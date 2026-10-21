# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Pantheon Makefile
#
# Builds Pantheon sources + Foundry sources into a single binary.
#
# Foundry location, in precedence order:
#   1. FOUNDRY=/path/to/foundry   explicit override
#   2. ./foundry                  the git submodule (git clone --recursive, or
#                                 git submodule update --init), or a symlink
#   3. ../foundry                 sibling checkout

CC ?= clang
# CPU target. Default -march=native: fastest, but the binary only runs on a CPU
# with the build machine's instruction set, so build on the machine that runs it.
# PORTABLE=1 builds a binary you can copy between machines: -march=x86-64-v2 on
# x86-64 (SSE4.2, no AVX; runs on any x86-64 CPU since ~2009), the compiler's
# generic default elsewhere.
ifeq ($(PORTABLE),1)
  ifneq ($(filter x86_64 amd64,$(shell uname -m)),)
    ARCH_FLAGS = -march=x86-64-v2
  else
    ARCH_FLAGS =
  endif
else
  ARCH_FLAGS = -march=native
endif
CFLAGS = -Wall -Wextra -std=c11 -O3 $(ARCH_FLAGS) -ffast-math
CFLAGS += -I$(FOUNDRY) -I.
ifneq ($(shell uname -s),Darwin)
  CFLAGS += -D_GNU_SOURCE
endif

# The serve layer (pantheon serve) is thread-per-connection, so pthreads are
# linked unconditionally. This is independent of the opt-in USE_THREADS compute
# thread pool below.
CFLAGS += -pthread
LDFLAGS += -pthread

FOUNDRY ?= $(if $(wildcard ./foundry/tensor/tensor.h),./foundry,../foundry)

ifeq ($(wildcard $(FOUNDRY)/tensor/tensor.h),)
  $(error Foundry not found at '$(FOUNDRY)'. Run 'git submodule update --init', \
symlink a checkout to ./foundry, clone it as ../foundry, or pass FOUNDRY=/path/to/foundry)
endif

# Debug build
ifdef DEBUG
CFLAGS = -Wall -Wextra -std=c11 -O0 -g -DDEBUG
CFLAGS += -I$(FOUNDRY) -I.
endif

# ============================================================
# Build provenance: the commit this binary came from.
#
# Stamped into src/accel/backend_identity.c so the process can print it next to
# its backend at startup and through `pantheon --backend`.  A tree with no git
# (a tarball, a copied checkout) reports "unknown" rather than failing the
# build — the point is that the binary never has to be reverse-engineered with
# `strings`, and "unknown" is itself an answer.
#
# Deliberately appended AFTER the DEBUG block, which REASSIGNS CFLAGS: put it
# above and a DEBUG=1 build silently loses the stamp.
# ============================================================
PN_BUILD_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
PN_BUILD_DIRTY  := $(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
CFLAGS += -DPN_BUILD_COMMIT='"$(PN_BUILD_COMMIT)$(PN_BUILD_DIRTY)"'

# pthread thread pool (opt-in: make USE_THREADS=1)
ifdef USE_THREADS
  CFLAGS += -DUSE_THREADS -pthread
  LDFLAGS += -pthread
  THREADPOOL_SRC = $(FOUNDRY)/threadpool/threadpool.c
else
  THREADPOOL_SRC =
endif

# BLAS acceleration (opt-in: make USE_BLAS=1)
ifdef USE_BLAS
  CFLAGS += -DUSE_BLAS
  ifeq ($(shell uname -s),Darwin)
    CFLAGS += -DACCELERATE_NEW_LAPACK
    LDFLAGS_BLAS = -framework Accelerate
  else
    CFLAGS += -DUSE_OPENBLAS
    LDFLAGS_BLAS = -lopenblas
  endif
else
  LDFLAGS_BLAS =
endif

# Metal GPU backend (opt-in: make USE_METAL=1, macOS only)
ifdef USE_METAL
  CFLAGS += -DUSE_METAL
  LDFLAGS_METAL = -framework Metal -framework MetalPerformanceShaders -framework Foundation
  # fm_metal is a device/queue/pipeline core plus one unit per kernel family
  # (foundry P1.0, cfa6727).  Globbed, never hand-listed: the family units are
  # added one per kernel through Phase 1, and a hand-listed link line fails as
  # an undefined symbol at the END of a 40-source compile.
  METAL_SRC = $(FOUNDRY)/metal/fm_metal.m $(wildcard $(FOUNDRY)/metal/kernels/*.m)
  OBJCFLAGS = -Wall -Wextra -O3 -ffast-math -fobjc-arc -DUSE_METAL -I$(FOUNDRY) -I.
  ifdef DEBUG
    OBJCFLAGS = -Wall -Wextra -O0 -g -DDEBUG -fobjc-arc -DUSE_METAL -I$(FOUNDRY) -I.
  endif
else
  LDFLAGS_METAL =
  METAL_SRC =
endif

# CUDA GPU backend (opt-in: make USE_CUDA=1, Linux only)
ifdef USE_CUDA
  CFLAGS += -DUSE_CUDA
  LDFLAGS_CUDA = -ldl
  # barswap.c is pulled in too: the 2-die layer-split forward in
  # qwen35moe_forward_cuda.c time-multiplexes the K80 dies through the single
  # MMIO slot via bs_acquire/bs_release. On a single-die host it is harmless
  # (only the 2-die path calls it, gated by a configured split layer).
  CUDA_SRC = $(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c
else
  LDFLAGS_CUDA =
  CUDA_SRC =
endif

# Tenstorrent Blackhole backend (opt-in: make USE_TT=1, Linux only)
#
# FT_TT_PHASE1B_NOC_ENABLED comes with it and is deliberately not a second knob.
# The fence exists so a DEFAULT foundry build cannot issue a NoC transaction
# during bring-up; a Pantheon serve that opens the card is past that by
# definition, and without it ft_tt_mem_info() cannot read a bank count, so
# pn_dm_init_tt() refuses every device. Coupling the two is what stops a build
# that links the backend and then silently declines to see the card.
#
# Device kernels are NOT built here: SFPI is needed only to bake them, and the kernel ELFs
# are pre-baked into tt/ft_tt_kernels.h — the same trick cuda/ uses for PTX.
#
# ft_tt_program_blob.c + ft_tt_program_resolve.c are what make that true of whole
# PROGRAMS as well, and they are named here rather than dragged in by a foundry
# header on purpose: ft_tt_program_resolve.c is the only file that pulls the
# 2.8 MB tt/ft_tt_programs.h into a link, so a build that has not adopted
# name-based program resolution links exactly what it linked before. Pantheon
# HAS adopted it (pn_tt_device.c, pn_tt_oncard.c), which is why the binary now
# carries its own Tensix programs instead of reading them out of a home
# directory at start-up.
#
# The ft_tt_udrv_* files are foundry's OWN USERSPACE DRIVER, linked in so that
# FT_TT_DRIVER=udrv can reach the card on a machine with no vendor module.
# Linking them is not using them: the default transport is still tt-kmd and
# ft_tt_runtime_open() is the only thing that ever chooses. What linking buys is
# that ONE binary runs on a board with tt-kmd and on the board that will not
# have it, which is the whole point of Phase 12 — the alternative is a second
# build configuration nobody tests until the card has already moved.
ifdef USE_TT
  # tt/convert/ sources include tt/ headers by bare name (ft_tt_op_l1_layout.h), which
  # resolves in the convert tools' own build via -I../.. and resolves nowhere here.
  CFLAGS += -DUSE_TT -DFT_TT_PHASE1B_NOC_ENABLED -I$(FOUNDRY)/tt
  TT_SRC = $(FOUNDRY)/tt/ft_tt_kmd.c \
	$(FOUNDRY)/tt/ft_tt_arch.c \
	$(FOUNDRY)/tt/ft_tt_arch_blackhole.c \
	$(FOUNDRY)/tt/ft_tt_device.c \
	$(FOUNDRY)/tt/ft_tt_tlb.c \
	$(FOUNDRY)/tt/ft_tt_arc_telemetry.c \
	$(FOUNDRY)/tt/ft_tt_coord.c \
	$(FOUNDRY)/tt/ft_tt_noc.c \
	$(FOUNDRY)/tt/ft_tt_buffer.c \
	$(FOUNDRY)/tt/ft_tt_elf.c \
	$(FOUNDRY)/tt/ft_tt_kernel_blob.c \
	$(FOUNDRY)/tt/ft_tt_program_blob.c \
	$(FOUNDRY)/tt/ft_tt_program_resolve.c \
	$(FOUNDRY)/tt/ft_tt_launch.c \
	$(FOUNDRY)/tt/ft_tt_core_status.c \
	$(FOUNDRY)/tt/ft_tt_core_readiness.c \
	$(FOUNDRY)/tt/ft_tt_multicast.c \
	$(FOUNDRY)/tt/ft_tt_group_shared_operand.c \
	$(FOUNDRY)/tt/ft_tt_dram_stripe.c \
	$(FOUNDRY)/tt/ft_tt_op_matmul_cooperative.c \
	$(FOUNDRY)/tt/ft_tt_group_broadcast_partition.c \
	$(FOUNDRY)/tt/ft_tt_op_matmul_resident_multicast.c \
	$(FOUNDRY)/tt/ft_tt_tilize.c \
	$(FOUNDRY)/tt/ft_tt_tile_row0.c \
	$(FOUNDRY)/tt/ft_tt_watchdog.c \
	$(FOUNDRY)/tt/ft_tt_queue.c \
	$(FOUNDRY)/tt/ft_tt_group.c \
	$(FOUNDRY)/tt/ft_tt_runtime.c \
	$(FOUNDRY)/tt/ft_tt_runtime_udrv.c \
	$(FOUNDRY)/tt/ft_tt_udrv_pci_sysfs.c \
	$(FOUNDRY)/tt/ft_tt_udrv_tlb_encode.c \
	$(FOUNDRY)/tt/ft_tt_udrv_claim.c \
	$(FOUNDRY)/tt/ft_tt_udrv_noc.c \
	$(FOUNDRY)/tt/ft_tt_udrv_noc_bank.c \
	$(FOUNDRY)/tt/ft_tt_udrv_arc_msg.c \
	$(FOUNDRY)/tt/ft_tt_udrv_arc_probe.c \
	$(FOUNDRY)/tt/ft_tt_udrv_reset.c \
	$(FOUNDRY)/tt/ft_tt_fttw.c \
	$(FOUNDRY)/tt/ft_tt_op_dispatch.c \
	$(FOUNDRY)/tt/ft_tt_op_matmul.c \
	$(FOUNDRY)/tt/ft_tt_op_memory.c \
	$(FOUNDRY)/tt/ft_tt_op_rms_norm.c \
	$(FOUNDRY)/tt/ft_tt_op_rope.c \
	$(FOUNDRY)/tt/ft_tt_op_sample.c \
	$(FOUNDRY)/tt/ft_tt_op_vector.c \
	$(FOUNDRY)/tt/ft_tt_op_attention.c \
	$(FOUNDRY)/tt/ft_tt_op_gdn.c \
	$(FOUNDRY)/tt/ft_tt_op_gdnc.c \
	$(FOUNDRY)/tt/ft_tt_op_replicas.c \
	$(FOUNDRY)/tt/convert/ft_bfp.c \
	$(FOUNDRY)/tt/convert/ft_float_convert.c \
	$(FOUNDRY)/tt/convert/ft_tt_precision_ledger.c \
	$(FOUNDRY)/tt/convert/ft_gguf_map.c \
	$(FOUNDRY)/tt/convert/ft_gguf_dequant.c \
	$(FOUNDRY)/tt/convert/ft_tt_weight_convert.c \
	$(FOUNDRY)/tt/convert/ft_tt_capacity_plan.c \
	$(FOUNDRY)/tt/convert/ft_tt_artifact_provenance.c \
	$(FOUNDRY)/tt/convert/ft_tt_artifact_cache.c
else
  TT_SRC =
endif

# ============================================================
# Foundry sources we depend on
# ============================================================

FOUNDRY_SRC = \
	$(FOUNDRY)/tensor/tensor.c \
	$(FOUNDRY)/safetensors/safetensors.c \
	$(FOUNDRY)/tokenizer/tokenizer.c \
	$(FOUNDRY)/quantization/quantization.c \
	$(FOUNDRY)/pth/pth.c \
	$(THREADPOOL_SRC) \
	$(METAL_SRC) \
	$(CUDA_SRC) \
	$(TT_SRC)

# ============================================================
# Pantheon sources
# ============================================================

PN_SRC = \
	src/core/engine.c \
	src/core/mem/pn_host_signals.c \
	src/core/mem/pn_memwatch.c \
	src/core/mem/pn_admission.c \
	src/core/mem/pn_kv_geometry.c \
	src/core/mem/pn_ladder.c \
	src/core/mem/pn_lazy_zero.c \
	src/core/bench.c \
	src/core/sample_profiler.c \
	src/core/qweight.c \
	src/core/sampling_modifiers.c \
	src/core/grammar_mask_shortlist.c \
	src/core/chat.c \
	src/core/bpe_pretokenize.c \
	src/core/ngram_draft.c \
	src/core/prefix_cache.c \
	src/core/session_snapshot.c \
	src/memory/arena.c \
	src/accel/simd.c \
	src/accel/die_manager.c \
	src/accel/die_layer_split.c \
	src/accel/backend_identity.c \
	src/accel/tt/pn_tt_device.c \
	src/accel/tt/pn_tt_matmul_group.c \
	src/accel/tt/pn_tt_residency.c \
	src/accel/tt/pn_tt_issue.c \
	src/accel/tt/pn_tt_core_boot_guard.c \
	src/accel/tt/pn_tt_oncard.c \
	src/accel/tt/pn_tt_eltwise_group.c \
	src/accel/tt/pn_tt_eltwise_split.c \
	src/accel/tt/pn_tt_reader.c \
	src/accel/tt/pn_tt_attention.c \
	src/accel/tt/pn_tt_gdn.c \
	src/accel/tt/pn_tt_gdn_checkpoint.c \
	src/accel/tt/pn_tt_gdnc.c \
	src/accel/tt/pn_tt_gdngate.c \
	src/accel/tt/pn_tt_stage_pipeline.c \
	src/accel/tt/pn_tt_head_norm.c \
	src/accel/tt/pn_tt_render.c \
	src/accel/tt/pn_tt_gate.c \
	src/accel/tt/pn_tt_fp32_strip.c \
	src/accel/tt/pn_tt_column_runs.c \
	src/accel/tt/pn_tt_descriptor_census.c \
	src/accel/tt/pn_tt_name_list.c \
	src/accel/tt/pn_tt_gdn_layout.c \
	src/accel/tt/pn_tt_gdn_core_scaling.c \
	src/accel/tt/pn_tt_replica_map.c \
	src/accel/tt/pn_tt_dispatch.c \
	src/formats/gguf.c \
	src/formats/gguf_dequant.c \
	src/formats/gguf_layer_bytes.c \
	src/models/llama.c \
	src/models/llama_state_snapshot.c \
	src/models/llama_attention_trace.c \
	src/models/llama_residual_trace.c \
	src/models/llama_forward_cuda.c \
	src/models/llama_forward_tt.c \
	src/models/pixtral_vision.c \
	src/models/phi3.c \
	src/models/rwkvx.c \
	src/models/gpu_attn_path_tracker.c \
	src/models/gpu_graph_capture_tracker.c \
	src/models/qwen35moe.c \
	src/models/qwen35moe_attn.c \
	src/models/qwen35moe_ssm.c \
	src/models/qwen35moe_moe.c \
	src/models/qwen35moe_forward.c \
	src/models/qwen35moe_forward_cuda.c \
	src/models/qwen35moe_forward_tt.c \
	src/models/qwen35moe_resident_tt.c \
	src/models/qwen35moe_gdn_tt.c \
	src/models/qwen35moe_gdn_strip_tt.c \
	src/models/qwen35moe_attn_tt.c \
	src/models/qwen35moe_prefill_tt.c \
	src/models/qwen35moe_prefill_checkpoint_tt.c \
	src/models/qwen35moe_checkpoint_band.c \
	src/models/qwen35moe_prefill_prefetch_tt.c \
	src/models/qwen35moe_prefill_overlap_tt.c \
	src/models/qwen35moe_prefill_tt_wall.c \
	src/models/qwen35moe_gdn_parity_tt.c \
	src/models/qwen35moe_gdn_normcheck_tt.c \
	src/models/qwen35moe_forward_metal.c \
	src/models/qwen35moe_metal_kernel_registry.c \
	src/models/qwen35moe_metal_coverage.c \
	src/models/qwen35moe_state_snapshot.c \
	src/models/bonsai_forward.c

# OpenAI-compatible HTTP serving layer (pantheon serve). Pure C + POSIX
# sockets; no extra link deps. Compiled into the CLI and the static library.
SERVE_SRC = \
	src/serve/json_reader.c \
	src/serve/json_writer.c \
	src/serve/attn_residency_json.c \
	src/serve/graph_residency_json.c \
	src/serve/http_server.c \
	src/serve/json_value_grammar.c \
	src/serve/tool_calling.c \
	src/serve/tool_grammar.c \
	src/serve/tool_grammar_json.c \
	src/serve/grammar_spec_chooser.c \
	src/serve/tool_param_empty_value_rule.c \
	src/serve/response_format_grammar.c \
	src/serve/serve_emitter.c \
	src/serve/json_canonical_stream.c \
	src/serve/tool_call_stream.c \
	src/serve/repetition_loop_detector.c \
	src/serve/serve_prompt_tokens.c \
	src/serve/serve_spec_lookup.c \
	src/serve/serve_generation.c \
	src/serve/model_registry.c \
	src/serve/model_session.c \
	src/serve/batch_sched.c \
	src/serve/base64.c \
	src/serve/model_admin.c \
	src/serve/openai_chat.c \
	src/serve/request_profile.c \
	src/serve/signal_handlers.c \
	src/serve/serve_shutdown.c \
	src/serve/serve_backend_guard.c \
	src/serve/serve_command.c \
	src/serve/session_watch.c \
	src/serve/serve_registry.c \
	src/serve/registry.c \
	src/serve/session_affinity.c \
	src/serve/registry_client.c \
	src/serve/http_client.c \
	src/serve/router.c \
	src/serve/router_command.c

# ============================================================
# Header dependencies — DISCOVERED, never hand-listed
#
# Every harness/test binary below is a one-shot whole-program compile whose
# prerequisites are .c files ONLY.  A changed HEADER therefore left make saying
# "is up to date" and handing back the PREVIOUS binary.
#
# For $(FOUNDRY)/cuda/fc_cuda_ptx.h — the GENERATED header carrying the
# embedded PTX for every CUDA kernel (regenerated on a CUDA host by
# cuda_kernels/Makefile) — that fails in the worst possible direction: the A/B
# reads 1.000x and the bit-exactness gate passes vacuously.  Both look like an
# honest "no change" rather than a broken build.
#
# The list is produced by find, NOT written by hand, so it cannot go stale the
# way this file's link lines already have once (Pantheon 8616f92, five targets
# with stale hand-maintained link lines).  It is deliberately over-approximate:
# these targets recompile every .c on every build anyway, so per-target
# precision would buy nothing.  The one place objects ARE compiled separately
# (libpantheon.a) uses real -MMD -MP depfiles instead.
#
# `pantheon` and `pantheon-convert` are .PHONY and so always rebuild — that is
# deliberate and is what keeps a flag change (USE_CUDA=1 etc., which make
# cannot see) from silently reusing a binary built with different flags.
# ============================================================
PN_HDR      := $(shell find -L src include cli tests -name '*.h' 2>/dev/null)
FOUNDRY_HDR := $(shell find -L $(FOUNDRY) -name '.git' -prune -o -name '*.h' -print 2>/dev/null)
ALL_HDR     := $(PN_HDR) $(FOUNDRY_HDR)

# ============================================================
# PER-BACKEND OUTPUT NAMES — the three builds are three programs
#
# `make`, `make USE_CUDA=1` and `make USE_TT=1` produce binaries that cannot run
# each other's work, and until this block they all wrote ONE file, `pantheon`.
# Last build won, silently.  That is how a CPU-only binary came to sit at the
# path a deploy config named as its binary and serve a 35B
# MoE off a 2-core no-AVX Pentium with all 8 K80 dies idle at 0 MiB, for an
# unknown period, with measurements taken against it: a Tenstorrent session ran
# `rm -f pantheon && make USE_TT=1 pantheon` in that checkout, and a later bare
# `make` left a CPU build behind.
#
# So each backend gets its OWN name.  A TT build is now physically incapable of
# overwriting the deployed CUDA binary — it writes a different file.
#
# `pantheon` survives as a SYMLINK to whichever backend was built last, because
# deploy scripts, tests/serve_*.sh and the docs all name it.  The symlink keeps every
# existing invocation working AND fixes the damage case: a stray TT build now
# only re-points a link, so `ln -sfn pantheon-cuda pantheon` restores a CUDA
# deployment in one command instead of a ~10-minute CUDA rebuild.  The link is not the
# safety net, though — `pantheon --backend` is (check it in a deploy preflight),
# and the serve refuses to start when --dies names dies it cannot use.
#
# Combination builds get a combination name so no two configurations can ever
# collide on one path.
# ============================================================
PN_BACKEND_SUFFIX :=
ifdef USE_CUDA
  PN_BACKEND_SUFFIX := $(PN_BACKEND_SUFFIX)-cuda
endif
ifdef USE_TT
  PN_BACKEND_SUFFIX := $(PN_BACKEND_SUFFIX)-tt
endif
ifdef USE_METAL
  PN_BACKEND_SUFFIX := $(PN_BACKEND_SUFFIX)-metal
endif
ifeq ($(PN_BACKEND_SUFFIX),)
  PN_BACKEND_SUFFIX := -cpu
endif

PN_BIN        := pantheon$(PN_BACKEND_SUFFIX)
PN_COMPAT_BIN := pantheon

# Every per-backend name this build system can emit — `clean` has to remove all
# of them, not just the one the current flags select, or a stale binary from
# another configuration outlives the clean that was supposed to remove it.
PN_ALL_BINS := pantheon-cpu pantheon-cuda pantheon-tt pantheon-metal \
	pantheon-cuda-tt pantheon-cuda-metal pantheon-tt-metal \
	pantheon-cuda-tt-metal

# ============================================================
# Targets
# ============================================================

.PHONY: all clean test pantheon pantheon-convert $(PN_BIN)

all: pantheon

# CLI binary, named for its backend.  PHONY (as `pantheon` always was): make
# cannot see a flag change, so a rebuild is the only thing that keeps
# USE_CUDA=1 from silently reusing a CPU object graph.
$(PN_BIN): cli/main.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC)
	$(CC) $(CFLAGS) -o $@ cli/main.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)
	@echo "built $@ (backend $(PN_BACKEND_SUFFIX:-%=%), commit $(PN_BUILD_COMMIT)$(PN_BUILD_DIRTY))"

# Back-compat: `make pantheon` still works and still leaves something runnable
# at ./pantheon — now a symlink, so the OTHER backends' binaries survive it.
# rm-then-ln rather than `ln -sfn`: the deployed ./pantheon is a regular FILE
# until the first build after this change, and rm-then-ln replaces both a file
# and a link identically on GNU and BSD.
pantheon: $(PN_BIN)
	@rm -f $(PN_COMPAT_BIN)
	@ln -s $(PN_BIN) $(PN_COMPAT_BIN)
	@echo "./$(PN_COMPAT_BIN) -> $(PN_BIN)  (verify with: ./$(PN_COMPAT_BIN) --backend)"

# `make pantheon-cuda` WITHOUT USE_CUDA=1 must not quietly produce a CPU binary
# wearing a CUDA name — that is precisely the confusion the rename exists to
# remove, and it would be worse than the original bug because the file name
# would now be an active lie. Every backend name that is not the one these flags
# select refuses and prints the flag it wanted.
PN_WRONG_FLAG_BINS := $(filter-out $(PN_BIN),$(PN_ALL_BINS))
.PHONY: $(PN_WRONG_FLAG_BINS)
$(PN_WRONG_FLAG_BINS):
	@echo "make: '$@' needs its backend flag — these flags build $(PN_BIN)." >&2
	@echo "  cuda:  make USE_CUDA=1        tt:  make USE_TT=1" >&2
	@echo "  metal: make USE_METAL=1       cpu: make" >&2
	@false

# ============================================================
# Static library for embedding in another program — same objects as the CLI
# minus cli/main.c. Build with the SAME flags you intend to link against, e.g.:
#   make USE_BLAS=1 USE_THREADS=1 libpantheon.a
# then link: -lpantheon <Accelerate/Metal/openblas as used> -lm
# ============================================================
LIB_OBJDIR = build/lib
LIB_C_ONLY = $(filter %.c,$(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC))
LIB_OBJS   = $(patsubst %,$(LIB_OBJDIR)/%.o,$(LIB_C_ONLY))
ifdef USE_METAL
LIB_OBJS  += $(patsubst %,$(LIB_OBJDIR)/%.o,$(METAL_SRC))
endif
LIB_DEPS   = $(LIB_OBJS:.o=.d)

# Real depfiles: these are the only separately-compiled objects in the build,
# so the precise mechanism applies.  -MMD writes
# build/lib/<path>.c.d beside the object; -MP adds phony targets so a DELETED
# header does not wedge the build.
$(LIB_OBJDIR)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Same mechanism for the Objective-C Metal units.  A pattern rule, not one rule
# per file, so a new kernel family unit needs no Makefile edit.
$(LIB_OBJDIR)/%.m.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

-include $(LIB_DEPS)

libpantheon.a: $(LIB_OBJS)
	ar rcs $@ $(LIB_OBJS)
	@echo "built $@ (public header: include/pantheon.h)"
	@echo "link with: -lpantheon $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA) -lm"

# Build + run tests
test: tests/test_backend_guard tests/test_build tests/test_gguf tests/test_gguf_layer_bytes tests/test_admission_reserve tests/test_die_scratch_ctx tests/test_die_layer_split tests/test_arena tests/test_qmv tests/test_simd tests/test_simd_props tests/test_chat tests/test_tool_calling tests/test_tool_call_defects tests/test_tool_call_truncation tests/test_empty_argument tests/test_frame_capture tests/test_json_value_grammar tests/test_tool_grammar tests/test_tool_grammar_json tests/test_grammar_spec_chooser tests/test_grammar_mask_shortlist tests/test_grammar_topn_sampler tests/test_response_format_grammar tests/test_sampling_modifiers tests/test_chat_request tests/test_serve_emitter tests/test_tool_call_stream tests/test_repetition_loop_detector tests/test_serve_spec_lookup tests/test_registry tests/test_base64 tests/test_prefix_cache_sys tests/test_llama_state_snapshot tests/test_prefix_projection tests/test_session_watch tests/test_session_affinity tests/test_bs_load tests/test_bpe_pretokenize tests/test_pretok_props tests/test_dequant_props tests/test_rwkvx tests/test_qwen35moe tests/test_qwen35_prefill_causality tests/test_qwen35moe_metal_coverage tests/test_qwen35moe_metal_coverage_fake tests/test_die_manager tests/test_model_registry tests/test_model_unload_reclaim tests/test_embed_row tests/test_tt_column_runs tests/test_qwen35_ckpt_band tests/test_tt_name_list tests/test_tt_eltwise_split tests/test_tt_descriptor_census tests/test_tt_stage_pipeline tests/test_graph_capture_retry tests/test_attn_ring_retry tests/test_serve_shutdown
	./tests/test_backend_guard
	./tests/test_build
	./tests/test_gguf
	./tests/test_gguf_layer_bytes
	./tests/test_admission_reserve
	./tests/test_die_scratch_ctx
	./tests/test_die_layer_split
	./tests/test_arena
	./tests/test_qmv
	./tests/test_simd
	./tests/test_simd_props
	./tests/test_chat
	./tests/test_tool_calling
	./tests/test_tool_call_defects
	./tests/test_tool_call_truncation
	./tests/test_empty_argument
	./tests/test_frame_capture
	./tests/test_json_value_grammar
	./tests/test_tool_grammar
	./tests/test_tool_grammar_json
	./tests/test_grammar_spec_chooser
	./tests/test_grammar_mask_shortlist
	./tests/test_grammar_topn_sampler
	./tests/test_grammar_topn_sampler --selftest
	./tests/test_response_format_grammar
	./tests/test_sampling_modifiers
	./tests/test_chat_request
	./tests/test_serve_emitter
	./tests/test_tool_call_stream
	./tests/test_repetition_loop_detector
	./tests/test_registry
	./tests/test_base64
	./tests/test_prefix_cache_sys
	./tests/test_llama_state_snapshot
	./tests/test_prefix_projection
	./tests/test_session_watch
	./tests/test_session_affinity
	./tests/test_bs_load
	./tests/test_bpe_pretokenize
	./tests/test_pretok_props
	./tests/test_dequant_props
	./tests/test_rwkvx
	./tests/test_qwen35moe
	./tests/test_qwen35_prefill_causality
	./tests/test_qwen35moe_metal_coverage
	./tests/test_qwen35moe_metal_coverage_fake
	./tests/test_die_manager
	./tests/test_model_registry
	./tests/test_model_unload_reclaim
	./tests/test_embed_row
	./tests/test_tt_column_runs
	./tests/test_qwen35_ckpt_band
	./tests/test_tt_eltwise_split
	./tests/test_tt_descriptor_census
	./tests/test_graph_capture_retry
	./tests/test_graph_capture_retry --latch
	./tests/test_attn_ring_retry
	./tests/test_attn_ring_retry --latch
	./tests/test_serve_shutdown

tests/test_build: tests/test_build.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_build.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_gguf: tests/test_gguf.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_gguf.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_gguf_layer_bytes: tests/test_gguf_layer_bytes.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_gguf_layer_bytes.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_admission_reserve: tests/test_admission_reserve.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_admission_reserve.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_die_layer_split: tests/test_die_layer_split.c src/accel/die_layer_split.c src/accel/die_layer_split.h
	$(CC) $(CFLAGS) -o $@ tests/test_die_layer_split.c src/accel/die_layer_split.c -lm

tests/test_die_scratch_ctx: tests/test_die_scratch_ctx.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_die_scratch_ctx.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_arena: tests/test_arena.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_arena.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_qmv: tests/test_qmv.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qmv.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_simd: tests/test_simd.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_simd.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Clean-room property tests for the CPU vector kernels (spec-derived, flag-independent).
tests/test_simd_props: tests/test_simd_props.c src/accel/simd.c src/accel/simd.h \
		src/formats/gguf_dequant.c src/formats/gguf.c src/formats/gguf_dequant.h src/formats/gguf.h
	$(CC) $(CFLAGS) -o $@ tests/test_simd_props.c src/accel/simd.c \
		src/formats/gguf_dequant.c src/formats/gguf.c -lm

tests/test_chat: tests/test_chat.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_chat.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# FRAME CAPTURE: a forced-close turn whose compelled XML frame swallowed the
# call the model was still writing. Synthetic fixtures in the observed failure
# shapes (tests/frame_capture_fixtures.h). Same deps as the parser
# suite below — pure C, no engine/model/CUDA.
tests/test_frame_capture: tests/test_frame_capture.c tests/frame_capture_fixtures.h src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_frame_capture.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# tc_parse_tool_calls (+ render) is pure C string parsing over the JSON DOM;
# links only the three serve string modules, no engine/model/CUDA deps.
tests/test_tool_calling: tests/test_tool_calling.c tests/abandoned_call_fixture.h src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_calling.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Structurally-wrong tool-call shapes (leaked dialect tail / reasoning leak /
# missing + unknown name) and the truncated-JSON argument repair — the two
# serve-side items from the Qwen harness audit. Same deps as the parser suite
# above: pure C, no engine/model/CUDA.
tests/test_tool_call_defects: tests/test_tool_call_defects.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_call_defects.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Truncated tool calls: the grammar and the parser must agree about where a value
# ended, and a call the grammar never closed must not be handed back as a call.
# The only test that links BOTH halves — the defect lives in the gap between
# them. Still pure C: no engine/model/CUDA.
tests/test_tool_call_truncation: tests/test_tool_call_truncation.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_call_truncation.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# EMPTY IS NOT ABSENT: a parameter the model CLOSED around an empty value is
# emitted as "" rather than dropped, so the client can tell "the model meant
# empty" (edit's new_string:"" is a deletion) from "the argument never arrived"
# (a truncated call, which the client charges against a task-ending budget).
# Pure C string parsing over the JSON DOM — no engine/model/CUDA.
tests/test_empty_argument: tests/test_empty_argument.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_empty_argument.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Incremental JSON-value well-formedness walk — the shared core that makes a
# container-typed tool argument unfailable (grammar side) and its raw splice
# safe (parser side).  Pure C, libc-only: no engine/model/CUDA, fully local.
tests/test_json_value_grammar: tests/test_json_value_grammar.c src/serve/json_value_grammar.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_json_value_grammar.c src/serve/json_value_grammar.c \
		-lm $(LDFLAGS)

# Grammar-constrained tool decoding FSM (Phase 5). Pure C string/DOM parsing
# over the tools schema; links only the grammar module + the JSON reader — no
# engine/model/CUDA deps, fully local.
tests/test_tool_grammar: tests/test_tool_grammar.c src/serve/tool_grammar.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar.c src/serve/tool_grammar.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

tests/test_tool_grammar_json: tests/test_tool_grammar_json.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_json.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Speculative decoding under the tool-call grammar: drafts verified through the
# grammar-aware chooser must reproduce plain constrained decoding token for
# token (self-falsifying simulation).  Links the real JSON grammar and the
# parser; no engine/model/device.
tests/test_grammar_spec_chooser: tests/test_grammar_spec_chooser.c src/serve/grammar_spec_chooser.c src/core/grammar_mask_shortlist.c src/serve/tool_grammar_json.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_spec_chooser.c src/serve/grammar_spec_chooser.c src/core/grammar_mask_shortlist.c src/serve/tool_grammar_json.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# The grammar mask decided from a shortlist of the highest logits must leave the
# same greedy pick / top-K set as the full-vocabulary mask.  libc-only.
tests/test_grammar_mask_shortlist: tests/test_grammar_mask_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampling_modifiers.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_mask_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampling_modifiers.c \
		-lm $(LDFLAGS)

# Full-logits ARMING WINDOW (PN_GRAMMAR_ARM_WINDOW): is the row a masked step
# reads armed one token EARLY enough?  Models the decode loop's ordering over
# the three real grammar walks; self-falsifying — exits 2 VACUOUS if the
# deliberately-stale policy fails to break.  No engine/model/CUDA deps.
tests/test_grammar_arm_window: tests/test_grammar_arm_window.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_grammar_json.c src/serve/response_format_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_arm_window.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_grammar_json.c src/serve/response_format_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# response_format (json_object) constrained-decoding FSM.  Pure C char walk with
# no schema/DOM input at all — links only the grammar module.  No engine/model/
# CUDA deps, fully local.
tests/test_response_format_grammar: tests/test_response_format_grammar.c src/serve/response_format_grammar.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_response_format_grammar.c src/serve/response_format_grammar.c \
		-lm $(LDFLAGS)

# OpenAI logits-row sampling controls (logit_bias / presence+frequency penalties /
# seeded RNG / logprobs).  Pure float-array math — links only the modifier module.
tests/test_sampling_modifiers: tests/test_sampling_modifiers.c src/core/sampling_modifiers.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampling_modifiers.c src/core/sampling_modifiers.c \
		-lm $(LDFLAGS)

# parse_chat_request tool_choice handling + forced_name lifetime (Phase 5 BUG 2).
# #includes openai_chat.c directly to reach its static parse fn; the request-parse
# path only touches the JSON reader + tool_calling renderer (linked), and tiny
# link stubs in the test satisfy the engine-side handlers it never calls. No
# engine/model/CUDA deps, fully local. serve_shutdown.c is linked because the
# parse's DEFAULT abort predicate is the shutdown check — every turn has to be
# stoppable by a SIGTERM — and that module is libc + pthread only.
# src/accel/backend_identity.c joins the link line because oai_handle_models now
# names the compiled backend on /v1/models. The module is a handful of string
# literals with no dependencies of its own, so it costs this harness nothing.
tests/test_chat_request: tests/test_chat_request.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c src/serve/serve_shutdown.c src/serve/openai_chat.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/accel/backend_identity.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_chat_request.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c src/serve/request_profile.c src/serve/serve_shutdown.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/accel/backend_identity.c \
		-lm $(LDFLAGS)

# Stream tool-call detection (serve_generation.c emitter, Phase 2). The emitter
# lives in serve_emitter.c and depends only on libc, so this links just that one
# module — no engine/model/CUDA deps.
# Where the multi-turn prefill checkpoint sits (a band boundary, at or below the history the
# next turn resends). Pure arithmetic, host-only — no card.
tests/test_qwen35_ckpt_band: tests/test_qwen35_ckpt_band.c src/models/qwen35moe_checkpoint_band.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_ckpt_band.c src/models/qwen35moe_checkpoint_band.c \
		-lm $(LDFLAGS)

tests/test_serve_emitter: tests/test_serve_emitter.c src/serve/serve_emitter.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_emitter.c src/serve/serve_emitter.c \
		-lm $(LDFLAGS)

# Incremental tool-call streaming (tool_call_stream.c + json_canonical_stream.c):
# the streamed argument fragments must concatenate to exactly the arguments the
# end-of-turn parse returns.  libc-only — the emitter, the streamer and the
# parser, no engine.
tests/test_tool_call_stream: tests/test_tool_call_stream.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_call_stream.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Decode-time repetition-cycle detector (the saber-orch Q3_K_M collapse: one
# paragraph re-emitted 31 times, reported as a normal finish). The detector
# depends only on libc, so this links just that one module — no engine/model/
# CUDA. The NEGATIVE tests (legitimate code, JSON, tables, log excerpts that
# must NOT trip it) are the ones that matter; run the red proof by relaxing
# RLD_DEFAULT_MIN_PERIOD to 1 and watching them go red.
tests/test_serve_spec_lookup: tests/test_serve_spec_lookup.c src/serve/serve_spec_lookup.c src/core/ngram_draft.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_spec_lookup.c src/serve/serve_spec_lookup.c \
		src/core/ngram_draft.c -lm $(LDFLAGS)

tests/test_repetition_loop_detector: tests/test_repetition_loop_detector.c src/serve/repetition_loop_detector.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_repetition_loop_detector.c \
		src/serve/repetition_loop_detector.c -lm $(LDFLAGS)

# Dispatch-layer registry unit test (Phase G1). Links only the registry + the
# two JSON modules — no engine/model/CUDA. Host-side, deterministic (injected
# clock for the TTL test).
tests/test_registry: tests/test_registry.c src/serve/registry.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_registry.c src/serve/registry.c \
		src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# System-prefix cache unit test (SYS1). Links the prefix cache + the state
# snapshot module only; the test stubs pn_state_new/reset/free over hand-built
# qwen35moe dual states — no engine/model/CUDA, builds and runs on the Mac.
tests/test_base64: tests/test_base64.c src/serve/base64.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_base64.c src/serve/base64.c \
		-lm $(LDFLAGS)

# Retry policy + three-state residency gauge for the CUDA-graph capture tracker
# (tasks #41/#42). Pure host state machine, no GPU, runs anywhere. Run the red
# arm too — `tests/test_graph_capture_retry --latch` replays the OLD one-shot
# latch and must fail its expectations, which is what proves the gate is not
# vacuous.
tests/test_graph_capture_retry: tests/test_graph_capture_retry.c \
		src/models/gpu_graph_capture_tracker.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_graph_capture_retry.c \
		src/models/gpu_graph_capture_tracker.c -lm $(LDFLAGS)

# Deferred-shutdown mechanism for `pantheon serve` (SIGTERM/SIGINT retiring the
# card residencies). Links serve_shutdown.c alone — no engine, no serve, no
# card — and raises REAL signals through a real sigaction to prove the handler
# path. The retirement it gates is device-side and is proved on silicon.
tests/test_serve_shutdown: tests/test_serve_shutdown.c \
		src/serve/serve_shutdown.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_shutdown.c \
		src/serve/serve_shutdown.c -lm $(LDFLAGS)

# Bounded retry after a failed per-slot KV-ring allocation + the ondie/host
# attention residency gauge. Pure host state machine, no GPU, runs anywhere.
# Run the red arm too — `tests/test_attn_ring_retry --latch` replays the OLD
# sticky latch (one attempt per geometry, then never again) and must fail its
# expectations, which is what proves the gate is not vacuous.
tests/test_attn_ring_retry: tests/test_attn_ring_retry.c \
		src/models/gpu_attn_path_tracker.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_attn_ring_retry.c \
		src/models/gpu_attn_path_tracker.c -lm $(LDFLAGS)

# Llama-family session snapshot (the primitive session export/restore rides on
# for every arch whose per-session state is a plain causal KV cache — the family
# the Tenstorrent backend serves). Pure host memory, no engine, no accelerator:
# the test hand-builds the tm_tensors so it links only the snapshot unit and the
# arch dispatcher. Byte comparison throughout; PCC would be blind to every
# failure this format can have.
tests/test_llama_state_snapshot: tests/test_llama_state_snapshot.c src/models/llama_state_snapshot.c src/models/qwen35moe_state_snapshot.c src/core/session_snapshot.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_llama_state_snapshot.c \
		src/models/llama_state_snapshot.c src/models/qwen35moe_state_snapshot.c \
		src/core/session_snapshot.c \
		-lm $(LDFLAGS)

tests/test_prefix_cache_sys: tests/test_prefix_cache_sys.c tests/tt_prefix_cache_link_stubs.c src/core/prefix_cache.c src/core/session_snapshot.c src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c src/serve/batch_sched.c src/serve/request_profile.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefix_cache_sys.c tests/tt_prefix_cache_link_stubs.c \
		src/core/prefix_cache.c src/core/session_snapshot.c \
		src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c \
		src/serve/batch_sched.c src/serve/request_profile.c \
		-lm $(LDFLAGS)

# Liveness-based session release (§C.4). Links the watch manager + prefix cache
# + snapshot machinery + http_server (hs_client_alive); mr_acquire/mr_release +
# pn_state_* are test stubs. Host-side, deterministic (manual manager + sweep
# hooks), no engine/model/CUDA.
tests/test_session_watch: tests/test_session_watch.c tests/tt_prefix_cache_link_stubs.c src/serve/session_watch.c src/core/prefix_cache.c src/core/session_snapshot.c src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c src/serve/http_server.c src/serve/model_session.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_session_watch.c tests/tt_prefix_cache_link_stubs.c \
		src/serve/session_watch.c src/core/prefix_cache.c src/core/session_snapshot.c \
		src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c \
		src/serve/http_server.c \
		src/serve/model_session.c src/serve/batch_sched.c src/serve/request_profile.c \
		-lm $(LDFLAGS)

# Session-affinity unit test (PC0.2). Links the affinity map + registry + the two
# JSON modules — no engine/model/CUDA. Host-side, deterministic.
tests/test_session_affinity: tests/test_session_affinity.c src/serve/session_affinity.c src/serve/registry.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_session_affinity.c \
		src/serve/session_affinity.c src/serve/registry.c \
		src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Dispatch-layer end-to-end test (Phase G1+G2): boots the real `pantheon router`
# + Python stub upstreams and asserts dedup / least-busy / failover / SSE
# pass-through / TTL eviction. Mac-only (host networking; no GPU/CUDA).
# Requires the built `pantheon` binary + python3 + curl. Runs ~20s (TTL wait).
.PHONY: test-dispatch-e2e
test-dispatch-e2e: pantheon
	bash tests/dispatch/e2e.sh

# Shared by every parity harness: the paired PPL/KL/gap scorer, the accelerator-arm
# selection that refuses a vacuous CPU-vs-CPU run, and the reproducible Gaussian
# perturbations. One copy so a gate and a debug sweep can never disagree about what a
# perplexity ratio means.
PARITY_LIB_SRC = tests/parity_score.c tests/parity_accel_arm.c tests/parity_noise.c \
                 tests/parity_gate_criteria.c tests/parity_logit_dump.c
PARITY_LIB_HDR = tests/parity_score.h tests/parity_accel_arm.h tests/parity_noise.h \
                 tests/parity_gate_criteria.h tests/parity_logit_dump.h

# The chat-templated, real-vocab, per-position accelerator parity GATE. Pass/fail on the
# perplexity a chat-framed prompt actually costs, not on one next-token argmax.
tests/test_chat_parity_gate: tests/test_chat_parity_gate.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR) $(PARITY_LIB_HDR) tests/chat_corpus.h
	$(CC) $(CFLAGS) -o $@ tests/test_chat_parity_gate.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# THE SAME GATE AS TWO SEPARATE PASSES, paired offline. The interleaved gate above puts a
# ~26 s CPU-oracle forward between two consecutive card forwards, which retires the resident
# triads: 802 / 2406 / 442 descriptor refusals against 0 on single-arm runs. This one dumps
# each arm's per-position logit rows on its own and scores them afterwards, which also makes
# the oracle pass CACHEABLE across arms (~3 h once, not per arm). The pairing step loads no
# model and opens no card, so it builds and runs with no accelerator flag at all.
tests/test_two_pass_parity_gate: tests/test_two_pass_parity_gate.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR) $(PARITY_LIB_HDR) tests/chat_corpus.h
	$(CC) $(CFLAGS) -o $@ tests/test_two_pass_parity_gate.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# L11b: the gdnc reader's on-card decay tile + gate runs vs the hardware multiply (FTZ+DAZ) and
# the host builders, bit for bit. Run it on an x86 host (where the builders run).
tests/test_gdnc_compact: tests/test_gdnc_compact.c $(FOUNDRY)/tt/kernels/ftgc_compact.h $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_gdnc_compact.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# PHASE B's CHEAP GATE: the prompt-lookup drafting ACCEPTANCE rate, two passes. --generate
# decodes a corpus greedily on the shipped arm and records the token ids (needs a card);
# --score replays those ids as speculative rounds (no model, no card, seconds); --selftest is
# a known-answer arm at both ends of the scale. Greedy verification accepts exactly the
# drafts that match the plain stream, so NO verify kernel is needed to measure this.
tests/bench_ngram_accept: tests/bench_ngram_accept.c tests/ngram_accept_stream.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR) $(PARITY_LIB_HDR) tests/chat_corpus.h tests/ngram_accept_stream.h
	$(CC) $(CFLAGS) -o $@ tests/bench_ngram_accept.c tests/ngram_accept_stream.c tests/chat_corpus.c $(PARITY_LIB_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The GDN host STAGER's address arithmetic, against the layout accessors the two dataflow
# kernels compute their own offsets with. Needs USE_TT=1 for the foundry tile code, but NOT a
# card: it opens no device, so it runs on the Mac where the stager is written.
tests/test_gdn_layout: tests/test_gdn_layout.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_gdn_layout.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# BYTE-EXACTNESS of the batched TT prefill against the one-at-a-time path. Byte comparison
# and not PCC: what makes a 32-row pass safe is row INDEPENDENCE, and a correlation cannot
# see independence. Needs USE_TT=1 and a card; without one it refuses rather than passing.
# THE BOOT GUARD AGAINST A REAL LEFTOVER. Two invocations, one card, NO MODEL —
# --leak strands a spinning compute thread and exits without closing the device,
# --read reports what the card says about that core writing nothing, and --arm is
# the process that used to wedge it. Needs USE_TT=1 and a card. It is a CARD ARM
# and belongs behind the flock runner (tt/tests/run_gdn_writer_capacity_arms.sh's
# pattern), but it is not a model load and must never become one.
tests/test_tt_boot_guard_leftover: tests/test_tt_boot_guard_leftover.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_boot_guard_leftover.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_tt_prefill_parity: tests/test_tt_prefill_parity.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_prefill_parity.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# THE THREE MEASUREMENTS PHASE 4 IS GATED ON, on Ornith's own strips: what a whole-tile
# transfer costs against the row-0 one (there is no on-card row gather, so a batched pass
# reaches row r through the link or not at all), whether the 32-row tile quantum survives 32
# LIVE rows rather than 31 zeros, and what the two mixers cost alone — they are the terms a
# batched pass cannot batch. Needs USE_TT=1 and a card.
tests/bench_tt_prefill_quantum: tests/bench_tt_prefill_quantum.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_tt_prefill_quantum.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Speculative greedy decode gate + bench (n-gram drafts + batched verify):
# the speculative token stream must be identical to plain greedy. Build with
# USE_CUDA=1 on a CUDA host; run under QWEN35_FUSE=0x37 for the strict gate.
tests/test_spec_qwen35: tests/test_spec_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Multi-user decode gate + bench (pn_qwen35moe_decode_slots): K users, one
# token each per forward pass — each user's batched stream must be identical
# to that user decoding alone. Build with USE_CUDA=1 on a CUDA host; run under
# QWEN35_FUSE=0x37 for the strict gate.
tests/test_users_qwen35: tests/test_users_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_users_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# What temperature costs the HOST per token: times the real sampler
# (pn_sample_next_token) at temp 0 vs temp 0.7 over the sparse device-top-k
# logits buffer the serve actually hands it. The delta bounds every "move the
# sampling off the host" lever. CPU-only — no GPU, no model, runs anywhere.
tests/bench_sampler_temp_delta: tests/bench_sampler_temp_delta.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_sampler_temp_delta.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Ship gate for the sparse-survivor sampler: the fast path must return the
# SAME token as the full-vocab path for the same row, params and RNG state.
# `--selftest` proves the gate can go red (and that the fast path actually
# ran) — a green run from a gate that cannot fail means nothing.
tests/test_sparse_sampler_identity: tests/test_sparse_sampler_identity.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sparse_sampler_identity.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The grammar mask on a backend's top-N row (shortlist / dense chunks / whole-row
# refill) must draw the token the full mask draws on the dense row.
# `--selftest`: a subject that skips the mask must be caught.
tests/test_grammar_topn_sampler: tests/test_grammar_topn_sampler.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_topn_sampler.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the top-N exactness check the sampler uses before sampling a top-N row
# (`--selftest`: a check blind to the penalties must be caught).
tests/test_topn_row_exact: tests/test_topn_row_exact.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_topn_row_exact.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Context-DEPTH decode bench: per-user steady-state DECODE tok/s as the seeded
# context deepens (isolates the 10 O(context) attention layers). Cold-prefills a
# large seed per user (untimed), then times only the batched decode. Build with
# USE_CUDA=1 on a CUDA host; run under QWEN35_FUSE=0x37.
tests/bench_depth_qwen35: tests/bench_depth_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_depth_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Session-lifetime regression gate: bind a decode slot immediately after a
# session free. Asserts no decode slot still holds a back-pointer into the
# freed state (address compare only), then performs the real bind-after-free —
# the sequence the serve runs on every session release and idle reap. Build
# with DEBUG=1 USE_CUDA=1 on a CUDA host so the ASAN arm runs too.
tests/test_slot_free_bind_qwen35: tests/test_slot_free_bind_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_slot_free_bind_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Phase 9.1 speculative-decoding GATE bench: cost of a D-row batched verify vs
# D sequential single-token forwards, plus the batched-slots row-cost curve.
# Pure measurement, changes no engine behaviour. Build with USE_CUDA=1 on a
# CUDA host; run under QWEN35_FUSE=0x37 to match the serve's kernel selection.
tests/bench_spec_verify_gate: tests/bench_spec_verify_gate.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_spec_verify_gate.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Phase 9.2 ACCEPTANCE bench: what fraction of prompt-lookup drafts the verify
# actually accepts, on real code-edit prompts, counted separately for the
# reasoning and answer segments — the measurement 9.1's break-even has to be
# tested against. Re-runs the exactness gate too (spec stream == plain greedy).
# Build with USE_CUDA=1 on a CUDA host; run under QWEN35_FUSE=0x37.
tests/bench_spec_accept: tests/bench_spec_accept.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_spec_accept.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Serve-scheduler gate (L1/L2/L3 — batched+chunked prefill, in-driver
# sampling): concurrent sg_run_completion_batched users vs the single-user
# reference (greedy text must be identical), join-during-decode, slow-client
# isolation. Build with USE_CUDA=1 on a CUDA host; run under QWEN35_FUSE=0x37.
tests/test_serve_sched_qwen35: tests/test_serve_sched_qwen35.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_sched_qwen35.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# PC-batched gate: prefix caching wired into the MULTI-TENANT batched path.
# 4 concurrent users x 3 growing turns through the batch scheduler with
# session affinity + delta-prefill; every user's per-turn reply must be
# token-identical (argmax-exact) to a non-cached single-user reference. Build
# with USE_CUDA=1 on a CUDA host; run under QWEN35_BATCH_SLOTS=1 QWEN35_FUSE=0x37.
tests/test_serve_sched_prefix_cache: tests/test_serve_sched_prefix_cache.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_sched_prefix_cache.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# True per-user capacity bench — realistic growing-context workload with prefix
# caching ON (steady-state delta-prefill), K concurrent users through the
# batched serve path. Reports per-user + aggregate tok/s + cost/1M. Build with
# USE_CUDA=1 on a CUDA host; run under QWEN35_BATCH_SLOTS=1 QWEN35_FUSE=0x37.
tests/bench_serve_prefix_cache: tests/bench_serve_prefix_cache.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_serve_prefix_cache.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Prefix-cache gate + latency bench (pn_prefix_cache): a multi-turn agentic
# scenario run with caching ON (delta-prefill only) vs OFF (re-prefill the
# whole context each turn) must produce token-identical replies (argmax-exact),
# and the cached per-turn prefill must be far cheaper. Build with USE_CUDA=1 on
# a CUDA host; run under QWEN35_FUSE=0x37 for the strict argmax gate.
tests/test_prefix_cache_qwen35: tests/test_prefix_cache_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefix_cache_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Quality-eval driver (perplexity + greedy generation) for A/B-ing two engine
# builds whose logits are not bit-identical; driven by evals/eval_quality.sh.
# Build with USE_CUDA=1 on a CUDA host for the GPU path. Needs a model.
tests/eval_ppl: tests/eval_ppl.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/eval_ppl.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_prefix_projection: tests/test_prefix_projection.c $(PN_SRC) $(FOUNDRY_SRC) \
		src/serve/serve_prompt_tokens.c src/serve/tool_calling.c \
		src/serve/json_value_grammar.c \
		src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefix_projection.c $(PN_SRC) $(FOUNDRY_SRC) \
		src/serve/serve_prompt_tokens.c src/serve/tool_calling.c \
		src/serve/json_value_grammar.c \
		src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_bpe_pretokenize: tests/test_bpe_pretokenize.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_bpe_pretokenize.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Clean-room property tests: pre-token split oracle over every code point, and
# bit-exact dequant/half-conversion checks against per-element references.
tests/test_pretok_props: tests/test_pretok_props.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_pretok_props.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_dequant_props: tests/test_dequant_props.c src/formats/gguf_dequant.c src/formats/gguf.c \
		src/formats/gguf_dequant.h src/formats/gguf.h
	$(CC) $(CFLAGS) -o $@ tests/test_dequant_props.c src/formats/gguf_dequant.c src/formats/gguf.c -lm

tests/test_rwkvx: tests/test_rwkvx.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_rwkvx.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_qwen35moe: tests/test_qwen35moe.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35moe.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Batched-prefill causality (reference-free) on every qwen35 batched path the build has:
# CUDA forward_batch + 9b/9c, TT prefill. SKIPs without PN_QWEN35_MODEL or such a path.
# Red proof: tests/causality_red/run_red.sh.
tests/test_qwen35_prefill_causality: tests/test_qwen35_prefill_causality.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_prefill_causality.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Metal kernel-coverage gate: every weight the forward DISPATCHES must have a
# Metal matvec kernel registered for its quant type, or the stock mixed-quant
# GGUF turns into garbage logits at Phase-5 bring-up.  Built TWICE from one
# source: the `_fake` binary links a stub fm_matvec_q5k, so Q5_K must flip to
# covered while the other missing types do not.  A gate that printed a constant
# could not tell the two binaries apart — that difference IS the proof it
# measures registration.  Both are model-gated on PN_QWEN35_MODEL.
tests/test_qwen35moe_metal_coverage: tests/test_qwen35moe_metal_coverage.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35moe_metal_coverage.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_qwen35moe_metal_coverage_fake: tests/test_qwen35moe_metal_coverage.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -DPN_COVERAGE_FAKE_KERNEL -o $@ \
		tests/test_qwen35moe_metal_coverage.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Losslessness gate for greedy speculative decoding: proves the speculative
# stream is bit-identical to plain greedy decode for any draft. Needs a model.
tests/test_spec_decode: tests/test_spec_decode.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_decode.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# End-to-end losslessness + realized-speedup measurement for the speculative
# generation loop (pn_spec_generate). Needs a model.
tests/test_spec_generate: tests/test_spec_generate.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_generate.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# M1 bench: verify-pass cost vs number of rows M (picks the default draft width
# and confirms the batched-verify payoff). Needs a model.
tests/bench_verify_rows: tests/bench_verify_rows.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_verify_rows.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# M2/M3 gate: pn_llama_verify_batch ragged multi-sequence verify is bit-identical
# (memcmp) to an independent per-slot forward_batch replay. Needs a model.
tests/test_verify_batch: tests/test_verify_batch.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_verify_batch.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# M4/M5 gate: batched speculative decode (pn_batch_decoder_step_spec) emits a
# token stream bit-identical to plain forward and plain batch decode for any
# draft, across batch sizes, K widths, mixed spec/plain slots, and EOS. Needs a model.
tests/test_spec_batch: tests/test_spec_batch.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_batch.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# MemoryAware Phase 1 unit test (host signals + memwatch publish/read + level
# derivation + reserve policy). Self-contained: links only the mem module + the
# host-signal reader + libm, so it builds and runs on the Mac (no model/GPU).
MEM_SRC = src/core/mem/pn_host_signals.c src/core/mem/pn_memwatch.c \
	src/core/mem/pn_admission.c src/core/mem/pn_ladder.c \
	src/serve/signal_handlers.c
tests/test_memwatch: tests/test_memwatch.c $(MEM_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_memwatch.c $(MEM_SRC) $(CUDA_SRC) -lm $(LDFLAGS) $(LDFLAGS_CUDA)

# Die-manager unit test (Phase 7 placement primitive). Self-contained: logical
# dies only — links just the die-manager module, no model/GPU, so it builds
# and runs on the Mac. The pn_dm_init_cuda probe is validated on the K80 test machine.
tests/test_die_manager: tests/test_die_manager.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_die_manager.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Model-registry unit test (Phase 7 multi-model serving). Bookkeeping asserts
# run everywhere; the model-backed part (two names resident, generation,
# unload-drain) uses models/qwen3-0.6b-Q8_0.gguf and skips if it is absent
# (override with PN_TEST_MODEL).
tests/test_model_registry: tests/test_model_registry.c src/serve/model_registry.c src/serve/batch_sched.c src/serve/request_profile.c src/serve/serve_backend_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_model_registry.c src/serve/model_registry.c src/serve/batch_sched.c src/serve/request_profile.c src/serve/serve_backend_guard.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Unload-reclaim regression (the /admin/models/unload die leak). Proves the die
# budget comes BACK — a DIFFERENT model loads onto the SAME die in the SAME
# process, on a die sized to fit exactly one — and that an unload which cannot
# drain REFUSES instead of reporting success and stranding the dies. Same link
# set as test_model_registry; logical dies only, so it runs on the Mac.
tests/test_model_unload_reclaim: tests/test_model_unload_reclaim.c src/serve/model_registry.c src/serve/batch_sched.c src/serve/request_profile.c src/serve/serve_backend_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_model_unload_reclaim.c src/serve/model_registry.c src/serve/batch_sched.c src/serve/request_profile.c src/serve/serve_backend_guard.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The silent-deploy class: compiled-backend identity, the refuse-to-start guard
# for a `--dies` this binary cannot honour, and the SERVED-vs-configured
# concurrency the two JSON surfaces advertise. Links the whole serve + engine
# (like test_serve_sched_qwen35) so the JSON assertions run the REAL
# oai_handle_models / ma_handle_models over a socketpair rather than a
# reimplementation of them. No GPU, no model file, no network.
tests/test_backend_guard: tests/test_backend_guard.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_backend_guard.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Dispatch-layer bs_get_load() accessor test (Phase G0). Exercises the pure
# slot/queue snapshot (sched==NULL single-user + hand-built cap=4 states);
# links batch_sched.c + the engine (batch_sched.c's other fns need it), no GPU.
tests/test_bs_load: tests/test_bs_load.c src/serve/batch_sched.c src/serve/request_profile.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_bs_load.c src/serve/batch_sched.c src/serve/request_profile.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# A4: unit tests for pn_gguf_dequantize_row (per-row byte-offset math +
# per-row == slice-of-full-tensor) and pn_qwen35moe_model_embed_row.
# Part A always runs (synthetic data); Part B needs PN_TEST_MODEL (a real GGUF).
tests/test_embed_row: tests/test_embed_row.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_embed_row.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The column-tile permutation that de-interleaves a joint Q+gate projection. Links ONLY the
# module under test: it has no device, no artifact and no USE_TT, which is the point — the
# de-interleave is the one Phase 5 item with no proven shape to inherit, so its arithmetic
# is settled with no card and on any host.
tests/test_tt_column_runs: tests/test_tt_column_runs.c src/accel/tt/pn_tt_column_runs.c \
                           src/accel/tt/pn_tt_column_runs.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_column_runs.c src/accel/tt/pn_tt_column_runs.c -lm

# The elementwise core split's tile arithmetic. Links ONLY the module under test — no device,
# no artifact, no USE_TT — because the one thing a split can get wrong that nothing reports is
# an ADDRESS, and that is settled here rather than on a card.
tests/test_tt_name_list: tests/test_tt_name_list.c \
	src/accel/tt/pn_tt_name_list.c src/accel/tt/pn_tt_name_list.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_name_list.c \
	  src/accel/tt/pn_tt_name_list.c -lm

# The staging pipeline's release contract, off hardware: a released item is always built.
# Build it with -DPN_TT_STAGE_PIPELINE_RED to see it fail.
tests/test_tt_stage_pipeline: tests/test_tt_stage_pipeline.c \
	src/accel/tt/pn_tt_stage_pipeline.c src/accel/tt/pn_tt_stage_pipeline.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_stage_pipeline.c src/accel/tt/pn_tt_stage_pipeline.c

tests/test_tt_eltwise_split: tests/test_tt_eltwise_split.c \
	src/accel/tt/pn_tt_eltwise_split.c src/accel/tt/pn_tt_eltwise_split.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_eltwise_split.c \
	  src/accel/tt/pn_tt_eltwise_split.c -lm

tests/test_tt_descriptor_census: tests/test_tt_descriptor_census.c \
	src/accel/tt/pn_tt_descriptor_census.c src/accel/tt/pn_tt_descriptor_census.h \
	src/accel/tt/pn_tt_name_list.c src/accel/tt/pn_tt_name_list.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_descriptor_census.c \
	  src/accel/tt/pn_tt_descriptor_census.c src/accel/tt/pn_tt_name_list.c -lm

# Self-contained differential parity test for the optimized token sampler:
# diffs a verbatim copy of the original sampler against the optimized one over
# thousands of random cases. No model/engine deps, so it links only libm.
tests/test_sampler_parity: tests/test_sampler_parity.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_parity.c -lm

# GPU-vs-CPU parity harness for the qwen35moe CUDA kernels. Always links the
# Foundry CUDA backend (dlopen, -ldl) so it builds + runs everywhere: on a
# host with no CUDA driver it self-gates on fc_available() and prints
# [SKIP] no CUDA. The actual GPU diffs run on a K80 host. The backend is
# pulled in directly (not via USE_CUDA) so the harness is self-contained.
tests/test_kernel_parity: tests/test_kernel_parity.c $(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c $(ALL_HDR)
	$(CC) $(CFLAGS) -DUSE_CUDA -o $@ tests/test_kernel_parity.c \
		$(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c -lm -ldl $(LDFLAGS)

# GATING smoke test for the 2-die barswap context swap (Task 1). Proves two
# live CUDA contexts (one per K80 die, both ordinal 0) coexist and survive
# bs_acquire/release multiplexing through the single MMIO slot. Self-gates on
# fc_driver_load(): on a host with no CUDA driver it prints [SKIP] no CUDA.
# Linked like test_kernel_parity (fc_cuda.c + barswap.c, dlopen -ldl). The
# operator runs this FIRST — it gates the 2-die layer-split forward.
tests/test_2die_swap: tests/test_2die_swap.c $(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c $(ALL_HDR)
	$(CC) $(CFLAGS) -DUSE_CUDA -o $@ tests/test_2die_swap.c \
		$(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c -lm -ldl $(LDFLAGS)

# Regression guard for the GLOBAL stream-capture bug (task #28). Captures a
# graph on one die while a second thread allocates/syncs on another, and fails
# if either side is disturbed — which is what CU_STREAM_CAPTURE_MODE_GLOBAL did
# process-wide, showing up as unexplained 900/901 in the production log. Needs
# two visible devices; self-gates to [SKIP] otherwise. Run it on idle dies,
# e.g. CUDA_VISIBLE_DEVICES=2,3 tests/test_graph_capture_concurrency.
# Linked like test_kernel_parity (fc_cuda.c + barswap.c, dlopen -ldl).
tests/test_graph_capture_concurrency: tests/test_graph_capture_concurrency.c $(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c $(ALL_HDR)
	$(CC) $(CFLAGS) -DUSE_CUDA -o $@ tests/test_graph_capture_concurrency.c \
		$(FOUNDRY)/cuda/fc_cuda.c $(FOUNDRY)/barswap/barswap.c -lm -ldl -lpthread $(LDFLAGS)

# CPU-reference golden-vector generator for the qwen35moe GPU kernels (no model).
# Builds the generator, then runs it to (re)produce tests/golden/*.bin.
tests/gen_golden_qwen35: tests/gen_golden_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/gen_golden_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

.PHONY: golden
golden: tests/gen_golden_qwen35
	./tests/gen_golden_qwen35

# Converter CLI
pantheon-convert: cli/convert.c $(FOUNDRY)/pth/pth.c $(FOUNDRY)/safetensors/safetensors.c
	$(CC) $(CFLAGS) -o $@ cli/convert.c $(FOUNDRY)/pth/pth.c $(FOUNDRY)/safetensors/safetensors.c -lm

# Clean. Test/bench/debug binaries go by pattern (sources are .c/.h), the same rule as
# .gitignore, so a new test cannot be missed.
clean:
	rm -f pantheon pantheon-convert libpantheon.a $(foreach b,cpu cuda tt metal cuda-tt cuda-metal tt-metal cuda-tt-metal,pantheon-$(b))
	find tests -maxdepth 1 -type f \( -name 'test_*' -o -name 'bench_*' -o -name 'gen_golden_*' \) ! -name '*.c' ! -name '*.h' -delete
	rm -rf *.dSYM tests/*.dSYM
	rm -f *.o src/core/*.o src/formats/*.o src/models/*.o src/memory/*.o src/accel/*.o
	rm -rf $(LIB_OBJDIR)
