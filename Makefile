# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Pantheon Makefile
#
# Builds Pantheon sources + Foundry sources into a single binary.
#
# Foundry location, in precedence order:
#   1. FOUNDRY=/path/to/foundry   explicit override
#   2. ./foundry                  the git submodule in the public repository (git
#                                 clone --recursive, or git submodule update
#                                 --init), or a symlink in a development checkout
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
	$(FOUNDRY)/tt/ft_tt_dram_nearest.c \
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
	$(FOUNDRY)/tt/ft_tt_op_norm_role.c \
	$(FOUNDRY)/tt/ft_tt_op_attention.c \
	$(FOUNDRY)/tt/ft_tt_op_gdn.c \
	$(FOUNDRY)/tt/ft_tt_op_gdnc.c \
	$(FOUNDRY)/tt/ft_tt_op_replicas.c \
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
	$(FOUNDRY)/tt/convert/ft_bfp.c \
	$(THREADPOOL_SRC) \
	$(METAL_SRC) \
	$(CUDA_SRC) \
	$(TT_SRC)

# ============================================================
# Pantheon sources
# ============================================================

# Host-RAM admission, for tests that link a routed source without PN_SRC.
HOST_ADMIT_SRC = src/core/mem/pn_host_admission.c src/core/mem/pn_host_signals.c

# FTTW tile decoding (the card-format oracle, gguf_tiled.h): every target that links
# gguf_dequant.c links these too. foundry's ft_bfp.c is the one block-float codec.
TILED_SRC = src/formats/gguf_tiled.c $(FOUNDRY)/tt/convert/ft_bfp.c

PN_SRC = \
	src/core/engine.c \
	src/core/load_guard.c \
	src/core/load_progress.c \
	src/core/mem/pn_host_signals.c \
	src/core/mem/pn_memwatch.c \
	src/core/mem/pn_admission.c \
	src/core/mem/pn_host_admission.c \
	src/core/mem/pn_session_host_fit.c \
	src/core/mem/pn_kv_geometry.c \
	src/core/mem/pn_ladder.c \
	src/core/mem/pn_lazy_zero.c \
	src/core/mem/pn_host_heap.c \
	src/core/bench.c \
	src/core/sample_profiler.c \
	src/core/act_stats.c \
	src/core/act_stats_file.c \
	src/core/act_stats_xtx.c \
	src/core/qweight.c \
	src/core/qweight_tiled.c \
	src/core/sampling_modifiers.c \
	src/core/host_stage_timer.c \
	src/core/host_row_dump.c \
	src/core/sampler_margin_draw.c \
	src/core/grammar_mask_shortlist.c \
	src/core/grammar_row_dump.c \
	src/core/sampler_topn_tail.c \
	src/core/sampler_topn_surgery.c \
	src/core/chat.c \
	src/core/bpe_pretokenize.c \
	src/core/ngram_draft.c \
	src/core/spec_sampling_rule.c \
	src/core/spec_sampling_chooser.c \
	src/core/spec_depth.c \
	src/core/dump_reqlog.c \
	src/core/spec_mtp.c \
	src/core/spec_mtp_step.c \
	src/core/spec_mtp_couple.c \
	src/core/prefix_cache.c \
	src/core/session_snapshot.c \
	src/memory/arena.c \
	src/accel/simd.c \
	src/accel/die_manager.c \
	src/accel/die_layer_split.c \
	src/accel/backend_identity.c \
	src/accel/tt/pn_tt_device.c \
	src/accel/tt/pn_tt_matmul_group.c \
	src/accel/tt/pn_tt_core_list.c \
	src/accel/tt/pn_tt_matmul_pull.c \
	src/accel/tt/pn_tt_unmirrored_probe.c \
	src/accel/tt/pn_tt_weight_banks.c \
	src/accel/tt/pn_tt_residency.c \
	src/accel/tt/pn_tt_group_pull.c \
	src/accel/tt/pn_tt_moe.c \
	src/accel/tt/pn_tt_issue.c \
	src/accel/tt/pn_tt_core_boot_guard.c \
	src/accel/tt/pn_tt_oncard.c \
	src/accel/tt/pn_tt_eltwise_group.c \
	src/accel/tt/pn_tt_eltwise_split.c \
	src/accel/tt/pn_tt_norm_schedule.c \
	src/accel/tt/pn_tt_norm_role.c \
	src/accel/tt/pn_tt_norm_role_plan.c \
	src/accel/tt/pn_tt_role_table.c \
	src/accel/tt/pn_tt_reader.c \
	src/accel/tt/pn_tt_attention.c \
	src/accel/tt/pn_tt_attention_lanes.c \
	src/accel/tt/pn_tt_attention_splitk.c \
	src/accel/tt/pn_tt_attention_grid.c \
	src/accel/tt/pn_tt_attn_deep.c \
	src/accel/tt/pn_tt_lend_ledger.c \
	src/accel/tt/pn_tt_prefill_lend.c \
	src/accel/tt/pn_tt_attn_splitk.c \
	src/accel/tt/pn_tt_attn_fast.c \
	src/accel/tt/pn_tt_grid_probe.c \
	src/accel/tt/pn_tt_grid_msplit.c \
	src/accel/tt/pn_tt_grid_marks.c \
	src/accel/tt/pn_tt_kv_banks.c \
	src/accel/tt/pn_tt_kv_append_split.c \
	src/accel/tt/pn_tt_kv_format.c \
	src/accel/tt/pn_tt_kv_rotate.c \
	src/accel/tt/pn_tt_gdn.c \
	src/accel/tt/pn_tt_gdn_checkpoint.c \
	src/accel/tt/pn_tt_gdn_verify_slots.c \
	src/accel/tt/pn_tt_gdn_conv_share.c \
	src/accel/tt/pn_tt_gdn_conv_split.c \
	src/accel/tt/pn_tt_matmul_rows.c \
	src/accel/tt/pn_tt_gdnc.c \
	src/accel/tt/pn_tt_gdngate.c \
	src/accel/tt/pn_tt_stage_pipeline.c \
	src/accel/tt/pn_tt_head_norm.c \
	src/accel/tt/pn_tt_render.c \
	src/accel/tt/pn_tt_gate.c \
	src/accel/tt/pn_tt_fp32_strip.c \
	src/accel/tt/pn_tt_column_runs.c \
	src/accel/tt/pn_tt_weight_view.c \
	src/accel/tt/pn_tt_weight_split.c \
	src/accel/tt/pn_tt_strip_placement.c \
	src/accel/tt/pn_tt_mixw_fit.c \
	src/accel/tt/pn_tt_prefill_reserve.c \
	src/accel/tt/pn_tt_descriptor_census.c \
	src/accel/tt/pn_tt_name_list.c \
	src/accel/tt/pn_tt_gdn_layout.c \
	src/accel/tt/pn_tt_gdn_core_scaling.c \
	src/accel/tt/pn_tt_replica_map.c \
	src/accel/tt/pn_tt_late_mirrors.c \
	src/accel/tt/pn_tt_swiglu_fuse.c \
	src/accel/tt/pn_tt_resadd_fuse.c \
	src/accel/tt/pn_tt_dispatch.c \
	src/accel/tt/pn_tt_dispatch_hoist.c \
	src/accel/tt/pn_tt_precondition.c \
	src/accel/tt/pn_tt_precondition_verify.c \
	src/accel/tt/pn_tt_precondition_dump.c \
	src/accel/tt/pn_tt_host_gap.c \
	src/accel/tt/pn_tt_topk_tail.c \
	src/accel/tt/pn_tt_topk_rows.c \
	src/formats/gguf.c \
	src/formats/gguf_dequant.c \
	src/formats/gguf_tiled.c \
	src/formats/gguf_fttw_overlay.c \
	src/formats/gguf_fttw_repack.c \
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
	src/models/qwen35moe_mtp_ring.c \
	src/models/qwen35moe.c \
	src/models/qwen35moe_die_projection.c \
	src/models/qwen35moe_attn.c \
	src/models/qwen35moe_ssm.c \
	src/models/qwen35moe_gate_tables.c \
	src/models/qwen35moe_moe.c \
	src/models/qwen35moe_routing_dump.c \
	src/models/qwen35moe_forward.c \
	src/models/qwen35moe_forward_cuda.c \
	src/models/qwen35moe_forward_tt.c \
	src/models/qwen35moe_resident_tt.c \
	src/models/qwen35moe_moe_tt.c \
	src/models/qwen35moe_gdn_tt.c \
	src/models/qwen35moe_gdn_strip_tt.c \
	src/models/qwen35moe_attn_tt.c \
	src/models/qwen35moe_prefill_tt.c \
	src/models/qwen35moe_prefill_checkpoint_tt.c \
	src/models/qwen35moe_fit_tt.c \
	src/models/qwen35moe_tt_fit_plan.c \
	src/models/qwen35moe_tt_mixw_census.c \
	src/models/qwen35moe_mtp_prompt_defer_tt.c \
	src/models/qwen35moe_outproj_mirrors_tt.c \
	src/models/qwen35moe_checkpoint_band.c \
	src/models/qwen35moe_prefill_prefetch_tt.c \
	src/models/qwen35moe_prefill_overlap_tt.c \
	src/models/qwen35moe_prefill_ov_plan.c \
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
	src/serve/field_pattern.c \
	src/serve/field_patterns_request.c \
	src/serve/response_format_grammar.c \
	src/serve/serve_emitter.c \
	src/serve/json_canonical_stream.c \
	src/serve/tool_call_stream.c \
	src/serve/repetition_loop_detector.c \
	src/serve/serve_prompt_tokens.c \
	src/serve/serve_spec_lookup.c \
	src/serve/serve_spec_census.c \
	src/serve/serve_spec_mtp_cuda.c \
	src/serve/serve_request_dump.c \
	src/serve/serve_dump_reqlog.c \
	src/serve/serve_generation.c \
	src/serve/context_over_budget.c \
	src/serve/model_registry.c \
	src/serve/model_replica_route.c \
	src/serve/replica_affinity.c \
	src/serve/replica_pin.c \
	src/serve/model_session.c \
	src/serve/batch_sched.c \
	src/serve/batch_sched_slots.c \
	src/serve/batch_sched_turn_checkpoint.c \
	src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c \
	src/serve/batch_sched_spec.c \
	src/serve/base64.c \
	src/serve/model_admin.c src/serve/load_env_scope.c \
	src/serve/openai_chat.c \
	src/serve/sampling_defaults.c \
	src/serve/request_profile.c \
	src/serve/request_timing_line.c \
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

# Foundry's Ingot codec (foundry ingot/): Pantheon Local decodes catalog .ingot downloads
# (src/local/ingot_fetch.c). An archive compiled with the codec's own flags — never
# -ffast-math — and listed last in SERVE_SRC so every link of the serve resolves against it.
# safetensors.c is in it for the module tests; links that compile FOUNDRY_SRC take theirs.
# Switching build configs needs `rm -rf build`, as for libpantheon.a.
INGOT_OBJDIR = build/ingot
INGOT_LIB    = $(INGOT_OBJDIR)/libpn_ingot.a
# Every codec source foundry's ingot/ has (a new foundry file links without a Makefile change).
INGOT_SRC = $(wildcard $(FOUNDRY)/ingot/core/*.c $(FOUNDRY)/ingot/codec/*.c \
	$(FOUNDRY)/ingot/container/*.c) $(FOUNDRY)/tt/convert/ft_sha256.c \
	$(FOUNDRY)/safetensors/safetensors.c
INGOT_OBJS = $(addprefix $(INGOT_OBJDIR)/,$(notdir $(INGOT_SRC:.c=.o)))
INGOT_HDR  = $(wildcard $(FOUNDRY)/ingot/*/*.h) $(FOUNDRY)/tt/convert/ft_sha256.h \
	$(FOUNDRY)/safetensors/safetensors.h
INGOT_CFLAGS = -std=c11 -O2 $(ARCH_FLAGS) -Wall -Wextra -pedantic -pthread -I$(FOUNDRY) \
	-D_DARWIN_C_SOURCE -D_DEFAULT_SOURCE

# Pantheon Local (appliance mode, `pantheon local`). Its own folder; linked into
# the CLI with the serve (docs/local/README.md).
LOCAL_SRC = $(wildcard src/local/*.c)
# Monocypher 4.0.3 (lib/monocypher/README.md): Ed25519 + BLAKE2b for the catalog
# signature. Vendored, CC0-1.0 OR BSD-2-Clause.
MONOCYPHER_SRC = lib/monocypher/monocypher.c lib/monocypher/monocypher-ed25519.c
SERVE_SRC += $(LOCAL_SRC) $(MONOCYPHER_SRC) $(INGOT_LIB)
# make LOCAL_TEST_CATALOG_KEY=1: also trust tests/local/fixtures/test_catalog.pub.
# TEST ONLY — for the on-hardware e2e scripts; never for a shipped build.
ifdef LOCAL_TEST_CATALOG_KEY
  CFLAGS += -DPN_LOCAL_TEST_CATALOG_KEY
endif

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
# The Ingot objects (minus safetensors, which FOUNDRY_SRC already brings).
LIB_OBJS  += $(filter-out %/safetensors.o,$(INGOT_OBJS))
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

# The Ingot archive (INGOT_SRC above), with the codec's own flags.
define INGOT_OBJ_RULE
$(INGOT_OBJDIR)/$(notdir $(1:.c=.o)): $(1) $(INGOT_HDR)
	@mkdir -p $(INGOT_OBJDIR)
	$$(CC) $$(INGOT_CFLAGS) -c $(1) -o $$@
endef
$(foreach s,$(INGOT_SRC),$(eval $(call INGOT_OBJ_RULE,$(s))))

$(INGOT_LIB): $(INGOT_OBJS)
	@rm -f $@
	ar rcs $@ $(INGOT_OBJS)

# Pantheon Local tests, defined before test: so its prerequisite list sees them
# (a rule's prerequisites expand when the rule is read).
LOCAL_TESTS_FULL = tests/test_local_empty_start tests/test_local_hw_report tests/test_local_keys \
	tests/test_local_flow tests/test_local_reload_placement tests/test_local_replicas \
	tests/test_local_load_progress tests/test_load_guard tests/test_local_unit_types \
	tests/test_load_budget_release tests/test_local_replica_pin tests/test_replica_pin_serve \
	tests/test_local_pairing_devices tests/test_local_people_routes tests/test_local_hook_adopt \
	tests/test_local_host_routes tests/test_local_session_routes tests/test_local_relay_routes
LOCAL_TESTS = tests/test_sampling_defaults tests/test_local_unit_place tests/test_local_mdns_packet tests/test_local_pick_port \
	tests/test_local_minisign tests/test_local_catalog tests/test_local_download \
	tests/test_local_loaded_set tests/test_local_load_options tests/test_replica_route \
	tests/test_local_store_cleanup tests/test_local_people_store tests/test_local_key_migration \
	tests/test_local_route_classes tests/test_local_session_registry tests/test_local_hosts_hub \
	tests/test_local_attach_ticket tests/test_local_hub_relays $(LOCAL_TESTS_FULL)
# Ingot Phase 1b: .ingot catalog downloads.
LOCAL_TESTS += tests/test_local_ingot
# People registry P5: loads, keys and seats by person.
LOCAL_TESTS_FULL += tests/test_local_loads_by_person tests/test_local_serve_principal \
	tests/test_local_seats

# Build + run tests
test: tests/test_backend_guard tests/test_build tests/test_gguf tests/test_gguf_layer_bytes tests/test_admission_reserve tests/test_context_budget tests/test_reasoning_budget tests/test_host_admission tests/test_host_admit_serve tests/test_die_scratch_ctx tests/test_qwen35_die_projection tests/test_die_layer_split tests/test_arena tests/test_qmv tests/test_tiled_fast_decode tests/test_simd tests/test_simd_props tests/test_chat tests/test_tool_calling tests/test_tool_call_defects tests/test_tool_call_truncation tests/test_empty_argument tests/test_frame_capture tests/test_json_value_grammar tests/test_tool_grammar tests/test_tool_grammar_json tests/test_tool_grammar_reasoned tests/test_tool_grammar_json_reasoned tests/test_field_pattern tests/test_tool_grammar_field_patterns tests/test_tool_grammar_json_field_patterns tests/test_tool_grammar_no_repeat_keys tests/test_field_pattern_steer tests/test_tool_grammar_forced_close_call tests/test_grammar_spec_chooser tests/test_grammar_mask_shortlist tests/test_grammar_topn_sampler tests/test_grammar_repeat_penalty tests/test_topn_surgery tests/test_grammar_exact_bound tests/test_grammar_dense_shortlist tests/test_grammar_select_shortlist tests/test_sampler_margin_draw tests/test_response_format_grammar tests/test_sampling_modifiers tests/test_chat_request tests/test_serve_emitter tests/test_tool_call_stream tests/test_repetition_loop_detector tests/test_serve_spec_lookup tests/test_registry tests/test_base64 tests/test_prefix_cache_sys tests/test_host_heap_churn tests/test_sampler_thread_scratch tests/test_llama_state_snapshot tests/test_prefix_projection tests/test_think_history_as_sent tests/test_think_history_drop_last tests/test_template_parity_qwen tests/test_session_watch tests/test_session_affinity tests/test_bs_load tests/test_bs_finalize_yield tests/test_request_timing_line tests/test_bs_prefill_share tests/test_bs_pd_budget tests/test_decode_watchdog tests/test_think_flip_backlog tests/test_bpe_pretokenize tests/test_pretok_props tests/test_dequant_props tests/test_rwkvx tests/test_qwen35moe tests/test_qwen35_prefill_causality tests/test_qwen35moe_metal_coverage tests/test_qwen35moe_metal_coverage_fake tests/test_die_manager tests/test_model_registry tests/test_model_unload_reclaim tests/test_unload_closes_watch tests/test_embed_row tests/test_tt_column_runs tests/test_qwen35_ckpt_band tests/test_tt_name_list tests/test_tt_core_list tests/test_tt_eltwise_split tests/test_tt_norm_schedule tests/test_tt_norm_role tests/test_tt_role_table tests/test_tt_descriptor_census tests/test_tt_q38_mixw_fit tests/test_tt_q38_fit_banks tests/test_tt_q38_fit_levers tests/test_tt_weight_view tests/test_tt_gate_tables tests/test_tt_reader_runs tests/test_tt_grid_msplit tests/test_tt_grid_marks tests/test_qwen35_ov_plan tests/test_tt_attn_splitk tests/test_tt_dispatch_shape_key tests/test_tt_attn_fast tests/test_tt_attn_deep tests/test_tt_lend_ledger tests/test_tt_dispatch_hoist tests/test_tt_precondition tests/test_tt_gdn_conv_split tests/test_tt_matmul_rows tests/test_tt_stage_pipeline tests/test_graph_capture_retry tests/test_attn_ring_retry tests/test_mtp_ring_clear tests/test_serve_shutdown tests/test_spec_depth tests/test_spec_mtp_couple tests/test_spec_grammar_draw tests/test_qwen35_tp_forget tests/test_qwen35_tp_restore_sync tests/test_dump_reqlog tests/test_act_stats tests/test_act_stats_model tests/test_act_stats_xtx tools/oracle/act_stats_merge tests/test_load_env_scope $(LOCAL_TESTS)
	./tests/test_backend_guard
	./tests/test_build
	./tests/test_gguf
	./tests/test_gguf_layer_bytes
	./tests/test_admission_reserve
	./tests/test_context_budget
	./tests/test_reasoning_budget
	./tests/test_host_admission
	./tests/test_host_admit_serve
	./tests/test_die_scratch_ctx
	./tests/test_qwen35_die_projection
	./tests/test_die_layer_split
	./tests/test_arena
	./tests/test_qmv
	./tests/test_tiled_fast_decode
	PN_TILED_RED=1 ./tests/test_tiled_fast_decode
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
	./tests/test_tool_grammar_reasoned
	./tests/test_tool_grammar_json_reasoned
	./tests/test_field_pattern
	./tests/test_tool_grammar_field_patterns
	./tests/test_tool_grammar_json_field_patterns
	./tests/test_tool_grammar_no_repeat_keys
	./tests/test_field_pattern_steer
	./tests/test_tool_grammar_forced_close_call
	./tests/test_grammar_spec_chooser
	./tests/test_grammar_mask_shortlist
	./tests/test_grammar_topn_sampler
	./tests/test_grammar_repeat_penalty
	./tests/test_topn_surgery
	./tests/test_topn_surgery --selftest
	./tests/test_grammar_exact_bound
	./tests/test_grammar_exact_bound --red
	./tests/test_grammar_exact_bound --selftest
	./tests/test_grammar_dense_shortlist
	./tests/test_grammar_dense_shortlist --red
	./tests/test_grammar_dense_shortlist --selftest
	./tests/test_grammar_dense_shortlist --red-hint
	./tests/test_grammar_select_shortlist
	./tests/test_grammar_select_shortlist --selftest
	./tests/test_sampler_margin_draw
	./tests/test_sampler_margin_draw --selftest
	./tests/test_sampler_margin_draw --red-wrong
	./tests/test_grammar_topn_sampler --selftest
	./tests/test_response_format_grammar
	./tests/test_sampling_modifiers
	./tests/test_chat_request
	./tests/test_serve_emitter
	./tests/test_tool_call_stream
	./tests/test_repetition_loop_detector
	./tests/test_spec_depth
	./tests/test_spec_mtp_couple
	./tests/test_spec_mtp_couple --selftest
	./tests/test_spec_grammar_draw
	./tests/test_spec_grammar_draw --selftest
	./tests/test_qwen35_tp_forget
	./tests/test_qwen35_tp_restore_sync
	./tests/test_dump_reqlog
	mkdir -p build && ./tests/test_act_stats
	mkdir -p build && ./tests/test_act_stats_model
	mkdir -p build && ./tests/test_act_stats_xtx
	./tools/oracle/act_stats_merge build/act_stats/merged.imatrix build/act_stats/h1.imatrix \
		build/act_stats/h2.imatrix && cmp build/act_stats/merged.imatrix build/act_stats/full.imatrix
	./tests/test_load_env_scope
	./tests/test_registry
	./tests/test_base64
	./tests/test_prefix_cache_sys
	./tests/test_host_heap_churn
	./tests/test_sampler_thread_scratch
	./tests/test_llama_state_snapshot
	./tests/test_prefix_projection
	./tests/test_think_history_as_sent
	./tests/test_think_history_drop_last
	./tests/test_template_parity_qwen
	./tests/test_session_watch
	./tests/test_session_affinity
	./tests/test_bs_load
	./tests/test_bs_finalize_yield
	./tests/test_request_timing_line
	./tests/test_bs_prefill_share
	./tests/test_bs_pd_budget
	./tests/test_decode_watchdog
	./tests/test_think_flip_backlog
	./tests/test_think_flip_backlog --red
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
	./tests/test_unload_closes_watch
	./tests/test_embed_row
	./tests/test_tt_column_runs
	./tests/test_qwen35_ckpt_band
	./tests/test_tt_eltwise_split
	./tests/test_tt_norm_schedule
	./tests/test_tt_norm_role
	./tests/test_tt_role_table
	./tests/test_tt_descriptor_census
	./tests/test_tt_q38_mixw_fit
	./tests/test_tt_q38_fit_banks
	./tests/test_tt_q38_fit_levers
	./tests/test_tt_weight_view
	./tests/test_tt_gate_tables
	./tests/test_tt_reader_runs
	./tests/test_tt_grid_msplit
	./tests/test_tt_grid_marks
	./tests/test_qwen35_ov_plan
	./tests/test_tt_core_list
	./tests/test_tt_attn_splitk
	./tests/test_tt_dispatch_shape_key
	./tests/test_tt_attn_fast

	./tests/test_tt_attn_deep
	./tests/test_tt_lend_ledger
	./tests/test_tt_dispatch_hoist
	./tests/test_tt_precondition
	./tests/test_tt_gdn_conv_split
	./tests/test_tt_matmul_rows
	./tests/test_graph_capture_retry
	./tests/test_graph_capture_retry --latch
	./tests/test_attn_ring_retry
	./tests/test_attn_ring_retry --latch
	./tests/test_mtp_ring_clear
	./tests/test_mtp_ring_clear --no-clear
	./tests/test_serve_shutdown
	@for t in $(LOCAL_TESTS); do echo "./$$t"; ./$$t || exit 1; done

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

tests/test_context_budget: tests/test_context_budget.c src/serve/context_over_budget.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_context_budget.c src/serve/context_over_budget.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The per-turn reasoning plan (answer reserve + line-end close); header-only.
tests/test_reasoning_budget: tests/test_reasoning_budget.c src/serve/reasoning_budget.h
	$(CC) $(CFLAGS) -o $@ tests/test_reasoning_budget.c

tests/test_host_admission: tests/test_host_admission.c src/core/mem/pn_host_admission.c src/core/mem/pn_host_signals.c src/core/mem/pn_host_admission.h
	$(CC) $(CFLAGS) -o $@ tests/test_host_admission.c src/core/mem/pn_host_admission.c \
		src/core/mem/pn_host_signals.c -lpthread

tests/test_die_layer_split: tests/test_die_layer_split.c src/accel/die_layer_split.c src/accel/die_layer_split.h
	$(CC) $(CFLAGS) -o $@ tests/test_die_layer_split.c src/accel/die_layer_split.c -lm

tests/test_qwen35_die_projection: tests/test_qwen35_die_projection.c src/models/qwen35moe_die_projection.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_die_projection.c src/models/qwen35moe_die_projection.c -lm

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
		src/formats/gguf_dequant.c src/formats/gguf.c $(TILED_SRC) $(HOST_ADMIT_SRC) -lm

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
tests/test_tool_call_truncation: tests/test_tool_call_truncation.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_call_truncation.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
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
tests/test_tool_grammar: tests/test_tool_grammar.c src/serve/tool_grammar.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar.c src/serve/tool_grammar.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

tests/test_tool_grammar_reasoned: tests/test_tool_grammar_reasoned.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_reasoned.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

tests/test_tool_grammar_json_reasoned: tests/test_tool_grammar_json_reasoned.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_json_reasoned.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# `field_patterns`: the regex (field_pattern.c) and its use in both tool-call
# grammars.  Pure C, no engine deps.
tests/test_field_pattern: tests/test_field_pattern.c src/serve/field_pattern.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_field_pattern.c src/serve/field_pattern.c -lm $(LDFLAGS)

tests/test_tool_grammar_field_patterns: tests/test_tool_grammar_field_patterns.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_field_patterns.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

tests/test_tool_grammar_json_field_patterns: tests/test_tool_grammar_json_field_patterns.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_json_field_patterns.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Each parameter key at most once per call, both dialects, and the head+tail raw
# excerpt on the dropped-call log line.  Pure C, no engine deps.
tests/test_tool_grammar_no_repeat_keys: tests/test_tool_grammar_no_repeat_keys.c src/serve/tool_grammar.c src/serve/tool_grammar_json.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_no_repeat_keys.c src/serve/tool_grammar.c src/serve/tool_grammar_json.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Mask cost per step with and without a field pattern (not part of `make test`).
tests/bench_field_pattern_mask: tests/bench_field_pattern_mask.c src/serve/tool_grammar.c src/serve/tool_grammar_json.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_field_pattern_mask.c src/serve/tool_grammar.c src/serve/tool_grammar_json.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

tests/test_tool_grammar_json: tests/test_tool_grammar_json.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_json.c src/serve/tool_grammar_json.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# Speculative decoding under the tool-call grammar: drafts verified through the
# grammar-aware chooser must reproduce plain constrained decoding token for
# token (self-falsifying simulation).  Links the real JSON grammar and the
# parser; no engine/model/device.
tests/test_grammar_spec_chooser: tests/test_grammar_spec_chooser.c src/serve/grammar_spec_chooser.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c src/serve/tool_grammar_json.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_spec_chooser.c src/serve/grammar_spec_chooser.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c src/serve/tool_grammar_json.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
		-lm $(LDFLAGS)

# The grammar mask decided from a shortlist of the highest logits must leave the
# same greedy pick / top-K set as the full-vocabulary mask.  libc-only.
tests/test_grammar_mask_shortlist: tests/test_grammar_mask_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c src/core/sampling_modifiers.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_mask_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c src/core/sampling_modifiers.c \
		-lm $(LDFLAGS)

# The shortlist select's SSE prefilter returns the scalar select's ids and bound to the bit.
tests/test_grammar_select_shortlist: tests/test_grammar_select_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_select_shortlist.c src/core/grammar_mask_shortlist.c src/core/sampler_topn_tail.c \
		-lm $(LDFLAGS)

# Full-logits ARMING WINDOW (PN_GRAMMAR_ARM_WINDOW): is the row a masked step
# reads armed one token EARLY enough?  Models the decode loop's ordering over
# the three real grammar walks; self-falsifying — exits 2 VACUOUS if the
# deliberately-stale policy fails to break.  No engine/model/CUDA deps.
tests/test_grammar_arm_window: tests/test_grammar_arm_window.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_grammar_json.c src/serve/response_format_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_arm_window.c src/serve/tool_grammar.c src/serve/json_value_grammar.c src/serve/tool_grammar_json.c src/serve/response_format_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c \
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
tests/test_chat_request: tests/test_chat_request.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c src/serve/serve_shutdown.c src/serve/serve_request_dump.c src/serve/openai_chat.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/serve/sampling_defaults.c src/serve/field_pattern.c src/serve/field_patterns_request.c src/accel/backend_identity.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_chat_request.c src/serve/sampling_defaults.c src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c src/serve/json_writer.c src/serve/request_profile.c src/serve/serve_shutdown.c src/serve/serve_request_dump.c src/serve/tool_call_stream.c src/serve/json_canonical_stream.c src/serve/serve_emitter.c src/serve/field_pattern.c src/serve/field_patterns_request.c src/accel/backend_identity.c \
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

# The MTP head's KV ring is zeroed at allocation, so a draft that attends rows nothing
# wrote (a long resumed session after a backlog reset) reads K = V = 0, not recycled
# VRAM that decodes as NaN. Host buffers in the device layout, no GPU. The red arm
# `--no-clear` replays the uncleared ring and must fail its expectations.
# -fno-fast-math: the reference attention must propagate NaN/inf the way the device does,
# or isfinite() folds to 1 and the gate is vacuous.
tests/test_mtp_ring_clear: tests/test_mtp_ring_clear.c src/models/qwen35moe_mtp_ring.c \
		src/formats/gguf_dequant.c src/formats/gguf.c $(ALL_HDR)
	$(CC) $(CFLAGS) -fno-fast-math -o $@ tests/test_mtp_ring_clear.c \
		src/models/qwen35moe_mtp_ring.c src/formats/gguf_dequant.c src/formats/gguf.c \
		$(TILED_SRC) $(HOST_ADMIT_SRC) -lm $(LDFLAGS)

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
		$(HOST_ADMIT_SRC) -lm $(LDFLAGS)

tests/test_prefix_cache_sys: tests/test_prefix_cache_sys.c tests/tt_prefix_cache_link_stubs.c src/core/prefix_cache.c src/core/session_snapshot.c src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/request_profile.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefix_cache_sys.c tests/tt_prefix_cache_link_stubs.c \
		src/core/prefix_cache.c src/core/session_snapshot.c \
		src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c \
		src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/request_profile.c \
		$(HOST_ADMIT_SRC) -lm $(LDFLAGS)

# think_history as_sent / drop_last on the batched ladder and the single-user cache: same
# hand-built-state harness as test_prefix_cache_sys (tests/think_history_ladder_harness.h).
THINK_HISTORY_LADDER_SRC = tests/tt_prefix_cache_link_stubs.c src/core/prefix_cache.c \
	src/core/session_snapshot.c src/models/qwen35moe_state_snapshot.c \
	src/models/llama_state_snapshot.c src/serve/batch_sched.c \
	src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c \
	src/serve/batch_sched_watchdog.c src/serve/request_profile.c \
	src/models/qwen35moe_checkpoint_band.c
tests/test_think_history_as_sent: tests/test_think_history_as_sent.c \
		tests/think_history_ladder_harness.h $(THINK_HISTORY_LADDER_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_think_history_as_sent.c $(THINK_HISTORY_LADDER_SRC) \
		$(HOST_ADMIT_SRC) -lm $(LDFLAGS)
tests/test_think_history_drop_last: tests/test_think_history_drop_last.c \
		tests/think_history_ladder_harness.h $(THINK_HISTORY_LADDER_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_think_history_drop_last.c $(THINK_HISTORY_LADDER_SRC) \
		$(HOST_ADMIT_SRC) -lm $(LDFLAGS)

# Liveness-based session release (§C.4). Links the watch manager + prefix cache
# + snapshot machinery + http_server (hs_client_alive); mr_acquire/mr_release +
# pn_state_* are test stubs. Host-side, deterministic (manual manager + sweep
# hooks), no engine/model/CUDA.
tests/test_session_watch: tests/test_session_watch.c tests/tt_prefix_cache_link_stubs.c src/serve/session_watch.c src/core/prefix_cache.c src/core/session_snapshot.c src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c src/serve/http_server.c src/serve/model_session.c src/serve/replica_affinity.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_session_watch.c tests/tt_prefix_cache_link_stubs.c src/serve/replica_affinity.c \
		src/serve/session_watch.c src/core/prefix_cache.c src/core/session_snapshot.c \
		src/models/qwen35moe_state_snapshot.c src/models/llama_state_snapshot.c \
		src/serve/http_server.c \
		src/serve/model_session.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/request_profile.c \
		$(HOST_ADMIT_SRC) -lm $(LDFLAGS)

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

# The qwen35 TT accessors against a NON-qwen35 model: none may read its weights as a
# pn_qwen35moe_model (the 1,891-token Argus CHAT crash). Needs USE_TT=1, NOT a card.
tests/test_tt_family_guard: tests/test_tt_family_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_family_guard.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The prefill KV append split: its per-head descriptors decode (memory kernel protocol) to the
# banked pair's placement, and spread over the cores. Needs USE_TT=1, NOT a card.
tests/test_tt_kv_append_split: tests/test_tt_kv_append_split.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_kv_append_split.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Qwen3.8-27B on one p150 at 32k: the attn_q halves, mirror-first tall placement and no tall
# attn_qg, against the tt-full-load census. Needs USE_TT=1, NOT a card.
# Also checks the shipped catalog's Qwen3.8 p150 profiles load as the shapes it fits.
TT_Q38_FIT_LOCAL_SRC = src/local/catalog_manifest.c src/local/load_options.c \
	src/local/unit_place.c src/local/catalog_match.c src/local/local_file_io.c \
	src/serve/json_reader.c src/serve/json_writer.c src/serve/sampling_defaults.c \
	src/serve/load_env_scope.c
tests/test_tt_q38_fit: tests/test_tt_q38_fit.c $(PN_SRC) $(FOUNDRY_SRC) $(TT_Q38_FIT_LOCAL_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_fit.c $(PN_SRC) $(FOUNDRY_SRC) $(TT_Q38_FIT_LOCAL_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# K4H (pn_tt_kv_rotate.h): H exact in every weight tile format, the matmul view emulated
# against a per-head rotation, q.k invariance, the state-file round trip, knob + catalog.
# Needs USE_TT=1, NOT a card.
tests/test_tt_kv_rotate: tests/test_tt_kv_rotate.c $(PN_SRC) $(FOUNDRY_SRC) $(TT_Q38_FIT_LOCAL_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_kv_rotate.c $(PN_SRC) $(FOUNDRY_SRC) $(TT_Q38_FIT_LOCAL_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The verify's GDN snapshot-slot map (legacy and compact) against a model of the foundry GDN
# kernels' read/write order, and the Qwen3.8 MTP contexts it buys. Pure: no card, no USE_TT.
tests/test_tt_gdn_verify_slots: tests/test_tt_gdn_verify_slots.c src/accel/tt/pn_tt_gdn_verify_slots.c src/accel/tt/pn_tt_gdn_verify_slots.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_gdn_verify_slots.c src/accel/tt/pn_tt_gdn_verify_slots.c

# L2: the M2 pull only at a chain's end, late reserve-respecting mirrors, and the decode
# out-projection inputs' range records (with the RED arm as a child). USE_TT=1, no card.
tests/test_tt_q38_l2: tests/test_tt_q38_l2.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_l2.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# S1: group completion by pull for the split vector families — the table, the a3/a4/a5 words
# and ONE await per split, against a fake card (with the RED arm as a child). USE_TT=1, no card.
tests/test_tt_group_pull: tests/test_tt_group_pull.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_group_pull.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# S2 on-card completion through the real dispatch recorder + replay builder against a fake
# card: knob 0 byte-identical to the base code's program (tests/tt_oncard_golden.sh), knob 1
# sidecar entries, the precondition_early red aimed short. USE_TT=1, no card.
tests/test_tt_oncard_dispatch: tests/test_tt_oncard_dispatch.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_oncard_dispatch.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# S3b: fused SwiGLU in the gate/up members — knob off moves nothing, the members' replica table,
# the fused descriptor words and every refusal (knob-on half as a child). USE_TT=1, no card.
tests/test_tt_swiglu_fuse: tests/test_tt_swiglu_fuse.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_swiglu_fuse.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Qwen3.8-27B on one p150: the contexts the load accepts once every post-load prefill buffer
# is counted, where the prefix checkpoint fits, and the abort invalidation. USE_TT=1, no card.
tests/test_tt_q38_reserve: tests/test_tt_q38_reserve.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_reserve.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The GDN PHASE CONV split (4 / 8 / 16 members, narrow and wide column encodings) against the
# whole descriptor on a host model of the kernels' column walk, all 48 Qwen3.8 layers, with
# the slice reds. Pure: the production slicer, foundry's encoder and the kernels' decode.
tests/test_tt_gdn_conv_split: tests/test_tt_gdn_conv_split.c src/accel/tt/pn_tt_gdn_conv_split.c \
	src/accel/tt/pn_tt_gdn_conv_split.h $(FOUNDRY)/tt/ft_tt_op_gdn.c $(FOUNDRY)/tt/ft_tt_op_gdn.h \
	$(FOUNDRY)/tt/kernels/resident_gdn_protocol.h
	$(CC) $(CFLAGS) -I$(FOUNDRY)/tt -o $@ tests/test_tt_gdn_conv_split.c \
	  src/accel/tt/pn_tt_gdn_conv_split.c $(FOUNDRY)/tt/ft_tt_op_gdn.c -lm

# The decode matmul's A-reuse ring per model and the Qwen3.8 classes it covers. Pure.
tests/test_tt_matmul_rows: tests/test_tt_matmul_rows.c src/accel/tt/pn_tt_matmul_rows.c \
	src/accel/tt/pn_tt_matmul_rows.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_matmul_rows.c src/accel/tt/pn_tt_matmul_rows.c

# Option D: one conv weight for both GDN families — the production placement on a host
# allocator (decode weights aliased, 30 MiB a bank freed) and the row-0 identity. USE_TT=1, no card.
tests/test_tt_q38_conv_share: tests/test_tt_q38_conv_share.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_conv_share.c $(PN_SRC) $(FOUNDRY_SRC) \
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

# The sampler's survivor hint (pn_sampler_row_hint) must draw the token the row scan draws.
# `--selftest`: a hint missing the best survivor must go red.
tests/test_sampler_row_hint: tests/test_sampler_row_hint.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_row_hint.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The grammar mask on a backend's top-N row (shortlist / dense chunks / whole-row
# refill) must draw the token the full mask draws on the dense row.
# `--selftest`: a subject that skips the mask must be caught.
tests/test_grammar_topn_sampler: tests/test_grammar_topn_sampler.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_topn_sampler.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Repetition penalty stands down while the tool grammar's mask is live.
TEST_GRP_SRC = src/serve/tool_grammar.c src/serve/tool_grammar_json.c src/serve/json_value_grammar.c src/serve/tool_param_empty_value_rule.c src/serve/field_pattern.c src/serve/json_reader.c src/serve/json_writer.c
tests/test_grammar_repeat_penalty: tests/test_grammar_repeat_penalty.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_repeat_penalty.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# field_patterns steer: a refused close on a patterned value becomes the pattern's
# forced continuation (tg_steer / tgj_steer via pn_tool_grammar_hook.steer).
tests/test_field_pattern_steer: tests/test_field_pattern_steer.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_field_pattern_steer.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# tool_choice "required": only the call opener after the thinking close (forced or
# the model's own); no landing window, bounded whitespace before the opener.
tests/test_tool_grammar_forced_close_call: tests/test_tool_grammar_forced_close_call.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tool_grammar_forced_close_call.c $(TEST_GRP_SRC) $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the top-N exactness check the sampler uses before sampling a top-N row
# (`--selftest`: a check blind to the penalties must be caught).
tests/test_topn_row_exact: tests/test_topn_row_exact.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_topn_row_exact.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the top-N verdict under client logit surgery (logit bias, presence/frequency):
# greedy and sampled draws it calls exact are the dense row's (`--selftest`: a verdict blind to
# the bias must be caught).
tests/test_topn_surgery: tests/test_topn_surgery.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_topn_surgery.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the grammar shortlist's outside bound: a masked top-64 row it decides draws the plain
# path's token (dense row, full mask) for every seed, at verify row 0 and past it (`--red`: a
# bound blind to the outside must mismatch; `--selftest`: the comparison can fail).
tests/test_grammar_exact_bound: tests/test_grammar_exact_bound.c tests/grammar_exact/grammar_row_compare.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_exact_bound.c tests/grammar_exact/grammar_row_compare.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the grammar shortlist on a dense row (the batched CUDA wiring): the draw with dense_row set
# (no-op refill) is the full-scan draw for every seed (--red: a bound blind to the outside, --selftest: a subject without the grammar, must mismatch).
tests/test_grammar_dense_shortlist: tests/test_grammar_dense_shortlist.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_grammar_dense_shortlist.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Census of real grammar-masked rows dumped on the Mac (PN_GRAMMAR_ROW_DUMP): what the outside
# bound decides, and its draws against the plain path's (tools/grammar_exact/run_mac_rows.sh).
tools/grammar_exact/grammar_row_census: tools/grammar_exact/grammar_row_census.c tests/grammar_exact/grammar_row_compare.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tools/grammar_exact/grammar_row_census.c tests/grammar_exact/grammar_row_compare.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Gate for the top-N tail planner: a sampled draw with top-k off or past the N, taken from a
# top-64 row (plain and as a deferring verify row), is distributed as the dense draw
# (`--selftest`: the reds tail_nobound / tail_nocarry must fail).
tests/test_topn_tail_sampler: tests/test_topn_tail_sampler.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_topn_tail_sampler.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# True speculative sampling: the rule's emitted distribution is p (analytic TVD + chi-square
# over many draws, against p and against the plain sampler's draws) on random rows at the
# shipped settings; `--selftest`: the reds accept-always / residual = p must fail.
tests/test_spec_sampling_rule: tests/test_spec_sampling_rule.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_sampling_rule.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# forget_state under QWEN35_TP drops both rank halves' host bindings (CUDA build; no GPU
# touched). Red: CFLAGS+=-DQW_FORGET_RED must fail.
tests/test_qwen35_tp_forget: tests/test_qwen35_tp_forget.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_tp_forget.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# A TP host restore then a resume at pos0 > 0 syncs both rank halves' KV rings (CUDA build,
# one die; skips without one). Red: CFLAGS+=-DQW_TP_SYNC_RED must fail.
tests/test_qwen35_tp_restore_sync: tests/test_qwen35_tp_restore_sync.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_tp_restore_sync.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Coupled MTP drafts (spec_mtp_couple.h): emitted stream byte-identical to plain draws
# (seeded and unseeded), q = p accepts every draft, coupled > argmax acceptance;
# `--selftest`: the red couple_consume must fail.
tests/test_spec_mtp_couple: tests/test_spec_mtp_couple.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_mtp_couple.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# MTP grammar draw (spec_mtp.h): verify rows drawn over a scratch walk = plain grammar
# decoding's stream (tokens, RNG, walk), greedy and seeded; `--selftest`: both reds must fail.
tests/test_spec_grammar_draw: tests/test_spec_grammar_draw.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_spec_grammar_draw.c $(PN_SRC) $(FOUNDRY_SRC) \
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

# One-slot fused launches on the decode-slots path: all built, and bit-identical
# to the unfused slots sequence (tests/run_slots_fused_k80.sh runs both arms).
# Needs the model + a CUDA GPU; build with USE_CUDA=1 on a CUDA host.
tests/test_qwen35_slots_fused: tests/test_qwen35_slots_fused.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_slots_fused.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Turn-start checkpoint exactness gate (rung 1c): restore == never decoded,
# bytewise logits. CUDA host + model; falsifier TCKPT_SKIP_RESTORE=1 must fail.
tests/test_turn_checkpoint_qwen35: tests/test_turn_checkpoint_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_turn_checkpoint_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Aborted-prefill exactness gate: an aborted pipelined prefill, kept and
# resumed == one uninterrupted prefill, bytewise logits. CUDA host + model;
# falsifier PABORT_POS_LAG=1 must fail.
tests/test_prefill_abort_resume_qwen35: tests/test_prefill_abort_resume_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefill_abort_resume_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Prefill/decode overlap engine gate: K decoders + a cursor prefill (overlap
# cycles, quiesce, finish) == the sequential run, per-step logits and the
# prefill's. CUDA host + model, 4 dies TP2; reds PN_PDO_RED_SHARED_SCRATCH and
# PN_PDO_RED_NO_HOP_FENCE must fail.
tests/test_pd_overlap_engine_qwen35: tests/test_pd_overlap_engine_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_pd_overlap_engine_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Phase 9.1 speculative-decoding GATE bench: cost of a D-row batched verify vs
# D sequential single-token forwards, plus the batched-slots row-cost curve.
# Pure measurement, changes no engine behaviour. Build with USE_CUDA=1 on a
# CUDA host; run under QWEN35_FUSE=0x37 to match the serve's kernel selection.
tests/bench_spec_verify_gate: tests/bench_spec_verify_gate.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_spec_verify_gate.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Prefill (TTFT) bench: seq chunk loop vs the wide die pipeline, wall time +
# last-logits hash. Build with USE_CUDA=1 on a CUDA host.
tests/bench_prefill_qwen35: tests/bench_prefill_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_prefill_qwen35.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_batch_verify_exact: tests/test_batch_verify_exact.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_batch_verify_exact.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# One-cycle CUDA MTP step (spec_mtp_step.h) vs plain decode, token-exact; red: PN_MTP_RED=1.
tests/test_cuda_spec_step: tests/test_cuda_spec_step.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_cuda_spec_step.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Dense-FFN qwen35 (Ornith/Bonsai): batched prefill + decode-slots vs single-token, greedy.
tests/test_cuda_dense_slots: tests/test_cuda_dense_slots.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_cuda_dense_slots.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Phase 2F end-to-end: MTP speculative decode (adaptive depth) vs plain decode, token-exact.
tests/bench_mtp_e2e: tests/bench_mtp_e2e.c src/serve/serve_spec_lookup.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_mtp_e2e.c src/serve/serve_spec_lookup.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_spec_depth: tests/test_spec_depth.c src/core/spec_depth.c src/core/spec_depth.h
	$(CC) $(CFLAGS) -o $@ tests/test_spec_depth.c src/core/spec_depth.c -lm

# The environment one model load sets and puts back (model_admin.c). Pure: no model.
tests/test_load_env_scope: tests/test_load_env_scope.c src/serve/load_env_scope.c src/serve/load_env_scope.h
	$(CC) $(CFLAGS) -o $@ tests/test_load_env_scope.c src/serve/load_env_scope.c

tests/test_dump_reqlog: tests/test_dump_reqlog.c src/core/dump_reqlog.c src/core/dump_reqlog.h \
		src/serve/serve_dump_reqlog.c src/serve/serve_dump_reqlog.h src/local/sha256.c
	$(CC) $(CFLAGS) -o $@ tests/test_dump_reqlog.c src/core/dump_reqlog.c \
		src/serve/serve_dump_reqlog.c src/local/sha256.c -lpthread

# PN_ACT_STATS (mixw imatrix): every qweight entry point vs a naive per-channel reference,
# outputs byte-identical with the stats on; then a tiny synthetic qwen35 model end to end
# (logits identical knob on/off, coverage, post-norm semantics, split halves merge = whole;
# the test recipe then merges the halves with tools/oracle/act_stats_merge: = the unsplit file).
tests/test_act_stats: tests/test_act_stats.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_act_stats.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The batched X^T X accumulator (dsyrk/ssyrk vs the scalar path, f32) and the sequential .xtx
# write streamed through a FIFO (src/core/act_stats_xtx.h).
tests/test_act_stats_xtx: tests/test_act_stats_xtx.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_act_stats_xtx.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_act_stats_model: tests/test_act_stats_model.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_act_stats_model.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Sums PN_ACT_STATS files (split halves, separate runs): src/core/act_stats_file.h.
tools/oracle/act_stats_merge: tools/oracle/act_stats_merge.c src/core/act_stats_file.c src/core/act_stats_file.h
	$(CC) $(CFLAGS) -o $@ tools/oracle/act_stats_merge.c src/core/act_stats_file.c

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

# Session-switch gate: two conversations taking STRICT turns on one batched
# scheduler (A1 B1 A2 B2 A3 B3) must each resume warm (cached >= the previous
# turn's prompt) and stay token-identical to a cold reference. Build with
# USE_CUDA=1 on a CUDA host; run under QWEN35_BATCH_SLOTS=1 QWEN35_KV_Q4=1.
tests/test_serve_sched_session_switch: tests/test_serve_sched_session_switch.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_serve_sched_session_switch.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
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

# Per-layer residual probe of one forward (tests/qwen3_parity). Needs a model.
tests/qwen3_parity/probe_layers: tests/qwen3_parity/probe_layers.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/qwen3_parity/probe_layers.c $(PN_SRC) $(FOUNDRY_SRC) \
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

# Pantheon's prompt rendering vs the Qwen3.6 / Qwen3.8 chat templates (goldens
# from tests/host/render_qwen_template_goldens.py; no Python at test time). Reaches
# the static request parse by #including openai_chat.c, like test_chat_request.
TEMPLATE_PARITY_SERVE_SRC = src/serve/serve_prompt_tokens.c src/serve/sampling_defaults.c \
	src/serve/tool_calling.c src/serve/json_value_grammar.c src/serve/json_reader.c \
	src/serve/json_writer.c src/serve/request_profile.c src/serve/serve_shutdown.c \
	src/serve/serve_request_dump.c src/serve/tool_call_stream.c \
	src/serve/json_canonical_stream.c src/serve/serve_emitter.c \
	src/serve/field_pattern.c src/serve/field_patterns_request.c
tests/test_template_parity_qwen: tests/test_template_parity_qwen.c \
		tests/template_parity_qwen_pins.h tests/openai_chat_link_stubs.h \
		src/serve/openai_chat.c $(TEMPLATE_PARITY_SERVE_SRC) $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_template_parity_qwen.c $(TEMPLATE_PARITY_SERVE_SRC) \
		$(PN_SRC) $(FOUNDRY_SRC) -lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

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
	$(CC) $(CFLAGS) -o $@ tests/test_dequant_props.c src/formats/gguf_dequant.c src/formats/gguf.c \
		$(TILED_SRC) $(HOST_ADMIT_SRC) -lm

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

# B5: host anon RAM stays flat across many concurrent requests (glibc only;
# SKIPs elsewhere). Runs the unmanaged heap (must grow) and pn_host_heap
# (must stay flat) in child processes.
tests/test_host_heap_churn: tests/test_host_heap_churn.c src/core/mem/pn_host_heap.c src/models/qwen35moe_state_snapshot.c $(HOST_ADMIT_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_host_heap_churn.c src/core/mem/pn_host_heap.c \
		src/models/qwen35moe_state_snapshot.c $(HOST_ADMIT_SRC) -lm $(LDFLAGS)

# Offline sampler bench on dense rows dumped by a serve (PN_HOST_ROW_DUMP).
tests/bench_sampler_rows: tests/bench_sampler_rows.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/bench_sampler_rows.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The margin draw returns the dense draw's token for the same uniform (sampler_margin_draw.h).
tests/test_sampler_margin_draw: tests/test_sampler_margin_draw.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_margin_draw.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# B6: a thread that samples frees its _Thread_local sampler buffers at exit
# (the serve samples token #1 on a per-request thread).
tests/test_sampler_thread_scratch: tests/test_sampler_thread_scratch.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampler_thread_scratch.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

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
tests/test_model_registry: tests/test_model_registry.c src/serve/model_registry.c src/serve/model_session.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_model_registry.c src/serve/model_registry.c src/serve/model_session.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Unload-reclaim regression (the /admin/models/unload die leak). Proves the die
# budget comes BACK — a DIFFERENT model loads onto the SAME die in the SAME
# process, on a die sized to fit exactly one — and that an unload which cannot
# drain REFUSES instead of reporting success and stranding the dies. Same link
# set as test_model_registry; logical dies only, so it runs on the Mac.
tests/test_model_unload_reclaim: tests/test_model_unload_reclaim.c src/serve/model_registry.c src/serve/model_session.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_model_unload_reclaim.c src/serve/model_registry.c src/serve/model_session.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Unload vs open /session/watch (Argus Unload refused while a chat tab was
# open). Real registry + real watch manager + poll thread over a socketpair: an
# idle or in-grace watch never blocks an unload and gets `model_unloaded`; a
# live turn is waited for, then cancelled; a turn that will not stop refuses.
tests/test_unload_closes_watch: tests/test_unload_closes_watch.c src/serve/model_registry.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/session_watch.c src/serve/model_session.c src/serve/http_server.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_unload_closes_watch.c src/serve/model_registry.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/session_watch.c src/serve/model_session.c src/serve/http_server.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/serve_backend_guard.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The silent-deploy class: compiled-backend identity, the refuse-to-start guard
# for a `--dies` this binary cannot honour, and the SERVED-vs-configured
# concurrency the two JSON surfaces advertise. Links the whole serve + engine
# (like test_serve_sched_qwen35) so the JSON assertions run the REAL
# oai_handle_models / ma_handle_models over a socketpair rather than a
# reimplementation of them. No GPU, no model file, no network.
tests/test_host_admit_serve: tests/test_host_admit_serve.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_host_admit_serve.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_backend_guard: tests/test_backend_guard.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_backend_guard.c $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Dispatch-layer bs_get_load() accessor test (Phase G0). Exercises the pure
# slot/queue snapshot (sched==NULL single-user + hand-built cap=4 states);
# links batch_sched.c + the engine (batch_sched.c's other fns need it), no GPU.
tests/test_bs_load: tests/test_bs_load.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_bs_load.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# A finished slot queued in bs_slot_finalize makes a long prefill yield at its next
# chunk boundary (its reply waits on finalize). Hand-built scheduler, no GPU.
tests/test_bs_finalize_yield: tests/test_bs_finalize_yield.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_bs_finalize_yield.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The per-request [req] timing line: fields, decode rate, arrival mark, release stats.
tests/test_request_timing_line: tests/test_request_timing_line.c src/serve/request_timing_line.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_request_timing_line.c src/serve/request_timing_line.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# B3/B4: a decode slot's owner always gets control back (watched token wait,
# legacy wait limit, SSE write failure + send timeout). Hand-built scheduler.
tests/test_bs_prefill_share: tests/test_bs_prefill_share.c src/serve/batch_sched_prefill_share.h
	$(CC) $(CFLAGS) -o $@ tests/test_bs_prefill_share.c -lm

tests/test_bs_pd_budget: tests/test_bs_pd_budget.c src/serve/batch_sched_pd_budget.h
	$(CC) $(CFLAGS) -o $@ tests/test_bs_pd_budget.c -lm

tests/test_decode_watchdog: tests/test_decode_watchdog.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/http_server.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_decode_watchdog.c src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/http_server.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_think_flip_backlog: tests/test_think_flip_backlog.c src/serve/serve_emitter.c $(TEST_GRP_SRC) src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/http_server.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_think_flip_backlog.c src/serve/serve_emitter.c $(TEST_GRP_SRC) src/serve/batch_sched.c src/serve/batch_sched_turn_checkpoint.c src/serve/batch_sched_admission.c src/serve/batch_sched_watchdog.c src/serve/batch_sched_slots.c src/serve/batch_sched_spec.c src/serve/serve_spec_mtp_cuda.c src/serve/request_profile.c src/serve/http_server.c \
		$(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# A4: unit tests for pn_gguf_dequantize_row (per-row byte-offset math +
# per-row == slice-of-full-tensor) and pn_qwen35moe_model_embed_row.
# Part A always runs (synthetic data); Part B needs PN_TEST_MODEL (a real GGUF).
# The card-format oracle's weight path (gguf_fttw_overlay + gguf_tiled + qweight_tiled, and the
# PN_MIXW_BFP4 repack in gguf_fttw_repack), with
# fixtures built by foundry's own converter. Green run, then the red run (one tile per column
# decoded with its exponent off by one) which must FAIL.
FTTW_ORACLE_TEST_SRC = tests/test_fttw_oracle.c src/formats/gguf.c src/formats/gguf_dequant.c \
	src/formats/gguf_tiled.c src/formats/gguf_fttw_overlay.c src/formats/gguf_fttw_repack.c \
	src/core/qweight.c src/core/qweight_tiled.c \
	src/accel/simd.c $(HOST_ADMIT_SRC) $(THREADPOOL_SRC) \
	$(FOUNDRY)/tt/convert/ft_bfp.c $(FOUNDRY)/tt/convert/ft_float_convert.c $(FOUNDRY)/tt/convert/ft_gguf_map.c \
	$(FOUNDRY)/tt/convert/ft_gguf_dequant.c $(FOUNDRY)/tt/convert/ft_gguf_write.c \
	$(FOUNDRY)/tt/convert/ft_tt_precision_ledger.c $(FOUNDRY)/tt/convert/ft_tt_weight_convert.c \
	$(FOUNDRY)/tt/ft_tt_tilize.c
# The oracle's tile-column decode (NEON fast path) vs foundry's codec + the scalar transpose, and the
# fused one-row matvec vs pn_dot_f32 over that reference.
tests/test_tiled_fast_decode: tests/test_tiled_fast_decode.c src/formats/gguf_tiled.c src/formats/gguf_tiled.h \
	src/accel/simd.c src/formats/gguf_dequant.c src/formats/gguf.c $(HOST_ADMIT_SRC) $(FOUNDRY)/tt/convert/ft_bfp.c
	$(CC) $(CFLAGS) -o $@ tests/test_tiled_fast_decode.c src/formats/gguf_tiled.c src/accel/simd.c \
		src/formats/gguf_dequant.c src/formats/gguf.c $(HOST_ADMIT_SRC) $(FOUNDRY)/tt/convert/ft_bfp.c -lm

tests/test_fttw_oracle: $(FTTW_ORACLE_TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $(FTTW_ORACLE_TEST_SRC) -lm $(LDFLAGS) $(LDFLAGS_BLAS)

# The oracle's dumper, whole-program like the tests.
tools/oracle/qwen36_oracle_dump: tools/oracle/qwen36_oracle_dump.c $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tools/oracle/qwen36_oracle_dump.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tools/oracle/qwen36_oracle_compare: tools/oracle/qwen36_oracle_compare.c
	$(CC) $(CFLAGS) -o $@ tools/oracle/qwen36_oracle_compare.c -lm

test-fttw-oracle: tests/test_fttw_oracle
	mkdir -p build && ./tests/test_fttw_oracle
	! PN_TILED_RED=1 ./tests/test_fttw_oracle --red

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
# The grid M split's row partition (pn_tt_grid_msplit.h): the module alone, no card.
tests/test_tt_grid_msplit: tests/test_tt_grid_msplit.c src/accel/tt/pn_tt_grid_msplit.c \
                           src/accel/tt/pn_tt_grid_msplit.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_grid_msplit.c src/accel/tt/pn_tt_grid_msplit.c

# G.2 grid step marks (pn_tt_grid_marks.h): the module alone, no card.
tests/test_tt_grid_marks: tests/test_tt_grid_marks.c src/accel/tt/pn_tt_grid_marks.c \
                          src/accel/tt/pn_tt_grid_marks.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_grid_marks.c src/accel/tt/pn_tt_grid_marks.c

# The prefill overlap's step placement (qwen35moe_prefill_ov_plan.h): the module alone.
tests/test_qwen35_ov_plan: tests/test_qwen35_ov_plan.c src/models/qwen35moe_prefill_ov_plan.c \
                           src/models/qwen35moe_prefill_ov_plan.h
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_ov_plan.c src/models/qwen35moe_prefill_ov_plan.c

# Split-K decode attention's plan (pn_tt_attn_splitk.h): the module alone, no card, no USE_TT.
tests/test_tt_attn_splitk: tests/test_tt_attn_splitk.c src/accel/tt/pn_tt_attn_splitk.c \
                           src/accel/tt/pn_tt_attn_splitk.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_attn_splitk.c src/accel/tt/pn_tt_attn_splitk.c -lm

# The same with every chunk overlapping its predecessor by a tile: must FAIL. Rebuilt every
# time (macOS make 3.81 one-second mtimes).
test-tt-attn-splitk-red:
	rm -f tests/test_tt_attn_splitk_red
	$(CC) $(CFLAGS) -DPN_TT_ATTN_SPLITK_RED -o tests/test_tt_attn_splitk_red \
	  tests/test_tt_attn_splitk.c src/accel/tt/pn_tt_attn_splitk.c -lm
	! ./tests/test_tt_attn_splitk_red

# Dispatch replay across split-K's plan changes (pn_tt_attn_splitk_plan_key): no card, no USE_TT.
tests/test_tt_dispatch_shape_key: tests/test_tt_dispatch_shape_key.c src/accel/tt/pn_tt_attn_splitk.c \
                                  src/accel/tt/pn_tt_attn_splitk.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_dispatch_shape_key.c src/accel/tt/pn_tt_attn_splitk.c

# The same with a key that ignores chunk lengths (misses 9,217): must FAIL. Rebuilt every time
# (macOS make 3.81 one-second mtimes).
test-tt-dispatch-shape-key-red:
	rm -f tests/test_tt_dispatch_shape_key_red
	$(CC) $(CFLAGS) -DPN_TT_ATTN_SPLITK_KEY_RED -o tests/test_tt_dispatch_shape_key_red \
	  tests/test_tt_dispatch_shape_key.c src/accel/tt/pn_tt_attn_splitk.c
	! ./tests/test_tt_dispatch_shape_key_red

# Attention fast-math program selection (pn_tt_attn_fast.h): the module alone, no card, no USE_TT.
tests/test_tt_attn_fast: tests/test_tt_attn_fast.c src/accel/tt/pn_tt_attn_fast.c \
                         src/accel/tt/pn_tt_attn_fast.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_attn_fast.c src/accel/tt/pn_tt_attn_fast.c

# The same with `exp` mapped to the LoFi program: must FAIL. Rebuilt every time (macOS make
# 3.81 one-second mtimes).
test-tt-attn-fast-red:
	rm -f tests/test_tt_attn_fast_red
	$(CC) $(CFLAGS) -DPN_TT_ATTN_FAST_RED -o tests/test_tt_attn_fast_red \
	  tests/test_tt_attn_fast.c src/accel/tt/pn_tt_attn_fast.c
	! ./tests/test_tt_attn_fast_red

# The "96-110,0-15" core-list parser (PN_TT_MATMUL_CORE_LIST): the module alone, no card.
# Build with -DPN_TT_CORE_LIST_RED to see it fail.
tests/test_tt_core_list: tests/test_tt_core_list.c src/accel/tt/pn_tt_core_list.c \
                         src/accel/tt/pn_tt_core_list.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_core_list.c src/accel/tt/pn_tt_core_list.c

# L1a's prefill lend ledger (pn_tt_lend_ledger.h): the module alone, no card.
tests/test_tt_lend_ledger: tests/test_tt_lend_ledger.c src/accel/tt/pn_tt_lend_ledger.c \
                           src/accel/tt/pn_tt_lend_ledger.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_lend_ledger.c src/accel/tt/pn_tt_lend_ledger.c

# The same with stuck cores handed back for a boot, then with deep attention off by default
# (L1a never on unasked): each must FAIL.
test-tt-lend-ledger-red:
	rm -f tests/test_tt_lend_ledger_red
	$(CC) $(CFLAGS) -DPN_TT_LEND_LEDGER_RED -o tests/test_tt_lend_ledger_red \
	  tests/test_tt_lend_ledger.c src/accel/tt/pn_tt_lend_ledger.c
	! ./tests/test_tt_lend_ledger_red
	rm -f tests/test_tt_lend_ledger_red
	$(CC) $(CFLAGS) -DPN_TT_LEND_DEFAULT_RED -o tests/test_tt_lend_ledger_red \
	  tests/test_tt_lend_ledger.c src/accel/tt/pn_tt_lend_ledger.c
	! ./tests/test_tt_lend_ledger_red

# Deep-queued prefill attention's schedule (pn_tt_attn_deep.h): the module alone, no card.
tests/test_tt_attn_deep: tests/test_tt_attn_deep.c src/accel/tt/pn_tt_attn_deep.c \
                         src/accel/tt/pn_tt_attn_deep.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_attn_deep.c src/accel/tt/pn_tt_attn_deep.c

# The same with the last unit never queued, and with no settle before a full ring: both must FAIL.
test-tt-attn-deep-red:
	rm -f tests/test_tt_attn_deep_red
	$(CC) $(CFLAGS) -DPN_TT_ATTN_DEEP_RED -o tests/test_tt_attn_deep_red \
	  tests/test_tt_attn_deep.c src/accel/tt/pn_tt_attn_deep.c
	! ./tests/test_tt_attn_deep_red
	rm -f tests/test_tt_attn_deep_red_overrun
	$(CC) $(CFLAGS) -DPN_TT_ATTN_DEEP_RED_NO_SETTLE -o tests/test_tt_attn_deep_red_overrun \
	  tests/test_tt_attn_deep.c src/accel/tt/pn_tt_attn_deep.c
	! ./tests/test_tt_attn_deep_red_overrun

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

# P3: the norm schedules and the residency extents they need, off hardware.
tests/test_tt_norm_schedule: tests/test_tt_norm_schedule.c \
	src/accel/tt/pn_tt_norm_schedule.c src/accel/tt/pn_tt_norm_schedule.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_norm_schedule.c src/accel/tt/pn_tt_norm_schedule.c

# P0: the persistent norm role's step map and forward cursor, off hardware.
tests/test_tt_norm_role: tests/test_tt_norm_role.c \
	src/accel/tt/pn_tt_norm_role_plan.c src/accel/tt/pn_tt_norm_role_plan.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_norm_role.c src/accel/tt/pn_tt_norm_role_plan.c

# P0: the role table's claims and mode-switch order, off hardware.
tests/test_tt_role_table: tests/test_tt_role_table.c \
	src/accel/tt/pn_tt_role_table.c src/accel/tt/pn_tt_role_table.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_role_table.c src/accel/tt/pn_tt_role_table.c

tests/test_tt_descriptor_census: tests/test_tt_descriptor_census.c \
	src/accel/tt/pn_tt_descriptor_census.c src/accel/tt/pn_tt_descriptor_census.h \
	src/accel/tt/pn_tt_name_list.c src/accel/tt/pn_tt_name_list.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_descriptor_census.c \
	  src/accel/tt/pn_tt_descriptor_census.c src/accel/tt/pn_tt_name_list.c -lm

# A per-tensor Bfp8/Bfp4 format map priced per DRAM bank for Qwen3.8 (mixw plan 1.3): the
# modules alone, no card, no USE_TT. The catalog parser checks the 262,144 tier profile.
TT_TIER_CATALOG_SRC = src/local/catalog_manifest.c src/local/load_options.c \
	src/local/unit_place.c src/local/catalog_match.c src/serve/json_reader.c \
	src/serve/sampling_defaults.c src/serve/json_writer.c
tests/test_tt_q38_mixw_fit: tests/test_tt_q38_mixw_fit.c src/accel/tt/pn_tt_mixw_fit.c \
	src/accel/tt/pn_tt_mixw_fit.h src/models/qwen35moe_tt_mixw_census.c \
	src/models/qwen35moe_tt_mixw_census.h src/models/qwen35moe_tt_fit_plan.c \
	src/models/qwen35moe_tt_fit_plan.h $(TT_TIER_CATALOG_SRC) \
	tests/local/fixtures/catalog_q38_tier.json catalog/catalog.json
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_mixw_fit.c src/accel/tt/pn_tt_mixw_fit.c \
	  src/models/qwen35moe_tt_mixw_census.c src/models/qwen35moe_tt_fit_plan.c \
	  $(TT_TIER_CATALOG_SRC) -lm

# The fit policy's bank rule (128 MiB a bank, verify slots and the checkpoint in): the trigger
# alone, no card, no USE_TT.
tests/test_tt_q38_fit_banks: tests/test_tt_q38_fit_banks.c src/models/qwen35moe_tt_fit_plan.c \
	src/models/qwen35moe_tt_fit_plan.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_fit_banks.c src/models/qwen35moe_tt_fit_plan.c

# A column view of a resident weight reads its source's bytes in place (the MTP draft head).
tests/test_tt_weight_view: tests/test_tt_weight_view.c src/accel/tt/pn_tt_weight_view.c \
	src/accel/tt/pn_tt_weight_view.h src/accel/tt/pn_tt_column_runs.c
	$(CC) $(CFLAGS) -DUSE_TT -o $@ tests/test_tt_weight_view.c src/accel/tt/pn_tt_weight_view.c \
	src/accel/tt/pn_tt_column_runs.c -lm

# The fit policy's levers (GDNC copies, tall bands, unmirrored gated strips) for the bank rule.
tests/test_tt_q38_fit_levers: tests/test_tt_q38_fit_levers.c src/models/qwen35moe_tt_fit_plan.c \
	src/models/qwen35moe_tt_fit_plan.h $(TT_TIER_CATALOG_SRC) catalog/catalog.json
	$(CC) $(CFLAGS) -o $@ tests/test_tt_q38_fit_levers.c src/models/qwen35moe_tt_fit_plan.c \
	  $(TT_TIER_CATALOG_SRC) -lm

# The exact card gates' tables vs the host gates they replace, and the refusal that keeps a
# load on the host gates: no card, no USE_TT.
tests/test_tt_gate_tables: tests/test_tt_gate_tables.c $(FOUNDRY)/tt/kernels/ftgc_table.h $(PN_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_tt_gate_tables.c $(PN_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# The alpha/beta readback's tile runs (rider (a)): no card, no USE_TT.
tests/test_tt_reader_runs: tests/test_tt_reader_runs.c src/accel/tt/pn_tt_reader_runs.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_reader_runs.c

tests/test_tt_dispatch_hoist: tests/test_tt_dispatch_hoist.c \
	src/accel/tt/pn_tt_dispatch_hoist.c src/accel/tt/pn_tt_dispatch_hoist.h \
	src/accel/tt/pn_tt_precondition.c src/accel/tt/pn_tt_precondition_verify.c \
	src/accel/tt/pn_tt_precondition.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_dispatch_hoist.c src/accel/tt/pn_tt_dispatch_hoist.c \
	  src/accel/tt/pn_tt_precondition.c src/accel/tt/pn_tt_precondition_verify.c

# S2 on-card completion: the AWAIT -> wait-sidecar rewrite of the puller's programs and the
# plan check that guards it, on hand-built cases and a synthesized Qwen3.8 decode token
# (replay + stream), with mutated plans and the `precondition_early` red refused. Pure.
tests/test_tt_precondition: tests/test_tt_precondition.c src/accel/tt/pn_tt_dispatch_hoist.c \
	src/accel/tt/pn_tt_dispatch_hoist.h src/accel/tt/pn_tt_precondition.c \
	src/accel/tt/pn_tt_precondition_verify.c src/accel/tt/pn_tt_precondition_dump.c \
	src/accel/tt/pn_tt_precondition.h
	$(CC) $(CFLAGS) -o $@ tests/test_tt_precondition.c src/accel/tt/pn_tt_dispatch_hoist.c \
	  src/accel/tt/pn_tt_precondition.c src/accel/tt/pn_tt_precondition_verify.c \
	  src/accel/tt/pn_tt_precondition_dump.c

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
	find tests -maxdepth 1 -type f \( -name 'test_*' -o -name 'bench_*' -o -name 'gen_golden_*' \) ! -name '*.c' ! -name '*.h' ! -name '*.py' -delete
	rm -rf *.dSYM tests/*.dSYM
	rm -f *.o src/core/*.o src/formats/*.o src/models/*.o src/memory/*.o src/accel/*.o
	rm -rf $(LIB_OBJDIR)
	rm -rf $(INGOT_OBJDIR)

# ============================================================
# Pantheon Local tests (tests/test_local_*.c, tests/test_sampling_defaults.c).
# LOCAL_TESTS_FULL link the whole serve (they fork a `pantheon local` child);
# the rest link only the module under test.
# ============================================================
# LOCAL_TESTS / LOCAL_TESTS_FULL are defined above the test: rule.

$(LOCAL_TESTS_FULL): tests/%: tests/%.c tests/local_test_support.h $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ $< $(PN_SRC) $(SERVE_SRC) $(FOUNDRY_SRC) \
		-lm $(LDFLAGS) $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

tests/test_sampling_defaults: tests/test_sampling_defaults.c src/serve/sampling_defaults.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_sampling_defaults.c src/serve/sampling_defaults.c \
		src/serve/json_reader.c src/serve/json_writer.c -lm $(LDFLAGS)

tests/test_local_unit_place: tests/test_local_unit_place.c src/local/unit_place.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_unit_place.c src/local/unit_place.c -lm $(LDFLAGS)

tests/test_local_pick_port: tests/test_local_pick_port.c tests/local_test_support.h
	$(CC) $(CFLAGS) -o $@ tests/test_local_pick_port.c $(LDFLAGS)

tests/test_local_mdns_packet: tests/test_local_mdns_packet.c src/local/mdns_packet.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_mdns_packet.c src/local/mdns_packet.c -lm $(LDFLAGS)

tests/test_local_minisign: tests/test_local_minisign.c src/local/minisign_verify.c src/local/sha256.c src/serve/base64.c $(MONOCYPHER_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_minisign.c src/local/minisign_verify.c \
		src/local/sha256.c src/serve/base64.c $(MONOCYPHER_SRC) -lm $(LDFLAGS)

LOCAL_CATALOG_TEST_SRC = src/local/catalog.c src/local/catalog_manifest.c src/local/catalog_match.c src/local/load_options.c \
	src/local/unit_place.c src/local/minisign_verify.c src/local/curl_runner.c \
	src/local/local_file_io.c src/serve/sampling_defaults.c src/serve/base64.c \
	src/serve/json_reader.c src/serve/json_writer.c $(MONOCYPHER_SRC)
tests/test_local_catalog: tests/test_local_catalog.c $(LOCAL_CATALOG_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_catalog.c $(LOCAL_CATALOG_TEST_SRC) -lm $(LDFLAGS)

LOCAL_DOWNLOAD_TEST_SRC = src/local/downloader.c src/local/install_worker.c src/local/model_store.c \
	src/local/curl_runner.c src/local/sha256.c src/local/catalog_manifest.c src/local/unit_place.c \
	src/local/ingot_fetch.c src/serve/sampling_defaults.c src/serve/json_reader.c \
	src/serve/json_writer.c $(INGOT_LIB)
tests/test_local_download: tests/test_local_download.c tests/local_fixture_httpd.h $(LOCAL_DOWNLOAD_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_download.c $(LOCAL_DOWNLOAD_TEST_SRC) -lm $(LDFLAGS)

# Ingot Phase 1b: .ingot catalog downloads decoded while they arrive (bounded memory,
# resume after a cut, corrupt / delta base refusals).
tests/test_local_ingot: tests/test_local_ingot.c tests/local_fixture_httpd.h $(LOCAL_DOWNLOAD_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_ingot.c $(LOCAL_DOWNLOAD_TEST_SRC) -lm $(LDFLAGS)

# H2m: store listing + uninstall by catalog id.
tests/test_local_store_cleanup: tests/test_local_store_cleanup.c src/local/model_store_cleanup.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_store_cleanup.c src/local/model_store_cleanup.c -lm $(LDFLAGS)

# Static file server standing in for Hugging Face in the on-hardware scripts.
build/fixture_httpd: tests/local/fixture_httpd.c tests/local_fixture_httpd.h
	@mkdir -p build
	$(CC) -O2 -Wall -Wextra -std=c11 $(if $(filter Darwin,$(shell uname -s)),,-D_GNU_SOURCE) -o $@ tests/local/fixture_httpd.c

# L3: load options (grid, fit, refusals, the catalog's memory model vs measured peaks).
LOCAL_OPTIONS_TEST_SRC = src/local/load_options.c src/local/unit_place.c src/local/catalog_manifest.c \
	src/local/catalog_match.c src/local/local_file_io.c src/serve/sampling_defaults.c \
	src/serve/json_reader.c src/serve/json_writer.c src/core/mem/pn_ladder.c
tests/test_local_load_options: tests/test_local_load_options.c $(LOCAL_OPTIONS_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_load_options.c $(LOCAL_OPTIONS_TEST_SRC) -lm $(LDFLAGS)

# L4: one served name on several replicas (session routing; hand-built registry).
tests/test_replica_route: tests/test_replica_route.c src/serve/model_replica_route.c src/serve/replica_affinity.c src/serve/replica_pin.c src/serve/http_server.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_replica_route.c src/serve/model_replica_route.c \
		src/serve/replica_affinity.c src/serve/replica_pin.c src/serve/http_server.c \
		src/serve/json_reader.c src/serve/json_writer.c -lm $(LDFLAGS)

tests/test_local_loaded_set: tests/test_local_loaded_set.c src/local/loaded_set.c src/local/local_file_io.c src/serve/json_reader.c src/serve/json_writer.c $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_loaded_set.c src/local/loaded_set.c \
		src/local/local_file_io.c src/serve/json_reader.c src/serve/json_writer.c -lm $(LDFLAGS)

# People registry P1.1: people.json and the admin tree.
LOCAL_PEOPLE_TEST_SRC = src/local/people_store.c src/local/local_file_io.c src/local/local_random.c \
	src/local/sha256.c src/serve/json_reader.c src/serve/json_writer.c
tests/test_local_people_store: tests/test_local_people_store.c $(LOCAL_PEOPLE_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_people_store.c $(LOCAL_PEOPLE_TEST_SRC) -lm $(LDFLAGS)

# People registry P1.2: keys.json v2 and the v1 migration.
LOCAL_KEYS_TEST_SRC = $(LOCAL_PEOPLE_TEST_SRC) src/local/key_store.c src/local/key_migration.c
tests/test_local_key_migration: tests/test_local_key_migration.c $(LOCAL_KEYS_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_key_migration.c $(LOCAL_KEYS_TEST_SRC) -lm $(LDFLAGS)

# People registry P1.3: route classes x callers through the real auth hook.
LOCAL_AUTH_TEST_SRC = $(LOCAL_KEYS_TEST_SRC) src/local/local_auth.c src/local/local_caller.c \
	src/local/rate_limit.c src/serve/http_server.c
tests/test_local_route_classes: tests/test_local_route_classes.c $(LOCAL_AUTH_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_route_classes.c $(LOCAL_AUTH_TEST_SRC) -lm $(LDFLAGS)

# People registry P2.2: sessions.json and the registry rules (no sockets).
LOCAL_REGISTRY_TEST_SRC = src/local/session_registry.c src/local/local_file_io.c \
	src/serve/json_reader.c src/serve/json_writer.c $(MONOCYPHER_SRC)
tests/test_local_session_registry: tests/test_local_session_registry.c $(LOCAL_REGISTRY_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_session_registry.c $(LOCAL_REGISTRY_TEST_SRC) -lm $(LDFLAGS)

# People registry P2.3: the hub's held waits over socketpairs.
LOCAL_HUB_TEST_SRC = $(LOCAL_REGISTRY_TEST_SRC) src/local/hosts_hub.c src/local/hub_relays.c \
	src/local/used_nonces.c src/local/local_random.c src/local/sha256.c src/serve/http_server.c
tests/test_local_hosts_hub: tests/test_local_hosts_hub.c $(LOCAL_HUB_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_hosts_hub.c $(LOCAL_HUB_TEST_SRC) -lm $(LDFLAGS)

# People registry P4.1: relay pairs in the hub over socketpairs.
tests/test_local_hub_relays: tests/test_local_hub_relays.c $(LOCAL_HUB_TEST_SRC) $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_hub_relays.c $(LOCAL_HUB_TEST_SRC) -lm $(LDFLAGS)

# People registry P3.1: HMAC-SHA256 and attach tickets.
LOCAL_TICKET_TEST_SRC = src/local/attach_ticket.c src/local/sha256.c src/local/local_random.c \
	src/serve/json_reader.c src/serve/json_writer.c $(MONOCYPHER_SRC)
tests/test_local_attach_ticket: tests/test_local_attach_ticket.c $(LOCAL_TICKET_TEST_SRC) \
	tests/local_fixture_replay.h $(ALL_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_local_attach_ticket.c $(LOCAL_TICKET_TEST_SRC) -lm $(LDFLAGS)

# People registry P1.4+: the contract fixtures are replayed against a forked box.
tests/test_local_pairing_devices tests/test_local_people_routes tests/test_local_host_routes \
	tests/test_local_session_routes tests/test_local_relay_routes: tests/local_fixture_replay.h tests/local_people_test_support.h \
	tests/local_host_wait_support.h

# The Mac end-to-end flow trusts the TEST ONLY catalog key (this binary only).
tests/test_local_flow: CFLAGS += -DPN_LOCAL_TEST_CATALOG_KEY
tests/test_local_unit_types: CFLAGS += -DPN_LOCAL_TEST_CATALOG_KEY
