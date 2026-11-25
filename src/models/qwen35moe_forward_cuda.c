/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_forward_cuda.c
 * @brief GPU forward path for Qwen3.5-MoE, single-die or N-die layer-split.
 *        Mirrors the CPU orchestration in qwen35moe_forward.c op-for-op, but
 *        uploads every weight to a persistent GPU fc_buffer at load and runs the
 *        40-layer sandwich (attn_norm -> SSM|attn block -> +residual ->
 *        post_attn_norm -> shared-MoE -> +residual) through Foundry's fc_*_ctx
 *        CUDA kernels, with GPU-resident dual state (SSM conv+recurrent state
 *        and KV cache) and per-layer scratch. Embed -> per-layer loop ->
 *        output_norm -> LM head. The LM-head result is reduced ON the die to
 *        the top-k (id, logit) pairs by default (gpu_fetch_logits /
 *        QWEN35_GPU_TOPK) — only k pairs cross the 0.4 GB/s host link, not
 *        the 248k-vocab buffer; QWEN35_GPU_FULL_LOGITS=1 restores the full
 *        logits D2H copy.
 *
 *        N-die split: an explicit placement of N dies (pn_load_on / serve's
 *        --dies) splits the layers N ways via pn_plan_die_layer_split — the
 *        SAME partitioner admission prices the load with, so the projection and
 *        the upload cannot disagree. Die i holds layers [layer_lo[i],
 *        layer_hi[i]) in its own VRAM with its own fc_cuda_ctx; token_embd is
 *        never uploaded (host row copy at embed time) and the final norm + LM
 *        head ride on the LAST die. The forward walks the dies in order,
 *        handing the residual stream across each of the N-1 boundaries — one
 *        contiguous [positions][n_embd] f32 copy, device-to-device over the
 *        peer link where the pair has it, else staged through the host
 *        (QWEN35_HANDOFF_HOST=1 forces the host path). The SSM/attention dual
 *        state never crosses dies.
 *
 *        More dies buys CAPACITY, not speed: every die's layer maths costs
 *        exactly what it did, and each extra boundary adds a sync plus a
 *        transfer per step. Split a model across the FEWEST dies that hold it
 *        at the context you want; go wider only to buy KV headroom.
 *
 *        Legacy barswap (QWEN35_GPU_BARSWAP) time-multiplexes one MMIO slot and
 *        supports at most a 2-die split; wider placements are refused there.
 *        The 2-die-only optimizations (T3 vocab-split head, L5 die pipeline,
 *        QWEN35_TWO_THREAD, the C2 async phase machine) stay off above two
 *        dies — each is a two-stage construction, and deepening them would add
 *        boundary hops to a trade that is already paying them.
 *
 * @status HOST C only. Compiles on the Mac with -DUSE_CUDA via dlopen (no GPU
 *         present here). The operator validates it layer-by-layer against the
 *         CPU reference on a K80 host (the A8 per-layer-dump bisection): the
 *         per-layer dump tags are byte-identical to the CPU path
 *         (qwen35moe_forward.c qwen35moe_dbg) and fire for EVERY layer on both
 *         dies, so all 40 GPU-vs-CPU diffs line up. The 2-die context-swap
 *         mechanism is gated by tests/test_2die_swap (run that FIRST).
 *
 * @issues Several ops have no dedicated GPU kernel yet and are done host-side on
 *         a read-back buffer (then re-uploaded). Each is marked `TODO kernel:`
 *         inline and listed in the file's final report. They are correctness-
 *         neutral (they call the same CPU helpers the reference uses) but cost a
 *         D2H/H2D round trip; closing them is follow-up kernel work.
 *
 *         The boundary hand-off's peer copy MUST be followed by a sync of the
 *         RECEIVING die (gpu_handoff_import). cuMemcpyPeer lands on its own
 *         stream, so without it the next die reads a stale residual stream and
 *         the model emits fluent-shaped garbage instead of failing — the failure
 *         mode is silent, which is why it is written down here rather than
 *         trusted to memory. Reproduced on K80 2026-08-21 at both 2 and 3 dies.
 *
 *         g->bhost[] is ONE host staging set shared by every die. The
 *         sequential paths are safe because only one die is ever live, but the
 *         9c wide prefill pipeline runs all of them at once, so any layer path
 *         that stages through bhost would have two dies writing the same host
 *         bytes — silent corruption again, not a failure. The default batched
 *         path never does (flash attention is on-die, the lazy KV mirror skips
 *         the host store, MoE and SSM never leave the die), and
 *         pn_qwen35moe_prefill_range_pipelined_cuda REFUSES the two settings
 *         that reintroduce it (QWEN35_HOST_KV_MIRROR=1, and flash unavailable).
 *         Per-die staging buffers would lift that restriction.
 *
 *         QWEN35_GPU_MAX_LAYERS (single-die debug cap) now prices the KV for
 *         the layers it actually uploads rather than for the whole model, so
 *         its gate is looser than before. It is a validation knob, unset in
 *         production.
 * @todo
 *   - [ ] Close the TODO-kernel gaps (joint Q+gate split, GQA head broadcast +
 *         L2-norm, decay gate / softplus, sigmoid gates, SwiGLU expert matmuls
 *         for non-Q3_K experts, attention softmax over the KV window).
 *   - [ ] K80: per-layer GPU-vs-CPU validation against the oracle dump, and
 *         confirm tests/test_2die_swap PASSes before trusting the split.
 *   - [ ] K80 (C1): regenerate fc_cuda_ptx.h (cd foundry/cuda_kernels &&
 *         make), then gate the segment graphs — argmax-exact generation vs
 *         QWEN35_GRAPH_FULL=0 and decode tok/s A/B (see gpu_segment_forward).
 */

/* 1. This module's own header */
#include "qwen35moe.h"

/* 2. Standard library */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
/* Unconditional: the live-GPU-context registry below is guarded by a mutex in
 * every build, USE_THREADS or not — a model unload can race a session free. */
#include <pthread.h>

/* 3. System / platform headers */
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#include <pthread.h>
#endif

/* 4. Project headers */
#include "../accel/die_layer_split.h"
#include "../accel/simd.h"
#include "../core/load_guard.h"
#include "../core/load_progress.h"
#include "../core/mem/pn_admission.h"
#include "../core/mem/pn_kv_geometry.h"
#include "../core/mem/pn_memwatch.h"
#include "../core/mem/pn_ladder.h"
#include "../core/qweight.h"
#include "../formats/gguf_layer_bytes.h"
#include "gpu_attn_path_tracker.h"
#include "gpu_graph_capture_tracker.h"
#include "qwen35moe_die_projection.h"
#include "qwen35moe_internal.h"
#include "qwen35moe_kv_q4.h"
#include "qwen35moe_mtp_ring.h"
#include "qwen35moe_pcie_stage.h"
#include "qwen35moe_step_ledger.h"

#ifdef USE_CUDA

#include "cuda/fc_cuda.h"
#include "barswap/barswap.h"

/* The no-placement boundary default (the legacy single-model serve path only —
 * an explicit placement supersedes it entirely). With no QWEN35_GPU_SPLIT_LAYER
 * override the split is disabled for a model of this depth or shallower, so
 * every layer runs on one die (the validated single-die path). A boundary below
 * n_layer implies two dies, named by QWEN35_GPU_DIE_A/B. See gpu_plan_dies(). */
#define GPU_DIE_SPLIT_LAYER 40

/* Default die ids for an unplaced 2-die split (override via QWEN35_GPU_DIE_A /
 * QWEN35_GPU_DIE_B). Two dies of one K80 card. */
#define GPU_DIE_A_DEFAULT 0
#define GPU_DIE_B_DEFAULT 1

/* Cap for the host-side attention-softmax scratch and the top-k expert
 * selection read-back. Matches the QWEN35_MAX_SEQ ceiling
 * (PN_QWEN35MOE_MAX_SEQ_CAP, see qwen35moe_forward.c). At 256K the per-head
 * scores array is 1 MB, which is too large for the stack — the three
 * host-softmax sites therefore malloc a `max_seq`-sized scratch on the heap
 * (freed at end of block) rather than stack-allocating this cap. Top-k <= 32. */
#define PN_QWEN35MOE_GPU_MAX_SEQ_SCORES 262144
#define PN_QWEN35MOE_GPU_MAX_TOPK 32

/* Decode-mode flash attention (QWEN35_FLASH_DECODE): the single decode query
 * splits the cached context across blocks so the die is not near-idle (npos=1
 * prefill launches only n_head_kv=2 blocks). Aim for ~KEYS_PER_SPLIT keys per
 * block, capped at SPLIT_MAX splits (partial scratch scales with the cap). */
#define PN_QWEN35MOE_DEC_KEYS_PER_SPLIT 256
#define PN_QWEN35MOE_DEC_SPLIT_MAX 32

/* Multi-user decode: max concurrent slots (per-user SSM state images are
 * ~2.1 MB per SSM layer per slot; 16 slots is ~1 GB across the model —
 * within a die pair's headroom next to the split weights). */
#define PN_QWEN35MOE_GPU_MAX_SLOTS 16
/* QWEN35_TP in-layer exchanges a die records an event for: core gather, out
 * gather, then down reduce (dense) or hid gather + output gather (MoE). */
#define TP_N_XCHG 4
/* Buffer lanes (QWEN35_BS_PD_OVERLAP): the decode-slots step runs on its own
 * batch scratch, die scratch and TP exchange set (QW_LANE_DECODE), so it can be
 * issued between two layers of a prefill chunk that sits on the default set
 * (QW_LANE_PREFILL). Same kernels on the same inputs: the lane only moves
 * addresses. g->lane selects the live one; 0 everywhere but inside a decode
 * step with the switch on. */
#define QW_LANE_PREFILL 0
#define QW_LANE_DECODE  1
#define QW_N_LANES      2

/* ============================================================
 * GPU-resident weight handle
 *
 * One fc_buffer per weight matrix (raw quant bytes, uploaded once) plus the
 * F32 norm/router vectors. Held on the model via pn_qwen35moe_model_set_gpu so
 * the per-token forward never re-uploads. Mirrors rwkvx.c's persistent
 * q->cuda_data pattern, generalized to the qwen35moe layer zoo.
 * ============================================================ */

/* A single weight matrix resident on the GPU: its raw quant bytes plus the dims
 * and quant type the matvec kernel needs. row_stride is the qweight's
 * bytes-per-row (for Q3_K, (cols/256)*110). */
typedef struct {
    fc_buffer *bytes;      /* raw quant bytes uploaded H2D (NULL if absent) */
    pn_gguf_quant type;    /* quant type (drives kernel dispatch) */
    int rows;
    int cols;
    int row_stride;        /* bytes per row */
    const pn_qweight *cpu; /* back-pointer to the CPU qweight (host fallback) */
    fc_cuda_ctx *ctx;      /* owning die context (which die holds this weight) */
    int kpar;              /* 1 = Q8_0 parity slab of a TP K-split (cols = full K) */
} gpu_weight;

/* GPU copy of one F32 vector (norms, router bias, etc.). */
typedef struct {
    fc_buffer *buf;
    int n;
    fc_cuda_ctx *ctx;      /* owning die context */
} gpu_vec;

/* GPU MoE block weights (router + per-expert SwiGLU + shared expert).
 *
 * Grouped mode (grouped == 1, the fast path): the merged 3D GGUF expert
 * tensors are uploaded as ONE contiguous buffer per matrix (gate_all/up_all/
 * down_all) and the whole routed-expert block runs in two launches
 * (fc_moe_gateup_ctx + fc_moe_down_accum_ctx) that index experts on device
 * from the router's top_idx — no per-expert buffers, no host read-back of the
 * selection. Both launches dispatch on the tensor's gguf quant: requires
 * gate/up Q3_K|Q4_K (gate and up the same) and down Q3_K|Q4_K|Q6_K, covering both
 * the Q3_K_M mini's mixed layout and a pure Q4_K_M, plus contiguous expert
 * slices; anything else falls back to the per-expert path
 * (gate_exps/up_exps/down_exps + the old dispatch loop). */
/* One cached fused-launch descriptor (fc_fused_multi): built lazily on first
 * use from persistent weight/scratch buffers, then reused every token. desc
 * == NULL after tried != 0 means the segment mix is unsupported and the
 * caller keeps its unfused launch sequence. */
typedef struct {
    fc_buffer *desc;
    int nseg;
    int blocks;
    int sh_bytes;
    int tried;
} gpu_fused;

typedef struct {
    gpu_weight ffn_gate_inp;     /* router logits */
    int        grouped;          /* 1 = grouped kernels, 0 = per-expert path */
    gpu_weight gate_all;         /* merged [n_expert*ff, n_embd] (grouped) */
    gpu_weight up_all;           /* merged [n_expert*ff, n_embd] (grouped) */
    gpu_weight down_all;         /* merged [n_expert*n_embd, ff] (grouped) */
    gpu_weight *gate_exps;       /* [n_expert] */
    gpu_weight *up_exps;         /* [n_expert] */
    gpu_weight *down_exps;       /* [n_expert] */
    gpu_weight ffn_gate_shexp;
    gpu_weight ffn_up_shexp;
    gpu_weight ffn_down_shexp;
    gpu_vec    ffn_gate_inp_shexp; /* shared-expert scalar gate [n_embd] */
    const float *gate_inp_shexp_host; /* host mmap pointer to the same vector
                                       * (constant weights — saves a per-token
                                       * D2H in the shared-gate dot) */
    int n_expert;
    /* Fused launches: router + shexp scalar-gate dot + shexp gate/up SwiGLU
     * in one kernel (fx_pre), and shexp down + sigmoid gate + combine +
     * residual in one kernel (fx_post). */
    gpu_fused fx_pre;
    gpu_fused fx_post;
    /* The same two on the batched-decode (slots) path at one slot, over the
     * batch scratch of generation fx_s_gen: see gpu_slots_fused_prepare. */
    gpu_fused fx_pre_s;
    gpu_fused fx_post_s;
    unsigned fx_s_gen;
    int sdiag_logged;   /* QWEN35_SDIAG: this layer's structure reported once */
} gpu_moe;

typedef struct {
    gpu_vec    attn_norm;
    gpu_weight attn_qkv;
    gpu_weight attn_gate;
    gpu_vec    ssm_conv1d;        /* [conv_dim * ssm_conv] */
    gpu_vec    ssm_a;             /* [n_v_heads] */
    gpu_weight ssm_alpha;
    gpu_weight ssm_beta;
    gpu_vec    ssm_dt;            /* [n_v_heads] */
    gpu_vec    ssm_norm;          /* [ssm_state] */
    gpu_weight ssm_out;
    gpu_vec    post_attention_norm;
    gpu_moe    moe;
    fc_cuda_ctx *ctx;   /* die context this layer's weights live on */
    /* QWEN35_TP: this struct is rank tp_rank's half of the layer (tp_on), with
     * the local dims of pn_qwen35moe_gpu.tp_cfg (see the TP section). */
    int tp_on, tp_rank;
    /* Persistent GPU-resident SSM dual state (conv + recurrent), allocated
     * ONCE on first use and then living entirely on this layer's die — no
     * per-token H2D/D2H round trip. The CONTENTS are (re)seeded from the host
     * state by an H2D copy when host_state changes or a sequence restarts at
     * pos 0; the buffer ADDRESSES never change after the first allocation, so
     * the layer's captured CUDA graph (below) stays valid across re-seeds. */
    fc_buffer *conv_state_g;
    fc_buffer *ssm_state_g;
    const void *host_state;
    /* Fused launches: the four from-x_norm projections (+ sigmoid/decay
     * epilogues) in one kernel, and ssm_out + residual in one kernel. */
    gpu_fused fx_proj;
    gpu_fused fx_out;
    /* The same two on the slots path at one slot (gpu_slots_fused_prepare). */
    gpu_fused fx_proj_s;
    gpu_fused fx_out_s;
    unsigned fx_s_gen;
    /* Captured CUDA graph of this layer's ENTIRE position-invariant launch
     * sequence (attn_norm -> SSM block -> post_attention_norm -> MoE, ~13
     * launches), replayed with one cuGraphLaunch per token. Captured lazily
     * at pos >= 1 (after the fused descriptors were built at pos 0).
     * graph_capture owns the retry policy for a FAILED capture (bounded
     * backoff, then a permanent stop) and the residency state the
     * graphs_live/graphs_total gauge reads — see
     * gpu_graph_capture_tracker.h. */
    fc_graph *graph;
    pn_graph_capture_tracker graph_capture;
    /* Set once this layer's body has run DIRECTLY (uncaptured) in this
     * process — capture is only attempted after that, so the fused
     * descriptors/scratch it builds on first use are never allocated
     * mid-capture (cuMemAlloc during capture invalidates it; the trap
     * fires when a BATCHED prefill precedes the first single-token
     * forward, since the batch path never builds the fused descs). */
    int body_ran;
    /* Speculative-verify snapshots (see pn_qwen35moe_spec_verify_tokens_cuda):
     * per-position dual-state images written by the seq scans during a spec
     * verify batch — conv_snaps[t] / ssm_snaps[t] = this layer's state AFTER
     * position t — so a partial accept of M < n positions restores
     * state-after-M with one D2D copy. Lazily allocated on the first spec
     * verify, sized snaps_cap positions. */
    fc_buffer *conv_snaps;
    fc_buffer *ssm_snaps;
    int snaps_cap;
    /* Multi-user decode slot state (see pn_qwen35moe_decode_slots_cuda):
     * slot-major per-user dual-state images — conv_slots [slots_cap][conv_n],
     * ssm_slots [slots_cap][ssm_n] — each slot advancing independently one
     * token per batched decode step. Seeded per slot by the bind call
     * (D2D from the live conv_state_g/ssm_state_g right after that user's
     * prefill); sized by the bind's n_slots. */
    fc_buffer *conv_slots;
    fc_buffer *ssm_slots;
    int slots_cap;
    /* Turn-start checkpoint per slot row (QWEN35_TURN_CHECKPOINT, serve's
     * rung 1c): the row's recurrent image at its checkpoint position,
     * [ckpt_cap][conv_n] / [ckpt_cap][ssm_n]. Carved with the session blocks
     * (pn_qwen35moe_slot_checkpoints_carve); NULL when the feature is off. */
    fc_buffer *conv_ckpt;
    fc_buffer *ssm_ckpt;
    int ckpt_cap;
    /* L9 — captured CUDA graph of this SSM layer's BATCHED-decode body
     * (attn_norm -> slot-SSM block -> post_attention_norm -> batched MoE,
     * ~30 launches -> 1), replayed once per batched decode step. The slots
     * path is position-invariant (per-slot device state at stable addresses,
     * batch scratch persistent per die) so the same graph replays every
     * step — the only per-step variance (the token embed + boundary handoff
     * + logits fetch) stays OUTSIDE, exactly like the single-token per-layer
     * SSM graph above. Keyed by the captured slot count: a change in nslots
     * changes the launch grids, so the graph is dropped and recaptured.
     * The 10 attention layers are NOT graphed (their per-layer host KV-mirror
     * D2H sync breaks capture); this covers the 30 SSM layers = ~75% of the
     * per-step launches. */
    fc_graph *slots_graph;
    pn_graph_capture_tracker slots_graph_capture;
    int slots_graph_n;   /* nslots the graph was captured against (0 = none) */
    int slots_graph_base; /* L5 — slot_base baked into the graph's ring-offset
                           * views; a different group (die-interleave) reuses
                           * this same layer with a different base → the graph
                           * is dropped + recaptured on a base change. */
    const void *slots_graph_scratch;  /* b->hidden addr baked into the graph:
                                       * a batch-scratch realloc (a wider
                                       * prefill) moves it → drop + recapture */
} gpu_ssm_layer;

typedef struct {
    gpu_vec    attn_norm;
    gpu_weight attn_q;
    gpu_weight attn_k;
    gpu_weight attn_v;
    gpu_vec    attn_q_norm;       /* [head_dim] */
    gpu_vec    attn_k_norm;       /* [head_dim] */
    gpu_weight attn_output;
    gpu_vec    post_attention_norm;
    gpu_moe    moe;
    fc_cuda_ctx *ctx;   /* die context this layer's weights live on */
    int tp_on, tp_rank; /* QWEN35_TP rank half (see gpu_ssm_layer) */
    /* Fused launches: q/k/v projections in one kernel, and attn_output +
     * residual in one kernel. */
    gpu_fused fx_qkv;
    gpu_fused fx_out;
    /* attn_output + residual on the slots path at one slot
     * (gpu_slots_fused_prepare). */
    gpu_fused fx_out_s;
    unsigned fx_s_gen;
    /* On-die flash-attention KV ring cache (batched prefill): K/V rows in
     * the host cache's exact [n_head_kv][max_seq][head_dim] layout, on this
     * layer's die. kv_host_id names the state whose cache the GPU copy
     * mirrors (its host k pointer); kv_synced_pos is how many leading rows
     * are valid — gpu_attn_kv_cache_sync re-uploads missing rows from the
     * host mirror after a state switch or host-side (single-token decode)
     * appends. */
    fc_buffer *k_cache_g;
    fc_buffer *v_cache_g;
    /* fp16 sink + recent-window shadow (q4 mode): the flash kernel reads these
     * for the attention sink and the last W tokens instead of q4, so greedy
     * argmax stops flipping. Sized to PN_KVQ4_SINKMAX / PN_KVQ4_WMAX rows/head;
     * filled by the append kernel and re-uploaded from the host shadow on a
     * non-contiguous re-sync. */
    fc_buffer *k_sink_g, *v_sink_g;
    fc_buffer *k_win_g, *v_win_g;
    /* Phase 9.2 — undo buffer for the recent-window ring during a speculative
     * verify. The ring is keyed `pos % W`, so appending a row at position p
     * overwrites the fp16 shadow of position p - W, which is STILL inside the
     * window and still read. A rejected draft therefore has to put that slot
     * back, and the only lossless copy is the one taken before the append (the
     * q4 body is coarser, so restoring from it would not be bit-exact).
     * [n_head_kv][spec_win_cap][head_dim] fp16, mirroring the ring's layout;
     * spec_win_pos0/n record which verify the snapshot belongs to. */
    fc_buffer *k_win_snap_g, *v_win_snap_g;
    int spec_win_cap;
    int spec_win_pos0, spec_win_n;
    int cache_cap;              /* allocated max_seq rows (0 = none) */
    const void *kv_host_id;
    int kv_synced_pos;
    /* Decode split-attention partials (QWEN35_FLASH_DECODE): per (head, split)
     * running max / sum and the unnormalized accumulator, on this die. Sized to
     * PN_QWEN35MOE_DEC_SPLIT_MAX splits and merged by the combine kernel. */
    fc_buffer *dec_m_g, *dec_l_g, *dec_acc_g;
    int dec_splits_cap;
    /* Segment-graph (C1) capture gates, mirroring gpu_ssm_layer.body_ran:
     * body_ran = this layer's full sandwich ran DIRECTLY once (fused descs +
     * per-die scratch built, so nothing allocates mid-capture); flash_ran =
     * the on-die decode-flash path (not the host-softmax fallback) ran, so
     * the GPU KV ring + dec_* partial scratch exist. */
    int body_ran;
    int flash_ran;
    /* L7 — multi-user decode slot KV rings (QWEN35_FLASH_SLOTS): one ring
     * per slot, [nslots][n_head_kv][max_seq][row], each slot slice
     * byte-identical to the single-user ring above, on this layer's die,
     * plus the per-slot fp16 sink/window shadows (q4 mode) and the per-slot
     * decode-split partial scratch. kv_slot_host_id / kv_slot_synced are the
     * per-slot watermarks (same convention as kv_host_id / kv_synced_pos);
     * gpu_attn_block_slots_flash re-uploads missing rows from the host
     * mirror, which the slots path keeps CURRENT every step (unlike the
     * single-token B2 deferral), so bind/move/fallback stay trivially
     * correct. */
    fc_buffer *k_slots_g, *v_slots_g;
    fc_buffer *k_sink_slots_g, *v_sink_slots_g;
    fc_buffer *k_win_slots_g, *v_win_slots_g;
    /* Turn-start checkpoint per slot row (q4 mode): the fp16 sink/window
     * shadows at the checkpoint position. The window ring is overwritten by
     * every later token, so it is the one piece of attention state a rewind
     * needs; KV rows below the checkpoint are never rewritten. */
    fc_buffer *k_sink_ckpt, *v_sink_ckpt;
    fc_buffer *k_win_ckpt, *v_win_ckpt;
    int ckpt_cap;
    fc_buffer *dec_slots_m_g, *dec_slots_l_g, *dec_slots_acc_g;
    int kv_slots_cap_seq;       /* allocated max_seq rows per slot (0 = none) */
    int kv_slots_cap_n;         /* allocated slot count */
    int kv_slots_q4;            /* cache mode the rings were allocated for */
    int dec_slots_cap;          /* slots the dec partial scratch covers */
    /* Ring-allocation failure geometry (Phase 7). A layer whose per-slot KV
     * rings do not fit records the geometry that failed. It used to STOP
     * retrying that geometry forever, because without any brake the layer
     * re-freed and re-attempted a multi-hundred-MiB cuMemAlloc on EVERY decode
     * step (Phase 6 measured 3504 failed 144 MiB allocations over ~1750 steps
     * at ctx_cap 131072 x K=4 — a 62% collapse instead of the ~10-15%
     * host-softmax degradation the fallback intends). But a permanent latch
     * means ONE transient failure pins the layer to the host path for the life
     * of the process, which is a measured 10.7x cliff (13.02 -> 1.22 tok/s at
     * depth 4096, K=2) that nothing reports.
     *
     * So the geometry is still recorded — it is what tells a SMALLER request
     * apart from a same-or-larger one — but the retry policy now lives in
     * `path` (gpu_attn_path_tracker.h): a bounded 8/32/128/512-eligible-step
     * backoff for a transient failure, an immediate stop for one that could not
     * fit an empty die, and an instant budget reset when the geometry shrinks.
     * The per-step cost the latch was protecting against is gone independently:
     * slots_ring_gen no longer bumps on an attempt that neither freed nor
     * allocated anything, so a retry cannot invalidate every layer's captured
     * graph. 0 = no failure recorded. */
    int kv_slots_fail_seq;      /* max_seq of the failed ring allocation */
    int kv_slots_fail_n;        /* slot count of the failed ring allocation */
    int kv_slots_fail_q4;       /* cache mode of the failed ring allocation */
    int dec_slots_fail_n;       /* slot count of the failed dec-scratch alloc */
    /* Which attention path this layer is on right now, why, and the ring
     * allocation's retry budget. Read by pn_qwen35moe_gpu_attn_residency and
     * published on /v1/models + /admin/models, so a host-softmax fallback is
     * something an operator can SEE rather than infer from throughput. */
    pn_attn_path_tracker path;
    const void *kv_slot_host_id[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int kv_slot_synced[PN_QWEN35MOE_GPU_MAX_SLOTS];
    /* Phase 10b (lazy host mirror) — parallel per-slot bookkeeping.
     * kv_slot_gen pairs with kv_slot_host_id: a watermark is valid only when
     * the pointer AND the content generation match (a reset or snapshot-
     * restored state keeps its pointer but bumps its generation).
     * kv_slot_kv points at the bound kv struct so a ring realloc can rescue
     * live rows into the host mirror before freeing the only copy. kv_gen is
     * the single-user-ring counterpart, paired with kv_host_id. */
    unsigned kv_slot_gen[PN_QWEN35MOE_GPU_MAX_SLOTS];
    pn_qwen35moe_kv *kv_slot_kv[PN_QWEN35MOE_GPU_MAX_SLOTS];
    unsigned kv_gen;
    int sdiag_flash_logged;     /* QWEN35_SDIAG: flash/host decision reported */
    /* Bumped whenever the slot rings / decode partial scratch are (re)allocated
     * — a captured graph holds their old device addresses. */
    unsigned slots_ring_gen;
    /* L10 (Phase 3) — captured CUDA graph of this ATTENTION layer's batched-
     * decode body: attn_norm -> on-die slots flash -> out-proj + residual ->
     * post_attention_norm -> batched MoE.
     *
     * These layers used to report `graph=never` because the host KV mirror sat
     * MID-body (a blocking fc_sync_ctx + D2H at step 6), and a mid-body host
     * sync invalidates capture. It does not have to sit there: b->a_k / b->a_v
     * still hold this layer's roped K/V after the whole sandwich runs (nothing
     * downstream touches the attention scratch), so the mirror is now issued
     * AFTER the body. The ring re-sync H2D — a no-op in steady state, live only
     * right after a bind/move — is likewise hoisted AHEAD of the body. What is
     * left inside is pure device work over persistent buffers.
     *
     * Position-invariance comes free: the slot kernels already read every
     * user's position from the device pos buffer and recompute each user's
     * split plan on-device. The one host-computed launch argument is
     * `launch_splits` (the flash grid width = max over users of their split
     * plan), which steps up once every `kps` tokens of context — so it is part
     * of the key and costs at most PN_QWEN35MOE_DEC_SPLIT_MAX recaptures over
     * a session's whole life. */
    fc_graph *slots_graph;
    pn_graph_capture_tracker slots_graph_capture;
    int slots_graph_n;          /* nslots at capture (0 = none) */
    int slots_graph_base;       /* slot_base at capture */
    int slots_graph_splits;     /* launch_splits (flash grid) at capture */
    int slots_graph_q4;         /* KV cache mode at capture */
    int slots_graph_max_seq;    /* ring max_seq at capture */
    unsigned slots_graph_gen;   /* slots_ring_gen at capture */
    const void *slots_graph_scratch;  /* b->hidden addr baked into the graph */
} gpu_attn_layer;

/* Persistent per-die GPU scratch for the per-layer blocks. Every buffer is
 * fully overwritten by its producer before use each call, so reuse across
 * layers/tokens is safe. Killing the ~800 cuMemAlloc/cuMemFree pairs per token
 * that the per-call allocations cost was worth ~30% of decode time. */
typedef struct {
    fc_cuda_ctx *ctx;             /* die this scratch belongs to (NULL = unset) */
    /* SSM block */
    fc_buffer *s_qkv, *s_z, *s_beta, *s_alpha, *s_conv_out;
    fc_buffer *s_qg, *s_kg, *s_vg, *s_gbuf, *s_y, *s_gated;
    /* attention block */
    fc_buffer *a_qg, *a_q, *a_z, *a_k, *a_v, *a_out;
    /* MoE block */
    fc_buffer *m_logits, *m_tidx, *m_twt, *m_gate, *m_up, *m_hid;
    fc_buffer *m_exp, *m_out, *m_shexp;
    fc_buffer *m_sgate;   /* [1] sigmoid'd shared-expert gate (fused path) */
    /* residual stream + norm/block scratch + LM-head logits */
    fc_buffer *r_hidden, *r_xnorm, *r_blockout, *r_logits;
    /* [1] int32 decode position for the segment graph (C1): the ONLY
     * per-token input the captured attention kernels read. The host writes
     * pos here (4-byte H2D, outside the graph) then replays the graph; the
     * device-pos kernels (rope_pos, kv_append_dev, flash_decode_dev) derive
     * everything position-dependent from it. */
    fc_buffer *dec_pos;
} gpu_die_scratch;

static void gpu_die_scratch_free(gpu_die_scratch *sc);
static void gpu_fused_free(gpu_fused *f);

/* One die segment's captured full-step graph (C1) + the KV geometry baked
 * into its recorded kernels. */
typedef struct {
    fc_graph *graph;
    pn_graph_capture_tracker capture;  /* bounded retry + residency state */
    int cap_max_seq;        /* kv->max_seq at capture (0 = no attn) */
    int cap_q4;             /* kv->q4 at capture (-1 = no attn layers) */
} gpu_seg_graph;

/* Persistent per-die GPU scratch for the BATCHED (prefill) forward: the
 * per-token buffers above, times the batch capacity n_cap, token-major
 * ([n][dim]; position t's vector at offset t*dim). Allocated lazily on the
 * first batch call and re-allocated only if a later batch is bigger. Every
 * buffer is fully overwritten by its producer for the live n positions. */
typedef struct {
    fc_cuda_ctx *ctx;             /* die this scratch belongs to (NULL = unset) */
    int n_cap;                    /* allocated batch capacity */
    unsigned gen;                 /* bumped on every (re)allocation, unique
                                   * across dies: what a cached descriptor
                                   * over these buffers is keyed on */
    /* residual stream + per-block scratch */
    fc_buffer *hidden, *x_norm, *block_out;             /* [n][n_embd] */
    /* SSM block */
    fc_buffer *s_qkv, *s_conv;                          /* [n][conv_dim] */
    fc_buffer *s_z, *s_qg, *s_kg, *s_vg, *s_y, *s_gated;/* [n][val_dim] */
    fc_buffer *s_beta, *s_alpha, *s_g;                  /* [n][n_v_heads] */
    /* attention block */
    fc_buffer *a_qg;                                    /* [n][2*attn_dim] */
    fc_buffer *a_q, *a_z, *a_out;                       /* [n][attn_dim] */
    fc_buffer *a_k, *a_v;                               /* [n][kv_dim] */
    /* MoE block */
    fc_buffer *m_logits;                                /* [n][n_expert] */
    fc_buffer *m_tidx, *m_twt;                          /* [n][k] */
    fc_buffer *m_gate, *m_up;                           /* [n][shared_ff] */
    fc_buffer *m_hid;                                   /* [n][k*expert_ff] */
    fc_buffer *m_out, *m_shexp;                         /* [n][n_embd] */
    fc_buffer *m_sgate;                                 /* [n] sigmoid'd
                                                         * shexp gate (fused
                                                         * one-slot path) */
    /* Expert-union dedup request lists — allocated ONLY when
     * gpu_moe_dedup_enabled(), so the default-off build carries no extra
     * VRAM and the decode path is untouched. */
    fc_buffer *m_ecount;                                /* [n_expert] int32 */
    fc_buffer *m_ereq;                                  /* [n_expert][n] int32 */
    /* Compacted (expert, request-tile) work list, for the compacted-grid
     * gate+up (FC_MOE_DEDUP_COMPACT, default off). [fc_moe_dedup_wmax()] int2,
     * i.e. 2.8 KB at the prefill shape. */
    fc_buffer *m_wlist;
    /* Decode-slots row map for the group this scratch is running (set with
     * the positions by gpu_pos_buf_upload): rows_h[u] is the persistent
     * ring/state row of launch position u, rows_g the same map on the die
     * (a BORROWED view into the die's pos buffer — not freed here). */
    fc_buffer *rows_g;
    int rows_h[PN_QWEN35MOE_GPU_MAX_SLOTS];
} gpu_batch_scratch;

static void gpu_batch_scratch_free(gpu_batch_scratch *b);

/* Host-side scratch carried across the GPU forward (see the section header
 * further down). Owned by pn_qwen35moe_gpu so the buffers are allocated once,
 * not per token (3 x n_vocab-float callocs per token was measurable mmap/
 * page-fault churn on the 2-core K80 host). */
typedef struct {
    float *a;     /* general scratch, >= max(conv_dim, n_head*head_dim*2, val_dim) */
    float *b;
    float *c;
    int cap;
} gpu_host_scratch;

static void gpu_host_scratch_free(gpu_host_scratch *h);

/* QWEN35_ASYNC_KVMIRROR — deferred host-KV-mirror store (per die).
 *
 * The flash-attn slots path (gpu_attn_block_slots_flash step 6) normally does a
 * BLOCKING fc_sync_ctx + cuMemcpyDtoH of the roped K/V into the host mirror at
 * EVERY attention layer. On the single-threaded 2-die driver that per-layer
 * stream drain defeats the die-pipeline's overlap (a die can't stay busy while
 * the host blocks reaping the other's mirror copy).
 *
 * The host mirror is only READ at step BOUNDARIES (slot bind / move / session
 * preserve) or to re-sync the device ring right after such a boundary — never
 * mid-step in steady-state flash decode (the device ring is authoritative). So
 * the mirror store can be safely deferred: per attention layer we (1) queue an
 * ASYNC DtoH of that layer's roped K/V into a per-layer host staging slot on
 * the die's own stream (no drain), and (2) record a pending-store descriptor.
 * At the END of the decode step, once per die, we fc_sync_ctx ONCE (draining
 * the already-queued async copies) and replay every pending host store. The
 * mirror is thus identical to the per-layer path by the time any boundary
 * consumer runs, but zero per-layer host stalls happen in between.
 *
 * Each descriptor carries everything the store loop needs; the staged K/V bytes
 * live in a per-die growable arena (kv_stage) so overlapping layers never
 * collide. Gated behind QWEN35_ASYNC_KVMIRROR (default OFF). */
typedef struct {
    pn_qwen35moe_kv *kv[PN_QWEN35MOE_GPU_MAX_SLOTS];  /* per-user host cache */
    int   pos[PN_QWEN35MOE_GPU_MAX_SLOTS];            /* this slot's position */
    int   nslots;
    int   kv_q4;
    int   kv_sink, kv_win;
    int   n_head_kv, head_dim, max_seq, kv_dim;
    size_t stage_off;   /* byte offset into the die arena of this layer's K||V */
} qw_kv_store;

typedef struct {
    qw_kv_store *stores;   /* [n_cap] pending descriptors this step */
    int   n;               /* live descriptor count */
    int   n_cap;
    float *stage;          /* host staging arena: 2*kv_dim floats per descriptor */
    size_t stage_bytes;    /* allocated arena size */
    size_t stage_used;     /* bytes handed out this step */
} qw_kv_defer;

/* Top-level GPU residency handle, stored on the model.
 *
 * The model's layers are split across `n_die` dies (1 .. the placement width
 * the die manager bound). Die i owns global layers [layer_lo[i], layer_hi[i]),
 * planned by pn_plan_die_layer_split — the SAME partitioner admission prices
 * the load with, so what is projected and what is uploaded cannot disagree.
 * token_embd is never uploaded (the embedding is a host row copy at embed
 * time); the final norm + LM head ride on the LAST die.
 *
 * Single-die (n_die == 1): head_ctx is NULL, every layer's .ctx == ctx[0], bs
 * is NULL (no barswap needed — the operator acquires one die externally).
 *
 * Split (n_die >= 2): each context is bound to its own die's CUDA ordinal in
 * small-BAR mode, or to ordinal 0 while that die sits in the barswap MMIO slot
 * in the legacy regime, where the forward bs_acquire()s the die before touching
 * its context. Between consecutive dies the forward hands the residual stream
 * across — device-to-device when the pair has peer access (peer_ok[i]), else
 * staged through the host. */
/* A captured verify segment (see verify_segment): per die, per row count. */
typedef struct {
    fc_graph *graph;
    uint64_t key;               /* device-address fingerprint at capture */
    int splits;                 /* rows-attention grid width at capture */
    long captures, failures, replays;
} gpu_verify_graph;

/* The MTP draft head's device side (see the MTP section near the end of this file). */
typedef struct {
    int on;                       /* uploaded (QWEN35_MTP=1 and the GGUF has a head) */
    fc_cuda_ctx *ctx;             /* the head die (the LM head's) */
    gpu_attn_layer d;             /* attention + MoE weights; its own q4 KV ring */
    gpu_weight eh_e, eh_h;        /* eh_proj halves: embedding / hidden */
    gpu_vec enorm, hnorm, head_norm;
    gpu_weight head_lo;           /* LM-head rows [0, sub_n) (aliases g->output) */
    gpu_weight head_hi;           /* rows [tail_first, n_vocab) (a view of g->output) */
    int sub_n, tail_first, n_tail;
    fc_buffer *e, *en, *hn, *x, *xn, *k, *v;   /* [QW_MTP_ROWS_CAP rows] */
    fc_buffer *qg, *q, *z, *aout, *blk, *tn, *mout, *h_chain, *logits;
    fc_buffer *m_g, *l_g, *acc_g; /* rows-kernel split partials */
    float *e_host;                /* embedding rows staged for upload */
    fc_buffer *h_ver;             /* the last trunk batch's output_norm'd rows */
    int ver_cap, ver_n;
    fc_buffer *h_pend;            /* backlog rows' h, positions pend_base.. */
    uint32_t pend_next[320];      /* backlog rows' next tokens (last: set by the draft) */
    int pend_n, pend_base, next_pos;
    const void *state;            /* the session the head's KV ring belongs to */
    long drafts, kv_rows, full_rows;
    double draft_ms;
    double prof_ms[4];            /* QWEN35_MTP_PROF: kv rows / attention / MoE / head, sync'd */
    double kv_prof_ms[3];         /* QWEN35_MTP_PROF=2: kv rows' host embed / upload / kernels */
    pn_mtp_pick_fn pick;          /* the draft choice from the head's pairs (NULL = argmax) */
    void *pick_ctx;
    int pick_index;               /* the draft mtp_pick is choosing (0-based) */
} gpu_mtp;

/* QWEN35_TP in-layer exchange buffers of one lane, per die: the core gather,
 * the out-projection gather, then the dense FFN down reduce or the MoE hidden
 * and output gathers. Every copy is ordered on the dies' streams, so ONE set
 * serves a stream of exchanges; a second lane (QW_LANE_DECODE) is what lets a
 * decode stage run between two layers of a prefill chunk without touching the
 * chunk's exchange state. */
typedef struct {
    int cap;                                         /* rows the buffers hold */
    fc_buffer *core[PN_DIE_LAYER_SPLIT_MAX_DIES];    /* [cap][core] full order */
    fc_buffer *half[PN_DIE_LAYER_SPLIT_MAX_DIES];    /* [cap][n_embd/2] out rows */
    fc_buffer *blk[PN_DIE_LAYER_SPLIT_MAX_DIES];     /* [cap][n_embd] */
    fc_buffer *recv[PN_DIE_LAYER_SPLIT_MAX_DIES];    /* [cap][n_embd] partner partial */
    fc_buffer *hidl[PN_DIE_LAYER_SPLIT_MAX_DIES];    /* MoE: [cap*(k+1)][ff/2] rank hid */
    fc_buffer *hidf[PN_DIE_LAYER_SPLIT_MAX_DIES];    /* MoE: [cap*(k+1)][ff] full hid */
    fc_buffer *mol[PN_DIE_LAYER_SPLIT_MAX_DIES];     /* MoE: [2*cap][n_embd/2] rank rows */
    fc_buffer *mof[PN_DIE_LAYER_SPLIT_MAX_DIES];     /* MoE: [2*cap][n_embd] routed+shared */
    fc_event *ev[PN_DIE_LAYER_SPLIT_MAX_DIES][TP_N_XCHG];
    /* QWEN35_TP_XCHG (default on): gathers are fc_xchg_2d launches ordered by
     * these per-die counters (TP_N_XCHG slots, zeroed together) instead of
     * events + copies; xchg = 0 when off or not allocated. */
    fc_buffer *sync[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int xchg;
} qw_tp_xset;

typedef struct pn_qwen35moe_gpu {
    gpu_ssm_layer  *ssm;          /* [n_ssm] */
    gpu_attn_layer *attn;         /* [n_attn] */
    int n_ssm;
    int n_attn;
    gpu_vec output_norm;          /* final RMSNorm [n_embd] (on the LAST die) */
    gpu_weight output;            /* LM head (on the LAST die) */
    /* token_embd is read host-side at embed time (just a row copy), so it is
     * not uploaded; the embedded row is uploaded fresh each token to ctx[0]. */

    /* T3 — vocab-split LM head (QWEN35_VOCAB_SPLIT, EXACTLY 2 dies). It pairs
     * the head die with die 0 specifically; at n_die > 2 those are not adjacent
     * in the pipeline and the head's partner would be a die the residual stream
     * left several hops ago, so the path stays off (see vocab_split_eligible).
     * When ON the LM head is split by vocab rows: die A holds the TOP half
     * [0, vocab_half)
     * in output_a + a copy of the final norm (output_norm_a), die B holds the
     * BOTTOM half [vocab_half, n_vocab) in `output` (re-uploaded to the slice).
     * Memory-neutral: die B's head shrinks to a half exactly as die A grows by
     * a half. After the final norm on die B, x_norm [K][n_embd] is P2P-copied
     * to die A (xnorm_a); each die computes its half's logits + local top-k and
     * the two per-slot top-k lists are merged order-stably on the host. Head-
     * phase only (once per step, outside the layer loop) → no per-layer sync
     * hazard. vocab_split_ready gates the whole path; falls back to the single-
     * die `output` head if setup failed / peer unavailable. */
    gpu_weight output_a;          /* LM head TOP half [0,vocab_half) (on ctx_a) */
    gpu_weight output_b;          /* LM head BOTTOM half [vocab_half,nv) (ctx_b) */
    gpu_vec output_norm_a;        /* final RMSNorm copy (on ctx_a) */
    fc_buffer *xnorm_a;           /* [cap][n_embd] x_norm mirror on die A */
    int xnorm_a_cap;
    fc_buffer *spec_logits_a;     /* [cap][vocab_half] logits on die A */
    int spec_logits_a_cap;
    fc_buffer *spec_logits_b;     /* [cap][n_vocab-vocab_half] logits on die B */
    int spec_logits_b_cap;
    int vocab_half;               /* rows in the die-A (top) half */
    int vocab_split_ready;        /* 1 = split head uploaded + peer enabled */

    /* QWEN35_TP (dense FFN or grouped Q8_0 MoE, exactly 4 dies): dies (0,1) and (2,3) are rank
     * pairs of one board; die 2s+p is rank p of stage s and holds half of
     * every layer in [layer_lo, layer_hi) of its stage (both ranks list the
     * stage's range). Rank 0's halves sit in ssm[idx] / attn[idx], rank 1's at
     * idx + tp_n_ssm / idx + tp_n_attn (the arrays are doubled, so every
     * per-layer loop sees both). tp_cfg carries a rank's local dims. The
     * per-die exchange buffers and events serve the in-layer exchanges (core
     * gather, out-projection gather, then the dense FFN down reduce or the MoE
     * hidden and output gathers). */
    int tp;
    int tp_n_ssm, tp_n_attn;
    pn_qwen35moe_config tp_cfg;
    /* The exchange buffers, per lane (see qw_tp_xset; tpx[g->lane] is live). */
    qw_tp_xset tpx[QW_N_LANES];
    /* MoE TP decode-slots step: one captured graph per stage spanning both
     * ranks (QWEN35_TP_GRAPH), keyed by tp_slots_graph_key; tp_gen moves when
     * an exchange or slot-state buffer is reallocated; tp_gev fence a graph
     * against the partner die's stream. */
    unsigned tp_gen;
    int tp_capturing;   /* a stage graph is being captured: gathers are copy kernels */
    fc_graph *tp_sgraph[2];
    uint64_t tp_sgraph_key[2], tp_sgraph_warm[2];
    int tp_sgraph_fail[2];
    fc_event *tp_gev[PN_DIE_LAYER_SPLIT_MAX_DIES];
    /* QWEN35_PCIE_*: staging of the step's input / hop / position copies (qw_pcie_for);
     * NULL = legacy blocking copies. pcie_failed: creation failed once, never retried. */
    qw_pcie_stage *pcie;
    int pcie_failed;
    /* Per rank: global conv channel of each local one, global v-head of each
     * local one (host state transfers gather / scatter through these). */
    int *tp_chan[2], *tp_vhead[2];

    /* N-die residency / multiplexing. Die i holds layers [layer_lo[i],
     * layer_hi[i]); the ranges come from pn_plan_die_layer_split. */
    int n_die;                    /* 1 .. PN_DIE_LAYER_SPLIT_MAX_DIES */
    fc_cuda_ctx *ctx[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int die[PN_DIE_LAYER_SPLIT_MAX_DIES];     /* CUDA ordinal / barswap die id */
    int layer_lo[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int layer_hi[PN_DIE_LAYER_SPLIT_MAX_DIES];
    /* peer_ok[i] = the die i -> die i+1 hand-off can go device-to-device. */
    int peer_ok[PN_DIE_LAYER_SPLIT_MAX_DIES];
    /* The die the final norm + LM head live on: the LAST one. NULL/-1 mirrors
     * are kept for the head die rather than recomputed at every use, because
     * "the head's context" is read far more often than the split changes. */
    fc_cuda_ctx *head_ctx;        /* ctx[n_die-1], or NULL when n_die == 1 */
    int head_die;                 /* die[n_die-1] */
    int split_layer;              /* layer_hi[0]; == n_layer when n_die == 1 */
    bs_context  *bs;              /* barswap handle (NULL in single-die mode) */
    int active_die;               /* die currently in the MMIO slot (-1 = none) */

    /* Lazily-created persistent scratch, one per die (see gpu_die_scratch). */
    gpu_die_scratch scratch[PN_DIE_LAYER_SPLIT_MAX_DIES];
    /* The decode lane's own die and batch scratch (QW_LANE_DECODE), and the
     * live lane: gpu_get_scratch / gpu_get_batch_scratch / TPX hand out the
     * buffers of g->lane. */
    gpu_die_scratch scratch_dec[PN_DIE_LAYER_SPLIT_MAX_DIES];
    gpu_batch_scratch bscratch_dec[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int lane;
    /* Inside an overlap cycle's head (pn_qwen35moe_pd_cycle_cuda): no solo-step
     * MTP row capture. */
    int pd_cycle;

    /* Batched-forward scratch, one per die, plus its host staging regions
     * (sized for bhost_cap positions; see gpu_get_batch_host). */
    gpu_batch_scratch bscratch[PN_DIE_LAYER_SPLIT_MAX_DIES];
    float *bhost[4];
    /* L5 — a SECOND host-staging set so the die-pipeline can run group0's
     * die-B host mirror concurrently with group1's die-A host mirror without
     * the two colliding on shared bhost regions. Lazily allocated (only when
     * QWEN35_DIE_PIPELINE engages) at the same per_pos*cap size as bhost. */
    float *bhost2[4];
    int bhost_cap;

    /* Persistent host-side scratch (allocated on first forward). */
    gpu_host_scratch host;
    int host_ok;

    /* Device top-k logits fetch (QWEN35_GPU_TOPK, see gpu_fetch_logits): the
     * bookkeeping for the sparse host presentation lives on the pn_qwen35moe_state
     * that OWNS the host logits buffer (s->topk_filled / s->topk_prev_ids /
     * s->topk_prev_n), not here — see the comment on gpu_fetch_logits. */

    /* Speculative-verify bookkeeping: [n][n_vocab] LM-head buffer on the
     * head's die (lazily sized spec_logits_cap positions) and the position
     * count of the LAST verify (0 = none pending) — spec_commit validates
     * n_accept against it. */
    fc_buffer *spec_logits;
    int spec_logits_cap;
    /* pn_qwen35moe_spec_verify_topk: per-row top-k into these host arrays (k = 0: argmax). */
    int spec_topk_k;
    int *spec_topk_ids;
    float *spec_topk_vals;
    gpu_mtp mtp;
    int spec_n;
    /* Phase 9.2 — the state and base position of that pending verify, so
     * spec_commit can fold the ACCEPTED KV rows into each attention layer's
     * ring watermark (the verify itself leaves it at the committed frontier;
     * see gpu_attn_block_batch_flash). NULL/0 when no verify is pending. */
    pn_qwen35moe_state *spec_state;
    int spec_pos0;

    /* Multi-user decode bookkeeping (see the decode-slots section):
     * which host state each slot is bound to and the per-die int32 RoPE
     * position buffers (indexed by die).  The sparse top-k presentation state
     * is NOT per slot — it belongs to the per-user pn_qwen35moe_state whose
     * logits buffer it describes. */
    const void *slot_state[PN_QWEN35MOE_GPU_MAX_SLOTS];
    /* Short logits row: a constrained slot's full row kept on the head die until its
     * draw (srow_state[r] = the state it belongs to, NULL = free; the buffers are kept
     * for reuse and freed with the model). */
    const void *srow_state[PN_QWEN35MOE_GPU_MAX_SLOTS];
    fc_buffer *srow_buf[PN_QWEN35MOE_GPU_MAX_SLOTS];
    fc_cuda_ctx *srow_ctx[PN_QWEN35MOE_GPU_MAX_SLOTS];
    /* What each slot row's SSM image is, so a resumed turn can re-seed the
     * shared SSM buffers ON THE CARD instead of from host RAM: the bound
     * state's KV content generation at bind and the position the image is at
     * (bind sets it, every decode step that steps the row advances it). A
     * reset or restore bumps the generation, which voids the image. */
    unsigned slot_gen[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int slot_pos[PN_QWEN35MOE_GPU_MAX_SLOTS];
    /* Turn-start checkpoint per row: whose it is, that state's KV content
     * generation when it was taken (a reset or restore voids it) and its
     * position. ckpt_state NULL = none. */
    const void *ckpt_state[PN_QWEN35MOE_GPU_MAX_SLOTS];
    unsigned ckpt_gen[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int ckpt_pos[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int n_slots;                     /* bound slot count (0 = none) */
    /* Per-die int32 [pos | rows] buffer: positions at [0, MAX), the row map
     * at [MAX, 2*MAX) — one fixed-size H2D per die per step carries both, and
     * neither device address ever moves (captured graphs stay valid). */
    fc_buffer *pos_buf[PN_DIE_LAYER_SPLIT_MAX_DIES];
    fc_buffer *rows_view[PN_DIE_LAYER_SPLIT_MAX_DIES];
    /* QWEN35_BATCH_VERIFY: per-die split partials of the R-row verify
     * attention (fc_attn_flash_decode_rows_q4_ctx), FC_ATTN_ROWS_MAX rows x
     * n_head x DEC_SPLIT_MAX. A die's attention layers run in stream order,
     * so one set per die serves all of them. */
    fc_buffer *bv_m_g[PN_DIE_LAYER_SPLIT_MAX_DIES];
    fc_buffer *bv_l_g[PN_DIE_LAYER_SPLIT_MAX_DIES];
    fc_buffer *bv_acc_g[PN_DIE_LAYER_SPLIT_MAX_DIES];
    fc_buffer *vpos_g[PN_DIE_LAYER_SPLIT_MAX_DIES];   /* verify [pos x5 | rows x5] */
    fc_buffer *vrows_g[PN_DIE_LAYER_SPLIT_MAX_DIES];  /* view: the rows half */
    gpu_verify_graph vgraph[PN_DIE_LAYER_SPLIT_MAX_DIES][5];
    /* The current step's row map (launch position u -> persistent row). */
    int step_rows[PN_QWEN35MOE_GPU_MAX_SLOTS];

    /* Full-step segment graphs (C1, QWEN35_GRAPH_FULL): ONE captured CUDA
     * graph per die segment covering EVERY layer body in that segment (SSM
     * and attention interleaved) — the whole per-die decode step replays
     * with a single cuGraphLaunch. Indexed by die: seg[i] covers layers
     * [layer_lo[i], layer_hi[i]); single-die uses seg[0] for all layers. The
     * embed upload, the boundary hand-offs and the end-of-step logits fetch
     * stay OUTSIDE (they host-sync). cap_max_seq / cap_q4 are the KV geometry
     * baked into the captured kernels — replay is refused (and the graph
     * dropped for recapture if the ring reallocates) when the live state
     * disagrees. */
    gpu_seg_graph seg[PN_DIE_LAYER_SPLIT_MAX_DIES];

    /* C2 (QWEN35_ASYNC_SCHED): the in-flight ASYNC decode step's phase
     * machine — begin() launches a die segment and returns without blocking;
     * poll() reaps completion via a CUDA event and advances to the next
     * phase, so a 2-core host keeps many cards' steps in flight at once.
     * One slot per model suffices: the serve layer's per-model gen_lock
     * already serializes generations on this model, and begin/poll are
     * called from that single generation thread. as_ev[i] belongs to ctx[i]
     * (lazily created; a failed create latches via as_ev_tried and async stays
     * off — the sync path is unaffected). The phase machine itself is a
     * two-segment one, so async engages only at n_die <= 2 (see
     * pn_qwen35moe_decode_begin_cuda). */
    int as_phase;               /* qw_async_phase */
    int as_pos;                 /* position of the in-flight token */
    fc_event *as_ev[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int as_ev_tried[PN_DIE_LAYER_SPLIT_MAX_DIES];

    /* QWEN35_ASYNC_KVMIRROR — per-die deferred host-KV-mirror store state,
     * indexed by die (see qw_kv_defer above). */
    qw_kv_defer defer[PN_DIE_LAYER_SPLIT_MAX_DIES];

    /* QWEN35_TWO_THREAD — persistent per-die worker threads (index 0 = die A,
     * 1 = die B). Each thread owns its die's CUDA context launch queue, so the
     * two dies' kernel launches are issued in parallel instead of serialized on
     * one host thread. Lazily spun up on first threaded decode; torn down at
     * model free. See qw_tt_* below. */
    struct qw_tt *tt;

    /* Intrusive link in the process-wide registry of live GPU contexts (see
     * qw_gpu_register below).  Typed void* only because this struct is an
     * anonymous typedef; it always holds a pn_qwen35moe_gpu *. */
    void *reg_next;
} pn_qwen35moe_gpu;

/* The live lane's TP exchange set. */
#define TPX(g) (&(g)->tpx[(g)->lane])

static void qw_defer_free(pn_qwen35moe_gpu *g);   /* teardown (impl below) */
/* MTP draft head (impl in the MTP section at the end of this file). */
static int mtp_upload(pn_qwen35moe_model *model, pn_qwen35moe_gpu *g,
                      const pn_qwen35moe_config *cfg);
static void mtp_free(pn_qwen35moe_gpu *g);
static int mtp_prof_on(void);
static void mtp_capture_rows(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx, fc_buffer *x_norm,
                             int n, int n_embd);
static int mtp_push(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, pn_qwen35moe_state *s,
                    int pos_end, const uint32_t *tokens, int n);
static void qw_tt_free(pn_qwen35moe_gpu *g);      /* TT teardown (impl below) */

/* ============================================================
 * Live-GPU-context registry — the session-state lifetime hook
 *
 * gpu_attn_layer keeps raw per-slot back-pointers into a session's
 * pn_qwen35moe_state (kv_slot_kv, kv_slot_host_id) and gpu_ssm_layer keeps one
 * (host_state).  pn_qwen35moe_state_free frees that state and has no model
 * handle, so it cannot reach the GPU layers directly — which is how a freed
 * kv survived as a dangling back-pointer and got dereferenced by the next
 * bind's lazy-mirror park copy (heap-use-after-free; that copy is gone, the
 * ring-realloc rescue and the slot-row SSM seed still dereference them).
 *
 * The lifetime is broken at the free site instead of guarded at the use site:
 * every live GPU context is registered here, and pn_qwen35moe_state_free calls
 * pn_qwen35moe_gpu_forget_state_cuda, which drops every reference to the dying
 * state before its memory goes back to the allocator.  A stale back-pointer
 * therefore cannot exist, so no consumer needs to (and none can) test one.
 *
 * The list is touched only at model load/free and session free, so a plain
 * mutex costs nothing measurable.
 * ============================================================ */
static pn_qwen35moe_gpu *qw_gpu_reg_head = NULL;
static pthread_mutex_t qw_gpu_reg_mtx = PTHREAD_MUTEX_INITIALIZER;

static void qw_gpu_register(pn_qwen35moe_gpu *g)
{
    pthread_mutex_lock(&qw_gpu_reg_mtx);
    g->reg_next = qw_gpu_reg_head;
    qw_gpu_reg_head = g;
    pthread_mutex_unlock(&qw_gpu_reg_mtx);
}

static void qw_gpu_unregister(pn_qwen35moe_gpu *g)
{
    pthread_mutex_lock(&qw_gpu_reg_mtx);
    pn_qwen35moe_gpu **pp = &qw_gpu_reg_head;
    while (*pp) {
        if (*pp == g) {
            *pp = (pn_qwen35moe_gpu *)g->reg_next;
            g->reg_next = NULL;
            break;
        }
        pp = (pn_qwen35moe_gpu **)&(*pp)->reg_next;
    }
    pthread_mutex_unlock(&qw_gpu_reg_mtx);
}


/* MemoryAware Phase 1: publish one die's post-upload VRAM truth (cuMemGetInfo
 * free + our own per-ctx alloc tally) into the live pn_mem_state, so admission
 * schedules against real numbers from the moment the model is resident. */
static void gpu_feed_die_mem_state(fc_cuda_ctx *ctx, int die_id)
{
    if (!ctx || !pn_memwatch_running())
        return;
    size_t die_free = 0, die_total = 0;
    if (fc_mem_info_ctx(ctx, &die_free, &die_total) != 0 || die_total == 0)
        return;
    uint64_t reserve = pn_die_reserve_bytes();
    uint64_t budget = die_total > reserve ? die_total - reserve : 0;
    pn_memwatch_set_die(die_id, die_free, budget, fc_ctx_alloc_bytes(ctx));
}

/* The KV window admission schedules against (mirrors qwen35moe_max_seq's
 * QWEN35_MAX_SEQ clamp). KV Phase 1: a per-model ctx_cap (set at load) wins so
 * the model-load fit gate budgets the SAME window the ring is sized to. Pass
 * the model to honor its cap; NULL keeps the global env behavior. */
static int gpu_admit_max_ctx_model(const pn_qwen35moe_model *model)
{
    const pn_qwen35moe_config *cfg =
        model ? pn_qwen35moe_model_config(model) : NULL;
    int max_ctx;
    if (cfg && cfg->ctx_cap > 0) {
        max_ctx = cfg->ctx_cap;
    } else {
        const char *seq_env = getenv("QWEN35_MAX_SEQ");
        max_ctx = seq_env ? atoi(seq_env) : 4096;
        if (max_ctx <= 0) max_ctx = 4096;
    }
    /* Clamp to the SAME ceiling the KV ring is allocated against
     * (qwen35moe_max_seq), so admission never promises less than the ring can
     * hold. */
    if (max_ctx > PN_QWEN35MOE_MAX_SEQ_CAP) max_ctx = PN_QWEN35MOE_MAX_SEQ_CAP;
    return max_ctx;
}

/* VRAM a pn_qweight costs once uploaded, mirroring gpu_weight_upload's own
 * arithmetic: the repacked row stride when the type repacks, the file stride
 * otherwise. Needed for the LM head because a TIED model has no output.weight
 * in the file at all — its head is the F32 materialization of token_embd
 * (qwen35moe.c), i.e. ~4x what the tensor table would suggest. */
static uint64_t gpu_qweight_upload_bytes(const pn_qweight *w)
{
    if (!w || !w->data || w->rows <= 0 || w->cols <= 0)
        return 0;
    size_t rstride = fc_repack_row_bytes((int)w->type, w->cols);
    if (rstride == 0)
        rstride = w->row_stride;
    return (uint64_t)w->rows * (uint64_t)rstride;
}

/* Bytes the LM head + final norm cost on whichever die holds them (die B in a
 * 2-die split, die A single-die). */
static uint64_t gpu_head_weight_bytes(pn_qwen35moe_model *model)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    uint64_t head = gpu_qweight_upload_bytes(pn_qwen35moe_model_output(model));
    if (cfg && cfg->n_embd > 0)
        head += (uint64_t)cfg->n_embd * sizeof(float);   /* output_norm */
    return head;
}

/* This die's shape for qwen35moe_die_projection.h: its layers by type, the
 * weights it holds (repacked layers + `extra_weight_bytes` + the MTP draft head
 * on the last die when kept), the session blocks the serve will carve and the
 * runtime scratch. Returns 0, or -1 when the tensor table yields no blk.*
 * bytes for the range (an unfamiliar layout: the caller gates conservatively). */
static int gpu_die_shape(pn_qwen35moe_model *model, int layer_lo, int layer_hi,
                         uint64_t extra_weight_bytes, int last_die, int max_ctx,
                         int kv_q4, int tp, pn_qwen35_die_shape *sh)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    pn_gguf_file *gf = pn_qwen35moe_gguf(model);

    memset(sh, 0, sizeof(*sh));
    sh->layer_weight_bytes = pn_gguf_layer_range_upload_bytes(gf, layer_lo, layer_hi);
    if (sh->layer_weight_bytes == 0)
        return -1;
    sh->extra_weight_bytes = extra_weight_bytes;
    /* mtp_upload puts the head next to the LM head: a full attention + MoE block
     * (~907 MiB on Qwen3.6-35B-A3B Q8_0) the gate used to leave out. */
    if (last_die && pn_qwen35moe_model_mtp(model) && cfg->mtp_layer >= 0)
        sh->mtp_weight_bytes = pn_gguf_layer_range_upload_bytes(gf, cfg->mtp_layer,
                                                                cfg->mtp_layer + 1);
    for (int layer = layer_lo; layer < layer_hi; layer++) {
        if (pn_qwen35moe_model_layer_type(model, layer) == PN_QWEN35MOE_LAYER_ATTN)
            sh->n_attn++;
        else
            sh->n_ssm++;
    }
    sh->session_rows = pn_qwen35moe_model_card_sessions(model);
    sh->max_seq = max_ctx;
    sh->kv_q4 = kv_q4 != 0;
    sh->n_head_kv = cfg->n_head_kv;
    sh->head_dim = cfg->head_dim;
    sh->ssm_conv = cfg->ssm_conv;
    sh->conv_dim = cfg->conv_dim;
    sh->ssm_state = cfg->ssm_state;
    sh->ssm_dt_rank = cfg->ssm_dt_rank;
    sh->scratch_bytes = PN_ADMIT_DIE_SCRATCH_BASE_BYTES;
    if (tp) {
        /* QWEN35_TP rank: half of every layer's rows and heads; a dense FFN
         * down is a parity slab, 80 B per 128 inputs instead of half of 144
         * (+1/18 of the full down); a MoE router is whole on both ranks. */
        uint64_t extra = 0;
        for (int layer = layer_lo; layer < layer_hi; layer++) {
            int idx = pn_qwen35moe_model_layer_index(model, layer);
            const pn_qwen35moe_moe *moe =
                pn_qwen35moe_model_layer_type(model, layer) == PN_QWEN35MOE_LAYER_ATTN
                    ? &pn_qwen35moe_model_attn_layer(model, idx)->moe
                    : &pn_qwen35moe_model_ssm_layer(model, idx)->moe;
            extra += cfg->n_expert ? gpu_qweight_upload_bytes(&moe->ffn_gate_inp) / 2
                                   : gpu_qweight_upload_bytes(&moe->ffn_down_shexp) / 18;
        }
        sh->layer_weight_bytes = sh->layer_weight_bytes / 2 + extra;
        sh->n_head_kv = cfg->n_head_kv / 2;
        sh->conv_dim = cfg->conv_dim / 2;
        sh->ssm_dt_rank = cfg->ssm_dt_rank / 2;
    }
    return 0;
}

/* The old proration for a layout whose tensor table names no blk.* tensors:
 * a share of the whole file + one KV window + the clamped scratch, gated against
 * the full per-die reserve. Conservative on purpose. */
static uint64_t gpu_die_need_prorated(pn_qwen35moe_model *model, int n_layers_here,
                                      int n_layer, int max_ctx, int kv_q4)
{
    pn_gguf_file *gf = pn_qwen35moe_gguf(model);
    uint64_t file_bytes = gf ? (uint64_t)gf->mmap_size : 0;
    uint64_t weights = n_layer > 0 ? file_bytes * (uint64_t)n_layers_here / (uint64_t)n_layer
                                   : file_bytes;
    uint64_t per_tok = pn_qwen35moe_kv_bytes_per_token(model, kv_q4 != 0);
    uint64_t kv = pn_admit_die_kv_bytes(per_tok, kv_q4 != 0, max_ctx, n_layers_here, n_layer);
    uint64_t reserve = pn_die_reserve_bytes(), margin = pn_die_load_margin_bytes();
    return weights + kv + pn_admit_die_scratch_bytes(max_ctx) +
           (reserve > margin ? reserve - margin : 0);
}

/* MemoryAware per-die fit gate (§6.1): verify ONE die of an N-die split — or
 * the single die of an unsplit load, which is the same question at N=1 — can
 * hold what it will hold while serving (qwen35moe_die_projection.h: its weights
 * at their on-die size incl. the MTP head, the session blocks the serve carves,
 * the runtime scratch) with PN_DIE_LOAD_MARGIN_BYTES to spare. Every term is an
 * allocation the build makes; the projection used to charge Q8_0 at file size,
 * leave out the MTP head and price the session rings through a clamped
 * per-token scratch, errors that only cancelled by luck. Feeds the die's measured
 * truth into memwatch and warns on stranded VRAM (§8.2) on the way. Returns 0 to
 * proceed, -1 to refuse — the caller fails the load cleanly instead of letting a
 * doomed multi-minute upload cascade into `cuMemAlloc failed: 2`. Fail-open
 * when the die's free/total cannot be read (no truth to gate against). */
static int gpu_admit_die_share(pn_qwen35moe_model *model, fc_cuda_ctx *ctx,
                               int die_id, int layer_lo, int layer_hi,
                               uint64_t extra_weight_bytes, int last_die,
                               int n_layer, int max_ctx, int kv_q4, int tp)
{
    int n_layers_here = layer_hi - layer_lo;
    size_t die_free = 0, die_total = 0;
    if (fc_mem_info_ctx(ctx, &die_free, &die_total) != 0 || die_total == 0)
        return 0;

    uint64_t reserve = pn_die_reserve_bytes();
    uint64_t budget = die_total > reserve ? die_total - reserve : 0;
    if (pn_memwatch_running())
        pn_memwatch_set_die(die_id, die_free, budget, fc_ctx_alloc_bytes(ctx));

    if (die_free < die_total / 2)
        fprintf(stderr, "[admit] WARNING die ordinal %d shows only "
                "%llu / %llu MiB free — possible stranded VRAM from a "
                "prior crash (see startup reclaim, MemoryAware §8.2)\n",
                die_id,
                (unsigned long long)(die_free >> 20),
                (unsigned long long)(die_total >> 20));

    pn_qwen35_die_shape sh;
    pn_qwen35_die_need nd = { 0 };
    uint64_t need;
    if (gpu_die_shape(model, layer_lo, layer_hi, extra_weight_bytes, last_die, max_ctx,
                      kv_q4, tp, &sh) == 0) {
        pn_qwen35_die_project(&sh, &nd);
        need = nd.total;
    } else {
        need = gpu_die_need_prorated(model, n_layers_here, n_layer, max_ctx, kv_q4);
    }
    uint64_t margin = pn_die_load_margin_bytes();
    uint64_t usable = die_free > margin ? die_free - margin : 0;
    if (need > usable) {
        fprintf(stderr, "[admit] REFUSE model load: die ordinal %d has "
                "%llu MiB free but its share needs ~%llu MiB "
                "(%d/%d layers + KV@%d + scratch) — refusing before the "
                "cuMemAlloc cascade\n",
                die_id,
                (unsigned long long)(die_free >> 20),
                (unsigned long long)(need >> 20),
                n_layers_here, n_layer, max_ctx);
        pn_accel_refusal_note("GPU die %d needs ~%llu MiB, has %llu MiB usable "
                              "(%d/%d layers + KV for %d tokens + scratch)",
                              die_id, (unsigned long long)(need >> 20),
                              (unsigned long long)(usable >> 20),
                              n_layers_here, n_layer, max_ctx);
        return -1;
    }
    fprintf(stderr, "[admit] die ordinal %d: needs ~%llu MiB (weights %llu + %d session "
            "block(s) %llu + scratch %llu) of %llu MiB free, %llu MiB spare over the "
            "%llu MiB margin\n", die_id, (unsigned long long)(need >> 20),
            (unsigned long long)(nd.weights >> 20),
            sh.session_rows > 0 ? sh.session_rows : 1,
            (unsigned long long)(nd.sessions >> 20), (unsigned long long)(nd.scratch >> 20),
            (unsigned long long)(die_free >> 20),
            (unsigned long long)((usable - need) >> 20), (unsigned long long)(margin >> 20));
    return 0;
}

/* The boundary knob, or 0 when unset. QWEN35_GPU_SPLIT_LAYER predates
 * placement and tunes ONE boundary, which describes a 2-die split completely
 * and nothing wider — pn_plan_die_layer_split honours it at that width and
 * ignores it above, and this reports the ignore rather than letting a knob
 * quietly stop working. */
static int gpu_split_layer_override(int n_die)
{
    const char *env = getenv("QWEN35_GPU_SPLIT_LAYER");
    if (!env)
        return 0;
    int v = atoi(env);
    if (v <= 0)
        return 0;
    if (n_die > 2) {
        fprintf(stderr, "qwen35moe: QWEN35_GPU_SPLIT_LAYER=%d ignored — it "
                "names ONE boundary and this is a %d-die split (%d "
                "boundaries); layers divide evenly instead\n",
                v, n_die, n_die - 1);
        return 0;
    }
    return v;
}

/* Plan the layer split and the die ids it runs on, onto `g`.
 *
 * Explicit placement (pn_load_on) is authoritative: N placed dies mean an
 * N-way split across exactly those dies, with the final norm + LM head on the
 * last. That is the whole point — a placement of three dies used to bind three
 * and upload a two-way split, so the third sat empty while the first two took
 * half the model each and the second one OOM'd.
 *
 * With no placement at all the legacy single-model serve path applies
 * unchanged: QWEN35_GPU_SPLIT_LAYER (default GPU_DIE_SPLIT_LAYER) picks the
 * boundary, a boundary below n_layer implies two dies, and QWEN35_GPU_DIE_A/B
 * name them. That is what a multi-die service unit would use.
 *
 * Returns 0 on success, -1 to refuse. A width this build cannot run is a
 * REFUSAL, never a narrower split: silently splitting a 3-die placement two
 * ways is the bug being fixed, and doing it "safely" would be the same bug. */
static int gpu_plan_dies(const pn_qwen35moe_model *model, int n_layer,
                         pn_qwen35moe_gpu *g)
{
    const int *placed = NULL;
    int n_placed = pn_qwen35moe_model_dies(model, &placed);

    if (n_placed > PN_DIE_LAYER_SPLIT_MAX_DIES) {
        fprintf(stderr, "qwen35moe: REFUSE — placed on %d dies but this build "
                "splits at most %d ways (PN_DIE_LAYER_SPLIT_MAX_DIES)\n",
                n_placed, PN_DIE_LAYER_SPLIT_MAX_DIES);
        return -1;
    }
    if (n_placed > n_layer) {
        fprintf(stderr, "qwen35moe: REFUSE — placed on %d dies but the model "
                "has only %d layers; a die with no layers is a mis-split, not "
                "a narrower one\n", n_placed, n_layer);
        return -1;
    }

    int n_die;
    if (n_placed >= 1 && placed) {
        n_die = n_placed;
        for (int i = 0; i < n_die; i++)
            g->die[i] = placed[i];
        if (getenv("QWEN35_GPU_DIE_A") || getenv("QWEN35_GPU_DIE_B"))
            fprintf(stderr, "qwen35moe: explicit die placement (%d dies) "
                    "supersedes QWEN35_GPU_DIE_A/B\n", n_die);
    } else if (getenv("QWEN35_GPU_DIES") && *getenv("QWEN35_GPU_DIES")) {
        /* No placement, but an explicit die list (tests and benches): an
         * N-way split across exactly these dies, the same as a placement of
         * them. Boundaries come from the planner (PN_DIE_SPLIT_LAYERS
         * honoured). A malformed list refuses rather than falling back. */
        const char *p = getenv("QWEN35_GPU_DIES");
        n_die = 0;
        while (*p) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p || v < 0 || n_die >= PN_DIE_LAYER_SPLIT_MAX_DIES ||
                (*end && *end != ',')) {
                fprintf(stderr, "qwen35moe: REFUSE — QWEN35_GPU_DIES='%s' is "
                        "not a die list\n", getenv("QWEN35_GPU_DIES"));
                return -1;
            }
            g->die[n_die++] = (int)v;
            p = *end ? end + 1 : end;
        }
        if (n_die < 1 || n_die > n_layer) {
            fprintf(stderr, "qwen35moe: REFUSE — QWEN35_GPU_DIES names %d dies "
                    "for %d layers\n", n_die, n_layer);
            return -1;
        }
    } else {
        /* No placement: the legacy env/compile-time regime. */
        int boundary = GPU_DIE_SPLIT_LAYER;
        const char *env = getenv("QWEN35_GPU_SPLIT_LAYER");
        if (env) { int v = atoi(env); if (v > 0) boundary = v; }
        if (boundary > n_layer)
            boundary = n_layer;
        n_die = (boundary < n_layer) ? 2 : 1;
        g->die[0] = GPU_DIE_A_DEFAULT;
        g->die[1] = GPU_DIE_B_DEFAULT;
        const char *ea = getenv("QWEN35_GPU_DIE_A");
        const char *eb = getenv("QWEN35_GPU_DIE_B");
        if (ea) g->die[0] = atoi(ea);
        if (eb) g->die[1] = atoi(eb);
    }

    /* A 1-die placement is authoritative — the boundary knob cannot carve it
     * up — which pn_plan_die_layer_split enforces by ignoring the override at
     * every width but 2. */
    pn_die_layer_split plan;
    if (pn_plan_die_layer_split(n_layer, n_die,
                                gpu_split_layer_override(n_die), &plan) != 0) {
        fprintf(stderr, "qwen35moe: REFUSE — cannot split %d layers across %d "
                "dies\n", n_layer, n_die);
        return -1;
    }
    for (int i = 0; i < n_die; i++) {
        g->layer_lo[i] = plan.layer_lo[i];
        g->layer_hi[i] = plan.layer_hi[i];
    }
    g->n_die = n_die;
    g->head_die = g->die[n_die - 1];
    g->split_layer = g->layer_hi[0];
    return 0;
}

/* Hand the residual stream across the boundary between die `i` and die `i+1`:
 * `bytes` of token-major [positions][n_embd] f32 out of `src` (on die i) into
 * `dst` (on die i+1).
 *
 * This is the whole cost of splitting a model: ONE contiguous activation copy
 * per boundary per step. An N-die split pays it N-1 times, which is why more
 * dies buys CAPACITY and not speed — nothing about the layer maths gets
 * cheaper, and each hop adds a host sync plus a transfer.
 *
 * Device-to-device across the peer link where the pair has it (on a K80 that
 * is the on-card PLX switch), else staged through the host buffer the caller
 * supplies. Both cost the same bytes; the peer path saves a round trip through
 * host memory and the PCIe hop in each direction. `src`'s die is synced first
 * either way — the copy reads what the layer loop just wrote.
 *
 * Split in two because the legacy barswap regime holds ONE die in its MMIO
 * slot at a time: the D2H must happen while the SENDING die is still selected
 * and the H2D after the RECEIVING one is, and collapsing them into a single
 * call put a sync + a read of the sending die's memory on the wrong side of
 * the swap. Phase 1 is everything that must happen before the switch. */
static void gpu_handoff_export(pn_qwen35moe_gpu *g, int from, fc_buffer *src,
                               void *host_stage, size_t bytes)
{
    fc_sync_ctx(g->ctx[from]);
    if (!g->peer_ok[from])
        fc_buffer_to_cpu_n(src, host_stage, bytes);
}

/* Phase 2: run this once die `from + 1` is the selected one and its scratch
 * exists, so `dst` is a real buffer on the receiving die. A peer copy that
 * fails here still has the host route available — peer_ok is only ever set
 * when both dies are live (never under barswap), so reading `src` from the
 * host at this point is legal. */
static void gpu_handoff_import(pn_qwen35moe_gpu *g, int from, fc_buffer *src,
                               fc_buffer *dst, void *host_stage, size_t bytes)
{
    if (g->peer_ok[from]) {
        if (fc_buffer_copy_peer(dst, src, bytes) == 0) {
            /* Sync the RECEIVING die, not just the sending one. cuMemcpyPeer
             * lands on its own context's stream, so without this the next
             * die's first layer can read `dst` before the copy has committed —
             * the residual stream is then whatever was in the buffer last
             * step, and the model emits fluent-looking garbage rather than
             * failing. This is the same sync the T3 vocab-split copy takes
             * after its x_norm peer copy, for the same reason. */
            fc_sync_ctx(g->ctx[from + 1]);
            return;
        }
        fc_buffer_to_cpu_n(src, host_stage, bytes);
    }
    fc_buffer_from_cpu_n(dst, host_stage, bytes);
}

/* The index of the die whose context is `ctx`, or -1 if it is not one of ours.
 * The per-die side tables (scratch, pos buffers, deferred KV mirrors, segment
 * graphs, async events) are all keyed by this. It used to be the expression
 * `ctx == g->ctx_b ? 1 : 0`, which silently answered "die 0" for every middle
 * die of a wider split. */
static int gpu_die_index_of_ctx(const pn_qwen35moe_gpu *g,
                                const fc_cuda_ctx *ctx)
{
    for (int i = 0; i < g->n_die; i++)
        if (g->ctx[i] == ctx)
            return i;
    return -1;
}

/* The CUDA ordinal / barswap die id behind `ctx`. Falls back to die 0 for an
 * unknown context, matching the old expression's behaviour for the only case
 * it could face (a single-die model's sole context). */
static int gpu_die_of_ctx(const pn_qwen35moe_gpu *g, const fc_cuda_ctx *ctx)
{
    int i = gpu_die_index_of_ctx(g, ctx);
    return g->die[i >= 0 ? i : 0];
}

/* Bring nvidia up after a die is acquired (blacklisted; each submodule needs
 * --ignore-install). Mirrors compute_persist / the test harnesses. Returns 0 on
 * success; harmless to call once per acquired die. */
static int gpu_load_nvidia(void)
{
    if (system("modprobe --ignore-install nvidia 2>/dev/null") != 0)
        return -1;
    if (system("modprobe --ignore-install nvidia_uvm 2>/dev/null") != 0)
        return -1;
    return 0;
}

/* Swap which die is in the barswap MMIO slot. No-op in single-die mode (bs
 * NULL) or when the requested die is already active. Returns 0 on success. */
static int gpu_select_die(pn_qwen35moe_gpu *g, int die)
{
    if (!g->bs)
        return 0; /* single-die: caller acquired the die externally */
    if (g->active_die == die)
        return 0;
    if (!bs_acquire(g->bs, die)) {
        fprintf(stderr, "qwen35moe: bs_acquire(die %d) failed\n", die);
        return -1;
    }
    g->active_die = die;
    return 0;
}

/* ============================================================
 * Per-layer debug dump — IDENTICAL tags/format to the CPU path
 * (qwen35moe_forward.c qwen35moe_dbg) so GPU output diffs against the CPU dump
 * line-for-line. Gated by QWEN35_DEBUG. Takes a host buffer; GPU callers read
 * back the relevant fc_buffer first (the dump is a validation aid, not a hot
 * path, so the extra D2H copy is acceptable).
 * ============================================================ */

/* ============================================================
 * Lightweight wall-clock profiler (QWEN35_PROF=1). Accumulates per-token time
 * by matvec quant type (kernel vs host-fallback) and by forward phase, printed
 * once per token. Adds fc_sync around timed ops for attribution, so enable
 * only when profiling.
 * ============================================================ */

#include <time.h>

static double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int prof_on(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_PROF") ? 1 : 0;
    return v;
}

#define PROF_MAX_TYPE 40
static double prof_mv_time[PROF_MAX_TYPE];  /* by quant type */
static long   prof_mv_cnt[PROF_MAX_TYPE];
static double prof_mv_fallback;             /* total host-fallback matvec time */
/* phases: 0=ssm_state_rt 1=attn_host 2=moe_total 3=lm_head 4=logits_d2h
 * 5=ssm_block 6=attn_block 7=moe_gateup 8=moe_down 9=moe_shexp */
static double prof_phase[10];

static void prof_reset(void)
{
    memset(prof_mv_time, 0, sizeof(prof_mv_time));
    memset(prof_mv_cnt, 0, sizeof(prof_mv_cnt));
    memset(prof_phase, 0, sizeof(prof_phase));
    prof_mv_fallback = 0.0;
}

static void prof_report(double total)
{
    fprintf(stderr, "PROF token total=%.3fs | matvec by type:", total);
    for (int t = 0; t < PROF_MAX_TYPE; t++)
        if (prof_mv_cnt[t])
            fprintf(stderr, " t%d=%.3fs/%ld", t, prof_mv_time[t], prof_mv_cnt[t]);
    fprintf(stderr, " | fallback=%.3fs\n", prof_mv_fallback);
    fprintf(stderr,
            "PROF phases: ssm_state_rt=%.3f attn_host=%.3f moe=%.3f head=%.3f "
            "logits_d2h=%.3f ssm_blk=%.3f attn_blk=%.3f moe_gateup=%.3f "
            "moe_down=%.3f moe_shexp=%.3f\n",
            prof_phase[0], prof_phase[1], prof_phase[2], prof_phase[3],
            prof_phase[4], prof_phase[5], prof_phase[6], prof_phase[7],
            prof_phase[8], prof_phase[9]);
}

/* Batch-path (prefill) phase profiler, gated by QWEN35_PROF like the
 * single-token profiler above: per-call values reset each batch call,
 * cumulative values span a whole chunked prefill. bprof_lap syncs the die
 * so each span's wall time is attributed to the phase that issued it. */
enum {
    BPROF_ATTN_MV = 0,   /* attn q/k/v + out projections (GPU) */
    BPROF_ATTN_SPLIT,    /* host q/gate split + its D2H/H2D (host path) */
    BPROF_ATTN_ROPE,     /* qk-rmsnorm + partial rope (+ device q/z split) */
    BPROF_ATTN_FLASH,    /* on-die kv sync/append + flash kernel + out gate */
    BPROF_ATTN_XFER,     /* attention D2H/H2D (kv mirror / host-softmax IO) */
    BPROF_ATTN_HOST,     /* HOST causal GQA softmax + kv append + out gate */
    BPROF_SSM,           /* whole SSM block (projections + scans) */
    BPROF_MOE,           /* whole MoE block (router + experts + shexp) */
    BPROF_COUNT
};
static const char *bprof_name[BPROF_COUNT] = {
    "attn_mv", "attn_split", "attn_rope", "attn_flash", "attn_xfer",
    "attn_host", "ssm", "moe"
};
static double bprof_call[BPROF_COUNT];
static double bprof_cum[BPROF_COUNT];
static double bprof_cum_total;
static long   bprof_cum_ntok;

static void bprof_add(int phase, double dt)
{
    bprof_call[phase] += dt;
    bprof_cum[phase] += dt;
}

/* Close the span started at *t0: drain ctx, book the time, restart. */
static void bprof_lap(int phase, fc_cuda_ctx *ctx, double *t0)
{
    if (!prof_on())
        return;
    fc_sync_ctx(ctx);
    double t = prof_now();
    bprof_add(phase, t - *t0);
    *t0 = t;
}

static void bprof_report(int n, int pos0, double call_total)
{
    double call_sum = 0.0, cum_sum = 0.0;
    bprof_cum_total += call_total;
    bprof_cum_ntok += n;
    fprintf(stderr, "BPROF call n=%d pos0=%d total=%.3fs |", n, pos0,
            call_total);
    for (int ph = 0; ph < BPROF_COUNT; ph++) {
        fprintf(stderr, " %s=%.3f", bprof_name[ph], bprof_call[ph]);
        call_sum += bprof_call[ph];
        cum_sum += bprof_cum[ph];
    }
    fprintf(stderr, " other=%.3f\n", call_total - call_sum);
    fprintf(stderr, "BPROF cum ntok=%ld total=%.3fs |", bprof_cum_ntok,
            bprof_cum_total);
    for (int ph = 0; ph < BPROF_COUNT; ph++)
        fprintf(stderr, " %s=%.3f(%.1f%%)", bprof_name[ph], bprof_cum[ph],
                100.0 * bprof_cum[ph] /
                    (bprof_cum_total > 0 ? bprof_cum_total : 1.0));
    fprintf(stderr, " other=%.3f\n", bprof_cum_total - cum_sum);
}

/* ============================================================
 * Slots-path (batched decode) phase profiler — QWEN35_SPROF=1 (M2).
 *
 * Sync-bracketed like bprof: each lap drains the issuing die so wall time is
 * attributed to the phase that queued the work. Instrumentation only — the
 * launch sequence and math are untouched; with the env unset every hook is a
 * single predictable branch. The extra syncs inflate the absolute step time,
 * so read the %% split from SPROF and the true step time from an unprofiled
 * run. sprof_active is set only inside pn_qwen35moe_decode_slots_tokens_cuda,
 * so the shared gpu_moe_block_batch laps stay dead on the prefill path.
 * ============================================================ */
enum {
    SPROF_DENSE = 0,     /* dense/shared matvecs (gpu_matvec_nx) in ssm/attn */
    SPROF_MOE_GATEUP,    /* routed experts: grouped gate+up SwiGLU */
    SPROF_MOE_DOWN,      /* routed experts: down proj + weighted accumulate */
    SPROF_MOE_SHEXP,     /* router (+topk) + shared expert + combine */
    SPROF_MOE_ROWFB,     /* M2b: per-row (non-grouped) MoE fallback loop */
    SPROF_ATTN_HOST,     /* host GQA softmax + q/gate split + PCIe + syncs */
    SPROF_ATTN_FLASH,    /* M2b: on-die slot flash (qk-norm/rope/append/kernel) */
    SPROF_ATTN_MIRROR,   /* M2b: host KV-mirror D2H + per-slot store */
    SPROF_SSM,           /* slot conv1d + recurrence + gated norm + glue */
    SPROF_LM_HEAD,       /* final norm + ONE batched LM-head GEMM */
    SPROF_LOGITS,        /* per-slot D2D + sync + top-k fetch loop */
    SPROF_XFER,          /* embed H2D + die-boundary D2H/H2D hand-off */
    SPROF_NCOUNT
};
static const char *sprof_pname[SPROF_NCOUNT] = {
    "dense_nx", "moe_gateup", "moe_down", "shexp_router", "moe_rowfb",
    "attn_host", "attn_flash", "attn_mirror",
    "ssm_slots", "lm_head", "logits_loop", "embed_xfer"
};
static double sprof_step_ph[SPROF_NCOUNT];   /* current step */
static double sprof_cum_ph[SPROF_NCOUNT];    /* across steps */
static double sprof_cum_total;
static long   sprof_nsteps;
static int    sprof_active;

static int sprof_on(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_SPROF") ? 1 : 0;
    return v;
}

/* QWEN35_PROF_SUB (with QWEN35_PROF): sub-split BPROF's two big buckets on the
 * batched PREFILL path — `moe=` into router+shared vs the two routed-expert
 * kernels (by borrowing the sprof spans below, since gpu_moe_block_batch is
 * shared with the slots path), and `ssm=` into projections / conv1d scan /
 * qk-norm / recurrence scan / out. Costs the same extra syncs bprof does, and
 * nests INSIDE the bprof spans, so it inflates the totals further still: read
 * the split, never the absolute time. */
static int prof_sub_on(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_PROF_SUB") ? 1 : 0;
    return v;
}

/* SSM-block sub-phases (QWEN35_PROF_SUB). Kept out of BPROF_COUNT so the
 * existing BPROF line and its `other=` residual are unchanged; ssm_lap does
 * not touch the enclosing BPROF_SSM span's clock, so ssm= stays the whole
 * block and these five partition it. */
enum {
    SSMP_PROJ = 0,   /* 4 projections + sigmoid + decay gate */
    SSMP_CONV,       /* fc_ssm_conv1d_silu_seq — sequential scan over n */
    SSMP_QKNORM,     /* fc_ssm_qknorm_gqa_nx */
    SSMP_RECUR,      /* fc_ssm_recurrence_seq — sequential scan over n */
    SSMP_OUT,        /* gated norm + out proj + residual add */
    SSMP_COUNT
};
static const char *ssmp_name[SSMP_COUNT] = {
    "proj", "conv_seq", "qknorm", "recur_seq", "out"
};
static double ssmp_call[SSMP_COUNT];
static double ssmp_cum[SSMP_COUNT];

static void ssm_lap(int phase, fc_cuda_ctx *ctx, double *t0)
{
    if (!prof_sub_on())
        return;
    fc_sync_ctx(ctx);
    double t = prof_now();
    ssmp_call[phase] += t - *t0;
    ssmp_cum[phase] += t - *t0;
    *t0 = t;
}

/* ============================================================
 * DECODE-slots sub-spans — QWEN35_SPROF=1 *and* QWEN35_PROF_SUB=1.
 *
 * `ssm_slots` and `attn_flash` are each ~6.7 ms of the batched decode step —
 * individually larger than the LM head — and neither had ever been split
 * internally: ssm_lap above only ever fired on the batched PREFILL path. These
 * two tables split them on the path the fleet actually decodes on, so the
 * fusion question ("is the chain's cost the GDN recurrence itself, or the glue
 * between the launches?") is answered with a measurement instead of a guess.
 *
 * Gating is deliberately the CONJUNCTION of both envs. The laps sync, and a
 * sync mid-layer breaks CUDA-graph capture — sprof already stands the slots
 * path down from graphs when it is active, so riding on sprof_active keeps
 * every new sync inside a configuration that had already given graphs up.
 * With QWEN35_PROF_SUB alone (the prefill split) the decode path is untouched.
 *
 * READ THE SPLIT, NOT THE ABSOLUTE. Two distortions run the same direction:
 * the extra syncs serialise work the real path overlaps, and graphs are off,
 * so per-launch overhead is charged at its un-amortised price. That makes the
 * glue share an UPPER BOUND on what fusion could recover under graphs.
 * ============================================================ */
enum {
    SSMD_DECAY = 0,  /* fc_sigmoid + fc_ssm_decay_gate_nx (unfused chain only) */
    SSMD_CONV,       /* fc_ssm_conv1d_silu_slots */
    SSMD_QKNORM,     /* fc_ssm_qknorm_gqa_nx */
    SSMD_RECUR,      /* fc_ssm_recurrence_slots, or the fused gdn_step */
    SSMD_GNORM,      /* fc_ssm_gated_norm */
    SSMD_COUNT
};
static const char *ssmd_name[SSMD_COUNT] = {
    "decay_gate", "conv1d", "qknorm", "recur", "gated_norm"
};
/* Launches per lap, so a ms/launch figure separates "this kernel is slow" from
 * "this lap holds two launches". Kept beside the names so they cannot drift. */
static const int ssmd_launches[SSMD_COUNT] = { 2, 1, 1, 1, 1 };

enum {
    ATTD_QZSPLIT = 0, /* fc_attn_qz_split */
    ATTD_QKNORM,      /* fc_qk_rms_norm x2 (q, k) */
    ATTD_ROPE,        /* fc_rope_neox_partial_pos x2 (q, k) */
    ATTD_KVAPPEND,    /* fc_attn_kv_append_slots[_q4] */
    ATTD_FLASH,       /* fc_attn_flash_decode_slots[_q4] — split-K + combine */
    ATTD_OUTGATE,     /* fc_sigmoid + fc_mul */
    ATTD_COUNT
};
static const char *attd_name[ATTD_COUNT] = {
    "qz_split", "qknorm", "rope", "kv_append", "flash", "out_gate"
};
/* -1 = not a fixed count: the flash lap dispatches launch_splits split-K
 * launches plus a combine, so it varies with KV depth. */
static const int attd_launches[ATTD_COUNT] = { 1, 2, 2, 1, -1, 2 };

/* shexp_router: the router and the shared expert around the two routed-expert
 * kernels. Ten launches per MoE layer, forty MoE layers -- the pool the fusion
 * plan calls latency-bound, so it gets the same per-launch split. */
enum {
    MOED_ROUTER = 0,  /* router matvec (ffn_gate_inp) */
    MOED_TOPK,        /* fc_moe_router_topk_nx */
    MOED_SH_GATEUP,   /* ffn_gate_shexp + ffn_up_shexp matvecs */
    MOED_SH_SILU,     /* fc_moe_silu_mul */
    MOED_SH_DOWN,     /* ffn_down_shexp matvec */
    MOED_SH_COMBINE,  /* fc_moe_shexp_dot_gate_nx + fc_add x2 */
    MOED_COUNT
};
static const char *moed_name[MOED_COUNT] = {
    "router_mv", "topk", "shexp_gateup_mv", "silu_mul", "shexp_down_mv", "combine"
};
static const int moed_launches[MOED_COUNT] = { 1, 1, 2, 1, 1, 3 };

static double ssmd_cum[SSMD_COUNT], attd_cum[ATTD_COUNT], moed_cum[MOED_COUNT];
static long   ssmd_laps[SSMD_COUNT], attd_laps[ATTD_COUNT], moed_laps[MOED_COUNT];

/* Both decode tables ride on sprof_active so they can never fire on a
 * graph-captured step. */
static int sub_decode_on(void)
{
    return sprof_active && prof_sub_on();
}

static void sub_begin(fc_cuda_ctx *ctx, double *t0)
{
    if (!sub_decode_on())
        return;
    fc_sync_ctx(ctx);
    *t0 = prof_now();
}

static void sub_lap(double *cum, long *laps, fc_cuda_ctx *ctx, double *t0)
{
    if (!sub_decode_on())
        return;
    fc_sync_ctx(ctx);
    double t = prof_now();
    *cum += t - *t0;
    (*laps)++;
    *t0 = t;
}

/* One table per phase: total ms/step, share of the phase, laps per step (= the
 * layer count the phase runs over) and ms per launch. `laps/step` is the number
 * that turns a share into a fusion argument: it says how many times the chain
 * pays whatever per-launch cost the glue rows are measuring. */
static void sub_decode_table(const char *tag, int n, const char *const *name,
                             const double *cum, const long *laps,
                             const int *launches, long nsteps)
{
    double tot = 0.0;
    for (int i = 0; i < n; i++)
        tot += cum[i];
    if (tot <= 0.0 || nsteps <= 0)
        return;
    fprintf(stderr, "SUBDEC %s steps=%ld phase=%.2fms/step |", tag, nsteps,
            1e3 * tot / (double)nsteps);
    for (int i = 0; i < n; i++) {
        double per_step_ms = 1e3 * cum[i] / (double)nsteps;
        double lps = (double)laps[i] / (double)nsteps;
        fprintf(stderr, " %s=%.2fms(%.0f%%,laps/step=%.0f", name[i],
                per_step_ms, 100.0 * cum[i] / tot, lps);
        if (launches[i] > 0 && laps[i] > 0)
            fprintf(stderr, ",%.3fms/launch",
                    per_step_ms / (lps * (double)launches[i]));
        fprintf(stderr, ")");
    }
    fprintf(stderr, "\n");
}

static void sub_decode_report(void)
{
    if (!sub_decode_on())
        return;
    sub_decode_table("ssm_slots", SSMD_COUNT, ssmd_name, ssmd_cum, ssmd_laps,
                     ssmd_launches, sprof_nsteps);
    sub_decode_table("attn_flash", ATTD_COUNT, attd_name, attd_cum, attd_laps,
                     attd_launches, sprof_nsteps);
    sub_decode_table("shexp_router", MOED_COUNT, moed_name, moed_cum, moed_laps,
                     moed_launches, sprof_nsteps);
}

/* Open a span: drain the die (so prior phases don't leak in) + timestamp. */
static void sprof_begin(fc_cuda_ctx *ctx, double *t0)
{
    if (!sprof_active)
        return;
    if (ctx)
        fc_sync_ctx(ctx);
    *t0 = prof_now();
}

/* Close the span started at *t0 into `phase`; restart the clock. */
static void sprof_lap(int phase, fc_cuda_ctx *ctx, double *t0)
{
    if (!sprof_active)
        return;
    if (ctx)
        fc_sync_ctx(ctx);
    double t = prof_now();
    sprof_step_ph[phase] += t - *t0;
    *t0 = t;
}

static void sprof_step_report(int nslots, double step_total)
{
    double sum = 0.0;
    sprof_nsteps++;
    sprof_cum_total += step_total;
    fprintf(stderr, "SPROF step K=%d total=%.1fms |", nslots,
            1e3 * step_total);
    for (int ph = 0; ph < SPROF_NCOUNT; ph++) {
        sprof_cum_ph[ph] += sprof_step_ph[ph];
        sum += sprof_step_ph[ph];
        fprintf(stderr, " %s=%.1f(%.0f%%)", sprof_pname[ph],
                1e3 * sprof_step_ph[ph],
                100.0 * sprof_step_ph[ph] / (step_total > 0 ? step_total : 1));
    }
    fprintf(stderr, " other=%.1f\n", 1e3 * (step_total - sum));
    if (sprof_nsteps % 10 == 0) {
        double csum = 0.0;
        fprintf(stderr, "SPROF cum steps=%ld avg=%.1fms |", sprof_nsteps,
                1e3 * sprof_cum_total / sprof_nsteps);
        for (int ph = 0; ph < SPROF_NCOUNT; ph++) {
            csum += sprof_cum_ph[ph];
            fprintf(stderr, " %s=%.1f%%", sprof_pname[ph],
                    100.0 * sprof_cum_ph[ph] /
                        (sprof_cum_total > 0 ? sprof_cum_total : 1));
        }
        fprintf(stderr, " other=%.1f%%\n",
                100.0 * (sprof_cum_total - csum) /
                    (sprof_cum_total > 0 ? sprof_cum_total : 1));
        sub_decode_report();
    }
}

/* ============================================================
 * Slots-path structural diagnostics — QWEN35_SDIAG=1 (M2b, Phase 0).
 *
 * ONE-SHOT facts about the batched decode step that a timing profile cannot
 * show: does the on-die flash slots path engage (and if not, which guard
 * rejected it), what did the per-slot KV ring allocation ask for and did it
 * succeed, which layers are `grouped` vs on the per-row MoE fallback, and
 * whether each SSM layer's slots graph actually captured. Each site latches
 * after its first report, so this is a handful of stderr lines per process,
 * not a per-step cost. Unlike sprof it inserts NO syncs and does NOT disable
 * graph capture or the die pipeline — so the facts it reports are the facts
 * of the REAL (unprofiled) path. Default off => production untouched.
 * ============================================================ */
static int sdiag_on(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_SDIAG") ? 1 : 0;
    return v;
}

static int qwen35moe_dbg_enabled(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_DEBUG") ? 1 : 0;
    return v;
}

static void qwen35moe_dbg(const char *tag, int layer, const float *x, int n)
{
    if (!qwen35moe_dbg_enabled())
        return;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    fprintf(stderr, "PN %-22s L%-2d [%12.6f %12.6f %12.6f ... %12.6f %12.6f "
            "%12.6f] sum=%.6f\n",
            tag, layer, x[0], x[1], x[2], x[n - 3], x[n - 2], x[n - 1], sum);
}

/* Read an fc_buffer's first n floats back to host and dump with the CPU tag.
 * ctx is the buffer's owning die context (so the sync hits the right stream). */
static void qwen35moe_dbg_gpu(fc_cuda_ctx *ctx, const char *tag, int layer,
                              fc_buffer *buf, int n, float *scratch)
{
    if (!qwen35moe_dbg_enabled())
        return;
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu(buf, scratch);
    qwen35moe_dbg(tag, layer, scratch, n);
}

/* QWEN35_BV_TRACE=1 (diagnostic, default off): after every layer, print an
 * FNV-1a hash of each row's residual stream, keyed by absolute position, from
 * the batched path (a verify) and the decode-slots path, so the first layer
 * where a verify row and the plain step at the same position diverge can be
 * read off the log ("BVT <path> L<layer> pos=<pos> <hash>"). Syncs per layer,
 * and the attention-step hooks sit inside graph-captured slots bodies: run it
 * with QWEN35_GRAPH=0, at K=1 (slot rows are keyed off pos[0]). Off, every hook
 * is one predictable branch. */
static int bv_trace_on(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_BV_TRACE") ? 1 : 0;
    return v;
}

static int bv_trace_slots_pos0;   /* pos[0] of the current slots step */

static void bv_trace_rows(fc_cuda_ctx *ctx, const char *path, const char *tag,
                          int layer, fc_buffer *buf, int nrows, int c,
                          const int *pos, int pos0)
{
    if (!bv_trace_on())
        return;
    float *h = malloc((size_t)nrows * c * sizeof(float));
    if (!h)
        return;
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(buf, h, (size_t)nrows * c * sizeof(float));
    for (int r = 0; r < nrows; r++) {
        uint32_t x = 2166136261u;
        const unsigned char *p = (const unsigned char *)(h + (size_t)r * c);
        for (size_t i = 0; i < (size_t)c * sizeof(float); i++)
            x = (x ^ p[i]) * 16777619u;
        fprintf(stderr, "BVT %s %s L%d pos=%d %08x\n", path, tag, layer,
                pos ? pos[r] : pos0 + r, x);
    }
    free(h);
}

/* 1/sqrt(n) for the attention (n = head_dim) and SSM recurrence (n = S)
 * scales — ONE out-of-line definition for every path. Under -ffast-math gcc
 * lowers 1.0f/sqrtf(x) to an approximate rsqrt + Newton step (0x3d7fffff at
 * 256, not 0.0625) or to an exact sqrt + divide, decided per call site and by
 * inlining: an unrelated edit to a caller has flipped one site, which put the
 * prefill and the decode step 1 ulp of scale apart (prefix-cache GATE 1 fail),
 * and the speculative verify must match the decode step byte for byte. Every
 * site reads this function, so all paths share the same bits — the rsqrt form
 * the decode and prefill paths had been running. */
__attribute__((noinline)) static float qw_scale_inv_sqrt(int n)
{
    return 1.0f / sqrtf((float)n);
}

/* ============================================================
 * Upload helpers
 * ============================================================ */

/* Upload one quantized weight matrix to a persistent fc_buffer.
 * Quant types with a repacked device layout (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 — see
 * fc_repack_weight in foundry fc_cuda.h) are converted ONCE here into the
 * 4-byte-aligned, scale-pre-unpacked form the matvec/MoE kernels read;
 * g->row_stride becomes the repacked stride. Values are bit-identical, only
 * the layout changes. Other types (F32 rows for the router/alpha/beta) upload
 * raw. The CPU qweight keeps pointing at the mmap'd GGUF blocks for the host
 * fallback. Returns 0 on success. */
static int vocab_split_on(void);   /* fwd decl (defined near the decode path) */

static int gpu_weight_upload(fc_cuda_ctx *ctx, gpu_weight *g, const pn_qweight *w)
{
    g->cpu = w;
    g->type = w->type;
    g->rows = w->rows;
    g->cols = w->cols;
    g->row_stride = (int)w->row_stride;
    g->bytes = NULL;
    g->ctx = ctx;
    if (!w->data || w->rows <= 0)
        return 0; /* absent/empty tensor — host fallback will handle it */
    size_t rstride = fc_repack_row_bytes((int)w->type, w->cols);
    if (rstride) {
        size_t rtotal = (size_t)w->rows * rstride;
        unsigned char *tmp = malloc(rtotal);
        if (!tmp)
            return -1;
        if (fc_repack_weight((int)w->type, w->data, tmp, w->rows, w->cols) != 0) {
            free(tmp);
            return -1;
        }
        g->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, rtotal);
        free(tmp);
        g->row_stride = (int)rstride;
        return g->bytes ? 0 : -1;
    }
    switch ((int)w->type) {
    case GGUF_QUANT_Q3_K:
    case GGUF_QUANT_Q4_K:
    case GGUF_QUANT_Q5_K:
    case GGUF_QUANT_Q6_K:
    case GGUF_QUANT_Q8_0:
        /* Repackable type with an odd cols the kernels couldn't handle
         * anyway — leave on the host (gpu_matvec falls back to pn_qmv). */
        return 0;
    default:
        break;
    }
    size_t total = (size_t)w->rows * w->row_stride;
    g->bytes = fc_buffer_from_ptr_ctx(ctx, w->data, total);
    return g->bytes ? 0 : -1;
}

/* T3 — upload a ROW SLICE [row_start, row_start+row_count) of a weight to ctx's
 * die. Same repack/copy path as gpu_weight_upload but over a contiguous sub-
 * range of the (row-major) source rows: row r of the source lives at
 * w->data + r*w->row_stride, and repack is per-row, so a slice is just a
 * shifted base pointer + a reduced row count. gw->rows becomes row_count and
 * gw->cpu stays the FULL weight (host fallback re-derives the slice from the
 * row offset when it runs — but the split path never uses the host fallback).
 * Used to give each die half the LM-head vocab rows (memory-neutral). */
static int gpu_weight_upload_rows(fc_cuda_ctx *ctx, gpu_weight *gw,
                                  const pn_qweight *w, int row_start,
                                  int row_count)
{
    if (!w || row_start < 0 || row_count <= 0 ||
        row_start + row_count > w->rows)
        return -1;
    gw->cpu = w;
    gw->type = w->type;
    gw->rows = row_count;
    gw->cols = w->cols;
    gw->row_stride = (int)w->row_stride;
    gw->bytes = NULL;
    gw->ctx = ctx;
    if (!w->data)
        return -1;
    const unsigned char *base =
        (const unsigned char *)w->data + (size_t)row_start * w->row_stride;
    size_t rstride = fc_repack_row_bytes((int)w->type, w->cols);
    if (rstride) {
        size_t rtotal = (size_t)row_count * rstride;
        unsigned char *tmp = malloc(rtotal);
        if (!tmp)
            return -1;
        if (fc_repack_weight((int)w->type, base, tmp, row_count, w->cols) != 0) {
            free(tmp);
            return -1;
        }
        gw->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, rtotal);
        free(tmp);
        gw->row_stride = (int)rstride;
        return gw->bytes ? 0 : -1;
    }
    /* Non-repacked types: only F32 (and raw quants with an odd cols) reach
     * here. Odd-cols quants have no GPU kernel — the split path requires a
     * GPU-resident head, so refuse rather than silently host-fall-back. */
    switch ((int)w->type) {
    case GGUF_QUANT_Q3_K:
    case GGUF_QUANT_Q4_K:
    case GGUF_QUANT_Q5_K:
    case GGUF_QUANT_Q6_K:
    case GGUF_QUANT_Q8_0:
        return -1;
    default:
        break;
    }
    size_t total = (size_t)row_count * w->row_stride;
    gw->bytes = fc_buffer_from_ptr_ctx(ctx, base, total);
    return gw->bytes ? 0 : -1;
}

/* Upload one F32 vector to a persistent fc_buffer on ctx's die. NULL src ->
 * empty (allowed for optional norms). Returns 0 on success. */
static int gpu_vec_upload(fc_cuda_ctx *ctx, gpu_vec *g, const float *src, int n)
{
    g->n = n;
    g->buf = NULL;
    g->ctx = ctx;
    if (!src || n <= 0)
        return 0;
    g->buf = fc_buffer_from_ptr_ctx(ctx, src, (size_t)n * sizeof(float));
    return g->buf ? 0 : -1;
}

static void gpu_weight_free(gpu_weight *g)
{
    if (g && g->bytes)
        fc_buffer_free(g->bytes);
}

static void gpu_vec_free(gpu_vec *g)
{
    if (g && g->buf)
        fc_buffer_free(g->buf);
}

/* Upload one MERGED expert tensor for the grouped kernels. Every type but Q8_0
 * uses the dense layout (gpu_weight_upload); Q8_0 experts use foundry's 288-B
 * expert super-blocks (fc_moe_repack_weight), which the grouped q80 kernels
 * read and the dense q80 matvec does not. Same bytes per row as the dense
 * repack (36 per 32 weights), so VRAM is unchanged. The temporary host copy
 * is one merged tensor (~300 MiB at Qwen3.6's shape), freed before the next. */
static int gpu_expert_upload(fc_cuda_ctx *ctx, gpu_weight *g,
                             const pn_qweight *w)
{
    if ((int)w->type != GGUF_QUANT_Q8_0)
        return gpu_weight_upload(ctx, g, w);
    size_t rstride = fc_moe_repack_row_bytes((int)w->type, w->cols);
    g->cpu = w;
    g->type = w->type;
    g->rows = w->rows;
    g->cols = w->cols;
    g->row_stride = (int)rstride;
    g->bytes = NULL;
    g->ctx = ctx;
    if (!w->data || w->rows <= 0 || rstride == 0)
        return -1;
    size_t total = (size_t)w->rows * rstride;
    unsigned char *tmp = malloc(total);
    if (!tmp)
        return -1;
    if (fc_moe_repack_weight((int)w->type, w->data, tmp, w->rows, w->cols) != 0) {
        free(tmp);
        return -1;
    }
    g->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, total);
    free(tmp);
    return g->bytes ? 0 : -1;
}

/* Q8_0 experts need cols % 256 == 0 for the expert super-blocks; every other
 * type keeps the grouped path's existing rules. */
static int gpu_expert_layout_ok(int type, int cols)
{
    return type != GGUF_QUANT_Q8_0 || fc_moe_repack_row_bytes(type, cols) != 0;
}

/* Upload one layer's shared-expert MoE weights. When dense (Bonsai), only the
 * three FFN projections (loaded into the shexp slots) exist — no router, no
 * routed experts, no scalar gate — so skip all of those and upload just the
 * SwiGLU triplet. */
static int gpu_moe_upload(fc_cuda_ctx *ctx, gpu_moe *g,
                          const pn_qwen35moe_moe *moe, int n_expert, int n_embd,
                          int dense)
{
    g->n_expert = n_expert;
    if (dense) {
        g->grouped = 0;
        g->gate_exps = g->up_exps = g->down_exps = NULL;
        if (gpu_weight_upload(ctx, &g->ffn_gate_shexp, &moe->ffn_gate_shexp) < 0)
            return -1;
        if (gpu_weight_upload(ctx, &g->ffn_up_shexp, &moe->ffn_up_shexp) < 0)
            return -1;
        if (gpu_weight_upload(ctx, &g->ffn_down_shexp, &moe->ffn_down_shexp) < 0)
            return -1;
        g->gate_inp_shexp_host = NULL;
        return 0;
    }
    if (gpu_weight_upload(ctx, &g->ffn_gate_inp, &moe->ffn_gate_inp) < 0)
        return -1;

    /* Grouped fast path: gate/up Q3_K|Q4_K + down Q3_K|Q4_K, and every expert a
     * contiguous slice of the merged 3D tensor (the GGUF loader slices them
     * that way — verify anyway, the kernels index e * rows * row_stride).
     *
     * gate and up must be the SAME type: one kernel decodes both matrices in
     * one warp body, so there is no per-matrix dispatch to give them. down is
     * dispatched separately and may differ (Q3_K|Q4_K|Q6_K) — a Q4_K_M mixes
     * Q4_K and Q6_K down across layers just as the Q3_K_M mini mixes Q3_K and
     * Q4_K.
     *
     * The *_quant_available_ctx probes are the second half of the gate — the
     * Q4_K gate+up and Q6_K down kernels are optional PTX symbols, and a
     * foundry fc_cuda_ptx.h regenerated before they existed must drop the model
     * to the per-expert fallback HERE rather than fail on every layer at run
     * time. That fallback is what makes a big Q4_K_M unloadable: it replaces 3
     * merged buffers a layer with 3*n_expert (768) separate ones. */
    int gu_type = (int)moe->gate_exps[0].type;
    int dn_type = (int)moe->down_exps[0].type;
    g->grouped = ((gu_type == GGUF_QUANT_Q3_K || gu_type == GGUF_QUANT_Q4_K ||
                   gu_type == GGUF_QUANT_F16 || gu_type == GGUF_QUANT_BF16 ||
                   gu_type == GGUF_QUANT_Q8_0) &&
                  (int)moe->up_exps[0].type == gu_type &&
                  gpu_expert_layout_ok(gu_type, moe->gate_exps[0].cols) &&
                  gpu_expert_layout_ok(dn_type, moe->down_exps[0].cols) &&
                  fc_moe_gateup_quant_available_ctx(ctx, gu_type) &&
                  fc_moe_down_quant_available_ctx(ctx, dn_type));
    if (g->grouped) {
        const pn_qweight *g0 = &moe->gate_exps[0];
        const pn_qweight *u0 = &moe->up_exps[0];
        const pn_qweight *d0 = &moe->down_exps[0];
        size_t g_stride = (size_t)g0->rows * g0->row_stride;
        size_t d_stride = (size_t)d0->rows * d0->row_stride;
        for (int e = 0; e < n_expert && g->grouped; e++) {
            if (moe->gate_exps[e].data != (const uint8_t *)g0->data + e * g_stride ||
                moe->up_exps[e].data != (const uint8_t *)u0->data + e * g_stride ||
                moe->down_exps[e].data != (const uint8_t *)d0->data + e * d_stride ||
                moe->gate_exps[e].type != g0->type ||
                moe->up_exps[e].type != u0->type ||
                moe->down_exps[e].type != d0->type)
                g->grouped = 0;
        }
    }
    if (g->grouped) {
        /* Upload each merged tensor as one buffer; describe it as a single
         * tall matrix (rows = n_expert * per-expert rows) so the row_stride/
         * type plumbing stays uniform. */
        pn_qweight merged;
        merged = *(&moe->gate_exps[0]);
        merged.rows = n_expert * moe->gate_exps[0].rows;
        if (gpu_expert_upload(ctx, &g->gate_all, &merged) < 0)
            return -1;
        merged = *(&moe->up_exps[0]);
        merged.rows = n_expert * moe->up_exps[0].rows;
        if (gpu_expert_upload(ctx, &g->up_all, &merged) < 0)
            return -1;
        merged = *(&moe->down_exps[0]);
        merged.rows = n_expert * moe->down_exps[0].rows;
        if (gpu_expert_upload(ctx, &g->down_all, &merged) < 0)
            return -1;
    } else {
        g->gate_exps = calloc((size_t)n_expert, sizeof(gpu_weight));
        g->up_exps = calloc((size_t)n_expert, sizeof(gpu_weight));
        g->down_exps = calloc((size_t)n_expert, sizeof(gpu_weight));
        if (!g->gate_exps || !g->up_exps || !g->down_exps)
            return -1;
        for (int e = 0; e < n_expert; e++) {
            if (gpu_weight_upload(ctx, &g->gate_exps[e], &moe->gate_exps[e]) < 0)
                return -1;
            if (gpu_weight_upload(ctx, &g->up_exps[e], &moe->up_exps[e]) < 0)
                return -1;
            if (gpu_weight_upload(ctx, &g->down_exps[e], &moe->down_exps[e]) < 0)
                return -1;
        }
    }
    if (gpu_weight_upload(ctx, &g->ffn_gate_shexp, &moe->ffn_gate_shexp) < 0)
        return -1;
    if (gpu_weight_upload(ctx, &g->ffn_up_shexp, &moe->ffn_up_shexp) < 0)
        return -1;
    if (gpu_weight_upload(ctx, &g->ffn_down_shexp, &moe->ffn_down_shexp) < 0)
        return -1;
    /* The shared-expert scalar gate is a 1-D [n_embd] F32 vector (see the CPU
     * qwen35moe_shared note); upload it as a vec, not a matvec weight. */
    g->gate_inp_shexp_host = (const float *)moe->ffn_gate_inp_shexp.data;
    return gpu_vec_upload(ctx, &g->ffn_gate_inp_shexp,
                          (const float *)moe->ffn_gate_inp_shexp.data, n_embd);
}

static void gpu_moe_free(gpu_moe *g)
{
    gpu_weight_free(&g->ffn_gate_inp);
    gpu_weight_free(&g->gate_all);
    gpu_weight_free(&g->up_all);
    gpu_weight_free(&g->down_all);
    if (g->gate_exps) {
        for (int e = 0; e < g->n_expert; e++) {
            gpu_weight_free(&g->gate_exps[e]);
            gpu_weight_free(&g->up_exps[e]);
            gpu_weight_free(&g->down_exps[e]);
        }
    }
    free(g->gate_exps);
    free(g->up_exps);
    free(g->down_exps);
    gpu_weight_free(&g->ffn_gate_shexp);
    gpu_weight_free(&g->ffn_up_shexp);
    gpu_weight_free(&g->ffn_down_shexp);
    gpu_vec_free(&g->ffn_gate_inp_shexp);
}

/* Upload one SSM layer's weights into slot d, tagged to ctx (its die). */
static int gpu_upload_ssm_layer(fc_cuda_ctx *ctx, gpu_ssm_layer *d,
                                const pn_qwen35moe_ssm_layer *l,
                                const pn_qwen35moe_config *cfg)
{
    int n_embd = cfg->n_embd;
    int conv_dim = cfg->conv_dim;
    int n_v_heads = cfg->ssm_dt_rank;
    d->ctx = ctx;
    if (gpu_vec_upload(ctx, &d->attn_norm, l->attn_norm, n_embd) < 0 ||
        gpu_weight_upload(ctx, &d->attn_qkv, &l->attn_qkv) < 0 ||
        gpu_weight_upload(ctx, &d->attn_gate, &l->attn_gate) < 0 ||
        gpu_vec_upload(ctx, &d->ssm_conv1d, l->ssm_conv1d, conv_dim * cfg->ssm_conv) < 0 ||
        gpu_vec_upload(ctx, &d->ssm_a, l->ssm_a, n_v_heads) < 0 ||
        gpu_weight_upload(ctx, &d->ssm_alpha, &l->ssm_alpha) < 0 ||
        gpu_weight_upload(ctx, &d->ssm_beta, &l->ssm_beta) < 0 ||
        gpu_vec_upload(ctx, &d->ssm_dt, l->ssm_dt, n_v_heads) < 0 ||
        gpu_vec_upload(ctx, &d->ssm_norm, l->ssm_norm, cfg->ssm_state) < 0 ||
        gpu_weight_upload(ctx, &d->ssm_out, &l->ssm_out) < 0 ||
        gpu_vec_upload(ctx, &d->post_attention_norm, l->post_attention_norm, n_embd) < 0 ||
        gpu_moe_upload(ctx, &d->moe, &l->moe, cfg->n_expert, n_embd, cfg->dense_ffn) < 0)
        return -1;
    return 0;
}

/* Upload one attention layer's weights into slot d, tagged to ctx (its die). */
static int gpu_upload_attn_layer(fc_cuda_ctx *ctx, gpu_attn_layer *d,
                                 const pn_qwen35moe_attn_layer *l,
                                 const pn_qwen35moe_config *cfg)
{
    int n_embd = cfg->n_embd;
    d->ctx = ctx;
    if (gpu_vec_upload(ctx, &d->attn_norm, l->attn_norm, n_embd) < 0 ||
        gpu_weight_upload(ctx, &d->attn_q, &l->attn_q) < 0 ||
        gpu_weight_upload(ctx, &d->attn_k, &l->attn_k) < 0 ||
        gpu_weight_upload(ctx, &d->attn_v, &l->attn_v) < 0 ||
        gpu_vec_upload(ctx, &d->attn_q_norm, l->attn_q_norm, cfg->head_dim) < 0 ||
        gpu_vec_upload(ctx, &d->attn_k_norm, l->attn_k_norm, cfg->head_dim) < 0 ||
        gpu_weight_upload(ctx, &d->attn_output, &l->attn_output) < 0 ||
        gpu_vec_upload(ctx, &d->post_attention_norm, l->post_attention_norm, n_embd) < 0 ||
        gpu_moe_upload(ctx, &d->moe, &l->moe, cfg->n_expert, n_embd, cfg->dense_ffn) < 0)
        return -1;
    return 0;
}

/* Phase A.1 (eight_card_serving_plan): once weights are resident in VRAM, the
 * host mmap of the quantized weight bytes is dead for the GPU path — but on the
 * 8 GB host that ~15 GB of page cache is exactly what drains MemAvailable and
 * trips admission. Advise the kernel to drop it. Safe: the mapping is
 * MAP_PRIVATE + PROT_READ, so a rare host-fallback matvec (missing kernel)
 * re-faults the clean pages straight from the file — correct, just slow. The
 * F32 token_embd and per-layer norms are separate heap copies (NOT in this
 * mmap), so they are untouched. Default-ON (the GPU-resident design) since the
 * host-RAM win + decode-neutrality were validated on a K80 host 2026-07-24 (ornith's
 * ~10 GB GGUF Rss collapsed to ~162 MiB, decode bit-correct); opt OUT with
 * QWEN35_DROP_WEIGHTS=0 for the rare host-fallback debug case. */
static void gpu_drop_weight_pages(pn_qwen35moe_model *model)
{
    const char *e = getenv("QWEN35_DROP_WEIGHTS");
    if (e && e[0] == '0')
        return;
    pn_gguf_file *gf = pn_qwen35moe_gguf(model);
    if (!gf || !gf->data || gf->data_size == 0)
        return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0)
        return;
    uintptr_t ds = (uintptr_t)gf->data;
    uintptr_t start = (ds + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1); /* page-align up */
    uintptr_t end = ds + gf->data_size;
    if (end <= start)
        return;
    size_t len = (size_t)(end - start) & ~((size_t)pg - 1); /* whole pages only */
    if (len == 0)
        return;
    if (madvise((void *)start, len, MADV_DONTNEED) == 0)
        fprintf(stderr, "qwen35moe: dropped %llu MiB host weight page cache "
                "(MADV_DONTNEED; weights resident in VRAM)\n",
                (unsigned long long)(len >> 20));
    else
        fprintf(stderr, "qwen35moe: MADV_DONTNEED on weight pages failed "
                "(errno %d); host cache not dropped\n", errno);
}

/* Upload the weights for global layers [layer_lo, layer_hi) onto `ctx`. Each
 * global layer maps to an SSM or attention local slot via the model's index
 * map; the slot is tagged with the context it now lives on, which is what lets
 * every later path resolve a layer's die from the layer itself. */
/* Live load / unload progress (load_progress.h): what every die of the split
 * holds now, from the contexts' own allocation tallies. A no-op unless the
 * loading thread opened a progress slot. */
static void gpu_progress_note_dies(const pn_qwen35moe_gpu *g)
{
    for (int i = 0; i < g->n_die; i++)
        if (g->ctx[i])
            pn_load_progress_set_bytes(g->die[i], fc_ctx_alloc_bytes(g->ctx[i]));
}

/* What die `i` holds beyond its layers once loaded: the LM head on the last die
 * and its share of the KV at the load's window, as admission prices it. */
static uint64_t gpu_progress_die_extra(pn_qwen35moe_model *model, const pn_qwen35moe_gpu *g,
                                       int i)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    int kv_q4 = pn_qwen35moe_model_kv_q4(model);
    uint64_t kv = pn_admit_die_kv_bytes(pn_qwen35moe_kv_bytes_per_token(model, kv_q4 != 0),
                                        kv_q4 != 0, gpu_admit_max_ctx_model(model),
                                        g->layer_hi[i] - g->layer_lo[i], cfg->n_layer);
    return kv + (i == g->n_die - 1 ? gpu_head_weight_bytes(model) : 0);
}

/* The progress totals once die `i` has uploaded layers [lo, done): a die's
 * device bytes per file byte (the repacked layout is larger than the GGUF)
 * measured so far, applied to what is left on this die and to the dies after
 * it. The slot still raises a total a die outgrows (MTP head, scratch). */
static void gpu_progress_project(pn_qwen35moe_model *model, const pn_qwen35moe_gpu *g, int i,
                                 uint64_t start, int done)
{
    pn_gguf_file *gf = pn_qwen35moe_gguf(model);
    uint64_t now = fc_ctx_alloc_bytes(g->ctx[i]);
    uint64_t file = pn_gguf_layer_range_bytes(gf, g->layer_lo[i], done);
    if (file == 0 || now <= start)
        return;
    double ratio = (double)(now - start) / (double)file;
    pn_load_progress_set_total(g->die[i], now + gpu_progress_die_extra(model, g, i) +
        (uint64_t)(ratio * (double)pn_gguf_layer_range_bytes(gf, done, g->layer_hi[i])));
    for (int j = i + 1; j < g->n_die; j++)
        if (g->ctx[j])
            pn_load_progress_set_total(g->die[j], fc_ctx_alloc_bytes(g->ctx[j]) +
                gpu_progress_die_extra(model, g, j) +
                (uint64_t)(ratio * (double)pn_gguf_layer_range_bytes(gf, g->layer_lo[j],
                                                                     g->layer_hi[j])));
}

/* ============================================================
 * QWEN35_TP — tensor-parallel weight upload (spec k80_q38_tp2 §3.2)
 *
 * Rank p of a pair holds, of every layer of its stage: whole output ROWS of
 * every projection (heads 12p.. of attention, kv heads 2p.., the GDN k-heads
 * 8p..8p+7 with the v-heads r*16 + 8p + j that the kernel's interleave
 * vh = r*n_k + kh pairs with them, out-projection rows [2560p, +2560), FFN
 * gate/up rows k with k % 2 == p) and, for FFN down, the Q8_0 parity slab of
 * the inputs k % 2 == p (fc_matvec_q80_par_nx_ctx). Every output is computed
 * whole by the same kernel as the layer split, so the halves are bit-exact; the
 * down partials sum in the kernel's own tree order (foundry fc_cuda.h).
 * GDN rows are ordered so the rank's local v-head lv = r*8 + j is global
 * r*16 + 8p + j: the kernels then run unchanged on the local dims.
 *
 * MoE (spec k80_q36_tp2 §3.2): the grouped Q8_0 expert kernels fuse the routing
 * weights and the sum over the k slots into down with their own lane trees, so
 * no K split of them is exact. Rank p holds rows [p*ff/2, +ff/2) of every
 * expert's (and the shared expert's) gate/up and output rows [p*n_embd/2, +)
 * of every down; the rank hidden is gathered whole before down, and the router
 * and the scalar gate are replicated. Every row is computed whole by today's
 * kernels.
 * ============================================================ */

/* Row indices of rank p's half of a weight, by kind; returns the count. */
enum {
    TP_ROWS_ATTN_Q, TP_ROWS_ATTN_KV, TP_ROWS_OUT, TP_ROWS_GDN_QKV, TP_ROWS_GDN_VHEAD,
    TP_ROWS_GDN_VDIM, TP_ROWS_FFN, TP_ROWS_SHEXP
};

static int tp_rows(int kind, int p, const pn_qwen35moe_config *cfg, int *rows)
{
    const int S = cfg->ssm_state, nk = cfg->ssm_n_group, nv = cfg->ssm_dt_rank;
    const int nkl = nk / 2, groups = nv / nk;
    int n = 0;

    switch (kind) {
    case TP_ROWS_ATTN_Q: {        /* [q_h | gate_h] per head, 2*head_dim rows a head */
        const int per = cfg->n_head / 2 * 2 * cfg->head_dim;
        for (int i = 0; i < per; i++)
            rows[n++] = p * per + i;
        break;
    }
    case TP_ROWS_ATTN_KV: {
        const int per = cfg->n_head_kv / 2 * cfg->head_dim;
        for (int i = 0; i < per; i++)
            rows[n++] = p * per + i;
        break;
    }
    case TP_ROWS_OUT:
        for (int i = 0; i < cfg->n_embd / 2; i++)
            rows[n++] = p * (cfg->n_embd / 2) + i;
        break;
    case TP_ROWS_GDN_QKV:         /* [q | k] rows of k-heads nkl*p.., then v rows */
        for (int part = 0; part < 2; part++)
            for (int i = 0; i < nkl * S; i++)
                rows[n++] = part * nk * S + p * nkl * S + i;
        for (int r = 0; r < groups; r++)
            for (int i = 0; i < nkl * S; i++)
                rows[n++] = 2 * nk * S + (r * nk + p * nkl) * S + i;
        break;
    case TP_ROWS_GDN_VHEAD:       /* one row (or element) per v-head */
        for (int r = 0; r < groups; r++)
            for (int j = 0; j < nkl; j++)
                rows[n++] = r * nk + p * nkl + j;
        break;
    case TP_ROWS_GDN_VDIM:        /* S rows per v-head (z gate) */
        for (int r = 0; r < groups; r++)
            for (int i = 0; i < nkl * S; i++)
                rows[n++] = (r * nk + p * nkl) * S + i;
        break;
    case TP_ROWS_FFN:
        for (int j = 0; j < cfg->ffn_length / 2; j++)
            rows[n++] = 2 * j + p;
        break;
    case TP_ROWS_SHEXP:           /* MoE shared expert gate/up: contiguous halves */
        for (int j = 0; j < cfg->shared_ff / 2; j++)
            rows[n++] = p * (cfg->shared_ff / 2) + j;
        break;
    }
    return n;
}

/* Upload the rows rows[0..nrows) of w (in that order) to ctx's die: the raw
 * rows are gathered, then repacked exactly as gpu_weight_upload does. */
static int gpu_weight_upload_row_map(fc_cuda_ctx *ctx, gpu_weight *gw, const pn_qweight *w,
                                     const int *rows, int nrows)
{
    memset(gw, 0, sizeof(*gw));
    gw->cpu = w;
    gw->type = w->type;
    gw->rows = nrows;
    gw->cols = w->cols;
    gw->row_stride = (int)w->row_stride;
    gw->ctx = ctx;
    if (!w->data || nrows <= 0)
        return -1;
    unsigned char *raw = malloc((size_t)nrows * w->row_stride);
    if (!raw)
        return -1;
    for (int i = 0; i < nrows; i++) {
        if (rows[i] < 0 || rows[i] >= w->rows) {
            free(raw);
            return -1;
        }
        memcpy(raw + (size_t)i * w->row_stride,
               (const unsigned char *)w->data + (size_t)rows[i] * w->row_stride,
               w->row_stride);
    }
    size_t rstride = fc_repack_row_bytes((int)w->type, w->cols);
    int rc = -1;
    if (rstride) {
        unsigned char *tmp = malloc((size_t)nrows * rstride);
        if (tmp && fc_repack_weight((int)w->type, raw, tmp, nrows, w->cols) == 0) {
            gw->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, (size_t)nrows * rstride);
            gw->row_stride = (int)rstride;
            rc = gw->bytes ? 0 : -1;
        }
        free(tmp);
    } else if ((int)w->type == GGUF_QUANT_F32) {
        gw->bytes = fc_buffer_from_ptr_ctx(ctx, raw, (size_t)nrows * w->row_stride);
        rc = gw->bytes ? 0 : -1;
    }
    free(raw);
    return rc;
}

/* Rank p's Q8_0 parity slab of w (all rows, the inputs k % 2 == p). */
static int gpu_weight_upload_parity(fc_cuda_ctx *ctx, gpu_weight *gw, const pn_qweight *w,
                                    int p)
{
    memset(gw, 0, sizeof(*gw));
    gw->cpu = w;
    gw->type = w->type;
    gw->rows = w->rows;
    gw->cols = w->cols;
    gw->ctx = ctx;
    gw->kpar = 1;
    size_t rstride = fc_repack_q80_parity_row_bytes(w->cols);
    if (!w->data || (int)w->type != GGUF_QUANT_Q8_0 || !rstride)
        return -1;
    unsigned char *tmp = malloc((size_t)w->rows * rstride);
    if (!tmp)
        return -1;
    if (fc_repack_weight_q80_parity(w->data, tmp, w->rows, w->cols, p) == 0)
        gw->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, (size_t)w->rows * rstride);
    free(tmp);
    gw->row_stride = (int)rstride;
    return gw->bytes ? 0 : -1;
}

/* Elements idx[i]*per .. +per of src, in order, as one F32 vector on ctx. */
static int gpu_vec_upload_map(fc_cuda_ctx *ctx, gpu_vec *gv, const float *src,
                              const int *idx, int n, int per)
{
    gv->n = n * per;
    gv->buf = NULL;
    gv->ctx = ctx;
    if (!src || n <= 0)
        return -1;
    float *tmp = malloc((size_t)n * per * sizeof(float));
    if (!tmp)
        return -1;
    for (int i = 0; i < n; i++)
        memcpy(tmp + (size_t)i * per, src + (size_t)idx[i] * per, (size_t)per * sizeof(float));
    gv->buf = fc_buffer_from_ptr_ctx(ctx, tmp, (size_t)n * per * sizeof(float));
    free(tmp);
    return gv->buf ? 0 : -1;
}

/* Rows [p*per/2, +per/2) of every one of the n_expert experts of the merged
 * Q8_0 tensor whose expert 0 is w (per rows each, contiguous), in expert order,
 * repacked to the expert layout like gpu_expert_upload. */
static int gpu_expert_upload_half(fc_cuda_ctx *ctx, gpu_weight *gw, const pn_qweight *w,
                                  int n_expert, int p)
{
    const int per = w->rows, keep = per / 2;
    const size_t rows = (size_t)n_expert * keep;
    size_t rstride = fc_moe_repack_row_bytes((int)w->type, w->cols);
    memset(gw, 0, sizeof(*gw));
    gw->cpu = w;
    gw->type = w->type;
    gw->rows = (int)rows;
    gw->cols = w->cols;
    gw->row_stride = (int)rstride;
    gw->ctx = ctx;
    if (!w->data || (int)w->type != GGUF_QUANT_Q8_0 || !rstride)
        return -1;
    unsigned char *raw = malloc(rows * w->row_stride);
    unsigned char *tmp = malloc(rows * rstride);
    int rc = -1;
    if (raw && tmp) {
        for (int e = 0; e < n_expert; e++)
            memcpy(raw + (size_t)e * keep * w->row_stride,
                   (const unsigned char *)w->data +
                       ((size_t)e * per + (size_t)p * keep) * w->row_stride,
                   (size_t)keep * w->row_stride);
        if (fc_moe_repack_weight((int)w->type, raw, tmp, (int)rows, w->cols) == 0) {
            gw->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, rows * rstride);
            rc = gw->bytes ? 0 : -1;
        }
    }
    free(raw);
    free(tmp);
    return rc;
}

/* Rank p's MoE: router and scalar gate whole; expert and shared gate/up rows
 * [p*ff/2, +ff/2); expert and shared down output rows [p*n_embd/2, +). */
static int gpu_tp_moe_upload(fc_cuda_ctx *ctx, gpu_moe *g, const pn_qwen35moe_moe *moe,
                             const pn_qwen35moe_config *cfg, int p, int *rows)
{
    g->n_expert = cfg->n_expert;
    g->grouped = 1;
    g->gate_exps = g->up_exps = g->down_exps = NULL;
    if (gpu_weight_upload(ctx, &g->ffn_gate_inp, &moe->ffn_gate_inp) < 0 ||
        gpu_expert_upload_half(ctx, &g->gate_all, &moe->gate_exps[0], cfg->n_expert, p) < 0 ||
        gpu_expert_upload_half(ctx, &g->up_all, &moe->up_exps[0], cfg->n_expert, p) < 0 ||
        gpu_expert_upload_half(ctx, &g->down_all, &moe->down_exps[0], cfg->n_expert, p) < 0)
        return -1;
    int nr = tp_rows(TP_ROWS_SHEXP, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &g->ffn_gate_shexp, &moe->ffn_gate_shexp, rows, nr) < 0 ||
        gpu_weight_upload_row_map(ctx, &g->ffn_up_shexp, &moe->ffn_up_shexp, rows, nr) < 0)
        return -1;
    nr = tp_rows(TP_ROWS_OUT, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &g->ffn_down_shexp, &moe->ffn_down_shexp, rows, nr) < 0)
        return -1;
    g->gate_inp_shexp_host = (const float *)moe->ffn_gate_inp_shexp.data;
    return gpu_vec_upload(ctx, &g->ffn_gate_inp_shexp,
                          (const float *)moe->ffn_gate_inp_shexp.data, cfg->n_embd);
}

/* Rank p's dense FFN: gate/up rows k % 2 == p, down's parity slab. */
static int gpu_tp_ffn_upload(fc_cuda_ctx *ctx, gpu_moe *g, const pn_qwen35moe_moe *moe,
                             const pn_qwen35moe_config *cfg, int p, int *rows)
{
    if (cfg->n_expert)
        return gpu_tp_moe_upload(ctx, g, moe, cfg, p, rows);
    g->n_expert = 0;
    g->grouped = 0;
    g->gate_exps = g->up_exps = g->down_exps = NULL;
    g->gate_inp_shexp_host = NULL;
    const int nr = tp_rows(TP_ROWS_FFN, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &g->ffn_gate_shexp, &moe->ffn_gate_shexp, rows, nr) < 0 ||
        gpu_weight_upload_row_map(ctx, &g->ffn_up_shexp, &moe->ffn_up_shexp, rows, nr) < 0 ||
        gpu_weight_upload_parity(ctx, &g->ffn_down_shexp, &moe->ffn_down_shexp, p) < 0)
        return -1;
    return 0;
}

static int gpu_tp_upload_ssm_layer(fc_cuda_ctx *ctx, gpu_ssm_layer *d,
                                   const pn_qwen35moe_ssm_layer *l,
                                   const pn_qwen35moe_config *cfg, int p, int *rows)
{
    const int n_embd = cfg->n_embd;
    int n;
    d->ctx = ctx;
    d->tp_on = 1;
    d->tp_rank = p;
    if (gpu_vec_upload(ctx, &d->attn_norm, l->attn_norm, n_embd) < 0 ||
        gpu_vec_upload(ctx, &d->ssm_norm, l->ssm_norm, cfg->ssm_state) < 0 ||
        gpu_vec_upload(ctx, &d->post_attention_norm, l->post_attention_norm, n_embd) < 0)
        return -1;
    n = tp_rows(TP_ROWS_GDN_QKV, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->attn_qkv, &l->attn_qkv, rows, n) < 0 ||
        gpu_vec_upload_map(ctx, &d->ssm_conv1d, l->ssm_conv1d, rows, n, cfg->ssm_conv) < 0)
        return -1;
    n = tp_rows(TP_ROWS_GDN_VDIM, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->attn_gate, &l->attn_gate, rows, n) < 0)
        return -1;
    n = tp_rows(TP_ROWS_GDN_VHEAD, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->ssm_alpha, &l->ssm_alpha, rows, n) < 0 ||
        gpu_weight_upload_row_map(ctx, &d->ssm_beta, &l->ssm_beta, rows, n) < 0 ||
        gpu_vec_upload_map(ctx, &d->ssm_a, l->ssm_a, rows, n, 1) < 0 ||
        gpu_vec_upload_map(ctx, &d->ssm_dt, l->ssm_dt, rows, n, 1) < 0)
        return -1;
    n = tp_rows(TP_ROWS_OUT, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->ssm_out, &l->ssm_out, rows, n) < 0)
        return -1;
    return gpu_tp_ffn_upload(ctx, &d->moe, &l->moe, cfg, p, rows);
}

static int gpu_tp_upload_attn_layer(fc_cuda_ctx *ctx, gpu_attn_layer *d,
                                    const pn_qwen35moe_attn_layer *l,
                                    const pn_qwen35moe_config *cfg, int p, int *rows)
{
    const int n_embd = cfg->n_embd;
    int n;
    d->ctx = ctx;
    d->tp_on = 1;
    d->tp_rank = p;
    if (gpu_vec_upload(ctx, &d->attn_norm, l->attn_norm, n_embd) < 0 ||
        gpu_vec_upload(ctx, &d->attn_q_norm, l->attn_q_norm, cfg->head_dim) < 0 ||
        gpu_vec_upload(ctx, &d->attn_k_norm, l->attn_k_norm, cfg->head_dim) < 0 ||
        gpu_vec_upload(ctx, &d->post_attention_norm, l->post_attention_norm, n_embd) < 0)
        return -1;
    n = tp_rows(TP_ROWS_ATTN_Q, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->attn_q, &l->attn_q, rows, n) < 0)
        return -1;
    n = tp_rows(TP_ROWS_ATTN_KV, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->attn_k, &l->attn_k, rows, n) < 0 ||
        gpu_weight_upload_row_map(ctx, &d->attn_v, &l->attn_v, rows, n) < 0)
        return -1;
    n = tp_rows(TP_ROWS_OUT, p, cfg, rows);
    if (gpu_weight_upload_row_map(ctx, &d->attn_output, &l->attn_output, rows, n) < 0)
        return -1;
    return gpu_tp_ffn_upload(ctx, &d->moe, &l->moe, cfg, p, rows);
}

/* Die i's halves of its stage's layers (rank i % 2). */
static int gpu_tp_upload_layer_range(pn_qwen35moe_model *model, pn_qwen35moe_gpu *g, int i,
                                     const pn_qwen35moe_config *cfg)
{
    const int p = i % 2;
    int rows_cap = cfg->ffn_length;
    if (rows_cap < cfg->conv_dim)
        rows_cap = cfg->conv_dim;
    if (rows_cap < 2 * cfg->n_head * cfg->head_dim)
        rows_cap = 2 * cfg->n_head * cfg->head_dim;
    int *rows = malloc((size_t)rows_cap * sizeof(int));
    if (!rows)
        return -1;
    uint64_t start = fc_ctx_alloc_bytes(g->ctx[i]);
    int rc = 0;
    for (int layer = g->layer_lo[i]; rc == 0 && layer < g->layer_hi[i]; layer++) {
        pn_load_progress_set_layer(layer, cfg->n_layer);
        int idx = pn_qwen35moe_model_layer_index(model, layer);
        if (pn_qwen35moe_model_layer_type(model, layer) == PN_QWEN35MOE_LAYER_ATTN)
            rc = gpu_tp_upload_attn_layer(g->ctx[i], &g->attn[idx + p * g->tp_n_attn],
                                          pn_qwen35moe_model_attn_layer(model, idx), cfg,
                                          p, rows);
        else
            rc = gpu_tp_upload_ssm_layer(g->ctx[i], &g->ssm[idx + p * g->tp_n_ssm],
                                         pn_qwen35moe_model_ssm_layer(model, idx), cfg, p,
                                         rows);
        pn_load_progress_set_layer(layer + 1, cfg->n_layer);
        gpu_progress_note_dies(g);
        gpu_progress_project(model, g, i, start, layer + 1);
    }
    free(rows);
    return rc;
}

static int gpu_upload_layer_range(pn_qwen35moe_model *model,
                                  pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                                  const pn_qwen35moe_config *cfg,
                                  int layer_lo, int layer_hi)
{
    int die_index = 0;
    while (die_index < g->n_die - 1 && g->ctx[die_index] != ctx)
        die_index++;
    uint64_t start = fc_ctx_alloc_bytes(ctx);
    for (int layer = layer_lo; layer < layer_hi; layer++) {
        pn_load_progress_set_layer(layer, cfg->n_layer);
        int idx = pn_qwen35moe_model_layer_index(model, layer);
        if (pn_qwen35moe_model_layer_type(model, layer) ==
            PN_QWEN35MOE_LAYER_ATTN) {
            const pn_qwen35moe_attn_layer *l =
                pn_qwen35moe_model_attn_layer(model, idx);
            if (gpu_upload_attn_layer(ctx, &g->attn[idx], l, cfg) < 0)
                return -1;
        } else {
            const pn_qwen35moe_ssm_layer *l =
                pn_qwen35moe_model_ssm_layer(model, idx);
            if (gpu_upload_ssm_layer(ctx, &g->ssm[idx], l, cfg) < 0)
                return -1;
        }
        pn_load_progress_set_layer(layer + 1, cfg->n_layer);
        gpu_progress_note_dies(g);
        gpu_progress_project(model, g, die_index, start, layer + 1);
    }
    return 0;
}

/* Whether the T3 vocab-split head can run on this placement: it pairs the head
 * die with die 0 over a peer link, which is only the adjacent pair when there
 * are EXACTLY two dies. At three or more, die 0 is several hops upstream of
 * the head and its x_norm mirror would be a copy across the whole pipeline for
 * no locality — so the path stays off rather than quietly costing a hop. */
static int vocab_split_eligible(const pn_qwen35moe_gpu *g, int small_bar)
{
    return vocab_split_on() && small_bar && g->n_die == 2;
}

/* Admission for ONE die of the split (§6.1). `ordinal` is what the die is
 * called in the memwatch/report space (its CUDA ordinal in small-BAR mode, its
 * index under barswap). `head_share_bytes` is any non-layer weight this die
 * holds beyond the head convention — used to charge die 0 for the vocab-split
 * half it may take. */
static int gpu_admit_one_die(pn_qwen35moe_model *model, pn_qwen35moe_gpu *g,
                             int i, int ordinal, uint64_t head_share_bytes)
{
    uint64_t extra = head_share_bytes;
    int last = i == g->n_die - 1;
    if (last)
        extra += gpu_head_weight_bytes(model);
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    return gpu_admit_die_share(model, g->ctx[i], ordinal,
                               g->layer_lo[i], g->layer_hi[i], extra, last,
                               cfg->n_layer, gpu_admit_max_ctx_model(model),
                               pn_qwen35moe_model_kv_q4(model), g->tp);
}

/* Bind a CUDA context to every die of the split and gate each die on its own
 * share, before any weight moves. In small-BAR mode all dies are live at once,
 * so the whole placement is proved to fit up front — a doomed split is refused
 * in milliseconds rather than after a multi-minute upload. Legacy barswap can
 * only see the die currently in its MMIO slot, so only die 0 is bound here and
 * the rest are bound + gated inside the upload loop.
 *
 * Die 0's context always binds here, split or not. */
static int gpu_bind_and_gate_dies(pn_qwen35moe_model *model,
                                  pn_qwen35moe_gpu *g, int small_bar)
{
    /* With the vocab split ON, die 0 also holds the head's top half. Charge it
     * for that AND still charge the head die the whole head: the split is
     * best-effort and falls back to the head die holding all of it, so both
     * dies stay upper-bounded either way. */
    uint64_t head_a = vocab_split_eligible(g, small_bar)
                    ? gpu_head_weight_bytes(model) / 2 : 0;

    int last = small_bar ? g->n_die : 1;
    for (int i = 0; i < last; i++) {
        int ordinal = small_bar ? g->die[i] : 0;
        g->ctx[i] = fc_cuda_init_device(ordinal);
        if (!g->ctx[i] || !fc_cuda_ctx_available(g->ctx[i])) {
            fprintf(stderr, "qwen35moe: fc_cuda_init_device for die %d "
                    "(ordinal %d) failed\n", i, ordinal);
            pn_accel_refusal_note("no CUDA context on GPU die %d", ordinal);
            return -1;
        }
        if (gpu_admit_one_die(model, g, i, ordinal, i == 0 ? head_a : 0) < 0)
            return -1;
        /* The progress bar's first total for this die (refined while it uploads). */
        pn_load_progress_set_total(g->die[i], fc_ctx_alloc_bytes(g->ctx[i]) +
                                   pn_gguf_layer_range_bytes(pn_qwen35moe_gguf(model),
                                                             g->layer_lo[i], g->layer_hi[i]) +
                                   gpu_progress_die_extra(model, g, i) + (i == 0 ? head_a : 0));
    }
    return 0;
}

/* Does the peer link between two contexts actually MOVE BYTES?
 *
 * Asking the driver is not enough, and this is not a theoretical worry. On the
 * K80 host, for two dies on DIFFERENT cards, cuDeviceCanAccessPeer answers yes,
 * cuCtxEnablePeerAccess succeeds, and cuMemcpyPeer returns CUDA_SUCCESS —
 * while copying nothing at all. The destination keeps whatever it held. Every
 * layer of the API reports success; only the data disagrees.
 *
 * Fed into a layer split that is silent, not loud: the receiving die runs its
 * layers on the previous step's residual stream and the model emits
 * fluent-shaped garbage. Measured 2026-08-21 on dies 4/6 and 5/6 (cross-card:
 * every one of 2048 floats unmoved) against dies 4/5 and 6/7 (same card, behind
 * the on-card PLX switch: all 2048 correct).
 *
 * So the link is PROVEN before it is trusted: poison the destination, copy a
 * known pattern across, read it back and compare. Once per pair at load, on 64
 * floats. Returns 1 if the bytes arrived, 0 otherwise — and 0 means the caller
 * stages that hop through the host, which always works.
 *
 * Thread-safe: no (touches both contexts; call at load, single-threaded). */
static int gpu_peer_link_moves_bytes(fc_cuda_ctx *from, fc_cuda_ctx *to)
{
    enum { PROBE_N = 64 };
    float probe[PROBE_N], back[PROBE_N];
    for (int i = 0; i < PROBE_N; i++) {
        probe[i] = (float)(i + 1) * 1.5f;
        back[i] = -1.0f;                  /* poison: a no-op copy leaves this */
    }
    size_t bytes = sizeof probe;
    fc_buffer *src = fc_buffer_alloc_ctx(from, bytes);
    fc_buffer *dst = fc_buffer_alloc_ctx(to, bytes);
    int ok = 0;
    if (src && dst) {
        fc_buffer_from_cpu_n(dst, back, bytes);
        fc_sync_ctx(to);
        fc_buffer_from_cpu_n(src, probe, bytes);
        fc_sync_ctx(from);
        if (fc_buffer_copy_peer(dst, src, bytes) == 0) {
            fc_sync_ctx(to);
            fc_buffer_to_cpu_n(dst, back, bytes);
            fc_sync_ctx(to);
            ok = memcmp(back, probe, bytes) == 0;
        }
    }
    fc_buffer_free(src);
    fc_buffer_free(dst);
    return ok;
}

/* T3 — vocab-split the LM head across the two dies (QWEN35_VOCAB_SPLIT).
 * `g->output` (the FULL head, already resident on the head die) is left
 * untouched so the single-token, prefill and pipeline head paths keep working;
 * the split uses SEPARATE half buffers only in the decode-slots head: die 0
 * gets output_a (top half [0,half)) + a norm copy, the head die gets output_b
 * (bottom half [half,nv)) as a zero-copy offset VIEW into the resident full
 * head, so it costs no new memory there. Best-effort — any failure leaves
 * vocab_split_ready = 0 and the decode-slots head runs the full head, so the
 * serve is never regressed. Returns 0 always: this cannot fail a load. */
static int gpu_setup_vocab_split(pn_qwen35moe_model *model,
                                 pn_qwen35moe_gpu *g, int small_bar,
                                 int n_embd)
{
    g->vocab_split_ready = 0;
    if (!vocab_split_eligible(g, small_bar))
        return 0;

    const pn_qweight *ow = pn_qwen35moe_model_output(model);
    int nv = ow ? ow->rows : 0;
    int half = nv / 2;              /* die 0 = [0,half), head die = [half,nv) */
    /* Same proof the boundary hand-off takes: this path P2Ps x_norm from the
     * head die back to die 0 every step, and a peer link that reports success
     * while moving nothing would feed die 0's half of the LM head a stale
     * x_norm — wrong top-k for half the vocab, silently. */
    if (nv < 2 || half < 1 || !g->output.bytes ||
        fc_cuda_devices_same_board(g->die[0], g->head_die) != 1 ||   /* never across boards */
        fc_cuda_enable_peer(g->ctx[0], g->head_ctx) != 0 ||
        !gpu_peer_link_moves_bytes(g->head_ctx, g->ctx[0])) {
        if (nv >= 2 && half >= 1 && g->output.bytes)
            fprintf(stderr, "qwen35moe: vocab-split OFF — no verified peer "
                    "link from die %d to die %d\n", g->head_die, g->die[0]);
        return 0;
    }

    g->output_b = g->output;        /* copy type/cols/rows/stride/ctx */
    g->output_b.bytes = fc_buffer_offset_view(g->output.bytes,
                            (size_t)half * g->output.row_stride);
    g->output_b.rows = nv - half;
    int ok = g->output_b.bytes != NULL &&
             (gpu_weight_upload_rows(g->ctx[0], &g->output_a, ow, 0, half) == 0) &&
             (gpu_vec_upload(g->ctx[0], &g->output_norm_a,
                             pn_qwen35moe_model_output_norm(model),
                             n_embd) == 0);
    if (!ok) {
        fc_buffer_free_view(g->output_b.bytes); g->output_b.bytes = NULL;
        fc_buffer_free(g->output_a.bytes);   g->output_a.bytes = NULL;
        fc_buffer_free(g->output_norm_a.buf); g->output_norm_a.buf = NULL;
        fprintf(stderr, "qwen35moe: vocab-split setup failed — decode-slots "
                "head stays on the full head-die head\n");
        return 0;
    }
    g->vocab_half = half;
    g->vocab_split_ready = 1;
    fc_sync_ctx(g->ctx[0]);
    fprintf(stderr, "qwen35moe: LM head vocab-split ON [die %d rows [0,%d) "
            "copy, die %d rows [%d,%d) view] (T3)\n",
            g->die[0], half, g->head_die, half, nv);
    return 0;
}

/* Enable — and then PROVE — peer access for every ADJACENT die pair, so the
 * per-step residual hand-off can go device-to-device instead of D2H + H2D
 * through the host. Only adjacent pairs matter: the residual stream walks the
 * dies in order, so die i only ever hands to die i+1.
 *
 * Per pair, and proven per pair (gpu_peer_link_moves_bytes): on a K80 host the
 * pairs that really carry bytes are the two dies of ONE card, behind its PLX
 * switch. A cross-card pair claims the capability and drops the data, so it
 * stages through the host — and the pairs that do work still do not.
 * QWEN35_HANDOFF_HOST=1 forces every hop through the host, which is how the two
 * routes are measured against each other. */
static void gpu_enable_handoff_peers(pn_qwen35moe_gpu *g)
{
    const char *force_host = getenv("QWEN35_HANDOFF_HOST");
    if (force_host && force_host[0] && force_host[0] != '0') {
        fprintf(stderr, "qwen35moe: QWEN35_HANDOFF_HOST — every boundary "
                "hand-off staged through the host\n");
        return;
    }
    if (g->bs)
        return;   /* barswap time-multiplexes one slot; no two dies are live */

    int n_peer = 0;
    for (int i = 0; i + 1 < g->n_die; i++) {
        /* Peer access is never even ENABLED across boards: only the two dies of
         * one multi-GPU board (fc_cuda_devices_same_board) get a peer link at all. */
        if (fc_cuda_devices_same_board(g->die[i], g->die[i + 1]) != 1) {
            g->peer_ok[i] = 0;
            fprintf(stderr, "qwen35moe: die %d -> %d not on one board: hand-off "
                    "staged through the host (peer access not enabled)\n",
                    g->die[i], g->die[i + 1]);
            continue;
        }
        g->peer_ok[i] = (fc_cuda_enable_peer(g->ctx[i], g->ctx[i + 1]) == 0) &&
                        gpu_peer_link_moves_bytes(g->ctx[i], g->ctx[i + 1]);
        if (!g->peer_ok[i])
            fprintf(stderr, "qwen35moe: die %d -> %d hand-off staged through "
                    "the host (no verified peer link)\n",
                    g->die[i], g->die[i + 1]);
        n_peer += g->peer_ok[i];
    }
    if (g->n_die > 1)
        fprintf(stderr, "qwen35moe: boundary hand-off — %d of %d hops "
                "device-to-device (verified), the rest staged through the "
                "host\n", n_peer, g->n_die - 1);
}

/* ---- QWEN35_TP setup (spec k80_q38_tp2 §3.1) ---- */

static int flash_on(void);   /* fwd decl (defined with the flash path) */

/* QWEN35_TP: unset = AUTO (TP2 wherever tp_refusal finds nothing), "2" = asked for
 * (a refusal is logged loudly), "0" = always the layer split. */
enum { TP_MODE_OFF, TP_MODE_AUTO, TP_MODE_ON };

static int tp_mode(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_TP");
        v = !e || !e[0] ? TP_MODE_AUTO : atoi(e) == 2 ? TP_MODE_ON : TP_MODE_OFF;
    }
    return v;
}

/* A MoE layer TP can split: router and scalar gate present, the experts merged
 * Q8_0 tensors (contiguous, as the grouped kernels index them) on the expert
 * layout, the shared down a row-sliceable matrix. */
static int tp_moe_weights_ok(const pn_qwen35moe_moe *moe, const pn_qwen35moe_config *cfg)
{
    const pn_qweight *g0 = &moe->gate_exps[0], *u0 = &moe->up_exps[0];
    const pn_qweight *d0 = &moe->down_exps[0];
    if (!moe->ffn_gate_inp.data || !moe->ffn_gate_inp_shexp.data || !moe->ffn_down_shexp.data ||
        !fc_repack_row_bytes((int)moe->ffn_down_shexp.type, moe->ffn_down_shexp.cols))
        return 0;
    for (int e = 0; e < cfg->n_expert; e++) {
        const pn_qweight *ge = &moe->gate_exps[e], *ue = &moe->up_exps[e];
        const pn_qweight *de = &moe->down_exps[e];
        if ((int)ge->type != GGUF_QUANT_Q8_0 || (int)ue->type != GGUF_QUANT_Q8_0 ||
            (int)de->type != GGUF_QUANT_Q8_0 ||
            ge->data != (const uint8_t *)g0->data + (size_t)e * g0->rows * g0->row_stride ||
            ue->data != (const uint8_t *)u0->data + (size_t)e * u0->rows * u0->row_stride ||
            de->data != (const uint8_t *)d0->data + (size_t)e * d0->rows * d0->row_stride)
            return 0;
    }
    return g0->rows == cfg->expert_ff && d0->rows == cfg->n_embd &&
           fc_moe_repack_row_bytes(GGUF_QUANT_Q8_0, g0->cols) &&
           fc_moe_repack_row_bytes(GGUF_QUANT_Q8_0, d0->cols);
}

/* Every matrix TP splits has a GPU form for a row slice; a dense FFN down is Q8_0
 * on 128-column groups (the parity slab), MoE experts as tp_moe_weights_ok. */
static int tp_weights_ok(const pn_qwen35moe_model *model)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        int idx = pn_qwen35moe_model_layer_index(model, layer);
        const pn_qwen35moe_moe *moe;
        const pn_qweight *w[6];
        int nw = 0;
        if (pn_qwen35moe_model_layer_type(model, layer) == PN_QWEN35MOE_LAYER_ATTN) {
            const pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer(model, idx);
            w[nw++] = &l->attn_q;
            w[nw++] = &l->attn_k;
            w[nw++] = &l->attn_v;
            w[nw++] = &l->attn_output;
            moe = &l->moe;
        } else {
            const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(model, idx);
            w[nw++] = &l->attn_qkv;
            w[nw++] = &l->attn_gate;
            w[nw++] = &l->ssm_out;
            moe = &l->moe;
        }
        w[nw++] = &moe->ffn_gate_shexp;
        w[nw++] = &moe->ffn_up_shexp;
        for (int i = 0; i < nw; i++)
            if (!w[i]->data || !fc_repack_row_bytes((int)w[i]->type, w[i]->cols))
                return 0;
        if (cfg->n_expert) {
            if (!tp_moe_weights_ok(moe, cfg))
                return 0;
        } else if ((int)moe->ffn_down_shexp.type != GGUF_QUANT_Q8_0 ||
            !fc_repack_q80_parity_row_bytes(moe->ffn_down_shexp.cols))
            return 0;
    }
    return 1;
}

/* NULL when this model and placement can run TP2, else why not. */
static const char *tp_refusal(const pn_qwen35moe_model *model, const pn_qwen35moe_gpu *g,
                              int small_bar)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    if (g->n_die != 4)
        return "needs exactly 4 dies (two board pairs)";
    if (!small_bar)
        return "barswap placement";
    if (!cfg->dense_ffn && !cfg->n_expert)
        return "neither a dense FFN nor routed experts";
    if (cfg->n_head % 2 || cfg->n_head_kv % 2 || (cfg->n_head / 2) % (cfg->n_head_kv / 2))
        return "attention heads do not halve";
    if (cfg->ssm_n_group % 2 || cfg->ssm_dt_rank % cfg->ssm_n_group)
        return "GDN heads do not halve";
    if (!cfg->n_expert && (cfg->ffn_length % 256 || cfg->n_embd % 64))
        return "FFN or hidden width does not halve on 128-column groups";
    /* MoE: rank gate/up launches cover whole 8-row blocks, down row pairs stay
     * whole, and the shared hidden rides the expert hidden's gather (same ff). */
    if (cfg->n_expert && (cfg->expert_ff % 16 || cfg->n_embd % 64 ||
                          cfg->shared_ff != cfg->expert_ff ||
                          cfg->n_expert_used > PN_QWEN35MOE_GPU_MAX_TOPK))
        return "MoE expert or hidden width does not halve";
    if (!flash_on())
        return "QWEN35_FLASH=0";
    if (pn_qwen35moe_host_kv_mirror_eager())
        return "QWEN35_HOST_KV_MIRROR=1 (eager host KV mirror)";
    /* Rank pairs must be the two dies of one board: peer access is never enabled
     * across boards. */
    if (fc_cuda_devices_same_board(g->die[0], g->die[1]) != 1 ||
        fc_cuda_devices_same_board(g->die[2], g->die[3]) != 1)
        return "dies 0+1 and 2+3 of the placement are not each one board";
    if (!tp_weights_ok(model))
        return cfg->n_expert
            ? "a split weight has no TP form (merged Q8_0 experts, row-sliceable shared expert)"
            : "a split weight has no TP form (Q8_0 FFN down on 128-column groups)";
    return NULL;
}

/* Turn the 4-die plan into two stages of rank pairs: dies (0,1) run layers
 * [0, split), dies (2,3) [split, n_layer). Doubles the layer arrays (rank 1 at
 * idx + count) and derives a rank's local dims. Leaves g->tp 0 (layer split)
 * when TP2 is not requested or refused. */
static int gpu_tp_plan(const pn_qwen35moe_model *model, pn_qwen35moe_gpu *g, int small_bar)
{
    const int mode = tp_mode();
    if (mode == TP_MODE_OFF)
        return 0;
    const char *why = tp_refusal(model, g, small_bar);
    if (why) {
        if (mode == TP_MODE_ON || g->n_die == 4)
            fprintf(stderr, "qwen35moe: tensor parallel %s (%s); layer split\n",
                    mode == TP_MODE_ON ? "refused" : "off", why);
        return 0;
    }
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    pn_die_layer_split plan;
    if (pn_plan_die_layer_split(cfg->n_layer, 2, gpu_split_layer_override(2), &plan) != 0)
        return -1;
    for (int i = 0; i < 4; i++) {
        g->layer_lo[i] = plan.layer_lo[i / 2];
        g->layer_hi[i] = plan.layer_hi[i / 2];
    }
    g->split_layer = g->layer_hi[0];
    gpu_ssm_layer *ssm = calloc((size_t)(2 * g->n_ssm) + 1, sizeof(gpu_ssm_layer));
    gpu_attn_layer *attn = calloc((size_t)(2 * g->n_attn) + 1, sizeof(gpu_attn_layer));
    if (!ssm || !attn) {
        free(ssm);
        free(attn);
        return -1;
    }
    free(g->ssm);
    free(g->attn);
    g->ssm = ssm;
    g->attn = attn;
    g->tp_n_ssm = g->n_ssm;
    g->tp_n_attn = g->n_attn;
    g->n_ssm *= 2;
    g->n_attn *= 2;
    pn_qwen35moe_config *l = &g->tp_cfg;
    *l = *cfg;
    l->n_head = cfg->n_head / 2;
    l->n_head_kv = cfg->n_head_kv / 2;
    l->ssm_n_group = cfg->ssm_n_group / 2;
    l->ssm_dt_rank = cfg->ssm_dt_rank / 2;
    l->ssm_d_inner = cfg->ssm_d_inner / 2;
    l->conv_dim = 2 * cfg->ssm_state * l->ssm_n_group + cfg->ssm_state * l->ssm_dt_rank;
    l->ffn_length = cfg->ffn_length / 2;
    for (int p = 0; p < 2; p++) {
        g->tp_chan[p] = malloc((size_t)l->conv_dim * sizeof(int));
        g->tp_vhead[p] = malloc((size_t)l->ssm_dt_rank * sizeof(int));
        if (!g->tp_chan[p] || !g->tp_vhead[p])
            return -1;
        tp_rows(TP_ROWS_GDN_QKV, p, cfg, g->tp_chan[p]);
        tp_rows(TP_ROWS_GDN_VHEAD, p, cfg, g->tp_vhead[p]);
    }
    g->tp = 1;
    fprintf(stderr, "qwen35moe: tensor parallel (QWEN35_TP %s) — dies %d+%d run layers [%d,%d), "
            "dies %d+%d [%d,%d), each die half of every layer\n",
            mode == TP_MODE_ON ? "2" : "auto; =0 off", g->die[0], g->die[1], g->layer_lo[0],
            g->layer_hi[0], g->die[2], g->die[3], g->layer_lo[2], g->layer_hi[2]);
    return 0;
}

/* Peer access for the two rank pairs ONLY (never across boards: cross-board
 * peer copies on this host report success and deliver garbage), each pair
 * proven both ways. */
static int gpu_tp_enable_pairs(pn_qwen35moe_gpu *g)
{
    for (int s = 0; s < 2; s++) {
        fc_cuda_ctx *a = g->ctx[2 * s], *b = g->ctx[2 * s + 1];
        if (fc_cuda_enable_peer(a, b) != 0 || !gpu_peer_link_moves_bytes(a, b) ||
            !gpu_peer_link_moves_bytes(b, a)) {
            fprintf(stderr, "qwen35moe: tensor parallel needs a verified peer link between "
                    "dies %d and %d (one board)\n", g->die[2 * s], g->die[2 * s + 1]);
            return -1;
        }
    }
    return 0;
}

/* ---- QWEN35_TP host state <-> a rank's local state ----
 * Host mirrors keep the full layout. A rank's KV heads are one contiguous block
 * of the head-major host ring (heads h0 .. h0 + nh), so KV transfers only shift
 * the host offset; its SSM channels and v-heads are spread (tp_chan / tp_vhead),
 * so SSM transfers gather / scatter. */

/* Host-state index of layer struct i (rank-1 halves follow rank 0's). */
static int qw_attn_hidx(const pn_qwen35moe_gpu *g, int i)
{
    return g->tp ? i % g->tp_n_attn : i;
}

static int qw_ssm_hidx(const pn_qwen35moe_gpu *g, int i)
{
    return g->tp ? i % g->tp_n_ssm : i;
}

/* Layer structs to visit against host state s (all of them under QWEN35_TP,
 * whose doubled arrays map back through qw_*_hidx). */
static int qw_ssm_lim(const pn_qwen35moe_gpu *g, const pn_qwen35moe_state *s)
{
    return g->tp ? g->n_ssm : (g->n_ssm < s->n_ssm ? g->n_ssm : s->n_ssm);
}

static int qw_attn_lim(const pn_qwen35moe_gpu *g, const pn_qwen35moe_state *s)
{
    return g->tp ? g->n_attn : (g->n_attn < s->n_attn ? g->n_attn : s->n_attn);
}

/* KV heads a layer struct holds, and the first one's host index. */
static int qw_kv_nh(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv)
{
    return d->tp_on ? kv->n_head_kv / 2 : kv->n_head_kv;
}

static int qw_kv_h0(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv)
{
    return d->tp_on ? d->tp_rank * (kv->n_head_kv / 2) : 0;
}

/* The dims a layer struct's state uses: a TP half's local ones or the model's. */
static const pn_qwen35moe_config *qw_state_cfg(const pn_qwen35moe_gpu *g, int tp_on,
                                               const pn_qwen35moe_config *cfg)
{
    return tp_on ? &g->tp_cfg : cfg;
}

/* Download one QWEN35_TP half's SSM images (conv at conv_off of conv_buf,
 * recurrent at ssm_off of ssm_buf) into host state st, or upload (to_dev) st's
 * part into them, through the rank's channel / v-head maps. */
static void qw_ssm_xfer_at(const pn_qwen35moe_gpu *g, const gpu_ssm_layer *d,
                           fc_buffer *conv_buf, size_t conv_off, fc_buffer *ssm_buf,
                           size_t ssm_off, pn_qwen35moe_ssm_state *st, int to_dev);

/* Upload (to_dev) or download host SSM state st for one layer struct. */
static void qw_ssm_xfer(const pn_qwen35moe_gpu *g, gpu_ssm_layer *d,
                        pn_qwen35moe_ssm_state *st, int to_dev)
{
    if (!d->tp_on) {
        if (to_dev) {
            fc_buffer_from_cpu(d->conv_state_g, st->conv_state);
            fc_buffer_from_cpu(d->ssm_state_g, st->ssm_state);
        } else {
            fc_buffer_to_cpu(d->conv_state_g, st->conv_state);
            fc_buffer_to_cpu(d->ssm_state_g, st->ssm_state);
        }
        return;
    }
    qw_ssm_xfer_at(g, d, d->conv_state_g, 0, d->ssm_state_g, 0, st, to_dev);
}

static void qw_ssm_xfer_at(const pn_qwen35moe_gpu *g, const gpu_ssm_layer *d,
                           fc_buffer *conv_buf, size_t conv_off, fc_buffer *ssm_buf,
                           size_t ssm_off, pn_qwen35moe_ssm_state *st, int to_dev)
{
    const pn_qwen35moe_config *l = &g->tp_cfg;
    const int km1 = st->conv_minus_1, S = st->state_size, p = d->tp_rank;
    const size_t conv_n = (size_t)km1 * l->conv_dim, rec = (size_t)S * S;
    const size_t ssm_n = rec * l->ssm_dt_rank;
    float *tmp = malloc((conv_n + ssm_n) * sizeof(float));
    if (!tmp) {
        fprintf(stderr, "[qwen35moe] TP SSM state transfer: out of host memory\n");
        return;
    }
    if (!to_dev) {
        fc_buffer_to_cpu_at(conv_buf, tmp, conv_off, conv_n * sizeof(float));
        fc_buffer_to_cpu_at(ssm_buf, tmp + conv_n, ssm_off, ssm_n * sizeof(float));
    }
    for (int c = 0; c < l->conv_dim; c++) {
        float *h = st->conv_state + (size_t)g->tp_chan[p][c] * km1;
        if (to_dev)
            memcpy(tmp + (size_t)c * km1, h, (size_t)km1 * sizeof(float));
        else
            memcpy(h, tmp + (size_t)c * km1, (size_t)km1 * sizeof(float));
    }
    for (int v = 0; v < l->ssm_dt_rank; v++) {
        float *h = st->ssm_state + (size_t)g->tp_vhead[p][v] * rec;
        if (to_dev)
            memcpy(tmp + conv_n + (size_t)v * rec, h, rec * sizeof(float));
        else
            memcpy(h, tmp + conv_n + (size_t)v * rec, rec * sizeof(float));
    }
    if (to_dev) {
        fc_buffer_from_cpu_at(conv_buf, conv_off, tmp, conv_n * sizeof(float));
        fc_buffer_from_cpu_at(ssm_buf, ssm_off, tmp + conv_n, ssm_n * sizeof(float));
    }
    free(tmp);
}

int pn_qwen35moe_gpu_upload(pn_qwen35moe_model *model)
{
    if (!model)
        return -1;
    if (!fc_available()) {
        /* No GPU at runtime (the Mac case): graceful skip — the CPU forward
         * stays selected. Not an error. */
        return 0;
    }
    if (pn_qwen35moe_model_gpu(model))
        return 0; /* already uploaded */

    /* MemoryAware load-spike budget (§6.1 refinement, measured on a K80 host
     * 2026-07-09): the GPU upload transiently costs ~2.4 GB of host anon RSS
     * for ~55 s (dequant/staging buffers) before releasing to ~1.1 GB. On the
     * 8 GB box THIS transient — not per-request prefill, which is ~0 on the
     * GPU path — is the real squeeze. Check it ONCE per load attempt, before
     * any die context spins up, so a squeezed host refuses cleanly instead of
     * tipping over mid-upload. Fail-open when host truth is unavailable. */
    {
        pn_mem_state mst;
        pn_memwatch_poll_now();
        pn_memwatch_read(&mst);
        uint64_t spike = 0;
        if (mst.host_total_bytes > 0 &&
            !pn_admit_load_spike(mst.host_avail_bytes, mst.host_total_bytes,
                                 &spike)) {
            fprintf(stderr, "[admit] REFUSE model load: host has %llu MiB "
                    "available but the GPU-upload transient needs ~%llu MiB "
                    "on top of the %llu MiB host reserve "
                    "(PN_LOAD_SPIKE_BYTES) — refusing before the upload\n",
                    (unsigned long long)(mst.host_avail_bytes >> 20),
                    (unsigned long long)(spike >> 20),
                    (unsigned long long)
                        (pn_host_reserve_bytes(mst.host_total_bytes) >> 20));
            pn_accel_refusal_note("host RAM for the GPU upload: needs ~%llu MiB "
                                  "beyond the reserve, %llu MiB available",
                                  (unsigned long long)(spike >> 20),
                                  (unsigned long long)(mst.host_avail_bytes >> 20));
            return -1;
        }
    }

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    int n_ssm = pn_qwen35moe_model_n_ssm(model);
    int n_attn = pn_qwen35moe_model_n_attn(model);
    int n_embd = cfg->n_embd;
    int n_layer = cfg->n_layer;

    pn_qwen35moe_gpu *g = calloc(1, sizeof(*g));
    if (!g)
        return -1;
    /* Registered before anything can bind a slot on it, and unregistered in
     * pn_qwen35moe_gpu_free (which the `fail:` path below also runs), so the
     * session-free hook always sees exactly the live contexts. */
    qw_gpu_register(g);
    g->n_ssm = n_ssm;
    g->n_attn = n_attn;
    g->active_die = -1;
    g->ssm = n_ssm ? calloc((size_t)n_ssm, sizeof(gpu_ssm_layer)) : NULL;
    g->attn = n_attn ? calloc((size_t)n_attn, sizeof(gpu_attn_layer)) : NULL;
    if ((n_ssm && !g->ssm) || (n_attn && !g->attn))
        goto fail;

    /* Plan the split first: it decides how many contexts to bind, what each
     * one uploads, and what admission prices. A width this build cannot run
     * refuses here, before a die is touched. */
    if (gpu_plan_dies(model, n_layer, g) < 0)
        goto fail;
    int n_die = g->n_die;

    /* Single-die prefix cap (single-die mode only): the full model (~16 GB)
     * exceeds one 12 GB die, so QWEN35_GPU_MAX_LAYERS uploads + validates only
     * the first N global layers on one die. Ignored once a real split is
     * configured (the split itself is what makes the full model fit). */
    int max_layers = n_layer;
    if (n_die == 1) {
        const char *cap_env = getenv("QWEN35_GPU_MAX_LAYERS");
        if (cap_env) { int v = atoi(cap_env); if (v > 0 && v < max_layers) max_layers = v; }
        g->layer_hi[0] = max_layers;
    }

    /* Mode: small-BAR (default) maps all dies at once, so each context binds
     * directly to its die's real CUDA ordinal and no barswap multiplexing is
     * needed. Opt into the legacy barswap single-slot regime (>16 GB BARs) with
     * QWEN35_GPU_BARSWAP=1. In barswap mode every context inits device 0 (the
     * one MMIO slot) and the forward bs_acquire()s each die before touching it. */
    int small_bar = (getenv("QWEN35_GPU_BARSWAP") == NULL);
    if (!small_bar && n_die > 2) {
        /* Barswap can only bind a context to a die that is currently in its
         * single MMIO slot, so the contexts have to be created one at a time,
         * interleaved with the uploads — which means a doomed wide split is
         * only discovered several minutes in. Refuse instead of building a
         * second, untested binding order for a regime the fleet no longer runs
         * (the BARs are resized; small-BAR is the default). */
        fprintf(stderr, "qwen35moe: REFUSE — QWEN35_GPU_BARSWAP supports at "
                "most a 2-die split, not %d\n", n_die);
        goto fail;
    }

    if (gpu_tp_plan(model, g, small_bar) < 0)
        goto fail;

    if (n_die >= 2 && !small_bar) {
        g->bs = bs_init();
        if (!g->bs) {
            fprintf(stderr, "qwen35moe: bs_init failed (a layer split needs "
                    "bar_swap.ko)\n");
            goto fail;
        }
        if (gpu_select_die(g, g->die[0]) < 0)
            goto fail;
        gpu_load_nvidia();   /* harmless if already up */
    }

    /* Bind every die's context and gate every die's share BEFORE uploading a
     * single weight, so a split that cannot fit is refused instantly instead
     * of after a multi-minute upload. (Legacy barswap can only see one die at
     * a time, so its second context + gate happen after die 0's upload.) */
    if (gpu_bind_and_gate_dies(model, g, small_bar) < 0)
        goto fail;
    if (g->tp && gpu_tp_enable_pairs(g) < 0)
        goto fail;

    for (int i = 0; i < n_die; i++) {
        /* Legacy barswap: bind + gate die i now that it can be swapped in. */
        if (i > 0 && !small_bar) {
            if (gpu_select_die(g, g->die[i]) < 0)
                goto fail;
            gpu_load_nvidia();
            g->ctx[i] = fc_cuda_init_device(0);
            if (!g->ctx[i] || !fc_cuda_ctx_available(g->ctx[i])) {
                fprintf(stderr, "qwen35moe: fc_cuda_init_device for die %d "
                        "(ordinal 0) failed\n", i);
                goto fail;
            }
            if (gpu_admit_one_die(model, g, i, i /* barswap slot ordinal */,
                                  0) < 0)
                goto fail;
        }
        if (g->tp ? gpu_tp_upload_layer_range(model, g, i, cfg) < 0
                  : gpu_upload_layer_range(model, g, g->ctx[i], cfg,
                                           g->layer_lo[i], g->layer_hi[i]) < 0)
            goto fail;
        /* The final norm + LM head ride on the LAST die (which is also die 0
         * when the model is not split). */
        if (i == n_die - 1) {
            if (gpu_vec_upload(g->ctx[i], &g->output_norm,
                               pn_qwen35moe_model_output_norm(model),
                               n_embd) < 0 ||
                gpu_weight_upload(g->ctx[i], &g->output,
                                  pn_qwen35moe_model_output(model)) < 0)
                goto fail;
        }
        fc_sync_ctx(g->ctx[i]);
    }
    g->head_ctx = (n_die > 1) ? g->ctx[n_die - 1] : NULL;
    if (mtp_upload(model, g, cfg) < 0) {
        fprintf(stderr, "qwen35moe: MTP head upload failed\n");
        goto fail;
    }
    gpu_progress_note_dies(g);

    if (gpu_setup_vocab_split(model, g, small_bar, n_embd) < 0)
        goto fail;
    if (!g->tp)
        gpu_enable_handoff_peers(g);

    for (int i = 0; i < n_die; i++)
        gpu_feed_die_mem_state(g->ctx[i], small_bar ? g->die[i] : i);

    pn_qwen35moe_model_set_gpu(model, g);
    if (n_die == 1) {
        fprintf(stderr,
                "qwen35moe: GPU weights uploaded (%d SSM + %d attn, first %d "
                "global) [single die %d]\n",
                n_ssm, n_attn, max_layers, g->die[0]);
    } else {
        fprintf(stderr, "qwen35moe: GPU weights uploaded [%d-die layer split:",
                n_die);
        for (int i = 0; i < n_die; i++)
            fprintf(stderr, " die %d = [%d,%d)%s", g->die[i], g->layer_lo[i],
                    g->layer_hi[i], i == n_die - 1 ? " + head" : "");
        fprintf(stderr, "]\n");
    }
    gpu_drop_weight_pages(model);
    return 0;

fail:
    if (!pn_accel_refusal_last()[0])
        pn_accel_refusal_note("the upload to GPU die %d failed (an allocation; "
                              "see the log above)", g->n_die > 0 ? g->die[0] : -1);
    /* Reuse the public free on a temporarily-attached handle so every uploaded
     * buffer is released. */
    pn_qwen35moe_model_set_gpu(model, g);
    pn_qwen35moe_gpu_free(model);
    return -1;
}

/* Fold one residency class into the `all` total. */
static void gpu_graph_residency_merge(pn_graph_residency *all,
                                      const pn_graph_residency *part)
{
    all->total            += part->total;
    all->live             += part->live;
    all->failed           += part->failed;
    all->gave_up          += part->gave_up;
    all->ineligible       += part->ineligible;
    all->pending          += part->pending;
    all->lifetime_captures += part->lifetime_captures;
    all->lifetime_failures += part->lifetime_failures;
}

int pn_qwen35moe_gpu_graph_residency(const pn_qwen35moe_model *model,
                                     pn_model_graph_residency *out)
{
    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(model);
    if (!g)
        return -1;

    for (int i = 0; i < g->n_attn; i++)
        pn_graph_residency_add(&out->batched_attn,
                               &g->attn[i].slots_graph_capture);
    for (int i = 0; i < g->n_ssm; i++) {
        pn_graph_residency_add(&out->batched_ssm, &g->ssm[i].slots_graph_capture);
        pn_graph_residency_add(&out->single_ssm, &g->ssm[i].graph_capture);
    }
    for (int i = 0; i < (int)(sizeof(g->seg) / sizeof(g->seg[0])); i++)
        pn_graph_residency_add(&out->segment, &g->seg[i].capture);

    gpu_graph_residency_merge(&out->all, &out->batched_attn);
    gpu_graph_residency_merge(&out->all, &out->batched_ssm);
    gpu_graph_residency_merge(&out->all, &out->single_ssm);
    gpu_graph_residency_merge(&out->all, &out->segment);
    return 0;
}

int pn_qwen35moe_gpu_attn_residency(const pn_qwen35moe_model *model,
                                    pn_model_attn_residency *out)
{
    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(model);
    if (!g)
        return -1;

    for (int i = 0; i < g->n_attn; i++)
        pn_attn_residency_add(&out->batched_attn, &g->attn[i].path);
    /* Only one class today, so `all` is a copy rather than a merge — the
     * single-token attention path is not instrumented, because the serve
     * routes every request through the batch scheduler. */
    out->all = out->batched_attn;
    return 0;
}

void pn_qwen35moe_gpu_free(pn_qwen35moe_model *model)
{
    if (!model)
        return;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(model);
    if (!g)
        return;
    qw_gpu_unregister(g);
    for (int i = 0; i < g->n_ssm; i++) {
        gpu_progress_note_dies(g);
        gpu_ssm_layer *d = &g->ssm[i];
        gpu_vec_free(&d->attn_norm);
        gpu_weight_free(&d->attn_qkv);
        gpu_weight_free(&d->attn_gate);
        gpu_vec_free(&d->ssm_conv1d);
        gpu_vec_free(&d->ssm_a);
        gpu_weight_free(&d->ssm_alpha);
        gpu_weight_free(&d->ssm_beta);
        gpu_vec_free(&d->ssm_dt);
        gpu_vec_free(&d->ssm_norm);
        gpu_weight_free(&d->ssm_out);
        gpu_vec_free(&d->post_attention_norm);
        gpu_moe_free(&d->moe);
        fc_buffer_free(d->conv_state_g);
        fc_buffer_free(d->ssm_state_g);
        fc_buffer_free(d->conv_snaps);
        fc_buffer_free(d->ssm_snaps);
        fc_buffer_free(d->conv_slots);
        fc_buffer_free(d->ssm_slots);
        fc_buffer_free(d->conv_ckpt);
        fc_buffer_free(d->ssm_ckpt);
        gpu_fused_free(&d->fx_proj);
        gpu_fused_free(&d->fx_out);
        gpu_fused_free(&d->fx_proj_s);
        gpu_fused_free(&d->fx_out_s);
        gpu_fused_free(&d->moe.fx_pre);
        gpu_fused_free(&d->moe.fx_post);
        gpu_fused_free(&d->moe.fx_pre_s);
        gpu_fused_free(&d->moe.fx_post_s);
        fc_graph_free(d->graph);
        fc_graph_free(d->slots_graph);
    }
    for (int i = 0; i < g->n_attn; i++) {
        gpu_progress_note_dies(g);
        gpu_attn_layer *d = &g->attn[i];
        gpu_vec_free(&d->attn_norm);
        gpu_weight_free(&d->attn_q);
        gpu_weight_free(&d->attn_k);
        gpu_weight_free(&d->attn_v);
        gpu_vec_free(&d->attn_q_norm);
        gpu_vec_free(&d->attn_k_norm);
        gpu_weight_free(&d->attn_output);
        gpu_vec_free(&d->post_attention_norm);
        gpu_moe_free(&d->moe);
        fc_buffer_free(d->k_cache_g);
        fc_buffer_free(d->v_cache_g);
        fc_buffer_free(d->k_sink_g);
        fc_buffer_free(d->v_sink_g);
        fc_buffer_free(d->k_win_g);
        fc_buffer_free(d->v_win_g);
        fc_buffer_free(d->k_win_snap_g);
        fc_buffer_free(d->v_win_snap_g);
        fc_buffer_free(d->dec_m_g);
        fc_buffer_free(d->dec_l_g);
        fc_buffer_free(d->dec_acc_g);
        fc_buffer_free(d->k_slots_g);
        fc_buffer_free(d->v_slots_g);
        fc_buffer_free(d->k_sink_slots_g);
        fc_buffer_free(d->v_sink_slots_g);
        fc_buffer_free(d->k_win_slots_g);
        fc_buffer_free(d->v_win_slots_g);
        fc_buffer_free(d->k_sink_ckpt);
        fc_buffer_free(d->v_sink_ckpt);
        fc_buffer_free(d->k_win_ckpt);
        fc_buffer_free(d->v_win_ckpt);
        fc_buffer_free(d->dec_slots_m_g);
        fc_buffer_free(d->dec_slots_l_g);
        fc_buffer_free(d->dec_slots_acc_g);
        gpu_fused_free(&d->fx_qkv);
        gpu_fused_free(&d->fx_out);
        gpu_fused_free(&d->fx_out_s);
        gpu_fused_free(&d->moe.fx_pre);
        gpu_fused_free(&d->moe.fx_post);
        gpu_fused_free(&d->moe.fx_pre_s);
        gpu_fused_free(&d->moe.fx_post_s);
        fc_graph_free(d->slots_graph);
    }
    for (int i = 0; i < PN_DIE_LAYER_SPLIT_MAX_DIES; i++) {
        fc_graph_free(g->seg[i].graph);
        fc_event_free(g->as_ev[i]);
    }
    qw_pcie_stage_free(g->pcie);
    g->pcie = NULL;
    for (int i = 0; i < PN_DIE_LAYER_SPLIT_MAX_DIES; i++) {   /* QWEN35_TP */
        for (int l = 0; l < QW_N_LANES; l++) {
            qw_tp_xset *x = &g->tpx[l];
            fc_buffer_free(x->core[i]);
            fc_buffer_free(x->half[i]);
            fc_buffer_free(x->blk[i]);
            fc_buffer_free(x->recv[i]);
            fc_buffer_free(x->hidl[i]);
            fc_buffer_free(x->hidf[i]);
            fc_buffer_free(x->mol[i]);
            fc_buffer_free(x->mof[i]);
            for (int k = 0; k < TP_N_XCHG; k++)
                fc_event_free(x->ev[i][k]);
            fc_buffer_free(x->sync[i]);
        }
        fc_event_free(g->tp_gev[i]);
    }
    for (int s = 0; s < 2; s++)
        fc_graph_free(g->tp_sgraph[s]);
    for (int p = 0; p < 2; p++) {
        free(g->tp_chan[p]);
        free(g->tp_vhead[p]);
    }
    free(g->ssm);
    free(g->attn);
    mtp_free(g);   /* before g->output: its draft-vocabulary views alias it */
    gpu_vec_free(&g->output_norm);
    gpu_weight_free(&g->output);
    /* T3 vocab-split head buffers (no-ops when the split was never enabled).
     * output_b.bytes is a non-owning offset VIEW into g->output — free only the
     * view wrapper, never the aliased device memory (that is g->output's). */
    gpu_weight_free(&g->output_a);
    fc_buffer_free_view(g->output_b.bytes);
    g->output_b.bytes = NULL;
    gpu_vec_free(&g->output_norm_a);
    fc_buffer_free(g->xnorm_a);
    fc_buffer_free(g->spec_logits_a);
    fc_buffer_free(g->spec_logits_b);
    for (int i = 0; i < PN_DIE_LAYER_SPLIT_MAX_DIES; i++) {
        gpu_die_scratch_free(&g->scratch[i]);
        gpu_batch_scratch_free(&g->bscratch[i]);
        gpu_die_scratch_free(&g->scratch_dec[i]);
        gpu_batch_scratch_free(&g->bscratch_dec[i]);
        fc_buffer_free_view(g->rows_view[i]);
        fc_buffer_free(g->pos_buf[i]);
        fc_buffer_free(g->bv_m_g[i]);
        fc_buffer_free(g->bv_l_g[i]);
        fc_buffer_free(g->bv_acc_g[i]);
        fc_buffer_free_view(g->vrows_g[i]);
        fc_buffer_free(g->vpos_g[i]);
        for (int r = 0; r < 5; r++)
            fc_graph_free(g->vgraph[i][r].graph);
    }
    fc_buffer_free(g->spec_logits);
    for (int r = 0; r < PN_QWEN35MOE_GPU_MAX_SLOTS; r++)
        fc_buffer_free(g->srow_buf[r]);
    for (int i = 0; i < 4; i++) {
        free(g->bhost[i]);
        free(g->bhost2[i]);
    }
    qw_tt_free(g);
    qw_defer_free(g);
    if (g->host_ok)
        gpu_host_scratch_free(&g->host);

    /* Tear down the per-die contexts and the barswap handle. NOTE: on the K80
     * barswap rig the runtime path uses _Exit() and never reaches this (CUDA's
     * context teardown wedges a thread); this free is for the upload-failure
     * path and non-CUDA hosts. Each fc_buffer_free above self-enters its owning
     * context, so a buffer's die must be the one currently in the slot for its
     * cuMemFree to land on mapped BARs — in split mode the operator should
     * prefer process-death reclamation over calling this mid-run. */
    for (int i = 0; i < g->n_die; i++)
        fc_cuda_ctx_free(g->ctx[i]);
    if (g->bs)
        bs_free(g->bs);
    free(g);
    pn_qwen35moe_model_set_gpu(model, NULL);
}

void pn_qwen35moe_gpu_forget_state_cuda(const pn_qwen35moe_state *s)
{
    if (!s)
        return;
    pthread_mutex_lock(&qw_gpu_reg_mtx);
    for (pn_qwen35moe_gpu *g = qw_gpu_reg_head; g;
         g = (pn_qwen35moe_gpu *)g->reg_next) {
        /* Slot rows: slot_state[] is dereferenced by the slot-row SSM seed
         * (qw_ssm_seed_shared), so a freed state must leave no row naming it. */
        for (int r = 0; r < PN_QWEN35MOE_GPU_MAX_SLOTS; r++)
            if (g->slot_state[r] == (const void *)s)
                g->slot_state[r] = NULL;
        for (int r = 0; r < PN_QWEN35MOE_GPU_MAX_SLOTS; r++)
            if (g->ckpt_state[r] == (const void *)s)
                g->ckpt_state[r] = NULL;
        for (int r = 0; r < PN_QWEN35MOE_GPU_MAX_SLOTS; r++)
            if (g->srow_state[r] == (const void *)s)
                g->srow_state[r] = NULL;
        /* Attention: the per-slot bind bookkeeping. kv_slot_kv is the pointer
         * the ring-realloc rescue dereferences; kv_slot_host_id / kv_gen /
         * kv_synced_pos name the same session's ring buffers by ADDRESS, so
         * leaving them behind would let a recycled malloc alias a new
         * session's cache onto this one's stale watermark. Both go together. */
        for (int i = 0; i < g->n_attn; i++) {
            gpu_attn_layer *d = &g->attn[i];
            for (int j = 0; j < s->n_attn; j++) {
                const pn_qwen35moe_kv *kv = &s->kv[j];
                const void *id_k = (const void *)kv->k;
                const void *id_q = (const void *)kv->kq;
                for (int sl = 0; sl < PN_QWEN35MOE_GPU_MAX_SLOTS; sl++) {
                    const void *hid = d->kv_slot_host_id[sl];
                    if (d->kv_slot_kv[sl] != kv &&
                        !(hid && (hid == id_k || hid == id_q)))
                        continue;
                    d->kv_slot_kv[sl] = NULL;
                    d->kv_slot_host_id[sl] = NULL;
                    d->kv_slot_gen[sl] = 0;
                    d->kv_slot_synced[sl] = 0;
                }
                if (d->kv_host_id &&
                    (d->kv_host_id == id_k || d->kv_host_id == id_q)) {
                    d->kv_host_id = NULL;
                    d->kv_synced_pos = 0;
                }
            }
        }
        /* SSM: host_state names the session whose recurrent image the live
         * GPU buffers hold. Clearing it costs one H2D reseed on the next bind
         * and removes the same recycled-address aliasing. */
        for (int i = 0; i < g->n_ssm; i++) {
            gpu_ssm_layer *d = &g->ssm[i];
            for (int j = 0; j < s->n_ssm; j++)
                if (d->host_state == (const void *)&s->ssm[j])
                    d->host_state = NULL;
        }
    }
    pthread_mutex_unlock(&qw_gpu_reg_mtx);
}

int pn_qwen35moe_gpu_slot_kv_refs_cuda(const void *base, size_t bytes)
{
    if (!base || !bytes)
        return 0;
    uintptr_t lo = (uintptr_t)base, hi = lo + (uintptr_t)bytes;
    int n = 0;
    pthread_mutex_lock(&qw_gpu_reg_mtx);
    for (pn_qwen35moe_gpu *g = qw_gpu_reg_head; g;
         g = (pn_qwen35moe_gpu *)g->reg_next)
        for (int i = 0; i < g->n_attn; i++)
            for (int sl = 0; sl < PN_QWEN35MOE_GPU_MAX_SLOTS; sl++) {
                uintptr_t p = (uintptr_t)g->attn[i].kv_slot_kv[sl];
                if (p >= lo && p < hi)
                    n++;
            }
    pthread_mutex_unlock(&qw_gpu_reg_mtx);
    return n;
}

/* Test hook: how many one-slot fused descriptors of the slots path are built
 * across every live GPU model (gpu_slots_fused_prepare), and in *possible how
 * many there are to build (4 per SSM layer, 3 per attention layer). */
int pn_qwen35moe_gpu_slots_fused_count_cuda(int *possible)
{
    int n = 0, all = 0;
    pthread_mutex_lock(&qw_gpu_reg_mtx);
    for (pn_qwen35moe_gpu *g = qw_gpu_reg_head; g;
         g = (pn_qwen35moe_gpu *)g->reg_next) {
        for (int i = 0; i < g->n_ssm; i++) {
            const gpu_ssm_layer *d = &g->ssm[i];
            n += !!d->fx_proj_s.desc + !!d->fx_out_s.desc +
                 !!d->moe.fx_pre_s.desc + !!d->moe.fx_post_s.desc;
            all += 4;
        }
        for (int i = 0; i < g->n_attn; i++) {
            const gpu_attn_layer *d = &g->attn[i];
            n += !!d->fx_out_s.desc + !!d->moe.fx_pre_s.desc +
                 !!d->moe.fx_post_s.desc;
            all += 3;
        }
    }
    pthread_mutex_unlock(&qw_gpu_reg_mtx);
    if (possible)
        *possible = all;
    return n;
}

/* ============================================================
 * GPU matvec dispatch
 *
 * fc_matvec_q3k consumes raw Q3_K bytes directly. For any other quant type
 * (the GGUF mix can carry Q6_K output, etc.) there is no dedicated kernel yet,
 * so we fall back to a host-side pn_qmv on a read-back of x and write the result
 * back to the GPU out buffer. That fallback is correctness-faithful (same path
 * the CPU reference uses) but costs a D2H/H2D round trip.
 *
 * Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q2_0/F32/F16/BF16 all have a kernel now; the host
 * fallback below only remains for types outside that set.
 * ============================================================ */

/* y[rows] = dequant(W) . x[cols]. x and y are GPU buffers; x_host/y_host are
 * caller-provided host scratch sized >= cols / rows for the fallback path. The
 * matvec runs on the weight's owning die context (w->ctx), so the same call
 * works for layers on die A or die B. x and y MUST belong to that same ctx. */
static void gpu_matvec(const gpu_weight *w, fc_buffer *x, fc_buffer *y,
                       float *x_host, float *y_host)
{
    if (w->kpar) {   /* a TP parity slab has no single-token or host form */
        fprintf(stderr, "[qwen35moe] TP parity weight reached gpu_matvec\n");
        return;
    }
    double t0 = 0.0;
    int prof = prof_on();
    if (prof) {
        fc_sync_ctx(w->ctx);
        t0 = prof_now();
    }
    /* On-device dispatch by quant type. Every type the Q3_K_M mini actually
     * mixes in has a kernel now (matvec_q3k.cu + matvec_kquants.cu); the host
     * fallback below only remains for exotic types outside that set. */
    int on_gpu = 1;
    if (w->bytes) {
        switch ((int)w->type) {
        case GGUF_QUANT_Q3_K:
            fc_matvec_q3k_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_Q4_K:
            fc_matvec_q4k_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_Q5_K:
            fc_matvec_q5k_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_Q6_K:
            fc_matvec_q6k_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_Q8_0:
            fc_matvec_q80_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_Q2_0_G128:
            /* Bonsai native ternary: warp-per-row conditional add/sub, no
             * dequant. Weights repacked to [f32 d][32 codes] at upload (B2). */
            fc_matvec_q2_0_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                               w->row_stride);
            break;
        case GGUF_QUANT_F32:
            fc_matvec_f32_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_F16:
            /* The DENSE half of an F16 model. The launcher is silent when the
             * kernel is absent (a pre-F16 fc_cuda_ptx.h), so probe first and
             * let the host fallback below take it — a no-op launch would leave
             * y holding the previous token's values. */
            if (!fc_matvec_f16_available_ctx(w->ctx)) {
                on_gpu = 0;
                break;
            }
            fc_matvec_f16_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride);
            break;
        case GGUF_QUANT_BF16:
            /* The same dense tensors unrounded — the checkpoints ship bf16, so
             * this is the arm an F16 conversion has to be measured AGAINST.
             * Same probe-first rule as F16 above. */
            if (!fc_matvec_bf16_available_ctx(w->ctx)) {
                on_gpu = 0;
                break;
            }
            fc_matvec_bf16_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                               w->row_stride);
            break;
        default:
            on_gpu = 0;
            break;
        }
    } else {
        on_gpu = 0;
    }
    if (on_gpu) {
        if (prof) {
            fc_sync_ctx(w->ctx);
            double dt = prof_now() - t0;
            int t = (int)w->type;
            if (t >= 0 && t < PROF_MAX_TYPE) {
                prof_mv_time[t] += dt;
                prof_mv_cnt[t]++;
            }
        }
        return;
    }
    /* Host fallback via the CPU qweight (exotic quant types only). */
    fc_sync_ctx(w->ctx);
    fc_buffer_to_cpu(x, x_host);
    pn_qmv(w->cpu, x_host, y_host);
    fc_buffer_from_cpu(y, y_host);
    if (prof) {
        fc_sync_ctx(w->ctx);
        double dt = prof_now() - t0;
        int t = (int)w->type;
        if (t >= 0 && t < PROF_MAX_TYPE) {
            prof_mv_time[t] += dt;
            prof_mv_cnt[t]++;
        }
        prof_mv_fallback += dt;
    }
}

/* ============================================================
 * Fused per-layer launches
 *
 * Decode is launch-bound on the K80 (~80 us/launch), so the per-layer
 * launch storms are collapsed with fc_fused_multi (see fused_layer.cu):
 * each fused launch batches several independent matvecs that read the same
 * x, with the tiny glue ops (sigmoid, decay gate, silu*mul, scalar gate,
 * combine/residual adds) folded into their epilogues. Every segment's math
 * is bit-identical to the stand-alone kernels it replaces. Descriptors are
 * built ONCE per layer (all referenced buffers are persistent) and cached
 * in gpu_fused; a build failure (exotic quant mix) leaves desc NULL and the
 * caller keeps the original unfused sequence.
 * ============================================================ */

/* Fusion-site bitmask (QWEN35_FUSE, default all on) — bisection aid:
 * bit0 ssm proj4, bit1 ssm out+res, bit2 attn qkv, bit3 attn norm+rope,
 * bit4 moe pre, bit5 moe post; on the batched/slots path, bit6 the shexp
 * gate + combine + residual as one fc_moe_shexp_gate_combine_nx launch and
 * bit7 the slots attention output gate folded into the split combine. */
static int gpu_fuse_mask(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_FUSE");
        v = env ? (int)strtol(env, NULL, 0) : 0xFF;
    }
    return v;
}

/* CUDA-graph replay of the SSM layers (QWEN35_GRAPH, default on; set 0 to
 * disable — bisection aid). */
static int gpu_graph_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_GRAPH");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Full-step segment graphs (C1): capture EVERY layer of a die segment —
 * SSM and attention interleaved — into one graph and replay the whole
 * per-die decode step with a single cuGraphLaunch (QWEN35_GRAPH_FULL,
 * default on; set 0 for the previous behavior: per-layer SSM graphs +
 * direct attention — the A/B baseline for the K80 hardware gate). Also requires
 * QWEN35_GRAPH itself on, and self-disables when the device-pos attention
 * kernels are missing from the PTX blob. */
static int gpu_graph_full_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_GRAPH_FULL");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* L9 — per-layer graphs on the BATCHED-decode (slots) path (QWEN35_GRAPH_SLOTS,
 * default on; set 0 for the ungraphed A/B baseline). A sub-gate of QWEN35_GRAPH,
 * the same way QWEN35_GRAPH_FULL is: it only engages when the master graph gate
 * is on. */
static int gpu_graph_slots_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_GRAPH_SLOTS");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* L10 — the ATTENTION half of the slots capture (QWEN35_GRAPH_SLOTS_ATTN,
 * default on; set 0 for the Phase-2 baseline: SSM layers captured, attention
 * layers direct). A sub-gate of QWEN35_GRAPH_SLOTS, so that the one thing
 * Phase 3 changed can be A/B'd on a single binary and a single deployed fleet
 * — which matters here more than anywhere else, since QWEN35_SPROF disables
 * capture outright and cannot be used to price it. */
static int gpu_graph_slots_attn_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_GRAPH_SLOTS_ATTN");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Expert-union dedup for the batched routed gate+up (QWEN35_MOE_DEDUP,
 * default OFF).
 *
 * PREFILL ONLY, by construction — see gpu_moe_block_batch's allow_dedup
 * argument. The dedup kernel's win is n*k / |union of routed experts|, which
 * a 128-token prefill chunk makes ~4x and an 8-slot decode step makes ~1.1x;
 * at decode that does not cover the wider grid and it measured 7-9% SLOWER.
 * Enabling this flag must therefore never reach the decode path.
 *
 * Kept default-off until the prefill A/B is settled; when off, not one
 * device buffer or launch on either path changes. */
static int gpu_moe_dedup_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_MOE_DEDUP");
        v = (env && env[0] == '1') ? 1 : 0;
    }
    return v;
}

/* QWEN35_MOE_DEDUP_DECODE=1 — MEASUREMENT ESCAPE HATCH, never a serving knob.
 *
 * The reason decode is excluded above is a regression figure that had no
 * measurement behind it in this repo. This lets gpu_moe_block_slots opt in so
 * the number can be reproduced on demand instead of taken on faith. It is a
 * SECOND flag on top of QWEN35_MOE_DEDUP: with only QWEN35_MOE_DEDUP=1 set,
 * decode still cannot reach the dedup kernel. */
static int gpu_moe_dedup_decode_override(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_MOE_DEDUP_DECODE");
        v = (env && env[0] == '1') ? 1 : 0;
    }
    return v;
}

/* QWEN35_MOE_DEDUP_Q80 (default on; =0 keeps fc_moe_gateup_nx_ctx): batched
 * PREFILL gate+up over Q8_0 experts through the expert-grouped kernel
 * (fc_moe_gateup_q80_dedup), which reads each expert row once per chunk instead
 * of once per routed token. Bit-exact with the nx kernel per request (foundry
 * test_moe_q80_dedup). Same width floor as the Q3_K dedup (n * k >= n_expert),
 * so decode and verify never reach it. */
static int gpu_moe_dedup_q80_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_MOE_DEDUP_Q80");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Fill one plain matvec segment from a resident weight. */
static fc_fused_seg gpu_seg_mv(const gpu_weight *w, fc_buffer *out, int ep)
{
    fc_fused_seg s;
    memset(&s, 0, sizeof(s));
    s.w = w->bytes;
    s.out = out;
    s.rows = w->rows;
    s.cols = w->cols;
    s.row_stride = w->row_stride;
    s.kind = ((int)w->type == GGUF_QUANT_F32) ? 1 : 0;
    s.qtype = (int)w->type;
    s.ep = ep;
    return s;
}

/* Build (once) and launch a cached fused descriptor. Returns 0 if the fused
 * launch ran, -1 if unsupported (caller must run its unfused sequence). */
static int gpu_fused_run(fc_cuda_ctx *ctx, gpu_fused *f,
                         const fc_fused_seg *segs, int nseg, fc_buffer *x)
{
    if (!f->tried) {
        f->tried = 1;
        f->nseg = nseg;
        f->desc = fc_fused_build_ctx(ctx, segs, nseg, &f->blocks, &f->sh_bytes);
    }
    if (!f->desc)
        return -1;
    fc_fused_multi_ctx(ctx, f->desc, f->nseg, f->blocks, f->sh_bytes, x);
    return 0;
}

static void gpu_fused_free(gpu_fused *f)
{
    if (f && f->desc) {
        fc_buffer_free(f->desc);
        f->desc = NULL;
    }
}

/* ============================================================
 * Host-side scratch carried across the GPU forward
 *
 * The fc_* kernels handle the dense, regular ops (rms_norm, matvec_q3k, rope,
 * qk_rms_norm, ssm_recurrence, conv1d, gated_norm, moe router/gate, silu/add/
 * mul). The irregular glue ops that lack a kernel are done on these host
 * buffers between kernel calls (each marked TODO kernel inline). One allocation
 * sized to the largest intermediate (val_dim / conv_dim / n_head*head_dim).
 * The typedef lives up top (pn_qwen35moe_gpu owns one persistent copy).
 * ============================================================ */

static int gpu_host_scratch_init(gpu_host_scratch *h, const pn_qwen35moe_config *cfg)
{
    int val_dim = cfg->ssm_state * cfg->ssm_dt_rank;     /* 4096 */
    int qg = cfg->head_dim * cfg->n_head * 2;            /* 8192 */
    int cap = cfg->conv_dim;
    if (val_dim > cap) cap = val_dim;
    if (qg > cap) cap = qg;
    if (cfg->n_vocab > cap) cap = cfg->n_vocab;
    h->cap = cap;
    h->a = calloc((size_t)cap, sizeof(float));
    h->b = calloc((size_t)cap, sizeof(float));
    h->c = calloc((size_t)cap, sizeof(float));
    return (h->a && h->b && h->c) ? 0 : -1;
}

static void gpu_host_scratch_free(gpu_host_scratch *h)
{
    free(h->a);
    free(h->b);
    free(h->c);
}

/* Free every buffer in one die's persistent scratch. */
static void gpu_die_scratch_free(gpu_die_scratch *sc)
{
    fc_buffer **bufs[] = {
        &sc->s_qkv, &sc->s_z, &sc->s_beta, &sc->s_alpha, &sc->s_conv_out,
        &sc->s_qg, &sc->s_kg, &sc->s_vg, &sc->s_gbuf, &sc->s_y, &sc->s_gated,
        &sc->a_qg, &sc->a_q, &sc->a_z, &sc->a_k, &sc->a_v, &sc->a_out,
        &sc->m_logits, &sc->m_tidx, &sc->m_twt, &sc->m_gate, &sc->m_up,
        &sc->m_hid, &sc->m_exp, &sc->m_out, &sc->m_shexp, &sc->m_sgate,
        &sc->r_hidden, &sc->r_xnorm, &sc->r_blockout, &sc->r_logits,
        &sc->dec_pos,
    };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        fc_buffer_free(*bufs[i]);
        *bufs[i] = NULL;
    }
    sc->ctx = NULL;
}

/* Get (lazily creating) the persistent scratch for `ctx`'s die. Returns NULL
 * for a context that is not one of the split's dies, or on allocation failure.
 *
 * Keyed by DIE INDEX rather than by scanning a two-entry free list: the list
 * had exactly as many slots as the old split had dies, so a third context
 * would have found none free and silently returned NULL mid-forward. */
static gpu_die_scratch *gpu_get_scratch(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                                        const pn_qwen35moe_config *cfg)
{
    int die_idx = gpu_die_index_of_ctx(g, ctx);
    if (die_idx < 0)
        return NULL;
    gpu_die_scratch *sc = g->lane == QW_LANE_DECODE ? &g->scratch_dec[die_idx]
                                                    : &g->scratch[die_idx];
    if (sc->ctx == ctx)
        return sc;

    const int n_embd = cfg->n_embd;
    const int conv_dim = cfg->conv_dim;
    const int val_dim = cfg->ssm_state * cfg->ssm_dt_rank;
    const int n_v_heads = cfg->ssm_dt_rank;
    const int qg_dim = cfg->head_dim * cfg->n_head * 2;
    const int kv_dim = cfg->head_dim * cfg->n_head_kv;
    const int attn_dim = cfg->head_dim * cfg->n_head;
    int ff = cfg->expert_ff > cfg->shared_ff ? cfg->expert_ff : cfg->shared_ff;

    sc->s_qkv      = fc_buffer_alloc_ctx(ctx, (size_t)conv_dim * sizeof(float));
    sc->s_z        = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->s_beta     = fc_buffer_alloc_ctx(ctx, (size_t)n_v_heads * sizeof(float));
    sc->s_alpha    = fc_buffer_alloc_ctx(ctx, (size_t)n_v_heads * sizeof(float));
    sc->s_conv_out = fc_buffer_alloc_ctx(ctx, (size_t)conv_dim * sizeof(float));
    sc->s_qg       = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->s_kg       = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->s_vg       = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->s_gbuf     = fc_buffer_alloc_ctx(ctx, (size_t)n_v_heads * sizeof(float));
    sc->s_y        = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->s_gated    = fc_buffer_alloc_ctx(ctx, (size_t)val_dim * sizeof(float));
    sc->a_qg       = fc_buffer_alloc_ctx(ctx, (size_t)qg_dim * sizeof(float));
    sc->a_q        = fc_buffer_alloc_ctx(ctx, (size_t)attn_dim * sizeof(float));
    sc->a_z        = fc_buffer_alloc_ctx(ctx, (size_t)attn_dim * sizeof(float));
    sc->a_k        = fc_buffer_alloc_ctx(ctx, (size_t)kv_dim * sizeof(float));
    sc->a_v        = fc_buffer_alloc_ctx(ctx, (size_t)kv_dim * sizeof(float));
    sc->a_out      = fc_buffer_alloc_ctx(ctx, (size_t)attn_dim * sizeof(float));
    /* Dense-FFN (Bonsai) has no router / routed experts, so n_expert and
     * n_expert_used are 0 — but a 0-byte alloc returns NULL and would trip the
     * validity check below. Floor these MoE-only buffers to 1 element; the
     * dense FFN path never reads them. */
    int n_expert_alloc = cfg->n_expert > 0 ? cfg->n_expert : 1;
    int k_alloc = cfg->n_expert_used > 0 ? cfg->n_expert_used : 1;
    sc->m_logits   = fc_buffer_alloc_ctx(ctx, (size_t)n_expert_alloc * sizeof(float));
    sc->m_tidx     = fc_buffer_alloc_ctx(ctx, (size_t)PN_QWEN35MOE_GPU_MAX_TOPK * sizeof(int32_t));
    sc->m_twt      = fc_buffer_alloc_ctx(ctx, (size_t)PN_QWEN35MOE_GPU_MAX_TOPK * sizeof(float));
    sc->m_gate     = fc_buffer_alloc_ctx(ctx, (size_t)ff * sizeof(float));
    sc->m_up       = fc_buffer_alloc_ctx(ctx, (size_t)ff * sizeof(float));
    /* m_hid holds all k selected experts' SwiGLU activations at once for the
     * grouped kernels (hid[s*ff + row]); k*ff also covers the old per-expert
     * path's single-ff use. */
    sc->m_hid      = fc_buffer_alloc_ctx(ctx, (size_t)k_alloc *
                                              (size_t)ff * sizeof(float));
    sc->m_exp      = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->m_out      = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->m_shexp    = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->m_sgate    = fc_buffer_alloc_ctx(ctx, sizeof(float));
    sc->r_hidden   = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->r_xnorm    = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->r_blockout = fc_buffer_alloc_ctx(ctx, (size_t)n_embd * sizeof(float));
    sc->r_logits   = fc_buffer_alloc_ctx(ctx, (size_t)cfg->n_vocab * sizeof(float));
    sc->dec_pos    = fc_buffer_alloc_ctx(ctx, sizeof(int32_t));

    if (!sc->s_qkv || !sc->s_z || !sc->s_beta || !sc->s_alpha ||
        !sc->s_conv_out || !sc->s_qg || !sc->s_kg || !sc->s_vg || !sc->s_gbuf ||
        !sc->s_y || !sc->s_gated || !sc->a_qg || !sc->a_q || !sc->a_z ||
        !sc->a_k || !sc->a_v || !sc->a_out || !sc->m_logits || !sc->m_tidx ||
        !sc->m_twt ||
        !sc->m_gate || !sc->m_up || !sc->m_hid || !sc->m_exp || !sc->m_out ||
        !sc->m_shexp || !sc->m_sgate || !sc->r_hidden || !sc->r_xnorm ||
        !sc->r_blockout || !sc->r_logits || !sc->dec_pos) {
        gpu_die_scratch_free(sc);
        return NULL;
    }
    sc->ctx = ctx;
    return sc;
}

/* ============================================================
 * SSM (Gated-Delta-Net) block on the GPU
 *
 * Mirrors pn_qwen35moe_ssm_block. Projections + conv1d + recurrence + gated
 * norm run on-device; the per-head L2-norm + interleaved GQA broadcast and the
 * decay-gate softplus are host-side glue (TODO kernels).
 * ============================================================ */

/* GPU SSM block. x_norm is the attn_norm'd input (GPU buffer). Writes the block
 * output (pre-residual) to out_buf (GPU buffer) AND adds it into the residual
 * stream `hidden` (fused into the ssm_out epilogue, or an explicit fc_add on
 * the fallback path). dual state buffers (conv_state, ssm_state) are
 * GPU-resident and updated in place. */
static int gpu_ssm_block(gpu_ssm_layer *d, const pn_qwen35moe_config *cfg,
                         fc_buffer *x_norm, fc_buffer *conv_state_g,
                         fc_buffer *ssm_state_g, fc_buffer *out_buf,
                         fc_buffer *hidden, gpu_die_scratch *sc,
                         gpu_host_scratch *h)
{
    const int conv_dim = cfg->conv_dim;
    const int S = cfg->ssm_state;
    const int n_k_heads = cfg->ssm_n_group;
    const int n_v_heads = cfg->ssm_dt_rank;
    const int kv_groups = n_v_heads / n_k_heads;
    const int key_dim = S * n_k_heads;
    const int val_dim = S * n_v_heads;
    const float eps = cfg->rms_norm_eps;
    fc_cuda_ctx *ctx = d->ctx;   /* this layer's die context */

    /* Persistent per-die scratch (every buffer fully overwritten below). */
    fc_buffer *qkv = sc->s_qkv, *z = sc->s_z, *beta = sc->s_beta;
    fc_buffer *alpha = sc->s_alpha, *conv_out = sc->s_conv_out;
    fc_buffer *qg = sc->s_qg, *kg = sc->s_kg, *vg = sc->s_vg;
    fc_buffer *gbuf = sc->s_gbuf, *y = sc->s_y, *gated = sc->s_gated;

    /* 1.+2. The four projections from x_norm — qkv, gate z, beta, alpha —
     * batched into ONE fused launch, with beta's sigmoid and alpha's decay
     * gate (g = ssm_a * softplus(alpha + ssm_dt), written straight to gbuf)
     * folded into the F32 segments' epilogues. Falls back to the original
     * 6-launch sequence if the quant mix is unsupported. */
    {
        fc_fused_seg segs[4];
        segs[0] = gpu_seg_mv(&d->attn_qkv, qkv, 0);
        segs[1] = gpu_seg_mv(&d->attn_gate, z, 0);
        segs[2] = gpu_seg_mv(&d->ssm_beta, beta, 1);      /* sigmoid */
        segs[3] = gpu_seg_mv(&d->ssm_alpha, gbuf, 2);     /* decay gate */
        segs[3].aux0 = d->ssm_dt.buf;
        segs[3].aux1 = d->ssm_a.buf;
        if (!(gpu_fuse_mask() & 1) ||
            gpu_fused_run(ctx, &d->fx_proj, segs, 4, x_norm) != 0) {
            gpu_matvec(&d->attn_qkv, x_norm, qkv, h->a, h->b);
            gpu_matvec(&d->attn_gate, x_norm, z, h->a, h->b);
            gpu_matvec(&d->ssm_beta, x_norm, beta, h->a, h->b);
            gpu_matvec(&d->ssm_alpha, x_norm, alpha, h->a, h->b);
            /* beta = sigmoid(beta), on device. */
            fc_sigmoid_ctx(ctx, beta, n_v_heads);
            /* Decay gate g = ssm_a * softplus(alpha + ssm_dt), on device. */
            fc_ssm_decay_gate_ctx(ctx, alpha, d->ssm_dt.buf, d->ssm_a.buf,
                                  gbuf, n_v_heads);
        }
    }

    /* 3. Causal conv1d + SiLU over the conv_dim stream; rolls conv_state. */
    fc_ssm_conv1d_silu_ctx(ctx, qkv, d->ssm_conv1d.buf, conv_state_g, conv_out,
                           conv_dim, cfg->ssm_conv);

    /* 3b. q/k per-head L2-norm + GQA interleaved-repeat (vh = r*n_k_heads + kh),
     * v straight through — on device. Mirrors the CPU ssm / ggml_repeat. */
    fc_ssm_qknorm_gqa_ctx(ctx, conv_out, qg, kg, vg, n_k_heads, kv_groups, S,
                          key_dim, val_dim, eps);

    /* 4. Per-token delta-rule recurrence (scale = 1/sqrt(S)). On device. */
    fc_ssm_recurrence_ctx(ctx, ssm_state_g, qg, kg, vg, gbuf, beta, y, n_v_heads,
                          S, qw_scale_inv_sqrt(S));

    /* 5. gated = RMSNorm(y, ssm_norm) * silu(z), then output projection with
     * the +residual folded into the epilogue (out_buf keeps the pre-residual
     * block output for the debug dumps; hidden += out_buf). */
    fc_ssm_gated_norm_ctx(ctx, y, d->ssm_norm.buf, z, gated, n_v_heads, S, eps);
    {
        fc_fused_seg seg = gpu_seg_mv(&d->ssm_out, out_buf, 3);
        seg.aux1 = hidden;
        if (!(gpu_fuse_mask() & 2) ||
            gpu_fused_run(ctx, &d->fx_out, &seg, 1, gated) != 0) {
            gpu_matvec(&d->ssm_out, gated, out_buf, h->a, h->b);
            fc_add_ctx(ctx, hidden, out_buf, hidden, cfg->n_embd);
        }
    }

    return 0;
}

/* On-die decode flash attention — defined after gpu_attn_kv_cache_sync (which
 * it reuses). Returns 0 = ran, 1 = unavailable, <0 = error. */
static int gpu_attn_block_decode_flash(gpu_attn_layer *d,
                                       const pn_qwen35moe_config *cfg,
                                       fc_buffer *x_norm, pn_qwen35moe_kv *kv,
                                       int pos, gpu_die_scratch *sc,
                                       gpu_host_scratch *h);

/* Full-step segment graph (C1) — defined after the flash-decode block whose
 * machinery it reuses. Runs die segment [lo, hi) as ONE captured CUDA graph
 * (replaying, or lazily capturing at pos >= 1). Returns 0 = segment ran via
 * the graph, 1 = not eligible (caller runs the per-layer loop), -1 = error. */
static int gpu_segment_forward(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                               fc_cuda_ctx *ctx, int seg_idx, int lo, int hi,
                               fc_buffer *hidden, fc_buffer *x_norm,
                               fc_buffer *block_out, pn_qwen35moe_state *s,
                               int pos, gpu_host_scratch *h);

/* Phase 10b (lazy host KV mirror) — forward decls; definitions live with
 * gpu_attn_kv_cache_sync below. */
static int qw_kv_mirror_lazy(void);
static int qw_kv_layer_materialize(gpu_attn_layer *d, pn_qwen35moe_kv *kv,
                                   int rows);
static size_t qw_kv_row_bytes(const pn_qwen35moe_kv *kv);
static size_t qw_kv_sink_bytes(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv);
static size_t qw_kv_win_bytes(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv);
static int gpu_attn_slot_ring_ensure(gpu_attn_layer *d,
                                     const pn_qwen35moe_config *cfg,
                                     int q4, int max_seq, int nslots,
                                     int kv_idx);

/* ============================================================
 * Full-attention block on the GPU
 *
 * Mirrors pn_qwen35moe_attn_block: joint Q+gate proj + split, QK-RMSNorm,
 * partial NEOX RoPE, GQA softmax over the KV window, sigmoid output gate,
 * output proj. Projections/RoPE/QK-norm run on-device; the Q/gate split, the
 * GQA softmax attention, and the sigmoid gate are host-side glue (TODO kernels).
 * ============================================================ */

static int gpu_attn_block(gpu_attn_layer *d, const pn_qwen35moe_config *cfg,
                          fc_buffer *x_norm, pn_qwen35moe_kv *kv,
                          int pos, fc_buffer *out_buf,
                          fc_buffer *hidden, gpu_die_scratch *sc,
                          gpu_host_scratch *h)
{
    /* Phase 10b (lazy mirror): this host-softmax path reads the host KV
     * mirror [0, pos). If earlier steps ran on-die with the lazy mirror the
     * rows live only in a GPU ring — pull them down first. Once this path
     * runs it keeps the mirror current itself (it appends every token). */
    if (kv->kv_mirror_stale &&
        qw_kv_layer_materialize(d, kv,
                                (pos < kv->max_seq) ? pos : kv->max_seq) != 0) {
        fprintf(stderr, "[attn] host attn: mirror stale with no ring source "
                "(pos=%d) — failing step\n", pos);
        return -1;
    }
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int n_kv_groups = n_head / n_head_kv;
    const int q_stride = head_dim * 2;
    const float eps = cfg->rms_norm_eps;
    const int max_seq = kv->max_seq;
    float *k_cache = kv->k;      /* fp32 mode; NULL under q4 */
    float *v_cache = kv->v;
    const int kv_q4 = kv->q4;
    const int kv_sink = kv_q4 ? pn_kvq4_sink() : 0;
    const int kv_win = kv_q4 ? pn_kvq4_win() : 0;
    fc_cuda_ctx *ctx = d->ctx;   /* this layer's die context */

    /* Persistent per-die scratch (every buffer fully overwritten below). */
    fc_buffer *qg = sc->a_qg, *qbuf = sc->a_q, *kbuf = sc->a_k;
    fc_buffer *vbuf = sc->a_v, *attn_out = sc->a_out;

    /* On-die decode flash attention (QWEN35_FLASH_DECODE): the whole block runs
     * on the die (device split -> QK-norm/RoPE -> KV append -> split flash ->
     * gate), leaving attn_out gated in sc->a_out. Returns 0 = ran (skip to the
     * output projection), 1 = unavailable (fall through to the host-softmax
     * path below), <0 = error. */
    int df = gpu_attn_block_decode_flash(d, cfg, x_norm, kv, pos, sc, h);
    if (df < 0)
        return -1;
    if (df == 0)
        goto out_proj;

    /* 1. Joint Q+gate, K and V projections — ONE fused launch (all three read
     * x_norm; K/V moving ahead of the Q split touches disjoint buffers). */
    {
        fc_fused_seg segs[3];
        segs[0] = gpu_seg_mv(&d->attn_q, qg, 0);
        segs[1] = gpu_seg_mv(&d->attn_k, kbuf, 0);
        segs[2] = gpu_seg_mv(&d->attn_v, vbuf, 0);
        if (!(gpu_fuse_mask() & 4) ||
            gpu_fused_run(ctx, &d->fx_qkv, segs, 3, x_norm) != 0) {
            gpu_matvec(&d->attn_q, x_norm, qg, h->a, h->b);
            gpu_matvec(&d->attn_k, x_norm, kbuf, h->a, h->b);
            gpu_matvec(&d->attn_v, x_norm, vbuf, h->a, h->b);
        }
    }

    /* Split per head: first head_dim = Q, second head_dim = gate z. TODO kernel:
     * the strided Q/gate split. Host-side; q -> qbuf on device, z kept in h->c. */
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu(qg, h->a);
    {
        float *qonly = h->b;        /* [n_head*head_dim] */
        float *zgate = h->c;        /* [n_head*head_dim] */
        for (int hd = 0; hd < n_head; hd++) {
            const float *src = h->a + (size_t)hd * q_stride;
            memcpy(qonly + (size_t)hd * head_dim, src, (size_t)head_dim * sizeof(float));
            memcpy(zgate + (size_t)hd * head_dim, src + head_dim,
                   (size_t)head_dim * sizeof(float));
        }
        fc_buffer_from_cpu(qbuf, qonly);
        /* zgate stays in h->c for the sigmoid output gate below. */
    }

    /* 2.-4. Per-head QK-RMSNorm + partial NEOX RoPE on Q and K — ONE fused
     * launch (was 4; per-head math identical). V is not normed/roped. */
    if (!(gpu_fuse_mask() & 8) ||
        fc_attn_qknorm_rope_ctx(ctx, qbuf, kbuf, d->attn_q_norm.buf,
                                d->attn_k_norm.buf, n_head, n_head_kv,
                                head_dim, eps, cfg->rope_dim, pos,
                                cfg->rope_theta) != 0) {
        fc_qk_rms_norm_ctx(ctx, qbuf, d->attn_q_norm.buf, n_head, head_dim, eps);
        fc_qk_rms_norm_ctx(ctx, kbuf, d->attn_k_norm.buf, n_head_kv, head_dim, eps);
        fc_rope_neox_partial_ctx(ctx, qbuf, n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        fc_rope_neox_partial_ctx(ctx, kbuf, n_head_kv, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
    }

    /* 5. Append this token's K/V to the host-side cache, then GQA-attend.
     * TODO kernel: the KV-cache append + GQA softmax attention over the window
     * (no fc kernel for windowed attention; fc_softmax is per-row only). Done
     * host-side on read-backs of qbuf/kbuf/vbuf. The KV cache mirrors the CPU
     * pn_qwen35moe_kv layout [n_head_kv][max_seq][head_dim]. */
    {
        double tah0 = 0.0;
        if (prof_on()) { fc_sync_ctx(ctx); tah0 = prof_now(); }
        int cache_pos = pos % max_seq;
        float *kvec = h->a;      /* [n_head_kv*head_dim] */
        float *vvec = h->b;
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu(kbuf, kvec);
        fc_buffer_to_cpu(vbuf, vvec);
        for (int hd = 0; hd < n_head_kv; hd++) {
            if (kv_q4) {
                pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd, max_seq,
                                  cache_pos, kvec + (size_t)hd * head_dim,
                                  head_dim, kv_sink, kv_win);
                pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd, max_seq,
                                  cache_pos, vvec + (size_t)hd * head_dim,
                                  head_dim, kv_sink, kv_win);
            } else {
                memcpy(k_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                       kvec + (size_t)hd * head_dim, (size_t)head_dim * sizeof(float));
                memcpy(v_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                       vvec + (size_t)hd * head_dim, (size_t)head_dim * sizeof(float));
            }
        }

        int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
        float *qhost = h->b;     /* reuse: read q back */
        fc_buffer_to_cpu(qbuf, qhost);
        float *aout = h->a;      /* [n_head*head_dim] */
        float scale = qw_scale_inv_sqrt(head_dim);
        /* per-head GQA softmax attention (mirrors qwen35moe_compute_heads);
         * under q4 the K/V rows are dequantized on the fly. scores is heap
         * scratch (seq_len can reach the 256K cap — too large for the stack). */
        float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
        float krow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        float vrow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        if (!scores) { if (prof_on()) prof_phase[1] += prof_now() - tah0; return -1; }
        for (int qh = 0; qh < n_head; qh++) {
            int kv_h = qh / n_kv_groups;
            const float *q_vec = qhost + (size_t)qh * head_dim;
            const float *k_base = kv_q4 ? NULL :
                k_cache + (size_t)kv_h * max_seq * head_dim;
            const float *v_base = kv_q4 ? NULL :
                v_cache + (size_t)kv_h * max_seq * head_dim;
            float *out_vec = aout + (size_t)qh * head_dim;
            for (int s = 0; s < seq_len; s++) {
                const float *k_vec;
                if (kv_q4) {
                    pn_kvq4_load_row(kv->kq, kv->k_sink, kv->k_win, kv_h,
                                     max_seq, s, pos, krow, head_dim,
                                     kv_sink, kv_win);
                    k_vec = krow;
                } else {
                    k_vec = k_base + (size_t)s * head_dim;
                }
                scores[s] = pn_dot_f32(q_vec, k_vec, head_dim) * scale;
            }
            float mx = scores[0];
            for (int s = 1; s < seq_len; s++)
                if (scores[s] > mx) mx = scores[s];
            float sum_exp = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                scores[s] = expf(scores[s] - mx);
                sum_exp += scores[s];
            }
            float inv_sum = 1.0f / sum_exp;
            memset(out_vec, 0, (size_t)head_dim * sizeof(float));
            for (int s = 0; s < seq_len; s++) {
                float w_s = scores[s] * inv_sum;
                const float *v_vec;
                if (kv_q4) {
                    pn_kvq4_load_row(kv->vq, kv->v_sink, kv->v_win, kv_h,
                                     max_seq, s, pos, vrow, head_dim,
                                     kv_sink, kv_win);
                    v_vec = vrow;
                } else {
                    v_vec = v_base + (size_t)s * head_dim;
                }
                for (int dd = 0; dd < head_dim; dd++)
                    out_vec[dd] += w_s * v_vec[dd];
            }
        }
        free(scores);

        /* 6. Output gate: attn_out *= sigmoid(z). TODO kernel: sigmoid gate.
         * z is in h->c (set above). */
        int attn_dim = n_head * head_dim;
        for (int i = 0; i < attn_dim; i++) {
            float gv = 1.0f / (1.0f + expf(-h->c[i]));
            aout[i] *= gv;
        }
        fc_buffer_from_cpu(attn_out, aout);
        if (prof_on()) prof_phase[1] += prof_now() - tah0;
    }

    /* 7. Output projection -> [n_embd] with the +residual folded into the
     * epilogue (out_buf keeps the pre-residual output for the debug dumps). */
out_proj:
    {
        fc_fused_seg seg = gpu_seg_mv(&d->attn_output, out_buf, 3);
        seg.aux1 = hidden;
        if (gpu_fused_run(ctx, &d->fx_out, &seg, 1, attn_out) != 0) {
            gpu_matvec(&d->attn_output, attn_out, out_buf, h->a, h->b);
            fc_add_ctx(ctx, hidden, out_buf, hidden, cfg->n_embd);
        }
    }

    return 0;
}

/* ============================================================
 * Shared-expert MoE FFN on the GPU
 *
 * Mirrors pn_qwen35moe_moe_block. Router matvec + softmax/top-k + per-expert
 * SwiGLU + shared expert run via fc_* kernels; the gate logit dot is host-side.
 * ============================================================ */

/* SwiGLU for one expert on device: out = (silu(gate.x) * (up.x)) . down. All
 * buffers + weights belong to ctx (this layer's die). */
static void gpu_swiglu_expert(fc_cuda_ctx *ctx, const gpu_weight *gate,
                              const gpu_weight *up, const gpu_weight *down,
                              fc_buffer *x, int ff, fc_buffer *t_gate,
                              fc_buffer *t_up, fc_buffer *t_hid, fc_buffer *out,
                              gpu_host_scratch *h)
{
    gpu_matvec(gate, x, t_gate, h->a, h->b);
    gpu_matvec(up, x, t_up, h->a, h->b);
    fc_silu_ctx(ctx, t_gate, t_gate, ff);     /* silu(gate) */
    fc_mul_ctx(ctx, t_gate, t_up, t_hid, ff); /* * up */
    gpu_matvec(down, t_hid, out, h->a, h->b);
}

static int gpu_moe_block(fc_cuda_ctx *ctx, gpu_moe *d,
                         const pn_qwen35moe_config *cfg, fc_buffer *x_norm,
                         fc_buffer *out_buf, fc_buffer *hidden,
                         gpu_die_scratch *sc, gpu_host_scratch *h)
{
    const int n_embd = cfg->n_embd;
    const int n_expert = cfg->n_expert;
    const int k = cfg->n_expert_used;
    const int expert_ff = cfg->expert_ff;
    const int shared_ff = cfg->shared_ff;

    /* Persistent per-die scratch (every buffer fully overwritten below;
     * moe_out is explicitly zeroed before accumulation). */
    fc_buffer *logits = sc->m_logits, *top_idx = sc->m_tidx, *top_wt = sc->m_twt;
    fc_buffer *t_gate = sc->m_gate, *t_up = sc->m_up, *t_hid = sc->m_hid;
    fc_buffer *exp_out = sc->m_exp, *moe_out = sc->m_out, *shexp = sc->m_shexp;

    /* Dense-FFN (Bonsai): a single plain SwiGLU, no router/experts/scalar-gate.
     * gate/up/down are in the shexp slots; ffn dim = ffn_length. Mirrors the
     * CPU dense path (qwen35moe_dense_ffn) and the shared-expert unfused
     * sequence minus the sigmoid gate, then adds the residual. */
    if (cfg->dense_ffn) {
        const int ff = cfg->ffn_length;
        gpu_matvec(&d->ffn_gate_shexp, x_norm, t_gate, h->a, h->b);
        gpu_matvec(&d->ffn_up_shexp, x_norm, t_up, h->a, h->b);
        fc_moe_silu_mul_ctx(ctx, t_gate, t_up, t_gate, ff);   /* silu(gate)*up */
        gpu_matvec(&d->ffn_down_shexp, t_gate, out_buf, h->a, h->b);
        fc_add_ctx(ctx, hidden, out_buf, hidden, n_embd);      /* residual */
        return 0;
    }

    /* 1. Fused MoE prologue — ONE launch for everything that reads x_norm:
     * the router logits, the shared-expert scalar-gate dot (sigmoid'd into
     * m_sgate), and the shared expert's gate+up SwiGLU (silu(g.x)*(u.x)
     * straight into t_gate; one warp computes both dots, replacing gate
     * matvec + up matvec + silu_mul). Only on the grouped path — the
     * per-expert fallback reuses t_gate for the routed experts. */
    int pre_fused = 0;
    if (d->grouped && (gpu_fuse_mask() & 16) &&
        d->ffn_gate_shexp.type == d->ffn_up_shexp.type &&
        d->ffn_gate_shexp.rows == d->ffn_up_shexp.rows &&
        d->ffn_gate_shexp.row_stride == d->ffn_up_shexp.row_stride) {
        double tp0 = 0.0;
        if (prof_on()) { fc_sync_ctx(ctx); tp0 = prof_now(); }
        fc_fused_seg segs[3];
        segs[0] = gpu_seg_mv(&d->ffn_gate_inp, logits, 0);
        memset(&segs[1], 0, sizeof(segs[1]));      /* scalar-gate dot */
        segs[1].w = d->ffn_gate_inp_shexp.buf;
        segs[1].out = sc->m_sgate;
        segs[1].rows = 1;
        segs[1].cols = n_embd;
        segs[1].row_stride = n_embd * (int)sizeof(float);
        segs[1].kind = 1;
        segs[1].ep = 1;                            /* sigmoid */
        segs[2] = gpu_seg_mv(&d->ffn_gate_shexp, t_gate, 0);
        segs[2].w2 = d->ffn_up_shexp.bytes;
        segs[2].kind = 2;                          /* gate+up SwiGLU pair */
        pre_fused = (gpu_fused_run(ctx, &d->fx_pre, segs, 3, x_norm) == 0);
        if (prof_on()) { fc_sync_ctx(ctx); prof_phase[9] += prof_now() - tp0; }
    }

    /* Router matvec (unfused path) + softmax/top-k/norm_topk_prob/
     * expert_weights_scale on device. */
    if (!pre_fused)
        gpu_matvec(&d->ffn_gate_inp, x_norm, logits, h->a, h->b);
    fc_moe_router_topk_ctx(ctx, logits, top_idx, top_wt, n_expert, k,
                           cfg->expert_weights_scale);

    if (d->grouped) {
        /* Grouped fast path: the whole routed-expert block in two launches
         * that read top_idx/top_wt on device — no sync, no read-back, no
         * zero pass. hid[s*ff+row] then out[row] = sum_s wt_s * down_e_s.hid_s
         * (bit-identical reduction/accumulate order to the per-expert path). */
        double tg0 = 0.0;
        if (prof_on()) { fc_sync_ctx(ctx); tg0 = prof_now(); }
        fc_moe_gateup_ctx(ctx, (int)d->gate_all.type, d->gate_all.bytes,
                          d->up_all.bytes, x_norm, top_idx, t_hid, k,
                          expert_ff, n_embd, d->gate_all.row_stride);
        if (prof_on()) {
            fc_sync_ctx(ctx);
            double tn = prof_now();
            prof_phase[7] += tn - tg0;
            tg0 = tn;
        }
        fc_moe_down_accum_ctx(ctx, (int)d->down_all.type, d->down_all.bytes,
                              t_hid, top_idx, top_wt, moe_out, k, n_embd,
                              expert_ff, d->down_all.row_stride);
        if (prof_on()) { fc_sync_ctx(ctx); prof_phase[8] += prof_now() - tg0; }
    } else {
        /* Per-expert fallback (off-menu expert layouts): read the selection
         * back and launch each expert's SwiGLU separately. */
        memset(h->a, 0, (size_t)n_embd * sizeof(float));
        fc_buffer_from_cpu(moe_out, h->a);

        int32_t idx_host[PN_QWEN35MOE_GPU_MAX_TOPK];
        float wt_host[PN_QWEN35MOE_GPU_MAX_TOPK];
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu(top_idx, idx_host);
        fc_buffer_to_cpu(top_wt, wt_host);

        for (int s = 0; s < k; s++) {
            int e = idx_host[s];
            if (e < 0 || e >= n_expert)
                continue;
            gpu_swiglu_expert(ctx, &d->gate_exps[e], &d->up_exps[e],
                              &d->down_exps[e], x_norm, expert_ff, t_gate,
                              t_up, t_hid, exp_out, h);
            fc_moe_weighted_add_ctx(ctx, exp_out, moe_out, wt_host[s], n_embd);
        }
    }

    /* 2.+3. Shared-expert epilogue + combine. Fused path: ONE launch — the
     * shexp down projection with epilogue out_buf[r] = moe_out[r] +
     * dot*sigmoid_gate, hidden[r] += out_buf[r] (replaces down matvec +
     * dot_gate + combine add + the caller's residual add). Fallback: the
     * original sequence plus the residual add moved in from the caller. */
    double tsx0 = 0.0;
    if (prof_on()) { fc_sync_ctx(ctx); tsx0 = prof_now(); }
    int post_fused = 0;
    if (pre_fused && (gpu_fuse_mask() & 32)) {
        fc_fused_seg seg = gpu_seg_mv(&d->ffn_down_shexp, out_buf, 4);
        seg.aux0 = moe_out;
        seg.aux1 = hidden;
        seg.aux2 = sc->m_sgate;
        post_fused = (gpu_fused_run(ctx, &d->fx_post, &seg, 1, t_gate) == 0);
    }
    if (!post_fused) {
        if (!pre_fused) {
            gpu_matvec(&d->ffn_gate_shexp, x_norm, t_gate, h->a, h->b);
            gpu_matvec(&d->ffn_up_shexp, x_norm, t_up, h->a, h->b);
            fc_moe_silu_mul_ctx(ctx, t_gate, t_up, t_gate, shared_ff);
        }
        gpu_matvec(&d->ffn_down_shexp, t_gate, shexp, h->a, h->b);
        fc_moe_shexp_dot_gate_ctx(ctx, d->ffn_gate_inp_shexp.buf, x_norm,
                                  shexp, n_embd);
        fc_add_ctx(ctx, moe_out, shexp, out_buf, n_embd);
        fc_add_ctx(ctx, hidden, out_buf, hidden, n_embd);
    }
    if (prof_on()) { fc_sync_ctx(ctx); prof_phase[9] += prof_now() - tsx0; }

    return 0;
}

/* ============================================================
 * Per-layer forward (the sandwich) + the full token forward
 * ============================================================ */

/* One SSM layer's COMPLETE launch sequence: attn_norm -> SSM block (+residual
 * fused into its output epilogue) -> post_attention_norm -> shared-MoE
 * (+residual fused). Every op is an async launch on the layer's stream over
 * persistent buffers with position-invariant grids/args, so the whole body is
 * CUDA-graph-capturable — one cuGraphLaunch replays it exactly (decode is
 * launch-bound). The dbg dumps and prof timers inside are no-ops unless their
 * env knobs are set, and those knobs disable graph capture (they sync
 * mid-body). */
static int gpu_ssm_layer_body(gpu_ssm_layer *d, const pn_qwen35moe_config *cfg,
                              int layer, fc_buffer *hidden, fc_buffer *x_norm,
                              fc_buffer *block_out, gpu_die_scratch *sc,
                              gpu_host_scratch *h)
{
    fc_cuda_ctx *ctx = d->ctx;
    int c = cfg->n_embd;
    int rc;

    fc_rms_norm_ctx(ctx, hidden, d->attn_norm.buf, x_norm, c, cfg->rms_norm_eps);
    qwen35moe_dbg_gpu(ctx, "attn_norm", layer, x_norm, c, h->a);

    double tb0 = prof_on() ? prof_now() : 0.0;
    rc = gpu_ssm_block(d, cfg, x_norm, d->conv_state_g, d->ssm_state_g,
                       block_out, hidden, sc, h);
    if (prof_on()) { fc_sync_ctx(ctx); prof_phase[5] += prof_now() - tb0; }
    if (rc < 0)
        return rc;

    /* block_out keeps the pre-residual output so the debug dumps still line
     * up with the CPU reference. */
    qwen35moe_dbg_gpu(ctx, "block_out", layer, block_out, c, h->a);
    qwen35moe_dbg_gpu(ctx, "attn_residual", layer, hidden, c, h->a);

    /* Shared-expert MoE on the post_attention_norm'd hidden state. */
    fc_rms_norm_ctx(ctx, hidden, d->post_attention_norm.buf, x_norm, c,
                    cfg->rms_norm_eps);
    qwen35moe_dbg_gpu(ctx, "attn_post_norm", layer, x_norm, c, h->a);
    double tm0 = prof_on() ? prof_now() : 0.0;
    rc = gpu_moe_block(ctx, &d->moe, cfg, x_norm, block_out, hidden, sc, h);
    if (prof_on()) { fc_sync_ctx(ctx); prof_phase[2] += prof_now() - tm0; }
    if (rc < 0)
        return rc;
    qwen35moe_dbg_gpu(ctx, "ffn_out", layer, block_out, c, h->a);
    qwen35moe_dbg_gpu(ctx, "l_out", layer, hidden, c, h->a);

    return 0;
}


/* Seed this SSM layer's SHARED live buffers with state `st` at position `pos`.
 * A decode slot row whose image is exactly this state's at this position (the
 * session's own block on the card) is the preferred source — device to device,
 * so a resumed turn never re-reads its recurrent state from host RAM; the host
 * mirror is used only for a fresh state (pos 0) or when no row holds it. */
static void qw_ssm_seed_shared(pn_qwen35moe_gpu *g, gpu_ssm_layer *d,
                               const pn_qwen35moe_ssm_state *st, int pos)
{
    const int idx = qw_ssm_hidx(g, (int)(d - g->ssm));
    const pn_qwen35moe_config *dc = &g->tp_cfg;   /* read only for a TP half */
    for (int r = 0; pos > 0 && r < g->n_slots && r < d->slots_cap; r++) {
        const pn_qwen35moe_state *ps =
            (const pn_qwen35moe_state *)g->slot_state[r];
        if (!ps || &ps->ssm[idx] != st || g->slot_pos[r] != pos ||
            g->slot_gen[r] != (ps->n_attn > 0 ? ps->kv[0].content_gen : 0u) ||
            !d->conv_slots || !d->ssm_slots)
            continue;
        /* QWEN35_TP: a half's images are its local dims. */
        size_t conv_b = (size_t)st->conv_minus_1 *
                        (d->tp_on ? dc->conv_dim : st->conv_dim) * sizeof(float);
        size_t ssm_b = (size_t)st->state_size * st->state_size *
                       (d->tp_on ? dc->ssm_dt_rank : st->n_v_heads) * sizeof(float);
        fc_buffer_copy_d2d_ctx(d->ctx, d->conv_state_g, 0, d->conv_slots,
                               (size_t)r * conv_b, conv_b);
        fc_buffer_copy_d2d_ctx(d->ctx, d->ssm_state_g, 0, d->ssm_slots,
                               (size_t)r * ssm_b, ssm_b);
        d->host_state = (const void *)st;
        return;
    }
    qw_ssm_xfer(g, d, (pn_qwen35moe_ssm_state *)st, 1);
    d->host_state = (const void *)st;
}

/* Run one layer on the GPU: attn_norm -> (SSM|attn) -> +residual ->
 * post_attention_norm -> shared-MoE -> +residual. hidden is the GPU residual
 * stream (updated in place). x_norm/block_out are reusable GPU scratch. All of
 * hidden/x_norm/block_out MUST belong to `ctx` — this layer's die context, the
 * caller swaps the barswap slot to that die before calling.
 *
 * SSM layers replay a captured CUDA graph of their whole launch sequence
 * (captured lazily at pos >= 1, after token 0 built the fused descriptors);
 * attention layers keep the direct path HERE (fallback). When the segment
 * graph is live (gpu_segment_forward, QWEN35_GRAPH_FULL) this function is
 * bypassed entirely — the whole die segment, attention layers included,
 * replays as one graph via the device-pos flash-decode kernels. */
static int gpu_layer_forward(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                             fc_cuda_ctx *ctx, int layer, fc_buffer *hidden,
                             fc_buffer *x_norm, fc_buffer *block_out,
                             pn_qwen35moe_state *s, int pos, gpu_host_scratch *h)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    int rc;

    gpu_die_scratch *sc = gpu_get_scratch(g, ctx, cfg);
    if (!sc)
        return -1;

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx];
        fc_rms_norm_ctx(ctx, hidden, d->attn_norm.buf, x_norm, c,
                        cfg->rms_norm_eps);
        qwen35moe_dbg_gpu(ctx, "attn_norm", layer, x_norm, c, h->a);
        pn_qwen35moe_kv *kv = &s->kv[idx];
        double tb0 = prof_on() ? prof_now() : 0.0;
        rc = gpu_attn_block(d, cfg, x_norm, kv, pos,
                            block_out, hidden, sc, h);
        if (prof_on()) { fc_sync_ctx(ctx); prof_phase[6] += prof_now() - tb0; }
        if (rc < 0)
            return rc;
        qwen35moe_dbg_gpu(ctx, "block_out", layer, block_out, c, h->a);
        qwen35moe_dbg_gpu(ctx, "attn_residual", layer, hidden, c, h->a);

        fc_rms_norm_ctx(ctx, hidden, d->post_attention_norm.buf, x_norm, c,
                        cfg->rms_norm_eps);
        qwen35moe_dbg_gpu(ctx, "attn_post_norm", layer, x_norm, c, h->a);
        double tm0 = prof_on() ? prof_now() : 0.0;
        rc = gpu_moe_block(ctx, &d->moe, cfg, x_norm, block_out, hidden, sc, h);
        if (prof_on()) { fc_sync_ctx(ctx); prof_phase[2] += prof_now() - tm0; }
        if (rc < 0)
            return rc;
        qwen35moe_dbg_gpu(ctx, "ffn_out", layer, block_out, c, h->a);
        qwen35moe_dbg_gpu(ctx, "l_out", layer, hidden, c, h->a);
        d->body_ran = 1;   /* segment-graph gate: fused descs/scratch built */
        return 0;
    }

    /* ---- SSM layer. ---- */
    gpu_ssm_layer *d = &g->ssm[idx];
    pn_qwen35moe_ssm_state *st = &s->ssm[idx];

    /* Persistent GPU-resident dual state (shadows the host copies in
     * s->ssm[idx]; the host mirror is NOT read back per token — enable
     * QWEN35_DEBUG bisection only with the CPU path or accept that the host
     * copy goes stale). Allocated ONCE on this layer's die; a state swap or
     * a sequence restart at pos 0 re-seeds the CONTENTS with an H2D copy so
     * the buffer addresses stay stable for the captured graph. */
    size_t conv_n = (size_t)st->conv_minus_1 * st->conv_dim;
    size_t ssm_n = (size_t)st->state_size * st->state_size * st->n_v_heads;
    double ts0 = prof_on() ? prof_now() : 0.0;
    if (!d->conv_state_g)
        d->conv_state_g = fc_buffer_alloc_ctx(ctx, conv_n * sizeof(float));
    if (!d->ssm_state_g)
        d->ssm_state_g = fc_buffer_alloc_ctx(ctx, ssm_n * sizeof(float));
    if (!d->conv_state_g || !d->ssm_state_g)
        return -1;
    if (d->host_state != (const void *)st || pos == 0)
        qw_ssm_seed_shared(g, d, st, pos);
    if (prof_on()) { fc_sync_ctx(ctx); prof_phase[0] += prof_now() - ts0; }

    /* Fast path: replay the layer's captured graph (~13 launches -> 1). */
    if (d->graph) {
        if (fc_graph_launch(d->graph) == 0)
            return 0;
        /* Replay failed — the graph is no use to anyone. Drop it and re-arm a
         * fresh capture rather than re-launching a dead graph every token. */
        fc_graph_free(d->graph);
        d->graph = NULL;
        pn_graph_capture_dropped(&d->graph_capture);
    }

    /* Capture lazily: only after a DIRECT body run (body_ran) built the fused
     * descriptors and per-die scratch (those creations alloc/copy and are not
     * capturable — see body_ran). prof/dbg sync mid-body — direct path only.
     * The non-grouped MoE fallback reads the expert selection back to the host
     * mid-block, so it is not capturable either. */
    int graph_gate_ok = gpu_graph_enabled() && !prof_on() &&
                        !qwen35moe_dbg_enabled() &&
                        (d->moe.grouped || cfg->dense_ffn);
    if (!graph_gate_ok) {
        pn_graph_capture_ineligible(&d->graph_capture,
            !gpu_graph_enabled() ? PN_GRAPH_INELIGIBLE_GRAPH_OFF
            : (prof_on() || qwen35moe_dbg_enabled())
                                 ? PN_GRAPH_INELIGIBLE_PROFILING
                                 : PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);
    }

    /* pos>=1 && body_ran are warm-up, not ineligibility — they clear by
     * themselves, so the tracker stays PENDING through them rather than
     * reporting a healthy layer as "will never capture". */
    if (graph_gate_ok && pos >= 1 && d->body_ran &&
        pn_graph_capture_should_attempt(&d->graph_capture)) {
        int captured = 0;
        if (fc_graph_begin_ctx(ctx) == 0) {
            rc = gpu_ssm_layer_body(d, cfg, layer, hidden, x_norm, block_out,
                                    sc, h);
            d->graph = fc_graph_end_ctx(ctx);
            if (pn_graph_capture_inject_now(&d->graph_capture) && d->graph) {
                fc_graph_free(d->graph);   /* QWEN35_GRAPH_CAPTURE_FAIL_INJECT */
                d->graph = NULL;
            }
            /* The recorded launches did NOT execute — replay the graph now,
             * or fall through and run the body directly. */
            if (rc == 0 && d->graph && fc_graph_launch(d->graph) == 0)
                captured = 1;
            else if (d->graph) {
                fc_graph_free(d->graph);
                d->graph = NULL;
            }
        }
        if (captured) {
            pn_graph_capture_succeeded(&d->graph_capture);
            return 0;
        }
        pn_graph_capture_failed(&d->graph_capture);
    }

    rc = gpu_ssm_layer_body(d, cfg, layer, hidden, x_norm, block_out, sc, h);
    if (rc == 0)
        d->body_ran = 1;
    return rc;
}

/* Decode-time logits fetch mode. Default is device top-k (k = 40, enough for
 * both greedy argmax and the serving sampler's top_k=40): the 248k-vocab
 * logits never cross the 0.4 GB/s host link — the die reduces them to k
 * (id, logit) pairs (~4 ms/token -> microseconds).
 *   QWEN35_GPU_TOPK=<k>        override k (1 = pure argmax), clamped to
 *                              [1, FC_TOPK_MAX]
 *   QWEN35_GPU_FULL_LOGITS=1   restore the full n_vocab D2H copy (anything
 *                              that needs ALL logits)
 * QWEN35_DEBUG also forces the full path so the result_output dump stays
 * real. Returns 0 for full mode. */
/* The per-state survivor-id array must be able to hold a full k. */
_Static_assert(PN_QWEN35MOE_TOPK_MAX >= FC_TOPK_MAX,
               "PN_QWEN35MOE_TOPK_MAX must cover FC_TOPK_MAX survivors");

/* Does any state in this batch need the full logits row (constrained decoding)?
 * The die reduces every slot in one kernel, so the batch is all-or-nothing; full
 * logits are a superset of top-k, so answering "yes" for the whole step is
 * correct for the unconstrained slots too — they just get more than they need,
 * and only while a constrained request is actually in flight. */
static int batch_needs_full_logits(pn_qwen35moe_state *const *sts, int n)
{
    for (int i = 0; i < n; i++)
        if (sts[i] && sts[i]->full_logits) return 1;
    return 0;
}

static int gpu_topk_k(void)
{
    static int v = -1;
    if (v < 0) {
        if (getenv("QWEN35_GPU_FULL_LOGITS") || qwen35moe_dbg_enabled()) {
            v = 0;
        } else {
            v = 40;
            const char *env = getenv("QWEN35_GPU_TOPK");
            if (env) {
                v = atoi(env);
                if (v < 0) v = 0;
                if (v > FC_TOPK_MAX) v = FC_TOPK_MAX;
            }
        }
    }
    return v;
}

/* Present k (id, val) survivors sparsely in s->logits.
 *
 * The sparse contract: s->logits is -inf everywhere except this step's k
 * survivors. To avoid an n_vocab -inf memset per token it is filled in full
 * ONCE and thereafter patched incrementally — clear the ids the PREVIOUS fetch
 * wrote, write this fetch's.  That is only sound if the bookkeeping describes
 * every write the buffer has seen, which is why it lives on the state that
 * OWNS the buffer (s->topk_*) and not on the shared per-model GPU context.
 *
 * It used to live on the GPU context, in two independent copies: g->topk_*
 * (used by the prefill / single-user fetch) and g->slot_topk_*[slot] (used by
 * the batched decode fetch).  Both maintained the SAME s->logits buffer while
 * each cleared only its OWN previous ids, so whatever the other last wrote
 * lingered as phantom finite entries — forever.  A turn that ended on EOS left
 * EOS behind at its real logit; the next turn's prefill did not clear it (the
 * prefill tracker's "already filled" guard still matched the buffer address),
 * EOS won at step 0, and the turn emitted ZERO tokens with finish_reason
 * "stop" — which, generating nothing, left the phantom in place and latched
 * the model until reload.  One tracker per buffer, owned by the buffer, is the
 * fix: every writer now clears exactly what the previous writer left. */
static void gpu_present_topk(pn_qwen35moe_state *s, const int *ids,
                             const float *vals, int k, int n_vocab)
{
    float *dst = s->logits;

    s->short_row_live = 0;          /* a short row is marked by its presenter, after this */
    if (!s->topk_filled) {          /* first fetch into this state's buffer */
        for (int i = 0; i < n_vocab; i++)
            dst[i] = -INFINITY;
        s->topk_filled = 1;
        s->topk_prev_n = 0;
    }
    for (int i = 0; i < s->topk_prev_n; i++)
        dst[s->topk_prev_ids[i]] = -INFINITY;
    for (int i = 0; i < k; i++)
        dst[ids[i]] = vals[i];
    memcpy(s->topk_prev_ids, ids, (size_t)k * sizeof(int));
    s->topk_prev_n = k;
}

/* Read the LM-head result off the die into s->logits[n_vocab].
 *
 * Top-k mode (default): fc_topk_logits_ctx reduces the logits ON the die
 * (bit-exact iterative argmax, host tie-break) and copies back only k
 * (id, logit) pairs, presented SPARSELY by gpu_present_topk above — so
 * downstream consumers are untouched: the host argmax scan finds the identical
 * token, and any sampler with top_k <= k sees exactly the survivors it would
 * have kept.
 *
 * Full mode (k == 0, or any top-k failure): the original full n_vocab D2H
 * copy. */
static int short_row_on(void);
static unsigned int short_row_topk_opts(void);
static int short_row_presented(void);
static int short_row_keep(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx, fc_buffer *logits_all,
                          int u, pn_qwen35moe_state *s, int n_vocab);

static void gpu_fetch_logits(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx, fc_buffer *logits_g,
                             pn_qwen35moe_state *s, int n_vocab)
{
    int k = gpu_topk_k();
    if (s && s->full_logits) k = 0;   /* constrained decode needs every logit */
    int ids[FC_TOPK_MAX];
    float vals[FC_TOPK_MAX];
    int nan_row = 1;

    /* Short row for a single-row decode (g given; see gpu_present_slots_logits). */
    if (g && s && s->full_logits && s->short_row_ok && short_row_on() &&
        fc_topk_logits_slots_opt_ctx(ctx, logits_g, 1, n_vocab, PN_QWEN35MOE_SHORT_ROW_K, ids,
                                     vals, &nan_row, short_row_topk_opts()) == 0 &&
        nan_row == 0 && vals[PN_QWEN35MOE_SHORT_ROW_K - 2] > vals[PN_QWEN35MOE_SHORT_ROW_K - 1] &&
        short_row_keep(g, ctx, logits_g, 0, s, n_vocab) == 0) {
        gpu_present_topk(s, ids, vals, short_row_presented(), n_vocab);
        s->short_row_bound = vals[PN_QWEN35MOE_SHORT_ROW_K - 1];
        s->short_row_live = 1;
        return;
    }

    if (k > 0 &&
        fc_topk_logits_ctx(ctx, logits_g, n_vocab, k, ids, vals) == 0) {
        gpu_present_topk(s, ids, vals, k, n_vocab);
        return;
    }

    fc_buffer_to_cpu(logits_g, s->logits);
    s->topk_filled = 0;             /* logits holds real values everywhere */
    s->short_row_live = 0;
}

/* QWEN35_PERSLOT_FULL_LOGITS (default on; =0 decides for the whole batch):
 * decide top-k vs full logits PER SLOT instead of for the whole batch. Exact:
 * each slot's sampled stream is unchanged (k80-mtp-tool-grammar, 12/12 identical;
 * our K80 host has served it since 2026-11-20). */
static int perslot_full_logits_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_PERSLOT_FULL_LOGITS");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* QWEN35_SHORT_ROW (default on; "0" = off): a constrained slot whose turn allows it
 * (short_row_ok) gets its exact top PN_QWEN35MOE_SHORT_ROW_K instead of the full
 * 1 MB row over the host link, the full row kept on the die for the sampler's fill
 * (engine.h pn_sampler_short_row; why the draw stays exact: engine.c SHORT ROW). */
static int short_row_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_SHORT_ROW");
        v = qw_env_flag_default_on(env);
    }
    return v;
}

/* QWEN35_TOPK_FAST (default on; "0" = off): the short row's device top-k ranks over several
 * blocks per row and replays as a captured graph with a pinned result copy
 * (fc_topk_logits_slots_opt_ctx). Same pairs and NaN flags; ~0.15 ms less a step at 1
 * slot, ~0.6 ms at 4 on a K80. */
static unsigned int short_row_topk_opts(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_TOPK_FAST");
        v = qw_env_flag_default_on(env);
    }
    return v ? FC_TOPK_OPT_ROW_BLOCKS | FC_TOPK_OPT_GRAPH : 0u;
}

/* Red arms (gates only; PN_SHORT_ROW_RED): `fill` = the fill reports success and writes
 * nothing, so every draw that needed the full row samples the short one; `k40` = that, with
 * only the top 40 presented (the short row always trusted, and too short). */
static int short_row_red(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PN_SHORT_ROW_RED");
        v = !e ? 0 : strcmp(e, "fill") == 0 ? 1 : strcmp(e, "k40") == 0 ? 2 : 0;
    }
    return v;
}
static int short_row_red_fill(void) { return short_row_red() != 0; }
static int short_row_presented(void) { return short_row_red() == 2 ? 40 : PN_QWEN35MOE_SHORT_ROW_K; }

/* Keep row u of `logits_all` (on ctx) on the die for s's draw. 0 on success; -1 = no
 * keep slot or no memory (the caller copies the full row as before). */
static int short_row_keep(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx, fc_buffer *logits_all,
                          int u, pn_qwen35moe_state *s, int n_vocab)
{
    const size_t row_bytes = (size_t)n_vocab * sizeof(float);
    int r = -1;

    for (int i = 0; i < PN_QWEN35MOE_GPU_MAX_SLOTS && r < 0; i++)
        if (g->srow_state[i] == (const void *)s)
            r = i;
    for (int i = 0; i < PN_QWEN35MOE_GPU_MAX_SLOTS && r < 0; i++)
        if (!g->srow_state[i])
            r = i;
    if (r < 0)
        return -1;
    if (!g->srow_buf[r] || g->srow_ctx[r] != ctx) {
        fc_buffer_free(g->srow_buf[r]);
        g->srow_buf[r] = fc_buffer_alloc_ctx(ctx, row_bytes);
        g->srow_ctx[r] = g->srow_buf[r] ? ctx : NULL;
        if (!g->srow_buf[r])
            return -1;
    }
    fc_buffer_copy_d2d_ctx(ctx, g->srow_buf[r], 0, logits_all, (size_t)u * row_bytes, row_bytes);
    g->srow_state[r] = s;
    s->short_row_slot = r;
    s->short_row_g = g;
    return 0;
}

int pn_qwen35moe_short_row_fill_cuda(pn_qwen35moe_state *s, float *row, int n_vocab)
{
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)s->short_row_g;
    const int r = s->short_row_slot;

    if (!g || r < 0 || r >= PN_QWEN35MOE_GPU_MAX_SLOTS || g->srow_state[r] != (const void *)s ||
        !g->srow_buf[r])
        return -1;
    s->short_row_live = 0;
    if (row == s->logits)
        s->topk_filled = 0;          /* the row holds real values everywhere */
    if (short_row_red_fill())
        return 0;
    /* The keep copy runs on the head's stream; commit it before the legacy-stream D2H. */
    fc_sync_ctx(g->srow_ctx[r]);
    fc_buffer_to_cpu_at(g->srow_buf[r], row, 0, (size_t)n_vocab * sizeof(float));
    return 0;
}

/* Deliver each slot's LM-head result out of the batched [nslots][n_vocab]
 * `logits_all` into that slot's own s->logits.
 *
 * WHY THIS IS NOT JUST THE OLD LOOP.  A slot that asked for constrained
 * decoding (tool grammar, logit_bias or logprobs — see sg_needs_full_logits)
 * sets s->full_logits for the WHOLE TURN, and the old code answered that
 * question for the entire batch: one such slot and every slot dropped onto the
 * per-slot fallback, which costs, per slot, a 993 KB device-to-device stage
 * into sc->r_logits, a full stream sync, and its own 2k+1 top-k launches over
 * its own row.  At k=40 that is 81 launches and ~40 MB of device reads PER
 * SLOT where the batched reduction does all K rows in the SAME 81 launches
 * (blockIdx.y is the slot).  The innocent slots paid for the constrained one.
 *
 * Per-slot instead: run the batched reduction once over all K rows exactly as
 * an unconstrained step does, hand the unconstrained slots their sparse
 * presentation from it, and give ONLY the constrained slots a full row —
 * copied straight out of `logits_all` at that slot's row offset, so even they
 * lose the D2D stage and the extra sync.  When EVERY slot wants full logits
 * (K=1 with a grammar, the live single-user tool-calling shape) the reduction
 * is skipped entirely rather than run and thrown away.
 *
 * Bit-exact, per slot, either way.  fc_topk_logits_slots_ctx is documented and
 * built to be per-row identical to fc_topk_logits_ctx — same nparts, scan
 * order, comparator and reduction shapes — and it restores the row it clobbers
 * during its rounds before it syncs, so the full-row copy taken afterwards is
 * the same bytes fc_buffer_to_cpu would have taken.
 *
 * @param ctx        the die the head ran on; `logits_all` and `sc` belong to it
 * @param logits_all [nslots][n_vocab] batched head output
 * @param sc         per-die scratch, for the legacy fallback's single-row stage
 * @param sts        [nslots] states, already offset to this group's base
 * @param nslots     rows in `logits_all`
 * @param n_vocab    vocabulary size
 * @return 0 on success, -1 if a state is missing
 *
 * Thread-safe: no (per-die, called inside the model's GPU turn)
 */
static int gpu_present_slots_logits(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                                    fc_buffer *logits_all, gpu_die_scratch *sc,
                                    pn_qwen35moe_state *const *sts, int nslots,
                                    int n_vocab)
{
    int k = gpu_topk_k();
    size_t row_bytes = (size_t)n_vocab * sizeof(float);
    int ids_all[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];
    float vals_all[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];

    for (int u = 0; u < nslots; u++) {
        if (!sts[u]) return -1;
        sts[u]->short_row_live = 0;
    }

    /* Short rows (short_row_on): one exact top-K' over every row, K' = max(k, 257); the
     * unconstrained slots take its first k (the same pairs a k call returns: the order is
     * total), an eligible constrained slot its 257 when the row is clean (no NaN, 256th
     * strictly above 257th), else the full row as below. */
    if (k > 0 && perslot_full_logits_on() && short_row_on()) {
        const int ks = k > PN_QWEN35MOE_SHORT_ROW_K ? k : PN_QWEN35MOE_SHORT_ROW_K;
        int n_short = 0, nan_rows[PN_QWEN35MOE_GPU_MAX_SLOTS];
        for (int u = 0; u < nslots; u++)
            if (sts[u]->full_logits && sts[u]->short_row_ok) n_short++;
        if (n_short > 0 && ks <= FC_TOPK_MAX && nslots <= PN_QWEN35MOE_GPU_MAX_SLOTS &&
            fc_topk_logits_slots_opt_ctx(ctx, logits_all, nslots, n_vocab, ks, ids_all,
                                         vals_all, nan_rows, short_row_topk_opts()) == 0) {
            if (qw_ledger_on())
                qw_ledger_lap(QW_LH_PRESENT_WAIT);
            for (int u = 0; u < nslots; u++) {
                const int *iu = ids_all + (size_t)u * ks;
                const float *vu = vals_all + (size_t)u * ks;
                if (!sts[u]->full_logits) {
                    gpu_present_topk(sts[u], iu, vu, k, n_vocab);
                } else if (sts[u]->short_row_ok && nan_rows[u] == 0 &&
                           vu[PN_QWEN35MOE_SHORT_ROW_K - 2] > vu[PN_QWEN35MOE_SHORT_ROW_K - 1] &&
                           short_row_keep(g, ctx, logits_all, u, sts[u], n_vocab) == 0) {
                    gpu_present_topk(sts[u], iu, vu, short_row_presented(), n_vocab);
                    sts[u]->short_row_bound = vu[PN_QWEN35MOE_SHORT_ROW_K - 1];
                    sts[u]->short_row_live = 1;
                } else {
                    fc_buffer_to_cpu_at(logits_all, sts[u]->logits,
                                        (size_t)u * row_bytes, row_bytes);
                    sts[u]->topk_filled = 0;
                }
            }
            if (qw_ledger_on())
                qw_ledger_lap(QW_LH_PRESENT_COPY);
            return 0;
        }
        /* No eligible slot, or the K' call failed: exactly the path below. */
    }

    if (k > 0 && perslot_full_logits_on()) {
        int n_topk = 0;
        for (int u = 0; u < nslots; u++)
            if (!sts[u]->full_logits) n_topk++;

        int reduced = 0;
        if (n_topk > 0) {
            reduced = (fc_topk_logits_slots_ctx(ctx, logits_all, nslots,
                                                n_vocab, k, ids_all,
                                                vals_all) == 0);
        } else {
            /* No slot wants a sparse row, so nothing reduces the head — but
             * the copies below are on the legacy NULL stream and the head
             * matvec is on ctx->stream, so commit it explicitly. */
            fc_sync_ctx(ctx);
        }
        if (qw_ledger_on())
            qw_ledger_lap(QW_LH_PRESENT_WAIT);
        if (reduced || n_topk == 0) {
            for (int u = 0; u < nslots; u++) {
                if (sts[u]->full_logits) {
                    fc_buffer_to_cpu_at(logits_all, sts[u]->logits,
                                        (size_t)u * row_bytes, row_bytes);
                    sts[u]->topk_filled = 0;
                    sts[u]->short_row_live = 0;
                } else {
                    gpu_present_topk(sts[u], ids_all + (size_t)u * k,
                                     vals_all + (size_t)u * k, k, n_vocab);
                }
            }
            if (qw_ledger_on())
                qw_ledger_lap(QW_LH_PRESENT_COPY);
            return 0;
        }
        /* The batched reduction failed: fall through to the legacy path, which
         * is the same fallback the batch-wide code takes. */
    }

    if (batch_needs_full_logits(sts, nslots)) k = 0;
    if (k > 0 && fc_topk_logits_slots_ctx(ctx, logits_all, nslots, n_vocab, k,
                                          ids_all, vals_all) == 0) {
        for (int u = 0; u < nslots; u++)
            gpu_present_topk(sts[u], ids_all + (size_t)u * k,
                             vals_all + (size_t)u * k, k, n_vocab);
        return 0;
    }
    for (int u = 0; u < nslots; u++) {
        fc_buffer_copy_d2d_ctx(ctx, sc->r_logits, 0, logits_all,
                               (size_t)u * row_bytes, row_bytes);
        fc_sync_ctx(ctx);
        gpu_fetch_logits(NULL, ctx, sc->r_logits, sts[u], n_vocab);
    }
    return 0;
}

static int gpu_get_batch_host(pn_qwen35moe_gpu *g,
                              const pn_qwen35moe_config *cfg, int n);
static int tp_batch_walk(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, const uint32_t *tokens,
                         int n, pn_qwen35moe_state *st, int pos0, int spec,
                         fc_cuda_ctx **ctx_out, gpu_batch_scratch **b_out,
                         gpu_die_scratch **sc_out);

/* QWEN35_TP: commit a decode row the TP walk appended at pos (spec form, so the
 * attention watermark waits at pos): advance every half still bound to st. The
 * SSM scans already hold state-after-1. */
static void tp_commit_row(pn_qwen35moe_gpu *g, pn_qwen35moe_state *st, int pos)
{
    for (int i = 0; i < g->n_attn; i++) {
        gpu_attn_layer *da = &g->attn[i];
        const pn_qwen35moe_kv *kv = &st->kv[qw_attn_hidx(g, i)];
        const void *id = kv->q4 ? (const void *)kv->kq : (const void *)kv->k;
        if (da->kv_host_id == id && da->kv_gen == kv->content_gen && da->kv_synced_pos == pos)
            da->kv_synced_pos = pos + 1;
    }
}

/* QWEN35_TP single-token forward: the TP walk at n = 1 with the decode-form
 * (verify) kernels, then the single-token head and logits fetch on the head die. */
static float *tp_forward_token(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, uint32_t token,
                               pn_qwen35moe_state *s, int pos)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    fc_cuda_ctx *ctx = NULL;
    gpu_batch_scratch *b = NULL;
    gpu_die_scratch *sc = NULL;

    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0)
            return NULL;
        g->host_ok = 1;
    }
    if ((s->n_attn > 0 && pos + 1 > s->kv[0].max_seq) || gpu_get_batch_host(g, cfg, 1) < 0 ||
        tp_batch_walk(m, g, &token, 1, s, pos, 1, &ctx, &b, &sc) < 0)
        return NULL;
    tp_commit_row(g, s, pos);
    fc_rms_norm_ctx(ctx, b->hidden, g->output_norm.buf, sc->r_xnorm, cfg->n_embd,
                    cfg->rms_norm_eps);
    gpu_matvec(&g->output, sc->r_xnorm, sc->r_logits, g->host.a, g->host.b);
    fc_sync_ctx(ctx);
    gpu_fetch_logits(g, ctx, sc->r_logits, s, cfg->n_vocab);
    return s->logits;
}

float *pn_qwen35moe_forward_token_cuda(pn_qwen35moe_model *m, uint32_t token,
                                       pn_qwen35moe_state *s, int pos)
{
    if (!m || !s)
        return NULL;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return NULL; /* not uploaded — caller should fall back to the CPU path */
    if (g->tp)
        return tp_forward_token(m, g, token, s, pos);

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    if ((int)token >= cfg->n_vocab)
        return NULL;

    /* NOTE: no sparse-top-k reset is needed here any more.  The tracking now
     * lives on the state that owns s->logits (see gpu_present_topk), so a
     * fresh state starts un-filled by construction (calloc) and a reused one
     * carries bookkeeping that still describes its own buffer.  The old
     * pos==0 reset existed only to paper over the tracking living on the
     * shared GPU context, keyed by the buffer's address. */

    /* Persistent host scratch: allocated on the first forward and reused —
     * the previous per-token calloc/free of ~3 x n_vocab floats was pure
     * mmap/page-fault churn on the 2-core K80 host. */
    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0)
            return NULL;
        g->host_ok = 1;
    }
    gpu_host_scratch h = g->host;   /* pointer copies — the buffers persist */

    double prof_t0 = 0.0;
    if (prof_on()) {
        prof_reset();
        prof_t0 = prof_now();
    }

    /* The residual stream + scratch + the LM-head output buffer all come from
     * the per-die persistent scratch (gpu_die_scratch), so each die works in
     * its OWN memory. The boundary `hidden` is handed to the next die between
     * segments (gpu_handoff_hidden); the SSM/attention dual state never
     * crosses. */
    fc_buffer *logits_g = NULL;
    gpu_die_scratch *sc = NULL;

    /* Embed: fetch the token's embedding row (per-row from the mmap when the
     * F32 heap was dropped, A2c), then upload it as die A's residual stream. */
    pn_qwen35moe_model_embed_row(m, token, h.a);
    qwen35moe_dbg("input_embed", -1, h.a, c);

    /* ---- Walk the dies in order. Die i runs its layer range against its own
     * scratch, then hands the residual stream to die i+1. The last die also
     * runs the final norm + LM head. ---- */
    for (int d = 0; d < g->n_die; d++) {
        if (gpu_select_die(g, g->die[d]) < 0)
            goto fail;
        sc = gpu_get_scratch(g, g->ctx[d], cfg);
        if (!sc)
            goto fail;
        if (d == 0)
            fc_buffer_from_cpu(sc->r_hidden, h.a);   /* seed the stream */

        int layer_lo = g->layer_lo[d];
        int layer_hi = g->layer_hi[d];
        /* Single-die prefix cap (same env as upload, single-die mode only):
         * run only the first N global layers so the per-layer l_out dumps can
         * be diffed against the CPU on one die. The final norm/head are
         * meaningless when capped (partial hidden) — only the per-layer dumps
         * matter in that mode. */
        if (g->n_die == 1) {
            const char *cap_env = getenv("QWEN35_GPU_MAX_LAYERS");
            if (cap_env) { int v = atoi(cap_env);
                           if (v > 0 && v < layer_hi) layer_hi = v; }
        }

        /* Segment graph first (C1): this die's whole segment as ONE
         * cuGraphLaunch. Falls back to the per-layer loop (with its per-layer
         * SSM graphs) when not eligible — pos 0, capture failed, ring wrap,
         * prof/dbg, etc. */
        int seg_rc = gpu_segment_forward(m, g, g->ctx[d], d, layer_lo, layer_hi,
                                         sc->r_hidden, sc->r_xnorm,
                                         sc->r_blockout, s, pos, &h);
        if (seg_rc < 0)
            goto fail;
        if (seg_rc != 0) {
            for (int layer = layer_lo; layer < layer_hi; layer++) {
                if (gpu_layer_forward(m, g, g->ctx[d], layer, sc->r_hidden,
                                      sc->r_xnorm, sc->r_blockout, s, pos,
                                      &h) < 0)
                    goto fail;
            }
        }

        if (d + 1 < g->n_die) {
            /* Boundary hand-off, then move onto the next die. Export runs
             * while THIS die is still selected; import after the switch, once
             * the receiving die's scratch exists. */
            size_t hidden_bytes = (size_t)c * sizeof(float);
            gpu_handoff_export(g, d, sc->r_hidden, h.a, hidden_bytes);
            if (gpu_select_die(g, g->die[d + 1]) < 0)
                goto fail;
            gpu_die_scratch *next = gpu_get_scratch(g, g->ctx[d + 1], cfg);
            if (!next)
                goto fail;
            gpu_handoff_import(g, d, sc->r_hidden, next->r_hidden, h.a,
                               hidden_bytes);
            continue;
        }

        /* ---- Last die: final norm + LM head. ---- */
        logits_g = sc->r_logits;
        fc_rms_norm_ctx(g->ctx[d], sc->r_hidden, g->output_norm.buf,
                        sc->r_xnorm, c, cfg->rms_norm_eps);
        qwen35moe_dbg_gpu(g->ctx[d], "result_norm", -1, sc->r_xnorm, c, h.a);
        double th0 = prof_on() ? prof_now() : 0.0;
        gpu_matvec(&g->output, sc->r_xnorm, logits_g, h.a, h.b);
        if (prof_on()) { fc_sync_ctx(g->ctx[d]); prof_phase[3] += prof_now() - th0; }
        double tl0 = prof_on() ? prof_now() : 0.0;
        fc_sync_ctx(g->ctx[d]);
        gpu_fetch_logits(g, g->ctx[d], logits_g, s, cfg->n_vocab);
        if (prof_on()) prof_phase[4] += prof_now() - tl0;
        qwen35moe_dbg("result_output", -1, s->logits, cfg->n_vocab);
    }

    /* All buffers (device gpu_die_scratch + host g->host) are persistent —
     * nothing to free per token. */
    if (prof_on())
        prof_report(prof_now() - prof_t0);
    return s->logits;

fail:
    return NULL;
}

/* ============================================================
 * BATCHED PREFILL forward — n tokens of ONE sequence in one pass
 *
 * The serving-foundation path: all n positions' activations live in
 * token-major [n][dim] GPU buffers and every per-token matvec becomes ONE
 * fc_matvec_*_nx launch (the weight is streamed once per 8-column tile —
 * FC_NX_COLS, L6-widened — instead of once per token). The SSM conv/recurrence are scans — one
 * launch walks the n positions inside the kernel, carrying state exactly
 * as n sequential tokens would. Attention appends all n K/V rows to the
 * host cache and runs the same host GQA softmax per position (causal by
 * construction: position t attends to cache entries [0, pos0+t]). MoE
 * routes each position to its own top-8 experts in batched launches.
 *
 * BIT-EXACT: every kernel used here is per-position bit-identical to the
 * single-token path's kernel (see the fc_*_nx headers), the host glue is
 * the same code per position, and the accumulation orders are unchanged —
 * so the per-position logits, the final KV cache, and the carried SSM
 * state are bit-identical to running the n tokens one at a time through
 * pn_qwen35moe_forward_token_cuda.
 *
 * The single-token fused/graph fast paths are NOT used here (the batch is
 * compute-dense enough that launch count no longer dominates); the batch
 * path is additive and leaves the single-token path untouched.
 * ============================================================ */

static void gpu_batch_scratch_free(gpu_batch_scratch *b)
{
    fc_buffer **bufs[] = {
        &b->hidden, &b->x_norm, &b->block_out,
        &b->s_qkv, &b->s_conv, &b->s_z, &b->s_qg, &b->s_kg, &b->s_vg,
        &b->s_y, &b->s_gated, &b->s_beta, &b->s_alpha, &b->s_g,
        &b->a_qg, &b->a_q, &b->a_z, &b->a_out, &b->a_k, &b->a_v,
        &b->m_logits, &b->m_tidx, &b->m_twt, &b->m_gate, &b->m_up,
        &b->m_hid, &b->m_out, &b->m_shexp, &b->m_sgate, &b->m_ecount,
        &b->m_ereq, &b->m_wlist,
    };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        fc_buffer_free(*bufs[i]);
        *bufs[i] = NULL;
    }
    b->ctx = NULL;
    b->n_cap = 0;
}

/* Get (lazily creating / growing) the batch scratch for ctx's die, sized for
 * at least n positions. Returns NULL for a context that is not one of the
 * split's dies, or on allocation failure. Keyed by die index — see
 * gpu_get_scratch for why the old two-slot scan could not survive a wider
 * split. */
static gpu_batch_scratch *gpu_get_batch_scratch(pn_qwen35moe_gpu *g,
                                                fc_cuda_ctx *ctx,
                                                const pn_qwen35moe_config *cfg,
                                                int n)
{
    int die_idx = gpu_die_index_of_ctx(g, ctx);
    if (die_idx < 0)
        return NULL;
#ifndef PN_PDO_RED_SHARED_SCRATCH
    gpu_batch_scratch *b = g->lane == QW_LANE_DECODE ? &g->bscratch_dec[die_idx]
                                                     : &g->bscratch[die_idx];
#else
    /* Red build: the decode lane shares the prefill's batch scratch, so an
     * overlap cycle's decode stage overwrites a chunk sitting half-way through
     * a stage. The exactness gates must see it. */
    gpu_batch_scratch *b = &g->bscratch[die_idx];
#endif
    if (b->ctx == ctx && b->n_cap >= n)
        return b;
    if (b->ctx)
        gpu_batch_scratch_free(b);

    const size_t c = (size_t)cfg->n_embd;
    const size_t conv_dim = (size_t)cfg->conv_dim;
    const size_t val_dim = (size_t)cfg->ssm_state * cfg->ssm_dt_rank;
    const size_t nvh = (size_t)cfg->ssm_dt_rank;
    const size_t attn_dim = (size_t)cfg->head_dim * cfg->n_head;
    const size_t kv_dim = (size_t)cfg->head_dim * cfg->n_head_kv;
    const size_t nn = (size_t)n;
    size_t ff = (size_t)(cfg->expert_ff > cfg->shared_ff ? cfg->expert_ff
                                                         : cfg->shared_ff);
    /* Dense-FFN (Bonsai, Ornith) has n_expert == n_expert_used == 0, and a
     * 0-byte alloc returns NULL, which failed this whole scratch: every batched
     * decode step and every batched prefill chunk of a dense model died here.
     * Floor the MoE-only buffers to 1, as gpu_get_scratch does; the dense FFN
     * never reads them. */
    const size_t n_exp = (size_t)(cfg->n_expert > 0 ? cfg->n_expert : 1);
    const size_t k_exp = (size_t)(cfg->n_expert_used > 0 ? cfg->n_expert_used
                                                         : 1);
#define BALLOC(field, elems) \
    b->field = fc_buffer_alloc_ctx(ctx, nn * (elems) * sizeof(float))
    BALLOC(hidden, c); BALLOC(x_norm, c); BALLOC(block_out, c);
    BALLOC(s_qkv, conv_dim); BALLOC(s_conv, conv_dim);
    BALLOC(s_z, val_dim); BALLOC(s_qg, val_dim); BALLOC(s_kg, val_dim);
    BALLOC(s_vg, val_dim); BALLOC(s_y, val_dim); BALLOC(s_gated, val_dim);
    BALLOC(s_beta, nvh); BALLOC(s_alpha, nvh); BALLOC(s_g, nvh);
    BALLOC(a_qg, 2 * attn_dim); BALLOC(a_q, attn_dim); BALLOC(a_z, attn_dim);
    BALLOC(a_out, attn_dim);
    BALLOC(a_k, kv_dim); BALLOC(a_v, kv_dim);
    BALLOC(m_logits, n_exp);
    b->m_tidx = fc_buffer_alloc_ctx(ctx, nn * k_exp * sizeof(int32_t));
    BALLOC(m_twt, k_exp);
    BALLOC(m_gate, ff); BALLOC(m_up, ff);
    BALLOC(m_hid, k_exp * ff);
    BALLOC(m_out, c); BALLOC(m_shexp, c); BALLOC(m_sgate, 1);
#undef BALLOC
    /* Expert-union dedup request lists (flag-gated, so the default build
     * allocates nothing here). ereq is capped at n requests per expert: a
     * token's k slots are distinct, so it can appear in an expert's list at
     * most once and R <= n. */
    if ((gpu_moe_dedup_enabled() || gpu_moe_dedup_q80_enabled()) &&
        cfg->n_expert > 0) {
        b->m_ecount = fc_buffer_alloc_ctx(ctx, (size_t)cfg->n_expert *
                                               sizeof(int32_t));
        b->m_ereq = fc_buffer_alloc_ctx(ctx, (size_t)cfg->n_expert * nn *
                                             sizeof(int32_t));
        if (!b->m_ecount || !b->m_ereq) {
            gpu_batch_scratch_free(b);
            return NULL;
        }
    }
    if ((gpu_moe_dedup_enabled() || gpu_moe_dedup_q80_enabled()) &&
        cfg->n_expert > 0) {
        /* Compacted work list for the compacted-grid gate+up. Sized at the
         * EXACT host-side bound on the (expert, request-tile) count, which is
         * also the grid.y that launch uses — 352 int2 = 2.8 KB at the prefill
         * shape. Allocated whenever the dedup path is on: it is negligible, and
         * a NULL here is what disables the compacted arm. wmax is monotone in
         * ntok, so sizing it at n_cap covers every chunk this scratch serves. */
        int wmax = fc_moe_dedup_wmax(cfg->n_expert, cfg->n_expert_used, n, n);
        if (wmax > 0)
            b->m_wlist = fc_buffer_alloc_ctx(ctx, (size_t)wmax * 2 *
                                                  sizeof(int32_t));
    }

    if (!b->hidden || !b->x_norm || !b->block_out || !b->s_qkv || !b->s_conv ||
        !b->s_z || !b->s_qg || !b->s_kg || !b->s_vg || !b->s_y || !b->s_gated ||
        !b->s_beta || !b->s_alpha || !b->s_g || !b->a_qg || !b->a_q ||
        !b->a_z || !b->a_out || !b->a_k || !b->a_v || !b->m_logits ||
        !b->m_tidx ||
        !b->m_twt || !b->m_gate || !b->m_up || !b->m_hid || !b->m_out ||
        !b->m_shexp || !b->m_sgate) {
        fprintf(stderr, "[qwen35moe] batch scratch for %d rows on die %d: "
                "alloc failed\n", n, die_idx);
        gpu_batch_scratch_free(b);
        return NULL;
    }
    static unsigned batch_gen;
    b->ctx = ctx;
    b->n_cap = n;
    b->gen = ++batch_gen;
    return b;
}

/* Host staging for the batch path: 4 float regions sized against the
 * largest batched read-back (the joint Q+gate projection, 2*attn_dim wide).
 * r0: aqg read-back / embed + boundary staging / attn-out staging
 * r1: q-only staging + roped-q read-back;  r2: the persisted gate z
 * r3: k|v read-back (2*kv_dim per position). */
static int gpu_get_batch_host(pn_qwen35moe_gpu *g,
                              const pn_qwen35moe_config *cfg, int n)
{
    size_t per_pos = 2 * (size_t)cfg->head_dim * cfg->n_head; /* 2*attn_dim */
    if (per_pos < (size_t)cfg->conv_dim)
        per_pos = (size_t)cfg->conv_dim;
    if (g->bhost_cap >= n && g->bhost[0])
        return 0;
    for (int i = 0; i < 4; i++) {
        free(g->bhost[i]);
        g->bhost[i] = malloc((size_t)n * per_pos * sizeof(float));
        if (!g->bhost[i])
            return -1;
    }
    g->bhost_cap = n;
    return 0;
}

/* Batched matvec dispatch: y[t][rows] = dequant(w) . x[t], t in [0, n).
 * Returns -1 if the weight has no batched kernel (caller must abort the
 * batch and fall back to the single-token path). */
static int gpu_matvec_nx(const gpu_weight *w, fc_buffer *x, fc_buffer *y,
                         int n)
{
    if (!w->bytes)
        return -1;
    if (w->kpar) {   /* TP down: this rank's parity partial over its compact input */
        if ((int)w->type != GGUF_QUANT_Q8_0)
            return -1;
        fc_matvec_q80_par_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols / 2,
                                 w->row_stride, n);
        return 0;
    }
    switch ((int)w->type) {
    case GGUF_QUANT_Q3_K:
        fc_matvec_q3k_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_Q4_K:
        fc_matvec_q4k_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_Q5_K:
        fc_matvec_q5k_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_Q6_K:
        fc_matvec_q6k_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_Q8_0:
        fc_matvec_q80_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_F32:
        fc_matvec_f32_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_F16:
        /* No batched host fallback exists (a -1 here fails the whole batched
         * MoE block), so an absent kernel must refuse, not launch nothing. */
        if (!fc_matvec_f16_available_ctx(w->ctx))
            return -1;
        fc_matvec_f16_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                             w->row_stride, n);
        return 0;
    case GGUF_QUANT_BF16:
        if (!fc_matvec_bf16_available_ctx(w->ctx))
            return -1;
        fc_matvec_bf16_nx_ctx(w->ctx, w->bytes, x, y, w->rows, w->cols,
                              w->row_stride, n);
        return 0;
    default:
        return -1;
    }
}

/* Should this batched MoE call take the expert-union dedup gate+up?
 *
 * Five conditions, all required:
 *
 * 1. `allow_dedup` — the caller is a path where dedup is allowed at all
 *    (prefill; see gpu_moe_block_batch). This is the policy gate.
 * 2. The flag is on. Default off.
 * 3. The batch is wide enough to break even. The dedup kernel launches
 *    n_expert blocks per FF row-group no matter how few positions there are,
 *    while _nx launches n*k; below n*k == n_expert the dedup grid is strictly
 *    bigger AND the expert union is nearly as large as the application count,
 *    so it can only lose. n_expert/k is 32 positions for nex-n2-mini. This is
 *    a block-count derivation, not a measured optimum — the measured point is
 *    n = 128, the prefill chunk.
 * 4. The scratch exists and the kernels are actually in the loaded PTX. The
 *    last check matters because a stale fc_cuda_ptx.h leaves the two dedup
 *    kernels NULL while the rest of the grouped path loads fine; without it
 *    the dedup wrapper would silently no-op and hand the down kernel an
 *    UNWRITTEN hid.
 * 5. The gate/up experts are Q3_K. The dedup kernels decode Q3_K and ONLY
 *    Q3_K (fc_q3k_decode_sb) — there is no Q4_K twin, because dedup is an
 *    opt-in prefill speed lever, not a format the engine must support. A
 *    Q4_K_M model therefore keeps the batched gate+up on fc_moe_gateup_nx_ctx,
 *    which does dispatch on quant. Without this the flag would hand Q4_K bytes
 *    to a Q3_K decode — silently wrong output, the exact failure this whole
 *    change exists to remove. */
static int gpu_moe_batch_use_dedup(fc_cuda_ctx *ctx,
                                   const pn_qwen35moe_config *cfg,
                                   gpu_batch_scratch *b, int n,
                                   int allow_dedup, int gateup_quant)
{
    if (!gpu_moe_dedup_enabled())
        return 0;
    /* The decode override exists to measure the LOSING configuration, so it
     * bypasses the width floor as well — otherwise an 8-slot step would fail
     * condition 3 and the override would measure nothing. */
    int wide = n * cfg->n_expert_used >= cfg->n_expert ||
               gpu_moe_dedup_decode_override();
    int ready = b->m_ecount && b->m_ereq && fc_moe_dedup_available_ctx(ctx) &&
                gateup_quant == GGUF_QUANT_Q3_K;
    int use = allow_dedup && wide && ready;

    /* Announce each DISTINCT verdict once. QWEN35_MOE_DEDUP=1 asked for a
     * different kernel; silently running the old one would make an A/B measure
     * nothing, and a missing-kernel fallback is exactly the case a caller needs
     * told about.
     *
     * Keyed on (allow_dedup, use) rather than announced-once, because a single
     * process reaches here from BOTH the prefill and the decode call sites and
     * a once-only line reports whichever ran first. Verifying "the flag is on
     * and decode still declined" needs to see the decode verdict, so all four
     * combinations get one line each — four lines, worst case, per process. */
    static int announced[2][2];
    if (!announced[allow_dedup ? 1 : 0][use ? 1 : 0]) {
        announced[allow_dedup ? 1 : 0][use ? 1 : 0] = 1;
        fprintf(stderr, "[qwen35moe] MoE dedup gate+up: %s (n=%d k=%d "
                "n_expert=%d, prefill_path=%d wide=%d kernels=%d "
                "gateup_quant=%d)\n",
                use ? "ON" : "OFF", n, cfg->n_expert_used, cfg->n_expert,
                allow_dedup, wide, ready, gateup_quant);
    }
    return use;
}

/* QWEN35_MOE_DEDUP_DIAG=1: read the per-expert request counts back and report
 * the expert union for the first few layers, then stop.
 *
 * The dedup's whole saving is n*k / |union|, and that ratio has only ever been
 * a binomial estimate. This measures it. Diagnostic only — it forces a host
 * sync per call, so it is off by default and self-limits to LAYERS layers of
 * the first chunk. */
static void gpu_moe_dedup_report_union(fc_cuda_ctx *ctx,
                                       const pn_qwen35moe_config *cfg,
                                       gpu_batch_scratch *b, int n)
{
    enum { LAYERS = 4 };
    static int diag = -1;
    static int reported = 0;
    if (diag < 0) {
        const char *env = getenv("QWEN35_MOE_DEDUP_DIAG");
        diag = (env && env[0] == '1') ? 1 : 0;
    }
    if (!diag || reported >= LAYERS)
        return;
    reported++;

    int32_t *counts = malloc((size_t)cfg->n_expert * sizeof(int32_t));
    if (!counts)
        return;
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->m_ecount, counts,
                       (size_t)cfg->n_expert * sizeof(int32_t));
    int unique = 0, maxr = 0, total = 0;
    for (int e = 0; e < cfg->n_expert; e++) {
        int r = counts[e];
        if (r <= 0)
            continue;
        unique++;
        total += r;
        if (r > maxr)
            maxr = r;
    }
    /* tiles = the z-tiles the kernel actually does work in, summed over
     * experts: this is how many times a weight stream is re-read, so
     * applications/tiles is the REALISED decode saving, not the ideal one. */
    int tiles = 0;
    for (int e = 0; e < cfg->n_expert; e++)
        if (counts[e] > 0)
            tiles += (counts[e] + FC_MOE_DEDUP_RT - 1) / FC_MOE_DEDUP_RT;
    fprintf(stderr, "[qwen35moe] dedup union: n=%d apps=%d unique=%d/%d "
            "maxR=%d meanR=%.2f tiles=%d ideal=%.2fx realised=%.2fx\n",
            n, total, unique, cfg->n_expert, maxr,
            unique ? (double)total / unique : 0.0, tiles,
            unique ? (double)total / unique : 0.0,
            tiles ? (double)total / tiles : 0.0);
    free(counts);
}

/* Batched shared-expert MoE: mirrors gpu_moe_block's UNFUSED sequence per
 * position (the fused fx_pre/fx_post launches are bit-identical to that
 * sequence, so this matches the single-token path either way). Requires
 * the grouped expert layout.
 *
 * `allow_dedup` is the PREFILL DISCRIMINANT, and it is a parameter rather
 * than an internal env read on purpose. This function has two callers with
 * opposite economics: the prefill chunk loop (n = 128 positions, ~4x expert
 * reuse — dedup wins) and gpu_moe_block_slots, the multi-user DECODE step
 * (n = live slots, ~1.1x reuse — dedup measured 7-9% SLOWER and is what got
 * the whole idea shelved). Reading QWEN35_MOE_DEDUP in here would flip both
 * at once. The callers decide; only prefill passes 1. */
/* Batched dense FFN (Bonsai, Ornith): the single-token dense SwiGLU of
 * gpu_moe_block over n rows — gate/up/down in the shexp slots, ffn dim =
 * ffn_length, no router, no scalar gate — then the residual. Without it a dense
 * model had no batched FFN at all: every prefill chunk fell back to one forward
 * per token, and a decode step ran the single-token block once per slot row. */
static int gpu_dense_ffn_batch(fc_cuda_ctx *ctx, gpu_moe *d,
                               const pn_qwen35moe_config *cfg,
                               gpu_batch_scratch *b, int n)
{
    double sp_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);
    if (gpu_matvec_nx(&d->ffn_gate_shexp, b->x_norm, b->m_gate, n) < 0 ||
        gpu_matvec_nx(&d->ffn_up_shexp, b->x_norm, b->m_up, n) < 0)
        return -1;
    fc_moe_silu_mul_ctx(ctx, b->m_gate, b->m_up, b->m_gate,
                        n * cfg->ffn_length);
    if (gpu_matvec_nx(&d->ffn_down_shexp, b->m_gate, b->block_out, n) < 0)
        return -1;
    fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
    sprof_lap(SPROF_MOE_SHEXP, ctx, &sp_t0);
    return 0;
}

/* ============================================================
 * One-slot fused launches on the batched-decode (slots) path
 *
 * At nslots == 1 the slots path runs the single-token path's fused
 * descriptors (fc_fused_multi) over the batch scratch: proj4 (qkv + z + beta
 * + alpha, raw — the GDN step applies sigmoid/decay), ssm_out + residual,
 * attn_output + residual, MoE pre (router + shexp scalar-gate dot + shexp
 * gate/up SwiGLU) and MoE post (shexp down + gate + combine + residual).
 * Every segment runs its stand-alone kernel's per-row body (Q8_0 the
 * fc_matvec_q80/_nx body, F32 fc_matvec_f32's) and every epilogue the small
 * kernel it replaces, so the step is bit-identical to the unfused slots
 * sequence. Same QWEN35_FUSE bits as the single-token sites (attn out has
 * none there either).
 *
 * The descriptors point into the batch scratch, so they are keyed on its
 * generation and built by gpu_slots_fused_prepare, which the layer dispatch
 * runs before any graph capture (a build allocates and uploads; a capture
 * cannot contain that). The block bodies only launch a built descriptor and
 * otherwise run the unfused sequence.
 * ============================================================ */

static void gpu_fused_reset(gpu_fused *f)
{
    gpu_fused_free(f);
    memset(f, 0, sizeof(*f));
}

static void gpu_fused_build(fc_cuda_ctx *ctx, gpu_fused *f,
                            const fc_fused_seg *segs, int nseg)
{
    f->tried = 1;
    f->nseg = nseg;
    f->desc = fc_fused_build_ctx(ctx, segs, nseg, &f->blocks, &f->sh_bytes);
}

/* Launch f if it is built for batch scratch generation `gen` and the step
 * has one slot; -1 = caller runs its unfused sequence. */
static int gpu_fused_launch_one(fc_cuda_ctx *ctx, const gpu_fused *f,
                                unsigned f_gen, const gpu_batch_scratch *b,
                                int nslots, fc_buffer *x)
{
    if (nslots != 1 || f_gen != b->gen || !f->desc)
        return -1;
    fc_fused_multi_ctx(ctx, f->desc, f->nseg, f->blocks, f->sh_bytes, x);
    return 0;
}

static void gpu_moe_slots_fused_prepare(fc_cuda_ctx *ctx, gpu_moe *d,
                                        const pn_qwen35moe_config *cfg,
                                        const gpu_batch_scratch *b)
{
    if (d->fx_s_gen == b->gen)
        return;
    gpu_fused_reset(&d->fx_pre_s);
    gpu_fused_reset(&d->fx_post_s);
    d->fx_s_gen = b->gen;
    if (!d->grouped || cfg->dense_ffn || !(gpu_fuse_mask() & 16) ||
        d->ffn_gate_shexp.type != d->ffn_up_shexp.type ||
        d->ffn_gate_shexp.rows != d->ffn_up_shexp.rows ||
        d->ffn_gate_shexp.row_stride != d->ffn_up_shexp.row_stride)
        return;
    fc_fused_seg segs[3];
    segs[0] = gpu_seg_mv(&d->ffn_gate_inp, b->m_logits, 0);
    memset(&segs[1], 0, sizeof(segs[1]));      /* scalar-gate dot */
    segs[1].w = d->ffn_gate_inp_shexp.buf;
    segs[1].out = b->m_sgate;
    segs[1].rows = 1;
    segs[1].cols = cfg->n_embd;
    segs[1].row_stride = cfg->n_embd * (int)sizeof(float);
    segs[1].kind = 1;
    segs[1].ep = 1;                            /* sigmoid */
    segs[2] = gpu_seg_mv(&d->ffn_gate_shexp, b->m_gate, 0);
    segs[2].w2 = d->ffn_up_shexp.bytes;
    segs[2].kind = 2;                          /* gate+up SwiGLU pair */
    gpu_fused_build(ctx, &d->fx_pre_s, segs, 3);
    if (!d->fx_pre_s.desc || !(gpu_fuse_mask() & 32))
        return;
    fc_fused_seg post = gpu_seg_mv(&d->ffn_down_shexp, b->block_out, 4);
    post.aux0 = b->m_out;
    post.aux1 = b->hidden;
    post.aux2 = b->m_sgate;
    gpu_fused_build(ctx, &d->fx_post_s, &post, 1);
}

static void gpu_ssm_slots_fused_prepare(gpu_ssm_layer *d,
                                        const gpu_batch_scratch *b)
{
    if (d->fx_s_gen == b->gen)
        return;
    gpu_fused_reset(&d->fx_proj_s);
    gpu_fused_reset(&d->fx_out_s);
    d->fx_s_gen = b->gen;
    if (gpu_fuse_mask() & 1) {
        fc_fused_seg segs[4];
        segs[0] = gpu_seg_mv(&d->attn_qkv, b->s_qkv, 0);
        segs[1] = gpu_seg_mv(&d->attn_gate, b->s_z, 0);
        segs[2] = gpu_seg_mv(&d->ssm_beta, b->s_beta, 0);
        segs[3] = gpu_seg_mv(&d->ssm_alpha, b->s_alpha, 0);
        gpu_fused_build(d->ctx, &d->fx_proj_s, segs, 4);
    }
    if (gpu_fuse_mask() & 2) {
        fc_fused_seg seg = gpu_seg_mv(&d->ssm_out, b->block_out, 3);
        seg.aux1 = b->hidden;
        gpu_fused_build(d->ctx, &d->fx_out_s, &seg, 1);
    }
}

static void gpu_attn_slots_fused_prepare(gpu_attn_layer *d,
                                         const gpu_batch_scratch *b)
{
    if (d->fx_s_gen == b->gen)
        return;
    gpu_fused_reset(&d->fx_out_s);
    d->fx_s_gen = b->gen;
    fc_fused_seg seg = gpu_seg_mv(&d->attn_output, b->block_out, 3);
    seg.aux1 = b->hidden;
    gpu_fused_build(d->ctx, &d->fx_out_s, &seg, 1);
}

/* Build (once per batch scratch) the one-slot descriptors of layer `idx`.
 * Outside any graph capture. */
static void gpu_slots_fused_prepare(pn_qwen35moe_gpu *g,
                                    const pn_qwen35moe_config *cfg, int is_attn,
                                    int idx, const gpu_batch_scratch *b,
                                    int nslots)
{
    if (nslots != 1)
        return;
    if (is_attn) {
        gpu_attn_layer *d = &g->attn[idx];
        gpu_attn_slots_fused_prepare(d, b);
        gpu_moe_slots_fused_prepare(d->ctx, &d->moe, cfg, b);
    } else {
        gpu_ssm_layer *d = &g->ssm[idx];
        gpu_ssm_slots_fused_prepare(d, b);
        gpu_moe_slots_fused_prepare(d->ctx, &d->moe, cfg, b);
    }
}

/* Batched routed gate+up into hid ([n*k][ff]): the Q8_0 expert-grouped kernel
 * over the compacted work list when it applies (QWEN35_MOE_DEDUP_Q80 on, Q8_0
 * experts, n * k >= n_expert, lists allocated, kernels in the PTX, and
 * allow_dedup — a prefill caller),
 * else fc_moe_gateup_nx_ctx. Both write the same bits. Returns 0 on launch. */
static int gpu_moe_gateup_batch(fc_cuda_ctx *ctx, const pn_qwen35moe_config *cfg,
                                const gpu_moe *d, gpu_batch_scratch *b,
                                fc_buffer *hid, int ff, int n, int allow_dedup)
{
    const int k = cfg->n_expert_used;
    const int wmax = fc_moe_dedup_wmax(cfg->n_expert, k, n, n);
    if (allow_dedup && gpu_moe_dedup_q80_enabled() &&
        d->gate_all.type == GGUF_QUANT_Q8_0 && n * k >= cfg->n_expert &&
        b->m_ecount && b->m_ereq && b->m_wlist && wmax > 0 &&
        fc_moe_dedup_q80_available_ctx(ctx)) {
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr, "[qwen35moe] MoE Q8_0 grouped gate+up: ON (n=%d k=%d "
                    "n_expert=%d ff=%d)\n", n, k, cfg->n_expert, ff);
        }
        fc_moe_dedup_build_ctx(ctx, b->m_tidx, b->m_ecount, b->m_ereq,
                               cfg->n_expert, k, n, n);
        fc_moe_dedup_build_wl_ctx(ctx, b->m_ecount, b->m_wlist, cfg->n_expert,
                                  wmax);
        if (fc_moe_gateup_q80_dedup_ctx(ctx, d->gate_all.bytes, d->up_all.bytes,
                                        b->x_norm, b->m_ereq, b->m_wlist, hid, k,
                                        ff, cfg->n_embd, d->gate_all.row_stride,
                                        wmax, n) == 0)
            return 0;
    }
    return fc_moe_gateup_nx_ctx(ctx, (int)d->gate_all.type, d->gate_all.bytes,
                                d->up_all.bytes, b->x_norm, b->m_tidx, hid, k, ff,
                                cfg->n_embd, d->gate_all.row_stride, n);
}

/* attn_output + residual for the slots path: one fused launch at one slot
 * when built, else the matvec and the add. */
static int gpu_attn_out_slots(gpu_attn_layer *d, const pn_qwen35moe_config *cfg,
                              gpu_batch_scratch *b, int nslots)
{
    if (gpu_fused_launch_one(d->ctx, &d->fx_out_s, d->fx_s_gen, b, nslots,
                             b->a_out) == 0)
        return 0;
    if (gpu_matvec_nx(&d->attn_output, b->a_out, b->block_out, nslots) < 0)
        return -1;
    fc_add_ctx(d->ctx, b->hidden, b->block_out, b->hidden,
               nslots * cfg->n_embd);
    return 0;
}

static int gpu_moe_block_batch(fc_cuda_ctx *ctx, gpu_moe *d,
                               const pn_qwen35moe_config *cfg,
                               gpu_batch_scratch *b, int n, int allow_dedup)
{
    const int n_embd = cfg->n_embd;
    const int k = cfg->n_expert_used;

    if (cfg->dense_ffn)
        return gpu_dense_ffn_batch(ctx, d, cfg, b, n);
    if (!d->grouped)
        return -1;

    double sp_t0 = 0.0, sub_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);
    sub_begin(ctx, &sub_t0);

    /* Router logits + per-position softmax/top-k. At one slot the fused
     * pre launch also computes the shexp scalar gate and gate/up SwiGLU. */
    int pre = gpu_fused_launch_one(ctx, &d->fx_pre_s, d->fx_s_gen, b, n,
                                   b->x_norm) == 0;
    if (!pre && gpu_matvec_nx(&d->ffn_gate_inp, b->x_norm, b->m_logits, n) < 0)
        return -1;
    sub_lap(&moed_cum[MOED_ROUTER], &moed_laps[MOED_ROUTER], ctx, &sub_t0);
    fc_moe_router_topk_nx_ctx(ctx, b->m_logits, b->m_tidx, b->m_twt,
                              cfg->n_expert, k, cfg->expert_weights_scale, n);
    sub_lap(&moed_cum[MOED_TOPK], &moed_laps[MOED_TOPK], ctx, &sub_t0);
    sprof_lap(SPROF_MOE_SHEXP, ctx, &sp_t0);

    /* Routed experts: grouped gate+up SwiGLU + down/accumulate, all
     * positions in two launches — or three, when the gate+up runs over the
     * expert union instead of over every (token, slot) pair. The dedup hid is
     * bit-exact with the _nx hid, so the down kernel below is unchanged
     * either way and the two paths are a clean A/B. */
    if (gpu_moe_batch_use_dedup(ctx, cfg, b, n, allow_dedup,
                                (int)d->gate_all.type)) {
        fc_moe_dedup_build_ctx(ctx, b->m_tidx, b->m_ecount, b->m_ereq,
                               cfg->n_expert, k, n, n);
        gpu_moe_dedup_report_union(ctx, cfg, b, n);
        /* FC_MOE_DEDUP_COMPACT=1 (default off, and inert unless the PTX was
         * built with -DFC_MOE_DEDUP_COMPACT=1) folds the (expert, request-tile)
         * grid axes into one over a device-built work list: 22,528 blocks
         * instead of 262,144, bit-exact. One extra tiny launch to compact
         * ecount; ereq, hid and the down kernel below are unchanged. */
        int wmax = b->m_wlist
                 ? fc_moe_dedup_wmax(cfg->n_expert, k, n, n) : 0;
        if (wmax > 0 && fc_moe_dedup_compact_enabled_ctx(ctx)) {
            fc_moe_dedup_build_wl_ctx(ctx, b->m_ecount, b->m_wlist,
                                      cfg->n_expert, wmax);
            fc_moe_gateup_q3k_dedup_compact_ctx(
                ctx, d->gate_all.bytes, d->up_all.bytes, b->x_norm, b->m_ereq,
                b->m_wlist, b->m_hid, k, cfg->expert_ff, n_embd,
                d->gate_all.row_stride, wmax, n);
        } else {
            fc_moe_gateup_q3k_dedup_ctx(ctx, d->gate_all.bytes,
                                        d->up_all.bytes, b->x_norm,
                                        b->m_ecount, b->m_ereq, b->m_hid, k,
                                        cfg->expert_ff, n_embd,
                                        d->gate_all.row_stride, cfg->n_expert,
                                        n);
        }
    } else {
        gpu_moe_gateup_batch(ctx, cfg, d, b, b->m_hid, cfg->expert_ff, n,
                             allow_dedup);
    }
    sprof_lap(SPROF_MOE_GATEUP, ctx, &sp_t0);
    fc_moe_down_accum_nx_ctx(ctx, (int)d->down_all.type, d->down_all.bytes,
                             b->m_hid, b->m_tidx, b->m_twt, b->m_out, k,
                             n_embd, cfg->expert_ff,
                             d->down_all.row_stride, n);
    sprof_lap(SPROF_MOE_DOWN, ctx, &sp_t0);
    sub_begin(ctx, &sub_t0);

    /* Shared expert + scalar gate + combine + residual. */
    if (!pre) {
        if (gpu_matvec_nx(&d->ffn_gate_shexp, b->x_norm, b->m_gate, n) < 0 ||
            gpu_matvec_nx(&d->ffn_up_shexp, b->x_norm, b->m_up, n) < 0)
            return -1;
        sub_lap(&moed_cum[MOED_SH_GATEUP], &moed_laps[MOED_SH_GATEUP], ctx,
                &sub_t0);
        fc_moe_silu_mul_ctx(ctx, b->m_gate, b->m_up, b->m_gate,
                            n * cfg->shared_ff);
        sub_lap(&moed_cum[MOED_SH_SILU], &moed_laps[MOED_SH_SILU], ctx,
                &sub_t0);
    }
    if (pre && gpu_fused_launch_one(ctx, &d->fx_post_s, d->fx_s_gen, b, n,
                                    b->m_gate) == 0) {
        sub_lap(&moed_cum[MOED_SH_COMBINE], &moed_laps[MOED_SH_COMBINE], ctx,
                &sub_t0);
        sprof_lap(SPROF_MOE_SHEXP, ctx, &sp_t0);
        return 0;
    }
    if (gpu_matvec_nx(&d->ffn_down_shexp, b->m_gate, b->m_shexp, n) < 0)
        return -1;
    sub_lap(&moed_cum[MOED_SH_DOWN], &moed_laps[MOED_SH_DOWN], ctx, &sub_t0);
    if (!(gpu_fuse_mask() & 64) ||
        fc_moe_shexp_gate_combine_nx_ctx(ctx, d->ffn_gate_inp_shexp.buf,
                                         b->x_norm, b->m_shexp, b->m_out,
                                         b->block_out, b->hidden, n_embd,
                                         n) != 0) {
        fc_moe_shexp_dot_gate_nx_ctx(ctx, d->ffn_gate_inp_shexp.buf,
                                     b->x_norm, b->m_shexp, n_embd, n);
        fc_add_ctx(ctx, b->m_out, b->m_shexp, b->block_out, n * n_embd);
        fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * n_embd);
    }
    sub_lap(&moed_cum[MOED_SH_COMBINE], &moed_laps[MOED_SH_COMBINE], ctx, &sub_t0);
    sprof_lap(SPROF_MOE_SHEXP, ctx, &sp_t0);
    return 0;
}

/* MULTI-USER DECODE MoE: the batched grouped block when this layer HAS a
 * grouped expert layout, else the single-token MoE run once per slot row.
 *
 * gpu_moe_block_batch is grouped-only: the merged-expert kernels exist for
 * gate/up Q3_K|Q4_K with down Q3_K|Q4_K|Q6_K, and gpu_moe_setup clears `grouped` for any
 * other expert quant (or a dense FFN).  The single-token forward has always
 * carried a per-expert fallback for exactly that case, so a model with an
 * off-menu expert quant still decodes — it just loses the grouped kernels on
 * those layers.  The slots path had no such fallback: one non-grouped layer
 * made gpu_moe_block_batch return -1, which aborted the WHOLE batched step at
 * any slot count.  (Qwen3.6-35B-A3B "saber": blk.0/blk.1 ffn_down_exps are
 * Q5_K, so every batched decode step died on layer 0 even with one user.)
 *
 * The fallback re-runs the validated single-token gpu_moe_block over 1-row
 * views of the batch scratch — identical math to what that user would get
 * decoding alone, at the cost of the grouped kernels on those layers only.
 * Slot rows are independent here (the MoE is position-invariant and carries no
 * state), so row-by-row is exact, not an approximation.
 *
 * `sc` (per-die scratch) and `h` (host scratch) are only touched on the
 * fallback; callers may pass NULL when the layer is grouped.
 *
 * Passes allow_dedup = 0: at decode-step slot counts the expert union is
 * nearly as large as the application count, so the dedup gate+up costs more
 * than it saves. Decode stays on the _nx path regardless of QWEN35_MOE_DEDUP;
 * only the explicit QWEN35_MOE_DEDUP_DECODE measurement override lifts it. */
static int gpu_moe_block_slots(fc_cuda_ctx *ctx, gpu_moe *d,
                               const pn_qwen35moe_config *cfg,
                               gpu_batch_scratch *b, int n,
                               gpu_die_scratch *sc, gpu_host_scratch *h)
{
    if (d->grouped || cfg->dense_ffn)
        return gpu_moe_block_batch(ctx, d, cfg, b, n,
                                   gpu_moe_dedup_decode_override());
    if (!sc || !h) {
        fprintf(stderr, "[qwen35moe] slots MoE fallback needs die/host "
                "scratch (non-grouped experts)\n");
        return -1;
    }

    const size_t row_bytes = (size_t)cfg->n_embd * sizeof(float);
    double sp_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);
    for (int u = 0; u < n; u++) {
        fc_buffer *x_norm = b->x_norm, *out_buf = b->block_out,
                  *hidden = b->hidden;
        int rc;
        if (u > 0) {
            x_norm  = fc_buffer_offset_view(b->x_norm, (size_t)u * row_bytes);
            out_buf = fc_buffer_offset_view(b->block_out,
                                            (size_t)u * row_bytes);
            hidden  = fc_buffer_offset_view(b->hidden, (size_t)u * row_bytes);
            if (!x_norm || !out_buf || !hidden) {
                fc_buffer_free_view(x_norm);
                fc_buffer_free_view(out_buf);
                fc_buffer_free_view(hidden);
                return -1;
            }
        }
        rc = gpu_moe_block(ctx, d, cfg, x_norm, out_buf, hidden, sc, h);
        if (u > 0) {
            fc_buffer_free_view(x_norm);
            fc_buffer_free_view(out_buf);
            fc_buffer_free_view(hidden);
        }
        if (rc < 0)
            return rc;
    }
    sprof_lap(SPROF_MOE_ROWFB, ctx, &sp_t0);
    return 0;
}

/* Batched SSM (Gated-Delta-Net) block: projections/glue batched, conv1d +
 * recurrence as position scans over the GPU-resident dual state. When spec
 * is set, the scans additionally write per-position state snapshots into
 * d->conv_snaps / d->ssm_snaps (allocated by the caller) for the partial-
 * accept rollback; the math is unchanged either way. */
/* core_only (QWEN35_TP): stop at the gated-norm output s_gated, before the out
 * projection and residual (the TP body gathers both ranks' cores first). */
static int gpu_ssm_block_batch(gpu_ssm_layer *d,
                               const pn_qwen35moe_config *cfg,
                               gpu_batch_scratch *b, int n, int spec, int core_only)
{
    const int conv_dim = cfg->conv_dim;
    const int S = cfg->ssm_state;
    const int n_k_heads = cfg->ssm_n_group;
    const int n_v_heads = cfg->ssm_dt_rank;
    const int kv_groups = n_v_heads / n_k_heads;
    const int key_dim = S * n_k_heads;
    const int val_dim = S * n_v_heads;
    const float eps = cfg->rms_norm_eps;
    fc_cuda_ctx *ctx = d->ctx;

    double sst0 = 0.0;
    if (prof_sub_on()) { fc_sync_ctx(ctx); sst0 = prof_now(); }

    /* Four projections from x_norm (same unfused order as gpu_ssm_block). */
    if (gpu_matvec_nx(&d->attn_qkv, b->x_norm, b->s_qkv, n) < 0 ||
        gpu_matvec_nx(&d->attn_gate, b->x_norm, b->s_z, n) < 0 ||
        gpu_matvec_nx(&d->ssm_beta, b->x_norm, b->s_beta, n) < 0 ||
        gpu_matvec_nx(&d->ssm_alpha, b->x_norm, b->s_alpha, n) < 0)
        return -1;
    fc_sigmoid_ctx(ctx, b->s_beta, n * n_v_heads);
    fc_ssm_decay_gate_nx_ctx(ctx, b->s_alpha, d->ssm_dt.buf, d->ssm_a.buf,
                             b->s_g, n_v_heads, n);
    ssm_lap(SSMP_PROJ, ctx, &sst0);

    /* Conv1d + recurrence: scans over the n positions (state in place),
     * with per-position snapshots in spec mode. */
    fc_ssm_conv1d_silu_seq_ctx(ctx, b->s_qkv, d->ssm_conv1d.buf,
                               d->conv_state_g, b->s_conv, conv_dim,
                               cfg->ssm_conv, n,
                               spec ? d->conv_snaps : NULL);
    ssm_lap(SSMP_CONV, ctx, &sst0);
    fc_ssm_qknorm_gqa_nx_ctx(ctx, b->s_conv, b->s_qg, b->s_kg, b->s_vg,
                             n_k_heads, kv_groups, S, key_dim, val_dim,
                             eps, n);
    ssm_lap(SSMP_QKNORM, ctx, &sst0);
    fc_ssm_recurrence_seq_ctx(ctx, d->ssm_state_g, b->s_qg, b->s_kg, b->s_vg,
                              b->s_g, b->s_beta, b->s_y, n_v_heads, S, n,
                              qw_scale_inv_sqrt(S),
                              spec ? d->ssm_snaps : NULL);
    ssm_lap(SSMP_RECUR, ctx, &sst0);

    /* Gated norm (heads are uniform across positions, so the single-token
     * kernel covers the batch with n*n_v_heads head-blocks) + out proj. */
    fc_ssm_gated_norm_ctx(ctx, b->s_y, d->ssm_norm.buf, b->s_z, b->s_gated,
                          n * n_v_heads, S, eps);
    if (core_only)
        return 0;
    if (gpu_matvec_nx(&d->ssm_out, b->s_gated, b->block_out, n) < 0)
        return -1;
    fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
    ssm_lap(SSMP_OUT, ctx, &sst0);
    return 0;
}

/* ============================================================
 * ON-DIE FLASH ATTENTION — batched prefill
 *
 * The long-context lever: the host causal GQA softmax is O(N^2) on the
 * 2-core K80 host (87.5% of an 8k prefill), so the batched path keeps a
 * GPU-resident copy of each attention layer's KV ring cache and runs the
 * whole attention block on the die (device q/z split -> QK-norm/RoPE ->
 * KV append -> fc_attn_flash_prefill -> sigmoid gate). The host cache is
 * still maintained as the mirror (a small D2H per call) so the single-
 * token decode path and cross-path re-syncs stay correct.
 *
 * NOT bit-exact vs the host softmax: the flash kernel's online softmax
 * and 16-lane dots reduce in a different order (~1e-6 relative drift,
 * far below Q3_K noise). Gate: argmax-exact on the reference sequence +
 * eval-harness A/B (QWEN35_FLASH=0 vs 1). QWEN35_FLASH=0 disables.
 * ============================================================ */

/* QWEN35_FLASH gate (default ON; =0 keeps the host-softmax path). */
static int flash_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_FLASH");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* StreamingLLM sliding-window attention gate. QWEN35_ATTN_WINDOW=<W> is the
 * recent-window size; 0/unset => full attention (the A/B baseline, a true
 * no-op on both the flash and host-softmax paths). QWEN35_ATTN_SINK=<S> is the
 * attention-sink prefix (default 4). Parsed once, cached. Every attention
 * layer (the 10 global i%4==3 layers) uses the same window. */
static int attn_window(void)
{
    static int w = -1;
    if (w < 0) {
        const char *env = getenv("QWEN35_ATTN_WINDOW");
        long v = env ? strtol(env, NULL, 10) : 0;
        w = (v > 0) ? (int)v : 0;
    }
    return w;
}

static int attn_sink(void)
{
    static int s = -1;
    if (s < 0) {
        const char *env = getenv("QWEN35_ATTN_SINK");
        long v = env ? strtol(env, NULL, 10) : 4;
        s = (v >= 0) ? (int)v : 4;
    }
    return s;
}

/* ============================================================
 * Phase 10b (INV3) — lazy host KV mirror helpers
 *
 * In lazy mode (QWEN35_HOST_KV_MIRROR unset/0) the batched slots path stops
 * writing the per-token host KV mirror: the GPU rings are authoritative for
 * a bound session, rebinds are seeded device-to-device, and the mirror is
 * materialized from a ring only when a real consumer needs host bytes
 * (session export, sys-prefix capture, prefill yield, host-softmax
 * fallback, ring realloc). kv_mirror_stale = 1 marks "the mirror is missing
 * rows that live in a GPU ring".
 * ============================================================ */

static int qw_kv_mirror_lazy(void)
{
    return !pn_qwen35moe_host_kv_mirror_eager();
}

static size_t qw_kv_row_bytes(const pn_qwen35moe_kv *kv)
{
    return kv->q4 ? (size_t)pn_kvq4_row_bytes(kv->head_dim)
                  : (size_t)kv->head_dim * sizeof(float);
}

/* Per-slot fp16 sink / window shadow byte sizes (q4 mode) of layer struct d
 * (a QWEN35_TP half holds its own heads only). */
static size_t qw_kv_sink_bytes(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv)
{
    return (size_t)qw_kv_nh(d, kv) * PN_KVQ4_SINKMAX * kv->head_dim *
           sizeof(uint16_t);
}

static size_t qw_kv_win_bytes(const gpu_attn_layer *d, const pn_qwen35moe_kv *kv)
{
    return (size_t)qw_kv_nh(d, kv) * PN_KVQ4_WMAX * kv->head_dim *
           sizeof(uint16_t);
}

/* Materialize one layer's host KV mirror rows [0, rows) (plus the fp16
 * sink/window shadows) from whichever GPU ring currently holds this kv's
 * content — the single-user ring or a decode slot ring, identified by the
 * (pointer, generation) watermark. Blocking D2H on the layer's ctx; caller
 * holds the GPU turn. Returns 0 and clears kv_mirror_stale on success, -1
 * if no ring holds the content (mirror left stale — the caller must treat
 * the mirror as unusable, NOT read it). */
static int qw_kv_layer_materialize(gpu_attn_layer *d, pn_qwen35moe_kv *kv,
                                   int rows)
{
    const void *host_id = kv->q4 ? (const void *)kv->kq
                                 : (const void *)kv->k;
    size_t row_b = qw_kv_row_bytes(kv);
    /* QWEN35_TP: this struct's heads [h0, h0 + nh) land at their host rows; the
     * caller materializes both halves (qw_kv_layer_materialize_all). */
    const int nh = qw_kv_nh(d, kv), h0 = qw_kv_h0(d, kv);
    const size_t head_h = (size_t)h0 * kv->max_seq * row_b;
    const size_t sink_h = (size_t)h0 * PN_KVQ4_SINKMAX * kv->head_dim * sizeof(uint16_t);
    const size_t win_h = (size_t)h0 * PN_KVQ4_WMAX * kv->head_dim * sizeof(uint16_t);
    unsigned char *kdst = (kv->q4 ? kv->kq : (unsigned char *)kv->k) + head_h;
    unsigned char *vdst = (kv->q4 ? kv->vq : (unsigned char *)kv->v) + head_h;
    if (rows < 0) {
        /* "As many rows as the source holds": the caller has no pos at hand
         * (slot bind); take the matching ring's own watermark. */
        rows = 0;
        if (d->k_cache_g && d->kv_host_id == host_id &&
            d->kv_gen == kv->content_gen && d->kv_synced_pos > rows)
            rows = d->kv_synced_pos;
        if (d->k_slots_g && d->kv_slots_cap_seq == kv->max_seq)
            for (int su = 0; su < d->kv_slots_cap_n; su++)
                if (d->kv_slot_host_id[su] == host_id &&
                    d->kv_slot_gen[su] == kv->content_gen &&
                    d->kv_slot_synced[su] > rows)
                    rows = d->kv_slot_synced[su];
        if (rows == 0)
            return -1;   /* no ring holds this content */
    }
    if (rows > kv->max_seq)
        rows = kv->max_seq;
    if (rows <= 0) {
        kv->kv_mirror_stale = 0;
        return 0;
    }

    /* Single-user ring (post-prefill / single-user decode). */
    if (d->k_cache_g && d->v_cache_g && d->cache_cap == kv->max_seq &&
        d->kv_host_id == host_id && d->kv_gen == kv->content_gen &&
        d->kv_synced_pos >= rows) {
        for (int hd = 0; hd < nh; hd++) {
            size_t off = (size_t)hd * kv->max_seq * row_b;
            fc_buffer_to_cpu_at(d->k_cache_g, kdst + off, off,
                                (size_t)rows * row_b);
            fc_buffer_to_cpu_at(d->v_cache_g, vdst + off, off,
                                (size_t)rows * row_b);
        }
        if (kv->q4 && d->k_sink_g) {
            const size_t sink_b = qw_kv_sink_bytes(d, kv), win_b = qw_kv_win_bytes(d, kv);
            fc_buffer_to_cpu_at(d->k_sink_g, (unsigned char *)kv->k_sink + sink_h, 0, sink_b);
            fc_buffer_to_cpu_at(d->v_sink_g, (unsigned char *)kv->v_sink + sink_h, 0, sink_b);
            fc_buffer_to_cpu_at(d->k_win_g, (unsigned char *)kv->k_win + win_h, 0, win_b);
            fc_buffer_to_cpu_at(d->v_win_g, (unsigned char *)kv->v_win + win_h, 0, win_b);
        }
        kv->kv_mirror_stale = 0;
        return 0;
    }

    /* Decode slot ring (batched sessions). */
    if (d->k_slots_g && d->v_slots_g && d->kv_slots_cap_seq == kv->max_seq &&
        d->kv_slots_q4 == kv->q4) {
        for (int su = 0; su < d->kv_slots_cap_n; su++) {
            if (d->kv_slot_host_id[su] != host_id ||
                d->kv_slot_gen[su] != kv->content_gen ||
                d->kv_slot_synced[su] < rows)
                continue;
            for (int hd = 0; hd < nh; hd++) {
                size_t dst_off = (size_t)hd * kv->max_seq * row_b;
                size_t src_off = ((size_t)su * nh + hd) *
                                 kv->max_seq * row_b;
                fc_buffer_to_cpu_at(d->k_slots_g, kdst + dst_off, src_off,
                                    (size_t)rows * row_b);
                fc_buffer_to_cpu_at(d->v_slots_g, vdst + dst_off, src_off,
                                    (size_t)rows * row_b);
            }
            if (kv->q4 && d->k_sink_slots_g) {
                size_t sink_b = qw_kv_sink_bytes(d, kv);
                size_t win_b = qw_kv_win_bytes(d, kv);
                fc_buffer_to_cpu_at(d->k_sink_slots_g, (unsigned char *)kv->k_sink + sink_h,
                                    (size_t)su * sink_b, sink_b);
                fc_buffer_to_cpu_at(d->v_sink_slots_g, (unsigned char *)kv->v_sink + sink_h,
                                    (size_t)su * sink_b, sink_b);
                fc_buffer_to_cpu_at(d->k_win_slots_g, (unsigned char *)kv->k_win + win_h,
                                    (size_t)su * win_b, win_b);
                fc_buffer_to_cpu_at(d->v_win_slots_g, (unsigned char *)kv->v_win + win_h,
                                    (size_t)su * win_b, win_b);
            }
            kv->kv_mirror_stale = 0;
            return 0;
        }
    }
    return -1;
}

/* qw_kv_layer_materialize over every half of attention layer struct idx (both
 * QWEN35_TP ranks; just idx otherwise). The mirror is clean only if all landed. */
static int qw_kv_layer_materialize_all(pn_qwen35moe_gpu *g, int idx, pn_qwen35moe_kv *kv,
                                       int rows)
{
    if (!g->tp)
        return qw_kv_layer_materialize(&g->attn[idx], kv, rows);
    int rc = qw_kv_layer_materialize(&g->attn[idx], kv, rows);
    if (qw_kv_layer_materialize(&g->attn[idx + g->tp_n_attn], kv, rows) != 0 || rc != 0) {
        kv->kv_mirror_stale = 1;
        return -1;
    }
    return 0;
}

/* Ensure this layer's GPU KV ring cache exists and holds rows [0, pos0)
 * of `kv`, uploading missing rows from the host mirror (contiguous per
 * head) when the bound state changed or host-side appends ran past the
 * GPU copy. Rows >= pos0 need no sync: this batch call appends them
 * before any query reads them. Returns 0 on success. */
static int gpu_attn_kv_cache_sync(gpu_attn_layer *d,
                                  const pn_qwen35moe_kv *kv, int pos0)
{
    /* Per-position row size: q4 -> ggml-Q4_0 bytes, else fp32. The GPU cache
     * mirrors the host cache byte-for-byte (host is quantized too under q4), so
     * a re-upload is a plain byte copy in either mode. */
    const int q4 = kv->q4;
    size_t row_b = q4 ? (size_t)pn_kvq4_row_bytes(kv->head_dim)
                      : (size_t)kv->head_dim * sizeof(float);
    size_t head_b = (size_t)kv->max_seq * row_b;
    const void *host_id = q4 ? (const void *)kv->kq : (const void *)kv->k;
    /* QWEN35_TP: this struct holds heads [h0, h0 + nh) of the host ring. */
    const int nh = qw_kv_nh(d, kv), h0 = qw_kv_h0(d, kv);

    if (!d->k_cache_g || d->cache_cap < kv->max_seq) {
        fc_buffer_free(d->k_cache_g);
        fc_buffer_free(d->v_cache_g);
        fc_buffer_free(d->k_sink_g);
        fc_buffer_free(d->v_sink_g);
        fc_buffer_free(d->k_win_g);
        fc_buffer_free(d->v_win_g);
        fc_buffer_free(d->k_win_snap_g);
        fc_buffer_free(d->v_win_snap_g);
        d->k_sink_g = d->v_sink_g = d->k_win_g = d->v_win_g = NULL;
        d->k_win_snap_g = d->v_win_snap_g = NULL;
        d->spec_win_cap = d->spec_win_n = 0;
        size_t bytes = (size_t)nh * head_b;
        d->k_cache_g = fc_buffer_alloc_ctx(d->ctx, bytes);
        d->v_cache_g = fc_buffer_alloc_ctx(d->ctx, bytes);
        d->cache_cap = (d->k_cache_g && d->v_cache_g) ? kv->max_seq : 0;
        if (q4 && d->cache_cap) {
            /* fp16 sink + recent-window shadow, sized to the maxima (host and
             * device rings are indexed identically). */
            size_t sink_b = qw_kv_sink_bytes(d, kv);
            size_t win_b = qw_kv_win_bytes(d, kv);
            d->k_sink_g = fc_buffer_alloc_ctx(d->ctx, sink_b);
            d->v_sink_g = fc_buffer_alloc_ctx(d->ctx, sink_b);
            d->k_win_g = fc_buffer_alloc_ctx(d->ctx, win_b);
            d->v_win_g = fc_buffer_alloc_ctx(d->ctx, win_b);
            if (!d->k_sink_g || !d->v_sink_g || !d->k_win_g || !d->v_win_g)
                d->cache_cap = 0;
        }
        d->kv_host_id = NULL;
        if (!d->cache_cap)
            return -1;
    }

    /* A sequence RESTART (pos0 == 0) invalidates the whole GPU cache for this
     * layer: on a REUSED state (same host_id — same kq buffer — but reset +
     * re-prefilled) the incremental watermark and, critically, the fp16
     * q4 sink/window SHADOW still hold the PREVIOUS sequence's residue.  The
     * append below only (re)writes [pos0, pos0+n); the shadow re-sync only
     * runs when synced < pos0, which is vacuously false at pos0 == 0 — so
     * without this the restarted sequence attends stale sink/window K/V and
     * its prefill logits diverge (observed: a reused serve slot's long-prompt
     * completion differed from a fresh state).  Drop the watermark so the
     * block below force-re-syncs the shadow from the (reset-zeroed, then
     * re-prefilled) host mirror. */
    if (pos0 == 0) {
        d->kv_host_id    = NULL;
        d->kv_synced_pos = 0;
    }

    int synced = (d->kv_host_id == host_id &&
                  d->kv_gen == kv->content_gen) ? d->kv_synced_pos : 0;
    if (synced > pos0)
        synced = pos0; /* spec re-run: stale tail rows get re-appended */
    /* Phase 9.2: a speculative verify used to force `synced = 0` here and
     * re-upload the whole [0, pos0) window from the host KV mirror on every
     * call — 48 us per context position, ~90 ms of the verify's 110 ms at
     * depth 1024, and outright BROKEN since Phase 10b/INV3 made that mirror
     * lazy (it is stale, so the refusal below fired and verify returned -1).
     *
     * It is not needed. The committed frontier [0, pos0) in this ring is
     * exactly as trustworthy for a verify as it is for the prefill and
     * decode appends that wrote it: the watermark tracks it, the clamp above
     * re-appends any tail past pos0, and rejected draft rows only ever sit at
     * ring slots >= pos0 (attention at position p reads only [0, p], so they
     * are never attended, and the next append overwrites them). What the old
     * comment called "holes" is the one case the watermark really cannot see
     * — rows committed into a DIFFERENT ring (a decode slot) — and that is
     * handled by the D2D branch below, which now runs for spec too.
     *
     * The accepted rows a verify appends are folded into the watermark by
     * pn_qwen35moe_spec_commit_cuda, which is what closes the loop. */
    int host_uploaded = 0;
    if (synced < pos0) {
        /* Phase 10b: prefer sourcing the missing rows on-die from a decode
         * slot ring that already holds this content (the multi-turn resume
         * case: the previous turn decoded in a slot ring; the delta prefill
         * runs here in the single-user ring). D2D, no host round-trip.
         * Slot rings only ever hold committed content — the batched decode
         * path never speculates — so this is a valid source for a verify. */
        int d2d_done = 0;
        if (d->k_slots_g && d->v_slots_g &&
            d->kv_slots_cap_seq == kv->max_seq && d->kv_slots_q4 == q4) {
            for (int su = 0; su < d->kv_slots_cap_n; su++) {
                if (d->kv_slot_host_id[su] != host_id ||
                    d->kv_slot_gen[su] != kv->content_gen ||
                    d->kv_slot_synced[su] < pos0)
                    continue;
                for (int hd = 0; hd < nh; hd++) {
                    size_t dst_off = (size_t)hd * head_b +
                                     (size_t)synced * row_b;
                    size_t src_off = ((size_t)su * nh + hd) *
                                     head_b + (size_t)synced * row_b;
                    size_t bytes = (size_t)(pos0 - synced) * row_b;
                    fc_buffer_copy_d2d_ctx(d->ctx, d->k_cache_g, dst_off,
                                           d->k_slots_g, src_off, bytes);
                    fc_buffer_copy_d2d_ctx(d->ctx, d->v_cache_g, dst_off,
                                           d->v_slots_g, src_off, bytes);
                }
                if (q4 && d->k_sink_slots_g && d->k_sink_g) {
                    size_t sink_b = qw_kv_sink_bytes(d, kv);
                    size_t win_b = qw_kv_win_bytes(d, kv);
                    fc_buffer_copy_d2d_ctx(d->ctx, d->k_sink_g, 0,
                                           d->k_sink_slots_g,
                                           (size_t)su * sink_b, sink_b);
                    fc_buffer_copy_d2d_ctx(d->ctx, d->v_sink_g, 0,
                                           d->v_sink_slots_g,
                                           (size_t)su * sink_b, sink_b);
                    fc_buffer_copy_d2d_ctx(d->ctx, d->k_win_g, 0,
                                           d->k_win_slots_g,
                                           (size_t)su * win_b, win_b);
                    fc_buffer_copy_d2d_ctx(d->ctx, d->v_win_g, 0,
                                           d->v_win_slots_g,
                                           (size_t)su * win_b, win_b);
                }
                d2d_done = 1;
                break;
            }
        }
        if (!d2d_done) {
            if (qw_kv_mirror_lazy() && kv->kv_mirror_stale) {
                /* The mirror is missing rows and no ring holds them — every
                 * lazy-mode flow materializes before reaching here, so this
                 * is an invariant breach. Refuse to upload garbage. */
                fprintf(stderr, "[attn] kv_cache_sync: rows [%d,%d) needed, "
                        "host mirror STALE and no GPU ring source — refusing "
                        "stale upload\n", synced, pos0);
                return -1;
            }
            const unsigned char *ksrc = q4 ? kv->kq
                                           : (const unsigned char *)kv->k;
            const unsigned char *vsrc = q4 ? kv->vq
                                           : (const unsigned char *)kv->v;
            for (int hd = 0; hd < nh; hd++) {
                size_t off_b = (size_t)hd * head_b + (size_t)synced * row_b;
                size_t host_b = off_b + (size_t)h0 * head_b;
                size_t bytes = (size_t)(pos0 - synced) * row_b;
                fc_buffer_from_cpu_at(d->k_cache_g, off_b, ksrc + host_b, bytes);
                fc_buffer_from_cpu_at(d->v_cache_g, off_b, vsrc + host_b, bytes);
            }
            host_uploaded = 1;
        }
    }
    /* Re-sync the fp16 q4 sink/window shadow from the host mirror on any
     * non-incremental HOST entry: a host re-upload (the D2D branch above
     * refreshed the shadow on-die already) OR a sequence RESTART (pos0 == 0
     * — a reused state whose shadow still holds the previous sequence's
     * residue; the append below only rewrites the ring rows, never the
     * shadow, and at pos0 == 0 the row-upload branch above is skipped).
     * The incremental append path keeps the shadow live so this is cheap
     * and rare (a few hundred KB/layer, once per sequence start). */
    if (q4 && (host_uploaded || pos0 == 0)) {
        size_t sink_b = qw_kv_sink_bytes(d, kv);
        size_t win_b = qw_kv_win_bytes(d, kv);
        size_t sink_h = (size_t)h0 * PN_KVQ4_SINKMAX * kv->head_dim * sizeof(uint16_t);
        size_t win_h = (size_t)h0 * PN_KVQ4_WMAX * kv->head_dim * sizeof(uint16_t);
        fc_buffer_from_cpu_at(d->k_sink_g, 0, (const unsigned char *)kv->k_sink + sink_h, sink_b);
        fc_buffer_from_cpu_at(d->v_sink_g, 0, (const unsigned char *)kv->v_sink + sink_h, sink_b);
        fc_buffer_from_cpu_at(d->k_win_g, 0, (const unsigned char *)kv->k_win + win_h, win_b);
        fc_buffer_from_cpu_at(d->v_win_g, 0, (const unsigned char *)kv->v_win + win_h, win_b);
    }
    d->kv_host_id = host_id;
    d->kv_gen = kv->content_gen;
    d->kv_synced_pos = pos0;
    return 0;
}

/* QWEN35_FLASH_DECODE gate (default ON; =0 keeps the host-softmax decode path;
 * A/B baseline). Set to any value except "0" to route single-token decode
 * attention through the on-die split flash kernel. */
static int flash_decode_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_FLASH_DECODE");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Target keys per decode split. QWEN35_FLASH_DECODE_KPS overrides the default
 * for tuning / forcing the multi-split combine path in tests; clamped to the
 * tile width so split_len rounding stays sane. */
static int dec_keys_per_split(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_FLASH_DECODE_KPS");
        v = env ? atoi(env) : PN_QWEN35MOE_DEC_KEYS_PER_SPLIT;
        if (v < 16)   /* FC_ATTN_TILE_KEYS: split_len is rounded to this */
            v = 16;
    }
    return v;
}

/* On-die decode flash attention: a single query (the new token at `pos`)
 * attends the cached context [0, pos] entirely on the die. Mirrors
 * gpu_attn_block_batch_flash with npos=1 but keeps everything in the per-die
 * scratch and splits the KV sequence across blocks (fc_attn_flash_decode) so
 * the die is not left near-idle. On success attn_out (sc->a_out) holds the
 * sigmoid-gated attention output, ready for the shared output projection.
 * Returns 0 = ran, 1 = unavailable (caller uses the host path), <0 = error. */
static int gpu_attn_block_decode_flash(gpu_attn_layer *d,
                                       const pn_qwen35moe_config *cfg,
                                       fc_buffer *x_norm, pn_qwen35moe_kv *kv,
                                       int pos, gpu_die_scratch *sc,
                                       gpu_host_scratch *h)
{
    fc_cuda_ctx *ctx = d->ctx;
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int attn_dim = n_head * head_dim;
    const int max_seq = kv->max_seq;
    const int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;

    /* Availability: flag on, kernels present, no ring wrap in the window, and
     * the q4 decode kernel present when the cache is quantized. */
    if (!flash_decode_on() || !fc_attn_flash_available_ctx(ctx) ||
        !fc_attn_flash_decode_available_ctx(ctx) || !sc->a_z ||
        pos + 1 > max_seq ||
        (kv->q4 && !fc_attn_flash_decode_q4_available_ctx(ctx)))
        return 1;

    /* Split count from context length: ~KEYS_PER_SPLIT keys/block, capped so
     * the partial scratch stays bounded. split_len is a multiple of the tile
     * width so tile boundaries line up. */
    int kps = dec_keys_per_split();
    int n_splits = (seq_len + kps - 1) / kps;
    if (n_splits < 1)
        n_splits = 1;
    if (n_splits > PN_QWEN35MOE_DEC_SPLIT_MAX)
        n_splits = PN_QWEN35MOE_DEC_SPLIT_MAX;
    int split_len = (seq_len + n_splits - 1) / n_splits;
    split_len = (split_len + 15) & ~15;
    /* split_len rounding can leave the tail splits empty; drop them so no
     * fully-masked block is launched (combine ignores -inf partials anyway). */
    n_splits = (seq_len + split_len - 1) / split_len;

    /* Partial scratch (m, l, acc) on this die, allocated once to the cap. */
    if (!d->dec_acc_g || d->dec_splits_cap < PN_QWEN35MOE_DEC_SPLIT_MAX) {
        fc_buffer_free(d->dec_m_g);
        fc_buffer_free(d->dec_l_g);
        fc_buffer_free(d->dec_acc_g);
        size_t np = (size_t)n_head * PN_QWEN35MOE_DEC_SPLIT_MAX;
        d->dec_m_g = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        d->dec_l_g = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        d->dec_acc_g = fc_buffer_alloc_ctx(ctx,
                                           np * (size_t)head_dim * sizeof(float));
        d->dec_splits_cap = (d->dec_m_g && d->dec_l_g && d->dec_acc_g)
                                ? PN_QWEN35MOE_DEC_SPLIT_MAX : 0;
        if (!d->dec_splits_cap)
            return -1;
    }

    /* 1. Q+gate / K / V projections (single position). */
    gpu_matvec(&d->attn_q, x_norm, sc->a_qg, h->a, h->b);
    gpu_matvec(&d->attn_k, x_norm, sc->a_k, h->a, h->b);
    gpu_matvec(&d->attn_v, x_norm, sc->a_v, h->a, h->b);

    /* 2.-4. Device q/gate split (z persists in a_z for the output gate),
     * QK-RMSNorm, partial NEOX RoPE — all on the die. */
    fc_attn_qz_split_ctx(ctx, sc->a_qg, sc->a_q, sc->a_z, n_head, head_dim, 1);
    fc_qk_rms_norm_ctx(ctx, sc->a_q, d->attn_q_norm.buf, n_head, head_dim,
                       cfg->rms_norm_eps);
    fc_qk_rms_norm_ctx(ctx, sc->a_k, d->attn_k_norm.buf, n_head_kv, head_dim,
                       cfg->rms_norm_eps);
    fc_rope_neox_partial_ctx(ctx, sc->a_q, n_head, head_dim, cfg->rope_dim, pos,
                             cfg->rope_theta);
    fc_rope_neox_partial_ctx(ctx, sc->a_k, n_head_kv, head_dim, cfg->rope_dim,
                             pos, cfg->rope_theta);

    /* 5. Sync the GPU KV ring to rows [0, pos) (re-uploads any host-only
     * appends), append this token's row, split flash over [0, pos]. */
    if (gpu_attn_kv_cache_sync(d, kv, pos) < 0)
        return -1;
    float scale = qw_scale_inv_sqrt(head_dim);
    if (kv->q4) {
        fc_attn_kv_append_q4_ctx(ctx, sc->a_k, sc->a_v, d->k_cache_g,
                                 d->v_cache_g, d->k_sink_g, d->v_sink_g,
                                 d->k_win_g, d->v_win_g, n_head_kv, head_dim,
                                 max_seq, pos, 1, pn_kvq4_sink(),
                                 pn_kvq4_win());
        d->kv_synced_pos = pos + 1;
        if (fc_attn_flash_decode_q4_ctx(ctx, sc->a_q, d->k_cache_g,
                                        d->v_cache_g, d->k_sink_g, d->v_sink_g,
                                        d->k_win_g, d->v_win_g, d->dec_m_g,
                                        d->dec_l_g, d->dec_acc_g, sc->a_out,
                                        n_head, n_head_kv, head_dim, max_seq,
                                        seq_len, n_splits, split_len,
                                        pn_kvq4_sink(), pn_kvq4_win(),
                                        attn_sink(), attn_window(),
                                        scale) < 0)
            return -1;
    } else {
        fc_attn_kv_append_ctx(ctx, sc->a_k, sc->a_v, d->k_cache_g, d->v_cache_g,
                              n_head_kv, head_dim, max_seq, pos, 1);
        d->kv_synced_pos = pos + 1;
        if (fc_attn_flash_decode_ctx(ctx, sc->a_q, d->k_cache_g, d->v_cache_g,
                                     d->dec_m_g, d->dec_l_g, d->dec_acc_g,
                                     sc->a_out, n_head, n_head_kv, head_dim,
                                     max_seq, seq_len, n_splits, split_len,
                                     attn_sink(), attn_window(), scale) < 0)
            return -1;
    }

    /* 6. Sigmoid output gate on the die: attn_out *= sigmoid(z). */
    fc_sigmoid_ctx(ctx, sc->a_z, attn_dim);
    fc_mul_ctx(ctx, sc->a_out, sc->a_z, sc->a_out, attn_dim);

    /* 7. B2 — deferred KV host mirror: mark the host mirror stale instead of
     * paying ~2 D2H/attn-layer per decode token. The GPU ring is the live
     * source of truth; pn_qwen35moe_gpu_state_to_host() refreshes the host
     * mirror on-demand when this session is preserved (session switch / LRU
     * eviction), before the GPU KV buffers are reused for another session. */
    kv->kv_mirror_stale = 1;
    d->flash_ran = 1;   /* segment-graph gate: ring + dec_* scratch exist */
    return 0;
}

/* ============================================================
 * Full-step segment graph (C1, QWEN35_GRAPH_FULL)
 *
 * One captured CUDA graph per die segment replays EVERY layer body —
 * SSM and attention interleaved — with a single cuGraphLaunch per
 * token (~300 launches -> 1 on a 20-layer segment). Made possible by
 * B2 (no per-token KV-mirror D2H) plus the device-position kernels:
 * the only per-token variance inside an attention body is the token
 * position, so it is moved into a device int (sc->dec_pos, one 4-byte
 * H2D per die per token, OUTSIDE the graph) and the RoPE / KV-append /
 * split-flash / combine kernels read it from device memory, recomputing
 * the split plan on-device with the exact host formula (bit-exact).
 *
 * What stays OUTSIDE the graph (host-synchronous or per-token-variant):
 *   - the token-embedding row fetch + residual-stream H2D seed
 *   - the 2-die boundary hand-off (sync + D2H + barswap + H2D)
 *   - the end-of-step LM head top-k logits fetch (D2H)
 *   - SSM state re-seed H2D on session switch; KV ring re-sync H2D on
 *     session resume (both no-ops in steady-state decode)
 *   - host bookkeeping (kv_synced_pos watermark, kv_mirror_stale)
 * ============================================================ */

/* Graph-capturable attention-layer body: the exact launch sequence of
 * gpu_attn_block's flash-decode path + out-proj + post-norm + MoE, with the
 * three position-dependent host-arg kernels swapped for their device-pos
 * variants (rope_pos, kv_append*_dev, flash_decode*_dev reading sc->dec_pos).
 * No debug dumps / prof syncs — the segment graph only engages with both off.
 * Caller has done the per-token host work (KV ring sync, dec_pos upload) and
 * guarantees flash eligibility (no ring wrap, dec scratch + ring allocated,
 * grouped MoE). */
static int gpu_attn_layer_body_graph(gpu_attn_layer *d,
                                     const pn_qwen35moe_config *cfg,
                                     fc_buffer *hidden, fc_buffer *x_norm,
                                     fc_buffer *block_out,
                                     pn_qwen35moe_kv *kv,
                                     gpu_die_scratch *sc, gpu_host_scratch *h)
{
    fc_cuda_ctx *ctx = d->ctx;
    const int c = cfg->n_embd;
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int attn_dim = n_head * head_dim;
    const int max_seq = kv->max_seq;
    const float eps = cfg->rms_norm_eps;
    float scale = qw_scale_inv_sqrt(head_dim);
    int rc;

    fc_rms_norm_ctx(ctx, hidden, d->attn_norm.buf, x_norm, c, eps);

    /* 1. Q+gate / K / V projections — same three launches as the direct
     * flash path (NOT the fused fx_qkv: keep the launch sequence identical
     * to the path the K80 hardware gate diffs against). */
    gpu_matvec(&d->attn_q, x_norm, sc->a_qg, h->a, h->b);
    gpu_matvec(&d->attn_k, x_norm, sc->a_k, h->a, h->b);
    gpu_matvec(&d->attn_v, x_norm, sc->a_v, h->a, h->b);

    /* 2.-4. Device q/gate split, QK-RMSNorm, RoPE at the DEVICE position. */
    fc_attn_qz_split_ctx(ctx, sc->a_qg, sc->a_q, sc->a_z, n_head, head_dim, 1);
    fc_qk_rms_norm_ctx(ctx, sc->a_q, d->attn_q_norm.buf, n_head, head_dim,
                       eps);
    fc_qk_rms_norm_ctx(ctx, sc->a_k, d->attn_k_norm.buf, n_head_kv, head_dim,
                       eps);
    fc_rope_neox_partial_pos_ctx(ctx, sc->a_q, n_head, head_dim, cfg->rope_dim,
                                 sc->dec_pos, cfg->rope_theta, 1);
    fc_rope_neox_partial_pos_ctx(ctx, sc->a_k, n_head_kv, head_dim,
                                 cfg->rope_dim, sc->dec_pos, cfg->rope_theta,
                                 1);

    /* 5. Ring append + split flash + combine, all at the device position. */
    if (kv->q4) {
        fc_attn_kv_append_q4_dev_ctx(ctx, sc->a_k, sc->a_v, d->k_cache_g,
                                     d->v_cache_g, d->k_sink_g, d->v_sink_g,
                                     d->k_win_g, d->v_win_g, n_head_kv,
                                     head_dim, max_seq, sc->dec_pos,
                                     pn_kvq4_sink(), pn_kvq4_win());
        rc = fc_attn_flash_decode_q4_dev_ctx(ctx, sc->a_q, d->k_cache_g,
                                             d->v_cache_g, d->k_sink_g,
                                             d->v_sink_g, d->k_win_g,
                                             d->v_win_g, d->dec_m_g,
                                             d->dec_l_g, d->dec_acc_g,
                                             sc->a_out, n_head, n_head_kv,
                                             head_dim, max_seq, sc->dec_pos,
                                             dec_keys_per_split(),
                                             PN_QWEN35MOE_DEC_SPLIT_MAX,
                                             pn_kvq4_sink(), pn_kvq4_win(),
                                             attn_sink(), attn_window(),
                                             scale);
    } else {
        fc_attn_kv_append_dev_ctx(ctx, sc->a_k, sc->a_v, d->k_cache_g,
                                  d->v_cache_g, n_head_kv, head_dim, max_seq,
                                  sc->dec_pos);
        rc = fc_attn_flash_decode_dev_ctx(ctx, sc->a_q, d->k_cache_g,
                                          d->v_cache_g, d->dec_m_g,
                                          d->dec_l_g, d->dec_acc_g, sc->a_out,
                                          n_head, n_head_kv, head_dim,
                                          max_seq, sc->dec_pos,
                                          dec_keys_per_split(),
                                          PN_QWEN35MOE_DEC_SPLIT_MAX,
                                          attn_sink(), attn_window(), scale);
    }
    if (rc < 0)
        return -1;

    /* 6. Sigmoid output gate. */
    fc_sigmoid_ctx(ctx, sc->a_z, attn_dim);
    fc_mul_ctx(ctx, sc->a_out, sc->a_z, sc->a_out, attn_dim);

    /* 7. Output projection (+residual), same fused/unfused choice as the
     * direct path. */
    {
        fc_fused_seg seg = gpu_seg_mv(&d->attn_output, block_out, 3);
        seg.aux1 = hidden;
        if (gpu_fused_run(ctx, &d->fx_out, &seg, 1, sc->a_out) != 0) {
            gpu_matvec(&d->attn_output, sc->a_out, block_out, h->a, h->b);
            fc_add_ctx(ctx, hidden, block_out, hidden, c);
        }
    }

    /* post_attention_norm + shared-MoE (+residual). */
    fc_rms_norm_ctx(ctx, hidden, d->post_attention_norm.buf, x_norm, c, eps);
    return gpu_moe_block(ctx, &d->moe, cfg, x_norm, block_out, hidden, sc, h);
}

static int gpu_segment_forward(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                               fc_cuda_ctx *ctx, int seg_idx, int lo, int hi,
                               fc_buffer *hidden, fc_buffer *x_norm,
                               fc_buffer *block_out, pn_qwen35moe_state *s,
                               int pos, gpu_host_scratch *h)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    gpu_seg_graph *seg = &g->seg[seg_idx];
    int layer, rc;

    /* Global gates (all env-latched — constant across the process). */
    if (!gpu_graph_enabled() || !gpu_graph_full_enabled() ||
        !flash_decode_on() || prof_on() || qwen35moe_dbg_enabled()) {
        pn_graph_capture_ineligible(&seg->capture,
            !gpu_graph_enabled() ? PN_GRAPH_INELIGIBLE_GRAPH_OFF
            : (prof_on() || qwen35moe_dbg_enabled())
                                 ? PN_GRAPH_INELIGIBLE_PROFILING
                                 : PN_GRAPH_INELIGIBLE_SLOTS_OFF);
        return 1;
    }
    if (pos < 1)
        return 1;   /* pos 0 runs direct: builds descriptors + scratch */
    /* A capture that FAILED gets bounded retries, not a permanent latch: the
     * cooldown is consumed here, the one point this segment reaches once per
     * token. The per-layer eligibility passes below can still refuse the step
     * afterwards; that costs at most one cooldown tick and no state. */
    if (!seg->graph && !pn_graph_capture_should_attempt(&seg->capture))
        return 1;

    gpu_die_scratch *sc = gpu_get_scratch(g, ctx, cfg);
    if (!sc)
        return -1;

    /* ---- Pass 1: eligibility. Pure host checks, no device work, so a
     * mid-loop refusal leaves nothing half-done. ---- */
    int q4 = -1, max_seq = 0;
    for (layer = lo; layer < hi; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
            gpu_attn_layer *d = &g->attn[idx];
            pn_qwen35moe_kv *kv = &s->kv[idx];
            if (!d->moe.grouped && !cfg->dense_ffn) {
                /* Permanent: expert quant decides `grouped` at upload, so this
                 * segment can never be captured on this model. */
                pn_graph_capture_ineligible(&seg->capture,
                                            PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);
                return 1;
            }
            if (!d->body_ran || !d->flash_ran)
                return 1;
            if (pos + 1 > kv->max_seq)
                return 1;   /* ring wrap: the host-softmax path owns this */
            if (q4 < 0) {
                q4 = kv->q4;
                max_seq = kv->max_seq;
                if (!fc_attn_decode_graph_available_ctx(ctx, q4))
                    return 1;
            } else if (kv->q4 != q4 || kv->max_seq != max_seq) {
                return 1;   /* mixed KV geometry — refuse, run direct */
            }
            if (d->cache_cap < kv->max_seq) {
                /* The direct path is about to REALLOCATE this layer's GPU
                 * ring — any captured graph holds the old addresses. Drop
                 * it and recapture once the new ring exists. */
                if (seg->graph) {
                    fc_graph_free(seg->graph);
                    seg->graph = NULL;
                }
                pn_graph_capture_dropped(&seg->capture);
                return 1;
            }
        } else {
            gpu_ssm_layer *d = &g->ssm[idx];
            if (!d->moe.grouped && !cfg->dense_ffn) {
                pn_graph_capture_ineligible(&seg->capture,
                                            PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);
                return 1;
            }
            if (!d->body_ran || !d->conv_state_g || !d->ssm_state_g)
                return 1;
        }
    }
    /* A replayed graph bakes in the KV geometry (max_seq strides, q4 kernel
     * choice). Refuse replay when the live state disagrees; the graph stays
     * for the geometry it was captured against. */
    if (seg->graph && q4 >= 0 &&
        (q4 != seg->cap_q4 || max_seq != seg->cap_max_seq))
        return 1;

    /* ---- Pass 2: per-token host pre-work (all OUTSIDE the graph). H2D
     * copies are synchronous and the stream is idle between tokens (the
     * logits fetch synced it), so nothing races the previous replay. ---- */
    for (layer = lo; layer < hi; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
            gpu_attn_layer *d = &g->attn[idx];
            /* Session resume / host-side appends: re-upload missing ring
             * rows [synced, pos). Steady-state decode: watermark == pos,
             * no copy. */
            if (gpu_attn_kv_cache_sync(d, &s->kv[idx], pos) < 0)
                return 1;
        } else {
            gpu_ssm_layer *d = &g->ssm[idx];
            pn_qwen35moe_ssm_state *st = &s->ssm[idx];
            /* Session switch: re-seed the GPU dual state CONTENTS (the
             * addresses the graph recorded never change). */
            if (d->host_state != (const void *)st)
                qw_ssm_seed_shared(g, d, st, pos);
        }
    }

    /* The per-token device position — the ONLY per-token input the captured
     * kernels read. */
    int32_t p32 = (int32_t)pos;
    fc_buffer_from_cpu(sc->dec_pos, &p32);

    if (seg->graph) {
        if (getenv("BONSAI_GRAPH_DBG")) { static int rc_[2]={0,0}; if(seg_idx<2 && (++rc_[seg_idx])%64==1) fprintf(stderr, "[graphdbg] REPLAY seg %d (count %d)\n", seg_idx, rc_[seg_idx]); }
        if (fc_graph_launch(seg->graph) != 0)
            return 1;   /* direct path takes this token; retry next */
        goto post;
    }

    /* ---- Capture lazily (mirrors the per-layer SSM capture): the recorded
     * launches do NOT execute; on any failure the direct path takes this token
     * and nothing was half-applied. ---- */
    if (fc_graph_begin_ctx(ctx) != 0) {
        pn_graph_capture_failed(&seg->capture);
        return 1;
    }
    rc = 0;
    for (layer = lo; layer < hi && rc == 0; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN)
            rc = gpu_attn_layer_body_graph(&g->attn[idx], cfg, hidden, x_norm,
                                           block_out, &s->kv[idx], sc, h);
        else
            rc = gpu_ssm_layer_body(&g->ssm[idx], cfg, layer, hidden, x_norm,
                                    block_out, sc, h);
    }
    seg->graph = fc_graph_end_ctx(ctx);
    if (pn_graph_capture_inject_now(&seg->capture) && seg->graph) {
        fc_graph_free(seg->graph);   /* fault injection, test only */
        seg->graph = NULL;
    }
    if (rc != 0 || !seg->graph) {
        if (seg->graph) {
            fc_graph_free(seg->graph);
            seg->graph = NULL;
        }
        pn_graph_capture_failed(&seg->capture);
        return 1;
    }
    seg->cap_q4 = q4;
    seg->cap_max_seq = max_seq;
    if (getenv("BONSAI_GRAPH_DBG"))
        fprintf(stderr, "[graphdbg] CAPTURED seg %d layers [%d,%d) dense=%d\n", seg_idx, lo, hi, cfg->dense_ffn);
    if (fc_graph_launch(seg->graph) != 0) {
        fc_graph_free(seg->graph);
        seg->graph = NULL;
        pn_graph_capture_failed(&seg->capture);
        return 1;
    }
    pn_graph_capture_succeeded(&seg->capture);
    /* This segment graph now owns every step in [lo,hi), so the per-layer SSM
     * graph slots underneath it are out of play — drop them from the residency
     * gauge until the segment graph is dropped and the per-layer path runs
     * again. Without this a per-layer slot frozen mid-retry keeps reporting a
     * deficit that nothing will ever work off: a healthy serve read
     * `live 2 / total 32 / failed 30`. */
    for (layer = lo; layer < hi; layer++) {
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
            pn_graph_capture_superseded(
                &g->ssm[pn_qwen35moe_model_layer_index(m, layer)].graph_capture);
    }

post:
    /* Host bookkeeping the graphed appends imply (the direct path does this
     * inside gpu_attn_block_decode_flash). */
    for (layer = lo; layer < hi; layer++) {
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
            continue;
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        g->attn[idx].kv_synced_pos = pos + 1;
        s->kv[idx].kv_mirror_stale = 1;   /* B2: GPU ring is source of truth */
    }
    return 0;
}

/* ---- Phase 9.2: speculative-verify undo for the q4 recent-window ring ----
 *
 * The fp16 window shadow is a ring of PN_KVQ4_WMAX slots addressed
 * `pos % PN_KVQ4_WMAX`, and the flash kernel reads a key at position s through
 * it whenever `s > cur_pos - W` for `W = pn_kvq4_win()`.  Appending a row at
 * position p therefore destroys the shadow of `p - WMAX`.  For prefill and
 * decode that is right (p - WMAX aged out of the W-wide window long ago).  For
 * a speculative verify a rejected draft row is never re-processed, so a slot it
 * clobbered is never rewritten — if that slot were still inside somebody's
 * window, every later query wanting it would read the wrong K/V, resyncing a
 * token or two later and looking like a sampling wobble rather than a cache
 * bug.
 *
 * WHEN THAT CAN HAPPEN — and at the depths we run, it cannot.  A verify block
 * spans [pos0, pos0+n); the oldest position anything can still want from the
 * shadow is > pos0 - W, and the newest eviction the block causes is
 * pos0 + n - 1 - WMAX.  So the block is harmless iff
 *
 *     n <= WMAX - W + 1        (129 rows at the default W = 128)
 *
 * and verify depths are D <= 16.  This machinery was written when the ring
 * lapped at W instead of WMAX, where the bound was `n <= 1` — i.e. always
 * unsafe.  Lapping at WMAX (foundry 9353d0b)
 * removed the hazard; qw_spec_win_needed now returns 0 in that regime and the
 * snapshot is skipped entirely.  It is kept, correct, for the out-of-envelope
 * case rather than deleted, because the bound is a property of two tunables
 * (QWEN35_KV_Q4_WIN and the caller's D) and not of the code.
 *
 * Restoring from the q4 body is not an option — that is the coarser copy the
 * shadow exists to avoid, so it would break bit-exactness.  Instead the verify
 * snapshots the slots it is about to touch, and spec_commit copies back the
 * ones belonging to rejected rows.
 *
 * A block that sits entirely inside the first ring lap has no prior occupant,
 * so it is skipped outright.
 */
static int qw_spec_win_needed(const pn_qwen35moe_kv *kv, int pos0, int n)
{
    int win = pn_kvq4_win();
    if (!kv->q4 || win <= 0 || pos0 + n <= PN_KVQ4_WMAX)
        return 0;                       /* no q4 shadow, or nothing to evict */
    if (n <= PN_KVQ4_WMAX - win + 1)
        return 0;                       /* every eviction is outside every
                                         * live W-wide window — see above */
    return 1;
}

/* Copy rows [t0, t1) of the verify block between the live ring and the undo
 * buffer.  The slots are consecutive modulo the ring, so this is one or two
 * runs per KV head, not one copy per row. */
static void qw_spec_win_copy(gpu_attn_layer *d, const pn_qwen35moe_kv *kv,
                             int pos0, int t0, int t1, int to_snap)
{
    const size_t row_b = (size_t)kv->head_dim * sizeof(uint16_t);
    for (int h = 0; h < kv->n_head_kv; h++) {
        int t = t0;
        while (t < t1) {
            int slot = (pos0 + t) % PN_KVQ4_WMAX;
            int run = PN_KVQ4_WMAX - slot;
            if (run > t1 - t) run = t1 - t;
            size_t ring_off = ((size_t)h * PN_KVQ4_WMAX + (size_t)slot) * row_b;
            size_t snap_off = ((size_t)h * (size_t)d->spec_win_cap +
                               (size_t)t) * row_b;
            size_t bytes = (size_t)run * row_b;
            if (to_snap) {
                fc_buffer_copy_d2d_ctx(d->ctx, d->k_win_snap_g, snap_off,
                                       d->k_win_g, ring_off, bytes);
                fc_buffer_copy_d2d_ctx(d->ctx, d->v_win_snap_g, snap_off,
                                       d->v_win_g, ring_off, bytes);
            } else {
                fc_buffer_copy_d2d_ctx(d->ctx, d->k_win_g, ring_off,
                                       d->k_win_snap_g, snap_off, bytes);
                fc_buffer_copy_d2d_ctx(d->ctx, d->v_win_g, ring_off,
                                       d->v_win_snap_g, snap_off, bytes);
            }
            t += run;
        }
    }
}

/* Called from the batched flash path before a SPECULATIVE append. */
static int qw_spec_win_snapshot(gpu_attn_layer *d, const pn_qwen35moe_kv *kv,
                                int pos0, int n)
{
    d->spec_win_n = 0;
    if (!qw_spec_win_needed(kv, pos0, n) || !d->k_win_g || !d->v_win_g)
        return 0;
    if (d->spec_win_cap < n) {
        fc_buffer_free(d->k_win_snap_g);
        fc_buffer_free(d->v_win_snap_g);
        size_t bytes = (size_t)kv->n_head_kv * (size_t)n *
                       (size_t)kv->head_dim * sizeof(uint16_t);
        d->k_win_snap_g = fc_buffer_alloc_ctx(d->ctx, bytes);
        d->v_win_snap_g = fc_buffer_alloc_ctx(d->ctx, bytes);
        d->spec_win_cap = (d->k_win_snap_g && d->v_win_snap_g) ? n : 0;
        if (!d->spec_win_cap)
            return -1;
    }
    qw_spec_win_copy(d, kv, pos0, 0, n, 1);
    d->spec_win_pos0 = pos0;
    d->spec_win_n = n;
    return 0;
}

/* Called from spec_commit for a PARTIAL accept: put back the window slots the
 * rejected rows [n_accept, n) evicted. */
static void qw_spec_win_restore(gpu_attn_layer *d, const pn_qwen35moe_kv *kv,
                                int pos0, int n, int n_accept)
{
    if (d->spec_win_n != n || d->spec_win_pos0 != pos0 || n_accept >= n)
        return;
    if (!qw_spec_win_needed(kv, pos0, n))
        return;
    qw_spec_win_copy(d, kv, pos0, n_accept, n, 0);
    d->spec_win_n = 0;
}

/* QWEN35_BATCH_VERIFY (default ON; =0 off): run a speculative verify's
 * attention through the R-row decode split kernel
 * (fc_attn_flash_decode_rows_q4_ctx) instead of the prefill flash kernel. The
 * prefill kernel walks the whole context serially in one block per (row, kv
 * head) — 2-10 blocks on a 13-SMX die, ~6x the decode step's per-position
 * cost — and reduces in a different order than decode. The rows kernel splits
 * the keys like the decode step, reads each K/V tile once for every row, and
 * is bit-exact per row against the plain decode step at that row's position.
 * Latched once. */
static int batch_verify_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_BATCH_VERIFY");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Verify attention over the single-user ring, rows already appended: a_q ->
 * a_out for the n rows at pos0.., with the sigmoid output gate folded into the
 * combine when the slots path folds it (QWEN35_FUSE bit7), exactly as the plain
 * decode step does. *gated reports which. Returns 0 = ran, 1 = not eligible
 * (caller runs the prefill kernel), -1 = error. */
static int gpu_attn_slot_plan_splits(int seq_len, int kps);
static int gpu_attn_verify_rows(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                const pn_qwen35moe_config *cfg,
                                const pn_qwen35moe_kv *kv, int pos0,
                                gpu_batch_scratch *b, int n, int *gated)
{
    fc_cuda_ctx *ctx = d->ctx;
    int di = gpu_die_index_of_ctx(g, ctx);
    *gated = 0;
    if (!batch_verify_on() || !kv->q4 || n > FC_ATTN_ROWS_MAX || di < 0 ||
        attn_window() > 0 || !fc_attn_flash_decode_rows_q4_available_ctx(ctx))
        return 1;
    if (!g->bv_acc_g[di]) {
        size_t np = (size_t)FC_ATTN_ROWS_MAX * cfg->n_head *
                    PN_QWEN35MOE_DEC_SPLIT_MAX;
        g->bv_m_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        g->bv_l_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        g->bv_acc_g[di] = fc_buffer_alloc_ctx(ctx, np * (size_t)cfg->head_dim *
                                                   sizeof(float));
        if (!g->bv_m_g[di] || !g->bv_l_g[di] || !g->bv_acc_g[di])
            return -1;
    }
    int fold = (gpu_fuse_mask() & 128) != 0;
    if (fc_attn_flash_decode_rows_q4_ctx(
            ctx, b->a_q, d->k_cache_g, d->v_cache_g, d->k_sink_g, d->v_sink_g,
            d->k_win_g, d->v_win_g, g->bv_m_g[di], g->bv_l_g[di],
            g->bv_acc_g[di], fold ? b->a_z : NULL, b->a_out, cfg->n_head,
            cfg->n_head_kv, cfg->head_dim, kv->max_seq, pos0, n,
            dec_keys_per_split(), PN_QWEN35MOE_DEC_SPLIT_MAX, pn_kvq4_sink(),
            pn_kvq4_win(), attn_sink(), attn_window(),
            qw_scale_inv_sqrt(cfg->head_dim)) != 0)
        return 1;
    *gated = fold;
    return 0;
}

/* Batched full-attention block, ON-DIE flash path (see section header).
 * Returns 0 on success, 1 if flash cannot run for this call (kernels
 * missing / disabled / ring wrap — caller uses the host path), -1 on
 * error. */
/* core_only (QWEN35_TP, lazy host KV mirror only): stop at the gated attention
 * output a_out, before the out projection and residual. */
static int gpu_attn_block_batch_flash(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                      const pn_qwen35moe_config *cfg,
                                      pn_qwen35moe_kv *kv, int pos0,
                                      gpu_batch_scratch *b, int n, int spec,
                                      int core_only)
{
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int attn_dim = n_head * head_dim;
    const int kv_dim = n_head_kv * head_dim;
    fc_cuda_ctx *ctx = d->ctx;

    if (!flash_on() || !fc_attn_flash_available_ctx(ctx) || !b->a_z ||
        pos0 + n > kv->max_seq ||
        (kv->q4 && !fc_attn_flash_q4_available_ctx(ctx)))
        return 1;

    double bt0 = prof_on() ? prof_now() : 0.0;

    /* 1. Joint Q+gate, K, V projections for all positions. */
    if (gpu_matvec_nx(&d->attn_q, b->x_norm, b->a_qg, n) < 0 ||
        gpu_matvec_nx(&d->attn_k, b->x_norm, b->a_k, n) < 0 ||
        gpu_matvec_nx(&d->attn_v, b->x_norm, b->a_v, n) < 0)
        return -1;
    bprof_lap(BPROF_ATTN_MV, ctx, &bt0);

    /* 2.-4. Device q/gate split (z persists in a_z for the output gate),
     * QK-RMSNorm, batched partial NEOX RoPE — same math as the host path,
     * minus the split's D2H/H2D round trip. */
    fc_attn_qz_split_ctx(ctx, b->a_qg, b->a_q, b->a_z, n_head, head_dim, n);
    fc_qk_rms_norm_ctx(ctx, b->a_q, d->attn_q_norm.buf, n * n_head, head_dim,
                       cfg->rms_norm_eps);
    fc_qk_rms_norm_ctx(ctx, b->a_k, d->attn_k_norm.buf, n * n_head_kv,
                       head_dim, cfg->rms_norm_eps);
    fc_rope_neox_partial_nx_ctx(ctx, b->a_q, n_head, head_dim, cfg->rope_dim,
                                pos0, cfg->rope_theta, n);
    fc_rope_neox_partial_nx_ctx(ctx, b->a_k, n_head_kv, head_dim,
                                cfg->rope_dim, pos0, cfg->rope_theta, n);
    bprof_lap(BPROF_ATTN_ROPE, ctx, &bt0);
    if (spec) {
        bv_trace_rows(ctx, "verify", "q", -1, b->a_q, n, attn_dim, NULL, pos0);
        bv_trace_rows(ctx, "verify", "k", -1, b->a_k, n, kv_dim, NULL, pos0);
        bv_trace_rows(ctx, "verify", "v", -1, b->a_v, n, kv_dim, NULL, pos0);
        bv_trace_rows(ctx, "verify", "z", -1, b->a_z, n, attn_dim, NULL, pos0);
    }

    /* 5. On-die attention: sync the GPU KV ring to pos0, append the n new
     * rows, causal GQA flash kernel, sigmoid output gate. */
    if (gpu_attn_kv_cache_sync(d, kv, pos0) < 0)
        return -1;
    /* A verify's rows may be rejected, and the q4 window append is destructive
     * one ring-lap back (see k_win_snap_g) — snapshot what it is about to
     * overwrite so spec_commit can undo the rejected tail. */
    if (spec && qw_spec_win_snapshot(d, kv, pos0, n) < 0)
        return -1;
    if (kv->q4)
        fc_attn_kv_append_q4_ctx(ctx, b->a_k, b->a_v, d->k_cache_g,
                                 d->v_cache_g, d->k_sink_g, d->v_sink_g,
                                 d->k_win_g, d->v_win_g, n_head_kv, head_dim,
                                 kv->max_seq, pos0, n, pn_kvq4_sink(),
                                 pn_kvq4_win());
    else
        fc_attn_kv_append_ctx(ctx, b->a_k, b->a_v, d->k_cache_g, d->v_cache_g,
                              n_head_kv, head_dim, kv->max_seq, pos0, n);
    /* A speculative verify appends n DRAFT rows to the GPU ring, but a partial
     * accept keeps only n_accept of them, so the watermark stays at the
     * committed frontier until the caller says how many landed —
     * pn_qwen35moe_spec_commit_cuda advances it by n_accept. A prefill append
     * is durable, so it advances by n immediately. */
    d->kv_synced_pos = spec ? pos0 : pos0 + n;
    int gated = 0, rows_rc = spec ? gpu_attn_verify_rows(g, d, cfg, kv, pos0,
                                                         b, n, &gated) : 1;
    if (rows_rc < 0)
        return -1;
    if (rows_rc == 0) {
        /* verify rows: a_out written by the decode split kernel */
    } else if (kv->q4) {
        if (fc_attn_flash_prefill_q4_ctx(ctx, b->a_q, d->k_cache_g,
                                         d->v_cache_g, d->k_sink_g, d->v_sink_g,
                                         d->k_win_g, d->v_win_g, b->a_out, n_head,
                                         n_head_kv, head_dim, kv->max_seq,
                                         pos0, n, pn_kvq4_sink(), pn_kvq4_win(),
                                         attn_sink(), attn_window(),
                                         qw_scale_inv_sqrt(head_dim)) < 0)
            return -1;
    } else if (fc_attn_flash_prefill_ctx(ctx, b->a_q, d->k_cache_g,
                                         d->v_cache_g, b->a_out, n_head,
                                         n_head_kv, head_dim, kv->max_seq,
                                         pos0, n, attn_sink(), attn_window(),
                                         qw_scale_inv_sqrt(head_dim)) < 0)
        return -1;
    if (!gated) {
        fc_sigmoid_ctx(ctx, b->a_z, n * attn_dim);
        fc_mul_ctx(ctx, b->a_out, b->a_z, b->a_out, n * attn_dim);
    }
    bprof_lap(BPROF_ATTN_FLASH, ctx, &bt0);
    if (spec) {
        bv_trace_rows(ctx, "verify", "rowsk", rows_rc, b->a_out, n, attn_dim,
                      NULL, pos0);
    }

    /* 6. Host KV mirror (session-resume source of truth): D2H the roped K/V
     * rows and append them to the host ring cache. This is batched prefill —
     * once per context, not per decode token. After this write the mirror is
     * current; decode tokens deferred by B2 will set kv_mirror_stale again.
     *
     * Phase 10b (lazy mirror): skip the host store entirely — the ring this
     * call just appended to is authoritative, downstream consumers (slot
     * bind, delta prefill) seed device-to-device, and anything that truly
     * needs host bytes materializes them via qw_kv_layer_materialize. Mark
     * the mirror stale so no one trusts it. Also skips the fc_sync_ctx:
     * step 7 is device work on the same stream.
     *
     * Phase 9.2: this also covers the speculative verify, which used to be
     * excluded (`&& !spec`) because the old verify path treated the host
     * mirror as authoritative. It no longer does — the ring is — and the
     * exclusion was actively wrong under the lazy mirror: it made a verify
     * D2H its DRAFT rows into the mirror and then clear kv_mirror_stale,
     * claiming a mirror whose [0, pos0) rows had never been written.
     */
    if (qw_kv_mirror_lazy()) {
        kv->kv_mirror_stale = 1;
        bprof_lap(BPROF_ATTN_XFER, ctx, &bt0);
        if (core_only)
            return 0;
        if (gpu_matvec_nx(&d->attn_output, b->a_out, b->block_out, n) < 0)
            return -1;
        fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
        bprof_lap(BPROF_ATTN_MV, ctx, &bt0);
        return 0;
    }
    if (core_only)
        return -1;   /* the eager host mirror has no TP form (tp_refusal) */
    fc_sync_ctx(ctx);
    float *hb_kv = g->bhost[3];
    fc_buffer_to_cpu_n(b->a_k, hb_kv, (size_t)n * kv_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_v, hb_kv + (size_t)n * kv_dim,
                       (size_t)n * kv_dim * sizeof(float));
    const int kv_sink = kv->q4 ? pn_kvq4_sink() : 0;
    const int kv_win = kv->q4 ? pn_kvq4_win() : 0;
    for (int t = 0; t < n; t++) {
        int cache_pos = (pos0 + t) % kv->max_seq;
        const float *kvec = hb_kv + (size_t)t * kv_dim;
        const float *vvec = hb_kv + (size_t)(n + t) * kv_dim;
        for (int hd = 0; hd < n_head_kv; hd++) {
            if (kv->q4) {
                pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd,
                                  kv->max_seq, cache_pos,
                                  kvec + (size_t)hd * head_dim, head_dim,
                                  kv_sink, kv_win);
                pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd,
                                  kv->max_seq, cache_pos,
                                  vvec + (size_t)hd * head_dim, head_dim,
                                  kv_sink, kv_win);
            } else {
                memcpy(kv->k + ((size_t)hd * kv->max_seq + cache_pos) * head_dim,
                       kvec + (size_t)hd * head_dim,
                       (size_t)head_dim * sizeof(float));
                memcpy(kv->v + ((size_t)hd * kv->max_seq + cache_pos) * head_dim,
                       vvec + (size_t)hd * head_dim,
                       (size_t)head_dim * sizeof(float));
            }
        }
    }
    /* B2 — host mirror is now current through pos0+n-1; clear the stale flag. */
    kv->kv_mirror_stale = 0;
    bprof_lap(BPROF_ATTN_XFER, ctx, &bt0);

    /* 7. Output projection + residual. */
    if (gpu_matvec_nx(&d->attn_output, b->a_out, b->block_out, n) < 0)
        return -1;
    fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
    bprof_lap(BPROF_ATTN_MV, ctx, &bt0);
    return 0;
}

/* Batched full-attention block, HOST-softmax fallback: batched
 * projections/norm/rope, then the host-side KV append + causal GQA softmax
 * per position (the same host code as gpu_attn_block, position by
 * position). Used when flash is disabled/unavailable or the window wraps
 * the ring. */
static int gpu_attn_block_batch(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                const pn_qwen35moe_config *cfg,
                                pn_qwen35moe_kv *kv,
                                int pos0, gpu_batch_scratch *b, int n)
{
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int n_kv_groups = n_head / n_head_kv;
    const int q_stride = head_dim * 2;
    const int attn_dim = n_head * head_dim;
    const int kv_dim = n_head_kv * head_dim;
    const float eps = cfg->rms_norm_eps;
    const int max_seq = kv->max_seq;
    float *k_cache = kv->k;      /* fp32 mode; NULL under q4 */
    float *v_cache = kv->v;
    const int kv_q4 = kv->q4;
    const int kv_sink = kv_q4 ? pn_kvq4_sink() : 0;
    const int kv_win = kv_q4 ? pn_kvq4_win() : 0;
    fc_cuda_ctx *ctx = d->ctx;
    /* Phase 10b (lazy mirror): this host-softmax prefill fallback reads the
     * host KV mirror [0, pos0+t). If earlier chunks went through the flash
     * path with the lazy mirror, pull the rows down from the GPU ring
     * first; this path keeps the mirror current itself from here on. */
    if (kv->kv_mirror_stale &&
        qw_kv_layer_materialize(d, kv,
                                (pos0 < max_seq) ? pos0 : max_seq) != 0) {
        fprintf(stderr, "[attn] host batch attn: mirror stale with no ring "
                "source (pos0=%d) — failing chunk\n", pos0);
        return -1;
    }
    float *hb_qg = g->bhost[0];
    float *hb_q = g->bhost[1];
    float *hb_z = g->bhost[2];
    float *hb_kv = g->bhost[3];
    double bt0 = prof_on() ? prof_now() : 0.0;

    /* 1. Joint Q+gate, K, V projections for all positions. */
    if (gpu_matvec_nx(&d->attn_q, b->x_norm, b->a_qg, n) < 0 ||
        gpu_matvec_nx(&d->attn_k, b->x_norm, b->a_k, n) < 0 ||
        gpu_matvec_nx(&d->attn_v, b->x_norm, b->a_v, n) < 0)
        return -1;
    bprof_lap(BPROF_ATTN_MV, ctx, &bt0);

    /* Q/gate split, host-side per position (same glue as the single-token
     * path; zgate persists in hb_z for the output gate below). */
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->a_qg, hb_qg,
                       (size_t)n * 2 * attn_dim * sizeof(float));
    for (int t = 0; t < n; t++) {
        const float *src_t = hb_qg + (size_t)t * 2 * attn_dim;
        float *q_t = hb_q + (size_t)t * attn_dim;
        float *z_t = hb_z + (size_t)t * attn_dim;
        for (int hd = 0; hd < n_head; hd++) {
            memcpy(q_t + (size_t)hd * head_dim,
                   src_t + (size_t)hd * q_stride,
                   (size_t)head_dim * sizeof(float));
            memcpy(z_t + (size_t)hd * head_dim,
                   src_t + (size_t)hd * q_stride + head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    fc_buffer_from_cpu_n(b->a_q, hb_q, (size_t)n * attn_dim * sizeof(float));
    bprof_lap(BPROF_ATTN_SPLIT, ctx, &bt0);

    /* 2.-4. QK-RMSNorm (heads uniform across positions -> single-token
     * kernel with n x the head count) + batched partial NEOX RoPE. */
    fc_qk_rms_norm_ctx(ctx, b->a_q, d->attn_q_norm.buf, n * n_head, head_dim,
                       eps);
    fc_qk_rms_norm_ctx(ctx, b->a_k, d->attn_k_norm.buf, n * n_head_kv,
                       head_dim, eps);
    fc_rope_neox_partial_nx_ctx(ctx, b->a_q, n_head, head_dim, cfg->rope_dim,
                                pos0, cfg->rope_theta, n);
    fc_rope_neox_partial_nx_ctx(ctx, b->a_k, n_head_kv, head_dim,
                                cfg->rope_dim, pos0, cfg->rope_theta, n);

    /* 5.+6. Host: append every position's K/V to the cache, then the causal
     * GQA softmax + sigmoid output gate per position (identical code to
     * gpu_attn_block, looped). aout accumulates in hb_qg (done with it). */
    bprof_lap(BPROF_ATTN_ROPE, ctx, &bt0);
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->a_q, hb_q, (size_t)n * attn_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_k, hb_kv, (size_t)n * kv_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_v, hb_kv + (size_t)n * kv_dim,
                       (size_t)n * kv_dim * sizeof(float));
    bprof_lap(BPROF_ATTN_XFER, ctx, &bt0);
    {
        float *aout_all = hb_qg;
        float scale = qw_scale_inv_sqrt(head_dim);
        /* heap scratch: max_seq can reach the 256K cap (too large for stack). */
        float *scores = (float *)malloc((size_t)max_seq * sizeof(float));
        float krow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        float vrow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        if (!scores) return -1;
        for (int t = 0; t < n; t++) {
            int pos = pos0 + t;
            int cache_pos = pos % max_seq;
            const float *kvec = hb_kv + (size_t)t * kv_dim;
            const float *vvec = hb_kv + (size_t)(n + t) * kv_dim;
            for (int hd = 0; hd < n_head_kv; hd++) {
                if (kv_q4) {
                    pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd,
                                      max_seq, cache_pos,
                                      kvec + (size_t)hd * head_dim, head_dim,
                                      kv_sink, kv_win);
                    pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd,
                                      max_seq, cache_pos,
                                      vvec + (size_t)hd * head_dim, head_dim,
                                      kv_sink, kv_win);
                } else {
                    memcpy(k_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                           kvec + (size_t)hd * head_dim,
                           (size_t)head_dim * sizeof(float));
                    memcpy(v_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                           vvec + (size_t)hd * head_dim,
                           (size_t)head_dim * sizeof(float));
                }
            }

            int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
            const float *qhost = hb_q + (size_t)t * attn_dim;
            float *aout = aout_all + (size_t)t * attn_dim;
            for (int qh = 0; qh < n_head; qh++) {
                int kv_h = qh / n_kv_groups;
                const float *q_vec = qhost + (size_t)qh * head_dim;
                const float *k_base = kv_q4 ? NULL :
                    k_cache + (size_t)kv_h * max_seq * head_dim;
                const float *v_base = kv_q4 ? NULL :
                    v_cache + (size_t)kv_h * max_seq * head_dim;
                float *out_vec = aout + (size_t)qh * head_dim;
                /* Sliding window (QWEN35_ATTN_WINDOW): attend key sp iff
                 * sp < aw_sink OR sp > pos - aw_win. Masked keys get score
                 * -inf (exp -> 0), so the softmax is over the windowed set.
                 * aw_win <= 0 attends all (full-attention baseline). */
                const int aw_win = attn_window();
                const int aw_sink = attn_sink();
                for (int sp = 0; sp < seq_len; sp++) {
                    if (aw_win > 0 && sp >= aw_sink && sp <= pos - aw_win) {
                        scores[sp] = -INFINITY;
                        continue;
                    }
                    const float *k_vec;
                    if (kv_q4) {
                        pn_kvq4_load_row(kv->kq, kv->k_sink, kv->k_win, kv_h,
                                         max_seq, sp, pos, krow, head_dim,
                                         kv_sink, kv_win);
                        k_vec = krow;
                    } else {
                        k_vec = k_base + (size_t)sp * head_dim;
                    }
                    scores[sp] = pn_dot_f32(q_vec, k_vec, head_dim) * scale;
                }
                float mx = -INFINITY;
                for (int sp = 0; sp < seq_len; sp++)
                    if (scores[sp] > mx) mx = scores[sp];
                float sum_exp = 0.0f;
                for (int sp = 0; sp < seq_len; sp++) {
                    scores[sp] = expf(scores[sp] - mx);
                    sum_exp += scores[sp];
                }
                float inv_sum = 1.0f / sum_exp;
                memset(out_vec, 0, (size_t)head_dim * sizeof(float));
                for (int sp = 0; sp < seq_len; sp++) {
                    /* Masked keys contributed exp()=0 above; skip their V. */
                    if (aw_win > 0 && sp >= aw_sink && sp <= pos - aw_win)
                        continue;
                    float w_s = scores[sp] * inv_sum;
                    const float *v_vec;
                    if (kv_q4) {
                        pn_kvq4_load_row(kv->vq, kv->v_sink, kv->v_win, kv_h,
                                         max_seq, sp, pos, vrow, head_dim,
                                         kv_sink, kv_win);
                        v_vec = vrow;
                    } else {
                        v_vec = v_base + (size_t)sp * head_dim;
                    }
                    for (int dd = 0; dd < head_dim; dd++)
                        out_vec[dd] += w_s * v_vec[dd];
                }
            }

            const float *z_t = hb_z + (size_t)t * attn_dim;
            for (int i = 0; i < attn_dim; i++) {
                float gv = 1.0f / (1.0f + expf(-z_t[i]));
                aout[i] *= gv;
            }
        }
        free(scores);
        bprof_lap(BPROF_ATTN_HOST, ctx, &bt0);
        fc_buffer_from_cpu_n(b->a_out, aout_all,
                             (size_t)n * attn_dim * sizeof(float));
        bprof_lap(BPROF_ATTN_XFER, ctx, &bt0);
    }

    /* 7. Output projection + residual. */
    if (gpu_matvec_nx(&d->attn_output, b->a_out, b->block_out, n) < 0)
        return -1;
    fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
    bprof_lap(BPROF_ATTN_MV, ctx, &bt0);
    return 0;
}

/* One layer of the batched forward: attn_norm -> (SSM scan | attn) ->
 * +residual -> post_attention_norm -> shared-MoE -> +residual, all over the
 * [n][n_embd] residual stream in b->hidden. */
static int gpu_layer_forward_batch(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                                   fc_cuda_ctx *ctx, int layer,
                                   gpu_batch_scratch *b,
                                   pn_qwen35moe_state *s, int pos0, int n,
                                   int spec)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    gpu_moe *moe;
    int rc;

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx];
        pn_qwen35moe_kv *kv = &s->kv[idx];
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c,
                           cfg->rms_norm_eps, n);
        rc = gpu_attn_block_batch_flash(g, d, cfg, kv, pos0, b, n, spec, 0);
        if (rc > 0) { /* flash unavailable for this call — host softmax */
            /* ...except under a speculative verify. The host fallback appends
             * into the HOST q4 window ring, which has the same one-lap-back
             * eviction problem (see qw_spec_win_snapshot) and no undo buffer,
             * so a rejected draft would silently corrupt a live shadow slot.
             * Fail the verify instead; every caller falls back to a plain
             * forward, which is correct and only marginally slower. */
            if (spec) {
                fprintf(stderr, "[attn] spec verify: flash unavailable, "
                        "refusing the host-softmax fallback\n");
                return -1;
            }
            rc = gpu_attn_block_batch(g, d, cfg, kv, pos0, b, n);
        }
        if (rc < 0)
            return rc;
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf,
                           b->x_norm, c, cfg->rms_norm_eps, n);
        moe = &d->moe;
    } else {
        gpu_ssm_layer *d = &g->ssm[idx];
        pn_qwen35moe_ssm_state *st = &s->ssm[idx];

        /* GPU-resident dual state, seeded exactly like the single-token
         * path (fresh buffers, a state swap, or a restart at pos 0). */
        size_t conv_n = (size_t)st->conv_minus_1 * st->conv_dim;
        size_t ssm_n = (size_t)st->state_size * st->state_size * st->n_v_heads;
        if (!d->conv_state_g)
            d->conv_state_g = fc_buffer_alloc_ctx(ctx, conv_n * sizeof(float));
        if (!d->ssm_state_g)
            d->ssm_state_g = fc_buffer_alloc_ctx(ctx, ssm_n * sizeof(float));
        if (!d->conv_state_g || !d->ssm_state_g)
            return -1;
        if (d->host_state != (const void *)st || pos0 == 0)
            qw_ssm_seed_shared(g, d, st, pos0);

        /* Spec verify: per-position snapshot buffers, grown to n. */
        if (spec && d->snaps_cap < n) {
            fc_buffer_free(d->conv_snaps);
            fc_buffer_free(d->ssm_snaps);
            d->conv_snaps = fc_buffer_alloc_ctx(ctx, (size_t)n * conv_n *
                                                     sizeof(float));
            d->ssm_snaps = fc_buffer_alloc_ctx(ctx, (size_t)n * ssm_n *
                                                    sizeof(float));
            d->snaps_cap = (d->conv_snaps && d->ssm_snaps) ? n : 0;
            if (!d->snaps_cap)
                return -1;
        }

        fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c,
                           cfg->rms_norm_eps, n);
        double st0 = prof_on() ? prof_now() : 0.0;
        rc = gpu_ssm_block_batch(d, cfg, b, n, spec, 0);
        if (rc < 0)
            return rc;
        bprof_lap(BPROF_SSM, ctx, &st0);
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf,
                           b->x_norm, c, cfg->rms_norm_eps, n);
        moe = &d->moe;
    }

    /* allow_dedup = !spec: this function serves both the PREFILL chunk loop
     * (n = QWEN35_PREFILL_CHUNK positions, wide enough for the expert union
     * to pay) and the speculative VERIFY (n = draft length, a handful of
     * positions — decode-shaped economics, so it stays on _nx). */
    if (spec)
        bv_trace_rows(ctx, "verify", "mix", layer, b->hidden, n, c, NULL, pos0);
    double mt0 = prof_on() ? prof_now() : 0.0;
    rc = gpu_moe_block_batch(ctx, moe, cfg, b, n, !spec);
    bprof_lap(BPROF_MOE, ctx, &mt0);
    if (spec)
        bv_trace_rows(ctx, "verify", "out", layer, b->hidden, n, c, NULL, pos0);
    return rc;
}

/* ============================================================
 * QWEN35_TP — the batched layer body and its exchanges (spec k80_q38_tp2 §3.4)
 *
 * Both ranks of a stage hold the identical residual stream; per layer they
 * exchange three times over the board's peer link:
 *   E1  core gather: each rank's attention / GDN core output (its heads) into a
 *       full-width row in the original column order, on both ranks;
 *   E2  out gather: each rank's out-projection rows into the full n_embd row;
 *   E3  down reduce: each rank's parity partial V_p + V_(p+2), summed A + B (the
 *       Q8_0 kernel's own final add; IEEE add commutes, so both ranks agree).
 * Every copy is PULLED: rank p records tp_ev[die][k] once its source is
 * written, and its partner waits on it and queues the copy on its OWN stream
 * (a pushed cross-die copy was seen incomplete after the source stream synced).
 * Destinations are written only by their own die. A source is rewritten only
 * after its writer has waited on the partner's NEXT exchange event, which the
 * partner records after its pull in stream order: E1's source (a_out / s_gated)
 * is next written in the following layer's core, after E2 and E3; E2's
 * (tp_half) at the next out projection, after E3 and E1; E3's (block_out) at
 * the next down, after E1 and E2. So one buffer set per die suffices.
 *
 * A MoE layer (spec k80_q36_tp2 §3.4-3.5) replaces E3 with two gathers:
 *   E3  hid gather: each rank's expert + shared gate/up rows (tp_hidl) into the
 *       full-ff hidden (tp_hidf) both downs read;
 *   E4  output gather: each rank's routed + shared down rows (tp_mol) into the
 *       full rows (tp_mof) the replicated combine reads.
 * Same rule: tp_hidl is next written at the next layer's gate/up (after E4, E1,
 * E2), tp_mol at the next down (after E1, E2, E3), and the core / tp_half
 * sources now also wait on E4 first.
 * ============================================================ */

/* QWEN35_TP_XCHG (default on; =0 keeps the event handshake + copies) for the
 * exchanges of at most TP_XCHG_MAX_BYTES a rank: decode and verify (n <= 5 is
 * <= 45 KiB); a 128-row prefill chunk moves ~1 MiB (copy engine, 8k wide prefill
 * 41.9 s vs 43.1 s through the kernel). */
#define TP_XCHG_MAX_BYTES ((size_t)256 << 10)
static int tp_xchg_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_TP_XCHG");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

/* QWEN35_BS_PD_OVERLAP (default on; =0 turns it off): the MoE TP decode-slots
 * step runs on the decode lane (QW_LANE_DECODE), the prerequisite for issuing
 * it between the layers of a prefill chunk. Exact: every gate bit-identical vs
 * off (#180); our K80 host has served it since 2026-11-21. Read once. */
static int qw_pd_overlap_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_BS_PD_OVERLAP");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

int pn_qwen35moe_pd_overlap_enabled(void)
{
    return qw_pd_overlap_on();
}

/* Exchange buffers sized for n rows on every die (and the events, once). */
static int tp_exchange_ready(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg, int n)
{
    if (TPX(g)->cap >= n)
        return 0;
    int cap = n < 8 ? 8 : n;
    size_t core = (size_t)cfg->n_head * cfg->head_dim;
    g->tp_gen++;   /* every exchange buffer moves */
    if (core < (size_t)cfg->ssm_state * cfg->ssm_dt_rank)
        core = (size_t)cfg->ssm_state * cfg->ssm_dt_rank;
    TPX(g)->cap = cap;
    for (int i = 0; i < 4; i++) {
        fc_buffer_free(TPX(g)->core[i]);
        fc_buffer_free(TPX(g)->half[i]);
        fc_buffer_free(TPX(g)->blk[i]);
        fc_buffer_free(TPX(g)->recv[i]);
        TPX(g)->core[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)cap * core * sizeof(float));
        TPX(g)->half[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)cap * cfg->n_embd / 2 *
                                                           sizeof(float));
        TPX(g)->blk[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)cap * cfg->n_embd * sizeof(float));
        TPX(g)->recv[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)cap * cfg->n_embd * sizeof(float));
        int ok = TPX(g)->core[i] && TPX(g)->half[i] && TPX(g)->blk[i] && TPX(g)->recv[i];
        if (cfg->n_expert) {
            /* k expert rows then one shared row of ff per token. */
            const size_t hrows = (size_t)cap * (cfg->n_expert_used + 1);
            fc_buffer_free(TPX(g)->hidl[i]);
            fc_buffer_free(TPX(g)->hidf[i]);
            fc_buffer_free(TPX(g)->mol[i]);
            fc_buffer_free(TPX(g)->mof[i]);
            TPX(g)->hidl[i] = fc_buffer_alloc_ctx(g->ctx[i], hrows * cfg->expert_ff / 2 *
                                                               sizeof(float));
            TPX(g)->hidf[i] = fc_buffer_alloc_ctx(g->ctx[i], hrows * cfg->expert_ff * sizeof(float));
            TPX(g)->mol[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)cap * cfg->n_embd * sizeof(float));
            TPX(g)->mof[i] = fc_buffer_alloc_ctx(g->ctx[i], (size_t)2 * cap * cfg->n_embd *
                                                              sizeof(float));
            ok = ok && TPX(g)->hidl[i] && TPX(g)->hidf[i] && TPX(g)->mol[i] && TPX(g)->mof[i];
        }
        for (int k = 0; k < TP_N_XCHG; k++) {
            if (!TPX(g)->ev[i][k])
                TPX(g)->ev[i][k] = fc_event_create_ctx(g->ctx[i]);
            ok = ok && TPX(g)->ev[i][k];
        }
        if (!g->tp_gev[i])
            g->tp_gev[i] = fc_event_create_ctx(g->ctx[i]);
        ok = ok && g->tp_gev[i];
        if (!ok)
            TPX(g)->cap = 0;
    }
    if (TPX(g)->cap && !TPX(g)->sync[0] && tp_xchg_on()) {
        /* Counters are allocated once, zeroed on every die together, and never
         * reset while a step can be in flight (a reset races a partner's wait). */
        static const uint8_t zero[TP_N_XCHG * FC_XCHG_SYNC_BYTES];
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            TPX(g)->sync[i] = fc_buffer_from_ptr_ctx(g->ctx[i], zero, sizeof(zero));
            ok = ok && TPX(g)->sync[i];
        }
        TPX(g)->xchg = ok;
        g->tp_gen++;
    }
    return TPX(g)->cap ? 0 : -1;
}

/* Stage s's ranks each mark exchange k's source written, then wait for the
 * partner's. */
static int tp_handshake(pn_qwen35moe_gpu *g, int s, int k)
{
    for (int p = 0; p < 2; p++)
        if (fc_event_record_ctx(g->ctx[2 * s + p], TPX(g)->ev[2 * s + p][k]) != 0)
            return -1;
    for (int p = 0; p < 2; p++)
        if (fc_event_wait_ctx(g->ctx[2 * s + p], TPX(g)->ev[2 * s + 1 - p][k]) != 0)
            return -1;
    return 0;
}

/* Gather: rank p's source holds, per row, `chunks` runs of `width` floats; run
 * r of rank p belongs at column r * (row / chunks) + p * width of the full
 * `row`-float rows. Each die copies its own runs and pulls its partner's into
 * dst[die]. */
static int tp_gather(pn_qwen35moe_gpu *g, int s, int k, fc_buffer *const src[2],
                     fc_buffer *const *dst, int width, int chunks, int row, int n)
{
    const size_t wb = (size_t)width * sizeof(float);
    const size_t pitch = (size_t)(row / chunks) * sizeof(float);
    const size_t height = (size_t)n * chunks;

    /* A graph capture cannot hold the peer copy, and a captured cross-die memcpy node
     * is issued from the launching die (a push); the copy kernel stays a pull. */
    int (*copy)(fc_buffer *, size_t, size_t, fc_buffer *, size_t, size_t, size_t, size_t,
                fc_cuda_ctx *) =
        g->tp_capturing ? fc_buffer_copy_2d_kernel_async : fc_buffer_copy_2d_async;

    if (TPX(g)->xchg && height * wb <= TP_XCHG_MAX_BYTES) {
        /* One launch a rank: own run, wait for the partner's launch, pull. Larger
         * (prefill-chunk) exchanges keep the copy engine, which moves them faster;
         * both schemes order a source's rewrite after the next exchange's wait, so
         * mixing them per call keeps the hazard rule. */
        for (int p = 0; p < 2; p++)
            if (fc_xchg_2d_async(dst[p], p * wb, (1 - p) * wb, pitch, src[p], src[1 - p], wb,
                                 wb, height, TPX(g)->sync[2 * s + p], TPX(g)->sync[2 * s + 1 - p],
                                 k, g->ctx[2 * s + p]) != 0)
                return -1;
        return 0;
    }
    if (tp_handshake(g, s, k) < 0)
        return -1;
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        if (copy(dst[p], p * wb, pitch, src[p], 0, wb, wb, height, ctx) != 0 ||
            copy(dst[p], (1 - p) * wb, pitch, src[1 - p], 0, wb, wb, height, ctx) != 0)
            return -1;
    }
    return 0;
}

/* Down reduce: recv[die] = own partial + the partner's (pulled), [n][c]. */
static int tp_reduce(pn_qwen35moe_gpu *g, int s, fc_buffer *const part[2],
                     fc_buffer *const *recv, int c, int n)
{
    const size_t bytes = (size_t)n * c * sizeof(float);

    if (tp_handshake(g, s, 2) < 0)
        return -1;
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        if (fc_buffer_copy_2d_async(recv[p], 0, bytes, part[1 - p], 0, bytes, bytes, 1, ctx) != 0)
            return -1;
        fc_add_ctx(ctx, part[p], recv[p], recv[p], n * c);
    }
    return 0;
}

/* QWEN35_TP_SYNC=1: sync both ranks after every exchange (debug arm: the hashes
 * must not depend on the event ordering). */
static int tp_sync_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_TP_SYNC");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* QWEN35_TP_GRAPH (default on; =0 issues directly): a MoE TP decode-slots step
 * or verify replays each stage's device work as one captured graph over both ranks. */
static int tp_graph_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_TP_GRAPH");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

static void tp_sync_stage(pn_qwen35moe_gpu *g, int s)
{
    if (tp_sync_on()) {
        fc_sync_ctx(g->ctx[2 * s]);
        fc_sync_ctx(g->ctx[2 * s + 1]);
    }
}

/* The MoE FFN of gpu_moe_block_batch on both ranks of stage s, n rows of
 * x_norm (replicated): router + top-k whole on each rank; the rank's expert and
 * shared gate/up rows; E3; the rank's down output rows over the whole hidden;
 * E4; the combine + residual whole on each rank (the same unfused launches
 * gpu_moe_block_batch issues at n > 1). */
static int tp_moe_block(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg, int s,
                        gpu_moe *const moe[2], gpu_batch_scratch *const b[2], int n)
{
    const int c = cfg->n_embd, k = cfg->n_expert_used, ff = cfg->expert_ff, fh = ff / 2;
    fc_buffer *hsh[2] = { NULL, NULL }, *fsh[2] = { NULL, NULL };
    fc_buffer *msh[2] = { NULL, NULL }, *osh[2] = { NULL, NULL };
    int rc = -1;

    /* Views of the shared-expert rows that follow the n*k expert rows. */
    for (int p = 0; p < 2; p++) {
        const int die = 2 * s + p;
        hsh[p] = fc_buffer_offset_view(TPX(g)->hidl[die], (size_t)n * k * fh * sizeof(float));
        fsh[p] = fc_buffer_offset_view(TPX(g)->hidf[die], (size_t)n * k * ff * sizeof(float));
        msh[p] = fc_buffer_offset_view(TPX(g)->mol[die], (size_t)n * (c / 2) * sizeof(float));
        osh[p] = fc_buffer_offset_view(TPX(g)->mof[die], (size_t)n * c * sizeof(float));
        if (!hsh[p] || !fsh[p] || !msh[p] || !osh[p])
            goto out;
    }
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        gpu_moe *d = moe[p];
        if (gpu_matvec_nx(&d->ffn_gate_inp, b[p]->x_norm, b[p]->m_logits, n) < 0)
            goto out;
        fc_moe_router_topk_nx_ctx(ctx, b[p]->m_logits, b[p]->m_tidx, b[p]->m_twt,
                                  cfg->n_expert, k, cfg->expert_weights_scale, n);
        if (gpu_moe_gateup_batch(ctx, cfg, d, b[p], TPX(g)->hidl[2 * s + p], fh, n, 1) != 0)
            goto out;
        if (gpu_matvec_nx(&d->ffn_gate_shexp, b[p]->x_norm, b[p]->m_gate, n) < 0 ||
            gpu_matvec_nx(&d->ffn_up_shexp, b[p]->x_norm, b[p]->m_up, n) < 0)
            goto out;
        fc_moe_silu_mul_ctx(ctx, b[p]->m_gate, b[p]->m_up, hsh[p], n * fh);
    }

    /* E3: the rank halves of every hidden row into the full-ff rows. */
    if (tp_gather(g, s, 2, TPX(g)->hidl + 2 * s, &TPX(g)->hidf[2 * s], fh, 1, ff, n * (k + 1)) < 0)
        goto out;
    tp_sync_stage(g, s);
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        gpu_moe *d = moe[p];
        fc_moe_down_accum_nx_ctx(ctx, (int)d->down_all.type, d->down_all.bytes,
                                 TPX(g)->hidf[2 * s + p], b[p]->m_tidx, b[p]->m_twt,
                                 TPX(g)->mol[2 * s + p], k, c / 2, ff, d->down_all.row_stride, n);
        if (gpu_matvec_nx(&d->ffn_down_shexp, fsh[p], msh[p], n) < 0)
            goto out;
    }

    /* E4: routed then shared down rows into the full rows; combine + residual. */
    if (tp_gather(g, s, 3, TPX(g)->mol + 2 * s, &TPX(g)->mof[2 * s], c / 2, 1, c, 2 * n) < 0)
        goto out;
    tp_sync_stage(g, s);
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        gpu_moe *d = moe[p];
        fc_buffer *mo = TPX(g)->mof[2 * s + p];
        if (!(gpu_fuse_mask() & 64) ||
            fc_moe_shexp_gate_combine_nx_ctx(ctx, d->ffn_gate_inp_shexp.buf, b[p]->x_norm,
                                             osh[p], mo, b[p]->block_out, b[p]->hidden, c,
                                             n) != 0) {
            fc_moe_shexp_dot_gate_nx_ctx(ctx, d->ffn_gate_inp_shexp.buf, b[p]->x_norm, osh[p],
                                         c, n);
            fc_add_ctx(ctx, mo, osh[p], b[p]->block_out, n * c);
            fc_add_ctx(ctx, b[p]->hidden, b[p]->block_out, b[p]->hidden, n * c);
        }
    }
    rc = 0;
out:
    for (int p = 0; p < 2; p++) {
        fc_buffer_free_view(hsh[p]);
        fc_buffer_free_view(fsh[p]);
        fc_buffer_free_view(msh[p]);
        fc_buffer_free_view(osh[p]);
    }
    return rc;
}

/* SSM state buffers of a TP rank's layer half: the local conv / recurrent sizes. */
static size_t tp_ssm_conv_n(const pn_qwen35moe_gpu *g, const pn_qwen35moe_ssm_state *st)
{
    return (size_t)st->conv_minus_1 * g->tp_cfg.conv_dim;
}

static size_t tp_ssm_rec_n(const pn_qwen35moe_gpu *g, const pn_qwen35moe_ssm_state *st)
{
    return (size_t)st->state_size * st->state_size * g->tp_cfg.ssm_dt_rank;
}

/* The rest of a TP layer once both ranks' cores (core[p], the rank's heads) are
 * written: E1, out-projection rows, E2, residual, post norm, then the FFN (dense:
 * parity down + E3 reduce; MoE: tp_moe_block). Shared by the batch and slots
 * bodies. */
static int tp_layer_tail(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg, int s,
                         int is_attn, int idx, fc_buffer *const core[2],
                         gpu_batch_scratch *const b[2], int n)
{
    const pn_qwen35moe_config *lc = &g->tp_cfg;
    const int c = cfg->n_embd;
    const gpu_weight *out_w[2];
    gpu_moe *moe[2];
    fc_buffer *half[2], *part[2];

    for (int p = 0; p < 2; p++) {
        if (is_attn) {
            gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
            out_w[p] = &d->attn_output;
            moe[p] = &d->moe;
        } else {
            gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
            out_w[p] = &d->ssm_out;
            moe[p] = &d->moe;
        }
        half[p] = TPX(g)->half[2 * s + p];
        part[p] = b[p]->block_out;
    }

    /* E1: the core (the rank's heads) into the full row, original order. GDN
     * rank heads come in groups of n_k/2 heads per v-group: n_v/n_k runs a row. */
    int rc = is_attn
        ? tp_gather(g, s, 0, core, &TPX(g)->core[2 * s], lc->n_head * cfg->head_dim, 1,
                    cfg->n_head * cfg->head_dim, n)
        : tp_gather(g, s, 0, core, &TPX(g)->core[2 * s], lc->ssm_n_group * cfg->ssm_state,
                    cfg->ssm_dt_rank / cfg->ssm_n_group, cfg->ssm_dt_rank * cfg->ssm_state, n);
    if (rc < 0)
        return -1;
    tp_sync_stage(g, s);
    for (int p = 0; p < 2; p++)
        if (gpu_matvec_nx(out_w[p], TPX(g)->core[2 * s + p], half[p], n) < 0)
            return -1;

    /* E2: out-projection rows into the full hidden-width row; residual. */
    if (tp_gather(g, s, 1, half, &TPX(g)->blk[2 * s], c / 2, 1, c, n) < 0)
        return -1;
    tp_sync_stage(g, s);
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        const gpu_vec *post = is_attn ? &g->attn[idx + p * g->tp_n_attn].post_attention_norm
                                      : &g->ssm[idx + p * g->tp_n_ssm].post_attention_norm;
        fc_add_ctx(ctx, b[p]->hidden, TPX(g)->blk[2 * s + p], b[p]->hidden, n * c);
        fc_rms_norm_nx_ctx(ctx, b[p]->hidden, post->buf, b[p]->x_norm, c, cfg->rms_norm_eps, n);
        if (cfg->n_expert)
            continue;
        /* The dense FFN of gpu_dense_ffn_batch on the rank's rows k % 2 == p; down
         * is the parity partial over exactly that SwiGLU output. */
        if (gpu_matvec_nx(&moe[p]->ffn_gate_shexp, b[p]->x_norm, b[p]->m_gate, n) < 0 ||
            gpu_matvec_nx(&moe[p]->ffn_up_shexp, b[p]->x_norm, b[p]->m_up, n) < 0)
            return -1;
        fc_moe_silu_mul_ctx(ctx, b[p]->m_gate, b[p]->m_up, b[p]->m_gate, n * lc->ffn_length);
        if (gpu_matvec_nx(&moe[p]->ffn_down_shexp, b[p]->m_gate, part[p], n) < 0)
            return -1;
    }

    if (cfg->n_expert)
        return tp_moe_block(g, cfg, s, moe, b, n);

    /* E3: down = A + B on both ranks; residual. */
    if (tp_reduce(g, s, part, &TPX(g)->recv[2 * s], c, n) < 0)
        return -1;
    tp_sync_stage(g, s);
    for (int p = 0; p < 2; p++)
        fc_add_ctx(g->ctx[2 * s + p], b[p]->hidden, TPX(g)->recv[2 * s + p], b[p]->hidden, n * c);
    return 0;
}

/* QWEN35_TP: sync BOTH ranks' GPU KV rings of attention layer struct idx to pos0
 * before either rank runs attention. The two halves read one shared host kv, and
 * a rank's flash append / verify prep marks that mirror stale (lazy mode) right
 * after syncing its own ring. After a host restore (sys-prefix snapshot, prefix-
 * cache turn, warm tier) content_gen is new, so no ring is a D2D source: run one
 * rank at a time and the second refuses the now-"stale" (still correct) mirror.
 * The later per-rank syncs are no-ops (watermark == pos0, same generation). */
static int tp_attn_kv_cache_sync_both(pn_qwen35moe_gpu *g, int idx, const pn_qwen35moe_kv *kv,
                                      int pos0)
{
#ifdef QW_TP_SYNC_RED
    return 0;   /* red build: the old per-rank order (tests/test_qwen35_tp_restore_sync) */
#endif
    if (pos0 == 0)
        return 0;   /* a restart sources nothing; each rank's own sync re-seeds */
    for (int p = 0; p < 2; p++)
        if (gpu_attn_kv_cache_sync(&g->attn[idx + p * g->tp_n_attn], kv, pos0) < 0)
            return -1;
    return 0;
}

/* One layer of stage s on both ranks, batched over n rows (prefill: spec 0;
 * verify: spec 1). b[p] is rank p's batch scratch (die 2s + p). */
static int tp_layer_forward_batch(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s, int layer,
                                  gpu_batch_scratch *const b[2], pn_qwen35moe_state *st,
                                  int pos0, int n, int spec)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const pn_qwen35moe_config *lc = &g->tp_cfg;
    const int c = cfg->n_embd;
    const int idx = pn_qwen35moe_model_layer_index(m, layer);
    const int is_attn = pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN;
    fc_buffer *core[2];

    if (is_attn && tp_attn_kv_cache_sync_both(g, idx, &st->kv[idx], pos0) < 0)
        return -1;
    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        if (is_attn) {
            gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
            fc_rms_norm_nx_ctx(ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, n);
            if (gpu_attn_block_batch_flash(g, d, lc, &st->kv[idx], pos0, b[p], n, spec, 1) != 0)
                return -1;
            core[p] = b[p]->a_out;
        } else {
            gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
            pn_qwen35moe_ssm_state *sst = &st->ssm[idx];
            size_t conv_n = tp_ssm_conv_n(g, sst), ssm_n = tp_ssm_rec_n(g, sst);
            if (!d->conv_state_g)
                d->conv_state_g = fc_buffer_alloc_ctx(ctx, conv_n * sizeof(float));
            if (!d->ssm_state_g)
                d->ssm_state_g = fc_buffer_alloc_ctx(ctx, ssm_n * sizeof(float));
            if (!d->conv_state_g || !d->ssm_state_g)
                return -1;
            if (d->host_state != (const void *)sst || pos0 == 0)
                qw_ssm_seed_shared(g, d, sst, pos0);
            if (spec && d->snaps_cap < n) {
                fc_buffer_free(d->conv_snaps);
                fc_buffer_free(d->ssm_snaps);
                d->conv_snaps = fc_buffer_alloc_ctx(ctx, (size_t)n * conv_n * sizeof(float));
                d->ssm_snaps = fc_buffer_alloc_ctx(ctx, (size_t)n * ssm_n * sizeof(float));
                d->snaps_cap = (d->conv_snaps && d->ssm_snaps) ? n : 0;
                if (!d->snaps_cap)
                    return -1;
            }
            fc_rms_norm_nx_ctx(ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, n);
            if (gpu_ssm_block_batch(d, lc, b[p], n, spec, 1) < 0)
                return -1;
            core[p] = b[p]->s_gated;
        }
    }
    return tp_layer_tail(g, cfg, s, is_attn, idx, core, b, n);
}

static int tp_verify_stage(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s,
                           gpu_batch_scratch *const b[2], pn_qwen35moe_state *st, int pos0,
                           int n);

/* The TP walk of gpu_batch_tokens_core: stage 0 embeds on both ranks, each stage
 * runs its layers on its pair, the hop to stage 1 is staged through the host
 * (never a cross-board peer copy), and the head runs on die 3 (head_ctx). On
 * return *ctx_out / *b_out / *sc_out are the head die's context and scratch. */
/* ---- QWEN35_PCIE_*: the TP step's per-step copies through qwen35moe_pcie_stage ---- */

/* Rows the input / hop staging holds: the widest decode-slots step or verify. */
#define QW_PCIE_MAX_ROWS \
    (PN_QWEN35MOE_GPU_MAX_SLOTS > FC_ATTN_ROWS_MAX ? PN_QWEN35MOE_GPU_MAX_SLOTS : FC_ATTN_ROWS_MAX)

/* The TP step's staging when a QWEN35_PCIE_* lever is on and `bytes` fit it (created on
 * first use), else NULL: the legacy blocking copies through bhost[0]. */
static qw_pcie_stage *qw_pcie_for(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg,
                                  size_t bytes)
{
    const qw_pcie_flags f = qw_pcie_flags_get();
    if (!g->tp || !qw_pcie_flags_any(f) || g->pcie_failed)
        return NULL;
    if (!g->pcie) {
        g->pcie = qw_pcie_stage_create(g->ctx,
                                       (size_t)QW_PCIE_MAX_ROWS * cfg->n_embd * sizeof(float),
                                       (size_t)2 * PN_QWEN35MOE_GPU_MAX_SLOTS * sizeof(int32_t),
                                       f);
        g->pcie_failed = g->pcie == NULL;
    }
    return g->pcie && bytes <= qw_pcie_stage_rows_cap(g->pcie) ? g->pcie : NULL;
}

/* Host rows for stage 0's embedded input: the staging's (once its earlier uploads have
 * read them) or bhost[0]. NULL on a device error. */
static float *qw_stage_input_rows(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg,
                                  size_t bytes)
{
    qw_pcie_stage *ps = qw_pcie_for(g, cfg, bytes);
    return ps ? qw_pcie_stage_input_rows(ps) : g->bhost[0];
}

/* Whether the host drains stage 0's rank 0 before the hop: every path but the
 * event-chained hop, which queues the hop behind it instead. */
static int qw_stage_hop_host_wait(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg,
                                  size_t bytes)
{
    qw_pcie_stage *ps = qw_pcie_for(g, cfg, bytes);
    return !ps || !qw_pcie_stage_flags(ps).hop_event;
}

/* Stage s's input rows onto both of its ranks: stage 0's from the input rows
 * (qw_stage_input_rows), stage 1's from stage 0's rank 0 (`prev`). Returns 0, or -1. */
static int qw_stage_input_upload(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg, int s,
                                 gpu_batch_scratch *prev, gpu_batch_scratch *const b[2],
                                 size_t bytes)
{
    qw_pcie_stage *ps = qw_pcie_for(g, cfg, bytes);
    if (ps)
        return s == 0 ? qw_pcie_stage_fan_in(ps, b[0]->hidden, b[1]->hidden, bytes)
                      : qw_pcie_stage_hop(ps, prev->hidden, b[0]->hidden, b[1]->hidden, bytes);
    if (s == 1)
        fc_buffer_to_cpu_n(prev->hidden, g->bhost[0], bytes);
    fc_buffer_from_cpu_n(b[0]->hidden, g->bhost[0], bytes);
    fc_buffer_from_cpu_n(b[1]->hidden, g->bhost[0], bytes);
    return 0;
}

static int tp_batch_walk(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, const uint32_t *tokens,
                         int n, pn_qwen35moe_state *st, int pos0, int spec,
                         fc_cuda_ctx **ctx_out, gpu_batch_scratch **b_out,
                         gpu_die_scratch **sc_out)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const int c = cfg->n_embd;
    const size_t hidden_bytes = (size_t)n * c * sizeof(float);
    gpu_batch_scratch *b[2] = { NULL, NULL };
    gpu_die_scratch *sc = NULL;

    if (tp_exchange_ready(g, cfg, n) < 0)
        return -1;
    for (int s = 0; s < 2; s++) {
        gpu_batch_scratch *prev = b[0];
        for (int p = 0; p < 2; p++) {
            b[p] = gpu_get_batch_scratch(g, g->ctx[2 * s + p], cfg, n);
            sc = gpu_get_scratch(g, g->ctx[2 * s + p], cfg);
            if (!b[p] || !sc)
                return -1;
        }
        if (s == 0) {
            float *rows = qw_stage_input_rows(g, cfg, hidden_bytes);
            if (!rows)
                return -1;
            for (int t = 0; t < n; t++)
                pn_qwen35moe_model_embed_row(m, tokens[t], rows + (size_t)t * c);
        } else if (qw_stage_hop_host_wait(g, cfg, hidden_bytes)) {
            fc_sync_ctx(g->ctx[0]);
        }
        if (qw_stage_input_upload(g, cfg, s, prev, b, hidden_bytes) < 0)
            return -1;
        /* A MoE verify runs the device-position bodies as a stage graph. */
        int vrc = spec && cfg->n_expert ? tp_verify_stage(m, g, s, b, st, pos0, n) : 1;
        if (vrc < 0)
            return -1;
        for (int layer = g->layer_lo[2 * s]; vrc == 1 && layer < g->layer_hi[2 * s]; layer++)
            if (tp_layer_forward_batch(m, g, s, layer, b, st, pos0, n, spec) < 0)
                return -1;
    }
    *ctx_out = g->ctx[3];
    *b_out = b[1];
    *sc_out = sc;
    return 0;
}

/* ============================================================
 * Speculative verify — device-position layer bodies + per-die CUDA graphs
 *
 * The verify (spec = 1) runs every position-dependent kernel in its device-position
 * form — RoPE (fc_rope_neox_partial_pos), the KV append (the slots append, every row
 * mapped to the single-user ring) and the rows attention (fc_attn_flash_decode_rows_q4_dev)
 * — with the rows' positions in a device buffer. Those are the kernels the plain decode
 * step itself runs, so the rows stay byte-exact, and a die's whole layer range becomes
 * position-invariant: it is captured ONCE per (rows, split-grid width) as a CUDA graph and
 * replayed with one launch (QWEN35_VERIFY_GRAPH, default on; =0 runs the same bodies
 * direct). The host-side work of a layer (KV ring watermark, window snapshot, SSM state
 * seeding / snapshot buffers) runs before the replay, outside the graph.
 * ============================================================ */

static int verify_graph_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_VERIFY_GRAPH");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Host prep of one layer for a verify of n rows at pos0 (no device work in the common
 * case). Returns 0, or -1. */
static int verify_layer_prep(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                             int layer, pn_qwen35moe_state *s, int pos0, int n)
{
    int idx = pn_qwen35moe_model_layer_index(m, layer);

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx];
        pn_qwen35moe_kv *kv = &s->kv[idx];
        if (gpu_attn_kv_cache_sync(d, kv, pos0) < 0 || qw_spec_win_snapshot(d, kv, pos0, n) < 0)
            return -1;
        d->kv_synced_pos = pos0;   /* the commit advances it by the accepted rows */
        kv->kv_mirror_stale = 1;
        return 0;
    }
    gpu_ssm_layer *d = &g->ssm[idx];
    pn_qwen35moe_ssm_state *st = &s->ssm[idx];
    size_t conv_n = (size_t)st->conv_minus_1 * st->conv_dim;
    size_t ssm_n = (size_t)st->state_size * st->state_size * st->n_v_heads;
    if (!d->conv_state_g)
        d->conv_state_g = fc_buffer_alloc_ctx(ctx, conv_n * sizeof(float));
    if (!d->ssm_state_g)
        d->ssm_state_g = fc_buffer_alloc_ctx(ctx, ssm_n * sizeof(float));
    if (!d->conv_state_g || !d->ssm_state_g)
        return -1;
    if (d->host_state != (const void *)st || pos0 == 0)
        qw_ssm_seed_shared(g, d, st, pos0);
    /* Snapshot buffers at the widest verify, so their addresses never move under a graph. */
    if (d->snaps_cap < FC_ATTN_ROWS_MAX) {
        fc_buffer_free(d->conv_snaps);
        fc_buffer_free(d->ssm_snaps);
        d->conv_snaps = fc_buffer_alloc_ctx(ctx, FC_ATTN_ROWS_MAX * conv_n * sizeof(float));
        d->ssm_snaps = fc_buffer_alloc_ctx(ctx, FC_ATTN_ROWS_MAX * ssm_n * sizeof(float));
        d->snaps_cap = (d->conv_snaps && d->ssm_snaps) ? FC_ATTN_ROWS_MAX : 0;
        if (!d->snaps_cap)
            return -1;
    }
    (void)n;
    return 0;
}

/* Verify attention runs the decode slots call (the rows as slots of one ring); QWEN35_VERIFY_KVF=0
 * keeps the rows kernel. Latched. */
static int verify_kvf_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_VERIFY_KVF");
        v = !(env && env[0] == '0');
    }
    return v;
}

/* The attention half of a verify layer, device-position kernels only. */
static int verify_attn_body(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                            const pn_qwen35moe_config *cfg, gpu_batch_scratch *b, int n,
                            fc_buffer *pos_g, fc_buffer *rows_g, int launch_splits, int di,
                            int max_seq, int core_only)
{
    fc_cuda_ctx *ctx = d->ctx;
    const int attn_dim = cfg->n_head * cfg->head_dim;
    int fold = (gpu_fuse_mask() & 128) != 0;

    if (gpu_matvec_nx(&d->attn_q, b->x_norm, b->a_qg, n) < 0 ||
        gpu_matvec_nx(&d->attn_k, b->x_norm, b->a_k, n) < 0 ||
        gpu_matvec_nx(&d->attn_v, b->x_norm, b->a_v, n) < 0)
        return -1;
    fc_attn_qz_split_ctx(ctx, b->a_qg, b->a_q, b->a_z, cfg->n_head, cfg->head_dim, n);
    fc_qk_rms_norm_ctx(ctx, b->a_q, d->attn_q_norm.buf, n * cfg->n_head, cfg->head_dim,
                       cfg->rms_norm_eps);
    fc_qk_rms_norm_ctx(ctx, b->a_k, d->attn_k_norm.buf, n * cfg->n_head_kv, cfg->head_dim,
                       cfg->rms_norm_eps);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_q, cfg->n_head, cfg->head_dim, cfg->rope_dim, pos_g,
                                 cfg->rope_theta, n);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_k, cfg->n_head_kv, cfg->head_dim, cfg->rope_dim,
                                 pos_g, cfg->rope_theta, n);
    fc_attn_kv_append_slots_q4_ctx(ctx, b->a_k, b->a_v, d->k_cache_g, d->v_cache_g,
                                   d->k_sink_g, d->v_sink_g, d->k_win_g, d->v_win_g,
                                   cfg->n_head_kv, cfg->head_dim, max_seq, pos_g, rows_g, n,
                                   pn_kvq4_sink(), pn_kvq4_win());
    if (verify_kvf_on()) {
        /* The n rows as n slots of the single-user ring (rows_g all 0, pos_g pos0..): the
         * decode step's own call, so each row is that step's kernel at its position. */
        float scale = qw_scale_inv_sqrt(cfg->head_dim);
        int kps = dec_keys_per_split();
        if (fold && fc_attn_flash_decode_slots_q4_gated_ctx(
                        ctx, b->a_q, d->k_cache_g, d->v_cache_g, d->k_sink_g, d->v_sink_g,
                        d->k_win_g, d->v_win_g, g->bv_m_g[di], g->bv_l_g[di], g->bv_acc_g[di],
                        b->a_out, cfg->n_head, cfg->n_head_kv, cfg->head_dim, max_seq, pos_g,
                        rows_g, n, launch_splits, kps, PN_QWEN35MOE_DEC_SPLIT_MAX,
                        pn_kvq4_sink(), pn_kvq4_win(), attn_sink(), attn_window(), scale,
                        b->a_z) != 0)
            fold = 0;   /* as the decode step: ungated call, then the gate below */
        if (!fold && fc_attn_flash_decode_slots_q4_ctx(
                         ctx, b->a_q, d->k_cache_g, d->v_cache_g, d->k_sink_g, d->v_sink_g,
                         d->k_win_g, d->v_win_g, g->bv_m_g[di], g->bv_l_g[di],
                         g->bv_acc_g[di], b->a_out, cfg->n_head, cfg->n_head_kv,
                         cfg->head_dim, max_seq, pos_g, rows_g, n, launch_splits, kps,
                         PN_QWEN35MOE_DEC_SPLIT_MAX, pn_kvq4_sink(), pn_kvq4_win(),
                         attn_sink(), attn_window(), scale) < 0)
            return -1;
    } else if (fc_attn_flash_decode_rows_q4_dev_ctx(ctx, b->a_q, d->k_cache_g, d->v_cache_g,
                                             d->k_sink_g, d->v_sink_g, d->k_win_g, d->v_win_g,
                                             g->bv_m_g[di], g->bv_l_g[di], g->bv_acc_g[di],
                                             fold ? b->a_z : NULL, b->a_out, cfg->n_head,
                                             cfg->n_head_kv, cfg->head_dim, max_seq, pos_g, n,
                                             launch_splits, dec_keys_per_split(),
                                             PN_QWEN35MOE_DEC_SPLIT_MAX, pn_kvq4_sink(),
                                             pn_kvq4_win(),
                                             qw_scale_inv_sqrt(cfg->head_dim)) != 0)
        return -1;
    if (!fold) {
        fc_sigmoid_ctx(ctx, b->a_z, n * attn_dim);
        fc_mul_ctx(ctx, b->a_out, b->a_z, b->a_out, n * attn_dim);
    }
    if (core_only)      /* QWEN35_TP: a_out holds the rank's heads */
        return 0;
    if (gpu_matvec_nx(&d->attn_output, b->a_out, b->block_out, n) < 0)
        return -1;
    fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden, n * cfg->n_embd);
    return 0;
}

/* One verify layer's device work (after verify_layer_prep): capturable. */
static int verify_layer_body(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                             int layer, gpu_batch_scratch *b, int n, fc_buffer *pos_g,
                             fc_buffer *rows_g, int launch_splits, int di, int max_seq)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd, idx = pn_qwen35moe_model_layer_index(m, layer);
    gpu_moe *moe;

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx];
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c, cfg->rms_norm_eps, n);
        if (verify_attn_body(g, d, cfg, b, n, pos_g, rows_g, launch_splits, di, max_seq, 0) < 0)
            return -1;
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf, b->x_norm, c,
                           cfg->rms_norm_eps, n);
        moe = &d->moe;
    } else {
        gpu_ssm_layer *d = &g->ssm[idx];
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c, cfg->rms_norm_eps, n);
        if (gpu_ssm_block_batch(d, cfg, b, n, 1, 0) < 0)
            return -1;
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf, b->x_norm, c,
                           cfg->rms_norm_eps, n);
        moe = &d->moe;
    }
    /* Decode economics: no dedup unless QWEN35_MOE_DEDUP_DECODE=1 asks to measure it (the
     * dedup hid is bit-exact with the _nx hid, so the rows stay byte-exact either way). */
    return gpu_moe_block_batch(ctx, moe, cfg, b, n, gpu_moe_dedup_decode_override());
}

/* Whether die di's layers can run the device-position verify at all (and be captured). */
static int verify_dev_eligible(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int di,
                               pn_qwen35moe_state *s, int n)
{
    fc_cuda_ctx *ctx = g->ctx[di];

    if (!batch_verify_on() || n > FC_ATTN_ROWS_MAX || attn_window() > 0 || prof_on() ||
        bv_trace_on() || !qw_kv_mirror_lazy() || s->n_attn < 1 || !s->kv[0].q4 ||
        !fc_attn_flash_decode_rows_q4_dev_available_ctx(ctx))
        return 0;
    for (int layer = g->layer_lo[di]; layer < g->layer_hi[di]; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        int attn = pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN;
        if (!(attn ? g->attn[idx].moe.grouped : g->ssm[idx].moe.grouped))
            return 0;
    }
    return 1;
}

/* A fingerprint of every device address a captured verify graph bakes in. */
static uint64_t verify_graph_key(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int di,
                                 gpu_batch_scratch *b)
{
    uint64_t h = 1469598103934665603ull;
#define VK_MIX(p) (h = (h ^ (uint64_t)(uintptr_t)(p)) * 1099511628211ull)
    VK_MIX(b->hidden); VK_MIX(b->x_norm); VK_MIX(b->a_qg); VK_MIX(b->m_hid); VK_MIX(b->s_qkv);
    VK_MIX(g->bv_acc_g[di]); VK_MIX(g->vpos_g[di]);
    for (int layer = g->layer_lo[di]; layer < g->layer_hi[di]; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
            gpu_attn_layer *d = &g->attn[idx];
            VK_MIX(d->k_cache_g); VK_MIX(d->v_cache_g); VK_MIX(d->k_win_g); VK_MIX(d->k_sink_g);
            VK_MIX(d->cache_cap);
        } else {
            gpu_ssm_layer *d = &g->ssm[idx];
            VK_MIX(d->conv_state_g); VK_MIX(d->ssm_state_g); VK_MIX(d->conv_snaps);
            VK_MIX(d->ssm_snaps);
        }
    }
#undef VK_MIX
    return h;
}

/* Run die di's layer range of a verify (n rows at pos0) on the device-position bodies,
 * replaying / capturing its graph. Returns 0, 1 (not eligible: caller runs the classic
 * per-layer path), or -1. */
static int verify_segment(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int di,
                          gpu_batch_scratch *b, pn_qwen35moe_state *s, int pos0, int n)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    fc_cuda_ctx *ctx = g->ctx[di];
    int32_t posv[2 * FC_ATTN_ROWS_MAX];
    int launch_splits = 1, max_seq = s->kv[0].max_seq;

    if (!verify_dev_eligible(m, g, di, s, n))
        return 1;
    if (!g->vpos_g[di]) {
        g->vpos_g[di] = fc_buffer_alloc_ctx(ctx, sizeof(posv));
        g->vrows_g[di] = g->vpos_g[di]
            ? fc_buffer_offset_view(g->vpos_g[di], FC_ATTN_ROWS_MAX * sizeof(int32_t)) : NULL;
        if (!g->vrows_g[di])
            return -1;
    }
    if (!g->bv_acc_g[di]) {
        size_t np = (size_t)FC_ATTN_ROWS_MAX * cfg->n_head * PN_QWEN35MOE_DEC_SPLIT_MAX;
        g->bv_m_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        g->bv_l_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        g->bv_acc_g[di] = fc_buffer_alloc_ctx(ctx, np * (size_t)cfg->head_dim * sizeof(float));
        if (!g->bv_m_g[di] || !g->bv_l_g[di] || !g->bv_acc_g[di])
            return -1;
    }
    for (int r = 0; r < FC_ATTN_ROWS_MAX; r++) {
        posv[r] = pos0 + (r < n ? r : n - 1);
        posv[FC_ATTN_ROWS_MAX + r] = 0;    /* every row appends to the single-user ring */
        int seq = pos0 + r + 1, sl;
        if (r < n) {
            int ns = (seq + dec_keys_per_split() - 1) / dec_keys_per_split();
            ns = ns < 1 ? 1 : ns > PN_QWEN35MOE_DEC_SPLIT_MAX ? PN_QWEN35MOE_DEC_SPLIT_MAX : ns;
            sl = (((seq + ns - 1) / ns) + 15) & ~15;
            if ((seq + sl - 1) / sl > launch_splits)
                launch_splits = (seq + sl - 1) / sl;
        }
    }
    fc_buffer_from_cpu_n(g->vpos_g[di], posv, sizeof(posv));
    for (int layer = g->layer_lo[di]; layer < g->layer_hi[di]; layer++)
        if (verify_layer_prep(m, g, ctx, layer, s, pos0, n) < 0)
            return -1;

    gpu_verify_graph *vg = &g->vgraph[di][n - 1];
    uint64_t key = verify_graph_key(m, g, di, b);
    if (vg->graph && (vg->key != key || vg->splits != launch_splits)) {
        fc_graph_free(vg->graph);
        vg->graph = NULL;
    }
    if (vg->graph && fc_graph_launch(vg->graph) == 0) {
        vg->replays++;
        return 0;
    }
    if (vg->graph) {
        fc_graph_free(vg->graph);
        vg->graph = NULL;
    }
    /* Capture (the replay of this very call runs it); a shape whose capture keeps
     * failing stops trying after 3 and runs direct. */
    if (gpu_graph_enabled() && verify_graph_on() && vg->failures < 3 &&
        fc_graph_begin_ctx(ctx) == 0) {
        int rc = 0;
        for (int layer = g->layer_lo[di]; layer < g->layer_hi[di] && rc == 0; layer++)
            rc = verify_layer_body(m, g, ctx, layer, b, n, g->vpos_g[di], g->vrows_g[di],
                                   launch_splits, di, max_seq);
        vg->graph = fc_graph_end_ctx(ctx);
        if (rc == 0 && vg->graph && fc_graph_launch(vg->graph) == 0) {
            vg->key = key;
            vg->splits = launch_splits;
            vg->captures++;
            return 0;
        }
        if (vg->graph) {
            fc_graph_free(vg->graph);
            vg->graph = NULL;
        }
        vg->failures++;
    }
    for (int layer = g->layer_lo[di]; layer < g->layer_hi[di]; layer++)
        if (verify_layer_body(m, g, ctx, layer, b, n, g->vpos_g[di], g->vrows_g[di],
                              launch_splits, di, max_seq) < 0)
            return -1;
    return 0;
}

/* Host prep of rank p's half of one layer for a QWEN35_TP verify (verify_layer_prep
 * with the half's local state sizes). */
static int tp_verify_layer_prep(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int p, int layer,
                                pn_qwen35moe_state *s, int pos0, int n)
{
    int idx = pn_qwen35moe_model_layer_index(m, layer);

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
        pn_qwen35moe_kv *kv = &s->kv[idx];
        if (gpu_attn_kv_cache_sync(d, kv, pos0) < 0 || qw_spec_win_snapshot(d, kv, pos0, n) < 0)
            return -1;
        d->kv_synced_pos = pos0;   /* the commit advances it by the accepted rows */
        kv->kv_mirror_stale = 1;
        return 0;
    }
    gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
    pn_qwen35moe_ssm_state *st = &s->ssm[idx];
    size_t conv_n = tp_ssm_conv_n(g, st), ssm_n = tp_ssm_rec_n(g, st);
    if (!d->conv_state_g)
        d->conv_state_g = fc_buffer_alloc_ctx(d->ctx, conv_n * sizeof(float));
    if (!d->ssm_state_g)
        d->ssm_state_g = fc_buffer_alloc_ctx(d->ctx, ssm_n * sizeof(float));
    if (!d->conv_state_g || !d->ssm_state_g)
        return -1;
    if (d->host_state != (const void *)st || pos0 == 0)
        qw_ssm_seed_shared(g, d, st, pos0);
    if (d->snaps_cap < FC_ATTN_ROWS_MAX) {
        fc_buffer_free(d->conv_snaps);
        fc_buffer_free(d->ssm_snaps);
        d->conv_snaps = fc_buffer_alloc_ctx(d->ctx, FC_ATTN_ROWS_MAX * conv_n * sizeof(float));
        d->ssm_snaps = fc_buffer_alloc_ctx(d->ctx, FC_ATTN_ROWS_MAX * ssm_n * sizeof(float));
        d->snaps_cap = (d->conv_snaps && d->ssm_snaps) ? FC_ATTN_ROWS_MAX : 0;
        if (!d->snaps_cap)
            return -1;
        g->tp_gen++;
    }
    return 0;
}

/* Device work of one layer of stage s for a QWEN35_TP verify: verify_layer_body's
 * device-position cores on each rank's half (local dims), then tp_layer_tail. */
static int tp_verify_layer_dev(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s, int layer,
                               gpu_batch_scratch *const b[2], int n, int launch_splits,
                               int max_seq)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const pn_qwen35moe_config *lc = &g->tp_cfg;
    const int c = cfg->n_embd, idx = pn_qwen35moe_model_layer_index(m, layer);
    const int is_attn = pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN;
    fc_buffer *core[2];

    for (int p = 0; p < 2; p++) {
        const int die = 2 * s + p;
        if (is_attn) {
            gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
            fc_rms_norm_nx_ctx(d->ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, n);
            if (verify_attn_body(g, d, lc, b[p], n, g->vpos_g[die], g->vrows_g[die],
                                 launch_splits, die, max_seq, 1) < 0)
                return -1;
            core[p] = b[p]->a_out;
        } else {
            gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
            fc_rms_norm_nx_ctx(d->ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, n);
            if (gpu_ssm_block_batch(d, lc, b[p], n, 1, 1) < 0)
                return -1;
            core[p] = b[p]->s_gated;
        }
    }
    return tp_layer_tail(g, cfg, s, is_attn, idx, core, b, n);
}

/* QWEN35_TP verify of stage s (n rows at pos0, both ranks) in verify_segment's
 * device-position form, replayed as one graph per (stage, n) that spans both ranks
 * (g->vgraph[2s][n - 1]), fenced against rank 1's stream like the slots stage graph.
 * Returns 1 when not eligible (the caller runs tp_layer_forward_batch). */
static int tp_verify_stage(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s,
                           gpu_batch_scratch *const b[2], pn_qwen35moe_state *st, int pos0,
                           int n)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const int lo = g->layer_lo[2 * s], hi = g->layer_hi[2 * s];
    fc_cuda_ctx *c0 = g->ctx[2 * s], *c1 = g->ctx[2 * s + 1];
    fc_event *e0 = g->tp_gev[2 * s], *e1 = g->tp_gev[2 * s + 1];
    int32_t posv[2 * FC_ATTN_ROWS_MAX];
    int launch_splits = 1, max_seq = st->kv[0].max_seq;

    for (int p = 0; p < 2; p++)
        if (!verify_dev_eligible(m, g, 2 * s + p, st, n))
            return 1;
    for (int r = 0; r < FC_ATTN_ROWS_MAX; r++) {
        posv[r] = pos0 + (r < n ? r : n - 1);
        posv[FC_ATTN_ROWS_MAX + r] = 0;    /* every row appends to the single-user ring */
        int seq = pos0 + r + 1, sl;
        if (r < n) {
            int ns = (seq + dec_keys_per_split() - 1) / dec_keys_per_split();
            ns = ns < 1 ? 1 : ns > PN_QWEN35MOE_DEC_SPLIT_MAX ? PN_QWEN35MOE_DEC_SPLIT_MAX : ns;
            sl = (((seq + ns - 1) / ns) + 15) & ~15;
            if ((seq + sl - 1) / sl > launch_splits)
                launch_splits = (seq + sl - 1) / sl;
        }
    }
    /* Rank 0's prep marks every layer's mirror stale before rank 1's prep runs. */
    for (int layer = lo; layer < hi; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN &&
            tp_attn_kv_cache_sync_both(g, idx, &st->kv[idx], pos0) < 0)
            return -1;
    }
    uint64_t key = 1469598103934665603ull;
#define TV_MIX(v) (key = (key ^ (uint64_t)(uintptr_t)(v)) * 1099511628211ull)
    for (int p = 0; p < 2; p++) {
        const int di = 2 * s + p;
        fc_cuda_ctx *ctx = g->ctx[di];
        if (!g->vpos_g[di]) {
            g->vpos_g[di] = fc_buffer_alloc_ctx(ctx, sizeof(posv));
            g->vrows_g[di] = g->vpos_g[di]
                ? fc_buffer_offset_view(g->vpos_g[di], FC_ATTN_ROWS_MAX * sizeof(int32_t))
                : NULL;
            if (!g->vrows_g[di])
                return -1;
        }
        if (!g->bv_acc_g[di]) {
            size_t np = (size_t)FC_ATTN_ROWS_MAX * cfg->n_head * PN_QWEN35MOE_DEC_SPLIT_MAX;
            g->bv_m_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
            g->bv_l_g[di] = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
            g->bv_acc_g[di] = fc_buffer_alloc_ctx(ctx, np * (size_t)cfg->head_dim * sizeof(float));
            if (!g->bv_m_g[di] || !g->bv_l_g[di] || !g->bv_acc_g[di])
                return -1;
        }
        fc_buffer_from_cpu_n(g->vpos_g[di], posv, sizeof(posv));
        for (int layer = lo; layer < hi; layer++)
            if (tp_verify_layer_prep(m, g, p, layer, st, pos0, n) < 0)
                return -1;
        TV_MIX(b[p]); TV_MIX(b[p]->gen); TV_MIX(g->bv_acc_g[di]); TV_MIX(g->vpos_g[di]);
        for (int layer = lo; layer < hi; layer++) {
            int idx = pn_qwen35moe_model_layer_index(m, layer);
            if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
                gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
                TV_MIX(d->k_cache_g); TV_MIX(d->v_cache_g); TV_MIX(d->k_win_g);
                TV_MIX(d->k_sink_g); TV_MIX(d->cache_cap);
            } else {
                gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
                TV_MIX(d->conv_state_g); TV_MIX(d->ssm_state_g); TV_MIX(d->conv_snaps);
                TV_MIX(d->ssm_snaps);
            }
        }
    }
    TV_MIX(g->tp_gen);
#undef TV_MIX

    gpu_verify_graph *vg = &g->vgraph[2 * s][n - 1];
    if (vg->graph && (vg->key != key || vg->splits != launch_splits)) {
        fc_graph_free(vg->graph);
        vg->graph = NULL;
    }
    if (!vg->graph && gpu_graph_enabled() && verify_graph_on() && tp_graph_on() &&
        !tp_sync_on() && vg->failures < 3 && fc_graph_begin_ctx(c0) == 0) {
        /* Rank 1's stream joins rank 0's capture and rejoins at the end. */
        g->tp_capturing = 1;
        int ok = fc_event_record_ctx(c0, e0) == 0 && fc_event_wait_ctx(c1, e0) == 0;
        for (int layer = lo; ok && layer < hi; layer++)
            ok = tp_verify_layer_dev(m, g, s, layer, b, n, launch_splits, max_seq) == 0;
        ok = fc_event_record_ctx(c1, e1) == 0 && fc_event_wait_ctx(c0, e1) == 0 && ok;
        g->tp_capturing = 0;
        fc_graph *gr = fc_graph_end_ctx(c0);
        if (ok && gr) {
            vg->graph = gr;
            vg->key = key;
            vg->splits = launch_splits;
            vg->captures++;
        } else {
            fc_graph_free(gr);
            vg->failures++;
        }
    }
    if (vg->graph) {
        if (fc_event_record_ctx(c1, e1) != 0 || fc_event_wait_ctx(c0, e1) != 0 ||
            fc_graph_launch(vg->graph) != 0 ||
            fc_event_record_ctx(c0, e0) != 0 || fc_event_wait_ctx(c1, e0) != 0)
            return -1;
        vg->replays++;
        return 0;
    }
    for (int layer = lo; layer < hi; layer++)
        if (tp_verify_layer_dev(m, g, s, layer, b, n, launch_splits, max_seq) < 0)
            return -1;
    return 0;
}

/* Phase 9b — prefill chunk pipelining (QWEN35_PREFILL_PIPELINE, default OFF).
 *
 * The two dies hold disjoint layer ranges (A = [0, split), B = [split,
 * n_layer)) and today every prefill chunk drains BOTH of them before the
 * serve's chunk loop advances: the die-A sync at the boundary hand-off, then
 * the die-B sync in front of the LM-head fetch. So die A idles for the whole
 * back half of every chunk and die B for the whole front half.
 *
 * The dependency graph does not require that. Chunk N+1's die-A work needs
 * only state die A itself wrote (its own KV rings and its own SSM recurrence
 * for layers [0, split)); it has NO dependency on chunk N's die-B work. The
 * SSM recurrence IS sequential across chunks, but it is sequential *per die*:
 * layer L's carry is consumed by layer L, and layer L lives on exactly one
 * die. Chunk N+1's die-A layers therefore serialize behind chunk N's die-A
 * layers — which is the same stream, so CUDA already orders it — and overlap
 * chunk N's die-B layers, which is the win.
 *
 * All this needs is for the host not to block on die B. Dropping the tail
 * (final norm + LM head + fc_sync_ctx + logits fetch) on every chunk but the
 * last does exactly that: the host issues die B's segment asynchronously and
 * walks straight into chunk N+1's die-A segment. The rendezvous that keeps it
 * correct is already in the code — the hand-off's `fc_buffer_from_cpu_n` into
 * bb->hidden is a legacy-null-stream HtoD in ctx_b, so it implicitly waits for
 * every outstanding blocking-stream op in that context, i.e. for die B to
 * finish chunk N before its input buffer is overwritten. Steady-state period
 * becomes max(t_A, t_B) instead of t_A + t_B.
 *
 * NOTHING about the arithmetic changes: same kernels, same order, same bytes
 * across the link, same per-die stream. Only the LM head of the discarded
 * intermediate chunks stops being computed — and the serve already threw those
 * logits away (sg_prefill_range tm_free()s every chunk but the last).
 *
 * Callers MUST drain (pn_qwen35moe_prefill_drain_cuda) before touching state
 * that an in-flight die-B segment is still writing — the range helpers do this
 * on every early exit. */
static int qw_prefill_pipeline_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_PREFILL_PIPELINE");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* ★ QWEN35_PROF makes the pipeline INERT, and the inertness is invisible.
 *
 * bprof_lap() / ssm_lap() / sprof_lap() each call fc_sync_ctx(ctx) on the die
 * that issued the phase, so with profiling on every layer of die B's segment is
 * drained the moment it is issued — the last thing a die-B layer does is
 * bprof_lap(BPROF_MOE, ctx_b, ...). By the time gpu_batch_tokens_core reaches
 * `if (no_tail) return`, there is nothing left in flight for the next chunk's
 * die-A segment to overlap. A profiled run is a SERIALISED run, on both entry
 * points, whatever QWEN35_PREFILL_PIPELINE says.
 *
 * The serve's `[prefill_pipeline] ON: ...` sentinel does NOT catch this: it
 * counts the code path the chunk loop chose, not work actually left in flight,
 * so it prints duty=100% serialised=0 for a fully serialised profiled run.
 * Neither does comparing BPROF's bucket sum to the same run's own wall clock —
 * both quantities come from the same serialised run.
 *
 * This cost the campaign a 1.85x error in its baseline: a profiled harness
 * measured 31.37 tok/s at 14 080 while the unprofiled serve does ~58-68 at the
 * same depth. Read absolute seconds and tok/s from an UNPROFILED run; read
 * shares and A/B ratios from a profiled one (they survive, because both dies
 * carry the same layer mix). */
static void qw_warn_prof_serialises_pipeline(void)
{
    static int warned = 0;
    if (warned || !prof_on())
        return;
    warned = 1;
    fprintf(stderr, "[qwen35moe] WARNING: QWEN35_PROF=1 with "
                    "QWEN35_PREFILL_PIPELINE=1 — the profiler's per-phase "
                    "fc_sync_ctx drains die B inside every chunk, so the two "
                    "dies run SERIALISED. Shares and A/B ratios are valid; "
                    "absolute seconds and tok/s are pipeline-OFF numbers and "
                    "are ~1.6-2.0x slower than the unprofiled serve.\n");
}

int pn_qwen35moe_prefill_pipeline_enabled(void)
{
    int on = qw_prefill_pipeline_on();
    if (on)
        qw_warn_prof_serialises_pipeline();
    return on;
}

/**
 * LM head for the LAST position of a batched pass, into s->logits.
 *
 * `b->x_norm` must already hold the final norm for all n positions; this slices
 * row n-1 out of it and runs the same single-column head matvec + fetch the
 * single-token path uses.
 *
 * Shared by the sequential batched tail and the pipelined prefill so both turn
 * the same activations into the same logit bytes — the pipeline's bit-exactness
 * argument depends on there being exactly one implementation of this.
 *
 * @return s->logits (borrowed; owned by the state), or NULL on failure.
 *
 * Thread-safe: no (mutates per-die scratch and s->logits)
 * Reentrant: no
 * Signal-safe: no
 */
static float *gpu_batch_head_last_row(pn_qwen35moe_gpu *g,
                                      const pn_qwen35moe_config *cfg,
                                      fc_cuda_ctx *ctx, gpu_batch_scratch *b,
                                      gpu_die_scratch *sc, gpu_host_scratch h,
                                      pn_qwen35moe_state *s, int n)
{
    int c = cfg->n_embd;
    fc_buffer_copy_d2d_ctx(ctx, sc->r_xnorm, 0, b->x_norm,
                           (size_t)(n - 1) * c * sizeof(float),
                           (size_t)c * sizeof(float));
    gpu_matvec(&g->output, sc->r_xnorm, sc->r_logits, h.a, h.b);
    fc_sync_ctx(ctx);
    gpu_fetch_logits(NULL, ctx, sc->r_logits, s, cfg->n_vocab);
    return s->logits;
}

/* Shared core of the batched prefill and the speculative verify. spec = 0:
 * the original prefill semantics (all_logits / last-position fetch). spec =
 * 1: every SSM layer's seq scans write per-position state snapshots, and the
 * tail computes each position's greedy argmax ON the die (batched LM head +
 * device top-k(1) per row) into argmax_out — s->logits is not updated.
 * no_tail = 1 (9b, prefill-only, spec=0 and all_logits=NULL): skip the whole
 * head phase and return WITHOUT syncing die B, leaving its segment in flight. */
static float *gpu_batch_tokens_core(pn_qwen35moe_model *m,
                                    const uint32_t *tokens, int n,
                                    pn_qwen35moe_state *s, int pos0,
                                    float *all_logits, int spec,
                                    int32_t *argmax_out, int no_tail)
{
    if (!m || !s || !tokens || n <= 0)
        return NULL;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return NULL;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    for (int t = 0; t < n; t++)
        if ((int)tokens[t] >= cfg->n_vocab)
            return NULL;

    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0)
            return NULL;
        g->host_ok = 1;
    }
    gpu_host_scratch h = g->host;
    if (gpu_get_batch_host(g, cfg, n) < 0)
        return NULL;

    /* ---- Walk the dies in order: die 0 embeds all n tokens, each die runs
     * its layer range, and each boundary hands the [n][c] activations to the
     * next die. ---- */
    fc_cuda_ctx *ctx = NULL;
    gpu_batch_scratch *b = NULL;
    gpu_die_scratch *sc = NULL;
    size_t hidden_bytes = (size_t)n * c * sizeof(float);

    if (g->tp && tp_batch_walk(m, g, tokens, n, s, pos0, spec, &ctx, &b, &sc) < 0)
        return NULL;
    for (int d = 0; !g->tp && d < g->n_die; d++) {
        if (d > 0)
            gpu_handoff_export(g, d - 1, b->hidden, g->bhost[0], hidden_bytes);
        if (gpu_select_die(g, g->die[d]) < 0)
            return NULL;
        gpu_batch_scratch *bd = gpu_get_batch_scratch(g, g->ctx[d], cfg, n);
        gpu_die_scratch *scd = gpu_get_scratch(g, g->ctx[d], cfg);
        if (!bd || !scd)
            return NULL;
        if (d == 0) {
            for (int t = 0; t < n; t++)
                pn_qwen35moe_model_embed_row(m, tokens[t],
                                             g->bhost[0] + (size_t)t * c);
            fc_buffer_from_cpu_n(bd->hidden, g->bhost[0], hidden_bytes);
        } else {
            gpu_handoff_import(g, d - 1, b->hidden, bd->hidden, g->bhost[0],
                               hidden_bytes);
        }

        int vrc = spec ? verify_segment(m, g, d, bd, s, pos0, n) : 1;
        if (vrc < 0)
            return NULL;
        for (int layer = g->layer_lo[d]; vrc > 0 && layer < g->layer_hi[d]; layer++) {
            if (gpu_layer_forward_batch(m, g, g->ctx[d], layer, bd, s, pos0, n,
                                        spec) < 0)
                return NULL;
        }
        ctx = g->ctx[d];
        b = bd;
        sc = scd;
    }

    /* 9b: an intermediate prefill chunk. Its logits are discarded by the
     * caller, so skip the head phase entirely and — the point of the whole
     * exercise — return without syncing the last die, leaving its segment in
     * flight so the next chunk's die-0 segment overlaps it. Returns the
     * success sentinel; s->logits is deliberately left stale (no caller in
     * this mode reads it, and the final chunk always refreshes it). With the MTP
     * head on, the chunk's output_norm'd rows are still captured (one norm on the
     * last die, queued behind its segment) so the serve can feed the head's
     * backlog; the trunk's state and logits are untouched either way. */
    if (no_tail) {
        if (g->mtp.on) {
            fc_rms_norm_nx_ctx(ctx, b->hidden, g->output_norm.buf, b->x_norm, c,
                               cfg->rms_norm_eps, n);
            mtp_capture_rows(g, ctx, b->x_norm, n, c);
        }
        return s->logits;
    }

    /* ---- Final norm over all positions + LM head. ---- */
    fc_rms_norm_nx_ctx(ctx, b->hidden, g->output_norm.buf, b->x_norm, c,
                       cfg->rms_norm_eps, n);
    mtp_capture_rows(g, ctx, b->x_norm, n, c);   /* the MTP head's h rows (no-op when off) */

    if (spec) {
        /* Per-position greedy argmax without leaving the die: ONE batched
         * LM-head launch into the [n][n_vocab] spec buffer, then a device
         * top-k(1) per row (bit-exact iterative argmax, same kernel the
         * serving fetch uses) — only n (id, logit) pairs cross the 0.4 GB/s
         * host link instead of n full vocab vectors. */
        if (!argmax_out)
            return NULL;
        if (g->spec_logits_cap < n) {
            fc_buffer_free(g->spec_logits);
            g->spec_logits = fc_buffer_alloc_ctx(ctx, (size_t)n *
                                                 cfg->n_vocab * sizeof(float));
            g->spec_logits_cap = g->spec_logits ? n : 0;
            if (!g->spec_logits_cap)
                return NULL;
        }
        if (gpu_matvec_nx(&g->output, b->x_norm, g->spec_logits, n) < 0)
            return NULL;
        if (g->spec_topk_k > 0) {
            /* Every row's top-k in ONE batched reduction, per row bit-exact to the plain
             * decode fetch (fc_topk_logits_slots_ctx): the caller decides the rows. */
            if (fc_topk_logits_slots_ctx(ctx, g->spec_logits, n, cfg->n_vocab,
                                         g->spec_topk_k, g->spec_topk_ids,
                                         g->spec_topk_vals) != 0)
                return NULL;
            return s->logits; /* success sentinel — contents NOT updated */
        }
        for (int t = 0; t < n; t++) {
            int id;
            float val;
            fc_buffer_copy_d2d_ctx(ctx, sc->r_logits, 0, g->spec_logits,
                                   (size_t)t * cfg->n_vocab * sizeof(float),
                                   (size_t)cfg->n_vocab * sizeof(float));
            if (fc_topk_logits_ctx(ctx, sc->r_logits, cfg->n_vocab, 1,
                                   &id, &val) != 0)
                return NULL;
            argmax_out[t] = id;
            /* The first draft mismatch ends the acceptable prefix — rows
             * past it are unreachable for the caller's accept scan, so
             * skip their top-k fetches (marked -1). */
            if (t + 1 < n && tokens[t + 1] != (uint32_t)id) {
                for (int r = t + 1; r < n; r++)
                    argmax_out[r] = -1;
                break;
            }
        }
        return s->logits; /* success sentinel — contents NOT updated */
    }

    if (all_logits) {
        /* Per-position logits (the bit-exact gate): slice each position's
         * x_norm and run the SAME single-column LM-head matvec + full D2H
         * the single-token path uses. */
        for (int t = 0; t < n; t++) {
            fc_buffer_copy_d2d_ctx(ctx, sc->r_xnorm, 0, b->x_norm,
                                   (size_t)t * c * sizeof(float),
                                   (size_t)c * sizeof(float));
            gpu_matvec(&g->output, sc->r_xnorm, sc->r_logits, h.a, h.b);
            fc_sync_ctx(ctx);
            fc_buffer_to_cpu(sc->r_logits,
                             all_logits + (size_t)t * cfg->n_vocab);
        }
        memcpy(s->logits, all_logits + (size_t)(n - 1) * cfg->n_vocab,
               (size_t)cfg->n_vocab * sizeof(float));
        s->topk_filled = 0;             /* s->logits holds real values */
        s->short_row_live = 0;
        return s->logits;
    }

    /* Serving path: only the LAST position's logits leave the die. */
    return gpu_batch_head_last_row(g, cfg, ctx, b, sc, h, s, n);
}

/* Largest row count one batched call may span before the q4 fp16 recent-window
 * ring aliases against itself.
 *
 * The ring laps at PN_KVQ4_WMAX (256) and a query row at cur_pos reads the last
 * `win` positions from it. A batched call appends ALL of its rows before ANY
 * row runs its attention, so the union of positions the call still needs is
 * (pos0 - win, pos0 + n - 1] — win + n - 1 rows. Keep that inside the ring and
 * every row reads its own true window:  n <= WMAX - win + 1. We cap at
 * WMAX - win (one row of slack, so the batch never touches the row it would
 * lap) — 128 rows at the default win=128, which is exactly serve's default
 * prefill chunk, so production never splits. Returns 0 for "no cap".
 *
 * Above the cap the kernel's win_live_lo guard keeps the result CORRECT (those
 * rows read the q4 body instead of the shadow) but no longer batch-invariant;
 * splitting here keeps a long chunk bit-identical to the same tokens prefilled
 * in serve-sized chunks. Non-q4 caches and win == 0 have no ring, no cap. */
static int qw_batch_row_cap(const pn_qwen35moe_state *s)
{
    if (!s || s->n_attn <= 0 || !s->kv || !s->kv[0].q4)
        return 0;
    int win = pn_kvq4_win();
    if (win <= 0)
        return 0;
    int cap = PN_KVQ4_WMAX - win;
    return cap < 1 ? 1 : cap;
}

static float *qw_batch_tokens_entry(pn_qwen35moe_model *m,
                                    const uint32_t *tokens, int n,
                                    pn_qwen35moe_state *s, int pos0,
                                    float *all_logits, int no_tail)
{
    /* Split an oversized batch into ring-safe sub-batches. Chunk boundaries
     * carry the KV/SSM state, so this is exactly the caller's own chunking —
     * the per-position logits land at the same offsets either way. */
    int cap = qw_batch_row_cap(s);
    if (cap > 0 && m && s && tokens && n > cap) {
        const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
        float *r = NULL;
        for (int off = 0; off < n; off += cap) {
            int nb = n - off < cap ? n - off : cap;
            r = qw_batch_tokens_entry(
                m, tokens + off, nb, s, pos0 + off,
                all_logits ? all_logits + (size_t)off * cfg->n_vocab : NULL,
                no_tail);
            if (!r)
                return NULL;
        }
        return r;
    }
    if (!prof_on())
        return gpu_batch_tokens_core(m, tokens, n, s, pos0, all_logits, 0,
                                     NULL, no_tail);
    memset(bprof_call, 0, sizeof(bprof_call));
    /* QWEN35_PROF_SUB additionally splits BPROF's two big buckets. The MoE
     * side needs no new instrumentation: gpu_moe_block_batch already laps
     * router+shared / routed gate+up / routed down for the slots profiler and
     * is shared with this path, so turning sprof_active on for the duration is
     * all it takes (sprof_active otherwise gates only decode-slots code, which
     * cannot run from here). The SSM side uses ssm_lap inside
     * gpu_ssm_block_batch. Both nest inside the BPROF spans. */
    int sub = prof_sub_on();
    if (sub) {
        memset(sprof_step_ph, 0, sizeof(sprof_step_ph));
        memset(ssmp_call, 0, sizeof(ssmp_call));
        sprof_active = 1;
    }
    double t0 = prof_now();
    float *r = gpu_batch_tokens_core(m, tokens, n, s, pos0, all_logits, 0,
                                     NULL, no_tail);
    if (sub) {
        sprof_active = 0;
        for (int ph = 0; ph < SPROF_NCOUNT; ph++)
            sprof_cum_ph[ph] += sprof_step_ph[ph];
    }
    if (r) {
        bprof_report(n, pos0, prof_now() - t0);
        if (sub) {
            double moe_sum = sprof_cum_ph[SPROF_MOE_SHEXP] +
                             sprof_cum_ph[SPROF_MOE_GATEUP] +
                             sprof_cum_ph[SPROF_MOE_DOWN];
            double ssm_sum = 0.0;
            for (int ph = 0; ph < SSMP_COUNT; ph++)
                ssm_sum += ssmp_cum[ph];
            fprintf(stderr, "MPROF cum n=%d moe |", n);
            fprintf(stderr, " router_shexp=%.3f(%.1f%%) gateup=%.3f(%.1f%%)"
                            " down=%.3f(%.1f%%)",
                    sprof_cum_ph[SPROF_MOE_SHEXP],
                    100.0 * sprof_cum_ph[SPROF_MOE_SHEXP] /
                        (moe_sum > 0 ? moe_sum : 1.0),
                    sprof_cum_ph[SPROF_MOE_GATEUP],
                    100.0 * sprof_cum_ph[SPROF_MOE_GATEUP] /
                        (moe_sum > 0 ? moe_sum : 1.0),
                    sprof_cum_ph[SPROF_MOE_DOWN],
                    100.0 * sprof_cum_ph[SPROF_MOE_DOWN] /
                        (moe_sum > 0 ? moe_sum : 1.0));
            fprintf(stderr, " | ssm");
            for (int ph = 0; ph < SSMP_COUNT; ph++)
                fprintf(stderr, " %s=%.3f(%.1f%%)", ssmp_name[ph], ssmp_cum[ph],
                        100.0 * ssmp_cum[ph] / (ssm_sum > 0 ? ssm_sum : 1.0));
            fprintf(stderr, "\n");
        }
    }
    return r;
}

float *pn_qwen35moe_forward_batch_tokens_cuda(pn_qwen35moe_model *m,
                                              const uint32_t *tokens, int n,
                                              pn_qwen35moe_state *s, int pos0,
                                              float *all_logits)
{
    return qw_batch_tokens_entry(m, tokens, n, s, pos0, all_logits, 0);
}

/* ============================================================
 * Phase 9b — prefill chunk pipelining, public entry points
 * ============================================================ */

static inline int qw_model_uses_qwen35moe_state(const pn_model *model);

/* Run ONE intermediate prefill chunk: everything the ordinary batched prefill
 * does except the head phase, and WITHOUT draining die B (see the long comment
 * on qw_prefill_pipeline_on). Advances state->pos by n. Returns 0, or -1 if
 * the batched path could not run — in which case NOTHING was advanced and the
 * caller must fall back to the ordinary chunk call, which is always correct.
 *
 * Single-die placement is a legal no-op: there is no die B to leave in flight,
 * so this just skips a discarded LM head. */
int pn_qwen35moe_prefill_chunk_cuda(pn_model *model, const uint32_t *tokens,
                                    int n, pn_state *state)
{
    if (!model || !state || !tokens || n <= 0)
        return -1;
    if (!qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    if (!pn_qwen35moe_model_gpu(m))
        return -1;
    if (!qw_batch_tokens_entry(m, tokens, n, s, state->pos, NULL, 1))
        return -1;
    state->pos += n;
    return 0;
}

/* Weight walk for pn_qwen35moe_dense_batch_prefill_ok: clears *ok on any
 * present weight whose quant has no batched (_nx) matvec in gpu_matvec_nx. */
static void qw_nx_quant_check(const pn_qwen35moe_weight_ref *ref, void *user)
{
    const pn_qweight *w = ref->w;
    if (!w || !w->data || w->rows <= 0)
        return;
    switch ((int)w->type) {
    case GGUF_QUANT_Q3_K:
    case GGUF_QUANT_Q4_K:
    case GGUF_QUANT_Q5_K:
    case GGUF_QUANT_Q6_K:
    case GGUF_QUANT_Q8_0:
    case GGUF_QUANT_F32:
        return;
    default:
        *(int *)user = 0;
    }
}

int pn_qwen35moe_dense_batch_prefill_ok(const pn_model *model)
{
    if (!qw_model_uses_qwen35moe_state(model))
        return 0;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    if (!m || !pn_qwen35moe_model_gpu(m) ||
        !pn_qwen35moe_model_config(m)->dense_ffn)
        return 0;
    int ok = 1;
    pn_qwen35moe_for_each_weight(m, qw_nx_quant_check, &ok);
    return ok;
}

/* ============================================================
 * Phase 9c — WIDE prefill pipelining: every die busy at once
 *
 * WHAT 9b LEAVES ON THE TABLE. 9b overlaps chunk N+1's FIRST die with chunk N's
 * LAST die, and nothing else: gpu_handoff_export() does a blocking
 * fc_sync_ctx() on the sending die at every boundary, so inside a chunk the
 * dies still run strictly one after another. On a 2-die split that is the whole
 * machine and 9b measured 1.96x of its 1.984x ceiling. On the 8-die split it
 * overlaps 2 of 8 dies: the period falls from 8t to 7t, worth 1.14x. Seven
 * eighths of the dies are still idle at any instant.
 *
 * WHAT THIS DOES. It runs the chunks as a proper software pipeline: at step T,
 * die d computes chunk T-d. Every die works on a DIFFERENT chunk at the same
 * time, so the steady-state period is one die-segment instead of n_die of them
 * and the ceiling is n_die rather than 2.
 *
 * WHY IT IS LEGAL — the same dependency argument 9b made, carried further. Die
 * d's work on chunk c needs (a) die d-1's output for chunk c, which die d-1
 * produced one step earlier, and (b) die d's OWN carried state — its KV rings
 * and its SSM recurrence for layers [lo[d], hi[d]). A layer lives on exactly
 * one die, so (b) is a per-die sequential dependency and die d visits chunks in
 * strictly increasing order, which honours it. There is no dependency from die
 * d's chunk c to die d-1's chunk c+1, and that is the one the pipeline breaks.
 *
 * ORDER OF A STEP, and why each part is where it is:
 *   1. Barrier every die that computed at step T-1. A die's output buffer is
 *      about to be read and its input buffer overwritten.
 *   2. Hand the residual stream forward, from the HIGHEST die down. Descending
 *      is not cosmetic: die d's hidden must be copied into die d+1 before die
 *      d-1's is copied into die d, or a chunk is overwritten before it moves.
 *   3. Die 0 embeds the newly-injected chunk T.
 *   4. Issue every active die's layer segment, with NO sync between them. This
 *      is the only step that costs GPU time, and all n_die of them overlap.
 *
 * BIT-EXACTNESS. Same kernels, same per-die order, same inputs, same bytes over
 * the same links; only the issue timing changes. Nothing reassociates and no
 * arithmetic is skipped. Verified with PN_PREFILL_LOGITS_HASH=1 against the
 * sequential path.
 *
 * WHAT IT COSTS. n_die - 1 boundary copies per chunk, exactly as today - the
 * pipeline does not add hops, it only stops idling between them. The barrier is
 * n_die fc_sync_ctx calls per step against one die-segment of compute.
 *
 * THE REAL CEILING IS POWER, NOT THIS CODE. A K80 is two GK210 dies behind one
 * board budget; driving both drags them off 875 MHz. An 8-die pipeline puts two
 * busy dies on each of the four boards, so the win is n_die x (throttled clock /
 * 875), not n_die. That is a property of the hardware, measured separately.
 * ============================================================ */

static int qw_prefill_pipeline_wide_on(void)
{
    /* DEFAULT ON. A layer split ran the dies in sequence, so prefill used 1.22
     * of 8 dies -- 16% occupancy -- with the host thread spinning in
     * fc_sync_ctx rather than working. Measured on the 8-die BF16 35B: 2.42x at
     * 475 tokens, 5.12x at 1,850, 6.40x at 4,637 (95% of the structural
     * 8n/(n+7) ceiling), taking TTFT at 4,637 tokens from 171 s to 26.7 s.
     * Bit-exact: an FNV-1a hash over the prefill's final logits is identical
     * with the pipeline off and on, at four seeded prompts.
     *
     * It falls off at depth -- 4.58x at 9k, 2.75x at 16.7k -- and the cause is
     * thermal, not algorithmic: die 2 reaches 91 C and the driver clocks it to
     * 478 then 256 MHz while the other seven hold 832-875, and a pipeline's
     * period is its slowest stage. Sequentially that never appears because the
     * neighbours are idle. The software answer is a cost-weighted split (die 2
     * sustains ~0.30 of the others' rate at depth and should hold ~1.5 layers,
     * not 5); until that lands, deep prefill is still faster than sequential,
     * just not 8x.
     *
     * QWEN35_PREFILL_PIPELINE_WIDE=0 restores the sequential path. The wide
     * path already refuses cleanly under QWEN35_PROF, a single die, barswap,
     * QWEN35_HOST_KV_MIRROR=1, or no flash. */
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_PREFILL_PIPELINE_WIDE");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

/* One chunk of the prefill range: where it starts, how many tokens, and the
 * absolute position its first token occupies. */
typedef struct {
    const uint32_t *tokens;
    int             n;
    int             pos0;
} qw_pipe_chunk;

static inline qw_pipe_chunk qw_pipe_chunk_at(const uint32_t *tokens,
                                             int n_total, int pos0,
                                             int chunk, int c)
{
    qw_pipe_chunk k;
    int off = c * chunk;
    k.tokens = tokens + off;
    k.n = n_total - off < chunk ? n_total - off : chunk;
    k.pos0 = pos0 + off;
    return k;
}

/* QWEN35_PIPE_PROF=1: where a pipelined range's wall time goes, without adding a sync —
 * the host's wait on each die at the step barriers (die d's wait is what it ran past the
 * dies drained before it), the host's issue time, and the MTP head feed's host time.
 * QWEN35_PIPE_PROF=2 also drains the last die around the feed, so "mtp feed" is its GPU
 * time (perturbs the overlap; diagnosis only). */
typedef struct {
    int on;
    double t0, issue_ms, mtp_ms;
    double wait_ms[PN_DIE_LAYER_SPLIT_MAX_DIES];
} qw_pipe_prof;

static int qw_pipe_prof_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_PIPE_PROF");
        v = e ? (atoi(e) > 1 ? 2 : 1) : 0;
    }
    return v;
}

/* QWEN35_MTP_PREFILL_FEED=0: the pipeline leaves the MTP head's backlog unfed (profiling
 * only: the head then drafts from an empty prompt history until a verify feeds it). */
static int qw_pipe_mtp_feed_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_MTP_PREFILL_FEED");
        v = !(e && atoi(e) == 0);
    }
    return v;
}

static void qw_pipe_prof_report(const qw_pipe_prof *pp, int n_die, int n_total,
                                int n_chunks)
{
    fprintf(stderr, "[pipe_prof] %d tokens %d chunks: wall %.0f ms, issue %.0f, mtp feed "
                    "%.0f, wait",
            n_total, n_chunks, (prof_now() - pp->t0) * 1e3, pp->issue_ms, pp->mtp_ms);
    for (int d = 0; d < n_die; d++)
        fprintf(stderr, " d%d %.0f", d, pp->wait_ms[d]);
    fprintf(stderr, "\n");
}

/* QWEN35_MTP_PROF=2: the head's cumulative KV-rows split (host embed / upload / kernels). */
static void qw_pipe_prof_report_mtp(const pn_qwen35moe_gpu *g)
{
    if (g->mtp.on && mtp_prof_on() > 1)
        fprintf(stderr, "[pipe_prof] mtp kv rows %ld: embed %.0f ms, upload %.0f, kernels %.0f\n",
                g->mtp.kv_rows, g->mtp.kv_prof_ms[0], g->mtp.kv_prof_ms[1],
                g->mtp.kv_prof_ms[2]);
}

/**
 * Run the whole prefill range as an n_die-stage software pipeline.
 *
 * @param inject_probe  optional; called once per step with the number of chunks
 *                      already injected. Returning non-zero stops NEW chunks
 *                      being injected — the pipeline then drains what is in
 *                      flight so the KV/SSM state still lands on a chunk
 *                      boundary — and the range reports an abort.
 * @param retired_out   receives the number of tokens that fully cleared the
 *                      pipeline (== n_total unless aborted).
 * @return 0 on success, 1 if aborted after draining, -1 on failure.
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
static int gpu_prefill_pipeline_run(pn_qwen35moe_model *m,
                                    const uint32_t *tokens, int n_total,
                                    pn_qwen35moe_state *s, int pos0, int chunk,
                                    int (*inject_probe)(void *), void *ud,
                                    int *retired_out)
{
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const int n_die = g->n_die;
    /* QWEN35_TP: the pipeline's stages are the two rank pairs; stage d's ranks
     * are dies 2d and 2d + 1, and the head die (3) closes the last stage. */
    const int n_stage = g->tp ? 2 : n_die;
    const int c = cfg->n_embd;
    int aborted = 0;

    *retired_out = 0;

    for (int t = 0; t < n_total; t++)
        if ((int)tokens[t] >= cfg->n_vocab)
            return -1;

    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0)
            return -1;
        g->host_ok = 1;
    }
    gpu_host_scratch h = g->host;

    /* Size every per-die scratch and the host staging set for the WIDEST chunk
     * up front. gpu_get_batch_scratch frees and reallocates when it is asked to
     * grow, and doing that mid-pipeline would pull buffers out from under a
     * segment still in flight. After this, every call is a cache hit. */
    if (gpu_get_batch_host(g, cfg, chunk) < 0)
        return -1;
    if (g->tp && tp_exchange_ready(g, cfg, chunk) < 0)
        return -1;
    gpu_batch_scratch *bs[PN_DIE_LAYER_SPLIT_MAX_DIES];
    gpu_die_scratch   *ds[PN_DIE_LAYER_SPLIT_MAX_DIES];
    for (int d = 0; d < n_die; d++) {
        if (gpu_select_die(g, g->die[d]) < 0)
            return -1;
        bs[d] = gpu_get_batch_scratch(g, g->ctx[d], cfg, chunk);
        ds[d] = gpu_get_scratch(g, g->ctx[d], cfg);
        if (!bs[d] || !ds[d])
            return -1;
    }

    const int n_chunks = (n_total + chunk - 1) / chunk;
    int injected = n_chunks;          /* lowered to stop injecting on abort */
    qw_pipe_prof pp = { .on = qw_pipe_prof_on(), .t0 = prof_now() };

    for (int step = 0; step < injected + n_stage - 1; step++) {
        /* 1. Drain step T-1 before touching any buffer it wrote. */
        for (int d = 0; d < n_stage; d++) {
            int prev = step - 1 - d;
            double tw = pp.on ? prof_now() : 0.0;
            if (prev >= 0 && prev < injected) {
                fc_sync_ctx(g->ctx[g->tp ? 2 * d : d]);
                if (g->tp)
                    fc_sync_ctx(g->ctx[2 * d + 1]);
            }
            if (pp.on)
                pp.wait_ms[d] += (prof_now() - tw) * 1e3;
        }
        double t_issue = pp.on ? prof_now() : 0.0;

        if (!aborted && inject_probe && inject_probe(ud)) {
            /* Stop feeding the pipeline, but keep stepping until what is
             * already in flight has cleared the last die: a half-walked chunk
             * would leave this session's layers at different positions. */
            aborted = 1;
            int in_flight = step;
            injected = in_flight < n_chunks ? in_flight : n_chunks;
            if (step >= injected + n_stage - 1)
                break;
        }

        /* 2. Boundary hand-offs, highest die first. */
        for (int d = n_stage - 1; d >= 1; d--) {
            int ck = step - d;
            if (ck < 0 || ck >= injected)
                continue;
            qw_pipe_chunk k = qw_pipe_chunk_at(tokens, n_total, pos0, chunk, ck);
            size_t bytes = (size_t)k.n * c * sizeof(float);
            if (g->tp) {   /* host-staged: stage d-1's rank 0 to both of stage d's ranks */
                fc_buffer_to_cpu_n(bs[2 * d - 2]->hidden, g->bhost[0], bytes);
                fc_buffer_from_cpu_n(bs[2 * d]->hidden, g->bhost[0], bytes);
                fc_buffer_from_cpu_n(bs[2 * d + 1]->hidden, g->bhost[0], bytes);
                continue;
            }
            gpu_handoff_export(g, d - 1, bs[d - 1]->hidden, g->bhost[0], bytes);
            if (gpu_select_die(g, g->die[d]) < 0)
                return -1;
            gpu_handoff_import(g, d - 1, bs[d - 1]->hidden, bs[d]->hidden,
                               g->bhost[0], bytes);
        }

        /* 3. Inject chunk `step` at die 0. */
        if (step < injected) {
            qw_pipe_chunk k = qw_pipe_chunk_at(tokens, n_total, pos0, chunk,
                                               step);
            if (gpu_select_die(g, g->die[0]) < 0)
                return -1;
            for (int t = 0; t < k.n; t++)
                pn_qwen35moe_model_embed_row(m, k.tokens[t],
                                             g->bhost[0] + (size_t)t * c);
            fc_buffer_from_cpu_n(bs[0]->hidden, g->bhost[0],
                                 (size_t)k.n * c * sizeof(float));
            if (g->tp)
                fc_buffer_from_cpu_n(bs[1]->hidden, g->bhost[0],
                                     (size_t)k.n * c * sizeof(float));
        }

        /* 4. Issue every active die's segment. No syncs — this is the overlap. */
        for (int sd = 0; sd < n_stage; sd++) {
            int ck = step - sd;
            if (ck < 0 || ck >= injected)
                continue;
            qw_pipe_chunk k = qw_pipe_chunk_at(tokens, n_total, pos0, chunk, ck);
            /* d: the die whose scratch carries the stage's output (the head die
             * on the last TP stage, rank 1). */
            const int d = g->tp ? (sd == n_stage - 1 ? n_die - 1 : 2 * sd) : sd;
            if (g->tp) {
                gpu_batch_scratch *pair[2] = { bs[2 * sd], bs[2 * sd + 1] };
                for (int layer = g->layer_lo[2 * sd]; layer < g->layer_hi[2 * sd]; layer++)
                    if (tp_layer_forward_batch(m, g, sd, layer, pair, s, k.pos0, k.n, 0) < 0)
                        return -1;
            } else {
                if (gpu_select_die(g, g->die[d]) < 0)
                    return -1;
                for (int layer = g->layer_lo[d]; layer < g->layer_hi[d]; layer++)
                    if (gpu_layer_forward_batch(m, g, g->ctx[d], layer, bs[d], s,
                                                k.pos0, k.n, 0) < 0)
                        return -1;
            }

            /* The final chunk's last die also runs the head, so the caller gets
             * real logits from the same activations the sequential path would
             * have produced. Intermediate chunks' heads are discarded by the
             * serve and are never computed, exactly as in 9b. With the MTP
             * head on, every chunk's output_norm'd rows also join the head's
             * backlog here (stream-ordered on the last die; the serve cannot
             * feed it chunk by chunk when the model owns the chunking). */
            /* After an abort the in-flight chunks still clear every die and
             * count in state->pos, so they still join the MTP backlog; only
             * the head is skipped (nobody reads those logits). */
            const int last_chunk = ck == n_chunks - 1;
            if (sd == n_stage - 1 && ((last_chunk && !aborted) || g->mtp.on)) {
                fc_rms_norm_nx_ctx(g->ctx[d], bs[d]->hidden,
                                   g->output_norm.buf, bs[d]->x_norm, c,
                                   cfg->rms_norm_eps, k.n);
                if (g->mtp.on && qw_pipe_mtp_feed_on()) {
                    if (pp.on > 1)
                        fc_sync_ctx(g->ctx[d]);   /* PROF=2: time the feed alone */
                    double tm = pp.on ? prof_now() : 0.0;
                    mtp_capture_rows(g, g->ctx[d], bs[d]->x_norm, k.n, c);
                    (void)mtp_push(m, g, s, k.pos0 + k.n, k.tokens, k.n);
                    if (pp.on > 1)
                        fc_sync_ctx(g->ctx[d]);
                    if (pp.on)
                        pp.mtp_ms += (prof_now() - tm) * 1e3;
                }
                if (last_chunk && !aborted &&
                    !gpu_batch_head_last_row(g, cfg, g->ctx[d], bs[d], ds[d], h,
                                             s, k.n))
                    return -1;
            }
        }
        if (pp.on)
            pp.issue_ms += (prof_now() - t_issue) * 1e3;
    }

    for (int d = 0; d < n_die; d++) {
        double tw = pp.on ? prof_now() : 0.0;
        fc_sync_ctx(g->ctx[d]);
        if (pp.on)
            pp.wait_ms[d] += (prof_now() - tw) * 1e3;
    }
    if (pp.on) {
        qw_pipe_prof_report(&pp, n_stage, n_total, n_chunks);
        qw_pipe_prof_report_mtp(g);
    }

    /* Every injected chunk has now cleared the last die (the loop ran until the
     * last one was issued there, and the barrier above waited for it), so the
     * session's layers all sit after chunk `injected - 1`. This is what
     * state->pos advances by, and it has to be exact: a client abort keeps the
     * state for the next request. (Counting at the top of each step, as this
     * did before, left the final in-flight chunk out: the layers ran one chunk
     * past state->pos.) */
    *retired_out = aborted ? (injected * chunk < n_total ? injected * chunk
                                                         : n_total)
                           : n_total;
    return aborted ? 1 : 0;
}

/* Drain both dies. Every early exit from a pipelined chunk loop (abort, error,
 * end of range) must call this before anything reads or resets the state an
 * in-flight die-B segment is still writing. Idempotent and cheap when the
 * pipeline is off or already drained. */
void pn_qwen35moe_prefill_drain_cuda(pn_model *model)
{
    if (!model || !qw_model_uses_qwen35moe_state(model))
        return;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    if (!m)
        return;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return;
    for (int i = 0; i < g->n_die; i++)
        if (g->ctx[i])
            fc_sync_ctx(g->ctx[i]);
}

int pn_qwen35moe_prefill_pipeline_wide_enabled(void)
{
    return qw_prefill_pipeline_wide_on();
}

int pn_qwen35moe_prefill_range_pipelined_cuda(pn_model *model,
                                              const uint32_t *tokens,
                                              int n_total, pn_state *state,
                                              int chunk,
                                              int (*should_abort)(void *),
                                              void *ud,
                                              tm_tensor **logits_out)
{
    if (logits_out)
        *logits_out = NULL;
    if (!qw_prefill_pipeline_wide_on())
        return -1;
    if (!model || !state || !tokens || n_total <= 0 || chunk <= 0)
        return -1;
    if (!qw_model_uses_qwen35moe_state(model))
        return -1;

    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return -1;

    /* A one-die placement has no stages to overlap, and barswap holds a single
     * die in its MMIO slot at a time, so no two dies are ever live. */
    if (g->n_die < 2 || g->bs)
        return -1;

    /* A dense-FFN model whose weights lack a batched kernel (Bonsai's ternary
     * Q2_0) would fail mid-pipeline with dies already advanced; refuse first. */
    if (pn_qwen35moe_model_config(m)->dense_ffn &&
        !pn_qwen35moe_dense_batch_prefill_ok(model))
        return -1;

    /* QWEN35_PROF drains the issuing die inside every phase lap, which
     * serialises the pipeline while still printing a full profile — the exact
     * failure that cost the 9b campaign a 1.85x baseline error. Refuse rather
     * than report pipeline-shaped numbers from a serialised run. */
    if (prof_on()) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[qwen35moe] QWEN35_PREFILL_PIPELINE_WIDE ignored: "
                    "QWEN35_PROF=1 drains each die inside every phase lap, so "
                    "the pipeline cannot overlap. Profile the sequential path "
                    "for shares; measure the pipeline unprofiled.\n");
        }
        return -1;
    }

    /* ★ The shared-host-buffer hazard. Several dies are live at once here, so
     * ANY layer path that stages through the shared g->bhost[] set would have
     * two dies writing the same host bytes — which would not fail, it would
     * produce fluent-looking garbage. The default batched layer path is clean:
     * flash attention does its q/gate split and softmax on the die, the lazy KV
     * mirror skips the host store, and the MoE and SSM blocks never leave the
     * die at all. Two configurations reintroduce the hazard, and both are
     * refused rather than raced:
     *   - QWEN35_HOST_KV_MIRROR=1 D2Hs every attention layer's K/V through
     *     bhost[3];
     *   - without flash, gpu_attn_block_batch runs the q/gate split and the
     *     causal softmax on the host through bhost[1..3].
     * Refusing costs nothing: the caller runs the sequential path, which is
     * correct under every configuration. */
    if (pn_qwen35moe_host_kv_mirror_eager() || !flash_on())
        return -1;
    for (int d = 0; d < g->n_die; d++)
        if (!fc_attn_flash_available_ctx(g->ctx[d]))
            return -1;

    /* Keep the pipeline's chunk inside the q4 recent-window ring cap. Above it
     * qw_batch_tokens_entry would silently re-split a chunk into sub-batches,
     * which would break the one-chunk-per-die-per-step invariant the schedule
     * is built on. */
    int cap = qw_batch_row_cap(s);
    if (cap > 0 && chunk > cap)
        chunk = cap;

    int retired = 0;
    int rc = gpu_prefill_pipeline_run(m, tokens, n_total, s, state->pos, chunk,
                                      should_abort, ud, &retired);
    if (rc < 0)
        return -1;                       /* state is undefined; caller resets */
    state->pos += retired;
    if (rc == 1)
        return 1;                        /* aborted after draining */

    /* Same owned [n_vocab] tensor the sequential batched entry hands back, so
     * the serve's chunk loop and this one are interchangeable to the caller. */
    if (logits_out) {
        int64_t shape[1] = { s->n_vocab };
        tm_tensor *out = tm_empty(shape, 1, TM_F32);
        if (!out)
            return -1;
        memcpy(tm_data_f32(out), s->logits,
               (size_t)s->n_vocab * sizeof(float));
        *logits_out = out;
    }
    return 0;
}

/* ============================================================
 * Speculative greedy decoding — batched verify + partial-accept commit
 *
 * verify runs [tokens[0..n)] (the committed pending token + the drafts)
 * through the batched forward at positions pos0..pos0+n-1 with per-position
 * SSM state snapshots, and reports each position's greedy argmax. It does
 * NOT commit anything: the caller scans the argmaxes against the drafts,
 * decides how many positions to accept (n_accept in [1, n]), and calls
 * commit.
 *
 * commit makes the dual state reflect exactly n_accept processed tokens:
 *   - SSM (conv + recurrent) state: restored from snapshot n_accept-1 with
 *     one D2D copy per layer (a no-op when n_accept == n — the in-place
 *     scan already left state-after-n).
 *   - KV cache: nothing to do. Entries the rejected positions wrote sit at
 *     ring slots > pos0+n_accept-1; attention for a position p only reads
 *     slots [0, p], so they are never read and are overwritten when those
 *     positions are re-processed with the corrected tokens. (verify refuses
 *     pos0 + n > max_seq, so the ring never wraps mid-verify.)
 * Every accepted position's argmax is the model's exact greedy token given
 * the same prefix — bit-exact vs plain single-token greedy decode.
 * ============================================================ */

int pn_qwen35moe_spec_verify_tokens_cuda(pn_qwen35moe_model *m,
                                         const uint32_t *tokens, int n,
                                         pn_qwen35moe_state *s, int pos0,
                                         int32_t *argmax_out)
{
    if (!m || !s || !tokens || n <= 0 || !argmax_out)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return -1;
    /* No KV-ring wraparound inside a verify block: rejected slots must stay
     * in the never-read tail of the ring. */
    if (s->n_attn > 0 && pos0 + n > s->kv[0].max_seq)
        return -1;

    g->spec_n = 0;
    g->spec_state = NULL;
    double t0 = prof_on() ? prof_now() : 0.0;
    if (prof_on())
        memset(bprof_call, 0, sizeof(bprof_call));
    if (!gpu_batch_tokens_core(m, tokens, n, s, pos0, NULL, 1, argmax_out, 0))
        return -1;
    if (prof_on())   /* QWEN35_PROF: the verify's split, "BPROF call n=<rows>" */
        bprof_report(n, pos0, prof_now() - t0);
    g->spec_n = n;
    g->spec_state = s;
    g->spec_pos0 = pos0;
    return 0;
}

int pn_qwen35moe_spec_commit_cuda(pn_qwen35moe_model *m, int n_accept)
{
    if (!m)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g || g->spec_n <= 0 || n_accept < 1 || n_accept > g->spec_n)
        return -1;

    int n = g->spec_n;
    pn_qwen35moe_state *s = g->spec_state;
    int pos0 = g->spec_pos0;
    g->spec_n = 0;
    g->spec_state = NULL;

    /* Phase 9.2 — KV ring watermark. The verify appended rows
     * [pos0, pos0+n) to the single-user GPU ring of every attention layer
     * but left the watermark at pos0, because only the caller knows how many
     * drafts landed. The first n_accept of those rows now ARE committed
     * content, so hand them to the watermark; the rejected tail stays above
     * the new frontier, where attention never reads it and the next append
     * overwrites it.
     *
     * This is what makes the next verify free. Without it the watermark
     * trails the frontier and gpu_attn_kv_cache_sync tries to close the gap
     * from the host KV mirror — which under the lazy mirror (Phase 10b/INV3)
     * does not hold those rows, so the verify fails outright, and under the
     * eager mirror costs a full-window H2D. That was the Phase 9.1 defect.
     *
     * Only layers still bound to this state's ring are advanced; anything
     * else has been rebound since the verify and must resync normally. */
    if (s) {
        for (int layer = 0; layer < pn_qwen35moe_model_config(m)->n_layer;
             layer++) {
            if (pn_qwen35moe_model_layer_type(m, layer) !=
                PN_QWEN35MOE_LAYER_ATTN)
                continue;
            int idx = pn_qwen35moe_model_layer_index(m, layer);
            pn_qwen35moe_kv *kv = &s->kv[idx];
            const void *host_id = kv->q4 ? (const void *)kv->kq
                                         : (const void *)kv->k;
            for (int p = 0; p < (g->tp ? 2 : 1); p++) {   /* QWEN35_TP: both halves */
                gpu_attn_layer *da = &g->attn[idx + p * g->tp_n_attn];
                if (da->kv_host_id == host_id && da->kv_gen == kv->content_gen &&
                    da->kv_synced_pos == pos0)
                    da->kv_synced_pos = pos0 + n_accept;
            }
        }
    }

    if (n_accept == n)
        return 0; /* full accept: the in-place scans already hold state-after-n */

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    /* QWEN35_TP: a rank's snapshots are its local state images. */
    const pn_qwen35moe_config *scfg = g->tp ? &g->tp_cfg : cfg;
    size_t conv_bytes = (size_t)(scfg->ssm_conv - 1) * scfg->conv_dim *
                        sizeof(float);
    size_t ssm_bytes = (size_t)scfg->ssm_state * scfg->ssm_state *
                       scfg->ssm_dt_rank * sizeof(float);
    size_t snap = (size_t)(n_accept - 1);

    /* Walk the layers in die order (front then back), syncing each die's
     * copies before leaving it, so barswap rigs swap at most once and never
     * touch a context whose die is out of the slot.  Attention layers ride
     * along to put back the recent-window shadow slots the rejected rows
     * evicted (see qw_spec_win_restore). */
    fc_cuda_ctx *cur = NULL;
    for (int lr = 0; lr < cfg->n_layer * (g->tp ? 2 : 1); lr++) {
        /* QWEN35_TP: every layer twice, rank 0's half then rank 1's. */
        int layer = lr % cfg->n_layer, rank = lr / cfg->n_layer;
        int lidx = pn_qwen35moe_model_layer_index(m, layer);
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_SSM) {
            gpu_attn_layer *da = &g->attn[lidx + rank * g->tp_n_attn];
            if (!s || da->spec_win_n != n)
                continue;
            if (da->ctx != cur) {
                if (cur)
                    fc_sync_ctx(cur);
                if (gpu_select_die(g, gpu_die_of_ctx(g, da->ctx)) < 0)
                    return -1;
                cur = da->ctx;
            }
            qw_spec_win_restore(da, &s->kv[lidx], pos0, n, n_accept);
            continue;
        }
        gpu_ssm_layer *d = &g->ssm[lidx + rank * g->tp_n_ssm];
        if (!d->conv_snaps || !d->ssm_snaps || d->snaps_cap < n)
            return -1;
        if (d->ctx != cur) {
            if (cur)
                fc_sync_ctx(cur);
            if (gpu_select_die(g, gpu_die_of_ctx(g, d->ctx)) < 0)
                return -1;
            cur = d->ctx;
        }
        fc_buffer_copy_d2d_ctx(d->ctx, d->conv_state_g, 0, d->conv_snaps,
                               snap * conv_bytes, conv_bytes);
        fc_buffer_copy_d2d_ctx(d->ctx, d->ssm_state_g, 0, d->ssm_snaps,
                               snap * ssm_bytes, ssm_bytes);
    }
    if (cur)
        fc_sync_ctx(cur);
    return 0;
}

/* ============================================================
 * MULTI-USER DECODE — K users, ONE token each, in one forward pass
 *
 * THE serving-throughput lever: decode is weight-bandwidth-bound, so
 * streaming every weight once for K users' tokens (the fc_matvec_*_nx /
 * grouped-MoE nx kernels, exactly as batched prefill uses them) raises
 * aggregate tok/s toward Kx. The difference from prefill: the K rows are
 * INDEPENDENT sequences, so
 *   - each slot owns its OWN SSM dual state (conv_slots/ssm_slots on the
 *     layer, slot-major) advanced by single-step slot kernels — no scan,
 *     no cross-slot dependency;
 *   - each slot attends to its OWN host KV cache at its OWN position
 *     (per-row RoPE positions via fc_rope_neox_partial_pos);
 *   - MoE routes each slot's token to its own top-8 (the nx kernels
 *     already index experts per row).
 *
 * BIT-EXACT PER USER: every kernel is per-row/per-slot bit-identical to
 * the single-token path's kernel and the host attention glue is the same
 * code per user, so each user's token stream when batched is IDENTICAL
 * to that user decoding alone. Gated by tests/test_users_qwen35 (run the
 * single-user reference under QWEN35_FUSE=0x37 — the fused qknorm+rope
 * contraction differs by ~1 ulp from the standalone kernels this path
 * mirrors, same caveat as the prefill/spec gates).
 *
 * Protocol: prefill each user (any path), then IMMEDIATELY bind it to a
 * slot (pn_qwen35moe_decode_slot_bind_cuda snapshots the live GPU SSM
 * state D2D into the slot — the host mirror is stale after GPU prefill),
 * then step all users together with pn_qwen35moe_decode_slots_tokens_cuda.
 * Fixed K for this increment; continuous batching (join/leave mid-flight,
 * prefill+decode mixing) is the follow-up.
 * ============================================================ */

/* NOTE: the batched decode path fetches each user's logits with plain
 * gpu_fetch_logits — there is no per-slot variant any more.  The sparse
 * bookkeeping hangs off the per-user pn_qwen35moe_state that owns the buffer
 * (gpu_present_topk), so K users stay independently maintained for free, and
 * the prefill fetch and the decode fetch share ONE tracker for the buffer they
 * both write into (they did not, which is what leaked EOS across turns). */

/* Upload one group's positions and row map to die `die_idx`'s int32
 * [pos | rows] buffer (one H2D), and point the group's batch scratch at the
 * row map: b->rows_h = rows (host), b->rows_g = the on-die copy. `rows` are
 * the group's absolute persistent rows (g->step_rows + group offset). Returns
 * the position buffer, or NULL on failure. */
static fc_buffer *gpu_pos_buf_upload(pn_qwen35moe_gpu *g,
                                     gpu_batch_scratch *b, fc_cuda_ctx *ctx,
                                     int die_idx, const int *pos,
                                     const int *rows, int nslots)
{
    const int cap = PN_QWEN35MOE_GPU_MAX_SLOTS;
    if (nslots <= 0 || nslots > cap)
        return NULL;
    if (!g->pos_buf[die_idx]) {
        g->pos_buf[die_idx] =
            fc_buffer_alloc_ctx(ctx, (size_t)2 * cap * sizeof(int32_t));
        if (!g->pos_buf[die_idx])
            return NULL;
        g->rows_view[die_idx] = fc_buffer_offset_view(
            g->pos_buf[die_idx], (size_t)cap * sizeof(int32_t));
        if (!g->rows_view[die_idx]) {
            fc_buffer_free(g->pos_buf[die_idx]);
            g->pos_buf[die_idx] = NULL;
            return NULL;
        }
    }
    int32_t staged[2 * PN_QWEN35MOE_GPU_MAX_SLOTS] = { 0 };
    for (int u = 0; u < nslots; u++) {
        staged[u] = pos[u];
        staged[cap + u] = rows[u];
        b->rows_h[u] = rows[u];
    }
    /* QWEN35_PCIE_POS: through the TP step's per-die staging, async. */
    int rc = g->tp && g->pcie && qw_pcie_stage_flags(g->pcie).pos && die_idx < 4 &&
                     ctx == g->ctx[die_idx]
                 ? qw_pcie_stage_upload_small(g->pcie, die_idx, g->pos_buf[die_idx], staged,
                                              sizeof(staged))
                 : 1;
    if (rc < 0)
        return NULL;
    if (rc > 0)
        fc_buffer_from_cpu_n(g->pos_buf[die_idx], staged, sizeof(staged));
    b->rows_g = g->rows_view[die_idx];
    return g->pos_buf[die_idx];
}

/* Bind user state `st` to decode slot `slot` (of n_slots total): allocate
 * (or re-allocate on capacity change — which invalidates every slot, so
 * bind ALL users after choosing K) the per-layer slot-state buffers, then
 * seed the slot's conv/ssm images. Source: the LIVE GPU state when this
 * user's forward was the model's most recent (d->host_state == st — the
 * bind-right-after-prefill protocol), else the host mirror (valid only for
 * a state that has not been through a GPU forward, e.g. fresh at pos 0). */
/* Rung 2 of the reactive ladder (MemoryAware.md §7): the pressure-capped decode
 * concurrency ceiling. At level OK this is the full compile-time
 * PN_QWEN35MOE_GPU_MAX_SLOTS; TIGHT halves it and CRITICAL floors it to 1.
 * Reads the live pn_mem_state; when memwatch is off (bare Mac) it returns the
 * full ceiling. Exported so the batched-decode scheduler can size its slot
 * count up-front — the bind guard below is the last-line enforcement. */
int pn_qwen35moe_gpu_ladder_max_slots(void)
{
    if (!pn_memwatch_running())
        return PN_QWEN35MOE_GPU_MAX_SLOTS;
    pn_mem_state st;
    pn_memwatch_read(&st);
    pn_ladder_plan lp;
    pn_ladder_plan_from_state(&st, PN_QWEN35MOE_GPU_MAX_SLOTS, 0, false, &lp);
    return lp.max_slots > 0 ? lp.max_slots : 1;
}

int pn_qwen35moe_decode_slot_bind_cuda(pn_qwen35moe_model *m,
                                       pn_qwen35moe_state *s, int pos,
                                       int slot, int n_slots)
{
    if (!m || !s || slot < 0 || n_slots <= slot ||
        n_slots > PN_QWEN35MOE_GPU_MAX_SLOTS)
        return -1;

    /* Rung 2 enforcement: refuse a bind above the current memory-pressure
     * concurrency ceiling (the scheduler should have sized to it via
     * pn_qwen35moe_gpu_ladder_max_slots; admission sheds the excess). At level
     * OK the ceiling is the full max, so this never fires on an unpressured
     * box. */
    int slot_ceiling = pn_qwen35moe_gpu_ladder_max_slots();
    if (n_slots > slot_ceiling) {
        fprintf(stderr, "[ladder] refusing %d decode slots: memory-pressure "
                "ceiling is %d\n", n_slots, slot_ceiling);
        return -1;
    }
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return -1;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    /* QWEN35_TP: the halves' images and rings use the rank's local dims. */
    const pn_qwen35moe_config *dcfg = qw_state_cfg(g, g->tp, cfg);
    size_t conv_bytes = (size_t)(dcfg->ssm_conv - 1) * dcfg->conv_dim *
                        sizeof(float);
    size_t ssm_bytes = (size_t)dcfg->ssm_state * dcfg->ssm_state *
                       dcfg->ssm_dt_rank * sizeof(float);

    if (g->n_slots != n_slots) {
        /* Slot count changed: every previously bound slot is void. */
        memset(g->slot_state, 0, sizeof(g->slot_state));
        g->n_slots = n_slots;
    }
    /* The row already holds THIS state's SSM image at this position (a resume
     * whose live shared buffers were taken by another session's forward):
     * the image is current, keep it rather than re-seeding from host RAM. */
    const unsigned gen = s->n_attn > 0 ? s->kv[0].content_gen : 0;
    const int ssm_in_row = g->slot_state[slot] == (const void *)s &&
                           g->slot_gen[slot] == gen && g->slot_pos[slot] == pos;


    /* Walk the layers in die order (front then back) to sync across the
     * die boundary.  No gpu_select_die / bs_acquire here: the K80 strap-quirk
     * maps all dies simultaneously (each die has its own persistent fc_cuda_ctx)
     * so there is no MMIO slot to acquire.  Every fc_buffer_*_ctx call below
     * already targets d->ctx directly — barswap is never needed on this path.
     * (Mirrors the async forward: "No gpu_select_die needed — async requires
     * small-BAR mode, where every die is live behind its own context.") */
    fc_cuda_ctx *cur = NULL;
    for (int lr = 0; lr < cfg->n_layer * (g->tp ? 2 : 1); lr++) {
        /* QWEN35_TP: every layer twice, rank 0's half then rank 1's. */
        int layer = lr % cfg->n_layer, rank = lr / cfg->n_layer;
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_SSM)
            continue;
        int sidx = pn_qwen35moe_model_layer_index(m, layer);
        gpu_ssm_layer *d = &g->ssm[sidx + rank * g->tp_n_ssm];
        pn_qwen35moe_ssm_state *st = &s->ssm[sidx];
        if (d->ctx != cur) {
            if (cur)
                fc_sync_ctx(cur);   /* flush previous die before switching */
            cur = d->ctx;
        }
        if (d->slots_cap < n_slots) {
            g->tp_gen++;   /* QWEN35_TP slots graphs bake these rows */
            fc_buffer_free(d->conv_slots);
            fc_buffer_free(d->ssm_slots);
            d->conv_slots = fc_buffer_alloc_ctx(d->ctx,
                                                (size_t)n_slots * conv_bytes);
            d->ssm_slots = fc_buffer_alloc_ctx(d->ctx,
                                               (size_t)n_slots * ssm_bytes);
            d->slots_cap = (d->conv_slots && d->ssm_slots) ? n_slots : 0;
            if (!d->slots_cap)
                return -1;
            /* L9 — the slot state buffers just moved: any captured slots
             * graph holds their old addresses. Drop it so the next decode
             * step recaptures against the new allocation. */
            if (d->slots_graph) {
                fc_graph_free(d->slots_graph);
                d->slots_graph = NULL;
            }
            pn_graph_capture_dropped(&d->slots_graph_capture);
            d->slots_graph_n = 0;
        }
        if (ssm_in_row && d->host_state != (const void *)st) {
            continue;   /* the row's image is already this user's, current */
        } else if (d->conv_state_g && d->host_state == (const void *)st) {
            /* Live GPU state belongs to this user — D2D snapshot. */
            fc_buffer_copy_d2d_ctx(d->ctx, d->conv_slots,
                                   (size_t)slot * conv_bytes,
                                   d->conv_state_g, 0, conv_bytes);
            fc_buffer_copy_d2d_ctx(d->ctx, d->ssm_slots,
                                   (size_t)slot * ssm_bytes,
                                   d->ssm_state_g, 0, ssm_bytes);
        } else {
            /* Host mirror (fresh state / CPU-prefilled state only). The fc
             * API has no offset H2D, so bounce through the live state_g
             * buffers (allocating them if this model never decoded). */
            if (!d->conv_state_g)
                d->conv_state_g = fc_buffer_alloc_ctx(d->ctx, conv_bytes);
            if (!d->ssm_state_g)
                d->ssm_state_g = fc_buffer_alloc_ctx(d->ctx, ssm_bytes);
            if (!d->conv_state_g || !d->ssm_state_g)
                return -1;
            qw_ssm_xfer(g, d, st, 1);
            d->host_state = (const void *)st;
            fc_buffer_copy_d2d_ctx(d->ctx, d->conv_slots,
                                   (size_t)slot * conv_bytes,
                                   d->conv_state_g, 0, conv_bytes);
            fc_buffer_copy_d2d_ctx(d->ctx, d->ssm_slots,
                                   (size_t)slot * ssm_bytes,
                                   d->ssm_state_g, 0, ssm_bytes);
        }
    }
    if (cur)
        fc_sync_ctx(cur);

    /* L7 — EAGER mode: invalidate this slot's KV-ring watermark on every
     * attention layer; a bind can put a RESET+re-prefilled state (same
     * pointer, new rows) into the slot, and the pointer alone cannot see
     * that. The host mirror is authoritative at bind time (bs_slot_bind
     * runs state_to_host), so the next flash step re-uploads [0, pos). */
    if (pn_qwen35moe_host_kv_mirror_eager()) {
        for (int i = 0; i < g->n_attn; i++)
            g->attn[i].kv_slot_host_id[slot] = NULL;
    } else {
        /* Phase 10b LAZY mode: hand the incoming state's KV rows to the
         * slot ring NOW, while the single-user ring still holds its
         * prefill — this bind runs under the same GPU turn that produced
         * it. Deferring the seed to the first group step is unsafe under
         * concurrency: another session's prefill reuses the single-user
         * ring in between (measured on a K80 host: a 16-way blast erred out every
         * slot with "no ring source"). A warm rebind of content the slot
         * already holds keeps its rows; anything unseedable falls back to
         * materializing the host mirror so the first step's legacy
         * mirror-upload path works. */
        for (int i = 0; i < g->n_attn; i++) {
            gpu_attn_layer *da = &g->attn[i];
            pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
            const void *id = kv->q4 ? (const void *)kv->kq
                                    : (const void *)kv->k;
            /* A warm rebind keeps the rows the slot already holds, but a
             * resumed turn has just delta-prefilled PAST them into the
             * single-user ring: top those rows up now too. Leaving them for
             * the first step lost them whenever another session's prefill
             * chunk took the single-user ring in between (overlapping turns:
             * the resumed turn erred out after one token). */
            int from = (da->kv_slot_host_id[slot] == id &&
                        da->kv_slot_gen[slot] == kv->content_gen)
                           ? da->kv_slot_synced[slot] : 0;
            int rows = (da->k_cache_g && da->v_cache_g &&
                        da->cache_cap == kv->max_seq &&
                        da->kv_host_id == id &&
                        da->kv_gen == kv->content_gen)
                           ? da->kv_synced_pos : 0;
            if (from > 0 && rows <= from)
                continue;   /* warm rebind: the slot already holds it all */
            int seeded = 0;
            if (rows > 0 &&
                gpu_attn_slot_ring_ensure(da, dcfg, kv->q4, kv->max_seq,
                                          n_slots, qw_attn_hidx(g, i)) == 0 &&
                da->kv_slots_cap_n > slot &&
                da->kv_slots_cap_seq == kv->max_seq) {
                size_t row_b = qw_kv_row_bytes(kv);
                size_t head_b = (size_t)kv->max_seq * row_b;
                const int nh = qw_kv_nh(da, kv);
                for (int hd = 0; hd < nh; hd++) {
                    size_t src_off = (size_t)hd * head_b + (size_t)from * row_b;
                    size_t dst_off = ((size_t)slot * nh + hd) *
                                     head_b + (size_t)from * row_b;
                    fc_buffer_copy_d2d_ctx(da->ctx, da->k_slots_g, dst_off,
                                           da->k_cache_g, src_off,
                                           (size_t)(rows - from) * row_b);
                    fc_buffer_copy_d2d_ctx(da->ctx, da->v_slots_g, dst_off,
                                           da->v_cache_g, src_off,
                                           (size_t)(rows - from) * row_b);
                }
                if (kv->q4 && da->k_sink_g && da->k_sink_slots_g) {
                    size_t sink_b = qw_kv_sink_bytes(da, kv);
                    size_t win_b = qw_kv_win_bytes(da, kv);
                    fc_buffer_copy_d2d_ctx(da->ctx, da->k_sink_slots_g,
                                           (size_t)slot * sink_b,
                                           da->k_sink_g, 0, sink_b);
                    fc_buffer_copy_d2d_ctx(da->ctx, da->v_sink_slots_g,
                                           (size_t)slot * sink_b,
                                           da->v_sink_g, 0, sink_b);
                    fc_buffer_copy_d2d_ctx(da->ctx, da->k_win_slots_g,
                                           (size_t)slot * win_b,
                                           da->k_win_g, 0, win_b);
                    fc_buffer_copy_d2d_ctx(da->ctx, da->v_win_slots_g,
                                           (size_t)slot * win_b,
                                           da->v_win_g, 0, win_b);
                }
                da->kv_slot_host_id[slot] = id;
                da->kv_slot_gen[slot] = kv->content_gen;
                da->kv_slot_synced[slot] = rows;
                da->kv_slot_kv[slot] = kv;
                seeded = 1;
            }
            if (!seeded) {
                /* QWEN35_TP: both halves at once, on rank 0's turn (a half
                 * alone would mark the mirror clean). */
                int both = g->tp ? i < g->tp_n_attn : 1;
                if (both && kv->kv_mirror_stale &&
                    (g->tp ? qw_kv_layer_materialize_all(g, i, kv, -1)
                           : qw_kv_layer_materialize(da, kv, -1)) != 0 &&
                    kv->kv_mirror_stale) {
                    fprintf(stderr, "[attn] slot_bind kv_idx=%d slot=%d: no "
                            "ring source to seed or materialize — first "
                            "step will refuse if rows are needed\n",
                            i, slot);
                }
                da->kv_slot_host_id[slot] = NULL;
                da->kv_slot_kv[slot] = NULL;
            }
        }
        for (int i = 0; i < g->n_die; i++)
            fc_sync_ctx(g->ctx[i]);
    }

    g->slot_state[slot] = (const void *)s;
    g->slot_gen[slot] = s->n_attn > 0 ? s->kv[0].content_gen : 0;
    g->slot_pos[slot] = pos;
    return 0;
}

/* Batched-decode SSM block: identical launch sequence to the prefill
 * gpu_ssm_block_batch except the conv1d and recurrence are SLOT kernels —
 * K independent single-step updates against the layer's slot-major state
 * (no scan; each user advances one position). core_only (QWEN35_TP): stop at
 * s_gated, before the out projection + residual. */
/* QWEN35_SSM_SLOTS_FUSED (default ON; =0 restores the three-launch chain for
 * A/B): the decode slots path runs beta sigmoid + decay gate + recurrence as
 * fc_ssm_gdn_step_slots_ctx — one warp-per-row launch, bit-exact. */
static int ssm_slots_fused_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_SSM_SLOTS_FUSED");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

static int gpu_ssm_block_slots(gpu_ssm_layer *d,
                               const pn_qwen35moe_config *cfg,
                               gpu_batch_scratch *b, int slot_base, int nslots,
                               int core_only)
{
    const int conv_dim = cfg->conv_dim;
    const int S = cfg->ssm_state;
    const int n_k_heads = cfg->ssm_n_group;
    const int n_v_heads = cfg->ssm_dt_rank;
    const int kv_groups = n_v_heads / n_k_heads;
    const int key_dim = S * n_k_heads;
    const int val_dim = S * n_v_heads;
    const float eps = cfg->rms_norm_eps;
    fc_cuda_ctx *ctx = d->ctx;

    /* Each launch position u reads and writes its user's OWN persistent
     * state row b->rows_h[u] (the row map rides on the die as b->rows_g), so
     * the users stepped together need not sit in contiguous rows and nothing
     * is moved when the active set changes. The batch scratch (b->*) stays
     * group-local [0,nslots). */
    (void)slot_base;
    fc_buffer *conv_ring = d->conv_slots;
    fc_buffer *ssm_ring  = d->ssm_slots;
    fc_buffer *rows_g = b->rows_g;

    double sp_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);

    if (gpu_fused_launch_one(ctx, &d->fx_proj_s, d->fx_s_gen, b, nslots,
                             b->x_norm) != 0 &&
        (gpu_matvec_nx(&d->attn_qkv, b->x_norm, b->s_qkv, nslots) < 0 ||
         gpu_matvec_nx(&d->attn_gate, b->x_norm, b->s_z, nslots) < 0 ||
         gpu_matvec_nx(&d->ssm_beta, b->x_norm, b->s_beta, nslots) < 0 ||
         gpu_matvec_nx(&d->ssm_alpha, b->x_norm, b->s_alpha, nslots) < 0))
        return -1;
    sprof_lap(SPROF_DENSE, ctx, &sp_t0);

    /* Sub-span clock is SEPARATE from sp_t0 on purpose: the five laps below
     * partition SPROF_SSM without shortening it, exactly as ssm_lap does for
     * BPROF_SSM on the prefill path. */
    double sub_t0 = 0.0;
    sub_begin(ctx, &sub_t0);

    /* Per-slot single-step conv1d over the slot-major state. */
    fc_ssm_conv1d_silu_slots_ctx(ctx, b->s_qkv, d->ssm_conv1d.buf,
                                 conv_ring, b->s_conv, conv_dim,
                                 cfg->ssm_conv, nslots, rows_g);
    sub_lap(&ssmd_cum[SSMD_CONV], &ssmd_laps[SSMD_CONV], ctx, &sub_t0);
    fc_ssm_qknorm_gqa_nx_ctx(ctx, b->s_conv, b->s_qg, b->s_kg, b->s_vg,
                             n_k_heads, kv_groups, S, key_dim, val_dim,
                             eps, nslots);
    sub_lap(&ssmd_cum[SSMD_QKNORM], &ssmd_laps[SSMD_QKNORM], ctx, &sub_t0);

    /* Beta sigmoid + decay gate + recurrence: one fused warp-per-row launch
     * when available (bit-exact), else the original three. The glue kernels
     * only feed the recurrence, so moving them after conv/qknorm is free. */
    const float recur_scale = qw_scale_inv_sqrt(S);
    if (!ssm_slots_fused_on() ||
        fc_ssm_gdn_step_slots_ctx(ctx, ssm_ring, b->s_qg, b->s_kg, b->s_vg,
                                  b->s_alpha, d->ssm_dt.buf, d->ssm_a.buf,
                                  b->s_beta, b->s_y, n_v_heads, S, nslots,
                                  recur_scale, rows_g) != 0) {
        fc_sigmoid_ctx(ctx, b->s_beta, nslots * n_v_heads);
        fc_ssm_decay_gate_nx_ctx(ctx, b->s_alpha, d->ssm_dt.buf, d->ssm_a.buf,
                                 b->s_g, n_v_heads, nslots);
        sub_lap(&ssmd_cum[SSMD_DECAY], &ssmd_laps[SSMD_DECAY], ctx, &sub_t0);
        fc_ssm_recurrence_slots_ctx(ctx, ssm_ring, b->s_qg, b->s_kg,
                                    b->s_vg, b->s_g, b->s_beta, b->s_y,
                                    n_v_heads, S, nslots, recur_scale,
                                    rows_g);
    }
    sub_lap(&ssmd_cum[SSMD_RECUR], &ssmd_laps[SSMD_RECUR], ctx, &sub_t0);

    fc_ssm_gated_norm_ctx(ctx, b->s_y, d->ssm_norm.buf, b->s_z, b->s_gated,
                          nslots * n_v_heads, S, eps);
    sub_lap(&ssmd_cum[SSMD_GNORM], &ssmd_laps[SSMD_GNORM], ctx, &sub_t0);
    sprof_lap(SPROF_SSM, ctx, &sp_t0);
    if (core_only)      /* QWEN35_TP: s_gated holds the rank's heads */
        return 0;
    if (gpu_fused_launch_one(ctx, &d->fx_out_s, d->fx_s_gen, b, nslots,
                             b->s_gated) != 0) {
        if (gpu_matvec_nx(&d->ssm_out, b->s_gated, b->block_out, nslots) < 0)
            return -1;
        fc_add_ctx(ctx, b->hidden, b->block_out, b->hidden,
                   nslots * cfg->n_embd);
    }
    sprof_lap(SPROF_DENSE, ctx, &sp_t0);
    return 0;
}

/* QWEN35_FLASH_SLOTS gate (default ON; =0 keeps the host-softmax slots path;
 * A/B baseline). Routes the batched-decode attention through the on-die
 * per-slot split flash kernels (L7). */
static int flash_slots_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_FLASH_SLOTS");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Host mirror of the decode split plan (fc_attn_dec_plan / the
 * gpu_attn_block_decode_flash formula): n_splits for one user's seq_len.
 * The slot kernels recompute each user's plan on-device; the host only
 * needs the MAX across users to size the launch. */
static int gpu_attn_slot_plan_splits(int seq_len, int kps)
{
    int n_splits = (seq_len + kps - 1) / kps;
    if (n_splits < 1)
        n_splits = 1;
    if (n_splits > PN_QWEN35MOE_DEC_SPLIT_MAX)
        n_splits = PN_QWEN35MOE_DEC_SPLIT_MAX;
    int split_len = (seq_len + n_splits - 1) / n_splits;
    split_len = (split_len + 15) & ~15;
    return (seq_len + split_len - 1) / split_len;
}

/* Ensure this layer's per-slot KV rings ([nslots][n_head_kv][max_seq][row] +
 * q4 fp16 shadows) and the per-slot decode partial scratch exist for
 * (nslots, max_seq, q4). A geometry change reallocates and invalidates every
 * slot watermark — safe, because the slots path keeps the host mirror
 * current, so the next step re-uploads each user's rows from it. Returns
 * 0 = ready, 1 = unavailable (allocation failed; caller uses the host
 * path). */
static int gpu_attn_slot_ring_ensure(gpu_attn_layer *d,
                                     const pn_qwen35moe_config *cfg,
                                     int q4, int max_seq, int nslots,
                                     int kv_idx)
{
    fc_cuda_ctx *ctx = d->ctx;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    size_t row_b = q4 ? (size_t)pn_kvq4_row_bytes(head_dim)
                      : (size_t)head_dim * sizeof(float);

    if (!d->k_slots_g || d->kv_slots_cap_seq != max_seq ||
        d->kv_slots_cap_n < nslots || d->kv_slots_q4 != q4) {
        size_t sink_b = q4 ? (size_t)nslots * n_head_kv * PN_KVQ4_SINKMAX *
                             head_dim * sizeof(uint16_t) : 0;
        size_t win_b  = q4 ? (size_t)nslots * n_head_kv * PN_KVQ4_WMAX *
                             head_dim * sizeof(uint16_t) : 0;
        size_t bytes = (size_t)nslots * n_head_kv * max_seq * row_b;
        size_t need = 2 * bytes + 2 * sink_b + 2 * win_b;

        /* A geometry STRICTLY SMALLER than the one that failed is a different
         * question, so the recorded verdict does not describe it: clear the
         * budget at once. This is what makes dropping K or ctx_cap recover the
         * on-die path immediately — no backoff to wait out, no reload. */
        int same_or_larger = (d->kv_slots_fail_n && q4 == d->kv_slots_fail_q4 &&
                              nslots >= d->kv_slots_fail_n &&
                              max_seq >= d->kv_slots_fail_seq);
        if (d->kv_slots_fail_n && !same_or_larger) {
            d->kv_slots_fail_seq = d->kv_slots_fail_n = d->kv_slots_fail_q4 = 0;
            pn_attn_ring_geometry_shrank(&d->path);
        }

        /* BOUNDED retry, not a latch (gpu_attn_path_tracker.h). A failure that
         * is merely contention gets 4 more chances on an 8/32/128/512
         * eligible-step backoff; one that cannot fit an empty die stops at
         * once. On a step where the answer is "not now" nothing is freed and
         * nothing is allocated, so we return BEFORE the free below and
         * slots_ring_gen does not move — claiming the ring addresses shifted on
         * a step where they did not is what turned the un-latched retry into a
         * 62% collapse (afcad12): it invalidates every layer's captured graph,
         * every step. */
        if (!pn_attn_ring_should_attempt(&d->path)) {
            pn_attn_path_host(&d->path, PN_ATTN_HOST_RING_ALLOC);
            return 1;
        }
        /* Captured BEFORE the frees below: whether this attempt actually moves
         * any ring address, and so whether slots_ring_gen may bump. */
        const int had_rings = (d->k_slots_g != NULL);
        /* Phase 10b (lazy mirror): the rings about to be freed may hold the
         * ONLY copy of live sessions' KV rows (the mirror is not maintained
         * per token). Rescue every bound slot's rows into its host mirror
         * before the free — bounded by the rows actually decoded, and rare
         * (a geometry change, not a per-step event). */
        if (qw_kv_mirror_lazy() && d->k_slots_g) {
            for (int su = 0; su < d->kv_slots_cap_n; su++) {
                pn_qwen35moe_kv *okv = d->kv_slot_kv[su];
                if (!okv || !d->kv_slot_host_id[su] || !okv->kv_mirror_stale)
                    continue;
                const void *oid = okv->q4 ? (const void *)okv->kq
                                          : (const void *)okv->k;
                if (oid != d->kv_slot_host_id[su] ||
                    okv->content_gen != d->kv_slot_gen[su] ||
                    okv->max_seq != d->kv_slots_cap_seq)
                    continue;   /* bookkeeping no longer describes this kv */
                qw_kv_layer_materialize(d, okv, d->kv_slot_synced[su]);
            }
        }
        fc_buffer_free(d->k_slots_g);
        fc_buffer_free(d->v_slots_g);
        fc_buffer_free(d->k_sink_slots_g);
        fc_buffer_free(d->v_sink_slots_g);
        fc_buffer_free(d->k_win_slots_g);
        fc_buffer_free(d->v_win_slots_g);
        d->k_slots_g = d->v_slots_g = NULL;
        d->k_sink_slots_g = d->v_sink_slots_g = NULL;
        d->k_win_slots_g = d->v_win_slots_g = NULL;
        memset(d->kv_slot_host_id, 0, sizeof(d->kv_slot_host_id));
        memset(d->kv_slot_gen, 0, sizeof(d->kv_slot_gen));
        memset(d->kv_slot_kv, 0, sizeof(d->kv_slot_kv));

        d->k_slots_g = fc_buffer_alloc_ctx(ctx, bytes);
        d->v_slots_g = fc_buffer_alloc_ctx(ctx, bytes);
        int ok = (d->k_slots_g && d->v_slots_g);
        if (ok && q4) {
            d->k_sink_slots_g = fc_buffer_alloc_ctx(ctx, sink_b);
            d->v_sink_slots_g = fc_buffer_alloc_ctx(ctx, sink_b);
            d->k_win_slots_g = fc_buffer_alloc_ctx(ctx, win_b);
            d->v_win_slots_g = fc_buffer_alloc_ctx(ctx, win_b);
            ok = (d->k_sink_slots_g && d->v_sink_slots_g &&
                  d->k_win_slots_g && d->v_win_slots_g);
        }
        /* Test-only fault injection: hand back the memory and report failure,
         * so the retry path is exercisable on real hardware without having to
         * genuinely exhaust a die. Default off, one cached compare. */
        if (ok && pn_attn_ring_inject_now(&d->path)) {
            fc_buffer_free(d->k_slots_g);
            fc_buffer_free(d->v_slots_g);
            fc_buffer_free(d->k_sink_slots_g);
            fc_buffer_free(d->v_sink_slots_g);
            fc_buffer_free(d->k_win_slots_g);
            fc_buffer_free(d->v_win_slots_g);
            d->k_slots_g = d->v_slots_g = NULL;
            d->k_sink_slots_g = d->v_sink_slots_g = NULL;
            d->k_win_slots_g = d->v_win_slots_g = NULL;
            ok = 0;
        }
        d->kv_slots_cap_seq = ok ? max_seq : 0;
        d->kv_slots_cap_n = ok ? nslots : 0;
        d->kv_slots_q4 = q4;
        /* Only claim the addresses moved when they actually did. A retry that
         * found NULL rings and failed again moved nothing, and bumping the
         * generation anyway would drop every layer's captured graph. */
        if (ok || had_rings)
            d->slots_ring_gen++;
        /* QWEN35_SDIAG: report what the ring allocation ASKED for and whether
         * it landed. This branch only runs on a geometry change, so it is at
         * most one line per attention layer per bind — the direct evidence for
         * "do the flash slot rings fit at this ctx_cap". */
        if (sdiag_on())
            fprintf(stderr, "[sdiag] ring_ensure kv_idx=%d nslots=%d "
                    "max_seq=%d q4=%d row_b=%zu k+v=%.1f MiB/slot "
                    "total=%.1f MiB -> %s\n",
                    kv_idx, nslots, max_seq, q4, row_b,
                    2.0 * (double)(bytes / nslots) / (1024.0 * 1024.0),
                    2.0 * (double)bytes / (1024.0 * 1024.0),
                    ok ? "OK" : "ALLOC FAILED (host softmax fallback)");
        if (!ok) {
            /* Transient or impossible? ASK THE DIE rather than guessing. A
             * request that exceeds this die's whole usable VRAM would fail on
             * an EMPTY die, so there is nothing to wait for and retrying it
             * four times is four rounds of pointless multi-hundred-MiB
             * cuMemAllocs. Anything smaller is contention — another model
             * loading, a sibling instance, a bigger geometry not yet released —
             * and contention is exactly what a bounded retry is for. */
            size_t die_free = 0, die_total = 0;
            int impossible = 0;
            if (pn_attn_ring_inject_now(&d->path)) {
                /* Injected fault: the die is fine, so there is nothing to
                 * measure and the log line below honestly reports 0 / 0 MiB.
                 * Test-only path — QWEN35_RING_ALLOC_FAIL_INJECT is unset in
                 * production, where the real reading is always taken. */
                impossible = pn_attn_ring_inject_impossible();
            } else if (fc_mem_info_ctx(ctx, &die_free, &die_total) == 0 &&
                       die_total > 0) {
                uint64_t reserve = pn_die_reserve_bytes();
                uint64_t usable = (uint64_t)die_total > reserve
                                      ? (uint64_t)die_total - reserve : 0;
                impossible = ((uint64_t)need > usable);
            }
            d->kv_slots_fail_seq = max_seq;
            d->kv_slots_fail_n = nslots;
            d->kv_slots_fail_q4 = q4;
            pn_attn_ring_alloc_failed(&d->path, impossible);
            pn_attn_path_host(&d->path, PN_ATTN_HOST_RING_ALLOC);
            /* Say so ONCE per failure, unconditionally — silence is what let
             * the 62% collapse hide behind healthy-looking throughput logs.
             * The retry budget bounds this to PN_ATTN_RING_MAX_FAILURES lines
             * per layer, and naming the verdict makes a transient squeeze
             * visibly different from a geometry that can never fit. */
            fprintf(stderr, "[attn] kv_idx=%d: on-die KV ring allocation "
                    "FAILED (%d slots x %d ctx = %.0f MiB, die free %.0f / "
                    "total %.0f MiB); this layer falls back to HOST softmax. "
                    "%s (attempt %d/%d). Lower ctx_cap or max_sessions to keep "
                    "attention on-die.\n",
                    kv_idx, nslots, max_seq,
                    (double)need / (1024.0 * 1024.0),
                    (double)die_free / (1024.0 * 1024.0),
                    (double)die_total / (1024.0 * 1024.0),
                    impossible ? "IMPOSSIBLE on this die — not retrying"
                               : (d->path.alloc_gave_up
                                      ? "retry budget spent — not retrying"
                                      : "will retry on a backoff"),
                    (int)d->path.alloc_failures, PN_ATTN_RING_MAX_FAILURES);
            return 1;   /* OOM: the host path still works */
        }
        /* Allocation succeeded at this geometry — clear the failure record. */
        d->kv_slots_fail_seq = d->kv_slots_fail_n = d->kv_slots_fail_q4 = 0;
    }

    if (!d->dec_slots_acc_g || d->dec_slots_cap < nslots) {
        /* Same bounded-retry rule as the rings above, sharing this layer's
         * budget: both allocations have to land for the layer to be on-die, so
         * one budget describes the layer. A smaller slot count is a different
         * question and resets it. */
        if (d->dec_slots_fail_n && nslots < d->dec_slots_fail_n) {
            d->dec_slots_fail_n = 0;
            pn_attn_ring_geometry_shrank(&d->path);
        }
        if (!pn_attn_ring_should_attempt(&d->path)) {
            pn_attn_path_host(&d->path, PN_ATTN_HOST_DEC_SCRATCH);
            return 1;
        }
        const int had_dec = (d->dec_slots_acc_g != NULL);
        fc_buffer_free(d->dec_slots_m_g);
        fc_buffer_free(d->dec_slots_l_g);
        fc_buffer_free(d->dec_slots_acc_g);
        size_t np = (size_t)nslots * cfg->n_head * PN_QWEN35MOE_DEC_SPLIT_MAX;
        d->dec_slots_m_g = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        d->dec_slots_l_g = fc_buffer_alloc_ctx(ctx, np * sizeof(float));
        d->dec_slots_acc_g = fc_buffer_alloc_ctx(ctx,
                                                 np * (size_t)head_dim *
                                                 sizeof(float));
        d->dec_slots_cap = (d->dec_slots_m_g && d->dec_slots_l_g &&
                            d->dec_slots_acc_g) ? nslots : 0;
        /* Same rule as the rings: an attempt that moved nothing must not claim
         * the scratch address changed. */
        if (d->dec_slots_cap || had_dec)
            d->slots_ring_gen++;
        if (!d->dec_slots_cap) {
            size_t dec_need = np * (2 * sizeof(float) +
                                    (size_t)head_dim * sizeof(float));
            size_t die_free = 0, die_total = 0;
            int impossible = 0;
            if (fc_mem_info_ctx(ctx, &die_free, &die_total) == 0 &&
                die_total > 0) {
                uint64_t reserve = pn_die_reserve_bytes();
                uint64_t usable = (uint64_t)die_total > reserve
                                      ? (uint64_t)die_total - reserve : 0;
                impossible = ((uint64_t)dec_need > usable);
            }
            d->dec_slots_fail_n = nslots;
            pn_attn_ring_alloc_failed(&d->path, impossible);
            pn_attn_path_host(&d->path, PN_ATTN_HOST_DEC_SCRATCH);
            fprintf(stderr, "[attn] kv_idx=%d: decode partial-scratch "
                    "allocation FAILED at %d slots (%.0f MiB, die free %.0f "
                    "MiB); this layer falls back to HOST softmax. %s "
                    "(attempt %d/%d).\n", kv_idx, nslots,
                    (double)dec_need / (1024.0 * 1024.0),
                    (double)die_free / (1024.0 * 1024.0),
                    impossible ? "IMPOSSIBLE on this die — not retrying"
                               : (d->path.alloc_gave_up
                                      ? "retry budget spent — not retrying"
                                      : "will retry on a backoff"),
                    (int)d->path.alloc_failures, PN_ATTN_RING_MAX_FAILURES);
            if (sdiag_on())
                fprintf(stderr, "[sdiag] ring_ensure kv_idx=%d nslots=%d "
                        "decode partial scratch ALLOC FAILED\n",
                        kv_idx, nslots);
            return 1;
        }
        d->dec_slots_fail_n = 0;
    }
    /* Both allocations are in place at this geometry: the layer is provably
     * fittable, so a LATER failure starts with a fresh budget. */
    pn_attn_ring_alloc_ok(&d->path);
    return 0;
}

/* QWEN35_ASYNC_KVMIRROR gate (default OFF). When on, gpu_attn_block_slots_flash
 * step 6 defers its host-KV-mirror store instead of blocking per attention
 * layer (see qw_kv_defer). Process-constant. */
static int async_kvmirror_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_ASYNC_KVMIRROR");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* Map a ctx to its per-die deferred-mirror slot. An unrecognised context
 * falls back to slot 0 the way the old two-way expression did. */
static int qw_defer_idx(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx)
{
    int i = gpu_die_index_of_ctx(g, ctx);
    return i >= 0 ? i : 0;
}

/* Queue one attention layer's host-KV-mirror store for deferred replay. Stages
 * the roped K/V (already async-D2H'd into the die arena at stage_off by the
 * caller) plus all the metadata the store loop needs. Grows the descriptor
 * array as needed. Returns the descriptor, or NULL on OOM. */
static qw_kv_store *qw_defer_push(qw_kv_defer *df)
{
    if (df->n >= df->n_cap) {
        int nc = df->n_cap ? df->n_cap * 2 : 64;
        qw_kv_store *ns = realloc(df->stores, (size_t)nc * sizeof(*ns));
        if (!ns)
            return NULL;
        df->stores = ns;
        df->n_cap = nc;
    }
    return &df->stores[df->n++];
}

/* Pre-size a die's staging arena for a whole decode step and reset its
 * watermarks. Called at step start for each active die so the arena is NEVER
 * realloc'd mid-step (an in-flight async DtoH writes into it — moving it would
 * corrupt earlier layers' queued copies). `n_attn` attention layers, each
 * staging 2*nslots*kv_dim floats. Returns 0 on success, -1 on OOM. */
static int qw_defer_reset(qw_kv_defer *df, int n_attn, int nslots, int kv_dim)
{
    df->n = 0;
    df->stage_used = 0;
    size_t need = (size_t)n_attn * 2 * nslots * (size_t)kv_dim * sizeof(float);
    if (need > df->stage_bytes) {
        float *ns = realloc(df->stage, need);
        if (!ns)
            return -1;
        df->stage = ns;
        df->stage_bytes = need;
    }
    return 0;
}

/* Hand out `bytes` of pre-sized per-die host staging arena for one layer's
 * K||V. The arena is reserved up front by qw_defer_reset, so this never
 * reallocs (which would invalidate in-flight async DtoH destinations). Returns
 * the byte offset, or (size_t)-1 if the pre-sized arena is exhausted. */
static size_t qw_defer_stage(qw_kv_defer *df, size_t bytes)
{
    if (df->stage_used + bytes > df->stage_bytes)
        return (size_t)-1;   /* pre-sizing was insufficient — caller falls back */
    size_t off = df->stage_used;
    df->stage_used += bytes;
    return off;
}

/* Flush the deferred host-KV-mirror stores for one die: ONE fc_sync_ctx drains
 * all the async D2H copies queued this step, then replay every pending store
 * into the host mirror (byte-identical to the per-layer store loop). Clears the
 * pending list + arena watermark for the next step. */
static void qw_defer_flush(pn_qwen35moe_gpu *g, int idx, fc_cuda_ctx *ctx)
{
    qw_kv_defer *df = &g->defer[idx];
    if (df->n == 0) {
        df->stage_used = 0;
        return;
    }
    fc_sync_ctx(ctx);   /* drain every async DtoH queued this step */
    for (int s = 0; s < df->n; s++) {
        qw_kv_store *st = &df->stores[s];
        const float *hb_kv = df->stage + st->stage_off / sizeof(float);
        for (int u = 0; u < st->nslots; u++) {
            pn_qwen35moe_kv *kv = st->kv[u];
            int cache_pos = st->pos[u] % st->max_seq;
            const float *kvec = hb_kv + (size_t)u * st->kv_dim;
            const float *vvec = hb_kv + (size_t)(st->nslots + u) * st->kv_dim;
            for (int hd = 0; hd < st->n_head_kv; hd++) {
                if (st->kv_q4) {
                    pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd,
                                      st->max_seq, cache_pos,
                                      kvec + (size_t)hd * st->head_dim,
                                      st->head_dim, st->kv_sink, st->kv_win);
                    pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd,
                                      st->max_seq, cache_pos,
                                      vvec + (size_t)hd * st->head_dim,
                                      st->head_dim, st->kv_sink, st->kv_win);
                } else {
                    memcpy(kv->k + ((size_t)hd * st->max_seq + cache_pos) *
                                   st->head_dim,
                           kvec + (size_t)hd * st->head_dim,
                           (size_t)st->head_dim * sizeof(float));
                    memcpy(kv->v + ((size_t)hd * st->max_seq + cache_pos) *
                                   st->head_dim,
                           vvec + (size_t)hd * st->head_dim,
                           (size_t)st->head_dim * sizeof(float));
                }
            }
        }
    }
    df->n = 0;
    df->stage_used = 0;
}

/* Free the per-die deferred-mirror arenas (teardown). */
static void qw_defer_free(pn_qwen35moe_gpu *g)
{
    for (int i = 0; i < PN_DIE_LAYER_SPLIT_MAX_DIES; i++) {
        free(g->defer[i].stores);
        free(g->defer[i].stage);
        memset(&g->defer[i], 0, sizeof(g->defer[i]));
    }
}

/* Batched-decode attention block, ON-DIE flash path (L7): one launch runs
 * every user's decode attention against its own GPU-resident KV ring —
 * replacing the host per-user softmax + its 2 syncs + PCIe round-trips per
 * layer (the K=8 steady-state ceiling; multi-tenant-speed-opt.md §3.2).
 * Each user's split plan is recomputed on-device from its own position, so
 * per-user outputs are bit-identical to the single-token decode-flash path
 * at the same position (the test_users_qwen35 gate).
 *
 * The host KV mirror stays CURRENT (step 6): the roped K/V rows are D2H'd
 * and stored per user exactly as the host path does — ~32 KB/layer, vs the
 * O(K*seq) softmax + ~0.5 MB it replaces — so bind (state_to_host),
 * slot-move, host-path fallback and session preserve all keep today's
 * semantics with no B2-style deferral bookkeeping.
 *
 * Returns 0 = ran, 1 = unavailable (caller uses the host path), <0 = error. */
/* L10 — host-computed plan for ONE attention layer's on-die slots flash path.
 *
 * Everything the device launch sequence needs that only the host can work out
 * (ring buffers + row map, cache geometry, the flash grid width) is resolved up
 * front into this struct. That is what leaves the launch sequence itself pure
 * device work — and therefore CUDA-graph capturable. It owns nothing. */
typedef struct {
    fc_buffer *k_ring, *v_ring;
    fc_buffer *k_sink, *v_sink, *k_win, *v_win;
    fc_buffer *rows_g;          /* borrowed row map (b->rows_g) */
    int   kv_q4, max_seq;
    int   sink_n, win_n;        /* q4 sink / recent-window row counts */
    int   kv_dim;
    int   launch_splits, kps;   /* flash grid width + keys per split */
    float scale;
} qw_attn_slots_plan;

/* Phase 1 of the slots attention block: pure HOST work — the availability
 * checks, the ring allocation, the row-map bounds check, the [synced, pos)
 * re-sync H2D from the host mirror (a no-op in steady-state decode), and the
 * flash grid width. No device compute is issued, so a refusal leaves nothing
 * half-done and the caller can take the host-softmax path.
 *
 * Returns 0 = ready (plan filled), 1 = flash unavailable (use host softmax,
 * plan untouched), <0 = hard error. */
static int attn_slots_flash_prep(gpu_attn_layer *d,
                                 const pn_qwen35moe_config *cfg, int kv_idx,
                                 pn_qwen35moe_state *const *sts,
                                 const int *pos, fc_buffer *pos_g,
                                 gpu_batch_scratch *b, int slot_base,
                                 int nslots, int total_slots,
                                 qw_attn_slots_plan *pl)
{
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    fc_cuda_ctx *ctx = d->ctx;

    memset(pl, 0, sizeof(*pl));

    /* Availability: flag on, kernels present, uniform KV geometry across
     * the slots, no ring wrap for any user. Env gates are process-constant,
     * so flash/host never alternate per step (only a ring wrap could mix
     * them — and the mirror-current design keeps that correct too). */
    /* QWEN35_SDIAG: name the guard that rejects the on-die path. Latches per
     * layer (first decision only), so a steady-state step logs nothing and the
     * default (env unset) path is a single predictable branch. */
    /* Every refusal also RECORDS itself on the layer's path tracker, which is
     * what /v1/models and /admin/models publish. The sdiag line is a debug
     * aid that latches after the first decision; the tracker is the live gauge
     * and is written every step, because "which path is this layer on RIGHT
     * NOW" is the question an operator staring at 1.22 tok/s is asking. One
     * byte store on a path that is already refusing — no hot-path cost. */
#define FLASH_NA(why, reason) do { \
        pn_attn_path_host(&d->path, (why)); \
        if (sdiag_on() && !d->sdiag_flash_logged) { \
            d->sdiag_flash_logged = 1; \
            fprintf(stderr, "[sdiag] flash_slots kv_idx=%d nslots=%d " \
                    "UNAVAILABLE: %s -> HOST softmax\n", \
                    kv_idx, nslots, (reason)); \
        } \
        return 1; \
    } while (0)

    if (!flash_slots_on())
        FLASH_NA(PN_ATTN_HOST_FLASH_OFF, "QWEN35_FLASH_SLOTS=0");
    if (!fc_attn_flash_available_ctx(ctx))
        FLASH_NA(PN_ATTN_HOST_NO_KERNELS, "no flash kernels on this ctx");
    if (!b->a_z || !pos_g)
        FLASH_NA(PN_ATTN_HOST_NO_SCRATCH, "batch scratch a_z/pos_g missing");
    const int kv_q4 = sts[0]->kv[kv_idx].q4;
    const int max_seq = sts[0]->kv[kv_idx].max_seq;
    if (kv_q4 ? !fc_attn_flash_decode_slots_q4_available_ctx(ctx)
              : !fc_attn_flash_decode_slots_available_ctx(ctx))
        FLASH_NA(PN_ATTN_HOST_NO_SLOT_KERNEL,
                 kv_q4 ? "no q4 slot-decode kernel"
                       : "no fp32 slot-decode kernel");
    for (int u = 0; u < nslots; u++) {
        const pn_qwen35moe_kv *kv = &sts[u]->kv[kv_idx];
        if (kv->q4 != kv_q4 || kv->max_seq != max_seq ||
            pos[u] + 1 > max_seq)
            FLASH_NA(PN_ATTN_HOST_RING_WRAP,
                     "non-uniform KV geometry / ring wrap across slots");
    }
    int rr = gpu_attn_slot_ring_ensure(d, cfg, kv_q4, max_seq, total_slots,
                                       kv_idx);
    if (rr > 0) {
        /* ring_ensure already recorded the precise reason (ring vs decode
         * scratch) and the retry verdict; do not overwrite it. */
        if (sdiag_on() && !d->sdiag_flash_logged) {
            d->sdiag_flash_logged = 1;
            fprintf(stderr, "[sdiag] flash_slots kv_idx=%d nslots=%d "
                    "UNAVAILABLE: slot KV ring allocation failed -> HOST "
                    "softmax\n", kv_idx, nslots);
        }
        return 1;
    }
    if (rr != 0)
        return rr;
    pn_attn_path_ondie(&d->path);
    if (sdiag_on() && !d->sdiag_flash_logged) {
        d->sdiag_flash_logged = 1;
        fprintf(stderr, "[sdiag] flash_slots kv_idx=%d nslots=%d max_seq=%d "
                "q4=%d ENGAGED (on-die attention)\n",
                kv_idx, nslots, max_seq, kv_q4);
    }
#undef FLASH_NA

    const int kv_sink = kv_q4 ? pn_kvq4_sink() : 0;
    const int kv_win = kv_q4 ? pn_kvq4_win() : 0;
    size_t row_b = kv_q4 ? (size_t)pn_kvq4_row_bytes(head_dim)
                         : (size_t)head_dim * sizeof(float);

    /* Users decode in their OWN persistent ring rows b->rows_h[u] (row map
     * on the die: b->rows_g), so the active set need not be contiguous and a
     * session's rows never move. The batch scratch (b->a_k etc.) stays
     * group-local [0,nslots). */
    (void)slot_base;
    const size_t sink_row_stride  = (size_t)n_head_kv * PN_KVQ4_SINKMAX *
                                    head_dim * sizeof(uint16_t);
    const size_t win_row_stride   = (size_t)n_head_kv * PN_KVQ4_WMAX *
                                    head_dim * sizeof(uint16_t);
    fc_buffer *k_ring = d->k_slots_g, *v_ring = d->v_slots_g;
    fc_buffer *k_sink = d->k_sink_slots_g, *v_sink = d->v_sink_slots_g;
    fc_buffer *k_win = d->k_win_slots_g, *v_win = d->v_win_slots_g;
    for (int u = 0; u < nslots; u++)
        if (b->rows_h[u] < 0 || b->rows_h[u] >= d->kv_slots_cap_n) {
            fprintf(stderr, "[attn] slots_prep kv_idx=%d: row %d outside the "
                    "%d carved ring rows\n", kv_idx, b->rows_h[u],
                    d->kv_slots_cap_n);
            return -1;
        }

    /* Sync each slot's ring rows [synced, pos[u]) — a no-op in steady-state
     * decode (watermark == pos); a full [0, pos) seed on the first step
     * after a bind / slot move / realloc. Watermarks are valid only when
     * both the buffer pointer AND the content generation match (Phase 10b).
     * Seed source preference: (1) the single-user ring right after prefill
     * (the bind hand-off) or (2) another slot holding this content (slot
     * move) — both on-die D2D; (3) the host mirror, only when it is
     * authoritative (eager mode, or a snapshot-restored session). */
    for (int u = 0; u < nslots; u++) {
        int su = b->rows_h[u];    /* this user's persistent ring row */
        const pn_qwen35moe_kv *kv = &sts[u]->kv[kv_idx];
        const void *host_id = kv_q4 ? (const void *)kv->kq
                                    : (const void *)kv->k;
        int synced = (d->kv_slot_host_id[su] == host_id &&
                      d->kv_slot_gen[su] == kv->content_gen)
                         ? d->kv_slot_synced[su] : 0;
        if (synced > pos[u])
            synced = pos[u];
        if (synced < pos[u]) {
            int seeded = 0;
            size_t seed_bytes = (size_t)(pos[u] - synced) * row_b;
            if (d->k_cache_g && d->v_cache_g && d->cache_cap == max_seq &&
                d->kv_host_id == host_id &&
                d->kv_gen == kv->content_gen &&
                d->kv_synced_pos >= pos[u]) {
                /* (1) D2D from the single-user ring (post-prefill bind). */
                for (int hd = 0; hd < n_head_kv; hd++) {
                    size_t src_off = ((size_t)hd * max_seq + synced) * row_b;
                    size_t dst_off = (((size_t)su * n_head_kv + hd) *
                                      max_seq + synced) * row_b;
                    fc_buffer_copy_d2d_ctx(ctx, d->k_slots_g, dst_off,
                                           d->k_cache_g, src_off,
                                           seed_bytes);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_slots_g, dst_off,
                                           d->v_cache_g, src_off,
                                           seed_bytes);
                }
                if (kv_q4 && d->k_sink_g && d->k_sink_slots_g) {
                    fc_buffer_copy_d2d_ctx(ctx, d->k_sink_slots_g,
                                           (size_t)su * sink_row_stride,
                                           d->k_sink_g, 0, sink_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_sink_slots_g,
                                           (size_t)su * sink_row_stride,
                                           d->v_sink_g, 0, sink_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->k_win_slots_g,
                                           (size_t)su * win_row_stride,
                                           d->k_win_g, 0, win_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_win_slots_g,
                                           (size_t)su * win_row_stride,
                                           d->v_win_g, 0, win_row_stride);
                }
                seeded = 1;
            }
            for (int s2 = 0; !seeded && s2 < d->kv_slots_cap_n; s2++) {
                /* (2) D2D from another slot that holds this content. */
                if (s2 == su || d->kv_slot_host_id[s2] != host_id ||
                    d->kv_slot_gen[s2] != kv->content_gen ||
                    d->kv_slot_synced[s2] < pos[u])
                    continue;
                for (int hd = 0; hd < n_head_kv; hd++) {
                    size_t src_off = (((size_t)s2 * n_head_kv + hd) *
                                      max_seq + synced) * row_b;
                    size_t dst_off = (((size_t)su * n_head_kv + hd) *
                                      max_seq + synced) * row_b;
                    fc_buffer_copy_d2d_ctx(ctx, d->k_slots_g, dst_off,
                                           d->k_slots_g, src_off,
                                           seed_bytes);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_slots_g, dst_off,
                                           d->v_slots_g, src_off,
                                           seed_bytes);
                }
                if (kv_q4 && d->k_sink_slots_g) {
                    fc_buffer_copy_d2d_ctx(ctx, d->k_sink_slots_g,
                                           (size_t)su * sink_row_stride,
                                           d->k_sink_slots_g,
                                           (size_t)s2 * sink_row_stride,
                                           sink_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_sink_slots_g,
                                           (size_t)su * sink_row_stride,
                                           d->v_sink_slots_g,
                                           (size_t)s2 * sink_row_stride,
                                           sink_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->k_win_slots_g,
                                           (size_t)su * win_row_stride,
                                           d->k_win_slots_g,
                                           (size_t)s2 * win_row_stride,
                                           win_row_stride);
                    fc_buffer_copy_d2d_ctx(ctx, d->v_win_slots_g,
                                           (size_t)su * win_row_stride,
                                           d->v_win_slots_g,
                                           (size_t)s2 * win_row_stride,
                                           win_row_stride);
                }
                seeded = 1;
            }
            if (!seeded && qw_kv_mirror_lazy() && kv->kv_mirror_stale) {
                fprintf(stderr, "[attn] slots_prep kv_idx=%d slot=%d: rows "
                        "[%d,%d) have no ring source and the host mirror is "
                        "stale — refusing stale attention\n",
                        kv_idx, su, synced, pos[u]);
                return -1;
            }
            if (!seeded) {
                /* (3) Host mirror upload (authoritative here). QWEN35_TP:
                 * this half's heads start at host head h0. */
                const int h0 = qw_kv_h0(d, kv);
                const unsigned char *ksrc = kv_q4
                    ? kv->kq : (const unsigned char *)kv->k;
                const unsigned char *vsrc = kv_q4
                    ? kv->vq : (const unsigned char *)kv->v;
                for (int hd = 0; hd < n_head_kv; hd++) {
                    size_t src_off = ((size_t)(h0 + hd) * max_seq + synced) * row_b;
                    size_t dst_off = (((size_t)su * n_head_kv + hd) * max_seq +
                                      synced) * row_b;
                    fc_buffer_from_cpu_at(k_ring, dst_off, ksrc + src_off,
                                          seed_bytes);
                    fc_buffer_from_cpu_at(v_ring, dst_off, vsrc + src_off,
                                          seed_bytes);
                }
                if (kv_q4) {
                    /* Re-seed this slot's fp16 sink/window shadow from the
                     * host shadow (cheap; only on a non-contiguous re-sync). */
                    size_t sink_b = (size_t)n_head_kv * PN_KVQ4_SINKMAX *
                                    head_dim * sizeof(uint16_t);
                    size_t win_b = (size_t)n_head_kv * PN_KVQ4_WMAX *
                                   head_dim * sizeof(uint16_t);
                    const size_t sink_h = (size_t)h0 * PN_KVQ4_SINKMAX * head_dim;
                    const size_t win_h = (size_t)h0 * PN_KVQ4_WMAX * head_dim;
                    fc_buffer_from_cpu_at(k_sink, (size_t)su * sink_b,
                                          kv->k_sink + sink_h, sink_b);
                    fc_buffer_from_cpu_at(v_sink, (size_t)su * sink_b,
                                          kv->v_sink + sink_h, sink_b);
                    fc_buffer_from_cpu_at(k_win, (size_t)su * win_b,
                                          kv->k_win + win_h, win_b);
                    fc_buffer_from_cpu_at(v_win, (size_t)su * win_b,
                                          kv->v_win + win_h, win_b);
                }
            }
        }
        d->kv_slot_host_id[su] = host_id;
        d->kv_slot_gen[su] = kv->content_gen;
        d->kv_slot_kv[su] = (pn_qwen35moe_kv *)kv;
        d->kv_slot_synced[su] = pos[u] + 1;   /* the append below lands it */
    }

    /* The flash grid width: the max of the per-user split plans. Every OTHER
     * launch argument below is position-invariant (the slot kernels read each
     * user's position from pos_g and recompute its own plan on-device), so
     * this is the single value a captured graph has to be keyed on. */
    int kps = dec_keys_per_split();
    int launch_splits = 1;
    for (int u = 0; u < nslots; u++) {
        int ns = gpu_attn_slot_plan_splits(pos[u] + 1, kps);
        if (ns > launch_splits)
            launch_splits = ns;
    }

    pl->k_ring = k_ring;   pl->v_ring = v_ring;
    pl->k_sink = k_sink;   pl->v_sink = v_sink;
    pl->k_win  = k_win;    pl->v_win  = v_win;
    pl->rows_g = b->rows_g;
    pl->kv_q4 = kv_q4;
    pl->max_seq = max_seq;
    pl->sink_n = kv_sink;
    pl->win_n = kv_win;
    pl->kv_dim = n_head_kv * head_dim;
    pl->launch_splits = launch_splits;
    pl->kps = kps;
    pl->scale = qw_scale_inv_sqrt(head_dim);
    return 0;
}

/* Phase 2 of the slots attention block: steps 1-5 + the sigmoid output gate,
 * as PURE DEVICE WORK over persistent buffers and the prep'd plan. No host
 * sync, no blocking copy, no allocation — this is the part that goes inside a
 * captured CUDA graph, and it must stay that way. */
static int attn_slots_flash_core(gpu_attn_layer *d,
                                 const pn_qwen35moe_config *cfg,
                                 gpu_batch_scratch *b, fc_buffer *pos_g,
                                 int nslots, const qw_attn_slots_plan *pl)
{
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int attn_dim = n_head * head_dim;
    const int kv_q4 = pl->kv_q4;
    const int max_seq = pl->max_seq;
    const int kv_sink = pl->sink_n, kv_win = pl->win_n;
    const int launch_splits = pl->launch_splits, kps = pl->kps;
    const float scale = pl->scale;
    fc_buffer *k_ring = pl->k_ring, *v_ring = pl->v_ring;
    fc_buffer *k_sink = pl->k_sink, *v_sink = pl->v_sink;
    fc_buffer *k_win = pl->k_win, *v_win = pl->v_win;
    fc_cuda_ctx *ctx = d->ctx;
    int gated = 0;
    double sp_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);

    /* 1. Joint Q+gate, K, V projections for all users. */
    if (gpu_matvec_nx(&d->attn_q, b->x_norm, b->a_qg, nslots) < 0 ||
        gpu_matvec_nx(&d->attn_k, b->x_norm, b->a_k, nslots) < 0 ||
        gpu_matvec_nx(&d->attn_v, b->x_norm, b->a_v, nslots) < 0)
        return -1;
    sprof_lap(SPROF_DENSE, ctx, &sp_t0);

    /* Separate clock: these laps partition SPROF_ATTN_FLASH, they do not
     * shorten it (same contract as the SSM sub-spans). */
    double sub_t0 = 0.0;
    sub_begin(ctx, &sub_t0);

    /* 2.-4. Device q/gate split (z persists in a_z for the output gate),
     * QK-RMSNorm, per-row-position partial NEOX RoPE — same math as the
     * host path, minus the split's D2H/H2D round trip. */
    fc_attn_qz_split_ctx(ctx, b->a_qg, b->a_q, b->a_z, n_head, head_dim,
                         nslots);
    sub_lap(&attd_cum[ATTD_QZSPLIT], &attd_laps[ATTD_QZSPLIT], ctx, &sub_t0);
    fc_qk_rms_norm_ctx(ctx, b->a_q, d->attn_q_norm.buf, nslots * n_head,
                       head_dim, cfg->rms_norm_eps);
    fc_qk_rms_norm_ctx(ctx, b->a_k, d->attn_k_norm.buf, nslots * n_head_kv,
                       head_dim, cfg->rms_norm_eps);
    sub_lap(&attd_cum[ATTD_QKNORM], &attd_laps[ATTD_QKNORM], ctx, &sub_t0);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_q, n_head, head_dim,
                                 cfg->rope_dim, pos_g, cfg->rope_theta,
                                 nslots);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_k, n_head_kv, head_dim,
                                 cfg->rope_dim, pos_g, cfg->rope_theta,
                                 nslots);
    sub_lap(&attd_cum[ATTD_ROPE], &attd_laps[ATTD_ROPE], ctx, &sub_t0);
    bv_trace_rows(ctx, "plain", "q", -1, b->a_q, nslots, attn_dim, NULL,
                  bv_trace_slots_pos0);
    bv_trace_rows(ctx, "plain", "k", -1, b->a_k, nslots, n_head_kv * head_dim,
                  NULL, bv_trace_slots_pos0);
    bv_trace_rows(ctx, "plain", "v", -1, b->a_v, nslots, n_head_kv * head_dim,
                  NULL, bv_trace_slots_pos0);
    bv_trace_rows(ctx, "plain", "z", -1, b->a_z, nslots, attn_dim, NULL,
                  bv_trace_slots_pos0);

    /* 5. Ring append (all K rows, one launch) + per-user split flash +
     * combine into a_out. */
    if (kv_q4) {
        fc_attn_kv_append_slots_q4_ctx(ctx, b->a_k, b->a_v, k_ring,
                                       v_ring, k_sink,
                                       v_sink, k_win,
                                       v_win, n_head_kv, head_dim,
                                       max_seq, pos_g, pl->rows_g, nslots,
                                       kv_sink, kv_win);
        sub_lap(&attd_cum[ATTD_KVAPPEND], &attd_laps[ATTD_KVAPPEND], ctx,
                &sub_t0);
        /* bit7: the sigmoid output gate rides the combine (bit-exact). */
        gated = (gpu_fuse_mask() & 128) &&
                fc_attn_flash_decode_slots_q4_gated_ctx(
                    ctx, b->a_q, k_ring, v_ring, k_sink, v_sink, k_win, v_win,
                    d->dec_slots_m_g, d->dec_slots_l_g, d->dec_slots_acc_g,
                    b->a_out, n_head, n_head_kv, head_dim, max_seq, pos_g,
                    pl->rows_g, nslots, launch_splits, kps, PN_QWEN35MOE_DEC_SPLIT_MAX,
                    kv_sink, kv_win, attn_sink(), attn_window(), scale,
                    b->a_z) == 0;
        if (!gated &&
            fc_attn_flash_decode_slots_q4_ctx(ctx, b->a_q, k_ring,
                                              v_ring, k_sink,
                                              v_sink,
                                              k_win,
                                              v_win,
                                              d->dec_slots_m_g,
                                              d->dec_slots_l_g,
                                              d->dec_slots_acc_g, b->a_out,
                                              n_head, n_head_kv, head_dim,
                                              max_seq, pos_g, pl->rows_g,
                                              nslots, launch_splits, kps,
                                              PN_QWEN35MOE_DEC_SPLIT_MAX,
                                              kv_sink, kv_win, attn_sink(),
                                              attn_window(), scale) < 0)
            return -1;
    } else {
        fc_attn_kv_append_slots_ctx(ctx, b->a_k, b->a_v, k_ring,
                                    v_ring, n_head_kv, head_dim,
                                    max_seq, pos_g, pl->rows_g, nslots);
        sub_lap(&attd_cum[ATTD_KVAPPEND], &attd_laps[ATTD_KVAPPEND], ctx,
                &sub_t0);
        if (fc_attn_flash_decode_slots_ctx(ctx, b->a_q, k_ring,
                                           v_ring, d->dec_slots_m_g,
                                           d->dec_slots_l_g,
                                           d->dec_slots_acc_g, b->a_out,
                                           n_head, n_head_kv, head_dim,
                                           max_seq, pos_g, pl->rows_g, nslots,
                                           launch_splits, kps,
                                           PN_QWEN35MOE_DEC_SPLIT_MAX,
                                           attn_sink(), attn_window(),
                                           scale) < 0)
            return -1;
    }

    sub_lap(&attd_cum[ATTD_FLASH], &attd_laps[ATTD_FLASH], ctx, &sub_t0);

    /* Sigmoid output gate on the die: a_out *= sigmoid(z). */
    if (!gated) {
        fc_sigmoid_ctx(ctx, b->a_z, nslots * attn_dim);
        fc_mul_ctx(ctx, b->a_out, b->a_z, b->a_out, nslots * attn_dim);
    }
    bv_trace_rows(ctx, "plain", "attn", -1, b->a_out, nslots, attn_dim, NULL,
                  bv_trace_slots_pos0);
    sub_lap(&attd_cum[ATTD_OUTGATE], &attd_laps[ATTD_OUTGATE], ctx, &sub_t0);
    sprof_lap(SPROF_ATTN_FLASH, ctx, &sp_t0);
    return 0;
}

/* Phase 3 of the slots attention block — step 6, the host KV mirror append:
 * D2H the roped K/V rows and store them into each user's host cache (identical
 * store code to the host path) so the mirror stays authoritative for
 * bind/move/fallback/preserve.
 *
 * L10: this used to sit MID-body, between the output gate and the out
 * projection, where its blocking fc_sync_ctx made the whole attention layer
 * un-capturable. It runs AFTER the layer body now. That is safe because
 * b->a_k / b->a_v are dedicated attention scratch: the out projection, the
 * post-attention norm and the batched MoE all write elsewhere (a_out /
 * block_out / hidden / x_norm / m_*), so the roped K/V is still intact when
 * this runs, and the NEXT attention layer's projections are the first thing
 * that overwrites it.
 *
 * QWEN35_ASYNC_KVMIRROR: instead of the per-layer blocking sync + D2H +
 * store, queue an ASYNC D2H of this layer's K||V into a per-die host arena
 * on the die's own stream and record the store for deferred replay at
 * end-of-step (qw_defer_flush). No per-layer host stall => the die stays
 * busy and the 2-die pipeline can actually overlap. Byte-identical result:
 * the mirror is fully replayed before any boundary consumer (bind/move/
 * preserve) runs. Falls back to the blocking path if async D2H or arena
 * allocation is unavailable. */
static void attn_slots_mirror(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                              const pn_qwen35moe_config *cfg, int kv_idx,
                              pn_qwen35moe_state *const *sts, const int *pos,
                              gpu_batch_scratch *b, int nslots,
                              const qw_attn_slots_plan *pl, float *hb_kv)
{
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int kv_dim = pl->kv_dim;
    const int kv_q4 = pl->kv_q4;
    const int max_seq = pl->max_seq;
    const int kv_sink = pl->sink_n, kv_win = pl->win_n;
    fc_cuda_ctx *ctx = d->ctx;
    double sp_t0 = 0.0;

    /* Phase 10b (lazy mirror): no per-token host append at all — the slot
     * ring this step appended to is authoritative, and the prep watermark
     * (kv_slot_synced = pos+1) records that the ring, not the mirror, holds
     * the rows. This removes a per-layer fc_sync + 2 D2H copies from every
     * decode step AND stops host RSS from growing with decode length. */
    if (qw_kv_mirror_lazy()) {
        for (int u = 0; u < nslots; u++)
            sts[u]->kv[kv_idx].kv_mirror_stale = 1;
        return;
    }
    sprof_begin(ctx, &sp_t0);

    if (async_kvmirror_on()) {
        int idx = qw_defer_idx(g, ctx);
        qw_kv_defer *df = &g->defer[idx];
        size_t layer_bytes = (size_t)2 * nslots * kv_dim * sizeof(float);
        size_t off = qw_defer_stage(df, layer_bytes);
        qw_kv_store *st = (off != (size_t)-1) ? qw_defer_push(df) : NULL;
        if (st) {
            float *stage = df->stage + off / sizeof(float);
            int ok = (fc_buffer_to_cpu_n_async(b->a_k, stage,
                          (size_t)nslots * kv_dim * sizeof(float)) == 0) &&
                     (fc_buffer_to_cpu_n_async(b->a_v,
                          stage + (size_t)nslots * kv_dim,
                          (size_t)nslots * kv_dim * sizeof(float)) == 0);
            if (ok) {
                st->nslots = nslots;
                st->kv_q4 = kv_q4;
                st->kv_sink = kv_sink;
                st->kv_win = kv_win;
                st->n_head_kv = n_head_kv;
                st->head_dim = head_dim;
                st->max_seq = max_seq;
                st->kv_dim = kv_dim;
                st->stage_off = off;
                for (int u = 0; u < nslots; u++) {
                    st->kv[u] = &sts[u]->kv[kv_idx];
                    st->pos[u] = pos[u];
                }
                sprof_lap(SPROF_ATTN_MIRROR, ctx, &sp_t0);
                return;
            }
            /* async D2H failed — un-push the descriptor + arena, fall back. */
            df->n--;
            df->stage_used = off;
        }
        /* fall through to the blocking path below (correctness preserved). */
    }
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->a_k, hb_kv, (size_t)nslots * kv_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_v, hb_kv + (size_t)nslots * kv_dim,
                       (size_t)nslots * kv_dim * sizeof(float));
    for (int u = 0; u < nslots; u++) {
        pn_qwen35moe_kv *kv = &sts[u]->kv[kv_idx];
        int cache_pos = pos[u] % max_seq;
        const float *kvec = hb_kv + (size_t)u * kv_dim;
        const float *vvec = hb_kv + (size_t)(nslots + u) * kv_dim;
        for (int hd = 0; hd < n_head_kv; hd++) {
            if (kv_q4) {
                pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd,
                                  max_seq, cache_pos,
                                  kvec + (size_t)hd * head_dim, head_dim,
                                  kv_sink, kv_win);
                pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd,
                                  max_seq, cache_pos,
                                  vvec + (size_t)hd * head_dim, head_dim,
                                  kv_sink, kv_win);
            } else {
                memcpy(kv->k + ((size_t)hd * max_seq + cache_pos) * head_dim,
                       kvec + (size_t)hd * head_dim,
                       (size_t)head_dim * sizeof(float));
                memcpy(kv->v + ((size_t)hd * max_seq + cache_pos) * head_dim,
                       vvec + (size_t)hd * head_dim,
                       (size_t)head_dim * sizeof(float));
            }
        }
    }
    sprof_lap(SPROF_ATTN_MIRROR, ctx, &sp_t0);
}

/* The on-die flash path for one batched-decode attention block, as the direct
 * (non-graph-captured) caller sees it: prep -> device core -> host KV mirror ->
 * out projection + residual. Same contract as before the L10 split — 0 = done,
 * 1 = flash unavailable (caller runs host softmax), <0 = error. */
static int gpu_attn_block_slots_flash(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                      const pn_qwen35moe_config *cfg,
                                      int kv_idx,
                                      pn_qwen35moe_state *const *sts,
                                      const int *pos, fc_buffer *pos_g,
                                      gpu_batch_scratch *b, int slot_base,
                                      int nslots, int total_slots,
                                      float *hb_kv)
{
    fc_cuda_ctx *ctx = d->ctx;
    qw_attn_slots_plan pl;
    int rc = attn_slots_flash_prep(d, cfg, kv_idx, sts, pos, pos_g, b,
                                   slot_base, nslots, total_slots, &pl);
    if (rc != 0)
        return rc;   /* 1 = host softmax path, <0 = error (plan freed) */

    rc = attn_slots_flash_core(d, cfg, b, pos_g, nslots, &pl);
    if (rc == 0) {
        attn_slots_mirror(g, d, cfg, kv_idx, sts, pos, b, nslots, &pl, hb_kv);
        /* 7. Output projection + residual. */
        double sp_t0 = 0.0;
        sprof_begin(ctx, &sp_t0);
        if (gpu_attn_out_slots(d, cfg, b, nslots) < 0)
            rc = -1;
        sprof_lap(SPROF_DENSE, ctx, &sp_t0);
    }
    return rc;
}

/* Batched-decode attention block: identical launch/glue sequence to the
 * prefill gpu_attn_block_batch except (a) RoPE takes each row's OWN
 * position from the die's pos buffer, and (b) the host KV append + causal
 * GQA softmax runs against each USER's own cache at its own position (the
 * same host code as gpu_attn_block, per user). */
static int gpu_attn_block_slots(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                const pn_qwen35moe_config *cfg, int kv_idx,
                                pn_qwen35moe_state *const *sts,
                                const int *pos, fc_buffer *pos_g,
                                gpu_batch_scratch *b, int slot_base,
                                int nslots, int total_slots,
                                float *const hb[4])
{
    const int n_head = cfg->n_head;
    const int n_head_kv = cfg->n_head_kv;
    const int head_dim = cfg->head_dim;
    const int n_kv_groups = n_head / n_head_kv;
    const int q_stride = head_dim * 2;
    const int attn_dim = n_head * head_dim;
    const int kv_dim = n_head_kv * head_dim;
    const float eps = cfg->rms_norm_eps;
    fc_cuda_ctx *ctx = d->ctx;
    /* Group-local host staging: hb[] = this group's 4 staging regions. For a
     * single group these are g->bhost[0..3]; the pipeline hands each group a
     * distinct set so overlapping die-A/die-B host mirrors never collide. */
    float *hb_qg = hb[0];
    float *hb_q  = hb[1];
    float *hb_z  = hb[2];
    float *hb_kv = hb[3];

    /* L7 — on-die per-slot flash attention (QWEN35_FLASH_SLOTS, default
     * on). Falls through to the host-softmax path below only when the slot
     * kernels are unavailable / disabled / a user's ring would wrap. */
    {
        int fr = gpu_attn_block_slots_flash(g, d, cfg, kv_idx, sts, pos,
                                            pos_g, b, slot_base, nslots,
                                            total_slots, hb_kv);
        if (fr <= 0)
            return fr;
    }

    double sp_t0 = 0.0;
    sprof_begin(ctx, &sp_t0);

    /* 1. Joint Q+gate, K, V projections for all users. */
    if (gpu_matvec_nx(&d->attn_q, b->x_norm, b->a_qg, nslots) < 0 ||
        gpu_matvec_nx(&d->attn_k, b->x_norm, b->a_k, nslots) < 0 ||
        gpu_matvec_nx(&d->attn_v, b->x_norm, b->a_v, nslots) < 0)
        return -1;
    sprof_lap(SPROF_DENSE, ctx, &sp_t0);

    /* Q/gate split, host-side per user (same glue as the single-token
     * path; zgate persists in hb_z for the output gate below). */
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->a_qg, hb_qg,
                       (size_t)nslots * 2 * attn_dim * sizeof(float));
    for (int u = 0; u < nslots; u++) {
        const float *src_u = hb_qg + (size_t)u * 2 * attn_dim;
        float *q_u = hb_q + (size_t)u * attn_dim;
        float *z_u = hb_z + (size_t)u * attn_dim;
        for (int hd = 0; hd < n_head; hd++) {
            memcpy(q_u + (size_t)hd * head_dim,
                   src_u + (size_t)hd * q_stride,
                   (size_t)head_dim * sizeof(float));
            memcpy(z_u + (size_t)hd * head_dim,
                   src_u + (size_t)hd * q_stride + head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    fc_buffer_from_cpu_n(b->a_q, hb_q,
                         (size_t)nslots * attn_dim * sizeof(float));

    /* 2.-4. QK-RMSNorm (heads uniform across rows) + per-row-position
     * partial NEOX RoPE (each user at its own context length). */
    fc_qk_rms_norm_ctx(ctx, b->a_q, d->attn_q_norm.buf, nslots * n_head,
                       head_dim, eps);
    fc_qk_rms_norm_ctx(ctx, b->a_k, d->attn_k_norm.buf, nslots * n_head_kv,
                       head_dim, eps);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_q, n_head, head_dim,
                                 cfg->rope_dim, pos_g, cfg->rope_theta,
                                 nslots);
    fc_rope_neox_partial_pos_ctx(ctx, b->a_k, n_head_kv, head_dim,
                                 cfg->rope_dim, pos_g, cfg->rope_theta,
                                 nslots);

    /* 5.+6. Host: per USER, append K/V to that user's cache, causal GQA
     * softmax over that user's window, sigmoid output gate — identical
     * code to gpu_attn_block per user. */
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(b->a_q, hb_q, (size_t)nslots * attn_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_k, hb_kv, (size_t)nslots * kv_dim * sizeof(float));
    fc_buffer_to_cpu_n(b->a_v, hb_kv + (size_t)nslots * kv_dim,
                       (size_t)nslots * kv_dim * sizeof(float));
    {
        float *aout_all = hb_qg;
        float scale = qw_scale_inv_sqrt(head_dim);
        /* heap scratch: max_seq (same for all slots) can reach the 256K cap —
         * too large for the stack. */
        int scores_cap = (nslots > 0) ? sts[0]->kv[kv_idx].max_seq : 4096;
        float *scores = (float *)malloc((size_t)scores_cap * sizeof(float));
        float krow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        float vrow[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
        if (!scores) return -1;
        for (int u = 0; u < nslots; u++) {
            pn_qwen35moe_kv *kv = &sts[u]->kv[kv_idx];
            float *k_cache = kv->k;
            float *v_cache = kv->v;
            const int kv_q4 = kv->q4;
            const int kv_sink = kv_q4 ? pn_kvq4_sink() : 0;
            const int kv_win = kv_q4 ? pn_kvq4_win() : 0;
            int max_seq = kv->max_seq;
            int p = pos[u];
            int cache_pos = p % max_seq;
            /* Phase 10b (lazy mirror): this host path READS the mirror
             * [0, seq_len) below. If decode ran on-die since the last
             * materialization the mirror is missing rows — pull them from
             * the GPU ring first. From here on this layer keeps the mirror
             * current itself (it appends every step), so this fires once
             * per fallback engagement, not per token. */
            if (kv->kv_mirror_stale &&
                qw_kv_layer_materialize(d, kv,
                                        (p < max_seq) ? p : max_seq) != 0) {
                fprintf(stderr, "[attn] slots host fallback kv_idx=%d "
                        "slot=%d: host mirror stale with no ring source — "
                        "failing step rather than attending garbage\n",
                        kv_idx, u);
                free(scores);
                return -1;
            }
            const float *kvec = hb_kv + (size_t)u * kv_dim;
            const float *vvec = hb_kv + (size_t)(nslots + u) * kv_dim;
            for (int hd = 0; hd < n_head_kv; hd++) {
                if (kv_q4) {
                    pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, hd,
                                      max_seq, cache_pos,
                                      kvec + (size_t)hd * head_dim, head_dim,
                                      kv_sink, kv_win);
                    pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, hd,
                                      max_seq, cache_pos,
                                      vvec + (size_t)hd * head_dim, head_dim,
                                      kv_sink, kv_win);
                } else {
                    memcpy(k_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                           kvec + (size_t)hd * head_dim,
                           (size_t)head_dim * sizeof(float));
                    memcpy(v_cache + ((size_t)hd * max_seq + cache_pos) * head_dim,
                           vvec + (size_t)hd * head_dim,
                           (size_t)head_dim * sizeof(float));
                }
            }

            int seq_len = (p + 1 < max_seq) ? p + 1 : max_seq;
            const float *qhost = hb_q + (size_t)u * attn_dim;
            float *aout = aout_all + (size_t)u * attn_dim;
            for (int qh = 0; qh < n_head; qh++) {
                int kv_h = qh / n_kv_groups;
                const float *q_vec = qhost + (size_t)qh * head_dim;
                const float *k_base = kv_q4 ? NULL :
                    k_cache + (size_t)kv_h * max_seq * head_dim;
                const float *v_base = kv_q4 ? NULL :
                    v_cache + (size_t)kv_h * max_seq * head_dim;
                float *out_vec = aout + (size_t)qh * head_dim;
                for (int sp = 0; sp < seq_len; sp++) {
                    const float *k_vec;
                    if (kv_q4) {
                        pn_kvq4_load_row(kv->kq, kv->k_sink, kv->k_win, kv_h,
                                         max_seq, sp, p, krow, head_dim,
                                         kv_sink, kv_win);
                        k_vec = krow;
                    } else {
                        k_vec = k_base + (size_t)sp * head_dim;
                    }
                    scores[sp] = pn_dot_f32(q_vec, k_vec, head_dim) * scale;
                }
                float mx = scores[0];
                for (int sp = 1; sp < seq_len; sp++)
                    if (scores[sp] > mx) mx = scores[sp];
                float sum_exp = 0.0f;
                for (int sp = 0; sp < seq_len; sp++) {
                    scores[sp] = expf(scores[sp] - mx);
                    sum_exp += scores[sp];
                }
                float inv_sum = 1.0f / sum_exp;
                memset(out_vec, 0, (size_t)head_dim * sizeof(float));
                for (int sp = 0; sp < seq_len; sp++) {
                    float w_s = scores[sp] * inv_sum;
                    const float *v_vec;
                    if (kv_q4) {
                        pn_kvq4_load_row(kv->vq, kv->v_sink, kv->v_win, kv_h,
                                         max_seq, sp, p, vrow, head_dim,
                                         kv_sink, kv_win);
                        v_vec = vrow;
                    } else {
                        v_vec = v_base + (size_t)sp * head_dim;
                    }
                    for (int dd = 0; dd < head_dim; dd++)
                        out_vec[dd] += w_s * v_vec[dd];
                }
            }

            const float *z_u = hb_z + (size_t)u * attn_dim;
            for (int i = 0; i < attn_dim; i++) {
                float gv = 1.0f / (1.0f + expf(-z_u[i]));
                aout[i] *= gv;
            }
        }
        free(scores);
        fc_buffer_from_cpu_n(b->a_out, aout_all,
                             (size_t)nslots * attn_dim * sizeof(float));
    }
    sprof_lap(SPROF_ATTN_HOST, ctx, &sp_t0);

    /* 7. Output projection + residual. */
    if (gpu_attn_out_slots(d, cfg, b, nslots) < 0)
        return -1;
    sprof_lap(SPROF_DENSE, ctx, &sp_t0);
    return 0;
}

/* L9 — the position-invariant SSM slot layer body (attn_norm -> slot-SSM
 * block -> post_attention_norm -> slots MoE -> +residual). On a GROUPED layer
 * this is pure stream-ordered: no host sync, no D2H, no allocation — every
 * buffer is persistent (the per-slot device state + the per-die batch
 * scratch), so the exact same launch sequence replays every step, and that is
 * the sequence captured into the per-layer slots graph. A non-grouped layer
 * takes the per-row MoE fallback (which does sync + read back the routing) and
 * is never graph-captured — the capture gate requires moe.grouped (or a
 * dense FFN, whose batched block is equally stream-ordered).
 * Returns 0 on success, <0 on error. */
static int gpu_ssm_slots_layer_body(gpu_ssm_layer *d, fc_cuda_ctx *ctx,
                                    const pn_qwen35moe_config *cfg,
                                    gpu_batch_scratch *b, int slot_base,
                                    int nslots, gpu_die_scratch *sc,
                                    gpu_host_scratch *h)
{
    int c = cfg->n_embd;
    int rc;
    fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c,
                       cfg->rms_norm_eps, nslots);
    rc = gpu_ssm_block_slots(d, cfg, b, slot_base, nslots, 0);
    if (rc < 0)
        return rc;
    fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf,
                       b->x_norm, c, cfg->rms_norm_eps, nslots);
    return gpu_moe_block_slots(ctx, &d->moe, cfg, b, nslots, sc, h);
}

/* L10 — graph-capturable body of ONE batched-decode ATTENTION layer:
 * attn_norm -> on-die slots flash core -> out projection + residual ->
 * post_attention_norm -> batched MoE.
 *
 * Pure device work by construction. The two host-touching pieces of the
 * attention block live outside it: the ring re-sync + plan (attn_slots_flash_
 * prep, before) and the host KV mirror (attn_slots_mirror, after). Everything
 * here reads persistent buffers at stable addresses, and the only launch
 * argument derived from the token positions is pl->launch_splits, which the
 * caller keys the graph on. */
static int gpu_attn_slots_layer_body(gpu_attn_layer *d, fc_cuda_ctx *ctx,
                                     const pn_qwen35moe_config *cfg,
                                     gpu_batch_scratch *b, fc_buffer *pos_g,
                                     int nslots,
                                     const qw_attn_slots_plan *pl,
                                     gpu_die_scratch *sc, gpu_host_scratch *h)
{
    int c = cfg->n_embd;
    fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c,
                       cfg->rms_norm_eps, nslots);
    if (attn_slots_flash_core(d, cfg, b, pos_g, nslots, pl) < 0)
        return -1;
    if (gpu_attn_out_slots(d, cfg, b, nslots) < 0)
        return -1;
    fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf,
                       b->x_norm, c, cfg->rms_norm_eps, nslots);
    return gpu_moe_block_slots(ctx, &d->moe, cfg, b, nslots, sc, h);
}

/* L10 — run one batched-decode ATTENTION layer through its captured CUDA
 * graph, capturing it first if needed.
 *
 * Returns 0 = layer complete, <0 = error, 1 = "not mine, run the direct path"
 * (only when the on-die flash core is unavailable and the host-softmax
 * fallback — which is capture-hostile — has to own the layer). In that case
 * prep did no device work and returned before the ring re-sync, so the direct
 * path re-running prep is a no-op. */
static int gpu_attn_slots_layer_graph(pn_qwen35moe_gpu *g, gpu_attn_layer *d,
                                      fc_cuda_ctx *ctx,
                                      const pn_qwen35moe_config *cfg,
                                      int kv_idx,
                                      pn_qwen35moe_state *const *sts,
                                      const int *pos, fc_buffer *pos_g,
                                      gpu_batch_scratch *b, int layer,
                                      int slot_base, int nslots,
                                      int total_slots, float *const hb[4],
                                      gpu_die_scratch *sc, gpu_host_scratch *h)
{
    qw_attn_slots_plan pl;
    int rc = attn_slots_flash_prep(d, cfg, kv_idx, sts, pos, pos_g, b,
                                   slot_base, nslots, total_slots, &pl);
    if (rc != 0)
        return rc > 0 ? 1 : rc;

    /* A captured graph bakes in the slot count and the ring-offset base (launch
     * grids), the batch-scratch addresses, the KV geometry, the ring
     * allocation, and the flash grid width. A change in any invalidates it —
     * drop + recapture. launch_splits is the only one that moves during a
     * session, and it steps up at most PN_QWEN35MOE_DEC_SPLIT_MAX times as the
     * context grows (once per `kps` tokens), so this is not a per-step
     * recapture. */
    if (d->slots_graph &&
        (d->slots_graph_n != nslots ||
         d->slots_graph_base != slot_base ||
         d->slots_graph_splits != pl.launch_splits ||
         d->slots_graph_q4 != pl.kv_q4 ||
         d->slots_graph_max_seq != pl.max_seq ||
         d->slots_graph_gen != d->slots_ring_gen ||
         d->slots_graph_scratch != (const void *)b->hidden)) {
        /* QWEN35_SDIAG: log EVERY drop, not just the first. A captured graph
         * that is silently invalidated and recaptured each step is worse than
         * no graph at all, and a per-step drop is indistinguishable from a
         * healthy one in a first-step-only report. One line per drop makes the
         * rate directly countable. */
        if (sdiag_on())
            fprintf(stderr, "[sdiag] layer %2d ATTN graph=DROPPED "
                    "n %d->%d base %d->%d splits %d->%d q4 %d->%d "
                    "max_seq %d->%d gen %u->%u scratch %s\n",
                    layer, d->slots_graph_n, nslots,
                    d->slots_graph_base, slot_base,
                    d->slots_graph_splits, pl.launch_splits,
                    d->slots_graph_q4, pl.kv_q4,
                    d->slots_graph_max_seq, pl.max_seq,
                    d->slots_graph_gen, d->slots_ring_gen,
                    d->slots_graph_scratch == (const void *)b->hidden
                        ? "same" : "MOVED");
        fc_graph_free(d->slots_graph);
        d->slots_graph = NULL;
        pn_graph_capture_dropped(&d->slots_graph_capture);
    }

    int done = 0;
    if (d->slots_graph) {
        if (fc_graph_launch(d->slots_graph) == 0) {
            done = 1;
        } else {
            /* Replay failed unexpectedly — drop it and run direct this step. */
            fc_graph_free(d->slots_graph);
            d->slots_graph = NULL;
            pn_graph_capture_dropped(&d->slots_graph_capture);
        }
    }

    /* Capture lazily, and RETRY a failed capture on a bounded backoff — a
     * capture lost to a transient collision must not cost this layer its graph
     * for the life of the process. The recorded launches do NOT execute, so on
     * any failure the direct body runs this step and nothing was
     * half-applied. */
    if (!done && pn_graph_capture_should_attempt(&d->slots_graph_capture)) {
        int brc = 0;
        int begin_ok = fc_graph_begin_ctx(ctx) == 0;
        if (begin_ok) {
            brc = gpu_attn_slots_layer_body(d, ctx, cfg, b, pos_g, nslots,
                                            &pl, sc, h);
            d->slots_graph = fc_graph_end_ctx(ctx);
            if (pn_graph_capture_inject_now(&d->slots_graph_capture) &&
                d->slots_graph) {
                fc_graph_free(d->slots_graph);   /* fault injection, test only */
                d->slots_graph = NULL;
            }
            if (brc == 0 && d->slots_graph &&
                fc_graph_launch(d->slots_graph) == 0) {
                d->slots_graph_n = nslots;
                d->slots_graph_base = slot_base;
                d->slots_graph_splits = pl.launch_splits;
                d->slots_graph_q4 = pl.kv_q4;
                d->slots_graph_max_seq = pl.max_seq;
                d->slots_graph_gen = d->slots_ring_gen;
                d->slots_graph_scratch = (const void *)b->hidden;
                done = 1;
            } else if (d->slots_graph) {
                fc_graph_free(d->slots_graph);
                d->slots_graph = NULL;
            }
        }

        if (done) {
            pn_graph_capture_succeeded(&d->slots_graph_capture);
            if (sdiag_on() && !d->moe.sdiag_logged) {
                d->moe.sdiag_logged = 1;
                fprintf(stderr, "[sdiag] layer %2d ATTN moe=%s "
                        "graph=CAPTURED n=%d base=%d splits=%d\n",
                        layer, cfg->dense_ffn ? "dense" : "grouped",
                        nslots, slot_base, pl.launch_splits);
            }
        } else {
            pn_graph_capture_failed(&d->slots_graph_capture);
            /* Log EVERY failure, not just the first: the retry budget bounds
             * this to PN_GRAPH_CAPTURE_MAX_FAILURES lines per layer, and the
             * attempt/cooldown make a permanent give-up visibly different from
             * a transient collision that recovers. */
            if (sdiag_on())
                fprintf(stderr, "[sdiag] layer %2d ATTN moe=%s "
                        "graph=%s (body rc=%d) attempt %d/%d -> %s\n", layer,
                        cfg->dense_ffn ? "dense" : "grouped",
                        begin_ok ? "CAPTURE FAILED" : "BEGIN FAILED", brc,
                        d->slots_graph_capture.consecutive_failures,
                        PN_GRAPH_CAPTURE_MAX_FAILURES,
                        pn_graph_capture_state_name(
                            d->slots_graph_capture.state));
        }
    }

    if (!done &&
        gpu_attn_slots_layer_body(d, ctx, cfg, b, pos_g, nslots, &pl,
                                  sc, h) < 0)
        return -1;

    /* Step 6, outside the graph: b->a_k / b->a_v still hold this layer's roped
     * K/V (nothing in the body after the flash core writes them). */
    attn_slots_mirror(g, d, cfg, kv_idx, sts, pos, b, nslots, &pl, hb[3]);
    return 0;
}

/* One layer of the batched decode: attn_norm -> (slot-SSM | per-user attn)
 * -> +residual -> post_attention_norm -> shared-MoE -> +residual over the
 * [K][n_embd] residual stream (the MoE block is the prefill one — routing
 * is already per-row). */
static int gpu_layer_forward_slots(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                                   fc_cuda_ctx *ctx, int layer,
                                   gpu_batch_scratch *b,
                                   pn_qwen35moe_state *const *sts,
                                   const int *pos, fc_buffer *pos_g,
                                   int slot_base, int nslots, int total_slots,
                                   float *const hb[4])
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    int rc;
    /* Only a NON-grouped layer needs the per-row MoE fallback's scratch;
     * resolving it lazily keeps the grouped (graph-captured) path allocation-
     * free, exactly as before. */
    gpu_host_scratch *host_scr = g->host_ok ? &g->host : NULL;
    /* Users live in persistent rows anywhere in [0, g->n_slots): size the
     * rings to every bound row, not to this step's width. */
    if (total_slots < g->n_slots)
        total_slots = g->n_slots;

    /* Which gate closed the graph path for this layer this step? Evaluated in
     * the same order the graph_ok expressions below test them, so the FIRST
     * failing gate is the one reported. */
    pn_graph_ineligible_reason gate_reason = PN_GRAPH_INELIGIBLE_NONE;
    int is_attn = pn_qwen35moe_model_layer_type(m, layer) ==
                  PN_QWEN35MOE_LAYER_ATTN;
    /* A dense FFN (Bonsai, Ornith, Qwen3.6/3.8-27B) has grouped = 0 but takes
     * gpu_dense_ffn_batch — pure stream-ordered launches into the persistent
     * batch scratch, as capturable as the grouped block. Same rule as the
     * single-user gpu_segment_forward gate (grouped || dense_ffn). */
    int layer_grouped = (is_attn ? g->attn[idx].moe.grouped
                                 : g->ssm[idx].moe.grouped) || cfg->dense_ffn;
    /* Before any capture below: one-slot fused descriptors build here. */
    gpu_slots_fused_prepare(g, cfg, is_attn, idx, b, nslots);
    const char *moe_kind = cfg->dense_ffn ? "dense" : "grouped";
    if (!gpu_graph_enabled())
        gate_reason = PN_GRAPH_INELIGIBLE_GRAPH_OFF;
    else if (!gpu_graph_slots_enabled())
        gate_reason = PN_GRAPH_INELIGIBLE_SLOTS_OFF;
    else if (is_attn && !gpu_graph_slots_attn_enabled())
        gate_reason = PN_GRAPH_INELIGIBLE_ATTN_OFF;
    else if (sprof_active)
        gate_reason = PN_GRAPH_INELIGIBLE_PROFILING;
    else if (!layer_grouped)
        gate_reason = PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED;
    else if (slot_base != 0)
        gate_reason = PN_GRAPH_INELIGIBLE_SLOT_BASE;

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        gpu_attn_layer *d = &g->attn[idx];
        gpu_die_scratch *die_scr = d->moe.grouped ? NULL
                                                  : gpu_get_scratch(g, ctx, cfg);
        /* L10 — captured-graph path for the attention layer. Same gates as the
         * SSM layers (graph env on, sprof off — its mid-body syncs would break
         * capture — grouped MoE, base-0 group only), plus the flash core being
         * available: the host-softmax fallback is unconditionally
         * capture-hostile and owns its layer end to end. */
        int graph_ok = gpu_graph_enabled() && gpu_graph_slots_enabled() &&
                       gpu_graph_slots_attn_enabled() && !sprof_active &&
                       layer_grouped && slot_base == 0;
        if (graph_ok) {
            rc = gpu_attn_slots_layer_graph(g, d, ctx, cfg, idx, sts, pos,
                                            pos_g, b, layer, slot_base, nslots,
                                            total_slots, hb, die_scr,
                                            host_scr);
            if (rc <= 0)
                return rc;   /* 0 = layer done, <0 = error; >0 = run direct */
        } else {
            pn_graph_capture_ineligible(&d->slots_graph_capture, gate_reason);
        }
        if (!graph_ok && sdiag_on() && !d->moe.sdiag_logged) {
            d->moe.sdiag_logged = 1;
            fprintf(stderr, "[sdiag] layer %2d ATTN moe=%s graph=%s\n", layer,
                    layer_grouped ? moe_kind : "PER-ROW FALLBACK",
                    !gpu_graph_enabled()         ? "off (QWEN35_GRAPH=0)"
                    : !gpu_graph_slots_enabled() ? "off (QWEN35_GRAPH_SLOTS=0)"
                    : !gpu_graph_slots_attn_enabled()
                                       ? "never (QWEN35_GRAPH_SLOTS_ATTN=0)"
                    : sprof_active               ? "disabled (sprof active)"
                    : !layer_grouped             ? "INELIGIBLE (MoE not grouped)"
                                                 : "skipped (slot_base>0)");
        }
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->attn_norm.buf, b->x_norm, c,
                           cfg->rms_norm_eps, nslots);
        rc = gpu_attn_block_slots(g, d, cfg, idx, sts, pos, pos_g, b,
                                  slot_base, nslots, total_slots, hb);
        if (rc < 0)
            return rc;
        fc_rms_norm_nx_ctx(ctx, b->hidden, d->post_attention_norm.buf,
                           b->x_norm, c, cfg->rms_norm_eps, nslots);
        return gpu_moe_block_slots(ctx, &d->moe, cfg, b, nslots, die_scr,
                                   host_scr);
    }

    /* ---- SSM layer. ---- */
    gpu_ssm_layer *d = &g->ssm[idx];
    gpu_die_scratch *die_scr = d->moe.grouped ? NULL
                                              : gpu_get_scratch(g, ctx, cfg);
    if (!d->conv_slots || !d->ssm_slots)
        return -1;  /* slots not bound */
    for (int u = 0; u < nslots; u++)
        if (b->rows_h[u] < 0 || b->rows_h[u] >= d->slots_cap)
            return -1;  /* a user's row was never carved */

    /* L9 — replay the captured slots graph (~30 launches -> 1) when the graph
     * gate is on, sprof is off (its mid-body syncs would break capture), MoE
     * is grouped, and the graph matches this step's slot count.
     * L5 — only the base-0 group uses the graph: a per-layer graph slot bakes
     * in one slot_base, and alternating two groups' bases per step would drop +
     * recapture every layer every step (thrash). The base-0 group keeps a
     * stable graph across steps; the other group (slot_base>0) runs direct. */
    int graph_ok = gpu_graph_enabled() && gpu_graph_slots_enabled() &&
                   !sprof_active && layer_grouped && slot_base == 0;

    /* A captured graph bakes in the slot count (launch grids), the batch-
     * scratch buffer addresses AND (L5) the ring-offset baked from slot_base;
     * a change in any invalidates it — drop + recapture. */
    if (d->slots_graph && (d->slots_graph_n != nslots ||
                           d->slots_graph_base != slot_base ||
                           d->slots_graph_scratch != (const void *)b->hidden)) {
        /* QWEN35_SDIAG: see the ATTN drop log — one line per drop so a
         * per-step recapture cannot hide behind a first-step-only report. */
        if (sdiag_on())
            fprintf(stderr, "[sdiag] layer %2d SSM  graph=DROPPED "
                    "n %d->%d base %d->%d scratch %s\n",
                    layer, d->slots_graph_n, nslots,
                    d->slots_graph_base, slot_base,
                    d->slots_graph_scratch == (const void *)b->hidden
                        ? "same" : "MOVED");
        fc_graph_free(d->slots_graph);
        d->slots_graph = NULL;
        pn_graph_capture_dropped(&d->slots_graph_capture);
    }

    if (graph_ok && d->slots_graph) {
        if (fc_graph_launch(d->slots_graph) == 0)
            return 0;
        /* Replay failed unexpectedly — drop it and run direct this step. */
        fc_graph_free(d->slots_graph);
        d->slots_graph = NULL;
        pn_graph_capture_dropped(&d->slots_graph_capture);
    }

    if (!graph_ok)
        pn_graph_capture_ineligible(&d->slots_graph_capture, gate_reason);

    /* Capture lazily, and RETRY a failed capture on a bounded backoff (see
     * gpu_graph_capture_tracker.h): the recorded launches do NOT execute, so on
     * any failure the direct body runs this step and nothing was
     * half-applied. */
    if (graph_ok && !d->slots_graph &&
        pn_graph_capture_should_attempt(&d->slots_graph_capture)) {
        int captured = 0;
        int begin_ok = fc_graph_begin_ctx(ctx) == 0;
        rc = 0;
        if (begin_ok) {
            rc = gpu_ssm_slots_layer_body(d, ctx, cfg, b, slot_base, nslots,
                                          die_scr, host_scr);
            d->slots_graph = fc_graph_end_ctx(ctx);
            if (pn_graph_capture_inject_now(&d->slots_graph_capture) &&
                d->slots_graph) {
                fc_graph_free(d->slots_graph);   /* fault injection, test only */
                d->slots_graph = NULL;
            }
            if (rc == 0 && d->slots_graph &&
                fc_graph_launch(d->slots_graph) == 0) {
                d->slots_graph_n = nslots;
                d->slots_graph_base = slot_base;
                d->slots_graph_scratch = (const void *)b->hidden;
                captured = 1;
            } else if (d->slots_graph) {
                fc_graph_free(d->slots_graph);
                d->slots_graph = NULL;
            }
        }

        if (captured) {
            pn_graph_capture_succeeded(&d->slots_graph_capture);
            if (sdiag_on() && !d->moe.sdiag_logged) {
                d->moe.sdiag_logged = 1;
                fprintf(stderr, "[sdiag] layer %2d SSM  moe=%s "
                        "graph=CAPTURED n=%d base=%d\n",
                        layer, moe_kind, nslots, slot_base);
            }
            return 0;
        }
        pn_graph_capture_failed(&d->slots_graph_capture);
        /* Every failure, bounded by the retry budget — see the ATTN log. */
        if (sdiag_on())
            fprintf(stderr, "[sdiag] layer %2d SSM  moe=%s "
                    "graph=%s (body rc=%d) attempt %d/%d -> %s\n", layer,
                    moe_kind,
                    begin_ok ? "CAPTURE FAILED" : "BEGIN FAILED", rc,
                    d->slots_graph_capture.consecutive_failures,
                    PN_GRAPH_CAPTURE_MAX_FAILURES,
                    pn_graph_capture_state_name(d->slots_graph_capture.state));
    }

    /* Not graph-eligible at all: say which precondition failed. */
    if (!graph_ok && sdiag_on() && !d->moe.sdiag_logged) {
        d->moe.sdiag_logged = 1;
        fprintf(stderr, "[sdiag] layer %2d SSM  moe=%s graph=%s\n", layer,
                layer_grouped ? moe_kind : "PER-ROW FALLBACK",
                !gpu_graph_enabled()       ? "off (QWEN35_GRAPH=0)"
                : !gpu_graph_slots_enabled() ? "off (QWEN35_GRAPH_SLOTS=0)"
                : sprof_active             ? "disabled (sprof active)"
                : !layer_grouped           ? "INELIGIBLE (MoE not grouped)"
                : slot_base != 0           ? "skipped (slot_base>0)"
                                           : "direct");
    }

    return gpu_ssm_slots_layer_body(d, ctx, cfg, b, slot_base, nslots,
                                    die_scr, host_scr);
}

/* ============================================================
 * QWEN35_TWO_THREAD — 2-host-thread die driver (TT)
 *
 * One persistent worker thread per die, each owning its die's CUDA context
 * launch queue. On a K80 in small-BAR mode both dies are live behind their own
 * persistent fc_cuda_ctx (no MMIO-slot barswap — g->bs == NULL), and CUDA's
 * context stack is thread-local, so two host threads issuing launches to two
 * different contexts run their dies' kernel queues in parallel. This removes
 * the single-host-thread launch-starvation term (one thread could not enqueue
 * both dies' hundreds of small launches fast enough — each die sat ~50% SM even
 * when "overlapping"). Reuses the async KV mirror (a per-die thread no longer
 * stalls on a blocking per-layer D2H).
 *
 * Coordination: the calling (control) thread posts a task to each worker and
 * blocks on the worker's completion via a per-worker mutex+condvar. Correct
 * synchronization only where a real data dependency exists — the die-A→die-B
 * boundary hidden-state handoff — expressed as: control waits for worker A's
 * done, does the [K][n_embd] handoff, then posts worker B. Everything each
 * worker touches (its die's ctx/stream/scratch/rings/defer arena) is per-die;
 * the shared host boundary staging (g->bhost) is written by A and read into B
 * only under the control thread's ordering, so no two threads touch it at once.
 *
 * NOTE (measured, honest): under a pure die-A-front / die-B-back LAYER split the
 * two dies CANNOT overlap within one decode step — die B's first layer depends
 * on die A's boundary output, and the next step's die-A front depends on this
 * step's sampled token (autoregressive). So TT alone does not create
 * concurrency for a single-batch step; it is the launch-parallel infrastructure
 * that a row-split TP scheme (both dies every layer) would ride on. Default OFF;
 * bit-exact to the sequential path. Process-constant gate. */
struct qw_tt {
    pthread_t     th[2];
    pthread_mutex_t mtx[2];
    pthread_cond_t  cv_go[2];       /* control -> worker: task ready */
    pthread_cond_t  cv_done[2];     /* worker -> control: task complete */
    int             go[2];          /* task pending for worker i */
    int             done[2];        /* worker i finished its task */
    int             rc[2];          /* worker i return code */
    int             quit;           /* teardown flag */
    pthread_mutex_t claim_mtx;      /* serializes worker index self-assignment */
    int             claim_next;     /* next free worker index (0 then 1) */
    /* Task payload (front/back layer range for the current step). */
    pn_qwen35moe_model      *m;
    pn_qwen35moe_gpu        *g;
    const pn_qwen35moe_config *cfg;
    pn_qwen35moe_state *const *sts;
    const int              *pos;
    fc_buffer              *pos_g[2];    /* per-die uploaded positions */
    gpu_batch_scratch      *bscr[2];     /* per-die batch scratch */
    int                     layer_lo[2];
    int                     layer_hi[2];
    int                     nslots;
};

static void *qw_tt_worker(void *arg)
{
    struct qw_tt *tt = (struct qw_tt *)arg;
    /* Each worker claims its die index once, race-free, under claim_mtx: the
     * first thread to reach here takes 0 (die A / ctx_a), the second takes 1
     * (die B / ctx_b). This avoids depending on tt->th[] being published. */
    pthread_mutex_lock(&tt->claim_mtx);
    int which = tt->claim_next++;
    pthread_mutex_unlock(&tt->claim_mtx);
    for (;;) {
        pthread_mutex_lock(&tt->mtx[which]);
        while (!tt->go[which] && !tt->quit)
            pthread_cond_wait(&tt->cv_go[which], &tt->mtx[which]);
        if (tt->quit) {
            pthread_mutex_unlock(&tt->mtx[which]);
            return NULL;
        }
        tt->go[which] = 0;
        pthread_mutex_unlock(&tt->mtx[which]);

        /* Run this die's assigned layer range on the full batch. Each
         * gpu_layer_forward_slots targets tt->g->ctx[0]/ctx_b directly (per-die
         * ctx); the fc_* calls push/pop that context on THIS thread's stack. */
        fc_cuda_ctx *ctx = (which == 0) ? tt->g->ctx[0] : tt->g->head_ctx;
        int rc = 0;
        for (int layer = tt->layer_lo[which];
             layer < tt->layer_hi[which] && rc == 0; layer++) {
            rc = gpu_layer_forward_slots(tt->m, tt->g, ctx, layer,
                                         tt->bscr[which], tt->sts, tt->pos,
                                         tt->pos_g[which], 0, tt->nslots,
                                         tt->nslots, tt->g->bhost);
        }
        /* Drain this die's stream so the control thread's post-work (the
         * boundary handoff read, or the head) sees committed results. */
        if (rc == 0)
            fc_sync_ctx(ctx);

        pthread_mutex_lock(&tt->mtx[which]);
        tt->rc[which] = rc;
        tt->done[which] = 1;
        pthread_cond_signal(&tt->cv_done[which]);
        pthread_mutex_unlock(&tt->mtx[which]);
    }
}

static void qw_tt_post(struct qw_tt *tt, int which)
{
    pthread_mutex_lock(&tt->mtx[which]);
    tt->done[which] = 0;
    tt->go[which] = 1;
    pthread_cond_signal(&tt->cv_go[which]);
    pthread_mutex_unlock(&tt->mtx[which]);
}

static int qw_tt_wait(struct qw_tt *tt, int which)
{
    pthread_mutex_lock(&tt->mtx[which]);
    while (!tt->done[which])
        pthread_cond_wait(&tt->cv_done[which], &tt->mtx[which]);
    int rc = tt->rc[which];
    pthread_mutex_unlock(&tt->mtx[which]);
    return rc;
}

static void qw_tt_free(pn_qwen35moe_gpu *g)
{
    struct qw_tt *tt = g->tt;
    if (!tt)
        return;
    tt->quit = 1;
    for (int i = 0; i < 2; i++) {
        pthread_mutex_lock(&tt->mtx[i]);
        pthread_cond_signal(&tt->cv_go[i]);
        pthread_mutex_unlock(&tt->mtx[i]);
    }
    for (int i = 0; i < 2; i++)
        pthread_join(tt->th[i], NULL);
    for (int i = 0; i < 2; i++) {
        pthread_mutex_destroy(&tt->mtx[i]);
        pthread_cond_destroy(&tt->cv_go[i]);
        pthread_cond_destroy(&tt->cv_done[i]);
    }
    pthread_mutex_destroy(&tt->claim_mtx);
    free(tt);
    g->tt = NULL;
}

/* Lazily spin up the two worker threads. Returns the handle or NULL (caller
 * falls back to the sequential path). The workers claim indices 0/1 by the
 * order they are created: we create th[0] first and let it settle before th[1]
 * so pthread_equal against th[0] is deterministic. */
static struct qw_tt *qw_tt_ensure(pn_qwen35moe_gpu *g)
{
    if (g->tt)
        return g->tt;
    struct qw_tt *tt = calloc(1, sizeof(*tt));
    if (!tt)
        return NULL;
    tt->g = g;
    pthread_mutex_init(&tt->claim_mtx, NULL);
    tt->claim_next = 0;
    for (int i = 0; i < 2; i++) {
        pthread_mutex_init(&tt->mtx[i], NULL);
        pthread_cond_init(&tt->cv_go[i], NULL);
        pthread_cond_init(&tt->cv_done[i], NULL);
    }
    /* Create worker 0 first; it will observe th[1] == 0 and claim which==0.
     * Then create worker 1. To make the pthread_equal test robust we set th[i]
     * before creating the thread is impossible; instead each worker computes
     * `which` from th[0], so worker 1 (created after th[0] is set) sees
     * !equal(self, th[0]) => 1, and worker 0 sees equal => 0. Serialize the
     * two creations so th[0] is fully published before worker 1 starts. */
    if (pthread_create(&tt->th[0], NULL, qw_tt_worker, tt) != 0) {
        free(tt);
        return NULL;
    }
    if (pthread_create(&tt->th[1], NULL, qw_tt_worker, tt) != 0) {
        tt->quit = 1;
        pthread_mutex_lock(&tt->mtx[0]);
        pthread_cond_signal(&tt->cv_go[0]);
        pthread_mutex_unlock(&tt->mtx[0]);
        pthread_join(tt->th[0], NULL);
        free(tt);
        return NULL;
    }
    g->tt = tt;
    return tt;
}

/* QWEN35_TWO_THREAD gate (default OFF). Two dies only; small-BAR mode (no
 * barswap: g->bs == NULL); nslots>=1. Process-constant. */
static int two_thread_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_TWO_THREAD");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* QWEN35_DIE_PIPELINE gate (default OFF). When on (and two dies, grouped MoE,
 * nslots>=2) the batched-decode step software-pipelines two batch groups
 * across the die pair so both physical dies compute at once (L5). Off = the
 * proven single-group sequential path runs untouched. Process-constant. */
static int die_pipeline_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_DIE_PIPELINE");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* T3 — vocab-split LM head across the 2 dies (QWEN35_VOCAB_SPLIT). Default
 * OFF; only meaningful in 2-die non-barswap mode with peer access. */
static int vocab_split_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_VOCAB_SPLIT");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* QWEN35_VOCAB_SPLIT_FULL_LOGITS=1 — serve a constrained slot its FULL row FROM
 * the split head (two offset D2Hs, one per die half) instead of abandoning the
 * split for the whole batch.  Default OFF: it is a decode-path change on the
 * live tool-calling shape and has not been through the agent-client gate. */
static int vocab_split_full_logits_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_VOCAB_SPLIT_FULL_LOGITS");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* ------------------------------------------------------------------
 * T3PROF — where does the vocab-split head's predicted saving actually go?
 *
 * The per-kernel model says a 50/50 row split of the 248320x2048 Q6_K head
 * (microbenchmark eta 1.993) should return ~2.4 ms of the phase table's 4.8 ms
 * lm_head. The serve measures ~1.01 ms. That ~1.4 ms gap is the single largest
 * uncertainty in the two-die campaign, because it is ALSO the calibration point
 * for whether per-kernel models over-predict on this engine generally (in which
 * case tensor parallelism is 1.09-1.15x, not 1.33-1.44x) or whether T3 just has
 * its own overhead that nothing else would inherit.
 *
 * So: time every stage of both head paths on the host, with an explicit drain
 * at each boundary so the attribution is real rather than "whatever had not
 * been enqueued yet". The drains INFLATE the totals -- read the split, not the
 * absolutes, exactly as QWEN35_PROF_SUB is read. Enable with
 * QWEN35_T3PROF=<steps between reports> (any non-numeric value means 32).
 * ------------------------------------------------------------------ */
enum {
    T3P_NORM = 0,    /* final RMS norm (both paths) */
    T3P_SYNCB,       /* split: fc_sync_ctx(ctx_b) before the x_norm peer copy */
    T3P_COPY,        /* split: fc_buffer_copy_peer of x_norm, 8 KB/slot */
    T3P_SYNCA,       /* split: fc_sync_ctx(ctx_a) after it */
    T3P_MATVEC,      /* split: BOTH half-head matvecs, drained (concurrent) */
    T3P_TOPKA,       /* split: die A's top-k over its half */
    T3P_TOPKB,       /* split: die B's top-k over its half */
    T3P_MERGE,       /* split: host 2-way merge + present */
    T3P_FULL_MV,     /* single-die: the whole 248320-row head, drained */
    T3P_FULL_PRES,   /* single-die: gpu_present_slots_logits (top-k + D2H) */
    T3P_COUNT
};
static const char *t3p_name[T3P_COUNT] = {
    "rms_norm", "sync(B)", "copy_peer", "sync(A)", "matvec_both",
    "topk_A", "topk_B", "host_merge", "matvec_full", "present_full"
};
static double t3p_cum[T3P_COUNT];
static long   t3p_nsplit, t3p_nfull;

static int t3prof_every(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_T3PROF");
        if (!e || !e[0]) v = 0;
        else { v = atoi(e); if (v <= 0) v = 32; }
    }
    return v;
}

static void t3p_lap(int slot, double *t0)
{
    double now = prof_now();
    t3p_cum[slot] += now - *t0;
    *t0 = now;
}

/* Report + reset. Called from whichever head path ran. */
static void t3p_report(void)
{
    int i;
    long n = t3p_nsplit ? t3p_nsplit : t3p_nfull;
    double tot = 0.0;
    if (n <= 0)
        return;
    for (i = 0; i < T3P_COUNT; i++)
        tot += t3p_cum[i];
    fprintf(stderr, "[T3PROF] %s head, %ld steps, %.3f ms/step total:",
            t3p_nsplit ? "SPLIT" : "single-die", n, tot * 1e3 / n);
    for (i = 0; i < T3P_COUNT; i++)
        if (t3p_cum[i] > 0.0)
            fprintf(stderr, " %s=%.3f", t3p_name[i], t3p_cum[i] * 1e3 / n);
    fprintf(stderr, "\n");
    fflush(stderr);
    memset(t3p_cum, 0, sizeof(t3p_cum));
    t3p_nsplit = t3p_nfull = 0;
}

/* Run one group's FRONT layers [0, front_end) on die A's ctx: embed the
 * group's tokens into hb[0], H2D into b->hidden, run the front layers. The
 * ring/scratch offsetting is handled inside gpu_layer_forward_slots via
 * slot_base. Leaves work queued on ctx_a's stream (no sync). Returns 0/-1. */
static int gpu_group_front(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                           const pn_qwen35moe_config *cfg,
                           const uint32_t *tokens,
                           pn_qwen35moe_state *const *sts, const int *pos,
                           gpu_batch_scratch *ba, fc_buffer *pos_a,
                           int slot_base, int count, int total_slots,
                           float *const hb[4], int front_end)
{
    int c = cfg->n_embd;
    for (int u = 0; u < count; u++)
        pn_qwen35moe_model_embed_row(m, tokens[slot_base + u],
                                     hb[0] + (size_t)u * c);
    fc_buffer_from_cpu_n(ba->hidden, hb[0], (size_t)count * c * sizeof(float));
    for (int layer = 0; layer < front_end; layer++) {
        if (gpu_layer_forward_slots(m, g, g->ctx[0], layer, ba, sts + slot_base,
                                    pos + slot_base, pos_a, slot_base, count,
                                    total_slots, hb) < 0)
            return -1;
    }
    return 0;
}

/* Run one group's BACK layers [split, n_layer) on die B's ctx from the handed-
 * off hidden already in hb[0] (H2D into bb->hidden), then leave work queued on
 * ctx_b (no sync). Returns 0/-1. */
static int gpu_group_back(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                          const pn_qwen35moe_config *cfg,
                          pn_qwen35moe_state *const *sts, const int *pos,
                          gpu_batch_scratch *bb, fc_buffer *pos_b,
                          int slot_base, int count, int total_slots,
                          float *const hb[4], int split, int n_layer)
{
    int c = cfg->n_embd;
    fc_buffer_from_cpu_n(bb->hidden, hb[0], (size_t)count * c * sizeof(float));
    for (int layer = split; layer < n_layer; layer++) {
        if (gpu_layer_forward_slots(m, g, g->head_ctx, layer, bb, sts + slot_base,
                                    pos + slot_base, pos_b, slot_base, count,
                                    total_slots, hb) < 0)
            return -1;
    }
    return 0;
}

/* Final norm + LM head + per-user top-k for one group's rows [slot_base,
 * slot_base+count), reusing g->spec_logits [count][n_vocab]. Runs on die B's
 * ctx (b = bb). Must be called after that group's back layers have synced.
 * Returns 0/-1. */
static int gpu_group_head(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx,
                          const pn_qwen35moe_config *cfg, gpu_batch_scratch *b,
                          gpu_die_scratch *sc, pn_qwen35moe_state *const *sts,
                          int slot_base, int count)
{
    int c = cfg->n_embd;
    fc_rms_norm_nx_ctx(ctx, b->hidden, g->output_norm.buf, b->x_norm, c,
                       cfg->rms_norm_eps, count);
    if (g->spec_logits_cap < count) {
        fc_buffer_free(g->spec_logits);
        g->spec_logits = fc_buffer_alloc_ctx(ctx, (size_t)count *
                                             cfg->n_vocab * sizeof(float));
        g->spec_logits_cap = g->spec_logits ? count : 0;
        if (!g->spec_logits_cap)
            return -1;
    }
    if (gpu_matvec_nx(&g->output, b->x_norm, g->spec_logits, count) < 0)
        return -1;
    return gpu_present_slots_logits(g, ctx, g->spec_logits, sc, sts + slot_base,
                                    count, cfg->n_vocab);
}

/* T3 — vocab-split LM head. `b->x_norm` [nslots][n_embd] is already computed
 * on die B (ctx_b). Split the head: die B computes the BOTTOM half
 * [vocab_half, n_vocab) local logits + top-k, die A computes the TOP half
 * [0, vocab_half) local logits + top-k (from an x_norm mirror P2P-copied from
 * die B), then the two descending per-slot top-k lists are merged order-stably
 * on the host and presented per slot. Bit-exact to the single-die head: each
 * die's per-row top-k is the exact iterative argmax over ITS vocab range, and
 * the 2-way merge (larger value first; ties → the die-A element, whose global
 * index is < vocab_half ≤ every die-B index) reproduces the single argmax's
 * value-desc / lowest-index tiebreak over the full vocab. Runs ONCE per step,
 * outside the layer loop, so there is no per-layer sync hazard. Returns 0 on
 * success, -1 on any error (caller falls back to the single-die head). */
static int gpu_split_head_slots(pn_qwen35moe_gpu *g,
                                const pn_qwen35moe_config *cfg,
                                gpu_batch_scratch *bb,
                                pn_qwen35moe_state *const *sts, int nslots)
{
    int c = cfg->n_embd;
    int nv = cfg->n_vocab;
    int half = g->vocab_half;              /* die A rows [0,half) */
    int nb = nv - half;                    /* die B rows [half,nv) */
    int k = gpu_topk_k();
    /* How many slots need the whole row this step (grammar mask live,
     * logit_bias, logprobs).  The split head keeps each half's logits on its
     * OWN die — spec_logits_a [K][half] on A, spec_logits_b [K][nb] on B — so
     * serving one of those slots means two offset D2H copies off two dies
     * instead of one contiguous copy off one.  That is the entire difference,
     * and it is why the whole batch used to be dropped off the split head the
     * moment any single slot asked: worth +1.9% K=1 / +3.0% K=2, given back
     * because one user sent `tools`.  QWEN35_VOCAB_SPLIT_FULL_LOGITS=1 does the
     * two copies instead. */
    int n_full = 0;
    for (int u = 0; u < nslots; u++) {
        if (!sts[u]) return -1;   /* same guard gpu_present_slots_logits makes */
        if (sts[u]->full_logits) n_full++;
    }
    if (n_full > 0 && !vocab_split_full_logits_on())
        k = 0;                              /* legacy: no split for this batch */
    if (k <= 0 || half <= 0 || nb <= 0)
        return -1;                          /* full-logits mode: no split */

    /* die-A x_norm mirror + die-A logits buffer (lazily sized). */
    if (g->xnorm_a_cap < nslots) {
        fc_buffer_free(g->xnorm_a);
        g->xnorm_a = fc_buffer_alloc_ctx(g->ctx[0],
                                         (size_t)nslots * c * sizeof(float));
        g->xnorm_a_cap = g->xnorm_a ? nslots : 0;
        if (!g->xnorm_a_cap)
            return -1;
    }
    if (g->spec_logits_a_cap < nslots) {
        fc_buffer_free(g->spec_logits_a);
        g->spec_logits_a = fc_buffer_alloc_ctx(g->ctx[0],
                                          (size_t)nslots * half * sizeof(float));
        g->spec_logits_a_cap = g->spec_logits_a ? nslots : 0;
        if (!g->spec_logits_a_cap)
            return -1;
    }
    /* die-B logits buffer (bottom half) — a DEDICATED [K][nb] buffer. */
    if (g->spec_logits_b_cap < nslots) {
        fc_buffer_free(g->spec_logits_b);
        g->spec_logits_b = fc_buffer_alloc_ctx(g->head_ctx,
                                          (size_t)nslots * nb * sizeof(float));
        g->spec_logits_b_cap = g->spec_logits_b ? nslots : 0;
        if (!g->spec_logits_b_cap)
            return -1;
    }

    /* P2P the [K][n_embd] x_norm from die B -> die A. bb->x_norm is on ctx_b;
     * xnorm_a on ctx_a. Sync die B so x_norm is fully written, then copy, then
     * sync die A so the copied data is committed and stream-ordered before the
     * die-A matvec reads it. */
    double t3t = 0.0;
    int t3p = t3prof_every();
    if (t3p) t3t = prof_now();

    fc_sync_ctx(g->head_ctx);
    if (t3p) t3p_lap(T3P_SYNCB, &t3t);
    if (fc_buffer_copy_peer(g->xnorm_a, bb->x_norm,
                            (size_t)nslots * c * sizeof(float)) != 0)
        return -1;
    if (t3p) t3p_lap(T3P_COPY, &t3t);
    fc_sync_ctx(g->ctx[0]);
    if (t3p) t3p_lap(T3P_SYNCA, &t3t);

    /* die-A top-half matvec + local top-k, die-B bottom-half matvec + local
     * top-k. The two run on separate real devices. */
    if (gpu_matvec_nx(&g->output_a, g->xnorm_a, g->spec_logits_a, nslots) < 0)
        return -1;
    if (gpu_matvec_nx(&g->output_b, bb->x_norm, g->spec_logits_b, nslots) < 0)
        return -1;
    if (t3p) {
        /* Drain BOTH so the concurrent matvec pair is attributed to itself and
         * not smeared into whichever top-k call happens to block first. */
        fc_sync_ctx(g->ctx[0]);
        fc_sync_ctx(g->head_ctx);
        t3p_lap(T3P_MATVEC, &t3t);
    }

    /* Per-die per-slot top-k. Die A indices are already global [0,half); die B
     * local indices [0,nb) are offset by +half to global. Both lists come back
     * value-descending (iterative argmax).
     *
     * Skipped entirely when EVERY slot wants the full row (K=1 with a live
     * grammar — the single-user tool-calling shape): there is nothing to reduce
     * for, and running the reduction to throw it away would cost 2 x 81
     * launches.  The two halves then still have to be COMMITTED before the
     * copies below, which run on the legacy NULL stream while the matvecs are on
     * each ctx's own stream — the same explicit-sync rule
     * gpu_present_slots_logits documents.  When the reduction does run it syncs
     * both dies itself. */
    int ids_a[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];
    float vals_a[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];
    int ids_b[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];
    float vals_b[PN_QWEN35MOE_GPU_MAX_SLOTS * FC_TOPK_MAX];
    int reduced = (n_full < nslots);
    if (reduced) {
        if (fc_topk_logits_slots_ctx(g->ctx[0], g->spec_logits_a, nslots, half, k,
                                     ids_a, vals_a) != 0)
            return -1;
        if (t3p) t3p_lap(T3P_TOPKA, &t3t);
        if (fc_topk_logits_slots_ctx(g->head_ctx, g->spec_logits_b, nslots, nb, k,
                                     ids_b, vals_b) != 0)
            return -1;
        if (t3p) t3p_lap(T3P_TOPKB, &t3t);
    } else {
        fc_sync_ctx(g->ctx[0]);
        fc_sync_ctx(g->head_ctx);
    }

    /* Present each slot: a constrained slot gets its FULL row assembled from
     * the two halves, everything else the order-stable 2-way merge of the two
     * top-k lists, truncated to k.
     *
     * The full row is bit-identical to the single-die head's.  Each output row
     * is an independent dot product of the same x_norm (die A reads a byte-copy
     * of die B's) against the same Q6_K weight row, the split is at a ROW
     * boundary (half = n_vocab/2, no K-block is cut), and fc_matvec_nx_launch
     * dispatches on the BATCH width, which both halves share — so row i of
     * spec_logits_a/b holds exactly the float the full head would have written.
     * The top-k above clobbers rows during its rounds and restores them before
     * it syncs, which is why the copies are taken after it and not before. */
    for (int u = 0; u < nslots; u++) {
        if (sts[u]->full_logits) {
            fc_buffer_to_cpu_at(g->spec_logits_a, sts[u]->logits,
                                (size_t)u * half * sizeof(float),
                                (size_t)half * sizeof(float));
            fc_buffer_to_cpu_at(g->spec_logits_b, sts[u]->logits + half,
                                (size_t)u * nb * sizeof(float),
                                (size_t)nb * sizeof(float));
            sts[u]->topk_filled = 0;    /* the row holds real values everywhere */
            sts[u]->short_row_live = 0;
            continue;
        }
        const int   *ia = ids_a  + (size_t)u * k;
        const float *va = vals_a + (size_t)u * k;
        const int   *ib = ids_b  + (size_t)u * k;
        const float *vb = vals_b + (size_t)u * k;
        int mids[FC_TOPK_MAX];
        float mvals[FC_TOPK_MAX];
        int ai = 0, bi = 0;
        for (int i = 0; i < k; i++) {
            /* Take the larger value; on a tie prefer die A (lower global
             * index) to match the single argmax's lowest-index tiebreak. */
            if (bi >= k || (ai < k && va[ai] >= vb[bi])) {
                mvals[i] = va[ai];
                mids[i]  = ia[ai];      /* already global [0,half) */
                ai++;
            } else {
                mvals[i] = vb[bi];
                mids[i]  = ib[bi] + half;   /* local -> global */
                bi++;
            }
        }
        gpu_present_topk(sts[u], mids, mvals, k, nv);
    }
    if (t3p) {
        t3p_lap(T3P_MERGE, &t3t);
        if (++t3p_nsplit >= t3p)
            t3p_report();
    }
    return 0;
}

/* Shared final-norm + LM head + per-slot top-k presentation, run on `ctx`
 * (die B in two-die mode, the sole die otherwise). Factored out so both the
 * sequential and the QWEN35_TWO_THREAD drivers use byte-identical head math.
 * Mirrors the tail of the sequential decode: final RMS norm over all K rows,
 * then either the T3 vocab-split head (2-die, peer-enabled, ctx==ctx_b) or the
 * single batched LM head + L8 batched top-k, presenting each slot's sparse
 * top-k into sts[u]->logits. `sp_step0` is the profiler step-start stamp (only
 * read when sprof is active). Returns 0 on success, -1 on error. */
static int gpu_head_from_slots(pn_qwen35moe_gpu *g,
                               const pn_qwen35moe_config *cfg, fc_cuda_ctx *ctx,
                               gpu_batch_scratch *b, gpu_die_scratch *sc,
                               pn_qwen35moe_state *const *sts, int nslots,
                               int two_die, double sp_step0)
{
    int c = cfg->n_embd;
    double sp_t0 = 0.0;
    double t3t = 0.0;
    int t3p = t3prof_every();
    const int lg = qw_ledger_on();
    sprof_begin(ctx, &sp_t0);
    if (t3p) t3t = prof_now();
    if (lg)
        qw_ledger_gpu(QW_LG_H_BEG, ctx);
    fc_rms_norm_nx_ctx(ctx, b->hidden, g->output_norm.buf, b->x_norm, c,
                       cfg->rms_norm_eps, nslots);
    /* A solo step's row can join the MTP head's backlog (pn_qwen35moe_mtp_push_rows after
     * the step), so a solo slot that stepped plain keeps its head history. */
    if (nslots == 1 && !g->pd_cycle)
        mtp_capture_rows(g, ctx, b->x_norm, 1, c);
    if (t3p) { fc_sync_ctx(ctx); t3p_lap(T3P_NORM, &t3t); }

    if (g->vocab_split_ready && vocab_split_on() && two_die &&
        ctx == g->head_ctx &&
        gpu_split_head_slots(g, cfg, b, sts, nslots) == 0) {
        sprof_lap(SPROF_LM_HEAD, ctx, &sp_t0);
        sprof_lap(SPROF_LOGITS, ctx, &sp_t0);
        if (sprof_active) {
            sprof_step_report(nslots, prof_now() - sp_step0);
            sprof_active = 0;
        }
        return 0;
    }

    if (g->spec_logits_cap < nslots) {
        fc_buffer_free(g->spec_logits);
        g->spec_logits = fc_buffer_alloc_ctx(ctx, (size_t)nslots *
                                             cfg->n_vocab * sizeof(float));
        g->spec_logits_cap = g->spec_logits ? nslots : 0;
        if (!g->spec_logits_cap) {
            fprintf(stderr, "[qwen35moe] decode_slots: LM-head logits buffer "
                    "(%d x %d) alloc failed\n", nslots, cfg->n_vocab);
            return -1;
        }
    }
    if (gpu_matvec_nx(&g->output, b->x_norm, g->spec_logits, nslots) < 0) {
        fprintf(stderr, "[qwen35moe] decode_slots: LM head has no batched "
                "kernel (quant type %d)\n", (int)g->output.type);
        return -1;
    }
    if (t3p) { fc_sync_ctx(ctx); t3p_lap(T3P_FULL_MV, &t3t); }
    if (lg) {
        qw_ledger_gpu(QW_LG_H_END, ctx);
        qw_ledger_lap(QW_LH_HEAD_ISSUE);
    }
    sprof_lap(SPROF_LM_HEAD, ctx, &sp_t0);
    if (gpu_present_slots_logits(g, ctx, g->spec_logits, sc, sts, nslots,
                                 cfg->n_vocab) < 0) {
        fprintf(stderr, "[qwen35moe] decode_slots: top-k presentation of %d "
                "rows failed\n", nslots);
        return -1;
    }
    if (t3p) {
        t3p_lap(T3P_FULL_PRES, &t3t);
        if (++t3p_nfull >= t3p)
            t3p_report();
    }
    sprof_lap(SPROF_LOGITS, ctx, &sp_t0);
    if (sprof_active) {
        sprof_step_report(nslots, prof_now() - sp_step0);
        sprof_active = 0;
    }
    return 0;
}

/* One batched decode step: K users, tokens[u] at position pos[u] of user
 * u's own sequence, in ONE pass. Each user's logits land in its own
 * sts[u]->logits (device top-k sparse presentation, same as single-user
 * decode). Slot u must be bound to sts[u]. Does NOT advance any position
 * bookkeeping (the pn_model adapter does). Returns 0 on success. */
/* QWEN35_TP decode step of one bound slot: the TP walk at n = 1 on the live
 * buffers (the slot row is parking only under TP: qw_ssm_seed_shared and
 * gpu_attn_kv_cache_sync pull it in when live is stale), the slots head, then
 * the advanced state parked back in the row. Bind records slot_pos = pos; the
 * caller (pn_qwen35moe_decode_slots_rows) bumps it with the row. */
static int tp_decode_slot_step(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, uint32_t token,
                               pn_qwen35moe_state *st, int pos, int row, int nslots)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    fc_cuda_ctx *ctx = NULL;
    gpu_batch_scratch *b = NULL;
    gpu_die_scratch *sc = NULL;
    pn_qwen35moe_state *sts[1] = { st };

    if (nslots != 1) {
        fprintf(stderr, "[qwen35moe] QWEN35_TP: one decode slot per step, not %d\n", nslots);
        return -1;
    }
    if ((st->n_attn > 0 && pos + 1 > st->kv[0].max_seq) ||
        tp_batch_walk(m, g, &token, 1, st, pos, 1, &ctx, &b, &sc) < 0)
        return -1;
    tp_commit_row(g, st, pos);
    if (gpu_head_from_slots(g, cfg, ctx, b, sc, sts, 1, 1, 0.0) < 0)
        return -1;
    return pn_qwen35moe_decode_slot_bind_cuda(m, st, pos, row, g->n_slots);
}

/* Device work of one layer of stage s on both ranks for a MoE QWEN35_TP
 * decode-slots step: the slots core of gpu_layer_forward_slots (on-die flash
 * attention over the rank's KV heads' slot rings with the plan pl[p] that
 * attn_slots_flash_prep made / the slot GDN step over the rank's heads' state
 * rows, local dims) up to the core output, then tp_layer_tail. No host work, so
 * a stage of these captures as one graph. */
static int tp_slots_layer_dev(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s, int layer,
                              gpu_batch_scratch *const b[2], fc_buffer *const pos_g[2],
                              const qw_attn_slots_plan *pl, int nslots)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const pn_qwen35moe_config *lc = &g->tp_cfg;
    const int c = cfg->n_embd;
    const int idx = pn_qwen35moe_model_layer_index(m, layer);
    const int is_attn = pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN;
    fc_buffer *core[2];

    for (int p = 0; p < 2; p++) {
        fc_cuda_ctx *ctx = g->ctx[2 * s + p];
        if (is_attn) {
            gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
            fc_rms_norm_nx_ctx(ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, nslots);
            if (attn_slots_flash_core(d, lc, b[p], pos_g[p], nslots, &pl[p]) != 0)
                return -1;
            core[p] = b[p]->a_out;
        } else {
            gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
            fc_rms_norm_nx_ctx(ctx, b[p]->hidden, d->attn_norm.buf, b[p]->x_norm, c,
                               cfg->rms_norm_eps, nslots);
            if (gpu_ssm_block_slots(d, lc, b[p], 0, nslots, 1) < 0)
                return -1;
            core[p] = b[p]->s_gated;
        }
    }
    return tp_layer_tail(g, cfg, s, is_attn, idx, core, b, nslots);
}

/* Host work of stage s before its device work: every attention half's ring
 * re-sync + plan (pls[2 * (layer - lo) + p]) and the SSM row checks. Returns a
 * key folding everything a captured stage graph bakes in, or 0 on error. */
static uint64_t tp_slots_stage_prep(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s,
                                    gpu_batch_scratch *const b[2], fc_buffer *const pos_g[2],
                                    pn_qwen35moe_state *const *sts, const int *pos,
                                    int nslots, qw_attn_slots_plan *pls)
{
    const pn_qwen35moe_config *lc = &g->tp_cfg;
    const int lo = g->layer_lo[2 * s], hi = g->layer_hi[2 * s];
    const int total_slots = nslots > g->n_slots ? nslots : g->n_slots;
    uint64_t key = 1469598103934665603ULL;
#define TP_KEY(v) (key = (key ^ (uint64_t)(v)) * 1099511628211ULL)
    TP_KEY(nslots);
    TP_KEY(g->tp_gen);
    for (int p = 0; p < 2; p++) {
        TP_KEY(b[p]->gen);
        TP_KEY((uintptr_t)b[p]);
    }
    for (int layer = lo; layer < hi; layer++) {
        const int idx = pn_qwen35moe_model_layer_index(m, layer);
        for (int p = 0; p < 2; p++) {
            if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
                gpu_attn_layer *d = &g->attn[idx + p * g->tp_n_attn];
                qw_attn_slots_plan *pl = &pls[2 * (layer - lo) + p];
                int rc = attn_slots_flash_prep(d, lc, idx, sts, pos, pos_g[p], b[p], 0, nslots,
                                               total_slots, pl);
                if (rc > 0)
                    fprintf(stderr, "[qwen35moe] QWEN35_TP: layer %d slot attention needs "
                            "the on-die flash path\n", layer);
                if (rc != 0)
                    return 0;
                TP_KEY(d->slots_ring_gen);
                TP_KEY(pl->launch_splits);
                TP_KEY(pl->kv_q4);
                TP_KEY(pl->max_seq);
            } else {
                gpu_ssm_layer *d = &g->ssm[idx + p * g->tp_n_ssm];
                if (!d->conv_slots || !d->ssm_slots)
                    return 0;  /* slots not bound */
                for (int u = 0; u < nslots; u++)
                    if (b[p]->rows_h[u] < 0 || b[p]->rows_h[u] >= d->slots_cap)
                        return 0;  /* a user's row was never carved */
            }
        }
    }
#undef TP_KEY
    return key ? key : 1;
}

/* Stage s's device work: replay its graph (key unchanged), capture it (second
 * step with this key; the first runs direct so every lazy buffer exists), or
 * issue it directly. A graph runs on rank 0's stream; tp_gev fences rank 1's
 * stream into it before and out of it after. */
static int tp_slots_stage_run(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, int s,
                              gpu_batch_scratch *const b[2], fc_buffer *const pos_g[2],
                              const qw_attn_slots_plan *pls, int nslots, uint64_t key)
{
    const int lo = g->layer_lo[2 * s], hi = g->layer_hi[2 * s];
    fc_cuda_ctx *c0 = g->ctx[2 * s], *c1 = g->ctx[2 * s + 1];
    fc_event *e0 = g->tp_gev[2 * s], *e1 = g->tp_gev[2 * s + 1];
    const int graph = tp_graph_on() && !tp_sync_on() && g->tp_sgraph_fail[s] < 4;

    if (graph && g->tp_sgraph[s] && g->tp_sgraph_key[s] != key) {
        fc_graph_free(g->tp_sgraph[s]);
        g->tp_sgraph[s] = NULL;
    }
    if (graph && !g->tp_sgraph[s] && g->tp_sgraph_warm[s] == key) {
        /* Capture: rank 1's stream joins rank 0's capture and rejoins at the end. */
        int ok = fc_graph_begin_ctx(c0) == 0;
        if (ok) {
            g->tp_capturing = 1;
            ok = fc_event_record_ctx(c0, e0) == 0 && fc_event_wait_ctx(c1, e0) == 0;
            for (int layer = lo; ok && layer < hi; layer++)
                ok = tp_slots_layer_dev(m, g, s, layer, b, pos_g, &pls[2 * (layer - lo)],
                                        nslots) == 0;
            ok = fc_event_record_ctx(c1, e1) == 0 && fc_event_wait_ctx(c0, e1) == 0 && ok;
            g->tp_capturing = 0;
            fc_graph *gr = fc_graph_end_ctx(c0);
            if (ok && gr) {
                g->tp_sgraph[s] = gr;
                g->tp_sgraph_key[s] = key;
                if (qw_ledger_on()) {
                    qw_ledger_stage_mode(s, 'C');
                    qw_ledger_graph_census(s, nslots, gr);
                }
            } else {
                fc_graph_free(gr);
                if (++g->tp_sgraph_fail[s] == 4)
                    fprintf(stderr, "[qwen35moe] QWEN35_TP: stage %d slots graph capture "
                            "failed 4 times; direct issue from now on\n", s);
            }
        }
    }
    if (graph && g->tp_sgraph[s]) {
        if (qw_ledger_on())
            qw_ledger_stage_mode(s, 'R');
        if (fc_event_record_ctx(c1, e1) != 0 || fc_event_wait_ctx(c0, e1) != 0 ||
            fc_graph_launch(g->tp_sgraph[s]) != 0 ||
            fc_event_record_ctx(c0, e0) != 0 || fc_event_wait_ctx(c1, e0) != 0)
            return -1;
        return 0;
    }
    g->tp_sgraph_warm[s] = key;
    if (qw_ledger_on())
        qw_ledger_stage_mode(s, 'D');
    for (int layer = lo; layer < hi; layer++)
        if (tp_slots_layer_dev(m, g, s, layer, b, pos_g, &pls[2 * (layer - lo)], nslots) < 0)
            return -1;
    return 0;
}

/* ============================================================
 * QWEN35_BS_PD_OVERLAP — stage-alternating prefill/decode overlap (MoE TP2)
 *
 * A TP2 decode step runs stage 0 on board (0,1), then stage 1 + the head on
 * board (2,3), then host sampling, so each board idles for about half of every
 * step. The prefill cursor fills those halves: it walks a prompt range through
 * the same 2-stage pipeline the wide prefill uses (chunk m on stage 1 while
 * chunk m+1 is on stage 0), but issues it a few LAYERS at a time ("pieces"),
 * and pn_qwen35moe_pd_cycle_cuda issues the pieces inside a decode step:
 *
 *   board 0+1:  S0(n)  P0 ................................  S0(n+1)  P0 ...
 *   board 2+3:  P1 ........  S1(n) + head  P1 ...........  S1(n+1) + head
 *
 * P0 is issued after S1 (it runs during S1, the head and sampling); P1 after the
 * head's logits came back (it runs during sampling and the next S0). Each die
 * has one stream, so a board runs one or the other, never both at once.
 *
 * Exactness: every layer of every chunk runs tp_layer_forward_batch on the same
 * inputs and in the same per-layer order as gpu_prefill_pipeline_run, and the
 * decode stages run tp_slots_step's bodies unchanged. Decode is on its own
 * buffer lane (QW_LANE_DECODE), so the only state the two share is the host
 * staging set bhost[0], and every use of it is a synchronous copy or an async
 * upload the driver stages before returning (fc_buffer_from_cpu_n_async).
 *
 * A piece is one layer of one chunk on one stage (~36 ms for a GDN layer of 128
 * tokens, ~2x for an attention layer at ~10k depth). The caller says how many
 * pieces each board may take per cycle (batch_sched_pd_budget.h: one).
 * ============================================================ */

struct pn_qwen35moe_pd_cursor {
    pn_qwen35moe_model *m;
    pn_qwen35moe_gpu *g;
    pn_qwen35moe_state *s;
    pn_state *state;
    const uint32_t *tokens;
    int n_total, pos0, chunk, n_chunks;
    int injected_max;            /* chunks allowed into stage 0 (lowered on abort) */
    int next_inject;             /* next chunk to enter stage 0 */
    int ck[2];                   /* chunk on stage s, -1 = none */
    int layer[2];                /* layers of ck[s] issued on stage s */
    int handed;                  /* chunks handed to stage 1 */
    int retired;                 /* chunks every layer of stage 1 was issued for */
    int normed_last;             /* the last chunk's x_norm is already on die 3 */
    int aborted;
    gpu_batch_scratch *bs[4];
    gpu_die_scratch *ds[4];
    double ph_ms[PN_QWEN35MOE_PD_PHASES];   /* the last cycle's host time per phase */
};

static int qw_pd_stage_layers(const pn_qwen35moe_gpu *g, int s)
{
    return g->layer_hi[2 * s] - g->layer_lo[2 * s];
}

/* Upload a staged host row block to a die: async when the driver allows it (the
 * pageable source is staged before the call returns, so bhost can be reused at
 * once), else the blocking copy. */
static void qw_pd_upload(fc_buffer *dst, const float *src, size_t bytes)
{
    if (fc_buffer_from_cpu_n_async(dst, src, bytes) != 0)
        fc_buffer_from_cpu_n(dst, src, bytes);
}

/* Chunk next_inject enters stage 0: embedded on the host, uploaded to both ranks. */
static int qw_pd_inject(pn_qwen35moe_pd_cursor *cur)
{
    pn_qwen35moe_gpu *g = cur->g;
    const int c = pn_qwen35moe_model_config(cur->m)->n_embd;
    qw_pipe_chunk k = qw_pipe_chunk_at(cur->tokens, cur->n_total, cur->pos0, cur->chunk,
                                       cur->next_inject);
    const size_t bytes = (size_t)k.n * c * sizeof(float);
    if (gpu_select_die(g, g->die[0]) < 0)
        return -1;
    for (int t = 0; t < k.n; t++)
        pn_qwen35moe_model_embed_row(cur->m, k.tokens[t], g->bhost[0] + (size_t)t * c);
    qw_pd_upload(cur->bs[0]->hidden, g->bhost[0], bytes);
    qw_pd_upload(cur->bs[1]->hidden, g->bhost[0], bytes);
    cur->ck[0] = cur->next_inject++;
    cur->layer[0] = 0;
    return 0;
}

/* Stage 1 issued the last layer of its chunk: feed the MTP head the chunk's
 * normed rows (as gpu_prefill_pipeline_run does) and retire it. The final
 * chunk's head is left to pn_qwen35moe_pd_finish_cuda. */
static void qw_pd_stage1_chunk_end(pn_qwen35moe_pd_cursor *cur)
{
    pn_qwen35moe_gpu *g = cur->g;
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(cur->m);
    const int d = g->n_die - 1;
    qw_pipe_chunk k = qw_pipe_chunk_at(cur->tokens, cur->n_total, cur->pos0, cur->chunk,
                                       cur->ck[1]);
    if (g->mtp.on && qw_pipe_mtp_feed_on()) {
        fc_rms_norm_nx_ctx(g->ctx[d], cur->bs[d]->hidden, g->output_norm.buf,
                           cur->bs[d]->x_norm, cfg->n_embd, cfg->rms_norm_eps, k.n);
        mtp_capture_rows(g, g->ctx[d], cur->bs[d]->x_norm, k.n, cfg->n_embd);
        (void)mtp_push(cur->m, g, cur->s, k.pos0 + k.n, k.tokens, k.n);
        if (cur->ck[1] == cur->n_chunks - 1)
            cur->normed_last = 1;
    }
    cur->retired++;
    cur->ck[1] = -1;
    cur->layer[1] = 0;
}

/* Issue up to max_pieces of stage s's next layers on the prefill lane (every
 * ready layer when max_pieces < 0). Stage 0 injects a new chunk only when its
 * last one was handed on; stage 1 picks up a handed chunk. Returns the number
 * of layers issued, or -1. */
static int qw_pd_issue_stage(pn_qwen35moe_pd_cursor *cur, int s, int max_pieces)
{
    pn_qwen35moe_gpu *g = cur->g;
    const int lane = g->lane;
    const int nl = qw_pd_stage_layers(g, s);
    int issued = 0;

    g->lane = QW_LANE_PREFILL;
    for (;;) {
        if (cur->ck[s] < 0) {
            if (s == 0) {
                if (cur->next_inject >= cur->injected_max || qw_pd_inject(cur) < 0)
                    break;
            } else {
                if (cur->handed <= cur->retired)
                    break;
                cur->ck[1] = cur->retired;
                cur->layer[1] = 0;
            }
        }
        if (cur->layer[s] >= nl)
            break;                       /* stage 0: waiting for its hand-off */
        if (max_pieces >= 0 && issued >= max_pieces)
            break;
        qw_pipe_chunk k = qw_pipe_chunk_at(cur->tokens, cur->n_total, cur->pos0, cur->chunk,
                                           cur->ck[s]);
        gpu_batch_scratch *pair[2] = { cur->bs[2 * s], cur->bs[2 * s + 1] };
        if (tp_layer_forward_batch(cur->m, g, s, g->layer_lo[2 * s] + cur->layer[s], pair,
                                   cur->s, k.pos0, k.n, 0) < 0) {
            g->lane = lane;
            return -1;
        }
        cur->layer[s]++;
        issued++;
        if (s == 1 && cur->layer[1] == nl)
            qw_pd_stage1_chunk_end(cur);
    }
    g->lane = lane;
    return issued;
}

/* Hand stage 0's finished chunk to stage 1 once stage 1 has issued all of its
 * own: rank 0's rows through the host to both stage-1 ranks, the uploads queued
 * behind stage 1's issued work. Blocks only on die 0 / die 1's streams.
 * Returns 1 when a chunk moved, 0 when none could, -1 on error. */
static int qw_pd_hand_off(pn_qwen35moe_pd_cursor *cur)
{
    pn_qwen35moe_gpu *g = cur->g;
    if (cur->ck[0] < 0 || cur->layer[0] < qw_pd_stage_layers(g, 0))
        return 0;
#ifndef PN_PDO_RED_NO_HOP_FENCE
    if (cur->ck[1] >= 0 || cur->handed > cur->retired)
        return 0;
#else
    /* Red build: the hop may overwrite stage 1's input while stage 1 still has
     * layers of the previous chunk to issue; the engine gate must fail. */
    if (cur->handed > cur->retired + 1)
        return 0;
#endif
    const int c = pn_qwen35moe_model_config(cur->m)->n_embd;
    qw_pipe_chunk k = qw_pipe_chunk_at(cur->tokens, cur->n_total, cur->pos0, cur->chunk,
                                       cur->ck[0]);
    const size_t bytes = (size_t)k.n * c * sizeof(float);
    fc_sync_ctx(g->ctx[0]);
    fc_buffer_to_cpu_n(cur->bs[0]->hidden, g->bhost[0], bytes);
    qw_pd_upload(cur->bs[2]->hidden, g->bhost[0], bytes);
    qw_pd_upload(cur->bs[3]->hidden, g->bhost[0], bytes);
    cur->handed++;
    cur->ck[0] = -1;
    cur->layer[0] = 0;
    return 1;
}

/* MoE QWEN35_TP decode-slots step over nslots bound rows (g->step_rows): the
 * walk of tp_batch_walk with the slots bodies (stage hop host-staged, head on
 * die 3), then the slots head. The slot rows are the live state, as on the
 * layer split; the caller bumps slot_pos. With a prefill cursor (overlap
 * cycle, decode lane), the cursor's pending hand-off happens while die 0 is
 * drained at the stage hop, stage 0's pieces are issued after stage 1's work
 * and stage 1's after the head (up to `pieces` each); the solo-step MTP row capture
 * is skipped (the head's backlog belongs to the prefilling session). */
static int tp_slots_step(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, const uint32_t *tokens,
                         pn_qwen35moe_state *const *sts, const int *pos, int nslots,
                         pn_qwen35moe_pd_cursor *cur, int pieces)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const int c = cfg->n_embd;
    const size_t hidden_bytes = (size_t)nslots * c * sizeof(float);
    gpu_batch_scratch *b[2] = { NULL, NULL };
    fc_buffer *pos_g[2] = { NULL, NULL };
    gpu_die_scratch *sc = NULL;
    qw_attn_slots_plan *pls = calloc((size_t)2 * cfg->n_layer, sizeof(*pls));
    const int lg = qw_ledger_on();
    int rc = -1;

    if (!pls)
        return -1;
    if (lg)
        qw_ledger_step_begin(nslots, pos, cur != NULL);
    double t_ph = cur ? prof_now() : 0.0;
#define QW_PD_PHASE(i) do { if (cur) { const double t_ = prof_now(); \
                                     cur->ph_ms[i] = 1000.0 * (t_ - t_ph); t_ph = t_; } } while (0)
    for (int u = 0; u < nslots; u++)
        if (sts[u]->n_attn > 0 && pos[u] + 1 > sts[u]->kv[0].max_seq)
            goto out;
    if (tp_exchange_ready(g, cfg, nslots) < 0)
        goto out;
    for (int s = 0; s < 2; s++) {
        gpu_batch_scratch *prev = b[0];
        for (int p = 0; p < 2; p++) {
            const int die = 2 * s + p;
            b[p] = gpu_get_batch_scratch(g, g->ctx[die], cfg, nslots);
            sc = gpu_get_scratch(g, g->ctx[die], cfg);
            pos_g[p] = b[p] ? gpu_pos_buf_upload(g, b[p], g->ctx[die], die, pos, g->step_rows,
                                                 nslots)
                            : NULL;
            if (!b[p] || !sc || !pos_g[p])
                goto out;
        }
        if (s == 0) {
            float *rows = qw_stage_input_rows(g, cfg, hidden_bytes);
            if (!rows)
                goto out;
            for (int u = 0; u < nslots; u++)
                pn_qwen35moe_model_embed_row(m, tokens[u], rows + (size_t)u * c);
#ifdef PN_PDO_RED
            /* Red build, PN_PDO_RED_ULP=1: the exactness gates must see a one-ulp change
             * on the decode lane. */
            if (g->lane == QW_LANE_DECODE && getenv("PN_PDO_RED_ULP"))
                rows[0] = nextafterf(rows[0], INFINITY);
#endif
        } else {
            QW_PD_PHASE(0);
            if (lg)
                qw_ledger_lap(QW_LH_MIRROR0);
            if (qw_stage_hop_host_wait(g, cfg, hidden_bytes))
                fc_sync_ctx(g->ctx[0]);
            QW_PD_PHASE(1);
            if (lg) {
                qw_ledger_lap(QW_LH_WAIT0);
                qw_ledger_inject("hop");
            }
            if (cur && qw_pd_hand_off(cur) < 0)
                goto out;
        }
        if (qw_stage_input_upload(g, cfg, s, prev, b, hidden_bytes) < 0)
            goto out;
        if (s == 1)
            QW_PD_PHASE(2);
        if (lg && s == 1)
            qw_ledger_lap(QW_LH_HOP);
        uint64_t key = tp_slots_stage_prep(m, g, s, b, pos_g, sts, pos, nslots, pls);
        if (lg) {
            qw_ledger_lap(s == 0 ? QW_LH_SETUP0 : QW_LH_SETUP1);
            qw_ledger_gpu(s == 0 ? QW_LG_S0_BEG : QW_LG_S1_BEG, g->ctx[2 * s]);
        }
        if (!key || tp_slots_stage_run(m, g, s, b, pos_g, pls, nslots, key) < 0) {
            fprintf(stderr, "[qwen35moe] QWEN35_TP slots step failed on stage %d (n=%d)\n", s,
                    nslots);
            goto out;
        }
        if (lg) {
            qw_ledger_gpu(s == 0 ? QW_LG_S0_END : QW_LG_S1_END, g->ctx[2 * s]);
            qw_ledger_lap(s == 0 ? QW_LH_LAUNCH0 : QW_LH_LAUNCH1);
            if (s == 0)
                qw_ledger_inject("shadow");
        }
        /* The host KV mirror (lazy under TP: marks it stale). */
        for (int layer = g->layer_lo[2 * s]; layer < g->layer_hi[2 * s]; layer++) {
            if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
                continue;
            const int idx = pn_qwen35moe_model_layer_index(m, layer);
            for (int p = 0; p < 2; p++)
                attn_slots_mirror(g, &g->attn[idx + p * g->tp_n_attn], &g->tp_cfg, idx, sts,
                                  pos, b[p], nslots,
                                  &pls[2 * (layer - g->layer_lo[2 * s]) + p], g->bhost[3]);
        }
        if (s == 1)
            QW_PD_PHASE(3);
        if (cur && s == 1 && qw_pd_issue_stage(cur, 0, pieces) < 0)
            goto out;
        if (s == 1)
            QW_PD_PHASE(4);
    }
    if (lg)
        qw_ledger_lap(QW_LH_MIRROR1);
    g->pd_cycle = cur != NULL;
    rc = gpu_head_from_slots(g, cfg, g->ctx[3], b[1], sc, sts, nslots, 1, 0.0);
    g->pd_cycle = 0;
    QW_PD_PHASE(5);
    if (rc == 0 && cur && qw_pd_issue_stage(cur, 1, pieces) < 0)
        rc = -1;
    QW_PD_PHASE(6);
#undef QW_PD_PHASE
out:
    if (lg)
        qw_ledger_step_end(rc);
    free(pls);
    return rc;
}

/* The checks and host staging every decode-slots step needs before any device
 * work: each user's row is in range and bound to its state (g->step_rows[u] =
 * the row), the host scratch exists and the host staging holds nslots rows. */
static int qw_slots_step_prepare(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg,
                                 const uint32_t *tokens, pn_qwen35moe_state *const *sts,
                                 const int *rows, int nslots)
{
    /* rows[u] = the persistent slot row user u was bound to (NULL = u). */
    for (int u = 0; u < nslots; u++) {
        int r = rows ? rows[u] : u;
        if (r < 0 || r >= g->n_slots || (int)tokens[u] >= cfg->n_vocab ||
            !sts[u] || g->slot_state[r] != (const void *)sts[u]) {
            fprintf(stderr, "[qwen35moe] decode_slots: user %d row %d rejected "
                    "(n_slots %d, token %u / vocab %d, %s)\n", u, r, g->n_slots,
                    tokens[u], cfg->n_vocab,
                    (r >= 0 && r < g->n_slots && sts[u] &&
                     g->slot_state[r] != (const void *)sts[u])
                        ? "row not bound to this state" : "out of range");
            return -1;
        }
        g->step_rows[u] = r;
    }

    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0) {
            fprintf(stderr, "[qwen35moe] decode_slots: host scratch alloc "
                    "failed\n");
            return -1;
        }
        g->host_ok = 1;
    }
    if (gpu_get_batch_host(g, cfg, nslots) < 0) {
        fprintf(stderr, "[qwen35moe] decode_slots: host staging for %d rows "
                "failed\n", nslots);
        return -1;
    }
    return 0;
}

int pn_qwen35moe_decode_slots_tokens_cuda(pn_qwen35moe_model *m,
                                          const uint32_t *tokens,
                                          pn_qwen35moe_state *const *sts,
                                          const int *pos, const int *rows,
                                          int nslots)
{
    if (!m || !tokens || !sts || !pos || nslots <= 0 ||
        nslots > PN_QWEN35MOE_GPU_MAX_SLOTS)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return -1;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int n_layer = cfg->n_layer;
    if (qw_slots_step_prepare(g, cfg, tokens, sts, rows, nslots) < 0)
        return -1;
    int two_die = (g->head_ctx != NULL);
    int split = g->split_layer;

    if (g->tp && cfg->n_expert) {
        g->lane = qw_pd_overlap_on() ? QW_LANE_DECODE : QW_LANE_PREFILL;
        int rc = tp_slots_step(m, g, tokens, sts, pos, nslots, NULL, 0);
        g->lane = QW_LANE_PREFILL;
        return rc;
    }
    if (g->tp)
        return tp_decode_slot_step(m, g, tokens[0], sts[0], pos[0], g->step_rows[0], nslots);

    /* QWEN35_ASYNC_KVMIRROR: pre-size + reset each active die's deferred-mirror
     * arena for this step (see qw_kv_defer). Pre-sizing to the full step up
     * front means the arena never reallocs mid-step, so in-flight async DtoH
     * destinations stay valid. On OOM, disable the deferral for this step by
     * leaving the arenas at n=0 (the flash path then falls back to blocking). */
    if (async_kvmirror_on()) {
        int kv_dim0 = cfg->n_head_kv * cfg->head_dim;
        for (int d = 0; d < g->n_die; d++)
            if (qw_defer_reset(&g->defer[d], g->n_attn, nslots, kv_dim0) < 0)
                g->defer[d].stage_bytes = 0;  /* used>bytes => stage() fails */
    }

    /* M2 slots-path profiler: arm the phase brackets for this step. */
    double sp_step0 = 0.0, sp_t0 = 0.0;
    if (sprof_on()) {
        sprof_active = 1;
        memset(sprof_step_ph, 0, sizeof(sprof_step_ph));
        sp_step0 = prof_now();
    }

    /* ============================================================
     * L5 — two-die-pair interleave (QWEN35_DIE_PIPELINE). Software-pipeline
     * two batch groups so both physical dies compute at once: group0-die-B
     * overlaps group1-die-A (separate contexts = the two dies run in
     * parallel). Bit-exact — independent users, unchanged per-user math, only
     * the launch ordering + per-group slot_base ring offsetting differ.
     * Off / not exactly two dies / <2 slots / sprof-on falls to the sequential
     * path. The interleave is a TWO-stage software pipeline by construction —
     * one front group overlapping one back group — so it runs at exactly two
     * dies; a wider split would need a deeper pipeline with as many groups as
     * stages, which is separate work and would not pay on this hardware (each
     * extra stage adds a boundary hop, and the measured 2-die overlap is only
     * 1.33-1.44x to begin with).
     * Requires small-BAR mode (g->bs == NULL): the two contexts are separate
     * real CUDA devices that run concurrently. Under legacy barswap the dies
     * time-share one MMIO slot (no true overlap), so the pipeline is skipped.
     * ============================================================ */
    if (g->n_die == 2 && !g->bs && die_pipeline_on() && nslots >= 2 &&
        !sprof_active) {
        int h = nslots / 2;                 /* group0 = [0,h), group1 = [h,K) */
        int n0 = h, n1 = nslots - h;
        /* Ensure the second host-staging set (group1) exists. */
        if (!g->bhost2[0]) {
            size_t per_pos = 2 * (size_t)cfg->head_dim * cfg->n_head;
            if (per_pos < (size_t)cfg->conv_dim)
                per_pos = (size_t)cfg->conv_dim;
            for (int i = 0; i < 4; i++) {
                g->bhost2[i] = malloc((size_t)g->bhost_cap * per_pos *
                                      sizeof(float));
                if (!g->bhost2[i])
                    return -1;
            }
        }
        float *const *hb0 = g->bhost;       /* group0 staging */
        float *const *hb1 = g->bhost2;      /* group1 staging */

        gpu_batch_scratch *ba = gpu_get_batch_scratch(g, g->ctx[0], cfg, nslots);
        gpu_batch_scratch *bb = gpu_get_batch_scratch(g, g->head_ctx, cfg, nslots);
        gpu_die_scratch *sc_b = gpu_get_scratch(g, g->head_ctx, cfg);
        if (!ba || !bb || !sc_b)
            return -1;

        /* --- group0 front on die A --- */
        if (gpu_select_die(g, g->die[0]) < 0)
            return -1;
        fc_buffer *pos_a = gpu_pos_buf_upload(g, ba, g->ctx[0], 0, pos,
                                              g->step_rows, n0);
        if (!pos_a ||
            gpu_group_front(m, g, cfg, tokens, sts, pos, ba, pos_a,
                            0, n0, nslots, hb0, split) < 0)
            return -1;
        /* handoff group0 A->B: sync die A, D2H group0 hidden into hb0[0]. */
        fc_sync_ctx(g->ctx[0]);
        fc_buffer_to_cpu_n(ba->hidden, hb0[0], (size_t)n0 * c * sizeof(float));

        /* --- launch group1 front on die A (async, NO sync) --- */
        pos_a = gpu_pos_buf_upload(g, ba, g->ctx[0], 0, pos + h,
                                   g->step_rows + h, n1);
        if (!pos_a ||
            gpu_group_front(m, g, cfg, tokens, sts, pos, ba, pos_a,
                            h, n1, nslots, hb1, split) < 0)
            return -1;

        /* --- group0 back on die B (async) — runs on the OTHER physical die
         * concurrently with group1's die-A work just launched above --- */
        if (gpu_select_die(g, g->head_die) < 0)
            return -1;
        fc_buffer *pos_b = gpu_pos_buf_upload(g, bb, g->head_ctx, 1, pos,
                                              g->step_rows, n0);
        if (!pos_b ||
            gpu_group_back(m, g, cfg, sts, pos, bb, pos_b,
                           0, n0, nslots, hb0, split, n_layer) < 0)
            return -1;

        /* group0 head (die B) — sync die B first, then norm+head+topk. */
        fc_sync_ctx(g->head_ctx);
        if (gpu_group_head(g, g->head_ctx, cfg, bb, sc_b, sts, 0, n0) < 0)
            return -1;

        /* --- handoff group1 A->B: sync die A (group1 front done), D2H --- */
        if (gpu_select_die(g, g->die[0]) < 0)
            return -1;
        fc_sync_ctx(g->ctx[0]);
        fc_buffer_to_cpu_n(ba->hidden, hb1[0], (size_t)n1 * c * sizeof(float));

        /* --- group1 back on die B, then group1 head --- */
        if (gpu_select_die(g, g->head_die) < 0)
            return -1;
        pos_b = gpu_pos_buf_upload(g, bb, g->head_ctx, 1, pos + h,
                                   g->step_rows + h, n1);
        if (!pos_b ||
            gpu_group_back(m, g, cfg, sts, pos, bb, pos_b,
                           h, n1, nslots, hb1, split, n_layer) < 0)
            return -1;
        fc_sync_ctx(g->head_ctx);
        if (gpu_group_head(g, g->head_ctx, cfg, bb, sc_b, sts, h, n1) < 0)
            return -1;
        /* Replay both dies' deferred host-KV mirror stores (both dies are drained
         * by their group syncs above) before returning, so the mirror is current
         * for the next step's bind/move/ring re-sync. */
        if (async_kvmirror_on()) {
            qw_defer_flush(g, 0, g->ctx[0]);
            qw_defer_flush(g, 1, g->head_ctx);
        }
        return 0;
    }

    /* ============================================================
     * QWEN35_TWO_THREAD — 2-host-thread layer-split driver. Runs the SAME
     * front(die A)/back(die B) layer split as the sequential path below, but
     * each die's layer range is issued from its own persistent worker thread so
     * the two dies' launch queues are fed in parallel (removes the single-thread
     * launch-starvation term). The die-A->die-B boundary hidden handoff is a
     * real data dependency: worker A must finish before the handoff, which must
     * finish before worker B starts — so for a single layer-split step the two
     * dies do NOT overlap (see the honest note at struct qw_tt). Bit-exact to
     * the sequential path (identical op sequence per die; only the host-side
     * issuer differs). Only EXACTLY two dies + small-BAR (g->bs==NULL) — the
     * worker pool is a pair, one thread per die. After the back layers this
     * sets ctx/b/sc to die B and FALLS THROUGH to the shared head.
     * ============================================================ */
    if (two_thread_on() && g->n_die == 2 && !g->bs) {
        struct qw_tt *tt = qw_tt_ensure(g);
        if (tt) {
            gpu_batch_scratch *ba = gpu_get_batch_scratch(g, g->ctx[0], cfg,
                                                          nslots);
            gpu_die_scratch *sc_a = gpu_get_scratch(g, g->ctx[0], cfg);
            fc_buffer *pos_a = ba ? gpu_pos_buf_upload(g, ba, g->ctx[0], 0, pos,
                                                       g->step_rows, nslots)
                                  : NULL;
            if (!ba || !sc_a || !pos_a)
                return -1;

            /* Embed + seed die A's residual (control thread; die A stream idle). */
            for (int u = 0; u < nslots; u++)
                pn_qwen35moe_model_embed_row(m, tokens[u],
                                             g->bhost[0] + (size_t)u * c);
            fc_buffer_from_cpu_n(ba->hidden, g->bhost[0],
                                 (size_t)nslots * c * sizeof(float));

            /* Worker A: front layers [0,split) on die A, full K. */
            tt->m = m; tt->g = g; tt->cfg = cfg; tt->sts = sts; tt->pos = pos;
            tt->nslots = nslots;
            tt->bscr[0] = ba; tt->pos_g[0] = pos_a;
            tt->layer_lo[0] = 0; tt->layer_hi[0] = split;
            qw_tt_post(tt, 0);
            if (qw_tt_wait(tt, 0) != 0)
                return -1;

            /* Boundary hand-off (control thread): die A drained by its worker's
             * fc_sync_ctx. D2H the boundary hidden, flush die A's KV mirror, then
             * seed die B and post worker B for the back layers. */
            fc_buffer_to_cpu_n(ba->hidden, g->bhost[0],
                               (size_t)nslots * c * sizeof(float));
            if (async_kvmirror_on())
                qw_defer_flush(g, 0, g->ctx[0]);

            gpu_batch_scratch *bb = gpu_get_batch_scratch(g, g->head_ctx, cfg,
                                                          nslots);
            gpu_die_scratch *sc_b = gpu_get_scratch(g, g->head_ctx, cfg);
            fc_buffer *pos_b = bb ? gpu_pos_buf_upload(g, bb, g->head_ctx, 1, pos,
                                                       g->step_rows, nslots)
                                  : NULL;
            if (!bb || !sc_b || !pos_b)
                return -1;
            fc_buffer_from_cpu_n(bb->hidden, g->bhost[0],
                                 (size_t)nslots * c * sizeof(float));

            tt->bscr[1] = bb; tt->pos_g[1] = pos_b;
            tt->layer_lo[1] = split; tt->layer_hi[1] = n_layer;
            qw_tt_post(tt, 1);
            if (qw_tt_wait(tt, 1) != 0)
                return -1;

            if (async_kvmirror_on())
                qw_defer_flush(g, 1, g->head_ctx);

            /* Fall through to the shared final-norm + LM head on die B. */
            gpu_head_from_slots(g, cfg, g->head_ctx, bb, sc_b, sts, nslots, two_die,
                                sp_step0);
            return 0;
        }
        /* qw_tt_ensure failed: fall through to the sequential path. */
    }

    /* ---- Sequential N-die walk: die 0 embeds all K tokens, each die runs
     * its layer range against its own scratch, and each boundary hands the
     * [K][c] activations to the next die. ---- */
    bv_trace_slots_pos0 = pos[0];
    fc_cuda_ctx *ctx = NULL;
    gpu_batch_scratch *b = NULL;
    gpu_die_scratch *sc = NULL;
    size_t hidden_bytes = (size_t)nslots * c * sizeof(float);

    for (int d = 0; d < g->n_die; d++) {
        if (d > 0) {
            sprof_begin(g->ctx[d - 1], &sp_t0);
            gpu_handoff_export(g, d - 1, b->hidden, g->bhost[0], hidden_bytes);
            /* The die we are leaving is drained by the export's sync, and is
             * still the selected one: replay its deferred host-KV mirror
             * stores here, so the mirror is current before any boundary
             * consumer reads it. */
            if (async_kvmirror_on())
                qw_defer_flush(g, d - 1, g->ctx[d - 1]);
        }
        if (gpu_select_die(g, g->die[d]) < 0)
            return -1;
        gpu_batch_scratch *bd = gpu_get_batch_scratch(g, g->ctx[d], cfg, nslots);
        gpu_die_scratch *scd = gpu_get_scratch(g, g->ctx[d], cfg);
        fc_buffer *pos_d = bd ? gpu_pos_buf_upload(g, bd, g->ctx[d], d, pos,
                                                   g->step_rows, nslots)
                               : NULL;
        if (!bd || !scd || !pos_d) {
            fprintf(stderr, "[qwen35moe] decode_slots: die %d scratch for %d "
                    "rows unavailable (batch %s, die %s, pos %s)\n", g->die[d],
                    nslots, bd ? "ok" : "FAILED", scd ? "ok" : "FAILED",
                    pos_d ? "ok" : "FAILED");
            return -1;
        }

        if (d == 0) {
            sprof_begin(g->ctx[d], &sp_t0);
            for (int u = 0; u < nslots; u++)
                pn_qwen35moe_model_embed_row(m, tokens[u],
                                             g->bhost[0] + (size_t)u * c);
            fc_buffer_from_cpu_n(bd->hidden, g->bhost[0], hidden_bytes);
        } else {
            gpu_handoff_import(g, d - 1, b->hidden, bd->hidden, g->bhost[0],
                               hidden_bytes);
        }
        sprof_lap(SPROF_XFER, g->ctx[d], &sp_t0);

        for (int layer = g->layer_lo[d]; layer < g->layer_hi[d]; layer++) {
            if (gpu_layer_forward_slots(m, g, g->ctx[d], layer, bd, sts, pos,
                                        pos_d, 0, nslots, nslots,
                                        g->bhost) < 0) {
                fprintf(stderr, "[qwen35moe] batched decode step failed on "
                        "die %d layer %d (n=%d)\n", g->die[d], layer, nslots);
                return -1;
            }
            bv_trace_rows(g->ctx[d], "plain", "out", layer, bd->hidden, nslots,
                          c, pos, 0);
        }
        ctx = g->ctx[d];
        b = bd;
        sc = scd;
    }

    /* QWEN35_ASYNC_KVMIRROR: replay the LAST die's deferred host-KV mirror
     * stores (every earlier die was flushed at its boundary hand-off above).
     * Done before the head/return so the mirror is current for the next step's
     * bind/move/re-sync. */
    if (async_kvmirror_on())
        qw_defer_flush(g, g->n_die - 1, ctx);

    /* ---- Final norm over all K rows + ONE batched LM head, then each user's
     * row reduced on-die to top-k pairs. Shared with the TWO_THREAD path. ---- */
    return gpu_head_from_slots(g, cfg, ctx, b, sc, sts, nslots, two_die,
                               sp_step0);
}

/* ============================================================
 * pn_forward_fn adapter (CUDA)
 *
 * Mirrors pn_qwen35moe_forward (the CPU adapter) but dispatches to the single-
 * die GPU token forward. The engine selects this when fc_available() and the
 * GPU weights are uploaded; otherwise it keeps the CPU adapter.
 * ============================================================ */

tm_tensor *pn_qwen35moe_forward_cuda(pn_model *model, uint32_t token,
                                     pn_state *state)
{
    if (!model || !state)
        return NULL;

    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return NULL;

    float *logits = pn_qwen35moe_forward_token_cuda(m, token, s, state->pos);
    if (!logits)
        return NULL;

    int64_t shape[1] = { s->n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);
    if (!out)
        return NULL;
    memcpy(tm_data_f32(out), logits, (size_t)s->n_vocab * sizeof(float));

    state->pos++;
    return out;
}

/* Batched prefill adapter: runs the n tokens through the batched forward in
 * ONE pass and returns the LAST position's logits (owned tensor), advancing
 * state->pos by n. per_pos_logits (optional, [n * n_vocab]) additionally
 * receives every position's full logits — the bit-exact gate hook.
 *
 * If the batched path cannot run (exotic quant / non-grouped experts), it
 * falls back to n sequential single-token forwards — same results, no
 * speedup — so callers never lose correctness. */
tm_tensor *pn_qwen35moe_forward_batch_cuda(pn_model *model,
                                           const uint32_t *tokens, int n,
                                           pn_state *state,
                                           float *per_pos_logits)
{
    if (!model || !state || !tokens || n <= 0)
        return NULL;

    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return NULL;

    float *logits = pn_qwen35moe_forward_batch_tokens_cuda(m, tokens, n, s,
                                                           state->pos,
                                                           per_pos_logits);
    if (logits) {
        state->pos += n;
    } else {
        /* Fallback: sequential single-token forwards (bit-identical by
         * definition; the batch path is purely an optimization). */
        for (int t = 0; t < n; t++) {
            logits = pn_qwen35moe_forward_token_cuda(m, tokens[t], s,
                                                     state->pos);
            if (!logits)
                return NULL;
            state->pos++;
            if (per_pos_logits)
                memcpy(per_pos_logits + (size_t)t * s->n_vocab, logits,
                       (size_t)s->n_vocab * sizeof(float));
        }
    }

    int64_t shape[1] = { s->n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);
    if (!out)
        return NULL;
    memcpy(tm_data_f32(out), logits, (size_t)s->n_vocab * sizeof(float));
    return out;
}

/* Every adapter below casts model->weights to pn_qwen35moe_model* and
 * state->kv_cache to pn_qwen35moe_state*.  The batch scheduler calls several of
 * them (forget_state, state_to_host, decode_slot_bind) on WHATEVER model owns
 * the slot — it is generic — so a GPU-placed llama model reached them and the
 * casts read another struct's fields: n_layer came out of garbage and the layer
 * loop walked off into a SIGSEGV that took the whole serve down.  Observed on
 * the K80 test machine with MiniCPM5-1B (PN_ARCH_LLAMA) under QWEN35_BATCH_SLOTS=1,
 * intermittent because the garbage was sometimes benign.
 *
 * Only the two archs that actually allocate the qwen35moe dual state
 * (pn_state_new: QWEN35MOE and BONSAI) may pass.  Everything else gets the
 * ordinary failure return, which every caller already treats as "no GPU fast
 * path for this state". */
static inline int qw_model_uses_qwen35moe_state(const pn_model *model)
{
    return model && (model->config.arch == PN_ARCH_QWEN35MOE ||
                     model->config.arch == PN_ARCH_BONSAI);
}

/* Speculative-decode adapters (see the verify/commit section above).
 * verify runs at state->pos WITHOUT advancing it; accept commits n_accept
 * positions and advances state->pos by n_accept. */
int pn_qwen35moe_spec_verify(pn_model *model, const uint32_t *tokens, int n,
                             pn_state *state, int32_t *argmax_out)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    return pn_qwen35moe_spec_verify_tokens_cuda(m, tokens, n, s, state->pos,
                                                argmax_out);
}

int pn_qwen35moe_spec_accept(pn_model *model, pn_state *state, int n_accept)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    if (!m)
        return -1;
    if (pn_qwen35moe_spec_commit_cuda(m, n_accept) != 0)
        return -1;
    state->pos += n_accept;
    return 0;
}

/* Test hook (the batched-verify exactness gate): copy row `row` of the LAST
 * verify's [n][n_vocab] LM-head output into out[n_vocab]. Every row is
 * computed by the batched head even when the argmax scan stops early. Call
 * between verify and accept. Returns 0, or -1 (no verify output / bad row). */
int pn_qwen35moe_spec_row_logits(pn_model *model, int row, float *out)
{
    if (!model || !out || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    if (!g || !g->spec_logits || row < 0 || row >= g->spec_n)
        return -1;
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    size_t row_bytes = (size_t)cfg->n_vocab * sizeof(float);
    if (gpu_select_die(g, g->die[g->n_die - 1]) < 0)
        return -1;
    fc_sync_ctx(g->ctx[g->n_die - 1]);
    fc_buffer_to_cpu_at(g->spec_logits, out, (size_t)row * row_bytes,
                        row_bytes);
    return 0;
}

/* Multi-user decode adapters (see the decode-slots section above). bind
 * snapshots `state`'s live SSM images into `slot`; step runs ONE batched
 * forward for all n_users (slot u == states[u]), fills each user's logits
 * (logits_out[u], owned by that user's state, valid until its next
 * forward) and advances each states[u]->pos by 1. */
int pn_qwen35moe_decode_slot_bind(pn_model *model, pn_state *state, int slot,
                                  int n_slots)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    return pn_qwen35moe_decode_slot_bind_cuda(m, s, (int)state->pos, slot,
                                              n_slots);
}


/* Carve the per-session decode blocks: `n_rows` persistent rows on every die
 * the model spans, each a full ctx_cap KV ring row (+ q4 sink/window shadows)
 * per attention layer and a conv + recurrent SSM image per SSM layer. Done ONCE
 * at load so the VRAM is reserved for the model's life and a request can never
 * be the thing that discovers it does not fit. `geom` supplies the KV geometry
 * (q4, max_seq). Returns 0, or -1 after naming the layer that did not fit. */
int pn_qwen35moe_decode_slots_carve(pn_model *model, pn_state *geom, int n_rows)
{
    if (!model || !geom || n_rows <= 0 || n_rows > PN_QWEN35MOE_GPU_MAX_SLOTS ||
        !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)geom->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    if (!s || !g)
        return -1;
    /* QWEN35_TP: every layer struct is a half; rows use the rank's local dims. */
    const pn_qwen35moe_config *cfg = qw_state_cfg(g, g->tp, pn_qwen35moe_model_config(m));
    size_t conv_bytes = (size_t)(cfg->ssm_conv - 1) * cfg->conv_dim *
                        sizeof(float);
    size_t ssm_bytes = (size_t)cfg->ssm_state * cfg->ssm_state *
                       cfg->ssm_dt_rank * sizeof(float);

    memset(g->slot_state, 0, sizeof(g->slot_state));
    g->n_slots = n_rows;
    for (int i = 0; i < g->n_ssm; i++) {
        gpu_ssm_layer *d = &g->ssm[i];
        if (d->slots_cap >= n_rows)
            continue;
        g->tp_gen++;   /* QWEN35_TP slots graphs bake these rows */
        fc_buffer_free(d->conv_slots);
        fc_buffer_free(d->ssm_slots);
        d->conv_slots = fc_buffer_alloc_ctx(d->ctx, (size_t)n_rows * conv_bytes);
        d->ssm_slots = fc_buffer_alloc_ctx(d->ctx, (size_t)n_rows * ssm_bytes);
        d->slots_cap = (d->conv_slots && d->ssm_slots) ? n_rows : 0;
        if (!d->slots_cap) {
            fprintf(stderr, "[qwen35moe] carve: SSM layer %d: no VRAM for %d "
                    "session blocks (%zu MiB)\n", i, n_rows,
                    ((size_t)n_rows * (conv_bytes + ssm_bytes)) >> 20);
            return -1;
        }
        if (d->slots_graph) {
            fc_graph_free(d->slots_graph);
            d->slots_graph = NULL;
        }
        pn_graph_capture_dropped(&d->slots_graph_capture);
        d->slots_graph_n = 0;
    }
    for (int i = 0; i < g->n_attn; i++) {
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        if (gpu_attn_slot_ring_ensure(&g->attn[i], cfg, kv->q4, kv->max_seq,
                                      n_rows, qw_attn_hidx(g, i)) != 0) {
            fprintf(stderr, "[qwen35moe] carve: attention layer %d: no VRAM "
                    "for %d session blocks of %d tokens\n", i, n_rows,
                    kv->max_seq);
            return -1;
        }
    }
    return 0;
}

/* Does decode row `row` still hold `state` exactly as committed — its SSM
 * image at state->pos and every attention layer's KV rows [0, pos) — so a
 * resumed turn can continue from the card without touching host RAM? Pure
 * bookkeeping, no device work. */
int pn_qwen35moe_slot_holds_state(pn_model *model, int row, pn_state *state)
{
    if (!model || !state || row < 0 || !qw_model_uses_qwen35moe_state(model))
        return 0;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)state->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    if (!s || !g || row >= g->n_slots || g->slot_state[row] != (const void *)s)
        return 0;
    const int pos = (int)state->pos;
    if (g->slot_pos[row] != pos ||
        g->slot_gen[row] != (s->n_attn > 0 ? s->kv[0].content_gen : 0u))
        return 0;
    for (int i = 0; i < g->n_attn; i++) {
        const gpu_attn_layer *d = &g->attn[i];
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        const void *id = kv->q4 ? (const void *)kv->kq : (const void *)kv->k;
        int need = pos < kv->max_seq ? pos : kv->max_seq;
        if (row >= d->kv_slots_cap_n || d->kv_slot_host_id[row] != id ||
            d->kv_slot_gen[row] != kv->content_gen ||
            d->kv_slot_synced[row] < need)
            return 0;
    }
    return 1;
}

/* BENCH ONLY (tests/bench_depth_qwen35, QWEN35_DEPTH_FAKE): advance a bound,
 * already-stepped decode row by `add` positions WITHOUT computing them, and mark
 * its ring rows synced so the next step does not seed them. The skipped KV rows
 * hold whatever the ring held, so the logits are meaningless; what it buys is the
 * attention cost of a deep context without the hours of prefill. Never called by
 * the serve. Returns 0, or -1 if the row is not current for this state. */
int pn_qwen35moe_bench_skip_positions(pn_model *model, pn_state *state, int row,
                                      int add)
{
    if (add <= 0 || !pn_qwen35moe_slot_holds_state(model, row, state))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    const int pos = (int)state->pos;
    if (s->n_attn > 0 && pos + add + 1 > s->kv[0].max_seq)
        return -1;
    for (int i = 0; i < g->n_attn; i++)
        g->attn[i].kv_slot_synced[row] = pos + add;
    g->slot_pos[row] += add;
    state->pos += add;
    return 0;
}

/* ============================================================
 * Turn-start checkpoints (the serve's rung 1c, QWEN35_TURN_CHECKPOINT).
 *
 * A session's block holds its state at the END of its last turn. When the
 * client drops or rewrites that turn's reply, the recurrent layers cannot be
 * rewound to the turn's start, so the start is saved before the turn decodes:
 * the recurrent images and the q4 fp16 sink/window shadows, device to device,
 * into per-row buffers carved with the blocks. KV rows below the checkpoint
 * are never rewritten (later tokens only append), so a restore copies the
 * saved images back into the row and sets the position back.
 * ============================================================ */

/* Recurrent image sizes of one SSM layer (the bind's own formulas). */
static void qw_ssm_image_bytes(const pn_qwen35moe_config *cfg,
                               size_t *conv_bytes, size_t *ssm_bytes)
{
    *conv_bytes = (size_t)(cfg->ssm_conv - 1) * cfg->conv_dim * sizeof(float);
    *ssm_bytes = (size_t)cfg->ssm_state * cfg->ssm_state * cfg->ssm_dt_rank *
                 sizeof(float);
}

static void qw_ckpt_free_all(pn_qwen35moe_gpu *g)
{
    for (int i = 0; i < g->n_ssm; i++) {
        fc_buffer_free(g->ssm[i].conv_ckpt);
        fc_buffer_free(g->ssm[i].ssm_ckpt);
        g->ssm[i].conv_ckpt = g->ssm[i].ssm_ckpt = NULL;
        g->ssm[i].ckpt_cap = 0;
    }
    for (int i = 0; i < g->n_attn; i++) {
        gpu_attn_layer *d = &g->attn[i];
        fc_buffer_free(d->k_sink_ckpt);
        fc_buffer_free(d->v_sink_ckpt);
        fc_buffer_free(d->k_win_ckpt);
        fc_buffer_free(d->v_win_ckpt);
        d->k_sink_ckpt = d->v_sink_ckpt = d->k_win_ckpt = d->v_win_ckpt = NULL;
        d->ckpt_cap = 0;
    }
    memset(g->ckpt_state, 0, sizeof(g->ckpt_state));
}

/* Checkpoints are optional, so they may only take VRAM the die gate did not
 * promise to the serve: after the carve every die must still have the runtime
 * scratch it was admitted with (allocated lazily, after this) plus the load
 * margin free. A cuMemAlloc that merely succeeds is not enough — it would let
 * the checkpoints eat the scratch and fail a later prefill instead. */
static int qw_ckpt_fit_dies(pn_qwen35moe_gpu *g, const pn_qwen35moe_state *s,
                            size_t conv_b, size_t ssm_b, int n_rows)
{
    uint64_t keep = PN_ADMIT_DIE_SCRATCH_BASE_BYTES + pn_die_load_margin_bytes();

    for (int die = 0; die < g->n_die; die++) {
        uint64_t bytes = 0;
        for (int i = 0; i < g->n_ssm; i++)
            if (g->ssm[i].ctx == g->ctx[die])
                bytes += (uint64_t)n_rows * (conv_b + ssm_b);
        for (int i = 0; i < qw_attn_lim(g, s); i++)
            if (g->attn[i].ctx == g->ctx[die] && s->kv[qw_attn_hidx(g, i)].q4)
                bytes += (uint64_t)n_rows * 2 *
                         (qw_kv_sink_bytes(&g->attn[i], &s->kv[qw_attn_hidx(g, i)]) +
                          qw_kv_win_bytes(&g->attn[i], &s->kv[qw_attn_hidx(g, i)]));
        size_t free_b = 0, total_b = 0;
        if (bytes == 0 || fc_mem_info_ctx(g->ctx[die], &free_b, &total_b) != 0)
            continue;
        if ((uint64_t)free_b < bytes + keep) {
            fprintf(stderr, "[qwen35moe] checkpoints: %d rows need %llu MiB on die %d, "
                    "which has %llu MiB free and keeps %llu MiB for scratch + margin; "
                    "not carved\n", n_rows, (unsigned long long)(bytes >> 20), g->die[die],
                    (unsigned long long)(free_b >> 20), (unsigned long long)(keep >> 20));
            return 0;
        }
    }
    return 1;
}

int pn_qwen35moe_slot_checkpoints_carve(pn_model *model, pn_state *geom,
                                        int n_rows)
{
    if (!model || !geom || n_rows <= 0 || n_rows > PN_QWEN35MOE_GPU_MAX_SLOTS ||
        !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)geom->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    if (!s || !g)
        return -1;
    size_t conv_b, ssm_b;   /* QWEN35_TP: a half's local images */
    qw_ssm_image_bytes(qw_state_cfg(g, g->tp, pn_qwen35moe_model_config(m)), &conv_b, &ssm_b);
    if (!qw_ckpt_fit_dies(g, s, conv_b, ssm_b, n_rows))
        return -1;
    size_t total = 0;
    int ok = 1;
    for (int i = 0; ok && i < g->n_ssm; i++) {
        gpu_ssm_layer *d = &g->ssm[i];
        d->conv_ckpt = fc_buffer_alloc_ctx(d->ctx, (size_t)n_rows * conv_b);
        d->ssm_ckpt = fc_buffer_alloc_ctx(d->ctx, (size_t)n_rows * ssm_b);
        ok = d->conv_ckpt && d->ssm_ckpt;
        d->ckpt_cap = ok ? n_rows : 0;
        total += (size_t)n_rows * (conv_b + ssm_b);
    }
    for (int i = 0; ok && i < g->n_attn; i++) {
        gpu_attn_layer *d = &g->attn[i];
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        if (kv->q4) {
            size_t sink_b = (size_t)n_rows * qw_kv_sink_bytes(d, kv);
            size_t win_b = (size_t)n_rows * qw_kv_win_bytes(d, kv);
            d->k_sink_ckpt = fc_buffer_alloc_ctx(d->ctx, sink_b);
            d->v_sink_ckpt = fc_buffer_alloc_ctx(d->ctx, sink_b);
            d->k_win_ckpt = fc_buffer_alloc_ctx(d->ctx, win_b);
            d->v_win_ckpt = fc_buffer_alloc_ctx(d->ctx, win_b);
            ok = d->k_sink_ckpt && d->v_sink_ckpt && d->k_win_ckpt &&
                 d->v_win_ckpt;
            total += 2 * (sink_b + win_b);
        }
        d->ckpt_cap = ok ? n_rows : 0;
    }
    if (!ok) {
        qw_ckpt_free_all(g);
        fprintf(stderr, "[qwen35moe] turn checkpoints: no VRAM for %d rows "
                "(%zu MiB); running without them\n", n_rows, total >> 20);
        return -1;
    }
    memset(g->ckpt_state, 0, sizeof(g->ckpt_state));
    /* What the checkpoints cost the sessions: the same bytes as KV rows. */
    uint64_t kv_tok_b = pn_kv_bytes_per_token(model, s->kv[0].q4 != 0);
    fprintf(stderr, "[qwen35moe] turn checkpoints: %d rows, %zu MiB on the "
            "card (all dies), %zu MiB per row; costs %llu tokens of KV room\n",
            n_rows, total >> 20, (total / (size_t)n_rows) >> 20,
            kv_tok_b ? (unsigned long long)(total / kv_tok_b) : 0ULL);
    return 0;
}

/* Copy checkpoint `ck`'s images, walking the layers in die order (the bind's
 * pattern). to_ckpt: live shared buffers -> checkpoint `ck`; else checkpoint
 * `ck` -> decode row `row`'s slot images. */
static void qw_ckpt_copy(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                         pn_qwen35moe_state *s, int row, int ck, int to_ckpt)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    size_t conv_b, ssm_b;   /* QWEN35_TP: a half's local images */
    qw_ssm_image_bytes(qw_state_cfg(g, g->tp, cfg), &conv_b, &ssm_b);
    fc_cuda_ctx *cur = NULL;
    for (int lr = 0; lr < cfg->n_layer * (g->tp ? 2 : 1); lr++) {
        /* QWEN35_TP: every layer twice, rank 0's half then rank 1's. */
        int layer = lr % cfg->n_layer, rank = lr / cfg->n_layer;
        int hidx = pn_qwen35moe_model_layer_index(m, layer);
        int is_ssm = pn_qwen35moe_model_layer_type(m, layer) ==
                     PN_QWEN35MOE_LAYER_SSM;
        int idx = hidx + rank * (is_ssm ? g->tp_n_ssm : g->tp_n_attn);
        fc_cuda_ctx *ctx = is_ssm ? g->ssm[idx].ctx : g->attn[idx].ctx;
        if (ctx != cur) {
            if (cur)
                fc_sync_ctx(cur);
            cur = ctx;
        }
        if (is_ssm) {
            gpu_ssm_layer *d = &g->ssm[idx];
            size_t co = (size_t)ck * conv_b, so = (size_t)ck * ssm_b;
            size_t rco = (size_t)row * conv_b, rso = (size_t)row * ssm_b;
            if (to_ckpt) {
                fc_buffer_copy_d2d_ctx(ctx, d->conv_ckpt, co, d->conv_state_g, 0, conv_b);
                fc_buffer_copy_d2d_ctx(ctx, d->ssm_ckpt, so, d->ssm_state_g, 0, ssm_b);
            } else {
                fc_buffer_copy_d2d_ctx(ctx, d->conv_slots, rco, d->conv_ckpt, co, conv_b);
                fc_buffer_copy_d2d_ctx(ctx, d->ssm_slots, rso, d->ssm_ckpt, so, ssm_b);
            }
            continue;
        }
        gpu_attn_layer *d = &g->attn[idx];
        const pn_qwen35moe_kv *kv = &s->kv[hidx];
        if (!kv->q4)
            continue;
        size_t sink_b = qw_kv_sink_bytes(d, kv), win_b = qw_kv_win_bytes(d, kv);
        size_t ko = (size_t)ck * sink_b, wo = (size_t)ck * win_b;
        size_t rko = (size_t)row * sink_b, rwo = (size_t)row * win_b;
        if (to_ckpt) {
            fc_buffer_copy_d2d_ctx(ctx, d->k_sink_ckpt, ko, d->k_sink_g, 0, sink_b);
            fc_buffer_copy_d2d_ctx(ctx, d->v_sink_ckpt, ko, d->v_sink_g, 0, sink_b);
            fc_buffer_copy_d2d_ctx(ctx, d->k_win_ckpt, wo, d->k_win_g, 0, win_b);
            fc_buffer_copy_d2d_ctx(ctx, d->v_win_ckpt, wo, d->v_win_g, 0, win_b);
        } else {
            fc_buffer_copy_d2d_ctx(ctx, d->k_sink_slots_g, rko, d->k_sink_ckpt, ko, sink_b);
            fc_buffer_copy_d2d_ctx(ctx, d->v_sink_slots_g, rko, d->v_sink_ckpt, ko, sink_b);
            fc_buffer_copy_d2d_ctx(ctx, d->k_win_slots_g, rwo, d->k_win_ckpt, wo, win_b);
            fc_buffer_copy_d2d_ctx(ctx, d->v_win_slots_g, rwo, d->v_win_ckpt, wo, win_b);
        }
    }
    for (int i = 0; i < g->n_die; i++)
        fc_sync_ctx(g->ctx[i]);
}

/* Do the live shared buffers hold `s` exactly at `pos` on every layer, with a
 * checkpoint row to copy it into? */
static int qw_ckpt_live_is(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                           pn_qwen35moe_state *s, int row, int pos)
{
    for (int i = 0; i < qw_ssm_lim(g, s); i++) {
        const gpu_ssm_layer *d = &g->ssm[i];
        if (d->ckpt_cap <= row || d->host_state != (const void *)&s->ssm[qw_ssm_hidx(g, i)] ||
            !d->conv_state_g || !d->ssm_state_g)
            return 0;
    }
    for (int i = 0; i < qw_attn_lim(g, s); i++) {
        const gpu_attn_layer *d = &g->attn[i];
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        if (d->ckpt_cap <= row)
            return 0;
        if (!kv->q4)
            continue;
        if (d->kv_host_id != (const void *)kv->kq || d->kv_gen != kv->content_gen ||
            d->kv_synced_pos != pos || !d->k_sink_g || !d->k_win_g)
            return 0;
    }
    (void)m;
    return 1;
}

/* Does row `row` still hold `s`'s KV rows [0, pos) and slot images to restore
 * checkpoint `ck` into? */
static int qw_ckpt_row_can_restore(pn_qwen35moe_gpu *g, pn_qwen35moe_state *s,
                                   int row, int ck, int pos)
{
    for (int i = 0; i < qw_ssm_lim(g, s); i++) {
        const gpu_ssm_layer *d = &g->ssm[i];
        if (d->ckpt_cap <= ck || d->slots_cap <= row || !d->conv_slots ||
            !d->ssm_slots)
            return 0;
    }
    for (int i = 0; i < qw_attn_lim(g, s); i++) {
        const gpu_attn_layer *d = &g->attn[i];
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        const void *id = kv->q4 ? (const void *)kv->kq : (const void *)kv->k;
        int need = pos < kv->max_seq ? pos : kv->max_seq;
        if (d->ckpt_cap <= ck || row >= d->kv_slots_cap_n ||
            d->kv_slot_host_id[row] != id ||
            d->kv_slot_gen[row] != kv->content_gen ||
            d->kv_slot_synced[row] < need)
            return 0;
        if (kv->q4 && (!d->k_sink_slots_g || !d->k_win_slots_g))
            return 0;
    }
    return 1;
}

int pn_qwen35moe_slot_checkpoint_save(pn_model *model, pn_state *state, int row)
{
    return pn_qwen35moe_slot_checkpoint_save_at(model, state, row);
}

int pn_qwen35moe_slot_checkpoint_save_at(pn_model *model, pn_state *state,
                                         int ck)
{
    const int row = ck;   /* the checkpoint index (bookkeeping arrays, buffers) */
    if (!model || !state || row < 0 || row >= PN_QWEN35MOE_GPU_MAX_SLOTS ||
        !qw_model_uses_qwen35moe_state(model) ||
        pn_qwen35moe_host_kv_mirror_eager())
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)state->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    const int pos = (int)state->pos;
    if (!s || !g || pos <= 0 || !qw_ckpt_live_is(m, g, s, row, pos))
        return -1;
    g->ckpt_state[row] = NULL;   /* void while the copy runs */
    qw_ckpt_copy(m, g, s, 0, row, 1);
    g->ckpt_state[row] = (const void *)s;
    g->ckpt_gen[row] = s->n_attn > 0 ? s->kv[0].content_gen : 0u;
    g->ckpt_pos[row] = pos;
    return 0;
}

int pn_qwen35moe_slot_checkpoint_restore(pn_model *model, pn_state *state,
                                         int row, int pos)
{
    return pn_qwen35moe_slot_checkpoint_restore_from(model, state, row, row, pos);
}

int pn_qwen35moe_slot_checkpoint_restore_from(pn_model *model, pn_state *state,
                                              int row, int ck, int pos)
{
    if (!model || !state || row < 0 || row >= PN_QWEN35MOE_GPU_MAX_SLOTS ||
        ck < 0 || ck >= PN_QWEN35MOE_GPU_MAX_SLOTS ||
        pos <= 0 || !qw_model_uses_qwen35moe_state(model) ||
        pn_qwen35moe_host_kv_mirror_eager())
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)state->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m)
                            : NULL;
    if (!s || !g)
        return -1;
    const unsigned gen = s->n_attn > 0 ? s->kv[0].content_gen : 0u;
    if (g->ckpt_state[ck] != (const void *)s || g->ckpt_gen[ck] != gen ||
        g->ckpt_pos[ck] != pos || row >= g->n_slots ||
        g->slot_state[row] != (const void *)s || g->slot_gen[row] != gen ||
        !qw_ckpt_row_can_restore(g, s, row, ck, pos))
        return -1;
    qw_ckpt_copy(m, g, s, row, ck, 0);
    g->slot_pos[row] = pos;
    for (int i = 0; i < g->n_attn; i++)
        if (g->attn[i].kv_slot_synced[row] > pos)
            g->attn[i].kv_slot_synced[row] = pos;   /* rows past it are stale */
    state->pos = pos;
    return 0;
}

int pn_qwen35moe_decode_slots(pn_model *model, const uint32_t *tokens,
                              pn_state **states, int n_users,
                              float **logits_out)
{
    return pn_qwen35moe_decode_slots_rows(model, tokens, states, NULL, n_users,
                                          logits_out);
}

int pn_qwen35moe_decode_slots_rows(pn_model *model, const uint32_t *tokens,
                                   pn_state **states, const int *rows,
                                   int n_users, float **logits_out)
{
    if (!model || !tokens || !states || n_users <= 0 ||
        n_users > PN_QWEN35MOE_GPU_MAX_SLOTS || !qw_model_uses_qwen35moe_state(model)) {
        fprintf(stderr, "[qwen35moe] decode_slots: bad call (n=%d, arch %d)\n",
                n_users, model ? (int)model->config.arch : -1);
        return -1;
    }
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    if (!m)
        return -1;

    pn_qwen35moe_state *sts[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int pos[PN_QWEN35MOE_GPU_MAX_SLOTS];
    for (int u = 0; u < n_users; u++) {
        if (!states[u] || !states[u]->kv_cache) {
            fprintf(stderr, "[qwen35moe] decode_slots: user %d has no state\n",
                    u);
            return -1;
        }
        sts[u] = (pn_qwen35moe_state *)states[u]->kv_cache;
        pos[u] = states[u]->pos;
    }

    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g) {
        fprintf(stderr, "[qwen35moe] decode_slots: model has no GPU weights\n");
        return -1;
    }
    g->mtp.ver_n = 0;   /* set again only by a solo step (gpu_head_from_slots) */
    if (pn_qwen35moe_decode_slots_tokens_cuda(m, tokens, sts, pos, rows,
                                              n_users) != 0)
        return -1;
    /* Each stepped row's SSM image advanced one position with its user. */
    for (int u = 0; u < n_users; u++)
        g->slot_pos[rows ? rows[u] : u]++;

    for (int u = 0; u < n_users; u++) {
        states[u]->pos++;
        if (logits_out)
            logits_out[u] = sts[u]->logits;
    }
    return 0;
}

/* ============================================================
 * QWEN35_BS_PD_OVERLAP — the prefill cursor's public calls (qwen35moe.h).
 * The pieces and the cycle's choreography are above tp_slots_step.
 * ============================================================ */

pn_qwen35moe_pd_cursor *pn_qwen35moe_pd_begin_cuda(pn_model *model, pn_state *state,
                                                   const uint32_t *tokens, int n_total,
                                                   int chunk)
{
    if (!qw_pd_overlap_on() || !qw_prefill_pipeline_wide_on())
        return NULL;
    if (!model || !state || !tokens || n_total <= 0 || chunk <= 0 ||
        !qw_model_uses_qwen35moe_state(model))
        return NULL;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = m ? (pn_qwen35moe_state *)state->kv_cache : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;
    if (!s || !g)
        return NULL;
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    /* The same refusals as pn_qwen35moe_prefill_range_pipelined_cuda, narrowed
     * to the layout the cycle is built for: MoE TP2 on 4 dies. */
    if (!g->tp || !cfg->n_expert || g->n_die != 4 || g->bs || prof_on() ||
        pn_qwen35moe_host_kv_mirror_eager() || !flash_on())
        return NULL;
    for (int d = 0; d < g->n_die; d++)
        if (!fc_attn_flash_available_ctx(g->ctx[d]))
            return NULL;
    for (int t = 0; t < n_total; t++)
        if ((int)tokens[t] >= cfg->n_vocab)
            return NULL;
    int cap = qw_batch_row_cap(s);
    if (cap > 0 && chunk > cap)
        chunk = cap;

    const int lane = g->lane;
    g->lane = QW_LANE_PREFILL;
    pn_qwen35moe_pd_cursor *cur = calloc(1, sizeof(*cur));
    int ok = cur != NULL;
    if (ok && !g->host_ok) {
        ok = gpu_host_scratch_init(&g->host, cfg) == 0;
        g->host_ok = ok;
    }
    ok = ok && gpu_get_batch_host(g, cfg, chunk) == 0 && tp_exchange_ready(g, cfg, chunk) == 0;
    for (int d = 0; ok && d < g->n_die; d++) {
        ok = gpu_select_die(g, g->die[d]) == 0;
        cur->bs[d] = ok ? gpu_get_batch_scratch(g, g->ctx[d], cfg, chunk) : NULL;
        cur->ds[d] = ok ? gpu_get_scratch(g, g->ctx[d], cfg) : NULL;
        ok = ok && cur->bs[d] && cur->ds[d];
    }
    g->lane = lane;
    if (!ok) {
        free(cur);
        return NULL;
    }
    cur->m = m;
    cur->g = g;
    cur->s = s;
    cur->state = state;
    cur->tokens = tokens;
    cur->n_total = n_total;
    cur->pos0 = state->pos;
    cur->chunk = chunk;
    cur->n_chunks = (n_total + chunk - 1) / chunk;
    cur->injected_max = cur->n_chunks;
    cur->ck[0] = cur->ck[1] = -1;
    return cur;
}

int pn_qwen35moe_pd_cycle_cuda(pn_model *model, pn_qwen35moe_pd_cursor *cur,
                               const uint32_t *tokens, pn_state **states, const int *rows,
                               int n_users, float **logits_out, int pieces)
{
    if (!model || !cur || !tokens || !states || pieces < 0 || n_users <= 0 ||
        n_users > PN_QWEN35MOE_GPU_MAX_SLOTS || model->weights != (void *)cur->m)
        return -1;
    pn_qwen35moe_model *m = cur->m;
    pn_qwen35moe_gpu *g = cur->g;
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    pn_qwen35moe_state *sts[PN_QWEN35MOE_GPU_MAX_SLOTS];
    int pos[PN_QWEN35MOE_GPU_MAX_SLOTS];
    for (int u = 0; u < n_users; u++) {
        if (!states[u] || !states[u]->kv_cache)
            return -1;
        sts[u] = (pn_qwen35moe_state *)states[u]->kv_cache;
        pos[u] = states[u]->pos;
    }
    g->mtp.ver_n = 0;
    if (qw_slots_step_prepare(g, cfg, tokens, sts, rows, n_users) < 0)
        return -1;
    g->lane = QW_LANE_DECODE;
    int rc = tp_slots_step(m, g, tokens, sts, pos, n_users, cur, pieces);
    g->lane = QW_LANE_PREFILL;
    if (rc != 0)
        return -1;
    for (int u = 0; u < n_users; u++) {
        g->slot_pos[rows ? rows[u] : u]++;
        states[u]->pos++;
        if (logits_out)
            logits_out[u] = sts[u]->logits;
    }
    return 0;
}

int pn_qwen35moe_pd_issued_all(const pn_qwen35moe_pd_cursor *cur)
{
    return cur && cur->retired >= cur->injected_max;
}

int pn_qwen35moe_pd_injected_all(const pn_qwen35moe_pd_cursor *cur)
{
    return cur && cur->next_inject >= cur->injected_max;
}

void pn_qwen35moe_pd_phase_ms(const pn_qwen35moe_pd_cursor *cur,
                              double ms[PN_QWEN35MOE_PD_PHASES])
{
    for (int i = 0; i < PN_QWEN35MOE_PD_PHASES; i++)
        ms[i] = cur ? cur->ph_ms[i] : 0.0;
}

int pn_qwen35moe_pd_remaining(const pn_qwen35moe_pd_cursor *cur)
{
    if (!cur)
        return 0;
    const int done = cur->retired * cur->chunk;
    return done < cur->n_total ? cur->n_total - done : 0;
}

/* Run the pipeline alone (no decode between pieces) until `target` chunks
 * retired: stage 1's whole chunk and stage 0's whole chunk are issued together
 * (the two boards overlap, as in the wide pipeline), then the hand-off. */
static int qw_pd_run_alone(pn_qwen35moe_pd_cursor *cur, int target)
{
    if (target > cur->injected_max)
        target = cur->injected_max;
    while (cur->retired < target) {
        int a = qw_pd_issue_stage(cur, 1, -1);
        int b = qw_pd_issue_stage(cur, 0, -1);
        if (a < 0 || b < 0)
            return -1;
        if (cur->retired >= target)
            break;
        if (cur->ck[1] < 0 && cur->ck[0] >= 0)
            fc_sync_ctx(cur->g->ctx[1]);   /* rank 1 of the chunk to hand on */
        int h = qw_pd_hand_off(cur);
        if (h < 0)
            return -1;
        if (!a && !b && !h)
            return -1;                     /* no progress: a bookkeeping bug */
    }
    return 0;
}

int pn_qwen35moe_pd_advance_cuda(pn_qwen35moe_pd_cursor *cur, int max_chunks)
{
    if (!cur || max_chunks <= 0)
        return -1;
    return qw_pd_run_alone(cur, cur->retired + max_chunks);
}

/* Drain every die and publish the retired position. */
static void qw_pd_drain(pn_qwen35moe_pd_cursor *cur)
{
    for (int d = 0; d < cur->g->n_die; d++)
        fc_sync_ctx(cur->g->ctx[d]);
    int done = cur->retired * cur->chunk;
    cur->state->pos = cur->pos0 + (done < cur->n_total ? done : cur->n_total);
}

int pn_qwen35moe_pd_quiesce_cuda(pn_qwen35moe_pd_cursor *cur)
{
    if (!cur)
        return -1;
    const int max = cur->injected_max;
    cur->injected_max = cur->next_inject;   /* nothing new enters */
    int rc = qw_pd_run_alone(cur, cur->injected_max);
    cur->injected_max = max;
    qw_pd_drain(cur);
    return rc;
}

int pn_qwen35moe_pd_finish_cuda(pn_qwen35moe_pd_cursor *cur, tm_tensor **logits_out)
{
    if (logits_out)
        *logits_out = NULL;
    if (!cur)
        return -1;
    pn_qwen35moe_gpu *g = cur->g;
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(cur->m);
    const int d = g->n_die - 1;
    int rc = qw_pd_run_alone(cur, cur->injected_max);
    qw_pd_drain(cur);
    if (rc == 0) {
        qw_pipe_chunk k = qw_pipe_chunk_at(cur->tokens, cur->n_total, cur->pos0, cur->chunk,
                                           cur->n_chunks - 1);
        const int lane = g->lane;
        g->lane = QW_LANE_PREFILL;
        if (!cur->normed_last)
            fc_rms_norm_nx_ctx(g->ctx[d], cur->bs[d]->hidden, g->output_norm.buf,
                               cur->bs[d]->x_norm, cfg->n_embd, cfg->rms_norm_eps, k.n);
        if (!gpu_batch_head_last_row(g, cfg, g->ctx[d], cur->bs[d], cur->ds[d], g->host,
                                     cur->s, k.n))
            rc = -1;
        g->lane = lane;
    }
    if (rc == 0 && logits_out) {
        int64_t shape[1] = { cur->s->n_vocab };
        tm_tensor *out = tm_empty(shape, 1, TM_F32);
        if (out)
            memcpy(tm_data_f32(out), cur->s->logits, (size_t)cur->s->n_vocab * sizeof(float));
        else
            rc = -1;
        *logits_out = out;
    }
    free(cur);
    return rc;
}

int pn_qwen35moe_pd_abort_cuda(pn_qwen35moe_pd_cursor *cur)
{
    if (!cur)
        return -1;
    cur->injected_max = cur->next_inject;
    cur->aborted = 1;
    int rc = qw_pd_run_alone(cur, cur->injected_max);
    qw_pd_drain(cur);
    free(cur);
    return rc;
}


/* ============================================================
 * PREFIX CACHING — snapshot the live GPU state back to the host (on-evict)
 *
 * B2 (KV mirror): single-token decode no longer writes D2H per token; instead
 * kv->kv_mirror_stale is set. This function refreshes any stale attn KV
 * mirrors by D2H-ing the full GPU KV ring back to kv->k/v (or kq/vq). The GPU
 * ring layout is byte-identical to the host ring, so a full fc_buffer_to_cpu of
 * k_cache_g/v_cache_g is sufficient. After the copy the mirror is authoritative
 * and gpu_attn_kv_cache_sync can H2D from it on session resume.
 *
 * B3 (SSM state): the 30 SSM layers' fixed-size recurrent state lives only in
 * the shared per-layer GPU buffers (conv_state_g / ssm_state_g) after a GPU
 * decode. The host copy goes stale when d->host_state matches this session.
 * This D2H copies it back so a session's pn_state is self-contained before the
 * GPU buffers are reused for another session.
 *
 * INVARIANT: called by preserve_live (prefix_cache.c) before c->live is
 * switched to a different session. At that point the GPU KV ring and SSM state
 * still hold this session's data. After this call:
 *  - kv->k/v (or kq/vq + k_sink/v_sink/k_win/v_win) are current → next resume
 *    can H2D via gpu_attn_kv_cache_sync.
 *  - st->conv_state/ssm_state are current → re-seeds on next forward.
 * A session is NEVER left with stale host state after its GPU buffers are
 * claimed by another session, so no KV is lost. ============================================================ */
/* SSM half of gpu_state_to_host: capture the SHARED single-user SSM buffers
 * into the host mirror for every layer whose GPU buffer holds THIS state's
 * image. Die-ordered walk so at most one die-select is needed. */
static int qw_state_ssm_shared_to_host(pn_qwen35moe_model *m,
                                       pn_qwen35moe_gpu *g,
                                       pn_qwen35moe_state *s)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int rc = 0;
    fc_cuda_ctx *cur = NULL;

    for (int layer = 0; layer < cfg->n_layer; layer++) {
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_SSM)
            continue;
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        pn_qwen35moe_ssm_state *st = &s->ssm[idx];
        for (int r = 0; rc == 0 && r < (g->tp ? 2 : 1); r++) {   /* QWEN35_TP: both halves */
            gpu_ssm_layer *d = &g->ssm[idx + r * g->tp_n_ssm];
            /* Only copy layers whose GPU buffer currently holds THIS state's
             * image — otherwise the host copy is already the source of truth. */
            if (!d->conv_state_g || !d->ssm_state_g ||
                d->host_state != (const void *)st)
                continue;
            if (d->ctx != cur) {
                if (cur)
                    fc_sync_ctx(cur);
                if (gpu_select_die(g, gpu_die_of_ctx(g, d->ctx)) < 0) {
                    rc = -1;
                    break;
                }
                cur = d->ctx;
            }
            qw_ssm_xfer(g, d, st, 0);
        }
        if (rc)
            break;
    }
    if (cur)
        fc_sync_ctx(cur);
    return rc;
}

int pn_qwen35moe_gpu_state_to_host(pn_model *model, pn_state *state)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return 0; /* CPU path: the host state is already authoritative. */

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int rc = 0;
    fc_cuda_ctx *cur = NULL;

    /* B2 / Phase 10b — KV mirror materialization (attn layers). Walk
     * die-ordered so at most one die-select is needed. Only layers whose
     * kv_mirror_stale flag is set need work; the rows come from whichever
     * GPU ring holds this state's content (single-user ring on the B2
     * single-user deferral, decode slot ring in lazy-mirror batched mode),
     * bounded to [0, min(pos, max_seq)) so the host commit is proportional
     * to the session's real context. */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
            continue;
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        gpu_attn_layer *d = &g->attn[idx];
        pn_qwen35moe_kv *kv = &s->kv[idx];
        if (!kv->kv_mirror_stale)
            continue;
        if (d->ctx != cur) {
            if (cur)
                fc_sync_ctx(cur);
            if (gpu_select_die(g, gpu_die_of_ctx(g, d->ctx)) < 0) {
                rc = -1;
                break;
            }
            cur = d->ctx;
        }
        int rows = state->pos < kv->max_seq ? state->pos : kv->max_seq;
        if (g->tp)
            fc_sync_ctx(g->attn[idx + g->tp_n_attn].ctx);   /* the rank-1 half's die */
        if (qw_kv_layer_materialize_all(g, idx, kv, rows) != 0) {
            fprintf(stderr, "[attn] state_to_host kv_idx=%d: host mirror "
                    "stale with no GPU ring source (rows=%d) — mirror left "
                    "stale\n", idx, rows);
            rc = -1;
        }
    }
    if (cur)
        fc_sync_ctx(cur);
    if (rc < 0)
        return rc;

    /* B3 — SSM state snapshot (shared single-user buffers). */
    return qw_state_ssm_shared_to_host(m, g, s);
}

int pn_qwen35moe_gpu_state_ssm_to_host(pn_model *model, pn_state *state)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return 0; /* CPU path: the host state is already authoritative. */
    return qw_state_ssm_shared_to_host(m, g, s);
}

/* ============================================================
 * PREFIX CACHING (batched path) — snapshot a DECODE SLOT's live SSM state
 * back into that slot's pn_state host mirror.
 *
 * pn_qwen35moe_gpu_state_to_host (above) captures only the SHARED single-user
 * SSM buffers (conv_state_g / ssm_state_g, guarded by d->host_state == st).
 * Batched decode instead advances the per-slot SSM images in the slot-major
 * arrays d->conv_slots / d->ssm_slots at row `gpu_idx` — those shared buffers
 * are NEVER touched — so after a batched turn a slot's pn_state SSM mirror is
 * stale even though its pos + KV mirror are current.  That leaves the slot's
 * state not delta-reusable by the prefix cache.
 *
 * This D2H-copies conv_slots[gpu_idx] → st->conv_state and
 * ssm_slots[gpu_idx] → st->ssm_state for every SSM layer, the exact inverse of
 * the host→slot seed in pn_qwen35moe_decode_slot_bind_cuda (same conv_bytes /
 * ssm_bytes stride).  After it the slot's pn_state SSM mirror is authoritative
 * — combined with the already-current KV mirror (bs_slot_bind / the L2 yield
 * keep it live) and pos == n_committed, the state is warm + self-contained and
 * a next turn can delta-prefill only the appended tokens.
 *
 * Also clears d->host_state so any layer whose SHARED buffer happens to point
 * at this st is re-seeded on its next single-user forward — the slot images
 * are the source of truth here, not the shared buffers.  Host mirror + GPU
 * read only; no slot compute.  Caller (the batched commit) must hold the GPU
 * turn so no decode step advances the slots mid-copy.  gpu_idx is the slot's
 * CURRENT GPU index (bs_slot.gpu_idx), which decode_slot_move keeps current.
 * ============================================================ */
int pn_qwen35moe_decode_slot_ssm_to_host(pn_model *model, int gpu_idx,
                                         pn_state *state)
{
    if (!model || !state || gpu_idx < 0 ||
        gpu_idx >= PN_QWEN35MOE_GPU_MAX_SLOTS || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return 0; /* CPU path: the host state is already authoritative. */
    if (g->n_slots <= gpu_idx)
        return -1;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    const pn_qwen35moe_config *dcfg = qw_state_cfg(g, g->tp, cfg);   /* QWEN35_TP halves */
    size_t conv_bytes = (size_t)(dcfg->ssm_conv - 1) * dcfg->conv_dim *
                        sizeof(float);
    size_t ssm_bytes  = (size_t)dcfg->ssm_state * dcfg->ssm_state *
                        dcfg->ssm_dt_rank * sizeof(float);

    int rc = 0;
    fc_cuda_ctx *cur = NULL;
    /* Die-ordered walk (front die then back) so at most one die-select per die,
     * mirroring the B3 loop in pn_qwen35moe_gpu_state_to_host. */
    for (int lr = 0; lr < cfg->n_layer * (g->tp ? 2 : 1); lr++) {
        int layer = lr % cfg->n_layer, rank = lr / cfg->n_layer;   /* QWEN35_TP halves */
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_SSM)
            continue;
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        gpu_ssm_layer *d = &g->ssm[idx + rank * g->tp_n_ssm];
        pn_qwen35moe_ssm_state *st = &s->ssm[idx];
        if (!d->conv_slots || !d->ssm_slots || d->slots_cap <= gpu_idx)
            return -1;   /* slots not allocated for this width */
        if (d->ctx != cur) {
            if (cur)
                fc_sync_ctx(cur);
            if (gpu_select_die(g, gpu_die_of_ctx(g, d->ctx)) < 0) {
                rc = -1;
                break;
            }
            cur = d->ctx;
        }
        if (d->tp_on) {
            qw_ssm_xfer_at(g, d, d->conv_slots, (size_t)gpu_idx * conv_bytes, d->ssm_slots,
                           (size_t)gpu_idx * ssm_bytes, st, 0);
        } else {
            fc_buffer_to_cpu_at(d->conv_slots, st->conv_state,
                                (size_t)gpu_idx * conv_bytes, conv_bytes);
            fc_buffer_to_cpu_at(d->ssm_slots, st->ssm_state,
                                (size_t)gpu_idx * ssm_bytes, ssm_bytes);
        }
        /* The slot images are now the source of truth for this state; drop any
         * shared-buffer "live here" binding so a later single-user forward
         * re-seeds from the (now-current) host mirror rather than trusting a
         * stale shared buffer. */
        if (d->host_state == (const void *)st)
            d->host_state = NULL;
    }
    if (cur)
        fc_sync_ctx(cur);
    return rc;
}

/* Forget every GPU-resident "this state is live here" binding for `state`, so
 * the NEXT forward re-seeds all per-layer GPU shadows from the host mirror as
 * if the state were fresh.  A serve slot's pn_state is REUSED across requests;
 * pn_qwen35moe_state_reset zeros only the HOST mirror, leaving d->host_state
 * (SSM) and kv_host_id/kv_synced_pos (single-user KV) still pointing at this
 * state with stale GPU contents from its previous life.  A short re-prefill
 * only re-touches low positions so the residue is never read (the reused slot
 * looked fine), but a LONG prefill attends/carries positions the reset never
 * cleared → its logits diverge from a fresh state.  Clearing the bindings
 * here forces gpu_attn_kv_cache_sync / the SSM re-seed gate to treat the
 * reused state as new (host_state != st → full re-seed from the zeroed +
 * re-prefilled host mirror).  Host bookkeeping only (no GPU work); the
 * scheduler calls it under the GPU turn right after pn_state_reset, so no
 * forward can race it.  No-op without a GPU handle (CPU path). */
/* Drop every layer struct's binding to host state s, so the next batch re-seeds SSM state and
 * re-syncs KV from it. Under QWEN35_TP the doubled arrays hold rank 1's halves at
 * [i + tp_n_*] (qw_*_lim / qw_*_hidx): both halves must drop. Rank 0's alone left rank 1
 * seeding its GDN heads from the shared buffers' stale prompt-end state on a session's turn 2
 * (the reply diverged from the layer split). QW_FORGET_RED (gate red build): rank 0's half. */
static void qw_forget_state_bindings(pn_qwen35moe_gpu *g, pn_qwen35moe_state *s)
{
#ifdef QW_FORGET_RED
    const int n_ssm = g->tp ? g->tp_n_ssm : qw_ssm_lim(g, s);
    const int n_attn = g->tp ? g->tp_n_attn : qw_attn_lim(g, s);
#else
    const int n_ssm = qw_ssm_lim(g, s), n_attn = qw_attn_lim(g, s);
#endif
    for (int i = 0; i < n_ssm; i++)
        if (g->ssm[i].host_state == (const void *)&s->ssm[qw_ssm_hidx(g, i)])
            g->ssm[i].host_state = NULL;   /* force SSM re-seed from host */
    for (int i = 0; i < n_attn; i++) {
        gpu_attn_layer *d = &g->attn[i];
        const pn_qwen35moe_kv *kv = &s->kv[qw_attn_hidx(g, i)];
        const void *host_id = kv->q4 ? (const void *)kv->kq : (const void *)kv->k;
        if (d->kv_host_id == host_id) {
            d->kv_host_id    = NULL;   /* force full KV re-sync + shadow */
            d->kv_synced_pos = 0;
        }
    }
}

void pn_qwen35moe_gpu_forget_ssm(pn_model *model, pn_state *state)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;
    if (!s || !g)
        return;
    for (int i = 0; i < qw_ssm_lim(g, s); i++)
        if (g->ssm[i].host_state == (const void *)&s->ssm[qw_ssm_hidx(g, i)])
            g->ssm[i].host_state = NULL;
}

void pn_qwen35moe_gpu_forget_state(pn_model *model, pn_state *state)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return;

    /* Any verify pending against this state is void once its bindings are
     * dropped — and spec_commit must not dereference a state the caller is
     * about to reset or free. */
    if (g->spec_state == s) {
        g->spec_state = NULL;
        g->spec_n = 0;
    }

    qw_forget_state_bindings(g, s);
}

/* Test hook (tests/test_qwen35_tp_forget): a TP layout of 2 SSM + 2 attention host layers
 * (4 + 4 layer structs, rank 1's halves at [i + tp_n_*]), every struct bound to the host
 * state, then the forget. @return the bindings left (0 = correct). */
int pn_qwen35moe_tp_forget_selftest(void)
{
    pn_qwen35moe_gpu *g = calloc(1, sizeof(*g));
    pn_qwen35moe_state *s = calloc(1, sizeof(*s));
    gpu_ssm_layer ssm[4];
    gpu_attn_layer attn[4];
    pn_qwen35moe_ssm_state hs[2];
    pn_qwen35moe_kv kv[2];
    unsigned char kq[2];
    int left = 0;

    if (!g || !s) {
        free(g);
        free(s);
        return -1;
    }
    memset(ssm, 0, sizeof(ssm));
    memset(attn, 0, sizeof(attn));
    memset(hs, 0, sizeof(hs));
    memset(kv, 0, sizeof(kv));
    g->tp = 1;
    g->tp_n_ssm = g->tp_n_attn = 2;
    g->n_ssm = g->n_attn = 4;
    g->ssm = ssm;
    g->attn = attn;
    s->n_ssm = s->n_attn = 2;
    s->ssm = hs;
    s->kv = kv;
    for (int i = 0; i < 2; i++) {
        kv[i].q4 = 1;
        kv[i].kq = &kq[i];
    }
    for (int i = 0; i < 4; i++) {
        ssm[i].host_state = &hs[i % 2];
        attn[i].kv_host_id = kv[i % 2].kq;
        attn[i].kv_synced_pos = 7;
    }
    qw_forget_state_bindings(g, s);
    for (int i = 0; i < 4; i++)
        left += (ssm[i].host_state != NULL) + (attn[i].kv_host_id != NULL) +
                (attn[i].kv_synced_pos != 0);
    free(g);
    free(s);
    return left;
}

/* The bytes of TP half d's GPU K ring rows [0, rows) that differ from the host mirror's. */
static size_t qw_tp_ring_diff_bytes(gpu_attn_layer *d, const pn_qwen35moe_kv *kv, int rows)
{
    size_t row_b = qw_kv_row_bytes(kv), head_b = (size_t)kv->max_seq * row_b, diff = 0;
    unsigned char *got = malloc((size_t)rows * row_b);
    if (!got)
        return (size_t)-1;
    for (int hd = 0; hd < qw_kv_nh(d, kv); hd++) {
        const unsigned char *want = kv->kq + (size_t)(qw_kv_h0(d, kv) + hd) * head_b;
        fc_buffer_to_cpu_at(d->k_cache_g, got, (size_t)hd * head_b, (size_t)rows * row_b);
        for (size_t i = 0; i < (size_t)rows * row_b; i++)
            diff += got[i] != want[i];
    }
    free(got);
    return diff;
}

/* Test hook (tests/test_qwen35_tp_restore_sync): one attention layer as two TP halves on
 * one die, a q4 host kv. Turn 1 prefills rows [0, 8) on both halves (lazy: mirror stale);
 * a restore then replaces the mirror (fresh, new content_gen, as snap_kv_restore does) and
 * turn 2 resumes at pos0 = 6 in the order tp_layer_forward_batch / tp_verify_stage run it.
 * Needs the lazy mirror (the default). @return refused syncs + differing ring bytes. */
int pn_qwen35moe_tp_restore_sync_selftest(void)
{
    enum { NH = 2, HD = 64, SEQ = 64, POS1 = 8, POS2 = 6 };
    fc_cuda_ctx *ctx = fc_cuda_init_device(0);
    if (!ctx || !fc_cuda_ctx_available(ctx))
        return -2;
    pn_qwen35moe_gpu *g = calloc(1, sizeof(*g));
    gpu_attn_layer attn[2];
    pn_qwen35moe_kv kv;
    size_t row_b = (size_t)pn_kvq4_row_bytes(HD), ring_b = (size_t)NH * SEQ * row_b;
    size_t sink_n = (size_t)NH * PN_KVQ4_SINKMAX * HD, win_n = (size_t)NH * PN_KVQ4_WMAX * HD;
    long bad = 0;

    memset(attn, 0, sizeof(attn));
    memset(&kv, 0, sizeof(kv));
    kv.q4 = 1;
    kv.n_head_kv = NH;
    kv.head_dim = HD;
    kv.max_seq = SEQ;
    kv.kq = malloc(ring_b);
    kv.vq = malloc(ring_b);
    kv.k_sink = calloc(sink_n, 2);
    kv.v_sink = calloc(sink_n, 2);
    kv.k_win = calloc(win_n, 2);
    kv.v_win = calloc(win_n, 2);
    if (!g || !kv.kq || !kv.vq || !kv.k_sink || !kv.v_sink || !kv.k_win || !kv.v_win) {
        bad = -1;
        goto out;
    }
    g->tp = 1;
    g->tp_n_attn = 1;
    g->n_attn = 2;
    g->attn = attn;
    for (int p = 0; p < 2; p++) {
        attn[p].ctx = ctx;
        attn[p].tp_on = 1;
        attn[p].tp_rank = p;
    }
    for (size_t i = 0; i < ring_b; i++) {
        kv.kq[i] = (unsigned char)(i * 7 + 1);
        kv.vq[i] = (unsigned char)(i * 13 + 5);
    }
    kv.content_gen = pn_qwen35moe_kv_next_content_gen();
    for (int p = 0; p < 2; p++) {               /* turn 1: prefill [0, POS1) */
        bad += gpu_attn_kv_cache_sync(&attn[p], &kv, 0) < 0;
        attn[p].kv_synced_pos = POS1;
        kv.kv_mirror_stale = 1;
    }
    for (size_t i = 0; i < ring_b; i++)         /* restore: new rows, fresh mirror */
        kv.kq[i] ^= 0x5a;
    kv.kv_mirror_stale = 0;
    kv.content_gen = pn_qwen35moe_kv_next_content_gen();
    bad += tp_attn_kv_cache_sync_both(g, 0, &kv, POS2) < 0;   /* turn 2 at POS2 */
    for (int p = 0; p < 2; p++) {
        bad += gpu_attn_kv_cache_sync(&attn[p], &kv, POS2) < 0;
        kv.kv_mirror_stale = 1;
    }
    for (int p = 0; p < 2; p++)
        bad += (long)qw_tp_ring_diff_bytes(&attn[p], &kv, POS2);
out:
    for (int p = 0; p < 2; p++) {
        gpu_attn_layer *d = &attn[p];
        fc_buffer_free(d->k_cache_g);
        fc_buffer_free(d->v_cache_g);
        fc_buffer_free(d->k_sink_g);
        fc_buffer_free(d->v_sink_g);
        fc_buffer_free(d->k_win_g);
        fc_buffer_free(d->v_win_g);
    }
    free(kv.kq);
    free(kv.vq);
    free(kv.k_sink);
    free(kv.v_sink);
    free(kv.k_win);
    free(kv.v_win);
    free(g);
    fc_cuda_ctx_free(ctx);
    return (int)(bad > 1000000 ? 1000000 : bad);
}

/* ============================================================
 * C2 — ASYNC decode step (launch / event-poll)
 *
 * The synchronous forward above ends every die segment with a blocking
 * fc_sync_ctx. That is fine for one card, but the K80 test machine's host has 2 cores
 * driving up to 8 cards: 8 connection threads blocked in
 * cuStreamSynchronize spin/yield for the two cores, so a card whose step
 * FINISHED waits a scheduler quantum for its thread to get CPU again and
 * launch the next step — the cards serialize at the sync points instead
 * of overlapping.
 *
 * The async step splits one token's forward at its natural host-sync
 * seams into launch + reap:
 *
 *   begin(): embed seed (H2D) -> segment-A graph replay (one
 *            cuGraphLaunch) -> [single-die: + final norm + LM head
 *            launches] -> fc_event_record -> RETURN (no blocking).
 *   poll():  fc_event_query (never blocks).
 *            WAIT_A done  -> boundary hidden D2H/H2D bounce ->
 *                            segment-B graph replay -> final norm +
 *                            LM head -> record -> WAIT_FINAL.
 *            WAIT_FINAL done -> top-k logits fetch (syncs only THIS
 *                            card's stream) -> s->logits ready.
 *
 * The caller (serve's generation loop) sleeps between polls, so a
 * waiting card costs ~zero host CPU and every active card's stream is
 * in flight simultaneously. Between begin/poll phases each ctx's stream
 * is provably idle at the points where the sync path relied on that
 * (gpu_segment_forward's pre-work H2Ds, the boundary bounce, dec_pos),
 * because the phase only advances once the event for ALL prior work on
 * that stream has completed — the same invariant the blocking fc_sync
 * gave, established without occupying a core.
 *
 * ELIGIBILITY mirrors the segment-graph gate (gpu_segment_forward): any
 * refusal — pos 0, capture failure, ring wrap, prof/dbg, KV-geometry
 * change — returns 1 from begin() and the caller runs the ordinary
 * blocking forward for that token, bit-identically. Legacy barswap mode
 * (g->bs != NULL, one global MMIO slot) and host-fallback LM heads are
 * also refused. With QWEN35_ASYNC_SCHED=0 begin() always returns 1 and
 * serving behaves exactly as before this change.
 *
 * Correctness under interleaving: every buffer begin/poll touches
 * (gpu_die_scratch, gpu_host_scratch, the KV/SSM state, the events, the
 * phase machine itself) hangs off THIS model's pn_qwen35moe_gpu / its
 * own per-die fc_cuda_ctx, and the per-model gen_lock guarantees one
 * generation (hence one in-flight step) per model. Different models
 * share no CUDA context, stream, or scratch, and every fc_* op
 * push/pops its own context — so N cards' steps interleave freely.
 * ============================================================ */

enum {
    QW_ASYNC_IDLE       = 0,
    QW_ASYNC_WAIT_A     = 1,   /* 2-die: front segment in flight on ctx_a */
    QW_ASYNC_WAIT_FINAL = 2,   /* back segment + LM head in flight */
};

/* Async dispatch (QWEN35_ASYNC_SCHED, default on; set 0 to restore the
 * fully synchronous serve path — bisection aid). */
static int gpu_async_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_ASYNC_SCHED");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Does the LM head have an on-device matvec kernel? The exotic-quant host
 * fallback in gpu_matvec does a blocking D2H of x (drains the whole
 * segment inside begin()), defeating the overlap — refuse async then. */
static int gpu_head_on_device(const pn_qwen35moe_gpu *g)
{
    if (!g->output.bytes)
        return 0;
    switch ((int)g->output.type) {
    case GGUF_QUANT_Q3_K:
    case GGUF_QUANT_Q4_K:
    case GGUF_QUANT_Q5_K:
    case GGUF_QUANT_Q6_K:
    case GGUF_QUANT_Q8_0:
    case GGUF_QUANT_F32:
        return 1;
    default:
        return 0;
    }
}

/* Lazily create the completion event for die slot `which` (0 = ctx_a,
 * 1 = ctx_b). A failed create latches so async stays off cheaply. */
static fc_event *gpu_async_event(pn_qwen35moe_gpu *g, int which,
                                 fc_cuda_ctx *ctx)
{
    if (!g->as_ev[which] && !g->as_ev_tried[which]) {
        g->as_ev_tried[which] = 1;
        g->as_ev[which] = fc_event_create_ctx(ctx);
    }
    return g->as_ev[which];
}

/* Arm `ev` behind everything queued on ctx's stream. If the record fails
 * we fall back to a blocking sync RIGHT HERE, so a later query on the
 * (possibly stale) event can never report "done" before the work is —
 * the stream is already drained. Degraded to sync, never wrong. */
static void gpu_async_arm(fc_cuda_ctx *ctx, fc_event *ev)
{
    if (fc_event_record_ctx(ctx, ev) != 0)
        fc_sync_ctx(ctx);
}

/* Reap one phase's event: 0 = complete, 1 = still running. An error from
 * the query degrades to a blocking sync (correct, just not async). */
static int gpu_async_reap(fc_cuda_ctx *ctx, fc_event *ev)
{
    int q = fc_event_query(ev);
    if (q > 0)
        return 1;
    if (q < 0)
        fc_sync_ctx(ctx);
    return 0;
}

/**
 * Launch one single-token decode step WITHOUT blocking. Returns 0 when the
 * step is in flight (the caller MUST then call .._poll until it returns
 * 0 or -1 before any other forward on this model), 1 when this token is
 * not async-eligible (caller runs the ordinary blocking forward — nothing
 * was launched), -1 on error.
 */
int pn_qwen35moe_forward_token_cuda_begin(pn_qwen35moe_model *m,
                                          uint32_t token,
                                          pn_qwen35moe_state *s, int pos)
{
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g || !gpu_async_enabled())
        return 1;
    if (g->bs)
        return 1;   /* legacy barswap: one global MMIO slot — stay sync */
    if (g->as_phase != QW_ASYNC_IDLE)
        return -1;  /* protocol misuse: a step is already in flight */

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int n_layer = cfg->n_layer;
    if ((int)token >= cfg->n_vocab)
        return 1;
    /* The phase machine has exactly two segment phases (front, then back), so
     * it drives a 1- or 2-die split. A wider split has more boundaries than
     * phases; rather than deepen the machine for a shape that trades tok/s for
     * capacity anyway, hand those steps to the sync N-die walk, which is the
     * only correctness-critical path. */
    if (g->n_die > 2)
        return 1;
    int two_die = (g->n_die == 2);
    int split = g->split_layer;

    /* Debug layer cap runs the (partial) sync path; exotic LM-head quant
     * would host-fallback inside begin (see gpu_head_on_device). */
    if (!two_die && getenv("QWEN35_GPU_MAX_LAYERS"))
        return 1;
    if (!gpu_head_on_device(g))
        return 1;

    /* The completion events must exist BEFORE anything is launched. */
    fc_event *ev_a = gpu_async_event(g, 0, g->ctx[0]);
    fc_event *ev_b = two_die ? gpu_async_event(g, 1, g->head_ctx) : NULL;
    if (!ev_a || (two_die && !ev_b))
        return 1;

    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, cfg) < 0)
            return -1;
        g->host_ok = 1;
    }
    gpu_host_scratch h = g->host;

    /* Embed + seed die A's residual stream. The H2D is synchronous but the
     * stream is idle between tokens (poll drained it via the event before
     * the previous step completed), so it is a pure ~n_embd-float copy. */
    gpu_die_scratch *sc_a = gpu_get_scratch(g, g->ctx[0], cfg);
    if (!sc_a)
        return -1;
    pn_qwen35moe_model_embed_row(m, token, h.a);
    fc_buffer_from_cpu(sc_a->r_hidden, h.a);

    /* Front segment as ONE graph replay (C1). Any refusal — pos 0, capture
     * failed, ring wrap, prof/dbg — means the sync path takes this token;
     * the seed above is harmlessly redone there and the segment's pass-2
     * pre-work H2Ds are idempotent. */
    int seg_rc = gpu_segment_forward(m, g, g->ctx[0], 0, 0,
                                     two_die ? split : n_layer,
                                     sc_a->r_hidden, sc_a->r_xnorm,
                                     sc_a->r_blockout, s, pos, &g->host);
    if (seg_rc < 0)
        return -1;
    if (seg_rc != 0)
        return 1;

    if (two_die) {
        gpu_async_arm(g->ctx[0], ev_a);
        g->as_phase = QW_ASYNC_WAIT_A;
        g->as_pos = pos;
        return 0;
    }

    /* Single die: final norm + LM head are stream-ordered launches behind
     * the segment graph — enqueue them now, then arm the event. (The
     * QWEN35_DEBUG result dumps are skipped: dbg refuses the segment path,
     * so async never engages while dumping.) */
    fc_rms_norm_ctx(g->ctx[0], sc_a->r_hidden, g->output_norm.buf,
                    sc_a->r_xnorm, cfg->n_embd, cfg->rms_norm_eps);
    gpu_matvec(&g->output, sc_a->r_xnorm, sc_a->r_logits, h.a, h.b);
    gpu_async_arm(g->ctx[0], ev_a);
    g->as_phase = QW_ASYNC_WAIT_FINAL;
    g->as_pos = pos;
    return 0;
}

/**
 * Reap / advance the in-flight step started by .._begin. Returns 1 while
 * the current phase's GPU work is still running (call again after a short
 * sleep — nothing blocks), 0 when the step is complete and s->logits
 * holds the token's logits (same presentation as the sync forward), -1 on
 * error (the machine resets; the generation should abort, exactly as a
 * NULL return from the sync forward).
 */
int pn_qwen35moe_forward_token_cuda_poll(pn_qwen35moe_model *m,
                                         pn_qwen35moe_state *s)
{
    if (!m || !s)
        return -1;
    pn_qwen35moe_gpu *g = (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m);
    if (!g || g->as_phase == QW_ASYNC_IDLE)
        return -1;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int n_layer = cfg->n_layer;
    int split = g->split_layer;
    int pos = g->as_pos;
    gpu_host_scratch h = g->host;

    if (g->as_phase == QW_ASYNC_WAIT_A) {
        if (gpu_async_reap(g->ctx[0], g->as_ev[0]) != 0)
            return 1;

        /* 2-die boundary hand-off (die A's stream is now idle): copy the
         * boundary hidden D2H, re-seed it H2D as die B's residual stream.
         * No gpu_select_die needed — async requires small-BAR mode, where
         * every die is live behind its own context. */
        gpu_die_scratch *sc_a = gpu_get_scratch(g, g->ctx[0], cfg);
        gpu_die_scratch *sc_b = gpu_get_scratch(g, g->head_ctx, cfg);
        if (!sc_a || !sc_b) {
            g->as_phase = QW_ASYNC_IDLE;
            return -1;
        }
        fc_buffer_to_cpu(sc_a->r_hidden, h.a);
        fc_buffer_from_cpu(sc_b->r_hidden, h.a);

        int seg_rc = gpu_segment_forward(m, g, g->head_ctx, 1, split, n_layer,
                                         sc_b->r_hidden, sc_b->r_xnorm,
                                         sc_b->r_blockout, s, pos, &g->host);
        if (seg_rc < 0) {
            g->as_phase = QW_ASYNC_IDLE;
            return -1;
        }
        if (seg_rc != 0) {
            /* Back segment can't replay this token (capture failure /
             * geometry change): finish it on the DIRECT per-layer path —
             * blocking host round-trips, rare, bit-identical. */
            for (int layer = split; layer < n_layer; layer++) {
                if (gpu_layer_forward(m, g, g->head_ctx, layer, sc_b->r_hidden,
                                      sc_b->r_xnorm, sc_b->r_blockout, s,
                                      pos, &g->host) < 0) {
                    g->as_phase = QW_ASYNC_IDLE;
                    return -1;
                }
            }
        }
        fc_rms_norm_ctx(g->head_ctx, sc_b->r_hidden, g->output_norm.buf,
                        sc_b->r_xnorm, cfg->n_embd, cfg->rms_norm_eps);
        gpu_matvec(&g->output, sc_b->r_xnorm, sc_b->r_logits, h.a, h.b);
        gpu_async_arm(g->head_ctx, g->as_ev[1]);
        g->as_phase = QW_ASYNC_WAIT_FINAL;
        return 1;
    }

    /* QW_ASYNC_WAIT_FINAL — the LM head result sits on the last die. */
    fc_cuda_ctx *ctx = g->head_ctx ? g->head_ctx : g->ctx[0];
    fc_event *ev = g->head_ctx ? g->as_ev[1] : g->as_ev[0];
    if (gpu_async_reap(ctx, ev) != 0)
        return 1;
    gpu_die_scratch *sc = gpu_get_scratch(g, ctx, cfg);
    if (!sc) {
        g->as_phase = QW_ASYNC_IDLE;
        return -1;
    }
    /* Top-k fetch syncs only THIS card's stream (already drained by the
     * event) — no global barrier across cards. */
    gpu_fetch_logits(g, ctx, sc->r_logits, s, cfg->n_vocab);
    g->as_phase = QW_ASYNC_IDLE;
    return 0;
}

/* pn_forward_step adapters (engine seam): pn_model/pn_state signatures
 * mirroring pn_qwen35moe_forward_cuda, advancing state->pos on completion
 * exactly as the sync adapter does. */
int pn_qwen35moe_forward_step_begin(pn_model *model, uint32_t token,
                                    pn_state *state)
{
    if (!model || !state)
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;
    return pn_qwen35moe_forward_token_cuda_begin(m, token, s, state->pos);
}

int pn_qwen35moe_forward_step_poll(pn_model *model, pn_state *state,
                                   tm_tensor **logits_out)
{
    if (!model || !state || !logits_out)
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return -1;

    int rc = pn_qwen35moe_forward_token_cuda_poll(m, s);
    if (rc != 0)
        return rc;

    int64_t shape[1] = { s->n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);
    if (!out)
        return -1;
    memcpy(tm_data_f32(out), s->logits, (size_t)s->n_vocab * sizeof(float));
    state->pos++;
    *logits_out = out;
    return 0;
}

/* ============================================================
 * MTP draft head (QWEN35_MTP=1)
 *
 * The GGUF's trailing NextN block as a draft model on the head die, outside the trunk
 * (llama.cpp graph_mtp): for the pair (h_j, t_{j+1}) — h_j the trunk's output_norm'd hidden
 * at position j, t_{j+1} the token after it —
 *   x  = eh_e . enorm(embed(t_{j+1})) + eh_h . hnorm(h_j)
 *   x += attn(attn_norm(x)) at position j over the head's OWN KV ring
 *   x += moe(post_attention_norm(x));  h' = shared_head_norm(x);  logits = head(h')
 * and the next draft chains on (h', draft).
 *
 * Every committed position owes the head one KV row. Only the K/V projections of a row are
 * needed for that (attention output and MoE feed nothing), so committed rows are caught up
 * "KV-only" in batches; the LAST row before the seed runs the full block and yields draft 0.
 * Rows wait in a backlog (h rows on the die + their next tokens) and are flushed in batches,
 * so a verify that drafts nothing (D = 0) adds no head launches.
 *
 * Drafts never change what is emitted: the verify decides every token. The head's numbers
 * only move acceptance, so it uses the prefill-shaped kernels freely and a draft vocabulary
 * (ids < QWEN35_MTP_VOCAB, default 65536, plus the special-token tail; 0 = the full head).
 * ============================================================ */

#define QW_MTP_ROWS_CAP   320   /* backlog + one prefill chunk (<= 256 rows) */
#define QW_MTP_FLUSH_AT    64   /* backlog rows that trigger a KV-only flush */
#define QW_MTP_KV_BATCH   128   /* rows per KV-only batch (one q4 window lap) */
#define QW_MTP_TOPK         8   /* pairs a draft fetches (argmax + confidence) */
#define QW_MTP_PICK_TOPK   PN_MTP_PICK_PAIRS /* pairs a draft fetches for a pick callback */
#define QW_MTP_TAIL_FIRST 248044 /* Qwen3.5/3.6 special tokens start here */

/* QWEN35_MTP_PROF=1: sync after each part of a draft and book its wall time (prof_ms). */
static int mtp_prof_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_MTP_PROF");
        v = e ? (atoi(e) > 1 ? 2 : 1) : 0;
    }
    return v;
}

static void mtp_lap(gpu_mtp *t, int part, double *t0)
{
    if (!mtp_prof_on())
        return;
    fc_sync_ctx(t->ctx);
    double now = prof_now();
    t->prof_ms[part] += (now - *t0) * 1e3;
    *t0 = now;
}

/* QWEN35_MTP_PROF=2: a KV-rows part (host embed / upload / kernels), drained. */
static void mtp_kv_lap(gpu_mtp *t, fc_cuda_ctx *ctx, int part, double *t0)
{
    fc_sync_ctx(ctx);
    double now = prof_now();
    t->kv_prof_ms[part] += (now - *t0) * 1e3;
    *t0 = now;
}

static int mtp_vocab_sub(void)
{
    const char *env = getenv("QWEN35_MTP_VOCAB");
    return env ? atoi(env) : 65536;
}

/* Upload the head next to the LM head (the last die). No-op without a kept head. */
static int mtp_upload(pn_qwen35moe_model *model, pn_qwen35moe_gpu *g,
                      const pn_qwen35moe_config *cfg)
{
    const pn_qwen35moe_mtp_head *h = pn_qwen35moe_model_mtp(model);
    gpu_mtp *t = &g->mtp;
    fc_cuda_ctx *ctx = g->ctx[g->n_die - 1];
    int nv = cfg->n_vocab, sub = mtp_vocab_sub();

    if (!h)
        return 0;
    t->ctx = ctx;
    if (gpu_upload_attn_layer(ctx, &t->d, &h->layer, cfg) < 0 ||
        gpu_weight_upload(ctx, &t->eh_e, &h->eh_e) < 0 ||
        gpu_weight_upload(ctx, &t->eh_h, &h->eh_h) < 0 ||
        gpu_vec_upload(ctx, &t->enorm, h->enorm, cfg->n_embd) < 0 ||
        gpu_vec_upload(ctx, &t->hnorm, h->hnorm, cfg->n_embd) < 0 ||
        gpu_vec_upload(ctx, &t->head_norm, h->head_norm, cfg->n_embd) < 0 ||
        !t->eh_e.bytes || !t->eh_h.bytes)
        return -1;
    /* Draft vocabulary: a row-prefix of the LM head plus the special tail, as views. */
    t->head_lo = g->output;
    t->sub_n = nv;
    t->n_tail = 0;
    if (sub > 0 && sub < QW_MTP_TAIL_FIRST && QW_MTP_TAIL_FIRST < nv) {
        t->head_lo.rows = sub;
        t->sub_n = sub;
        t->tail_first = QW_MTP_TAIL_FIRST;
        t->n_tail = nv - QW_MTP_TAIL_FIRST;
        t->head_hi = g->output;
        t->head_hi.rows = t->n_tail;
        t->head_hi.bytes = fc_buffer_offset_view(g->output.bytes,
                                                 (size_t)t->tail_first * g->output.row_stride);
        if (!t->head_hi.bytes)
            return -1;
    }
    t->on = 1;
    fprintf(stderr, "qwen35moe: MTP draft head on die %d (draft vocabulary %d + %d tail of %d)\n",
            g->die[g->n_die - 1], t->sub_n, t->n_tail, nv);
    return 0;
}

static void mtp_free(pn_qwen35moe_gpu *g)
{
    gpu_mtp *t = &g->mtp;
    gpu_attn_layer *d = &t->d;
    fc_buffer **bufs[] = { &t->e, &t->en, &t->hn, &t->x, &t->xn, &t->k, &t->v, &t->qg,
                           &t->q, &t->z, &t->aout, &t->blk, &t->tn, &t->mout, &t->h_chain,
                           &t->logits, &t->h_pend, &t->h_ver, &t->m_g, &t->l_g, &t->acc_g,
                           &d->k_cache_g, &d->v_cache_g, &d->k_sink_g, &d->v_sink_g,
                           &d->k_win_g, &d->v_win_g };

    if (!t->ctx)
        return;
    gpu_vec_free(&d->attn_norm);
    gpu_weight_free(&d->attn_q);
    gpu_weight_free(&d->attn_k);
    gpu_weight_free(&d->attn_v);
    gpu_vec_free(&d->attn_q_norm);
    gpu_vec_free(&d->attn_k_norm);
    gpu_weight_free(&d->attn_output);
    gpu_vec_free(&d->post_attention_norm);
    gpu_moe_free(&d->moe);
    gpu_weight_free(&t->eh_e);
    gpu_weight_free(&t->eh_h);
    gpu_vec_free(&t->enorm);
    gpu_vec_free(&t->hnorm);
    gpu_vec_free(&t->head_norm);
    fc_buffer_free_view(t->head_hi.bytes);   /* head_lo/head_hi alias g->output */
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        fc_buffer_free(*bufs[i]);
        *bufs[i] = NULL;
    }
    free(t->e_host);
    memset(t, 0, sizeof(*t));
}

/* pn_mtp_ring_fill_ops on the head die: stream-ordered upload, then on-stream copies. */
static int mtp_ring_put(void *ops_ctx, void *buf, const void *src, size_t bytes)
{
    (void)ops_ctx;
    if (fc_buffer_from_cpu_n_async((fc_buffer *)buf, src, bytes) != 0)
        fc_buffer_from_cpu_n((fc_buffer *)buf, src, bytes);
    return 0;
}

static int mtp_ring_copy(void *ops_ctx, void *buf, size_t dst_off, size_t src_off,
                         size_t bytes)
{
    fc_buffer_copy_d2d_ctx((fc_cuda_ctx *)ops_ctx, (fc_buffer *)buf, dst_off, (fc_buffer *)buf,
                           src_off, bytes);
    return 0;
}

/* Scratch + the head's KV ring (sized to the session's max_seq). */
static int mtp_alloc(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg,
                     const pn_qwen35moe_kv *kv0)
{
    gpu_mtp *t = &g->mtp;
    gpu_attn_layer *d = &t->d;
    fc_cuda_ctx *ctx = t->ctx;
    size_t c = (size_t)cfg->n_embd, cap = QW_MTP_ROWS_CAP;
    size_t attn = (size_t)cfg->n_head * cfg->head_dim, kvd = (size_t)cfg->n_head_kv * cfg->head_dim;
    size_t np = (size_t)FC_ATTN_ROWS_MAX * cfg->n_head * PN_QWEN35MOE_DEC_SPLIT_MAX;

    /* The head's ring below is its OWN, always q4 (the rows kernel is q4-only), filled
     * only by mtp_rows_kv from the head's projections: it never reads the trunk's KV,
     * so an fp32 trunk (Pantheon Local) drafts the same way. Drafts are only proposals;
     * the trunk verifies every token, so the output does not depend on this ring. */
    if (!t->x) {
        t->e = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->en = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->hn = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->x = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->xn = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->k = fc_buffer_alloc_ctx(ctx, cap * kvd * 4);
        t->v = fc_buffer_alloc_ctx(ctx, cap * kvd * 4);
        t->h_pend = fc_buffer_alloc_ctx(ctx, cap * c * 4);
        t->qg = fc_buffer_alloc_ctx(ctx, 2 * attn * 4);
        t->q = fc_buffer_alloc_ctx(ctx, attn * 4);
        t->z = fc_buffer_alloc_ctx(ctx, attn * 4);
        t->aout = fc_buffer_alloc_ctx(ctx, attn * 4);
        t->blk = fc_buffer_alloc_ctx(ctx, c * 4);
        t->tn = fc_buffer_alloc_ctx(ctx, c * 4);
        t->mout = fc_buffer_alloc_ctx(ctx, c * 4);
        t->h_chain = fc_buffer_alloc_ctx(ctx, c * 4);
        t->logits = fc_buffer_alloc_ctx(ctx, (size_t)(t->sub_n + t->n_tail) * 4);
        t->m_g = fc_buffer_alloc_ctx(ctx, np * 4);
        t->l_g = fc_buffer_alloc_ctx(ctx, np * 4);
        t->acc_g = fc_buffer_alloc_ctx(ctx, np * cfg->head_dim * 4);
        t->e_host = malloc(cap * c * sizeof(float));
        if (!t->e || !t->en || !t->hn || !t->x || !t->xn || !t->k || !t->v || !t->h_pend ||
            !t->qg || !t->q || !t->z || !t->aout || !t->blk || !t->tn || !t->mout ||
            !t->h_chain || !t->logits || !t->m_g || !t->l_g || !t->acc_g || !t->e_host)
            return -1;
    }
    if (d->cache_cap != kv0->max_seq) {
        pn_mtp_ring_sizes rs = pn_mtp_ring_sizes_for(cfg->n_head_kv, cfg->head_dim,
                                                     kv0->max_seq);
        pn_mtp_ring_fill_ops zero_ops = { mtp_ring_put, mtp_ring_copy, ctx };
        fc_buffer **ring[PN_MTP_RING_BUFFERS] = { &d->k_cache_g, &d->v_cache_g, &d->k_sink_g,
                                                  &d->v_sink_g, &d->k_win_g, &d->v_win_g };
        void *bufs[PN_MTP_RING_BUFFERS];
        for (size_t i = 0; i < PN_MTP_RING_BUFFERS; i++) {
            fc_buffer_free(*ring[i]);
            *ring[i] = NULL;
        }
        d->k_cache_g = fc_buffer_alloc_ctx(ctx, rs.cache_bytes);
        d->v_cache_g = fc_buffer_alloc_ctx(ctx, rs.cache_bytes);
        d->k_sink_g = fc_buffer_alloc_ctx(ctx, rs.sink_bytes);
        d->v_sink_g = fc_buffer_alloc_ctx(ctx, rs.sink_bytes);
        d->k_win_g = fc_buffer_alloc_ctx(ctx, rs.win_bytes);
        d->v_win_g = fc_buffer_alloc_ctx(ctx, rs.win_bytes);
        d->cache_cap = 0;
        if (!d->k_cache_g || !d->v_cache_g || !d->k_sink_g || !d->v_sink_g || !d->k_win_g ||
            !d->v_win_g)
            return -1;
        /* The head attends [0, pos) while only rows since its last backlog reset are this
         * session's; rows nothing ever wrote must decode finite (qwen35moe_mtp_ring.h). */
        for (size_t i = 0; i < PN_MTP_RING_BUFFERS; i++)
            bufs[i] = *ring[i];
        if (pn_mtp_ring_clear(bufs, &rs, &zero_ops) != 0)
            return -1;
        d->cache_cap = kv0->max_seq;
    }
    return 0;
}

/* Capture the trunk's output_norm'd rows of the batch that just ran (prefill chunk or
 * verify) on the head die: the backlog's h source. Called from gpu_batch_tokens_core. */
static void mtp_capture_rows(pn_qwen35moe_gpu *g, fc_cuda_ctx *ctx, fc_buffer *x_norm,
                             int n, int n_embd)
{
    gpu_mtp *t = &g->mtp;

    t->ver_n = 0;
    if (!t->on || ctx != t->ctx)
        return;
    if (t->ver_cap < n) {
        fc_buffer_free(t->h_ver);
        t->h_ver = fc_buffer_alloc_ctx(ctx, (size_t)n * n_embd * sizeof(float));
        t->ver_cap = t->h_ver ? n : 0;
        if (!t->h_ver)
            return;
    }
    fc_buffer_copy_d2d_ctx(ctx, t->h_ver, 0, x_norm, 0, (size_t)n * n_embd * sizeof(float));
    t->ver_n = n;
}

/* x = eh_proj([enorm(embed(next[r])) | hnorm(h[r])]) for n rows, attn_norm into xn, and the
 * rows' K/V appended to the head's ring at pos0.. (h: [n][c] on the head die). */
static int mtp_rows_kv(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                       const pn_qwen35moe_config *cfg, int pos0, int n,
                       const uint32_t *next, fc_buffer *h)
{
    gpu_mtp *t = &g->mtp;
    gpu_attn_layer *d = &t->d;
    fc_cuda_ctx *ctx = t->ctx;
    int c = cfg->n_embd;

    double t0 = prof_now(), tk = t0;
    const int fine = mtp_prof_on() > 1;
    if (fine)
        fc_sync_ctx(ctx);
    for (int r = 0; r < n; r++)
        pn_qwen35moe_model_embed_row(m, next[r], t->e_host + (size_t)r * c);
    if (fine)
        mtp_kv_lap(t, ctx, 0, &tk);
    /* Stream-ordered upload: the blocking copy would wait for the head die's whole
     * queue — during a pipelined prefill that is the in-flight trunk segment, which
     * serialised the 9b overlap (1k-token TTFT 9.0 s -> 13.8 s with the head fed). */
    if (fc_buffer_from_cpu_n_async(t->e, t->e_host, (size_t)n * c * sizeof(float)) != 0)
        fc_buffer_from_cpu_n(t->e, t->e_host, (size_t)n * c * sizeof(float));
    if (fine)
        mtp_kv_lap(t, ctx, 1, &tk);
    fc_rms_norm_nx_ctx(ctx, t->e, t->enorm.buf, t->en, c, cfg->rms_norm_eps, n);
    fc_rms_norm_nx_ctx(ctx, h, t->hnorm.buf, t->hn, c, cfg->rms_norm_eps, n);
    if (gpu_matvec_nx(&t->eh_e, t->en, t->x, n) < 0 ||
        gpu_matvec_nx(&t->eh_h, t->hn, t->e, n) < 0)
        return -1;
    fc_add_ctx(ctx, t->x, t->e, t->x, n * c);
    fc_rms_norm_nx_ctx(ctx, t->x, d->attn_norm.buf, t->xn, c, cfg->rms_norm_eps, n);
    if (gpu_matvec_nx(&d->attn_k, t->xn, t->k, n) < 0 ||
        gpu_matvec_nx(&d->attn_v, t->xn, t->v, n) < 0)
        return -1;
    fc_qk_rms_norm_ctx(ctx, t->k, d->attn_k_norm.buf, n * cfg->n_head_kv, cfg->head_dim,
                       cfg->rms_norm_eps);
    fc_rope_neox_partial_nx_ctx(ctx, t->k, cfg->n_head_kv, cfg->head_dim, cfg->rope_dim, pos0,
                                cfg->rope_theta, n);
    fc_attn_kv_append_q4_ctx(ctx, t->k, t->v, d->k_cache_g, d->v_cache_g, d->k_sink_g,
                             d->v_sink_g, d->k_win_g, d->v_win_g, cfg->n_head_kv,
                             cfg->head_dim, d->cache_cap, pos0, n, pn_kvq4_sink(),
                             pn_kvq4_win());
    if (fine)
        mtp_kv_lap(t, ctx, 2, &tk);
    mtp_lap(t, 0, &t0);
    t->kv_rows += n;
    return 0;
}

/* The draft from h_chain: draft-vocabulary logits, top-k, ids mapped back. The draft is the
 * argmax, or the pick callback's choice from the top QW_MTP_PICK_TOPK pairs; the confidence
 * is the argmax's softmax over the top QW_MTP_TOPK either way. */
static int mtp_pick(pn_qwen35moe_gpu *g, uint32_t *draft, float *conf)
{
    gpu_mtp *t = &g->mtp;
    gpu_host_scratch *hs = &g->host;
    const int k = t->pick ? QW_MTP_PICK_TOPK : QW_MTP_TOPK;
    int ids[QW_MTP_PICK_TOPK];
    float vals[QW_MTP_PICK_TOPK];
    double sum = 0.0;

    gpu_matvec(&t->head_lo, t->h_chain, t->logits, hs->a, hs->b);
    if (t->n_tail) {
        fc_buffer *hi = fc_buffer_offset_view(t->logits, (size_t)t->sub_n * sizeof(float));
        if (!hi)
            return -1;
        gpu_matvec(&t->head_hi, t->h_chain, hi, hs->a, hs->b);
        fc_buffer_free_view(hi);
    }
    if (fc_topk_logits_ctx(t->ctx, t->logits, t->sub_n + t->n_tail, k, ids, vals) != 0)
        return -1;
    for (int i = 0; i < k; i++)
        ids[i] = ids[i] < t->sub_n ? ids[i] : t->tail_first + (ids[i] - t->sub_n);
    for (int i = 0; i < QW_MTP_TOPK; i++)
        sum += exp((double)(vals[i] - vals[0]));
    *draft = t->pick ? t->pick(t->pick_ctx, t->pick_index, ids, vals, k) : (uint32_t)ids[0];
    *conf = sum > 0.0 ? (float)(1.0 / sum) : 1.0f;
    return 0;
}

/* The full block for row r of t->x (its K/V already appended at pos): attention, MoE,
 * shared_head_norm into h_chain, then the draft. */
static int mtp_row_full(pn_qwen35moe_gpu *g, const pn_qwen35moe_config *cfg, int r, int pos,
                        uint32_t *draft, float *conf)
{
    gpu_mtp *t = &g->mtp;
    gpu_attn_layer *d = &t->d;
    fc_cuda_ctx *ctx = t->ctx;
    gpu_host_scratch *hs = &g->host;
    gpu_die_scratch *sc = gpu_get_scratch(g, ctx, cfg);
    size_t row_b = (size_t)cfg->n_embd * sizeof(float);
    int attn = cfg->n_head * cfg->head_dim, rc = -1;
    fc_buffer *x = fc_buffer_offset_view(t->x, (size_t)r * row_b);
    fc_buffer *xn = fc_buffer_offset_view(t->xn, (size_t)r * row_b);

    double t0 = prof_now();
    if (!sc || !x || !xn)
        goto done;
    gpu_matvec(&d->attn_q, xn, t->qg, hs->a, hs->b);
    fc_attn_qz_split_ctx(ctx, t->qg, t->q, t->z, cfg->n_head, cfg->head_dim, 1);
    fc_qk_rms_norm_ctx(ctx, t->q, d->attn_q_norm.buf, cfg->n_head, cfg->head_dim,
                       cfg->rms_norm_eps);
    fc_rope_neox_partial_nx_ctx(ctx, t->q, cfg->n_head, cfg->head_dim, cfg->rope_dim, pos,
                                cfg->rope_theta, 1);
    if (fc_attn_flash_decode_rows_q4_ctx(ctx, t->q, d->k_cache_g, d->v_cache_g, d->k_sink_g,
                                         d->v_sink_g, d->k_win_g, d->v_win_g, t->m_g, t->l_g,
                                         t->acc_g, NULL, t->aout, cfg->n_head,
                                         cfg->n_head_kv, cfg->head_dim, d->cache_cap, pos, 1,
                                         dec_keys_per_split(), PN_QWEN35MOE_DEC_SPLIT_MAX,
                                         pn_kvq4_sink(), pn_kvq4_win(), attn_sink(), 0,
                                         qw_scale_inv_sqrt(cfg->head_dim)) != 0)
        goto done;
    fc_sigmoid_ctx(ctx, t->z, attn);
    fc_mul_ctx(ctx, t->aout, t->z, t->aout, attn);
    gpu_matvec(&d->attn_output, t->aout, t->blk, hs->a, hs->b);
    fc_add_ctx(ctx, x, t->blk, x, cfg->n_embd);
    fc_rms_norm_nx_ctx(ctx, x, d->post_attention_norm.buf, t->tn, cfg->n_embd,
                       cfg->rms_norm_eps, 1);
    mtp_lap(t, 1, &t0);
    if (gpu_moe_block(ctx, &d->moe, cfg, t->tn, t->mout, x, sc, hs) < 0)
        goto done;
    mtp_lap(t, 2, &t0);
    fc_rms_norm_nx_ctx(ctx, x, t->head_norm.buf, t->h_chain, cfg->n_embd, cfg->rms_norm_eps, 1);
    rc = mtp_pick(g, draft, conf);
    mtp_lap(t, 3, &t0);
    t->full_rows++;
done:
    fc_buffer_free_view(x);
    fc_buffer_free_view(xn);
    return rc;
}

/* KV-only catch-up of backlog rows [0, n) (their next tokens all known); the rest move down.
 * Batches of at most QW_MTP_KV_BATCH rows keep one append inside one lap of the q4 window. */
static int mtp_flush(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g,
                     const pn_qwen35moe_config *cfg, int n)
{
    gpu_mtp *t = &g->mtp;
    size_t row_b = (size_t)cfg->n_embd * sizeof(float);

    if (n <= 0)
        return 0;
    for (int off = 0; off < n; off += QW_MTP_KV_BATCH) {
        int nb = n - off < QW_MTP_KV_BATCH ? n - off : QW_MTP_KV_BATCH;
        fc_buffer *h = off ? fc_buffer_offset_view(t->h_pend, (size_t)off * row_b) : t->h_pend;
        int rc = h ? mtp_rows_kv(m, g, cfg, t->pend_base + off, nb, t->pend_next + off, h) : -1;
        if (off)
            fc_buffer_free_view(h);
        if (rc < 0)
            return -1;
    }
    for (int r = n; r < t->pend_n; r++) {
        fc_buffer_copy_d2d_ctx(t->ctx, t->h_pend, (size_t)(r - n) * row_b, t->h_pend,
                               (size_t)r * row_b, row_b);
        t->pend_next[r - n] = t->pend_next[r];
    }
    t->pend_n -= n;
    t->pend_base += n;
    return 0;
}

/* Join the last trunk batch's rows to the backlog: rows at positions state->pos - n.., whose
 * input tokens were tokens[0..n). Resets the head on a new session or a restart. */
static int mtp_push(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, pn_qwen35moe_state *s,
                    int pos_end, const uint32_t *tokens, int n)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    gpu_mtp *t = &g->mtp;
    size_t row_b = (size_t)cfg->n_embd * sizeof(float);
    int pos0 = pos_end - n;

    if (!t->on || n < 1 || n > QW_MTP_KV_BATCH || n > t->ver_n || s->n_attn < 1 ||
        mtp_alloc(g, cfg, &s->kv[0]) < 0)
        return -1;
    if (t->state != (const void *)s || pos0 == 0 ||
        (t->pend_n > 0 && t->pend_base + t->pend_n != pos0) ||
        (t->pend_n == 0 && t->next_pos != pos0)) {
        t->state = (const void *)s;
        t->pend_n = 0;
    }
    if (t->pend_n > 0)
        t->pend_next[t->pend_n - 1] = tokens[0];
    if (t->pend_n + n > QW_MTP_ROWS_CAP && mtp_flush(m, g, cfg, t->pend_n) < 0)
        return -1;
    if (t->pend_n == 0)
        t->pend_base = pos0;
    fc_buffer_copy_d2d_ctx(t->ctx, t->h_pend, (size_t)t->pend_n * row_b, t->h_ver, 0,
                           (size_t)n * row_b);
    for (int r = 0; r < n; r++)
        t->pend_next[t->pend_n + r] = r + 1 < n ? tokens[r + 1] : 0u;
    t->pend_n += n;
    t->next_pos = pos_end;
    if (t->pend_n > QW_MTP_FLUSH_AT && mtp_flush(m, g, cfg, t->pend_n - 1) < 0)
        return -1;
    return 0;
}

/* Up to kmax drafts for the token after `seed` (which sits at pos_seed): the backlog's
 * catch-up with its last row in full gives draft 0, then each draft chains on the head's
 * own output. `more` (NULL = always) decides after each draft whether to draft again. */
static int mtp_draft(pn_qwen35moe_model *m, pn_qwen35moe_gpu *g, pn_qwen35moe_state *s,
                     int pos_seed, uint32_t seed, int kmax, uint32_t *drafts, float *confs,
                     pn_mtp_more_fn more, void *more_ctx, pn_mtp_pick_fn pick, void *pick_ctx)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    gpu_mtp *t = &g->mtp;
    double t0 = prof_now();
    int n = t->pend_n, got = 0;

    if (!t->on || kmax < 1 || n < 1 || t->state != (const void *)s ||
        t->pend_base + n != pos_seed)
        return -1;
    t->pend_next[n - 1] = seed;
    t->pick = pick;
    t->pick_ctx = pick_ctx;
    t->pick_index = 0;
    if (mtp_rows_kv(m, g, cfg, t->pend_base, n, t->pend_next, t->h_pend) < 0 ||
        mtp_row_full(g, cfg, n - 1, pos_seed - 1, &drafts[0], &confs[0]) < 0) {
        t->pick = NULL;
        return -1;
    }
    t->pend_n = 0;
    for (got = 1; got < kmax; got++) {
        if (more && !more(more_ctx, got, confs[got - 1]))
            break;
        t->pick_index = got;
        if (mtp_rows_kv(m, g, cfg, pos_seed - 1 + got, 1, &drafts[got - 1], t->h_chain) < 0 ||
            mtp_row_full(g, cfg, 0, pos_seed - 1 + got, &drafts[got], &confs[got]) < 0)
            break;
    }
    t->pick = NULL;
    t->drafts += got;
    t->draft_ms += (prof_now() - t0) * 1e3;
    return got;
}

int pn_qwen35moe_mtp_ready(pn_model *model)
{
    pn_qwen35moe_model *m = model ? (pn_qwen35moe_model *)model->weights : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;

    return g && g->mtp.on;
}

int pn_qwen35moe_mtp_push_rows(pn_model *model, pn_state *state, const uint32_t *tokens,
                               int n)
{
    if (!model || !state || !tokens || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;

    if (!g || !s)
        return -1;
    return mtp_push(m, g, s, state->pos, tokens, n);
}

int pn_qwen35moe_mtp_rewind(pn_model *model, pn_state *state)
{
    if (!model || !state || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;

    if (!g || !g->mtp.on)
        return -1;
    g->mtp.state = state->kv_cache;
    g->mtp.pend_n = 0;
    g->mtp.next_pos = state->pos;
    return 0;
}

int pn_qwen35moe_mtp_draft_pick(pn_model *model, pn_state *state, uint32_t seed, int kmax,
                                uint32_t *drafts, float *confs, pn_mtp_more_fn more,
                                void *more_ctx, pn_mtp_pick_fn pick, void *pick_ctx)
{
    if (!model || !state || !drafts || !confs || !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;

    if (!g || !s)
        return -1;
    if (!g->host_ok) {
        if (gpu_host_scratch_init(&g->host, pn_qwen35moe_model_config(m)) < 0)
            return -1;
        g->host_ok = 1;
    }
    return mtp_draft(m, g, s, state->pos, seed, kmax, drafts, confs, more, more_ctx, pick,
                     pick_ctx);
}

int pn_qwen35moe_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int kmax,
                           uint32_t *drafts, float *confs, pn_mtp_more_fn more,
                           void *more_ctx)
{
    return pn_qwen35moe_mtp_draft_pick(model, state, seed, kmax, drafts, confs, more, more_ctx,
                                       NULL, NULL);
}

int pn_qwen35moe_mtp_stats(pn_model *model, pn_mtp_stats *out)
{
    pn_qwen35moe_model *m = model ? (pn_qwen35moe_model *)model->weights : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;

    if (!g || !out || !g->mtp.on)
        return -1;
    out->drafts = g->mtp.drafts;
    out->kv_rows = g->mtp.kv_rows;
    out->full_rows = g->mtp.full_rows;
    out->draft_ms = g->mtp.draft_ms;
    for (int i = 0; i < 4; i++)
        out->prof_ms[i] = g->mtp.prof_ms[i];
    return 0;
}

int pn_qwen35moe_verify_graph_stats(pn_model *model, long *captures, long *replays,
                                    long *failures)
{
    pn_qwen35moe_model *m = model ? (pn_qwen35moe_model *)model->weights : NULL;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;

    if (!g || !captures || !replays || !failures)
        return -1;
    *captures = *replays = *failures = 0;
    for (int d = 0; d < PN_DIE_LAYER_SPLIT_MAX_DIES; d++)
        for (int r = 0; r < 5; r++) {
            *captures += g->vgraph[d][r].captures;
            *replays += g->vgraph[d][r].replays;
            *failures += g->vgraph[d][r].failures;
        }
    return 0;
}

int pn_qwen35moe_spec_verify_topk(pn_model *model, const uint32_t *tokens, int n,
                                  pn_state *state, int k, int *ids, float *vals)
{
    if (!model || !state || !ids || !vals || k < 1 || k > FC_TOPK_MAX ||
        !qw_model_uses_qwen35moe_state(model))
        return -1;
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_gpu *g = m ? (pn_qwen35moe_gpu *)pn_qwen35moe_model_gpu(m) : NULL;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    int32_t argmax_unused[FC_ATTN_ROWS_MAX + 1];

    if (!g || !s || n > FC_ATTN_ROWS_MAX)
        return -1;
    g->spec_topk_k = k;
    g->spec_topk_ids = ids;
    g->spec_topk_vals = vals;
    int rc = pn_qwen35moe_spec_verify_tokens_cuda(m, tokens, n, s, state->pos, argmax_unused);
    g->spec_topk_k = 0;
    g->spec_topk_ids = NULL;
    g->spec_topk_vals = NULL;
    return rc;
}

#endif /* USE_CUDA */
