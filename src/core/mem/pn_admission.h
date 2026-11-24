/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_admission.h
 * @brief Pre-flight admission control — the load-bearing MemoryAware fix
 *        (MemoryAware.md §6). Before Pantheon starts any memory-growing work it
 *        *computes* what that work will need and refuses / caps it if it won't
 *        fit within reserved host + per-die headroom, instead of attempting the
 *        allocation and OOM-rebooting the box (the 32k full-attention prefill)
 *        or cascading into `cuMemAlloc failed: 2` (the model-load path).
 *
 *        Engine-level by decision (§11 Q7): pn_admit_request lives next to
 *        pn_load and is called from the serve layer AND is reachable by the
 *        eval path (eval_ppl) — which is exactly what crash-looped. The cost of
 *        an N-token prompt is estimated from a measured per-prompt-length LUT
 *        (§11 Q1) plus a linear per-token KV term; refine to a formula later.
 *
 * @status Phase 2 — admission gates, now PATH-AWARE: a GPU-resident model
 *         admits the full context up to max_seq (per-request host cost is ~0,
 *         measured on a K80 host); the conservative LUT models the CPU-fallback path
 *         only. Model loads additionally budget the fixed GPU-upload host
 *         spike (PN_LOAD_SPIKE_BYTES) — the real 8 GB-box squeeze.
 * @issues Cost LUT is seeded from the spec's measured points; re-measure on a
 *         K80 host at 1k/2k/4k/8k/16k/32k and update PN_ADMIT_LUT to tighten.
 *         mr_die_weight_share (model_registry.c) projects a die's weights from
 *         the GGUF tensor table, i.e. FILE bytes — but a quant that repacks on
 *         upload costs more resident VRAM than it costs on disk. Q4_K_M matches
 *         its file closely; Q3_K_M does not (K80: projected 7425 MiB/die,
 *         real residency ~11.8 GiB/die including scratch). Both this gate and
 *         gpu_admit_die_share inherit that under-projection. Fixing it needs
 *         upload-bytes accounting AND a rethink of the per-die reserve, since
 *         the real Q3_K_M residency legitimately exceeds total - reserve.
 * @todo None
 */

#ifndef PN_ADMISSION_H
#define PN_ADMISSION_H

#include <stdbool.h>
#include <stdint.h>

#include "pn_mem_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The verdict from an admission check. */
typedef enum {
    PN_ADMIT_OK      = 0,   /* fits — run it                                    */
    PN_ADMIT_CAP     = 1,   /* won't fit at N, but a smaller window would —     */
                            /* run capped to `fit_tokens` and tell the client   */
    PN_ADMIT_REJECT  = 2,   /* won't fit at all / host under stall — 503 + wait */
} pn_admit_verdict;

/* Result of pn_admit_request: the verdict, a client-facing reason, the largest
 * prompt length that WOULD fit (for the CAP path), and the projected bytes. */
typedef struct {
    pn_admit_verdict verdict;
    int              fit_tokens;      /* for CAP: the capped prompt length      */
    uint64_t         host_need_bytes; /* projected host cost of this request     */
    uint64_t         host_avail_bytes;/* host allocatable at decision time       */
    const char      *reason;          /* short, client-safe explanation          */
} pn_admit_result;

/* --- Cost model (measured LUT + linear KV; MemoryAware.md §6.2, §11 Q1) --- */

/* Per-token host KV cost — a GENERIC FALLBACK ONLY. The correct number is
 * per-model, computed from the loaded model's real geometry by
 * pn_kv_bytes_per_token() (pn_kv_geometry.h) and passed into the functions
 * below as `kv_per_token`. These constants are used ONLY when the caller passes
 * kv_per_token == 0 (geometry unavailable), so a mispriced model can never be
 * priced at zero. They model a dense all-layers-KV shape and will mis-price a
 * hybrid — always prefer the geometry value. */
#define PN_ADMIT_KV_BYTES_PER_TOKEN_Q4   5898ULL     /* ~5.76 KiB (fallback)  */
#define PN_ADMIT_KV_BYTES_PER_TOKEN_FP32 40960ULL    /* ~40 KiB   (fallback)  */

/* Resolve the effective per-token KV bytes: the model-real value when non-zero,
 * else the conservative generic constant for the KV precision. Pure. */
uint64_t pn_admit_kv_per_token(uint64_t kv_per_token, bool kv_q4);

/* The KV bytes ONE die of a split holds at `max_ctx`: the whole model's KV,
 * scaled by the fraction of layers that die owns. The weights are sliced by
 * layer range, so the KV projected on top of them has to be sliced the same
 * way — charging every die the whole model's KV is a clean 2x over-projection
 * of the KV term on a 2-die split, and it grows with the cap, which is where it
 * does damage.
 *
 * `kv_per_token` is the model-real whole-model figure when the caller has a
 * loaded model in hand, else 0 for the generic constant (pn_admit_kv_per_token).
 * `n_layer <= 0` or `n_layers_here <= 0` means the layout could not be read:
 * charge the WHOLE model's KV rather than guess a share, so an unfamiliar
 * layout is gated conservatively rather than admitted blind. Pure. */
uint64_t pn_admit_die_kv_bytes(uint64_t kv_per_token, bool kv_q4, int max_ctx,
                               int n_layers_here, int n_layer);

/* GPU-path per-die scratch reserve for the load gates: everything a load puts
 * on ONE die that is neither its weight slice nor the single-session KV the
 * gate prices separately. The host activation LUT does not apply here — on the
 * GPU path the O(N^2) scores live on-die in bounded tiles.
 *
 * This has two parts, and only one of them is fixed:
 *
 *   BASE — batch prefill activations, flash-attention tiles, the batched
 *   [slots][n_vocab] logits buffer, captured-graph descriptors and per-die
 *   staging. These do NOT grow with the context: prefill runs in
 *   QWEN35_PREFILL_CHUNK-sized chunks, and the decode split partials are
 *   allocated to PN_QWEN35MOE_DEC_SPLIT_MAX splits, not to seq_len
 *   (qwen35moe_forward_cuda.c).
 *
 *   PER-TOKEN — the batched serving path allocates the on-die KV as a PER-SLOT
 *   ring, [nslots][n_head_kv][max_seq][row] per layer plus q4 fp16 sink/window
 *   shadows (gpu_attn_slot_ring_ensure), while the gates' kv_bytes term prices
 *   ONE session. The slots the gate does not price are therefore proportional
 *   to the window the model is capped to: at the default 8 sessions and a
 *   half-model layer share that is ~24 KiB per token of cap.
 *
 * So a single fixed constant was a context measurement wearing a constant's
 * clothes — it charged a 32k-window price to a model capped at 2048. The
 * constants below are calibrated so the total lands EXACTLY on the previously
 * measured value at a 32768 window (2-die Q3_K_M @ 32k q4-KV: ~2 GiB non-weight
 * per die including the driver context, which the free-VRAM query has already
 * paid for):
 *
 *     512 MiB + 32768 * 32 KiB = 1536 MiB   — the old constant, unchanged
 *
 * and the old constant is also kept as a CEILING, so this repricing can only
 * ever charge LESS than before, never more. That matters because the per-token
 * slope is slot-dependent — it is calibrated at the default 8 sessions, and a
 * serve pinned to `--max-sessions 2` allocates a quarter of that ring. Letting
 * the line run past its calibration point would have charged a 131072-window
 * production serve 4.6 GiB for a ring that costs it 768 MiB, and refused a
 * split that runs today. Above 32768 we have no measurement, so the honest
 * thing is to keep charging exactly what we charged before rather than
 * extrapolate a number we never took.
 *
 * Behaviour at and above the one measured point is therefore identical; only
 * windows BELOW it are repriced. A 2048-token cap now costs 576 MiB, against a
 * measured ~400-480 MiB of real non-weight bytes actually added per die by
 * a 2-die Qwen3.6-35B-A3B Q4_K_M load at that cap. */
#define PN_ADMIT_DIE_SCRATCH_BASE_BYTES      (512ULL * 1024 * 1024)  /* 512 MiB */
#define PN_ADMIT_DIE_SCRATCH_BYTES_PER_TOKEN (32ULL * 1024)          /* 32 KiB  */
#define PN_ADMIT_DIE_SCRATCH_MAX_BYTES       (1536ULL * 1024 * 1024) /* 1.5 GiB */

/* The per-die GPU scratch reserve for a model capped to `max_ctx` tokens: the
 * fixed base plus the per-token ring term, clamped to
 * PN_ADMIT_DIE_SCRATCH_MAX_BYTES so this never charges more than the flat
 * constant it replaces. max_ctx <= 0 (no cap known) yields the base alone.
 * Pure. */
uint64_t pn_admit_die_scratch_bytes(int max_ctx);

/* Model-LOAD host spike (measured on a K80 host 2026-07-09): uploading weights to the
 * dies transiently costs ~2.4 GB of host anon RSS for ~55 s (dequant/staging
 * buffers riding the slow host link) before releasing to ~1.1 GB steady-state.
 * On the 8 GB box THIS transient — not per-request prefill, which is ~0 on the
 * GPU path — is the real squeeze, so model-load admission budgets it as a
 * fixed cost. Env override PN_LOAD_SPIKE_BYTES. */
#define PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT  (2400ULL * 1024 * 1024)  /* ~2.4 GB */

/* Estimate the peak *host* bytes a prefill of n_prompt tokens will transiently
 * need: the O(N^2)-ish activation peak (interpolated from the measured LUT) plus
 * the linear KV growth. `kv_q4` selects the per-token KV term. Pure — no I/O.
 * This is the number the OOM-reboot lacked a check against. */
uint64_t pn_admit_host_cost(int n_prompt, bool kv_q4, uint64_t kv_per_token);

/* Estimate the *VRAM* bytes a prefill of n_prompt tokens adds on the model's
 * die: the on-die KV growth + flash-attention scratch. Pure. `kv_per_token` is
 * the model-real per-token KV bytes (0 = use the generic constant). */
uint64_t pn_admit_die_cost(int n_prompt, bool kv_q4, uint64_t kv_per_token);

/* The effective model-load host-spike budget in bytes, honoring the
 * PN_LOAD_SPIKE_BYTES env override (values <= 0 or garbage fall back to the
 * default). Pure — safe to call anywhere. */
uint64_t pn_load_spike_bytes(void);

/* --- The two admission gates (MemoryAware.md §6.1, §6.2) --- */

/* Request admission (§6.2), PATH-AWARE. Given the prompt length and the live
 * memory state, decide fit / cap / reject.
 *
 * `gpu_resident` selects the cost model:
 *   - true (weights uploaded, the CUDA forward is selected): the per-request
 *     prefill host cost is ~0 — the O(N^2) scores live on-die in bounded flash
 *     tiles and both host + die KV are preallocated at max_seq by the load
 *     admission (measured on a K80 host 2026-07-09: serve RSS never moved to ~12.8k).
 *     The only length-triggered flash->host-softmax fallback fires at
 *     n > QWEN35_MAX_SEQ (qwen35moe_forward_cuda.c), so the FULL context is
 *     admitted up to `max_window` and CAPPED only past it. Host can still
 *     REJECT, but only on genuine pressure (sustained PSI stall, or
 *     MemAvailable already inside the reserve).
 *   - false (CPU-forward / host-fallback path): the conservative LUT + linear
 *     KV cost is checked against host (MemAvailable - reserve) AND, if any die
 *     is reporting, that die's free VRAM — the path that OOM-rebooted the box.
 *
 * `die_id` is the model's die (-1 = no die / CPU path, host-only check).
 * `max_window` caps the CAP fit_tokens to a runnable window, and on both paths a
 * prompt longer than it is CAPPED even when memory holds it. Fills *out.
 * Never allocates, never blocks.
 *
 * Thread-safe: yes (reads a snapshot copy)
 * Reentrant: yes */
void pn_admit_request(int n_prompt, bool kv_q4, uint64_t kv_per_token,
                      bool gpu_resident, int die_id, int max_window,
                      const pn_mem_state *state, pn_admit_result *out);

/* Model-load admission (§6.1). Before uploading weights to a die, check that
 * weights + projected KV @ max ctx + an activation reserve fit within the die's
 * free VRAM (from `die_free_bytes`) with the per-die reserve held back. Returns
 * true to proceed, false to refuse cleanly (caller logs + picks another die or
 * exits with a clear message instead of the failed:2 cascade). `need_out`, if
 * non-NULL, receives the projected total need in bytes.
 *
 * This is the CPU-RESIDENT gate: its activation reserve is the host LUT, which
 * models the CPU forward. A load that will run on an accelerator must use
 * pn_admit_model_load_die / _die_budgeted instead — the LUT extrapolates to
 * tens of GB at a wide window and refuses GPU loads that fit comfortably.
 *
 * `die_free_bytes` is RAW free VRAM as the hardware reports it — this function
 * holds the per-die reserve back itself. Callers holding a die-manager budget
 * (pn_dm_die.vram_budget, which is already total minus the reserve) must use a
 * _budgeted entry point instead, or the reserve is charged twice.
 *
 * Thread-safe: yes
 * Reentrant: yes */
bool pn_admit_model_load(uint64_t weights_bytes, int max_ctx, bool kv_q4,
                         uint64_t kv_per_token, uint64_t die_free_bytes,
                         uint64_t *need_out);

/* Per-die model-load admission for a GPU load (§6.1) — the split gate, and the
 * single-die gate that mirrors it. Checks that ONE die can hold its share of
 * the weights (`weights_bytes`, this die's slice, not the whole model) + its
 * share of the projected KV (`kv_bytes`, this die's layers only) + the GPU-path
 * scratch reserve for a `max_ctx`-token window, with the per-die reserve held
 * back from `die_free_bytes`. Unlike pn_admit_model_load it does NOT add the
 * host activation LUT — that models the CPU-forward path; a GPU load's on-die
 * scratch is bounded (tiled flash attention). Returns true to proceed, false to
 * refuse cleanly. `need_out`, if non-NULL, receives the projected need for this
 * die in bytes.
 *
 * The scratch reserve and the per-die reserve are NOT the same quantity under
 * two names, and summing them is not a double count:
 *
 *   - the scratch reserve is an INCREMENT — bytes this load will ADD to the
 *     die, on top of its weights and single-session KV;
 *   - the per-die reserve is a FLOOR on free VRAM that must survive the peak,
 *     so the driver can still satisfy its own late allocations without the
 *     `cuMemAlloc failed: 2` cascade.
 *
 * Nor does subtracting the reserve from FREE charge already-resident driver
 * overhead twice, which is how it reads at a glance. `die_free_bytes` is
 * `total - already_allocated`, so `free - reserve` is identically
 * `(total - reserve) - already_allocated`: the reserve is charged once, against
 * total, exactly as pn_memwatch.h defines it and as die_manager's
 * `vram_budget = total - reserve` applies it. K80 evidence that the sum is not
 * over-tight: a 2-die Qwen3.6-35B-A3B Q4_K_M @ ctx-cap 2048 peaked at 10,570 /
 * 10,879 MiB of 12,206, i.e. inside `total - reserve` (11,182) on both dies —
 * the floor was genuinely respected by the real run, not merely projected.
 *
 * Thread-safe: yes
 * Reentrant: yes */
bool pn_admit_model_load_die(uint64_t weights_bytes, uint64_t kv_bytes,
                             int max_ctx, uint64_t die_free_bytes,
                             uint64_t *need_out);

/* As pn_admit_model_load_die, but for a caller whose free figure is ALREADY net
 * of the per-die reserve — the die manager subtracts it once when it sets
 * vram_budget, so anything derived from vram_budget has paid it.
 *
 * The projection is identical; only the reserve handling differs. Getting that
 * pairing wrong has bitten twice, in opposite directions:
 *
 *   - mr_load passed a budget-derived free to the RAW entry point and so
 *     charged the 1 GiB reserve twice on every registry load, refusing a 2-die
 *     Qwen3.6-35B-A3B Q4_K_M whose own printed numbers said it fit;
 *   - and it reached for the CPU-resident projection, so once the gate started
 *     honouring --ctx-cap the host LUT it carries went with it. At the deployed config's
 *     --ctx-cap 131072 that charged 48.2 GiB of host activation to an
 *     accelerator load (K80, 2026-08-21: "die 4 has 11182 MiB free, needs
 *     56363 MiB", of which weights were 7425 and KV 737), which would have
 *     refused the live qwen36-stock serve at startup.
 *
 * A GPU placement is a GPU load. This is the entry point it belongs at.
 *
 * Thread-safe: yes
 * Reentrant: yes */
bool pn_admit_model_load_die_budgeted(uint64_t weights_bytes, uint64_t kv_bytes,
                                      int max_ctx,
                                      uint64_t die_budget_free_bytes,
                                      uint64_t *need_out);

/* Model-load HOST-spike admission (§6.1 refinement). The GPU upload transiently
 * costs ~PN_LOAD_SPIKE_BYTES of host anon RSS (see the constant above) — check
 * that MemAvailable can absorb that fixed spike on top of the host reserve
 * BEFORE the multi-minute upload starts, so a squeezed host refuses the load
 * cleanly instead of tipping over mid-upload. Applies to every GPU load path
 * (single-die and split) — call once per load attempt. Returns true to
 * proceed, false to refuse. `spike_out`, if non-NULL, receives the budgeted
 * spike in bytes.
 *
 * Thread-safe: yes
 * Reentrant: yes */
bool pn_admit_load_spike(uint64_t host_avail_bytes, uint64_t host_total_bytes,
                         uint64_t *spike_out);

/* Human-readable verdict name, for logging. */
const char *pn_admit_verdict_name(pn_admit_verdict v);

#ifdef __cplusplus
}
#endif

#endif /* PN_ADMISSION_H */
