/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe.h
 * @brief Qwen3.5-MoE (Qwen3-Next lineage) adapter for Pantheon — opaque model
 *        handle plus the pn_model/pn_state adapter API. Hybrid Gated-Delta-Net
 *        SSM + sparse full-attention + shared-expert MoE.
 *
 * @status A6: full hybrid forward + dual-state lifecycle assembled (forward
 *         orchestration in qwen35moe_forward.c). Runs end-to-end producing
 *         finite logits; numeric correctness vs the oracle is A8.
 * @issues None
 * @todo
 *   - [ ] A8: validate logits vs the llama.cpp oracle (numeric correctness)
 */

#ifndef PN_QWEN35MOE_H
#define PN_QWEN35MOE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../core/engine.h"
#include "../formats/gguf.h"
#include "gpu_attn_path_tracker.h"
#include "gpu_graph_capture_tracker.h"

/* Single source of truth for the env-overridable KV-window ceiling
 * (QWEN35_MAX_SEQ). The KV ring is allocated up to this cap (qwen35moe_max_seq),
 * so admission MUST promise the same ceiling — otherwise a QWEN35_MAX_SEQ above
 * the admission cap would let the ring allocate larger than admission ever
 * checked. Kept here (not in qwen35moe_forward.c) so the engine admission path
 * and the CUDA admit gate share the exact number. Raised to 256K for
 * long-context serving; the default window (PN_QWEN35MOE_MAX_SEQ) is unchanged,
 * so a plain run reserves no more memory than before. */
#define PN_QWEN35MOE_MAX_SEQ_CAP 262144

/* The hybrid KV ring's DEFAULT window (tokens) when neither a per-model ctx_cap
 * nor QWEN35_MAX_SEQ is set. In the header (shared with the engine admission
 * path) so admission promises exactly the window the ring is sized to. */
#define PN_QWEN35MOE_MAX_SEQ 4096

/* Opaque weight/model handle. Concrete definition lives in qwen35moe.c. */
typedef struct pn_qwen35moe_model pn_qwen35moe_model;

/**
 * Load a Qwen3.5-MoE GGUF model: parses all qwen35moe.* config keys, derives
 * the per-layer type map (full-attention when (i+1) % full_attention_interval
 * == 0, else SSM), and loads every per-layer + model-level tensor.
 * Keeps the GGUF file open (mmap backing for quantized weights).
 *
 * @param gguf    Open GGUF file handle (ownership stays with the caller).
 * @param config  Engine config to populate from GGUF metadata.
 * @return        Opaque model handle, or NULL on failure.
 *
 * Thread-safe: no (single load call)
 * Reentrant: no
 * Signal-safe: no
 */
pn_qwen35moe_model *pn_qwen35moe_load(pn_gguf_file *gguf, pn_config *config);

/**
 * Free a Qwen3.5-MoE model handle. Does NOT close the GGUF file — the engine
 * owns that and closes it after this returns.
 */
void pn_qwen35moe_free(pn_qwen35moe_model *model);

/**
 * Return the GGUF file backing this model (mmap source for quantized weights).
 * The engine uses this to close the file after pn_qwen35moe_free(). Does not
 * transfer ownership.
 */
pn_gguf_file *pn_qwen35moe_gguf(const pn_qwen35moe_model *model);

/**
 * Real per-token KV growth in bytes for THIS model, across its full-attention
 * layers only (the hybrid stack's SSM layers hold recurrent state, not a
 * per-token-growing KV cache). This is the honest number admission must use for
 * this arch — the generic PN_ADMIT_KV_BYTES_PER_TOKEN_* constants model a dense
 * all-layers-KV model and badly misprice a hybrid.
 *
 *   per_token = n_attn * n_head_kv * 2(K+V) * bytes_per_head_vec
 * where bytes_per_head_vec is the packed Q4 row size (pn_kvq4_row_bytes) under
 * `kv_q4`, else head_dim * sizeof(float).
 *
 * @param model  Loaded qwen35moe/bonsai model handle.
 * @param kv_q4  true if the KV cache is stored Q4 (the serving default).
 * @return       Whole-model KV bytes added per admitted token, or 0 if model
 *               is NULL / geometry is not yet available.
 *
 * Thread-safe: yes (reads immutable config)
 * Reentrant: yes
 * Signal-safe: yes
 */
uint64_t pn_qwen35moe_kv_bytes_per_token(const pn_qwen35moe_model *model,
                                         bool kv_q4);

/**
 * Forward pass conforming to pn_forward_fn (implemented in qwen35moe_forward.c).
 * Runs the full hybrid stack for one token and returns owned logits [n_vocab].
 */
tm_tensor *pn_qwen35moe_forward(pn_model *model, uint32_t token, pn_state *state);

#ifdef USE_CUDA
/**
 * Single-die GPU forward conforming to pn_forward_fn (implemented in
 * qwen35moe_forward_cuda.c). Selected by the engine when fc_available() and the
 * model's weights have been uploaded via pn_qwen35moe_gpu_upload(); otherwise
 * the engine keeps pn_qwen35moe_forward (CPU). Single die only; the 2-die swap
 * is a stubbed seam in qwen35moe_forward_cuda.c.
 */
tm_tensor *pn_qwen35moe_forward_cuda(pn_model *model, uint32_t token,
                                     pn_state *state);

/**
 * BATCHED prefill forward (implemented in qwen35moe_forward_cuda.c): runs n
 * tokens of ONE sequence through the whole stack in a single pass — batched
 * matvecs (fc_matvec_*_nx), SSM state scans, causal attention over all n
 * query positions, per-token MoE routing — and returns the LAST position's
 * logits [n_vocab] (owned tensor), advancing state->pos by n. The KV cache,
 * SSM state and per-position logits are bit-identical to n sequential
 * single-token forwards; per_pos_logits (optional, [n * n_vocab]) receives
 * every position's full logits — the correctness-gate hook. Falls back
 * internally to sequential single-token forwards if the batched path cannot
 * run (exotic quant mix), so callers never lose correctness.
 */
tm_tensor *pn_qwen35moe_forward_batch_cuda(pn_model *model,
                                           const uint32_t *tokens, int n,
                                           pn_state *state,
                                           float *per_pos_logits);

/**
 * Can a dense-FFN qwen35 model (Ornith, Bonsai) on the GPU take the batched
 * prefill? 1 when every weight's quant has a batched matvec (K-quants, Q8_0,
 * F32); 0 otherwise (e.g. Bonsai's native ternary Q2_0) or for any other model.
 * The serve keeps the per-token loop for a 0.
 *
 * @param model  Loaded model (NULL -> 0)
 * @return       1 = take pn_qwen35moe_forward_batch_cuda, 0 = per-token prefill
 *
 * Thread-safe: yes (reads the loaded weights only)
 * Reentrant: yes
 */
int pn_qwen35moe_dense_batch_prefill_ok(const pn_model *model);

/**
 * Phase 9b — prefill chunk pipelining (QWEN35_PREFILL_PIPELINE=1, default OFF).
 *
 * Under a 2-die layer split the dies run strictly sequentially per chunk: the
 * boundary hand-off syncs die A, the LM head syncs die B, so each idles for
 * the other's half. Chunk N+1's die-A work has no dependency on chunk N's
 * die-B work — the SSM recurrence and the KV rings are per-layer state and a
 * layer lives on exactly one die — so the two CAN overlap. The only thing
 * preventing it is the host blocking on die B for logits nobody reads.
 *
 * pn_qwen35moe_prefill_chunk_cuda() runs an intermediate chunk with the head
 * phase and the die-B sync omitted, advancing state->pos by n and leaving die
 * B's segment IN FLIGHT. Returns 0 on success, -1 if the batched path could
 * not run (nothing advanced; fall back to pn_qwen35moe_forward_batch_cuda,
 * which is always correct). The LAST chunk of a range must use the ordinary
 * call, so the caller gets real logits and both dies end up drained.
 *
 * pn_qwen35moe_prefill_drain_cuda() syncs both dies. Call it before any early
 * exit that reads or resets state an in-flight segment is still writing.
 *
 * Bit-exactness: identical kernels in an identical per-die order on identical
 * inputs — only the issue timing and the discarded intermediate LM heads
 * change. Nothing reassociates.
 */
int pn_qwen35moe_prefill_pipeline_enabled(void);
int pn_qwen35moe_prefill_chunk_cuda(pn_model *model, const uint32_t *tokens,
                                    int n, pn_state *state);
void pn_qwen35moe_prefill_drain_cuda(pn_model *model);

/**
 * Phase 9c — WIDE prefill pipelining
 * (QWEN35_PREFILL_PIPELINE_WIDE=1, default OFF).
 *
 * 9b above overlaps exactly two dies: chunk N+1's first against chunk N's last.
 * That is the whole machine on a 2-die split, but on an 8-die layer split it
 * leaves six dies idle at any instant, because the boundary hand-off syncs the
 * sending die at every hop inside a chunk.
 *
 * This entry runs a whole prefill range as an n_die-stage software pipeline: at
 * step T die d computes chunk T-d, so every die is working on a different chunk
 * at the same time and the steady-state period is ONE die-segment instead of
 * n_die of them. It owns the chunk loop, which 9b could not — a pipeline has to
 * see all the chunks to keep several in flight.
 *
 * Legal for the same reason 9b is: die d's chunk c needs die d-1's chunk c
 * (produced a step earlier) and die d's own carried KV/SSM state, which is
 * per-layer and therefore per-die, and which die d advances in chunk order.
 * There is no dependency from die d's chunk c to die d-1's chunk c+1.
 *
 * `should_abort` is polled once per step. On abort NO new chunk is injected but
 * the pipeline still drains what is in flight, so the session's layers all end
 * at the same position rather than half-walked; *`state->pos`* then advances by
 * the tokens that fully cleared, and the call reports 1.
 *
 * Bit-exact against the sequential path: identical kernels in an identical
 * per-die order on identical inputs, only the issue timing differs. Gate it
 * with PN_PREFILL_LOGITS_HASH=1.
 *
 * @param chunk  tokens per pipeline stage; clamped down to the q4 ring cap.
 * @param logits_out  receives an OWNED [n_vocab] tensor of the last position's
 *                    logits (caller frees), NULL on abort/refusal.
 * @return 0 = whole range prefilled; 1 = aborted after draining; -1 = refused
 *         or failed. On -1 nothing was advanced and the caller must fall back
 *         to the ordinary chunked path.
 *
 * Thread-safe: no (serialized by the caller's generation lock)
 * Reentrant: no
 * Signal-safe: no
 */
/**
 * @brief QWEN35_BS_PD_OVERLAP (default on; =0 off): the MoE TP decode-slots step runs
 *        on its own buffer lane so it can interleave with a prefill chunk's
 *        layers. CUDA builds only.
 * @return 1 when the switch is on, else 0. Read once.
 */
int pn_qwen35moe_pd_overlap_enabled(void);

/**
 * @brief Prefill cursor for the stage-alternating overlap (QWEN35_BS_PD_OVERLAP,
 *        MoE TP2 on 4 dies only). Walks tokens[0, n_total) at state->pos through
 *        the 2-stage wide pipeline a few layers at a time, so a decode step can
 *        run between its pieces (pn_qwen35moe_pd_cycle_cuda). Bit-exact with
 *        pn_qwen35moe_prefill_range_pipelined_cuda.
 *
 * Lifetime: begin → any mix of cycle / advance / quiesce → finish or abort
 * (each frees the cursor). `tokens` must stay valid until then. Every call is
 * made by the GPU turn owner.
 *
 * begin: NULL = refused (switch off, not MoE TP2, profiler or eager host KV
 *        mirror on); nothing was touched, the caller prefills another way.
 * cycle: one decode step over n_users slots exactly as
 *        pn_qwen35moe_decode_slots_rows (same arguments and results), with
 *        prefill pieces issued in each board's idle half, at most `pieces`
 *        layers per board. Returns 0 / -1.
 * advance: no decoders: run the pipeline alone until max_chunks more chunks
 *        retired (or all). Returns 0 / -1.
 * issued_all: every chunk's every layer was issued.
 * injected_all: the last chunk entered the pipeline. From then on no piece is
 *        left that would re-seed the live buffers, so a quiesce must not hand
 *        the turn over: finish has to follow under the same turn.
 * remaining: tokens of the range not yet retired.
 * quiesce: retire what is in flight, drain, and set state->pos to the retired
 *        position, so the GPU turn can be handed over. Returns 0 / -1.
 * finish: retire the rest, drain, run the head on the last row, set state->pos
 *        to the end of the range. *logits_out = an OWNED [n_vocab] tensor.
 *        Returns 0 / -1 (state undefined: the caller resets).
 * abort: retire only what entered the pipeline, drain, set state->pos there.
 *        Returns 0 / -1.
 */
typedef struct pn_qwen35moe_pd_cursor pn_qwen35moe_pd_cursor;
pn_qwen35moe_pd_cursor *pn_qwen35moe_pd_begin_cuda(pn_model *model, pn_state *state,
                                                   const uint32_t *tokens, int n_total,
                                                   int chunk);
int pn_qwen35moe_pd_cycle_cuda(pn_model *model, pn_qwen35moe_pd_cursor *cur,
                               const uint32_t *tokens, pn_state **states, const int *rows,
                               int n_users, float **logits_out, int pieces);
int pn_qwen35moe_pd_advance_cuda(pn_qwen35moe_pd_cursor *cur, int max_chunks);
int pn_qwen35moe_pd_issued_all(const pn_qwen35moe_pd_cursor *cur);
int pn_qwen35moe_pd_injected_all(const pn_qwen35moe_pd_cursor *cur);
int pn_qwen35moe_pd_remaining(const pn_qwen35moe_pd_cursor *cur);
/* The last overlap cycle's host wall time per phase (QWEN35_BS_STATS): stage-0
 * decode issue, die-0 drain at the hop, hand-off + hop copies, stage-1 decode
 * issue, stage-0 pieces, head (waits for die 3), stage-1 pieces. */
#define PN_QWEN35MOE_PD_PHASES 7
void pn_qwen35moe_pd_phase_ms(const pn_qwen35moe_pd_cursor *cur,
                              double ms[PN_QWEN35MOE_PD_PHASES]);
int pn_qwen35moe_pd_quiesce_cuda(pn_qwen35moe_pd_cursor *cur);
int pn_qwen35moe_pd_finish_cuda(pn_qwen35moe_pd_cursor *cur, tm_tensor **logits_out);
int pn_qwen35moe_pd_abort_cuda(pn_qwen35moe_pd_cursor *cur);
int pn_qwen35moe_prefill_pipeline_wide_enabled(void);
int pn_qwen35moe_prefill_range_pipelined_cuda(pn_model *model,
                                              const uint32_t *tokens,
                                              int n_total, pn_state *state,
                                              int chunk,
                                              int (*should_abort)(void *),
                                              void *ud,
                                              tm_tensor **logits_out);

/**
 * Speculative greedy decoding, increment 1 (temp 0, bit-exact by
 * construction). verify runs [pending, draft_1..draft_{n-1}] at positions
 * state->pos.. through the batched forward with per-position SSM state
 * snapshots and fills argmax_out[t] with each position's exact greedy
 * argmax — WITHOUT advancing state->pos or committing state. The caller
 * accepts the longest prefix where draft_t == argmax_out[t-1] (plus the
 * free correction/bonus token) and calls accept with n_accept in [1, n]:
 * every SSM layer's dual state is rolled to its position-(n_accept-1)
 * snapshot (the KV cache needs no rollback) and state->pos += n_accept.
 * Accepted tokens are exactly the model's greedy argmaxes, so the emitted
 * stream is IDENTICAL to plain greedy decode. verify returns -1 (no state
 * touched; fall back to plain decode) if the batched path cannot run.
 */


int pn_qwen35moe_spec_verify(pn_model *model, const uint32_t *tokens, int n,
                             pn_state *state, int32_t *argmax_out);
int pn_qwen35moe_spec_accept(pn_model *model, pn_state *state, int n_accept);
/** Test hook: row `row` of the last verify's full logits (n_vocab floats). */
int pn_qwen35moe_spec_row_logits(pn_model *model, int row, float *out);
/** pn_qwen35moe_spec_verify, but every row's top-k (id, value) pairs land in ids/vals
 *  ([n][k], descending, ties by ascending id) — the same pairs the plain decode step
 *  presents for that row — so the caller can SAMPLE each row exactly as plain decode
 *  would. n <= 5 (the rows kernel). Commit with pn_qwen35moe_spec_accept. */
int pn_qwen35moe_spec_verify_topk(pn_model *model, const uint32_t *tokens, int n,
                                  pn_state *state, int k, int *ids, float *vals);

/* ---- MTP draft head on the CUDA path (QWEN35_MTP=1; Phase 2F) ----
 * The head keeps its own KV over every committed position. After each trunk batch that
 * advanced `state` (a prefill chunk of <= 128 tokens, or a verify's accept), call
 * pn_qwen35moe_mtp_push_rows with that batch's input tokens and the count it advanced;
 * pn_qwen35moe_mtp_draft then drafts the tokens after the uncommitted `seed` (at
 * state->pos). Drafts never change the emitted stream; the verify decides it. */

/** Test hook: pn_qwen35moe_gpu_forget_state's binding drop on a fake TP layout (both rank
 *  halves bound). @return the bindings left: 0 = correct, -1 = no memory. */
int pn_qwen35moe_tp_forget_selftest(void);

/** Test hook (CUDA, one die): a fake TP attention layer restored from host (new content_gen,
 *  fresh mirror) then resumed at a pos0 > 0, run in the TP walk's order (both-rank pre-sync,
 *  then each rank's sync + stale mark). @return refused syncs + ring rows that differ from
 *  the host mirror: 0 = correct, -1 = no memory, -2 = no CUDA device (skip). */
int pn_qwen35moe_tp_restore_sync_selftest(void);

/** Whether the model carries an uploaded MTP head. */
int pn_qwen35moe_mtp_ready(pn_model *model);
/** Join the last trunk batch's first n rows (positions state->pos - n ..) to the head's
 *  backlog; tokens[0..n) are those rows' input tokens. 0, or -1 (head off / no rows). */
int pn_qwen35moe_mtp_push_rows(pn_model *model, pn_state *state, const uint32_t *tokens,
                               int n);
/** The trunk state was set back (a checkpoint restore): drop the head's backlog and
 *  continue it from state->pos. 0, or -1 (head off). */
int pn_qwen35moe_mtp_rewind(pn_model *model, pn_state *state);
/** Draft continuation: called after draft `drafted` (1-based count so far) with its
 *  confidence; return non-zero to draft one more. */
typedef int (*pn_mtp_more_fn)(void *ctx, int drafted, float conf);
/** Up to kmax drafts after `seed`; confs[i] = draft i's softmax over the head's top 8.
 *  @return the count (>= 1), or -1 (no head / backlog not at state->pos) */
int pn_qwen35moe_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int kmax,
                           uint32_t *drafts, float *confs, pn_mtp_more_fn more,
                           void *more_ctx);
/** Pairs a pick callback sees per draft. */
#define PN_MTP_PICK_PAIRS 20
/** Draft choice: draft `draft_index` (0-based) from the head's top `n` pairs (full-vocabulary
 *  ids, values descending, ties by ascending id). Return the token the head chains on. */
typedef uint32_t (*pn_mtp_pick_fn)(void *ctx, int draft_index, const int *ids,
                                   const float *vals, int n);
/** pn_qwen35moe_mtp_draft, each draft chosen by `pick` (NULL = the argmax) from the head's
 *  top PN_MTP_PICK_PAIRS pairs. confs[] stay the argmax's confidence. */
int pn_qwen35moe_mtp_draft_pick(pn_model *model, pn_state *state, uint32_t seed, int kmax,
                                uint32_t *drafts, float *confs, pn_mtp_more_fn more,
                                void *more_ctx, pn_mtp_pick_fn pick, void *pick_ctx);
typedef struct {
    long drafts;      /* drafts produced */
    long kv_rows;     /* head rows appended (catch-up + chained) */
    long full_rows;   /* rows run through the whole head block */
    double draft_ms;  /* wall time inside pn_qwen35moe_mtp_draft */
    double prof_ms[4];/* QWEN35_MTP_PROF=1: kv rows / attention / MoE / head pick (sync'd) */
} pn_mtp_stats;
int pn_qwen35moe_mtp_stats(pn_model *model, pn_mtp_stats *out);
/** Verify CUDA graphs (QWEN35_VERIFY_GRAPH): captures, replays and failed captures so far,
 *  summed over dies and row counts. */
int pn_qwen35moe_verify_graph_stats(pn_model *model, long *captures, long *replays,
                                    long *failures);

/**
 * MULTI-USER DECODE, increment 2 of the batched-forward work: K users each
 * decoding one token/step, processed in ONE forward pass — the weights
 * stream once for all K (the serving-throughput lever). Each user keeps its
 * OWN state (pn_state per user): per-slot SSM images on the GPU, per-user
 * host KV cache, per-user RoPE position. BIT-EXACT per user: a user's token
 * stream when batched is identical to that user decoding alone (run the
 * single-user reference under QWEN35_FUSE=0x37 — the fused qknorm+rope
 * kernel's known ~1-ulp contraction, same caveat as the prefill gate).
 *
 * Protocol (fixed K):
 *   1. prefill each user's pn_state (single-token or batched prefill);
 *   2. IMMEDIATELY after each user's prefill, bind it:
 *      pn_qwen35moe_decode_slot_bind(model, state_u, u, K) — this
 *      snapshots the live GPU SSM state into slot u (the host mirror is
 *      stale after a GPU prefill, so bind must precede any other user's
 *      forward);
 *   3. step all users: pn_qwen35moe_decode_slots(model, tokens, states, K,
 *      logits) runs ONE pass, fills logits[u] (each user's own logits
 *      buffer, device top-k presentation like single-user decode) and
 *      advances each states[u]->pos by 1.
 * Re-binding (or changing K) re-seeds slots; continuous batching
 * (join/leave mid-flight, prefill+decode mixing) is the follow-up.
 * Returns 0 on success, -1 on failure (no state corrupted on the
 * validation failures; a mid-pass kernel failure is fatal to the batch).
 */
int pn_qwen35moe_decode_slot_bind(pn_model *model, pn_state *state, int slot,
                                  int n_slots);
int pn_qwen35moe_decode_slots(pn_model *model, const uint32_t *tokens,
                              pn_state **states, int n_users,
                              float **logits_out);

/**
 * BENCH ONLY: advance bound, already-stepped decode row `row` of `state` by `add`
 * positions without computing them (the skipped KV rows are garbage; timing only).
 * Lets a decode-attention benchmark reach 64k-128k context without prefilling it.
 * Returns 0, or -1 if the row is not current for `state` or the ring would wrap.
 * Thread-safe: no (call between decode steps). Reentrant: no. Signal-safe: no.
 */
int pn_qwen35moe_bench_skip_positions(pn_model *model, pn_state *state, int row,
                                      int add);

/**
 * pn_qwen35moe_decode_slots with an explicit ROW MAP: user u decodes in the
 * persistent slot row rows[u] it was bound to (rows NULL = u). The rows need
 * not be contiguous or ordered, so a scheduler can give every session a fixed
 * block for its lifetime and step any subset of them with nothing moved.
 * Bit-exact per user vs the identity map. Returns 0 / -1 like decode_slots.
 */
int pn_qwen35moe_decode_slots_rows(pn_model *model, const uint32_t *tokens,
                                   pn_state **states, const int *rows,
                                   int n_users, float **logits_out);

/**
 * Carve `n_rows` per-session decode blocks on the model's dies: for each row a
 * ctx_cap KV ring row per attention layer (+ q4 sink/window shadows) and a
 * conv + recurrent SSM image per SSM layer. Call once at load, before any
 * bind; `geom` (any state of this model) supplies the KV geometry. Returns 0,
 * or -1 (logged) when the blocks do not fit — refuse the load then.
 */
int pn_qwen35moe_decode_slots_carve(pn_model *model, pn_state *geom, int n_rows);

/**
 * 1 when decode row `row` still holds `state` exactly as committed (its SSM
 * image at state->pos and all KV rows [0, pos) on every attention layer), so a
 * resumed turn can continue from the card with no host copy; 0 otherwise.
 * Bookkeeping only; call owning the GPU turn.
 */
int pn_qwen35moe_slot_holds_state(pn_model *model, int row, pn_state *state);

/**
 * Carve one turn-start checkpoint per decode row (the serve's rung 1c): per
 * SSM layer a conv + recurrent image, per q4 attention layer the fp16
 * sink/window shadows. Call once at load, after
 * pn_qwen35moe_decode_slots_carve. Returns 0, or -1 (logged, nothing kept)
 * when they do not fit; the serve then runs without checkpoints.
 */
int pn_qwen35moe_slot_checkpoints_carve(pn_model *model, pn_state *geom,
                                        int n_rows);

/**
 * Save `state`'s live image as row `row`'s checkpoint at state->pos. Call
 * owning the GPU turn right after a prefill of this state (its live shared
 * buffers hold it at state->pos on every layer). Returns 0, or -1 when the
 * live buffers do not hold it (nothing changed) or no checkpoints are carved.
 */
int pn_qwen35moe_slot_checkpoint_save(pn_model *model, pn_state *state, int row);

/**
 * Put decode row `row` back at its checkpoint: the checkpoint must be
 * `state`'s at position `pos`, taken since the state's last reset or restore,
 * and the row must still hold the state's KV rows [0, pos). Copies the saved
 * images into the row, sets the row's bookkeeping and state->pos to `pos`.
 * The caller then forgets the state's shared bindings so the next forward
 * re-seeds from the row. Returns 0, or -1 with nothing changed.
 */
int pn_qwen35moe_slot_checkpoint_restore(pn_model *model, pn_state *state,
                                         int row, int pos);

/**
 * The two calls above with the checkpoint index `ck` decoupled from the decode
 * row: save into checkpoint `ck` (any index below the carved row count), and
 * restore checkpoint `ck` into decode row `row`. Same contracts. The serve
 * carves 2 x slots and gives slot i the turn checkpoint i and the divergence
 * checkpoint slots + i.
 */
int pn_qwen35moe_slot_checkpoint_save_at(pn_model *model, pn_state *state,
                                         int ck);
int pn_qwen35moe_slot_checkpoint_restore_from(pn_model *model, pn_state *state,
                                              int row, int ck, int pos);

/**
 * The memory-pressure decode-concurrency ceiling (rung 2 of the reactive
 * ladder): the max n_slots decode_slot_bind will accept right now.  Full
 * PN_QWEN35MOE_GPU_MAX_SLOTS at level OK / when memwatch is off.  The batch
 * scheduler sizes its slot count with this up front so the per-bind guard
 * never refuses a lazily-initialized lane.
 */
int pn_qwen35moe_gpu_ladder_max_slots(void);

/**
 * PREFIX CACHING support: snapshot the live GPU-resident SSM dual state of
 * `state` back into its host pn_state, so the host state is fully
 * self-contained (host KV mirror + host SSM + pos). The prefix cache calls
 * this on a session BEFORE another session's forward overwrites the shared
 * per-layer GPU SSM buffers; the session is then restored later purely by the
 * ordinary seed-on-host-change path (H2D SSM re-seed + KV re-sync from the host
 * mirror) that the batched/single-token forward already runs when it sees a
 * different host state. No-op (returns 0) without a GPU handle — the CPU path
 * keeps the host state authoritative already. Only layers whose GPU buffer
 * currently holds THIS state's image are copied. Returns 0 on success, -1 on
 * failure.
 */
int pn_qwen35moe_gpu_state_to_host(pn_model *model, pn_state *state);



/**
 * Phase 10b (INV3): SSM-only variant of gpu_state_to_host — captures the
 * SHARED single-user SSM buffers into the host mirror but leaves the KV
 * mirror alone. The batch scheduler uses it at slot bind / turn finalize in
 * lazy-mirror mode, where the KV rows live in the GPU slot ring and are
 * seeded device-to-device on rebind (materializing them to host RAM there
 * would recreate exactly the per-session host footprint the lazy mode
 * removes). No-op (returns 0) without a GPU handle.
 */
int pn_qwen35moe_gpu_state_ssm_to_host(pn_model *model, pn_state *state);

/**
 * Phase 10b (INV3): QWEN35_HOST_KV_MIRROR gate. 1 = legacy EAGER host KV
 * mirror (every decode token D2H-appended to host RAM); 0 (default) = LAZY —
 * the GPU slot ring is authoritative while a session is bound and the host
 * mirror is materialized on demand (export, sys-prefix capture, prefill
 * yield, host-softmax fallback). Cached on first read.
 */
int pn_qwen35moe_host_kv_mirror_eager(void);

/**
 * PREFIX CACHING (batched path): snapshot a DECODE SLOT's live SSM state back
 * into that slot's pn_state host mirror.  gpu_state_to_host captures only the
 * SHARED single-user SSM buffers; batched decode advances the per-slot images
 * in the slot-major conv_slots/ssm_slots arrays at row `gpu_idx`, so after a
 * batched turn a slot's SSM host mirror is stale.  This D2H-copies
 * conv_slots[gpu_idx] → st->conv_state and ssm_slots[gpu_idx] → st->ssm_state
 * for every SSM layer (the inverse of decode_slot_bind's host→slot seed) and
 * clears any shared-buffer host_state binding for this state, leaving the
 * slot's pn_state warm + self-contained (host KV mirror + host SSM + pos) and
 * delta-reusable next turn.  `gpu_idx` is the slot's CURRENT GPU index
 * (bs_slot.gpu_idx).  Must be called owning the GPU turn (no decode step may
 * advance the slots mid-copy).  No-op (returns 0) without a GPU handle.
 * Returns 0 on success, -1 on failure.
 */
int pn_qwen35moe_decode_slot_ssm_to_host(pn_model *model, int gpu_idx,
                                         pn_state *state);

/**
 * Forget every GPU-resident "this state is live here" binding for `state`
 * (per-layer SSM d->host_state + single-user KV kv_host_id/kv_synced_pos), so
 * the next forward re-seeds all GPU shadows from the host mirror as if the
 * state were fresh.  A serve slot's pn_state is reused across requests and
 * pn_state_reset clears only the host mirror; without this a long re-prefill
 * on a reused slot reads stale GPU residue and diverges from a fresh state.
 * Host bookkeeping only; call under the GPU-owner lock right after
 * pn_state_reset.  No-op on the CPU path (host state already authoritative).
 */
void pn_qwen35moe_gpu_forget_state(pn_model *model, pn_state *state);

/**
 * The SSM half of pn_qwen35moe_gpu_forget_state: drop only the shared-buffer SSM binding
 * for `state`, so the next single-user forward re-seeds its recurrent image (from the
 * state's decode row when that row is current, D2D) while the shared KV ring keeps its
 * watermark and tops up only the rows it lacks. For a state that decoded in its slot since
 * its last single-user forward (solo-slot MTP). Host bookkeeping only. CUDA builds.
 */
void pn_qwen35moe_gpu_forget_ssm(pn_model *model, pn_state *state);

/**
 * ASYNC single-token decode step (C2, QWEN35_ASYNC_SCHED): begin launches
 * one token's GPU step (segment-graph replay + LM head) and returns WITHOUT
 * blocking; poll reaps completion via a CUDA event and advances the 2-die
 * boundary hand-off, so one 2-core host keeps up to 8 cards' generations in
 * flight simultaneously instead of serializing on fc_sync barriers.
 *
 * begin: 0 = step in flight (caller MUST poll to completion before any
 * other forward on this model), 1 = token not async-eligible (nothing
 * launched — run the ordinary blocking forward, bit-identical), -1 = error.
 * poll: 1 = still running (sleep briefly, poll again — never blocks),
 * 0 = done (*logits_out = owned logits tensor, state->pos advanced),
 * -1 = error (machine reset; abort the generation).
 *
 * Serialized per model by the serve layer's gen_lock; different models
 * (own die contexts) interleave freely from any thread.
 */
int pn_qwen35moe_forward_step_begin(pn_model *model, uint32_t token,
                                    pn_state *state);
int pn_qwen35moe_forward_step_poll(pn_model *model, pn_state *state,
                                   tm_tensor **logits_out);

/* Upload the model's weights to persistent GPU buffers (no-op without a GPU).
 * Defined in qwen35moe_forward_cuda.c; declared here so the engine loader can
 * call it after a successful pn_qwen35moe_load(). */
int  pn_qwen35moe_gpu_upload(pn_qwen35moe_model *model);
void pn_qwen35moe_gpu_free(pn_qwen35moe_model *model);

/* True once GPU weights are resident on the model (upload succeeded). */
int  pn_qwen35moe_model_gpu_loaded(const pn_qwen35moe_model *model);

/**
 * Snapshot this model's CUDA-graph residency — how many of its per-layer /
 * per-segment graph slots hold a live captured graph right now, how many
 * FAILED to capture, and how many are gated out as INELIGIBLE. Only the third
 * of those is normal-and-fine; only the second is a problem.
 *
 * A capture failure degrades to the direct kernel body with correct output, so
 * without this counter a serve can silently shed graph coverage layer by layer
 * (~1.8 ms/step for all 40 layers) and look completely healthy.
 *
 * @param model  Loaded qwen35moe model
 * @param out    Filled with the per-class + summed residency
 * @return       0 on success, -1 if the model has no GPU handle (CPU forward)
 *
 * Thread-safe: yes, lock-free — it reads per-die counters the driver threads
 *   write. A concurrent decode step can make the snapshot off by one slot; the
 *   numbers are a gauge, not an invariant.
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_qwen35moe_gpu_graph_residency(const pn_qwen35moe_model *model,
                                     pn_model_graph_residency *out);

/**
 * Snapshot which attention path every attention layer is on RIGHT NOW —
 * on-die flash, or the host-softmax fallback — and why.
 *
 * The fallback is correctness-faithful and silent: the same tokens come back,
 * an order of magnitude slower — 13.02 -> 1.22 tok/s measured on the depth
 * bench at K=2, and worse the deeper the context. This is the only thing that can
 * tell an operator it is happening.
 *
 * @param model  Loaded qwen35moe model
 * @param out    Filled with the per-class + summed residency
 * @return       0 on success, -1 if the model has no GPU handle (CPU forward)
 *
 * Thread-safe: yes, lock-free — same contract as the graph residency above.
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_qwen35moe_gpu_attn_residency(const pn_qwen35moe_model *model,
                                    pn_model_attn_residency *out);
#endif /* USE_CUDA */

#ifdef USE_METAL
/**
 * Make every quantized weight of the model addressable by the Apple GPU
 * (implemented in qwen35moe_forward_metal.c). Unified memory means there is no
 * device copy to make: a page-aligned mmap'd weight is wrapped in place with
 * fm_buffer_from_ptr and the handle stored on pn_qweight.gpu. Idempotent per
 * weight — a weight that already has a buffer is left alone — so calling this
 * twice is a no-op, and the CPU weights stay live either way.
 *
 * Deliberately does NOT create an MTLResidencySet: wiring the working set was
 * measured worth ~0.7% and costs the file-backed/evictable memory posture that
 * is the point of this backend (foundry/metal/qwen35moe_metal_spec.md §2).
 *
 * A weight whose pointer is not host-page aligned cannot take the zero-copy
 * path. Those are COUNTED AND LOGGED (census on stderr) and left un-uploaded
 * rather than copied, because on a real GGUF that is nearly every tensor and
 * the copy would duplicate the whole model in host RAM.
 *
 * @param model  Loaded qwen35moe/bonsai model (CPU weights resident).
 * @return       0 when the walk ran (even if every weight missed zero-copy),
 *               -1 when there is no Metal device or the handle is unusable.
 *
 * Thread-safe: no (mutates every weight's gpu field)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_metal_upload(pn_qwen35moe_model *model);

/**
 * Release every Metal buffer attached by pn_qwen35moe_metal_upload. NULL-safe
 * and safe on a model that was never uploaded. Frees only the wrappers — the
 * mmap'd weight bytes belong to the GGUF file.
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void pn_qwen35moe_metal_free(pn_qwen35moe_model *model);

/**
 * Metal forward conforming to pn_forward_fn (implemented in
 * qwen35moe_forward_metal.c). Selected by the engine when fm_available() and
 * pn_qwen35moe_metal_upload() succeeded.
 *
 * PHASE 0: no kernels exist yet. The token forward reports NOT-IMPLEMENTED and
 * this adapter delegates the token to pn_qwen35moe_forward (CPU), so the
 * emitted logits are bit-identical to a build without USE_METAL. It is a seam,
 * not a GPU forward — do not read a Metal selection in the log as GPU compute.
 *
 * Thread-safe: no (mutates state)
 * Reentrant: no
 * Signal-safe: no
 */
tm_tensor *pn_qwen35moe_forward_metal(pn_model *model, uint32_t token,
                                      pn_state *state);
#endif /* USE_METAL */

/**
 * Require the FULL logits row for this state (constrained decoding).
 *
 * The die normally reduces the LM head to the top-k (id, logit) pairs so only k
 * pairs cross the 0.4 GB/s host link; the host row is -inf everywhere else.  A
 * grammar mask must be able to force a token the model did not rank in its
 * top-k, so a turn decoding under a grammar sets this for its lifetime.
 *
 * Costs the full n_vocab D2H per token, so set it ONLY while a grammar is live.
 * No-op for a NULL state.
 *
 * Thread-safe: no (per-state flag; set before the turn's first forward)
 */
void pn_qwen35moe_state_set_full_logits(pn_state *state, int on);

/** True when `model`'s pn_state is a pn_qwen35moe_state (MoE + dense Bonsai). */
int pn_qwen35moe_owns_state(const pn_model *model);

/** True when `model` is a qwen35 model with a dense SwiGLU FFN (Qwen3.8, Ornith, Bonsai)
 *  rather than MoE experts. */
int pn_qwen35moe_dense_ffn(const pn_model *model);


/* Explicit die placement (pn_load_on, Phase 7). Stored on the model BEFORE the
 * GPU upload so its die selection prefers the placement over the
 * QWEN35_GPU_DIE_A/B env knobs. Unconditional (bookkeeping without CUDA);
 * declared here so the engine loader can call it. n == 0 clears. */
void pn_qwen35moe_model_set_dies(pn_qwen35moe_model *m, const int *dies, int n);
int  pn_qwen35moe_model_dies(const pn_qwen35moe_model *m, const int **dies_out);

/* KV cache format of this load (pn_kv_format, engine.h), set by the engine loader
 * before any state exists. kv_q4: the format every new KV cache starts in — the
 * explicit choice, else QWEN35_KV_Q4 (default ON). Thread-safe: set before publish. */
void pn_qwen35moe_model_set_kv_format(pn_qwen35moe_model *m, int kv_format);
int  pn_qwen35moe_model_kv_format(const pn_qwen35moe_model *m);
int  pn_qwen35moe_model_kv_q4(const pn_qwen35moe_model *m);

/* Decode blocks the serve will carve on the card for this load (pn_load_opts.
 * card_sessions), set by the engine loader before the upload so the die gate
 * prices them. 0 = none. Thread-safe: set before publish. */
void pn_qwen35moe_model_set_card_sessions(pn_qwen35moe_model *m, int n);
int  pn_qwen35moe_model_card_sessions(const pn_qwen35moe_model *m);

/* Dual-state lifecycle: per-SSM-layer conv/ssm state + per-attention-layer KV
 * cache + per-token scratch arena (implemented in qwen35moe_forward.c). */
void *pn_qwen35moe_state_new(void *weights);
void  pn_qwen35moe_state_free(void *state);
void  pn_qwen35moe_state_reset(void *state);

/* ============================================================
 * Host-side state snapshot at a token position (system-prefix caching) —
 * implemented in qwen35moe_state_snapshot.c. Pure host memory: on the GPU path
 * the caller must make the host mirror authoritative FIRST (snapshot after
 * pn_qwen35moe_gpu_state_to_host) and, after a restore, forget the state's GPU
 * bindings (pn_qwen35moe_gpu_forget_state) so the next forward re-seeds from
 * the restored mirror — a restore to a NON-ZERO pos does not trigger the
 * pos==0 restart re-sync on its own.
 * ============================================================ */

/**
 * Deep-copy the dual state `state` (a pn_qwen35moe_state*) as it stands at
 * position `pos`: every SSM layer's conv/recurrent state, every attention
 * layer's KV rows [0, pos), and the full fp16 sink/window shadows (q4 mode —
 * the window ring is later overwritten, so it must be captured).
 *
 * @param state  Dual state (host mirror must be current).
 * @param pos    Position the state sits at (> 0, <= every layer's max_seq).
 * @return       Opaque snapshot (free with pn_qwen35moe_state_snapshot_free),
 *               or NULL on failure.
 *
 * Thread-safe: no
 */
void *pn_qwen35moe_state_snapshot(const void *state, int pos);

/**
 * A RECURRENT snapshot at `pos`: every SSM layer's conv/recurrent state and the
 * q4 fp16 sink/window shadows, but NO KV rows. A restore puts the state back at
 * `pos` over the KV rows [0, pos) the host mirror still holds, so it is valid
 * only while no row below `pos` was rewritten (the single-user turn snapshot:
 * a drop_last request resumes at the last turn's start). Not serializable.
 * Same host-mirror protocol as pn_qwen35moe_state_snapshot.
 *
 * Thread-safe: no
 */
void *pn_qwen35moe_state_snapshot_recurrent(const void *state, int pos);

/**
 * Restore a snapshot into `state`'s host buffers (dims must match exactly —
 * same model, KV mode and window) and mark the host mirror authoritative
 * (kv_mirror_stale cleared). The caller sets pn_state->pos to the snapshot
 * position and forgets the GPU bindings (see above).
 *
 * @return  0 on success, -1 on dimension mismatch / bad args.
 *
 * Thread-safe: no
 */
int pn_qwen35moe_state_restore(void *state, const void *snap);

/** The position a snapshot was taken at, or -1. */
int pn_qwen35moe_state_snapshot_pos(const void *snap);

/** Free a snapshot. NULL-safe. */
void pn_qwen35moe_state_snapshot_free(void *snap);

/**
 * Serialize a snapshot (the scattered per-layer buffers produced by
 * pn_qwen35moe_state_snapshot) into ONE contiguous little-endian blob for the
 * session warm-tier (K80 serving-ops Phase 7): a fixed header
 * {magic, format_version, weights_version, pos, n_ssm, n_attn} followed by
 * every SSM layer's dims+conv+ssm bytes and every attention layer's
 * dims+KV-rows (+q4 sink/window shadows) in layer order. The blob is
 * self-describing (all dims are in it) so pn_qwen35moe_snap_deserialize is the
 * exact inverse — the round-trip snap→bytes→snap is byte-identical.
 *
 * @param snap             Snapshot to serialize (from pn_qwen35moe_state_snapshot).
 * @param weights_version  Model-identity tag stamped in the header (see
 *                         mr_weights_version); the restore side rejects a blob
 *                         whose tag does not match the target model.
 * @param bytes_out        Out: malloc'd blob (caller frees). NULL on failure.
 * @param len_out          Out: blob length in bytes.
 * @return                 0 on success, -1 on bad args / allocation failure.
 *
 * Thread-safe: no
 */
int pn_qwen35moe_snap_serialize(const void *snap, uint64_t weights_version,
                                unsigned char **bytes_out, size_t *len_out);

/**
 * Deserialize a blob produced by pn_qwen35moe_snap_serialize back into an
 * opaque snapshot (free with pn_qwen35moe_state_snapshot_free). Validates the
 * magic, format version, and internal length consistency; a truncated,
 * corrupt, or wrong-format blob is rejected (NULL). Does NOT check
 * weights_version — the caller compares *weights_version_out against the target
 * model's tag and refuses a mismatch (so it can quote a clean error).
 *
 * @param bytes                Blob bytes.
 * @param len                  Blob length.
 * @param weights_version_out  Out (nullable): the header's weights_version tag.
 * @return                     Opaque snapshot, or NULL on any validation failure.
 *
 * Thread-safe: no
 */
void *pn_qwen35moe_snap_deserialize(const unsigned char *bytes, size_t len,
                                    uint64_t *weights_version_out);


#ifdef USE_TT
/**
 * STEP 4 S2 (Tenstorrent): one greedy speculative step — verify [seed, drafts...] at
 * state->pos.., accept the longest draft prefix equal to the model's greedy tokens, commit the
 * seed + accepted drafts (state->pos advances by that many) and roll the carried state back
 * to the last committed row. `emit` receives the accepted drafts then the model's next token
 * (uncommitted: the next call's seed). @return the number of tokens in `emit`, or -1
 * (unsupported / failed: the caller falls back to plain decode).
 */
int pn_qwen35moe_tt_spec_step(pn_model *model, pn_state *state, uint32_t seed,
                              const uint32_t *drafts, int n_drafts, uint32_t *emit);
/**
 * A verify row's chooser for SPECULATIVE SAMPLING: given row r's sparse logits (the same
 * top-N (index, value) pairs a plain decode step's sparse row carries), return the token the
 * engine's sampler picks — called once per EMITTED token, in order, so the sampler's RNG and
 * repeat window advance exactly as plain decode's do (shared randomness: a draft is accepted
 * iff plain sampling would have drawn it).
 */
typedef uint32_t (*pn_spec_choose_fn)(void *ctx, const uint32_t *index, const float *value,
                                      uint32_t count);
/** A chooser's answer for "this row's top-N cannot decide the draw exactly" (rows >= 1 only:
 *  the cycle then ends with the previous row's accepted draft as the next, uncommitted
 *  token, and the next cycle meets this distribution again as its row 0, whose dense row
 *  can be refilled). */
#define PN_SPEC_CHOOSE_DEFER 0xFFFFFFFFu
/** pn_qwen35moe_tt_spec_step with a chooser (NULL = greedy). */
int pn_qwen35moe_tt_spec_step_ex(pn_model *model, pn_state *state, uint32_t seed,
                                 const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                 pn_spec_choose_fn choose, void *choose_ctx);
/** True when a greedy pick over the exact top-N row equals the dense argmax for these params:
 *  temperature <= 0 and no logit surgery that can promote a token from outside the top-N
 *  (logit_bias, presence/frequency penalties, a grammar hook). */
bool pn_qwen35moe_tt_greedy_exact_on_topn(const pn_sampling_params *params);
/** Per request, before the prompt's forwards (the prefill's last row is token 1): the logit
 *  rows. PN_TT_SPARSE_LOGITS unset: top-8 for exact greedy, top-64 for sampled requests with
 *  1 <= top_k <= 64 (no bias / grammar / logprobs), dense otherwise; =1 forces top-N, =0
 *  dense. @return the N chosen, 0 = dense. */
uint32_t pn_qwen35moe_tt_select_rows(pn_model *model, const pn_sampling_params *params,
                                     bool needs_full_logits);
/** pn_sampling_params.dense_row for this forward: the last head's logit row 0, dense
 *  (`model_ctx` is the pn_model). */
int pn_qwen35moe_tt_dense_row(void *model_ctx, float *out, int n_vocab);
/** pn_sampling_params.dense_chunks for this forward: only the 16-entry chunks of the
 *  dense row 0 marked in `need` (`model_ctx` is the pn_model). */
int pn_qwen35moe_tt_dense_chunks(void *model_ctx, const unsigned char *need, float *out,
                                 int n_vocab);
/** pn_sampling_params.tail_counts for this forward: the card counts logits row `row` of the
 *  last head (0 = decode, k = verify row k) into the key bins (pn_tt_topk_tail.h).
 *  `PN_TT_TAIL_CROSSCHECK=1`: row 0's counts are checked against the dense row on the host. */
int pn_qwen35moe_tt_tail_counts(void *model_ctx, int row, const uint32_t *keys, int bins,
                                uint32_t *counts);
/** True when the verify can hand a chooser each row's sparse logits AND plain decode's rows
 *  are the same sparse presentation (`PN_TT_SPARSE_LOGITS`): speculative sampling is then
 *  identical to plain sampling. */
bool pn_qwen35moe_tt_spec_can_sample(pn_model *model);
/** PM 5: keep the model's resident card families armed while idle (serve tick). */
int pn_qwen35moe_keepalive_tt(pn_model *model);
/** PM 6: a short internal prefill + 2 tokens at serve start, so the families armed during the
 *  load (and idle past the retire limit by the time the serve listens) are re-armed before the
 *  first request instead of inside it. Leaves no session state behind. `PN_TT_WARM=0` off. */
int pn_qwen35moe_warm_tt(pn_model *model);
/** RED `spec_lookup_unverified`: arm the next verify cycle's unverified (wrong) accept. */
void pn_qwen35moe_tt_red_accept_next_cycle(void);

/** STEP 4 S2b: up to `k` drafts for the token after `seed` (at state->pos) from the MTP head
 *  (`PN_TT_MTP`), after its owed catch-up. @return the count, or -1 (no MTP: draft elsewhere) */
int pn_qwen35moe_tt_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int k,
                              uint32_t *drafts);
/** The MTP head's draft chooser for SAMPLED drafts: draft `draft_index`'s top-N (index, value)
 *  pairs (vocabulary ids) -> the draft (pn_spec_sample_draft). */
typedef uint32_t (*pn_spec_draft_fn)(void *ctx, int draft_index, const uint32_t *index,
                                     const float *value, uint32_t count);
/** The speculative counters so far (cycles, drafts verified, drafts accepted): a caller takes
 *  differences around a request. @return 0, or -1 (no TT model) */
int pn_qwen35moe_tt_spec_counts(pn_model *model, uint64_t *cycles, uint64_t *drafted,
                                uint64_t *accepted);
/** pn_qwen35moe_tt_mtp_draft with a draft chooser (NULL = the head's argmax). The head then
 *  returns `PN_TT_SPEC_Q_PAIRS` pairs a draft (default: the request's top-N). */
int pn_qwen35moe_tt_mtp_draft_ex(pn_model *model, pn_state *state, uint32_t seed, int k,
                                 uint32_t *drafts, pn_spec_draft_fn draw, void *draw_ctx);

/**
 * Put this model's MATMULS on a Tenstorrent Blackhole card (qwen35moe_forward_tt.c).
 *
 * Opens the card, turns the checkpoint into a cached FTTW artifact if one is not already
 * beside it, binds every matmul weight by name, and installs the `qwen35moe_mv` seam so the
 * forward's matvecs leave for the card. Nothing else moves: the Gated-Delta-Net recurrence,
 * the conv1d, the attention softmax and every norm stay on the host, running the CPU
 * oracle's own code on its own fp32 state — so the two arms differ in the matmuls alone.
 *
 * Refuses rather than half-working. The MoE branch, a geometry that does not divide by the
 * 32-wide tile, and a binding that did not resolve are three DIFFERENT refusals, because
 * each sends a different person to a different fix.
 *
 * @return 0 on success, -1 on refusal or failure (the model stays on the CPU).
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
int pn_qwen35moe_tt_upload(pn_model *model);

/* 1 if the model has a live Tenstorrent residency attached. */
int pn_qwen35moe_model_tt_loaded(const pn_model *model);

/**
 * 1 if this model's attention K/V lives in CARD DRAM rather than in the host state.
 *
 * **THE HOST MIRROR IS NOT WRITTEN ON THAT ARM.** `qwen35moe_resident_tt.c` fills
 * `st->kv[idx]` only under `PN_TT_ATTN_COMPARE`, because the whole point of on-card attention
 * is that 32,768 B a position stops crossing the link — a 256k session would be 16 GiB of
 * host mirror on a 7.1 GB box. Anything that reads the host K/V as if it were the session
 * must therefore ASK first: the system-prefix snapshot and the session export both serialise
 * `st->kv`, and on the card arm they would capture zeros, restore them, and produce a fluent
 * wrong answer with nothing refusing anywhere.
 *
 * @return 1 when the K/V is on the card, 0 otherwise (no residency, or host attention)
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: no
 */
int pn_qwen35moe_model_tt_kv_on_card(const pn_model *model);

/**
 * MULTI-TURN: resume `state` from the card's prefill checkpoint for a prompt `tokens[0, n)`.
 *
 * The checkpoint is the carried GDN state at the last full pass boundary of this sequence's
 * last batched prefill, in card DRAM (never on the host's disk). When it belongs to `state`,
 * the card still holds `state`'s sequence, it lies below `n` and its tokens are a prefix of
 * `tokens`, the carried state is copied back and `state->pos` set to it; a batched prefill
 * from there runs the passes a cold prefill of `tokens` would, so the output is the cold one.
 *
 * @return the resumed position (> 0); 0 when nothing applies (nothing changed); -1 when the
 *         restore copy failed (the card state is undefined: start the sequence over)
 */
int pn_qwen35moe_tt_checkpoint_resume(pn_model *model, pn_state *state, const uint32_t *tokens,
                                      int n);

/**
 * MULTI-TURN: the next prefill of `state` keeps its checkpoint at or below `ceiling` — the end
 * of the prompt's history, i.e. where the chat template's forced generation prompt (the
 * "<think>\n" after the assistant header) begins. A client resends that turn WITHOUT those
 * tokens, so a checkpoint above `ceiling` could never be resumed and the next turn would
 * re-prefill the whole conversation. Applies to that one prefill; <= 0 sets no ceiling.
 */
void pn_qwen35moe_tt_checkpoint_ceiling(pn_model *model, const pn_state *state, int ceiling);

/** 1 when the model keeps the multi-turn prefill checkpoint on the card (enabled, the load
 *  reserved its twins, the recurrence on the card), else 0: think_history drop_last resumes
 *  from it. */
int pn_qwen35moe_tt_checkpoint_available(const pn_model *model);

/** 1 when the card's K/V and carried state hold `state`'s sequence (the last forward on the
 *  card was for it), else 0. An append that continues a sequence needs it. */
int pn_qwen35moe_tt_card_holds(const pn_model *model, const pn_state *state);

/* Release the residency and detach. Never parks the core. */
void pn_qwen35moe_tt_free(pn_model *model);

/**
 * One decode step with the matmuls on the card, conforming to pn_forward_fn.
 *
 * A DISTINCT function pointer from the CPU forward even though it delegates to it: the
 * engine classifies a residency by that pointer, and byte-identical output has never been
 * evidence that a card ran anything.
 */
tm_tensor *pn_qwen35moe_forward_tt(pn_model *model, uint32_t token, pn_state *state);

/**
 * BATCHED PREFILL: up to 32 prompt positions through one set of projections.
 *
 * DEFAULT OFF (`PN_TT_PREFILL=1` arms it) and it REFUSES rather than falls back — a NULL
 * return means the prompt was not touched and the caller's per-token loop is what runs.
 * Ornith's two mixers are both single-position on the card, so a batch buys the projections,
 * the norms, the SwiGLU and the residual adds, and pays a per-layer per-position link round
 * trip to reach each row. `qwen35moe_prefill_tt.c` states that trade and measures it.
 *
 * @return the LAST prompt position's logit row (caller owns), or NULL for NOT TAKEN
 */
tm_tensor *pn_qwen35moe_prefill_tt(pn_model *model, const uint32_t *tokens, int n_tokens,
                                   pn_state *state);

/**
 * What the LAST `pn_qwen35moe_prefill_tt()` call actually batched. It returns logits even
 * when its short tail pass aborted and the prompt ran per token, so a caller that must know
 * the batched path ran — a causality gate — reads this, not the return value.
 *
 * @param model  the loaded model
 * @return       positions that call ran through batched passes; -1 if it declined, aborted
 *               a pass, never ran, or `model` is not on the TT forward
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_qwen35moe_prefill_tt_last_batched(const pn_model *model);

/**
 * Did the last `pn_qwen35moe_prefill_tt()` ABORT part way (as opposed to declining)? Then the
 * card's state was partly written and the session was invalidated (`state->pos` is 0): the
 * caller must FAIL the request, not replay the prompt per token.
 *
 * @return 1 when it aborted, 0 otherwise (declined, succeeded, or not the TT forward)
 */
int pn_qwen35moe_prefill_tt_aborted(const pn_model *model);

/**
 * The matmul split for the run so far: how many went to the card, how many the seam
 * declined (a weight it does not hold), and how many the card refused.
 *
 * An acceptance run reads `on_host == 0 && failed == 0` alongside a non-zero `on_card`.
 * This exists because the card and a silent CPU fallback have returned the same bytes.
 */
void pn_qwen35moe_tt_matmul_counts(const pn_model *model, uint64_t *on_card,
                                   uint64_t *on_host, uint64_t *failed);
#endif /* USE_TT */

/**
 * Detach or reattach the accelerator's matvec seam, and report what it was.
 *
 * **THIS FAMILY'S ACCELERATOR IS NOT A SEPARATE FORWARD — IT IS A HOOK ON THE SHARED
 * CONFIG.** `qwen35moe_mv()` consults `cfg->mv.fn` on every projection, and the upload
 * installs the card's there. So a process whose card is armed has NO CPU forward: calling
 * `pn_qwen35moe_forward` or `pn_bonsai_forward` still routes all 249 projections to the
 * card. A parity harness that scores "the CPU forward" against "the TT forward" in one
 * process is therefore scoring the card against the card unless it detaches this first —
 * and against the RESIDENT-ACTIVATION arm it does not merely read as parity, it segfaults,
 * because that arm keeps the residual stream in card DRAM and the host path hands the seam
 * host buffers.
 *
 * `pn_qwen35moe_forward_tt` does not use the seam on the resident arm (it drives the card
 * directly through `pn_qwen35_tt_resident_forward`), so detaching the seam gives a genuine
 * CPU oracle without disturbing the accelerator arm.
 *
 * @param enable  0 detaches the seam, non-zero restores whatever was detached
 * @return        1 if the seam was attached before this call, 0 if not, -1 on a bad model
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35moe_model_accel_mv_enable(pn_qwen35moe_model *m, int enable);

/**
 * FORGET the accelerator seam entirely — both the live hook and the detached shadow.
 *
 * `..._mv_enable(m, 0)` is a PAUSE: it moves the hook into a shadow copy so a later
 * `enable(m, 1)` can put it back, which is exactly what a parity harness wants around an
 * oracle call. An accelerator being TORN DOWN wants the opposite, and the difference is a
 * use-after-free: the hook's context is the accelerator object, so freeing that object while
 * either copy still names it leaves the next projection reading freed memory. A teardown that
 * only paused the seam would still be holding the pointer in the shadow.
 *
 * Call this before freeing whatever `mv.ctx` points at. Idempotent.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_qwen35moe_model_accel_mv_forget(pn_qwen35moe_model *m);

#endif /* PN_QWEN35MOE_H */
