/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_forward_tt.c
 * @brief The `llama` decode forward on a Tenstorrent Blackhole card — projections, norms,
 *        the SwiGLU gate, the residual adds and the LM head's top-N all on the card, with
 *        the activation kept RESIDENT IN CARD DRAM between them.
 *
 *        Structurally this is still llama_forward_cuda.c: the same ordered layer and the
 *        same host RoPE / KV append / GQA softmax kept for bit-parity with the CPU oracle.
 *        What differs is that the hidden state does not come back to the host between ops.
 *
 * ---------------------------------------------------------------------------
 * WHY THE ACTIVATION STAYS ON THE CARD, AS A NUMBER
 * ---------------------------------------------------------------------------
 * Card->host is ~4.7 MB/s and there is no device->host DMA on Blackhole, so a forward that
 * returns every projection's result to the host spends the token on the link: 0.412 s of a
 * 0.568 s token, 73%, after two rounds of making that transfer as small as it can be. It
 * cannot be made smaller — a matvec's result is already only logical row 0 of each tile.
 * The only move left is not to make it, which means the ops BETWEEN the matmuls have to run
 * on the card too. They do, each on its own resident core (pn_tt_device.h), and they pass
 * activations to each other through DRAM.
 *
 * WHAT STILL CROSSES. `q`, `k` and `v` come back every layer and `attn` goes out. It USED
 * to be forced: the on-card RoPE kernel refuses the INTERLEAVED rotary pairing this model
 * family takes and the converter did not permute wq/wk. **The converter does now**
 * (`--rope-pairing split-half`, recorded in the artifact header), so the kernel's refusal
 * is no longer what holds the readback here — the host-side KV append and GQA softmax are.
 * Those need q/k/v on the host whatever the pairing is, so moving RoPE on-card ALONE
 * removes no transfer; RoPE, KV-append and attention have to move together.
 *
 * ---------------------------------------------------------------------------
 * THE INSTRUMENT — build it before you need it
 * ---------------------------------------------------------------------------
 * With 24 layers and 7 projections each, "the token is wrong" is 169 candidate ops. So
 * every matvec can be scored against Pantheon's own CPU `pn_qmv` on the SAME input, in the
 * same call, under `PN_TT_TRACE`:
 *
 *   PN_TT_TRACE=1   one line per op: shape, max|Δ|, PCC, and the worst element's index
 *   PN_TT_TRACE=2   the above, plus a per-layer summary of the worst op in that layer
 *
 * Tracing now DOWNLOADS each op's input and output, because they no longer pass through
 * host memory. That is a debug cost by construction and it is the reason the trace is a
 * knob rather than a build.
 *
 * Expect a NON-ZERO Δ everywhere and do not read it as a bug: the card's activation format
 * is bf16 and its weights are Bfp8_b, against an fp32 host reference over Q8_0. What the
 * trace is for is finding the op where Δ stops looking like rounding.
 *
 * ---------------------------------------------------------------------------
 * MOVING AN OP ON-CARD MOVES ITS NUMERICS
 * ---------------------------------------------------------------------------
 * The host computes norms, the SwiGLU gate and the residual adds in fp32 over an fp32
 * residual stream. The card computes them in bf16 with an fp32 destination accumulator and
 * writes bf16 back, so every stage moved is one more bf16 rounding of the residual stream.
 * `PN_TT_ONCARD` selects which ops move, so the A/B is one binary — and the difference it
 * shows is a RESULT, not a regression to tune away.
 *
 * @status Norms, SwiGLU, residual adds and top-N on the card behind PN_TT_ONCARD. The
 *         generation loop runs: token N+1 decodes from token N's KV state, token for token
 *         against the CPU oracle, through `pantheon --chat` and through `pantheon serve`.
 *         The artifact is produced from the `.gguf` and cached on first load — a serve gets
 *         a checkpoint path and nothing else (llama_tt_resolve_artifact).
 * @issues None
 * @todo
 *   - [ ] Move RoPE, the KV append and the GQA softmax on-card TOGETHER — separately none
 *         of them removes a byte, because q/k/v cross for whichever of the three is left.
 */

#ifdef USE_TT

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"
#include "llama_residual_trace.h"

#include "../accel/simd.h"
#include "../accel/tt/pn_tt_device.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "tt/convert/ft_float_convert.h"
#include "tt/convert/ft_tt_artifact_cache.h"
#include "tt/ft_tt_memmap.h"

/**
 * An artifact to use INSTEAD of the cache — an override, not the normal path.
 *
 * The normal path is `--model something.gguf` and nothing else: the artifact is produced
 * from the checkpoint on first load and cached beside it. This variable exists for the case
 * the cache cannot serve — an artifact converted with an experimental option, or two
 * artifacts under comparison — and it is an OVERRIDE in the strict sense: it is used
 * verbatim, its provenance is NOT checked against the checkpoint, and if it does not load
 * the run refuses rather than quietly converting a fresh one. That asymmetry is deliberate
 * and is the one `ft_tt_program_open_kernels()` already uses for a program-directory
 * override: an explicit choice wins or fails, and the default is the cache.
 */
#define PN_TT_ARTIFACT_ENV "PN_TT_ARTIFACT"

/** Where cached artifacts live. Unset keeps them beside the checkpoint. */
#define PN_TT_ARTIFACT_CACHE_DIR_ENV "PN_TT_ARTIFACT_CACHE_DIR"

/** Force a rung by name (`bfp8`/`bfp4`) instead of letting the fit choose. An INSTRUMENT:
 *  it still has to fit, and a forced rung that does not is refused with the arithmetic. */
#define PN_TT_RUNG_ENV "PN_TT_RUNG"

/** Force a rotary pairing by name. An instrument; the default is read off the checkpoint. */
#define PN_TT_ROPE_PAIRING_ENV "PN_TT_ROPE_PAIRING"

/** Context the rung choice must leave KV room for. Defaults to the model's own cap. */
#define PN_TT_CONTEXT_TOKENS_ENV "PN_TT_CONTEXT_TOKENS"

/**
 * Card DRAM to hold back from the weights, in MiB.
 *
 * NOT A MEASUREMENT. It is a reserve for everything the forward allocates that is neither a
 * weight nor the KV cache — the rotary table, the activation scratch, the reduce and
 * elementwise residencies' buffers — and it is stated as a round number rather than dressed
 * up as one, because nobody has instrumented the allocator's high-water mark. Raise it if a
 * model the planner says fits runs the allocator out.
 */
#define PN_TT_CARD_RESERVE_MIB_ENV "PN_TT_CARD_RESERVE_MIB"
#define PN_TT_CARD_RESERVE_MIB_DEFAULT 512u

/** Per-op divergence reporting. 0 off, 1 per op, 2 per op plus a per-layer worst. */
#define PN_TT_TRACE_ENV "PN_TT_TRACE"

/** Present the LM head as a SPARSE row built from the on-card top-N instead of reading the
 *  whole vocabulary back — which is the shape the K80 path already ships (`gpu_present_topk`)
 *  and the only arm in which the logit row genuinely never crosses the link. Off by default,
 *  because the parity harness scores a full row. */
#define PN_TT_SPARSE_LOGITS_ENV "PN_TT_SPARSE_LOGITS"

/**
 * Run BOTH attention arms every layer and compare them element for element.
 *
 * The crossed control for on-card attention: the host arm keeps its own fp32 KV cache and
 * computes the GQA softmax as it always did, the card arm keeps its bf16 cache and runs
 * flash decode, and the two answers are scored against each other. It costs the readback
 * on-card attention exists to remove, so it is a knob and never a build.
 */
#define PN_TT_ATTN_CHECK_ENV "PN_TT_ATTN_CHECK"

/**
 * Recover, from the card itself, WHICH rotary frequency it applied to WHICH feature slot.
 *
 * The standing hypothesis for the on-card attention defect is that the card's rotary re-pairs
 * what `--rope-pairing split-half` un-paired. An attention-output check cannot answer it:
 * `q . k` is invariant under any pairing applied to BOTH q and k, so a consistently wrong
 * pairing costs the attention output only its rounding — which is exactly the regime the
 * shipping attn-check already reads (max|d| 0.039). So this asks the question directly.
 *
 * A rotation is invertible from its own operands. With `x` the pre-RoPE strip and `y` the
 * post-RoPE strip, both read off the card, slot `j` and its partner `j+half` satisfy
 *
 *     y[j]      = x[j] cos t - x[j+half] sin t
 *     y[j+half] = x[j] sin t + x[j+half] cos t
 *
 * so `cos t` and `sin t` fall out of two dot products and `t = atan2(sin t, cos t)` is the
 * angle the card ACTUALLY used, to bf16 precision, with no assumption about the table.
 * Comparing it against `position * theta^(-2p/rot)` for every candidate pair index `p` names
 * the frequency the card put on that slot.
 */
#define PN_TT_ROPE_CHECK_ENV "PN_TT_ROPE_CHECK"

/**
 * Measure what rows 1-31 of the strips hold on the on-card attention arm.
 *
 * pn_tt_oncard.c invariant 1 records them as INFERRED harmless and NOT measured: flash decode
 * writes whole output tiles, so `attn` — and through the residual add, `hidden` — are no
 * longer known to be zero there. An inferred-harmless assumption underneath a live defect
 * investigation is worth one measurement.
 */
#define PN_TT_ROW_CENSUS_ENV "PN_TT_ROW_CENSUS"

/**
 * Emulate ONE of the card's attention roundings inside the HOST arm, so the gate can price it.
 *
 * Phase 7.7 records that on-card attention fails the gate where the host arm passes, and that
 * "which of the two on-card changes supplies that rounding is NOT separated" — separating them
 * on the card needs an fp32 KV program that does not exist. It does NOT need one HERE: the
 * host arm can be given each of the card's roundings one at a time, and the gate then prices
 * each in the only currency that matters, per-document perplexity.
 *
 * A comma-separated list of:
 *   `bf16table`  round the host rotary's cos/sin to bf16 — the card's table format. This is
 *                the one rounding that is COHERENT across a whole head: it perturbs the ANGLE
 *                at a position, so every score against that position moves together.
 *   `bf16rot`    round the host's rotated q to bf16 — the card's q strip is bf16.
 *   `bf16kv`     round k and v to bf16 on the way into the host KV cache — the card's cache.
 *   `bf16softmax` the flash-decode kernel's own INTERMEDIATE precision. `attention_float16b`
 *                makes EVERY CB Float16_b, so `cb_qs` (Q pre-scaled by 1/sqrt(head_dim)),
 *                `cb_s` (the scores, then the exponentials in place), the running max and the
 *                row sums all round to bf16 between passes, while the host holds every one of
 *                them in fp32. This is the only one of the four that is not a storage format
 *                anybody chose — it is what running the same arithmetic on the card costs.
 *   `bf16qscale` / `bf16scores` — the two HALVES of `bf16softmax`, so the half with a cheap
 *                fix can be priced alone. `bf16qscale` is only the pre-scale into `cb_qs`;
 *                `bf16scores` is everything downstream of it. `scale_query()` already records
 *                that the pre-scale could be fused into `exp_tile`'s `scale_en` argument for
 *                one fewer pass, so if that half carries the damage the fix is cheaper AND
 *                faster than what ships.
 *
 * They compose, so naming them all is the whole emulation and each name alone is its share.
 * This changes the HOST arm only; it is an instrument and never a default.
 */
#define PN_TT_HOST_EMULATE_ENV "PN_TT_HOST_EMULATE"

/** The widest head this file's on-stack per-head scratch serves. MiniCPM5-1B is 128 and
 *  Qwen3.8 is 256; a model past this is a refusal rather than a stack overrun. */
#define LLAMA_TT_MAX_HEAD_DIM 512

/** A recovered rotation is only readable where the pair carries signal: below this the
 *  denominator `x[j]^2 + x[j+half]^2` is bf16 noise and the angle is arbitrary. */
#define LLAMA_TT_ROPE_MIN_PAIR_ENERGY 1e-4

/** `-std=c11` sets `__STRICT_ANSI__` and glibc then does NOT define `M_PI`. Spelling it here
 *  is one line; discovering it on a Linux host's GCC after a Mac build succeeded is a round. */
#define LLAMA_TT_PI 3.14159265358979323846

/** Two candidate pair indices are only DISTINGUISHABLE when their angles differ by more than
 *  the bf16 table's own error. Slots below this are counted separately rather than voted. */
#define LLAMA_TT_ROPE_MIN_SEPARATION 0.05

/** How far apart two attention outputs may be before an element is called DIVERGED.
 *
 *  Not a pass/fail threshold — the report prints max|Δ| and the count on both sides of it.
 *  It is set at the scale a bf16 KV cache against an fp32 one is EXPECTED to differ by, so
 *  the count separates "rounding, everywhere" from "one element is wrong", which is the
 *  distinction a permutation or an off-by-one position makes and a norm does not. */
#define LLAMA_TT_ATTN_DIVERGENCE 0.05

/** Pairs asked of the on-card top-N. */
#define PN_TT_TOPK_PAIRS 8u

/** One layer's resident projections and the two norms in front of them. */
typedef struct {
    pn_tt_weight wq, wk, wv, wo;
    pn_tt_weight w_gate, w_up, w_down;
    pn_tt_norm_constants attn_norm, ffn_norm;
} pn_llama_tt_layer;

/**
 * Every activation of one token, resident in card DRAM.
 *
 * `hidden` is TWO strips and not one because the residual add is a stream op: the reader is
 * ahead of the writer, so writing the result back over an input would be safe today and
 * would be an unstated dependency on the reader's lead. Two strips and a slot index cost a
 * few KiB of a 31 GiB card and depend on nothing.
 */
typedef struct {
    pn_tt_activation hidden[2];
    pn_tt_activation normed;
    pn_tt_activation q, k, v;
    /** RoPE is out of place: the reader runs ahead of the writer, so rotating over the input
     *  would be an unstated dependency on that lead — the same argument `hidden` makes. */
    pn_tt_activation q_rot, k_rot;
    pn_tt_activation attn;   /**< the GQA output going back out for wo */
    pn_tt_activation project; /**< wo's and ffn_down's output, before the residual add */
    pn_tt_activation gate, up, swiglu;
    pn_tt_activation logits;
    int slot; /**< which `hidden` holds the live residual stream */
} pn_llama_tt_strips;

/** The model-level handle, stored on pn_model::gpu. */
typedef struct {
    pn_tt_device *device;
    pn_llama_tt_layer *layers;
    pn_tt_weight output;
    pn_tt_norm_constants output_norm;
    pn_llama_tt_strips card;
    int n_layers;
    bool loaded;
    bool sparse_logits;
    bool attn_check;      /**< run both attention arms and score them against each other */
    bool rope_check;      /**< recover the card's own rotary angles and name the pairing */
    bool row_census;      /**< measure rows 1-31 of the strips on the on-card arm */
    /* PN_TT_HOST_EMULATE: give the HOST arm one of the card's roundings so the gate can
     * price it. Three independent bits, because the point is to separate them. */
    bool emulate_bf16_table, emulate_bf16_rot, emulate_bf16_kv;
    /** `bf16softmax` is BOTH of these; `bf16qscale` and `bf16scores` are its two halves, so
     *  the one with a cheap fix can be priced on its own. */
    bool emulate_bf16_q_scale, emulate_bf16_scores;
    bool attn_armed;      /**< arming has been ATTEMPTED; see attn_refused for the verdict */
    /** Arming was asked for and FAILED. It latches, so every later token refuses too: a
     *  refusal that becomes a silent host-side fallback on token 2 is the exact defect the
     *  matmul residency's missing-program incident cost a round to. */
    bool attn_refused;
    int trace;

    /* Where the residual stream currently is. Both may be true after a sync; neither is
     * true only before the first token's embedding lands. */
    bool hidden_on_host;
    bool hidden_on_card;

    /* Per-BATCH host scratch — [rows x width] each, allocated the first time a batched
     * prefill runs and NULL until then, so a decode-only run pays nothing for them. Kept
     * SEPARATE from the per-token buffers below rather than resizing those: the single-token
     * forward is the shipping path and every recorded figure is its, so the batched path is
     * not allowed to change a line of it. */
    float *b_hidden, *b_normed, *b_q, *b_k, *b_v, *b_attn, *b_gate, *b_up, *b_scratch;
    int b_rows; /**< rows the batch buffers are sized for; 0 = not allocated */

    /* Per-token host scratch, allocated once at upload. */
    float *hidden, *normed, *q, *k, *v, *attn, *gate, *up, *ffn_out, *scores, *reference;
    float *trace_in, *trace_out;
    float *attn_card; /**< the card's attention output, under PN_TT_ATTN_CHECK only */
    /** The pre- and post-RoPE strips read straight off the card, under PN_TT_ROPE_CHECK, and
     *  one strip's rows 1-31 under PN_TT_ROW_CENSUS. Widest strip either needs. */
    float *probe_in, *probe_out;
    int probe_width;
    int scores_len;

    /* The crossed control's running verdict over the whole run. */
    double attn_check_worst;
    int attn_check_worst_layer;
    long attn_check_diverged;
    double kv_export_worst;
    int kv_export_worst_layer;

    /* The rotary probe's running verdict over the whole run. */
    long rope_slots_identity, rope_slots_reflected, rope_slots_other, rope_slots_degenerate;
    double rope_worst_identity_residual, rope_worst_norm_error;
    /** How far from its own index the winning candidate ever sat, and the angular gap between
     *  the two at that slot. A winner one index away whose angles differ by less than the bf16
     *  table's own error is the ladder being finer than the instrument, not a mis-pairing. */
    int rope_worst_other_distance;
    double rope_worst_other_separation;

    /* The row census's running verdict: the largest magnitude seen in rows 1-31, and how many
     * of those elements were not finite. */
    double row_census_worst;
    long row_census_nonfinite;

    /* The on-card top-N of the last token, and the host argmax it is gated against. */
    uint32_t topk_index[PN_TT_TOPK_PAIRS];
    float topk_value[PN_TT_TOPK_PAIRS];
    uint32_t topk_count;

    /* The device's phase counters as of the PREVIOUS token, so the report can print a
     * per-token cost. The counters themselves are monotonic and never reset — and a running
     * total printed under the words "token done" was read as a per-token figure and turned a
     * 1.9x dispatch regression into a reported 73x. Print both, label both. */
    double last_upload, last_dispatch, last_readback, last_oncard;
    uint64_t last_matvecs;
} pn_llama_tt;

/* ============================================================
 * Host RoPE + GQA-softmax attention
 *
 * Copied from llama_forward_cuda.c, which copied them from llama.c's oracle. They are
 * duplicated rather than shared because bit-parity with the CPU forward is the whole
 * acceptance criterion here: if these drift, an argmax mismatch stops being evidence about
 * the card.
 * ============================================================ */

/* Interleaved RoPE (Llama/Qwen2 permuted GGUF layout) — identical math to llama.c
 * apply_rope_raw. MiniCPM5 (PN_ARCH_LLAMA) takes this path, and it is also the reason the
 * on-card RoPE kernel cannot be used (pn_tt_device.h). */
static void llama_tt_rope_raw(float *data, int n_heads, int head_dim, int rope_dim, int pos,
                              float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < rot; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/** One value put through the card's storage format and back. */
static float llama_tt_to_bf16_and_back(float value)
{
    return ft_float_from_bf16(ft_bf16_from_float(value));
}

/** `count` values through the card's storage format, in place. */
static void llama_tt_round_to_bf16(float *values, int count)
{
    for (int i = 0; i < count; i++)
        values[i] = llama_tt_to_bf16_and_back(values[i]);
}

/**
 * NEOX RoPE with the cos/sin rounded to bf16 — the CARD's table, on the HOST's arm.
 *
 * Everything else is `llama_tt_rope_neox` character for character, so the gate reading is
 * attributable to the table format and to nothing else. The angle is still reduced in float
 * exactly as the shipping path reduces it; only the two table values are narrowed, which is
 * what `ft_tt_op_rope_build_table()` does on the card (`ft_bf16_from_float((float)cos(angle))`).
 *
 * This is an instrument reached only through PN_TT_HOST_EMULATE.
 */
static void llama_tt_rope_neox_bf16_table(float *data, int n_heads, int head_dim, int rope_dim,
                                          int pos, float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = llama_tt_to_bf16_and_back(cosf(angle));
            float sin_a = llama_tt_to_bf16_and_back(sinf(angle));
            float x0 = head[i], x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* NEOX RoPE (Qwen3 unpermuted GGUF layout) — identical to llama.c apply_rope_neox. */
static void llama_tt_rope_neox(float *data, int n_heads, int head_dim, int rope_dim, int pos,
                               float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/**
 * NEOX RoPE for a `--rope-pairing split-half-reversed` artifact: slot `i` carries the
 * model's rotary pair `half - 1 - i`, so it takes that pair's frequency.
 *
 * **This is exact, not an approximation.** The attention score is a sum of one term per
 * rotary pair, `q . k = sum_p (rotated q_p . rotated k_p)`; reflecting the slot index applies
 * the SAME reflection to q and to k, so the set of (frequency, feature-pair) terms is
 * identical and only the order they are summed in moves. That reordering is the whole point
 * of the experiment — it is also what re-draws which 16 output features share a block-float
 * exponent in `attn_q`/`attn_k`.
 *
 * Note the frequency is computed with exactly the expression `llama_tt_rope_neox` uses, on a
 * reflected index rather than a reflected result: `(float)(2*j)/(float)rot` for the same
 * integer `j` produces the same float, so slot `i` here gets bit-identical cos/sin to slot
 * `half-1-i` there and the arms differ only where they are meant to.
 */
/**
 * Interleaved RoPE for a `--rope-pairing interleaved-reversed` artifact: the partner is still
 * adjacent (GGUF's own grouping), but slot `p` carries the model's rotary pair `half - 1 - p`.
 *
 * This is the CROSSED CONTROL for the split-half result. `split-half` changed two things at
 * once — it moved a partner from adjacent to `+half`, AND it thereby stopped a pair's two
 * components sharing a block-float exponent. This arm reflects the pair index without moving
 * the partner, so it re-draws the exponent blocks while KEEPING each pair's two components in
 * one block. Exact in exact arithmetic, for the same reason as the split-half reflection.
 */
static void llama_tt_rope_raw_reversed(float *data, int n_heads, int head_dim, int rope_dim, int pos,
                                       float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < rot; i += 2) {
            int j = half - 1 - (i / 2); /* the model's pair index this slot holds */
            float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

static void llama_tt_rope_neox_reversed(float *data, int n_heads, int head_dim, int rope_dim, int pos,
                                        float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            int j = half - 1 - i; /* the model's pair index this slot holds */
            float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* GQA softmax attention for all heads — identical math to llama.c compute_attn_heads. */
static void llama_tt_attn_heads(const float *q, const float *k_cache, const float *v_cache,
                                float *attn_out, float *scores, int n_head, int n_kv_groups,
                                int head_dim, int max_seq, int seq_len, float scale)
{
    for (int qh = 0; qh < n_head; qh++) {
        int kv_h = qh / n_kv_groups;
        const float *q_vec = q + (size_t)qh * head_dim;
        const float *k_base = k_cache + (size_t)kv_h * max_seq * head_dim;
        const float *v_base = v_cache + (size_t)kv_h * max_seq * head_dim;
        float *out_vec = attn_out + (size_t)qh * head_dim;
        for (int s = 0; s < seq_len; s++)
            scores[s] = pn_dot_f32(q_vec, k_base + (size_t)s * head_dim, head_dim) * scale;
        float max_score = scores[0];
        for (int s = 1; s < seq_len; s++)
            if (scores[s] > max_score) max_score = scores[s];
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        memset(out_vec, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            float w_s = scores[s] * inv_sum;
            const float *v_vec = v_base + (size_t)s * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_vec[d] += w_s * v_vec[d];
        }
    }
}

/**
 * The GQA softmax with the flash-decode kernel's own INTERMEDIATE precision — the host arm
 * doing the card's arithmetic, so the gate can price what the card's circular buffers cost.
 *
 * `attention_float16b.cbtable` makes every CB Float16_b, and the kernel both produces and
 * consumes `cb_s` in place, so each of these lands in bf16 between passes where
 * `llama_tt_attn_heads` keeps fp32 throughout:
 *
 *   - **Q pre-scaled by 1/sqrt(head_dim) into `cb_qs`** — a rounding the host does not have at
 *     all, because it multiplies the SCORE by the scale instead (`scale_query()` explains why
 *     the card scales the smaller operand: `dim_tiles` tile multiplies against
 *     `blocks * block_tiles` score tiles).
 *   - the scores, written to `cb_s` out of an fp32 DST;
 *   - the row max out of `running_max()`;
 *   - the exponentials, written back over `cb_s` in place by `exponentiate_block()`;
 *   - the row sum and its reciprocal.
 *
 * The two accumulations stay fp32 because DST_ACCUM_MODE is 1 on that kernel: what is
 * emulated is where a value is STORED, which is exactly where the card is narrower.
 *
 * Not a claim to be bit-equal to the card — the SFPU exponential and the online block
 * rescaling are not reproduced here. It is a PRICE for the format path, and its value is that
 * it is the only one of the four candidates that is not a storage format anyone chose.
 */
static void llama_tt_attn_heads_bf16_softmax(const float *q, const float *k_cache,
                                             const float *v_cache, float *attn_out,
                                             float *scores, int n_head, int n_kv_groups,
                                             int head_dim, int max_seq, int seq_len, float scale,
                                             bool round_q_scale, bool round_scores)
{
    float q_scaled[LLAMA_TT_MAX_HEAD_DIM];

    for (int qh = 0; qh < n_head; qh++) {
        int kv_h = qh / n_kv_groups;
        const float *q_vec = q + (size_t)qh * head_dim;
        const float *k_base = k_cache + (size_t)kv_h * max_seq * head_dim;
        const float *v_base = v_cache + (size_t)kv_h * max_seq * head_dim;
        float *out_vec = attn_out + (size_t)qh * head_dim;
        float max_score, sum_exp = 0.0f, inv_sum;

        for (int d = 0; d < head_dim; d++) {
            float scaled = q_vec[d] * scale;

            q_scaled[d] = round_q_scale ? llama_tt_to_bf16_and_back(scaled) : scaled;
        }
        for (int s = 0; s < seq_len; s++) {
            float raw = pn_dot_f32(q_scaled, k_base + (size_t)s * head_dim, head_dim);

            scores[s] = round_scores ? llama_tt_to_bf16_and_back(raw) : raw;
        }
        max_score = scores[0];
        for (int s = 1; s < seq_len; s++)
            if (scores[s] > max_score) max_score = scores[s];
        if (round_scores)
            max_score = llama_tt_to_bf16_and_back(max_score);
        for (int s = 0; s < seq_len; s++) {
            float p = expf(scores[s] - max_score);

            scores[s] = round_scores ? llama_tt_to_bf16_and_back(p) : p;
            sum_exp += scores[s];
        }
        if (round_scores) {
            sum_exp = llama_tt_to_bf16_and_back(sum_exp);
            inv_sum = llama_tt_to_bf16_and_back(1.0f / sum_exp);
        } else {
            inv_sum = 1.0f / sum_exp;
        }
        memset(out_vec, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            float w_s = round_scores ? llama_tt_to_bf16_and_back(scores[s] * inv_sum)
                                     : scores[s] * inv_sum;
            const float *v_vec = v_base + (size_t)s * head_dim;

            for (int d = 0; d < head_dim; d++)
                out_vec[d] += w_s * v_vec[d];
        }
    }
}

/* ============================================================
 * The traced matvec — one op, and its divergence from the oracle
 * ============================================================ */

/** max|a-b| and PCC over n elements, plus the index of the worst element. Both metrics,
 *  always: a per-element scale error moves max-abs 35x while PCC barely notices, and a
 *  permutation moves neither — so these are the floor of the evidence, not the whole of it. */
static void llama_tt_score(const float *card, const float *host, int n, double *out_max_abs,
                           double *out_pcc, int *out_worst)
{
    double sum_a = 0.0, sum_b = 0.0, sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
    double worst = -1.0, denom;
    int worst_index = 0;

    for (int i = 0; i < n; i++) {
        double a = (double)card[i], b = (double)host[i];
        double delta = fabs(a - b);

        if (delta > worst) { worst = delta; worst_index = i; }
        sum_a += a; sum_b += b; sum_aa += a * a; sum_bb += b * b; sum_ab += a * b;
    }
    *out_max_abs = worst;
    *out_worst = worst_index;
    denom = sqrt(((double)n * sum_aa - sum_a * sum_a) * ((double)n * sum_bb - sum_b * sum_b));
    *out_pcc = denom > 0.0 ? ((double)n * sum_ab - sum_a * sum_b) / denom : 1.0;
}

/** A block-float exponent is SHARED by 16 consecutive values, so what it costs is the spread
 *  INSIDE a block, not the tensor's overall scale. */
#define LLAMA_TT_BLOCKFLOAT_SPAN 16

/**
 * Dynamic range of one activation vector — the INPUT side of the loss, which the per-op PCC
 * does not see. Three numbers, because they fail differently:
 *
 *   rms    the scale, so a range figure can be read relative to it
 *   crest  max|a| / rms — a single outlier in a vector moves this and nothing else
 *   spread mean over 16-value blocks of log2(block max|a| / block rms), in BITS. This is the
 *          quantity a shared exponent has to cover, so it is the direct test of the
 *          "structured prompts widen the range and block float pays for it" hypothesis.
 */
static void llama_tt_activation_range(const float *a, int n, double *out_rms, double *out_crest,
                                      double *out_spread)
{
    double sum_sq = 0.0, amax = 0.0, spread = 0.0;
    int blocks = 0;

    for (int i = 0; i < n; i += LLAMA_TT_BLOCKFLOAT_SPAN) {
        int span = n - i < LLAMA_TT_BLOCKFLOAT_SPAN ? n - i : LLAMA_TT_BLOCKFLOAT_SPAN;
        double block_sq = 0.0, block_max = 0.0;

        for (int j = i; j < i + span; j++) {
            double v = fabs((double)a[j]);

            block_sq += v * v;
            if (v > block_max) block_max = v;
        }
        sum_sq += block_sq;
        if (block_max > amax) amax = block_max;
        block_sq = sqrt(block_sq / span);
        if (block_sq > 0.0 && block_max > 0.0) {
            spread += log2(block_max / block_sq);
            blocks++;
        }
    }
    *out_rms = sqrt(sum_sq / n);
    *out_crest = *out_rms > 0.0 ? amax / *out_rms : 0.0;
    *out_spread = blocks > 0 ? spread / blocks : 0.0;
}

/**
 * One projection on the card, both operands resident, optionally scored against the CPU
 * oracle on the same input.
 *
 * The comparison is deliberately made on the OP and not on the token, because a wrong
 * answer that only shows up in the argmax has already been through 169 chances to cancel.
 * Under trace this DOWNLOADS both operands — they no longer pass through host memory at
 * all, which is the point of the file and the cost of the instrument.
 *
 * @return 0, or -1 (the caller aborts the forward — a failed matmul must not be absorbed)
 */
static int llama_tt_matvec(pn_llama_tt *g, const pn_tt_weight *weight, const pn_qweight *cpu,
                           const pn_tt_activation *in, const pn_tt_activation *out, int layer,
                           const char *label)
{
    if (pn_tt_matvec_resident(g->device, weight, in, out) != 0) {
        fprintf(stderr, "llama_tt: layer %d %s failed on the card\n", layer, label);
        return -1;
    }
    if (g->trace > 0 && cpu && g->reference && g->trace_in && g->trace_out) {
        double max_abs = 0.0, pcc = 1.0, rms = 0.0, crest = 0.0, spread = 0.0;
        int worst = 0;

        if (pn_tt_activation_download(g->device, in, g->trace_in, weight->in_features) != 0 ||
            pn_tt_activation_download(g->device, out, g->trace_out, weight->out_features) != 0)
            return -1;
        pn_qmv(cpu, g->trace_in, g->reference);
        llama_tt_score(g->trace_out, g->reference, (int)weight->out_features, &max_abs, &pcc,
                       &worst);
        llama_tt_activation_range(g->trace_in, (int)weight->in_features, &rms, &crest, &spread);
        fprintf(stderr, "llama_tt trace: L%02d %-12s Kt=%-4u Nt=%-5u max|d|=%.6f pcc=%.9f "
                        "worst[%d] card=%.6f cpu=%.6f in_rms=%.6f in_crest=%.4f "
                        "in_spread_bits=%.4f\n",
                layer, label, weight->tile_rows, weight->tile_cols, max_abs, pcc, worst,
                (double)g->trace_out[worst], (double)g->reference[worst], rms, crest, spread);
    }
    return 0;
}

/**
 * The same projection, ISSUED and not awaited — for the ones a layer can run together.
 *
 * Within a layer q/k/v read one `normed` and write three disjoint strips, and gate/up read
 * one `normed` and write two: nothing in either set reads what the set writes, so their
 * descriptors may sit in the members' rings at once. The card runs them back to back at its
 * own 0.39-0.45 us op-to-op instead of one host round trip apiece, and the host waits FOUR
 * times a layer instead of seven.
 *
 * UNDER TRACE IT DEGRADES TO THE WAITING FORM, deliberately: the trace scores the output
 * against the CPU oracle, and reading an output IS waiting for it. An instrument that
 * silently changed the thing it measures would be worse than a slower one.
 */
static int llama_tt_matvec_begin(pn_llama_tt *g, const pn_tt_weight *weight,
                                 const pn_qweight *cpu, const pn_tt_activation *in,
                                 const pn_tt_activation *out, int layer, const char *label)
{
    if (g->trace > 0 && cpu && g->reference && g->trace_in && g->trace_out)
        return llama_tt_matvec(g, weight, cpu, in, out, layer, label);
    if (pn_tt_matvec_enqueue(g->device, weight, in, out) != 0) {
        fprintf(stderr, "llama_tt: layer %d %s failed on the card\n", layer, label);
        return -1;
    }
    return 0;
}

/* ============================================================
 * Where the residual stream lives
 * ============================================================ */

/** The strip currently holding the residual stream, and the other one. */
static const pn_tt_activation *llama_tt_hidden(pn_llama_tt *g)
{
    return &g->card.hidden[g->card.slot];
}

static const pn_tt_activation *llama_tt_hidden_spare(pn_llama_tt *g)
{
    return &g->card.hidden[g->card.slot ^ 1];
}

/** Make the HOST copy current, reading it back only if it is not already. Every call site
 *  is a place where an op did NOT move on-card. */
static int llama_tt_hidden_to_host(pn_llama_tt *g, int n_embd)
{
    if (g->hidden_on_host)
        return 0;
    if (pn_tt_activation_download(g->device, llama_tt_hidden(g), g->hidden, (uint32_t)n_embd) != 0)
        return -1;
    g->hidden_on_host = true;
    return 0;
}

/** Make the CARD copy current, writing it only if it is not already. */
static int llama_tt_hidden_to_card(pn_llama_tt *g, int n_embd)
{
    if (g->hidden_on_card)
        return 0;
    if (pn_tt_activation_upload(g->device, llama_tt_hidden(g), g->hidden, (uint32_t)n_embd) != 0)
        return -1;
    g->hidden_on_card = true;
    return 0;
}

/** One RMS norm into `g->card.normed`, on the card or on the host, leaving the result
 *  where the next matmul reads it either way. */
static int llama_tt_norm(pn_llama_tt *g, const pn_tt_norm_constants *constants,
                         const float *host_gamma, const pn_config *cfg)
{
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM)) {
        if (llama_tt_hidden_to_card(g, cfg->n_embd) != 0)
            return -1;
        return pn_tt_rms_norm(g->device, constants, llama_tt_hidden(g), &g->card.normed);
    }
    if (llama_tt_hidden_to_host(g, cfg->n_embd) != 0)
        return -1;
    pn_vrms_norm_f32(g->hidden, host_gamma, g->normed, cfg->n_embd, cfg->rms_norm_eps);
    return pn_tt_activation_upload(g->device, &g->card.normed, g->normed, (uint32_t)cfg->n_embd);
}

/** hidden += `g->card.project`, on the card or on the host. */
static int llama_tt_residual(pn_llama_tt *g, const pn_config *cfg)
{
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_RESIDUAL)) {
        if (llama_tt_hidden_to_card(g, cfg->n_embd) != 0)
            return -1;
        if (pn_tt_add(g->device, llama_tt_hidden(g), &g->card.project,
                      llama_tt_hidden_spare(g)) != 0)
            return -1;
        g->card.slot ^= 1;
        g->hidden_on_card = true;
        g->hidden_on_host = false;
        return 0;
    }
    if (pn_tt_activation_download(g->device, &g->card.project, g->ffn_out,
                                  (uint32_t)cfg->n_embd) != 0)
        return -1;
    if (llama_tt_hidden_to_host(g, cfg->n_embd) != 0)
        return -1;
    pn_vadd_f32(g->hidden, g->ffn_out, g->hidden, cfg->n_embd);
    g->hidden_on_host = true;
    g->hidden_on_card = false;
    return 0;
}

/* ============================================================
 * Residency
 * ============================================================ */

/** Bind one artifact tensor name to a resident weight, reporting the miss by name. */
static int llama_tt_bind(pn_llama_tt *g, const char *name, pn_tt_weight *out)
{
    if (pn_tt_device_find_weight(g->device, name, out) != 0) {
        fprintf(stderr, "llama_tt: artifact has no %s\n", name);
        return -1;
    }
    return 0;
}

/** Every projection of one layer, by the GGUF names the converter preserved. */
static int llama_tt_bind_layer(pn_llama_tt *g, int layer)
{
    pn_llama_tt_layer *d = &g->layers[layer];
    char name[128];
    struct { const char *suffix; pn_tt_weight *slot; } bindings[] = {
        { "attn_q",      &d->wq },     { "attn_k",    &d->wk },
        { "attn_v",      &d->wv },     { "attn_output", &d->wo },
        { "ffn_gate",    &d->w_gate }, { "ffn_up",    &d->w_up },
        { "ffn_down",    &d->w_down },
    };

    for (size_t i = 0; i < sizeof(bindings) / sizeof(bindings[0]); i++) {
        if (snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, bindings[i].suffix) < 0)
            return -1;
        if (llama_tt_bind(g, name, bindings[i].slot) != 0)
            return -1;
    }
    return 0;
}

/** Allocate the per-token host scratch. One allocation per vector, once, at load.
 *
 *  `scores` is NOT here: its length is the KV ring's max_seq, which belongs to the session
 *  and not to the model, so it is sized on the first forward from the live cache. */
static int llama_tt_alloc_scratch(pn_llama_tt *g, const pn_config *cfg)
{
    int n_embd = cfg->n_embd;
    int q_width = cfg->n_head * pn_head_dim(cfg);
    int kv_width = cfg->n_kv_head * pn_head_dim(cfg);
    int widest = cfg->n_vocab > cfg->n_ff ? cfg->n_vocab : cfg->n_ff;

    g->hidden = calloc((size_t)n_embd, sizeof(float));
    g->normed = calloc((size_t)n_embd, sizeof(float));
    g->q = calloc((size_t)q_width, sizeof(float));
    g->k = calloc((size_t)kv_width, sizeof(float));
    g->v = calloc((size_t)kv_width, sizeof(float));
    g->attn = calloc((size_t)q_width, sizeof(float));
    g->gate = calloc((size_t)cfg->n_ff, sizeof(float));
    g->up = calloc((size_t)cfg->n_ff, sizeof(float));
    g->ffn_out = calloc((size_t)n_embd, sizeof(float));
    g->reference = g->trace > 0 ? calloc((size_t)widest, sizeof(float)) : NULL;
    g->trace_in = g->trace > 0 ? calloc((size_t)widest, sizeof(float)) : NULL;
    g->trace_out = g->trace > 0 ? calloc((size_t)widest, sizeof(float)) : NULL;
    g->attn_card = g->attn_check ? calloc((size_t)q_width, sizeof(float)) : NULL;
    /* One width serves both probes: the rotary check reads q strips (q_width) and the row
     * census reads `attn` (q_width) and `hidden` (n_embd), so the widest of the three. */
    g->probe_width = q_width > n_embd ? q_width : n_embd;
    g->probe_in = (g->rope_check || g->row_census)
                      ? calloc((size_t)g->probe_width, sizeof(float)) : NULL;
    g->probe_out = (g->rope_check || g->row_census)
                       ? calloc((size_t)g->probe_width, sizeof(float)) : NULL;
    return (g->hidden && g->normed && g->q && g->k && g->v && g->attn && g->gate && g->up &&
            g->ffn_out && (!g->attn_check || g->attn_card) &&
            (!(g->rope_check || g->row_census) || (g->probe_in && g->probe_out)) &&
            (g->trace == 0 || (g->reference && g->trace_in && g->trace_out))) ? 0 : -1;
}

/**
 * Every activation of one token, allocated ONCE in card DRAM and zeroed.
 *
 * The zeroing is load-bearing rather than tidy: a matvec writes only logical row 0 of each
 * tile and the other 31 rows are fed to the FPU anyway, so they have to be zero and have to
 * stay zero. Every on-card op here preserves that for free.
 */
static int llama_tt_alloc_strips(pn_llama_tt *g, const pn_config *cfg)
{
    pn_llama_tt_strips *s = &g->card;
    const pn_llama_tt_layer *l0 = &g->layers[0];
    uint32_t embd_tiles = (uint32_t)cfg->n_embd / 32u;
    uint32_t ff_tiles = (uint32_t)cfg->n_ff / 32u;

    if (pn_tt_activation_alloc(g->device, embd_tiles, &s->hidden[0]) != 0 ||
        pn_tt_activation_alloc(g->device, embd_tiles, &s->hidden[1]) != 0 ||
        pn_tt_activation_alloc(g->device, embd_tiles, &s->normed) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wq.tile_cols, &s->q) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wk.tile_cols, &s->k) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wv.tile_cols, &s->v) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wq.tile_cols, &s->q_rot) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wk.tile_cols, &s->k_rot) != 0 ||
        pn_tt_activation_alloc(g->device, l0->wo.tile_rows, &s->attn) != 0 ||
        pn_tt_activation_alloc(g->device, embd_tiles, &s->project) != 0 ||
        pn_tt_activation_alloc(g->device, ff_tiles, &s->gate) != 0 ||
        pn_tt_activation_alloc(g->device, ff_tiles, &s->up) != 0 ||
        pn_tt_activation_alloc(g->device, ff_tiles, &s->swiglu) != 0 ||
        pn_tt_activation_alloc(g->device, g->output.tile_cols, &s->logits) != 0)
        return -1;
    s->slot = 0;
    return 0;
}

/** Expand every norm weight into the card-side constants the reduce op reads. Once, at
 *  load: a per-token upload of gamma would put back a transfer this file exists to remove. */
static int llama_tt_upload_norms(pn_llama_tt *g, pn_llama_qweights *qw, const pn_config *cfg)
{
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        if (pn_tt_norm_constants_upload(g->device, qw->layers[layer].attn_norm,
                                        (uint32_t)cfg->n_embd, cfg->rms_norm_eps,
                                        &g->layers[layer].attn_norm) != 0)
            return -1;
        if (pn_tt_norm_constants_upload(g->device, qw->layers[layer].ffn_norm,
                                        (uint32_t)cfg->n_embd, cfg->rms_norm_eps,
                                        &g->layers[layer].ffn_norm) != 0)
            return -1;
    }
    return pn_tt_norm_constants_upload(g->device, qw->output_norm, (uint32_t)cfg->n_embd,
                                       cfg->rms_norm_eps, &g->output_norm);
}

/** Does layer 0 carry a host-side step between the projections and RoPE — the Qwen2
 *  biases or the Qwen3 per-head Q/K norm (pn_llama_qkv_finish)? The card does neither. */
static bool llama_tt_needs_qkv_finish(const pn_model *model)
{
    const pn_llama_qlayer *l = &((const pn_llama_qweights *)model->weights)->layers[0];

    return l->bq || l->bk || l->bv || l->attn_q_norm || l->attn_k_norm;
}

/**
 * Arm the three attention residencies and the card-resident KV cache, once, on the first
 * forward.
 *
 * It is LAZY rather than done at load because the cache's capacity is the SESSION's
 * `max_seq`, which does not exist at model load — sizing it from the model instead would
 * allocate the checkpoint's 131,072-position context (3 GiB) for a serve that asked for 2k.
 *
 * **A non-split-half artifact is a REFUSAL.** foundry's RoPE kernel implements NeoX pairing
 * and refuses interleaved (2i, 2i+1) pairing by shape; a llama-family artifact converted
 * WITHOUT `--rope-pairing split-half` holds q and k in interleaved order, and rotating those
 * as NeoX is a wrong answer with no error anywhere. The host arm reads the same flag to pick
 * its own rotation, so the two arms cannot disagree about it either.
 */
static int llama_tt_ensure_attention(pn_llama_tt *g, pn_model *model, int max_seq)
{
    pn_config *cfg = &model->config;

    if (g->attn_refused)
        return -1;
    if (g->attn_armed)
        return 0;
    g->attn_armed = true;
    if (!pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_ATTENTION))
        return 0;
    /* TWO refusals for TWO independent artifact bits, and the reflected one is tested FIRST
     * because a `split-half-reversed` artifact is split-half SHAPED and would otherwise pass
     * the second check and run silently wrong: the on-card RoPE table is built with the
     * frequency in SLOT order, so a reflected artifact gets every frequency on the wrong
     * feature. `interleaved-reversed` fails both, and it should say the reflection first
     * because that is the reason a fixed converter flag could not rescue it. */
    if (pn_tt_device_rope_pairs_reversed(g->device)) {
        fprintf(stderr, "llama_tt: REFUSED — this artifact reflects the rotary pair index "
                        "(slot p carries pair half-1-p). The on-card RoPE table is built in "
                        "SLOT order, so the card would apply every frequency to the wrong "
                        "feature. It is an instrument: run it with host attention.\n");
        g->attn_refused = true;
        return -1;
    }
    if (pn_tt_device_kv_type(g->device).rotate) {
        fprintf(stderr, "llama_tt: REFUSED — KV type k4h rotates Q and K after RoPE, which only "
                        "the Qwen3.5-family forward does. Use bfp4 here.\n");
        g->attn_refused = true;
        return -1;
    }
    if (llama_tt_needs_qkv_finish(model)) {
        fprintf(stderr, "llama_tt: REFUSED — on-card attention keeps q/k on the card, and this "
                        "model needs its q/k bias or per-head Q/K norm applied before RoPE, "
                        "which only the host arm does. Run it with host attention.\n");
        g->attn_refused = true;
        return -1;
    }
    if (cfg->arch != PN_ARCH_QWEN3 && !pn_tt_device_rope_is_split_half(g->device)) {
        fprintf(stderr, "llama_tt: REFUSED — on-card attention needs a `--rope-pairing "
                        "split-half` artifact; this one is interleaved and the on-card RoPE "
                        "kernel cannot express that pairing\n");
        g->attn_refused = true;
        return -1;
    }
    if (pn_tt_device_arm_attention(g->device, (uint32_t)cfg->n_head, (uint32_t)cfg->n_kv_head,
                                   (uint32_t)pn_head_dim(cfg), (uint32_t)cfg->rope_dim,
                                   (double)cfg->rope_theta, (uint32_t)max_seq,
                                   (uint32_t)cfg->n_layer) != 0) {
        g->attn_refused = true;
        return -1;
    }
    return 0;
}

/**
 * The session-export hook: move one layer's K or V between the card and a packed fp32 array.
 *
 * Installed on the KV cache when — and only when — attention is running on the card, because
 * that is the only arm on which the host tensors are stale. It is the ONLY place the card's
 * KV cache crosses the link, and it runs at export/restore time rather than per token; the
 * arithmetic for that choice is in pn_tt_device.h and llama_state_snapshot.c.
 */
static int llama_tt_kv_move(void *device_v, int layer, int values, int position_count,
                            float *rows, int capture)
{
    pn_tt_device *device = (pn_tt_device *)device_v;

    if (!device || layer < 0 || position_count <= 0 || !rows)
        return -1;
    if (capture)
        return pn_tt_kv_download(device, (uint32_t)layer, values != 0,
                                 (uint32_t)position_count, rows);
    return pn_tt_kv_upload(device, (uint32_t)layer, values != 0, (uint32_t)position_count,
                           rows);
}

/** Size the attention score row to the live KV ring, once. */
static int llama_tt_ensure_scores(pn_llama_tt *g, int max_seq)
{
    if (g->scores && g->scores_len >= max_seq)
        return 0;
    free(g->scores);
    g->scores = calloc((size_t)max_seq, sizeof(float));
    g->scores_len = g->scores ? max_seq : 0;
    return g->scores ? 0 : -1;
}

/**
 * KV bytes one token costs in card DRAM, across every layer.
 *
 * The card's KV cache is bf16 and holds K and V for every KV head at full head_dim
 * (ft_tt_kv_cache_layout_init), so this is 2 regions x heads x dim x 2 bytes per layer. It
 * is the model's geometry, not a guess, and it is what makes "how much context does this
 * rung leave" answerable.
 */
static uint64_t llama_tt_kv_bytes_per_token(const pn_config *cfg)
{
    uint64_t heads = (uint64_t)(cfg->n_kv_head > 0 ? cfg->n_kv_head : cfg->n_head);
    uint64_t dim = (uint64_t)pn_head_dim(cfg);

    return 2ull * heads * dim * 2ull * (uint64_t)cfg->n_layer;
}

/** The context a rung choice has to leave room for. */
static uint32_t llama_tt_context_tokens(const pn_config *cfg)
{
    const char *override_value = getenv(PN_TT_CONTEXT_TOKENS_ENV);

    if (override_value && *override_value) {
        unsigned long parsed = strtoul(override_value, NULL, 10);

        if (parsed > 0ul)
            return (uint32_t)parsed;
    }
    if (cfg->ctx_cap > 0)
        return (uint32_t)cfg->ctx_cap;
    return cfg->n_ctx > 0 ? (uint32_t)cfg->n_ctx : 4096u;
}

/** An env override that named something this build does not know. Refuse, never ignore. */
#define LLAMA_TT_ENV_BAD (-2)

/** A rung forced by the environment: an ft_tt_ledger_mode, -1 for "choose", or _ENV_BAD. */
static int llama_tt_forced_rung(void)
{
    const char *name = getenv(PN_TT_RUNG_ENV);
    int mode;

    if (!name || !*name)
        return -1;
    mode = ft_tt_ledger_mode_from_name(name);
    if (mode < 0) {
        fprintf(stderr, "llama_tt: REFUSED — %s='%s' is not a rung name\n", PN_TT_RUNG_ENV, name);
        return LLAMA_TT_ENV_BAD;
    }
    return mode;
}

/** A pairing forced by the environment: an ft_tt_rope_pairing, -1 for "read it off the
 *  checkpoint", or _ENV_BAD. A misspelling that silently fell back to the default would read
 *  as "this pairing costs nothing", which is the most expensive way to be wrong here. */
static int llama_tt_forced_rope_pairing(void)
{
    static const struct { const char *name; ft_tt_rope_pairing pairing; } NAMES[] = {
        {"source", FT_TT_ROPE_PAIRING_SOURCE},
        {"split-half", FT_TT_ROPE_PAIRING_SPLIT_HALF},
        {"split-half-reversed", FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED},
        {"interleaved-reversed", FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED}
    };
    const char *name = getenv(PN_TT_ROPE_PAIRING_ENV);
    size_t i;

    if (!name || !*name)
        return -1;
    for (i = 0; i < sizeof(NAMES) / sizeof(NAMES[0]); i++) {
        if (strcmp(name, NAMES[i].name) == 0)
            return (int)NAMES[i].pairing;
    }
    fprintf(stderr, "llama_tt: REFUSED — %s='%s' names none of source, split-half, "
                    "split-half-reversed, interleaved-reversed\n",
            PN_TT_ROPE_PAIRING_ENV, name);
    return LLAMA_TT_ENV_BAD;
}

/**
 * Turn a `.gguf` path into a card-ready artifact, converting and caching it if need be.
 *
 * THIS IS THE WHOLE PRODUCT REQUIREMENT IN ONE FUNCTION: a user downloads a GGUF and our
 * software makes it run. They are never asked which rung, never told their file is the
 * wrong format, and never run a converter by hand. What they see on a first load is a
 * progress bar, and on every load after that, nothing.
 *
 * @param model  a loaded model whose `path` is the checkpoint
 * @param out    receives the artifact path to load
 * @param out_bytes  its size
 * @return       0 and a usable path, or -1 with the reason already printed
 */
static int llama_tt_resolve_artifact(const pn_model *model, char *out, size_t out_bytes)
{
    const char *override_path = getenv(PN_TT_ARTIFACT_ENV);
    const char *reserve_mib = getenv(PN_TT_CARD_RESERVE_MIB_ENV);
    const pn_config *cfg = &model->config;
    ft_tt_artifact_cache_request request;
    ft_tt_artifact_cache_result result;
    char line[FT_TT_CAPACITY_REASON_BYTES];
    unsigned long reserve;
    int rc;

    if (override_path && *override_path) {
        if (snprintf(out, out_bytes, "%s", override_path) >= (int)out_bytes) {
            fprintf(stderr, "llama_tt: REFUSED — %s is longer than %zu bytes\n",
                    PN_TT_ARTIFACT_ENV, out_bytes);
            return -1;
        }
        fprintf(stderr, "llama_tt: artifact %s (%s override — its provenance is NOT checked "
                        "against the checkpoint)\n", out, PN_TT_ARTIFACT_ENV);
        return 0;
    }
    if (model->card_artifact && *model->card_artifact) {
        /* A card metadata GGUF's own artifact (pn_load attached it): nothing to convert. */
        if (snprintf(out, out_bytes, "%s", model->card_artifact) >= (int)out_bytes) {
            fprintf(stderr, "llama_tt: REFUSED — the card artifact's path does not fit\n");
            return -1;
        }
        fprintf(stderr, "llama_tt: artifact %s (prebuilt, beside the metadata GGUF)\n", out);
        return 0;
    }

    if (!model->path || !*model->path) {
        fprintf(stderr, "llama_tt: REFUSED — the model has no checkpoint path to convert "
                        "from, and %s is not set\n", PN_TT_ARTIFACT_ENV);
        return -1;
    }

    reserve = PN_TT_CARD_RESERVE_MIB_DEFAULT;
    if (reserve_mib && *reserve_mib)
        reserve = strtoul(reserve_mib, NULL, 10);

    memset(&request, 0, sizeof(request));
    request.gguf_path = model->path;
    request.cache_dir = getenv(PN_TT_ARTIFACT_CACHE_DIR_ENV);
    request.budget.card_bytes = (uint64_t)FT_TT_MEM_DRAM_SIZE * 8ull;
    request.budget.runtime_state_bytes = (uint64_t)reserve * 1024ull * 1024ull;
    request.budget.kv_bytes_per_token = llama_tt_kv_bytes_per_token(cfg);
    request.budget.context_tokens = llama_tt_context_tokens(cfg);
    request.progress_style = FT_TT_CACHE_PROGRESS_STDERR;

    request.forced_mode = llama_tt_forced_rung();
    request.rope_pairing = llama_tt_forced_rope_pairing();
    if (request.forced_mode == LLAMA_TT_ENV_BAD || request.rope_pairing == LLAMA_TT_ENV_BAD)
        return -1;

    rc = ft_tt_artifact_cache_ensure(&request, &result);
    if (rc != FT_TT_ARTIFACT_CACHE_OK) {
        /* A refusal carries the arithmetic. It is not a fallback: the caller stays on the
         * CPU and says so, rather than converting something that will not fit. */
        fprintf(stderr, "llama_tt: REFUSED — %s\n", result.reason);
        return -1;
    }

    /* THE ONE LINE. Which rung, why, what it left, and whether anything was converted. */
    fprintf(stderr, "llama_tt: %s — %s\n", result.produced ? "converted" : "cached",
            ft_tt_capacity_selection_line(&result.selection, &request.budget, line,
                                          sizeof(line)));
    if (result.produced && result.had_artifact && result.note[0])
        fprintf(stderr, "llama_tt: the cached artifact was NOT reusable and was not served — "
                        "%s\n", result.note);
    fprintf(stderr, "llama_tt: rotary pairing %s (%s); KV %llu B/token, %u tokens reserved\n",
            result.rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF ? "split-half" : "source",
            result.rope_reason ? result.rope_reason : "",
            (unsigned long long)request.budget.kv_bytes_per_token,
            request.budget.context_tokens);

    if (snprintf(out, out_bytes, "%s", result.artifact_path) >= (int)out_bytes) {
        fprintf(stderr, "llama_tt: REFUSED — the cached artifact's path does not fit\n");
        return -1;
    }
    return 0;
}

int pn_llama_tt_upload(pn_model *model)
{
    char artifact[FT_TT_ARTIFACT_CACHE_PATH_BYTES];
    uint32_t eltwise_tiles[2];
    pn_llama_qweights *qw;
    pn_llama_tt *g;
    pn_config *cfg;
    const char *trace;
    const char *sparse;
    const char *check;

    if (!model || !model->weights)
        return -1;
    qw = (pn_llama_qweights *)model->weights;
    cfg = &model->config;
    if (!qw->is_quantized) {
        fprintf(stderr, "llama_tt: F32 weights are not a converted artifact; staying on CPU\n");
        return -1;
    }
    if (cfg->n_experts > 0) {
        fprintf(stderr, "llama_tt: MoE is not built on this backend; staying on CPU\n");
        return -1;
    }
    g = calloc(1, sizeof(*g));
    if (!g)
        return -1;
    trace = getenv(PN_TT_TRACE_ENV);
    g->trace = trace ? atoi(trace) : 0;
    sparse = getenv(PN_TT_SPARSE_LOGITS_ENV);
    g->sparse_logits = sparse && atoi(sparse) != 0;
    check = getenv(PN_TT_ATTN_CHECK_ENV);
    g->attn_check = check && atoi(check) != 0;
    check = getenv(PN_TT_ROPE_CHECK_ENV);
    g->rope_check = check && atoi(check) != 0;
    check = getenv(PN_TT_ROW_CENSUS_ENV);
    g->row_census = check && atoi(check) != 0;
    check = getenv(PN_TT_HOST_EMULATE_ENV);
    if (check && *check) {
        g->emulate_bf16_table = strstr(check, "bf16table") != NULL;
        g->emulate_bf16_rot = strstr(check, "bf16rot") != NULL;
        g->emulate_bf16_kv = strstr(check, "bf16kv") != NULL;
        g->emulate_bf16_q_scale =
            strstr(check, "bf16softmax") != NULL || strstr(check, "bf16qscale") != NULL;
        g->emulate_bf16_scores =
            strstr(check, "bf16softmax") != NULL || strstr(check, "bf16scores") != NULL;
        /* A misspelled name that silently emulates nothing would read as "this rounding costs
         * nothing", which is the most expensive possible way to be wrong here. */
        if (!g->emulate_bf16_table && !g->emulate_bf16_rot && !g->emulate_bf16_kv &&
            !g->emulate_bf16_q_scale && !g->emulate_bf16_scores) {
            fprintf(stderr, "llama_tt: REFUSED — %s='%s' names none of bf16table, bf16rot, "
                            "bf16kv, bf16softmax, bf16qscale, bf16scores\n",
                    PN_TT_HOST_EMULATE_ENV, check);
            free(g);
            return -1;
        }
        if ((g->emulate_bf16_q_scale || g->emulate_bf16_scores) &&
            pn_head_dim(cfg) > LLAMA_TT_MAX_HEAD_DIM) {
            fprintf(stderr, "llama_tt: REFUSED — bf16softmax holds one head on the stack and "
                            "head_dim %d exceeds %d\n", pn_head_dim(cfg), LLAMA_TT_MAX_HEAD_DIM);
            free(g);
            return -1;
        }
        fprintf(stderr, "llama_tt: HOST ARM EMULATING the card — bf16table %d bf16rot %d "
                        "bf16kv %d bf16qscale %d bf16scores %d. This is an instrument; the "
                        "card arm is unaffected.\n",
                (int)g->emulate_bf16_table, (int)g->emulate_bf16_rot, (int)g->emulate_bf16_kv,
                (int)g->emulate_bf16_q_scale, (int)g->emulate_bf16_scores);
    }
    g->n_layers = cfg->n_layer;

    /* The card is opened BEFORE the artifact is resolved, and the order is load-bearing:
     * resolving can mean a twenty-minute conversion, and doing that on a machine whose card
     * then fails to open would burn the twenty minutes to end up on the CPU anyway. */
    if (pn_tt_device_open(0, &g->device) != 0)
        goto fail;
    if (llama_tt_resolve_artifact(model, artifact, sizeof(artifact)) != 0)
        goto fail;
    if (pn_tt_device_load_artifact(g->device, artifact) != 0)
        goto fail;
    /* The host arm applies the q/k bias and per-head norm in the GGUF's feature order. A
     * permuted artifact (split-half, or a reflected instrument) moves q/k features away from
     * their bias and gamma entries: a smoothly wrong answer, so refuse it. */
    if ((pn_tt_device_rope_is_split_half(g->device) ||
         pn_tt_device_rope_pairs_reversed(g->device)) && llama_tt_needs_qkv_finish(model)) {
        fprintf(stderr, "llama_tt: REFUSED — this artifact permutes q/k, and the model's q/k "
                        "bias / per-head norm are in the checkpoint's order\n");
        goto fail;
    }

    g->layers = calloc((size_t)cfg->n_layer, sizeof(*g->layers));
    if (!g->layers)
        goto fail;
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        if (llama_tt_bind_layer(g, layer) != 0)
            goto fail;
    }
    if (llama_tt_bind(g, "output.weight", &g->output) != 0)
        goto fail;

    /* The shape the artifact says must be the shape the engine says. A silent disagreement
     * here is a matmul with the right tile counts and the wrong meaning. */
    if ((int)g->output.out_features != cfg->n_vocab ||
        (int)g->layers[0].wq.in_features != cfg->n_embd) {
        fprintf(stderr, "llama_tt: artifact shape (vocab %u, n_embd %u) disagrees with the "
                        "checkpoint (vocab %d, n_embd %d)\n",
                g->output.out_features, g->layers[0].wq.in_features, cfg->n_vocab, cfg->n_embd);
        goto fail;
    }
    if (llama_tt_alloc_scratch(g, cfg) != 0)
        goto fail;
    if (llama_tt_alloc_strips(g, cfg) != 0)
        goto fail;

    /* The shapes are the MODEL's, not the artifact's: the reduce residency's buffers are
     * placed for ONE width and the elementwise residency runs ONE block depth for every
     * descriptor it will ever see, so both have to be decided against the widths this
     * forward will actually ask for. */
    eltwise_tiles[0] = (uint32_t)cfg->n_embd / 32u;
    eltwise_tiles[1] = (uint32_t)cfg->n_ff / 32u;
    if (pn_tt_device_arm_oncard(g->device, (uint32_t)cfg->n_embd / 32u, eltwise_tiles, 2u,
                                g->output.tile_cols) != 0)
        goto fail;
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM) && llama_tt_upload_norms(g, qw, cfg) != 0)
        goto fail;

    g->loaded = true;
    model->gpu = g;
    fprintf(stderr, "llama_tt: %u weights resident (%.1f MiB), %d layers, trace=%d\n",
            pn_tt_device_weight_count(g->device),
            (double)pn_tt_device_weight_bytes(g->device) / (1024.0 * 1024.0), cfg->n_layer,
            g->trace);
    return 0;

fail:
    model->gpu = g;
    pn_llama_tt_free(model);
    return -1;
}

int pn_llama_model_tt_loaded(const pn_model *model)
{
    const pn_llama_tt *g = model ? (const pn_llama_tt *)model->gpu : NULL;

    return (g && g->loaded) ? 1 : 0;
}

void pn_llama_tt_free(pn_model *model)
{
    pn_llama_tt *g = model ? (pn_llama_tt *)model->gpu : NULL;

    if (!g)
        return;
    pn_tt_device_close(g->device);
    free(g->layers);
    free(g->hidden); free(g->normed); free(g->q); free(g->k); free(g->v);
    free(g->attn); free(g->gate); free(g->up); free(g->ffn_out); free(g->scores);
    free(g->reference); free(g->trace_in); free(g->trace_out); free(g->attn_card);
    free(g->probe_in); free(g->probe_out);
    free(g->b_hidden); free(g->b_normed); free(g->b_q); free(g->b_k); free(g->b_v);
    free(g->b_attn); free(g->b_gate); free(g->b_up); free(g->b_scratch);
    free(g);
    model->gpu = NULL;
}

/* ============================================================
 * Forward
 * ============================================================ */

/**
 * ONE position's rotary, KV append and GQA softmax, over q/k/v ALREADY in host memory.
 *
 * This is the body `llama_tt_attention_host` used to hold between its download and its
 * return, factored out with the buffers as arguments and NOTHING else changed. The batched
 * prefill calls it once per row, so the two paths run the SAME arithmetic per position and a
 * divergence between them cannot be a second copy of this having drifted. `q` and `k` are
 * rotated IN PLACE, exactly as before.
 */
static int llama_tt_attend_one(pn_llama_tt *g, pn_model *model, pn_state *state, int layer,
                               int pos, float *q, float *k, float *v, float *out)
{
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    int head_dim = pn_head_dim(cfg);
    int max_seq = kv->max_seq;
    int n_kv_groups = cfg->n_head / cfg->n_kv_head;
    int q_width = cfg->n_head * head_dim;
    int kv_width = cfg->n_kv_head * head_dim;

    /* Biases (Qwen2) + per-head Q/K RMSNorm (Qwen3) BEFORE RoPE, the CPU oracle's own code.
     * The card's projections do neither; skipping this ran Qwen3 attention on un-normed q/k
     * (garbage text). It assumes the artifact keeps the GGUF's q/k feature order, which
     * llama_tt_ensure_attention() and pn_llama_tt_upload() check. */
    pn_llama_qkv_finish(&((pn_llama_qweights *)model->weights)->layers[layer], cfg, q, k, v);

    /* THE PAIRING COMES FROM THE ARTIFACT, NOT THE ARCHITECTURE. A `--rope-pairing
     * split-half` artifact holds wq/wk with llama.cpp's HF->GGUF permutation undone, so
     * its q and k arrive in NeoX order whatever `arch` says — and reading that pairing off
     * the architecture instead would rotate the wrong feature pairs with no symptom but a
     * wrong answer. Q.K is invariant under a permutation applied to both, so the two
     * artifacts agree to the last digit that block float allows. */
    if (pn_tt_device_rope_pairs_reversed(g->device)) {
        /* The two reflected INSTRUMENTS. The two artifact bits are independent — SPLIT_HALF
         * says where a partner is, PAIRS_REVERSED says which pair a slot holds — so the
         * reflected arm still has to ask the first question, and it is tested before the
         * plain split-half branch because that predicate is true here too. */
        if (pn_tt_device_rope_is_split_half(g->device)) {
            llama_tt_rope_neox_reversed(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            llama_tt_rope_neox_reversed(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos,
                                        cfg->rope_theta);
        } else {
            llama_tt_rope_raw_reversed(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            llama_tt_rope_raw_reversed(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos,
                                       cfg->rope_theta);
        }
    } else if (cfg->arch == PN_ARCH_QWEN3 || pn_tt_device_rope_is_split_half(g->device)) {
        if (g->emulate_bf16_table) {
            llama_tt_rope_neox_bf16_table(q, cfg->n_head, head_dim, cfg->rope_dim, pos,
                                          cfg->rope_theta);
            llama_tt_rope_neox_bf16_table(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos,
                                          cfg->rope_theta);
        } else {
            llama_tt_rope_neox(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            llama_tt_rope_neox(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos,
                               cfg->rope_theta);
        }
    } else {
        llama_tt_rope_raw(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        llama_tt_rope_raw(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
    }

    /* The card's q strip is bf16 and its KV cache is bf16. Each is applied on its own so the
     * gate prices Q's rounding and the cache's rounding separately — they are different
     * quantities: Q's moves this token's query, the cache's moves EVERY position it will ever
     * be read at. */
    if (g->emulate_bf16_rot)
        llama_tt_round_to_bf16(q, q_width);
    if (g->emulate_bf16_kv) {
        llama_tt_round_to_bf16(k, kv_width);
        llama_tt_round_to_bf16(v, kv_width);
    }

    float *k_cache = (float *)kv->k_cache[layer]->data;
    float *v_cache = (float *)kv->v_cache[layer]->data;
    int cache_pos = pos % max_seq;
    for (int h = 0; h < cfg->n_kv_head; h++) {
        memcpy(k_cache + ((size_t)h * max_seq + cache_pos) * head_dim,
               k + (size_t)h * head_dim, head_dim * sizeof(float));
        memcpy(v_cache + ((size_t)h * max_seq + cache_pos) * head_dim,
               v + (size_t)h * head_dim, head_dim * sizeof(float));
    }

    int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
    if (g->emulate_bf16_q_scale || g->emulate_bf16_scores)
        llama_tt_attn_heads_bf16_softmax(q, k_cache, v_cache, out, g->scores,
                                         cfg->n_head, n_kv_groups, head_dim, max_seq, seq_len,
                                         1.0f / sqrtf((float)head_dim),
                                         g->emulate_bf16_q_scale, g->emulate_bf16_scores);
    else
        llama_tt_attn_heads(q, k_cache, v_cache, out, g->scores, cfg->n_head,
                            n_kv_groups, head_dim, max_seq, seq_len,
                            1.0f / sqrtf((float)head_dim));
    return 0;
}

/**
 * RoPE, the KV append and the GQA softmax ON THE HOST — the shipping path, unchanged.
 *
 * It leaves `g->attn` holding the attention output and does NOT upload it, so the crossed
 * control can run this and the card arm over the same q/k/v and compare.
 */
static int llama_tt_attention_host(pn_llama_tt *g, pn_model *model, pn_state *state, int layer,
                                   int pos)
{
    pn_llama_tt_strips *s = &g->card;
    pn_config *cfg = &model->config;
    int q_width = cfg->n_head * pn_head_dim(cfg);
    int kv_width = cfg->n_kv_head * pn_head_dim(cfg);

    /* THE READBACK THAT IS LEFT when attention is on the host. q, k and v come back because
     * the KV append and the GQA softmax are host-side; the rotary pairing no longer decides
     * it. 10,240 B a layer, 245,760 a token, 52 ms at 4.7 MB/s — the whole readback phase. */
    if (pn_tt_activation_download(g->device, &s->q, g->q, (uint32_t)q_width) != 0 ||
        pn_tt_activation_download(g->device, &s->k, g->k, (uint32_t)kv_width) != 0 ||
        pn_tt_activation_download(g->device, &s->v, g->v, (uint32_t)kv_width) != 0)
        return -1;
    return llama_tt_attend_one(g, model, state, layer, pos, g->q, g->k, g->v, g->attn);
}

/**
 * The same three ops ON THE CARD, and NOTHING crosses the link.
 *
 * q, k and v stay in DRAM strips; RoPE rotates q and k into their own strips; the append
 * places k and v into this layer's card-resident cache; flash decode reads that cache and
 * writes `attn` — which is where `wo` already reads from.
 */
static int llama_tt_attention_card(pn_llama_tt *g, pn_model *model, pn_state *state, int layer,
                                   int pos)
{
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_llama_tt_strips *s = &g->card;
    pn_config *cfg = &model->config;
    int max_seq = kv->max_seq;
    int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;

    if (pn_tt_rope(g->device, &s->q, &s->q_rot, (uint32_t)cfg->n_head, (uint32_t)pos) != 0 ||
        pn_tt_rope(g->device, &s->k, &s->k_rot, (uint32_t)cfg->n_kv_head, (uint32_t)pos) != 0)
        return -1;
    if (pn_tt_kv_append(g->device, (uint32_t)layer, false, &s->k_rot, (uint32_t)pos) != 0 ||
        pn_tt_kv_append(g->device, (uint32_t)layer, true, &s->v, (uint32_t)pos) != 0)
        return -1;
    return pn_tt_attention(g->device, (uint32_t)layer, &s->q_rot, &s->attn,
                           (uint32_t)seq_len);
}

/**
 * THE CROSSED CONTROL. Run BOTH arms over the same q/k/v and compare the two attention
 * outputs element for element.
 *
 * This is not a PCC check and deliberately not one: the failures this op can have are a
 * wrong KV position, a group's query reading another group's head, a mask tail off by one and
 * a face-order mistake — every one of which is a PERMUTATION or a localised wrong element,
 * and PCC is blind to the first and saturates on the second (foundry/tt/DEV_NOTES.md §3). So
 * it reports max|Δ|, WHERE, and how many elements are past a threshold, which is the
 * resolution "one element of one head is wrong" needs.
 *
 * It costs the readback it exists to remove — both arms, plus a download of the card's
 * answer — so it is a knob, exactly like PN_TT_TRACE.
 */
static void llama_tt_attention_compare(pn_llama_tt *g, const pn_config *cfg, int layer)
{
    int q_width = cfg->n_head * pn_head_dim(cfg);
    double worst = 0.0;
    int worst_index = 0, diverged = 0;

    if (pn_tt_activation_download(g->device, &g->card.attn, g->attn_card,
                                  (uint32_t)q_width) != 0) {
        fprintf(stderr, "llama_tt attn-check: L%02d the card's attention output did not read "
                        "back\n", layer);
        return;
    }
    for (int i = 0; i < q_width; i++) {
        double delta = fabs((double)g->attn_card[i] - (double)g->attn[i]);

        if (delta > worst) { worst = delta; worst_index = i; }
        if (delta > LLAMA_TT_ATTN_DIVERGENCE) diverged++;
    }
    if (worst > g->attn_check_worst) {
        g->attn_check_worst = worst;
        g->attn_check_worst_layer = layer;
    }
    g->attn_check_diverged += diverged;
    fprintf(stderr, "llama_tt attn-check: L%02d max|d|=%.6f at %d (head %d dim %d) card=%.6f "
                    "host=%.6f, %d of %d past %.3f\n",
            layer, worst, worst_index, worst_index / pn_head_dim(cfg),
            worst_index % pn_head_dim(cfg), (double)g->attn_card[worst_index],
            (double)g->attn[worst_index], diverged, q_width, LLAMA_TT_ATTN_DIVERGENCE);
}

/**
 * THE EXPORT PATH, CHECKED ON SILICON — the card's KV cache read back and compared against
 * the host's, position for position.
 *
 * Session export downloads the card cache (llama_state_snapshot.c) and nothing else in the
 * forward ever reads it back, so without this the only proof that `pn_tt_kv_download()`
 * addresses the right tiles is off-hardware arithmetic. Under `PN_TT_ATTN_CHECK` BOTH caches
 * are live and current — the host arm appends its fp32 rows and the card arm appends its bf16
 * ones from the same k/v — so the host cache is a real reference for the card's.
 *
 * It is bf16 against fp32, so the bar is a bf16 rounding and not equality; what it actually
 * catches is a wrong tile, a wrong head stride, a transposed face map or an off-by-one
 * position, every one of which is a whole-value error rather than a rounding one.
 */
static void llama_tt_kv_export_check(pn_llama_tt *g, pn_model *model, pn_state *state,
                                     int layer, int pos)
{
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    int head_dim = pn_head_dim(cfg);
    int count = pos + 1;
    size_t elems = (size_t)cfg->n_kv_head * (size_t)count * (size_t)head_dim;
    float *card_k = malloc(elems * sizeof(float));
    float *card_v = malloc(elems * sizeof(float));
    double worst = 0.0;
    int worst_index = 0, which = 0;

    if (!card_k || !card_v) { free(card_k); free(card_v); return; }
    if (pn_tt_kv_download(g->device, (uint32_t)layer, false, (uint32_t)count, card_k) != 0 ||
        pn_tt_kv_download(g->device, (uint32_t)layer, true, (uint32_t)count, card_v) != 0) {
        fprintf(stderr, "llama_tt kv-export: L%02d the card KV did not read back\n", layer);
        free(card_k); free(card_v);
        return;
    }
    for (int kvside = 0; kvside < 2; kvside++) {
        const float *card = kvside ? card_v : card_k;
        const float *live = (const float *)(kvside ? kv->v_cache[layer]->data
                                                   : kv->k_cache[layer]->data);

        for (int h = 0; h < cfg->n_kv_head; h++) {
            for (int p = 0; p < count; p++) {
                for (int e = 0; e < head_dim; e++) {
                    size_t ci = ((size_t)h * count + p) * head_dim + e;
                    size_t li = ((size_t)h * kv->max_seq + p) * head_dim + e;
                    double delta = fabs((double)card[ci] - (double)live[li]);

                    if (delta > worst) { worst = delta; worst_index = (int)ci; which = kvside; }
                }
            }
        }
    }
    if (worst > g->kv_export_worst) {
        g->kv_export_worst = worst;
        g->kv_export_worst_layer = layer;
    }
    fprintf(stderr, "llama_tt kv-export: L%02d %d positions, max|d|=%.6f in %s at %d\n", layer,
            count, worst, which ? "V" : "K", worst_index);
    free(card_k);
    free(card_v);
}

/** An angle folded into (-pi, pi], so two angles can be compared without a branch per wrap. */
static double llama_tt_wrap_angle(double angle)
{
    double wrapped = fmod(angle + LLAMA_TT_PI, 2.0 * LLAMA_TT_PI);

    if (wrapped < 0.0)
        wrapped += 2.0 * LLAMA_TT_PI;
    return wrapped - LLAMA_TT_PI;
}

/**
 * Is this float an infinity or a NaN, read off its BITS?
 *
 * `isfinite()` cannot be used here: the build carries `-ffast-math`, which implies
 * `-ffinite-math-only`, under which the compiler is entitled to fold `isfinite(x)` to 1 — so
 * a census built on it would report zero non-finite elements whatever the card wrote. That is
 * exactly the vacuous-check shape this project keeps a catalogue of.
 */
static int llama_tt_float_is_non_finite(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return ((bits >> 23) & 0xFFu) == 0xFFu;
}

/** The angle a NeoX rotary applies to pair index `p` at `pos`, in the model's own terms. */
static double llama_tt_expected_rope_angle(int p, int rot, int pos, double theta_base)
{
    return llama_tt_wrap_angle((double)pos * pow(theta_base, -2.0 * (double)p / (double)rot));
}

/**
 * Recover the card's rotary angle per feature slot and report which pair index it belongs to.
 *
 * Reads the pre-RoPE strip and the post-RoPE strip off the card — neither is the host's, so
 * this measures the card alone and does not need the host arm to be running. See the
 * PN_TT_ROPE_CHECK block at the top of this file for the inversion.
 *
 * For every slot it scores the recovered angle against EVERY candidate pair index, so the
 * output is not "does it match what we expect" but "which frequency did it actually apply".
 * Slots whose identity and reflected candidates are closer together than the bf16 table's own
 * error are counted as degenerate rather than voted, because at those frequencies no
 * measurement could tell the two apart.
 */
static void llama_tt_rope_recover_check(pn_llama_tt *g, const pn_config *cfg, int layer, int pos)
{
    const pn_llama_tt_strips *s = &g->card;
    int head_dim = pn_head_dim(cfg);
    int q_width = cfg->n_head * head_dim;
    int rot = (cfg->rope_dim > 0 && cfg->rope_dim <= head_dim) ? cfg->rope_dim : head_dim;
    int half = rot / 2;
    long identity = 0, reflected = 0, other = 0, degenerate = 0, readable = 0;
    double worst_identity = 0.0, worst_norm_error = 0.0;

    if (pn_tt_activation_download(g->device, &s->q, g->probe_in, (uint32_t)q_width) != 0 ||
        pn_tt_activation_download(g->device, &s->q_rot, g->probe_out, (uint32_t)q_width) != 0) {
        fprintf(stderr, "llama_tt rope-check: L%02d the rotary strips did not read back\n",
                layer);
        return;
    }
    for (int h = 0; h < cfg->n_head; h++) {
        const float *x = g->probe_in + (size_t)h * head_dim;
        const float *y = g->probe_out + (size_t)h * head_dim;

        for (int j = 0; j < half; j++) {
            double x0 = x[j], x1 = x[j + half], y0 = y[j], y1 = y[j + half];
            double energy = x0 * x0 + x1 * x1;
            double cos_t, sin_t, recovered, best_residual, identity_residual, separation;
            int best_p = 0;

            if (energy < LLAMA_TT_ROPE_MIN_PAIR_ENERGY)
                continue;
            readable++;
            cos_t = (x0 * y0 + x1 * y1) / energy;
            sin_t = (x0 * y1 - x1 * y0) / energy;
            /* A rotation preserves the norm. Reported separately because a table that is not
             * a (cos, sin) pair at all fails HERE and not on the angle. */
            if (fabs(sqrt(cos_t * cos_t + sin_t * sin_t) - 1.0) > worst_norm_error)
                worst_norm_error = fabs(sqrt(cos_t * cos_t + sin_t * sin_t) - 1.0);
            recovered = atan2(sin_t, cos_t);

            identity_residual = fabs(llama_tt_wrap_angle(
                recovered - llama_tt_expected_rope_angle(j, rot, pos, cfg->rope_theta)));
            if (identity_residual > worst_identity)
                worst_identity = identity_residual;
            separation = fabs(llama_tt_wrap_angle(
                llama_tt_expected_rope_angle(j, rot, pos, cfg->rope_theta) -
                llama_tt_expected_rope_angle(half - 1 - j, rot, pos, cfg->rope_theta)));
            if (separation < LLAMA_TT_ROPE_MIN_SEPARATION) {
                degenerate++;
                continue;
            }
            best_residual = -1.0;
            for (int p = 0; p < half; p++) {
                double residual = fabs(llama_tt_wrap_angle(
                    recovered - llama_tt_expected_rope_angle(p, rot, pos, cfg->rope_theta)));

                if (best_residual < 0.0 || residual < best_residual) {
                    best_residual = residual;
                    best_p = p;
                }
            }
            if (best_p == j) {
                identity++;
            } else if (best_p == half - 1 - j) {
                reflected++;
            } else {
                int distance = best_p > j ? best_p - j : j - best_p;
                double gap = fabs(llama_tt_wrap_angle(
                    llama_tt_expected_rope_angle(j, rot, pos, cfg->rope_theta) -
                    llama_tt_expected_rope_angle(best_p, rot, pos, cfg->rope_theta)));

                other++;
                if (distance > g->rope_worst_other_distance)
                    g->rope_worst_other_distance = distance;
                if (gap > g->rope_worst_other_separation)
                    g->rope_worst_other_separation = gap;
            }
        }
    }
    g->rope_slots_identity += identity;
    g->rope_slots_reflected += reflected;
    g->rope_slots_other += other;
    g->rope_slots_degenerate += degenerate;
    if (worst_identity > g->rope_worst_identity_residual)
        g->rope_worst_identity_residual = worst_identity;
    if (worst_norm_error > g->rope_worst_norm_error)
        g->rope_worst_norm_error = worst_norm_error;
    fprintf(stderr, "llama_tt rope-check: L%02d pos=%d readable=%ld best_match identity=%ld "
                    "reflected=%ld other=%ld degenerate=%ld worst_identity_residual=%.6f rad "
                    "worst_norm_error=%.6f\n",
            layer, pos, readable, identity, reflected, other, degenerate, worst_identity,
            worst_norm_error);
}

/**
 * Read rows 1-31 of one strip off the card and report the largest magnitude in them.
 *
 * The strip invariant says they are zero. On the on-card attention arm that is INFERRED, not
 * measured (pn_tt_oncard.c invariant 1), and this is the measurement. Non-finite elements are
 * counted separately because an inf or a NaN in a row nothing reads is still an inf that a
 * future row-crossing op would consume.
 */
static void llama_tt_row_census_strip(pn_llama_tt *g, const pn_tt_activation *strip,
                                      const char *name, int width, int layer)
{
    double worst = 0.0, row0_worst = 0.0, row0_disagreement = 0.0;
    long nonfinite = 0;
    int worst_row = 0, worst_index = 0;

    /* THE CONTROL, and without it a reader that returned zeros would report the invariant
     * HOLDING. Row 0 is the row the forward actually uses, so it must come back non-zero, and
     * it must AGREE with the shipping row-0 reader — which reads different bytes by a
     * different path. A census that cannot go red is the vacuous-gate shape this project has
     * caught four times. */
    if (pn_tt_activation_download_tile_row(g->device, strip, 0u, g->probe_in,
                                           (uint32_t)width) == 0 &&
        pn_tt_activation_download(g->device, strip, g->probe_out, (uint32_t)width) == 0) {
        for (int i = 0; i < width; i++) {
            double delta = fabs((double)g->probe_in[i] - (double)g->probe_out[i]);

            if (fabs((double)g->probe_in[i]) > row0_worst)
                row0_worst = fabs((double)g->probe_in[i]);
            if (delta > row0_disagreement)
                row0_disagreement = delta;
        }
    }
    for (uint32_t row = 1; row < 32u; row++) {
        if (pn_tt_activation_download_tile_row(g->device, strip, row, g->probe_in,
                                               (uint32_t)width) != 0) {
            fprintf(stderr, "llama_tt row-census: L%02d %s row %u did not read back\n", layer,
                    name, row);
            return;
        }
        for (int i = 0; i < width; i++) {
            double value = (double)g->probe_in[i];

            if (llama_tt_float_is_non_finite(g->probe_in[i])) { nonfinite++; continue; }
            if (fabs(value) > worst) {
                worst = fabs(value);
                worst_row = (int)row;
                worst_index = i;
            }
        }
    }
    if (worst > g->row_census_worst)
        g->row_census_worst = worst;
    g->row_census_nonfinite += nonfinite;
    fprintf(stderr, "llama_tt row-census: L%02d %-7s rows 1-31 max|v|=%.6g at row %d column %d, "
                    "%ld non-finite of %d; CONTROL row 0 max|v|=%.6g, disagrees with the row-0 "
                    "reader by %.6g\n",
            layer, name, worst, worst_row, worst_index, nonfinite, 31 * width, row0_worst,
            row0_disagreement);
}

/** The attention half of one layer. Split out to keep the forward readable and to keep
 *  each half under a screen — the ordering inside it is the CUDA path's, unchanged. */
static int llama_tt_attention(pn_llama_tt *g, pn_model *model, pn_state *state, int layer,
                              int pos)
{
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_llama_qlayer *cpu = &qw->layers[layer];
    pn_llama_tt_layer *d = &g->layers[layer];
    pn_llama_tt_strips *s = &g->card;
    pn_config *cfg = &model->config;
    bool on_card = pn_tt_device_oncard(g->device, PN_TT_ONCARD_ATTENTION);
    int q_width = cfg->n_head * pn_head_dim(cfg);

    if (llama_tt_norm(g, &d->attn_norm, cpu->attn_norm, cfg) != 0)
        return -1;
    /* ONE WAIT FOR THREE PROJECTIONS. They share `normed` and write q, k and v, which are
     * three disjoint strips — so nothing here reads what this set writes and the host has no
     * reason to stop between them. The wait is the next thing that touches q/k/v, and every
     * device operation flushes on entry, so it happens whether or not anyone remembers. */
    if (llama_tt_matvec_begin(g, &d->wq, &cpu->wq, &s->normed, &s->q, layer, "attn_q") != 0 ||
        llama_tt_matvec_begin(g, &d->wk, &cpu->wk, &s->normed, &s->k, layer, "attn_k") != 0 ||
        llama_tt_matvec_begin(g, &d->wv, &cpu->wv, &s->normed, &s->v, layer, "attn_v") != 0)
        return -1;
    if (pn_tt_matvec_flush(g->device) != 0)
        return -1;

    /* The host arm runs FIRST under the check, because it is the one that consumes q/k/v as
     * they are — the card arm rotates q and k into their own strips and leaves the originals
     * untouched, so the order does not change either answer. */
    if (!on_card || g->attn_check) {
        if (llama_tt_attention_host(g, model, state, layer, pos) != 0)
            return -1;
        if (!on_card &&
            pn_tt_activation_upload(g->device, &s->attn, g->attn, (uint32_t)q_width) != 0)
            return -1;
    }
    if (on_card) {
        if (llama_tt_attention_card(g, model, state, layer, pos) != 0)
            return -1;
        /* Layer 0 only, and only past position 0. At position 0 every rotary angle is exactly
         * zero, so the recovery would be reading the identity out of the identity — a probe
         * that cannot distinguish anything is worse than no probe. */
        if (g->rope_check && layer == 0 && pos > 0)
            llama_tt_rope_recover_check(g, cfg, layer, pos);
        if (g->row_census && layer == (pos % cfg->n_layer))
            llama_tt_row_census_strip(g, &s->attn, "attn", q_width, layer);
        if (g->attn_check) {
            llama_tt_attention_compare(g, cfg, layer);
            /* ONE layer, not all 24: this reads whole tiles off the card and its cost grows
             * with the sequence, so checking every layer every token would dominate the run
             * and prove nothing the rotation does not. The layer rotates with the position so
             * a long enough run covers all of them. */
            if (layer == (pos % cfg->n_layer))
                llama_tt_kv_export_check(g, model, state, layer, pos);
        }
    }

    if (llama_tt_matvec(g, &d->wo, &cpu->wo, &s->attn, &s->project, layer, "attn_output") != 0)
        return -1;
    if (llama_tt_residual(g, cfg) != 0)
        return -1;
    /* AFTER the residual add, because the question is not what flash decode wrote but what
     * propagates: `hidden` is the strip every later layer and the LM head read. */
    if (on_card && g->row_census && layer == (pos % cfg->n_layer))
        llama_tt_row_census_strip(g, llama_tt_hidden(g), "hidden", cfg->n_embd, layer);
    return 0;
}

/** The SwiGLU half of one layer. Nothing here crosses the link when the gate and the
 *  residual add are on-card: normed, gate, up and the product are all DRAM strips. */
static int llama_tt_ffn(pn_llama_tt *g, pn_model *model, int layer)
{
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_llama_qlayer *cpu = &qw->layers[layer];
    pn_llama_tt_layer *d = &g->layers[layer];
    pn_llama_tt_strips *s = &g->card;
    pn_config *cfg = &model->config;

    if (llama_tt_norm(g, &d->ffn_norm, cpu->ffn_norm, cfg) != 0)
        return -1;
    /* ONE WAIT FOR BOTH HALVES OF THE GATE — same argument as q/k/v: one input, two
     * disjoint outputs, and the SwiGLU that consumes them flushes on entry. */
    if (llama_tt_matvec_begin(g, &d->w_gate, &cpu->w_gate, &s->normed, &s->gate, layer,
                              "ffn_gate") != 0 ||
        llama_tt_matvec_begin(g, &d->w_up, &cpu->w_up, &s->normed, &s->up, layer, "ffn_up") != 0)
        return -1;
    if (pn_tt_matvec_flush(g->device) != 0)
        return -1;

    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_SWIGLU)) {
        if (pn_tt_silu_mul(g->device, &s->gate, &s->up, &s->swiglu) != 0)
            return -1;
    } else {
        if (pn_tt_activation_download(g->device, &s->gate, g->gate, (uint32_t)cfg->n_ff) != 0 ||
            pn_tt_activation_download(g->device, &s->up, g->up, (uint32_t)cfg->n_ff) != 0)
            return -1;
        pn_vsilu_mul_f32(g->gate, g->up, g->gate, cfg->n_ff);
        if (pn_tt_activation_upload(g->device, &s->swiglu, g->gate, (uint32_t)cfg->n_ff) != 0)
            return -1;
    }

    if (llama_tt_matvec(g, &d->w_down, &cpu->w_down, &s->swiglu, &s->project, layer,
                        "ffn_down") != 0)
        return -1;
    return llama_tt_residual(g, cfg);
}

/**
 * The LM head, and the only place the vocabulary is involved.
 *
 * Two arms, and the difference is 261,120 bytes a token. With the on-card top-N the card
 * hands back eight (index, value) pairs and nothing else; `PN_TT_SPARSE_LOGITS` presents
 * exactly those, scattered into an all -INFINITY row, which is the shape the K80 path
 * already ships. Without it the whole row is read so the parity harness can score it — and
 * when both are true, the card's index is checked against the host's serial argmax over the
 * row it did NOT see, which is the gate this op is accepted on.
 */
static int llama_tt_head(pn_llama_tt *g, pn_model *model, float *logits_raw)
{
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_config *cfg = &model->config;
    bool have_topk = false;

    if (llama_tt_norm(g, &g->output_norm, qw->output_norm, cfg) != 0)
        return -1;
    /* The vector the LM head actually reads. It is the last row of the residual trace
     * because a divergence here is the one that reaches the logits directly. */
    if (pn_residual_trace_armed()) {
        if (pn_tt_activation_download(g->device, &g->card.normed, g->normed,
                                      (uint32_t)cfg->n_embd) != 0)
            return -1;
        pn_residual_trace_record(cfg->n_layer, g->normed, cfg->n_embd);
    }
    if (llama_tt_matvec(g, &g->output, &qw->output, &g->card.normed, &g->card.logits, -1,
                        "lm_head") != 0)
        return -1;

    g->topk_count = 0u;
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_TOPK)) {
        if (pn_tt_topk(g->device, &g->card.logits, PN_TT_TOPK_PAIRS, g->topk_index,
                       g->topk_value, &g->topk_count) != 0)
            return -1;
        have_topk = g->topk_count > 0u;
    }

    if (g->sparse_logits && have_topk) {
        for (int i = 0; i < cfg->n_vocab; i++)
            logits_raw[i] = -INFINITY;
        for (uint32_t i = 0; i < g->topk_count; i++) {
            if (g->topk_index[i] < (uint32_t)cfg->n_vocab)
                logits_raw[g->topk_index[i]] = g->topk_value[i];
        }
        return 0;
    }

    if (pn_tt_activation_download(g->device, &g->card.logits, logits_raw,
                                  (uint32_t)cfg->n_vocab) != 0)
        return -1;
    if (have_topk) {
        int argmax = 0;

        for (int i = 1; i < cfg->n_vocab; i++)
            if (logits_raw[i] > logits_raw[argmax]) argmax = i;
        fprintf(stderr, "llama_tt: on-card top-1 index %u vs host argmax %d — %s\n",
                g->topk_index[0], argmax,
                g->topk_index[0] == (uint32_t)argmax ? "MATCH" : "MISMATCH");
    }
    return 0;
}

/** The per-token cost and the run total, split by the machine that paid each. */
static void llama_tt_report(pn_llama_tt *g)
{
    double upload = 0.0, dispatch = 0.0, readback = 0.0, oncard = 0.0;
    uint64_t reg_r = 0, reg_w = 0, bulk_r = 0, bulk_w = 0;
    uint64_t resident = 0, oneshot = 0, restarts = 0, oncard_ops = 0, oncard_restarts = 0;
    uint64_t matvecs = pn_tt_device_matvec_count(g->device);

    pn_tt_device_phase_seconds(g->device, &upload, &dispatch, &readback);
    pn_tt_device_oncard_stats(g->device, &oncard, &oncard_ops, &oncard_restarts);
    /* The TOTAL is the sum of the four phases, not `matvec_seconds`: an activation that
     * crosses the link now does so OUTSIDE a matvec — a strip is uploaded once and read back
     * once, not per projection — so a total taken from the matvec timer would omit exactly
     * the transfers this file exists to remove and would improve on its own. */
    /* THIS TOKEN, then the run so far. The device's counters are cumulative and monotonic;
     * printing only those under "token done" is what let a 0.437 s token be read as a
     * 16.789 s one. The per-token line is first because it is the one anyone quotes. */
    fprintf(stderr, "llama_tt: THIS TOKEN — %llu matmuls, %.3f s "
                    "(upload %.3f + dispatch/compute %.3f + readback %.3f + on-card %.3f)\n",
            (unsigned long long)(matvecs - g->last_matvecs),
            (upload - g->last_upload) + (dispatch - g->last_dispatch) +
                (readback - g->last_readback) + (oncard - g->last_oncard),
            upload - g->last_upload, dispatch - g->last_dispatch, readback - g->last_readback,
            oncard - g->last_oncard);
    fprintf(stderr, "llama_tt: RUN TOTAL — %llu matmuls on the card, %.3f s total "
                    "(upload %.3f + dispatch/compute %.3f + readback %.3f + on-card %.3f "
                    "over %llu ops), %.2f MiB read back, %.2f MiB written\n",
            (unsigned long long)matvecs,
            upload + dispatch + readback + oncard, upload, dispatch, readback, oncard,
            (unsigned long long)oncard_ops,
            (double)pn_tt_device_bytes_read(g->device) / (1024.0 * 1024.0),
            (double)pn_tt_device_bytes_written(g->device) / (1024.0 * 1024.0));
    g->last_upload = upload;
    g->last_dispatch = dispatch;
    g->last_readback = readback;
    g->last_oncard = oncard;
    g->last_matvecs = matvecs;
    /* Which WINDOW carried those bytes. A register access on the bulk path, or bulk data
     * still on the register path, both look like a working token from here and only these
     * counters tell them apart. */
    pn_tt_device_window_traffic(g->device, &reg_r, &reg_w, &bulk_r, &bulk_w);
    {
        uint64_t reg_aims = 0, bulk_aims = 0, aim_failures = 0;

        /* The aim counts ride the same line because they answer the question the byte
         * counts cannot: the two transports move IDENTICAL bytes and differ only in what
         * an aim costs, so a wall-clock gap between them is aims x per-aim cost and there
         * is no way to price it without this number. */
        pn_tt_device_window_aims(g->device, &reg_aims, &bulk_aims, &aim_failures);
        fprintf(stderr, "llama_tt: windows — register(UC/word32) read %llu B write %llu B "
                        "%llu aims, bulk(WC/block16) read %llu B write %llu B %llu aims, "
                        "%llu aim failures\n",
                (unsigned long long)reg_r, (unsigned long long)reg_w,
                (unsigned long long)reg_aims, (unsigned long long)bulk_r,
                (unsigned long long)bulk_w, (unsigned long long)bulk_aims,
                (unsigned long long)aim_failures);
    }
    /* WHICH dispatch path carried the ops. Both produce identical numbers, so a silent
     * fallback to one-shot looks exactly like a resident run from the outside. */
    pn_tt_device_dispatch_counts(g->device, &resident, &oneshot, &restarts);
    /* HOW MANY PROJECTIONS ACTUALLY WENT WIDE. A group whose member count does not divide a
     * weight's Nt leaves that weight on one core, so a run can be configured for N cores and
     * still spend most of its seconds on one — which reads from outside as "multi-core did
     * not help" and is a different statement entirely. */
    {
        uint64_t split_ops = 0, solo_ops = 0;

        pn_tt_matmul_group_op_counts(g->device, &split_ops, &solo_ops);
        if (split_ops + solo_ops > 0)
            fprintf(stderr, "llama_tt: group — %llu projections split across the group, "
                            "%llu on one core\n",
                    (unsigned long long)split_ops, (unsigned long long)solo_ops);
    }
    /* WHAT THE DISPATCH PHASE IS MADE OF. `dispatch/compute` above is one number for two
     * very different things, and which of them grows with the member count is the whole
     * question a multi-core round has to answer. */
    pn_tt_matmul_group_time_report(g->device);
    fprintf(stderr, "llama_tt: dispatch — %llu resident, %llu one-shot, %llu matmul "
                    "residency restarts, %llu on-card residency restarts; "
                    "on-card norm %d swiglu %d residual %d topk %d attention %d\n",
            (unsigned long long)resident, (unsigned long long)oneshot,
            (unsigned long long)restarts, (unsigned long long)oncard_restarts,
            (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM),
            (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_SWIGLU),
            (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_RESIDUAL),
            (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_TOPK),
            (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_ATTENTION));
    /* The crossed control's verdict for the WHOLE run, not per layer: a single worst element
     * over every layer and every token is the number that says whether the card's attention
     * ever left the rounding band, and the count says how wide the excursion was. */
    if (g->attn_check)
        fprintf(stderr, "llama_tt attn-check: RUN WORST max|d|=%.6f at layer %d, %ld elements "
                        "past %.3f; KV EXPORT worst max|d|=%.6f at layer %d\n",
                g->attn_check_worst, g->attn_check_worst_layer, g->attn_check_diverged,
                LLAMA_TT_ATTN_DIVERGENCE, g->kv_export_worst, g->kv_export_worst_layer);
    /* The rotary verdict for the WHOLE run. `identity` is the artifact's own pairing; every
     * count in `reflected` or `other` is a feature the card rotated at another frequency. */
    if (g->rope_check)
        fprintf(stderr, "llama_tt rope-check: RUN TOTAL best_match identity=%ld reflected=%ld "
                        "other=%ld (degenerate, not voted=%ld); worst identity residual %.6f "
                        "rad, worst norm error %.6f; worst 'other' was %d index away with the "
                        "two candidates %.6f rad apart\n",
                g->rope_slots_identity, g->rope_slots_reflected, g->rope_slots_other,
                g->rope_slots_degenerate, g->rope_worst_identity_residual,
                g->rope_worst_norm_error, g->rope_worst_other_distance,
                g->rope_worst_other_separation);
    if (g->row_census)
        fprintf(stderr, "llama_tt row-census: RUN WORST |v| in rows 1-31 = %.6g, %ld non-finite "
                        "elements\n", g->row_census_worst, g->row_census_nonfinite);
}

tm_tensor *pn_llama_forward_tt(pn_model *model, uint32_t token, pn_state *state)
{
    if (!model || !state || !model->weights || !model->gpu)
        return NULL;
    pn_llama_tt *g = (pn_llama_tt *)model->gpu;
    if (!g->loaded)
        return NULL;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    int pos = state->pos;

    if (!kv || llama_tt_ensure_scores(g, kv->max_seq) != 0)
        return NULL;
    if (llama_tt_ensure_attention(g, model, kv->max_seq) != 0)
        return NULL;
    /* The card holds ONE KV region set. The host cache is the sequence's identity, and a
     * cache that comes back after being displaced is two sequences alternating on it. */
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_ATTENTION)) {
        if (pn_tt_kv_bind(g->device, kv) != 0)
            return NULL;
        /* SAY WHERE THE STATE IS, so session export cannot read the wrong place. On this arm
         * the host tensors are never written; llama_state_snapshot.c calls this hook and
         * moves the rows to and from the card instead. Set HERE and not at allocation,
         * because the forward is the only code that knows which arm is live — and under
         * PN_TT_ATTN_CHECK both caches ARE current, so the card is still the one to trust:
         * it is what the next token will actually attend to. */
        kv->kv_device = g->device;
        kv->kv_move = llama_tt_kv_move;
    }

    /* Token embedding, on the host: the artifact stores the table tiled and block-float,
     * which the on-card gather cannot read (pn_tt_device.h). The engine's own fp32 copy is
     * the same values and is already resident. It costs one 64-byte-a-tile UPLOAD, in the
     * direction the link is ~60x faster. */
    if (state->embd_override)
        memcpy(g->hidden, state->embd_override, (size_t)cfg->n_embd * sizeof(float));
    else
        memcpy(g->hidden, qw->tok_embd + (size_t)token * cfg->n_embd,
               (size_t)cfg->n_embd * sizeof(float));
    g->card.slot = 0;
    g->hidden_on_host = true;
    g->hidden_on_card = false;

    for (int layer = 0; layer < cfg->n_layer; layer++) {
        if (llama_tt_attention(g, model, state, layer, pos) != 0)
            return NULL;
        if (llama_tt_ffn(g, model, layer) != 0)
            return NULL;
        /* The card's residual stream at the same point the CPU oracle records its own. The
         * download is the instrument's cost and changes no arithmetic: the card copy stays
         * authoritative, so the next layer's norm still reads what it would have. */
        if (pn_residual_trace_armed()) {
            if (llama_tt_hidden_to_host(g, cfg->n_embd) != 0)
                return NULL;
            pn_residual_trace_record(layer, g->hidden, cfg->n_embd);
        }
        if (g->trace > 1) {
            double norm = 0.0;

            if (llama_tt_hidden_to_host(g, cfg->n_embd) != 0)
                return NULL;
            for (int i = 0; i < cfg->n_embd; i++)
                norm += (double)g->hidden[i] * (double)g->hidden[i];
            fprintf(stderr, "llama_tt trace: L%02d hidden |x|=%.6f\n", layer, sqrt(norm));
        }
    }

    float *logits_raw = malloc((size_t)cfg->n_vocab * sizeof(float));
    if (!logits_raw)
        return NULL;
    if (llama_tt_head(g, model, logits_raw) != 0) {
        free(logits_raw);
        return NULL;
    }

    int64_t logits_shape[] = { cfg->n_vocab };
    tm_tensor *logits_t = tm_from_data(logits_raw, logits_shape, 1, TM_F32);
    tm_tensor *logits = logits_t ? tm_clone(logits_t) : NULL;
    if (logits_t) tm_free(logits_t);
    free(logits_raw);

    llama_tt_report(g);
    state->pos++;
    return logits;
}


/* ============================================================
 * BATCHED PREFILL — the tile quantum IS the batch dimension
 *
 * ---------------------------------------------------------------------------
 * WHAT WAS WRONG, AND IT WAS NOT THE MATMUL
 * ---------------------------------------------------------------------------
 * A prompt has every one of its tokens in hand at once, so every weight tile read can serve
 * ALL of them. We were reading the model once per prompt token: a 166-token prompt read a
 * 1.15 GB artifact 166 times, ~190 GB, and measured TTFT 39.25 s. The matmul counter proved
 * it rather than the timing implying it — a 25-token prompt plus 24 decode tokens moved the
 * RUN TOTAL by exactly 8,281 = 49 x 169 forwards.
 *
 * ---------------------------------------------------------------------------
 * THE CARD WAS ALREADY DOING IT AND WE WERE THROWING 31 OF 32 ROWS AWAY
 * ---------------------------------------------------------------------------
 * A tile is 32x32 and a matvec is Mt=1, so the FPU multiplies 32 rows against every weight
 * tile it reads and 31 of them are the zero padding pn_tt_activation_alloc staged. The
 * reader floor — 81 core cycles per tile-pair, the AI clock — is a cost of READING a weight
 * tile and is paid per tile, not per row. So M prompt positions in one tile cost the card
 * EXACTLY what one position costs: same descriptor, same Mt, same Kt, same Nt, same bytes.
 *
 * Everything downstream was already 32-row shaped and nobody had to be told:
 *   - `pn_tt_rms_norm` submits `row_blocks = 1`, and one row block IS 32 rows, each reduced
 *     over its own row (resident_reduce_compute.c);
 *   - the elementwise family (SwiGLU, the residual adds) is per-TILE, so all 32 rows already;
 *   - the matmul is per-row in M by construction: C[m][n] sums over k and reads only row m
 *     of A, so a row cannot see another row.
 * The ONLY thing standing between the card and a batch was that nothing could put more than
 * one row across the link. pn_tt_activation_{upload,download}_rows is that, and this is its
 * caller.
 *
 * ---------------------------------------------------------------------------
 * WHY 32 AND NOT 32 x SOMETHING — THE L1 QUESTION, ANSWERED RATHER THAN ASSUMED
 * ---------------------------------------------------------------------------
 * 32 is the TILE QUANTUM, not a budget: it costs zero extra L1 (Mt does not appear in the
 * matmul's CB sizing, `2*depth*(in0_tile + in1_tile) + 2*out_tile <= 1,351,680`) and zero
 * extra DRAM (the strips are already 32 rows tall — that is what was being wasted).
 *
 * Going PAST 32 means Mt >= 2, and with the resident reader as it stands that buys no
 * bandwidth at all. Its loop is `for mt { for nt { for kt } }`, so B is re-read once per
 * tile-row and A once per output column: both traffics scale linearly in Mt and the bytes
 * PER TOKEN are flat. Mt > 1 would amortise the descriptor and nothing else. A blocked
 * reader that held Mt A tile-rows in L1 and streamed B past all of them WOULD, and that is
 * where L1 becomes the binding constraint: 2*depth*(Mt*2048 + 1088) + 2*Mt*2048 <= 1,351,680
 * gives Mt <= 6 at this model's depth=48 (192 rows) and Mt <= 164 at depth=1. That is a
 * kernel round, not this one.
 * ============================================================ */

/** Rows of one batched pass. 32 is the tile and the default; the env exists so a sweep can
 *  A/B the batch against itself, and 1 turns the batched path into the sequential one
 *  through the same code — which is what makes the byte comparison below a control. */
#define PN_TT_PREFILL_ROWS_ENV "PN_TT_PREFILL_ROWS"

/** `PN_TT_PREFILL=0` puts the whole prompt back on the one-at-a-time path, in ONE binary. */
#define PN_TT_PREFILL_ENV "PN_TT_PREFILL"

/**
 * THE BREAK-EVEN, AND IT IS A DIVISION RATHER THAN A TASTE.
 *
 * A batched pass reads WHOLE tiles: 2048 B per tile however many of the 32 rows carry a
 * token. The row-0 reader costs 128 B per tile per row (2 x 64 B, the card->host alignment).
 * So a pass of M rows costs 2048/M bytes a row against 128, and the two are equal at
 * **M = 16**. Below that the batch moves MORE bytes per token than the path it replaces, and
 * since prefill is now readback-bound that is a straight loss — measured, not feared: a pass
 * of 8 rows reads 0.94x of the sequential rate and a pass of 3 reads 0.43x.
 *
 * So a prompt shorter than this, and the ragged tail of any prompt, go through the ordinary
 * one-at-a-time forward. `PN_TT_PREFILL_MIN_ROWS` is the A/B knob; 0 disables the guard so
 * the loss is reproducible on demand.
 */
#define PN_TT_PREFILL_MIN_ROWS_ENV "PN_TT_PREFILL_MIN_ROWS"

static int llama_tt_prefill_min_rows(void)
{
    const char *env = getenv(PN_TT_PREFILL_MIN_ROWS_ENV);
    int min_rows = env ? atoi(env) : 16;

    if (min_rows < 1) min_rows = 1;
    if (min_rows > 32) min_rows = 32;
    return min_rows;
}

static int llama_tt_prefill_rows(void)
{
    const char *env = getenv(PN_TT_PREFILL_ROWS_ENV);
    int rows = env ? atoi(env) : 32;

    if (rows < 1) rows = 1;
    if (rows > 32) rows = 32;
    return rows;
}

static bool llama_tt_prefill_enabled(void)
{
    const char *env = getenv(PN_TT_PREFILL_ENV);

    return !(env && env[0] == '0');
}

/**
 * THE RED PROOFS, and the gate is not believed until each of them goes red.
 *
 * Same knob and same shape as `pn_tt_red` in the device layer, spelled here so the model
 * file does not have to reach into pn_tt_device_internal.h. Each mutation is HOST-SIDE
 * arithmetic only — no transfer changes length, so none of them can reach the zero-length
 * NoC hazard a shortening mutation would (DEV_NOTES.md §1).
 *
 *   prefill_pos      every row rotates and appends at the FIRST position of the pass, so a
 *                    batch of 32 writes 32 K/V entries all built for base_pos. This is what
 *                    "the batch forgot which row is which token" looks like.
 *   prefill_causal   the per-row loop runs in REVERSE, so row r reads cache entries later
 *                    rows have already written. The mask in this design IS the loop order,
 *                    so this is the mutation that proves the order is load-bearing.
 *   prefill_rowfill  q/k/v come back and rows 1..M-1 are overwritten with row 0 — exactly
 *                    the bug of a transfer that still only moves logical row 0, which is
 *                    the state this whole path replaces.
 */
static bool llama_tt_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

/** The batch host scratch, allocated the first time a batched prefill runs. */
static int llama_tt_alloc_batch(pn_llama_tt *g, const pn_config *cfg, int rows)
{
    int n_embd = cfg->n_embd;
    int q_width = cfg->n_head * pn_head_dim(cfg);
    int kv_width = cfg->n_kv_head * pn_head_dim(cfg);
    int widest = cfg->n_ff > n_embd ? cfg->n_ff : n_embd;

    if (g->b_rows >= rows)
        return 0;
    free(g->b_hidden); free(g->b_normed); free(g->b_q); free(g->b_k); free(g->b_v);
    free(g->b_attn); free(g->b_gate); free(g->b_up); free(g->b_scratch);
    g->b_rows = 0;
    g->b_hidden = calloc((size_t)rows * n_embd, sizeof(float));
    g->b_normed = calloc((size_t)rows * n_embd, sizeof(float));
    g->b_q = calloc((size_t)rows * q_width, sizeof(float));
    g->b_k = calloc((size_t)rows * kv_width, sizeof(float));
    g->b_v = calloc((size_t)rows * kv_width, sizeof(float));
    g->b_attn = calloc((size_t)rows * q_width, sizeof(float));
    g->b_gate = calloc((size_t)rows * cfg->n_ff, sizeof(float));
    g->b_up = calloc((size_t)rows * cfg->n_ff, sizeof(float));
    g->b_scratch = calloc((size_t)rows * widest, sizeof(float));
    if (!g->b_hidden || !g->b_normed || !g->b_q || !g->b_k || !g->b_v || !g->b_attn ||
        !g->b_gate || !g->b_up || !g->b_scratch)
        return -1;
    g->b_rows = rows;
    return 0;
}

/** One RMS norm over `rows` rows. On card it is the SAME descriptor the single-token path
 *  submits — `row_blocks = 1` already means 32 rows — so this branch is not a batched
 *  variant of the op, it is the op with its rows used. */
static int llama_tt_norm_rows(pn_llama_tt *g, const pn_tt_norm_constants *constants,
                              const float *host_gamma, const pn_config *cfg,
                              const pn_tt_activation *in, const pn_tt_activation *out,
                              int rows)
{
    int n_embd = cfg->n_embd;

    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM))
        return pn_tt_rms_norm(g->device, constants, in, out);
    if (pn_tt_activation_download_rows(g->device, in, g->b_scratch, (uint32_t)n_embd,
                                       (uint32_t)rows) != 0)
        return -1;
    for (int r = 0; r < rows; r++)
        pn_vrms_norm_f32(g->b_scratch + (size_t)r * n_embd, host_gamma,
                         g->b_normed + (size_t)r * n_embd, n_embd, cfg->rms_norm_eps);
    return pn_tt_activation_upload_rows(g->device, out, g->b_normed, (uint32_t)n_embd,
                                        (uint32_t)rows);
}

/** hidden += project, over `rows` rows, flipping the live slot exactly as the single-token
 *  residual does. The card arm is per-TILE, so it needed nothing. */
static int llama_tt_residual_rows(pn_llama_tt *g, const pn_config *cfg, int rows)
{
    pn_llama_tt_strips *s = &g->card;
    int n_embd = cfg->n_embd;

    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_RESIDUAL)) {
        if (pn_tt_add(g->device, llama_tt_hidden(g), &s->project, llama_tt_hidden_spare(g)) != 0)
            return -1;
        s->slot ^= 1;
        return 0;
    }
    if (pn_tt_activation_download_rows(g->device, &s->project, g->b_scratch,
                                       (uint32_t)n_embd, (uint32_t)rows) != 0)
        return -1;
    if (pn_tt_activation_download_rows(g->device, llama_tt_hidden(g), g->b_hidden,
                                       (uint32_t)n_embd, (uint32_t)rows) != 0)
        return -1;
    for (int r = 0; r < rows; r++)
        pn_vadd_f32(g->b_hidden + (size_t)r * n_embd, g->b_scratch + (size_t)r * n_embd,
                    g->b_hidden + (size_t)r * n_embd, n_embd);
    return pn_tt_activation_upload_rows(g->device, llama_tt_hidden(g), g->b_hidden,
                                        (uint32_t)n_embd, (uint32_t)rows);
}

/**
 * ONE batched pass: `rows` consecutive prompt positions starting at `base_pos`, through the
 * whole layer stack, leaving their K/V in the cache and NOT computing any logits.
 *
 * No LM head runs here at all. The caller hands the LAST prompt token to the ordinary
 * single-token forward, which is what produces the logits — so the head, the top-N and the
 * sparse-logit presentation are literally the shipping code and cannot diverge. It also
 * makes the acceptance argument short: if the K/V this writes is byte-identical to what the
 * sequential path writes, the logits are byte-identical because they come out of the same
 * function reading the same cache.
 */
static int llama_tt_prefill_pass(pn_llama_tt *g, pn_model *model, pn_state *state,
                                 const uint32_t *tokens, const float *const *embd_rows,
                                 int rows, int base_pos)
{
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_llama_tt_strips *s = &g->card;
    pn_config *cfg = &model->config;
    int n_embd = cfg->n_embd;
    int q_width = cfg->n_head * pn_head_dim(cfg);
    int kv_width = cfg->n_kv_head * pn_head_dim(cfg);

    for (int r = 0; r < rows; r++) {
        if (embd_rows && embd_rows[r])
            memcpy(g->b_hidden + (size_t)r * n_embd, embd_rows[r],
                   (size_t)n_embd * sizeof(float));
        else
            memcpy(g->b_hidden + (size_t)r * n_embd,
                   qw->tok_embd + (size_t)tokens[r] * n_embd,
                   (size_t)n_embd * sizeof(float));
    }
    s->slot = 0;
    g->hidden_on_host = false;
    g->hidden_on_card = true;
    if (pn_tt_activation_upload_rows(g->device, llama_tt_hidden(g), g->b_hidden,
                                     (uint32_t)n_embd, (uint32_t)rows) != 0)
        return -1;

    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_qlayer *cpu = &qw->layers[layer];
        pn_llama_tt_layer *d = &g->layers[layer];

        if (llama_tt_norm_rows(g, &d->attn_norm, cpu->attn_norm, cfg, llama_tt_hidden(g),
                               &s->normed, rows) != 0)
            return -1;
        /* The same chain the single-token path takes: one wait for three projections. */
        if (llama_tt_matvec_begin(g, &d->wq, NULL, &s->normed, &s->q, layer, "attn_q") != 0 ||
            llama_tt_matvec_begin(g, &d->wk, NULL, &s->normed, &s->k, layer, "attn_k") != 0 ||
            llama_tt_matvec_begin(g, &d->wv, NULL, &s->normed, &s->v, layer, "attn_v") != 0)
            return -1;
        if (pn_tt_matvec_flush(g->device) != 0)
            return -1;
        if (pn_tt_activation_download_rows(g->device, &s->q, g->b_q, (uint32_t)q_width,
                                           (uint32_t)rows) != 0 ||
            pn_tt_activation_download_rows(g->device, &s->k, g->b_k, (uint32_t)kv_width,
                                           (uint32_t)rows) != 0 ||
            pn_tt_activation_download_rows(g->device, &s->v, g->b_v, (uint32_t)kv_width,
                                           (uint32_t)rows) != 0)
            return -1;
        /* STRICTLY IN POSITION ORDER, and that is a correctness requirement rather than a
         * convenience: row r attends its own K/V, so row r's append has to have happened
         * before row r reads the cache, and rows r+1.. must NOT have. A row loop in order
         * gives exactly the causal mask the sequential path produces, with no mask code. */
        if (llama_tt_red("prefill_rowfill"))
            for (int r = 1; r < rows; r++) {
                memcpy(g->b_q + (size_t)r * q_width, g->b_q,
                       (size_t)q_width * sizeof(float));
                memcpy(g->b_k + (size_t)r * kv_width, g->b_k,
                       (size_t)kv_width * sizeof(float));
                memcpy(g->b_v + (size_t)r * kv_width, g->b_v,
                       (size_t)kv_width * sizeof(float));
            }
        for (int i = 0; i < rows; i++) {
            int r = llama_tt_red("prefill_causal") ? rows - 1 - i : i;
            int row_pos = llama_tt_red("prefill_pos") ? base_pos : base_pos + r;

            if (llama_tt_attend_one(g, model, state, layer, row_pos,
                                    g->b_q + (size_t)r * q_width,
                                    g->b_k + (size_t)r * kv_width,
                                    g->b_v + (size_t)r * kv_width,
                                    g->b_attn + (size_t)r * q_width) != 0)
                return -1;
        }
        if (pn_tt_activation_upload_rows(g->device, &s->attn, g->b_attn, (uint32_t)q_width,
                                         (uint32_t)rows) != 0)
            return -1;
        if (llama_tt_matvec(g, &d->wo, NULL, &s->attn, &s->project, layer, "attn_output") != 0)
            return -1;
        if (llama_tt_residual_rows(g, cfg, rows) != 0)
            return -1;

        if (llama_tt_norm_rows(g, &d->ffn_norm, cpu->ffn_norm, cfg, llama_tt_hidden(g),
                               &s->normed, rows) != 0)
            return -1;
        if (llama_tt_matvec_begin(g, &d->w_gate, NULL, &s->normed, &s->gate, layer,
                                  "ffn_gate") != 0 ||
            llama_tt_matvec_begin(g, &d->w_up, NULL, &s->normed, &s->up, layer, "ffn_up") != 0)
            return -1;
        if (pn_tt_matvec_flush(g->device) != 0)
            return -1;
        if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_SWIGLU)) {
            if (pn_tt_silu_mul(g->device, &s->gate, &s->up, &s->swiglu) != 0)
                return -1;
        } else {
            if (pn_tt_activation_download_rows(g->device, &s->gate, g->b_gate,
                                               (uint32_t)cfg->n_ff, (uint32_t)rows) != 0 ||
                pn_tt_activation_download_rows(g->device, &s->up, g->b_up,
                                               (uint32_t)cfg->n_ff, (uint32_t)rows) != 0)
                return -1;
            for (int r = 0; r < rows; r++)
                pn_vsilu_mul_f32(g->b_gate + (size_t)r * cfg->n_ff,
                                 g->b_up + (size_t)r * cfg->n_ff,
                                 g->b_gate + (size_t)r * cfg->n_ff, cfg->n_ff);
            if (pn_tt_activation_upload_rows(g->device, &s->swiglu, g->b_gate,
                                             (uint32_t)cfg->n_ff, (uint32_t)rows) != 0)
                return -1;
        }
        if (llama_tt_matvec(g, &d->w_down, NULL, &s->swiglu, &s->project, layer,
                            "ffn_down") != 0)
            return -1;
        if (llama_tt_residual_rows(g, cfg, rows) != 0)
            return -1;
    }
    state->pos = base_pos + rows;
    return 0;
}

/** Return every strip this file owns to the zeroed state a Mt=1 matvec's row 0 depends on.
 *  Nothing in the decode path reads rows 1-31 — `pn_tt_topk` names row 0 in its descriptor
 *  and every other op is per-row — so this is not load-bearing today. It is what keeps the
 *  strip invariant a property of the strip rather than a claim about who reads what. */
static int llama_tt_restage_strips(pn_llama_tt *g)
{
    pn_llama_tt_strips *s = &g->card;
    const pn_tt_activation *all[] = { &s->hidden[0], &s->hidden[1], &s->normed, &s->q, &s->k,
                                      &s->v, &s->q_rot, &s->k_rot, &s->attn, &s->project,
                                      &s->gate, &s->up, &s->swiglu, &s->logits };

    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
        if (pn_tt_activation_zero(g->device, all[i]) != 0)
            return -1;
    return 0;
}

tm_tensor *pn_llama_prefill_tt(pn_model *model, const uint32_t *tokens,
                               const float *const *embd_rows, int n_tokens, pn_state *state)
{
    pn_llama_tt *g;
    pn_llama_kv_cache *kv;
    tm_tensor *last = NULL;
    int rows, min_rows, done;

    if (!model || !state || !model->weights || !model->gpu || n_tokens <= 0 || !tokens)
        return NULL;
    if (model->forward != pn_llama_forward_tt)
        return NULL;
    g = (pn_llama_tt *)model->gpu;
    if (!g->loaded)
        return NULL;
    kv = (pn_llama_kv_cache *)state->kv_cache;
    if (!kv)
        return NULL;
    /* REFUSALS, each of which falls back to the caller's per-token loop rather than being
     * absorbed. A batch of one is the sequential path and buys nothing. */
    if (n_tokens < 2 || !llama_tt_prefill_enabled())
        return NULL;
    /* ON-CARD ATTENTION IS SINGLE-POSITION BY CONSTRUCTION: pn_tt_rope, pn_tt_kv_append and
     * the flash decode each take ONE position, so a 32-row pass has no way to express 32 of
     * them. Refusing is the honest answer; silently rotating every row to base_pos would be
     * a wrong answer with no symptom. */
    if (pn_tt_device_oncard(g->device, PN_TT_ONCARD_ATTENTION))
        return NULL;
    /* The trace scores an op against the CPU oracle at Mt=1 and would read row 0 of a
     * 32-row strip as if the other 31 were padding. An instrument that silently measured
     * something else is worse than not having it here. */
    if (g->trace > 0 || g->attn_check || g->rope_check || g->row_census)
        return NULL;
    /* SAME REASON, one layer down. PN_TT_MATVEC_DUMP appends logical ROW 0 of every
     * projection's output, which under a 32-row pass is the first token of the pass and
     * not the op — and the op COUNT changes too, so two dumps taken at different core
     * counts would no longer be comparable files. The multi-core byte gate is the thing
     * that instrument exists for; batching under it would quietly break that gate. */
    if (getenv("PN_TT_MATVEC_DUMP"))
        return NULL;
    /* A pass must not wrap the ring: every row of it is live at once, so a wrap would have
     * a later row overwrite an earlier row's K/V inside the same pass. */
    if (state->pos + n_tokens > kv->max_seq)
        return NULL;

    rows = llama_tt_prefill_rows();
    min_rows = llama_tt_prefill_min_rows();
    if (min_rows > rows)
        min_rows = rows;
    /* REFUSED BEFORE ANYTHING IS TOUCHED. A prompt with no full-width pass in it would be
     * SLOWER batched, so the honest answer is to hand it back rather than to run it. */
    if (n_tokens - 1 < min_rows)
        return NULL;
    if (llama_tt_ensure_scores(g, kv->max_seq) != 0)
        return NULL;
    if (llama_tt_alloc_batch(g, &model->config, rows) != 0)
        return NULL;

    /* Batch while a pass is still worth its whole-tile transfer; then the TAIL — the last
     * token always, plus whatever remainder is under the break-even — goes through the
     * ordinary forward. That is also what keeps the head, the top-N and the sparse-logit
     * presentation as the shipping code unmodified: no LM head runs in a batched pass at all. */
    done = 0;
    while (n_tokens - 1 - done >= min_rows) {
        int take = n_tokens - 1 - done;

        if (take > rows) take = rows;
        if (llama_tt_prefill_pass(g, model, state, tokens + done,
                                  embd_rows ? embd_rows + done : NULL, take,
                                  state->pos) != 0)
            return NULL;
        done += take;
    }
    if (llama_tt_restage_strips(g) != 0)
        return NULL;

    for (int i = done; i < n_tokens; i++) {
        if (last) tm_free(last);
        if (embd_rows && embd_rows[i]) {
            state->embd_override = embd_rows[i];
            last = pn_llama_forward_tt(model, 0, state);
            state->embd_override = NULL;
        } else {
            last = pn_llama_forward_tt(model, tokens[i], state);
        }
        if (!last)
            return NULL;
    }
    return last;
}

#endif /* USE_TT */
