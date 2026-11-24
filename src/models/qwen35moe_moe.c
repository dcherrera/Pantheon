/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_moe.c
 * @brief Qwen3.5-MoE shared-expert MoE FFN (A5). Runs on EVERY layer (both SSM
 *        and attention) on the post_attention_norm'd hidden state. Extends
 *        Pantheon's Mixtral-style routed MoE with softmax + norm_topk_prob
 *        gating and an always-on sigmoid-gated shared expert.
 *
 * @status A5 complete: routed softmax + top-k + norm_topk_prob + expert_weights
 *         _scale SwiGLU experts, plus the sigmoid-gated shared expert and add.
 *         Verified in isolation against a NumPy reference (< 1e-4 max abs diff).
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "qwen35moe_internal.h"

/* 2. Standard library */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 3. System / platform headers */
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

/* 4. Project headers */
#include "../accel/simd.h"
#include "../core/qweight.h"
#include "qwen35moe_routing_dump.h"

/* Hard cap on top-k experts — sized for the on-stack selection scratch. The
 * Qwen3.5-MoE configs use 8; Mixtral uses 2. */
#define PN_QWEN35MOE_MAX_TOPK 32

/* sigmoid(x) = 1 / (1 + exp(-x)). */
static inline float qwen35moe_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================
 * Routed experts: softmax router -> top-k -> norm_topk_prob -> SwiGLU sum
 * ============================================================ */

/* Softmax over the n_expert router logits, in place. */
static void qwen35moe_router_softmax(float *logits, int n_expert)
{
    float max_gl = logits[0];
    for (int e = 1; e < n_expert; e++)
        if (logits[e] > max_gl)
            max_gl = logits[e];
    float sum_exp = 0.0f;
    for (int e = 0; e < n_expert; e++) {
        logits[e] = expf(logits[e] - max_gl);
        sum_exp += logits[e];
    }
    float inv = 1.0f / sum_exp;
    for (int e = 0; e < n_expert; e++)
        logits[e] *= inv;
}

/* Select the top-k experts by probability (selection sort, k is small).
 * Fills top_idx/top_wt and returns the sum of the selected weights. */
static float qwen35moe_select_topk(const float *probs, int n_expert, int k,
                                   int *top_idx, float *top_wt)
{
    float wt_sum = 0.0f;
    for (int s = 0; s < k; s++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_expert; e++) {
            int skip = 0;
            for (int j = 0; j < s; j++)
                if (top_idx[j] == e) {
                    skip = 1;
                    break;
                }
            if (!skip && probs[e] > best_val) {
                best = e;
                best_val = probs[e];
            }
        }
        top_idx[s] = best;
        top_wt[s] = best_val;
        wt_sum += best_val;
    }
    return wt_sum;
}

/* SwiGLU for one expert: out = (silu(gate . x) * (up . x)) . down.
 * scratch_gate/up/hid are [expert_ff], out is [n_embd]. */
static void qwen35moe_swiglu_expert(const pn_qwen35moe_config *cfg,
                                    const pn_qweight *gate, const pn_qweight *up,
                                    const pn_qweight *down, const float *x,
                                    int expert_ff, float *scratch_gate,
                                    float *scratch_up, float *scratch_hid,
                                    float *out)
{
    qwen35moe_mv(cfg, gate, x, scratch_gate);
    qwen35moe_mv(cfg, up, x, scratch_up);
    pn_vsilu_mul_f32(scratch_gate, scratch_up, scratch_hid, expert_ff);
    qwen35moe_mv(cfg, down, scratch_hid, out);
}

/* Routed-MoE branch: softmax router, top-k, norm_topk_prob renormalization,
 * expert_weights_scale, then the weighted SwiGLU expert sum into moe_out. */
static int qwen35moe_routed(const pn_qwen35moe_moe *moe,
                            const pn_qwen35moe_config *cfg, const float *x,
                            pn_arena *arena, float *moe_out)
{
    const int n_embd = cfg->n_embd;
    const int n_expert = cfg->n_expert;
    const int k = cfg->n_expert_used;
    const int expert_ff = cfg->expert_ff;

    float *logits = pn_arena_alloc_f32(arena, n_expert);
    float *exp_gate = pn_arena_alloc_f32(arena, expert_ff);
    float *exp_up = pn_arena_alloc_f32(arena, expert_ff);
    float *exp_hid = pn_arena_alloc_f32(arena, expert_ff);
    float *exp_out = pn_arena_alloc_f32(arena, n_embd);
    if (!logits || !exp_gate || !exp_up || !exp_hid || !exp_out)
        return -1;

    qwen35moe_mv(cfg, &moe->ffn_gate_inp, x, logits);
    qwen35moe_router_softmax(logits, n_expert);

    int top_idx[PN_QWEN35MOE_MAX_TOPK];
    float top_wt[PN_QWEN35MOE_MAX_TOPK];
    float wt_sum = qwen35moe_select_topk(logits, n_expert, k, top_idx, top_wt);
    pn_qwen35moe_routing_dump(logits, n_expert, top_idx, k);

    /* norm_topk_prob: renormalize the selected weights to sum to 1, then apply
     * the routed-weight scale (1.0 unless the GGUF carries the key). */
    float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
    float scale = cfg->expert_weights_scale * inv_wt;
    for (int s = 0; s < k; s++)
        top_wt[s] *= scale;

    memset(moe_out, 0, (size_t)n_embd * sizeof(float));
    for (int s = 0; s < k; s++) {
        int e = top_idx[s];
        qwen35moe_swiglu_expert(cfg, &moe->gate_exps[e], &moe->up_exps[e],
                                &moe->down_exps[e], x, expert_ff, exp_gate,
                                exp_up, exp_hid, exp_out);
        float w = top_wt[s];
        for (int j = 0; j < n_embd; j++)
            moe_out[j] += w * exp_out[j];
    }
    return 0;
}

/* ============================================================
 * Shared expert (always on, sigmoid-gated)
 * ============================================================ */

/* Shared-expert branch: shexp = sigmoid(ffn_gate_inp_shexp . x) * SwiGLU_shexp(x).
 * The scalar sigmoid gate is applied ONLY to this branch. */
static int qwen35moe_shared(const pn_qwen35moe_moe *moe,
                            const pn_qwen35moe_config *cfg, const float *x,
                            pn_arena *arena, float *shexp_out)
{
    const int n_embd = cfg->n_embd;
    const int shared_ff = cfg->shared_ff;

    float *sh_gate = pn_arena_alloc_f32(arena, shared_ff);
    float *sh_up = pn_arena_alloc_f32(arena, shared_ff);
    float *sh_hid = pn_arena_alloc_f32(arena, shared_ff);
    if (!sh_gate || !sh_up || !sh_hid)
        return -1;

    qwen35moe_swiglu_expert(cfg, &moe->ffn_gate_shexp, &moe->ffn_up_shexp,
                            &moe->ffn_down_shexp, x, shared_ff, sh_gate, sh_up,
                            sh_hid, shexp_out);

    /* 1 scalar/token: project x -> single logit, sigmoid, scale the branch.
     * ffn_gate_inp_shexp is a 1-D [n_embd] vector; the GGUF loader stores 1-D
     * tensors as rows=n_embd/cols=1, so pn_qmv would emit n_embd outputs (and
     * overflow a scalar). Compute the dot product directly instead — a single
     * weight . x reduction over the embedding dim.
     *
     * Reported to the dispatch observer by hand precisely BECAUSE it bypasses
     * pn_qmv: a GPU backend still owes this site a kernel, so leaving it
     * unreported would let the coverage gate mistake a real dispatch for an
     * unused weight. */
    pn_qweight_note_dispatch(&moe->ffn_gate_inp_shexp, PN_QWEIGHT_OP_DOT);
    const float *gate_w = (const float *)moe->ffn_gate_inp_shexp.data;
    float gate_logit = 0.0f;
    for (int j = 0; j < n_embd; j++)
        gate_logit += gate_w[j] * x[j];
    float sgate = qwen35moe_sigmoid(gate_logit);
    for (int j = 0; j < n_embd; j++)
        shexp_out[j] *= sgate;
    return 0;
}

/* ============================================================
 * Block entry point
 * ============================================================ */

/* Dense-FFN branch (Bonsai): a single plain SwiGLU, no router, no scalar gate.
 * gate/up/down live in the shared-expert slots; dim = ffn_length (= shared_ff).
 * out = (silu(gate . x) * (up . x)) . down. Caller adds the residual. */
static int qwen35moe_dense_ffn(const pn_qwen35moe_moe *moe,
                               const pn_qwen35moe_config *cfg, const float *x,
                               pn_arena *arena, float *out)
{
    const int ff = cfg->ffn_length;
    float *g = pn_arena_alloc_f32(arena, ff);
    float *u = pn_arena_alloc_f32(arena, ff);
    float *h = pn_arena_alloc_f32(arena, ff);
    if (!g || !u || !h)
        return -1;
    qwen35moe_swiglu_expert(cfg, &moe->ffn_gate_shexp, &moe->ffn_up_shexp,
                            &moe->ffn_down_shexp, x, ff, g, u, h, out);
    return 0;
}

int pn_qwen35moe_moe_block(const pn_qwen35moe_moe *moe,
                           const pn_qwen35moe_config *cfg, const float *x,
                           pn_arena *arena, float *out, struct fw_pool *pool)
{
    (void)pool; /* per-token MoE is cheap; threading reserved for A6. */
    if (!moe || !cfg || !x || !arena || !out)
        return -1;

    /* Bonsai dense SwiGLU FFN — no MoE routing. */
    if (cfg->dense_ffn)
        return qwen35moe_dense_ffn(moe, cfg, x, arena, out);

    if (cfg->n_expert_used > PN_QWEN35MOE_MAX_TOPK)
        return -1;

    const int n_embd = cfg->n_embd;

    /* Routed experts -> out. */
    if (qwen35moe_routed(moe, cfg, x, arena, out) < 0)
        return -1;

    /* Shared expert (sigmoid-gated) -> shexp, then out += shexp. */
    float *shexp = pn_arena_alloc_f32(arena, n_embd);
    if (!shexp)
        return -1;
    if (qwen35moe_shared(moe, cfg, x, arena, shexp) < 0)
        return -1;
    pn_vadd_f32(out, shexp, out, n_embd);
    return 0;
}
