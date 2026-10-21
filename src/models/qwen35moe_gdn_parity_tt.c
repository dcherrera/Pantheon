/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gdn_parity_tt.c
 * @brief The ON-CARD GATE for the Gated-Delta-Net arm: a SHADOW host recurrence run beside
 *        the card's, on the same inputs, from the same zero initial condition, so a card
 *        answer can be priced against the oracle's every token of every SSM layer.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT COMPARES, AND WHY THOSE TWO
 * ---------------------------------------------------------------------------
 * Two numbers a layer, and they SPLIT THE OP:
 *
 *   conv    `g->gdn_conv_out` against `pn_qwen35moe_conv1d_silu()` — PHASE CONV alone,
 *           including the rolling window, so a wrong tap plane or a wrong live row shows
 *           here and nowhere else.
 *   gated   `g->h_gated` against the whole of `pn_qwen35moe_ssm_recurrence()` — the block's
 *           actual output. With `conv` clean, a `gated` miss is PHASE DELTA and the two
 *           host norms around it, and the norms are the oracle's own code on both arms.
 *
 * Both call the ORACLE'S OWN functions. A parity arm that reimplemented the conv1d would be
 * comparing the card against a third answer, which measures nothing.
 *
 * ---------------------------------------------------------------------------
 * THE SHADOW STATE IS THE HOST ARM'S OWN, AND IT EVOLVES INDEPENDENTLY
 * ---------------------------------------------------------------------------
 * `pn_qwen35moe_state_new()` allocates a zeroed `pn_qwen35moe_ssm_state` per SSM layer
 * whatever arm runs, and the on-card arm leaves it untouched. So the shadow costs no
 * allocation and starts where the card's DRAM state starts. It is NEVER resynced from the
 * card: the two recurrences run side by side for the whole sequence, which is the only way a
 * carried-state error is visible at all. A single-token compare passes on a recurrence that
 * is right once and wrong on the second token — the reason the acceptance is 48 tokens.
 *
 * **The consequence is that a per-token number is drift, not op error.** Position 0 is the
 * one place both states are zero, so the position-0 figures are the op's own error with no
 * compounding, and the trajectory after it is how that error propagates. Both are reported.
 *
 * ---------------------------------------------------------------------------
 * THE METRIC IS MAX-NORM RELATIVE, NOT PER-ELEMENT RELATIVE
 * ---------------------------------------------------------------------------
 * `max|card - host| / max|host|` over the tensor. Per-element relative error is unusable
 * here: a delta-rule output has elements many orders below its own scale and dividing by one
 * of those reports 10^6 for a difference that is numerical noise. Max-norm relative is
 * scale-invariant, has no arbitrary floor, and is what "1e-3 relative" means for a tensor.
 *
 * **THE BAR IS 1e-3 AND IT IS THE CUDA ARM'S, NOT THE CPU'S.** The oracle accumulates its
 * dots in fp64 deliberately; a Tensix cannot, so bit-equality was never available. That was
 * decided in `foundry/tt/tests/Makefile.gdn` before this file existed.
 *
 * @status Ran the first on-card GDN gate, 2026-08-18. It found two silent defects (a shared
 *   NoC coordinate word and a core collision with the matmul group) and then localised the
 *   remainder: at position 0 layer 0 `conv` is 6.872e-04 and `gated` is 1.000e+00, so PHASE
 *   CONV is inside the bar and PHASE DELTA is not.
 * @issues The per-token figures after position 0 are DRIFT, not op error — the two states
 *   evolve independently and are never resynced, which is deliberate (it is the only way a
 *   carried-state error is visible at all) but means only the position-0 column is a
 *   statement about one descriptor.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../memory/arena.h"

/** The gate. Fixed, not an environment variable: a tolerance a run can widen is not a gate. */
#define PN_GDN_PARITY_TOLERANCE 1.0e-3f

/** `PN_TT_GDN_PARITY=1` arms it. Off by default — it runs a second full host recurrence per
 *  SSM layer, which is the cost the on-card arm exists to remove. */
#define PN_GDN_PARITY_ENV "PN_TT_GDN_PARITY"

/** One tensor's worst disagreement, and where it was. */
typedef struct {
    float worst;      /**< max-norm relative error */
    int worst_pos;
    int worst_layer;
    float at_zero;    /**< the same figure at position 0, where neither state has drifted */
} pn_gdn_parity_track;

struct pn_qwen35_gdn_parity {
    float *host_conv_out; /**< conv_dim */
    float *host_gated;    /**< ssm_state * ssm_dt_rank */
    float *saved_window;  /**< (ssm_conv-1) * conv_dim — the shadow window across the probe */
    size_t window_floats;
    pn_gdn_parity_track conv;
    pn_gdn_parity_track gated;
    float token_conv_this_layer; /**< the layer being compared right now, for the pos-0 trace */
    float token_conv;     /**< worst of either tensor within the token being built */
    float token_gated;
    int token_conv_layer;
    int token_gated_layer;
    int tokens;
    int layers_compared;
};

/* ================================================================
 * The metric
 * ================================================================ */

/**
 * `max|card - host| / max|host|` over `count` elements.
 *
 * @param worst_index  written with the element that produced it, or -1 if the reference is
 *                     identically zero (in which case the answer is 0 or +INFINITY, and the
 *                     caller wants to know which element)
 * @return             the max-norm relative error; 0 when both tensors are identically zero
 */
static float gdn_parity_relative(const float *card, const float *host, size_t count,
                                 size_t *worst_index)
{
    float worst_diff = 0.0f;
    float reference = 0.0f;
    size_t index;
    size_t at = 0;

    for (index = 0; index < count; index++) {
        const float diff = fabsf(card[index] - host[index]);
        const float magnitude = fabsf(host[index]);

        if (diff > worst_diff) {
            worst_diff = diff;
            at = index;
        }
        if (magnitude > reference)
            reference = magnitude;
    }
    *worst_index = at;
    if (reference <= FLT_MIN)
        return worst_diff > 0.0f ? INFINITY : 0.0f;
    return worst_diff / reference;
}

/**
 * WHERE the disagreeing channels are, not just how big the worst one is.
 *
 * A tilized one-row strip has two coordinates a bug can follow — the TILE (`c / 32`) and the
 * COLUMN INSIDE it (`c % 32`) — and Tf32 rounding follows neither. Printing the histogram of
 * `c % 32` separates "a few channels round differently" from "one column of every tile is
 * wrong", which are a tolerance question and a layout bug respectively.
 */
static void gdn_parity_locate(const float *card, const float *host, size_t count)
{
    unsigned per_column[32] = { 0 };
    size_t first[6];
    size_t found = 0;
    size_t index;
    size_t bad = 0;

    for (index = 0; index < count; index++) {
        const float magnitude = fabsf(host[index]);

        if (fabsf(card[index] - host[index]) <= 1.0e-3f * magnitude + 1.0e-6f)
            continue;
        bad++;
        per_column[index % 32u]++;
        if (found < 6u)
            first[found++] = index;
    }
    fprintf(stderr, "gdn-parity:   %zu of %zu channels off by more than 1e-3 relative\n", bad,
            count);
    fprintf(stderr, "gdn-parity:   by column %% 32:");
    for (index = 0; index < 32u; index++)
        fprintf(stderr, " %u", per_column[index]);
    fprintf(stderr, "\n");
    fprintf(stderr, "gdn-parity:   first bad channels:");
    for (index = 0; index < found; index++)
        fprintf(stderr, " %zu(tile %zu col %zu)", first[index], first[index] / 32u,
                first[index] % 32u);
    fprintf(stderr, "\n");
}

/** Fold one comparison into a tensor's running worst. */
static void gdn_parity_record(pn_gdn_parity_track *track, float relative, int position,
                              int layer)
{
    if (relative > track->worst) {
        track->worst = relative;
        track->worst_pos = position;
        track->worst_layer = layer;
    }
    if (position == 0 && relative > track->at_zero)
        track->at_zero = relative;
}

/* ================================================================
 * Lifecycle
 * ================================================================ */

int pn_qwen35_tt_gdn_parity_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    const char *requested = getenv(PN_GDN_PARITY_ENV);
    struct pn_qwen35_gdn_parity *parity;

    if (!g || !cfg)
        return -1;
    if (!g->gdn_on_card || !requested || !*requested || strcmp(requested, "0") == 0)
        return 0;

    parity = calloc(1, sizeof(*parity));
    if (!parity)
        return -1;
    parity->window_floats = (size_t)(cfg->ssm_conv - 1) * (size_t)cfg->conv_dim;
    parity->host_conv_out = calloc((size_t)cfg->conv_dim, sizeof(float));
    parity->host_gated =
        calloc((size_t)cfg->ssm_state * (size_t)cfg->ssm_dt_rank, sizeof(float));
    parity->saved_window = calloc(parity->window_floats, sizeof(float));
    if (!parity->host_conv_out || !parity->host_gated || !parity->saved_window) {
        free(parity->host_conv_out);
        free(parity->host_gated);
        free(parity->saved_window);
        free(parity);
        return -1;
    }
    g->gdn_parity = parity;
    fprintf(stderr, "qwen35_tt: gdn PARITY armed — a shadow host recurrence runs beside every "
                    "SSM layer and the gate is %.1e max-norm relative. This roughly TRIPLES "
                    "the token; it is a gate, not a measurement arm.\n",
            (double)PN_GDN_PARITY_TOLERANCE);
    return 0;
}

void pn_qwen35_tt_gdn_parity_release(pn_qwen35_tt *g)
{
    if (!g || !g->gdn_parity)
        return;
    free(g->gdn_parity->host_conv_out);
    free(g->gdn_parity->host_gated);
    free(g->gdn_parity->saved_window);
    free(g->gdn_parity);
    g->gdn_parity = NULL;
}

/* ================================================================
 * The comparison
 * ================================================================ */

int pn_qwen35_tt_gdn_parity_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                  const pn_qwen35moe_config *cfg, int layer,
                                  const float *qkv, const float *z, const float *beta_raw,
                                  const float *alpha, pn_qwen35moe_ssm_state *state,
                                  pn_arena *arena, struct fw_pool *pool)
{
    struct pn_qwen35_gdn_parity *parity = g ? g->gdn_parity : NULL;
    const size_t conv_count = (size_t)cfg->conv_dim;
    const size_t gated_count = (size_t)cfg->ssm_state * (size_t)cfg->ssm_dt_rank;
    size_t where;
    float relative;

    if (!parity)
        return 0;
    if (!l || !qkv || !z || !beta_raw || !alpha || !state || !arena)
        return -1;

    /* PHASE CONV alone. The oracle's conv1d ROLLS the window, and the whole recurrence below
     * has to roll it exactly once for this token, so the window is saved and put back. */
    memcpy(parity->saved_window, state->conv_state, parity->window_floats * sizeof(float));
    pn_qwen35moe_conv1d_silu(l, cfg, qkv, state, parity->host_conv_out);
    memcpy(state->conv_state, parity->saved_window, parity->window_floats * sizeof(float));

    relative = gdn_parity_relative(g->gdn_conv_out, parity->host_conv_out, conv_count, &where);
    parity->token_conv_this_layer = relative;
    if (relative > parity->token_conv) {
        parity->token_conv = relative;
        parity->token_conv_layer = layer;
    }
    /* The first layer of the first token, once, AS BIT PATTERNS. At position 0 the window is
     * zero, so the card's answer is SiLU(qkv * w[.][taps-1]) and nothing else — the simplest
     * arithmetic this op ever does. A ratio cannot say which arm is wrong and a decimal
     * cannot say HOW: a swapped half-word, a shifted exponent and a wrong tile all read as
     * "garbage" in %g and are three different bugs in %08x. */
    if (parity->tokens == 0 && parity->layers_compared == 0) {
        union { float f; uint32_t u; } card, host;
        size_t index;

        for (index = 0; index < 8u; index++) {
            card.f = g->gdn_conv_out[index];
            host.f = parity->host_conv_out[index];
            fprintf(stderr, "gdn-parity:   conv_out[%zu] card %08x (%g)  host %08x (%g)  "
                            "qkv %g  w[.][taps-1] %g\n",
                    index, card.u, (double)card.f, host.u, (double)host.f,
                    (double)qkv[index],
                    (double)l->ssm_conv1d[index * (size_t)cfg->ssm_conv +
                                          (size_t)(cfg->ssm_conv - 1)]);
        }
        card.f = g->gdn_conv_out[where];
        host.f = parity->host_conv_out[where];
        fprintf(stderr, "gdn-parity:   worst element %zu card %08x (%g) host %08x (%g)\n",
                where, card.u, (double)card.f, host.u, (double)host.f);
        gdn_parity_locate(g->gdn_conv_out, parity->host_conv_out, conv_count);
    }

    /* The whole block, on the shadow state, from the same four projections. The arena was
     * reset at the top of this layer and the on-card arm allocates nothing from it. */
    pn_arena_reset(arena);
    if (pn_qwen35moe_ssm_recurrence(l, cfg, qkv, z, beta_raw, alpha, state, arena,
                                    parity->host_gated, pool) < 0)
        return -1;

    relative = gdn_parity_relative(g->h_gated, parity->host_gated, gated_count, &where);
    /* THE FIRST TOKEN, LAYER BY LAYER. A per-token maximum says how bad the block is; it
     * cannot say WHERE it starts, and at position 0 every carried state is zero, so a layer's
     * error is that layer's own plus whatever the residual stream carried into it. The layer
     * at which this stops being small is the layer the search belongs in.
     *
     * POSITION 1 IS PRINTED TOO, and it is a different statement: `conv` cannot drift, because
     * BOTH arms are handed the SAME `qkv` at every layer and the only other input is the
     * rolling window, which both roll from that same qkv. So a `conv` figure that is 7e-4 at
     * position 0 and 2.3 at position 1 is not drift — it is the ROLL. */
    if (parity->tokens < 2) {
        /* `gated` landing on exactly 1.000 means `max|card - host| == max|host|`, which for a
         * ratio is the signature of a card answer that is identically ZERO — and the block's
         * output is zero exactly when PHASE DELTA's `y` is. So the scale of `y` itself is
         * printed beside the ratio; a ratio cannot distinguish "wrong" from "absent". */
        float y_scale = 0.0f;
        size_t index;

        for (index = 0; index < gated_count; index++)
            if (fabsf(g->gdn_y[index]) > y_scale)
                y_scale = fabsf(g->gdn_y[index]);
        fprintf(stderr, "gdn-parity: pos %d layer %2d  conv %.3e  gated %.3e  max|card y| %.4g "
                        "(y[0..3] %g %g %g %g)\n",
                parity->tokens, layer, (double)parity->token_conv_this_layer, (double)relative,
                (double)y_scale, (double)g->gdn_y[0], (double)g->gdn_y[1],
                (double)g->gdn_y[2], (double)g->gdn_y[3]);
    }
    if (relative > parity->token_gated) {
        parity->token_gated = relative;
        parity->token_gated_layer = layer;
    }
    parity->layers_compared++;
    return 0;
}

void pn_qwen35_tt_gdn_parity_token(pn_qwen35_tt *g, int position)
{
    struct pn_qwen35_gdn_parity *parity = g ? g->gdn_parity : NULL;

    if (!parity)
        return;
    gdn_parity_record(&parity->conv, parity->token_conv, position, parity->token_conv_layer);
    gdn_parity_record(&parity->gated, parity->token_gated, position,
                      parity->token_gated_layer);
    fprintf(stderr, "gdn-parity: pos=%-4d conv %.3e (layer %2d)   gated %.3e (layer %2d)\n",
            position, (double)parity->token_conv, parity->token_conv_layer,
            (double)parity->token_gated, parity->token_gated_layer);
    parity->token_conv = 0.0f;
    parity->token_gated = 0.0f;
    parity->token_conv_layer = -1;
    parity->token_gated_layer = -1;
    parity->tokens++;
}

void pn_qwen35_tt_gdn_parity_report(const pn_qwen35_tt *g)
{
    const struct pn_qwen35_gdn_parity *parity = g ? g->gdn_parity : NULL;
    bool passed;

    if (!parity)
        return;
    passed = parity->conv.worst <= PN_GDN_PARITY_TOLERANCE &&
             parity->gated.worst <= PN_GDN_PARITY_TOLERANCE;
    fprintf(stderr,
            "gdn-parity: %d tokens, %d layer comparisons\n"
            "gdn-parity:   conv   worst %.3e at pos %d layer %d   |  pos-0 %.3e\n"
            "gdn-parity:   gated  worst %.3e at pos %d layer %d   |  pos-0 %.3e\n"
            "gdn-parity: GATE %.1e max-norm relative — %s\n",
            parity->tokens, parity->layers_compared, (double)parity->conv.worst,
            parity->conv.worst_pos, parity->conv.worst_layer, (double)parity->conv.at_zero,
            (double)parity->gated.worst, parity->gated.worst_pos, parity->gated.worst_layer,
            (double)parity->gated.at_zero, (double)PN_GDN_PARITY_TOLERANCE,
            passed ? "PASS" : "FAIL");
}

#endif /* USE_TT */
