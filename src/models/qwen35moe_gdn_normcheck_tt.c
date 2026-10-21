/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gdn_normcheck_tt.c
 * @brief SCORE THE TWO PER-HEAD NORMS ON THE CARD AGAINST THE ORACLE'S OWN FUNCTIONS, on
 *        real layer data, without moving them yet.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS MEASURES AND WHY IT IS A SEPARATE STEP
 * ---------------------------------------------------------------------------
 * The claim this project has restated several times without running it is that the GDN's
 * two per-head norms are "ordinary narrow reduces on a second residency". Two halves of
 * that claim were never tested:
 *
 *  1. **The LAYOUT.** A reduce residency reduces along a ROW, and the norms are per head.
 *     Every activation this backend has is a `[1, N]` strip. The resolution is that
 *     `resident_reduce_reader.c` reads row block `b` at tiles `b*tile_cols + wt`, so a
 *     descriptor with `tile_cols = S/32` and `row_blocks = heads` gives ONE reduce per head
 *     with no reshape at all — pn_tt_head_norm.h carries it. This runs it.
 *  2. **The PRECISION.** The host norms are fp64 (the L2's accumulator is `double` on
 *     purpose — `pn_qwen35moe_l2_norm()`'s own comment names a token-60-90 divergence onset
 *     for doing it in float32). The q/k norm feeds the delta rule's recurrent state, so
 *     whether the card's format is affordable there is the question the 48-token `cmp` will
 *     otherwise answer expensively. The FIRST run of this probe answered NO for the bf16
 *     `reduce` program — 8.599e-03 and 1.283e-02 against a 1e-3 gate — which is what put the
 *     residency on `reduce_fp32`.
 *
 * It is a PROBE and not a path: it stages the host's own `conv_out` and `y` into card strips,
 * runs the card, and scores. Nothing it does is on the shipped route and nothing downstream
 * reads its output — which is exactly why it can run beside a `cmp` that must stay identical.
 *
 * **THREE numbers, not two.** The q/k arm is one Float32 descriptor and scores one thing. The
 * gated arm is TWO descriptors on TWO residencies at TWO formats — a Float32 reduce and then
 * a bf16 elementwise gate — so it is scored before and after the crossing. Quoting one number
 * for it is what would hide which half costs what.
 *
 * **`PN_TT_GDN_NORMCHECK=1`**, with `PN_TT_ONCARD=norm,swiglu,residual,topk,gdn` — the
 * gated arm needs the ELEMENTWISE residency for its `silu(z) *` half, and that is the
 * shipped `pn_tt_silu_mul()`.
 *
 * @status Layout CONFIRMED 2026-08-19 on a p150a — the row-block-per-head strip is right and
 *         costs no reshape. On the bf16 `reduce` program the precision answer was NO: q/k L2
 *         8.599e-03 worst head (SSM layer 7), gated 1.283e-02 (layer 16), 146,844 of 196,608
 *         q/k elements past 1e-3.
 *
 *         ON `reduce_fp32`, 2026-08-19, 48 SSM layers, run twice with identical figures and
 *         0 refusals: q/k L2 **1.737e-03** (layer 19), RMS alone **3.165e-03** (layer 8),
 *         gated **1.023e-02** (layer 12); 25,718 of 196,608 q/k elements past 1e-3. So the
 *         Float32 reduce is a real 4.95x on the q/k norm and 5.7x on the element count, and
 *         NEITHER NORM CLEARS THE 1e-3 GATE. pn_tt_head_norm.h decomposes what is left.
 *
 *         Both red arms fire on the card. `PN_TT_RED=head_norm_bf16_constants` (constants at
 *         the 2048 page under a 4096 program) drives every figure to ~1.0e+00 with nothing
 *         refusing anywhere, and `PN_TT_RED=norm_eps` drives them to 8.6e-01/9.99e-01. The
 *         first leaves `cmp` IDENTICAL, which is the proof that this probe really is off the
 *         shipped path.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/simd.h"
#include "../accel/tt/pn_tt_fp32_strip.h"
#include "../accel/tt/pn_tt_head_norm.h"

/** How many SSM layers are scored before the probe goes quiet. Two tokens' worth at
 *  Ornith's 24, so a per-layer figure is available for a position that carries state and
 *  for one that does not. */
#define GDN_NORMCHECK_LAYER_BUDGET 48

/* ---------------------------------------------------------------------------
 * `PN_TT_RED=head_norm_synthetic` — THE ARM THAT SETTLES CARD-VERSUS-MODEL
 *
 * The off-hardware model of this exact program at this exact shape
 * (foundry `tests/test_vector_ops.c`, "the three walls under the Float32 head norms")
 * scores the L2 member 8.406e-04 and the RMS member 1.847e-03. The card scores 1.737e-03
 * and 3.165e-03 — **2.07x and 2.02x above it, the SAME factor on both members**, which is a
 * shared term rather than a per-member one. The three named candidates have now been priced
 * off-hardware and NONE of them is 2x: the SFPU rsqrt polynomial is 7.2e-08 (four orders
 * below the residue), reordering the chain is a null, and the card's 48x32 sample costs the
 * RMS member ~1.3x and the L2 member nothing.
 *
 * What is left unpriced is the DATA. The model runs uniform values in [-1, 1); the card runs
 * Ornith's real `conv_out` and `y`. This arm removes that difference in the only way that
 * settles it — it feeds the card the model's OWN stream and scores it against the same fp64
 * oracle:
 *
 *   card lands on ~8.4e-04 / ~1.8e-03  ->  the 2x is the DATA and the model is sound
 *   card stays at ~1.7e-03 / ~3.2e-03  ->  the 2x is the CARD and the model is missing a
 *                                          mechanism that is neither format nor chain
 *
 * The stream is xorshift64* at seed 0x1220E8, which is `ft_tt_vector_random_value()`. This
 * file cannot link foundry's test tree, so it spells the generator itself — and the two
 * spellings are held together by the eight words below, which `test_vector_ops.c` pins as
 * the same literals. A disagreement is a REFUSAL, not an incomparable number.
 * --------------------------------------------------------------------------- */
/** `pn_tt_red()` spelled here, because a model file does not reach into
 *  `pn_tt_device_internal.h` (llama_forward_tt.c makes the same argument for its own
 *  mutations). One `strcmp` against `PN_TT_RED`, exactly as the accel layer's does. */
static bool normcheck_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

static uint64_t g_normcheck_random_state = 0x1220E8ULL;

static float normcheck_random_value(void)
{
    uint32_t bits;

    g_normcheck_random_state ^= g_normcheck_random_state >> 12;
    g_normcheck_random_state ^= g_normcheck_random_state << 25;
    g_normcheck_random_state ^= g_normcheck_random_state >> 27;
    bits = (uint32_t)((g_normcheck_random_state * 0x2545F4914F6CDD1Dull) >> 40);
    return ((float)bits / 8388608.0f) - 1.0f;
}

/** The generator against the eight words test_vector_ops.c pins. Resets the stream. */
static bool normcheck_random_matches_the_model(void)
{
    static const uint32_t pinned[8] = { 0xBF508D2Au, 0x3EF4CC04u, 0x3EE46400u, 0xBDF7F450u,
                                        0xBE494EE8u, 0xBF4FF826u, 0x3E88AB9Cu, 0xBF4CD48Cu };
    unsigned index;

    g_normcheck_random_state = 0x1220E8ULL;
    for (index = 0; index < 8u; index++) {
        const float value = normcheck_random_value();
        uint32_t bits;

        memcpy(&bits, &value, sizeof(bits));
        if (bits != pinned[index])
            return false;
    }
    g_normcheck_random_state = 0x1220E8ULL;
    return true;
}

struct pn_qwen35_gdn_normcheck {
    /* THE REDUCE'S OWN OPERANDS ARE Float32 — `reduce_fp32` is placed by
     * reduce_float32.cbtable and its reader issues 4096-byte reads. */
    pn_tt_fp32_strip qk;     /**< q then k, 2*key_dim as one [1, N] strip */
    pn_tt_fp32_strip qk_out; /**< the card's L2 answer. SEPARATE from the input on purpose:
                              *   in-place would be a second question in the same run. */
    pn_tt_fp32_strip y;      /**< PHASE DELTA's answer, v_heads*S */
    pn_tt_fp32_strip y_out;  /**< the card's RMS answer */
    /* THE GATE'S ARE bf16 — `pn_tt_silu_mul()` runs on the ELEMENTWISE residency, which is
     * `eltwise_float16b.cbtable`. So the gated arm crosses fp32 -> bf16 between the two
     * descriptors, and scoring the RMS answer BEFORE that crossing is what separates the
     * reduce's error from the bridge's. */
    pn_tt_activation z;        /**< the attn_gate projection, v_heads*S */
    pn_tt_activation y_bridge; /**< the card's RMS answer, rendered back into a bf16 strip */
    pn_tt_activation gated;    /**< silu(z) * rmsnorm(y) */
    pn_tt_norm_constants l2;
    pn_tt_norm_constants *rms; /**< one per SSM layer — gamma is `ssm_norm`, per layer */
    bool *rms_ready;
    float *host;   /**< download scratch, the wider of the two shapes */
    float *want;   /**< the oracle's answer for the same input */
    float *synth;  /**< `PN_TT_RED=head_norm_synthetic`: the model's own stream */
    bool synthetic;
    int ssm_layers;
    int scored;
    double worst_l2;
    double worst_rms;
    double worst_gated;
    int worst_l2_layer;
    int worst_rms_layer;
    int worst_gated_layer;
    uint64_t l2_over_gate;    /**< elements past 1e-3 relative, across every scored layer */
    uint64_t rms_over_gate;
    uint64_t gated_over_gate;
    uint64_t elements;
};

/**
 * Max-norm relative error of `got` against `want`, PER HEAD, returning the worst head.
 *
 * Per head and not per tensor because a per-head fault is invisible in a per-tensor figure
 * — PHASE DELTA scored 1.000e+00 for three rounds on exactly that mistake.
 */
static double normcheck_score(const float *want, const float *got, int heads, int width,
                              uint64_t *out_over_gate)
{
    double worst = 0.0;
    int head;

    for (head = 0; head < heads; head++) {
        const float *w = want + (size_t)head * width;
        const float *g = got + (size_t)head * width;
        double max_abs = 0.0;
        double max_want = 0.0;
        int index;

        for (index = 0; index < width; index++) {
            const double diff = fabs((double)g[index] - (double)w[index]);
            const double magnitude = fabs((double)w[index]);

            if (diff > max_abs)
                max_abs = diff;
            if (magnitude > max_want)
                max_want = magnitude;
            if (magnitude > 0.0 && diff / magnitude > 1e-3)
                (*out_over_gate)++;
        }
        if (max_want > 0.0 && max_abs / max_want > worst)
            worst = max_abs / max_want;
    }
    return worst;
}

int pn_qwen35_tt_gdn_normcheck_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    const uint32_t state = (uint32_t)cfg->ssm_state;
    const uint32_t key_tiles = (uint32_t)(2 * cfg->ssm_state * cfg->ssm_n_group) / 32u;
    const uint32_t value_tiles = (uint32_t)(cfg->ssm_state * cfg->ssm_dt_rank) / 32u;
    struct pn_qwen35_gdn_normcheck *check;
    size_t widest;

    if (!g || !cfg || !getenv("PN_TT_GDN_NORMCHECK"))
        return 0;
    if (pn_tt_device_arm_head_norm(g->device, state) != 0)
        return 0;
    if (!pn_tt_head_norm_is_armed(g->device, state)) {
        fprintf(stderr, "qwen35_tt: PN_TT_GDN_NORMCHECK asked for the per-head norms and the "
                        "residency did not arm — nothing is scored\n");
        return 0;
    }

    check = calloc(1, sizeof(*check));
    if (!check)
        return -1;
    check->ssm_layers = cfg->n_layer;
    check->rms = calloc((size_t)cfg->n_layer, sizeof(*check->rms));
    check->rms_ready = calloc((size_t)cfg->n_layer, sizeof(*check->rms_ready));
    widest = (key_tiles > value_tiles ? key_tiles : value_tiles) * 32u;
    check->host = calloc(widest, sizeof(*check->host));
    check->want = calloc(widest, sizeof(*check->want));
    check->synth = calloc(widest, sizeof(*check->synth));
    if (!check->rms || !check->rms_ready || !check->host || !check->want || !check->synth)
        goto fail;
    if (normcheck_red("head_norm_synthetic")) {
        if (!normcheck_random_matches_the_model()) {
            fprintf(stderr, "qwen35_tt: PN_TT_RED=head_norm_synthetic — this file's xorshift "
                            "does NOT reproduce the model's pinned stream, so the two figures "
                            "would not be comparable. Refusing the arm.\n");
            goto fail;
        }
        check->synthetic = true;
    }

    /* The L2 member takes eps*eps: the oracle clamps the MAGNITUDE at eps and the eps tile
     * folds into the SQUARE. foundry ft_tt_op_vector.h carries the derivation. */
    if (pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, NULL, state,
                                         cfg->rms_norm_eps * cfg->rms_norm_eps,
                                         &check->l2) != 0)
        goto fail;
    if (pn_tt_fp32_strip_alloc(g->device, key_tiles, &check->qk) != 0 ||
        pn_tt_fp32_strip_alloc(g->device, key_tiles, &check->qk_out) != 0 ||
        pn_tt_fp32_strip_alloc(g->device, value_tiles, &check->y) != 0 ||
        pn_tt_fp32_strip_alloc(g->device, value_tiles, &check->y_out) != 0 ||
        pn_tt_activation_alloc(g->device, value_tiles, &check->z) != 0 ||
        pn_tt_activation_alloc(g->device, value_tiles, &check->y_bridge) != 0 ||
        pn_tt_activation_alloc(g->device, value_tiles, &check->gated) != 0)
        goto fail;

    g->gdn_normcheck = check;
    fprintf(stderr, "qwen35_tt: PN_TT_GDN_NORMCHECK is on — the two per-head norms run on "
                    "the card beside the host's and are scored per head over the first %d "
                    "SSM layers. This is a PROBE: nothing downstream reads it.%s\n",
            GDN_NORMCHECK_LAYER_BUDGET,
            check->synthetic ? " RED: head_norm_synthetic — the card is fed the OFF-HARDWARE "
                               "MODEL'S OWN uniform stream instead of Ornith's activations, "
                               "so l2/rms are directly comparable with 8.406e-04 / 1.847e-03; "
                               "`gated` is NOT scored, its oracle is the real layer's."
                             : "");
    return 0;

fail:
    free(check->rms);
    free(check->rms_ready);
    free(check->host);
    free(check->want);
    free(check->synth);
    free(check);
    return -1;
}

void pn_qwen35_tt_gdn_normcheck_release(pn_qwen35_tt *g)
{
    if (!g || !g->gdn_normcheck)
        return;
    free(g->gdn_normcheck->rms);
    free(g->gdn_normcheck->rms_ready);
    free(g->gdn_normcheck->host);
    free(g->gdn_normcheck->want);
    free(g->gdn_normcheck->synth);
    free(g->gdn_normcheck);
    g->gdn_normcheck = NULL;
}

/** The q/k arm: ONE descriptor over q|k, which the row-block-per-head layout makes 32
 *  independent norms of 16 q-heads followed by 16 k-heads. */
static int normcheck_l2(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index)
{
    struct pn_qwen35_gdn_normcheck *check = g->gdn_normcheck;
    const int state = cfg->ssm_state;
    const int k_heads = cfg->ssm_n_group;
    const uint32_t count = (uint32_t)(2 * state * k_heads);
    const float *source = g->gdn_conv_out;
    double worst;
    int head;

    if (check->synthetic) {
        uint32_t element;

        for (element = 0; element < count; element++)
            check->synth[element] = normcheck_random_value();
        source = check->synth;
    }
    if (pn_tt_fp32_strip_upload(g->device, &check->qk, source, count) != 0 ||
        pn_tt_head_norm(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &check->l2,
                        (uint32_t)(2 * k_heads), &check->qk, &check->qk_out) != 0 ||
        pn_tt_fp32_strip_download(g->device, &check->qk_out, check->host, count) != 0)
        return -1;

    if (check->synthetic) {
        /* The same oracle over the synthetic stream — `pn_qwen35moe_l2_norm()` in place,
         * which is what `gdn_split_and_normalise()` calls on the real one. */
        memcpy(check->want, source, (size_t)count * sizeof(float));
        for (head = 0; head < 2 * k_heads; head++)
            pn_qwen35moe_l2_norm(check->want + (size_t)head * state, state,
                                 cfg->rms_norm_eps);
    } else {
        /* The oracle's answer is already computed: `gdn_split_and_normalise()` wrote it. */
        memcpy(check->want, g->gdn_qn, (size_t)state * k_heads * sizeof(float));
        memcpy(check->want + (size_t)state * k_heads, g->gdn_kn,
               (size_t)state * k_heads * sizeof(float));
    }
    worst = normcheck_score(check->want, check->host, 2 * k_heads, state,
                            &check->l2_over_gate);
    if (worst > check->worst_l2) {
        check->worst_l2 = worst;
        check->worst_l2_layer = ssm_index;
    }
    check->elements += (uint64_t)count;
    return 0;
}

/**
 * The gated arm, SCORED TWICE, because it is two descriptors on two residencies at two
 * formats and one number cannot say which of them costs what.
 *
 *   `worst_rms`    the Float32 reduce's own answer against `pn_vrms_norm_f32()`, read back
 *                  straight off the fp32 output strip. This is the number the Float32
 *                  program moves.
 *   `worst_gated`  the whole of `pn_qwen35moe_gated_norm()`: that answer rendered into a
 *                  **bf16** strip, multiplied by `silu(z)` on the bf16 ELEMENTWISE
 *                  residency, and read back. The extra term is the fp32 -> bf16 BRIDGE
 *                  between the two families, and it is a different item's to remove.
 */
static int normcheck_gated(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                           const pn_qwen35moe_config *cfg, int ssm_index, const float *z,
                           const float *gated)
{
    struct pn_qwen35_gdn_normcheck *check = g->gdn_normcheck;
    const int state = cfg->ssm_state;
    const int v_heads = cfg->ssm_dt_rank;
    const uint32_t count = (uint32_t)(state * v_heads);
    const float *source = g->gdn_y;
    double worst;
    int head;

    if (check->synthetic) {
        uint32_t element;

        for (element = 0; element < count; element++)
            check->synth[element] = normcheck_random_value();
        source = check->synth;
    }
    if (!check->rms_ready[ssm_index]) {
        if (pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM,
                                             l->ssm_norm, (uint32_t)state, cfg->rms_norm_eps,
                                             &check->rms[ssm_index]) != 0)
            return -1;
        check->rms_ready[ssm_index] = true;
    }
    if (pn_tt_fp32_strip_upload(g->device, &check->y, source, count) != 0 ||
        pn_tt_head_norm(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &check->rms[ssm_index],
                        (uint32_t)v_heads, &check->y, &check->y_out) != 0 ||
        pn_tt_fp32_strip_download(g->device, &check->y_out, check->host, count) != 0)
        return -1;

    /* The reduce alone. `pn_qwen35moe_gated_norm()`'s own first half, per head. */
    for (head = 0; head < v_heads; head++)
        pn_vrms_norm_f32(source + (size_t)head * state, l->ssm_norm,
                         check->want + (size_t)head * state, state, cfg->rms_norm_eps);
    worst = normcheck_score(check->want, check->host, v_heads, state, &check->rms_over_gate);
    if (worst > check->worst_rms) {
        check->worst_rms = worst;
        check->worst_rms_layer = ssm_index;
    }

    /* The gate's oracle is the REAL layer's `gated`, which a synthetic `y` does not produce.
     * Scoring it under this arm would be a number against the wrong answer, so it is skipped
     * by name rather than reported as a large one. */
    if (check->synthetic)
        return 0;

    /* Then the bridge and the gate. */
    if (pn_tt_activation_upload(g->device, &check->y_bridge, check->host, count) != 0 ||
        pn_tt_activation_upload(g->device, &check->z, z, count) != 0 ||
        pn_tt_silu_mul(g->device, &check->z, &check->y_bridge, &check->gated) != 0 ||
        pn_tt_activation_download(g->device, &check->gated, check->host, count) != 0)
        return -1;

    worst = normcheck_score(gated, check->host, v_heads, state, &check->gated_over_gate);
    if (worst > check->worst_gated) {
        check->worst_gated = worst;
        check->worst_gated_layer = ssm_index;
    }
    return 0;
}

int pn_qwen35_tt_gdn_normcheck_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                     const pn_qwen35moe_config *cfg, int ssm_index,
                                     const float *z, const float *gated)
{
    struct pn_qwen35_gdn_normcheck *check;

    if (!g || !g->gdn_normcheck || !l || !cfg || !z || !gated)
        return 0;
    check = g->gdn_normcheck;
    if (check->scored >= GDN_NORMCHECK_LAYER_BUDGET || ssm_index >= check->ssm_layers)
        return 0;
    if (normcheck_l2(g, cfg, ssm_index) != 0 ||
        normcheck_gated(g, l, cfg, ssm_index, z, gated) != 0) {
        fprintf(stderr, "qwen35_tt: the per-head norm probe failed at SSM layer %d — no "
                        "score from here on\n", ssm_index);
        check->scored = GDN_NORMCHECK_LAYER_BUDGET;
        return -1;
    }
    fprintf(stderr, "qwen35_tt: [normcheck] ssm %2d  l2 %.3e  rms %.3e  gated %.3e\n",
            ssm_index, check->worst_l2, check->worst_rms, check->worst_gated);
    check->scored++;
    return 0;
}

void pn_qwen35_tt_gdn_normcheck_report(const pn_qwen35_tt *g)
{
    const struct pn_qwen35_gdn_normcheck *check;

    if (!g || !g->gdn_normcheck)
        return;
    check = g->gdn_normcheck;
    fprintf(stderr,
            "qwen35_tt: per-head norms ON CARD vs the oracle over %d SSM layers — "
            "q/k L2 worst %.3e (layer %d), RMS worst %.3e (layer %d), gated worst %.3e "
            "(layer %d); past 1e-3 relative: %llu of %llu q/k, %llu RMS, %llu gated. The RMS "
            "figure is the Float32 reduce alone; gated adds the fp32->bf16 bridge into the "
            "bf16 elementwise gate.\n",
            check->scored, check->worst_l2, check->worst_l2_layer, check->worst_rms,
            check->worst_rms_layer, check->worst_gated, check->worst_gated_layer,
            (unsigned long long)check->l2_over_gate, (unsigned long long)check->elements,
            (unsigned long long)check->rms_over_gate,
            (unsigned long long)check->gated_over_gate);
}

#endif /* USE_TT */
