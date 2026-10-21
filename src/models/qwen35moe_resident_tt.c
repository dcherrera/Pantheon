/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_resident_tt.c
 * @brief The `qwen35` hybrid forward with its ACTIVATIONS resident in card DRAM.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS RUNG REMOVES, AND WHY IT IS EXACTLY THIS MUCH
 * ---------------------------------------------------------------------------
 * The rung below this one (`qwen35moe_forward_tt.c`'s projection-only arm) put all 249
 * matmuls on the card and left every activation on the host, so each projection's result
 * came back over a link measured at 4.97 MB/s. That is 6.39 MiB and 1.286 s a token — 56%
 * of it. The bytes decompose exactly, and the decomposition is what decides this file:
 *
 *   | consumer                            | B/token   | share |
 *   |-------------------------------------|-----------|-------|
 *   | the dense SwiGLU (gate, up, down)   | 3,670,016 | 54.8% |
 *   | the LM head's logit row             |   993,280 | 14.8% |
 *   | the Gated-Delta-Net's operands      | 1,185,792 | 17.7% |
 *   | ssm_out + attn_output               |   327,680 |  4.9% |
 *   | the attention block's q/k/v         |   524,288 |  7.8% |
 *
 * The first two, and the fourth, need NO arithmetic the card cannot already do: an RMSNorm
 * (the reduce residency), silu(gate)*up (the elementwise residency), a residual add (the
 * same residency) and an exact top-N (the sampling residency) are all shipped ops with red
 * proofs. Moving them is this file. That is 74.5% of the readback, and it does not depend on
 * the recurrence at all.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT DELIBERATELY DOES NOT REMOVE, AND THE WARNING THAT APPLIES
 * ---------------------------------------------------------------------------
 * The Gated-Delta-Net recurrence and the attention core still run on the HOST, so their
 * operands still cross: `attn_qkv`, `attn_gate`, `ssm_alpha` and `ssm_beta` come down and
 * `gated` goes back up; `attn_q`, `attn_k` and `attn_v` come down and `attn_out` goes back
 * up. **Each value crosses the link exactly once in each direction — nothing here is
 * downloaded and re-uploaded unchanged.**
 *
 * The standing warning is that on-card GDN and resident activations "must land together or
 * the second cancels the first". That is a statement about the FLOOR, and this file makes it
 * measurable rather than argued: the host recurrence goes from 17.7% of the readback to
 * ~78% of what is left, but the readback itself falls ~4.4x. So this rung is not cancelled —
 * what the host recurrence sets is the floor beneath it, and that floor cannot be lowered by
 * anything in this file.
 *
 * Lowering it means moving the WHOLE Gated-Delta-Net block, not a piece of it, and that is
 * arithmetic rather than preference. Per SSM layer the host takes over after four
 * projections and 49,408 B come down. After an on-card conv1d it is still 49,408 — a
 * depthwise convolution is width-preserving. After an on-card conv1d + L2 norm + GQA repeat
 * it is 65,536, because the interleaved repeat DOUBLES q and k. Only with the delta rule and
 * the gated norm on the card too does it collapse, to 256 B: `alpha` and `beta` down for the
 * host `softplus` (there is no SFPU `log`) and `decay` and `beta` back up.
 *
 * **THAT IS `PN_TT_GDN_STRIP` AND IT IS IMPLEMENTED** — `qwen35moe_gdn_strip_tt.c`. This file
 * carries both routes and the flag chooses: the block above the `gdn_strip` branch downloads
 * nothing but `alpha` and `beta`.
 *
 * @status The residual stream, both RMSNorms, the dense SwiGLU, the LM head and the exact
 *         top-N run on the card. The attention core runs on the host. The GDN recurrence runs
 *         on the host too UNLESS `PN_TT_ONCARD` names `gdn`, in which case its two heavy
 *         phases are card descriptors over state resident in card DRAM
 *         (`qwen35moe_gdn_tt.c`) — an arm that has never executed on silicon.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

#include "../accel/tt/pn_tt_dispatch.h"
#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "../accel/tt/pn_tt_head_norm.h"
#include "../accel/tt/pn_tt_replica_map.h"
#include "../core/bench.h"
#include "../memory/arena.h"

/** A strip is a whole number of 32x32 tiles. */
#define PN_QWEN35_TT_TILE 32u

/** Force the projection-only arm even when everything armed — the discriminator. */
#define PN_QWEN35_TT_RESIDENT_ENV "PN_TT_RESIDENT"

/* ============================================================
 * Bring-up
 * ============================================================ */

/** Tiles wide enough to hold `features`, which the geometry gate has already made a
 *  multiple of the tile width for every strip this file allocates. */
static uint32_t qwen35_tt_tiles(uint32_t features)
{
    return (features + PN_QWEN35_TT_TILE - 1u) / PN_QWEN35_TT_TILE;
}

/**
 * Allocate every strip ONCE, zeroed.
 *
 * The zeroing is load-bearing rather than tidy: a matvec writes only logical row 0 of each
 * tile and the FPU is fed all 32 rows anyway, so rows 1-31 have to be zero and have to stay
 * zero. Every op used here preserves that for free — a matmul of a zero row is zero,
 * silu(0)*0 is zero, 0+0 is zero, and an RMS norm of a zero row is 0 * gamma * rsqrt(eps).
 *
 * Widths come from the WEIGHTS rather than from the config wherever a weight defines them,
 * so a strip is exactly what its producer writes and its consumer reads. A disagreement
 * between the artifact and the engine is then a refusal at `pn_tt_matvec_*` rather than a
 * silently truncated projection.
 */
static int qwen35_tt_alloc_strips(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                                  const pn_qwen35_tt_layer *ssm,
                                  const pn_qwen35_tt_layer *attn)
{
    pn_qwen35_tt_strips *s = &g->card;
    uint32_t embd_tiles = qwen35_tt_tiles((uint32_t)cfg->n_embd);
    /* THE TWO HOT STRIPS, AND ONLY THOSE. Counted from this model's own shapes, A's DRAM
     * bytes are `normed` 70.6%, `ffn_swiglu` 17.6% and `ssm_gated`/`attn_out` 11.8% — and
     * these two are exactly the pair whose producers (the REDUCE and ELTWISE residencies)
     * drain through the one shared writer that can mirror them. `normed` is the RMS-norm
     * output that feeds all seven projections of every layer; `ffn_swiglu` feeds
     * `ffn_down`. Mirroring anything else would cost writes and buy a bank nobody is
     * queueing on. pn_tt_replica_map.h has the arithmetic. */
    uint32_t replicas;
    const pn_tt_activation *mirrored[2];

    /* TWO BANDS FOR THIS MODEL: a batched prefill pass carries 64 positions, halving the pass
     * count — 282.3 -> ~300 tok/s with the 8-set replica table keeping the eltwise split, every
     * arm IDENTICAL. Decode reads band 0 and is unchanged. This model runs both chunk paths,
     * which a second band requires; any other model keeps 1. `PN_TT_STRIP_TILE_ROWS` wins. */
    (void)pn_tt_device_prefer_strip_tile_rows(g->device, 2u);
    replicas = pn_tt_activation_replica_count(g->device);

    if (pn_tt_activation_alloc(g->device, embd_tiles, &s->hidden[0]) != 0 ||
        pn_tt_activation_alloc(g->device, embd_tiles, &s->hidden[1]) != 0 ||
        pn_tt_activation_alloc_replicated(g->device, embd_tiles, replicas, &s->normed) != 0 ||
        pn_tt_activation_alloc(g->device, embd_tiles, &s->project) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->attn_qkv.tile_cols, &s->ssm_qkv) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->attn_gate.tile_cols, &s->ssm_z) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->ssm_alpha.tile_cols, &s->ssm_alpha) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->ssm_beta.tile_cols, &s->ssm_beta) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->ssm_out.tile_rows, &s->ssm_gated) != 0 ||
        pn_tt_activation_alloc(g->device, attn->attn_q.tile_cols, &s->attn_qg) != 0 ||
        pn_tt_activation_alloc(g->device, attn->attn_k.tile_cols, &s->attn_k) != 0 ||
        pn_tt_activation_alloc(g->device, attn->attn_v.tile_cols, &s->attn_v) != 0 ||
        pn_tt_activation_alloc(g->device, attn->attn_output.tile_rows, &s->attn_out) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->ffn_gate.tile_cols, &s->ffn_gate) != 0 ||
        pn_tt_activation_alloc(g->device, ssm->ffn_up.tile_cols, &s->ffn_up) != 0 ||
        pn_tt_activation_alloc_replicated(g->device, ssm->ffn_down.tile_rows, replicas,
                                          &s->ffn_swiglu) != 0 ||
        pn_tt_activation_alloc(g->device, g->output.tile_cols, &s->logits) != 0)
        return -1;
    /* TELL THE PRODUCERS. Allocating mirrors is half the mechanism; without this the copies
     * are allocated, zeroed and never written again, and a matmul member reading one gets a
     * strip of zeros with every kernel completing and no error word set. The map is
     * published even when it is EMPTY, so a core the previous process left a map on is
     * provably cleared rather than assumed to be. */
    mirrored[0] = &s->normed;
    mirrored[1] = &s->ffn_swiglu;
    if (pn_tt_publish_replica_map(g->device, mirrored, 2u, &s->ffn_swiglu) != 0)
        return -1;
    s->slot = 0;
    return 0;
}

/** Expand every norm weight into the card-side constants the reduce op reads. Once, at
 *  load: a per-token upload of gamma would put back a transfer this file exists to remove. */
static int qwen35_tt_upload_norms(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                  const pn_qwen35moe_config *cfg)
{
    uint32_t width = (uint32_t)cfg->n_embd;
    int layer;

    for (layer = 0; layer < cfg->n_layer; layer++) {
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        const float *pre;
        const float *post;

        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
            const pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer(m, idx);
            if (!l)
                return -1;
            pre = l->attn_norm;
            post = l->post_attention_norm;
        } else {
            const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, idx);
            if (!l)
                return -1;
            pre = l->attn_norm;
            post = l->post_attention_norm;
        }
        if (pn_tt_norm_constants_upload(g->device, pre, width, cfg->rms_norm_eps,
                                        &g->layers[layer].attn_norm) != 0 ||
            pn_tt_norm_constants_upload(g->device, post, width, cfg->rms_norm_eps,
                                        &g->layers[layer].post_norm) != 0)
            return -1;
    }
    return pn_tt_norm_constants_upload(g->device, pn_qwen35moe_model_output_norm(m), width,
                                       cfg->rms_norm_eps, &g->output_norm);
}

/** The host buffers the two host-side blocks read their projections into. Allocated once —
 *  a per-token malloc of 32 KiB inside a 32-layer loop is not free on a 2-core host. */
static int qwen35_tt_alloc_host_scratch(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                                        const pn_qwen35_tt_layer *ssm,
                                        const pn_qwen35_tt_layer *attn)
{
    g->h_qkv = calloc((size_t)ssm->attn_qkv.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_z = calloc((size_t)ssm->attn_gate.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_alpha = calloc((size_t)ssm->ssm_alpha.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_beta = calloc((size_t)ssm->ssm_beta.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_gated = calloc((size_t)ssm->ssm_out.tile_rows * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_qg = calloc((size_t)attn->attn_q.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_k = calloc((size_t)attn->attn_k.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_v = calloc((size_t)attn->attn_v.tile_cols * PN_QWEN35_TT_TILE, sizeof(float));
    g->h_attn = calloc((size_t)attn->attn_output.tile_rows * PN_QWEN35_TT_TILE,
                       sizeof(float));
    g->h_embed = calloc((size_t)cfg->n_embd, sizeof(float));
    return (g->h_qkv && g->h_z && g->h_alpha && g->h_beta && g->h_gated && g->h_qg &&
            g->h_k && g->h_v && g->h_attn && g->h_embed) ? 0 : -1;
}

/**
 * Find the first layer of each kind, so the strip widths can come from a real weight.
 *
 * A hybrid stack has both, and this model's own load-time gate has already refused a
 * geometry where either count is zero. Returning a refusal rather than assuming layer 0 is
 * an SSM layer is what keeps this file honest about a family it has not seen.
 */
static int qwen35_tt_representative_layers(const pn_qwen35_tt *g,
                                           const pn_qwen35_tt_layer **ssm,
                                           const pn_qwen35_tt_layer **attn)
{
    int layer;

    *ssm = NULL;
    *attn = NULL;
    for (layer = 0; layer < g->n_layers; layer++) {
        if (g->layers[layer].is_attention) {
            if (!*attn)
                *attn = &g->layers[layer];
        } else if (!*ssm) {
            *ssm = &g->layers[layer];
        }
    }
    return (*ssm && *attn) ? 0 : -1;
}

/**
 * PM 7 ITEM 5: arm the GDN GATE family (the gates computed on the card) when the chunkwise
 * prefill and its tall strips are up. A failure is SAID and leaves the host gate path, which is
 * the pre-item-5 arithmetic (libm) — so a run that did not arm is not the new reference.
 */
static void qwen35_tt_gdngate_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                  const pn_qwen35moe_config *cfg)
{
    const int n_ssm = pn_qwen35moe_model_n_ssm(m);
    const float **ssm_a, **ssm_dt;
    int i;

    if (!pn_tt_gdnc_is_armed(g->device) || g->tall_rows == 0u || n_ssm <= 0)
        return;
    ssm_a = calloc((size_t)n_ssm, sizeof(*ssm_a));
    ssm_dt = calloc((size_t)n_ssm, sizeof(*ssm_dt));
    for (i = 0; ssm_a && ssm_dt && i < n_ssm; i++) {
        const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, i);

        ssm_a[i] = l ? l->ssm_a : NULL;
        ssm_dt[i] = l ? l->ssm_dt : NULL;
        if (!ssm_a[i] || !ssm_dt[i])
            break;
    }
    if (!ssm_a || !ssm_dt || i < n_ssm ||
        pn_tt_gdngate_arm(g->device, (uint32_t)n_ssm, (uint32_t)cfg->ssm_dt_rank, ssm_a, ssm_dt,
                          g->tall_rows * 32u, pn_tt_gdnc_compact_scale_bits(g->device)) != 0)
        fprintf(stderr, "qwen35_tt: the GDN gates stay on the HOST (gdngate did not arm) — "
                        "alpha/beta read back as before, libm arithmetic\n");
    else {
        const char *sweep = getenv("PN_TT_GDNGATE_SWEEP");

        g->gdn_det_gates = true;
        fprintf(stderr, "qwen35_tt: GDN gates ON THE CARD (ft_det_math) — prefill, verify and "
                        "decode read no alpha/beta back\n");
        if (sweep && atoi(sweep) > 0)
            (void)pn_tt_gdngate_sweep(g->device, (uint32_t)atoi(sweep));
    }
    free(ssm_a);
    free(ssm_dt);
}

int pn_qwen35_tt_resident_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                              const pn_qwen35moe_config *cfg)
{
    const pn_qwen35_tt_layer *ssm;
    const pn_qwen35_tt_layer *attn;
    uint32_t eltwise_tiles[3];
    uint32_t eltwise_count = 2u;
    const char *want;

    if (!g || !g->device || !m || !cfg || !g->layers)
        return -1;

    want = getenv(PN_QWEN35_TT_RESIDENT_ENV);
    if (want && *want && atoi(want) == 0) {
        fprintf(stderr, "qwen35_tt: %s=0 — the resident-activation arm is OFF by request; "
                        "every projection round-trips through the host\n",
                PN_QWEN35_TT_RESIDENT_ENV);
        return 0;
    }
    if (qwen35_tt_representative_layers(g, &ssm, &attn) != 0) {
        fprintf(stderr, "qwen35_tt: the resident arm needs one layer of each kind and this "
                        "stack has only one — staying on the projection-only arm\n");
        return 0;
    }

    /* Every width an elementwise DESCRIPTOR will carry. Both residual adds are n_embd wide
     * and the SwiGLU is ffn wide; one residency serves both, at one block depth that has to
     * divide each of them (and twice itself into each) — pn_tt_device.h states the
     * invariant and the core it cost. */
    eltwise_tiles[0] = qwen35_tt_tiles((uint32_t)cfg->n_embd);
    eltwise_tiles[1] = ssm->ffn_gate.tile_cols;
    /* ON-CARD ATTENTION IS ON BY DEFAULT FOR THIS FAMILY AND FOR NO OTHER, and it is declared
     * HERE because a default belongs to a model rather than to the card. It passes a
     * calibrated chat-parity gate on this hybrid (0.986080, disagree 2.76%, +0.4% perplexity)
     * and is faster at every length measured; the same op family through `llama_forward_tt.c`
     * is recorded as FAILING that gate on MiniCPM5-1B, so the default cannot be global.
     * An explicit `PN_TT_ONCARD` list still wins over this. */
    pn_tt_device_oncard_default(g->device, PN_TT_ONCARD_ATTENTION, true);

    /* ON-CARD ATTENTION'S SIGMOID GATE IS A THIRD WIDTH, and it has to be declared HERE
     * rather than where it is used: the residency runs ONE block depth for every descriptor
     * it will ever see, so a width discovered after `place_cbs` is a descriptor the depth
     * cannot serve. `n_head*head_dim` is the gate's width and it is not `n_embd` on this
     * model — Ornith is 32 x 256 = 8,192 against 4,096.
     *
     * The question is only answerable because the call above re-read the selection: this ran
     * BEFORE anything populated `oncard_requested`, so it used to read a calloc'd false and
     * the third width was never declared at all. */
    if (pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_ATTENTION))
        eltwise_tiles[eltwise_count++] =
            qwen35_tt_tiles((uint32_t)(cfg->n_head * cfg->head_dim));
    if (pn_tt_device_arm_oncard(g->device, qwen35_tt_tiles((uint32_t)cfg->n_embd),
                                eltwise_tiles, eltwise_count, g->output.tile_cols) != 0)
        return 0;

    /* All three or none. A mixed arm would mean a per-op host fallback that downloads and
     * re-uploads the very strip this file exists to keep resident, which is slower than the
     * arm below it AND harder to attribute. A clean refusal is worth more than a fast-looking
     * hybrid. */
    if (!pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM) ||
        !pn_tt_device_oncard(g->device, PN_TT_ONCARD_SWIGLU) ||
        !pn_tt_device_oncard(g->device, PN_TT_ONCARD_RESIDUAL)) {
        fprintf(stderr, "qwen35_tt: the norm / SwiGLU / residual families did not all arm "
                        "(norm %d, swiglu %d, residual %d) — staying on the projection-only "
                        "arm\n",
                (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_NORM),
                (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_SWIGLU),
                (int)pn_tt_device_oncard(g->device, PN_TT_ONCARD_RESIDUAL));
        return 0;
    }

    if (qwen35_tt_alloc_strips(g, cfg, ssm, attn) != 0 ||
        qwen35_tt_upload_norms(g, m, cfg) != 0 ||
        qwen35_tt_alloc_host_scratch(g, cfg, ssm, attn) != 0) {
        fprintf(stderr, "qwen35_tt: the resident arm could not be staged — staying on the "
                        "projection-only arm\n");
        return 0;
    }

    /* A family that was ASKED for and did not arm is a refusal, not a configuration — say so
     * once rather than quietly reading the whole logit row back and reporting a slow token
     * nobody can attribute. */
    g->sparse_capable = pn_tt_device_oncard(g->device, PN_TT_ONCARD_TOPK);
    if (g->sparse_logits && !g->sparse_capable) {
        fprintf(stderr, "qwen35_tt: %s was asked for and the sampling residency did NOT arm "
                        "— the whole %u-tile logit row comes back instead\n",
                PN_QWEN35_TT_SPARSE_ENV, g->output.tile_cols);
        g->sparse_logits = false;
    }
    /* The Gated-Delta-Net block is the LAST family to arm and the only one that may fail
     * FATALLY: everything above it degrades to a host op, but a half-staged GDN residency
     * would leave some layers' carried state on the card and some on the host. */
    if (pn_qwen35_tt_gdn_arm(g, m, cfg) != 0) {
        fprintf(stderr, "qwen35_tt: the on-card Gated-Delta-Net arm was asked for and could "
                        "not be staged — the recurrence stays on the host\n");
        pn_qwen35_tt_gdn_release(g);
    }
    /* THE STRIP BRIDGE, on top of that arm and never instead of it. A refusal leaves the
     * host-mediated recurrence running, which is slower and not wrong. */
    if (pn_qwen35_tt_gdn_strip_arm(g, cfg) != 0) {
        fprintf(stderr, "qwen35_tt: the strip bridge was asked for and could not be staged — "
                        "the block stays on the host-mediated path\n");
        pn_qwen35_tt_gdn_strip_release(g);
    }
    /* THE DEFAULT-VERSUS-NAMED SPLIT, AND HERE IT IS NOT THEORETICAL. The GDN block WITHOUT
     * the bridge is a measured wall REGRESSION — 1.43 tok/s against the control's 2.20 —
     * because the per-head norms on the host force `conv_out` and `y` across the link as well.
     * So a DEFAULT arm that loses the bridge gives the recurrence back to the host rather than
     * shipping the slow half of the block; an arm the operator NAMED keeps it, because that
     * arm is the correctness vehicle they asked for and substituting it quietly would be a
     * different forward. The card-side residency stays armed and simply goes unused. */
    if (g->gdn_on_card && !g->gdn_strip &&
        !pn_tt_device_oncard_named(g->device, PN_TT_ONCARD_GDN)) {
        fprintf(stderr, "qwen35_tt: the Gated-Delta-Net block is on by DEFAULT and the strip "
                        "bridge did not arm. Without it the block is 1.43 tok/s against 2.20, "
                        "so the recurrence goes back to the host. Name `gdn` in PN_TT_ONCARD "
                        "to run it anyway.\n");
        pn_qwen35_tt_gdn_release(g);
    }
    if (pn_qwen35_tt_gdn_parity_arm(g, cfg) != 0)
        return -1;
    /* ON-CARD ATTENTION IS A REFUSAL WHEN IT WAS NAMED, AND A DEGRADATION WHEN IT IS ONLY THE
     * DEFAULT. Every family above it falls back to a host op with the same arithmetic; this
     * one moves the KV cache into card DRAM at bf16 and changes the accumulation order, so a
     * half-armed attention asked for BY NAME is a forward that is partly one model and partly
     * another and must not be substituted.
     *
     * But `return -1` here reaches `goto fail` in pn_qwen35moe_tt_upload, and that drops the
     * WHOLE model to the CPU forward — on Ornith, ~0.05 tok/s instead of the host-attention
     * arm's 2.2. That is the right price for a request the operator made explicitly and the
     * wrong one for a default, so the two are split rather than sharing a flag. The split is
     * inert while attention is opt-in; it is the prerequisite for defaulting it on. */
    if (pn_qwen35_tt_attn_arm(g, m, cfg) != 0) {
        pn_qwen35_tt_attn_release(g);
        if (pn_tt_device_oncard_named(g->device, PN_TT_ONCARD_ATTENTION)) {
            fprintf(stderr, "qwen35_tt: on-card attention was asked for BY NAME and did not "
                            "arm — refusing rather than silently running the host block with "
                            "a different KV cache\n");
            return -1;
        }
        fprintf(stderr, "qwen35_tt: on-card attention is on by DEFAULT and did not arm — "
                        "running the HOST attention block instead. The answer is correct and "
                        "O(context) on the host; it is not the card arm.\n");
    }
    /* STEP 4 S2b: the MTP draft head, after the attention arm reserved its KV region. A
     * failure leaves speculation to the n-gram drafter. */
    if (pn_qwen35_tt_attn_is_on_card(g) && pn_qwen35_tt_mtp_arm(g, m, cfg) != 0) {
        fprintf(stderr, "qwen35_tt: the MTP draft head did not arm — no MTP drafts\n");
        g->mtp.on = false;
    }
    if (pn_qwen35_tt_attn_is_on_card(g)) {
        /* Only the CROSSED CONTROL needs a host copy of the card's answer, and only when it
         * is asked for — the whole point of the arm is that nothing comes back. */
        if (getenv("PN_TT_ATTN_COMPARE")) {
            g->h_attn_card = calloc((size_t)cfg->n_head * (size_t)cfg->head_dim,
                                    sizeof(float));
            if (!g->h_attn_card)
                return -1;
        }
    }
    if (pn_qwen35_tt_prefill_tall_alloc(g) != 0)
        return -1;
    qwen35_tt_gdngate_arm(g, m, cfg);
    g->resident = true;
    fprintf(stderr, "qwen35_tt: resident activations ARMED — the residual stream, both "
                    "RMSNorms, the dense SwiGLU and the LM head stay in card DRAM; the "
                    "logit row %s\n",
            g->sparse_logits ? "stays there too (exact top-N crosses instead)"
                             : "still comes back in full");
    return 0;
}

void pn_qwen35_tt_resident_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    pn_qwen35_tt_gdn_parity_report(g);
    pn_qwen35_tt_gdn_normcheck_report(g);
    pn_qwen35_tt_gdn_parity_release(g);
    pn_qwen35_tt_gdn_strip_release(g);
    pn_qwen35_tt_gdn_release(g);
    pn_qwen35_tt_attn_release(g);
    free(g->h_qkv);
    free(g->h_z);
    free(g->h_alpha);
    free(g->h_beta);
    free(g->h_gated);
    free(g->h_qg);
    free(g->h_k);
    free(g->h_v);
    free(g->h_attn);
    free(g->h_embed);
    g->h_qkv = g->h_z = g->h_alpha = g->h_beta = g->h_gated = NULL;
    g->h_qg = g->h_k = g->h_v = g->h_attn = g->h_embed = NULL;
}

/* ============================================================
 * The forward
 * ============================================================ */

/** The strip currently carrying the residual stream, and the one the next add writes. */
static pn_tt_activation *qwen35_tt_hidden(pn_qwen35_tt *g)
{
    return &g->card.hidden[g->card.slot];
}

static pn_tt_activation *qwen35_tt_hidden_spare(pn_qwen35_tt *g)
{
    return &g->card.hidden[g->card.slot ^ 1];
}

/** hidden = hidden + project, on the card, into the spare strip. */
static int qwen35_tt_residual(pn_qwen35_tt *g)
{
    if (pn_tt_add(g->device, qwen35_tt_hidden(g), &g->card.project,
                  qwen35_tt_hidden_spare(g)) != 0)
        return -1;
    g->card.slot ^= 1;
    return 0;
}

/** One projection, ISSUED. Counted here because this is the only place the resident arm
 *  asks the card for a matmul, and the count is the evidence that it did. */
int pn_qwen35_tt_project(pn_qwen35_tt *g, const pn_tt_weight *w,
                         const pn_tt_activation *in, const pn_tt_activation *out, bool wait)
{
    int rc = wait ? pn_tt_matvec_resident(g->device, w, in, out)
                  : pn_tt_matvec_enqueue(g->device, w, in, out);

    if (rc != 0) {
        g->mv_failed++;
        if (!g->failure_reported) {
            g->failure_reported = 1;
            fprintf(stderr, "qwen35_tt: a card projection FAILED inside the resident "
                            "forward. There is no host arm for it at this rung, so the "
                            "token is refused rather than quietly computed elsewhere.\n");
        }
        return -1;
    }
    g->mv_on_card++;
    return 0;
}

/** ITEM 6: `PN_TT_GDNGATE_DECODE=0` keeps det gates but computes them on the host from read
 *  alpha/beta (same bits; the A/B for the card path's speed). */
static bool qwen35_tt_card_decode_gates(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_GDNGATE_DECODE");

        cached = (env && *env) ? (atoi(env) != 0) : 1;
    }
    return cached != 0;
}

/**
 * The Gated-Delta-Net block: four projections out, the recurrence on the host, one back.
 *
 * The four are INDEPENDENT — one input strip, four disjoint output strips — so they go in
 * the ring together and the host waits once. Four is exactly `PN_TT_MATVEC_MAX_CHAIN`.
 */
static int qwen35_tt_ssm_block(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                               const pn_qwen35_tt_layer *d,
                               const pn_qwen35moe_config *cfg, int ssm_index,
                               pn_qwen35moe_ssm_state *state, pn_arena *arena,
                               struct fw_pool *pool)
{
    pn_qwen35_tt_strips *s = &g->card;
    uint32_t gated_n = (uint32_t)(cfg->ssm_state * cfg->ssm_dt_rank);

    if (pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false) != 0)
        return -1;

    /* THE STRIP BRIDGE: `ssm_qkv` and `ssm_z` DO NOT COME DOWN. The block runs whole on the
     * card and the only two things that cross are `alpha` and `beta` — one tile each, 256 B —
     * because `sigmoid` and `exp(a * softplus(.))` need a `pow` the SFPU has no `log` for.
     * That is 49,408 B a layer removed; see qwen35moe_gdn_strip_tt.c. */
    if (g->gdn_strip) {
        /* ITEM 6: with det gates the card computes them — nothing is read back. */
        const bool card = g->gdn_det_gates && qwen35_tt_card_decode_gates();
        double started_ms;
        int rc;

        if (!card &&
            (pn_tt_activation_download(g->device, &s->ssm_alpha, g->h_alpha,
                                       s->ssm_alpha.tile_count * PN_QWEN35_TT_TILE) != 0 ||
             pn_tt_activation_download(g->device, &s->ssm_beta, g->h_beta,
                                       s->ssm_beta.tile_count * PN_QWEN35_TT_TILE) != 0))
            return -1;
        started_ms = pn_time_ms();
        rc = pn_qwen35_tt_gdn_strip_layer(g, l, cfg, ssm_index, &s->ssm_qkv, &s->ssm_z,
                                          card ? NULL : g->h_beta, card ? NULL : g->h_alpha,
                                          &s->ssm_gated);
        g->host_recurrence_seconds += (pn_time_ms() - started_ms) / 1000.0;
        if (rc != 0)
            return -1;
        (void)gated_n;
        (void)state;
        (void)arena;
        (void)pool;
        return pn_qwen35_tt_project(g, &d->ssm_out, &s->ssm_gated, &s->project, true);
    }

    /* The four downloads flush the chain themselves — pn_tt_device.h's one-seam rule. */
    if (pn_tt_activation_download(g->device, &s->ssm_qkv, g->h_qkv,
                                  s->ssm_qkv.tile_count * PN_QWEN35_TT_TILE) != 0 ||
        pn_tt_activation_download(g->device, &s->ssm_z, g->h_z,
                                  s->ssm_z.tile_count * PN_QWEN35_TT_TILE) != 0 ||
        pn_tt_activation_download(g->device, &s->ssm_alpha, g->h_alpha,
                                  s->ssm_alpha.tile_count * PN_QWEN35_TT_TILE) != 0 ||
        pn_tt_activation_download(g->device, &s->ssm_beta, g->h_beta,
                                  s->ssm_beta.tile_count * PN_QWEN35_TT_TILE) != 0)
        return -1;

    /* The CPU oracle's own conv1d, L2 norms, interleaved GQA repeat, delta rule and gated
     * norm, on the oracle's own fp32 state. Shared code, not a reimplementation — which is
     * what makes a disagreement with the oracle attributable to the matmuls. */
    {
        double started_ms = pn_time_ms();
        /* The on-card arm carries the state in card DRAM, so it takes neither `state` nor the
         * arena; the two device descriptors it issues are timed by the on-card counter and
         * what lands HERE is only the host arithmetic left around them. */
        int recurrence_rc =
            g->gdn_on_card
                ? pn_qwen35_tt_gdn_recurrence(g, l, cfg, ssm_index, g->h_qkv, g->h_z,
                                              g->h_beta, g->h_alpha, g->h_gated)
                : pn_qwen35moe_ssm_recurrence(l, cfg, g->h_qkv, g->h_z, g->h_beta, g->h_alpha,
                                              state, arena, g->h_gated, pool);
        g->host_recurrence_seconds += (pn_time_ms() - started_ms) / 1000.0;
        if (recurrence_rc < 0)
            return -1;
    }

    /* The on-card GATE, and it runs OUTSIDE the recurrence timer on purpose: it is a second
     * full host recurrence, so timing it inside would report the arm as three times its own
     * cost. A no-op unless PN_TT_GDN_PARITY armed it. */
    if (g->gdn_on_card && pn_qwen35_tt_gdn_parity_layer(g, l, cfg, ssm_index, g->h_qkv,
                                                        g->h_z, g->h_beta, g->h_alpha, state,
                                                        arena, pool) != 0)
        return -1;

    if (pn_tt_activation_upload(g->device, &s->ssm_gated, g->h_gated, gated_n) != 0)
        return -1;
    return pn_qwen35_tt_project(g, &d->ssm_out, &s->ssm_gated, &s->project, true);
}

/**
 * The full-attention block: three projections out, the attention core on the host, one back.
 * Q/K/V are independent, so they chain.
 *
 * THIS IS THE SHIPPED ARM AND THE ONE THE ORACLE WAS TAKEN ON. It is also the reference the
 * crossed control diffs the card arm against, which is why it is left whole and reachable
 * rather than replaced: `qwen35moe_attn_tt.c` runs BESIDE it under `PN_TT_ATTN_COMPARE`, not
 * instead of it.
 */
static int qwen35_tt_attn_block(pn_qwen35_tt *g, const pn_qwen35moe_attn_layer *l,
                                const pn_qwen35_tt_layer *d,
                                const pn_qwen35moe_config *cfg, pn_qwen35moe_kv *kv,
                                int pos, pn_arena *arena, struct fw_pool *pool)
{
    pn_qwen35_tt_strips *s = &g->card;
    uint32_t attn_n = (uint32_t)(cfg->n_head * cfg->head_dim);

    if (pn_qwen35_tt_project(g, &d->attn_q, &s->normed, &s->attn_qg, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_k, &s->normed, &s->attn_k, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_v, &s->normed, &s->attn_v, false) != 0)
        return -1;

    if (pn_tt_activation_download(g->device, &s->attn_qg, g->h_qg,
                                  s->attn_qg.tile_count * PN_QWEN35_TT_TILE) != 0 ||
        pn_tt_activation_download(g->device, &s->attn_k, g->h_k,
                                  s->attn_k.tile_count * PN_QWEN35_TT_TILE) != 0 ||
        pn_tt_activation_download(g->device, &s->attn_v, g->h_v,
                                  s->attn_v.tile_count * PN_QWEN35_TT_TILE) != 0)
        return -1;

    {
        double started_ms = pn_time_ms();
        int attention_rc = pn_qwen35moe_attn_core(l, cfg, g->h_qg, g->h_k, g->h_v, kv, pos,
                                                  arena, g->h_attn, pool);
        g->host_attention_seconds += (pn_time_ms() - started_ms) / 1000.0;
        if (attention_rc < 0)
            return -1;
    }

    if (pn_tt_activation_upload(g->device, &s->attn_out, g->h_attn, attn_n) != 0)
        return -1;
    return pn_qwen35_tt_project(g, &d->attn_output, &s->attn_out, &s->project, true);
}

/** The dense SwiGLU FFN, entirely on the card: gate and up chain, silu(gate)*up is one
 *  descriptor rather than two, and down writes straight into the residual's operand. */
static int qwen35_tt_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d)
{
    pn_qwen35_tt_strips *s = &g->card;

    if (pn_qwen35_tt_project(g, &d->ffn_gate, &s->normed, &s->ffn_gate, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_up, &s->normed, &s->ffn_up, false) != 0)
        return -1;
    if (pn_tt_matvec_flush(g->device) != 0)
        return -1;
    if (pn_tt_silu_mul(g->device, &s->ffn_gate, &s->ffn_up, &s->ffn_swiglu) != 0)
        return -1;
    return pn_qwen35_tt_project(g, &d->ffn_down, &s->ffn_swiglu, &s->project, true);
}

/** One layer: attn_norm -> (SSM | attention) -> +residual -> post_norm -> FFN -> +residual.
 *  The same order `qwen35moe_forward.c` runs, with the norms and the adds on the card. */
static int qwen35_tt_layer(pn_qwen35_tt *g, pn_qwen35moe_model *m, int layer,
                           const pn_qwen35moe_config *cfg, pn_qwen35moe_state *st, int pos,
                           struct fw_pool *pool)
{
    const pn_qwen35_tt_layer *d = &g->layers[layer];
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    int rc;

    if (pn_tt_rms_norm(g->device, &d->attn_norm, qwen35_tt_hidden(g), &g->card.normed) != 0)
        return -1;
    pn_arena_reset(st->arena);
    if (d->is_attention) {
        const pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer(m, idx);
        if (!l)
            return -1;
        if (pn_qwen35_tt_attn_is_on_card(g)) {
            const int max_seq = pn_qwen35moe_max_seq(cfg);
            const int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;

            /* THE CROSSED CONTROL, and the ORDER is the whole of its correctness. The host
             * block runs FIRST and leaves its answer in `g->h_attn`; the card block then
             * runs over freshly re-projected q/k/v and OVERWRITES `s->project`, so the
             * forward carries the CARD's answer and the host's is the shadow. Running it the
             * other way round would silently make the compare arm a host forward — a
             * measurement of the thing not under test.
             *
             * Both arms maintain their own KV cache: the host's `st->kv[idx]` and the card's
             * region. Neither is ever resynced from the other, which is what makes a
             * divergence at position N still visible at position N+1. */
            if (g->attn_compare &&
                qwen35_tt_attn_block(g, l, d, cfg, &st->kv[idx], pos, st->arena, pool) != 0)
                return -1;
            rc = pn_qwen35_tt_attn_block_card(g, d, cfg, idx, pos, seq_len);
            if (rc == 0 && g->attn_compare)
                pn_qwen35_tt_attn_compare(g, cfg, layer, g->h_attn);
        } else {
            rc = qwen35_tt_attn_block(g, l, d, cfg, &st->kv[idx], pos, st->arena, pool);
        }
    } else {
        const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, idx);
        if (!l)
            return -1;
        rc = qwen35_tt_ssm_block(g, l, d, cfg, idx, &st->ssm[idx], st->arena, pool);
    }
    if (rc != 0 || qwen35_tt_residual(g) != 0)
        return -1;

    if (pn_tt_rms_norm(g->device, &d->post_norm, qwen35_tt_hidden(g), &g->card.normed) != 0)
        return -1;
    if (qwen35_tt_ffn(g, d) != 0)
        return -1;
    return qwen35_tt_residual(g);
}

/**
 * The LM head, and the one place the logit row's 993,280 B a token is decided.
 *
 * With the sampling residency armed, the card selects the exact top-N and ~96 bytes of
 * (index, value) pairs cross instead of 248,320 floats; the row handed back is the same
 * sparse presentation the K80 path uses, `-INFINITY` everywhere the card did not name. The
 * comparator is exact — highest value wins, lowest index breaks a tie — so a greedy argmax
 * over this row is the argmax over the dense one, not an approximation of it.
 */
static int qwen35_tt_head_with(pn_qwen35_tt *g, pn_qwen35moe_state *st,
                               const pn_tt_norm_constants *norm)
{
    pn_qwen35_tt_strips *s = &g->card;
    uint32_t i;
    double started_ms;

    if (pn_tt_rms_norm(g->device, norm, qwen35_tt_hidden(g), &s->normed) != 0)
        return -1;
    if (pn_qwen35_tt_project(g, &g->output, &s->normed, &s->logits, true) != 0)
        return -1;

    if (!g->sparse_logits)
        return pn_tt_activation_download(g->device, &s->logits, st->logits,
                                         (uint32_t)st->n_vocab);

    g->topk_count = 0;
    if (pn_tt_topk(g->device, &s->logits, g->topk_pairs ? g->topk_pairs : PN_QWEN35_TT_TOPK_GREEDY,
                   g->topk_index, g->topk_value, &g->topk_count) != 0 || g->topk_count == 0u)
        return -1;

    /* `PN_TT_TOPN_CROSSCHECK=1`: the card's top-N against a host selection over the dense row
     * (value desc, index asc) — same ids, same values, same order. Instrumentation only. */
    if (getenv("PN_TT_TOPN_CROSSCHECK")) {
        static unsigned long checked, bad;
        float *dense = malloc((size_t)st->n_vocab * sizeof(float));

        if (dense && pn_tt_activation_download(g->device, &s->logits, dense,
                                               (uint32_t)st->n_vocab) == 0) {
            uint32_t k;
            int ok = 1;

            for (k = 0; k < g->topk_count && ok; k++) {
                uint32_t v, rank = 0;
                const uint32_t id = g->topk_index[k];

                if (id >= (uint32_t)st->n_vocab || dense[id] != g->topk_value[k]) { ok = 0; break; }
                for (v = 0; v < (uint32_t)st->n_vocab; v++)
                    if (dense[v] > dense[id] || (dense[v] == dense[id] && v < id))
                        rank++;
                if (rank != k) ok = 0;
            }
            checked++;
            if (!ok) bad++;
            if ((checked & 63u) == 0u || !ok)
                fprintf(stderr, "qwen35_tt: TOPN CROSSCHECK — %lu rows, %lu mismatched (N %u)\n",
                        checked, bad, g->topk_count);
        }
        free(dense);
    }
    /* 248,320 floats a token written twice — once to -INFINITY here, once more when the
     * caller copies the row into the engine's tensor. The card sent 96 bytes; the host still
     * touches 993,280. Timed because it is host work between two seams and therefore in
     * NONE of the device counters. */
    started_ms = pn_time_ms();
    for (i = 0; i < (uint32_t)st->n_vocab; i++)
        st->logits[i] = -INFINITY;
    for (i = 0; i < g->topk_count; i++) {
        if (g->topk_index[i] < (uint32_t)st->n_vocab)
            st->logits[g->topk_index[i]] = g->topk_value[i];
    }
    g->host_logit_seconds += (pn_time_ms() - started_ms) / 1000.0;
    return 0;
}

static int qwen35_tt_head(pn_qwen35_tt *g, pn_qwen35moe_state *st)
{
    return qwen35_tt_head_with(g, st, &g->output_norm);
}

/* S2c: the verify's heads in ONE pass — rows 0..rows-1 of `hidden` through the final norm and
 * the LM head together (one weight stream; the half-tile levers hold for <= 16 rows), then a
 * top-k per row (the kernel's row select) and the engine's greedy rule on each. */
int pn_qwen35_tt_head_rows(pn_qwen35_tt *g, pn_qwen35moe_state *st,
                           const pn_tt_activation *hidden, int rows, uint32_t *argmax_out)
{
    pn_qwen35_tt_strips *s = &g->card;
    int r, rc = 0;

    if (rows < 1 || rows > 16 || !g->sparse_logits)
        return -1;
    pn_tt_device_set_verify_rows(g->device, (uint32_t)rows);
    if (pn_tt_rms_norm(g->device, &g->output_norm, hidden, &s->normed) != 0 ||
        pn_qwen35_tt_project(g, &g->output, &s->normed, &s->logits, true) != 0)
        rc = -1;
    pn_tt_device_set_verify_rows(g->device, 0u);
    for (r = 0; rc == 0 && r < rows; r++) {
        uint32_t i;
        int best = -1;

        pn_tt_device_set_topk_row(g->device, (uint32_t)r);
        g->topk_count = 0;
        if (pn_tt_topk(g->device, &s->logits,
                       g->topk_pairs ? g->topk_pairs : PN_QWEN35_TT_TOPK_GREEDY, g->topk_index,
                       g->topk_value, &g->topk_count) != 0 || g->topk_count == 0u) {
            rc = -1;
            break;
        }
        /* The greedy rule over the sparse row: largest value, lowest index on a tie. */
        {
            float bv = -INFINITY;

            for (i = 0; i < g->topk_count; i++) {
                const int idx = (int)g->topk_index[i];
                const float v = g->topk_value[i];

                if (idx >= st->n_vocab)
                    continue;
                if (best < 0 || v > bv || (v == bv && idx < best)) {
                    best = idx;
                    bv = v;
                }
            }
        }
        if (best < 0) {
            rc = -1;
            break;
        }
        argmax_out[r] = (uint32_t)best;
        memcpy(g->spec_topk_index[r], g->topk_index, sizeof(g->spec_topk_index[r]));
        memcpy(g->spec_topk_value[r], g->topk_value, sizeof(g->spec_topk_value[r]));
        g->spec_topk_count[r] = g->topk_count;
    }
    pn_tt_device_set_topk_row(g->device, 0u);
    return rc;
}

int pn_qwen35_tt_head_row(pn_qwen35_tt *g, pn_qwen35moe_state *st)
{
    int rc;

    if (!getenv("PN_TT_HEAD_ROW0") || atoi(getenv("PN_TT_HEAD_ROW0")) != 0)
        pn_tt_device_set_single_row(g->device, true);
    rc = qwen35_tt_head(g, st);
    pn_tt_device_set_single_row(g->device, false);
    return rc;
}

/**
 * DFlash acceptance measurement (`PN_TT_DUMP_HIDDEN=<file>`): after layers 1,5,...,29 (the
 * drafter's `target_layer_ids`, HF `hidden_states[l+1]`) append this position's residual
 * row, plus layer 31 (the last: final norm + LM head must reproduce the next token, the dump's
 * own check). Record: int32 pos, int32 token, then 9 x n_embd float32. Instrumentation only — every
 * record is a card read; run it with PN_TT_SPEC=0 PN_TT_PREFILL=0 PN_TT_DISPATCH=0.
 */
static void qwen35_tt_dump_hidden(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int layer,
                                  uint32_t token, int pos)
{
    static FILE *out;
    static int armed = -1;
    static float *row;
    int32_t head[2];

    if (armed < 0) {
        const char *path = getenv("PN_TT_DUMP_HIDDEN");

        armed = 0;
        if (path && *path) {
            out = fopen(path, "wb");
            row = calloc((size_t)cfg->n_embd, sizeof(float));
            armed = (out && row) ? 1 : 0;
            if (!armed)
                fprintf(stderr, "qwen35_tt: PN_TT_DUMP_HIDDEN %s could not be opened\n", path);
        }
    }
    if (!armed || layer < 1 || ((layer - 1) % 4 != 0 && layer != 31) || layer > 31)
        return;
    if (layer == 1) {
        head[0] = pos;
        head[1] = (int32_t)token;
        fwrite(head, sizeof(head), 1, out);
    }
    if (pn_tt_activation_download(g->device, &g->card.hidden[g->card.slot], row,
                                  (uint32_t)cfg->n_embd) != 0)
        memset(row, 0, (size_t)cfg->n_embd * sizeof(float));
    fwrite(row, sizeof(float), (size_t)cfg->n_embd, out);
    if (layer == 31)
        fflush(out);
}

float *pn_qwen35_tt_resident_forward(pn_qwen35_tt *g, pn_qwen35moe_model *m, uint32_t token,
                                     pn_qwen35moe_state *st, int pos, struct fw_pool *pool)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int layer;

    if (!cfg || (int)token >= cfg->n_vocab)
        return NULL;

    /* THE CARD'S CARRIED STATE IS PER SEQUENCE and there is one region set per model, so a
     * new sequence has to zero it exactly where the host arm gets a freshly calloc'd
     * `pn_qwen35moe_ssm_state`. Position 0 is that seam. A caller resuming mid-sequence from
     * a prefix cache would not pass through it — stated because nothing here can detect it. */
    if (g->gdn_on_card && pos == 0 && pn_tt_gdn_state_reset(g->device) != 0)
        return NULL;
    /* STEP 3 (A): the dispatch recorder sees this whole forward, embedding to top-k. */
    pn_tt_dispatch_token_begin(g->device, g->tokens + 1u);

    /* The embedding table is not on the card — it is tiled block-float and no on-card op
     * reads it — so the residual stream starts with one upload a token. 4,096 bf16 at the
     * fast direction of the link is under 20 microseconds. */
    {
        double started_ms = pn_time_ms();
        pn_qwen35moe_model_embed_row(m, token, g->h_embed);
        g->host_embed_seconds += (pn_time_ms() - started_ms) / 1000.0;
    }
    g->card.slot = 0;
    if (pn_tt_activation_upload(g->device, &g->card.hidden[0], g->h_embed,
                                (uint32_t)cfg->n_embd) != 0)
        return NULL;

    /* ONE TOKEN: every strip's live row is row 0 (PHASE E row-0 path). Cleared on every exit
     * so a batched prefill after this never inherits the promise. */
    pn_tt_device_set_single_row(g->device, true);
    for (layer = 0; layer < cfg->n_layer; layer++) {
        if (qwen35_tt_layer(g, m, layer, cfg, st, pos, pool) != 0) {
            pn_tt_device_set_single_row(g->device, false);
            return NULL;
        }
        qwen35_tt_dump_hidden(g, cfg, layer, token, pos);
    }
    pn_tt_device_set_single_row(g->device, false);
    pn_qwen35_tt_gdn_parity_token(g, pos);
    /* STEP 3a: `PN_TT_HEAD_ROW0` (default 1; 0 = off) keeps the single-row promise over the head too (final
     * norm, LM head, top-k) — the LM head then reads row-0 half tiles like every projection. */
    if (!getenv("PN_TT_HEAD_ROW0") || atoi(getenv("PN_TT_HEAD_ROW0")) != 0)
        pn_tt_device_set_single_row(g->device, true);
    if (qwen35_tt_head(g, st) != 0) {
        pn_tt_device_set_single_row(g->device, false);
        pn_tt_dispatch_token_end(g->device, g->tokens + 1u);
        return NULL;
    }
    pn_tt_device_set_single_row(g->device, false);
    pn_tt_dispatch_token_end(g->device, g->tokens + 1u);
    return st->logits;
}

/* =======================================================================================
 * STEP 4 S2b — THE MTP DRAFT HEAD
 *
 * Ornith's KL-distilled nextn block (protoLabsAI/Ornith-1.0-9B-MTP-GGUF), converted by
 * tools/build_mtp_gguf.py + ft_tt_convert_gguf into a second artifact. One position:
 *   x = eh_proj_e @ enorm(emb(t_{p+1})) + eh_proj_h @ hnorm(h_p)
 *   x = x + attn(attn_norm(x)) at p (its own KV region);  x = x + ffn(post_norm(x))
 *   draft = argmax(lm_head(shared_head_norm(x)))   — the trunk's LM head (byte-identical)
 * and x chains as the next position's h. It only proposes: the verify decides every token.
 * ======================================================================================= */

static int qwen35_tt_mtp_norm(pn_qwen35_tt *g, const char *name, uint32_t n, float eps,
                              pn_tt_norm_constants *out)
{
    const float *v = pn_tt_device_extra_vector(g->device, name, n);

    return v ? pn_tt_norm_constants_upload(g->device, v, n, eps, out) : -1;
}

void pn_qwen35_tt_mtp_resolve(pn_qwen35_tt *g, const char *trunk_artifact)
{
    const char *env = getenv("PN_TT_MTP");
    const char *spec = getenv("PN_TT_SPEC");
    size_t n;

    g->mtp.path[0] = '\0';
    /* SHIPPED DEFAULT (2026-10-06, David): MTP speculative decoding on. Off with
     * `PN_TT_SPEC=0` (no MTP arms either) or `PN_TT_MTP=` (empty: no drafts from the head). */
    if (spec && atoi(spec) == 0)
        return;
    if (env) {
        snprintf(g->mtp.path, sizeof(g->mtp.path), "%s", env);
        return;
    }
    if (!trunk_artifact || (n = strlen(trunk_artifact)) < 5u ||
        strcmp(trunk_artifact + n - 5u, ".fttw") != 0 || n + 5u >= sizeof(g->mtp.path))
        return;
    snprintf(g->mtp.path, sizeof(g->mtp.path), "%.*s.mtp.fttw", (int)(n - 5u), trunk_artifact);
    if (access(g->mtp.path, R_OK) != 0) {
        fprintf(stderr, "qwen35_tt: no MTP artifact at %s — plain decode (no speculative "
                        "drafts from the head)\n", g->mtp.path);
        g->mtp.path[0] = '\0';
    }
}

int pn_qwen35_tt_mtp_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg)
{
    const char *path = g->mtp.path;
    pn_qwen35_tt_layer *d = &g->mtp.layer;
    const uint32_t n_embd = (uint32_t)cfg->n_embd, head_dim = (uint32_t)cfg->head_dim;
    const float *qn, *kn;
    struct { const char *name; pn_tt_weight *w; } place[] = {
        { "mtp.eh_proj_e.weight", &g->mtp.eh_e },  { "mtp.eh_proj_h.weight", &g->mtp.eh_h },
        { "mtp.attn_k.weight", &d->attn_k },       { "mtp.attn_v.weight", &d->attn_v },
        { "mtp.attn_output.weight", &d->attn_output },
        { "mtp.ffn_gate.weight", &d->ffn_gate },   { "mtp.ffn_up.weight", &d->ffn_up },
        { "mtp.ffn_down.weight", &d->ffn_down },
    };
    size_t i;

    (void)m;
    g->mtp.on = false;
    if (!path || !*path)
        return 0;
    if (!g->attn_on_card) {
        fprintf(stderr, "qwen35_tt: MTP refused — it needs on-card attention\n");
        return -1;
    }
    memset(d, 0, sizeof(*d));
    d->is_attention = true;
    if (pn_tt_device_extra_open(g->device, path) != 0)
        return -1;
    for (i = 0; i < sizeof(place) / sizeof(place[0]); i++)
        if (pn_tt_device_place_extra_weight(g->device, place[i].name, place[i].w) != 0)
            return -1;
    if (pn_qwen35_tt_attn_split_q(g, cfg, "mtp.attn_q.weight", true, d) != 0)
        return -1;
    qn = pn_tt_device_extra_vector(g->device, "mtp.attn_q_norm.weight", head_dim);
    kn = pn_tt_device_extra_vector(g->device, "mtp.attn_k_norm.weight", head_dim);
    if (!qn || !kn ||
        pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, qn, head_dim,
                                         cfg->rms_norm_eps, &d->attn_q_norm) != 0 ||
        pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, kn, head_dim,
                                         cfg->rms_norm_eps, &d->attn_k_norm) != 0 ||
        qwen35_tt_mtp_norm(g, "mtp.attn_norm.weight", n_embd, cfg->rms_norm_eps,
                           &d->attn_norm) != 0 ||
        qwen35_tt_mtp_norm(g, "mtp.post_attention_norm.weight", n_embd, cfg->rms_norm_eps,
                           &d->post_norm) != 0 ||
        qwen35_tt_mtp_norm(g, "mtp.enorm.weight", n_embd, cfg->rms_norm_eps,
                           &g->mtp.enorm) != 0 ||
        qwen35_tt_mtp_norm(g, "mtp.hnorm.weight", n_embd, cfg->rms_norm_eps,
                           &g->mtp.hnorm) != 0 ||
        qwen35_tt_mtp_norm(g, "mtp.shared_head_norm.weight", n_embd, cfg->rms_norm_eps,
                           &g->mtp.head_norm) != 0)
        return -1;
    if (pn_tt_activation_alloc(g->device, qwen35_tt_tiles(n_embd), &g->mtp.h) != 0 ||
        pn_tt_activation_alloc(g->device, qwen35_tt_tiles(n_embd), &g->mtp.e) != 0)
        return -1;
    g->mtp.sub = false;
    /* DEFAULT 2048 tiles (ids < 65,536): run_s2c5.sh — acceptance unchanged (0.81 short /
     * 0.75 long), MTP 27.3 -> 18.3 ms a cycle; 1024 lost acceptance (0.73). 0 = whole head. */
    if (!getenv("PN_TT_MTP_VOCAB_TILES") || atoi(getenv("PN_TT_MTP_VOCAB_TILES")) > 0) {
        const uint32_t total = g->output.tile_cols;
        const uint32_t n = getenv("PN_TT_MTP_VOCAB_TILES")
                               ? (uint32_t)atoi(getenv("PN_TT_MTP_VOCAB_TILES")) : 2048u;
        const uint32_t tail = 248044u / 32u; /* Ornith's special tokens start at 248044 */
        pn_tt_column_run runs[2] = { { 0u, n }, { tail, total - tail } };
        float *ninf = malloc((size_t)total * 32u * sizeof(float));
        uint32_t i;

        if (!ninf || n == 0u || n >= tail || tail >= total ||
            pn_tt_device_stage_weight_columns(g->device, "output.weight", runs, 2u,
                                              &g->mtp.head_sub) != 0 ||
            pn_tt_activation_alloc(g->device, total, &g->mtp.logits_sub) != 0) {
            free(ninf);
            return -1;
        }
        for (i = 0; i < total * 32u; i++)
            ninf[i] = -INFINITY;
        if (pn_tt_activation_upload(g->device, &g->mtp.logits_sub, ninf, total * 32u) != 0) {
            free(ninf);
            return -1;
        }
        free(ninf);
        g->mtp.sub_tiles = n;
        g->mtp.sub_tail_first = tail;
        g->mtp.sub = true;
        fprintf(stderr, "qwen35_tt: MTP draft vocabulary — ids < %u + the %u-tile special tail "
                        "(%u of %u LM-head columns)\n", n * 32u, total - tail,
                n + total - tail, total);
    }
    g->mtp.pend_n = 0;
    g->mtp.on = true;
    fprintf(stderr, "qwen35_tt: MTP draft head ARMED from %s (KV region %d)\n", path,
            g->mtp.attn_index);
    return 0;
}

/** The draft head on the decode hidden (row 0): shared_head_norm, the (draft-vocabulary) LM
 *  head, top-k, the greedy rule. */
/** The best value's softmax over the top-k values: an upper bound on its full-vocabulary
 *  probability, enough to say a draft is unlikely to be accepted. */
static float qwen35_tt_topk_confidence(const float *values, uint32_t count, float best)
{
    double sum = 0.0;
    uint32_t i;

    for (i = 0; i < count; i++)
        sum += exp((double)(values[i] - best));
    return sum > 0.0 ? (float)(1.0 / sum) : 1.0f;
}

static int qwen35_tt_mtp_head(pn_qwen35_tt *g, pn_qwen35moe_state *st, uint32_t *argmax_out)
{
    pn_qwen35_tt_strips *s = &g->card;
    int rc = 0;

    if (g->mtp.sub) {
        /* The draft vocabulary: subset logits into the -inf strip (the subset's columns are
         * its first tiles), top-k, the best pair mapped back to its vocabulary id. */
        uint32_t i;
        int best = -1;
        float bv = -INFINITY;

        if (pn_tt_rms_norm(g->device, &g->mtp.head_norm, qwen35_tt_hidden(g), &s->normed) != 0 ||
            pn_qwen35_tt_project(g, &g->mtp.head_sub, &s->normed, &g->mtp.logits_sub, true) != 0)
            rc = -1;
        g->topk_count = 0;
        if (rc == 0 && (pn_tt_topk(g->device, &g->mtp.logits_sub, PN_QWEN35_TT_TOPK_GREEDY,
                                   g->topk_index, g->topk_value, &g->topk_count) != 0 ||
                        g->topk_count == 0u))
            rc = -1;
        for (i = 0; rc == 0 && i < g->topk_count; i++) {
            const uint32_t sub = g->topk_index[i];
            const int id = (int)(sub < g->mtp.sub_tiles * 32u
                                     ? sub
                                     : g->mtp.sub_tail_first * 32u + (sub - g->mtp.sub_tiles * 32u));

            if (id >= st->n_vocab)
                continue;
            if (best < 0 || g->topk_value[i] > bv || (g->topk_value[i] == bv && id < best)) {
                best = id;
                bv = g->topk_value[i];
            }
        }
        if (rc == 0 && best < 0)
            rc = -1;
        if (rc == 0 && argmax_out)
            *argmax_out = (uint32_t)best;
        if (rc == 0)
            g->mtp.last_conf = qwen35_tt_topk_confidence(g->topk_value, g->topk_count, bv);
        g->mtp.heads++;
    } else {
        int v, best = 0;

        rc = qwen35_tt_head_with(g, st, &g->mtp.head_norm);
        for (v = 1; rc == 0 && v < st->n_vocab; v++)
            if (st->logits[v] > st->logits[best])
                best = v;
        if (rc == 0 && argmax_out)
            *argmax_out = (uint32_t)best;
        g->mtp.last_conf = 1.0f;
        g->mtp.heads++;
    }
    return rc;
}

int pn_qwen35_tt_mtp_head_row(pn_qwen35_tt *g, pn_qwen35moe_state *st,
                              const pn_tt_activation *src, uint32_t row, uint32_t *argmax_out)
{
    const double started = pn_time_ms();
    int rc;

    if (!g->mtp.on || pn_tt_row_copy(g->device, src, row, &g->card.hidden[g->card.slot], 0u) != 0 ||
        pn_tt_row_copy(g->device, src, row, &g->mtp.h, 0u) != 0)
        return -1;
    pn_tt_device_set_single_row(g->device, true);
    rc = qwen35_tt_mtp_head(g, st, argmax_out);
    pn_tt_device_set_single_row(g->device, false);
    g->mtp.ms += pn_time_ms() - started;
    return rc;
}

int pn_qwen35_tt_mtp_step(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg,
                          pn_qwen35moe_state *st, uint32_t token, const pn_tt_activation *h_src,
                          uint32_t h_row, int pos, bool want_head, uint32_t *argmax_out)
{
    pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35_tt_layer *d = &g->mtp.layer;
    const int max_seq = pn_qwen35moe_max_seq(cfg);
    const double started = pn_time_ms();
    int rc = -1;

    if (!g->mtp.on || pos < 0 || pos >= max_seq)
        return -1;
    if (h_src && pn_tt_row_copy(g->device, h_src, h_row, &g->mtp.h, 0u) != 0)
        return -1;
    pn_qwen35moe_model_embed_row(m, token, g->h_embed);
    if (pn_tt_activation_upload(g->device, &g->mtp.e, g->h_embed, (uint32_t)cfg->n_embd) != 0)
        return -1;
    pn_tt_device_set_single_row(g->device, true);
    s->slot = 0;
    if (pn_tt_rms_norm(g->device, &g->mtp.hnorm, &g->mtp.h, &s->normed) == 0 &&
        pn_qwen35_tt_project(g, &g->mtp.eh_h, &s->normed, &s->project, true) == 0 &&
        pn_tt_rms_norm(g->device, &g->mtp.enorm, &g->mtp.e, &s->normed) == 0 &&
        pn_qwen35_tt_project(g, &g->mtp.eh_e, &s->normed, &s->hidden[0], true) == 0 &&
        qwen35_tt_residual(g) == 0 &&
        pn_tt_rms_norm(g->device, &d->attn_norm, qwen35_tt_hidden(g), &s->normed) == 0 &&
        pn_qwen35_tt_attn_block_card(g, d, cfg, g->mtp.attn_index, pos, pos + 1) == 0 &&
        qwen35_tt_residual(g) == 0 &&
        pn_tt_rms_norm(g->device, &d->post_norm, qwen35_tt_hidden(g), &s->normed) == 0 &&
        qwen35_tt_ffn(g, d) == 0 && qwen35_tt_residual(g) == 0 &&
        pn_tt_row_copy(g->device, qwen35_tt_hidden(g), 0u, &g->mtp.h, 0u) == 0)
        rc = 0;
    if (rc == 0 && want_head)
        rc = qwen35_tt_mtp_head(g, st, argmax_out);
    pn_tt_device_set_single_row(g->device, false);
    g->mtp.steps++;
    g->mtp.ms += pn_time_ms() - started;
    return rc;
}

#endif /* USE_TT */
