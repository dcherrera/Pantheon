/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gdn_tt.c
 * @brief The Gated-Delta-Net recurrence with its two heavy phases ON THE CARD — the caller
 *        `foundry/tt/ft_tt_op_gdn.h` was written for, and the only thing that stands between
 *        the triad and its first instruction on silicon.
 *
 * ---------------------------------------------------------------------------
 * THE SAME FIVE STEPS AS THE ORACLE, IN THE SAME ORDER, WITH TWO OF THEM MOVED
 * ---------------------------------------------------------------------------
 * `pn_qwen35moe_ssm_recurrence()` in qwen35moe_ssm.c IS the definition of this block, value
 * for value, and it does five things between the input projections and the output one:
 *
 *   1. beta = sigmoid(beta_raw)                        host, and stays host
 *   2. g = ssm_a * softplus(alpha + ssm_dt)            host, and CANNOT move: `decay =
 *                                                      exp(g)` needs a `pow` and so a `log`,
 *                                                      and the SFPU has neither
 *   3. conv1d + SiLU over conv_dim, window rolled      -> PHASE CONV
 *   4. split q|k|v, L2-norm each q/k head, GQA repeat  host for now (see below); the repeat
 *                                                      is never materialised — it is the
 *                                                      reader's `vh % k_heads`
 *   5. the delta rule over every v-head                -> PHASE DELTA
 *   6. gated norm: RMSNorm(y, ssm_norm) * silu(z)      host for now
 *
 * Steps 1, 2, 4 and 6 call the ORACLE'S OWN functions rather than reimplementing them —
 * `pn_qwen35moe_beta_sigmoid`, `pn_qwen35moe_decay_gate`, `pn_qwen35moe_l2_norm`,
 * `pn_qwen35moe_gated_norm`. That is what makes a disagreement between this arm and the CPU
 * one attributable to the two descriptors and to nothing else, which is the whole point of
 * the gate this file exists to enable.
 *
 * ---------------------------------------------------------------------------
 * WHY 4 AND 6 ARE STILL ON THE HOST, AND WHAT THAT COSTS
 * ---------------------------------------------------------------------------
 * Both are per-head norms at `S/32` = 4 tile columns against the model's 128, and a reduce
 * residency is placed for exactly ONE width. `pn_tt_head_norm.c` is that second residency
 * and it is ARMED — but they still cannot move, and the reason is not the width.
 *
 * **MEASURED 2026-08-19, `PN_TT_GDN_NORMCHECK=1`, and it settles two things the scoping got
 * wrong in opposite directions.** The LAYOUT is free: a reduce with `row_blocks = heads`
 * over a `[1, heads*S]` strip is one norm per head with no reshape at all, because the
 * reader indexes row block `b` at tiles `b*tile_cols`. The PRECISION is not: the shipped
 * reduce family is bf16, and on Ornith's real per-layer data the card scores 8.599e-03 on
 * the q/k L2 norm and 1.283e-02 on the gated norm against these very functions, 9x and 13x
 * past the 1e-3 gate. The lever is the CB's format — the reduce alone is 6.1e-03 of it —
 * so the norms move when there is a **Float32 `reduce` program**, and not before.
 *
 * ---------------------------------------------------------------------------
 * AND THE NORMS DID NOT MOVE ALONE — THE PLAN ITEM WAS NOT SEPARABLE
 * ---------------------------------------------------------------------------
 * Moving them removes `conv_out` (32,768 B a layer) and `y` (16,384 B) from the readback,
 * which takes this arm from 2630.09 KiB/token back to the control's 1478.09 — PARITY, not a
 * gain. The gain is the OTHER 49,408 B a layer, and that needs `ssm_qkv` and `ssm_z` to stop
 * being downloaded too. **That is `qwen35moe_gdn_strip_tt.c`**, and it took a FORMAT bridge
 * (two render residencies, `pn_tt_render.h`) and a LAYOUT change (PHASE DELTA reads head-major
 * row-0 strips, so conv_out IS the delta input) rather than the bf16 GDN operand this comment
 * used to predict — a mixed CB table demotes the whole program's fp32 unpack, so the GDN op
 * cannot read a bf16 strip at any price.
 *
 * **THIS FILE IS THE HOST-MEDIATED ARM AND IT IS STILL SLOWER, ON PURPOSE.** Because the L2
 * norm sits between the two phases, `conv_out` has to come back (32,768 B a layer) and `y`
 * has to come back (16,384 B), on top of the 49,408 B of projections. So it roughly DOUBLES
 * the readback — ~1.18 MB a token more — to remove 0.0421 s a token of host delta-rule
 * arithmetic, and at the measured 5.26 MB/s that is a net LOSS of ~0.18 s a token. It is the
 * "conv1d alone is 1.00x" row of the split-point table from the other side, and it is why
 * this path is opt-in: it is the CORRECTNESS vehicle for the on-card gate — the arm that
 * calls the ORACLE'S OWN functions around the two descriptors, so a disagreement is
 * attributable to them and to nothing else. The speed rung is the strip bridge beside it.
 *
 * @status RUN ON SILICON 2026-08-18 and NOT YET CORRECT. Opt-in with
 *         `PN_TT_ONCARD=norm,swiglu,residual,topk,gdn`; `PN_TT_GDN_PARITY=1` runs the oracle
 *         beside it and prices the disagreement (`qwen35moe_gdn_parity_tt.c`).
 * @issues The six host steps around the two descriptors are the oracle's own and are not in
 *         question.
 *
 *         AT POSITION 0 BOTH PHASES ARE NOW INSIDE OR NEAR THE GATE. PHASE CONV is 1.6e-04 to
 *         8.7e-04 across all 24 SSM layers; PHASE DELTA agrees with its closed form on all 32
 *         heads at 2.7e-03 to 7.0e-03, uniformly, with a systematic `card/want` of 0.9955 to
 *         0.9977 that is Tf32 truncation over the delta rule's six unpack stages and NOT a
 *         layout fault. `gated` at layer 0 is 6.636e-04.
 *
 *         THE 48-TOKEN `cmp` PASSES AS OF 2026-08-19 — 189 B, md5 80fc6259, IDENTICAL to the
 *         oracle and to the host-recurrence control, at 0.99 tok/s against the control's 1.44
 *         (expected: the two per-head norms have not moved yet). The defect that had been
 *         failing it was the conv window's NEWEST plane, wrong on every CONV descriptor but
 *         the first; it was CB_IN carrying two producers across the two phases and it is
 *         fixed in foundry. `conv` is 8.677e-04 at position 0 and 1.098e-03 to 1.790e-03
 *         after it, and what is left in both phases is Tf32 truncation over the unpack
 *         stages, not layout.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdn_conv_share.h"
#include "../accel/tt/pn_tt_gdnc.h"

int pn_qwen35_tt_gdn_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                         const pn_qwen35moe_config *cfg)
{
    pn_tt_gdn_shape shape;
    int layer;
    int ssm_layers = 0;

    if (!g || !g->device || !m || !cfg)
        return -1;
    if (!pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_GDN))
        return 0;

    for (layer = 0; layer < cfg->n_layer; layer++) {
        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
            ssm_layers++;
    }
    if (ssm_layers == 0)
        return 0;

    shape.state_size = (uint32_t)cfg->ssm_state;
    shape.v_heads = (uint32_t)cfg->ssm_dt_rank;
    shape.k_heads = (uint32_t)cfg->ssm_n_group;
    shape.conv_dim = (uint32_t)cfg->conv_dim;
    shape.conv_taps = (uint32_t)cfg->ssm_conv;
    shape.layer_count = (uint32_t)ssm_layers;
    /* Option D, under the fit policy only: one conv weight for both GDN families. */
    pn_tt_gdn_conv_share_request(g->device, pn_tt_gdn_conv_share_wanted(g->fit_policy));
    if (pn_tt_device_arm_gdn(g->device, &shape) != 0)
        return -1;
    if (!pn_tt_gdn_is_armed(g->device))
        return 0;
    /* THE CHUNKWISE FAMILY ARMS BEHIND THE DECODE ONE AND ONLY BEHIND IT — it advances the
     * SAME conv window and the SAME recurrent state, so there is nothing for it to carry
     * without them. Default OFF; `PN_TT_PREFILL_GDNC=1` asks for it. It is never fatal: an
     * unarmed chunkwise family leaves the prefill on the per-position path. */
    if (pn_tt_device_arm_gdnc(g->device) != 0 || pn_tt_gdn_conv_share_bind(g->device) != 0)
        return -1;

    /* The conv weight is a per-layer CONSTANT and the transpose into tap-major planes is a
     * load-time cost, not a per-token one. Doing it per token would put a 32 KiB gather and a
     * 4 MiB write inside a 24-layer loop. */
    for (layer = 0; layer < cfg->n_layer; layer++) {
        const pn_qwen35moe_ssm_layer *l;
        int idx;

        if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN)
            continue;
        idx = pn_qwen35moe_model_layer_index(m, layer);
        l = pn_qwen35moe_model_ssm_layer(m, idx);
        if (!l || !l->ssm_conv1d ||
            pn_tt_gdn_conv_weight_upload(g->device, (uint32_t)idx, l->ssm_conv1d) != 0) {
            fprintf(stderr, "qwen35_tt: SSM layer %d's conv weight did not stage — the GDN "
                            "block stays on the host\n", layer);
            return -1;
        }
        /* THE SAME TRANSPOSE AT `valid_rows` 32 INSTEAD OF 1, into the chunkwise family's own
         * constant, plus the band tiles behind it. Two copies of one weight, and the second
         * exists so the SHIPPED decode plane is not touched by prefill work at all. */
        if (pn_tt_gdnc_is_armed(g->device) &&
            pn_tt_gdnc_conv_weight_upload(g->device, (uint32_t)idx, l->ssm_conv1d) != 0) {
            /* Shared, that constant IS the decode weight: nothing else will write it. */
            if (pn_tt_gdn_conv_weight_is_shared(g->device)) {
                fprintf(stderr, "qwen35_tt: SSM layer %d's shared conv weight did not stage — "
                                "the GDN block stays on the host\n", layer);
                return -1;
            }
            fprintf(stderr, "qwen35_tt: SSM layer %d's ROW-REPLICATED conv weight did not "
                            "stage — the chunkwise mixers are unusable and prefill stays "
                            "per-position\n", layer);
            pn_tt_gdnc_release(g->device);
        }
    }

    g->gdn_conv_out = calloc((size_t)cfg->conv_dim, sizeof(float));
    g->gdn_qn = calloc((size_t)cfg->ssm_state * (size_t)cfg->ssm_n_group, sizeof(float));
    g->gdn_kn = calloc((size_t)cfg->ssm_state * (size_t)cfg->ssm_n_group, sizeof(float));
    g->gdn_y = calloc((size_t)cfg->ssm_state * (size_t)cfg->ssm_dt_rank, sizeof(float));
    g->gdn_decay = calloc((size_t)cfg->ssm_dt_rank, sizeof(float));
    g->gdn_beta = calloc((size_t)cfg->ssm_dt_rank, sizeof(float));
    if (!g->gdn_conv_out || !g->gdn_qn || !g->gdn_kn || !g->gdn_y || !g->gdn_decay ||
        !g->gdn_beta)
        return -1;

    g->gdn_on_card = true;
    /* The PROBE, not the path: it needs `gdn_conv_out`/`gdn_y` allocated above, and it is
     * armed last so a failure there leaves the arm itself intact. */
    if (pn_qwen35_tt_gdn_normcheck_arm(g, cfg) != 0)
        fprintf(stderr, "qwen35_tt: the per-head norm probe did not arm; the recurrence is "
                        "unaffected\n");
    fprintf(stderr, "qwen35_tt: the Gated-Delta-Net block's two heavy phases run ON THE CARD "
                    "for %d SSM layers — two descriptors a layer, state resident, `cmp` "
                    "IDENTICAL to the oracle. WITHOUT PN_TT_GDN_STRIP the two per-head norms "
                    "stay on the host, so `conv_out` and `y` cross as well and this arm is "
                    "SLOWER than the host recurrence: it is the correctness vehicle, not a "
                    "speed rung.\n", ssm_layers);
    return 0;
}

void pn_qwen35_tt_gdn_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    pn_qwen35_tt_gdn_normcheck_release(g);
    free(g->gdn_conv_out);
    free(g->gdn_qn);
    free(g->gdn_kn);
    free(g->gdn_y);
    free(g->gdn_decay);
    free(g->gdn_beta);
    g->gdn_conv_out = g->gdn_qn = g->gdn_kn = g->gdn_y = NULL;
    g->gdn_decay = g->gdn_beta = NULL;
    g->gdn_on_card = false;
}

/**
 * The host half of step 4: split the convolved stream and L2-normalise each q and k HEAD.
 *
 * The GQA repeat the oracle does here is deliberately NOT done: `q` and `k` leave this
 * function with ONE row per k-head, which is what the DELTA reader takes, and the repeat is
 * its `vh % k_heads`. Materialising it would double both streams — the 1.33x-worse row of
 * the split-point table.
 */
static void gdn_split_and_normalise(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    const int state = cfg->ssm_state;
    const int k_heads = cfg->ssm_n_group;
    const size_t head_bytes = (size_t)state * sizeof(float);
    const float *q_src = g->gdn_conv_out;
    const float *k_src = g->gdn_conv_out + (size_t)state * (size_t)k_heads;
    int head;

    for (head = 0; head < k_heads; head++) {
        memcpy(g->gdn_qn + (size_t)head * state, q_src + (size_t)head * state, head_bytes);
        memcpy(g->gdn_kn + (size_t)head * state, k_src + (size_t)head * state, head_bytes);
        pn_qwen35moe_l2_norm(g->gdn_qn + (size_t)head * state, state, cfg->rms_norm_eps);
        pn_qwen35moe_l2_norm(g->gdn_kn + (size_t)head * state, state, cfg->rms_norm_eps);
    }
}

int pn_qwen35_tt_gdn_recurrence(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                const pn_qwen35moe_config *cfg, int ssm_index,
                                const float *qkv, const float *z, const float *beta_raw,
                                const float *alpha, float *gated)
{
    const int v_heads = cfg->ssm_dt_rank;
    const int key_dim = cfg->ssm_state * cfg->ssm_n_group;
    int head;

    if (!g || !l || !cfg || !qkv || !z || !beta_raw || !alpha || !gated)
        return -1;
    if (!g->gdn_on_card || ssm_index < 0)
        return -1;

    /* 1 and 2, the oracle's own, on the oracle's own inputs. `decay = exp(g)` is applied
     * HERE rather than on the card because the card has no `log` and therefore no `pow`; the
     * op takes the answer as 32 fp32 scalars, 128 B a layer. */
    pn_qwen35moe_beta_sigmoid(beta_raw, v_heads, g->gdn_beta);
    pn_qwen35moe_decay_gate(l, alpha, v_heads, g->gdn_decay);
    for (head = 0; head < v_heads; head++)
        g->gdn_decay[head] = expf(g->gdn_decay[head]);

    /* 3. PHASE CONV — conv1d + SiLU over the whole stream, and the window rolls on the card. */
    if (pn_tt_gdn_conv(g->device, (uint32_t)ssm_index, qkv, g->gdn_conv_out) != 0)
        return -1;

    /* 4. split and L2-normalise, WITHOUT the repeat. */
    gdn_split_and_normalise(g, cfg);

    /* 5. PHASE DELTA — v is the third slice of the convolved stream and crosses untouched. */
    if (pn_tt_gdn_delta(g->device, (uint32_t)ssm_index, g->gdn_qn, g->gdn_kn,
                        g->gdn_conv_out + 2 * key_dim, g->gdn_decay, g->gdn_beta,
                        g->gdn_y) != 0)
        return -1;

    /* 6. the gated norm, the oracle's own. */
    pn_qwen35moe_gated_norm(l, cfg, g->gdn_y, z, gated);

    /* AFTER 4 and 6, because it scores the card's answer against what they just wrote. */
    (void)pn_qwen35_tt_gdn_normcheck_layer(g, l, cfg, ssm_index, z, gated);
    return 0;
}

#endif /* USE_TT */
