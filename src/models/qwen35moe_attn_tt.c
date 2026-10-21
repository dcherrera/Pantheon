/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_attn_tt.c
 * @brief Ornith's FULL-ATTENTION block on the card: the de-interleaved Q/gate projections,
 *        the per-head QK-RMSNorms, partial NeoX RoPE, the KV append, flash decode and the
 *        SIGMOID output gate — with the KV cache itself living in card DRAM.
 *
 *        The op family is `src/accel/tt/pn_tt_attention.c` and it already existed, green on
 *        silicon, for MiniCPM5-1B. This file is what Ornith's SHAPE needs on top of it.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS THE PHASE WHOSE VALUE DOES NOT EVAPORATE
 * ---------------------------------------------------------------------------
 * Host attention is **O(context) at 0.53 ms per position** — measured as a straight line
 * (14.80 ms/token at position ~25, 19.80 at ~35, 26.27 at ~46.5), 8 layers re-reading
 * 262,144 B of fp32 KV per position. Every card-work win this project has made is worth
 * ~1.04x at 8k context, because at 8k the host attention core alone is 4.4 s of a 4.8 s
 * token. This block is the only one whose gain GROWS with the context the product runs at.
 *
 * ---------------------------------------------------------------------------
 * THE FOUR THINGS ORNITH'S SHAPE NEEDS, AND WHERE EACH LANDED
 * ---------------------------------------------------------------------------
 *  1. **The joint Q+gate projection, de-interleaved.** `blk.N.attn_q.weight` emits
 *     `[h0 Q(256) | h0 gate(256) | h1 Q | ...]`, and NO op downstream can read that in
 *     place: the QK-norm reads head `h` as row block `h` at tiles `h*8` consecutive, and
 *     flash decode reads a KV group as one contiguous run of `q_rows*dim_tiles = 16` tiles.
 *     Over the joint strip a group's run is head `2g`'s Q followed by head `2g`'s GATE.
 *     `foundry/tt/tests/test_attention_ops.c` asserts exactly that.
 *     **Solved in the WEIGHT**, at load: two column-permuted copies via
 *     `pn_tt_device_stage_weight_columns()`, so the projection emits Q and the gate already
 *     apart. Zero per-token cost, no kernel change. `pn_tt_column_runs.h` prices the three
 *     candidates against each other.
 *  2. **Per-head QK-RMSNorm on both Q and K**, at `head_dim/32 = 8` tile columns. The
 *     existing per-head reduce residency is placed for the GDN's Wt=4 and REFUSES any other
 *     width, so this is a SECOND slot on its own core. It is **bf16**, deliberately — see
 *     `pn_tt_head_norm.h`: flash decode is bf16 end to end, so an fp32 norm would need a
 *     render either side of it that the card does not have and would be re-rounded by the
 *     very next op.
 *  3. **The SIGMOID output gate.** `sigmoid(z) * attn` is NOT `silu(z) * attn`, so the
 *     SwiGLU member cannot serve it; it is SIGMOID then MUL, two descriptors on the existing
 *     elementwise residency, no kernel source change and no blob regeneration.
 *  4. The all-Float32 program question is deliberately NOT reopened here. It is the same
 *     strip-bridge item the GDN norms are blocked on, and (2) is the argument for why
 *     attention is on the other side of it.
 *
 * **A FIFTH THING WAS SCOPED AND IS NOT REAL.** The spec's "GQA repeat — interleaved,
 * `kh = vh % 16`" describes the GDN block's head repeat, NOT the attention layers'.
 * Pantheon's own attention oracle is `kv_h = qh / n_kv_groups` (`qwen35moe_attn.c`), which is
 * CONTIGUOUS and is exactly what `pn_tt_attention()` issues. Checked head by head in the
 * foundry gate rather than read off the spec.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS OPT-IN, AND IT IS NOT A HEDGE
 * ---------------------------------------------------------------------------
 * `PN_TT_ONCARD=all` does not select attention and must not. This arm moves WHERE THE KV
 * CACHE LIVES (card DRAM, bf16) and changes the order the attention sum accumulates in
 * (blocked online softmax against a two-pass host softmax whose denominator is a double).
 * The host cache is fp32 and may be Q4. It is a DIFFERENT FORWARD, and on top of that the
 * QK-norm's RMS member measured 3.165e-03 at Wt=4 against a 1e-3 gate. So the `cmp` bar for
 * this arm is not the shipped path's: the discriminating instrument is the CROSSED CONTROL
 * below, which runs both arms over the same q/k/v and reports max|delta| PER HEAD.
 *
 * @status WRITTEN AND BUILDS; NOT RUN ON SILICON. Authored in a wave with exclusive card
 *         access held by another agent, so every figure here is DERIVED or inherited from a
 *         prior measurement and none of it is a measurement of this code.
 * @issues The card holds ONE K/V region set (`pn_tt_attention.c`), so two sequences
 *         alternating is detected and refused rather than served.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_column_runs.h"
#include "../accel/tt/pn_tt_head_norm.h"

/** Elements per tile row, and the tile count of a width. */
#define QWEN35_ATTN_TILE 32u

/** A crossed control against the host arm, per head. `PN_TT_ATTN_COMPARE=1`. It costs the
 *  readback it exists to remove, so it is a knob exactly like `PN_TT_TRACE`. */
#define QWEN35_ATTN_COMPARE_ENV "PN_TT_ATTN_COMPARE"

static uint32_t qwen35_attn_tiles(uint32_t width)
{
    return (width + QWEN35_ATTN_TILE - 1u) / QWEN35_ATTN_TILE;
}

/* ================================================================
 * Arming
 * ================================================================ */

/**
 * Stage the two de-interleaved halves of one layer's joint Q+gate projection.
 *
 * The run lists are built by `pn_tt_column_runs_field()` and checked injective by the
 * placement path, so a shape this function got wrong is a refusal at load rather than a
 * projection that is quietly missing one head.
 */
int pn_qwen35_tt_attn_split_q(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, const char *name,
                              bool extra, pn_qwen35_tt_layer *slot)
{
    pn_tt_column_run runs[PN_TT_COLUMN_RUNS_MAX];
    const uint32_t head_tiles = qwen35_attn_tiles((uint32_t)cfg->head_dim);
    const uint32_t joint_tiles = 2u * head_tiles;
    const uint32_t heads = (uint32_t)cfg->n_head;
    int (*stage)(pn_tt_device *, const char *, const pn_tt_column_run *, uint32_t,
                 pn_tt_weight *) =
        extra ? pn_tt_device_stage_extra_columns : pn_tt_device_stage_weight_columns;

    if (pn_tt_column_runs_field(heads, joint_tiles, 0u, head_tiles, runs,
                                PN_TT_COLUMN_RUNS_MAX) != heads ||
        stage(g->device, name, runs, heads, &slot->attn_q_only) != 0) {
        fprintf(stderr, "qwen35_tt: could not stage the Q half of %s\n", name);
        return -1;
    }
    if (pn_tt_column_runs_field(heads, joint_tiles, head_tiles, head_tiles, runs,
                                PN_TT_COLUMN_RUNS_MAX) != heads ||
        stage(g->device, name, runs, heads, &slot->attn_gate_only) != 0) {
        fprintf(stderr, "qwen35_tt: could not stage the gate half of %s\n", name);
        return -1;
    }
    /* The two halves must reconstruct the joint weight's width exactly. A `head_dim` that
     * does not divide the projection is the one shape mistake the run list cannot see.
     * (An extra-artifact layer has no joint weight placed: its width is the two halves'.) */
    if (extra)
        slot->attn_q.tile_cols = slot->attn_q_only.tile_cols + slot->attn_gate_only.tile_cols;
    if (slot->attn_q_only.tile_cols + slot->attn_gate_only.tile_cols !=
        slot->attn_q.tile_cols) {
        fprintf(stderr, "qwen35_tt: %s is %u tile columns but 2 x %u heads x %u is %u — the "
                        "Q+gate interleave is not what this layer's weight holds\n", name,
                slot->attn_q.tile_cols, heads, head_tiles,
                slot->attn_q_only.tile_cols + slot->attn_gate_only.tile_cols);
        return -1;
    }
    return 0;
}

/** The extra strips the card arm needs beside the ones the resident forward already holds. */
static int qwen35_attn_alloc_strips(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    const uint32_t q_tiles = qwen35_attn_tiles((uint32_t)(cfg->n_head * cfg->head_dim));
    const uint32_t kv_tiles = qwen35_attn_tiles((uint32_t)(cfg->n_head_kv * cfg->head_dim));

    /* SEPARATE ROTATED STRIPS rather than RoPE in place: `pn_tt_rope` is a stream op whose
     * reader runs ahead of its writer, so an in-place rotation would be safe today and an
     * unstated dependency on that lead. The same argument the residual pair is written on. */
    return (pn_tt_activation_alloc(g->device, q_tiles, &a->q) == 0 &&
            pn_tt_activation_alloc(g->device, q_tiles, &a->q_rot) == 0 &&
            pn_tt_activation_alloc(g->device, q_tiles, &a->z) == 0 &&
            pn_tt_activation_alloc(g->device, q_tiles, &a->gate) == 0 &&
            pn_tt_activation_alloc(g->device, q_tiles, &a->gated) == 0 &&
            pn_tt_activation_alloc(g->device, kv_tiles, &a->k_rot) == 0)
               ? 0
               : -1;
}

/**
 * Is this a shape the on-card RoPE kernel will execute rather than approximate?
 *
 * `ft_tt_op_rope.h` refuses `half % 32 == 16` (the partner lands in a different FACE COLUMN)
 * and a `rot` that is not a multiple of 32 at all. Ornith's own RoPE is
 * `qwen35moe_rope_partial()`, which is the kernel's split-half NeoX form expression for
 * expression — the two agree on the pairing, so the only question left is the width, and it
 * is asked HERE because the alternative is finding out on the card.
 */
static bool qwen35_attn_rope_is_expressible(const pn_qwen35moe_config *cfg)
{
    const int rot = (cfg->rope_dim > 0 && cfg->rope_dim < cfg->head_dim) ? cfg->rope_dim
                                                                        : cfg->head_dim;

    if ((rot % 64) != 0) {
        fprintf(stderr, "qwen35_tt: REFUSED — on-card attention needs a rotary width that is "
                        "a multiple of 64 so `half` is a whole number of tiles; this model "
                        "rotates %d of %d\n", rot, cfg->head_dim);
        return false;
    }
    return true;
}

int pn_qwen35_tt_attn_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                          const pn_qwen35moe_config *cfg)
{
    const uint32_t head_width = (uint32_t)cfg->head_dim;
    int layer;

    if (!g || !m || !cfg)
        return -1;
    g->attn_on_card = false;
    if (!pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_ATTENTION))
        return 0;
    if (!qwen35_attn_rope_is_expressible(cfg))
        return -1;

    /* THE QK-NORM RESIDENCY FIRST, because it is the one that can decline for a reason the
     * caller can act on — a second head width when the GDN already took the last slot. */
    if (pn_tt_device_arm_head_norm_at(g->device, head_width, FT_TT_OP_TILE_BYTES_BF16) != 0)
        return -1;
    if (!pn_tt_head_norm_is_armed(g->device, head_width)) {
        fprintf(stderr, "qwen35_tt: REFUSED — on-card attention needs a per-head reduce "
                        "residency at Wt=%u for the QK-RMSNorms and none armed\n",
                head_width / QWEN35_ATTN_TILE);
        return -1;
    }

    if (pn_tt_device_arm_attention(g->device, (uint32_t)cfg->n_head,
                                   (uint32_t)cfg->n_head_kv, head_width,
                                   (uint32_t)cfg->rope_dim, (double)cfg->rope_theta,
                                   (uint32_t)pn_qwen35moe_max_seq(cfg),
                                   (uint32_t)pn_qwen35moe_model_n_attn(m) +
                                       (g->mtp.path[0] ? 1u : 0u)) !=
        0)
        return -1;
    /* STEP 4 S2b: the MTP block's KV region is the one after the trunk's. */
    g->mtp.attn_index = pn_qwen35moe_model_n_attn(m);
    if (!pn_tt_device_oncard(g->device, PN_TT_ONCARD_ATTENTION))
        return -1;
    if (qwen35_attn_alloc_strips(g, cfg) != 0) {
        fprintf(stderr, "qwen35_tt: REFUSED — the attention strips did not allocate\n");
        return -1;
    }

    for (layer = 0; layer < cfg->n_layer; layer++) {
        pn_qwen35_tt_layer *slot = &g->layers[layer];
        int index;
        const pn_qwen35moe_attn_layer *l;

        if (pn_qwen35moe_model_layer_type(m, layer) != PN_QWEN35MOE_LAYER_ATTN)
            continue;
        index = pn_qwen35moe_model_layer_index(m, layer);
        l = pn_qwen35moe_model_attn_layer(m, index);
        {
            char name[128];

            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
            if (!l || pn_qwen35_tt_attn_split_q(g, cfg, name, false, slot) != 0)
                return -1;
        }
        /* GAMMA IS SHARED ACROSS HEADS, which is what the row-block layout gives for free
         * and what `qwen35moe_qk_norm_heads()` does: one `attn_q_norm` applied identically
         * to all 32 Q heads, one `attn_k_norm` to all 16 KV heads. */
        if (pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM,
                                             l->attn_q_norm, head_width, cfg->rms_norm_eps,
                                             &slot->attn_q_norm) != 0 ||
            pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM,
                                             l->attn_k_norm, head_width, cfg->rms_norm_eps,
                                             &slot->attn_k_norm) != 0) {
            fprintf(stderr, "qwen35_tt: REFUSED — layer %d's QK-norm constants did not "
                            "upload\n", layer);
            return -1;
        }
    }

    g->attn_on_card = true;
    g->attn_compare = getenv(QWEN35_ATTN_COMPARE_ENV) != NULL;
    fprintf(stderr, "qwen35_tt: attention on the card — %d Q heads / %d KV groups, head_dim "
                    "%d, QK-norm on the Wt=%u per-head residency, sigmoid gate on the "
                    "elementwise one\n", cfg->n_head, cfg->n_head_kv, cfg->head_dim,
            head_width / QWEN35_ATTN_TILE);
    return 0;
}

bool pn_qwen35_tt_attn_is_on_card(const pn_qwen35_tt *g)
{
    return g && g->attn_on_card;
}

/* ================================================================
 * The block
 * ================================================================ */

int pn_qwen35_tt_attn_project_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d)
{
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;

    if (!g || !d)
        return -1;
    /* FOUR PROJECTIONS, NOT THREE, and the fourth is free. The joint weight is staged as two
     * halves, so Q and the gate come out of two matmuls over the same input instead of one
     * matmul and a per-token de-interleave. Same FLOPs, same weight bytes, one more
     * descriptor: 8 a token against 249. They are independent, so they chain. */
    if (pn_qwen35_tt_project(g, &d->attn_q_only, &s->normed, &a->q, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_gate_only, &s->normed, &a->z, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_k, &s->normed, &s->attn_k, false) != 0 ||
        pn_qwen35_tt_project(g, &d->attn_v, &s->normed, &s->attn_v, false) != 0)
        return -1;
    return 0;
}

int pn_qwen35_tt_attn_core_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                const pn_qwen35moe_config *cfg, int attn_index, int pos,
                                int seq_len)
{
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;

    if (!g || !d || !cfg || seq_len <= 0)
        return -1;

    /* IN PLACE IS CORRECT HERE and not elsewhere: the reduce reader has read a whole row
     * block before the writer can emit it, and the row blocks do not alias
     * (`pn_tt_head_norm.h`). V is NOT normed — that is the model, not an omission. */
    if (pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_q_norm,
                             (uint32_t)cfg->n_head, &a->q, &a->q) != 0 ||
        pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_k_norm,
                             (uint32_t)cfg->n_head_kv, &s->attn_k, &s->attn_k) != 0)
        return -1;

    if (pn_tt_rope(g->device, &a->q, &a->q_rot, (uint32_t)cfg->n_head, (uint32_t)pos) != 0 ||
        pn_tt_rope(g->device, &s->attn_k, &a->k_rot, (uint32_t)cfg->n_head_kv,
                   (uint32_t)pos) != 0)
        return -1;
    if (pn_tt_kv_append(g->device, (uint32_t)attn_index, false, &a->k_rot,
                        (uint32_t)pos) != 0 ||
        pn_tt_kv_append(g->device, (uint32_t)attn_index, true, &s->attn_v,
                        (uint32_t)pos) != 0)
        return -1;
    if (pn_tt_attention(g->device, (uint32_t)attn_index, &a->q_rot, &s->attn_out,
                        (uint32_t)seq_len) != 0)
        return -1;

    /* SIGMOID, NOT SILU. `silu(x) = x*sigmoid(x)`, so reusing the SwiGLU member here is off
     * by a factor of the gate itself — and both operands are `n_head*head_dim`, so no shape
     * check anywhere would see it. `PN_TT_RED=attn_gate_silu` is that mutation. */
    return pn_tt_sigmoid_mul(g->device, &a->z, &s->attn_out, &a->gate, &a->gated);
}

int pn_qwen35_tt_attn_core_chunk(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                 const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                                 int rows)
{
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;

    if (!g || !d || !cfg || rows <= 0 || rows > 32 || base_pos < 0 || (base_pos % 32) != 0)
        return -1;

    /* UNCHANGED, AND THAT IS THE FINDING. A per-head norm is `row_blocks = heads` over a
     * `[1, heads*S]` strip; a row reduce reduces ALONG the row, so row t of row block h is
     * position t of head h and 32 positions cost one descriptor exactly as one position
     * does. V is still not normed. */
    if (pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_q_norm,
                             (uint32_t)cfg->n_head, &a->q, &a->q) != 0 ||
        pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_k_norm,
                             (uint32_t)cfg->n_head_kv, &s->attn_k, &s->attn_k) != 0)
        return -1;

    if (pn_tt_rope_chunk(g->device, &a->q, &a->q_rot, (uint32_t)cfg->n_head,
                         (uint32_t)base_pos, (uint32_t)rows) != 0 ||
        pn_tt_rope_chunk(g->device, &s->attn_k, &a->k_rot, (uint32_t)cfg->n_head_kv,
                         (uint32_t)base_pos, (uint32_t)rows) != 0)
        return -1;
    if (pn_tt_kv_append_chunk(g->device, (uint32_t)attn_index, false, &a->k_rot,
                              (uint32_t)base_pos, (uint32_t)rows) != 0 ||
        pn_tt_kv_append_chunk(g->device, (uint32_t)attn_index, true, &s->attn_v,
                              (uint32_t)base_pos, (uint32_t)rows) != 0)
        return -1;
    if (pn_tt_attention_chunk(g->device, (uint32_t)attn_index, &a->q_rot, &s->attn_out,
                              (uint32_t)base_pos, (uint32_t)rows) != 0)
        return -1;

    /* SIGMOID, NOT SILU — the same note as the per-position arm, and per element, so the
     * chunk costs one descriptor. */
    return pn_tt_sigmoid_mul(g->device, &a->z, &s->attn_out, &a->gate, &a->gated);
}

int pn_qwen35_tt_attn_block_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                 const pn_qwen35moe_config *cfg, int attn_index, int pos,
                                 int seq_len)
{
    if (pn_qwen35_tt_attn_project_card(g, d) != 0 ||
        pn_qwen35_tt_attn_core_card(g, d, cfg, attn_index, pos, seq_len) != 0)
        return -1;
    return pn_qwen35_tt_project(g, &d->attn_output, &g->attn_card.gated, &g->card.project,
                                true);
}

/* ================================================================
 * The crossed control
 * ================================================================ */

/**
 * Run BOTH arms over the same projections and diff the attention outputs PER HEAD.
 *
 * NOT a PCC check, and deliberately not one. The failures this path can have are a wrong KV
 * position, a group's query reading another group's head, a mask tail off by one, a
 * de-interleave that swapped Q and the gate, and a QK-norm at the wrong width — every one of
 * which is a PERMUTATION or a localised wrong element. PCC is blind to the first and
 * saturates on the second. So this reports max|delta| and WHICH HEAD it was on, because
 * "one head of thirty-two is wrong" is a per-tensor figure of 1.0e+00 and a per-head figure
 * that names the head. A per-tensor max-norm hid a per-head fault in the GDN family for
 * three rounds.
 *
 * It costs a full download of both answers AND a second whole attention block, so it is off
 * unless `PN_TT_ATTN_COMPARE` is set. When it IS set, `host_attention_seconds` counts the
 * shadow arm and no timing from that run is comparable with one from a run without it.
 *
 * The caller runs the HOST block first and the CARD block second, so `s->project` — the one
 * the forward carries — is the CARD's. Reversed, this would be a host forward with a card
 * shadow, which is a measurement of the thing not under test.
 */
void pn_qwen35_tt_attn_compare(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int layer,
                               const float *host_attn)
{
    const int head_dim = cfg->head_dim;
    const uint32_t width = (uint32_t)(cfg->n_head * head_dim);
    double worst = 0.0;
    double worst_relative = 0.0;
    int worst_relative_head = -1;
    int worst_head = -1;
    int worst_lane = -1;
    int heads_out = 0;
    int head;

    if (!g->attn_compare || !host_attn || !g->h_attn_card)
        return;
    if (pn_tt_activation_download(g->device, &g->attn_card.gated, g->h_attn_card, width) != 0) {
        fprintf(stderr, "qwen35_tt: [attn compare] layer %d — the card answer did not come "
                        "back; NOTHING is scored for this layer\n", layer);
        return;
    }
    for (head = 0; head < cfg->n_head; head++) {
        double head_worst = 0.0;
        double scale = 0.0;
        int lane;

        for (lane = 0; lane < head_dim; lane++) {
            const size_t index = (size_t)head * (size_t)head_dim + (size_t)lane;
            const double delta = fabs((double)g->h_attn_card[index] -
                                      (double)host_attn[index]);

            if (fabs((double)host_attn[index]) > scale)
                scale = fabs((double)host_attn[index]);
            if (delta > head_worst) {
                head_worst = delta;
                if (delta > worst) {
                    worst = delta;
                    worst_head = head;
                    worst_lane = lane;
                }
            }
        }
        /* MAX-NORM RELATIVE, per head — the same figure the GDN parity gate reports, so the
         * two are readable against each other and against the 1e-3 bar.
         *
         * **THE COUNT ALONE DOES NOT DECIDE ANYTHING.** "16 of 16 heads past 1e-3" is the
         * same sentence for an arm 1.1x over the bar and for one 800x over it, and those are
         * a bf16 cache and a permutation respectively. The WORST relative figure is what
         * separates them, so it is carried out beside the count rather than thrown away
         * inside the comparison. */
        if (scale > 0.0) {
            const double relative = head_worst / scale;

            if (relative > 1.0e-3)
                heads_out++;
            if (relative > worst_relative) {
                worst_relative = relative;
                worst_relative_head = head;
            }
        }
    }
    fprintf(stderr, "qwen35_tt: [attn compare] layer %d — max|delta| %.3e at head %d lane %d, "
                    "worst RELATIVE %.3e at head %d, %d of %d heads past 1e-3 relative\n",
            layer, worst, worst_head, worst_lane, worst_relative, worst_relative_head,
            heads_out, cfg->n_head);
}

/* ================================================================
 * Release
 * ================================================================ */

void pn_qwen35_tt_attn_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    free(g->h_attn_card);
    g->h_attn_card = NULL;
    g->attn_on_card = false;
}

#endif /* USE_TT */
