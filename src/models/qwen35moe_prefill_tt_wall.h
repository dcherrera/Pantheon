/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_tt_wall.h
 * @brief The batched prefill's WALL, by region, against the batched/sequential/transfer lines.
 *
 * The `batched prefill —` line sums three brackets that live INSIDE the layer loop. A serve
 * request of 1,630 tokens took 7.07 s where that line said 4.02 s, so ~3 s lived outside it.
 * This file clocks every region of `pn_qwen35moe_prefill_tt()` and reports, per region, its
 * wall and how much of that wall the three lines already contain — the remainder is what
 * no line names. Printed under `PN_TT_PREFILL_REPORT`, once per call (not cumulative).
 *
 * status: measurement only; changes no arithmetic and no ordering.
 * issues: regions nest (residual and mtp inside a pass, state-reset and grid-begin inside setup),
 *         so the rows do not sum to total.
 * todo:   none.
 */
#ifndef PN_QWEN35MOE_PREFILL_TT_WALL_H
#define PN_QWEN35MOE_PREFILL_TT_WALL_H

#include <stdbool.h>

typedef enum {
    PN_PREFILL_WALL_SETUP,        /**< tall enter, row scratch, state zero, grid begin */
    PN_PREFILL_WALL_STATE_RESET,  /**< the carried GDN state + conv window zeroed (in SETUP) */
    PN_PREFILL_WALL_GRID_BEGIN,   /**< the prefill grid booted (in SETUP) */
    PN_PREFILL_WALL_EMBED,        /**< host embedding rows */
    PN_PREFILL_WALL_STAGE,        /**< strip zero + band uploads of the embeddings */
    PN_PREFILL_WALL_LAYERS,       /**< the whole layer loop */
    PN_PREFILL_WALL_RESIDUAL,     /**< residual adds (inside LAYERS) */
    PN_PREFILL_WALL_MTP,          /**< the MTP block over the prompt */
    PN_PREFILL_WALL_FLUSH,        /**< the final matvec flush */
    PN_PREFILL_WALL_GRID_END,
    PN_PREFILL_WALL_RESTAGE,      /**< re-zero every strip the prompt wrote */
    PN_PREFILL_WALL_TALL_EXIT,    /**< republish the decode mirror map */
    PN_PREFILL_WALL_PARAM_RESET,
    PN_PREFILL_WALL_TAIL,         /**< the last token's ordinary forward (the logits) */
    PN_PREFILL_WALL_COUNT
} pn_prefill_wall_region;

/** Start a call: zero every region, mark the total. */
void pn_prefill_wall_begin(double accounted_seconds);

/** Open / close a region. `accounted_seconds` = batched + sequential + transfer right now. */
void pn_prefill_wall_open(pn_prefill_wall_region region, double accounted_seconds);
void pn_prefill_wall_close(pn_prefill_wall_region region, double accounted_seconds);

/** The re-arm clock (seconds spent booting / starting / stopping residencies so far): each
 *  region then also reports how much of its wall was idle-retire re-arms. NULL = none. */
void pn_prefill_wall_set_rearm_clock(double (*clock)(void));

/** PROFILE (2026-10-26): one layer of a batched pass took `seconds` of wall (its norm, mixer,
 *  projections, FFN and residuals; a W2 layer includes the next layer's prep it ran). */
void pn_prefill_wall_layer(bool attention, double seconds);

/** PROFILE: one attention layer's streamed band loop (QK norms, RoPE, K/V appends, the chunk
 *  mixer, the gate) took `seconds`, open to finish. */
void pn_prefill_wall_attn_mixer(double seconds);

/** Print `prefill WALL —` and one row per region that moved. */
void pn_prefill_wall_report(int n_tokens, double accounted_seconds);

#endif /* PN_QWEN35MOE_PREFILL_TT_WALL_H */
