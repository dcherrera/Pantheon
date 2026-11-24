/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_schedule.h
 * @brief P3: HOW a norm's reduce descriptors are scheduled — block depth, streamed and half
 *        reads, the final-chain run length — and the residency extents that schedule needs.
 *
 * @status Off-hardware only: tests/test_tt_norm_schedule.c. Card arms: pn_tt_oncard.c
 *         (model norm), pn_tt_head_norm.c.
 * @issues None
 * @todo None
 *
 * A schedule moves WHEN tiles are read and computed, never what is computed: every tile
 * sees the same ops in the same order at the same rounding points, so any schedule is
 * bit-exact against the pre-P3 one (docs/q38dec_p3_norm.md). The wire half is foundry
 * `ft_tt_op_vector.h` (FT_TT_OP_REDUCE_SCHED_*) and `kernels/resident_reduce_protocol.h`.
 *
 * Pure integer planning, no card and no `USE_TT`, so `tests/test_tt_norm_schedule.c` proves
 * it. Two shapes:
 *
 *   model norm   one row block a decode token, Wt 160 on Qwen3.8, column-split two ways:
 *                streamed at depth 16, half reads, final chain in runs of 16.
 *   head norm    one row block a head, Wt 4 or 8: one block a row, CB_IN four rows deep so
 *                the reader runs four heads ahead, the chain in runs of a whole row.
 *
 * `fast` false is the pre-P3 schedule exactly: depth = Wt, everything else zero.
 */

#ifndef PN_TT_NORM_SCHEDULE_H
#define PN_TT_NORM_SCHEDULE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** `PN_TT_NORM_FAST`: 1 (default) the P3 schedule, 0 the pre-P3 one. */
#define PN_TT_NORM_FAST_ENV       "PN_TT_NORM_FAST"
/** `PN_TT_NORM_DEPTH`: the model norm's streamed block depth (default 16). */
#define PN_TT_NORM_DEPTH_ENV      "PN_TT_NORM_DEPTH"
#define PN_TT_NORM_DEPTH_DEFAULT  16u
/** `PN_TT_NORM_CHUNK`: the model norm's final-chain run length K (default 16). */
#define PN_TT_NORM_CHUNK_ENV      "PN_TT_NORM_CHUNK"
#define PN_TT_NORM_CHUNK_DEFAULT  16u
/** The kernel carries K in one byte. */
#define PN_TT_NORM_CHUNK_MAX      255u
/** How many heads the head-norm reader may run ahead: CB_IN rows. */
#define PN_TT_HEAD_NORM_INPUT_ROWS 4u

/** One norm residency's schedule. All-zero extents and flags are the pre-P3 schedule. */
typedef struct pn_tt_norm_schedule {
    uint32_t prefetch_tiles; /**< reader block depth asked for; the encoder fits it to Wt */
    uint32_t chain_chunk;    /**< final-chain run K; 0 = per tile */
    uint32_t stage_pages;    /**< residency: scratch and cb_out pages; 0 = pre-P3 */
    uint32_t input_rows;     /**< residency: CB_IN rows; 0 = pre-P3 (one) */
    bool stream_input;       /**< pipelined input, gamma after row 0 */
    bool half_read;          /**< faces 0-1 of input and gamma, where the kernel allows it */
} pn_tt_norm_schedule;

/**
 * The model norm's schedule at `tile_cols`.
 *
 * @param fast       false: the pre-P3 schedule
 * @param tile_cols  Wt (must be non-zero)
 * @param depth      streamed block depth asked for (0 = the default)
 * @param chunk      final-chain run asked for (0 = the default; clamped to 1..255 and Wt)
 * @param out        receives the schedule
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_norm_schedule_model(bool fast, uint32_t tile_cols, uint32_t depth, uint32_t chunk,
                               pn_tt_norm_schedule *out);

/**
 * The per-head norms' schedule at head width `tile_cols`.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_norm_schedule_head(bool fast, uint32_t tile_cols, pn_tt_norm_schedule *out);

/**
 * The final-chain run a descriptor finishing `slice_tiles` per row asks for: K, or the slice
 * when that is shorter. Never more than `stage_pages`, so the reader never refuses it.
 *
 * @return 0 for the pre-P3 per-tile chain
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_norm_schedule_chunk_for(const pn_tt_norm_schedule *schedule,
                                       uint32_t slice_tiles);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_NORM_SCHEDULE_H */
