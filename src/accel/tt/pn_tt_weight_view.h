/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_view.h
 * @brief A COLUMN VIEW of a resident weight: some of its output columns read IN PLACE, as a
 *        weight of their own, with no second copy in card DRAM.
 *
 * WHY. pn_tt_device_stage_weight_columns() uploads a column subset as a SECOND copy, because
 * the bump allocator cannot free a tensor out of the middle. The MTP draft head's LM-head
 * subset (2,057 of Qwen3.8's 7,760 column tiles, Bfp8) is 0.333 GiB of such a copy, ~42 MiB a
 * DRAM bank, and the bank rule needs it back at long context (lever V, qwen35moe_tt_fit_plan.h).
 *
 * HOW. A sharded weight's member m holds columns [first_m, first_m + cols_m) as an N-major
 * panel: column c of it is a contiguous Kt x tile run at `panel + (c - first_m) * Kt * tile`
 * (pn_tt_matmul_group_place_weight_columns). A view piece is a run of the view's columns that
 * lies inside ONE source panel: its B base is that panel plus the offset of its first column,
 * its door and bank the panel's, and its output offset its first VIEW column — exactly the
 * words a placed shard carries, so every reader and the prefill grid read it unchanged. The
 * pieces are split to spread the view over up to `members` members (widths within one of
 * each other inside a segment; a segment never crosses a source panel).
 *
 * The answer cannot move: each output column is the same dot product over the same bytes as
 * the staged copy's. The speed can: the subset's columns sit in the panels of the source
 * members that own them (Qwen3.8: 5 of 16, in 5 banks, plus the tail's), not spread over all.
 * A RAGGED split: SPLIT dispatch only — the caller refuses it when the group cooperates.
 *
 * No device: arithmetic over pn_tt_weight, tested on any host (`tests/test_tt_weight_view.c`).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_WEIGHT_VIEW_H
#define PN_TT_WEIGHT_VIEW_H

#ifdef USE_TT

#include <stdint.h>

#include "pn_tt_column_runs.h"
#include "pn_tt_device.h"

/**
 * Build `out`, the view of `source`'s columns `runs` (in view order), over at most `members`
 * pieces (<= PN_TT_MATMUL_MAX_CORES).
 *
 * @return 0, or -1: a bad argument, runs that leave the source or repeat a column, or more
 *         source segments than `members`
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_weight_column_view(const pn_tt_weight *source, const pn_tt_column_run *runs,
                             uint32_t run_count, uint32_t members, pn_tt_weight *out);

#endif /* USE_TT */

#endif /* PN_TT_WEIGHT_VIEW_H */
