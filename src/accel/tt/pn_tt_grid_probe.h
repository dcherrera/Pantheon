/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_probe.h
 * @brief PREFILL GRID PROBE (2026-10-26, measurement only): each prefill-grid projection
 *        timed alone — the chain flushed before it, the projection issued, then flushed — and
 *        booked by shape (Mt x Kt x Nt tiles), so the report gives achieved TFLOP/s per
 *        projection kind on the grid.
 *
 * Serialises the grid (a projection never overlaps another or the chain), so its wall is
 * not today's prefill: it is the op's own time. Meaningful only with the stream off
 * (`PN_TT_PREFILL_STREAM=0`): a streamed flush is answered at once.
 *
 * @status Opt-in: `PN_TT_GRID_PROBE=1`. Bit-identical by construction (same descriptors).
 * @issues Process-global; one prefill at a time.
 * @todo None
 */
#ifndef PN_TT_GRID_PROBE_H
#define PN_TT_GRID_PROBE_H

#include <stdbool.h>
#include <stdint.h>

/** `PN_TT_GRID_PROBE=1`. */
bool pn_tt_grid_probe_enabled(void);

/** Book one projection: its tile shape, the members it ran on, issue and await seconds. */
void pn_tt_grid_probe_book(uint32_t mt, uint32_t kt, uint32_t nt, uint32_t members,
                           double issue_seconds, double await_seconds);

/** Print `GRID PROBE —` rows (one per shape) and reset. */
void pn_tt_grid_probe_report(void);

#endif /* PN_TT_GRID_PROBE_H */
