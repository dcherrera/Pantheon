/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_idle.h
 * @brief GRID IDLE GUARD: the prefill grid gets the idle rule every other family has.
 *
 * WHY. A grid core's residency exits on its own after its idle spin limit (the fastest
 * measured exit is ~8 s). Residencies are retired and re-armed past 5000 ms idle, and the
 * decode group gets keepalive NOPs; the grid got neither. A host gap of ~8 s between two grid
 * descriptors (a pass boundary: the MTP layer, the keepalive, a 2048-row embedding gather
 * that faults from disk) let every grid core exit, and the next pass's first projection went
 * to dead cores: the puller AWAITed them 20 s, the stream failed, GRID FAIL "the host stopped
 * feeding the queue", BATCHED PREFILL ABORTED (q38-tt-grid-abort, 2026-11-20).
 *
 * THE RULE. `idle` is host seconds since the last grid descriptor went out. A core finishes
 * its descriptor after the host issued it, so the card's idle stretch is never longer than
 * that (the one exception: a descriptor captured in a stream chunk not yet launched).
 *   - keepalive point (`nop_ok`): idle >= min_nop and below limit - margin: NOP every core;
 *   - idle >= limit - margin at a keepalive point, or >= limit at an issue: RE-ARM (stop by
 *     asking, start fresh); a re-armed core runs the same descriptors, so bit-identical;
 *   - otherwise nothing. `limit` 0 turns the guard off.
 *
 * Pure: no device. Build the test with -DPN_TT_GRID_IDLE_RED (the guard never re-arms).
 *
 * @status q38-tt-grid-abort fix.
 * @issues None
 * @todo   None
 */
#ifndef PN_TT_GRID_IDLE_H
#define PN_TT_GRID_IDLE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PN_TT_GRID_IDLE_NONE = 0,
    PN_TT_GRID_IDLE_NOP,
    PN_TT_GRID_IDLE_REARM
} pn_tt_grid_idle_action;

/** No NOP this close to the limit: a NOP could reach a half-exited triad (the decode
 *  group's keepalive keeps the same margin). */
#define PN_TT_GRID_IDLE_MARGIN_S 0.5

/**
 * @param idle_s     host seconds since the last grid descriptor (< 0: none yet)
 * @param min_nop_s  keepalive: idle below this needs nothing
 * @param limit_s    the retire limit (0 = guard off)
 * @param nop_ok     a keepalive point (quiet, nothing pending), not an issue
 */
pn_tt_grid_idle_action pn_tt_grid_idle_decide(double idle_s, double min_nop_s, double limit_s,
                                              bool nop_ok);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_GRID_IDLE_H */
