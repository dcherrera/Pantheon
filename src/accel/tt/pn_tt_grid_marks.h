/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_marks.h
 * @brief GRID STEP MARKS (q38-gdn-chain G.2): wait for ONE detached grid projection, not all.
 *
 * WHY. The prefill overlap issues grid projections detached and later waits for them. A wait
 * for "every detached projection" drains the grid: it idles through the synchronous steps
 * after the wait (residual add, norm, SwiGLU) until the next projection is issued. With a
 * mark per projection — every grid core's issued count just after it — a wait names only the
 * projection it needs, and anything issued after it keeps the grid busy. Each grid core runs
 * its ring in order, so a core that reached the mark's count has finished that projection
 * and everything before it.
 *
 * Also the RING ROOM: the grid ring holds `limit` descriptors a core in flight. When an issue
 * would overflow it, wait for the OLDEST marks that make room instead of draining everything.
 *
 * Pure: no device. The caller records each issue, pushes a mark after it, and turns `take`'s
 * per-core counts into waits (streamed: puller AWAITs; otherwise a real full wait).
 *
 * @status G.2 of the prefill chain work. Build the test with
 *         -DPN_TT_GRID_MARKS_RED (take hands out the previous mark's counts) to see it fail.
 * @issues None
 * @todo   None
 */
#ifndef PN_TT_GRID_MARKS_H
#define PN_TT_GRID_MARKS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_TT_GRID_MARK_CORES 64u
#define PN_TT_GRID_MARKS 16u
#define PN_TT_GRID_MARK_NONE 0xFFFFFFFFu

typedef struct {
    uint32_t ring_total;     /**< ring depth issued through this mark (cumulative) */
    uint32_t detached_total; /**< detached projections issued through this mark */
    uint32_t count[PN_TT_GRID_MARK_CORES]; /**< each grid core's issued count */
} pn_tt_grid_mark;

typedef struct {
    pn_tt_grid_mark mark[PN_TT_GRID_MARKS]; /**< mark id `i` lives at `i % PN_TT_GRID_MARKS` */
    uint32_t cores;
    uint32_t next;   /**< the next mark id */
    uint32_t oldest; /**< the oldest mark still held (older ones were taken or evicted) */
    uint32_t done_through; /**< marks below this id are done (taken, or covered by one) */
    uint32_t ring_total, ring_done;
    uint32_t detached_total, detached_done;
    uint32_t awaited[PN_TT_GRID_MARK_CORES]; /**< the highest count waited for, a core */
} pn_tt_grid_marks;

/** Forget everything (a grid start). */
void pn_tt_grid_marks_reset(pn_tt_grid_marks *m, uint32_t cores);

/** A grid projection was issued: `depth` ring descriptors a core, `detached` or chained. */
void pn_tt_grid_marks_issued(pn_tt_grid_marks *m, uint32_t depth, bool detached);

/** Mark everything issued so far (`counts`: each core's issued count). @return the mark id.
 *  The 17th live mark evicts the oldest (a later wait for it falls back to a full wait). */
uint32_t pn_tt_grid_marks_push(pn_tt_grid_marks *m, const uint32_t *counts);

/**
 * Wait for mark `id`: fills `need[core]` with the count to wait for, or 0 where the core was
 * already waited that far, and books the ring and detached projections through it as done.
 *
 * @return how many cores need a wait, or -1 when the mark is gone (evicted, never pushed):
 *         the caller waits for everything instead (pn_tt_grid_marks_all)
 */
int pn_tt_grid_marks_take(pn_tt_grid_marks *m, uint32_t id, uint32_t *need);

/** Everything issued is done (a full wait): `counts` each core's issued count. */
void pn_tt_grid_marks_all(pn_tt_grid_marks *m, const uint32_t *counts);

/** Ring descriptors / detached projections still in flight. */
uint32_t pn_tt_grid_marks_ring(const pn_tt_grid_marks *m);
uint32_t pn_tt_grid_marks_detached(const pn_tt_grid_marks *m);

/**
 * Room for an issue of `depth` descriptors a core, with `chained` projections on the matvec
 * chain, under `limit` (ring descriptors a core; projections in flight below it).
 *
 * @return PN_TT_GRID_MARK_NONE when it already fits; else the OLDEST mark whose take makes it
 *         fit; or `(uint32_t)-2` when no live mark does (the caller waits for everything)
 */
uint32_t pn_tt_grid_marks_room(const pn_tt_grid_marks *m, uint32_t depth, uint32_t chained,
                               uint32_t limit);

#define PN_TT_GRID_MARK_FULL 0xFFFFFFFEu

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_GRID_MARKS_H */
