/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_deep.h
 * @brief DEEP-QUEUED PREFILL ATTENTION's schedule: which
 *        attention core runs which (band, Q head) CHUNK descriptor of one attention layer-pass.
 *        Pure arithmetic, no device: the card path is pn_tt_attention.c
 *        pn_tt_attention_chunk_queue().
 *
 * The shipped band loop runs one band at a time: its 24 head descriptors in rounds of one a
 * member, awaited per round, then the next band. The deep path appends every band's K/V first,
 * then queues every (band, head) unit of the layer-pass over the members without a barrier,
 * and settles once. Each unit's descriptor is byte for byte the shipped one; only the core and
 * the time it runs change, so the output is the same bytes (spec §7: later bands' K/V rows
 * sit past each band's live length and take the dead mask, as the stale rows they replace did).
 *
 * THE ORDER: band-major, head-minor, unit u on member `u % members` — every member gets
 * `units / members` units or one more, and no member count has to divide the heads (the grid
 * cores of plan 3.2 make it 60).
 *
 * @status Off by default (`PN_TT_ATTN_DEEP=1` on) until its card gate (plan 1.2).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ATTN_DEEP_H
#define PN_TT_ATTN_DEEP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The most members a schedule or a flow counts (12 attention + 48 grid, with room). */
#define PN_TT_ATTN_DEEP_MAX_MEMBERS 128u

/** One CHUNK descriptor of a layer-pass: band (32 rows), Q head, and the member that runs it. */
typedef struct pn_tt_attn_deep_unit {
    uint32_t band;
    uint32_t head;
    uint32_t member;
} pn_tt_attn_deep_unit;

/** The member unit `index` (issue order) runs on. `members` >= 1. */
uint32_t pn_tt_attn_deep_member(uint32_t index, uint32_t members);

/**
 * Fill `out` with the layer-pass's units in issue order.
 *
 * @return units written (`bands * heads`), or 0 on a zero count or when `max` is short
 */
uint32_t pn_tt_attn_deep_plan(uint32_t bands, uint32_t heads, uint32_t members,
                              pn_tt_attn_deep_unit *out, uint32_t max);

/**
 * Is `units` a legal schedule: every (band, head) exactly once, bands in non-decreasing order,
 * every member below `members`, and member loads within one of each other?
 *
 * @return NULL when legal, else the first defect by name
 */
const char *pn_tt_attn_deep_plan_check(const pn_tt_attn_deep_unit *units, uint32_t count,
                                       uint32_t bands, uint32_t heads, uint32_t members);

/**
 * THE FLOW CONTROL. A member's ring has none of its own (ft_tt_queue_stage() does not check
 * occupancy), so the issuer counts what each member has outstanding and settles a member before
 * an enqueue that would put more than `depth` on it. MEASURED, the hard way: a queue that left
 * this to a deferred residency's bookkeeping overran the ring by ~1,260 descriptors and wedged
 * 12 cores. `settle` must return only when the
 * member has nothing outstanding; the card path's is pn_tt_residency_await_now().
 */
typedef struct pn_tt_attn_deep_io {
    void *context;
    /** Enqueue unit (band, head) on `member`. 0, or -1 (the issue stops). */
    int (*enqueue)(void *context, uint32_t member, uint32_t band, uint32_t head);
    /** Return once `member` has nothing outstanding. 0, or -1. */
    int (*settle)(void *context, uint32_t member);
} pn_tt_attn_deep_io;

/** One layer-pass's issue state: the next unit's index and each member's outstanding count. */
typedef struct pn_tt_attn_deep_flow {
    uint32_t members;
    uint32_t depth;
    uint32_t cursor;
    uint32_t outstanding[PN_TT_ATTN_DEEP_MAX_MEMBERS];
} pn_tt_attn_deep_flow;

/** Start a layer-pass over `members` (1..PN_TT_ATTN_DEEP_MAX_MEMBERS), at most `depth` (>= 1)
 *  outstanding a member. @return 0, -1 on a bad count */
int pn_tt_attn_deep_flow_begin(pn_tt_attn_deep_flow *flow, uint32_t members, uint32_t depth);

/** Issue band `band`'s `heads` units, each on member pn_tt_attn_deep_member(cursor++); when the
 *  member already holds `depth`, every member is settled first (a wave: the round robin fills
 *  them evenly). @return 0, or -1 at the first failed call */
int pn_tt_attn_deep_issue_band(pn_tt_attn_deep_flow *flow, uint32_t band, uint32_t heads,
                               const pn_tt_attn_deep_io *io);

/** Settle every member with anything outstanding; the flow is then empty. @return 0, or -1 when
 *  any settle failed (every member is still settled) */
int pn_tt_attn_deep_flow_drain(pn_tt_attn_deep_flow *flow, const pn_tt_attn_deep_io *io);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_ATTN_DEEP_H */
