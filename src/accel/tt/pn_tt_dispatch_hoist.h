/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_dispatch_hoist.h
 * @brief L10 — HOISTING in the puller's replay program: a descriptor's parameter block and ring
 *        slot are written while the core the puller is awaiting still works, so after the
 *        await only the head bump stands between one core finishing and the next starting.
 *
 * The recorded program (pn_tt_dispatch.c) is the host's own event order: AWAIT P, then the
 * next descriptor's WRITE (rotating parameter block), RING (slot) and HEAD. On the card the
 * puller does all three AFTER P is done, so their copy + NoC write + barrier are on the
 * critical path once per handoff (~875 a plain token). This planner moves each movable
 * WRITE / RING to just before the nearest preceding AWAIT, and the builder puts a FLUSH entry
 * (resident_puller_abi.h) in front of each hoisted block, so the heads queued before the await
 * still go out first and the hoisted writes run under the await.
 *
 * WHY IT IS EXACT. The target reads a ring slot (and the parameter block its a0 names) only
 * once the HEAD says the slot is published, and heads never move — every core receives the
 * same descriptors in the same order after the same awaits. A write may move above an AWAIT
 * only when what it OVERWRITES is provably finished at the new position:
 *   - ring slot of descriptor i, capacity C: descriptor i-C has completed;
 *   - a parameter block at address A: the previous descriptor that used A has completed.
 * "Completed" is a lower bound on each queue's completed count, raised only by the AWAITs the
 * program has already passed (and, for matmul members the host never awaits, by their pull
 * leader's AWAIT: member 0 completes a descriptor with pull sequence s only after every member
 * published a group sequence >= s). Across tokens the bound from the end of the previous token
 * is carried, shifted by the queue's per-token index step. Unknown = never satisfied. Writes
 * never pass writes (each stream keeps its order: the dry run still compares byte for byte),
 * nothing crosses a segment end, and a write crosses at most ONE await.
 *
 * The TRACKER is the rule on its own (what each queue has provably completed); the replay
 * PLANNER runs it over a recorded token, the streamed path (S2d) one captured entry at a time.
 * Pure C over plain arrays — `tests/test_tt_dispatch_hoist.c` proves it off-card, including
 * that every unsafe move is refused.
 *
 * @status L10 B1 (replay) / B2 (stream).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_DISPATCH_HOIST_H
#define PN_TT_DISPATCH_HOIST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    PN_TT_HOIST_FIXED = 0, /* stays; nothing moves above it (non-parameter queue writes) */
    PN_TT_HOIST_WRITE,     /* a rotating parameter block: descriptor `index` of `queue` at `address` */
    PN_TT_HOIST_RING,      /* ring slots [index, index + slots) of `queue`, or of every member of `group` */
    PN_TT_HOIST_HEAD,      /* a head bump of `queue` / `group`: movable writes pass it */
    PN_TT_HOIST_AWAIT,     /* the puller waits for `queue`'s completed count to reach `index` */
    PN_TT_HOIST_SEGMENT    /* a segment end */
} pn_tt_hoist_kind;

typedef struct {
    uint8_t kind;
    int16_t group;           /* RING / HEAD: multicast group index, -1 = unicast `queue` */
    uint16_t queue;
    uint32_t index;          /* see the kind */
    uint32_t slots;          /* RING */
    uint32_t address;        /* WRITE */
    uint32_t pull_sequence;  /* RING on a pull leader: the a4 its members must reach (0 none) */
    uint32_t group_sequence; /* RING on a member: its a5 (0 none) */
} pn_tt_hoist_item;

typedef struct {
    uint32_t capacity; /* ring slots */
    uint32_t step;     /* descriptor indices a token advances */
    int32_t leader;    /* a never-awaited member: the queue whose pull implies it, else -1 */
} pn_tt_hoist_queue;

typedef struct {
    uint16_t queues[4];
    uint32_t count;
} pn_tt_hoist_group;

typedef struct {
    size_t moved;     /* items placed above an await */
    size_t blocked;   /* movable items left where they were (a bound not proven) */
    size_t flushes;   /* FLUSH entries the plan asks for */
} pn_tt_hoist_stats;

typedef struct {
    uint32_t queue;
    int64_t count; /* the completed count the queue must have reached */
} pn_tt_hoist_need;

/* ---- the tracker ---- */
typedef struct pn_tt_hoist_tracker pn_tt_hoist_tracker;

/** Bounds for `max_queues` queues, every one unknown. NULL when out of memory. */
pn_tt_hoist_tracker *pn_tt_hoist_tracker_create(uint32_t max_queues);
void pn_tt_hoist_tracker_destroy(pn_tt_hoist_tracker *t);
/** Every bound unknown again, every history empty (a new stream, a re-armed queue). */
void pn_tt_hoist_tracker_forget(pn_tt_hoist_tracker *t);
/** Queue `q`'s ring capacity, per-token step (0 for a stream) and pull leader (-1 none). */
void pn_tt_hoist_tracker_queue(pn_tt_hoist_tracker *t, uint32_t q, uint32_t capacity,
                               uint32_t step, int32_t leader);
/** The multicast groups RING / HEAD items may name (kept by pointer). */
void pn_tt_hoist_tracker_groups(pn_tt_hoist_tracker *t, const pn_tt_hoist_group *groups,
                                uint32_t count);
/** What a movable item overwrites, as completed counts; 0 = it may not move. */
uint32_t pn_tt_hoist_tracker_needs(pn_tt_hoist_tracker *t, const pn_tt_hoist_item *it,
                                   pn_tt_hoist_need need[4]);
/** Are the needs met just before the AWAIT whose bound changes are [undo_from, undo_to)
 *  (the marks around its apply)? With no await crossed, pass undo_from == undo_to. */
bool pn_tt_hoist_tracker_met(const pn_tt_hoist_tracker *t, const pn_tt_hoist_need *need,
                             uint32_t count, size_t undo_from, size_t undo_to);
/** Apply an item (after its needs were read). @return the undo mark before it */
size_t pn_tt_hoist_tracker_apply(pn_tt_hoist_tracker *t, const pn_tt_hoist_item *it);
size_t pn_tt_hoist_tracker_undo_mark(const pn_tt_hoist_tracker *t);
/** Forget the undo history (no await before this point will be crossed any more). */
void pn_tt_hoist_tracker_drop_undo(pn_tt_hoist_tracker *t);
bool pn_tt_hoist_tracker_failed(const pn_tt_hoist_tracker *t);
/** Carry the bounds into the next token of a replay: every bound and parameter user shifted
 *  back by its queue's step, the pull spans and the undo history dropped. */
void pn_tt_hoist_tracker_next_token(pn_tt_hoist_tracker *t);

/* ---- the replay planner ---- */

/** In `order`, this value is a FLUSH entry (every other value is an item index). */
#define PN_TT_HOIST_FLUSH_MARK 0xFFFFFFFFu

/**
 * Plan the program order. `order` receives every item index once (plus FLUSH marks) in the
 * order the entries are to be emitted; it needs room for 2 * n entries.
 *
 * @return entries written to `order`, or 0 if the input is malformed (a queue index out of
 *         range, out of memory) — the caller then emits the original order.
 */
size_t pn_tt_hoist_plan(const pn_tt_hoist_item *items, size_t n, const pn_tt_hoist_queue *queues,
                        uint32_t queue_count, const pn_tt_hoist_group *groups,
                        uint32_t group_count, uint32_t *order, pn_tt_hoist_stats *stats);

#endif /* PN_TT_DISPATCH_HOIST_H */
