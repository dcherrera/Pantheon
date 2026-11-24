/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_precondition.h
 * @brief S2 ON-CARD COMPLETION in the puller's programs: an AWAIT followed by heads becomes a
 *        WAIT SIDECAR entry on those heads' descriptors (their readers check the producer on
 *        the card) plus a real AWAIT `lookahead` windows later.
 *
 * Today the puller AWAITs window W_{k-1} (the descriptors issued since the previous await)
 * before it issues W_k, so every handoff costs a poll, a head flush and the consumer's own
 * pickup in series. This planner
 * rewrites a program, in the order the builder would emit it, so that:
 *   - the first descriptor W_k opens on each queue carries a PRECOND entry "producer P has
 *     completed C" (the last AWAIT before W_k), written after its ring slot and before its
 *     head; its reader waits for it on the card (foundry resident_precondition.h);
 *   - that AWAIT is REALIZED (emitted as a real puller AWAIT) once window W_{k+lookahead}
 *     begins, so the puller runs at most `lookahead` windows ahead.
 *
 * WHY IT IS EXACT. Every descriptor must start only after every AWAIT that preceded its head
 * in the original program holds. With lookahead 1 the puller has realized every earlier
 * window's AWAIT before it issues W_k, and the entry covers W_k's own; with a larger
 * lookahead an older unrealized AWAIT is kept only when the entry's producer descriptor was
 * itself issued after it (it then started after that AWAIT held), and realized otherwise.
 * A later descriptor on the same queue inherits the first one's start (the reader is
 * sequential). Everything the planner cannot prove keeps a real AWAIT where it stood:
 *   - a ring slot or rotating parameter block whose previous user is not proven complete by
 *     the REALIZED awaits (the puller writes it; pn_tt_dispatch_hoist.h's tracker decides);
 *   - any write it does not track (data, parameter cache, other queue words, a row copy):
 *     every converted AWAIT is realized first (a BARRIER);
 *   - a segment end (the host touches data next): everything realized;
 *   - several AWAITs in one run: all but the last are realized (one entry names one
 *     producer); a target queue whose reader does not honour the sidecar: realized;
 *   - an AWAIT on a queue about to publish past its count (a head on that queue): realized
 *     first. The puller's AWAIT takes a count only with the token of descriptor count - 1
 *     beside it, so it must go out while that is still the queue's last descriptor.
 * pn_tt_precondition_verify() checks a plan against its input independently of these rules
 * (a vector-clock model of what each descriptor's start is guaranteed), and the builder runs
 * it on every replay program before it uses one.
 *
 * Online: items go in one at a time and actions come out through a callback, so the replay
 * builder (a whole recorded token) and the streamed verify (one captured access at a time)
 * share it. Pure C over plain arrays; tests/test_tt_precondition.c proves it off-card.
 *
 * @status S2 2.3. On by default (David 2026-11-17); card
 *         gate passed.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_PN_TT_PRECONDITION_H
#define PANTHEON_PN_TT_PRECONDITION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pn_tt_dispatch_hoist.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The widest lookahead accepted (`PN_TT_ONCARD_LOOKAHEAD`). */
#define PN_TT_PRE_MAX_LOOKAHEAD 8u
/** Converted AWAITs held unrealized at once; the oldest is realized past this. */
#define PN_TT_PRE_MAX_PENDING 64u

typedef enum {
    PN_TT_PRE_NEUTRAL = 0, /* orders nothing (a FLUSH) */
    PN_TT_PRE_BARRIER,     /* an untracked write or a row copy: realize everything first */
    PN_TT_PRE_WRITE,       /* rotating parameter block of `queue`: descriptor `index`, `address` */
    PN_TT_PRE_RING,        /* ring slots [index, index + slots) of `queue` / `group` */
    PN_TT_PRE_HEAD,        /* head of `queue` / `group`: descriptors [index, index + slots) */
    PN_TT_PRE_AWAIT,       /* the program waits for `queue`'s completed count to reach `index` */
    PN_TT_PRE_SEGMENT      /* a segment end */
} pn_tt_pre_kind;

typedef struct {
    uint8_t kind;
    int16_t group;           /* RING / HEAD: multicast group, -1 = unicast `queue` */
    uint16_t queue;
    uint32_t index;          /* see the kind; UINT32_MAX on a HEAD = first descriptor unknown */
    uint32_t slots;          /* RING slots; HEAD descriptors */
    uint32_t address;        /* WRITE */
    uint32_t step;           /* per-token advance of `index` (a replay); 0 in a stream */
    uint32_t pull_sequence;  /* RING: as pn_tt_hoist_item (the pull-leader implication) */
    uint32_t group_sequence;
} pn_tt_pre_item;

typedef enum {
    PN_TT_PRE_OUT_ITEM = 0, /* emit input item `item` as it stands */
    PN_TT_PRE_OUT_AWAIT,    /* a real AWAIT of `queue` for `count` (realizes input item `item`) */
    PN_TT_PRE_OUT_PRECOND   /* a sidecar entry: descriptor `index` of `queue` / `group` waits for
                               `producer` to complete `count` (before input HEAD `item`) */
} pn_tt_pre_out_kind;

typedef struct {
    uint8_t kind;
    int16_t group;
    uint16_t queue;
    uint16_t producer;
    uint32_t item;
    uint32_t index, index_step;
    uint32_t count, count_step;
} pn_tt_pre_action;

/** Why AWAITs were realized (a census for the logs and tests). */
typedef struct {
    size_t awaits_in;       /* AWAIT items given */
    size_t awaits_out;      /* real AWAITs emitted */
    size_t preconds;        /* sidecar entries emitted */
    size_t windows;         /* windows that opened with a precondition */
    size_t merged;          /* AWAITs implied by a later one in their run (same queue) */
    size_t by_lag, by_write, by_barrier, by_segment, by_run, by_chain, by_refusal, by_pending;
    size_t by_publish;      /* realized before a head on its own queue published past it */
    size_t by_held;         /* HELD: its queue published past it already; left in place */
} pn_tt_pre_stats;

typedef struct {
    uint32_t lookahead;  /* windows the puller may run ahead, >= 1 */
    bool red_early;      /* RED `precondition_early`: every entry aims one count early */
} pn_tt_pre_config;

typedef void (*pn_tt_pre_emit_fn)(void *user, const pn_tt_pre_action *action);

typedef struct pn_tt_pre_planner pn_tt_pre_planner;

/** A planner for queues 0..max_queues-1. NULL when out of memory or `config` is invalid
 *  (lookahead 0 or past PN_TT_PRE_MAX_LOOKAHEAD). Caller frees with ..._destroy(). */
pn_tt_pre_planner *pn_tt_precondition_create(uint32_t max_queues, const pn_tt_pre_config *config,
                                             pn_tt_pre_emit_fn emit, void *user);
void pn_tt_precondition_destroy(pn_tt_pre_planner *p);
/** Queue `q`'s ring capacity, per-token step, pull leader (-1) and whether its reader honours
 *  the sidecar. Unset queues have capacity 0 (every ring write unproven) and honour it. */
void pn_tt_precondition_queue(pn_tt_pre_planner *p, uint32_t q, uint32_t capacity,
                              uint32_t step, int32_t leader, bool honours);
/** The multicast groups RING / HEAD items may name (kept by pointer). */
void pn_tt_precondition_groups(pn_tt_pre_planner *p, const pn_tt_hoist_group *groups,
                               uint32_t count);
/** Replay only: take the completion bounds and parameter users a whole token leaves behind,
 *  shifted back one token, as the starting knowledge (pn_tt_dispatch_hoist.c's carry). */
void pn_tt_precondition_carry(pn_tt_pre_planner *p, const pn_tt_pre_item *items, size_t n);
/** Feed input item `index` (callers number items 0, 1, ... in feed order). Actions for it —
 *  realized AWAITs, PRECONDs, then the item itself — come out through the callback.
 *  @return 0, or -1 if the item is malformed or memory ran out (the plan is then void). */
int pn_tt_precondition_push(pn_tt_pre_planner *p, uint32_t index, const pn_tt_pre_item *it);
/** Realize every converted AWAIT now (a stream ends; the caller then emits its own end). */
int pn_tt_precondition_drain(pn_tt_pre_planner *p);
const pn_tt_pre_stats *pn_tt_precondition_stats(const pn_tt_pre_planner *p);

/** What a plan check found. */
typedef struct {
    size_t heads, writes, segments, barriers;
    size_t failures;
    char first[192]; /* the first failure, readable */
    uint32_t first_queue, first_value; /* its first two numbers (a queue, a count / index) */
} pn_tt_pre_report;

typedef struct {
    uint32_t capacity; /* ring slots, 0 = unknown */
    bool honours;
} pn_tt_pre_queue_info;

/**
 * Check `out` (a plan's actions) against `in` (the items it was planned from), from first
 * principles: every descriptor's start is guaranteed after every AWAIT that preceded its head
 * in `in`; every ring slot and rotating parameter block the plan writes is proven free at
 * least wherever the original program proved it; every segment end and BARRIER is preceded
 * by every AWAIT before it; every AWAIT and entry names a published descriptor; entries sit
 * between their descriptor's previous head and its own and only on queues that honour them.
 * @return true when the plan passes; `report` (may be NULL) says what was checked and the
 *         first failure. Out of memory is a failure.
 *
 * Thread-safe: yes (no shared state)
 * Reentrant:   yes
 * Signal-safe: no (allocates)
 */
bool pn_tt_precondition_verify(const pn_tt_pre_item *in, size_t n_in, const pn_tt_pre_action *out,
                               size_t n_out, const pn_tt_pre_queue_info *queues,
                               uint32_t queue_count, const pn_tt_hoist_group *groups,
                               uint32_t group_count, pn_tt_pre_report *report);

/* ---- `PN_TT_ONCARD_DUMP`: a plan and everything it was planned from, as text ---- */

/** One planning problem and (optionally) its plan. Owned arrays when read from a file. */
typedef struct {
    pn_tt_pre_config config;
    uint32_t queue_count;
    uint32_t *capacity, *step;
    int32_t *leader;
    bool *honours;
    uint32_t group_count;
    pn_tt_hoist_group *groups;
    pn_tt_pre_item *items;
    size_t n_items;
    pn_tt_pre_action *actions;
    size_t n_actions;
} pn_tt_pre_dump;

/**
 * Write `d` as text (one line a queue, group, item and action; "PNPRE 1" first, "end" last).
 * @return 0, or -1 on a write error
 *
 * Thread-safe: yes (the caller owns `f`)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_tt_precondition_dump_write(FILE *f, const pn_tt_pre_dump *d);
/**
 * Read a dump written by pn_tt_precondition_dump_write() into freshly allocated arrays.
 * @return 0, or -1 on a malformed file or out of memory (`d` is then freed and zeroed)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_tt_precondition_dump_read(FILE *f, pn_tt_pre_dump *d);
/** Free what pn_tt_precondition_dump_read() allocated. */
void pn_tt_precondition_dump_free(pn_tt_pre_dump *d);

#ifdef __cplusplus
}
#endif

#endif /* PANTHEON_PN_TT_PRECONDITION_H */
