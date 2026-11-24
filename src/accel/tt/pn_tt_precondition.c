/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_precondition.c
 * @brief S2 on-card completion: the planner that turns a puller program's AWAITs into wait
 *        sidecar entries plus lagged AWAITs. See pn_tt_precondition.h for the rule and why it
 *        is exact; pn_tt_precondition_verify.c checks its output from first principles.
 *
 * State, all in input order:
 *   - `run`: the AWAITs since the last head (a window opens at the next head);
 *   - `pending`: converted AWAITs not yet realized, oldest first, each with its window;
 *   - `known`: what the REALIZED AWAITs prove each queue has completed (the hoist tracker,
 *     with its pull-leader implication) — the only knowledge a puller-side write may use;
 *   - per queue: the window whose entry it carries, and its recent heads (first, end and the
 *     AWAITs seen before each) for the chain test of a lookahead past 1.
 *
 * @status S2 2.3.
 * @issues None
 * @todo None
 */
#include "pn_tt_precondition.h"

#include <stdlib.h>
#include <string.h>

/* Heads remembered per queue for the chain test; an older producer descriptor is unproven. */
#define PRE_ISSUE_HISTORY 32u

typedef struct {
    uint32_t item;     /* the input AWAIT */
    uint16_t queue;
    uint32_t count, step;
    uint32_t window;   /* pending: the window it opened */
    uint64_t ordinal;  /* AWAITs fed before it */
} pre_await;

typedef struct {
    uint32_t first, end;  /* descriptors [first, end) */
    uint64_t ordinal;     /* AWAITs fed before the head */
} pre_issue;

struct pn_tt_pre_planner {
    pn_tt_pre_config config;
    pn_tt_pre_emit_fn emit;
    void *user;
    uint32_t max_queues;
    pn_tt_hoist_tracker *known;
    bool *honours;
    uint32_t *entry_window;
    pre_issue *issue;      /* PRE_ISSUE_HISTORY a queue, a ring */
    uint32_t *issue_count; /* heads recorded a queue */
    uint32_t *published;   /* a queue's published count, once a head named its first index */
    bool *published_known;
    const pn_tt_hoist_group *groups;
    uint32_t group_count;
    pre_await run[PN_TT_PRE_MAX_PENDING];
    uint32_t run_count;
    pre_await pending[PN_TT_PRE_MAX_PENDING];
    uint32_t pending_count;
    bool has_window;
    uint32_t window_item;  /* the input AWAIT that is the current window's precondition */
    uint32_t window;
    uint64_t awaits_seen;
    pn_tt_pre_stats stats;
    bool failed;
};

pn_tt_pre_planner *pn_tt_precondition_create(uint32_t max_queues, const pn_tt_pre_config *config,
                                             pn_tt_pre_emit_fn emit, void *user)
{
    pn_tt_pre_planner *p;
    uint32_t q;

    if (!config || !emit || config->lookahead == 0u ||
        config->lookahead > PN_TT_PRE_MAX_LOOKAHEAD || max_queues == 0u)
        return NULL;
    p = calloc(1u, sizeof(*p));
    if (!p)
        return NULL;
    p->config = *config;
    p->emit = emit;
    p->user = user;
    p->max_queues = max_queues;
    p->known = pn_tt_hoist_tracker_create(max_queues);
    p->honours = calloc(max_queues, sizeof(*p->honours));
    p->entry_window = calloc(max_queues, sizeof(*p->entry_window));
    p->issue = calloc((size_t)max_queues * PRE_ISSUE_HISTORY, sizeof(*p->issue));
    p->issue_count = calloc(max_queues, sizeof(*p->issue_count));
    p->published = calloc(max_queues, sizeof(*p->published));
    p->published_known = calloc(max_queues, sizeof(*p->published_known));
    if (!p->known || !p->honours || !p->entry_window || !p->issue || !p->issue_count ||
        !p->published || !p->published_known) {
        pn_tt_precondition_destroy(p);
        return NULL;
    }
    for (q = 0; q < max_queues; q++)
        p->honours[q] = true;
    return p;
}

void pn_tt_precondition_destroy(pn_tt_pre_planner *p)
{
    if (!p)
        return;
    pn_tt_hoist_tracker_destroy(p->known);
    free(p->honours);
    free(p->entry_window);
    free(p->issue);
    free(p->issue_count);
    free(p->published);
    free(p->published_known);
    free(p);
}

void pn_tt_precondition_queue(pn_tt_pre_planner *p, uint32_t q, uint32_t capacity,
                              uint32_t step, int32_t leader, bool honours)
{
    if (!p || q >= p->max_queues)
        return;
    pn_tt_hoist_tracker_queue(p->known, q, capacity, step, leader);
    p->honours[q] = honours;
}

void pn_tt_precondition_groups(pn_tt_pre_planner *p, const pn_tt_hoist_group *groups,
                               uint32_t count)
{
    if (!p)
        return;
    p->groups = groups;
    p->group_count = count;
    pn_tt_hoist_tracker_groups(p->known, groups, count);
}

const pn_tt_pre_stats *pn_tt_precondition_stats(const pn_tt_pre_planner *p)
{
    return p ? &p->stats : NULL;
}

static pn_tt_hoist_item pre_hoist_item(const pn_tt_pre_item *it)
{
    pn_tt_hoist_item h;

    memset(&h, 0, sizeof(h));
    switch (it->kind) {
    case PN_TT_PRE_WRITE: h.kind = PN_TT_HOIST_WRITE; break;
    case PN_TT_PRE_RING: h.kind = PN_TT_HOIST_RING; break;
    case PN_TT_PRE_HEAD: h.kind = PN_TT_HOIST_HEAD; break;
    case PN_TT_PRE_AWAIT: h.kind = PN_TT_HOIST_AWAIT; break;
    case PN_TT_PRE_SEGMENT: h.kind = PN_TT_HOIST_SEGMENT; break;
    default: h.kind = PN_TT_HOIST_FIXED; break;
    }
    h.group = it->group;
    h.queue = it->queue;
    h.index = it->index;
    h.slots = it->slots;
    h.address = it->address;
    h.pull_sequence = it->pull_sequence;
    h.group_sequence = it->group_sequence;
    return h;
}

void pn_tt_precondition_carry(pn_tt_pre_planner *p, const pn_tt_pre_item *items, size_t n)
{
    size_t i;

    if (!p)
        return;
    for (i = 0; i < n; i++) {
        const pn_tt_hoist_item h = pre_hoist_item(&items[i]);

        (void)pn_tt_hoist_tracker_apply(p->known, &h);
    }
    pn_tt_hoist_tracker_next_token(p->known);
}

/** The queues an item names: its group's members, or its one queue. */
static uint32_t pre_item_queues(const pn_tt_pre_planner *p, const pn_tt_pre_item *it,
                                uint32_t out[4])
{
    if (it->group >= 0) {
        const pn_tt_hoist_group *g = &p->groups[it->group];
        uint32_t k, n = g->count < 4u ? g->count : 4u;

        for (k = 0; k < n; k++)
            out[k] = g->queues[k];
        return n;
    }
    out[0] = it->queue;
    return 1u;
}

/** Emit a real AWAIT for `a` and count it under `reason`. */
static void pre_realize(pn_tt_pre_planner *p, const pre_await *a, size_t *reason)
{
    pn_tt_pre_action act;
    pn_tt_hoist_item h;

    memset(&act, 0, sizeof(act));
    act.kind = PN_TT_PRE_OUT_AWAIT;
    act.group = -1;
    act.queue = a->queue;
    act.item = a->item;
    act.count = a->count;
    act.count_step = a->step;
    p->emit(p->user, &act);
    memset(&h, 0, sizeof(h));
    h.kind = PN_TT_HOIST_AWAIT;
    h.group = -1;
    h.queue = a->queue;
    h.index = a->count;
    (void)pn_tt_hoist_tracker_apply(p->known, &h);
    pn_tt_hoist_tracker_drop_undo(p->known);
    p->stats.awaits_out++;
    (*reason)++;
    if (p->has_window && p->window_item == a->item)
        p->has_window = false;
}

/** Realize the pending AWAITs of window `upto` or older, oldest first. */
static void pre_realize_pending(pn_tt_pre_planner *p, uint32_t upto, size_t *reason)
{
    uint32_t i, kept = 0u;

    for (i = 0; i < p->pending_count; i++) {
        if (p->pending[i].window <= upto)
            pre_realize(p, &p->pending[i], reason);
        else
            p->pending[kept++] = p->pending[i];
    }
    p->pending_count = kept;
}

/** Realize everything: pending (older) first, then the run. */
static void pre_realize_all(pn_tt_pre_planner *p, size_t *reason)
{
    uint32_t i;

    pre_realize_pending(p, UINT32_MAX, reason);
    for (i = 0; i < p->run_count; i++)
        pre_realize(p, &p->run[i], reason);
    p->run_count = 0u;
}

static void pre_emit_item(pn_tt_pre_planner *p, uint32_t index)
{
    pn_tt_pre_action act;

    memset(&act, 0, sizeof(act));
    act.kind = PN_TT_PRE_OUT_ITEM;
    act.group = -1;
    act.item = index;
    p->emit(p->user, &act);
}

/** Are the needs proven by the realized AWAITs? A need of count <= 0 overwrites nothing:
 *  descriptor indices count from the queue's arm, so the slot was never used. */
static bool pre_met(const pn_tt_pre_planner *p, const pn_tt_hoist_need *need, uint32_t needs)
{
    uint32_t k;

    for (k = 0; k < needs; k++)
        if (need[k].count > 0 && !pn_tt_hoist_tracker_met(p->known, &need[k], 1u, 0u, 0u))
            return false;
    return needs > 0u;
}

/** A ring slot or rotating parameter block: what it overwrites must be proven complete by
 *  the realized AWAITs; realize oldest first until it is, everything if it never is. */
static void pre_write(pn_tt_pre_planner *p, const pn_tt_pre_item *it)
{
    const pn_tt_hoist_item h = pre_hoist_item(it);
    pn_tt_hoist_need need[4];
    const uint32_t needs = pn_tt_hoist_tracker_needs(p->known, &h, need);

    while (!pre_met(p, need, needs) && p->pending_count > 0u) {
        pre_realize(p, &p->pending[0], &p->stats.by_write);
        memmove(&p->pending[0], &p->pending[1], (p->pending_count - 1u) * sizeof(*p->pending));
        p->pending_count--;
    }
    if (!pre_met(p, need, needs))
        pre_realize_all(p, &p->stats.by_write);
}

/** Is pending AWAIT `a` implied by precondition `pre` (the producer descriptor pre names was
 *  itself issued after `a`, or names the same queue at a count at least as high)? */
static bool pre_implied(const pn_tt_pre_planner *p, const pre_await *a, const pre_await *pre)
{
    const pre_issue *h = &p->issue[(size_t)pre->queue * PRE_ISSUE_HISTORY];
    const uint32_t target = pre->count - 1u;
    uint32_t n = p->issue_count[pre->queue], k;

    if (a->queue == pre->queue && (int32_t)(pre->count - a->count) >= 0)
        return true;
    if (n > PRE_ISSUE_HISTORY)
        n = PRE_ISSUE_HISTORY;
    for (k = 0; k < n; k++) {
        const pre_issue *e = &h[k];

        if ((int32_t)(target - e->first) >= 0 && (int32_t)(e->end - target) > 0)
            return e->ordinal > a->ordinal;
    }
    return false;
}

/** The first head after a run of AWAITs: the run's last one becomes the window's
 *  precondition; the others, and older pending ones it does not imply, are realized. */
static void pre_open_window(pn_tt_pre_planner *p)
{
    const pre_await last = p->run[p->run_count - 1u];
    uint32_t i, kept = 0u;

    for (i = 0; i + 1u < p->run_count; i++) {
        if (p->run[i].queue == last.queue && (int32_t)(last.count - p->run[i].count) >= 0)
            p->stats.merged++;
        else
            pre_realize(p, &p->run[i], &p->stats.by_run);
    }
    p->run_count = 0u;
    for (i = 0; i < p->pending_count; i++) {
        if (pre_implied(p, &p->pending[i], &last))
            p->pending[kept++] = p->pending[i];
        else
            pre_realize(p, &p->pending[i], &p->stats.by_chain);
    }
    p->pending_count = kept;
    if (last.count == 0u)
        return; /* nothing to wait for */
    if (p->pending_count == PN_TT_PRE_MAX_PENDING) {
        pre_realize(p, &p->pending[0], &p->stats.by_pending);
        memmove(&p->pending[0], &p->pending[1], (p->pending_count - 1u) * sizeof(*p->pending));
        p->pending_count--;
    }
    p->window++;
    p->pending[p->pending_count] = last;
    p->pending[p->pending_count].window = p->window;
    p->pending_count++;
    p->has_window = true;
    p->window_item = last.item;
    p->stats.windows++;
}

/** Realize every pending AWAIT on a queue in `qs` before a head publishes past its count. The
 *  puller accepts an AWAIT's count only with the token of descriptor count - 1 beside it, so
 *  once a later descriptor of that queue may complete, an AWAIT left for later can never be
 *  satisfied (resident_puller.c kPlrErrAwaitToken; card D=8, 2026-11-17). */
static void pre_before_publish(pn_tt_pre_planner *p, const uint32_t *qs, uint32_t nq)
{
    uint32_t i, k, kept = 0u;

    for (i = 0; i < p->pending_count; i++) {
        bool named = false;

        for (k = 0; k < nq; k++)
            named = named || p->pending[i].queue == qs[k];
        if (named)
            pre_realize(p, &p->pending[i], &p->stats.by_publish);
        else
            p->pending[kept++] = p->pending[i];
    }
    p->pending_count = kept;
}

/** The window's entry on a head's descriptors, unless every queue it names has one. */
static void pre_head(pn_tt_pre_planner *p, uint32_t index, const pn_tt_pre_item *it)
{
    uint32_t qs[4], nq = pre_item_queues(p, it, qs), k;
    bool missing = false, honoured = it->index != UINT32_MAX;

    if (p->run_count > 0u)
        pre_open_window(p);
    pre_before_publish(p, qs, nq);
    for (k = 0; k < nq; k++) {
        missing = missing || p->entry_window[qs[k]] != p->window;
        honoured = honoured && p->honours[qs[k]];
    }
    if (p->has_window && missing && !honoured)
        pre_realize_all(p, &p->stats.by_refusal);
    if (p->has_window && missing) {
        const pre_await *pre = &p->pending[p->pending_count - 1u];
        pn_tt_pre_action act;

        memset(&act, 0, sizeof(act));
        act.kind = PN_TT_PRE_OUT_PRECOND;
        act.group = it->group;
        act.queue = it->queue;
        act.producer = pre->queue;
        act.item = index;
        act.index = it->index;
        act.index_step = it->step;
        act.count = pre->count - (p->config.red_early && pre->count > 1u ? 1u : 0u);
        act.count_step = pre->step;
        p->emit(p->user, &act);
        for (k = 0; k < nq; k++)
            p->entry_window[qs[k]] = p->window;
        p->stats.preconds++;
    }
    pre_emit_item(p, index);
    for (k = 0; k < nq && it->index == UINT32_MAX; k++)
        p->published_known[qs[k]] = false; /* an unplaced head: the count is unknown again */
    for (k = 0; k < nq && it->index != UINT32_MAX; k++) {
        pre_issue *e = &p->issue[(size_t)qs[k] * PRE_ISSUE_HISTORY +
                                 p->issue_count[qs[k]] % PRE_ISSUE_HISTORY];

        e->first = it->index;
        e->end = it->index + it->slots;
        e->ordinal = p->awaits_seen;
        p->issue_count[qs[k]]++;
        /* the plan check's rule (pn_tt_precondition_verify.c ver_head): a count only grows */
        if (!p->published_known[qs[k]] || p->published[qs[k]] == it->index ||
            (int32_t)(it->index + it->slots - p->published[qs[k]]) > 0)
            p->published[qs[k]] = it->index + it->slots;
        p->published_known[qs[k]] = true;
    }
}

static void pre_await_item(pn_tt_pre_planner *p, uint32_t index, const pn_tt_pre_item *it)
{
    pre_await a;

    memset(&a, 0, sizeof(a));
    a.item = index;
    a.queue = it->queue;
    a.count = it->index;
    a.step = it->step;
    /* HELD: the program already published this queue past the count it awaits (a descriptor
     * out before its await, finishing only after it). The puller takes the count only with
     * descriptor count - 1's token beside it, so the await goes out HERE, where the program
     * has it — any later and the held descriptor may finish first (card replay with
     * PN_TT_MATMUL_PULL_DEFER=0, 2026-11-17). Not converted, not part of a window. */
    if (p->published_known[it->queue] && p->published[it->queue] != it->index) {
        a.ordinal = p->awaits_seen++;
        p->stats.awaits_in++;
        pre_realize(p, &a, &p->stats.by_held);
        return;
    }

    /* A new run: window `window + 1` may open, so the puller realizes what is lookahead
     * windows back (all of it at lookahead 1) before anything of that window goes out. */
    if (p->run_count == 0u && p->window + 1u >= p->config.lookahead)
        pre_realize_pending(p, p->window + 1u - p->config.lookahead, &p->stats.by_lag);
    if (p->run_count == PN_TT_PRE_MAX_PENDING) {
        pre_realize(p, &p->run[0], &p->stats.by_run);
        memmove(&p->run[0], &p->run[1], (p->run_count - 1u) * sizeof(*p->run));
        p->run_count--;
    }
    a.ordinal = p->awaits_seen++;
    p->run[p->run_count++] = a;
    p->stats.awaits_in++;
}

int pn_tt_precondition_push(pn_tt_pre_planner *p, uint32_t index, const pn_tt_pre_item *it)
{
    if (!p || !it || p->failed)
        return -1;
    if ((it->kind != PN_TT_PRE_NEUTRAL && it->kind != PN_TT_PRE_BARRIER &&
         it->kind != PN_TT_PRE_SEGMENT && it->queue >= p->max_queues) ||
        (it->group >= 0 && (uint32_t)it->group >= p->group_count)) {
        p->failed = true;
        return -1;
    }
    switch (it->kind) {
    case PN_TT_PRE_AWAIT:
        pre_await_item(p, index, it);
        break;
    case PN_TT_PRE_HEAD:
        pre_head(p, index, it);
        break;
    case PN_TT_PRE_WRITE:
    case PN_TT_PRE_RING: {
        const pn_tt_hoist_item h = pre_hoist_item(it);

        pre_write(p, it);
        pre_emit_item(p, index);
        (void)pn_tt_hoist_tracker_apply(p->known, &h);
        break;
    }
    case PN_TT_PRE_SEGMENT:
        pre_realize_all(p, &p->stats.by_segment);
        pre_emit_item(p, index);
        break;
    case PN_TT_PRE_BARRIER:
        pre_realize_all(p, &p->stats.by_barrier);
        pre_emit_item(p, index);
        break;
    default:
        pre_emit_item(p, index);
        break;
    }
    if (pn_tt_hoist_tracker_failed(p->known))
        p->failed = true;
    return p->failed ? -1 : 0;
}

int pn_tt_precondition_drain(pn_tt_pre_planner *p)
{
    if (!p || p->failed)
        return -1;
    pre_realize_all(p, &p->stats.by_segment);
    return 0;
}
