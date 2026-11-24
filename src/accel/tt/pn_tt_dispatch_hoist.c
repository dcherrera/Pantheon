/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_dispatch_hoist.c
 * @brief L10 — the hoisting rule (a tracker of what each queue has provably completed) and the
 *        replay planner built on it. See pn_tt_dispatch_hoist.h for why a move is exact.
 *
 * The replay planner makes two passes over a token's items in the host's order:
 *   1. SIMULATE: the completed-count bound each queue reaches by the end of the token, and
 *      the last descriptor that used each parameter address — the previous token's state as
 *      the next one sees it (shifted back by the queue's step).
 *   2. PLAN: walk again from that carried state; a movable item goes just before the nearest
 *      preceding AWAIT (only HEADs between them) when what it overwrites is finished there.
 * The streamed path (pn_tt_dispatch.c, S2d) drives the same tracker one entry at a time.
 *
 * @status L10 B1/B2.
 * @issues None
 * @todo None
 */
#include "pn_tt_dispatch_hoist.h"

#include <stdlib.h>
#include <string.h>

#define HOIST_LB_NONE INT64_MIN
#define HOIST_PARAM_ADDRESSES 32u
#define HOIST_SPANS_KEEP 256u /* a long stream keeps the most recent descriptors' sequences */

typedef struct {
    uint32_t start, end, sequence;
} hoist_span;

typedef struct {
    hoist_span *spans;
    size_t count, capacity;
} hoist_spans;

typedef struct {
    int64_t lb;                 /* completed count known reached, HOIST_LB_NONE = unknown */
    hoist_spans lead;           /* this queue's descriptors' pull sequences (a pull leader) */
    hoist_spans member;         /* this queue's descriptors' group sequences (a member) */
    uint32_t param_address[HOIST_PARAM_ADDRESSES];
    int64_t param_last[HOIST_PARAM_ADDRESSES]; /* last descriptor index using the address */
    uint32_t param_count;
} hoist_queue_state;

typedef struct {
    uint32_t queue;
    int64_t before;
} hoist_undo;

struct pn_tt_hoist_tracker {
    pn_tt_hoist_queue *queues;
    hoist_queue_state *state;
    uint32_t max_queues;
    const pn_tt_hoist_group *groups;
    uint32_t group_count;
    hoist_undo *undo;
    size_t undo_count, undo_capacity;
    bool failed;
};

pn_tt_hoist_tracker *pn_tt_hoist_tracker_create(uint32_t max_queues)
{
    pn_tt_hoist_tracker *t = calloc(1u, sizeof(*t));
    uint32_t q;

    if (!t)
        return NULL;
    t->max_queues = max_queues ? max_queues : 1u;
    t->queues = calloc(t->max_queues, sizeof(*t->queues));
    t->state = calloc(t->max_queues, sizeof(*t->state));
    if (!t->queues || !t->state) {
        pn_tt_hoist_tracker_destroy(t);
        return NULL;
    }
    for (q = 0; q < t->max_queues; q++) {
        t->queues[q].leader = -1;
        t->state[q].lb = HOIST_LB_NONE;
    }
    return t;
}

void pn_tt_hoist_tracker_destroy(pn_tt_hoist_tracker *t)
{
    uint32_t q;

    if (!t)
        return;
    for (q = 0; t->state && q < t->max_queues; q++) {
        free(t->state[q].lead.spans);
        free(t->state[q].member.spans);
    }
    free(t->state);
    free(t->queues);
    free(t->undo);
    free(t);
}

void pn_tt_hoist_tracker_forget(pn_tt_hoist_tracker *t)
{
    uint32_t q;

    for (q = 0; q < t->max_queues; q++) {
        t->state[q].lb = HOIST_LB_NONE;
        t->state[q].lead.count = 0u;
        t->state[q].member.count = 0u;
        t->state[q].param_count = 0u;
    }
    t->undo_count = 0u;
}

void pn_tt_hoist_tracker_queue(pn_tt_hoist_tracker *t, uint32_t q, uint32_t capacity,
                               uint32_t step, int32_t leader)
{
    if (q >= t->max_queues)
        return;
    t->queues[q].capacity = capacity;
    t->queues[q].step = step;
    t->queues[q].leader = leader;
}

void pn_tt_hoist_tracker_groups(pn_tt_hoist_tracker *t, const pn_tt_hoist_group *groups,
                                uint32_t count)
{
    t->groups = groups;
    t->group_count = count;
}

size_t pn_tt_hoist_tracker_undo_mark(const pn_tt_hoist_tracker *t)
{
    return t->undo_count;
}

void pn_tt_hoist_tracker_drop_undo(pn_tt_hoist_tracker *t)
{
    t->undo_count = 0u;
}

bool pn_tt_hoist_tracker_failed(const pn_tt_hoist_tracker *t)
{
    return t->failed;
}

static bool hoist_spans_push(hoist_spans *s, uint32_t start, uint32_t end, uint32_t sequence)
{
    if (s->count >= 2u * HOIST_SPANS_KEEP) {
        memmove(s->spans, s->spans + (s->count - HOIST_SPANS_KEEP),
                HOIST_SPANS_KEEP * sizeof(*s->spans));
        s->count = HOIST_SPANS_KEEP;
    }
    if (s->count == s->capacity) {
        const size_t grow = s->capacity ? s->capacity * 2u : 64u;
        hoist_span *more = realloc(s->spans, grow * sizeof(*more));

        if (!more)
            return false;
        s->spans = more;
        s->capacity = grow;
    }
    s->spans[s->count].start = start;
    s->spans[s->count].end = end;
    s->spans[s->count].sequence = sequence;
    s->count++;
    return true;
}

static int64_t *hoist_param_slot(hoist_queue_state *q, uint32_t address)
{
    uint32_t i;

    for (i = 0; i < q->param_count; i++)
        if (q->param_address[i] == address)
            return &q->param_last[i];
    if (q->param_count == HOIST_PARAM_ADDRESSES)
        return NULL;
    q->param_address[q->param_count] = address;
    q->param_last[q->param_count] = HOIST_LB_NONE;
    return &q->param_last[q->param_count++];
}

/** Raise queue `q`'s bound to `value`; the old value goes on the undo list. */
static void hoist_raise(pn_tt_hoist_tracker *t, uint32_t q, int64_t value)
{
    hoist_queue_state *s = &t->state[q];

    if (s->lb != HOIST_LB_NONE && s->lb >= value)
        return;
    if (t->undo_count == t->undo_capacity) {
        const size_t grow = t->undo_capacity ? t->undo_capacity * 2u : 1024u;
        hoist_undo *more = realloc(t->undo, grow * sizeof(*more));

        if (!more) {
            t->failed = true;
            return;
        }
        t->undo = more;
        t->undo_capacity = grow;
    }
    t->undo[t->undo_count].queue = q;
    t->undo[t->undo_count].before = s->lb;
    t->undo_count++;
    s->lb = value;
}

/** An AWAIT on `q` reaching `wanted`: q's own bound, and its members' (a member's
 *  descriptors with a group sequence <= the pull sequence member 0 waited for). */
static void hoist_await(pn_tt_hoist_tracker *t, uint32_t q, uint32_t wanted)
{
    const hoist_spans *lead = &t->state[q].lead;
    uint32_t sequence = 0u, m;
    size_t i;
    bool found = false;

    hoist_raise(t, q, (int64_t)wanted);
    if (wanted == 0u)
        return;
    for (i = lead->count; i-- > 0;)
        if (lead->spans[i].start <= wanted - 1u && wanted - 1u < lead->spans[i].end) {
            sequence = lead->spans[i].sequence;
            found = sequence != 0u;
            break;
        }
    if (!found)
        return;
    for (m = 0; m < t->max_queues; m++) {
        const hoist_spans *mem = &t->state[m].member;

        if (t->queues[m].leader != (int32_t)q)
            continue;
        for (i = mem->count; i-- > 0;)
            if ((int32_t)(mem->spans[i].sequence - sequence) <= 0) {
                size_t k;
                bool all_posted = true;

                /* a descriptor that posts no sequence (0) breaks the chain of implication */
                for (k = 0; k <= i; k++)
                    if (mem->spans[k].sequence == 0u)
                        all_posted = false;
                if (all_posted)
                    hoist_raise(t, m, (int64_t)mem->spans[i].end);
                break;
            }
    }
}

/** The queues a RING / HEAD / WRITE item touches (a group's members, or its one queue). */
static uint32_t hoist_item_queues(const pn_tt_hoist_tracker *t, const pn_tt_hoist_item *it,
                                  uint32_t out[4])
{
    if (it->group >= 0 && (uint32_t)it->group < t->group_count) {
        const pn_tt_hoist_group *g = &t->groups[it->group];
        uint32_t k;

        for (k = 0; k < g->count && k < 4u; k++)
            out[k] = g->queues[k];
        return g->count < 4u ? g->count : 4u;
    }
    out[0] = it->queue;
    return 1u;
}

size_t pn_tt_hoist_tracker_apply(pn_tt_hoist_tracker *t, const pn_tt_hoist_item *it)
{
    const size_t from = t->undo_count;
    uint32_t qs[4], k, nq;

    switch (it->kind) {
    case PN_TT_HOIST_WRITE: {
        int64_t *last = hoist_param_slot(&t->state[it->queue], it->address);

        if (last)
            *last = (int64_t)it->index;
        break;
    }
    case PN_TT_HOIST_RING:
        nq = hoist_item_queues(t, it, qs);
        for (k = 0; k < nq; k++) {
            hoist_queue_state *s = &t->state[qs[k]];

            if (it->pull_sequence != 0u &&
                !hoist_spans_push(&s->lead, it->index, it->index + it->slots, it->pull_sequence))
                t->failed = true;
            if (t->queues[qs[k]].leader >= 0 &&
                !hoist_spans_push(&s->member, it->index, it->index + it->slots,
                                  it->group_sequence))
                t->failed = true;
        }
        break;
    case PN_TT_HOIST_AWAIT:
        hoist_await(t, it->queue, it->index);
        break;
    default:
        break;
    }
    return from;
}

uint32_t pn_tt_hoist_tracker_needs(pn_tt_hoist_tracker *t, const pn_tt_hoist_item *it,
                                   pn_tt_hoist_need out[4])
{
    uint32_t qs[4], k, nq;

    if (it->kind == PN_TT_HOIST_WRITE) {
        int64_t *last = hoist_param_slot(&t->state[it->queue], it->address);

        if (!last || *last == HOIST_LB_NONE)
            return 0u;
        out[0].queue = it->queue;
        out[0].count = *last + 1;
        return 1u;
    }
    if (it->kind != PN_TT_HOIST_RING || it->slots == 0u)
        return 0u;
    nq = hoist_item_queues(t, it, qs);
    for (k = 0; k < nq; k++) {
        const uint32_t capacity = t->queues[qs[k]].capacity;

        if (capacity == 0u || it->slots > capacity)
            return 0u;
        out[k].queue = qs[k];
        out[k].count = (int64_t)it->index + (int64_t)it->slots - (int64_t)capacity;
    }
    return nq;
}

bool pn_tt_hoist_tracker_met(const pn_tt_hoist_tracker *t, const pn_tt_hoist_need *need,
                             uint32_t count, size_t undo_from, size_t undo_to)
{
    uint32_t k;

    if (count == 0u)
        return false;
    for (k = 0; k < count; k++) {
        int64_t lb = t->state[need[k].queue].lb;
        size_t u;

        for (u = undo_from; u < undo_to && u < t->undo_count; u++)
            if (t->undo[u].queue == need[k].queue) {
                lb = t->undo[u].before; /* the first change that await made: its old value */
                break;
            }
        if (lb == HOIST_LB_NONE || lb < need[k].count)
            return false;
    }
    return true;
}

/* ---------------------------------------------------------------------------------------
 * The replay planner
 * --------------------------------------------------------------------------------------- */

void pn_tt_hoist_tracker_next_token(pn_tt_hoist_tracker *t)
{
    uint32_t q, a;

    for (q = 0; q < t->max_queues; q++) {
        hoist_queue_state *s = &t->state[q];
        const int64_t step = (int64_t)t->queues[q].step;

        if (s->lb != HOIST_LB_NONE)
            s->lb -= step;
        for (a = 0; a < s->param_count; a++)
            if (s->param_last[a] != HOIST_LB_NONE)
                s->param_last[a] -= step;
        s->lead.count = 0u;
        s->member.count = 0u;
    }
    t->undo_count = 0u;
}

/** Pass 1: the state at the end of the token, carried into the next (shifted by the step). */
static void hoist_carry(pn_tt_hoist_tracker *t, const pn_tt_hoist_item *items, size_t n)
{
    size_t i;

    for (i = 0; i < n && !t->failed; i++)
        (void)pn_tt_hoist_tracker_apply(t, &items[i]);
    pn_tt_hoist_tracker_next_token(t);
}

typedef struct {
    uint32_t item;
    bool hoisted;
    size_t undo_from, undo_to; /* an AWAIT's changes */
} hoist_slot;

size_t pn_tt_hoist_plan(const pn_tt_hoist_item *items, size_t n, const pn_tt_hoist_queue *queues,
                        uint32_t queue_count, const pn_tt_hoist_group *groups,
                        uint32_t group_count, uint32_t *order, pn_tt_hoist_stats *stats)
{
    pn_tt_hoist_tracker *t;
    hoist_slot *out;
    size_t len = 0u, segment_start = 0u, i, written = 0u;
    uint32_t q;
    bool failed;

    if (stats)
        memset(stats, 0, sizeof(*stats));
    for (i = 0; i < n; i++)
        if (items[i].kind != PN_TT_HOIST_SEGMENT && items[i].kind != PN_TT_HOIST_FIXED &&
            (items[i].queue >= queue_count ||
             (items[i].group >= 0 && (uint32_t)items[i].group >= group_count)))
            return 0u;
    t = pn_tt_hoist_tracker_create(queue_count);
    out = malloc((n ? n : 1u) * sizeof(*out));
    failed = !t || !out;
    for (q = 0; !failed && q < queue_count; q++)
        pn_tt_hoist_tracker_queue(t, q, queues[q].capacity, queues[q].step, queues[q].leader);
    if (!failed) {
        pn_tt_hoist_tracker_groups(t, groups, group_count);
        hoist_carry(t, items, n);
    }
    for (i = 0; i < n && !failed && !t->failed; i++) {
        const pn_tt_hoist_item *it = &items[i];
        pn_tt_hoist_need need[4];
        const uint32_t needs = pn_tt_hoist_tracker_needs(t, it, need);
        size_t at = len;

        if (it->kind == PN_TT_HOIST_WRITE || it->kind == PN_TT_HOIST_RING) {
            size_t j = len;

            while (j > segment_start && items[out[j - 1u].item].kind == PN_TT_HOIST_HEAD)
                j--;
            if (j > segment_start && items[out[j - 1u].item].kind == PN_TT_HOIST_AWAIT &&
                pn_tt_hoist_tracker_met(t, need, needs, out[j - 1u].undo_from,
                                        out[j - 1u].undo_to))
                at = j - 1u;
            if (stats) {
                if (at != len)
                    stats->moved++;
                else
                    stats->blocked++;
            }
        }
        memmove(&out[at + 1u], &out[at], (len - at) * sizeof(*out));
        out[at].item = (uint32_t)i;
        out[at].hoisted = at != len;
        out[at].undo_from = pn_tt_hoist_tracker_apply(t, it);
        out[at].undo_to = t->undo_count;
        len++;
        if (it->kind == PN_TT_HOIST_SEGMENT)
            segment_start = len;
    }
    failed = failed || t->failed;
    if (!failed) {
        for (i = 0; i < len; i++) {
            if (out[i].hoisted && (i == 0u || !out[i - 1u].hoisted)) {
                order[written++] = PN_TT_HOIST_FLUSH_MARK;
                if (stats)
                    stats->flushes++;
            }
            order[written++] = out[i].item;
        }
    }
    pn_tt_hoist_tracker_destroy(t);
    free(out);
    return failed ? 0u : written;
}
