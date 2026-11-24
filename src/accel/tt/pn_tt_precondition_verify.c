/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_precondition_verify.c
 * @brief The S2 plan check (pn_tt_precondition_verify, see pn_tt_precondition.h): a model of
 *        the card that knows nothing of the planner's rules.
 *
 * THE MODEL. Every published descriptor d gets a START VECTOR S(d): for each queue, the
 * completed count d's start is guaranteed to come after. Completing d then guarantees
 * C(d) = S(d) + "d's queue reached index(d) + 1". The puller's knowledge K is the join of C
 * over the descriptors its real AWAITs waited for. At a head, each published descriptor gets
 *   S = K  joined with  C(producer descriptor) of the sidecar entry written for it (if it
 *        matches its index and its queue honours entries)  joined with  S(previous descriptor
 *        on its queue) (the reader is sequential).
 * The checks, against the input program:
 *   - HEAD: S of every descriptor it publishes covers every AWAIT that preceded the head in
 *     the input (the running maximum per queue). By induction that is today's whole
 *     guarantee, transitive facts included.
 *   - RING / WRITE: what the slot or block overwrites is proven complete by K wherever the
 *     input program (simulated the same way, every AWAIT real) proved it, up to the need.
 *   - SEGMENT / BARRIER: K covers every AWAIT before it.
 *   - every real AWAIT and every entry names a descriptor already published (else a wait
 *     that can never end); an entry sits between its descriptor's previous head and its own.
 * Pull-leader implications are not modelled, on either side, so a member's knowledge is never
 * credited — the comparison stays like for like.
 *
 * @status S2 2.3.
 * @issues None
 * @todo None
 */
#include "pn_tt_precondition.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VER_NONE INT64_MIN
#define VER_PARAM_ADDRESSES 32u

typedef struct {
    int64_t *s;          /* start vectors, `cap` of them, queue_count words each */
    uint32_t base, count, cap;
    bool based;
    uint32_t published;
    bool entry;          /* a sidecar entry is written and its head not yet seen */
    uint32_t entry_index, entry_count;
    uint16_t entry_producer;
    uint32_t param_address[VER_PARAM_ADDRESSES];
    int64_t param_last[VER_PARAM_ADDRESSES];
    uint32_t param_count;
} ver_queue;

typedef struct {
    const pn_tt_pre_item *in;
    size_t n_in;
    const pn_tt_pre_queue_info *queues;
    uint32_t nq;
    const pn_tt_hoist_group *groups;
    uint32_t ng;
    ver_queue *q;
    int64_t *k;          /* the puller's knowledge */
    int64_t *tmp;
    int64_t *rmax;       /* the input's AWAITs so far, per queue */
    size_t in_cursor;    /* input items folded into rmax */
    int64_t *write_known; /* per write and member: the input program's K at that write */
    size_t writes;
    uint32_t *await_published; /* per input item: an AWAIT's queue's published count where the
                                  input has it, UINT32_MAX none yet */
    pn_tt_pre_report *report;
    bool oom;
} ver_ctx;

static void ver_fail(ver_ctx *v, const char *what, uint32_t a, uint32_t b, int64_t c, int64_t d)
{
    if (v->report->failures++ == 0u) {
        (void)snprintf(v->report->first, sizeof(v->report->first), "%s (%u, %u: %lld vs %lld)",
                       what, a, b, (long long)c, (long long)d);
        v->report->first_queue = a;
        v->report->first_value = b;
    }
}

static void ver_join(int64_t *dst, const int64_t *src, uint32_t n)
{
    uint32_t i;

    for (i = 0; i < n; i++)
        if (src[i] > dst[i])
            dst[i] = src[i];
}

static int64_t *ver_start(ver_ctx *v, uint32_t q, uint32_t index)
{
    ver_queue *s = &v->q[q];

    if (!s->based || (int32_t)(index - s->base) < 0 || index - s->base >= s->count)
        return NULL;
    return &s->s[(size_t)(index - s->base) * v->nq];
}

/** C(q, index) into v->tmp: its start vector plus its own completion. */
static void ver_completion(ver_ctx *v, uint32_t q, uint32_t index)
{
    const int64_t *s = ver_start(v, q, index);
    uint32_t i;

    for (i = 0; i < v->nq; i++)
        v->tmp[i] = s ? s[i] : VER_NONE;
    if ((int64_t)index + 1 > v->tmp[q])
        v->tmp[q] = (int64_t)index + 1;
}

static uint32_t ver_item_queues(const ver_ctx *v, int16_t group, uint16_t queue, uint32_t out[4])
{
    if (group >= 0 && (uint32_t)group < v->ng) {
        const pn_tt_hoist_group *g = &v->groups[group];
        uint32_t k, n = g->count < 4u ? g->count : 4u;

        for (k = 0; k < n; k++)
            out[k] = g->queues[k];
        return n;
    }
    out[0] = queue;
    return 1u;
}

/** Room for start vectors of descriptors [base, end) of queue q. */
static bool ver_reserve(ver_ctx *v, uint32_t q, uint32_t first, uint32_t end)
{
    ver_queue *s = &v->q[q];

    if (!s->based) {
        s->based = true;
        s->base = first;
        s->published = first;
    }
    if ((int32_t)(first - s->base) < 0)
        return false;
    while (end - s->base > s->cap) {
        const uint32_t grow = s->cap ? s->cap * 2u : 64u;
        int64_t *more = realloc(s->s, (size_t)grow * v->nq * sizeof(*more));

        if (!more) {
            v->oom = true;
            return false;
        }
        s->s = more;
        s->cap = grow;
    }
    while (s->count < end - s->base) {
        uint32_t i;

        for (i = 0; i < v->nq; i++)
            s->s[(size_t)s->count * v->nq + i] = VER_NONE;
        s->count++;
    }
    return true;
}

static void ver_head(ver_ctx *v, const pn_tt_pre_item *it, bool check)
{
    uint32_t qs[4], nq = ver_item_queues(v, it->group, it->queue, qs), k, i, j;

    v->report->heads += check ? 1u : 0u;
    for (k = 0; k < nq; k++) {
        ver_queue *s = &v->q[qs[k]];
        int64_t *start, *prev;

        if (it->index == UINT32_MAX) {
            /* Unplaceable (a stream head whose first index is unknown): its start is at
             * least K, so K alone must cover the input's AWAITs; the queue is unknown again. */
            for (i = 0; check && i < v->nq; i++)
                if (v->k[i] < v->rmax[i]) {
                    ver_fail(v, "an unplaced head before an AWAIT it followed is realized",
                             qs[k], i, v->k[i], v->rmax[i]);
                    break;
                }
            s->based = false;
            s->count = 0u;
            s->entry = false;
            continue;
        }
        if (!ver_reserve(v, qs[k], it->index, it->index + it->slots)) {
            ver_fail(v, "a head the model cannot place", qs[k], it->index, 0, 0);
            continue;
        }
        start = ver_start(v, qs[k], it->index);
        memcpy(start, v->k, v->nq * sizeof(*start));
        if (s->entry && s->entry_index == it->index && v->queues[qs[k]].honours) {
            ver_completion(v, s->entry_producer, s->entry_count - 1u);
            ver_join(start, v->tmp, v->nq);
        } else if (s->entry) {
            ver_fail(v, "an entry for another descriptor than the head's first", qs[k],
                     it->index, s->entry_index, 0);
        }
        s->entry = false;
        prev = it->index > 0u ? ver_start(v, qs[k], it->index - 1u) : NULL;
        if (prev)
            ver_join(start, prev, v->nq);
        for (j = 1; j < it->slots; j++)
            memcpy(ver_start(v, qs[k], it->index + j), start, v->nq * sizeof(*start));
        if (s->published == it->index || (int32_t)(it->index + it->slots - s->published) > 0)
            s->published = it->index + it->slots;
        for (i = 0; check && i < v->nq; i++)
            if (start[i] < v->rmax[i]) {
                ver_fail(v, "a descriptor may start before an AWAIT its head followed (queue, "
                            "descriptor: guaranteed vs awaited)", qs[k], it->index, start[i],
                         v->rmax[i]);
                break;
            }
    }
}

/** A ring slot or parameter block: its need per queue; record (input pass) or compare. */
static void ver_write(ver_ctx *v, const pn_tt_pre_item *it, bool check)
{
    uint32_t qs[4], nq = ver_item_queues(v, it->group, it->queue, qs), k;

    for (k = 0; k < nq && v->writes < v->n_in * 4u; k++) {
        ver_queue *s = &v->q[qs[k]];
        int64_t need = VER_NONE;

        if (it->kind == PN_TT_PRE_RING && v->queues[qs[k]].capacity != 0u) {
            need = (int64_t)it->index + (int64_t)it->slots - (int64_t)v->queues[qs[k]].capacity;
        } else if (it->kind == PN_TT_PRE_WRITE) {
            uint32_t a;

            for (a = 0; a < s->param_count && s->param_address[a] != it->address; a++)
                ;
            if (a == s->param_count && a < VER_PARAM_ADDRESSES) {
                s->param_address[a] = it->address;
                s->param_last[a] = VER_NONE;
                s->param_count++;
            }
            if (a < s->param_count) {
                need = s->param_last[a] == VER_NONE ? VER_NONE : s->param_last[a] + 1;
                s->param_last[a] = (int64_t)it->index;
            }
        }
        if (!check) {
            v->write_known[v->writes++] = v->k[qs[k]];
            continue;
        }
        v->report->writes++;
        {
            const int64_t today = v->write_known[v->writes++];
            const int64_t want = need == VER_NONE || need <= 0 ? VER_NONE
                                                                : (need < today ? need : today);

            if (v->k[qs[k]] < want)
                ver_fail(v, "a slot or block is overwritten before its user is proven done "
                            "(queue, index: known vs needed)", qs[k], it->index, v->k[qs[k]],
                         want);
        }
    }
}

/** Was descriptor count - 1 of q published by now? A queue with no head yet in the program
 *  was published before it (a replay token awaits what the previous one issued). */
static bool ver_published(const ver_ctx *v, uint32_t q, uint32_t count)
{
    return !v->q[q].based || (int32_t)(count - v->q[q].published) <= 0;
}

/** The puller accepts an AWAIT's count only with the token of descriptor count - 1 beside it
 *  (resident_puller.c kPlrErrAwaitToken), so a plan's real AWAIT for input item `item` must go
 *  out before a later descriptor of its queue can complete:
 *   - the usual AWAIT (the input has it at its queue's published count) while its queue has
 *     published nothing past `count`;
 *   - a HELD one (the input already has its queue published past `count`: a descriptor out
 *     before its await, finishing only after the puller looks) exactly where the input has it,
 *     before any later item — moved later, the held descriptor may finish first. `next` is
 *     the plan's next input item. */
static void ver_exact(ver_ctx *v, uint32_t q, uint32_t count, uint32_t item, size_t next)
{
    const uint32_t at_input = item < v->n_in ? v->await_published[item] : UINT32_MAX;
    const uint32_t now = v->q[q].based ? v->q[q].published : count;

    if (at_input != UINT32_MAX && at_input != count) {
        if (now != at_input)
            ver_fail(v, "a HELD AWAIT (its queue published past it in the input) is moved past "
                        "a publish (queue, count: published now vs in the input)", q, count,
                     now, at_input);
        else if (next > (size_t)item + 1u)
            ver_fail(v, "a HELD AWAIT (its queue published past it in the input) is moved past "
                        "a later item (queue, count: its input item vs the plan's next item)",
                     q, count, item, (int64_t)next);
    } else if (now != count) {
        ver_fail(v, "an AWAIT is emitted after its queue published past it (the puller "
                    "refuses the token)", q, count, now, 0);
    }
}

static void ver_await(ver_ctx *v, uint32_t q, uint32_t count)
{
    if (count == 0u)
        return;
    if (!ver_published(v, q, count))
        ver_fail(v, "an AWAIT names a descriptor not yet published", q, count,
                 v->q[q].published, 0);
    ver_completion(v, q, count - 1u);
    ver_join(v->k, v->tmp, v->nq);
}

static void ver_precond(ver_ctx *v, const pn_tt_pre_action *a)
{
    uint32_t qs[4], nq = ver_item_queues(v, a->group, a->queue, qs), k;

    if (a->producer >= v->nq || a->count == 0u || !ver_published(v, a->producer, a->count))
        ver_fail(v, "an entry names a descriptor not yet published", a->producer, a->count, 0, 0);
    for (k = 0; k < nq; k++) {
        if (!v->queues[qs[k]].honours)
            ver_fail(v, "an entry on a queue whose reader does not honour it", qs[k], a->index,
                     0, 0);
        v->q[qs[k]].entry = true;
        v->q[qs[k]].entry_index = a->index;
        v->q[qs[k]].entry_count = a->count;
        v->q[qs[k]].entry_producer = a->producer;
    }
}

/** Fold the input's AWAITs before item `upto` into the running maximum. */
static void ver_fold_input(ver_ctx *v, size_t upto)
{
    for (; v->in_cursor < upto && v->in_cursor < v->n_in; v->in_cursor++) {
        const pn_tt_pre_item *it = &v->in[v->in_cursor];

        if (it->kind == PN_TT_PRE_AWAIT && it->queue < v->nq && (int64_t)it->index > v->rmax[it->queue])
            v->rmax[it->queue] = (int64_t)it->index;
    }
}

static void ver_covers_all(ver_ctx *v, const char *what, uint32_t at)
{
    uint32_t i;

    for (i = 0; i < v->nq; i++)
        if (v->k[i] < v->rmax[i]) {
            ver_fail(v, what, i, at, v->k[i], v->rmax[i]);
            return;
        }
}

/** One input item, run as the model sees it. `check`: the plan's pass (else the input's). */
static void ver_item(ver_ctx *v, size_t index, bool check)
{
    const pn_tt_pre_item *it = &v->in[index];

    if (check)
        ver_fold_input(v, index);
    switch (it->kind) {
    case PN_TT_PRE_HEAD:
        ver_head(v, it, check);
        break;
    case PN_TT_PRE_RING:
    case PN_TT_PRE_WRITE:
        ver_write(v, it, check);
        break;
    case PN_TT_PRE_AWAIT:
        if (!check)
            v->await_published[index] = v->q[it->queue].based ? v->q[it->queue].published
                                                               : UINT32_MAX;
        ver_await(v, it->queue, it->index);
        break;
    case PN_TT_PRE_SEGMENT:
    case PN_TT_PRE_BARRIER:
        if (check) {
            if (it->kind == PN_TT_PRE_SEGMENT)
                v->report->segments++;
            else
                v->report->barriers++;
            ver_covers_all(v, it->kind == PN_TT_PRE_SEGMENT
                                  ? "a segment ends before an AWAIT in it is realized"
                                  : "an untracked write goes out before an AWAIT before it",
                           (uint32_t)index);
        }
        break;
    default:
        break;
    }
}

static void ver_reset(ver_ctx *v)
{
    uint32_t i;

    for (i = 0; i < v->nq; i++) {
        free(v->q[i].s);
        memset(&v->q[i], 0, sizeof(v->q[i]));
        v->k[i] = VER_NONE;
        v->rmax[i] = VER_NONE;
    }
    v->writes = 0u;
    v->in_cursor = 0u;
}

static bool ver_alloc(ver_ctx *v)
{
    v->q = calloc(v->nq, sizeof(*v->q));
    v->k = malloc(v->nq * sizeof(*v->k));
    v->tmp = malloc(v->nq * sizeof(*v->tmp));
    v->rmax = malloc(v->nq * sizeof(*v->rmax));
    v->write_known = malloc((v->n_in * 4u + 1u) * sizeof(*v->write_known));
    v->await_published = malloc((v->n_in + 1u) * sizeof(*v->await_published));
    return v->q && v->k && v->tmp && v->rmax && v->write_known && v->await_published;
}

static void ver_free(ver_ctx *v)
{
    uint32_t i;

    for (i = 0; v->q && i < v->nq; i++)
        free(v->q[i].s);
    free(v->q);
    free(v->k);
    free(v->tmp);
    free(v->rmax);
    free(v->write_known);
    free(v->await_published);
}

/** The plan's pass: its actions in order; items must come out in input order. */
static void ver_plan(ver_ctx *v, const pn_tt_pre_action *out, size_t n_out)
{
    size_t i, next = 0u;

    for (i = 0; i < n_out && !v->oom; i++) {
        const pn_tt_pre_action *a = &out[i];

        if (a->kind == PN_TT_PRE_OUT_ITEM) {
            if (a->item >= v->n_in || a->item < next) {
                ver_fail(v, "an item out of order or out of range", a->item, (uint32_t)next, 0, 0);
                return;
            }
            next = a->item + 1u;
            ver_item(v, a->item, true);
        } else if (a->kind == PN_TT_PRE_OUT_AWAIT && a->queue < v->nq) {
            if (a->count != 0u)
                ver_exact(v, a->queue, a->count, a->item, next);
            ver_await(v, a->queue, a->count);
        } else if (a->kind == PN_TT_PRE_OUT_PRECOND) {
            ver_precond(v, a);
        } else {
            ver_fail(v, "an action the model does not know", a->kind, a->queue, 0, 0);
        }
    }
    for (; next < v->n_in; next++)
        if (v->in[next].kind != PN_TT_PRE_AWAIT)
            ver_fail(v, "an input item the plan dropped", (uint32_t)next, v->in[next].kind, 0, 0);
}

bool pn_tt_precondition_verify(const pn_tt_pre_item *in, size_t n_in, const pn_tt_pre_action *out,
                               size_t n_out, const pn_tt_pre_queue_info *queues,
                               uint32_t queue_count, const pn_tt_hoist_group *groups,
                               uint32_t group_count, pn_tt_pre_report *report)
{
    pn_tt_pre_report local;
    ver_ctx v;
    size_t i;

    memset(&v, 0, sizeof(v));
    v.report = report ? report : &local;
    memset(v.report, 0, sizeof(*v.report));
    if (!in || !out || !queues || queue_count == 0u) {
        ver_fail(&v, "nothing to check", 0, 0, 0, 0);
        return false;
    }
    for (i = 0; i < n_in; i++) {
        const pn_tt_pre_item *it = &in[i];
        const bool queued = it->kind == PN_TT_PRE_HEAD || it->kind == PN_TT_PRE_RING ||
                            it->kind == PN_TT_PRE_WRITE || it->kind == PN_TT_PRE_AWAIT;

        if ((queued && it->queue >= queue_count) ||
            (it->group >= 0 && ((uint32_t)it->group >= group_count || !groups))) {
            ver_fail(&v, "an input item names a queue or group out of range", (uint32_t)i,
                     it->queue, 0, 0);
            return false;
        }
    }
    v.in = in;
    v.n_in = n_in;
    v.queues = queues;
    v.nq = queue_count;
    v.groups = groups;
    v.ng = group_count;
    if (!ver_alloc(&v)) {
        ver_free(&v);
        ver_fail(&v, "out of memory", 0, 0, 0, 0);
        return false;
    }
    ver_reset(&v);
    for (i = 0; i < n_in && !v.oom; i++) /* the input program, every AWAIT real */
        ver_item(&v, i, false);
    ver_reset(&v);
    if (!v.oom)
        ver_plan(&v, out, n_out);
    if (v.oom)
        ver_fail(&v, "out of memory", 0, 0, 0, 0);
    ver_free(&v);
    return v.report->failures == 0u;
}
