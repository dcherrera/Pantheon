/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_precondition.c
 * @brief Off-card proof of S2 on-card completion's program rewrite (src/accel/tt/
 *        pn_tt_precondition.c) and of the plan check that guards it (pn_tt_precondition_verify.c).
 *
 * Three layers:
 *   1. CASES: small programs whose right plan is known — each rule the planner applies, and
 *      for each, a MUTATED plan (the realization that rule forced, deleted) the check must
 *      refuse. A check that passes a plan with a live hazard is worth nothing.
 *   2. A QWEN3.8 DECODE TOKEN, synthesized from the forward's issue sequence (fusion spec
 *      2.1: 48 GDN + 16 attention layers, the S1 group awaits, the matmul's 16 members and its
 *      4-core multicast groups, rotating parameter slots on every vector family), as the
 *      replay builder sees it (hoisted, carried, groups, per-token steps) and as the streamed
 *      verify sees it (no groups, no carry, data writes). No recorded token trace exists in the
 *      tree (the recorder prints a census, not its events), so this is the stand-in; its
 *      shape is the hand count of the spec's table.
 *   3. RED `precondition_early` (every entry one count short): the check must refuse it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_dispatch_hoist.h"
#include "../src/accel/tt/pn_tt_precondition.h"

static int g_failed;

typedef struct {
    pn_tt_pre_action *a;
    size_t n, cap;
} plan;

static void plan_record(void *user, const pn_tt_pre_action *action)
{
    plan *p = user;

    if (p->n == p->cap) {
        const size_t grow = p->cap ? p->cap * 2u : 256u;
        pn_tt_pre_action *more = realloc(p->a, grow * sizeof(*more));

        if (!more) {
            fprintf(stderr, "out of memory\n");
            exit(2);
        }
        p->a = more;
        p->cap = grow;
    }
    p->a[p->n++] = *action;
}

static pn_tt_pre_item mk(uint8_t kind, uint16_t queue, uint32_t index, uint32_t slots)
{
    pn_tt_pre_item it;

    memset(&it, 0, sizeof(it));
    it.kind = kind;
    it.group = -1;
    it.queue = queue;
    it.index = index;
    it.slots = slots;
    it.address = 0x179000u;
    return it;
}

typedef struct {
    pn_tt_pre_queue_info info[64];
    int32_t leader[64];
    uint32_t step[64];
    uint32_t count;
    const pn_tt_hoist_group *groups;
    uint32_t group_count;
} setup;

static void setup_init(setup *s, uint32_t count, uint32_t capacity)
{
    uint32_t q;

    memset(s, 0, sizeof(*s));
    s->count = count;
    for (q = 0; q < count; q++) {
        s->info[q].capacity = capacity;
        s->info[q].honours = true;
        s->leader[q] = -1;
    }
}

/** Plan `items` with `lookahead` (carry: replay's carried knowledge first). */
static void make_plan(const setup *s, const pn_tt_pre_item *items, size_t n, uint32_t lookahead,
                      bool red_early, bool carry, plan *out, pn_tt_pre_stats *stats)
{
    const pn_tt_pre_config config = { lookahead, red_early };
    pn_tt_pre_planner *p = pn_tt_precondition_create(s->count, &config, plan_record, out);
    uint32_t q;
    size_t i;

    out->n = 0u;
    if (!p) {
        printf("FAIL planner not created\n");
        g_failed++;
        return;
    }
    for (q = 0; q < s->count; q++)
        pn_tt_precondition_queue(p, q, s->info[q].capacity, s->step[q], s->leader[q],
                                 s->info[q].honours);
    pn_tt_precondition_groups(p, s->groups, s->group_count);
    if (carry)
        pn_tt_precondition_carry(p, items, n);
    for (i = 0; i < n; i++)
        if (pn_tt_precondition_push(p, (uint32_t)i, &items[i]) != 0) {
            printf("FAIL push refused item %zu\n", i);
            g_failed++;
            break;
        }
    if (stats)
        *stats = *pn_tt_precondition_stats(p);
    pn_tt_precondition_destroy(p);
}

static bool verify(const setup *s, const pn_tt_pre_item *items, size_t n, const plan *pl,
                   pn_tt_pre_report *report)
{
    return pn_tt_precondition_verify(items, n, pl->a, pl->n, s->info, s->count, s->groups,
                                     s->group_count, report);
}

static void expect(const char *name, bool ok, const char *detail)
{
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail ? ": " : "", detail ? detail : "");
    if (!ok)
        g_failed++;
}

/** The plan passes the check. */
static void expect_valid(const char *name, const setup *s, const pn_tt_pre_item *items, size_t n,
                         const plan *pl)
{
    pn_tt_pre_report r;
    const bool ok = verify(s, items, n, pl, &r);
    char line[256];

    (void)snprintf(line, sizeof(line), "%zu heads, %zu writes, %zu segments checked%s%s", r.heads,
                   r.writes, r.segments, ok ? "" : " — ", ok ? "" : r.first);
    expect(name, ok, line);
}

/** The plan must FAIL the check (a red). */
static void expect_refused(const char *name, const setup *s, const pn_tt_pre_item *items,
                           size_t n, const plan *pl)
{
    pn_tt_pre_report r;
    const bool ok = verify(s, items, n, pl, &r);
    char line[256];

    (void)snprintf(line, sizeof(line), "%zu failures, first: %s", r.failures, r.first);
    expect(name, !ok, line);
}

static size_t count_kind(const plan *pl, uint8_t kind)
{
    size_t i, c = 0u;

    for (i = 0; i < pl->n; i++)
        c += pl->a[i].kind == kind ? 1u : 0u;
    return c;
}

/** Index in the plan of the k-th action of `kind` realizing / naming input item `item`. */
static long find_action(const plan *pl, uint8_t kind, uint32_t item)
{
    size_t i;

    for (i = 0; i < pl->n; i++)
        if (pl->a[i].kind == kind && pl->a[i].item == item)
            return (long)i;
    return -1;
}

/** A copy of `pl` without action `at`. */
static void plan_without(const plan *pl, long at, plan *out)
{
    size_t i;

    out->n = 0u;
    for (i = 0; i < pl->n; i++)
        if ((long)i != at)
            plan_record(out, &pl->a[i]);
}

/* P -> C -> D, each awaited: C and D carry entries; each AWAIT is realized one window late. */
static void test_chain(void)
{
    setup s;
    pn_tt_pre_item it[10];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    long at;

    setup_init(&s, 3, 64u);
    it[0] = mk(PN_TT_PRE_RING, 0, 10, 1);
    it[1] = mk(PN_TT_PRE_HEAD, 0, 10, 1);
    it[2] = mk(PN_TT_PRE_AWAIT, 0, 11, 0);
    it[3] = mk(PN_TT_PRE_RING, 1, 20, 1);
    it[4] = mk(PN_TT_PRE_HEAD, 1, 20, 1);
    it[5] = mk(PN_TT_PRE_AWAIT, 1, 21, 0);
    it[6] = mk(PN_TT_PRE_RING, 2, 30, 1);
    it[7] = mk(PN_TT_PRE_HEAD, 2, 30, 1);
    it[8] = mk(PN_TT_PRE_AWAIT, 2, 31, 0);
    it[9] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, 10, 1u, false, false, &pl, &st);
    {
        /* I0 I1 I3 P I4 A(2) I6 P I7 A(5) A(8) I9 */
        const bool shape = pl.n == 12u && pl.a[3].kind == PN_TT_PRE_OUT_PRECOND &&
                           pl.a[3].queue == 1u && pl.a[3].producer == 0u &&
                           pl.a[3].count == 11u && pl.a[3].index == 20u &&
                           pl.a[5].kind == PN_TT_PRE_OUT_AWAIT && pl.a[5].item == 2u &&
                           pl.a[7].kind == PN_TT_PRE_OUT_PRECOND && pl.a[7].queue == 2u &&
                           pl.a[7].producer == 1u && pl.a[9].item == 5u && pl.a[10].item == 8u;
        char line[96];

        (void)snprintf(line, sizeof(line), "%zu actions, %zu entries, %zu real awaits of %zu",
                       pl.n, st.preconds, st.awaits_out, st.awaits_in);
        expect("chain: entries on C and D, each await one window late", shape, line);
    }
    expect_valid("chain: the check passes the plan", &s, it, 10, &pl);
    at = find_action(&pl, PN_TT_PRE_OUT_PRECOND, 7u);
    plan_without(&pl, at, &bad);
    expect_refused("RED chain without D's entry (D may start before C is done)", &s, it, 10, &bad);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 8u);
    plan_without(&pl, at, &bad);
    expect_refused("RED chain without the segment end's await", &s, it, 10, &bad);
    make_plan(&s, it, 10, 1u, true, false, &pl, NULL);
    expect_refused("RED precondition_early on the chain", &s, it, 10, &pl);
    free(pl.a);
    free(bad.a);
}

/* Two AWAITs in one run before a head: one entry names one producer, so the first is real. */
static void test_run(void)
{
    setup s;
    pn_tt_pre_item it[9];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;

    setup_init(&s, 3, 8u);
    it[0] = mk(PN_TT_PRE_RING, 0, 0, 1);
    it[1] = mk(PN_TT_PRE_HEAD, 0, 0, 1);
    it[2] = mk(PN_TT_PRE_RING, 1, 0, 1);
    it[3] = mk(PN_TT_PRE_HEAD, 1, 0, 1);
    it[4] = mk(PN_TT_PRE_AWAIT, 0, 1, 0);
    it[5] = mk(PN_TT_PRE_AWAIT, 1, 1, 0);
    it[6] = mk(PN_TT_PRE_RING, 2, 0, 1);
    it[7] = mk(PN_TT_PRE_HEAD, 2, 0, 1);
    it[8] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, 9, 1u, false, false, &pl, &st);
    expect("run of two: the first realized, the last the entry",
           st.by_run == 1u && st.preconds == 1u &&
               find_action(&pl, PN_TT_PRE_OUT_AWAIT, 4u) <
                   find_action(&pl, PN_TT_PRE_OUT_ITEM, 7u),
           NULL);
    expect_valid("run of two: valid", &s, it, 9, &pl);
    plan_without(&pl, find_action(&pl, PN_TT_PRE_OUT_AWAIT, 4u), &bad);
    expect_refused("RED run of two with the first await dropped", &s, it, 9, &bad);
    free(pl.a);
    free(bad.a);
}

/* Lookahead 2. x (q1) and w (q4) issued together; await x; y (q2); await w; z (q3). z's entry
 * names w, which was issued BEFORE "await x": w's start never waited for x, so z's entry does
 * not imply x is done — the planner must realize "await x" before z. When w is issued after
 * "await x" instead, the entry implies it and it may stay pending. */
static void test_lookahead_chain(void)
{
    setup s;
    pn_tt_pre_item it[12];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    long at;

    setup_init(&s, 5, 8u);
    it[0] = mk(PN_TT_PRE_RING, 1, 0, 1);
    it[1] = mk(PN_TT_PRE_HEAD, 1, 0, 1);
    it[2] = mk(PN_TT_PRE_RING, 4, 0, 1);
    it[3] = mk(PN_TT_PRE_HEAD, 4, 0, 1);
    it[4] = mk(PN_TT_PRE_AWAIT, 1, 1, 0);
    it[5] = mk(PN_TT_PRE_RING, 2, 0, 1);
    it[6] = mk(PN_TT_PRE_HEAD, 2, 0, 1);
    it[7] = mk(PN_TT_PRE_AWAIT, 4, 1, 0);
    it[8] = mk(PN_TT_PRE_RING, 3, 0, 1);
    it[9] = mk(PN_TT_PRE_HEAD, 3, 0, 1);
    it[10] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, 11, 2u, false, false, &pl, &st);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 4u);
    expect("lookahead 2: an await the entry does not imply is realized before z",
           st.by_chain == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, 9u), NULL);
    expect_valid("lookahead 2, unimplied: valid", &s, it, 11, &pl);
    plan_without(&pl, at, &bad);
    expect_refused("RED lookahead 2 trusting an unimplied await (z may start before x)", &s, it,
                   11, &bad);
    /* w issued after "await x": the chain holds, nothing extra is realized */
    it[2] = mk(PN_TT_PRE_AWAIT, 1, 1, 0);
    it[3] = mk(PN_TT_PRE_RING, 4, 0, 1);
    it[4] = mk(PN_TT_PRE_HEAD, 4, 0, 1);
    it[5] = mk(PN_TT_PRE_RING, 2, 0, 1);
    it[6] = mk(PN_TT_PRE_HEAD, 2, 0, 1);
    make_plan(&s, it, 11, 2u, false, false, &pl, &st);
    expect("lookahead 2: an implied await stays pending", st.by_chain == 0u, NULL);
    expect_valid("lookahead 2, implied: valid", &s, it, 11, &pl);
    free(pl.a);
    free(bad.a);
}

/* A ring of 2: C publishes descriptors 0 and 1, the program awaits both, and C's descriptor 2
 * reuses descriptor 0's slot. Only "await C 2" proves descriptor 0 done, and the plan would
 * convert it: the ring write must realize it first (the window then keeps a real await). */
static void test_ring_reuse(void)
{
    setup s;
    pn_tt_pre_item it[16];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0;
    long at;

    setup_init(&s, 2, 8u);
    s.info[1].capacity = 2u;
    it[n++] = mk(PN_TT_PRE_RING, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 1, 0);
    it[n++] = mk(PN_TT_PRE_RING, 1, 0, 2);
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 0, 2);
    it[n++] = mk(PN_TT_PRE_AWAIT, 1, 2, 0); /* 5 */
    it[n++] = mk(PN_TT_PRE_RING, 1, 2, 1); /* 6: slot of descriptor 0 */
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 2, 1);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 5u);
    expect("ring reuse: the slot's last user is realized before the slot is written",
           st.by_write == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, 6u), NULL);
    expect_valid("ring reuse: valid", &s, it, n, &pl);
    plan_without(&pl, at, &bad);
    expect_refused("RED ring slot rewritten under a live descriptor", &s, it, n, &bad);
    free(pl.a);
    free(bad.a);
}

/* The vector families' single rotating parameter block: the same queue in two windows in a
 * row rewrites a block its previous descriptor may still read. */
static void test_param_reuse(void)
{
    setup s;
    pn_tt_pre_item it[16];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0;
    long at;

    setup_init(&s, 2, 64u);
    it[n++] = mk(PN_TT_PRE_WRITE, 0, 100, 0);
    it[n++] = mk(PN_TT_PRE_RING, 0, 100, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 100, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 101, 0); /* 3 */
    it[n++] = mk(PN_TT_PRE_WRITE, 0, 101, 0); /* 4: rewrites descriptor 100's block */
    it[n++] = mk(PN_TT_PRE_RING, 0, 101, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 101, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 102, 0);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 3u);
    expect("parameter block reuse back to back keeps its await",
           st.by_write == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, 4u), NULL);
    expect_valid("parameter block reuse: valid", &s, it, n, &pl);
    plan_without(&pl, at, &bad);
    expect_refused("RED parameter block rewritten under its reader", &s, it, n, &bad);
    free(pl.a);
    free(bad.a);
}

/* An untracked write (a data write, a row copy) and a segment end realize everything first;
 * a queue whose reader does not honour the sidecar gets a real await, never an entry. */
static void test_barrier_segment_refusal(void)
{
    setup s;
    pn_tt_pre_item it[12];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0;

    setup_init(&s, 3, 8u);
    it[n++] = mk(PN_TT_PRE_RING, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 1, 0);
    it[n++] = mk(PN_TT_PRE_RING, 1, 0, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 0, 1);
    it[n++] = mk(PN_TT_PRE_BARRIER, 0, 0, 0); /* 5 */
    it[n++] = mk(PN_TT_PRE_AWAIT, 1, 1, 0);
    it[n++] = mk(PN_TT_PRE_RING, 2, 0, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 2, 0, 1);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    expect("barrier: the converted await is realized before the data write",
           st.by_barrier == 1u && find_action(&pl, PN_TT_PRE_OUT_AWAIT, 2u) <
                                      find_action(&pl, PN_TT_PRE_OUT_ITEM, 5u),
           NULL);
    expect_valid("barrier: valid", &s, it, n, &pl);
    plan_without(&pl, find_action(&pl, PN_TT_PRE_OUT_AWAIT, 2u), &bad);
    expect_refused("RED data write before the await it followed", &s, it, n, &bad);
    s.info[2].honours = false;
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    expect("refusal: a queue that does not honour entries gets a real await",
           st.by_refusal == 1u && count_kind(&pl, PN_TT_PRE_OUT_PRECOND) == 1u, NULL);
    expect_valid("refusal: valid", &s, it, n, &pl);
    s.info[2].honours = true;
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    s.info[2].honours = false;
    expect_refused("RED an entry on a queue that ignores it", &s, it, n, &pl);
    free(pl.a);
    free(bad.a);
}

/* A multicast group's head gets ONE entry (the members share the index). A queue awaiting its
 * own previous descriptor keeps a REAL await before its next head: once that head is out, the
 * queue can complete past the count and the puller's AWAIT (count with its token) could never
 * be met (card D=8, 2026-11-17). RED: the plan that realizes it after the head is refused. */
static void test_group_and_self(void)
{
    static const pn_tt_hoist_group groups[1] = { { { 1, 2, 3, 0 }, 3u } };
    setup s;
    pn_tt_pre_item it[12];
    plan pl = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0;
    long at;

    setup_init(&s, 4, 8u);
    s.groups = groups;
    s.group_count = 1u;
    it[n++] = mk(PN_TT_PRE_RING, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 0, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 1, 0);
    it[n] = mk(PN_TT_PRE_RING, 1, 0, 1); it[n++].group = 0;
    it[n] = mk(PN_TT_PRE_HEAD, 1, 0, 1); it[n++].group = 0;
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 1, 0);
    it[n++] = mk(PN_TT_PRE_RING, 0, 1, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 1, 1);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    at = find_action(&pl, PN_TT_PRE_OUT_PRECOND, 4u);
    expect("group head: one multicast entry", at >= 0 && pl.a[at].group == 0 &&
                                                   count_kind(&pl, PN_TT_PRE_OUT_PRECOND) == 1u,
           NULL);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 5u);
    expect("self: an await on the head's own queue is realized before that head",
           st.by_publish == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, 7u) &&
               find_action(&pl, PN_TT_PRE_OUT_PRECOND, 7u) < 0,
           NULL);
    expect_valid("group and self: valid", &s, it, n, &pl);
    {
        /* the old plan: the self await realized only at the segment end, after the head */
        plan late = { 0 };
        size_t i;

        for (i = 0; i < pl.n; i++)
            if ((long)i != at)
                plan_record(&late, &pl.a[i]);
        late.n--; /* drop the segment end, put the late await before it */
        plan_record(&late, &pl.a[at]);
        plan_record(&late, &pl.a[pl.n - 1u]);
        expect_refused("RED a self await realized after its queue published past it", &s, it,
                       n, &late);
        free(late.a);
    }
    pl.a[find_action(&pl, PN_TT_PRE_OUT_PRECOND, 4u)].index = 1u;
    expect_refused("RED an entry for another index than the head's first", &s, it, n, &pl);
    free(pl.a);
}

/* A HELD await: the program publishes C's descriptor 11 before awaiting C at 11 (its count
 * is below C's published 12 — the shape the card's plain-decode replay showed, 128 a token on
 * queue 0, once L2's PN_TT_MATMUL_PULL_DEFER went to 0). The puller takes count 11 only with
 * descriptor 10's token beside it, so the await must stay exactly where it is: moved a window
 * later, descriptor 11 may finish first and the AWAIT can never be met. RED: the plan that
 * moves it past the next item is refused. */
static void test_held(void)
{
    setup s;
    pn_tt_pre_item it[12];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0, i;
    long at, after;

    setup_init(&s, 3, 64u);
    it[n++] = mk(PN_TT_PRE_RING, 0, 10, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 10, 1);
    it[n++] = mk(PN_TT_PRE_RING, 0, 11, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 11, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 11, 0); /* 4: held (published 12) */
    it[n++] = mk(PN_TT_PRE_RING, 1, 20, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 20, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 1, 21, 0);
    it[n++] = mk(PN_TT_PRE_RING, 2, 30, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 2, 30, 1);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, n, 1u, false, false, &pl, &st);
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, 4u);
    expect("held: the await stays where it is, before the next item",
           st.by_held == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, 5u) &&
               find_action(&pl, PN_TT_PRE_OUT_PRECOND, 6u) < 0,
           NULL);
    expect_valid("held: valid", &s, it, n, &pl);
    after = find_action(&pl, PN_TT_PRE_OUT_ITEM, 6u);
    bad.n = 0u;
    for (i = 0; i < pl.n; i++) {
        if ((long)i != at)
            plan_record(&bad, &pl.a[i]);
        if ((long)i == after && at >= 0)
            plan_record(&bad, &pl.a[at]);
    }
    expect_refused("RED a held await moved past the next item", &s, it, n, &bad);
    free(pl.a);
    free(bad.a);
}

/* A held await under a HOISTED order: the hoist moves the next op's parameter block and ring
 * slot (another queue, proven free) above the held AWAIT, as td_hoist_program does on the
 * card. The held await may stay after them — they publish nothing and touch no descriptor of
 * its queue, and the knob-0 program has the same order — but nothing that FOLLOWS it in the
 * hoisted order may pass it. The plan passes the check; the plan with the held await after
 * the next head is refused. */
static void test_held_hoisted(void)
{
    const pn_tt_hoist_queue q[3] = { { 64u, 0u, -1 }, { 64u, 0u, -1 }, { 64u, 0u, -1 } };
    static const uint8_t to_hoist[] = { PN_TT_HOIST_FIXED, PN_TT_HOIST_FIXED, PN_TT_HOIST_WRITE,
                                        PN_TT_HOIST_RING, PN_TT_HOIST_HEAD, PN_TT_HOIST_AWAIT,
                                        PN_TT_HOIST_SEGMENT };
    setup s;
    pn_tt_pre_item it[16], hoisted[32];
    pn_tt_hoist_item h[16];
    uint32_t order[40];
    plan pl = { 0 }, bad = { 0 };
    pn_tt_pre_stats st;
    size_t n = 0, got, k, i;
    long at, head;
    bool moved = false;

    setup_init(&s, 3, 64u);
    it[n++] = mk(PN_TT_PRE_WRITE, 1, 19, 0); /* q1's last user: 19, done by "await q1 20" */
    it[n++] = mk(PN_TT_PRE_RING, 1, 19, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 19, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 1, 20, 0);
    it[n++] = mk(PN_TT_PRE_RING, 0, 10, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 10, 1);
    it[n++] = mk(PN_TT_PRE_RING, 0, 11, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 0, 11, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 0, 11, 0); /* held */
    it[n++] = mk(PN_TT_PRE_WRITE, 1, 20, 0); /* hoisted above the held await */
    it[n++] = mk(PN_TT_PRE_RING, 1, 20, 1);
    it[n++] = mk(PN_TT_PRE_HEAD, 1, 20, 1);
    it[n++] = mk(PN_TT_PRE_AWAIT, 1, 21, 0);
    it[n++] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    for (k = 0; k < n; k++) {
        memset(&h[k], 0, sizeof(h[k]));
        h[k].kind = to_hoist[it[k].kind];
        h[k].group = -1;
        h[k].queue = it[k].queue;
        h[k].index = it[k].kind == PN_TT_PRE_HEAD ? it[k].index + 1u : it[k].index;
        h[k].slots = it[k].slots;
        h[k].address = it[k].address;
    }
    got = pn_tt_hoist_plan(h, n, q, 3, NULL, 0, order, NULL);
    for (k = 0; k < got; k++) {
        hoisted[k] = order[k] == PN_TT_HOIST_FLUSH_MARK ? mk(PN_TT_PRE_NEUTRAL, 0, 0, 0)
                                                        : it[order[k]];
    }
    for (k = 0; k + 1u < got; k++)
        if (order[k] == 10u && order[k + 1u] == 8u)
            moved = true; /* q1's slot went above the held await */
    make_plan(&s, hoisted, got, 1u, false, false, &pl, &st);
    expect("held + hoist: the hoist moved a write above the held await", moved, NULL);
    for (k = 0; k < got && order[k] != 8u; k++)
        ;
    at = find_action(&pl, PN_TT_PRE_OUT_AWAIT, (uint32_t)k);
    expect("held + hoist: the held await is realized at its hoisted position",
           st.by_held == 1u && at >= 0 && at < find_action(&pl, PN_TT_PRE_OUT_ITEM, (uint32_t)k + 1u),
           NULL);
    expect_valid("held + hoist: valid", &s, hoisted, got, &pl);
    head = -1;
    for (i = 0; i < pl.n; i++)
        if (pl.a[i].kind == PN_TT_PRE_OUT_ITEM && hoisted[pl.a[i].item].kind == PN_TT_PRE_HEAD &&
            pl.a[i].item > k)
            head = head < 0 ? (long)i : head;
    bad.n = 0u;
    for (i = 0; i < pl.n; i++) {
        if ((long)i != at)
            plan_record(&bad, &pl.a[i]);
        if ((long)i == head && at >= 0)
            plan_record(&bad, &pl.a[at]);
    }
    expect_refused("RED held + hoist: the held await after the next head", &s, hoisted, got,
                   &bad);
    free(pl.a);
    free(bad.a);
}

/* ---- a Qwen3.8-27B decode token ---- */

enum {
    Q_MM = 0,      /* matmul members 0..15 */
    Q_ELT = 16,    /* eltwise 4 */
    Q_RED = 20,    /* reduce + reduce-member */
    Q_WIDE = 22,   /* render widen 2 */
    Q_NARROW = 24, /* render narrow 2 */
    Q_GDN = 26,    /* gdn CONV/DELTA 2 */
    Q_GATE = 28,   /* gdngate */
    Q_HNL2 = 29,   /* head-norm L2 2 */
    Q_HNRMS = 31,  /* head-norm RMS 2 */
    Q_ATTN = 33,   /* attention lanes 4 */
    Q_KV = 37,     /* KV append split 2 */
    Q_ROPE = 39,
    Q_TOPK = 40,
    Q_COUNT = 41
};

static const pn_tt_hoist_group g_mm_groups[4] = {
    { { 1, 2, 3, 4 }, 4u }, { { 5, 6, 7, 8 }, 4u }, { { 9, 10, 11, 12 }, 4u },
    { { 13, 14, 15, 0 }, 3u }
};

typedef struct {
    pn_tt_pre_item *items;
    size_t n, cap;
    uint32_t next[Q_COUNT];
    uint32_t sequence;
    bool replay; /* multicast groups + per-token steps */
    bool held;   /* every norm publishes one more descriptor on member 0 before its await */
    const uint32_t *step;
} q38_gen;

static void gen_push(q38_gen *g, pn_tt_pre_item it)
{
    if (g->n == g->cap) {
        g->cap = g->cap ? g->cap * 2u : 4096u;
        g->items = realloc(g->items, g->cap * sizeof(*g->items));
        if (!g->items)
            exit(2);
    }
    if (g->step && it.kind != PN_TT_PRE_NEUTRAL && it.kind != PN_TT_PRE_SEGMENT &&
        it.kind != PN_TT_PRE_BARRIER)
        it.step = g->step[it.queue];
    g->items[g->n++] = it;
}

/** One op on members [first, first + members): parameter block (vector families), ring slots,
 *  head, then ONE await on member 0 (S1). */
static void gen_op(q38_gen *g, uint32_t first, uint32_t members, uint32_t slots, bool param)
{
    uint32_t m;
    pn_tt_pre_item it;

    for (m = 0; m < members; m++) {
        const uint32_t q = first + m;

        if (param) {
            it = mk(PN_TT_PRE_WRITE, (uint16_t)q, g->next[q], 0);
            gen_push(g, it);
        }
        it = mk(PN_TT_PRE_RING, (uint16_t)q, g->next[q], slots);
        if (members > 1u) {
            if (m == 0u)
                it.pull_sequence = g->sequence;
            else
                it.group_sequence = g->sequence;
        }
        gen_push(g, it);
        gen_push(g, mk(PN_TT_PRE_HEAD, (uint16_t)q, g->next[q], slots));
        g->next[q] += slots;
    }
    g->sequence++;
    if (g->held && first == Q_RED) {
        /* the await names the norm's count; member 0's next descriptor is already out */
        const uint32_t awaited = g->next[first];

        gen_push(g, mk(PN_TT_PRE_WRITE, Q_RED, g->next[Q_RED], 0));
        gen_push(g, mk(PN_TT_PRE_RING, Q_RED, g->next[Q_RED], 1));
        gen_push(g, mk(PN_TT_PRE_HEAD, Q_RED, g->next[Q_RED], 1));
        g->next[Q_RED] += 1u;
        gen_push(g, mk(PN_TT_PRE_AWAIT, (uint16_t)first, awaited, 0));
        return;
    }
    gen_push(g, mk(PN_TT_PRE_AWAIT, (uint16_t)first, g->next[first], 0));
}

/** A matmul flush on all 16 members; replay sends members 1..15 as four multicast groups. */
static void gen_matmul(q38_gen *g, uint32_t slots)
{
    uint32_t k, m;
    pn_tt_pre_item it;

    it = mk(PN_TT_PRE_RING, Q_MM, g->next[Q_MM], slots);
    it.pull_sequence = g->sequence;
    gen_push(g, it);
    gen_push(g, mk(PN_TT_PRE_HEAD, Q_MM, g->next[Q_MM], slots));
    for (k = 0; k < 4u; k++) {
        if (g->replay) {
            it = mk(PN_TT_PRE_RING, g_mm_groups[k].queues[0], g->next[g_mm_groups[k].queues[0]],
                    slots);
            it.group = (int16_t)k;
            it.group_sequence = g->sequence;
            gen_push(g, it);
            it = mk(PN_TT_PRE_HEAD, g_mm_groups[k].queues[0], g->next[g_mm_groups[k].queues[0]],
                    slots);
            it.group = (int16_t)k;
            gen_push(g, it);
        } else {
            for (m = 0; m < g_mm_groups[k].count; m++) {
                const uint16_t q = g_mm_groups[k].queues[m];

                it = mk(PN_TT_PRE_RING, q, g->next[q], slots);
                it.group_sequence = g->sequence;
                gen_push(g, it);
                gen_push(g, mk(PN_TT_PRE_HEAD, q, g->next[q], slots));
            }
        }
    }
    for (m = 0; m < 16u; m++)
        g->next[Q_MM + m] += slots;
    g->sequence++;
    gen_push(g, mk(PN_TT_PRE_AWAIT, Q_MM, g->next[Q_MM], 0));
}

/** A host data access: a segment end in a replay, a captured data write in the stream. */
static void gen_boundary(q38_gen *g)
{
    gen_push(g, mk(g->replay ? PN_TT_PRE_SEGMENT : PN_TT_PRE_BARRIER, 0, 0, 0));
}

static void gen_ffn_tail(q38_gen *g)
{
    gen_matmul(g, 1);                      /* out-proj */
    gen_op(g, Q_ELT, 4, 1, true);          /* residual add */
    gen_op(g, Q_RED, 2, 1, true);          /* post-norm */
    gen_matmul(g, 2);                      /* gate + up */
    gen_op(g, Q_ELT, 4, 1, true);          /* silu_mul */
    gen_matmul(g, 1);                      /* down */
    gen_op(g, Q_ELT, 4, 1, true);          /* residual add */
}

/** One decode token: 64 layers (every 4th attention), the head, top-k. */
static void gen_token(q38_gen *g)
{
    uint32_t layer;

    for (layer = 0; layer < 64u; layer++) {
        gen_op(g, Q_RED, 2, 1, true);      /* pre-norm */
        gen_matmul(g, 4);                  /* in-proj */
        if (layer % 4u != 3u) {
            gen_op(g, Q_WIDE, 2, 1, true);
            gen_op(g, Q_GDN, 2, 1, true);  /* CONV */
            gen_op(g, Q_HNL2, 2, 1, true);
            gen_op(g, Q_GATE, 1, 1, true);
            gen_op(g, Q_GDN, 2, 1, true);  /* DELTA */
            gen_op(g, Q_HNRMS, 2, 1, true);
            gen_op(g, Q_NARROW, 2, 1, true);
            gen_op(g, Q_ELT, 4, 1, true);  /* silu_mul */
        } else {
            gen_op(g, Q_HNL2, 2, 1, true); /* head-norm q */
            gen_op(g, Q_HNL2, 2, 1, true); /* head-norm k: same queues back to back */
            gen_boundary(g);               /* the position words */
            gen_op(g, Q_ROPE, 1, 1, true);
            gen_op(g, Q_ROPE, 1, 1, true);
            gen_op(g, Q_KV, 2, 1, true);
            gen_op(g, Q_ATTN, 4, 1, true);
            gen_op(g, Q_ELT, 4, 1, true);  /* sigmoid */
            gen_op(g, Q_ELT, 4, 1, true);  /* mul */
        }
        gen_ffn_tail(g);
        if (layer % 2u == 1u)
            gen_boundary(g);
    }
    gen_op(g, Q_RED, 2, 1, true);          /* final norm */
    gen_matmul(g, 1);                      /* lm_head */
    gen_op(g, Q_TOPK, 1, 1, true);
    gen_push(g, mk(PN_TT_PRE_SEGMENT, 0, 0, 0));
}

static void q38_setup(setup *s, bool replay)
{
    uint32_t m;

    setup_init(s, Q_COUNT, 64u);
    for (m = 0; m < 16u; m++)
        s->info[Q_MM + m].capacity = 8u; /* the matmul group's ring */
    if (replay) {
        s->groups = g_mm_groups;
        s->group_count = 4u;
    }
    for (m = 1; m < 16u; m++)
        s->leader[Q_MM + m] = Q_MM;
    for (m = 1; m < 4u; m++)
        s->leader[Q_ELT + m] = Q_ELT;
    s->leader[Q_RED + 1] = Q_RED;
}

/** Replay: the hoisted order (FLUSH marks as NEUTRAL items), as td_hoist_program emits it. */
static size_t q38_hoisted(const setup *s, const pn_tt_pre_item *items, size_t n,
                          pn_tt_pre_item *out)
{
    pn_tt_hoist_item *h = calloc(n, sizeof(*h));
    pn_tt_hoist_queue queues[Q_COUNT];
    uint32_t *order = calloc(2u * n + 2u, sizeof(*order));
    size_t i, got, k = 0;
    uint32_t q;

    if (!h || !order)
        exit(2);
    for (q = 0; q < Q_COUNT; q++) {
        queues[q].capacity = s->info[q].capacity;
        queues[q].step = s->step[q];
        queues[q].leader = s->leader[q];
    }
    for (i = 0; i < n; i++) {
        static const uint8_t kinds[] = { PN_TT_HOIST_FIXED, PN_TT_HOIST_FIXED, PN_TT_HOIST_WRITE,
                                         PN_TT_HOIST_RING, PN_TT_HOIST_HEAD, PN_TT_HOIST_AWAIT,
                                         PN_TT_HOIST_SEGMENT };

        h[i].kind = kinds[items[i].kind];
        h[i].group = items[i].group;
        h[i].queue = items[i].queue;
        h[i].index = items[i].index;
        h[i].slots = items[i].slots;
        h[i].address = items[i].address;
        h[i].pull_sequence = items[i].pull_sequence;
        h[i].group_sequence = items[i].group_sequence;
    }
    got = pn_tt_hoist_plan(h, n, queues, Q_COUNT, s->groups, s->group_count, order, NULL);
    for (i = 0; i < got; i++)
        out[k++] = order[i] == PN_TT_HOIST_FLUSH_MARK ? mk(PN_TT_PRE_NEUTRAL, 0, 0, 0)
                                                      : items[order[i]];
    free(h);
    free(order);
    return k;
}

static void q38_run(const char *name, const setup *s, const pn_tt_pre_item *items, size_t n,
                    bool carry)
{
    static const uint32_t lookaheads[4] = { 1u, 2u, 4u, 8u };
    plan pl = { 0 };
    pn_tt_pre_stats st;
    unsigned l;
    char line[320];

    for (l = 0; l < 4u; l++) {
        make_plan(s, items, n, lookaheads[l], false, carry, &pl, &st);
        (void)snprintf(line, sizeof(line),
                       "%s, lookahead %u: %zu awaits, %zu converted to %zu entries (each "
                       "realized a window late), %zu kept in place; realized by lag %zu, write "
                       "%zu, barrier %zu, segment %zu, run %zu, chain %zu, refusal %zu, publish "
                       "%zu, held %zu; merged %zu",
                       name, lookaheads[l], st.awaits_in, st.windows, st.preconds,
                       st.awaits_in - st.merged - st.windows, st.by_lag, st.by_write,
                       st.by_barrier, st.by_segment, st.by_run, st.by_chain, st.by_refusal,
                       st.by_publish, st.by_held, st.merged);
        expect("q38 token converts", st.preconds > 0u && st.windows > 0u, line);
        /* every input await is realized somewhere (lag or otherwise) or merged into one */
        expect("q38 token: every await realized or merged",
               st.awaits_out + st.merged == st.awaits_in, NULL);
        expect_valid("q38 token: the check passes the plan", s, items, n, &pl);
    }
    make_plan(s, items, n, 1u, true, carry, &pl, &st);
    expect_refused("RED precondition_early on the q38 token", s, items, n, &pl);
    free(pl.a);
}

static void test_q38_token(bool held)
{
    setup s;
    q38_gen g;
    uint32_t q, start[Q_COUNT], step[Q_COUNT];
    pn_tt_pre_item *hoisted;
    size_t n;

    /* replay: learn the per-token steps from a first token, plan the second */
    memset(&g, 0, sizeof(g));
    g.replay = true;
    g.held = held;
    printf("     q38 token, %s\n", held ? "HELD awaits (2 a layer: each norm's member 0 publishes "
                                           "again before its await)" : "every await at its count");
    for (q = 0; q < Q_COUNT; q++)
        g.next[q] = start[q] = 1000u;
    gen_token(&g);
    for (q = 0; q < Q_COUNT; q++)
        step[q] = g.next[q] - start[q];
    g.n = 0u;
    g.step = step;
    gen_token(&g);
    q38_setup(&s, true);
    memcpy(s.step, step, sizeof(step));
    hoisted = calloc(2u * g.n + 2u, sizeof(*hoisted));
    if (!hoisted)
        exit(2);
    n = q38_hoisted(&s, g.items, g.n, hoisted);
    printf("     q38 replay token: %zu items (%zu after the hoist's flushes)\n", g.n, n);
    q38_run("replay", &s, hoisted, n, true);
    free(hoisted);
    /* stream: no groups, no steps, no carry, data writes instead of segment ends */
    g.n = 0u;
    g.step = NULL;
    g.replay = false;
    gen_token(&g);
    q38_setup(&s, false);
    printf("     q38 stream cycle: %zu items\n", g.n);
    q38_run("stream", &s, g.items, g.n, false);
    free(g.items);
}

/** `--replay FILE`: a `PN_TT_ONCARD_DUMP` from the card, re-planned and re-checked here. Prints
 *  both checks' reports, whether the re-plan reproduces the card's plan, and the input items
 *  around the first failure. @return 0 when the re-plan passes its check. */
static int replay_dump(const char *path)
{
    FILE *f = fopen(path, "r");
    pn_tt_pre_dump d;
    pn_tt_pre_report card, again;
    pn_tt_pre_queue_info *info;
    plan pl = { 0 };
    pn_tt_pre_planner *p;
    pn_tt_pre_stats st;
    bool card_ok, again_ok, same;
    uint32_t q;
    size_t i;

    if (!f || pn_tt_precondition_dump_read(f, &d) != 0) {
        printf("replay: %s is not a PN_TT_ONCARD_DUMP\n", path);
        if (f)
            fclose(f);
        return 2;
    }
    fclose(f);
    info = calloc(d.queue_count + 1u, sizeof(*info));
    p = info ? pn_tt_precondition_create(d.queue_count, &d.config, plan_record, &pl) : NULL;
    if (!p) {
        pn_tt_precondition_dump_free(&d);
        free(info);
        return 2;
    }
    for (q = 0; q < d.queue_count; q++) {
        info[q].capacity = d.capacity[q];
        info[q].honours = d.honours[q];
        pn_tt_precondition_queue(p, q, d.capacity[q], d.step[q], d.leader[q], d.honours[q]);
    }
    pn_tt_precondition_groups(p, d.groups, d.group_count);
    pn_tt_precondition_carry(p, d.items, d.n_items);
    for (i = 0; i < d.n_items; i++)
        (void)pn_tt_precondition_push(p, (uint32_t)i, &d.items[i]);
    st = *pn_tt_precondition_stats(p);
    pn_tt_precondition_destroy(p);
    card_ok = pn_tt_precondition_verify(d.items, d.n_items, d.actions, d.n_actions, info,
                                        d.queue_count, d.groups, d.group_count, &card);
    again_ok = pn_tt_precondition_verify(d.items, d.n_items, pl.a, pl.n, info, d.queue_count,
                                         d.groups, d.group_count, &again);
    same = pl.n == d.n_actions && memcmp(pl.a, d.actions, pl.n * sizeof(*pl.a)) == 0;
    printf("replay %s: %zu items, %u queues, %u groups, lookahead %u\n", path, d.n_items,
           d.queue_count, d.group_count, d.config.lookahead);
    printf("  the card's plan: %zu actions, check %s (%zu failures%s%s)\n", d.n_actions,
           card_ok ? "PASSED" : "FAILED", card.failures, card_ok ? "" : "; first: ",
           card_ok ? "" : card.first);
    printf("  re-planned here: %zu actions (%s the card's), check %s (%zu failures%s%s); held %zu, "
           "publish %zu, %zu entries\n", pl.n, same ? "identical to" : "DIFFERENT from",
           again_ok ? "PASSED" : "FAILED", again.failures, again_ok ? "" : "; first: ",
           again_ok ? "" : again.first, st.by_held, st.by_publish, st.preconds);
    if (!again_ok || !card_ok) {
        const pn_tt_pre_report *r = again_ok ? &card : &again;

        /* the failure names (queue, count) of an AWAIT or (queue, descriptor) of a head */
        for (i = 0; i < d.n_items; i++) {
            const pn_tt_pre_item *it = &d.items[i];

            if (it->queue == r->first_queue &&
                ((it->kind == PN_TT_PRE_AWAIT && it->index == r->first_value) ||
                 (it->kind == PN_TT_PRE_HEAD && it->index <= r->first_value &&
                  r->first_value < it->index + it->slots)))
                break;
        }
        printf("  input items around the first one naming queue %u, %u (item %zu):\n",
               r->first_queue, r->first_value, i);
        for (size_t j = i > 12u ? i - 12u : 0u; j < d.n_items && j <= i + 6u; j++)
            printf("    %6zu kind %u group %d queue %u index %u slots %u\n", j, d.items[j].kind,
                   d.items[j].group, d.items[j].queue, d.items[j].index, d.items[j].slots);
    }
    free(pl.a);
    free(info);
    pn_tt_precondition_dump_free(&d);
    return again_ok ? 0 : 1;
}

/** The dump format round-trips a plan, and the replay re-plans it identically. */
static void test_dump_round_trip(void)
{
    setup s;
    pn_tt_pre_item it[10];
    plan pl = { 0 };
    pn_tt_pre_dump d, back;
    char path[] = "build/test_tt_precondition_dump.txt";
    FILE *f;
    bool ok;

    setup_init(&s, 3, 64u);
    it[0] = mk(PN_TT_PRE_RING, 0, 10, 1);
    it[1] = mk(PN_TT_PRE_HEAD, 0, 10, 1);
    it[2] = mk(PN_TT_PRE_AWAIT, 0, 11, 0);
    it[3] = mk(PN_TT_PRE_RING, 1, 20, 1);
    it[4] = mk(PN_TT_PRE_HEAD, 1, 20, 1);
    it[5] = mk(PN_TT_PRE_SEGMENT, 0, 0, 0);
    make_plan(&s, it, 6, 1u, false, false, &pl, NULL);
    memset(&d, 0, sizeof(d));
    d.config.lookahead = 1u;
    d.queue_count = 3u;
    d.capacity = (uint32_t[3]){ 64u, 64u, 64u };
    d.step = (uint32_t[3]){ 0u, 0u, 0u };
    d.leader = (int32_t[3]){ -1, -1, -1 };
    d.honours = (bool[3]){ true, true, true };
    d.items = it;
    d.n_items = 6u;
    d.actions = pl.a;
    d.n_actions = pl.n;
    f = fopen(path, "w");
    ok = f && pn_tt_precondition_dump_write(f, &d) == 0;
    if (f)
        fclose(f);
    f = ok ? fopen(path, "r") : NULL;
    ok = f && pn_tt_precondition_dump_read(f, &back) == 0;
    if (f)
        fclose(f);
    ok = ok && back.n_items == 6u && back.n_actions == pl.n &&
         memcmp(back.items, it, 6u * sizeof(*it)) == 0 &&
         memcmp(back.actions, pl.a, pl.n * sizeof(*pl.a)) == 0;
    if (ok)
        pn_tt_precondition_dump_free(&back);
    expect("dump: a plan round-trips through PN_TT_ONCARD_DUMP's text", ok, NULL);
    expect("dump: --replay re-plans it identically and passes", ok && replay_dump(path) == 0,
           NULL);
    free(pl.a);
}

int main(int argc, char **argv)
{
    if (argc > 2 && strcmp(argv[1], "--replay") == 0)
        return replay_dump(argv[2]);
    test_chain();
    test_run();
    test_lookahead_chain();
    test_ring_reuse();
    test_param_reuse();
    test_barrier_segment_refusal();
    test_group_and_self();
    test_held();
    test_held_hoisted();
    test_dump_round_trip();
    test_q38_token(false);
    test_q38_token(true);
    printf("%s\n", g_failed ? "test_tt_precondition: FAILED" : "test_tt_precondition: ALL PASS");
    return g_failed ? 1 : 0;
}
