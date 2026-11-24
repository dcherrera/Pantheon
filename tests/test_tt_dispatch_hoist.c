/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_dispatch_hoist.c
 * @brief Off-card proof of the L10 hoisting planner (src/accel/tt/pn_tt_dispatch_hoist.c):
 *        every move it makes is one the overwrite rule allows, and every move the rule forbids
 *        is refused — each red case below is a hoist that WOULD corrupt a live descriptor.
 */
#include <stdio.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_dispatch_hoist.h"
#include "../src/accel/tt/pn_tt_precondition.h"

static int g_failed;

#define F PN_TT_HOIST_FLUSH_MARK

static pn_tt_hoist_item item(uint8_t kind, uint16_t queue, uint32_t index)
{
    pn_tt_hoist_item it;

    memset(&it, 0, sizeof(it));
    it.kind = kind;
    it.queue = queue;
    it.group = -1;
    it.index = index;
    it.slots = kind == PN_TT_HOIST_RING ? 1u : 0u;
    it.address = 0x179000u;
    return it;
}

static void expect(const char *name, const pn_tt_hoist_item *items, size_t n,
                   const pn_tt_hoist_queue *queues, uint32_t nq, const pn_tt_hoist_group *groups,
                   uint32_t ng, const uint32_t *want, size_t want_n)
{
    uint32_t order[64];
    pn_tt_hoist_stats stats;
    const size_t got = pn_tt_hoist_plan(items, n, queues, nq, groups, ng, order, &stats);
    bool ok = got == want_n && memcmp(order, want, want_n * sizeof(*want)) == 0;
    size_t i;

    printf("%s %s (moved %zu, blocked %zu, flushes %zu):", ok ? "PASS" : "FAIL", name,
           stats.moved, stats.blocked, stats.flushes);
    for (i = 0; i < got; i++)
        if (order[i] == F)
            printf(" F");
        else
            printf(" %u", order[i]);
    printf("\n");
    if (!ok)
        g_failed++;
}

/* P feeds C; both awaited every token. C's block and slot go out under P's await. */
static void test_chain(void)
{
    const pn_tt_hoist_queue q[2] = { { 8u, 1u, -1 }, { 8u, 1u, -1 } };
    pn_tt_hoist_item it[8];
    static const uint32_t want[] = { 0, 1, F, 3, 4, 2, 5, 6, 7 };

    it[0] = item(PN_TT_HOIST_RING, 0, 10);
    it[1] = item(PN_TT_HOIST_HEAD, 0, 11);
    it[2] = item(PN_TT_HOIST_AWAIT, 0, 11);
    it[3] = item(PN_TT_HOIST_WRITE, 1, 5);
    it[4] = item(PN_TT_HOIST_RING, 1, 5);
    it[5] = item(PN_TT_HOIST_HEAD, 1, 6);
    it[6] = item(PN_TT_HOIST_AWAIT, 1, 6);
    it[7] = item(PN_TT_HOIST_SEGMENT, 0, 0);
    expect("chain: C's block + slot under P's await", it, 8, q, 2, NULL, 0, want,
           sizeof(want) / sizeof(*want));
}

/* RED: C runs twice back to back with no await on C between — its second parameter block
 * would overwrite the first while the first may not have started. Must stay. */
static void test_param_reuse_red(void)
{
    const pn_tt_hoist_queue q[2] = { { 8u, 1u, -1 }, { 8u, 2u, -1 } };
    pn_tt_hoist_item it[10];
    static const uint32_t want[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    it[0] = item(PN_TT_HOIST_WRITE, 1, 5);
    it[1] = item(PN_TT_HOIST_RING, 1, 5);
    it[2] = item(PN_TT_HOIST_HEAD, 1, 6);
    it[3] = item(PN_TT_HOIST_RING, 0, 10);
    it[4] = item(PN_TT_HOIST_HEAD, 0, 11);
    it[5] = item(PN_TT_HOIST_AWAIT, 0, 11);
    it[6] = item(PN_TT_HOIST_WRITE, 1, 6); /* same address as descriptor 5: not done yet */
    it[7] = item(PN_TT_HOIST_RING, 1, 6);
    it[8] = item(PN_TT_HOIST_HEAD, 1, 7);
    it[9] = item(PN_TT_HOIST_AWAIT, 1, 7);
    expect("RED parameter block still in use: stays", it, 10, q, 2, NULL, 0, want,
           sizeof(want) / sizeof(*want));
}

/* Capacity 2: slot i holds i-2. Proven done (C awaited at 5, carried bound 4): every write
 * moves. RED: without the await on C, slot 6 would overwrite descriptor 4, not proven done —
 * that ring write stays while the others (proven) still move. */
static void test_ring_capacity_red(void)
{
    const pn_tt_hoist_queue q[2] = { { 8u, 1u, -1 }, { 2u, 3u, -1 } };
    pn_tt_hoist_item it[12];
    static const uint32_t want[] = { 0, 1, F, 3, 5, 2, 4, 6, F, 8, 9, 7, 10, 11 };
    static const uint32_t want_red[] = { 0, 1, F, 3, 5, 2, 4, 6, F, 8, 7, 9, 10, 11 };

    it[0] = item(PN_TT_HOIST_RING, 1, 4);
    it[1] = item(PN_TT_HOIST_HEAD, 1, 5);
    it[2] = item(PN_TT_HOIST_AWAIT, 1, 5);
    it[3] = item(PN_TT_HOIST_RING, 1, 5);
    it[4] = item(PN_TT_HOIST_HEAD, 1, 6);
    it[5] = item(PN_TT_HOIST_RING, 0, 10);
    it[6] = item(PN_TT_HOIST_HEAD, 0, 11);
    it[7] = item(PN_TT_HOIST_AWAIT, 0, 11);
    it[8] = item(PN_TT_HOIST_WRITE, 1, 6);
    it[9] = item(PN_TT_HOIST_RING, 1, 6);
    it[10] = item(PN_TT_HOIST_HEAD, 1, 7);
    it[11] = item(PN_TT_HOIST_AWAIT, 1, 7);
    expect("capacity 2, i-2 done: moves", it, 12, q, 2, NULL, 0, want,
           sizeof(want) / sizeof(*want));
    it[2] = item(PN_TT_HOIST_AWAIT, 0, 9);
    expect("RED capacity 2, i-2 live: that slot stays", it, 12, q, 2, NULL, 0, want_red,
           sizeof(want_red) / sizeof(*want_red));
}

/* A never-awaited member M (leader L, ring capacity 1): its next slot overwrites the
 * descriptor L's pull just proved done (member sequence 70 <= pull sequence 70) -> it moves.
 * RED: the pull waits only for 69 -> M's descriptor is not proven done -> it stays (and L's
 * slot, behind it, with it); with no leader at all M never moves. */
static void test_member(void)
{
    pn_tt_hoist_queue q[3] = { { 8u, 1u, -1 }, { 4u, 2u, -1 }, { 1u, 2u, 0 } };
    pn_tt_hoist_item it[16];
    size_t n = 0;
    static const uint32_t want[] = { 0, 1, 2, 3, F, 5, 4, 6, F, 8, 10, 7, 9, 11, 12, 13 };
    static const uint32_t want_red[] = { 0, 1, 2, 3, F, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13 };

    it[n] = item(PN_TT_HOIST_RING, 2, 8); it[n].group_sequence = 70; n++;      /* 0 M */
    it[n] = item(PN_TT_HOIST_HEAD, 2, 9); n++;                                  /* 1 */
    it[n] = item(PN_TT_HOIST_RING, 0, 20); it[n].pull_sequence = 70; n++;      /* 2 L */
    it[n] = item(PN_TT_HOIST_HEAD, 0, 21); n++;                                 /* 3 */
    it[n] = item(PN_TT_HOIST_AWAIT, 0, 21); n++;                                /* 4 */
    it[n] = item(PN_TT_HOIST_RING, 1, 3); n++;                                  /* 5 X */
    it[n] = item(PN_TT_HOIST_HEAD, 1, 4); n++;                                  /* 6 */
    it[n] = item(PN_TT_HOIST_AWAIT, 1, 4); n++;                                 /* 7 */
    it[n] = item(PN_TT_HOIST_RING, 2, 9); it[n].group_sequence = 71; n++;      /* 8 M */
    it[n] = item(PN_TT_HOIST_HEAD, 2, 10); n++;                                 /* 9 */
    it[n] = item(PN_TT_HOIST_RING, 0, 21); it[n].pull_sequence = 71; n++;      /* 10 L */
    it[n] = item(PN_TT_HOIST_HEAD, 0, 22); n++;                                 /* 11 */
    it[n] = item(PN_TT_HOIST_AWAIT, 0, 22); n++;                                /* 12 */
    it[n] = item(PN_TT_HOIST_SEGMENT, 0, 0); n++;                               /* 13 */
    expect("member under its leader's pull: moves", it, n, q, 3, NULL, 0, want,
           sizeof(want) / sizeof(*want));
    it[2].pull_sequence = 69;
    expect("RED pull short of the member's sequence: stays", it, n, q, 3, NULL, 0, want_red,
           sizeof(want_red) / sizeof(*want_red));
    it[2].pull_sequence = 70;
    q[2].leader = -1;
    expect("RED member with no leader: stays", it, n, q, 3, NULL, 0, want_red,
           sizeof(want_red) / sizeof(*want_red));
}

/* Nothing crosses a segment end, and a malformed queue index refuses the plan. */
static void test_segment_and_malformed(void)
{
    const pn_tt_hoist_queue q[2] = { { 8u, 1u, -1 }, { 8u, 1u, -1 } };
    pn_tt_hoist_item it[6];
    static const uint32_t want[] = { 0, 1, 2, 3, 4, 5 };
    uint32_t order[16];

    it[0] = item(PN_TT_HOIST_RING, 0, 10);
    it[1] = item(PN_TT_HOIST_HEAD, 0, 11);
    it[2] = item(PN_TT_HOIST_AWAIT, 0, 11);
    it[3] = item(PN_TT_HOIST_SEGMENT, 0, 0);
    it[4] = item(PN_TT_HOIST_RING, 1, 5);
    it[5] = item(PN_TT_HOIST_HEAD, 1, 6);
    expect("segment end is a wall", it, 6, q, 2, NULL, 0, want, 6);
    it[4].queue = 9;
    if (pn_tt_hoist_plan(it, 6, q, 2, NULL, 0, order, NULL) != 0u) {
        printf("FAIL malformed queue index accepted\n");
        g_failed++;
    } else {
        printf("PASS malformed queue index refused\n");
    }
}

/* The replay's token carry (pn_tt_hoist_tracker_next_token): a bound and a parameter user
 * learned in token k are shifted back by the queue's step, so they name the right descriptors
 * of token k+1. RED: without the shift, descriptor 11's slot (capacity 4, its user 7) would be
 * called free on token k's bound 10. */
static void test_next_token(void)
{
    pn_tt_hoist_tracker *t = pn_tt_hoist_tracker_create(1u);
    pn_tt_hoist_item it = item(PN_TT_HOIST_AWAIT, 0, 10);
    pn_tt_hoist_need need[4];
    uint32_t n;
    bool free10, free11, block;

    pn_tt_hoist_tracker_queue(t, 0, 4u, 3u, -1);
    (void)pn_tt_hoist_tracker_apply(t, &it);
    it = item(PN_TT_HOIST_WRITE, 0, 9);
    (void)pn_tt_hoist_tracker_apply(t, &it);
    pn_tt_hoist_tracker_next_token(t);
    it = item(PN_TT_HOIST_RING, 0, 10);
    n = pn_tt_hoist_tracker_needs(t, &it, need);
    free10 = pn_tt_hoist_tracker_met(t, need, n, 0u, 0u);
    it = item(PN_TT_HOIST_RING, 0, 11);
    n = pn_tt_hoist_tracker_needs(t, &it, need);
    free11 = pn_tt_hoist_tracker_met(t, need, n, 0u, 0u);
    it = item(PN_TT_HOIST_WRITE, 0, 12);
    n = pn_tt_hoist_tracker_needs(t, &it, need);
    block = n == 1u && need[0].count == 7 && pn_tt_hoist_tracker_met(t, need, n, 0u, 0u);
    printf("%s next token: bound 10 -> 7 (slot 10 free %d, slot 11 free %d), block user 9 -> 6 "
           "(%d)\n", free10 && !free11 && block ? "PASS" : "FAIL", free10, free11, block);
    if (!(free10 && !free11 && block))
        g_failed++;
    pn_tt_hoist_tracker_destroy(t);
}

typedef struct {
    pn_tt_pre_action a[64];
    size_t n;
} s2_plan;

static void s2_record(void *user, const pn_tt_pre_action *action)
{
    s2_plan *p = user;

    if (p->n < 64u)
        p->a[p->n++] = *action;
}

/* S2 runs on the HOISTED order (pn_tt_precondition.h): the chain's hoisted block and slot
 * stand ahead of P's await, which S2 then turns into C's entry. The plan check passes it, and
 * refuses it once C's entry is dropped. */
static void test_hoist_then_s2(void)
{
    const pn_tt_hoist_queue q[2] = { { 8u, 1u, -1 }, { 8u, 1u, -1 } };
    const pn_tt_pre_queue_info info[2] = { { 8u, true }, { 8u, true } };
    const pn_tt_pre_config config = { 1u, false };
    static const uint8_t kinds[] = { PN_TT_PRE_NEUTRAL, PN_TT_PRE_WRITE, PN_TT_PRE_RING,
                                     PN_TT_PRE_HEAD, PN_TT_PRE_AWAIT, PN_TT_PRE_SEGMENT };
    pn_tt_hoist_item it[8];
    pn_tt_pre_item pre[16];
    uint32_t order[32];
    s2_plan plan = { .n = 0 };
    pn_tt_pre_planner *p = pn_tt_precondition_create(2u, &config, s2_record, &plan);
    size_t got, k, entry = 0u;
    bool valid, refused;

    it[0] = item(PN_TT_HOIST_RING, 0, 10);
    it[1] = item(PN_TT_HOIST_HEAD, 0, 11);
    it[2] = item(PN_TT_HOIST_AWAIT, 0, 11);
    it[3] = item(PN_TT_HOIST_WRITE, 1, 5);
    it[4] = item(PN_TT_HOIST_RING, 1, 5);
    it[5] = item(PN_TT_HOIST_HEAD, 1, 6);
    it[6] = item(PN_TT_HOIST_AWAIT, 1, 6);
    it[7] = item(PN_TT_HOIST_SEGMENT, 0, 0);
    got = pn_tt_hoist_plan(it, 8, q, 2, NULL, 0, order, NULL);
    for (k = 0; k < got; k++) {
        const pn_tt_hoist_item *h = order[k] == F ? NULL : &it[order[k]];

        memset(&pre[k], 0, sizeof(pre[k]));
        pre[k].group = -1;
        if (!h)
            continue;
        pre[k].kind = kinds[h->kind];
        pre[k].queue = h->queue;
        pre[k].index = h->index;
        pre[k].slots = h->slots;
        pre[k].address = h->address;
        if (h->kind == PN_TT_HOIST_HEAD) { /* a head names its first descriptor */
            pre[k].index = h->index - 1u;
            pre[k].slots = 1u;
        }
    }
    for (k = 0; p && k < got; k++)
        (void)pn_tt_precondition_push(p, (uint32_t)k, &pre[k]);
    pn_tt_precondition_destroy(p);
    valid = pn_tt_precondition_verify(pre, got, plan.a, plan.n, info, 2, NULL, 0, NULL);
    for (k = 0; k < plan.n; k++)
        if (plan.a[k].kind == PN_TT_PRE_OUT_PRECOND)
            entry = k;
    memmove(&plan.a[entry], &plan.a[entry + 1u], (plan.n - entry - 1u) * sizeof(*plan.a));
    plan.n--;
    refused = !pn_tt_precondition_verify(pre, got, plan.a, plan.n, info, 2, NULL, 0, NULL);
    printf("%s hoisted chain under S2: plan valid %d, without C's entry refused %d\n",
           valid && refused && entry > 0u ? "PASS" : "FAIL", valid, refused);
    if (!(valid && refused && entry > 0u))
        g_failed++;
}

int main(void)
{
    test_chain();
    test_param_reuse_red();
    test_ring_capacity_red();
    test_member();
    test_segment_and_malformed();
    test_next_token();
    test_hoist_then_s2();
    printf("%s\n", g_failed ? "test_tt_dispatch_hoist: FAILED" : "test_tt_dispatch_hoist: ALL PASS");
    return g_failed ? 1 : 0;
}
