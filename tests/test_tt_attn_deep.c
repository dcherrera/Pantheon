/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_attn_deep.c — deep-queued prefill attention's SCHEDULE (pn_tt_attn_deep.h), with NO
 * CARD and NO USE_TT.
 *
 * A unit the schedule drops is a band whose head output is whatever the strip held, and a unit
 * queued twice is wasted work that still reads right — neither shows in a correlation. Checked:
 *   - for bands 1..64, heads {16, 24, 32}, members 1..64: the plan passes the check (every
 *     (band, head) once, bands in order, members in range, loads within one);
 *   - the member rule is the one the card path calls (pn_tt_attn_deep_member);
 *   - negative controls: a dropped unit, a duplicated unit, an out-of-range member, a reordered
 *     band and an unbalanced load are each refused by name;
 *   - THE FLOW CONTROL against a fake ring per member that never completes anything until it is
 *     settled (the worst card): for bands 1..64, members 1..60, depths 1..16, no ring ever holds
 *     more than `depth`, every (band, head) is enqueued once on the member the schedule names,
 *     and the drain leaves every ring empty; a failed settle stops the issue.
 * Build with -DPN_TT_ATTN_DEEP_RED (the last unit is never queued) or
 * -DPN_TT_ATTN_DEEP_RED_NO_SETTLE (no settle before a full ring: the overrun that wedged 12 cores)
 * to see it fail.
 */
#include "../src/accel/tt/pn_tt_attn_deep.h"

#include <stdio.h>
#include <string.h>

#define MAX_UNITS (64u * 32u)

static int g_failed;
static int g_checks;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failed++;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static void test_sweep(void)
{
    static const uint32_t heads_list[] = { 16u, 24u, 32u };
    static pn_tt_attn_deep_unit units[MAX_UNITS];
    uint32_t bands, h, members;
    int bad = 0;

    for (bands = 1u; bands <= 64u; bands++)
        for (h = 0; h < sizeof(heads_list) / sizeof(heads_list[0]); h++)
            for (members = 1u; members <= 64u; members++) {
                const uint32_t heads = heads_list[h];
                const uint32_t count =
                    pn_tt_attn_deep_plan(bands, heads, members, units, MAX_UNITS);
                const char *defect =
                    pn_tt_attn_deep_plan_check(units, count, bands, heads, members);
                uint32_t index;

                if (defect) {
                    if (bad++ < 3)
                        fprintf(stderr, "  bands %u heads %u members %u: %s\n", bands, heads,
                                members, defect);
                    continue;
                }
                for (index = 0; index < count; index++)
                    if (units[index].member != pn_tt_attn_deep_member(index, members))
                        bad++;
            }
    check(bad == 0, "every swept schedule is legal and uses the card path's member rule");
}

static void test_negative_controls(void)
{
    static pn_tt_attn_deep_unit units[MAX_UNITS];
    const uint32_t bands = 4u, heads = 24u, members = 12u;
    uint32_t count = pn_tt_attn_deep_plan(bands, heads, members, units, MAX_UNITS);
    pn_tt_attn_deep_unit saved;

    check(pn_tt_attn_deep_plan(0u, heads, members, units, MAX_UNITS) == 0u &&
              pn_tt_attn_deep_plan(bands, heads, members, units, 10u) == 0u,
          "a zero count or a short buffer plans nothing");
    check(pn_tt_attn_deep_plan_check(units, count - 1u, bands, heads, members) != NULL,
          "a dropped unit is refused");
    saved = units[5];
    units[5] = units[4];
    check(pn_tt_attn_deep_plan_check(units, count, bands, heads, members) != NULL,
          "a duplicated unit is refused");
    units[5] = saved;
    units[7].member = members;
    check(pn_tt_attn_deep_plan_check(units, count, bands, heads, members) != NULL,
          "an out-of-range member is refused");
    units[7].member = pn_tt_attn_deep_member(7u, members);
    saved = units[30];
    units[30] = units[0];
    units[0] = saved;
    check(pn_tt_attn_deep_plan_check(units, count, bands, heads, members) != NULL,
          "a band queued out of order is refused");
    units[0] = units[30];
    units[30] = saved;
    count = pn_tt_attn_deep_plan(bands, heads, members, units, MAX_UNITS);
    units[0].member = 1u;
    check(pn_tt_attn_deep_plan_check(units, count, bands, heads, members) != NULL,
          "an unbalanced load is refused");
}

/** A member's ring on the worst card: nothing completes until it is settled. */
typedef struct fake_card {
    uint32_t occupancy[PN_TT_ATTN_DEEP_MAX_MEMBERS];
    uint32_t capacity;
    uint32_t overruns;
    uint32_t enqueued;
    uint32_t misplaced;
    uint32_t members;
    int fail_settle_member; /**< -1 none */
    unsigned char seen[64u * 32u];
} fake_card;

static int fake_enqueue(void *context, uint32_t member, uint32_t band, uint32_t head)
{
    fake_card *card = context;

    if (++card->occupancy[member] > card->capacity)
        card->overruns++;
    if (member != pn_tt_attn_deep_member(card->enqueued, card->members))
        card->misplaced++;
    card->seen[band * 32u + head]++;
    card->enqueued++;
    return 0;
}

static int fake_settle(void *context, uint32_t member)
{
    fake_card *card = context;

    if ((int)member == card->fail_settle_member)
        return -1;
    card->occupancy[member] = 0u;
    return 0;
}

static void test_flow(void)
{
    static fake_card card;
    pn_tt_attn_deep_flow flow;
    pn_tt_attn_deep_io io = { &card, fake_enqueue, fake_settle };
    uint32_t bands, members, depth, band, m, i;
    int bad = 0, overran = 0;

    for (bands = 1u; bands <= 64u; bands += 7u)
        for (members = 1u; members <= 60u; members += 3u)
            for (depth = 1u; depth <= 16u; depth += 5u) {
                memset(&card, 0, sizeof(card));
                card.capacity = depth;
                card.members = members;
                card.fail_settle_member = -1;
                if (pn_tt_attn_deep_flow_begin(&flow, members, depth) != 0) {
                    bad++;
                    continue;
                }
                for (band = 0; band < bands; band++)
                    if (pn_tt_attn_deep_issue_band(&flow, band, 24u, &io) != 0)
                        bad++;
                if (pn_tt_attn_deep_flow_drain(&flow, &io) != 0)
                    bad++;
                if (card.overruns)
                    overran++;
                for (m = 0; m < members; m++)
                    if (card.occupancy[m] != 0u || flow.outstanding[m] != 0u)
                        bad++;
                for (i = 0; i < bands * 32u; i++)
                    if ((i % 32u) < 24u ? card.seen[i] != 1u : card.seen[i] != 0u)
                        bad++;
                if (card.misplaced || card.enqueued != bands * 24u || flow.cursor != 0u)
                    bad++;
            }
    check(overran == 0, "no member's ring ever holds more than the flow's depth");
    check(bad == 0, "every unit once, on the scheduled member, and the drain empties every ring");

    memset(&card, 0, sizeof(card));
    card.capacity = 2u;
    card.members = 3u;
    card.fail_settle_member = 1;
    (void)pn_tt_attn_deep_flow_begin(&flow, 3u, 2u);
    check(pn_tt_attn_deep_issue_band(&flow, 0u, 24u, &io) != 0 && card.enqueued < 24u,
          "a failed settle stops the issue");
    check(pn_tt_attn_deep_flow_begin(&flow, 0u, 2u) != 0 &&
              pn_tt_attn_deep_flow_begin(&flow, PN_TT_ATTN_DEEP_MAX_MEMBERS + 1u, 2u) != 0 &&
              pn_tt_attn_deep_flow_begin(&flow, 4u, 0u) != 0,
          "a zero or oversized member count, or a zero depth, is refused");
}

int main(void)
{
    test_sweep();
    test_negative_controls();
    test_flow();
    printf("test_tt_attn_deep: %d of %d checks passed\n", g_checks - g_failed, g_checks);
    return g_failed == 0 ? 0 : 1;
}
