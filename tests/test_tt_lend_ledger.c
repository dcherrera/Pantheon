/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_lend_ledger.c — the L1a prefill lend ledger (pn_tt_lend_ledger.h), NO CARD, NO USE_TT.
 *
 * A lent core restored twice is two firmware boots under one triad; a lent core never restored is a
 * decode family short a member for the rest of the process; a core whose borrowed member did not
 * exit, booted again, is the wedge. Checked:
 *   - the threshold: no lending without deep attention or below the row floor; 0 floor = never;
 *   - the defaults: a dense model lends from 16,384 rows unasked, MoE never; the knobs win both ways;
 *     unset PN_TT_ATTN_DEEP is on exactly for a prompt with lent cores;
 *   - every lent core comes back once, in lending order, with its member index;
 *   - a duplicate core and a full ledger are refused;
 *   - a stuck core is never handed back, the others still are;
 *   - a second take restores nothing; a popped entry (gone to the grid) is not restored here.
 * Build with -DPN_TT_LEND_LEDGER_RED (stuck cores handed back) or -DPN_TT_LEND_DEFAULT_RED (deep
 * off unless asked) to see it fail.
 */
#include "../src/accel/tt/pn_tt_lend_ledger.h"

#include <stdio.h>

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

static void test_threshold(void)
{
    check(!pn_tt_lend_wanted(32768u, 4096u, false), "no lending without deep attention");
    check(!pn_tt_lend_wanted(4095u, 4096u, true), "no lending below the row floor");
    check(pn_tt_lend_wanted(4096u, 4096u, true), "lending at the row floor");
    check(!pn_tt_lend_wanted(32768u, 0u, true), "a zero floor never lends");
}

static void test_defaults(void)
{
    check(pn_tt_lend_min_rows(NULL, true) == PN_TT_LEND_DENSE_MIN_ROWS, "dense: the default floor");
    check(pn_tt_lend_min_rows("", true) == PN_TT_LEND_DENSE_MIN_ROWS, "dense: empty is unset");
    check(pn_tt_lend_min_rows(NULL, false) == 0u, "MoE: off by default");
    check(pn_tt_lend_min_rows("0", true) == 0u, "PN_TT_PREFILL_LEND=0 turns a dense model off");
    check(pn_tt_lend_min_rows("4096", false) == 4096u, "PN_TT_PREFILL_LEND=<rows> on for MoE");
    check(pn_tt_lend_min_rows("4k", true) == PN_TT_LEND_DENSE_MIN_ROWS, "not a number: default");
    check(pn_tt_lend_deep_on(NULL, true), "deep unset: on for a prompt with lent cores");
    check(!pn_tt_lend_deep_on(NULL, false), "deep unset: off without lent cores");
    check(!pn_tt_lend_deep_on("0", true), "PN_TT_ATTN_DEEP=0 wins over lent cores");
    check(pn_tt_lend_deep_on("1", false), "PN_TT_ATTN_DEEP=1 wins without lent cores");
    check(pn_tt_lend_wanted(32768u, pn_tt_lend_min_rows(NULL, true), pn_tt_lend_deep_on(NULL, true)),
          "a 32k Qwen3.8 prompt lends with no knob set");
    check(!pn_tt_lend_wanted(7680u, pn_tt_lend_min_rows(NULL, true), pn_tt_lend_deep_on(NULL, true)),
          "an 8k Qwen3.8 prompt does not");
}

static void test_round_trip(void)
{
    static const uint32_t cores[] = {133u, 134u, 135u, 50u, 51u, 90u};
    uint32_t members[PN_TT_LEND_MAX], back[PN_TT_LEND_MAX], index, taken;
    pn_tt_lend_ledger ledger;
    int in_order = 1;

    pn_tt_lend_ledger_reset(&ledger);
    for (index = 0; index < 6u; index++)
        check(pn_tt_lend_ledger_add(&ledger, index + 1u, cores[index]), "a core is lent");
    check(!pn_tt_lend_ledger_add(&ledger, 9u, 134u), "a core already lent is refused");
    taken = pn_tt_lend_ledger_take_restorable(&ledger, members, back);
    check(taken == 6u, "every lent core comes back");
    for (index = 0; index < taken && index < 6u; index++)
        in_order &= back[index] == cores[index] && members[index] == index + 1u;
    check(in_order, "in lending order, with its member index");
    check(pn_tt_lend_ledger_take_restorable(&ledger, members, back) == 0u,
          "a second take restores nothing");
}

static void test_stuck(void)
{
    uint32_t members[PN_TT_LEND_MAX], back[PN_TT_LEND_MAX], index, taken;
    pn_tt_lend_ledger ledger;
    int stuck_back = 0;

    pn_tt_lend_ledger_reset(&ledger);
    for (index = 0; index < 4u; index++)
        (void)pn_tt_lend_ledger_add(&ledger, index, 100u + index);
    pn_tt_lend_ledger_mark_stuck(&ledger, 102u);
    taken = pn_tt_lend_ledger_take_restorable(&ledger, members, back);
    for (index = 0; index < taken; index++)
        stuck_back |= back[index] == 102u;
    check(!stuck_back, "a stuck core is never booted again");
    check(taken == 3u, "the other cores still come back");
}

static void test_pop(void)
{
    uint32_t members[PN_TT_LEND_MAX], back[PN_TT_LEND_MAX], member = 0u, core = 0u;
    pn_tt_lend_ledger ledger;

    pn_tt_lend_ledger_reset(&ledger);
    (void)pn_tt_lend_ledger_add(&ledger, 0u, 50u);
    (void)pn_tt_lend_ledger_add(&ledger, 7u, 90u);
    check(pn_tt_lend_ledger_pop(&ledger, &member, &core) && member == 7u && core == 90u,
          "pop returns the last entry");
    check(pn_tt_lend_ledger_take_restorable(&ledger, members, back) == 1u && back[0] == 50u,
          "a popped core is not restored by the ledger");
    check(!pn_tt_lend_ledger_pop(&ledger, &member, &core), "pop on an empty ledger refuses");
}

static void test_full(void)
{
    pn_tt_lend_ledger ledger;
    uint32_t index;
    int all = 1;

    pn_tt_lend_ledger_reset(&ledger);
    for (index = 0; index < PN_TT_LEND_MAX; index++)
        all &= pn_tt_lend_ledger_add(&ledger, index, index);
    check(all, "the ledger holds PN_TT_LEND_MAX cores");
    check(!pn_tt_lend_ledger_add(&ledger, 99u, 999u), "a full ledger refuses");
}

int main(void)
{
    test_threshold();
    test_defaults();
    test_round_trip();
    test_stuck();
    test_pop();
    test_full();
    printf("test_tt_lend_ledger: %d of %d checks passed\n", g_checks - g_failed, g_checks);
    return g_failed ? 1 : 0;
}
