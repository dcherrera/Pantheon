/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_bs_pd_budget.c — the prefill an overlap cycle may issue
 * (batch_sched_pd_budget.h): QWEN35_BS_PD_PIECES parsing and the plain-step EMA
 * the stats line compares a cycle with, and when an overlap prefill may hand
 * the turn over (bs_pd_may_yield).
 */

#include "../src/serve/batch_sched_pd_budget.h"

#include <math.h>
#include <stdio.h>

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { g_failed++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static int near(double a, double b) { return fabs(a - b) < 1e-9; }

int main(void)
{
    CHECK(bs_pd_pieces_parse(NULL) == BS_PD_PIECES_DEFAULT, "unset -> default");
    CHECK(bs_pd_pieces_parse("") == BS_PD_PIECES_DEFAULT, "empty -> default");
    CHECK(bs_pd_pieces_parse("0") == BS_PD_PIECES_DEFAULT, "zero -> default");
    CHECK(bs_pd_pieces_parse("-3") == BS_PD_PIECES_DEFAULT, "negative -> default");
    CHECK(bs_pd_pieces_parse("2") == 2, "2 kept");
    CHECK(bs_pd_pieces_parse("999") == BS_PD_PIECES_MAX, "clamped to max");

    bs_pd_budget b;
    bs_pd_budget_init(&b, 2);
    CHECK(b.pieces == 2 && near(b.plain_ms[3], 0.0), "init");

    bs_pd_budget_note_plain(&b, 3, 80.0);
    CHECK(near(b.plain_ms[3], 80.0), "first plain step seeds the EMA");
    bs_pd_budget_note_plain(&b, 3, 90.0);
    CHECK(near(b.plain_ms[3], 81.0), "EMA 0.9/0.1");
    bs_pd_budget_note_plain(&b, 3, -1.0);
    CHECK(near(b.plain_ms[3], 81.0), "non-positive time ignored");

    /* Out-of-range widths are ignored, not indexed. */
    bs_pd_budget_note_plain(&b, 0, 50.0);
    bs_pd_budget_note_plain(&b, BS_PD_MAX_WIDTH + 1, 50.0);
    CHECK(near(b.plain_ms[0], 0.0), "width 0 ignored");

    /* Hand-over: only for a waiter, and never once the last chunk is in (the
     * turn checkpoint and the head need the live buffers it leaves). */
    CHECK(!bs_pd_may_yield(false, 0, false), "nobody waiting -> keep the turn");
    CHECK(bs_pd_may_yield(false, 1, false), "finalize waiting, chunks left -> yield");
    CHECK(bs_pd_may_yield(false, 0, true), "shorter prefill waiting, chunks left -> yield");
    CHECK(!bs_pd_may_yield(true, 1, false), "finalize waiting, last chunk in -> keep");
    CHECK(!bs_pd_may_yield(true, 0, true), "shorter waiting, last chunk in -> keep");
    CHECK(!bs_pd_may_yield(true, 2, true), "both waiting, last chunk in -> keep");

    printf("test_bs_pd_budget: %d/%d checks passed\n", g_checks - g_failed, g_checks);
    return g_failed ? 1 : 0;
}
