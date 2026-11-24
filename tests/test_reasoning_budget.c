/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_reasoning_budget.c — the per-turn reasoning plan (src/serve/reasoning_budget.h).
 *
 *   Reserve: with the CLI's --reasoning-budget 8000 and Argus's max_tokens 8192, a
 *   forced close left ~190 tokens for the tool call, so a write was cut and refused.
 *   The plan keeps PN_REASONING_RESERVE_DEFAULT of max_tokens for the answer.
 *   Clean close: the close landed on exactly token N, mid-sentence. The plan closes
 *   on the first line end in a window just before the cap.
 *
 * run_turn() walks one turn the way both serve decode loops do: count each
 * reasoning token, ask pn_reasoning_should_close, inject one "</think>" token,
 * then emit the tool call until max_tokens runs out.
 */

#include "../src/serve/reasoning_budget.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

typedef struct {
    int closed_at;   /* reasoning tokens when the close fired (0 = never) */
    int closed_on_line_end;
    int call_toks;   /* tool-call tokens emitted before max_tokens ran out */
} turn_out;

/* A think that never self-closes (the hard turn), breaking lines every
 * `nl_every` tokens (0 = never), then a call of `call_len` tokens. */
static turn_out run_turn(const pn_reasoning_plan *plan, int max_tokens, int nl_every,
                         int call_len)
{
    turn_out o = { 0, 0, 0 };
    int toks = 0, in_think = 1;

    for (int generated = 0; generated < max_tokens; generated++) {
        if (in_think) {
            int line_end = nl_every > 0 && (toks + 1) % nl_every == 0;
            const char *piece = line_end ? " done.\n" : " word";
            toks++;
            if (pn_reasoning_should_close(plan, toks, pn_reasoning_piece_ends_line(piece))) {
                o.closed_at = toks;
                o.closed_on_line_end = line_end;
                generated++;          /* the injected "</think>" */
                in_think = 0;
            }
        } else if (o.call_toks < call_len) {
            o.call_toks++;
        } else {
            break;
        }
    }
    return o;
}

/* The audit case: budget 8000, max_tokens 8192, a 1,500-token write after a
 * think that runs to the cap. The pre-plan cap ({8000, 8000}) leaves 191. */
static void test_long_think_leaves_room_for_the_call(void)
{
    printf("long_think_leaves_room_for_the_call:\n");
    pn_reasoning_plan old = { 8000, 8000 };
    pn_reasoning_plan p = pn_reasoning_plan_make(8000, 8192, 0, 0);
    turn_out a = run_turn(&old, 8192, 0, 1500);
    turn_out b = run_turn(&p, 8192, 0, 1500);

    CHECK(a.call_toks == 191, "pre-plan cap: 191 tokens left, the write is cut");
    CHECK(p.hard == 8192 - PN_REASONING_RESERVE_DEFAULT, "cap lowered to keep the reserve");
    CHECK(b.closed_at == p.hard, "the close fires at the lowered cap");
    CHECK(b.call_toks == 1500, "the whole 1,500-token call fits");
    CHECK(8192 - b.closed_at - 1 >= PN_REASONING_RESERVE_DEFAULT - 1,
          "at least the reserve is left after the close");
}

static void test_plan_math(void)
{
    pn_reasoning_plan p;

    printf("plan_math:\n");
    p = pn_reasoning_plan_make(2000, 8192, 0, 256);
    CHECK(p.hard == 2000 && p.soft == 1750, "the K80 host today: 2000 untouched, window 1750-2000");
    p = pn_reasoning_plan_make(8000, 8192, 0, 256);
    CHECK(p.hard == 5120 && p.soft == 4864, "CLI 8000 / 8192: cap 5120, window from 4864");
    p = pn_reasoning_plan_make(5120, 8192, 3072, 256);
    CHECK(p.hard == 5120 && p.soft == 4864, "proposed K80 value: 5120 + 3072 = 8192");
    p = pn_reasoning_plan_make(8000, 8192, 1024, 256);
    CHECK(p.hard == 7168, "a per-profile reserve of 1024");
    p = pn_reasoning_plan_make(8000, 1024, 0, 256);
    CHECK(p.hard == 512 && p.soft == 448, "small max_tokens: reserve capped at half");
    p = pn_reasoning_plan_make(8000, 0, 0, 256);
    CHECK(p.hard == 8000, "max_tokens unknown: no reserve clamp");
    p = pn_reasoning_plan_make(0, 8192, 0, 256);
    CHECK(p.hard == 0 && p.soft == 0, "budget 0: no cap");
    p = pn_reasoning_plan_make(-5, 8192, 0, 256);
    CHECK(p.hard == 0, "negative budget: no cap");
    p = pn_reasoning_plan_make(2000, 8192, 0, 0);
    CHECK(p.soft == p.hard, "grace 0: the old exact cut point");
    p = pn_reasoning_plan_make(4, 8192, 0, 256);
    CHECK(p.hard == 4 && p.soft == 4, "tiny budget: window is hard/8 = 0");
    p = pn_reasoning_plan_make(8000, 1, 0, 256);
    CHECK(p.hard >= 1, "max_tokens 1: the cap never drops below one token");
}

/* The close lands on a whole line: the first line end inside the window, never
 * one before it, and the hard cap when no line ends in the window. */
static void test_close_on_line_end(void)
{
    printf("close_on_line_end:\n");
    pn_reasoning_plan p = pn_reasoning_plan_make(2000, 8192, 0, 256);
    turn_out o = run_turn(&p, 8192, 60, 100);

    /* Line ends at 60, 120, ...: 1740 < soft 1750, so the first in-window one is 1800. */
    CHECK(o.closed_at == 1800 && o.closed_on_line_end, "closes on the line end at 1800");
    o = run_turn(&p, 8192, 0, 100);
    CHECK(o.closed_at == 2000 && !o.closed_on_line_end, "no line end: closes at the cap");
    o = run_turn(&p, 8192, 1750, 100);
    CHECK(o.closed_at == 1750, "a line end on the soft point itself closes");
    o = run_turn(&p, 8192, 1749, 100);
    CHECK(o.closed_at == 2000, "a line end one token before the window does not");
    pn_reasoning_plan none = pn_reasoning_plan_make(0, 8192, 0, 256);
    o = run_turn(&none, 8192, 60, 100);
    CHECK(o.closed_at == 0, "no plan: never closes");
}

static void test_pieces_and_drafts(void)
{
    printf("pieces_and_drafts:\n");
    CHECK(pn_reasoning_piece_ends_line(".\n") && pn_reasoning_piece_ends_line("\n\n") &&
          pn_reasoning_piece_ends_line("a\nb"), "pieces holding a newline end a line");
    CHECK(!pn_reasoning_piece_ends_line(" word") && !pn_reasoning_piece_ends_line("") &&
          !pn_reasoning_piece_ends_line(NULL), "others do not");

    pn_reasoning_plan p = pn_reasoning_plan_make(2000, 8192, 0, 256);
    CHECK(pn_reasoning_draft_room(&p, 1700) == 49, "drafts stop short of the window");
    CHECK(pn_reasoning_draft_room(&p, 1749) == 0, "no drafts on the token before it");
    pn_reasoning_plan none = { 0, 0 };
    CHECK(pn_reasoning_draft_room(&none, 5) == INT_MAX, "no plan: drafts unbounded");
    CHECK(!pn_reasoning_should_close(NULL, 99999, true), "NULL plan never closes");
}

static void test_grace_env(void)
{
    printf("grace_env:\n");
    unsetenv("PN_REASONING_LINE_GRACE");
    CHECK(pn_reasoning_line_grace() == PN_REASONING_LINE_GRACE_DEFAULT, "unset = default");
    setenv("PN_REASONING_LINE_GRACE", "0", 1);
    CHECK(pn_reasoning_line_grace() == 0, "0 turns the window off");
    pn_reasoning_plan p = pn_reasoning_plan_for_turn(2000, 8192, 0, "m");
    CHECK(p.hard == 2000 && p.soft == 2000, "for_turn reads the env");
    setenv("PN_REASONING_LINE_GRACE", "64", 1);
    CHECK(pn_reasoning_line_grace() == 64, "a set value");
    setenv("PN_REASONING_LINE_GRACE", "12x", 1);
    CHECK(pn_reasoning_line_grace() == PN_REASONING_LINE_GRACE_DEFAULT, "junk = default");
    setenv("PN_REASONING_LINE_GRACE", "-3", 1);
    CHECK(pn_reasoning_line_grace() == PN_REASONING_LINE_GRACE_DEFAULT, "negative = default");
    unsetenv("PN_REASONING_LINE_GRACE");
}

int main(void)
{
    test_long_think_leaves_room_for_the_call();
    test_plan_math();
    test_close_on_line_end();
    test_pieces_and_drafts();
    test_grace_env();
    printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
