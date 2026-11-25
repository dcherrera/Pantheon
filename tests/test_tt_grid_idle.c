/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_grid_idle.c — the grid idle guard's decision (src/accel/tt/pn_tt_grid_idle.h), NO CARD.
 *
 * The failure it guards: every grid core idles past its spin limit (~8 s) and the next pass's
 * first projection goes to dead cores. Checks, over idle 0..20 s in 10 ms steps, at the 5 s
 * limit and 2 s NOP floor: an issue re-arms exactly from the limit up; a keepalive point NOPs
 * in [2, 4.5) s and re-arms from 4.5 s; nothing ever lets a core reach 8 s idle without a NOP
 * or re-arm being due first; limit 0 and "no descriptor yet" do nothing.
 * Build with -DPN_TT_GRID_IDLE_RED (the guard never re-arms) to see it fail.
 */
#include "../src/accel/tt/pn_tt_grid_idle.h"

#include <stdio.h>

static int g_failed, g_checks;

static void check(int ok, const char *what, double idle)
{
    g_checks++;
    if (!ok) {
        g_failed++;
        if (g_failed < 20)
            fprintf(stderr, "FAIL %s at idle %.2f s\n", what, idle);
    }
}

int main(void)
{
    const double limit = 5.0, floor_s = 2.0, exit_s = 8.0;

    for (int step = 0; step <= 2000; step++) {
        const double idle = step * 0.01;
        const pn_tt_grid_idle_action at_issue = pn_tt_grid_idle_decide(idle, floor_s, limit, false);
        const pn_tt_grid_idle_action at_keep = pn_tt_grid_idle_decide(idle, floor_s, limit, true);

        check((at_issue == PN_TT_GRID_IDLE_REARM) == (idle >= limit), "issue re-arms from the limit",
              idle);
        check(at_issue != PN_TT_GRID_IDLE_NOP, "an issue never NOPs", idle);
        if (idle < floor_s)
            check(at_keep == PN_TT_GRID_IDLE_NONE, "keepalive below the floor does nothing", idle);
        else if (idle < limit - PN_TT_GRID_IDLE_MARGIN_S)
            check(at_keep == PN_TT_GRID_IDLE_NOP, "keepalive NOPs below the margin", idle);
        else
            check(at_keep == PN_TT_GRID_IDLE_REARM, "keepalive re-arms inside the margin", idle);
        if (idle >= exit_s - 3.0)
            check(at_issue == PN_TT_GRID_IDLE_REARM && at_keep == PN_TT_GRID_IDLE_REARM,
                  "no descriptor reaches a core near its idle exit", idle);
        check(pn_tt_grid_idle_decide(idle, floor_s, 0.0, false) == PN_TT_GRID_IDLE_NONE &&
                  pn_tt_grid_idle_decide(idle, floor_s, 0.0, true) == PN_TT_GRID_IDLE_NONE,
              "limit 0 is off", idle);
    }
    check(pn_tt_grid_idle_decide(-1.0, floor_s, limit, false) == PN_TT_GRID_IDLE_NONE,
          "no descriptor yet", -1.0);
    printf("test_tt_grid_idle: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
