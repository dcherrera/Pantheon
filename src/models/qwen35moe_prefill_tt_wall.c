/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_tt_wall.c
 * @brief Region wall clocks for the TT batched prefill (see the header).
 *
 * status: measurement only.
 * issues: process-global; the prefill is single-threaded and one call at a time.
 * todo:   none.
 */
#include "qwen35moe_prefill_tt_wall.h"

#include <stdio.h>

#include "../core/bench.h"

static const char *const kPrefillWallNames[PN_PREFILL_WALL_COUNT] = {
    "setup", "state-reset", "grid-begin", "embed", "stage", "layers", "residual", "mtp",
    "flush", "grid-end", "restage", "tall-exit", "param-reset", "tail-forward"
};

static double g_wall_seconds[PN_PREFILL_WALL_COUNT];
static double g_wall_accounted[PN_PREFILL_WALL_COUNT];
static double g_open_wall[PN_PREFILL_WALL_COUNT];
static double g_open_accounted[PN_PREFILL_WALL_COUNT];
static double g_total_mark;
static double g_total_accounted_mark;
static double g_wall_rearm[PN_PREFILL_WALL_COUNT];
static double g_open_rearm[PN_PREFILL_WALL_COUNT];
static double g_total_rearm_mark;
static double (*g_rearm_clock)(void);

static double prefill_rearm_now(void)
{
    return g_rearm_clock ? g_rearm_clock() : 0.0;
}

void pn_prefill_wall_set_rearm_clock(double (*clock)(void))
{
    g_rearm_clock = clock;
}

static double prefill_wall_now(void)
{
    return pn_time_ms() / 1000.0;
}

void pn_prefill_wall_begin(double accounted_seconds)
{
    for (int i = 0; i < PN_PREFILL_WALL_COUNT; i++) {
        g_wall_seconds[i] = 0.0;
        g_wall_accounted[i] = 0.0;
        g_wall_rearm[i] = 0.0;
    }
    g_total_rearm_mark = prefill_rearm_now();
    g_total_accounted_mark = accounted_seconds;
    g_total_mark = prefill_wall_now();
}

void pn_prefill_wall_open(pn_prefill_wall_region region, double accounted_seconds)
{
    g_open_accounted[region] = accounted_seconds;
    g_open_rearm[region] = prefill_rearm_now();
    g_open_wall[region] = prefill_wall_now();
}

void pn_prefill_wall_close(pn_prefill_wall_region region, double accounted_seconds)
{
    g_wall_seconds[region] += prefill_wall_now() - g_open_wall[region];
    g_wall_accounted[region] += accounted_seconds - g_open_accounted[region];
    g_wall_rearm[region] += prefill_rearm_now() - g_open_rearm[region];
}

void pn_prefill_wall_report(int n_tokens, double accounted_seconds)
{
    const double total = prefill_wall_now() - g_total_mark;
    const double accounted = accounted_seconds - g_total_accounted_mark;

    fprintf(stderr, "qwen35_tt: prefill WALL — %d tokens in %.4f s (%.1f tok/s); "
                    "batched+sequential+transfer %.4f s, OUTSIDE them %.4f s; re-arms %.4f s\n",
            n_tokens, total, total > 0.0 ? n_tokens / total : 0.0, accounted,
            total - accounted, prefill_rearm_now() - g_total_rearm_mark);
    for (int i = 0; i < PN_PREFILL_WALL_COUNT; i++)
        if (g_wall_seconds[i] > 0.0)
            fprintf(stderr, "qwen35_tt:   wall %-12s %.4f s, of it in the lines %.4f, "
                            "outside %.4f, re-arms %.4f\n",
                    kPrefillWallNames[i], g_wall_seconds[i], g_wall_accounted[i],
                    g_wall_seconds[i] - g_wall_accounted[i], g_wall_rearm[i]);
}
