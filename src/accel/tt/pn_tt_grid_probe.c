/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_probe.c
 * @brief Per-shape timing of prefill-grid projections. See pn_tt_grid_probe.h.
 * @status Opt-in measurement (`PN_TT_GRID_PROBE=1`).
 * @issues None
 * @todo None
 */
#include "pn_tt_grid_probe.h"

#include <stdio.h>
#include <stdlib.h>

#define PN_TT_GRID_PROBE_SHAPES 32u

typedef struct {
    uint32_t mt, kt, nt, members;
    uint64_t count;
    double issue_seconds, await_seconds;
} pn_tt_grid_probe_shape;

static pn_tt_grid_probe_shape g_shapes[PN_TT_GRID_PROBE_SHAPES];
static uint32_t g_shape_count;

bool pn_tt_grid_probe_enabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *env = getenv("PN_TT_GRID_PROBE");

        enabled = env && *env && atoi(env) != 0;
    }
    return enabled == 1;
}

void pn_tt_grid_probe_book(uint32_t mt, uint32_t kt, uint32_t nt, uint32_t members,
                           double issue_seconds, double await_seconds)
{
    pn_tt_grid_probe_shape *shape = NULL;

    for (uint32_t i = 0; i < g_shape_count; i++)
        if (g_shapes[i].mt == mt && g_shapes[i].kt == kt && g_shapes[i].nt == nt &&
            g_shapes[i].members == members)
            shape = &g_shapes[i];
    if (!shape) {
        if (g_shape_count == PN_TT_GRID_PROBE_SHAPES)
            return;
        shape = &g_shapes[g_shape_count++];
        shape->mt = mt;
        shape->kt = kt;
        shape->nt = nt;
        shape->members = members;
    }
    shape->count++;
    shape->issue_seconds += issue_seconds;
    shape->await_seconds += await_seconds;
}

void pn_tt_grid_probe_report(void)
{
    for (uint32_t i = 0; i < g_shape_count; i++) {
        const pn_tt_grid_probe_shape *s = &g_shapes[i];
        /* One 32x32x32 tile product is 2 * 32^3 FLOP. */
        const double flop = 65536.0 * (double)s->mt * s->kt * s->nt * (double)s->count;
        const double total = s->issue_seconds + s->await_seconds;

        fprintf(stderr, "pn_tt: GRID PROBE — Mt %u Kt %u Nt %u on %u cores: %llu x, issue "
                        "%.3f ms + await %.3f ms each = %.2f TFLOP/s (await only %.2f)\n",
                s->mt, s->kt, s->nt, s->members, (unsigned long long)s->count,
                1e3 * s->issue_seconds / (double)s->count,
                1e3 * s->await_seconds / (double)s->count,
                total > 0.0 ? flop / total / 1e12 : 0.0,
                s->await_seconds > 0.0 ? flop / s->await_seconds / 1e12 : 0.0);
    }
    g_shape_count = 0u;
}
