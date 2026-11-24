/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file die_layer_split.c
 * @brief Implementation of the shared N-die layer partitioner.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "die_layer_split.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* The boundary before die `idx` of an n_die-way split of n_layer layers.
 * Computed in `long` because n_layer * idx overflows int for absurd inputs
 * long before the caller's validation would notice, and a wrapped boundary is
 * a silently-wrong split rather than a crash. */
static int split_boundary(int n_layer, int n_die, int idx)
{
    return (int)((long)n_layer * (long)idx / (long)n_die);
}

/* Shared validation for both entry points. */
static int split_args_ok(int n_layer, int n_die)
{
    return n_layer > 0 && n_die >= 1 &&
           n_die <= PN_DIE_LAYER_SPLIT_MAX_DIES && n_die <= n_layer;
}

/* PN_DIE_SPLIT_LAYERS=b1,b2,...: the N-1 boundaries of an N-die split, for
 * placements where the flat split does not fit (the head die also carries the
 * LM head and the MTP head). Used only when it names exactly n_die-1 strictly
 * increasing boundaries inside (0, n_layer); anything else is ignored with a
 * warning, never half-applied. Fills b[0..n_die] (b[0] = 0, b[n_die] =
 * n_layer) and returns 1 when it applies. */
static int env_boundaries(int n_layer, int n_die, int *b)
{
    const char *env = getenv("PN_DIE_SPLIT_LAYERS");
    if (!env || !*env || n_die < 2)
        return 0;
    const char *p = env;
    int n = 0;
    b[0] = 0;
    while (*p && n < n_die - 1) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p || v <= b[n] || v >= n_layer)
            break;
        b[++n] = (int)v;
        p = (*end == ',') ? end + 1 : end;
        if (*end != ',' && *end)
            break;
    }
    if (n != n_die - 1 || *p) {
        fprintf(stderr, "die_layer_split: PN_DIE_SPLIT_LAYERS='%s' ignored — "
                "needs %d increasing boundaries in (0,%d)\n", env, n_die - 1,
                n_layer);
        return 0;
    }
    b[n_die] = n_layer;
    return 1;
}

int pn_plan_die_layer_split(int n_layer, int n_die, int override_boundary,
                            pn_die_layer_split *out)
{
    if (!out || !split_args_ok(n_layer, n_die))
        return -1;

    int eb[PN_DIE_LAYER_SPLIT_MAX_DIES + 1];
    if (env_boundaries(n_layer, n_die, eb)) {
        for (int i = 0; i < n_die; i++) {
            out->layer_lo[i] = eb[i];
            out->layer_hi[i] = eb[i + 1];
        }
        out->n_die = n_die;
        return 0;
    }
    for (int i = 0; i < n_die; i++) {
        out->layer_lo[i] = split_boundary(n_layer, n_die, i);
        out->layer_hi[i] = (i == n_die - 1) ? n_layer
                                            : split_boundary(n_layer, n_die, i + 1);
    }
    /* A single boundary describes a 2-die split completely and nothing wider,
     * so the override applies at exactly that width. Clamped inside (0,n_layer)
     * so an out-of-range knob narrows the split rather than emptying a die. */
    if (n_die == 2 && override_boundary > 0) {
        int b = override_boundary;
        if (b >= n_layer)
            b = n_layer - 1;
        out->layer_hi[0] = b;
        out->layer_lo[1] = b;
    }
    out->n_die = n_die;
    return 0;
}

int pn_die_layer_range(int n_layer, int n_die, int idx, int *lo_out,
                       int *hi_out)
{
    if (!split_args_ok(n_layer, n_die) || idx < 0 || idx >= n_die)
        return -1;

    int eb[PN_DIE_LAYER_SPLIT_MAX_DIES + 1];
    if (env_boundaries(n_layer, n_die, eb)) {
        if (lo_out)
            *lo_out = eb[idx];
        if (hi_out)
            *hi_out = eb[idx + 1];
        return 0;
    }

    if (lo_out)
        *lo_out = split_boundary(n_layer, n_die, idx);
    if (hi_out)
        *hi_out = (idx == n_die - 1) ? n_layer
                                     : split_boundary(n_layer, n_die, idx + 1);
    return 0;
}
