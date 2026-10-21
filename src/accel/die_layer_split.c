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

int pn_plan_die_layer_split(int n_layer, int n_die, int override_boundary,
                            pn_die_layer_split *out)
{
    if (!out || !split_args_ok(n_layer, n_die))
        return -1;

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

    if (lo_out)
        *lo_out = split_boundary(n_layer, n_die, idx);
    if (hi_out)
        *hi_out = (idx == n_die - 1) ? n_layer
                                     : split_boundary(n_layer, n_die, idx + 1);
    return 0;
}
