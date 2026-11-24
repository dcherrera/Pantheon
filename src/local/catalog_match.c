/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog_match.c
 * @brief  Implementation of profile matching (see catalog_match.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "catalog_match.h"

#include <stdio.h>
#include <string.h>

#include "load_options.h"

static int units_of_type(const pn_hw_report *hw, const char *type)
{
    int n = 0;

    for (int i = 0; i < hw->n_units; i++)
        n += strcmp(hw->units[i].type, type) == 0;
    return n;
}

bool pn_cat_request_fits(const pn_unit_request *r, uint64_t min_host_ram, const pn_hw_report *hw,
                         uint64_t host_ram, char *reason, size_t cap)
{
    int ids[PN_UNIT_PLACE_MAX], n = 0, total = units_of_type(hw, r->unit_type);

    reason[0] = '\0';
    if (total < r->count) {
        snprintf(reason, cap, "needs %d %s, this box has %d", r->count, r->unit_type, total);
        return false;
    }
    if (min_host_ram && host_ram < min_host_ram) {
        snprintf(reason, cap, "needs %.1f GB host RAM, this box has %.1f GB",
                 (double)min_host_ram / 1e9, (double)host_ram / 1e9);
        return false;
    }
    return pn_unit_place(hw, r, ids, &n, reason, cap) == 0;
}

bool pn_cat_profile_fits(const pn_cat_profile *p, const pn_hw_report *hw, uint64_t host_ram,
                         char *reason, size_t cap)
{
    pn_load_option opts[PN_LOAD_OPTIONS_MAX], chosen;
    int n;

    if (!pn_cat_request_fits(&p->hardware, p->min_host_ram, hw, host_ram, reason, cap))
        return false;
    if (!p->options.present)
        return true;
    /* Smart load options: it fits when at least one choice fits the units now. */
    n = pn_load_options_fit(p, hw, opts, PN_LOAD_OPTIONS_MAX);
    for (int i = 0; i < n; i++)
        if (opts[i].fits)
            return true;
    if (n > 0)
        pn_load_options_choose(p, hw, opts[0].ctx_cap, opts[0].max_sessions, &chosen, reason,
                               cap);
    return false;
}

/* >0 when x belongs after y: fitting first, then lower rank first. */
static int match_order(const pn_cat_match *x, const pn_cat_match *y)
{
    if (x->fits != y->fits)
        return x->fits ? -1 : 1;
    return (x->profile->rank > y->profile->rank) - (x->profile->rank < y->profile->rank);
}

int pn_cat_match_profiles(const pn_catalog_manifest *m, const pn_hw_report *hw,
                          uint64_t host_ram, pn_cat_match *out, int max)
{
    int n = 0;

    for (int i = 0; i < m->n_models; i++) {
        for (int j = 0; j < m->models[i].n_profiles && n < max; j++) {
            pn_cat_match *e = &out[n++];
            e->model = &m->models[i];
            e->profile = &m->models[i].profiles[j];
            e->fits = pn_cat_profile_fits(e->profile, hw, host_ram, e->reason, sizeof(e->reason));
        }
    }
    /* Insertion sort: stable, so catalog order holds inside a rank. */
    for (int i = 1; i < n; i++) {
        pn_cat_match key = out[i];
        int j = i - 1;
        while (j >= 0 && match_order(&out[j], &key) > 0) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
    return n;
}
