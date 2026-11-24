/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_unload_rule.c
 * @brief  Implementation of the unload rule (see local_unload_rule.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_unload_rule.h"

#include <stdio.h>
#include <string.h>

#include "people_store.h"

const mr_info *pn_local_registry_row_of(const pn_loaded_entry *e, const mr_info *info,
                                        int n_info)
{
    int first = e->n_units > 0 ? e->units[0] : -1;

    /* The registry numbers copies as the loaded list does, so a CPU copy (no
     * units) is found by its number. */
    for (int i = 0; i < n_info; i++)
        if (strcmp(info[i].name, e->catalog_id) == 0 &&
            (first < 0 ? info[i].n_dies == 0 && info[i].replica == e->replica
                       : (info[i].n_dies > 0 && info[i].dies[0] == first)))
            return &info[i];
    return NULL;
}

/* Who loaded it, for the reason: the handle, else the id of a removed person. */
static void loader_name(pn_local_ctx *ctx, const char *person, char *out, size_t cap)
{
    pn_person p;

    if (ctx->people && person[0] && pn_people_get(ctx->people, person, &p) == PN_PEOPLE_OK)
        snprintf(out, cap, "%s", p.handle);
    else
        snprintf(out, cap, "%s", person[0] ? "a person no longer on this box" : "nobody known");
}

/* The refusal for one remembered replica, or 0. */
static int entry_refusal(pn_local_ctx *ctx, const pn_loaded_entry *e, const char *person,
                         const mr_info *info, int n_info, char *why, size_t cap)
{
    const mr_info *row = pn_local_registry_row_of(e, info, n_info);
    char who[PN_PERSON_HANDLE_MAX + 32];

    if (strcmp(e->loaded_by, person) != 0) {
        loader_name(ctx, e->loaded_by, who, sizeof(who));
        snprintf(why, cap, "instance %d of %s was loaded by %s", e->replica, e->catalog_id, who);
        return -1;
    }
    if (row && row->sessions_used > 0) {
        snprintf(why, cap, "instance %d of %s has %d session%s on it", e->replica,
                 e->catalog_id, row->sessions_used, row->sessions_used == 1 ? "" : "s");
        return -1;
    }
    return 0;
}

/* Is registry row `row` one of the remembered replicas? */
static int row_is_remembered(const mr_info *row, const pn_loaded_entry *list, int n,
                             const mr_info *info, int n_info)
{
    for (int i = 0; i < n; i++)
        if (pn_local_registry_row_of(&list[i], info, n_info) == row)
            return 1;
    return 0;
}

int pn_local_unload_refusal(pn_local_ctx *ctx, const char *catalog_id, int replica,
                            const char *person, char *why, size_t cap)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    mr_info info[MR_MAX_MODELS];
    int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
    int n_info = ctx->reg ? mr_snapshot(ctx->reg, info, MR_MAX_MODELS) : 0;

    for (int i = 0; i < n; i++) {
        if (strcmp(list[i].catalog_id, catalog_id) != 0 ||
            (replica > 0 && list[i].replica != replica))
            continue;
        if (entry_refusal(ctx, &list[i], person, info, n_info, why, cap) != 0)
            return -1;
    }
    if (replica > 0)
        return 0;
    for (int i = 0; i < n_info; i++) {
        if (strcmp(info[i].name, catalog_id) != 0 ||
            row_is_remembered(&info[i], list, n, info, n_info))
            continue;
        snprintf(why, cap, "instance %d of %s was not loaded through the box (an admin may "
                 "unload it)", info[i].replica, catalog_id);
        return -1;
    }
    return 0;
}
