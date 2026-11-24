/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_uninstall.c
 * @brief  Implementation of the uninstall route (see local_uninstall.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_uninstall.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/json_reader.h"
#include "catalog.h"
#include "install_worker.h"
#include "loaded_set.h"
#include "local_load.h"
#include "model_store.h"
#include "model_store_cleanup.h"

static int refuse(int *status, int code, char *err, size_t errlen, const char *msg)
{
    *status = code;
    snprintf(err, errlen, "%s", msg);
    return -1;
}

/* "GPU 0, 1" / "the p150" / "the CPU": where a registry row sits, in words. */
static void describe_units(const mr_info *row, const char *unit_type, char *out, size_t cap)
{
    size_t n = 0;

    if (row->n_dies == 0) {
        snprintf(out, cap, "the CPU");
        return;
    }
    if (strcmp(unit_type, "p150") == 0) {
        snprintf(out, cap, "the p150");
        return;
    }
    n = (size_t)snprintf(out, cap, "GPU ");
    for (int i = 0; i < row->n_dies && n < cap; i++)
        n += (size_t)snprintf(out + n, cap - n, "%s%d", i ? ", " : "", row->dies[i]);
}

/* Why catalog_id cannot be removed now, or "" when it can. Loads lock held. */
static void why_in_use(pn_local_ctx *ctx, const char *id, char *why, size_t cap)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    mr_info info[MR_MAX_MODELS];
    int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
    int n_info = ctx->reg ? mr_snapshot(ctx->reg, info, MR_MAX_MODELS) : 0;
    const char *unit_type = "";
    pn_install_status st;
    char where[64];

    why[0] = '\0';
    for (int i = 0; i < n; i++)
        if (strcmp(list[i].catalog_id, id) == 0)
            unit_type = list[i].hardware.unit_type;
    for (int i = 0; i < n_info; i++) {
        if (strcmp(info[i].name, id) != 0)
            continue;
        describe_units(&info[i], unit_type, where, sizeof(where));
        snprintf(why, cap, "%s is loaded on %s. Unload it, then remove it.", id, where);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (strcmp(list[i].catalog_id, id) != 0)
            continue;
        snprintf(why, cap, "%s is set to load when Pantheon Local starts. Unload it, then "
                 "remove it.", id);
        return;
    }
    if (ctx->installer && pn_install_worker_status(ctx->installer, id, &st) == 0 &&
        st.state != PN_INSTALL_DONE && st.state != PN_INSTALL_FAILED)
        snprintf(why, cap, "%s is still installing. Wait for it to finish, then remove it.", id);
}

int pn_local_uninstall(pn_local_ctx *ctx, const char *catalog_id, uint64_t *freed,
                       int *n_removed, int *status, char *err, size_t errlen)
{
    char why[256];
    int rc;

    *freed = 0;
    *n_removed = 0;
    pn_local_loads_lock();
    why_in_use(ctx, catalog_id, why, sizeof(why));
    if (why[0])
        rc = refuse(status, 409, err, errlen, why);
    else if (pn_model_store_remove_id(ctx->store_dir, catalog_id, freed, n_removed, why,
                                      sizeof(why)) != 0)
        rc = refuse(status, 500, err, errlen, why);
    else if (*n_removed == 0)
        rc = refuse(status, 404, err, errlen, "nothing of that model is in the store");
    else
        rc = 0;
    pn_local_loads_unlock();
    if (*n_removed)
        fprintf(stderr, "local: removed %s from the store (%d file(s), %.1f GB)\n", catalog_id,
                *n_removed, (double)*freed / 1e9);
    return rc;
}

/* Whether the files some profile of the model needs are all in the store (a
 * K80 install holds no p150-only files, spec §16). */
static bool installed_for_a_profile(const pn_local_ctx *ctx, const pn_cat_model *model)
{
    pn_cat_model *need = calloc(1, sizeof(*need));
    bool installed = false;

    for (int i = 0; need && !installed && i < model->n_profiles; i++) {
        pn_cat_model_for_units(model, pn_cat_unit_type_bit(model->profiles[i].hardware.unit_type),
                               need);
        installed = pn_model_store_installed(ctx->store_dir, need);
    }
    free(need);
    return installed;
}

/* Whether a catalog model owns every finished entry of this store group: it
 * is listed and its current files are what is stored. */
static bool group_is_current(const pn_store_group *g, const pn_local_ctx *ctx,
                             const pn_catalog_manifest *m)
{
    const pn_cat_model *model = m ? pn_cat_find_model(m, g->catalog_id) : NULL;

    if (!model)
        return false;
    /* Only downloads in progress (resumable): an install under way, not a leftover. */
    if (g->n_partial == g->n_files)
        return true;
    return installed_for_a_profile(ctx, model);
}

void pn_local_write_store_only(json_writer *jw, const pn_local_ctx *ctx,
                               const pn_catalog_manifest *m)
{
    pn_store_group *groups = calloc(PN_STORE_GROUPS_MAX, sizeof(*groups));
    int n = groups ? pn_model_store_scan(ctx->store_dir, groups, PN_STORE_GROUPS_MAX) : 0;
    const char *sep = "";

    jw_raw(jw, "\"store_only\":[");
    for (int i = 0; i < n; i++) {
        if (group_is_current(&groups[i], ctx, m))
            continue;
        jw_printf(jw, "%s{\"catalog_id\":", sep);
        jw_string(jw, groups[i].catalog_id);
        jw_printf(jw, ",\"bytes\":%llu,\"files\":%d}", (unsigned long long)groups[i].bytes,
                  groups[i].n_files);
        sep = ",";
    }
    jw_raw(jw, "]");
    free(groups);
}

int pn_local_uninstall_route(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    uint64_t freed = 0;
    int n = 0, status = 500;
    char err[256], out[PN_CAT_ID_MAX + 128];

    if (strcmp(req->method, "POST") != 0 || strcmp(req->path, "/local/models/uninstall") != 0)
        return 0;
    jr_value *body = req->body ? jr_parse(req->body) : NULL;
    const char *id = body ? jr_as_string(jr_object_get(body, "catalog_id")) : NULL;
    char id_copy[PN_CAT_ID_MAX], probe[PN_CAT_ID_MAX + 16];

    /* The id must name store entries exactly: `<id>-<sha12>.` parses back to it. */
    snprintf(probe, sizeof(probe), "%s-000000000000.x", id ? id : "");
    if (!id || !*id || strchr(id, '/') ||
        pn_model_store_entry_id(probe, id_copy, sizeof(id_copy)) != 0 || strcmp(id_copy, id) != 0)
        hs_send_error(cfd, 400, "invalid_request_error", "'catalog_id' is required");
    else if (pn_local_uninstall(ctx, id, &freed, &n, &status, err, sizeof(err)) != 0)
        hs_send_error(cfd, status, status == 409 ? "model_in_use" : "uninstall_failed", err);
    else {
        snprintf(out, sizeof(out), "{\"catalog_id\":\"%s\",\"removed_files\":%d,\"freed\":%llu}",
                 id, n, (unsigned long long)freed);
        hs_send_json(cfd, 200, out);
    }
    jr_free(body);
    return 1;
}
