/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_load.c
 * @brief  Implementation of catalog loads and the boot reload (see local_load.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_load.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/die_manager.h"
#include "../core/load_progress.h"
#include "../serve/model_admin.h"
#include "../serve/serve_registry.h"
#include "catalog.h"
#include "catalog_match.h"
#include "hw_mem_poller.h"
#include "load_options.h"
#include "local_routes.h"
#include "local_unload_rule.h"
#include "model_store.h"
#include "people_store.h"

static pthread_mutex_t g_load_lock = PTHREAD_MUTEX_INITIALIZER;

/* The load running now (loads are one at a time), for GET /local/models/loaded
 * before the model is in the loaded set. */
static pthread_mutex_t g_inflight_lock = PTHREAD_MUTEX_INITIALIZER;
static pn_loaded_entry g_inflight;
static bool            g_inflight_on;

static void inflight_set(const pn_loaded_entry *e)
{
    pthread_mutex_lock(&g_inflight_lock);
    if (e)
        g_inflight = *e;
    g_inflight_on = e != NULL;
    pthread_mutex_unlock(&g_inflight_lock);
}

int pn_local_load_in_flight(pn_loaded_entry *out)
{
    pthread_mutex_lock(&g_inflight_lock);
    int on = g_inflight_on;
    if (on)
        *out = g_inflight;
    pthread_mutex_unlock(&g_inflight_lock);
    return on ? 0 : -1;
}

void pn_local_loads_lock(void)
{
    pthread_mutex_lock(&g_load_lock);
}

void pn_local_loads_unlock(void)
{
    pthread_mutex_unlock(&g_load_lock);
}

static int refuse(int *status, int code, char *err, size_t errlen, const char *msg)
{
    if (status)
        *status = code;
    snprintf(err, errlen, "%s", msg);
    return -1;
}

static const pn_hw_unit *find_unit(const pn_hw_report *hw, int id)
{
    for (int i = 0; i < hw->n_units; i++)
        if (hw->units[i].id == id)
            return &hw->units[i];
    return NULL;
}

/* The caller picked units: same type, free, right count, card rule. */
static int check_units(const pn_hw_report *hw, const pn_unit_request *r, const int *units,
                       int n, char *err, size_t errlen)
{
    if (n != r->count) {
        snprintf(err, errlen, "this profile needs %d %s unit(s), %d given", r->count,
                 r->unit_type, n);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        const pn_hw_unit *u = find_unit(hw, units[i]);
        if (!u || strcmp(u->type, r->unit_type) != 0 || !pn_hw_unit_available(u) ||
            u->mem_free < r->min_mem_per_unit) {
            snprintf(err, errlen, "unit %d is not a free %s with enough memory", units[i],
                     r->unit_type);
            return -1;
        }
        if (r->same_card && u->card != find_unit(hw, units[0])->card) {
            snprintf(err, errlen, "this profile needs its units on one card");
            return -1;
        }
        int on_card = 0;
        for (int j = 0; j < n; j++)
            on_card += find_unit(hw, units[j]) && find_unit(hw, units[j])->card == u->card;
        if (r->per_card > 0 && !r->same_card && on_card != r->per_card) {
            snprintf(err, errlen, "this profile needs %d units on each card it uses",
                     r->per_card);
            return -1;
        }
    }
    return 0;
}

/* Copy the profile's settings into the entry. */
static void fill_entry(const pn_cat_model *m, const pn_cat_profile *p, const char *path,
                       pn_loaded_entry *out)
{
    memset(out, 0, sizeof(*out));
    snprintf(out->catalog_id, sizeof(out->catalog_id), "%s", m->id);
    snprintf(out->profile_id, sizeof(out->profile_id), "%s", p->id);
    snprintf(out->path, sizeof(out->path), "%s", path);
    out->hardware = p->hardware;
    out->ctx_cap = p->ctx_cap;
    out->max_sessions = p->max_sessions;
    out->max_concurrent = p->max_concurrent;
    out->reasoning_budget = p->reasoning_budget;
    out->reasoning_reserve = p->reasoning_reserve;
    out->mtp = p->mtp;
    out->mtp_drafts = p->mtp_drafts;
    if (p->options.present) {
        snprintf(out->kv_type, sizeof(out->kv_type), "%s", p->options.kv_type);
        snprintf(out->kv_long, sizeof(out->kv_long), "%s", p->options.kv_long);
    }
    out->sampling = p->sampling;
}

/* The load option (ctx x users) and the placement it needs: the request's
 * choice must fit the units free now; the units then need its memory. */
static int choose_and_place(const pn_cat_profile *p, const pn_local_load_request *rq,
                            const pn_hw_report *hw, uint64_t host_ram, pn_loaded_entry *out,
                            char *why, size_t cap)
{
    pn_load_option opt;

    if (pn_load_options_choose(p, hw, rq->ctx_cap, rq->max_sessions, &opt, why, cap) != 0)
        return -1;
    out->ctx_cap = opt.ctx_cap;
    out->max_sessions = p->options.present || p->max_sessions > 0 ? opt.max_sessions : 0;
    out->hardware = pn_load_option_request(p, &opt, hw);
    if (!pn_cat_request_fits(&out->hardware, p->min_host_ram, hw, host_ram, why, cap))
        return -1;
    if (rq->n_units > 0) {
        if (check_units(hw, &out->hardware, rq->units, rq->n_units, why, cap) != 0)
            return -1;
        memcpy(out->units, rq->units, (size_t)rq->n_units * sizeof(int));
        out->n_units = strcmp(p->hardware.unit_type, "cpu") == 0 ? 0 : rq->n_units;
        return 0;
    }
    return pn_unit_place(hw, &out->hardware, out->units, &out->n_units, why, cap);
}

int pn_local_resolve_load(pn_local_ctx *ctx, const pn_local_load_request *rq,
                          pn_loaded_entry *out, int *status, char *err, size_t errlen)
{
    pn_catalog_manifest *m = pn_catalog_snapshot(ctx->catalog);
    const pn_cat_model *model = m ? pn_cat_find_model(m, rq->catalog_id) : NULL;
    const pn_cat_profile *p = model ? pn_cat_find_profile(model, rq->profile_id) : NULL;
    pn_cat_model *need = calloc(1, sizeof(*need));
    char path[1024], why[200];
    pn_host_info host;
    pn_hw_report hw;
    int rc = -1;

    if (!p || !need) {
        refuse(status, need ? 404 : 500, err, errlen,
               !need ? "out of memory" : m ? "no such catalog model / profile" : "no catalog yet");
        goto done;
    }
    /* Only the files this profile's unit type needs (spec §16), and ITS weights file: a
     * model may ship one per unit type (Qwen3.6: the K80 GGUF, the p150 metadata GGUF). */
    pn_cat_model_for_units(model, pn_cat_unit_type_bit(p->hardware.unit_type), need);
    if (!pn_model_store_installed(ctx->store_dir, need) ||
        pn_model_store_weights_path(ctx->store_dir, need, path, sizeof(path)) != 0) {
        refuse(status, 409, err, errlen, "not installed: POST /local/models/install first");
        goto done;
    }
    if (pn_model_store_link_companions(ctx->store_dir, need, why, sizeof(why)) != 0) {
        refuse(status, 507, err, errlen, why);
        goto done;
    }
    pn_local_snapshot_hardware_fresh(ctx, &host, &hw);
    fill_entry(model, p, path, out);
    if (rq->loaded_by)
        snprintf(out->loaded_by, sizeof(out->loaded_by), "%s", rq->loaded_by);
    out->replica = ctx->loaded ? pn_loaded_set_next_replica(ctx->loaded, model->id) : 1;
    if (choose_and_place(p, rq, &hw, host.ram_total, out, why, sizeof(why)) != 0) {
        refuse(status, 409, err, errlen, why);
        goto done;
    }
    rc = 0;
done:
    free(need);
    free(m);
    return rc;
}

/* The registry's row for `name` on first_die (< 0: any row of that name), or -1. */
static int registry_row(mr_registry *reg, const char *name, int first_die, mr_info *out)
{
    mr_info info[MR_MAX_MODELS];
    int n = reg ? mr_snapshot(reg, info, MR_MAX_MODELS) : 0;

    for (int i = 0; i < n; i++) {
        if (strcmp(info[i].name, name) == 0 &&
            (first_die < 0 || (info[i].n_dies > 0 && info[i].dies[0] == first_die))) {
            *out = info[i];
            return 0;
        }
    }
    return -1;
}

/* An accelerator profile that ended up on the CPU forward (the upload failed, or
 * this build has no forward for it on that card) is not what the profile promised:
 * unload it and refuse, never serve it as if it were on the card (H2g). */
static int verify_placement(pn_local_ctx *ctx, const pn_loaded_entry *e, char *err,
                            size_t errlen)
{
    mr_info row;
    char why[160];

    int first = e->n_units > 0 ? e->units[0] : -1;

    if (strcmp(e->hardware.unit_type, "cpu") == 0 ||
        registry_row(ctx->reg, e->catalog_id, first, &row) != 0 || row.n_dies == e->n_units)
        return 0;
    if (mr_unload_replica(ctx->reg, e->catalog_id, row.n_dies > 0 ? row.dies[0] : -1, why,
                          sizeof(why)) != 0)
        fprintf(stderr, "local: could not unload %s after a wrong placement: %s\n", e->catalog_id,
                why);
    snprintf(err, errlen, "this build could not run %s on the %s (it would run on the CPU); "
             "not loaded", e->catalog_id, e->hardware.unit_type);
    return -1;
}

/* ma_load_model + remember. g_load_lock held. */
static int load_entry_locked(pn_local_ctx *ctx, const pn_loaded_entry *e, char *err,
                             size_t errlen)
{
    ma_load_request lr;
    char tt_spec[8];

    memset(&lr, 0, sizeof(lr));
    lr.path = e->path;
    lr.name = e->catalog_id;
    memcpy(lr.dies, e->units, (size_t)e->n_units * sizeof(int));
    lr.n_dies = e->n_units;
    lr.opts.ctx_cap = e->ctx_cap;
    /* `cpu` is the host CPU forward in every build: never auto-placed on a card. */
    lr.opts.cpu_only = strcmp(e->hardware.unit_type, "cpu") == 0;
    lr.opts.kv_format = pn_local_kv_format_of(e->kv_type);
    lr.reasoning_budget = e->reasoning_budget;
    lr.reasoning_reserve = e->reasoning_reserve;
    lr.max_sessions = e->max_sessions;
    lr.max_concurrent = e->max_concurrent;
#ifdef USE_CUDA
    lr.mtp = e->mtp;
#endif
    /* A p150 profile's KV type and MTP drafts, for this load only (load_options.h). */
    lr.tt_kv_type = pn_local_tt_kv_type_of(e->hardware.unit_type, e->kv_type);
    lr.tt_kv_long = pn_local_tt_kv_type_of(e->hardware.unit_type, e->kv_long);
    lr.tt_spec = pn_local_tt_spec_of(e->hardware.unit_type, e->mtp, e->mtp_drafts, tt_spec,
                                     sizeof(tt_spec));
    lr.sampling = e->sampling;
    lr.replica = e->replica;   /* the registry numbers the copy as the loaded list does */
    fprintf(stderr, "local: loading %s replica %d (profile %s, %d tokens x %d user(s), KV %s%s%s) "
            "on %d unit(s)%s\n", e->catalog_id, e->replica, e->profile_id, e->ctx_cap,
            e->max_sessions > 0 ? e->max_sessions : 1, e->kv_type[0] ? e->kv_type : "default",
            lr.tt_spec ? ", PN_TT_SPEC=" : "", lr.tt_spec ? lr.tt_spec : "",
            e->n_units, lr.opts.cpu_only ? " (cpu)" : "");
    inflight_set(e);
    pn_load_progress_begin(e->catalog_id, e->replica, PN_LOAD_PHASE_LOADING, e->units, NULL,
                           e->n_units);
    int rc = ma_load_model(ctx->reg, &lr, err, errlen);
    pn_load_progress_end();
    inflight_set(NULL);
    pn_hw_mem_poller_kick();
    if (rc != 0 || verify_placement(ctx, e, err, errlen) != 0)
        return -1;
    if (ctx->loaded && pn_loaded_set_add(ctx->loaded, e) != 0)
        fprintf(stderr, "local: loaded %s but could not save the loaded set\n", e->catalog_id);
    pn_local_reload_note(e->catalog_id, e->replica, PN_RELOAD_LOADED, NULL);
    return 0;
}

int pn_local_load_catalog(pn_local_ctx *ctx, const pn_local_load_request *rq,
                          pn_loaded_entry *out, int *status, char *err, size_t errlen)
{
    int rc = -1;

    /* Placement and load under one lock: two loads never pick the same units. */
    pthread_mutex_lock(&g_load_lock);
    if (pn_local_resolve_load(ctx, rq, out, status, err, errlen) == 0) {
        rc = load_entry_locked(ctx, out, err, errlen);
        if (rc != 0 && status)
            *status = strstr(err, "already") ? 409 : 507;
    }
    pthread_mutex_unlock(&g_load_lock);
    return rc;
}

/* The first die of the registry's first row named `name`; -2 when none (-1: a
 * row on the CPU). */
static int first_row_die(mr_registry *reg, const char *name)
{
    mr_info row;

    if (registry_row(reg, name, -1, &row) != 0)
        return -2;
    return row.n_dies > 0 ? row.dies[0] : -1;
}

/* mr_unload_replica with a progress slot: the bars drain from what each unit
 * holds now (the die manager's bindings) to nothing. */
static int unload_with_progress(pn_local_ctx *ctx, const pn_loaded_entry *e, int first,
                                char *why, size_t cap)
{
    uint64_t held[PN_UNIT_PLACE_MAX] = { 0 };
    pn_dm_die d;

    for (int u = 0; u < e->n_units; u++)
        if (pn_dm_get(e->units[u], &d) == 0)
            held[u] = d.vram_bound;
    pn_load_progress_begin(e->catalog_id, e->replica, PN_LOAD_PHASE_UNLOADING, e->units, held,
                           e->n_units);
    for (int u = 0; u < e->n_units; u++)
        pn_load_progress_set_bytes(e->units[u], held[u]);
    int rc = mr_unload_replica(ctx->reg, e->catalog_id, first, why, cap);
    pn_load_progress_end();
    pn_hw_mem_poller_kick();
    return rc;
}

/* Unload the remembered replica `replica` (its registry row, if loaded). */
static int unload_one_replica(pn_local_ctx *ctx, const char *id, int replica, int *n_out,
                              int *status, char *err, size_t errlen)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    mr_info info[MR_MAX_MODELS];
    int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
    int n_info = ctx->reg ? mr_snapshot(ctx->reg, info, MR_MAX_MODELS) : 0;
    char why[160];

    for (int i = 0; i < n; i++) {
        if (strcmp(list[i].catalog_id, id) != 0 || list[i].replica != replica)
            continue;
        int first = list[i].n_units > 0 ? list[i].units[0] : -1;
        /* Only this replica's own row: a CPU replica that is not loaded must not
         * take down another copy of the name (the unload rule checked this one). */
        if (pn_local_registry_row_of(&list[i], info, n_info)) {
            int rc = unload_with_progress(ctx, &list[i], first, why, sizeof(why));
            if (rc != 0)
                return refuse(status, strstr(why, "busy") ? 409 : 500, err, errlen, why);
            (*n_out)++;
        }
        pn_loaded_set_remove(ctx->loaded, id, replica);
        return 0;
    }
    snprintf(why, sizeof(why), "%s has no replica %d", id, replica);
    return refuse(status, 404, err, errlen, why);
}

int pn_local_unload(pn_local_ctx *ctx, const char *catalog_id, int replica, int *n_out,
                    int *status, char *err, size_t errlen)
{
    return pn_local_unload_as(ctx, catalog_id, replica, NULL, n_out, status, err, errlen);
}

int pn_local_unload_as(pn_local_ctx *ctx, const char *catalog_id, int replica,
                       const char *person, int *n_out, int *status, char *err, size_t errlen)
{
    int rc = 0, first;
    char why[PN_RELOAD_REASON_MAX];

    *n_out = 0;
    pthread_mutex_lock(&g_load_lock);
    if (person && pn_local_unload_refusal(ctx, catalog_id, replica, person, why,
                                          sizeof(why)) != 0) {
        pthread_mutex_unlock(&g_load_lock);
        fprintf(stderr, "local: unload of %s refused for %s: %s\n", catalog_id, person, why);
        return refuse(status, 403, err, errlen, why);
    }
    if (replica > 0) {
        rc = unload_one_replica(ctx, catalog_id, replica, n_out, status, err, errlen);
    } else {
        /* Every remembered replica (each leaves the set once it is down), then
         * anything else under the name (a model loaded by hand). */
        pn_loaded_entry list[PN_LOADED_MAX];
        int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
        bool known = false;
        for (int i = 0; i < n && rc == 0; i++) {
            if (strcmp(list[i].catalog_id, catalog_id) != 0)
                continue;
            known = true;
            rc = unload_one_replica(ctx, catalog_id, list[i].replica, n_out, status, err,
                                    errlen);
        }
        while (rc == 0 && (first = first_row_die(ctx->reg, catalog_id)) != -2) {
            if (mr_unload_replica(ctx->reg, catalog_id, first, why, sizeof(why)) != 0)
                rc = refuse(status, strstr(why, "busy") ? 409 : 500, err, errlen, why);
            else
                (*n_out)++;
        }
        if (rc == 0 && *n_out == 0 && !known)
            rc = refuse(status, 404, err, errlen, "no such model");
    }
    if (first_row_die(ctx->reg, catalog_id) == -2)
        serve_registry_remove(catalog_id);
    pthread_mutex_unlock(&g_load_lock);
    pn_hw_mem_poller_kick();
    if (*n_out)
        fprintf(stderr, "local: unloaded %d replica(s) of %s\n", *n_out, catalog_id);
    return rc;
}

int pn_local_loads_claim_unowned(pn_local_ctx *ctx)
{
    pn_person owner;
    int claimed = 0;

    if (!ctx->loaded || !ctx->people || pn_people_admin_count(ctx->people) == 0 ||
        pn_people_owner_or_create(ctx->people, NULL, NULL, &owner, NULL) != PN_PEOPLE_OK)
        return 0;
    claimed = pn_loaded_set_claim_unowned(ctx->loaded, owner.id);
    if (claimed > 0)
        fprintf(stderr, "local: %d remembered model(s) from before people belong to %s\n",
                claimed, owner.id);
    return claimed < 0 ? -1 : 0;
}

/* Boot reload outcomes, per remembered model (GET /local/models/loaded). */
typedef struct {
    char              catalog_id[PN_CAT_ID_MAX];
    int               replica;
    pn_reload_state   state;
    char              reason[PN_RELOAD_REASON_MAX];
} reload_row;

static reload_row      g_reload[PN_LOADED_MAX];
static int             g_n_reload;
static pthread_mutex_t g_reload_lock = PTHREAD_MUTEX_INITIALIZER;

void pn_local_reload_note(const char *catalog_id, int replica, pn_reload_state state,
                          const char *reason)
{
    pthread_mutex_lock(&g_reload_lock);
    int i = 0;
    while (i < g_n_reload && (strcmp(g_reload[i].catalog_id, catalog_id) != 0 ||
                              g_reload[i].replica != replica))
        i++;
    if (i == g_n_reload && g_n_reload < PN_LOADED_MAX) {
        g_reload[g_n_reload].replica = replica;
        snprintf(g_reload[g_n_reload++].catalog_id, PN_CAT_ID_MAX, "%s", catalog_id);
    }
    if (i < g_n_reload) {
        g_reload[i].state = state;
        snprintf(g_reload[i].reason, sizeof(g_reload[i].reason), "%s", reason ? reason : "");
    }
    pthread_mutex_unlock(&g_reload_lock);
}

pn_reload_state pn_local_reload_state(const char *catalog_id, int replica, char *reason,
                                      size_t cap)
{
    pn_reload_state st = PN_RELOAD_NONE;

    if (reason && cap)
        reason[0] = '\0';
    pthread_mutex_lock(&g_reload_lock);
    for (int i = 0; i < g_n_reload; i++) {
        if (strcmp(g_reload[i].catalog_id, catalog_id) != 0 || g_reload[i].replica != replica)
            continue;
        st = g_reload[i].state;
        if (reason && cap)
            snprintf(reason, cap, "%s", g_reload[i].reason);
    }
    pthread_mutex_unlock(&g_reload_lock);
    return st;
}

/* Re-check a remembered entry against the hardware found now (H2f): the box must
 * still have units of the profile's type (a `cpu` profile needs a box with no
 * accelerator, as the catalog match says), then its remembered units if still
 * free, else free ones placed again. */
static int recheck_and_place(pn_local_ctx *ctx, pn_loaded_entry *e, char *why, size_t cap)
{
    pn_host_info host;
    pn_hw_report hw;
    char reason[160];

    pn_local_snapshot_hardware_fresh(ctx, &host, &hw);
    if (!pn_cat_request_fits(&e->hardware, 0, &hw, host.ram_total, reason, sizeof(reason))) {
        snprintf(why, cap, "profile %s no longer fits this box: %s", e->profile_id, reason);
        return -1;
    }
    if (strcmp(e->hardware.unit_type, "cpu") == 0) {
        e->n_units = 0;
        return 0;
    }
    if (check_units(&hw, &e->hardware, e->units, e->n_units, why, cap) == 0)
        return 0;
    return pn_unit_place(&hw, &e->hardware, e->units, &e->n_units, why, cap);
}

void pn_local_reload_now(pn_local_ctx *ctx)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
    char why[PN_RELOAD_REASON_MAX];

    for (int i = 0; i < n; i++)
        pn_local_reload_note(list[i].catalog_id, list[i].replica, PN_RELOAD_PENDING, NULL);
    for (int i = 0; i < n; i++) {
        pthread_mutex_lock(&g_load_lock);
        if (recheck_and_place(ctx, &list[i], why, sizeof(why)) != 0) {
            fprintf(stderr, "local: reload of %s skipped: %s\n", list[i].catalog_id, why);
            pn_local_reload_note(list[i].catalog_id, list[i].replica, PN_RELOAD_SKIPPED, why);
        } else if (load_entry_locked(ctx, &list[i], why, sizeof(why)) != 0) {
            fprintf(stderr, "local: reload of %s failed: %s\n", list[i].catalog_id, why);
            pn_local_reload_note(list[i].catalog_id, list[i].replica, PN_RELOAD_FAILED, why);
        }
        pthread_mutex_unlock(&g_load_lock);
    }
    if (n)
        fprintf(stderr, "local: reload of the remembered models finished\n");
}

static void *reload_main(void *arg)
{
    pn_local_reload_now(arg);
    return NULL;
}

void pn_local_reload_async(pn_local_ctx *ctx)
{
    pthread_t t;

    if (!ctx->loaded || pthread_create(&t, NULL, reload_main, ctx) != 0)
        return;
    pthread_detach(t);
}
