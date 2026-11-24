/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_model_routes.c
 * @brief  Implementation of the model routes (see local_model_routes.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_model_routes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "../core/load_progress.h"
#include "../serve/model_admin.h"
#include "catalog.h"
#include "curl_runner.h"
#include "hw_report.h"
#include "install_worker.h"
#include "loaded_set.h"
#include "local_caller.h"
#include "local_load.h"
#include "local_people_wire.h"
#include "local_routes.h"
#include "local_unload_rule.h"
#include "model_store.h"
#include "people_store.h"

static void send_status(int cfd, int code, const pn_install_status *st)
{
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_printf(&jw, "{\"catalog_id\":\"%s\",\"state\":\"%s\",\"bytes\":%llu,\"total\":%llu,"
                   "\"error\":", st->model_id, pn_install_state_name(st->state),
              (unsigned long long)st->bytes, (unsigned long long)st->total);
    if (st->error[0])
        jw_string(&jw, st->error);
    else
        jw_raw(&jw, "null");
    jw_raw(&jw, "}");
    hs_send_json(cfd, code, jw.data);
    jw_free(&jw);
}

/* The unit types an install is for: the named profile's, or (no profile_id)
 * every unit type this box has, plus its CPU. -1 for an unknown profile. */
static int install_units(pn_local_ctx *ctx, const pn_cat_model *model, const char *profile_id,
                         unsigned *units)
{
    pn_host_info host;
    pn_hw_report hw;

    if (profile_id) {
        const pn_cat_profile *p = pn_cat_find_profile(model, profile_id);
        *units = p ? pn_cat_unit_type_bit(p->hardware.unit_type) : 0;
        return p ? 0 : -1;
    }
    pn_local_snapshot_hardware(ctx, &host, &hw);
    *units = PN_CAT_UNIT_CPU;
    for (int i = 0; i < hw.n_units; i++)
        *units |= pn_cat_unit_type_bit(hw.units[i].type);
    return 0;
}

/* Already installed: a done row with the needed files' total. */
static void send_installed(int cfd, const pn_cat_model *need)
{
    pn_install_status st;

    memset(&st, 0, sizeof(st));
    snprintf(st.model_id, sizeof(st.model_id), "%s", need->id);
    st.state = PN_INSTALL_DONE;
    for (int i = 0; i < need->n_files; i++)
        st.total += need->files[i].size;
    st.bytes = st.total;
    send_status(cfd, 200, &st);
}

/* Install the files the box (or the named profile) needs: a K80 install of a
 * model with p150-only extras skips them (spec §16). */
static void handle_install(int cfd, pn_local_ctx *ctx, const jr_value *body)
{
    const char *id = jr_as_string(jr_object_get(body, "catalog_id"));
    const char *profile_id = jr_as_string(jr_object_get(body, "profile_id"));
    pn_catalog_manifest *m = pn_catalog_snapshot(ctx->catalog);
    const pn_cat_model *model = m && id ? pn_cat_find_model(m, id) : NULL;
    pn_cat_model *need = calloc(1, sizeof(*need));
    pn_install_status st;
    unsigned units = 0;
    char err[256];
    bool have_units = need && model && install_units(ctx, model, profile_id, &units) == 0;

    if (have_units)
        pn_cat_model_for_units(model, units, need);
    if (!need) {
        hs_send_error(cfd, 500, "server_error", "oom");
    } else if (!model) {
        hs_send_error(cfd, 404, "not_found", m ? "no such catalog model" : "no catalog yet");
    } else if (!have_units) {
        hs_send_error(cfd, 404, "not_found", "no such profile of that catalog model");
    } else if (!pn_curl_path() || !ctx->installer) {
        hs_send_error(cfd, 503, "curl_missing", "installs need curl on the box (not found)");
    } else if (pn_model_store_installed(ctx->store_dir, need)) {
        send_installed(cfd, need);
    } else if (pn_install_worker_submit(ctx->installer, need, m, err, sizeof(err)) != 0) {
        hs_send_error(cfd, 409, "install_busy", err);
    } else {
        pn_install_worker_status(ctx->installer, need->id, &st);
        send_status(cfd, 202, &st);
    }
    free(need);
    free(m);
}

static void handle_install_status(int cfd, pn_local_ctx *ctx, const char *id)
{
    pn_install_status st;

    if (!ctx->installer || pn_install_worker_status(ctx->installer, id, &st) != 0)
        hs_send_error(cfd, 404, "not_found", "no install of that model since the box started");
    else
        send_status(cfd, 200, &st);
}

/* The fields every replica row shares: ids, replica number, load option, and
 * who loaded it (short person; a removed person keeps its id, spec §4.7). */
static void write_entry_head(json_writer *jw, pn_local_ctx *ctx, const pn_loaded_entry *e)
{
    pn_person p;

    jw_printf(jw, "{\"catalog_id\":\"%s\",\"profile_id\":\"%s\",\"name\":\"%s\","
                  "\"replica\":%d,\"ctx_cap\":%d,\"max_sessions\":%d,\"loaded_by\":",
              e->catalog_id, e->profile_id, e->catalog_id, e->replica, e->ctx_cap,
              e->max_sessions > 0 ? e->max_sessions : 1);
    if (ctx->people && pn_people_get(ctx->people, e->loaded_by, &p) == PN_PEOPLE_OK)
        pn_local_write_person_short(jw, &p);
    else
        pn_local_write_person_short_of(jw, NULL, 0, e->loaded_by);
    jw_raw(jw, ",\"units\":[");
}

static void write_entry(json_writer *jw, pn_local_ctx *ctx, const pn_loaded_entry *e)
{
    write_entry_head(jw, ctx, e);
    for (int i = 0; i < e->n_units; i++)
        jw_printf(jw, "%s%d", i ? "," : "", e->units[i]);
    jw_raw(jw, "]}");
}

static void handle_load(int cfd, pn_local_ctx *ctx, const jr_value *body)
{
    const jr_value *ju = jr_object_get(body, "units");
    const pn_local_caller *caller = pn_local_caller_get();
    pn_local_load_request rq = {
        .catalog_id = jr_as_string(jr_object_get(body, "catalog_id")),
        .profile_id = jr_as_string(jr_object_get(body, "profile_id")),
        .ctx_cap = (int)jr_as_number(jr_object_get(body, "ctx_cap"), 0),
        .max_sessions = (int)jr_as_number(jr_object_get(body, "max_sessions"), 0),
        .loaded_by = caller ? caller->person : NULL,
    };
    int status = 500;
    pn_loaded_entry e;
    char err[256];
    json_writer jw;

    if (!rq.catalog_id || !rq.profile_id) {
        hs_send_error(cfd, 400, "invalid_request_error", "catalog_id and profile_id are required");
        return;
    }
    if (rq.ctx_cap < 0 || rq.max_sessions < 0) {
        hs_send_error(cfd, 400, "invalid_request_error", "ctx_cap / max_sessions must be > 0");
        return;
    }
    for (int i = 0; i < jr_array_len(ju) && rq.n_units < PN_UNIT_PLACE_MAX; i++)
        rq.units[rq.n_units++] = (int)jr_as_number(jr_array_get(ju, i), -1);
    if (pn_local_load_catalog(ctx, &rq, &e, &status, err, sizeof(err)) != 0) {
        hs_send_error(cfd, status, "load_refused", err);
        return;
    }
    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    write_entry(&jw, ctx, &e);
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
}

/* A p150's bytes during a load come from its DRAM allocator (the TT upload
 * does not report progress itself); K80 dies report through the upload. */
static void fill_device_bytes(pn_load_progress *p)
{
    pn_hw_dev_mem m;

    if (p->phase != PN_LOAD_PHASE_LOADING || pn_hw_family_detect() != PN_HW_FAMILY_P150)
        return;
    for (int i = 0; i < p->n_units; i++) {
        if (pn_hw_report_read_device(PN_HW_FAMILY_P150, p->units[i].die, &m) != 0 ||
            m.used_self <= p->units[i].bytes)
            continue;
        p->units[i].bytes = m.used_self;
        if (p->units[i].total < m.used_self)
            p->units[i].total = m.used_self;
    }
}

/* ,"progress":{phase, bytes, total, layer, n_layers, units:[{id, bytes, total}]}
 * while a load / unload of this replica runs, else ,"progress":null (H2n). */
static void write_progress(json_writer *jw, const pn_load_progress *p)
{
    uint64_t bytes = 0, total = 0;

    if (!p) {
        jw_raw(jw, ",\"progress\":null");
        return;
    }
    for (int i = 0; i < p->n_units; i++) {
        bytes += p->units[i].bytes;
        total += p->units[i].total;
    }
    jw_printf(jw, ",\"progress\":{\"phase\":\"%s\",\"bytes\":%llu,\"total\":%llu,\"layer\":%d,"
                  "\"n_layers\":%d,\"units\":[",
              p->phase == PN_LOAD_PHASE_UNLOADING ? "unloading" : "loading",
              (unsigned long long)bytes, (unsigned long long)total, p->layer, p->n_layers);
    for (int i = 0; i < p->n_units; i++)
        jw_printf(jw, "%s{\"id\":%d,\"bytes\":%llu,\"total\":%llu}", i ? "," : "",
                  p->units[i].die, (unsigned long long)p->units[i].bytes,
                  (unsigned long long)p->units[i].total);
    jw_raw(jw, "]}");
}

/* The progress of this replica's load / unload, or NULL. */
static const pn_load_progress *progress_of(const pn_loaded_entry *e, pn_load_progress *buf)
{
    if (pn_load_progress_find(e->catalog_id, e->replica, buf) != 0)
        return NULL;
    fill_device_bytes(buf);
    return buf;
}

/* One remembered model as it is NOW: loaded (units = the dies it really holds,
 * placement "cpu" when it holds none), unloading, loading, or not loaded, with
 * why (H2f / H2g), and the live progress of a load / unload (H2n). */
static void write_loaded_status(json_writer *jw, pn_local_ctx *ctx, const pn_loaded_entry *e,
                                const mr_info *info, int n_info)
{
    pn_load_progress buf;
    const pn_load_progress *prog = progress_of(e, &buf);
    const mr_info *row = pn_local_registry_row_of(e, info, n_info);
    char reason[PN_RELOAD_REASON_MAX];

    write_entry_head(jw, ctx, e);
    for (int i = 0; row && i < row->n_dies; i++)
        jw_printf(jw, "%s%d", i ? "," : "", row->dies[i]);
    if (row) {
        jw_printf(jw, "],\"state\":\"%s\",\"placement\":\"%s\",\"reason\":null,"
                      "\"sessions_used\":%d,\"slots_held\":%d,\"seat_free\":%s",
                  prog && prog->phase == PN_LOAD_PHASE_UNLOADING ? "unloading" : "loaded",
                  row->n_dies > 0 ? e->hardware.unit_type : "cpu", row->sessions_used,
                  row->slots_held, row->seat_free ? "true" : "false");
        write_progress(jw, prog);
        jw_raw(jw, "}");
        return;
    }
    pn_reload_state st = pn_local_reload_state(e->catalog_id, e->replica, reason, sizeof(reason));
    /* An unload hides the registry row before it frees the model: still "unloading". */
    const char *state = prog ? (prog->phase == PN_LOAD_PHASE_UNLOADING ? "unloading" : "loading")
                        : st == PN_RELOAD_PENDING ? "loading" : "not_loaded";
    jw_printf(jw, "],\"state\":\"%s\",\"placement\":null,\"reason\":", state);
    if (reason[0] && !prog)
        jw_string(jw, reason);
    else
        jw_raw(jw, "null");
    write_progress(jw, prog);
    jw_raw(jw, "}");
}

/* The load running now when the loaded set does not list it yet: state
 * "loading" on the units it was placed on, with its progress. */
static void write_in_flight(json_writer *jw, pn_local_ctx *ctx, const pn_loaded_entry *list,
                            int n, bool first)
{
    pn_loaded_entry e;
    pn_load_progress buf;

    if (pn_local_load_in_flight(&e) != 0)
        return;
    for (int i = 0; i < n; i++)
        if (strcmp(list[i].catalog_id, e.catalog_id) == 0 && list[i].replica == e.replica)
            return;
    if (!first)
        jw_raw(jw, ",");
    write_entry_head(jw, ctx, &e);
    for (int i = 0; i < e.n_units; i++)
        jw_printf(jw, "%s%d", i ? "," : "", e.units[i]);
    jw_raw(jw, "],\"state\":\"loading\",\"placement\":null,\"reason\":null");
    write_progress(jw, progress_of(&e, &buf));
    jw_raw(jw, "}");
}

void pn_local_write_loaded(json_writer *jw, pn_local_ctx *ctx)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    mr_info info[MR_MAX_MODELS];
    int n = ctx->loaded ? pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) : 0;
    int n_info = ctx->reg ? mr_snapshot(ctx->reg, info, MR_MAX_MODELS) : 0;

    jw_raw(jw, "{\"models\":[");
    for (int i = 0; i < n; i++) {
        if (i)
            jw_raw(jw, ",");
        write_loaded_status(jw, ctx, &list[i], info, n_info);
    }
    write_in_flight(jw, ctx, list, n, n == 0);
    jw_raw(jw, "]}");
}

static void handle_loaded(int cfd, pn_local_ctx *ctx)
{
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    pn_local_write_loaded(&jw, ctx);
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
}

/* G3 + L4: {"name"} unloads every replica of the model, {"name","replica":N}
 * one of them; the remembered set follows. A non-admin may unload only replicas
 * they loaded that hold no session (spec §4.7: 403 unload_refused with why). */
static void handle_admin_unload(int cfd, pn_local_ctx *ctx, const jr_value *body)
{
    const char *name = jr_as_string(jr_object_get(body, "name"));
    const pn_local_caller *caller = pn_local_caller_get();
    const char *as_person = caller && !caller->admin ? caller->person : NULL;
    int replica = (int)jr_as_number(jr_object_get(body, "replica"), 0), n = 0, status = 500;
    char err[PN_RELOAD_REASON_MAX];
    json_writer jw;

    if (!name || !*name || replica < 0) {
        hs_send_error(cfd, 400, "invalid_request_error", "'name' is required (replica >= 1)");
        return;
    }
    if (pn_local_unload_as(ctx, name, replica, as_person, &n, &status, err, sizeof(err)) != 0) {
        hs_send_error(cfd, status, status == 403 ? "unload_refused" : "unload_failed", err);
        return;
    }
    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_raw(&jw, "{\"unloaded\":");
    jw_string(&jw, name);
    jw_printf(&jw, ",\"replicas\":%d}", n);
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
}

int pn_local_model_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const char *p = req->path;
    const int post = strcmp(req->method, "POST") == 0, get = strcmp(req->method, "GET") == 0;

    if (get && strncmp(p, "/local/models/install/", 22) == 0 && p[22]) {
        handle_install_status(cfd, ctx, p + 22);
        return 1;
    }
    if (get && strcmp(p, "/local/models/loaded") == 0) {
        handle_loaded(cfd, ctx);
        return 1;
    }
    if (!post || (strcmp(p, "/local/models/install") != 0 && strcmp(p, "/local/models/load") != 0 &&
                  strcmp(p, "/admin/models/unload") != 0))
        return 0;
    jr_value *body = req->body ? jr_parse(req->body) : NULL;
    if (!body || jr_kind(body) != JR_OBJECT)
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object body is required");
    else if (strcmp(p, "/admin/models/unload") == 0)
        handle_admin_unload(cfd, ctx, body);
    else if (strcmp(p, "/local/models/install") == 0)
        handle_install(cfd, ctx, body);
    else
        handle_load(cfd, ctx, body);
    jr_free(body);
    return 1;
}
