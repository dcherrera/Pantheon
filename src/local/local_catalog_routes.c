/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_catalog_routes.c
 * @brief  Implementation of the catalog routes (see local_catalog_routes.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_catalog_routes.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/json_writer.h"
#include "catalog.h"
#include "catalog_match.h"
#include "curl_runner.h"
#include "load_options.h"
#include "local_routes.h"
#include "local_uninstall.h"
#include "model_store.h"

static void write_sampling(json_writer *jw, const pn_model_sampling *s)
{
    const char *sep = "";

    jw_raw(jw, "{");
    if (s->has_temperature) { jw_printf(jw, "\"temperature\":%g", s->temperature); sep = ","; }
    if (s->has_top_p) { jw_printf(jw, "%s\"top_p\":%g", sep, s->top_p); sep = ","; }
    if (s->has_top_k) { jw_printf(jw, "%s\"top_k\":%d", sep, s->top_k); sep = ","; }
    if (s->has_min_p) { jw_printf(jw, "%s\"min_p\":%g", sep, s->min_p); sep = ","; }
    if (s->has_presence_penalty)
        jw_printf(jw, "%s\"presence_penalty\":%g", sep, s->presence_penalty);
    jw_raw(jw, "}");
}

static void write_profile(json_writer *jw, const pn_cat_profile *p)
{
    const pn_cat_measured *m = &p->measured;

    jw_printf(jw, "{\"id\":\"%s\",\"rank\":%d,\"hardware\":{\"unit_type\":\"%s\",\"count\":%d,"
                  "\"same_card\":%s,\"per_card\":%d,\"min_mem_per_unit\":%llu,"
                  "\"min_host_ram\":%llu},",
              p->id, p->rank, p->hardware.unit_type, p->hardware.count,
              p->hardware.same_card ? "true" : "false", p->hardware.per_card,
              (unsigned long long)p->hardware.min_mem_per_unit,
              (unsigned long long)p->min_host_ram);
    jw_printf(jw, "\"load\":{\"ctx_cap\":%d,\"max_sessions\":%d,\"max_concurrent\":%d,"
                  "\"reasoning_budget\":%d,",
              p->ctx_cap, p->max_sessions, p->max_concurrent, p->reasoning_budget);
    if (p->reasoning_reserve > 0)   /* absent = the serve default */
        jw_printf(jw, "\"reasoning_reserve\":%d,", p->reasoning_reserve);
    jw_printf(jw, "\"mtp\":%s},\"sampling\":", p->mtp ? "true" : "false");
    write_sampling(jw, &p->sampling);
    jw_printf(jw, ",\"measured\":{\"decode_tok_s\":{\"8k\":%g,\"14k\":%g,\"32k\":%g},"
                  "\"ttft\":{\"prompt_tokens\":%d,\"seconds\":%g},\"context_window\":%d}}",
              m->decode_tok_s_8k, m->decode_tok_s_14k, m->decode_tok_s_32k,
              m->ttft_prompt_tokens, m->ttft_s, m->context_window);
}

/* L3: "options": the context x users choices that fit the units free now,
 * the recommended one flagged. A profile without a memory model offers its
 * fixed settings only. */
static void write_options(json_writer *jw, const pn_cat_profile *p, const pn_hw_report *hw)
{
    pn_load_option opts[PN_LOAD_OPTIONS_MAX];
    const pn_cat_load_options *o = &p->options;
    int n = pn_load_options_fit(p, hw, opts, PN_LOAD_OPTIONS_MAX);
    const char *sep = "";

    if (o->present)
        jw_printf(jw, "{\"kv_type\":\"%s\",\"ctx_min\":%d,\"ctx_max\":%d,\"sessions_max\":%d,",
                  o->kv_type, o->ctx_min, o->ctx_max, o->sessions_max);
    else
        jw_printf(jw, "{\"kv_type\":null,\"ctx_min\":%d,\"ctx_max\":%d,\"sessions_max\":%d,",
                  p->ctx_cap, p->ctx_cap, p->max_sessions > 0 ? p->max_sessions : 1);
    jw_raw(jw, "\"choices\":[");
    for (int i = 0; i < n; i++) {
        if (!opts[i].fits)
            continue;
        jw_printf(jw, "%s{\"ctx_cap\":%d,\"max_sessions\":%d,\"need_per_unit\":%llu,"
                      "\"recommended\":%s}", sep, opts[i].ctx_cap, opts[i].max_sessions,
                  (unsigned long long)opts[i].need_per_unit,
                  opts[i].recommended ? "true" : "false");
        sep = ",";
    }
    jw_raw(jw, "]}");
}

/* `need` = the model's files for this row's unit type: "size" is what its install
 * downloads (a p150 row of Qwen3.6 is the metadata GGUF + the .fttw files, not the
 * K80 GGUF). */
static void write_model(json_writer *jw, const pn_cat_model *m, const pn_cat_model *need,
                        bool installed)
{
    uint64_t size = 0;

    for (int i = 0; i < need->n_files; i++)
        size += need->files[i].size;

    jw_printf(jw, "{\"id\":\"%s\",\"name\":", m->id);
    jw_string(jw, m->name);
    jw_raw(jw, ",\"description\":");
    jw_string(jw, m->description);
    jw_raw(jw, ",\"licence\":{\"id\":");
    jw_string(jw, m->licence_id);
    jw_raw(jw, ",\"url\":");
    jw_string(jw, m->licence_url);
    jw_raw(jw, "},\"capabilities\":[");
    for (int i = 0; i < m->n_capabilities; i++)
        jw_printf(jw, "%s\"%s\"", i ? "," : "", m->capabilities[i]);
    jw_printf(jw, "],\"size\":%llu,\"installed\":%s}", (unsigned long long)size,
              installed ? "true" : "false");
}

/* Whether the files this profile's unit type needs are in the store. */
static bool profile_installed(const pn_local_ctx *ctx, const pn_cat_match *e,
                              pn_cat_model *scratch)
{
    unsigned units = pn_cat_unit_type_bit(e->profile->hardware.unit_type);

    pn_cat_model_for_units(e->model, units, scratch);
    return pn_model_store_installed(ctx->store_dir, scratch);
}

static void handle_catalog(int cfd, pn_local_ctx *ctx)
{
    pn_catalog_manifest *m = pn_catalog_snapshot(ctx->catalog);
    pn_host_info host;
    pn_hw_report hw;
    json_writer jw;

    pn_cat_match *entries = calloc(PN_CAT_MAX_ENTRIES, sizeof(*entries));
    pn_cat_model *scratch = calloc(1, sizeof(*scratch));
    if (!entries || !scratch || !jw_init(&jw)) {
        free(entries);
        free(scratch);
        free(m);
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    pn_local_snapshot_hardware_fresh(ctx, &host, &hw);
    int n = m ? pn_cat_match_profiles(m, &hw, host.ram_total, entries, PN_CAT_MAX_ENTRIES) : 0;
    if (m)
        jw_printf(&jw, "{\"version\":%lld,\"entries\":[", m->version);
    else
        jw_raw(&jw, "{\"version\":null,\"entries\":[");
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, i ? ",{\"model\":" : "{\"model\":");
        bool installed = profile_installed(ctx, &entries[i], scratch);
        write_model(&jw, entries[i].model, scratch, installed);
        jw_raw(&jw, ",\"profile\":");
        write_profile(&jw, entries[i].profile);
        jw_raw(&jw, ",\"options\":");
        write_options(&jw, entries[i].profile, &hw);
        jw_printf(&jw, ",\"fits\":%s,\"reason\":", entries[i].fits ? "true" : "false");
        if (entries[i].fits)
            jw_raw(&jw, "null}");
        else {
            jw_string(&jw, entries[i].reason);
            jw_raw(&jw, "}");
        }
    }
    jw_raw(&jw, "],");
    pn_local_write_store_only(&jw, ctx, m);
    jw_raw(&jw, "}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
    free(entries);
    free(scratch);
    free(m);
}

static const char *protocols_of(const pn_local_ctx *ctx)
{
    return ctx->allow_http ? PN_CURL_PROTOCOLS_ANY : PN_CURL_PROTOCOLS_HTTPS;
}

static void handle_refresh(int cfd, pn_local_ctx *ctx)
{
    char err[256], out[64];

    if (!ctx->catalog_url[0]) {
        hs_send_error(cfd, 400, "no_catalog_url", "this Pantheon Local was started with --catalog-url '' (no catalog)");
        return;
    }
    if (pn_catalog_refresh(ctx->catalog, ctx->catalog_url, protocols_of(ctx), err,
                           sizeof(err)) != 0) {
        fprintf(stderr, "local: %s\n", err);
        hs_send_error(cfd, 422, "catalog_refused", err);
        return;
    }
    snprintf(out, sizeof(out), "{\"version\":%lld}", pn_catalog_version(ctx->catalog));
    hs_send_json(cfd, 200, out);
}

int pn_local_catalog_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/local/catalog") == 0)
        handle_catalog(cfd, ctx);
    else if (strcmp(req->method, "POST") == 0 && strcmp(req->path, "/local/catalog/refresh") == 0)
        handle_refresh(cfd, ctx);
    else
        return 0;
    return 1;
}

static void *fetch_main(void *arg)
{
    pn_local_ctx *ctx = arg;
    char err[256];

    if (pn_catalog_refresh(ctx->catalog, ctx->catalog_url, protocols_of(ctx), err,
                           sizeof(err)) != 0)
        fprintf(stderr, "local: catalog fetch at start failed: %s\n", err);
    return NULL;
}

void pn_local_catalog_fetch_async(pn_local_ctx *ctx)
{
    pthread_t t;

    if (!ctx->catalog_url[0] || !ctx->catalog)
        return;
    if (pthread_create(&t, NULL, fetch_main, ctx) == 0)
        pthread_detach(t);
}
