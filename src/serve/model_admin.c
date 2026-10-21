/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_admin.c
 * @brief Implementation of the runtime load/unload admin endpoints (see
 *        model_admin.h). Thin JSON shims over model_registry + die_manager —
 *        all policy (duplicates, drain, placement, VRAM admission) lives
 *        there.
 *
 * @status Live on the CPU host; die numbers become real on the K80 test machine.
 * @issues Registry refusals surface as one reason string; the HTTP status is
 *         picked by matching it (see status_for_reason) — coarse but enough
 *         for an operator surface.
 * @todo None
 */

#include "model_admin.h"

#include "attn_residency_json.h"
#include "graph_residency_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/die_manager.h"
#include "../core/prefix_cache.h"
#include "../core/session_snapshot.h"
#include "base64.h"
#include "http_server.h"
#include "json_reader.h"
#include "json_writer.h"
#include "model_session.h"
#include "serve_registry.h"

/* Map a registry refusal reason to an HTTP status: duplicates conflict (409),
 * capacity refusals are 507 (insufficient storage), a missing file is 404,
 * a `dies` array this binary cannot honour is the CALLER's mistake (400),
 * anything else a 500. */
static int status_for_reason(const char *reason)
{
    /* serve_backend_guard's short reasons. 400, not 500: nothing is broken —
     * the request named accelerator dies on a process that has no accelerator
     * backend compiled in, or none visible. Retrying is pointless; the fix is a
     * different binary or an omitted `dies`. */
    if (strstr(reason, "no accelerator"))
        return 400;
    /* 409, not 500: a busy model is a CONFLICTING STATE, not a broken server.
     * An unload that could not drain refuses rather than reporting success and
     * stranding the dies, and the operator's move is to retry — same family as
     * "already unloading". */
    if (strstr(reason, "already loaded") || strstr(reason, "already unloading") ||
        strstr(reason, "model is busy"))
        return 409;
    if (strstr(reason, "admission") || strstr(reason, "binding") ||
        strstr(reason, "table full"))
        return 507;
    if (strstr(reason, "not found") || strstr(reason, "no such"))
        return 404;
    return 500;
}

/* Write one model row {"id":..,"object":"model","dies":[..],"vram_bytes":..}.
 *
 * The admin surface carries BOTH context numbers because an operator's question
 * is different from a client's: `context_window` is what sessions actually get
 * right now, `context_window_configured` is what --ctx-cap asked for. They are
 * equal unless the memory ladder is clamping this instant, and seeing them
 * differ is the whole point — that is the signal that pressure is costing
 * context. (/v1/models advertises only the effective one; a client has no use
 * for a window it cannot have.) */
static void write_model_row(json_writer *jw, const mr_info *m)
{
    jw_raw(jw, "{\"id\":");
    jw_string(jw, m->name);
    jw_raw(jw, ",\"object\":\"model\",\"path\":");
    jw_string(jw, m->path);
    jw_raw(jw, ",\"dies\":[");
    for (int i = 0; i < m->n_dies; i++)
        jw_printf(jw, "%s%d", i ? "," : "", m->dies[i]);
    /* Both concurrency numbers, for exactly the reason the row already carries
     * both context numbers: `max_concurrent` is what this model SERVES right now
     * (bs_get_load's capacity), `max_concurrent_configured` is what
     * --max-concurrent asked for. They differ whenever there is no scheduler at
     * all (no CUDA compiled in, QWEN35_BATCH_SLOTS off, bs_create failed) or the
     * memory-pressure ladder clamped the slot count — and seeing them differ IS
     * the signal. The incident serve reported max_concurrent 2 with a NULL
     * scheduler and an actual served concurrency of 1. */
    jw_printf(jw, "],\"vram_bytes\":%llu,\"max_sessions\":%d,"
                  "\"max_concurrent\":%d,\"max_concurrent_configured\":%d",
              (unsigned long long)m->vram_bytes, m->max_sessions,
              m->served_concurrency, m->max_concurrent);
    if (m->context_window > 0)
        jw_printf(jw, ",\"context_window\":%d", m->context_window);
    if (m->context_window_configured > 0)
        jw_printf(jw, ",\"context_window_configured\":%d",
                  m->context_window_configured);
    /* The operator gets the per-class breakdown and the lifetime counters: a
     * model at live 38/40 with capture_failures 2 lost two layers' graphs to
     * something transient, which is a completely different story from a model
     * reporting 40 ineligible layers because its experts are not Q3_K. */
    gr_write_detail(jw, &m->graphs);
    /* And which attention path each layer is actually on. `host` > 0 is the
     * fault; `reasons` names it, and `gave_up` says the layer has stopped
     * trying to get back on-die at this geometry. */
    ar_write_detail(jw, &m->attn);
    jw_raw(jw, "}");
}

void ma_handle_load(int client_fd, mr_registry *reg, const char *body)
{
    if (!body) {
        hs_send_error(client_fd, 400, "invalid_request_error", "empty request body");
        return;
    }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        jr_free(root);
        return;
    }
    const char *path = jr_as_string(jr_object_get(root, "path"));
    const char *name = jr_as_string(jr_object_get(root, "name"));
    if (!path || !*path) {
        hs_send_error(client_fd, 400, "invalid_request_error", "'path' is required");
        jr_free(root);
        return;
    }

    int dies[PN_MODEL_MAX_DIES];
    int n_dies = 0;
    const jr_value *jd = jr_object_get(root, "dies");
    if (jr_kind(jd) == JR_ARRAY) {
        int len = jr_array_len(jd);
        if (len > PN_MODEL_MAX_DIES) {
            hs_send_error(client_fd, 400, "invalid_request_error", "too many dies");
            jr_free(root);
            return;
        }
        for (int i = 0; i < len; i++)
            dies[n_dies++] = (int)jr_as_number(jr_array_get(jd, i), -1);
    }

    pn_load_opts opts = {
        .quantize = jr_as_bool(jr_object_get(root, "quantize"), false),
    };

    /* Per-model context cap (KV Phase 1): the KV ring is sized to this window at
     * load, so it must be in the load opts. Absent/<=0 = engine default
     * (QWEN35_MAX_SEQ / PN_SESSION_MAX_CTX). This is how ornith gets a 40K
     * window without moving saber's. */
    int ctx_cap = (int)jr_as_number(jr_object_get(root, "ctx_cap"), 0);
    if (ctx_cap < 0) ctx_cap = 0;
    opts.ctx_cap = ctx_cap;

    /* Per-model AUTO-path reasoning cap (tokens); absent/<=0 = disabled. */
    int reasoning_budget = (int)jr_as_number(
        jr_object_get(root, "reasoning_budget"), 0);
    if (reasoning_budget < 0) reasoning_budget = 0;

    /* Per-model live-session (RESIDENCY) cap; absent/<=0 = the engine default
     * (BS_MAX_USERS). mr_load clamps into [1, BS_MAX_USERS]. */
    int max_sessions = (int)jr_as_number(
        jr_object_get(root, "max_sessions"), 0);
    if (max_sessions < 0) max_sessions = 0;

    /* Per-model CONCURRENCY cap = the batch scheduler's decode slots;
     * absent/<=0 = max_sessions. mr_load clamps into [1, BS_MAX_USERS]. */
    int max_concurrent = (int)jr_as_number(
        jr_object_get(root, "max_concurrent"), 0);
    if (max_concurrent < 0) max_concurrent = 0;

    char reason[160] = "";
    if (mr_load(reg, path, name, &opts, n_dies ? dies : NULL, n_dies,
                reasoning_budget, max_sessions, max_concurrent,
                reason, sizeof(reason)) != 0) {
        hs_send_error(client_fd, status_for_reason(reason), "load_failed", reason);
        jr_free(root);
        return;
    }

    /* Answer with the freshly-registered row (resolve the served name — it
     * may have been derived from the basename). */
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);
    const mr_info *loaded = NULL;
    for (int i = 0; i < n; i++)
        if (strcmp(info[i].path, path) == 0 &&
            (!name || strcmp(info[i].name, name) == 0))
            loaded = &info[i];

    /* Register the newly-loaded model with the dispatch router so it becomes
     * routable — startup registration only covers the launch model, so an
     * /admin-loaded model would otherwise be invisible to the router (503 on
     * dispatch). Uses the resolved served name (may be derived from the path). */
    {
        const char *reg_name = loaded ? loaded->name : name;
        if (reg_name && *reg_name)
            serve_registry_add(reg_name);
    }

    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); jr_free(root); return; }
    if (loaded) {
        write_model_row(&jw, loaded);
        hs_send_json(client_fd, 200, jw.data);
    } else {
        hs_send_json(client_fd, 200, "{\"status\":\"loaded\"}");
    }
    jw_free(&jw);
    jr_free(root);
}

void ma_handle_unload(int client_fd, mr_registry *reg, const char *body)
{
    if (!body) {
        hs_send_error(client_fd, 400, "invalid_request_error", "empty request body");
        return;
    }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        jr_free(root);
        return;
    }
    const char *name = jr_as_string(jr_object_get(root, "name"));
    if (!name || !*name) {
        hs_send_error(client_fd, 400, "invalid_request_error", "'name' is required");
        jr_free(root);
        return;
    }

    char reason[160] = "";
    if (mr_unload(reg, name, reason, sizeof(reason)) != 0) {
        hs_send_error(client_fd, status_for_reason(reason), "unload_failed", reason);
        jr_free(root);
        return;
    }

    /* Deregister from the router (only if no replica of this name remains). */
    serve_registry_remove(name);

    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); jr_free(root); return; }
    jw_raw(&jw, "{\"unloaded\":");
    jw_string(&jw, name);
    jw_raw(&jw, "}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
    jr_free(root);
}

/* Serialize the exported snapshot to a blob, base64 it, and answer with the
 * session-export JSON. Owns `snap` (frees it). Returns after answering. */
static void session_export_reply(int client_fd, pn_arch arch, void *snap,
                                 uint64_t weights_version, const char *sid,
                                 int n_tok, uint32_t *tokens)
{
    unsigned char *blob = NULL;
    size_t blob_len = 0;
    if (pn_session_snapshot_serialize(arch, snap, weights_version, &blob,
                                      &blob_len) != 0) {
        pn_session_snapshot_free(arch, snap);
        free(tokens);
        hs_send_error(client_fd, 500, "server_error", "snapshot serialize failed");
        return;
    }
    pn_session_snapshot_free(arch, snap);

    char *b64 = pn_base64_encode(blob, blob_len);
    free(blob);
    if (!b64) {
        free(tokens);
        hs_send_error(client_fd, 500, "server_error", "oom");
        return;
    }

    json_writer jw;
    if (!jw_init(&jw)) { free(b64); free(tokens); hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    jw_raw(&jw, "{\"session_id\":");
    jw_string(&jw, sid);
    jw_printf(&jw, ",\"weights_version\":%llu,\"n_tok\":%d,\"blob_bytes\":%zu"
              ",\"blob_b64\":", (unsigned long long)weights_version, n_tok, blob_len);
    jw_string(&jw, b64);
    /* Committed token ids the blob covers. The client MUST send these back on
     * /session/restore (len == the snapshot's pos), so export has to provide
     * them — a blob-only client has no other way to get them. */
    jw_raw(&jw, ",\"tokens\":[");
    for (int i = 0; i < n_tok; i++) {
        if (i) jw_raw(&jw, ",");
        jw_printf(&jw, "%u", tokens ? tokens[i] : 0u);
    }
    jw_raw(&jw, "]}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
    free(b64);
    free(tokens);
}

void ma_handle_session_export(int client_fd, mr_registry *reg, const char *body)
{
    if (!body) {
        hs_send_error(client_fd, 400, "invalid_request_error", "empty request body");
        return;
    }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        jr_free(root);
        return;
    }
    const char *name = jr_as_string(jr_object_get(root, "name"));
    const char *sid = jr_as_string(jr_object_get(root, "session_id"));
    if (!sid || !*sid) {
        hs_send_error(client_fd, 400, "invalid_request_error", "'session_id' is required");
        jr_free(root);
        return;
    }

    mr_entry *entry = mr_acquire(reg, name);
    if (!entry) {
        hs_send_error(client_fd, 404, "model_not_found",
                      "the requested model is not loaded (see GET /v1/models)");
        jr_free(root);
        return;
    }
    /* An architecture with no snapshot family is a REFUSAL WITH ITS OWN REASON,
     * not a lookup miss. Before this existed the only predicate in the path was
     * "is this qwen35moe?", so a live, generating session on any other arch —
     * every model the Tenstorrent backend serves, and every dense Llama/Qwen on
     * CPU or CUDA — came back 404 session_not_found. The operator was told the
     * session was gone; it was sitting in the prefix cache resuming turns. */
    pn_arch arch = entry->model->config.arch;
    if (!pn_session_snapshot_supported(arch)) {
        mr_release(reg, entry);
        hs_send_error(client_fd, 400, "session_export_unsupported",
                      "this model's architecture has no session snapshot "
                      "(its per-session state cannot be serialized); "
                      "replay the conversation instead");
        jr_free(root);
        return;
    }
    uint64_t weights_version = mr_weights_version(entry);
    int n_tok = 0;
    uint32_t *tokens = NULL;
    void *snap = NULL;

    /* Route through the session facade. A BATCHED model's session lives in the
     * scheduler's warm decode slot, NOT in the prefix cache (which holds only
     * admission bookkeeping on that path) — looking it up in the prefix cache,
     * what this used to do, made export answer session_not_found for every
     * session on every batched model, i.e. everything the K80 test machine actually runs.
     * The facade materializes the KV rows from the GPU decode-slot ring on that
     * path (Phase 10b / INV3) and takes the model's session lock either way, so
     * an in-flight turn cannot mutate the state mid-export. */
    ms_result msr = ms_session_export(entry, sid, &snap, &tokens, &n_tok);
    if (msr != MS_OK) {
        mr_release(reg, entry);
        if (msr == MS_NO_CACHE)
            hs_send_error(client_fd, 400, "session_caching_disabled",
                          "this model has no session cache to export from");
        else if (msr == MS_BUSY)
            hs_send_error(client_fd, 429, "session_busy",
                          "a turn is in flight on that session; "
                          "export again once it completes");
        else if (msr == MS_NOT_FOUND)
            hs_send_error(client_fd, 404, "session_not_found",
                          "no live session with that id (or nothing to export)");
        else
            hs_send_error(client_fd, 500, "server_error",
                          "session export failed");
        jr_free(root);
        return;
    }

    session_export_reply(client_fd, arch, snap, weights_version, sid, n_tok,
                         tokens);
    mr_release(reg, entry);
    jr_free(root);
}

void ma_handle_models(int client_fd, mr_registry *reg)
{
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);

    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    jw_raw(&jw, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < n; i++) {
        if (i) jw_raw(&jw, ",");
        write_model_row(&jw, &info[i]);
    }
    jw_raw(&jw, "]}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
}

void ma_handle_dies(int client_fd)
{
    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    int n = pn_dm_count();
    jw_printf(&jw, "{\"object\":\"list\",\"n_dies\":%d,\"data\":[", n);
    for (int i = 0; i < n; i++) {
        pn_dm_die d;
        if (pn_dm_get(i, &d) != 0)
            continue;
        if (i) jw_raw(&jw, ",");
        jw_printf(&jw, "{\"die\":%d,\"card\":%d,\"model\":", i, pn_dm_card_of(i));
        jw_string(&jw, d.model_name);
        jw_printf(&jw, ",\"n_models\":%d,\"vram_budget\":%llu,\"vram_used\":%llu}",
                  d.n_models, (unsigned long long)d.vram_budget,
                  (unsigned long long)d.vram_used);
    }
    jw_raw(&jw, "]}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
}
