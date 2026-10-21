/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file registry.c
 * @brief Implementation of the live-instance registry (see registry.h). Pure C,
 *        one mutex, fixed-size table. Uses the serve layer's json_reader for
 *        parsing register/heartbeat/deregister bodies and json_writer for the
 *        /v1/models and /registry/fleet responses — no new deps.
 *
 * @status G1 complete.
 * @issues None
 * @todo None
 */

#include "registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json_reader.h"
#include "json_writer.h"

/* Copy a JSON string field into a fixed dst buffer (truncating, always NUL
 * terminated). Returns true if the field was present and a string. */
static bool copy_str_field(const jr_value *obj, const char *key, char *dst,
                           size_t cap)
{
    const jr_value *v = jr_object_get(obj, key);
    const char *s = jr_as_string(v);
    if (!s) { dst[0] = '\0'; return false; }
    size_t n = strlen(s);
    if (n >= cap) n = cap - 1;
    memcpy(dst, s, n);
    dst[n] = '\0';
    return true;
}

int reg_init(reg_registry *r)
{
    memset(r, 0, sizeof(*r));
    if (pthread_mutex_init(&r->mu, NULL) != 0)
        return -1;
    return 0;
}

void reg_destroy(reg_registry *r)
{
    pthread_mutex_destroy(&r->mu);
}

/* Find the row index for an instance_id, or -1. Caller holds the lock. */
static int find_instance(reg_registry *r, const char *instance_id)
{
    for (int i = 0; i < r->n; i++)
        if (strcmp(r->inst[i].instance_id, instance_id) == 0)
            return i;
    return -1;
}

reg_status reg_register_json(reg_registry *r, const char *body)
{
    if (!body) return REG_BAD_REQUEST;
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) { jr_free(root); return REG_BAD_REQUEST; }

    reg_instance tmp;
    memset(&tmp, 0, sizeof(tmp));

    bool have_model = copy_str_field(root, "model_id", tmp.model_id, sizeof(tmp.model_id));
    bool have_inst  = copy_str_field(root, "instance_id", tmp.instance_id, sizeof(tmp.instance_id));
    bool have_ep    = copy_str_field(root, "endpoint", tmp.endpoint, sizeof(tmp.endpoint));
    if (!have_model || !have_inst || !have_ep) { jr_free(root); return REG_BAD_REQUEST; }

    copy_str_field(root, "node_id", tmp.node_id, sizeof(tmp.node_id));
    if (!copy_str_field(root, "transport", tmp.transport, sizeof(tmp.transport)))
        snprintf(tmp.transport, sizeof(tmp.transport), "%s", RW_TRANSPORT_LOCAL);
    tmp.capacity   = (int)jr_as_number(jr_object_get(root, "capacity"), 0);
    /* Effective context window (tokens). Absent/negative ⇒ 0 = unreported, and
     * every emit below then omits the field rather than advertising a zero a
     * client would have to special-case. */
    tmp.context_window =
        (int)jr_as_number(jr_object_get(root, "context_window"), 0);
    if (tmp.context_window < 0) tmp.context_window = 0;

    /* A fresh registration starts fully free + healthy until the first
     * heartbeat refines the load. */
    tmp.free_slots = tmp.capacity;
    tmp.healthy    = true;
    tmp.last_seen  = time(NULL);

    jr_free(root);

    pthread_mutex_lock(&r->mu);
    int idx = find_instance(r, tmp.instance_id);
    reg_status st = REG_OK;
    if (idx >= 0) {
        r->inst[idx] = tmp;                 /* idempotent re-register */
    } else if (r->n < REG_MAX_INSTANCES) {
        r->inst[r->n++] = tmp;
    } else {
        st = REG_FULL;
    }
    pthread_mutex_unlock(&r->mu);
    return st;
}

reg_status reg_heartbeat_json(reg_registry *r, const char *body)
{
    if (!body) return REG_BAD_REQUEST;
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) { jr_free(root); return REG_BAD_REQUEST; }

    char instance_id[RW_INSTANCE_ID_MAX];
    if (!copy_str_field(root, "instance_id", instance_id, sizeof(instance_id))) {
        jr_free(root);
        return REG_BAD_REQUEST;
    }

    const jr_value *load = jr_object_get(root, "load");
    /* Read load fields up-front (borrowed pointers valid until jr_free). */
    bool have_load = (load && jr_kind(load) == JR_OBJECT);
    int active = have_load ? (int)jr_as_number(jr_object_get(load, "active_slots"), 0) : 0;
    int queue  = have_load ? (int)jr_as_number(jr_object_get(load, "queue_depth"), 0) : 0;
    int free_s = have_load ? (int)jr_as_number(jr_object_get(load, "free_slots"), 0) : 0;
    /* The window rides the heartbeat too: the memory ladder can clamp it after
     * registration, so a register-time value would go stale exactly when it
     * matters. <=0 (or absent) leaves the prior value alone — an old serve that
     * doesn't send it must not blank out what it reported at register. */
    int ctxw = have_load
                   ? (int)jr_as_number(jr_object_get(load, "context_window"), 0)
                   : 0;
    const jr_value *hv = jr_object_get(root, "healthy");
    bool have_healthy = (jr_kind(hv) == JR_BOOL);
    bool healthy = jr_as_bool(hv, true);

    jr_free(root);

    pthread_mutex_lock(&r->mu);
    int idx = find_instance(r, instance_id);
    reg_status st = REG_OK;
    if (idx < 0) {
        st = REG_NOT_FOUND;
    } else {
        reg_instance *e = &r->inst[idx];
        if (have_load) {
            e->active_slots = active;
            e->queue_depth  = queue;
            e->free_slots   = free_s;
            if (ctxw > 0)
                e->context_window = ctxw;
        }
        if (have_healthy) e->healthy = healthy;
        e->last_seen = time(NULL);
    }
    pthread_mutex_unlock(&r->mu);
    return st;
}

reg_status reg_deregister_json_collect(reg_registry *r, const char *body,
                                       char *removed_out, size_t out_cap)
{
    if (removed_out && out_cap > 0)
        removed_out[0] = '\0';
    if (!body) return REG_BAD_REQUEST;
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) { jr_free(root); return REG_BAD_REQUEST; }

    char instance_id[RW_INSTANCE_ID_MAX];
    bool ok = copy_str_field(root, "instance_id", instance_id, sizeof(instance_id));
    jr_free(root);
    if (!ok) return REG_BAD_REQUEST;

    pthread_mutex_lock(&r->mu);
    int idx = find_instance(r, instance_id);
    reg_status st = REG_NOT_FOUND;
    if (idx >= 0) {
        if (removed_out && out_cap > 0)
            snprintf(removed_out, out_cap, "%s", r->inst[idx].instance_id);
        /* Swap-remove: move the last row into the hole. Order is irrelevant. */
        r->inst[idx] = r->inst[r->n - 1];
        r->n--;
        st = REG_OK;
    }
    pthread_mutex_unlock(&r->mu);
    return st;
}

reg_status reg_deregister_json(reg_registry *r, const char *body)
{
    return reg_deregister_json_collect(r, body, NULL, 0);
}

int reg_sweep_collect(reg_registry *r, time_t now,
                      char (*evicted_out)[RW_INSTANCE_ID_MAX], int max_out,
                      int *n_out)
{
    int evicted = 0;
    int reported = 0;
    pthread_mutex_lock(&r->mu);
    int i = 0;
    while (i < r->n) {
        if (now - r->inst[i].last_seen > RW_TTL_SECONDS) {
            if (evicted_out && reported < max_out)
                snprintf(evicted_out[reported++], RW_INSTANCE_ID_MAX, "%s",
                         r->inst[i].instance_id);
            r->inst[i] = r->inst[r->n - 1];
            r->n--;
            evicted++;
            /* do not advance i: the swapped-in row must be checked too */
        } else {
            i++;
        }
    }
    pthread_mutex_unlock(&r->mu);
    if (n_out)
        *n_out = reported;
    return evicted;
}

int reg_sweep(reg_registry *r, time_t now)
{
    return reg_sweep_collect(r, now, NULL, 0, NULL);
}

/* An instance is a live dispatch candidate iff healthy and within TTL. Caller
 * holds the lock. `now` is passed so callers use one consistent clock read. */
static bool is_live(const reg_instance *e, time_t now)
{
    return e->healthy && (now - e->last_seen <= RW_TTL_SECONDS);
}

char *reg_models_json(reg_registry *r)
{
    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, "{\"object\":\"list\",\"data\":[");

    pthread_mutex_lock(&r->mu);
    time_t now = time(NULL);
    int emitted = 0;
    for (int i = 0; i < r->n; i++) {
        if (!is_live(&r->inst[i], now)) continue;
        /* Dedup by model_id: only emit the FIRST live instance of each id, but
         * aggregate live count + free slots across all live replicas of it. */
        bool first = true;
        for (int j = 0; j < i; j++) {
            if (is_live(&r->inst[j], now) &&
                strcmp(r->inst[j].model_id, r->inst[i].model_id) == 0) {
                first = false;
                break;
            }
        }
        if (!first) continue;

        /* Window across replicas: the router may dispatch this model_id to ANY
         * live replica, so the only window it can honestly promise is the
         * SMALLEST one on offer. A replica that reports nothing (0) makes even
         * that unknowable — one unreported replica suppresses the field
         * entirely, because "undetected" is a state the client handles and a
         * window it can be denied is not. */
        int live = 0, total_free = 0, min_window = 0;
        bool window_known = true;
        for (int j = 0; j < r->n; j++) {
            if (is_live(&r->inst[j], now) &&
                strcmp(r->inst[j].model_id, r->inst[i].model_id) == 0) {
                live++;
                total_free += r->inst[j].free_slots;
                int w = r->inst[j].context_window;
                if (w <= 0)
                    window_known = false;
                else if (min_window == 0 || w < min_window)
                    min_window = w;
            }
        }
        if (emitted++) jw_raw(&jw, ",");
        jw_raw(&jw, "{\"id\":");
        jw_string(&jw, r->inst[i].model_id);
        jw_printf(&jw, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"pantheon\","
                       "\"live\":%d,\"free_slots\":%d", live, total_free);
        if (window_known && min_window > 0)
            jw_printf(&jw, ",\"context_window\":%d", min_window);
        jw_raw(&jw, "}");
    }
    pthread_mutex_unlock(&r->mu);

    jw_raw(&jw, "]}");
    if (jw.oom) { jw_free(&jw); return NULL; }
    char *out = jw.data;   /* transfer ownership */
    jw.data = NULL;
    jw_free(&jw);
    return out;
}

char *reg_fleet_json(reg_registry *r)
{
    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, "{\"object\":\"fleet\",\"models\":[");

    pthread_mutex_lock(&r->mu);
    time_t now = time(NULL);
    int emitted = 0;
    for (int i = 0; i < r->n; i++) {
        if (!is_live(&r->inst[i], now)) continue;
        bool first = true;
        for (int j = 0; j < i; j++) {
            if (is_live(&r->inst[j], now) &&
                strcmp(r->inst[j].model_id, r->inst[i].model_id) == 0) {
                first = false;
                break;
            }
        }
        if (!first) continue;

        int live = 0, total_free = 0, min_window = 0;
        bool window_known = true;
        if (emitted++) jw_raw(&jw, ",");
        jw_raw(&jw, "{\"model_id\":");
        jw_string(&jw, r->inst[i].model_id);
        jw_raw(&jw, ",\"instances\":[");
        int inst_emitted = 0;
        /* Collect distinct nodes while we walk the replicas. */
        char nodes[REG_MAX_INSTANCES][RW_NODE_ID_MAX];
        int n_nodes = 0;
        for (int j = 0; j < r->n; j++) {
            if (!is_live(&r->inst[j], now) ||
                strcmp(r->inst[j].model_id, r->inst[i].model_id) != 0)
                continue;
            reg_instance *e = &r->inst[j];
            live++;
            total_free += e->free_slots;
            if (inst_emitted++) jw_raw(&jw, ",");
            jw_raw(&jw, "{\"instance_id\":");
            jw_string(&jw, e->instance_id);
            jw_raw(&jw, ",\"node_id\":");
            jw_string(&jw, e->node_id);
            jw_raw(&jw, ",\"endpoint\":");
            jw_string(&jw, e->endpoint);
            jw_raw(&jw, ",\"transport\":");
            jw_string(&jw, e->transport);
            jw_printf(&jw, ",\"capacity\":%d,\"active_slots\":%d,\"queue_depth\":%d,"
                           "\"free_slots\":%d,\"healthy\":%s",
                      e->capacity, e->active_slots, e->queue_depth, e->free_slots,
                      e->healthy ? "true" : "false");
            if (e->context_window > 0)
                jw_printf(&jw, ",\"context_window\":%d", e->context_window);
            jw_raw(&jw, "}");
            if (e->context_window <= 0)
                window_known = false;
            else if (min_window == 0 || e->context_window < min_window)
                min_window = e->context_window;
            /* track distinct node */
            bool seen_node = false;
            for (int b = 0; b < n_nodes; b++)
                if (strcmp(nodes[b], e->node_id) == 0) { seen_node = true; break; }
            if (!seen_node && n_nodes < REG_MAX_INSTANCES)
                snprintf(nodes[n_nodes++], RW_NODE_ID_MAX, "%s", e->node_id);
        }
        jw_raw(&jw, "],\"nodes\":[");
        for (int b = 0; b < n_nodes; b++) {
            if (b) jw_raw(&jw, ",");
            jw_string(&jw, nodes[b]);
        }
        jw_printf(&jw, "],\"live\":%d,\"total_free_slots\":%d", live, total_free);
        /* Model-level window = the smallest a dispatch could land on; suppressed
         * outright if any live replica hasn't reported one (same rule as
         * reg_models_json — never promise a window a replica can't honour). */
        if (window_known && min_window > 0)
            jw_printf(&jw, ",\"context_window\":%d", min_window);
        jw_raw(&jw, "}");
    }
    pthread_mutex_unlock(&r->mu);

    jw_raw(&jw, "]}");
    if (jw.oom) { jw_free(&jw); return NULL; }
    char *out = jw.data;
    jw.data = NULL;
    jw_free(&jw);
    return out;
}

bool reg_pick_least_busy(reg_registry *r, const char *model_id,
                         const char *const *exclude, int n_exclude,
                         reg_instance *out)
{
    pthread_mutex_lock(&r->mu);
    time_t now = time(NULL);

    int best = -1;
    for (int i = 0; i < r->n; i++) {
        reg_instance *e = &r->inst[i];
        if (strcmp(e->model_id, model_id) != 0) continue;
        if (!is_live(e, now)) continue;

        bool excluded = false;
        for (int x = 0; x < n_exclude; x++)
            if (exclude && exclude[x] && strcmp(exclude[x], e->instance_id) == 0) {
                excluded = true;
                break;
            }
        if (excluded) continue;

        if (best < 0) { best = i; continue; }
        reg_instance *b = &r->inst[best];
        /* max free_slots, then min queue_depth. Ties fall through to the
         * round-robin bias below (best stays; equal candidates alternate). */
        if (e->free_slots > b->free_slots ||
            (e->free_slots == b->free_slots && e->queue_depth < b->queue_depth)) {
            best = i;
        }
    }

    bool found = false;
    if (best >= 0) {
        /* Round-robin across the set of candidates that TIE with the winner on
         * (free_slots, queue_depth), so equal-load replicas alternate rather
         * than always hitting the first. */
        reg_instance *b = &r->inst[best];
        int tie[REG_MAX_INSTANCES];
        int n_tie = 0;
        for (int i = 0; i < r->n; i++) {
            reg_instance *e = &r->inst[i];
            if (strcmp(e->model_id, model_id) != 0) continue;
            if (!is_live(e, now)) continue;
            bool excluded = false;
            for (int x = 0; x < n_exclude; x++)
                if (exclude && exclude[x] &&
                    strcmp(exclude[x], e->instance_id) == 0) { excluded = true; break; }
            if (excluded) continue;
            if (e->free_slots == b->free_slots && e->queue_depth == b->queue_depth)
                tie[n_tie++] = i;
        }
        int chosen = best;
        if (n_tie > 1)
            chosen = tie[r->rr_counter++ % (unsigned)n_tie];
        *out = r->inst[chosen];
        found = true;
    }
    pthread_mutex_unlock(&r->mu);
    return found;
}

bool reg_find_live(reg_registry *r, const char *model_id,
                   const char *instance_id, reg_instance *out)
{
    if (!model_id || !instance_id)
        return false;

    bool found = false;
    pthread_mutex_lock(&r->mu);
    time_t now = time(NULL);
    int idx = find_instance(r, instance_id);
    if (idx >= 0 && is_live(&r->inst[idx], now) &&
        strcmp(r->inst[idx].model_id, model_id) == 0) {
        *out = r->inst[idx];
        found = true;
    }
    pthread_mutex_unlock(&r->mu);
    return found;
}
