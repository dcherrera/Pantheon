/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_registry.c
 * @brief Implementation of the serve-side dispatch self-registration lifecycle
 *        (see serve_registry.h). Fully gated on rc_enabled() (PANTHEON_REGISTRY_URL):
 *        with it unset serve_registry_start() returns immediately having done
 *        nothing — no registration, no thread, and serve_registry_deregister_all()
 *        is likewise a no-op — so the serve process is byte-for-byte unchanged.
 *
 * @status G0 wiring + runtime add/remove (models loaded via /admin after startup
 *         now register too — the instance table is mutex-guarded).
 * @issues None
 * @todo None
 */

#include "serve_registry.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "batch_sched.h"
#include "registry_client.h"
#include "registry_wire.h"

/* Registered-instance table. Seeded by serve_registry_start(), then mutated at
 * runtime by serve_registry_add()/_remove() (an /admin load/unload). Guarded by
 * g_lock — the table is no longer write-once, so every read (heartbeat,
 * deregister) and write takes the lock. Network calls (rc_register/heartbeat/
 * deregister) are done OUTSIDE the lock so a slow POST never blocks the table. */
typedef struct {
    char model_id[RW_MODEL_ID_MAX];
    char instance_id[RW_INSTANCE_ID_MAX];
} sr_instance;

static sr_instance     g_instances[MR_MAX_MODELS];
static int             g_n_instances = 0;
static mr_registry    *g_reg  = NULL;      /* borrowed (process-lifetime) */
static char            g_host[100] = "";   /* set by serve_registry_start */
static int             g_port = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *env_or(const char *name, const char *fallback)
{
    const char *v = getenv(name);
    return (v && v[0]) ? v : fallback;
}

/* Register one model (by served name) with the dispatch registry and record it
 * for heartbeat/deregister. Idempotent: a model whose instance_id is already in
 * the table is skipped (so loading a second replica of the same name — same
 * host:port:model — does not double-register). Safe to call from any thread once
 * g_reg/g_host/g_port are set. */
static void sr_register_model(const char *name)
{
    if (!g_reg || !name || !*name)
        return;

    rc_identity id = {0};
    snprintf(id.model_id,    sizeof id.model_id,    "%s", name);
    snprintf(id.instance_id, sizeof id.instance_id, "%s:%d:%s", g_host, g_port, name);
    snprintf(id.node_id,     sizeof id.node_id,     "%s", env_or("PANTHEON_NODE_ID", "node0"));
    snprintf(id.endpoint,    sizeof id.endpoint,    "http://%s:%d", g_host, g_port);
    snprintf(id.transport,   sizeof id.transport,   "%s",
             env_or("PANTHEON_REGISTRY_TRANSPORT", RW_TRANSPORT_LOCAL));

    /* Dedupe + capacity guard (quick, under lock). */
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_n_instances; i++) {
        if (strcmp(g_instances[i].instance_id, id.instance_id) == 0) {
            pthread_mutex_unlock(&g_lock);
            return;   /* already registered (e.g. a replica of the same name) */
        }
    }
    if (g_n_instances >= MR_MAX_MODELS) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    pthread_mutex_unlock(&g_lock);

    /* capacity = served concurrency = batch-slot count (1 single-user). */
    bs_load   load  = {0};
    mr_entry *entry = mr_acquire(g_reg, name);
    if (entry) {
        bs_get_load(entry->sched, &load);
        mr_release(g_reg, entry);
    } else {
        load.capacity = 1;
    }
    id.capacity = load.capacity;

    /* Network register OUTSIDE the lock. */
    if (!rc_register(&id)) {
        fprintf(stderr, "serve: WARNING failed to register '%s' with dispatch registry "
                        "(will rely on next heartbeat / TTL)\n", id.model_id);
        return;
    }

    /* Append under the lock (re-check dedupe against a concurrent add). */
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_n_instances; i++) {
        if (strcmp(g_instances[i].instance_id, id.instance_id) == 0) {
            pthread_mutex_unlock(&g_lock);
            return;
        }
    }
    if (g_n_instances < MR_MAX_MODELS) {
        snprintf(g_instances[g_n_instances].model_id,
                 sizeof g_instances[g_n_instances].model_id, "%s", id.model_id);
        snprintf(g_instances[g_n_instances].instance_id,
                 sizeof g_instances[g_n_instances].instance_id, "%s", id.instance_id);
        g_n_instances++;
        fprintf(stderr, "serve: registered '%s' (instance %s, cap %d) with dispatch registry\n",
                id.model_id, id.instance_id, id.capacity);
    }
    pthread_mutex_unlock(&g_lock);
}

/* Background heartbeat: every RW_HEARTBEAT_SECONDS snapshot the instance table
 * under the lock, then (lock released) for each instance re-acquire its model
 * entry, read live load from its batch_sched, and POST a heartbeat. */
static void *heartbeat_thread(void *arg)
{
    (void)arg;
    for (;;) {
        sleep(RW_HEARTBEAT_SECONDS);

        sr_instance snap[MR_MAX_MODELS];
        int n;
        pthread_mutex_lock(&g_lock);
        n = g_n_instances;
        memcpy(snap, g_instances, (size_t)n * sizeof(sr_instance));
        pthread_mutex_unlock(&g_lock);

        for (int i = 0; i < n; i++) {
            bs_load    load  = {0};
            mr_entry  *entry = mr_acquire(g_reg, snap[i].model_id);
            if (entry) {
                bs_get_load(entry->sched, &load);
                mr_release(g_reg, entry);
            } else {
                /* Model unloaded out from under us: report unhealthy so the
                 * router stops dispatching to it (the TTL sweep also evicts). */
                load.capacity = load.free_slots = 0;
            }
            rc_load hb = {
                .active_slots = load.active_slots,
                .queue_depth  = load.queue_depth,
                .free_slots   = load.free_slots,
                .healthy      = (entry != NULL),
            };
            rc_heartbeat(snap[i].instance_id, &hb);
        }
    }
    return NULL;
}

void serve_registry_start(mr_registry *reg, const char *host, int port)
{
    if (!rc_enabled())          /* PANTHEON_REGISTRY_URL unset ⇒ fully off */
        return;

    g_reg  = reg;
    g_port = port;
    snprintf(g_host, sizeof g_host, "%s", host ? host : "127.0.0.1");

    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);
    for (int i = 0; i < n; i++)
        sr_register_model(info[i].name);

    pthread_mutex_lock(&g_lock);
    int registered = g_n_instances;
    pthread_mutex_unlock(&g_lock);
    if (registered == 0)
        return;   /* nothing registered ⇒ no heartbeat thread */

    pthread_t tid;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        fprintf(stderr, "serve: WARNING could not start dispatch heartbeat thread\n");
        return;
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&tid, &attr, heartbeat_thread, NULL) != 0)
        fprintf(stderr, "serve: WARNING could not start dispatch heartbeat thread\n");
    pthread_attr_destroy(&attr);
}

void serve_registry_add(const char *model_id)
{
    if (!rc_enabled() || !g_reg)   /* not started / disabled ⇒ no-op */
        return;
    sr_register_model(model_id);
}

void serve_registry_remove(const char *model_id)
{
    if (!rc_enabled() || !g_reg || !model_id || !*model_id)
        return;

    /* Only deregister when NO replica of this name remains loaded — unloading one
     * of two same-name replicas must keep the shared instance registered (the
     * serve still serves that name). */
    mr_entry *e = mr_acquire(g_reg, model_id);
    if (e) {
        mr_release(g_reg, e);
        return;
    }

    char inst[RW_INSTANCE_ID_MAX] = "";
    pthread_mutex_lock(&g_lock);
    int found = -1;
    for (int i = 0; i < g_n_instances; i++) {
        if (strcmp(g_instances[i].model_id, model_id) == 0) {
            found = i;
            snprintf(inst, sizeof inst, "%s", g_instances[i].instance_id);
            break;
        }
    }
    if (found >= 0) {
        g_instances[found] = g_instances[g_n_instances - 1];   /* swap-remove */
        g_n_instances--;
    }
    pthread_mutex_unlock(&g_lock);

    if (found >= 0)
        rc_deregister(inst);   /* best-effort, outside the lock */
}

void serve_registry_deregister_all(void)
{
    if (!rc_enabled())
        return;

    sr_instance snap[MR_MAX_MODELS];
    int n;
    pthread_mutex_lock(&g_lock);
    n = g_n_instances;
    memcpy(snap, g_instances, (size_t)n * sizeof(sr_instance));
    pthread_mutex_unlock(&g_lock);

    for (int i = 0; i < n; i++)
        rc_deregister(snap[i].instance_id);   /* best-effort */
}
