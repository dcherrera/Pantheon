/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   replica_affinity.c
 * @brief  Implementation of the session -> replica table (see replica_affinity.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "replica_affinity.h"

#include "../core/session_family.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char     key[PN_REPLICA_SID_MAX];
    uint64_t serial;     /* 0 = empty row */
    uint64_t last_use;
} affinity_row;

static affinity_row    g_rows[PN_REPLICA_AFFINITY_MAX];
static uint64_t        g_clock;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

uint64_t pn_replica_affinity_lookup(const char *sid, const uint64_t *serials, int n)
{
    uint64_t found = 0;

    if (!sid || !*sid)
        return 0;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX && !found; i++) {
        if (!g_rows[i].serial || !pn_session_same_family(g_rows[i].key, sid))
            continue;
        for (int j = 0; j < n; j++) {
            if (serials[j] == g_rows[i].serial) {
                found = serials[j];
                g_rows[i].last_use = ++g_clock;
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
    return found;
}

void pn_replica_affinity_bind(const char *sid, uint64_t serial)
{
    int slot = -1;

    if (!sid || !*sid || !serial)
        return;
    pthread_mutex_lock(&g_lock);
    /* The same key, else an empty row, else the least recently used one. */
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX && slot < 0; i++)
        if (g_rows[i].serial && strcmp(g_rows[i].key, sid) == 0)
            slot = i;
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX && slot < 0; i++)
        if (!g_rows[i].serial)
            slot = i;
    if (slot < 0) {
        slot = 0;
        for (int i = 1; i < PN_REPLICA_AFFINITY_MAX; i++)
            if (g_rows[i].last_use < g_rows[slot].last_use)
                slot = i;
    }
    snprintf(g_rows[slot].key, sizeof(g_rows[slot].key), "%s", sid);
    g_rows[slot].serial = serial;
    g_rows[slot].last_use = ++g_clock;
    pthread_mutex_unlock(&g_lock);
}

int pn_replica_affinity_count(uint64_t serial)
{
    int n = 0;

    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX; i++)
        n += serial && g_rows[i].serial == serial;
    pthread_mutex_unlock(&g_lock);
    return n;
}

void pn_replica_affinity_forget(const char *sid)
{
    if (!sid || !*sid)
        return;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX; i++)
        if (g_rows[i].serial && pn_session_same_family(g_rows[i].key, sid))
            memset(&g_rows[i], 0, sizeof(g_rows[i]));
    pthread_mutex_unlock(&g_lock);
}

void pn_replica_affinity_unbind(const char *sid, uint64_t serial)
{
    if (!sid || !*sid || !serial)
        return;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX; i++)
        if (g_rows[i].serial == serial && strcmp(g_rows[i].key, sid) == 0)
            memset(&g_rows[i], 0, sizeof(g_rows[i]));
    pthread_mutex_unlock(&g_lock);
}

void pn_replica_affinity_rebind(const char *sid, uint64_t from, uint64_t to)
{
    bool moved = false;

    if (!sid || !*sid || !to)
        return;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX; i++) {
        if (g_rows[i].serial && g_rows[i].serial == from &&
            pn_session_same_family(g_rows[i].key, sid)) {
            g_rows[i].serial = to;
            g_rows[i].last_use = ++g_clock;
            moved = true;
        }
    }
    pthread_mutex_unlock(&g_lock);
    if (!moved)
        pn_replica_affinity_bind(sid, to);
}

void pn_replica_affinity_forget_serial(uint64_t serial)
{
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < PN_REPLICA_AFFINITY_MAX; i++)
        if (serial && g_rows[i].serial == serial)
            memset(&g_rows[i], 0, sizeof(g_rows[i]));
    pthread_mutex_unlock(&g_lock);
}

void pn_replica_affinity_reset(void)
{
    pthread_mutex_lock(&g_lock);
    memset(g_rows, 0, sizeof(g_rows));
    g_clock = 0;
    pthread_mutex_unlock(&g_lock);
}
