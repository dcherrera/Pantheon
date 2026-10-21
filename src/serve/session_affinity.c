/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_affinity.c
 * @brief Implementation of the session -> instance affinity map (see
 *        session_affinity.h). Pure C, one mutex, fixed-size table, LRU eviction
 *        — the same shape as registry.c. No new deps.
 *
 * @status PC0.2 complete.
 * @issues None
 * @todo None
 */

#include "session_affinity.h"

#include <stdio.h>
#include <string.h>

int sa_init(session_affinity *sa)
{
    memset(sa, 0, sizeof(*sa));
    if (pthread_mutex_init(&sa->mu, NULL) != 0)
        return -1;
    return 0;
}

void sa_destroy(session_affinity *sa)
{
    if (!sa)
        return;
    pthread_mutex_destroy(&sa->mu);
}

/* Find the row index for a session_id, or -1. Caller holds the lock. */
static int sa_find(const session_affinity *sa, const char *session_id)
{
    for (int i = 0; i < sa->n; i++)
        if (strcmp(sa->ent[i].session_id, session_id) == 0)
            return i;
    return -1;
}

/* Index of the least-recently-used row (smallest seq). Caller holds the lock;
 * only called when the table is full, so n >= 1. */
static int sa_lru_index(const session_affinity *sa)
{
    int lru = 0;
    for (int i = 1; i < sa->n; i++)
        if (sa->ent[i].seq < sa->ent[lru].seq)
            lru = i;
    return lru;
}

bool sa_lookup(session_affinity *sa, const char *session_id,
               char *out, size_t out_cap)
{
    if (!sa || !session_id || !session_id[0] || !out || out_cap == 0)
        return false;

    bool found = false;
    pthread_mutex_lock(&sa->mu);
    int idx = sa_find(sa, session_id);
    if (idx >= 0) {
        sa->ent[idx].seq = ++sa->clock;   /* refresh LRU on every hit */
        snprintf(out, out_cap, "%s", sa->ent[idx].instance_id);
        found = true;
    }
    pthread_mutex_unlock(&sa->mu);
    return found;
}

void sa_record(session_affinity *sa, const char *session_id,
               const char *instance_id)
{
    if (!sa || !session_id || !session_id[0] || !instance_id || !instance_id[0])
        return;

    pthread_mutex_lock(&sa->mu);
    int idx = sa_find(sa, session_id);
    if (idx < 0) {
        /* New session: append, or evict the LRU row when the table is full. */
        if (sa->n < SA_MAX_SESSIONS)
            idx = sa->n++;
        else
            idx = sa_lru_index(sa);
        snprintf(sa->ent[idx].session_id, sizeof(sa->ent[idx].session_id),
                 "%s", session_id);
    }
    snprintf(sa->ent[idx].instance_id, sizeof(sa->ent[idx].instance_id),
             "%s", instance_id);
    sa->ent[idx].seq = ++sa->clock;
    pthread_mutex_unlock(&sa->mu);
}

int sa_forget_instance(session_affinity *sa, const char *instance_id)
{
    if (!sa || !instance_id || !instance_id[0])
        return 0;

    int removed = 0;
    pthread_mutex_lock(&sa->mu);
    int i = 0;
    while (i < sa->n) {
        if (strcmp(sa->ent[i].instance_id, instance_id) == 0) {
            sa->ent[i] = sa->ent[sa->n - 1];   /* swap-remove; order irrelevant */
            sa->n--;
            removed++;
            /* do not advance i: the swapped-in row must be checked too */
        } else {
            i++;
        }
    }
    pthread_mutex_unlock(&sa->mu);
    return removed;
}
