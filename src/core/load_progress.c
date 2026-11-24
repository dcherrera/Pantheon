/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_progress.c
 * @brief  Implementation of the load / unload progress slots (see
 *         load_progress.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "load_progress.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    bool             open;
    pn_load_progress p;
} progress_slot;

static progress_slot          g_slots[PN_LOAD_PROGRESS_SLOTS];
static pthread_mutex_t        g_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local int      g_current = -1;   /* this thread's slot */

/* The unit of `die` in the current slot, added when missing. g_lock held. */
static pn_load_progress_unit *current_unit(int die)
{
    if (g_current < 0 || !g_slots[g_current].open)
        return NULL;
    pn_load_progress *p = &g_slots[g_current].p;
    for (int i = 0; i < p->n_units; i++)
        if (p->units[i].die == die)
            return &p->units[i];
    if (p->n_units >= PN_LOAD_PROGRESS_UNITS)
        return NULL;
    pn_load_progress_unit *u = &p->units[p->n_units++];
    memset(u, 0, sizeof(*u));
    u->die = die;
    return u;
}

int pn_load_progress_begin(const char *tag, int replica, pn_load_phase phase, const int *dies,
                           const uint64_t *totals, int n_dies)
{
    pn_load_progress_end();
    pthread_mutex_lock(&g_lock);
    int s = 0;
    while (s < PN_LOAD_PROGRESS_SLOTS && g_slots[s].open)
        s++;
    if (s < PN_LOAD_PROGRESS_SLOTS) {
        pn_load_progress *p = &g_slots[s].p;
        memset(p, 0, sizeof(*p));
        snprintf(p->tag, sizeof(p->tag), "%s", tag ? tag : "");
        p->replica = replica;
        p->phase = phase;
        for (int i = 0; i < n_dies && i < PN_LOAD_PROGRESS_UNITS; i++) {
            p->units[i].die = dies[i];
            p->units[i].total = totals ? totals[i] : 0;
            p->n_units++;
        }
        g_slots[s].open = true;
        g_current = s;
    }
    pthread_mutex_unlock(&g_lock);
    return s < PN_LOAD_PROGRESS_SLOTS ? 0 : -1;
}

void pn_load_progress_end(void)
{
    if (g_current < 0)
        return;
    pthread_mutex_lock(&g_lock);
    g_slots[g_current].open = false;
    pthread_mutex_unlock(&g_lock);
    g_current = -1;
}

void pn_load_progress_set_total(int die, uint64_t total)
{
    if (g_current < 0)
        return;
    pthread_mutex_lock(&g_lock);
    pn_load_progress_unit *u = current_unit(die);
    if (u)
        u->total = total > u->bytes ? total : u->bytes;
    pthread_mutex_unlock(&g_lock);
}

void pn_load_progress_set_bytes(int die, uint64_t bytes)
{
    if (g_current < 0)
        return;
    pthread_mutex_lock(&g_lock);
    pn_load_progress_unit *u = current_unit(die);
    if (u) {
        u->bytes = bytes;
        /* An unload's total is what it started from; a load's grows to its bytes. */
        if (u->total < bytes)
            u->total = bytes;
    }
    pthread_mutex_unlock(&g_lock);
}

void pn_load_progress_set_layer(int layer, int n_layers)
{
    if (g_current < 0)
        return;
    pthread_mutex_lock(&g_lock);
    if (g_slots[g_current].open) {
        g_slots[g_current].p.layer = layer;
        g_slots[g_current].p.n_layers = n_layers;
    }
    pthread_mutex_unlock(&g_lock);
}

int pn_load_progress_list(pn_load_progress *out, int max)
{
    int n = 0;

    pthread_mutex_lock(&g_lock);
    for (int s = 0; s < PN_LOAD_PROGRESS_SLOTS && n < max; s++)
        if (g_slots[s].open)
            out[n++] = g_slots[s].p;
    pthread_mutex_unlock(&g_lock);
    return n;
}

int pn_load_progress_find(const char *tag, int replica, pn_load_progress *out)
{
    int rc = -1;

    pthread_mutex_lock(&g_lock);
    for (int s = 0; s < PN_LOAD_PROGRESS_SLOTS && rc != 0; s++) {
        if (!g_slots[s].open || strcmp(g_slots[s].p.tag, tag) != 0 ||
            g_slots[s].p.replica != replica)
            continue;
        *out = g_slots[s].p;
        rc = 0;
    }
    pthread_mutex_unlock(&g_lock);
    return rc;
}

uint64_t pn_load_progress_die_bytes(int die, int *active)
{
    uint64_t bytes = 0;
    int hit = 0;

    pthread_mutex_lock(&g_lock);
    for (int s = 0; s < PN_LOAD_PROGRESS_SLOTS; s++) {
        if (!g_slots[s].open)
            continue;
        for (int i = 0; i < g_slots[s].p.n_units; i++) {
            if (g_slots[s].p.units[i].die != die)
                continue;
            bytes += g_slots[s].p.units[i].bytes;
            hit = 1;
        }
    }
    pthread_mutex_unlock(&g_lock);
    if (active)
        *active = hit;
    return bytes;
}
