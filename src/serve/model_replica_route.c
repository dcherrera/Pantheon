/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   model_replica_route.c
 * @brief  mr_acquire_session / mr_acquire_session_on (model_registry.h): route a
 *         session to one replica of a name. A bound session returns to its
 *         replica while it holds a seat there; anything else goes to the
 *         requested replica when one is named (never elsewhere), else to the
 *         least busy one with a free session seat (live turns + seated sessions;
 *         any replica when all are full, which admission then refuses), then is
 *         bound there. mr_failover_session moves a refused, seatless session to
 *         another copy with a seat; mr_acquire_replicas takes every copy.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include <stdio.h>
#include <string.h>

#include "model_registry.h"
#include "model_session.h"
#include "replica_affinity.h"

/* Ready entries named `name`, in load order (serial). */
static int replicas_locked(mr_registry *reg, const char *name, mr_entry **out)
{
    int n = 0;

    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *c = &reg->entries[i];
        if (!c->used || !c->model || c->unloading || strncmp(c->name, name, MR_MAX_NAME) != 0)
            continue;
        int j = n++;
        while (j > 0 && out[j - 1]->serial > c->serial) {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = c;
    }
    return n;
}

/* The lowest-slot ready entry's name (the default model), or NULL. */
static const char *default_name_locked(mr_registry *reg)
{
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *c = &reg->entries[i];
        if (c->used && c->model && !c->unloading)
            return c->name;
    }
    return NULL;
}

/* The replica `sid` is bound to, or NULL. */
static mr_entry *bound_locked(mr_entry **cand, int n, const char *sid)
{
    uint64_t serials[MR_MAX_MODELS], bound;

    for (int i = 0; i < n; i++)
        serials[i] = cand[i]->serial;
    bound = pn_replica_affinity_lookup(sid, serials, n);
    for (int i = 0; bound && i < n; i++)
        if (cand[i]->serial == bound)
            return cand[i];
    return NULL;
}

/* Least busy: live turns + sessions seated on it (ms_sessions_used); ties to
 * the first loaded. Not bindings: an idle client's watch binds its session
 * without a seat, and after a restart every watch reconnects to the first copy
 * up, which then looked full while it served nothing. A replica with a free
 * session seat beats a full one whatever their loads, so a new session is
 * never refused on a full copy while another copy could seat it; when every
 * copy is full the least busy one refuses it. */
static mr_entry *least_busy_locked(mr_entry **cand, int n)
{
    mr_entry *best = NULL;
    int best_load = 0;
    bool best_free = false;

    for (int i = 0; i < n; i++) {
        int load = cand[i]->refs + ms_sessions_used(cand[i]);
        bool seat = ms_session_seat_free(cand[i]);
        if (!best || (seat && !best_free) ||
            (seat == best_free && load < best_load)) {
            best = cand[i];
            best_load = load;
            best_free = seat;
        }
    }
    return best;
}

/* `sid` is bound to `bound` but holds no seat there (a watch bound it while only
 * that copy was loaded, or the serve restarted): nothing of it would be lost by
 * moving, so place it as a new session would be. Returns where it goes and
 * moves its bindings when that is another copy. Unpinned routing only. */
static mr_entry *replace_unseated_locked(mr_entry **cand, int n, mr_entry *bound,
                                         const char *sid, mr_route *route)
{
    if (n < 2 || ms_session_holds_seat(bound, sid))
        return bound;
    mr_entry *p = least_busy_locked(cand, n);
    if (!p || p == bound || !ms_session_seat_free(p))
        return bound;
    pn_replica_affinity_rebind(sid, bound->serial, p->serial);
    route->moved = true;
    fprintf(stderr, "registry: session %.63s held no seat on '%s' replica %d; placed on "
            "replica %d (die %d)\n", sid, bound->name, bound->replica, p->replica,
            p->n_dies > 0 ? p->dies[0] : -1);
    return p;
}

/* Candidate `replica` (its copy number), or NULL. */
static mr_entry *numbered_locked(mr_entry **cand, int n, int replica)
{
    for (int i = 0; i < n; i++)
        if (cand[i]->replica == replica)
            return cand[i];
    return NULL;
}

/* Record the name's copy numbers on the route (for a caller's error text). */
static void note_replicas(mr_route *route, mr_entry **cand, int n)
{
    route->n_replicas = n;
    for (int i = 0; i < n; i++)
        route->replicas[i] = cand[i]->replica;
}

mr_entry *mr_acquire_session_on(mr_registry *reg, const char *name, const char *sid,
                                mr_route *route)
{
    mr_entry *cand[MR_MAX_MODELS], *e = NULL;
    mr_route none = {0};
    int n;

    if (!route)
        route = &none;
    route->status = MR_ROUTE_NO_MODEL;
    route->bound_now = route->kept_binding = route->moved = false;
    route->n_replicas = 0;
    route->name[0] = '\0';
    if (!reg)
        return NULL;
    pthread_mutex_lock(&reg->lock);
    if (!name || !*name)
        name = default_name_locked(reg);
    n = name ? replicas_locked(reg, name, cand) : 0;
    note_replicas(route, cand, n);
    snprintf(route->name, sizeof(route->name), "%s", name ? name : "");
    if (n > 0) {
        e = bound_locked(cand, n, sid);
        if (e && route->place_unseated && route->replica <= 0)
            e = replace_unseated_locked(cand, n, e, sid, route);
        if (e) {
            route->kept_binding = route->replica > 0 && e->replica != route->replica;
        } else {
            /* A new session: bind it where it lands (one row per session family);
             * a requested copy is the only place it may land. */
            e = route->replica > 0 ? numbered_locked(cand, n, route->replica)
                                   : least_busy_locked(cand, n);
            if (e && sid && *sid) {
                pn_replica_affinity_bind(sid, e->serial);
                route->bound_now = true;
                if (n > 1)   /* once per new session, never per turn */
                    fprintf(stderr, "registry: session %.63s -> '%s' replica %d on die %d%s\n",
                            sid, e->name, e->replica, e->n_dies > 0 ? e->dies[0] : -1,
                            route->replica > 0 ? " (requested)" : "");
            }
        }
        if (e) {
            e->refs++;
            route->status = MR_ROUTE_OK;
        } else {
            route->status = MR_ROUTE_NO_REPLICA;
        }
    }
    pthread_mutex_unlock(&reg->lock);
    return e;
}

mr_entry *mr_failover_session(mr_registry *reg, const mr_entry *from, const char *sid,
                              mr_route *route)
{
    mr_entry *cand[MR_MAX_MODELS], *others[MR_MAX_MODELS], *e = NULL;
    int n, m = 0;

    if (!reg || !from)
        return NULL;
    pthread_mutex_lock(&reg->lock);
    n = replicas_locked(reg, from->name, cand);
    for (int i = 0; i < n; i++)
        if (cand[i] != from && ms_session_seat_free(cand[i]))
            others[m++] = cand[i];
    e = m > 0 ? least_busy_locked(others, m) : NULL;
    if (e) {
        e->refs++;
        if (sid && *sid)
            pn_replica_affinity_rebind(sid, from->serial, e->serial);
        if (route) {
            route->status = MR_ROUTE_OK;
            route->moved = true;
        }
        fprintf(stderr, "registry: session %.63s refused on '%s' replica %d; failed over "
                "to replica %d (die %d)\n", sid && *sid ? sid : "-", from->name,
                from->replica, e->replica, e->n_dies > 0 ? e->dies[0] : -1);
    }
    pthread_mutex_unlock(&reg->lock);
    return e;
}

int mr_acquire_replicas(mr_registry *reg, const char *name, mr_entry **out, int max)
{
    mr_entry *cand[MR_MAX_MODELS];
    int n, k = 0;

    if (!reg || !name || !out || max <= 0)
        return 0;
    pthread_mutex_lock(&reg->lock);
    n = replicas_locked(reg, name, cand);
    for (int i = 0; i < n && k < max; i++) {
        cand[i]->refs++;
        out[k++] = cand[i];
    }
    pthread_mutex_unlock(&reg->lock);
    return k;
}

mr_entry *mr_acquire_session(mr_registry *reg, const char *name, const char *sid)
{
    return mr_acquire_session_on(reg, name, sid, NULL);
}
