/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_session.c
 * @brief The session-state facade — see model_session.h for the contract and
 *        for why serve code must not reach past it into `entry->prefix_cache`
 *        or `entry->sched` — and the table of which person each serve session
 *        belongs to (people registry §4.8).
 *
 * Lock discipline (identical on every operation here):
 *   - BATCHED path: the scheduler owns the state (and admits sessions), and
 *     `bs_session_*` take `sched->mu` + the GPU turnstile themselves.  The
 *     model's prefix cache is not used on this path.
 *   - SINGLE-USER path: `gen_lock` serializes the cache ops against that
 *     model's generation (the cache is documented thread-unsafe and the
 *     single-user generation mutates the very states we touch).
 */

#include "model_session.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "batch_sched.h"
#include "replica_affinity.h"
#include "../core/prefix_cache.h"
#include "../core/session_family.h"
#include "../core/session_snapshot.h"

/* Both stores keep a session id's first 63 bytes, so a binding does too. */
#define MS_SID_KEY 64

/* One session id bound to a person (ms_session_principal_check). */
typedef struct {
    char     sid[MS_SID_KEY];
    char     principal[MS_PRINCIPAL_MAX];
    uint64_t serial;    /* the entry of its last bind (single-user seats) */
    uint64_t last_ms;   /* last use: the least recent goes when the table is full */
} ms_binding;

static ms_binding      g_bindings[MS_PRINCIPAL_BINDINGS_MAX];
static int             g_n_bindings;
static pthread_mutex_t g_bindings_lock = PTHREAD_MUTEX_INITIALIZER;

uint64_t ms_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static uint64_t g_idle_ttl_default_ms = 12u * 60u * 60u * 1000u;   /* 12 hours */

void ms_set_session_idle_ttl_default_ms(uint64_t ttl_ms)
{
    g_idle_ttl_default_ms = ttl_ms;
}

uint64_t ms_session_idle_ttl_ms(void)
{
    static uint64_t cached = 0;
    static int loaded = 0;
    if (!loaded) {
        const char *env = getenv("PN_SESSION_IDLE_TTL_MS");
        if (env && *env) {
            char *end = NULL;
            long v = strtol(env, &end, 10);
            cached = (end != env && *end == '\0' && v >= 0)
                   ? (uint64_t)v : g_idle_ttl_default_ms;
        } else {
            cached = g_idle_ttl_default_ms;
        }
        loaded = 1;
    }
    return cached;
}

/* Map a scheduler result onto the facade's. */
static ms_result ms_from_bs(bs_session_result r)
{
    switch (r) {
    case BS_SESSION_OK:        return MS_OK;
    case BS_SESSION_NOT_FOUND: return MS_NOT_FOUND;
    case BS_SESSION_BUSY:      return MS_BUSY;
    default:                   return MS_ERROR;
    }
}

ms_result ms_session_export(mr_entry *entry, const char *sid,
                            void **snap_out, uint32_t **tokens_out,
                            int *n_tok_out)
{
    if (snap_out)   *snap_out = NULL;
    if (tokens_out) *tokens_out = NULL;
    if (n_tok_out)  *n_tok_out = 0;
    if (!entry || !sid || !sid[0] || !snap_out)
        return MS_ERROR;

    /* ── Batched arm: the session lives in its warm decode slot. ── */
    if (entry->sched)
        return ms_from_bs(bs_session_export(entry->sched, sid, snap_out,
                                            tokens_out, n_tok_out));

    /* ── Single-user arm: the session lives in the prefix cache. ── */
    if (!entry->prefix_cache)
        return MS_NO_CACHE;

    pthread_mutex_lock(&entry->gen_lock);
    int n_tok = 0;
    const uint32_t *tok_ref = NULL;
    void *snap = pn_prefix_cache_export_snapshot(entry->prefix_cache, sid,
                                                 &n_tok, &tok_ref);
    /* Copy the committed ids while STILL under the lock: tok_ref points into
     * the live entry and is only valid until an in-flight turn mutates it.
     * `snap` is already a deep copy, so it is safe post-unlock. */
    uint32_t *tokens = NULL;
    int oom = 0;
    if (snap && n_tok > 0 && tok_ref) {
        tokens = (uint32_t *)malloc((size_t)n_tok * sizeof(uint32_t));
        if (tokens) memcpy(tokens, tok_ref, (size_t)n_tok * sizeof(uint32_t));
        else oom = 1;
    }
    pthread_mutex_unlock(&entry->gen_lock);

    if (!snap)
        return MS_NOT_FOUND;
    if (oom) {
        pn_session_snapshot_free(entry->model->config.arch, snap);
        return MS_ERROR;
    }
    *snap_out = snap;
    if (tokens_out) *tokens_out = tokens; else free(tokens);
    if (n_tok_out)  *n_tok_out = n_tok;
    return MS_OK;
}

ms_result ms_session_restore(mr_entry *entry, const char *sid,
                             const void *snap, const uint32_t *tokens,
                             int n_tok)
{
    if (!entry || !sid || !sid[0] || !snap || !tokens || n_tok <= 0)
        return MS_ERROR;

    /* ── Batched arm.  Note what this REPLACES: the old code imported into the
     * prefix cache even for batched models, which allocated a fresh pn_state
     * (and its VRAM KV cache) that no batched generation ever reads — the
     * restore reported success and did nothing. ── */
    if (entry->sched) {
        bs_session_result r = bs_session_restore(entry->sched, sid, snap,
                                                 tokens, n_tok);
        if (r == BS_SESSION_ERROR)
            return MS_BAD_SNAPSHOT;   /* dim / pos mismatch → replay instead */
        return ms_from_bs(r);
    }

    /* ── Single-user arm.  Admit + import under ONE continuously-held lock:
     * the admission reaper frees idle slots, so it must not race the import or
     * a concurrent generation, and a gap between the two would let another
     * request take the last slot so import LRU-evicts a live session. ── */
    if (!entry->prefix_cache)
        return MS_NO_CACHE;

    pthread_mutex_lock(&entry->gen_lock);
    pn_session_admit adm = pn_prefix_cache_admit(entry->prefix_cache, sid,
                                                 ms_now_ms(),
                                                 ms_session_idle_ttl_ms(),
                                                 NULL);
    if (adm == PN_SESSION_ADMIT_AT_CAPACITY) {
        pthread_mutex_unlock(&entry->gen_lock);
        return MS_BUSY;
    }
    int rc = pn_prefix_cache_import_snapshot(entry->prefix_cache, sid, snap,
                                             tokens, n_tok);
    if (rc == 0)
        pn_prefix_cache_touch(entry->prefix_cache, sid, ms_now_ms());
    pthread_mutex_unlock(&entry->gen_lock);

    return rc == 0 ? MS_OK : MS_BAD_SNAPSHOT;
}

/* Forget the bindings of `sid` (by_prefix: of every id starting with it). */
static void principal_forget(const char *sid, bool by_prefix)
{
    char key[MS_SID_KEY];
    size_t n = 0;

    snprintf(key, sizeof(key), "%s", sid);
    n = strlen(key);
    pthread_mutex_lock(&g_bindings_lock);
    for (int i = 0; i < g_n_bindings;) {
        const char *b = g_bindings[i].sid;
        if (by_prefix ? strncmp(b, key, n) != 0 : strcmp(b, key) != 0) {
            i++;
            continue;
        }
        g_bindings[i] = g_bindings[--g_n_bindings];
    }
    pthread_mutex_unlock(&g_bindings_lock);
}

void ms_session_release(mr_entry *entry, const char *sid)
{
    if (!entry || !sid || !sid[0])
        return;
    /* Batched: the session (and its seat) is its warm decode slot; the prefix
     * cache holds nothing on this path. */
    if (entry->sched) {
        bs_session_release(entry->sched, sid);
    } else if (entry->prefix_cache) {
        pthread_mutex_lock(&entry->gen_lock);
        pn_prefix_cache_release_session(entry->prefix_cache, sid);
        pthread_mutex_unlock(&entry->gen_lock);
    }
    principal_forget(sid, false);
}

void ms_session_release_by_prefix(mr_entry *entry, const char *base_prefix)
{
    if (!entry || !base_prefix || !base_prefix[0])
        return;
    if (entry->sched) {
        bs_session_release_by_prefix(entry->sched, base_prefix);
    } else if (entry->prefix_cache) {
        pthread_mutex_lock(&entry->gen_lock);
        pn_prefix_cache_release_by_prefix(entry->prefix_cache, base_prefix);
        pthread_mutex_unlock(&entry->gen_lock);
    }
    principal_forget(base_prefix, true);
}

/* A slot for a new binding: a free one, else the least recently used (logged:
 * that session can then be bound again by whoever uses it next). */
static int principal_slot_locked(void)
{
    int oldest = 0;

    if (g_n_bindings < MS_PRINCIPAL_BINDINGS_MAX)
        return g_n_bindings++;
    for (int i = 1; i < g_n_bindings; i++)
        if (g_bindings[i].last_ms < g_bindings[oldest].last_ms)
            oldest = i;
    fprintf(stderr, "serve: session owner table full (%d); the least recently used "
            "binding is forgotten\n", MS_PRINCIPAL_BINDINGS_MAX);
    return oldest;
}

ms_result ms_session_principal_check(mr_entry *entry, const char *sid, const char *principal,
                                     bool admin, ms_principal_use use)
{
    char key[MS_SID_KEY];
    int exact = -1, mine = 0;

    if (!principal || !principal[0] || !sid || !sid[0])
        return MS_OK;
    snprintf(key, sizeof(key), "%s", sid);
    pthread_mutex_lock(&g_bindings_lock);
    for (int i = 0; i < g_n_bindings; i++) {
        if (!pn_session_same_family(g_bindings[i].sid, key))
            continue;
        if (strcmp(g_bindings[i].principal, principal) != 0) {
            pthread_mutex_unlock(&g_bindings_lock);
            return MS_NOT_YOURS;
        }
        mine = 1;
        if (strcmp(g_bindings[i].sid, key) == 0)
            exact = i;
    }
    if (use == MS_PRINCIPAL_EXPORT && !mine && !admin) {
        pthread_mutex_unlock(&g_bindings_lock);
        return MS_NOT_YOURS;
    }
    if (use == MS_PRINCIPAL_BIND) {
        if (exact < 0) {
            exact = principal_slot_locked();
            memcpy(g_bindings[exact].sid, key, sizeof(key));
            snprintf(g_bindings[exact].principal, MS_PRINCIPAL_MAX, "%s", principal);
        }
        g_bindings[exact].serial = entry ? entry->serial : 0;
    }
    if (exact >= 0)
        g_bindings[exact].last_ms = ms_now_ms();
    pthread_mutex_unlock(&g_bindings_lock);
    return MS_OK;
}

/* Does binding `b` hold a seat on `entry` (ms_seats_by_principal)? */
static bool binding_holds_seat(mr_entry *entry, const ms_binding *b)
{
    if (entry->sched)
        return bs_session_family_holds(entry->sched, b->sid);
    return b->serial == entry->serial;
}

int ms_seats_by_principal(mr_entry *entry, ms_principal_seats *out, int max)
{
    ms_binding *copy = NULL;
    bool *held = NULL;
    int n = 0, n_out = 0;

    if (!entry || !out || max <= 0)
        return 0;
    copy = malloc(sizeof(*copy) * MS_PRINCIPAL_BINDINGS_MAX);
    held = calloc(MS_PRINCIPAL_BINDINGS_MAX, sizeof(*held));
    if (!copy || !held)
        goto done;
    pthread_mutex_lock(&g_bindings_lock);
    n = g_n_bindings;
    memcpy(copy, g_bindings, sizeof(*copy) * (size_t)n);
    pthread_mutex_unlock(&g_bindings_lock);
    for (int i = 0; i < n; i++) {
        held[i] = binding_holds_seat(entry, &copy[i]);
        /* One family of one person is one seat, however many of its ids are bound. */
        for (int j = 0; held[i] && j < i; j++)
            if (held[j] && strcmp(copy[j].principal, copy[i].principal) == 0 &&
                pn_session_same_family(copy[j].sid, copy[i].sid))
                held[i] = false;
        if (!held[i])
            continue;
        int k = 0;
        while (k < n_out && strcmp(out[k].principal, copy[i].principal) != 0)
            k++;
        if (k == n_out && n_out < max) {
            memset(&out[k], 0, sizeof(out[k]));
            snprintf(out[k].principal, sizeof(out[k].principal), "%s", copy[i].principal);
            n_out++;
        }
        if (k < n_out)
            out[k].held++;
    }
done:
    free(copy);
    free(held);
    return n_out;
}

int ms_sessions_used(mr_entry *entry)
{
    if (!entry)
        return 0;
    if (entry->sched)
        return bs_session_count(entry->sched);
    if (entry->prefix_cache)
        return pn_prefix_cache_session_count(entry->prefix_cache);
    /* No session store (a CPU copy): a session is only its routing binding. */
    return pn_replica_affinity_count(entry->serial);
}

void ms_session_occupancy(const mr_entry *entry, ms_occupancy *out)
{
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    if (!entry)
        return;
    if (entry->sched) {
        bs_occupancy o;
        bs_session_occupancy(entry->sched, &o);
        out->sessions_used = o.families;
        out->max_sessions = o.max_sessions;
        out->slots_held = o.slots_held;
        out->slots_free = o.slots_free;
        out->seat_free = o.seat_free;
        return;
    }
    /* No decode slots: a seat is one session, so held == used. */
    out->max_sessions = entry->max_sessions;
    if (entry->prefix_cache) {
        out->sessions_used = pn_prefix_cache_session_count(entry->prefix_cache);
        out->max_sessions = pn_prefix_cache_max_sessions(entry->prefix_cache);
        out->seat_free = out->sessions_used < out->max_sessions;
    } else {
        out->sessions_used = pn_replica_affinity_count(entry->serial);
        out->seat_free = true;
    }
    out->slots_held = out->sessions_used;
    out->slots_free = out->max_sessions > out->sessions_used
                    ? out->max_sessions - out->sessions_used : 0;
}

bool ms_session_holds_seat(mr_entry *entry, const char *sid)
{
    if (!entry || !sid || !sid[0])
        return false;
    if (entry->sched)
        return bs_session_family_holds(entry->sched, sid);
    return true;
}

bool ms_session_seat_free(mr_entry *entry)
{
    if (!entry)
        return false;
    if (entry->sched)
        return bs_session_seat_free(entry->sched);
    if (entry->prefix_cache)   /* the cache's own cap is what admission enforces */
        return pn_prefix_cache_session_count(entry->prefix_cache) <
               pn_prefix_cache_max_sessions(entry->prefix_cache);
    return true;   /* nothing to hold: no session cap is enforced */
}
