/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_session.c
 * @brief The session-state facade — see model_session.h for the contract and
 *        for why serve code must not reach past it into `entry->prefix_cache`
 *        or `entry->sched`.
 *
 * Lock discipline (identical on every operation here):
 *   - BATCHED path: the scheduler owns the state, and `bs_session_*` take
 *     `sched->mu` + the GPU turnstile themselves.  `pc_lock` is still taken
 *     around the prefix-cache bookkeeping calls (admit/touch) because those
 *     run concurrently from request threads.
 *   - SINGLE-USER path: `gen_lock` serializes the cache ops against that
 *     model's generation (the cache is documented thread-unsafe and the
 *     single-user generation mutates the very states we touch).
 */

#include "model_session.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "batch_sched.h"
#include "../core/prefix_cache.h"
#include "../core/session_snapshot.h"

uint64_t ms_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
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
                   ? (uint64_t)v : 12u * 60u * 60u * 1000u;
        } else {
            cached = 12u * 60u * 60u * 1000u;   /* 12 hours */
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

void ms_session_release(mr_entry *entry, const char *sid)
{
    if (!entry || !sid || !sid[0])
        return;
    if (entry->sched)
        bs_session_release(entry->sched, sid);
    if (!entry->prefix_cache)
        return;
    /* The prefix cache still holds this model's admission bookkeeping on the
     * batched path (and the whole session on the single-user path), so clear
     * it either way. */
    pthread_mutex_t *lock = entry->sched ? &entry->pc_lock : &entry->gen_lock;
    pthread_mutex_lock(lock);
    pn_prefix_cache_release_session(entry->prefix_cache, sid);
    pthread_mutex_unlock(lock);
}

void ms_session_release_by_prefix(mr_entry *entry, const char *base_prefix)
{
    if (!entry || !base_prefix || !base_prefix[0])
        return;
    if (entry->sched)
        bs_session_release_by_prefix(entry->sched, base_prefix);
    if (!entry->prefix_cache)
        return;
    pthread_mutex_t *lock = entry->sched ? &entry->pc_lock : &entry->gen_lock;
    pthread_mutex_lock(lock);
    pn_prefix_cache_release_by_prefix(entry->prefix_cache, base_prefix);
    pthread_mutex_unlock(lock);
}
