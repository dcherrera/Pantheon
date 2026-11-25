/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_admission.c
 * @brief Implementation of the batch scheduler's session admission — see
 *        batch_sched_admission.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "batch_sched_admission.h"

#include <stdio.h>
#include <string.h>

#include "../core/session_family.h"

const char *bs_slot_session_owner(const bs_slot *sl)
{
    if (sl->status != BS_SLOT_IDLE && sl->claim_sid[0])
        return sl->claim_sid;
    if (sl->sid[0])
        return sl->sid;
    if (sl->sys_sid[0])
        return sl->sys_sid;
    if (sl->ckpt.valid && sl->ckpt.sid[0])
        return sl->ckpt.sid;
    if (sl->partial.valid && sl->partial.sid[0])
        return sl->partial.sid;
    return NULL;
}

bool bs_family_holds_slot_locked(const pn_batch_sched *sched, const char *sid)
{
    for (int i = 0; i < sched->cap; i++) {
        const char *owner = bs_slot_session_owner(&sched->slots[i]);
        if (owner && pn_session_same_family(owner, sid))
            return true;
    }
    return false;
}

int bs_family_count_locked(const pn_batch_sched *sched)
{
    int n = 0;

    for (int i = 0; i < sched->cap; i++) {
        const char *owner = bs_slot_session_owner(&sched->slots[i]);
        bool first = owner != NULL;
        for (int j = 0; j < i && first; j++) {
            const char *prev = bs_slot_session_owner(&sched->slots[j]);
            if (prev && pn_session_same_family(prev, owner))
                first = false;
        }
        n += first;
    }
    return n;
}

int bs_unheld_count_locked(const pn_batch_sched *sched)
{
    int n = 0;

    for (int i = 0; i < sched->cap; i++)
        n += !bs_slot_session_owner(&sched->slots[i]);
    return n;
}

/* Is some slot unheld: idle now, or running a session-less turn, which leaves
 * it unheld?  A new family (or no session) may only ever run on such a slot:
 * every other slot stays its family's until released or reaped, so waiting
 * for one would only end in a timeout. */
static bool any_slot_unheld_locked(const pn_batch_sched *sched)
{
    return bs_unheld_count_locked(sched) > 0;
}

static int session_limit(const pn_batch_sched *sched)
{
    return sched->max_sessions > 0 ? sched->max_sessions : sched->cap;
}

bool bs_family_may_take_extra_locked(const pn_batch_sched *sched)
{
    return bs_unheld_count_locked(sched) - 1 >=
           session_limit(sched) - bs_family_count_locked(sched);
}

bool bs_session_family_holds(pn_batch_sched *sched, const char *sid)
{
    if (!sched || !sid || !sid[0])
        return false;
    pthread_mutex_lock(&sched->mu);
    bool holds = bs_family_holds_slot_locked(sched, sid);
    pthread_mutex_unlock(&sched->mu);
    return holds;
}

bool bs_admission_seat_free_locked(const pn_batch_sched *sched)
{
    return bs_family_count_locked(sched) < session_limit(sched) &&
           any_slot_unheld_locked(sched);
}

bs_acquire_result bs_admission_check_locked(const pn_batch_sched *sched,
                                            const char *sid)
{
    if (!sid || !sid[0])
        return any_slot_unheld_locked(sched) ? BS_ACQUIRE_OK
                                             : BS_ACQUIRE_AT_CAPACITY;
    if (bs_family_holds_slot_locked(sched, sid))
        return BS_ACQUIRE_OK;   /* an existing session always resumes */
    return bs_admission_seat_free_locked(sched) ? BS_ACQUIRE_OK
                                                : BS_ACQUIRE_AT_CAPACITY;
}

/* Is `sl` an idle slot of another family, untouched for at least the TTL?
 * Copies its holder into `owner`. Under mu. */
static bool reap_candidate_locked(const pn_batch_sched *sched,
                                  const bs_slot *sl, const char *sid,
                                  double now_ms, char owner[64])
{
    const char *held = bs_slot_session_owner(sl);

    if (sl->status != BS_SLOT_IDLE || !held || sl->last_active_ms <= 0.0 ||
        now_ms - sl->last_active_ms < (double)sched->session_idle_ttl_ms)
        return false;
    if (sid && sid[0] && pn_session_same_family(held, sid))
        return false;
    snprintf(owner, 64, "%s", held);
    return true;
}

void bs_admission_reap_idle(pn_batch_sched *sched, const char *sid,
                            double now_ms)
{
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        char owner[64], again[64];

        pthread_mutex_lock(&sched->mu);
        bool reap = sched->session_idle_ttl_ms > 0 &&
                    reap_candidate_locked(sched, sl, sid, now_ms, owner);
        pn_session_watched_fn watched = sched->watch_fn;
        void *watch_ctx = sched->watch_ctx;
        pthread_mutex_unlock(&sched->mu);
        /* The watch hook takes the watch manager's lock: never under mu. A
         * watched session is the watch's to release, not the backstop's. */
        if (!reap || (watched && watched(owner, watch_ctx)))
            continue;

        /* Re-check: the slot may have been claimed or touched meanwhile. */
        pthread_mutex_lock(&sched->mu);
        reap = reap_candidate_locked(sched, sl, sid, now_ms, again) &&
               strcmp(owner, again) == 0;
        if (reap)
            sl->status = BS_SLOT_PREFILL;   /* reserved: the fields are ours */
        double idle_ms = now_ms - sl->last_active_ms;
        pthread_mutex_unlock(&sched->mu);
        if (!reap)
            continue;

        bs_slot_drop_session(sched, sl);
        pthread_mutex_lock(&sched->mu);
        sl->status = BS_SLOT_IDLE;
        pthread_cond_broadcast(&sched->cv_idle);
        pthread_mutex_unlock(&sched->mu);
        fprintf(stderr, "[batch_sched] reaped idle session slot=%d sid=%s "
                "(idle %.0fms >= %llums, no watch)\n", i, owner, idle_ms,
                (unsigned long long)sched->session_idle_ttl_ms);
    }
}

/* Is `sl` an idle slot held by another family than `sid`'s? Copies its holder
 * into `owner`. Under mu. */
static bool evict_candidate_locked(const bs_slot *sl, const char *sid, char owner[64])
{
    const char *held = bs_slot_session_owner(sl);

    if (sl->status != BS_SLOT_IDLE || !held)
        return false;
    if (sid && sid[0] && pn_session_same_family(held, sid))
        return false;
    snprintf(owner, 64, "%s", held);
    return true;
}

bool bs_admission_evict_lru_idle(pn_batch_sched *sched, const char *sid)
{
    int order[BS_MAX_USERS];
    int n = 0;

    /* Candidates, least recently active first (cap is small: insertion sort). */
    pthread_mutex_lock(&sched->mu);
    for (int i = 0; i < sched->cap && n < BS_MAX_USERS; i++) {
        char owner[64];
        if (!evict_candidate_locked(&sched->slots[i], sid, owner))
            continue;
        int j = n++;
        while (j > 0 && sched->slots[order[j - 1]].last_active_ms >
                        sched->slots[i].last_active_ms) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = i;
    }
    pn_session_watched_fn watched = sched->watch_fn;
    void *watch_ctx = sched->watch_ctx;
    pthread_mutex_unlock(&sched->mu);

    for (int k = 0; k < n; k++) {
        bs_slot *sl = &sched->slots[order[k]];
        char owner[64], again[64];

        pthread_mutex_lock(&sched->mu);
        bool ok = evict_candidate_locked(sl, sid, owner);
        pthread_mutex_unlock(&sched->mu);
        /* The watch hook takes the watch manager's lock: never under mu. */
        if (!ok || (watched && watched(owner, watch_ctx)))
            continue;
        pthread_mutex_lock(&sched->mu);
        ok = evict_candidate_locked(sl, sid, again) && strcmp(owner, again) == 0;
        if (ok)
            sl->status = BS_SLOT_PREFILL;   /* reserved: the fields are ours */
        pthread_mutex_unlock(&sched->mu);
        if (!ok)
            continue;

        bs_slot_drop_session(sched, sl);
        pthread_mutex_lock(&sched->mu);
        sl->status = BS_SLOT_IDLE;
        pthread_cond_broadcast(&sched->cv_idle);
        pthread_mutex_unlock(&sched->mu);
        /* No session id here: in gateway mode ids are customer data. */
        fprintf(stderr, "[batch_sched] evicted an idle session seat slot=%d "
                "(seats are evictable)\n", order[k]);
        return true;
    }
    return false;
}
