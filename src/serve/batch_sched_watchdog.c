/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   batch_sched_watchdog.c
 * @brief  The watched token wait (see batch_sched_watchdog.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "batch_sched_watchdog.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#define BS_DECODE_STALL_MS_DEFAULT 120000
#define BS_POP_POLL_MS_DEFAULT     1000

int bs_decode_stall_ms(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_DECODE_STALL_MS");
        char *end = NULL;
        long v = env && *env ? strtol(env, &end, 10) : -1;
        cached = (env && *env && end && *end == '\0' && v >= 0 && v <= 86400000L)
               ? (int)v : BS_DECODE_STALL_MS_DEFAULT;
    }
    return cached;
}

static double monotonic_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* The slot cvs use the default (realtime) clock: deadline `ms` from now. */
static struct timespec realtime_deadline(int ms)
{
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    long ns = ts.tv_nsec + (long)ms * 1000000L;
    ts.tv_sec += ns / 1000000000L;
    ts.tv_nsec = ns % 1000000000L;
    return ts;
}

/* One look at the slot under mu: a queued token, an error, a finish, or
 * nothing yet (-9). The drain order is bs_slot_pop_token's. */
static int slot_ready_locked(const pn_batch_sched *sched, bs_slot *sl, uint32_t *tok)
{
    if (sl->ring && sl->ring_r < sl->ring_w) {
        *tok = sl->ring[sl->ring_r++];
        return BS_POP_TOKEN;
    }
    if (sched->shutdown || sl->step_error)
        return BS_POP_ERROR;
    if (sl->finished)
        return BS_POP_FINISHED;
    return -9;
}

bs_pop_status bs_slot_pop_token_watched(pn_batch_sched *sched, int idx, uint32_t *tok,
                                        const bs_pop_watch *watch)
{
    if (!sched || idx < 0 || idx >= sched->cap || !tok)
        return BS_POP_ERROR;
    int poll_ms = watch && watch->poll_ms > 0 ? watch->poll_ms : BS_POP_POLL_MS_DEFAULT;
    int stall_ms = watch ? watch->stall_ms : 0;
    double start = monotonic_ms();
    bs_slot *sl = &sched->slots[idx];

    pthread_mutex_lock(&sched->mu);
    for (;;) {
        int r = slot_ready_locked(sched, sl, tok);
        if (r != -9) {
            pthread_mutex_unlock(&sched->mu);
            return (bs_pop_status)r;
        }
        if (!watch) {
            pthread_cond_wait(&sl->cv, &sched->mu);
            continue;
        }
        struct timespec until = realtime_deadline(poll_ms);
        int rc = pthread_cond_timedwait(&sl->cv, &sched->mu, &until);
        if (rc != ETIMEDOUT)
            continue;   /* woken: look again */
        r = slot_ready_locked(sched, sl, tok);
        if (r != -9) {
            pthread_mutex_unlock(&sched->mu);
            return (bs_pop_status)r;
        }
        /* The abort check may take other locks: never under mu. */
        pthread_mutex_unlock(&sched->mu);
        if (watch->should_abort && watch->should_abort(watch->abort_ud))
            return BS_POP_ABORTED;
        if (stall_ms > 0 && monotonic_ms() - start >= (double)stall_ms)
            return BS_POP_STALLED;
        pthread_mutex_lock(&sched->mu);
    }
}
