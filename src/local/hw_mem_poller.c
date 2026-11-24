/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hw_mem_poller.c
 * @brief  Implementation of the background device memory reads (see
 *         hw_mem_poller.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "hw_mem_poller.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

typedef struct {
    bool          ok;
    pn_hw_dev_mem mem;
    uint64_t      read_ms;        /* monotonic time of the reading */
} die_reading;

static struct {
    pthread_mutex_t  lock;
    pthread_cond_t   wake;        /* the thread: a kick or stop */
    pthread_cond_t   done;        /* waiters: a pass finished */
    pthread_t        thread;
    bool             running, stop;
    pn_hw_family     family;
    int              n_dies, interval_ms;
    pn_hw_mem_reader reader;
    die_reading      dies[PN_HW_MAX_UNITS];
    uint64_t         kicks;       /* kicks asked */
    uint64_t         kicks_done;  /* kicks covered by a finished pass */
    uint64_t         passes;
    uint64_t         pass_ms;     /* last pass length */
} g = { .lock = PTHREAD_MUTEX_INITIALIZER, .wake = PTHREAD_COND_INITIALIZER,
        .done = PTHREAD_COND_INITIALIZER };

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* A deadline `ms` from now on the clock pthread_cond_timedwait uses. */
static struct timespec deadline_in(uint64_t ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += (time_t)(ms / 1000u);
    ts.tv_nsec += (long)(ms % 1000u) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    return ts;
}

/* One pass over every die, the lock released while reading. */
static void read_all_dies(void)
{
    uint64_t kicks = g.kicks, start = now_ms();

    for (int die = 0; die < g.n_dies; die++) {
        die_reading r = { 0 };
        pthread_mutex_unlock(&g.lock);
        r.ok = g.reader(g.family, die, &r.mem) == 0;
        r.read_ms = now_ms();
        pthread_mutex_lock(&g.lock);
        g.dies[die] = r;
    }
    g.pass_ms = now_ms() - start;
    g.kicks_done = kicks;
    g.passes++;
    pthread_cond_broadcast(&g.done);
}

static void *poller_main(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&g.lock);
    while (!g.stop) {
        uint64_t began = now_ms();
        read_all_dies();
        uint64_t gap = (uint64_t)g.interval_ms;
        if (gap < g.pass_ms * PN_HW_POLL_DUTY_DIVISOR)
            gap = g.pass_ms * PN_HW_POLL_DUTY_DIVISOR;
        struct timespec until = deadline_in(gap > now_ms() - began ? gap - (now_ms() - began) : 0);
        while (!g.stop && g.kicks == g.kicks_done &&
               pthread_cond_timedwait(&g.wake, &g.lock, &until) != ETIMEDOUT)
            ;
    }
    pthread_mutex_unlock(&g.lock);
    return NULL;
}

int pn_hw_mem_poller_start(pn_hw_family family, int n_dies, pn_hw_mem_reader reader,
                           int interval_ms)
{
    if (g.running || !reader || n_dies <= 0 || n_dies > PN_HW_MAX_UNITS || interval_ms <= 0)
        return -1;
    pthread_mutex_lock(&g.lock);
    memset(g.dies, 0, sizeof(g.dies));
    g.family = family;
    g.n_dies = n_dies;
    g.reader = reader;
    g.interval_ms = interval_ms;
    g.stop = false;
    g.kicks = g.kicks_done = g.passes = 0;
    uint64_t passes = g.passes;
    if (pthread_create(&g.thread, NULL, poller_main, NULL) != 0) {
        pthread_mutex_unlock(&g.lock);
        return -1;
    }
    g.running = true;
    while (g.passes == passes)
        pthread_cond_wait(&g.done, &g.lock);
    pthread_mutex_unlock(&g.lock);
    return 0;
}

void pn_hw_mem_poller_stop(void)
{
    if (!g.running)
        return;
    pthread_mutex_lock(&g.lock);
    g.stop = true;
    pthread_cond_broadcast(&g.wake);
    pthread_mutex_unlock(&g.lock);
    pthread_join(g.thread, NULL);
    pthread_mutex_lock(&g.lock);
    g.running = false;
    pthread_cond_broadcast(&g.done);
    pthread_mutex_unlock(&g.lock);
}

int pn_hw_mem_poller_running(void)
{
    pthread_mutex_lock(&g.lock);
    int r = g.running;
    pthread_mutex_unlock(&g.lock);
    return r;
}

int pn_hw_mem_poller_get(int die, pn_hw_dev_mem *out, uint64_t *age_ms)
{
    int rc = -1;

    pthread_mutex_lock(&g.lock);
    if (g.running && die >= 0 && die < g.n_dies && g.dies[die].ok) {
        *out = g.dies[die].mem;
        if (age_ms)
            *age_ms = now_ms() - g.dies[die].read_ms;
        rc = 0;
    }
    pthread_mutex_unlock(&g.lock);
    return rc;
}

void pn_hw_mem_poller_kick(void)
{
    pthread_mutex_lock(&g.lock);
    g.kicks++;
    pthread_cond_broadcast(&g.wake);
    pthread_mutex_unlock(&g.lock);
}

/* The oldest reading's time, 0 when a die has none. g.lock held. */
static uint64_t oldest_read_ms(void)
{
    uint64_t oldest = UINT64_MAX;

    for (int die = 0; die < g.n_dies; die++)
        if (g.dies[die].read_ms < oldest)
            oldest = g.dies[die].read_ms;
    return oldest == UINT64_MAX ? 0 : oldest;
}

int pn_hw_mem_poller_ensure_fresh(int max_age_ms, int timeout_ms)
{
    int rc = 0;

    pthread_mutex_lock(&g.lock);
    if (!g.running) {
        pthread_mutex_unlock(&g.lock);
        return -1;
    }
    if (g.kicks == g.kicks_done && now_ms() - oldest_read_ms() <= (uint64_t)max_age_ms) {
        pthread_mutex_unlock(&g.lock);
        return 0;
    }
    uint64_t want = ++g.kicks;
    struct timespec until = deadline_in((uint64_t)timeout_ms);
    pthread_cond_broadcast(&g.wake);
    while (g.running && g.kicks_done < want && rc == 0)
        if (pthread_cond_timedwait(&g.done, &g.lock, &until) == ETIMEDOUT)
            rc = -1;
    pthread_mutex_unlock(&g.lock);
    return rc;
}

uint64_t pn_hw_mem_poller_passes(void)
{
    pthread_mutex_lock(&g.lock);
    uint64_t n = g.passes;
    pthread_mutex_unlock(&g.lock);
    return n;
}
