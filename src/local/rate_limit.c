/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   rate_limit.c
 * @brief  Implementation of the per-key token buckets (see rate_limit.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "rate_limit.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char   id[24];
    double tokens;
    double last;     /* refill time, also the LRU stamp */
    bool   used;
} bucket;

struct pn_rate_limiter {
    pthread_mutex_t lock;
    double          rate, burst;
    bucket          buckets[PN_RATE_MAX_BUCKETS];
};

double pn_rate_now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

pn_rate_limiter *pn_rate_limiter_new(double per_second, double burst)
{
    pn_rate_limiter *rl = calloc(1, sizeof(*rl));

    if (!rl)
        return NULL;
    pthread_mutex_init(&rl->lock, NULL);
    rl->rate = per_second;
    rl->burst = burst;
    return rl;
}

void pn_rate_limiter_free(pn_rate_limiter *rl)
{
    if (!rl)
        return;
    pthread_mutex_destroy(&rl->lock);
    free(rl);
}

/* The bucket for id, creating one (or recycling the stalest). Lock held. */
static bucket *find_bucket(pn_rate_limiter *rl, const char *id, double now)
{
    bucket *victim = &rl->buckets[0];

    for (int i = 0; i < PN_RATE_MAX_BUCKETS; i++) {
        bucket *b = &rl->buckets[i];
        if (b->used && strcmp(b->id, id) == 0)
            return b;
        if (!b->used || (victim->used && b->last < victim->last))
            victim = b;
    }
    memset(victim, 0, sizeof(*victim));
    snprintf(victim->id, sizeof(victim->id), "%s", id);
    victim->tokens = rl->burst;
    victim->last = now;
    victim->used = true;
    return victim;
}

bool pn_rate_limiter_take(pn_rate_limiter *rl, const char *id, double now_s)
{
    bool ok = false;

    pthread_mutex_lock(&rl->lock);
    bucket *b = find_bucket(rl, id, now_s);
    if (now_s > b->last) {
        b->tokens += (now_s - b->last) * rl->rate;
        if (b->tokens > rl->burst)
            b->tokens = rl->burst;
        b->last = now_s;
    }
    if (b->tokens >= 1.0) {
        b->tokens -= 1.0;
        ok = true;
    }
    pthread_mutex_unlock(&rl->lock);
    return ok;
}
