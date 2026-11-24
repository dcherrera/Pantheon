/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   rate_limit.h
 * @brief  Per-key token buckets (spec §7: rate limits per key). One bucket per
 *         id string (a key id, or a fixed name such as "pair"); the table is
 *         fixed-size and the least recently used bucket is recycled.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_RATE_LIMIT_H
#define PN_LOCAL_RATE_LIMIT_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_RATE_MAX_BUCKETS 128

typedef struct pn_rate_limiter pn_rate_limiter;

/**
 * @param per_second  sustained refill rate
 * @param burst       bucket size
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_rate_limiter *pn_rate_limiter_new(double per_second, double burst);

/** NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_rate_limiter_free(pn_rate_limiter *rl);

/**
 * Take one token from `id`'s bucket at time `now_s` (seconds, monotonic).
 * @return true if allowed, false if the bucket is empty
 *
 * Thread-safe: yes (internal lock)
 * Reentrant:   yes
 * Signal-safe: no
 */
bool pn_rate_limiter_take(pn_rate_limiter *rl, const char *id, double now_s);

/** Monotonic seconds, for callers. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
double pn_rate_now(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_RATE_LIMIT_H */
