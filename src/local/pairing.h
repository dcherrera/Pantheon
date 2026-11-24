/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pairing.h
 * @brief  The pairing code (spec §7, D3): 8 random digits, made at start, printed
 *         once to the log when it is made, and written to <state-dir>/pair_code
 *         (0600) for `pantheon local pair-code`. It rotates after a successful
 *         pair, after `ttl` seconds, and after too many wrong tries. No network
 *         route ever returns it.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_PAIRING_H
#define PN_LOCAL_PAIRING_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_PAIR_CODE_DIGITS      8
#define PN_PAIR_DEFAULT_TTL_S    600   /* 10 min */
#define PN_PAIR_MAX_FAILURES     5     /* wrong tries before the code rotates */

typedef struct pn_pairing pn_pairing;

/**
 * Make the first code and write it. With run_thread, a 1 s ticker rotates it
 * on expiry (tests drive pn_pairing_tick by hand instead).
 * @return the pairing state, or NULL (file or RNG failure)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_pairing *pn_pairing_start(const char *code_path, int ttl_s, int run_thread);

/** Stop the ticker, delete the code file, free. NULL-safe.
 *  Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_pairing_stop(pn_pairing *p);

/**
 * Try a code (spaces and dashes ignored). A match rotates the code; a miss
 * counts, and PN_PAIR_MAX_FAILURES misses rotate it too (the caller then voids
 * every live invite, people registry spec §3.3).
 * @return 0 on a match, -2 on the miss that rotated the code, -1 on another miss
 *
 * Thread-safe: yes (internal lock)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_pairing_redeem(pn_pairing *p, const char *code);

/**
 * Forget the misses so far (an invite was redeemed: misses count since the last
 * success, box-wide).
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
void pn_pairing_clear_misses(pn_pairing *p);

/**
 * Rotate if the code is older than its ttl at time `now`.
 * @return 1 if it rotated, 0 otherwise
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_pairing_tick(pn_pairing *p, time_t now);

/** Copy the current code (tests). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
void pn_pairing_current(pn_pairing *p, char out[PN_PAIR_CODE_DIGITS + 1]);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_PAIRING_H */
