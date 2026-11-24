/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   used_nonces.h
 * @brief  A bounded set of used one-time nonces, each remembered until its own
 *         expiry. The relay (people registry spec §4.6) opens at most one relay
 *         per attach-ticket nonce, so it keeps every nonce it has seen until
 *         the ticket can no longer be accepted (exp + skew). Expired entries
 *         are reused; a set whose entries are all live refuses new ones rather
 *         than forget a live nonce.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_USED_NONCES_H
#define PN_LOCAL_USED_NONCES_H

#ifdef __cplusplus
extern "C" {
#endif

#define PN_USED_NONCE_MAX 33   /* 32 hex + NUL */

typedef enum {
    PN_USED_NONCE_OK     = 0,
    PN_USED_NONCE_SEEN   = -1,   /* already used, and not expired */
    PN_USED_NONCE_FULL   = -2,   /* every entry is still live */
    PN_USED_NONCE_E_ARGS = -3,   /* empty or too long */
} pn_used_nonce_result;

typedef struct pn_used_nonces pn_used_nonces;

/**
 * A set of at most `max` live nonces. NULL on OOM or max < 1. Free with
 * pn_used_nonces_free.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_used_nonces *pn_used_nonces_new(int max);

/** Free the set. Thread-safe: no. Reentrant: yes. Signal-safe: no */
void pn_used_nonces_free(pn_used_nonces *set);

/**
 * Check `nonce` was not used and mark it used until `until` (unix s). An entry
 * whose `until` < `now` is expired: it no longer counts and its room is reused.
 * @return PN_USED_NONCE_OK (marked), _SEEN, _FULL, or _E_ARGS
 * Thread-safe: no (the caller locks). Reentrant: no. Signal-safe: no
 */
int pn_used_nonces_mark(pn_used_nonces *set, const char *nonce, long long until, long long now);

/** 1 when `nonce` is used and live at `now`, else 0. Thread-safe: no (the caller locks). Reentrant: no. Signal-safe: no */
int pn_used_nonces_seen(const pn_used_nonces *set, const char *nonce, long long now);

/** Live entries at `now`. Thread-safe: no (the caller locks). Reentrant: no. Signal-safe: no */
int pn_used_nonces_live(const pn_used_nonces *set, long long now);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_USED_NONCES_H */
