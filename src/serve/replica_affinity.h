/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   replica_affinity.h
 * @brief  Session -> replica binding inside one serve. A model loaded on several
 *         die-sets under one name (replicas) keeps each session's KV on the
 *         replica that ran its first turn; this table remembers which one
 *         (by mr_entry.serial), so later turns, its /session/watch and its
 *         export/restore/release go back there.
 *
 *         Keys match by base: a recorded key K matches a session id S when
 *         S == K, S starts with "K-", or K starts with "S-" (a client watches
 *         the base tab id while its turns run on "<base>-<mode>" ids). A
 *         binding ends when its session is released, its replica unloads, or
 *         the table is full (least recently used goes first).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_REPLICA_AFFINITY_H
#define PN_SERVE_REPLICA_AFFINITY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_REPLICA_AFFINITY_MAX 512
#define PN_REPLICA_SID_MAX      64

/**
 * The replica serial `sid` is bound to among `serials[0..n)`, or 0.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
uint64_t pn_replica_affinity_lookup(const char *sid, const uint64_t *serials, int n);

/**
 * Bind `sid` to `serial` (replaces an earlier binding of the same key).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_affinity_bind(const char *sid, uint64_t serial);

/**
 * Sessions bound to `serial`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_replica_affinity_count(uint64_t serial);

/**
 * Drop every binding matching `sid` (by the base rule above).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_affinity_forget(const char *sid);

/**
 * Drop exactly the binding of key `sid` to `serial` (no base matching): a
 * session a request bound and then refused before it ran.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_affinity_unbind(const char *sid, uint64_t serial);

/**
 * Move every binding matching `sid` (by the base rule above) from `from` to
 * `to`; bind `sid` to `to` when none matched. A session re-placed onto another
 * replica takes its whole family along.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_affinity_rebind(const char *sid, uint64_t from, uint64_t to);

/**
 * Drop every binding to `serial` (its replica unloaded).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_affinity_forget_serial(uint64_t serial);

/** Empty the table (tests). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
void pn_replica_affinity_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_REPLICA_AFFINITY_H */
