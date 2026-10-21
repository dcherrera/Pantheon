/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_affinity.h
 * @brief Session -> instance affinity map for the dispatch router (PC0.2).
 *        Remembers which serve instance last handled a given X-Session-Id so a
 *        multi-turn chat keeps landing on the SAME instance — where the serve's
 *        prefix cache already holds that session's prefilled KV. Without this,
 *        least-busy routing can send turn 2 to a different instance (a KV miss =
 *        a full cold prefill), defeating the prefix cache.
 *
 * Concurrency: one mutex guards the whole table, matching the registry's design
 * (registry.h). Operations are O(N) over a small N; all public calls take/return
 * copies so callers never hold pointers into the table across the lock.
 *
 * Lifecycle: a mapping is recorded on the first least-busy pick for a session and
 * refreshed on every hit. It is cleared when its instance is deregistered or
 * TTL-evicted (sa_forget_instance) so a dead instance never pins a session.
 *
 * @status PC0.2 complete: record/lookup/forget, mutex-guarded fixed table.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_SESSION_AFFINITY_H
#define PN_SERVE_SESSION_AFFINITY_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

#include "registry_wire.h"

/* Max distinct sessions tracked. A session entry is tiny; 1024 covers a large
 * fleet of concurrent chat tabs without dynamic growth. When full, the oldest
 * (least-recently-used) entry is evicted to admit a new session. */
#define SA_MAX_SESSIONS 1024

/* Max session-id length tracked (matches the serve prefix cache's 63+1 key). */
#define SA_SESSION_ID_MAX 64

/* One session -> instance binding. POD row; strings are fixed-size. */
typedef struct {
    char   session_id[SA_SESSION_ID_MAX];
    char   instance_id[RW_INSTANCE_ID_MAX];
    long   seq;    /* monotonic use counter for LRU eviction (higher = newer) */
} sa_entry;

/* The affinity map. Zero-initialize then sa_init(). */
typedef struct {
    pthread_mutex_t mu;
    sa_entry        ent[SA_MAX_SESSIONS];
    int             n;
    long            clock;   /* monotonic counter stamped onto seq on each touch */
} session_affinity;

/**
 * Initialize the map (mutex + empty table).
 * @return 0 on success, -1 if the mutex could not be created.
 * Thread-safe: no (call once at startup, before any worker threads)
 */
int sa_init(session_affinity *sa);

/** Destroy the map's mutex. Table memory is inline (no frees). NULL-safe. */
void sa_destroy(session_affinity *sa);

/**
 * Look up the instance a session is currently bound to. On a hit, copies the
 * bound instance_id into `out` (size `out_cap`), refreshes the entry's LRU
 * stamp, and returns true. Returns false if the session has no binding.
 *
 * @param session_id  the X-Session-Id (NULL/empty ⇒ always a miss)
 * @param out         buffer for the bound instance_id (>= RW_INSTANCE_ID_MAX)
 * @param out_cap     size of `out`
 * Thread-safe: yes (internal locking)
 */
bool sa_lookup(session_affinity *sa, const char *session_id,
               char *out, size_t out_cap);

/**
 * Record (or refresh) a session -> instance binding. Overwrites an existing
 * binding for the same session. When the table is full and the session is new,
 * evicts the least-recently-used entry to make room. A NULL/empty session_id or
 * instance_id is ignored (no-op).
 *
 * Thread-safe: yes (internal locking)
 */
void sa_record(session_affinity *sa, const char *session_id,
               const char *instance_id);

/**
 * Forget every binding pointing at `instance_id`. Called when an instance is
 * deregistered or TTL-evicted so a dead instance stops pinning its sessions
 * (they re-route least-busy on their next turn). A NULL/empty id is a no-op.
 *
 * @return number of bindings removed.
 * Thread-safe: yes (internal locking)
 */
int sa_forget_instance(session_affinity *sa, const char *instance_id);

#endif /* PN_SERVE_SESSION_AFFINITY_H */
