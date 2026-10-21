/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file registry.h
 * @brief In-memory live-instance registry for the dispatch layer (Phase G1).
 *        A thread-safe table of serve instances keyed by `instance_id` and
 *        indexed (by scan) on `model_id`. HTTP handler threads register /
 *        heartbeat / deregister; a background TTL sweep evicts instances that
 *        stopped heartbeating. The router (G2) reads it to pick the least-busy
 *        live instance of a requested model.
 *
 * Concurrency: one mutex guards the whole table. Operations are O(N) over a
 * small N (a handful of serve processes), so a single lock is simplest and
 * correct; no per-row locking needed. All public calls take/return copies —
 * callers never hold pointers into the table across the lock.
 *
 * @status G1 complete: register/heartbeat/deregister, TTL eviction, deduped
 *         /v1/models emit, model-level /registry/fleet emit, least-busy pick.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_REGISTRY_H
#define PN_SERVE_REGISTRY_H

#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include "registry_wire.h"

/* Max live instances tracked. The local fleet is ~5 models × a few replicas;
 * 256 leaves ample headroom for multi-node federation without dynamic growth. */
#define REG_MAX_INSTANCES 256

/* One live serve instance. POD row; strings are fixed-size (see registry_wire). */
typedef struct {
    char   model_id[RW_MODEL_ID_MAX];
    char   instance_id[RW_INSTANCE_ID_MAX];
    char   node_id[RW_NODE_ID_MAX];
    char   endpoint[RW_ENDPOINT_MAX];      /* upstream base URL to proxy to */
    char   transport[RW_TRANSPORT_MAX];    /* "local" | "wg" */
    int    capacity;                       /* total batch slots */

    /* Live load (from the last heartbeat). */
    int    active_slots;
    int    queue_depth;
    int    free_slots;
    int    context_window;                 /* effective per-session context window
                                            * in tokens as of the last register /
                                            * heartbeat; 0 = not reported. Sits
                                            * with the LOAD fields, not the
                                            * identity ones, because the memory
                                            * ladder can move it at runtime. */
    bool   healthy;

    time_t last_seen;                      /* wall-clock of last register/heartbeat */
} reg_instance;

/* The registry. Zero-initialize then reg_init(). */
typedef struct {
    pthread_mutex_t mu;
    reg_instance    inst[REG_MAX_INSTANCES];
    int             n;
    /* Monotonic counter used only to break least-busy ties with round-robin
     * (incremented each pick so equal-load replicas alternate). */
    unsigned        rr_counter;
} reg_registry;

/* Result of a register/heartbeat/deregister parse+apply. */
typedef enum {
    REG_OK = 0,
    REG_BAD_REQUEST = -1,   /* malformed JSON / missing required field */
    REG_FULL = -2,          /* table is full (register only) */
    REG_NOT_FOUND = -3,     /* instance_id unknown (heartbeat/deregister) */
} reg_status;

/** Initialize the registry (mutex + empty table). Returns 0, or -1 on error. */
int reg_init(reg_registry *r);

/** Destroy the registry mutex. Table memory is inline (no frees). */
void reg_destroy(reg_registry *r);

/**
 * Apply a POST /registry/register body (JSON). Inserts a new instance or
 * updates an existing one with the same instance_id (idempotent re-register).
 * Required fields: model_id, instance_id, endpoint. Optional: node_id,
 * transport (default "local"), capacity (default 0). Sets last_seen = now.
 */
reg_status reg_register_json(reg_registry *r, const char *body);

/**
 * Apply a POST /registry/heartbeat body (JSON). Updates load + healthy +
 * last_seen for the named instance_id. Required: instance_id. load{} and
 * healthy are optional (missing load leaves the prior values).
 */
reg_status reg_heartbeat_json(reg_registry *r, const char *body);

/**
 * Apply a DELETE /registry/deregister body (JSON). Removes the instance.
 * Required: instance_id.
 */
reg_status reg_deregister_json(reg_registry *r, const char *body);

/**
 * Like reg_deregister_json but, on a successful removal, copies the removed
 * instance_id into `removed_out` (size `out_cap`). Lets the router drop that
 * instance's session-affinity bindings. `removed_out` may be NULL to ignore it.
 *
 * Thread-safe: yes (internal locking)
 */
reg_status reg_deregister_json_collect(reg_registry *r, const char *body,
                                       char *removed_out, size_t out_cap);

/**
 * Evict every instance whose last_seen is older than RW_TTL_SECONDS. Called
 * periodically by the TTL sweep thread (and usable directly in tests).
 * @param now  the reference time (usually time(NULL)); passed in for testability.
 * @return number of instances evicted.
 */
int reg_sweep(reg_registry *r, time_t now);

/**
 * Like reg_sweep but appends each evicted instance_id into the caller-provided
 * `evicted_out` array (each slot RW_INSTANCE_ID_MAX bytes), up to `max_out`
 * entries. Lets the router drop those instances' session-affinity bindings.
 * Eviction still proceeds for instances past `max_out` — only the reporting is
 * capped. `evicted_out` may be NULL (then this behaves like reg_sweep).
 *
 * @param evicted_out  array of char[RW_INSTANCE_ID_MAX], or NULL
 * @param max_out      capacity of evicted_out in entries
 * @param n_out        if non-NULL, receives how many ids were written
 * @return total number of instances evicted (may exceed n_out).
 * Thread-safe: yes (internal locking)
 */
int reg_sweep_collect(reg_registry *r, time_t now,
                      char (*evicted_out)[RW_INSTANCE_ID_MAX], int max_out,
                      int *n_out);

/**
 * Emit the deduped GET /v1/models body into a malloc'd, NUL-terminated string
 * (caller frees). One entry per model_id, with a live-instance count and
 * aggregate free capacity. Returns NULL on OOM.
 */
char *reg_models_json(reg_registry *r);

/**
 * Emit the GET /registry/fleet body into a malloc'd string (caller frees).
 * Model-level: per model_id {live, instances, total_free_slots, nodes[]}.
 * Returns NULL on OOM.
 */
char *reg_fleet_json(reg_registry *r);

/**
 * Pick the least-busy live instance of `model_id`, skipping any instance whose
 * instance_id is in `exclude` (for failover retries; may be NULL/empty).
 * Selection: max free_slots, then min queue_depth, tiebreak round-robin. Only
 * healthy instances within TTL are considered.
 *
 * On success copies the chosen instance into *out and returns true. Returns
 * false if no eligible instance exists (⇒ router responds 503).
 *
 * @param exclude    array of instance_id strings to skip (already-failed picks)
 * @param n_exclude  length of exclude
 */
bool reg_pick_least_busy(reg_registry *r, const char *model_id,
                         const char *const *exclude, int n_exclude,
                         reg_instance *out);

/**
 * Look up a SPECIFIC live instance by instance_id, requiring it to serve
 * `model_id` and be within TTL + healthy. On success copies it into *out and
 * returns true. Used for session affinity: route a session back to the exact
 * instance holding its prefilled prefix cache. Returns false if the instance is
 * unknown, serves a different model, or is not live (⇒ caller falls back to
 * least-busy).
 *
 * @param model_id     the model the request is for (must match)
 * @param instance_id  the instance to look up
 * Thread-safe: yes (internal locking)
 */
bool reg_find_live(reg_registry *r, const char *model_id,
                   const char *instance_id, reg_instance *out);

#endif /* PN_SERVE_REGISTRY_H */
