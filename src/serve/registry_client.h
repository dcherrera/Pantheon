/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file registry_client.h
 * @brief Serve-side self-registration client (Phase G0). A Pantheon serve
 *        process uses this to announce its model instances to the dispatch
 *        registry, heartbeat its live load, and deregister on shutdown.
 *
 *        GATED behind the PANTHEON_REGISTRY_URL environment variable: if it is
 *        unset, EVERY call here is a no-op and serve behaves exactly as today.
 *        This is why wiring it into serve_command.c is safe — with no env set,
 *        nothing changes. (This build ships the client + documents the hook
 *        points; it does NOT live-integrate it into the running serve loop.)
 *
 * Wire protocol: see registry_wire.h. The endpoints hit are
 *   POST   $PANTHEON_REGISTRY_URL/registry/register
 *   POST   $PANTHEON_REGISTRY_URL/registry/heartbeat
 *   DELETE $PANTHEON_REGISTRY_URL/registry/deregister
 *
 * @status G0 client complete + unit-tested against a stub registry; the
 *         serve_command.c startup/heartbeat/SIGTERM wiring is documented but
 *         intentionally NOT applied in this build (a separate human step).
 * @issues None
 * @todo Wire the three call sites in serve_command.c (see the handoff doc).
 */

#ifndef PN_SERVE_REGISTRY_CLIENT_H
#define PN_SERVE_REGISTRY_CLIENT_H

#include <stdbool.h>

#include "registry_wire.h"

/* Identity + reachability of this serve instance, filled once at startup. */
typedef struct {
    char model_id[RW_MODEL_ID_MAX];       /* the shared model identity */
    char instance_id[RW_INSTANCE_ID_MAX]; /* unique per serve process */
    char node_id[RW_NODE_ID_MAX];
    char endpoint[RW_ENDPOINT_MAX];       /* how the router reaches THIS serve */
    char transport[RW_TRANSPORT_MAX];     /* "local" | "wg" */
    int  capacity;                        /* batch-slot count */
    int  context_window;                  /* effective per-session context window
                                           * in tokens (pn_effective_ctx_window)
                                           * at registration; 0 = don't report */
} rc_identity;

/* The live load to report on a heartbeat (read from batch_sched). */
typedef struct {
    int  active_slots;
    int  queue_depth;
    int  free_slots;
    int  context_window;   /* effective per-session context window in tokens
                            * RIGHT NOW; 0 = don't report (leaves the registry's
                            * last known value alone). Reported per heartbeat,
                            * not just at register, because the memory ladder can
                            * move it at runtime. */
    bool healthy;
} rc_load;

/**
 * @return true if PANTHEON_REGISTRY_URL is set (⇒ registration is active). When
 *         false, all rc_* calls below are no-ops that return true (success).
 */
bool rc_enabled(void);

/**
 * POST /registry/register with the instance identity. No-op (returns true) if
 * registration is disabled. Returns false only on a real network/HTTP failure
 * while enabled.
 */
bool rc_register(const rc_identity *id);

/** POST /registry/heartbeat with the current load. Same no-op/failure rules. */
bool rc_heartbeat(const char *instance_id, const rc_load *load);

/** DELETE /registry/deregister for a graceful stop. Same rules. */
bool rc_deregister(const char *instance_id);

#endif /* PN_SERVE_REGISTRY_CLIENT_H */
