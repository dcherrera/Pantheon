/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_registry.h
 * @brief Serve-side dispatch self-registration lifecycle (Phase G0 wiring).
 *        Ties the register / heartbeat / deregister client (registry_client.h)
 *        into the live serve process: one register per served model_id at
 *        startup, a background heartbeat thread reporting live load from
 *        batch_sched, and a best-effort deregister on controlled shutdown.
 *
 *        FULLY GATED on PANTHEON_REGISTRY_URL (via rc_enabled()): with it unset
 *        every function here is a no-op and the serve process is byte-for-byte
 *        unchanged (no registration, no heartbeat thread spawned, no network).
 *
 *        Identity semantics (see registry_wire.h): the shared `model_id` is the
 *        served name clients request (mr_info.name) — two serve processes
 *        hosting the same weights advertise the SAME model_id and become a
 *        router replica pool; the `instance_id` is unique per serve process
 *        ("<host>:<port>:<model>").
 *
 * @status G0 wiring. Live-registration behaviour is verified on a K80 host separately
 *         (no GPU/model on the Mac); the Mac build path exercises only the
 *         unset-env no-op and the compile.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_REGISTRY_H
#define PN_SERVE_REGISTRY_H

#include "model_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Startup registration: for each resident model in `reg`, build an rc_identity
 * (model_id = served name, unique instance_id = "<host>:<port>:<name>",
 * endpoint = "http://<host>:<port>", capacity from the model's batch_sched) and
 * rc_register() it, recording it for later heartbeat + deregister. Then spawn a
 * detached heartbeat thread that every RW_HEARTBEAT_SECONDS reports each
 * instance's live load (bs_get_load under mr_acquire/mr_release).
 *
 * No-op (does not touch `reg`, spawns no thread) when rc_enabled() is false.
 * Call once, after hs_listen succeeds and all models are loaded.
 *
 * `reg`, `host` must outlive the serve process (both are process-lifetime in
 * pn_serve_main); the heartbeat thread borrows `reg` and never frees it.
 *
 * Thread-safe: no (call once at startup, before the accept loop).
 */
void serve_registry_start(mr_registry *reg, const char *host, int port);

/**
 * Register a model loaded AFTER serve_registry_start() (an /admin/models/load),
 * using the host/port captured at start. Idempotent: a model whose instance is
 * already registered (e.g. a second replica of the same served name) is skipped.
 * No-op when rc_enabled() is false or serve_registry_start() has not run. The
 * running heartbeat thread picks the new instance up on its next cycle.
 *
 * Thread-safe: yes (guards the instance table; runs from the request thread).
 */
void serve_registry_add(const char *model_id);

/**
 * Deregister a model unloaded via /admin/models/unload — but only if no replica
 * of the same served name remains loaded (unloading one of two same-name
 * replicas keeps the shared instance registered). No-op when disabled/unstarted.
 *
 * Thread-safe: yes.
 */
void serve_registry_remove(const char *model_id);

/**
 * Best-effort deregister of every instance registered by
 * serve_registry_start(). Called from the SIGINT/SIGTERM shutdown handler so a
 * clean stop drops the instances from the router immediately (the registry TTL
 * also evicts them if this is skipped). No-op when rc_enabled() is false or
 * nothing was registered. Each rc_deregister uses a short connect timeout.
 *
 * Safe to call from the controlled-shutdown signal handler (same trade as the
 * CUDA teardown there: operator-initiated stop, not a crash mid-kernel).
 */
void serve_registry_deregister_all(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_REGISTRY_H */
