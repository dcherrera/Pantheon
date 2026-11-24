/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_load.h
 * @brief  Load a catalog model with a profile (spec G1/G2): resolve the profile,
 *         place it on free units (or check the units the caller picked), build
 *         the load (dies, ctx-cap, sessions, concurrency, reasoning budget,
 *         sampling defaults, MTP) and run it through the serve's own
 *         ma_load_model; remember it in the loaded set. A `cpu` profile is the
 *         host CPU forward in every build; an accelerator profile that lands on
 *         the CPU is unloaded and refused. Also the reload at boot, one model at
 *         a time, each re-checked against the hardware found now. Loads are
 *         serialized.
 *
 * @status None
 * @issues MTP from a profile is applied on CUDA builds (`--mtp`); a TT build
 *         follows its own shipped MTP default (on when the artifact has a head).
 * @todo   None
 */

#ifndef PN_LOCAL_LOAD_H
#define PN_LOCAL_LOAD_H

#include <stddef.h>

#include "loaded_set.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One load request (POST /local/models/load). */
typedef struct {
    const char *catalog_id, *profile_id;
    int         units[PN_UNIT_PLACE_MAX];   /* the caller's pick */
    int         n_units;                    /* 0 = place automatically */
    int         ctx_cap, max_sessions;      /* the load option; 0 = recommended */
    const char *loaded_by;                  /* the caller's person (may be NULL) */
} pn_local_load_request;

/**
 * Build the load: resolve the profile, choose the load option (it must be one
 * that fits the units free now, load_options.h), place it (or check the
 * caller's units) and number the replica (a model already loaded gets the
 * next number: another copy on other units, the same served name).
 * @param status  HTTP status for a refusal (400 / 404 / 409)
 * @return 0, or -1 with a reason
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_resolve_load(pn_local_ctx *ctx, const pn_local_load_request *rq,
                          pn_loaded_entry *out, int *status, char *err, size_t errlen);

/**
 * Resolve + place + load (one lock around all three) and remember the model.
 * @param status  HTTP status for a refusal (404 / 409, 507 when the load fails)
 * @return 0, or -1 with a reason
 *
 * Thread-safe: yes (serialized). Reentrant: no. Signal-safe: no
 */
int pn_local_load_catalog(pn_local_ctx *ctx, const pn_local_load_request *rq,
                          pn_loaded_entry *out, int *status, char *err, size_t errlen);

/**
 * Unload replica `replica` of catalog_id (0 = every replica), and anything else
 * the registry holds under that name when replica is 0. The remembered set
 * follows. *n_out = replicas unloaded.
 * @param status  HTTP status for a refusal (404 / 409 busy)
 * @return 0, or -1 with a reason (some replicas may be gone already)
 *
 * Thread-safe: yes (serialized with loads). Reentrant: no. Signal-safe: no
 */
int pn_local_unload(pn_local_ctx *ctx, const char *catalog_id, int replica, int *n_out,
                    int *status, char *err, size_t errlen);

/**
 * pn_local_unload on behalf of a non-admin person (spec §4.7): under the same
 * lock, every targeted replica must have been loaded by `person` and hold no
 * session, else nothing is unloaded and the answer is 403 with why. NULL person =
 * an admin: pn_local_unload.
 *
 * Thread-safe: yes (serialized with loads). Reentrant: no. Signal-safe: no
 */
int pn_local_unload_as(pn_local_ctx *ctx, const char *catalog_id, int replica,
                       const char *person, int *n_out, int *status, char *err, size_t errlen);

/**
 * The load running now (resolved, not yet in the loaded set).
 * @return 0, or -1 when no load runs
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_load_in_flight(pn_loaded_entry *out);

/**
 * Hold / release the lock every load and unload runs under, so another change
 * to the loaded models (an uninstall's check + removal) cannot interleave.
 *
 * Thread-safe: yes. Reentrant: no (not recursive). Signal-safe: no
 */
void pn_local_loads_lock(void);
void pn_local_loads_unlock(void);

/**
 * Give remembered replicas from before people (no `loaded_by`) to person 1, the
 * oldest admin (spec §4.7). Nothing happens on a box with no person yet.
 * @return 0, or -1 when the loaded set could not be saved
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int pn_local_loads_claim_unowned(pn_local_ctx *ctx);

/* What the boot reload did with a remembered model. */
typedef enum {
    PN_RELOAD_NONE = 0,   /* not part of a reload */
    PN_RELOAD_PENDING,    /* waiting for its turn */
    PN_RELOAD_LOADED,     /* loaded (by the reload or a later load) */
    PN_RELOAD_SKIPPED,    /* its profile does not fit the hardware found now */
    PN_RELOAD_FAILED,     /* the load itself failed */
} pn_reload_state;

#define PN_RELOAD_REASON_MAX 256

/**
 * Reload the remembered set in a background thread, in order, one at a time.
 * Each model's profile is re-checked against the hardware found now; one that no
 * longer fits is not loaded, stays remembered, and is reported (log +
 * pn_local_reload_state).
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
void pn_local_reload_async(pn_local_ctx *ctx);

/**
 * The same reload, on the calling thread.
 *
 * Thread-safe: yes (loads serialize). Reentrant: no. Signal-safe: no
 */
void pn_local_reload_now(pn_local_ctx *ctx);

/**
 * The reload outcome of catalog_id's replica; reason (may be NULL) gets why it was not
 * loaded ("" otherwise).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_reload_state pn_local_reload_state(const char *catalog_id, int replica, char *reason,
                                      size_t cap);

/**
 * Record a reload outcome (the reload, and a successful load, call this).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_reload_note(const char *catalog_id, int replica, pn_reload_state state,
                          const char *reason);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_LOAD_H */
