/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_auth.h
 * @brief  The appliance's HTTP auth hook (hs_auth_fn; people registry spec §4.1).
 *         Every route has a class; every class but NONE needs a key, sent as
 *         `Authorization: Bearer pnk_...`, and rights come from the key's person:
 *           - NONE:      OPTIONS (CORS preflight) and POST /local/pair;
 *           - INFERENCE: any key: /v1/..., /session/..., GET /, and POST
 *                        /admin/session/export (the serve refuses another
 *                        person's session, model_session.h);
 *           - DEVICE:    a device key of any person (the box, me, devices,
 *                        people list, invites, keys, hosts, sessions, relay);
 *           - LOADER:    a device key whose person may load (admins always):
 *                        install, load, and /admin/models/unload (whose
 *                        route adds the §4.7 rule for non-admins);
 *           - ADMIN:     a device key of an admin: everything else.
 *         Missing/unknown key (or a removed person) → 401; wrong class → 403
 *         `forbidden`; LOADER without may_load → 403 `may_load_off`; over the
 *         per-key rate → 429. The resolved caller goes to local_caller.h, and
 *         its person to the serve's thread principal (hs_set_thread_principal).
 *         Admin actions are logged with the key id (never the key).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_AUTH_H
#define PN_LOCAL_AUTH_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-key limits: sustained requests/second and burst. */
#define PN_LOCAL_RATE_PER_S  10.0
#define PN_LOCAL_RATE_BURST  60.0
/* Pair attempts, box-wide (the code is the only thing guarding the box). */
#define PN_LOCAL_PAIR_PER_S  0.2
#define PN_LOCAL_PAIR_BURST  5.0

/* Who may call a route (spec §4.1), in increasing rights. */
typedef enum {
    PN_ROUTE_NONE = 0,
    PN_ROUTE_INFERENCE,
    PN_ROUTE_DEVICE,
    PN_ROUTE_LOADER,
    PN_ROUTE_ADMIN,
} pn_local_route_class;

/**
 * The class of a route (`path` may carry a query string).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
pn_local_route_class pn_local_route_class_of(const char *method, const char *path);

/** "none" / "inference" / "device" / "loader" / "admin".
 *  Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const char *pn_local_route_class_name(pn_local_route_class cls);

/**
 * hs_auth_fn for the appliance; `user` is the pn_local_ctx. Clears this
 * thread's caller, then sets it when the request passes with a key.
 * @return 0 to route the request, 1 if it was refused (response sent)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_auth_check(int client_fd, const hs_request *req, void *user);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_AUTH_H */
