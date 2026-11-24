/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_host_routes.h
 * @brief  The registry routes of a host (people registry spec §4.4), DEVICE class:
 *           POST /local/hosts                    register or update (owner = the caller)
 *           GET  /local/hosts/<key>/wait?since=n the beat: held by the hub up to 30 s
 *           POST /local/hosts/<key>/delete       the owner, from any device
 *         Plus the hooks other routes call when a person or a device key goes
 *         away, so the held waits hear `close`.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_HOST_ROUTES_H
#define PN_LOCAL_HOST_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer a /local/hosts request.
 * @return 0 when it is not one of these routes, 1 when answered, 2 when the hub
 *         adopted the fd (a held wait: the caller must not close it)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_host_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

/**
 * A person was removed: delete their rows (their waits get `close` `removed`)
 * and drop them from every share. Returns the rows deleted.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_hosts_person_removed(pn_local_ctx *ctx, const char *person);

/**
 * A device key was revoked: its held waits get `close` `revoked`.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_local_hosts_key_revoked(pn_local_ctx *ctx, const char *key_id);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HOST_ROUTES_H */
