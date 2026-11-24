/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_session_routes.h
 * @brief  The registry routes of a client (people registry spec §4.5-4.6),
 *         DEVICE class:
 *           GET  /local/sessions[?rev=n]        own rows + rows shared with the caller
 *           POST /local/sessions/<key>/share    the owner shares, changes or unshares
 *           POST /local/sessions/<key>/attach   an attach ticket for the caller's role
 *         The host side (register, wait, delete) is local_host_routes.h.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_SESSION_ROUTES_H
#define PN_LOCAL_SESSION_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer a /local/sessions request.
 * @return 0 when it is not one of these routes, 1 when answered
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_session_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_SESSION_ROUTES_H */
