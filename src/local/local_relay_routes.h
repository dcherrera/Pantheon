/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_relay_routes.h
 * @brief  The relay routes (people registry spec §4.6), DEVICE class, used when a
 *         client cannot reach a host directly:
 *           POST /local/relay/<key>       the client, with its attach ticket: the
 *                                         box checks the ticket, holds the fd and
 *                                         queues `relay {id}` on the host's wait
 *           POST /local/relay/<id>/host   the row's owner (the host) joins
 *         Both legs then read the 200 header block and the hub pipes bytes
 *         between them; the box keeps none of them.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_RELAY_ROUTES_H
#define PN_LOCAL_RELAY_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer a /local/relay request.
 * @return 0 when it is not one of these routes, 1 when answered, 2 when the hub
 *         adopted the fd (a relay leg: the caller must not close it)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_relay_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_RELAY_ROUTES_H */
