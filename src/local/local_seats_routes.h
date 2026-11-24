/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_seats_routes.h
 * @brief  `GET /local/seats/people` (ADMIN; people registry spec §4.5): per person,
 *         their registered sessions by state (from the session registry) and the
 *         serve seats they hold per model and instance (model_session.h). Counts
 *         only: no titles, addresses, keys or content.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_SEATS_ROUTES_H
#define PN_LOCAL_SEATS_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer GET /local/seats/people (the auth hook already required an admin).
 * @return 0 when it is not this route, 1 when answered
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_seats_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_SEATS_ROUTES_H */
