/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_catalog_routes.h
 * @brief  Catalog routes (spec §5.5):
 *           GET  /local/catalog          every profile, fitting ones first, the
 *                                        rest greyed out with the reason
 *           POST /local/catalog/refresh  fetch + verify the manifest now
 *         plus the background fetch at startup.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CATALOG_ROUTES_H
#define PN_LOCAL_CATALOG_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @return 1 if handled, 0 if not a catalog route
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_catalog_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

/**
 * Fetch the catalog in a detached thread (no-op without --catalog-url).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_catalog_fetch_async(pn_local_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CATALOG_ROUTES_H */
