/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_model_routes.h
 * @brief  Model routes of the appliance (spec F4, G1, G3):
 *           POST /local/models/install      {catalog_id}        → 202, download + verify
 *           GET  /local/models/install/<id>                     → progress
 *           POST /local/models/load         {catalog_id, profile_id, units?}
 *           GET  /local/models/loaded                           → the remembered set
 *                                            (+ the load running now), each with
 *                                            state loaded / loading / unloading /
 *                                            not_loaded (+ reason), the units and
 *                                            placement it really holds, and the
 *                                            live progress of a load / unload
 *           POST /admin/models/unload       (wrapped: the serve's own unload, then
 *                                            the model leaves the loaded set)
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_MODEL_ROUTES_H
#define PN_LOCAL_MODEL_ROUTES_H

#include "../serve/http_server.h"
#include "../serve/json_writer.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @return 1 if handled, 0 if not one of these routes
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_model_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

/**
 * The GET /local/models/loaded body: {"models":[{catalog_id, profile_id, name,
 * units, state, placement, reason, progress}]}; progress is null, or
 * {phase, bytes, total, layer, n_layers, units:[{id, bytes, total}]}.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_write_loaded(json_writer *jw, pn_local_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MODEL_ROUTES_H */
