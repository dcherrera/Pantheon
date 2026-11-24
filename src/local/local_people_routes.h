/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_people_routes.h
 * @brief  Me, device, people and invite routes (people registry spec §4.2-4.3):
 *           GET  /local/me              the caller's person and device
 *           POST /local/me/device       {device_id?, device_name?}: claim once, rename
 *           GET  /local/devices[?all=1] the caller's devices (every device for an admin)
 *           GET  /local/people          full objects for an admin, short otherwise
 *           POST /local/people          {handle, name, may_load?} (admin)
 *           POST /local/people/<id>     {name?, handle?, may_load?, admin?} (admin per the
 *                                       admin tree; anyone may rename themselves)
 *           POST /local/people/remove   {id} (admin): revokes keys, voids invites,
 *                                       deletes their registry rows
 *           POST /local/invites         {person?}: a one-time code (self; admin: anyone)
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_PEOPLE_ROUTES_H
#define PN_LOCAL_PEOPLE_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer a me / device / people route. The caller comes from local_caller.h.
 * @return 1 if handled, 0 if the path is not one of these
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_people_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_PEOPLE_ROUTES_H */
