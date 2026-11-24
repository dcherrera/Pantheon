/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_key_routes.h
 * @brief  Pairing and key routes (people registry spec §4.2-4.3):
 *           POST /local/pair         {code, label, device_id?, device_name?, handle?,
 *                                    name?} → a device key of the code's person
 *                                    (returned once) + the person
 *           GET  /local/keys[?all=1] the caller's keys (all of them for an admin)
 *           POST /local/keys         {label, person?} → an inference key (returned once)
 *           POST /local/keys/revoke  {id}: own keys; an admin any key
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_KEY_ROUTES_H
#define PN_LOCAL_KEY_ROUTES_H

#include "../serve/http_server.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer a pairing / key route.
 * @return 1 if handled, 0 if the path is not one of these
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_key_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_KEY_ROUTES_H */
