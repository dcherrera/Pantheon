/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_uninstall.h
 * @brief  Remove an installed model from the box (H2m):
 *           POST /local/models/uninstall {catalog_id}
 *             → 200 {catalog_id, removed_files, freed}
 *             → 409 while the model is loaded, remembered for the next start,
 *               or still installing (a plain reason: unload / wait first)
 *             → 404 when nothing of it is in the store
 *         Works for a model the catalog no longer lists, too: those are listed
 *         in GET /local/catalog as "store_only" (pn_local_write_store_only).
 *
 * @status None
 * @issues An install submitted in the same instant as the removal can lose
 *         its `.part`; the next install starts it again.
 * @todo   None
 */

#ifndef PN_LOCAL_UNINSTALL_H
#define PN_LOCAL_UNINSTALL_H

#include <stddef.h>
#include <stdint.h>

#include "../serve/http_server.h"
#include "../serve/json_writer.h"
#include "catalog_manifest.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Uninstall catalog_id: refused while loaded / remembered / installing, else
 * every store entry of it is removed (model_store_cleanup.h).
 * @param status  HTTP status of a refusal (404 / 409 / 500)
 * @return 0, or -1 with a plain reason
 *
 * Thread-safe: yes (holds the loads lock). Reentrant: no. Signal-safe: no
 */
int pn_local_uninstall(pn_local_ctx *ctx, const char *catalog_id, uint64_t *freed,
                       int *n_removed, int *status, char *err, size_t errlen);

/**
 * Append `"store_only":[{"catalog_id","bytes","files"}]`: models whose files
 * are in the store but that the catalog `m` (may be NULL) does not list, or
 * lists with other files (an older revision; downloads in progress excepted).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_write_store_only(json_writer *jw, const pn_local_ctx *ctx,
                               const pn_catalog_manifest *m);

/**
 * @return 1 if this was POST /local/models/uninstall (answered), else 0
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_local_uninstall_route(int client_fd, pn_local_ctx *ctx, const hs_request *req);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_UNINSTALL_H */
