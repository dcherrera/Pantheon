/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_routes.h
 * @brief  The appliance's HTTP routes under /local/ (hardware, pairing, keys,
 *         catalog, install, load). Called from the serve's route hook after the
 *         auth hook let the request through.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_ROUTES_H
#define PN_LOCAL_ROUTES_H

#include "../serve/http_server.h"
#include "host_info.h"
#include "hw_report.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Answer one /local/... request.
 * @return 1 if the route was handled (response sent), 2 if handled and the fd was
 *         adopted (a held host wait: the caller must not close it), 0 if it is not a
 *         /local route
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_routes_dispatch(int client_fd, pn_local_ctx *ctx, const hs_request *req);

/**
 * Host facts + card tree of the box now, device memory from the background
 * snapshot as it is (instant; used by /local/hardware and discovery).
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_local_snapshot_hardware(pn_local_ctx *ctx, pn_host_info *host,
                                pn_hw_report *hw);

/**
 * The same with device memory fresh enough to place a load on: read at most
 * PN_HW_FIT_MAX_AGE_MS ago and after the last load or unload (waits for a
 * background pass when it is not; used by the catalog fit and loads).
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_local_snapshot_hardware_fresh(pn_local_ctx *ctx, pn_host_info *host,
                                pn_hw_report *hw);

/**
 * Last accepted catalog version, or -1 when no catalog was accepted yet.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
long long pn_local_catalog_version(const pn_local_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_ROUTES_H */
