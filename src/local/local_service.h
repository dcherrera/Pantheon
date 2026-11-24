/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_service.h
 * @brief  Wires the appliance into the serve: builds the subsystems from the
 *         context, installs the serve hooks (routes, startup, shutdown) and the
 *         HTTP auth hook, and tears them down at shutdown.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_SERVICE_H
#define PN_LOCAL_SERVICE_H

#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Open the subsystems and install the serve + auth hooks. Call before
 * pn_serve_main.
 * @return 0, or -1 (message printed)
 *
 * Thread-safe: no (startup)
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_local_service_install(pn_local_ctx *ctx);

/**
 * Free what pn_local_service_install opened (after the serve returned).
 *
 * Thread-safe: no (teardown)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_local_service_release(pn_local_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_SERVICE_H */
