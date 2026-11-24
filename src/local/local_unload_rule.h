/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_unload_rule.h
 * @brief  Who may unload which replica (people registry spec §4.7): an admin
 *         anything; anyone else only replicas they loaded that hold no session.
 *         Also the match of a remembered replica to its serve registry row, which
 *         the rule and GET /local/models/loaded share.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_UNLOAD_RULE_H
#define PN_LOCAL_UNLOAD_RULE_H

#include <stddef.h>

#include "../serve/model_registry.h"
#include "loaded_set.h"
#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The registry row of remembered replica `e`: the same name on its first unit,
 * or for a CPU replica (no units) the CPU row with its replica number.
 * @return the row in `info`, or NULL when that replica is not loaded now
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
const mr_info *pn_local_registry_row_of(const pn_loaded_entry *e, const mr_info *info,
                                        int n_info);

/**
 * Check a non-admin's unload of replica `replica` of catalog_id (0 = every
 * replica, and every registry row under the name): each must have `loaded_by` ==
 * person and no session on it. A registry row the loaded set does not know (a
 * model loaded by hand) is refused too. Targets that do not exist are left to the
 * unload itself (404).
 * @param why  receives the reason ("instance 2 of x was loaded by alice")
 * @return 0 when the unload may go ahead, -1 when refused
 *
 * Thread-safe: yes (call under the loads lock so nothing changes meanwhile).
 * Reentrant: yes. Signal-safe: no
 */
int pn_local_unload_refusal(pn_local_ctx *ctx, const char *catalog_id, int replica,
                            const char *person, char *why, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_UNLOAD_RULE_H */
