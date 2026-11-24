/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   key_migration.h
 * @brief  keys.json v1 → v2 at start (people registry spec §3.2): every v1 key
 *         becomes a key of person 1, the box's oldest admin. When there are no
 *         people yet, person 1 `owner` / "Owner" (admin) is created and
 *         people.json saved FIRST; then the v1 file is kept as keys.json.v1 and
 *         keys.json is written as v2. A crash between the two saves re-runs
 *         cleanly: the next start finds the admin and reuses it.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_KEY_MIGRATION_H
#define PN_LOCAL_KEY_MIGRATION_H

#include "key_store.h"
#include "people_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Migrate `ks` when it was read from a v1 file; nothing to do otherwise.
 * @param backup_path  where the v1 file is kept (keys.json.v1)
 * @return 0 (migrated or nothing to do), -1 on a failure (logged; serve nothing)
 *
 * Thread-safe: no (call before serving)
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_key_migrate_v1(pn_key_store *ks, pn_people *people, const char *backup_path);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_KEY_MIGRATION_H */
