/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   model_store_cleanup.h
 * @brief  Uninstall from the model store (H2m): find every store entry of one
 *         catalog id and remove it. A store entry's name starts with
 *         `<catalog id>-<12 hex>.` — the verified files (`<id>-<sha12>.<ext>`),
 *         their downloads in progress (`.part`), the companion links
 *         (`<id>-<weights sha12>.<rest>`) and the backend caches written beside
 *         the weights (the TT card artifact `<id>-<sha12>.<rung>.fttw` and its
 *         `.tmp.<pid>`). Also lists the store by catalog id, so models a newer
 *         catalog dropped can be shown and removed.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_MODEL_STORE_CLEANUP_H
#define PN_LOCAL_MODEL_STORE_CLEANUP_H

#include <stddef.h>
#include <stdint.h>

#include "catalog_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_STORE_GROUPS_MAX 64

/* The store entries of one catalog id. */
typedef struct {
    char     catalog_id[PN_CAT_ID_MAX];
    uint64_t bytes;      /* regular files (links count 0) */
    int      n_files;    /* entries of any kind, downloads in progress included */
    int      n_partial;  /* of which `.part` downloads in progress */
} pn_store_group;

/**
 * The catalog id of a store entry name: everything before the first
 * `-<12 hex>.`.
 * @return 0, or -1 when the name is not a store entry
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_model_store_entry_id(const char *name, char *id, size_t cap);

/**
 * Group the store's entries by catalog id, in directory order.
 * @return groups written (0 when the store cannot be read)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_model_store_scan(const char *store_dir, pn_store_group *out, int max);

/**
 * Remove every store entry of `catalog_id` (files and links; never a
 * directory). *freed = bytes of the regular files removed, *n_removed =
 * entries removed.
 * @return 0 (also when nothing matched: *n_removed is 0), -1 with the first
 *         entry that could not be removed in `err` (the rest are still tried)
 *
 * Thread-safe: no (the caller keeps installs and loads of the id out).
 * Reentrant: no. Signal-safe: no
 */
int pn_model_store_remove_id(const char *store_dir, const char *catalog_id, uint64_t *freed,
                             int *n_removed, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MODEL_STORE_CLEANUP_H */
