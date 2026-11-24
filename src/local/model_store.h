/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   model_store.h
 * @brief  The appliance's model store (spec §4.3): one directory (default
 *         ~/.pantheon/models/), each catalog file stored as
 *         `<catalog-id>-<first 12 hex of its sha256>.<ext>`. A final name exists
 *         only after the file was verified (downloads land in `<name>.part`),
 *         so "the file is there" means "the file is verified". Free-space check
 *         before a download. Companion links: an extra file named after the
 *         weights file (`<weights stem>.<rest>`) is linked as
 *         `<id>-<weights sha12>.<rest>`, where the backend looks for it.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_MODEL_STORE_H
#define PN_LOCAL_MODEL_STORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "catalog_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kept free on the store's filesystem after a download (logs, state, swap). */
#define PN_STORE_MARGIN_BYTES ((uint64_t)1 << 30)

/**
 * Store path of the file a load opens: the weights file of `need`, the model's copy for
 * ONE unit type (pn_cat_model_for_units), so a p150 load opens its metadata GGUF and a
 * K80 load its GGUF.
 * @return 0, or -1 (no weights file, or truncation)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_model_store_weights_path(const char *store_dir, const pn_cat_model *need, char *out,
                                size_t cap);

/**
 * Final path of a model's file in the store.
 * @return 0, or -1 on truncation
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_model_store_path(const char *store_dir, const pn_cat_model *m, const pn_cat_file *f,
                        char *out, size_t cap);

/**
 * Whether every file of the model is in the store with its final name and
 * catalog size.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
bool pn_model_store_installed(const char *store_dir, const pn_cat_model *m);

/**
 * Store path of an installed catalog file by its sha256: a file of any model of
 * `catalog` whose sha256 is `sha256` and whose final store file has its catalog
 * size (the store holds a final name only after verifying the file against that
 * sha256). Finds the base of a delta .ingot.
 * @return 0, or -1 when none is installed (or on truncation)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_model_store_find_sha256(const char *store_dir, const pn_catalog_manifest *catalog,
                               const char *sha256, char *out, size_t cap);

/**
 * Refuse when `need` bytes plus the margin do not fit on the store's
 * filesystem.
 * @return 0 if it fits, -1 with "needs X GB free on <dir>, has Y GB"
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_model_store_check_space(const char *store_dir, uint64_t need, char *reason, size_t cap);

/**
 * Link every companion extra file next to the stored weights file. An extra
 * whose filename is `<weights stem>.<rest>` (stem = the weights filename minus
 * its last extension, e.g. `Ornith-1.5-9B-Q8_0.bfp8.mtp.fttw` for
 * `Ornith-1.5-9B-Q8_0.gguf`) gets the symlink `<id>-<weights sha12>.<rest>` ->
 * its own verified store file, replaced atomically when it points elsewhere.
 * This is how the TT backend finds the MTP head (`<trunk>.mtp.fttw`) of a
 * catalog model. Call only when the model is installed.
 * @return 0, or -1 with the reason in `err`
 *
 * Thread-safe: no (one load at a time). Reentrant: no. Signal-safe: no
 */
int pn_model_store_link_companions(const char *store_dir, const pn_cat_model *m, char *err,
                                   size_t errlen);

/**
 * Size of a file, 0 if missing.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
uint64_t pn_model_store_file_size(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MODEL_STORE_H */
