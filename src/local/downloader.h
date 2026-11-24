/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   downloader.h
 * @brief  Download + verify one catalog file into the model store (spec §6):
 *         curl into `<final>.part` (resumed from its size), check the size, hash
 *         it, and only then rename to the final name. A sha256 mismatch deletes
 *         the part file and refuses. The URL is built only from manifest fields
 *         that passed validation (catalog_manifest.c). A file with an .ingot
 *         (files[].ingot) is fetched as the .ingot and decoded into the part file
 *         (ingot_fetch.h) when that moves fewer bytes; a delta .ingot only when
 *         its base is installed, else (or when the base is rejected) the whole file.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_DOWNLOADER_H
#define PN_LOCAL_DOWNLOADER_H

#include <stddef.h>
#include <stdint.h>

#include "catalog_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *store_dir;
    const char *base_url;       /* "" / NULL = https://huggingface.co */
    const char *protocols;      /* curl --proto value */
    const pn_catalog_manifest *catalog;   /* optional: finds the installed base of a delta
                                           * .ingot; NULL = such files download whole */
} pn_download_cfg;

/* Progress phases, for the install status. */
typedef enum {
    PN_DL_DOWNLOADING = 0,
    PN_DL_VERIFYING,
} pn_download_phase;

/* Called before each phase; the caller reads the part file's size for bytes. */
typedef void (*pn_download_phase_fn)(void *user, pn_download_phase phase, const char *part_path);

/**
 * Make one file of a model present and verified in the store. A file already
 * at its final name is taken as verified (it got there only through this path).
 *
 * @return 0, or -1 with a reason in err (a whole-file part is kept for a resume
 *         unless the hash was wrong; a failed .ingot decode deletes its part)
 *
 * Thread-safe: yes (one call per file at a time). Reentrant: yes. Signal-safe: no
 */
int pn_download_file(const pn_download_cfg *cfg, const pn_cat_model *m, const pn_cat_file *f,
                     pn_download_phase_fn on_phase, void *user, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_DOWNLOADER_H */
