/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   install_worker.h
 * @brief  One worker thread that installs catalog models into the store, one
 *         download at a time (spec F4). Submitting while busy is refused; status
 *         (state, bytes / total from the part file, error) is kept per model for
 *         the last installs.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_INSTALL_WORKER_H
#define PN_LOCAL_INSTALL_WORKER_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "catalog_manifest.h"
#include "downloader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_INSTALL_HISTORY 16

typedef enum {
    PN_INSTALL_QUEUED = 0,
    PN_INSTALL_DOWNLOADING,
    PN_INSTALL_VERIFYING,
    PN_INSTALL_DONE,
    PN_INSTALL_FAILED,
} pn_install_state;

typedef struct {
    char             model_id[PN_CAT_ID_MAX];
    pn_install_state state;
    uint64_t         bytes;        /* on disk so far (finished files + part file) */
    uint64_t         total;        /* sum of the model's file sizes */
    char             error[256];
    time_t           started, finished;
} pn_install_status;

typedef struct pn_install_worker pn_install_worker;

/**
 * Start the worker. cfg's strings are copied; cfg->catalog is not used (each job brings
 * its own, pn_install_worker_submit).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_install_worker *pn_install_worker_start(const pn_download_cfg *cfg);

/** Stop (after the current file) and free. NULL-safe. Thread-safe: no. */
void pn_install_worker_stop(pn_install_worker *w);

/**
 * Queue the install of `m` (copied). `catalog` is the manifest `m` came from (copied for
 * the job; NULL = a delta .ingot file downloads whole): it finds installed delta bases.
 * @return 0, or -1 when an install is already running (err says which) or out of memory
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_install_worker_submit(pn_install_worker *w, const pn_cat_model *m,
                             const pn_catalog_manifest *catalog, char *err, size_t errlen);

/**
 * Status of the latest install of model_id.
 * @return 0, or -1 if it was never submitted
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_install_worker_status(pn_install_worker *w, const char *model_id, pn_install_status *out);

/** "queued", "downloading", "verifying", "done", "failed". Thread-safe: yes. */
const char *pn_install_state_name(pn_install_state s);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_INSTALL_WORKER_H */
