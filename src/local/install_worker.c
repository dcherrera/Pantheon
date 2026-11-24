/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   install_worker.c
 * @brief  Implementation of the install worker (see install_worker.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "install_worker.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_store.h"

struct pn_install_worker {
    pthread_mutex_t   lock;
    pthread_cond_t    wake;
    pthread_t         thread;
    bool              stop, busy, has_job;
    char              store_dir[1024], base_url[256], protocols[64];
    pn_cat_model      job;
    pn_catalog_manifest *catalog;        /* the job's catalog (copy), for delta bases */
    char              part_path[1100];     /* current part file, for progress */
    uint64_t          done_bytes;          /* finished files of the current job */
    pn_install_status history[PN_INSTALL_HISTORY];
    int               n_history, next_slot;
};

const char *pn_install_state_name(pn_install_state s)
{
    static const char *names[] = { "queued", "downloading", "verifying", "done", "failed" };
    return (unsigned)s < 5 ? names[s] : "unknown";
}

/* The history row of model_id, or a fresh one. Lock held. */
static pn_install_status *row_for(pn_install_worker *w, const char *model_id, bool create)
{
    for (int i = 0; i < w->n_history; i++)
        if (strcmp(w->history[i].model_id, model_id) == 0)
            return &w->history[i];
    if (!create)
        return NULL;
    pn_install_status *r = &w->history[w->next_slot];
    w->next_slot = (w->next_slot + 1) % PN_INSTALL_HISTORY;
    if (w->n_history < PN_INSTALL_HISTORY)
        w->n_history++;
    memset(r, 0, sizeof(*r));
    snprintf(r->model_id, sizeof(r->model_id), "%s", model_id);
    return r;
}

static void on_phase(void *user, pn_download_phase phase, const char *part)
{
    pn_install_worker *w = user;

    pthread_mutex_lock(&w->lock);
    pn_install_status *r = row_for(w, w->job.id, true);
    r->state = phase == PN_DL_VERIFYING ? PN_INSTALL_VERIFYING : PN_INSTALL_DOWNLOADING;
    snprintf(w->part_path, sizeof(w->part_path), "%s", part);
    pthread_mutex_unlock(&w->lock);
}

/* Install every file of the job; returns 0 or -1 with err. */
static int run_job(pn_install_worker *w, char *err, size_t errlen)
{
    pn_download_cfg cfg = { w->store_dir, w->base_url, w->protocols, w->catalog };

    for (int i = 0; i < w->job.n_files; i++) {
        if (pn_download_file(&cfg, &w->job, &w->job.files[i], on_phase, w, err, errlen) != 0)
            return -1;
        pthread_mutex_lock(&w->lock);
        w->done_bytes += w->job.files[i].size;
        w->part_path[0] = '\0';
        pthread_mutex_unlock(&w->lock);
    }
    return 0;
}

static void *worker_main(void *arg)
{
    pn_install_worker *w = arg;
    char err[256];

    pthread_mutex_lock(&w->lock);
    for (;;) {
        while (!w->stop && !w->has_job)
            pthread_cond_wait(&w->wake, &w->lock);
        if (w->stop)
            break;
        w->has_job = false;
        w->done_bytes = 0;
        pthread_mutex_unlock(&w->lock);
        fprintf(stderr, "local: installing %s\n", w->job.id);
        err[0] = '\0';
        int rc = run_job(w, err, sizeof(err));
        pthread_mutex_lock(&w->lock);
        pn_install_status *r = row_for(w, w->job.id, true);
        r->state = rc == 0 ? PN_INSTALL_DONE : PN_INSTALL_FAILED;
        r->bytes = rc == 0 ? r->total : w->done_bytes;
        r->finished = time(NULL);
        snprintf(r->error, sizeof(r->error), "%s", rc == 0 ? "" : err);
        w->busy = false;
        fprintf(stderr, "local: install of %s %s%s%s\n", w->job.id, rc == 0 ? "done" : "failed",
                rc == 0 ? "" : ": ", rc == 0 ? "" : err);
    }
    pthread_mutex_unlock(&w->lock);
    return NULL;
}

pn_install_worker *pn_install_worker_start(const pn_download_cfg *cfg)
{
    pn_install_worker *w = calloc(1, sizeof(*w));

    if (!w)
        return NULL;
    pthread_mutex_init(&w->lock, NULL);
    pthread_cond_init(&w->wake, NULL);
    snprintf(w->store_dir, sizeof(w->store_dir), "%s", cfg->store_dir);
    snprintf(w->base_url, sizeof(w->base_url), "%s", cfg->base_url ? cfg->base_url : "");
    snprintf(w->protocols, sizeof(w->protocols), "%s", cfg->protocols);
    if (pthread_create(&w->thread, NULL, worker_main, w) != 0) {
        pthread_cond_destroy(&w->wake);
        pthread_mutex_destroy(&w->lock);
        free(w);
        return NULL;
    }
    return w;
}

void pn_install_worker_stop(pn_install_worker *w)
{
    if (!w)
        return;
    pthread_mutex_lock(&w->lock);
    w->stop = true;
    bool busy = w->busy;
    pthread_cond_signal(&w->wake);
    pthread_mutex_unlock(&w->lock);
    if (busy) {
        /* Never hold shutdown behind a multi-GB download: the curl child dies
         * with the process (PR_SET_PDEATHSIG) and the .part file resumes on the
         * next install. The worker is left to the process exit. */
        fprintf(stderr, "local: install of %s interrupted; it resumes on the next install\n",
                w->job.id);
        pthread_detach(w->thread);
        return;
    }
    pthread_join(w->thread, NULL);
    pthread_cond_destroy(&w->wake);
    pthread_mutex_destroy(&w->lock);
    free(w->catalog);
    free(w);
}

int pn_install_worker_submit(pn_install_worker *w, const pn_cat_model *m,
                             const pn_catalog_manifest *catalog, char *err, size_t errlen)
{
    pn_catalog_manifest *copy = catalog ? malloc(sizeof(*copy)) : NULL;
    int rc = -1;

    if (catalog && !copy) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (copy)
        *copy = *catalog;
    pthread_mutex_lock(&w->lock);
    if (w->busy) {
        snprintf(err, errlen, "an install is already running (%s); one at a time", w->job.id);
        free(copy);
    } else {
        w->job = *m;
        free(w->catalog);
        w->catalog = copy;
        w->busy = w->has_job = true;
        pn_install_status *r = row_for(w, m->id, true);
        memset(r->error, 0, sizeof(r->error));
        r->state = PN_INSTALL_QUEUED;
        r->bytes = 0;
        r->total = 0;
        for (int i = 0; i < m->n_files; i++)
            r->total += m->files[i].size;
        r->started = time(NULL);
        r->finished = 0;
        pthread_cond_signal(&w->wake);
        rc = 0;
    }
    pthread_mutex_unlock(&w->lock);
    return rc;
}

int pn_install_worker_status(pn_install_worker *w, const char *model_id, pn_install_status *out)
{
    int rc = -1;

    pthread_mutex_lock(&w->lock);
    pn_install_status *r = row_for(w, model_id, false);
    if (r) {
        *out = *r;
        bool current = w->busy && strcmp(w->job.id, model_id) == 0;
        if (current && (r->state == PN_INSTALL_DOWNLOADING || r->state == PN_INSTALL_VERIFYING))
            out->bytes = w->done_bytes + (w->part_path[0] ? pn_model_store_file_size(w->part_path)
                                                          : 0);
        rc = 0;
    }
    pthread_mutex_unlock(&w->lock);
    return rc;
}
