/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_stage_pipeline.h
 * @brief A one-worker BUILD pipeline: a persistent thread builds items [0, total) in batches
 *        while the caller consumes the batches already built — so host-side construction of a
 *        staged block overlaps its own MMIO upload instead of preceding it.
 *
 * @status Used by the gdnc DELTA staging (`PN_TT_GDNC_STAGE_THREAD`).
 * @issues None
 * @todo None
 *
 * The builder must write only the bytes of the items it is handed, and the consumer must read
 * only items `pn_tt_stage_pipeline_wait()` has released; that partition is the whole of the
 * synchronisation contract. Progress is published with RELEASE and read with ACQUIRE, so a
 * released item's bytes are visible to the consumer.
 */
#ifndef PN_TT_STAGE_PIPELINE_H
#define PN_TT_STAGE_PIPELINE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

/** Build items [first, first + count). Returns 0, or non-zero to fail the whole job. */
typedef int (*pn_tt_stage_build_fn)(void *context, uint32_t first, uint32_t count);

typedef struct pn_tt_stage_pipeline {
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    bool started;   /**< the thread exists */
    bool quit;      /**< asked to exit */
    bool has_job;   /**< a job is posted and not yet picked up */
    pn_tt_stage_build_fn fn;
    void *context;
    uint32_t total, batch;
    uint32_t progress; /**< items built so far; atomic, release/acquire */
    int failed;        /**< non-zero once any batch failed; atomic */
} pn_tt_stage_pipeline;

/** Start a build of `total` items in batches of `batch`; spawns the worker on first use.
 *  @return 0, or -1 when the worker could not be started (the caller builds inline) */
int pn_tt_stage_pipeline_start(pn_tt_stage_pipeline *pipeline, pn_tt_stage_build_fn fn,
                               void *context, uint32_t total, uint32_t batch);

/** Block until at least `items` are built or the job failed. @return 0, or -1 on failure */
int pn_tt_stage_pipeline_wait(pn_tt_stage_pipeline *pipeline, uint32_t items);

/** Stop and join the worker, if any. Safe on a pipeline that never started. */
void pn_tt_stage_pipeline_stop(pn_tt_stage_pipeline *pipeline);

#endif /* PN_TT_STAGE_PIPELINE_H */
