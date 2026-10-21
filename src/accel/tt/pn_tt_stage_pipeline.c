/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_stage_pipeline.c
 * @brief The one-worker build pipeline. See the header for the contract.
 *
 * @status Used by the gdnc DELTA staging.
 * @issues None
 * @todo None
 */
#include "pn_tt_stage_pipeline.h"

#include <sched.h>
#include <string.h>

/** The worker: wait for a job, build it batch by batch, publish progress, repeat. */
static void *stage_pipeline_worker(void *argument)
{
    pn_tt_stage_pipeline *pipeline = (pn_tt_stage_pipeline *)argument;

    for (;;) {
        pn_tt_stage_build_fn fn;
        void *context;
        uint32_t total, batch, done = 0u;

        pthread_mutex_lock(&pipeline->lock);
        while (!pipeline->has_job && !pipeline->quit)
            pthread_cond_wait(&pipeline->wake, &pipeline->lock);
        if (pipeline->quit) {
            pthread_mutex_unlock(&pipeline->lock);
            return NULL;
        }
        pipeline->has_job = false;
        fn = pipeline->fn;
        context = pipeline->context;
        total = pipeline->total;
        batch = pipeline->batch;
        pthread_mutex_unlock(&pipeline->lock);

        while (done < total) {
            const uint32_t count = (total - done < batch) ? total - done : batch;

            if (fn(context, done, count) != 0) {
                __atomic_store_n(&pipeline->failed, 1, __ATOMIC_RELEASE);
                break;
            }
            done += count;
            __atomic_store_n(&pipeline->progress, done, __ATOMIC_RELEASE);
        }
    }
}

int pn_tt_stage_pipeline_start(pn_tt_stage_pipeline *pipeline, pn_tt_stage_build_fn fn,
                               void *context, uint32_t total, uint32_t batch)
{
    if (!pipeline || !fn || batch == 0u)
        return -1;
    if (!pipeline->started) {
        memset(pipeline, 0, sizeof(*pipeline));
        if (pthread_mutex_init(&pipeline->lock, NULL) != 0)
            return -1;
        if (pthread_cond_init(&pipeline->wake, NULL) != 0) {
            pthread_mutex_destroy(&pipeline->lock);
            return -1;
        }
        if (pthread_create(&pipeline->thread, NULL, stage_pipeline_worker, pipeline) != 0) {
            pthread_cond_destroy(&pipeline->wake);
            pthread_mutex_destroy(&pipeline->lock);
            return -1;
        }
        pipeline->started = true;
    }
    pthread_mutex_lock(&pipeline->lock);
    __atomic_store_n(&pipeline->progress, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&pipeline->failed, 0, __ATOMIC_RELEASE);
    pipeline->fn = fn;
    pipeline->context = context;
    pipeline->total = total;
    pipeline->batch = batch;
    pipeline->has_job = true;
    pthread_cond_signal(&pipeline->wake);
    pthread_mutex_unlock(&pipeline->lock);
    return 0;
}

int pn_tt_stage_pipeline_wait(pn_tt_stage_pipeline *pipeline, uint32_t items)
{
    /* A SPIN, NOT A CONDITION WAIT: a batch is a few microseconds of work and the consumer is
     * between MMIO uploads, so a sleep/wake round trip would cost more than the wait. The box
     * has four CPUs and this runs on two of them. */
    for (;;) {
        if (__atomic_load_n(&pipeline->failed, __ATOMIC_ACQUIRE) != 0)
            return -1;
        if (__atomic_load_n(&pipeline->progress, __ATOMIC_ACQUIRE) >= items)
            return 0;
        sched_yield();
    }
}

void pn_tt_stage_pipeline_stop(pn_tt_stage_pipeline *pipeline)
{
    if (!pipeline || !pipeline->started)
        return;
    pthread_mutex_lock(&pipeline->lock);
    pipeline->quit = true;
    pthread_cond_signal(&pipeline->wake);
    pthread_mutex_unlock(&pipeline->lock);
    pthread_join(pipeline->thread, NULL);
    pthread_cond_destroy(&pipeline->wake);
    pthread_mutex_destroy(&pipeline->lock);
    pipeline->started = false;
}
