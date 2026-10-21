/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_stage_pipeline.c
 * @brief The build pipeline's release contract: an item the consumer was released for is
 *        always fully built, across many jobs reusing one worker.
 *
 * status: off-hardware, runs anywhere with pthreads.
 * issues: none.  todo: none.
 *
 * Build with -DPN_TT_STAGE_PIPELINE_RED to have the consumer read WITHOUT waiting — that run
 * must report stale items (it is the proof the check can fail).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_stage_pipeline.h"

#define ITEMS 32u
#define WORDS 512u /* 2 KiB an item, so a build takes long enough to race */

static uint32_t g_buffer[ITEMS][WORDS];
static uint32_t g_job;

static int build_items(void *context, uint32_t first, uint32_t count)
{
    uint32_t item, word;

    (void)context;
    for (item = first; item < first + count; item++)
        for (word = 0; word < WORDS; word++)
            g_buffer[item][word] = g_job * 100000u + item * 1000u + word;
    return 0;
}

int main(void)
{
    pn_tt_stage_pipeline pipeline;
    unsigned long stale = 0;
    uint32_t job;

    memset(&pipeline, 0, sizeof(pipeline));
    for (job = 1; job <= 2000u; job++) {
        uint32_t item;

        g_job = job;
        if (pn_tt_stage_pipeline_start(&pipeline, build_items, NULL, ITEMS, 4u) != 0) {
            printf("FAIL the worker did not start\n");
            return 1;
        }
        for (item = 0; item < ITEMS; item++) {
#if !defined(PN_TT_STAGE_PIPELINE_RED)
            if (pn_tt_stage_pipeline_wait(&pipeline, item + 1u) != 0) {
                printf("FAIL wait reported a failed job\n");
                return 1;
            }
#endif
            if (g_buffer[item][WORDS - 1u] != job * 100000u + item * 1000u + WORDS - 1u)
                stale++;
        }
        (void)pn_tt_stage_pipeline_wait(&pipeline, ITEMS);
    }
    pn_tt_stage_pipeline_stop(&pipeline);
    printf("%lu stale item reads over 2000 jobs x %u items\n", stale, ITEMS);
    printf("%d failed\n", stale ? 1 : 0);
    return stale ? 1 : 0;
}
