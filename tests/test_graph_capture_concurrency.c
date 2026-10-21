/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_graph_capture_concurrency.c — regression guard for the GLOBAL stream
 * capture bug (task #28, foundry e6af3d4).
 *
 * WHY THIS EXISTS
 *   fc_graph_begin_ctx used to open capture in CU_STREAM_CAPTURE_MODE_GLOBAL,
 *   which makes the driver prohibit "potentially unsafe" CUDA calls across the
 *   WHOLE PROCESS for the duration of the capture. A serve runs one decode
 *   thread per batched model plus a connection thread per HTTP request, with no
 *   GPU-wide lock across models, so a second model's ordinary cuMemAlloc /
 *   cuStreamSynchronize both failed (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
 *   900) and invalidated the other die's open capture (901). Output stayed
 *   correct - the recorded launches never execute - but the per-layer retry
 *   latches, so one collision cost that layer its graph for the life of the
 *   process. It showed up as unexplained 900/901 in the production log.
 *
 *   Measured on the K80 (driver 470.256.02) before the fix: 20 of 300 captures
 *   destroyed and 21 of 108 of the other thread's allocations rejected. After:
 *   300/300 and zero rejections.
 *
 * WHAT IT ASSERTS
 *   With a second thread allocating/syncing on a SECOND die, every capture on
 *   the first die still succeeds, and none of the second thread's allocations
 *   are refused. Both are exact - any failure at all is the bug returning.
 *
 * REQUIREMENTS
 *   Two visible CUDA devices. Self-gates otherwise (prints [SKIP]), so it is
 *   safe to run anywhere. On the K80 test machine, pin it to an idle die:
 *       CUDA_VISIBLE_DEVICES=6,7 tests/test_graph_capture_concurrency
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include "cuda/fc_cuda.h"

#define ROUNDS          200
#define LAUNCHES_PER    13      /* about one layer body */
#define SCRATCH_N       4096

static fc_cuda_ctx *ctx_a, *ctx_b;
static volatile int stop_b;
static int cap_ok, cap_fail, launch_fail;
static int b_alloc_ok, b_alloc_fail;

/* Stands in for a second model's connection thread: allocate scratch, wait for
 * its die, free. Every one of these is a call GLOBAL capture used to reject. */
static void *interferer(void *unused)
{
    (void)unused;
    while (!stop_b) {
        fc_buffer *t = fc_buffer_alloc_ctx(ctx_b, 1u << 20);
        if (!t) { b_alloc_fail++; usleep(200); continue; }
        b_alloc_ok++;
        fc_sync_ctx(ctx_b);
        fc_buffer_free(t);
    }
    return NULL;
}

static fc_buffer *x, *y, *z;

static void capture_round(void)
{
    if (fc_graph_begin_ctx(ctx_a) != 0) { cap_fail++; return; }
    for (int i = 0; i < LAUNCHES_PER; i++) {
        fc_add_ctx(ctx_a, x, y, z, SCRATCH_N);
        fc_mul_ctx(ctx_a, z, y, x, SCRATCH_N);
    }
    fc_graph *g = fc_graph_end_ctx(ctx_a);
    if (!g) { cap_fail++; return; }
    if (fc_graph_launch(g) != 0) launch_fail++;
    else cap_ok++;
    fc_graph_free(g);
}

int main(void)
{
    setbuf(stdout, NULL);

    ctx_a = fc_cuda_init_device(0);
    if (!ctx_a) {
        printf("[SKIP] no CUDA device 0\n");
        return 0;
    }
    ctx_b = fc_cuda_init_device(1);
    if (!ctx_b) {
        printf("[SKIP] needs two visible CUDA devices "
               "(try CUDA_VISIBLE_DEVICES=6,7)\n");
        fc_cuda_ctx_free(ctx_a);
        return 0;
    }
    printf("capture die=%d  interferer die=%d  rounds=%d\n",
           fc_cuda_ctx_device_id(ctx_a), fc_cuda_ctx_device_id(ctx_b), ROUNDS);

    x = fc_buffer_alloc_ctx(ctx_a, SCRATCH_N * sizeof(float));
    y = fc_buffer_alloc_ctx(ctx_a, SCRATCH_N * sizeof(float));
    z = fc_buffer_alloc_ctx(ctx_a, SCRATCH_N * sizeof(float));
    if (!x || !y || !z) {
        printf("[SKIP] could not allocate capture scratch on die 0\n");
        return 0;
    }

    /* Baseline: uncontended. If this is not clean the box has a problem that
     * has nothing to do with what we are testing. */
    for (int i = 0; i < ROUNDS; i++) capture_round();
    if (cap_ok != ROUNDS) {
        printf("FAIL: uncontended capture is already broken "
               "(%d/%d ok, %d failed, %d launch failed)\n",
               cap_ok, ROUNDS, cap_fail, launch_fail);
        return 1;
    }
    printf("  uncontended: %d/%d captures OK\n", cap_ok, ROUNDS);

    cap_ok = cap_fail = launch_fail = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, interferer, NULL) != 0) {
        printf("[SKIP] could not start interferer thread\n");
        return 0;
    }
    for (int i = 0; i < ROUNDS; i++) capture_round();
    stop_b = 1;
    pthread_join(th, NULL);

    printf("  contended:   %d/%d captures OK (%d failed, %d launch failed); "
           "interferer allocs %d OK / %d refused\n",
           cap_ok, ROUNDS, cap_fail, launch_fail, b_alloc_ok, b_alloc_fail);

    int bad = 0;
    if (cap_ok != ROUNDS) {
        printf("FAIL: a second die's thread destroyed %d of %d captures — "
               "stream capture is process-GLOBAL again\n",
               ROUNDS - cap_ok, ROUNDS);
        bad = 1;
    }
    if (b_alloc_fail != 0) {
        printf("FAIL: %d of the interferer's %d allocations were refused — "
               "a capture on another die is blocking unrelated work\n",
               b_alloc_fail, b_alloc_ok + b_alloc_fail);
        bad = 1;
    }
    if (b_alloc_ok == 0) {
        printf("FAIL: interferer never allocated; the test proved nothing\n");
        bad = 1;
    }

    fc_buffer_free(x); fc_buffer_free(y); fc_buffer_free(z);
    fc_cuda_ctx_free(ctx_a);
    fc_cuda_ctx_free(ctx_b);

    printf(bad ? "[FAIL] graph capture concurrency\n"
               : "[PASS] graph capture concurrency\n");
    return bad;
}
