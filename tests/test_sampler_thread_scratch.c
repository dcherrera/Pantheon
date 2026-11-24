/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_sampler_thread_scratch.c — B6: a thread that samples must not leave its
 * sampler buffers behind when it exits.
 *
 * The serve runs every request on its own short-lived thread and samples
 * token #1 there. The sampler keeps two n_vocab float buffers per thread
 * (_Thread_local); before B6 nothing freed them at thread exit, so each
 * request leaked ~1.9 MiB at qwen3.6's vocab (~2 GiB per 1,000 requests on
 * the bank).
 *
 * Measures LIVE heap bytes (glibc mallinfo2 / macOS malloc zone statistics),
 * not RSS, so it reports allocations still held, not allocator retention.
 *   1. N short-lived threads each sample one qwen3.6-sized row: live bytes
 *      must not grow by more than FLAT_MB (pre-B6: N x ~1.9 MiB).
 *   2. pn_sampler_thread_release on a thread that sampled frees its buffers.
 */

#include "../src/core/engine.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GLIBC__)
#include <malloc.h>
static double live_heap_mb(void)
{
    struct mallinfo2 mi = mallinfo2();
    return (double)(mi.uordblks + mi.hblkhd) / (1024.0 * 1024.0);
}
#elif defined(__APPLE__)
#include <malloc/malloc.h>
static double live_heap_mb(void)
{
    malloc_statistics_t st;
    malloc_zone_statistics(NULL, &st);
    return (double)st.size_in_use / (1024.0 * 1024.0);
}
#else
static double live_heap_mb(void) { return -1.0; }
#endif

#define N_VOCAB   248320   /* qwen3.6 */
#define N_THREADS 64
#define FLAT_MB   8.0      /* pre-B6 growth: N_THREADS x ~1.9 MiB = ~121 MiB */

static void sample_one_row(void)
{
    float *logits = malloc((size_t)N_VOCAB * sizeof(float));
    if (!logits)
        return;
    for (int i = 0; i < N_VOCAB; i++)
        logits[i] = (float)((i * 2654435761u) % 1000u) / 100.0f;
    pn_sampling_params p = pn_sampling_defaults();
    p.temperature = 0.6f;
    p.top_p = 0.95f;
    p.top_k = 20;
    (void)pn_sample_next_token(logits, N_VOCAB, &p, NULL, 0);
    free(logits);
}

static void *request_thread(void *arg)
{
    (void)arg;
    sample_one_row();
    return NULL;
}

static int check_threads_leave_nothing(void)
{
    pthread_t t;
    pthread_create(&t, NULL, request_thread, NULL);   /* warm-up */
    pthread_join(t, NULL);
    double before = live_heap_mb();
    for (int i = 0; i < N_THREADS; i++) {
        pthread_create(&t, NULL, request_thread, NULL);
        pthread_join(t, NULL);
    }
    double grew = live_heap_mb() - before;
    int ok = grew <= FLAT_MB;
    printf("%s %d sampling threads exited: live heap %+.1f MiB (limit %.1f)\n",
           ok ? "PASS" : "FAIL", N_THREADS, grew, FLAT_MB);
    return ok;
}

static int check_explicit_release(void)
{
    sample_one_row();
    double held = live_heap_mb();
    pn_sampler_thread_release();
    double freed = held - live_heap_mb();
    double want = 2.0 * N_VOCAB * sizeof(float) / (1024.0 * 1024.0) * 0.9;
    int ok = freed >= want;
    printf("%s pn_sampler_thread_release frees %.1f MiB (want >= %.1f)\n",
           ok ? "PASS" : "FAIL", freed, want);
    return ok;
}

int main(void)
{
    if (live_heap_mb() < 0.0) {
        printf("SKIP test_sampler_thread_scratch: no live-heap statistics on "
               "this platform\n");
        return 0;
    }
    int ok = check_threads_leave_nothing();
    ok &= check_explicit_release();
    return ok ? 0 : 1;
}
