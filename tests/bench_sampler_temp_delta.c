/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_sampler_temp_delta.c
 * @brief What does temperature actually cost the HOST, per token?
 *
 * @status Complete.  CPU-only: no GPU, no model, no serve.
 * @issues None
 * @todo None
 *
 * The serving question this answers: production runs temperature 0.7 and
 * measures fewer tok/s than temperature 0.  Part of that gap is the host
 * sampler — temp 0 returns after ONE argmax scan, temp 0.7 runs the full
 * softmax / top-k / top-p / draw pipeline.  This times the REAL sampler
 * (pn_sample_next_token, linked from the engine, not a copy) on the REAL
 * shape the serve hands it: an n_vocab buffer that is -INFINITY everywhere
 * except the k survivors the device top-k wrote (gpu_present_topk).
 *
 * Both arms restore the buffer from the same pristine snapshot inside the
 * timed region, so the memcpy cancels exactly in the temp0.7 - temp0 delta.
 * That delta is the ONLY per-token cost that temperature adds on the host,
 * and therefore the entire budget any "move sampling off the host" lever can
 * ever recover.  Anything the serve loses beyond it is somewhere else.
 *
 * Since the sparse fast path also covers greedy, the temp-0 arm is measured
 * BOTH ways: the full-vocab argmax and the survivor argmax.
 *
 * Usage: bench_sampler_temp_delta [n_vocab] [k] [iters]
 *   defaults 248320 (nex-n2-mini) / 40 (serve top_k) / 2000
 */
#include "../src/core/engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

/* The sparse contract gpu_present_topk leaves behind: -inf everywhere, k
 * finite survivors at scattered ids, values descending (iterative argmax). */
static void build_sparse(float *snap, int n_vocab, int k, unsigned seed)
{
    for (int i = 0; i < n_vocab; i++) snap[i] = -INFINITY;
    uint64_t s = seed ? seed : 1;
    for (int i = 0; i < k; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        int id = (int)((s >> 33) % (uint64_t)n_vocab);
        snap[id] = 8.0f - 0.15f * (float)i;
    }
}

/* One timed arm: iters * (restore from snapshot + one real sampler call). */
static double time_arm(float *buf, const float *snap, int n_vocab,
                       pn_sampling_params *p, uint32_t *recent, int n_recent,
                       int iters, int sparse, uint32_t *sink)
{
    size_t bytes = (size_t)n_vocab * sizeof(float);
    uint32_t acc = 0;
    pn_sampler_set_sparse(sparse);
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        memcpy(buf, snap, bytes);
        acc += pn_sample_next_token(buf, n_vocab, p, recent, n_recent);
    }
    double dt = now_ms() - t0;
    pn_sampler_clear_sparse();
    *sink = acc;
    return dt / (double)iters;
}

int main(int argc, char **argv)
{
    int n_vocab = argc > 1 ? atoi(argv[1]) : 248320;
    int k       = argc > 2 ? atoi(argv[2]) : 40;
    int iters   = argc > 3 ? atoi(argv[3]) : 2000;
    if (n_vocab <= 0 || k <= 0 || k > n_vocab || iters <= 0) {
        fprintf(stderr, "usage: %s [n_vocab] [k] [iters]\n", argv[0]);
        return 2;
    }

    float *snap = malloc((size_t)n_vocab * sizeof(float));
    float *buf  = malloc((size_t)n_vocab * sizeof(float));
    if (!snap || !buf) { fprintf(stderr, "oom\n"); free(snap); free(buf); return 1; }
    build_sparse(snap, n_vocab, k, 1234);

    /* The serve's real defaults (pn_sampling_defaults), temperature overridden
     * per arm.  repeat_penalty 1.1 over a 64-token window is ON in production,
     * so the window is populated for both arms. */
    uint32_t recent[64];
    for (int i = 0; i < 64; i++) recent[i] = (uint32_t)(i * 977 % n_vocab);

    pn_sampling_params greedy = pn_sampling_defaults();
    greedy.temperature = 0.0f;
    pn_sampling_params sampled = pn_sampling_defaults();
    sampled.temperature = 0.7f;

    /* Warm the pages and the engine's thread-local sampler scratch so the
     * first-call allocation is not charged to any arm. */
    uint32_t sink = 0;
    time_arm(buf, snap, n_vocab, &greedy,  recent, 64, 32, 0, &sink);
    time_arm(buf, snap, n_vocab, &sampled, recent, 64, 32, 0, &sink);
    time_arm(buf, snap, n_vocab, &sampled, recent, 64, 32, 1, &sink);

    time_arm(buf, snap, n_vocab, &greedy,  recent, 64, 32, 1, &sink);

    double t_greedy  = time_arm(buf, snap, n_vocab, &greedy,  recent, 64, iters, 0, &sink);
    double t_sampled = time_arm(buf, snap, n_vocab, &sampled, recent, 64, iters, 0, &sink);
    unsigned long h0 = pn_sampler_sparse_hits();
    double t_sparse  = time_arm(buf, snap, n_vocab, &sampled, recent, 64, iters, 1, &sink);
    unsigned long hits = pn_sampler_sparse_hits() - h0;
    unsigned long hg0 = pn_sampler_sparse_hits();
    double t_gsparse = time_arm(buf, snap, n_vocab, &greedy,  recent, 64, iters, 1, &sink);
    unsigned long hits_greedy = pn_sampler_sparse_hits() - hg0;

    /* Restore-only arm: what the shared memcpy costs, so the reader can see
     * how much of each absolute number is bench overhead rather than sampler. */
    size_t bytes = (size_t)n_vocab * sizeof(float);
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) memcpy(buf, snap, bytes);
    double t_restore = (now_ms() - t0) / (double)iters;

    printf("n_vocab=%d k=%d iters=%d (sink=%u)\n", n_vocab, k, iters, sink);
    printf("  restore only (every arm pays this) : %.4f ms\n", t_restore);
    printf("  temp 0.0  full-vocab argmax        : %.4f ms\n", t_greedy);
    printf("  temp 0.7  full-vocab sampler       : %.4f ms\n", t_sampled);
    printf("  temp 0.7  sparse-survivor sampler  : %.4f ms   (%lu fast-path hits)\n",
           t_sparse, hits);
    printf("  temp 0.0  sparse-survivor argmax   : %.4f ms   (%lu fast-path hits)\n",
           t_gsparse, hits_greedy);
    if (hits != (unsigned long)iters || hits_greedy != (unsigned long)iters) {
        printf("  !! the fast path did not run on every call — this number is a lie\n");
        free(snap); free(buf);
        return 1;
    }
    printf("\n");
    printf("  temperature costs the host TODAY   : %.4f ms/token\n",
           t_sampled - t_greedy);
    printf("  the sampler fast path recovers     : %.4f ms/token\n",
           t_sampled - t_sparse);
    printf("  the greedy fast path recovers      : %.4f ms/token\n",
           t_greedy - t_gsparse);
    printf("  residual vs greedy after the fix   : %.4f ms/token\n",
           t_sparse - t_greedy);
    printf("\n");
    printf("  If a step costs S ms, recovering R ms/token takes tok/s from\n");
    printf("  1000/S to 1000/(S-R). At S=52.63 (19.0 tok/s):\n");
    printf("    temp 0.7 arm: %.2f tok/s   temp 0.0 arm: %.2f tok/s\n",
           1000.0 / (52.63 - (t_sampled - t_sparse)),
           1000.0 / (52.63 - (t_greedy - t_gsparse)));

    free(snap); free(buf);
    return 0;
}
