/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Bit-identical + throughput check for the parallelized pn_qmv.
 *
 * Compares single-threaded vs pool-parallel pn_qmv on real Q4_K tensors from
 * the LightOnOCR LM, then times both. Bounded: a few seconds total. */

#include "src/formats/gguf.h"
#include "src/core/qweight.h"
#include "threadpool/threadpool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void fill_vec(float *v, int n, unsigned seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    }
}

static int test_tensor(pn_gguf_file *g, const char *name, fw_pool *pool, int iters) {
    pn_qweight w;
    if (pn_qweight_from_gguf(&w, g, name) != 0) {
        printf("  [skip] %s not found\n", name);
        return 0;
    }

    float *vec = malloc((size_t)w.cols * sizeof(float));
    float *o_ref = malloc((size_t)w.rows * sizeof(float));
    float *o_mt  = malloc((size_t)w.rows * sizeof(float));
    fill_vec(vec, w.cols, 12345u);

    /* Single-threaded reference (no pool registered) */
    pn_qmv_set_pool(NULL);
    pn_qmv(&w, vec, o_ref);

    /* Pool-parallel */
    pn_qmv_set_pool(pool);
    pn_qmv(&w, vec, o_mt);

    double max_abs = 0.0;
    int nbad = 0;
    for (int r = 0; r < w.rows; r++) {
        double d = fabs((double)o_ref[r] - (double)o_mt[r]);
        if (d > max_abs) max_abs = d;
        if (o_ref[r] != o_mt[r]) nbad++;
    }

    /* Bench single-threaded */
    pn_qmv_set_pool(NULL);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) pn_qmv(&w, vec, o_ref);
    double st = (now_ms() - t0) / iters;

    /* Bench parallel */
    pn_qmv_set_pool(pool);
    t0 = now_ms();
    for (int i = 0; i < iters; i++) pn_qmv(&w, vec, o_mt);
    double mt = (now_ms() - t0) / iters;

    printf("  %-22s rows=%6d cols=%5d  maxdiff=%.3g  bitdiff=%d  1T=%.4f ms  MT=%.4f ms  %.2fx\n",
           name, w.rows, w.cols, max_abs, nbad, st, mt, st / mt);

    free(vec); free(o_ref); free(o_mt);
    return nbad;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "models/lightonocr-lm-Q4_K_M.gguf";
    pn_gguf_file *g = pn_gguf_open(path);
    if (!g) { fprintf(stderr, "open failed: %s\n", path); return 1; }

    fw_pool *pool = fw_pool_new(0);
    printf("pn_qmv thread validation (pool=%p)\n", (void *)pool);

    int total_bad = 0;
    total_bad += test_tensor(g, "token_embd.weight",      pool, 30);
    total_bad += test_tensor(g, "output.weight",          pool, 50);
    total_bad += test_tensor(g, "blk.0.ffn_down.weight",  pool, 200);
    total_bad += test_tensor(g, "blk.0.ffn_gate.weight",  pool, 200);
    total_bad += test_tensor(g, "blk.0.attn_q.weight",    pool, 200);
    total_bad += test_tensor(g, "blk.0.attn_v.weight",    pool, 200);

    printf("\n%s  (total bit-differing rows = %d)\n",
           total_bad == 0 ? "BIT-IDENTICAL: PASS" : "MISMATCH: FAIL", total_bad);

    fw_pool_free(pool);
    pn_gguf_close(g);
    return total_bad == 0 ? 0 : 1;
}
