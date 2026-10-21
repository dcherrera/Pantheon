/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Batched multi-sequence decode benchmark.
 *
 * Decodes B independent sequences together via pn_llama_forward_batch (one
 * M=B GEMM per step, dequantizing each weight once and applying it to all B
 * rows) and reports throughput vs batch size. The point: single-token decode
 * is dequant-bound, so batching should raise TOTAL tok/s ~linearly with B
 * while per-sequence tok/s stays roughly flat — i.e. B pages for ~the price of
 * one. Each B builds `ctx` tokens of context (untimed) then times `n_gen`
 * steady-state steps.
 *
 * Usage: bench_batch_decode <model.gguf> [n_gen] [ctx] [Bmax]
 */

#include "pantheon.h"
#include "src/core/engine.h"
#include "src/models/llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint32_t argmax(const float *v, int n) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return (uint32_t)best;
}

/* Decode B sequences together. Returns ms per BATCH step (one M=B forward),
 * measured over n_gen timed steps after building `ctx` tokens of context. */
static double run_batch(pn_model *model, int B, int ctx, int n_gen) {
    int n_vocab = model->config.n_vocab;
    pn_state **st = calloc(B, sizeof(pn_state *));
    uint32_t *tok = malloc(B * sizeof(uint32_t));
    float *logits = malloc((size_t)B * n_vocab * sizeof(float));
    if (!st || !tok || !logits) { free(st); free(tok); free(logits); return -1; }

    for (int b = 0; b < B; b++) {
        st[b] = pn_state_new(model);
        tok[b] = (uint32_t)(1 + b);   /* distinct seed per sequence */
    }

    /* Build context (untimed): each step is a real M=B decode from pos b. */
    for (int i = 0; i < ctx; i++) {
        if (pn_llama_forward_batch(model, tok, NULL, st, B, logits) != 0) {
            fprintf(stderr, "forward_batch unsupported\n");
            for (int b = 0; b < B; b++) pn_state_free(st[b]);
            free(st); free(tok); free(logits); return -1;
        }
        for (int b = 0; b < B; b++) tok[b] = argmax(logits + (size_t)b * n_vocab, n_vocab);
    }

    /* Timed steady-state decode. */
    double t0 = now_ms();
    for (int i = 0; i < n_gen; i++) {
        pn_llama_forward_batch(model, tok, NULL, st, B, logits);
        for (int b = 0; b < B; b++) tok[b] = argmax(logits + (size_t)b * n_vocab, n_vocab);
    }
    double ms_per_step = (now_ms() - t0) / n_gen;

    for (int b = 0; b < B; b++) pn_state_free(st[b]);
    free(st); free(tok); free(logits);
    return ms_per_step;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int n_gen = argc > 2 ? atoi(argv[2]) : 32;
    int ctx   = argc > 3 ? atoi(argv[3]) : 96;
    int Bmax  = argc > 4 ? atoi(argv[4]) : 8;

    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }

    printf("batched decode on %s\n", path);
    printf("  ctx=%d, timed n_gen=%d/seq\n\n", ctx, n_gen);
    printf("  %3s | %10s | %12s | %12s | %8s\n",
           "B", "ms/step", "per-seq t/s", "total t/s", "speedup");
    printf("  ----+------------+--------------+--------------+---------\n");

    double base_total = 0;
    for (int B = 1; B <= Bmax; B *= 2) {
        double ms = run_batch(model, B, ctx, n_gen);
        if (ms < 0) { pn_model_free(model); return 1; }
        double per_seq = 1000.0 / ms;      /* each sequence emits 1 tok per step */
        double total   = per_seq * B;      /* across all B sequences */
        if (B == 1) base_total = total;
        printf("  %3d | %10.2f | %12.2f | %12.2f | %7.2fx\n",
               B, ms, per_seq, total, total / base_total);
    }
    printf("\n  (per-seq t/s ~flat + total t/s scaling with B = the batching win)\n");

    pn_model_free(model);
    return 0;
}
