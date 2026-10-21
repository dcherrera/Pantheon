/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* End-to-end per-token generation throughput: single-thread vs pool-parallel
 * pn_qmv on the LightOnOCR LM. Drives the decoder token-by-token (greedy) and
 * verifies the two paths produce the identical token sequence. Bounded run. */

#include "pantheon.h"
#include "src/core/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Greedy-generate n_gen tokens starting from a fixed seed token. Records the
 * produced token ids and the mean ms/token. Returns 0 on success. */
static int run_gen(pn_model *model, int n_gen, uint32_t seed_tok,
                   uint32_t *toks_out, double *ms_per_tok) {
    pn_state *st = pn_state_new(model);
    if (!st) return -1;
    uint32_t tok = seed_tok;
    double t0 = now_ms();
    for (int i = 0; i < n_gen; i++) {
        pn_output *o = pn_infer(model, st, tok);
        if (!o || o->type != PN_OUT_LOGITS) { pn_state_free(st); return -1; }
        int best = 0;
        float bv = o->data[0];
        for (int v = 1; v < o->n; v++) {
            if (o->data[v] > bv) { bv = o->data[v]; best = v; }
        }
        tok = (uint32_t)best;
        toks_out[i] = tok;
        pn_output_free(o);
    }
    *ms_per_tok = (now_ms() - t0) / n_gen;
    pn_state_free(st);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "models/lightonocr-lm-Q4_K_M.gguf";
    int n_gen = argc > 2 ? atoi(argv[2]) : 32;

    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }

    uint32_t *a = malloc(n_gen * sizeof(uint32_t));
    uint32_t *b = malloc(n_gen * sizeof(uint32_t));
    double ms_mt = 0, ms_1t = 0;
    uint32_t seed = 1;  /* arbitrary fixed start token */

    /* Warm up (page-in weights) */
    double junk; run_gen(model, 2, seed, a, &junk);

    /* AFTER: multithreaded pn_qmv (model->pool registered each forward) */
    struct fw_pool *saved_pool = model->pool;
    if (run_gen(model, n_gen, seed, a, &ms_mt) != 0) { fprintf(stderr, "gen MT failed\n"); return 1; }

    /* BEFORE: force single-thread by detaching the pool. pn_llama_forward will
     * register NULL, so pn_qmv runs serially. */
    model->pool = NULL;
    if (run_gen(model, n_gen, seed, b, &ms_1t) != 0) { fprintf(stderr, "gen 1T failed\n"); return 1; }
    model->pool = saved_pool;

    int mismatch = 0;
    for (int i = 0; i < n_gen; i++) if (a[i] != b[i]) mismatch++;

    printf("\ngenerate %d tokens on %s\n", n_gen, path);
    printf("  single-thread pn_qmv : %.2f ms/token  (%.1f tok/s)\n", ms_1t, 1000.0 / ms_1t);
    printf("  pool-parallel pn_qmv : %.2f ms/token  (%.1f tok/s)\n", ms_mt, 1000.0 / ms_mt);
    printf("  speedup              : %.2fx\n", ms_1t / ms_mt);
    printf("  token-seq mismatch   : %d  -> %s\n", mismatch,
           mismatch == 0 ? "IDENTICAL OUTPUT" : "DIVERGED");

    free(a); free(b);
    pn_model_free(model);
    return mismatch == 0 ? 0 : 1;
}
