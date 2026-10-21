/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Bounded per-token generation benchmark for the Q4_K decoder.
 * Greedy (temp 0) generation from a fixed seed. Prints ms/token, effective
 * GB/s, and the produced token ids so before/after runs can be diffed. */

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
    const char *path = argc > 1 ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int n_gen = argc > 2 ? atoi(argv[2]) : 48;
    const double gb_per_tok = 0.4;  /* ~0.4 GB weights read per token */

    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }

    uint32_t *a = malloc(n_gen * sizeof(uint32_t));
    double ms = 0, junk;
    uint32_t seed = 1;

    /* Warm up (page-in weights) */
    run_gen(model, 4, seed, a, &junk);

    if (run_gen(model, n_gen, seed, a, &ms) != 0) { fprintf(stderr, "gen failed\n"); return 1; }

    double gbps = gb_per_tok / (ms / 1000.0);
    printf("\ngenerate %d tokens on %s\n", n_gen, path);
    printf("  ms/token   : %.2f\n", ms);
    printf("  tok/s      : %.2f\n", 1000.0 / ms);
    printf("  eff GB/s   : %.2f  (assuming %.2f GB/token)\n", gbps, gb_per_tok);
    printf("  tokens     :");
    for (int i = 0; i < n_gen; i++) printf(" %u", a[i]);
    printf("\n");

    free(a);
    pn_model_free(model);
    return 0;
}
