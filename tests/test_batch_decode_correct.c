/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Correctness check for pn_llama_forward_batch.
 *
 * (1) A single sequence decoded via forward_batch(B=1) must produce the SAME
 *     greedy token stream as the sequential pn_llama_forward (pn_infer).
 * (2) Two sequences with different seeds, decoded TOGETHER in one batch, must
 *     each match their own independent single-stream decode — proving the B
 *     caches don't cross-contaminate.
 */

#include "pantheon.h"
#include "src/core/engine.h"
#include "src/models/llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t argmax(const float *v, int n) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return (uint32_t)best;
}

/* Reference: sequential decode of `n` greedy tokens from a seed. */
static void decode_seq(pn_model *m, uint32_t seed, int n, uint32_t *out) {
    pn_state *st = pn_state_new(m);
    uint32_t tok = seed;
    for (int i = 0; i < n; i++) {
        pn_output *o = pn_infer(m, st, tok);
        tok = argmax(o->data, o->n);
        out[i] = tok;
        pn_output_free(o);
    }
    pn_state_free(st);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int n = argc > 2 ? atoi(argv[2]) : 24;
    pn_model *m = pn_load(path);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    int nv = m->config.n_vocab;

    uint32_t seedA = 5, seedB = 99;
    uint32_t *refA = malloc(n * sizeof(uint32_t));
    uint32_t *refB = malloc(n * sizeof(uint32_t));
    decode_seq(m, seedA, n, refA);
    decode_seq(m, seedB, n, refB);

    /* Batched decode of both sequences together. */
    pn_state *st[2] = { pn_state_new(m), pn_state_new(m) };
    uint32_t tok[2] = { seedA, seedB };
    float *logits = malloc((size_t)2 * nv * sizeof(float));
    uint32_t *batA = malloc(n * sizeof(uint32_t));
    uint32_t *batB = malloc(n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) {
        if (pn_llama_forward_batch(m, tok, NULL, st, 2, logits) != 0) {
            fprintf(stderr, "forward_batch unsupported\n"); return 1;
        }
        tok[0] = argmax(logits + 0 * nv, nv);
        tok[1] = argmax(logits + 1 * nv, nv);
        batA[i] = tok[0];
        batB[i] = tok[1];
    }

    int mismatchA = 0, mismatchB = 0;
    for (int i = 0; i < n; i++) {
        if (batA[i] != refA[i]) mismatchA++;
        if (batB[i] != refB[i]) mismatchB++;
    }
    printf("seq A: %d/%d tokens match sequential decode\n", n - mismatchA, n);
    printf("seq B: %d/%d tokens match sequential decode\n", n - mismatchB, n);
    printf("%s\n", (mismatchA == 0 && mismatchB == 0) ? "PASS: batched == sequential, caches independent"
                                                      : "FAIL: batched decode diverges");

    pn_state_free(st[0]); pn_state_free(st[1]);
    free(refA); free(refB); free(batA); free(batB); free(logits);
    pn_model_free(m);
    return (mismatchA == 0 && mismatchB == 0) ? 0 : 1;
}
