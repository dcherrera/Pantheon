/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* bench_verify_rows.c - M1 milestone bench for batched speculative decoding.
 *
 * Measures the cost of one pn_llama_verify pass as a function of the number of
 * rows M (candidate tokens verified in one shot). The verify pass is the kernel
 * the batched speculative decoder runs once per step over the concatenation of
 * every slot's [pending, drafts...] block, so this table confirms the payoff
 * (verify of M rows costs far less than M separate single-token forwards) and
 * picks a sane default draft width K.
 *
 * The pass runs on a state prefilled to a realistic context length so the
 * attention cost (which grows with committed length) is representative. pos is
 * never advanced by verify, so the same committed cache is reused for every M
 * and every repetition.
 *
 * Usage: bench_verify_rows [model.gguf] [ctx_len] [reps]
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int ctx_len = (argc > 2) ? atoi(argv[2]) : 256;
    int reps    = (argc > 3) ? atoi(argv[3]) : 20;

    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    pn_state *state = pn_state_new(model);
    if (!state) { fprintf(stderr, "state failed\n"); return 1; }
    int nv = model->config.n_vocab;

    /* Prefill a deterministic context so verify runs against a warm cache. */
    uint32_t *prompt = malloc((size_t)ctx_len * sizeof(uint32_t));
    uint32_t x = 0x9e3779b9u;
    for (int i = 0; i < ctx_len; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        prompt[i] = (x % (uint32_t)(nv - 4)) + 2;
    }
    pn_state_reset(state);
    tm_tensor *pl = pn_llama_prefill(model, prompt, NULL, ctx_len, state);
    if (!pl) {
        printf("[SKIP] model not on the quantized non-MoE verify path\n");
        free(prompt); pn_state_free(state); pn_model_free(model);
        return 0;
    }
    tm_free(pl);
    printf("model: %s  n_vocab=%d n_layer=%d n_embd=%d  ctx_len=%d reps=%d\n",
           path, nv, model->config.n_layer, model->config.n_embd, ctx_len, reps);

    const int Ms[] = { 6, 8, 12, 16, 24, 32, 48, 64 };
    const int nM = (int)(sizeof(Ms) / sizeof(Ms[0]));

    uint32_t cand[64];
    for (int i = 0; i < 64; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        cand[i] = (x % (uint32_t)(nv - 4)) + 2;
    }
    float *logits = malloc((size_t)64 * (size_t)nv * sizeof(float));
    if (!logits) { fprintf(stderr, "oom\n"); return 1; }

    printf("\n  M   ms/call   ms/row   rows/s\n");
    printf("  --  --------  -------  --------\n");
    for (int mi = 0; mi < nM; mi++) {
        int M = Ms[mi];
        /* one warm-up */
        if (pn_llama_verify(model, cand, NULL, M, state, logits) != 0) {
            fprintf(stderr, "verify failed at M=%d\n", M); return 1;
        }
        double t0 = now_ms();
        for (int r = 0; r < reps; r++)
            pn_llama_verify(model, cand, NULL, M, state, logits);
        double dt = (now_ms() - t0) / reps;
        printf("  %2d  %8.3f  %7.4f  %8.1f\n", M, dt, dt / M, 1000.0 * M / dt);
    }

    free(logits); free(prompt);
    pn_state_free(state); pn_model_free(model);
    return 0;
}
