/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_prefill.c — validate + benchmark the batched multi-token prefill
 * against the sequential single-token forward.
 *
 * Usage: test_prefill <model.gguf> [P_validate] [P_bench]
 *
 * Feeds identical fixed input embeddings through:
 *   (a) P sequential pn_llama_forward() calls (embd_override per token)
 *   (b) one pn_llama_prefill() batched call
 * and compares the resulting last-token logits.
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
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

static float* make_embeds(int P, int n_embd, unsigned seed) {
    srand(seed);
    float *e = malloc((size_t)P * n_embd * sizeof(float));
    for (int i = 0; i < P * n_embd; i++)
        e[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    return e;
}

/* Sequential reference: returns owned logits for the last token. */
static float* seq_last_logits(pn_model *model, pn_state *state,
                              const float *emb, int P, int n_embd, int n_vocab) {
    pn_state_reset(state);
    tm_tensor *logits = NULL;
    for (int p = 0; p < P; p++) {
        if (logits) tm_free(logits);
        state->embd_override = emb + (size_t)p * n_embd;
        logits = model->forward(model, 0, state);
        state->embd_override = NULL;
        if (!logits) return NULL;
    }
    float *out = malloc((size_t)n_vocab * sizeof(float));
    memcpy(out, logits->data, (size_t)n_vocab * sizeof(float));
    tm_free(logits);
    return out;
}

static float* batched_last_logits(pn_model *model, pn_state *state,
                                  const float *emb, int P, int n_embd, int n_vocab) {
    pn_state_reset(state);
    const float **rows = calloc((size_t)P, sizeof(float *));
    uint32_t *toks = calloc((size_t)P, sizeof(uint32_t));
    for (int p = 0; p < P; p++) rows[p] = emb + (size_t)p * n_embd;
    tm_tensor *logits = pn_llama_prefill(model, toks, rows, P, state);
    free(rows); free(toks);
    if (!logits) return NULL;
    float *out = malloc((size_t)n_vocab * sizeof(float));
    memcpy(out, logits->data, (size_t)n_vocab * sizeof(float));
    tm_free(logits);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [Pval] [Pbench]\n", argv[0]); return 2; }
    int Pval   = argc > 2 ? atoi(argv[2]) : 64;
    int Pbench = argc > 3 ? atoi(argv[3]) : 256;

    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    pn_state *state = pn_state_new(model);
    if (!state) { fprintf(stderr, "state failed\n"); return 1; }

    int n_embd = model->config.n_embd;
    int n_vocab = model->config.n_vocab;
    printf("model: n_embd=%d n_vocab=%d n_layer=%d n_head=%d arch=%d\n",
           n_embd, n_vocab, model->config.n_layer, model->config.n_head, model->config.arch);

    /* ---- Validation ---- */
    float *emb = make_embeds(Pval, n_embd, 1234);
    float *seqL = seq_last_logits(model, state, emb, Pval, n_embd, n_vocab);
    if (!seqL) { fprintf(stderr, "sequential failed\n"); return 1; }
    float *batL = batched_last_logits(model, state, emb, Pval, n_embd, n_vocab);
    if (!batL) { fprintf(stderr, "BATCHED PREFILL UNSUPPORTED (returned NULL)\n"); return 3; }

    double max_abs = 0, dot = 0, na = 0, nb = 0;
    for (int v = 0; v < n_vocab; v++) {
        double d = fabs((double)seqL[v] - (double)batL[v]);
        if (d > max_abs) max_abs = d;
        dot += (double)seqL[v] * batL[v];
        na += (double)seqL[v] * seqL[v];
        nb += (double)batL[v] * batL[v];
    }
    double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("VALIDATE P=%d: max_abs_diff=%.3e  cosine=%.8f\n", Pval, max_abs, cosine);
    int ok = (max_abs < 1e-3) || (cosine > 0.9999);
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    free(emb); free(seqL); free(batL);

    /* ---- Benchmark ---- */
    float *embB = make_embeds(Pbench, n_embd, 99);

    double t0 = now_ms();
    float *s = seq_last_logits(model, state, embB, Pbench, n_embd, n_vocab);
    double seq_ms = now_ms() - t0;
    free(s);

    t0 = now_ms();
    float *b = batched_last_logits(model, state, embB, Pbench, n_embd, n_vocab);
    double bat_ms = now_ms() - t0;
    free(b);

    printf("BENCH P=%d: sequential=%.1f ms (%.2f ms/tok)  batched=%.1f ms (%.2f ms/tok)  speedup=%.2fx\n",
           Pbench, seq_ms, seq_ms / Pbench, bat_ms, bat_ms / Pbench, seq_ms / bat_ms);

    free(embB);
    pn_state_free(state);
    pn_model_free(model);
    return ok ? 0 : 1;
}
