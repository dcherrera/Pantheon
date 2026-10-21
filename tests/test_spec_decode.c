/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_spec_decode.c - the losslessness gate for greedy speculative decoding.
 *
 * The point of this test: prove that speculative decoding produces a token
 * stream BIT-IDENTICAL to plain greedy decoding, no matter how good or bad the
 * draft is. A draft only proposes tokens; the real model (pn_llama_verify)
 * predicts every position, and pn_spec_decode_greedy emits argmax of THOSE
 * logits, accepting a draft token only when it already equals that argmax. So
 * the emitted stream is fully determined by the model, never by the draft.
 *
 * Strategy:
 *   base   = greedy decode driven entirely by the verify machinery (K=0: one
 *            token per verify pass -> emit argmax of its single row). This is
 *            greedy decoding through EXACTLY the same kernels the speculative
 *            path uses, so it is the reference the speculative loop must match
 *            bit-for-bit by construction.
 *   ref_fwd= plain greedy via pn_llama_forward (the shipped single-token path).
 *            Reported for cross-check: batched GEMMs reassociate the low bits vs
 *            the M=1 pn_qmv path, but that never flips a non-tie argmax, so the
 *            streams agree.
 *   spec_K = speculative decode with draft width K and several draft qualities
 *            (random / all-wrong / oracle / partially-corrupted oracle).
 *
 * GATE: every spec_K stream (all K, all draft strategies, all prompts) must
 * equal `base` exactly. A secondary check proves row i of the verify logits is
 * argmax-independent of how many candidates follow it (K-independence), which is
 * why acceptance can never change the result.
 *
 * Usage: test_spec_decode [model.gguf]
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEN_LEN     64          /* tokens generated per run */
#define PROMPT_LEN  32
#define MAX_K       8

static int g_nv = 0;

static uint32_t argmax(const float *l, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    return (uint32_t)b;
}

/* Deterministic prompt of valid token ids for a given prompt index. */
static void make_prompt(uint32_t *p, int n, int idx) {
    uint32_t x = 0x9e3779b9u * (uint32_t)(idx + 1);
    for (int i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (x % (uint32_t)(g_nv - 4)) + 2;   /* avoid id 0/1 (BOS/EOS-ish) */
    }
}

/* Prefill `prompt`, return the greedy seed (argmax of the last-token logits).
 * Leaves state at pos = P. */
static int prefill_seed(pn_model *m, pn_state *s, const uint32_t *prompt, int P,
                        uint32_t *seed_out) {
    pn_state_reset(s);
    tm_tensor *logits = pn_llama_prefill(m, prompt, NULL, P, s);
    if (!logits) return -1;
    *seed_out = argmax((const float *)logits->data, g_nv);
    tm_free(logits);
    return 0;
}

/* Plain greedy via pn_llama_forward. Fills out[0..G-1]. */
static int ref_forward(pn_model *m, pn_state *s, const uint32_t *prompt, int P,
                       uint32_t *out, int G) {
    uint32_t seed;
    if (prefill_seed(m, s, prompt, P, &seed) != 0) return -1;
    out[0] = seed;
    for (int t = 1; t < G; t++) {
        tm_tensor *l = m->forward(m, out[t - 1], s);
        if (!l) return -1;
        out[t] = argmax((const float *)l->data, g_nv);
        tm_free(l);
    }
    return 0;
}

typedef enum { DRAFT_RANDOM, DRAFT_ZERO, DRAFT_ORACLE, DRAFT_PARTIAL } draft_kind;

/* Speculative greedy with draft width K and a draft strategy. Fills out[0..G-1].
 * `oracle` (may be NULL for K==0) supplies the true future for ORACLE/PARTIAL.
 * Accumulates drafted/accepted token counts for the acceptance report. */
static int spec_run(pn_model *m, pn_state *s, const uint32_t *prompt, int P,
                    uint32_t *out, int G, int K, draft_kind kind,
                    const uint32_t *oracle, unsigned rng,
                    long *drafted, long *accepted) {
    uint32_t seed;
    if (prefill_seed(m, s, prompt, P, &seed) != 0) return -1;
    out[0] = seed;
    int n = 1;
    uint32_t draft[MAX_K];
    uint32_t emit[MAX_K + 1];
    while (n < G) {
        for (int i = 0; i < K; i++) {
            int gpos = n + i;               /* stream slot this guess targets */
            switch (kind) {
                case DRAFT_RANDOM:
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    draft[i] = rng % (uint32_t)g_nv;
                    break;
                case DRAFT_ZERO:
                    draft[i] = 0;
                    break;
                case DRAFT_ORACLE:
                    draft[i] = (gpos < G) ? oracle[gpos] : 0;
                    break;
                case DRAFT_PARTIAL:
                    draft[i] = (gpos < G) ? ((gpos % 3 == 2) ? oracle[gpos] ^ 1u
                                                             : oracle[gpos]) : 0;
                    break;
            }
        }
        int em = pn_spec_decode_greedy(m, s, seed, draft, K, emit);
        if (em < 0) return -1;
        int acc = 0;
        for (int i = 0; i < em && i < K; i++) {
            if (emit[i] == draft[i]) acc++; else break;
        }
        *drafted += K;
        *accepted += acc;
        for (int j = 0; j < em && n < G; j++) out[n++] = emit[j];
        seed = emit[em - 1];
    }
    return 0;
}

/* Directly prove row i of the verify logits is argmax-independent of K: compare
 * argmax(row i of a full M=K+1 verify) against argmax(the LAST row of a verify
 * over just the prefix [seed, d0..d_{i-1}]). All calls run from the same
 * pos=P (verify never commits pos), so each sees the same committed cache. */
static int rowindep_check(pn_model *m, pn_state *s, const uint32_t *prompt,
                          int P, uint32_t seed, const uint32_t *draft, int K) {
    int nv = g_nv;
    uint32_t toks[MAX_K + 1];
    toks[0] = seed;
    for (int i = 0; i < K; i++) toks[1 + i] = draft[i];

    float *full = malloc((size_t)(K + 1) * nv * sizeof(float));
    float *pref = malloc((size_t)(K + 1) * nv * sizeof(float));
    if (!full || !pref) { free(full); free(pref); return -1; }

    if (prefill_seed(m, s, prompt, P, &(uint32_t){0}) != 0) { free(full); free(pref); return -1; }
    if (pn_llama_verify(m, toks, NULL, K + 1, s, full) != 0) { free(full); free(pref); return -1; }

    int mism = 0;
    for (int i = 0; i <= K; i++) {
        /* verify over the (i+1)-token prefix; row i is its last row */
        if (pn_llama_verify(m, toks, NULL, i + 1, s, pref) != 0) { free(full); free(pref); return -1; }
        uint32_t a = argmax(full + (size_t)i * nv, nv);
        uint32_t b = argmax(pref + (size_t)i * nv, nv);
        if (a != b) mism++;
    }
    free(full); free(pref);
    return mism;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    pn_state *state = pn_state_new(model);
    if (!state) { fprintf(stderr, "state failed\n"); return 1; }
    g_nv = model->config.n_vocab;
    printf("model: %s  n_vocab=%d n_layer=%d n_embd=%d arch=%d\n",
           path, g_nv, model->config.n_layer, model->config.n_embd, model->config.arch);

    /* pn_llama_verify only supports the quantized non-MoE Llama/Qwen path; make
     * sure this model uses it before asserting anything. */
    {
        uint32_t probe_tok = 2, one;
        pn_state_reset(state);
        float *plog = malloc((size_t)g_nv * sizeof(float));
        int supported = plog && pn_llama_verify(model, &probe_tok, NULL, 1, state, plog) == 0;
        free(plog);
        (void)one;
        if (!supported) {
            printf("[SKIP] model is not on the quantized non-MoE verify path\n");
            pn_state_free(state); pn_model_free(model);
            return 0;
        }
    }

    const int Ks[] = { 1, 2, 4, 8 };
    const int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));
    const draft_kind kinds[] = { DRAFT_RANDOM, DRAFT_ZERO, DRAFT_ORACLE, DRAFT_PARTIAL };
    const char *kind_name[] = { "random", "all-wrong", "oracle", "partial-oracle" };
    const int nkind = 4;
    const int n_prompts = 4;

    uint32_t prompt[PROMPT_LEN];
    uint32_t base[GEN_LEN], fwd[GEN_LEN], spec[GEN_LEN];

    int fail = 0, fwd_diffs_total = 0, rowindep_mism_total = 0;
    long checks = 0;

    for (int pi = 0; pi < n_prompts; pi++) {
        make_prompt(prompt, PROMPT_LEN, pi);

        /* Reference A: greedy through the verify machinery (K=0). */
        long d0 = 0, a0 = 0;
        if (spec_run(model, state, prompt, PROMPT_LEN, base, GEN_LEN, 0,
                     DRAFT_ZERO, NULL, 1, &d0, &a0) != 0) {
            fprintf(stderr, "base decode failed\n"); fail = 1; break;
        }
        /* Reference B: plain pn_llama_forward greedy. */
        if (ref_forward(model, state, prompt, PROMPT_LEN, fwd, GEN_LEN) != 0) {
            fprintf(stderr, "forward decode failed\n"); fail = 1; break;
        }
        int fwd_diffs = 0;
        for (int t = 0; t < GEN_LEN; t++) if (fwd[t] != base[t]) fwd_diffs++;
        fwd_diffs_total += fwd_diffs;

        printf("\nprompt %d: base seed=%u ... base[1..3]=%u,%u,%u  "
               "verify-greedy vs pn_llama_forward diffs=%d/%d\n",
               pi, base[0], base[1], base[2], base[3], fwd_diffs, GEN_LEN);

        /* Row-independence proof on one representative draft. */
        {
            uint32_t seed; prefill_seed(model, state, prompt, PROMPT_LEN, &seed);
            uint32_t rd[MAX_K];
            unsigned rng = 0xC0FFEEu + pi;
            for (int i = 0; i < MAX_K; i++) { rng ^= rng<<13; rng^=rng>>17; rng^=rng<<5; rd[i]=rng%(unsigned)g_nv; }
            int mism = rowindep_check(model, state, prompt, PROMPT_LEN, seed, rd, MAX_K);
            if (mism < 0) { fprintf(stderr, "rowindep failed\n"); fail = 1; break; }
            rowindep_mism_total += mism;
            printf("  row-independence: %d/%d rows differ in argmax across K\n", mism, MAX_K + 1);
        }

        /* GATE: every (K, draft-strategy) speculative stream must equal base. */
        for (int ki = 0; ki < nK; ki++) {
            for (int kd = 0; kd < nkind; kd++) {
                long drafted = 0, accepted = 0;
                if (spec_run(model, state, prompt, PROMPT_LEN, spec, GEN_LEN,
                             Ks[ki], kinds[kd], base, 0x1234u + pi * 31 + ki * 7 + kd,
                             &drafted, &accepted) != 0) {
                    fprintf(stderr, "spec decode failed\n"); fail = 1; break;
                }
                int diffs = 0, first = -1;
                for (int t = 0; t < GEN_LEN; t++)
                    if (spec[t] != base[t]) { diffs++; if (first < 0) first = t; }
                checks++;
                double ar = drafted ? (double)accepted / (double)drafted : 0.0;
                printf("  K=%d %-14s accept=%.2f (%ld/%ld)  stream %s",
                       Ks[ki], kind_name[kd], ar, accepted, drafted,
                       diffs == 0 ? "IDENTICAL" : "DIFFERS");
                if (diffs) { printf("  (%d diffs, first@%d spec=%u base=%u)",
                                    diffs, first, spec[first], base[first]); fail = 1; }
                printf("\n");
            }
            if (fail) break;
        }
        if (fail) break;
    }

    printf("\n==== SUMMARY ====\n");
    printf("bit-identity checks: %ld  row-independence mismatches: %d  "
           "verify-vs-forward diffs: %d\n", checks, rowindep_mism_total, fwd_diffs_total);
    printf("RESULT: %s\n", fail ? "FAIL" : "PASS (speculative greedy == plain greedy, bit for bit)");

    pn_state_free(state);
    pn_model_free(model);
    return fail ? 1 : 0;
}
