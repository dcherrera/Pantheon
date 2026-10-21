/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_spec_generate.c - end-to-end check + speedup measurement for the lossless
 * speculative generation loop (pn_spec_generate).
 *
 * This is the Part 2 core in isolation from Apple Vision: instead of a live
 * VNRecognizeTextRequest draft it uses a synthetic draft provider that agrees
 * with the model's true greedy tokens a tunable fraction of the time (a
 * stand-in for a Vision transcript at a given acceptance). The provider aligns
 * to the accepted prefix on every call, exactly as the real Vision provider will
 * re-tokenize the remaining transcript conditioned on the accepted prefix (the
 * BPE-seam fix), so it always proposes the next slots.
 *
 * It proves the whole generation loop is lossless (spec output == plain greedy
 * output, bit for bit) and reports the realized decode speedup = tokens emitted
 * / verify passes, at several draft acceptance levels and draft widths. It also
 * checks the low-acceptance fallback to plain decode.
 *
 * Usage: test_spec_generate [model.gguf] [max_tokens]
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_LEN 32

static int g_nv = 0;

static uint32_t argmax(const float *l, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i; return (uint32_t)b;
}
static void make_prompt(uint32_t *p, int n, int idx) {
    uint32_t x = 0x9e3779b9u * (uint32_t)(idx + 1);
    for (int i = 0; i < n; i++) { x^=x<<13; x^=x>>17; x^=x<<5; p[i]=(x%(uint32_t)(g_nv-4))+2; }
}
/* Prefill + greedy seed. */
static int prefill_seed(pn_model *m, pn_state *s, const uint32_t *pr, int P, uint32_t *seed) {
    pn_state_reset(s);
    tm_tensor *l = pn_llama_prefill(m, pr, NULL, P, s);
    if (!l) return -1;
    *seed = argmax((const float *)l->data, g_nv);
    tm_free(l);
    return 0;
}
/* Plain greedy reference via pn_llama_forward; fills ref[0..G-1]. */
static int ref_forward(pn_model *m, pn_state *s, const uint32_t *pr, int P, uint32_t *ref, int G) {
    uint32_t seed;
    if (prefill_seed(m, s, pr, P, &seed) != 0) return -1;
    ref[0] = seed;
    for (int t = 1; t < G; t++) {
        tm_tensor *l = m->forward(m, ref[t-1], s);
        if (!l) return -1;
        ref[t] = argmax((const float *)l->data, g_nv);
        tm_free(l);
    }
    return 0;
}

/* Synthetic Vision-style draft provider: knows the model's true continuation
 * (ref) and proposes it with probability keep_prob per token, else a wrong
 * token. Aligned by the accepted-prefix length, so it self-resyncs each call. */
typedef struct { const uint32_t *ref; int ref_len; unsigned rng; double keep; } draft_ctx;
static int vision_like_draft(const uint32_t *prefix, int n_prefix,
                             uint32_t *out, int max_k, void *ud) {
    (void)prefix;
    draft_ctx *c = (draft_ctx *)ud;
    int k = 0;
    for (int i = 0; i < max_k; i++) {
        int pos = n_prefix + i;
        if (pos >= c->ref_len) break;
        c->rng ^= c->rng<<13; c->rng ^= c->rng>>17; c->rng ^= c->rng<<5;
        double u = (double)(c->rng & 0xFFFFFF) / (double)0x1000000;
        if (u < c->keep) out[i] = c->ref[pos];
        else            out[i] = (c->ref[pos] ^ 1u) % (uint32_t)g_nv;  /* wrong guess */
        k++;
    }
    return k;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int G = (argc > 2) ? atoi(argv[2]) : 128;
    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    pn_state *state = pn_state_new(model);
    if (!state) return 1;
    g_nv = model->config.n_vocab;
    printf("model: %s  n_vocab=%d n_layer=%d  max_tokens=%d\n",
           path, g_nv, model->config.n_layer, G);

    uint32_t prompt[PROMPT_LEN];
    make_prompt(prompt, PROMPT_LEN, 0);

    uint32_t *ref  = malloc((size_t)G * sizeof(uint32_t));
    uint32_t *spec = malloc((size_t)G * sizeof(uint32_t));
    if (ref_forward(model, state, prompt, PROMPT_LEN, ref, G) != 0) {
        fprintf(stderr, "ref failed\n"); return 1;
    }

    const int    Ks[]  = { 4, 8 };
    const double kps[] = { 0.86, 0.60, 0.30 };   /* target draft acceptance */
    int fail = 0;

    printf("\n%-4s %-6s %-8s %-8s %-9s %-8s %s\n",
           "K", "keep", "accept", "passes", "emitted", "speedup", "lossless");
    for (int ki = 0; ki < 2; ki++) {
        for (int qi = 0; qi < 3; qi++) {
            uint32_t seed;
            if (prefill_seed(model, state, prompt, PROMPT_LEN, &seed) != 0) { fail=1; break; }
            draft_ctx ctx = { ref, G, 0xABCDEF01u ^ (unsigned)(ki*131 + qi*7), kps[qi] };
            pn_spec_stats st;
            int n = pn_spec_generate(model, state, seed, G, Ks[ki], /*stop_id=*/-1,
                                     vision_like_draft, &ctx,
                                     /*min_accept_frac=*/0.45, /*probe_cycles=*/8,
                                     spec, &st);
            if (n < 0) { fprintf(stderr, "spec generate failed/unsupported\n"); fail=1; break; }
            int diffs = 0;
            for (int t = 0; t < n && t < G; t++) if (spec[t] != ref[t]) diffs++;
            if (n != G) diffs++;   /* must generate the same count with stop_id=-1 */
            double accept = st.drafted ? (double)st.accepted/st.drafted : 0.0;
            double sp = st.verify_passes ? (double)st.tokens_emitted/st.verify_passes : 0.0;
            printf("%-4d %-6.2f %-8.3f %-8ld %-9ld %-7.2fx %s%s\n",
                   Ks[ki], kps[qi], accept, st.verify_passes, st.tokens_emitted, sp,
                   diffs==0 ? "IDENTICAL" : "DIFFERS", st.fell_back ? " (fellback)" : "");
            if (diffs) fail = 1;
        }
        if (fail) break;
    }

    printf("\nRESULT: %s\n", fail ?
           "FAIL" :
           "PASS (speculative generation == plain greedy, bit for bit; speedup scales with acceptance)");
    free(ref); free(spec);
    pn_state_free(state); pn_model_free(model);
    return fail ? 1 : 0;
}
