/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Speculative greedy decode gate + bench for qwen35moe on a K80 host.
 *
 * Phase 1 (reference): plain greedy decode of n_gen tokens from the prefix
 * via single-token forwards. Phase 2: speculative greedy decode of the same
 * n_gen tokens — prompt-lookup n-gram drafts (pn_ngram_draft over prefix +
 * committed output) verified in one batched pass (pn_qwen35moe_spec_verify /
 * pn_qwen35moe_spec_accept), falling back to a plain forward when no draft
 * matches. The two token streams MUST be identical (bit-exact gate); the
 * bench lines report decode tok/s for both, the speedup, and the acceptance
 * stats.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_spec_qwen35
 * Run:  QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *         QWEN35_FUSE=0x37 ./tests/test_spec_qwen35 <model.gguf> [n_gen] \
 *         [rep] [tok0 ...]
 *   n_gen  tokens to decode (default 64)
 *   rep    repeat the prefix sequence rep times (default 1) — a repetitive
 *          prefix is the n-gram drafter's favorable case
 *   tok0.. prefix tokens (default: the 12-token reference sequence)
 * Env: QWEN35_SPEC_K (max drafts/verify, default 6), QWEN35_SPEC_NG_MAX (3),
 *      QWEN35_SPEC_NG_MIN (2).
 * QWEN35_FUSE=0x37 makes the single-token attn path float-identical to the
 * batched verify kernels (the fused qknorm+rope contraction differs by ~1
 * ulp); the strict gate runs under it.
 * NOT part of the test suite (needs the model + a CUDA GPU). */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/core/ngram_draft.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define SPEC_MAX_PREFIX 4096
#define SPEC_MAX_GEN    2048
#define SPEC_MAX_K      32

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static int env_int(const char *name, int def)
{
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

/* Feed the prefix through single-token forwards; return the LAST logits
 * tensor (caller frees) or NULL. */
static tm_tensor *prefill(pn_model *model, pn_state *st,
                          const uint32_t *prefix, int n_prefix)
{
    tm_tensor *out = NULL;
    for (int i = 0; i < n_prefix; i++) {
        if (out) tm_free(out);
        out = model->forward(model, prefix[i], st);
        if (!out) return NULL;
    }
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_gen] [rep] [tok0 ...]\n",
                argv[0]);
        return 1;
    }

    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int n_vocab = model->config.n_vocab;

    int n_gen = argc > 2 ? atoi(argv[2]) : 64;
    if (n_gen < 2) n_gen = 2;
    if (n_gen > SPEC_MAX_GEN) n_gen = SPEC_MAX_GEN;
    int rep = argc > 3 ? atoi(argv[3]) : 1;
    if (rep < 1) rep = 1;

    static const uint32_t ref_toks[12] = { 9419, 11, 100, 200, 300, 400,
                                           500, 600, 700, 800, 900, 1000 };
    uint32_t unit[SPEC_MAX_PREFIX];
    int n_unit;
    if (argc > 4) {
        n_unit = argc - 4 < SPEC_MAX_PREFIX ? argc - 4 : SPEC_MAX_PREFIX;
        for (int i = 0; i < n_unit; i++)
            unit[i] = (uint32_t)atoi(argv[4 + i]);
    } else {
        n_unit = 12;
        memcpy(unit, ref_toks, sizeof(ref_toks));
    }
    static uint32_t prefix[SPEC_MAX_PREFIX];
    int n_prefix = 0;
    for (int r = 0; r < rep && n_prefix + n_unit <= SPEC_MAX_PREFIX; r++) {
        memcpy(prefix + n_prefix, unit, (size_t)n_unit * sizeof(uint32_t));
        n_prefix += n_unit;
    }

    int max_k = env_int("QWEN35_SPEC_K", 6);
    if (max_k < 1) max_k = 1;
    if (max_k > SPEC_MAX_K) max_k = SPEC_MAX_K;
    int ng_max = env_int("QWEN35_SPEC_NG_MAX", 3);
    int ng_min = env_int("QWEN35_SPEC_NG_MIN", 3);
    pn_spec_throttle thr;
    pn_spec_throttle_init(&thr, env_int("QWEN35_SPEC_MIN_ACC", 4),
                          env_int("QWEN35_SPEC_MISS", 2),
                          env_int("QWEN35_SPEC_COOL", 48));

    fprintf(stderr,
            "spec gate: prefix=%d (unit %d x %d) n_gen=%d K=%d ngram=%d..%d\n",
            n_prefix, n_unit, rep, n_gen, max_k, ng_min, ng_max);

    static uint32_t gen_plain[SPEC_MAX_GEN], gen_spec[SPEC_MAX_GEN];
    static uint32_t hist[SPEC_MAX_PREFIX + SPEC_MAX_GEN + SPEC_MAX_K];

    /* ---- Phase 1: plain greedy reference. ---- */
    pn_state *st = pn_state_new(model);
    if (!st) { fprintf(stderr, "state failed\n"); return 1; }
    tm_tensor *out = prefill(model, st, prefix, n_prefix);
    if (!out) { fprintf(stderr, "prefill failed\n"); return 1; }
    double t0 = now_s();
    uint32_t tok = (uint32_t)argmax(tm_data_f32(out), n_vocab);
    gen_plain[0] = tok;
    for (int i = 1; i < n_gen; i++) {
        tm_free(out);
        out = model->forward(model, tok, st);
        if (!out) { fprintf(stderr, "plain forward failed @%d\n", i); return 1; }
        tok = (uint32_t)argmax(tm_data_f32(out), n_vocab);
        gen_plain[i] = tok;
    }
    double dt_plain = now_s() - t0;
    tm_free(out);
    pn_state_free(st);

    /* ---- Phase 2: speculative greedy decode. ---- */
    st = pn_state_new(model);
    if (!st) { fprintf(stderr, "state failed\n"); return 1; }
    out = prefill(model, st, prefix, n_prefix);
    if (!out) { fprintf(stderr, "spec prefill failed\n"); return 1; }

    long verifies = 0, drafted = 0, accepted = 0, plain_steps = 0;
    t0 = now_s();
    uint32_t pending = (uint32_t)argmax(tm_data_f32(out), n_vocab);
    tm_free(out);
    out = NULL;
    memcpy(hist, prefix, (size_t)n_prefix * sizeof(uint32_t));
    int nh = n_prefix;
    gen_spec[0] = pending;
    hist[nh++] = pending;
    int n_emitted = 1;

    while (n_emitted < n_gen) {
        uint32_t drafts[SPEC_MAX_K], block[SPEC_MAX_K + 1];
        int32_t am[SPEC_MAX_K + 1];
        int k = 0;
        int stepped = 0;

        if (pn_spec_throttle_gate(&thr))
            k = pn_ngram_draft(hist, nh, ng_max, ng_min, max_k, drafts);
        if (k >= thr.min_accept) {
            block[0] = pending;
            memcpy(block + 1, drafts, (size_t)k * sizeof(uint32_t));
            if (pn_qwen35moe_spec_verify(model, block, k + 1, st, am) == 0) {
                int acc = 1;
                while (acc <= k && drafts[acc - 1] == (uint32_t)am[acc - 1])
                    acc++;
                if (pn_qwen35moe_spec_accept(model, st, acc) != 0) {
                    fprintf(stderr, "spec accept failed\n");
                    return 1;
                }
                verifies++;
                drafted += k;
                accepted += acc - 1;
                pn_spec_throttle_report(&thr, acc - 1);
                for (int i = 0; i < acc && n_emitted < n_gen; i++) {
                    gen_spec[n_emitted++] = (uint32_t)am[i];
                    hist[nh++] = (uint32_t)am[i];
                }
                pending = (uint32_t)am[acc - 1];
                stepped = 1;
            }
        }
        if (!stepped) {
            out = model->forward(model, pending, st);
            if (!out) { fprintf(stderr, "spec forward failed\n"); return 1; }
            pending = (uint32_t)argmax(tm_data_f32(out), n_vocab);
            tm_free(out);
            out = NULL;
            plain_steps++;
            gen_spec[n_emitted++] = pending;
            hist[nh++] = pending;
        }
    }
    double dt_spec = now_s() - t0;
    pn_state_free(st);

    /* ---- Gate: identical token streams. ---- */
    int first_diff = -1;
    for (int i = 0; i < n_gen; i++) {
        if (gen_plain[i] != gen_spec[i]) { first_diff = i; break; }
    }
    fprintf(stderr, "plain :");
    for (int i = 0; i < n_gen; i++) fprintf(stderr, " %u", gen_plain[i]);
    fprintf(stderr, "\nspec  :");
    for (int i = 0; i < n_gen; i++) fprintf(stderr, " %u", gen_spec[i]);
    fprintf(stderr, "\n");

    double fwd_plain = n_gen - 1;                  /* forwards after seed */
    double fwd_spec = (double)(verifies + plain_steps);
    fprintf(stderr,
            "BENCH plain %.3fs (%.2f tok/s) | spec %.3fs (%.2f tok/s) | "
            "speedup %.2fx\n",
            dt_plain, fwd_plain / dt_plain, dt_spec, fwd_plain / dt_spec,
            dt_plain / dt_spec);
    fprintf(stderr,
            "SPEC verifies=%ld plain_steps=%ld drafted=%ld accepted=%ld "
            "accept_rate=%.2f tokens/forward=%.2f\n",
            verifies, plain_steps, drafted, accepted,
            drafted ? (double)accepted / (double)drafted : 0.0,
            fwd_spec ? (double)(n_gen - 1) / fwd_spec : 0.0);
    if (first_diff < 0)
        fprintf(stderr, "GATE: BIT-EXACT PASS\n");
    else
        fprintf(stderr, "GATE: FAIL first_diff=%d (plain=%u spec=%u)\n",
                first_diff, gen_plain[first_diff], gen_spec[first_diff]);

    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(first_diff < 0 ? 0 : 1);
    return first_diff < 0 ? 0 : 1;
}
#endif
