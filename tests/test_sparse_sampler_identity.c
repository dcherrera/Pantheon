/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_sparse_sampler_identity.c
 * @brief Ship gate for the sparse-survivor sampler: identical token, or fail.
 *
 * @status Complete.  CPU-only — no GPU, no model.
 * @issues None
 * @todo None
 *
 * The fast path in engine.c samples over the <=k finite survivors of a sparse
 * logits row instead of walking all n_vocab entries eight times.  It ships only
 * if it is BIT-EXACT: for the same row, the same params and the same RNG state
 * it must return the token the full-vocab path returns.  Not "close", not
 * "same distribution" — the same token, or generation diverges from every
 * existing reference.
 *
 * TWO THINGS THIS GATE DOES THAT THE PREVIOUS ONE DID NOT
 *
 * 1. It asserts the fast path RAN.  pn_sampler_sparse_hits() must advance on
 *    every arm-B call.  The previous attempt at this optimisation detected
 *    survivors with `logits[i] > -INFINITY`, which -ffast-math folds to a
 *    constant true; every row therefore looked dense, the fast path never once
 *    executed, and the bit-exactness gate passed 18/18 by comparing the slow
 *    path against itself.  A green gate over an inert change is worse than a
 *    red one.
 *
 * 2. It proves it can go red.  `--selftest` runs the same comparison against a
 *    deliberately WRONG arm (top_p nudged off the requested value, which moves
 *    the nucleus cut and therefore the draw).  If that still reports a match,
 *    the comparison is not sensitive to the sampler's decision and no green
 *    result from it means anything.  `--selftest` exits 0 only when the
 *    mismatch is detected.
 *
 * Usage: test_sparse_sampler_identity [--selftest]
 */
#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Smaller than the real 248320 so the grid runs in a second; the fast path's
 * correctness does not depend on n_vocab, only on the row being sparse. A
 * final case re-runs at the production vocab to be sure. */
#define VOCAB 20011

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

/* Exactly what gpu_present_topk leaves in the host row: -inf everywhere, k
 * survivors at scattered ascending-scanned ids. `spread` shapes the survivor
 * logits so the grid covers ties, wide gaps and underflow-to-zero probs. */
typedef enum { SPREAD_WIDE, SPREAD_TIGHT, SPREAD_ALL_EQUAL, SPREAD_EXTREME } spread_t;

static void build_row(float *row, int n_vocab, int k, spread_t sp, uint64_t seed)
{
    for (int i = 0; i < n_vocab; i++) row[i] = -INFINITY;
    uint64_t s = seed ? seed : 1;
    int placed = 0;
    while (placed < k) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        int id = (int)((s >> 33) % (uint64_t)n_vocab);
        uint32_t bits;
        memcpy(&bits, &row[id], sizeof bits);
        if (bits != 0xff800000u) continue;      /* already a survivor */
        float v;
        switch (sp) {
        case SPREAD_WIDE:      v = 12.0f - 0.9f * (float)placed; break;
        case SPREAD_TIGHT:     v = 4.0f - 0.0009f * (float)placed; break;
        case SPREAD_ALL_EQUAL: v = 3.25f; break;
        default:               v = 30.0f - 25.0f * (float)placed; break; /* underflow tail */
        }
        row[id] = v;
        placed++;
    }
}

/* One arm: fresh copy of the row, fresh RNG state from `seed`, one sample. */
static uint32_t run_arm(const float *row, int n_vocab, pn_sampling_params p,
                        uint32_t *recent, int n_recent, uint64_t seed, int sparse_on)
{
    float *buf = malloc((size_t)n_vocab * sizeof(float));
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    memcpy(buf, row, (size_t)n_vocab * sizeof(float));
    p.rng_seeded = 1;
    p.rng_state  = pn_seed_to_rng_state(seed);
    pn_sampler_set_sparse(sparse_on);
    uint32_t tok = pn_sample_next_token(buf, n_vocab, &p, recent, n_recent);
    pn_sampler_clear_sparse();
    free(buf);
    return tok;
}

static const char *spread_name(spread_t s)
{
    switch (s) {
    case SPREAD_WIDE: return "wide";
    case SPREAD_TIGHT: return "tight";
    case SPREAD_ALL_EQUAL: return "all-equal";
    default: return "extreme";
    }
}

/* Compare the two arms over one configuration. Returns 1 when they match. */
static int compare(int n_vocab, int k, spread_t sp, float temp, int top_k,
                   float top_p, float rp, uint64_t seed, int verbose,
                   float top_p_arm_b)
{
    float *row = malloc((size_t)n_vocab * sizeof(float));
    if (!row) { fprintf(stderr, "oom\n"); exit(1); }
    build_row(row, n_vocab, k, sp, seed * 7919u + 13u);

    /* A repetition window that deliberately straddles the survivor set: some
     * entries are survivors (the penalty moves their logit), some are -inf
     * non-survivors (the penalty must be a no-op that leaves them -inf, which
     * is what lets the fast path ignore them). */
    uint32_t recent[64];
    int n_recent = 64;
    int found = 0;
    for (int i = 0; i < n_vocab && found < 8; i++) {
        uint32_t bits; memcpy(&bits, &row[i], sizeof bits);
        if (bits != 0xff800000u) recent[found++] = (uint32_t)i;
    }
    for (int i = found; i < n_recent; i++)
        recent[i] = (uint32_t)((i * 977u + 5u) % (unsigned)n_vocab);

    pn_sampling_params p = pn_sampling_defaults();
    p.temperature = temp;
    p.top_k = top_k;
    p.top_p = top_p;
    p.repeat_penalty = rp;
    p.repeat_last_n = n_recent;

    unsigned long hits_before = pn_sampler_sparse_hits();
    uint32_t a = run_arm(row, n_vocab, p, recent, n_recent, seed, 0);
    pn_sampling_params pb = p;
    pb.top_p = top_p_arm_b;                 /* == top_p except in --selftest */
    uint32_t b = run_arm(row, n_vocab, pb, recent, n_recent, seed, 1);
    unsigned long hits_after = pn_sampler_sparse_hits();

    int ran = (hits_after == hits_before + 1);
    int match = (a == b);

    if (verbose || !match || !ran) {
        printf("  k=%-4d %-9s temp=%.2f top_k=%-6d top_p=%.2f rp=%.2f seed=%llu"
               " -> A=%u B=%u %s%s\n",
               k, spread_name(sp), (double)temp, top_k, (double)top_p, (double)rp,
               (unsigned long long)seed, a, b,
               match ? "match" : "MISMATCH",
               ran ? "" : "  [fast path DID NOT RUN]");
    }

    free(row);
    if (!ran) return -1;    /* vacuous: the arms were the same code */
    return match;
}

int main(int argc, char **argv)
{
    int selftest = (argc > 1 && strcmp(argv[1], "--selftest") == 0);

    if (selftest) {
        /* Falsification: arm B samples with a top_p that is NOT the requested
         * one. The nucleus cut moves, so the draw must land somewhere else at
         * least once. A harness that reports "all match" here is measuring
         * nothing and no green run from it can be trusted. */
        printf("SELFTEST: arm B given the wrong top_p — the gate must go RED\n");
        int mismatches = 0, vacuous = 0, cases = 0;
        for (uint64_t seed = 1; seed <= 60; seed++) {
            int r = compare(VOCAB, 32, SPREAD_TIGHT, 0.7f, 40, 0.95f, 1.1f,
                            seed, 0, 0.30f);
            cases++;
            if (r < 0) vacuous++;
            else if (r == 0) mismatches++;
        }
        printf("  %d cases, %d mismatches detected, %d vacuous\n",
               cases, mismatches, vacuous);
        if (vacuous) {
            printf("SELFTEST FAILED: the fast path did not run — comparison is vacuous\n");
            return 1;
        }
        if (mismatches == 0) {
            printf("SELFTEST FAILED: a deliberately wrong arm still matched — "
                   "this gate cannot detect a broken sampler\n");
            return 1;
        }
        printf("SELFTEST PASSED: the gate detects a wrong sampler (%d/%d cases)\n",
               mismatches, cases);
        return 0;
    }

    printf("Sparse-survivor sampler — token-identity gate (vocab %d)\n", VOCAB);

    const float temps[]  = { 0.3f, 0.7f, 1.0f, 2.0f };
    const int   topks[]  = { 1, 5, 40, 0, 100000 };
    const float toppz[]  = { 1.0f, 0.95f, 0.5f, 0.1f };
    const float rps[]    = { 1.0f, 1.1f };
    const int   ks[]     = { 1, 2, 8, 32, 40, 256 };
    const spread_t sps[] = { SPREAD_WIDE, SPREAD_TIGHT, SPREAD_ALL_EQUAL, SPREAD_EXTREME };

    int vacuous = 0;
    uint64_t seed = 1;
    for (size_t a = 0; a < sizeof temps / sizeof *temps; a++)
    for (size_t b = 0; b < sizeof topks / sizeof *topks; b++)
    for (size_t c = 0; c < sizeof toppz / sizeof *toppz; c++)
    for (size_t d = 0; d < sizeof rps / sizeof *rps; d++)
    for (size_t e = 0; e < sizeof ks / sizeof *ks; e++)
    for (size_t f = 0; f < sizeof sps / sizeof *sps; f++) {
        int r = compare(VOCAB, ks[e], sps[f], temps[a], topks[b], toppz[c],
                        rps[d], seed++, 0, toppz[c]);
        if (r < 0) { vacuous++; g_fail++; g_checks++; }
        else CHECK(r == 1, "token mismatch at case %llu", (unsigned long long)seed - 1);
    }
    printf("  grid: %d comparisons, %d failures, %d vacuous\n",
           g_checks, g_fail, vacuous);

    /* Production shape: the real vocab, the real serve defaults, many seeds. */
    printf("Production shape (vocab 248320, k=40, temp 0.7, top_k 40, top_p 0.9, rp 1.1)\n");
    int prod_bad = 0, prod_vacuous = 0;
    for (uint64_t s = 1000; s < 1200; s++) {
        int r = compare(248320, 40, SPREAD_TIGHT, 0.7f, 40, 0.9f, 1.1f, s, 0, 0.9f);
        if (r < 0) prod_vacuous++;
        else if (r == 0) prod_bad++;
    }
    g_checks += 200;
    if (prod_bad || prod_vacuous) g_fail += prod_bad + prod_vacuous;
    printf("  200 turns: %d mismatches, %d vacuous\n", prod_bad, prod_vacuous);

    /* Greedy (temp 0) takes the SAME fast path since the argmax addition: it
     * must RUN and it must return the identical token.  The spreads matter
     * here — SPREAD_ALL_EQUAL is the tie-break case, where every survivor
     * carries the same logit and both scans must land on the LOWEST survivor
     * id, and SPREAD_EXTREME puts the winner in a 25-per-step tail. */
    printf("Greedy (temp 0) must take the fast path and return the same token\n");
    {
        const spread_t gsp[] = { SPREAD_WIDE, SPREAD_TIGHT, SPREAD_ALL_EQUAL,
                                 SPREAD_EXTREME };
        const int gks[] = { 1, 2, 40, 256 };
        for (size_t i = 0; i < sizeof gsp / sizeof *gsp; i++)
        for (size_t j = 0; j < sizeof gks / sizeof *gks; j++) {
            float *row = malloc(248320 * sizeof(float));
            if (!row) { fprintf(stderr, "oom\n"); exit(1); }
            build_row(row, 248320, gks[j], gsp[i], 4242 + i * 31 + j);
            pn_sampling_params p = pn_sampling_defaults();
            p.temperature = 0.0f;
            unsigned long h0 = pn_sampler_sparse_hits();
            uint32_t ga = run_arm(row, 248320, p, NULL, 0, 7, 0);
            uint32_t gb = run_arm(row, 248320, p, NULL, 0, 7, 1);
            unsigned long h1 = pn_sampler_sparse_hits();
            CHECK(ga == gb, "greedy token changed (%s k=%d): %u vs %u",
                  spread_name(gsp[i]), gks[j], ga, gb);
            CHECK(h1 == h0 + 1, "greedy fast path did not run (%s k=%d, %lu hits)",
                  spread_name(gsp[i]), gks[j], h1 - h0);
            free(row);
        }
    }

    /* Greedy on a DENSE row must still fall through — full_logits mode hands
     * the sampler a real row and there is no survivor set to shortcut. */
    printf("Greedy on a dense row must fall through to the full scan\n");
    {
        int nv = 5000;
        float *row = malloc((size_t)nv * sizeof(float));
        if (!row) { fprintf(stderr, "oom\n"); exit(1); }
        uint64_t s = 424242;
        for (int i = 0; i < nv; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            row[i] = (float)((int)((s >> 33) % 2000) - 1000) * 0.01f;
        }
        pn_sampling_params p = pn_sampling_defaults();
        p.temperature = 0.0f;
        unsigned long h0 = pn_sampler_sparse_hits();
        uint32_t da = run_arm(row, nv, p, NULL, 0, 31, 0);
        uint32_t db = run_arm(row, nv, p, NULL, 0, 31, 1);
        unsigned long h1 = pn_sampler_sparse_hits();
        CHECK(da == db, "dense greedy row diverged: %u vs %u", da, db);
        CHECK(h1 == h0, "the fast path ran on a dense greedy row (%lu hits)",
              h1 - h0);
        free(row);
    }

    /* A DENSE row (what full_logits mode hands the sampler for a tool-grammar,
     * logit_bias or logprobs turn) must fall through, not be mis-sampled. */
    printf("Dense row must fall through to the full path\n");
    {
        int nv = 5000;
        float *row = malloc((size_t)nv * sizeof(float));
        uint64_t s = 99;
        for (int i = 0; i < nv; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            row[i] = (float)((int)((s >> 33) % 2000) - 1000) * 0.01f;
        }
        pn_sampling_params p = pn_sampling_defaults();
        p.temperature = 0.7f;
        unsigned long h0 = pn_sampler_sparse_hits();
        uint32_t da = run_arm(row, nv, p, NULL, 0, 31, 0);
        uint32_t db = run_arm(row, nv, p, NULL, 0, 31, 1);
        unsigned long h1 = pn_sampler_sparse_hits();
        CHECK(da == db, "dense row diverged: %u vs %u", da, db);
        CHECK(h1 == h0, "the fast path ran on a dense row (%lu hits)", h1 - h0);
        free(row);
    }

    printf("\n%s: %d checks, %d failures\n", g_fail ? "FAILED" : "PASSED",
           g_checks, g_fail);
    return g_fail ? 1 : 0;
}
