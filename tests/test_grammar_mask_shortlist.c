/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the grammar mask's shortlist decision
 * (src/core/grammar_mask_shortlist.c).
 *
 * Property, over thousands of random rows, allowed-sets and penalty settings:
 * whenever the shortlist DECIDES, the draw it leaves is the one the full
 * vocabulary mask leaves — greedy: the same argmax; sampling: the same top-K
 * set after the penalties (the only tokens the sampler's draw can reach).  Rows
 * come both dense and as a backend's exact top-64 row (reference = the dense
 * row it was cut from).  The test also requires the shortlist to decide most
 * rows (a decider that never decides would pass vacuously) and checks the
 * undecidable cases (logit_bias, raising penalties, top_k off) leave the row
 * untouched.
 *
 * Pure C, libc-only.
 */

#include "lib/test.h"
#include "../src/core/grammar_mask_shortlist.h"
#include "../src/core/sampling_modifiers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NV 3000
#define TOPN 64

static uint64_t g_rng = 88172645463325252ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rnd_unit(void) { return (float)(rnd() % 1000000) / 1000000.0f; }

typedef struct { uint32_t salt; uint32_t keep_per_100; } allow_rule;
static bool allowed_by_rule(void *ctx, uint32_t id) {
    const allow_rule *r = (const allow_rule *)ctx;
    return ((id * 2654435761u) ^ r->salt) % 100u < r->keep_per_100;
}

/* The sampler's view after a FULL mask: disallowed → -1e30, then the
 * penalties in the sampler's order. */
static void full_mask_view(const float *row, float *out, const gms_sampling *s,
                           const allow_rule *rule) {
    for (int i = 0; i < NV; i++)
        out[i] = allowed_by_rule((void *)rule, (uint32_t)i) ? row[i] : -1e30f;
    pn_apply_presence_frequency_penalties(out, NV, s->presence_penalty, s->frequency_penalty,
                                          s->recent, s->n_recent);
    if (s->temperature > 0.0f)
        for (int i = 0; i < s->n_recent; i++) {
            uint32_t t = s->recent[i];
            out[t] = out[t] > 0.0f ? out[t] / s->repeat_penalty : out[t] * s->repeat_penalty;
        }
}

/* The same view of the shortlist's rewritten row (-inf stays -inf). */
static void short_view(const float *rewritten, float *out, const gms_sampling *s) {
    memcpy(out, rewritten, sizeof(float) * NV);
    pn_apply_presence_frequency_penalties(out, NV, s->presence_penalty, s->frequency_penalty,
                                          s->recent, s->n_recent);
    if (s->temperature > 0.0f)
        for (int i = 0; i < s->n_recent; i++) {
            uint32_t t = s->recent[i];
            if (out[t] < -1e37f) continue;
            out[t] = out[t] > 0.0f ? out[t] / s->repeat_penalty : out[t] * s->repeat_penalty;
        }
}

static int argmax(const float *v) {
    int best = 0;
    for (int i = 1; i < NV; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Top-K membership: value >= the K-th largest (ties included, as the sampler). */
static void topk_set(const float *v, int k, bool *in) {
    float *tmp = malloc(sizeof(float) * NV);
    memcpy(tmp, v, sizeof(float) * NV);
    for (int a = 0; a < k; a++) {          /* partial selection, small k */
        int best = a;
        for (int i = a + 1; i < NV; i++) if (tmp[i] > tmp[best]) best = i;
        float t = tmp[a]; tmp[a] = tmp[best]; tmp[best] = t;
    }
    float kth = tmp[k - 1];
    for (int i = 0; i < NV; i++) in[i] = v[i] >= kth && v[i] > -1e29f;
    free(tmp);
}

/* An LM-like row: a long tail of noise and a few hundred plausible tokens. */
static void make_row(float *row) {
    for (int i = 0; i < NV; i++) row[i] = rnd_unit() * 8.0f - 4.0f;
    for (int j = 0; j < 300; j++) row[rnd() % NV] = 4.0f + rnd_unit() * rnd_unit() * 16.0f;
}

/* The backend's exact top-N presentation of a dense row. */
static float neg_inf(void) {
    uint32_t bits = 0xff800000u;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static void make_topn(const float *dense, float *topn) {
    bool taken[NV] = { false };
    for (int i = 0; i < NV; i++) topn[i] = neg_inf();
    for (int k = 0; k < TOPN; k++) {
        int best = -1;
        for (int i = 0; i < NV; i++)
            if (!taken[i] && (best < 0 || dense[i] > dense[best])) best = i;
        taken[best] = true;
        topn[best] = dense[best];
    }
}

static void test_shortlist_matches_full_mask(void) {
    static float dense[NV], row[NV], ref[NV], got[NV];
    uint32_t recent[64], ids[GMS_MAX];
    int decided = 0, trials = 0, mismatches = 0, decided_greedy = 0, decided_sampled = 0;
    for (int t = 0; t < 3000; t++) {
        make_row(dense);
        allow_rule rule = { rnd(), 5u + rnd() % 96u };
        /* The repeat window hits the row's TOP tokens (an agent re-emitting its
         * own recent text), with repeats, so penalties push shortlisted tokens
         * below the bound — the case the bound and the penalty replica exist
         * for. */
        uint32_t top[64], bound_ids[GMS_MAX];
        float top_bound;
        int n_top = gms_select_shortlist(dense, NV, bound_ids, &top_bound);
        for (int i = 0; i < 64; i++) top[i] = bound_ids[rnd() % (uint32_t)n_top];
        int n_recent = (int)(rnd() % 64);
        for (int i = 0; i < n_recent; i++)
            recent[i] = (i % 4 == 0) ? rnd() % NV : top[rnd() % 16];
        gms_sampling s = {
            .temperature = (t % 2) ? 0.7f : 0.0f, .top_k = 1 + (int)(rnd() % 40),
            .repeat_penalty = 1.0f + (float)(rnd() % 6) * 0.1f,
            .presence_penalty = (float)(rnd() % 3) * 1.0f,
            .frequency_penalty = (float)(rnd() % 3) * 0.5f,
            .recent = recent, .n_recent = n_recent,
        };
        bool use_topn = (t / 2) % 2;
        if (use_topn) make_topn(dense, row); else memcpy(row, dense, sizeof(row));
        float bound;
        int n = gms_select_shortlist(row, NV, ids, &bound);
        trials++;
        if (!gms_mask_from_shortlist(row, NV, ids, n, bound, &s, allowed_by_rule, &rule))
            continue;
        decided++;
        if (s.temperature <= 0.0f) decided_greedy++; else decided_sampled++;
        full_mask_view(dense, ref, &s, &rule);
        short_view(row, got, &s);
        if (s.temperature <= 0.0f) {
            if (argmax(ref) != argmax(got)) mismatches++;
        } else {
            bool in_ref[NV], in_got[NV];
            topk_set(ref, s.top_k, in_ref);
            topk_set(got, s.top_k, in_got);
            if (memcmp(in_ref, in_got, sizeof(in_ref)) != 0) mismatches++;
        }
    }
    printf("    trials=%d decided=%d (greedy %d, sampled %d) mismatches=%d\n", trials, decided,
           decided_greedy, decided_sampled, mismatches);
    ASSERT_INT_EQ(mismatches, 0);
    ASSERT_TRUE(decided_greedy > trials / 4);
    ASSERT_TRUE(decided_sampled > trials / 20);
}

static void test_undecidable_leaves_row(void) {
    static float row[NV], before[NV];
    uint32_t ids[GMS_MAX];
    float bound;
    allow_rule all = { 0, 100 };
    make_row(row);
    memcpy(before, row, sizeof(row));
    int n = gms_select_shortlist(row, NV, ids, &bound);
    gms_sampling bias = { .temperature = 0.0f, .top_k = 40, .repeat_penalty = 1.0f,
                          .has_logit_bias = true };
    ASSERT_FALSE(gms_mask_from_shortlist(row, NV, ids, n, bound, &bias, allowed_by_rule, &all));
    gms_sampling raising = { .temperature = 0.7f, .top_k = 40, .repeat_penalty = 0.9f };
    ASSERT_FALSE(gms_mask_from_shortlist(row, NV, ids, n, bound, &raising, allowed_by_rule, &all));
    gms_sampling no_topk = { .temperature = 0.7f, .top_k = 0, .repeat_penalty = 1.0f };
    ASSERT_FALSE(gms_mask_from_shortlist(row, NV, ids, n, bound, &no_topk, allowed_by_rule, &all));
    allow_rule none = { 0, 0 };
    gms_sampling greedy = { .temperature = 0.0f, .top_k = 40, .repeat_penalty = 1.0f };
    ASSERT_FALSE(gms_mask_from_shortlist(row, NV, ids, n, bound, &greedy, allowed_by_rule, &none));
    ASSERT_TRUE(memcmp(row, before, sizeof(row)) == 0);
}

static void test_select_shortlist(void) {
    static float row[NV], topn[NV];
    uint32_t ids[GMS_MAX];
    float bound;
    make_row(row);
    int n = gms_select_shortlist(row, NV, ids, &bound);
    ASSERT_INT_EQ(n, GMS_MAX);
    int above = 0;
    for (int i = 0; i < NV; i++) if (row[i] > bound) above++;
    ASSERT_TRUE(above <= GMS_MAX);
    for (int i = 0; i < n; i++) ASSERT_TRUE(row[ids[i]] >= bound);
    make_topn(row, topn);
    n = gms_select_shortlist(topn, NV, ids, &bound);
    ASSERT_INT_EQ(n, TOPN);
    float min = 1e30f;
    for (int i = 0; i < n; i++) if (topn[ids[i]] < min) min = topn[ids[i]];
    ASSERT_TRUE(bound == min);
}

int main(void) {
    printf("\n=== grammar_mask_shortlist ===\n\n");
    RUN_TEST(select_shortlist);
    RUN_TEST(undecidable_leaves_row);
    RUN_TEST(shortlist_matches_full_mask);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
