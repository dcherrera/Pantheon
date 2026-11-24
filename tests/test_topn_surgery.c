/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_topn_surgery.c
 * @brief Gate for the top-N verdict under client logit surgery (sampler_topn_surgery.h): when
 *        pn_sampler_topn_row_verdict says a top-64 row decides a draw carrying a logit bias or
 *        a presence/frequency penalty, the dense row really decides it the same way.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Random rows (vocab 4096, a heavy head), repeat windows that hit the head, presence/frequency
 * penalties (sometimes negative), and logit biases on head and tail tokens (positive and
 * negative, some large enough to lift a tail token over the head).
 *   GREEDY (temperature 0): truth = the dense row's argmax after the surgery (lowest id on a
 *     tie) equals the survivors' argmax after the same surgery. EXACT over a false truth is a
 *     FAILURE.
 *   SAMPLED with a bias, top_k 1..64: truth = the dense row's top-k set after the surgery and
 *     the repetition penalty lies inside the 64.
 * Both verdicts must occur in both parts. `--selftest`: a verdict computed blind to the bias
 * must be caught.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"

#define NV 4096
#define N 64
#define TRIALS 4000

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }
static float xf(void) { return (float)(xr() & 0xFFFFFF) / 16777216.0f; }

typedef struct {
    float row[NV], sparse[NV], adj[NV];
    int top[N], in[NV];
    uint32_t recent[64];
    int n_recent;
    pn_logit_bias_entry bias[8];
} trial_row;

/* A random row, its top-64 (the card's kernel order), a window and a bias list. */
static void make_trial(trial_row *t, pn_sampling_params *p, int greedy)
{
    int i, j, n = 0;

    for (i = 0; i < NV; i++) t->row[i] = 4.0f * xf() - 2.0f;
    for (i = 0; i < 100; i++) t->row[xr() % NV] = 6.0f + 6.0f * xf();
    for (i = 0; i < NV; i++) {
        if (n < N || t->row[i] > t->row[t->top[n - 1]]) {
            int at = n < N ? n++ : N - 1;
            t->top[at] = i;
            for (j = at; j > 0 && t->row[t->top[j]] > t->row[t->top[j - 1]]; j--) {
                int s = t->top[j]; t->top[j] = t->top[j - 1]; t->top[j - 1] = s;
            }
        }
    }
    memset(t->in, 0, sizeof(t->in));
    for (i = 0; i < NV; i++) t->sparse[i] = -INFINITY;
    for (i = 0; i < N; i++) { t->sparse[t->top[i]] = t->row[t->top[i]]; t->in[t->top[i]] = 1; }
    t->n_recent = (int)(xr() % 65);
    for (i = 0; i < t->n_recent; i++)
        t->recent[i] = (xr() % 4) ? (uint32_t)t->top[xr() % 8] : xr() % NV;
    *p = pn_sampling_defaults();
    p->temperature = greedy ? 0.0f : 0.3f + 1.2f * xf();
    p->top_k = 1 + (int)(xr() % 64);
    p->presence_penalty = (xr() % 3) ? 2.0f * xf() : 0.0f;
    p->frequency_penalty = (xr() % 3) ? 1.0f * xf() : 0.0f;
    if (xr() % 16 == 0) p->presence_penalty = -0.5f;
    p->n_logit_bias = (int)(xr() % 4) + (greedy ? 0 : 1);
    for (i = 0; i < p->n_logit_bias; i++) {
        t->bias[i].token_id = (xr() % 2) ? (uint32_t)t->top[xr() % N] : xr() % NV;
        t->bias[i].bias = (xr() % 3) ? 12.0f * xf() - 6.0f : 20.0f * xf();
    }
    p->logit_bias = t->bias;
}

/* The dense row after the client surgery (and, sampled, the repetition penalty). */
static void dense_adjusted(trial_row *t, const pn_sampling_params *p)
{
    memcpy(t->adj, t->row, sizeof(t->adj));
    pn_apply_logit_bias(t->adj, NV, p->logit_bias, p->n_logit_bias);
    pn_apply_presence_frequency_penalties(t->adj, NV, p->presence_penalty, p->frequency_penalty,
                                          t->recent, t->n_recent);
    if (p->temperature > 0.0f)
        for (int i = 0; i < t->n_recent; i++)
            t->adj[t->recent[i]] = t->adj[t->recent[i]] > 0.0f
                                       ? t->adj[t->recent[i]] / p->repeat_penalty
                                       : t->adj[t->recent[i]] * p->repeat_penalty;
}

static int truth_greedy(trial_row *t, const pn_sampling_params *p)
{
    int i, dense = 0, best = -1;

    dense_adjusted(t, p);
    for (i = 1; i < NV; i++) if (t->adj[i] > t->adj[dense]) dense = i;
    for (i = 0; i < NV; i++)
        if (t->in[i] && (best < 0 || t->adj[i] > t->adj[best])) best = i;
    return best == dense;
}

static int truth_sampled(trial_row *t, const pn_sampling_params *p)
{
    static float tmp[NV];
    int i, k;
    float kth;

    dense_adjusted(t, p);
    memcpy(tmp, t->adj, sizeof(tmp));
    for (k = 0; k < p->top_k; k++) {
        int best = k;
        for (i = k + 1; i < NV; i++) if (tmp[i] > tmp[best]) best = i;
        float s = tmp[k]; tmp[k] = tmp[best]; tmp[best] = s;
    }
    kth = tmp[p->top_k - 1];
    for (i = 0; i < NV; i++)
        if (!t->in[i] && t->adj[i] >= kth) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    static trial_row t;
    int part, failures = 0, bad = 0;

    for (part = 0; part < 2; part++) {
        const int greedy = part == 0;
        int trial, said_exact = 0, said_refill = 0, false_exact = 0;

        for (trial = 0; trial < TRIALS; trial++) {
            pn_sampling_params p, q;
            make_trial(&t, &p, greedy);
            q = p;
            if (selftest) q.n_logit_bias = 0;   /* blind to the bias */
            q.dense_row = NULL;
            const int exact = pn_sampler_topn_row_verdict(t.sparse, NV, &q, t.recent, t.n_recent,
                                                          NULL) == PN_TOPN_EXACT;
            const int truth = greedy ? truth_greedy(&t, &p) : truth_sampled(&t, &p);

            if (exact) said_exact++; else said_refill++;
            if (exact && !truth) false_exact++;
        }
        printf("topn_surgery %s: %d trials, exact %d / refill %d, false-exact %d\n",
               greedy ? "greedy " : "sampled", TRIALS, said_exact, said_refill, false_exact);
        failures += false_exact;
        if (!said_exact || !said_refill) bad = 1;
    }
    if (selftest) {
        printf(failures ? "SELFTEST PASSED: a verdict blind to the bias is caught (%d)\n"
                        : "SELFTEST FAILED: the gate cannot see a blind verdict\n", failures);
        return failures ? 0 : 1;
    }
    if (failures || bad) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
