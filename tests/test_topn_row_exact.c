/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_topn_row_exact.c
 * @brief Gate for pn_sampler_topn_row_exact: when it says a top-N row decides a sampled draw,
 *        the dense row's penalised top-k set really lies inside the N.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Random rows (vocab 4096, a heavy head), random repeat windows that hit the head hard,
 * random penalties and top_k 1..64. For each: the top-64 row the card would hand over, the
 * check's verdict, and the brute-force truth (the dense row penalised exactly as the sampler
 * does, its k-th largest value, and whether any token outside the 64 reaches it). A "true"
 * verdict over a false truth is a FAILURE. Both verdicts must occur (the check is not
 * constant). `--selftest`: a check that ignores the penalties must be caught.
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

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }
static float xf(void) { return (float)(xr() & 0xFFFFFF) / 16777216.0f; }

/* Brute force: does the dense row's penalised top-k set lie inside `inside`? */
static int truth_inside(const float *row, const int *inside, const pn_sampling_params *p,
                        const uint32_t *recent, int n_recent)
{
    static float adj[NV];
    static int in[NV];
    int i, k, above_out = 0;
    float kth;

    memcpy(adj, row, sizeof(adj));
    pn_apply_presence_frequency_penalties(adj, NV, p->presence_penalty, p->frequency_penalty,
                                          recent, n_recent);
    for (i = 0; i < n_recent; i++)
        adj[recent[i]] = adj[recent[i]] > 0.0f ? adj[recent[i]] / p->repeat_penalty
                                               : adj[recent[i]] * p->repeat_penalty;
    memset(in, 0, sizeof(in));
    for (i = 0; i < N; i++) in[inside[i]] = 1;
    /* k-th largest over the whole row */
    {
        static float tmp[NV];
        memcpy(tmp, adj, sizeof(tmp));
        for (k = 0; k < p->top_k; k++) {
            int best = k;
            for (i = k + 1; i < NV; i++) if (tmp[i] > tmp[best]) best = i;
            float t = tmp[k]; tmp[k] = tmp[best]; tmp[best] = t;
        }
        kth = tmp[p->top_k - 1];
    }
    for (i = 0; i < NV; i++)
        if (!in[i] && adj[i] >= kth) above_out = 1;
    return !above_out;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    static float row[NV], sparse[NV];
    int top[N];
    uint32_t recent[64];
    int trial, failures = 0, said_true = 0, said_false = 0;

    for (trial = 0; trial < 3000; trial++) {
        pn_sampling_params p = pn_sampling_defaults();
        int i, j, n = 0, n_recent;

        for (i = 0; i < NV; i++) row[i] = 4.0f * xf() - 2.0f;
        for (i = 0; i < 100; i++) row[xr() % NV] = 6.0f + 6.0f * xf();
        p.temperature = 0.3f + 1.2f * xf();
        p.top_k = 1 + (int)(xr() % 64);
        p.repeat_penalty = 1.0f + 0.8f * xf();
        p.presence_penalty = (xr() & 1) ? 0.0f : 2.0f * xf();
        p.frequency_penalty = (xr() & 1) ? 0.0f : 1.0f * xf();
        /* top-64, kernel comparator */
        for (i = 0; i < NV; i++) {
            if (n < N || row[i] > row[top[n - 1]]) {
                int at = n < N ? n++ : N - 1;
                top[at] = i;
                for (j = at; j > 0 && row[top[j]] > row[top[j - 1]]; j--) {
                    int t = top[j]; top[j] = top[j - 1]; top[j - 1] = t;
                }
            }
        }
        for (i = 0; i < NV; i++) sparse[i] = -INFINITY;
        for (i = 0; i < N; i++) sparse[top[i]] = row[top[i]];
        n_recent = (int)(xr() % 65);
        for (i = 0; i < n_recent; i++)
            recent[i] = (xr() % 4) ? (uint32_t)top[xr() % N] : xr() % NV;
        {
            pn_sampling_params q = p;
            if (selftest) { q.repeat_penalty = 1.0f; q.presence_penalty = 0.0f; q.frequency_penalty = 0.0f; }
            const int verdict = pn_sampler_topn_row_exact(sparse, NV, &q, recent, n_recent);
            const int truth = truth_inside(row, top, &p, recent, n_recent);

            if (verdict) said_true++; else said_false++;
            if (verdict && !truth) failures++;
        }
    }
    printf("topn_row_exact: 3000 trials, verdict true %d / false %d, false-exact %d\n",
           said_true, said_false, failures);
    if (selftest) {
        printf(failures ? "SELFTEST PASSED: a check blind to penalties is caught (%d)\n"
                        : "SELFTEST FAILED: the gate cannot see a blind check\n", failures);
        return failures ? 0 : 1;
    }
    if (failures || !said_true || !said_false) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
