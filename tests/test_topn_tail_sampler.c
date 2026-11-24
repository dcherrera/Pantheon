/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_topn_tail_sampler.c
 * @brief Gate for the top-N tail planner (sampler_topn_tail.h): a sampled draw whose top-k is
 *        off or past the N, taken from a top-64 row through the production sampler, is
 *        distributed exactly as the dense draw.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Random bf16-rounded rows (vocab 2048, flat to peaked), top_k 0 / 100, top_p 1 / 0.9,
 * temperature 0.6 / 1.0 / 1.4, with and without penalties over a repeat window that hits the
 * head. For each case:
 *   - the exact distribution (double, the sampler's semantics: penalties, full softmax, top-k
 *     with ties, renormalised, top-p, renormalised);
 *   - PLAN: the planner's implied distribution (NUCLEUS: the survivors at the cut; MIXTURE:
 *     a h + max(p - a h, 0)) against the exact one, TVD <= 1e-5;
 *   - DRAWS through pn_sample_next_token on the dense row (control), on the top-64 row with a
 *     dense refill, and through pn_spec_sample_choose as a verify row past 0 (deferring to a
 *     row-0 draw of the same row): each empirical TVD within 2x its expected sampling noise
 *     + 1e-3;
 *   - the spec route's token SEQUENCE equals the plain route's for the same seed.
 * Every plan kind and the residual branch must occur. The suite runs twice: with the trivial
 * tail bound and with the device's tail counts (emulated over the dense row); the reds run
 * with the counts. `--selftest`: the reds
 * `PN_TT_RED=tail_nobound` (a planner that ignores the tail) and `tail_nocarry` (a deferred
 * row drops its draw) must each fail.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/core/sampler_topn_tail.h"
#include "../src/core/sampling_modifiers.h"

#define NV 2048
#define N 64
#define DRAWS 60000
#define SEQ 2000

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }
static double xf(void) { return ((double)(xr() & 0xFFFFFF) + 0.5) / 16777216.0; }

static float bf16_round(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* One case: the dense row, its top-N row, params and the repeat window. */
typedef struct {
    float dense[NV], sparse[NV];
    uint32_t idx[N];
    float val[N];
    uint32_t recent[64];
    int n_recent;
    pn_sampling_params p;
} tcase;

static const float *g_refill_src;
static int g_use_counts;   /* the device's tail counts, emulated over the dense row */

static int counts_from_dense(void *ctx, int row, const uint32_t *keys, int bins,
                             uint32_t *counts)
{
    tnt_tail_bins tb;

    (void)ctx; (void)row;
    tb.bins = bins;
    tb.xmin_index = -1;
    memcpy(tb.keys, keys, (size_t)bins * sizeof(keys[0]));
    tnt_tail_bins_count(&tb, g_refill_src, NV, counts);
    return 0;
}
static int refill_copy(void *ctx, float *logits, int n)
{
    (void)ctx;
    memcpy(logits, g_refill_src, (size_t)n * sizeof(float));
    return 0;
}

static void make_case(tcase *c, double scale, float temp, int top_k, float top_p, int penalties)
{
    int i, j, n = 0;
    int top[N];

    for (i = 0; i < NV; i++) {
        const double z = sqrt(-2.0 * log(xf())) * cos(6.283185307179586 * xf());
        c->dense[i] = bf16_round((float)(scale * z));
    }
    for (i = 0; i < 3; i++) c->dense[xr() % NV] += bf16_round((float)(scale * 1.5));
    for (i = 0; i < NV; i++) {   /* top-N: value desc, index asc (the kernel's comparator) */
        if (n < N || c->dense[i] > c->dense[top[n - 1]]) {
            int at = n < N ? n++ : N - 1;
            top[at] = i;
            for (j = at; j > 0 && c->dense[top[j]] > c->dense[top[j - 1]]; j--) {
                int t = top[j]; top[j] = top[j - 1]; top[j - 1] = t;
            }
        }
    }
    for (i = 0; i < NV; i++) c->sparse[i] = -INFINITY;
    for (i = 0; i < N; i++) {
        c->sparse[top[i]] = c->dense[top[i]];
        c->idx[i] = (uint32_t)top[i];
        c->val[i] = c->dense[top[i]];
    }
    c->n_recent = penalties ? 64 : 0;
    for (i = 0; i < c->n_recent; i++)   /* half the head, half the tail */
        c->recent[i] = (i & 1) ? (uint32_t)top[xr() % 8] : xr() % NV;
    c->p = pn_sampling_defaults();
    c->p.temperature = temp;
    c->p.top_k = top_k;
    c->p.top_p = top_p;
    c->p.repeat_penalty = penalties ? 1.1f : 1.0f;
    c->p.presence_penalty = penalties ? 0.3f : 0.0f;
    c->p.frequency_penalty = penalties ? 0.2f : 0.0f;
    c->p.rng_seeded = 1;
}

/* The exact distribution (double) of the sampler's semantics over the dense row. */
static void exact_probs(const tcase *c, double *p)
{
    static float adj[NV];
    static double work[NV];
    double m, z = 0.0, thr, cum;
    int i, kk = c->p.top_k <= 0 || c->p.top_k > NV ? NV : c->p.top_k, nc = 0;

    memcpy(adj, c->dense, sizeof(adj));
    pn_apply_presence_frequency_penalties(adj, NV, c->p.presence_penalty, c->p.frequency_penalty,
                                          c->recent, c->n_recent);
    for (i = 0; i < c->n_recent; i++)
        adj[c->recent[i]] = adj[c->recent[i]] > 0.0f ? adj[c->recent[i]] / c->p.repeat_penalty
                                                     : adj[c->recent[i]] * c->p.repeat_penalty;
    m = adj[0];
    for (i = 1; i < NV; i++) if (adj[i] > m) m = adj[i];
    for (i = 0; i < NV; i++) { p[i] = exp(((double)adj[i] - m) / c->p.temperature); z += p[i]; }
    for (i = 0; i < NV; i++) p[i] /= z;
    if (kk < NV) {
        int a, b;
        memcpy(work, p, sizeof(work));
        for (a = 0; a < kk; a++) {   /* partial sort: kk-th largest */
            int best = a;
            for (b = a + 1; b < NV; b++) if (work[b] > work[best]) best = b;
            double t = work[a]; work[a] = work[best]; work[best] = t;
        }
        thr = work[kk - 1];
        z = 0.0;
        for (i = 0; i < NV; i++) { if (p[i] < thr) p[i] = 0.0; z += p[i]; }
        for (i = 0; i < NV; i++) p[i] /= z;
    }
    if (c->p.top_p < 1.0f) {
        for (i = 0; i < NV; i++) if (p[i] > 0.0) work[nc++] = p[i];
        for (int a = 0; a < nc; a++) {
            int best = a;
            for (int b = a + 1; b < nc; b++) if (work[b] > work[best]) best = b;
            double t = work[a]; work[a] = work[best]; work[best] = t;
        }
        cum = 0.0; thr = 0.0;
        for (i = 0; i < nc && cum < c->p.top_p; i++) { cum += work[i]; thr = work[i]; }
        z = 0.0;
        for (i = 0; i < NV; i++) { if (p[i] < thr) p[i] = 0.0; z += p[i]; }
        for (i = 0; i < NV; i++) p[i] /= z;
    }
}

static double tvd_counts(const double *p, const unsigned *cnt, int draws, double *noise)
{
    double t = 0.0, e = 0.0;

    for (int i = 0; i < NV; i++) {
        t += fabs(p[i] - (double)cnt[i] / draws);
        e += sqrt(p[i] * (1.0 - p[i]) / draws);
    }
    *noise = 0.5 * e * 0.7978845608;   /* E|N(0, s)| = s sqrt(2 / pi) */
    return 0.5 * t;
}

/* The planner's implied distribution over the case, against the exact one. */
static double plan_tvd(const tcase *c, const double *pe, tnt_kind *kind)
{
    static float pen[NV];
    float vals[N];
    int ids[N], S = 0, i;
    double q[NV], t = 0.0;
    tnt_plan plan;
    tnt_sampling ts = { c->p.temperature, c->p.top_p, c->p.top_k, NV, 1.0, -1.0, 0 };
    float bound = c->val[N - 1];

    memcpy(pen, c->sparse, sizeof(pen));
    pn_apply_presence_frequency_penalties(pen, NV, c->p.presence_penalty, c->p.frequency_penalty,
                                          c->recent, c->n_recent);
    for (i = 0; i < c->n_recent; i++)
        pen[c->recent[i]] = pen[c->recent[i]] > 0.0f ? pen[c->recent[i]] / c->p.repeat_penalty
                                                     : pen[c->recent[i]] * c->p.repeat_penalty;
    for (i = 0; i < NV; i++) if (c->sparse[i] > -1e30f) { ids[S] = i; vals[S] = pen[i]; S++; }
    if (g_use_counts) {   /* production's effective bound (sampler_tail_counts) */
        tnt_tail_bins tb;
        uint32_t counts[TNT_TAIL_BINS + 1];
        float raw[N], x_min = vals[0];
        double m = vals[0], B, trivial;
        int in_topk;

        for (i = 0; i < S; i++) {
            raw[i] = c->sparse[ids[i]];
            if (vals[i] > m) m = vals[i];
            if (vals[i] < x_min) x_min = vals[i];
        }
        tnt_tail_bins_make(bound, x_min, c->p.temperature, NV, &tb);
        tnt_tail_bins_count(&tb, c->dense, NV, counts);
        if (tnt_tail_from_counts(&tb, counts, raw, S, m, 1.0 / c->p.temperature, x_min,
                                 c->p.top_k, &B, &in_topk) == 0) {
            trivial = (double)(NV - S) * exp(((double)bound - m) / c->p.temperature);
            ts.tail_mass = B < trivial ? B : trivial;
            ts.survivors_in_topk = in_topk;
        }
    }
    *kind = tnt_plan_draw(vals, S, bound, &ts, &plan);
    if (*kind == TNT_DENSE) return 0.0;
    for (i = 0; i < NV; i++) q[i] = (*kind == TNT_MIXTURE) ? pe[i] : 0.0;
    if (*kind == TNT_NUCLEUS) {
        double w[N], th, z = 0.0, srt[N];
        double m = vals[0];
        for (i = 0; i < S; i++) if (vals[i] > m) m = vals[i];
        for (i = 0; i < S; i++) srt[i] = w[i] = exp(((double)vals[i] - m) / c->p.temperature);
        for (int a = 0; a < plan.nucleus_k; a++) {
            int best = a;
            for (int b = a + 1; b < S; b++) if (srt[b] > srt[best]) best = b;
            double tt = srt[a]; srt[a] = srt[best]; srt[best] = tt;
        }
        th = srt[plan.nucleus_k - 1];
        for (i = 0; i < S; i++) if (w[i] >= th) z += w[i];
        for (i = 0; i < S; i++) q[ids[i]] = w[i] >= th ? w[i] / z : 0.0;
    } else {
        for (i = 0; i < S; i++) {
            const double ah = plan.head_mass * plan.head_prob[i], r = pe[ids[i]] - ah;
            q[ids[i]] = ah + (r > 0.0 ? r : 0.0);
        }
    }
    for (i = 0; i < NV; i++) t += fabs(pe[i] - q[i]);
    return 0.5 * t;
}

/* `route`: 0 dense control, 1 the top-N row with a refill, 2 the chooser as a row past 0. */
static uint32_t draw_once(tcase *c, int route, pn_sampling_params *p, float *row)
{
    static uint32_t recent[64];
    pn_spec_sample_ctx ctx;
    uint32_t tok;

    memcpy(recent, c->recent, sizeof(recent));
    if (route == 0) {
        memcpy(row, c->dense, sizeof(float) * NV);
        return pn_sample_next_token(row, NV, p, recent, c->n_recent);
    }
    p->dense_row = refill_copy;
    p->tail_counts = g_use_counts ? counts_from_dense : NULL;
    g_refill_src = c->dense;
    if (route == 1) {
        memcpy(row, c->sparse, sizeof(float) * NV);
        tok = pn_sample_next_token(row, NV, p, recent, c->n_recent);
    } else {
        ctx = (pn_spec_sample_ctx){ p, row, NV, recent, c->n_recent, 64, 1 };
        tok = pn_spec_sample_choose(&ctx, c->idx, c->val, N);
        if (tok == PN_SPEC_CHOOSE_DEFER_TOKEN) {
            memcpy(recent, c->recent, sizeof(recent));
            ctx = (pn_spec_sample_ctx){ p, row, NV, recent, c->n_recent, 64, 0 };
            tok = pn_spec_sample_choose(&ctx, c->idx, c->val, N);
        }
    }
    p->dense_row = NULL;
    p->tail_counts = NULL;
    return tok;
}

/* Draw `DRAWS` tokens along `route`; returns 1 when the empirical TVD is within the noise. */
static int route_ok(tcase *c, const double *pe, int route, uint64_t seed, double *tvd_out)
{
    static unsigned cnt[NV];
    static float row[NV];
    pn_sampling_params p = c->p;
    double noise;

    memset(cnt, 0, sizeof(cnt));
    p.rng_state = seed;
    for (int d = 0; d < DRAWS; d++) {
        const uint32_t t = draw_once(c, route, &p, row);
        if (t < NV) cnt[t]++;
    }
    *tvd_out = tvd_counts(pe, cnt, DRAWS, &noise);
    return *tvd_out <= 2.0 * noise + 1e-3;
}

/* Plain route vs spec route: the same seed must give the same token sequence. */
static int sequences_match(tcase *c, uint64_t seed)
{
    static float row[NV];
    pn_sampling_params a = c->p, b = c->p;

    a.rng_state = b.rng_state = seed;
    for (int d = 0; d < SEQ; d++)
        if (draw_once(c, 1, &a, row) != draw_once(c, 2, &b, row))
            return 0;
    return 1;
}

static int run_suite(int verbose)
{
    static tcase c;
    static double pe[NV];
    const double scales[] = { 1.0, 2.5, 6.0 };
    const float temps[] = { 0.6f, 1.0f, 1.4f };
    const int topks[] = { 0, 100 };
    const float topps[] = { 1.0f, 0.9f };
    int fails = 0, kinds[3] = { 0, 0, 0 };
    unsigned long from_n0, res0, dense0, from_n, res, dense;

    g_rng = 0x2545F4914F6CDD1Dull;
    pn_sampler_tail_counts(&from_n0, &res0, &dense0);
    for (int si = 0; si < 3; si++)
    for (int ti = 0; ti < 3; ti++)
    for (int ki = 0; ki < 2; ki++)
    for (int pi = 0; pi < 2; pi++) {
        const int pen = (si + ti + ki + pi) & 1;
        double t0, t1, t2, tp;
        tnt_kind kind;
        int ok;

        make_case(&c, scales[si], temps[ti], topks[ki], topps[pi], pen);
        exact_probs(&c, pe);
        tp = plan_tvd(&c, pe, &kind);
        kinds[kind]++;
        ok = tp <= 1e-5;
        ok &= route_ok(&c, pe, 0, 0x1234u + (uint64_t)si, &t0);
        ok &= route_ok(&c, pe, 1, 0x5678u + (uint64_t)ti, &t1);
        ok &= route_ok(&c, pe, 2, 0x9ABCu + (uint64_t)ki, &t2);
        ok &= sequences_match(&c, 0xDEF0u + (uint64_t)pi);
        if (!ok) fails++;
        if (verbose || !ok)
            printf("  %s scale %.1f T %.1f top_k %3d top_p %.1f pen %d plan %d planTVD %.1e "
                   "draws TVD dense %.4f topN %.4f spec %.4f\n", ok ? "ok  " : "FAIL",
                   scales[si], temps[ti], topks[ki], topps[pi], pen, (int)kind, tp, t0, t1, t2);
    }
    pn_sampler_tail_counts(&from_n, &res, &dense);
    printf("plans: dense %d nucleus %d mixture %d | draws from N %lu, residual %lu, dense %lu\n",
           kinds[TNT_DENSE], kinds[TNT_NUCLEUS], kinds[TNT_MIXTURE], from_n - from_n0,
           res - res0, dense - dense0);
    if ((!g_use_counts && !kinds[TNT_DENSE]) || !kinds[TNT_NUCLEUS] || !kinds[TNT_MIXTURE] ||
        res == res0) {
        printf("a plan kind or the residual branch never occurred: the suite is not covering\n");
        fails++;
    }
    return fails;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    const int verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    int fails, red;

    printf("-- trivial bound (no device counts)\n");
    fails = run_suite(verbose);
    printf("-- device tail counts\n");
    g_use_counts = 1;
    fails += run_suite(verbose);
    printf("test_topn_tail_sampler: %d failing cases\n", fails);
    if (!selftest)
        return fails ? 1 : 0;
    setenv("PN_TT_RED", "tail_nobound", 1);
    red = run_suite(0);
    printf("selftest red tail_nobound: %d failing cases (must be > 0)\n", red);
    setenv("PN_TT_RED", "tail_nocarry", 1);
    fails += !red;
    red = run_suite(0);
    printf("selftest red tail_nocarry: %d failing cases (must be > 0)\n", red);
    unsetenv("PN_TT_RED");
    fails += !red;
    return fails ? 1 : 0;
}
