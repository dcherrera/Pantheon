/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_spec_sampling_rule.c
 * @brief Gate for true speculative sampling (spec_sampling_rule.h, spec_sampling_chooser.h):
 *        the token a verify row emits is distributed exactly as the plain sampler's.
 *
 * @status Complete. CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Random top-64 target rows (vocab 1024, a heavy head, a repeat window that hits it) and
 * draft rows that share ~3/4 of the target's top-64 (the rest outside p: always rejected)
 * at five settings: Ornith's 0.6 / 20 / 0.95, the CLI's 0.8 / 40 / 0.9, 1.0 / 20 / 1.0,
 * and with top-k off 0.7 / - / 0.95 and 1.0 / - / 1.0. Every row: M rounds of the PRODUCTION
 * path — pn_spec_sample_draft (the draft drawn from q) then pn_spec_sample_choose (the verify
 * row) — on one seeded stream, against (a) p, the plain sampler's final distribution
 * (chi-square), (b) M draws of the plain sampler itself, pn_sample_next_token (two-sample
 * chi-square), (c) the analytic emitted-vs-p TVD. Top-k off has no exact top-N row, so the
 * chooser would fall back to the plain draw: there the rule is driven directly.
 * Also: point-mass drafts, same seed -> same tokens, and the rule's acceptance is above the
 * shared-randomness rule's p(argmax q) on average.
 * `--selftest`: the reds accept-always and residual = p must FAIL the gates.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"
#include "../src/core/spec_sampling_chooser.h"

#define NV 1024
#define N 64
#define ROWS 24
#define M 4000

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }
static float xf(void) { return (float)(xr() & 0xFFFFFF) / 16777216.0f; }
static float xn(void) { float u = xf() + 1e-7f, v = xf(); return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v); }

typedef struct { float t; int k; float p; const char *name; } setting;
static const setting SETTINGS[] = {
    { 0.6f, 20, 0.95f, "0.6/20/0.95 (Ornith)" },
    { 0.8f, 40, 0.90f, "0.8/40/0.9 (CLI)" },
    { 1.0f, 20, 1.00f, "1.0/20/1.0" },
    { 0.7f, 0, 0.95f, "0.7/-/0.95 (no top-k)" },
    { 1.0f, 0, 1.00f, "1.0/-/1.0 (no top-k)" },
};

typedef struct {
    double chi2, df, chi2_plain, df_plain, max_tvd, sum_acc, sum_shared;
    unsigned long outside;
} tally;

/* A target top-64 row: a heavy head of `N` ids, -inf elsewhere. */
static void make_target(int *ids, float *vals)
{
    int i, j;

    for (i = 0; i < N; i++) {
        int id;
        do { id = (int)(xr() % NV); for (j = 0; j < i && ids[j] != id; j++); } while (j < i);
        ids[i] = id;
        vals[i] = 14.0f - 2.2f * powf((float)i, 0.55f) + 0.6f * xn();
    }
}

/* The draft head's row: the target's values plus noise on ~3/4 of its ids, new ids for the
 * rest (q mass outside p's support). */
static void make_draft(const int *t_ids, const float *t_vals, uint32_t *ids, float *vals)
{
    int i, j;

    for (i = 0; i < N; i++) {
        if (xr() % 4) {
            ids[i] = (uint32_t)t_ids[i];
            vals[i] = t_vals[i] + 0.9f * xn();
        } else {
            uint32_t id;
            do { id = xr() % NV; for (j = 0; j < N && t_ids[j] != (int)id; j++); } while (j < N);
            ids[i] = id;
            vals[i] = t_vals[i] - 1.0f + 0.9f * xn();
        }
    }
}

static void row_of(float *row, const uint32_t *ids, const float *vals)
{
    const uint32_t neg_inf_bits = 0xff800000u;
    float neg_inf;
    int i;
    memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));
    for (i = 0; i < NV; i++) row[i] = neg_inf;
    for (i = 0; i < N; i++) row[ids[i]] = vals[i];
}

static void chi2_bins(tally *t, const unsigned long *hist, const unsigned long *plain,
                      const float *probs, int S)
{
    double se = 0.0, so = 0.0, sp = 0.0;
    int i, b = 0, bp = 0;

    for (i = 0; i <= S; i++) {
        const double e = i < S ? (double)M * probs[i] : 0.0;
        if (e < 5.0) { se += e; so += hist[i]; sp += plain[i]; continue; }
        t->chi2 += (hist[i] - e) * (hist[i] - e) / e;
        t->chi2_plain += ((double)hist[i] - plain[i]) * ((double)hist[i] - plain[i]) /
                         (double)(hist[i] + plain[i]);
        b++; bp++;
    }
    if (se >= 1.0) { t->chi2 += (so - se) * (so - se) / se; b++; }
    if (so + sp >= 10.0) { t->chi2_plain += (so - sp) * (so - sp) / (so + sp); bp++; }
    t->df += b > 1 ? b - 1 : 0;
    t->df_plain += bp > 1 ? bp - 1 : 0;
}

static float uniform_of(void *params) { return pn_sampler_uniform((pn_sampling_params *)params); }

/* One setting: ROWS rows x M rounds. Returns the tally. */
static tally run_setting(const setting *st, uint64_t seed)
{
    static float row[NV], work[NV];
    tally t = { 0 };
    int r, i;

    for (r = 0; r < ROWS; r++) {
        pn_sampling_params p = pn_sampling_defaults();
        pn_spec_sample_ctx c = { .params = &p, .n_vocab = NV, .max_recent = 64 };
        uint32_t recent[64], drow_ids[N], trow_u[N];
        float trow_v[N], drow_v[N];
        int t_ids[N], ids[N];
        float probs[N];
        unsigned long hist[N + 1] = { 0 }, plain[N + 1] = { 0 };
        int n_recent, S, m;
        const bool exact_path = st->k > 0;

        p.temperature = st->t; p.top_k = st->k; p.top_p = st->p;
        p.rng_seeded = 1; p.rng_state = seed + (uint64_t)r;
        make_target(t_ids, trow_v);
        for (i = 0; i < N; i++) trow_u[i] = (uint32_t)t_ids[i];
        make_draft(t_ids, trow_v, drow_ids, drow_v);
        n_recent = (int)(xr() % 40);
        for (i = 0; i < n_recent; i++) recent[i] = (uint32_t)t_ids[xr() % 12];
        /* p: the plain sampler's final distribution */
        row_of(work, trow_u, trow_v);
        S = pn_sampler_row_final_probs(work, NV, &p, recent, n_recent, false, ids, probs, N);
        if (S < 1) { printf("  row %d: no p\n", r); t.outside++; continue; }
        c.row = row; c.recent = malloc(64 * sizeof(uint32_t));
        if (!c.recent || pn_spec_sample_rule_arm(&c) != 0) { printf("alloc\n"); exit(2); }
        for (m = 0; m < M; m++) {
            uint32_t y;
            int at = S;

            memcpy(c.recent, recent, (size_t)n_recent * sizeof(uint32_t));
            c.n_recent = n_recent; c.row_index = 0;
            pn_spec_sample_begin_drafts(&c);
            (void)pn_spec_sample_draft(&c, 0, drow_ids, drow_v, N);
            if (exact_path) {
                y = pn_spec_sample_choose(&c, trow_u, trow_v, N);
            } else {
                row_of(work, trow_u, trow_v);
                S = pn_sampler_row_final_probs(work, NV, &p, recent, n_recent, false, ids, probs, N);
                y = pn_spec_rule_verify(ids, probs, S, &c.q[0], uniform_of, &p, NULL);
            }
            for (i = 0; i < S; i++) if ((uint32_t)ids[i] == y) { at = i; break; }
            if (at == S || !(probs[at] > 0.0f)) t.outside++;
            hist[at]++;
            /* the plain sampler itself on the same row */
            row_of(work, trow_u, trow_v);
            y = pn_sample_next_token(work, NV, &p, recent, n_recent);
            for (at = S, i = 0; i < S; i++) if ((uint32_t)ids[i] == y) { at = i; break; }
            plain[at]++;
        }
        if (exact_path && c.stats.rule_rows != (unsigned long)M)
            printf("  row %d: only %lu of %d rounds took the rule (%lu fell back)\n", r,
                   c.stats.rule_rows, M, c.stats.fallback);
        chi2_bins(&t, hist, plain, probs, S);
        {
            const double tvd = pn_spec_rule_emitted_tvd(ids, probs, S, &c.q[0]);
            double best = 0.0, pq = 0.0;
            if (tvd > t.max_tvd) t.max_tvd = tvd;
            t.sum_acc += pn_spec_rule_acceptance(ids, probs, S, &c.q[0]);
            /* the shared-randomness rule accepts the argmax draft with p(argmax q) */
            for (i = 0; i < c.q[0].n; i++)
                if (c.q[0].probs[i] > best) { best = c.q[0].probs[i]; pq = 0.0;
                    for (int j = 0; j < S; j++) if ((uint32_t)ids[j] == c.q[0].ids[i]) pq = probs[j]; }
            t.sum_shared += pq;
        }
        pn_spec_sample_rule_free(&c);
        free(c.recent);
    }
    return t;
}

static double z_of(double chi2, double df) { return df > 0 ? (chi2 - df) / sqrt(2.0 * df) : 0.0; }

/* Gates on one tally: |z| < 5 both ways, TVD < 1e-5, nothing emitted outside p. */
static int tally_pass(const tally *t)
{
    return fabs(z_of(t->chi2, t->df)) < 5.0 && fabs(z_of(t->chi2_plain, t->df_plain)) < 5.0 &&
           t->max_tvd < 1e-5 && t->outside == 0;
}

static void tally_print(const char *name, const tally *t)
{
    printf("  %-24s vs p chi2 %8.1f / %4.0f df (z %6.2f) | vs plain draws chi2 %8.1f / %4.0f df "
           "(z %6.2f) | TVD max %.2e | outside %lu | accept %.3f (shared-rand %.3f)\n",
           name, t->chi2, t->df, z_of(t->chi2, t->df), t->chi2_plain, t->df_plain,
           z_of(t->chi2_plain, t->df_plain), t->max_tvd, t->outside, t->sum_acc / ROWS,
           t->sum_shared / ROWS);
}

/* Same seed, same tokens: the production pair run twice on one row. */
static int determinism_ok(void)
{
    static float row[NV];
    uint32_t out[2][64], d_ids[N], t_u[N];
    float d_v[N], t_v[N];
    int t_ids[N], pass, i;

    make_target(t_ids, t_v);
    for (i = 0; i < N; i++) t_u[i] = (uint32_t)t_ids[i];
    make_draft(t_ids, t_v, d_ids, d_v);
    for (pass = 0; pass < 2; pass++) {
        pn_sampling_params p = pn_sampling_defaults();
        pn_spec_sample_ctx c = { .params = &p, .n_vocab = NV, .max_recent = 64, .row = row };
        uint32_t rec[64];

        p.temperature = 0.6f; p.top_k = 20; p.top_p = 0.95f; p.rng_seeded = 1; p.rng_state = 1234;
        c.recent = rec;
        pn_spec_sample_rule_arm(&c);
        for (i = 0; i < 64; i++) {
            c.n_recent = 0; c.row_index = 0;
            pn_spec_sample_begin_drafts(&c);
            (void)pn_spec_sample_draft(&c, 0, d_ids, d_v, N);
            out[pass][i] = pn_spec_sample_choose(&c, t_u, t_v, N);
        }
        pn_spec_sample_rule_free(&c);
    }
    return memcmp(out[0], out[1], sizeof(out[0])) == 0;
}

/* Point-mass drafts (prompt lookup): accept with p(x), else p without x. */
static int point_mass_ok(void)
{
    const int ids[4] = { 3, 7, 9, 11 };
    const float probs[4] = { 0.5f, 0.3f, 0.2f, 0.0f };
    pn_spec_proposal q = { .token = 7, .n = 0 };
    pn_spec_proposal q_out = { .token = 11, .n = 0 };   /* p(11) = 0: never accepted */

    return pn_spec_rule_emitted_tvd(ids, probs, 4, &q) < 1e-7 &&
           pn_spec_rule_emitted_tvd(ids, probs, 4, &q_out) < 1e-7 &&
           fabs(pn_spec_rule_acceptance(ids, probs, 4, &q) - 0.3) < 1e-7;
}

/* The draft's pair path equals the row path (same penalties, same tail) on random rows, to
 * float rounding (-ffast-math may associate the penalty steps differently). */
static int pairs_match_rows(void)
{
    static float row[NV];
    int trial, i, bad = 0;

    for (trial = 0; trial < 500; trial++) {
        pn_sampling_params p = pn_sampling_defaults();
        uint32_t ids_u[N], recent[64];
        float vals[N], pr[N], pp[N];
        int t_ids[N], a[N], b[N], n_recent = (int)(xr() % 64), Sa, Sb;

        p.temperature = 0.3f + xf(); p.top_k = 1 + (int)(xr() % 64); p.top_p = 0.5f + 0.5f * xf();
        p.repeat_penalty = 1.0f + 0.5f * xf();
        p.presence_penalty = (xr() & 1) ? 0.5f * xf() : 0.0f;
        p.frequency_penalty = (xr() & 1) ? 0.5f * xf() : 0.0f;
        make_target(t_ids, vals);
        for (i = 0; i < N; i++) ids_u[i] = (uint32_t)t_ids[i];
        for (i = 0; i < n_recent; i++) recent[i] = (uint32_t)t_ids[xr() % 16];
        row_of(row, ids_u, vals);
        Sa = pn_sampler_row_final_probs(row, NV, &p, recent, n_recent, false, a, pr, N);
        Sb = pn_sampler_pairs_final_probs(ids_u, vals, N, NV, &p, recent, n_recent, b, pp, N);
        int far = 0;
        for (i = 0; i < Sa && Sa == Sb; i++) far += fabsf(pr[i] - pp[i]) > 1e-5f;
        /* q needs no bit identity (any q is exact): the same distribution to rounding. */
        if (Sa != Sb || memcmp(a, b, (size_t)Sa * sizeof(int)) || far) {
            if (!bad) {
                printf("  first mismatch: S %d / %d", Sa, Sb);
                for (i = 0; i < Sa && i < Sb; i++)
                    if (a[i] != b[i] || fabsf(pr[i] - pp[i]) > 1e-5f) {
                        printf(", at %d: id %d/%d p %.9g/%.9g", i, a[i], b[i], pr[i], pp[i]);
                        break;
                    }
                printf("\n");
            }
            bad++;
        }
    }
    return bad == 0;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    const size_t n_set = sizeof(SETTINGS) / sizeof(SETTINGS[0]);
    size_t s;
    int fails = 0;

    if (selftest) {
        const pn_spec_rule_red reds[2] = { PN_SPEC_RULE_RED_ACCEPT_ALWAYS, PN_SPEC_RULE_RED_RESIDUAL_P };
        const char *names[2] = { "accept_always", "residual_p" };
        int caught = 0, k;

        for (k = 0; k < 2; k++) {
            tally t;
            pn_spec_rule_set_red(reds[k]);
            t = run_setting(&SETTINGS[0], 99);
            printf("RED %s:\n", names[k]);
            tally_print(SETTINGS[0].name, &t);
            if (!tally_pass(&t)) caught++;
            else printf("  RED %s NOT CAUGHT\n", names[k]);
        }
        pn_spec_rule_set_red(PN_SPEC_RULE_RED_NONE);
        printf(caught == 2 ? "SELFTEST PASSED: both reds fail the gates\n"
                           : "SELFTEST FAILED: a red passed the gates\n");
        return caught == 2 ? 0 : 1;
    }
    pn_spec_rule_set_red(PN_SPEC_RULE_RED_NONE);
    printf("spec_sampling_rule: %d rows x %d rounds per setting\n", ROWS, M);
    for (s = 0; s < n_set; s++) {
        const tally t = run_setting(&SETTINGS[s], 1000 + s * 77);
        tally_print(SETTINGS[s].name, &t);
        if (!tally_pass(&t)) { printf("  FAIL\n"); fails++; }
    }
    if (!determinism_ok()) { printf("determinism: same seed gave different tokens  FAIL\n"); fails++; }
    else printf("determinism: same seed, same 64 tokens\n");
    if (!pairs_match_rows()) { printf("pair path != row path  FAIL\n"); fails++; }
    else printf("draft pair path == row path (to 1e-5), 500 random rows\n");
    if (!point_mass_ok()) { printf("point-mass drafts  FAIL\n"); fails++; }
    else printf("point-mass drafts: emitted == p, acceptance = p(x)\n");
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
