/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_sampler_margin_draw.c
 * @brief The margin draw (sampler_margin_draw.h) returns the dense full path's token whenever
 *        it decides, and refuses near every float-sensitive boundary.
 *
 * @status Complete. CPU-only — no device, no model.
 * @issues None
 * @todo None
 *
 * Part A, the margin draw against the dense draw. Reference: pn_sample_next_token with the
 * margin draw OFF (the dense full path), the uniform injected through carried_uniform so both
 * sides draw against the same u. Rows: realistic (a peaked or flat head of ~40 logits over a
 * noisy tail, full 248,320 vocabulary), plus exact ties and near ties at the top-k edge.
 * Uniforms: random, and placed within a few ulps of every cumulative boundary of the plan
 * (where a float difference can flip the token). Every decided draw must match; the decided
 * share must be high (it is the speed).
 *
 * Part B, the sampler with the margin draw ON against OFF, through pn_sample_next_token as the
 * serve calls it (repeat penalty over a recent window, the request's top_k / top_p / T): the
 * same token for the same uniform on every row and every uniform of part A, and a seeded run
 * of draws per row ends on the same tokens and the same RNG state. The margin counter must
 * move (the path is not inert).
 *
 * `--selftest`: PN_MARGIN_RED=1 (every margin 0); exits 0 only if part A mismatches.
 * `--red-wrong`: PN_MARGIN_RED=2 (a wrong decided token); exits 0 only if part B mismatches.
 */
#include "../src/core/engine.h"
#include "../src/core/sampler_margin_draw.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 248320

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;

static double urand(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

static double nrand(void)
{
    double a = urand(), b = urand();
    return sqrt(-2.0 * log(a + 1e-300)) * cos(6.283185307179586 * b);
}

/* kind 0 peaked, 1 flat head, 2 exact tie at the k-th, 3 near tie at the k-th; head[] gets
 * the head's ids, highest first for kinds 2 and 3. */
static void make_row(float *row, int kind, int top_k, uint32_t *head)
{
    for (int i = 0; i < VOCAB; i++) row[i] = (float)(nrand() * 2.0 - 12.0);
    double v = 22.0 + 4.0 * urand();
    const double step = kind == 1 ? 0.15 : 0.4;
    for (int j = 0; j < 48; j++) {
        head[j] = (uint32_t)(urand() * VOCAB);
        row[head[j]] = (float)v;
        v -= step * -log(urand() + 1e-12);
    }
    if (kind == 2 || kind == 3) {
        /* Sort the head by value to find the k-th and k+1-th, then tie them. */
        float vals[48];
        for (int j = 0; j < 48; j++) vals[j] = row[head[j]];
        for (int a = 1; a < 48; a++)
            for (int b = a; b > 0 && vals[b] > vals[b - 1]; b--) {
                float t = vals[b]; vals[b] = vals[b - 1]; vals[b - 1] = t;
                uint32_t h = head[b]; head[b] = head[b - 1]; head[b - 1] = h;
            }
        row[head[top_k]] = kind == 2 ? row[head[top_k - 1]]
                                     : nextafterf(row[head[top_k - 1]], -FLT_MAX);
    }
}

static pn_sampling_params params_for(float t, int k, float p, float rp)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = t;
    sp.top_k = k;
    sp.top_p = p;
    sp.repeat_penalty = rp;
    return sp;
}

/* One draw through the sampler with the margin draw on or off; u < 0 = the request's RNG. */
static uint32_t sampler_token(const float *row, float *scratch, pn_sampling_params *sp, float u,
                              int margin_on, const uint32_t *recent, int n_recent)
{
    uint32_t rec[64];
    if (u >= 0.0f) {
        sp->carried_uniform_set = 1;
        sp->carried_uniform = u;
    }
    if (n_recent > 0)
        memcpy(rec, recent, (size_t)n_recent * sizeof(uint32_t));
    memcpy(scratch, row, (size_t)VOCAB * sizeof(float));
    pn_margin_draw_set_enabled(margin_on);
    uint32_t tok = pn_sample_next_token(scratch, VOCAB, sp, n_recent > 0 ? rec : NULL, n_recent);
    pn_margin_draw_set_enabled(-1);
    return tok;
}

/* Part B on one row: on vs off for every uniform, then a seeded run of 32 draws. */
static long on_off_mismatches(const float *row, float *scratch, const float *us, int nu, float t,
                              int k, float p, const uint32_t *head, long *compared)
{
    uint32_t recent[8];
    long bad = 0;
    for (int i = 0; i < 8; i++) recent[i] = head[i * 3];   /* the penalty hits the head */
    for (int i = 0; i < nu; i++) {
        pn_sampling_params a = params_for(t, k, p, 1.1f), b = a;
        bad += sampler_token(row, scratch, &a, us[i], 1, recent, 8) !=
               sampler_token(row, scratch, &b, us[i], 0, recent, 8);
        (*compared)++;
    }
    pn_sampling_params a = params_for(t, k, p, 1.1f), b = a;
    a.rng_seeded = b.rng_seeded = 1;
    a.rng_state = b.rng_state = 0x243f6a8885a308d3ull ^ (uint64_t)nu;
    for (int i = 0; i < 32; i++) {
        bad += sampler_token(row, scratch, &a, -1.0f, 1, recent, 8) !=
               sampler_token(row, scratch, &b, -1.0f, 0, recent, 8);
        (*compared)++;
    }
    return bad + (a.rng_state != b.rng_state);
}

static uint32_t reference_token(const float *row, float *scratch, float u, float t, int k,
                                float p)
{
    pn_sampling_params sp = params_for(t, k, p, 1.0f);
    return sampler_token(row, scratch, &sp, u, 0, NULL, 0);
}

/* Random uniforms, a few ulps around every boundary of the plan, and just outside the margin
 * on both sides (decided, and still next to a boundary). Returns the count. */
static int make_uniforms(const pn_margin_plan *plan, int plan_ok, float *us, int cap)
{
    static const int offs[] = { -64, -8, -2, -1, 0, 1, 2, 8, 64, 1024 };
    int nu = 0;

    for (int i = 0; i < 64; i++) us[nu++] = (float)urand();
    for (int j = 0; plan_ok && j < plan->n; j++) {
        for (int o = 0; o < 10 && nu < cap; o++) {
            float u = (float)plan->cum[j];
            for (int s = 0; s < abs(offs[o]); s++)
                u = nextafterf(u, offs[o] < 0 ? 0.0f : 1.0f);
            if (u >= 0.0f && u < 1.0f) us[nu++] = u;
        }
        const double lo = plan->cum[j] - 1.5 * PN_MARGIN_BETA;
        const double hi = plan->cum[j] + 1.5 * PN_MARGIN_BETA;
        if (lo > 0.0 && nu < cap) us[nu++] = (float)lo;
        if (hi < 1.0 && nu < cap) us[nu++] = (float)hi;
    }
    return nu;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    const int red_wrong = argc > 1 && strcmp(argv[1], "--red-wrong") == 0;
    const int rows = argc > 2 ? atoi(argv[2]) : 24;
    if (selftest || red_wrong)
        setenv("PN_MARGIN_RED", selftest ? "1" : "2", 1);
    float *row = malloc((size_t)VOCAB * sizeof(float));
    float *scratch = malloc((size_t)VOCAB * sizeof(float));
    long decided = 0, refused = 0, mismatched = 0, plans = 0, plan_refused = 0;
    long on_off_bad = 0, on_off_compared = 0;
    const float temps[] = { 0.6f, 1.0f };
    const int ks[] = { 20, 40 };
    const float tps[] = { 0.95f, 1.0f, 0.8f };

    if (!row || !scratch)
        return 2;
    const unsigned long margin_before = pn_sampler_margin_draws();
    for (int r = 0; r < rows; r++) {
        const float t = temps[r % 2];
        const int k = ks[(r / 2) % 2];
        const float p = tps[(r / 4) % 3];
        uint32_t head[48];
        make_row(row, r % 4, k, head);
        pn_margin_plan plan;
        const int rc = pn_margin_plan_dense(row, VOCAB, t, k, p, &plan);
        plans++;
        plan_refused += rc != PN_MARGIN_OK;
        if (getenv("MARGIN_TEST_VERBOSE"))
            fprintf(stderr, "row %d kind %d T %.1f k %d p %.2f: plan %d\n", r, r % 4, t, k, p, rc);
        float us[64 + PN_MARGIN_CAND * 12];
        const int nu = make_uniforms(&plan, rc == PN_MARGIN_OK, us, (int)(sizeof us / sizeof us[0]));
        for (int i = 0; rc == PN_MARGIN_OK && i < nu; i++) {
            uint32_t tok;
            if (!pn_margin_decide(&plan, us[i], &tok)) {
                refused++;
                continue;
            }
            decided++;
            const uint32_t ref = reference_token(row, scratch, us[i], t, k, p);
            if (ref != tok && ++mismatched <= 5)
                fprintf(stderr, "mismatch row %d (T %.1f k %d p %.2f) u=%.9g: margin %u dense %u\n",
                        r, t, k, p, us[i], tok, ref);
        }
        on_off_bad += on_off_mismatches(row, scratch, us, nu, t, k, p, head, &on_off_compared);
    }
    const unsigned long margin_used = pn_sampler_margin_draws() - margin_before;
    printf("test_sampler_margin_draw%s: plans %ld (refused %ld), draws decided %ld refused %ld, "
           "mismatched %ld; sampler on vs off: %ld draws, %ld mismatched, margin decided %lu\n",
           selftest ? " --selftest" : red_wrong ? " --red-wrong" : "", plans, plan_refused,
           decided, refused, mismatched, on_off_compared, on_off_bad, margin_used);
    free(row);
    free(scratch);
    if (selftest || red_wrong) {
        const int caught = selftest ? mismatched > 0 : on_off_bad > 0;
        printf(caught ? "SELFTEST PASS (red build caught)\n" : "SELFTEST FAIL (red build not caught)\n");
        return caught ? 0 : 1;
    }
    if (mismatched || on_off_bad || decided == 0 || margin_used == 0 || plan_refused * 2 > plans) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
