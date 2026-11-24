/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_sampler_rows.c
 * @brief Offline sampler bench on real dense decode rows (PN_HOST_ROW_DUMP files):
 *        replays each unmasked row through the production sampler (must reproduce the serve's
 *        token), times it with the margin draw (sampler_margin_draw.h) off and on, checks on
 *        against off on many uniforms (same token per uniform), and times the shortlist
 *        select against the scalar loop it replaced (same ids and bound). Exit 1 on any
 *        mismatch.
 *
 * usage: bench_sampler_rows <rows.bin> [uniforms_per_row=150] [reps=5]
 *
 * @status Instrument only.
 * @issues None
 * @todo None
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../src/core/engine.h"
#include "../src/core/grammar_mask_shortlist.h"
#include "../src/core/sampler_margin_draw.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* The scalar shortlist select gms_select_shortlist replaced (no SSE prefilter): the bench
 * times both and checks they agree (ids, order, bound value). */
static int ref_bits_neg_inf(const float *p)
{
    uint32_t b;
    memcpy(&b, p, 4);
    return b == 0xff800000u;
}

static void ref_heap_down(uint32_t *heap, const float *logits, int n, int at)
{
    for (;;) {
        int least = at, l = 2 * at + 1, r = l + 1;
        if (l < n && logits[heap[l]] < logits[heap[least]]) least = l;
        if (r < n && logits[heap[r]] < logits[heap[least]]) least = r;
        if (least == at) return;
        uint32_t t = heap[at]; heap[at] = heap[least]; heap[least] = t;
        at = least;
    }
}

static int ref_select_scalar(const float *logits, int n_vocab, uint32_t *ids, float *bound)
{
    int n = 0;
    const uint32_t ninf_bits = 0xff800000u;
    float left_out;
    memcpy(&left_out, &ninf_bits, 4);
    for (int i = 0; i < n_vocab; i++) {
        if (ref_bits_neg_inf(&logits[i])) continue;
        if (n < GMS_MAX) {
            ids[n++] = (uint32_t)i;
            if (n == GMS_MAX)
                for (int j = GMS_MAX / 2 - 1; j >= 0; j--) ref_heap_down(ids, logits, n, j);
            continue;
        }
        if (logits[i] <= logits[ids[0]]) {
            if (logits[i] > left_out) left_out = logits[i];
            continue;
        }
        if (logits[ids[0]] > left_out) left_out = logits[ids[0]];
        ids[0] = (uint32_t)i;
        ref_heap_down(ids, logits, n, 0);
    }
    if (n == 0) return 0;
    if (n < GMS_MAX || ref_bits_neg_inf(&left_out)) {
        float mn = logits[ids[0]];
        for (int j = 1; j < n; j++) if (logits[ids[j]] < mn) mn = logits[ids[j]];
        *bound = mn;
    } else {
        *bound = left_out;
    }
    return n;
}

typedef struct {
    int n_vocab, top_k, repeat_last_n, rng_seeded, masked, token, n_recent;
    float temperature, top_p, repeat_penalty, presence, frequency;
    uint64_t rng_state;
    uint32_t *recent;
    float *row;
} dumped_row;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int read_row(FILE *f, dumped_row *r)
{
    char magic[4];
    uint32_t version;
    float f2[2], pf[2];

    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PNRD", 4) != 0)
        return 0;
    if (fread(&version, 4, 1, f) != 1 || version != 1 || fread(&r->n_vocab, 4, 1, f) != 1 ||
        fread(f2, 4, 2, f) != 2 || fread(&r->top_k, 4, 1, f) != 1 ||
        fread(&r->repeat_penalty, 4, 1, f) != 1 || fread(&r->repeat_last_n, 4, 1, f) != 1 ||
        fread(pf, 4, 2, f) != 2 || fread(&r->rng_seeded, 4, 1, f) != 1 ||
        fread(&r->rng_state, 8, 1, f) != 1 || fread(&r->masked, 4, 1, f) != 1 ||
        fread(&r->token, 4, 1, f) != 1 || fread(&r->n_recent, 4, 1, f) != 1)
        return 0;
    r->temperature = f2[0];
    r->top_p = f2[1];
    r->presence = pf[0];
    r->frequency = pf[1];
    r->recent = r->n_recent > 0 ? malloc((size_t)r->n_recent * 4) : NULL;
    r->row = malloc((size_t)r->n_vocab * 4);
    if (!r->row || (r->n_recent > 0 && !r->recent))
        return 0;
    if ((r->n_recent > 0 && fread(r->recent, 4, (size_t)r->n_recent, f) != (size_t)r->n_recent) ||
        fread(r->row, 4, (size_t)r->n_vocab, f) != (size_t)r->n_vocab)
        return 0;
    return 1;
}

/* The serve's grammar slots hand an already-dense row: the refill is a no-op. */
static int dense_already(void *ctx, float *logits, int n_vocab)
{
    (void)ctx; (void)logits; (void)n_vocab;
    return 0;
}

static pn_sampling_params params_of(const dumped_row *r)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = r->temperature;
    sp.top_p = r->top_p;
    sp.top_k = r->top_k;
    sp.repeat_penalty = r->repeat_penalty;
    sp.repeat_last_n = r->repeat_last_n;
    sp.presence_penalty = r->presence;
    sp.frequency_penalty = r->frequency;
    sp.rng_seeded = r->rng_seeded;
    sp.rng_state = r->rng_state;
    sp.dense_row = dense_already;
    return sp;
}

/* The production draw on a copy, the margin draw on or off; u < 0 = the recorded RNG stream,
 * else an injected uniform. */
static uint32_t production_draw(const dumped_row *r, float *scratch, float u, int margin_on)
{
    pn_sampling_params sp = params_of(r);
    uint32_t recent[4096];
    const int nr = r->n_recent < 4096 ? r->n_recent : 4096;

    if (u >= 0.0f) {
        sp.carried_uniform_set = 1;
        sp.carried_uniform = u;
    }
    memcpy(recent, r->recent, (size_t)nr * 4);
    memcpy(scratch, r->row, (size_t)r->n_vocab * 4);
    pn_margin_draw_set_enabled(margin_on);
    const uint32_t tok = pn_sample_next_token(scratch, r->n_vocab, &sp, recent, nr);
    pn_margin_draw_set_enabled(-1);
    return tok;
}

/* Fastest of `reps` production draws (recorded RNG stream), margin draw on or off. */
static double time_draw(const dumped_row *r, float *scratch, int reps, int margin_on)
{
    double best = 1e9;
    for (int k = 0; k < reps; k++) {
        const double t0 = now_ms();
        production_draw(r, scratch, -1.0f, margin_on);
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* The row the full path receives: the repetition penalty applied (presence/frequency 0 here
 * is checked by the caller), exactly as sampler_apply_repeat_penalty does. */
static void penalised_row(const dumped_row *r, float *out)
{
    memcpy(out, r->row, (size_t)r->n_vocab * 4);
    if (r->repeat_penalty == 1.0f)
        return;
    for (int i = 0; i < r->n_recent; i++) {
        uint32_t t = r->recent[i];
        if (t < (uint32_t)r->n_vocab)
            out[t] = out[t] > 0.0f ? out[t] / r->repeat_penalty : out[t] * r->repeat_penalty;
    }
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return n ? v[n / 2] : 0.0;
}

/* The shortlist select (the masked path's first pass) on one row: production (SSE prefilter)
 * vs the scalar reference; 1 when they disagree on the ids, their order or the bound. */
static int bench_select(const dumped_row *r, int reps, double *t_ref, double *t_prod)
{
    uint32_t ids[GMS_MAX], ids2[GMS_MAX];
    float bound = 0.0f, bound2 = 0.0f;
    int n1 = 0, n2 = 0;

    *t_ref = *t_prod = 1e9;
    for (int k = 0; k < reps; k++) {
        const double t0 = now_ms();
        n1 = ref_select_scalar(r->row, r->n_vocab, ids, &bound);
        const double t1 = now_ms();
        n2 = gms_select_shortlist(r->row, r->n_vocab, ids2, &bound2);
        const double t2 = now_ms();
        if (t1 - t0 < *t_ref) *t_ref = t1 - t0;
        if (t2 - t1 < *t_prod) *t_prod = t2 - t1;
    }
    return n1 != n2 || memcmp(ids, ids2, (size_t)n1 * 4) != 0 || !(bound == bound2);
}

/* The sampler with the margin draw on against off on one row, for n_u random uniforms and
 * uniforms next to every boundary of the row's plan: mismatches (decided counts in *n). */
static long on_off_row(const dumped_row *r, float *scratch, float *pen, int n_u, uint64_t *lcg,
                       long *n)
{
    pn_margin_plan plan;
    long bad = 0;

    penalised_row(r, pen);
    const int ok = pn_margin_plan_dense(pen, r->n_vocab, r->temperature, r->top_k, r->top_p,
                                        &plan) == PN_MARGIN_OK;
    for (int i = 0; i < n_u + (ok ? plan.n * 6 : 0); i++) {
        float u;
        if (i < n_u) {
            *lcg = *lcg * 6364136223846793005ull + 1442695040888963407ull;
            u = (float)((*lcg >> 40) / 16777216.0);
        } else {
            static const double off[6] = { -2e-4, -1.01e-4, -1e-6, 1e-6, 1.01e-4, 2e-4 };
            u = (float)(plan.cum[(i - n_u) / 6] + off[(i - n_u) % 6]);
            if (!(u >= 0.0f && u < 1.0f)) continue;
        }
        bad += production_draw(r, scratch, u, 1) != production_draw(r, scratch, u, 0);
        (*n)++;
    }
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <rows.bin> [uniforms_per_row] [reps]\n", argv[0]);
        return 2;
    }
    const int n_u = argc > 2 ? atoi(argv[2]) : 150;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    FILE *f = fopen(argv[1], "rb");
    if (!f)
        return 2;
    dumped_row r;
    float *scratch = NULL, *pen = NULL;
    int rows = 0, unmasked = 0, reproduced = 0, sel_mismatch = 0;
    int n_dense = 0, n_margin = 0, n_sel = 0;
    long compared = 0, mismatched = 0;
    static double t_dense[4096], t_margin[4096], t_sel[4096], t_sel2[4096];
    uint64_t lcg = 12345;
    const unsigned long margin_before = pn_sampler_margin_draws();

    while (read_row(f, &r)) {
        rows++;
        if (!scratch) {
            scratch = malloc((size_t)r.n_vocab * 4);
            pen = malloc((size_t)r.n_vocab * 4);
        }
        if (n_sel < 4096) {
            sel_mismatch += bench_select(&r, reps, &t_sel[n_sel], &t_sel2[n_sel]);
            n_sel++;
        }
        if (!r.masked && r.temperature > 0.0f && r.presence == 0.0f && r.frequency == 0.0f) {
            unmasked++;
            reproduced += (int)production_draw(&r, scratch, -1.0f, 1) == r.token;
            if (n_dense < 4096) t_dense[n_dense++] = time_draw(&r, scratch, reps, 0);
            if (n_margin < 4096) t_margin[n_margin++] = time_draw(&r, scratch, reps, 1);
            mismatched += on_off_row(&r, scratch, pen, n_u, &lcg, &compared);
        }
        free(r.row);
        free(r.recent);
    }
    fclose(f);
    printf("rows %d, unmasked sampled %d, serve token reproduced (margin on) %d/%d\n", rows,
           unmasked, reproduced, unmasked);
    printf("sampler draw, unmasked dense row: margin off median %.3f ms, on %.3f ms "
           "(margin decided %lu of all draws)\n", median(t_dense, n_dense),
           median(t_margin, n_margin), pn_sampler_margin_draws() - margin_before);
    printf("margin on vs off, same uniform: %ld draws, %ld mismatched\n", compared, mismatched);
    printf("shortlist select (all rows): scalar median %.3f ms, production (SSE) %.3f ms; "
           "mismatched %d\n", median(t_sel, n_sel), median(t_sel2, n_sel), sel_mismatch);
    free(scratch);
    free(pen);
    return mismatched || sel_mismatch || reproduced != unmasked ? 1 : 0;
}
