/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_topn_tail.c
 * @brief The top-N tail planner and the residual draw (see sampler_topn_tail.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#include "sampler_topn_tail.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Relative margin on the top-p cut: the dense path sums float probabilities normalised by
 * a float sum over the whole vocabulary; a cut closer than this to the boundary goes dense. */
#define TNT_CUT_MARGIN 1e-4
/* The nucleus's last value must clear the tail's bound by this much (in v / T units): the
 * dense top-p keeps ties at its threshold, so a tail token equal to it would join. */
#define TNT_TIE_MARGIN 1e-4

static int tnt_cmp_desc(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

/* NUCLEUS or DENSE: the top-p cut over the survivors' unnormalised masses w (any order),
 * for every normaliser in [W, W + B]. */
static tnt_kind tnt_plan_nucleus(const double *w, int S, double W, double B, double vmax_t,
                                 float bound, const tnt_sampling *s, tnt_plan *out)
{
    double sorted[TNT_MAX], prefix = 0.0, before = 0.0, last;
    const double lo = (double)s->top_p * W, hi = (double)s->top_p * (W + B);
    int j;

    for (j = 0; j < S; j++) sorted[j] = w[j];
    qsort(sorted, (size_t)S, sizeof(double), tnt_cmp_desc);
    for (j = 0; j < S; j++) {
        before = prefix;
        prefix += sorted[j];
        if (prefix >= hi * (1.0 + TNT_CUT_MARGIN))
            break;
    }
    if (j == S || !(before < lo * (1.0 - TNT_CUT_MARGIN)))
        return TNT_DENSE;
    /* The cut's value against the tail: log(sorted[j]) = (v - m) / T. */
    last = log(sorted[j]) + vmax_t;
    if (!(last - (double)bound / (double)s->temperature > TNT_TIE_MARGIN))
        return TNT_DENSE;
    out->nucleus_k = j + 1;
    return TNT_NUCLEUS;
}

tnt_kind tnt_plan_draw(const float *vals, int S, float bound, const tnt_sampling *s,
                       tnt_plan *out)
{
    double w[TNT_MAX], W = 0.0, B, m;
    const double inv_t = s && s->temperature > 0.0f ? 1.0 / (double)s->temperature : 0.0;
    int i;

    out->kind = TNT_DENSE;
    out->nucleus_k = 0;
    out->head_mass = 0.0;
    if (!vals || !s || S < 1 || S > TNT_MAX || inv_t <= 0.0 || S >= s->n_vocab)
        return TNT_DENSE;
    m = vals[0];
    for (i = 0; i < S; i++) {
        /* Under a partial top-k a survivor penalised below the tail may leave the top-k set. */
        if (s->top_k > 0 && s->top_k < s->n_vocab && !s->survivors_in_topk &&
            !(vals[i] >= bound))
            return TNT_DENSE;
        if (vals[i] > m) m = vals[i];
    }
    for (i = 0; i < S; i++) {
        w[i] = exp(((double)vals[i] - m) * inv_t);
        W += w[i];
    }
    B = s->tail_scale * (s->tail_mass >= 0.0 ? s->tail_mass
                                             : (double)(s->n_vocab - S) *
                                                   exp(((double)bound - m) * inv_t));
    if (s->top_p < 1.0f) {
        out->kind = tnt_plan_nucleus(w, S, W, B, m * inv_t, bound, s, out);
        return out->kind;
    }
    out->head_mass = W / (W + B);
    for (i = 0; i < S; i++) out->head_prob[i] = w[i] / W;
    out->kind = TNT_MIXTURE;
    return TNT_MIXTURE;
}

uint32_t tnt_residual_draw(const float *probs, int n_vocab, const int *ids,
                           const tnt_plan *plan, int S, float u)
{
    const double target = (double)u - plan->head_mass;
    double cum = 0.0, r;
    int i, at = 0, last = -1;

    for (i = 0; i < n_vocab; i++) {
        r = (double)probs[i];
        if (at < S && ids[at] == i)
            r -= plan->head_mass * plan->head_prob[at++];
        if (r <= 0.0)
            continue;
        last = i;
        cum += r;
        if (target < cum)
            return (uint32_t)i;
    }
    return last >= 0 ? (uint32_t)last : 0u;
}

uint32_t tnt_key_from_float(float value)
{
    uint32_t bits, u;

    memcpy(&bits, &value, sizeof(bits));
    u = bits >> 16;
    return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}

static double tnt_float_from_key(uint32_t key)
{
    const uint32_t u = (key & 0x8000u) ? (key & 0x7FFFu) : (~key & 0xFFFFu);
    const uint32_t bits = u << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return (double)value;
}

/* The smallest order key whose value is strictly above x (x below the largest finite). */
static uint32_t tnt_key_above(float x)
{
    uint32_t key = tnt_key_from_float(x);

    while (key < 0xFFFFu && !(tnt_float_from_key(key) > (double)x)) key++;
    while (key > 0u && tnt_float_from_key(key - 1u) > (double)x) key--;
    return key;
}

void tnt_tail_bins_make(float bound, float x_min, float temperature, int n_vocab,
                        tnt_tail_bins *out)
{
    const double span = (double)temperature * (log((double)(n_vocab > 1 ? n_vocab : 2)) + 18.0);
    const double delta = span / (double)(TNT_TAIL_BINS - 2);
    const uint32_t kx = x_min < bound ? tnt_key_above(x_min) : 0x10000u;
    int j, n = 1, at;

    out->keys[0] = tnt_key_from_float(bound);
    for (j = 1; j < TNT_TAIL_BINS - 1 && out->keys[n - 1] > 0u; j++) {
        uint32_t key = tnt_key_from_float((float)((double)bound - j * delta));

        out->keys[n] = key < out->keys[n - 1] ? key : out->keys[n - 1] - 1u;
        n++;
    }
    /* The smallest survivor's own threshold: the count above it is then exact. */
    out->xmin_index = -1;
    if (kx < out->keys[0]) {
        for (at = 0; at < n && out->keys[at] > kx; at++)
            ;
        if (at == n || out->keys[at] != kx) {
            memmove(&out->keys[at + 1], &out->keys[at], (size_t)(n - at) * sizeof(out->keys[0]));
            out->keys[at] = kx;
            n++;
        }
        out->xmin_index = at;
    }
    out->bins = n;
    out->upper[0] = (double)bound;
    for (j = 1; j <= n; j++)
        out->upper[j] = tnt_float_from_key(out->keys[j - 1]);
}

int tnt_tail_bin(const tnt_tail_bins *tb, uint32_t key)
{
    int low = 0, high = tb->bins;

    while (low < high) {
        const int mid = (low + high) >> 1;

        if (tb->keys[mid] <= key)
            high = mid;
        else
            low = mid + 1;
    }
    return low;
}

void tnt_tail_bins_count(const tnt_tail_bins *tb, const float *row, int n, uint32_t *counts)
{
    int i;

    memset(counts, 0, (size_t)(tb->bins + 1) * sizeof(counts[0]));
    for (i = 0; i < n; i++)
        counts[tnt_tail_bin(tb, tnt_key_from_float(row[i]))]++;
}

int tnt_tail_from_counts(const tnt_tail_bins *tb, const uint32_t *counts, const float *raw, int S,
                         double m, double inv_t, float x_min, int top_k, double *B, int *in_topk)
{
    long tail[TNT_TAIL_BINS + 1];
    long above = 0;
    double b = 0.0;
    int i, j;

    for (j = 0; j <= tb->bins; j++) tail[j] = (long)counts[j];
    for (i = 0; i < S; i++) tail[tnt_tail_bin(tb, tnt_key_from_float(raw[i]))]--;
    for (j = 0; j <= tb->bins; j++) {
        if (tail[j] < 0)
            return -1;
        b += (double)tail[j] * exp((tb->upper[j] - m) * inv_t);
        if (tb->xmin_index >= 0 ? j <= tb->xmin_index : tb->upper[j] > (double)x_min)
            above += tail[j];
    }
    *B = b;
    *in_topk = top_k > 0 && (long)(S - 1) + above < (long)top_k;
    return 0;
}
