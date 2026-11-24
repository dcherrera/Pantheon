/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_margin_draw.c
 * @brief The dense full-path draw decided from the row's top candidates (sampler_margin_draw.h).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "sampler_margin_draw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#define MARGIN_NEG_INF_BITS 0xff800000u

/* Red arms (gates only; a gate that stays green under one proves nothing):
 * PN_MARGIN_RED=1 every margin 0, so a draw within rounding of a boundary is decided from the
 * reals instead of refused; PN_MARGIN_RED=2 every decided draw returns the NEXT candidate. */
static int margin_red(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PN_MARGIN_RED");
        v = (e && (e[0] == '1' || e[0] == '2')) ? e[0] - '0' : 0;
    }
    return v;
}

/* PN_MARGIN_DRAW=0 turns the margin draw off (the dense path draws every row). */
static int g_margin_override = -1;

int pn_margin_draw_enabled(void)
{
    static int v = -1;
    if (g_margin_override >= 0)
        return g_margin_override;
    if (v < 0) {
        const char *e = getenv("PN_MARGIN_DRAW");
        v = !(e && e[0] == '0');
    }
    return v;
}

void pn_margin_draw_set_enabled(int on) { g_margin_override = on < 0 ? -1 : (on != 0); }

static double margin_mu(void) { return margin_red() == 1 ? 0.0 : PN_MARGIN_MU; }
static double margin_beta(void) { return margin_red() == 1 ? 0.0 : PN_MARGIN_BETA; }

/* Value desc, id asc: "a ranks above b". */
static int margin_above(float va, uint32_t ia, float vb, uint32_t ib)
{
    return va > vb || (va == vb && ia < ib);
}

/* Min-heap by rank (root = the lowest-ranked kept entry), sifted down from `at`. */
static void margin_heap_down(uint32_t *ids, float *vals, int n, int at)
{
    for (;;) {
        int low = at, l = 2 * at + 1, r = l + 1;
        if (l < n && margin_above(vals[low], ids[low], vals[l], ids[l])) low = l;
        if (r < n && margin_above(vals[low], ids[low], vals[r], ids[r])) low = r;
        if (low == at)
            return;
        float tv = vals[at]; vals[at] = vals[low]; vals[low] = tv;
        uint32_t ti = ids[at]; ids[at] = ids[low]; ids[low] = ti;
        at = low;
    }
}

static int margin_is_neg_inf(const float *p)
{
    uint32_t bits;
    memcpy(&bits, p, sizeof(bits));
    return bits == MARGIN_NEG_INF_BITS;
}

/* NaN or +inf (read through the bits: fast-math folds float tests for them). The dense path's
 * arithmetic on such a row is not modelled, so the plan refuses it. */
static int margin_is_unmodelled(const float *p)
{
    uint32_t bits;
    memcpy(&bits, p, sizeof(bits));
    return (bits & 0x7f800000u) == 0x7f800000u && bits != MARGIN_NEG_INF_BITS;
}

/* Offer entry i to the heap (n entries, capacity c); *bad set on NaN / +inf. */
static void margin_offer(const float *row, uint32_t i, uint32_t *ids, float *vals, int *n, int c,
                         int *bad)
{
    if (margin_is_neg_inf(&row[i]))
        return;
    if (margin_is_unmodelled(&row[i])) {
        *bad = 1;
        return;
    }
    if (*n < c) {
        ids[*n] = i;
        vals[*n] = row[i];
        if (++*n == c)
            for (int j = c / 2 - 1; j >= 0; j--) margin_heap_down(ids, vals, c, j);
        return;
    }
    if (!margin_above(row[i], i, vals[0], ids[0]))
        return;
    ids[0] = i;
    vals[0] = row[i];
    margin_heap_down(ids, vals, c, 0);
}

int pn_margin_select_top(const float *row, int n, int c, uint32_t *ids, float *vals)
{
    int got = 0, i = 0, bad = 0;

    if (c < 1 || c > PN_MARGIN_CAND || n < 1)
        return 0;
    for (; i < n && got < c; i++)
        margin_offer(row, (uint32_t)i, ids, vals, &got, c, &bad);
#if defined(__SSE2__)
    /* Once full, an entry can only enter by ranking above the root: skip any 4 that are all
     * ordered and <= it (an equal value with a higher id never ranks above, and every later id
     * is higher). -inf never passes; a NaN lane goes to the scalar offer, which flags it. */
    if (got == c) {
        for (; i + 4 <= n; i += 4) {
            const __m128 v = _mm_loadu_ps(row + i);
            if (_mm_movemask_ps(_mm_cmple_ps(v, _mm_set1_ps(vals[0]))) == 0xf)
                continue;
            for (int j = 0; j < 4; j++)
                margin_offer(row, (uint32_t)(i + j), ids, vals, &got, c, &bad);
        }
    }
#endif
    for (; i < n; i++)
        margin_offer(row, (uint32_t)i, ids, vals, &got, c, &bad);
    if (bad)
        return -1;
    /* Heap -> rank order (insertion sort: c <= 64). */
    for (int a = 1; a < got; a++) {
        float v = vals[a];
        uint32_t id = ids[a];
        int b = a - 1;
        for (; b >= 0 && margin_above(v, id, vals[b], ids[b]); b--) {
            vals[b + 1] = vals[b];
            ids[b + 1] = ids[b];
        }
        vals[b + 1] = v;
        ids[b + 1] = id;
    }
    return got;
}

/* First index after the run of entries tied with d[at] (d is non-increasing). */
static int margin_tie_end(const double *d, int n, int at)
{
    int e = at + 1;
    while (e < n && d[e] == d[at]) e++;
    return e;
}

/* Kept count after the top-p stop over q[0..nk) (value desc); <= 0 = a refusal. */
static int margin_top_p_count(const double *q, const double *d, int nk, double top_p)
{
    double cum = 0.0;
    int t = 0;

    /* The dense loop adds q[t] while the mass before it is < top_p. */
    while (t < nk && cum < top_p) cum += q[t++];
    /* Mass before the last added entry was clear of top_p from below ... */
    if (top_p - (cum - q[t - 1]) <= margin_beta())
        return PN_MARGIN_EDGE_TOPP;
    /* ... and, when the loop stopped early, the mass after it from above. */
    if (t < nk && cum - top_p <= margin_beta())
        return PN_MARGIN_EDGE_TOPP;
    /* Kept = q >= the last added value: its exact ties ride along; a near tie refuses. */
    const int end = margin_tie_end(d, nk, t - 1);
    if (end < nk && d[t - 1] - d[end] <= margin_mu())
        return PN_MARGIN_EDGE_TOPP;
    return end;
}

int pn_margin_plan_dense(const float *logits, int n_vocab, float temperature, int top_k,
                         float top_p, pn_margin_plan *plan)
{
    uint32_t ids[PN_MARGIN_CAND];
    float vals[PN_MARGIN_CAND];
    double d[PN_MARGIN_CAND], q[PN_MARGIN_CAND], sum = 0.0;

    if (!logits || !plan || !(temperature > 0.0f) || top_k < 1 || top_k > PN_MARGIN_CAND - 2 ||
        n_vocab <= PN_MARGIN_CAND)
        return PN_MARGIN_INELIGIBLE;
    const int c = pn_margin_select_top(logits, n_vocab, PN_MARGIN_CAND, ids, vals);
    if (c < PN_MARGIN_CAND)
        return PN_MARGIN_INELIGIBLE;   /* fewer finite entries, or a NaN / +inf in the row */
    /* The dense path's own arithmetic up to the exponent: x = l * (1/T), d = x - max. */
    const float inv_temp = 1.0f / temperature;
    const float x_max = vals[0] * inv_temp;
    for (int j = 0; j < c; j++) {
        const float x = vals[j] * inv_temp;
        d[j] = (double)(x - x_max);
    }
    /* Top-k: everything tied with the k-th; the next value down must be MU clear. */
    const int nk = margin_tie_end(d, c, top_k - 1);
    if (nk >= c || d[top_k - 1] - d[nk] <= margin_mu())
        return PN_MARGIN_EDGE_TOPK;
    if (d[nk - 1] < -60.0)
        return PN_MARGIN_RANGE;
    for (int j = 0; j < nk; j++) {
        q[j] = exp(d[j]);
        sum += q[j];
    }
    for (int j = 0; j < nk; j++) q[j] /= sum;
    int np = nk;
    if (top_p < 1.0f) {
        np = margin_top_p_count(q, d, nk, (double)top_p);
        if (np <= 0)
            return np;
    }
    /* The final scan: P in ascending id order, renormalised. */
    uint32_t pid[PN_MARGIN_CAND];
    double pq[PN_MARGIN_CAND], psum = 0.0;
    for (int j = 0; j < np; j++) {
        pid[j] = ids[j];
        pq[j] = q[j];
        psum += q[j];
    }
    for (int a = 1; a < np; a++) {
        uint32_t id = pid[a];
        double v = pq[a];
        int b = a - 1;
        for (; b >= 0 && pid[b] > id; b--) {
            pid[b + 1] = pid[b];
            pq[b + 1] = pq[b];
        }
        pid[b + 1] = id;
        pq[b + 1] = v;
    }
    double cum = 0.0;
    for (int j = 0; j < np; j++) {
        cum += pq[j] / psum;
        plan->ids[j] = pid[j];
        plan->cum[j] = cum;
    }
    plan->n = np;
    return PN_MARGIN_OK;
}

int pn_margin_decide(const pn_margin_plan *plan, float u, uint32_t *token)
{
    const double r = (double)u;
    double below = 0.0;

    if (!plan || plan->n < 1 || !token)
        return 0;
    for (int j = 0; j < plan->n; j++) {
        if (r < plan->cum[j]) {
            if (plan->cum[j] - r <= margin_beta())
                return 0;
            if (j > 0 && r - below <= margin_beta())
                return 0;
            *token = plan->ids[margin_red() == 2 ? (j + 1) % plan->n : j];
            return 1;
        }
        below = plan->cum[j];
    }
    return 0;   /* at or past the total mass: the dense path's fallback scan decides */
}

int pn_margin_draw(const float *logits, int n_vocab, float temperature, int top_k, float top_p,
                   float u, uint32_t *token)
{
    pn_margin_plan plan;

    return pn_margin_plan_dense(logits, n_vocab, temperature, top_k, top_p, &plan) ==
               PN_MARGIN_OK &&
           pn_margin_decide(&plan, u, token);
}
