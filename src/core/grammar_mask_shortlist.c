/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Grammar mask from a shortlist — see grammar_mask_shortlist.h for why the
 * decision is exact.  Pure C, libc + libm (the outside bound's planner,
 * sampler_topn_tail.c).
 */
#include "grammar_mask_shortlist.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#define GMS_NEG_INF_BITS 0xff800000u

/* -inf built from its bits: under -ffast-math the INFINITY macro is not
 * reliable, and the sampler recognises an excluded entry by these exact bits. */
static float gms_neg_inf(void)
{
    const uint32_t bits = GMS_NEG_INF_BITS;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* Reads the BITS through the pointer: a float value passed through fast-math
 * code may be assumed finite, and the compare folded away. */
static bool gms_is_neg_inf(const float *p)
{
    uint32_t bits;
    memcpy(&bits, p, sizeof(bits));
    return bits == GMS_NEG_INF_BITS;
}

/* The logit the sampler will see for `id` after the penalties, replicating
 * pn_apply_presence_frequency_penalties and the repeat-penalty loop in their
 * order (presence/frequency per occurrence, then — sampling only — the repeat
 * penalty once per occurrence). */
static float gms_penalised(float v, uint32_t id, const gms_sampling *s)
{
    int count = 0;
    for (int i = 0; i < s->n_recent; i++) {
        if (s->recent[i] != id) continue;
        v -= s->frequency_penalty;
        if (count == 0) v -= s->presence_penalty;
        count++;
    }
    if (s->temperature > 0.0f && s->repeat_penalty != 1.0f) {
        for (int i = 0; i < count; i++)
            v = v > 0.0f ? v / s->repeat_penalty : v * s->repeat_penalty;
    }
    return v;
}

/* Margin over `bound` that absorbs a rounding difference between this replica
 * and the sampler's own penalty arithmetic. */
static float gms_margin(float bound)
{
    return 1e-4f + 1e-4f * fabsf(bound);
}

/* The outside allowed tokens' mass bound B, in units of exp((v - m) / T): `count` of them at
 * `bound`, or with the device's tail counts the `count` tokens in the highest bins first (the
 * shortlisted tokens taken out of the counts). Negative when the counts contradict the row. */
static double gms_outside_mass(const gms_outside *o, const float *logits, const uint32_t *ids,
                               int n_ids, float bound, double m, double inv_t)
{
    long tail[TNT_TAIL_BINS + 1];
    double left = o->count, mass = 0.0;

    if (!o->bins || !o->bin_counts)
        return o->count * exp(((double)bound - m) * inv_t);
    for (int j = 0; j <= o->bins->bins; j++) tail[j] = (long)o->bin_counts[j];
    for (int i = 0; i < n_ids; i++)
        tail[tnt_tail_bin(o->bins, tnt_key_from_float(logits[ids[i]]))]--;
    for (int j = 0; j <= o->bins->bins && left > 0.0; j++) {
        const double taken = (double)tail[j] < left ? (double)tail[j] : left;

        if (tail[j] < 0)
            return -1.0;
        mass += taken * exp((o->bins->upper[j] - m) * inv_t);
        left -= taken;
    }
    return mass;
}

/* Fewer than K allowed tokens clear `bound`: does the outside bound still prove the dense
 * draw keeps only shortlisted tokens (the planner's NUCLEUS, see the header)? The survivors
 * handed to the planner are the CLEAR tokens (penalised above `bound` + margin: all of them in
 * the dense top-K, since fewer than K do). An allowed shortlisted token penalised below that —
 * the repeat penalty on a recent token — may or may not make the dense top-K, so its mass joins
 * the outside's in the uncertain tail, and the cut must clear it as it clears `bound`. */
static bool gms_outside_decides(const float *logits, int n_vocab, const uint32_t *ids, int n_ids,
                                const float *penalised, int n_kept, float bound, int k,
                                const gms_sampling *s)
{
    const double inv_t = 1.0 / (double)s->temperature;
    const float strict = bound + gms_margin(bound);
    float clear[GMS_MAX];
    double m = 0.0, tail_mass;
    int n_clear = 0;
    tnt_plan plan;

    if (!(s->top_p > 0.0f && s->top_p < 1.0f))
        return false;
    for (int i = 0; i < n_kept; i++)
        if (penalised[i] > strict) {
            if (n_clear == 0 || penalised[i] > m) m = penalised[i];
            clear[n_clear++] = penalised[i];
        }
    if (n_clear < 1 || n_clear >= k)
        return false;
    tail_mass = gms_outside_mass(s->outside, logits, ids, n_ids, bound, m, inv_t);
    if (tail_mass < 0.0)
        return false;
    for (int i = 0; i < n_kept; i++)
        if (!(penalised[i] > strict)) tail_mass += exp(((double)penalised[i] - m) * inv_t);
    /* `strict` for the planner's checks (every survivor in the top-k, the cut above the tail)
     * absorbs a rounding difference between this replica's penalties and the sampler's; the
     * outside's mass itself is charged at the true `bound`. */
    tnt_sampling ts = { s->temperature, s->top_p, k < n_vocab ? k : 0, n_vocab,
                        s->outside->scale, tail_mass, 0 };
    return tnt_plan_draw(clear, n_clear, strict, &ts, &plan) == TNT_NUCLEUS;
}

int gms_mask_from_shortlist(float *logits, int n_vocab, const uint32_t *ids, int n_ids,
                             float bound, const gms_sampling *s,
                             gms_allowed_fn allowed, void *allowed_ctx)
{
    uint32_t kept_ids[GMS_MAX];
    float kept_vals[GMS_MAX], penalised[GMS_MAX];
    int n_kept = 0, n_clear = 0, verdict = GMS_DECIDED_CLEAR;
    const bool bounded = s->outside && s->temperature > 0.0f;
    const int k = s->temperature <= 0.0f ? 1
                : (bounded && (s->top_k <= 0 || s->top_k > n_vocab)) ? n_vocab : s->top_k;

    if (n_ids < 1 || n_ids > GMS_MAX || k < 1 || (k > n_ids && !bounded) || s->has_logit_bias ||
        s->repeat_penalty < 1.0f || s->presence_penalty < 0.0f || s->frequency_penalty < 0.0f)
        return GMS_UNDECIDED;
    for (int i = 0; i < n_ids; i++) {
        const uint32_t id = ids[i];
        if (gms_is_neg_inf(&logits[id]) || !allowed(allowed_ctx, id)) continue;
        kept_ids[n_kept] = id;
        kept_vals[n_kept] = logits[id];
        penalised[n_kept] = gms_penalised(logits[id], id, s);
        if (penalised[n_kept++] > bound + gms_margin(bound)) n_clear++;
    }
    if (n_clear < k) {
        if (!(bounded && gms_outside_decides(logits, n_vocab, ids, n_ids, penalised, n_kept,
                                             bound, k, s)))
            return GMS_UNDECIDED;
        verdict = GMS_DECIDED_BOUND;
    }
    const float neg_inf = gms_neg_inf();
    for (int i = 0; i < n_vocab; i++)
        logits[i] = neg_inf;
    for (int i = 0; i < n_kept; i++)
        logits[kept_ids[i]] = kept_vals[i];
    return verdict;
}

/* Sift the min-heap (by value) down from `at`. */
static void gms_heap_down(uint32_t *heap, const float *logits, int n, int at)
{
    for (;;) {
        int least = at, l = 2 * at + 1, r = l + 1;
        if (l < n && logits[heap[l]] < logits[heap[least]]) least = l;
        if (r < n && logits[heap[r]] < logits[heap[least]]) least = r;
        if (least == at) return;
        uint32_t t = heap[at];
        heap[at] = heap[least];
        heap[least] = t;
        at = least;
    }
}

#if defined(__SSE2__)
/* Offer entry i to the shortlist heap (the scalar select's loop body). Always inlined, so
 * n and left_out stay in registers in the callers' loops. */
static inline __attribute__((always_inline)) void
gms_offer(const float *restrict logits, int i, uint32_t *restrict ids, int *restrict n,
          float *restrict left_out)
{
    if (gms_is_neg_inf(&logits[i])) return;
    if (*n < GMS_MAX) {
        ids[(*n)++] = (uint32_t)i;
        if (*n == GMS_MAX)
            for (int j = GMS_MAX / 2 - 1; j >= 0; j--) gms_heap_down(ids, logits, *n, j);
        return;
    }
    if (logits[i] <= logits[ids[0]]) {
        if (logits[i] > *left_out) *left_out = logits[i];
        return;
    }
    if (logits[ids[0]] > *left_out) *left_out = logits[ids[0]];
    ids[0] = (uint32_t)i;
    gms_heap_down(ids, logits, *n, 0);
}

/* PN_GMS_SELECT_RED=1 (gates only): the block skip also takes entries up to 1e-3 above the
 * root, so a block holding a candidate can be passed over. */
static float gms_select_red_slack(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PN_GMS_SELECT_RED");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v ? 1e-3f : 0.0f;
}

/* Once the heap is full, the scalar loop leaves the heap alone for an entry <= its root and
 * only raises left_out to it. A block of 4 entries that are all <= the root therefore changes
 * nothing but left_out, by their maximum: kept here as a running vector maximum and merged at
 * the end (a maximum does not depend on the order it is taken in; -inf leaves it as it was).
 * Any other block (a candidate, or a NaN: unordered compares false) runs the scalar body.
 * Returns the index reached. */
static int gms_select_blocks(const float *logits, int n_vocab, int i, uint32_t *ids, int *n,
                             float *left_out)
{
    const float slack = gms_select_red_slack();
    __m128 skipped = _mm_set1_ps(*left_out);
    float lanes[4];

    for (; i + 4 <= n_vocab; i += 4) {
        const __m128 v = _mm_loadu_ps(logits + i);
        if (_mm_movemask_ps(_mm_cmple_ps(v, _mm_set1_ps(logits[ids[0]] + slack))) == 0xf) {
            skipped = _mm_max_ps(skipped, v);
            continue;
        }
        for (int j = 0; j < 4; j++) gms_offer(logits, i + j, ids, n, left_out);
    }
    _mm_storeu_ps(lanes, skipped);
    for (int j = 0; j < 4; j++)
        if (lanes[j] > *left_out) *left_out = lanes[j];
    return i;
}
#endif

int gms_select_shortlist(const float *logits, int n_vocab, uint32_t *ids, float *bound)
{
    int n = 0, i = 0;
    float left_out = gms_neg_inf();

#if defined(__SSE2__)
    for (; i < n_vocab && n < GMS_MAX; i++) gms_offer(logits, i, ids, &n, &left_out);
    if (n == GMS_MAX)
        i = gms_select_blocks(logits, n_vocab, i, ids, &n, &left_out);
#endif
    for (; i < n_vocab; i++) {
        if (gms_is_neg_inf(&logits[i])) continue;
        if (n < GMS_MAX) {
            ids[n++] = (uint32_t)i;
            if (n == GMS_MAX)
                for (int j = GMS_MAX / 2 - 1; j >= 0; j--) gms_heap_down(ids, logits, n, j);
            continue;
        }
        if (logits[i] <= logits[ids[0]]) {
            if (logits[i] > left_out) left_out = logits[i];
            continue;
        }
        if (logits[ids[0]] > left_out) left_out = logits[ids[0]];
        ids[0] = (uint32_t)i;
        gms_heap_down(ids, logits, n, 0);
    }
    if (n == 0) return 0;
    if (n < GMS_MAX || gms_is_neg_inf(&left_out)) {
        /* Every finite entry is listed: a top-N row, whose unlisted TRUE logits
         * sit at or below its smallest entry. */
        float min = logits[ids[0]];
        for (int i = 1; i < n; i++)
            if (logits[ids[i]] < min) min = logits[ids[i]];
        *bound = min;
    } else {
        *bound = left_out;
    }
    return n;
}
