/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Grammar mask from a shortlist — see grammar_mask_shortlist.h for why the
 * decision is exact.  Pure C, libc-only.
 */
#include "grammar_mask_shortlist.h"

#include <math.h>
#include <string.h>

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

bool gms_mask_from_shortlist(float *logits, int n_vocab, const uint32_t *ids, int n_ids,
                             float bound, const gms_sampling *s,
                             gms_allowed_fn allowed, void *allowed_ctx)
{
    uint32_t kept_ids[GMS_MAX];
    float kept_vals[GMS_MAX];
    int n_kept = 0, n_clear = 0;
    const int k = s->temperature <= 0.0f ? 1 : s->top_k;

    if (n_ids < 1 || n_ids > GMS_MAX || k < 1 || k > n_ids || s->has_logit_bias ||
        s->repeat_penalty < 1.0f || s->presence_penalty < 0.0f || s->frequency_penalty < 0.0f)
        return false;
    for (int i = 0; i < n_ids; i++) {
        const uint32_t id = ids[i];
        if (gms_is_neg_inf(&logits[id]) || !allowed(allowed_ctx, id)) continue;
        kept_ids[n_kept] = id;
        kept_vals[n_kept++] = logits[id];
        if (gms_penalised(logits[id], id, s) > bound + gms_margin(bound)) n_clear++;
    }
    if (n_clear < k)
        return false;
    const float neg_inf = gms_neg_inf();
    for (int i = 0; i < n_vocab; i++)
        logits[i] = neg_inf;
    for (int i = 0; i < n_kept; i++)
        logits[kept_ids[i]] = kept_vals[i];
    return true;
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

int gms_select_shortlist(const float *logits, int n_vocab, uint32_t *ids, float *bound)
{
    int n = 0;
    float left_out = gms_neg_inf();

    for (int i = 0; i < n_vocab; i++) {
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
