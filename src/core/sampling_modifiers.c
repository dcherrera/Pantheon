/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * OpenAI sampling controls that act on a logits row — see sampling_modifiers.h
 * for the contract and the ordering rationale.
 *
 * Pure C, libc + libm only.  No engine/model/CUDA deps, so the whole file is
 * unit-tested against real logit rows without a forward pass.
 */

#include "sampling_modifiers.h"

#include <math.h>

void pn_apply_logit_bias(float *logits, int n_vocab,
                         const pn_logit_bias_entry *bias, int n)
{
    if (!logits || !bias || n <= 0) return;
    for (int i = 0; i < n; i++) {
        uint32_t id = bias[i].token_id;
        if (id < (uint32_t)n_vocab)
            logits[id] += bias[i].bias;
    }
}

void pn_apply_presence_frequency_penalties(float *logits, int n_vocab,
                                           float presence_penalty,
                                           float frequency_penalty,
                                           const uint32_t *recent_tokens,
                                           int n_recent)
{
    if (!logits) return;
    if (presence_penalty == 0.0f && frequency_penalty == 0.0f) return;
    if (!recent_tokens || n_recent <= 0) return;

    for (int i = 0; i < n_recent; i++) {
        uint32_t tok = recent_tokens[i];
        if (tok >= (uint32_t)n_vocab) continue;
        /* Every occurrence pays the frequency penalty; the FIRST occurrence in
         * the window also pays the presence penalty.  Scanning backwards for an
         * earlier copy keeps this allocation-free — the window is repeat_last_n
         * entries (64 by default), so the O(n²) is a few thousand compares,
         * nothing beside a full-vocab softmax. */
        int seen_before = 0;
        for (int j = 0; j < i; j++) {
            if (recent_tokens[j] == tok) { seen_before = 1; break; }
        }
        logits[tok] -= frequency_penalty;
        if (!seen_before) logits[tok] -= presence_penalty;
    }
}

uint64_t pn_seed_to_rng_state(uint64_t seed)
{
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z ? z : 0x9E3779B97F4A7C15ULL;
}

float pn_seeded_rng_next_float(uint64_t *state)
{
    if (!state) return 0.0f;
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    /* Top 24 bits of the scrambled word → a float in [0,1) with no modulo bias
     * and no rounding to exactly 1.0f. */
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int pn_logprobs_extract(const float *logits, int n_vocab, uint32_t chosen,
                        int n_top, float *chosen_out,
                        uint32_t *top_ids_out, float *top_lp_out)
{
    if (!logits || n_vocab <= 0) return 0;
    if (n_top < 0) n_top = 0;
    if (n_top > PN_LOGPROBS_MAX_TOP) n_top = PN_LOGPROBS_MAX_TOP;

    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += expf(logits[i] - max_logit);
    /* log Z, so logprob(t) == logits[t] - log_z. */
    float log_z = max_logit + logf(sum);

    if (chosen_out)
        *chosen_out = ((int)chosen < n_vocab) ? logits[chosen] - log_z : 0.0f;

    if (n_top == 0 || !top_ids_out || !top_lp_out) return 0;

    /* Bounded insertion sort.  The `<= worst` early-out makes this ~O(n_vocab)
     * in practice and n_top is capped at 20, so a full 150k-vocab row costs a
     * single linear scan plus a handful of shifts. */
    int   n_kept = 0;
    float worst  = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        if (n_kept == n_top && logits[i] <= worst) continue;
        int pos = (n_kept < n_top) ? n_kept : n_top - 1;
        while (pos > 0 && top_lp_out[pos - 1] < logits[i]) {
            top_lp_out[pos]  = top_lp_out[pos - 1];
            top_ids_out[pos] = top_ids_out[pos - 1];
            pos--;
        }
        top_lp_out[pos]  = logits[i];
        top_ids_out[pos] = (uint32_t)i;
        if (n_kept < n_top) n_kept++;
        worst = top_lp_out[n_kept - 1];
    }
    for (int i = 0; i < n_kept; i++) top_lp_out[i] -= log_z;
    return n_kept;
}
