/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ngram_draft.h
 * @brief Prompt-lookup n-gram drafter for speculative decoding: proposes the
 *        continuation of the most recent earlier occurrence of the current
 *        token-history suffix. Host-side and model-free — the "draft model"
 *        is the sequence's own repetition (code, quoted context, templated
 *        text), so drafting costs microseconds and every accepted draft is
 *        verified by the real model before it is emitted.
 *
 * @status Working; used by the qwen35moe speculative greedy decode drivers.
 * @issues None
 * @todo None
 */

#ifndef PN_NGRAM_DRAFT_H
#define PN_NGRAM_DRAFT_H

#include <stdint.h>

/**
 * Propose up to max_k draft tokens from the token history (prompt + all
 * committed output, oldest first).
 *
 * For g = ngram_max down to ngram_min, take the last g history tokens as the
 * key and search BACKWARD (most recent match first) for an earlier occurrence
 * of that g-gram; on a hit, propose the tokens that followed it. Longer
 * n-grams are tried first (higher precision), the backward scan biases toward
 * recent repetition, and the trailing self-match is excluded.
 *
 * @param hist       Token history, oldest first.
 * @param n_hist     Number of history tokens.
 * @param ngram_max  Longest suffix length to try (e.g. 3).
 * @param ngram_min  Shortest suffix length to try (e.g. 2, >= 1).
 * @param max_k      Draft budget.
 * @param out        Receives the draft tokens (up to max_k).
 * @return           Number of draft tokens written (0 = no match; the caller
 *                   should fall back to plain single-token decode).
 *
 * Thread-safe: yes (no state)
 * Reentrant: yes
 * Signal-safe: yes
 */
int pn_ngram_draft(const uint32_t *hist, int n_hist, int ngram_max,
                   int ngram_min, int max_k, uint32_t *out);

/* ============================================================
 * Draft throttle — the economics guard.
 *
 * On the K80 a batched verify of K+1 positions costs several plain
 * forwards, so a verify only pays when enough drafts are accepted
 * (min_accept ~ verify_cost / forward_cost). Output phases that draft
 * poorly (e.g. free-form reasoning text) must ride at plain-decode speed:
 * after miss_limit consecutive verifies accepting < min_accept drafts,
 * drafting pauses for cool_steps plain steps, then probes again. Purely a
 * WHEN-to-draft policy — accepted tokens are always the model's exact
 * greedy argmaxes, so output is identical under any throttle setting.
 * ============================================================ */

typedef struct {
    int min_accept;  /* accepted drafts below this counts as a miss */
    int miss_limit;  /* consecutive misses before cooling */
    int cool_steps;  /* plain steps to ride before probing again */
    int miss;        /* state: consecutive misses so far */
    int cool;        /* state: plain steps left in the cooldown */
} pn_spec_throttle;

/* Initialize (also resets state). Non-positive args pick the defaults
 * (min_accept 4, miss_limit 2, cool_steps 48). */
void pn_spec_throttle_init(pn_spec_throttle *t, int min_accept,
                           int miss_limit, int cool_steps);

/* 1 = drafting allowed this step; 0 = ride plain (consumes a cool step). */
int pn_spec_throttle_gate(pn_spec_throttle *t);

/* Report a verify's outcome (number of ACCEPTED DRAFTS, i.e. n_accept-1). */
void pn_spec_throttle_report(pn_spec_throttle *t, int accepted_drafts);

#endif /* PN_NGRAM_DRAFT_H */
