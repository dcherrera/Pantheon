/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampling_modifiers.h
 * @brief The OpenAI sampling controls that act on a logits row: `logit_bias`,
 *        `presence_penalty` / `frequency_penalty`, the per-request `seed` RNG,
 *        and `logprobs`.
 *
 * These are pure functions over a float array plus a recent-token window — no
 * model, no tokenizer, no engine state — so they live apart from the sampler
 * that calls them (engine.c) and are unit-tested directly
 * (tests/test_sampling_modifiers.c) instead of through a full forward pass.
 * That separation is deliberate: the last bug in this area (a repetition penalty
 * applied on the wrong side of the greedy branch) was an ORDERING bug that a
 * test over the real logits row would have caught immediately.
 *
 * @status Complete: bias, presence/frequency penalties, seeded RNG and logprob
 *         extraction all implemented and tested.
 * @issues None
 * @todo None
 */

#ifndef PN_SAMPLING_MODIFIERS_H
#define PN_SAMPLING_MODIFIERS_H

#include <stdint.h>

#include "../../include/pantheon.h"   /* pn_logit_bias_entry, PN_LOGPROBS_MAX_TOP */

/**
 * Add each entry's bias to its token's logit, in place.
 *
 * Runs FIRST in the sampling pipeline — before penalties, temperature and
 * top-k/top-p — because OpenAI defines `logit_bias` as a modification of the
 * model's own logits, and because a -100 bias must be able to veto a token
 * regardless of how the rest of the pipeline would have scored it.  Entries
 * whose id is out of range are skipped.  No-op when `bias` is NULL or `n` <= 0.
 *
 * @param logits   [n_vocab] logits row, modified in place
 * @param n_vocab  vocabulary size
 * @param bias     borrowed id→bias array (may be NULL)
 * @param n        number of entries
 *
 * Thread-safe: yes (operates only on the caller's buffers)
 */
void pn_apply_logit_bias(float *logits, int n_vocab,
                         const pn_logit_bias_entry *bias, int n);

/**
 * Apply OpenAI's presence and frequency penalties over a recent-token window:
 *
 *     logit[t] -= frequency_penalty * occurrences(t in window)
 *     logit[t] -= presence_penalty  * (occurrences(t in window) > 0)
 *
 * No-op when both penalties are 0 (the default) or the window is empty, so a
 * request that does not ask for them leaves the logits untouched.
 *
 * ORDERING CONTRACT: unlike the server's default `repeat_penalty`, these are
 * only ever non-zero because a client asked, and they are a deterministic
 * function of the already-generated prefix — so the caller applies them on the
 * greedy (temperature<=0) path as well, and greedy stays reproducible.
 *
 * @param logits             [n_vocab] logits row, modified in place
 * @param n_vocab            vocabulary size
 * @param presence_penalty   OpenAI `presence_penalty`
 * @param frequency_penalty  OpenAI `frequency_penalty`
 * @param recent_tokens      borrowed window of recently emitted ids (may be NULL)
 * @param n_recent           window length
 *
 * Thread-safe: yes
 */
void pn_apply_presence_frequency_penalties(float *logits, int n_vocab,
                                           float presence_penalty,
                                           float frequency_penalty,
                                           const uint32_t *recent_tokens,
                                           int n_recent);

/**
 * Diffuse a client-supplied `seed` into a usable RNG state (splitmix64).
 *
 * The common seeds are tiny (0, 1, 42) and the xorshift64* stream mixes poorly
 * out of a low-entropy state, which would make neighbouring seeds produce
 * visibly correlated first tokens.  Never returns 0 (xorshift's fixed point).
 *
 * Thread-safe: yes (pure)
 */
uint64_t pn_seed_to_rng_state(uint64_t seed);

/**
 * Draw the next uniform value in [0, 1) from a per-request xorshift64* stream,
 * advancing `*state` in place.  A per-request stream is what makes `seed`
 * meaningful at all: the process-global rand() is shared by every connection
 * thread and by the batched driver's per-slot loop, so it can never be
 * deterministic for one request.
 *
 * Thread-safe: yes, provided each caller owns its own `state`.
 */
float pn_seeded_rng_next_float(uint64_t *state);

/**
 * Log-softmax a logits row and extract the OpenAI `logprobs` view of it.
 *
 * The caller must pass the RAW logits — the model's own distribution — not a row
 * that temperature, penalties, top-k/top-p or a grammar mask have already
 * rewritten; that is what the OpenAI field means.
 *
 * @param logits         [n_vocab] RAW logits (NOT modified)
 * @param n_vocab        vocabulary size
 * @param chosen         the token that was actually sampled
 * @param n_top          how many alternatives to report (clamped to
 *                       [0, PN_LOGPROBS_MAX_TOP])
 * @param chosen_out     receives logprob(chosen) (may be NULL)
 * @param top_ids_out    receives the top ids, descending (>= n_top capacity)
 * @param top_lp_out     receives their logprobs, descending
 * @return               how many alternatives were written
 *
 * Thread-safe: yes
 */
int pn_logprobs_extract(const float *logits, int n_vocab, uint32_t chosen,
                        int n_top, float *chosen_out,
                        uint32_t *top_ids_out, float *top_lp_out);

#endif /* PN_SAMPLING_MODIFIERS_H */
