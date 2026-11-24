/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_topn_surgery.h
 * @brief Can a backend's top-N row decide a draw exactly when the client's logit SURGERY
 *        (OpenAI `logit_bias`, `presence_penalty`, `frequency_penalty`) has to be applied?
 *
 * A top-N row names the N largest RAW logits; every other token's raw logit is <= v_min,
 * the smallest of the N. The surgery is applied to the row before the draw (bias added, then
 * presence/frequency subtracted per occurrence in the repeat window). A token outside the N
 * can then reach at most
 *     limit = v_min + max(0, sum of the biases on that token)
 * as long as both penalties are >= 0 (they only lower a logit); a negative penalty can raise
 * an outside token by an amount the row cannot bound from what it shows, so it is undecidable
 * here (the caller refills the dense row).
 *
 *   GREEDY: the dense argmax is a survivor iff the best survivor AFTER the surgery is strictly
 *     above `limit` (an outside token can then neither beat nor tie it).
 *   SAMPLED with 1 <= top_k <= N: the dense top-k set lies inside the N iff the k-th largest
 *     survivor after the surgery (and the repetition penalty) is above `limit` — the same test
 *     pn_sampler_topn_row_verdict makes against v_min, with v_min replaced by `limit`.
 *
 * The survivors' values here are a DECISION copy: the draw itself still runs the sampler's own
 * surgery on the row. A relative margin of 1e-4 covers any rounding difference between the two.
 *
 * Pure C, libc + libm only; tests/test_topn_surgery.c checks it against a dense reference.
 *
 * @status Used by pn_sampler_topn_row_verdict (engine.c). `PN_TT_TOPN_SURGERY=0` keeps such
 *         requests on dense rows (pn_qwen35moe_tt_select_rows) — the behaviour before.
 *         Red `PN_TT_RED=topn_surgery_blind`: the outside limit ignores the biases.
 * @issues None
 * @todo None
 */
#ifndef PN_SAMPLER_TOPN_SURGERY_H
#define PN_SAMPLER_TOPN_SURGERY_H

#include <stdbool.h>
#include <stdint.h>

#include "../../include/pantheon.h"

/** Does this request carry client logit surgery (a bias, or a nonzero presence/frequency)? */
bool pn_topn_surgery_present(const pn_sampling_params *params);

/**
 * Apply the client surgery to the survivors' decision copy `vals[0..S)` (ids[] their token
 * ids): each bias entry for a survivor added, then frequency per occurrence and presence once
 * per token in the window.
 */
void pn_topn_surgery_apply(const int *ids, float *vals, int S, const pn_sampling_params *params,
                           const uint32_t *recent, int n_recent);

/**
 * The largest value any token OUTSIDE the survivors can reach after the surgery, given every
 * outside raw logit is <= v_min. False when no bound exists (a negative presence or frequency
 * penalty with a non-empty window).
 */
bool pn_topn_surgery_outside_limit(const int *ids, int S, float v_min,
                                   const pn_sampling_params *params, int n_recent, float *limit);

/** True when `value` beats `limit` by the decision margin (strictly, relative 1e-4). */
bool pn_topn_surgery_clears(float value, float limit);

/**
 * Does the request carry a positive logit bias (a bias that can RAISE a token)? On a SAMPLED
 * request such a bias leaves the top-k decision to the k-th survivor against v_min + bias, which
 * a top-64 row almost never clears at top_k ~40 — every row would refill, so those requests
 * keep dense rows and plain decode (measured). Greedy clears it against the best survivor and keeps the top-N rows.
 */
bool pn_topn_surgery_raising_bias(const pn_sampling_params *params);

/** Is the topn-surgery path on (`PN_TT_TOPN_SURGERY`, default 1; 0 = the old dense rows)? */
bool pn_topn_surgery_enabled(void);

#endif /* PN_SAMPLER_TOPN_SURGERY_H */
