/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_sampling_chooser.h
 * @brief True speculative sampling on a pn_spec_sample_ctx: the MTP drafts are DRAWN from q
 *        (the draft head's top-N row through the request's own sampler transform) and each
 *        verify row is decided by min(1, p/q) + residual (spec_sampling_rule.h).
 *
 * @status Complete. CPU code; the card hands it (index, value) rows.
 * @issues None
 * @todo None
 *
 * q is the sampler's final distribution (penalties over the draft's own repeat window,
 * temperature, top-k renormalised, top-p) over the draft head's top-N, stop tokens removed
 * (a cycle never drafts a stop: cutting a DRAWN draft by its value would bias the output).
 * Any q is exact; this one is close to p, so acceptance is sum min(p, q).
 * p is the plain sampler's final distribution over the verify row's top-N, used only when the
 * top-N decides it exactly (pn_sampler_topn_row_exact); otherwise the row falls back to the
 * plain draw (accept iff it equals the draft), which is exact for any draft.
 * One RNG stream (the request's params): a uniform per draft, one per accept test, one per
 * residual — the same seed gives the same output; the output is not plain decode's.
 *
 * `PN_SPEC_SAMPLE_DIST_CHECK=M` (gates): every rule row is re-drawn M times (draft from q,
 * then the rule) on a private RNG; chi-square of the emitted histogram against p, a
 * two-sample chi-square against M draws of the plain sampler's scan, and the analytic
 * emitted-vs-p TVD — reported as `SPEC SAMPLE DIST —` lines.
 */
#ifndef PN_SPEC_SAMPLING_CHOOSER_H
#define PN_SPEC_SAMPLING_CHOOSER_H

#include <stdbool.h>
#include <stdint.h>

#include "engine.h"
#include "spec_sampling_rule.h"

/** q's temperature as a multiple of the request's (`PN_TT_SPEC_Q_TEMP` overrides): 0.7 was
 *  the best of 1 / 0.7 / 0.5 / 0.3 / 0.15 at all three probed settings. */
#define PN_SPEC_Q_TEMP_DEFAULT 0.7f
/** The lowest request temperature `PN_TT_SPEC_SAMPLE_RULE=2` turns the rule on for. */
#define PN_SPEC_RULE_MIN_TEMP 0.9f
/** Proposals a cycle can hold (the verify's 15 drafts + the bonus row). */
#define PN_SPEC_SAMPLE_MAX_DRAFTS 16

/** Does this request take the rule? OFF by default (`PN_TT_SPEC_SAMPLE_RULE` unset or 0).
 *  Sampled with 1 <= top_k <= 64 (the top-N rows then give p exactly; a row that does not
 *  falls back per row); `=2` only at temperature >= 0.9: on the same rows
 *  the MTP head's q accepts sum min(p,q) 0.622 vs the argmax draft's p(argmax) 0.557 at 1.0,
 *  but 0.620 vs 0.617 at 0.8 and 0.641 vs 0.649 at Ornith's 0.6 (`PN_SPEC_SAMPLE_Q_PROBE`).
 *  Other requests keep argmax drafts + the shared-randomness rule (byte-identical to plain
 *  sampling). `=1`: at any temperature. */
bool pn_spec_rule_applies(const pn_sampling_params *params);

/** Allocate c->q and c->draft_recent (c->max_recent set). @return 0, or -1 */
int pn_spec_sample_rule_arm(pn_spec_sample_ctx *c);
void pn_spec_sample_rule_free(pn_spec_sample_ctx *c);

/** Before a cycle's drafts: the window the first draft sees is c->recent[0..n_recent). */
void pn_spec_sample_begin_drafts(pn_spec_sample_ctx *c);

/**
 * The MTP head's draft `draft_index` (a pn_spec_draft_fn): q from its top-N (index, value),
 * the draft drawn from it with one uniform from the request's stream. A q left empty (every
 * candidate a stop token) becomes a point mass at the best pair. @return the draft
 */
uint32_t pn_spec_sample_draft(void *ctx, int draft_index, const uint32_t *index,
                              const float *value, uint32_t count);

/** Drafts from..n-1 replaced by deterministic ones (prompt lookup): point masses. */
void pn_spec_sample_point_drafts(pn_spec_sample_ctx *c, int from, const uint32_t *drafts, int n);

/** After the cycle's drafts are cut to `n_drafts`: rows past them are never draft rows. */
void pn_spec_sample_cut_drafts(pn_spec_sample_ctx *c, int n_drafts);

/**
 * Verify row `row` (< c->n_q) with its logits already in c->row: when p is exact on the top-N,
 * decide it by min(1, p/q) + residual and return true with *token; else false (row untouched).
 */
bool pn_spec_rule_choose_row(pn_spec_sample_ctx *c, int row, uint32_t *token);

/** One `[spec sampling]` line: drafts, rule rows, accepted, mean sum min(p, q), fallbacks. */
void pn_spec_rule_report(const pn_spec_sample_ctx *c, const char *who);

#endif /* PN_SPEC_SAMPLING_CHOOSER_H */
