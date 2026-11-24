/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp_step.h
 * @brief One MTP speculative cycle per call for a qwen35moe session on the CUDA path, with
 *        the same contract as the Tenstorrent draft/step pair the serve's spec loop calls
 *        (pn_qwen35moe_tt_mtp_draft, pn_qwen35moe_tt_spec_step).
 *
 * The serve keeps its decode loop: it drafts after the uncommitted seed, runs one step, emits
 * the returned tokens and uses the last one as the next seed. Exact by draw-and-compare
 * (spec_mtp.h): every verify row is drawn with the request's own sampler, RNG stream and
 * repeat window, one draw per emitted token.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#ifndef PN_SPEC_MTP_STEP_H
#define PN_SPEC_MTP_STEP_H

#include "../../include/pantheon.h"

#include <stdint.h>

#ifdef USE_CUDA

/** Largest draft count per cycle (the verify takes <= 5 rows: the seed + 4 drafts). */
#define PN_CUDA_SPEC_MAX_DRAFTS 4

/**
 * Up to k drafts for the token after `seed` (uncommitted, at state->pos) from the MTP head.
 * The head's backlog must sit at state->pos (pn_qwen35moe_mtp_push_rows after every trunk
 * batch that advanced the state).
 *
 * @return the draft count (>= 1), or -1 (k outside 1..4, no head, or backlog behind)
 */
int pn_qwen35moe_cuda_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int k,
                                uint32_t *drafts);

/**
 * One verify cycle: verify [seed, drafts...] at state->pos, draw each row with the request's
 * sampler exactly as plain decode would (top-k 40 sparse row, `params`' RNG, the repeat window
 * recent[0..*n_recent) of capacity repeat_window, one draw per emitted token), accept the
 * drafts up to the first one that differs from its row's draw, commit the seed + accepted
 * drafts (state->pos advances by that many) and feed those rows to the MTP head's backlog.
 * `emit` receives the accepted drafts then the drawn next token (uncommitted: the next call's
 * seed). n_drafts = 0 is one plain step through the verify. With params->tool_grammar set
 * (its `st` a scratch clone of the request's walk) every row is fetched dense and drawn as
 * plain grammar decoding draws it (pn_mtp_draw_rows_grammar).
 * RED arm (gates only): PN_MTP_RED=1 accepts every draft without the compare.
 *
 * @return the number of tokens in `emit` (1..n_drafts + 1), or -1 (bad args / model failure)
 */
int pn_qwen35moe_cuda_spec_step(pn_model *model, pn_state *state, uint32_t seed,
                                const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                pn_sampling_params *params, uint32_t *recent, int *n_recent,
                                int repeat_window);

struct pn_mtp_couple_s;
/** pn_qwen35moe_cuda_spec_step, with the cycle's coupled-draft state (spec_mtp_couple.h; NULL
 *  = none) for the acceptance probe. The draws are the same either way. */
int pn_qwen35moe_cuda_spec_step_ex(pn_model *model, pn_state *state, uint32_t seed,
                                   const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                   pn_sampling_params *params, uint32_t *recent, int *n_recent,
                                   int repeat_window, const struct pn_mtp_couple_s *couple);

#endif /* USE_CUDA */

#endif /* PN_SPEC_MTP_STEP_H */
