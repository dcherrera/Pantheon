/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_spec.h
 * @brief Solo-slot MTP speculation for the batched decode driver (batch_sched.c): when
 *        exactly one slot of an MTP model (serve --mtp) is decoding, the driver runs one
 *        exact MTP cycle on it (serve_spec_mtp_cuda.h) instead of a one-token decode step.
 *
 * The cycle verifies on the model's shared single-user buffers, which the forward seeds
 * device-to-device from the slot's own block (KV ring rows + SSM image); afterwards the slot
 * is re-bound (decode_slot_bind tops its block up from the shared buffers), so the block is
 * current after every cycle and a second user joining — or another session's prefill taking
 * the shared buffers — finds nothing stale. A slot that decoded plain since its last cycle
 * has its shared-buffer binding forgotten first, so the verify re-seeds from the block.
 * D = 0 (the controller expects no draft to pay) is the plain decode-slots step; its output
 * row still joins the MTP head's backlog, so the next cycle drafts from an unbroken history.
 *
 * Exact: the verify rows equal the decode-slots rows byte for byte (K=1, and per user under
 * batching), each drawn with the slot's own sampler, RNG and repeat window, one draw per
 * emitted token; drafts never cross the token budget or the reasoning-budget trip.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#ifndef BATCH_SCHED_SPEC_H
#define BATCH_SCHED_SPEC_H

#include "batch_sched.h"

#ifdef USE_CUDA

/** Whether slot `sl` may speculate this step (under sched->mu): an MTP model, in-driver
 *  sampling, no logprobs / logit bias, no force-close pending, and no grammar hook or a
 *  tool walk with clone look-ahead (arming window on) whose mask is not live for this row. */
bool bs_spec_eligible(const pn_batch_sched *sched, const bs_slot *sl);

/** Whether a slot with this tool-grammar hook speculates with the grammar draw
 *  (spec_mtp_step.h): the hook can clone its walk, has no steer, the arming window is off (so
 *  plain decode draws the slot's rows dense, as the cycle does) and PN_MTP_GRAMMAR=1 (default
 *  off).
 *  The cycle's rows are drawn over a scratch clone of the walk; the deposit advances the live
 *  walk over the emitted tokens, so the stream is plain grammar decoding's. */
bool bs_spec_grammar_draws(const pn_tool_grammar_hook *tgh);

/** sg_mtp_cut_fn for the grammar draw (ctx = the slot's hook): advance a clone of the live
 *  walk over the drafts and keep those before the first one the live mask refuses or the
 *  first one that completes the call (which may then only be the cycle's last token — the
 *  same rule as EOS, so the state never commits past the call's end). */
int bs_spec_grammar_complete_cut(void *ctx, const uint32_t *drafts, int nd);

/** sg_mtp_cut_fn for a tool grammar (AUTO or FORCED) (ctx = the slot's pn_tool_grammar_hook): advance
 *  a clone of the live walk over the drafts and keep those after which the next row is
 *  still unmasked. A row drawn outside the mask is the hook-free draw (engine.c: the mask
 *  is skipped while inactive), so the cycle draws with the hook removed and the driver
 *  advances the live walk over every emitted token itself. */
int bs_spec_grammar_cut(void *ctx, const uint32_t *drafts, int nd);

/** Why a request on an MTP model never speculates (logged once at bind), or NULL. */
const char *bs_spec_plain_reason(const pn_batch_sched *sched, const pn_sampling_params *sp);

/** After a solo slot's plain decode-slots step (GPU turn owned, sched->mu held): book its
 *  cost for the controller and push the stepped row to the MTP head's backlog. */
void bs_spec_after_plain_step(pn_batch_sched *sched, bs_slot *sl, uint32_t token, double ms);

/** PN_MTP_SHADOW=1 (measurement only): a solo grammar slot, which decodes plain, drafts
 *  after every emitted token as a cycle would, and the drafts are scored against the tokens
 *  plain decoding emits next — the acceptance an exact grammar-aware verify would see (its
 *  draws ARE plain decoding's tokens). Never set in production: each draft costs a head pass
 *  under the driver's lock. */
bool bs_spec_shadow_on(void);

/** After a solo grammar slot emitted sl->next_token on a plain step (GPU turn owned,
 *  sched->mu held): score the pending probe drafts, then draft after that token. */
void bs_spec_shadow_step(pn_batch_sched *sched, bs_slot *sl);

/** At release (sched->mu held, ring still live): score what is left and log one line. */
void bs_spec_shadow_report(bs_slot *sl, int slot);

/** QWEN35_MTP_STRESS_PLAIN=N (tests only): every Nth step of a solo MTP slot is a plain
 *  decode-slots step, so cycle <-> plain hand-overs run constantly. */
bool bs_spec_stress_plain(const bs_slot *sl);

/** A fresh request's controller on the batched path: D = 0 is the decode-slots step, priced
 *  as such (QWEN35_MTP_D0_PLAIN=0 keeps the one-row-verify pricing, for A/B). */
void bs_spec_arm_controller(bs_slot *sl);

/** The most drafts the next cycle may commit (under sched->mu): the token budget and the
 *  reasoning-budget trip land on the cycle's last (uncommitted) token at the latest. */
int bs_spec_draft_cap(const bs_slot *sl);

/** Copy the slot's repeat window into its cycle buffer and push the seed onto it, as the
 *  plain deposit does for the stepped token before it samples (under sched->mu).
 *  @return the window length */
int bs_spec_stage_window(bs_slot *sl, uint32_t seed);

/**
 * The GPU half of a cycle drafting `depth` (>= 1; GPU turn owned, sched->mu NOT held):
 * forget a stale shared binding, run the MTP cycle on the slot's state with `sp` (a copy of the slot's sampler),
 * re-bind the slot's block (row `gpu_row` of `n_slots`). emit[] receives the accepted drafts
 * then the next token. @return tokens in emit, or -1 (the slot must error out)
 */
int bs_spec_cycle_gpu(pn_batch_sched *sched, bs_slot *sl, int gpu_row, uint32_t seed,
                      int depth, int cap, pn_sampling_params *sp, int n_recent, uint32_t *emit);

#endif /* USE_CUDA */

#endif /* BATCH_SCHED_SPEC_H */
