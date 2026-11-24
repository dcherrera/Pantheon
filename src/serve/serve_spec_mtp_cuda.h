/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_mtp_cuda.h
 * @brief The CUDA backend of the serve's single-user speculative loop (serve_generation.c):
 *        one MTP cycle per call — adaptive depth (spec_depth.h), MTP-head drafts, one verify
 *        drawn with the request's own sampler (spec_mtp_step.h).
 *
 * Exact: every emitted token is drawn from a verify row that is byte-identical to the plain
 * decode-slots step's row for that position, with the request's sampler, RNG stream and
 * repeat window, one draw per emitted token (spec_mtp.h "draw-and-compare"). The depth only
 * decides how many rows a verify carries; a cycle with no drafts is a one-row verify, so the
 * emitted stream never depends on the depth.
 *
 * The decode loop keeps its shape: after an emitted seed (not yet in the state) a cycle
 * commits the seed + accepted drafts and returns them + the next token (uncommitted); the
 * loop emits the committed ones with no forward and uses the last as the next seed — the
 * same contract as the Tenstorrent step.
 *
 * @status complete
 * @issues The MTP head keeps one session's KV: a request cannot draft while another session
 *         prefills (it then takes one-row verifies, still exact).
 * @todo None
 */
#ifndef SERVE_SPEC_MTP_CUDA_H
#define SERVE_SPEC_MTP_CUDA_H

#include "../../include/pantheon.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef USE_CUDA

#include "../core/spec_depth.h"
#include "../core/spec_mtp_couple.h"
#include "serve_spec_lookup.h"

/** One probe draft set (PN_MTP_SHADOW=1, batch_sched_spec.h): drafted beside a plain step
 *  after the token at ring index `at`, scored against the tokens plain decoding emits next. */
typedef struct {
    int at, n;
    bool masked;              /* the mask was live for the row after `at` */
    uint32_t d[PN_SPEC_DEPTH_MAX];
} sg_mtp_shadow_draft;

#define SG_MTP_SHADOW_PENDING 8

/** Per-request speculation state of the CUDA backend. */
typedef struct {
    pn_spec_depth depth;      /* adaptive draft depth (one controller per request) */
    uint32_t *recent;         /* the cycle's copy of the loop's repeat window */
    int recent_cap;           /* allocation of recent (kept across begin calls) */
    int max_recent;
    float confs[PN_SPEC_DEPTH_MAX];
    long cycles, drafted, accepted, draft_failures, grammar_cuts;
    long depth_hist[PN_SPEC_DEPTH_MAX + 1];
    double ms_draft, ms_verify;   /* wall time in drafts / verify+draw+commit */
    double ms_bind, ms_cycle;     /* batched path: slot re-bind / whole cycle incl. deposit */
    long plain_steps;             /* batched path: solo plain steps (D = 0) and their ms */
    double ms_plain;
    long plain_feed_misses;       /* plain steps whose row did not reach the head */
    pn_mtp_couple couple;         /* coupled drafts for sampled requests (spec_mtp_couple.h) */
    sg_spec_lookup *lookup;       /* prompt-lookup drafts (sg_spec_lookup_init_cuda), or NULL */
    long lookup_cycles, lookup_drafted, lookup_accepted;
    long grammar_rows, grammar_masked;   /* batched grammar draw: rows drawn, under a live mask */
    sg_mtp_shadow_draft shadow[SG_MTP_SHADOW_PENDING];   /* unscored probe drafts */
    int shadow_n;
    long shadow_hist[2][PN_SPEC_DEPTH_MAX + 1];  /* [masked][drafts a verify would accept] */
    long shadow_drafts;
    double shadow_ms;             /* probe draft time (inside the driver's sample window) */
} sg_mtp_cuda;

/**
 * Whether `model` runs the shared Qwen3.5 CUDA forward: the MoE models
 * (pn_qwen35moe_forward_cuda) and the dense-FFN ones, Ornith and Bonsai
 * (pn_bonsai_forward_cuda, which delegates to it).
 */
bool sg_qwen35_cuda_forward(const pn_model *model);

/** Whether `model` can speculate here: a CUDA qwen35 model (MoE or dense) with its MTP head
 *  uploaded. */
bool sg_mtp_cuda_available(pn_model *model);

/**
 * Arm one request for `model` (largest draft depth: QWEN35_MTP_DMAX, default 4; a dense-FFN
 * model takes the dense cost priors). `mtp` must be zeroed before its first begin; a later
 * begin reuses its window buffer (sg_mtp_cuda_end frees it).
 * @return 0, or -1 (no memory: the request decodes plain)
 */
int sg_mtp_cuda_begin(sg_mtp_cuda *mtp, const pn_model *model, int max_recent);

/** A caller's draft cut (NULL = none): how many of drafts[0..nd) the cycle may keep —
 *  e.g. the AUTO tool grammar's look-ahead, which stops before any row its mask touches. */
typedef int (*sg_mtp_cut_fn)(void *ctx, const uint32_t *drafts, int nd);

/** The controller's draft depth for the next cycle at `pos`, bounded by `cap` (0 = the
 *  speculation would not pay here: a caller with a cheaper plain step may take it). */
int sg_mtp_cuda_choose(sg_mtp_cuda *mtp, int pos, int cap);

/**
 * One cycle with `seed` (the token just emitted, not yet in the state) at state->pos, drafting
 * `depth` (from sg_mtp_cuda_choose; < 0 = choose here). `cap` bounds the drafts (token / reasoning budget; <= 0: a one-row verify). Drafts are cut
 * before any eos_id / im_end, then by `cut` (may be NULL). With mtp->lookup on, a matching
 * context suffix drafts its continuation instead (QWEN35_MTP_LOOKUP=1: in place of the head,
 * which then drafts nothing; =2: only when its first draft is the head's). Draws with `params` (its RNG advances once per emitted token)
 * over a copy of recent[0..n_recent) (recent may be mtp->recent itself, already filled).
 * emit[] receives the accepted drafts then the next token.
 * @return tokens in emit (>= 1), or -1 (model failure: the request must decode plain)
 */
int sg_mtp_cuda_cycle(sg_mtp_cuda *mtp, pn_model *model, pn_state *state, uint32_t seed,
                      int depth, int cap, int eos_id, int im_end, sg_mtp_cut_fn cut,
                      void *cut_ctx, pn_sampling_params *params, const uint32_t *recent,
                      int n_recent, uint32_t *emit);

/** CLOCK_MONOTONIC in ms (the cycle's timers). */
double sg_mtp_cuda_now_ms(void);

/** One log line for a request that speculated. */
void sg_mtp_cuda_report(const sg_mtp_cuda *mtp, int generated);

void sg_mtp_cuda_end(sg_mtp_cuda *mtp);

#endif /* USE_CUDA */

#endif /* SERVE_SPEC_MTP_CUDA_H */
