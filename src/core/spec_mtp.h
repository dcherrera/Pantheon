/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp.h
 * @brief MTP speculative generation for one qwen35moe session on the CUDA path: MTP-head
 *        drafts, one batched verify per cycle, every row sampled exactly as plain decode
 *        samples it (draw-and-compare), depth from the adaptive controller (spec_depth.h).
 *
 * EXACT: each verify row's top-k pairs equal the plain decode step's for that position
 * (pn_qwen35moe_spec_verify_topk), the row is presented the same way (-inf outside the
 * pairs) and drawn with the same sampler, RNG stream and repeat window; a draft only counts
 * when the drawn token equals it, and exactly one draw is made per emitted token. So the
 * emitted stream equals plain decode's (greedy or sampled with a seeded RNG) for ANY draft
 * depth. When the controller finds plain decode faster from here on it binds the session to
 * decode slot 0 and finishes on the plain step (the serve's decode path).
 *
 * @status complete
 * @issues Bench driver only: not wired into serve.
 * @todo None
 */
#ifndef PN_SPEC_MTP_H
#define PN_SPEC_MTP_H

#include "../../include/pantheon.h"
#include "spec_depth.h"

#include <stdint.h>

typedef struct {
    int fixed_d;          /* >= 0: always draft this many (no controller); -1: adaptive */
    int dmax;             /* adaptive: largest D */
    int plain_switch;     /* 1: allow the one-way switch to plain decode */
    int use_more;         /* 1: per-draft continue decision from the head's confidence */
    int topk;             /* pairs per verify row: must equal the plain step's (40) */
    int repeat_window;    /* the repeat-penalty ring length the plain arm keeps */
    int couple;           /* 1: coupled drafts for sampled draws (spec_mtp_couple.h) */
    /* Prompt lookup (lookup_draft NULL = off): up to 4 drafts from the context's recurring
     * suffix — mode 1 in place of the head (which then drafts nothing), mode 2 only when the
     * first agrees with the head's first. The caller keeps the history: push gets every
     * emitted token, report every lookup cycle's outcome. */
    int (*lookup_draft)(void *ctx, int cap, uint32_t *drafts);
    void (*lookup_report)(void *ctx, int drafted, int accepted);
    void (*lookup_push)(void *ctx, uint32_t token);
    void *lookup_ctx;
    int lookup_mode;
} pn_mtp_gen_opts;

typedef struct {
    long cycles;                          /* verifies run */
    long drafted, accepted, emitted;
    long depth_hist[PN_SPEC_DEPTH_MAX + 1];
    long plain_steps;                     /* steps after the switch */
    long lookup_cycles, lookup_accepted;  /* cycles the lookup drafted, its drafts kept */
    int switched_at;                      /* position of the switch, -1 = never */
    double ms_draft, ms_verify, ms_plain, ms_sample;
} pn_mtp_gen_stats;

/**
 * Generate after a prefill: `state` sits at the seed's position (seed = the first sampled
 * token, not yet in the state) and the MTP head's backlog holds the prompt's rows
 * (pn_qwen35moe_mtp_push_rows after each prefill chunk). Draws with `params` (its seeded
 * RNG advances once per emitted token), keeps recent[0..*n_recent) as the repeat window,
 * stops at stop_id (not emitted) or max_tokens. out[] receives the generated tokens after
 * the seed (the seed is NOT written). `ctl` may be NULL when fixed_d >= 0.
 *
 * @return tokens written to out, or -1 on a model/path failure
 */
int pn_mtp_spec_generate(pn_model *model, pn_state *state, uint32_t seed, int max_tokens,
                         int stop_id, pn_sampling_params *params, uint32_t *recent,
                         int *n_recent, const pn_mtp_gen_opts *opts, pn_spec_depth *ctl,
                         uint32_t *out, pn_mtp_gen_stats *stats);

/** Draw a verify row (top-k pairs presented, -inf elsewhere) exactly as plain decode would:
 *  pn_sample_next_token_topn (the sparse-survivor path, the same token), or
 *  pn_sample_next_token when QWEN35_SPEC_SPARSE=0. */
uint32_t pn_mtp_draw_row(float *row, int n_vocab, pn_sampling_params *params,
                         uint32_t *recent, int n_recent);

/** Fetch verify row `row`'s DENSE logits (n_vocab floats) into out; 0, or -1. */
typedef int (*pn_mtp_row_fetch_fn)(void *ctx, int row, float *out);

/**
 * Draw verify rows under a tool grammar exactly as plain grammar decoding draws them. Row r
 * (r = 0..n_drafts, toks[0] the seed, toks[r + 1] the draft row r is compared with) is fetched
 * DENSE — plain grammar decoding's full-logits row for that position — into `row` and drawn
 * with pn_sample_next_token over `params`. params->tool_grammar must point at a hook whose `st`
 * is a scratch clone of the live walk: each draw refreshes, masks and advances it, so row r + 1
 * is masked by the state the accepted prefix reached, and a draft the grammar forbids cannot
 * equal its row's draw. Stops after the first draw that differs from its draft, or row
 * n_drafts. One draw per emitted token; the repeat window recent[0..*n_recent) (capacity
 * repeat_window) takes every draw, as plain decode's does.
 * RED (gates only), PN_MTP_RED: "1" emits each draft as accepted (rows still drawn);
 * "grammar_live_walk" draws with the hook's advance removed (every row masked by the walk as
 * the cycle found it). Either must break the identity with plain grammar decoding.
 * @return draws written to emit (1..n_drafts + 1), or -1 (a fetch failed / bad args)
 */
int pn_mtp_draw_rows_grammar(pn_mtp_row_fetch_fn fetch, void *fetch_ctx, const uint32_t *toks,
                             int n_drafts, float *row, int n_vocab, pn_sampling_params *params,
                             uint32_t *recent, int *n_recent, int repeat_window, uint32_t *emit);

/** Append tok to the repeat window recent[0..*n) of capacity cap (oldest drops out). */
void pn_mtp_recent_push(uint32_t *recent, int *n, int cap, uint32_t tok);

/** Present (ids, vals)[0..k) into row (-inf elsewhere; the previous ids prev[0..*n_prev)
 *  are cleared first) — the plain decode step's sparse row, byte for byte. */
void pn_mtp_present_row(float *row, int *prev, int *n_prev, const int *ids, const float *vals,
                        int k);

#endif /* PN_SPEC_MTP_H */
