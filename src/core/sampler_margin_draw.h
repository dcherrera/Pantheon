/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_margin_draw.h
 * @brief The dense full-path draw (temperature -> full-vocabulary softmax -> top-k renormalised
 *        -> top-p -> renormalised -> cumulative scan) decided from the row's top candidates,
 *        with the SAME token, or refused so the caller runs the dense path.
 *
 * WHY. On a 248k-vocabulary row the dense path spends ~4-5 ms of a 2-core host per sampled
 * token (an expf and a dozen full-row passes), and inside a tool-call turn every unmasked row
 * takes it. The token it returns depends, in exact arithmetic, only on the top-k logits: the
 * full-vocabulary normaliser cancels when top-k renormalises. In float it also depends on
 * rounding, but only when a decision sits within rounding error of a boundary.
 *
 * HOW. One pass picks the top PN_MARGIN_CAND raw logits. The plan computes the draw's
 * distribution from them in double precision, as exact reals, and keeps every decision the
 * dense path makes as a margin: the top-k edge and the top-p edge as log-ratio gaps between
 * neighbouring logits (PN_MARGIN_MU), the top-p stop and the final scan as distances from the
 * cumulative boundaries (PN_MARGIN_BETA). The dense path's float results differ from those
 * reals by at most ~(4E + (4k + 10)u) (E = expf's relative error, u = 2^-24, k = top_k; the
 * full normaliser's own rounding cancels), ~1e-5 for k = 20 even with E = 16u, so a draw
 * clear of every boundary by BETA = 1e-4 is the dense path's draw. Anything closer is
 * refused and the caller runs the dense path with the same uniform: identical tokens and an
 * identical RNG stream, never a different token.
 *
 * Callers keep the dense path for everything the plan does not model: top_k < 1 or above
 * PN_MARGIN_CAND - 2, temperature <= 0, a row with fewer finite entries than candidates or
 * any NaN / +inf.
 *
 * Used by sample_token_core (engine.c) for every row that reaches the dense path; off with
 * PN_MARGIN_DRAW=0. Gates: tests/test_sampler_margin_draw.c (the margin draw against the dense
 * draw, and the sampler with it on against off, token and RNG state per uniform); red arms
 * PN_MARGIN_RED=1 (margins 0) and PN_MARGIN_RED=2 (a wrong decided token, for serve gates).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#ifndef PN_SAMPLER_MARGIN_DRAW_H
#define PN_SAMPLER_MARGIN_DRAW_H

#include <stdint.h>

#define PN_MARGIN_CAND 64          /* top raw logits the plan reads */
#define PN_MARGIN_MU   1e-4        /* log-ratio gap required at the top-k / top-p edges */
#define PN_MARGIN_BETA 1e-4        /* cumulative-mass distance required at every boundary */

typedef struct {
    int      n;                    /* tokens the final scan crosses (P) */
    uint32_t ids[PN_MARGIN_CAND];  /* P in ascending id order (the dense scan's order) */
    double   cum[PN_MARGIN_CAND];  /* exact-real cumulative mass after ids[j] */
} pn_margin_plan;

/** Reasons a plan was refused (pn_margin_plan_dense's return when not 1). */
enum {
    PN_MARGIN_OK = 1,
    PN_MARGIN_INELIGIBLE = 0,      /* parameters or row outside the model */
    PN_MARGIN_EDGE_TOPK = -1,      /* top-k edge within MU */
    PN_MARGIN_EDGE_TOPP = -2,      /* top-p stop within BETA, or its edge within MU */
    PN_MARGIN_RANGE = -3,          /* kept logits too far below the max (subnormal risk) */
};

/**
 * Plan the draw over `logits` (the row exactly as the dense path would receive it: bias and
 * penalties applied, temperature not). Reads the row only. Returns PN_MARGIN_OK with `plan`
 * filled, else a refusal code (nothing drawn: the caller runs the dense path as usual).
 */
int pn_margin_plan_dense(const float *logits, int n_vocab, float temperature, int top_k,
                         float top_p, pn_margin_plan *plan);

/**
 * With the uniform `u` the dense path draws: 1 and *token when u is at least BETA from every
 * cumulative boundary of the plan, else 0 (the caller runs the dense path with this same u).
 */
int pn_margin_decide(const pn_margin_plan *plan, float u, uint32_t *token);

/** Plan + decide in one call: 1 and *token when the plan decides the draw with `u`, else 0
 *  (the caller runs the dense path with this same u). */
int pn_margin_draw(const float *logits, int n_vocab, float temperature, int top_k, float top_p,
                   float u, uint32_t *token);

/** PN_MARGIN_DRAW=0 turns the draw off in the sampler (default on). */
int pn_margin_draw_enabled(void);

/** Tests: 1 on, 0 off, -1 back to the environment. */
void pn_margin_draw_set_enabled(int on);

/** Top `c` (c <= PN_MARGIN_CAND) finite entries of `row` by value, value descending, id
 *  ascending on ties; returns how many were found, or -1 when the row holds a NaN or +inf.
 *  Exposed for tests. */
int pn_margin_select_top(const float *row, int n, int c, uint32_t *ids, float *vals);

#endif /* PN_SAMPLER_MARGIN_DRAW_H */
