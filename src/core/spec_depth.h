/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_depth.h
 * @brief Adaptive speculative draft depth: picks D (0 = no drafts) per verify from a cost
 *        model of measured verify/draft costs at the current position and the running
 *        acceptance, so speculation is only run where it is expected to pay.
 *
 * The choice only decides HOW MANY rows the next verify carries; the verify decides every
 * emitted token, so no setting here can change the output.
 *
 * Cost model (ms):  verify(n rows, pos) = (c[n] + s[n] * pos) * ratio
 *                   plain(pos)          = (plain_c + plain_s * pos) * plain_ratio
 *                   draft(D)            = draft_first + (D - 1) * draft_next
 *                   cycle(D)            = draft(D) + verify(D + 1, pos) + extra
 * c/s are priors (measured on the K80, overridable with QWEN35_SPEC_COST); ratio and the
 * draft costs are running averages of what the loop observes. Expected tokens from D drafts
 * with per-depth conditional acceptances a_i: 1 + sum_{i<=D} prod_{j<=i} a_j.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#ifndef PN_SPEC_DEPTH_H
#define PN_SPEC_DEPTH_H

#define PN_SPEC_DEPTH_MAX 4   /* drafts; a verify carries at most 5 rows (the rows kernel) */

typedef struct {
    int dmax;                               /* largest D considered (<= PN_SPEC_DEPTH_MAX) */
    double verify_c[PN_SPEC_DEPTH_MAX + 2]; /* [n] fixed ms of an n-row verify+commit */
    double verify_s[PN_SPEC_DEPTH_MAX + 2]; /* [n] ms per context position */
    double ratio;                           /* running observed / modelled verify cost */
    double plain_c, plain_s;                /* the plain decode step */
    double draft_first, draft_next;         /* ms: first draft (with catch-up), each next */
    double accept[PN_SPEC_DEPTH_MAX + 1];   /* [i] P(draft i accepted | draft i-1 was) */
    double conf_cal;                        /* running accepted / sum(confidence) */
    double alpha;                           /* running-average weight */
    long d0_run;                            /* consecutive D = 0 choices */
    int d0_plain;                           /* D = 0 is the caller's plain step, not a
                                               one-row verify (pn_spec_depth_d0_is_plain) */
    double plain_ratio;                     /* running observed / modelled plain step */
    long plain_seen;                        /* plain steps observed (0 with d0_plain: the
                                               plain step follows ratio — heat slows both) */
    double extra_ms;                        /* running per-cycle cost outside draft + verify */
    int env_costs;                          /* QWEN35_SPEC_COST set every prior */
    int dense;                              /* dense-FFN priors (pn_spec_depth_dense_priors) */
} pn_spec_depth;

/**
 * Priors: the K80 costs measured for the batched verify (or QWEN35_SPEC_COST =
 * "c1,s1;c2,s2;...;c5,s5;plain_c,plain_s;draft_first,draft_next") and Gate 1's
 * acceptance (temp 0.7). dmax is clamped to [0, PN_SPEC_DEPTH_MAX].
 */
void pn_spec_depth_init(pn_spec_depth *c, int dmax);

/**
 * D = 0 means the caller's plain decode step (priced by the plain model, observed with
 * pn_spec_depth_observe_plain), not a one-row verify: the batched serve, whose solo slot
 * takes the decode-slots step when no draft pays. The plain prior becomes the measured
 * decode-slots step (unless QWEN35_SPEC_COST set it).
 */
void pn_spec_depth_d0_is_plain(pn_spec_depth *c);

/**
 * Priors for a dense-FFN model (Qwen3.8 measured on 4 K80 dies) instead of the MoE ones: a
 * dense Q8 verify streams the same weights for every row, so extra rows cost little until
 * the 5-row kernel (Qwen3.8: 1 / 4 / 5 rows = 260 / 354 / 572 ms), where the MoE verify
 * pays for each row's experts. The running ratio scales them to a smaller dense model. Call
 * after pn_spec_depth_init (before or after pn_spec_depth_d0_is_plain); QWEN35_SPEC_COST
 * still wins.
 */
void pn_spec_depth_dense_priors(pn_spec_depth *c);

/** Expected tokens per ms when the next verify drafts D tokens at position pos. */
double pn_spec_depth_rate(const pn_spec_depth *c, int d, int pos);

/** The plain step's tokens per ms at pos. */
double pn_spec_depth_plain_rate(const pn_spec_depth *c, int pos);

/** The D in [0, dmax] with the best expected rate at pos (ties: the smaller D). */
int pn_spec_depth_choose(pn_spec_depth *c, int pos);

/**
 * After `drafted` drafts with confidences confs[0..drafted), whether drafting on is expected
 * to raise the rate: stopping is compared with every deeper D up to dmax (draft and verify
 * cost against the chance the drafts land), not only the next one, since a costly next row
 * can be followed by a cheap one.
 */
int pn_spec_depth_more(const pn_spec_depth *c, int pos, int drafted, const float *confs);

/**
 * Whether the plain step beats every D even at an optimistic acceptance (the larger of the
 * running value and `optimistic`, per depth): speculation cannot pay from pos on, since
 * every verify row grows with position at least as fast as the plain step.
 */
int pn_spec_depth_plain_wins(const pn_spec_depth *c, int pos, double optimistic);

/**
 * The one-way switch to plain decode for a loop whose D = 0 is a one-row verify: plain wins
 * at an optimistic 0.9 acceptance, or a run of 16+ D = 0 choices finds the one-row verify
 * 3% slower than the plain step.
 */
int pn_spec_depth_switch_plain(const pn_spec_depth *c, int pos);

/** Fold one verify (n rows, at pos, wall ms including commit) into the running ratio. */
void pn_spec_depth_observe_verify(pn_spec_depth *c, int n, int pos, double ms);

/** Fold one plain decode step (at pos, wall ms) into the plain ratio. */
void pn_spec_depth_observe_plain(pn_spec_depth *c, int pos, double ms);

/** Fold one cycle's cost outside its draft and verify (e.g. a slot re-bind, ms). */
void pn_spec_depth_observe_extra(pn_spec_depth *c, double ms);

/** Fold one draft call (n drafts, wall ms) into the draft costs. */
void pn_spec_depth_observe_draft(pn_spec_depth *c, int n, double ms);

/** Fold one verify's outcome: `drafted` drafts, the first `accepted` landed (depths past
 *  `drafted` relax toward the prior). */
void pn_spec_depth_observe_accept(pn_spec_depth *c, int drafted, int accepted,
                                  const float *confs);

#endif /* PN_SPEC_DEPTH_H */
