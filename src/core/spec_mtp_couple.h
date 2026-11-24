/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp_couple.h
 * @brief Coupled MTP drafts for sampled requests: each draft is drawn from the head's
 *        distribution with the SAME uniform its verify row will be drawn with.
 *
 * Draw-and-compare (spec_mtp.h) accepts draft i iff verify row i's draw equals it, and row i
 * draws with the request's next uniform u_i (one uniform per emitted token, the plain
 * sampler's ascending-id cumulative scan). The argmax draft is accepted with p(argmax q).
 * Peeking u_i (a copy of the request's stream, never advanced) and drafting q's own scan at
 * u_i accepts whenever the two scans land on the same token: p = q accepts every draft, and
 * any overlap of the two distributions' CDF intervals counts, not just one token's mass.
 *
 * EXACT by construction: drafts only propose; every emitted token is still the plain
 * sampler's draw from the request's own stream, so the output is byte-identical to plain
 * decode (and to argmax drafts) for any seed. A request without an OpenAI seed (the
 * process-global rand()) cannot be peeked: its first coupled cycle gives it a private stream
 * seeded from rand() — the same distribution, now peekable.
 *
 * q is the request's sampler (penalties over the window the row will see, temperature,
 * top-k, top-p) over the head's top PN_MTP_PICK_PAIRS pairs. Any q is exact; this one is
 * close to p. Greedy requests, a grammar hook or top-N refills keep the argmax draft.
 *
 * Env: QWEN35_MTP_COUPLE=0 turns coupling off (argmax drafts). PN_MTP_PROBE=1 books, per
 * draft position, the acceptance of the argmax draft p(argmax q), of rejection sampling
 * sum min(p, q), and of the coupled draft (CDF-interval overlap), on every verify row a
 * coupled cycle reaches, and prints them at exit (diagnostics only). RED (gates only):
 * PN_MTP_RED=couple_consume peeks on the request's own stream (advancing it), which breaks
 * the byte-identity; tests/test_spec_mtp_couple --selftest must fail on it.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#ifndef PN_SPEC_MTP_COUPLE_H
#define PN_SPEC_MTP_COUPLE_H

#include "../../include/pantheon.h"

#include <stdbool.h>
#include <stdint.h>

/** Drafts a cycle can couple (the verify's 4 drafts, with room). */
#define PN_MTP_COUPLE_MAX 8
/** Pairs kept per draft (= the head's pick pairs). */
#define PN_MTP_COUPLE_PAIRS 20
/** Largest repeat window a coupled cycle tracks (larger windows keep argmax drafts). */
#define PN_MTP_COUPLE_WINDOW 256

typedef struct pn_mtp_couple_s {
    pn_sampling_params *params;     /* the request's (peeked; seeded once if unseeded) */
    const uint32_t *recent;         /* the window row 0 draws under */
    int n_recent, max_recent, n_vocab;
    bool on;                        /* this cycle's drafts are coupled */
    int n_u;
    float u[PN_MTP_COUPLE_MAX];     /* the uniforms rows 0.. will draw with */
    int n_drafts;
    uint32_t drafts[PN_MTP_COUPLE_MAX];
    int n_pairs[PN_MTP_COUPLE_MAX]; /* the head's pairs per draft (the probe's q) */
    uint32_t ids[PN_MTP_COUPLE_MAX][PN_MTP_COUPLE_PAIRS];
    float vals[PN_MTP_COUPLE_MAX][PN_MTP_COUPLE_PAIRS];
    uint32_t window[PN_MTP_COUPLE_WINDOW];
    long cycles, coupled_drafts, argmax_drafts;
} pn_mtp_couple;

/** QWEN35_MTP_COUPLE (default on). */
bool pn_mtp_couple_enabled(void);

/**
 * Arm one cycle of up to k drafts: peek the k uniforms rows 0..k-1 will draw with. Off
 * (argmax drafts) for greedy, a grammar hook, top-N refills, a window over
 * PN_MTP_COUPLE_WINDOW or QWEN35_MTP_COUPLE=0. An unseeded request is seeded here once.
 * @return true when the cycle is coupled
 */
bool pn_mtp_couple_begin(pn_mtp_couple *c, pn_sampling_params *params, const uint32_t *recent,
                         int n_recent, int max_recent, int n_vocab, int k);

/** A pn_mtp_pick_fn: draft `i` from the head's pairs — q's scan at u_i when coupled, else the
 *  first pair (the argmax). Records the pairs and the draft. */
uint32_t pn_mtp_couple_pick(void *ctx, int i, const int *ids, const float *vals, int n);

/** PN_MTP_PROBE=1? */
bool pn_mtp_couple_probe_on(void);

/**
 * Probe one reached verify row `row` (< the cycle's drafts): p from the row's top-k pairs
 * under `window` (the window the row draws with), q from draft `row`'s recorded pairs;
 * `accepted` = the live outcome. No-op unless the probe is on and the pairs were recorded
 * (the drafts went through pn_mtp_couple_pick).
 */
void pn_mtp_couple_probe_row(const pn_mtp_couple *c, int row, const int *p_ids,
                             const float *p_vals, int k, const uint32_t *window, int n_window,
                             bool accepted);

/** Acceptance of the coupled draft for one row: the overlap of p's and q's CDF intervals in
 *  ascending-id order (both (id, prob) lists ascending by id). Exposed for the tests. */
double pn_mtp_couple_overlap(const int *p_ids, const float *p_probs, int Sp, const int *q_ids,
                             const float *q_probs, int Sq);

#endif /* PN_SPEC_MTP_COUPLE_H */
