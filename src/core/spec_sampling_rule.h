/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_sampling_rule.h
 * @brief True speculative sampling (Leviathan et al. / Chen et al. 2023) over final sampler
 *        distributions: the draft is DRAWN from a proposal q, the target accepts it with
 *        probability min(1, p(x)/q(x)), a rejection draws from normalize(max(0, p - q)).
 *        The emitted token is distributed exactly as p, whatever q is.
 *
 * @status Complete. Pure math over (id, probability) lists — no model, no card.
 * @issues None
 * @todo None
 *
 * p and q are the sampler's FINAL distributions (penalties, temperature, top-k, top-p
 * applied): p over the target row's survivors (zero for every token not listed), q over the
 * draft head's. A token q proposes outside p's list has p = 0 and is always rejected; the
 * residual lives on p's list only. A deterministic draft (n = 0: q is a point mass at
 * `token`) is the same rule with q(token) = 1 — accept with probability p(token).
 *
 * Reds (`PN_TT_RED`, or pn_spec_rule_set_red): `spec_accept_always` accepts every draft,
 * `spec_residual_p` draws a rejection's token from p instead of the residual. Both change
 * the emitted distribution; the distribution gates must catch them.
 */
#ifndef PN_SPEC_SAMPLING_RULE_H
#define PN_SPEC_SAMPLING_RULE_H

#include <stdbool.h>
#include <stdint.h>

/** Largest proposal support kept (the card's top-N ceiling). */
#define PN_SPEC_PROPOSAL_MAX 64

/** One draft and the distribution it was drawn from. */
typedef struct pn_spec_proposal {
    uint32_t token;                        /**< the draft */
    int n;                                 /**< support size; 0 = point mass at `token` */
    uint32_t ids[PN_SPEC_PROPOSAL_MAX];
    float probs[PN_SPEC_PROPOSAL_MAX];     /**< > 0, summing to `total` */
    double total;                          /**< sum of probs (1 up to rounding) */
} pn_spec_proposal;

/** A uniform draw in [0, 1). */
typedef float (*pn_spec_uniform_fn)(void *ctx);

typedef enum {
    PN_SPEC_RULE_RED_NONE = 0,
    PN_SPEC_RULE_RED_ACCEPT_ALWAYS,
    PN_SPEC_RULE_RED_RESIDUAL_P,
} pn_spec_rule_red;

/** Override the red (tests); PN_SPEC_RULE_RED_NONE also stops reading `PN_TT_RED`. */
void pn_spec_rule_set_red(pn_spec_rule_red red);

/** q(token): 0 outside the support, 1 for a point mass at it. */
double pn_spec_proposal_prob(const pn_spec_proposal *q, uint32_t token);

/**
 * Build q from a final distribution (ids[i], probs[i]), the `excluded` tokens removed and the
 * rest renormalised, and draw the draft from it with uniform `u`.
 * @return 0, or -1 when nothing is left (q untouched)
 */
int pn_spec_proposal_build(pn_spec_proposal *q, const int *ids, const float *probs, int n,
                           const uint32_t *excluded, int n_excluded, float u);

/** A draw from q with uniform `u` (the same scan pn_spec_proposal_build draws with). */
uint32_t pn_spec_proposal_sample(const pn_spec_proposal *q, float u);

/**
 * The verify of draft q->token against the target's final distribution p (p_ids, p_probs,
 * S entries, zero elsewhere). Takes one uniform for the accept test and, on a rejection, one
 * for the residual draw. @return the emitted token (q->token iff accepted)
 */
uint32_t pn_spec_rule_verify(const int *p_ids, const float *p_probs, int S,
                             const pn_spec_proposal *q, pn_spec_uniform_fn uniform,
                             void *uniform_ctx, bool *accepted);

/** Sum over tokens of min(p, q): the probability a draft from q is accepted. */
double pn_spec_rule_acceptance(const int *p_ids, const float *p_probs, int S,
                               const pn_spec_proposal *q);

/**
 * Total variation distance between p and the distribution the rule EMITS when the draft is
 * drawn from q (computed from the rule's own accept probability and residual, so a broken
 * rule shows here): 0 up to float rounding for a correct rule.
 */
double pn_spec_rule_emitted_tvd(const int *p_ids, const float *p_probs, int S,
                                const pn_spec_proposal *q);

#endif /* PN_SPEC_SAMPLING_RULE_H */
