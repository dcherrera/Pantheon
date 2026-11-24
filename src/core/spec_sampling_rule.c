/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_sampling_rule.c
 * @brief Implementation of true speculative sampling's accept / residual rule
 *        (spec_sampling_rule.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#include "spec_sampling_rule.h"

#include <stdlib.h>
#include <string.h>

static int g_red = -1;   /* -1: read PN_TT_RED on first use */

void pn_spec_rule_set_red(pn_spec_rule_red red) { g_red = (int)red; }

static pn_spec_rule_red spec_rule_red(void)
{
    if (g_red < 0) {
        const char *env = getenv("PN_TT_RED");

        g_red = PN_SPEC_RULE_RED_NONE;
        if (env && strcmp(env, "spec_accept_always") == 0)
            g_red = PN_SPEC_RULE_RED_ACCEPT_ALWAYS;
        else if (env && strcmp(env, "spec_residual_p") == 0)
            g_red = PN_SPEC_RULE_RED_RESIDUAL_P;
    }
    return (pn_spec_rule_red)g_red;
}

double pn_spec_proposal_prob(const pn_spec_proposal *q, uint32_t token)
{
    int i;

    if (q->n == 0)
        return token == q->token ? 1.0 : 0.0;
    for (i = 0; i < q->n; i++)
        if (q->ids[i] == token)
            return (double)q->probs[i] / q->total;
    return 0.0;
}

static double spec_p_of(const int *p_ids, const float *p_probs, int S, uint32_t token)
{
    int i;

    for (i = 0; i < S; i++)
        if ((uint32_t)p_ids[i] == token)
            return (double)p_probs[i];
    return 0.0;
}

/* min(1, p/q): the accept probability of a draft with these two masses. */
static double spec_accept_prob(double p, double q)
{
    if (spec_rule_red() == PN_SPEC_RULE_RED_ACCEPT_ALWAYS)
        return 1.0;
    if (q <= 0.0)
        return 0.0;          /* never proposed: no accept event to price */
    return p >= q ? 1.0 : p / q;
}

/* The residual's weight for p's entry i: max(0, p - q). Red `spec_residual_p`: p. */
static double spec_residual_weight(const int *p_ids, const float *p_probs, int i,
                                   const pn_spec_proposal *q)
{
    const double w = (double)p_probs[i] - pn_spec_proposal_prob(q, (uint32_t)p_ids[i]);

    if (spec_rule_red() == PN_SPEC_RULE_RED_RESIDUAL_P)
        return (double)p_probs[i];
    return w > 0.0 ? w : 0.0;
}

uint32_t pn_spec_proposal_sample(const pn_spec_proposal *q, float u)
{
    const double target = (double)u * q->total;
    double cum = 0.0;
    int i;

    if (q->n == 0)
        return q->token;
    for (i = 0; i < q->n; i++) {
        cum += (double)q->probs[i];
        if (target < cum)
            return q->ids[i];
    }
    return q->ids[q->n - 1];
}

int pn_spec_proposal_build(pn_spec_proposal *q, const int *ids, const float *probs, int n,
                           const uint32_t *excluded, int n_excluded, float u)
{
    int i, j, kept = 0;
    double total = 0.0;

    for (i = 0; i < n && kept < PN_SPEC_PROPOSAL_MAX; i++) {
        bool skip = !(probs[i] > 0.0f);

        for (j = 0; !skip && j < n_excluded; j++)
            skip = (uint32_t)ids[i] == excluded[j];
        if (skip)
            continue;
        q->ids[kept] = (uint32_t)ids[i];
        q->probs[kept] = probs[i];
        total += (double)probs[i];
        kept++;
    }
    if (kept == 0 || !(total > 0.0))
        return -1;
    q->n = kept;
    q->total = total;
    q->token = pn_spec_proposal_sample(q, u);
    return 0;
}

/* A draw from the residual max(0, p - q) (red: from p) with uniform u; p itself when the
 * residual has no mass (p == q to rounding: a rejection is then a rounding event). */
static uint32_t spec_residual_draw(const int *p_ids, const float *p_probs, int S,
                                   const pn_spec_proposal *q, float u)
{
    double w[PN_SPEC_PROPOSAL_MAX * 4];
    double total = 0.0, target, cum = 0.0;
    int i, last = -1;

    if (S > (int)(sizeof(w) / sizeof(w[0])))
        S = (int)(sizeof(w) / sizeof(w[0]));
    for (i = 0; i < S; i++) {
        w[i] = spec_residual_weight(p_ids, p_probs, i, q);
        total += w[i];
    }
    if (!(total > 0.0)) {
        for (i = 0; i < S; i++) {
            w[i] = (double)p_probs[i];
            total += w[i];
        }
    }
    target = (double)u * total;
    for (i = 0; i < S; i++) {
        if (w[i] <= 0.0)
            continue;
        last = i;
        cum += w[i];
        if (target < cum)
            return (uint32_t)p_ids[i];
    }
    return last >= 0 ? (uint32_t)p_ids[last] : q->token;
}

uint32_t pn_spec_rule_verify(const int *p_ids, const float *p_probs, int S,
                             const pn_spec_proposal *q, pn_spec_uniform_fn uniform,
                             void *uniform_ctx, bool *accepted)
{
    const double p = spec_p_of(p_ids, p_probs, S, q->token);
    const double a = spec_accept_prob(p, pn_spec_proposal_prob(q, q->token));
    const float u = uniform(uniform_ctx);

    if ((double)u < a) {
        if (accepted) *accepted = true;
        return q->token;
    }
    if (accepted) *accepted = false;
    return spec_residual_draw(p_ids, p_probs, S, q, uniform(uniform_ctx));
}

double pn_spec_rule_acceptance(const int *p_ids, const float *p_probs, int S,
                               const pn_spec_proposal *q)
{
    double sum = 0.0;
    int i;

    for (i = 0; i < S; i++) {
        const double p = (double)p_probs[i];
        const double qq = pn_spec_proposal_prob(q, (uint32_t)p_ids[i]);

        sum += p < qq ? p : qq;
    }
    return sum;
}

/* The emitted mass of every token: q(y) a(y) on the accept branch, plus the rejection mass
 * spread over the residual. Tokens q proposes outside p's list are accepted with a(y) =
 * min(1, 0 / q) = 0 — unless the rule is broken, which is why they are scored too. */
double pn_spec_rule_emitted_tvd(const int *p_ids, const float *p_probs, int S,
                                const pn_spec_proposal *q)
{
    double emit[PN_SPEC_PROPOSAL_MAX * 4] = { 0 };
    double w[PN_SPEC_PROPOSAL_MAX * 4];
    double reject = 1.0, w_total = 0.0, tvd = 0.0;
    const int nq = q->n == 0 ? 1 : q->n;
    int i, j;

    if (S > (int)(sizeof(w) / sizeof(w[0])))
        return 1.0;
    for (j = 0; j < nq; j++) {
        const uint32_t y = q->n == 0 ? q->token : q->ids[j];
        const double qy = pn_spec_proposal_prob(q, y);
        const double a = spec_accept_prob(spec_p_of(p_ids, p_probs, S, y), qy);
        bool in_p = false;

        reject -= qy * a;
        for (i = 0; i < S && !in_p; i++) {
            if ((uint32_t)p_ids[i] == y) {
                emit[i] += qy * a;
                in_p = true;
            }
        }
        if (!in_p)
            tvd += qy * a;   /* emitted where p is zero */
    }
    for (i = 0; i < S; i++) {
        w[i] = spec_residual_weight(p_ids, p_probs, i, q);
        w_total += w[i];
    }
    for (i = 0; i < S; i++) {
        const double r = w_total > 0.0 ? w[i] / w_total : (double)p_probs[i];
        const double d = emit[i] + (reject > 0.0 ? reject : 0.0) * r - (double)p_probs[i];

        tvd += d < 0.0 ? -d : d;
    }
    return 0.5 * tvd;
}
