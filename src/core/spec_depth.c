/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_depth.c
 * @brief Implementation of the adaptive speculative draft depth (spec_depth.h).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "spec_depth.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* K80 TP2 (dies 4-7, pinned 2505/875), verify attention through the decode slots call, per-die
 * CUDA graphs: bench_spec_verify_gate verify+commit of n rows = c + s * pos, the least-squares
 * line through 8k / 19k / 32k (within 1 ms of each). Verify+commit ms at 8k / 19k / 32k:
 * 27.2 / 29.8 / 32.9, 39.5 / 42.1 / 45.9, 53.1 / 60.2 / 69.2, 62.5 / 67.8 / 76.9, 72.2 / 81.3 /
 * 95.1. The third row costs more than the others (the n = 3 matvec shape). */
static const double k_verify_c[PN_SPEC_DEPTH_MAX + 2] = { 0.0, 25.27, 37.24, 47.64, 57.19,
                                                          64.00 };
static const double k_verify_s[PN_SPEC_DEPTH_MAX + 2] = { 0.0, 0.000233, 0.000261, 0.000655,
                                                          0.000588, 0.000936 };
/* The batched decode-slots step at K = 1 (the batched serve's D = 0), same runs:
 * 29.8 / 33.0 / 35.7 ms at 8k / 19k / 32k. */
static const double k_slots_c = 28.02, k_slots_s = 0.000239;
/* Gate 1, temp 0.7, the 9 prompts measured at every depth: per-position acceptance
 * 0.80 / 0.63 / 0.50 / 0.42 -> conditional 0.80, 0.79, 0.79, 0.84. */
static const double k_accept[PN_SPEC_DEPTH_MAX + 1] = { 1.0, 0.80, 0.79, 0.79, 0.84 };
/* Dense FFN: Qwen3.8-27B Q8_0 on K80 dies 8-11 under TP2 (k80-q38-tp2), bench_spec_verify_gate
 * verify + commit of n rows at depth 518 / 8198 (140.1 / 145.6, 164.1 / 178.5, 178.2 / 199.4,
 * 189.7 / 219.2, 203.8 / 243.6 ms; 2054 within 3% of the line), the line through both; the
 * plain decode-slots step 154.3 ms at 512 (k80-q38-tp2) with the 1-row slope; 8.2 ms an MTP
 * draft (bench_mtp_e2e, k80-q38-mtp-accept). A 5-row verify costs 1.07-1.11x a 4-row one
 * here (the r4c5 tile; the layer split is alike: 333 / 371 ms at 4k), not the 1.62x of the
 * pre-TP2 priors, which kept the controller off D = 4 even at 0.9 acceptance. */
static const double k_dense_verify_c[PN_SPEC_DEPTH_MAX + 2] = { 0.0, 139.77, 163.17, 176.76,
                                                                187.76, 201.06 };
static const double k_dense_verify_s[PN_SPEC_DEPTH_MAX + 2] = { 0.0, 0.000710, 0.001875,
                                                                0.002755, 0.003832, 0.005190 };
static const double k_dense_plain_c = 154.0, k_dense_plain_s = 0.000710;
static const double k_dense_draft = 8.2;

/* QWEN35_SPEC_COST="c1,s1;...;c5,s5;plain_c,plain_s;draft_first,draft_next" */
static void spec_depth_env_costs(pn_spec_depth *c)
{
    const char *p = getenv("QWEN35_SPEC_COST");
    double v[2 * (PN_SPEC_DEPTH_MAX + 3)];
    int n = 0;

    while (p && *p && n < (int)(sizeof(v) / sizeof(v[0]))) {
        char *end = NULL;
        v[n] = strtod(p, &end);
        if (end == p)
            break;
        n++;
        p = (*end == ',' || *end == ';') ? end + 1 : end;
    }
    if (n != (int)(sizeof(v) / sizeof(v[0]))) {
        if (p)
            fprintf(stderr, "spec_depth: QWEN35_SPEC_COST needs %d numbers, got %d — priors kept\n",
                    (int)(sizeof(v) / sizeof(v[0])), n);
        return;
    }
    for (int r = 1; r <= PN_SPEC_DEPTH_MAX + 1; r++) {
        c->verify_c[r] = v[2 * (r - 1)];
        c->verify_s[r] = v[2 * (r - 1) + 1];
    }
    c->env_costs = 1;
    c->plain_c = v[2 * (PN_SPEC_DEPTH_MAX + 1)];
    c->plain_s = v[2 * (PN_SPEC_DEPTH_MAX + 1) + 1];
    c->draft_first = v[2 * (PN_SPEC_DEPTH_MAX + 2)];
    c->draft_next = v[2 * (PN_SPEC_DEPTH_MAX + 2) + 1];
}

void pn_spec_depth_init(pn_spec_depth *c, int dmax)
{
    memset(c, 0, sizeof(*c));
    c->dmax = dmax < 0 ? 0 : dmax > PN_SPEC_DEPTH_MAX ? PN_SPEC_DEPTH_MAX : dmax;
    memcpy(c->verify_c, k_verify_c, sizeof(k_verify_c));
    memcpy(c->verify_s, k_verify_s, sizeof(k_verify_s));
    memcpy(c->accept, k_accept, sizeof(k_accept));
    /* The single-session plain step (model->forward, full logits + host argmax), same runs as
     * the verify priors: 31.9 / 33.2 / 35.7 ms at 8k / 19k / 32k. It runs the verify core with
     * one row, so it shares the verify's attention kernel (the rows kernel gave 0.61 us/pos). */
    c->plain_c = 30.47;
    c->plain_s = 0.000155;
    c->draft_first = 4.5;
    c->draft_next = 4.5;
    c->conf_cal = 1.0;
    c->ratio = 1.0;
    c->plain_ratio = 1.0;
    c->alpha = 0.1;
    spec_depth_env_costs(c);
}

static double spec_verify_ms(const pn_spec_depth *c, int n, int pos)
{
    return (c->verify_c[n] + c->verify_s[n] * pos) * c->ratio;
}

static double spec_draft_ms(const pn_spec_depth *c, int d)
{
    return d > 0 ? c->draft_first + (d - 1) * c->draft_next : 0.0;
}

/* Expected tokens of a verify with d drafts at conditional acceptances acc[1..d]. */
static double spec_expected_tokens(const double *acc, int d)
{
    double e = 1.0, p = 1.0;

    for (int i = 1; i <= d; i++) {
        p *= acc[i];
        e += p;
    }
    return e;
}

void pn_spec_depth_dense_priors(pn_spec_depth *c)
{
    c->dense = 1;
    if (c->env_costs)
        return;
    memcpy(c->verify_c, k_dense_verify_c, sizeof(k_dense_verify_c));
    memcpy(c->verify_s, k_dense_verify_s, sizeof(k_dense_verify_s));
    /* The dense plain step is the decode-slots step on both paths. */
    c->plain_c = k_dense_plain_c;
    c->plain_s = k_dense_plain_s;
    c->draft_first = k_dense_draft;
    c->draft_next = k_dense_draft;
}

void pn_spec_depth_d0_is_plain(pn_spec_depth *c)
{
    c->d0_plain = 1;
    if (!c->env_costs && !c->dense) {
        c->plain_c = k_slots_c;
        c->plain_s = k_slots_s;
    }
}

double pn_spec_depth_plain_rate(const pn_spec_depth *c, int pos)
{
    /* An unobserved plain step follows the verify's ratio: a hot die, or a model the priors
     * were not measured on, slows both alike. */
    double heat = c->plain_seen ? c->plain_ratio : c->ratio;
    return 1.0 / ((c->plain_c + c->plain_s * pos) * heat);
}

double pn_spec_depth_rate(const pn_spec_depth *c, int d, int pos)
{
    if (d == 0 && c->d0_plain)
        return pn_spec_depth_plain_rate(c, pos);
    return spec_expected_tokens(c->accept, d) /
           (spec_draft_ms(c, d) + spec_verify_ms(c, d + 1, pos) + c->extra_ms);
}

int pn_spec_depth_choose(pn_spec_depth *c, int pos)
{
    int best = 0;
    double best_rate = pn_spec_depth_rate(c, 0, pos);

    for (int d = 1; d <= c->dmax; d++) {
        double r = pn_spec_depth_rate(c, d, pos);
        if (r > best_rate) {
            best = d;
            best_rate = r;
        }
    }
    c->d0_run = best == 0 ? c->d0_run + 1 : 0;
    /* A D = 0 verify observes no acceptance, so a dip would latch D = 0 for good: while it
     * lasts, let the estimate relax toward the prior so drafting gets re-tried. */
    if (best == 0)
        for (int i = 1; i <= PN_SPEC_DEPTH_MAX; i++)
            c->accept[i] += 0.05 * (k_accept[i] - c->accept[i]);
    return best;
}

int pn_spec_depth_more(const pn_spec_depth *c, int pos, int drafted, const float *confs)
{
    double e = 1.0, p = 1.0;

    if (drafted < 1 || drafted >= c->dmax)
        return 0;
    for (int i = 0; i < drafted; i++) {
        double a = c->conf_cal * confs[i];
        p *= a > 0.99 ? 0.99 : a;
        e += p;
    }
    double stop = e / (spec_draft_ms(c, drafted) + spec_verify_ms(c, drafted + 1, pos) +
                       c->extra_ms);
    for (int d = drafted + 1; d <= c->dmax; d++) {
        p *= c->accept[d];
        e += p;
        if (e / (spec_draft_ms(c, d) + spec_verify_ms(c, d + 1, pos) + c->extra_ms) > stop)
            return 1;
    }
    return 0;
}

int pn_spec_depth_plain_wins(const pn_spec_depth *c, int pos, double optimistic)
{
    pn_spec_depth o = *c;

    for (int i = 1; i <= PN_SPEC_DEPTH_MAX; i++)
        if (o.accept[i] < optimistic)
            o.accept[i] = optimistic;
    for (int d = c->d0_plain ? 1 : 0; d <= o.dmax; d++)
        if (pn_spec_depth_rate(&o, d, pos) >= pn_spec_depth_plain_rate(c, pos))
            return 0;
    return 1;
}

int pn_spec_depth_switch_plain(const pn_spec_depth *c, int pos)
{
    return pn_spec_depth_plain_wins(c, pos, 0.9) ||
           (c->d0_run >= 16 &&
            pn_spec_depth_rate(c, 0, pos) < 0.97 * pn_spec_depth_plain_rate(c, pos));
}

void pn_spec_depth_observe_verify(pn_spec_depth *c, int n, int pos, double ms)
{
    if (n < 1 || n > PN_SPEC_DEPTH_MAX + 1)
        return;
    /* One running ratio observed / modelled, shared by every verify width: a hot die or a
     * context past the fitted range slows every width alike, and widths the loop has not
     * tried lately keep their measured proportions. */
    double model = c->verify_c[n] + c->verify_s[n] * pos;
    if (model > 0.0)
        c->ratio += c->alpha * (ms / model - c->ratio);
}

void pn_spec_depth_observe_plain(pn_spec_depth *c, int pos, double ms)
{
    double model = c->plain_c + c->plain_s * pos;
    if (model <= 0.0)
        return;
    if (c->plain_seen++ == 0)
        c->plain_ratio = c->ratio;   /* start from the verify's heat, not the cold prior */
    c->plain_ratio += c->alpha * (ms / model - c->plain_ratio);
}

void pn_spec_depth_observe_extra(pn_spec_depth *c, double ms)
{
    c->extra_ms += c->alpha * (ms - c->extra_ms);
}

void pn_spec_depth_observe_draft(pn_spec_depth *c, int n, double ms)
{
    if (n < 1)
        return;
    /* The first draft carries the catch-up; split the call by the running ratio. */
    double first = ms - (n - 1) * c->draft_next;
    if (n > 1)
        c->draft_next += c->alpha * ((ms - c->draft_first) / (n - 1) - c->draft_next);
    c->draft_first += c->alpha * ((first > 0.0 ? first : ms / n) - c->draft_first);
}

void pn_spec_depth_observe_accept(pn_spec_depth *c, int drafted, int accepted,
                                  const float *confs)
{
    double acc = 0.0, conf = 0.0;

    /* Draft i was put to the test only if every draft before it landed. */
    for (int i = 1; i <= drafted && i <= PN_SPEC_DEPTH_MAX; i++) {
        if (i > accepted + 1)
            break;
        int hit = i <= accepted;
        c->accept[i] += c->alpha * ((double)hit - c->accept[i]);
        acc += hit;
        conf += confs ? confs[i - 1] : 1.0;
    }
    /* Depths past the drafts went untested: like a D = 0 run, a dip at depth i would latch
     * D < i for good, so they relax toward the prior and get re-tried. */
    for (int i = drafted + 1; i <= PN_SPEC_DEPTH_MAX; i++)
        c->accept[i] += 0.05 * (k_accept[i] - c->accept[i]);
    if (conf > 0.0)
        c->conf_cal += c->alpha * (acc / conf - c->conf_cal);
}
