/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp_couple.c
 * @brief Implementation of coupled MTP drafts and the acceptance probe (spec_mtp_couple.h).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "spec_mtp_couple.h"

#include "engine.h"
#include "sampling_modifiers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool pn_mtp_couple_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_MTP_COUPLE");
        v = !(e && atoi(e) == 0);
    }
    return v;
}

bool pn_mtp_couple_probe_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PN_MTP_PROBE");
        v = e && atoi(e) != 0;
    }
    return v;
}

bool pn_mtp_couple_begin(pn_mtp_couple *c, pn_sampling_params *params, const uint32_t *recent,
                         int n_recent, int max_recent, int n_vocab, int k)
{
    c->params = params;
    c->recent = recent;
    c->n_recent = n_recent;
    c->max_recent = max_recent;
    c->n_vocab = n_vocab;
    c->n_drafts = 0;
    c->n_u = 0;
    c->on = false;
    memset(c->n_pairs, 0, sizeof(c->n_pairs));
    c->cycles++;
    if (!pn_mtp_couple_enabled() || !params || params->temperature <= 0.0f ||
        params->tool_grammar || params->dense_row || params->n_logit_bias > 0 || k < 1 ||
        max_recent > PN_MTP_COUPLE_WINDOW || n_recent > PN_MTP_COUPLE_WINDOW ||
        (n_recent > 0 && !recent))
        return false;
    if (k > PN_MTP_COUPLE_MAX)
        k = PN_MTP_COUPLE_MAX;
    if (!params->rng_seeded) {
        /* rand() cannot be peeked: a private stream seeded from it draws the same way. */
        uint64_t seed = ((uint64_t)(unsigned)rand() << 32) ^ (uint64_t)(unsigned)rand();
        params->rng_state = pn_seed_to_rng_state(seed);
        params->rng_seeded = 1;
    }
    /* A copy advances; the request's stream does not. RED (gates only):
     * PN_MTP_RED=couple_consume peeks on the request's own stream, which must break the
     * byte-identity with plain decode. */
    const char *red = getenv("PN_MTP_RED");
    pn_sampling_params peek = *params;
    pn_sampling_params *src = red && strcmp(red, "couple_consume") == 0 ? params : &peek;
    for (int i = 0; i < k; i++)
        c->u[i] = pn_sampler_uniform(src);
    c->n_u = k;
    c->on = true;
    return true;
}

/* The window row i draws under: recent, then drafts 0..i-1, sliding at max_recent. */
static int couple_window(pn_mtp_couple *c, int i)
{
    int n = c->n_recent;

    if (c->max_recent <= 0)
        return 0;
    memcpy(c->window, c->recent, (size_t)n * sizeof(uint32_t));
    for (int j = 0; j < i; j++) {
        if (n < c->max_recent) {
            c->window[n++] = c->drafts[j];
        } else {
            memmove(c->window, c->window + 1, (size_t)(c->max_recent - 1) * sizeof(uint32_t));
            c->window[c->max_recent - 1] = c->drafts[j];
        }
    }
    return n;
}

/* The plain sampler's scan (sparse_sample): the first survivor whose running sum passes r. */
static int couple_scan(const float *probs, int S, float r)
{
    float cum = 0.0f;

    for (int i = 0; i < S; i++) {
        cum += probs[i];
        if (r < cum)
            return i;
    }
    for (int i = S - 1; i >= 0; i--)
        if (probs[i] > 0.0f)
            return i;
    return -1;
}

uint32_t pn_mtp_couple_pick(void *ctx, int i, const int *ids, const float *vals, int n)
{
    pn_mtp_couple *c = (pn_mtp_couple *)ctx;
    uint32_t tok = n > 0 ? (uint32_t)ids[0] : 0u;

    if (i < 0 || i >= PN_MTP_COUPLE_MAX || n < 1)
        return tok;
    if (n > PN_MTP_COUPLE_PAIRS)
        n = PN_MTP_COUPLE_PAIRS;
    for (int j = 0; j < n; j++) {
        c->ids[i][j] = (uint32_t)ids[j];
        c->vals[i][j] = vals[j];
    }
    c->n_pairs[i] = n;
    if (c->on && i < c->n_u) {
        int qi[PN_MTP_COUPLE_PAIRS];
        float qp[PN_MTP_COUPLE_PAIRS];
        int nw = couple_window(c, i);
        int S = pn_sampler_pairs_final_probs(c->ids[i], c->vals[i], (uint32_t)n, c->n_vocab,
                                             c->params, c->window, nw, qi, qp,
                                             PN_MTP_COUPLE_PAIRS);
        int at = S >= 1 ? couple_scan(qp, S, c->u[i]) : -1;
        if (at >= 0) {
            tok = (uint32_t)qi[at];
            c->coupled_drafts++;
        } else {
            c->argmax_drafts++;
        }
    } else {
        c->argmax_drafts++;
    }
    c->drafts[i] = tok;
    if (i + 1 > c->n_drafts)
        c->n_drafts = i + 1;
    return tok;
}

double pn_mtp_couple_overlap(const int *p_ids, const float *p_probs, int Sp, const int *q_ids,
                             const float *q_probs, int Sq)
{
    /* Walk the union of ids ascending; each id owns [lo, hi) of [0, 1) in either scan. */
    double plo = 0.0, qlo = 0.0, acc = 0.0;
    int a = 0, b = 0;

    while (a < Sp || b < Sq) {
        int id = a < Sp && (b >= Sq || p_ids[a] <= q_ids[b]) ? p_ids[a] : q_ids[b];
        double pp = a < Sp && p_ids[a] == id ? (double)p_probs[a++] : 0.0;
        double qq = b < Sq && q_ids[b] == id ? (double)q_probs[b++] : 0.0;
        double lo = plo > qlo ? plo : qlo, hi = plo + pp < qlo + qq ? plo + pp : qlo + qq;

        if (pp > 0.0 && qq > 0.0 && hi > lo)
            acc += hi - lo;
        plo += pp;
        qlo += qq;
    }
    return acc;
}

/* ---- the probe ---- */

#define PROBE_POS PN_MTP_COUPLE_MAX
#define PROBE_GAPS 5
static const float PROBE_GAP_EDGE[PROBE_GAPS - 1] = { 0.05f, 0.1f, 0.25f, 0.5f };

static struct {
    /* sampled rows, [0] argmax-drafted cycles / [1] coupled cycles */
    long rows[2][PROBE_POS], acc[2][PROBE_POS];
    double argmax[2][PROBE_POS], rs[2][PROBE_POS], coupled[2][PROBE_POS];
    /* greedy rows */
    long grows[PROBE_POS], gacc[PROBE_POS];
    long gap[PROBE_GAPS + 1];   /* greedy misses: head logit gap argmax - target token */
    int registered;
} g_probe;

static void probe_report(void)
{
    for (int i = 0; i < PROBE_POS; i++) {
        for (int c = 0; c < 2; c++)
            if (g_probe.rows[c][i])
                fprintf(stderr, "MTP PROBE sampled %s pos %d: rows %ld live %.3f | p(argmax q) "
                                "%.3f | sum min(p,q) %.3f | coupled %.3f\n",
                        c ? "coupled" : "argmax", i + 1, g_probe.rows[c][i],
                        (double)g_probe.acc[c][i] / g_probe.rows[c][i],
                        g_probe.argmax[c][i] / g_probe.rows[c][i],
                        g_probe.rs[c][i] / g_probe.rows[c][i],
                        g_probe.coupled[c][i] / g_probe.rows[c][i]);
        if (g_probe.grows[i])
            fprintf(stderr, "MTP PROBE greedy pos %d: rows %ld accepted %.3f\n", i + 1,
                    g_probe.grows[i], (double)g_probe.gacc[i] / g_probe.grows[i]);
    }
    if (g_probe.gap[0] + g_probe.gap[1] + g_probe.gap[2] + g_probe.gap[3] + g_probe.gap[4] +
        g_probe.gap[5])
        fprintf(stderr, "MTP PROBE greedy misses by head logit gap (argmax - target): <0.05 %ld "
                        "| <0.1 %ld | <0.25 %ld | <0.5 %ld | >=0.5 %ld | target not in head "
                        "top %d %ld\n", g_probe.gap[0], g_probe.gap[1], g_probe.gap[2],
                g_probe.gap[3], g_probe.gap[4], PN_MTP_COUPLE_PAIRS, g_probe.gap[5]);
}

void pn_mtp_couple_probe_row(const pn_mtp_couple *c, int row, const int *p_ids,
                             const float *p_vals, int k, const uint32_t *window, int n_window,
                             bool accepted)
{
    if (!pn_mtp_couple_probe_on() || !c || row < 0 || row >= PROBE_POS || row >= c->n_drafts ||
        c->n_pairs[row] < 1 || !c->params)
        return;
    if (!g_probe.registered) {
        atexit(probe_report);
        g_probe.registered = 1;
    }
    if (c->params->temperature <= 0.0f) {
        g_probe.grows[row]++;
        g_probe.gacc[row] += accepted;
        if (!accepted) {
            /* The target's argmax (descending pairs: the first) in the head's pairs. */
            int b = 5;
            for (int j = 0; j < c->n_pairs[row]; j++)
                if (c->ids[row][j] == (uint32_t)p_ids[0]) {
                    float gap = c->vals[row][0] - c->vals[row][j];
                    for (b = 0; b < PROBE_GAPS - 1 && gap >= PROBE_GAP_EDGE[b]; b++)
                        ;
                    break;
                }
            g_probe.gap[b]++;
        }
        return;
    }
    uint32_t pidx[64];
    int pi[64], qi[PN_MTP_COUPLE_PAIRS];
    float pp[64], qp[PN_MTP_COUPLE_PAIRS];
    if (k > 64)
        k = 64;
    for (int j = 0; j < k; j++)
        pidx[j] = (uint32_t)p_ids[j];
    int Sp = pn_sampler_pairs_final_probs(pidx, p_vals, (uint32_t)k, c->n_vocab, c->params,
                                          window, n_window, pi, pp, 64);
    int Sq = pn_sampler_pairs_final_probs(c->ids[row], c->vals[row], (uint32_t)c->n_pairs[row],
                                          c->n_vocab, c->params, window, n_window, qi, qp,
                                          PN_MTP_COUPLE_PAIRS);
    if (Sp < 1 || Sq < 1)
        return;
    double am = 0.0, rs = 0.0;
    for (int a = 0; a < Sp; a++) {
        if ((uint32_t)pi[a] == c->ids[row][0])
            am = pp[a];
        for (int b = 0; b < Sq; b++)
            if (qi[b] == pi[a])
                rs += pp[a] < qp[b] ? pp[a] : qp[b];
    }
    const int on = c->on ? 1 : 0;
    g_probe.rows[on][row]++;
    g_probe.acc[on][row] += accepted;
    g_probe.argmax[on][row] += am;
    g_probe.rs[on][row] += rs;
    g_probe.coupled[on][row] += pn_mtp_couple_overlap(pi, pp, Sp, qi, qp, Sq);
}
