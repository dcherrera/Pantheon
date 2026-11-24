/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_sampling_chooser.c
 * @brief Implementation of true speculative sampling on the chooser context
 *        (spec_sampling_chooser.h): drafts drawn from q, rows decided by the rule, the
 *        distribution check.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#include "spec_sampling_chooser.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool pn_spec_rule_applies(const pn_sampling_params *params)
{
    const char *env = getenv("PN_TT_SPEC_SAMPLE_RULE");
    const int mode = env && *env ? atoi(env) : 0;

    if (mode == 0 || !params || params->temperature <= 0.0f || params->tool_grammar ||
        params->top_k < 1 || params->top_k > PN_SPEC_PROPOSAL_MAX)
        return false;
    /* Off by default: +~1% at temperature 1.0 is not worth losing byte-identity with plain
     * sampling. `=2` only where it accepts more than the argmax draft (the proposal probe). */
    return mode == 1 || (mode == 2 && params->temperature >= PN_SPEC_RULE_MIN_TEMP);
}

int pn_spec_sample_rule_arm(pn_spec_sample_ctx *c)
{
    c->q = calloc(PN_SPEC_SAMPLE_MAX_DRAFTS, sizeof(*c->q));
    c->draft_recent = malloc((size_t)(c->max_recent > 0 ? c->max_recent : 1) * sizeof(uint32_t));
    c->n_q = 0;
    if (c->q && c->draft_recent)
        return 0;
    pn_spec_sample_rule_free(c);
    return -1;
}

void pn_spec_sample_rule_free(pn_spec_sample_ctx *c)
{
    free(c->q);
    free(c->draft_recent);
    c->q = NULL;
    c->draft_recent = NULL;
    c->n_q = 0;
}

void pn_spec_sample_begin_drafts(pn_spec_sample_ctx *c) { c->n_q = 0; }

/* The repeat window plain decode would have at draft i: the cycle's base window (the seed
 * included) followed by drafts 0..i-1, sliding at max_recent. */
static int spec_draft_window(pn_spec_sample_ctx *c, int i)
{
    int n = c->n_recent, j;

    if (c->max_recent <= 0)
        return 0;
    memcpy(c->draft_recent, c->recent, (size_t)n * sizeof(uint32_t));
    for (j = 0; j < i; j++) {
        if (n < c->max_recent) {
            c->draft_recent[n++] = c->q[j].token;
        } else {
            memmove(c->draft_recent, c->draft_recent + 1,
                    (size_t)(c->max_recent - 1) * sizeof(uint32_t));
            c->draft_recent[c->max_recent - 1] = c->q[j].token;
        }
    }
    return n;
}

/* The best pair: largest value, lowest id on a tie (the head's greedy rule). */
static uint32_t spec_best_pair(const uint32_t *index, const float *value, uint32_t count,
                               int n_vocab)
{
    uint32_t i, best = 0u;
    int found = 0;

    for (i = 0; i < count; i++) {
        if (index[i] >= (uint32_t)n_vocab)
            continue;
        if (!found || value[i] > value[best] || (value[i] == value[best] && index[i] < index[best])) {
            best = i;
            found = 1;
        }
    }
    return found ? index[best] : 0u;
}

static void spec_probe_keep(int i, const uint32_t *index, const float *value, uint32_t count);

/* `PN_TT_SPEC_Q_TEMP=s`: q is the sampler at temperature T * s (s <= 0: a point mass at the
 * best pair, the old rule's draft). Any q is exact; a q sharper than the request's own
 * temperature suits a draft head flatter than the target (spec "TRUE SPECULATIVE SAMPLING"). */
static float spec_q_temp_scale(void)
{
    static float s = -2.0f;

    if (s < -1.5f) {
        const char *env = getenv("PN_TT_SPEC_Q_TEMP");

        s = env && *env ? (float)atof(env) : PN_SPEC_Q_TEMP_DEFAULT;
    }
    return s;
}

/* The best pair that may be drafted (not a stop token); any pair when none. */
static uint32_t spec_best_draftable(const pn_spec_sample_ctx *c, const uint32_t *index,
                                    const float *value, uint32_t count)
{
    uint32_t i, kept[PN_SPEC_PROPOSAL_MAX];
    float kv[PN_SPEC_PROPOSAL_MAX];
    uint32_t n = 0;
    int j;

    for (i = 0; i < count && n < PN_SPEC_PROPOSAL_MAX; i++) {
        bool stop = false;
        for (j = 0; j < c->n_no_draft; j++) stop = stop || index[i] == c->no_draft[j];
        if (!stop) { kept[n] = index[i]; kv[n++] = value[i]; }
    }
    return n ? spec_best_pair(kept, kv, n, c->n_vocab) : spec_best_pair(index, value, count, c->n_vocab);
}

uint32_t pn_spec_sample_draft(void *opaque, int draft_index, const uint32_t *index,
                              const float *value, uint32_t count)
{
    pn_spec_sample_ctx *c = (pn_spec_sample_ctx *)opaque;
    int ids[PN_SPEC_PROPOSAL_MAX];
    float probs[PN_SPEC_PROPOSAL_MAX];
    pn_sampling_params qp;
    pn_spec_proposal *q;
    const float scale = spec_q_temp_scale();
    int n_win, S = 0;
    float u;

    if (!c->q || draft_index < 0 || draft_index >= PN_SPEC_SAMPLE_MAX_DRAFTS)
        return spec_best_pair(index, value, count, c->n_vocab);
    q = &c->q[draft_index];
    spec_probe_keep(draft_index, index, value, count);
    n_win = spec_draft_window(c, draft_index);
    qp = *c->params;
    qp.temperature *= scale;
    /* A uniform carried for a deferred position belongs to that position's draw (the tail
     * planner, top-k off or past 64 — requests the rule does not arm for): never spend it on
     * a draft; the draft is then the best pair as a point mass. */
    if (scale > 0.0f && !c->params->carried_uniform_set)
        S = pn_sampler_pairs_final_probs(index, value, count, c->n_vocab, &qp, c->draft_recent,
                                         n_win, ids, probs, PN_SPEC_PROPOSAL_MAX);
    u = S >= 1 ? pn_sampler_uniform(c->params) : 0.0f;
    if (S < 1 || pn_spec_proposal_build(q, ids, probs, S, c->no_draft, c->n_no_draft, u) != 0) {
        q->n = 0;
        q->token = spec_best_draftable(c, index, value, count);
    }
    c->n_q = draft_index + 1;
    c->stats.drafted++;
    return q->token;
}

void pn_spec_sample_point_drafts(pn_spec_sample_ctx *c, int from, const uint32_t *drafts, int n)
{
    int i;

    if (!c->q)
        return;
    if (n > PN_SPEC_SAMPLE_MAX_DRAFTS)
        n = PN_SPEC_SAMPLE_MAX_DRAFTS;
    for (i = from; i < n; i++) {
        c->q[i].n = 0;
        c->q[i].token = drafts[i];
    }
    c->n_q = n;
}

void pn_spec_sample_cut_drafts(pn_spec_sample_ctx *c, int n_drafts)
{
    if (c->n_q > n_drafts)
        c->n_q = n_drafts > 0 ? n_drafts : 0;
}

/* ------------------------------------------------------------------------------------------
 * The distribution check (`PN_SPEC_SAMPLE_DIST_CHECK=M`)
 * ------------------------------------------------------------------------------------------ */

static struct {
    unsigned long rows, outside;
    double max_tvd, sum_tvd;
    double chi2, df;             /* one-sample: emitted histogram vs p */
    double chi2_plain, df_plain; /* two-sample: emitted vs the plain sampler's draws */
    double sum_acceptance;
    int draws;
} g_dist;
static uint64_t g_dist_rng = 0x2545F4914F6CDD1Dull;

static float spec_dist_uniform(void *ctx)
{
    (void)ctx;
    g_dist_rng ^= g_dist_rng << 13;
    g_dist_rng ^= g_dist_rng >> 7;
    g_dist_rng ^= g_dist_rng << 17;
    return (float)((g_dist_rng >> 40) & 0xFFFFFF) / 16777216.0f;
}

static void spec_dist_report(void)
{
    const double z = g_dist.df > 0.0 ? (g_dist.chi2 - g_dist.df) / sqrt(2.0 * g_dist.df) : 0.0;
    const double zp = g_dist.df_plain > 0.0
                          ? (g_dist.chi2_plain - g_dist.df_plain) / sqrt(2.0 * g_dist.df_plain)
                          : 0.0;

    if (!g_dist.rows)
        return;
    fprintf(stderr, "SPEC SAMPLE DIST — %lu rows x %d draws: emitted-vs-p TVD max %.3e mean %.3e "
                    "| vs p chi2 %.1f on %.0f df (z %.2f) | vs plain draws chi2 %.1f on %.0f df "
                    "(z %.2f) | draws outside p %lu | mean sum min(p,q) %.3f\n",
            g_dist.rows, g_dist.draws, g_dist.max_tvd, g_dist.sum_tvd / (double)g_dist.rows,
            g_dist.chi2, g_dist.df, z, g_dist.chi2_plain, g_dist.df_plain, zp, g_dist.outside,
            g_dist.sum_acceptance / (double)g_dist.rows);
}

/* The plain sampler's draw over its final probs (sparse_sample's scan). */
static int spec_plain_scan(const float *probs, int S, float r)
{
    float cum = 0.0f;
    int i;

    for (i = 0; i < S; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    for (i = S - 1; i >= 0; i--)
        if (probs[i] > 0.0f) return i;
    return -1;
}

/* One row: M draws of (draft ~ q, the rule) and M plain draws; chi-square bins pool every
 * token whose expected count is under 5 (with draws outside p) into one bin. */
static void spec_dist_check_row(const int *ids, const float *probs, int S,
                                const pn_spec_proposal *q, int M)
{
    static int registered;
    unsigned long hist[PN_SPEC_PROPOSAL_MAX + 1] = { 0 }, plain[PN_SPEC_PROPOSAL_MAX + 1] = { 0 };
    pn_spec_proposal draw = *q;
    double small_exp = 0.0, small_obs = 0.0, small_plain = 0.0;
    int m, i, bins = 0, bins_plain = 0;

    if (!registered) { atexit(spec_dist_report); registered = 1; }
    for (m = 0; m < M; m++) {
        uint32_t y;
        int at = S;

        draw.token = pn_spec_proposal_sample(q, spec_dist_uniform(NULL));
        y = pn_spec_rule_verify(ids, probs, S, &draw, spec_dist_uniform, NULL, NULL);
        for (i = 0; i < S; i++)
            if ((uint32_t)ids[i] == y) { at = i; break; }
        if (at == S || !(probs[at] > 0.0f)) g_dist.outside++;
        hist[at]++;
        at = spec_plain_scan(probs, S, spec_dist_uniform(NULL));
        plain[at < 0 ? S : at]++;
    }
    for (i = 0; i <= S; i++) {
        const double e = i < S ? (double)M * (double)probs[i] : 0.0;

        if (e < 5.0) {
            small_exp += e;
            small_obs += (double)hist[i];
            small_plain += (double)plain[i];
            continue;
        }
        g_dist.chi2 += ((double)hist[i] - e) * ((double)hist[i] - e) / e;
        bins++;
        if (hist[i] + plain[i] == 0u)   /* a red can empty an expected bin on both sides */
            continue;
        g_dist.chi2_plain += ((double)hist[i] - (double)plain[i]) *
                             ((double)hist[i] - (double)plain[i]) / (double)(hist[i] + plain[i]);
        bins_plain++;
    }
    if (small_exp >= 1.0) {
        g_dist.chi2 += (small_obs - small_exp) * (small_obs - small_exp) / small_exp;
        bins++;
    }
    if (small_obs + small_plain >= 10.0) {
        g_dist.chi2_plain += (small_obs - small_plain) * (small_obs - small_plain) /
                             (small_obs + small_plain);
        bins_plain++;
    }
    g_dist.df += bins > 1 ? bins - 1 : 0;
    g_dist.df_plain += bins_plain > 1 ? bins_plain - 1 : 0;
}

static void spec_dist_check(const int *ids, const float *probs, int S, const pn_spec_proposal *q)
{
    static int M = -1;
    double tvd;

    if (M < 0) {
        const char *env = getenv("PN_SPEC_SAMPLE_DIST_CHECK");

        M = env ? atoi(env) : 0;
    }
    if (M <= 0)
        return;
    g_dist.draws = M;
    tvd = pn_spec_rule_emitted_tvd(ids, probs, S, q);
    if (tvd > g_dist.max_tvd) g_dist.max_tvd = tvd;
    g_dist.sum_tvd += tvd;
    g_dist.sum_acceptance += pn_spec_rule_acceptance(ids, probs, S, q);
    spec_dist_check_row(ids, probs, S, q, M);
    ++g_dist.rows;
}

/* ------------------------------------------------------------------------------------------ */

static float spec_request_uniform(void *params)
{
    return pn_sampler_uniform((pn_sampling_params *)params);
}

/* ------------------------------------------------------------------------------------------
 * The proposal probe (`PN_SPEC_SAMPLE_Q_PROBE=1`, diagnostics): on every rule row, the
 * acceptance sum min(p, q') of other proposals q' built from the same draft-head pairs — q's
 * temperature x {1, 0.7, 0.5, 0.3, 0.15}, over 64 and over the top 8 pairs — and the
 * shared-randomness rule's p(argmax): the same rows for every candidate, no trajectory noise.
 * ------------------------------------------------------------------------------------------ */
#define SPEC_PROBE_SCALES 5
static const float SPEC_PROBE_SCALE[SPEC_PROBE_SCALES] = { 1.0f, 0.7f, 0.5f, 0.3f, 0.15f };
static struct {
    uint32_t ids[PN_SPEC_SAMPLE_MAX_DRAFTS][PN_SPEC_PROPOSAL_MAX];
    float vals[PN_SPEC_SAMPLE_MAX_DRAFTS][PN_SPEC_PROPOSAL_MAX];
    uint32_t n[PN_SPEC_SAMPLE_MAX_DRAFTS];
    double sum[2][SPEC_PROBE_SCALES], argmax;
    unsigned long rows;
    int on;
} g_probe = { .on = -1 };

static bool spec_probe_on(void)
{
    if (g_probe.on < 0)
        g_probe.on = getenv("PN_SPEC_SAMPLE_Q_PROBE") && atoi(getenv("PN_SPEC_SAMPLE_Q_PROBE")) != 0;
    return g_probe.on;
}

static void spec_probe_keep(int i, const uint32_t *index, const float *value, uint32_t count)
{
    if (!spec_probe_on() || i < 0 || i >= PN_SPEC_SAMPLE_MAX_DRAFTS)
        return;
    if (count > PN_SPEC_PROPOSAL_MAX) count = PN_SPEC_PROPOSAL_MAX;
    memcpy(g_probe.ids[i], index, count * sizeof(uint32_t));
    memcpy(g_probe.vals[i], value, count * sizeof(float));
    g_probe.n[i] = count;
}

/* The top `keep` pairs by value (all when keep >= n). */
static uint32_t spec_probe_top(const uint32_t *ids, const float *vals, uint32_t n, uint32_t keep,
                               uint32_t *out_ids, float *out_vals)
{
    uint32_t i, j, m = 0;

    for (i = 0; i < n; i++) {
        uint32_t rank = 0;
        for (j = 0; j < n; j++)
            rank += vals[j] > vals[i] || (vals[j] == vals[i] && j < i);
        if (rank < keep) { out_ids[m] = ids[i]; out_vals[m++] = vals[i]; }
    }
    return m;
}

static void spec_probe_row(pn_spec_sample_ctx *c, int row, const int *ids, const float *probs, int S)
{
    uint32_t top_ids[PN_SPEC_PROPOSAL_MAX];
    float top_vals[PN_SPEC_PROPOSAL_MAX];
    int qi[PN_SPEC_PROPOSAL_MAX], w, k;
    float qp[PN_SPEC_PROPOSAL_MAX];

    if (!spec_probe_on() || row >= PN_SPEC_SAMPLE_MAX_DRAFTS || !g_probe.n[row])
        return;
    for (w = 0; w < 2; w++) {
        const uint32_t n = spec_probe_top(g_probe.ids[row], g_probe.vals[row], g_probe.n[row],
                                          w ? 8u : PN_SPEC_PROPOSAL_MAX, top_ids, top_vals);
        for (k = 0; k < SPEC_PROBE_SCALES; k++) {
            pn_sampling_params qpar = *c->params;
            pn_spec_proposal q;
            int Sq;

            qpar.temperature *= SPEC_PROBE_SCALE[k];
            Sq = pn_sampler_pairs_final_probs(top_ids, top_vals, n, c->n_vocab, &qpar, c->recent,
                                              c->n_recent, qi, qp, PN_SPEC_PROPOSAL_MAX);
            if (Sq >= 1 && pn_spec_proposal_build(&q, qi, qp, Sq, c->no_draft, c->n_no_draft,
                                                  0.0f) == 0)
                g_probe.sum[w][k] += pn_spec_rule_acceptance(ids, probs, S, &q);
        }
    }
    {
        pn_spec_proposal pm = { .n = 0 };
        pm.token = spec_best_draftable(c, g_probe.ids[row], g_probe.vals[row], g_probe.n[row]);
        g_probe.argmax += pn_spec_rule_acceptance(ids, probs, S, &pm);
    }
    g_probe.rows++;
}

static void spec_probe_report(void)
{
    const double r = g_probe.rows ? (double)g_probe.rows : 1.0;
    int k;

    if (!g_probe.rows)
        return;
    fprintf(stderr, "SPEC Q PROBE — %lu rows: p(argmax) %.3f | sum min(p,q) at q temp x",
            g_probe.rows, g_probe.argmax / r);
    for (k = 0; k < SPEC_PROBE_SCALES; k++)
        fprintf(stderr, " %.2f: %.3f/%.3f", SPEC_PROBE_SCALE[k], g_probe.sum[0][k] / r,
                g_probe.sum[1][k] / r);
    fprintf(stderr, " (64 pairs / top 8)\n");
}

bool pn_spec_rule_choose_row(pn_spec_sample_ctx *c, int row, uint32_t *token)
{
    int ids[PN_SPEC_PROPOSAL_MAX];
    float probs[PN_SPEC_PROPOSAL_MAX];
    const pn_spec_proposal *q = &c->q[row];
    bool accepted = false;
    const int S = pn_sampler_row_final_probs(c->row, c->n_vocab, c->params, c->recent,
                                             c->n_recent, true, ids, probs, PN_SPEC_PROPOSAL_MAX);

    if (S < 1) {
        c->stats.fallback++;
        return false;
    }
    spec_dist_check(ids, probs, S, q);
    spec_probe_row(c, row, ids, probs, S);
    *token = pn_spec_rule_verify(ids, probs, S, q, spec_request_uniform, c->params, &accepted);
    c->stats.rule_rows++;
    c->stats.accepted += accepted ? 1u : 0u;
    c->stats.sum_acceptance += pn_spec_rule_acceptance(ids, probs, S, q);
    return true;
}

void pn_spec_rule_report(const pn_spec_sample_ctx *c, const char *who)
{
    const pn_spec_rule_stats *s = &c->stats;

    if (!c->q || !s->drafted)
        return;
    spec_dist_report();   /* cumulative over the process; silent without the check */
    spec_probe_report();
    fprintf(stderr, "[spec sampling] %s: drafts drawn %lu, rule rows %lu, accepted %lu (%.3f), "
                    "mean sum min(p,q) %.3f, plain-draw rows %lu\n",
            who, s->drafted, s->rule_rows, s->accepted,
            s->rule_rows ? (double)s->accepted / (double)s->rule_rows : 0.0,
            s->rule_rows ? s->sum_acceptance / (double)s->rule_rows : 0.0, s->fallback);
}
