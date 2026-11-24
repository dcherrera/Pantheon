/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_row_compare.c
 * @brief Plain vs top-N draws of one grammar-masked row (see grammar_row_compare.h).
 *
 * @status Test support (CPU only).
 * @issues None
 * @todo None
 */
#include "grammar_row_compare.h"
#include "../../src/core/engine.h"
#include "../../src/core/sampler_topn_tail.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRC_TEXT 12

static char *g_texts;              /* n_vocab texts of GRC_TEXT bytes: "@@<id>" */
static int g_texts_n;
static const grc_row *g_row;       /* the row the device callbacks serve */
static unsigned long g_refills;

int grc_init(int n_vocab)
{
    if (g_texts_n >= n_vocab)
        return 0;
    free(g_texts);
    g_texts = malloc((size_t)n_vocab * GRC_TEXT);
    if (!g_texts)
        return -1;
    for (int i = 0; i < n_vocab; i++)
        snprintf(g_texts + (size_t)i * GRC_TEXT, GRC_TEXT, "@@%d", i);
    g_texts_n = n_vocab;
    return 0;
}

static const char *grc_token_text(void *user, uint32_t id)
{
    (void)user;
    return (int)id < g_texts_n ? g_texts + (size_t)id * GRC_TEXT : NULL;
}

/* "@" and "@@" are live prefixes of every token; a whole token is its verdict. */
static int grc_allowed(void *st, const char *text)
{
    if (text[0] != '@') return 0;
    if (!text[1]) return 1;
    if (text[1] != '@') return 0;
    if (!text[2]) return 1;
    return ((const unsigned char *)st)[strtoul(text + 2, NULL, 10)] != 0;
}

static int grc_dense_row(void *ctx, float *logits, int n)
{
    (void)ctx;
    memcpy(logits, g_row->dense, (size_t)n * sizeof(float));
    g_refills++;
    return 0;
}

static int grc_dense_chunks(void *ctx, const unsigned char *need, float *logits, int n)
{
    (void)ctx;
    for (int c = 0; c < (n + 15) / 16; c++)
        for (int i = c * 16; need[c] && i < c * 16 + 16 && i < n; i++) logits[i] = g_row->dense[i];
    return 0;
}

/* The card's tail counts of the raw row (pn_qwen35moe_tt_tail_counts), on the host. */
static int grc_tail_counts(void *ctx, int row, const uint32_t *keys, int bins, uint32_t *counts)
{
    tnt_tail_bins tb;

    (void)ctx;
    if (row != 0 || bins > TNT_TAIL_BINS)
        return -1;
    tb.bins = bins;
    tb.xmin_index = -1;
    memcpy(tb.keys, keys, (size_t)bins * sizeof(keys[0]));
    tnt_tail_bins_count(&tb, g_row->dense, g_row->n_vocab, counts);
    return 0;
}

static int grc_row_already_dense(void *ctx, float *logits, int n)
{
    (void)ctx; (void)logits; (void)n;
    return 0;
}

/* The TT kernel's top-N row: value desc, id asc; -inf elsewhere. */
static void grc_top_n_row(const float *dense, int n_vocab, int topn, float *row)
{
    uint32_t top[GRC_TOPN], neg_inf_bits = 0xff800000u;
    const int cap = topn > 0 && topn < GRC_TOPN ? topn : GRC_TOPN;
    int n = 0;

    for (int i = 0; i < n_vocab; i++) {
        memcpy(&row[i], &neg_inf_bits, sizeof(float));
        if (n == cap && !(dense[i] > dense[top[n - 1]]))
            continue;
        int at = n < cap ? n++ : cap - 1;
        top[at] = (uint32_t)i;
        for (; at > 0 && dense[top[at]] > dense[top[at - 1]]; at--) {
            uint32_t t = top[at];
            top[at] = top[at - 1];
            top[at - 1] = t;
        }
    }
    for (int i = 0; i < n; i++) row[top[i]] = dense[top[i]];
}

/* The top-N side of one seeded draw. Returns the token, or PN_SPEC_CHOOSE_DEFER_TOKEN. */
static uint32_t grc_subject_draw(float *sparse, const grc_row *r, pn_sampling_params p,
                                 grc_route route, uint32_t *recent, grc_result *out)
{
    const unsigned long bound0 = pn_sampler_grammar_bound_steps();
    const unsigned long short0 = pn_sampler_grammar_shortlist_steps();
    const unsigned long chunk0 = pn_sampler_grammar_chunk_steps();
    const unsigned long refill0 = g_refills;
    uint32_t token;

    if (route == GRC_ROW0) {
        p.dense_row = grc_dense_row;
        p.dense_chunks = grc_dense_chunks;
        p.tail_counts = grc_tail_counts;
        token = pn_sample_next_token(sparse, r->n_vocab, &p, recent, r->n_recent);
        out->shortlist += pn_sampler_grammar_shortlist_steps() - short0;
        out->chunk_steps += pn_sampler_grammar_chunk_steps() - chunk0;
    } else {
        /* serve_generation.c sg_gs_masked_pick, a verify row past 0 — including its
         * repeat_penalty = 1: a masked draw carries no repetition penalty (sample_token). */
        p.dense_row = grc_row_already_dense;
        p.repeat_penalty = 1.0f;
        if (!pn_sampler_grammar_shortlist(sparse, r->n_vocab, &p, recent, r->n_recent))
            return PN_SPEC_CHOOSE_DEFER_TOKEN;
        out->shortlist++;
        p.tool_grammar = NULL;
        token = pn_sample_decided_row(sparse, r->n_vocab, &p, recent, r->n_recent);
    }
    out->bound += pn_sampler_grammar_bound_steps() - bound0;
    out->refills += g_refills - refill0;
    return token;
}

void grc_compare(const grc_row *r, grc_route route, int n_seeds, uint64_t seed_base,
                 int grammar_off_subject, grc_result *out)
{
    float *plain = malloc((size_t)r->n_vocab * sizeof(float));
    float *sparse = malloc((size_t)r->n_vocab * sizeof(float));
    float *top_n = malloc((size_t)r->n_vocab * sizeof(float));
    uint32_t recent_a[256], recent_b[256];
    const int n_recent = r->n_recent < 256 ? r->n_recent : 256;
    pn_tool_grammar_hook hook;

    if (!plain || !sparse || !top_n || grc_init(r->n_vocab) != 0) {
        fprintf(stderr, "grc_compare: out of memory\n");
        free(plain); free(sparse); free(top_n);
        return;
    }
    g_row = r;
    memset(&hook, 0, sizeof(hook));
    hook.st = (void *)r->verdict;
    hook.token_text = grc_token_text;
    hook.allowed = grc_allowed;
    hook.active = 1;
    hook.n_vocab = r->n_vocab;
    grc_top_n_row(r->dense, r->n_vocab, r->topn, top_n);
    for (int seed = 0; seed < n_seeds; seed++) {
        pn_sampling_params p = r->params;
        uint32_t want, got;

        p.tool_grammar = &hook;
        p.rng_seeded = 1;
        p.rng_state = seed_base ^ (0x9E3779B97F4A7C15ull * (uint64_t)(seed + 1));
        pn_sampling_params q = p;
        if (grammar_off_subject) q.tool_grammar = NULL;
        memcpy(plain, r->dense, (size_t)r->n_vocab * sizeof(float));
        memcpy(sparse, top_n, (size_t)r->n_vocab * sizeof(float));
        if (n_recent > 0) {
            memcpy(recent_a, r->recent, (size_t)n_recent * sizeof(uint32_t));
            memcpy(recent_b, r->recent, (size_t)n_recent * sizeof(uint32_t));
        }
        want = pn_sample_next_token(plain, r->n_vocab, &p, recent_a, n_recent);
        got = grc_subject_draw(sparse, r, q, route, recent_b, out);
        if (got == PN_SPEC_CHOOSE_DEFER_TOKEN) {
            out->deferred++;
            continue;
        }
        out->draws++;
        out->mismatches += got != want;
    }
    free(plain); free(sparse); free(top_n);
}
