/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp.c
 * @brief Implementation of MTP speculative generation (spec_mtp.h).
 *
 * @status complete
 * @issues Bench driver only: not wired into serve.
 * @todo None
 */
#include "spec_mtp.h"
#include "spec_mtp_couple.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine.h"
#include "../models/qwen35moe.h"

uint32_t pn_mtp_draw_row(float *row, int n_vocab, pn_sampling_params *params,
                         uint32_t *recent, int n_recent)
{
    static int sparse = -1;
    if (sparse < 0) {
        const char *e = getenv("QWEN35_SPEC_SPARSE");
        sparse = !(e && atoi(e) == 0);
    }
    return sparse ? pn_sample_next_token_topn(row, n_vocab, params, recent, n_recent)
                  : pn_sample_next_token(row, n_vocab, params, recent, n_recent);
}

void pn_mtp_recent_push(uint32_t *recent, int *n, int cap, uint32_t tok)
{
    if (cap <= 0)
        return;
    if (*n < cap) {
        recent[(*n)++] = tok;
        return;
    }
    memmove(recent, recent + 1, (size_t)(cap - 1) * sizeof(uint32_t));
    recent[cap - 1] = tok;
}

int pn_mtp_draw_rows_grammar(pn_mtp_row_fetch_fn fetch, void *fetch_ctx, const uint32_t *toks,
                             int n_drafts, float *row, int n_vocab, pn_sampling_params *params,
                             uint32_t *recent, int *n_recent, int repeat_window, uint32_t *emit)
{
    const pn_tool_grammar_hook *hook = params ? params->tool_grammar : NULL;
    const char *red = getenv("PN_MTP_RED");
    const bool red_accept = red && strcmp(red, "1") == 0;
    pn_tool_grammar_hook stale;
    int n_emit = 0;

    if (!fetch || !toks || !row || !hook || !recent || !n_recent || !emit || n_drafts < 0)
        return -1;
    if (red && strcmp(red, "grammar_live_walk") == 0) {
        stale = *hook;
        stale.advance = NULL;
        params->tool_grammar = &stale;
    }
    for (int r = 0; r <= n_drafts; r++) {
        if (fetch(fetch_ctx, r, row) != 0) {
            n_emit = -1;
            break;
        }
        uint32_t tok = pn_sample_next_token(row, n_vocab, params, recent, *n_recent);
        if (r < n_drafts && red_accept)
            tok = toks[r + 1];
        pn_mtp_recent_push(recent, n_recent, repeat_window, tok);
        emit[n_emit++] = tok;
        if (r == n_drafts || tok != toks[r + 1])
            break;
    }
    params->tool_grammar = hook;
    return n_emit;
}

void pn_mtp_present_row(float *row, int *prev, int *n_prev, const int *ids, const float *vals,
                        int k)
{
    for (int i = 0; i < *n_prev; i++)
        row[prev[i]] = -INFINITY;
    for (int i = 0; i < k; i++) {
        row[ids[i]] = vals[i];
        prev[i] = ids[i];
    }
    *n_prev = k;
}

#ifdef USE_CUDA

#define MTP_TOPK_MAX 64

static double mtp_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

typedef struct {
    pn_model *model;
    pn_state *state;
    pn_sampling_params *params;
    uint32_t *recent;
    int *n_recent;
    const pn_mtp_gen_opts *opts;
    pn_spec_depth *ctl;
    pn_mtp_gen_stats *st;
    float *row;          /* [n_vocab] presented row, -inf outside the current pairs */
    int prev[MTP_TOPK_MAX];
    int n_prev;
    pn_mtp_couple couple; /* this cycle's coupled drafts / the probe's pairs */
} mtp_gen;

/* Draw one token from a presented row exactly as the plain decode loop does. `draft` >= 0:
 * the row checks that draft (index `row`), booked by the probe. */
static uint32_t mtp_draw(mtp_gen *g, const int *ids, const float *vals, int row, int64_t draft)
{
    double t0 = mtp_now_ms();
    pn_mtp_present_row(g->row, g->prev, &g->n_prev, ids, vals, g->opts->topk);
    uint32_t tok = pn_mtp_draw_row(g->row, g->model->config.n_vocab, g->params, g->recent,
                                   *g->n_recent);
    if (draft >= 0)
        pn_mtp_couple_probe_row(&g->couple, row, ids, vals, g->opts->topk, g->recent,
                                *g->n_recent, tok == (uint32_t)draft);
    pn_mtp_recent_push(g->recent, g->n_recent, g->opts->repeat_window, tok);
    g->st->ms_sample += mtp_now_ms() - t0;
    return tok;
}

/* One plain decode step (slot 0) after the switch. Returns the drawn token, or -1. */
static int64_t mtp_plain_step(mtp_gen *g, uint32_t seed)
{
    float *logits[1];
    pn_state *sts[1] = { g->state };
    double t0 = mtp_now_ms();

    if (pn_qwen35moe_decode_slots(g->model, &seed, sts, 1, logits) != 0)
        return -1;
    g->st->ms_plain += mtp_now_ms() - t0;
    g->st->plain_steps++;
    t0 = mtp_now_ms();
    uint32_t tok = pn_sample_next_token(logits[0], g->model->config.n_vocab, g->params,
                                        g->recent, *g->n_recent);
    pn_mtp_recent_push(g->recent, g->n_recent, g->opts->repeat_window, tok);
    g->st->ms_sample += mtp_now_ms() - t0;
    return tok;
}

/* The depth for this cycle: fixed, or the controller's (with the one-way plain switch). */
static int mtp_pick_depth(mtp_gen *g, int pos, int *switch_now)
{
    *switch_now = 0;
    if (g->opts->fixed_d >= 0 || !g->ctl)
        return g->opts->fixed_d < 0 ? 0 : g->opts->fixed_d;
    int d = pn_spec_depth_choose(g->ctl, pos);
    if (g->opts->plain_switch && pn_spec_depth_switch_plain(g->ctl, pos))
        *switch_now = 1;
    return d;
}

typedef struct {
    pn_spec_depth *ctl;
    int pos;
    float confs[PN_SPEC_DEPTH_MAX];
} mtp_more_ctx;

static int mtp_more_cb(void *ctx, int drafted, float conf)
{
    mtp_more_ctx *m = (mtp_more_ctx *)ctx;
    if (drafted >= 1 && drafted <= PN_SPEC_DEPTH_MAX)
        m->confs[drafted - 1] = conf;
    return pn_spec_depth_more(m->ctl, m->pos, drafted, m->confs);
}

/* One speculative cycle: drafts, verify, draws, commit, backlog. Writes the emitted tokens
 * to emit[] and returns their count (>= 1), or -1. *stopped = the stop token was drawn. */
static int mtp_cycle(mtp_gen *g, uint32_t seed, int d, int stop_id, uint32_t *emit,
                     int *stopped)
{
    const int k = g->opts->topk, pos = g->state->pos;
    uint32_t toks[PN_SPEC_DEPTH_MAX + 1];
    float confs[PN_SPEC_DEPTH_MAX] = { 0 };
    int ids[(PN_SPEC_DEPTH_MAX + 1) * MTP_TOPK_MAX];
    float vals[(PN_SPEC_DEPTH_MAX + 1) * MTP_TOPK_MAX];
    int nd = 0, n_emit = 0, rows;
    double t0;

    toks[0] = seed;
    g->couple.n_drafts = 0;
    uint32_t looked[PN_SPEC_DEPTH_MAX];
    int nl = g->opts->lookup_draft ? g->opts->lookup_draft(g->opts->lookup_ctx,
                                                           PN_SPEC_DEPTH_MAX, looked) : 0;
    bool looked_up = nl > 0 && g->opts->lookup_mode == 1;
    if (looked_up) {
        memcpy(toks + 1, looked, (size_t)nl * sizeof(uint32_t));
        nd = nl;
    } else if (d > 0) {
        mtp_more_ctx mc = { .ctl = g->ctl, .pos = pos };
        t0 = mtp_now_ms();
        bool pick = pn_mtp_couple_begin(&g->couple, g->params, g->recent, *g->n_recent,
                                        g->opts->repeat_window, g->model->config.n_vocab,
                                        g->opts->couple ? d : 0) ||
                    pn_mtp_couple_probe_on();
        nd = pn_qwen35moe_mtp_draft_pick(g->model, g->state, seed, d, toks + 1, confs,
                                         g->opts->use_more && g->ctl ? mtp_more_cb : NULL, &mc,
                                         pick ? pn_mtp_couple_pick : NULL,
                                         pick ? &g->couple : NULL);
        if (nd < 0)
            nd = 0;
        g->st->ms_draft += mtp_now_ms() - t0;
        if (g->ctl && nd > 0)
            pn_spec_depth_observe_draft(g->ctl, nd, mtp_now_ms() - t0);
        if (nl > 0 && nd >= 1 && g->opts->lookup_mode == 2 && looked[0] == toks[1]) {
            memcpy(toks + 1, looked, (size_t)nl * sizeof(uint32_t));
            nd = nl;
            looked_up = true;
            g->couple.n_drafts = 1;   /* only draft 0 is the head's (the probe) */
        }
    }
    t0 = mtp_now_ms();
    if (pn_qwen35moe_spec_verify_topk(g->model, toks, nd + 1, g->state, k, ids, vals) != 0)
        return -1;
    double t_verify = mtp_now_ms() - t0;
    *stopped = 0;
    for (rows = 0; rows <= nd; rows++) {
        uint32_t tok = mtp_draw(g, ids + (size_t)rows * k, vals + (size_t)rows * k, rows,
                                rows < nd ? (int64_t)toks[rows + 1] : -1);
        emit[n_emit++] = tok;
        if ((int)tok == stop_id) {
            *stopped = 1;
            break;
        }
        if (rows == nd || tok != toks[rows + 1])
            break;
    }
    int accepted = n_emit - 1;   /* drafts that matched their row's draw */
    t0 = mtp_now_ms();
    if (!*stopped && (pn_qwen35moe_spec_accept(g->model, g->state, accepted + 1) != 0 ||
                      pn_qwen35moe_mtp_push_rows(g->model, g->state, toks, accepted + 1) != 0))
        return -1;
    t_verify += mtp_now_ms() - t0;
    g->st->ms_verify += t_verify;
    if (g->ctl) {
        pn_spec_depth_observe_verify(g->ctl, nd + 1, pos, t_verify);
        if (nd > 0 && !looked_up)
            pn_spec_depth_observe_accept(g->ctl, nd, accepted, confs);
    }
    if (looked_up) {
        g->opts->lookup_report(g->opts->lookup_ctx, nd, accepted);
        g->st->lookup_cycles++;
        g->st->lookup_accepted += accepted;
    }
    g->st->cycles++;
    g->st->drafted += nd;
    g->st->accepted += accepted;
    g->st->depth_hist[nd]++;
    return n_emit;
}

int pn_mtp_spec_generate(pn_model *model, pn_state *state, uint32_t seed, int max_tokens,
                         int stop_id, pn_sampling_params *params, uint32_t *recent,
                         int *n_recent, const pn_mtp_gen_opts *opts, pn_spec_depth *ctl,
                         uint32_t *out, pn_mtp_gen_stats *stats)
{
    mtp_gen g = { .model = model, .state = state, .params = params, .recent = recent,
                  .n_recent = n_recent, .opts = opts, .ctl = ctl, .st = stats };
    uint32_t emit[PN_SPEC_DEPTH_MAX + 1];
    int n = 0, switched = 0, stopped = 0, failed = 0;

    if (!model || !state || !opts || !out || !stats || opts->topk < 1 ||
        opts->topk > MTP_TOPK_MAX)
        return -1;
    memset(stats, 0, sizeof(*stats));
    stats->switched_at = -1;
    g.row = malloc((size_t)model->config.n_vocab * sizeof(float));
    if (!g.row)
        return -1;
    for (int i = 0; i < model->config.n_vocab; i++)
        g.row[i] = -INFINITY;
    while (n < max_tokens && !stopped) {
        int switch_now = 0, ne;
        int d = switched ? 0 : mtp_pick_depth(&g, state->pos, &switch_now);
        if (switch_now && !switched) {
            if (pn_qwen35moe_decode_slot_bind(model, state, 0, 1) != 0) {
                failed = 1;
                break;
            }
            switched = 1;
            stats->switched_at = state->pos;
        }
        if (switched) {
            int64_t tok = mtp_plain_step(&g, seed);
            if (tok < 0) {
                failed = 1;
                break;
            }
            emit[0] = (uint32_t)tok;
            ne = 1;
            stopped = (int)tok == stop_id;
        } else if ((ne = mtp_cycle(&g, seed, d, stop_id, emit, &stopped)) < 0) {
            failed = 1;
            break;
        }
        for (int i = 0; i < ne && n < max_tokens; i++) {
            if (opts->lookup_push)
                opts->lookup_push(opts->lookup_ctx, emit[i]);
            if ((int)emit[i] == stop_id)
                break;
            out[n++] = emit[i];
        }
        seed = emit[ne - 1];
    }
    stats->emitted = n;
    free(g.row);
    return failed ? -1 : n;
}

#endif /* USE_CUDA */
