/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file spec_mtp_step.c
 * @brief Implementation of the one-cycle CUDA MTP draft/step pair (spec_mtp_step.h).
 *
 * RED arm (exactness gates only): PN_MTP_RED=1 makes pn_qwen35moe_cuda_spec_step accept every
 * draft without comparing it to its row's draw, so the emitted stream MUST differ from plain
 * decode wherever a draft was wrong. A gate that stays green under it proves nothing.
 *
 * Under a tool grammar (params->tool_grammar set: the batched driver's scratch-walk hook) the
 * rows are drawn DENSE (pn_mtp_draw_rows_grammar over pn_qwen35moe_spec_row_logits), because
 * plain grammar decoding draws from the full row. RED PN_MTP_RED=grammar_topn draws them from
 * the top-k presentation with the hook detached instead (must differ from plain).
 *
 * @status complete
 * @issues None
 * @todo Keep the presented row per request instead of a malloc + -inf fill per cycle.
 */
#include "spec_mtp_step.h"

#ifdef USE_CUDA

#include "spec_mtp.h"
#include "spec_mtp_couple.h"
#include "../models/qwen35moe.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* The plain decode step's top-k: every verify row is presented with exactly these pairs. */
#define CUDA_SPEC_TOPK 40

int pn_qwen35moe_cuda_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int k,
                                uint32_t *drafts)
{
    float confs[8];

    if (!model || !state || !drafts || k < 1 || k > PN_CUDA_SPEC_MAX_DRAFTS)
        return -1;
    return pn_qwen35moe_mtp_draft(model, state, seed, k, drafts, confs, NULL, NULL);
}

/* PN_MTP_RED=1: accept drafts unchecked (read once). */
static int cuda_spec_red_on(void)
{
    static int red = -1;
    if (red < 0) {
        const char *v = getenv("PN_MTP_RED");
        red = v && strcmp(v, "1") == 0;
        if (red)
            fprintf(stderr, "spec_mtp_step: RED PN_MTP_RED=1 — drafts are NOT checked\n");
    }
    return red;
}

/* Draw verify rows in order until a draw differs from the next draft (or the last row).
 * Writes the draws to emit[] and returns their count, or -1 (no row memory). Under the red
 * arm every draft is emitted as accepted (each row is still drawn, so the RNG advances). */
static int cuda_spec_draw_rows(pn_model *model, const uint32_t *toks, int n_drafts,
                               const int *ids, const float *vals, uint32_t *emit,
                               pn_sampling_params *params, uint32_t *recent, int *n_recent,
                               int repeat_window, const pn_mtp_couple *couple)
{
    const bool probe = couple && pn_mtp_couple_probe_on();
    const int n_vocab = model->config.n_vocab;
    float *row = malloc((size_t)n_vocab * sizeof(float));
    int prev[CUDA_SPEC_TOPK], n_prev = 0, n_emit = 0;

    if (!row)
        return -1;
    for (int i = 0; i < n_vocab; i++)
        row[i] = -INFINITY;
    for (int r = 0; r <= n_drafts; r++) {
        pn_mtp_present_row(row, prev, &n_prev, ids + (size_t)r * CUDA_SPEC_TOPK,
                           vals + (size_t)r * CUDA_SPEC_TOPK, CUDA_SPEC_TOPK);
        uint32_t tok = pn_mtp_draw_row(row, n_vocab, params, recent, *n_recent);
        if (r < n_drafts && cuda_spec_red_on())
            tok = toks[r + 1];
        if (probe && r < n_drafts)
            pn_mtp_couple_probe_row(couple, r, ids + (size_t)r * CUDA_SPEC_TOPK,
                                    vals + (size_t)r * CUDA_SPEC_TOPK, CUDA_SPEC_TOPK, recent,
                                    *n_recent, tok == toks[r + 1]);
        pn_mtp_recent_push(recent, n_recent, repeat_window, tok);
        emit[n_emit++] = tok;
        if (r == n_drafts || tok != toks[r + 1])
            break;
    }
    free(row);
    return n_emit;
}

/* pn_mtp_row_fetch_fn over the last verify's device logits. */
static int cuda_spec_fetch_row(void *ctx, int row, float *out)
{
    return pn_qwen35moe_spec_row_logits((pn_model *)ctx, row, out);
}

/* The grammar draw (spec_mtp.h): dense rows, the scratch-walk hook in params. Returns the
 * draws, or -1. */
static int cuda_spec_draw_rows_grammar(pn_model *model, const uint32_t *toks, int n_drafts,
                                       uint32_t *emit, pn_sampling_params *params,
                                       uint32_t *recent, int *n_recent, int repeat_window)
{
    const int n_vocab = model->config.n_vocab;
    float *row = malloc((size_t)n_vocab * sizeof(float));

    if (!row)
        return -1;
    int n_emit = pn_mtp_draw_rows_grammar(cuda_spec_fetch_row, model, toks, n_drafts, row,
                                          n_vocab, params, recent, n_recent, repeat_window,
                                          emit);
    free(row);
    return n_emit;
}

/* PN_MTP_RED=grammar_topn (gates only). */
static bool cuda_spec_red_grammar_topn(void)
{
    const char *v = getenv("PN_MTP_RED");
    return v && strcmp(v, "grammar_topn") == 0;
}

int pn_qwen35moe_cuda_spec_step(pn_model *model, pn_state *state, uint32_t seed,
                                const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                pn_sampling_params *params, uint32_t *recent, int *n_recent,
                                int repeat_window)
{
    return pn_qwen35moe_cuda_spec_step_ex(model, state, seed, drafts, n_drafts, emit, params,
                                          recent, n_recent, repeat_window, NULL);
}

int pn_qwen35moe_cuda_spec_step_ex(pn_model *model, pn_state *state, uint32_t seed,
                                   const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                   pn_sampling_params *params, uint32_t *recent, int *n_recent,
                                   int repeat_window, const pn_mtp_couple *couple)
{
    uint32_t toks[PN_CUDA_SPEC_MAX_DRAFTS + 1];
    int ids[(PN_CUDA_SPEC_MAX_DRAFTS + 1) * CUDA_SPEC_TOPK];
    float vals[(PN_CUDA_SPEC_MAX_DRAFTS + 1) * CUDA_SPEC_TOPK];

    if (!model || !state || !emit || !params || !n_recent || n_drafts < 0 ||
        n_drafts > PN_CUDA_SPEC_MAX_DRAFTS || (n_drafts > 0 && !drafts))
        return -1;
    toks[0] = seed;
    for (int i = 0; i < n_drafts; i++)
        toks[i + 1] = drafts[i];
    if (pn_qwen35moe_spec_verify_topk(model, toks, n_drafts + 1, state, CUDA_SPEC_TOPK, ids,
                                      vals) != 0)
        return -1;
    int n_emit;
    if (params->tool_grammar && !cuda_spec_red_grammar_topn()) {
        n_emit = cuda_spec_draw_rows_grammar(model, toks, n_drafts, emit, params, recent,
                                             n_recent, repeat_window);
    } else {
        const pn_tool_grammar_hook *hook = params->tool_grammar;
        params->tool_grammar = NULL;   /* RED grammar_topn: the hook-free top-k draw */
        n_emit = cuda_spec_draw_rows(model, toks, n_drafts, ids, vals, emit, params, recent,
                                     n_recent, repeat_window, couple);
        params->tool_grammar = hook;
    }
    if (n_emit < 1)
        return -1;
    if (pn_qwen35moe_spec_accept(model, state, n_emit) != 0 ||
        pn_qwen35moe_mtp_push_rows(model, state, toks, n_emit) != 0)
        return -1;
    return n_emit;
}

#endif /* USE_CUDA */
