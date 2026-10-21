/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_attention_trace.c
 * @brief Implementation of the per-head attention-shape capture declared in
 *        llama_attention_trace.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "llama_attention_trace.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/** The armed capture, or NULL. One slot: the only caller is a single-threaded harness. */
static pn_attention_trace *g_armed_attn_trace = NULL;

int pn_attention_trace_alloc(pn_attention_trace *trace, int n_layer, int n_head)
{
    if (!trace || n_layer <= 0 || n_head <= 0)
        return -1;
    memset(trace, 0, sizeof(*trace));
    trace->heads = calloc((size_t)n_layer * (size_t)n_head, sizeof(*trace->heads));
    trace->normed_rms = calloc((size_t)n_layer, sizeof(*trace->normed_rms));
    if (!trace->heads || !trace->normed_rms) {
        pn_attention_trace_free(trace);
        return -1;
    }
    trace->n_layer = n_layer;
    trace->n_head = n_head;
    return 0;
}

void pn_attention_trace_free(pn_attention_trace *trace)
{
    if (!trace)
        return;
    if (g_armed_attn_trace == trace)
        g_armed_attn_trace = NULL;
    free(trace->heads);
    free(trace->normed_rms);
    memset(trace, 0, sizeof(*trace));
}

void pn_attention_trace_capture_into(pn_attention_trace *trace)
{
    g_armed_attn_trace = trace;
    if (trace) {
        trace->recorded = 0;
        trace->seq_len = 0;
    }
}

int pn_attention_trace_armed(void)
{
    return g_armed_attn_trace != NULL;
}

/** Mean of the squares of `v`. */
static double mean_square(const float *v, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += (double)v[i] * (double)v[i];
    return n > 0 ? sum / (double)n : 0.0;
}

/**
 * Per-component mean square of each cached V row of ONE kv head, summed against the
 * attention weights of one query head. Kept separate from the shape statistics because the
 * kv head is shared across a group of query heads and this is the expensive term.
 */
static double weighted_v_mean_square(const float *v_base, const float *weights, int seq_len,
                                     int head_dim)
{
    double acc = 0.0;

    for (int s = 0; s < seq_len; s++) {
        double p = (double)weights[s];

        if (p > 0.0)
            acc += p * mean_square(v_base + (size_t)s * (size_t)head_dim, head_dim);
    }
    return acc;
}

/** Entropy, collision and peak of one already-normalised weight vector. */
static void weight_shape(const float *weights, int seq_len, pn_attn_head_record *rec)
{
    double entropy = 0.0, collision = 0.0, peak = 0.0;
    int peak_at = 0;

    for (int s = 0; s < seq_len; s++) {
        double p = (double)weights[s];

        if (p > 0.0)
            entropy -= p * log(p);
        collision += p * p;
        if (p > peak) {
            peak = p;
            peak_at = s;
        }
    }
    rec->top_pos_rel = seq_len > 1 ? (double)peak_at / (double)(seq_len - 1) : 0.0;
    rec->self_weight = (double)weights[seq_len - 1];
    rec->entropy = entropy;
    rec->collision = collision;
    rec->max_weight = peak;
    rec->sink_weight = (double)weights[0];
}

void pn_attention_trace_record(int layer, const pn_attention_trace_layer *in)
{
    pn_attention_trace *trace = g_armed_attn_trace;

    if (!trace || !in || layer < 0 || layer >= trace->n_layer || in->n_head != trace->n_head)
        return;
    if (!in->scores || !in->attn_out || !in->v_cache || in->seq_len <= 0)
        return;

    for (int h = 0; h < in->n_head; h++) {
        pn_attn_head_record *rec = &trace->heads[(size_t)layer * (size_t)trace->n_head + h];
        const float *weights = in->scores + (size_t)h * (size_t)in->max_seq;
        const float *v_base = in->v_cache + (size_t)(h / in->n_kv_groups) *
                                                (size_t)in->max_seq * (size_t)in->head_dim;

        weight_shape(weights, in->seq_len, rec);
        rec->out_rms = sqrt(mean_square(in->attn_out + (size_t)h * (size_t)in->head_dim,
                                        in->head_dim));
        rec->vbar_rms = sqrt(weighted_v_mean_square(v_base, weights, in->seq_len, in->head_dim));
        rec->v0_rms = sqrt(mean_square(v_base, in->head_dim));
    }
    if (in->normed)
        trace->normed_rms[layer] = sqrt(mean_square(in->normed, in->n_embd));
    trace->seq_len = in->seq_len;
    if (layer + 1 > trace->recorded)
        trace->recorded = layer + 1;
}

const pn_attn_head_record *pn_attention_trace_head(const pn_attention_trace *trace, int layer,
                                                   int head)
{
    if (!trace || !trace->heads || layer < 0 || layer >= trace->recorded || head < 0 ||
        head >= trace->n_head)
        return NULL;
    return &trace->heads[(size_t)layer * (size_t)trace->n_head + head];
}
