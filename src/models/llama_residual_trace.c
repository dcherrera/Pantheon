/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_residual_trace.c
 * @brief Implementation of the per-layer residual-stream capture declared in
 *        llama_residual_trace.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "llama_residual_trace.h"

#include <stdlib.h>
#include <string.h>

/** The armed capture, or NULL. One slot: the only caller is a single-threaded harness. */
static pn_residual_trace *g_armed_trace = NULL;

int pn_residual_trace_alloc(pn_residual_trace *trace, int n_rows, int width)
{
    if (!trace || n_rows <= 0 || width <= 0)
        return -1;
    trace->rows = calloc((size_t)n_rows * (size_t)width, sizeof(float));
    if (!trace->rows)
        return -1;
    trace->n_rows = n_rows;
    trace->width = width;
    trace->recorded = 0;
    return 0;
}

void pn_residual_trace_free(pn_residual_trace *trace)
{
    if (!trace)
        return;
    if (g_armed_trace == trace)
        g_armed_trace = NULL;
    free(trace->rows);
    trace->rows = NULL;
    trace->n_rows = 0;
    trace->width = 0;
    trace->recorded = 0;
}

void pn_residual_trace_capture_into(pn_residual_trace *trace)
{
    g_armed_trace = trace;
    if (trace)
        trace->recorded = 0;
}

void pn_residual_trace_record(int row, const float *hidden, int width)
{
    pn_residual_trace *trace = g_armed_trace;

    if (!trace || !hidden || row < 0 || row >= trace->n_rows || width != trace->width)
        return;
    memcpy(trace->rows + (size_t)row * (size_t)trace->width, hidden,
           (size_t)trace->width * sizeof(float));
    if (row + 1 > trace->recorded)
        trace->recorded = row + 1;
}

int pn_residual_trace_armed(void)
{
    return g_armed_trace != NULL;
}

const float *pn_residual_trace_row(const pn_residual_trace *trace, int row)
{
    if (!trace || !trace->rows || row < 0 || row >= trace->recorded)
        return NULL;
    return trace->rows + (size_t)row * (size_t)trace->width;
}
