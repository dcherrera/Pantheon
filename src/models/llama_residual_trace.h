/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_residual_trace.h
 * @brief Capture the residual stream layer by layer out of ANY llama forward, so two
 *        implementations can be compared on their ACCUMULATED STATE rather than on the
 *        local error of each op.
 *
 *        The per-op trace in llama_forward_tt.c feeds the card's own input to the CPU
 *        reference and scores each op in isolation. That measures per-op error and is blind
 *        to the divergence of the stream those ops write into: the ChatML defect shows the
 *        same per-op error as plain prose (1.06x) with five times the end-to-end logit
 *        divergence, which is only visible here.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_LLAMA_RESIDUAL_TRACE_H
#define PN_LLAMA_RESIDUAL_TRACE_H

/**
 * One forward's residual stream. Row L (0 <= L < n_layer) is the hidden state AFTER layer
 * L's FFN residual add; row n_layer is the final normed vector the LM head reads.
 */
typedef struct {
    float *rows;     /**< n_rows * width, owned */
    int    n_rows;
    int    width;
    int    recorded; /**< how many rows the forward now running has written */
} pn_residual_trace;

/**
 * Allocate a capture buffer.
 *
 * @param n_rows  layers + 1 — the extra row is the final normed vector
 * @param width   n_embd
 * @return        0 on success, -1 on allocation failure
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int pn_residual_trace_alloc(pn_residual_trace *trace, int n_rows, int width);

void pn_residual_trace_free(pn_residual_trace *trace);

/**
 * Direct the next forward's per-layer records into `trace`, or NULL to stop capturing.
 * Resets `trace->recorded` so a capture is always one forward's worth.
 *
 * Deliberately a single process-wide slot rather than a parameter on the forward: the
 * forward signature is the engine's public contract and a debug instrument does not get to
 * change it. The one caller is a debug harness that runs one forward at a time.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
void pn_residual_trace_capture_into(pn_residual_trace *trace);

/**
 * Record one row. Called by a forward implementation; a no-op — one predictable
 * branch on a global — when no capture is armed, which is every production call.
 *
 * @param row     row index; ignored if outside the armed buffer
 * @param hidden  the vector to copy, `width` floats
 */
void pn_residual_trace_record(int row, const float *hidden, int width);

/** @return whether a capture is armed, so a forward can skip work it only does to record. */
int pn_residual_trace_armed(void);

/** @return row `row` of a completed capture, or NULL if it was never written. */
const float *pn_residual_trace_row(const pn_residual_trace *trace, int row);

#endif /* PN_LLAMA_RESIDUAL_TRACE_H */
