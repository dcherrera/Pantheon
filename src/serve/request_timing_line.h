/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_timing_line.h
 * @brief One always-on `[req]` line per completed batched request: wall-clock
 *        timestamp, prompt / newly prefilled tokens, TTFT, the request's own
 *        decode rate (batched steps included), the active-slot range it decoded
 *        in, whether MTP ran, and the finalize wait.
 *
 *   [req] 2026-11-19T14:03:22.123-0600 slot=1 prompt=12345 prefill=1830
 *         ttft_ms=10234.5 gen=210 decode_ms=5123.4 tok_s=40.8 active=1-3
 *         mtp=no fin_ms=0.4 finish=tool_calls
 *
 * Cost: a few clock reads per request and one fprintf after the reply is built;
 * nothing on the per-token path.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_REQUEST_TIMING_LINE_H
#define PANTHEON_REQUEST_TIMING_LINE_H

#include <stddef.h>
#include <time.h>

/** One request's timings, filled by the batched completion path. */
typedef struct {
    int         slot;
    int         prompt_tok;    /* prompt tokens after templating */
    int         prefill_tok;   /* tokens actually prefilled (prompt - cached) */
    double      ttft_ms;       /* request arrival -> first token fed to the emitter */
    int         generated;     /* tokens generated, the first one included */
    double      decode_ms;     /* first token -> last token popped */
    int         active_min;    /* fewest / most decoding slots in this request's */
    int         active_max;    /*   steps; 0/0 = it never stepped */
    long        mtp_cycles;    /* MTP cycles run for it; 0 = MTP did not run */
    double      finalize_ms;   /* waiting for + running bs_slot_finalize */
    const char *finish;        /* finish_reason */
} sg_request_timing;

/** CLOCK_MONOTONIC in ms — the clock every field above is measured on. */
double sg_request_timing_now_ms(void);

/** Decode rate over the request's own decode time: tokens after the first
 *  (the first comes from the prefill) per second; 0 when it cannot be measured. */
double sg_request_timing_tok_s(const sg_request_timing *t);

/**
 * Format the line (no trailing newline) for wall time `wall` (local time with
 * its UTC offset, ms precision) into buf.
 * @return the snprintf length (>= cap means truncated)
 */
int sg_request_timing_format(const sg_request_timing *t, const struct timespec *wall,
                             char *buf, size_t cap);

/** Format at the current wall time and write the line to stderr. */
void sg_request_timing_log(const sg_request_timing *t);

/** Mark this connection thread's request as fully received (serve_command). */
void sg_request_arrival_mark(void);

/** This thread's arrival mark (sg_request_timing_now_ms clock), then clear it so
 *  a second completion in the same request (n > 1) does not reuse it.
 *  @return the mark, or 0.0 if none was set */
double sg_request_arrival_take(void);

#endif /* PANTHEON_REQUEST_TIMING_LINE_H */
