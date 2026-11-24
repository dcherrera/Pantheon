/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   batch_sched_pd_budget.h
 * @brief  How much prefill a stage-alternating overlap cycle may issue
 *         (QWEN35_BS_PD_OVERLAP): a fixed number of layer pieces per board
 *         per cycle, plus the plain-step times the stats line compares with.
 *
 *         An overlap cycle is one decode step with prefill pieces issued in
 *         each board's idle half (pn_qwen35moe_pd_cycle_cuda). A piece is one
 *         layer of one 128-token chunk: ~36 ms for a GDN layer, ~2x for an
 *         attention layer at ~10k depth. A board's idle half is only the other
 *         board's decode stage plus the head and the host's turn-around
 *         (~25-30 ms), so a piece overruns it a little, and every piece more
 *         in the same cycle overruns it in full: the next stage's decode
 *         queues behind it on that die's one stream. One piece per board per
 *         cycle keeps decode at ~0.6-0.85x of a plain step and the prefill at
 *         ~0.65-0.75x of running alone (both together ~1.35-1.48x today's
 *         50/50 time slicing); two pieces nearly double the cycle for a
 *         little more prefill (~1.2-1.3x). Measured on a K80 host: a credit budget
 *         that let 0-2 pieces per cycle ran at 1.7x the plain step and ~1.02x
 *         today.
 *
 *         Scheduling only: what a piece or a step computes never depends on
 *         the piece count, so every session's tokens are bit-identical at any
 *         setting.
 *
 * @status Complete
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_BATCH_SCHED_PD_BUDGET_H
#define PN_SERVE_BATCH_SCHED_PD_BUDGET_H

#include <stdbool.h>
#include <stdlib.h>

#define BS_PD_MAX_WIDTH        16
#define BS_PD_PIECES_DEFAULT   1
#define BS_PD_PIECES_MAX       8
/* Shortest-prefill-first: hand the turn to a waiting prefill only when it is
 * this many tokens shorter than what is left (a hand-over drains the pipeline,
 * ~one chunk-stage of bubble, plus a state save). */
#define BS_PD_SRPT_MARGIN      512

typedef struct {
    int    pieces;                         /* layer pieces per board per cycle */
    double plain_ms[BS_PD_MAX_WIDTH + 1];  /* EMA of a plain step by width (0 = none) */
} bs_pd_budget;

/* Parse QWEN35_BS_PD_PIECES: NULL/empty/non-positive = default, clamped. */
static inline int bs_pd_pieces_parse(const char *value)
{
    if (!value || !value[0])
        return BS_PD_PIECES_DEFAULT;
    int n = atoi(value);
    if (n < 1)
        return BS_PD_PIECES_DEFAULT;
    return n > BS_PD_PIECES_MAX ? BS_PD_PIECES_MAX : n;
}

static inline void bs_pd_budget_init(bs_pd_budget *b, int pieces)
{
    b->pieces = pieces;
    for (int k = 0; k <= BS_PD_MAX_WIDTH; k++)
        b->plain_ms[k] = 0.0;
}

/* A plain (no prefill) step of `width` slots took `ms`. */
static inline void bs_pd_budget_note_plain(bs_pd_budget *b, int width, double ms)
{
    if (width < 1 || width > BS_PD_MAX_WIDTH || ms <= 0.0)
        return;
    double *e = &b->plain_ms[width];
    *e = *e > 0.0 ? 0.9 * *e + 0.1 * ms : ms;
}

/* May the prefiller hand the GPU turn over before its next overlap cycle?
 * Only for a waiting finalize or a shorter prefill, and only while chunks are
 * left to inject. A hand-over retires what is in flight and lets any forward
 * take the live buffers and the batch scratch; the chunks still to come walk
 * every layer again, re-seeding each from the slot's block. Once the last
 * chunk is in, nothing would win them back: the turn checkpoint saved at the
 * range's end would find them holding another session, and the head would
 * read another prefill's rows. So the range ends under the same turn, as
 * every other prefill path does (sg_prefill_range_sched). */
static inline bool bs_pd_may_yield(bool injected_all, int finalizers,
                                   bool shorter_waiting)
{
    return !injected_all && (finalizers > 0 || shorter_waiting);
}

#endif /* PN_SERVE_BATCH_SCHED_PD_BUDGET_H */
