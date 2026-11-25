/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_step_ledger.h
 * @brief  `QWEN35_STEP_LEDGER=<file>`: one line per two-stage (TP2 x 2) decode step that splits
 *         its wall time into GPU-busy time and host time, by phase.
 *
 * WHY. A decode step is two stage graphs on two die pairs, a host-staged hand-off between
 * them, the LM head and the logits read-back. Whatever part of the step's wall time no die
 * is executing is host cost: launch latency, sync wake-up, host-side copies and staging.
 * This prices it per phase, so a lever (graphs, fewer syncs, a faster host) can be judged
 * by the time it could actually remove.
 *
 * HOW. Host phases are CLOCK_MONOTONIC laps on the step's thread. GPU spans come from
 * timing events recorded on the streams OUTSIDE the captured graphs (before and after each
 * stage graph, around the head), so the graphs themselves are not perturbed. The call census
 * (launches, syncs, memcpys and their bytes) is diffed per step and between steps.
 * Unset, every hook is one predictable branch.
 *
 * Line format (tab-separated key=value, one step a line):
 *   ts n pd pmin pmax graph wall period s0 s1 head  <host phases>  <census in step>  <census off step>
 * GPU-busy = s0 + s1 + head; host-in-step = wall - GPU-busy; off-step = period - wall.
 *
 * `QWEN35_STEP_LEDGER_INJECT=<point>:<us>[,<point>:<us>]` adds host sleeps inside the step
 * (the harness check). `hop` sits on the critical path (after stage 0 drains): it must land in
 * the hop phase and in wall. `shadow` sits under the running stage 0: it lands in mirror0,
 * wait0 shrinks by the same, and wall must not move.
 *
 * @status Instrument only.
 * @issues Single writer: the step runs inside the model's GPU turn.
 * @todo   None
 */
#ifndef PN_QWEN35MOE_STEP_LEDGER_H
#define PN_QWEN35MOE_STEP_LEDGER_H

#ifdef USE_CUDA

#include "cuda/fc_cuda.h"

/** Host phases of a two-stage decode step, in step order. */
typedef enum {
    QW_LH_SETUP0 = 0,   /* exchange check, scratch, position uploads, embed, hidden upload, prep */
    QW_LH_LAUNCH0,      /* stage 0 graph launch call (or capture / direct issue) */
    QW_LH_MIRROR0,      /* host work under stage 0: KV-mirror marks, stage 1 scratch + positions */
    QW_LH_WAIT0,        /* host blocked until stage 0 drains */
    QW_LH_HOP,          /* the board hand-off: hidden D2H + two H2D (host-staged) */
    QW_LH_SETUP1,       /* stage 1 prep */
    QW_LH_LAUNCH1,      /* stage 1 graph launch call */
    QW_LH_MIRROR1,      /* stage 1 host KV-mirror bookkeeping + overlap pieces */
    QW_LH_HEAD_ISSUE,   /* final norm + LM head launches */
    QW_LH_PRESENT_WAIT, /* host blocked until the head drains (incl. the top-k reduction) */
    QW_LH_PRESENT_COPY, /* logits rows D2H + sparse presentation */
    QW_LH_TAIL,         /* the rest of the step (overlap pieces, return) */
    QW_LH_N
} qw_ledger_phase;

/** GPU timing points (each a timing event on one die's stream). */
typedef enum {
    QW_LG_S0_BEG = 0, QW_LG_S0_END,
    QW_LG_S1_BEG, QW_LG_S1_END,
    QW_LG_H_BEG, QW_LG_H_END,
    QW_LG_N
} qw_ledger_point;

/** True when QWEN35_STEP_LEDGER names a writable file (read once). */
int qw_ledger_on(void);

/** Start a step: stamps t0, snapshots the call census. `pd` = an overlap cycle's step.
 *  Laps and GPU points outside a begun step are ignored. */
void qw_ledger_step_begin(int nslots, const int *pos, int pd);

/** Charge now - last stamp to `phase`. */
void qw_ledger_lap(qw_ledger_phase phase);

/** Record GPU point `pt` on `ctx`'s stream (outside any capture). */
void qw_ledger_gpu(qw_ledger_point pt, fc_cuda_ctx *ctx);

/** Note how stage `s` ran: 'R' graph replay, 'C' capture + replay, 'D' direct issue. */
void qw_ledger_stage_mode(int s, char mode);

/** A captured stage graph's node census (printed once per capture). */
void qw_ledger_graph_census(int s, int nslots, const fc_graph *gr);

/** Sleep here if QWEN35_STEP_LEDGER_INJECT names this point. */
void qw_ledger_inject(const char *point);

/** End the step: GPU spans, census diff, one line out. rc < 0 drops the step. */
void qw_ledger_step_end(int rc);

#endif /* USE_CUDA */
#endif /* PN_QWEN35MOE_STEP_LEDGER_H */
