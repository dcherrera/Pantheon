/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_host_gap.h
 * @brief HOST GAP clock (`PN_TT_HOST_GAP=1`, timing only): the card time the host's own
 *        decisions put in a decode cycle.
 *
 * A gap opens when the host has just waited for EVERY piece of work it handed the card (a
 * stream boundary's flush, a stream's end, a replay segment's await, a synchronous puller op):
 * from then on the card has nothing to do. It closes at the next hand-off (a PULL enqueued).
 * Each gap is booked under "<phase>:<why>": the phase the model code named last (verify, mtp,
 * plain, ...) and why the card went idle (the boundary's access class, "end" of a stream, ...).
 * The sum is the card-idle time an on-card token loop could remove; nothing else is.
 *
 * @status measurement only; no effect unless `PN_TT_HOST_GAP=1`.
 * @issues the gap opens when the host OBSERVES completion, so a poll's latency is not in it
 *         (an under-estimate by one poll per gap).
 * @todo None
 */
#ifndef PN_TT_HOST_GAP_H
#define PN_TT_HOST_GAP_H

#include <stdbool.h>

/** true when `PN_TT_HOST_GAP=1`. */
bool pn_tt_host_gap_enabled(void);

/** The model code's current phase (a string literal; kept by pointer). */
void pn_tt_host_gap_phase(const char *phase);

/** The host has waited for all card work: the card is idle from now, because of `why`
 *  (a string literal). */
void pn_tt_host_gap_card_idle(const char *why);

/** The host just handed the card work: closes an open gap. */
void pn_tt_host_gap_card_busy(void);

/** Splits an OPEN gap: the time so far stays under its label, the rest goes under
 *  "<phase>:<why>" (both string literals). No effect when the card is busy. */
void pn_tt_host_gap_mark(const char *phase, const char *why);

/** `pn_tt_host_gap_mark` under the current phase: splits an open gap at a host step. */
void pn_tt_host_gap_split(const char *why);

/** RED for the clock itself (`PN_TT_HOST_GAP_RED_US=<us>`): spins that long on the host. Put
 *  where the card is idle, the matching gap must grow by it; nothing else changes. */
void pn_tt_host_gap_red_delay(void);

/** Prints the gaps per `divisor` (cycles or tokens) under `label`. */
void pn_tt_host_gap_report(const char *label, double divisor);

#endif /* PN_TT_HOST_GAP_H */
