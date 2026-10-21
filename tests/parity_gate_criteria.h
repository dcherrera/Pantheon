/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_gate_criteria.h
 * @brief The accelerator parity gate's THRESHOLDS and its verdict, in one place so an
 *        interleaved gate and a two-pass gate can never disagree about what a passing arm is.
 *
 * The derivation of every number below is in test_chat_parity_gate.c's header, which is where
 * the calibration ladder was run. They are repeated here as named constants and nowhere else:
 * a magic 1.05 in an `if` is a threshold nobody can find when they need to argue with it, and
 * two copies of it are two thresholds waiting to drift apart.
 *
 * Summary of where they come from, all measured on this card:
 *
 *   ratio <= 1.05          identity control is exactly 1.000000; the card on plain prose is
 *                          1.0036 and 0.9937; iid logit noise costs 1.0091 at sigma 0.2 and
 *                          1.0902 at 0.4; the chat-framing defect is 1.2207.
 *   t <= 4.0               one-sided paired t on the per-position dNLL — only being worse
 *                          counts. Card on plain prose +1.69 / -1.47; the defect +6.76.
 *   disagree <= 8%         plain prose 3.5% and 2.6%; the defect 16.9%. MEASURED AS THE
 *                          BINDING criterion, not the loosest — at sigma 0.2 it fails while
 *                          perplexity is still comfortably inside 1.05.
 *   n >= 200               "not enough evidence" is not a pass.
 *   worst document <= 1.05 a serve does not average over a corpus; a user sends one prompt.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_PARITY_GATE_CRITERIA_H
#define PN_TESTS_PARITY_GATE_CRITERIA_H

#include "parity_score.h"

#define PARITY_GATE_MAX_PPL_RATIO     1.05
#define PARITY_GATE_MAX_PAIRED_T      4.0
#define PARITY_GATE_MAX_DISAGREE_RATE 0.08
#define PARITY_GATE_MIN_SCORED        200
#define PARITY_GATE_MAX_DOC_PPL_RATIO 1.05
#define PARITY_GATE_MIN_DOC_SCORED    100
#define PARITY_GATE_MAX_DOCUMENTS     64

/**
 * The worst per-document perplexity ratio among documents long enough to threshold.
 *
 * Documents shorter than PARITY_GATE_MIN_DOC_SCORED are reported by the caller but not
 * thresholded, because a handful of positions is not evidence either way.
 *
 * @param worst_doc  set to that document's index, or -1 when none was long enough
 * @return           the worst ratio, or 0.0 when there was no eligible document
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
double parity_gate_worst_document_ratio(const parity_score *per_doc, int docs, int *worst_doc);

/**
 * Apply every criterion, print one line per criterion and the VERDICT line.
 *
 * @param identity      when set, every figure must additionally be exactly zero-divergence —
 *                      a harness whose identity control is not exactly zero is measuring
 *                      itself and no accelerator number from it can be believed
 * @param noise_sigma   the injected perturbation, for the report line; a positive value also
 *                      SKIPS the exact-zero assertion, because a non-zero result is the point
 * @param framing       "chat" or "plain", for the report line
 * @return              0 pass, 1 fail
 *
 * Thread-safe: no (writes stdout)   Reentrant: no   Signal-safe: no
 */
int parity_gate_verdict(const parity_summary *m, const parity_score *per_doc, int docs,
                        int identity, double noise_sigma, const char *framing);

#endif /* PN_TESTS_PARITY_GATE_CRITERIA_H */
