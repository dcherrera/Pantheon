/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_score.h
 * @brief Paired teacher-forced scoring of one accelerator logit row against the CPU oracle's
 *        — perplexity, KL and the greedy-quality gap — shared by every parity harness.
 *
 *        An argmax disagreement is not a worse token: two candidates can be near-equally
 *        good. These are the numbers that turn "they disagree" into "it is worse, by this
 *        much", and they live in one place so a gate and a debug sweep can never disagree
 *        about what a perplexity ratio means.
 *
 *   PPL   teacher-forced perplexity of the SAME token stream under each arm. Position i is
 *         scored on the actual next token, so both arms are graded against the corpus rather
 *         than against each other. Paired: the per-position difference cancels the content
 *         variance that dominates either arm's absolute figure, and a paired t on it detects
 *         a systematic loss far below the absolute spread.
 *   KL    KL(P_cpu || P_accel) in nats — how far the accelerator's whole distribution has
 *         moved, needing no reference corpus at all.
 *   GAP   logP_cpu(cpu argmax) − logP_cpu(accel argmax), the accelerator's greedy pick
 *         priced under the ORACLE's own distribution. Small means the disagreements were
 *         ties.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_PARITY_SCORE_H
#define PN_TESTS_PARITY_SCORE_H

/**
 * Is this value finite — asked of its BITS, because this tree is built `-ffast-math`.
 *
 * TWO WAYS TO GET A VACUOUS NON-FINITE CHECK, AND THIS AVOIDS BOTH. `if (x > worst)` never
 * selects a NaN, because every comparison against one is false — that is how a broken
 * chunkwise arm scored 7.775e-15 and read green. And `isfinite()` is not the fix here:
 * `-ffast-math` implies `-ffinite-math-only`, under which the compiler is entitled to fold it
 * to 1 and delete the guard, leaving a check that cannot fail. An exponent field of all ones
 * is infinity or NaN by definition and no optimisation flag can reason it away.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int parity_float_is_finite(float value);

/** The double form of parity_float_is_finite, for the gate's own derived figures. */
int parity_double_is_finite(double value);

/** Running paired score over every scored position of a run. */
typedef struct {
    double nll_accel;    /**< sum of −logP_accel(next token) */
    double nll_cpu;      /**< sum of −logP_cpu(next token) */
    double d_sum;        /**< sum of the per-position difference of those two */
    double d_sumsq;
    double kl_sum;       /**< sum of KL(P_cpu || P_accel), nats */
    double gap_sum;      /**< sum of logP_cpu(cpu argmax) − logP_cpu(accel argmax), nats */
    double gap_disagree; /**< the same, restricted to positions where the argmaxes differ */
    long   n;
    long   n_disagree;
} parity_score;

/** The derived figures of a completed score, so a caller can threshold them by name
 *  instead of re-deriving them from the sums and getting a different answer. */
typedef struct {
    double accel_ppl;
    double cpu_ppl;
    double ratio;         /**< accel_ppl / cpu_ppl — 1.0 is parity, >1 is worse */
    double dnll_mean;
    double dnll_sd;
    double t;             /**< paired t on the per-position ΔNLL */
    double kl_mean;
    double gap_mean;
    double gap_mean_disagree;
    double disagree_rate; /**< fraction of scored positions whose argmaxes differ */
    long   n;
    long   n_disagree;
} parity_summary;

/** log sum exp of a logit row, taken against the row max so a 130k-wide row cannot
 *  overflow. */
double parity_row_logsumexp(const float *row, int n);

/**
 * KL(P_cpu || P_accel) in nats, in one pass and without materialising either distribution:
 * sum p_c (x_c − lse_c − x_a + lse_a) = (lse_a − lse_c) + sum p_c (x_c − x_a).
 */
double parity_row_kl(const float *cpu, const float *accel, int n, double lse_c, double lse_a);

/** Index of the largest and second-largest element of `row`. The gap between them is the
 *  margin a divergence has to cross, which is what tells a near-tie from a disagreement. */
void parity_top2_indices(const float *row, int n, int *first, int *second);

/**
 * Score one position against the token that actually follows it.
 *
 * @param next  the corpus's next token id — the label both arms are graded on
 * @param a1    the accelerator arm's argmax, @param c1 the CPU oracle's
 * @param out   optional [2]: per-position NLL of the accelerator and the oracle
 */
void parity_score_add(parity_score *s, const float *accel, const float *cpu, int n, int next,
                      int a1, int c1, double *out);

/** Fold a per-sequence score into a running total. */
void parity_score_merge(parity_score *total, const parity_score *add);

/** Derive the reportable figures. Zero-filled when nothing was scored. */
void parity_score_summarise(const parity_score *s, parity_summary *out);

/** One summary line on stdout, prefixed "PPL". `label` distinguishes a per-sequence
 *  roll-up from the grand total. */
void parity_score_report(const parity_score *s, const char *label);

#endif /* PN_TESTS_PARITY_SCORE_H */
