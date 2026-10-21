/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_gate_criteria.c
 * @brief Implementation of the shared parity-gate verdict declared in parity_gate_criteria.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "parity_gate_criteria.h"

#include <stddef.h>
#include <stdio.h>

/** One criterion's verdict, so the report can NAME what failed instead of just failing. */
typedef struct {
    const char *name;
    double      value;
    double      limit;
    int         failed;
} parity_gate_criterion;

/**
 * Fill one criterion and return whether it failed. `higher_is_worse` throughout.
 *
 * **THE TEST IS `!(value <= limit)` AND NOT `value > limit`, AND THAT IS THE WHOLE POINT.**
 * Every comparison against a NaN is false, so `value > limit` reads PASS on a figure that is
 * not a number at all — the same defect that let a broken chunkwise arm score 7.775e-15 and
 * read green (`if (difference > worst)` never accepting a NaN as the new worst). Negating the
 * ordered form makes a NaN fail instead of vanish, on every criterion at once.
 */
static int parity_gate_check(parity_gate_criterion *c, const char *name, double value,
                             double limit)
{
    c->name = name;
    c->value = value;
    c->limit = limit;
    c->failed = !(value <= limit);
    return c->failed;
}

/**
 * How many of the summary's figures are not finite.
 *
 * The criteria above are NaN-safe individually, but a run whose arithmetic produced a NaN
 * anywhere should say SO rather than fail one threshold for an unexplained reason. This
 * counts them so the report names the condition. `parity_double_is_finite` asks the BITS —
 * `isfinite()` is foldable to 1 under this tree's `-ffast-math`.
 */
static int parity_gate_non_finite_figures(const parity_summary *m)
{
    const double figures[] = {m->accel_ppl, m->cpu_ppl, m->ratio, m->dnll_mean,
                              m->dnll_sd,   m->t,       m->kl_mean};
    int count = 0;

    for (size_t i = 0; i < sizeof(figures) / sizeof(figures[0]); i++)
        if (!parity_double_is_finite(figures[i]))
            count++;
    return count;
}

double parity_gate_worst_document_ratio(const parity_score *per_doc, int docs, int *worst_doc)
{
    double worst = 0.0;

    if (!worst_doc)
        return 0.0;
    *worst_doc = -1;
    if (!per_doc)
        return 0.0;
    for (int i = 0; i < docs; i++) {
        parity_summary m;

        parity_score_summarise(&per_doc[i], &m);
        if (m.n < PARITY_GATE_MIN_DOC_SCORED)
            continue;
        if (*worst_doc < 0 || m.ratio > worst) {
            worst = m.ratio;
            *worst_doc = i;
        }
    }
    return worst;
}

int parity_gate_verdict(const parity_summary *m, const parity_score *per_doc, int docs,
                        int identity, double noise_sigma, const char *framing)
{
    parity_gate_criterion criteria[6];
    int worst_doc = -1;
    double worst_ratio = parity_gate_worst_document_ratio(per_doc, docs, &worst_doc);
    int failed = 0;

    failed |= parity_gate_check(&criteria[0], "ppl_ratio", m->ratio, PARITY_GATE_MAX_PPL_RATIO);
    failed |= parity_gate_check(&criteria[1], "paired_t", m->t, PARITY_GATE_MAX_PAIRED_T);
    failed |= parity_gate_check(&criteria[2], "disagree_rate", m->disagree_rate,
                                PARITY_GATE_MAX_DISAGREE_RATE);
    /* Inverted on purpose: too FEW scored positions is the failure, so the criterion is
     * expressed as a shortfall against the minimum and read the same way as the others. */
    failed |= parity_gate_check(&criteria[3], "scored_shortfall",
                                (double)(PARITY_GATE_MIN_SCORED - m->n), 0.0);
    failed |= parity_gate_check(&criteria[4], "worst_doc_ppl_ratio", worst_ratio,
                                PARITY_GATE_MAX_DOC_PPL_RATIO);
    /* Named rather than inferred. A gate that merely happened to fail a threshold because one
     * of its inputs was a NaN would send the reader to the wrong question. */
    failed |= parity_gate_check(&criteria[5], "non_finite_figures",
                                (double)parity_gate_non_finite_figures(m), 0.0);

    for (int i = 0; i < 6; i++)
        printf("GATE criterion %-19s value=%-12.6g limit=%-10.6g %s\n", criteria[i].name,
               criteria[i].value, criteria[i].limit, criteria[i].failed ? "FAIL" : "pass");
    printf("GATE worst_document index=%d ratio=%.6f (documents below %d scored positions are "
           "reported but not thresholded)\n", worst_doc, worst_ratio,
           PARITY_GATE_MIN_DOC_SCORED);

    /* Under injected noise the two arms differ BY CONSTRUCTION, so the exact-zero assertion is
     * not merely relaxed — it would be the wrong question. What the run is asking is whether
     * the criteria move at all, and a PASS at a sigma the thresholds were calibrated to catch
     * is the finding. */
    if (identity && noise_sigma > 0.0) {
        printf("GATE identity_control skipped (--noise %.4g is injected divergence; a "
               "non-zero result is what this arm is for)\n", noise_sigma);
        identity = 0;
    }
    if (identity) {
        int exact = m->ratio == 1.0 && m->dnll_mean == 0.0 && m->kl_mean == 0.0 &&
                    m->n_disagree == 0;

        printf("GATE identity_control exact_zero=%s\n", exact ? "yes" : "NO");
        if (!exact) {
            printf("GATE VERDICT framing=%s FAIL (identity control is not exactly zero — this "
                   "run measured the harness and the harness is broken)\n", framing);
            return 1;
        }
    }
    printf("GATE VERDICT framing=%s noise=%.4g %s (n=%ld ratio=%.6f t=%.3f disagree=%.4f "
           "kl=%.6g)\n", framing, noise_sigma, failed ? "FAIL" : "PASS", m->n, m->ratio, m->t,
           m->disagree_rate, m->kl_mean);
    fflush(stdout);
    return failed ? 1 : 0;
}
