/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_score.c
 * @brief Implementation of the paired teacher-forced scorer declared in parity_score.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "parity_score.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int parity_float_is_finite(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return ((bits >> 23) & 0xFFu) != 0xFFu;
}

int parity_double_is_finite(double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return ((bits >> 52) & 0x7FFu) != 0x7FFu;
}

double parity_row_logsumexp(const float *row, int n)
{
    double max = row[0], sum = 0.0;

    for (int j = 1; j < n; j++)
        if ((double)row[j] > max) max = (double)row[j];
    for (int j = 0; j < n; j++)
        sum += exp((double)row[j] - max);
    return max + log(sum);
}

double parity_row_kl(const float *cpu, const float *accel, int n, double lse_c, double lse_a)
{
    double acc = 0.0;

    for (int j = 0; j < n; j++) {
        double p = exp((double)cpu[j] - lse_c);

        acc += p * ((double)cpu[j] - (double)accel[j]);
    }
    return acc + (lse_a - lse_c);
}

void parity_top2_indices(const float *row, int n, int *first, int *second)
{
    int a = 0, b = -1;

    for (int j = 1; j < n; j++)
        if (row[j] > row[a]) a = j;
    for (int j = 0; j < n; j++) {
        if (j == a) continue;
        if (b < 0 || row[j] > row[b]) b = j;
    }
    *first = a;
    *second = b;
}

void parity_score_add(parity_score *s, const float *accel, const float *cpu, int n, int next,
                      int a1, int c1, double *out)
{
    double lse_a = parity_row_logsumexp(accel, n);
    double lse_c = parity_row_logsumexp(cpu, n);
    double nll_a = lse_a - (double)accel[next];
    double nll_c = lse_c - (double)cpu[next];
    double gap = (double)cpu[c1] - (double)cpu[a1]; /* the lse cancels */

    s->nll_accel += nll_a;
    s->nll_cpu += nll_c;
    s->d_sum += nll_a - nll_c;
    s->d_sumsq += (nll_a - nll_c) * (nll_a - nll_c);
    s->kl_sum += parity_row_kl(cpu, accel, n, lse_c, lse_a);
    s->gap_sum += gap;
    if (a1 != c1) {
        s->gap_disagree += gap;
        s->n_disagree++;
    }
    s->n++;
    if (out) {
        out[0] = nll_a;
        out[1] = nll_c;
    }
}

void parity_score_merge(parity_score *total, const parity_score *add)
{
    total->nll_accel += add->nll_accel;
    total->nll_cpu += add->nll_cpu;
    total->d_sum += add->d_sum;
    total->d_sumsq += add->d_sumsq;
    total->kl_sum += add->kl_sum;
    total->gap_sum += add->gap_sum;
    total->gap_disagree += add->gap_disagree;
    total->n += add->n;
    total->n_disagree += add->n_disagree;
}

void parity_score_summarise(const parity_score *s, parity_summary *out)
{
    double mean_d, var, sd;

    if (!out)
        return;
    if (!s || s->n <= 0) {
        parity_summary empty = {0};

        *out = empty;
        return;
    }
    mean_d = s->d_sum / (double)s->n;
    var = s->n > 1 ? (s->d_sumsq - (double)s->n * mean_d * mean_d) / (double)(s->n - 1) : 0.0;
    sd = var > 0.0 ? sqrt(var) : 0.0;
    out->accel_ppl = exp(s->nll_accel / (double)s->n);
    out->cpu_ppl = exp(s->nll_cpu / (double)s->n);
    out->ratio = out->accel_ppl / out->cpu_ppl;
    out->dnll_mean = mean_d;
    out->dnll_sd = sd;
    out->t = sd > 0.0 ? mean_d / (sd / sqrt((double)s->n)) : 0.0;
    out->kl_mean = s->kl_sum / (double)s->n;
    out->gap_mean = s->gap_sum / (double)s->n;
    out->gap_mean_disagree = s->n_disagree > 0 ? s->gap_disagree / (double)s->n_disagree : 0.0;
    out->disagree_rate = (double)s->n_disagree / (double)s->n;
    out->n = s->n;
    out->n_disagree = s->n_disagree;
}

void parity_score_report(const parity_score *s, const char *label)
{
    parity_summary m;

    if (!s || s->n <= 0) {
        printf("PPL %s n=0\n", label);
        fflush(stdout);
        return;
    }
    parity_score_summarise(s, &m);
    printf("PPL %s n=%ld accel_ppl=%.6f cpu_ppl=%.6f ratio=%.6f "
           "dnll_mean=%.6g dnll_sd=%.6g t=%.3f kl_mean=%.6g "
           "gap_mean=%.6g disagree=%ld gap_mean_disagree=%.6g\n",
           label, m.n, m.accel_ppl, m.cpu_ppl, m.ratio, m.dnll_mean, m.dnll_sd, m.t, m.kl_mean,
           m.gap_mean, m.n_disagree, m.gap_mean_disagree);
    fflush(stdout);
}
