/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats_xtx.c
 * @brief  Implementation of the batched X^T X accumulator (act_stats_xtx.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "act_stats_xtx.h"

#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#include "act_stats_file.h"

/* Square blocks the in-place mirror walks, so the transposed reads stay in cache. */
#define XTX_MIRROR_BLOCK 64u

static int xtx_is_f32(const pn_act_xtx *x) {
    return (x->flags & PN_ACT_XTX_F32) != 0;
}

int pn_act_xtx_init(pn_act_xtx *x, const char *name, uint32_t n_in, uint32_t batch_rows,
                    unsigned flags) {
    memset(x, 0, sizeof(*x));
    x->n_in = n_in;
    x->flags = flags;
    x->batch_rows = batch_rows ? batch_rows : PN_ACT_XTX_BATCH_ROWS;
    const size_t elem = xtx_is_f32(x) ? sizeof(float) : sizeof(double);
    const size_t n = n_in;
    x->name = strdup(name ? name : "");
    x->acc = n ? calloc(n * n, elem) : NULL;
    x->rows = n ? malloc((size_t)x->batch_rows * n * elem) : NULL;
    x->diag = n ? calloc(n, sizeof(double)) : NULL;
    if (!x->name || !x->acc || !x->rows || !x->diag) {
        fprintf(stderr, "act_stats: %s: cannot allocate a %u x %u %s X^T X\n", name ? name : "?",
                n_in, n_in, xtx_is_f32(x) ? "f32" : "f64");
        pn_act_xtx_free(x);
        return -1;
    }
    return 0;
}

/* The original rank-1 loop, one buffered row at a time: upper triangle, j >= i. */
static void xtx_flush_scalar(pn_act_xtx *x) {
    const uint32_t n = x->n_in;
    for (uint32_t r = 0; r < x->buffered; r++) {
        if (xtx_is_f32(x)) {
            const float *row = (const float *)x->rows + (size_t)r * n;
            for (uint32_t i = 0; i < n; i++) {
                float xi = row[i], *dst = (float *)x->acc + (size_t)i * n;
                for (uint32_t j = i; j < n; j++)
                    dst[j] += xi * row[j];
            }
        } else {
            const double *row = (const double *)x->rows + (size_t)r * n;
            for (uint32_t i = 0; i < n; i++) {
                double xi = row[i], *dst = (double *)x->acc + (size_t)i * n;
                for (uint32_t j = i; j < n; j++)
                    dst[j] += xi * row[j];
            }
        }
    }
}

static void xtx_flush(pn_act_xtx *x) {
    if (x->buffered == 0)
        return;
#ifdef USE_BLAS
    if (!(x->flags & PN_ACT_XTX_SCALAR)) {
        /* C (n x n, upper) += A^T A, A = the buffered rows (k x n, row-major). */
        const int n = (int)x->n_in, k = (int)x->buffered;
        if (xtx_is_f32(x))
            cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, n, k, 1.0f, (const float *)x->rows,
                        n, 1.0f, (float *)x->acc, n);
        else
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n, k, 1.0, (const double *)x->rows,
                        n, 1.0, (double *)x->acc, n);
        x->buffered = 0;
        return;
    }
#endif
    xtx_flush_scalar(x);
    x->buffered = 0;
}

void pn_act_xtx_add_rows(pn_act_xtx *x, const float *rows, uint32_t count) {
    const uint32_t n = x->n_in;
    if (x->finished || !x->acc)
        return;
    for (uint32_t r = 0; r < count; r++) {
        const float *src = rows + (size_t)r * n;
        /* The imatrix's own expression (act_stats.c), so the diagonal matches it exactly. */
        for (uint32_t j = 0; j < n; j++)
            x->diag[j] += (double)src[j] * (double)src[j];
        if (xtx_is_f32(x)) {
            memcpy((float *)x->rows + (size_t)x->buffered * n, src, (size_t)n * sizeof(float));
        } else {
            double *dst = (double *)x->rows + (size_t)x->buffered * n;
            for (uint32_t j = 0; j < n; j++)
                dst[j] = src[j];
        }
        if (++x->buffered == x->batch_rows)
            xtx_flush(x);
    }
    x->count += count;
}

/* Lower triangle := upper, in place, block by block. */
static void xtx_mirror(pn_act_xtx *x) {
    const size_t n = x->n_in;
    for (size_t bi = 0; bi < n; bi += XTX_MIRROR_BLOCK) {
        const size_t ei = bi + XTX_MIRROR_BLOCK < n ? bi + XTX_MIRROR_BLOCK : n;
        for (size_t bj = 0; bj <= bi; bj += XTX_MIRROR_BLOCK) {
            for (size_t i = bi; i < ei; i++) {
                const size_t ej = bj + XTX_MIRROR_BLOCK < i ? bj + XTX_MIRROR_BLOCK : i;
                for (size_t j = bj; j < ej; j++) {
                    if (xtx_is_f32(x))
                        ((float *)x->acc)[i * n + j] = ((float *)x->acc)[j * n + i];
                    else
                        ((double *)x->acc)[i * n + j] = ((double *)x->acc)[j * n + i];
                }
            }
        }
    }
}

void pn_act_xtx_finish(pn_act_xtx *x) {
    if (x->finished || !x->acc)
        return;
    xtx_flush(x);
    for (size_t i = 0; i < x->n_in; i++) {
        if (xtx_is_f32(x))
            ((float *)x->acc)[i * x->n_in + i] = (float)x->diag[i];
        else
            ((double *)x->acc)[i * x->n_in + i] = x->diag[i];
    }
    xtx_mirror(x);
    x->finished = 1;
}

double pn_act_xtx_at(const pn_act_xtx *x, uint32_t i, uint32_t j) {
    const size_t at = (size_t)i * x->n_in + j;
    if (x->finished && i == j)
        return x->diag[i];
    return xtx_is_f32(x) ? (double)((const float *)x->acc)[at] : ((const double *)x->acc)[at];
}

int pn_act_xtx_write_payload(pn_act_xtx *x, FILE *f) {
    const size_t n = x->n_in;
    double *row = NULL;
    int rc = 0;

    pn_act_xtx_finish(x);
    if (!xtx_is_f32(x)) {
        for (size_t i = 0; i < n && rc == 0; i++)
            rc = pn_act_stats_write_f64(f, (const double *)x->acc + i * n, n);
        return rc;
    }
    if (!(row = malloc(n * sizeof(*row)))) {
        fprintf(stderr, "act_stats: %s: no row buffer for the write\n", x->name);
        return -1;
    }
    for (size_t i = 0; i < n && rc == 0; i++) {
        const float *src = (const float *)x->acc + i * n;
        for (size_t j = 0; j < n; j++)
            row[j] = src[j];
        row[i] = x->diag[i];
        rc = pn_act_stats_write_f64(f, row, n);
    }
    free(row);
    return rc;
}

void pn_act_xtx_free(pn_act_xtx *x) {
    free(x->name);
    free(x->acc);
    free(x->rows);
    free(x->diag);
    memset(x, 0, sizeof(*x));
}
