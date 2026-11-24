/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats_xtx.h
 * @brief  One tensor's X^T X accumulator for PN_ACT_STATS (act_stats.h): activation rows are
 *         buffered and flushed as one rank-k update, into f64 or f32 accumulators, and the
 *         result is written as the "mixw imatrix v1" X^T X payload (f64, act_stats_file.h)
 *         strictly sequentially, row by row.
 *
 * WHY BATCHED. A rank-1 update per row is n^2 / 2 multiply-adds per token through a scalar
 * loop; for ffn_down (n = 17,408) over ~17.7k tokens and 64 layers that is days. Rows are
 * buffered (`batch_rows`, 128 by default) and flushed with cblas_dsyrk / cblas_ssyrk under
 * USE_BLAS. Without BLAS (or with PN_ACT_XTX_SCALAR) the flush runs the original scalar
 * upper-triangle loop row by row, so a no-BLAS build accumulates exactly as before.
 *
 * F32 ACCUMULATORS (PN_ACT_XTX_F32) halve the n^2 buffer (1.21 GB instead of 2.42 GB for
 * ffn_down). The file stays f64: each row is widened while it is written.
 *
 * THE DIAGONAL is accumulated apart, sequentially in f64, with the imatrix's own expression,
 * and replaces the accumulator's diagonal when the matrix is finished. So diag(X^T X) equals
 * the imatrix's sum x^2 bit for bit at either precision and in either flush path.
 *
 * MEMORY. n^2 accumulators + batch_rows x n buffered rows + n f64 (the diagonal). Nothing
 * else of order n^2 is allocated, at write time included: the lower triangle is mirrored in
 * place, then rows go out through one n-double row buffer.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_ACT_STATS_XTX_H
#define PN_ACT_STATS_XTX_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Rows buffered between flushes when the caller passes 0. */
#define PN_ACT_XTX_BATCH_ROWS 128u

/** f32 accumulators (default f64). */
#define PN_ACT_XTX_F32 0x1u
/** The scalar rank-1 loop even when built with BLAS (what a no-BLAS build always runs). */
#define PN_ACT_XTX_SCALAR 0x2u

/** One tensor's accumulator. Fields are read-only to callers. */
typedef struct pn_act_xtx {
    char     *name;
    uint32_t  n_in;
    uint64_t  count;     /**< rows accumulated (buffered rows included) */
    unsigned  flags;
    uint32_t  batch_rows;
    uint32_t  buffered;  /**< rows waiting in `rows` */
    int       finished;  /**< flushed, diagonal set and mirrored: no more rows */
    void     *acc;       /**< n_in^2 f64 or f32, row-major; upper triangle until finished */
    void     *rows;      /**< batch_rows x n_in, f64 (f64 mode) or f32 */
    double   *diag;      /**< n_in, sequential f64 sum x^2 */
} pn_act_xtx;

/**
 * Allocate a zeroed accumulator.
 *
 * @param x           receives the accumulator (overwritten)
 * @param name        tensor name, copied
 * @param n_in        row length (the weight's input dimension)
 * @param batch_rows  rows a flush; 0 = PN_ACT_XTX_BATCH_ROWS
 * @param flags       PN_ACT_XTX_F32 | PN_ACT_XTX_SCALAR
 * @return 0, or -1 (allocation failure; stderr names the tensor and size)
 *
 * Thread-safe: yes (distinct x)   Reentrant: yes   Signal-safe: no
 */
int pn_act_xtx_init(pn_act_xtx *x, const char *name, uint32_t n_in, uint32_t batch_rows,
                    unsigned flags);

/**
 * Add `count` rows of n_in floats (contiguous). Flushes whenever the buffer fills.
 * Ignored once the accumulator is finished.
 *
 * Thread-safe: yes (distinct x)   Reentrant: yes   Signal-safe: no
 */
void pn_act_xtx_add_rows(pn_act_xtx *x, const float *rows, uint32_t count);

/**
 * Flush the buffered remainder, set the diagonal and mirror the upper triangle onto the lower
 * one, in place. Idempotent; no rows are accepted afterwards.
 *
 * Thread-safe: yes (distinct x)   Reentrant: yes   Signal-safe: no
 */
void pn_act_xtx_finish(pn_act_xtx *x);

/**
 * Element (i, j) as f64. Before pn_act_xtx_finish() only j >= i of the flushed rows is set.
 *
 * Thread-safe: yes (read only)   Reentrant: yes   Signal-safe: yes
 */
double pn_act_xtx_at(const pn_act_xtx *x, uint32_t i, uint32_t j);

/**
 * Finish, then write the payload (n_in^2 f64, row-major, little-endian) to `f`, sequentially.
 *
 * @return 0, or -1 (allocation or write failure; stderr)
 *
 * Thread-safe: yes (distinct x)   Reentrant: yes   Signal-safe: no
 */
int pn_act_xtx_write_payload(pn_act_xtx *x, FILE *f);

/**
 * Free everything and zero `x`.
 *
 * Thread-safe: yes (distinct x)   Reentrant: yes   Signal-safe: no
 */
void pn_act_xtx_free(pn_act_xtx *x);

#ifdef __cplusplus
}
#endif

#endif /* PN_ACT_STATS_XTX_H */
