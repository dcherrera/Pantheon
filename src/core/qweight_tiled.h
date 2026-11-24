/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qweight_tiled.h
 * @brief  Matvec / matmul over a PN_QUANT_TILED_* weight (foundry FTTW tiles): the
 *         card-format oracle's compute, decoding each 32-row tile column once per call.
 *
 * Numerics: every output is one pn_dot_f32 of a decoded fp32 row with the input row — the
 * same dot the dense types use — so the oracle differs from a Q8_0 run only by the weight
 * VALUES the tiles hold.
 *
 * @status None
 * @issues Rows must be a multiple of 32 (whole tile columns); others print and write nothing.
 * @todo   None
 */
#ifndef PN_QWEIGHT_TILED_H
#define PN_QWEIGHT_TILED_H

#include "qweight.h"

#ifdef __cplusplus
extern "C" {
#endif

struct fw_pool;

/**
 * out[b * rows + r] = row r of `w` dotted with acts[b * cols ...], for b < B.
 *
 * @param pool  splits tile columns over the pool when non-NULL; NULL runs serially
 *
 * Thread-safe: no (uses the pool)    Reentrant: no    Signal-safe: no
 */
void pn_qmm_tiled(const pn_qweight *w, const float *acts, int B, float *out, struct fw_pool *pool);

#ifdef __cplusplus
}
#endif

#endif /* PN_QWEIGHT_TILED_H */
