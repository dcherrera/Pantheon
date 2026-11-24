/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qweight.h
 * @brief  Quantized weight handle and its matvec / matmul / dot kernels.
 *
 * Written from a behaviour specification (cleanroom_r2_qweight_spec.md, recorded in docs/cleanroom/README.md); no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues - Block kernels ignore a partial trailing block; Q2_0_g128 processes it.
 *         - Ternary is unsupported through pn_qdot (stderr + 0) but works via pn_qmv.
 *         - pn_qmm / pn_qmm_prefill output is undefined for types the row dequant does
 *           not cover and for non-block-multiple cols.
 *         - pn_qweight_from_gguf ignores middle dims, never writes scales/group_size.
 * @todo   None
 */
#ifndef PN_QWEIGHT_H
#define PN_QWEIGHT_H

#include <stddef.h>

#include "../formats/gguf.h"
#include "../formats/gguf_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Largest batch pn_qmm runs through the shared-dequant path; larger B loops pn_qmv. */
#define PN_QMM_MAX_B 64

/**
 * A quantized weight matrix [rows, cols]. Plain public struct: callers build it by hand.
 *
 * data        row 0 of the quantized bytes; row r starts at data + r*row_stride
 * type        quant type of every row
 * rows        output dimension
 * cols        input dimension
 * row_stride  bytes between consecutive rows
 * scales      PN_QUANT_TERNARY only: per-group scales, scales[r*n_groups + g]
 * group_size  PN_QUANT_TERNARY only: elements per scale group
 * gpu         fm_buffer * holding the weight bytes on a GPU device, or NULL
 *             (unconditional field: the struct layout never depends on build flags)
 */
typedef struct {
    const void   *data;
    pn_gguf_quant type;
    int           rows;
    int           cols;
    size_t        row_stride;
    const float  *scales;
    int           group_size;
    void         *gpu;
} pn_qweight;

/** Kind of dispatch reported to the dispatch observer. */
typedef enum {
    PN_QWEIGHT_OP_MATVEC,
    PN_QWEIGHT_OP_MATMUL,
    PN_QWEIGHT_OP_DOT
} pn_qweight_op;

/** Observer called once per reported dispatch, on the dispatching thread. */
typedef void (*pn_qweight_dispatch_observer_fn)(const pn_qweight *w,
                                                pn_qweight_op op, void *user);

/**
 * Activation observer: called once per pn_qmv / pn_qmv_group entry / pn_qmm / pn_qmm_prefill
 * dispatch, on the dispatching thread, BEFORE the product, with the B input rows ([B, w->cols]
 * row-major) exactly as the caller passed them. Internal per-row fallbacks never re-report.
 */
typedef void (*pn_qweight_activation_observer_fn)(const pn_qweight *w, const float *acts,
                                                  int B, void *user);

struct fw_pool;

/**
 * Fill a pn_qweight from a named GGUF tensor (data points into the mmap).
 *
 * @param qw    Output handle; scales and group_size are never written
 * @param gguf  Open GGUF file
 * @param name  Tensor name (exact match)
 * @return      0 on success; -1 on NULL args, a missing tensor (stderr line), or a type
 *              that cannot be sized (silent; data/type/gpu/rows/cols already written)
 *
 * Thread-safe: yes (writes only *qw)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_qweight_from_gguf(pn_qweight *qw, pn_gguf_file *gguf, const char *name);

/**
 * out[r] = row r of w dotted with vec, for r in [0, rows).
 *
 * @param w    Weight (NULL: no-op)
 * @param vec  Input [cols] (NULL: no-op)
 * @param out  Output [rows] (NULL: no-op)
 *
 * Thread-safe: no (uses the registered pool and GPU staging buffers)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmv(const pn_qweight *w, const float *vec, float *out);

/**
 * out[i] = w[i] @ vec[i] for i < min(n, 8), in one parallel dispatch when possible.
 * Results equal per-weight pn_qmv exactly.
 *
 * @param w    Array of n weight pointers (NULL array: no-op)
 * @param vec  Array of n input vectors
 * @param out  Array of n output vectors
 * @param n    Number of weights (<= 0: no-op; > 8 is clamped to 8)
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmv_group(const pn_qweight *const *w, const float *const *vec,
                  float *const *out, int n);

/**
 * Batched matvec: out[b*rows + r] = row r of w dotted with acts[b*cols ...].
 *
 * @param w     Weight
 * @param acts  Activations [B, cols] row-major
 * @param B     Batch size (<= 0: no-op; > PN_QMM_MAX_B loops pn_qmv)
 * @param out   Output [B, rows] row-major
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmm(const pn_qweight *w, const float *acts, int B, float *out);

/**
 * Prefill matmul: as pn_qmm with no B cap and panels of at least 64 rows.
 *
 * @param w     Weight
 * @param acts  Activations [B, cols] row-major
 * @param B     Batch size (<= 0: no-op)
 * @param out   Output [B, rows] row-major
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmm_prefill(const pn_qweight *w, const float *acts, int B, float *out);

/**
 * Install (or clear, fn = NULL) the process-wide dispatch observer.
 *
 * @param fn    Observer, or NULL to disable
 * @param user  Passed back to fn on every call (stored regardless of fn)
 *
 * Thread-safe: no (call before the forward)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qweight_set_dispatch_observer(pn_qweight_dispatch_observer_fn fn, void *user);

/**
 * Report a dispatch to the observer (if one is installed and w is non-NULL).
 *
 * @param w   Weight being dispatched
 * @param op  Kind of dispatch
 *
 * Thread-safe: yes, if the observer is
 * Reentrant:   yes, if the observer is
 * Signal-safe: no
 */
void pn_qweight_note_dispatch(const pn_qweight *w, pn_qweight_op op);

/**
 * Install (or clear, fn = NULL) the process-wide activation observer (PN_ACT_STATS uses it).
 * Unset, each entry point pays one untaken branch.
 *
 * @param fn    Observer, or NULL to disable
 * @param user  Passed back to fn on every call
 *
 * Thread-safe: no (call before the forward)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qweight_set_activation_observer(pn_qweight_activation_observer_fn fn, void *user);

/**
 * Register the thread pool the matvec / matmul entry points split rows over.
 * Ignored in builds without USE_THREADS.
 *
 * @param pool  Pool, or NULL to run serially
 *
 * Thread-safe: no (call before the forward)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmv_set_pool(struct fw_pool *pool);

/**
 * Dot one quantized row with a float vector.
 *
 * @param row_data  Row bytes (any alignment)
 * @param vec       Input [n]
 * @param n         Elements; block types use whole blocks only (Q2_0_g128 rounds up)
 * @param type      Quant type; unsupported types print one stderr line and return 0
 * @return          The dot product
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
float pn_qdot(const void *row_data, const float *vec, int n, pn_gguf_quant type);

/**
 * @return  1 if the Metal GPU matvec is available (USE_METAL builds), else 0
 *
 * Thread-safe: TBD (follows fm_available)
 * Reentrant:   TBD
 * Signal-safe: no
 */
int pn_qmv_gpu_available(void);

/**
 * Upload a Q4_K weight's bytes to the GPU and attach the buffer to w->gpu.
 *
 * @param w  Weight (NULL: returns 0)
 * @return   1 if w->gpu is (already or now) set; 0 otherwise. Always 0 without USE_METAL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_qmv_gpu_upload(pn_qweight *w);

/**
 * Free any buffer attached to w->gpu and clear it (USE_METAL builds; else no-op).
 *
 * @param w  Weight (NULL: no-op)
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmv_gpu_free(pn_qweight *w);

/**
 * Enable (non-zero) or disable GPU decode for pn_qmm on GPU-resident Q4_K weights.
 * Off by default: GPU decode is slower than the CPU path; weights upload for prefill.
 *
 * @param on  Flag, stored as-is
 *
 * Thread-safe: no (call before the forward)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_qmm_set_gpu_decode(int on);

#ifdef __cplusplus
}
#endif

#endif /* PN_QWEIGHT_H */
