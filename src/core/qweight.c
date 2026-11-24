/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qweight.c
 * @brief  Quantized weight matvec / matmul / dot kernels, threading and GPU dispatch.
 *
 * Written from a behaviour specification (cleanroom_r2_qweight_spec.md, recorded in docs/cleanroom/README.md); no ggml or
 * llama.cpp source was consulted.
 *
 * Release results depend on clang fast-math codegen for the exact expression shapes below
 * (statement order, accumulator variables, helper boundaries). Keep the shapes.
 *
 * @status None
 * @issues - Block kernels ignore a partial trailing block; Q2_0_g128 processes it.
 *         - Ternary is unsupported through pn_qdot but works through pn_qmv.
 *         - Ternary 4-wide loop mis-decodes when group_size % 4 != 0; group_size <= 0
 *           divides by zero.
 *         - Q2_0 release math is amax*RN(1/127) and 127/amax; debug computes the literal
 *           source, so release and debug may differ at ties.
 *         - pn_qmm / pn_qmm_prefill output is undefined for types the row dequant does
 *           not cover and for non-block-multiple cols; dequant return codes are ignored.
 *         - pn_qmv_group under USE_METAL dereferences a NULL entry; its fallback path
 *           reports each weight to the observer twice.
 *         - pn_qweight_from_gguf ignores middle dims, casts dims to int, floors the row
 *           stride, leaves partial fields written on the silent -1 path.
 *         - Non-Metal pn_qmv_gpu_upload returns 0 and pn_qmv_gpu_free leaves gpu set.
 *         - The four Metal staging capacity globals have external linkage.
 *         - pn_qmv_set_pool is ignored without USE_THREADS.
 * @todo   None
 */
#include "qweight.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define PN_QWEIGHT_NEON 1
#endif

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#include "../accel/simd.h"
#include "../formats/gguf_tiled.h"
#include "qweight_tiled.h"

#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

#ifdef USE_METAL
#include "metal/fm_metal.h"
#endif

/* Rows below this run on the calling thread; the dispatch costs more than it saves. */
#define QWEIGHT_THREAD_MIN_ROWS 128
/* pn_qmv_group coalesces at most this many weights into one dispatch. */
#define QWEIGHT_GROUP_MAX 8
/* BLAS panel target: 131072 floats (~512 KB) per dequantized panel. */
#define QWEIGHT_PANEL_FLOATS 131072
#define QWEIGHT_PANEL_MAX_ROWS 128
/* Prefill panels never drop below this so sgemm's N stays AMX-friendly. */
#define QWEIGHT_PREFILL_MIN_ROWS 64

/* ------------------------------------------------------------------------------------------
 * Process-wide state
 * ---------------------------------------------------------------------------------------- */

#ifdef USE_THREADS
static fw_pool *g_pool = NULL;
#endif

static int g_gpu_decode = 0;
static pn_qweight_dispatch_observer_fn g_observer_fn = NULL;
static void *g_observer_user = NULL;
static pn_qweight_activation_observer_fn g_activation_fn = NULL;
static void *g_activation_user = NULL;

#ifdef USE_METAL
static fm_buffer *g_gpu_in = NULL;
static fm_buffer *g_gpu_out = NULL;
static fm_buffer *g_qmm_in = NULL;
static fm_buffer *g_qmm_out = NULL;
size_t g_gpu_in_cap = 0;
size_t g_gpu_out_cap = 0;
size_t g_qmm_in_cap = 0;
size_t g_qmm_out_cap = 0;
#endif

/* ------------------------------------------------------------------------------------------
 * Per-type row dot kernels
 * ---------------------------------------------------------------------------------------- */

static float dot_row_f16(const void *row, const float *vec, int n) {
    const uint16_t *src16 = (const uint16_t *)row;
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += pn_half_to_float(src16[i]) * vec[i];
    return sum;
}

static float dot_row_q4_0(const uint8_t *row, const float *vec, int n) {
    int nb = n / 32;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++)
        sum += pn_dot_q4_0_f32(row + 18 * i, vec + 32 * i);
    return sum;
}

static float dot_row_q8_0(const uint8_t *row, const float *vec, int n) {
    int nb = n / 32;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++)
        sum += pn_dot_q8_0_f32(row + 34 * i, vec + 32 * i);
    return sum;
}

typedef void (*dequant_block_fn)(const void *block, float *out);

/* Dequantize each 256-element block to a stack scratch, then dot it. */
static float dot_row_kquant_scratch(const uint8_t *row, const float *vec, int n,
                                    size_t block_bytes, dequant_block_fn dequant) {
    int nb = n / 256;
    float sum = 0.0f;
    float tmp[256];
    for (int i = 0; i < nb; i++) {
        dequant(row + (size_t)i * block_bytes, tmp);
        sum += pn_dot_f32(tmp, vec + 256 * i, 256);
    }
    return sum;
}

#ifdef PN_QWEIGHT_NEON

/* Widen 16 quant bytes to floats, accumulate q*v into *qd and the plain sum of v into *vs. */
static inline void acc16(uint8x16_t q, const float *v, float32x4_t *qd, float32x4_t *vs) {
    uint16x8_t q_lo = vmovl_u8(vget_low_u8(q));
    uint16x8_t q_hi = vmovl_high_u8(q);
    float32x4_t q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo)));
    float32x4_t q1 = vcvtq_f32_u32(vmovl_high_u16(q_lo));
    float32x4_t q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi)));
    float32x4_t q3 = vcvtq_f32_u32(vmovl_high_u16(q_hi));
    float32x4_t v0 = vld1q_f32(v);
    float32x4_t v1 = vld1q_f32(v + 4);
    float32x4_t v2 = vld1q_f32(v + 8);
    float32x4_t v3 = vld1q_f32(v + 12);
    *qd = vfmaq_f32(*qd, q0, v0);
    *qd = vfmaq_f32(*qd, q1, v1);
    *qd = vfmaq_f32(*qd, q2, v2);
    *qd = vfmaq_f32(*qd, q3, v3);
    *vs = vaddq_f32(*vs, vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
}

static void unpack_q4_k_scales(const uint8_t *s, uint8_t *sc, uint8_t *mn) {
    for (int j = 0; j < 4; j++) {
        sc[j] = s[j] & 63;
        mn[j] = s[j + 4] & 63;
    }
    for (int j = 4; j < 8; j++) {
        sc[j] = (uint8_t)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn[j] = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

/* One Q4_K block (256 elements) folded into the row accumulator acc. */
static inline float32x4_t q4_k_block_acc(const uint8_t *blk, const float *vec,
                                         float32x4_t acc) {
    uint16_t d_bits = 0;
    uint16_t dmin_bits = 0;
    memcpy(&d_bits, blk, sizeof(d_bits));
    memcpy(&dmin_bits, blk + 2, sizeof(dmin_bits));
    float d = pn_half_to_float(d_bits);
    float dmin = pn_half_to_float(dmin_bits);
    uint8_t sc[8];
    uint8_t mn[8];
    unpack_q4_k_scales(blk + 4, sc, mn);
    const uint8_t *qs = blk + 16;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (int k = 0; k < 4; k++) {
        uint8x16_t p0 = vld1q_u8(qs + 32 * k);
        uint8x16_t p1 = vld1q_u8(qs + 32 * k + 16);
        float32x4_t qd = vdupq_n_f32(0.0f);
        float32x4_t vs = vdupq_n_f32(0.0f);
        acc16(vandq_u8(p0, mask_lo), vec, &qd, &vs);
        acc16(vandq_u8(p1, mask_lo), vec + 16, &qd, &vs);
        acc = vfmaq_n_f32(acc, qd, d * (float)sc[2 * k]);
        acc = vfmaq_n_f32(acc, vs, -dmin * (float)mn[2 * k]);

        qd = vdupq_n_f32(0.0f);
        vs = vdupq_n_f32(0.0f);
        acc16(vshrq_n_u8(p0, 4), vec + 32, &qd, &vs);
        acc16(vshrq_n_u8(p1, 4), vec + 48, &qd, &vs);
        acc = vfmaq_n_f32(acc, qd, d * (float)sc[2 * k + 1]);
        acc = vfmaq_n_f32(acc, vs, -dmin * (float)mn[2 * k + 1]);
        vec += 64;
    }
    return acc;
}

static float dot_row_q4_k(const uint8_t *row, const float *vec, int n) {
    int nb = n / 256;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++)
        acc = q4_k_block_acc(row + (size_t)i * 144, vec + 256 * i, acc);
    float sum = 0.0f;
    sum += vaddvq_f32(acc);
    return sum;
}

/* One quant group of a Q6_K block: q against 16 inputs, scale s, bias -32 folded in. */
static inline float32x4_t q6_k_group_acc(uint8x16_t q, const float *input, float d,
                                         int8_t s, float32x4_t acc) {
    float32x4_t qd = vdupq_n_f32(0.0f);
    float32x4_t vs = vdupq_n_f32(0.0f);
    acc16(q, input, &qd, &vs);
    float coef = d * (float)s;
    acc = vfmaq_n_f32(acc, qd, coef);
    acc = vfmaq_n_f32(acc, vs, -32.0f * coef);
    return acc;
}

/* One Q6_K block (256 elements) folded into the row accumulator acc. */
static inline float32x4_t q6_k_block_acc(const uint8_t *blk, const float *vec,
                                         float32x4_t acc) {
    uint16_t d_bits = 0;
    memcpy(&d_bits, blk + 208, sizeof(d_bits));
    float d = pn_half_to_float(d_bits);
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const uint8x16_t mask_2 = vdupq_n_u8(3);

    for (int h = 0; h < 2; h++) {
        const float *vo = vec + 128 * h;
        for (int g = 0; g < 2; g++) {
            int o = 16 * g;
            uint8x16_t la = vld1q_u8(ql + o);
            uint8x16_t lb = vld1q_u8(ql + 32 + o);
            uint8x16_t hb = vld1q_u8(qh + o);
            uint8x16_t q1 = vorrq_u8(vandq_u8(la, mask_lo),
                                     vshlq_n_u8(vandq_u8(hb, mask_2), 4));
            uint8x16_t q2 = vorrq_u8(vandq_u8(lb, mask_lo),
                                     vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 2), mask_2), 4));
            uint8x16_t q3 = vorrq_u8(vshrq_n_u8(la, 4),
                                     vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 4), mask_2), 4));
            uint8x16_t q4 = vorrq_u8(vshrq_n_u8(lb, 4), vshlq_n_u8(vshrq_n_u8(hb, 6), 4));
            acc = q6_k_group_acc(q1, vo + o, d, sc[0 + g], acc);
            acc = q6_k_group_acc(q2, vo + 32 + o, d, sc[2 + g], acc);
            acc = q6_k_group_acc(q3, vo + 64 + o, d, sc[4 + g], acc);
            acc = q6_k_group_acc(q4, vo + 96 + o, d, sc[6 + g], acc);
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
}

static float dot_row_q6_k(const uint8_t *row, const float *vec, int n) {
    int nb = n / 256;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++)
        acc = q6_k_block_acc(row + (size_t)i * 210, vec + 256 * i, acc);
    float sum = 0.0f;
    sum += vaddvq_f32(acc);
    return sum;
}

#else /* !PN_QWEIGHT_NEON */

static float dot_row_q4_k(const uint8_t *row, const float *vec, int n) {
    return dot_row_kquant_scratch(row, vec, n, 144, pn_dequantize_block_q4_k);
}

static float dot_row_q6_k(const uint8_t *row, const float *vec, int n) {
    return dot_row_kquant_scratch(row, vec, n, 210, pn_dequantize_block_q6_k);
}

#endif /* PN_QWEIGHT_NEON */

/* Q2_0_g128 with activations quantized to int8 per 32-element sub-block. */
static float dot_row_q2_0_g128(const uint8_t *row, const float *vec, int n) {
    float sum = 0.0f;
    int nblk = (n + 127) / 128;
    for (int blk = 0; blk < nblk; blk++) {
        const uint8_t *b = row + 34 * blk;
        uint16_t d_bits = 0;
        memcpy(&d_bits, b, sizeof(d_bits));
        float d_w = pn_half_to_float(d_bits);
        const uint8_t *qs = b + 2;
        int start = 128 * blk;
        int cnt = n - start;
        if (cnt > 128)
            cnt = 128;
        float blk_acc = 0.0f;
        int sb = 0;
        while (sb + 32 <= cnt) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = fabsf(vec[start + sb + j]);
                if (a > amax)
                    amax = a;
            }
            float d_pre = amax / 127.0f;
            float id = (d_pre != 0.0f) ? 1.0f / d_pre : 0.0f;
            float d_act = pn_half_to_float(pn_float_to_half(d_pre));
            int isum = 0;
            for (int j = 0; j < 32; j++) {
                int k = sb + j;
                int code = (qs[k >> 2] >> (2 * (k & 3))) & 3;
                isum += (code - 1) * (int)roundf(vec[start + sb + j] * id);
            }
            blk_acc += d_act * (float)isum;
            sb += 32;
        }
        for (int j = sb; j < cnt; j++) {
            int code = (qs[j >> 2] >> (2 * (j & 3))) & 3;
            blk_acc += (float)(code - 1) * vec[start + j];
        }
        sum += d_w * blk_acc;
    }
    return sum;
}

/* Packed 2-bit ternary row (00=-1, 01=0, 10=+1, 11=nothing) with per-group scales. */
static float dot_row_ternary(const uint8_t *packed, const float *vec, int n,
                             const float *scales, int gs) {
    int n_groups = (n + gs - 1) / gs;
    float sum = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        float scale = scales[g];
        int start = g * gs;
        int end = start + gs;
        if (end > n)
            end = n;
        float gs_acc = 0.0f;
        int i = start;
        /* Always decodes byte i>>2 from bit pair 0, even when i % 4 != 0 [AS-IS]. */
        while (i + 3 < end) {
            uint8_t byte = packed[i >> 2];
            for (int k = 0; k < 4; k++) {
                int c = (byte >> (2 * k)) & 3;
                if (c == 0)
                    gs_acc -= vec[i + k];
                else if (c == 2)
                    gs_acc += vec[i + k];
            }
            i += 4;
        }
        while (i < end) {
            int c = (packed[i >> 2] >> (2 * (i & 3))) & 3;
            if (c == 0)
                gs_acc -= vec[i];
            else if (c == 2)
                gs_acc += vec[i];
            i++;
        }
        sum += scale * gs_acc;
    }
    return sum;
}

float pn_qdot(const void *row_data, const float *vec, int n, pn_gguf_quant type) {
    const uint8_t *row = (const uint8_t *)row_data;
    switch (type) {
    case GGUF_QUANT_F32:
        return pn_dot_f32((const float *)row_data, vec, n);
    case GGUF_QUANT_F16:
        return dot_row_f16(row_data, vec, n);
    case GGUF_QUANT_Q4_0:
        return dot_row_q4_0(row, vec, n);
    case GGUF_QUANT_Q8_0:
        return dot_row_q8_0(row, vec, n);
    case GGUF_QUANT_Q3_K:
        return dot_row_kquant_scratch(row, vec, n, 110, pn_dequantize_block_q3_k);
    case GGUF_QUANT_Q4_K:
        return dot_row_q4_k(row, vec, n);
    case GGUF_QUANT_Q5_K:
        return dot_row_kquant_scratch(row, vec, n, 176, pn_dequantize_block_q5_k);
    case GGUF_QUANT_Q6_K:
        return dot_row_q6_k(row, vec, n);
    case GGUF_QUANT_Q2_0_G128:
        return dot_row_q2_0_g128(row, vec, n);
    default:
        fprintf(stderr, "pn_qdot: unsupported type %d\n", (int)type);
        return 0.0f;
    }
}

/* ------------------------------------------------------------------------------------------
 * Observer and process-wide setters
 * ---------------------------------------------------------------------------------------- */

void pn_qweight_set_dispatch_observer(pn_qweight_dispatch_observer_fn fn, void *user) {
    g_observer_fn = fn;
    g_observer_user = user;
}

void pn_qweight_note_dispatch(const pn_qweight *w, pn_qweight_op op) {
    if (g_observer_fn && w)
        g_observer_fn(w, op, g_observer_user);
}

void pn_qweight_set_activation_observer(pn_qweight_activation_observer_fn fn, void *user) {
    g_activation_fn = fn;
    g_activation_user = user;
}

/* Called once per entry-point dispatch (never from the internal per-row fallbacks), so each
 * activation row reaches the observer exactly once. One predictable branch when unset. */
static inline void note_activations(const pn_qweight *w, const float *acts, int B) {
    if (g_activation_fn)
        g_activation_fn(w, acts, B, g_activation_user);
}

void pn_qmv_set_pool(struct fw_pool *pool) {
#ifdef USE_THREADS
    g_pool = pool;
#else
    (void)pool;
#endif
}

void pn_qmm_set_gpu_decode(int on) {
    g_gpu_decode = on;
}

/* ------------------------------------------------------------------------------------------
 * GPU (Metal) staging, matvec and residency
 * ---------------------------------------------------------------------------------------- */

#ifdef USE_METAL

/* Reuse a staging buffer if big enough, else regrow it with 25% + 4 KB slack. */
static fm_buffer *gpu_stage(fm_buffer **buf, size_t *cap, size_t bytes) {
    if (*buf && *cap >= bytes)
        return *buf;
    if (*buf) {
        fm_buffer_free(*buf);
        *buf = NULL;
    }
    size_t want = bytes + (bytes >> 2) + 4096;
    *buf = fm_buffer_alloc(want);
    *cap = *buf ? want : 0;
    return *buf;
}

/* Returns 1 if the GPU computed out, 0 if the caller must run the CPU path. */
static int gpu_matvec(const pn_qweight *w, const float *vec, float *out) {
    if (!w->gpu || w->type != GGUF_QUANT_Q4_K || !fm_available())
        return 0;
    size_t in_bytes = (size_t)w->cols * sizeof(float);
    size_t out_bytes = (size_t)w->rows * sizeof(float);
    fm_buffer *in = gpu_stage(&g_gpu_in, &g_gpu_in_cap, in_bytes);
    fm_buffer *outbuf = gpu_stage(&g_gpu_out, &g_gpu_out_cap, out_bytes);
    if (!in || !outbuf)
        return 0;
    memcpy(fm_buffer_data(in), vec, in_bytes);
    fm_matvec_q4_k((fm_buffer *)w->gpu, in, outbuf, w->rows, w->cols, (int)w->row_stride);
    fm_sync();
    memcpy(out, fm_buffer_data(outbuf), out_bytes);
    return 1;
}

/* Returns 1 if the GPU computed out, 0 if the caller must run the CPU path. */
static int gpu_qmm(const pn_qweight *w, const float *acts, int B, float *out) {
    if (!w->gpu || !g_gpu_decode || w->type != GGUF_QUANT_Q4_K || B > 32 || !fm_available())
        return 0;
    size_t in_bytes = (size_t)B * (size_t)w->cols * sizeof(float);
    size_t out_bytes = (size_t)B * (size_t)w->rows * sizeof(float);
    fm_buffer *in = gpu_stage(&g_qmm_in, &g_qmm_in_cap, in_bytes);
    fm_buffer *outbuf = gpu_stage(&g_qmm_out, &g_qmm_out_cap, out_bytes);
    if (!in || !outbuf)
        return 0;
    memcpy(fm_buffer_data(in), acts, in_bytes);
    fm_sgemm_q4_k((fm_buffer *)w->gpu, in, outbuf, w->rows, w->cols, (int)w->row_stride, B);
    fm_sync();
    memcpy(out, fm_buffer_data(outbuf), out_bytes);
    return 1;
}

#endif /* USE_METAL */

int pn_qmv_gpu_available(void) {
#ifdef USE_METAL
    return fm_available() ? 1 : 0;
#else
    return 0;
#endif
}

int pn_qmv_gpu_upload(pn_qweight *w) {
#ifdef USE_METAL
    if (!w)
        return 0;
    if (w->gpu)
        return 1;
    if (w->type != GGUF_QUANT_Q4_K)
        return 0;
    if (fm_init() != 0 || !fm_available())
        return 0;
    size_t bytes = (size_t)w->rows * w->row_stride;
    fm_buffer *buf = fm_buffer_alloc(bytes);
    if (!buf)
        return 0;
    memcpy(fm_buffer_data(buf), w->data, bytes);
    w->gpu = buf;
    return 1;
#else
    (void)w;
    return 0;
#endif
}

void pn_qmv_gpu_free(pn_qweight *w) {
#ifdef USE_METAL
    if (w && w->gpu) {
        fm_buffer_free((fm_buffer *)w->gpu);
        w->gpu = NULL;
    }
#else
    (void)w;
#endif
}

/* ------------------------------------------------------------------------------------------
 * pn_qmv
 * ---------------------------------------------------------------------------------------- */

typedef struct {
    const pn_qweight *w;
    const float *vec;
    float *out;
    int n_groups;
} qmv_task;

static void qmv_ternary_range(int start, int end, void *arg) {
    const qmv_task *t = (const qmv_task *)arg;
    const pn_qweight *w = t->w;
    for (int r = start; r < end; r++) {
        const uint8_t *row = (const uint8_t *)w->data + (size_t)r * w->row_stride;
        const float *row_scales = w->scales + (size_t)r * (size_t)t->n_groups;
        t->out[r] = dot_row_ternary(row, t->vec, w->cols, row_scales, w->group_size);
    }
}

static void qmv_range(int start, int end, void *arg) {
    const qmv_task *t = (const qmv_task *)arg;
    const pn_qweight *w = t->w;
    for (int r = start; r < end; r++) {
        const uint8_t *row = (const uint8_t *)w->data + (size_t)r * w->row_stride;
        t->out[r] = pn_qdot(row, t->vec, w->cols, w->type);
    }
}

/* Run fn over [0, rows), split over the pool when one is set and rows clear the threshold. */
static void run_rows(int rows, void (*fn)(int, int, void *), void *task) {
#ifdef USE_THREADS
    if (g_pool && rows >= QWEIGHT_THREAD_MIN_ROWS) {
        fw_for(g_pool, rows, fn, task);
        return;
    }
#endif
    fn(0, rows, task);
}

/* The pool the tiled path splits over, or NULL. */
static struct fw_pool *tiled_pool(void) {
#ifdef USE_THREADS
    return (struct fw_pool *)g_pool;
#else
    return NULL;
#endif
}

/* pn_qmv after its argument check and activation report: the per-row fallbacks of pn_qmm and
 * pn_qmv_group call this, so their rows are reported once, by their own entry point. */
static void qmv_dispatch(const pn_qweight *w, const float *vec, float *out) {
    pn_qweight_note_dispatch(w, PN_QWEIGHT_OP_MATVEC);
    if (pn_tiled_is_type(w->type)) {
        pn_qmm_tiled(w, vec, 1, out, tiled_pool());
        return;
    }

    qmv_task task = { w, vec, out, 0 };
    if (w->type == PN_QUANT_TERNARY) {
        task.n_groups = (w->cols + w->group_size - 1) / w->group_size;
        run_rows(w->rows, qmv_ternary_range, &task);
        return;
    }
#ifdef USE_METAL
    if (w->gpu && gpu_matvec(w, vec, out))
        return;
#endif
    run_rows(w->rows, qmv_range, &task);
}

void pn_qmv(const pn_qweight *w, const float *vec, float *out) {
    if (!w || !vec || !out)
        return;
    note_activations(w, vec, 1);
    qmv_dispatch(w, vec, out);
}

/* ------------------------------------------------------------------------------------------
 * pn_qmv_group
 * ---------------------------------------------------------------------------------------- */

#ifdef USE_THREADS

typedef struct {
    const pn_qweight *const *w;
    const float *const *vec;
    float *const *out;
    int off[QWEIGHT_GROUP_MAX + 1];
    int n;
} qmv_group_task;

/* Global rows [start, end) across the concatenated weights. */
static void qmv_group_range(int start, int end, void *arg) {
    const qmv_group_task *t = (const qmv_group_task *)arg;
    int s = 0;
    while (s < t->n && start >= t->off[s + 1])
        s++;
    for (int r = start; r < end; r++) {
        while (r >= t->off[s + 1])
            s++;
        const pn_qweight *w = t->w[s];
        int lr = r - t->off[s];
        const uint8_t *row = (const uint8_t *)w->data + (size_t)lr * w->row_stride;
        t->out[s][lr] = pn_qdot(row, t->vec[s], w->cols, w->type);
    }
}

/* Returns 1 if all n weights were computed in one pool dispatch. */
static int qmv_group_coalesced(const pn_qweight *const *w, const float *const *vec,
                               float *const *out, int n) {
    int coalesce = g_pool != NULL;
    for (int i = 0; i < n && coalesce; i++) {
        if (!w[i] || w[i]->type == PN_QUANT_TERNARY || pn_tiled_is_type(w[i]->type))
            coalesce = 0;
#ifdef USE_METAL
        /* Dereferences a NULL entry [AS-IS]. */
        if (w[i]->gpu != NULL && w[i]->type == GGUF_QUANT_Q4_K)
            coalesce = 0;
#endif
    }
    if (!coalesce)
        return 0;

    qmv_group_task task = { w, vec, out, {0}, n };
    task.off[0] = 0;
    for (int i = 0; i < n; i++)
        task.off[i + 1] = task.off[i] + w[i]->rows;
    if (task.off[n] < QWEIGHT_THREAD_MIN_ROWS)
        return 0;
    fw_for(g_pool, task.off[n], qmv_group_range, &task);
    return 1;
}

#endif /* USE_THREADS */

void pn_qmv_group(const pn_qweight *const *w, const float *const *vec,
                  float *const *out, int n) {
    if (n <= 0 || !w)
        return;
    if (n > QWEIGHT_GROUP_MAX)
        n = QWEIGHT_GROUP_MAX;
    for (int i = 0; i < n; i++) {
        pn_qweight_note_dispatch(w[i], PN_QWEIGHT_OP_MATVEC);
        if (w[i] && vec[i] && out[i])
            note_activations(w[i], vec[i], 1);
    }
#ifdef USE_THREADS
    if (qmv_group_coalesced(w, vec, out, n))
        return;
#endif
    for (int i = 0; i < n; i++) {
        if (w[i] && vec[i] && out[i])
            qmv_dispatch(w[i], vec[i], out[i]);
    }
}

/* ------------------------------------------------------------------------------------------
 * pn_qmm / pn_qmm_prefill
 * ---------------------------------------------------------------------------------------- */

/* Dequantize one full row; -1 (nothing written) for types it does not cover. */
static int dequant_row(const uint8_t *row, pn_gguf_quant type, int cols, float *out) {
    switch (type) {
    case GGUF_QUANT_F32:
        memcpy(out, row, (size_t)cols * sizeof(float));
        return 0;
    case GGUF_QUANT_F16: {
        const uint16_t *src16 = (const uint16_t *)row;
        for (int i = 0; i < cols; i++)
            out[i] = pn_half_to_float(src16[i]);
        return 0;
    }
    case GGUF_QUANT_Q4_0:
        for (int i = 0; i < cols / 32; i++)
            pn_dequantize_block_q4_0(row + (size_t)i * 18, out + 32 * i);
        return 0;
    case GGUF_QUANT_Q8_0:
        for (int i = 0; i < cols / 32; i++)
            pn_dequantize_block_q8_0(row + (size_t)i * 34, out + 32 * i);
        return 0;
    case GGUF_QUANT_Q3_K:
        for (int i = 0; i < cols / 256; i++)
            pn_dequantize_block_q3_k(row + (size_t)i * 110, out + 256 * i);
        return 0;
    case GGUF_QUANT_Q4_K:
        for (int i = 0; i < cols / 256; i++)
            pn_dequantize_block_q4_k(row + (size_t)i * 144, out + 256 * i);
        return 0;
    case GGUF_QUANT_Q5_K:
        for (int i = 0; i < cols / 256; i++)
            pn_dequantize_block_q5_k(row + (size_t)i * 176, out + 256 * i);
        return 0;
    case GGUF_QUANT_Q6_K:
        for (int i = 0; i < cols / 256; i++)
            pn_dequantize_block_q6_k(row + (size_t)i * 210, out + 256 * i);
        return 0;
    default:
        return -1;
    }
}

/* Rows per ~512 KB dequantized panel, clamped to [1, 128]. */
static int panel_rows(int cols) {
    int c = cols > 1 ? cols : 1;
    int rows = QWEIGHT_PANEL_FLOATS / c;
    if (rows < 1)
        rows = 1;
    if (rows > QWEIGHT_PANEL_MAX_ROWS)
        rows = QWEIGHT_PANEL_MAX_ROWS;
    return rows;
}

typedef struct {
    const pn_qweight *w;
    const float *acts;
    int B;
    float *out;
    int panel;
} qmm_task;

static const uint8_t *qweight_row(const pn_qweight *w, int r) {
    return (const uint8_t *)w->data + (size_t)r * w->row_stride;
}

/* out[b*rows + r0 + i] for rows r0 .. r0+rb-1, from rb dequantized rows in panel. */
static void qmm_panel_product(const qmm_task *t, const float *panel, int r0, int rb) {
    const pn_qweight *w = t->w;
    int cols = w->cols;
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, t->B, rb, cols,
                1.0f, t->acts, cols, panel, cols, 0.0f, t->out + r0, w->rows);
#else
    for (int i = 0; i < rb; i++) {
        for (int b = 0; b < t->B; b++)
            t->out[(size_t)b * w->rows + r0 + i] =
                pn_dot_f32(panel + (size_t)i * cols, t->acts + (size_t)b * cols, cols);
    }
#endif
}

/* Direct quantized dots for rows [r0, r1): the allocation-failure fallback. */
static void qmm_rows_direct(const qmm_task *t, int r0, int r1) {
    const pn_qweight *w = t->w;
    for (int r = r0; r < r1; r++) {
        for (int b = 0; b < t->B; b++)
            t->out[(size_t)b * w->rows + r] =
                pn_qdot(qweight_row(w, r), t->acts + (size_t)b * w->cols, w->cols, w->type);
    }
}

#ifdef USE_BLAS
/* Panelled sgemm over rows [start, end). Returns -1 if the panel cannot be allocated. */
static int qmm_range_blas(const qmm_task *t, int start, int end) {
    const pn_qweight *w = t->w;
    int cols = w->cols;
    int panel_cap = panel_rows(cols);
    float *panel = malloc((size_t)panel_cap * (size_t)cols * sizeof(float));
    if (!panel)
        return -1;
    for (int r0 = start; r0 < end; r0 += panel_cap) {
        int rb = end - r0 < panel_cap ? end - r0 : panel_cap;
        for (int i = 0; i < rb; i++)
            dequant_row(qweight_row(w, r0 + i), w->type, cols, panel + (size_t)i * cols);
        qmm_panel_product(t, panel, r0, rb);
    }
    free(panel);
    return 0;
}
#endif

static void qmm_range(int start, int end, void *arg) {
    const qmm_task *t = (const qmm_task *)arg;
    const pn_qweight *w = t->w;
#ifdef USE_BLAS
    if (qmm_range_blas(t, start, end) == 0)
        return;
#endif
    float *scratch = malloc((size_t)w->cols * sizeof(float));
    if (!scratch) {
        qmm_rows_direct(t, start, end);
        return;
    }
    for (int r = start; r < end; r++) {
        dequant_row(qweight_row(w, r), w->type, w->cols, scratch);
        for (int b = 0; b < t->B; b++)
            t->out[(size_t)b * w->rows + r] =
                pn_dot_f32(scratch, t->acts + (size_t)b * w->cols, w->cols);
    }
    free(scratch);
}

/* Per-batch pn_qmv loop: ternary weights and batches above the shared-dequant cap. */
static void qmm_per_batch(const pn_qweight *w, const float *acts, int B, float *out) {
    for (int b = 0; b < B; b++)
        qmv_dispatch(w, acts + (size_t)b * w->cols, out + (size_t)b * w->rows);
}

void pn_qmm(const pn_qweight *w, const float *acts, int B, float *out) {
    if (!w || !acts || !out || B <= 0)
        return;
    note_activations(w, acts, B);
    pn_qweight_note_dispatch(w, PN_QWEIGHT_OP_MATMUL);
    if (pn_tiled_is_type(w->type)) {
        pn_qmm_tiled(w, acts, B, out, tiled_pool());
        return;
    }
#ifdef USE_METAL
    if (gpu_qmm(w, acts, B, out))
        return;
#endif
    if (w->type == PN_QUANT_TERNARY || B > PN_QMM_MAX_B) {
        qmm_per_batch(w, acts, B, out);
        return;
    }
    qmm_task task = { w, acts, B, out, 0 };
    run_rows(w->rows, qmm_range, &task);
}

/* Panel blocks [bs, be) of task->panel rows each. */
static void prefill_range(int bs, int be, void *arg) {
    const qmm_task *t = (const qmm_task *)arg;
    const pn_qweight *w = t->w;
    int panel = t->panel;
    float *pbuf = malloc((size_t)panel * (size_t)w->cols * sizeof(float));
    if (!pbuf) {
        for (int blk = bs; blk < be; blk++) {
            int r0 = blk * panel;
            int rb = w->rows - r0 < panel ? w->rows - r0 : panel;
            qmm_rows_direct(t, r0, r0 + rb);
        }
        return;
    }
    for (int blk = bs; blk < be; blk++) {
        int r0 = blk * panel;
        int rb = w->rows - r0 < panel ? w->rows - r0 : panel;
        for (int i = 0; i < rb; i++)
            dequant_row(qweight_row(w, r0 + i), w->type, w->cols, pbuf + (size_t)i * w->cols);
        qmm_panel_product(t, pbuf, r0, rb);
    }
    free(pbuf);
}

void pn_qmm_prefill(const pn_qweight *w, const float *acts, int B, float *out) {
    if (!w || !acts || !out || B <= 0)
        return;
    note_activations(w, acts, B);
    pn_qweight_note_dispatch(w, PN_QWEIGHT_OP_MATMUL);
    if (pn_tiled_is_type(w->type)) {
        pn_qmm_tiled(w, acts, B, out, tiled_pool());
        return;
    }
    if (w->type == PN_QUANT_TERNARY) {
        qmm_per_batch(w, acts, B, out);
        return;
    }
    int panel = panel_rows(w->cols);
    if (panel < QWEIGHT_PREFILL_MIN_ROWS)
        panel = QWEIGHT_PREFILL_MIN_ROWS;
    int nblk = (w->rows + panel - 1) / panel;
    qmm_task task = { w, acts, B, out, panel };
#ifdef USE_THREADS
    if (g_pool && nblk > 1) {
        fw_for(g_pool, nblk, prefill_range, &task);
        return;
    }
#endif
    prefill_range(0, nblk, &task);
}

/* ------------------------------------------------------------------------------------------
 * GGUF loading
 * ---------------------------------------------------------------------------------------- */

int pn_qweight_from_gguf(pn_qweight *qw, pn_gguf_file *gguf, const char *name) {
    if (!qw || !gguf || !name)
        return -1;
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(gguf, name);
    if (!info) {
        fprintf(stderr, "pn_qweight_from_gguf: tensor '%s' not found\n", name);
        return -1;
    }
    qw->data = (const char *)gguf->data + info->offset;
    qw->type = info->type;
    qw->gpu = NULL;
    if (info->n_dims >= 2) {
        qw->rows = (int)info->dims[info->n_dims - 1];
        qw->cols = (int)info->dims[0];
    } else {
        qw->rows = (int)info->dims[0];
        qw->cols = 1;
    }
    size_t bpb = pn_gguf_bytes_per_block(qw->type);
    int bs = pn_gguf_block_size(qw->type);
    if (bs > 0 && bpb > 0) {
        qw->row_stride = ((size_t)qw->cols / (size_t)bs) * bpb;
        return 0;
    }
    qw->row_stride = 0;
    return -1;
}
