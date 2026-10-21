/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   simd.c
 * @brief  NEON / AVX2+FMA / plain-C implementations of the CPU vector kernels.
 *
 * Written from a behaviour specification (plans/cleanroom_r2_simd_spec.md); no ggml or
 * llama.cpp source was consulted.
 *
 * Release builds use -ffast-math, so output bits depend on the shape of this source (which
 * loops are explicit SIMD, accumulator count, statement split), not only on the maths.
 * Change the shape of any loop and the bits change.
 *
 * @status None
 * @issues - -ffast-math re-associates silu*up into (up*x)/(1+e); release and debug differ.
 *         - x86 vectorized silu uses a reciprocal estimate; its scalar tail divides exactly.
 *         - NEON Q4_0 scales per element, scalar Q4_0 gets d hoisted by the compiler, Q8_0
 *           scales at the end: the paths do not agree bit-for-bit.
 *         - The first FMA into a zero accumulator folds to a multiply (keeps -0).
 *         - NaN payload choice follows compiled operand order.
 *         - rms computes NaN for n == 0 (nothing is written).
 *         - No NULL or range checks.
 *         - pn_vmul_f32, pn_vscale_f32, pn_vsum_f32 have no callers.
 *         - The shipping x86 target (the bank, no AVX/FMA) never compiles the AVX2 path.
 * @todo   None
 */
#include "simd.h"

#include <math.h>

#include "../formats/gguf_dequant.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PN_SIMD_PATH_NEON 1
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#define PN_SIMD_PATH_AVX2 1
#include <immintrin.h>
#endif

#ifdef PN_SIMD_PATH_AVX2
/* Low half + high half, then two horizontal pair-adds; lane 0 holds the total. */
static float avx2_horizontal_sum(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    return _mm_cvtss_f32(low);
}
#endif

float pn_dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float sum;
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 7 < n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    acc0 = vaddq_f32(acc0, acc1);
    sum = vaddvq_f32(acc0);
#elif defined(PN_SIMD_PATH_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m256 a_lo = _mm256_loadu_ps(a + i);
        __m256 a_hi = _mm256_loadu_ps(a + i + 8);
        __m256 b_lo = _mm256_loadu_ps(b + i);
        __m256 b_hi = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a_lo, b_lo, acc0);
        acc1 = _mm256_fmadd_ps(a_hi, b_hi, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    sum = avx2_horizontal_sum(acc0);
#else
    sum = 0.0f;
#endif
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

void pn_vadd_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
#if defined(PN_SIMD_PATH_NEON)
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#elif defined(PN_SIMD_PATH_AVX2)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#endif
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

void pn_vmul_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
#if defined(PN_SIMD_PATH_NEON)
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#elif defined(PN_SIMD_PATH_AVX2)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#endif
    for (; i < n; i++)
        out[i] = a[i] * b[i];
}

void pn_vscale_f32(const float *a, float scale, float *out, int n) {
    int i = 0;
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(out + i, vmulq_f32(va, vscale));
    }
#elif defined(PN_SIMD_PATH_AVX2)
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vscale));
    }
#endif
    for (; i < n; i++)
        out[i] = a[i] * scale;
}

float pn_vsum_f32(const float *a, int n) {
    int i = 0;
    float sum;
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4)
        acc = vaddq_f32(acc, vld1q_f32(a + i));
    sum = vaddvq_f32(acc);
#elif defined(PN_SIMD_PATH_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    sum = avx2_horizontal_sum(acc);
#else
    sum = 0.0f;
#endif
    for (; i < n; i++)
        sum += a[i];
    return sum;
}

/* Plain loop on every path: the compiler vectorizes it (and, under -ffast-math, reorders
 * the arithmetic); hand-written SIMD here would change the output bits. */
void pn_vsilu_mul_f32(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = gate[i];
        float s = x / (1.0f + expf(-x));
        out[i] = s * up[i];
    }
}

void pn_vrms_norm_f32(const float *x, const float *w, float *out, int n, float eps) {
    /* Double accumulation: callers rely on its precision over long rows. */
    double ss = 0.0;
    for (int k = 0; k < n; k++)
        ss += (double)x[k] * (double)x[k];
    float r = (float)(1.0 / sqrt(ss / (double)n + (double)eps));

    int i = 0;
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t vr = vdupq_n_f32(r);
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vr), vw));
    }
#elif defined(PN_SIMD_PATH_AVX2)
    __m256 vr = _mm256_set1_ps(r);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vr), vw));
    }
#endif
    for (; i < n; i++)
        out[i] = x[i] * r * w[i];
}

float pn_dot_q4_0_f32(const void *block, const float *vec) {
    const pn_block_q4_0 *blk = (const pn_block_q4_0 *)block;
    float df = pn_half_to_float(blk->d);
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t vdf = vdupq_n_f32(df);
    float32x4_t voffset = vdupq_n_f32(8.0f);
    for (int j = 0; j < 16; j += 4) {
        float lo[4];
        float hi[4];
        for (int k = 0; k < 4; k++) {
            lo[k] = (float)(blk->qs[j + k] & 0x0F);
            hi[k] = (float)(blk->qs[j + k] >> 4);
        }
        float32x4_t vlo = vmulq_f32(vsubq_f32(vld1q_f32(lo), voffset), vdf);
        float32x4_t vhi = vmulq_f32(vsubq_f32(vld1q_f32(hi), voffset), vdf);
        acc0 = vfmaq_f32(acc0, vlo, vld1q_f32(vec + j));
        acc1 = vfmaq_f32(acc1, vhi, vld1q_f32(vec + j + 16));
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        int lo = blk->qs[j] & 0x0F;
        int hi = blk->qs[j] >> 4;
        sum += df * (float)(lo - 8) * vec[j];
        sum += df * (float)(hi - 8) * vec[j + 16];
    }
    return sum;
#endif
}

float pn_dot_q8_0_f32(const void *block, const float *vec) {
    const pn_block_q8_0 *blk = (const pn_block_q8_0 *)block;
    float df = pn_half_to_float(blk->d);
#if defined(PN_SIMD_PATH_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (int j = 0; j < 32; j += 8) {
        int16x8_t q16 = vmovl_s8(vld1_s8(blk->qs + j));
        float32x4_t q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        float32x4_t q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        acc0 = vfmaq_f32(acc0, q_lo, vld1q_f32(vec + j));
        acc1 = vfmaq_f32(acc1, q_hi, vld1q_f32(vec + j + 4));
    }
    return df * vaddvq_f32(vaddq_f32(acc0, acc1));
#else
    float sum = 0.0f;
    for (int j = 0; j < 32; j++)
        sum += (float)blk->qs[j] * vec[j];
    sum *= df;
    return sum;
#endif
}
