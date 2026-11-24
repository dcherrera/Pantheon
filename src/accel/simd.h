/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   simd.h
 * @brief  CPU vector kernels: dot products, element-wise ops, RMS norm, Q4_0/Q8_0 block dots.
 *
 * Written from a behaviour specification (cleanroom_r2_simd_spec.md, recorded in docs/cleanroom/README.md); no ggml or
 * llama.cpp source was consulted.
 *
 * One code path is chosen at compile time: NEON (arm64), AVX2+FMA (x86 built with both),
 * otherwise plain C that the compiler may auto-vectorize. There is no runtime dispatch.
 *
 * Common contract for every function taking `n`: for n <= 0 nothing is read or written and
 * reductions return +0.0f. Pointers need only float (4-byte) alignment. In-place calls are
 * supported: out == a, out == b, out == gate, out == up, out == x, w == x, a == b.
 * No NULL checks: a NULL pointer with n > 0 crashes.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_SIMD_H
#define PN_SIMD_H

#include <stddef.h>

/**
 * Dot product of two float vectors.
 *
 * @param a  First vector, n floats
 * @param b  Second vector, n floats
 * @param n  Element count
 * @return   Sum of a[i] * b[i]; +0.0f when n <= 0
 *
 * Thread-safe: yes (pure)
 * Reentrant:   yes
 * Signal-safe: yes
 */
float pn_dot_f32(const float *a, const float *b, int n);

/**
 * Element-wise add: out[i] = a[i] + b[i].
 *
 * @param a    First input, n floats
 * @param b    Second input, n floats
 * @param out  Output, n floats (may equal a or b)
 * @param n    Element count
 *
 * Thread-safe: yes (pure apart from writing out)
 * Reentrant:   yes
 * Signal-safe: yes
 */
void pn_vadd_f32(const float *a, const float *b, float *out, int n);

/**
 * Element-wise multiply: out[i] = a[i] * b[i].
 *
 * @param a    First input, n floats
 * @param b    Second input, n floats
 * @param out  Output, n floats (may equal a or b)
 * @param n    Element count
 *
 * Thread-safe: yes (pure apart from writing out)
 * Reentrant:   yes
 * Signal-safe: yes
 */
void pn_vmul_f32(const float *a, const float *b, float *out, int n);

/**
 * SwiGLU combine: out[i] = silu(gate[i]) * up[i], silu(x) = x / (1 + expf(-x)).
 *
 * @param gate  Gate activations, n floats
 * @param up    Up projections, n floats
 * @param out   Output, n floats (may equal gate or up)
 * @param n     Element count
 *
 * Thread-safe: yes (pure apart from writing out)
 * Reentrant:   yes
 * Signal-safe: yes (libm expf only)
 */
void pn_vsilu_mul_f32(const float *gate, const float *up, float *out, int n);

/**
 * RMS norm: out[i] = x[i] * r * w[i], r = 1 / sqrt(mean(x^2) + eps), mean taken in double.
 *
 * @param x    Input, n floats
 * @param w    Per-element weight, n floats (may equal x)
 * @param out  Output, n floats (may equal x)
 * @param n    Element count
 * @param eps  Added to the mean square before the square root; not clamped
 *
 * Thread-safe: yes (pure apart from writing out)
 * Reentrant:   yes
 * Signal-safe: yes (libm sqrt only)
 */
void pn_vrms_norm_f32(const float *x, const float *w, float *out, int n, float eps);

/**
 * Dot of one Q4_0 block (32 elements, 18 bytes) with 32 floats.
 *
 * @param block  Q4_0 block (pn_block_q4_0); needs 2-byte alignment
 * @param vec    32 floats
 * @return       Sum over j of (code_j - 8) * d * vec[j]
 *
 * Thread-safe: yes (pure)
 * Reentrant:   yes
 * Signal-safe: yes
 */
float pn_dot_q4_0_f32(const void *block, const float *vec);

/**
 * Dot of one Q8_0 block (32 elements, 34 bytes) with 32 floats.
 *
 * @param block  Q8_0 block (pn_block_q8_0); needs 2-byte alignment
 * @param vec    32 floats
 * @return       d * sum over j of qs[j] * vec[j]
 *
 * Thread-safe: yes (pure)
 * Reentrant:   yes
 * Signal-safe: yes
 */
float pn_dot_q8_0_f32(const void *block, const float *vec);

/**
 * Scale a vector: out[i] = a[i] * scale.
 *
 * @param a      Input, n floats
 * @param scale  Multiplier
 * @param out    Output, n floats (may equal a)
 * @param n      Element count
 *
 * Thread-safe: yes (pure apart from writing out)
 * Reentrant:   yes
 * Signal-safe: yes
 */
void pn_vscale_f32(const float *a, float scale, float *out, int n);

/**
 * Sum of a float vector.
 *
 * @param a  Input, n floats
 * @param n  Element count
 * @return   Sum of a[i]; +0.0f when n <= 0
 *
 * Thread-safe: yes (pure)
 * Reentrant:   yes
 * Signal-safe: yes
 */
float pn_vsum_f32(const float *a, int n);

#endif /* PN_SIMD_H */
