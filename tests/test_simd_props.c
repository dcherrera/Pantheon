/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Property tests for the CPU vector kernels (src/accel/simd.c), derived from the
 * behaviour contract only. Each property holds under release (-ffast-math) and debug flags.
 *
 * - n <= 0: reductions return +0.0f bits, nothing is written.
 * - Element-wise ops write exactly n elements with correctly rounded values.
 * - In-place calls give the same bits as out-of-place calls.
 * - dot/vsum over small integers are exact for every tail length.
 * - Q4_0/Q8_0 element mapping via one-hot vectors (exact).
 * - rms_norm accumulates in double (a case float accumulation gets visibly wrong).
 * - silu*up against a double reference.
 */

#include "lib/test.h"
#include "../src/accel/simd.h"
#include "../src/formats/gguf_dequant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PROP_MAX_N 300
#define PROP_GUARD 8
#define PROP_GUARD_VALUE 12345.0f

static uint32_t fbits(float f) {
    volatile union { float f; uint32_t u; } x;
    x.f = f;
    return x.u;
}

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 16);
}

/* Uniform-ish float in [-4, 4) with a few random mantissa bits. */
static float rnd_float(void) {
    return ((float)(rnd() % 65536u) - 32768.0f) / 8192.0f;
}

static void fill_guard(float *buf, int len) {
    for (int i = 0; i < len; i++)
        buf[i] = PROP_GUARD_VALUE;
}

static int guard_intact(const float *buf, int from, int to) {
    for (int i = from; i < to; i++) {
        if (fbits(buf[i]) != fbits(PROP_GUARD_VALUE))
            return 0;
    }
    return 1;
}

static void test_nonpositive_n(void) {
    float a[16], b[16], out[16];
    for (int i = 0; i < 16; i++) {
        a[i] = 1.5f;
        b[i] = 2.0f;
    }
    const int sizes[] = {-8, -1, 0};
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        ASSERT_INT_EQ(fbits(pn_dot_f32(a, b, n)), 0);
        ASSERT_INT_EQ(fbits(pn_vsum_f32(a, n)), 0);
        fill_guard(out, 16);
        pn_vadd_f32(a, b, out, n);
        pn_vmul_f32(a, b, out, n);
        pn_vscale_f32(a, 3.0f, out, n);
        pn_vsilu_mul_f32(a, b, out, n);
        pn_vrms_norm_f32(a, b, out, n, 1e-5f);
        ASSERT_TRUE(guard_intact(out, 0, 16));
    }
}

static void test_elementwise_extent_and_values(void) {
    float a[64], b[64], out[64 + PROP_GUARD];
    for (int n = 1; n <= 64; n++) {
        for (int i = 0; i < n; i++) {
            a[i] = rnd_float();
            b[i] = rnd_float();
        }
        volatile float scale = rnd_float();

        fill_guard(out, n + PROP_GUARD);
        pn_vadd_f32(a, b, out, n);
        for (int i = 0; i < n; i++) {
            volatile float want = a[i] + b[i];
            ASSERT_INT_EQ(fbits(out[i]), fbits(want));
        }
        ASSERT_TRUE(guard_intact(out, n, n + PROP_GUARD));

        fill_guard(out, n + PROP_GUARD);
        pn_vmul_f32(a, b, out, n);
        for (int i = 0; i < n; i++) {
            volatile float want = a[i] * b[i];
            ASSERT_INT_EQ(fbits(out[i]), fbits(want));
        }
        ASSERT_TRUE(guard_intact(out, n, n + PROP_GUARD));

        fill_guard(out, n + PROP_GUARD);
        pn_vscale_f32(a, scale, out, n);
        for (int i = 0; i < n; i++) {
            volatile float want = a[i] * scale;
            ASSERT_INT_EQ(fbits(out[i]), fbits(want));
        }
        ASSERT_TRUE(guard_intact(out, n, n + PROP_GUARD));

        fill_guard(out, n + PROP_GUARD);
        pn_vsilu_mul_f32(a, b, out, n);
        ASSERT_TRUE(guard_intact(out, n, n + PROP_GUARD));

        fill_guard(out, n + PROP_GUARD);
        pn_vrms_norm_f32(a, b, out, n, 1e-5f);
        ASSERT_TRUE(guard_intact(out, n, n + PROP_GUARD));
    }
}

/* silu is excluded: on x86 the in-place alias fallback divides exactly while the
 * vectorized body uses a reciprocal estimate, so its bits legitimately differ. */
static void test_in_place_matches_out_of_place(void) {
    float a[64], b[64], ref[64], work[64];
    for (int n = 1; n <= 64; n++) {
        for (int i = 0; i < n; i++) {
            a[i] = rnd_float();
            b[i] = rnd_float();
        }

        pn_vadd_f32(a, b, ref, n);
        memcpy(work, a, sizeof(float) * (size_t)n);
        pn_vadd_f32(work, b, work, n);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);
        memcpy(work, b, sizeof(float) * (size_t)n);
        pn_vadd_f32(a, work, work, n);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);

        pn_vmul_f32(a, b, ref, n);
        memcpy(work, a, sizeof(float) * (size_t)n);
        pn_vmul_f32(work, b, work, n);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);

        pn_vscale_f32(a, 0.75f, ref, n);
        memcpy(work, a, sizeof(float) * (size_t)n);
        pn_vscale_f32(work, 0.75f, work, n);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);

        pn_vrms_norm_f32(a, b, ref, n, 1e-6f);
        memcpy(work, a, sizeof(float) * (size_t)n);
        pn_vrms_norm_f32(work, b, work, n, 1e-6f);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);

        pn_vrms_norm_f32(a, a, ref, n, 1e-6f);
        memcpy(work, a, sizeof(float) * (size_t)n);
        pn_vrms_norm_f32(work, work, work, n, 1e-6f);
        ASSERT_INT_EQ(memcmp(work, ref, sizeof(float) * (size_t)n), 0);
    }
}

/* Small integers keep every partial sum exact, so any summation order gives the same
 * answer and a skipped, doubled or misplaced element shows up as a wrong integer. */
static void test_integer_reductions_exact(void) {
    float a[PROP_MAX_N], b[PROP_MAX_N];
    for (int n = 1; n <= PROP_MAX_N; n++) {
        long dot_want = 0, sum_want = 0;
        for (int i = 0; i < n; i++) {
            int ai = (int)(rnd() % 17u) - 8;
            int bi = (int)(rnd() % 17u) - 8;
            a[i] = (float)ai;
            b[i] = (float)bi;
            dot_want += (long)ai * bi;
            sum_want += ai;
        }
        ASSERT_INT_EQ((long)pn_dot_f32(a, b, n), dot_want);
        ASSERT_INT_EQ((long)pn_vsum_f32(a, n), sum_want);
    }
}

static void test_q4_0_element_mapping(void) {
    pn_block_q4_0 blk;
    float vec[32];
    for (int trial = 0; trial < 64; trial++) {
        blk.d = 0x3C00; /* 1.0 */
        for (int j = 0; j < 16; j++)
            blk.qs[j] = (uint8_t)rnd();
        for (int e = 0; e < 32; e++) {
            memset(vec, 0, sizeof(vec));
            vec[e] = 1.0f;
            int code = e < 16 ? (blk.qs[e] & 0x0F) : (blk.qs[e - 16] >> 4);
            ASSERT_INT_EQ((long)pn_dot_q4_0_f32(&blk, vec), code - 8);
        }
        blk.d = 0x4000; /* 2.0 */
        long want = 0;
        for (int e = 0; e < 32; e++) {
            vec[e] = 1.0f;
            int code = e < 16 ? (blk.qs[e] & 0x0F) : (blk.qs[e - 16] >> 4);
            want += 2 * (code - 8);
        }
        ASSERT_INT_EQ((long)pn_dot_q4_0_f32(&blk, vec), want);
    }
}

static void test_q8_0_element_mapping(void) {
    pn_block_q8_0 blk;
    float vec[32];
    for (int trial = 0; trial < 64; trial++) {
        blk.d = 0x4000; /* 2.0 */
        for (int j = 0; j < 32; j++)
            blk.qs[j] = (int8_t)(rnd() & 0xFF);
        long want = 0;
        for (int e = 0; e < 32; e++) {
            memset(vec, 0, sizeof(vec));
            vec[e] = 1.0f;
            ASSERT_INT_EQ((long)pn_dot_q8_0_f32(&blk, vec), 2L * blk.qs[e]);
            want += 2L * blk.qs[e];
        }
        for (int e = 0; e < 32; e++)
            vec[e] = 1.0f;
        ASSERT_INT_EQ((long)pn_dot_q8_0_f32(&blk, vec), want);
    }
}

/* One big element (square 2^24) then many ones: a float sum loses most of the ones,
 * a double sum is exact. With x = w = 1 the output is r itself. */
static void test_rms_double_accumulation(void) {
    enum { RMS_N = 4097 };
    static float x[RMS_N], w[RMS_N], out[RMS_N];
    x[0] = 4096.0f;
    w[0] = 1.0f;
    for (int i = 1; i < RMS_N; i++) {
        x[i] = 1.0f;
        w[i] = 1.0f;
    }
    double ss = 16777216.0 + (double)(RMS_N - 1);
    float r_want = (float)(1.0 / sqrt(ss / (double)RMS_N));
    pn_vrms_norm_f32(x, w, out, RMS_N, 0.0f);
    for (int i = 1; i < RMS_N; i++)
        ASSERT_INT_EQ(fbits(out[i]), fbits(r_want));
}

static void test_silu_mul_reference(void) {
    float gate[64], up[64], out[64];
    for (int i = 0; i < 64; i++) {
        gate[i] = rnd_float();
        up[i] = rnd_float();
    }
    for (int n = 1; n <= 64; n++) {
        pn_vsilu_mul_f32(gate, up, out, n);
        for (int i = 0; i < n; i++) {
            double g = gate[i];
            double want = g / (1.0 + exp(-g)) * up[i];
            ASSERT_TRUE(fabs(out[i] - want) <= 1e-5 * (fabs(want) + 1e-3));
        }
    }
}

int main(void) {
    printf("\n  SIMD kernel property tests\n\n");

    RUN_TEST(nonpositive_n);
    RUN_TEST(elementwise_extent_and_values);
    RUN_TEST(in_place_matches_out_of_place);
    RUN_TEST(integer_reductions_exact);
    RUN_TEST(q4_0_element_mapping);
    RUN_TEST(q8_0_element_mapping);
    RUN_TEST(rms_double_accumulation);
    RUN_TEST(silu_mul_reference);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
