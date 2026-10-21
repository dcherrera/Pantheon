/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * SIMD kernel tests.
 */

#include "lib/test.h"
#include "../src/accel/simd.h"
#include "../src/formats/gguf_dequant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Test dot product */
static void test_simd_dot(void) {
    float a[64], b[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }

    /* sum(0..63) = 63*64/2 = 2016 */
    float result = pn_dot_f32(a, b, 64);
    ASSERT_FLT_NEAR(2016.0f, result, 1e-2f);

    /* Test small sizes */
    result = pn_dot_f32(a, b, 3);
    ASSERT_FLT_NEAR(3.0f, result, 1e-4f);  /* 0+1+2 */

    /* Test n=1 */
    float x = 3.0f, y = 4.0f;
    result = pn_dot_f32(&x, &y, 1);
    ASSERT_FLT_NEAR(12.0f, result, 1e-4f);
}

/* Test element-wise add */
static void test_simd_add(void) {
    float a[64], b[64], out[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (float)i;
        b[i] = 100.0f;
    }

    pn_vadd_f32(a, b, out, 64);

    for (int i = 0; i < 64; i++) {
        ASSERT_FLT_NEAR((float)i + 100.0f, out[i], 1e-4f);
    }

    /* In-place */
    pn_vadd_f32(a, b, a, 64);
    ASSERT_FLT_NEAR(100.0f, a[0], 1e-4f);
    ASSERT_FLT_NEAR(163.0f, a[63], 1e-4f);
}

/* Test SiLU + multiply */
static void test_simd_silu(void) {
    float gate[4] = { 0.0f, 1.0f, -1.0f, 10.0f };
    float up[4]   = { 1.0f, 1.0f,  1.0f,  1.0f };
    float out[4];

    pn_vsilu_mul_f32(gate, up, out, 4);

    /* silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 */
    ASSERT_FLT_NEAR(0.0f, out[0], 1e-4f);

    /* silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311 */
    ASSERT_FLT_NEAR(0.7311f, out[1], 1e-3f);

    /* silu(-1) = -1 / (1 + exp(1)) ≈ -0.2689 */
    ASSERT_FLT_NEAR(-0.2689f, out[2], 1e-3f);

    /* silu(10) ≈ 10 (sigmoid(10) ≈ 1) */
    ASSERT_FLT_NEAR(10.0f, out[3], 1e-2f);
}

/* Test fused Q4_0 dot */
static void test_simd_q4_0_dot(void) {
    pn_block_q4_0 block;
    block.d = 0x3C00;  /* scale = 1.0 */

    /* Set all nibbles to 8 (zero after dequant) */
    memset(block.qs, 0x88, 16);

    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;

    float result = pn_dot_q4_0_f32(&block, vec);
    ASSERT_FLT_NEAR(0.0f, result, 1e-4f);

    /* lo=15, hi=0: dequant = [7, 7, ...16x, -8, -8, ...16x]
     * dot with ones = 16*7 + 16*(-8) = -16 */
    memset(block.qs, 0x0F, 16);
    result = pn_dot_q4_0_f32(&block, vec);
    ASSERT_FLT_NEAR(-16.0f, result, 1e-2f);
}

/* Test fused Q8_0 dot */
static void test_simd_q8_0_dot(void) {
    pn_block_q8_0 block;
    block.d = 0x3C00;  /* scale = 1.0 */
    memset(block.qs, 2, 32);  /* all values = 2 */

    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;

    /* dot = 1.0 * 32 * 2 = 64 */
    float result = pn_dot_q8_0_f32(&block, vec);
    ASSERT_FLT_NEAR(64.0f, result, 1e-2f);
}

/* Test RMS normalization */
static void test_simd_rms_norm(void) {
    float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float w[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float out[4];

    pn_vrms_norm_f32(x, w, out, 4, 1e-5f);

    /* rms = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386 */
    /* out[0] = 1.0 / 2.7386 ≈ 0.3651 */
    float rms = sqrtf(30.0f / 4.0f + 1e-5f);
    for (int i = 0; i < 4; i++) {
        float expected = x[i] / rms * w[i];
        ASSERT_FLT_NEAR(expected, out[i], 1e-4f);
    }
}

int main(void) {
    printf("SIMD Kernel Tests\n");
    printf("=================\n\n");

    RUN_TEST(simd_dot);
    RUN_TEST(simd_add);
    RUN_TEST(simd_silu);
    RUN_TEST(simd_q4_0_dot);
    RUN_TEST(simd_q8_0_dot);
    RUN_TEST(simd_rms_norm);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
