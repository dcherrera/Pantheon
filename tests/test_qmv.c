/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Quantized matvec tests.
 */

#include "lib/test.h"
#include "../src/core/qweight.h"
#include "../src/formats/gguf_dequant.h"
#include "../src/accel/simd.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Test Q4_0 quantized dot product matches dequant+dot reference */
static void test_qmv_q4_0(void) {
    /* Create a Q4_0 block with known values */
    pn_block_q4_0 block;
    block.d = 0x3C00;  /* FP16 1.0 */
    memset(block.qs, 0x88, 16);  /* All nibbles = 8 -> (8-8)*1.0 = 0.0 */

    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;

    /* Quantized dot should be 0 (all weights are zero after dequant) */
    float result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q4_0);
    ASSERT_FLT_NEAR(0.0f, result, 1e-4f);

    /* Now test with non-trivial values: lo=10, hi=6 -> (10-8)=2, (6-8)=-2 */
    block.d = 0x3C00;  /* FP16 1.0 */
    memset(block.qs, 0x6A, 16);  /* lo=0xA=10, hi=0x6=6 */

    /* Reference: sum of (10-8)*1.0*1.0 for 16 lo + (6-8)*1.0*1.0 for 16 hi = 32 + (-32) = 0 */
    result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q4_0);
    ASSERT_FLT_NEAR(0.0f, result, 1e-4f);

    /* Asymmetric test: lo=12, hi=4 -> (12-8)=4, (4-8)=-4, same sum = 0 */
    memset(block.qs, 0x4C, 16);  /* lo=0xC=12, hi=0x4=4 */
    result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q4_0);
    ASSERT_FLT_NEAR(0.0f, result, 1e-4f);

    /* Test where sum != 0: lo=15, hi=0 -> (15-8)=7, (0-8)=-8
     * With all vec=1: 16*7 + 16*(-8) = 112 - 128 = -16 */
    memset(block.qs, 0x0F, 16);  /* lo=15, hi=0 */
    result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q4_0);
    ASSERT_FLT_NEAR(-16.0f, result, 1e-4f);
}

/* Test Q8_0 quantized dot product */
static void test_qmv_q8_0(void) {
    pn_block_q8_0 block;
    block.d = 0x3C00;  /* FP16 1.0 */

    /* Set all qs to 1 */
    memset(block.qs, 1, 32);

    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;

    /* dot = d * sum(qs[i] * vec[i]) = 1.0 * 32 * 1 = 32.0 */
    float result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q8_0);
    ASSERT_FLT_NEAR(32.0f, result, 1e-4f);

    /* Scale = 0.5 */
    block.d = 0x3800;
    result = pn_qdot(&block, vec, 32, GGUF_QUANT_Q8_0);
    ASSERT_FLT_NEAR(16.0f, result, 1e-4f);
}

/* ---- Q2_0_g128 (GGUF type 42 native ternary) ---------------------------- */

/* Pack n codes (0..2) into a Q2_0_g128 row: per 128-group -> [fp16 d][32 code bytes].
 * Returns malloc'd row buffer; caller frees. Fills codes_out[] and dvals_out[] (per group). */
static uint8_t *q2_pack_row(const uint8_t *codes, int n, const float *dvals, int *row_bytes) {
    int n_blocks = (n + 127) / 128;
    int bytes = n_blocks * 34;
    uint8_t *row = (uint8_t *)calloc(bytes, 1);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *blk = row + (size_t)b * 34;
        uint16_t dh = pn_float_to_half(dvals[b]);
        memcpy(blk, &dh, 2);
        uint8_t *qs = blk + 2;
        int start = b * 128;
        int end = start + 128; if (end > n) end = n;
        for (int i = start; i < end; i++) {
            int li = i - start;
            qs[li >> 2] |= (uint8_t)((codes[i] & 3) << ((li & 3) * 2));
        }
    }
    *row_bytes = bytes;
    return row;
}

/* INDEPENDENT dequant-then-dot reference: run the real pn_dequantize (type 42)
 * to expand the packed row to a dense float weight vector (w[i] = d*(code-1)),
 * then a plain sequential dot with vec[]. This is the oracle for CPU correctness.
 * (It uses per-element scale multiply + naive summation, so it differs from the
 *  group-factored kernel only in the last ULP -> checked with a tight tolerance;
 *  the STRICT per-element bit-exact bar is GPU-vs-CPU, where both orders are pinned.) */
static float q2_ref_dot(const uint8_t *row, int n, const float *vec) {
    float *w = (float *)malloc(n * sizeof(float));
    pn_dequantize(row, w, (size_t)n, GGUF_QUANT_Q2_0_G128);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += w[i] * vec[i];
    free(w);
    return sum;
}

/* Worst-case bound on the error dot_q2_0_g128_row is ENTITLED to.
 *
 * The kernel Q8_0-quantizes the ACTIVATION in 32-element sub-blocks and
 * integer-dots the ternary codes against it, so each vec element lands within
 * d_act/2 of its true value with d_act = fp16(amax/127) over that sub-block.
 * The dot's error is therefore bounded by SUM |w_i| * d_act_i / 2. Derived from
 * the same public inputs the caller has — it does not reach into the kernel, so
 * it cannot inherit a bug from it the way a re-implementation would. */
static float q2_activation_error_bound(const uint8_t *row, int n,
                                       const float *vec) {
    float *w = (float *)malloc(n * sizeof(float));
    pn_dequantize(row, w, (size_t)n, GGUF_QUANT_Q2_0_G128);
    float bound = 0.0f;
    for (int start = 0; start < n; start += 32) {
        int cnt = (start + 32 <= n) ? 32 : n - start;
        float amax = 0.0f, wsum = 0.0f;
        for (int j = 0; j < cnt; j++) {
            float a = fabsf(vec[start + j]);
            if (a > amax) amax = a;
            wsum += fabsf(w[start + j]);
        }
        bound += wsum * (amax / 127.0f) * 0.5f;
    }
    free(w);
    return bound * 1.05f + 1e-4f;   /* + the fp16 storage of d_act itself */
}

static void q2_shape_check(int n, unsigned seed) {
    int n_blocks = (n + 127) / 128;
    uint8_t *codes = (uint8_t *)malloc(n);
    float *dvals = (float *)malloc(n_blocks * sizeof(float));
    float *vec = (float *)malloc(n * sizeof(float));
    srand(seed);
    for (int i = 0; i < n; i++) codes[i] = (uint8_t)(rand() % 3);        /* {0,1,2} true ternary */
    for (int b = 0; b < n_blocks; b++) dvals[b] = ((rand() % 2000) - 1000) / 10000.0f; /* ~fp16 scales */
    for (int i = 0; i < n; i++) vec[i] = ((rand() % 4000) - 2000) / 1000.0f;

    int row_bytes = 0;
    uint8_t *row = q2_pack_row(codes, n, dvals, &row_bytes);

    float got = pn_qdot(row, vec, n, GGUF_QUANT_Q2_0_G128);
    float ref = q2_ref_dot(row, n, vec);

    /* Independent dequant-then-dot oracle. The kernel is NOT bit-exact against
     * it and must not be asserted to be: it quantizes the activation to int8
     * (see q2_activation_error_bound). Two assertions, because one alone is
     * either unsound or useless:
     *
     *   SOUNDNESS — the error stays inside what the algorithm is entitled to.
     *     Never flakes, catches gross breakage.
     *   ACCURACY  — and it stays FAR inside it. The worst-case bound sums n
     *     per-element errors, but their SIGNS are independent, so the realized
     *     error grows like sqrt(n), not n: err ~ bound * K / sqrt(n). Measured
     *     K on the real Bonsai shapes is 0.63 .. 1.19, so K = 2 keeps ~1.7x
     *     headroom and still fails a change that doubles the error — 6-bit
     *     activations instead of 7-bit, say, which the sound check alone waves
     *     through. Seeds are fixed, so this is deterministic, not sampled.
     *
     *     ONLY above n = 512. At n = 128 the walk is 4 sub-blocks long and K
     *     measures 1.44 with no way to tell a real regression from the small-
     *     sample variance; asserting there would be a coin flip dressed as a
     *     bound. The small shapes keep the soundness check only. */
    float bound = q2_activation_error_bound(row, n, vec);
    float err = fabsf(ref - got);
    ASSERT_TRUE(err <= bound);
    if (n >= 512)
        ASSERT_TRUE(err <= 2.0f * bound / sqrtf((float)n));

    free(codes); free(dvals); free(vec); free(row);
}

static void test_qmv_q2_0_g128(void) {
    /* Trivial: all code=2 (+1), d=1.0, vec=1 -> sum = n.
     *
     * NOT bit-exact, and deliberately so. dot_q2_0_g128_row Q8_0-quantizes the
     * ACTIVATION in 32-elem sub-blocks and integer-dots against it (matching
     * ggml's ternary matvec), so the sub-block scale d_act = fp16(amax/127)
     * carries fp16 mantissa error even when the weights are exact integers.
     * Here that is d_act = fp16(1/127) = 0.007873535 against 1/127 =
     * 0.007874016, and 16256 * d_act = 127.99219 rather than 128 — a relative
     * error of 2^-11, the fp16 mantissa step, not a defect. An exact-equality
     * assertion here would be an assertion that the kernel does NOT quantize
     * activations, which is the opposite of its design.
     *
     * So: the same relative tolerance the randomized shape check uses. The
     * all-zero arm below stays tight, because zero is exact under any scale. */
    int n = 128;
    uint8_t *codes = (uint8_t *)malloc(n);
    float dvals[1] = { 1.0f };
    float *vec = (float *)malloc(n * sizeof(float));
    memset(codes, 2, n);
    for (int i = 0; i < n; i++) vec[i] = 1.0f;
    int rb;
    float tol = 1e-3f * ((float)n + 1.0f);
    uint8_t *row = q2_pack_row(codes, n, dvals, &rb);
    float r = pn_qdot(row, vec, n, GGUF_QUANT_Q2_0_G128);
    ASSERT_FLT_NEAR((float)n, r, tol);
    /* all code=0 (-1) -> -n */
    memset(codes, 0, n); free(row); row = q2_pack_row(codes, n, dvals, &rb);
    r = pn_qdot(row, vec, n, GGUF_QUANT_Q2_0_G128);
    ASSERT_FLT_NEAR(-(float)n, r, tol);
    /* all code=1 (0) -> 0, and this one IS exact: every term is 0*q. */
    memset(codes, 1, n); free(row); row = q2_pack_row(codes, n, dvals, &rb);
    r = pn_qdot(row, vec, n, GGUF_QUANT_Q2_0_G128);
    ASSERT_FLT_NEAR(0.0f, r, 1e-4f);
    free(codes); free(vec); free(row);

    /* Representative Bonsai shapes (the contiguous/ne0 dim = cols): 5120, 17408, 6144 */
    q2_shape_check(5120, 1);    /* attn proj 5120x5120 col dim */
    q2_shape_check(17408, 2);   /* ffn down 5120x17408 col dim */
    q2_shape_check(6144, 3);    /* ssm inner */
    q2_shape_check(5120, 4);
    q2_shape_check(640, 5);     /* small, exercises multi-block */
    q2_shape_check(128, 6);     /* single block */
}

/* Test F32 qmv matches tm_mv reference */
static void test_qmv_f32(void) {
    /* 2x4 F32 weight matrix */
    float w_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };

    pn_qweight qw;
    qw.data = w_data;
    qw.type = GGUF_QUANT_F32;
    qw.rows = 2;
    qw.cols = 4;
    qw.row_stride = 4 * sizeof(float);

    float vec[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float out[2];
    pn_qmv(&qw, vec, out);

    /* row 0: 1+2+3+4 = 10, row 1: 5+6+7+8 = 26 */
    ASSERT_FLT_NEAR(10.0f, out[0], 1e-4f);
    ASSERT_FLT_NEAR(26.0f, out[1], 1e-4f);
}

/* Test null safety */
static void test_qmv_null_safety(void) {
    float vec[4] = {1, 2, 3, 4};
    float out[2] = {0};

    /* NULL weight should not crash */
    pn_qmv(NULL, vec, out);
    ASSERT_FLT_NEAR(0.0f, out[0], 1e-6f);

    /* NULL vec */
    pn_qweight qw = {0};
    pn_qmv(&qw, NULL, out);
    ASSERT_FLT_NEAR(0.0f, out[0], 1e-6f);
}

int main(void) {
    printf("Quantized Matvec Tests\n");
    printf("======================\n\n");

    RUN_TEST(qmv_q4_0);
    RUN_TEST(qmv_q8_0);
    RUN_TEST(qmv_q2_0_g128);
    RUN_TEST(qmv_f32);
    RUN_TEST(qmv_null_safety);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
