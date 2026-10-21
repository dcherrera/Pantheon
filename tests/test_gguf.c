/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * GGUF parser and dequantization tests.
 */

#include "lib/test.h"
#include "../src/formats/gguf.h"
#include "../src/formats/gguf_dequant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================
 * Dequantization tests
 * ============================================================ */

/* Test Q4_0 dequantization with known values */
static void test_dequant_q4_0(void) {
    pn_block_q4_0 block;
    block.d = 0x3C00;  /* FP16 1.0 */
    memset(block.qs, 0x88, 16);  /* All nibbles = 8 -> (8-8)*1.0 = 0.0 */

    float out[32];
    pn_dequantize_block_q4_0(&block, out);

    for (int i = 0; i < 32; i++) {
        ASSERT_FLT_NEAR(0.0f, out[i], 1e-6f);
    }

    /* Test with scale=2.0, first byte has lo=0, hi=15 */
    block.d = 0x4000;  /* FP16 2.0 */
    memset(block.qs, 0, 16);
    block.qs[0] = 0xF0;  /* lo=0, hi=15 */

    pn_dequantize_block_q4_0(&block, out);

    ASSERT_FLT_NEAR(-16.0f, out[0], 1e-4f);
    ASSERT_FLT_NEAR(14.0f, out[16], 1e-4f);
}

/* Test Q8_0 dequantization with known values */
static void test_dequant_q8_0(void) {
    pn_block_q8_0 block;
    block.d = 0x3C00;  /* FP16 1.0 */

    memset(block.qs, 0, 32);
    block.qs[0] = 0;
    block.qs[1] = 1;
    block.qs[2] = -1;
    block.qs[3] = 127;
    block.qs[4] = -128;

    float out[32];
    pn_dequantize_block_q8_0(&block, out);

    ASSERT_FLT_NEAR(0.0f, out[0], 1e-6f);
    ASSERT_FLT_NEAR(1.0f, out[1], 1e-6f);
    ASSERT_FLT_NEAR(-1.0f, out[2], 1e-6f);
    ASSERT_FLT_NEAR(127.0f, out[3], 1e-6f);
    ASSERT_FLT_NEAR(-128.0f, out[4], 1e-6f);

    /* Test with scale=0.5 */
    block.d = 0x3800;  /* FP16 0.5 */
    pn_dequantize_block_q8_0(&block, out);
    ASSERT_FLT_NEAR(0.0f, out[0], 1e-6f);
    ASSERT_FLT_NEAR(0.5f, out[1], 1e-6f);
    ASSERT_FLT_NEAR(-0.5f, out[2], 1e-6f);
}

/* Test F16 dequantization */
static void test_dequant_f16(void) {
    uint16_t src[] = { 0x0000, 0x3C00, 0xBC00, 0x4000, 0x3800 };
    float dst[5];

    int ret = pn_dequantize(src, dst, 5, GGUF_QUANT_F16);
    ASSERT_INT_EQ(0, ret);

    ASSERT_FLT_NEAR(0.0f, dst[0], 1e-6f);
    ASSERT_FLT_NEAR(1.0f, dst[1], 1e-6f);
    ASSERT_FLT_NEAR(-1.0f, dst[2], 1e-6f);
    ASSERT_FLT_NEAR(2.0f, dst[3], 1e-6f);
    ASSERT_FLT_NEAR(0.5f, dst[4], 1e-6f);
}

/* Test F32 passthrough */
static void test_dequant_f32(void) {
    float src[] = { 1.0f, -2.5f, 3.14159f, 0.0f };
    float dst[4];

    int ret = pn_dequantize(src, dst, 4, GGUF_QUANT_F32);
    ASSERT_INT_EQ(0, ret);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(src[i], dst[i], 1e-7f);
    }
}

/* Test block size and type size functions */
static void test_block_sizes(void) {
    ASSERT_INT_EQ(1,  pn_gguf_block_size(GGUF_QUANT_F32));
    ASSERT_INT_EQ(1,  pn_gguf_block_size(GGUF_QUANT_F16));
    ASSERT_INT_EQ(32, pn_gguf_block_size(GGUF_QUANT_Q4_0));
    ASSERT_INT_EQ(32, pn_gguf_block_size(GGUF_QUANT_Q8_0));

    /* K-quant block sizes */
    ASSERT_INT_EQ(256, pn_gguf_block_size(GGUF_QUANT_Q4_K));
    ASSERT_INT_EQ(256, pn_gguf_block_size(GGUF_QUANT_Q5_K));
    ASSERT_INT_EQ(256, pn_gguf_block_size(GGUF_QUANT_Q6_K));

    /* Type sizes: 128 elements */
    ASSERT_INT_EQ(128 * 4, (int)pn_gguf_type_size(GGUF_QUANT_F32, 128));
    ASSERT_INT_EQ(128 * 2, (int)pn_gguf_type_size(GGUF_QUANT_F16, 128));
    /* Q4_0: 128/32 blocks * 18 bytes = 72 */
    ASSERT_INT_EQ(72, (int)pn_gguf_type_size(GGUF_QUANT_Q4_0, 128));
    /* Q8_0: 128/32 blocks * 34 bytes = 136 */
    ASSERT_INT_EQ(136, (int)pn_gguf_type_size(GGUF_QUANT_Q8_0, 128));

    /* K-quant type sizes: 256 elements */
    ASSERT_INT_EQ(144, (int)pn_gguf_type_size(GGUF_QUANT_Q4_K, 256));
    ASSERT_INT_EQ(176, (int)pn_gguf_type_size(GGUF_QUANT_Q5_K, 256));
    ASSERT_INT_EQ(210, (int)pn_gguf_type_size(GGUF_QUANT_Q6_K, 256));

    /* Bytes per block */
    ASSERT_INT_EQ(144, (int)pn_gguf_bytes_per_block(GGUF_QUANT_Q4_K));
    ASSERT_INT_EQ(176, (int)pn_gguf_bytes_per_block(GGUF_QUANT_Q5_K));
    ASSERT_INT_EQ(210, (int)pn_gguf_bytes_per_block(GGUF_QUANT_Q6_K));
}

/* Test unsupported type returns -1 */
static void test_dequant_unsupported(void) {
    float dummy_src = 0.0f;
    float dummy_dst = 0.0f;
    int ret = pn_dequantize(&dummy_src, &dummy_dst, 1, GGUF_QUANT_Q2_K);
    ASSERT_INT_EQ(-1, ret);
}

/* Test bulk Q4_0 dequantization (multiple blocks) */
static void test_dequant_q4_0_bulk(void) {
    pn_block_q4_0 blocks[2];

    blocks[0].d = 0x3C00;
    memset(blocks[0].qs, 0x88, 16);

    blocks[1].d = 0x3800;
    memset(blocks[1].qs, 0x99, 16);

    float out[64];
    int ret = pn_dequantize(blocks, out, 64, GGUF_QUANT_Q4_0);
    ASSERT_INT_EQ(0, ret);

    for (int i = 0; i < 32; i++) {
        ASSERT_FLT_NEAR(0.0f, out[i], 1e-6f);
    }
    for (int i = 32; i < 64; i++) {
        ASSERT_FLT_NEAR(0.5f, out[i], 1e-4f);
    }
}

/* ============================================================
 * K-quant dequantization tests
 * ============================================================ */

/* Test Q4_K dequantization */
static void test_dequant_q4_k(void) {
    pn_block_q4_k block;
    memset(&block, 0, sizeof(block));

    /* d=1.0, dmin=0.0 — simplifies formula to: d * sc[j] * nibble */
    block.d = 0x3C00;     /* FP16 1.0 */
    block.dmin = 0x0000;  /* FP16 0.0 */

    /* Set scale[0] = 1 (low 6 bits of scales_raw[0]), rest = 0 */
    block.scales[0] = 1;
    memset(block.scales + 1, 0, 11);

    /* ggml layout: 32-byte qs chunk, low nibbles → sub-block 0, high nibbles → sub-block 1.
     * Set all 32 bytes of first chunk to 0x05 so low nibble = 5, high nibble = 0. */
    memset(block.qs, 0x05, 32);  /* lo=5, hi=0 for 32 bytes */
    memset(block.qs + 32, 0, 96);

    float out[256];
    pn_dequantize_block_q4_k(&block, out);

    /* Sub-block 0 (low nibbles): d * sc[0] * 5 - 0 = 5.0 */
    for (int i = 0; i < 32; i++) {
        ASSERT_FLT_NEAR(5.0f, out[i], 1e-4f);
    }

    /* Sub-block 1+ (sc[1..7]=0): 0.0 */
    for (int i = 32; i < 256; i++) {
        ASSERT_FLT_NEAR(0.0f, out[i], 1e-4f);
    }
}

/* Test Q6_K dequantization */
static void test_dequant_q6_k(void) {
    pn_block_q6_k block;
    memset(&block, 0, sizeof(block));

    /* d=1.0, scale[0]=1, rest=0 */
    block.d = 0x3C00;  /* FP16 1.0 */
    block.scales[0] = 1;

    /* Set first 16 elements to have q6 = 32 (which gives value = 1*(32-32) = 0) */
    /* q6 = low4 | (high2 << 4). For q6=32: low4=0, high2=2 */
    /* For first 8 elements (indices 0-7): ql bytes 0-3 contain pairs, qh byte 0 has pairs */
    /* Just test that zero block produces all zeros minus offset */
    float out[256];
    pn_dequantize_block_q6_k(&block, out);

    /* All zeros with scale=1: q6=0 for all, value = 1 * 1 * (0 - 32) = -32 */
    ASSERT_FLT_NEAR(-32.0f, out[0], 1e-4f);

    /* Sub-blocks with scale=0 should produce 0 regardless of q6 value */
    for (int i = 16; i < 256; i++) {
        ASSERT_FLT_NEAR(0.0f, out[i], 1e-4f);
    }
}

/* Test Q5_K dequantization */
static void test_dequant_q5_k(void) {
    pn_block_q5_k block;
    memset(&block, 0, sizeof(block));

    /* d=1.0, dmin=0.0, scale[0]=1 */
    block.d = 0x3C00;
    block.dmin = 0x0000;
    block.scales[0] = 1;

    /* ggml layout: 32-byte qs chunk, low nibbles → sub-block 0, high nibbles → sub-block 1.
     * Set all 32 bytes to 0x03 so low nibble = 3, high nibble = 0. */
    memset(block.qs, 0x03, 32);  /* lo=3, hi=0 for 32 bytes */
    memset(block.qs + 32, 0, 96);
    memset(block.qh, 0, 32);    /* all high bits = 0 */

    float out[256];
    pn_dequantize_block_q5_k(&block, out);

    /* Sub-block 0 (low nibbles): d * sc[0] * 3 - 0 = 3.0 */
    for (int i = 0; i < 32; i++) {
        ASSERT_FLT_NEAR(3.0f, out[i], 1e-4f);
    }

    /* Rest should be 0 */
    for (int i = 32; i < 256; i++) {
        ASSERT_FLT_NEAR(0.0f, out[i], 1e-4f);
    }
}

/* Test K-quant struct sizes match expected */
static void test_kquant_struct_sizes(void) {
    ASSERT_INT_EQ(144, (int)sizeof(pn_block_q4_k));
    ASSERT_INT_EQ(176, (int)sizeof(pn_block_q5_k));
    ASSERT_INT_EQ(210, (int)sizeof(pn_block_q6_k));
}

int main(void) {
    printf("GGUF & Dequantization Tests\n");
    printf("===========================\n\n");

    RUN_TEST(dequant_q4_0);
    RUN_TEST(dequant_q8_0);
    RUN_TEST(dequant_f16);
    RUN_TEST(dequant_f32);
    RUN_TEST(block_sizes);
    RUN_TEST(dequant_unsupported);
    RUN_TEST(dequant_q4_0_bulk);
    RUN_TEST(dequant_q4_k);
    RUN_TEST(dequant_q5_k);
    RUN_TEST(dequant_q6_k);
    RUN_TEST(kquant_struct_sizes);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
