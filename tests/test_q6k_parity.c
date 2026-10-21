/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Parity test for the fused NEON Q6_K dequant-dot in src/core/qweight.c.
 *
 * The fused kernel (reached through pn_qdot / dot_q6_k on NEON) must produce the
 * same dot product as the reference path that dequantizes each Q6_K super-block
 * with pn_dequantize_block_q6_k and then calls pn_dot_f32. We build many random
 * VALID Q6_K blocks (random ql/qh bytes, random int8 sub-block scales, a finite
 * f16 super-scale) and assert the two agree within a tight relative tolerance,
 * for single-block and multi-block rows.
 *
 * Build (NEON, Apple):
 *   cc -O3 -std=c11 -Iinclude -I. -I./foundry \
 *      tests/test_q6k_parity.c libpantheon.a -framework Accelerate -lm \
 *      -o tests/test_q6k_parity
 */

#include "src/core/qweight.h"
#include "src/formats/gguf_dequant.h"
#include "src/accel/simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define Q6K_BYTES 210   /* sizeof(pn_block_q6_k) */

/* Fused NEON accumulation (FMA + reordered horizontal sum, under -ffast-math)
 * differs from the scalar dequant-then-dot reference only at the fp32 rounding
 * level. Near a dot-product zero-crossing that rounding is large *relative* to
 * the tiny result, so a pure relative test is not meaningful there. We use the
 * standard combined bound |got-ref| <= ATOL + RTOL*|ref|. RTOL sits just above
 * the worst reorder noise seen over these adversarial full-range-scale blocks
 * (observed max rel err ~1.6e-3); real Q6_K weights and normalized activations
 * stay far tighter. ATOL covers the exact zero-crossings where rel is undefined. */
#define RTOL   3e-3f
#define ATOL   1e-3f

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t xrand(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

/* Fill one valid Q6_K block: ql[128] low nibbles, qh[64] 2-bit highs,
 * scales[16] int8, d one finite fp16. Layout must match pn_block_q6_k exactly. */
static void make_block(uint8_t *blk) {
    for (int i = 0; i < 128; i++) blk[i] = (uint8_t)(xrand() & 0xFF);       /* ql */
    for (int i = 0; i < 64; i++)  blk[128 + i] = (uint8_t)(xrand() & 0xFF); /* qh */
    for (int i = 0; i < 16; i++)  blk[192 + i] = (int8_t)(xrand() & 0xFF);  /* scales */
    /* d: pick a finite, non-tiny fp16 (exponent ~2^-4..2^-1, random mantissa).
     * Both paths decode d identically, so any finite value proves parity. */
    uint16_t dbits = (uint16_t)(0x2C00u | (xrand() & 0x03FFu));
    memcpy(blk + 208, &dbits, 2);
}

static void make_vec(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = ((float)(xrand() % 4000) / 1000.0f) - 2.0f;   /* [-2, 2) */
    }
}

/* Reference: dequantize each block then pn_dot_f32. */
static float ref_dot(const uint8_t *row, const float *vec, int n_blocks) {
    float tmp[256];
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; i++) {
        pn_dequantize_block_q6_k(row + (size_t)i * Q6K_BYTES, tmp);
        sum += pn_dot_f32(tmp, vec + i * 256, 256);
    }
    return sum;
}

int main(void) {
    if (sizeof(pn_block_q6_k) != Q6K_BYTES) {
        fprintf(stderr, "unexpected sizeof(pn_block_q6_k)=%zu\n",
                sizeof(pn_block_q6_k));
        return 1;
    }

    int fails = 0;
    int total = 0;
    float max_rel = 0.0f;
    float max_abs = 0.0f;

    /* Part 1: single-block parity, many random blocks. */
    const int N_SINGLE = 20000;
    for (int t = 0; t < N_SINGLE; t++) {
        uint8_t blk[Q6K_BYTES];
        float vec[256];
        make_block(blk);
        make_vec(vec, 256);

        float ref = ref_dot(blk, vec, 1);
        float got = pn_qdot(blk, vec, 256, GGUF_QUANT_Q6_K);

        float aerr = fabsf(got - ref);
        float rel = aerr / (fabsf(ref) > 1e-6f ? fabsf(ref) : 1e-6f);
        if (rel > max_rel)  max_rel = rel;
        if (aerr > max_abs) max_abs = aerr;
        total++;
        if (aerr > ATOL + RTOL * fabsf(ref)) {
            if (fails < 5)
                fprintf(stderr, "single #%d: ref=%.6f got=%.6f abs=%.2e rel=%.2e\n",
                        t, ref, got, aerr, rel);
            fails++;
        }
    }

    /* Part 2: multi-block rows (2..32 blocks), exercising accumulation. */
    const int N_MULTI = 4000;
    for (int t = 0; t < N_MULTI; t++) {
        int nb = 2 + (int)(xrand() % 31);   /* 2..32 blocks */
        int n = nb * 256;
        uint8_t *row = malloc((size_t)nb * Q6K_BYTES);
        float *vec = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < nb; i++) make_block(row + (size_t)i * Q6K_BYTES);
        make_vec(vec, n);

        float ref = ref_dot(row, vec, nb);
        float got = pn_qdot(row, vec, n, GGUF_QUANT_Q6_K);

        float aerr = fabsf(got - ref);
        float rel = aerr / (fabsf(ref) > 1e-6f ? fabsf(ref) : 1e-6f);
        if (rel > max_rel)  max_rel = rel;
        if (aerr > max_abs) max_abs = aerr;
        total++;
        if (aerr > ATOL + RTOL * fabsf(ref)) {
            if (fails < 5)
                fprintf(stderr, "multi #%d (nb=%d): ref=%.6f got=%.6f abs=%.2e rel=%.2e\n",
                        t, nb, ref, got, aerr, rel);
            fails++;
        }
        free(row);
        free(vec);
    }

    printf("Q6_K parity: %d/%d within |err| <= %.0e + %.0e*|ref| "
           "(max abs err %.3e, max rel err %.3e)\n",
           total - fails, total, (double)ATOL, (double)RTOL,
           (double)max_abs, (double)max_rel);
    if (fails == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (%d mismatches)\n", fails);
    return 1;
}
