/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_tiled.c
 * @brief  Implementation of the FTTW tile decoders (foundry ft_bfp_unpack_tiles underneath).
 *
 * `PN_TILED_RED=1` is the red arm of the oracle's exactness test: the first tile of every
 * decoded tile column / row is scaled by 2 — its exponent off by one — so a forward over
 * it must DIFFER from the reference.
 *
 * Bfp8_b / Bfp4_b tile columns (every CPU-oracle matvec) decode through a NEON path on arm64
 * (tiled_fast_column): the same bits as ft_bfp_unpack_tiles + the scalar transpose. A one-row
 * matvec over them (pn_tiled_dot_column) also skips the fp32 panel, whose DRAM round trip bound
 * the oracle, keeping pn_dot_f32's exact accumulation order (mixw plan 2.3). Other hosts and
 * formats keep the codec path. tests/test_tiled_fast_decode checks every datum x exponent.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "gguf_tiled.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) /* = pn_dot_f32's NEON path (simd.c) */
#include <arm_neon.h>
#define TILED_FAST_NEON 1
#endif

#include "tt/convert/ft_bfp.h"

/* ft_tt_tensor_format / ft_tt_tensor_layout values written into FTTW directories. */
#define TILED_FTTW_FORMAT_FP32 0u
#define TILED_FTTW_FORMAT_BFP8_B 2u
#define TILED_FTTW_FORMAT_BFP4_B 3u
#define TILED_FTTW_FORMAT_BFP6_B 4u
#define TILED_FTTW_LAYOUT_SOURCE 0u
#define TILED_FTTW_LAYOUT_NMAJOR 1u
#define TILED_FP32_TILE_BYTES 4096u
#define TILED_FACE 16

int pn_tiled_is_type(pn_gguf_quant type)
{
    return type == PN_QUANT_TILED_FP32 || type == PN_QUANT_TILED_BFP8 || type == PN_QUANT_TILED_BFP4 ||
           type == PN_QUANT_TILED_BFP6 || type == PN_QUANT_TILED_ROWS_FP32 || type == PN_QUANT_TILED_ROWS_BFP8;
}

static int tiled_is_rows_layout(pn_gguf_quant type)
{
    return type == PN_QUANT_TILED_ROWS_FP32 || type == PN_QUANT_TILED_ROWS_BFP8;
}

/** The block-float format of a tiled type; -1 for fp32 tiles and non-tiled types. */
static int tiled_bfp_format(pn_gguf_quant type, ft_bfp_format *out)
{
    switch (type) {
    case PN_QUANT_TILED_BFP8:
    case PN_QUANT_TILED_ROWS_BFP8:
        *out = FT_BFP_BFP8_B;
        return 0;
    case PN_QUANT_TILED_BFP4:
        *out = FT_BFP_BFP4_B;
        return 0;
    case PN_QUANT_TILED_BFP6:
        *out = FT_BFP_BFP6_B;
        return 0;
    default:
        return -1;
    }
}

size_t pn_tiled_tile_bytes(pn_gguf_quant type)
{
    ft_bfp_format format;

    if (type == PN_QUANT_TILED_FP32 || type == PN_QUANT_TILED_ROWS_FP32)
        return TILED_FP32_TILE_BYTES;
    return tiled_bfp_format(type, &format) == 0 ? ft_bfp_tile_bytes(format) : 0u;
}

pn_gguf_quant pn_tiled_type_for(uint32_t fttw_format, uint32_t fttw_layout)
{
    int rows = fttw_layout == TILED_FTTW_LAYOUT_SOURCE;

    if (!rows && fttw_layout != TILED_FTTW_LAYOUT_NMAJOR)
        return (pn_gguf_quant)0;
    switch (fttw_format) {
    case TILED_FTTW_FORMAT_FP32:
        return rows ? PN_QUANT_TILED_ROWS_FP32 : PN_QUANT_TILED_FP32;
    case TILED_FTTW_FORMAT_BFP8_B:
        return rows ? PN_QUANT_TILED_ROWS_BFP8 : PN_QUANT_TILED_BFP8;
    case TILED_FTTW_FORMAT_BFP4_B:
        return rows ? (pn_gguf_quant)0 : PN_QUANT_TILED_BFP4;
    case TILED_FTTW_FORMAT_BFP6_B:
        return rows ? (pn_gguf_quant)0 : PN_QUANT_TILED_BFP6;
    default:
        return (pn_gguf_quant)0;
    }
}

/** One fp32 tile, face-ordered on disk (faces (0,0) (0,1) (1,0) (1,1), 16x16 each), to row-major. */
static void tiled_decode_fp32_tile(const unsigned char *tile, float *out)
{
    for (int r = 0; r < PN_TILED_DIM; r++)
        for (int c = 0; c < PN_TILED_DIM; c++) {
            size_t face = (size_t)(r / TILED_FACE) * 2u + (size_t)(c / TILED_FACE);

            memcpy(&out[r * PN_TILED_DIM + c],
                   tile + (face * TILED_FACE * TILED_FACE + (size_t)(r % TILED_FACE) * TILED_FACE +
                           (size_t)(c % TILED_FACE)) * sizeof(float),
                   sizeof(float));
        }
}

/**
 * Decode `count` consecutive stored tiles to row-major [32 x 32] fp32 each, in their STORED
 * orientation. Block-float tiles are swizzled (row-major) on disk and go through foundry's codec
 * in ONE call (its datum table is built once a call); fp32 tiles are face-ordered, as foundry's
 * converter writes them.
 */
static int tiled_decode_tiles(pn_gguf_quant type, const unsigned char *tiles, size_t count, float *out)
{
    ft_bfp_format format;

    if (tiled_bfp_format(type, &format) == 0)
        return ft_bfp_unpack_tiles(format, tiles, count, out) == FT_BFP_OK ? 0 : -1;
    for (size_t t = 0; t < count; t++)
        tiled_decode_fp32_tile(tiles + t * TILED_FP32_TILE_BYTES, out + t * PN_TILED_DIM * PN_TILED_DIM);
    return 0;
}

/** The red arm: is it on, decided once. */
static int tiled_red(void)
{
    static int red = -1;

    if (red < 0)
        red = getenv("PN_TILED_RED") && atoi(getenv("PN_TILED_RED")) == 1;
    return red;
}

static void tiled_apply_red(float *tile)
{
    for (int i = 0; i < PN_TILED_DIM * PN_TILED_DIM; i++)
        tile[i] *= 2.0f;
}

#ifdef TILED_FAST_NEON
/**
 * 16 block-float datums (sign bit 7, magnitude bits 6..0) with one fp32-biased block exponent
 * `e` -> 16 fp32 at `out`. ft_bfp_convert_datum's expression, lane-wise: magnitude 0 gives
 * sign << 31; else lz = leading zeros of the 7-bit magnitude and the bits are
 * sign << 31 | ((mag << lz) & 0x3f) << 17 | (uint32)(e - lz) << 23 (same uint32 wrap).
 */
static inline void tiled_fast_block(uint8x16_t d, uint32_t e, float *out)
{
    const uint8x16_t mag = vandq_u8(d, vdupq_n_u8(0x7f));
    const uint8x16_t lz = vsubq_u8(vclzq_u8(mag), vdupq_n_u8(1));
    const uint8x16_t frac = vandq_u8(vshlq_u8(mag, vreinterpretq_s8_u8(lz)), vdupq_n_u8(0x3f));
    const uint8x16_t zero = vceqq_u8(mag, vdupq_n_u8(0));
    const uint8x16_t sign = vshrq_n_u8(d, 7);
    const uint32x4_t ev = vdupq_n_u32(e);
    const uint16x8_t lz16[2] = { vmovl_u8(vget_low_u8(lz)), vmovl_u8(vget_high_u8(lz)) };
    const uint16x8_t fr16[2] = { vmovl_u8(vget_low_u8(frac)), vmovl_u8(vget_high_u8(frac)) };
    const uint16x8_t sg16[2] = { vmovl_u8(vget_low_u8(sign)), vmovl_u8(vget_high_u8(sign)) };
    const uint16x8_t z16[2] = { vmovl_u8(vget_low_u8(zero)), vmovl_u8(vget_high_u8(zero)) };

    for (int h = 0; h < 2; h++)
        for (int q = 0; q < 2; q++) {
            uint32x4_t l = q ? vmovl_high_u16(lz16[h]) : vmovl_u16(vget_low_u16(lz16[h]));
            uint32x4_t f = q ? vmovl_high_u16(fr16[h]) : vmovl_u16(vget_low_u16(fr16[h]));
            uint32x4_t s = q ? vmovl_high_u16(sg16[h]) : vmovl_u16(vget_low_u16(sg16[h]));
            uint32x4_t z = q ? vmovl_high_u16(z16[h]) : vmovl_u16(vget_low_u16(z16[h]));
            uint32x4_t sbits = vshlq_n_u32(s, 31);
            uint32x4_t full = vorrq_u32(vorrq_u32(sbits, vshlq_n_u32(f, 17)), vshlq_n_u32(vsubq_u32(ev, l), 23));
            /* z lanes are 0x00ff after widening a 0xff byte: any nonzero = magnitude 0. */
            uint32x4_t bits = vbslq_u32(vtstq_u32(z, z), sbits, full);

            vst1q_f32(out + 8 * h + 4 * q, vreinterpretq_f32_u32(bits));
        }
}

/**
 * One bfp8_b (width 8) or bfp4_b (width 4) tile -> row-major [32 x 32] fp32, bit-identical to
 * ft_bfp_unpack_tiles: 64 blocks of 16 datums, block b at tile row (b/32)*16 + b%16, columns
 * ((b/16)%2)*16..+15; bfp4 datums are nibbles, low first, read as datum = nibble << 4.
 */
static void tiled_fast_tile(int width, const unsigned char *packed, float *tile)
{
    const unsigned char *datums = packed + FT_BFP_TILE_EXPONENT_BYTES;

    for (int b = 0; b < 64; b++) {
        const int face = b / TILED_FACE;
        float *target = tile + ((face / 2) * TILED_FACE + b % TILED_FACE) * PN_TILED_DIM + (face % 2) * TILED_FACE;
        uint8x16_t d;

        if (width == 8) {
            d = vld1q_u8(datums + 16 * b);
        } else {
            uint8x8_t n = vld1_u8(datums + 8 * b);
            uint8x8_t lo = vshl_n_u8(n, 4);
            uint8x8_t hi = vand_u8(n, vdup_n_u8(0xf0));
            d = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
        }
        tiled_fast_block(d, packed[b], target);
    }
}

/** tile[k*32 + n] -> panel[n*cols + k] for a 32x32 tile, by 4x4 blocks. */
static void tiled_fast_transpose(const float *tile, float *panel, size_t cols)
{
    for (int k0 = 0; k0 < PN_TILED_DIM; k0 += 4)
        for (int n0 = 0; n0 < PN_TILED_DIM; n0 += 4) {
            float32x4_t r0 = vld1q_f32(tile + (k0 + 0) * PN_TILED_DIM + n0);
            float32x4_t r1 = vld1q_f32(tile + (k0 + 1) * PN_TILED_DIM + n0);
            float32x4_t r2 = vld1q_f32(tile + (k0 + 2) * PN_TILED_DIM + n0);
            float32x4_t r3 = vld1q_f32(tile + (k0 + 3) * PN_TILED_DIM + n0);
            float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1);
            float32x4_t t2 = vtrn1q_f32(r2, r3), t3 = vtrn2q_f32(r2, r3);
            float32x4_t c0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
            float32x4_t c1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
            float32x4_t c2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
            float32x4_t c3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));

            vst1q_f32(panel + (size_t)(n0 + 0) * cols + k0, c0);
            vst1q_f32(panel + (size_t)(n0 + 1) * cols + k0, c1);
            vst1q_f32(panel + (size_t)(n0 + 2) * cols + k0, c2);
            vst1q_f32(panel + (size_t)(n0 + 3) * cols + k0, c3);
        }
}

/**
 * The fused matvec of one tile column: 32 rows x cols dotted with `vec`, never materialising
 * the panel. Each row keeps pn_dot_f32's NEON state (acc0 for i % 8 < 4, acc1 for the rest,
 * one fma per 4 lanes in increasing i) across tiles, then acc0 + acc1 and the across-vector
 * add, so out[n] is bit-identical to pn_dot_f32(panel row n, vec, cols) for cols % 32 == 0.
 */
static void tiled_fast_dot_column(int width, const unsigned char *column, int tiles, const float *vec, float *out)
{
    const size_t tile_bytes = FT_BFP_TILE_EXPONENT_BYTES + (size_t)PN_TILED_DIM * PN_TILED_DIM * (size_t)width / 8u;
    float tile[PN_TILED_DIM * PN_TILED_DIM], rows[PN_TILED_DIM * PN_TILED_DIM];
    float32x4_t acc[PN_TILED_DIM][2];

    for (int n = 0; n < PN_TILED_DIM; n++)
        acc[n][0] = acc[n][1] = vdupq_n_f32(0.0f);
    for (int kt = 0; kt < tiles; kt++) {
        const float *x = vec + (size_t)kt * PN_TILED_DIM;
        float32x4_t xv[8];

        tiled_fast_tile(width, column + (size_t)kt * tile_bytes, tile);
        if (kt == 0 && tiled_red())
            tiled_apply_red(tile);
        tiled_fast_transpose(tile, rows, PN_TILED_DIM);
        for (int j = 0; j < 8; j++)
            xv[j] = vld1q_f32(x + 4 * j);
        for (int n = 0; n < PN_TILED_DIM; n++) {
            const float *r = rows + n * PN_TILED_DIM;
            float32x4_t a0 = acc[n][0], a1 = acc[n][1];

            for (int j = 0; j < 8; j += 2) {
                a0 = vfmaq_f32(a0, vld1q_f32(r + 4 * j), xv[j]);
                a1 = vfmaq_f32(a1, vld1q_f32(r + 4 * j + 4), xv[j + 1]);
            }
            acc[n][0] = a0;
            acc[n][1] = a1;
        }
    }
    for (int n = 0; n < PN_TILED_DIM; n++)
        out[n] = vaddvq_f32(vaddq_f32(acc[n][0], acc[n][1]));
}

/** A bfp8_b / bfp4_b tile column straight into the panel (tile by tile, through L1). */
static void tiled_fast_column(int width, const unsigned char *column, int tiles, int cols, float *panel)
{
    const size_t tile_bytes = FT_BFP_TILE_EXPONENT_BYTES + (size_t)PN_TILED_DIM * PN_TILED_DIM * (size_t)width / 8u;
    float tile[PN_TILED_DIM * PN_TILED_DIM];

    for (int kt = 0; kt < tiles; kt++) {
        tiled_fast_tile(width, column + (size_t)kt * tile_bytes, tile);
        if (kt == 0 && tiled_red())
            tiled_apply_red(tile);
        tiled_fast_transpose(tile, panel + (size_t)kt * PN_TILED_DIM, (size_t)cols);
    }
}
#endif

int pn_tiled_decode_column(pn_gguf_quant type, const void *column_base, int cols, float *panel,
                           float *scratch)
{
    const size_t tile_values = PN_TILED_DIM * PN_TILED_DIM;
    const int tiles = cols / PN_TILED_DIM;

#ifdef TILED_FAST_NEON
    if ((type == PN_QUANT_TILED_BFP8 || type == PN_QUANT_TILED_BFP4) && cols % PN_TILED_DIM == 0) {
        tiled_fast_column(type == PN_QUANT_TILED_BFP8 ? 8 : 4, column_base, tiles, cols, panel);
        return 0;
    }
#endif
    if (!pn_tiled_tile_bytes(type) || tiled_is_rows_layout(type) || cols % PN_TILED_DIM != 0 ||
        tiled_decode_tiles(type, column_base, (size_t)tiles, scratch) != 0)
        return -1;
    if (tiled_red())
        tiled_apply_red(scratch);
    for (int kt = 0; kt < tiles; kt++) {
        const float *tile = scratch + (size_t)kt * tile_values;

        /* tile[k*32 + n] = W[n][32kt + k]: transpose into the panel's rows. */
        for (int n = 0; n < PN_TILED_DIM; n++)
            for (int k = 0; k < PN_TILED_DIM; k++)
                panel[(size_t)n * (size_t)cols + (size_t)kt * PN_TILED_DIM + (size_t)k] = tile[k * PN_TILED_DIM + n];
    }
    return 0;
}

int pn_tiled_dot_column(pn_gguf_quant type, const void *column_base, int cols, const float *vec, float *out)
{
#ifdef TILED_FAST_NEON
    if ((type == PN_QUANT_TILED_BFP8 || type == PN_QUANT_TILED_BFP4) && cols > 0 && cols % PN_TILED_DIM == 0) {
        tiled_fast_dot_column(type == PN_QUANT_TILED_BFP8 ? 8 : 4, column_base, cols / PN_TILED_DIM, vec, out);
        return 0;
    }
#else
    (void)type;
    (void)column_base;
    (void)cols;
    (void)vec;
    (void)out;
#endif
    return -1;
}

/**
 * One bfp8_b datum (sign bit 7, magnitude bits 6..0) under block exponent `e` -> fp32 bits:
 * ft_bfp_convert_datum's expression, as tiled_fast_block evaluates it lane-wise. Magnitude 0 gives
 * sign << 31; else lz = leading zeros of the 7-bit magnitude and the bits are
 * sign << 31 | ((mag << lz) & 0x3f) << 17 | (uint32)(e - lz) << 23 (same uint32 wrap).
 */
static inline uint32_t tiled_bfp8_datum(uint8_t datum, uint32_t e)
{
    const uint32_t sign = (uint32_t)datum >> 7, mag = (uint32_t)datum & 0x7fu;
    uint32_t lz;

    if (mag == 0)
        return sign << 31;
    lz = (uint32_t)__builtin_clz(mag) - 25u;
    return (sign << 31) | (((mag << lz) & 0x3fu) << 17) | ((e - lz) << 23);
}

/**
 * Row `r` of one stored ROWS-layout tile -> 32 fp32, the same bits as decoding the whole tile and
 * keeping row r. Bfp8_b: the row is blocks (r/16)*32 + r%16 (columns 0..15) and that + 16
 * (columns 16..31), each datum by tiled_bfp8_datum. Fp32: the row's 16 values in faces (r/16)*2
 * and (r/16)*2 + 1.
 */
static void tiled_decode_tile_row(pn_gguf_quant type, const unsigned char *tile, size_t r, float *out)
{
    for (size_t h = 0; h < 2; h++) {
        const size_t face = (r / TILED_FACE) * 2u + h;

        if (type == PN_QUANT_TILED_ROWS_BFP8) {
            const size_t block = face * TILED_FACE + r % TILED_FACE;
            const unsigned char *datums = tile + FT_BFP_TILE_EXPONENT_BYTES + block * TILED_FACE;

            for (size_t i = 0; i < TILED_FACE; i++) {
                const uint32_t bits = tiled_bfp8_datum(datums[i], tile[block]);

                memcpy(&out[h * TILED_FACE + i], &bits, sizeof(bits));
            }
        } else {
            memcpy(out + h * TILED_FACE,
                   tile + (face * TILED_FACE * TILED_FACE + (r % TILED_FACE) * TILED_FACE) * sizeof(float),
                   TILED_FACE * sizeof(float));
        }
    }
}

int pn_tiled_decode_row(pn_gguf_quant type, const void *data, size_t row, size_t cols, float *dst)
{
    float scratch[PN_TILED_DIM * PN_TILED_DIM];
    const size_t grid = cols / PN_TILED_DIM;
    const size_t tile_bytes = pn_tiled_tile_bytes(type);
    const size_t r = row % PN_TILED_DIM;
    const int transposed = !tiled_is_rows_layout(type);
    const unsigned char *base;

    if (!tile_bytes || cols % PN_TILED_DIM != 0)
        return -1;
    /* Both layouts keep a 32-row group's tiles together, `grid` of them. */
    base = (const unsigned char *)data + (row / PN_TILED_DIM) * grid * tile_bytes;
    /* The embedding: decode only the token's row of each tile, not all 32 (one embed row a token
     * in prefill decoded 32x the values it kept). The red arm doubles the first tile's part. */
    if (!transposed) {
        for (size_t t = 0; t < grid; t++)
            tiled_decode_tile_row(type, base + t * tile_bytes, r, dst + t * PN_TILED_DIM);
        if (tiled_red())
            for (size_t c = 0; c < PN_TILED_DIM; c++)
                dst[c] *= 2.0f;
        return 0;
    }
    for (size_t t = 0; t < grid; t++) {
        if (tiled_decode_tiles(type, base + t * tile_bytes, 1u, scratch) != 0)
            return -1;
        if (t == 0 && tiled_red())
            tiled_apply_red(scratch);
        for (size_t c = 0; c < PN_TILED_DIM; c++)
            dst[t * PN_TILED_DIM + c] = scratch[c * PN_TILED_DIM + r];
    }
    return 0;
}
