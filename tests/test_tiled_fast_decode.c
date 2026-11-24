/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_tiled_fast_decode.c
 * @brief  pn_tiled_decode_column (the CPU oracle's tile-column decode, NEON fast path on arm64)
 *         vs the reference: foundry ft_bfp_unpack_tiles + the scalar transpose, bit for bit; and
 *         pn_tiled_dot_column (fused one-row matvec) vs pn_dot_f32 over that reference panel; and
 *         pn_tiled_decode_row on the ROWS layout (the embedding, one row of each tile) vs whole
 *         tiles decoded by the codec with the row kept.
 *
 * Coverage: bfp8_b and bfp4_b; every datum value under every block exponent 0..255 (incl. the
 * small exponents where e - lz wraps); random columns at the real widths (5120, 17408) for the
 * transpose stride; PN_TILED_RED=1 (first tile x2) on both paths. The harness is first shown
 * able to fail (a one-bit-flipped reference must be reported). Prints the decode speedup.
 * Row decode: rows_bfp8 every datum x exponent and every row 0..31; random 5120-wide row groups;
 * rows_fp32 against the matrix the face-ordered tiles were written from; a neighbouring row's
 * reference must differ. Prints the row decode speedup.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/accel/simd.h"
#include "../src/formats/gguf_tiled.h"
#include "tt/convert/ft_bfp.h"

static int g_checks, g_failures;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok)
        g_failures++;
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
}

/* The reference: decode every tile with foundry's codec, optional red, scalar transpose. */
static int reference_column(ft_bfp_format format, const unsigned char *column, int cols, float *panel, int red)
{
    const int tiles = cols / PN_TILED_DIM;
    float *scratch = malloc((size_t)tiles * PN_TILED_DIM * PN_TILED_DIM * sizeof(float));

    if (!scratch || ft_bfp_unpack_tiles(format, column, (size_t)tiles, scratch) != FT_BFP_OK) {
        free(scratch);
        return -1;
    }
    if (red)
        for (int i = 0; i < PN_TILED_DIM * PN_TILED_DIM; i++)
            scratch[i] *= 2.0f;
    for (int kt = 0; kt < tiles; kt++)
        for (int n = 0; n < PN_TILED_DIM; n++)
            for (int k = 0; k < PN_TILED_DIM; k++)
                panel[(size_t)n * cols + kt * PN_TILED_DIM + k] =
                    scratch[(size_t)kt * PN_TILED_DIM * PN_TILED_DIM + k * PN_TILED_DIM + n];
    free(scratch);
    return 0;
}

/* The row reference: every tile of the row group through the codec, optional red, row r kept. */
static int reference_row(const unsigned char *group, int cols, int r, float *dst, int red)
{
    const int tiles = cols / PN_TILED_DIM;
    float *scratch = malloc((size_t)tiles * PN_TILED_DIM * PN_TILED_DIM * sizeof(float));

    if (!scratch || ft_bfp_unpack_tiles(FT_BFP_BFP8_B, group, (size_t)tiles, scratch) != FT_BFP_OK) {
        free(scratch);
        return -1;
    }
    if (red)
        for (int i = 0; i < PN_TILED_DIM * PN_TILED_DIM; i++)
            scratch[i] *= 2.0f;
    for (int t = 0; t < tiles; t++)
        memcpy(dst + t * PN_TILED_DIM, scratch + ((size_t)t * PN_TILED_DIM + r) * PN_TILED_DIM,
               PN_TILED_DIM * sizeof(float));
    free(scratch);
    return 0;
}

/* Every row of one row group both ways; returns the number of rows that differ (-1 on error). */
static int compare_rows(const unsigned char *group, int cols, int red, float *a, float *b)
{
    int bad = 0;

    for (int r = 0; r < PN_TILED_DIM; r++) {
        if (pn_tiled_decode_row(PN_QUANT_TILED_ROWS_BFP8, group, (size_t)r, (size_t)cols, a) != 0 ||
            reference_row(group, cols, r, b, red) != 0)
            return -1;
        bad += memcmp(a, b, (size_t)cols * sizeof(float)) != 0;
    }
    return bad;
}

static uint32_t rng_state = 0x2545f491u;

static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Fill `tiles` packed tiles: mode 0 = random bytes; mode 1 = exponent `e` everywhere and the
 * datum stream counting 0,1,2,... so every datum value appears in every tile. */
static void fill(unsigned char *buf, size_t tile_bytes, int tiles, int mode, int e)
{
    for (int t = 0; t < tiles; t++) {
        unsigned char *tile = buf + (size_t)t * tile_bytes;

        for (size_t i = 0; i < tile_bytes; i++)
            tile[i] = mode == 0 ? (unsigned char)rng() : (unsigned char)(i + (size_t)t * 37u);
        if (mode == 1)
            memset(tile, e, FT_BFP_TILE_EXPONENT_BYTES);
    }
}

/* Decode one column both ways; 0 = bit-identical. */
static int compare(pn_gguf_quant type, ft_bfp_format format, const unsigned char *column, int cols, int red,
                   float *a, float *b)
{
    if (pn_tiled_decode_column(type, column, cols, a, b + (size_t)PN_TILED_DIM * cols) != 0 ||
        reference_column(format, column, cols, b, red) != 0)
        return -1;
    return memcmp(a, b, (size_t)PN_TILED_DIM * cols * sizeof(float)) == 0 ? 0 : 1;
}

/* Fused dots vs pn_dot_f32 over the reference panel; 0 = bit-identical, 2 = no fused path. */
static int compare_dot(pn_gguf_quant type, ft_bfp_format format, const unsigned char *column, int cols, int red,
                       const float *vec, float *panel)
{
    float got[PN_TILED_DIM], want[PN_TILED_DIM];

    if (pn_tiled_dot_column(type, column, cols, vec, got) != 0)
        return 2;
    if (reference_column(format, column, cols, panel, red) != 0)
        return -1;
    for (int n = 0; n < PN_TILED_DIM; n++)
        want[n] = pn_dot_f32(panel + (size_t)n * cols, vec, cols);
    return memcmp(got, want, sizeof(got)) == 0 ? 0 : 1;
}

static double now_s(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const int red = getenv("PN_TILED_RED") && atoi(getenv("PN_TILED_RED")) == 1;
    const struct {
        pn_gguf_quant type;
        ft_bfp_format format;
        const char *name;
    } fmts[2] = { { PN_QUANT_TILED_BFP8, FT_BFP_BFP8_B, "bfp8_b" }, { PN_QUANT_TILED_BFP4, FT_BFP_BFP4_B, "bfp4_b" } };
    const int max_cols = 17408;
    unsigned char *buf = malloc((size_t)(max_cols / PN_TILED_DIM) * 1088u);
    float *a = malloc((size_t)PN_TILED_DIM * max_cols * sizeof(float));
    float *b = malloc((size_t)2 * PN_TILED_DIM * max_cols * sizeof(float));
    char what[160];

    (void)argc;
    (void)argv;
    if (!buf || !a || !b)
        return 2;
    printf("test_tiled_fast_decode%s\n", red ? " (PN_TILED_RED=1)" : "");

    /* The harness can fail: a reference with one flipped bit must not compare equal. */
    {
        size_t tb = ft_bfp_tile_bytes(FT_BFP_BFP8_B);

        fill(buf, tb, 2, 0, 0);
        pn_tiled_decode_column(PN_QUANT_TILED_BFP8, buf, 64, a, b + PN_TILED_DIM * 64);
        reference_column(FT_BFP_BFP8_B, buf, 64, b, red);
        b[PN_TILED_DIM * 64 - 7] = -b[PN_TILED_DIM * 64 - 7] + 1.0f;
        check(memcmp(a, b, (size_t)PN_TILED_DIM * 64 * sizeof(float)) != 0, "harness: a one-value change is caught");
    }

    /* Row decode on the ROWS layout (the embedding). */
    {
        const size_t tb = ft_bfp_tile_bytes(FT_BFP_BFP8_B);
        const int cols = 5120, groups = 3;
        int bad = 0, err = 0, rc;

        for (int e = 0; e < 256; e++) {
            fill(buf, tb, 2, 1, e);
            rc = compare_rows(buf, 64, red, a, b);
            err += rc < 0;
            bad += rc > 0;
        }
        snprintf(what, sizeof(what), "rows_bfp8: row decode, every datum x exponent 0..255 x row 0..31: %d of 256 "
                 "groups differ, %d errors", bad, err);
        check(bad == 0 && err == 0, what);

        bad = err = 0;
        for (int rep = 0; rep < 4; rep++) {
            fill(buf, tb, cols / PN_TILED_DIM, 0, 0);
            rc = compare_rows(buf, cols, red, a, b);
            err += rc < 0;
            bad += rc > 0;
        }
        snprintf(what, sizeof(what), "rows_bfp8: random 5120-wide row groups: %d of 4 differ, %d errors", bad, err);
        check(bad == 0 && err == 0, what);

        /* A whole tensor of `groups` row groups: a row past the first group reads its own group. */
        fill(buf, tb, groups * 2, 0, 0);
        pn_tiled_decode_row(PN_QUANT_TILED_ROWS_BFP8, buf, 2 * PN_TILED_DIM + 5, 64, a);
        reference_row(buf + 2 * 2 * tb, 64, 5, b, red);
        check(memcmp(a, b, 64 * sizeof(float)) == 0, "rows_bfp8: row 69 of a 3-group tensor is group 2's row 5");

        /* The comparison can fail: row 6's reference is not row 7's decode. */
        fill(buf, tb, 2, 0, 0);
        pn_tiled_decode_row(PN_QUANT_TILED_ROWS_BFP8, buf, 7, 64, a);
        reference_row(buf, 64, 6, b, red);
        check(memcmp(a, b, 64 * sizeof(float)) != 0, "rows_bfp8 harness: a neighbouring row's reference is caught");

        /* rows_fp32: face-ordered tiles written from a known matrix; every row must read it back. */
        {
            float m[PN_TILED_DIM][64];
            unsigned char fp[2 * 4096];

            for (int r = 0; r < PN_TILED_DIM; r++)
                for (int c = 0; c < 64; c++)
                    m[r][c] = ((float)(rng() % 20001u) - 10000.0f) * 1e-4f;
            for (int t = 0; t < 2; t++)
                for (int r = 0; r < PN_TILED_DIM; r++)
                    for (int c = 0; c < PN_TILED_DIM; c++) {
                        size_t face = (size_t)(r / 16) * 2u + (size_t)(c / 16);

                        memcpy(fp + t * 4096 + (face * 256 + (size_t)(r % 16) * 16 + (size_t)(c % 16)) * 4,
                               &m[r][t * PN_TILED_DIM + c], 4);
                    }
            bad = 0;
            for (int r = 0; r < PN_TILED_DIM; r++) {
                pn_tiled_decode_row(PN_QUANT_TILED_ROWS_FP32, fp, (size_t)r, 64, a);
                if (red)
                    for (int c = 0; c < PN_TILED_DIM; c++)
                        a[c] *= 0.5f;
                bad += memcmp(a, m[r], sizeof(m[r])) != 0;
            }
            snprintf(what, sizeof(what), "rows_fp32: every row reads back the written matrix: %d of 32 differ", bad);
            check(bad == 0, what);
        }

        /* Speed (informational): one embedding row at 5120. */
        {
            const int reps = 2000;
            double t0, t_new, t_ref;

            fill(buf, tb, cols / PN_TILED_DIM, 0, 0);
            t0 = now_s();
            for (int i = 0; i < reps; i++)
                pn_tiled_decode_row(PN_QUANT_TILED_ROWS_BFP8, buf, (size_t)(i % PN_TILED_DIM), (size_t)cols, a);
            t_new = now_s() - t0;
            t0 = now_s();
            for (int i = 0; i < reps; i++)
                reference_row(buf, cols, i % PN_TILED_DIM, b, 0);
            t_ref = now_s() - t0;
            printf("  info  rows_bfp8 row decode at 5120: %.1f us/row (whole tiles %.1f): %.1fx\n",
                   t_new * 1e6 / reps, t_ref * 1e6 / reps, t_ref / t_new);
        }
    }

    for (int f = 0; f < 2; f++) {
        size_t tb = ft_bfp_tile_bytes(fmts[f].format);
        int bad = 0;

        /* Every datum under every exponent: 2-tile columns (cols 64). */
        for (int e = 0; e < 256; e++) {
            fill(buf, tb, 2, 1, e);
            bad += compare(fmts[f].type, fmts[f].format, buf, 64, red, a, b) != 0;
        }
        snprintf(what, sizeof(what), "%s: every datum x exponent 0..255: %d of 256 columns differ", fmts[f].name, bad);
        check(bad == 0, what);

        bad = 0;
        for (int rep = 0; rep < 8; rep++) {
            int cols = rep % 2 ? 17408 : 5120;

            fill(buf, tb, cols / PN_TILED_DIM, 0, 0);
            bad += compare(fmts[f].type, fmts[f].format, buf, cols, red, a, b) != 0;
        }
        snprintf(what, sizeof(what), "%s: random columns at 5120 / 17408: %d of 8 differ", fmts[f].name, bad);
        check(bad == 0, what);

        /* Fused dots: random activations incl. large/small magnitudes, every exponent column too. */
        {
            float *vec = malloc((size_t)max_cols * sizeof(float));
            int none = 0;

            bad = 0;
            for (int i = 0; i < max_cols; i++)
                vec[i] = ((float)(rng() % 20001u) - 10000.0f) * (i % 7 == 0 ? 1e-3f : 1e-5f);
            for (int e = 100; e < 140; e++) {
                int rc;

                fill(buf, tb, 2, 1, e);
                rc = compare_dot(fmts[f].type, fmts[f].format, buf, 64, red, vec, b);
                bad += rc == 1 || rc < 0;
                none += rc == 2;
            }
            for (int rep = 0; rep < 8; rep++) {
                int cols = rep % 2 ? 17408 : 5120, rc;

                fill(buf, tb, cols / PN_TILED_DIM, 0, 0);
                for (size_t t = 0; t < (size_t)cols / PN_TILED_DIM; t++)  /* weight-like exponents */
                    for (int x = 0; x < FT_BFP_TILE_EXPONENT_BYTES; x++)
                        buf[t * tb + x] = (unsigned char)(112 + rng() % 16u);
                rc = compare_dot(fmts[f].type, fmts[f].format, buf, cols, red, vec, b);
                bad += rc == 1 || rc < 0;
                none += rc == 2;
            }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            snprintf(what, sizeof(what), "%s: fused dots == pn_dot_f32(reference row): %d of 48 differ, %d unfused",
                     fmts[f].name, bad, none);
            check(bad == 0 && none == 0, what);
#else
            snprintf(what, sizeof(what), "%s: no fused path on this host (all 48 refused)", fmts[f].name);
            check(none == 48, what);
#endif
            free(vec);
        }

        /* Speed (informational): 17408-wide column, both paths. */
        {
            const int reps = 200, cols = 17408;
            double t0, t_new, t_ref;

            fill(buf, tb, cols / PN_TILED_DIM, 0, 0);
            t0 = now_s();
            for (int r = 0; r < reps; r++)
                pn_tiled_decode_column(fmts[f].type, buf, cols, a, b);
            t_new = now_s() - t0;
            t0 = now_s();
            for (int r = 0; r < reps; r++)
                reference_column(fmts[f].format, buf, cols, b, 0);
            t_ref = now_s() - t0;
            printf("  info  %s decode %.2f ns/value (reference %.2f): %.1fx\n", fmts[f].name,
                   t_new * 1e9 / ((double)reps * cols * PN_TILED_DIM), t_ref * 1e9 / ((double)reps * cols * PN_TILED_DIM),
                   t_ref / t_new);
        }
    }
    printf("\n%d checks, %d failed\n", g_checks, g_failures);
    free(buf);
    free(a);
    free(b);
    return g_failures ? 1 : 0;
}
