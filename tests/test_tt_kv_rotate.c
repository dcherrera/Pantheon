/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_kv_rotate.c
 * @brief K4H's Hadamard rotation (pn_tt_kv_rotate.h), off the card.
 *
 *   1. H: symmetric, H*H = I, entries +-1/16; the host transform equals the explicit matrix
 *      product and is its own inverse.
 *   2. The weight: in bf16, Bfp8_b and Bfp4_b tiles H decodes back EXACTLY, tile (k, n) at
 *      n * Kt + k of the N-major grid.
 *   3. The card op, emulated: a q_rot / k_rot strip (Qwen3.8: 24 / 4 heads x 8 tiles, one band
 *      of 32 positions, face-ordered bf16 tiles) read through pn_tt_kv_rotate_view() with the
 *      resident matmul's addressing (A tile (m, k) at m * Kt + k, C tile (m, n) at m * Nt + n,
 *      B N-major) equals every row of every head times H. Red: the view of the transposed
 *      shape (heads tiles a row) does not.
 *   4. Attention invariance: (qH).(kH) = q.k within bf16 rounding on rows with outlier
 *      channels; and the point of it: the Bfp4 cache's score error drops when K is rotated.
 *   5. State files: export (unpack, un-rotate) then import (rotate, pack) gives back the
 *      cache's bytes, so a k4h session round-trips through the canonical frame.
 *   6. The knob (`k4h`, `PN_TT_KV_LONG`) and the catalog field
 *      options.kv_long.
 *
 * `PN_TT_RED=kv_rotate_h_sign` (one sign of H flipped) must fail 1 and 3.
 *
 * Build: `make USE_TT=1 tests/test_tt_kv_rotate` (Mac or Linux, no card).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_kv_format.h"
#include "../src/accel/tt/pn_tt_kv_rotate.h"
#include "../src/local/catalog_manifest.h"
#include "../src/local/local_file_io.h"
#include "tt/convert/ft_bfp.h"
#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_tilize.h"

#define DIM 256u
#define TILES (DIM / 32u)
#define Q_HEADS 24u
#define KV_HEADS 4u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/* A fixed generator, so a failure reproduces. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static double uniform(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / (double)(1ull << 53);
}

static double gaussian(void)
{
    const double u = uniform() + 1e-12, v = uniform();

    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static float bf16_round(float x)
{
    return ft_float_from_bf16(ft_bf16_from_float(x));
}

/** A post-norm K or Q row: Gaussian with a few fixed outlier channels (KIVI's K shape). */
static void fill_row(float *row, double outlier)
{
    for (uint32_t i = 0; i < DIM; i++)
        row[i] = (float)gaussian();
    row[7] *= (float)outlier;
    row[40] *= (float)outlier;
    row[131] *= (float)(outlier * 0.7);
}

/* ---- 1. H ---- */
static void check_h(void)
{
    double worst_identity = 0.0, worst_symmetry = 0.0;
    bool entries = true, transform = true, involution = true;
    float row[DIM], copy[DIM], twice[DIM];

    for (uint32_t i = 0; i < DIM; i++) {
        for (uint32_t j = 0; j < DIM; j++) {
            double dot = 0.0;
            const float e = pn_tt_kv_rotate_entry(DIM, i, j);

            for (uint32_t k = 0; k < DIM; k++)
                dot += (double)pn_tt_kv_rotate_entry(DIM, i, k) *
                       (double)pn_tt_kv_rotate_entry(DIM, k, j);
            worst_identity = fmax(worst_identity, fabs(dot - (i == j ? 1.0 : 0.0)));
            worst_symmetry = fmax(worst_symmetry, fabs(e - pn_tt_kv_rotate_entry(DIM, j, i)));
            entries = entries && fabsf(fabsf(e) - 0.0625f) == 0.0f && bf16_round(e) == e;
        }
    }
    check(worst_identity < 1e-12, "H*H = I (H orthogonal and its own inverse)");
    check(worst_symmetry == 0.0, "H is symmetric");
    check(entries, "every entry of H is +-1/16, exact in bf16");
    check(pn_tt_kv_rotate_entry(255u, 0u, 0u) == 0.0f && pn_tt_kv_rotate_rows(row, 1u, 96u) != 0,
          "a width that is not a power of two is refused");

    for (int trial = 0; trial < 16; trial++) {
        fill_row(row, 20.0);
        memcpy(copy, row, sizeof(row));
        check(pn_tt_kv_rotate_rows(copy, 1u, DIM) == 0, "the host transform runs");
        for (uint32_t j = 0; j < DIM; j++) {
            double sum = 0.0;

            for (uint32_t k = 0; k < DIM; k++)
                sum += (double)row[k] * (double)pn_tt_kv_rotate_entry(DIM, k, j);
            transform = transform && fabs(sum - copy[j]) <= 1e-5 * (1.0 + fabs(sum));
        }
        memcpy(twice, copy, sizeof(copy));
        pn_tt_kv_rotate_rows(twice, 1u, DIM);
        for (uint32_t j = 0; j < DIM; j++)
            involution = involution && fabsf(twice[j] - row[j]) <= 1e-5f * (1.0f + fabsf(row[j]));
    }
    check(transform, "the fast transform = row * H (explicit product)");
    check(involution, "the transform applied twice is the identity");
}

/* ---- 2. The weight tiles ---- */
static void check_weight(void)
{
    static const uint32_t formats[] = { 2048u, 1088u, 576u };

    for (size_t f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
        const uint32_t bytes = formats[f];
        uint8_t *packed = malloc((size_t)TILES * TILES * bytes);
        float tile[1024];
        bool exact = true;
        char what[160];

        check(packed && pn_tt_kv_rotate_weight_tiles(DIM, bytes, packed) == 0,
              "the weight tiles build");
        for (uint32_t n = 0; packed && n < TILES; n++) {
            for (uint32_t k = 0; k < TILES; k++) {
                const uint8_t *src = packed + ((size_t)n * TILES + k) * bytes;

                if (bytes == 2048u) {
                    const uint16_t *face = (const uint16_t *)src;

                    for (uint32_t r = 0; r < 32u; r++)
                        for (uint32_t c = 0; c < 32u; c++)
                            tile[r * 32u + c] =
                                ft_float_from_bf16(face[ft_tt_tile_face_index(r, c)]);
                } else {
                    ft_bfp_unpack_tiles(bytes == 1088u ? FT_BFP_BFP8_B : FT_BFP_BFP4_B, src, 1u,
                                        tile);
                }
                for (uint32_t r = 0; r < 32u; r++)
                    for (uint32_t c = 0; c < 32u; c++)
                        exact = exact && tile[r * 32u + c] ==
                                             pn_tt_kv_rotate_entry(DIM, k * 32u + r,
                                                                   n * 32u + c);
            }
        }
        snprintf(what, sizeof(what), "H in %u-byte tiles decodes back exactly, N-major", bytes);
        check(exact, what);
        free(packed);
    }
    check(pn_tt_kv_rotate_weight_tiles(DIM, 1024u, (uint8_t[8]){ 0 }) != 0,
          "an unknown weight tile size is refused");
}

/* ---- 3. The card op, emulated ---- */

/** C(32x32, row-major, float) += A(face-ordered bf16) * B(row-major float). */
static void tile_mac(float *c, const uint16_t *a, const float *b)
{
    for (uint32_t r = 0; r < 32u; r++)
        for (uint32_t k = 0; k < 32u; k++) {
            const float x = ft_float_from_bf16(a[ft_tt_tile_face_index(r, k)]);

            for (uint32_t col = 0; col < 32u; col++)
                c[r * 32u + col] += x * b[k * 32u + col];
        }
}

/**
 * The resident matmul over a view, by its addressing (pn_tt_device.h "the second tile-row"):
 * A tile (m, k) at `m * Kt + k` of the view's base, B tile (k, n) at `n * Kt + k` (N-major),
 * C tile (m, n) at `m * Nt + n`; the product rounded to bf16 as the packer writes it.
 */
static void emulate_matmul(const uint16_t *strip, size_t strip_tiles,
                           const pn_tt_activation *view, const uint8_t *weight_bf16,
                           uint16_t *out)
{
    const uint32_t kt = TILES, nt = TILES;
    float b[1024], c[1024];

    for (uint32_t m = 0; m < view->active_rows; m++) {
        for (uint32_t n = 0; n < nt; n++) {
            memset(c, 0, sizeof(c));
            for (uint32_t k = 0; k < kt; k++) {
                const uint16_t *bt =
                    (const uint16_t *)(weight_bf16 + ((size_t)n * kt + k) * 2048u);

                for (uint32_t r = 0; r < 32u; r++)
                    for (uint32_t col = 0; col < 32u; col++)
                        b[r * 32u + col] = ft_float_from_bf16(bt[ft_tt_tile_face_index(r, col)]);
                const size_t a = (size_t)m * view->tile_count + k;

                /* Past the strip is past the allocation: count it as garbage, not a read. */
                if (a >= strip_tiles) {
                    c[0] += 1e6f;
                    continue;
                }
                tile_mac(c, strip + a * 1024u, b);
            }
            if ((size_t)m * nt + n >= strip_tiles)
                continue;
            for (uint32_t r = 0; r < 32u; r++)
                for (uint32_t col = 0; col < 32u; col++)
                    out[((size_t)m * nt + n) * 1024u + ft_tt_tile_face_index(r, col)] =
                        ft_bf16_from_float(c[r * 32u + col]);
        }
    }
}

/** Strip value of head h, position r, channel d (tiles row-major across the strip). */
static float strip_at(const uint16_t *strip, uint32_t h, uint32_t r, uint32_t d)
{
    return ft_float_from_bf16(
        strip[((size_t)h * TILES + d / 32u) * 1024u + ft_tt_tile_face_index(r, d % 32u)]);
}

/** Does the emulated op over `view` rotate every row of every head? Returns the worst error. */
static double rotation_error(uint32_t heads, uint32_t rows, const pn_tt_activation *view)
{
    const size_t tiles = (size_t)heads * TILES;
    uint16_t *strip = calloc(tiles * 1024u, sizeof(uint16_t));
    uint16_t *out = calloc(tiles * 1024u, sizeof(uint16_t));
    uint8_t *weight = malloc((size_t)TILES * TILES * 2048u);
    float row[DIM];
    double worst = 0.0;

    if (!strip || !out || !weight || pn_tt_kv_rotate_weight_tiles(DIM, 2048u, weight) != 0) {
        free(strip), free(out), free(weight);
        return 1e30;
    }
    /* Rows past `rows` stay zero, as a strip's padding rows are (pn_tt_device.h). */
    for (uint32_t h = 0; h < heads; h++)
        for (uint32_t r = 0; r < rows; r++) {
            fill_row(row, 10.0);
            for (uint32_t d = 0; d < DIM; d++)
                strip[((size_t)h * TILES + d / 32u) * 1024u + ft_tt_tile_face_index(r, d % 32u)] =
                    ft_bf16_from_float(row[d]);
        }
    emulate_matmul(strip, tiles, view, weight, out);
    for (uint32_t h = 0; h < heads; h++)
        for (uint32_t r = 0; r < 32u; r++) {
            for (uint32_t d = 0; d < DIM; d++)
                row[d] = strip_at(strip, h, r, d);
            pn_tt_kv_rotate_rows(row, 1u, DIM);
            for (uint32_t d = 0; d < DIM; d++) {
                const double got = strip_at(out, h, r, d);

                /* One bf16 rounding of the product: half an ulp, 2^-9 relative. */
                worst = fmax(worst, fabs(got - row[d]) / (fabs(row[d]) + 1e-3));
            }
        }
    free(strip), free(out), free(weight);
    return worst;
}

static void check_card_op(void)
{
    pn_tt_activation strip, view, bad;
    char what[200];

    memset(&strip, 0, sizeof(strip));
    strip.bank = 3u;
    strip.address = 0x123000u;
    strip.tile_count = Q_HEADS * TILES;
    strip.replica_count = 2u; /* a mirrored strip: the view reads the primary only */
    strip.replica_bank[0] = 3u;
    strip.replica_address[0] = 0x123000u;
    strip.tile_rows = strip.active_rows = 1u;
    check(pn_tt_kv_rotate_view(&strip, Q_HEADS, TILES, &view) == 0 && view.bank == 3u &&
              view.address == 0x123000u && view.tile_count == TILES &&
              view.tile_rows == Q_HEADS && view.active_rows == Q_HEADS &&
              view.replica_count == 1u && view.replica_address[0] == 0x123000u,
          "the Q view: 24 tile rows of 8 at the strip's band, its primary mirror");
    check(pn_tt_kv_rotate_view(&strip, Q_HEADS + 1u, TILES, &bad) != 0,
          "a view past the strip is refused");

    for (uint32_t pass = 0; pass < 2u; pass++) {
        const uint32_t heads = pass ? KV_HEADS : Q_HEADS;
        static const uint32_t rows_case[] = { 1u, 2u, 32u };

        strip.tile_count = heads * TILES;
        pn_tt_kv_rotate_view(&strip, heads, TILES, &view);
        for (size_t c = 0; c < 3u; c++) {
            const double err = rotation_error(heads, rows_case[c], &view);

            snprintf(what, sizeof(what), "%u heads, %u rows: the matmul over the view rotates "
                     "every row of every head (worst rel err %.2e <= 2^-8)", heads,
                     rows_case[c], err);
            check(err <= 1.0 / 256.0, what);
        }
    }
    /* RED: the transposed reading — `heads` tiles a row, 8 rows — mixes heads. */
    strip.tile_count = KV_HEADS * TILES;
    pn_tt_kv_rotate_view(&strip, KV_HEADS, TILES, &view);
    bad = view;
    bad.tile_count = KV_HEADS;
    bad.tile_rows = bad.active_rows = TILES;
    {
        /* The emulation walks Kt = 8 tiles a row, so a 4-tile row pitch reads across heads. */
        const double err = rotation_error(KV_HEADS, 32u, &bad);

        snprintf(what, sizeof(what), "RED: a transposed view does NOT rotate the heads "
                 "(worst rel err %.2e)", err);
        check(err > 0.1, what);
    }
}

/* ---- 4. Attention invariance and the quantizer ---- */

/** Bfp4 round trip of one row of `DIM` (16-value face-row groups, as the cache packs). */
static void bfp4_row(const float *in, float *out)
{
    uint16_t bf[1024];
    uint8_t packed[576];

    for (uint32_t t = 0; t < TILES; t++) {
        memset(bf, 0, sizeof(bf));
        for (uint32_t c = 0; c < 32u; c++)
            bf[ft_tt_tile_face_index(0u, c)] = ft_bf16_from_float(in[t * 32u + c]);
        pn_tt_kv_format_tiles_from_bf16(PN_TT_KV_FORMAT_BFP4, bf, 1u, packed);
        pn_tt_kv_format_tiles_to_bf16(PN_TT_KV_FORMAT_BFP4, packed, 1u, bf);
        for (uint32_t c = 0; c < 32u; c++)
            out[t * 32u + c] = ft_float_from_bf16(bf[ft_tt_tile_face_index(0u, c)]);
    }
}

static void check_attention(void)
{
    enum { KEYS = 256 };
    float q[DIM], k[DIM], qh[DIM], kh[DIM], kq[DIM];
    double noise_bf16 = 0.0, noise_rot = 0.0, err_plain = 0.0, err_rot = 0.0;
    char what[200];

    for (int trial = 0; trial < 32; trial++) {
        fill_row(q, 12.0);
        for (int key = 0; key < KEYS; key++) {
            double dot = 0.0, dot_h = 0.0, dot_q = 0.0, dot_qh = 0.0, exact = 0.0;

            fill_row(k, 12.0);
            memcpy(qh, q, sizeof(q));
            memcpy(kh, k, sizeof(k));
            pn_tt_kv_rotate_rows(qh, 1u, DIM);
            pn_tt_kv_rotate_rows(kh, 1u, DIM);
            for (uint32_t d = 0; d < DIM; d++) {
                qh[d] = bf16_round(qh[d]); /* the card writes both rotated strips in bf16 */
                kh[d] = bf16_round(kh[d]);
                dot += (double)bf16_round(q[d]) * bf16_round(k[d]);
                dot_h += (double)qh[d] * kh[d];
                exact += (double)q[d] * k[d];
            }
            noise_bf16 += (dot - exact) * (dot - exact);
            noise_rot += (dot_h - exact) * (dot_h - exact);
            /* The score error the Bfp4 cache adds, plain K vs rotated K. */
            bfp4_row(k, kq);
            for (uint32_t d = 0; d < DIM; d++)
                dot_q += (double)bf16_round(q[d]) * kq[d];
            bfp4_row(kh, kq);
            for (uint32_t d = 0; d < DIM; d++)
                dot_qh += (double)qh[d] * kq[d];
            err_plain += (dot_q - dot) * (dot_q - dot);
            err_rot += (dot_qh - dot) * (dot_qh - dot);
        }
    }
    /* The rotation costs one more bf16 rounding of each operand: its score noise vs the fp32
     * q.k is at most ~sqrt(2)x that of plain bf16 operands (allowed 2x). */
    snprintf(what, sizeof(what), "(qH).(kH) = q.k within bf16 rounding (score noise RMS %.4f vs "
             "plain bf16 %.4f, <= 2x)", sqrt(noise_rot), sqrt(noise_bf16));
    printf("  %s\n", what);
    check(noise_rot <= 4.0 * noise_bf16, what);
    snprintf(what, sizeof(what), "rotated K cuts the Bfp4 score error on outlier rows "
             "(RMS %.3f -> %.3f, %.2fx)", sqrt(err_plain), sqrt(err_rot),
             sqrt(err_plain / err_rot));
    printf("  %s\n", what);
    check(err_rot < 0.7 * err_plain, what);
}

/* ---- 5. State files ---- */
static void check_state_roundtrip(void)
{
    enum { POS = 64 };
    float *canon = malloc(sizeof(float) * POS * DIM);
    uint16_t *bf = calloc((size_t)TILES * 2u * 1024u, sizeof(uint16_t));
    uint8_t *cache = malloc((size_t)TILES * 2u * 576u), *again = malloc((size_t)TILES * 2u * 576u);
    bool ok = canon && bf && cache && again;

    /* The card's cache: rows k*H, bf16, packed (pos tile p, dim tile t at p * 8 + t). */
    for (uint32_t p = 0; ok && p < POS; p++) {
        float row[DIM];

        fill_row(row, 15.0);
        pn_tt_kv_rotate_rows(row, 1u, DIM);
        for (uint32_t d = 0; d < DIM; d++)
            bf[((size_t)(p / 32u) * TILES + d / 32u) * 1024u +
               ft_tt_tile_face_index(p % 32u, d % 32u)] = ft_bf16_from_float(row[d]);
    }
    if (ok)
        pn_tt_kv_format_tiles_from_bf16(PN_TT_KV_FORMAT_BFP4, bf, TILES * 2u, cache);
    /* Export (pn_tt_kv_download): unpack, then un-rotate into the file's canonical rows. */
    if (ok)
        pn_tt_kv_format_tiles_to_bf16(PN_TT_KV_FORMAT_BFP4, cache, TILES * 2u, bf);
    for (uint32_t p = 0; ok && p < POS; p++) {
        for (uint32_t d = 0; d < DIM; d++)
            canon[p * DIM + d] = ft_float_from_bf16(
                bf[((size_t)(p / 32u) * TILES + d / 32u) * 1024u +
                   ft_tt_tile_face_index(p % 32u, d % 32u)]);
        pn_tt_kv_rotate_rows(canon + p * DIM, 1u, DIM);
    }
    /* Import (pn_tt_kv_upload): rotate, bf16, pack. */
    if (ok)
        memset(bf, 0, (size_t)TILES * 2u * 1024u * sizeof(uint16_t));
    for (uint32_t p = 0; ok && p < POS; p++) {
        float row[DIM];

        memcpy(row, canon + p * DIM, sizeof(row));
        pn_tt_kv_rotate_rows(row, 1u, DIM);
        for (uint32_t d = 0; d < DIM; d++)
            bf[((size_t)(p / 32u) * TILES + d / 32u) * 1024u +
               ft_tt_tile_face_index(p % 32u, d % 32u)] = ft_bf16_from_float(row[d]);
    }
    if (ok)
        pn_tt_kv_format_tiles_from_bf16(PN_TT_KV_FORMAT_BFP4, bf, TILES * 2u, again);
    check(ok && memcmp(cache, again, (size_t)TILES * 2u * 576u) == 0,
          "k4h export (un-rotate) then import (rotate) gives back the cache's bytes");
    free(canon), free(bf), free(cache), free(again);
}

/* ---- 6. Knob and catalog ---- */
/** The shipped catalog with `"kv_long": kv_long` added to one profile's options: the p150
 *  K8V4 profile (`p150`) or the first K80 one (`k80`). 0 parsed with the field read back. */
static int catalog_with_kv_long(bool p150, const char *kv_long, char *err, size_t cap)
{
    const char *anchor = p150 ? "\"kv_type\": \"k8v4\"," : "\"kv_type\": \"q4\",";
    const char *profile = p150 ? "p150-one" : "k80-pair";
    const char *model = p150 ? "qwen3.8-27b-q8" : "qwen3.6-35b-a3b-q3km";
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char *text = NULL, *json = NULL, *at;
    int rc = -3;

    if (m && pn_local_read_file("catalog/catalog.json", 4u << 20, &text, NULL) == 0 &&
        (at = strstr(text, anchor)) != NULL &&
        (json = malloc(strlen(text) + 64u)) != NULL) {
        const size_t head = (size_t)(at - text) + strlen(anchor);

        memcpy(json, text, head);
        sprintf(json + head, " \"kv_long\": \"%s\",%s", kv_long, text + head);
        rc = pn_cat_manifest_parse(json, m, err, cap);
        if (rc == 0) {
            const pn_cat_model *q = pn_cat_find_model(m, model);
            const pn_cat_profile *pr = q ? pn_cat_find_profile(q, profile) : NULL;

            if (!pr || strcmp(pr->options.kv_long, kv_long) != 0)
                rc = -2;
        }
    }
    free(text), free(json), free(m);
    return rc;
}

static void check_knob_and_catalog(void)
{
    pn_tt_kv_type type, other;
    char err[256];

    check(pn_tt_kv_type_parse("k4h", &type) && type.rotate &&
              type.k_format == PN_TT_KV_FORMAT_BFP4 && type.v_format == PN_TT_KV_FORMAT_BFP4 &&
              strcmp(pn_tt_kv_type_name(type), "k4h") == 0 &&
              strcmp(pn_tt_kv_type_attention_program(type), "attention_bfp4") == 0,
          "k4h = Bfp4 K and V, rotated, read by attention_bfp4");
    check(pn_tt_kv_type_parse("bfp4", &other) && !other.rotate &&
              strcmp(pn_tt_kv_type_name(other), "bfp4") == 0,
          "bfp4 stays unrotated and keeps its name");

    unsetenv(PN_TT_KV_LONG_ENV);
    check(!pn_tt_kv_type_long(&type), "PN_TT_KV_LONG unset: no long type");
    setenv(PN_TT_KV_LONG_ENV, "k4h", 1);
    check(pn_tt_kv_type_long(&type) && type.rotate, "PN_TT_KV_LONG=k4h: the long type is k4h");
    setenv(PN_TT_KV_LONG_ENV, "bf16", 1);
    check(!pn_tt_kv_type_long(&type), "PN_TT_KV_LONG=bf16 is no long type");
    setenv(PN_TT_KV_LONG_ENV, "k9", 1);
    check(!pn_tt_kv_type_long(&type), "an unknown PN_TT_KV_LONG is no long type");
    unsetenv(PN_TT_KV_LONG_ENV);

    check(pn_cat_kv_type_allowed("p150", "k4h") && !pn_cat_kv_type_allowed("k80-gpu", "k4h"),
          "k4h is a p150 catalog kv_type only");
    check(catalog_with_kv_long(true, "k4h", err, sizeof(err)) == 0,
          "options.kv_long k4h parses on a p150 profile");
    check(catalog_with_kv_long(false, "k4h", err, sizeof(err)) == -1,
          "options.kv_long on a k80 profile is refused");
    check(catalog_with_kv_long(true, "bf16", err, sizeof(err)) == -1,
          "options.kv_long bf16 is refused");
    check(catalog_with_kv_long(true, "k9", err, sizeof(err)) == -1,
          "an unknown options.kv_long is refused");
}

int main(void)
{
    check_h();
    check_weight();
    check_card_op();
    check_attention();
    check_state_roundtrip();
    check_knob_and_catalog();
    printf("test_tt_kv_rotate: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("test_tt_kv_rotate: SKIPPED (build with USE_TT=1)\n");
    return 0;
}

#endif /* USE_TT */
