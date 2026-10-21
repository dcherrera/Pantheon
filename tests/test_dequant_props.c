/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Property tests for GGUF dequantization (src/formats/gguf_dequant.c).
 *
 * - Block struct sizes and field offsets.
 * - pn_half_to_float on all 65536 inputs vs ldexp-based math (bit-compared).
 * - pn_float_to_half on all 2^32 inputs vs an independent RNE reference
 *   (frexp + nearbyint in double), and half->float->half identity for non-NaN.
 * - Every block format vs a per-element reference written straight from the
 *   format description (double math, one final rounding), on random blocks and
 *   on single-bit blocks (catches index / bit-order mistakes).
 * - Edge scales (zero, subnormal, max, Inf, NaN): class checks via bits.
 * - pn_dequantize argument errors, unsupported types (exact stderr), partial tails.
 * - Size helpers and names; tensor/row helpers on a synthetic pn_gguf_file.
 */

#include "lib/test.h"
#include "../src/formats/gguf.h"
#include "../src/formats/gguf_dequant.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Bits via a volatile union: under -ffinite-math-only the compiler may otherwise
 * "know" a float is finite and fold Inf/NaN bit tests away. */
static uint32_t fbits(float f) {
    volatile union { float f; uint32_t u; } x;
    x.f = f;
    return x.u;
}
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 16);
}

/* ---- stderr capture ---- */
static int cap_saved = -1, cap_pipe[2];
static void cap_begin(void) {
    fflush(stderr);
    if (pipe(cap_pipe) != 0) return;
    cap_saved = dup(2);
    dup2(cap_pipe[1], 2);
    close(cap_pipe[1]);
}
static void cap_end(char *out, size_t cap) {
    fflush(stderr);
    dup2(cap_saved, 2);
    close(cap_saved);
    ssize_t r = read(cap_pipe[0], out, cap - 1);
    out[r > 0 ? r : 0] = 0;
    close(cap_pipe[0]);
}

/* ------------------------------------------------------------------ */

static void test_struct_layout(void) {
    ASSERT_INT_EQ((int)sizeof(pn_block_q4_0), 18);
    ASSERT_INT_EQ((int)offsetof(pn_block_q4_0, qs), 2);
    ASSERT_INT_EQ((int)sizeof(pn_block_q8_0), 34);
    ASSERT_INT_EQ((int)offsetof(pn_block_q8_0, qs), 2);
    ASSERT_INT_EQ((int)sizeof(pn_block_q2_0_g128), 34);
    ASSERT_INT_EQ((int)offsetof(pn_block_q2_0_g128, qs), 2);
    ASSERT_INT_EQ((int)sizeof(pn_block_q3_k), 110);
    ASSERT_INT_EQ((int)offsetof(pn_block_q3_k, hmask), 0);
    ASSERT_INT_EQ((int)offsetof(pn_block_q3_k, qs), 32);
    ASSERT_INT_EQ((int)offsetof(pn_block_q3_k, scales), 96);
    ASSERT_INT_EQ((int)offsetof(pn_block_q3_k, d), 108);
    ASSERT_INT_EQ((int)sizeof(pn_block_q4_k), 144);
    ASSERT_INT_EQ((int)offsetof(pn_block_q4_k, dmin), 2);
    ASSERT_INT_EQ((int)offsetof(pn_block_q4_k, scales), 4);
    ASSERT_INT_EQ((int)offsetof(pn_block_q4_k, qs), 16);
    ASSERT_INT_EQ((int)sizeof(pn_block_q5_k), 176);
    ASSERT_INT_EQ((int)offsetof(pn_block_q5_k, qh), 16);
    ASSERT_INT_EQ((int)offsetof(pn_block_q5_k, qs), 48);
    ASSERT_INT_EQ((int)sizeof(pn_block_q6_k), 210);
    ASSERT_INT_EQ((int)offsetof(pn_block_q6_k, qh), 128);
    ASSERT_INT_EQ((int)offsetof(pn_block_q6_k, scales), 192);
    ASSERT_INT_EQ((int)offsetof(pn_block_q6_k, d), 208);
}

/* Reference widening: ldexp math for finite values, bit formula for Inf/NaN. */
static uint32_t ref_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31;
    int e = (h >> 10) & 31, m = h & 0x3FF;
    if (e == 31) return s | 0x7F800000u | ((uint32_t)m << 13);
    double v = e == 0 ? ldexp((double)m, -24) : ldexp(1024.0 + m, e - 25);
    return s | fbits((float)v);
}

static void test_half_to_float_all(void) {
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++)
        if (fbits(pn_half_to_float((uint16_t)h)) != ref_h2f((uint16_t)h)) {
            if (bad++ < 5) printf("    h2f mismatch 0x%04X\n", h);
        }
    ASSERT_INT_EQ(bad, 0);
}

/* Reference narrowing via double: exact scaling by powers of two + nearbyint (RNE). */
static uint16_t ref_f2h(uint32_t x) {
    uint16_t s = (uint16_t)((x >> 16) & 0x8000);
    uint32_t E = (x >> 23) & 0xFF, M = x & 0x7FFFFF;
    if (E == 0xFF) return (uint16_t)(s | 0x7C00 | (M ? 0x200 : 0));
    if (E == 0) return s; /* float zero/subnormal: far below half range */
    double v = ldexp((double)(M | 0x800000), (int)E - 150); /* |f| exactly */
    if (v >= 65520.0) return (uint16_t)(s | 0x7C00);
    if (v < ldexp(1.0, -14)) {
        double q = nearbyint(ldexp(v, 24));
        return (uint16_t)(s | (uint16_t)q);
    }
    int ex;
    double fr = frexp(v, &ex);          /* v = fr * 2^ex, fr in [0.5, 1) */
    double q = nearbyint(ldexp(fr, 11)); /* 1024..2048 */
    int e = ex - 1;
    if (q >= 2048.0) { q = 1024.0; e++; }
    if (e + 15 >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | ((e + 15) << 10) | ((uint16_t)q - 1024));
}

static void test_float_to_half_all(void) {
    uint64_t bad = 0;
    for (uint64_t x = 0; x <= 0xFFFFFFFFull; x++) {
        uint32_t u = (uint32_t)x;
        uint16_t got = pn_float_to_half(bitsf(u));
        if (got != ref_f2h(u)) {
            if (bad++ < 5) printf("    f2h mismatch 0x%08X: got 0x%04X want 0x%04X\n", u, got, ref_f2h(u));
        }
    }
    ASSERT_INT_EQ((int)bad, 0);
}

static void test_half_round_trip(void) {
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        if (((h >> 10) & 31) == 31 && (h & 0x3FF)) continue; /* NaN */
        if (pn_float_to_half(pn_half_to_float((uint16_t)h)) != h) bad++;
    }
    ASSERT_INT_EQ(bad, 0);
    /* NaNs canonicalise to the quiet NaN, sign kept. */
    ASSERT_INT_EQ(pn_float_to_half(pn_half_to_float(0x7C01)), 0x7E00);
    ASSERT_INT_EQ(pn_float_to_half(pn_half_to_float(0xFD55)), 0xFE00);
    /* Signalling half NaN payload survives widening. */
    ASSERT_INT_EQ((int)(fbits(pn_half_to_float(0x7C01)) == 0x7F802000u), 1);
}

/* ---- per-element reference decoders (straight from the format description) ---- */

static double H(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return (double)bitsf(ref_h2f(h)); }

static float ref_q4_0(const uint8_t *b, int e) {
    int nib = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4);
    return (float)(H(b) * (nib - 8));
}
static float ref_q8_0(const uint8_t *b, int e) { return (float)(H(b) * (int8_t)b[2 + e]); }
static float ref_q2_0(const uint8_t *b, int e) {
    int code = (b[2 + e / 4] >> (2 * (e % 4))) & 3;
    return (float)(H(b) * (code - 1));
}
static float ref_q3_k(const uint8_t *b, int e) {
    const uint8_t *hmask = b, *qs = b + 32, *S = b + 96;
    int h = e / 128, j = (e % 128) / 32, l = e % 32;
    int k = 8 * h + 2 * j + l / 16;
    /* six-bit scale k: low 4 bits from S[k % 8] (nibble k / 8), high 2 bits from
     * S[8 + k % 4] at bit position 2 * (k / 4). */
    int lo = k < 8 ? (S[k] & 0xF) : (S[k - 8] >> 4);
    int hi = (S[8 + k % 4] >> (2 * (k / 4))) & 3;
    int sc = (lo | (hi << 4)) - 32;
    int q = ((qs[32 * h + l] >> (2 * j)) & 3) - (((hmask[l] >> (4 * h + j)) & 1) ? 0 : 4);
    return (float)(H(b + 108) * sc * q);
}
static void ref_scale_min(const uint8_t *S, int k, int *sc, int *mn) {
    if (k < 4) { *sc = S[k] & 63; *mn = S[k + 4] & 63; }
    else {
        *sc = (S[k + 4] & 0xF) | ((S[k - 4] >> 6) << 4);
        *mn = (S[k + 4] >> 4) | ((S[k] >> 6) << 4);
    }
}
static float ref_q45_k(const uint8_t *b, int e, int five) {
    const uint8_t *S = b + 4, *qh = b + 16, *qs = five ? b + 48 : b + 16;
    int p = e / 64, hi = (e % 64) / 32, l = e % 32, k = 2 * p + hi;
    int q = hi ? (qs[32 * p + l] >> 4) : (qs[32 * p + l] & 0xF);
    if (five && ((qh[l] >> k) & 1)) q += 16;
    int sc, mn;
    ref_scale_min(S, k, &sc, &mn);
    return (float)(H(b) * sc * q - H(b + 2) * mn);
}
static float ref_q6_k(const uint8_t *b, int e) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    int h = e / 128, r = e % 128, quad = r / 32, l = r % 32;
    int low = quad & 1 ? ql[64 * h + l + 32] : ql[64 * h + l];
    low = quad >= 2 ? low >> 4 : low & 0xF;
    int high = (qh[32 * h + l] >> (2 * quad)) & 3;
    int q = (low | (high << 4)) - 32;
    return (float)(H(b + 208) * sc[8 * h + 2 * quad + l / 16] * q);
}

typedef struct {
    const char *name;
    pn_gguf_quant type;
    int elems, bytes;
    int d_off, dmin_off;    /* dmin_off < 0 when absent */
    void (*fn)(const void *, float *);
    float (*ref)(const uint8_t *, int);
} fmt;

static float ref_q4_k(const uint8_t *b, int e) { return ref_q45_k(b, e, 0); }
static float ref_q5_k(const uint8_t *b, int e) { return ref_q45_k(b, e, 1); }

static const fmt FMTS[] = {
    { "Q4_0", GGUF_QUANT_Q4_0, 32, 18, 0, -1, pn_dequantize_block_q4_0, ref_q4_0 },
    { "Q8_0", GGUF_QUANT_Q8_0, 32, 34, 0, -1, pn_dequantize_block_q8_0, ref_q8_0 },
    { "Q2_0_G128", GGUF_QUANT_Q2_0_G128, 128, 34, 0, -1, pn_dequantize_block_q2_0_g128, ref_q2_0 },
    { "Q3_K", GGUF_QUANT_Q3_K, 256, 110, 108, -1, pn_dequantize_block_q3_k, ref_q3_k },
    { "Q4_K", GGUF_QUANT_Q4_K, 256, 144, 0, 2, pn_dequantize_block_q4_k, ref_q4_k },
    { "Q5_K", GGUF_QUANT_Q5_K, 256, 176, 0, 2, pn_dequantize_block_q5_k, ref_q5_k },
    { "Q6_K", GGUF_QUANT_Q6_K, 256, 210, 208, -1, pn_dequantize_block_q6_k, ref_q6_k },
};
#define N_FMTS (int)(sizeof FMTS / sizeof FMTS[0])

/* A finite half with a moderate exponent (keeps the K-quant min subtraction exact
 * in double, so the reference is the correctly rounded value). */
static uint16_t rand_half(void) {
    uint16_t e = (uint16_t)(5 + rnd() % 20);
    return (uint16_t)(((rnd() & 1) << 15) | (e << 10) | (rnd() & 0x3FF));
}

static int compare_block(const fmt *f, const uint8_t *blk) {
    float out[256];
    f->fn(blk, out);
    int bad = 0;
    for (int e = 0; e < f->elems; e++)
        if (fbits(out[e]) != fbits(f->ref(blk, e))) {
            if (bad++ < 3) printf("    %s elem %d: got %a want %a\n", f->name, e, out[e], f->ref(blk, e));
        }
    return bad;
}

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

static void test_blocks_random(void) {
    int bad = 0;
    uint8_t blk[256 + 3];
    for (int fi = 0; fi < N_FMTS; fi++) {
        const fmt *f = &FMTS[fi];
        for (int it = 0; it < 2000; it++) {
            uint8_t *b = blk + (it % 3); /* exercise unaligned reads */
            for (int i = 0; i < f->bytes; i++) b[i] = (uint8_t)rnd();
            put16(b + f->d_off, rand_half());
            if (f->dmin_off >= 0) put16(b + f->dmin_off, rand_half());
            bad += compare_block(f, b);
        }
    }
    ASSERT_INT_EQ(bad, 0);
}

static int in_scale_field(const fmt *f, int byte) {
    return (byte >= f->d_off && byte < f->d_off + 2) ||
           (f->dmin_off >= 0 && byte >= f->dmin_off && byte < f->dmin_off + 2);
}

/* One bit set in an otherwise-zero payload (scales 1.0 / 0.5), every bit position. */
static void test_blocks_single_bit(void) {
    int bad = 0;
    uint8_t b[256];
    for (int fi = 0; fi < N_FMTS; fi++) {
        const fmt *f = &FMTS[fi];
        for (int byte = 0; byte < f->bytes; byte++) {
            if (in_scale_field(f, byte)) continue;
            for (int bit = 0; bit < 8; bit++) {
                for (int base = 0; base < 2; base++) {
                    memset(b, base ? 0xFF : 0x00, (size_t)f->bytes);
                    put16(b + f->d_off, 0x3C00);
                    if (f->dmin_off >= 0) put16(b + f->dmin_off, 0x3800);
                    b[byte] ^= (uint8_t)(1u << bit);
                    bad += compare_block(f, b);
                }
            }
        }
    }
    ASSERT_INT_EQ(bad, 0);
}

static int is_zero(float v) { return (fbits(v) & 0x7FFFFFFFu) == 0; }
static int is_nonfinite(float v) { return (fbits(v) & 0x7F800000u) == 0x7F800000u; }

static void test_blocks_edge_scales(void) {
    int bad = 0;
    uint8_t b[256];
    static const uint16_t finite_edges[] = { 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x0400, 0x03FF };
    for (int fi = 0; fi < N_FMTS; fi++) {
        const fmt *f = &FMTS[fi];
        for (int it = 0; it < 50; it++) {
            for (int i = 0; i < f->bytes; i++) b[i] = (uint8_t)rnd();
            /* finite edges: exact reference applies */
            for (size_t k = 0; k < sizeof finite_edges / 2; k++) {
                put16(b + f->d_off, finite_edges[k]);
                if (f->dmin_off >= 0) put16(b + f->dmin_off, finite_edges[(k + 1) % 6]);
                bad += compare_block(f, b);
            }
            float out[256];
            /* zero scales -> zero outputs */
            put16(b + f->d_off, (uint16_t)((it & 1) ? 0x8000 : 0x0000));
            if (f->dmin_off >= 0) put16(b + f->dmin_off, 0x0000);
            f->fn(b, out);
            for (int e = 0; e < f->elems; e++) if (!is_zero(out[e])) { if (bad++ < 3) printf("    %s zero-scale elem %d = %08X\n", f->name, e, fbits(out[e])); }
            /* Inf / NaN d (formats without a min term): non-finite wherever the
             * d-term is non-zero (probed with d = 1), and everywhere for NaN. */
            if (f->dmin_off >= 0) continue;
            static const uint16_t nonfinite[] = { 0x7C00, 0xFC00, 0x7E00, 0x7C01 };
            float probe[256];
            put16(b + f->d_off, 0x3C00);
            f->fn(b, probe);
            for (int k = 0; k < 4; k++) {
                put16(b + f->d_off, nonfinite[k]);
                f->fn(b, out);
                for (int e = 0; e < f->elems; e++) {
                    if ((!is_zero(probe[e]) || k >= 2) && !is_nonfinite(out[e])) {
                        if (bad++ < 3) printf("    %s d=%04X elem %d = %08X\n", f->name, nonfinite[k], e, fbits(out[e]));
                    }
                }
            }
        }
    }
    ASSERT_INT_EQ(bad, 0);
}

static void test_dequantize_args_and_tail(void) {
    char err[256];
    float dst[300];
    uint8_t src[2048];
    for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)rnd();
    for (int i = 0; i < 8; i++) put16(src + 210 * i + 208, 0x3C00);

    cap_begin();
    int r1 = pn_dequantize(NULL, dst, 4, GGUF_QUANT_F32);
    int r2 = pn_dequantize(src, NULL, 4, GGUF_QUANT_F32);
    int r3 = pn_dequantize(src, dst, 0, GGUF_QUANT_F32);
    int r4 = pn_dequantize(src, dst, 0, (pn_gguf_quant)99);
    cap_end(err, sizeof err);
    ASSERT_INT_EQ(r1, -1); ASSERT_INT_EQ(r2, -1); ASSERT_INT_EQ(r3, -1); ASSERT_INT_EQ(r4, -1);
    ASSERT_INT_EQ((int)strlen(err), 0);

    static const int unsupported[] = { 3, 6, 7, 9, 10, 15, 100, 4, 5, 16, 29, 31, 141, 255, -1 };
    for (size_t i = 0; i < sizeof unsupported / sizeof unsupported[0]; i++) {
        char want[96];
        snprintf(want, sizeof want, "pn_dequantize: unsupported quant type %d\n", unsupported[i]);
        cap_begin();
        int r = pn_dequantize(src, dst, 256, (pn_gguf_quant)unsupported[i]);
        cap_end(err, sizeof err);
        ASSERT_INT_EQ(r, -1);
        ASSERT_INT_EQ(strcmp(err, want), 0);
    }

    /* Partial trailing block: tail untouched, returns 0. */
    const float sentinel = -12345.0f;
    for (int i = 0; i < 300; i++) dst[i] = sentinel;
    ASSERT_INT_EQ(pn_dequantize(src, dst, 300, GGUF_QUANT_Q6_K), 0);
    int tail_ok = 1;
    for (int i = 256; i < 300; i++) if (fbits(dst[i]) != fbits(sentinel)) tail_ok = 0;
    ASSERT_TRUE(tail_ok);
    for (int i = 0; i < 300; i++) dst[i] = sentinel;
    ASSERT_INT_EQ(pn_dequantize(src, dst, 100, GGUF_QUANT_Q4_K), 0);
    tail_ok = 1;
    for (int i = 0; i < 300; i++) if (fbits(dst[i]) != fbits(sentinel)) tail_ok = 0;
    ASSERT_TRUE(tail_ok);

    /* Bulk == per-block for every block format. */
    int bad = 0;
    for (int fi = 0; fi < N_FMTS; fi++) {
        const fmt *f = &FMTS[fi];
        for (int i = 0; i < 2; i++) put16(src + f->bytes * i + f->d_off, 0x3555);
        float bulk[512], one[256];
        ASSERT_INT_EQ(pn_dequantize(src, bulk, (size_t)(2 * f->elems), f->type), 0);
        for (int i = 0; i < 2; i++) {
            f->fn(src + f->bytes * i, one);
            if (memcmp(one, bulk + f->elems * i, (size_t)f->elems * 4) != 0) bad++;
        }
    }
    ASSERT_INT_EQ(bad, 0);

    /* F16 / BF16 / F32 are bit moves (subnormals and NaN payloads preserved). */
    uint16_t h[4] = { 0x0001, 0x7C01, 0x8000, 0x7F81 };
    float o[4];
    ASSERT_INT_EQ(pn_dequantize(h, o, 4, GGUF_QUANT_BF16), 0);
    for (int i = 0; i < 4; i++) ASSERT_INT_EQ((int)(fbits(o[i]) == (uint32_t)h[i] << 16), 1);
    ASSERT_INT_EQ(pn_dequantize(h, o, 4, GGUF_QUANT_F16), 0);
    for (int i = 0; i < 4; i++) ASSERT_INT_EQ((int)(fbits(o[i]) == ref_h2f(h[i])), 1);
    uint32_t fsrc[3] = { 0x00000001u, 0x7F800001u, 0x80000000u }, fdst[3];
    ASSERT_INT_EQ(pn_dequantize(fsrc, (float *)fdst, 3, GGUF_QUANT_F32), 0);
    ASSERT_INT_EQ(memcmp(fsrc, fdst, sizeof fsrc), 0);
}

static void test_size_helpers_and_names(void) {
    static const struct { int t, bs; int bpb; const char *name; } T[] = {
        { 0, 1, 4, "F32" }, { 1, 1, 2, "F16" }, { 30, 1, 2, "BF16" },
        { 2, 32, 18, "Q4_0" }, { 3, 32, 0, "Q4_1" }, { 6, 32, 0, "Q5_0" }, { 7, 32, 0, "Q5_1" },
        { 8, 32, 34, "Q8_0" }, { 9, 32, 0, "Q8_1" },
        { 10, 256, 0, "Q2_K" }, { 11, 256, 110, "Q3_K" }, { 12, 256, 144, "Q4_K" },
        { 13, 256, 176, "Q5_K" }, { 14, 256, 210, "Q6_K" }, { 15, 256, 0, "Q8_K" },
        { 142, 128, 34, "Q2_0_G128" },
        { 100, 0, 0, "type100" }, { 4, 0, 0, "type4" }, { 5, 0, 0, "type5" },
        { 16, 0, 0, "type16" }, { 29, 0, 0, "type29" }, { 31, 0, 0, "type31" },
        { 141, 0, 0, "type141" }, { 255, 0, 0, "type255" }, { -1, 0, 0, "type-1" },
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
        pn_gguf_quant q = (pn_gguf_quant)T[i].t;
        char buf[PN_GGUF_QUANT_NAME_MAX];
        if (pn_gguf_block_size(q) != T[i].bs) { bad++; printf("    block_size %d\n", T[i].t); }
        if ((int)pn_gguf_bytes_per_block(q) != T[i].bpb) { bad++; printf("    bpb %d\n", T[i].t); }
        size_t want = (T[i].bs > 0 && T[i].bpb > 0) ? (1000 / (size_t)T[i].bs) * (size_t)T[i].bpb : 0;
        if (pn_gguf_type_size(q, 1000) != want) { bad++; printf("    type_size %d\n", T[i].t); }
        const char *r = pn_gguf_quant_name(q, buf, sizeof buf);
        if (r != buf || strcmp(buf, T[i].name) != 0) { bad++; printf("    name %d: %s\n", T[i].t, buf); }
    }
    ASSERT_INT_EQ(bad, 0);

    char small[4];
    ASSERT_TRUE(pn_gguf_quant_name(GGUF_QUANT_Q2_0_G128, small, sizeof small) == small);
    ASSERT_INT_EQ(strcmp(small, "Q2_"), 0);
    char one[1] = { 'x' };
    pn_gguf_quant_name(GGUF_QUANT_F32, one, 1);
    ASSERT_INT_EQ(one[0], 0);
    char keep[4] = "abc";
    const char *r = pn_gguf_quant_name(GGUF_QUANT_F32, keep, 0);
    ASSERT_TRUE(r != keep && r[0] == 0);
    ASSERT_INT_EQ(strcmp(keep, "abc"), 0);
    r = pn_gguf_quant_name(GGUF_QUANT_F32, NULL, 8);
    ASSERT_TRUE(r != NULL && r[0] == 0);
}

/* ---- synthetic file ---- */

static void test_tensor_and_row_helpers(void) {
    enum { ROWS = 4, COLS = 512 };
    static uint8_t data[64 + ROWS * (COLS / 256) * 144];
    for (size_t i = 0; i < sizeof data; i++) data[i] = (uint8_t)rnd();
    for (int b = 0; b < ROWS * COLS / 256; b++) {
        put16(data + 64 + 144 * b, rand_half());
        put16(data + 64 + 144 * b + 2, rand_half());
    }

    pn_gguf_tensor_info ti[3];
    memset(ti, 0, sizeof ti);
    static char n0[] = "w.q4k", n1[] = "w.q4_1", n2[] = "w.tern";
    ti[0].name.data = n0; ti[0].name.len = strlen(n0);
    ti[0].n_dims = 2; ti[0].dims[0] = COLS; ti[0].dims[1] = ROWS;
    ti[0].type = GGUF_QUANT_Q4_K; ti[0].offset = 64;
    ti[1].name.data = n1; ti[1].name.len = strlen(n1);
    ti[1].n_dims = 1; ti[1].dims[0] = 64; ti[1].type = GGUF_QUANT_Q4_1;
    ti[2].name.data = n2; ti[2].name.len = strlen(n2);
    ti[2].n_dims = 1; ti[2].dims[0] = 64; ti[2].type = PN_QUANT_TERNARY;

    pn_gguf_file file;
    memset(&file, 0, sizeof file);
    file.n_tensors = 3;
    file.tensors = ti;
    file.data = data;
    file.data_size = sizeof data;

    char err[512];
    size_t numel = 0;
    float *full = pn_gguf_dequantize_tensor(&file, "w.q4k", &numel);
    ASSERT_NOT_NULL(full);
    ASSERT_INT_EQ((int)numel, ROWS * COLS);

    float row[COLS];
    int bad = 0;
    for (int r = 0; r < ROWS; r++) {
        if (pn_gguf_dequantize_row(&file, "w.q4k", (size_t)r, COLS, row) != 0) bad++;
        if (memcmp(row, full + r * COLS, sizeof row) != 0) bad++;
    }
    ASSERT_INT_EQ(bad, 0);
    free(full);

    cap_begin();
    ASSERT_NULL(pn_gguf_dequantize_tensor(NULL, "w.q4k", &numel));
    ASSERT_NULL(pn_gguf_dequantize_tensor(&file, NULL, &numel));
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "missing", 0, COLS, row), -1);
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "w.q4k", 1, 100, row), -1);  /* unaligned */
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "w.q4k", 0, 0, row), -1);
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "w.tern", 0, 32, row), -1);  /* no block size */
    ASSERT_INT_EQ(pn_gguf_dequantize_row(NULL, "w.q4k", 0, COLS, row), -1);
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "w.q4k", 0, COLS, NULL), -1);
    cap_end(err, sizeof err);
    ASSERT_INT_EQ((int)strlen(err), 0);

    cap_begin();
    ASSERT_NULL(pn_gguf_dequantize_tensor(&file, "missing", &numel));
    cap_end(err, sizeof err);
    ASSERT_INT_EQ(strcmp(err, "pn_gguf_dequantize_tensor: tensor 'missing' not found\n"), 0);

    cap_begin();
    ASSERT_NULL(pn_gguf_dequantize_tensor(&file, "w.q4_1", &numel));
    cap_end(err, sizeof err);
    ASSERT_INT_EQ(strcmp(err, "pn_dequantize: unsupported quant type 3\n"
                              "pn_gguf_dequantize_tensor: unsupported type 3 for 'w.q4_1'\n"), 0);

    /* Unsizeable type through the row helper: byte_off 0, then the dequant error. */
    cap_begin();
    ASSERT_INT_EQ(pn_gguf_dequantize_row(&file, "w.q4_1", 1, 32, row), -1);
    cap_end(err, sizeof err);
    ASSERT_INT_EQ(strcmp(err, "pn_dequantize: unsupported quant type 3\n"), 0);
}

int main(void) {
    printf("\n  GGUF dequant property tests\n\n");

    RUN_TEST(struct_layout);
    RUN_TEST(half_to_float_all);
    RUN_TEST(half_round_trip);
    RUN_TEST(float_to_half_all);
    RUN_TEST(blocks_random);
    RUN_TEST(blocks_single_bit);
    RUN_TEST(blocks_edge_scales);
    RUN_TEST(dequantize_args_and_tail);
    RUN_TEST(size_helpers_and_names);
    RUN_TEST(tensor_and_row_helpers);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
