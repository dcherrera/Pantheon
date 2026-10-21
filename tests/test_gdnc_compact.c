/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_gdnc_compact.c
 * @brief L11b: the gdnc reader's on-card decay tile + gate runs (foundry/tt/kernels/
 *        ftgc_compact.h) against (1) the hardware fp32 multiply under FTZ+DAZ and (2) the
 *        host builders `ft_tt_op_gdnc_build_decay(_at)` / `_build_gate(_at)`, BIT FOR BIT.
 *        x86 is the authority: run it on an x86 host.
 *
 *   make USE_TT=1 tests/test_gdnc_compact && ./tests/test_gdnc_compact
 *
 * Red: `--red` flips one rounding (ties go up) and must FAIL.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "tt/ft_tt_op_gdnc.h"
#include "tt/kernels/ftgc_compact.h"

static int g_red;

static uint32_t f2u(float f)
{
    uint32_t u;

    memcpy(&u, &f, 4);
    return u;
}

static float u2f(uint32_t u)
{
    float f;

    memcpy(&f, &u, 4);
    return f;
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 16);
}

/** The card's multiply, or the red mutation of it (ties away from even). */
static uint32_t card_mul(uint32_t a, uint32_t b)
{
    uint32_t r = ftgc_mul(a, b);

    if (g_red && ((rnd() & 1023u) == 0u))
        r ^= 1u;
    return r;
}

static void set_ftz_daz(void)
{
#if defined(__x86_64__)
    _mm_setcsr(_mm_getcsr() | 0x8040u); /* FTZ | DAZ */
#elif defined(__aarch64__)
    uint64_t fpcr;

    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= 1ull << 24; /* FZ: AArch64 flushes on tininess BEFORE rounding — not the host's */
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    fprintf(stderr, "test_gdnc_compact: AArch64 FZ — informational; x86 is the authority\n");
#else
    fprintf(stderr, "test_gdnc_compact: not x86 — the hardware leg is not the host's\n");
#endif
}

static int mul_leg(void)
{
    volatile float va, vb;
    uint64_t n, bad = 0;

    for (n = 0; n < 200000000ull; n++) {
        uint32_t a = rnd() & 0x7FFFFFFFu, b = rnd() & 0x7FFFFFFFu;

        if (n & 1u) { /* near the underflow edge: exponents summing to ~127-126 */
            a = (a & 0x807FFFFFu) | ((60u + (rnd() % 8u)) << 23);
            b = (b & 0x807FFFFFu) | ((rnd() % 8u) << 23);
        }
        if ((a & 0x7F800000u) == 0x7F800000u || (b & 0x7F800000u) == 0x7F800000u)
            continue;
        va = u2f(a);
        vb = u2f(b);
        if (f2u(va * vb) != card_mul(a, b)) {
            if (bad < 5)
                fprintf(stderr, "  mul %08x * %08x: hw %08x card %08x\n", a, b,
                        f2u(va * vb), ftgc_mul(a, b));
            bad++;
        }
    }
    printf("mul leg: %llu of %llu differ\n", (unsigned long long)bad, (unsigned long long)n);
    return bad != 0;
}

static float draw_decay(void)
{
    /* exp(g) with g from -20 (1e-9: underflows inside a chunk) to 0 */
    const double g = -20.0 * pow((double)(rnd() % 1000000u) / 1e6, 3.0);

    return (float)exp(g);
}

static int builder_leg(void)
{
    uint64_t cases = 0, bad = 0;
    unsigned trial;
    float scale;

    {
        float d1 = 1.0f, b1 = 0.0f, out[FT_TT_OP_GDNC_GATE_STRIDE];

        if (ft_tt_op_gdnc_build_gate(&d1, &b1, 1u, 128u, out, FT_TT_OP_GDNC_GATE_STRIDE) !=
            FT_TT_OK)
            return 1;
        scale = out[FT_TT_OP_GDNC_GATE_PS * 32u];
    }
    for (trial = 0; trial < 20000u; trial++) {
        const uint32_t first_row = (trial & 1u) ? 16u : 0u;
        const uint32_t max_rows = 32u - first_row;
        const uint32_t chunk_rows = 1u + rnd() % max_rows;
        float decay[32], beta[32], host_gate[FT_TT_OP_GDNC_GATE_STRIDE];
        unsigned char host_tile[FT_TT_OP_GDNC_FP32_TILE_BYTES];
        uint32_t card_decay[32], card_beta[32], card_tile[1024], card_gate[160];
        uint32_t i;

        for (i = 0; i < 32u; i++) {
            decay[i] = (i < chunk_rows) ? draw_decay() : 0.0f;
            beta[i] = (i < chunk_rows) ? (float)(rnd() % 1000000u) / 1e6f : 0.0f;
            card_decay[i] = f2u(decay[i]);
            card_beta[i] = f2u(beta[i]);
        }
        if (ft_tt_op_gdnc_build_decay_at(decay, first_row, chunk_rows, host_tile,
                                         sizeof(host_tile)) != FT_TT_OK ||
            ft_tt_op_gdnc_build_gate_at(decay, beta, first_row, chunk_rows, 128u, host_gate,
                                        FT_TT_OP_GDNC_GATE_STRIDE) != FT_TT_OK)
            return 1;
        ftgc_compact_decay_tile(card_decay, first_row, chunk_rows, card_tile);
        ftgc_compact_gate(card_decay, card_beta, first_row, chunk_rows, f2u(scale), card_gate);
        if (g_red)
            card_tile[ftgc_tile_index(chunk_rows - 1u + first_row, first_row)] ^= 1u;
        cases++;
        if (memcmp(host_tile, card_tile, sizeof(host_tile)) != 0 ||
            memcmp(host_gate, card_gate, sizeof(card_gate)) != 0) {
            if (bad < 5)
                fprintf(stderr, "  builder case %u: first_row %u rows %u — tile %s, gate %s\n",
                        trial, first_row, chunk_rows,
                        memcmp(host_tile, card_tile, sizeof(host_tile)) ? "DIFFERS" : "same",
                        memcmp(host_gate, card_gate, sizeof(card_gate)) ? "DIFFERS" : "same");
            bad++;
        }
    }
    printf("builder leg: %llu of %llu cases differ (scale %08x)\n", (unsigned long long)bad,
           (unsigned long long)cases, f2u(scale));
    return bad != 0;
}

int main(int argc, char **argv)
{
    int fail;

    g_red = argc > 1 && strcmp(argv[1], "--red") == 0;
    set_ftz_daz();
    fail = mul_leg();
    fail |= builder_leg();
    printf("%s%s\n", fail ? "FAIL" : "PASS", g_red ? " (red arm: FAIL is the pass)" : "");
    return fail;
}
