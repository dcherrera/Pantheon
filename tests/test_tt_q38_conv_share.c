/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_conv_share.c
 * @brief Option D of the Qwen3.8 fit: one conv weight for both GDN
 *        families. The census of the production GDN/GDNC placement on a host allocator, and
 *        the row-0 identity that makes the alias bit-exact.
 *
 * Opens no device. A host `pn_tt_device` holds only a real `ft_tt_allocator` (8 p150 banks)
 * and a DRAM writer that discards; the placement code is production's own:
 * `pn_tt_gdn_place_regions()`, `pn_tt_gdnc_place_regions()`, `pn_tt_gdn_conv_share_bind()`
 * and `pn_tt_gdn_checkpoint_need()`.
 *
 *   1. Qwen3.8 shared: every decode weight IS gdnc copy 0, no decode weight is allocated
 *      (5.00 MiB a layer = 30 MiB in every bank), and every window, state, gdnc copy and
 *      checkpoint twin keeps its bank. At every starting cursor.
 *   2. Not shared (Ornith, Qwen3.6, Qwen3.8 off): main's census — each layer its own weight,
 *      weight/window/state on consecutive round-robin banks, main's byte total.
 *   3. Fallback: sharing asked for, gdnc not armed -> own weights, same bytes.
 *   4. The gate: fit policy, strip, gdnc and the knob, each able to decline.
 *   5. Row 0 of the gdnc tile is the decode plane byte for byte (pn_tt_gdn_layout.h).
 *
 * Build: `make USE_TT=1 tests/test_tt_q38_conv_share` (Mac or Linux, no card).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_device_internal.h"
#include "../src/accel/tt/pn_tt_gdn_checkpoint.h"
#include "../src/accel/tt/pn_tt_gdn_conv_share.h"
#include "../src/accel/tt/pn_tt_gdn_layout.h"
#include "tt/ft_tt_buffer.h"

#define BANKS 8u
#define MIB (1024.0 * 1024.0)
#define TILE 4096u
#define MAX_LAYERS 64u
/* test_tt_q38_reserve.c's D_FREED_PER_BANK: 6 decode weights of 5.00 MiB a bank. */
#define Q38_FREED_PER_BANK (6ull * 5u * 1024u * 1024u)

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/* ================================================================
 * A host device: an allocator and a DRAM writer that discards
 * ================================================================ */

static int discard_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                         size_t bytes)
{
    (void)user, (void)x, (void)y, (void)address, (void)src, (void)bytes;
    return FT_TT_OK;
}

typedef struct {
    const char *name;
    pn_tt_gdn_shape shape;
} model_shape;

/* GGUF geometry: state, v-heads, k-heads, conv_dim, taps, SSM layers. */
static const model_shape Q38 = { "Qwen3.8-27B", { 128u, 48u, 16u, 10240u, 4u, 48u } };
static const model_shape ORNITH = { "Ornith-1.5-9B", { 128u, 32u, 16u, 8192u, 4u, 24u } };
static const model_shape Q36 = { "Qwen3.6-35B-A3B", { 128u, 32u, 16u, 8192u, 4u, 30u } };

/** What one placement left: per-bank use, every region, gdnc copy 0, the twins' need. */
typedef struct {
    int rc;
    uint8_t share;
    uint64_t used[BANKS];
    uint64_t total;
    pn_tt_gdn_region region[MAX_LAYERS];
    pn_tt_gdn_buffer constant[MAX_LAYERS];
    pn_tt_bank_need twins;
} placement;

static placement place(const pn_tt_gdn_shape *shape, bool share, bool gdnc_armed,
                       uint32_t cursor)
{
    pn_tt_device *d = calloc(1, sizeof(*d));
    ft_tt_grid_info grid;
    placement p;

    memset(&p, 0, sizeof(p));
    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.dram_bank_count = BANKS;
    if (!d || ft_tt_allocator_init(&d->allocator, NULL, &grid, 0u) != FT_TT_OK) {
        p.rc = -1;
        free(d);
        return p;
    }
    d->dram_bank_count = BANKS;
    d->runtime.bulk_dram.write = discard_write;
    d->strip_bank_cursor = cursor;
    d->gdn_shape = *shape;

    /* The arm sequence of pn_qwen35_tt_gdn_arm(), minus the residencies. */
    pn_tt_gdn_conv_share_request(d, share);
    p.rc = pn_tt_gdn_place_regions(d);
    d->gdn.running = true;
    if (p.rc == 0)
        p.rc = pn_tt_gdnc_place_regions(d);
    d->gdnc.running = gdnc_armed;
    if (p.rc == 0)
        p.rc = pn_tt_gdn_conv_share_bind(d);
    if (p.rc == 0) {
        p.share = d->gdn_conv_share;
        for (uint32_t b = 0; b < BANKS; b++) {
            p.used[b] = ft_tt_allocator_used_bytes(&d->allocator, FT_TT_MEM_SPACE_DRAM, b);
            p.total += p.used[b];
        }
        memcpy(p.region, d->gdn_region, shape->layer_count * sizeof(p.region[0]));
        memcpy(p.constant, d->gdnc_constant, shape->layer_count * sizeof(p.constant[0]));
        pn_tt_bank_need_init(&p.twins, BANKS);
        pn_tt_gdn_checkpoint_need(d, &p.twins);
    }
    /* Host tables only; the allocator lives in `d`. */
    free(d->gdn_region);
    free(d->gdnc_constant);
    free(d->gdnc_constant_replica);
    free(d);
    return p;
}

static uint64_t weight_bytes(const pn_tt_gdn_shape *s)
{
    return (uint64_t)(s->conv_dim / 32u) * s->conv_taps * TILE;
}

static bool same_buffer(const pn_tt_gdn_buffer *a, const pn_tt_gdn_buffer *b)
{
    return a->bank == b->bank && a->address == b->address;
}

/* ================================================================
 * (1) Qwen3.8, shared
 * ================================================================ */

static void check_q38_shared(void)
{
    const pn_tt_gdn_shape *s = &Q38.shape;
    bool aliased = true, banks_kept = true, freed_even = true, twins_kept = true;
    uint64_t freed_total = 0u;

    for (uint32_t cursor = 0; cursor < BANKS; cursor++) {
        const placement own = place(s, false, true, cursor);
        const placement shared = place(s, true, true, cursor);

        check(own.rc == 0 && shared.rc == 0, "Qwen3.8 places with and without sharing");
        check(own.share == PN_TT_GDN_CONV_OWN && shared.share == PN_TT_GDN_CONV_SHARED,
              "Qwen3.8: sharing engages only when asked");
        for (uint32_t l = 0; l < s->layer_count; l++) {
            aliased &= same_buffer(&shared.region[l].weight, &shared.constant[l]);
            banks_kept &= shared.region[l].window.bank == own.region[l].window.bank &&
                          shared.region[l].state.bank == own.region[l].state.bank &&
                          shared.constant[l].bank == own.constant[l].bank;
        }
        for (uint32_t b = 0; b < BANKS; b++) {
            freed_even &= own.used[b] - shared.used[b] == Q38_FREED_PER_BANK;
            twins_kept &= own.twins.bank[b] == shared.twins.bank[b];
        }
        freed_total = own.total - shared.total;
    }
    printf("  Qwen3.8 shared: decode weight = gdnc copy 0 %s; %.1f MiB freed (%.1f MiB a bank)\n",
           aliased ? "every layer" : "NOT every layer", (double)freed_total / MIB,
           (double)Q38_FREED_PER_BANK / MIB);
    check(aliased, "Qwen3.8 shared: every decode weight IS its gdnc copy 0 (not allocated)");
    check(freed_total == s->layer_count * weight_bytes(s),
          "Qwen3.8 shared: 48 x 5.00 MiB (240 MiB) less DRAM");
    check(freed_even, "Qwen3.8 shared: exactly 30 MiB freed in EVERY bank, every cursor");
    check(banks_kept, "Qwen3.8 shared: windows, states and gdnc copies keep their banks");
    check(twins_kept, "Qwen3.8 shared: the checkpoint twins need the same bytes per bank");
}

/* ================================================================
 * (2) Not shared: main's census
 * ================================================================ */

static void check_main_census(const model_shape *m)
{
    const pn_tt_gdn_shape *s = &m->shape;
    const placement p = place(s, false, true, 3u);
    const uint64_t state = (uint64_t)s->v_heads * (s->state_size / 32u) * (s->state_size / 32u) *
                           TILE;
    const uint64_t window = (uint64_t)(s->conv_dim / 32u) * (s->conv_taps - 1u) * TILE;
    bool own = true, round_robin = true;
    uint64_t regions = 0u;
    char what[160];

    for (uint32_t l = 0; l < s->layer_count; l++) {
        const uint32_t first = (3u + 3u * l) % BANKS;

        own &= !same_buffer(&p.region[l].weight, &p.constant[l]);
        round_robin &= p.region[l].weight.bank == first &&
                       p.region[l].window.bank == (first + 1u) % BANKS &&
                       p.region[l].state.bank == (first + 2u) % BANKS;
    }
    regions = s->layer_count * (state + window + weight_bytes(s));
    snprintf(what, sizeof(what), "%s (not shared): each layer allocates its own conv weight",
             m->name);
    check(p.rc == 0 && p.share == PN_TT_GDN_CONV_OWN && own, what);
    snprintf(what, sizeof(what), "%s (not shared): weight, window, state on consecutive "
             "round-robin banks (main)", m->name);
    check(round_robin, what);
    /* Main's regions plus the shared staging and the gdnc constants: at least the regions and
     * the 8 constant copies, i.e. nothing went missing. */
    snprintf(what, sizeof(what), "%s (not shared): DRAM holds main's regions and 8 gdnc copies",
             m->name);
    check(p.total >= regions + 8u * s->layer_count * weight_bytes(s), what);
    printf("  %s not shared: %.1f MiB of GDN+GDNC DRAM\n", m->name, (double)p.total / MIB);
}

/* ================================================================
 * (3) Fallback and (4) the gate
 * ================================================================ */

static void check_fallback(void)
{
    const placement own = place(&Q38.shape, false, false, 0u);
    const placement fell = place(&Q38.shape, true, false, 0u);
    bool own_weights = true;

    for (uint32_t l = 0; l < Q38.shape.layer_count; l++)
        own_weights &= !same_buffer(&fell.region[l].weight, &fell.constant[l]);
    check(fell.rc == 0 && fell.share == PN_TT_GDN_CONV_OWN && own_weights,
          "gdnc not armed: sharing falls back to the decode family's own weights");
    check(fell.total == own.total, "gdnc not armed: the fallback allocates main's bytes");
}

static void check_gate(void)
{
    static const char *const knobs[] = { PN_TT_GDN_CONV_SHARE_ENV, "PN_TT_GDN_CONV_STRIP",
                                         "PN_TT_PREFILL_GDNC" };

    for (size_t i = 0; i < 3u; i++)
        unsetenv(knobs[i]);
    check(pn_tt_gdn_conv_share_wanted(true), "the fit policy alone engages sharing");
    check(!pn_tt_gdn_conv_share_wanted(false), "no fit policy, no sharing (old models)");
    for (size_t i = 0; i < 3u; i++) {
        char what[96];

        setenv(knobs[i], "0", 1);
        snprintf(what, sizeof(what), "%s=0 declines sharing under the fit policy", knobs[i]);
        check(!pn_tt_gdn_conv_share_wanted(true), what);
        setenv(knobs[i], "1", 1);
        check(pn_tt_gdn_conv_share_wanted(true) && !pn_tt_gdn_conv_share_wanted(false),
              "=1 keeps the fit policy's answer");
        unsetenv(knobs[i]);
    }
}

/* ================================================================
 * (5) Row 0 of the gdnc tile is the decode plane
 * ================================================================ */

static void check_row0_identity(void)
{
    const uint32_t conv_dim = Q38.shape.conv_dim, taps = Q38.shape.conv_taps;
    const uint32_t cols = conv_dim / 32u;
    float *weight = malloc((size_t)conv_dim * taps * sizeof(float));
    float *scratch = malloc((size_t)conv_dim * sizeof(float));
    unsigned char *decode = malloc((size_t)cols * TILE);
    unsigned char gdnc[TILE];
    float plane[32u * 32u];
    uint32_t row0_off = 0u, replica_off = 0u, decode_pad = 0u;
    uint32_t seed = 12345u;

    if (!weight || !scratch || !decode) {
        check(false, "row-0 identity: allocation");
        return;
    }
    for (size_t i = 0; i < (size_t)conv_dim * taps; i++) {
        seed = seed * 1664525u + 1013904223u;
        weight[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 0.25f;
    }
    weight[5] = -0.0f; /* the sign of zero must survive both stagers */
    for (uint32_t tap = 0; tap < taps; tap++) {
        if (pn_tt_gdn_stage_plane(weight, tap, taps, conv_dim, scratch, decode) != 0) {
            check(false, "row-0 identity: the decode plane stages");
            break;
        }
        for (uint32_t col = 0; col < cols; col++) {
            const unsigned char *dt = decode + (size_t)col * TILE;

            if (pn_tt_gdnc_stage_weight_column(weight, taps, tap, col, false, plane, gdnc) != 0) {
                row0_off++;
                continue;
            }
            for (uint32_t chunk = 0; chunk < 2u; chunk++) {
                const uint32_t r0 = pn_tt_gdn_row_offset(0u, 0u, chunk);

                row0_off += memcmp(dt + r0, gdnc + r0, 64u) != 0;
                for (uint32_t row = 1; row < 32u; row++) {
                    const uint32_t rr = pn_tt_gdn_row_offset(0u, row, chunk);
                    static const unsigned char zeros[64];

                    replica_off += memcmp(gdnc + rr, gdnc + r0, 64u) != 0;
                    decode_pad += memcmp(dt + rr, zeros, 64u) != 0;
                }
            }
        }
    }
    printf("  row 0: %u of %u half-rows differ; gdnc rows 1..31 differing from row 0: %u; "
           "decode rows 1..31 nonzero: %u\n", row0_off, taps * cols * 2u, replica_off,
           decode_pad);
    check(row0_off == 0u, "row 0 of every gdnc tile is the decode plane byte for byte");
    check(replica_off == 0u, "the gdnc planes are row-replicated");
    check(decode_pad == 0u, "the decode plane's rows 1..31 are zero (the strip never reads them)");
    free(weight);
    free(scratch);
    free(decode);
}

int main(void)
{
    check_q38_shared();
    check_main_census(&ORNITH);
    check_main_census(&Q36);
    check_main_census(&Q38);
    check_fallback();
    check_gate();
    check_row0_identity();
    printf("test_tt_q38_conv_share: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("test_tt_q38_conv_share: needs USE_TT=1 (no card)\n");
    return 1;
}

#endif /* USE_TT */
