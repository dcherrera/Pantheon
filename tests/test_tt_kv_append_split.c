/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_kv_append_split.c
 * @brief The prefill KV append split (pn_tt_kv_append_split.h): its descriptors place exactly
 *        what the shipped ONE banked descriptor places, head for head, and spread over the cores.
 *
 * Opens no device. A host `pn_tt_device` holds a real `ft_tt_allocator` (8 p150 banks), the
 * production KV bank groups (`pn_tt_kv_banks_configure()` / `_alloc()`) and a DRAM writer that
 * discards. Every descriptor is DECODED with the memory kernel's own protocol header
 * (foundry `resident_memory_protocol.h`, the reader's per-group source and the writer's
 * banked / unbanked destination arithmetic), so "same placement" is the card's reading of
 * the words, not the encoder's.
 *
 *   1. Placement: for K8V4, Bfp8 and bf16 caches, layers, aligned bands and a partial last
 *      band, the split pair's 2G descriptors place every head's rows at the same source tile,
 *      destination core, address, row range and format as the banked K and V descriptors.
 *   2. Shape: K's G descriptors first then V's, one head each, each carrying its own format.
 *   3. Spread: over 8 members every descriptor has its own core; over 4, 2 and 1 each member
 *      runs whole descriptors and every descriptor runs exactly once.
 *   4. Knobs: the member count (bf16 never splits, default 8, clamped) and the core list.
 *
 * Build: `make USE_TT=1 tests/test_tt_kv_append_split` (Mac or Linux, no card).
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
#include "../src/accel/tt/pn_tt_kv_append_split.h"
#include "../src/accel/tt/pn_tt_kv_banks.h"
#include "../src/accel/tt/pn_tt_kv_format.h"
#include "tt/ft_tt_buffer.h"
#include "tt/kernels/resident_memory_protocol.h"

#define BANKS 8u
#define LAYERS 16u
#define KV_HEADS 4u
#define HEAD_DIM 256u
#define CAPACITY 30720u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

static int discard_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                         size_t bytes)
{
    (void)user, (void)x, (void)y, (void)address, (void)src, (void)bytes;
    return FT_TT_OK;
}

/** A host device with production's KV bank groups for `k_format` / `v_format`. */
static pn_tt_device *host_device(uint32_t k_format, uint32_t v_format)
{
    pn_tt_device *d = calloc(1, sizeof(*d));
    ft_tt_grid_info grid;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.dram_bank_count = BANKS;
    if (!d || ft_tt_allocator_init(&d->allocator, NULL, &grid, 0u) != FT_TT_OK) {
        free(d);
        return NULL;
    }
    d->dram_bank_count = BANKS;
    for (uint32_t b = 0; b < BANKS; b++) {
        d->dram[b].x = (uint16_t)(1u + b);
        d->dram[b].y = (uint16_t)(20u + 2u * b);
    }
    d->runtime.bulk_dram.write = discard_write;
    d->strip_bank_cursor = 3u;
    d->kv_type.k_format = k_format;
    d->kv_type.v_format = v_format;
    d->attn_kv_heads = KV_HEADS;
    d->attn_dim_tiles = HEAD_DIM / 32u;
    d->memory.running = true; /* the request builder's own precondition */
    if (ft_tt_kv_cache_layout_init(KV_HEADS, HEAD_DIM, CAPACITY,
                                   pn_tt_kv_format_tile_bytes(k_format), &d->kv_layout) != FT_TT_OK ||
        ft_tt_kv_cache_layout_init(KV_HEADS, HEAD_DIM, CAPACITY,
                                   pn_tt_kv_format_tile_bytes(v_format), &d->kv_v_layout) !=
            FT_TT_OK ||
        pn_tt_kv_banks_configure(d) != 0 || !pn_tt_kv_banks_alloc(d, LAYERS)) {
        free(d->kv_region);
        free(d);
        return NULL;
    }
    return d;
}

static void free_device(pn_tt_device *d)
{
    if (d)
        free(d->kv_region);
    free(d);
}

/** One head's rows as the card places them: where the reader reads, where the writer writes. */
typedef struct {
    uint32_t src_x, src_y, src_base, src_row0;
    uint32_t dst_x, dst_y, dst_base, row_in_tile, rows, group_tiles, format;
    bool whole, fits;
} placement;

/** Decode group `group` of `w` the way resident_gather_reader.c and resident_place_writer.c
 *  do (source: base + group * stride; destination: banked or not, + slot * stride + tile row). */
static bool decode_group(const uint32_t *w, uint32_t group, uint32_t dst_tile_bytes,
                         placement *out)
{
    const uint32_t flags = w[kMemFlags];
    const uint32_t tile_row = w[kMemDstRow0] / (uint32_t)kMemTileRows;
    const uint32_t group_bytes = w[kMemGroupTiles] * dst_tile_bytes;
    uint32_t x = w[kMemDstX], y = w[kMemDstY], slot = group;

    if (group >= w[kMemGroupCount])
        return false;
    if ((flags & (uint32_t)kMemFlagDstBanked) != 0u &&
        !ftm_banked_destination(flags, w[kMemDstX], w[kMemDstY], group, &x, &y, &slot))
        return false;
    memset(out, 0, sizeof(*out));
    out->src_x = w[kMemSrcX];
    out->src_y = w[kMemSrcY];
    out->src_base = w[kMemSrcBase] + group * w[kMemSrcGroupStride];
    out->src_row0 = w[kMemSrcRow0];
    out->dst_x = x;
    out->dst_y = y;
    out->dst_base = w[kMemDstBase] + slot * w[kMemDstGroupStride] + tile_row * group_bytes;
    out->row_in_tile = w[kMemDstRow0] % (uint32_t)kMemTileRows;
    out->rows = w[kMemRowCount];
    out->group_tiles = w[kMemGroupTiles];
    out->format = (flags >> kMemKvFormatShift) & (uint32_t)kMemKvFormatMask;
    out->whole = (flags & (uint32_t)kMemFlagDstWholeTiles) != 0u;
    out->fits = ftm_destination_span_fits(w[kMemDstGroupStride], tile_row * group_bytes,
                                          group_bytes);
    return true;
}

static bool same_placement(const placement *a, const placement *b)
{
    return a->src_x == b->src_x && a->src_y == b->src_y && a->src_base == b->src_base &&
           a->src_row0 == b->src_row0 && a->dst_x == b->dst_x && a->dst_y == b->dst_y &&
           a->dst_base == b->dst_base && a->row_in_tile == b->row_in_tile &&
           a->rows == b->rows && a->group_tiles == b->group_tiles && a->format == b->format &&
           a->whole == b->whole && a->fits && b->fits;
}

static pn_tt_activation source_strip(uint32_t bank, uint32_t address)
{
    pn_tt_activation a;

    memset(&a, 0, sizeof(a));
    a.bank = bank;
    a.address = address;
    a.tile_count = KV_HEADS * (HEAD_DIM / 32u);
    a.tile_rows = a.active_rows = 1u;
    return a;
}

/** 1 + 2: the split pair against the shipped banked K and V descriptors, every head. */
static void check_placement(uint32_t k_format, uint32_t v_format, const char *name)
{
    static const uint32_t positions[] = { 0u, 32u, 2816u, 28224u, CAPACITY - 32u };
    static const uint32_t layers[] = { 0u, 7u, LAYERS - 1u };
    pn_tt_device *d = host_device(k_format, v_format);
    pn_tt_activation k = source_strip(2u, 0x01000000u), v = source_strip(5u, 0x01400000u);
    char what[192];

    snprintf(what, sizeof(what), "%s: host device with KV bank groups", name);
    check(d != NULL && d->kv_bank_groups == KV_HEADS && d->kv_append_banked, what);
    if (!d || d->kv_bank_groups != KV_HEADS || !d->kv_append_banked) {
        free_device(d);
        return;
    }
    for (uint32_t li = 0; li < sizeof(layers) / sizeof(layers[0]); li++) {
        for (uint32_t pi = 0; pi < sizeof(positions) / sizeof(positions[0]); pi++) {
            for (uint32_t rows = 32u; rows >= 29u; rows -= 3u) {
                uint32_t split[2u * PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS];
                const uint32_t layer = layers[li], position = positions[pi];
                const uint32_t count = pn_tt_kv_append_pair_words(d, layer, &k, &v, position,
                                                                  rows, split);
                bool all_same = count == 2u * KV_HEADS;

                for (uint32_t half = 0; half < 2u && all_same; half++) {
                    const bool values = half == 1u;
                    const uint32_t tile = pn_tt_kv_layout_of(d, values)->tile_bytes;
                    ft_tt_op_kv_append_request request;
                    uint32_t banked[PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS];
                    uint32_t n = 0u;

                    all_same = pn_tt_kv_append_chunk_request(d, layer, values, values ? &v : &k,
                                                             position, rows, &request) == 0 &&
                               pn_tt_kv_append_words(d, &request, layer, values, banked, &n) ==
                                   0 &&
                               n == 1u && banked[0][kMemGroupCount] == KV_HEADS;
                    for (uint32_t head = 0; head < KV_HEADS && all_same; head++) {
                        const uint32_t *w = split[half * KV_HEADS + head];
                        placement ref, got;

                        all_same = w[kMemGroupCount] == 1u &&
                                   (w[kMemFlags] & (uint32_t)kMemFlagDstBanked) == 0u &&
                                   decode_group(banked[0], head, tile, &ref) &&
                                   decode_group(w, 0u, tile, &got) &&
                                   same_placement(&ref, &got);
                    }
                }
                snprintf(what, sizeof(what), "%s: layer %u position %u rows %u — %u split "
                         "descriptors place every K and V head as the banked pair does", name,
                         layer, position, rows, count);
                check(all_same, what);
            }
        }
    }
    free_device(d);
}

/** 3: every descriptor runs exactly once, whole, and 8 members give each its own core. */
static void check_spread(void)
{
    static const uint32_t member_counts[] = { 8u, 4u, 2u, 1u, 3u };

    for (uint32_t c = 0; c < sizeof(member_counts) / sizeof(member_counts[0]); c++) {
        const uint32_t members = member_counts[c];
        uint32_t runs[PN_TT_KV_APPEND_MAX_MEMBERS] = { 0 };
        bool in_range = true;
        char what[128];

        for (uint32_t i = 0; i < 2u * KV_HEADS; i++) {
            const uint32_t m = pn_tt_kv_append_member_of(i, members);

            in_range = in_range && m < members;
            if (m < PN_TT_KV_APPEND_MAX_MEMBERS)
                runs[m]++;
        }
        snprintf(what, sizeof(what), "spread over %u: every descriptor on a member in range",
                 members);
        check(in_range, what);
        for (uint32_t m = 0; m < members; m++) {
            const uint32_t want = (2u * KV_HEADS) / members + (m < (2u * KV_HEADS) % members);

            snprintf(what, sizeof(what), "spread over %u: member %u runs %u descriptors (want %u)",
                     members, m, runs[m], want);
            check(runs[m] == want, what);
        }
    }
}

/** 4: the member count and the core list. */
static void check_knobs(void)
{
    static const uint32_t defaults[] = { 67u, 68u, 69u, 70u, 121u, 126u, 127u };

    check(pn_tt_kv_append_members_wanted(false, NULL, 8u) == 1u, "bf16 cache: 1 core");
    check(pn_tt_kv_append_members_wanted(false, "8", 8u) == 1u, "bf16 cache ignores the knob");
    check(pn_tt_kv_append_members_wanted(true, NULL, 8u) == 8u, "block float: 8 by default");
    check(pn_tt_kv_append_members_wanted(true, "", 8u) == 8u, "empty knob: the default");
    check(pn_tt_kv_append_members_wanted(true, "4", 8u) == 4u, "knob 4: 4");
    check(pn_tt_kv_append_members_wanted(true, "1", 8u) == 1u, "knob 1: off");
    check(pn_tt_kv_append_members_wanted(true, "0", 8u) == 8u, "knob 0 is not a count: default");
    check(pn_tt_kv_append_members_wanted(true, "4x", 8u) == 8u, "malformed knob: default");
    check(pn_tt_kv_append_members_wanted(true, "20", 8u) == 8u, "knob above the maximum: clamped");
    for (uint32_t i = 0; i < sizeof(defaults) / sizeof(defaults[0]); i++) {
        char what[96];

        snprintf(what, sizeof(what), "default list core %u is %u", i, defaults[i]);
        check(pn_tt_kv_append_list_core(NULL, i) == defaults[i], what);
    }
    check(pn_tt_kv_append_list_core(NULL, 7u) == UINT32_MAX, "default list ends after 7");
    check(pn_tt_kv_append_list_core("", 0u) == 67u, "empty list: the default");
    check(pn_tt_kv_append_list_core("90,91", 1u) == 91u, "custom list position 1");
    check(pn_tt_kv_append_list_core("90,91", 2u) == UINT32_MAX, "custom list ends");
    check(pn_tt_kv_append_list_core("90,x", 1u) == UINT32_MAX, "malformed entry: refused");
    check(pn_tt_kv_append_list_core("90;91", 0u) == UINT32_MAX, "bad separator: refused");
}

int main(void)
{
    check_placement(PN_TT_KV_FORMAT_BFP8, PN_TT_KV_FORMAT_BFP4, "k8v4");
    check_placement(PN_TT_KV_FORMAT_BFP8, PN_TT_KV_FORMAT_BFP8, "bfp8");
    check_placement(PN_TT_KV_FORMAT_BF16, PN_TT_KV_FORMAT_BF16, "bf16");
    check_spread();
    check_knobs();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("test_tt_kv_append_split: needs USE_TT=1 — skipped\n");
    return 0;
}

#endif /* USE_TT */
