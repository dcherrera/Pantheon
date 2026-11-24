/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_l2.c
 * @brief L2, no card:
 *   1. the M2 pull lands on the chain's LAST member-0 descriptor only, aimed at the last split;
 *   2. late mirrors take only banks with room plus the reserve, zero every copy, move nothing
 *      allocated before them, and leave the strip untouched when no bank has room;
 *   3. the published replica map fans a write anywhere inside a decode out-projection input
 *      out to every mirror at the same offset (the kernel's own lookup,
 *      ftr_resolve_destinations), skips a tall strip in the slot, and the RED arm
 *      (`PN_TT_RED=decode_outproj_mirror_stale`, run as a child) fans out nothing.
 * Needs USE_TT=1, NOT a card.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_device_internal.h"
#include "../src/accel/tt/pn_tt_late_mirrors.h"
#include "../src/accel/tt/pn_tt_matmul_pull.h"
#include "../src/accel/tt/pn_tt_replica_map.h"
#include "tt/ft_tt_buffer.h"
#include "tt/kernels/resident_output_replicas.h"

#define BANKS 8u
#define MIB (1024ull * 1024ull)
#define GATED_TILES 192u /* Qwen3.8 ssm_out Kt: 48 v-heads x 128 / 32 */
#define GATED_ROWS 2u
#define GATED_BYTES ((uint64_t)GATED_TILES * GATED_ROWS * 2048u)

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
 * 1. Pull placement
 * ================================================================ */

/** One chain as the issue path sees it: `split[i]` per projection, sequences from `seq0`.
 *  Returns how many member-0 descriptors carried a pull; `*a3`/`*a4` the last one's words. */
static uint32_t pull_chain_run(const bool *split, uint32_t n, uint32_t seq0, bool defer,
                               uint32_t *a3, uint32_t *a4)
{
    pn_tt_pull_chain chain = { 0u, 0u };
    uint32_t pulled = 0u, i;

    *a3 = *a4 = 0u;
    for (i = 0; i < n; i++) {
        pn_tt_pull_words words;

        pn_tt_pull_chain_note(&chain, split[i], 16u, seq0 + i);
        words = pn_tt_pull_words_for(&chain, !defer || i + 1u == n, false, false);
        if (words.a3 != 0u)
            pulled++;
        if (i + 1u == n) {
            *a3 = words.a3;
            *a4 = words.a4;
        }
    }
    return pulled;
}

static void test_pull_placement(void)
{
    const bool in_proj[4] = { true, true, true, true };
    const bool split_solo[2] = { true, false };
    const bool solo[2] = { false, false };
    const bool split_solo_split_solo[4] = { true, false, true, false };
    uint32_t a3, a4;
    pn_tt_pull_words w;
    pn_tt_pull_chain c = { 16u, 7u };

    check(pull_chain_run(in_proj, 4u, 100u, true, &a3, &a4) == 1u,
          "deferred in-proj chain of 4: one pull");
    check(a3 == 16u && a4 == 103u, "the pull is the last projection's, 16 members");
    check(pull_chain_run(in_proj, 4u, 100u, false, &a3, &a4) == 4u,
          "not deferred: every split projection pulls (the shipped behaviour)");
    check(pull_chain_run(split_solo, 2u, 10u, true, &a3, &a4) == 1u && a3 == 16u && a4 == 10u,
          "split then solo: the solo carries the split's pull (v2)");
    check(pull_chain_run(solo, 2u, 10u, true, &a3, &a4) == 0u && a3 == 0u,
          "a solo-only chain pulls nothing");
    check(pull_chain_run(split_solo_split_solo, 4u, 50u, true, &a3, &a4) == 1u && a4 == 52u,
          "split, solo, split, solo: one pull aimed at the second split");
    w = pn_tt_pull_words_for(&c, true, true, true);
    check(w.a3 == (16u | PN_TT_PULL_RED_SKIP | PN_TT_PULL_SERIAL) && w.a4 == 7u,
          "RED skip and serial sweep are bits 31 and 30 over the member count");
    w = pn_tt_pull_words_for(&c, false, true, true);
    check(w.a3 == 0u && w.a4 == 0u, "a descriptor that does not end its chain pulls nothing");
}

/* ================================================================
 * Fake device: a host allocator, DRAM zeroing captured, L1 per core stored
 * ================================================================ */

#define CORES 16u
static uint64_t g_zeroed[BANKS]; /* bytes zeroed per bank, through bulk_dram */
static uint32_t g_l1[CORES][FT_TT_REPLICA_TABLE_WORDS];
static uint16_t g_core_x[CORES], g_core_y[CORES];

static int core_slot(uint16_t x, uint16_t y)
{
    for (uint32_t i = 0; i < CORES; i++)
        if (g_core_x[i] == x && g_core_y[i] == y)
            return (int)i;
    return -1;
}

static int fake_bulk_write(void *user, uint16_t x, uint16_t y, uint64_t address,
                           const void *src, size_t bytes)
{
    const pn_tt_device *d = user;

    (void)address, (void)src;
    for (uint32_t b = 0; b < BANKS; b++)
        if (d->dram[b].x == x && d->dram[b].y == y)
            g_zeroed[b] += bytes;
    return FT_TT_OK;
}

static int fake_noc_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                          size_t bytes)
{
    const int slot = core_slot(x, y);

    (void)user;
    if (slot < 0 || address != FT_TT_REPLICA_TABLE_ADDRESS || bytes > sizeof(g_l1[0]))
        return FT_TT_OK;
    memcpy(g_l1[slot], src, bytes);
    return FT_TT_OK;
}

static int fake_noc_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                         size_t bytes)
{
    const int slot = core_slot(x, y);

    (void)user;
    if (slot < 0 || address != FT_TT_REPLICA_TABLE_ADDRESS || bytes > sizeof(g_l1[0]))
        return FT_TT_ERROR;
    memcpy(dst, g_l1[slot], bytes);
    return FT_TT_OK;
}

static pn_tt_device *fake_device(void)
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
        d->dram[b].x = (uint16_t)(30u + b);
        d->dram[b].y = 1u;
    }
    d->runtime.bulk_dram.write = fake_bulk_write;
    d->runtime.bulk_dram.user = d;
    d->runtime.noc.write = fake_noc_write;
    d->runtime.noc.read = fake_noc_read;
    memset(g_zeroed, 0, sizeof(g_zeroed));
    return d;
}

/** Leave bank `bank` with exactly `left` free bytes. */
static void fill_bank(pn_tt_device *d, uint32_t bank, uint64_t left)
{
    uint64_t free_bytes[BANKS];
    ft_tt_buffer buffer;

    (void)pn_tt_device_dram_free_banks(d, free_bytes, BANKS);
    if (free_bytes[bank] > left)
        (void)ft_tt_buffer_alloc(&d->allocator, FT_TT_MEM_SPACE_DRAM, bank,
                                 (size_t)(free_bytes[bank] - left), &buffer);
}

static pn_tt_activation gated_strip(pn_tt_device *d, uint32_t bank)
{
    pn_tt_activation s;
    ft_tt_buffer buffer;

    memset(&s, 0, sizeof(s));
    (void)ft_tt_buffer_alloc(&d->allocator, FT_TT_MEM_SPACE_DRAM, bank, (size_t)GATED_BYTES,
                             &buffer);
    s.bank = bank;
    s.address = (uint32_t)buffer.address;
    s.tile_count = GATED_TILES;
    s.tile_rows = GATED_ROWS;
    s.active_rows = 1u;
    s.replica_count = 1u;
    s.replica_bank[0] = bank;
    s.replica_address[0] = s.address;
    return s;
}

/* ================================================================
 * 2. Late mirrors
 * ================================================================ */

static void test_late_mirrors(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_activation s, t;
    uint64_t before[BANKS], after[BANKS];
    bool distinct = true, skipped_full = true, kept_reserve = true, zeroed = true;
    int n;

    if (!d) {
        check(false, "fake device");
        return;
    }
    s = gated_strip(d, 2u);
    fill_bank(d, 4u, 3u * MIB);  /* the served MTP load's fullest bank: 3 MiB */
    fill_bank(d, 6u, 17u * MIB); /* room for the copy + 16 MiB reserve, barely */
    (void)pn_tt_device_dram_free_banks(d, before, BANKS);
    n = pn_tt_activation_mirror_late(d, &s, 8u, 16u * MIB);
    (void)pn_tt_device_dram_free_banks(d, after, BANKS);
    check(n == 7, "8 wanted, bank 4 has no room: 7 copies");
    check(s.replica_count == 7u && s.replica_bank[0] == 2u, "replica 0 stays the strip");
    for (uint32_t i = 0; i < s.replica_count; i++)
        for (uint32_t j = i + 1u; j < s.replica_count; j++)
            if (s.replica_bank[i] == s.replica_bank[j])
                distinct = false;
    for (uint32_t i = 0; i < s.replica_count; i++)
        if (s.replica_bank[i] == 4u)
            skipped_full = false;
    for (uint32_t b = 0; b < BANKS; b++)
        if (after[b] != before[b] && after[b] < 16u * MIB)
            kept_reserve = false;
    for (uint32_t i = 0; i < s.replica_count; i++)
        if (g_zeroed[s.replica_bank[i]] < GATED_BYTES)
            zeroed = false;
    check(distinct, "every copy in its own bank");
    check(skipped_full, "the 3 MiB bank is skipped");
    check(kept_reserve, "every used bank keeps the reserve");
    check(zeroed, "the strip and every copy zeroed");
    check(before[4] == after[4], "the skipped bank is untouched");
    check(pn_tt_activation_mirror_late(d, &s, 8u, 0u) == -1, "an already-mirrored strip refused");

    /* No room anywhere: the strip is left exactly as it was. */
    for (uint32_t b = 0; b < BANKS; b++)
        fill_bank(d, b, 1u * MIB);
    t = gated_strip(d, 1u);
    n = pn_tt_activation_mirror_late(d, &t, 8u, 16u * MIB);
    check(n == 1 && t.replica_count == 1u, "no bank with room: replica count stays 1");
    free(d);
}

/* ================================================================
 * 3. The published map fans the decode out-projection input out
 * ================================================================ */

/** How many destinations a write at `offset` into `s` resolves to on core slot `core`, and
 *  whether each lands at the same offset from its own copy. */
static uint32_t fan_out(uint32_t core, const pn_tt_device *d, const pn_tt_activation *s,
                        uint32_t offset, bool *same_offset)
{
    uint32_t x[FT_TT_REPLICA_MAX_DESTINATIONS], y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t n = ftr_resolve_destinations(g_l1[core], d->dram[s->bank].x,
                                                d->dram[s->bank].y, s->address + offset, x, y,
                                                base);

    *same_offset = true;
    for (uint32_t i = 0; i < n && i < s->replica_count; i++)
        if (x[i] != d->dram[s->replica_bank[i]].x ||
            base[i] != s->replica_address[i] + offset)
            *same_offset = false;
    return n;
}

static int test_published_map(bool red)
{
    pn_tt_device *d = fake_device();
    pn_tt_activation normed, gated, tall;
    const pn_tt_activation *slots[1];
    const pn_tt_activation *mirrored[1];
    const uint32_t last = (uint32_t)(GATED_BYTES - 2048u);
    bool same = false;
    uint32_t n;

    if (!d) {
        check(false, "fake device");
        return 1;
    }
    for (uint32_t i = 0; i < CORES; i++) {
        g_core_x[i] = (uint16_t)(1u + i);
        g_core_y[i] = 2u;
    }
    memset(g_l1, 0, sizeof(g_l1));
    d->reduce.running = true;
    d->reduce.core_x = g_core_x[0], d->reduce.core_y = g_core_y[0];
    d->eltwise.running = true;
    d->eltwise.core_x = g_core_x[1], d->eltwise.core_y = g_core_y[1];

    normed = gated_strip(d, 0u);
    normed.tile_count = 160u; /* hidden 5120 */
    gated = gated_strip(d, 3u);
    if (pn_tt_activation_mirror_late(d, &gated, 8u, 0u) != 8) {
        check(false, "8 copies of the gated strip");
        free(d);
        return 1;
    }
    slots[0] = &gated;
    mirrored[0] = &normed; /* unmirrored here: adds no key */
    pn_tt_replica_map_set_decode_ranged(slots, 1u);
    if (pn_tt_publish_replica_map(d, mirrored, 1u, NULL) != 0) {
        check(false, "the map publishes and reads back");
        free(d);
        return 1;
    }
    if (red) {
        n = fan_out(1u, d, &gated, 0u, &same);
        check(n == 1u, "RED decode_outproj_mirror_stale: the eltwise writer fans out nothing");
        free(d);
        return 0;
    }
    n = fan_out(1u, d, &gated, 0u, &same);
    check(n == 8u && same, "eltwise: a write at the strip's base reaches all 8 copies");
    n = fan_out(1u, d, &gated, GATED_TILES * 2048u + 48u * 2048u, &same);
    check(n == 8u && same, "eltwise: band 1, member 1's slice, same offset in every copy");
    n = fan_out(1u, d, &gated, last, &same);
    check(n == 8u && same, "the strip's last tile is inside the range");
    n = fan_out(1u, d, &gated, (uint32_t)GATED_BYTES, &same);
    check(n == 1u, "one tile past the strip: not fanned out");
    n = fan_out(0u, d, &gated, 0u, &same);
    check(n == 8u && same, "the reduce core's table carries it too");

    /* The slot holding a TALL strip (a batched pass swapped it in) adds no decode record. */
    tall = gated;
    tall.tile_rows = 64u;
    slots[0] = &tall;
    pn_tt_replica_map_set_decode_ranged(slots, 1u);
    (void)pn_tt_publish_replica_map(d, mirrored, 1u, NULL);
    n = fan_out(1u, d, &gated, 0u, &same);
    check(n == 1u, "a tall strip in the slot is not keyed as a decode range");
    pn_tt_replica_map_set_decode_ranged(NULL, 0u);
    free(d);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--red-child") == 0) {
        test_published_map(true);
        printf("test_tt_q38_l2 (red child): %d checks, %d failed\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    }
    test_pull_placement();
    test_late_mirrors();
    test_published_map(false);
    {
        char command[1024];
        int rc;

        snprintf(command, sizeof(command), "PN_TT_RED=decode_outproj_mirror_stale %s --red-child",
                 argv[0]);
        rc = system(command);
        check(rc == 0, "the RED child (its own process: PN_TT_RED is read once)");
    }
    printf("test_tt_q38_l2: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else
int main(void)
{
    printf("test_tt_q38_l2: needs USE_TT=1\n");
    return 0;
}
#endif
