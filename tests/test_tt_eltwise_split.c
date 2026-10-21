/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_eltwise_split.c — the arithmetic of splitting one elementwise descriptor across N
 * Tensix, scored with NO CARD and NO USE_TT.
 *
 * WHAT IT PROVES, and why each part is here rather than left to a card arm:
 *
 *   1. THE SLICES TILE THE DESCRIPTOR EXACTLY AND OVERLAP NOWHERE. Every member of every
 *      legal split is walked and the union of their tile ranges is checked to be [0, W) with
 *      no gap and no repeat, at every width Ornith carries and at every member count. An
 *      overlap here is TWO CORES WRITING ONE DRAM BYTE — the one failure in this whole lever
 *      that produces a plausible wrong answer with nothing refusing anywhere.
 *
 *   2. THE BYTE OFFSET IS THE TILE OFFSET TIMES THE PAGE SIZE, and it is asked about the
 *      OPERAND's page size rather than the family's, because the render pair proves the two
 *      differ (fp32 in, bf16 out) and a split that assumed 2048 would be right by luck.
 *
 *   3. A RAGGED SPLIT IS REFUSED, NOT ROUNDED. The matmul group hands its remainder out one
 *      column at a time; this family cannot, because a slice off the `2 * depth` grid walks
 *      the CB write pointer up L1 with no error. So a member count that does not divide is
 *      refused at the top (`member_limit` never returns it) and refused again at the bottom
 *      (`slice` returns false), and BOTH are checked — a single guard is a single edit away
 *      from being the only one.
 *
 *   4. THE MEMBER LIMIT IS THE LARGEST DIVISOR AT OR BELOW WHAT WAS ASKED FOR, not the GCD
 *      and not the request. Ornith's three widths are 128 / 256 / 384, whose common divisors
 *      run 1, 2, 4, 8, 16, 32, 64, 128 — so asking for 3, 6 or 12 must come back 2, 4 and 8
 *      rather than 1, and asking for 12 must not come back 12.
 *
 *   5. ONE MEMBER IS THE PRE-SPLIT PATH. `members = 1` yields the whole descriptor at offset
 *      zero, which is the claim that makes `PN_TT_ELTWISE_CORES=1` a control rather than a
 *      configuration.
 *
 * RED ARMS. Each of these is a one-line mutation of the module that this file must go red
 * on, and each was run:
 *   - `first_tile = member * per_member` -> `member * tile_count` .......... 16 failures
 *   - the `(tile_count % members) != 0u` refusal deleted ................... 2 failures
 *   - `byte_offset = first_tile * tile_bytes` -> `first_tile` ............. 16 failures
 *   - `member_limit` walking UP instead of down from the request ........... 8 failures
 */

#include "../src/accel/tt/pn_tt_eltwise_split.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_checked = 0;

#define CHECK(cond, msg) do { \
    g_checked++; \
    if (!(cond)) { \
        g_failed++; \
        printf("  FAIL %s\n", (msg)); \
    } \
} while (0)

/** Ornith's three elementwise widths: the residual adds and the GDN gate at 128, the
 *  attention gate at 256, the FFN SwiGLU at 384. Every figure in this file is about these. */
static const uint32_t k_ornith_widths[] = { 128u, 256u, 384u };
#define ORNITH_WIDTHS 3u

/** The widest descriptor there is, so a cover map can be a fixed array. */
#define MAX_TILES 4096u

/**
 * Walk every member of an `members`-way split of `tile_count` and check the ranges TILE the
 * descriptor: every tile covered exactly once, in order, starting at zero.
 */
static void check_slices_tile_exactly(uint32_t tile_count, uint32_t members,
                                      uint32_t tile_bytes)
{
    unsigned char covered[MAX_TILES];
    uint32_t expected_first = 0u;
    uint32_t member, tile;
    bool all_ok = true;

    memset(covered, 0, sizeof(covered));
    for (member = 0; member < members; member++) {
        pn_tt_eltwise_slice slice;

        if (!pn_tt_eltwise_split_slice(tile_count, members, member, tile_bytes, &slice)) {
            all_ok = false;
            break;
        }
        if (slice.first_tile != expected_first ||
            slice.byte_offset != slice.first_tile * tile_bytes ||
            slice.tile_count != tile_count / members)
            all_ok = false;
        for (tile = slice.first_tile; tile < slice.first_tile + slice.tile_count; tile++) {
            if (tile >= MAX_TILES || covered[tile] != 0u)
                all_ok = false;
            else
                covered[tile] = 1u;
        }
        expected_first += slice.tile_count;
    }
    for (tile = 0; tile < tile_count && tile < MAX_TILES; tile++) {
        if (covered[tile] != 1u)
            all_ok = false;
    }
    CHECK(all_ok && expected_first == tile_count,
          "every member's range is disjoint and the set covers the descriptor exactly");
}

/** 1 + 2: the slices tile the descriptor, at every width and every legal member count. */
static void test_the_slices_tile_the_descriptor(void)
{
    static const uint32_t members[] = { 1u, 2u, 4u, 8u, 16u };
    unsigned width, index;

    printf("the slices tile the descriptor and overlap nowhere\n");
    for (width = 0; width < ORNITH_WIDTHS; width++) {
        for (index = 0; index < sizeof(members) / sizeof(members[0]); index++)
            check_slices_tile_exactly(k_ornith_widths[width], members[index], 2048u);
    }
    /* THE PAGE SIZE IS THE OPERAND'S, NOT THE FAMILY'S. The strip bridge runs the same
     * eltwise sources at MIXED page sizes — fp32 in, bf16 out — so a split that assumed 2048
     * would be right only by luck. */
    check_slices_tile_exactly(128u, 4u, 4096u);
    check_slices_tile_exactly(384u, 8u, 4096u);
    {
        pn_tt_eltwise_slice slice;

        CHECK(pn_tt_eltwise_split_slice(384u, 8u, 3u, 4096u, &slice) &&
                  slice.first_tile == 144u && slice.byte_offset == 144u * 4096u,
              "the byte offset is the tile offset times the OPERAND's page size");
        CHECK(pn_tt_eltwise_split_slice(384u, 8u, 3u, 2048u, &slice) &&
                  slice.byte_offset == 144u * 2048u,
              "and the same tile offset at a bf16 page is half the bytes");
    }
}

/** W1a: the block-ragged wide split tiles every prefill width exactly, every slice whole
 *  blocks, members past the last block empty at the end. */
static void test_the_block_split_tiles_prefill_widths(void)
{
    static const uint32_t mts[] = { 1u, 2u, 13u, 24u, 51u, 64u };
    static const uint32_t wides[] = { 128u, 384u };
    uint32_t i, w, members, member;

    printf("the block-ragged wide split tiles every prefill width\n");
    for (i = 0; i < sizeof(mts) / sizeof(mts[0]); i++)
        for (w = 0; w < 2u; w++)
            for (members = 1u; members <= PN_TT_ELTWISE_SPLIT_MAX_MEMBERS; members++) {
                const uint32_t tiles = mts[i] * wides[w];
                uint32_t next = 0u, most = 0u, least = 0xffffffffu;
                bool ok = true;

                for (member = 0; member < members; member++) {
                    pn_tt_eltwise_slice slice;

                    if (!pn_tt_eltwise_split_slice_blocks(tiles, members, member, 64u, 2048u,
                                                          &slice) ||
                        slice.first_tile != next || (slice.tile_count % 64u) != 0u ||
                        slice.byte_offset != slice.first_tile * 2048u)
                        ok = false;
                    next = slice.first_tile + slice.tile_count;
                    if (slice.tile_count > most) most = slice.tile_count;
                    if (slice.tile_count < least) least = slice.tile_count;
                }
                CHECK(ok && next == tiles && most - least <= 64u,
                      "contiguous whole-block slices covering the descriptor, within one block");
            }
    {
        pn_tt_eltwise_slice slice;

        CHECK(!pn_tt_eltwise_split_slice_blocks(19584u + 32u, 15u, 0u, 64u, 2048u, &slice),
              "a width that is not whole blocks is refused");
        CHECK(pn_tt_eltwise_split_slice_blocks(19584u, 15u, 14u, 64u, 2048u, &slice) &&
                  slice.first_tile == 19584u - 1280u && slice.tile_count == 1280u,
              "51 x 384 over 15: 306 blocks, the last member owns 20 of them at the end");
        CHECK(pn_tt_eltwise_split_slice_blocks(128u, 4u, 3u, 64u, 2048u, &slice) &&
                  slice.tile_count == 0u && slice.first_tile == 128u,
              "a member past the last block is empty at the end");
        CHECK(!pn_tt_eltwise_split_slice_blocks(128u, 4u, 4u, 64u, 2048u, &slice),
              "a member index equal to the count is refused");
    }
}

/** 5: one member is the whole descriptor at offset zero — the control arm's claim. */
static void test_one_member_is_the_pre_split_path(void)
{
    unsigned width;

    printf("one member is the pre-split path\n");
    for (width = 0; width < ORNITH_WIDTHS; width++) {
        pn_tt_eltwise_slice slice;

        CHECK(pn_tt_eltwise_split_slice(k_ornith_widths[width], 1u, 0u, 2048u, &slice) &&
                  slice.first_tile == 0u && slice.byte_offset == 0u &&
                  slice.tile_count == k_ornith_widths[width],
              "members = 1 is the whole descriptor at offset zero");
    }
}

/** 3: a ragged split is refused at the slice, and so is every out-of-range argument. */
static void test_a_ragged_split_is_refused(void)
{
    pn_tt_eltwise_slice slice;

    printf("a ragged split is refused rather than rounded\n");
    CHECK(!pn_tt_eltwise_split_slice(128u, 3u, 0u, 2048u, &slice),
          "128 tiles over 3 members is refused, not rounded to 42");
    CHECK(!pn_tt_eltwise_split_slice(384u, 5u, 0u, 2048u, &slice),
          "384 tiles over 5 members is refused");
    CHECK(!pn_tt_eltwise_split_slice(128u, 4u, 4u, 2048u, &slice),
          "a member index equal to the member count is refused");
    CHECK(!pn_tt_eltwise_split_slice(128u, 4u, 9u, 2048u, &slice),
          "a member index past the member count is refused");
    CHECK(!pn_tt_eltwise_split_slice(0u, 4u, 0u, 2048u, &slice),
          "a zero tile count is refused");
    CHECK(!pn_tt_eltwise_split_slice(128u, 0u, 0u, 2048u, &slice),
          "a zero member count is refused");
    CHECK(!pn_tt_eltwise_split_slice(128u, PN_TT_ELTWISE_SPLIT_MAX_MEMBERS + 1u, 0u, 2048u,
                                     &slice),
          "a member count past the module's bound is refused");
    CHECK(!pn_tt_eltwise_split_slice(128u, 4u, 0u, 0u, &slice),
          "a zero page size is refused — it would make every offset zero");
    CHECK(!pn_tt_eltwise_split_slice(128u, 4u, 0u, 2048u, NULL), "a NULL output is refused");
}

/** 4: the limit is the largest divisor at or below the request. */
static void test_the_member_limit_is_the_largest_divisor(void)
{
    printf("the member limit is the largest divisor at or below the request\n");
    CHECK(pn_tt_eltwise_split_member_limit(8u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 8u,
          "8 divides 128, 256 and 384, so 8 is 8");
    CHECK(pn_tt_eltwise_split_member_limit(16u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 16u,
          "and so does 16");
    /* THE ANSWER IS NOT THE GCD. gcd(128,256,384) is 128, and asking for 3 must come back 2
     * rather than 1 or 128 — the largest DIVISOR AT OR BELOW the request. */
    CHECK(pn_tt_eltwise_split_member_limit(3u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 2u,
          "3 comes back 2, not 1 and not the gcd");
    CHECK(pn_tt_eltwise_split_member_limit(6u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 4u,
          "6 comes back 4");
    CHECK(pn_tt_eltwise_split_member_limit(12u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 8u,
          "12 comes back 8 — 12 does not divide 128");
    CHECK(pn_tt_eltwise_split_member_limit(5u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 4u,
          "5 comes back 4");
    CHECK(pn_tt_eltwise_split_member_limit(7u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 4u,
          "7 comes back 4");

    /* The clamps. */
    CHECK(pn_tt_eltwise_split_member_limit(64u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 16u,
          "the caller's bound clamps before the widths are consulted");
    CHECK(pn_tt_eltwise_split_member_limit(64u, k_ornith_widths, ORNITH_WIDTHS, 999u) ==
              PN_TT_ELTWISE_SPLIT_MAX_MEMBERS,
          "and the module's own bound clamps the caller's");
    CHECK(pn_tt_eltwise_split_member_limit(1u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 1u,
          "1 is 1");
    CHECK(pn_tt_eltwise_split_member_limit(0u, k_ornith_widths, ORNITH_WIDTHS, 16u) == 1u,
          "0 is 1 — there is no zero-member split");

    /* A width that shares nothing collapses the split for ALL of them, which is the whole
     * "one depth, one placement, every member" rule. */
    {
        static const uint32_t odd[] = { 128u, 256u, 384u, 129u };

        CHECK(pn_tt_eltwise_split_member_limit(8u, odd, 4u, 16u) == 1u,
              "one indivisible width collapses the split for every width");
    }
    {
        static const uint32_t zero_width[] = { 128u, 0u };

        CHECK(pn_tt_eltwise_split_member_limit(8u, zero_width, 2u, 16u) == 1u,
              "a zero width refuses the split rather than being skipped");
    }
    CHECK(pn_tt_eltwise_split_member_limit(8u, NULL, 3u, 16u) == 1u, "a NULL table is 1");
    CHECK(pn_tt_eltwise_split_member_limit(8u, k_ornith_widths, 0u, 16u) == 1u,
          "an empty table is 1");
}

/**
 * 3, the other half: what the limit returns is ALWAYS sliceable.
 *
 * The two guards are deliberately redundant — the limit never returns a ragged count and the
 * slice refuses one anyway — and a redundancy nobody checks is a redundancy one edit removes.
 * This walks every request from 1 to 40 and holds the pair to each other.
 */
static void test_the_two_guards_agree(void)
{
    uint32_t requested;
    bool all_ok = true;

    printf("what the member limit returns is always sliceable\n");
    for (requested = 1u; requested <= 40u; requested++) {
        uint32_t members = pn_tt_eltwise_split_member_limit(requested, k_ornith_widths,
                                                            ORNITH_WIDTHS, 16u);
        unsigned width;

        if (members == 0u || members > PN_TT_ELTWISE_SPLIT_MAX_MEMBERS ||
            members > (requested < 16u ? requested : 16u))
            all_ok = false;
        for (width = 0; width < ORNITH_WIDTHS; width++) {
            pn_tt_eltwise_slice slice;

            if (!pn_tt_eltwise_split_slice(k_ornith_widths[width], members, members - 1u,
                                           2048u, &slice))
                all_ok = false;
        }
    }
    CHECK(all_ok, "every count the limit returns slices every width, for requests 1..40");
}

int main(void)
{
    printf("test_tt_eltwise_split — the elementwise core split's arithmetic, off-hardware\n\n");
    test_the_slices_tile_the_descriptor();
    test_one_member_is_the_pre_split_path();
    test_a_ragged_split_is_refused();
    test_the_member_limit_is_the_largest_divisor();
    test_the_two_guards_agree();
    test_the_block_split_tiles_prefill_widths();
    printf("\n%d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
