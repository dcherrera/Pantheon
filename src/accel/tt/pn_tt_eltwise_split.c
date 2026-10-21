/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_eltwise_split.c
 * @brief Implementation of the elementwise split arithmetic — member count and tile range.
 *
 * No device, no allocation, no I/O, and deliberately so: the header states why an address
 * is the one thing here that fails silently, and this file is what a test with no card can
 * hold to that.
 *
 * @status 0 checks failed off-hardware, `tests/test_tt_eltwise_split.c`.
 * @issues None
 * @todo None
 */

#include "pn_tt_eltwise_split.h"

uint32_t pn_tt_eltwise_split_member_limit(uint32_t requested, const uint32_t *tile_counts,
                                          uint32_t count, uint32_t max_members)
{
    uint32_t members;

    if (!tile_counts || count == 0u || requested == 0u || max_members == 0u)
        return 1u;
    if (max_members > PN_TT_ELTWISE_SPLIT_MAX_MEMBERS)
        max_members = PN_TT_ELTWISE_SPLIT_MAX_MEMBERS;
    members = requested < max_members ? requested : max_members;
    for (; members > 1u; members--) {
        uint32_t index;
        bool divides = true;

        for (index = 0; index < count && divides; index++) {
            /* A zero width is a caller defect and cannot be split at all. Refusing the whole
             * split rather than skipping the row keeps "every width this residency serves"
             * literally true of the answer. */
            if (tile_counts[index] == 0u || (tile_counts[index] % members) != 0u)
                divides = false;
        }
        if (divides)
            return members;
    }
    return 1u;
}

bool pn_tt_eltwise_split_slice(uint32_t tile_count, uint32_t members, uint32_t member,
                               uint32_t tile_bytes, pn_tt_eltwise_slice *out_slice)
{
    uint32_t per_member;

    if (!out_slice || members == 0u || members > PN_TT_ELTWISE_SPLIT_MAX_MEMBERS ||
        member >= members || tile_count == 0u || tile_bytes == 0u)
        return false;
    if ((tile_count % members) != 0u)
        return false;
    per_member = tile_count / members;
    out_slice->first_tile = member * per_member;
    out_slice->tile_count = per_member;
    out_slice->byte_offset = out_slice->first_tile * tile_bytes;
    return true;
}

bool pn_tt_eltwise_split_slice_blocks(uint32_t tile_count, uint32_t members, uint32_t member,
                                      uint32_t block_tiles, uint32_t tile_bytes,
                                      pn_tt_eltwise_slice *out_slice)
{
    uint32_t blocks, base, extra, first_block, own;

    if (!out_slice || members == 0u || members > PN_TT_ELTWISE_SPLIT_MAX_MEMBERS ||
        member >= members || tile_count == 0u || tile_bytes == 0u || block_tiles == 0u ||
        (tile_count % block_tiles) != 0u)
        return false;
    blocks = tile_count / block_tiles;
    base = blocks / members;
    extra = blocks % members;
    first_block = member * base + (member < extra ? member : extra);
    own = base + (member < extra ? 1u : 0u);
    out_slice->first_tile = first_block * block_tiles;
    out_slice->tile_count = own * block_tiles;
    out_slice->byte_offset = out_slice->first_tile * tile_bytes;
    return true;
}
