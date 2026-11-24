/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_conv_split.c
 * @brief The GDN member count and per-member slices. See pn_tt_gdn_conv_split.h.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_gdn_conv_split.h"

#include <stdlib.h>

bool pn_tt_gdn_conv_split_fits(uint32_t cols, uint32_t members, uint32_t granule,
                               uint32_t col_max)
{
    const uint32_t per = members ? cols / members : 0u;

    /* The last member starts at `cols - per`, the furthest first column of the split. */
    return per > 0u && granule > 0u && cols % members == 0u && per % granule == 0u &&
           per <= col_max && cols - per <= col_max;
}

uint32_t pn_tt_gdn_conv_member_count(uint32_t cols, uint32_t armed, uint32_t asked,
                                     uint32_t granule, uint32_t col_max)
{
    uint32_t members = armed;

    if (asked >= 1u && asked < armed && cols % asked == 0u && granule > 0u &&
        (cols / asked) % granule == 0u)
        members = asked;
    while (members > 1u && !pn_tt_gdn_conv_split_fits(cols, members, granule, col_max))
        members--;
    return members ? members : 1u;
}

int pn_tt_gdn_slice(const ft_tt_op_gdn_request *request, uint32_t members, uint32_t index,
                    uint32_t depth, pn_tt_gdn_slice_red red, ft_tt_op_gdn_request *out)
{
    uint32_t per;

    if (!request || !out || members == 0u || index >= members)
        return -1;
    *out = *request;
    if (request->phase != FT_TT_OP_GDN_PHASE_CONV) {
        per = request->v_heads / members;
        out->head_first = index * per;
        out->head_count = per;
        return 0;
    }
    /* Absolute columns: the kernels index `tap * conv_tile_cols + column`, so a slice is only
     * the loop bounds of every pass. */
    per = (request->conv_dim / FT_TT_OP_TILE_DIM) / members;
    out->col_first = index * per;
    out->col_count = per;
    switch (red) {
    case PN_TT_GDN_SLICE_RED_COL_DUP:
        /* Every member convolves member 0's columns: the rest are never written. */
        out->col_first = 0u;
        break;
    case PN_TT_GDN_SLICE_RED_COL_DROP:
        /* The last member starts one block early, overlapping its neighbour: the stream's last
         * block is never written. Every slice keeps the exact arm's size, so the card runs the
         * same descriptor shapes (a short slice is refused on the card instead of computed). */
        if (index + 1u == members) {
            if (depth == 0u || per <= depth)
                return -1;
            out->col_first -= depth;
        }
        break;
    case PN_TT_GDN_SLICE_RED_COL_WRAP:
        /* What the narrow bytes alone would address: a slice past 255 lands on another
         * member's columns and its own are never written. */
        out->col_first &= PN_TT_GDN_COL_BYTE_MAX;
        out->col_count &= PN_TT_GDN_COL_BYTE_MAX;
        break;
    case PN_TT_GDN_SLICE_EXACT:
        break;
    }
    return 0;
}

bool pn_tt_gdn_conv_wide_enabled(const char *value)
{
    char *end = NULL;
    unsigned long parsed;

    if (!value || !*value)
        return true;
    parsed = strtoul(value, &end, 0);
    return end == value || *end != '\0' || parsed != 0ul;
}
