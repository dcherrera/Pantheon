/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_conv_split.h
 * @brief How many GDN members a PHASE CONV (or DELTA) descriptor splits over, and each
 *        member's slice. Pure: no device, so the host model (tests/test_tt_gdn_conv_split.c)
 *        drives the same code the card path runs.
 *
 * CONV divides by TILE COLUMN of the conv_dim stream (each member convolves its columns in
 * every tap plane, at the full plane stride); DELTA divides by v-head. A CONV slice rides in
 * the descriptor as `col_first` / `col_count`. Their low bytes are bytes 1 and 2 of the
 * CONV-taps word; foundry's wide encoding (`kGdnByteColFirstHigh`) carries the high bytes in
 * the tile-column word. WIDE (the default) lets Qwen3.8's 320 columns split 8 or 16 ways;
 * NARROW (`PN_TT_GDN_CONV_WIDE=0`) keeps every slice under 256 columns, which caps them at 4
 * members of 80.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GDN_CONV_SPLIT_H
#define PN_TT_GDN_CONV_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

#include "tt/ft_tt_op_gdn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The furthest column a NARROW slice may start at or span: the taps word's bytes alone. */
#define PN_TT_GDN_COL_BYTE_MAX 255u

/** `PN_TT_GDN_CONV_WIDE` (default 1): CONV slices may address every column foundry encodes;
 *  `=0` is the narrow cap. */
#define PN_TT_GDN_CONV_WIDE_ENV "PN_TT_GDN_CONV_WIDE"

/**
 * Is the wide CONV encoding on, for a `PN_TT_GDN_CONV_WIDE` value? DEFAULT ON (measured
 * 2026-11-17: 16-way k1 +14.7%, plain +8.0%, md5-exact); only a value parsing to 0 turns it off.
 *
 * @param value  the environment value, or NULL when unset
 * @return       true for wide (every column), false for the narrow cap
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_gdn_conv_wide_enabled(const char *value);

/** A deliberate mutation of the CONV slices (the `PN_TT_RED` names); each must DIFFER. */
typedef enum pn_tt_gdn_slice_red {
    PN_TT_GDN_SLICE_EXACT = 0,
    PN_TT_GDN_SLICE_RED_COL_DUP,  /**< `gdn_conv_col_dup`: every member takes member 0's */
    PN_TT_GDN_SLICE_RED_COL_DROP, /**< `gdn_conv_col_drop`: the stream's last block unwritten */
    PN_TT_GDN_SLICE_RED_COL_WRAP, /**< `gdn_conv_col_wrap`: first/count lose their high byte */
} pn_tt_gdn_slice_red;

/**
 * Does splitting `cols` tile columns over `members` give slices the family can run and
 * encode: equal, each a multiple of the CONV slice granule, every first column and count at
 * most `col_max`?
 *
 * The granule, not the block depth: an 18-column slice (a depth multiple) ran its CONV on
 * the card and broke the DELTA after it (foundry FT_TT_OP_GDN_CONV_SLICE_GRANULE).
 *
 * @param cols     tile columns of the stream (conv_dim / 32)
 * @param members  the member count
 * @param granule  `ft_tt_op_gdn_conv_slice_granule()` (4 at Qwen3.8's placement)
 * @param col_max  PN_TT_GDN_COL_BYTE_MAX (narrow) or FT_TT_OP_GDN_COL_MAX (wide)
 * @return         true when it fits
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_gdn_conv_split_fits(uint32_t cols, uint32_t members, uint32_t granule,
                               uint32_t col_max);

/**
 * How many members PHASE CONV runs over: `asked` when it is a valid cap below `armed`, else
 * `armed`, then stepped down to the largest count whose slices fit. 1 = the whole-core CONV.
 *
 * @param cols     tile columns of the stream
 * @param armed    GDN members this run armed
 * @param asked    `PN_TT_GDN_CONV_CORES` (defaults to `armed`)
 * @param granule  the CONV slice granule (`ft_tt_op_gdn_conv_slice_granule()`)
 * @param col_max  PN_TT_GDN_COL_BYTE_MAX or FT_TT_OP_GDN_COL_MAX
 * @return         the member count, at least 1
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_gdn_conv_member_count(uint32_t cols, uint32_t armed, uint32_t asked,
                                     uint32_t granule, uint32_t col_max);

/**
 * Member `index`'s slice of `request` split `members` ways: tile columns for CONV, v-heads
 * for DELTA. The caller has already checked that the count divides.
 *
 * @param request  the whole descriptor's request
 * @param members  the member count, at least 1
 * @param index    this member, below `members`
 * @param depth    the CONV block depth (the block a drop red skips)
 * @param red      a CONV slice mutation, or PN_TT_GDN_SLICE_EXACT (ignored for DELTA)
 * @param out      receives the sliced request
 * @return         0, or -1 on a NULL argument, an index out of range, or a drop red with
 *                 no block to drop
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_gdn_slice(const ft_tt_op_gdn_request *request, uint32_t members, uint32_t index,
                    uint32_t depth, pn_tt_gdn_slice_red red, ft_tt_op_gdn_request *out);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_GDN_CONV_SPLIT_H */
