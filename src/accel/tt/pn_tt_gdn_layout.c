/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_layout.c
 * @brief Implementation of the GDN stager's address arithmetic. See pn_tt_gdn_layout.h.
 *
 * @status Pure; `tests/test_gdn_layout.c` proves it against literals derived from the
 *   protocol header and the CPU oracle rather than from these functions.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdn_layout.h"

#include <stddef.h>
#include <string.h>

#include "tt/ft_tt_tilize.h"

uint32_t pn_tt_gdn_head_tile(uint32_t head, uint32_t state)
{
    return head * (state / PN_TT_GDN_LAYOUT_TILE_DIM);
}

uint32_t pn_tt_gdn_row_offset(uint32_t tile, uint32_t row, uint32_t chunk)
{
    /* Four 16x16 faces in order, so the face holding `[row][chunk*16 .. chunk*16+15]` is
     * `(row/16)*2 + chunk`, and each face is `TILE_BYTES/4` — 1024 B in fp32. Inside the face
     * the row is `row % 16` rows of 16 fp32 down. */
    const uint32_t face = (row / 16u) * 2u + chunk;

    return tile * PN_TT_GDN_LAYOUT_TILE_BYTES + face * (PN_TT_GDN_LAYOUT_TILE_BYTES / 4u) +
           (row % 16u) * PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES;
}

uint32_t pn_tt_gdn_row0_offset(uint32_t tile, uint32_t chunk)
{
    /* Face 0 is elements [0,256) of the tile and face 1 is [256,512), so the two halves of
     * row 0 are `TILE_BYTES/4` apart — 1024 B in fp32. */
    return pn_tt_gdn_row_offset(tile, 0u, chunk);
}

int pn_tt_gdn_stage_plane(const float *source, uint32_t first, uint32_t stride,
                          uint32_t conv_dim, float *scratch, unsigned char *tiles_out)
{
    uint32_t channel;

    if (!source || !scratch || !tiles_out || conv_dim == 0u || stride == 0u ||
        (conv_dim % PN_TT_GDN_LAYOUT_TILE_DIM) != 0u)
        return -1;
    for (channel = 0; channel < conv_dim; channel++)
        scratch[channel] = source[(size_t)first + (size_t)channel * stride];
    if (ft_tt_tilize_strip(scratch, sizeof(*scratch), 1u, conv_dim,
                           (size_t)conv_dim * sizeof(*scratch), tiles_out) != FT_TT_TILIZE_OK)
        return -1;
    return 0;
}

int pn_tt_gdnc_stage_weight_column(const float *weight, uint32_t taps, uint32_t tap,
                                   uint32_t column, bool row0_only, float *plane,
                                   unsigned char *tile_out)
{
    const uint32_t dim = PN_TT_GDN_LAYOUT_TILE_DIM;
    uint32_t row;
    uint32_t lane;

    if (!weight || !plane || !tile_out || taps == 0u || tap >= taps)
        return -1;
    memset(plane, 0, (size_t)dim * dim * sizeof(*plane));
    /* `w[c][tap]` at EVERY row: the tap coefficient is constant across a chunk's positions and
     * the broadcast surface is BROADCAST_COL only — there is no row broadcast. */
    for (row = 0; row < (row0_only ? 1u : dim); row++)
        for (lane = 0; lane < dim; lane++)
            plane[row * dim + lane] = weight[((size_t)column * dim + lane) * taps + tap];
    if (ft_tt_tilize_strip(plane, sizeof(*plane), dim, dim, (size_t)dim * sizeof(*plane),
                           tile_out) != FT_TT_TILIZE_OK)
        return -1;
    return 0;
}

#endif /* USE_TT */
