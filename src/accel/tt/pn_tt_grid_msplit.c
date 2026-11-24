/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_msplit.c
 * @brief Grid M split arithmetic. See pn_tt_grid_msplit.h.
 * @status Default on (`PN_TT_GRID_MSPLIT=0` off); stash chunks (`PN_TT_GRID_STASH_CHUNKS=0` off).
 * @issues None
 * @todo   None
 */
#include "pn_tt_grid_msplit.h"

#include <stdlib.h>

bool pn_tt_grid_msplit_applies(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                               uint32_t mt, uint32_t pieces, uint32_t max_columns)
{
    if (max_columns == 0u || pieces < 2u || columns == 0u || columns > max_columns)
        return false;
    if (stash_tiles == 0u || (uint64_t)columns * kt > stash_tiles)
        return false;
    return mt >= pieces;
}

void pn_tt_grid_msplit_rows(uint32_t mt, uint32_t pieces, uint32_t row_block, uint32_t piece,
                            uint32_t *out_first, uint32_t *out_rows)
{
    const uint32_t block = row_block ? row_block : 1u;
    const uint32_t blocks = (mt + block - 1u) / block;
    const uint32_t base = blocks / pieces, extra = blocks % pieces;
    const uint32_t first_block = piece * base + (piece < extra ? piece : extra);
    const uint32_t count = base + (piece < extra ? 1u : 0u);
    uint32_t first = first_block * block, end = (first_block + count) * block;

    if (end > mt)
        end = mt;
    if (first > mt)
        first = mt;
    *out_first = first;
    *out_rows = end - first;
}

uint32_t pn_tt_grid_msplit_max_columns(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *on = getenv("PN_TT_GRID_MSPLIT");
        const char *cols = getenv("PN_TT_GRID_MSPLIT_COLS");

        cached = (on && *on && atoi(on) == 0) ? 0 : (cols && *cols ? atoi(cols) : 8);
        if (cached < 0)
            cached = 0;
    }
    return (uint32_t)cached;
}

uint32_t pn_tt_grid_stash_chunks(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                                 bool enabled)
{
    uint32_t widest;

    if (!enabled || columns == 0u || kt == 0u || stash_tiles == 0u ||
        (uint64_t)columns * kt <= stash_tiles)
        return 1u;
    widest = stash_tiles / kt;
    if (widest == 0u)
        return 1u;
    return (columns + widest - 1u) / widest;
}

void pn_tt_grid_stash_chunk_columns(uint32_t columns, uint32_t chunks, uint32_t chunk,
                                    uint32_t *out_first, uint32_t *out_cols)
{
    const uint32_t n = chunks ? chunks : 1u;
    const uint32_t base = columns / n, extra = columns % n;

    *out_first = chunk * base + (chunk < extra ? chunk : extra);
    *out_cols = chunk < n ? base + (chunk < extra ? 1u : 0u) : 0u;
}

bool pn_tt_grid_stash_chunks_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_GRID_STASH_CHUNKS");

        cached = (env && *env && atoi(env) == 0) ? 0 : 1;
    }
    return cached != 0;
}

uint32_t pn_tt_grid_unsplit_pieces(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                                   uint32_t mt, uint32_t row_block, uint32_t grid_cores,
                                   uint32_t max_columns)
{
    const uint32_t block = row_block ? row_block : 1u;
    uint32_t blocks;

    if (max_columns == 0u || columns == 0u || columns > max_columns || grid_cores < 2u)
        return 0u;
    if (stash_tiles == 0u || (uint64_t)columns * kt > stash_tiles)
        return 0u;
    blocks = (mt + block - 1u) / block;
    if (blocks < 2u)
        return 0u;
    return blocks < grid_cores ? blocks : grid_cores;
}

uint32_t pn_tt_grid_unsplit_max_columns(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_GRID_UNSPLIT");

        cached = env && *env ? atoi(env) : 8;
        if (cached < 0)
            cached = 0;
    }
    return (uint32_t)cached;
}
