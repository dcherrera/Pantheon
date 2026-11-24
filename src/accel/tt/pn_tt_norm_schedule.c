/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_schedule.c
 * @brief The P3 norm schedules. See pn_tt_norm_schedule.h.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_norm_schedule.h"

#include <string.h>

/** The pre-P3 schedule: one block a row, nothing else. */
static void pn_tt_norm_schedule_off(uint32_t tile_cols, pn_tt_norm_schedule *out)
{
    memset(out, 0, sizeof(*out));
    out->prefetch_tiles = tile_cols;
}

void pn_tt_norm_schedule_model(bool fast, uint32_t tile_cols, uint32_t depth, uint32_t chunk,
                               pn_tt_norm_schedule *out)
{
    uint32_t run = chunk ? chunk : PN_TT_NORM_CHUNK_DEFAULT;

    pn_tt_norm_schedule_off(tile_cols, out);
    if (!fast || tile_cols == 0u)
        return;
    if (run > PN_TT_NORM_CHUNK_MAX)
        run = PN_TT_NORM_CHUNK_MAX;
    if (run > tile_cols)
        run = tile_cols;
    out->prefetch_tiles = depth ? depth : PN_TT_NORM_DEPTH_DEFAULT;
    out->chain_chunk = run;
    out->stage_pages = run;
    out->stream_input = true;
    out->half_read = true;
}

void pn_tt_norm_schedule_head(bool fast, uint32_t tile_cols, pn_tt_norm_schedule *out)
{
    pn_tt_norm_schedule_off(tile_cols, out);
    if (!fast || tile_cols == 0u)
        return;
    /* One block a row: a head is a handful of tiles, and the reader's run-ahead is across
     * heads (input_rows), not within one. */
    out->chain_chunk = tile_cols > PN_TT_NORM_CHUNK_MAX ? PN_TT_NORM_CHUNK_MAX : tile_cols;
    out->stage_pages = out->chain_chunk;
    out->input_rows = PN_TT_HEAD_NORM_INPUT_ROWS;
    out->stream_input = true;
    out->half_read = true;
}

uint32_t pn_tt_norm_schedule_chunk_for(const pn_tt_norm_schedule *schedule,
                                       uint32_t slice_tiles)
{
    if (!schedule || schedule->chain_chunk == 0u)
        return 0u;
    return slice_tiles < schedule->chain_chunk ? slice_tiles : schedule->chain_chunk;
}
