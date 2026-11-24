/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_reader_runs.h
 * @brief Which bytes of a bf16 strip tile the reader thread reads (pn_tt_reader.c): the rows'
 *        faces, and with a `live` column count only the faces those columns occupy — the
 *        alpha/beta rider (a): Qwen3.8's 48 heads leave tile 1's right faces (columns 48-63)
 *        unread, a quarter of the readback.
 *
 * Pure: tests/test_tt_reader_runs.c.
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifndef PN_TT_READER_RUNS_H
#define PN_TT_READER_RUNS_H

#include <stdbool.h>
#include <stdint.h>

/**
 * The runs of tile `tile` (of `tiles`) a job of `rows` rows reads, as offsets/lengths inside the
 * tile of `tile_bytes`. Faces 0/1 hold rows 0-15 (columns 0-15 / 16-31), faces 2/3 rows 16-31;
 * a tile whose live columns stop at 16 (`live` != 0) needs only faces 0 and 2. Rows <= 16 read
 * 64 B-rounded runs of the top faces (pn_tt_activation_download_rows()'s shape).
 * @return the run count (1 or 2)
 */
static inline uint32_t pn_tt_reader_tile_runs(uint32_t tile, uint32_t tiles, uint32_t rows,
                                              uint32_t live, uint32_t tile_bytes,
                                              uint32_t offset[2], uint32_t length[2])
{
    const bool left_only = live != 0u && tile + 1u == tiles && live <= tile * 32u + 16u;
    const uint32_t face = tile_bytes / 4u;
    uint32_t n = 0u;

    if (rows <= 16u) {
        const uint32_t run = (rows * 32u + 63u) & ~63u;

        offset[n] = 0u;
        length[n++] = run;
        if (!left_only) {
            offset[n] = face;
            length[n++] = run;
        }
        return n;
    }
    if (!left_only) {
        offset[0] = 0u;
        length[0] = tile_bytes;
        return 1u;
    }
    offset[n] = 0u;
    length[n++] = face;
    offset[n] = 2u * face;
    length[n++] = face;
    return n;
}

#endif /* PN_TT_READER_RUNS_H */
