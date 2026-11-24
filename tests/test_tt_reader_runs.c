/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_reader_runs.c
 * @brief The reader's tile runs (src/accel/tt/pn_tt_reader_runs.h): with `live` = 48 of a
 *        2-tile alpha/beta band, every live column's bytes are read and tile 1's right faces
 *        are not — a quarter less readback, the live values unchanged.
 *
 * Opens no device: emulates the bf16 face layout of a band and what pn_tt_reader.c's widening
 * loop does with the runs it read.
 *
 * Build: `make tests/test_tt_reader_runs` (any host).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_reader_runs.h"

#define TILE_BYTES 2048u
#define TILES 2u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok)
        g_failures++;
}

static uint32_t face_index(uint32_t row, uint32_t column)
{
    return ((row / 16u) * 2u + column / 16u) * 256u + (row % 16u) * 16u + column % 16u;
}

/** Read `rows` rows of a 2-tile band through the runs; widen the first `live` columns.
 *  @return bytes read; `mismatch` counts live elements that differ from the card's. */
static uint32_t read_band(const uint16_t *card, uint32_t rows, uint32_t live, uint32_t *mismatch)
{
    uint16_t stage[TILES * TILE_BYTES / 2u];
    uint32_t bytes = 0u;

    memset(stage, 0xEE, sizeof(stage));
    for (uint32_t tile = 0; tile < TILES; tile++) {
        uint32_t offset[2], length[2];
        const uint32_t runs =
            pn_tt_reader_tile_runs(tile, TILES, rows, live, TILE_BYTES, offset, length);

        for (uint32_t r = 0; r < runs; r++) {
            memcpy((uint8_t *)stage + tile * TILE_BYTES + offset[r],
                   (const uint8_t *)card + tile * TILE_BYTES + offset[r], length[r]);
            bytes += length[r];
        }
    }
    *mismatch = 0u;
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t col = 0; col < (live ? live : TILES * 32u); col++) {
            const uint32_t at = (col / 32u) * (TILE_BYTES / 2u) + face_index(row, col % 32u);

            *mismatch += stage[at] != card[at];
        }
    return bytes;
}

int main(void)
{
    uint16_t card[TILES * TILE_BYTES / 2u];
    char what[160];

    printf("test_tt_reader_runs: the alpha/beta rider (a)\n");
    for (uint32_t i = 0; i < TILES * TILE_BYTES / 2u; i++)
        card[i] = (uint16_t)(i * 2654435761u >> 16);
    {
        static const uint32_t rows_list[] = {32u, 17u, 16u, 7u, 1u};

        for (uint32_t k = 0; k < sizeof(rows_list) / sizeof(rows_list[0]); k++) {
            const uint32_t rows = rows_list[k];
            uint32_t bad_all, bad_live;
            const uint32_t all = read_band(card, rows, 0u, &bad_all);
            const uint32_t live = read_band(card, rows, 48u, &bad_live);

            snprintf(what, sizeof(what), "%u rows: every column right with all read (%u B); the "
                     "48 live right with %u B (%.0f%% of it)", rows, all, live,
                     100.0 * live / all);
            check(bad_all == 0u && bad_live == 0u && live * 4u == all * 3u, what);
        }
    }
    {
        uint32_t offset[2], length[2], bad;

        check(pn_tt_reader_tile_runs(0u, TILES, 32u, 48u, TILE_BYTES, offset, length) == 1u &&
                  length[0] == TILE_BYTES,
              "tile 0 (columns 0-31, all live) is read whole");
        check(read_band(card, 32u, 49u, &bad) == TILES * TILE_BYTES && bad == 0u,
              "49 live columns reach face 1 of tile 1: everything read");
        check(read_band(card, 32u, 40u, &bad) * 4u == TILES * TILE_BYTES * 3u && bad == 0u,
              "40 live columns: the same quarter saved");
    }
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
