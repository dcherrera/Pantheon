/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_grid_msplit.c — the grid M split's row partition, with NO CARD and NO USE_TT.
 *
 * A piece's rows become A and C ADDRESSES on the card, so a partition that drops or repeats a
 * row is a smoothly wrong projection nothing reports. Checked: for every Mt 1..96, pieces
 * 2..4 and row blocks 1..4 the pieces cover [0, Mt) exactly once, in order, each start on a
 * row-block boundary; and the eligibility rule (stash, width, Mt >= pieces). The STASH CHUNKS:
 * for every piece width 1..40 and Kt in {128, 160, 192, 544} the chunks cover the piece once,
 * in order, each within the stash, widths within one of each other.
 * The UNSPLIT rows (O.0): a narrow one-shard weight's pieces cover Mt over up to every grid core,
 * each piece at least one row block; the rule refuses wide, stashless and one-block shapes.
 * Build with -DPN_TT_GRID_MSPLIT_RED (rows), -DPN_TT_GRID_CHUNK_RED (chunks) or
 * -DPN_TT_GRID_UNSPLIT_RED (unsplit pieces) to see them fail.
 */
#include "../src/accel/tt/pn_tt_grid_msplit.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed;

static void check(int ok, const char *what)
{
    if (!ok) {
        g_failed++;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static void check_partition(uint32_t mt, uint32_t pieces, uint32_t block)
{
    uint32_t next = 0u;
    char what[96];

    for (uint32_t p = 0; p < pieces; p++) {
        uint32_t first = 0u, rows = 0u;

        pn_tt_grid_msplit_rows(mt, pieces, block, p, &first, &rows);
#ifdef PN_TT_GRID_MSPLIT_RED
        if (p == 1u && rows > 0u)
            first++;
#endif
        snprintf(what, sizeof(what), "mt %u pieces %u block %u piece %u", mt, pieces, block, p);
        if (rows == 0u)
            continue;
        check(first == next, what);
        check(first % block == 0u, what);
        next = first + rows;
    }
    snprintf(what, sizeof(what), "mt %u pieces %u block %u covers", mt, pieces, block);
    check(next == mt, what);
}

static void check_chunks(uint32_t columns, uint32_t kt, uint32_t stash)
{
    const uint32_t chunks = pn_tt_grid_stash_chunks(columns, kt, stash, true);
    uint32_t next = 0u, widest = 0u, narrowest = UINT32_MAX;
    char what[96];

    snprintf(what, sizeof(what), "cols %u kt %u stash %u", columns, kt, stash);
    for (uint32_t c = 0; c < chunks; c++) {
        uint32_t first = 0u, cols = 0u;

        pn_tt_grid_stash_chunk_columns(columns, chunks, c, &first, &cols);
#ifdef PN_TT_GRID_CHUNK_RED
        if (c == 1u)
            first++;
#endif
        check(first == next && cols > 0u, what);
        if (chunks > 1u)
            check((uint64_t)cols * kt <= stash, what);
        widest = cols > widest ? cols : widest;
        narrowest = cols < narrowest ? cols : narrowest;
        next = first + cols;
    }
    check(next == columns, what);
    check(widest - narrowest <= 1u, what);
}

/* The unsplit rule's pieces, dealt by pn_tt_grid_msplit_rows, cover [0, Mt) once and none is
 * empty (an empty piece would be a descriptor with Mt 0). */
static void check_unsplit(uint32_t mt, uint32_t block, uint32_t cores)
{
    uint32_t pieces = pn_tt_grid_unsplit_pieces(2u, 160u, 1152u, mt, block, cores, 8u);
    uint32_t next = 0u;
    char what[96];

#ifdef PN_TT_GRID_UNSPLIT_RED
    if (pieces > 1u)
        pieces++;
#endif
    snprintf(what, sizeof(what), "unsplit mt %u block %u cores %u", mt, block, cores);
    if ((mt + block - 1u) / block < 2u) {
        check(pieces == 0u, what);
        return;
    }
    check(pieces >= 2u && pieces <= cores, what);
    for (uint32_t p = 0; p < pieces; p++) {
        uint32_t first = 0u, rows = 0u;

        pn_tt_grid_msplit_rows(mt, pieces, block, p, &first, &rows);
        check(rows > 0u && first == next && first % block == 0u, what);
        next = first + rows;
    }
    check(next == mt, what);
}

int main(void)
{
    static const uint32_t kts[] = { 128u, 160u, 192u, 544u };

    for (uint32_t i = 0; i < 4u; i++)
        for (uint32_t columns = 1u; columns <= 40u; columns++)
            check_chunks(columns, kts[i], 1152u);
    check(pn_tt_grid_stash_chunks(12u, 160u, 1152u, true) == 2u, "Q3.8 gate/up piece: 2 chunks");
    check(pn_tt_grid_stash_chunks(4u, 544u, 1152u, true) == 2u, "Q3.8 down piece: 2 chunks");
    check(pn_tt_grid_stash_chunks(7u, 160u, 1152u, true) == 1u, "fits: one");
    check(pn_tt_grid_stash_chunks(12u, 160u, 1152u, false) == 1u, "off: one");
    check(pn_tt_grid_stash_chunks(3u, 2048u, 1152u, true) == 1u, "no column fits: one (B streams)");
    check(pn_tt_grid_stash_chunks(12u, 160u, 0u, true) == 1u, "no stash: one");
    for (uint32_t mt = 1u; mt <= 96u; mt++)
        for (uint32_t pieces = 2u; pieces <= 4u; pieces++)
            for (uint32_t block = 1u; block <= 4u; block++)
                check_partition(mt, pieces, block);
    check(pn_tt_grid_msplit_applies(8u, 128u, 1152u, 64u, 3u, 8u), "z splits");
    check(pn_tt_grid_msplit_applies(2u, 128u, 1152u, 64u, 3u, 8u), "k/v split");
    check(!pn_tt_grid_msplit_applies(16u, 128u, 1152u, 64u, 3u, 8u), "qkv (16 wide) stays");
    check(!pn_tt_grid_msplit_applies(8u, 384u, 1152u, 64u, 3u, 8u), "ffn_down (past stash)");
    check(!pn_tt_grid_msplit_applies(8u, 128u, 0u, 64u, 3u, 8u), "no stash, no split");
    check(!pn_tt_grid_msplit_applies(8u, 128u, 1152u, 2u, 3u, 8u), "Mt < pieces");
    check(!pn_tt_grid_msplit_applies(8u, 128u, 1152u, 64u, 3u, 0u), "off");
    for (uint32_t mt = 1u; mt <= 96u; mt++)
        for (uint32_t block = 1u; block <= 4u; block++) {
            check_unsplit(mt, block, 48u);
            check_unsplit(mt, block, 3u);
        }
    check(pn_tt_grid_unsplit_pieces(2u, 160u, 1152u, 13u, 1u, 48u, 8u) == 13u,
          "Q3.8 alpha/beta super-band: 13 cores");
    check(pn_tt_grid_unsplit_pieces(2u, 160u, 1152u, 64u, 1u, 48u, 8u) == 48u,
          "Q3.8 alpha/beta pass: 48 cores");
    check(pn_tt_grid_unsplit_pieces(2u, 160u, 1152u, 1u, 1u, 48u, 8u) == 0u, "one row: none");
    check(pn_tt_grid_unsplit_pieces(16u, 160u, 1152u, 64u, 1u, 48u, 8u) == 0u, "wide: none");
    check(pn_tt_grid_unsplit_pieces(2u, 160u, 0u, 64u, 1u, 48u, 8u) == 0u, "no stash: none");
    check(pn_tt_grid_unsplit_pieces(2u, 160u, 1152u, 64u, 1u, 48u, 0u) == 0u, "off: none");
    printf("test_tt_grid_msplit: %d failed\n", g_failed);
    return g_failed ? 1 : 0;
}
