/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_weight_view.c
 * @brief The column view of a resident weight (pn_tt_weight_view.h): every view column reads
 *        exactly the bytes, door and bank its source column does.
 *
 * Opens no device. The source is Qwen3.8-27B's `output.weight` as
 * pn_tt_matmul_group_place_weight_columns() places it: 7,760 Bfp8 column tiles, Kt 160, over
 * 16 members of 485 in banks (cursor + m) % 8. The view is the MTP draft head's vocabulary
 * (pn_qwen35_tt_mtp_arm): ids < 65,536 (2,048 tiles) and the 9-tile special tail.
 *   1. EXACT: for every view column, the member that owns it reads `source column`'s address,
 *      door and bank, and writes output column = view column; widths are positive, running sums.
 *   2. SPREAD: 16 pieces over the 6 source panels the subset touches.
 *   3. REFUSALS: a run off the end, a repeated column, fewer members than segments, no members,
 *      an empty source.
 *   4. AN UNSHARDED source (one panel) splits over the members too.
 *
 * Build: `make tests/test_tt_weight_view` (any host).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef USE_TT
#define USE_TT 1
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_weight_view.h"

#define Q38_VOCAB_TILES 7760u
#define Q38_EMBD_TILES 160u
#define BFP8_TILE 1088u
#define MEMBERS 16u
#define BANKS 8u
#define DRAFT_TILES 2048u
#define TAIL_FIRST (248044u / 32u)

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/** The placed source: member m's panel at a distinct address in bank (3 + m) % 8, door (m, b). */
static void source_of(pn_tt_weight *w, uint32_t members)
{
    uint32_t running = 0u;

    memset(w, 0, sizeof(*w));
    w->tile_rows = Q38_EMBD_TILES;
    w->tile_cols = Q38_VOCAB_TILES;
    w->tile_bytes = BFP8_TILE;
    w->in_features = Q38_EMBD_TILES * 32u;
    w->out_features = Q38_VOCAB_TILES * 32u;
    for (uint32_t m = 0; m < members; m++) {
        w->shard_tile_cols[m] = Q38_VOCAB_TILES / members + (m < Q38_VOCAB_TILES % members);
        w->shard_column_first[m] = running;
        w->shard_address[m] = 0x10000000u + m * 0x01000000u;
        w->shard_bank[m] = (3u + m) % BANKS;
        w->shard_x[m] = (uint16_t)(100u + m);
        w->shard_y[m] = (uint16_t)(200u + w->shard_bank[m]);
        running += w->shard_tile_cols[m];
    }
    w->shard_count = members;
    w->bank = w->shard_bank[0];
    w->address = w->shard_address[0];
}

/** Where `source` keeps column `c`: address, door, bank. */
static void source_column(const pn_tt_weight *w, uint32_t c, uint32_t *address, uint16_t *x,
                          uint16_t *y, uint32_t *bank)
{
    for (uint32_t m = 0; m < w->shard_count; m++)
        if (c >= w->shard_column_first[m] && c < w->shard_column_first[m] + w->shard_tile_cols[m]) {
            *address = w->shard_address[m] + (c - w->shard_column_first[m]) * w->tile_rows *
                                                 w->tile_bytes;
            *x = w->shard_x[m];
            *y = w->shard_y[m];
            *bank = w->shard_bank[m];
            return;
        }
    *address = 0u;
}

/** Does every view column read its source column? Pieces positive, running sums, total. */
static bool view_exact(const pn_tt_weight *src, const pn_tt_weight *v,
                       const pn_tt_column_run *runs, uint32_t run_count)
{
    uint32_t view_column = 0u, running = 0u;

    if (v->tile_rows != src->tile_rows || v->tile_bytes != src->tile_bytes ||
        v->out_features != v->tile_cols * 32u || v->shard_count == 0u)
        return false;
    for (uint32_t p = 0; p < v->shard_count; p++) {
        if (v->shard_tile_cols[p] == 0u || v->shard_column_first[p] != running)
            return false;
        running += v->shard_tile_cols[p];
    }
    if (running != v->tile_cols)
        return false;
    for (uint32_t r = 0; r < run_count; r++)
        for (uint32_t c = runs[r].first; c < runs[r].first + runs[r].count; c++, view_column++) {
            uint32_t want_address, want_bank, p = 0u;
            uint16_t want_x, want_y;

            source_column(src, c, &want_address, &want_x, &want_y, &want_bank);
            while (p < v->shard_count &&
                   view_column >= v->shard_column_first[p] + v->shard_tile_cols[p])
                p++;
            if (p == v->shard_count ||
                v->shard_address[p] + (view_column - v->shard_column_first[p]) * v->tile_rows *
                        v->tile_bytes != want_address ||
                v->shard_x[p] != want_x || v->shard_y[p] != want_y || v->shard_bank[p] != want_bank)
                return false;
        }
    return view_column == v->tile_cols && v->bank == v->shard_bank[0] &&
           v->address == v->shard_address[0];
}

static void test_draft_vocab(void)
{
    const pn_tt_column_run runs[2] = { { 0u, DRAFT_TILES },
                                       { TAIL_FIRST, Q38_VOCAB_TILES - TAIL_FIRST } };
    pn_tt_weight src, v;
    uint32_t widest = 0u, banks_seen = 0u;
    char what[160];

    source_of(&src, MEMBERS);
    check(pn_tt_weight_column_view(&src, runs, 2u, MEMBERS, &v) == 0, "the draft view builds");
    check(v.tile_cols == 2057u, "2,057 view columns (2,048 + the 9-tile tail)");
    check(view_exact(&src, &v, runs, 2u),
          "EXACT: every view column reads its source column's address, door and bank");
    for (uint32_t p = 0; p < v.shard_count; p++) {
        widest = v.shard_tile_cols[p] > widest ? v.shard_tile_cols[p] : widest;
        banks_seen |= 1u << v.shard_bank[p];
    }
    snprintf(what, sizeof(what), "SPREAD: %u pieces (of 16), widest %u, over %d banks",
             v.shard_count, widest, __builtin_popcount(banks_seen));
    /* 2,048 columns live in members 0..4 (5 panels), the tail in member 15: 6 segments, the 10
     * spare members split the wide ones; one source panel of 485 -> pieces of <= 162. */
    check(v.shard_count == MEMBERS && widest <= 162u && __builtin_popcount(banks_seen) == 6, what);

    /* A mutation of the arithmetic must be caught by the exactness check: one piece one column
     * late reads the wrong bytes. */
    v.shard_address[3] += Q38_EMBD_TILES * BFP8_TILE;
    check(!view_exact(&src, &v, runs, 2u), "RED: one piece one column late is caught");
}

static void test_refusals(void)
{
    const pn_tt_column_run off_end[1] = { { 7700u, 61u } };
    const pn_tt_column_run repeat[2] = { { 0u, 10u }, { 5u, 10u } };
    const pn_tt_column_run spread[3] = { { 0u, 1u }, { 1000u, 1u }, { 7000u, 1u } };
    pn_tt_weight src, v, empty;

    source_of(&src, MEMBERS);
    memset(&empty, 0, sizeof(empty));
    check(pn_tt_weight_column_view(&src, off_end, 1u, MEMBERS, &v) != 0, "a run off the end");
    check(pn_tt_weight_column_view(&src, repeat, 2u, MEMBERS, &v) != 0, "a repeated column");
    check(pn_tt_weight_column_view(&src, spread, 3u, 2u, &v) != 0,
          "3 segments over 2 members: refused");
    check(pn_tt_weight_column_view(&src, spread, 3u, 3u, &v) == 0 && v.shard_count == 3u &&
              view_exact(&src, &v, spread, 3u),
          "3 one-column segments over 3 members: one piece each, exact");
    check(pn_tt_weight_column_view(&src, spread, 3u, 0u, &v) != 0, "no members");
    check(pn_tt_weight_column_view(&empty, spread, 3u, MEMBERS, &v) != 0, "an empty source");
}

static void test_unsharded(void)
{
    const pn_tt_column_run runs[2] = { { 0u, DRAFT_TILES },
                                       { TAIL_FIRST, Q38_VOCAB_TILES - TAIL_FIRST } };
    pn_tt_weight src, v;

    source_of(&src, 1u);
    check(src.shard_tile_cols[0] == Q38_VOCAB_TILES, "the one-panel source");
    check(pn_tt_weight_column_view(&src, runs, 2u, MEMBERS, &v) == 0 && v.shard_count == MEMBERS &&
              view_exact(&src, &v, runs, 2u),
          "an unsharded source splits over 16 members, exact");
    check(pn_tt_weight_column_view(&src, runs, 2u, 1u, &v) != 0,
          "two runs over one member: two segments, refused");
}

int main(void)
{
    printf("test_tt_weight_view: column views of a resident weight\n");
    test_draft_vocab();
    test_refusals();
    test_unsharded();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
