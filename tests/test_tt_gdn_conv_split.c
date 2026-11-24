/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_gdn_conv_split.c
 * @brief The GDN PHASE CONV split, 4 / 8 / 16 ways, against the whole descriptor on an exact
 *        host model of the kernels' column walk. No card.
 *
 * The production pieces: `pn_tt_gdn_conv_member_count()` and `pn_tt_gdn_slice()` (the code
 * `gdn_submit_delta_split()` runs), foundry's encoder `ft_tt_op_gdn_parameters()` /
 * `ft_tt_op_gdn_row_extension()`, and the KERNELS' OWN decode `ftg_decode()` and
 * `ftg_conv_plane_tile()`. The model is the reader / compute / writer contract of
 * `resident_gdn_{reader,writer}.c` on a host DRAM pool: for every column of the decoded
 * slice, the tap sequence (window planes, then stream rows), SiLU of the weighted sum to the
 * output row, the rolled window to the state base (or extension word r, multi-row). Tiles
 * are modelled as 4 rows x 32 lanes — the rows a strip descriptor touches; whole-tile mode
 * runs every modelled element.
 *
 *   1. Member counts: narrow keeps Qwen3.8 at 4 (shipped), wide takes 16, or 8 when asked.
 *   2. Narrow descriptors are the pre-change bytes (high half-word zero, taps word as was).
 *   3. EXACT: at 8 and 16 (and 4, and one explicit [0, 320) slice) the sliced members leave
 *      the DRAM pool byte for byte equal to the whole descriptor, members run forward and
 *      reversed, every tile written exactly as often — for all 48 layers (per-layer data
 *      and addresses), strip one-row, strip two-row (the k1 verify) and whole-tile modes.
 *   4. REDS DIFFER: a duplicated, a dropped and a wrapped (high byte lost) slice set. The
 *      wrap is a no-op at 4 ways — the narrow split never needs the high byte.
 *
 * Build: `make tests/test_tt_gdn_conv_split` (in `make test`; no USE_TT, no card).
 * Mutation reds: tests/run_tt_gdn_conv_split_mutants.sh.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/accel/tt/pn_tt_gdn_conv_split.h"
#include "tt/ft_tt_op_gdn.h"

/* The L1 pointer attribute is the SFPI fork's; a host compiler reads it as nothing. */
#define FTQ_L1_PTR
#include "tt/kernels/resident_gdn_protocol.h"

/* The model's tile: the 4 rows a strip descriptor can touch here, 32 lanes each. */
#define MODEL_ROWS 4u
#define LANES 32u
#define TILE_FLOATS (MODEL_ROWS * LANES)
#define TILE_ADDRESS_BYTES 4096u /* the fp32 tile the addresses step by */
#define LAYERS 48u
#define TAPS 4u
#define MAX_ROWS 2u
#define MAX_MEMBERS 16u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/* ft_tt_op_gdn.c's CB placement is never called here; this keeps the link to the encoder. */
int ft_tt_op_cb_place(const ft_tt_op_cb_request *requests, unsigned count,
                      ft_tt_launch_cb_entry *out_entries, void *buffer, size_t buffer_bytes,
                      uint64_t *out_mask)
{
    (void)requests, (void)count, (void)out_entries, (void)buffer, (void)buffer_bytes;
    (void)out_mask;
    return FT_TT_EINVAL;
}

/* ================================================================
 * The host DRAM pool and the kernels' CONV walk
 * ================================================================ */

typedef struct {
    float *tiles;          /**< `count` model tiles */
    uint32_t *writes;      /**< how often each tile was written */
    uint32_t count;
    uint32_t out_of_range; /**< tile indices past the pool a descriptor addressed */
    float scratch[TILE_FLOATS];
} dram_pool;

/** One layer's buffers, as tile indices into the pool. */
typedef struct {
    uint32_t in, out, weight, window, rows_out[MAX_ROWS];
} layer_layout;

static uint32_t tile_of(uint32_t address)
{
    return address / TILE_ADDRESS_BYTES;
}

/** A tile row of the pool; an address past it is counted and lands in scratch. */
static float *tile_row(dram_pool *pool, uint32_t tile, uint32_t row)
{
    if (tile >= pool->count || row >= MODEL_ROWS) {
        pool->out_of_range++;
        return pool->scratch;
    }
    return pool->tiles + (size_t)tile * TILE_FLOATS + (size_t)row * LANES;
}

static void count_write(dram_pool *pool, uint32_t tile)
{
    if (tile < pool->count)
        pool->writes[tile]++;
}

/**
 * One descriptor, as the reader / compute / writer triad runs it: decoded by `ftg_decode()`,
 * columns [col_first, +col_count), planes at `ftg_conv_plane_tile()`. Every column's operands
 * are read before any of its results are written (the reader's passes precede the writer's).
 */
static void model_conv(uint32_t words[FT_TT_OP_GDN_WORDS],
                       const uint32_t extension[FT_TT_OP_GDN_WORDS], dram_pool *pool)
{
    const ftg_shape shape = ftg_decode(words);
    const bool strip = (words[kGdnFlags] & (uint32_t)kGdnFlagConvStrip) != 0u;
    const uint32_t in = tile_of(words[kGdnActivationIn]);
    const uint32_t out = tile_of(words[kGdnActivationOut]);
    const uint32_t weight = tile_of(words[kGdnConstantWeight]);
    const uint32_t window = tile_of(words[kGdnStateConv]);
    const uint32_t state = tile_of((words[kGdnFlags] & (uint32_t)kGdnFlagStateOut)
                                       ? words[kGdnStateRecurrent]
                                       : words[kGdnStateConv]);
    const uint32_t lanes = strip ? LANES : TILE_FLOATS;
    const uint32_t taps = shape.conv_taps;
    float sequence[TAPS + MAX_ROWS][TILE_FLOATS];
    uint32_t column, row, tap, lane;

    if (!ftg_shape_is_supported(&shape) || taps != TAPS || shape.row_count > MAX_ROWS)
        return;
    for (column = shape.col_first; column < shape.col_first + shape.col_count; column++) {
        /* The tap sequence: the live window planes, then the stream rows. */
        for (tap = 0; tap + 1u < taps; tap++)
            memcpy(sequence[tap], tile_row(pool, window + ftg_conv_plane_tile(&shape, tap, column), 0u),
                   lanes * sizeof(float));
        for (row = 0; row < shape.row_count; row++)
            memcpy(sequence[taps - 1u + row],
                   tile_row(pool, in + column, strip ? shape.row_first + row : 0u),
                   lanes * sizeof(float));
        for (row = 0; row < shape.row_count; row++) {
            const uint32_t base = shape.row_count > 1u ? tile_of(extension[row]) : state;
            float *dst = tile_row(pool, out + column, strip ? shape.row_first + row : 0u);

            for (lane = 0; lane < lanes; lane++) {
                float sum = 0.0f;

                for (tap = 0; tap < taps; tap++)
                    sum += tile_row(pool, weight + ftg_conv_plane_tile(&shape, tap, column),
                                    0u)[lane] * sequence[tap + row][lane];
                dst[lane] = sum / (1.0f + expf(-sum));
            }
            count_write(pool, out + column);
            for (tap = 0; tap + 1u < taps; tap++) {
                memcpy(tile_row(pool, base + ftg_conv_plane_tile(&shape, tap, column), 0u),
                       sequence[tap + 1u + row], lanes * sizeof(float));
                count_write(pool, base + ftg_conv_plane_tile(&shape, tap, column));
            }
        }
    }
}

/* ================================================================
 * Requests, pools and arms
 * ================================================================ */

typedef enum { MODE_STRIP_ONE_ROW, MODE_STRIP_TWO_ROWS, MODE_WHOLE_TILE, MODE_COUNT } conv_mode;

static const char *const MODE_NAMES[MODE_COUNT] = {"strip, one row", "strip, two rows (k1)",
                                                   "whole tile"};

static ft_tt_op_operand pool_operand(uint32_t tile)
{
    ft_tt_op_operand operand;

    operand.x = 1u;
    operand.y = 2u;
    operand.address = tile * TILE_ADDRESS_BYTES;
    operand.tile_bytes = FT_TT_OP_TILE_BYTES_FP32;
    return operand;
}

/** Per-layer addresses: each layer's buffers start `layer` tiles further in, a gap apart. */
static layer_layout layout_for(uint32_t cols, uint32_t layer, uint32_t *out_tiles)
{
    layer_layout layout;
    uint32_t cursor = 1u + layer;

    layout.in = cursor, cursor += cols + 3u;
    layout.out = cursor, cursor += cols + 5u;
    layout.weight = cursor, cursor += TAPS * cols + 7u;
    layout.window = cursor, cursor += (TAPS - 1u) * cols + 11u;
    layout.rows_out[0] = cursor, cursor += (TAPS - 1u) * cols + 13u;
    layout.rows_out[1] = cursor, cursor += (TAPS - 1u) * cols + 17u;
    *out_tiles = cursor;
    return layout;
}

static ft_tt_op_gdn_request conv_request(uint32_t conv_dim, const layer_layout *layout,
                                         conv_mode mode)
{
    ft_tt_op_gdn_request request;

    memset(&request, 0, sizeof(request));
    request.phase = FT_TT_OP_GDN_PHASE_CONV;
    request.state_size = 128u;
    request.v_heads = 48u;
    request.k_heads = 16u;
    request.conv_dim = conv_dim;
    request.conv_taps = TAPS;
    request.activation_in = pool_operand(layout->in);
    request.activation_out = pool_operand(layout->out);
    request.conv_weight = pool_operand(layout->weight);
    request.state_conv = pool_operand(layout->window);
    if (mode != MODE_WHOLE_TILE)
        request.flags = FT_TT_OP_GDN_FLAG_CONV_STRIP;
    if (mode == MODE_STRIP_TWO_ROWS) {
        request.row_count = 2u;
        request.state_rows_out[0] = pool_operand(layout->rows_out[0]);
        request.state_rows_out[1] = pool_operand(layout->rows_out[1]);
    }
    return request;
}

/** Deterministic per-layer data in every tile (outputs too: an unwritten tile shows). */
static void pool_fill(dram_pool *pool, uint32_t seed)
{
    uint32_t state = 0x9E3779B9u ^ (seed * 0x85EBCA6Bu);
    size_t index;

    for (index = 0; index < (size_t)pool->count * TILE_FLOATS; index++) {
        state = state * 1664525u + 1013904223u;
        pool->tiles[index] = (float)((int32_t)(state >> 8) - 0x800000) / 8388608.0f;
    }
    memset(pool->writes, 0, pool->count * sizeof(*pool->writes));
    pool->out_of_range = 0u;
}

static bool pool_alloc(dram_pool *pool, uint32_t count)
{
    pool->count = count;
    pool->tiles = calloc((size_t)count * TILE_FLOATS, sizeof(*pool->tiles));
    pool->writes = calloc(count, sizeof(*pool->writes));
    return pool->tiles && pool->writes;
}

static void pool_free(dram_pool *pool)
{
    free(pool->tiles);
    free(pool->writes);
    pool->tiles = NULL;
    pool->writes = NULL;
}

static void pool_copy(dram_pool *dst, const dram_pool *src)
{
    memcpy(dst->tiles, src->tiles, (size_t)src->count * TILE_FLOATS * sizeof(*src->tiles));
    memcpy(dst->writes, src->writes, src->count * sizeof(*src->writes));
    dst->out_of_range = src->out_of_range;
}

static bool pool_equal(const dram_pool *a, const dram_pool *b)
{
    return a->out_of_range == 0u && b->out_of_range == 0u &&
           memcmp(a->tiles, b->tiles, (size_t)a->count * TILE_FLOATS * sizeof(*a->tiles)) == 0 &&
           memcmp(a->writes, b->writes, a->count * sizeof(*a->writes)) == 0;
}

/** Encode `request` (and its row extension) and run it on `pool`. false = refused. */
static bool run_descriptor(const ft_tt_op_gdn_request *request, dram_pool *pool)
{
    uint32_t words[FT_TT_OP_GDN_WORDS], extension[FT_TT_OP_GDN_WORDS];

    memset(extension, 0, sizeof(extension));
    if (ft_tt_op_gdn_parameters(request, words) != FT_TT_OK)
        return false;
    if (request->row_count > 1u && ft_tt_op_gdn_row_extension(request, extension) != FT_TT_OK)
        return false;
    model_conv(words, extension, pool);
    return true;
}

/** The split arm: every member's production slice, in order or reversed. */
static bool run_split(const ft_tt_op_gdn_request *request, uint32_t members,
                      pn_tt_gdn_slice_red red, bool reversed, dram_pool *pool)
{
    uint32_t step;

    for (step = 0; step < members; step++) {
        const uint32_t index = reversed ? members - 1u - step : step;
        ft_tt_op_gdn_request sliced;

        if (pn_tt_gdn_slice(request, members, index, ft_tt_op_gdn_conv_depth(), red,
                            &sliced) != 0 ||
            !run_descriptor(&sliced, pool))
            return false;
    }
    return true;
}

/* ================================================================
 * The checks
 * ================================================================ */

static void test_member_counts(void)
{
    const uint32_t granule = ft_tt_op_gdn_conv_slice_granule(128u, TAPS);
    ft_tt_op_gdn_request request;
    uint32_t tiles, words[FT_TT_OP_GDN_WORDS];
    layer_layout layout = layout_for(320u, 0u, &tiles);

    printf("member counts (Qwen3.8 320 tile columns, 16 armed; Ornith 256)\n");
    check(granule == 4u, "the CONV slice granule at Qwen3.8's placement is 4 columns");
    check(pn_tt_gdn_conv_wide_enabled(NULL) && pn_tt_gdn_conv_wide_enabled("") &&
              pn_tt_gdn_conv_wide_enabled("1") && !pn_tt_gdn_conv_wide_enabled("0"),
          "PN_TT_GDN_CONV_WIDE: unset or empty is wide (the default), =0 the narrow cap");
    check(pn_tt_gdn_conv_member_count(320u, 16u, 16u, granule, PN_TT_GDN_COL_BYTE_MAX) == 4u,
          "narrow (shipped): Qwen3.8 CONV over 4 members of 80");
    check(pn_tt_gdn_conv_member_count(320u, 16u, 8u, granule, PN_TT_GDN_COL_BYTE_MAX) == 4u,
          "narrow: an 8-way ask still falls to 4 (column 280 is past the bytes)");
    check(pn_tt_gdn_conv_member_count(320u, 16u, 16u, granule, FT_TT_OP_GDN_COL_MAX) == 16u,
          "wide: Qwen3.8 CONV over all 16 members of 20");
    check(pn_tt_gdn_conv_member_count(320u, 16u, 8u, granule, FT_TT_OP_GDN_COL_MAX) == 8u,
          "wide: PN_TT_GDN_CONV_CORES=8 gives 8 members of 40");
    check(pn_tt_gdn_conv_member_count(256u, 16u, 16u, granule, PN_TT_GDN_COL_BYTE_MAX) == 16u &&
              pn_tt_gdn_conv_member_count(256u, 16u, 16u, granule, FT_TT_OP_GDN_COL_MAX) == 16u,
          "Ornith's 256 columns split 16 ways either way");
    check(!pn_tt_gdn_conv_split_fits(320u, 16u, granule, PN_TT_GDN_COL_BYTE_MAX) &&
              pn_tt_gdn_conv_split_fits(320u, 16u, granule, FT_TT_OP_GDN_COL_MAX) &&
              !pn_tt_gdn_conv_split_fits(320u, 64u, granule, FT_TT_OP_GDN_COL_MAX),
          "fits: 16 narrow no, 16 wide yes, 64 (5 columns, not a granule multiple) no");
    /* 288 / 16 = 18: a depth multiple, the slice size that broke DELTA on the card. */
    check(!pn_tt_gdn_conv_split_fits(288u, 16u, granule, FT_TT_OP_GDN_COL_MAX) &&
              pn_tt_gdn_conv_member_count(288u, 16u, 16u, granule, FT_TT_OP_GDN_COL_MAX) == 12u,
          "18-column slices never split: 288 columns step down to 12 members of 24");
    request = conv_request(10240u, &layout, MODE_STRIP_ONE_ROW);
    request.col_first = 300u;
    request.col_count = 18u;
    check(ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK,
          "the old drop red's [300, 318) is refused by the encoder");
}

/** Narrow (4-way) descriptors are the bytes the pre-change encoder emitted. */
static void test_narrow_bytes(void)
{
    layer_layout layout;
    ft_tt_op_gdn_request request, sliced;
    uint32_t tiles, index, words[FT_TT_OP_GDN_WORDS];
    bool same = true;

    printf("narrow descriptors are the shipped bytes\n");
    layout = layout_for(320u, 0u, &tiles);
    request = conv_request(10240u, &layout, MODE_STRIP_ONE_ROW);
    for (index = 0; index < 4u; index++) {
        const uint32_t taps_word = TAPS | ((index * 80u) << 8) | (80u << 16) |
                                   ((ft_tt_op_gdn_conv_depth() & 0xFFu) << 24);

        if (pn_tt_gdn_slice(&request, 4u, index, ft_tt_op_gdn_conv_depth(),
                            PN_TT_GDN_SLICE_EXACT, &sliced) != 0 ||
            ft_tt_op_gdn_parameters(&sliced, words) != FT_TT_OK ||
            words[FT_TT_OP_GDN_CONV_TILE_COLS] != 320u ||
            words[FT_TT_OP_GDN_CONV_TAPS] != taps_word)
            same = false;
    }
    check(same, "every 4-way member: tile-column word 320, taps word first<<8 | count<<16");
}

/** All 48 layers, every mode: sliced == whole, byte for byte. */
static void test_exact(void)
{
    static const uint32_t splits[] = {4u, 8u, 16u};
    dram_pool initial, whole, split;
    uint32_t tiles, layer, mode, s;
    layer_layout layout = layout_for(320u, LAYERS - 1u, &tiles);
    bool ok[MODE_COUNT][4];
    char what[160];

    printf("exact: sliced CONV == whole descriptor, 48 layers\n");
    memset(ok, 1, sizeof(ok));
    if (!pool_alloc(&initial, tiles) || !pool_alloc(&whole, tiles) || !pool_alloc(&split, tiles)) {
        check(false, "pool allocation");
        return;
    }
    for (layer = 0; layer < LAYERS; layer++) {
        layout = layout_for(320u, layer, &tiles);
        for (mode = 0; mode < MODE_COUNT; mode++) {
            const ft_tt_op_gdn_request request = conv_request(10240u, &layout, (conv_mode)mode);
            ft_tt_op_gdn_request explicit_whole = request;

            pool_fill(&initial, layer * MODE_COUNT + mode);
            pool_copy(&whole, &initial);
            if (!run_descriptor(&request, &whole)) {
                ok[mode][0] = false;
                continue;
            }
            for (s = 0; s < 3u; s++) {
                pool_copy(&split, &initial);
                if (!run_split(&request, splits[s], PN_TT_GDN_SLICE_EXACT, false, &split) ||
                    !pool_equal(&split, &whole))
                    ok[mode][s] = false;
                pool_copy(&split, &initial);
                if (!run_split(&request, splits[s], PN_TT_GDN_SLICE_EXACT, true, &split) ||
                    !pool_equal(&split, &whole))
                    ok[mode][s] = false;
            }
            /* One member owning [0, 320) explicitly: the count needs its high byte. */
            explicit_whole.col_first = 0u;
            explicit_whole.col_count = 320u;
            pool_copy(&split, &initial);
            if (!run_descriptor(&explicit_whole, &split) || !pool_equal(&split, &whole))
                ok[mode][3] = false;
        }
    }
    for (mode = 0; mode < MODE_COUNT; mode++) {
        for (s = 0; s < 3u; s++) {
            snprintf(what, sizeof(what), "%u-way == whole, forward and reversed (%s)",
                     splits[s], MODE_NAMES[mode]);
            check(ok[mode][s], what);
        }
        snprintf(what, sizeof(what), "explicit [0, 320) == unsliced (%s)", MODE_NAMES[mode]);
        check(ok[mode][3], what);
    }
    pool_free(&initial);
    pool_free(&whole);
    pool_free(&split);
}

/** Each red differs from the whole descriptor; the wrap is a no-op at 4 ways. */
static void test_reds(void)
{
    static const struct {
        pn_tt_gdn_slice_red red;
        uint32_t members;
        bool differs;
        const char *what;
    } arms[] = {
        {PN_TT_GDN_SLICE_RED_COL_DUP, 16u, true, "red gdn_conv_col_dup at 16 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_DUP, 8u, true, "red gdn_conv_col_dup at 8 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_DROP, 16u, true, "red gdn_conv_col_drop at 16 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_DROP, 8u, true, "red gdn_conv_col_drop at 8 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_WRAP, 16u, true, "red gdn_conv_col_wrap at 16 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_WRAP, 8u, true, "red gdn_conv_col_wrap at 8 ways DIFFERS"},
        {PN_TT_GDN_SLICE_RED_COL_WRAP, 4u, false,
         "red gdn_conv_col_wrap at 4 ways is IDENTICAL (narrow needs no high byte)"},
    };
    dram_pool initial, whole, split;
    uint32_t tiles, arm, layer;
    layer_layout layout = layout_for(320u, LAYERS - 1u, &tiles);

    printf("reds: overlapping, dropped and wrapped slices\n");
    if (!pool_alloc(&initial, tiles) || !pool_alloc(&whole, tiles) || !pool_alloc(&split, tiles)) {
        check(false, "pool allocation");
        return;
    }
    for (arm = 0; arm < sizeof(arms) / sizeof(arms[0]); arm++) {
        bool all = true;

        for (layer = 0; layer < LAYERS; layer += LAYERS - 1u) {
            const ft_tt_op_gdn_request request =
                conv_request(10240u, (layout = layout_for(320u, layer, &tiles), &layout),
                             MODE_STRIP_ONE_ROW);
            bool differs;

            pool_fill(&initial, 1000u + layer);
            pool_copy(&whole, &initial);
            pool_copy(&split, &initial);
            differs = !run_descriptor(&request, &whole) ||
                      !run_split(&request, arms[arm].members, arms[arm].red, false, &split) ||
                      !pool_equal(&split, &whole);
            all = all && differs == arms[arm].differs;
        }
        check(all, arms[arm].what);
    }
    pool_free(&initial);
    pool_free(&whole);
    pool_free(&split);
}

int main(void)
{
    test_member_counts();
    test_narrow_bytes();
    test_exact();
    test_reds();
    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
