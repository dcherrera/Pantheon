/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdn_layout.c
 * @brief OFF-HARDWARE acceptance for the GATED-DELTA-NET host STAGER's address arithmetic —
 *        the four conversions in `src/accel/tt/pn_tt_gdn_layout.c`.
 *
 * Opens no device. Build with `make USE_TT=1 tests/test_gdn_layout` and run it anywhere,
 * including on the Mac where the code is written.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS FOR, AND WHY IT SIMULATES THE READER
 * ---------------------------------------------------------------------------
 * The stager and the two dataflow kernels are the two ends of a contract NOTHING checks at
 * run time: a stager that disagrees about where a channel or a head sits produces plausible
 * numbers from the first token instead of an error. `foundry/tt/tests/test_gdn_ops.c` pins
 * the ACCESSORS the kernels compute their offsets with; this pins what the HOST writes into
 * the addresses those accessors name.
 *
 * So every check below reads the staged bytes back the way the READER would — through
 * `ftg_delta_head_tile()`, `ftg_conv_plane_tile()`, `ftg_y_tile()` and the face index — and
 * asserts it recovers the value the CPU ORACLE (`pn_qwen35moe_ssm_recurrence()`) says belongs
 * there. Nothing here is compared against the function that produced it.
 *
 * THE ONE THAT MATTERS MOST is the conv weight. The oracle stores it row-major
 * `[conv_dim, taps]` and the card wants `taps` TAP-MAJOR PLANES, so the stager TRANSPOSES.
 * A reshape is the same bytes, the same length and a different answer, and the arm below
 * asserts that the two are not merely different in principle but disagree at a named channel.
 *
 * ---------------------------------------------------------------------------
 * q | k | v AND y ARE HEAD-MAJOR ROW-0 STRIPS, AND THERE IS NO STAGER LEFT FOR THEM
 * ---------------------------------------------------------------------------
 * A head's `state` values are `state/32` CONSECUTIVE tiles, all of the data in ROW 0 of each,
 * head `h` starting at `pn_tt_gdn_head_tile(h, state)` = `h * state/32`. That is the geometry
 * `pn_tt_gdn_row0_offset()` already addresses and the geometry PHASE CONV's output already
 * has, so there is nothing to tilize: `pn_tt_gdn_stage_stream()` and its inverse are GONE and
 * the traffic is `pn_tt_fp32_row0_move()`, two 64 B chunks a tile.
 *
 * They used to be tilized `[heads, state]` matrices with ONE ROW PER HEAD — 32x denser, and it
 * forced the host to de-tilize conv_out, L2-normalise on the CPU and re-tilize between the two
 * phases, 49,408 B a layer over PCIe. The arms below therefore test what a MOVE plus the
 * offsets place, not what a tilizer produces: `strip_write()` here is `pn_tt_fp32_row0_move()`
 * with the NoC swapped for a memcpy, which is the only part of it a machine without a card can
 * run. The one-row-per-head arithmetic is asserted to DISAGREE at a named head, the same way
 * the reshape is for the conv weight.
 */

#include "../src/accel/tt/pn_tt_gdn_layout.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/ft_tt_op_gdn.h"
#include "tt/ft_tt_tilize.h"

/* The device's L1 pointer attribute is the SFPI GCC fork's, not the contract's; the header
 * guards it so a host compiler can read the file. Same escape test_gdn_ops.c uses. */
#define FTQ_L1_PTR
#include "tt/kernels/resident_gdn_protocol.h"

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                                   \
    do {                                                                                   \
        g_checks++;                                                                        \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "  FAIL  %s (%s:%d)\n", (msg), __FILE__, __LINE__);            \
            g_failed++;                                                                    \
        } else {                                                                           \
            fprintf(stderr, "  ok    %s\n", (msg));                                        \
        }                                                                                  \
    } while (0)

/** Ornith's shape, read off `ft_tt_op_gdn.h`'s own constants rather than retyped. */
static ftg_shape ornith_shape(uint32_t phase)
{
    ftg_shape shape;

    shape.phase = phase;
    shape.state_tiles = FT_TT_OP_GDN_QWEN35_STATE / 32u;
    shape.v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    shape.k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    shape.conv_tile_cols = FT_TT_OP_GDN_QWEN35_CONV_DIM / 32u;
    shape.conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    shape.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    shape.tile_count = 0u;
    return shape;
}

/** The float a tilized strip holds at [row][column] of tile `tile`. The reader's own
 *  arithmetic, spelled from `ft_tt_tilize.h` rather than from the stager. */
static float tile_element(const unsigned char *tiles, uint32_t tile, uint32_t row,
                          uint32_t column)
{
    const float *base = (const float *)(const void *)tiles;

    return base[(size_t)tile * 1024u + ft_tt_tile_face_index(row, column)];
}

/* ================================================================
 * Logical row 0 of a strip of fp32 tiles
 * ================================================================ */

static void test_row0_offsets(void)
{
    const uint32_t conv_dim = 256u;
    const uint32_t tiles = conv_dim / 32u;
    float *source = malloc((size_t)conv_dim * sizeof(float));
    unsigned char *staged = calloc((size_t)tiles, PN_TT_GDN_LAYOUT_TILE_BYTES);
    uint32_t index;
    int recovered = 1;

    fprintf(stderr, "\nlogical row 0 of an fp32 tile — the two 64 B pieces\n");

    /* Worked out from ft_tt_tilize.h: face 0 is elements [0,256) and face 1 is [256,512), so
     * in fp32 the two halves of row 0 are 1024 B apart inside a 4096 B tile. */
    CHECK(pn_tt_gdn_row0_offset(0u, 0u) == 0u, "tile 0 chunk 0 is at byte 0");
    CHECK(pn_tt_gdn_row0_offset(0u, 1u) == 1024u, "tile 0 chunk 1 is 1024 B on, not 512");
    CHECK(pn_tt_gdn_row0_offset(1u, 0u) == 4096u, "tile 1 chunk 0 is one whole tile on");
    CHECK(pn_tt_gdn_row0_offset(3u, 1u) == 3u * 4096u + 1024u, "tile 3 chunk 1 is 13312");
    CHECK(PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES == 64u,
          "an fp32 row-0 chunk is exactly the 64 B DRAM read granule — nothing discarded");

    /* And the semantic version: a strip tilized as ONE row, read back through those offsets,
     * has to reproduce the source. This is what a wrong face stride breaks. */
    for (index = 0; index < conv_dim; index++)
        source[index] = (float)index + 0.5f;
    if (pn_tt_gdn_stage_plane(source, 0u, 1u, conv_dim, source, staged) != 0)
        recovered = 0;
    for (index = 0; index < conv_dim && recovered; index++) {
        const uint32_t tile = index / 32u;
        const uint32_t chunk = (index % 32u) / 16u;
        const float *piece =
            (const float *)(const void *)(staged + pn_tt_gdn_row0_offset(tile, chunk));

        if (piece[index % 16u] != (float)index + 0.5f)
            recovered = 0;
    }
    CHECK(recovered, "row 0 read through the chunk offsets reproduces the strip, element for "
                     "element");
    free(source);
    free(staged);
}

/* ================================================================
 * ANY logical row — the chunk phase-isolation arm's row shuttle
 * ================================================================ */

/**
 * The arm that says WHICH chunk kernel owns the measured prefill divergence shuttles row `t`
 * of a chunk's buffer to row 0 and back, so every one of its answers rests on this offset.
 *
 * THE PLAUSIBLE WRONG ANSWER IS A FLAT `row * 64`, which is right for rows 0-15 and wrong for
 * 16-31 because the tile is four 16x16 FACES, not one 32x32 block. That mistake would read
 * half of every chunk out of the wrong face and the attribution would be an artefact — so it
 * is asserted against here, not just avoided.
 */
static void test_row_offsets(void)
{
    const uint32_t rows = 32u;
    const uint32_t cols = 128u;
    const uint32_t tiles = cols / 32u;
    float *source = malloc((size_t)rows * cols * sizeof(float));
    unsigned char *staged = calloc((size_t)tiles, PN_TT_GDN_LAYOUT_TILE_BYTES);
    uint32_t row;
    uint32_t col;
    int recovered = 1;

    fprintf(stderr, "\nany logical row of an fp32 tile — the phase-isolation arm's shuttle\n");

    CHECK(pn_tt_gdn_row_offset(0u, 0u, 0u) == pn_tt_gdn_row0_offset(0u, 0u) &&
              pn_tt_gdn_row_offset(2u, 0u, 1u) == pn_tt_gdn_row0_offset(2u, 1u),
          "row 0 of the general offset IS pn_tt_gdn_row0_offset, tile and chunk alike");
    CHECK(pn_tt_gdn_row_offset(0u, 1u, 0u) == 64u,
          "row 1 is 64 B down inside face 0");
    CHECK(pn_tt_gdn_row_offset(0u, 16u, 0u) == 2048u,
          "row 16 crosses into FACE 2 — the boundary a flat row*64 misses");
    CHECK(pn_tt_gdn_row_offset(0u, 16u, 0u) != 16u * 64u,
          "and it is not 1024: the flat formula is the wrong answer this test exists for");
    CHECK(pn_tt_gdn_row_offset(0u, 31u, 1u) == 3u * 1024u + 15u * 64u,
          "row 31 chunk 1 is the last 64 B of face 3");
    CHECK(pn_tt_gdn_row_offset(2u, 5u, 1u) == 2u * 4096u + 1024u + 5u * 64u,
          "tile 2 row 5 chunk 1 composes tile, face and in-face row");

    /* The semantic version: a REAL 32-row tilization, read back through these offsets, has to
     * reproduce every element — the row-0 test's argument at 32 rows instead of one. */
    for (row = 0; row < rows; row++)
        for (col = 0; col < cols; col++)
            source[(size_t)row * cols + col] = (float)(row * 1000u + col) + 0.5f;
    if (ft_tt_tilize_strip(source, sizeof(float), rows, cols, (size_t)cols * sizeof(float),
                           staged) != FT_TT_TILIZE_OK)
        recovered = 0;
    for (row = 0; row < rows && recovered; row++) {
        for (col = 0; col < cols; col++) {
            const uint32_t tile = col / 32u;
            const uint32_t chunk = (col % 32u) / 16u;
            const float *piece = (const float *)(const void *)(staged +
                                 pn_tt_gdn_row_offset(tile, row, chunk));

            if (piece[col % 16u] != (float)(row * 1000u + col) + 0.5f) {
                recovered = 0;
                break;
            }
        }
    }
    CHECK(recovered, "every element of a 32x128 tilization is where pn_tt_gdn_row_offset says");
    free(source);
    free(staged);
}

/* ================================================================
 * The conv weight — the TRANSPOSE
 * ================================================================ */

static void test_conv_weight_transpose(void)
{
    const ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_CONV);
    const uint32_t conv_dim = 256u; /* a small stand-in; the arithmetic is per channel */
    const uint32_t taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    const uint32_t tiles = conv_dim / 32u;
    float *weight = malloc((size_t)conv_dim * taps * sizeof(float));
    float *scratch = malloc((size_t)conv_dim * sizeof(float));
    unsigned char *planes = calloc((size_t)tiles * taps, PN_TT_GDN_LAYOUT_TILE_BYTES);
    uint32_t channel, tap;
    int transposed = 1;
    int rows_padded = 1;

    fprintf(stderr, "\nthe conv1d weight — TAP-MAJOR PLANES, which is a TRANSPOSE\n");

    /* The oracle's order: `w[c*taps + tap]` — qwen35moe_ssm.c's `cw = w + c*k; cw[tap]`. The
     * value encodes both indices so a mis-stage cannot land on a coincidence. */
    for (channel = 0; channel < conv_dim; channel++)
        for (tap = 0; tap < taps; tap++)
            weight[(size_t)channel * taps + tap] = (float)channel * 16.0f + (float)tap;

    for (tap = 0; tap < taps; tap++) {
        /* THE TRANSPOSE: stride `taps`, starting at `tap`. */
        if (pn_tt_gdn_stage_plane(weight, tap, taps, conv_dim, scratch,
                                  planes + (size_t)tap * tiles *
                                               PN_TT_GDN_LAYOUT_TILE_BYTES) != 0)
            transposed = 0;
    }
    for (tap = 0; tap < taps && transposed; tap++) {
        for (channel = 0; channel < conv_dim; channel++) {
            /* Read it back the way the READER does: plane tile, row 0, column c%32. The
             * plane base is `ftg_conv_plane_tile()`, which is the accessor the kernel uses. */
            const uint32_t tile = tap * tiles + channel / 32u;
            const float got = tile_element(planes, tile, 0u, channel % 32u);

            if (got != (float)channel * 16.0f + (float)tap)
                transposed = 0;
        }
    }
    CHECK(transposed, "plane t, tile c/32, [0][c%32] holds w[c][t] — the oracle's own value");

    /* The plane index the reader computes for a NON-zero column, against a literal. */
    CHECK(ftg_conv_plane_tile(&shape, 3u, 5u) == 3u * 256u + 5u,
          "the reader reaches tap 3 column 5 at plane tile 773 (tap-major, not channel-major)");

    /* Rows 1-31 are the tile quantum. They must be ZERO: the compute thread multiplies whole
     * tiles, and a non-zero pad row would come back through SiLU as a non-zero conv_out row,
     * which is the strip invariant the rest of the forward depends on. */
    for (channel = 0; channel < conv_dim && rows_padded; channel++) {
        uint32_t row;

        for (row = 1u; row < 32u && rows_padded; row++) {
            if (tile_element(planes, channel / 32u, row, channel % 32u) != 0.0f)
                rows_padded = 0;
        }
    }
    CHECK(rows_padded, "rows 1-31 of every plane tile are zero — the tile quantum, staged");

    /* AND THE RESHAPE IS A DIFFERENT ANSWER. Not "could be": at channel 1 of plane 1 the
     * transpose gives w[1][1] = 17 and the reshape gives w[64][1] = 1025. Asserting they
     * disagree is what makes the distinction a fact rather than a caution. */
    {
        float transposed_value, reshaped_value;

        (void)pn_tt_gdn_stage_plane(weight, 1u, taps, conv_dim, scratch, planes);
        transposed_value = tile_element(planes, 0u, 0u, 1u);
        (void)pn_tt_gdn_stage_plane(weight, 1u * conv_dim, 1u, conv_dim, scratch, planes);
        reshaped_value = tile_element(planes, 0u, 0u, 1u);
        CHECK(transposed_value == 1.0f * 16.0f + 1.0f && reshaped_value != transposed_value,
              "the RESHAPE (first = t*conv_dim, stride 1) disagrees with the transpose at "
              "channel 1 — the mutation PN_TT_RED=gdn_conv_reshape injects");
    }
    free(weight);
    free(scratch);
    free(planes);
}

/* ================================================================
 * The DELTA input — q | k | v, and y coming back. HEAD-MAJOR ROW-0 STRIPS.
 * ================================================================ */

/** Fill `[heads, state]` with a value that encodes both indices. */
static void fill_matrix(float *matrix, uint32_t heads, uint32_t state, float tag)
{
    uint32_t head, index;

    for (head = 0; head < heads; head++)
        for (index = 0; index < state; index++)
            matrix[(size_t)head * state + index] = tag + (float)head * 1024.0f + (float)index;
}

/**
 * `pn_tt_fp32_row0_move()` with the NoC replaced by a memcpy — the ONLY part of that function
 * a machine with no card can run, and the part the layout lives in. Two 64 B chunks a tile, at
 * `pn_tt_gdn_row0_offset(first_tile + t, chunk)`. Everything the real one adds is a transport.
 */
static void strip_write(unsigned char *strip, uint32_t first_tile, const float *values,
                        uint32_t tiles)
{
    uint32_t tile, chunk;

    for (tile = 0; tile < tiles; tile++) {
        for (chunk = 0; chunk < PN_TT_GDN_LAYOUT_ROW0_CHUNKS; chunk++) {
            const float *host = values + (size_t)tile * PN_TT_GDN_LAYOUT_TILE_DIM +
                                (size_t)chunk * PN_TT_GDN_LAYOUT_ROW0_CHUNK_ELEMS;

            memcpy(strip + pn_tt_gdn_row0_offset(first_tile + tile, chunk), host,
                   PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES);
        }
    }
}

/** The read half of the same primitive. */
static void strip_read(const unsigned char *strip, uint32_t first_tile, float *values,
                       uint32_t tiles)
{
    uint32_t tile, chunk;

    for (tile = 0; tile < tiles; tile++) {
        for (chunk = 0; chunk < PN_TT_GDN_LAYOUT_ROW0_CHUNKS; chunk++) {
            float *host = values + (size_t)tile * PN_TT_GDN_LAYOUT_TILE_DIM +
                          (size_t)chunk * PN_TT_GDN_LAYOUT_ROW0_CHUNK_ELEMS;

            memcpy(host, strip + pn_tt_gdn_row0_offset(first_tile + tile, chunk),
                   PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES);
        }
    }
}

/**
 * `pn_tt_gdn_head_tile()` on its own, and composed with `pn_tt_gdn_row0_offset()` into the
 * BYTE offset of head `h`'s tile `t` — which is what a caller actually passes to the move.
 */
static void test_head_tile_offsets(void)
{
    const uint32_t state = FT_TT_OP_GDN_QWEN35_STATE; /* 128 */
    const uint32_t per_head = state / 32u;            /* 4 */
    const ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);
    uint32_t head;
    int agrees = 1;
    int strided = 1;

    fprintf(stderr, "\nhead-major ROW-0 strips — where head h's four tiles start\n");

    CHECK(pn_tt_gdn_head_tile(0u, state) == 0u && pn_tt_gdn_head_tile(1u, state) == 4u &&
              pn_tt_gdn_head_tile(31u, state) == 124u,
          "head 0, 1 and 31 start at tiles 0, 4 and 124 — h * state/32");
    CHECK(pn_tt_gdn_head_tile(32u, state) == 128u && pn_tt_gdn_head_tile(63u, state) == 252u,
          "and it keeps going past 32 heads: there is no row block to wrap into");
    CHECK(pn_tt_gdn_head_tile(2u, 256u) == 16u && pn_tt_gdn_head_tile(2u, 64u) == 4u,
          "the stride is state/32, so S=256 puts head 2 at tile 16 and S=64 at tile 4");

    /* THE BYTE OFFSET, which is the number the move is handed: head 3's tile 2, chunk 1. Head
     * 3 starts at tile 12, so that is tile 14 chunk 1 = 14*4096 + 1024 = 58368. */
    CHECK(pn_tt_gdn_row0_offset(pn_tt_gdn_head_tile(3u, state) + 2u, 1u) == 58368u,
          "head 3's tile 2, chunk 1 is byte 58368 of the strip");

    /* The HOST's head_tile and the DEVICE's accessors are two spellings of one contract: the
     * y tile the writer computes and the v tile the reader computes must both be the stager's
     * offset, or the two ends disagree with nothing to catch it. */
    for (head = 0; head < shape.v_heads; head++) {
        if (ftg_y_tile(&shape, head, 0u) != pn_tt_gdn_head_tile(head, state))
            agrees = 0;
        if (ftg_delta_head_tile(&shape, kGdnStreamV, head) !=
            ftg_delta_stream_base(&shape, kGdnStreamV) + pn_tt_gdn_head_tile(head, state))
            agrees = 0;
    }
    CHECK(agrees, "pn_tt_gdn_head_tile agrees with ftg_y_tile and ftg_delta_head_tile at every "
                  "one of the 32 v-heads — the host and the kernels compute the same tile");

    /* AND THE ONE-ROW-PER-HEAD ARITHMETIC IS A DIFFERENT ANSWER. It gave every head of a row
     * block the SAME `(h/32)*state_tiles` tiles and told them apart by row `h%32`, so its
     * stride was 0 where this one's is `state/32`. Asserting they disagree at head 1 is what
     * makes the change a fact rather than a note. */
    for (head = 0; head + 1u < 64u; head++) {
        if (pn_tt_gdn_head_tile(head + 1u, state) - pn_tt_gdn_head_tile(head, state) != per_head)
            strided = 0;
    }
    CHECK(strided && pn_tt_gdn_head_tile(1u, state) != (1u / 32u) * per_head,
          "consecutive heads are state/32 apart — the old (h/32)*state_tiles packing put "
          "heads 0 and 1 on the SAME tile and this disagrees with it at head 1");
}

static void test_delta_input_layout(void)
{
    const ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);
    const uint32_t state = FT_TT_OP_GDN_QWEN35_STATE;
    const uint32_t per_head = state / 32u;
    const uint32_t k_heads = shape.k_heads;
    const uint32_t v_heads = shape.v_heads;
    const uint32_t input_tiles = ftg_delta_input_tiles(&shape);
    float *q = malloc((size_t)k_heads * state * sizeof(float));
    float *k = malloc((size_t)k_heads * state * sizeof(float));
    float *v = malloc((size_t)v_heads * state * sizeof(float));
    unsigned char *staged = calloc(input_tiles, PN_TT_GDN_LAYOUT_TILE_BYTES);
    uint32_t head, index;
    int placed = 1;
    int repeat_reads = 1;
    int pad_rows_zero = 1;

    fprintf(stderr, "\nthe DELTA q | k | v — ONE head-major row-0 strip, q and k UN-repeated\n");

    fill_matrix(q, k_heads, state, 100000.0f);
    fill_matrix(k, k_heads, state, 200000.0f);
    fill_matrix(v, v_heads, state, 300000.0f);

    /* THE WHOLE OF THE STAGING, now: three moves at three head-major offsets. No tilize
     * buffer, no scratch, nothing to get wrong except the offset. */
    for (head = 0; head < k_heads; head++) {
        strip_write(staged,
                    ftg_delta_stream_base(&shape, kGdnStreamQ) +
                        pn_tt_gdn_head_tile(head, state),
                    q + (size_t)head * state, per_head);
        strip_write(staged,
                    ftg_delta_stream_base(&shape, kGdnStreamK) +
                        pn_tt_gdn_head_tile(head, state),
                    k + (size_t)head * state, per_head);
    }
    for (head = 0; head < v_heads; head++)
        strip_write(staged,
                    ftg_delta_stream_base(&shape, kGdnStreamV) +
                        pn_tt_gdn_head_tile(head, state),
                    v + (size_t)head * state, per_head);

    /* Read every element back the way `read_one_head()` does: `ftg_delta_head_tile()` for the
     * head's FIRST tile, then tile `index/32` of the head's own consecutive run, ROW 0. */
    for (head = 0; head < v_heads && placed; head++) {
        for (index = 0; index < state; index++) {
            const uint32_t tile =
                ftg_delta_head_tile(&shape, kGdnStreamV, head) + index / 32u;

            if (tile_element(staged, tile, (uint32_t)kGdnStripRow, index % 32u) !=
                300000.0f + (float)head * 1024.0f + (float)index)
                placed = 0;
        }
    }
    CHECK(placed, "v-head h's element j is at ftg_delta_head_tile(V, h) + j/32, row 0, "
                  "column j%32");

    /* The heads do not overlap: head h owns tiles [h*state/32, (h+1)*state/32) and NOTHING of
     * head h+1's run carries head h's tag. Reading the FIRST tile of every head's successor
     * and finding the successor's own value is what proves the runs are disjoint. */
    for (head = 0; head + 1u < v_heads && placed; head++) {
        const uint32_t next = ftg_delta_head_tile(&shape, kGdnStreamV, head + 1u);

        if (next - ftg_delta_head_tile(&shape, kGdnStreamV, head) != per_head)
            placed = 0;
        if (tile_element(staged, next, (uint32_t)kGdnStripRow, 0u) !=
            300000.0f + (float)(head + 1u) * 1024.0f)
            placed = 0;
    }
    CHECK(placed, "consecutive v-heads are state/32 tiles apart and head h+1's first tile "
                  "holds head h+1's own element 0 — the runs do not overlap");

    /* Rows 1-31 of a strip tile are never written and never read. A move that touched them
     * would be writing over the neighbouring 64 B granule. */
    for (index = 0; index < input_tiles && pad_rows_zero; index++) {
        uint32_t row;

        for (row = 1u; row < 32u && pad_rows_zero; row++) {
            if (tile_element(staged, index, row, 0u) != 0.0f ||
                tile_element(staged, index, row, 31u) != 0.0f)
                pad_rows_zero = 0;
        }
    }
    CHECK(pad_rows_zero, "rows 1-31 of every strip tile are untouched — the move writes only "
                         "the two 64 B chunks of row 0");

    /* The GQA repeat is the READER's modulo, so v-head 17 has to find k-head 1's FOUR TILES in
     * the UN-repeated k block — 16 heads' worth, not 32. */
    for (head = 0; head < v_heads && repeat_reads; head++) {
        const uint32_t k_head = ftg_k_head_for(head, k_heads);
        const uint32_t tile = ftg_delta_head_tile(&shape, kGdnStreamK, k_head);

        if (tile_element(staged, tile, (uint32_t)kGdnStripRow, 7u) !=
            200000.0f + (float)k_head * 1024.0f + 7.0f)
            repeat_reads = 0;
    }
    CHECK(repeat_reads, "v-head vh reads k-head vh%16's four tiles out of a 16-head k block — "
                        "the repeat is an address, never materialised");

    /* And it is NOT the blocked mapping, which is the plausible wrong answer with a recorded
     * gibberish bug behind it. */
    CHECK(ftg_k_head_for(17u, 16u) == 1u && 17u / (32u / 16u) == 8u,
          "v-head 17 pairs with k-head 1, not with k-head 8");

    /* q, k and v do not overlap, and the whole staging buffer is exactly 256 tiles at Ornith:
     * 16*4 + 16*4 + 32*4. It was 12 as three one-row-per-head matrices. */
    CHECK(ftg_delta_stream_base(&shape, kGdnStreamQ) == 0u &&
              ftg_delta_stream_base(&shape, kGdnStreamK) == 64u &&
              ftg_delta_stream_base(&shape, kGdnStreamV) == 128u && input_tiles == 256u,
          "q | k | v sit at tiles 0, 64, 128 and the staged input is 256 tiles");
    /* The last v-head's run ENDS on the buffer's last tile — no hole, nothing off the end. */
    CHECK(ftg_delta_head_tile(&shape, kGdnStreamV, v_heads - 1u) + per_head == input_tiles,
          "v-head 31's run ends exactly at tile 256 — the three blocks tile the buffer");

    free(q);
    free(k);
    free(v);
    free(staged);
}

static void test_y_strip(void)
{
    const ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);
    const uint32_t state = FT_TT_OP_GDN_QWEN35_STATE;
    const uint32_t per_head = state / 32u;
    const uint32_t v_heads = shape.v_heads;
    const uint32_t tiles = ftg_y_tile(&shape, v_heads - 1u, per_head - 1u) + 1u;
    float *y = malloc((size_t)v_heads * state * sizeof(float));
    float *back = calloc((size_t)v_heads * state, sizeof(float));
    unsigned char *staged = calloc(tiles, PN_TT_GDN_LAYOUT_TILE_BYTES);
    uint32_t head, index;
    int writer_places = 1;
    int injective = 1;
    unsigned char *seen = calloc(tiles, 1u);

    fprintf(stderr, "\ny — a head-major ROW-0 strip, 128 tiles at Ornith\n");

    CHECK(tiles == v_heads * per_head && tiles == 128u,
          "y is v_heads * state/32 = 128 tiles, not one shared row block of 4");

    /* The forward reads y back with the same move it wrote the input with — head h's four
     * tiles, in order. Round-tripping the whole strip head by head is the de-tilize that used
     * to be `pn_tt_gdn_unstage_stream()`. */
    fill_matrix(y, v_heads, state, 400000.0f);
    for (head = 0; head < v_heads; head++)
        strip_write(staged, ftg_y_tile(&shape, head, 0u), y + (size_t)head * state, per_head);
    for (head = 0; head < v_heads; head++)
        strip_read(staged, ftg_y_tile(&shape, head, 0u), back + (size_t)head * state, per_head);
    CHECK(memcmp(y, back, (size_t)v_heads * state * sizeof(float)) == 0,
          "write then read at ftg_y_tile(h, 0) is the identity on [32, 128] — the forward's "
          "own read-back path");

    /* The WRITER's own placement: head h's tile-row jt goes to `ftg_y_tile(h, jt)`, ROW 0.
     * Reading the strip at exactly those coordinates has to find y. */
    for (head = 0; head < v_heads && writer_places; head++) {
        for (index = 0; index < state; index++) {
            if (tile_element(staged, ftg_y_tile(&shape, head, index / 32u),
                             (uint32_t)kGdnStripRow, index % 32u) !=
                400000.0f + (float)head * 1024.0f + (float)index)
                writer_places = 0;
        }
    }
    CHECK(writer_places, "the writer's ftg_y_tile(h, jt) / row 0 finds y[h][32*jt + j]");

    /* INJECTIVITY, which is the property the old geometry did not have: every head of a row
     * block used to land on the SAME `state_tiles` tiles and was told apart by its row, so
     * (h, jt) -> tile was 32-to-1. On a strip there is no row to hide in, and two heads on one
     * tile would be one head's answer overwriting another's. */
    for (head = 0; head < v_heads && injective; head++) {
        uint32_t row;

        for (row = 0; row < per_head && injective; row++) {
            const uint32_t tile = ftg_y_tile(&shape, head, row);

            if (tile >= tiles || seen[tile])
                injective = 0;
            else
                seen[tile] = 1u;
        }
    }
    CHECK(injective, "(head, tile_row) -> y tile is INJECTIVE over all 32 x 4 — the old "
                     "one-row-per-head y collapsed all 32 heads onto tiles 0-3");

    free(seen);
    free(y);
    free(back);
    free(staged);
}

/** 64 v-heads — twice Ornith's. Under the old packing this was the case that needed a SECOND
 *  row block; on a strip it is simply twice as long, which is the point. */
static void test_sixty_four_heads(void)
{
    const uint32_t state = 128u;
    const uint32_t heads = 64u;
    const uint32_t per_head = state / 32u;
    ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);
    float *source = malloc((size_t)heads * state * sizeof(float));
    float *back = calloc((size_t)heads * state, sizeof(float));
    unsigned char *staged = calloc((size_t)heads * per_head, PN_TT_GDN_LAYOUT_TILE_BYTES);
    int round_trips = 1;
    int index;
    uint32_t head;

    fprintf(stderr, "\n64 heads — 256 tiles, not two row blocks of 4\n");
    shape.v_heads = heads;

    fill_matrix(source, heads, state, 500000.0f);
    for (head = 0; head < heads; head++)
        strip_write(staged, pn_tt_gdn_head_tile(head, state), source + (size_t)head * state,
                    per_head);
    for (head = 0; head < heads; head++)
        strip_read(staged, pn_tt_gdn_head_tile(head, state), back + (size_t)head * state,
                   per_head);
    if (memcmp(source, back, (size_t)heads * state * sizeof(float)) != 0)
        round_trips = 0;
    CHECK(round_trips, "write/read round-trips 64 heads across 256 tiles");

    /* Head 32 is at tile 128 and holds its OWN values — not row 0 of a second block at tile 4,
     * which is where the old arithmetic sent it. */
    index = 1;
    for (head = 32u; head < heads && index; head++) {
        uint32_t element;

        for (element = 0; element < state; element++) {
            if (tile_element(staged, ftg_y_tile(&shape, head, element / 32u),
                             (uint32_t)kGdnStripRow, element % 32u) !=
                500000.0f + (float)head * 1024.0f + (float)element)
                index = 0;
        }
    }
    CHECK(index && ftg_y_tile(&shape, 32u, 0u) == 128u,
          "head 32 is at tile 128 with its own values, not at tile 4 as row 0 of a second "
          "row block");

    free(source);
    free(back);
    free(staged);
}

int main(void)
{
    fprintf(stderr, "Gated-Delta-Net host stager — off-hardware layout acceptance (no device "
                    "is opened)\n");
    test_row0_offsets();
    test_row_offsets();
    test_conv_weight_transpose();
    test_head_tile_offsets();
    test_delta_input_layout();
    test_y_strip();
    test_sixty_four_heads();
    fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
