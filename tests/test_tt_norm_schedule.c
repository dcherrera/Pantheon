/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_norm_schedule.c — the P3 norm schedules (pn_tt_norm_schedule.h), scored with NO
 * CARD and NO USE_TT.
 *
 * WHAT IT PROVES:
 *
 *   1. OFF IS THE PRE-P3 SCHEDULE. `PN_TT_NORM_FAST=0` must put the pre-P3 descriptor and
 *      placement on the card byte for byte: depth = Wt, chunk 0, no stream, no half reads,
 *      default extents.
 *   2. THE CHUNK NEVER EXCEEDS THE SCRATCH IT IS PLACED WITH. The reader refuses a chunk past
 *      the scratch CBs' pages (a pack that waits forever otherwise), so a schedule that asked
 *      for one would turn every norm into a re-arm. Swept over widths, slices and knob values.
 *   3. THE QWEN3.8 SHAPES get the schedule the P3 design prices: model norm Wt 160 at depth
 *      16 (a divisor), K 16; head norms Wt 4 and 8 four rows deep, K a whole head.
 */

#include "../src/accel/tt/pn_tt_norm_schedule.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_checked = 0;

#define CHECK(cond, msg) do { \
    g_checked++; \
    if (!(cond)) { \
        g_failed++; \
        printf("  FAIL %s\n", (msg)); \
    } \
} while (0)

static bool is_pre_p3(const pn_tt_norm_schedule *s, uint32_t tile_cols)
{
    return s->prefetch_tiles == tile_cols && s->chain_chunk == 0u && s->stage_pages == 0u &&
           s->input_rows == 0u && !s->stream_input && !s->half_read;
}

static void test_off_is_the_pre_p3_schedule(void)
{
    pn_tt_norm_schedule s;
    uint32_t wt;
    bool all = true;

    for (wt = 1u; wt <= 320u; wt++) {
        pn_tt_norm_schedule_model(false, wt, 16u, 16u, &s);
        all = all && is_pre_p3(&s, wt) && pn_tt_norm_schedule_chunk_for(&s, wt) == 0u;
        pn_tt_norm_schedule_head(false, wt, &s);
        all = all && is_pre_p3(&s, wt) && pn_tt_norm_schedule_chunk_for(&s, wt) == 0u;
    }
    CHECK(all, "fast off: depth Wt, chunk 0, no stream/half, default extents, every Wt 1..320");
}

static void test_chunk_fits_its_scratch(void)
{
    static const uint32_t knobs[] = { 0u, 1u, 2u, 8u, 16u, 40u, 80u, 255u, 256u, 1000u };
    pn_tt_norm_schedule s;
    bool all = true;
    uint32_t wt, k, slice;

    for (wt = 1u; wt <= 320u; wt++) {
        for (k = 0; k < sizeof(knobs) / sizeof(knobs[0]); k++) {
            pn_tt_norm_schedule_model(true, wt, 16u, knobs[k], &s);
            all = all && s.chain_chunk >= 1u && s.chain_chunk <= PN_TT_NORM_CHUNK_MAX &&
                  s.chain_chunk <= s.stage_pages;
            for (slice = 1u; slice <= wt; slice++) {
                const uint32_t chunk = pn_tt_norm_schedule_chunk_for(&s, slice);

                all = all && chunk >= 1u && chunk <= s.stage_pages && chunk <= slice;
            }
        }
        pn_tt_norm_schedule_head(true, wt, &s);
        all = all && pn_tt_norm_schedule_chunk_for(&s, wt) <= s.stage_pages &&
              s.stage_pages <= PN_TT_NORM_CHUNK_MAX;
    }
    CHECK(all, "every chunk asked for is >= 1 and fits the scratch placed with it");
}

static void test_qwen38_shapes(void)
{
    pn_tt_norm_schedule s;

    pn_tt_norm_schedule_model(true, 160u, 0u, 0u, &s);
    CHECK(s.prefetch_tiles == 16u && 160u % s.prefetch_tiles == 0u,
          "model norm Wt 160: depth 16, a divisor of Wt");
    CHECK(s.chain_chunk == 16u && s.stage_pages == 16u, "model norm: K 16 on 16 scratch pages");
    CHECK(s.stream_input && s.half_read && s.input_rows == 0u,
          "model norm: streamed, half reads, CB_IN one row");
    CHECK(pn_tt_norm_schedule_chunk_for(&s, 80u) == 16u, "model norm, column slice 80: K 16");

    pn_tt_norm_schedule_head(true, 8u, &s);
    CHECK(s.prefetch_tiles == 8u && s.input_rows == PN_TT_HEAD_NORM_INPUT_ROWS,
          "attention head norm Wt 8: one block a head, four heads ahead");
    CHECK(pn_tt_norm_schedule_chunk_for(&s, 8u) == 8u && s.stage_pages == 8u,
          "attention head norm: the chain in runs of a whole head");
    pn_tt_norm_schedule_head(true, 4u, &s);
    CHECK(s.prefetch_tiles == 4u && pn_tt_norm_schedule_chunk_for(&s, 4u) == 4u &&
              s.stage_pages == 4u && s.stream_input && s.half_read,
          "GDN head norm Wt 4: one block a head, runs of 4, streamed, half reads");
    CHECK(pn_tt_norm_schedule_chunk_for(NULL, 4u) == 0u, "no schedule: per-tile chain");
}

int main(void)
{
    printf("test_tt_norm_schedule — the P3 norm schedules, off-hardware\n\n");
    test_off_is_the_pre_p3_schedule();
    test_chunk_fits_its_scratch();
    test_qwen38_shapes();
    printf("\n%d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
