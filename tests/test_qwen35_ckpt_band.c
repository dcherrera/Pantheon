/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the multi-turn prefill checkpoint position (qwen35moe_checkpoint_band.c).
 *
 * The live defect (serve log, 2026-10-13): a 35,617-token prompt ending in the
 * template's generation prompt "<think>\n" (ids 248068 198) prefilled from a checkpoint at
 * 34,912 with its batched rows ending at 35,616 — a band multiple — so the checkpoint was taken
 * AT 35,616, covering "<think>". The next turn resent that turn without it (lcp 35,615), the
 * checkpoint could not be resumed, and the whole conversation re-prefilled (79 s). These tests
 * pin that the checkpoint never lands above the ceiling (the assistant header's end).
 */

#include "lib/test.h"
#include "../src/models/qwen35moe_checkpoint_band.h"

/* The recorded turn, without a ceiling: the checkpoint covers the generation prompt. */
static void test_live_turn_without_ceiling_covers_think(void)
{
    ASSERT_INT_EQ(pn_qwen35_ckpt_band_at(34912, 35616, 0), 35616);
}

/* With the ceiling at the header's end (35,617 - 2), it lands where the next turn can resume
 * (at or below its lcp, 35,615). */
static void test_live_turn_with_ceiling_is_resumable(void)
{
    int at = pn_qwen35_ckpt_band_at(34912, 35616, 35617 - 2);
    ASSERT_INT_EQ(at, 35584);
    ASSERT_TRUE(at <= 35615);
}

/* Every prompt length and base: never above the ceiling, never above the batched rows, always
 * a band boundary above base, and the LAST such boundary (nothing resumable is given away). */
static void test_band_at_is_last_boundary_at_or_below_both_limits(void)
{
    for (int base = 0; base <= 256; base += 32)
        for (int end = base; end <= base + 300; end++)
            for (int forced = 0; forced <= 6; forced++) {
                int ceiling = end + 1 - forced;   /* prompt n = end + 1, header end n - forced */
                int at = pn_qwen35_ckpt_band_at(base, end, ceiling);
                int limit = ceiling < end ? ceiling : end;
                if (at == 0) {
                    ASSERT_TRUE(limit - base < 32);
                    continue;
                }
                ASSERT_TRUE(at > base);
                ASSERT_INT_EQ(at % 32, 0);
                ASSERT_TRUE(at <= limit);
                ASSERT_TRUE(limit - at < 32);
            }
}

/* No ceiling (<= 0) is the old rule exactly. */
static void test_no_ceiling_is_the_old_rule(void)
{
    for (int end = 0; end < 200; end++) {
        int old = ((end / 32) * 32 > 0) ? (end / 32) * 32 : 0;
        ASSERT_INT_EQ(pn_qwen35_ckpt_band_at(0, end, 0), old);
        ASSERT_INT_EQ(pn_qwen35_ckpt_band_at(0, end, -5), old);
    }
}

int main(void)
{
    printf("\n  qwen35 checkpoint band\n\n");
    RUN_TEST(live_turn_without_ceiling_covers_think);
    RUN_TEST(live_turn_with_ceiling_is_resumable);
    RUN_TEST(band_at_is_last_boundary_at_or_below_both_limits);
    RUN_TEST(no_ceiling_is_the_old_rule);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
