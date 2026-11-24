/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_bs_prefill_share.c — the decode share a yielding prefiller grants its
 * decoding neighbours (batch_sched_prefill_share.h): env parsing and the
 * stand-aside window.
 */

#include "../src/serve/batch_sched_prefill_share.h"

#include <math.h>
#include <stdio.h>

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { g_failed++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

static int near(double a, double b) { return fabs(a - b) < 1e-9; }

int main(void)
{
    CHECK(bs_prefill_decode_share_parse(NULL) == BS_PREFILL_DECODE_SHARE_DEFAULT,
          "unset -> default");
    CHECK(bs_prefill_decode_share_parse("") == BS_PREFILL_DECODE_SHARE_DEFAULT,
          "empty -> default");
    CHECK(bs_prefill_decode_share_parse("0") == 0, "0 -> off (one step per chunk)");
    CHECK(bs_prefill_decode_share_parse("-5") == 0, "negative clamps to 0");
    CHECK(bs_prefill_decode_share_parse("75") == 75, "75 kept");
    CHECK(bs_prefill_decode_share_parse("100") == BS_PREFILL_DECODE_SHARE_MAX,
          "100 clamps to max (prefill always progresses)");

    CHECK(near(bs_prefill_decode_window_ms(3000.0, 0), 0.0), "share 0 -> no window");
    CHECK(near(bs_prefill_decode_window_ms(0.0, 50), 0.0), "no hold -> no window");
    CHECK(near(bs_prefill_decode_window_ms(3000.0, 50), 3000.0), "50% -> equal time");
    CHECK(near(bs_prefill_decode_window_ms(3000.0, 75), 9000.0), "75% -> 3x the chunk");
    CHECK(near(bs_prefill_decode_window_ms(1000.0, 100), 9000.0), "share capped at 90%");

    printf("test_bs_prefill_share: %d/%d checks passed\n", g_checks - g_failed, g_checks);
    return g_failed ? 1 : 0;
}
