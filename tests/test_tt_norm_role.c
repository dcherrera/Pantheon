/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_norm_role.c
 * @brief The persistent norm role's host planning off the card: the step <-> norm map over a 64-layer token, the
 *        forward's step cursor over three tokens, and the orders it must refuse — a missed
 *        residual, a doubled residual, a step past the token (the OVERRUN) and a token that
 *        ends short. `NORM_ROLE_TEST_RED=overrun` builds a cursor check that lets the overrun
 *        through, and the test must fail.
 *
 * @status P0.
 * @issues None
 * @todo None
 */

#include "../src/accel/tt/pn_tt_norm_role_plan.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_checked = 0;

#define CHECK(cond, msg)                                                                     \
    do {                                                                                     \
        g_checked++;                                                                         \
        if (!(cond)) {                                                                       \
            g_failed++;                                                                      \
            fprintf(stderr, "FAIL: %s\n", msg);                                              \
        }                                                                                    \
    } while (0)

#define LAYERS 64u

/** One forward in the order qwen35_tt_layer / qwen35_tt_head run it. */
static int run_forward(pn_tt_norm_role_cursor *cursor, uint32_t *steps_seen)
{
    uint32_t step, layer;

    pn_tt_norm_role_cursor_begin(cursor, pn_tt_norm_role_steps(LAYERS));
    for (layer = 0; layer < LAYERS; layer++) {
        if (pn_tt_norm_role_cursor_norm(cursor, &step) != 0 || step != steps_seen[0]++)
            return -1;
        if (pn_tt_norm_role_cursor_residual(cursor) != 0 ||
            pn_tt_norm_role_cursor_norm(cursor, &step) != 0 || step != steps_seen[0]++ ||
            pn_tt_norm_role_cursor_residual(cursor) != 0)
            return -1;
    }
    if (pn_tt_norm_role_cursor_norm(cursor, &step) != 0 || step != steps_seen[0]++)
        return -1;
    return pn_tt_norm_role_cursor_end(cursor);
}

static void test_step_map(void)
{
    pn_tt_norm_role_norm norm;
    uint32_t layer, attn = 0u, post = 0u, output = 0u;

    CHECK(pn_tt_norm_role_steps(LAYERS) == 129u, "64 layers are 129 steps");
    for (uint32_t k = 0; k < 129u; k++) {
        if (!pn_tt_norm_role_step_norm(k, LAYERS, &norm, &layer))
            continue;
        attn += norm == PN_TT_NORM_ROLE_ATTN && layer == k / 2u;
        post += norm == PN_TT_NORM_ROLE_POST && layer == (k - 1u) / 2u;
        output += norm == PN_TT_NORM_ROLE_OUTPUT && k == 128u;
    }
    CHECK(attn == 64u && post == 64u && output == 1u, "every norm of the token, once, in order");
    CHECK(!pn_tt_norm_role_step_norm(129u, LAYERS, &norm, &layer), "step 129 is past the token");
    CHECK(pn_tt_norm_role_mode_from("1") == 1 && pn_tt_norm_role_mode_from("0") == 0 &&
              pn_tt_norm_role_mode_from(NULL) == 0 && pn_tt_norm_role_mode_from("2") == 0,
          "PN_TT_PERSIST_NORM: only 1 is on");
    CHECK(pn_tt_norm_role_exchange_from(NULL) == 0 && pn_tt_norm_role_exchange_from("") == 0 &&
              pn_tt_norm_role_exchange_from("faces,stagger") == 3 &&
              pn_tt_norm_role_exchange_from("whole") == 0 &&
              pn_tt_norm_role_exchange_from("faces") == 1 &&
              pn_tt_norm_role_exchange_from("stagger") == 2,
          "PN_TT_PERSIST_NORM_EXCHANGE: the four modes, default whole");
    CHECK(pn_tt_norm_role_exchange_from("Faces") == -1 &&
              pn_tt_norm_role_exchange_from("stagger,faces") == -1 &&
              pn_tt_norm_role_exchange_from("3") == -1,
          "PN_TT_PERSIST_NORM_EXCHANGE: anything else is refused");
}

static void test_cursor(void)
{
    pn_tt_norm_role_cursor cursor;
    uint32_t step, tokens_ok = 0u;

    for (uint32_t token = 0; token < 3u; token++) {
        uint32_t seen = 0u;

        tokens_ok += run_forward(&cursor, &seen) == 0 && seen == 129u;
    }
    CHECK(tokens_ok == 3u, "three tokens of norm, residual, norm, ... walk every step");

    pn_tt_norm_role_cursor_begin(&cursor, 129u);
    CHECK(pn_tt_norm_role_cursor_residual(&cursor) != 0, "a residual before step 0 is refused");
    CHECK(pn_tt_norm_role_cursor_norm(&cursor, &step) == 0 && step == 0u, "step 0 is a LOAD");
    CHECK(pn_tt_norm_role_cursor_norm(&cursor, &step) != 0,
          "a norm with no residual folded in (a missed add) is refused");
    CHECK(pn_tt_norm_role_cursor_residual(&cursor) == 0 &&
              pn_tt_norm_role_cursor_residual(&cursor) != 0,
          "two residuals in a row are refused");
    CHECK(pn_tt_norm_role_cursor_end(&cursor) != 0, "a token that ends short is refused");

    /* THE OVERRUN: a full token, then one more norm. */
    {
        uint32_t seen = 0u;
        int extra;

        (void)run_forward(&cursor, &seen);
#if defined(NORM_ROLE_TEST_RED_OVERRUN)
        cursor.steps++; /* a cursor that does not know where the token ends */
#endif
        (void)pn_tt_norm_role_cursor_residual(&cursor);
        extra = pn_tt_norm_role_cursor_norm(&cursor, &step);
        CHECK(extra != 0, "a step past the token's last (the overrun) is refused");
    }
}

int main(void)
{
    test_step_map();
    test_cursor();
    printf("test_tt_norm_role: %d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
