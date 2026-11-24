/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_core_list.c — the "96-110,0-15" core-list parser (pn_tt_core_list.h), NO CARD.
 *
 * A wrong list boots matmul members on cores other families hold, or drops one silently, so
 * every malformed, reversed, repeated, out-of-range and over-long list must be refused, and a
 * good one must come back in the order written. Build with -DPN_TT_CORE_LIST_RED to see the
 * order check fail (the expected list is reversed).
 */
#include "../src/accel/tt/pn_tt_core_list.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed, g_run;

static void check(int ok, const char *what)
{
    g_run++;
    if (!ok) {
        g_failed++;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static void check_refused(const char *text, uint32_t capacity, const char *what)
{
    uint32_t cores[64];

    check(pn_tt_core_list_parse(text, cores, capacity, 139u) == -1, what);
}

int main(void)
{
    uint32_t cores[64];
    int count = pn_tt_core_list_parse("96-110,0-15", cores, 64u, 139u);
    int ordered = count == 31;

    for (uint32_t index = 0; ordered && index < 31u; index++) {
#ifdef PN_TT_CORE_LIST_RED
        const uint32_t want = index < 15u ? 110u - index : 15u - (index - 15u);
#else
        const uint32_t want = index < 15u ? 96u + index : index - 15u;
#endif
        ordered = cores[index] == want;
    }
    check(ordered, "96-110,0-15 is 31 cores in the order written");
    check(pn_tt_core_list_parse("7", cores, 64u, 139u) == 1 && cores[0] == 7u, "one index");
    check(pn_tt_core_list_parse("", cores, 64u, 139u) == 0, "empty text is an empty list");
    check(pn_tt_core_list_parse(NULL, cores, 64u, 139u) == 0, "NULL is an empty list");
    check(pn_tt_core_list_parse("139", cores, 64u, 139u) == 1, "the largest index is legal");
    check_refused("140", 64u, "an index past max_index");
    check_refused("10-5", 64u, "a reversed range");
    check_refused("3,4,3", 64u, "a repeated index");
    check_refused("0-3,2", 64u, "an index repeated inside a range");
    check_refused("1,,2", 64u, "an empty entry");
    check_refused("1,2,", 64u, "a trailing comma");
    check_refused("a", 64u, "not a number");
    check_refused("1 2", 64u, "a space separator");
    check_refused("-3", 64u, "a range with no start");
    check_refused("0-4", 4u, "longer than the capacity");
    check(pn_tt_core_list_parse("0-3", cores, 4u, 139u) == 4, "exactly the capacity fits");
    printf("test_tt_core_list: %d checks, %d failed\n", g_run, g_failed);
    return g_failed ? 1 : 0;
}
