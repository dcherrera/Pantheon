/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_name_list.c
 * @brief The FILTER and the SELECTOR disagree about the empty list and agree about everything
 *        else — which is the whole contract, and the one this backend has already paid for.
 *
 * `pn_tt_name_list_contains()` answers **true** for an unset list: an unset knob does not
 * restrict. `pn_tt_name_list_selects()` answers **false**: an unset knob arms nothing. The two
 * calls read identically at a call site, so nothing but a test can tell them apart — and using
 * the filter to arm `PN_TT_RED=residency_await_early` armed that red arm on EVERY run, refused
 * every descriptor of every family at token 0 with "completed_count does not match the number
 * enqueued", and cost four card sessions and a cross-agent bisect that landed on the wrong
 * commit. Silicon, 2026-08-25.
 *
 * No device, no allocation, no I/O.
 *
 * @status Green off-hardware.
 * @issues None
 * @todo None
 */

#include <stdio.h>

#include "../src/accel/tt/pn_tt_name_list.h"

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

int main(void)
{
    printf("tt name list — the filter and the selector, and where they must differ\n");

    /* THE ASYMMETRY. This pair is the entire reason the second function exists. */
    check(pn_tt_name_list_contains(NULL, "residency_await_early"),
          "FILTER: an unset list CONTAINS every name — an unset knob does not restrict");
    check(!pn_tt_name_list_selects(NULL, "residency_await_early"),
          "SELECTOR: an unset list SELECTS nothing — an unset knob arms nothing");
    check(pn_tt_name_list_contains("", "x"), "FILTER: an EMPTY list contains every name");
    check(!pn_tt_name_list_selects("", "x"), "SELECTOR: an EMPTY list selects nothing");

    /* AND THE RED ARM ITSELF, spelled the way the caller spells it. A regression that flips
     * this back to the filter fails here by name. */
    check(!pn_tt_name_list_selects(NULL, "residency_await_early") &&
              !pn_tt_name_list_selects("", "residency_await_early"),
          "RED ARM: `residency_await_early` is NOT armed when PN_TT_RED is unset or empty");
    check(pn_tt_name_list_selects("residency_await_early", "residency_await_early"),
          "RED ARM: it IS armed when named alone");
    check(pn_tt_name_list_selects("gdn_decay_raw,residency_await_early", "residency_await_early"),
          "RED ARM: and when named in a list");
    check(!pn_tt_name_list_selects("gdn_decay_raw,attn_group", "residency_await_early"),
          "RED ARM: and NOT when the list names other arms");

    /* ON A NON-EMPTY LIST THE TWO MUST NOT DIVERGE. If they do, the selector has grown its own
     * matcher and the pair can disagree about a membership as well as about the empty list. */
    {
        static const char *const lists[] = { "a", "a,b", "b,a", "render,render-widen",
                                             "aa", "a,,b", ",a" };
        static const char *const names[] = { "a", "b", "render", "render-widen", "aa", "" };
        unsigned li, ni;
        int agree = 1;

        for (li = 0; li < sizeof(lists) / sizeof(lists[0]); li++)
            for (ni = 0; ni < sizeof(names) / sizeof(names[0]); ni++)
                if (pn_tt_name_list_contains(lists[li], names[ni]) !=
                    pn_tt_name_list_selects(lists[li], names[ni]))
                    agree = 0;
        check(agree, "on every NON-EMPTY list the filter and the selector agree exactly");
    }

    /* The whole-word rule the filter documents must survive in the selector. */
    check(!pn_tt_name_list_selects("render-widen", "render"),
          "SELECTOR: `render` does not take `render-widen` — compared WHOLE");
    check(!pn_tt_name_list_selects("residency_await", "residency_await_early"),
          "SELECTOR: a PREFIX of the arm's name does not arm it");
    check(!pn_tt_name_list_selects("x", NULL) && !pn_tt_name_list_selects("x", ""),
          "SELECTOR: a NULL or empty name is never selected");

    printf("\n%d/%d checks failed\n", checks_failed, checks_run);
    return checks_failed == 0 ? 0 : 1;
}
