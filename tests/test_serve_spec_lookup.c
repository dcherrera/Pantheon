/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_serve_spec_lookup.c
 * @brief The serve's prompt-lookup drafts: matches, the minimum suffix, the cap, the history
 *        growth and the throttle. `--selftest` breaks the expected continuation and must fail.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/serve/serve_spec_lookup.h"

static int g_failed;

#define CHECK(cond, ...)                                                                    \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            g_failed++;                                                                     \
            fprintf(stderr, "  FAIL: " __VA_ARGS__);                                        \
            fprintf(stderr, "\n");                                                          \
        }                                                                                   \
    } while (0)

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    /* "a b c d e f g ... a b c" : the suffix a b c recurs, d e f g follow it. */
    const uint32_t prompt[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2 };
    uint32_t drafts[16];
    sg_spec_lookup lookup;
    int n, i;

    setenv("PN_TT_SPEC_LOOKUP", "1", 1);
    setenv("PN_TT_SPEC_LOOKUP_K", "4", 1);
    setenv("PN_TT_SPEC_LOOKUP_MIN", "3", 1);
    CHECK(sg_spec_lookup_init(&lookup, prompt, 14, 1) == 0 && lookup.on, "init");
    /* Suffix "1 2" is two tokens: below the minimum of three. */
    CHECK(sg_spec_lookup_draft(&lookup, 15, drafts) == 0, "a 2-token match drafted");
    sg_spec_lookup_push(&lookup, 3); /* grows past the initial capacity of 14 + 1 + 16 */
    n = sg_spec_lookup_draft(&lookup, 15, drafts);
    CHECK(n == 4, "drafts %d, want 4", n);
    for (i = 0; i < n; i++)
        CHECK(drafts[i] == (uint32_t)(4 + i + (selftest ? 1 : 0)), "draft %d = %u", i,
              drafts[i]);
    CHECK(sg_spec_lookup_draft(&lookup, 2, drafts) == 2, "the cap");
    for (i = 0; i < 40; i++)
        sg_spec_lookup_push(&lookup, 100u + (uint32_t)i);
    CHECK(lookup.count == 55 && lookup.on, "history grew to %d", lookup.count);
    /* The throttle: two lookup cycles accepting < 3 rest it for 16 cycles. */
    for (i = 0; i < 40; i++)
        sg_spec_lookup_push(&lookup, 1u + (uint32_t)(i % 3));
    CHECK(sg_spec_lookup_draft(&lookup, 15, drafts) > 0, "a repeating tail drafts");
    sg_spec_lookup_report(&lookup, 4, 0);
    sg_spec_lookup_report(&lookup, 4, 1);
    CHECK(sg_spec_lookup_draft(&lookup, 15, drafts) == 0, "throttled");
    CHECK(sg_spec_lookup_agrees(&lookup, 5u, 6u), "mode 1 takes lookup whatever MTP says");
    sg_spec_lookup_free(&lookup);
    setenv("PN_TT_SPEC_LOOKUP", "2", 1);
    CHECK(sg_spec_lookup_init(&lookup, prompt, 14, 1) == 0 && lookup.on, "mode 2 init");
    CHECK(sg_spec_lookup_agrees(&lookup, 5u, 5u) && !sg_spec_lookup_agrees(&lookup, 5u, 6u),
          "mode 2 takes lookup only when the first drafts agree");
    sg_spec_lookup_free(&lookup);
    unsetenv("PN_TT_SPEC_LOOKUP");
    CHECK(sg_spec_lookup_init(&lookup, prompt, 14, 1) == 0 && lookup.on && lookup.mode == 2,
          "mode 2 by default (David 2026-10-07)");
    sg_spec_lookup_free(&lookup);
    setenv("PN_TT_SPEC_LOOKUP", "0", 1);
    CHECK(sg_spec_lookup_init(&lookup, prompt, 14, 1) == 0 && !lookup.on, "off by env");
    CHECK(sg_spec_lookup_draft(&lookup, 15, drafts) == 0, "off drafts nothing");
    sg_spec_lookup_free(&lookup);
    unsetenv("PN_TT_SPEC_LOOKUP");
    /* The CUDA cycle's lookup: its own env, off by default, <= 4 drafts, suffix >= 5. */
    CHECK(sg_spec_lookup_init_cuda(&lookup, prompt, 14, 1) == 0 && !lookup.on, "cuda: off by default");
    setenv("QWEN35_MTP_LOOKUP", "1", 1);
    setenv("QWEN35_MTP_LOOKUP_K", "9", 1);
    CHECK(sg_spec_lookup_init_cuda(&lookup, prompt, 14, 1) == 0 && lookup.on && lookup.mode == 1 &&
          lookup.k == 4 && lookup.min_ngram == 5, "cuda: mode 1, k clamped to 4, min 5");
    sg_spec_lookup_free(&lookup);
    unsetenv("QWEN35_MTP_LOOKUP");
    unsetenv("QWEN35_MTP_LOOKUP_K");
    if (g_failed) {
        printf("=== %d FAILED ===\n", g_failed);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
