/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ngram_draft.c
 * @brief Prompt-lookup n-gram drafter (see ngram_draft.h). A linear backward
 *        scan per n-gram length: history stays a plain array (a few thousand
 *        tokens at serving lengths), so a scan is ~microseconds against a
 *        ~50 ms K80 forward — not worth an index.
 *
 * @status Working; covered by tests/test_spec_qwen35.c on the K80 test machine.
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "ngram_draft.h"

/* 2. Standard library */
#include <string.h>

int pn_ngram_draft(const uint32_t *hist, int n_hist, int ngram_max,
                   int ngram_min, int max_k, uint32_t *out)
{
    if (!hist || !out || max_k <= 0 || ngram_min < 1 ||
        ngram_max < ngram_min)
        return 0;

    for (int g = ngram_max; g >= ngram_min; g--) {
        if (n_hist < g + 1)
            continue;
        const uint32_t *key = hist + n_hist - g;
        /* Most recent earlier occurrence; start excludes the trailing
         * self-match (start + g <= n_hist - 1 leaves >= 1 continuation). */
        for (int start = n_hist - g - 1; start >= 0; start--) {
            if (memcmp(hist + start, key, (size_t)g * sizeof(*hist)) != 0)
                continue;
            int avail = n_hist - (start + g);
            int k = avail < max_k ? avail : max_k;
            memcpy(out, hist + start + g, (size_t)k * sizeof(*hist));
            return k;
        }
    }
    return 0;
}

void pn_spec_throttle_init(pn_spec_throttle *t, int min_accept,
                           int miss_limit, int cool_steps)
{
    t->min_accept = min_accept > 0 ? min_accept : 4;
    t->miss_limit = miss_limit > 0 ? miss_limit : 2;
    t->cool_steps = cool_steps > 0 ? cool_steps : 48;
    t->miss = 0;
    t->cool = 0;
}

int pn_spec_throttle_gate(pn_spec_throttle *t)
{
    if (t->cool > 0) {
        t->cool--;
        return 0;
    }
    return 1;
}

void pn_spec_throttle_report(pn_spec_throttle *t, int accepted_drafts)
{
    if (accepted_drafts >= t->min_accept) {
        t->miss = 0;
        return;
    }
    if (++t->miss >= t->miss_limit) {
        t->miss = 0;
        t->cool = t->cool_steps;
    }
}
