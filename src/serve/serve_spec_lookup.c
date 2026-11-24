/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_lookup.c
 * @brief Prompt-lookup drafts for the serve's speculative decode. See serve_spec_lookup.h.
 *
 * @status Working off-hardware (tests/test_serve_spec_lookup.c).
 * @issues None
 * @todo None
 */

#include "serve_spec_lookup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Longest suffix tried first (higher precision). */
#define LOOKUP_NGRAM_MAX 6

/** The throttle: a lookup cycle accepting fewer drafts than this is a miss; after this many
 *  misses in a row lookup rests for this many cycles (the MTP head drafts meanwhile). */
#define LOOKUP_MIN_ACCEPT 3
#define LOOKUP_MISS_LIMIT 2
#define LOOKUP_COOL_CYCLES 16

static int lookup_env(const char *name, int fallback)
{
    const char *value = getenv(name);

    return value && *value ? atoi(value) : fallback;
}

/* Copy the prompt and arm `mode` with at most k drafts a cycle and suffix matches >= min. */
static int lookup_arm(sg_spec_lookup *lookup, const uint32_t *prompt, int n_prompt, int max_new,
                      int mode, int k, int min_ngram)
{
    memset(lookup, 0, sizeof(*lookup));
    lookup->mode = mode;
    if (lookup->mode <= 0 || !prompt || n_prompt < 0 || max_new < 0)
        return 0;
    lookup->k = k;
    lookup->min_ngram = min_ngram;
    if (lookup->k < 1 || lookup->min_ngram < 1)
        return 0;
    if (lookup->k > 15)
        lookup->k = 15;
    lookup->capacity = n_prompt + max_new + 16;
    lookup->history = malloc((size_t)lookup->capacity * sizeof(*lookup->history));
    if (!lookup->history)
        return -1;
    memcpy(lookup->history, prompt, (size_t)n_prompt * sizeof(*prompt));
    lookup->count = n_prompt;
    pn_spec_throttle_init(&lookup->throttle, LOOKUP_MIN_ACCEPT, LOOKUP_MISS_LIMIT,
                          LOOKUP_COOL_CYCLES);
    lookup->on = true;
    return 0;
}

int sg_spec_lookup_init(sg_spec_lookup *lookup, const uint32_t *prompt, int n_prompt,
                        int max_new)
{
    return lookup_arm(lookup, prompt, n_prompt, max_new, lookup_env("PN_TT_SPEC_LOOKUP", 2),
                      lookup_env("PN_TT_SPEC_LOOKUP_K", 6), lookup_env("PN_TT_SPEC_LOOKUP_MIN", 3));
}

int sg_spec_lookup_init_cuda(sg_spec_lookup *lookup, const uint32_t *prompt, int n_prompt,
                             int max_new)
{
    int k = lookup_env("QWEN35_MTP_LOOKUP_K", 4);

    if (k > 4)
        k = 4;   /* the CUDA verify takes <= 5 rows */
    return lookup_arm(lookup, prompt, n_prompt, max_new, lookup_env("QWEN35_MTP_LOOKUP", 0), k,
                      lookup_env("QWEN35_MTP_LOOKUP_MIN", 5));
}

bool sg_spec_lookup_agrees(const sg_spec_lookup *lookup, uint32_t looked_first,
                           uint32_t mtp_first)
{
    return lookup->mode != 2 || looked_first == mtp_first;
}

void sg_spec_lookup_push(sg_spec_lookup *lookup, uint32_t token)
{
    if (!lookup->on)
        return;
    if (lookup->count == lookup->capacity) {
        uint32_t *grown = realloc(lookup->history,
                                  (size_t)lookup->capacity * 2u * sizeof(*grown));

        if (!grown) {
            lookup->on = false;
            return;
        }
        lookup->history = grown;
        lookup->capacity *= 2;
    }
    lookup->history[lookup->count++] = token;
}

int sg_spec_lookup_draft(sg_spec_lookup *lookup, int cap, uint32_t *drafts)
{
    const int want = cap < lookup->k ? cap : lookup->k;
    const int longest = LOOKUP_NGRAM_MAX > lookup->min_ngram ? LOOKUP_NGRAM_MAX
                                                             : lookup->min_ngram;

    if (!lookup->on || want < 1 || !pn_spec_throttle_gate(&lookup->throttle))
        return 0;
    return pn_ngram_draft(lookup->history, lookup->count, longest, lookup->min_ngram, want,
                          drafts);
}

void sg_spec_lookup_report(sg_spec_lookup *lookup, int drafted, int accepted)
{
    lookup->cycles++;
    lookup->drafted += (uint64_t)drafted;
    lookup->accepted += (uint64_t)accepted;
    pn_spec_throttle_report(&lookup->throttle, accepted);
}

void sg_spec_lookup_free(sg_spec_lookup *lookup)
{
    if (lookup->cycles)
        fprintf(stderr, "serve: LOOKUP — %llu cycles, %llu drafted, %llu accepted (%.2f)\n",
                (unsigned long long)lookup->cycles, (unsigned long long)lookup->drafted,
                (unsigned long long)lookup->accepted,
                lookup->drafted ? (double)lookup->accepted / (double)lookup->drafted : 0.0);
    free(lookup->history);
    memset(lookup, 0, sizeof(*lookup));
}
