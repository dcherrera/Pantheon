/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_exact_bound.c
 * @brief Gate for the grammar shortlist's OUTSIDE BOUND (grammar_mask_shortlist.h): a masked
 *        top-64 row it decides draws the token the plain path draws — the dense row under the
 *        full-vocabulary mask — for every seed.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Synthetic rows shaped like a tool call's structural steps: a heavy head, a handful of
 * allowed tokens inside the top-64 (often fewer than top_k), and allowed tokens outside
 * whose mass ranges from negligible to dominant (some sit just under the 64th logit), and
 * flat heads where the repeat penalty on recent allowed tokens drops them under the 64th.
 * Temperatures 0.3-1.2, top_k 40 (sometimes 20, 100 or off), top_p 0.8-0.99, the server's
 * repeat penalty 1.1 over a window that hits the head. Each row is drawn through the real
 * sampler (tests/grammar_exact/grammar_row_compare.c) at verify row 0 (every callback: the
 * card's tail counts, chunks, the dense refill) and as a verify row past 0 (no counts;
 * undecided = defer), 40 seeds each.
 *
 * PASS: 0 mismatches, and the bound decides draws on both routes while others still refill
 * or defer (a bound that never decides, or always does, fails).
 * `--red`: PN_TT_RED=grammar_bound_blind (the outside charged nothing) must mismatch.
 * `--selftest`: a top-N side that drops the grammar must mismatch (the comparison can fail).
 */
#include "grammar_exact/grammar_row_compare.h"
#include "../src/core/engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 20000
#define ROWS 250
#define SEEDS 40

static uint64_t g_rng = 0x7A3F1C2B9D4E5F61ull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 11); }
static float xf(void) { return (float)(xr() & 0xFFFFFF) / 16777216.0f; }

static float g_dense[VOCAB];
static unsigned char g_verdict[VOCAB];
static uint32_t g_recent[64];

/* The 64th largest logit (the top-N row's bound). */
static float topn_bound(void)
{
    static float sorted[VOCAB];
    memcpy(sorted, g_dense, sizeof(sorted));
    for (int k = 0; k < 64; k++)
        for (int i = k + 1; i < VOCAB; i++)
            if (sorted[i] > sorted[k]) { float t = sorted[i]; sorted[i] = sorted[k]; sorted[k] = t; }
    return sorted[63];
}

/* One structural-step row: allowed inside / outside, the outside's weight, the parameters. */
static void make_row(grc_row *r)
{
    const int inside = 1 + (int)(xr() % 30);
    const int heavy = xr() % 4 == 0;
    const int flat = xr() % 4 == 0;   /* a head within ~2 of the bound: penalties cross it */
    float bound;

    for (int i = 0; i < VOCAB; i++) g_dense[i] = 6.0f * xf() - 4.0f;
    for (int i = 0; i < 100; i++) g_dense[xr() % VOCAB] = 6.0f + (flat ? 2.0f : 12.0f) * xf();
    bound = topn_bound();
    memset(g_verdict, 0, sizeof(g_verdict));
    for (int i = 0; i < VOCAB; i++)
        if (g_dense[i] < bound && xr() % 1000 < (heavy ? 300u : 1u + xr() % 200u))
            g_verdict[i] = 1;
    for (int n = 0; n < inside; ) {   /* allowed head tokens: the inside set */
        const int i = (int)(xr() % VOCAB);
        if (g_dense[i] >= bound && !g_verdict[i]) { g_verdict[i] = 1; n++; }
    }
    if (heavy)   /* outside allowed tokens just under the bound: real mass outside */
        for (int n = 0; n < 200; n++) {
            const int i = (int)(xr() % VOCAB);
            if (g_dense[i] < bound) { g_dense[i] = bound - 0.5f * xf(); g_verdict[i] = 1; }
        }
    r->n_vocab = VOCAB;
    r->dense = g_dense;
    r->verdict = g_verdict;
    r->params = pn_sampling_defaults();
    r->params.temperature = 0.3f + 0.9f * xf();
    r->params.top_k = (int[]){ 40, 40, 40, 20, 100, 0 }[xr() % 6];
    r->params.top_p = (float[]){ 0.8f, 0.9f, 0.95f, 0.95f, 0.99f }[xr() % 5];
    r->n_recent = (int)(xr() % 65);
    for (int i = 0; i < r->n_recent; i++) {
        int head = (int)(xr() % VOCAB);
        for (int tries = 0; tries < 50 && g_dense[head] < bound; tries++) head = (int)(xr() % VOCAB);
        g_recent[i] = xr() % 2 ? (uint32_t)head : xr() % VOCAB;
    }
    if (flat)   /* recent ALLOWED head tokens: the repeat penalty drops some under the bound */
        for (int i = 0; i < r->n_recent; i++)
            for (int tries = 0; tries < 200; tries++) {
                const int id = (int)(xr() % VOCAB);
                if (g_verdict[id] && g_dense[id] >= bound) { g_recent[i] = (uint32_t)id; break; }
            }
    r->recent = g_recent;
    r->topn = 0;
}

int main(int argc, char **argv)
{
    const int red = argc > 1 && strcmp(argv[1], "--red") == 0;
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    grc_result row0 = { 0 }, later = { 0 };

    if (red) setenv("PN_TT_RED", "grammar_bound_blind", 1);
    if (grc_init(VOCAB) != 0) return 1;
    for (int t = 0; t < ROWS; t++) {
        grc_row r;
        make_row(&r);
        grc_compare(&r, GRC_ROW0, SEEDS, (uint64_t)t * 7919u, selftest, &row0);
        if (!selftest) grc_compare(&r, GRC_ROW_LATER, SEEDS, (uint64_t)t * 7919u, 0, &later);
    }
    printf("row 0      : %lu draws, %lu mismatches, bound %lu, shortlist %lu, chunks %lu, refills %lu\n",
           row0.draws, row0.mismatches, row0.bound, row0.shortlist, row0.chunk_steps, row0.refills);
    printf("verify row : %lu decided draws, %lu mismatches, bound %lu, deferred %lu\n",
           later.draws, later.mismatches, later.bound, later.deferred);
    const unsigned long wrong = row0.mismatches + later.mismatches;
    if (red || selftest) {
        printf(wrong ? "%s: caught (%lu mismatches)\n" : "%s: VACUOUS (0 mismatches)\n",
               red ? "RED grammar_bound_blind" : "SELFTEST grammar dropped", wrong);
        return wrong ? 0 : 1;
    }
    if (wrong || !row0.bound || !later.bound || !later.deferred ||
        !(row0.chunk_steps + row0.refills)) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
