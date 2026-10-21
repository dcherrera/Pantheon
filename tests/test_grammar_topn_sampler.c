/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_topn_sampler.c
 * @brief The grammar mask on a backend's TOP-N row draws the token the full
 *        mask draws on the dense row — same token, greedy and seeded sampling.
 *
 * @status Complete.  CPU-only — no device, no model.
 * @issues None
 * @todo None
 *
 * A Tenstorrent turn keeps top-N rows (the card's exact top 64) while a tool
 * grammar is armed, so its unmasked steps stay fast.  A masked step then has
 * three ways to its draw inside sample_token (engine.c):
 *   shortlist  — decided from the 64 alone (grammar_mask_shortlist.h);
 *   chunks     — fetch only the dense chunks holding allowed tokens
 *                (pn_sampling_params.dense_chunks);
 *   refill     — the whole dense row (dense_row), then the full mask.
 * Each must return the token the plain path returns: the full mask over the
 * dense row.  This drives sample_token through pn_sample_next_token with a fake
 * grammar hook whose allowed set is prefix-closed (as every real grammar walk
 * is: DEAD is absorbing), on random rows, at temperature 0 and at 0.7 with a
 * seeded RNG, and requires every path to have run.
 *
 * `--selftest` runs the comparison against a WRONG subject (the mask skipped on
 * the top-N row) and exits 0 only if that mismatch is detected.
 */
#include "../src/core/engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 20000
#define TOPN 64

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static char g_text[VOCAB][8];
static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}

/* The fake grammar: text = two letters + digits.  Allowed iff the first letter
 * is in A, the second in B, and (whole token) the number is not a multiple of
 * `drop`.  Prefix-closed by construction. */
typedef struct { unsigned a_mask, b_mask, drop; } fake_grammar;

static int fake_allowed(void *st, const char *t) {
    const fake_grammar *g = (const fake_grammar *)st;
    if (!t[0]) return 1;
    if (t[0] < 'a' || t[0] > 'z' || !(g->a_mask >> (t[0] - 'a') & 1u)) return 0;
    if (!t[1]) return 1;
    if (t[1] < 'a' || t[1] > 'z' || !(g->b_mask >> (t[1] - 'a') & 1u)) return 0;
    if (!t[2]) return 1;
    return (atoi(t + 2) % (int)g->drop) != 0;
}
static const char *fake_text(void *user, uint32_t id) {
    (void)user;
    return id < VOCAB ? g_text[id] : NULL;
}

/* end_allowed: "the turn may not end here" (a required walk before its call). */
static int fake_end_refused(void *st) { (void)st; return 0; }

static float g_dense[VOCAB];
static unsigned long g_chunks_read, g_rows_read;
static int dense_row_copy(void *ctx, float *logits, int n) {
    (void)ctx;
    memcpy(logits, g_dense, sizeof(float) * (size_t)n);
    g_rows_read++;
    return 0;
}
static int dense_chunks_copy(void *ctx, const unsigned char *need, float *logits, int n) {
    (void)ctx;
    for (int c = 0; c < (n + 15) / 16; c++) {
        if (!need[c]) continue;
        for (int i = c * 16; i < c * 16 + 16 && i < n; i++) logits[i] = g_dense[i];
        g_chunks_read++;
    }
    return 0;
}

static void make_topn(const float *dense, float *row) {
    static unsigned char taken[VOCAB];
    uint32_t neg = 0xff800000u;
    memset(taken, 0, sizeof(taken));
    for (int i = 0; i < VOCAB; i++) memcpy(&row[i], &neg, sizeof(float));
    for (int k = 0; k < TOPN; k++) {
        int best = -1;
        for (int i = 0; i < VOCAB; i++)
            if (!taken[i] && (best < 0 || dense[i] > dense[best])) best = i;
        taken[best] = 1;
        row[best] = dense[best];
    }
}

static int run(int selftest) {
    static float ref_row[VOCAB], row[VOCAB];
    uint32_t recent[32];
    int mismatches = 0, trials = 0, end_drawn = 0;
    unsigned long sl0 = pn_sampler_grammar_shortlist_steps();
    unsigned long ch0 = pn_sampler_grammar_chunk_steps();
    for (int t = 0; t < 600; t++) {
        fake_grammar g = { rnd() | 1u, (t % 3 == 0) ? 0x3FFFFFFu : (rnd() | 1u),
                           2u + rnd() % 5u };
        for (int i = 0; i < VOCAB; i++)
            g_dense[i] = (float)(rnd() % 100000) / 10000.0f - 4.0f;
        for (int j = 0; j < 120; j++) g_dense[rnd() % VOCAB] = 6.0f + (float)(rnd() % 1000) / 80.0f;
        int n_recent = (int)(rnd() % 32);
        for (int i = 0; i < n_recent; i++) recent[i] = rnd() % VOCAB;
        pn_tool_grammar_hook hook;
        memset(&hook, 0, sizeof(hook));
        hook.token_text = fake_text;
        hook.allowed = fake_allowed;
        hook.st = &g;
        hook.n_vocab = VOCAB;
        hook.active = 1;
        const bool end_case = (t % 5 == 0);   /* id 0 is the end token (no text) */
        if (end_case) {
            hook.end_allowed = fake_end_refused;
            g_dense[0] = 60.0f;                /* the model wants to stop */
        }
        pn_sampling_params ref = pn_sampling_defaults();
        ref.temperature = (t % 2) ? 0.7f : 0.0f;
        ref.top_k = 40;
        ref.top_p = 0.95f;
        ref.tool_grammar = &hook;
        ref.rng_seeded = 1;
        ref.rng_state = 0x9E3779B97F4A7C15ull ^ (uint64_t)t;
        pn_sampling_params sub = ref;
        sub.dense_row = dense_row_copy;
        sub.dense_chunks = (t % 4 == 3) ? NULL : dense_chunks_copy;   /* some refills too */
        sub.dense_ctx = NULL;
        if (selftest) sub.tool_grammar = NULL;
        memcpy(ref_row, g_dense, sizeof(ref_row));
        make_topn(g_dense, row);
        uint32_t a = pn_sample_next_token(ref_row, VOCAB, &ref, recent, n_recent);
        uint32_t b = pn_sample_next_token(row, VOCAB, &sub, recent, n_recent);
        trials++;
        if (a != b) mismatches++;
        /* (A grammar that allows nothing at all fails open by design.) */
        bool any_allowed = false;
        for (int i = 1; i < VOCAB && !any_allowed; i++)
            any_allowed = fake_allowed(&g, g_text[i]) != 0;
        if (end_case && any_allowed && !selftest && (a == 0 || b == 0)) end_drawn++;
    }
    unsigned long shortlisted = pn_sampler_grammar_shortlist_steps() - sl0;
    unsigned long chunked = pn_sampler_grammar_chunk_steps() - ch0;
    printf("  trials=%d mismatches=%d shortlist_steps=%lu chunk_steps=%lu whole_rows=%lu "
           "chunks_read=%lu (whole row = %d chunks)\n", trials, mismatches, shortlisted, chunked,
           g_rows_read, g_chunks_read, VOCAB / 16);
    if (selftest) return mismatches > 0 ? 0 : 1;
    CHECK(mismatches == 0, "top-N subject drew a different token in %d of %d", mismatches, trials);
    CHECK(shortlisted > 0, "the shortlist path never ran");
    CHECK(chunked > 0, "the chunk path never ran");
    CHECK(g_rows_read > 0, "the whole-row refill never ran");
    CHECK(end_drawn == 0, "an end token was drawn %d times where end_allowed refused it", end_drawn);
    return 0;
}

int main(int argc, char **argv) {
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    for (int i = 0; i < VOCAB; i++)
        snprintf(g_text[i], sizeof(g_text[i]), "%c%c%d", 'a' + (i * 7) % 26, 'a' + (i * 11) % 26,
                 i % 1000);
    g_text[0][0] = '\0';   /* id 0: an end token, no text */
    printf("=== grammar mask on top-N rows%s ===\n", selftest ? " (selftest)" : "");
    int rc = run(selftest);
    if (selftest) {
        printf(rc == 0 ? "selftest: mismatch detected (good)\n"
                       : "selftest: VACUOUS — a wrong subject matched\n");
        return rc;
    }
    printf("%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
