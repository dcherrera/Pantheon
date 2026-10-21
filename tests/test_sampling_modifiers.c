/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_sampling_modifiers.c
 * @brief Unit tests for the OpenAI logits-row sampling controls
 *        (src/core/sampling_modifiers.c): `logit_bias`,
 *        `presence_penalty`/`frequency_penalty`, the seeded `seed` RNG, and
 *        `logprobs`/`top_logprobs` extraction.
 *
 * These assert on the ARITHMETIC and the ORDERING, not on a generated string —
 * the bug class this area already shipped once was an ordering bug (a penalty
 * applied on the wrong side of the greedy branch), which no end-to-end text
 * check would have localized.
 *
 * @status Complete for the four fields implemented.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "lib/test.h"
#include "../src/core/sampling_modifiers.h"

#define VOCAB 8

static void reset(float *logits)
{
    for (int i = 0; i < VOCAB; i++) logits[i] = 0.0f;
}

/* ── logit_bias ── */

static void test_logit_bias_adds_to_the_named_tokens(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[3] = 1.0f;
    const pn_logit_bias_entry bias[] = { { 3, -5.0f }, { 5, 2.5f } };
    pn_apply_logit_bias(logits, VOCAB, bias, 2);
    ASSERT_FLT_NEAR(logits[3], -4.0f, 1e-6f);
    ASSERT_FLT_NEAR(logits[5], 2.5f, 1e-6f);
    ASSERT_FLT_NEAR(logits[0], 0.0f, 1e-6f);   /* untouched */
}

static void test_logit_bias_ignores_out_of_range_ids(void)
{
    float logits[VOCAB];
    reset(logits);
    const pn_logit_bias_entry bias[] = { { VOCAB, 100.0f }, { 999999, -100.0f } };
    pn_apply_logit_bias(logits, VOCAB, bias, 2);
    for (int i = 0; i < VOCAB; i++) ASSERT_FLT_NEAR(logits[i], 0.0f, 1e-6f);
}

static void test_logit_bias_off_is_a_noop(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[2] = 7.0f;
    pn_apply_logit_bias(logits, VOCAB, NULL, 0);
    ASSERT_FLT_NEAR(logits[2], 7.0f, 1e-6f);
}

/* ── presence / frequency penalties ── */

static void test_frequency_penalty_scales_with_count(void)
{
    float logits[VOCAB];
    reset(logits);
    /* token 4 appears three times, token 6 once. */
    const uint32_t recent[] = { 4, 6, 4, 4 };
    pn_apply_presence_frequency_penalties(logits, VOCAB, 0.0f, 1.0f, recent, 4);
    ASSERT_FLT_NEAR(logits[4], -3.0f, 1e-6f);
    ASSERT_FLT_NEAR(logits[6], -1.0f, 1e-6f);
    ASSERT_FLT_NEAR(logits[0], 0.0f, 1e-6f);
}

static void test_presence_penalty_is_charged_once(void)
{
    float logits[VOCAB];
    reset(logits);
    const uint32_t recent[] = { 4, 4, 4 };
    pn_apply_presence_frequency_penalties(logits, VOCAB, 2.0f, 0.0f, recent, 3);
    ASSERT_FLT_NEAR(logits[4], -2.0f, 1e-6f);   /* NOT -6 */
}

static void test_both_penalties_compose(void)
{
    float logits[VOCAB];
    reset(logits);
    const uint32_t recent[] = { 1, 1 };
    pn_apply_presence_frequency_penalties(logits, VOCAB, 0.5f, 0.25f, recent, 2);
    /* 2 occurrences * 0.25 frequency + 1 * 0.5 presence */
    ASSERT_FLT_NEAR(logits[1], -1.0f, 1e-6f);
}

static void test_penalties_off_is_a_noop(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[1] = 3.0f;
    const uint32_t recent[] = { 1, 1 };
    pn_apply_presence_frequency_penalties(logits, VOCAB, 0.0f, 0.0f, recent, 2);
    ASSERT_FLT_NEAR(logits[1], 3.0f, 1e-6f);
    /* …and an empty window is a no-op even with penalties set. */
    pn_apply_presence_frequency_penalties(logits, VOCAB, 1.0f, 1.0f, NULL, 0);
    ASSERT_FLT_NEAR(logits[1], 3.0f, 1e-6f);
}

static void test_negative_penalty_boosts(void)
{
    /* OpenAI allows -2.0..2.0; a negative value must ENCOURAGE repetition. */
    float logits[VOCAB];
    reset(logits);
    const uint32_t recent[] = { 2 };
    pn_apply_presence_frequency_penalties(logits, VOCAB, -1.0f, 0.0f, recent, 1);
    ASSERT_FLT_NEAR(logits[2], 1.0f, 1e-6f);
}

/* ── seeded RNG ── */

static void test_same_seed_gives_the_same_stream(void)
{
    uint64_t a = pn_seed_to_rng_state(42);
    uint64_t b = pn_seed_to_rng_state(42);
    for (int i = 0; i < 64; i++) {
        float x = pn_seeded_rng_next_float(&a);
        float y = pn_seeded_rng_next_float(&b);
        ASSERT_FLT_NEAR(x, y, 0.0f);
    }
}

static void test_different_seeds_diverge_immediately(void)
{
    /* Neighbouring small seeds are the common case (0, 1, 42) — they must not
     * produce a correlated FIRST draw, which is what the splitmix64 diffuse in
     * pn_seed_to_rng_state is for. */
    uint64_t a = pn_seed_to_rng_state(0);
    uint64_t b = pn_seed_to_rng_state(1);
    float x = pn_seeded_rng_next_float(&a);
    float y = pn_seeded_rng_next_float(&b);
    ASSERT_TRUE(fabsf(x - y) > 0.01f);
}

static void test_seed_zero_is_not_a_degenerate_state(void)
{
    /* 0 is xorshift's fixed point: an undiffused seed of 0 would return 0.0
     * forever and pin every draw to token 0. */
    uint64_t s = pn_seed_to_rng_state(0);
    ASSERT_TRUE(s != 0);
    int nonzero = 0;
    for (int i = 0; i < 32; i++)
        if (pn_seeded_rng_next_float(&s) > 0.0f) nonzero++;
    ASSERT_TRUE(nonzero > 25);
}

static void test_rng_stays_in_unit_interval(void)
{
    uint64_t s = pn_seed_to_rng_state(12345);
    for (int i = 0; i < 4096; i++) {
        float v = pn_seeded_rng_next_float(&s);
        ASSERT_TRUE(v >= 0.0f);
        ASSERT_TRUE(v < 1.0f);
    }
}

/* ── logprobs ── */

static void test_logprobs_sum_to_one(void)
{
    float logits[VOCAB];
    for (int i = 0; i < VOCAB; i++) logits[i] = (float)i * 0.5f;
    uint32_t ids[PN_LOGPROBS_MAX_TOP];
    float    lps[PN_LOGPROBS_MAX_TOP];
    float    chosen = 0.0f;
    int n = pn_logprobs_extract(logits, VOCAB, 3, VOCAB, &chosen, ids, lps);
    ASSERT_INT_EQ(n, VOCAB);
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += expf(lps[i]);
    ASSERT_FLT_NEAR(total, 1.0f, 1e-4f);
    /* Every logprob is <= 0 (a probability). */
    for (int i = 0; i < n; i++) ASSERT_TRUE(lps[i] <= 1e-6f);
}

static void test_logprobs_are_descending_and_top_is_argmax(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[5] = 10.0f;
    logits[2] = 9.0f;
    logits[7] = 8.0f;
    uint32_t ids[PN_LOGPROBS_MAX_TOP];
    float    lps[PN_LOGPROBS_MAX_TOP];
    int n = pn_logprobs_extract(logits, VOCAB, 5, 3, NULL, ids, lps);
    ASSERT_INT_EQ(n, 3);
    ASSERT_INT_EQ((int)ids[0], 5);
    ASSERT_INT_EQ((int)ids[1], 2);
    ASSERT_INT_EQ((int)ids[2], 7);
    ASSERT_TRUE(lps[0] > lps[1]);
    ASSERT_TRUE(lps[1] > lps[2]);
}

static void test_chosen_logprob_matches_its_entry(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[5] = 10.0f;
    logits[2] = 9.0f;
    uint32_t ids[PN_LOGPROBS_MAX_TOP];
    float    lps[PN_LOGPROBS_MAX_TOP];
    float    chosen = 0.0f;
    /* Chosen is the RUNNER-UP: its reported logprob must equal its top-list
     * entry, not the argmax's. */
    pn_logprobs_extract(logits, VOCAB, 2, 2, &chosen, ids, lps);
    ASSERT_INT_EQ((int)ids[1], 2);
    ASSERT_FLT_NEAR(chosen, lps[1], 1e-6f);
}

static void test_logprobs_top_is_clamped(void)
{
    float logits[VOCAB];
    reset(logits);
    uint32_t ids[PN_LOGPROBS_MAX_TOP];
    float    lps[PN_LOGPROBS_MAX_TOP];
    /* Asking for more than the cap must not overrun the caller's arrays. */
    int n = pn_logprobs_extract(logits, VOCAB, 0, 10000, NULL, ids, lps);
    ASSERT_TRUE(n <= PN_LOGPROBS_MAX_TOP);
    ASSERT_INT_EQ(n, VOCAB);   /* vocab is smaller than the cap here */
}

static void test_logprobs_zero_top_still_reports_chosen(void)
{
    float logits[VOCAB];
    reset(logits);
    logits[4] = 5.0f;
    float chosen = 123.0f;
    int n = pn_logprobs_extract(logits, VOCAB, 4, 0, &chosen, NULL, NULL);
    ASSERT_INT_EQ(n, 0);
    ASSERT_TRUE(chosen < 0.0f);      /* a real logprob was written */
    ASSERT_TRUE(chosen > -1.0f);     /* and token 4 dominates, so it is near 0 */
}

static void test_logprobs_does_not_modify_the_row(void)
{
    float logits[VOCAB];
    for (int i = 0; i < VOCAB; i++) logits[i] = (float)i;
    float before[VOCAB];
    memcpy(before, logits, sizeof(before));
    uint32_t ids[PN_LOGPROBS_MAX_TOP];
    float    lps[PN_LOGPROBS_MAX_TOP];
    pn_logprobs_extract(logits, VOCAB, 1, 4, NULL, ids, lps);
    for (int i = 0; i < VOCAB; i++) ASSERT_FLT_NEAR(logits[i], before[i], 0.0f);
}

int main(void)
{
    printf("\n=== sampling modifiers (logit_bias / penalties / seed / logprobs) ===\n\n");
    RUN_TEST(logit_bias_adds_to_the_named_tokens);
    RUN_TEST(logit_bias_ignores_out_of_range_ids);
    RUN_TEST(logit_bias_off_is_a_noop);
    RUN_TEST(frequency_penalty_scales_with_count);
    RUN_TEST(presence_penalty_is_charged_once);
    RUN_TEST(both_penalties_compose);
    RUN_TEST(penalties_off_is_a_noop);
    RUN_TEST(negative_penalty_boosts);
    RUN_TEST(same_seed_gives_the_same_stream);
    RUN_TEST(different_seeds_diverge_immediately);
    RUN_TEST(seed_zero_is_not_a_degenerate_state);
    RUN_TEST(rng_stays_in_unit_interval);
    RUN_TEST(logprobs_sum_to_one);
    RUN_TEST(logprobs_are_descending_and_top_is_argmax);
    RUN_TEST(chosen_logprob_matches_its_entry);
    RUN_TEST(logprobs_top_is_clamped);
    RUN_TEST(logprobs_zero_top_still_reports_chosen);
    RUN_TEST(logprobs_does_not_modify_the_row);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
