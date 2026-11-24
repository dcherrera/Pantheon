/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_dispatch_shape_key.c — the dispatch replay across split-K's plan changes
 * (pn_tt_attn_splitk_plan_key, pn_tt_dispatch_shape_key), with NO CARD and NO USE_TT.
 *
 * THE WEDGE (q38-tt-regression-bisect, 2026-11-20): an 8k chat's turn-1 decode crossed
 * 8,192 positions under dispatch replay. The replay predicts each token's queue writes
 * linearly from two learning tokens and the puller writes the prediction BEFORE the host's
 * write is compared; split-K's switch-over is not linear, so the attention cores got the
 * lanes descriptor at forward k=339 ("DISPATCH REPLAY DIVERGED at forward k=339") and turn 2
 * found them stuck mid-descriptor.
 *
 * Modelled here: every attention descriptor field a decode token sends (lanes per group, or
 * split-K's PARTIAL slots and MERGEs), for the Qwen3.8 p150 geometry (12 members + 2 extras,
 * 4 KV groups, block 16, 30,720 positions); the replay's learning window (from forward 2 or
 * 4, the H slide on a shape change) and its linear prediction. Checked:
 *   - the shape key changes exactly at 8,192 and every 3,072 after 9,216, nowhere else;
 *   - WITHOUT the key, the failing transition (7,852-token prompt, 512 tokens, learning from
 *     forward 2) mispredicts at k=339, the card's own forward, and a decode from 9,000 at
 *     9,217;
 *   - WITH the key, every replayed token of a decode through all of 7,851..30,720 is
 *     predicted exactly (from 2 and from 4), re-anchoring once per change;
 *   - where replay worked before (2.9k, 28k), the key never changes and nothing re-anchors.
 * Build with -DPN_TT_ATTN_SPLITK_KEY_RED (the key ignores chunk lengths) to see it fail.
 */
#include "../src/accel/tt/pn_tt_attn_splitk.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CAPACITY_POSITIONS 30720u
#define FIELDS_MAX 512u
#define RELEARN_LIMIT 8u

static int g_failed;
static int g_checks;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failed++;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static const pn_tt_attn_splitk_run k_run[1] = { { 0u, CAPACITY_POSITIONS } };

/** Qwen3.8 p150 decode: one row at `live` over 14 split cores and 4 KV groups. */
static pn_tt_attn_splitk_geometry geometry_at(uint32_t live)
{
    pn_tt_attn_splitk_geometry g;

    memset(&g, 0, sizeof(g));
    g.live_positions = live;
    g.shortest_live = live;
    g.kv_heads = 4u;
    g.cores = 14u;
    g.block_tiles = 16u;
    g.capacity_tiles = CAPACITY_POSITIONS / 32u;
    g.runs = k_run;
    g.run_count = 1u;
    return g;
}

static pn_tt_attn_splitk_plan plan_at(uint32_t live)
{
    pn_tt_attn_splitk_config config;
    pn_tt_attn_splitk_geometry g = geometry_at(live);
    pn_tt_attn_splitk_plan plan;

    (void)pn_tt_attn_splitk_config_parse(NULL, NULL, NULL, &config);
    if (pn_tt_attn_splitk_plan_make(&config, &g, &plan) != 0)
        memset(&plan, 0, sizeof(plan));
    return plan;
}

static uint64_t key_at(uint32_t live)
{
    pn_tt_attn_splitk_plan plan = plan_at(live);

    return pn_tt_attn_splitk_plan_key(&plan);
}

/** The descriptor fields one decode token sends at `live`, in issue order. @return count */
static uint32_t token_fields(uint32_t live, uint32_t *fields)
{
    pn_tt_attn_splitk_geometry g = geometry_at(live);
    pn_tt_attn_splitk_plan plan = plan_at(live);
    uint32_t n = 0u;

    if (plan.chunk_count == 0u) {
        for (uint32_t group = 0; group < g.kv_heads; group++) {
            fields[n++] = group; /* core */
            fields[n++] = 0u;    /* flags: lanes */
            fields[n++] = 0u;    /* first tile */
            fields[n++] = live;  /* kv_positions */
            fields[n++] = g.capacity_tiles;
        }
        return n;
    }
    for (uint32_t d = 0; d < plan.partials; d++) {
        pn_tt_attn_splitk_slot slot;

        (void)pn_tt_attn_splitk_slot_of(&g, &plan, d, &slot);
        fields[n++] = slot.core;
        fields[n++] = 1u | (slot.last ? 2u : 0u); /* flags: PARTIAL, last */
        fields[n++] = slot.first_tile;
        fields[n++] = slot.kv_positions;
        fields[n++] = slot.capacity_tiles;
        fields[n++] = slot.slot;
    }
    for (uint32_t group = 0; group < g.kv_heads; group++) {
        fields[n++] = group;                        /* core */
        fields[n++] = 4u;                           /* flags: MERGE */
        fields[n++] = group * plan.chunk_count;     /* first slot */
        fields[n++] = plan.chunk_count;
    }
    return n;
}

typedef struct {
    uint32_t mispredicted_live; /* the first token the model got wrong, 0 = none */
    uint32_t mispredicted_k;    /* its forward offset from the model's first token */
    uint32_t replayed, reanchors;
} replay_result;

/**
 * The replay's rule over a decode of positions [first_pos, first_pos + tokens): forward f
 * (from 1) attends pos + 1 positions. Learning tokens are forwards sequence_first + from - 1
 * and the next; two of different shapes slide the window (H, at most 8); after them every
 * token is predicted word by word as a + (b - a) * k and replay stops at the first miss.
 * `use_key`: a key change re-anchors (sequence_first = f), as pn_tt_dispatch_token_begin.
 */
static replay_result replay(uint32_t first_pos, uint32_t tokens, uint32_t from, int use_key)
{
    static uint32_t model[2][FIELDS_MAX], current[FIELDS_MAX];
    uint32_t count[2] = { 0u, 0u }, relearns = 0u;
    uint64_t sequence_first = 1u, anchor = key_at(first_pos + 1u);
    replay_result r;
    int model_ok = 0;

    memset(&r, 0, sizeof(r));
    for (uint64_t f = 1u; f <= tokens; f++) {
        const uint32_t live = first_pos + (uint32_t)f;
        const uint64_t key = key_at(live);
        uint64_t first;
        uint32_t n;

        if (use_key && key != anchor) {
            anchor = key;
            sequence_first = f;
            model_ok = 0;
            relearns = 0u;
            r.reanchors++;
        }
        first = sequence_first + from - 1u;
        n = token_fields(live, current);
        if (f == first) {
            memcpy(model[0], current, n * sizeof(current[0]));
            count[0] = n;
        } else if (f == first + 1u) {
            memcpy(model[1], current, n * sizeof(current[0]));
            count[1] = n;
            model_ok = count[0] == count[1];
            if (!model_ok && relearns < RELEARN_LIMIT) {
                memcpy(model[0], model[1], n * sizeof(current[0]));
                count[0] = n;
                sequence_first++;
                relearns++;
            }
        } else if (f >= first + 2u && model_ok) {
            const uint32_t k = (uint32_t)(f - first);
            int hit = n == count[0];

            for (uint32_t w = 0; hit && w < n; w++)
                hit = model[0][w] + (model[1][w] - model[0][w]) * k == current[w];
            if (!hit) {
                r.mispredicted_live = live;
                r.mispredicted_k = k;
                return r;
            }
            r.replayed++;
        }
    }
    return r;
}

static void test_key_changes(void)
{
    static const uint32_t want[] = { 8192u, 9217u, 12289u, 15361u, 18433u, 21505u, 24577u,
                                     27649u };
    uint32_t got[16], n = 0u;
    uint64_t previous = key_at(1u);

    check(previous == 0u, "lanes depths key 0");
    for (uint32_t live = 2u; live <= CAPACITY_POSITIONS; live++) {
        const uint64_t key = key_at(live);

        if (key != previous && n < 16u)
            got[n++] = live;
        previous = key;
    }
    check(n == sizeof(want) / sizeof(want[0]), "eight key changes over 30,720 positions");
    for (uint32_t i = 0; i < n && i < sizeof(want) / sizeof(want[0]); i++) {
        if (got[i] != want[i])
            fprintf(stderr, "  key change %u at %u, want %u\n", i, got[i], want[i]);
        check(got[i] == want[i], "key change where the plan steps");
    }
    {
        pn_tt_attn_splitk_plan none;

        memset(&none, 0, sizeof(none));
        check(pn_tt_attn_splitk_plan_key(&none) == 0u && pn_tt_attn_splitk_plan_key(NULL) == 0u,
              "no plan, key 0");
    }
}

static void test_failing_transition_without_key(void)
{
    /* The card: 7,852-token prompt, batched to 7,851, decode from position 7,851, H2. */
    const replay_result t8 = replay(7851u, 512u, 2u, 0);
    const replay_result deep = replay(9000u, 400u, 4u, 0);

    if (t8.mispredicted_live != 8192u || t8.mispredicted_k != 339u)
        fprintf(stderr, "  t8: mispredicted at live %u k=%u\n", t8.mispredicted_live,
                t8.mispredicted_k);
    check(t8.mispredicted_live == 8192u && t8.mispredicted_k == 339u,
          "no key: the 8k chat mispredicts at live 8192, forward k=339 (the card's)");
    check(deep.mispredicted_live == 9217u, "no key: a decode from 9,000 mispredicts at 9,217");
}

static void test_key_replays_exactly(void)
{
    static const uint32_t froms[] = { 2u, 4u };

    for (unsigned i = 0; i < 2u; i++) {
        const replay_result t8 = replay(7851u, 512u, froms[i], 1);
        const replay_result all = replay(7851u, CAPACITY_POSITIONS - 7851u, froms[i], 1);

        if (all.mispredicted_live)
            fprintf(stderr, "  from %u: mispredicted at live %u\n", froms[i],
                    all.mispredicted_live);
        check(t8.mispredicted_live == 0u && t8.reanchors == 1u && t8.replayed > 400u,
              "key: the 8k chat replays every token after one re-anchor at 8192");
        check(all.mispredicted_live == 0u && all.reanchors == 8u,
              "key: 7,851..30,720 replays exactly, one re-anchor per plan change");
    }
}

static void test_unchanged_where_it_worked(void)
{
    const replay_result short_run = replay(2843u, 128u, 2u, 1);
    const replay_result long_run = replay(28255u, 64u, 2u, 1);
    const replay_result long_old = replay(28255u, 64u, 2u, 0);

    check(short_run.reanchors == 0u && short_run.mispredicted_live == 0u,
          "2.9k: lanes, no re-anchor");
    check(long_run.reanchors == 0u && long_run.mispredicted_live == 0u &&
              long_run.replayed == long_old.replayed,
          "28k: one plan, no re-anchor, the same tokens replayed as before");
}

int main(void)
{
    test_key_changes();
    test_failing_transition_without_key();
    test_key_replays_exactly();
    test_unchanged_where_it_worked();
    printf("test_tt_dispatch_shape_key: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
