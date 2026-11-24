/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_attn_splitk.c — split-K decode attention's PLAN (pn_tt_attn_splitk.h), with NO CARD
 * and NO USE_TT.
 *
 * A chunk becomes a K / V base address and a live length on the card, so a plan that drops,
 * repeats or misplaces positions is a smoothly wrong attention nothing reports. Checked:
 *   - the knobs parse (unset = on), and with PN_TT_ATTN_SPLITK=0 no call at any depth is split
 *     (the lanes path, unchanged), nor one under the switch-over;
 *   - for a sweep of depths, cores, KV groups, blocks, verify rows and bank-run layouts, every
 *     plan passes pn_tt_attn_splitk_plan_check (cover once, in order, aligned, minimum, runs,
 *     causal limit, windows, counts), uses at most cores / groups chunks a group, and its
 *     slots cover every (group, chunk) once with contiguous merges;
 *   - a MERGE EMULATION in double: per-chunk (m, l, O) partials over the plan, folded in slot
 *     order as the merge descriptor folds them, equal the direct softmax to 1e-12 for every
 *     row's own causal limit;
 *   - negative controls: overlapping, gapped, run-crossing, misaligned, short and
 *     limit-missing plans are each refused by name, and a dropped partial breaks the merge.
 * Build with -DPN_TT_ATTN_SPLITK_RED (chunks overlap by one tile) to see it fail.
 */
#include "../src/accel/tt/pn_tt_attn_splitk.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DIM 32u
#define MAX_POS 262144u

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

static pn_tt_attn_splitk_config config_on(void)
{
    pn_tt_attn_splitk_config config;

    (void)pn_tt_attn_splitk_config_parse("1", NULL, NULL, &config);
    return config;
}

static void test_config(void)
{
    pn_tt_attn_splitk_config c;

    check(pn_tt_attn_splitk_config_parse(NULL, NULL, NULL, &c) == 0 && c.enabled &&
              c.min_positions == 8192u && c.min_chunk_positions == 1024u,
          "unset: on, defaults 8192 / 1024");
    check(pn_tt_attn_splitk_config_parse("", NULL, NULL, &c) == 0 && c.enabled, "empty: on");
    check(pn_tt_attn_splitk_config_parse("0", "", "", &c) == 0 && !c.enabled, "=0: off");
    check(pn_tt_attn_splitk_config_parse("1", "16384", "2048", &c) == 0 && c.enabled &&
              c.min_positions == 16384u && c.min_chunk_positions == 2048u,
          "=1 with both knobs");
    check(pn_tt_attn_splitk_config_parse("yes", NULL, NULL, &c) == -1 && !c.enabled,
          "a non-number refuses and stays off");
    check(pn_tt_attn_splitk_config_parse("1", NULL, "0", &c) == -1 && !c.enabled,
          "a zero minimum chunk refuses and stays off");
}

static pn_tt_attn_splitk_geometry geometry_of(uint32_t live, uint32_t shortest, uint32_t heads,
                                              uint32_t cores, uint32_t block,
                                              const pn_tt_attn_splitk_run *runs,
                                              uint32_t run_count)
{
    pn_tt_attn_splitk_geometry g;

    memset(&g, 0, sizeof(g));
    g.live_positions = live;
    g.shortest_live = shortest;
    g.kv_heads = heads;
    g.cores = cores;
    g.block_tiles = block;
    g.capacity_tiles = MAX_POS / 32u + 2u * block;
    g.runs = runs;
    g.run_count = run_count;
    return g;
}

static const pn_tt_attn_splitk_run k_one_run[1] = { { 0u, MAX_POS + 4096u } };

static void test_knob_off_is_lanes(void)
{
    pn_tt_attn_splitk_config off, on = config_on();
    pn_tt_attn_splitk_plan plan;
    int split = 0;

    (void)pn_tt_attn_splitk_config_parse("0", NULL, NULL, &off);
    for (uint32_t live = 1u; live <= MAX_POS; live += 4093u) {
        const pn_tt_attn_splitk_geometry g = geometry_of(live, live, 4u, 64u, 16u, k_one_run, 1u);

        check(pn_tt_attn_splitk_plan_make(&off, &g, &plan) == 0, "off: plan_make");
        split += plan.chunk_count != 0u;
    }
    check(split == 0, "PN_TT_ATTN_SPLITK=0: no depth splits (the lanes path)");
    for (uint32_t live = 1u; live < 8192u; live += 97u) {
        const pn_tt_attn_splitk_geometry g = geometry_of(live, live, 4u, 64u, 16u, k_one_run, 1u);

        (void)pn_tt_attn_splitk_plan_make(&on, &g, &plan);
        split += plan.chunk_count != 0u;
    }
    check(split == 0, "on: nothing under the 8,192 switch-over splits");
    {
        const pn_tt_attn_splitk_geometry g = geometry_of(8192u, 8192u, 4u, 16u, 16u, k_one_run, 1u);

        (void)pn_tt_attn_splitk_plan_make(&on, &g, &plan);
        check(plan.chunk_count == 4u && plan.partials == 16u && plan.rounds == 1u,
              "on at 8,192 over 16 cores and 4 groups: 4 chunks a group, one round");
    }
}

/** Every slot once, contiguous per group, cores within range, live lengths summing up. */
static void check_slots(const pn_tt_attn_splitk_geometry *g, const pn_tt_attn_splitk_plan *plan,
                        const char *what)
{
    uint32_t seen[PN_TT_ATTN_SPLITK_MAX_PARTIALS];
    uint32_t live_sum[PN_TT_ATTN_SPLITK_MAX_PARTIALS];
    int ok = 1;

    memset(seen, 0, sizeof(seen));
    memset(live_sum, 0, sizeof(live_sum));
    for (uint32_t d = 0; d < plan->partials; d++) {
        pn_tt_attn_splitk_slot s;

        if (pn_tt_attn_splitk_slot_of(g, plan, d, &s) != 0) {
            ok = 0;
            continue;
        }
        ok &= s.core < g->cores && s.round < plan->rounds && s.group < g->kv_heads;
        ok &= s.slot == s.group * plan->chunk_count + s.chunk; /* a merge's run */
        ok &= s.last == (s.chunk + 1u == plan->chunk_count);
        ok &= s.capacity_tiles + s.first_tile == g->capacity_tiles;
        seen[s.group * plan->chunk_count + s.chunk]++;
        live_sum[s.group] += s.kv_positions;
    }
    for (uint32_t i = 0; i < plan->partials; i++)
        ok &= seen[i] == 1u;
    for (uint32_t h = 0; h < g->kv_heads; h++)
        ok &= live_sum[h] == g->live_positions;
    ok &= pn_tt_attn_splitk_slot_of(g, plan, plan->partials, &(pn_tt_attn_splitk_slot){ 0 }) == -1;
    check(ok, what);
}

static void test_sweep(void)
{
    static const uint32_t lives[] = { 8192u, 8193u, 9000u, 15000u, 28672u, 28673u,
                                      33000u, 65536u, 131072u, 262144u };
    static const uint32_t cores[] = { 4u, 8u, 12u, 16u, 32u, 48u, 64u };
    static const uint32_t heads[] = { 1u, 2u, 4u, 8u };
    static const uint32_t blocks[] = { 8u, 16u };
    static const uint32_t verify[] = { 1u, 2u, 5u };
    static const pn_tt_attn_splitk_run k_runs[3] = { { 0u, 4096u },
                                                     { 4096u, 20480u },
                                                     { 20480u, MAX_POS + 4096u } };
    const pn_tt_attn_splitk_config on = config_on();
    uint32_t plans = 0u, split = 0u;
    char what[160];

    for (size_t a = 0; a < sizeof(lives) / sizeof(lives[0]); a++)
    for (size_t b = 0; b < sizeof(cores) / sizeof(cores[0]); b++)
    for (size_t c = 0; c < sizeof(heads) / sizeof(heads[0]); c++)
    for (size_t d = 0; d < sizeof(blocks) / sizeof(blocks[0]); d++)
    for (size_t e = 0; e < sizeof(verify) / sizeof(verify[0]); e++)
    for (uint32_t layout = 0; layout < 2u; layout++) {
        const uint32_t live = lives[a];
        const uint32_t shortest = live - verify[e] + 1u;
        const pn_tt_attn_splitk_geometry g =
            geometry_of(live, shortest, heads[c], cores[b], blocks[d],
                        layout ? k_runs : k_one_run, layout ? 3u : 1u);
        pn_tt_attn_splitk_plan plan;
        const char *broken;

        if ((shortest - 1u) / 32u != (live - 1u) / 32u)
            continue; /* the rows path refuses verifies straddling a tile */
        plans++;
        snprintf(what, sizeof(what), "live %u cores %u heads %u block %u verify %u runs %u",
                 live, cores[b], heads[c], blocks[d], verify[e], layout ? 3u : 1u);
        check(pn_tt_attn_splitk_plan_make(&on, &g, &plan) == 0, what);
        broken = pn_tt_attn_splitk_plan_check(&on, &g, &plan);
        if (broken)
            fprintf(stderr, "  %s: %s\n", what, broken);
        check(broken == NULL, what);
        if (plan.chunk_count == 0u) {
            check(cores[b] / heads[c] < 2u, what); /* only too few cores keeps lanes here */
            continue;
        }
        split++;
        check(layout == 1u || plan.chunk_count <= cores[b] / heads[c], what);
        check_slots(&g, &plan, what);
    }
    printf("  sweep: %u plans checked, %u split\n", plans, split);
    check(split > 0u, "the sweep split something");
}

/* ---- the merge emulation, in double ---- */

static float g_q[8u * DIM];
static float g_k[40000u * DIM];
static float g_v[40000u * DIM];

static uint64_t g_state = 1u;

static double random_unit(void)
{
    g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(g_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static double score(uint32_t row, uint32_t position)
{
    double s = 0.0;

    for (uint32_t e = 0; e < DIM; e++)
        s += (double)g_q[row * DIM + e] * (double)g_k[position * DIM + e];
    return s / sqrt((double)DIM);
}

typedef struct partial {
    double m, l, o[DIM];
} partial;

/** One chunk's (m, l, O) for `row`, over the chunk's positions below the row's limit. */
static void chunk_partial(uint32_t row, uint32_t first, uint32_t end, partial *out)
{
    out->m = -HUGE_VAL;
    out->l = 0.0;
    memset(out->o, 0, sizeof(out->o));
    for (uint32_t p = first; p < end; p++)
        if (score(row, p) > out->m)
            out->m = score(row, p);
    for (uint32_t p = first; p < end; p++) {
        const double w = exp(score(row, p) - out->m);

        out->l += w;
        for (uint32_t e = 0; e < DIM; e++)
            out->o[e] += w * (double)g_v[p * DIM + e];
    }
}

/** The plan's partials for `row` folded in slot order, as the MERGE descriptor folds them. */
static void merge_by_plan(const pn_tt_attn_splitk_geometry *g, const pn_tt_attn_splitk_plan *plan,
                          uint32_t row, uint32_t live, uint32_t drop, double *out)
{
    partial acc, part;
    int first = 1;

    for (uint32_t d = 0; d < plan->chunk_count; d++) { /* group 0's slots */
        pn_tt_attn_splitk_slot s;
        uint32_t end;

        (void)pn_tt_attn_splitk_slot_of(g, plan, d, &s);
        if (d == drop)
            continue;
        end = s.first_tile * 32u + s.kv_positions;
        chunk_partial(row, s.first_tile * 32u, s.last && live < end ? live : end, &part);
        if (first) {
            acc = part;
            first = 0;
            continue;
        }
        {
            const double m = part.m > acc.m ? part.m : acc.m;
            const double a = exp(acc.m - m), b = exp(part.m - m);

            acc.l = a * acc.l + b * part.l;
            for (uint32_t e = 0; e < DIM; e++)
                acc.o[e] = a * acc.o[e] + b * part.o[e];
            acc.m = m;
        }
    }
    for (uint32_t e = 0; e < DIM; e++)
        out[e] = acc.o[e] / acc.l;
}

static void direct(uint32_t row, uint32_t live, double *out)
{
    partial whole;

    chunk_partial(row, 0u, live, &whole);
    for (uint32_t e = 0; e < DIM; e++)
        out[e] = whole.o[e] / whole.l;
}

static double merge_error(const pn_tt_attn_splitk_geometry *g, const pn_tt_attn_splitk_plan *plan,
                          uint32_t rows, uint32_t drop)
{
    double worst = 0.0;

    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t live = g->live_positions - rows + r + 1u;
        double a[DIM], b[DIM];

        merge_by_plan(g, plan, r, live, drop, a);
        direct(r, live, b);
        for (uint32_t e = 0; e < DIM; e++)
            if (fabs(a[e] - b[e]) > worst)
                worst = fabs(a[e] - b[e]);
    }
    return worst;
}

static void test_merge_emulation(void)
{
    static const struct {
        uint32_t live, rows, cores;
    } cases[] = { { 8192u, 1u, 16u }, { 9000u, 2u, 12u }, { 28673u, 2u, 32u }, { 33000u, 5u, 64u },
                  { 33001u, 8u, 48u } };
    const pn_tt_attn_splitk_config on = config_on();
    char what[128];

    for (uint32_t i = 0; i < 8u * DIM; i++)
        g_q[i] = (float)(random_unit() * 3.0);
    for (uint32_t i = 0; i < 40000u * DIM; i++) {
        g_k[i] = (float)random_unit();
        g_v[i] = (float)random_unit();
    }
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const pn_tt_attn_splitk_geometry g =
            geometry_of(cases[c].live, cases[c].live - cases[c].rows + 1u, 4u, cases[c].cores,
                        16u, k_one_run, 1u);
        pn_tt_attn_splitk_plan plan;
        double error;

        (void)pn_tt_attn_splitk_plan_make(&on, &g, &plan);
        snprintf(what, sizeof(what), "merge emulation: live %u, %u row(s), %u chunks",
                 cases[c].live, cases[c].rows, plan.chunk_count);
        error = plan.chunk_count ? merge_error(&g, &plan, cases[c].rows, UINT32_MAX) : 1.0;
        printf("  %s: max abs %.3e\n", what, error);
        check(plan.chunk_count >= 2u && error < 1e-12, what);
        if (plan.chunk_count >= 2u) {
            snprintf(what, sizeof(what), "negative control: dropping chunk 1 of live %u breaks it",
                     cases[c].live);
            check(merge_error(&g, &plan, cases[c].rows, 1u) > 1e-6, what);
        }
    }
}

static void expect_broken(const pn_tt_attn_splitk_geometry *g, const pn_tt_attn_splitk_plan *plan,
                          const char *expected)
{
    const pn_tt_attn_splitk_config on = config_on();
    const char *broken = pn_tt_attn_splitk_plan_check(&on, g, plan);
    char what[160];

    snprintf(what, sizeof(what), "refused as \"%s\" (got \"%s\")", expected,
             broken ? broken : "accepted");
    check(broken != NULL && strcmp(broken, expected) == 0, what);
}

static void test_negative_controls(void)
{
    static const pn_tt_attn_splitk_run runs[2] = { { 0u, 16384u }, { 16384u, MAX_POS + 4096u } };
    const pn_tt_attn_splitk_config on = config_on();
    const pn_tt_attn_splitk_geometry g = geometry_of(28672u, 28672u, 4u, 16u, 16u, k_one_run, 1u);
    const pn_tt_attn_splitk_geometry g2 = geometry_of(28672u, 28672u, 4u, 16u, 16u, runs, 2u);
    pn_tt_attn_splitk_plan good, bad;

    (void)pn_tt_attn_splitk_plan_make(&on, &g, &good);
    check(good.chunk_count == 4u && pn_tt_attn_splitk_plan_check(&on, &g, &good) == NULL,
          "28,672 over 16 cores: 4 valid chunks");
    bad = good;
    bad.chunk[1].first_tile -= 1u;
    bad.chunk[1].tile_count += 1u;
    expect_broken(&g, &bad, "chunks overlap");
    bad = good;
    bad.chunk[2].first_tile += 32u;
    bad.chunk[2].tile_count -= 32u;
    expect_broken(&g, &bad, "a gap between chunks");
    bad = good;
    bad.chunk[0].tile_count -= 1u;
    bad.chunk[1].first_tile -= 1u;
    bad.chunk[1].tile_count += 1u;
    expect_broken(&g, &bad, "a chunk that is not whole double blocks");
    bad = good;
    bad.chunk_count = 3u;
    bad.partials = 12u;
    expect_broken(&g, &bad, "chunks that do not end at the live length");
    bad = good;
    bad.chunk[0].tile_count = 0u;
    expect_broken(&g, &bad, "an empty chunk");
    {
        /* 4 chunks of 224 tiles: chunk 2 spans tile 512 (position 16,384), the run edge. */
        pn_tt_attn_splitk_plan crossing = good;

        expect_broken(&g2, &crossing, "a chunk crossing a bank run");
        (void)pn_tt_attn_splitk_plan_make(&on, &g2, &crossing);
        check(pn_tt_attn_splitk_plan_check(&on, &g2, &crossing) == NULL &&
                  crossing.chunk_count == 5u,
              "with a run edge at 16,384 the planner cuts there (5 chunks)");
    }
    {
        const pn_tt_attn_splitk_geometry rows = geometry_of(28672u, 20000u, 4u, 16u, 16u,
                                                            k_one_run, 1u);

        expect_broken(&rows, &good, "a row's causal limit outside the last chunk");
    }
}

int main(void)
{
    test_config();
    test_knob_off_is_lanes();
    test_sweep();
    test_merge_emulation();
    test_negative_controls();
    printf("test_tt_attn_splitk: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
