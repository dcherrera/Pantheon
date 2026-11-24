/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_grid_marks.c — grid step marks (src/accel/tt/pn_tt_grid_marks.h), NO CARD.
 *
 * A wait that stops short of its projection lets the next step read rows the grid has not
 * written (a wrong answer nothing reports); a wait past it drains the grid for nothing; a ring
 * count that drifts overruns descriptors the card has not read. Replays seeded random streams
 * of projections (each over a random subset of 1..64 cores, depth 1..2, detached or chained),
 * marks after each, and takes, room checks and full waits in between, against a model that
 * keeps every projection's per-core counts. Checks: a take waits each core exactly to the
 * mark's count (never short, never past, never twice); a taken or covered mark needs no wait;
 * an evicted mark refuses; ring / detached in flight equal the model's; room names the OLDEST
 * mark that makes an issue fit, or "fits" / "full" correctly.
 * Build with -DPN_TT_GRID_MARKS_RED (take hands out the previous mark's counts) to see it fail.
 */
#include "../src/accel/tt/pn_tt_grid_marks.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed;

static void check(int ok, const char *what)
{
    if (!ok) {
        g_failed++;
        if (g_failed < 20)
            fprintf(stderr, "FAIL %s\n", what);
    }
}

static uint32_t g_seed;

static uint32_t rnd(uint32_t n)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (g_seed >> 8) % n;
}

#define MAXP 4096u

static struct {
    uint32_t count[MAXP][PN_TT_GRID_MARK_CORES]; /* each core's issued count after projection p */
    uint32_t ring[MAXP], detached[MAXP];         /* cumulative after p */
} g_model;

static void run(uint32_t seed, uint32_t cores, uint32_t limit)
{
    pn_tt_grid_marks m;
    uint32_t issued[PN_TT_GRID_MARK_CORES] = { 0 }, awaited[PN_TT_GRID_MARK_CORES] = { 0 };
    uint32_t need[PN_TT_GRID_MARK_CORES], ring = 0, det = 0, done_ring = 0, done_det = 0;
    uint32_t taken = 0; /* marks below are done */
    char what[96];

    g_seed = seed;
    snprintf(what, sizeof(what), "seed %u cores %u limit %u", seed, cores, limit);
    pn_tt_grid_marks_reset(&m, cores);
    for (uint32_t p = 0; p < 600u; p++) {
        const uint32_t depth = 1u + rnd(2u);
        const bool detached = rnd(4u) != 0u;
        const uint32_t room = pn_tt_grid_marks_room(&m, depth, 0u, limit);
        const uint32_t oldest = taken > (p >= PN_TT_GRID_MARKS ? p - PN_TT_GRID_MARKS : 0u)
                                    ? taken
                                    : (p >= PN_TT_GRID_MARKS ? p - PN_TT_GRID_MARKS : 0u);
        uint32_t id;

        check(m.oldest == oldest, what);
        /* Room: the oldest live mark whose take fits the issue. */
        if (ring - done_ring + depth <= limit && det - done_det < limit) {
            check(room == PN_TT_GRID_MARK_NONE, what);
        } else {
            uint32_t want = PN_TT_GRID_MARK_FULL;

            for (uint32_t q = oldest; q < p; q++)
                if (ring - g_model.ring[q] + depth <= limit && det - g_model.detached[q] < limit) {
                    want = q;
                    break;
                }
            check(room == want, what);
        }
        for (uint32_t c = 0; c < cores; c++)
            if (rnd(3u) != 0u)
                issued[c] += depth;
        ring += depth;
        det += detached ? 1u : 0u;
        pn_tt_grid_marks_issued(&m, depth, detached);
        id = pn_tt_grid_marks_push(&m, issued);
        check(id == p, what);
        memcpy(g_model.count[p], issued, sizeof(issued));
        g_model.ring[p] = ring;
        g_model.detached[p] = det;

        switch (rnd(8u)) {
        case 0: case 1: case 2: { /* take a recent mark (sometimes one already done) */
            const uint32_t back = rnd(4u), q = p >= back ? p - back : 0u;
            const bool evicted = q >= taken && p + 1u - q > PN_TT_GRID_MARKS;
            const int n = pn_tt_grid_marks_take(&m, q, need);
            int expect = 0;

            if (q < taken) {
                check(n == 0, what);
                break;
            }
            if (evicted) {
                check(n == -1, what);
                break;
            }
            for (uint32_t c = 0; c < cores; c++) {
                const uint32_t want = g_model.count[q][c] > awaited[c] ? g_model.count[q][c] : 0u;

                check(need[c] == want, what);
                if (want) {
                    awaited[c] = want;
                    expect++;
                }
            }
            check(n == expect, what);
            if (g_model.ring[q] > done_ring)
                done_ring = g_model.ring[q];
            if (g_model.detached[q] > done_det)
                done_det = g_model.detached[q];
            taken = q + 1u;
            break;
        }
        case 3: /* a full wait */
            if (rnd(4u) == 0u) {
                pn_tt_grid_marks_all(&m, issued);
                memcpy(awaited, issued, sizeof(issued));
                done_ring = ring;
                done_det = det;
                taken = p + 1u;
                check(pn_tt_grid_marks_take(&m, p, need) == 0, what);
            }
            break;
        default:
            break;
        }
        check(pn_tt_grid_marks_ring(&m) == ring - done_ring, what);
        check(pn_tt_grid_marks_detached(&m) == det - done_det, what);
    }
    /* Eviction: 17 marks without a take refuse the oldest. */
    {
        uint32_t first = 0;

        for (uint32_t i = 0; i <= PN_TT_GRID_MARKS; i++) {
            pn_tt_grid_marks_issued(&m, 1u, true);
            const uint32_t id = pn_tt_grid_marks_push(&m, issued);

            if (i == 0u)
                first = id;
        }
        check(pn_tt_grid_marks_take(&m, first, need) == -1, what);
        check(pn_tt_grid_marks_take(&m, PN_TT_GRID_MARK_NONE, need) == 0, what);
    }
}

int main(void)
{
    static const uint32_t cores[] = { 1u, 2u, 13u, 48u, 64u };
    static const uint32_t limits[] = { 2u, 6u, 12u };

    for (uint32_t seed = 1u; seed <= 40u; seed++)
        for (uint32_t i = 0; i < sizeof(cores) / sizeof(cores[0]); i++)
            for (uint32_t l = 0; l < sizeof(limits) / sizeof(limits[0]); l++)
                run(seed * 7919u + i, cores[i], limits[l]);
    printf("test_tt_grid_marks: %d failed\n", g_failed);
    return g_failed ? 1 : 0;
}
