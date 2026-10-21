/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_column_runs.c — the column-tile permutation that de-interleaves Ornith's joint
 * Q+gate projection, scored with NO CARD and NO USE_TT.
 *
 * WHAT IT PROVES, and why each part is here rather than left to the card:
 *
 *   1. The FIELD LIST is Ornith's. 32 heads x 16 column tiles a head, Q at offset 0 and the
 *      gate at offset 8, each 8 wide. Both halves are 256 destination columns and together
 *      they are a PARTITION of the 512-column source — checked as a partition, not as two
 *      counts that happen to add up.
 *
 *   2. The MAPPING agrees with a brute-force model built the other way round. A permutation
 *      that drops one column and repeats another has the right total and the right shape and
 *      is a smoothly wrong projection on the card, where nothing can see it.
 *
 *   3. The CONTIGUOUS RUN LENGTH is right, because it is what turns the permutation into 32
 *      transfers instead of 256. A `*out_contiguous` that is too LARGE is a read across a
 *      run boundary — the source of a whole head's wrong weights — and a value that is too
 *      small is only slow, so the two failures are asymmetric and both are checked.
 *
 *   4. The REFUSALS fire: a run past the end, two runs that overlap, a zero-width run, a
 *      field wider than its record.
 *
 * The de-interleave is the one item of Phase 5 with no existing shape to inherit, so it is
 * the one whose arithmetic has to be settled before a model is ever loaded.
 */

#include "../src/accel/tt/pn_tt_column_runs.h"

#include <stdio.h>
#include <string.h>

/* Ornith-9B: 32 Q heads, head_dim 256 (8 column tiles), a joint Q+gate projection. */
#define ORNITH_HEADS       32u
#define ORNITH_HEAD_TILES  8u
#define ORNITH_JOINT_TILES (2u * ORNITH_HEAD_TILES)
#define ORNITH_SOURCE_COLS (ORNITH_HEADS * ORNITH_JOINT_TILES)

static int g_failed = 0;
static int g_checked = 0;

#define CHECK(cond, msg) do { \
    g_checked++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

/** The de-interleave, written the OTHER way round: destination column `d` of the field at
 *  `offset` is source column `(d / width) * stride + offset + (d % width)`. If this and the
 *  run list ever disagree, one of them is the defect and the test says which column. */
static uint32_t model_source_column(uint32_t destination, uint32_t stride, uint32_t offset,
                                    uint32_t width)
{
    return (destination / width) * stride + offset + (destination % width);
}

static void test_ornith_field_lists(void)
{
    pn_tt_column_run q[PN_TT_COLUMN_RUNS_MAX];
    pn_tt_column_run gate[PN_TT_COLUMN_RUNS_MAX];
    unsigned char seen[ORNITH_SOURCE_COLS];
    uint32_t q_runs, gate_runs, index;

    printf("Ornith's joint Q+gate projection\n");

    q_runs = pn_tt_column_runs_field(ORNITH_HEADS, ORNITH_JOINT_TILES, 0u, ORNITH_HEAD_TILES,
                                     q, PN_TT_COLUMN_RUNS_MAX);
    gate_runs = pn_tt_column_runs_field(ORNITH_HEADS, ORNITH_JOINT_TILES, ORNITH_HEAD_TILES,
                                        ORNITH_HEAD_TILES, gate, PN_TT_COLUMN_RUNS_MAX);
    CHECK(q_runs == ORNITH_HEADS && gate_runs == ORNITH_HEADS,
          "each half is one run per head — 32 transfers, not 256");
    CHECK(pn_tt_column_runs_total(q, q_runs) == ORNITH_HEADS * ORNITH_HEAD_TILES,
          "the Q half is 256 destination column tiles (8192 features)");
    CHECK(pn_tt_column_runs_total(gate, gate_runs) == ORNITH_HEADS * ORNITH_HEAD_TILES,
          "the gate half is 256 destination column tiles");
    CHECK(pn_tt_column_runs_are_injective(q, q_runs, ORNITH_SOURCE_COLS),
          "the Q half uses no source column twice");
    CHECK(pn_tt_column_runs_are_injective(gate, gate_runs, ORNITH_SOURCE_COLS),
          "the gate half uses no source column twice");

    /* A PARTITION, not two halves that add up. Two lists can both be injective, both total
     * 256, and both miss the same column. */
    memset(seen, 0, sizeof(seen));
    for (index = 0; index < ORNITH_HEADS * ORNITH_HEAD_TILES; index++) {
        uint32_t source, contiguous;

        if (pn_tt_column_runs_map(q, q_runs, index, &source, &contiguous) &&
            source < ORNITH_SOURCE_COLS)
            seen[source]++;
        if (pn_tt_column_runs_map(gate, gate_runs, index, &source, &contiguous) &&
            source < ORNITH_SOURCE_COLS)
            seen[source]++;
    }
    {
        bool partition = true;

        for (index = 0; index < ORNITH_SOURCE_COLS; index++) {
            if (seen[index] != 1u)
                partition = false;
        }
        CHECK(partition, "Q and gate PARTITION the 512-column joint weight — every source "
                         "column lands in exactly one half");
    }

    /* The first head, spelled out, because a permutation that is right on average is the
     * failure this whole module exists to make impossible. */
    CHECK(q[0].first == 0u && q[0].count == 8u && q[1].first == 16u && q[1].count == 8u,
          "Q head 0 is source tiles 0..7 and Q head 1 is 16..23");
    CHECK(gate[0].first == 8u && gate[0].count == 8u && gate[1].first == 24u,
          "gate head 0 is source tiles 8..15 and gate head 1 starts at 24");
    CHECK(q[ORNITH_HEADS - 1u].first == 496u && gate[ORNITH_HEADS - 1u].first == 504u,
          "the last head's Q starts at 496 and its gate at 504");
}

static void test_mapping_against_the_model(void)
{
    pn_tt_column_run runs[PN_TT_COLUMN_RUNS_MAX];
    uint32_t run_count;
    uint32_t offset;
    bool agrees = true;
    bool contiguity_agrees = true;

    printf("the mapping, against a brute-force model built the other way round\n");

    for (offset = 0u; offset <= ORNITH_HEAD_TILES; offset += ORNITH_HEAD_TILES) {
        uint32_t destination;

        run_count = pn_tt_column_runs_field(ORNITH_HEADS, ORNITH_JOINT_TILES, offset,
                                            ORNITH_HEAD_TILES, runs, PN_TT_COLUMN_RUNS_MAX);
        for (destination = 0; destination < ORNITH_HEADS * ORNITH_HEAD_TILES; destination++) {
            uint32_t source = 0u, contiguous = 0u;
            uint32_t want_source = model_source_column(destination, ORNITH_JOINT_TILES, offset,
                                                       ORNITH_HEAD_TILES);
            uint32_t want_contiguous =
                ORNITH_HEAD_TILES - (destination % ORNITH_HEAD_TILES);

            if (!pn_tt_column_runs_map(runs, run_count, destination, &source, &contiguous) ||
                source != want_source)
                agrees = false;
            if (contiguous != want_contiguous)
                contiguity_agrees = false;
            /* The run length must never claim more than the source actually has in a row:
             * an over-long range is a read across a head boundary. */
            if (contiguous == 0u ||
                model_source_column(destination + contiguous - 1u, ORNITH_JOINT_TILES, offset,
                                    ORNITH_HEAD_TILES) != source + contiguous - 1u)
                contiguity_agrees = false;
        }
    }
    CHECK(agrees, "every one of the 512 destination columns maps where the model says");
    CHECK(contiguity_agrees, "and the contiguous run length is exact — never past a head "
                             "boundary, never short of one");

    {
        uint32_t source, contiguous;

        run_count = pn_tt_column_runs_field(ORNITH_HEADS, ORNITH_JOINT_TILES, 0u,
                                            ORNITH_HEAD_TILES, runs, PN_TT_COLUMN_RUNS_MAX);
        CHECK(!pn_tt_column_runs_map(runs, run_count, ORNITH_HEADS * ORNITH_HEAD_TILES,
                                     &source, &contiguous),
              "one destination column past the end is refused, not wrapped");
    }
}

static void test_refusals(void)
{
    pn_tt_column_run runs[4];
    uint32_t source, contiguous;

    printf("the refusals\n");

    runs[0].first = 0u;
    runs[0].count = 0u;
    CHECK(pn_tt_column_runs_total(runs, 1u) == 0u, "a zero-width run is refused");
    CHECK(!pn_tt_column_runs_map(runs, 1u, 0u, &source, &contiguous),
          "...and nothing maps through it");

    runs[0].first = 0u;
    runs[0].count = 8u;
    runs[1].first = 504u;
    runs[1].count = 16u;
    CHECK(!pn_tt_column_runs_are_injective(runs, 2u, ORNITH_SOURCE_COLS),
          "a run that ends past the source is refused");

    runs[1].first = 4u;
    runs[1].count = 8u;
    CHECK(!pn_tt_column_runs_are_injective(runs, 2u, ORNITH_SOURCE_COLS),
          "two runs that OVERLAP are refused — a duplicated column is a smoothly wrong "
          "projection nothing on the card can see");

    runs[1].first = 8u;
    CHECK(pn_tt_column_runs_are_injective(runs, 2u, ORNITH_SOURCE_COLS),
          "...and two runs that merely abut are fine");

    CHECK(pn_tt_column_runs_field(ORNITH_HEADS, 8u, 0u, 16u, runs, 4u) == 0u,
          "a field WIDER than its record is refused — it would overlap the next group");
    CHECK(pn_tt_column_runs_field(ORNITH_HEADS, 16u, 12u, 8u, runs, 4u) == 0u,
          "a field that starts too late in its record is refused");
    CHECK(pn_tt_column_runs_field(ORNITH_HEADS, 16u, 0u, 8u, runs, 4u) == 0u,
          "a run array too small for the group count is refused, not truncated");
    CHECK(pn_tt_column_runs_field(0u, 16u, 0u, 8u, runs, 4u) == 0u, "zero groups is refused");
}

/** The identity permutation has to be expressible, because the shared placement path takes
 *  a run list for every weight and the ordinary ones must come out byte for byte unchanged. */
static void test_identity(void)
{
    pn_tt_column_run identity;
    uint32_t index;
    bool ok = true;

    printf("the identity permutation\n");

    identity.first = 0u;
    identity.count = 512u;
    CHECK(pn_tt_column_runs_total(&identity, 1u) == 512u, "identity totals the whole weight");
    CHECK(pn_tt_column_runs_are_injective(&identity, 1u, 512u), "identity is injective");
    for (index = 0; index < 512u; index++) {
        uint32_t source, contiguous;

        if (!pn_tt_column_runs_map(&identity, 1u, index, &source, &contiguous) ||
            source != index || contiguous != 512u - index)
            ok = false;
    }
    CHECK(ok, "identity maps every column to itself in ONE transfer");
}

int main(void)
{
    printf("test_tt_column_runs — the joint-projection de-interleave, off-hardware\n\n");
    test_ornith_field_lists();
    test_mapping_against_the_model();
    test_refusals();
    test_identity();
    printf("\n%d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
