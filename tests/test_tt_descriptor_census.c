/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_descriptor_census.c — the (family, function, tile count) descriptor census and the
 * fixed/per-tile fit it supports, scored with NO CARD and NO USE_TT.
 *
 * WHAT IT PROVES, and why each part is here rather than left to a decode run:
 *
 *   1. The KEY IS ALL THREE FIELDS. An eltwise residency runs six functions over three
 *      widths; a census that keyed on the family would reproduce exactly the "293.8 us each"
 *      average it exists to replace, and a census that keyed on (family, function) would
 *      merge the FFN SwiGLU with a gate-width one. Both merges are checked as merges that
 *      must NOT happen.
 *
 *   2. THE FIT RECOVERS A LINE IT DID NOT SEE. Synthetic rows on an exact
 *      `fixed + per_tile * tiles` line come back to 1e-12, and a SECOND line with different
 *      coefficients comes back different — so the fit is reading the data and not returning
 *      a constant. Weighting by descriptor count is checked the same way, with the ops
 *      piled onto one shape.
 *
 *   3. THE FIT REFUSES ONE SHAPE. Two unknowns need two equations. Answering a one-shape
 *      family is precisely the arithmetic that produced the retired "958 ns per single-tile
 *      op", and it is refused rather than approximated.
 *
 *   4. AN OVERFLOW IS LOUD. A full table does not fold the 65th key into an "other" bucket:
 *      a truncated census that still sums to the family total is indistinguishable from a
 *      complete one, so the drop is counted separately and the row count stays put.
 *
 *   5. A NaN NEVER REACHES THE FIT. `failed = value > limit` is false for a NaN and
 *      `isfinite()` folds to 1 under this tree's -ffast-math, so finiteness is asked of the
 *      exponent bits at RECORD time — before a poisoned row can exist.
 *
 *   6. AN ABSENT FAMILY IS AN ERROR, NOT ZERO. A family that never ran must not read as a
 *      family that ran for free.
 *
 * The census is the instrument the elementwise and render decomposition is taken with, so
 * its own arithmetic is settled before a model is ever loaded.
 */

#include "../src/accel/tt/pn_tt_descriptor_census.h"
#include "../src/accel/tt/pn_tt_name_list.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_checked = 0;

#define CHECK(cond, msg) do { \
    g_checked++; \
    if (!(cond)) { \
        g_failed++; \
        printf("  FAIL %s\n", (msg)); \
    } \
} while (0)

/** Relative closeness, so a slope of 1e-7 and an intercept of 1e-4 use the same test. */
static bool close_enough(double got, double want, double tolerance)
{
    double scale = want < 0.0 ? -want : want;
    double error = got - want;

    if (error < 0.0)
        error = -error;
    if (scale < 1.0)
        scale = 1.0;
    return error <= tolerance * scale;
}

/**
 * Do two figures differ by at least `fraction` of the larger?
 *
 * `close_enough()` floors its scale at 1.0 so a tolerance can be read as absolute, which
 * makes its negation useless at microsecond magnitudes — every pair of times is "close".
 * A discriminating arm needs a genuinely RELATIVE test, and this is it.
 */
static bool differs_relatively(double a, double b, double fraction)
{
    double difference = a - b;
    double scale = (a < 0.0 ? -a : a) > (b < 0.0 ? -b : b) ? (a < 0.0 ? -a : a)
                                                           : (b < 0.0 ? -b : b);

    if (difference < 0.0)
        difference = -difference;
    if (scale == 0.0)
        return difference > 0.0;
    return difference / scale >= fraction;
}

/** A NaN built from bits, because a literal 0.0/0.0 is folded by -ffast-math. */
static double make_nan(void)
{
    uint64_t bits = 0x7FF8000000000000ull;
    double value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * A POSITIVE infinity, from bits for the same reason.
 *
 * THIS IS THE ARM THAT DISCRIMINATES. A NaN is refused by the finiteness test AND — measured,
 * on this tree's `-ffast-math` — by the `seconds < 0.0` test beside it, because the compiler
 * is licensed to assume no NaN exists and may answer that comparison either way. So deleting
 * the finiteness test entirely leaves a NaN-only suite reading a clean green: the ninth shape
 * of vacuous gate in this project, inside the check for the eleventh. `+INFINITY` is
 * non-finite and NOT negative, so only the finiteness test can refuse it.
 */
static double make_positive_infinity(void)
{
    uint64_t bits = 0x7FF0000000000000ull;
    double value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* ------------------------------------------------------------------------ */

static void test_the_key_is_all_three_fields(void)
{
    pn_tt_descriptor_census census;
    const pn_tt_descriptor_census_row *row = NULL;

    printf("the key is (family, function, tile count)\n");
    memset(&census, 0, sizeof(census));

    /* Same key three times aggregates into ONE row. */
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, 1e-4) == 0,
          "record accepts a well-formed descriptor");
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, 2e-4);
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, 3e-4);
    CHECK(census.row_count == 1u, "three descriptors on one key make one row");
    CHECK(pn_tt_descriptor_census_row_at(&census, 0u, &row), "row 0 reads back");
    CHECK(row && row->ops == 3u, "the row counts three descriptors");
    CHECK(row && close_enough(row->seconds, 6e-4, 1e-12), "the row sums their time");

    /* A DIFFERENT FUNCTION at the same width is a different row. This is the merge that a
     * family-keyed census would make, and it is the merge that hides a unary sigmoid inside
     * a binary silu_mul's average. */
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "silu_mul", 128u, 1e-4);
    CHECK(census.row_count == 2u, "a different function at the same width is a new row");

    /* A DIFFERENT WIDTH with the same function is a different row. This is the merge that a
     * (family, function)-keyed census would make, and it is the one that makes the fit
     * impossible — the fit needs the widths APART. */
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 384u, 1e-4);
    CHECK(census.row_count == 3u, "a different width with the same function is a new row");

    /* A different family never merges, however alike the rest of the key. */
    (void)pn_tt_descriptor_census_record(&census, "render-widen", "add", 128u, 1e-4);
    CHECK(census.row_count == 4u, "a different family is a new row");

    {
        uint64_t ops = 0;
        double seconds = 0.0;
        uint32_t shapes = 0;

        CHECK(pn_tt_descriptor_census_family_totals(&census, "eltwise", &ops, &seconds,
                                                    &shapes) == 0,
              "eltwise totals resolve");
        CHECK(ops == 5u, "eltwise totals five descriptors, not six");
        CHECK(shapes == 2u, "eltwise ran two distinct widths");
        CHECK(close_enough(seconds, 8e-4, 1e-12), "eltwise totals its own time only");
        CHECK(pn_tt_descriptor_census_family_totals(&census, NULL, &ops, NULL, NULL) == 0 &&
              ops == 6u, "a NULL family sums every row");
    }
}

static void test_truncation_is_a_known_property(void)
{
    pn_tt_descriptor_census census;

    printf("key truncation, stated rather than discovered\n");
    memset(&census, 0, sizeof(census));

    /* Names that differ INSIDE the key width stay apart. */
    (void)pn_tt_descriptor_census_record(&census, "render-widen", "widen", 4u, 1e-4);
    (void)pn_tt_descriptor_census_record(&census, "render-narrow", "narrow", 4u, 1e-4);
    CHECK(census.row_count == 2u, "the two render directions do not collide");

    /* Names that differ only PAST the width merge, and that is a property callers must know
     * about rather than meet in a report. 24 bytes holds 23 characters. */
    (void)pn_tt_descriptor_census_record(
        &census, "aaaaaaaaaaaaaaaaaaaaaaaaLONG", "f", 4u, 1e-4);
    (void)pn_tt_descriptor_census_record(
        &census, "aaaaaaaaaaaaaaaaaaaaaaaaWIDE", "f", 4u, 1e-4);
    CHECK(census.row_count == 3u, "names differing only past the key width MERGE");
}

static void test_refusals_are_counted_apart(void)
{
    pn_tt_descriptor_census census;

    printf("refusals, and none of them silent\n");
    memset(&census, 0, sizeof(census));

    CHECK(pn_tt_descriptor_census_record(NULL, "eltwise", "add", 128u, 1e-4) == -1,
          "a NULL census is refused");
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 0u, 1e-4) == -1,
          "a zero-width descriptor is refused");
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, make_nan()) == -1,
          "a NaN duration is refused");
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u,
                                         make_positive_infinity()) == -1,
          "a POSITIVE INFINITY duration is refused — only finiteness can catch this one");
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, -1e-4) == -1,
          "a negative duration is refused");
    CHECK(census.refused == 4u, "the four refusals with a census are counted");
    CHECK(census.row_count == 0u, "a refused descriptor creates no row");
    CHECK(census.dropped == 0u, "a refusal is not a drop");
}

static void test_an_overflow_is_loud(void)
{
    pn_tt_descriptor_census census;
    uint32_t index;

    printf("a full table drops LOUDLY, it does not bucket\n");
    memset(&census, 0, sizeof(census));

    for (index = 0; index < PN_TT_DESCRIPTOR_CENSUS_ROWS; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", index + 1u, 1e-4);
    CHECK(census.row_count == PN_TT_DESCRIPTOR_CENSUS_ROWS, "the table fills");
    CHECK(census.dropped == 0u, "nothing dropped while it had room");

    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 9999u, 7e-4) == -1,
          "the 65th distinct key is refused");
    CHECK(census.dropped == 1u, "and counted as a DROP");
    CHECK(close_enough(census.dropped_seconds, 7e-4, 1e-12), "with its time kept apart");
    CHECK(census.row_count == PN_TT_DESCRIPTOR_CENSUS_ROWS, "the row count does not grow");
    {
        uint64_t ops = 0;

        (void)pn_tt_descriptor_census_family_totals(&census, "eltwise", &ops, NULL, NULL);
        CHECK(ops == PN_TT_DESCRIPTOR_CENSUS_ROWS,
              "the dropped descriptor is NOT in the family total");
    }
    /* A key already present still records after the table is full — the limit is on KEYS. */
    CHECK(pn_tt_descriptor_census_record(&census, "eltwise", "add", 1u, 1e-4) == 0,
          "an existing key still records with the table full");
}

/** Fill `census` with an exact line, `ops[i]` descriptors at width `widths[i]`. */
static void plant_a_line(pn_tt_descriptor_census *census, const char *family, double fixed,
                         double per_tile, const uint32_t *widths, const uint32_t *ops,
                         uint32_t shapes)
{
    uint32_t shape, copy;

    for (shape = 0; shape < shapes; shape++) {
        double seconds = fixed + per_tile * (double)widths[shape];

        for (copy = 0; copy < ops[shape]; copy++)
            (void)pn_tt_descriptor_census_record(census, family, "add", widths[shape],
                                                 seconds);
    }
}

static void test_the_fit_recovers_a_line(void)
{
    static const uint32_t widths[3] = { 4u, 128u, 384u };
    static const uint32_t ops[3] = { 8u, 48u, 24u };
    pn_tt_descriptor_census census;
    double fixed = 0.0, per_tile = 0.0;
    uint32_t shapes = 0;

    printf("the fit recovers a line it did not see\n");

    memset(&census, 0, sizeof(census));
    plant_a_line(&census, "eltwise", 60e-6, 1.7e-6, widths, ops, 3u);
    CHECK(pn_tt_descriptor_census_fit(&census, "eltwise", &fixed, &per_tile, &shapes) == 0,
          "a three-shape family fits");
    CHECK(shapes == 3u, "the fit used three shapes");
    CHECK(close_enough(fixed, 60e-6, 1e-9), "the intercept comes back");
    CHECK(close_enough(per_tile, 1.7e-6, 1e-9), "the slope comes back");

    /* THE DISCRIMINATING ARM: a different line must give different coefficients, or the fit
     * is returning a constant and every figure it produced is meaningless. */
    memset(&census, 0, sizeof(census));
    plant_a_line(&census, "eltwise", 200e-6, 0.4e-6, widths, ops, 3u);
    CHECK(pn_tt_descriptor_census_fit(&census, "eltwise", &fixed, &per_tile, NULL) == 0,
          "the second line fits too");
    CHECK(close_enough(fixed, 200e-6, 1e-9), "and its OWN intercept comes back");
    CHECK(close_enough(per_tile, 0.4e-6, 1e-9), "and its OWN slope comes back");

    /* Weighting, part one: pile the descriptors onto one shape. On an EXACT line the answer
     * is unchanged, which is what says the weights are not distorting a clean fit. This check
     * on its own cannot see whether weighting happens at all — part two can. */
    {
        static const uint32_t lopsided[3] = { 1u, 500u, 1u };

        memset(&census, 0, sizeof(census));
        plant_a_line(&census, "eltwise", 60e-6, 1.7e-6, widths, lopsided, 3u);
        CHECK(pn_tt_descriptor_census_fit(&census, "eltwise", &fixed, &per_tile, NULL) == 0,
              "a lopsided family still fits");
        CHECK(close_enough(fixed, 60e-6, 1e-9), "weighting does not move the intercept");
        CHECK(close_enough(per_tile, 1.7e-6, 1e-9), "weighting does not move the slope");
    }

    /* Weighting, part two — THE DISCRIMINATING ARM. Off a line the weights must matter, or
     * the "48 wide descriptors and 8 narrow ones" census is being read as 2 equal shapes.
     * Three non-collinear points, fitted twice with only the descriptor counts changed. */
    {
        static const uint32_t narrow[3] = { 1u, 2u, 3u };
        double even_fixed = 0.0, even_slope = 0.0;
        double heavy_fixed = 0.0, heavy_slope = 0.0;
        uint32_t index;

        memset(&census, 0, sizeof(census));
        (void)pn_tt_descriptor_census_record(&census, "bent", "add", narrow[0], 1e-6);
        (void)pn_tt_descriptor_census_record(&census, "bent", "add", narrow[1], 2e-6);
        (void)pn_tt_descriptor_census_record(&census, "bent", "add", narrow[2], 10e-6);
        CHECK(pn_tt_descriptor_census_fit(&census, "bent", &even_fixed, &even_slope,
                                          NULL) == 0,
              "three equally-weighted non-collinear points fit");

        for (index = 0; index < 99u; index++)
            (void)pn_tt_descriptor_census_record(&census, "bent", "add", narrow[2], 10e-6);
        CHECK(pn_tt_descriptor_census_fit(&census, "bent", &heavy_fixed, &heavy_slope,
                                          NULL) == 0,
              "and again with the third point 100x as common");
        CHECK(differs_relatively(heavy_slope, even_slope, 0.01),
              "the descriptor COUNT moves the slope — the fit is weighted");
        CHECK(differs_relatively(heavy_fixed, even_fixed, 0.01),
              "the descriptor COUNT moves the intercept");
    }
}

static void test_the_fit_refuses_what_it_cannot_answer(void)
{
    static const uint32_t one_width[1] = { 128u };
    static const uint32_t one_op[1] = { 48u };
    pn_tt_descriptor_census census;
    double fixed = -1.0, per_tile = -1.0;

    printf("the fit refuses one shape, and an absent family\n");

    memset(&census, 0, sizeof(census));
    plant_a_line(&census, "eltwise", 60e-6, 1.7e-6, one_width, one_op, 1u);
    CHECK(pn_tt_descriptor_census_fit(&census, "eltwise", &fixed, &per_tile, NULL) == -1,
          "ONE distinct width is refused, not divided");
    CHECK(fixed == -1.0 && per_tile == -1.0, "and the outputs are left untouched");

    /* Two rows, same width, different functions: still ONE equation. A shape count that
     * counted ROWS instead of distinct widths would answer this one. */
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "silu_mul", 128u, 3e-4);
    CHECK(pn_tt_descriptor_census_fit(&census, "eltwise", &fixed, &per_tile, NULL) == -1,
          "two rows at ONE width are still one equation");

    CHECK(pn_tt_descriptor_census_fit(&census, "render-widen", &fixed, &per_tile, NULL) == -1,
          "a family with no rows is refused");
    CHECK(pn_tt_descriptor_census_fit(NULL, "eltwise", &fixed, &per_tile, NULL) == -1,
          "a NULL census is refused");
    CHECK(pn_tt_descriptor_census_family_totals(&census, "render-widen", NULL, NULL,
                                                NULL) == -1,
          "an absent family TOTALS to an error, not to zero");
}

/**
 * The recorded elementwise figure, decomposed and re-composed.
 *
 * 136 descriptors a token at 293.8 us each is 0.03996 s — the plan records 0.0374 s, which is
 * the same family measured on the shipped arm. This check does not assert either number: it
 * asserts that a census whose rows are a PARTITION of the family reproduces the family's own
 * mean exactly, so a decomposition can never disagree with the aggregate it refines.
 */
static void test_a_decomposition_recomposes(void)
{
    pn_tt_descriptor_census census;
    uint64_t ops = 0;
    double seconds = 0.0;
    uint32_t index;

    printf("a decomposition re-composes to the aggregate it refines\n");
    memset(&census, 0, sizeof(census));

    /* 136 = 120 + 16, the recorded split: 120 from the 24 SSM + 8 attention layers' residual
     * and SwiGLU traffic, 16 from the attention gate's SIGMOID and MUL pair over 8 layers.
     * The times here are arbitrary; the PARTITION is the point. */
    for (index = 0; index < 48u; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, 150e-6);
    for (index = 0; index < 24u; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "silu_mul", 384u, 700e-6);
    for (index = 0; index < 48u; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 384u, 300e-6);
    for (index = 0; index < 8u; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "sigmoid", 128u, 120e-6);
    for (index = 0; index < 8u; index++)
        (void)pn_tt_descriptor_census_record(&census, "eltwise", "mul", 128u, 140e-6);

    CHECK(pn_tt_descriptor_census_family_totals(&census, "eltwise", &ops, &seconds,
                                                NULL) == 0,
          "the census resolves");
    CHECK(ops == 136u, "the rows partition 136 descriptors");
    CHECK(census.dropped == 0u && census.refused == 0u, "nothing was dropped or refused");
    CHECK(close_enough(seconds, 48.0 * 150e-6 + 24.0 * 700e-6 + 48.0 * 300e-6 +
                                8.0 * 120e-6 + 8.0 * 140e-6, 1e-12),
          "and their time sums to the family's");
    /* The two extremes differ by 5.8x here — which is the whole reason one mean over the
     * family cannot price a lever against any of them. */
    CHECK(700e-6 / 120e-6 > 5.0, "the widest and narrowest descriptors differ by >5x");
}

/**
 * The comma-list filter behind `PN_TT_ONCARD_BATCH_FAMILY`.
 *
 * It lives in this suite because it is the guard that keeps the batch ladder ANSWER-PRESERVING
 * — it restricts the queue-depth instrument to the families whose descriptors are idempotent.
 * A filter that matched nothing would leave K = 1 everywhere, the ladder would read flat, and
 * the conclusion would be "batching this family buys nothing" rather than "the filter is
 * broken". A restricting knob fails silently into the shape of a negative result.
 */
static void test_the_batch_family_filter(void)
{
    printf("the comma-list filter behind PN_TT_ONCARD_BATCH_FAMILY\n");

    /* An UNSET knob does not restrict — otherwise turning the filter on by accident would
     * silently take every family out of the instrument. */
    CHECK(pn_tt_name_list_contains(NULL, "eltwise"), "a NULL list takes every name");
    CHECK(pn_tt_name_list_contains("", "eltwise"), "an EMPTY list takes every name");

    CHECK(pn_tt_name_list_contains("eltwise", "eltwise"), "a one-element list matches");
    CHECK(!pn_tt_name_list_contains("eltwise", "reduce"), "and rejects anything else");

    /* First, middle and last element, because an off-by-one in the comma walk takes exactly
     * one of the three. */
    CHECK(pn_tt_name_list_contains("eltwise,render-widen,render-narrow", "eltwise"),
          "the FIRST element matches");
    CHECK(pn_tt_name_list_contains("eltwise,render-widen,render-narrow", "render-widen"),
          "the MIDDLE element matches");
    CHECK(pn_tt_name_list_contains("eltwise,render-widen,render-narrow", "render-narrow"),
          "the LAST element matches");
    CHECK(!pn_tt_name_list_contains("eltwise,render-widen,render-narrow", "gdn"),
          "and the GDN recurrence — the family that must NOT be batched — is rejected");

    /* WHOLE-element comparison. `render` must not take `render-widen`: the two render
     * directions are separate residencies and a substring test would silently widen the arm
     * past what the operator asked for. And a prefix of a listed name must not match either. */
    CHECK(!pn_tt_name_list_contains("render-widen", "render"),
          "a PREFIX of a listed name does not match");
    CHECK(!pn_tt_name_list_contains("render", "render-widen"),
          "a listed PREFIX does not take the longer name");
    CHECK(!pn_tt_name_list_contains("eltwise,render-widen", "wise"),
          "an interior substring does not match");

    /* Degenerate lists a real env var produces. */
    CHECK(!pn_tt_name_list_contains("eltwise", NULL), "a NULL name is never a member");
    CHECK(!pn_tt_name_list_contains("eltwise", ""), "an EMPTY name is never a member");
    CHECK(pn_tt_name_list_contains("eltwise,", "eltwise"), "a trailing comma is tolerated");
    CHECK(!pn_tt_name_list_contains("eltwise,", ""),
          "and the empty element a trailing comma leaves matches NOTHING");
    CHECK(pn_tt_name_list_contains(",eltwise", "eltwise"), "a leading comma is tolerated");
}

/**
 * A REPLAYED DESCRIPTOR IS A TEARDOWN, NOT A DESCRIPTOR'S COST.
 *
 * Every resident family submits under a recovery — refusal, re-arm, resubmit — inside ONE
 * stopwatch, and folding that bracket into `seconds / ops` prices the teardown as arithmetic.
 * It has already produced a wrong headline: PHASE CONV read 4,502 us a descriptor against
 * 738.2 for no reason but a circular-buffer write pointer that made every CONV after a DELTA
 * refuse. Those two numbers are the ones planted here.
 *
 * THE ARMS THAT DISCRIMINATE, because a check that only asks "did the counter move" would
 * pass on a census that still averaged the two together:
 *
 *   - the clean per-descriptor figure must equal 738.2 us AND differ RELATIVELY from the
 *     contaminated mean, so a census that quietly summed both goes red;
 *   - the FIT must return the clean line's own coefficients with a recovered-only shape
 *     present at a wildly different time — if recovery reached the normal equations the
 *     intercept moves;
 *   - a family whose every descriptor refused must be REFUSED by the fit, not fitted to its
 *     own recovery path.
 */
static void test_a_replayed_descriptor_is_not_a_cost(void)
{
    static const uint32_t widths[2] = { 128u, 384u };
    static const uint32_t ops[2] = { 12u, 12u };
    pn_tt_descriptor_census census;
    const pn_tt_descriptor_census_row *row = NULL;
    uint64_t recovered_ops = 0, rearms = 0;
    double recovered_seconds = 0.0;
    double fixed = 0.0, per_tile = 0.0;
    uint64_t clean_ops = 0;
    double clean_seconds = 0.0;
    uint32_t index;

    printf("a refused-and-replayed descriptor is kept out of the per-descriptor figure\n");
    memset(&census, 0, sizeof(census));

    for (index = 0; index < 4u; index++)
        (void)pn_tt_descriptor_census_record(&census, "gdn", "conv", 256u, 738.2e-6);
    for (index = 0; index < 4u; index++)
        CHECK(pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 256u,
                                                       4502.0e-6, 1u) == 0,
              "a replayed descriptor records");

    CHECK(pn_tt_descriptor_census_row_at(&census, 0u, &row) && row, "one row, both kinds");
    CHECK(census.row_count == 1u, "recovery does not create a second row");
    CHECK(row->ops == 4u, "only the clean descriptors are in ops");
    CHECK(row->recovered_ops == 4u, "and the replayed ones are counted apart");
    CHECK(row->rearms == 4u, "with their re-arms");
    CHECK(close_enough(1e6 * row->seconds / (double)row->ops, 738.2, 1e-9),
          "the per-descriptor figure is the CLEAN 738.2 us");
    /* THE ARM. A census that averaged both would read 2620.1 us here. */
    CHECK(differs_relatively(1e6 * row->seconds / (double)row->ops,
                             1e6 * (row->seconds + row->recovered_seconds) /
                                 (double)(row->ops + row->recovered_ops),
                             0.5),
          "and it differs from the contaminated mean by more than half — so a census that "
          "summed them would go red here");
    CHECK(close_enough(1e6 * row->recovered_seconds / (double)row->recovered_ops, 4502.0,
                       1e-9),
          "the recovery bracket keeps its own 4502 us");

    CHECK(pn_tt_descriptor_census_family_totals(&census, "gdn", &clean_ops, &clean_seconds,
                                                NULL) == 0 &&
              clean_ops == 4u && close_enough(clean_seconds, 4.0 * 738.2e-6, 1e-12),
          "the family total is the CLEAN total");
    CHECK(pn_tt_descriptor_census_family_recovery(&census, "gdn", &recovered_ops,
                                                  &recovered_seconds, &rearms) == 0 &&
              recovered_ops == 4u && rearms == 4u &&
              close_enough(recovered_seconds, 4.0 * 4502.0e-6, 1e-12),
          "and the recovery reads back beside it");

    /* A clean family answers ZEROS rather than refusing — "never refused" and "never ran" are
     * different facts and a reader must be able to print the first. An absent one refuses. */
    memset(&census, 0, sizeof(census));
    (void)pn_tt_descriptor_census_record(&census, "eltwise", "add", 128u, 1e-4);
    recovered_ops = 7u;
    rearms = 7u;
    CHECK(pn_tt_descriptor_census_family_recovery(&census, "eltwise", &recovered_ops, NULL,
                                                  &rearms) == 0 &&
              recovered_ops == 0u && rearms == 0u,
          "a family that never refused reads 0 re-arms, not an absence");
    CHECK(pn_tt_descriptor_census_family_recovery(&census, "topk", NULL, NULL, NULL) == -1,
          "a family that never ran is refused");

    /* rearms == 0 is a clean descriptor under the wrong name and must not be a way to move
     * time out of the fit. */
    memset(&census, 0, sizeof(census));
    CHECK(pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 256u, 1e-4,
                                                   0u) == -1,
          "a recovered record with no re-arm is refused");
    CHECK(census.refused == 1u && census.row_count == 0u,
          "counted as a refusal, and it creates no row");
    CHECK(pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 0u, 1e-4, 1u) == -1,
          "and the width and finiteness guards still apply to the recovered path");
    CHECK(pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 256u,
                                                   make_positive_infinity(), 1u) == -1,
          "including the infinity the finiteness test alone can catch");
    CHECK(census.refused == 3u, "all three refusals counted");

    /* THE FIT SEES CLEAN ROWS ONLY. Two clean shapes on an exact line, plus a THIRD shape
     * that only ever refused, at a time nowhere near that line. */
    memset(&census, 0, sizeof(census));
    plant_a_line(&census, "gdn", 67.3e-6, 4.52e-6, widths, ops, 2u);
    for (index = 0; index < 40u; index++)
        (void)pn_tt_descriptor_census_record_recovered(&census, "gdn", "add", 32u, 9000e-6,
                                                       1u);
    CHECK(pn_tt_descriptor_census_fit(&census, "gdn", &fixed, &per_tile, NULL) == 0,
          "the family still fits");
    CHECK(close_enough(fixed, 67.3e-6, 1e-9),
          "the intercept is the CLEAN line's — recovery reached no normal equation");
    CHECK(close_enough(per_tile, 4.52e-6, 1e-9), "and so is the slope");

    /* And a family that NEVER retired a first attempt has no line at all. */
    memset(&census, 0, sizeof(census));
    (void)pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 128u, 4502e-6, 1u);
    (void)pn_tt_descriptor_census_record_recovered(&census, "gdn", "conv", 384u, 9000e-6, 1u);
    CHECK(pn_tt_descriptor_census_fit(&census, "gdn", NULL, NULL, NULL) == -1,
          "a family that only ever refused is REFUSED, not fitted to its recovery");
}

int main(void)
{
    printf("test_tt_descriptor_census — the on-card descriptor census, off-hardware\n\n");
    test_the_key_is_all_three_fields();
    test_truncation_is_a_known_property();
    test_refusals_are_counted_apart();
    test_an_overflow_is_loud();
    test_the_fit_recovers_a_line();
    test_the_fit_refuses_what_it_cannot_answer();
    test_a_decomposition_recomposes();
    test_a_replayed_descriptor_is_not_a_cost();
    test_the_batch_family_filter();
    printf("\n%d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
