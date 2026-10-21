/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_descriptor_census.c
 * @brief Implementation of the (family, function, tile count) descriptor census. See
 *        pn_tt_descriptor_census.h, which carries why the per-residency report is not
 *        enough for the elementwise family and does not see the render pair at all.
 *
 *        No device, no allocation, no I/O — deliberately, so it is provable with no card.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_descriptor_census.h"

#include <string.h>

/**
 * Is `value` finite, asked of the EXPONENT BITS THROUGH A VOLATILE SLOT.
 *
 * `isfinite()` is foldable to a literal 1 under `-ffast-math` (which implies
 * `-ffinite-math-only`), and this tree compiles with `-ffast-math`. Asking the bits is the
 * recorded fix — AND THE RECORDED FIX IS NOT ENOUGH ON ITS OWN.
 *
 * Measured here, Apple clang, `-O3 -ffast-math`: a plain `memcpy`-then-shift written inline
 * in the calling translation unit answers **1 for a positive infinity and 1 for a NaN**. The
 * compiler recognises the idiom, applies `-ffinite-math-only` to it, and folds it — the exact
 * fold the bit test was adopted to defeat. It is not constant propagation: it survives the
 * value arriving through a `volatile uint64_t`. The same code compiled without `-ffast-math`
 * answers 0 for both.
 *
 * The `volatile double` slot below fixes it (as does reading through a `const double *`, or
 * `__attribute__((optnone))`): a volatile store forbids the compiler from reasoning about the
 * value it just wrote. `tests/parity_score.c`'s `parity_double_is_finite()` has the unguarded
 * shape and is safe only because it lives in its own translation unit with no LTO — turning
 * LTO on would silently re-open the eleventh vacuous-gate class.
 */
static int census_double_is_finite(double value)
{
    volatile double opaque = value;
    double copy = opaque;
    uint64_t bits;

    memcpy(&bits, &copy, sizeof(bits));
    return ((bits >> 52) & 0x7FFu) != 0x7FFu;
}

/** Copy `source` into a key field, always NUL-terminated, truncating rather than refusing. */
static void census_copy_key(char *destination, const char *source)
{
    size_t index = 0;

    if (!source) {
        destination[0] = '\0';
        return;
    }
    while (index + 1u < PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES && source[index] != '\0') {
        destination[index] = source[index];
        index++;
    }
    destination[index] = '\0';
}

/** Does `row` already carry this key? The key is all three fields, never two. */
static bool census_row_matches(const pn_tt_descriptor_census_row *row, const char *family,
                               const char *function, uint32_t tile_count)
{
    char wanted_family[PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES];
    char wanted_function[PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES];

    if (row->tile_count != tile_count)
        return false;
    census_copy_key(wanted_family, family);
    census_copy_key(wanted_function, function);
    return strcmp(row->family, wanted_family) == 0 &&
           strcmp(row->function, wanted_function) == 0;
}

/**
 * Find or create the row for a key, or say why not.
 *
 * @param out_row  set on success only
 * @return         0, or -1 with `refused` or `dropped` already accounted
 */
static int census_row_for_key(pn_tt_descriptor_census *census, const char *family,
                              const char *function, uint32_t tile_count, double seconds,
                              pn_tt_descriptor_census_row **out_row)
{
    uint32_t index;

    /* A zero width or a non-finite duration is a defect in the CALLER, and folding it into a
     * row would make the fit answer with authority about a number nobody measured. */
    if (tile_count == 0u || !census_double_is_finite(seconds) || seconds < 0.0) {
        census->refused++;
        return -1;
    }
    for (index = 0; index < census->row_count; index++) {
        if (census_row_matches(&census->rows[index], family, function, tile_count)) {
            *out_row = &census->rows[index];
            return 0;
        }
    }
    if (census->row_count >= PN_TT_DESCRIPTOR_CENSUS_ROWS) {
        /* NOT an "other" bucket. A truncated census that still sums to the family total is
         * indistinguishable from a complete one. */
        census->dropped++;
        census->dropped_seconds += seconds;
        return -1;
    }
    index = census->row_count++;
    memset(&census->rows[index], 0, sizeof(census->rows[index]));
    census_copy_key(census->rows[index].family, family);
    census_copy_key(census->rows[index].function, function);
    census->rows[index].tile_count = tile_count;
    *out_row = &census->rows[index];
    return 0;
}

int pn_tt_descriptor_census_record(pn_tt_descriptor_census *census, const char *family,
                                   const char *function, uint32_t tile_count, double seconds)
{
    pn_tt_descriptor_census_row *row = NULL;

    if (!census ||
        census_row_for_key(census, family, function, tile_count, seconds, &row) != 0)
        return -1;
    row->ops++;
    row->seconds += seconds;
    return 0;
}

int pn_tt_descriptor_census_record_recovered(pn_tt_descriptor_census *census,
                                             const char *family, const char *function,
                                             uint32_t tile_count, double seconds,
                                             uint64_t rearms)
{
    pn_tt_descriptor_census_row *row = NULL;

    if (!census)
        return -1;
    /* A "recovered" descriptor that re-armed nothing is a CLEAN one under the wrong name, and
     * accepting it would let a caller move time out of the fit by mislabelling it. */
    if (rearms == 0u) {
        census->refused++;
        return -1;
    }
    if (census_row_for_key(census, family, function, tile_count, seconds, &row) != 0)
        return -1;
    row->recovered_ops++;
    row->recovered_seconds += seconds;
    row->rearms += rearms;
    return 0;
}

bool pn_tt_descriptor_census_row_at(const pn_tt_descriptor_census *census, uint32_t index,
                                    const pn_tt_descriptor_census_row **out_row)
{
    if (!census || index >= census->row_count)
        return false;
    if (out_row)
        *out_row = &census->rows[index];
    return true;
}

/** Does `row` belong to `family`? A NULL family means every row. */
static bool census_row_in_family(const pn_tt_descriptor_census_row *row, const char *family)
{
    char wanted[PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES];

    if (!family)
        return true;
    census_copy_key(wanted, family);
    return strcmp(row->family, wanted) == 0;
}

/**
 * Distinct tile counts among a family's CLEAN rows, counted the O(n^2) way; n is at most 64.
 *
 * Clean because this is the fit's shape guard: a row that only ever refused contributes no
 * equation, so counting it would let a one-point fit past the first of the two redundant
 * guards and leave the determinant test alone with the job.
 */
static uint32_t census_distinct_shapes(const pn_tt_descriptor_census *census,
                                       const char *family)
{
    uint32_t distinct = 0;
    uint32_t index;

    for (index = 0; index < census->row_count; index++) {
        uint32_t earlier;
        bool seen = false;

        if (!census_row_in_family(&census->rows[index], family) || census->rows[index].ops == 0u)
            continue;
        for (earlier = 0; earlier < index && !seen; earlier++)
            seen = census_row_in_family(&census->rows[earlier], family) &&
                   census->rows[earlier].ops != 0u &&
                   census->rows[earlier].tile_count == census->rows[index].tile_count;
        if (!seen)
            distinct++;
    }
    return distinct;
}

int pn_tt_descriptor_census_family_totals(const pn_tt_descriptor_census *census,
                                          const char *family, uint64_t *out_ops,
                                          double *out_seconds, uint32_t *out_shapes)
{
    uint64_t ops = 0;
    double seconds = 0.0;
    uint32_t rows = 0;
    uint32_t index;

    if (!census)
        return -1;
    for (index = 0; index < census->row_count; index++) {
        if (!census_row_in_family(&census->rows[index], family))
            continue;
        ops += census->rows[index].ops;
        seconds += census->rows[index].seconds;
        rows++;
    }
    if (rows == 0u)
        return -1;
    if (out_ops)
        *out_ops = ops;
    if (out_seconds)
        *out_seconds = seconds;
    if (out_shapes)
        *out_shapes = census_distinct_shapes(census, family);
    return 0;
}

int pn_tt_descriptor_census_family_recovery(const pn_tt_descriptor_census *census,
                                            const char *family, uint64_t *out_recovered_ops,
                                            double *out_recovered_seconds,
                                            uint64_t *out_rearms)
{
    uint64_t recovered_ops = 0;
    double recovered_seconds = 0.0;
    uint64_t rearms = 0;
    uint32_t rows = 0;
    uint32_t index;

    if (!census)
        return -1;
    for (index = 0; index < census->row_count; index++) {
        if (!census_row_in_family(&census->rows[index], family))
            continue;
        recovered_ops += census->rows[index].recovered_ops;
        recovered_seconds += census->rows[index].recovered_seconds;
        rearms += census->rows[index].rearms;
        rows++;
    }
    /* Only "the family never ran" refuses. A family that ran and never refused answers zeros,
     * because a reader has to be able to print "0 re-arms" rather than print nothing. */
    if (rows == 0u)
        return -1;
    if (out_recovered_ops)
        *out_recovered_ops = recovered_ops;
    if (out_recovered_seconds)
        *out_recovered_seconds = recovered_seconds;
    if (out_rearms)
        *out_rearms = rearms;
    return 0;
}

int pn_tt_descriptor_census_fit(const pn_tt_descriptor_census *census, const char *family,
                                double *out_fixed_seconds, double *out_seconds_per_tile,
                                uint32_t *out_shapes)
{
    double weight_sum = 0.0, weighted_x = 0.0, weighted_y = 0.0;
    double weighted_xx = 0.0, weighted_xy = 0.0;
    uint32_t shapes;
    uint32_t index;
    double determinant;

    if (!census)
        return -1;
    shapes = census_distinct_shapes(census, family);
    /* TWO UNKNOWNS NEED TWO EQUATIONS. A family that ran at one width cannot say which part
     * of its descriptor is fixed, and answering anyway is how "958 ns per single-tile op"
     * was produced from a division. */
    if (shapes < 2u)
        return -1;
    for (index = 0; index < census->row_count; index++) {
        const pn_tt_descriptor_census_row *row = &census->rows[index];
        double weight, x, y;

        if (!census_row_in_family(row, family) || row->ops == 0u)
            continue;
        weight = (double)row->ops;
        x = (double)row->tile_count;
        y = row->seconds / weight;
        weight_sum += weight;
        weighted_x += weight * x;
        weighted_y += weight * y;
        weighted_xx += weight * x * x;
        weighted_xy += weight * x * y;
    }
    determinant = weight_sum * weighted_xx - weighted_x * weighted_x;
    /* Scale-free degeneracy test: the absolute determinant of a fit over 7,760-tile shapes is
     * enormous while the same fit over 4-tile shapes is tiny, so a fixed epsilon would refuse
     * one family and accept a collinear other. */
    if (!census_double_is_finite(determinant) ||
        determinant <= 1e-12 * weight_sum * weighted_xx)
        return -1;
    if (out_seconds_per_tile)
        *out_seconds_per_tile = (weight_sum * weighted_xy - weighted_x * weighted_y) /
                                determinant;
    if (out_fixed_seconds)
        *out_fixed_seconds = (weighted_xx * weighted_y - weighted_x * weighted_xy) /
                             determinant;
    if (out_shapes)
        *out_shapes = shapes;
    return 0;
}
