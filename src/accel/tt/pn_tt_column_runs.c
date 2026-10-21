/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_column_runs.c
 * @brief Implementation of the column-tile permutation arithmetic. See pn_tt_column_runs.h,
 *        which carries why a joint projection is de-interleaved in the WEIGHT.
 *
 *        No device, no allocation, no I/O — deliberately, so it is provable with no card.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_column_runs.h"

uint32_t pn_tt_column_runs_total(const pn_tt_column_run *runs, uint32_t run_count)
{
    uint32_t total = 0u;
    uint32_t index;

    if (!runs || run_count == 0u || run_count > PN_TT_COLUMN_RUNS_MAX)
        return 0u;
    for (index = 0; index < run_count; index++) {
        if (runs[index].count == 0u)
            return 0u;
        /* A destination wider than any weight this backend admits is a caller bug that would
         * otherwise wrap and produce a plausible small number. */
        if (total > UINT32_MAX - runs[index].count)
            return 0u;
        total += runs[index].count;
    }
    return total;
}

bool pn_tt_column_runs_are_injective(const pn_tt_column_run *runs, uint32_t run_count,
                                     uint32_t source_columns)
{
    uint32_t index;

    if (pn_tt_column_runs_total(runs, run_count) == 0u || source_columns == 0u)
        return false;
    for (index = 0; index < run_count; index++) {
        uint32_t other;

        if (runs[index].first >= source_columns ||
            runs[index].count > source_columns - runs[index].first)
            return false;
        /* Pairwise rather than a bitmap: `run_count` is at most 128, so this is at worst
         * 8,128 comparisons ONCE at model load, and a bitmap would need a bound on
         * `source_columns` that this module has no business having. */
        for (other = 0; other < index; other++) {
            const uint32_t a_end = runs[index].first + runs[index].count;
            const uint32_t b_end = runs[other].first + runs[other].count;

            if (runs[index].first < b_end && runs[other].first < a_end)
                return false;
        }
    }
    return true;
}

bool pn_tt_column_runs_map(const pn_tt_column_run *runs, uint32_t run_count,
                           uint32_t destination_column, uint32_t *out_source_column,
                           uint32_t *out_contiguous)
{
    uint32_t walked = 0u;
    uint32_t index;

    if (pn_tt_column_runs_total(runs, run_count) == 0u || !out_source_column ||
        !out_contiguous)
        return false;
    for (index = 0; index < run_count; index++) {
        if (destination_column < walked + runs[index].count) {
            const uint32_t inside = destination_column - walked;

            *out_source_column = runs[index].first + inside;
            *out_contiguous = runs[index].count - inside;
            return true;
        }
        walked += runs[index].count;
    }
    return false;
}

uint32_t pn_tt_column_runs_field(uint32_t groups, uint32_t stride, uint32_t offset,
                                 uint32_t width, pn_tt_column_run *out_runs, uint32_t max_runs)
{
    uint32_t index;

    if (!out_runs || groups == 0u || width == 0u || groups > max_runs ||
        groups > PN_TT_COLUMN_RUNS_MAX)
        return 0u;
    /* The field has to fit inside the record, or consecutive groups would overlap and the
     * injectivity check downstream would report a defect this function created. */
    if (stride < width || offset > stride - width)
        return 0u;
    for (index = 0; index < groups; index++) {
        out_runs[index].first = index * stride + offset;
        out_runs[index].count = width;
    }
    return groups;
}
