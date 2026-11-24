/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_view.c
 * @brief Implementation of the column view of a resident weight (pn_tt_weight_view.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_weight_view.h"

#include <string.h>

/** One run of view columns inside one source panel. */
typedef struct view_segment {
    uint32_t shard;  /**< the source panel */
    uint32_t first;  /**< its first SOURCE column */
    uint32_t count;  /**< columns */
    uint32_t pieces; /**< members it is split over */
} view_segment;

static uint32_t view_ceil_div(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

int pn_tt_weight_column_view(const pn_tt_weight *source, const pn_tt_column_run *runs,
                             uint32_t run_count, uint32_t members, pn_tt_weight *out)
{
    view_segment segment[PN_TT_MATMUL_MAX_CORES];
    const uint32_t shards = source ? source->shard_count : 0u;
    uint32_t segments = 0u, total, spare, view_column = 0u, piece = 0u;

    if (!source || !runs || !out || shards == 0u || shards > PN_TT_MATMUL_MAX_CORES ||
        members == 0u || members > PN_TT_MATMUL_MAX_CORES || source->tile_rows == 0u ||
        source->tile_bytes == 0u)
        return -1;
    total = pn_tt_column_runs_total(runs, run_count);
    if (total == 0u || !pn_tt_column_runs_are_injective(runs, run_count, source->tile_cols))
        return -1;

    /* The segments: every run cut at the source's panel edges. */
    for (uint32_t r = 0; r < run_count; r++) {
        const uint32_t run_end = runs[r].first + runs[r].count;

        for (uint32_t s = 0; s < shards; s++) {
            const uint32_t lo = source->shard_column_first[s];
            const uint32_t hi = lo + source->shard_tile_cols[s];
            const uint32_t from = runs[r].first > lo ? runs[r].first : lo;
            const uint32_t to = run_end < hi ? run_end : hi;

            if (from >= to)
                continue;
            if (segments == members)
                return -1;
            segment[segments].shard = s;
            segment[segments].first = from;
            segment[segments].count = to - from;
            segment[segments].pieces = 1u;
            segments++;
        }
    }
    {
        uint32_t covered = 0u;

        for (uint32_t i = 0; i < segments; i++)
            covered += segment[i].count;
        if (covered != total)
            return -1; /* a source whose panels do not cover its columns */
    }

    /* The spare members go, one at a time, to the segment with the widest piece. */
    for (spare = members - segments; spare > 0u; spare--) {
        uint32_t best = segments, widest = 1u;

        for (uint32_t i = 0; i < segments; i++) {
            const uint32_t width = view_ceil_div(segment[i].count, segment[i].pieces);

            if (segment[i].pieces < segment[i].count && width > widest) {
                widest = width;
                best = i;
            }
        }
        if (best == segments)
            break;
        segment[best].pieces++;
    }

    memset(out, 0, sizeof(*out));
    out->tile_rows = source->tile_rows;
    out->tile_cols = total;
    out->tile_bytes = source->tile_bytes;
    out->in_features = source->in_features;
    out->out_features = total * 32u;
    out->probe_selected = source->probe_selected;
    for (uint32_t i = 0; i < segments; i++) {
        const uint32_t s = segment[i].shard;
        const uint32_t base = segment[i].count / segment[i].pieces;
        const uint32_t extra = segment[i].count % segment[i].pieces;
        uint32_t column = segment[i].first;

        for (uint32_t p = 0; p < segment[i].pieces; p++, piece++) {
            const uint32_t width = base + (p < extra ? 1u : 0u);

            out->shard_tile_cols[piece] = width;
            out->shard_column_first[piece] = view_column;
            out->shard_address[piece] =
                source->shard_address[s] + (column - source->shard_column_first[s]) *
                                               source->tile_rows * source->tile_bytes;
            out->shard_x[piece] = source->shard_x[s];
            out->shard_y[piece] = source->shard_y[s];
            out->shard_bank[piece] = source->shard_bank[s];
            column += width;
            view_column += width;
        }
    }
    out->shard_count = piece;
    out->bank = out->shard_bank[0];
    out->address = out->shard_address[0];
    return 0;
}

#endif /* USE_TT */
