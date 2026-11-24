/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_strip_placement.c
 * @brief Implementation of mirror-first strip placement and its dry run
 *        (pn_tt_strip_placement.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_strip_placement.h"

#include <stdbool.h>

size_t pn_tt_strip_placement_order(const pn_tt_strip_request *requests, size_t count,
                                   size_t *out_order)
{
    size_t written = 0u;

    if (!requests || !out_order)
        return 0u;
    for (size_t i = 0u; i < count; i++)
        if (requests[i].replicas > 1u)
            out_order[written++] = i;
    for (size_t i = 0u; i < count; i++)
        if (requests[i].replicas <= 1u)
            out_order[written++] = i;
    return written;
}

uint32_t pn_tt_strip_placement_emptiest_bank(const uint64_t *free_bytes, uint32_t bank_count,
                                             uint64_t bytes)
{
    uint32_t best = bank_count;

    if (!free_bytes)
        return bank_count;
    for (uint32_t bank = 0u; bank < bank_count; bank++)
        if (free_bytes[bank] >= bytes &&
            (best == bank_count || free_bytes[bank] > free_bytes[best]))
            best = bank;
    return best;
}

/* pn_tt_dram_alloc_any: the cursor's bank, else the emptiest bank that fits. */
static uint32_t simulate_round_robin_bank(const uint64_t *free_bytes, uint32_t bank_count,
                                          uint32_t *cursor, uint64_t bytes)
{
    const uint32_t first = (*cursor)++ % bank_count;

    if (free_bytes[first] >= bytes)
        return first;
    return pn_tt_strip_placement_emptiest_bank(free_bytes, bank_count, bytes);
}

int pn_tt_strip_placement_simulate(const pn_tt_strip_request *requests, const size_t *order,
                                   size_t count, pn_tt_strip_policy policy,
                                   uint64_t bytes_per_tile, uint64_t *free_bytes,
                                   uint32_t bank_count, uint32_t *cursor, size_t *out_failed)
{
    if (!requests || !order || !free_bytes || !cursor || bank_count == 0u)
        return -1;
    for (size_t step = 0u; step < count; step++) {
        const pn_tt_strip_request *strip = &requests[order[step]];
        const uint64_t bytes = (uint64_t)strip->tile_count * bytes_per_tile;
        const bool round_robin =
            strip->replicas > 1u || policy == PN_TT_STRIP_POLICY_ROUND_ROBIN;
        uint32_t home;

        if (strip->tile_count == 0u)
            continue;
        home = round_robin ? simulate_round_robin_bank(free_bytes, bank_count, cursor, bytes)
                           : pn_tt_strip_placement_emptiest_bank(free_bytes, bank_count, bytes);
        bool fits = home < bank_count && strip->replicas <= bank_count;

        if (fits)
            free_bytes[home] -= bytes;
        for (uint32_t copy = 1u; fits && copy < strip->replicas; copy++) {
            const uint32_t bank = (home + copy) % bank_count;

            fits = free_bytes[bank] >= bytes;
            if (fits)
                free_bytes[bank] -= bytes;
        }
        if (!fits) {
            if (out_failed)
                *out_failed = order[step];
            return -1;
        }
    }
    return 0;
}
