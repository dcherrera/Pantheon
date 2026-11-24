/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_late_mirrors.c
 * @brief Late, reserve-respecting mirrors for an allocated strip (pn_tt_late_mirrors.h).
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_late_mirrors.h"

#include <stdio.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_prefill_reserve.h"

#include "tt/ft_tt_buffer.h"

/* Undo the mirrors taken so far. Each is the newest allocation of its own bank's arena, which
 * is the only buffer the bump allocator frees. */
static void late_mirrors_release(pn_tt_device *device, const ft_tt_buffer *taken,
                                 uint32_t count)
{
    while (count > 0u) {
        count--;
        if (ft_tt_buffer_free(&device->allocator, &taken[count]) != FT_TT_OK)
            fprintf(stderr, "pn_tt: late mirror in bank %u did not free — its %zu B stay "
                            "allocated, unused\n", taken[count].logical_bank, taken[count].bytes);
    }
}

int pn_tt_activation_mirror_late(pn_tt_device *device, pn_tt_activation *strip,
                                 uint32_t replicas, uint64_t reserve_bytes)
{
    uint64_t free_bytes[PN_TT_FIT_MAX_BANKS];
    ft_tt_buffer taken[PN_TT_ACTIVATION_MAX_REPLICAS];
    uint32_t banks, step, count = 0u;
    size_t bytes;

    if (!device || !strip || strip->tile_count == 0u || strip->tile_rows == 0u ||
        strip->replica_count != 1u)
        return -1;
    banks = pn_tt_device_dram_free_banks(device, free_bytes, PN_TT_FIT_MAX_BANKS);
    if (banks == 0u || strip->bank >= banks)
        return -1;
    if (replicas > banks)
        replicas = banks;
    if (replicas > PN_TT_ACTIVATION_MAX_REPLICAS)
        replicas = PN_TT_ACTIVATION_MAX_REPLICAS;
    bytes = (size_t)strip->tile_count * strip->tile_rows * PN_TT_TILE_BF16;
    for (step = 1u; step < banks && 1u + count < replicas; step++) {
        const uint32_t bank = (strip->bank + step) % banks;

        if (free_bytes[bank] < (uint64_t)bytes + reserve_bytes)
            continue;
        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes,
                               &taken[count]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: late mirror of a %u-tile strip refused in bank %u\n",
                    strip->tile_count, bank);
            late_mirrors_release(device, taken, count);
            return -1;
        }
        count++;
    }
    if (count == 0u)
        return 1;
    /* The strip and its copies identical by construction: all zero (see the header). */
    if (pn_tt_zero_dram(device, strip->bank, strip->address, bytes) != 0) {
        late_mirrors_release(device, taken, count);
        return -1;
    }
    for (step = 0u; step < count; step++)
        if (pn_tt_zero_dram(device, taken[step].logical_bank, (uint32_t)taken[step].address,
                            bytes) != 0) {
            late_mirrors_release(device, taken, count);
            return -1;
        }
    for (step = 0u; step < count; step++) {
        strip->replica_bank[1u + step] = taken[step].logical_bank;
        strip->replica_address[1u + step] = (uint32_t)taken[step].address;
    }
    strip->replica_count = 1u + count;
    return (int)strip->replica_count;
}

#endif /* USE_TT */
