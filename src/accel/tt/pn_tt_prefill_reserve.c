/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_prefill_reserve.c
 * @brief Implementation of the per-bank need arithmetic (pn_tt_prefill_reserve.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_prefill_reserve.h"

#include <string.h>

static uint64_t align_dram(uint64_t bytes)
{
    return (bytes + PN_TT_FIT_DRAM_ALIGN - 1u) & ~(uint64_t)(PN_TT_FIT_DRAM_ALIGN - 1u);
}

void pn_tt_bank_need_init(pn_tt_bank_need *need, uint32_t bank_count)
{
    if (!need)
        return;
    memset(need, 0, sizeof(*need));
    need->bank_count = bank_count > PN_TT_FIT_MAX_BANKS ? PN_TT_FIT_MAX_BANKS : bank_count;
}

void pn_tt_bank_need_pin(pn_tt_bank_need *need, uint32_t bank, uint64_t bytes)
{
    if (need && bank < need->bank_count)
        need->bank[bank] += align_dram(bytes);
}

void pn_tt_bank_need_any(pn_tt_bank_need *need, uint64_t bytes)
{
    const uint64_t aligned = align_dram(bytes);

    if (!need)
        return;
    need->any += aligned;
    if (aligned > need->any_max)
        need->any_max = aligned;
}

uint64_t pn_tt_bank_need_total(const pn_tt_bank_need *need)
{
    uint64_t total = 0u;

    if (!need)
        return 0u;
    for (uint32_t b = 0; b < need->bank_count; b++)
        total += need->bank[b];
    return total + need->any;
}

bool pn_tt_bank_need_fits(const pn_tt_bank_need *need, const uint64_t *free_bytes,
                          uint32_t *short_bank, uint64_t *short_bytes)
{
    uint64_t left = 0u, widest = 0u;

    if (!need || !free_bytes)
        return false;
    for (uint32_t b = 0; b < need->bank_count; b++) {
        if (need->bank[b] > free_bytes[b]) {
            if (short_bank)
                *short_bank = b;
            if (short_bytes)
                *short_bytes = need->bank[b] - free_bytes[b];
            return false;
        }
        left += free_bytes[b] - need->bank[b];
        if (free_bytes[b] - need->bank[b] > widest)
            widest = free_bytes[b] - need->bank[b];
    }
    if (need->any <= left && need->any_max <= widest)
        return true;
    if (short_bank)
        *short_bank = need->bank_count;
    if (short_bytes)
        *short_bytes = need->any > left ? need->any - left : need->any_max - widest;
    return false;
}

void pn_tt_bank_need_take(const pn_tt_bank_need *need, uint64_t *free_bytes)
{
    uint64_t any;

    if (!need || !free_bytes)
        return;
    for (uint32_t b = 0; b < need->bank_count; b++)
        free_bytes[b] -= need->bank[b] < free_bytes[b] ? need->bank[b] : free_bytes[b];
    for (any = need->any; any > 0u;) {
        uint32_t best = 0u;

        for (uint32_t b = 1; b < need->bank_count; b++)
            if (free_bytes[b] > free_bytes[best])
                best = b;
        if (free_bytes[best] == 0u)
            return;
        {
            const uint64_t step = any < free_bytes[best] ? any : free_bytes[best];

            free_bytes[best] -= step;
            any -= step;
        }
    }
}

void pn_tt_bank_need_state_split(pn_tt_bank_need *need, const uint32_t *state_bank,
                                 uint32_t layers, uint32_t members, uint64_t slice_bytes)
{
    if (!need || !state_bank || need->bank_count == 0u)
        return;
    for (uint32_t layer = 0; layer < layers; layer++)
        for (uint32_t m = 0; m < members; m++)
            pn_tt_bank_need_pin(need, (state_bank[layer] + 1u + m) % need->bank_count,
                                slice_bytes);
}

void pn_tt_bank_need_twins(pn_tt_bank_need *need, const uint32_t *window_bank,
                           const uint32_t *state_bank, uint32_t layers, uint64_t window_bytes,
                           uint64_t state_bytes)
{
    if (!need || !window_bank || !state_bank)
        return;
    for (uint32_t layer = 0; layer < layers; layer++) {
        pn_tt_bank_need_pin(need, window_bank[layer], window_bytes);
        pn_tt_bank_need_pin(need, state_bank[layer], state_bytes);
    }
}
