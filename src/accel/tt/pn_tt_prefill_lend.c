/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_prefill_lend.c
 * @brief L1a: decode GDN cores lent to prefill attention for one prompt (pn_tt_prefill_lend.h).
 *
 * @status L1a.5, default on for dense-FFN qwen35 (pn_tt_prefill_lend.h).
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_prefill_lend.h"

#include <stdio.h>

#include "pn_tt_attention_grid.h"
#include "pn_tt_device_internal.h"
#include "pn_tt_gdn.h"
#include "pn_tt_matmul_group.h"

/** Restore the cores that never took an attention member (their residency retired cleanly). */
static void prefill_lend_restore_from(pn_tt_device *device, uint32_t first)
{
    pn_tt_lend_ledger *ledger = &device->prefill_lend;
    uint32_t index;

    for (index = first; index < ledger->count; index++)
        if (!pn_tt_gdn_restore(device, ledger->member[index], ledger->core[index]))
            fprintf(stderr, "pn_tt: prefill LEND — gdn member %u did not restart on core %u\n",
                    ledger->member[index], ledger->core[index]);
    if (first < ledger->count)
        ledger->count = first;
}

uint32_t pn_tt_prefill_lend_begin(pn_tt_device *device, uint32_t rows, uint32_t min_rows,
                                  bool deep)
{
    pn_tt_lend_ledger *ledger = &device->prefill_lend;
    const uint32_t lendable = pn_tt_gdn_lend_count(device);
    const uint32_t cap = pn_tt_env_u32("PN_TT_PREFILL_LEND_CORES", lendable);
    const double started = pn_tt_now_seconds();
    uint32_t member, armed, members = 0u;

    if (!pn_tt_lend_wanted(rows, min_rows, deep) ||
        ledger->count != 0u || device->attention_grid_count != 0u || lendable == 0u)
        return 0u;
    for (member = 0; member < lendable && ledger->count < cap; member++) {
        const uint32_t core = pn_tt_gdn_lend(device, member);

        if (core != UINT32_MAX && !pn_tt_lend_ledger_add(ledger, member, core) &&
            !pn_tt_gdn_restore(device, member, core))
            fprintf(stderr, "pn_tt: prefill LEND — gdn member %u did not restart on core %u\n",
                    member, core);
    }
    /* L1a.3: the last lent core completes the grid's 64 with decode members 1-15. */
    device->prefill_lend_grid_core = UINT32_MAX;
    if (ledger->count > 1u && pn_tt_env_u32("PN_TT_PREFILL_LEND_GRID", 0u) != 0u &&
        pn_tt_matmul_grid_extend(device, ledger->core[ledger->count - 1u]) != 0u)
        (void)pn_tt_lend_ledger_pop(ledger, &device->prefill_lend_grid_member,
                                    &device->prefill_lend_grid_core);
    armed = ledger->count ? pn_tt_attention_grid_arm(device, ledger->core, ledger->count) : 0u;
    prefill_lend_restore_from(device, armed);
    /* And the decode matmul members 1-15 (idle in prefill; the grid serves the projections):
     * parked, attention booted on their cores, the matmul firmware booted back at the end. */
    if (armed != 0u && device->prefill_lend_grid_core == UINT32_MAX &&
        pn_tt_env_u32("PN_TT_PREFILL_LEND_MEMBERS", 1u) != 0u) {
        uint32_t cores[PN_TT_LEND_MAX];
        const uint32_t lent = pn_tt_matmul_members_lend(device, cores, PN_TT_LEND_MAX);

        members = lent ? pn_tt_attention_grid_arm(device, cores, lent) : 0u;
    }
    fprintf(stderr, "pn_tt: prefill LEND — %u decode GDN + %u decode matmul cores lent to "
                    "attention (%u members), grid %s, %.3f s\n", armed, members,
            pn_tt_attention_queue_members(device),
            device->prefill_lend_grid_core != UINT32_MAX ? "extended to 64" : "unchanged",
            pn_tt_now_seconds() - started);
    return armed + members;
}

bool pn_tt_prefill_lend_active(const pn_tt_device *device)
{
    return device && (device->prefill_lend.count != 0u || device->group.grid_lent ||
                      device->group.members_lent);
}

/** L1a.3: the grid back to 48 and the decode group re-armed; the 64th core to its GDN member.
 *  @return 0, or -1 (named) */
static int prefill_lend_grid_end(pn_tt_device *device)
{
    bool parked = false;
    int rc;

    if (!device->group.grid_lent)
        return 0;
    rc = pn_tt_matmul_grid_retract(device, &parked);
    if (device->prefill_lend_grid_core == UINT32_MAX)
        return rc;
    if (!parked) {
        fprintf(stderr, "pn_tt: prefill LEND — grid core %u did not exit; gdn member %u stays "
                        "down\n", device->prefill_lend_grid_core, device->prefill_lend_grid_member);
        rc = -1;
    } else if (!pn_tt_gdn_restore(device, device->prefill_lend_grid_member,
                                  device->prefill_lend_grid_core)) {
        fprintf(stderr, "pn_tt: prefill LEND — gdn member %u did not restart on core %u\n",
                device->prefill_lend_grid_member, device->prefill_lend_grid_core);
        rc = -1;
    }
    device->prefill_lend_grid_core = UINT32_MAX;
    return rc;
}

int pn_tt_prefill_lend_end(pn_tt_device *device)
{
    pn_tt_lend_ledger *ledger = &device->prefill_lend;
    uint32_t stuck[PN_TT_ATTN_GRID_MAX_MEMBERS], stuck_count = 0u, index, restored = 0u;
    uint32_t members[PN_TT_LEND_MAX], cores[PN_TT_LEND_MAX], count;
    const double started = pn_tt_now_seconds();
    int rc;

    rc = prefill_lend_grid_end(device);
    if (ledger->count == 0u && !device->group.members_lent)
        return rc;
    if (pn_tt_attention_grid_park_all(device, stuck, &stuck_count) != 0)
        rc = -1;
    if (pn_tt_matmul_members_return(device, stuck, stuck_count) != 0)
        rc = -1;
    for (index = 0; index < stuck_count; index++)
        pn_tt_lend_ledger_mark_stuck(ledger, stuck[index]);
    count = pn_tt_lend_ledger_take_restorable(ledger, members, cores);
    for (index = 0; index < count; index++) {
        if (pn_tt_gdn_restore(device, members[index], cores[index])) {
            restored++;
            continue;
        }
        fprintf(stderr, "pn_tt: prefill LEND — gdn member %u did not restart on core %u\n",
                members[index], cores[index]);
        rc = -1;
    }
    fprintf(stderr, "pn_tt: prefill LEND — %u of %u cores back to decode GDN (%u stuck), %.3f s\n",
            restored, count + stuck_count, stuck_count, pn_tt_now_seconds() - started);
    return rc;
}

#endif /* USE_TT */
