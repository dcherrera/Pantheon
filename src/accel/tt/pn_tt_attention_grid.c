/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_grid.c
 * @brief The attention-on-the-grid swap probe (pn_tt_attention_grid.h).
 *
 * @status Probe and C (2026-11-18); C off by default.
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_attention_grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_attention_splitk.h"
#include "pn_tt_device_internal.h"
#include "tt/ft_tt_launch.h"
#include "tt/ft_tt_queue.h"

bool pn_tt_attention_grid_cycle_wanted(void)
{
    return pn_tt_env_u32("PN_TT_GRID_CYCLE_ATTN", 0u) != 0u;
}

/** Retire `member`, then hold its RISCs in reset when its triad exited. @return true parked */
static bool attention_grid_park_one(pn_tt_device *device, pn_tt_residency *member)
{
    uint32_t pending = PN_TT_ONCARD_TRIAD_MASK;

    pn_tt_residency_release(member);
    if (!pn_tt_residency_retired(member) ||
        ft_tt_queue_all_processors_exited(&device->runtime.noc, &member->queue,
                                          PN_TT_ONCARD_TRIAD_MASK, &pending) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: GRID ATTENTION core %u did not exit (0x%x) — NOT parked\n",
                member->core_index, pending);
        return false;
    }
    if (ft_tt_launch_assert_reset(&device->runtime.noc, member->core_x, member->core_y,
                                  FT_TT_SOFT_RESET_ALL_RISCS) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: GRID ATTENTION core %u could not be parked\n",
                member->core_index);
        return false;
    }
    member->core_booted = false;
    return true;
}

int pn_tt_attention_grid_cycle(pn_tt_device *device, const uint32_t *cores, uint32_t count,
                               pn_tt_attention_grid_cycle_time *out_time)
{
    pn_tt_residency *members;
    uint32_t armed = 0u, index;
    double at;
    int rc = 0;

    if (!device || !cores || count == 0u || !out_time || !device->attention.running)
        return -1;
    members = calloc(count, sizeof(*members));
    if (!members)
        return -1;
    at = pn_tt_now_seconds();
    for (index = 0; index < count; index++) {
        if (!pn_tt_arm_attention_member(device, &members[index], cores[index])) {
            fprintf(stderr, "pn_tt: GRID ATTENTION core %u did not arm\n", cores[index]);
            rc = -1;
            break;
        }
        armed++;
    }
    out_time->arm_seconds = pn_tt_now_seconds() - at;
    at = pn_tt_now_seconds();
    for (index = 0; index < armed; index++)
        if (!attention_grid_park_one(device, &members[index]))
            rc = -1;
    out_time->park_seconds = pn_tt_now_seconds() - at;
    free(members);
    return rc;
}

uint32_t pn_tt_attention_grid_arm(pn_tt_device *device, const uint32_t *cores, uint32_t count)
{
    uint32_t index, armed = 0u;

    if (!device || !cores || count == 0u || !device->attention.running)
        return 0u;
    if (!device->attention_grid) {
        device->attention_grid = calloc(PN_TT_ATTN_GRID_MAX_MEMBERS, sizeof(pn_tt_residency));
        if (!device->attention_grid)
            return 0u;
    }
    /* APPENDS after the members already borrowed (L1a arms two sets in turn). */
    for (index = 0; index < count && device->attention_grid_count < PN_TT_ATTN_GRID_MAX_MEMBERS;
         index++) {
        pn_tt_residency *member = &device->attention_grid[device->attention_grid_count];

        memset(member, 0, sizeof(*member));
        if (!pn_tt_arm_attention_member(device, member, cores[index])) {
            fprintf(stderr, "pn_tt: GRID ATTENTION core %u did not arm — %u borrowed\n",
                    cores[index], device->attention_grid_count);
            break;
        }
        device->attention_grid_count++;
        armed++;
    }
    return armed;
}

int pn_tt_attention_grid_park_all(pn_tt_device *device, uint32_t *stuck, uint32_t *stuck_count)
{
    uint32_t index;
    int rc = 0;

    if (stuck_count)
        *stuck_count = 0u;
    if (!device)
        return -1;
    for (index = 0; index < device->attention_grid_count; index++) {
        /* Captured first: release zeroes the handle. */
        const uint32_t core = device->attention_grid[index].core_index;

        if (attention_grid_park_one(device, &device->attention_grid[index]))
            continue;
        rc = -1;
        if (stuck && stuck_count)
            stuck[(*stuck_count)++] = core;
    }
    device->attention_grid_count = 0u;
    return rc;
}

uint32_t pn_tt_attention_queue_members(const pn_tt_device *device)
{
    return device->attention_members + device->attention_grid_count;
}

pn_tt_residency *pn_tt_attention_queue_member(pn_tt_device *device, uint32_t m)
{
    if (m == 0u)
        return &device->attention;
    if (m < device->attention_members)
        return &device->attention_member[m - 1u];
    return &device->attention_grid[m - device->attention_members];
}

#endif /* USE_TT */
