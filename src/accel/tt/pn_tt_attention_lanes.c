/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_lanes.c
 * @brief Decode attention's KV groups over several attention cores. See
 *        pn_tt_attention_lanes.h.
 * @status Default 4 lanes (`PN_TT_ATTN_DECODE_LANES=1` off).
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_attention_lanes.h"

#include <stdio.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_group_pull.h"
#include "tt/ft_tt_queue.h"

#define PN_TT_ATTN_DECODE_LANES_ENV "PN_TT_ATTN_DECODE_LANES"
#define PN_TT_ATTN_DECODE_LANES_DEFAULT 4u

uint32_t pn_tt_attention_decode_lanes(const pn_tt_device *device)
{
    static int asked = -1;
    uint32_t lanes;

    if (asked < 0)
        asked = (int)pn_tt_env_u32(PN_TT_ATTN_DECODE_LANES_ENV,
                                     PN_TT_ATTN_DECODE_LANES_DEFAULT);
    if (!device || asked <= 1)
        return 1u;
    lanes = (uint32_t)asked;
    if (lanes > device->attention_members)
        lanes = device->attention_members;
    if (lanes > device->attn_kv_heads)
        lanes = device->attn_kv_heads;
    return lanes < 1u ? 1u : lanes;
}

static pn_tt_residency *attention_lane(pn_tt_device *device, uint32_t lane)
{
    return lane == 0u ? &device->attention : &device->attention_member[lane - 1u];
}

static bool attention_lane_rearm(pn_tt_device *device, uint32_t lane)
{
    return lane == 0u ? pn_tt_attention_main_rearm(device)
                      : pn_tt_attention_member_rearm(device, lane);
}

/** RED `attn_lanes_no_await`: lanes 1.. are never awaited, so nothing orders the output's
 *  consumer after them. Timing-dependent by design; reported, not gated on. */
static bool attention_lanes_red_no_await(void)
{
    return pn_tt_red("attn_lanes_no_await");
}

/** S1: is this round's lane 0 to post for the others (pn_tt_group_pull.h)? Not while the
 *  residencies defer their awaits (the lanes have no settle that knows the pull) nor under
 *  the no-await red. The table names every attention member, so a short round pulls fewer. */
static bool attention_lanes_pulled(pn_tt_device *device, uint32_t count,
                                   pn_tt_residency **set)
{
    uint32_t lane;

    for (lane = 0; lane < device->attention_members && lane < PN_TT_ATTN_MAX_MEMBERS; lane++)
        set[lane] = attention_lane(device, lane);
    return !device->attention.defer_await && !attention_lanes_red_no_await() &&
           pn_tt_group_pull_decide(&device->attention_pull, pn_tt_group_pull_decode(device),
                                   set, device->attention_members, count, "attention");
}

/** One round: group `first + k` on lane k for k < count; enqueue all, then await all (lane 0
 *  alone when it posts for the round). */
static int attention_lanes_round(pn_tt_device *device,
                                 uint32_t words[][FT_TT_OP_ATTENTION_WORDS], uint32_t first,
                                 uint32_t count)
{
    const double started = pn_tt_now_seconds();
    pn_tt_residency *set[PN_TT_ATTN_MAX_MEMBERS];
    bool bad[PN_TT_ATTN_MAX_MEMBERS];
    uint32_t lane;
    int failed = 0;

    memset(bad, 0, sizeof(bad));
    if (attention_lanes_pulled(device, count, set)) {
        (void)pn_tt_group_pull_enqueue(&device->attention_pull, true, set, count,
                                       FT_TT_QUEUE_OP_ATTENTION, words[first], bad);
        failed = pn_tt_group_pull_await(&device->attention_pull, true, set, count, bad, NULL);
    } else {
        for (lane = 0; lane < count; lane++)
            if (pn_tt_residency_enqueue(attention_lane(device, lane), FT_TT_QUEUE_OP_ATTENTION,
                                        words[first + lane], 1u) != 0)
                failed = bad[lane] = 1;
        for (lane = 0; lane < count; lane++) {
            if (bad[lane] || (lane > 0u && attention_lanes_red_no_await()))
                continue;
            if (pn_tt_residency_await(attention_lane(device, lane), PN_TT_OP_TIMEOUT_MS) != 0)
                failed = bad[lane] = 1;
        }
    }
    for (lane = 0; failed && lane < count; lane++) {
        if (!bad[lane])
            continue;
        if (!attention_lane_rearm(device, lane) ||
            pn_tt_residency_submit(attention_lane(device, lane), FT_TT_QUEUE_OP_ATTENTION,
                                   words[first + lane], PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    device->oncard_seconds += pn_tt_now_seconds() - started;
    device->oncard_ops += count;
    return 0;
}

int pn_tt_attention_issue_lanes(pn_tt_device *device,
                                uint32_t words[][FT_TT_OP_ATTENTION_WORDS], uint32_t groups)
{
    const uint32_t lanes = pn_tt_attention_decode_lanes(device);
    uint32_t first;

    if (!device || !words || groups == 0u || groups > PN_TT_ATTN_LANE_MAX_GROUPS)
        return -1;
    for (first = 0; first < groups; first += lanes) {
        const uint32_t count = groups - first < lanes ? groups - first : lanes;

        if (attention_lanes_round(device, words, first, count) != 0)
            return -1;
    }
    return 0;
}

#endif /* USE_TT */
