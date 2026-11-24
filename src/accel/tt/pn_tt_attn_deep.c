/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_deep.c
 * @brief Deep-queued prefill attention's schedule (pn_tt_attn_deep.h).
 *
 * @status Off by default (`PN_TT_ATTN_DEEP=1` on).
 * @issues None
 * @todo None
 */
#include "pn_tt_attn_deep.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

uint32_t pn_tt_attn_deep_member(uint32_t index, uint32_t members)
{
    return members == 0u ? 0u : index % members;
}

uint32_t pn_tt_attn_deep_plan(uint32_t bands, uint32_t heads, uint32_t members,
                              pn_tt_attn_deep_unit *out, uint32_t max)
{
    uint32_t count = 0u, band, head;

    if (!out || bands == 0u || heads == 0u || members == 0u || bands > max / heads)
        return 0u;
    for (band = 0; band < bands; band++)
        for (head = 0; head < heads; head++) {
#ifdef PN_TT_ATTN_DEEP_RED
            /* RED: the last head of the last band is never queued. Must FAIL the check. */
            if (band + 1u == bands && head + 1u == heads)
                continue;
#endif
            out[count].band = band;
            out[count].head = head;
            out[count].member = pn_tt_attn_deep_member(count, members);
            count++;
        }
    return count;
}

const char *pn_tt_attn_deep_plan_check(const pn_tt_attn_deep_unit *units, uint32_t count,
                                       uint32_t bands, uint32_t heads, uint32_t members)
{
    uint32_t load[PN_TT_ATTN_DEEP_MAX_MEMBERS] = { 0 };
    uint32_t low = UINT32_MAX, high = 0u, index;
    unsigned char *seen;
    const char *defect = NULL;

    if (!units || bands == 0u || heads == 0u || members == 0u ||
        members > PN_TT_ATTN_DEEP_MAX_MEMBERS)
        return "bad arguments";
    if (count != bands * heads)
        return "unit count is not bands x heads";
    seen = calloc((size_t)bands * heads, 1u);
    if (!seen)
        return "out of memory";
    for (index = 0; index < count && !defect; index++) {
        const pn_tt_attn_deep_unit *unit = &units[index];

        if (unit->band >= bands || unit->head >= heads)
            defect = "a unit is out of range";
        else if (unit->member >= members)
            defect = "a unit's member is out of range";
        else if (index > 0u && unit->band < units[index - 1u].band)
            defect = "bands are not in issue order";
        else if (seen[(size_t)unit->band * heads + unit->head]++)
            defect = "a (band, head) is queued twice";
        else
            load[unit->member]++;
    }
    for (index = 0; index < members && !defect; index++) {
        if (load[index] < low)
            low = load[index];
        if (load[index] > high)
            high = load[index];
    }
    if (!defect && high - low > 1u)
        defect = "member loads differ by more than one";
    free(seen);
    return defect;
}

int pn_tt_attn_deep_flow_begin(pn_tt_attn_deep_flow *flow, uint32_t members, uint32_t depth)
{
    if (!flow || members == 0u || members > PN_TT_ATTN_DEEP_MAX_MEMBERS || depth == 0u)
        return -1;
    memset(flow, 0, sizeof(*flow));
    flow->members = members;
    flow->depth = depth;
    return 0;
}

int pn_tt_attn_deep_issue_band(pn_tt_attn_deep_flow *flow, uint32_t band, uint32_t heads,
                               const pn_tt_attn_deep_io *io)
{
    uint32_t head;

    if (!flow || !io || !io->enqueue || !io->settle || flow->members == 0u)
        return -1;
    for (head = 0; head < heads; head++) {
        const uint32_t member = pn_tt_attn_deep_member(flow->cursor, flow->members);

#ifndef PN_TT_ATTN_DEEP_RED_NO_SETTLE
        /* A WAVE, NOT ONE MEMBER. The settle waits for everything a member holds, so settling
         * only the full one left it a single descriptor until the issue came round again —
         * every member starved after the first lap (M: band loop 49.6 vs 47.7 s at 32k). The
         * round robin fills every member evenly, so when one is full all are: settle them all
         * and refill the next wave whole. */
        if (flow->outstanding[member] >= flow->depth) {
            uint32_t other;

            for (other = 0; other < flow->members; other++) {
                if (flow->outstanding[other] == 0u)
                    continue;
                if (io->settle(io->context, other) != 0)
                    return -1;
                flow->outstanding[other] = 0u;
            }
        }
#endif
        if (io->enqueue(io->context, member, band, head) != 0)
            return -1;
        flow->outstanding[member]++;
        flow->cursor++;
    }
    return 0;
}

int pn_tt_attn_deep_flow_drain(pn_tt_attn_deep_flow *flow, const pn_tt_attn_deep_io *io)
{
    uint32_t member;
    int rc = 0;

    if (!flow || !io || !io->settle)
        return -1;
    for (member = 0; member < flow->members; member++) {
        if (flow->outstanding[member] == 0u)
            continue;
        if (io->settle(io->context, member) != 0)
            rc = -1;
        flow->outstanding[member] = 0u;
    }
    flow->cursor = 0u;
    return rc;
}
