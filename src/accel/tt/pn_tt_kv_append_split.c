/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_append_split.c
 * @brief The prefill KV append split (pn_tt_kv_append_split.h): members, and one band's K and V
 *        appends enqueued over them and awaited once.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_kv_append_split.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint32_t pn_tt_kv_append_members_wanted(bool block_float, const char *env_value,
                                        uint32_t max_members)
{
    uint32_t want = PN_TT_KV_APPEND_CORES_DEFAULT;

    if (!block_float || max_members == 0u)
        return 1u;
    if (env_value && *env_value) {
        char *end = NULL;
        const unsigned long parsed = strtoul(env_value, &end, 10);

        if (end && *end == '\0' && parsed > 0ul)
            want = parsed > max_members ? max_members : (uint32_t)parsed;
    }
    return want > max_members ? max_members : want;
}

uint32_t pn_tt_kv_append_list_core(const char *list, uint32_t position)
{
    const char *at = (list && *list) ? list : PN_TT_KV_APPEND_CORE_LIST_DEFAULT;
    uint32_t index = 0u;

    while (*at) {
        char *end = NULL;
        const unsigned long core = strtoul(at, &end, 10);

        if (end == at || (*end != ',' && *end != '\0'))
            return UINT32_MAX;
        if (index++ == position)
            return (uint32_t)core;
        at = *end == ',' ? end + 1 : end;
    }
    return UINT32_MAX;
}

uint32_t pn_tt_kv_append_member_of(uint32_t descriptor, uint32_t members)
{
    return members > 1u ? descriptor % members : 0u;
}

#ifdef USE_TT

#include "pn_tt_group_pull.h"
#include "pn_tt_kv_banks.h"
#include "pn_tt_kv_format.h"
#include "pn_tt_residency.h"

static bool kv_cache_is_block_float(const pn_tt_device *device)
{
    return device->kv_type.k_format != PN_TT_KV_FORMAT_BF16 ||
           device->kv_type.v_format != PN_TT_KV_FORMAT_BF16;
}

void pn_tt_kv_append_members_arm(pn_tt_device *device)
{
    const char *list = getenv(PN_TT_KV_APPEND_CORE_LIST_ENV);
    uint32_t want, up = 1u, position;

    if (!device)
        return;
    device->memory_members = 1u;
    if (!device->memory.running)
        return;
    want = pn_tt_kv_append_members_wanted(kv_cache_is_block_float(device),
                                          getenv(PN_TT_KV_APPEND_CORES_ENV),
                                          PN_TT_KV_APPEND_MAX_MEMBERS);
    for (position = 0u; up < want; position++) {
        const uint32_t core = pn_tt_kv_append_list_core(list, position);

        if (core == UINT32_MAX)
            break;
        if (core == device->memory.core_index)
            continue;
        /* A core another residency holds is refused by name there; try the next one. */
        if (pn_tt_arm_memory_member(device, &device->memory_member[up - 1u], core))
            up++;
    }
    device->memory_members = up;
    if (want > 1u) {
        fprintf(stderr, "pn_tt: KV APPEND SPLIT — asked %u, running %u core(s): %u then",
                want, up, device->memory.core_index);
        for (position = 1u; position < up; position++)
            fprintf(stderr, "%s%u", position > 1u ? "," : " ",
                    device->memory_member[position - 1u].core_index);
        fprintf(stderr, "\n");
    }
}

void pn_tt_kv_append_members_release(pn_tt_device *device)
{
    if (!device)
        return;
    for (uint32_t member = 1u; member < device->memory_members; member++)
        pn_tt_residency_release(&device->memory_member[member - 1u]);
    device->memory_members = 0u;
}

/** Retire and re-arm member `index` (0 = the memory family) on its own core. */
static bool kv_append_member_rearm(pn_tt_device *device, uint32_t index)
{
    pn_tt_residency *member;
    uint32_t core;

    if (index == 0u)
        return pn_tt_attention_memory_rearm(device);
    member = &device->memory_member[index - 1u];
    core = member->core_index;
    fprintf(stderr, "pn_tt: memory member %u refused a descriptor — retiring it by ASKING and "
                    "arming a fresh one on core %u\n", index, core);
    pn_tt_residency_stop(member);
    pn_tt_residency_release(member);
    device->oncard_restarts++;
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: memory member %u did not retire — REFUSING to arm a fresh one "
                        "on core %u\n", index, core);
        return false;
    }
    return pn_tt_arm_memory_member(device, member, core);
}

uint32_t pn_tt_kv_append_pair_words(pn_tt_device *device, uint32_t layer,
                                    const pn_tt_activation *k_source,
                                    const pn_tt_activation *v_source, uint32_t first_position,
                                    uint32_t rows,
                                    uint32_t words[2u * PN_TT_KV_MAX_BANK_GROUPS]
                                                  [FT_TT_OP_MEMORY_WORDS])
{
    uint32_t count = 0u;

    for (uint32_t half = 0u; half < 2u; half++) {
        const bool values = half == 1u;
        ft_tt_op_kv_append_request request;
        uint32_t n = 0u;

        if (pn_tt_kv_append_chunk_request(device, layer, values, values ? v_source : k_source,
                                          first_position, rows, &request) != 0 ||
            pn_tt_kv_append_words_per_group(device, &request, layer, values, &words[count],
                                            &n) != 0) {
            fprintf(stderr, "pn_tt: the %u-position split KV append at %u of layer %u is not "
                            "encodable\n", rows, first_position, layer);
            return 0u;
        }
        count += n;
    }
    return count;
}

pn_tt_group_pull_words pn_tt_kv_append_pull_words(uint32_t index, uint32_t count,
                                                  uint32_t members, uint32_t sequence)
{
    const uint32_t used = count < members ? count : members;
    pn_tt_group_pull_words words =
        pn_tt_group_pull_words_for(pn_tt_kv_append_member_of(index, members), used, sequence,
                                   pn_tt_red("group_pull_short"));

    /* The deal is round-robin, so the members that received one are a prefix. */
    if (index + members < count) {
        words.a3 = 0u;
        words.a4 = 0u;
        words.a5 = sequence - 1u;
    }
    return words;
}

int pn_tt_kv_append_chunk_pair(pn_tt_device *device, uint32_t layer,
                               const pn_tt_activation *k_source,
                               const pn_tt_activation *v_source, uint32_t first_position,
                               uint32_t rows)
{
    uint32_t words[2u * PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS];
    pn_tt_residency *set[PN_TT_KV_APPEND_MAX_MEMBERS];
    bool used[PN_TT_KV_APPEND_MAX_MEMBERS], bad[PN_TT_KV_APPEND_MAX_MEMBERS];
    uint32_t members, count, index, m, sequence = 0u;
    double started;
    bool pulled;

    if (!device)
        return -1;
    members = device->memory_members;
    if (members <= 1u)
        return (pn_tt_kv_append_chunk(device, layer, false, k_source, first_position, rows) !=
                    0 ||
                pn_tt_kv_append_chunk(device, layer, true, v_source, first_position, rows) != 0)
                   ? -1
                   : 0;
    count = pn_tt_kv_append_pair_words(device, layer, k_source, v_source, first_position, rows,
                                       words);
    if (count == 0u)
        return -1;

    memset(used, 0, sizeof(used));
    memset(bad, 0, sizeof(bad));
    for (m = 0u; m < members; m++)
        set[m] = m == 0u ? &device->memory : &device->memory_member[m - 1u];
    /* S1: `memory` posts only once every member that got a descriptor has; one await. */
    pulled = pn_tt_group_pull_decide(&device->memory_pull, pn_tt_group_pull_decode(device), set,
                                     members, count < members ? count : members, "memory");
    if (pulled)
        sequence = pn_tt_group_pull_next(&device->memory_pull);
    started = pn_tt_now_seconds();
    for (index = 0u; index < count; index++) {
        const pn_tt_group_pull_words pull = pn_tt_kv_append_pull_words(index, count, members,
                                                                       sequence);

        m = pn_tt_kv_append_member_of(index, members);
        used[m] = true;
        if (!bad[m] &&
            (pulled ? pn_tt_residency_enqueue_pull(set[m], FT_TT_QUEUE_OP_KV_APPEND,
                                                   words[index], pull.a3, pull.a4, pull.a5)
                    : pn_tt_residency_enqueue(set[m], FT_TT_QUEUE_OP_KV_APPEND, words[index],
                                              1u)) != 0)
            bad[m] = true;
    }
    set[0]->seconds += pn_tt_now_seconds() - started;
    if (pulled) {
        (void)pn_tt_group_pull_await(&device->memory_pull, true, set,
                                     count < members ? count : members, bad, NULL);
    } else {
        for (m = 0u; m < members; m++) {
            const double at = pn_tt_now_seconds();

            if (!used[m] || bad[m])
                continue;
            if (pn_tt_residency_await(set[m], PN_TT_OP_TIMEOUT_MS) != 0)
                bad[m] = true;
            set[m]->seconds += pn_tt_now_seconds() - at;
        }
    }
    /* A refused member is re-armed and runs ITS descriptors again, in order: an append
     * rewrites the same rows with the same bytes, and no other member writes its heads. */
    for (m = 0u; m < members; m++) {
        if (!bad[m])
            continue;
        if (!kv_append_member_rearm(device, m))
            return -1;
        set[m] = m == 0u ? &device->memory : &device->memory_member[m - 1u];
        for (index = 0u; index < count; index++)
            if (pn_tt_kv_append_member_of(index, members) == m &&
                pn_tt_residency_submit(set[m], FT_TT_QUEUE_OP_KV_APPEND, words[index],
                                       PN_TT_OP_TIMEOUT_MS) != 0)
                return -1;
    }
    device->oncard_seconds += pn_tt_now_seconds() - started;
    device->oncard_ops += count;
    return 0;
}

#endif /* USE_TT */
