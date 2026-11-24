/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_splitk.c
 * @brief Split-K decode attention on the card, and the attention program choice (split-K and
 *        `PN_TT_ATTN_FAST`, pn_tt_attn_fast.h). See pn_tt_attention_splitk.h.
 * @status On by default (`PN_TT_ATTN_SPLITK=0` restores lanes). Card gate plan 3.3 passed
 *         (Qwen3.8 K8V4 at 30k, one session).
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_attention_splitk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_attention_lanes.h"
#include "pn_tt_attn_fast.h"
#include "pn_tt_device_internal.h"
#include "pn_tt_kv_format.h"
#include "tt/ft_tt_buffer.h"
#include "tt/ft_tt_queue.h"

#define PN_TT_ATTN_SPLITK_ENV           "PN_TT_ATTN_SPLITK"
#define PN_TT_ATTN_SPLITK_MIN_POS_ENV   "PN_TT_ATTN_SPLITK_MIN_POSITIONS"
#define PN_TT_ATTN_SPLITK_MIN_CHUNK_ENV "PN_TT_ATTN_SPLITK_MIN_CHUNK"
#define PN_TT_ATTN_SPLITK_CORES_ENV     "PN_TT_ATTN_SPLITK_CORE_LIST"
/** The extras when the list is unset: the two cores the Qwen3.8 p150 serve map leaves free
 *  (the kv-append split holds the rest; plan 3.3 measured split-K on these 12 + 2). */
#define PN_TT_ATTN_SPLITK_CORES_DEFAULT "87,128"
/** The one KV type with a split build, and its program. */
#define PN_TT_ATTN_SPLITK_BASE_PROGRAM  "attention_k8v4"
#define PN_TT_ATTN_SPLITK_PROGRAM       "attention_k8v4_split"
#define PN_TT_ATTN_SPLITK_MAX_CORES     (PN_TT_ATTN_MAX_MEMBERS + PN_TT_ATTN_SPLITK_MAX_EXTRA)

static const pn_tt_attn_splitk_config *splitk_config(void)
{
    static pn_tt_attn_splitk_config config;
    static bool read;

    if (!read) {
        if (pn_tt_attn_splitk_config_parse(getenv(PN_TT_ATTN_SPLITK_ENV),
                                           getenv(PN_TT_ATTN_SPLITK_MIN_POS_ENV),
                                           getenv(PN_TT_ATTN_SPLITK_MIN_CHUNK_ENV),
                                           &config) != 0)
            fprintf(stderr, "pn_tt: %s / %s / %s is not a number — split-K stays off\n",
                    PN_TT_ATTN_SPLITK_ENV, PN_TT_ATTN_SPLITK_MIN_POS_ENV,
                    PN_TT_ATTN_SPLITK_MIN_CHUNK_ENV);
        read = true;
    }
    return &config;
}

const char *pn_tt_attention_program_name(const pn_tt_device *device)
{
    if (device->attn_fast_program)
        return device->attn_fast_program;
    return device->attn_splitk ? PN_TT_ATTN_SPLITK_PROGRAM
                               : pn_tt_kv_type_attention_program(device->kv_type);
}

void pn_tt_attention_fast_choose(pn_tt_device *device)
{
    const char *knob = getenv(PN_TT_ATTN_FAST_ENV);
    const char *shipped;
    const char *chosen;
    char why[256];

    device->attn_fast_program = NULL;
    shipped = pn_tt_attention_program_name(device);
    chosen = pn_tt_attn_fast_choose_program(knob, shipped, pn_tt_program_is_present, why,
                                            sizeof(why));
    if (why[0])
        fprintf(stderr, "pn_tt: REFUSED attention fast-math: %s\n", why);
    if (!chosen || !shipped || strcmp(chosen, shipped) == 0)
        return;
    device->attn_fast_program = chosen;
    fprintf(stderr, "pn_tt: attention fast-math %s=%s — every attention core runs `%s` (decode "
                    "too) instead of `%s`\n", PN_TT_ATTN_FAST_ENV, knob, chosen, shipped);
}

void pn_tt_attention_splitk_choose(pn_tt_device *device)
{
    const char *base = pn_tt_kv_type_attention_program(device->kv_type);

    const char *knob = getenv(PN_TT_ATTN_SPLITK_ENV);

    device->attn_splitk = false;
    if (!splitk_config()->enabled)
        return;
    if (!base || strcmp(base, PN_TT_ATTN_SPLITK_BASE_PROGRAM) != 0) {
        /* On by default: only a cache type without a split build stays quiet unless asked. */
        if (knob && *knob)
            fprintf(stderr, "pn_tt: split-K asked for, but only a k8v4 KV cache has a split "
                            "program (this one is %s) — lanes\n",
                    pn_tt_kv_type_name(device->kv_type));
        return;
    }
    if (!pn_tt_program_is_present(PN_TT_ATTN_SPLITK_PROGRAM)) {
        fprintf(stderr, "pn_tt: split-K asked for, but `%s` is not in the program root — "
                        "lanes\n", PN_TT_ATTN_SPLITK_PROGRAM);
        return;
    }
    device->attn_splitk = true;
}

/** Arm the split-only extras from the core list; a core that will not come up is skipped. */
static void splitk_arm_extras(pn_tt_device *device)
{
    const char *list = getenv(PN_TT_ATTN_SPLITK_CORES_ENV);
    char *end = NULL;

    if (!list)
        list = PN_TT_ATTN_SPLITK_CORES_DEFAULT; /* set but empty = no extras */

    device->attention_splitk_extras = 0u;
    while (list && *list && device->attention_splitk_extras < PN_TT_ATTN_SPLITK_MAX_EXTRA) {
        const unsigned long core = strtoul(list, &end, 10);
        pn_tt_residency *extra = &device->attention_splitk_extra[device->attention_splitk_extras];

        if (end == list)
            break;
        if (pn_tt_arm_attention_member(device, extra, (uint32_t)core))
            device->attention_splitk_extras++;
        else
            fprintf(stderr, "pn_tt: split-K extra core %lu did not arm — skipped\n", core);
        list = *end == ',' ? end + 1 : end;
    }
}

/** The partial buffer (every slot) on any bank, and the zero mask on the constants' bank:
 *  a ROWS descriptor carries one coordinate for the constants and its mask. */
static bool splitk_alloc(pn_tt_device *device)
{
    const size_t partials = (size_t)PN_TT_ATTN_SPLITK_MAX_PARTIALS *
                            ft_tt_op_attention_partial_bytes(device->attn_dim_tiles,
                                                             PN_TT_TILE_BF16);
    ft_tt_buffer buffer;

    if (pn_tt_dram_alloc_any(device, partials, &buffer) != 0)
        return false;
    device->attn_partial_bank = buffer.logical_bank;
    device->attn_partial_address = (uint32_t)buffer.address;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, device->attn_const_bank,
                           PN_TT_TILE_BF16, &buffer) != FT_TT_OK)
        return false;
    device->attn_zero_mask_address = (uint32_t)buffer.address;
    return pn_tt_zero_dram(device, device->attn_const_bank, device->attn_zero_mask_address,
                           PN_TT_TILE_BF16) == 0;
}

void pn_tt_attention_splitk_arm(pn_tt_device *device)
{
    if (!device->attn_splitk)
        return;
    splitk_arm_extras(device);
    if (!splitk_alloc(device)) {
        device->attn_partial_address = 0u;
        fprintf(stderr, "pn_tt: split-K's partial buffer did not allocate — the split "
                        "program serves lanes only\n");
        return;
    }
    fprintf(stderr, "pn_tt: split-K decode attention ARMED — %u core(s) (%u member(s) + %u "
                    "extra), switch-over %u positions, chunks >= %u positions\n",
            pn_tt_attention_splitk_cores(device), device->attention_members,
            device->attention_splitk_extras, splitk_config()->min_positions,
            splitk_config()->min_chunk_positions);
}

void pn_tt_attention_splitk_release(pn_tt_device *device)
{
    for (uint32_t i = 0; i < device->attention_splitk_extras; i++)
        pn_tt_residency_release(&device->attention_splitk_extra[i]);
    device->attention_splitk_extras = 0u;
}

uint32_t pn_tt_attention_splitk_cores(const pn_tt_device *device)
{
    if (!device->attn_splitk || device->attn_partial_address == 0u)
        return 0u;
    return device->attention_members + device->attention_splitk_extras;
}

static pn_tt_residency *splitk_core(pn_tt_device *device, uint32_t core)
{
    if (core == 0u)
        return &device->attention;
    if (core < device->attention_members)
        return &device->attention_member[core - 1u];
    return &device->attention_splitk_extra[core - device->attention_members];
}

static bool splitk_core_rearm(pn_tt_device *device, uint32_t core)
{
    pn_tt_residency *extra;
    uint32_t index;

    if (core == 0u)
        return pn_tt_attention_main_rearm(device);
    if (core < device->attention_members)
        return pn_tt_attention_member_rearm(device, core);
    extra = splitk_core(device, core);
    index = extra->core_index;
    pn_tt_residency_stop(extra);
    pn_tt_residency_release(extra);
    device->oncard_restarts++;
    return pn_tt_residency_retired(extra) && pn_tt_arm_attention_member(device, extra, index);
}

/** Descriptors [0, count) on cores [0, count): enqueue all, await all, re-arm and resubmit a
 *  refusal alone after every other core was awaited (the lanes round, wider). */
static int splitk_round(pn_tt_device *device, uint32_t words[][FT_TT_OP_ATTENTION_WORDS],
                        uint32_t count)
{
    const double started = pn_tt_now_seconds();
    bool bad[PN_TT_ATTN_SPLITK_MAX_CORES];
    int failed = 0;

    memset(bad, 0, sizeof(bad));
    for (uint32_t k = 0; k < count; k++)
        if (pn_tt_residency_enqueue(splitk_core(device, k), FT_TT_QUEUE_OP_ATTENTION, words[k],
                                    1u) != 0)
            failed = bad[k] = 1;
    for (uint32_t k = 0; k < count; k++)
        if (!bad[k] && pn_tt_residency_await(splitk_core(device, k), PN_TT_OP_TIMEOUT_MS) != 0)
            failed = bad[k] = 1;
    for (uint32_t k = 0; failed && k < count; k++)
        if (bad[k] && (!splitk_core_rearm(device, k) ||
                       pn_tt_residency_submit(splitk_core(device, k), FT_TT_QUEUE_OP_ATTENTION,
                                              words[k], PN_TT_OP_TIMEOUT_MS) != 0))
            return -1;
    device->oncard_seconds += pn_tt_now_seconds() - started;
    device->oncard_ops += count;
    return 0;
}

static ft_tt_op_operand splitk_partial_operand(const pn_tt_device *device, uint32_t slot)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[device->attn_partial_bank].x;
    operand.y = (uint16_t)device->dram[device->attn_partial_bank].y;
    operand.address = device->attn_partial_address +
                      slot * ft_tt_op_attention_partial_bytes(device->attn_dim_tiles,
                                                              PN_TT_TILE_BF16);
    operand.tile_bytes = PN_TT_TILE_BF16;
    return operand;
}

/** PARTIAL `slot`: the group's lanes request over one chunk. */
static int splitk_partial_words(const pn_tt_device *device,
                                const ft_tt_op_attention_request *lanes,
                                const pn_tt_attn_splitk_slot *slot,
                                uint32_t words[FT_TT_OP_ATTENTION_WORDS])
{
    ft_tt_op_attention_request request = *lanes;
    const uint32_t tiles = slot->first_tile * request.dim_tiles;

    request.flags |= FT_TT_OP_ATTENTION_FLAG_PARTIAL;
    request.keys.address += tiles * request.keys.tile_bytes;
    request.values.address += tiles * request.values.tile_bytes;
    request.kv_positions = slot->kv_positions;
    request.capacity_tiles = slot->capacity_tiles;
    if ((request.flags & FT_TT_OP_ATTENTION_FLAG_ROWS) != 0u && !slot->last)
        request.rows_mask_address = device->attn_zero_mask_address;
    request.output = splitk_partial_operand(device, slot->slot);
    return ft_tt_op_attention_parameters(&request, words) == FT_TT_OK ? 0 : -1;
}

/** MERGE for one group: its `chunks` contiguous slots into the lanes request's output. RED
 *  `attn_splitk_drop_chunk`: one chunk fewer is folded — every output moves, must DIFFER. */
static int splitk_merge_words(const pn_tt_device *device, const ft_tt_op_attention_request *lanes,
                              uint32_t first_slot, uint32_t chunks,
                              uint32_t words[FT_TT_OP_ATTENTION_WORDS])
{
    ft_tt_op_attention_request request;

    memset(&request, 0, sizeof(request));
    request.q_rows = lanes->q_rows;
    request.dim_tiles = lanes->dim_tiles;
    request.block_tiles = lanes->block_tiles;
    request.capacity_tiles = lanes->capacity_tiles;
    request.query_row = lanes->query_row;
    request.positions = lanes->positions;
    request.flags = FT_TT_OP_ATTENTION_FLAG_MERGE | (lanes->flags & FT_TT_OP_ATTENTION_FLAG_ROWS);
    request.partial_count = pn_tt_red("attn_splitk_drop_chunk") ? chunks - 1u : chunks;
    request.query = splitk_partial_operand(device, first_slot);
    request.output = lanes->output;
    return ft_tt_op_attention_parameters(&request, words) == FT_TT_OK ? 0 : -1;
}

/** Every bank run of a head: one, from position 0. Each head's K (and V) is one contiguous
 *  run in its bank group's bank (pn_tt_kv_banks.h: head-major, F2), so no chunk can cross a
 *  bank whatever the plan; a layout that splits a head would list its runs here. */
static uint32_t splitk_head_runs(const pn_tt_device *device, pn_tt_attn_splitk_run *runs)
{
    runs[0].first_position = 0u;
    runs[0].end_position = device->kv_layout.capacity_positions;
    return 1u;
}

/** The planner's view of one call on this device; `runs` backs geometry->runs. */
static void splitk_geometry_of(const pn_tt_device *device, uint32_t live, uint32_t shortest_live,
                               uint32_t groups, uint32_t block_tiles, uint32_t capacity_tiles,
                               pn_tt_attn_splitk_run *runs, pn_tt_attn_splitk_geometry *geometry)
{
    memset(geometry, 0, sizeof(*geometry));
    geometry->live_positions = live;
    geometry->shortest_live = shortest_live;
    geometry->kv_heads = groups;
    geometry->cores = pn_tt_attention_splitk_cores(device);
    geometry->block_tiles = block_tiles;
    geometry->capacity_tiles = capacity_tiles;
    geometry->runs = runs;
    geometry->run_count = splitk_head_runs(device, runs);
}

uint64_t pn_tt_attention_splitk_shape_key(const pn_tt_device *device, uint32_t live_positions)
{
    pn_tt_attn_splitk_run runs[PN_TT_ATTN_SPLITK_MAX_RUNS];
    pn_tt_attn_splitk_geometry geometry;
    pn_tt_attn_splitk_plan plan;

    if (!device || live_positions == 0u || pn_tt_attention_splitk_cores(device) < 2u)
        return 0u;
    splitk_geometry_of(device, live_positions, live_positions, device->attn_kv_heads,
                       device->attn_block_tiles, device->kv_layout.capacity_tiles, runs,
                       &geometry);
    if (pn_tt_attn_splitk_plan_make(splitk_config(), &geometry, &plan) != 0)
        return 0u;
    return pn_tt_attn_splitk_plan_key(&plan);
}

int pn_tt_attention_splitk_issue(pn_tt_device *device, const ft_tt_op_attention_request *requests,
                                 uint32_t groups, uint32_t shortest_live)
{
    uint32_t words[PN_TT_ATTN_SPLITK_MAX_CORES][FT_TT_OP_ATTENTION_WORDS];
    pn_tt_attn_splitk_run runs[PN_TT_ATTN_SPLITK_MAX_RUNS];
    pn_tt_attn_splitk_geometry geometry;
    pn_tt_attn_splitk_plan plan;
    const char *broken;

    if (!device || !requests || groups == 0u)
        return -1;
    if (pn_tt_attention_splitk_cores(device) < 2u)
        return 1;
    splitk_geometry_of(device, requests[0].kv_positions, shortest_live, groups,
                       requests[0].block_tiles, requests[0].capacity_tiles, runs, &geometry);
    if (pn_tt_attn_splitk_plan_make(splitk_config(), &geometry, &plan) != 0)
        return -1;
    if (plan.chunk_count == 0u)
        return 1;
    broken = pn_tt_attn_splitk_plan_check(splitk_config(), &geometry, &plan);
    if (broken) {
        fprintf(stderr, "pn_tt: split-K plan refused (%s) at %u positions\n", broken,
                geometry.live_positions);
        return -1;
    }
    for (uint32_t round = 0; round < plan.rounds; round++) {
        uint32_t count = 0u;

        for (uint32_t d = round * geometry.cores; d < plan.partials && count < geometry.cores;
             d++, count++) {
            pn_tt_attn_splitk_slot slot;

            if (pn_tt_attn_splitk_slot_of(&geometry, &plan, d, &slot) != 0 ||
                splitk_partial_words(device, &requests[slot.group], &slot, words[count]) != 0)
                return -1;
        }
        if (splitk_round(device, words, count) != 0)
            return -1;
    }
    for (uint32_t first = 0; first < groups; first += geometry.cores) {
        const uint32_t count = groups - first < geometry.cores ? groups - first : geometry.cores;

        for (uint32_t k = 0; k < count; k++)
            if (splitk_merge_words(device, &requests[first + k], (first + k) * plan.chunk_count,
                                   plan.chunk_count, words[k]) != 0)
                return -1;
        if (splitk_round(device, words, count) != 0)
            return -1;
    }
    return 0;
}

#endif /* USE_TT */
