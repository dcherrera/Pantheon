/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_banks.c
 * @brief KV bank groups: a layer's K / V regions split by KV head over banks. See
 *        pn_tt_kv_banks.h.
 * @status Default on, G 4 (`PN_TT_KV_BANK_GROUPS=1` off).
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_kv_banks.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/ft_tt_buffer.h"

#define PN_TT_KV_BANK_GROUPS_ENV "PN_TT_KV_BANK_GROUPS"
#define PN_TT_KV_APPEND_BANKED_ENV "PN_TT_KV_APPEND_BANKED"
/* DEFAULT 4 since 2026-10-26 (David): with the banked append it costs decode nothing and
 * cuts serve TTFT at 28.7k 52.6 -> 40.7 s (with the 16 attention members). */
#define PN_TT_KV_BANK_GROUPS_DEFAULT 4u

/** The default G: the largest of 4, 2, 1 that divides the KV heads (4 for Ornith's 4). */
static uint32_t kv_bank_groups_default(uint32_t heads)
{
    uint32_t groups = PN_TT_KV_BANK_GROUPS_DEFAULT;

    while (groups > 1u && heads % groups != 0u)
        groups /= 2u;
    return groups;
}

const ft_tt_kv_cache_layout *pn_tt_kv_layout_of(const pn_tt_device *device, bool values)
{
    return values ? &device->kv_v_layout : &device->kv_layout;
}

static const ft_tt_kv_cache_layout *kv_group_layout_of(const pn_tt_device *device, bool values)
{
    return values ? &device->kv_v_group_layout : &device->kv_group_layout;
}

int pn_tt_kv_banks_configure(pn_tt_device *device)
{
    const uint32_t heads = device->kv_layout.head_count;
    uint32_t groups = pn_tt_env_u32(PN_TT_KV_BANK_GROUPS_ENV, kv_bank_groups_default(heads));

    if (groups < 1u || groups > PN_TT_KV_MAX_BANK_GROUPS || heads % groups != 0u) {
        fprintf(stderr, "pn_tt: PN_TT_KV_BANK_GROUPS=%u must divide the %u KV heads and be at "
                        "most %u — one bank a region\n", groups, heads,
                PN_TT_KV_MAX_BANK_GROUPS);
        groups = 1u;
    }
    device->kv_bank_groups = groups;
    /* One BANKED descriptor a region (foundry FT_TT_OP_MEMORY_FLAG_DST_BANKED) instead of one
     * per group: the decode cost of the groups. `PN_TT_KV_APPEND_BANKED=0` restores G. */
    device->kv_append_banked = groups > 1u && groups <= FT_TT_OP_MEMORY_MAX_BANKS &&
                               pn_tt_env_u32(PN_TT_KV_APPEND_BANKED_ENV, 1u) != 0u;
    if (ft_tt_kv_cache_layout_init(heads / groups, device->kv_layout.head_dim,
                                   device->kv_layout.capacity_positions,
                                   device->kv_layout.tile_bytes,
                                   &device->kv_group_layout) != FT_TT_OK ||
        ft_tt_kv_cache_layout_init(heads / groups, device->kv_v_layout.head_dim,
                                   device->kv_v_layout.capacity_positions,
                                   device->kv_v_layout.tile_bytes,
                                   &device->kv_v_group_layout) != FT_TT_OK)
        return -1;
    if (groups > 1u)
        fprintf(stderr, "pn_tt: KV BANK GROUPS — each layer's K and V split into %u groups of "
                        "%u head(s), one bank each; append %s\n", groups, heads / groups,
                device->kv_append_banked ? "ONE banked descriptor" : "one descriptor a group");
    return 0;
}

/** One group's allocation in the next bank of the rotation (the strip cursor's). */
static bool kv_group_alloc(pn_tt_device *device, size_t bytes, uint32_t *out_bank,
                           uint32_t *out_address)
{
    ft_tt_buffer buffer;
    const uint32_t bank = device->strip_bank_cursor++ % device->dram_bank_count;

    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes, &buffer) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full allocating %zu bytes for a KV bank group\n", bytes);
        return false;
    }
    *out_bank = buffer.logical_bank;
    *out_address = (uint32_t)buffer.address;
    return pn_tt_zero_dram(device, *out_bank, *out_address, bytes) == 0;
}

/**
 * Pad the next `groups` banks of the rotation up to one common cursor, so the groups about
 * to be allocated there share an address (the BANKED append has one base word). The banks
 * draw equally after the first set, so the padding is paid about once.
 */
static bool kv_group_set_align(pn_tt_device *device, uint32_t groups)
{
    uint64_t top = 0u;

    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t bank = (device->strip_bank_cursor + g) % device->dram_bank_count;

        if (device->allocator.dram[bank].cursor > top)
            top = device->allocator.dram[bank].cursor;
    }
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t bank = (device->strip_bank_cursor + g) % device->dram_bank_count;
        const uint64_t gap = top - device->allocator.dram[bank].cursor;
        ft_tt_buffer pad;

        if (gap != 0u && ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank,
                                            (size_t)gap, &pad) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: DRAM full padding bank %u for a KV bank group set\n", bank);
            return false;
        }
    }
    return true;
}

/** One layer's K (or V) groups, the next `groups` banks; aligned when appends are banked. */
static bool kv_group_set_alloc(pn_tt_device *device, size_t group_bytes, uint32_t groups,
                               uint32_t *banks, uint32_t *addresses)
{
    if (device->kv_append_banked && !kv_group_set_align(device, groups))
        return false;
    for (uint32_t g = 0; g < groups; g++)
        if (!kv_group_alloc(device, group_bytes, &banks[g], &addresses[g]))
            return false;
    for (uint32_t g = 1; g < groups && device->kv_append_banked; g++)
        if (addresses[g] != addresses[0]) {
            fprintf(stderr, "pn_tt: KV bank group set not aligned (%#x vs %#x) — banked append "
                            "off, one descriptor a group\n", addresses[g], addresses[0]);
            device->kv_append_banked = false;
        }
    return true;
}

/** The fullest bank of the next `groups` banks of the rotation, starting `skip` banks on. */
static uint64_t kv_group_set_top(const pn_tt_device *device, uint32_t skip, uint32_t groups)
{
    uint64_t top = 0u;

    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t bank = (device->strip_bank_cursor + skip + g) % device->dram_bank_count;

        if (device->allocator.dram[bank].cursor > top)
            top = device->allocator.dram[bank].cursor;
    }
    return top;
}

bool pn_tt_kv_layer_v_first(uint64_t k_group_bytes, uint64_t v_group_bytes,
                            uint64_t first_set_top, uint64_t second_set_top)
{
    if (k_group_bytes == v_group_bytes)
        return false;
    return k_group_bytes > v_group_bytes ? first_set_top > second_set_top
                                         : second_set_top > first_set_top;
}

static bool kv_layer_v_first(const pn_tt_device *device, uint32_t groups)
{
    return pn_tt_kv_layer_v_first(device->kv_group_layout.region_bytes,
                                  device->kv_v_group_layout.region_bytes,
                                  kv_group_set_top(device, 0u, groups),
                                  kv_group_set_top(device, groups, groups));
}

bool pn_tt_kv_banks_alloc(pn_tt_device *device, uint32_t layer_count)
{
    const size_t k_group_bytes = (size_t)device->kv_group_layout.region_bytes;
    const size_t v_group_bytes = (size_t)device->kv_v_group_layout.region_bytes;
    const uint32_t groups = device->kv_bank_groups ? device->kv_bank_groups : 1u;

    if (device->dram_bank_count == 0u)
        return false;
    device->kv_region = calloc(layer_count, sizeof(*device->kv_region));
    if (!device->kv_region)
        return false;
    device->kv_layer_count = layer_count;
    for (uint32_t layer = 0; layer < layer_count; layer++) {
        pn_tt_kv_region *region = &device->kv_region[layer];
        const bool v_first = kv_layer_v_first(device, groups);

        /* K's groups then V's, each the next bank: at G = 4 a layer spans all 8 banks, and
         * at G = 1 this is the pre-group K bank, V bank order exactly. A mixed cache may
         * take V's first (kv_layer_v_first). */
        if (v_first && !kv_group_set_alloc(device, v_group_bytes, groups, region->v_group_bank,
                                           region->v_group_address))
            return false;
        if (!kv_group_set_alloc(device, k_group_bytes, groups, region->k_group_bank,
                                region->k_group_address))
            return false;
        if (!v_first && !kv_group_set_alloc(device, v_group_bytes, groups, region->v_group_bank,
                                            region->v_group_address))
            return false;
        region->k_bank = region->k_group_bank[0];
        region->k_address = region->k_group_address[0];
        region->v_bank = region->v_group_bank[0];
        region->v_address = region->v_group_address[0];
    }
    return true;
}

static uint32_t kv_group_of(const pn_tt_device *device, uint32_t head)
{
    const uint32_t per = device->kv_group_layout.head_count ? device->kv_group_layout.head_count
                                                            : 1u;
    const uint32_t group = head / per;

    return group < device->kv_bank_groups ? group : 0u;
}

uint32_t pn_tt_kv_head_bank(const pn_tt_device *device, uint32_t layer, bool values,
                            uint32_t head)
{
    const pn_tt_kv_region *region = &device->kv_region[layer];
    const uint32_t group = kv_group_of(device, head);

    return values ? region->v_group_bank[group] : region->k_group_bank[group];
}

ft_tt_op_operand pn_tt_kv_region_operand(const pn_tt_device *device, uint32_t layer,
                                         bool values, uint32_t head)
{
    const pn_tt_kv_region *region = &device->kv_region[layer];
    const uint32_t group = kv_group_of(device, head);
    const uint32_t first = group * device->kv_group_layout.head_count;
    const uint32_t bank = values ? region->v_group_bank[group] : region->k_group_bank[group];
    const uint32_t stride = (uint32_t)pn_tt_kv_layout_of(device, values)->head_stride;
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[bank].x;
    operand.y = (uint16_t)device->dram[bank].y;
    /* The group's base minus the full layout's offset of its first head (uint32 wrap): a
     * caller adding `head * stride + position` lands at `base + (head - first) * stride +
     * position`. At G = 1 first = 0 and this is the region's own base. */
    operand.address = (values ? region->v_group_address[group]
                              : region->k_group_address[group]) - first * stride;
    /* bf16, or block float; K's and V's may differ (pn_tt_kv_format.h) */
    operand.tile_bytes = pn_tt_kv_layout_of(device, values)->tile_bytes;
    return operand;
}

int pn_tt_kv_head_operand(const pn_tt_device *device, uint32_t layer, bool values,
                          uint32_t head, ft_tt_op_operand *out)
{
    uint64_t run_offset = 0;
    uint32_t run_bytes = 0;

    if (ft_tt_kv_cache_run(pn_tt_kv_layout_of(device, values), head, 0u, PN_TT_TILE_DIM,
                           &run_offset, &run_bytes) != FT_TT_OK)
        return -1;
    *out = pn_tt_kv_region_operand(device, layer, values, head);
    out->address += (uint32_t)run_offset;
    return 0;
}

/**
 * The BANKED form: one descriptor, each group's heads into its own bank at the common
 * address. Red `kv_bank_group_shift`: group g's heads land in group g+1's bank.
 */
static int kv_append_words_banked(const pn_tt_device *device,
                                  const ft_tt_op_kv_append_request *request, uint32_t layer,
                                  bool values, uint32_t words[FT_TT_OP_MEMORY_WORDS])
{
    const uint32_t groups = device->kv_bank_groups;
    const uint32_t per = device->kv_group_layout.head_count;
    const bool red = pn_tt_red("kv_bank_group_shift");
    ft_tt_op_kv_append_request one = *request;

    one.region = pn_tt_kv_region_operand(device, layer, values, 0u);
    one.bank_count = groups;
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t target = red ? (g + 1u) % groups : g;

        one.bank_region[g] = pn_tt_kv_region_operand(device, layer, values, target * per);
        one.bank_region[g].address +=
            target * per * (uint32_t)pn_tt_kv_layout_of(device, values)->head_stride;
    }
    /* The full layout: every head in one descriptor, slot stride one head (the encoder puts
     * heads-per-bank in FLAGS). */
    one.cache = pn_tt_kv_layout_of(device, values);
    return ft_tt_op_kv_append_parameters(&one, words) == FT_TT_OK ? 0 : -1;
}

int pn_tt_kv_append_words(const pn_tt_device *device, const ft_tt_op_kv_append_request *request,
                          uint32_t layer, bool values,
                          uint32_t words[PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS],
                          uint32_t *out_count)
{
    if (device->kv_append_banked) {
        *out_count = 1u;
        return kv_append_words_banked(device, request, layer, values, words[0]);
    }
    return pn_tt_kv_append_words_per_group(device, request, layer, values, words, out_count);
}

int pn_tt_kv_append_words_per_group(const pn_tt_device *device,
                                    const ft_tt_op_kv_append_request *request, uint32_t layer,
                                    bool values,
                                    uint32_t words[PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS],
                                    uint32_t *out_count)
{
    const uint32_t groups = device->kv_bank_groups ? device->kv_bank_groups : 1u;
    const uint32_t per = device->kv_layout.head_count / groups;
    /* RED `kv_bank_group_shift` (G > 1 only): group g appends group (g+1)'s heads — every
     * K/V row written with another head's values. Must DIFFER. */
    const bool red = groups > 1u && pn_tt_red("kv_bank_group_shift");

    for (uint32_t g = 0; g < groups; g++) {
        ft_tt_op_kv_append_request one = *request;
        const uint32_t source_group = red ? (g + 1u) % groups : g;
        const uint32_t first = g * per;

        if (groups > 1u) {
            one.cache = kv_group_layout_of(device, values);
            /* The SOURCE is the bf16 projection strip whatever the cache format. */
            one.source.address += source_group * per * device->kv_layout.dim_tiles *
                                  PN_TT_TILE_BF16;
        }
        one.region = pn_tt_kv_region_operand(device, layer, values, first);
        one.region.address += first * (uint32_t)pn_tt_kv_layout_of(device, values)->head_stride;
        if (ft_tt_op_kv_append_parameters(&one, words[g]) != FT_TT_OK)
            return -1;
    }
    *out_count = groups;
    return 0;
}

#endif /* USE_TT */
