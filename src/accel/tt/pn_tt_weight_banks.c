/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_banks.c
 * @brief Weight bank map + probe selection (pn_tt_weight_banks.h carries the why).
 *
 * @status NEW 2026-10-22
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_weight_banks.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_name_list.h"
#include "tt/ft_tt_coord.h"
#include "tt/ft_tt_dram_nearest.h"

/** The NoC the matmul reader (BRISC) issues its weight reads on. */
#define PN_TT_WEIGHT_READER_NOC 0u

void pn_tt_weight_class(const char *name, char *out, uint32_t out_size)
{
    const char *start = name ? name : "";
    const char *end;
    size_t length;

    if (!out || out_size == 0u)
        return;
    if (strncmp(start, "blk.", 4) == 0) {
        const char *dot = strchr(start + 4, '.');

        if (dot)
            start = dot + 1;
    }
    end = strstr(start, ".weight");
    length = end ? (size_t)(end - start) : strlen(start);
    if (length >= out_size)
        length = out_size - 1u;
    memcpy(out, start, length);
    out[length] = '\0';
}

uint8_t pn_tt_weight_probe_selected(const char *name)
{
    const char *list = getenv(PN_TT_MATMUL_PROBE_ONLY_ENV);
    char klass[64];

    if (!list || !*list)
        return 1u;
    pn_tt_weight_class(name, klass, sizeof(klass));
    return pn_tt_name_list_contains(list, klass) ? 1u : 0u;
}

/** Parse the mode once: 0 rotate, 1 near, 2 nearbank, 3 far, 4 fixed (controls); -1 a name
 *  nobody knows. */
static int pn_tt_weight_banks_mode(void)
{
    const char *value = getenv(PN_TT_WEIGHT_BANKS_ENV);

    if (!value || !*value || strcmp(value, "rotate") == 0)
        return 0;
    if (strcmp(value, "near") == 0)
        return 1;
    if (strcmp(value, "nearbank") == 0)
        return 2;
    if (strcmp(value, "far") == 0)
        return 3;
    if (strcmp(value, "fixed") == 0)
        return 4;
    if (strncmp(value, "map:", 4) == 0)
        return 5;
    fprintf(stderr, "pn_tt: REFUSED — %s='%s' is not rotate|near|nearbank|far|fixed|map:<b.p,...>\n",
            PN_TT_WEIGHT_BANKS_ENV, value);
    return -1;
}

/** `map:3.1,6.2,...` — an explicit bank.door per member (a searched map under test). */
static int pn_tt_weight_banks_parse_map(pn_tt_device *device, const char *text)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t member = 0u, load[8] = {0};
    const char *cursor = text;

    while (*cursor && member < group->member_count) {
        char *end = NULL;
        unsigned long bank = strtoul(cursor, &end, 10), port;

        if (end == cursor || *end != '.')
            return -1;
        cursor = end + 1;
        port = strtoul(cursor, &end, 10);
        if (end == cursor || bank >= device->dram_bank_count || port > 2u)
            return -1;
        group->weight_near_bank[member] = (uint32_t)bank;
        group->weight_near_port[member] = (uint32_t)port;
        load[bank % 8u]++;
        member++;
        cursor = *end == ',' ? end + 1 : end;
    }
    if (member != group->member_count || *cursor != '\0')
        return -1;
    for (member = 0; member < 8u; member++)
        if (load[member] > (group->member_count + device->dram_bank_count - 1u) /
                               device->dram_bank_count)
            return -1; /* an unbalanced map is refused: one bank's ceiling would cap them all */
    return 0;
}

/** The nearest-bank map for the group's members, computed once and printed. */
static int pn_tt_weight_banks_build_near(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    ft_tt_noc_coord cores[PN_TT_MATMUL_MAX_CORES];
    uint32_t members = group->member_count, member;
    const uint32_t banks = device->dram_bank_count;
    const uint32_t per_bank = (members + banks - 1u) / banks;

    if (members > FT_TT_DRAM_NEAREST_MAX_READERS)
        return -1;
    if (group->weight_bank_mode == 5u) {
        if (pn_tt_weight_banks_parse_map(device, getenv(PN_TT_WEIGHT_BANKS_ENV) + 4) != 0)
            return -1;
        goto print_map;
    }
    for (member = 0; member < members; member++)
        if (ft_tt_coord_tensix_by_index(&device->runtime.grid, group->member[member].core_index,
                                        FT_TT_COORD_NOC0, &cores[member]) != FT_TT_OK)
            return -1;
    if (ft_tt_dram_nearest_assign(&device->runtime.grid, cores, members,
                                  PN_TT_WEIGHT_READER_NOC, per_bank ? per_bank : 1u,
                                  group->weight_bank_mode == 3u ? FT_TT_DRAM_NEAREST_GOAL_FARTHEST
                                                                : FT_TT_DRAM_NEAREST_GOAL_NEAREST,
                                  group->weight_near_bank, group->weight_near_port) != FT_TT_OK)
        return -1;
    /* FIXED: the rotation's cursor-0 banks for every tensor — the shipped bank set without
     * the per-tensor rotation, so ORDER (same banks every projection) and DISTANCE separate. */
    if (group->weight_bank_mode == 4u) {
        ft_tt_dram_endpoint doors[PN_TT_MATMUL_MAX_CORES];

        for (member = 0; member < members; member++)
            group->weight_near_bank[member] = member % banks;
        if (ft_tt_dram_stripe_choose_ports(&device->runtime.grid, group->weight_near_bank,
                                           members, FT_TT_COORD_TRANSLATED, doors) != FT_TT_OK)
            return -1;
        for (member = 0; member < members; member++)
            group->weight_near_port[member] = doors[member].port;
    }
print_map:;
    static const char *const names[] = {"rotate", "near", "nearbank", "far", "fixed", "map"};

    fprintf(stderr, "pn_tt: weight banks %s — member:bank.door",
            names[group->weight_bank_mode]);
    for (member = 0; member < members; member++)
        fprintf(stderr, " %u:%u.%u", member, group->weight_near_bank[member],
                group->weight_near_port[member]);
    fprintf(stderr, "\n");
    group->weight_near_ready = true;
    return 0;
}

int pn_tt_weight_banks_choose(pn_tt_device *device, uint32_t members, uint32_t *banks,
                              ft_tt_dram_endpoint *doors)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    if (!group->weight_near_ready) {
        int mode = pn_tt_weight_banks_mode();

        if (mode < 0)
            return -1;
        group->weight_bank_mode = (uint32_t)mode;
        if (mode == 0)
            return 0;
        if (pn_tt_weight_banks_build_near(device) != 0) {
            fprintf(stderr, "pn_tt: REFUSED — could not build the nearest-bank map\n");
            return -1;
        }
    }
    if (group->weight_bank_mode == 0u)
        return 0;
    if (members != group->member_count)
        return 0; /* a panel count the map was not built for keeps the rotation */
    for (member = 0; member < members; member++)
        banks[member] = group->weight_near_bank[member];
    if (group->weight_bank_mode == 2u)
        return ft_tt_dram_stripe_choose_ports(&device->runtime.grid, banks, members,
                                              FT_TT_COORD_TRANSLATED, doors) == FT_TT_OK
                   ? 1
                   : -1;
    for (member = 0; member < members; member++) {
        memset(&doors[member], 0, sizeof(doors[member]));
        doors[member].logical_bank = banks[member];
        doors[member].port = group->weight_near_port[member];
        if (ft_tt_coord_dram_port(&device->runtime.grid, banks[member],
                                  group->weight_near_port[member], FT_TT_COORD_TRANSLATED,
                                  &doors[member].coord) != FT_TT_OK)
            return -1;
    }
    return 1;
}

#endif /* USE_TT */
