/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_mixw_fit.c
 * @brief Implementation of the format-map fit: map checks, then per-bank pricing over the
 *        load's weight rotation (see pn_tt_mixw_fit.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_mixw_fit.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/** printf into `reason` when there is one. */
static void mixw_refuse(char *reason, size_t reason_bytes, const char *format, ...)
{
    va_list args;

    if (!reason || reason_bytes == 0u)
        return;
    va_start(args, format);
    vsnprintf(reason, reason_bytes, format, args);
    va_end(args);
}

uint32_t pn_tt_mixw_format_tile_bytes(const char *format_name)
{
    if (!format_name)
        return 0u;
    if (strcmp(format_name, "bfp8_b") == 0)
        return PN_TT_MIXW_TILE_BYTES_BFP8;
    if (strcmp(format_name, "bfp4_b") == 0)
        return PN_TT_MIXW_TILE_BYTES_BFP4;
    return 0u;
}

uint64_t pn_tt_mixw_placement_bytes(const pn_tt_mixw_placement *placement, uint32_t tile_bytes)
{
    if (!placement)
        return 0u;
    return (uint64_t)placement->tile_rows * placement->tile_cols * tile_bytes;
}

size_t pn_tt_mixw_group_size(const pn_tt_mixw_placement *placements, size_t count,
                             const char *group_name)
{
    size_t members = 0u;

    for (size_t i = 0; placements && group_name && i < count; i++)
        if (strcmp(placements[i].group_name, group_name) == 0)
            members++;
    return members;
}

/** First placement whose `group_name` (or, with `by_placement`, `placement_name`) is `name`. */
static const pn_tt_mixw_placement *mixw_find(const pn_tt_mixw_placement *placements,
                                             size_t count, const char *name, bool by_placement)
{
    for (size_t i = 0; i < count; i++) {
        const char *key = by_placement ? placements[i].placement_name : placements[i].group_name;

        if (strcmp(key, name) == 0)
            return &placements[i];
    }
    return NULL;
}

/** Is every placement of `group_name` host-side? */
static bool mixw_group_is_host(const pn_tt_mixw_placement *placements, size_t count,
                               const char *group_name)
{
    for (size_t i = 0; i < count; i++)
        if (strcmp(placements[i].group_name, group_name) == 0 &&
            placements[i].kind != PN_TT_MIXW_PLACE_HOST)
            return false;
    return true;
}

/**
 * One map line against the placements: a known group the converter may map, a format a
 * resident program serves, listed once. A name that is one placement of another group is a
 * SPLIT of that group and is said so, because it is the one refusal a reader could mistake
 * for a typo.
 */
static int mixw_check_line(const pn_tt_mixw_placement *placements, size_t count,
                           const pn_tt_mixw_map_entry *map, size_t line, char *reason,
                           size_t reason_bytes)
{
    const pn_tt_mixw_map_entry *entry = &map[line];
    const pn_tt_mixw_placement *group, *member;

    if (!entry->tensor_name || !entry->format_name) {
        mixw_refuse(reason, reason_bytes, "map line %zu is missing a field", line + 1u);
        return -1;
    }
    for (size_t earlier = 0; earlier < line; earlier++)
        if (map[earlier].tensor_name && strcmp(map[earlier].tensor_name, entry->tensor_name) == 0) {
            mixw_refuse(reason, reason_bytes, "map line %zu: %s is listed twice", line + 1u,
                        entry->tensor_name);
            return -1;
        }
    if (pn_tt_mixw_format_tile_bytes(entry->format_name) == 0u) {
        mixw_refuse(reason, reason_bytes, "map line %zu: %s asks for '%s'; a dense resident "
                    "matmul serves only bfp8_b and bfp4_b", line + 1u, entry->tensor_name,
                    entry->format_name);
        return -1;
    }
    group = mixw_find(placements, count, entry->tensor_name, false);
    if (!group) {
        member = mixw_find(placements, count, entry->tensor_name, true);
        if (member)
            mixw_refuse(reason, reason_bytes, "map line %zu: %s is one placement of %s (%zu "
                        "placements, one source); naming it alone splits that group — map %s",
                        line + 1u, entry->tensor_name, member->group_name,
                        pn_tt_mixw_group_size(placements, count, member->group_name),
                        member->group_name);
        else
            mixw_refuse(reason, reason_bytes, "map line %zu: %s is not a tensor of this census",
                        line + 1u, entry->tensor_name);
        return -1;
    }
    if (group->map_refusal || mixw_group_is_host(placements, count, entry->tensor_name)) {
        mixw_refuse(reason, reason_bytes, "map line %zu: %s may not be mapped: %s", line + 1u,
                    entry->tensor_name,
                    group->map_refusal ? group->map_refusal : "host-side, 0 card bytes");
        return -1;
    }
    return 0;
}

/** Shapes the rotation can place: a non-empty grid, parts that divide a LOAD entry. */
static int mixw_check_placements(const pn_tt_mixw_placement *placements, size_t count,
                                 const pn_tt_mixw_card *card, char *reason, size_t reason_bytes)
{
    if (card->bank_count == 0u || card->bank_count > PN_TT_MIXW_MAX_BANKS ||
        card->members == 0u || (card->default_tile_bytes != PN_TT_MIXW_TILE_BYTES_BFP8 &&
                                card->default_tile_bytes != PN_TT_MIXW_TILE_BYTES_BFP4)) {
        mixw_refuse(reason, reason_bytes, "card: %u banks, %u members, default tile %u B is "
                    "not a priceable card", card->bank_count, card->members,
                    card->default_tile_bytes);
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        const pn_tt_mixw_placement *p = &placements[i];
        const uint32_t parts = p->parts ? p->parts : 1u;

        if (!p->placement_name || !p->group_name || p->tile_rows == 0u || p->tile_cols == 0u ||
            p->tile_cols % parts != 0u ||
            (parts > 1u && p->kind != PN_TT_MIXW_PLACE_LOAD)) {
            mixw_refuse(reason, reason_bytes, "placement %zu (%s) is not a placeable shape", i,
                        p->placement_name ? p->placement_name : "(unnamed)");
            return -1;
        }
    }
    return 0;
}

/** One upload through pn_tt_matmul_group_place_weight_columns(): shards, then cursor + 1. */
static void mixw_place_one(const pn_tt_mixw_card *card, uint32_t tile_rows, uint32_t tile_cols,
                           uint32_t tile_bytes, uint32_t *cursor, uint64_t *bank_bytes)
{
    const uint64_t column_bytes = (uint64_t)tile_rows * tile_bytes;
    const uint32_t base = tile_cols / card->members, remainder = tile_cols % card->members;

    if (card->members < 2u || base == 0u) {
        bank_bytes[*cursor % card->bank_count] += column_bytes * tile_cols;
    } else {
        for (uint32_t member = 0; member < card->members; member++)
            bank_bytes[(*cursor + member) % card->bank_count] +=
                column_bytes * (base + (member < remainder ? 1u : 0u));
    }
    *cursor = (*cursor + 1u) % card->bank_count;
}

/** The tile bytes the map gives `group_name`, or the rung's when no line names it. */
static uint32_t mixw_group_tile_bytes(const pn_tt_mixw_map_entry *map, size_t map_count,
                                      const char *group_name, uint32_t default_tile_bytes)
{
    for (size_t i = 0; i < map_count; i++)
        if (strcmp(map[i].tensor_name, group_name) == 0)
            return pn_tt_mixw_format_tile_bytes(map[i].format_name);
    return default_tile_bytes;
}

int pn_tt_mixw_fit_price(const pn_tt_mixw_placement *placements, size_t count,
                         const pn_tt_mixw_map_entry *map, size_t map_count,
                         const pn_tt_mixw_card *card, pn_tt_mixw_fit *out,
                         uint32_t *out_tile_bytes, char *reason, size_t reason_bytes)
{
    uint32_t cursor;

    if (!placements || !card || !out || (!map && map_count)) {
        mixw_refuse(reason, reason_bytes, "bad arguments");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (mixw_check_placements(placements, count, card, reason, reason_bytes) != 0)
        return -1;
    for (size_t line = 0; line < map_count; line++)
        if (mixw_check_line(placements, count, map, line, reason, reason_bytes) != 0)
            return -1;

    cursor = card->first_bank % card->bank_count;
    for (size_t i = 0; i < count; i++) {
        const pn_tt_mixw_placement *p = &placements[i];
        const uint32_t tile_bytes = mixw_group_tile_bytes(map, map_count, p->group_name,
                                                          card->default_tile_bytes);
        const uint64_t bytes = pn_tt_mixw_placement_bytes(p, tile_bytes);

        if (out_tile_bytes)
            out_tile_bytes[i] = tile_bytes;
        if (p->kind == PN_TT_MIXW_PLACE_HOST) {
            out->host_bytes += bytes;
            continue;
        }
        if (p->kind == PN_TT_MIXW_PLACE_LOAD) {
            const uint32_t parts = p->parts ? p->parts : 1u, start = cursor;

            for (uint32_t part = 0; part < parts; part++)
                mixw_place_one(card, p->tile_rows, p->tile_cols / parts, tile_bytes, &cursor,
                               out->bank_bytes);
            cursor = (start + 1u) % card->bank_count;
            out->load_bytes += bytes;
        } else {
            mixw_place_one(card, p->tile_rows, p->tile_cols, tile_bytes, &cursor,
                           out->bank_bytes);
            out->staged_bytes += bytes;
        }
        if (tile_bytes == PN_TT_MIXW_TILE_BYTES_BFP4)
            out->bfp4_placements++;
    }
    for (uint32_t bank = 0; bank < card->bank_count; bank++) {
        out->card_bytes += out->bank_bytes[bank];
        if (out->bank_bytes[bank] > out->bank_bytes[out->fullest_bank])
            out->fullest_bank = bank;
    }
    out->mapped_lines = (uint32_t)map_count;
    return 0;
}
