/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_split.c
 * @brief Implementation of the load-time whole-or-parts decision (pn_tt_weight_split.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_weight_split.h"

#include <string.h>

int pn_tt_weight_split_rule_init(pn_tt_weight_split_rule *rule, const char *suffix,
                                 uint32_t fields, uint32_t field_tiles, uint32_t part_tiles)
{
    size_t length;
    uint32_t parts;

    if (!rule || !suffix)
        return -1;
    length = strlen(suffix);
    if (length == 0u || length >= PN_TT_WEIGHT_SPLIT_SUFFIX_BYTES || fields == 0u ||
        part_tiles == 0u || field_tiles % part_tiles != 0u)
        return -1;
    parts = field_tiles / part_tiles;
    if (parts < 2u || parts > PN_TT_WEIGHT_SPLIT_MAX_PARTS || fields > PN_TT_COLUMN_RUNS_MAX)
        return -1;
    memset(rule, 0, sizeof(*rule));
    memcpy(rule->suffix, suffix, length + 1u);
    rule->fields = fields;
    rule->field_tiles = field_tiles;
    rule->part_tiles = part_tiles;
    return 0;
}

uint32_t pn_tt_weight_split_part_count(const pn_tt_weight_split_rule *rule)
{
    return rule && rule->part_tiles ? rule->field_tiles / rule->part_tiles : 0u;
}

bool pn_tt_weight_split_applies(const pn_tt_weight_split_rule *rule, const char *name,
                                uint32_t tile_cols)
{
    size_t name_length, suffix_length;

    if (!rule || !name || rule->suffix[0] == '\0')
        return false;
    name_length = strlen(name);
    suffix_length = strlen(rule->suffix);
    return name_length >= suffix_length &&
           strcmp(name + name_length - suffix_length, rule->suffix) == 0 &&
           tile_cols == rule->fields * rule->field_tiles;
}

uint32_t pn_tt_weight_upload_parts(const pn_tt_weight_split_rule *rule, const char *name,
                                   uint32_t tile_cols, uint64_t byte_count,
                                   pn_tt_weight_upload_part *out_parts)
{
    uint64_t column_bytes;
    uint32_t parts, part;

    if (!name || !out_parts || tile_cols == 0u || byte_count % tile_cols != 0u)
        return 0u;
    column_bytes = byte_count / tile_cols;
    if (!pn_tt_weight_split_applies(rule, name, tile_cols)) {
        memset(&out_parts[0], 0, sizeof(out_parts[0]));
        out_parts[0].part = -1;
        out_parts[0].tile_cols = tile_cols;
        out_parts[0].bytes = byte_count;
        return 1u;
    }
    parts = pn_tt_weight_split_part_count(rule);
    for (part = 0u; part < parts; part++) {
        pn_tt_weight_upload_part *out = &out_parts[part];

        memset(out, 0, sizeof(*out));
        out->part = (int32_t)part;
        out->run_count = pn_tt_column_runs_field(rule->fields, rule->field_tiles,
                                                 part * rule->part_tiles, rule->part_tiles,
                                                 out->runs, PN_TT_COLUMN_RUNS_MAX);
        if (out->run_count != rule->fields)
            return 0u;
        out->tile_cols = pn_tt_column_runs_total(out->runs, out->run_count);
        out->bytes = (uint64_t)out->tile_cols * column_bytes;
    }
    return parts;
}
