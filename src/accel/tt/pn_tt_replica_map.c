/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_replica_map.c
 * @brief Implementation of the activation-strip mirror map.
 *
 * @status Written 2026-08-19.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_replica_map.h"

#include <stdio.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_eltwise_group.h"

#include "tt/ft_tt_op_replicas.h"

uint32_t pn_tt_activation_replica_count(const pn_tt_device *device)
{
    if (!device)
        return 1u;
    return device->activation_replicas ? device->activation_replicas : 1u;
}

/**
 * Add ONE SLICE of one strip's mirrors to the map, or say nothing if it has none.
 *
 * `byte_offset` is what makes a split addressable. The kernel's lookup is the exact triple
 * `(x, y, base)`, so a descriptor writing the strip from tile `t` onward needs its OWN set,
 * whose primary is the strip's base plus that offset and whose mirrors are each mirror's
 * base plus the SAME offset — the mirrors are whole copies of the strip, so one offset
 * addresses all of them. At offset 0 this is the unsliced set exactly.
 */
static int replica_map_add_slice(ft_tt_op_replica_table *table, const pn_tt_device *device,
                                 const pn_tt_activation *strip, uint32_t byte_offset)
{
    ft_tt_op_replica_set *entry;
    uint32_t index;

    if (!strip || strip->replica_count < 2u)
        return 0;
    if (table->set_count >= FT_TT_REPLICA_MAX_SETS) {
        fprintf(stderr, "pn_tt: more mirrored strips than the replica table holds (%u)\n",
                FT_TT_REPLICA_MAX_SETS);
        return -1;
    }
    entry = &table->set[table->set_count];
    entry->count = strip->replica_count;
    for (index = 0u; index < strip->replica_count; index++) {
        const uint32_t bank = strip->replica_bank[index];

        if (bank >= device->dram_bank_count) {
            fprintf(stderr, "pn_tt: mirror %u names bank %u of %u\n", index, bank,
                    device->dram_bank_count);
            return -1;
        }
        /* THE BANK'S OWN COORDINATE, not a door. A door is a read-side lever: the writer's
         * cost is the barrier, not the port, and picking a port here would make the map
         * depend on the matmul's member count. */
        entry->destination[index].x = (uint16_t)device->dram[bank].x;
        entry->destination[index].y = (uint16_t)device->dram[bank].y;
        entry->destination[index].base = strip->replica_address[index] + byte_offset;
    }
    table->set_count++;
    return 0;
}

/** Publish to one core and prove it landed. */
static int replica_map_publish_to(pn_tt_device *device, const pn_tt_residency *residency,
                                  const ft_tt_op_replica_table *table, const char *label)
{
    if (!residency->running)
        return 0;
    if (ft_tt_op_replica_table_write(&device->runtime.noc, residency->core_x,
                                     residency->core_y, table) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: could not write the replica map to %s (core %u)\n", label,
                residency->core_index);
        return -1;
    }
    /* READ IT BACK. A map that half-landed leaves one mirror stale, and a stale mirror is a
     * plausible wrong answer with every kernel completing — there is nothing downstream
     * that could refuse it.
     * S2e (`PN_TT_REPLICA_VERIFY_ONCE`, default 1): a table this core has already read back
     * byte for byte is not read back again — the verify alternates the decode map and its own
     * twice a cycle, and each 808 B read-back costs ~0.56 ms on this link. */
    {
        static struct { uint16_t x, y; uint64_t hash[4]; uint32_t n; } seen[64];
        static uint32_t seen_count;
        static int once = -1;
        const unsigned char *bytes = (const unsigned char *)table;
        uint64_t hash = 1469598103934665603ull;
        uint32_t i, h;
        size_t b;

        if (once < 0)
            once = (int)pn_tt_env_u32("PN_TT_REPLICA_VERIFY_ONCE", 1u);
        for (b = 0; b < sizeof(*table); b++)
            hash = (hash ^ bytes[b]) * 1099511628211ull;
        for (i = 0; i < seen_count; i++)
            if (seen[i].x == residency->core_x && seen[i].y == residency->core_y)
                break;
        if (once && i < seen_count)
            for (h = 0; h < seen[i].n; h++)
                if (seen[i].hash[h] == hash)
                    return 0;
        if (ft_tt_op_replica_table_verify(&device->runtime.noc, residency->core_x,
                                          residency->core_y, table) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the replica map did not read back from %s (core %u)\n",
                    label, residency->core_index);
            return -1;
        }
        if (i == seen_count && seen_count < 64u) {
            seen[i].x = (uint16_t)residency->core_x;
            seen[i].y = (uint16_t)residency->core_y;
            seen[i].n = 0u;
            seen_count++;
        }
        if (i < seen_count)
            seen[i].hash[seen[i].n++ % 4u] = hash, seen[i].n = seen[i].n > 4u ? 4u : seen[i].n;
    }
    return 0;
}

static uint32_t g_replica_span;
static uint32_t g_replica_superband;

void pn_tt_replica_map_set_span(uint32_t span)
{
    g_replica_span = span;
}

void pn_tt_replica_map_set_superband(uint32_t bands)
{
    g_replica_superband = bands;
}

/** Add the set keyed at `byte_offset` unless this table already holds that key. */
static int replica_map_add_key(ft_tt_op_replica_table *table, const pn_tt_device *device,
                               const pn_tt_activation *strip, uint32_t byte_offset)
{
    uint32_t set;

    for (set = 0u; set < table->set_count; set++)
        if (table->set[set].destination[0].base == strip->replica_address[0] + byte_offset &&
            table->set[set].destination[0].x == device->dram[strip->replica_bank[0]].x &&
            table->set[set].destination[0].y == device->dram[strip->replica_bank[0]].y)
            return 0;
    return replica_map_add_slice(table, device, strip, byte_offset);
}

/**
 * The keys of the SPLIT strip that ONE writer needs: eltwise member `member` (or, for
 * `member < 0`, a core that never writes it sliced — it gets the base keys member 0 has).
 *
 * PER CORE SINCE PHASE W1a. The kernel's lookup is the exact `(x, y, base)` of the descriptor
 * a core runs, and a core only ever runs ITS slice, so a key for another member's slice is a
 * set no descriptor on that core can hit. One shared table held every member's keys and
 * capped the split at the table's 8 sets; per core it holds that member's keys only — at most
 * one a band (decode) or one a span (tall) — whatever the member count.
 */
static int replica_map_split_keys(pn_tt_device *device, ft_tt_op_replica_table *table,
                                  const pn_tt_activation *strip, int member)
{
    const uint32_t m = member < 0 ? 0u : (uint32_t)member;
    const uint32_t bands = strip->tile_rows ? strip->tile_rows : 1u;
    const bool skip = pn_tt_red("eltwise_wide_key_skip");
    pn_tt_eltwise_slice slice;
    uint32_t band, which, members;

    /* PHASE G: a TALL strip (more than 4 bands) is keyed by the producer's span — decode
     * (span 1) and the prefill span `g_replica_span` — not a set per band. */
    if (bands > 4u) {
        const uint32_t spans[2] = { 1u, g_replica_span ? g_replica_span : 1u };

        for (which = 0u; which < 2u; which++) {
            const uint32_t flat = spans[which] * strip->tile_count;

            if (which == 1u && spans[1] == 1u)
                break;
            members = pn_tt_eltwise_group_members_for(device, flat, FT_TT_OP_VECTOR_FN_SILU_MUL);
            if (members > 1u && (flat % members) != 0u &&
                members <= device->eltwise_decode_members)
                members = 1u; /* that width runs whole on member 0 */
            if (m >= members ||
                !pn_tt_eltwise_group_slice(device, flat, members, m, PN_TT_TILE_BF16, &slice))
                continue;
            /* RED `eltwise_wide_key_skip`: a WIDE member is not told its prefill key — its
             * slice's mirrors go stale and the grid's ffn_down reads them; must DIFFER. */
            if (skip && m >= device->eltwise_decode_members && which == 1u)
                continue;
            if (replica_map_add_key(table, device, strip, slice.byte_offset) != 0)
                return -1;
        }
        /* PHASE W2: a SUPER-BAND producer writes `sb` bands at band j*sb (the last one
         * shorter), so its member's slice of THAT width is a key of its own. */
        if (g_replica_superband && g_replica_span > g_replica_superband) {
            uint32_t first;

            for (first = 0u; first < g_replica_span; first += g_replica_superband) {
                const uint32_t n = g_replica_span - first < g_replica_superband
                                       ? g_replica_span - first : g_replica_superband;
                const uint32_t flat = n * strip->tile_count;

                members = pn_tt_eltwise_group_members_for(device, flat, FT_TT_OP_VECTOR_FN_SILU_MUL);
                if (members > 1u && (flat % members) != 0u &&
                    members <= device->eltwise_decode_members)
                    members = 1u;
                if (m >= members ||
                    !pn_tt_eltwise_group_slice(device, flat, members, m, PN_TT_TILE_BF16,
                                               &slice))
                    continue;
                if (replica_map_add_key(table, device, strip,
                                        first * strip->tile_count * PN_TT_TILE_BF16 +
                                            slice.byte_offset) != 0)
                    return -1;
            }
        }
        return 0;
    }
    members = pn_tt_eltwise_group_members_for(device, strip->tile_count,
                                              FT_TT_OP_VECTOR_FN_SILU_MUL);
    if (members > 1u && (strip->tile_count % members) != 0u)
        members = 1u;
    if (m >= members ||
        !pn_tt_eltwise_group_slice(device, strip->tile_count, members, m, PN_TT_TILE_BF16,
                                   &slice))
        return 0;
    for (band = 0u; band < bands; band++)
        if (replica_map_add_key(table, device, strip,
                                band * strip->tile_count * PN_TT_TILE_BF16 +
                                    slice.byte_offset) != 0) {
            fprintf(stderr, "pn_tt: the replica table holds %u sets and %u band(s) need more "
                            "— lower %s, or the mirrors of the upper bands go STALE\n",
                    FT_TT_REPLICA_MAX_SETS, bands, PN_TT_STRIP_TILE_ROWS_ENV);
            return -1;
        }
    return 0;
}

/** The whole table ONE writer gets: every unsliced mirrored strip, and the split strip's keys
 *  for `member` (see replica_map_split_keys). */
static int replica_map_build(pn_tt_device *device, const pn_tt_activation *const *strips,
                             uint32_t strip_count, const pn_tt_activation *split_output,
                             int member, ft_tt_op_replica_table *table)
{
    uint32_t index, band;

    memset(table, 0, sizeof(*table));
    for (index = 0u; index < strip_count; index++) {
        const pn_tt_activation *strip = strips[index];
        const uint32_t bands = strip && strip->tile_rows ? strip->tile_rows : 1u;

        if (!strip || strip->replica_count < 2u)
            continue;
        /* BY PRODUCER (W2): the split strip (SwiGLU) is written by the eltwise members only,
         * every other mirrored strip (`normed`) by the reduce/head-norm cores only — so each
         * table holds only keys its own descriptors can hit, and the super-band keys fit. */
        if (strip == split_output) {
            if (member >= 0 && replica_map_split_keys(device, table, strip, member) != 0)
                return -1;
            continue;
        }
        if (member >= 0 && split_output)
            continue;
        /* A BAND IS A SECOND SET OF KEYS: a producer writing band `m` writes `base + m *
         * tile_count * 2048`, an address a one-set table never matches — measured, the
         * plausible wrong answer at two bands (2026-08-22). A TALL strip's producers write
         * the whole span in one descriptor, so its key is the base. */
        for (band = 0u; band < (bands > 4u ? 1u : bands); band++)
            if (replica_map_add_key(table, device, strip,
                                    band * strip->tile_count * PN_TT_TILE_BF16) != 0)
                return -1;
        if (bands > 4u && g_replica_superband && g_replica_span > g_replica_superband)
            for (band = g_replica_superband; band < g_replica_span; band += g_replica_superband)
                if (replica_map_add_key(table, device, strip,
                                        band * strip->tile_count * PN_TT_TILE_BF16) != 0)
                    return -1;
    }
    return 0;
}

int pn_tt_publish_replica_map(pn_tt_device *device, const pn_tt_activation *const *strips,
                              uint32_t strip_count, const pn_tt_activation *split_output)
{
    ft_tt_op_replica_table table;
    uint32_t index;

    if (!device || (!strips && strip_count))
        return -1;
    /* RED PROOF `PN_TT_RED=replica_stale`: allocate the mirrors and tell NOBODY about them.
     * The producers then write only replica 0 and mirrors 1..R-1 stay at the zeros the
     * allocation staged, so every matmul member except those on replica 0 multiplies a zero
     * row. THE ANSWER IS GARBAGE, deliberately; it MUST differ. */
    if (pn_tt_red("replica_stale")) {
        fprintf(stderr, "pn_tt: RED PROOF replica_stale — the mirrors are allocated and the "
                        "producers are never told, so every mirror but the first stays "
                        "ZERO. THE ANSWER IS GARBAGE unless nothing reads them.\n");
        strip_count = 0u;
    }
    /* EVEN AN EMPTY MAP IS PUBLISHED: the magic word with a zero set count is a different
     * state from "this L1 region was never written". Every core running
     * `resident_vector_writer.c` gets its OWN table (replica_map_split_keys). */
    if (replica_map_build(device, strips, strip_count, split_output, -1, &table) != 0 ||
        replica_map_publish_to(device, &device->reduce, &table, "the reduce residency") != 0 ||
        /* L4: the column-split member writes the same strips — without the table its half
         * of every mirror goes stale (measured 2026-10-05: DIFFERS, deterministic). */
        replica_map_publish_to(device, &device->reduce_member, &table,
                               "the reduce member") != 0)
        return -1;
    for (index = 0u; index < device->head_norm_count; index++)
        if (replica_map_publish_to(device, &device->head_norm[index].residency, &table,
                                   "a head-norm residency") != 0)
            return -1;
    for (index = 0u; index < pn_tt_eltwise_group_members(device); index++) {
        const pn_tt_residency *writer =
            index == 0u ? &device->eltwise : &device->eltwise_member[index - 1u];

        if (replica_map_build(device, strips, strip_count, split_output, (int)index,
                              &table) != 0 ||
            replica_map_publish_to(device, writer, &table,
                                   index == 0u ? "the eltwise residency"
                                               : "an eltwise split member") != 0)
            return -1;
        if (index + 1u == pn_tt_eltwise_group_members(device) && table.set_count)
            fprintf(stderr, "pn_tt: replica map published — eltwise member %u holds %u "
                            "set(s), base (%u,%u)+0x%x\n",
                    index, table.set_count, table.set[0].destination[0].x,
                    table.set[0].destination[0].y, table.set[0].destination[0].base);
    }
    return 0;
}

int pn_tt_publish_empty_replica_map(pn_tt_device *device, const pn_tt_residency *residency,
                                    const char *label)
{
    ft_tt_op_replica_table table;

    if (!device || !residency)
        return -1;
    memset(&table, 0, sizeof(table));
    return replica_map_publish_to(device, residency, &table, label);
}

#endif /* USE_TT */
