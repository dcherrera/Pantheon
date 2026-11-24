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
#include "pn_tt_superband.h"

#include <stdio.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_eltwise_group.h"
#include "pn_tt_norm_role.h"
#include "pn_tt_swiglu_fuse.h"

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

/**
 * OUT-PROJECTIONS ON THE GRID (2026-10-27): ONE RANGE RECORD for a whole TALL strip — any
 * descriptor writing anywhere in it (any band, slice, span or super-band) fans out at the
 * same offset into every mirror (foundry range records, kernels/resident_output_replicas.h).
 * In ranged mode every tall mirrored strip is keyed this way, which is what lets the
 * band-by-band producers of `ssm_gated` / attention `gated` be mirrored at all: an exact key
 * a band is 64 sets. RED `range_record_skip_gated`: the gated strips' records are left out,
 * so their mirrors go stale and the grid's out-projection reads them — must DIFFER.
 */
static bool g_replica_ranged;
static const pn_tt_activation *g_range_only[4];
static uint32_t g_range_only_count;

void pn_tt_replica_map_set_ranged(bool ranged, const pn_tt_activation *const *gated,
                                  uint32_t gated_count)
{
    uint32_t index;

    g_replica_ranged = ranged;
    g_range_only_count = 0u;
    for (index = 0u; ranged && index < gated_count && index < 4u; index++)
        g_range_only[g_range_only_count++] = gated[index];
}

static const pn_tt_activation *g_decode_range[4];
static uint32_t g_decode_range_count;

void pn_tt_replica_map_set_decode_ranged(const pn_tt_activation *const *strips, uint32_t count)
{
    uint32_t index;

    g_decode_range_count = 0u;
    for (index = 0u; strips && index < count && index < 4u; index++)
        g_decode_range[g_decode_range_count++] = strips[index];
}

static int replica_map_add_range(ft_tt_op_replica_table *table, const pn_tt_device *device,
                                 const pn_tt_activation *strip)
{
    const uint32_t bytes = strip->tile_rows * strip->tile_count * PN_TT_TILE_BF16;

    if (replica_map_add_slice(table, device, strip, 0u) != 0)
        return -1;
    if (strip->replica_count >= 2u)
        table->set[table->set_count - 1u].range_bytes = bytes;
    return 0;
}

/** The core a table goes to: a residency's, or a matmul member's (S3b). */
typedef struct replica_map_core {
    bool running;
    uint16_t core_x, core_y;
    uint32_t core_index;
} replica_map_core;

static int replica_map_publish_core(pn_tt_device *device, const replica_map_core *core,
                                    const ft_tt_op_replica_table *table, const char *label);

/** Publish to one residency's core and prove it landed. */
static int replica_map_publish_to(pn_tt_device *device, const pn_tt_residency *residency,
                                  const ft_tt_op_replica_table *table, const char *label)
{
    const replica_map_core core = { residency->running, residency->core_x, residency->core_y,
                                    residency->core_index };

    return replica_map_publish_core(device, &core, table, label);
}

/** Publish to one core and prove it landed. */
static int replica_map_publish_core(pn_tt_device *device, const replica_map_core *residency,
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
         * shorter, or folded — pn_tt_superband.h), so its member's slice of THAT width is a
         * key of its own. */
        if (g_replica_superband && g_replica_span > g_replica_superband) {
            const uint32_t jobs = pn_tt_superband_jobs(g_replica_span, g_replica_superband);

            for (uint32_t job = 0u; job < jobs; job++) {
                uint32_t first, n;

                pn_tt_superband_job(g_replica_span, g_replica_superband, job, &first, &n);
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
        if (g_replica_ranged && bands > 4u) {
            if (replica_map_add_range(table, device, strip) != 0)
                return -1;
            continue;
        }
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
            for (uint32_t job = 1u;
                 job < pn_tt_superband_jobs(g_replica_span, g_replica_superband); job++) {
                uint32_t first, n;

                pn_tt_superband_job(g_replica_span, g_replica_superband, job, &first, &n);
                if (replica_map_add_key(table, device, strip,
                                        first * strip->tile_count * PN_TT_TILE_BF16) != 0)
                    return -1;
            }
    }
    if (!pn_tt_red("range_record_skip_gated") && !pn_tt_red("replica_stale"))
        for (index = 0u; index < g_range_only_count; index++)
            if (g_range_only[index]->tile_rows > 4u &&
                replica_map_add_range(table, device, g_range_only[index]) != 0)
                return -1;
    /* L2: the DECODE out-projection inputs (at most 4 bands; the slot holds the tall strip
     * while a batched pass has swapped it in, and the tall one is keyed above). RED
     * `decode_outproj_mirror_stale`: leave them out, so their mirrors stay zero — must DIFFER. */
    if (!pn_tt_red("decode_outproj_mirror_stale") && !pn_tt_red("replica_stale"))
        for (index = 0u; index < g_decode_range_count; index++)
            if (g_decode_range[index]->tile_rows <= 4u &&
                replica_map_add_range(table, device, g_decode_range[index]) != 0)
                return -1;
    return 0;
}

/* ===== S3b: the matmul members' table, for fused SwiGLU descriptors ===================== */

static const pn_tt_activation *g_fused_strips[PN_TT_REPLICA_MAP_FUSED_MAX];
static uint32_t g_fused_count;
static bool g_fused_published;
/** What the LAST successful publish put on the members — the strips as they were then — so a
 *  strip mirrored or allocated after it is not taken as covered by a table that lacks it. */
static ft_tt_op_replica_table g_fused_table;

void pn_tt_replica_map_set_fused_swiglu(const pn_tt_activation *const *strips, uint32_t count)
{
    uint32_t index;

    g_fused_count = 0u;
    g_fused_published = false;
    for (index = 0u; strips && index < count && index < PN_TT_REPLICA_MAP_FUSED_MAX; index++)
        if (strips[index])
            g_fused_strips[g_fused_count++] = strips[index];
}

/** Bytes from a strip's band 0 to its seeked band (pn_tt_activation_seek moves every copy). */
static uint32_t replica_map_seek_bytes(const pn_tt_activation *strip)
{
    return strip->row_offset * strip->tile_count * PN_TT_TILE_BF16;
}

/**
 * ONE RANGE RECORD per registered mirrored strip, keyed at its band-0 primary and covering
 * every band: member m's fused descriptor writes `strip + m's first column` of whichever band
 * the strip is seeked to, and every mirror gets the same offset. RED `fuse_swiglu_mirror_skip`:
 * the last mirror is left out of each record, so it is never written — must DIFFER.
 */
static int replica_map_fused_table(const pn_tt_device *device, ft_tt_op_replica_table *table)
{
    const bool skip = pn_tt_red("fuse_swiglu_mirror_skip");
    uint32_t index, copy;

    memset(table, 0, sizeof(*table));
    for (index = 0u; index < g_fused_count; index++) {
        const pn_tt_activation *strip = g_fused_strips[index];
        const uint32_t count = strip->replica_count - (skip ? 1u : 0u);
        ft_tt_op_replica_set *entry;

        if (strip->replica_count < 2u || count < 2u)
            continue;
        if (table->set_count >= FT_TT_REPLICA_MAX_SETS)
            return -1;
        entry = &table->set[table->set_count++];
        entry->count = count;
        entry->range_bytes = strip->tile_rows * strip->tile_count * PN_TT_TILE_BF16;
        for (copy = 0u; copy < count; copy++) {
            const uint32_t bank = strip->replica_bank[copy];

            if (bank >= device->dram_bank_count)
                return -1;
            entry->destination[copy].x = (uint16_t)device->dram[bank].x;
            entry->destination[copy].y = (uint16_t)device->dram[bank].y;
            entry->destination[copy].base =
                strip->replica_address[copy] - replica_map_seek_bytes(strip);
        }
    }
    return 0;
}

int pn_tt_replica_map_publish_fused(pn_tt_device *device)
{
    ft_tt_op_replica_table table;
    uint32_t member;

    if (!device)
        return -1;
    g_fused_published = false;
    if (!pn_tt_fuse_swiglu_mode() || g_fused_count == 0u)
        return 0;
    if (replica_map_fused_table(device, &table) != 0) {
        fprintf(stderr, "pn_tt: the fused SwiGLU strips do not fit the members' replica table\n");
        return -1;
    }
    /* EVERY MEMBER OR NONE: a member that is not running now is armed later, and the arm
     * re-plants (pn_tt_matmul_group_arm); until then the fused path stays refused. */
    for (member = 0u; member < device->group.member_count; member++)
        if (!device->group.member[member].running)
            return 0;
    for (member = 0u; member < device->group.member_count; member++) {
        const pn_tt_matmul_member *slot = &device->group.member[member];
        const replica_map_core core = { slot->running, slot->core_x, slot->core_y,
                                        slot->core_index };

        if (replica_map_publish_core(device, &core, &table, "a matmul member") != 0)
            return -1;
    }
    g_fused_table = table;
    g_fused_published = device->group.member_count > 0u;
    return 0;
}

bool pn_tt_replica_map_fused_covers(const pn_tt_device *device, const pn_tt_activation *output)
{
    const uint32_t base0 = output ? output->replica_address[0] - replica_map_seek_bytes(output)
                                  : 0u;
    const uint32_t bytes = output ? output->tile_rows * output->tile_count * PN_TT_TILE_BF16 : 0u;
    const bool skip = pn_tt_red("fuse_swiglu_mirror_skip");
    uint32_t set, copy;

    if (!device || !output)
        return false;
    if (output->replica_count < 2u)
        return true;
    if (!g_fused_published || output->replica_bank[0] >= device->dram_bank_count)
        return false;
    /* COVERED = the published table holds a record keyed at this strip's band-0 primary, over
     * all of it, naming every one of its copies (all but the last under the RED). */
    for (set = 0u; set < g_fused_table.set_count; set++) {
        const ft_tt_op_replica_set *entry = &g_fused_table.set[set];
        bool same = entry->count == output->replica_count - (skip ? 1u : 0u) &&
                    entry->range_bytes >= bytes &&
                    entry->destination[0].x == device->dram[output->replica_bank[0]].x &&
                    entry->destination[0].y == device->dram[output->replica_bank[0]].y &&
                    entry->destination[0].base == base0;

        for (copy = 1u; same && copy < entry->count; copy++)
            same = output->replica_bank[copy] < device->dram_bank_count &&
                   entry->destination[copy].x == device->dram[output->replica_bank[copy]].x &&
                   entry->destination[copy].base ==
                       output->replica_address[copy] - replica_map_seek_bytes(output);
        if (same)
            return true;
    }
    return false;
}

/* The last published strip set, for the persistent norm role's members, which arm after the
 * map was first published (pn_tt_replica_map_publish_role). */
#define REPLICA_MAP_SAVED_MAX 64u
static const pn_tt_activation *g_saved_strips[REPLICA_MAP_SAVED_MAX];
static uint32_t g_saved_count;
static const pn_tt_activation *g_saved_split;

int pn_tt_replica_map_publish_role(pn_tt_device *device)
{
    ft_tt_op_replica_table table;
    pn_tt_residency *member;

    if (!device || replica_map_build(device, g_saved_strips, g_saved_count, g_saved_split, -1,
                                     &table) != 0)
        return -1;
    /* The role writes `normed` slices exactly as a reduce member does, so it gets the reduce
     * residency's table (the base keys, offsets added in the writer). */
    for (uint32_t m = 0; (member = pn_tt_norm_role_member_mut(device, m)) != NULL; m++)
        if (member->running && replica_map_publish_to(device, member, &table,
                                                      "a norm role member") != 0)
            return -1;
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
        replica_map_publish_to(device, &device->reduce, &table, "the reduce residency") != 0)
        return -1;
    /* L4: the column-split members write the same strips — without the table a member's slice
     * of every mirror goes stale (measured 2026-10-05: DIFFERS, deterministic). */
    for (index = 0u; index + 1u < PN_TT_REDUCE_MAX_MEMBERS; index++)
        if (replica_map_publish_to(device, &device->reduce_member[index], &table,
                                   "a reduce member") != 0)
            return -1;
    g_saved_count = strip_count < REPLICA_MAP_SAVED_MAX ? strip_count : REPLICA_MAP_SAVED_MAX;
    for (index = 0u; index < g_saved_count; index++)
        g_saved_strips[index] = strips[index];
    g_saved_split = split_output;
    if (pn_tt_replica_map_publish_role(device) != 0)
        return -1;
    if ((g_range_only_count || g_decode_range_count) && device->gate.residency.running &&
        replica_map_publish_to(device, &device->gate.residency, &table,
                               "the gate-narrow residency") != 0)
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
    /* S3b: the matmul members' table is independent of the strips passed here (it names the
     * registered fused strips by slot), so republishing it costs one write a member and keeps
     * every path that republishes the map also re-planting it. */
    return pn_tt_replica_map_publish_fused(device);
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
