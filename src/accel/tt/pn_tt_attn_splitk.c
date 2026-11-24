/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_splitk.c
 * @brief Split-K decode attention's plan. See pn_tt_attn_splitk.h.
 * @status On by default (`PN_TT_ATTN_SPLITK=0` restores lanes).
 * @issues None
 * @todo None
 */
#include "pn_tt_attn_splitk.h"

#include <stdlib.h>
#include <string.h>

#define SPLITK_TILE_POSITIONS 32u

static uint32_t tiles_of(uint32_t positions)
{
    return (positions + SPLITK_TILE_POSITIONS - 1u) / SPLITK_TILE_POSITIONS;
}

static uint32_t round_up(uint32_t value, uint32_t multiple)
{
    return (value + multiple - 1u) / multiple * multiple;
}

/** "N" as a u32; NULL / empty keeps `fallback`. */
static int parse_u32(const char *text, uint32_t fallback, uint32_t *out)
{
    char *end = NULL;
    unsigned long value;

    *out = fallback;
    if (!text || !*text)
        return 0;
    value = strtoul(text, &end, 10);
    if (!end || *end != '\0' || value > UINT32_MAX)
        return -1;
    *out = (uint32_t)value;
    return 0;
}

int pn_tt_attn_splitk_config_parse(const char *enabled, const char *min_positions,
                                   const char *min_chunk, pn_tt_attn_splitk_config *out)
{
    uint32_t on = 0u;

    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    out->min_positions = PN_TT_ATTN_SPLITK_MIN_POSITIONS_DEFAULT;
    out->min_chunk_positions = PN_TT_ATTN_SPLITK_MIN_CHUNK_DEFAULT;
    if (parse_u32(enabled, 1u, &on) != 0 ||
        parse_u32(min_positions, PN_TT_ATTN_SPLITK_MIN_POSITIONS_DEFAULT,
                  &out->min_positions) != 0 ||
        parse_u32(min_chunk, PN_TT_ATTN_SPLITK_MIN_CHUNK_DEFAULT,
                  &out->min_chunk_positions) != 0 ||
        out->min_chunk_positions == 0u) {
        out->enabled = false;
        out->min_positions = PN_TT_ATTN_SPLITK_MIN_POSITIONS_DEFAULT;
        out->min_chunk_positions = PN_TT_ATTN_SPLITK_MIN_CHUNK_DEFAULT;
        return -1;
    }
    out->enabled = on == 1u;
    return 0;
}

static bool geometry_is_valid(const pn_tt_attn_splitk_geometry *geometry)
{
    return geometry->live_positions != 0u && geometry->shortest_live != 0u &&
           geometry->shortest_live <= geometry->live_positions && geometry->kv_heads != 0u &&
           geometry->cores != 0u && geometry->block_tiles != 0u &&
           geometry->capacity_tiles != 0u && geometry->runs != NULL &&
           geometry->run_count != 0u && geometry->run_count <= PN_TT_ATTN_SPLITK_MAX_RUNS;
}

/** Runs ascending, contiguous from 0, tile-aligned, covering the live positions. */
static bool runs_are_tiled(const pn_tt_attn_splitk_geometry *geometry)
{
    uint32_t next = 0u;

    for (uint32_t r = 0; r < geometry->run_count; r++) {
        const pn_tt_attn_splitk_run *run = &geometry->runs[r];

        if (run->first_position != next || run->end_position <= run->first_position ||
            run->first_position % SPLITK_TILE_POSITIONS != 0u)
            return false;
        next = run->end_position;
    }
    return next >= geometry->live_positions;
}

/** Chunks of `per` tiles from each run's start, the last of a run ragged. */
static uint32_t cut_chunks(const pn_tt_attn_splitk_geometry *geometry, uint32_t per,
                           pn_tt_attn_splitk_chunk *chunks)
{
    const uint32_t live_tiles = tiles_of(geometry->live_positions);
    uint32_t count = 0u;

    for (uint32_t r = 0; r < geometry->run_count; r++) {
        const uint32_t first = geometry->runs[r].first_position / SPLITK_TILE_POSITIONS;
        uint32_t end = tiles_of(geometry->runs[r].end_position);

        if (end > live_tiles)
            end = live_tiles;
        for (uint32_t tile = first; tile < end; tile += per) {
            if (count == PN_TT_ATTN_SPLITK_MAX_PARTIALS)
                return UINT32_MAX;
            chunks[count].first_tile = tile;
            chunks[count].tile_count = end - tile < per ? end - tile : per;
#ifdef PN_TT_ATTN_SPLITK_RED
            /* RED (make test-tt-attn-splitk-red): every chunk after the first starts one tile
             * early, re-reading its predecessor's last tile. Must FAIL. */
            if (count > 0u) {
                chunks[count].first_tile--;
                chunks[count].tile_count++;
            }
#endif
            count++;
        }
    }
    return count;
}

int pn_tt_attn_splitk_plan_make(const pn_tt_attn_splitk_config *config,
                                const pn_tt_attn_splitk_geometry *geometry,
                                pn_tt_attn_splitk_plan *out)
{
    const uint32_t align = geometry ? 2u * geometry->block_tiles : 0u;
    uint32_t wanted, live_tiles, per, count;

    if (!config || !geometry || !out)
        return -1;
    memset(out, 0, sizeof(*out));
    if (!geometry_is_valid(geometry))
        return -1;
    if (!config->enabled || geometry->live_positions < config->min_positions ||
        !runs_are_tiled(geometry))
        return 0;
    wanted = geometry->cores / geometry->kv_heads;
    if (wanted > geometry->live_positions / config->min_chunk_positions)
        wanted = geometry->live_positions / config->min_chunk_positions;
    if (wanted > PN_TT_ATTN_SPLITK_MAX_PARTIALS / geometry->kv_heads)
        wanted = PN_TT_ATTN_SPLITK_MAX_PARTIALS / geometry->kv_heads;
    if (wanted < 2u)
        return 0;
    live_tiles = tiles_of(geometry->live_positions);
    per = round_up((live_tiles + wanted - 1u) / wanted, align);
    if (per < round_up(tiles_of(config->min_chunk_positions), align))
        per = round_up(tiles_of(config->min_chunk_positions), align);
    /* Run edges can add chunks; widen until every group's chunks fit the partial slots. */
    for (count = cut_chunks(geometry, per, out->chunk);
         (count == UINT32_MAX || count * geometry->kv_heads > PN_TT_ATTN_SPLITK_MAX_PARTIALS) &&
         per < live_tiles;
         count = cut_chunks(geometry, per, out->chunk))
        per += align;
    if (count < 2u || count == UINT32_MAX ||
        count * geometry->kv_heads > PN_TT_ATTN_SPLITK_MAX_PARTIALS) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->chunk_count = count;
    out->partials = count * geometry->kv_heads;
    out->rounds = (out->partials + geometry->cores - 1u) / geometry->cores;
    return 0;
}

/** The run holding tile `tile`, or run_count. */
static uint32_t run_of(const pn_tt_attn_splitk_geometry *geometry, uint32_t tile)
{
    for (uint32_t r = 0; r < geometry->run_count; r++)
        if (tile * SPLITK_TILE_POSITIONS >= geometry->runs[r].first_position &&
            tile * SPLITK_TILE_POSITIONS < geometry->runs[r].end_position)
            return r;
    return geometry->run_count;
}

/** The padded window the reader fetches for a chunk: its live tiles to double blocks. */
static uint32_t padded_tiles(const pn_tt_attn_splitk_geometry *geometry,
                             const pn_tt_attn_splitk_chunk *chunk)
{
    return round_up(chunk->tile_count, 2u * geometry->block_tiles);
}

static const char *chunk_check(const pn_tt_attn_splitk_config *config,
                               const pn_tt_attn_splitk_geometry *geometry,
                               const pn_tt_attn_splitk_plan *plan, uint32_t c, uint32_t next)
{
    const pn_tt_attn_splitk_chunk *chunk = &plan->chunk[c];
    const bool last = c + 1u == plan->chunk_count;

    if (chunk->tile_count == 0u)
        return "an empty chunk";
    if (chunk->first_tile < next)
        return "chunks overlap";
    if (chunk->first_tile > next)
        return "a gap between chunks";
    if (!last && chunk->tile_count % (2u * geometry->block_tiles) != 0u &&
        run_of(geometry, chunk->first_tile + chunk->tile_count) ==
            run_of(geometry, chunk->first_tile))
        return "a chunk that is not whole double blocks";
    if (!last && chunk->tile_count * SPLITK_TILE_POSITIONS < config->min_chunk_positions &&
        run_of(geometry, chunk->first_tile + chunk->tile_count) ==
            run_of(geometry, chunk->first_tile))
        return "a chunk shorter than the minimum";
    if (run_of(geometry, chunk->first_tile) !=
        run_of(geometry, chunk->first_tile + chunk->tile_count - 1u))
        return "a chunk crossing a bank run";
    if (chunk->first_tile + padded_tiles(geometry, chunk) > geometry->capacity_tiles)
        return "a padded read window past the region";
    return NULL;
}

const char *pn_tt_attn_splitk_plan_check(const pn_tt_attn_splitk_config *config,
                                         const pn_tt_attn_splitk_geometry *geometry,
                                         const pn_tt_attn_splitk_plan *plan)
{
    uint32_t next = 0u;

    if (!config || !geometry || !plan || !geometry_is_valid(geometry))
        return "a NULL or zero argument";
    if (plan->chunk_count == 0u)
        return plan->partials == 0u ? NULL : "partials without chunks";
    if (plan->chunk_count < 2u || plan->chunk_count > PN_TT_ATTN_SPLITK_MAX_PARTIALS)
        return "a chunk count outside 2..max";
    for (uint32_t c = 0; c < plan->chunk_count; c++) {
        const char *broken = chunk_check(config, geometry, plan, c, next);

        if (broken)
            return broken;
        next = plan->chunk[c].first_tile + plan->chunk[c].tile_count;
    }
    if (next != tiles_of(geometry->live_positions))
        return "chunks that do not end at the live length";
    if (plan->chunk[plan->chunk_count - 1u].first_tile >
        (geometry->shortest_live - 1u) / SPLITK_TILE_POSITIONS)
        return "a row's causal limit outside the last chunk";
    if (plan->partials != plan->chunk_count * geometry->kv_heads ||
        plan->partials > PN_TT_ATTN_SPLITK_MAX_PARTIALS)
        return "a partial count that is not chunks x groups within the maximum";
    if (plan->rounds != (plan->partials + geometry->cores - 1u) / geometry->cores)
        return "rounds that do not cover the partials";
    return NULL;
}

int pn_tt_attn_splitk_slot_of(const pn_tt_attn_splitk_geometry *geometry,
                              const pn_tt_attn_splitk_plan *plan, uint32_t index,
                              pn_tt_attn_splitk_slot *out)
{
    const pn_tt_attn_splitk_chunk *chunk;
    uint32_t end;

    if (!geometry || !plan || !out || plan->chunk_count == 0u || index >= plan->partials ||
        geometry->cores == 0u)
        return -1;
    memset(out, 0, sizeof(*out));
    out->group = index / plan->chunk_count;
    out->chunk = index % plan->chunk_count;
    out->core = index % geometry->cores;
    out->round = index / geometry->cores;
    out->slot = index;
    chunk = &plan->chunk[out->chunk];
    end = (chunk->first_tile + chunk->tile_count) * SPLITK_TILE_POSITIONS;
    out->first_tile = chunk->first_tile;
    out->kv_positions = (geometry->live_positions < end ? geometry->live_positions : end) -
                        chunk->first_tile * SPLITK_TILE_POSITIONS;
    out->capacity_tiles = geometry->capacity_tiles - chunk->first_tile;
    out->last = out->chunk + 1u == plan->chunk_count;
    return 0;
}

/** FNV-1a over one 32-bit word. */
static uint64_t key_mix(uint64_t hash, uint32_t word)
{
    for (unsigned byte = 0; byte < 4u; byte++) {
        hash ^= (word >> (8u * byte)) & 0xFFu;
        hash *= 0x100000001B3ull;
    }
    return hash;
}

uint64_t pn_tt_attn_splitk_plan_key(const pn_tt_attn_splitk_plan *plan)
{
    uint64_t hash = 0xCBF29CE484222325ull;

    if (!plan || plan->chunk_count == 0u)
        return 0u;
    hash = key_mix(hash, plan->chunk_count);
#ifndef PN_TT_ATTN_SPLITK_KEY_RED
    /* RED (make test-tt-dispatch-shape-key-red): only the chunk count is keyed, so a step in
     * the chunk length (9,217) keeps the key. Must FAIL. */
    for (uint32_t c = 0; c < plan->chunk_count && c < PN_TT_ATTN_SPLITK_MAX_PARTIALS; c++) {
        hash = key_mix(hash, plan->chunk[c].first_tile);
        if (c + 1u < plan->chunk_count)
            hash = key_mix(hash, plan->chunk[c].tile_count);
    }
#endif
    return hash ? hash : 1u;
}
