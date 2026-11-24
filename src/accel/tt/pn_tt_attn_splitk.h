/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_splitk.h
 * @brief SPLIT-K decode attention's PLAN: which
 *        position chunks each KV group's flash-decode is cut into, which core and partial
 *        slot each chunk's PARTIAL descriptor takes, and where each group's MERGE reads.
 *        Pure arithmetic, no device: the card path is pn_tt_attention_splitk.c.
 *
 * Today's lanes path sends one descriptor per KV group, so one core walks a group's whole
 * KV: 34 ms of attention a token at 28k, linear in depth. Split-K cuts each group's
 * positions into chunks over every split core; each chunk is a PARTIAL descriptor emitting
 * its un-normalised O, its max m and its sum l (foundry resident_attention_protocol.h §7),
 * and one MERGE descriptor per group folds them after the host has awaited them all.
 *
 * THE CHUNKS: per group `C = min(cores / kv_heads, live / min_chunk_positions)`, at least 2
 * (fewer stays on lanes, as does a call below `min_positions`, the switch-over depth). Every
 * chunk but the last is a whole number of DOUBLE BLOCKS (2 * block_tiles position tiles,
 * 1,024 positions at the shipped 16): the reader pads each descriptor to double blocks, so
 * aligned chunks read nothing twice. Chunks never cross a BANK RUN (a position range
 * contiguous in one DRAM bank; today every head is one run from position 0, so this costs
 * nothing, but a layout that splits a head across banks must not make one descriptor read
 * two banks). The last chunk holds every row's causal limit, because a verify's rows all
 * end in one position tile (pn_tt_attention_rows' precondition).
 *
 * SLOTS: descriptor d is group `d / C`, chunk `d % C`, on core `d % cores` in round
 * `d / cores`, writing partial slot d. A group's partials are therefore contiguous: its
 * merge reads slots [group * C, group * C + C) in chunk order.
 *
 * @status On by default (K8V4 caches only, program `attention_k8v4_split`; card gate plan
 *         3.3 passed); `PN_TT_ATTN_SPLITK=0` restores lanes. `PN_TT_ATTN_SPLITK_MIN_POSITIONS`
 *         (default 8,192 switch-over), `PN_TT_ATTN_SPLITK_MIN_CHUNK` (default 1,024 positions).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ATTN_SPLITK_H
#define PN_TT_ATTN_SPLITK_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Partial descriptors one call may send: foundry FT_TT_OP_ATTENTION_MAX_PARTIALS. */
#define PN_TT_ATTN_SPLITK_MAX_PARTIALS 64u
/** Bank runs one head may be split across. */
#define PN_TT_ATTN_SPLITK_MAX_RUNS 8u

#define PN_TT_ATTN_SPLITK_MIN_POSITIONS_DEFAULT 8192u
#define PN_TT_ATTN_SPLITK_MIN_CHUNK_DEFAULT     1024u

/** The three knobs. */
typedef struct pn_tt_attn_splitk_config {
    bool enabled;                 /**< PN_TT_ATTN_SPLITK */
    uint32_t min_positions;       /**< switch-over: shorter calls stay on lanes */
    uint32_t min_chunk_positions; /**< no chunk but the last is shorter */
} pn_tt_attn_splitk_config;

/** Positions [first, end) of a head, contiguous in one DRAM bank (K and V alike). */
typedef struct pn_tt_attn_splitk_run {
    uint32_t first_position;
    uint32_t end_position;
} pn_tt_attn_splitk_run;

/** What one attention call looks like to the planner. */
typedef struct pn_tt_attn_splitk_geometry {
    uint32_t live_positions;     /**< the longest row's limit: the call's kv_positions */
    uint32_t shortest_live;      /**< the shortest row's limit (= live for one decode row) */
    uint32_t kv_heads;           /**< KV groups, one merge each */
    uint32_t cores;              /**< split cores armed */
    uint32_t block_tiles;        /**< the residency's flash-decode block */
    uint32_t capacity_tiles;     /**< position tiles the K / V region holds */
    const pn_tt_attn_splitk_run *runs; /**< a head's bank runs, ascending, from 0 */
    uint32_t run_count;
} pn_tt_attn_splitk_geometry;

/** Position tiles [first_tile, first_tile + tile_count). */
typedef struct pn_tt_attn_splitk_chunk {
    uint32_t first_tile;
    uint32_t tile_count;
} pn_tt_attn_splitk_chunk;

/** One call's plan: the same chunks for every KV group. chunk_count 0 = stay on lanes. */
typedef struct pn_tt_attn_splitk_plan {
    uint32_t chunk_count;                                   /**< per group */
    pn_tt_attn_splitk_chunk chunk[PN_TT_ATTN_SPLITK_MAX_PARTIALS];
    uint32_t partials;                                      /**< kv_heads * chunk_count */
    uint32_t rounds;                                        /**< ceil(partials / cores) */
} pn_tt_attn_splitk_plan;

/** Where PARTIAL descriptor `index` goes and what it reads. */
typedef struct pn_tt_attn_splitk_slot {
    uint32_t group;           /**< KV group */
    uint32_t chunk;           /**< chunk within the group */
    uint32_t core;            /**< split core, 0 = the attention residency */
    uint32_t round;           /**< rounds run in order, each awaited */
    uint32_t slot;            /**< partial slot written (= index) */
    uint32_t first_tile;      /**< folded into the K and V bases */
    uint32_t kv_positions;    /**< chunk-relative live length */
    uint32_t capacity_tiles;  /**< the region's tiles past first_tile */
    bool last;                /**< holds the causal limit: tail / staircase mask */
} pn_tt_attn_splitk_slot;

/**
 * Parse the knobs. NULL or empty = the default (on); `enabled` is "1" to arm, "0" for lanes.
 * @return 0, or -1 for a value that is not a number (the config is then the defaults, off)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_attn_splitk_config_parse(const char *enabled, const char *min_positions,
                                   const char *min_chunk, pn_tt_attn_splitk_config *out);

/**
 * Plan one call. Writes chunk_count 0 (lanes) when split-K is off, the call is under the
 * switch-over, fewer than two chunks fit, or a bank run is not tile-aligned.
 * @return 0, or -1 for a NULL argument or a geometry with zeros in it
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_attn_splitk_plan_make(const pn_tt_attn_splitk_config *config,
                                const pn_tt_attn_splitk_geometry *geometry,
                                pn_tt_attn_splitk_plan *out);

/**
 * Is `plan` executable for `geometry`? Checks every rule the card relies on: the chunks
 * cover [0, live tiles) exactly once in order, none empty, all but the last double-block
 * aligned and at least min_chunk_positions long, none crossing a bank run, the last holding
 * every row's limit, every padded read window inside the region, the partial count within
 * PN_TT_ATTN_SPLITK_MAX_PARTIALS and the rounds what the cores need.
 * @return NULL when it is, else a static string naming the first broken rule
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const char *pn_tt_attn_splitk_plan_check(const pn_tt_attn_splitk_config *config,
                                         const pn_tt_attn_splitk_geometry *geometry,
                                         const pn_tt_attn_splitk_plan *plan);

/**
 * PARTIAL descriptor `index` (< plan->partials) of `plan`.
 * @return 0, or -1 for an index past the plan
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_attn_splitk_slot_of(const pn_tt_attn_splitk_geometry *geometry,
                              const pn_tt_attn_splitk_plan *plan, uint32_t index,
                              pn_tt_attn_splitk_slot *out);

/**
 * The plan's SHAPE KEY, for on-card dispatch replay (pn_tt_dispatch_shape_key): 0 for lanes,
 * else a non-zero hash of the chunk count and every chunk's first tile and, but for the
 * last, its length. Two depths with one key send the same descriptors to the same cores
 * with every word equal or moving with the depth (only the last chunk's live length grows),
 * which is what the replay's linear model predicts. A key change is a token it cannot: the
 * switch-over (8,192), then every depth where the chunk length steps up a double block
 * (Qwen3.8 p150: 9,217, 12,289, ... every 3,072).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_tt_attn_splitk_plan_key(const pn_tt_attn_splitk_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_ATTN_SPLITK_H */
