/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_mixw_fit.h
 * @brief Price a per-tensor FORMAT MAP (Bfp8_b / Bfp4_b) on the card, per DRAM bank, with the
 *        load's weight rotation. Refuses a map the converter or the runtime could not honour.
 *
 * WHAT IS PRICED. A list of card PLACEMENTS in the order the load makes them: the trunk
 * directory's N-major entries, then the arms' staged copies and the extra (MTP) artifact.
 * Each placement carries a GROUP: the GGUF tensor whose format it has. A map line names a
 * group, never a placement, because every placement of one group is the same source bytes:
 *   - one entry uploaded as column parts (the `attn_q` Q and gate halves);
 *   - one entry staged again later from the same bytes (`attn_q` halves beside the joint, the
 *     MTP draft-vocabulary copy of `output.weight` unless staging repacks it);
 *   - one GGUF tensor the converter writes as two entries (`nextn.eh_proj` ->
 *     `mtp.eh_proj_e` + `mtp.eh_proj_h`).
 * A map that names a placement apart from its group would give one source two formats: it is
 * refused, as are unknown names, names listed twice, groups the converter refuses to map
 * (`map_refusal`, mirroring foundry ft_tt_ledger_map_refusal()) and any format but `bfp8_b` /
 * `bfp4_b` (the only dense matmul tiles a resident program serves).
 *
 * THE ROTATION (default `PN_TT_WEIGHT_BANKS`, the only mode priced). From
 * pn_tt_device_open() / pn_tt_place_resident_entry() / pn_tt_matmul_group_place_weight_columns():
 *   - the cursor starts at bank `first_bank` (2);
 *   - a placement over M members gives member m `cols / M` columns, the first `cols % M`
 *     members one more, in bank (cursor + m) % banks; with fewer columns than members (or
 *     M < 2) it goes whole into bank cursor; either way the cursor then moves on by ONE;
 *   - a LOAD entry's parts take consecutive cursor positions from the entry's own, and the
 *     cursor then moves on by ONE from where the entry started.
 * DRAM buffers are 64 B aligned; 1,088 B and 576 B tiles are multiples, so no padding.
 *
 * No device: arithmetic over a list, tested on any host (`tests/test_tt_q38_mixw_fit.c`).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_MIXW_FIT_H
#define PN_TT_MIXW_FIT_H

#include <stddef.h>
#include <stdint.h>

/** Bytes of one 32x32 weight tile on the card. */
#define PN_TT_MIXW_TILE_BYTES_BFP8 1088u
#define PN_TT_MIXW_TILE_BYTES_BFP4 576u

/** Most DRAM banks priced (a p150 has 8). */
#define PN_TT_MIXW_MAX_BANKS 16u

/** Room for a refusal, which names the map line and the group. */
#define PN_TT_MIXW_REASON_BYTES 320u

/** How a placement reaches (or does not reach) the card. */
typedef enum pn_tt_mixw_placement_kind {
    /** A trunk directory entry uploaded at load, `parts` equal column parts. */
    PN_TT_MIXW_PLACE_LOAD = 0,
    /** One placement after the load: an extra-artifact weight or a staged column copy. */
    PN_TT_MIXW_PLACE_STAGED = 1,
    /** Not on the card: `token_embd`, whose row the host decodes from the mmap. */
    PN_TT_MIXW_PLACE_HOST = 2,
} pn_tt_mixw_placement_kind;

/** One weight placement. Strings are borrowed for the call. */
typedef struct pn_tt_mixw_placement {
    const char *placement_name; /**< what lands: the artifact entry, or a staged copy's label */
    const char *group_name;     /**< the GGUF tensor whose format it carries: the map key */
    const char *map_refusal;    /**< NULL when a map may name the group; else why not */
    uint32_t tile_rows;         /**< Kt */
    uint32_t tile_cols;         /**< Nt of the whole placement (all parts together) */
    uint32_t parts;             /**< LOAD: equal column parts uploaded; otherwise 1 */
    pn_tt_mixw_placement_kind kind;
} pn_tt_mixw_placement;

/** One map line, as the shared map file holds it: `<gguf tensor name> <fttw format name>`. */
typedef struct pn_tt_mixw_map_entry {
    const char *tensor_name;
    const char *format_name;
} pn_tt_mixw_map_entry;

/** The card the map is priced on. */
typedef struct pn_tt_mixw_card {
    uint32_t bank_count;         /**< DRAM banks (8 on a p150) */
    uint32_t members;            /**< decode matmul members a weight is sharded over (16) */
    uint32_t first_bank;         /**< the rotation's starting bank (2) */
    uint32_t default_tile_bytes; /**< the artifact rung's tile for unmapped groups (1,088) */
} pn_tt_mixw_card;

/** A priced map. */
typedef struct pn_tt_mixw_fit {
    uint64_t bank_bytes[PN_TT_MIXW_MAX_BANKS]; /**< weight bytes per bank */
    uint64_t card_bytes;       /**< all banks */
    uint64_t load_bytes;       /**< LOAD placements: what the load log calls "weights" */
    uint64_t staged_bytes;     /**< STAGED placements */
    uint64_t host_bytes;       /**< HOST placements at their format (artifact bytes, no card) */
    uint32_t fullest_bank;     /**< most weight bytes; the lowest index wins a tie */
    uint32_t bfp4_placements;  /**< card placements priced at 576 B */
    uint32_t mapped_lines;     /**< map lines applied */
} pn_tt_mixw_fit;

/**
 * Tile bytes of an FTTW format name a map may ask for: `bfp8_b` 1,088, `bfp4_b` 576, else 0.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_mixw_format_tile_bytes(const char *format_name);

/**
 * Bytes of one placement at `tile_bytes`.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_tt_mixw_placement_bytes(const pn_tt_mixw_placement *placement, uint32_t tile_bytes);

/**
 * How many placements belong to `group_name` (a group of more than one is a set the map
 * moves together).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
size_t pn_tt_mixw_group_size(const pn_tt_mixw_placement *placements, size_t count,
                             const char *group_name);

/**
 * Check `map` against `placements`, then price every placement per bank.
 *
 * @param placements     in load order
 * @param map            the map lines (NULL with `map_count` 0 = the artifact's rung)
 * @param card           banks, members, rotation start, default tile
 * @param out            the priced map; zeroed first
 * @param out_tile_bytes optional, `count` entries: each placement's tile bytes
 * @param reason         receives why a map or a placement list was refused; may be NULL
 * @return 0, or -1 with `reason` filled (nothing priced)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int pn_tt_mixw_fit_price(const pn_tt_mixw_placement *placements, size_t count,
                         const pn_tt_mixw_map_entry *map, size_t map_count,
                         const pn_tt_mixw_card *card, pn_tt_mixw_fit *out,
                         uint32_t *out_tile_bytes, char *reason, size_t reason_bytes);

#endif /* PN_TT_MIXW_FIT_H */
