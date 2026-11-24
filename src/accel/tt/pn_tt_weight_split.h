/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_split.h
 * @brief Which column parts of an artifact weight go up to the card at load: the whole
 *        tensor, or only its de-interleaved parts (a joint Q+gate `attn_q` as Q and gate).
 *
 * On-card attention reads `attn_q` only as two column-permuted halves
 * (`pn_tt_column_runs.h`). Uploading the joint tensor as well kept a second copy resident
 * that no on-card op reads: 63.75 MiB a layer, 0.996 GiB over Qwen3.8-27B's 16 attention
 * layers. A split rule says "upload this tensor as its parts instead", and this file is the
 * per-tensor decision plus the parts' run lists and byte counts. No device, so it is tested
 * on any host (`tests/test_tt_q38_fit.c`).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_WEIGHT_SPLIT_H
#define PN_TT_WEIGHT_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_column_runs.h"

/** Longest tensor-name suffix a rule can match, terminator included. */
#define PN_TT_WEIGHT_SPLIT_SUFFIX_BYTES 48u

/** Most parts one field can be cut into. Q and gate are 2. */
#define PN_TT_WEIGHT_SPLIT_MAX_PARTS 4u

/**
 * A tensor whose output columns are `fields` interleaved records of `field_tiles` column
 * tiles, uploaded as `field_tiles / part_tiles` compact parts. Part p is columns
 * `p*part_tiles .. (p+1)*part_tiles` of every record. For `attn_q` at head_dim 256:
 * `fields` = heads, `field_tiles` 16, `part_tiles` 8; part 0 is Q, part 1 the gate.
 */
typedef struct pn_tt_weight_split_rule {
    char suffix[PN_TT_WEIGHT_SPLIT_SUFFIX_BYTES]; /**< e.g. ".attn_q.weight" */
    uint32_t fields;
    uint32_t field_tiles;
    uint32_t part_tiles;
} pn_tt_weight_split_rule;

/** One upload of one tensor: the whole of it, or one column part. */
typedef struct pn_tt_weight_upload_part {
    int32_t part;           /**< -1 = the whole tensor, else the part index */
    uint32_t tile_cols;     /**< destination column tiles */
    uint64_t bytes;         /**< card DRAM it takes */
    uint32_t run_count;     /**< 0 for the whole tensor */
    pn_tt_column_run runs[PN_TT_COLUMN_RUNS_MAX];
} pn_tt_weight_upload_part;

/**
 * Build a rule. Refuses a shape the parts cannot tile exactly.
 *
 * @return 0, or -1 when `suffix` is empty or too long, or `part_tiles` does not divide
 *         `field_tiles` into 2..PN_TT_WEIGHT_SPLIT_MAX_PARTS parts
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_weight_split_rule_init(pn_tt_weight_split_rule *rule, const char *suffix,
                                 uint32_t fields, uint32_t field_tiles, uint32_t part_tiles);

/** Parts per field under `rule`. */
uint32_t pn_tt_weight_split_part_count(const pn_tt_weight_split_rule *rule);

/**
 * Does `rule` apply to this tensor? Its name must end in the suffix and its width must be
 * exactly `fields * field_tiles`. A name that matches at another width is NOT split: it is
 * a different shape and the caller refuses it where the parts are read.
 */
bool pn_tt_weight_split_applies(const pn_tt_weight_split_rule *rule, const char *name,
                                uint32_t tile_cols);

/**
 * What to upload for one N-major tensor.
 *
 * @param rule        NULL for none
 * @param name        artifact tensor name
 * @param tile_cols   the tensor's column tiles
 * @param byte_count  the tensor's bytes (an N-major column tile is `byte_count / tile_cols`)
 * @param out_parts   receives the uploads, PN_TT_WEIGHT_SPLIT_MAX_PARTS of room
 * @return            uploads written: 1 for the whole tensor, the part count when split,
 *                    0 on a malformed shape
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_weight_upload_parts(const pn_tt_weight_split_rule *rule, const char *name,
                                   uint32_t tile_cols, uint64_t byte_count,
                                   pn_tt_weight_upload_part *out_parts);

#endif /* PN_TT_WEIGHT_SPLIT_H */
