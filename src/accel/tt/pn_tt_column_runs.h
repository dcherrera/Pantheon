/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_column_runs.h
 * @brief A PERMUTATION of a weight's output columns, expressed as runs of COLUMN TILES —
 *        the whole of the "de-interleave a joint projection" problem, with no card in it.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS: A JOINT PROJECTION IS INTERLEAVED PER HEAD
 * ---------------------------------------------------------------------------
 * Ornith's `blk.N.attn_q.weight` emits Q **and** the attention output gate from one matmul,
 * interleaved per head:
 *
 *     [ head 0 Q (head_dim) | head 0 gate (head_dim) | head 1 Q | head 1 gate | ... ]
 *
 * Every op downstream wants them apart and compact. The QK-RMSNorm reads head `h` as row
 * block `h` at tiles `h*tile_cols`, consecutive; flash decode reads a KV group as ONE
 * contiguous run of `q_rows*dim_tiles` tiles. Over the joint strip, neither lands: group
 * `g`'s run would be head `2g`'s Q followed by head `2g`'s GATE.
 * `foundry/tt/tests/test_attention_ops.c` asserts exactly that, so this is a measured
 * requirement rather than a tidiness one.
 *
 * ---------------------------------------------------------------------------
 * AND WHY IT IS SOLVED IN THE WEIGHT RATHER THAN PER TOKEN
 * ---------------------------------------------------------------------------
 * The three places it could be fixed, and what each costs:
 *
 *   1. a per-token card-side gather — a new op, 512 tiles a layer, 8 layers a token;
 *   2. a stride word in `resident_attention_reader.c` AND one in `resident_reduce_reader.c` —
 *      two kernel source changes, a program rebuild, and the reduce one is under the model's
 *      OWN RMS norms on the shipped default path;
 *   3. a PERMUTED UPLOAD of the weight at load — no kernel change, no per-token cost, and
 *      every downstream op runs at its already-proven shape.
 *
 * (3), and this file is its arithmetic. An N-major artifact lays column tile `c` as a
 * contiguous `tile_rows * tile_bytes` run, so a column permutation is a permutation of
 * contiguous byte ranges and the upload is one `ft_tt_fttw_upload_range()` per run per
 * shard. The cost is a second copy of the tensor in card DRAM — 71.3 MiB a layer for
 * Ornith's `attn_q`, 570 MiB over its 8 attention layers, against 32 GB of board memory.
 *
 * Nothing here touches a device, so it is provable with no card:
 * `tests/test_tt_column_runs.c`.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_COLUMN_RUNS_H
#define PN_TT_COLUMN_RUNS_H

#include <stdbool.h>
#include <stdint.h>

/** How many runs one destination weight may be assembled from. 32 heads x 2 halves is 64;
 *  this is the ceiling a caller's array has to satisfy and it is checked, not assumed. */
#define PN_TT_COLUMN_RUNS_MAX 128u

/** One contiguous run of SOURCE column tiles, in the order the DESTINATION wants them. */
typedef struct pn_tt_column_run {
    uint32_t first; /**< first source column tile */
    uint32_t count; /**< consecutive source column tiles, > 0 */
} pn_tt_column_run;

/**
 * Destination column tiles this run list produces.
 *
 * @return the sum of the counts, or 0 for an empty or malformed list
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_column_runs_total(const pn_tt_column_run *runs, uint32_t run_count);

/**
 * Do these runs stay inside a `source_columns`-wide weight, and does no source column appear
 * twice?
 *
 * The second half is what makes a wrong run list a REFUSAL rather than a weight with one
 * column duplicated and another missing — which would be a smoothly wrong projection with
 * nothing on the card able to see it.
 *
 * @return true when every run is in range and the destination uses each source column at
 *         most once
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_column_runs_are_injective(const pn_tt_column_run *runs, uint32_t run_count,
                                     uint32_t source_columns);

/**
 * Where destination column `destination_column` comes from, and how many destination columns
 * after it are contiguous in the SOURCE too.
 *
 * The second output is what turns a permutation into a small number of large transfers: a
 * caller uploads `*out_contiguous` columns in one range and asks again.
 *
 * @param out_source_column  the source column tile it reads
 * @param out_contiguous     destination columns from here that share the run, >= 1
 * @return                   false when `destination_column` is past the end
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_column_runs_map(const pn_tt_column_run *runs, uint32_t run_count,
                           uint32_t destination_column, uint32_t *out_source_column,
                           uint32_t *out_contiguous);

/**
 * Build the run list that lifts ONE interleaved field out of a joint projection.
 *
 * Field `offset..offset+width` of each of `groups` records of `stride` column tiles. For
 * Ornith's `attn_q` at head_dim 256 (dim_tiles 8): `groups` 32, `stride` 16, `width` 8, and
 * `offset` 0 for Q or 8 for the gate.
 *
 * @param out_runs  receives `groups` runs
 * @param max_runs  capacity of `out_runs`
 * @return          runs written, or 0 on a bad shape or too small an array
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_column_runs_field(uint32_t groups, uint32_t stride, uint32_t offset,
                                 uint32_t width, pn_tt_column_run *out_runs,
                                 uint32_t max_runs);

#endif /* PN_TT_COLUMN_RUNS_H */
