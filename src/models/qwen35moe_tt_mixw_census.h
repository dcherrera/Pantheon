/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_tt_mixw_census.h
 * @brief The qwen35 dense hybrid's card weight placements (Qwen3.8-27B, Ornith, Bonsai), in
 *        load order and grouped by the GGUF tensor a format map names (pn_tt_mixw_fit.h).
 *
 * ORDER (what pn_tt_mixw_fit prices the rotation over):
 *   1. the trunk artifact's N-major entries in directory order: `output.weight`, then each
 *      block's entries by name. `attn_q` is
 *      ONE entry of two parts when the fit policy splits it at load (qwen35moe_fit_tt.c);
 *   2. without that split, the attention arm's staged Q and gate halves (a second copy);
 *   3. with MTP, the extra artifact in pn_qwen35_tt_mtp_arm() order (`eh_proj_e`, `eh_proj_h`,
 *      `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`), its `attn_q` halves,
 *      then the draft-vocabulary copy staged from `output.weight` columns.
 * `token_embd` is listed as HOST: one row a token is decoded on the host from the mmap.
 *
 * GROUPS that hold more than one placement (one source, one format):
 *   - `blk.N.attn_q.weight`: its two halves (split at load), or the joint + two staged halves;
 *   - `blk.L.attn_q.weight` (L = n_layer, the MTP block): the MTP Q and gate halves;
 *   - `blk.L.nextn.eh_proj.weight`: `mtp.eh_proj_e` + `mtp.eh_proj_h`;
 *   - `output.weight`: the LM head + the MTP draft-vocabulary copy, UNLESS staging repacks the
 *     copy to its own format (spec §6.1, plan 4.1): then the copy is its own group,
 *     `mtp.draft_vocab.weight` — a Pantheon-side name, not a GGUF tensor.
 * No two GGUF tensors share a descriptor: every decode and prefill matmul dispatch takes ONE
 * weight (pn_tt_matmul_group_enqueue, pn_tt_matmul_grid_issue_detached), and the resident
 * program reads the format per descriptor (spec §6.1 R-a).
 *
 * `ssm_alpha` / `ssm_beta` and `token_embd` carry the converter's map refusal (foundry
 * ft_tt_ledger_map_refusal()).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef QWEN35MOE_TT_MIXW_CENSUS_H
#define QWEN35MOE_TT_MIXW_CENSUS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../accel/tt/pn_tt_mixw_fit.h"

/** Room for every placement of a 64-layer trunk, its staged halves and the MTP head. */
#define PN_QWEN35_TT_MIXW_MAX_PLACEMENTS 1024u
#define PN_QWEN35_TT_MIXW_NAME_BYTES 64u

/** The shape, in 32-wide tiles, and the load choices that move placements. */
typedef struct pn_qwen35_tt_mixw_shape {
    uint32_t n_layer;       /**< trunk layers */
    uint32_t attn_interval; /**< layer L is full attention when (L + 1) % interval == 0 */
    uint32_t embd_tiles, ffn_tiles, conv_tiles, inner_tiles, dt_tiles;
    uint32_t head_tiles;    /**< n_head x head_dim / 32: one half of the joint attn_q */
    uint32_t kv_tiles;      /**< n_head_kv x head_dim / 32 */
    uint32_t vocab_tiles;
    bool split_q_at_load;   /**< fit policy A: attn_q uploaded as its halves, no second copy */
    bool mtp;               /**< the dense MTP head (extra artifact) is armed */
    uint32_t draft_cols;    /**< MTP draft-vocabulary columns staged (2,057); 0 = none */
    bool draft_repacked;    /**< staging writes the draft copy at its own format (plan 4.1) */
} pn_qwen35_tt_mixw_shape;

/** The placements and the names they point into. Large: keep it static or on the heap. */
typedef struct pn_qwen35_tt_mixw_census {
    pn_tt_mixw_placement placements[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];
    char names[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS][2][PN_QWEN35_TT_MIXW_NAME_BYTES];
    size_t count;
} pn_qwen35_tt_mixw_census;

/**
 * Fill `out` with the shape's placements in load order (see the file note).
 *
 * @return 0, or -1 when the shape is empty or does not fit the census
 *
 * Thread-safe: yes (only `out`)    Reentrant: yes    Signal-safe: yes
 */
int pn_qwen35_tt_mixw_census_build(const pn_qwen35_tt_mixw_shape *shape,
                                   pn_qwen35_tt_mixw_census *out);

#endif /* QWEN35MOE_TT_MIXW_CENSUS_H */
