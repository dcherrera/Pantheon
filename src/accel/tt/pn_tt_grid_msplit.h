/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_grid_msplit.h
 * @brief GRID M SPLIT (2026-10-26): the prefill grid's narrow shapes cut by ROWS, not columns.
 *
 * WHY. The grid gives each of a decode shard's `pieces` cores a column piece over ALL Mt rows,
 * so every core reads the whole activation A. For the Nt 128 shapes (ssm z / attn gate: 8
 * columns a shard, pieces 3 / 3 / 2) and attn k / v (2 columns: pieces 1 / 1 / 0, a third of
 * the cores idle) the projection is bound by those A reads, not by compute. Cut instead by
 * rows: each of the shard's cores takes ALL the shard's columns over a third of the rows, so
 * A is read once per shard (a third of the bytes) and B (which fits the L1 stash) is read
 * three times — B is ~7% of A's bytes at 2,048 rows. Every output tile is the same sum in the
 * same Kt order: only which core computes it changes.
 *
 * Also the GRID STASH CHUNKS: a piece too wide for the stash split into stash-sized column
 * chunks (`pn_tt_grid_stash_chunks`).
 *
 * @status Default on (`PN_TT_GRID_MSPLIT=0` off). Red `grid_msplit_a_shift`. Stash chunks
 *         default on (`PN_TT_GRID_STASH_CHUNKS=0` off), red `grid_chunk_b_shift`.
 * @issues None
 * @todo   Out-projections reach the grid only once their inputs are mirrored.
 */
#ifndef PN_TT_GRID_MSPLIT_H
#define PN_TT_GRID_MSPLIT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Does a shard of `columns` tile columns, inner `kt`, take the row split?
 *
 * @param columns      the shard's tile columns
 * @param kt           the weight's inner tiles
 * @param stash_tiles  the grid core's L1 B stash (0: no stash — no split, B would stream)
 * @param mt           tile rows of this projection
 * @param pieces       grid cores a shard
 * @param max_columns  widest shard that splits (`PN_TT_GRID_MSPLIT_COLS`, 0 = off)
 * @return true when every piece gets at least one row block and the shard fits the stash
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
bool pn_tt_grid_msplit_applies(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                               uint32_t mt, uint32_t pieces, uint32_t max_columns);

/**
 * Rows of piece `piece`: whole row blocks of `row_block` dealt in order, the first
 * `blocks % pieces` pieces one block more; the last block may be short.
 *
 * @param out_first  first tile row
 * @param out_rows   tile rows (0 = this piece has none)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void pn_tt_grid_msplit_rows(uint32_t mt, uint32_t pieces, uint32_t row_block, uint32_t piece,
                            uint32_t *out_first, uint32_t *out_rows);

/** `PN_TT_GRID_MSPLIT_COLS` (default 8; `PN_TT_GRID_MSPLIT=0` -> 0). Read once. */
uint32_t pn_tt_grid_msplit_max_columns(void);

/**
 * GRID STASH CHUNKS (2026-11-15): descriptors a column piece of `columns` x `kt` tiles takes so
 * that each one's B fits the stash. A piece wider than the stash streams B from DRAM once per
 * row block (Qwen3.8's FFN: Mt 64 sweeps of ~2 MB a core); as balanced column chunks each
 * loads its B into the stash once and re-reads only A. Each output tile is still one
 * descriptor's full Kt sum on the same core: the same bits.
 *
 * @return 1 when the piece fits (or `enabled` is false, or no column fits: B streams as
 *         before), else ceil(columns / (stash_tiles / kt))
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t pn_tt_grid_stash_chunks(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                                 bool enabled);

/**
 * Columns of chunk `chunk` of `chunks`: balanced, the first `columns % chunks` one wider.
 *
 * @param out_first  first column inside the piece
 * @param out_cols   columns
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void pn_tt_grid_stash_chunk_columns(uint32_t columns, uint32_t chunks, uint32_t chunk,
                                    uint32_t *out_first, uint32_t *out_cols);

/** `PN_TT_GRID_STASH_CHUNKS` (default on; `=0` off). Read once. */
bool pn_tt_grid_stash_chunks_enabled(void);

/**
 * GRID UNSPLIT ROWS (q38-gdn-chain O.0): grid cores a multi-row projection of an UNSPLIT weight
 * is cut over, by rows. Such a weight (Qwen3.8's `ssm_alpha` / `ssm_beta`: Nt 2, one shard)
 * otherwise runs on ONE decode member sweeping every row — at a 13-band super-band 1.13 ms
 * each, on the chain's critical path (the wavefront flushes them). Cut by rows over the grid,
 * each core takes ALL the columns over its rows: every output tile is the same full-Kt sum in
 * the same order, only the core that computes it moves.
 *
 * @param columns      the weight's tile columns
 * @param kt           the weight's inner tiles
 * @param stash_tiles  the grid core's L1 B stash (0: no stash — no split)
 * @param mt           tile rows of this projection
 * @param row_block    the matmul row block (rows are dealt in whole blocks)
 * @param grid_cores   grid cores available
 * @param max_columns  widest weight that splits (`pn_tt_grid_unsplit_max_columns`, 0 = off)
 * @return pieces (each gets >= one row block), or 0 when the rule does not apply
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t pn_tt_grid_unsplit_pieces(uint32_t columns, uint32_t kt, uint32_t stash_tiles,
                                   uint32_t mt, uint32_t row_block, uint32_t grid_cores,
                                   uint32_t max_columns);

/** `PN_TT_GRID_UNSPLIT` (default on: 8 columns; `=0` off). Read once. */
uint32_t pn_tt_grid_unsplit_max_columns(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_GRID_MSPLIT_H */
