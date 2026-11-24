/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_tiled.h
 * @brief  Decode foundry FTTW block-float tiles (the card's weight bytes) on the CPU: the
 *         PN_QUANT_TILED_* types an attached artifact contributes (gguf_fttw_overlay.h).
 *
 * The card-format oracle runs qwen35moe on the CPU over
 * the SAME bytes the card reads, so format loss and card arithmetic can be told apart.
 * Every tile is decoded by foundry's own ft_bfp_unpack_tiles — no second codec.
 *
 * Layouts (foundry tt/convert/ft_tt_weight_convert.h):
 *   PN_QUANT_TILED_BFP{8,4,6}  a matmul weight W [rows = N out, cols = K in], stored
 *       TRANSPOSED as K x N tiles in a column-major tile grid: tile column nt (output rows
 *       [32nt, 32nt+32)) is Kt = cols/32 consecutive tiles, and inside a decoded tile the
 *       value at [k*32 + n] is W[32nt + n][32kt + k]. Tile column nt starts at
 *       data + 32nt * row_stride (row_stride = Kt * tile_bytes / 32).
 *   PN_QUANT_TILED_ROWS_BFP8   the embedding: NOT transposed, row-major tile grid. Tile row
 *       rt (rows [32rt, 32rt+32)) is Ct = cols/32 tiles; decoded value [r*32 + c].
 *   PN_QUANT_TILED_FP32 / _ROWS_FP32  the same two layouts in fp32 tiles (face-ordered on
 *       disk), the exact-values arm the oracle is checked against the GGUF with.
 * cols (K) must be whole tiles. rows (N) of a plain 2-D weight may end in a partial tile: the
 * converter zero-pads the last strip and the matvec keeps the valid rows (the overlay refuses
 * every other shape).
 * `PN_TILED_RED=1` scales the first tile of each decoded column / row by 2 (a red arm).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_GGUF_TILED_H
#define PN_GGUF_TILED_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Elements along a tile edge. */
#define PN_TILED_DIM 32

/** 1 for a PN_QUANT_TILED_* type. */
int pn_tiled_is_type(pn_gguf_quant type);

/** Bytes of one 32x32 tile of `type` (1088 / 576 / 832), or 0 for a non-tiled type. */
size_t pn_tiled_tile_bytes(pn_gguf_quant type);

/** The PN_QUANT_TILED_* type for an FTTW (format, layout) pair, or 0 for one not modelled. */
pn_gguf_quant pn_tiled_type_for(uint32_t fttw_format, uint32_t fttw_layout);

/**
 * Decode 32 consecutive output rows of a TRANSPOSED weight into `panel` [32 x cols],
 * row-major (one output feature per row) — the rows a matvec dots against its input.
 *
 * @param column_base  the tile column's first byte (data + r0 * row_stride, r0 = 32nt)
 * @param scratch      caller-owned, 32 * cols floats (every tile of the column at once)
 * @return 0, or -1 for a type that is not a transposed tiled type
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int pn_tiled_decode_column(pn_gguf_quant type, const void *column_base, int cols, float *panel,
                           float *scratch);

/**
 * One-row matvec of a tile column: out[n] = pn_dot_f32(decoded row n, vec, cols) for the 32 rows
 * of the column, bit for bit, without the [32 x cols] panel. NEON hosts, bfp8_b / bfp4_b only.
 *
 * @param column_base  as pn_tiled_decode_column
 * @param out          32 floats (rows past a short last column are padding; callers drop them)
 * @return 0, or -1 when this host/type has no fused path (use pn_tiled_decode_column + dots)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int pn_tiled_dot_column(pn_gguf_quant type, const void *column_base, int cols, const float *vec, float *out);

/**
 * Decode one row of a tiled tensor ([rows x cols] logical) into `dst` [cols]. For the
 * embedding (ROWS layout) this is the token's row; for a transposed weight it gathers the
 * row's column out of each of its tile column's Kt tiles.
 *
 * @param data  the tensor's first byte
 * @return 0, or -1 for a non-tiled type or a `cols` that is not whole tiles
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int pn_tiled_decode_row(pn_gguf_quant type, const void *data, size_t row, size_t cols, float *dst);

#ifdef __cplusplus
}
#endif

#endif /* PN_GGUF_TILED_H */
