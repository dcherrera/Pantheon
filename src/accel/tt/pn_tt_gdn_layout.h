/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_layout.h
 * @brief The GDN stager's ADDRESS ARITHMETIC, with no device in it — the four conversions
 *        between a host float array and the DRAM layouts
 *        `foundry/tt/kernels/resident_gdn_protocol.h` settles.
 *
 *        Separate from `pn_tt_gdn.c` for one reason: these are the functions a wrong answer
 *        hides in, and split out they are provable with no card in the machine
 *        (`tests/test_gdn_layout.c`). Everything left in `pn_tt_gdn.c` is allocation,
 *        descriptor encoding and NoC calls — things a test cannot reach anyway.
 *
 * ---------------------------------------------------------------------------
 * WHAT "TILIZED" MEANS HERE, ONCE
 * ---------------------------------------------------------------------------
 * What `ft_tt_tilize_strip()` produces: `ceil(rows/32)` row blocks in order, each
 * `ceil(cols/32)` FACE-ORDERED 32x32 tiles in order, so element `[r][c]` lands at tile
 * `(r/32)*ceil(cols/32) + c/32`, position `[r%32][c%32]`. Every operand of this family is
 * fp32, so a tile is 4096 bytes.
 *
 * @status Pure C, no device, no allocation beyond what the caller hands in.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GDN_LAYOUT_H
#define PN_TT_GDN_LAYOUT_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

/** Bytes of one fp32 tile, and the elements along one of its edges. */
#define PN_TT_GDN_LAYOUT_TILE_BYTES 4096u
#define PN_TT_GDN_LAYOUT_TILE_DIM   32u

/** The two 64-byte pieces logical row 0 of an fp32 tile is split across — columns 0-15 at the
 *  top of face 0, columns 16-31 at the top of face 1. */
#define PN_TT_GDN_LAYOUT_ROW0_CHUNKS      2u
#define PN_TT_GDN_LAYOUT_ROW0_CHUNK_ELEMS 16u
#define PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES (PN_TT_GDN_LAYOUT_ROW0_CHUNK_ELEMS * 4u)

/**
 * Byte offset, within a strip of consecutive fp32 tiles, of one of logical row 0's two
 * pieces.
 *
 * The ONE place that layout is expressed, and separate for the reason `ft_tt_tile_row0.h`
 * gives about its own: the two mutations that matter — a wrong face stride, and the two
 * chunks swapped — are wrong values out of exactly this function.
 *
 * **In fp32 both pieces are exactly 64 bytes, which is the DRAM READ alignment.** The bf16
 * row-0 path has to read 64 and discard 32; this one wastes nothing, which is why the conv
 * stream crosses the link as row 0 rather than as whole tiles.
 *
 * @param tile   which tile of the strip
 * @param chunk  0 for columns 0-15, 1 for columns 16-31
 * @return       byte offset from the strip's base
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_gdn_row0_offset(uint32_t tile, uint32_t chunk);

/**
 * The same offset for ANY logical row of the tile, not just row 0.
 *
 * An fp32 tile is four FACES of 16x16, face-ordered: face `(row/16)*2 + (col/16)` sits at
 * `face * TILE_BYTES/4`, and inside a face row `row%16` sits at `(row%16) * 64`. Row 0 is the
 * `row = 0` case and nothing else, so `pn_tt_gdn_row0_offset()` is written in terms of THIS
 * function and the layout is still expressed once.
 *
 * **IT EXISTS FOR THE CHUNK PHASE-ISOLATION ARM AND NOTHING ELSE.** The decode GDN family
 * reads and writes logical row 0; the chunkwise family reads and writes rows 0..C-1 of the
 * SAME buffers. Attributing a divergence to one of the two chunk kernels means running the
 * OTHER phase per position across a chunk's rows, which needs row `t` moved to row 0 and the
 * answer moved back — exact fp32 copies, so the arm adds no arithmetic of its own.
 *
 * @param tile   which tile of the strip
 * @param row    0..31; anything else is clamped by the caller, not here
 * @param chunk  0 for columns 0-15, 1 for columns 16-31
 * @return       byte offset from the strip's base
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_gdn_row_offset(uint32_t tile, uint32_t row, uint32_t chunk);

/**
 * Gather `conv_dim` channels at STRIDE `stride` starting at `first`, and tilize them as a
 * ONE-ROW strip of `conv_dim/32` fp32 tiles.
 *
 * **THE CONV WEIGHT'S TAP PLANE `t` IS `(first = t, stride = taps)`, AND THAT IS THE WHOLE
 * TRANSPOSE.** The oracle stores the weight row-major `[conv_dim, taps]` — `w[c*taps + t]` —
 * and the card wants `taps` planes, plane `t` holding `w[.][t]` laid out exactly like the qkv
 * stream. The plausible wrong answer is a RESHAPE, `(first = t*conv_dim, stride = 1)`: the
 * same bytes, the same length, a different answer, and nothing on the card can see the
 * difference. Spelling both as two arguments of one function is what makes the difference a
 * thing a test can state.
 *
 * One row is `valid_rows = 1`: rows 1-31 are the tile quantum and the tilizer zeroes them,
 * which is correct because the compute thread's whole-tile multiply against a zero stream row
 * is zero and `SiLU(0) = 0`.
 *
 * @param source    at least `first + (conv_dim-1)*stride + 1` floats
 * @param scratch   `conv_dim` floats of caller-owned scratch
 * @param tiles_out `conv_dim/32 * 4096` bytes
 * @return          0, or -1 on a bad argument
 *
 * Thread-safe: yes (caller owns both buffers)    Reentrant: yes    Signal-safe: no
 */
int pn_tt_gdn_stage_plane(const float *source, uint32_t first, uint32_t stride,
                          uint32_t conv_dim, float *scratch, unsigned char *tiles_out);

/**
 * One tile column of one tap plane of the CHUNKWISE family's conv constant: `w[c][tap]` for
 * the column's 32 channels at EVERY row (ROW-REPLICATED), or at row 0 alone when `row0_only`.
 *
 * Row 0 is the decode family's plane (`pn_tt_gdn_stage_plane()`) byte for byte: both gather
 * `w[c*taps + tap]` and both tilize fp32 with `ft_tt_tilize_strip()`. That identity is what
 * lets the decode CONV read this constant instead of its own copy (pn_tt_gdn_conv_share.h).
 *
 * @param weight    the oracle's row-major `[conv_dim, taps]`
 * @param plane     `32*32` floats of caller-owned scratch
 * @param tile_out  4096 bytes
 * @return          0, or -1 on a bad argument
 *
 * Thread-safe: yes (caller owns both buffers)    Reentrant: yes    Signal-safe: no
 */
int pn_tt_gdnc_stage_weight_column(const float *weight, uint32_t taps, uint32_t tap,
                                   uint32_t column, bool row0_only, float *plane,
                                   unsigned char *tile_out);

/**
 * The first tile of head `head`'s `state` values inside a HEAD-MAJOR ROW-0 STRIP.
 *
 * `q`, `k`, `v` at `kGdnActivationIn` and `y` at `kGdnActivationOut` are all row-0 strips now:
 * a head's `state` values are `state/32` CONSECUTIVE tiles, all of them in row 0, exactly the
 * geometry `pn_tt_gdn_row0_offset()` addresses and exactly the geometry PHASE CONV's output
 * already has. They used to be tilized `[heads, state]` matrices with one row per head, which
 * is 32x denser in DRAM and forced the host to de-tilize, L2-normalise and re-tilize between
 * the two phases — 49,408 B a layer over PCIe.
 *
 * There is no `pn_tt_gdn_stage_stream()` any more because there is nothing left to stage: a
 * row-0 strip is written and read by `pn_tt_fp32_row0_move()`, the same primitive the conv
 * stream and the per-head reduce residency's strips already use.
 */
uint32_t pn_tt_gdn_head_tile(uint32_t head, uint32_t state);

#endif /* USE_TT */
#endif /* PN_TT_GDN_LAYOUT_H */
