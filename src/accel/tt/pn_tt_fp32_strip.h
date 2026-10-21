/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_fp32_strip.h
 * @brief A `[1, 32*tile_count]` activation strip of **Float32** tiles — what a program
 *        placed at a 4096-byte page reads and writes, where `pn_tt_activation` is the
 *        2048-byte bf16 one.
 *
 * ---------------------------------------------------------------------------
 * WHY A SECOND STRIP TYPE RATHER THAN A FLAG ON THE FIRST
 * ---------------------------------------------------------------------------
 * `pn_tt_activation` is bf16 in three separate places — the allocation size, the row-0
 * chunk geometry (`ft_tt_tile_row0.h` reads 64 bytes and discards 32) and the operand's
 * `tile_bytes` — and EVERY op in the backend but one takes it. Adding a page size to it
 * would put a runtime branch on the shipped decode path to serve one probe and one
 * residency, and a strip whose format is a field is a strip that can be handed to the wrong
 * program with no refusal anywhere. Two types cannot be confused by the compiler.
 *
 * **In fp32 logical row 0 is two EXACTLY 64-byte pieces**, which is the DRAM read
 * granularity — the bf16 path has to read 64 and discard 32. So this strip is cheaper per
 * element to move than the bf16 one, not more expensive.
 *
 * The address arithmetic is `pn_tt_gdn_row0_offset()`. It is named for the family that
 * needed it first, but it is the generic fp32-tile row-0 layout and it is the one piece of
 * this file that is separately proven (`tests/test_gdn_layout.c`).
 *
 * @status Used by the per-head reduce residency (`pn_tt_head_norm.c`) and by the GDN family,
 *         which delegates its own row-0 mover here.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_FP32_STRIP_H
#define PN_TT_FP32_STRIP_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "tt/ft_tt_op_dispatch.h"

/** One strip of consecutive Float32 tiles in card DRAM. */
typedef struct pn_tt_fp32_strip {
    uint32_t bank;       /**< logical DRAM bank */
    uint32_t address;    /**< byte offset within that bank */
    uint32_t tile_count; /**< features / 32 */
} pn_tt_fp32_strip;

/**
 * Carve a `tile_count`-tile Float32 strip and stage it ZEROED.
 *
 * The zeroing is the precondition every row-0 write depends on and none of them can check:
 * only row 0 is ever written, so rows 1-31 have to start zero and stay zero for a whole-tile
 * reduce over them to contribute nothing.
 *
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_fp32_strip_alloc(pn_tt_device *device, uint32_t tile_count,
                           pn_tt_fp32_strip *out_strip);

/** Write `count` floats into logical row 0 of a strip, with no conversion at all. */
int pn_tt_fp32_strip_upload(pn_tt_device *device, const pn_tt_fp32_strip *strip,
                            const float *values, uint32_t count);

/** Read logical row 0 of a strip back as `count` floats. */
int pn_tt_fp32_strip_download(pn_tt_device *device, const pn_tt_fp32_strip *strip,
                              float *out_values, uint32_t count);

/** An operand descriptor for a Float32 strip: its endpoint and the 4096-byte page. */
ft_tt_op_operand pn_tt_fp32_strip_operand(const pn_tt_device *device,
                                          const pn_tt_fp32_strip *strip);

/** The same for an ordinary bf16 activation strip: its endpoint and the 2048-byte page.
 *
 *  It lives beside the fp32 one because the two are read TOGETHER now — the strip bridge and
 *  the per-head norms take raw operands so one body can serve both pages, and a caller
 *  outside `src/accel/tt/` needs to name both sides of the seam. */
ft_tt_op_operand pn_tt_operand(const pn_tt_device *device, const pn_tt_activation *activation);

/**
 * Move logical row 0 of `tiles` consecutive fp32 tiles at a DRAM address, in either
 * direction. The primitive both this file's upload/download and the GDN stager use.
 *
 * `values` is `tiles * 32` floats and the two pieces of each row are ADJACENT in it, so no
 * host staging buffer is needed in either direction.
 *
 * @param reading  true reads the card into `values`, false writes `values` to the card
 * @return         0, or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_fp32_row0_move(pn_tt_device *device, uint32_t bank, uint32_t address, uint32_t tiles,
                         float *values, bool reading);

/**
 * The same move for ANY logical row of the tiles. `pn_tt_fp32_row0_move()` is `row = 0`.
 *
 * Only the chunk phase-isolation arm uses a non-zero row: the decode GDN family works in row
 * 0 and the chunkwise family works in rows 0..C-1 of the same buffers, so isolating one of
 * the two chunk kernels means shuttling row `t` to row 0 and the answer back. Every move is
 * an exact fp32 copy — the arm changes WHICH KERNEL runs and nothing about the arithmetic.
 *
 * @param row  0..31
 * @return     0, or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_fp32_row_move(pn_tt_device *device, uint32_t bank, uint32_t address, uint32_t row,
                        uint32_t tiles, float *values, bool reading);

#endif /* USE_TT */
#endif /* PN_TT_FP32_STRIP_H */
