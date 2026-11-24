/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_resadd_fuse.h
 * @brief S3a, the residual ADD folded into the out-projection and FFN down members: the knob
 *        and the descriptor words (foundry tt/kernels/resident_matmul_resadd_protocol.h).
 *
 * `PN_TT_FUSE_RESADD=1` arms the fused resident program (`rtriad_bfp8_swiglu`, which carries
 * the RESADD code too) and turns a decode / verify-row `ssm_out` / `attn_output` / `ffn_down`
 * projection into `project` plus the eltwise `hidden' = hidden + project` into one member
 * descriptor each that writes hidden + bf16(input . W) straight into the spare residual strip.
 * 0 (the default) is today's path. A shape the fused descriptor cannot serve is refused before
 * anything is written and runs unfused.
 *
 * Exactness: the fused compute packs the projection bf16 into its panel CB (the same pack of the
 * same DST the unfused matmul writes to `project`), then runs eltwise ADD's own verbs on
 * (residual tile, panel tile) with the residual as SrcA, as pn_tt_add(hidden, project) does,
 * and packs bf16: both rounding points, in order (foundry tests/test_resadd_fusion.c).
 *
 * RED (PN_TT_RED): `fuse_resadd_skip` sets word-15 bit 28, so the kernel skips the add and the
 * residual stream loses every layer's input (must DIFFER).
 *
 * @status Added with S3a (q38dec-p5-s3). Card gate not run.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_RESADD_FUSE_H
#define PN_TT_RESADD_FUSE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** `PN_TT_FUSE_RESADD`: 0 today's path (default), 1 fuse decode and verify-row residual adds. */
#define PN_TT_FUSE_RESADD_ENV "PN_TT_FUSE_RESADD"

/**
 * The knob, read once per process.
 *
 * @return 0 off, 1 on
 *
 * Thread-safe: no (first call caches)    Reentrant: no    Signal-safe: no
 */
int pn_tt_fuse_resadd_mode(void);

/**
 * The word-15 bits of a RESADD descriptor (ORed over the block word).
 *
 * @param red_skip  RED fuse_resadd_skip: the kernel skips the add
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_resadd_block_bits(bool red_skip);

/**
 * True when `residual` can feed a RESADD into `output`: both unmirrored (the writer writes one
 * destination), both at least `columns` tiles wide, and distinct strips.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_resadd_strips_pair(const pn_tt_activation *residual, const pn_tt_activation *output,
                              uint32_t columns);

/**
 * Descriptor a1/a2 of a member whose columns start at `column_first`: the residual strip's
 * address at that column and its bank coordinate (`x`, `y` = the bank's NoC door).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_resadd_words(const pn_tt_activation *residual, uint32_t column_first, uint32_t x,
                        uint32_t y, uint32_t *a1, uint32_t *a2);

#endif /* USE_TT */
#endif /* PN_TT_RESADD_FUSE_H */
