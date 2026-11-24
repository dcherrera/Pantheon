/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_topk_rows.h
 * @brief MULTI-ROW TOP-K (2026-10-21): the verify heads' per-row top-k as ONE descriptor per
 *        split member that reads the logit tiles once and selects for up to
 *        FT_TT_OP_SAMPLE_MAX_ROWS consecutive token rows (foundry
 *        kernels/resident_topk_select.c). Plus the split's shared host half: reading a
 *        member's result block and the exact merge in the kernel's order.
 *
 * Each row's merged answer is the answer the single-row split gives for that row: the same
 * kernel selection over the same elements in the same order, the same merge.
 *
 * @status Used by pn_qwen35_tt_head_rows (`PN_TT_TOPK_ROWS`, default on; `=0` off).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_TOPK_ROWS_H
#define PN_TT_TOPK_ROWS_H

#ifdef USE_TT

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** Pairs one top-k selects at most (FT_TT_OP_SAMPLE_MAX_SELECT). */
#define PN_TT_TOPK_SELECT_MAX 64u

/** Rows one multi-row top-k descriptor takes (FT_TT_OP_SAMPLE_MAX_ROWS). */
#define PN_TT_TOPK_ROWS_MAX 4u

/**
 * The helper area for a top-k whose result block is at `result_address` (the three TRISCs
 * scan too, foundry FT_TT_OP_SAMPLE_HELPER_AREA). `PN_TT_TOPK_HELPERS` (default 1; `=0` off:
 * NCRISC scans alone).
 * @return the L1 address, or 0 = no helpers
 */
uint32_t pn_tt_topk_helper_area(uint32_t result_address);

/** On a failed top-k: print member 0's five processor status slots (state, cursor, error,
 *  last opcode, spins) and its helper area's handshake words. Diagnostic only. */
void pn_tt_topk_dump_failure(pn_tt_device *device);

/** The kernel's order key of an fp32 that holds a bf16: larger key = larger value. */
uint32_t pn_tt_topk_order_key(float value);

/**
 * Read `bytes` of a result block at L1 `address` on core (x, y): one bulk read
 * (`PN_TT_TOPK_BULK`, default 1), else the register accessor.
 * @return FT_TT_OK or the accessor's error
 */
int pn_tt_topk_read_result(pn_tt_device *device, uint32_t x, uint32_t y, uint32_t address,
                           uint32_t *block, size_t bytes);

/**
 * Read every split member's result block at `block_offset` bytes past its result address and
 * merge them exactly: key descending, then index ascending, the first `count` kept.
 * @param slice_offset_red  RED `topk_split_no_offset`: members' indices are not shifted
 * @return 0, or -1 (a block that did not read back or does not decode)
 */
int pn_tt_topk_merge_members(pn_tt_device *device, uint32_t count, uint32_t block_offset,
                             bool slice_offset_red, uint32_t *out_indices, float *out_values,
                             uint32_t *out_count);

/**
 * Top-`count` pairs of `rows` consecutive token rows (from the device's top-k row) of
 * `logits`, one descriptor per split member. Row r's pairs land at
 * out_indices[r * stride ..], out_values[r * stride ..], its count at out_counts[r].
 * @param stride  >= count
 * @return 0; 1 when this path does not apply (fewer than 2 members, tail counts asked,
 *         rows outside 2..FT_TT_OP_SAMPLE_MAX_ROWS) — the caller runs rows one by one;
 *         -1 on a refused or failed member (the caller may retry row by row)
 */
int pn_tt_topk_rows(pn_tt_device *device, const pn_tt_activation *logits, uint32_t count,
                    uint32_t rows, uint32_t stride, uint32_t *out_indices, float *out_values,
                    uint32_t *out_counts);

#endif /* USE_TT */

#endif /* PN_TT_TOPK_ROWS_H */
