/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdngate.h
 * @brief The GDN GATE family host half: the Gated DeltaNet's per-position decay and beta
 * computed ON THE CARD from the alpha/beta projection strips, as the gdnc DELTA reader's
 * COMPACT gate blocks, so the prefill never reads alpha/beta back to the host.
 *
 * WHY THE BITS ARE THE HOST'S: the card reader and this file compile the same integer-only
 * `ft_det_math.h` (foundry/tt/kernels), through the same block builder
 * (`ftgg_build_block()` in resident_gdngate_protocol.h).
 *
 * THE SHAPE. One member per core of `PN_TT_GDNGATE_CORE_LIST` (default 121,126,127,128 — the
 * cores no family holds). Each SSM layer owns a gate array of
 * 2 * `max_bands` blocks in the gdnc DELTA constant bank (the DELTA descriptor reads its gate
 * at the constant word's xy); block (band, sub) is at `layer_base + (2 * band + sub) * stride`.
 * A layer's bands are dealt to the members in contiguous ranges, issued WITHOUT awaiting, and
 * settled once before the layer's band loop (a puller AWAIT when streamed).
 *
 * @status Written PM 7 (2026-10-08). Unexecuted.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_PN_TT_GDNGATE_H
#define PANTHEON_PN_TT_GDNGATE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** Arm the members, upload every layer's constants (`ssm_a`, `ssm_dt` for each SSM layer, row
 *  `layer` = 32 + 32 fp32), and allocate the gate arrays for up to `max_rows` positions a
 *  pass. `scale_bits` is the host's 1/sqrt(S) word (header word 0). @return 0, or -1 (the
 *  family is then simply not used: the host gate path stays). */
int pn_tt_gdngate_arm(pn_tt_device *device, uint32_t layers, uint32_t heads,
                      const float *const *ssm_a, const float *const *ssm_dt, uint32_t max_rows,
                      uint32_t scale_bits);

/** Is the family armed and wanted (`PN_TT_GDNGATE=0` off)? */
bool pn_tt_gdngate_enabled(const pn_tt_device *device);

/** Issue layer `layer`'s blocks for rows [0, rows) of the alpha/beta strips (any seek; band 0
 *  is computed), with the DELTA sub-chunk length `sub_rows`. Not awaited. */
int pn_tt_gdngate_issue(pn_tt_device *device, uint32_t layer, const pn_tt_activation *alpha,
                        const pn_tt_activation *beta, uint32_t rows, uint32_t sub_rows);

/** The same for bands [first_band, first_band + band_count) of a `rows`-row layer — the W2
 *  wavefront issues the next layer's gates super-band by super-band. */
int pn_tt_gdngate_issue_bands(pn_tt_device *device, uint32_t layer,
                              const pn_tt_activation *alpha, const pn_tt_activation *beta,
                              uint32_t rows, uint32_t sub_rows, uint32_t first_band,
                              uint32_t band_count);

/** Await every issued descriptor (streamed: puller AWAITs). A refused member is re-armed and
 *  its descriptor replayed (the op is a pure function of strips nobody has rewritten yet). */
int pn_tt_gdngate_settle(pn_tt_device *device);

/** ITEM 6: the DECODE gate layout for rows [0, rows) of band 0 of the strips (flat
 *  decay[H] | beta[H] a row at `row_stride`), written at `dst_bank`/`dst_address` — where the
 *  gdn DELTA reads it (pn_tt_gdn_gate_target). Enqueued AND awaited. @return 0 or -1. */
int pn_tt_gdngate_flat(pn_tt_device *device, uint32_t layer, const pn_tt_activation *alpha,
                       const pn_tt_activation *beta, uint32_t rows, uint32_t dst_bank,
                       uint32_t dst_address, uint32_t row_stride);

/** The DRAM address of block (band, sub) of `layer`, in the gdnc DELTA constant bank. */
uint32_t pn_tt_gdngate_block_address(const pn_tt_device *device, uint32_t layer, uint32_t band,
                                     uint32_t sub);

/** VERIFY (`PN_TT_GDNGATE_VERIFY=1`, gates only): read layer `layer`'s blocks for `rows` back
 *  and compare them word for word with the host build of the same block from host-read
 *  alpha/beta. @return mismatching words, or -1 on a read failure. */
long pn_tt_gdngate_verify_layer(pn_tt_device *device, uint32_t layer,
                                const pn_tt_activation *alpha, const pn_tt_activation *beta,
                                uint32_t rows, uint32_t sub_rows);

/** THE CARD == HOST SWEEP (`PN_TT_GDNGATE_SWEEP=<rotations>`, gates only): synthetic strips
 *  holding every bf16 bit pattern as alpha and as beta, rotated over the head columns, through
 *  the family with every layer's real constants, each block compared with the host build.
 *  @return mismatching words, or -1. Leaves the DRAM it used allocated (a gate arm). */
long pn_tt_gdngate_sweep(pn_tt_device *device, uint32_t rotations);

/** One stderr line: issues (and how many at a layer start), settles and their wait, FLATs. */
void pn_tt_gdngate_report(const pn_tt_device *device);

/** Retire and release every member (device close). */
void pn_tt_gdngate_release(pn_tt_device *device);

#endif /* USE_TT */

#endif /* PANTHEON_PN_TT_GDNGATE_H */
