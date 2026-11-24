/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_prefill_lend.h
 * @brief L1a PREFILL ROLE TABLE, attention half:
 *        the decode GDN family's 16 cores sit idle for a whole prompt (role census), so for a long
 *        prompt they are lent to deep-queued prefill attention and given back before decode.
 *
 * Begin: retire each decode GDN residency (by asking; its carried state is in DRAM), then arm the
 * attention program on its core as an extra deep-attention member (pn_tt_attention_grid_arm).
 * End: retire and park those members, then boot and start each GDN residency on its own core
 * again. A core whose attention member did not exit is never booted (pn_tt_lend_ledger.h).
 * Exact: the same attention descriptors on more members (the deep queue's schedule), the same
 * GDN residencies after.
 *
 * And the decode matmul members 1-15 (96-110, idle in prefill: the grid serves the projections)
 * are parked and lent to attention too (`PN_TT_PREFILL_LEND_MEMBERS`, default 1): matmul firmware
 * booted back and the decode group re-armed whole at the end (pn_tt_matmul_members_lend/return).
 *
 * L1a.3 (`PN_TT_PREFILL_LEND_GRID=1`, default off): one lent core also completes a 64-core grid
 * with the decode matmul members 1-15 (pn_tt_matmul_grid_extend), given back at the end with the
 * decode group re-armed whole. Exact on card, but no faster at 7,680 rows (SSM pass 90.7 vs 87.4
 * ms): every grid core reads all of A, so more cores add A traffic (spec §4b).
 *
 * `PN_TT_PREFILL_LEND=<rows>`: lend for prompts of at least <rows> batched rows (0 off); unset, a
 * dense-FFN model (Qwen3.8) lends from 16,384 rows and MoE never (pn_tt_lend_min_rows). Deep
 * attention follows (`PN_TT_ATTN_DEEP` unset = on exactly for a prompt with lent cores).
 * `PN_TT_PREFILL_LEND_CORES=N` caps the cores lent; `PN_TT_PREFILL_LEND_GRID=1` extends the grid.
 * Card (p150, Qwen3.8): 8k and 32k prefill bit-identical to main, decode unchanged; 32k
 * TTFT 134.2 -> 113.4 s, 8k 25.06 -> 25.49 s.
 *
 * @status L1a.5, default on for dense-FFN qwen35 prompts of 16,384+ rows.
 * @issues Re-booting cores is the standing wedge hazard (pn_tt_matmul_group.c pn_tt_grid_park).
 * @todo None
 */
#ifndef PN_TT_PREFILL_LEND_H
#define PN_TT_PREFILL_LEND_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/**
 * Lend the decode GDN cores to prefill attention for this prompt, when it qualifies.
 * @param rows      the prompt's batched rows
 * @param min_rows  the row floor (pn_tt_lend_min_rows; 0 never lends)
 * @param deep      deep-queued attention may run for this prompt
 * @return attention members borrowed (0: nothing lent, the prompt runs as before)
 */
uint32_t pn_tt_prefill_lend_begin(pn_tt_device *device, uint32_t rows, uint32_t min_rows,
                                  bool deep);

/** Are cores lent right now (C, which borrows into the same set, then stays off)? */
bool pn_tt_prefill_lend_active(const pn_tt_device *device);

/** Give every lent core back to its decode residency. Safe to call when nothing is lent.
 *  @return 0, or -1 when a member did not exit or a residency did not restart (named) */
int pn_tt_prefill_lend_end(pn_tt_device *device);

#endif /* USE_TT */

#endif /* PN_TT_PREFILL_LEND_H */
