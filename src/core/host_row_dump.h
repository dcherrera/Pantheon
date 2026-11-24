/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file host_row_dump.h
 * @brief Dump real dense decode rows with their sampler inputs and the drawn token
 *        (`PN_HOST_ROW_DUMP=<file>`), for offline exactness and speed tests of the sampler.
 *
 * Record (little-endian): "PNRD" u32 version=1, i32 n_vocab, f32 temperature, f32 top_p,
 * i32 top_k, f32 repeat_penalty, i32 repeat_last_n, f32 presence, f32 frequency,
 * i32 rng_seeded, u64 rng_state (before the draw), i32 masked, i32 token (drawn),
 * i32 n_recent, u32 recent[n_recent], f32 row[n_vocab] (as the sampler received it).
 * `PN_HOST_ROW_DUMP_EVERY` (default 8) picks every Nth candidate row, `PN_HOST_ROW_DUMP_MAX`
 * (default 64) caps the count. With `PN_HOST_ROW_DUMP_GATE=<path>` rows are taken only
 * while that file exists. Single caller thread (the batched driver).
 *
 * @status Instrument only; off unless PN_HOST_ROW_DUMP is set.
 * @issues None
 * @todo None
 */
#ifndef PN_HOST_ROW_DUMP_H
#define PN_HOST_ROW_DUMP_H

#include <stdint.h>
#include "../../include/pantheon.h"

/** True when PN_HOST_ROW_DUMP names a file that opened. */
int pn_hrd_enabled(void);

/** Snapshot this row and its sampler inputs when it is picked; returns 1 when it was. */
int pn_hrd_begin(const float *row, int n_vocab, const pn_sampling_params *sp,
                 const uint32_t *recent, int n_recent, int masked);

/** Write the snapshot taken by the last pn_hrd_begin that returned 1, with its token. */
void pn_hrd_end(uint32_t token);

#endif /* PN_HOST_ROW_DUMP_H */
