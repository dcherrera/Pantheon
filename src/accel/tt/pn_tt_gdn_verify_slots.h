/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_verify_slots.h
 * @brief Which GDN snapshot slot each row of a speculative verify writes, and which slot a
 *        rollback reads. Pure: no device, so the host tests drive it directly.
 *
 * A verify runs rows 0..rows-1 (the seed, then the drafts). S_r is the carried state BEFORE
 * row r: S_0 is the state the verify starts from, S_rows the state after the last row. A
 * rollback that keeps rows 0..j needs S_(j+1), for j in 0..rows-2 (j = rows-1 keeps every row
 * and needs nothing). S_0 is therefore read once, by row 0, and never again.
 *
 * LEGACY (rows slots): row r writes slot r and the two trade places, so afterwards slot r holds
 * S_r and the live region S_rows. Slot 0 ends holding S_0, which nothing reads.
 *
 * COMPACT (rows - 1 slots): the LAST row writes into the buffer that held S_0 instead of a
 * slot of its own. Afterwards the live region holds S_rows and slot (r mod (rows - 1)) holds
 * S_r for r in 1..rows-1. Why the last row and not row 0 (writing row 0 in place): the
 * multi-row CONV reader (foundry resident_gdn_reader.c read_one_conv_rows) reads the LIVE
 * window planes again for rows 1..taps-2, so a row-0 write in place would be read back by
 * them. Every read of S_0's buffer, on every walk, is issued (and barrier-completed) before the
 * last row's compute starts, so the last row's write follows them all.
 *
 * - Per-row walk: row r writes pn_gdn_verify_row_slot() (-1 = in place, no trade: a one-row
 *   compact verify, the plain decode descriptor) and then trades with the live region.
 * - Multi-row descriptor: row r writes pn_gdn_verify_rows_out_slot() (-1 = the live buffer);
 *   COMPACT does no trades afterwards, LEGACY trades every row in order.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GDN_VERIFY_SLOTS_H
#define PN_TT_GDN_VERIFY_SLOTS_H

#include <stdbool.h>
#include <stdint.h>

/** Snapshot slots a verify of `rows` rows needs (rows, or rows - 1 when compact). */
uint32_t pn_gdn_verify_slot_count(bool compact, uint32_t rows);

/** Per-row walk: the slot row `row` writes before it trades with the live region, or -1 when
 *  it writes in place and trades with nothing. */
int pn_gdn_verify_row_slot(bool compact, uint32_t rows, uint32_t row);

/** Multi-row descriptor: the slot row `row`'s state goes to, or -1 for the live buffer. */
int pn_gdn_verify_rows_out_slot(bool compact, uint32_t rows, uint32_t row);

/** Does a multi-row descriptor trade each row's slot with the live region afterwards? */
bool pn_gdn_verify_rows_trade(bool compact);

/** The slot holding the state after row `last_kept` (0 <= last_kept <= rows - 2) once a verify
 *  of `rows` rows has run. */
uint32_t pn_gdn_verify_rollback_slot(bool compact, uint32_t rows, uint32_t last_kept);

#endif /* PN_TT_GDN_VERIFY_SLOTS_H */
