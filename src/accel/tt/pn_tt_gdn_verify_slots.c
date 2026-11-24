/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_verify_slots.c
 * @brief The verify's snapshot-slot map, legacy and compact. See pn_tt_gdn_verify_slots.h.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_gdn_verify_slots.h"

#include <stdlib.h>
#include <string.h>

/** RED `gdn_compact_rollback_slot` (host and card, compact only): the rollback reads the slot
 *  one row early — must DIFFER from 2 drafts up. */
static bool verify_slots_red(void)
{
    const char *red = getenv("PN_TT_RED");

    return red && strcmp(red, "gdn_compact_rollback_slot") == 0;
}

uint32_t pn_gdn_verify_slot_count(bool compact, uint32_t rows)
{
    if (!compact)
        return rows;
    return rows > 0u ? rows - 1u : 0u;
}

int pn_gdn_verify_row_slot(bool compact, uint32_t rows, uint32_t row)
{
    if (!compact)
        return (int)row;
    if (rows < 2u)
        return -1;
    /* The last row writes S_0's buffer: after row 0's trade that is slot 0. */
    return row + 1u < rows ? (int)row : 0;
}

int pn_gdn_verify_rows_out_slot(bool compact, uint32_t rows, uint32_t row)
{
    if (!compact)
        return (int)row;
    if (row + 1u >= rows)
        return -1;
    /* Row r's state is S_(r+1): straight to the slot the per-row walk leaves it in. */
    return (int)((row + 1u) % (rows - 1u));
}

bool pn_gdn_verify_rows_trade(bool compact)
{
    return !compact;
}

uint32_t pn_gdn_verify_rollback_slot(bool compact, uint32_t rows, uint32_t last_kept)
{
    if (!compact || rows < 2u)
        return last_kept + 1u;
    return (verify_slots_red() ? last_kept : last_kept + 1u) % (rows - 1u);
}
