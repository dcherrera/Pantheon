/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_rows.c
 * @brief Pure A-side decisions for the resident decode matmul (see pn_tt_matmul_rows.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_matmul_rows.h"

uint32_t pn_tt_a_reuse_ring(int env_set, uint32_t env_tiles, uint32_t preferred)
{
    if (env_set)
        return env_tiles;
    return preferred != 0u ? preferred : PN_TT_A_REUSE_DEFAULT_TILES;
}

bool pn_tt_a_reuse_fires(uint32_t ring, uint32_t tile_inner, uint32_t row_block)
{
    return ring != 0u && row_block == 1u && ring == tile_inner;
}
