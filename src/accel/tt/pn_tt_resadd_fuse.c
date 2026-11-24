/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_resadd_fuse.c
 * @brief The S3a residual-add knob and descriptor words (pn_tt_resadd_fuse.h).
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_resadd_fuse.h"

#include "pn_tt_device_internal.h"

#include "tt/kernels/resident_matmul_resadd_protocol.h"

int pn_tt_fuse_resadd_mode(void)
{
    static int mode = -1;

    if (mode < 0)
        mode = pn_tt_env_u32(PN_TT_FUSE_RESADD_ENV, 0u) != 0u ? 1 : 0;
    return mode;
}

uint32_t pn_tt_resadd_block_bits(bool red_skip)
{
    return FT_TT_MATMUL_RESADD_FLAG | (red_skip ? FT_TT_MATMUL_RESADD_RED_SKIP : 0u);
}

bool pn_tt_resadd_strips_pair(const pn_tt_activation *residual, const pn_tt_activation *output,
                              uint32_t columns)
{
    if (!residual || !output || residual == output || columns == 0u)
        return false;
    if (residual->replica_count > 1u || output->replica_count > 1u)
        return false;
    if (residual->tile_count < columns || output->tile_count < columns)
        return false;
    return !(residual->bank == output->bank && residual->address == output->address);
}

void pn_tt_resadd_words(const pn_tt_activation *residual, uint32_t column_first, uint32_t x,
                        uint32_t y, uint32_t *a1, uint32_t *a2)
{
    *a1 = residual->address + ftra_residual_offset(column_first, PN_TT_TILE_BF16);
    *a2 = ftra_residual_xy(x, y);
}

#endif /* USE_TT */
