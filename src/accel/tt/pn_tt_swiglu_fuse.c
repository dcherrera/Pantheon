/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_swiglu_fuse.c
 * @brief The fused-SwiGLU knob, program name and descriptor words (pn_tt_swiglu_fuse.h).
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_swiglu_fuse.h"

#include "pn_tt_device_internal.h"
#include "pn_tt_resadd_fuse.h"

#include "tt/kernels/resident_matmul_swiglu_protocol.h"

int pn_tt_fuse_swiglu_mode(void)
{
    static int mode = -1;

    if (mode < 0)
        mode = pn_tt_env_u32(PN_TT_FUSE_SWIGLU_ENV, 1u) != 0u ? 1 : 0;
    return mode;
}

bool pn_tt_fused_program_armed(void)
{
    return pn_tt_fuse_swiglu_mode() || pn_tt_fuse_resadd_mode();
}

const char *pn_tt_resident_program_name(void)
{
    return pn_tt_fused_program_armed() ? PN_TT_RESIDENT_PROGRAM_SWIGLU
                                       : PN_TT_RESIDENT_PROGRAM_PLAIN;
}

bool pn_tt_eltwise_silu_row0(const pn_tt_device *device)
{
    static int row0_mode = -1;

    if (row0_mode < 0)
        row0_mode = (int)pn_tt_env_u32("PN_TT_ELTWISE_ROW0", 1u);
    return row0_mode == 2 ||
           (row0_mode == 1 && device &&
            (device->single_row || (device->verify_rows && device->verify_rows <= 4u)));
}

bool pn_tt_swiglu_shards_pair(const pn_tt_weight *gate, const pn_tt_weight *up)
{
    uint32_t member;

    if (!gate || !up || gate->shard_count < 2u || gate->shard_count != up->shard_count ||
        gate->shard_count > PN_TT_MATMUL_MAX_CORES || gate->tile_rows != up->tile_rows ||
        gate->tile_cols != up->tile_cols || gate->tile_bytes != up->tile_bytes)
        return false;
    for (member = 0u; member < gate->shard_count; member++)
        if (gate->shard_tile_cols[member] != up->shard_tile_cols[member] ||
            gate->shard_column_first[member] != up->shard_column_first[member] ||
            gate->shard_tile_cols[member] == 0u)
            return false;
    return true;
}

uint32_t pn_tt_swiglu_block_bits(bool silu_row0, bool red_no_stage)
{
    return FT_TT_MATMUL_SWIGLU_FLAG | (silu_row0 ? FT_TT_MATMUL_SWIGLU_SILU_ROW0 : 0u) |
           (red_no_stage ? FT_TT_MATMUL_SWIGLU_RED_NO_STAGE : 0u);
}

void pn_tt_swiglu_up_words(const pn_tt_weight *up, uint32_t member, uint32_t *a1,
                           uint32_t *a2)
{
    *a1 = up->shard_address[member];
    *a2 = ftsw_up_xy(up->shard_x[member], up->shard_y[member]);
}

#endif /* USE_TT */
