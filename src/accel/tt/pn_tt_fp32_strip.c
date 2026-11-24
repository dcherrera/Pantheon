/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_fp32_strip.c
 * @brief Implementation of the Float32 activation strip. See pn_tt_fp32_strip.h for why it
 *        is a second type and not a flag on `pn_tt_activation`.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_fp32_strip.h"

#include <stdio.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_gdn_layout.h"

int pn_tt_fp32_row0_move(pn_tt_device *device, uint32_t bank, uint32_t address, uint32_t tiles,
                         float *values, bool reading)
{
    return pn_tt_fp32_row_move(device, bank, address, 0u, tiles, values, reading);
}

int pn_tt_fp32_row_move(pn_tt_device *device, uint32_t bank, uint32_t address, uint32_t row,
                        uint32_t tiles, float *values, bool reading)
{
    const ft_tt_noc_access *bulk;
    uint16_t x;
    uint16_t y;
    uint32_t tile;
    uint32_t chunk;

    if (!device || !values || bank >= device->dram_bank_count ||
        row >= PN_TT_GDN_LAYOUT_TILE_DIM)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    bulk = &device->runtime.bulk_dram;
    x = (uint16_t)device->dram[bank].x;
    y = (uint16_t)device->dram[bank].y;
    for (tile = 0; tile < tiles; tile++) {
        for (chunk = 0; chunk < PN_TT_GDN_LAYOUT_ROW0_CHUNKS; chunk++) {
            float *host = values + (size_t)tile * PN_TT_TILE_DIM +
                          (size_t)chunk * PN_TT_GDN_LAYOUT_ROW0_CHUNK_ELEMS;
            const uint64_t at = (uint64_t)address + pn_tt_gdn_row_offset(tile, row, chunk);
            const int rc =
                reading ? bulk->read(bulk->user, x, y, at, host,
                                     PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES)
                        : bulk->write(bulk->user, x, y, at, host,
                                      PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES);

            if (rc != FT_TT_OK)
                return -1;
        }
    }
    if (reading)
        device->bytes_read +=
            (size_t)tiles * PN_TT_GDN_LAYOUT_ROW0_CHUNKS * PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES;
    else
        device->bytes_written +=
            (size_t)tiles * PN_TT_GDN_LAYOUT_ROW0_CHUNKS * PN_TT_GDN_LAYOUT_ROW0_CHUNK_BYTES;
    return 0;
}

int pn_tt_fp32_strip_alloc(pn_tt_device *device, uint32_t tile_count,
                           pn_tt_fp32_strip *out_strip)
{
    const size_t bytes = (size_t)tile_count * PN_TT_TILE_FP32;
    ft_tt_buffer buffer;

    if (!device || !out_strip || tile_count == 0u || device->dram_bank_count == 0u)
        return -1;
    if (pn_tt_dram_alloc_any(device, bytes, &buffer) != 0) {
        fprintf(stderr, "pn_tt: DRAM full allocating a %u-tile Float32 strip\n", tile_count);
        return -1;
    }
    out_strip->bank = buffer.logical_bank;
    out_strip->address = (uint32_t)buffer.address;
    out_strip->tile_count = tile_count;
    /* THE PRECONDITION every row-0 write depends on and none of them can check. */
    if (pn_tt_zero_dram(device, out_strip->bank, out_strip->address, bytes) != 0) {
        fprintf(stderr, "pn_tt: could not stage a %u-tile Float32 strip zeroed\n", tile_count);
        return -1;
    }
    return 0;
}

int pn_tt_fp32_strip_upload(pn_tt_device *device, const pn_tt_fp32_strip *strip,
                            const float *values, uint32_t count)
{
    if (!strip || !values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (count / PN_TT_TILE_DIM > strip->tile_count)
        return -1;
    /* The mover never writes through `values` on the outbound leg; the cast is the price of
     * one function serving both directions. */
    return pn_tt_fp32_row0_move(device, strip->bank, strip->address,
                                count / PN_TT_TILE_DIM, (float *)values, false);
}

int pn_tt_fp32_strip_download(pn_tt_device *device, const pn_tt_fp32_strip *strip,
                              float *out_values, uint32_t count)
{
    if (!strip || !out_values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (count / PN_TT_TILE_DIM > strip->tile_count)
        return -1;
    return pn_tt_fp32_row0_move(device, strip->bank, strip->address,
                                count / PN_TT_TILE_DIM, out_values, true);
}

ft_tt_op_operand pn_tt_fp32_strip_operand(const pn_tt_device *device,
                                          const pn_tt_fp32_strip *strip)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[strip->bank].x;
    operand.y = (uint16_t)device->dram[strip->bank].y;
    operand.address = strip->address;
    operand.tile_bytes = PN_TT_TILE_FP32;
    return operand;
}

#endif /* USE_TT */
