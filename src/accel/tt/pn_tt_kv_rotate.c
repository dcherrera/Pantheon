/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_rotate.c
 * @brief K4H's Hadamard rotation of Q and K: the host transform, H as a matmul weight, the
 *        strip view and the card call. See pn_tt_kv_rotate.h.
 * @status Opt-in (`PN_TT_KV_TYPE=k4h`).
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_kv_rotate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_matmul_group.h"
#include "tt/convert/ft_bfp.h"
#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_tilize.h"

#define ROT_TILE 32u
#define ROT_TILE_ELEMENTS (ROT_TILE * ROT_TILE)

static bool rot_dim_ok(uint32_t dim)
{
    return dim >= 2u && (dim & (dim - 1u)) == 0u;
}

float pn_tt_kv_rotate_entry(uint32_t dim, uint32_t i, uint32_t j)
{
    if (!rot_dim_ok(dim) || i >= dim || j >= dim)
        return 0.0f;
    /* RED `kv_rotate_h_sign`: one entry of row 0 flips, so H is no longer orthogonal and q.k
     * moves for any row with weight there — the host test and a card run must both see it. */
    if (i == 0u && j == 1u && pn_tt_red("kv_rotate_h_sign"))
        return (float)(-1.0 / sqrt((double)dim));
    return (float)(((__builtin_popcount(i & j) & 1u) ? -1.0 : 1.0) / sqrt((double)dim));
}

int pn_tt_kv_rotate_rows(float *rows, size_t row_count, uint32_t dim)
{
    double *work;
    const double scale = 1.0 / sqrt((double)dim);

    if (!rot_dim_ok(dim) || (!rows && row_count))
        return -1;
    work = malloc((size_t)dim * sizeof(*work));
    if (!work)
        return -1;
    for (size_t r = 0; r < row_count; r++) {
        float *row = rows + r * dim;

        for (uint32_t i = 0; i < dim; i++)
            work[i] = row[i];
        /* The butterfly in Sylvester order: after the pass at stride h, element i holds the
         * sum over j of (-1)^popcount(i & j & (2h-1)) x_j for the bits below 2h. */
        for (uint32_t h = 1u; h < dim; h <<= 1) {
            for (uint32_t i = 0; i < dim; i += h << 1) {
                for (uint32_t j = i; j < i + h; j++) {
                    const double a = work[j], b = work[j + h];

                    work[j] = a + b;
                    work[j + h] = a - b;
                }
            }
        }
        for (uint32_t i = 0; i < dim; i++)
            row[i] = (float)(work[i] * scale);
    }
    free(work);
    return 0;
}

int pn_tt_kv_rotate_weight_tiles(uint32_t dim, uint32_t tile_bytes, uint8_t *out)
{
    const uint32_t tiles = dim / ROT_TILE;
    float block[ROT_TILE_ELEMENTS];

    if (!out || !rot_dim_ok(dim) || dim < ROT_TILE)
        return -1;
    if (tile_bytes != 2048u && tile_bytes != ft_bfp_tile_bytes(FT_BFP_BFP8_B) &&
        tile_bytes != ft_bfp_tile_bytes(FT_BFP_BFP4_B))
        return -1;
    for (uint32_t n = 0; n < tiles; n++) {
        for (uint32_t k = 0; k < tiles; k++) {
            /* N-MAJOR: column panel n holds its Kt tiles contiguously (ft_tt_weight_convert.h
             * FT_TT_LAYOUT_MATMUL_NMAJOR), as the reader walks it at B_N_MAJOR = 1. */
            uint8_t *tile = out + ((size_t)n * tiles + k) * tile_bytes;

            for (uint32_t r = 0; r < ROT_TILE; r++)
                for (uint32_t c = 0; c < ROT_TILE; c++)
                    block[r * ROT_TILE + c] =
                        pn_tt_kv_rotate_entry(dim, k * ROT_TILE + r, n * ROT_TILE + c);
            if (tile_bytes == 2048u) {
                uint16_t *face = (uint16_t *)tile;

                for (uint32_t r = 0; r < ROT_TILE; r++)
                    for (uint32_t c = 0; c < ROT_TILE; c++)
                        face[ft_tt_tile_face_index(r, c)] =
                            ft_bf16_from_float(block[r * ROT_TILE + c]);
            } else if (ft_bfp_pack_tiles(tile_bytes == ft_bfp_tile_bytes(FT_BFP_BFP8_B)
                                             ? FT_BFP_BFP8_B
                                             : FT_BFP_BFP4_B,
                                         block, 1u, tile) != FT_BFP_OK) {
                return -1;
            }
        }
    }
    return 0;
}

int pn_tt_kv_rotate_view(const pn_tt_activation *strip, uint32_t heads, uint32_t head_tiles,
                         pn_tt_activation *view)
{
    if (!strip || !view || heads == 0u || head_tiles == 0u ||
        strip->tile_count < heads * head_tiles)
        return -1;
    memset(view, 0, sizeof(*view));
    view->bank = strip->bank;
    view->address = strip->address;
    view->tile_count = head_tiles;
    view->replica_count = 1u;
    view->replica_bank[0] = strip->bank;
    view->replica_address[0] = strip->address;
    /* TILE-ROW-MAJOR IS THE WHOLE CLAIM: the reader takes A tile (m, k) at
     * `(m * tile_inner + k) * 2048` and the writer puts C tile (m, n) at `(m * Nt + n) * 2048`
     * (pn_tt_device.h, the second tile-row), which for Kt = Nt = head_tiles is head m's tile k
     * of the strip, in place. */
    view->tile_rows = heads;
    view->active_rows = heads;
    view->row_offset = 0u;
    return 0;
}

bool pn_tt_kv_rotating(const pn_tt_device *device)
{
    return device && device->kv_type.rotate && device->kv_rotate_armed;
}

int pn_tt_kv_rotate_arm(pn_tt_device *device)
{
    const pn_tt_matmul_group *group;
    const uint32_t dim = device ? device->attn_head_dim : 0u;
    const uint32_t tiles = dim / ROT_TILE;
    uint32_t tile_bytes, members, first = 0u;
    uint64_t column_bytes;
    uint8_t *packed;
    pn_tt_weight *w;

    if (!device)
        return -1;
    if (!device->kv_type.rotate || device->kv_rotate_armed)
        return 0;
    group = &device->group;
    tile_bytes = device->resident_tile_bytes;
    if (dim != PN_TT_KV_ROTATE_DIM || !device->resident_running || tile_bytes == 0u) {
        fprintf(stderr, "pn_tt: REFUSED — KV type k4h needs head_dim %u (got %u) and the "
                        "matmul residency up (%s)\n", PN_TT_KV_ROTATE_DIM, dim,
                device->resident_running ? "up" : "down");
        return -1;
    }
    column_bytes = (uint64_t)tiles * tile_bytes;
    packed = malloc((size_t)(tiles * column_bytes));
    if (!packed || pn_tt_kv_rotate_weight_tiles(dim, tile_bytes, packed) != 0) {
        free(packed);
        fprintf(stderr, "pn_tt: REFUSED — no %u-byte Hadamard tiles for k4h\n", tile_bytes);
        return -1;
    }

    /* SHARDED BY COLUMN over the group, one panel a member, so the Mt = 24 Q rotation is not
     * 1,536 tile products on member 0. Each member reads the whole A view; the panels
     * concatenate along the output row exactly as an artifact weight's do. */
    w = &device->kv_rotate_weight;
    memset(w, 0, sizeof(*w));
    w->tile_rows = tiles;
    w->tile_cols = tiles;
    w->tile_bytes = tile_bytes;
    w->in_features = dim;
    w->out_features = dim;
    members = group->member_count < tiles ? group->member_count : tiles;
    if (members == 0u)
        members = 1u;
    for (uint32_t m = 0; m < members; m++) {
        const uint32_t cols = tiles / members + (m < tiles % members ? 1u : 0u);
        ft_tt_buffer buffer;

        if (pn_tt_dram_alloc_any(device, (size_t)(cols * column_bytes), &buffer) != 0 ||
            pn_tt_write_dram(device, buffer.logical_bank, (uint32_t)buffer.address,
                             packed + (size_t)first * column_bytes,
                             (size_t)(cols * column_bytes)) != 0) {
            free(packed);
            fprintf(stderr, "pn_tt: REFUSED — the k4h Hadamard panel %u did not place\n", m);
            return -1;
        }
        if (m == 0u) {
            w->bank = buffer.logical_bank;
            w->address = (uint32_t)buffer.address;
        }
        w->shard_tile_cols[m] = cols;
        w->shard_column_first[m] = first;
        w->shard_address[m] = (uint32_t)buffer.address;
        w->shard_x[m] = (uint16_t)device->dram[buffer.logical_bank].x;
        w->shard_y[m] = (uint16_t)device->dram[buffer.logical_bank].y;
        w->shard_bank[m] = buffer.logical_bank;
        first += cols;
        device->bytes_written += cols * column_bytes;
    }
    w->shard_count = members;
    free(packed);
    device->kv_rotate_armed = true;
    fprintf(stderr, "pn_tt: k4h — Q and K Hadamard-rotated after RoPE (%ux%u, %u-byte tiles, "
                    "%u members)\n", dim, dim, tile_bytes, members);
    return 0;
}

int pn_tt_kv_rotate(pn_tt_device *device, const pn_tt_activation *input,
                    const pn_tt_activation *output, uint32_t heads)
{
    pn_tt_activation in_view, out_view;
    const uint32_t head_tiles = device ? device->attn_dim_tiles : 0u;

    if (!pn_tt_kv_rotating(device) || !input || !output || input == output ||
        (input->bank == output->bank && input->address == output->address)) {
        fprintf(stderr, "pn_tt: k4h rotation refused (not armed, or in place)\n");
        return -1;
    }
    if (pn_tt_kv_rotate_view(input, heads, head_tiles, &in_view) != 0 ||
        pn_tt_kv_rotate_view(output, heads, head_tiles, &out_view) != 0) {
        fprintf(stderr, "pn_tt: k4h rotation of %u heads x %u tiles does not fit the strips "
                        "(%u, %u tiles)\n", heads, head_tiles, input->tile_count,
                output->tile_count);
        return -1;
    }
    /* THE ONE-SHOT ESCAPE IS Mt = 1 (pn_tt_matvec_dispatch), so it would rotate head 0 alone
     * and leave a plausible wrong answer. This goes to the residency or refuses the token. */
    if (pn_tt_matvec_flush(device) != 0 || !device->resident_running) {
        fprintf(stderr, "pn_tt: k4h rotation refused — the matmul residency is not up\n");
        return -1;
    }
    if (pn_tt_matmul_group_dispatch(device, &device->kv_rotate_weight, &in_view, &out_view) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: k4h rotation (Mt=%u Kt=%u Nt=%u) refused by the matmul "
                        "residency\n", heads, head_tiles, head_tiles);
        return -1;
    }
    device->matvec_count++;
    return 0;
}

#endif /* USE_TT */
