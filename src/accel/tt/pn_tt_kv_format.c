/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_format.c
 * @brief The on-card KV cache type knob (K and V formats) and the link-side tile
 *        conversions. See pn_tt_kv_format.h.
 * @status Opt-in (default bf16).
 * @issues None
 * @todo None
 */
#include "pn_tt_kv_format.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/kernels/ft_kv_block_float.h"

/** Every `PN_TT_KV_TYPE` value, its K and V formats and the program that reads it. */
static const struct {
    const char *name;
    pn_tt_kv_type type;
    const char *program;
} k_kv_types[] = {
    { "bf16", { PN_TT_KV_FORMAT_BF16, PN_TT_KV_FORMAT_BF16, false }, "attention" },
    { "bfp8", { PN_TT_KV_FORMAT_BFP8, PN_TT_KV_FORMAT_BFP8, false }, "attention_bfp8" },
    { "bfp4", { PN_TT_KV_FORMAT_BFP4, PN_TT_KV_FORMAT_BFP4, false }, "attention_bfp4" },
    { "k8v4", { PN_TT_KV_FORMAT_BFP8, PN_TT_KV_FORMAT_BFP4, false }, "attention_k8v4" },
    { "k16v4", { PN_TT_KV_FORMAT_BF16, PN_TT_KV_FORMAT_BFP4, false }, "attention_k16v4" },
    { "k4v16", { PN_TT_KV_FORMAT_BFP4, PN_TT_KV_FORMAT_BF16, false }, "attention_k4v16" },
    { "k16v8", { PN_TT_KV_FORMAT_BF16, PN_TT_KV_FORMAT_BFP8, false }, "attention_k16v8" },
    /* K4H: the Bfp4 cache with Q and K rotated (pn_tt_kv_rotate.h); the program is Bfp4's. */
    { "k4h", { PN_TT_KV_FORMAT_BFP4, PN_TT_KV_FORMAT_BFP4, true }, "attention_bfp4" },
};
#define KV_TYPE_COUNT (sizeof(k_kv_types) / sizeof(k_kv_types[0]))

static int kv_type_index(pn_tt_kv_type type)
{
    for (size_t i = 0; i < KV_TYPE_COUNT; i++)
        if (k_kv_types[i].type.k_format == type.k_format &&
            k_kv_types[i].type.v_format == type.v_format &&
            k_kv_types[i].type.rotate == type.rotate)
            return (int)i;
    return -1;
}

bool pn_tt_kv_type_parse(const char *value, pn_tt_kv_type *out)
{
    *out = k_kv_types[0].type;
    if (!value || !*value)
        return true;
    for (size_t i = 0; i < KV_TYPE_COUNT; i++)
        if (strcmp(value, k_kv_types[i].name) == 0) {
            *out = k_kv_types[i].type;
            return true;
        }
    return false;
}

pn_tt_kv_type pn_tt_kv_type_requested(void)
{
    static bool warned;
    const char *value = getenv(PN_TT_KV_TYPE_ENV);
    pn_tt_kv_type type;

    if (!pn_tt_kv_type_parse(value, &type) && !warned) {
        fprintf(stderr, "pn_tt: %s=%s is not bf16, bfp8, bfp4, k8v4, k16v4, k4v16, k16v8 or "
                        "k4h — the KV cache stays bf16\n", PN_TT_KV_TYPE_ENV, value);
        warned = true;
    }
    return type;
}

bool pn_tt_kv_type_long(pn_tt_kv_type *out)
{
    const char *value = getenv(PN_TT_KV_LONG_ENV);
    pn_tt_kv_type type;

    if (!out || !value || !*value || !pn_tt_kv_type_parse(value, &type) ||
        (type.k_format == PN_TT_KV_FORMAT_BF16 && type.v_format == PN_TT_KV_FORMAT_BF16))
        return false;
    *out = type;
    return true;
}

const char *pn_tt_kv_type_name(pn_tt_kv_type type)
{
    const int i = kv_type_index(type);

    return i < 0 ? "?" : k_kv_types[i].name;
}

const char *pn_tt_kv_type_attention_program(pn_tt_kv_type type)
{
    const int i = kv_type_index(type);

    return i < 0 ? NULL : k_kv_types[i].program;
}

uint32_t pn_tt_kv_format_tile_bytes(uint32_t format)
{
    const uint32_t bytes = ft_kv_bf_tile_bytes(format);

    return bytes ? bytes : (uint32_t)kKvBfBf16TileBytes;
}

const char *pn_tt_kv_format_name(uint32_t format)
{
    return format == PN_TT_KV_FORMAT_BFP8 ? "bfp8"
           : format == PN_TT_KV_FORMAT_BFP4 ? "bfp4"
                                            : "bf16";
}

void pn_tt_kv_format_tiles_to_bf16(uint32_t format, const uint8_t *packed, size_t tiles,
                                   uint16_t *bf16_out)
{
    const uint32_t packed_bytes = pn_tt_kv_format_tile_bytes(format);

    for (size_t tile = 0; tile < tiles; tile++) {
        const uint8_t *source = packed + tile * packed_bytes;
        uint16_t *destination = bf16_out + tile * 1024u;

        for (uint32_t face = 0; face < 4u; face++) {
            for (uint32_t face_row = 0; face_row < 16u; face_row++) {
                uint32_t bits[16];

                ft_kv_bf_unpack_row(format, source, face, face_row, bits);
                /* Exact: a datum has at most 6 fraction bits below its leading one. */
                for (uint32_t i = 0; i < 16u; i++)
                    destination[face * 256u + face_row * 16u + i] = (uint16_t)(bits[i] >> 16);
            }
        }
    }
}

void pn_tt_kv_format_tiles_from_bf16(uint32_t format, const uint16_t *bf16, size_t tiles,
                                     uint8_t *packed_out)
{
    const uint32_t packed_bytes = pn_tt_kv_format_tile_bytes(format);

    for (size_t tile = 0; tile < tiles; tile++)
        ft_kv_bf_pack_tile_rows(format, bf16 + tile * 1024u, packed_out + tile * packed_bytes,
                                0u, 32u);
}
