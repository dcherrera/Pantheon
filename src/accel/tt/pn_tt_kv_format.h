/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_format.h
 * @brief The on-card KV cache's element formats: bf16 (default), Bfp8_b or Bfp4_b, for K and
 *        V together or each its own (`PN_TT_KV_TYPE=bf16|bfp8|bfp4|k8v4|k16v4|k4v16|k16v8|k4h`);
 *        `k4h` is Bfp4_b K and V with Q and K Hadamard-rotated after RoPE (pn_tt_kv_rotate.h).
 *        Block float frees card DRAM: a position of one attention layer costs 4,096 B in bf16,
 *        2,176 B in Bfp8_b, 1,152 B in Bfp4_b, 1,664 B as K Bfp8_b / V Bfp4_b at Qwen3.8's 4 KV
 *        heads x 256.
 *
 * The append packs each new face row on the card (foundry ft_tt_op_memory.h §1.6; the K and V
 * appends each pack to their own region's format), attention runs the matching program
 * (`attention_bfp8`, `attention_k8v4`, ...), and export / import convert at the link. Opt-in:
 * the default stays bf16 and the bf16 programs are untouched.
 *
 * @status Opt-in (default bf16).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_KV_FORMAT_H
#define PN_TT_KV_FORMAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PN_TT_KV_TYPE_ENV "PN_TT_KV_TYPE"
/** The KV type a profile allows at a context its own type does not fit (catalog
 *  options.kv_long, set by Local for the load); the TT fit policy decides whether to use it. */
#define PN_TT_KV_LONG_ENV "PN_TT_KV_LONG"

/** Wire values of foundry's FT_TT_KV_FORMAT_* / kKvFormat*. */
#define PN_TT_KV_FORMAT_BF16 0u
#define PN_TT_KV_FORMAT_BFP8 1u
#define PN_TT_KV_FORMAT_BFP4 2u

/** A KV cache type: K's format and V's (PN_TT_KV_FORMAT_*), equal unless mixed; `rotate`
 *  stores K (and reads Q) Hadamard-rotated (pn_tt_kv_rotate.h). */
typedef struct pn_tt_kv_type {
    uint32_t k_format;
    uint32_t v_format;
    bool rotate;
} pn_tt_kv_type;

/** The type `PN_TT_KV_TYPE` asks for; unset or empty is bf16. An unknown value is bf16 and
 *  says so on stderr, once. */
pn_tt_kv_type pn_tt_kv_type_requested(void);

/** The long-context type `PN_TT_KV_LONG` allows: true and `*out` for a known block-float
 *  type; false (`*out` untouched) when unset, bf16 or unknown. */
bool pn_tt_kv_type_long(pn_tt_kv_type *out);

/** The type a `PN_TT_KV_TYPE` value names; false (and *out bf16) for an unknown value. */
bool pn_tt_kv_type_parse(const char *value, pn_tt_kv_type *out);

/** "bf16", "bfp8", "bfp4", "k8v4", "k16v4", "k4v16", "k16v8", "k4h"; "?" for a type no knob
 *  value names. */
const char *pn_tt_kv_type_name(pn_tt_kv_type type);

/** The attention program that reads a cache of `type`; NULL for a pair with no program. */
const char *pn_tt_kv_type_attention_program(pn_tt_kv_type type);

/** Bytes of one 32x32 tile in `format`: 2048, 1088, 576. */
uint32_t pn_tt_kv_format_tile_bytes(uint32_t format);

/** "bf16", "bfp8", "bfp4". */
const char *pn_tt_kv_format_name(uint32_t format);


/** Packed cache tiles -> face-ordered bf16 tiles (2048 B each), `tiles` of them. */
void pn_tt_kv_format_tiles_to_bf16(uint32_t format, const uint8_t *packed, size_t tiles,
                                   uint16_t *bf16_out);

/** Face-ordered bf16 tiles -> packed cache tiles, the bytes the card's append would write. */
void pn_tt_kv_format_tiles_from_bf16(uint32_t format, const uint16_t *bf16, size_t tiles,
                                     uint8_t *packed_out);

#endif /* PN_TT_KV_FORMAT_H */
