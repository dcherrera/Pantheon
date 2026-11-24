/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_dequant.c
 * @brief  Dequantization of GGUF tensor blocks to float32, half conversions, size helpers.
 *
 * Rewritten from a behaviour specification
 * (cleanroom_bpe_dequant_spec.md, recorded in docs/cleanroom/README.md) and public GGUF format documentation;
 * the implementer did not see the previous code.
 *
 * Bit-exactness under -ffast-math: every scale x quant product here is exactly
 * representable in binary32, so grouping does not change results; Q4_K/Q5_K round once
 * at the min subtraction. All math is float (no double, no explicit fma), and pure bit
 * moves (F32/BF16 copy, half<->float) are integer-only so FTZ/DAZ cannot touch them.
 *
 * @status None
 * @issues - Partial trailing blocks are skipped silently (tail of dst left untouched).
 *         - numel == 0 makes pn_dequantize fail; unsupported types via
 *           pn_gguf_dequantize_tensor print two lines.
 *         - Q4_1/Q5_0/Q5_1/Q8_1/Q2_K/Q8_K: block size known, bytes per block 0.
 *         - pn_float_to_half canonicalises NaNs (not a round trip with pn_half_to_float).
 *         - pn_gguf_dequantize_row: no bounds check, offset multiply can wrap.
 * @todo   None
 */
#include "gguf_dequant.h"
#include "gguf_tiled.h"
#include "../core/mem/pn_host_admission.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Bit helpers
 * ------------------------------------------------------------------------- */

static inline float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static inline uint32_t float_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* Little-endian 16-bit load from an arbitrarily aligned address. */
static inline uint16_t load_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

/* ---------------------------------------------------------------------------
 * Half precision
 * ------------------------------------------------------------------------- */

float pn_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;

    if (exp == 0) {
        if (man == 0) return bits_to_float(sign);
        /* Subnormal: value = man * 2^-24. Find the leading one at bit b (0..9). */
        uint32_t b = 9;
        while (!(man & (1u << b))) b--;
        uint32_t frac = (man & ((1u << b) - 1u)) << (23 - b);
        return bits_to_float(sign | ((b + 103u) << 23) | frac);
    }
    if (exp == 31) return bits_to_float(sign | 0x7F800000u | (man << 13));
    return bits_to_float(sign | ((exp + 112u) << 23) | (man << 13));
}

uint16_t pn_float_to_half(float f) {
    uint32_t x = float_to_bits(f);
    uint16_t s16 = (uint16_t)((x >> 16) & 0x8000u);
    uint32_t E = (x >> 23) & 0xFFu;
    uint32_t M = x & 0x7FFFFFu;
    int e = (int)E - 112;

    if (E == 0xFF) return (uint16_t)(s16 | 0x7C00u | (M ? 0x0200u : 0u));
    if (e >= 31) return (uint16_t)(s16 | 0x7C00u);

    if (e <= 0) {
        if (e < -10) return s16;
        uint32_t mm = M | 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);          /* 14..24 */
        uint32_t q = mm >> shift;
        uint32_t r = mm & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (r > halfway || (r == halfway && (q & 1u))) q++;
        return (uint16_t)(s16 | q);                   /* carry into 0x400 = min normal */
    }

    uint32_t q = M >> 13;
    uint32_t r = M & 0x1FFFu;
    if (r > 0x1000u || (r == 0x1000u && (q & 1u))) q++;
    if (q == 0x400u) {
        q = 0;
        e += 1;
        if (e >= 31) return (uint16_t)(s16 | 0x7C00u);
    }
    return (uint16_t)(s16 | ((uint32_t)e << 10) | q);
}

/* ---------------------------------------------------------------------------
 * Block decoders. Field offsets come from the public structs so the layouts live in
 * one place; reads go through byte pointers so any alignment is fine.
 * ------------------------------------------------------------------------- */

#define FIELD(blk, type, field) ((const uint8_t *)(blk) + offsetof(type, field))

void pn_dequantize_block_q4_0(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q4_0, d)));
    const uint8_t *qs = FIELD(block, pn_block_q4_0, qs);
    for (int j = 0; j < 16; j++) {
        int lo = (qs[j] & 0x0F) - 8;
        int hi = (qs[j] >> 4) - 8;
        out[j]      = d * (float)lo;
        out[j + 16] = d * (float)hi;
    }
}

void pn_dequantize_block_q8_0(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q8_0, d)));
    const int8_t *qs = (const int8_t *)FIELD(block, pn_block_q8_0, qs);
    for (int j = 0; j < 32; j++)
        out[j] = d * (float)qs[j];
}

void pn_dequantize_block_q2_0_g128(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q2_0_g128, d)));
    const uint8_t *qs = FIELD(block, pn_block_q2_0_g128, qs);
    for (int j = 0; j < 128; j++) {
        int code = (qs[j / 4] >> (2 * (j % 4))) & 3;
        out[j] = (float)(code - 1) * d;
    }
}

void pn_dequantize_block_q3_k(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q3_k, d)));
    const uint8_t *hmask = FIELD(block, pn_block_q3_k, hmask);
    const uint8_t *qs = FIELD(block, pn_block_q3_k, qs);
    const uint8_t *S = FIELD(block, pn_block_q3_k, scales);

    /* 16 six-bit scales: low nibbles from S[0..7], top two bits from S[8..11]. */
    int8_t sc[16];
    for (int i = 0; i < 4; i++) {
        uint8_t hi = S[8 + i];
        sc[i]      = (int8_t)(((S[i]     & 0x0F) | (((hi >> 0) & 3) << 4)) - 32);
        sc[i + 4]  = (int8_t)(((S[i + 4] & 0x0F) | (((hi >> 2) & 3) << 4)) - 32);
        sc[i + 8]  = (int8_t)(((S[i]     >> 4)   | (((hi >> 4) & 3) << 4)) - 32);
        sc[i + 12] = (int8_t)(((S[i + 4] >> 4)   | (((hi >> 6) & 3) << 4)) - 32);
    }

    for (int h = 0; h < 2; h++) {
        for (int j = 0; j < 4; j++) {
            for (int g = 0; g < 2; g++) {             /* 16-element group within l */
                const int k = 8 * h + 2 * j + g;
                const float dl = d * (float)sc[k];
                for (int l = 16 * g; l < 16 * g + 16; l++) {
                    int low2 = (qs[32 * h + l] >> (2 * j)) & 3;
                    int hbit = (hmask[l] >> (4 * h + j)) & 1;
                    int q = low2 - (hbit ? 0 : 4);
                    out[128 * h + 32 * j + l] = dl * (float)q;
                }
            }
        }
    }
}

/* 6-bit scale/min unpack shared by Q4_K and Q5_K. */
static void unpack_scale_min(const uint8_t *S, uint8_t sc[8], uint8_t mn[8]) {
    for (int k = 0; k < 4; k++) {
        sc[k] = S[k] & 63;
        mn[k] = S[k + 4] & 63;
    }
    for (int k = 4; k < 8; k++) {
        sc[k] = (uint8_t)((S[k + 4] & 0x0F) | ((S[k - 4] >> 6) << 4));
        mn[k] = (uint8_t)((S[k + 4] >> 4) | ((S[k] >> 6) << 4));
    }
}

void pn_dequantize_block_q4_k(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q4_k, d)));
    const float dm = pn_half_to_float(load_u16(FIELD(block, pn_block_q4_k, dmin)));
    const uint8_t *qs = FIELD(block, pn_block_q4_k, qs);
    uint8_t sc[8], mn[8];
    unpack_scale_min(FIELD(block, pn_block_q4_k, scales), sc, mn);

    for (int p = 0; p < 4; p++) {
        const float d0 = d * (float)sc[2 * p],     m0 = dm * (float)mn[2 * p];
        const float d1 = d * (float)sc[2 * p + 1], m1 = dm * (float)mn[2 * p + 1];
        const uint8_t *c = qs + 32 * p;
        float *o = out + 64 * p;
        for (int l = 0; l < 32; l++) {
            o[l]      = d0 * (float)(c[l] & 0xF) - m0;
            o[l + 32] = d1 * (float)(c[l] >> 4) - m1;
        }
    }
}

void pn_dequantize_block_q5_k(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q5_k, d)));
    const float dm = pn_half_to_float(load_u16(FIELD(block, pn_block_q5_k, dmin)));
    const uint8_t *qh = FIELD(block, pn_block_q5_k, qh);
    const uint8_t *qs = FIELD(block, pn_block_q5_k, qs);
    uint8_t sc[8], mn[8];
    unpack_scale_min(FIELD(block, pn_block_q5_k, scales), sc, mn);

    for (int p = 0; p < 4; p++) {
        const float d0 = d * (float)sc[2 * p],     m0 = dm * (float)mn[2 * p];
        const float d1 = d * (float)sc[2 * p + 1], m1 = dm * (float)mn[2 * p + 1];
        const uint8_t *c = qs + 32 * p;
        const uint8_t bit_lo = (uint8_t)(1u << (2 * p));
        const uint8_t bit_hi = (uint8_t)(1u << (2 * p + 1));
        float *o = out + 64 * p;
        for (int l = 0; l < 32; l++) {
            int qlo = (c[l] & 0xF) + ((qh[l] & bit_lo) ? 16 : 0);
            int qhi = (c[l] >> 4) + ((qh[l] & bit_hi) ? 16 : 0);
            o[l]      = d0 * (float)qlo - m0;
            o[l + 32] = d1 * (float)qhi - m1;
        }
    }
}

void pn_dequantize_block_q6_k(const void *block, float *out) {
    const float d = pn_half_to_float(load_u16(FIELD(block, pn_block_q6_k, d)));
    const uint8_t *ql_all = FIELD(block, pn_block_q6_k, ql);
    const uint8_t *qh_all = FIELD(block, pn_block_q6_k, qh);
    const int8_t *sc_all = (const int8_t *)FIELD(block, pn_block_q6_k, scales);

    for (int h = 0; h < 2; h++) {
        const uint8_t *QL = ql_all + 64 * h;
        const uint8_t *QH = qh_all + 32 * h;
        const int8_t *SC = sc_all + 8 * h;
        float *o = out + 128 * h;
        for (int l = 0; l < 32; l++) {
            const int t = l / 16;
            int q1 = (QL[l] & 0xF)      | (((QH[l] >> 0) & 3) << 4);
            int q2 = (QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4);
            int q3 = (QL[l] >> 4)       | (((QH[l] >> 4) & 3) << 4);
            int q4 = (QL[l + 32] >> 4)  | (((QH[l] >> 6) & 3) << 4);
            o[l]      = d * (float)SC[0 + t] * (float)(q1 - 32);
            o[l + 32] = d * (float)SC[2 + t] * (float)(q2 - 32);
            o[l + 64] = d * (float)SC[4 + t] * (float)(q3 - 32);
            o[l + 96] = d * (float)SC[6 + t] * (float)(q4 - 32);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Bulk dispatch
 * ------------------------------------------------------------------------- */

typedef void (*block_fn)(const void *block, float *out);

static void run_blocks(const void *src, float *dst, size_t numel,
                       size_t elems, size_t bytes, block_fn fn) {
    const uint8_t *p = (const uint8_t *)src;
    size_t nb = numel / elems;
    for (size_t i = 0; i < nb; i++)
        fn(p + i * bytes, dst + i * elems);
}

int pn_dequantize(const void *src, float *dst, size_t numel, pn_gguf_quant type) {
    if (src == NULL || dst == NULL || numel == 0) return -1;
    const uint8_t *bytes = (const uint8_t *)src;

    switch (type) {
    case GGUF_QUANT_F32:
        memcpy(dst, src, numel * sizeof(float));
        return 0;
    case GGUF_QUANT_F16:
        for (size_t i = 0; i < numel; i++)
            dst[i] = pn_half_to_float(load_u16(bytes + 2 * i));
        return 0;
    case GGUF_QUANT_BF16:
        for (size_t i = 0; i < numel; i++) {
            uint32_t w = (uint32_t)load_u16(bytes + 2 * i) << 16;
            memcpy(&dst[i], &w, sizeof w);
        }
        return 0;
    case GGUF_QUANT_Q4_0:
        run_blocks(src, dst, numel, 32, sizeof(pn_block_q4_0), pn_dequantize_block_q4_0);
        return 0;
    case GGUF_QUANT_Q8_0:
        run_blocks(src, dst, numel, 32, sizeof(pn_block_q8_0), pn_dequantize_block_q8_0);
        return 0;
    case GGUF_QUANT_Q2_0_G128:
        run_blocks(src, dst, numel, 128, sizeof(pn_block_q2_0_g128),
                   pn_dequantize_block_q2_0_g128);
        return 0;
    case GGUF_QUANT_Q3_K:
        run_blocks(src, dst, numel, 256, sizeof(pn_block_q3_k), pn_dequantize_block_q3_k);
        return 0;
    case GGUF_QUANT_Q4_K:
        run_blocks(src, dst, numel, 256, sizeof(pn_block_q4_k), pn_dequantize_block_q4_k);
        return 0;
    case GGUF_QUANT_Q5_K:
        run_blocks(src, dst, numel, 256, sizeof(pn_block_q5_k), pn_dequantize_block_q5_k);
        return 0;
    case GGUF_QUANT_Q6_K:
        run_blocks(src, dst, numel, 256, sizeof(pn_block_q6_k), pn_dequantize_block_q6_k);
        return 0;
    default:
        fprintf(stderr, "pn_dequantize: unsupported quant type %d\n", (int)type);
        return -1;
    }
}

/* ---------------------------------------------------------------------------
 * Size helpers and names
 * ------------------------------------------------------------------------- */

int pn_gguf_block_size(pn_gguf_quant type) {
    switch (type) {
    case GGUF_QUANT_F32:
    case GGUF_QUANT_F16:
    case GGUF_QUANT_BF16:
        return 1;
    case GGUF_QUANT_Q4_0:
    case GGUF_QUANT_Q4_1:
    case GGUF_QUANT_Q5_0:
    case GGUF_QUANT_Q5_1:
    case GGUF_QUANT_Q8_0:
    case GGUF_QUANT_Q8_1:
        return 32;
    case GGUF_QUANT_Q2_K:
    case GGUF_QUANT_Q3_K:
    case GGUF_QUANT_Q4_K:
    case GGUF_QUANT_Q5_K:
    case GGUF_QUANT_Q6_K:
    case GGUF_QUANT_Q8_K:
        return 256;
    case GGUF_QUANT_Q2_0_G128:
        return 128;
    case PN_QUANT_TILED_FP32:
    case PN_QUANT_TILED_BFP8:
    case PN_QUANT_TILED_BFP4:
    case PN_QUANT_TILED_BFP6:
    case PN_QUANT_TILED_ROWS_FP32:
    case PN_QUANT_TILED_ROWS_BFP8:
        return PN_TILED_DIM; /* a block = 32 elements of one tile (gguf_tiled.h) */
    default:
        return 0;
    }
}

size_t pn_gguf_bytes_per_block(pn_gguf_quant type) {
    switch (type) {
    case GGUF_QUANT_F32:       return 4;
    case GGUF_QUANT_F16:
    case GGUF_QUANT_BF16:      return 2;
    case GGUF_QUANT_Q4_0:      return sizeof(pn_block_q4_0);
    case GGUF_QUANT_Q8_0:      return sizeof(pn_block_q8_0);
    case GGUF_QUANT_Q2_0_G128: return sizeof(pn_block_q2_0_g128);
    case GGUF_QUANT_Q3_K:      return sizeof(pn_block_q3_k);
    case GGUF_QUANT_Q4_K:      return sizeof(pn_block_q4_k);
    case GGUF_QUANT_Q5_K:      return sizeof(pn_block_q5_k);
    case GGUF_QUANT_Q6_K:      return sizeof(pn_block_q6_k);
    case PN_QUANT_TILED_FP32:
    case PN_QUANT_TILED_BFP8:
    case PN_QUANT_TILED_BFP4:
    case PN_QUANT_TILED_BFP6:
    case PN_QUANT_TILED_ROWS_FP32:
    case PN_QUANT_TILED_ROWS_BFP8:
        return pn_tiled_tile_bytes(type) / PN_TILED_DIM; /* 128 / 34 / 18 / 26: exact */
    default:                   return 0;   /* includes the deliberately unsizeable types */
    }
}

size_t pn_gguf_type_size(pn_gguf_quant type, size_t numel) {
    int bs = pn_gguf_block_size(type);
    if (bs <= 0) return 0;
    size_t bpb = pn_gguf_bytes_per_block(type);
    if (bpb == 0) return 0;
    return (numel / (size_t)bs) * bpb;
}

const char *pn_gguf_quant_name(pn_gguf_quant type, char *buf, size_t buf_len) {
    if (buf == NULL || buf_len == 0) return "";
    const char *name = NULL;
    switch (type) {
    case GGUF_QUANT_F32:       name = "F32"; break;
    case GGUF_QUANT_F16:       name = "F16"; break;
    case GGUF_QUANT_BF16:      name = "BF16"; break;
    case GGUF_QUANT_Q4_0:      name = "Q4_0"; break;
    case GGUF_QUANT_Q4_1:      name = "Q4_1"; break;
    case GGUF_QUANT_Q5_0:      name = "Q5_0"; break;
    case GGUF_QUANT_Q5_1:      name = "Q5_1"; break;
    case GGUF_QUANT_Q8_0:      name = "Q8_0"; break;
    case GGUF_QUANT_Q8_1:      name = "Q8_1"; break;
    case GGUF_QUANT_Q2_K:      name = "Q2_K"; break;
    case GGUF_QUANT_Q3_K:      name = "Q3_K"; break;
    case GGUF_QUANT_Q4_K:      name = "Q4_K"; break;
    case GGUF_QUANT_Q5_K:      name = "Q5_K"; break;
    case GGUF_QUANT_Q6_K:      name = "Q6_K"; break;
    case GGUF_QUANT_Q8_K:      name = "Q8_K"; break;
    case GGUF_QUANT_Q2_0_G128: name = "Q2_0_G128"; break;
    case PN_QUANT_TILED_FP32:  name = "TILED_FP32"; break;
    case PN_QUANT_TILED_ROWS_FP32: name = "TILED_ROWS_FP32"; break;
    case PN_QUANT_TILED_BFP8:  name = "TILED_BFP8_B"; break;
    case PN_QUANT_TILED_BFP4:  name = "TILED_BFP4_B"; break;
    case PN_QUANT_TILED_BFP6:  name = "TILED_BFP6_B"; break;
    case PN_QUANT_TILED_ROWS_BFP8: name = "TILED_ROWS_BFP8_B"; break;
    default:                   break;
    }
    if (name != NULL)
        snprintf(buf, buf_len, "%s", name);
    else
        snprintf(buf, buf_len, "type%d", (int)type);
    return buf;
}

/* ---------------------------------------------------------------------------
 * Tensor-level helpers
 * ------------------------------------------------------------------------- */

float *pn_gguf_dequantize_tensor(pn_gguf_file *file, const char *name, size_t *numel_out) {
    if (file == NULL || name == NULL) return NULL;
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(file, name);
    if (info == NULL) {
        fprintf(stderr, "pn_gguf_dequantize_tensor: tensor '%s' not found\n", name);
        return NULL;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < info->n_dims; i++)
        numel *= (size_t)info->dims[i];

    const char *raw = (const char *)file->data + info->offset;
    /* An anonymous copy of a weight is host RAM the mmap does not cover:
     * admitted first, charged to the model being loaded (pn_load_on). */
    if (pn_host_admit_weight_copy((uint64_t)numel * sizeof(float), name) != 0)
        return NULL;
    float *out = malloc(numel * sizeof(float));
    if (out == NULL) return NULL;

    if (pn_dequantize(raw, out, numel, info->type) < 0) {
        fprintf(stderr, "pn_gguf_dequantize_tensor: unsupported type %d for '%s'\n",
                (int)info->type, name);
        free(out);
        return NULL;
    }
    if (numel_out != NULL) *numel_out = numel;
    return out;
}

int pn_gguf_dequantize_row(pn_gguf_file *file, const char *name,
                           size_t row, size_t row_numel, float *dst) {
    if (file == NULL || name == NULL || dst == NULL) return -1;
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(file, name);
    if (info == NULL) return -1;
    if (pn_tiled_is_type(info->type))
        return pn_tiled_decode_row(info->type, (const char *)file->data + info->offset, row, row_numel, dst);

    int bs = pn_gguf_block_size(info->type);
    if (bs <= 0) return -1;
    size_t off = row * row_numel;
    if (off % (size_t)bs != 0 || row_numel % (size_t)bs != 0) return -1;

    size_t byte_off = pn_gguf_type_size(info->type, off);
    const char *src = (const char *)file->data + info->offset + byte_off;
    return pn_dequantize(src, dst, row_numel, info->type);
}
