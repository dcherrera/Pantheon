/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_dequant.h
 * @brief  GGUF tensor block layouts, dequantization to float32, and size helpers.
 *
 * Rewritten from a behaviour specification and public GGUF format documentation;
 * the implementer did not see the previous code.
 *
 * @status None
 * @issues - pn_dequantize returns -1 for numel == 0 and silently skips a partial trailing
 *           block (returns 0, tail untouched); pn_gguf_dequantize_tensor then hands that
 *           tail back uninitialised.
 *         - An unsupported type through pn_gguf_dequantize_tensor prints two lines.
 *         - Q4_1/Q5_0/Q5_1/Q8_1/Q2_K/Q8_K have a block size but 0 bytes per block, so
 *           they size as 0 (deliberately unsizeable).
 *         - pn_float_to_half canonicalises NaNs; pn_half_to_float keeps NaN payloads.
 *         - pn_gguf_quant_name returns a literal "" (not buf) for a NULL/empty buffer.
 *         - pn_gguf_dequantize_row has no bounds check; its offset multiply can wrap.
 * @todo   None
 */
#ifndef PN_GGUF_DEQUANT_H
#define PN_GGUF_DEQUANT_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

/* ---------------------------------------------------------------------------
 * Block layouts (ABI: field order and sizes are relied on across the tree).
 * `d` / `dmin` are IEEE binary16 bit patterns.
 * ------------------------------------------------------------------------- */

/** Q4_0: 32 elements, 4-bit codes offset by 8. */
typedef struct {
    uint16_t d;
    uint8_t  qs[16];
} pn_block_q4_0;

/** Q8_0: 32 elements, signed 8-bit codes. */
typedef struct {
    uint16_t d;
    int8_t   qs[32];
} pn_block_q8_0;

/** Q2_0_g128: 128 elements, 2-bit codes (0..3 -> -1..+2), low bits first. */
typedef struct {
    uint16_t d;
    uint8_t  qs[32];
} pn_block_q2_0_g128;

/** Q3_K: 256 elements, 2 low bits + 1 high bit per element, 16 six-bit scales. */
typedef struct {
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
} pn_block_q3_k;

/** Q4_K: 256 elements, 4-bit codes, 8 sub-blocks with 6-bit scale and min. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} pn_block_q4_k;

/** Q5_K: as Q4_K plus one high bit per element in qh. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
} pn_block_q5_k;

/** Q6_K: 256 elements, 4 low + 2 high bits per element, 16 signed 8-bit scales. */
typedef struct {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
} pn_block_q6_k;

_Static_assert(sizeof(pn_block_q4_0) == 18, "pn_block_q4_0 must be 18 bytes");
_Static_assert(sizeof(pn_block_q8_0) == 34, "pn_block_q8_0 must be 34 bytes");
_Static_assert(sizeof(pn_block_q2_0_g128) == 34, "pn_block_q2_0_g128 must be 34 bytes");
_Static_assert(sizeof(pn_block_q3_k) == 110, "pn_block_q3_k must be 110 bytes");
_Static_assert(sizeof(pn_block_q4_k) == 144, "pn_block_q4_k must be 144 bytes");
_Static_assert(sizeof(pn_block_q5_k) == 176, "pn_block_q5_k must be 176 bytes");
_Static_assert(sizeof(pn_block_q6_k) == 210, "pn_block_q6_k must be 210 bytes");

/** Buffer size that holds any pn_gguf_quant_name result. */
#define PN_GGUF_QUANT_NAME_MAX 16

/**
 * Dequantize `numel` elements of `type` from `src` into `dst`.
 * F32/F16/BF16 convert every element. Block types convert floor(numel / block) whole
 * blocks; a partial trailing block is left untouched in `dst`.
 * Returns 0 on success; -1 (silently) if src/dst is NULL or numel is 0; -1 with a
 * "pn_dequantize: unsupported quant type N" line on stderr for any other type.
 */
int pn_dequantize(const void *src, float *dst, size_t numel, pn_gguf_quant type);

/** Dequantize one block at `block` (any alignment) into `out`. No checks. */
void pn_dequantize_block_q4_0(const void *block, float *out);      /* 32 floats  */
void pn_dequantize_block_q8_0(const void *block, float *out);      /* 32 floats  */
void pn_dequantize_block_q2_0_g128(const void *block, float *out); /* 128 floats */
void pn_dequantize_block_q3_k(const void *block, float *out);      /* 256 floats */
void pn_dequantize_block_q4_k(const void *block, float *out);      /* 256 floats */
void pn_dequantize_block_q5_k(const void *block, float *out);      /* 256 floats */
void pn_dequantize_block_q6_k(const void *block, float *out);      /* 256 floats */

/** Exact binary16 -> binary32 widening (subnormals normalised, NaN payload kept). */
float pn_half_to_float(uint16_t h);

/** binary32 -> binary16, round to nearest even; overflow -> Inf; NaN -> 0x7E00 | sign. */
uint16_t pn_float_to_half(float f);

/**
 * Dequantize a whole named tensor into a new malloc'd buffer (caller frees).
 * Sets *numel_out (if non-NULL) on success. Returns NULL if file/name is NULL, the
 * tensor is missing (stderr line), malloc fails, or the type is unsupported (stderr).
 */
float *pn_gguf_dequantize_tensor(pn_gguf_file *file, const char *name, size_t *numel_out);

/**
 * Dequantize row `row` (row_numel elements, block-aligned) of a named tensor into dst.
 * Returns the pn_dequantize result, or -1 on NULL args, missing tensor, a type with no
 * block size, or a misaligned row. No bounds check against the tensor dims.
 */
int pn_gguf_dequantize_row(pn_gguf_file *file, const char *name,
                           size_t row, size_t row_numel, float *dst);

/** Elements per block (1 for F32/F16/BF16); 0 for unknown types. */
int pn_gguf_block_size(pn_gguf_quant type);

/** Bytes per block; 0 for types this code cannot size (incl. Q4_1, Q2_K, ...). */
size_t pn_gguf_bytes_per_block(pn_gguf_quant type);

/** Bytes for `numel` elements (whole blocks only); 0 if the type cannot be sized. */
size_t pn_gguf_type_size(pn_gguf_quant type, size_t numel);

/**
 * Write the type's name ("Q4_K", or "type<N>" if unknown) into buf with snprintf
 * truncation and return buf. Returns the literal "" when buf is NULL or buf_len is 0.
 */
const char *pn_gguf_quant_name(pn_gguf_quant type, char *buf, size_t buf_len);

#endif /* PN_GGUF_DEQUANT_H */
