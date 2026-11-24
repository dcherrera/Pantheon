/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_GGUF_H
#define PN_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * GGUF file format parser.
 *
 * GGUF layout:
 *   1. Magic: "GGUF" (4 bytes)
 *   2. Version: uint32 (v2 or v3)
 *   3. Tensor count: uint64
 *   4. Metadata count: uint64
 *   5. Metadata entries: typed key-value pairs
 *   6. Tensor info: name, dims, quant type, offset
 *   7. Padding (32-byte aligned, overridable via general.alignment)
 *   8. Tensor data (binary weights)
 */

/* GGUF metadata value types */
typedef enum {
    GGUF_TYPE_UINT8    = 0,
    GGUF_TYPE_INT8     = 1,
    GGUF_TYPE_UINT16   = 2,
    GGUF_TYPE_INT16    = 3,
    GGUF_TYPE_UINT32   = 4,
    GGUF_TYPE_INT32    = 5,
    GGUF_TYPE_FLOAT32  = 6,
    GGUF_TYPE_BOOL     = 7,
    GGUF_TYPE_STRING   = 8,
    GGUF_TYPE_ARRAY    = 9,
    GGUF_TYPE_UINT64   = 10,
    GGUF_TYPE_INT64    = 11,
    GGUF_TYPE_FLOAT64  = 12,
} pn_gguf_type;

/* GGUF quantization types (subset — expand as needed) */
typedef enum {
    GGUF_QUANT_F32   = 0,
    GGUF_QUANT_F16   = 1,
    GGUF_QUANT_Q4_0  = 2,
    GGUF_QUANT_Q4_1  = 3,
    GGUF_QUANT_Q5_0  = 6,
    GGUF_QUANT_Q5_1  = 7,
    GGUF_QUANT_Q8_0  = 8,
    GGUF_QUANT_Q8_1  = 9,
    GGUF_QUANT_Q2_K  = 10,
    GGUF_QUANT_Q3_K  = 11,
    GGUF_QUANT_Q4_K  = 12,
    GGUF_QUANT_Q5_K  = 13,
    GGUF_QUANT_Q6_K  = 14,
    GGUF_QUANT_Q8_K  = 15,
    GGUF_QUANT_BF16  = 30,  /* 1|8|7. The format the checkpoints actually ship in: same 2 bytes/weight as F16 but the fp32 exponent range, so widening is the top-16-bits reinterpret and no weight is re-rounded. */
    GGUF_QUANT_Q2_0_G128 = 142,  /* PrismML Bonsai native ternary (g128 release): 128-elem blocks, fp16 group scale + 2-bit codes. NOTE: the shipped Q2_0_g128 pack tags tensors type 142 / general.file_type 141; the older g64 pack used id 42. The 34-byte block layout ([fp16 d][32 codes], 128 elems) is identical — only the type id differs. */
    PN_QUANT_TERNARY = 100,  /* Packed 2-bit ternary with per-group float32 scales */
    /* Foundry FTTW tiles, read through an attached artifact (gguf_fttw_overlay.h): the
     * card-format CPU oracle. NOT GGML ids: 2000 + the
     * artifact's ft_tt_tensor_format. 32x32 tiles; a "block" is 32 elements of
     * tile_bytes/32 bytes, so numel/32 * bpb is the exact size of whole tiles and a 32-row
     * group starts at data + r0 * row_stride. gguf_tiled.h has the layout. */
    PN_QUANT_TILED_FP32 = 2000,      /* N-major, fp32 face-ordered tiles (the exact-values arm) */
    PN_QUANT_TILED_BFP8 = 2002,      /* N-major: K x N tiles, column-major grid (matmul weights) */
    PN_QUANT_TILED_BFP4 = 2003,
    PN_QUANT_TILED_BFP6 = 2004,
    PN_QUANT_TILED_ROWS_FP32 = 2100, /* source layout, fp32 face-ordered tiles */
    PN_QUANT_TILED_ROWS_BFP8 = 2102, /* source layout: row-major tile grid (the embedding) */
} pn_gguf_quant;

/* GGUF string (length-prefixed) */
typedef struct {
    uint64_t len;
    char *data;
} pn_gguf_string;

/* GGUF metadata key-value pair */
typedef struct {
    pn_gguf_string key;
    pn_gguf_type type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        uint64_t  u64;
        int64_t   i64;
        float     f32;
        double    f64;
        bool      b;
        pn_gguf_string str;
    } value;
} pn_gguf_kv;

/* GGUF tensor info entry */
typedef struct {
    pn_gguf_string name;
    uint32_t n_dims;
    uint64_t dims[4];
    pn_gguf_quant type;
    uint64_t offset;        /* Offset from start of tensor data section */
} pn_gguf_tensor_info;

/* GGUF file handle */
typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    pn_gguf_kv *kv;                /* Metadata array [n_kv] */
    pn_gguf_tensor_info *tensors;  /* Tensor info array [n_tensors] */
    void *data;                     /* Pointer to tensor data section */
    size_t data_size;
    void *mmap_base;                /* mmap base pointer (for munmap) */
    size_t mmap_size;               /* mmap size */
    void *overlay_base;             /* an attached FTTW artifact's mapping, or NULL */
    size_t overlay_size;
    int fd;                         /* File descriptor */
} pn_gguf_file;

/* Get metadata uint64 value. Returns default_val if not found. */
uint64_t pn_gguf_get_u64(pn_gguf_file *file, const char *key, uint64_t default_val);

/* Get metadata int32 value. Returns default_val if not found. */
int32_t pn_gguf_get_i32(pn_gguf_file *file, const char *key, int32_t default_val);

/* Get metadata array length. Returns 0 if not found or not an array. */
uint64_t pn_gguf_get_arr_n(pn_gguf_file *file, const char *key);

/* GGUF array metadata entry */
typedef struct {
    pn_gguf_type elem_type;
    uint64_t n;
    void *data;             /* Raw pointer into mmap'd array data */
} pn_gguf_arr;

/* Get metadata array. Returns NULL if not found or not an array. */
pn_gguf_arr* pn_gguf_get_arr(pn_gguf_file *file, const char *key);

/* Get string from array at index. Returns NULL on error. */
const char* pn_gguf_arr_str(pn_gguf_arr *arr, uint64_t idx);

/* Open and parse a GGUF file (header + metadata + tensor info, mmap data) */
pn_gguf_file* pn_gguf_open(const char *path);

/* Close and free a GGUF file */
void pn_gguf_close(pn_gguf_file *file);

/* Look up a metadata value by key. Returns NULL if not found. */
pn_gguf_kv* pn_gguf_find_kv(pn_gguf_file *file, const char *key);

/* Get a metadata uint32 value. Returns default_val if not found. */
uint32_t pn_gguf_get_u32(pn_gguf_file *file, const char *key, uint32_t default_val);

/* Get a metadata float value. Returns default_val if not found. */
float pn_gguf_get_f32(pn_gguf_file *file, const char *key, float default_val);

/* Get a metadata string value. Returns NULL if not found. */
const char* pn_gguf_get_str(pn_gguf_file *file, const char *key);

/* Get pointer to tensor data by name. Returns NULL if not found. */
void* pn_gguf_tensor_data(pn_gguf_file *file, const char *name);

/* Get tensor info by name. Returns NULL if not found. */
pn_gguf_tensor_info* pn_gguf_find_tensor(pn_gguf_file *file, const char *name);

#endif /* PN_GGUF_H */
