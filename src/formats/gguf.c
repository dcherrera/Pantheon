/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ============================================================
 * Cursor-based reader helpers
 * All reads are little-endian (GGUF spec).
 * ============================================================ */

typedef struct {
    const uint8_t *base;
    const uint8_t *cur;
    const uint8_t *end;
} cursor_t;

#define CHECK_BOUNDS(c, n) do { \
    if ((c)->cur + (n) > (c)->end) { \
        fprintf(stderr, "gguf: unexpected end of file at offset %zu (need %zu bytes)\n", \
                (size_t)((c)->cur - (c)->base), (size_t)(n)); \
        goto fail; \
    } \
} while(0)

static inline uint8_t read_u8(cursor_t *c) {
    uint8_t v = c->cur[0];
    c->cur += 1;
    return v;
}

static inline uint16_t read_u16(cursor_t *c) {
    uint16_t v;
    memcpy(&v, c->cur, 2);
    c->cur += 2;
    return v;
}

static inline uint32_t read_u32(cursor_t *c) {
    uint32_t v;
    memcpy(&v, c->cur, 4);
    c->cur += 4;
    return v;
}

static inline uint64_t read_u64(cursor_t *c) {
    uint64_t v;
    memcpy(&v, c->cur, 8);
    c->cur += 8;
    return v;
}

static inline int8_t read_i8(cursor_t *c) { return (int8_t)read_u8(c); }
static inline int16_t read_i16(cursor_t *c) { return (int16_t)read_u16(c); }
static inline int32_t read_i32(cursor_t *c) { return (int32_t)read_u32(c); }
static inline int64_t read_i64(cursor_t *c) { return (int64_t)read_u64(c); }

static inline float read_f32(cursor_t *c) {
    float v;
    memcpy(&v, c->cur, 4);
    c->cur += 4;
    return v;
}

static inline double read_f64(cursor_t *c) {
    double v;
    memcpy(&v, c->cur, 8);
    c->cur += 8;
    return v;
}

/* Read a GGUF string: 8-byte length + data. Returns malloc'd null-terminated copy. */
static char* read_string(cursor_t *c, uint64_t *len_out) {
    uint64_t len = read_u64(c);
    if (len_out) *len_out = len;
    if (c->cur + len > c->end) return NULL;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, c->cur, len);
    s[len] = '\0';
    c->cur += len;
    return s;
}

/* Size of a single GGUF metadata type value (excluding arrays/strings) */
static size_t gguf_type_size(pn_gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

/* Skip over an array value in the cursor (used during metadata parsing) */
static int skip_array(cursor_t *c) {
    if (c->cur + 12 > c->end) return -1;  /* type(4) + count(8) */
    pn_gguf_type elem_type = (pn_gguf_type)read_u32(c);
    uint64_t count = read_u64(c);

    if (elem_type == GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < count; i++) {
            if (c->cur + 8 > c->end) return -1;
            uint64_t slen = read_u64(c);
            if (c->cur + slen > c->end) return -1;
            c->cur += slen;
        }
    } else {
        size_t esz = gguf_type_size(elem_type);
        if (esz == 0) return -1;
        size_t total = esz * count;
        if (c->cur + total > c->end) return -1;
        c->cur += total;
    }
    return 0;
}

/* ============================================================
 * pn_gguf_open — parse GGUF header, metadata, tensor info
 * ============================================================ */

pn_gguf_file* pn_gguf_open(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gguf: cannot open %s\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void *mmap_base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_base == MAP_FAILED) {
        fprintf(stderr, "gguf: mmap failed for %s\n", path);
        close(fd);
        return NULL;
    }

    cursor_t c = {
        .base = (const uint8_t *)mmap_base,
        .cur  = (const uint8_t *)mmap_base,
        .end  = (const uint8_t *)mmap_base + file_size,
    };

    /* 1. Magic "GGUF" */
    CHECK_BOUNDS(&c, 4);
    if (memcmp(c.cur, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf: bad magic in %s\n", path);
        goto fail;
    }
    c.cur += 4;

    /* 2. Version (expect v3) */
    CHECK_BOUNDS(&c, 4);
    uint32_t version = read_u32(&c);
    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf: unsupported version %u in %s\n", version, path);
        goto fail;
    }

    /* 3. Tensor count and metadata KV count */
    CHECK_BOUNDS(&c, 16);
    uint64_t n_tensors = read_u64(&c);
    uint64_t n_kv = read_u64(&c);

    /* Allocate file handle */
    pn_gguf_file *file = calloc(1, sizeof(pn_gguf_file));
    if (!file) goto fail;

    file->version = version;
    file->n_tensors = n_tensors;
    file->n_kv = n_kv;
    file->mmap_base = mmap_base;
    file->mmap_size = file_size;
    file->fd = fd;

    /* 4. Parse metadata KV pairs */
    file->kv = calloc(n_kv, sizeof(pn_gguf_kv));
    if (!file->kv && n_kv > 0) goto fail_file;

    for (uint64_t i = 0; i < n_kv; i++) {
        /* Key string */
        CHECK_BOUNDS(&c, 8);
        char *key = read_string(&c, &file->kv[i].key.len);
        if (!key) goto fail_file;
        file->kv[i].key.data = key;

        /* Value type */
        CHECK_BOUNDS(&c, 4);
        file->kv[i].type = (pn_gguf_type)read_u32(&c);

        /* Value — read or skip based on type */
        switch (file->kv[i].type) {
            case GGUF_TYPE_UINT8:   CHECK_BOUNDS(&c, 1); file->kv[i].value.u8  = read_u8(&c); break;
            case GGUF_TYPE_INT8:    CHECK_BOUNDS(&c, 1); file->kv[i].value.i8  = read_i8(&c); break;
            case GGUF_TYPE_UINT16:  CHECK_BOUNDS(&c, 2); file->kv[i].value.u16 = read_u16(&c); break;
            case GGUF_TYPE_INT16:   CHECK_BOUNDS(&c, 2); file->kv[i].value.i16 = read_i16(&c); break;
            case GGUF_TYPE_UINT32:  CHECK_BOUNDS(&c, 4); file->kv[i].value.u32 = read_u32(&c); break;
            case GGUF_TYPE_INT32:   CHECK_BOUNDS(&c, 4); file->kv[i].value.i32 = read_i32(&c); break;
            case GGUF_TYPE_FLOAT32: CHECK_BOUNDS(&c, 4); file->kv[i].value.f32 = read_f32(&c); break;
            case GGUF_TYPE_BOOL:    CHECK_BOUNDS(&c, 1); file->kv[i].value.b   = read_u8(&c) != 0; break;
            case GGUF_TYPE_UINT64:  CHECK_BOUNDS(&c, 8); file->kv[i].value.u64 = read_u64(&c); break;
            case GGUF_TYPE_INT64:   CHECK_BOUNDS(&c, 8); file->kv[i].value.i64 = read_i64(&c); break;
            case GGUF_TYPE_FLOAT64: CHECK_BOUNDS(&c, 8); file->kv[i].value.f64 = read_f64(&c); break;
            case GGUF_TYPE_STRING: {
                CHECK_BOUNDS(&c, 8);
                char *val = read_string(&c, &file->kv[i].value.str.len);
                if (!val) goto fail_file;
                file->kv[i].value.str.data = val;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                /* Store pointer to array data for later parsing */
                const uint8_t *arr_start = c.cur;
                if (skip_array(&c) < 0) goto fail_file;
                /* Store raw array pointer in the string field (reinterpreted later) */
                file->kv[i].value.str.data = (char *)arr_start;
                file->kv[i].value.str.len = (uint64_t)(c.cur - arr_start);
                break;
            }
            default:
                fprintf(stderr, "gguf: unknown metadata type %d for key '%s'\n",
                        file->kv[i].type, key);
                goto fail_file;
        }
    }

    /* 5. Parse tensor info entries */
    file->tensors = calloc(n_tensors, sizeof(pn_gguf_tensor_info));
    if (!file->tensors && n_tensors > 0) goto fail_file;

    for (uint64_t i = 0; i < n_tensors; i++) {
        /* Name */
        CHECK_BOUNDS(&c, 8);
        char *name = read_string(&c, &file->tensors[i].name.len);
        if (!name) goto fail_file;
        file->tensors[i].name.data = name;

        /* Number of dimensions */
        CHECK_BOUNDS(&c, 4);
        file->tensors[i].n_dims = read_u32(&c);
        if (file->tensors[i].n_dims > 4) {
            fprintf(stderr, "gguf: tensor '%s' has %u dims (max 4)\n",
                    name, file->tensors[i].n_dims);
            goto fail_file;
        }

        /* Dimension sizes */
        CHECK_BOUNDS(&c, file->tensors[i].n_dims * 8);
        for (uint32_t d = 0; d < file->tensors[i].n_dims; d++) {
            file->tensors[i].dims[d] = read_u64(&c);
        }

        /* Quantization type */
        CHECK_BOUNDS(&c, 4);
        file->tensors[i].type = (pn_gguf_quant)read_u32(&c);

        /* Offset within tensor data section */
        CHECK_BOUNDS(&c, 8);
        file->tensors[i].offset = read_u64(&c);
    }

    /* 6. Calculate data section start (align to GGUF_DEFAULT_ALIGNMENT from file start) */
    size_t header_end = (size_t)(c.cur - c.base);
    size_t alignment = 32;  /* GGUF default alignment (GGUF_DEFAULT_ALIGNMENT = 32) */

    /* Check for custom alignment in metadata */
    pn_gguf_kv *align_kv = pn_gguf_find_kv(file, "general.alignment");
    if (align_kv && align_kv->type == GGUF_TYPE_UINT32) {
        alignment = align_kv->value.u32;
    }

    size_t data_offset = (header_end + alignment - 1) & ~(alignment - 1);
    file->data = (uint8_t *)mmap_base + data_offset;
    file->data_size = file_size - data_offset;

    fprintf(stderr, "gguf: opened %s (v%u, %llu tensors, %llu metadata)\n",
           path, version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    return file;

fail_file:
    pn_gguf_close(file);
    return NULL;

fail:
    munmap(mmap_base, file_size);
    close(fd);
    return NULL;
}

/* ============================================================
 * Close
 * ============================================================ */

void pn_gguf_close(pn_gguf_file *file) {
    if (!file) return;

    /* Free metadata key strings and string values */
    if (file->kv) {
        for (uint64_t i = 0; i < file->n_kv; i++) {
            free(file->kv[i].key.data);
            if (file->kv[i].type == GGUF_TYPE_STRING) {
                free(file->kv[i].value.str.data);
            }
            /* Array values point into mmap — don't free */
        }
        free(file->kv);
    }

    /* Free tensor name strings */
    if (file->tensors) {
        for (uint64_t i = 0; i < file->n_tensors; i++) {
            free(file->tensors[i].name.data);
        }
        free(file->tensors);
    }

    /* Unmap file */
    if (file->mmap_base) {
        munmap(file->mmap_base, file->mmap_size);
    }
    if (file->fd >= 0) {
        close(file->fd);
    }

    free(file);
}

/* ============================================================
 * Metadata lookup helpers
 * ============================================================ */

pn_gguf_kv* pn_gguf_find_kv(pn_gguf_file *file, const char *key) {
    if (!file || !key) return NULL;
    for (uint64_t i = 0; i < file->n_kv; i++) {
        if (file->kv[i].key.data && strcmp(file->kv[i].key.data, key) == 0) {
            return &file->kv[i];
        }
    }
    return NULL;
}

uint32_t pn_gguf_get_u32(pn_gguf_file *file, const char *key, uint32_t default_val) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.u32;
    if (kv->type == GGUF_TYPE_INT32 && kv->value.i32 >= 0) return (uint32_t)kv->value.i32;
    return default_val;
}

uint64_t pn_gguf_get_u64(pn_gguf_file *file, const char *key, uint64_t default_val) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT64) return kv->value.u64;
    if (kv->type == GGUF_TYPE_UINT32) return kv->value.u32;
    if (kv->type == GGUF_TYPE_INT32 && kv->value.i32 >= 0) return (uint64_t)kv->value.i32;
    return default_val;
}

int32_t pn_gguf_get_i32(pn_gguf_file *file, const char *key, int32_t default_val) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_INT32) return kv->value.i32;
    if (kv->type == GGUF_TYPE_UINT32 && kv->value.u32 <= INT32_MAX) return (int32_t)kv->value.u32;
    return default_val;
}

float pn_gguf_get_f32(pn_gguf_file *file, const char *key, float default_val) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv || kv->type != GGUF_TYPE_FLOAT32) return default_val;
    return kv->value.f32;
}

const char* pn_gguf_get_str(pn_gguf_file *file, const char *key) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return NULL;
    return kv->value.str.data;
}

/* ============================================================
 * Array metadata access
 * ============================================================ */

uint64_t pn_gguf_get_arr_n(pn_gguf_file *file, const char *key) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return 0;
    /* Array data pointer stored in str.data, parse element count */
    const uint8_t *p = (const uint8_t *)kv->value.str.data;
    uint64_t count;
    memcpy(&count, p + 4, 8);  /* skip type(4), read count(8) */
    return count;
}

pn_gguf_arr* pn_gguf_get_arr(pn_gguf_file *file, const char *key) {
    pn_gguf_kv *kv = pn_gguf_find_kv(file, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return NULL;

    const uint8_t *p = (const uint8_t *)kv->value.str.data;
    pn_gguf_arr *arr = malloc(sizeof(pn_gguf_arr));
    if (!arr) return NULL;

    uint32_t elem_type;
    memcpy(&elem_type, p, 4);
    memcpy(&arr->n, p + 4, 8);
    arr->elem_type = (pn_gguf_type)elem_type;
    arr->data = (void *)(p + 12);  /* past type(4) + count(8) */

    return arr;
}

const char* pn_gguf_arr_str(pn_gguf_arr *arr, uint64_t idx) {
    if (!arr || arr->elem_type != GGUF_TYPE_STRING || idx >= arr->n) return NULL;

    /* Walk string array from start to find idx-th entry */
    const uint8_t *p = (const uint8_t *)arr->data;
    for (uint64_t i = 0; i < idx; i++) {
        uint64_t slen;
        memcpy(&slen, p, 8);
        p += 8 + slen;
    }

    uint64_t slen;
    memcpy(&slen, p, 8);
    /* Strings in GGUF arrays are NOT null-terminated, but we return
     * a pointer into the mmap'd data. The caller must handle length
     * or we need to copy. For now return pointer (length available via slen). */
    /* Actually for BPE token extraction we need null-terminated copies.
     * The caller should use the length. For convenience, check if next byte is \0. */
    return (const char *)(p + 8);
}

/* ============================================================
 * Tensor lookup
 * ============================================================ */

pn_gguf_tensor_info* pn_gguf_find_tensor(pn_gguf_file *file, const char *name) {
    if (!file || !name) return NULL;
    for (uint64_t i = 0; i < file->n_tensors; i++) {
        if (file->tensors[i].name.data && strcmp(file->tensors[i].name.data, name) == 0) {
            return &file->tensors[i];
        }
    }
    return NULL;
}

void* pn_gguf_tensor_data(pn_gguf_file *file, const char *name) {
    if (!file || !file->data || !name) return NULL;
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(file, name);
    if (!info) return NULL;
    return (char *)file->data + info->offset;
}
