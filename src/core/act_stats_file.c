/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats_file.c
 * @brief  Implementation of the "mixw imatrix v1" table, reader, writer and merge
 *         (format in act_stats_file.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "act_stats_file.h"

#include <stdlib.h>
#include <string.h>

/* Largest n_in either kind accepts: 2^20 channels (imatrix), 2^16 (X^T X: 32 GiB already). */
#define ACT_STATS_MAX_N_IN (1u << 20)
#define ACT_STATS_MAX_N_IN_XTX (1u << 16)
/* Longest tensor name accepted by the reader. */
#define ACT_STATS_MAX_NAME 4096u

uint64_t pn_act_stats_payload_len(int xtx, uint32_t n_in) {
    if (n_in == 0 || n_in > (xtx ? ACT_STATS_MAX_N_IN_XTX : ACT_STATS_MAX_N_IN))
        return 0;
    return xtx ? (uint64_t)n_in * n_in : n_in;
}

pn_act_stats_entry *pn_act_stats_table_find(pn_act_stats_table *t, const char *name) {
    for (uint32_t i = 0; i < t->n_entries; i++) {
        if (strcmp(t->entries[i].name, name) == 0)
            return &t->entries[i];
    }
    return NULL;
}

pn_act_stats_entry *pn_act_stats_table_append(pn_act_stats_table *t, const char *name,
                                              uint32_t n_in) {
    uint64_t len = pn_act_stats_payload_len(t->xtx, n_in);
    if (!len) {
        fprintf(stderr, "act_stats: %s: n_in %u out of range for a %s table\n", name, n_in,
                t->xtx ? "X^T X" : "imatrix");
        return NULL;
    }
    if (t->n_entries == t->cap) {
        uint32_t cap = t->cap ? 2 * t->cap : 256;
        pn_act_stats_entry *grown = realloc(t->entries, (size_t)cap * sizeof(*grown));
        if (!grown)
            return NULL;
        t->entries = grown;
        t->cap = cap;
    }
    pn_act_stats_entry *e = &t->entries[t->n_entries];
    e->name = strdup(name);
    e->payload = calloc((size_t)len, sizeof(double));
    if (!e->name || !e->payload) {
        fprintf(stderr, "act_stats: %s: cannot allocate %llu doubles\n", name,
                (unsigned long long)len);
        free(e->name);
        free(e->payload);
        return NULL;
    }
    e->n_in = n_in;
    e->count = 0;
    t->n_entries++;
    return e;
}

int pn_act_stats_table_merge(pn_act_stats_table *dst, const pn_act_stats_table *src) {
    if (dst->xtx != src->xtx) {
        fprintf(stderr, "act_stats: cannot merge an imatrix table with an X^T X table\n");
        return -1;
    }
    for (uint32_t i = 0; i < src->n_entries; i++) {
        const pn_act_stats_entry *s = &src->entries[i];
        pn_act_stats_entry *d = pn_act_stats_table_find(dst, s->name);
        if (!d && !(d = pn_act_stats_table_append(dst, s->name, s->n_in)))
            return -1;
        if (d->n_in != s->n_in) {
            fprintf(stderr, "act_stats: %s: n_in %u vs %u, refusing to merge\n", s->name, d->n_in,
                    s->n_in);
            return -1;
        }
        uint64_t len = pn_act_stats_payload_len(dst->xtx, s->n_in);
        for (uint64_t j = 0; j < len; j++)
            d->payload[j] += s->payload[j];
        d->count += s->count;
    }
    return 0;
}

/* Little-endian serialization, independent of the host's byte order. */
static int put_u32(FILE *f, uint32_t v) {
    unsigned char b[4] = { (unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16),
                           (unsigned char)(v >> 24) };
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}

static int put_u64(FILE *f, uint64_t v) {
    return put_u32(f, (uint32_t)v) || put_u32(f, (uint32_t)(v >> 32)) ? -1 : 0;
}

/* Doubles in little-endian byte order, staged through a chunk so an X^T X payload is a few
 * thousand fwrite calls, not tens of millions. */
int pn_act_stats_write_f64(FILE *f, const double *v, uint64_t n) {
    unsigned char chunk[8 * 512];
    while (n) {
        size_t k = n < 512 ? (size_t)n : 512;
        for (size_t i = 0; i < k; i++) {
            uint64_t bits;
            memcpy(&bits, &v[i], sizeof(bits));
            for (int b = 0; b < 8; b++)
                chunk[8 * i + (size_t)b] = (unsigned char)(bits >> (8 * b));
        }
        if (fwrite(chunk, 8, k, f) != k)
            return -1;
        v += k;
        n -= k;
    }
    return 0;
}

static int get_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4)
        return -1;
    *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 0;
}

static int get_u64(FILE *f, uint64_t *v) {
    uint32_t lo, hi;
    if (get_u32(f, &lo) || get_u32(f, &hi))
        return -1;
    *v = (uint64_t)hi << 32 | lo;
    return 0;
}

static int get_f64_array(FILE *f, double *v, uint64_t n) {
    unsigned char chunk[8 * 512];
    while (n) {
        size_t k = n < 512 ? (size_t)n : 512;
        if (fread(chunk, 8, k, f) != k)
            return -1;
        for (size_t i = 0; i < k; i++) {
            uint64_t bits = 0;
            for (int b = 0; b < 8; b++)
                bits |= (uint64_t)chunk[8 * i + (size_t)b] << (8 * b);
            memcpy(&v[i], &bits, sizeof(bits));
        }
        v += k;
        n -= k;
    }
    return 0;
}

int pn_act_stats_write_header(FILE *f, uint32_t n_tensors, uint64_t n_tokens) {
    return fwrite(PN_ACT_STATS_MAGIC, 1, 4, f) != 4 || put_u32(f, PN_ACT_STATS_VERSION) ||
                   put_u32(f, n_tensors) || put_u64(f, n_tokens)
               ? -1
               : 0;
}

int pn_act_stats_write_entry_head(FILE *f, const char *name, uint32_t n_in, uint64_t count) {
    const uint32_t len = (uint32_t)strlen(name);
    return put_u32(f, len) || fwrite(name, 1, len, f) != len || put_u32(f, n_in) ||
                   put_u64(f, count)
               ? -1
               : 0;
}

int pn_act_stats_table_write_stream(const pn_act_stats_table *t, FILE *f) {
    uint64_t n_tokens = 0;
    for (uint32_t i = 0; i < t->n_entries; i++) {
        if (t->entries[i].count > n_tokens)
            n_tokens = t->entries[i].count;
    }
    int bad = pn_act_stats_write_header(f, t->n_entries, n_tokens);
    for (uint32_t i = 0; i < t->n_entries && !bad; i++) {
        const pn_act_stats_entry *e = &t->entries[i];
        bad = pn_act_stats_write_entry_head(f, e->name, e->n_in, e->count) ||
              pn_act_stats_write_f64(f, e->payload, pn_act_stats_payload_len(t->xtx, e->n_in));
    }
    if (bad || fflush(f) != 0) {
        fprintf(stderr, "act_stats: write failed\n");
        return -1;
    }
    return 0;
}

int pn_act_stats_table_write(const pn_act_stats_table *t, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    int rc = pn_act_stats_table_write_stream(t, f);
    if (fclose(f) != 0)
        rc = -1;
    if (rc)
        fprintf(stderr, "act_stats: %s not written\n", path);
    return rc;
}

/* One tensor record into t. Returns 0, or -1 with stderr. */
static int read_entry(pn_act_stats_table *t, FILE *f, const char *path) {
    uint32_t len, n_in;
    uint64_t count;
    char name[ACT_STATS_MAX_NAME + 1];

    if (get_u32(f, &len) || len == 0 || len > ACT_STATS_MAX_NAME || fread(name, 1, len, f) != len) {
        fprintf(stderr, "act_stats: %s: bad or truncated tensor name\n", path);
        return -1;
    }
    name[len] = '\0';
    if (get_u32(f, &n_in) || get_u64(f, &count)) {
        fprintf(stderr, "act_stats: %s: %s truncated\n", path, name);
        return -1;
    }
    if (pn_act_stats_table_find(t, name)) {
        fprintf(stderr, "act_stats: %s: duplicate tensor %s\n", path, name);
        return -1;
    }
    pn_act_stats_entry *e = pn_act_stats_table_append(t, name, n_in);
    if (!e)
        return -1;
    e->count = count;
    if (get_f64_array(f, e->payload, pn_act_stats_payload_len(t->xtx, n_in))) {
        fprintf(stderr, "act_stats: %s: %s payload truncated\n", path, name);
        return -1;
    }
    return 0;
}

int pn_act_stats_table_read(pn_act_stats_table *t, const char *path, int xtx) {
    memset(t, 0, sizeof(*t));
    t->xtx = xtx;
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    char magic[4];
    uint32_t version = 0, n = 0;
    int rc = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, PN_ACT_STATS_MAGIC, 4) != 0 ||
        get_u32(f, &version) || version != PN_ACT_STATS_VERSION || get_u32(f, &n) ||
        get_u64(f, &t->n_tokens)) {
        fprintf(stderr, "act_stats: %s: not a mixw imatrix v1 file\n", path);
        rc = -1;
    }
    for (uint32_t i = 0; i < n && rc == 0; i++)
        rc = read_entry(t, f, path);
    if (rc == 0 && fgetc(f) != EOF) {
        fprintf(stderr, "act_stats: %s: trailing bytes (wrong kind? xtx=%d)\n", path, xtx);
        rc = -1;
    }
    fclose(f);
    if (rc)
        pn_act_stats_table_free(t);
    return rc;
}

void pn_act_stats_table_free(pn_act_stats_table *t) {
    for (uint32_t i = 0; i < t->n_entries; i++) {
        free(t->entries[i].name);
        free(t->entries[i].payload);
    }
    free(t->entries);
    t->entries = NULL;
    t->n_entries = 0;
    t->cap = 0;
    t->n_tokens = 0;
}
