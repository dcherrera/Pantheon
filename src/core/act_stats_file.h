/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats_file.h
 * @brief  The "mixw imatrix v1" activation-statistics file: in-memory table, reader, writer
 *         and the additive merge. Written by PN_ACT_STATS (act_stats.h); read by the mixw
 *         proxy scorer.
 *
 * FORMAT (little-endian, no padding):
 *
 *   header   char magic[4] = "MXIM"; u32 version = 1; u32 n_tensors; u64 n_tokens
 *   tensor   u32 name_len; name bytes (GGUF tensor name, no NUL); u32 n_in; u64 count;
 *            f64 payload[]
 *
 *   imatrix file (`<file>`):      payload = sum_x2[n_in], sum over activation rows of x_j^2
 *   X^T X file   (`<file>.xtx`):  payload = xtx[n_in * n_in] row-major, sum over rows of x x^T
 *
 * x is the activation row that multiplies the weight (after any norm, as the CPU forward feeds
 * it); count is the number of rows accumulated into that tensor; n_tokens is the largest count
 * in the file (= forward tokens for a one-row-per-token forward). The two kinds share the
 * header; the reader is told which one it reads.
 *
 * MERGE RULE: files are additive. Same name: n_in must match, counts add, payloads add. A name
 * in only one input is copied. n_tokens is recomputed as the largest count. So the halves of a
 * PN_SPLIT_LAYERS run (disjoint tensors, same tokens) and separate runs over different tokens
 * (same tensors) both merge to the stats of the whole forward over all those tokens.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_ACT_STATS_FILE_H
#define PN_ACT_STATS_FILE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_ACT_STATS_MAGIC "MXIM"
#define PN_ACT_STATS_VERSION 1u

/** One tensor's accumulator. payload has n_in (imatrix) or n_in * n_in (X^T X) doubles. */
typedef struct {
    char     *name;
    uint32_t  n_in;
    uint64_t  count;
    double   *payload;
} pn_act_stats_entry;

/** A whole file in memory. xtx = 1: payloads are n_in * n_in. */
typedef struct {
    int                 xtx;
    uint64_t            n_tokens;
    uint32_t            n_entries;
    uint32_t            cap;
    pn_act_stats_entry *entries;
} pn_act_stats_table;

/**
 * Number of doubles in one entry's payload for this table kind (0 if it would overflow).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t pn_act_stats_payload_len(int xtx, uint32_t n_in);

/**
 * Find an entry by exact name.
 *
 * @return the entry, or NULL
 *
 * Thread-safe: yes (read only)   Reentrant: yes   Signal-safe: no
 */
pn_act_stats_entry *pn_act_stats_table_find(pn_act_stats_table *t, const char *name);

/**
 * Append a zeroed entry (count 0, payload all zero). The caller ensures the name is new.
 *
 * @return the entry, or NULL (allocation failure or an n_in whose payload overflows; stderr)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
pn_act_stats_entry *pn_act_stats_table_append(pn_act_stats_table *t, const char *name,
                                              uint32_t n_in);

/**
 * Add `src` into `dst` by the merge rule above. Both must be the same kind.
 *
 * @return 0, or -1 (kind mismatch, an n_in mismatch for a shared name, or allocation failure;
 *         stderr names the tensor). On -1, dst may hold a partial merge.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_table_merge(pn_act_stats_table *dst, const pn_act_stats_table *src);

/**
 * Write the file header: magic, version, `n_tensors`, `n_tokens`. A streaming writer knows both
 * before its first byte; nothing is patched afterwards.
 *
 * @return 0, or -1 on a write error
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_write_header(FILE *f, uint32_t n_tensors, uint64_t n_tokens);

/**
 * Write one tensor record's head: name_len, name, n_in, count. Its payload follows.
 *
 * @return 0, or -1 on a write error
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_write_entry_head(FILE *f, const char *name, uint32_t n_in, uint64_t count);

/**
 * Write `n` doubles, little-endian.
 *
 * @return 0, or -1 on a write error
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_write_f64(FILE *f, const double *v, uint64_t n);

/**
 * Write the table to `f` in the format above; n_tokens written = the largest count.
 *
 * @return 0, or -1 on a write error (stderr)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_table_write_stream(const pn_act_stats_table *t, FILE *f);

/**
 * Write the table to `path` (created / truncated).
 *
 * @return 0, or -1 (stderr says why)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_table_write(const pn_act_stats_table *t, const char *path);

/**
 * Read `path` into a fresh table of kind `xtx` (t is overwritten; free it after).
 *
 * @return 0, or -1 (unreadable, bad magic/version, truncated, duplicate name; stderr)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_table_read(pn_act_stats_table *t, const char *path, int xtx);

/**
 * Free every entry and reset the table to empty (kind kept).
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
void pn_act_stats_table_free(pn_act_stats_table *t);

#ifdef __cplusplus
}
#endif

#endif /* PN_ACT_STATS_FILE_H */
