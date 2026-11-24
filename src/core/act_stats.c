/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats.c
 * @brief  Implementation of PN_ACT_STATS (act_stats.h): the weight-pointer -> tensor-name map,
 *         the activation observer that accumulates sum x^2 / X^T X (act_stats_xtx.h), and the
 *         end-of-run write.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "act_stats.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include "act_stats_file.h"
#include "act_stats_xtx.h"
#include "qweight.h"

/* One GGUF tensor's data pointer and what has been resolved for it. */
typedef struct {
    const void *data;     /* NULL = empty hash slot */
    const char *name;     /* the GGUF's own (NUL-terminated) name string */
    uint32_t    n_in;     /* the GGUF's dims[0] */
    int32_t     imatrix;  /* entry index in the imatrix table; -1 until first dispatch */
    int32_t     xtx;      /* index in g_stats.xtx, or -1 */
} act_stats_slot;

static struct {
    int                active;
    int                paused;
    int                failed;           /* an accumulator could not be allocated */
    const void        *owner;
    char              *path;
    char              *xtx_path;
    FILE              *out;
    FILE              *out_xtx;
    act_stats_slot    *slots;
    uint32_t           mask;             /* slots has mask + 1 entries, a power of two */
    pn_act_stats_table imatrix;
    pn_act_xtx        *xtx;              /* the PN_ACT_STATS_FULL tensors, in list order */
    uint32_t           n_xtx;
    unsigned           xtx_flags;        /* PN_ACT_XTX_F32 */
    int                xtx_fifo;         /* the .xtx path is a FIFO: opened at the write */
    uint64_t           unnamed;          /* dispatches on a weight with no GGUF name */
    uint64_t           mismatched;       /* dispatches whose cols disagree with the tensor */
} g_stats;

static uint32_t slot_hash(const void *data) {
    uint64_t h = (uint64_t)(uintptr_t)data * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(h >> 32);
}

static act_stats_slot *slot_lookup(const void *data) {
    for (uint32_t i = slot_hash(data) & g_stats.mask;; i = (i + 1) & g_stats.mask) {
        act_stats_slot *s = &g_stats.slots[i];
        if (s->data == data)
            return s;
        if (!s->data)
            return NULL;
    }
}

/* Map every 2-D tensor's data pointer to its name. A pointer seen twice keeps the first. */
static int map_tensor_names(const pn_gguf_file *gguf) {
    uint32_t cap = 1024;
    while (cap < 2 * gguf->n_tensors)
        cap *= 2;
    g_stats.slots = calloc(cap, sizeof(*g_stats.slots));
    if (!g_stats.slots)
        return -1;
    g_stats.mask = cap - 1;
    for (uint64_t t = 0; t < gguf->n_tensors; t++) {
        const pn_gguf_tensor_info *info = &gguf->tensors[t];
        const void *data = (const char *)gguf->data + info->offset;
        if (info->n_dims < 2 || !info->name.data || slot_lookup(data))
            continue;
        uint32_t i = slot_hash(data) & g_stats.mask;
        while (g_stats.slots[i].data)
            i = (i + 1) & g_stats.mask;
        g_stats.slots[i] = (act_stats_slot){ data, info->name.data, (uint32_t)info->dims[0], -1, -1 };
    }
    return 0;
}

/* Append one zeroed X^T X accumulator; its index, or -1. An n_in the file reader refuses
 * (act_stats_file.h) is refused here, before hours of accumulation. */
static int32_t append_xtx(const char *name, uint32_t n_in) {
    if (!pn_act_stats_payload_len(1, n_in)) {
        fprintf(stderr, "act_stats: %s: n_in %u out of range for an X^T X\n", name, n_in);
        return -1;
    }
    pn_act_xtx *grown = realloc(g_stats.xtx, (size_t)(g_stats.n_xtx + 1) * sizeof(*grown));
    if (!grown)
        return -1;
    g_stats.xtx = grown;
    if (pn_act_xtx_init(&g_stats.xtx[g_stats.n_xtx], name, n_in, 0, g_stats.xtx_flags) != 0)
        return -1;
    return (int32_t)g_stats.n_xtx++;
}

/* Allocate the X^T X accumulator for one name of PN_ACT_STATS_FULL and bind its slots. */
static int bind_xtx_name(const char *name) {
    for (uint32_t t = 0; t < g_stats.n_xtx; t++)
        if (strcmp(g_stats.xtx[t].name, name) == 0)
            return 0;
    int32_t idx = -1;
    for (uint32_t i = 0; i <= g_stats.mask; i++) {
        act_stats_slot *s = &g_stats.slots[i];
        if (!s->data || strcmp(s->name, name) != 0)
            continue;
        if (idx < 0 && (idx = append_xtx(name, s->n_in)) < 0)
            return -1;
        s->xtx = idx;
    }
    if (idx < 0)
        fprintf(stderr, "act_stats: PN_ACT_STATS_FULL: no 2-D tensor named '%s'\n", name);
    return idx < 0 ? -1 : 0;
}

static int bind_xtx_list(const char *full_list) {
    char *list = strdup(full_list ? full_list : "");
    if (!list)
        return -1;
    int rc = 0;
    for (char *save = NULL, *tok = strtok_r(list, ",", &save); tok && rc == 0;
         tok = strtok_r(NULL, ",", &save))
        rc = bind_xtx_name(tok);
    free(list);
    return rc;
}

/* The imatrix entry for a slot, created at its first dispatch (so the file is in forward order). */
static pn_act_stats_entry *slot_entry(act_stats_slot *s, int n_in) {
    if (s->imatrix < 0) {
        pn_act_stats_entry *e = pn_act_stats_table_find(&g_stats.imatrix, s->name);
        if (!e && !(e = pn_act_stats_table_append(&g_stats.imatrix, s->name, (uint32_t)n_in))) {
            g_stats.failed = 1;
            return NULL;
        }
        s->imatrix = (int32_t)(e - g_stats.imatrix.entries);
    }
    return &g_stats.imatrix.entries[s->imatrix];
}

static void act_stats_observe(const pn_qweight *w, const float *acts, int B, void *user) {
    (void)user;
    if (g_stats.paused)
        return;
    act_stats_slot *s = slot_lookup(w->data);
    if (!s) {
        g_stats.unnamed++;
        return;
    }
    pn_act_stats_entry *e = slot_entry(s, w->cols);
    if (!e)
        return;
    if (e->n_in != (uint32_t)w->cols) {
        g_stats.mismatched++;
        return;
    }
    for (int b = 0; b < B; b++) {
        const float *x = acts + (size_t)b * e->n_in;
        for (uint32_t j = 0; j < e->n_in; j++)
            e->payload[j] += (double)x[j] * (double)x[j];
    }
    e->count += (uint64_t)B;
    if (s->xtx >= 0 && g_stats.xtx[s->xtx].n_in == e->n_in)
        pn_act_xtx_add_rows(&g_stats.xtx[s->xtx], acts, (uint32_t)B);
}

/* Free everything and return to the idle state (the observer must already be uninstalled). */
static void act_stats_reset(void) {
    if (g_stats.out)
        fclose(g_stats.out);
    if (g_stats.out_xtx)
        fclose(g_stats.out_xtx);
    pn_act_stats_table_free(&g_stats.imatrix);
    for (uint32_t t = 0; t < g_stats.n_xtx; t++)
        pn_act_xtx_free(&g_stats.xtx[t]);
    free(g_stats.xtx);
    free(g_stats.slots);
    free(g_stats.path);
    free(g_stats.xtx_path);
    memset(&g_stats, 0, sizeof(g_stats));
}

/* The imatrix file, and the X^T X file at `xtx_path` (NULL = `path`.xtx). A FIFO is not opened
 * here: opening one for writing blocks until its reader opens it, so it is opened at the write
 * and the forward never waits on the consumer. */
static int open_outputs(const char *path, const char *xtx_path) {
    size_t n = strlen(path);
    struct stat st;
    g_stats.path = strdup(path);
    if (!g_stats.path || !(g_stats.out = fopen(path, "wb"))) {
        perror(path);
        return -1;
    }
    if (g_stats.n_xtx == 0)
        return 0;
    if (xtx_path && *xtx_path) {
        g_stats.xtx_path = strdup(xtx_path);
    } else if ((g_stats.xtx_path = malloc(n + 5))) {
        memcpy(g_stats.xtx_path, path, n);
        memcpy(g_stats.xtx_path + n, ".xtx", 5);
    }
    if (!g_stats.xtx_path)
        return -1;
    g_stats.xtx_fifo = stat(g_stats.xtx_path, &st) == 0 && S_ISFIFO(st.st_mode);
    if (!g_stats.xtx_fifo && !(g_stats.out_xtx = fopen(g_stats.xtx_path, "wb"))) {
        perror(g_stats.xtx_path);
        return -1;
    }
    return 0;
}

int pn_act_stats_begin(const char *path, const char *full_list, const pn_gguf_file *gguf,
                       const void *owner) {
    return pn_act_stats_begin_ex(path, full_list, gguf, owner, NULL);
}

int pn_act_stats_begin_ex(const char *path, const char *full_list, const pn_gguf_file *gguf,
                          const void *owner, const pn_act_stats_options *options) {
    if (g_stats.active || !path || !*path || !gguf) {
        fprintf(stderr, "act_stats: %s\n", g_stats.active ? "already active" : "no path or GGUF");
        return -1;
    }
    g_stats.xtx_flags = options && options->xtx_f32 ? PN_ACT_XTX_F32 : 0u;
    if (map_tensor_names(gguf) < 0 || bind_xtx_list(full_list) < 0 ||
        open_outputs(path, options ? options->xtx_path : NULL) < 0) {
        fprintf(stderr, "act_stats: PN_ACT_STATS=%s refused\n", path);
        act_stats_reset();
        return -1;
    }
    g_stats.active = 1;
    g_stats.owner = owner;
    pn_qweight_set_activation_observer(act_stats_observe, NULL);
    fprintf(stderr, "act_stats: accumulating to %s (%u X^T X tensors%s%s%s%s)\n", path,
            g_stats.n_xtx, g_stats.xtx_path ? " -> " : "",
            g_stats.xtx_path ? g_stats.xtx_path : "",
            g_stats.n_xtx && (g_stats.xtx_flags & PN_ACT_XTX_F32) ? ", f32 accumulators" : "",
            g_stats.xtx_fifo ? ", a FIFO opened at the write" : "");
    return 0;
}

int pn_act_stats_begin_from_env(const pn_gguf_file *gguf, const void *owner) {
    const char *path = getenv("PN_ACT_STATS");
    const char *f32 = getenv("PN_ACT_STATS_XTX_F32");
    pn_act_stats_options options = { getenv("PN_ACT_STATS_XTX_PATH"), f32 && atoi(f32) != 0 };
    if (!path || !*path)
        return 0;
    return pn_act_stats_begin_ex(path, getenv("PN_ACT_STATS_FULL"), gguf, owner, &options) == 0
               ? 1
               : -1;
}

void pn_act_stats_set_paused(int paused) {
    g_stats.paused = paused;
}

int pn_act_stats_active(void) {
    return g_stats.active;
}

/* The X^T X file, strictly sequential (a FIFO reader sees it in one pass): header with the
 * tensor count and the largest row count, then each tensor in PN_ACT_STATS_FULL order. */
static int write_xtx(void) {
    uint64_t n_tokens = 0;
    int bad = 0;
    if (g_stats.xtx_fifo && !(g_stats.out_xtx = fopen(g_stats.xtx_path, "wb"))) {
        perror(g_stats.xtx_path);
        return -1;
    }
    for (uint32_t t = 0; t < g_stats.n_xtx; t++)
        n_tokens = g_stats.xtx[t].count > n_tokens ? g_stats.xtx[t].count : n_tokens;
    fprintf(stderr, "act_stats: X^T X order (%u tensors) ->", g_stats.n_xtx);
    for (uint32_t t = 0; t < g_stats.n_xtx; t++)
        fprintf(stderr, " %s", g_stats.xtx[t].name);
    fprintf(stderr, "\n");
    bad = pn_act_stats_write_header(g_stats.out_xtx, g_stats.n_xtx, n_tokens);
    for (uint32_t t = 0; t < g_stats.n_xtx && !bad; t++) {
        pn_act_xtx *x = &g_stats.xtx[t];
        bad = pn_act_stats_write_entry_head(g_stats.out_xtx, x->name, x->n_in, x->count) ||
              pn_act_xtx_write_payload(x, g_stats.out_xtx);
        /* Free each accumulator once written: a FIFO reader (mixw gptq) holds one H at a time,
         * and keeping a band's whole set (5.3 GB for 4 layers) alongside it swapped the Mac. */
        pn_act_xtx_free(x);
    }
    if (bad || fflush(g_stats.out_xtx) != 0) {
        fprintf(stderr, "act_stats: %s: write failed\n", g_stats.xtx_path);
        return -1;
    }
    return 0;
}

int pn_act_stats_finish(const void *owner) {
    if (!g_stats.active || owner != g_stats.owner)
        return 0;
    pn_qweight_set_activation_observer(NULL, NULL);
    int rc = pn_act_stats_table_write_stream(&g_stats.imatrix, g_stats.out);
    if (g_stats.n_xtx && write_xtx() != 0)
        rc = -1;
    if (g_stats.unnamed || g_stats.mismatched)
        fprintf(stderr, "act_stats: skipped %llu dispatches on weights with no GGUF name, %llu "
                "whose cols disagree with the tensor's\n", (unsigned long long)g_stats.unnamed,
                (unsigned long long)g_stats.mismatched);
    if (g_stats.failed) {
        fprintf(stderr, "act_stats: an accumulator could not be allocated; the file is partial\n");
        rc = -1;
    }
    fprintf(stderr, "act_stats: %s %s (%u tensors)\n", rc ? "FAILED writing" : "wrote",
            g_stats.path, g_stats.imatrix.n_entries);
    act_stats_reset();
    return rc;
}
