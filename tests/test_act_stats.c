/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_act_stats.c
 * @brief PN_ACT_STATS at the dispatch level: random weights and activations through every
 *   qweight entry point (pn_qmv, pn_qmv_group, pn_qmm small and above PN_QMM_MAX_B,
 *   pn_qmm_prefill) against a naive per-channel reference.
 *
 *   - outputs with the stats on equal outputs with them off, byte for byte
 *   - sum x^2 per input channel and the row count per tensor equal the naive reference (rows
 *     reported once: the per-row fallbacks do not double count)
 *   - the full X^T X of a named tensor equals a naive full n x n reference (both halves)
 *   - a weight with no GGUF name and a paused dispatch record nothing
 *   - the file round-trips; merge is additive over shared names and copies the rest
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../src/core/act_stats.h"
#include "../src/core/act_stats_file.h"
#include "../src/core/qweight.h"

#define N_TENSORS 3
#define OUT_DIR "build/act_stats"

static int checks, failures;

static void check(int ok, const char *what) {
    checks++;
    failures += !ok;
    if (!ok)
        printf("  FAIL  %s\n", what);
}

static float rand_value(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state >> 8) / 16777216.0f - 0.5f) * 4.0f;
}

/* Three named F32 weights of different shapes plus one unnamed, in one fake GGUF. */
static const char *const tensor_names[N_TENSORS] = { "blk.0.attn_qkv.weight",
                                                     "blk.0.ffn_down.weight", "output.weight" };
static const int tensor_rows[N_TENSORS] = { 40, 24, 33 };
static const int tensor_cols[N_TENSORS] = { 24, 40, 17 };

typedef struct {
    pn_gguf_file        gguf;
    pn_gguf_tensor_info infos[N_TENSORS];
    float              *data;
    pn_qweight          w[N_TENSORS];
    pn_qweight          unnamed;
    double             *ref[N_TENSORS];   /* naive sum x^2 */
    unsigned long long  ref_count[N_TENSORS];
    double             *ref_xtx;          /* naive full X^T X of tensor 0 */
} dispatch_fixture;

static void fixture_build(dispatch_fixture *f) {
    memset(f, 0, sizeof(*f));
    size_t total = 0, offsets[N_TENSORS + 1];
    for (int t = 0; t < N_TENSORS; t++) {
        offsets[t] = total;
        total += (size_t)tensor_rows[t] * (size_t)tensor_cols[t];
    }
    offsets[N_TENSORS] = total;
    total += (size_t)tensor_rows[0] * (size_t)tensor_cols[0];   /* the unnamed copy */
    f->data = malloc(total * sizeof(float));
    unsigned seed = 7;
    for (size_t i = 0; i < total; i++)
        f->data[i] = rand_value(&seed);
    f->gguf.data = f->data;
    f->gguf.n_tensors = N_TENSORS;
    f->gguf.tensors = f->infos;
    for (int t = 0; t < N_TENSORS; t++) {
        f->infos[t].name.data = (char *)tensor_names[t];
        f->infos[t].name.len = strlen(tensor_names[t]);
        f->infos[t].n_dims = 2;
        f->infos[t].dims[0] = (uint64_t)tensor_cols[t];
        f->infos[t].dims[1] = (uint64_t)tensor_rows[t];
        f->infos[t].type = GGUF_QUANT_F32;
        f->infos[t].offset = offsets[t] * sizeof(float);
        f->w[t] = (pn_qweight){ .data = f->data + offsets[t], .type = GGUF_QUANT_F32,
                                .rows = tensor_rows[t], .cols = tensor_cols[t],
                                .row_stride = (size_t)tensor_cols[t] * sizeof(float) };
        f->ref[t] = calloc((size_t)tensor_cols[t], sizeof(double));
    }
    f->unnamed = f->w[0];
    f->unnamed.data = f->data + offsets[N_TENSORS];
    f->ref_xtx = calloc((size_t)tensor_cols[0] * (size_t)tensor_cols[0], sizeof(double));
}

static void fixture_free(dispatch_fixture *f) {
    for (int t = 0; t < N_TENSORS; t++)
        free(f->ref[t]);
    free(f->ref_xtx);
    free(f->data);
}

/* The naive reference: every row of acts multiplies tensor t once. */
static void reference_add(dispatch_fixture *f, int t, const float *acts, int B) {
    int n = tensor_cols[t];
    for (int b = 0; b < B; b++) {
        const float *x = acts + (size_t)b * n;
        for (int j = 0; j < n; j++)
            f->ref[t][j] += (double)x[j] * (double)x[j];
        if (t == 0) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    f->ref_xtx[(size_t)i * n + j] += (double)x[i] * (double)x[j];
        }
    }
    f->ref_count[t] += (unsigned long long)B;
}

typedef enum { ENTRY_QMV, ENTRY_QMM, ENTRY_QMM_PREFILL, ENTRY_GROUP } dispatch_entry;

/* One scripted dispatch. tensor -1 = the unnamed weight; ENTRY_GROUP runs tensors 0, 1, 2. */
typedef struct {
    dispatch_entry entry;
    int            tensor;
    int            B;
    int            paused;
} dispatch_op;

static const dispatch_op dispatch_script[] = {
    { ENTRY_QMV, 0, 1, 0 },          { ENTRY_QMV, 0, 1, 0 },        { ENTRY_QMV, 0, 1, 0 },
    { ENTRY_QMM, 1, 3, 0 },          { ENTRY_QMM, 2, PN_QMM_MAX_B + 6, 0 },
    { ENTRY_QMM_PREFILL, 0, 5, 0 },  { ENTRY_GROUP, 0, 1, 0 },      { ENTRY_QMV, -1, 1, 0 },
    { ENTRY_QMV, 1, 1, 1 },          { ENTRY_QMM, 2, 2, 0 },        { ENTRY_QMM_PREFILL, 1, 1, 0 },
};
#define N_OPS (int)(sizeof(dispatch_script) / sizeof(dispatch_script[0]))

/* Random activations for one op's weight (B rows), deterministic per (op, slot). */
static float *op_acts(int op, int slot, int B, int cols) {
    unsigned seed = 1000u + 31u * (unsigned)op + (unsigned)slot;
    float *acts = malloc((size_t)B * (size_t)cols * sizeof(float));
    for (int i = 0; i < B * cols; i++)
        acts[i] = rand_value(&seed);
    return acts;
}

static const pn_qweight *op_weight(dispatch_fixture *f, int t) {
    return t < 0 ? &f->unnamed : &f->w[t];
}

/* Run op `op`, outputs into out[0..2] (one per weight for a group). Records the reference when
 * `reference` is set and the op is live and named. */
static void run_op(dispatch_fixture *f, int op, float *out[3], int reference) {
    const dispatch_op *d = &dispatch_script[op];
    if (d->entry == ENTRY_GROUP) {
        const pn_qweight *ws[3];
        float *acts[3];
        for (int k = 0; k < 3; k++) {
            ws[k] = &f->w[k];
            acts[k] = op_acts(op, k, 1, f->w[k].cols);
        }
        pn_qmv_group(ws, (const float *const *)acts, out, 3);
        for (int k = 0; k < 3; k++) {
            if (reference)
                reference_add(f, k, acts[k], 1);
            free(acts[k]);
        }
        return;
    }
    const pn_qweight *w = op_weight(f, d->tensor);
    float *acts = op_acts(op, 0, d->B, w->cols);
    pn_act_stats_set_paused(d->paused);
    if (d->entry == ENTRY_QMV)
        pn_qmv(w, acts, out[0]);
    else if (d->entry == ENTRY_QMM)
        pn_qmm(w, acts, d->B, out[0]);
    else
        pn_qmm_prefill(w, acts, d->B, out[0]);
    pn_act_stats_set_paused(0);
    if (reference && !d->paused && d->tensor >= 0)
        reference_add(f, d->tensor, acts, d->B);
    free(acts);
}

/* Output buffers for every op, large enough for any weight of the fixture. */
static float *op_outputs[N_OPS][3];

static void alloc_outputs(float *bufs[N_OPS][3]) {
    for (int op = 0; op < N_OPS; op++)
        for (int k = 0; k < 3; k++)
            bufs[op][k] = calloc((size_t)(PN_QMM_MAX_B + 6) * 64, sizeof(float));
}

static void free_outputs(float *bufs[N_OPS][3]) {
    for (int op = 0; op < N_OPS; op++)
        for (int k = 0; k < 3; k++)
            free(bufs[op][k]);
}

static void compare_with_reference(dispatch_fixture *f, const char *path) {
    pn_act_stats_table t, x;
    check(pn_act_stats_table_read(&t, path, 0) == 0, "imatrix file reads back");
    char xpath[256];
    snprintf(xpath, sizeof(xpath), "%s.xtx", path);
    check(pn_act_stats_table_read(&x, xpath, 1) == 0, "X^T X file reads back");
    check(t.n_entries == N_TENSORS, "one entry per named tensor (unnamed weight skipped)");
    unsigned long long max_count = 0;
    for (int k = 0; k < N_TENSORS; k++) {
        pn_act_stats_entry *e = pn_act_stats_table_find(&t, tensor_names[k]);
        check(e != NULL, tensor_names[k]);
        if (!e)
            continue;
        max_count = f->ref_count[k] > max_count ? f->ref_count[k] : max_count;
        check(e->n_in == (uint32_t)tensor_cols[k], "n_in = weight cols");
        check(e->count == f->ref_count[k], "row count = reference (no double count)");
        int bad = 0;
        for (int j = 0; j < tensor_cols[k]; j++)
            bad += fabs(e->payload[j] - f->ref[k][j]) > 1e-12 * (1.0 + fabs(f->ref[k][j]));
        check(bad == 0, "sum x^2 per input channel = naive reference");
    }
    check(t.n_tokens == max_count, "header n_tokens = largest count");
    pn_act_stats_entry *xe = pn_act_stats_table_find(&x, tensor_names[0]);
    check(x.n_entries == 1 && xe != NULL && xe->count == f->ref_count[0], "X^T X entry and count");
    if (xe) {
        int n = tensor_cols[0], bad = 0;
        for (int i = 0; i < n * n; i++)
            bad += fabs(xe->payload[i] - f->ref_xtx[i]) > 1e-9 * (1.0 + fabs(f->ref_xtx[i]));
        check(bad == 0, "X^T X = naive full reference (both triangles)");
    }
    pn_act_stats_table_free(&t);
    pn_act_stats_table_free(&x);
}

static void test_dispatch_against_reference(void) {
    dispatch_fixture f;
    fixture_build(&f);
    const char *path = OUT_DIR "/dispatch.imatrix";
    int owner = 0, other = 0;
    size_t out_bytes = (size_t)(PN_QMM_MAX_B + 6) * 64 * sizeof(float);

    /* Knob off: no observer installed; these are the reference outputs. */
    alloc_outputs(op_outputs);
    for (int op = 0; op < N_OPS; op++)
        run_op(&f, op, op_outputs[op], 0);

    check(pn_act_stats_begin(path, "nope.weight", &f.gguf, &owner) == -1, "unknown FULL name refused");
    check(!pn_act_stats_active(), "refused begin leaves nothing installed");
    check(pn_act_stats_begin(path, tensor_names[0], &f.gguf, &owner) == 0, "begin");
    check(pn_act_stats_begin(path, NULL, &f.gguf, &owner) == -1, "second begin refused");

    float *live[N_OPS][3];
    alloc_outputs(live);
    for (int op = 0; op < N_OPS; op++)
        run_op(&f, op, live[op], 1);
    int differ = 0;
    for (int op = 0; op < N_OPS; op++)
        for (int k = 0; k < 3; k++)
            differ += memcmp(live[op][k], op_outputs[op][k], out_bytes) != 0;
    check(differ == 0, "every output byte-identical, stats on vs off");

    check(pn_act_stats_finish(&other) == 0 && pn_act_stats_active(), "finish by a non-owner is a no-op");
    check(pn_act_stats_finish(&owner) == 0 && !pn_act_stats_active(), "finish writes and stops");
    compare_with_reference(&f, path);
    free_outputs(live);
    free_outputs(op_outputs);
    fixture_free(&f);
}

/* Write two tables, read them back, merge: shared names add, the rest are copied. */
static void test_file_merge(void) {
    pn_act_stats_table a = { 0 }, b = { 0 }, ra, rb;
    pn_act_stats_entry *e;
    e = pn_act_stats_table_append(&a, "blk.0.ffn_up.weight", 4);
    e->count = 3;
    for (int j = 0; j < 4; j++)
        e->payload[j] = 1.5 * j;
    e = pn_act_stats_table_append(&a, "blk.1.ffn_up.weight", 2);
    e->count = 3;
    e->payload[1] = 9.0;
    e = pn_act_stats_table_append(&b, "blk.0.ffn_up.weight", 4);
    e->count = 5;
    for (int j = 0; j < 4; j++)
        e->payload[j] = 0.25;
    e = pn_act_stats_table_append(&b, "output.weight", 3);
    e->count = 5;
    e->payload[2] = -1.0;
    check(pn_act_stats_table_write(&a, OUT_DIR "/a.imatrix") == 0 &&
          pn_act_stats_table_write(&b, OUT_DIR "/b.imatrix") == 0, "write two tables");
    check(pn_act_stats_table_read(&ra, OUT_DIR "/a.imatrix", 0) == 0 &&
          pn_act_stats_table_read(&rb, OUT_DIR "/b.imatrix", 0) == 0, "read them back");
    check(ra.n_tokens == 3 && rb.n_tokens == 5, "n_tokens = largest count");
    pn_act_stats_table_free(&rb);
    check(pn_act_stats_table_read(&rb, OUT_DIR "/b.imatrix", 1) == -1, "read as the wrong kind refused");
    check(pn_act_stats_table_read(&rb, OUT_DIR "/b.imatrix", 0) == 0, "re-read");
    check(pn_act_stats_table_merge(&ra, &rb) == 0, "merge");
    e = pn_act_stats_table_find(&ra, "blk.0.ffn_up.weight");
    check(ra.n_entries == 3 && e && e->count == 8 && e->payload[3] == 4.75, "shared name adds");
    e = pn_act_stats_table_find(&ra, "output.weight");
    check(e && e->count == 5 && e->payload[2] == -1.0, "a name in one input is copied");
    e = pn_act_stats_table_find(&ra, "blk.1.ffn_up.weight");
    check(e && e->count == 3 && e->payload[1] == 9.0, "the other input's own name is kept");
    pn_act_stats_table bad = { 0 };
    pn_act_stats_table_append(&bad, "output.weight", 7);
    check(pn_act_stats_table_merge(&ra, &bad) == -1, "n_in mismatch refused");
    pn_act_stats_table_free(&a);
    pn_act_stats_table_free(&b);
    pn_act_stats_table_free(&ra);
    pn_act_stats_table_free(&rb);
    pn_act_stats_table_free(&bad);
}

int main(void) {
    mkdir("build", 0755);
    mkdir(OUT_DIR, 0755);
    test_dispatch_against_reference();
    test_file_merge();
    printf("test_act_stats: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
