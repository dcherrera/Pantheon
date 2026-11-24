/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_act_stats_xtx.c
 * @brief The batched X^T X accumulator (src/core/act_stats_xtx.h) and the sequential .xtx write.
 *
 *   - the scalar rank-1 path and batched f64 (rows flushed through cblas_dsyrk under USE_BLAS)
 *     equal a naive f64 reference within 1e-12 of each entry's scale sqrt(D_ii D_jj), for row
 *     counts that are and are not multiples of the batch; f32 accumulators within 1e-5
 *   - the diagonal equals the sequential f64 sum x^2 bit for bit, and the matrix is symmetric
 *   - the f32 accumulator's written payload is f64 and matches its own entries
 *   - PN_ACT_STATS_XTX_PATH on a FIFO: a child reads the stream while the run writes it, and the
 *     bytes equal the same run written to a regular file (a writer that seeks cannot pass)
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/core/act_stats.h"
#include "../src/core/act_stats_file.h"
#include "../src/core/act_stats_xtx.h"
#include "../src/core/qweight.h"

#define OUT_DIR "build/act_stats_xtx"

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

static float *random_rows(uint32_t rows, uint32_t n, unsigned seed) {
    float *x = malloc((size_t)rows * n * sizeof(float));
    for (size_t i = 0; x && i < (size_t)rows * n; i++)
        x[i] = rand_value(&seed);
    return x;
}

/* Feed `rows` rows in uneven dispatches (1, 3, 7, ... rows), as a forward would. */
static void feed(pn_act_xtx *x, const float *data, uint32_t rows) {
    uint32_t done = 0, step = 1;
    while (done < rows) {
        uint32_t k = step < rows - done ? step : rows - done;
        pn_act_xtx_add_rows(x, data + (size_t)done * x->n_in, k);
        done += k;
        step = step * 2 + 1;
    }
}

/* The naive full reference: every row's outer product, in f64, in row order. */
static double *naive_xtx(const float *data, uint32_t rows, uint32_t n) {
    double *ref = calloc((size_t)n * n, sizeof(double));
    for (uint32_t r = 0; ref && r < rows; r++) {
        const float *x = data + (size_t)r * n;
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < n; j++)
                ref[(size_t)i * n + j] += (double)x[i] * (double)x[j];
    }
    return ref;
}

/* Largest |a - ref| / sqrt(ref_ii ref_jj) over every entry (an off-diagonal sum can cancel to
 * ~0, so the Cauchy-Schwarz bound is the scale, not |ref|). */
static double worst_scaled_diff(const pn_act_xtx *a, const double *ref) {
    const size_t n = a->n_in;
    double worst = 0.0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double scale = sqrt(ref[i * n + i] * ref[j * n + j]) + 1e-300;
            double d = fabs(pn_act_xtx_at(a, (uint32_t)i, (uint32_t)j) - ref[i * n + j]) / scale;
            worst = d > worst ? d : worst;
        }
    return worst;
}

static void test_batched_equals_scalar(uint32_t n, uint32_t rows, uint32_t batch) {
    pn_act_xtx scalar, batched, f32;
    float *data = random_rows(rows, n, 11u + n + rows);
    double *diag = calloc(n, sizeof(double));
    char what[200];
    int diag_bad = 0, sym_bad = 0;

    check(pn_act_xtx_init(&scalar, "s", n, batch, PN_ACT_XTX_SCALAR) == 0 &&
              pn_act_xtx_init(&batched, "b", n, batch, 0u) == 0 &&
              pn_act_xtx_init(&f32, "f", n, batch, PN_ACT_XTX_F32) == 0 && data && diag,
          "three accumulators allocate");
    feed(&scalar, data, rows);
    feed(&batched, data, rows);
    feed(&f32, data, rows);
    pn_act_xtx_finish(&scalar);
    pn_act_xtx_finish(&batched);
    pn_act_xtx_finish(&f32);
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t j = 0; j < n; j++)
            diag[j] += (double)data[(size_t)r * n + j] * (double)data[(size_t)r * n + j];
    for (uint32_t i = 0; i < n; i++) {
        diag_bad += pn_act_xtx_at(&batched, i, i) != diag[i] ||
                    pn_act_xtx_at(&f32, i, i) != diag[i] || pn_act_xtx_at(&scalar, i, i) != diag[i];
        for (uint32_t j = 0; j < i; j++)
            sym_bad += pn_act_xtx_at(&batched, i, j) != pn_act_xtx_at(&batched, j, i) ||
                       pn_act_xtx_at(&f32, i, j) != pn_act_xtx_at(&f32, j, i);
    }
    double *ref = naive_xtx(data, rows, n);
    const double d_scalar = ref ? worst_scaled_diff(&scalar, ref) : 1.0;
    const double d64 = ref ? worst_scaled_diff(&batched, ref) : 1.0;
    const double d32 = ref ? worst_scaled_diff(&f32, ref) : 1.0;
    free(ref);
    printf("  n %u, %u rows, batch %u, vs the naive f64 reference: scalar %.2e, batched f64 "
           "%.2e, f32 %.2e\n", n, rows, batch, d_scalar, d64, d32);
    snprintf(what, sizeof(what), "n %u, %u rows: the scalar path = the naive reference (%.2e)",
             n, rows, d_scalar);
    check(d_scalar <= 1e-12, what);
    snprintf(what, sizeof(what), "n %u, %u rows (batch %u): batched f64 within 1e-12 (%.2e)", n,
             rows, batch, d64);
    check(d64 <= 1e-12, what);
    snprintf(what, sizeof(what), "n %u, %u rows: f32 accumulators within 1e-5 (%.2e)", n, rows,
             d32);
    check(d32 <= 1e-5, what);
    check(scalar.count == rows && batched.count == rows && f32.count == rows, "counts = rows");
    check(diag_bad == 0, "diagonal = the sequential f64 sum x^2, bit for bit, every path");
    check(sym_bad == 0, "symmetric after finish");
    pn_act_xtx_add_rows(&batched, data, 1);
    check(batched.count == rows, "no rows accepted once finished");
    pn_act_xtx_free(&scalar);
    pn_act_xtx_free(&batched);
    pn_act_xtx_free(&f32);
    free(data);
    free(diag);
}

/* The f32 accumulator writes f64: read the payload back and compare with its own entries. */
static void test_f32_payload(void) {
    const uint32_t n = 29, rows = 77;
    pn_act_xtx x;
    float *data = random_rows(rows, n, 5u);
    double *back = malloc((size_t)n * n * sizeof(double));
    FILE *f = fopen(OUT_DIR "/f32.payload", "w+b");
    int bad = 0;

    check(f && back && pn_act_xtx_init(&x, "t", n, 16, PN_ACT_XTX_F32) == 0,
          "f32 payload fixture");
    if (!f || !back) {
        free(data);
        free(back);
        return;
    }
    feed(&x, data, rows);
    check(pn_act_xtx_write_payload(&x, f) == 0, "f32 payload writes");
    check(ftell(f) == (long)((size_t)n * n * sizeof(double)), "payload is n^2 f64");
    rewind(f);
    check(fread(back, sizeof(double), (size_t)n * n, f) == (size_t)n * n, "payload reads back");
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++)
            bad += back[(size_t)i * n + j] != pn_act_xtx_at(&x, i, j);
    check(bad == 0, "written f64 = the f32 entries widened (diagonal the f64 sum)");
    fclose(f);
    pn_act_xtx_free(&x);
    free(data);
    free(back);
}

/* ---- the whole PN_ACT_STATS path, to a regular file and to a FIFO ---- */

#define RUN_N 48
#define RUN_ROWS 333
static const char *const run_names[2] = { "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight" };

/* Two named F32 weights (n_in 48 and 40) in a fake GGUF; RUN_ROWS rows through pn_qmm. */
static int stats_run(const char *path, const char *xtx_path, int f32) {
    static const int cols[2] = { RUN_N, 40 }, rows_out = 8;
    pn_gguf_file gguf;
    pn_gguf_tensor_info infos[2];
    pn_qweight w[2];
    float *weights = random_rows(2, RUN_N * rows_out, 3u);
    float *acts = random_rows(RUN_ROWS, RUN_N, 9u), *out = malloc((size_t)rows_out * sizeof(float));
    pn_act_stats_options options = { xtx_path, f32 };
    int owner = 0, rc = -1;

    memset(&gguf, 0, sizeof(gguf));
    memset(infos, 0, sizeof(infos));
    gguf.data = weights;
    gguf.n_tensors = 2;
    gguf.tensors = infos;
    for (int t = 0; t < 2; t++) {
        infos[t].name.data = (char *)run_names[t];
        infos[t].name.len = strlen(run_names[t]);
        infos[t].n_dims = 2;
        infos[t].dims[0] = (uint64_t)cols[t];
        infos[t].dims[1] = (uint64_t)rows_out;
        infos[t].type = GGUF_QUANT_F32;
        infos[t].offset = (size_t)t * RUN_N * rows_out * sizeof(float);
        w[t] = (pn_qweight){ .data = weights + (size_t)t * RUN_N * rows_out,
                             .type = GGUF_QUANT_F32, .rows = rows_out, .cols = cols[t],
                             .row_stride = (size_t)cols[t] * sizeof(float) };
    }
    if (weights && acts && out &&
        pn_act_stats_begin_ex(path, "blk.0.ffn_down.weight,blk.0.ffn_gate.weight", &gguf, &owner,
                              &options) == 0) {
        for (int r = 0; r < RUN_ROWS; r++) {
            pn_qmv(&w[0], acts + (size_t)r * RUN_N, out);
            pn_qmv(&w[1], acts + (size_t)r * RUN_N, out);
        }
        rc = pn_act_stats_finish(&owner);
    }
    free(weights);
    free(acts);
    free(out);
    return rc;
}

static long file_bytes(const char *path, unsigned char **out) {
    FILE *f = fopen(path, "rb");
    long n = -1;
    if (f && fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) >= 0 && fseek(f, 0, SEEK_SET) == 0 &&
        (*out = malloc((size_t)n + 1u)) && fread(*out, 1, (size_t)n, f) != (size_t)n)
        n = -1;
    if (f)
        fclose(f);
    return n;
}

/* The child copies the FIFO to `copy` byte for byte, then exits 0. */
static void fifo_reader(const char *fifo, const char *copy) {
    FILE *in = fopen(fifo, "rb"), *out = fopen(copy, "wb");
    unsigned char buf[65536];
    size_t k;
    if (!in || !out)
        _exit(2);
    while ((k = fread(buf, 1, sizeof(buf), in)) > 0)
        if (fwrite(buf, 1, k, out) != k)
            _exit(3);
    _exit(fclose(out) == 0 && !ferror(in) ? 0 : 4);
}

static void test_fifo_stream(int f32) {
    const char *fifo = OUT_DIR "/h.fifo";
    unsigned char *reg = NULL, *piped = NULL;
    pn_act_stats_table t;
    int status = -1;
    char what[160];
    pid_t child;

    check(stats_run(OUT_DIR "/reg.imatrix", OUT_DIR "/reg.xtx", f32) == 0,
          "a run writes X^T X to PN_ACT_STATS_XTX_PATH (regular file)");
    unlink(fifo);
    check(mkfifo(fifo, 0600) == 0, "mkfifo");
    child = fork();
    if (child == 0)
        fifo_reader(fifo, OUT_DIR "/fifo_copy.xtx");
    if (stats_run(OUT_DIR "/fifo.imatrix", fifo, f32) != 0) {
        check(0, "the same run streams into the FIFO");
        /* A writer that never opened the FIFO leaves the reader blocked in open(): release it. */
        int fd = open(fifo, O_WRONLY | O_NONBLOCK);
        if (fd >= 0)
            close(fd);
    }
    check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0, "the FIFO reader read to EOF");
    const long a = file_bytes(OUT_DIR "/reg.xtx", &reg);
    const long b = file_bytes(OUT_DIR "/fifo_copy.xtx", &piped);
    snprintf(what, sizeof(what), "%s accumulators: FIFO stream (%ld B) = regular file (%ld B), "
             "byte for byte", f32 ? "f32" : "f64", b, a);
    check(a > 0 && a == b && memcmp(reg, piped, (size_t)a) == 0, what);
    check(pn_act_stats_table_read(&t, OUT_DIR "/fifo_copy.xtx", 1) == 0 && t.n_entries == 2 &&
              strcmp(t.entries[0].name, run_names[1]) == 0 &&
              strcmp(t.entries[1].name, run_names[0]) == 0 && t.n_tokens == RUN_ROWS &&
              t.entries[0].count == RUN_ROWS,
          "the stream reads back: FULL-list order, header counts right");
    pn_act_stats_table_free(&t);
    unlink(fifo);
    free(reg);
    free(piped);
}

int main(void) {
    mkdir("build", 0755);
    mkdir(OUT_DIR, 0755);
    test_batched_equals_scalar(37, 300, 128);   /* 2 full batches + 44 */
    test_batched_equals_scalar(130, 256, 128);  /* exactly 2 batches */
    test_batched_equals_scalar(257, 1000, 64);  /* many batches + 40 */
    test_f32_payload();
    test_fifo_stream(0);
    test_fifo_stream(1);
    printf("test_act_stats_xtx: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
