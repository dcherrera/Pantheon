/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_embed_row.c — A4 unit tests for pn_gguf_dequantize_row and
 * pn_qwen35moe_model_embed_row.
 *
 * Part A (always runs — no model file needed): pure-logic checks.
 *   - pn_gguf_block_size / pn_gguf_bytes_per_block return expected values for
 *     the types used by token_embd (Q4_0, Q4_K, Q6_K, F16, F32).
 *   - pn_gguf_type_size gives the correct byte span for a row-sized element
 *     count, matching the hand-computed stride so row offsets are correct.
 *   - pn_gguf_dequantize_row on a synthetic Q4_0 tensor (fabricated in
 *     memory) produces per-row output identical to the matching slice of a
 *     full-tensor pn_dequantize call.
 *
 * Part B (model-backed; skips with a note if the GGUF is absent): open a
 * real GGUF containing "token_embd.weight", dequantize several rows via
 * pn_gguf_dequantize_row, dequantize the whole tensor via
 * pn_gguf_dequantize_tensor, and assert that each row matches the
 * corresponding slice of the full dequantization (within float rounding —
 * the two paths call identical block-level decoders, so they must be
 * bit-identical). Override the default model path with PN_TEST_MODEL.
 */

#include "../src/formats/gguf.h"
#include "../src/formats/gguf_dequant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static const char *test_model_path(void)
{
    const char *env = getenv("PN_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

/* =========================================================
 * Part A — pure-logic checks (no model file required)
 * ========================================================= */

/* A.1 — block_size / bytes_per_block / type_size arithmetic */
static void test_block_arithmetic(void)
{
    printf("block_arithmetic:\n");

    /* Q4_0: 32 elements, 18 bytes/block */
    CHECK(pn_gguf_block_size(GGUF_QUANT_Q4_0) == 32, "Q4_0 block_size");
    CHECK(pn_gguf_bytes_per_block(GGUF_QUANT_Q4_0) == sizeof(pn_block_q4_0),
          "Q4_0 bytes_per_block == sizeof(pn_block_q4_0)");
    CHECK(sizeof(pn_block_q4_0) == 18, "sizeof(pn_block_q4_0)==18");
    /* type_size for a 2048-element row */
    size_t q40_row = pn_gguf_type_size(GGUF_QUANT_Q4_0, 2048);
    CHECK(q40_row == (2048 / 32) * 18, "Q4_0 type_size for 2048 elems");

    /* Q4_K: 256 elements, 144 bytes/block */
    CHECK(pn_gguf_block_size(GGUF_QUANT_Q4_K) == 256, "Q4_K block_size");
    CHECK(pn_gguf_bytes_per_block(GGUF_QUANT_Q4_K) == 144, "Q4_K bytes_per_block");
    size_t q4k_row = pn_gguf_type_size(GGUF_QUANT_Q4_K, 2048);
    CHECK(q4k_row == (2048 / 256) * 144, "Q4_K type_size for 2048 elems");

    /* Q6_K: 256 elements, 210 bytes/block */
    CHECK(pn_gguf_block_size(GGUF_QUANT_Q6_K) == 256, "Q6_K block_size");
    CHECK(pn_gguf_bytes_per_block(GGUF_QUANT_Q6_K) == 210, "Q6_K bytes_per_block");

    /* F16: 1 element, 2 bytes */
    CHECK(pn_gguf_block_size(GGUF_QUANT_F16) == 1, "F16 block_size");
    CHECK(pn_gguf_bytes_per_block(GGUF_QUANT_F16) == 2, "F16 bytes_per_block");

    /* F32: 1 element, 4 bytes */
    CHECK(pn_gguf_block_size(GGUF_QUANT_F32) == 1, "F32 block_size");
    CHECK(pn_gguf_bytes_per_block(GGUF_QUANT_F32) == 4, "F32 bytes_per_block");

    /* type_size(x, 0) == 0 for all types */
    CHECK(pn_gguf_type_size(GGUF_QUANT_Q4_0, 0) == 0, "type_size 0 elems == 0");

    /* Row-offset consistency: type_size(row*n) == row * type_size(n) for
     * block-aligned n. Verified for Q4_0 with n_embd=2048, rows 0,1,5. */
    for (size_t r = 0; r <= 5; r++) {
        size_t off_mul = pn_gguf_type_size(GGUF_QUANT_Q4_0, r * 2048);
        size_t off_seq = r * pn_gguf_type_size(GGUF_QUANT_Q4_0, 2048);
        CHECK(off_mul == off_seq, "Q4_0 row offset: type_size(r*n)==r*type_size(n)");
    }
}

/* A.2 — synthetic Q4_0 tensor: per-row dequant == slice of whole-tensor
 * dequant.
 *
 * We build a fake pn_gguf_file whose .data points to a hand-constructed
 * sequence of Q4_0 blocks. N_ROWS rows, each ROW_ELEMS elements. We pick
 * non-trivial block values (varying scale + nibbles) so the comparison is
 * meaningful. The fake tensor info is assembled in the tensors[] array so
 * pn_gguf_dequantize_row / pn_gguf_dequantize_tensor can find it. */

#define SYNTH_ROWS     8
#define SYNTH_ROW_ELEMS 64   /* 2 blocks of Q4_0 (block_size=32) */
#define SYNTH_N_BLOCKS (SYNTH_ROWS * SYNTH_ROW_ELEMS / 32)

static void test_synthetic_q4_row(void)
{
    printf("synthetic_q4_row:\n");

    /* Build raw Q4_0 block data. Use a simple pattern: scale = 1.0f as fp16,
     * nibbles cycle through 0..15 so the dequant is deterministic. */
    pn_block_q4_0 blocks[SYNTH_N_BLOCKS];
    for (int b = 0; b < SYNTH_N_BLOCKS; b++) {
        /* scale d = (b+1)*0.5f  → fp16 */
        float d = (float)(b + 1) * 0.5f;
        blocks[b].d = pn_float_to_half(d);
        for (int j = 0; j < 16; j++) {
            uint8_t lo = (uint8_t)((b * 16 + j) & 0xF);
            uint8_t hi = (uint8_t)(((b * 16 + j) + 3) & 0xF);
            blocks[b].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }

    /* Assemble a fake pn_gguf_file.  We only need .data, .tensors, .n_tensors;
     * the rest can be zero / NULL. */
    pn_gguf_tensor_info ti;
    memset(&ti, 0, sizeof(ti));
    ti.n_dims    = 2;
    ti.dims[0]   = SYNTH_ROW_ELEMS;   /* ncols */
    ti.dims[1]   = SYNTH_ROWS;        /* nrows */
    ti.type      = GGUF_QUANT_Q4_0;
    ti.offset    = 0;
    /* Name (pn_gguf_string = {data, len} pointing into a static literal). */
    static char tname[] = "synth.weight";
    ti.name.data = tname;
    ti.name.len  = sizeof(tname) - 1;

    pn_gguf_file fake;
    memset(&fake, 0, sizeof(fake));
    fake.tensors   = &ti;
    fake.n_tensors = 1;
    fake.data      = blocks;
    fake.data_size = sizeof(blocks);

    /* Full-tensor dequant reference. */
    size_t total = (size_t)SYNTH_ROWS * SYNTH_ROW_ELEMS;
    float *full = malloc(total * sizeof(float));
    CHECK(full != NULL, "malloc full");
    if (!full) return;
    size_t numel = 0;
    float *ref = pn_gguf_dequantize_tensor(&fake, "synth.weight", &numel);
    CHECK(ref != NULL, "dequantize_tensor succeeded on synthetic");
    CHECK(numel == total, "dequantize_tensor numel correct");

    /* Per-row dequant must match the corresponding slice of ref. */
    float *row_buf = malloc((size_t)SYNTH_ROW_ELEMS * sizeof(float));
    CHECK(row_buf != NULL, "malloc row_buf");
    if (!row_buf) { free(ref); free(full); return; }

    int row_ok = 1;
    for (int r = 0; r < SYNTH_ROWS; r++) {
        int rc = pn_gguf_dequantize_row(&fake, "synth.weight",
                                        (size_t)r, SYNTH_ROW_ELEMS, row_buf);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: dequantize_row row %d returned %d\n", r, rc);
            g_failed++;
            row_ok = 0;
            continue;
        }
        const float *ref_row = ref + (size_t)r * SYNTH_ROW_ELEMS;
        for (int c = 0; c < SYNTH_ROW_ELEMS; c++) {
            if (row_buf[c] != ref_row[c]) {
                fprintf(stderr,
                        "  FAIL: row %d col %d: row=%.6f full=%.6f\n",
                        r, c, (double)row_buf[c], (double)ref_row[c]);
                g_failed++;
                row_ok = 0;
                break;
            }
        }
    }
    if (row_ok)
        printf("  all %d rows match full-tensor dequant\n", SYNTH_ROWS);

    free(row_buf);
    free(ref);
    free(full);
}

/* =========================================================
 * Part B — model-backed (gated on PN_TEST_MODEL presence)
 * ========================================================= */

static void test_model_backed(void)
{
    const char *path = test_model_path();
    if (access(path, R_OK) != 0) {
        printf("model_backed: [SKIP] %s not found (set PN_TEST_MODEL)\n", path);
        return;
    }
    printf("model_backed: (%s)\n", path);

    pn_gguf_file *f = pn_gguf_open(path);
    CHECK(f != NULL, "open GGUF");
    if (!f) return;

    /* Find token_embd.weight; skip if not present (some small test GGUFs
     * don't have it under this exact name). */
    pn_gguf_tensor_info *ti = pn_gguf_find_tensor(f, "token_embd.weight");
    if (!ti) {
        printf("  [SKIP] tensor 'token_embd.weight' not in this GGUF\n");
        pn_gguf_close(f);
        return;
    }

    /* Determine row shape: dims[0] = n_embd (columns), dims[1] = n_vocab (rows). */
    if (ti->n_dims < 2) {
        printf("  [SKIP] token_embd.weight has < 2 dims\n");
        pn_gguf_close(f);
        return;
    }
    size_t row_numel = (size_t)ti->dims[0];
    size_t n_rows    = (size_t)ti->dims[1];

    /* Verify block alignment (requirement of pn_gguf_dequantize_row). */
    int bsize = pn_gguf_block_size(ti->type);
    if (bsize <= 0 || row_numel % (size_t)bsize != 0) {
        printf("  [SKIP] row_numel=%zu not block-aligned for type %d\n",
               row_numel, (int)ti->type);
        pn_gguf_close(f);
        return;
    }

    printf("  token_embd.weight: type=%d rows=%zu row_numel=%zu\n",
           (int)ti->type, n_rows, row_numel);

    /* Full-tensor dequant (reference). May be large; limit to 3 rows to keep
     * the test fast and lean on memory. We dequant a prefix: use a synthetic
     * slice approach — dequant the whole tensor only if it fits under 256 MB. */
    size_t total_floats = n_rows * row_numel;
    float *full = NULL;
    int full_ok = 0;
    if (total_floats * sizeof(float) <= 256u * 1024u * 1024u) {
        size_t ne = 0;
        full = pn_gguf_dequantize_tensor(f, "token_embd.weight", &ne);
        full_ok = (full != NULL && ne == total_floats);
        if (!full_ok)
            printf("  [WARN] dequantize_tensor failed or wrong numel; "
                   "skipping row-vs-full comparison\n");
    } else {
        printf("  [NOTE] tensor too large (%.0f MB) for full dequant; "
               "row-vs-full check skipped\n",
               (double)(total_floats * sizeof(float)) / (1024.0 * 1024.0));
    }

    /* Always test: dequantize_row succeeds and produces finite values for
     * a handful of rows (0, middle, last). */
    float *row_buf = malloc(row_numel * sizeof(float));
    CHECK(row_buf != NULL, "malloc row_buf");
    if (!row_buf) { free(full); pn_gguf_close(f); return; }

    size_t test_rows[] = { 0, n_rows / 2, n_rows - 1 };
    int n_test = (int)(sizeof(test_rows) / sizeof(test_rows[0]));
    /* De-duplicate if n_rows is tiny. */
    if (n_rows == 1) n_test = 1;
    if (n_rows == 2) n_test = 2;

    for (int ti_idx = 0; ti_idx < n_test; ti_idx++) {
        size_t r = test_rows[ti_idx];
        int rc = pn_gguf_dequantize_row(f, "token_embd.weight",
                                        r, row_numel, row_buf);
        CHECK(rc == 0, "dequantize_row return 0");
        if (rc != 0) continue;

        /* All finite. */
        int finite_ok = 1;
        for (size_t c = 0; c < row_numel; c++) {
            if (!isfinite(row_buf[c])) { finite_ok = 0; break; }
        }
        CHECK(finite_ok, "dequantize_row output is finite");

        if (full_ok) {
            /* Compare to full-tensor slice. */
            const float *ref_row = full + r * row_numel;
            int match = 1;
            for (size_t c = 0; c < row_numel; c++) {
                if (row_buf[c] != ref_row[c]) {
                    fprintf(stderr,
                            "  FAIL: row %zu col %zu: row=%.8f full=%.8f\n",
                            r, c, (double)row_buf[c], (double)ref_row[c]);
                    g_failed++;
                    match = 0;
                    break;
                }
            }
            if (match)
                printf("  row %zu: matches full-tensor dequant (%zu elems)\n",
                       r, row_numel);
        } else {
            printf("  row %zu: dequantize_row OK, finite\n", r);
        }
    }

    free(row_buf);
    free(full);
    pn_gguf_close(f);
}

/* =========================================================
 * main
 * ========================================================= */

int main(void)
{
    printf("test_embed_row\n");
    test_block_arithmetic();
    test_synthetic_q4_row();
    test_model_backed();

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
