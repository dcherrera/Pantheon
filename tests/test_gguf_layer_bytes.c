/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_gguf_layer_bytes.c — per-layer weight accounting for the multi-die
 * admission gate. Self-contained: builds a synthetic tensor table in memory, no
 * GGUF file, no model, no GPU.
 *
 * The regression it pins (2026-08-21): gpu_admit_die_share projected a die's
 * resident weights as `mmap_size * n_layers_here / n_layer`. The file also
 * carries token_embd (never uploaded — the embed is a host row copy) and
 * output.weight (one die's, not every die's), so every die was charged for
 * both. On Qwen3.6-35B-A3B Q4_K_M that over-projected each die by ~15% and
 * refused a 2-die load that fits with ~1.3 GiB of headroom per die.
 *
 * Covers:
 *   - non-layer tensors are excluded from a layer range
 *   - ranges partition: [0,split) + [split,n) == the whole layer stack
 *   - mixed quants per layer sum by tensor, not by layer count
 *   - the proration bug is measurably worse than the tensor-table sum
 *   - blk.<i>. name parsing rejects near-misses
 *   - absent / empty / malformed inputs return 0, not garbage
 */

#include "../src/formats/gguf_layer_bytes.h"
#include "../src/formats/gguf_dequant.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define MAX_T 256

/* A synthetic GGUF whose tensor names own their storage, so name.data is the
 * NUL-terminated char* pn_gguf_find_tensor's strcmp expects. */
typedef struct {
    pn_gguf_file file;
    pn_gguf_tensor_info t[MAX_T];
    char names[MAX_T][64];
} fake_gguf;

static void fg_init(fake_gguf *f)
{
    memset(f, 0, sizeof(*f));
    f->file.tensors = f->t;
    f->file.n_tensors = 0;
}

static void fg_add(fake_gguf *f, const char *name, pn_gguf_quant type,
                   uint64_t rows, uint64_t cols)
{
    uint64_t i = f->file.n_tensors;
    if (i >= MAX_T) return;
    snprintf(f->names[i], sizeof(f->names[i]), "%s", name);
    f->t[i].name.data = f->names[i];
    f->t[i].name.len  = strlen(f->names[i]);
    f->t[i].type = type;
    f->t[i].n_dims = 2;
    f->t[i].dims[0] = cols;
    f->t[i].dims[1] = rows;
    f->file.n_tensors = i + 1;
}

static uint64_t bytes_of(pn_gguf_quant type, uint64_t rows, uint64_t cols)
{
    return (uint64_t)pn_gguf_type_size(type, (size_t)(rows * cols));
}

/* 8 layers shaped like the Q3_K_M mini: the merged ffn_down_exps is Q6_K on the
 * back half and Q4_K on the front, which is exactly the asymmetry that makes a
 * layer-COUNT split and a layer-BYTE split different numbers. */
#define N_LAYER 8
#define SPLIT   4
#define ATTN_ROWS 32
#define ATTN_COLS 256
#define FFN_ROWS  64
#define FFN_COLS  256

static void build_model(fake_gguf *f)
{
    fg_init(f);
    fg_add(f, "token_embd.weight", GGUF_QUANT_Q6_K, 1024, 256);
    for (int i = 0; i < N_LAYER; i++) {
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.attn_qkv.weight", i);
        fg_add(f, n, GGUF_QUANT_Q4_K, ATTN_ROWS, ATTN_COLS);
        snprintf(n, sizeof(n), "blk.%d.ffn_down_exps.weight", i);
        fg_add(f, n, (i < SPLIT) ? GGUF_QUANT_Q4_K : GGUF_QUANT_Q6_K,
               FFN_ROWS, FFN_COLS);
    }
    fg_add(f, "output_norm.weight", GGUF_QUANT_F32, 1, 256);
    fg_add(f, "output.weight", GGUF_QUANT_Q6_K, 1024, 256);
}

static void test_excludes_non_layer_tensors(void)
{
    printf("excludes_non_layer_tensors:\n");
    fake_gguf f;
    build_model(&f);

    uint64_t front_expect =
        SPLIT * (bytes_of(GGUF_QUANT_Q4_K, ATTN_ROWS, ATTN_COLS) +
                 bytes_of(GGUF_QUANT_Q4_K, FFN_ROWS, FFN_COLS));
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, SPLIT) == front_expect,
          "front range is its own tensors only");

    /* token_embd + output.weight are the two big non-layer tensors; neither may
     * appear in any layer range. */
    uint64_t whole = pn_gguf_layer_range_bytes(&f.file, 0, N_LAYER);
    uint64_t embd  = pn_gguf_tensor_file_bytes(&f.file, "token_embd.weight");
    uint64_t head  = pn_gguf_tensor_file_bytes(&f.file, "output.weight");
    CHECK(embd > 0 && head > 0, "the non-layer tensors are really there");

    uint64_t sum_all = 0;
    for (uint64_t i = 0; i < f.file.n_tensors; i++)
        sum_all += bytes_of(f.t[i].type, f.t[i].dims[1], f.t[i].dims[0]);
    uint64_t norm = pn_gguf_tensor_file_bytes(&f.file, "output_norm.weight");
    CHECK(whole + embd + head + norm == sum_all,
          "layers + non-layer tensors == every tensor, no double count");
}

static void test_ranges_partition(void)
{
    printf("ranges_partition:\n");
    fake_gguf f;
    build_model(&f);

    uint64_t front = pn_gguf_layer_range_bytes(&f.file, 0, SPLIT);
    uint64_t back  = pn_gguf_layer_range_bytes(&f.file, SPLIT, N_LAYER);
    uint64_t whole = pn_gguf_layer_range_bytes(&f.file, 0, N_LAYER);
    CHECK(front + back == whole, "the two die shares partition the stack");
    CHECK(front > 0 && back > 0, "both shares are non-empty");

    /* Same layer COUNT, different BYTES — the Q6_K back half is heavier. This
     * is why an equal-count split is not automatically an equal-byte split. */
    CHECK(back > front, "Q6_K back half outweighs the Q4_K front half");

    CHECK(pn_gguf_layer_range_bytes(&f.file, 3, 3) == 0, "empty range is 0");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 5, 2) == 0, "inverted range is 0");
    CHECK(pn_gguf_layer_range_bytes(&f.file, N_LAYER, N_LAYER + 4) == 0,
          "range past the last layer is 0");
    CHECK(pn_gguf_layer_range_bytes(&f.file, -3, SPLIT) == front,
          "negative lo clamps to 0 rather than swallowing non-layer tensors");
}

static void test_proration_over_projects(void)
{
    printf("proration_over_projects:\n");
    fake_gguf f;
    build_model(&f);

    /* What the gate used to compute: a layer-count share of the whole file. */
    uint64_t file_bytes = 0;
    for (uint64_t i = 0; i < f.file.n_tensors; i++)
        file_bytes += bytes_of(f.t[i].type, f.t[i].dims[1], f.t[i].dims[0]);

    uint64_t prorated_front = file_bytes * SPLIT / N_LAYER;
    uint64_t real_front     = pn_gguf_layer_range_bytes(&f.file, 0, SPLIT);
    CHECK(prorated_front > real_front,
          "the old proration charges die A for tensors it does not hold");

    /* And it is not a rounding difference: the excess is the non-layer weight,
     * which on this fixture is a large fraction of a die's real share. */
    uint64_t excess = prorated_front - real_front;
    CHECK(excess * 10 > real_front,
          "the over-projection is >10% of the die's real weight, not noise");

    /* The fixed accounting: die A = its layers, die B = its layers + the head. */
    uint64_t head = pn_gguf_tensor_file_bytes(&f.file, "output.weight") +
                    pn_gguf_tensor_file_bytes(&f.file, "output_norm.weight");
    uint64_t fixed_a = real_front;
    uint64_t fixed_b = pn_gguf_layer_range_bytes(&f.file, SPLIT, N_LAYER) + head;
    uint64_t embd = pn_gguf_tensor_file_bytes(&f.file, "token_embd.weight");
    CHECK(fixed_a + fixed_b + embd == file_bytes,
          "the two dies plus the never-uploaded embedding account for the file");
    CHECK(fixed_a < prorated_front, "die A's projection drops");
}

static void test_name_parsing(void)
{
    printf("name_parsing:\n");
    fake_gguf f;
    fg_init(&f);
    fg_add(&f, "blk.2.real.weight",  GGUF_QUANT_Q4_K, 8, 256);
    fg_add(&f, "blk.2suffix.weight", GGUF_QUANT_Q4_K, 8, 256);  /* no dot */
    fg_add(&f, "blkX.2.weight",      GGUF_QUANT_Q4_K, 8, 256);  /* not blk. */
    fg_add(&f, "blk..weight",        GGUF_QUANT_Q4_K, 8, 256);  /* no index */
    fg_add(&f, "blk.weight",         GGUF_QUANT_Q4_K, 8, 256);
    fg_add(&f, "ablk.2.weight",      GGUF_QUANT_Q4_K, 8, 256);  /* prefix only */

    uint64_t one = bytes_of(GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 64) == one,
          "only the real blk.<i>. tensor counts");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 2, 3) == one, "layer 2 exactly");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 2) == 0, "layers 0-1 are empty");

    /* Multi-digit indices are real: a 40-layer model has blk.12 and blk.7. */
    fg_init(&f);
    fg_add(&f, "blk.7.w",  GGUF_QUANT_Q4_K, 8, 256);
    fg_add(&f, "blk.12.w", GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 10) == one, "blk.12 is not in [0,10)");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 10, 20) == one, "blk.12 is in [10,20)");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 20) == 2 * one, "both in [0,20)");
}

static void test_degenerate_inputs(void)
{
    printf("degenerate_inputs:\n");
    CHECK(pn_gguf_layer_range_bytes(NULL, 0, 8) == 0, "NULL file -> 0");
    CHECK(pn_gguf_tensor_file_bytes(NULL, "output.weight") == 0, "NULL file -> 0");

    fake_gguf f;
    build_model(&f);
    CHECK(pn_gguf_tensor_file_bytes(&f.file, NULL) == 0, "NULL name -> 0");
    CHECK(pn_gguf_tensor_file_bytes(&f.file, "no.such.tensor") == 0,
          "absent tensor -> 0, and the caller reads 0 as 'unknown'");

    /* A tensor with no dims, a zero dim, or too many dims is malformed; report
     * 0 rather than a wrapped product. */
    fg_init(&f);
    fg_add(&f, "blk.0.w", GGUF_QUANT_Q4_K, 8, 256);
    f.t[0].n_dims = 0;
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) == 0, "0 dims -> 0");
    f.t[0].n_dims = 2; f.t[0].dims[0] = 0;
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) == 0, "zero dim -> 0");

    /* dims[] is 4 wide. Fill all four so the only thing that can stop a 5-dim
     * read is the bound itself — otherwise a zero dim short-circuits the loop
     * and the guard is never exercised. */
    f.t[0].dims[0] = 256; f.t[0].dims[1] = 8;
    f.t[0].dims[2] = 1;   f.t[0].dims[3] = 1;
    f.t[0].n_dims = 4;
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) ==
          bytes_of(GGUF_QUANT_Q4_K, 8, 256), "n_dims == 4 is accepted");
    f.t[0].n_dims = 5;
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) == 0,
          "n_dims > 4 -> 0, never a read past dims[3]");

    /* An unsupported quant has no known block size; it contributes 0 rather
     * than a bogus size, and does not poison its neighbours. */
    /* An unrecognised quant must make the WHOLE range unmeasurable, not
     * silently contribute 0. I asserted the opposite here and it was wrong:
     * this sum is a safety projection, so "I cannot size this" must never read
     * downstream as "it is free". BF16 shipped in the enum without a block-size
     * entry and every expert tensor sized as zero, projecting 10 MiB per die
     * for a model needing ~8,300. 0 makes the caller fall back to its prorated
     * estimate -- wrong, but never optimistic. */
    fg_init(&f);
    fg_add(&f, "blk.0.w", (pn_gguf_quant)9999, 8, 256);
    fg_add(&f, "blk.1.w", GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) == 0,
          "an unsizeable quant refuses the whole range rather than costing 0");

    /* BF16 specifically, because that is the one that got shipped half-added. */
    fg_init(&f);
    fg_add(&f, "blk.0.w", GGUF_QUANT_BF16, 8, 256);
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, 8) == 8 * 256 * 2,
          "BF16 sizes as 2 bytes per element");
}

static void test_layer_count(void)
{
    printf("layer_count:\n");
    fake_gguf f;
    build_model(&f);
    CHECK(pn_gguf_layer_count(&f.file) == N_LAYER,
          "layer count comes off the tensor table, no arch prefix needed");
    CHECK(pn_gguf_layer_count(NULL) == 0, "NULL file -> 0");

    /* Non-layer tensors must not raise the count, and a gap must not lower it:
     * the count is max index + 1, which is what a [0,n) range needs. */
    fg_init(&f);
    fg_add(&f, "token_embd.weight", GGUF_QUANT_Q6_K, 8, 256);
    fg_add(&f, "output.weight", GGUF_QUANT_Q6_K, 8, 256);
    CHECK(pn_gguf_layer_count(&f.file) == 0, "no blk.* tensors -> 0 layers");
    fg_add(&f, "blk.0.w", GGUF_QUANT_Q4_K, 8, 256);
    fg_add(&f, "blk.39.w", GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_count(&f.file) == 40, "highest index + 1");

    /* A trailing MTP / NextN block is NOT a layer. Qwen3.6-35B-A3B ships
     * block_count=41 with a speculative-draft head the forward never runs and
     * never uploads; counting it puts weight on a die that will not hold it.
     * Same rule the loader applies (qwen35moe.c). */
    fg_init(&f);
    for (int i = 0; i < 41; i++) {
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.w", i);
        fg_add(&f, n, GGUF_QUANT_Q4_K, 8, 256);
    }
    CHECK(pn_gguf_layer_count(&f.file) == 41, "41 plain blocks are 41 layers");
    fg_add(&f, "blk.40.nextn.eh_proj.weight", GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_count(&f.file) == 40,
          "the trailing nextn block is dropped");
    CHECK(pn_gguf_layer_range_bytes(&f.file, 0, pn_gguf_layer_count(&f.file)) ==
          40 * bytes_of(GGUF_QUANT_Q4_K, 8, 256),
          "and a [0,count) range stops before it");

    /* Only the LAST block, and only for nextn: an eh_proj in the middle is a
     * real layer's tensor, not a head to drop. */
    fg_init(&f);
    for (int i = 0; i < 41; i++) {
        char n[64];
        snprintf(n, sizeof(n), "blk.%d.w", i);
        fg_add(&f, n, GGUF_QUANT_Q4_K, 8, 256);
    }
    fg_add(&f, "blk.7.nextn.eh_proj.weight", GGUF_QUANT_Q4_K, 8, 256);
    CHECK(pn_gguf_layer_count(&f.file) == 41,
          "a nextn tensor on a middle block drops nothing");
}

static void test_tied_head_sizing(void)
{
    printf("tied_head_sizing:\n");
    /* A TIED model carries no output.weight — the LM head is token_embd
     * materialized to F32 at load. Sizing it from the packed tensor would
     * under-project the die that holds it by ~4x, so the numel is the number a
     * caller needs. */
    fake_gguf f;
    fg_init(&f);
    fg_add(&f, "blk.0.w", GGUF_QUANT_Q4_K, 8, 256);
    fg_add(&f, "token_embd.weight", GGUF_QUANT_Q6_K, 1024, 256);

    CHECK(pn_gguf_tensor_file_bytes(&f.file, "output.weight") == 0,
          "a tied model really has no output.weight");
    uint64_t numel = pn_gguf_tensor_numel(&f.file, "token_embd.weight");
    CHECK(numel == 1024 * 256, "numel is the element count, not bytes");

    uint64_t packed = pn_gguf_tensor_file_bytes(&f.file, "token_embd.weight");
    uint64_t f32    = numel * sizeof(float);
    CHECK(f32 > packed * 4, "the F32 head is >4x the packed embedding");

    CHECK(pn_gguf_tensor_numel(&f.file, "no.such.tensor") == 0, "absent -> 0");
    CHECK(pn_gguf_tensor_numel(NULL, "token_embd.weight") == 0, "NULL -> 0");
}

int main(void)
{
    printf("test_gguf_layer_bytes\n");
    test_excludes_non_layer_tensors();
    test_ranges_partition();
    test_proration_over_projects();
    test_name_parsing();
    test_degenerate_inputs();
    test_layer_count();
    test_tied_head_sizing();

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
