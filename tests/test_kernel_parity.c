/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_kernel_parity.c — GPU-vs-CPU parity harness for the qwen35moe kernels
 *
 * Runs each new Foundry CUDA kernel (Q3_K matvec, partial NEOX RoPE,
 * QK-RMSNorm, MoE routing) through the fc_cuda dlopen backend and diffs
 * its output against the trusted CPU-reference golden vectors in
 * the tests/golden directory. Prints, per kernel:
 *
 *   [PASS]/[FAIL] <kernel>  max_abs_err=...  max_rel_err=...
 *
 * MoE top-k indices are integer and must match EXACTLY (golden README).
 *
 * Gated on fc_available(): with no CUDA driver (e.g. this Mac) it prints
 * "[SKIP] no CUDA" and exits 0, so the harness compiles, links, and runs
 * everywhere; the real GPU diffs happen when the operator runs it on the
 * K80 host.
 *
 * Golden format (tests/golden/README.md):
 *   line 1: "PNGOLDEN1 <name> <dtype> <count> seed=0x9E3779B9\n"
 *   then:   <count> little-endian <dtype> values (f32 / i32 / u8).
 *
 * @status GPU diffs pending K80 host; SKIP path validated on Mac.
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "barswap/barswap.h"

/* Bring nvidia up after a die is acquired (it's blacklisted; each submodule
 * needs --ignore-install). Mirrors compute_persist's load_nvidia. */
static int parity_load_nvidia(void)
{
    if (system("modprobe --ignore-install nvidia 2>/dev/null") != 0) return -1;
    sleep(1);
    if (system("modprobe --ignore-install nvidia_uvm 2>/dev/null") != 0) return -1;
    sleep(1);
    return 0;
}

/* qwen35moe config constants (golden README "Config constants used"). */
#define PARITY_N_EMBD          2048
#define PARITY_N_HEAD          16
#define PARITY_N_HEAD_KV       2
#define PARITY_HEAD_DIM        256
#define PARITY_ROPE_DIM        64
#define PARITY_ROPE_THETA      1e7f
#define PARITY_RMS_NORM_EPS    1e-6f
#define PARITY_N_EXPERT        256
#define PARITY_EXPERT_USED     8
#define PARITY_EXPERT_WT_SCALE 1.0f

/* Q3_K matvec test shape (golden README Kernel 1). */
#define PARITY_Q3K_ROWS        24
#define PARITY_Q3K_COLS        1024
#define PARITY_Q3K_BLOCK_BYTES 110   /* sizeof(pn_block_q3_k) — packed, no pad */

/* Gated-Delta-Net SSM dimensions (golden README Kernels 5-7). */
#define PARITY_SSM_N_V_HEADS   32
#define PARITY_SSM_STATE_SIZE  128
#define PARITY_SSM_CONV_DIM    8192
#define PARITY_SSM_CONV_KERNEL 4

/* Tolerances (golden README). fp32 trig/softmax on Kepler. */
#define PARITY_REL_TOL         1e-3f

#define GOLDEN_DIR             "tests/golden/"

/* ============================================================
 * Golden-vector loader
 * ============================================================ */

/* A loaded golden file: the payload past the PNGOLDEN1 header line.
 * `count` is the element count from the header; `bytes` is the raw
 * payload length. Ownership: caller frees `data`. */
typedef struct {
    char    name[64];
    char    dtype[8];
    size_t  count;
    size_t  bytes;
    void   *data;
} golden_vec;

/* sizeof one element for a golden dtype. Returns 0 for unknown dtype. */
static size_t golden_dtype_size(const char *dtype)
{
    if (strcmp(dtype, "f32") == 0) return 4;
    if (strcmp(dtype, "i32") == 0) return 4;
    if (strcmp(dtype, "u8") == 0)  return 1;
    return 0;
}

/* Load a golden .bin (header line + little-endian payload) from GOLDEN_DIR.
 * Returns 0 on success and fills *out; caller frees out->data. The host is
 * assumed little-endian (x86_64 / aarch64), matching the file format. */
static int golden_load(const char *filename, golden_vec *out)
{
    char path[256];
    int n = snprintf(path, sizeof(path), "%s%s", GOLDEN_DIR, filename);
    if (n < 0 || (size_t)n >= sizeof(path))
        return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[parity] cannot open %s\n", path);
        return -1;
    }

    memset(out, 0, sizeof(*out));
    int rc = -1;

    /* Header line: PNGOLDEN1 <name> <dtype> <count> seed=0x... */
    char magic[16] = {0};
    unsigned long count = 0;
    if (fscanf(f, "%15s %63s %7s %lu", magic, out->name, out->dtype, &count) != 4)
        goto cleanup;
    if (strcmp(magic, "PNGOLDEN1") != 0)
        goto cleanup;

    /* Consume the rest of the header line up to and including the newline. */
    int c;
    while ((c = fgetc(f)) != '\n' && c != EOF)
        ;

    size_t elem = golden_dtype_size(out->dtype);
    if (elem == 0)
        goto cleanup;

    out->count = (size_t)count;
    out->bytes = out->count * elem;
    out->data = malloc(out->bytes ? out->bytes : 1);
    if (!out->data)
        goto cleanup;

    if (fread(out->data, 1, out->bytes, f) != out->bytes)
        goto cleanup;

    rc = 0;

cleanup:
    fclose(f);
    if (rc != 0) {
        free(out->data);
        out->data = NULL;
    }
    return rc;
}

static void golden_free(golden_vec *g)
{
    if (g) {
        free(g->data);
        g->data = NULL;
    }
}

/* ============================================================
 * Comparison + reporting
 * ============================================================ */

static int g_failures = 0;

/* Max abs and max relative error between two f32 arrays. rel uses
 * |a-b| / max(|gold|, tiny) so near-zero golden values do not blow up. */
static void f32_errors(const float *got, const float *gold, size_t n,
                       float *max_abs, float *max_rel)
{
    float ma = 0.0f, mr = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(got[i] - gold[i]);
        if (diff > ma) ma = diff;
        float denom = fabsf(gold[i]);
        float rel = diff / (denom > 1e-6f ? denom : 1e-6f);
        if (rel > mr) mr = rel;
    }
    *max_abs = ma;
    *max_rel = mr;
}

/* Absolute-error floor: an element this close in absolute terms is correct
 * regardless of its relative error (avoids near-zero-denominator inflation,
 * e.g. a ~2e-5 state element with a 2e-8 fp32 reduction-order diff). */
#define PARITY_ABS_TOL 1e-5f

/* Report an f32 kernel result and tally failures. Pass = within EITHER the
 * relative OR the absolute tolerance — the standard numerical-parity criterion. */
static void report_f32(const char *kernel, const float *got,
                       const float *gold, size_t n, float rel_tol)
{
    float max_abs, max_rel;
    f32_errors(got, gold, n, &max_abs, &max_rel);
    bool pass = (max_rel <= rel_tol) || (max_abs <= PARITY_ABS_TOL);
    if (!pass) g_failures++;
    printf("  [%s] %-22s max_abs_err=%.3e  max_rel_err=%.3e\n",
           pass ? "PASS" : "FAIL", kernel, (double)max_abs, (double)max_rel);
}

/* ============================================================
 * Per-kernel parity tests
 * ============================================================ */

/* Kernel 1: Q3_K dequant + matvec. Loads the raw quantized weight bytes,
 * the fp32 input vector, and the golden output; runs fc_matvec_q3k and
 * diffs. row_stride = (cols/256) * 110, matching the CPU pn_qmv layout. */
static void test_q3k_matvec(void)
{
    golden_vec w = {0}, x = {0}, out = {0};
    if (golden_load("q3k_matvec_weight.bin", &w) != 0 ||
        golden_load("q3k_matvec_input.bin", &x) != 0 ||
        golden_load("q3k_matvec_output.bin", &out) != 0) {
        fprintf(stderr, "[parity] q3k_matvec: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    int rows = PARITY_Q3K_ROWS;
    int cols = PARITY_Q3K_COLS;
    int row_stride = (cols / 256) * PARITY_Q3K_BLOCK_BYTES;

    fc_buffer *d_w = fc_buffer_from_ptr(w.data, w.bytes);
    fc_buffer *d_x = fc_buffer_from_ptr(x.data, x.bytes);
    fc_buffer *d_out = fc_buffer_alloc((size_t)rows * sizeof(float));

    fc_matvec_q3k(d_w, d_x, d_out, rows, cols, row_stride);
    fc_sync();

    float *got = malloc((size_t)rows * sizeof(float));
    fc_buffer_to_cpu(d_out, got);
    report_f32("q3k_matvec", got, (const float *)out.data, (size_t)rows,
               PARITY_REL_TOL);

    free(got);
    fc_buffer_free(d_w);
    fc_buffer_free(d_x);
    fc_buffer_free(d_out);

cleanup:
    golden_free(&w);
    golden_free(&x);
    golden_free(&out);
}

/* Rotate one tensor at one position and diff against its golden.
 * data is rotated in place on the GPU, so it is re-uploaded per call. */
static void rope_one(const char *kernel, const float *input, size_t count,
                     int n_heads, int pos, const char *gold_file)
{
    golden_vec gold = {0};
    if (golden_load(gold_file, &gold) != 0) {
        fprintf(stderr, "[parity] rope: golden load %s failed\n", gold_file);
        g_failures++;
        return;
    }

    fc_buffer *d_data = fc_buffer_from_ptr(input, count * sizeof(float));
    fc_rope_neox_partial(d_data, n_heads, PARITY_HEAD_DIM, PARITY_ROPE_DIM,
                         pos, PARITY_ROPE_THETA);
    fc_sync();

    float *got = malloc(count * sizeof(float));
    fc_buffer_to_cpu(d_data, got);
    report_f32(kernel, got, (const float *)gold.data, count, PARITY_REL_TOL);

    free(got);
    fc_buffer_free(d_data);
    golden_free(&gold);
}

/* Kernel 2: partial NEOX RoPE. Q (16 heads) and K (2 heads) at a couple of
 * positions, including pos 0 (exact identity). */
static void test_rope(void)
{
    golden_vec qin = {0}, kin = {0};
    if (golden_load("rope_q_input.bin", &qin) != 0 ||
        golden_load("rope_k_input.bin", &kin) != 0) {
        fprintf(stderr, "[parity] rope: input load failed\n");
        g_failures++;
        goto cleanup;
    }

    rope_one("rope_q_pos0", (const float *)qin.data, qin.count,
             PARITY_N_HEAD, 0, "rope_q_pos0.bin");
    rope_one("rope_q_pos100", (const float *)qin.data, qin.count,
             PARITY_N_HEAD, 100, "rope_q_pos100.bin");
    rope_one("rope_k_pos0", (const float *)kin.data, kin.count,
             PARITY_N_HEAD_KV, 0, "rope_k_pos0.bin");
    rope_one("rope_k_pos100", (const float *)kin.data, kin.count,
             PARITY_N_HEAD_KV, 100, "rope_k_pos100.bin");

cleanup:
    golden_free(&qin);
    golden_free(&kin);
}

/* Kernel 3: QK-RMSNorm. Per-head RMS over head_dim with the shared weight,
 * applied in place to the 16-head Q layout. */
static void test_qk_rms_norm(void)
{
    golden_vec in = {0}, weight = {0}, out = {0};
    if (golden_load("qknorm_input.bin", &in) != 0 ||
        golden_load("qknorm_weight.bin", &weight) != 0 ||
        golden_load("qknorm_output.bin", &out) != 0) {
        fprintf(stderr, "[parity] qknorm: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    fc_buffer *d_data = fc_buffer_from_ptr(in.data, in.bytes);
    fc_buffer *d_weight = fc_buffer_from_ptr(weight.data, weight.bytes);

    fc_qk_rms_norm(d_data, d_weight, PARITY_N_HEAD, PARITY_HEAD_DIM,
                   PARITY_RMS_NORM_EPS);
    fc_sync();

    float *got = malloc(in.bytes);
    fc_buffer_to_cpu(d_data, got);
    report_f32("qk_rms_norm", got, (const float *)out.data, in.count,
               PARITY_REL_TOL);

    free(got);
    fc_buffer_free(d_data);
    fc_buffer_free(d_weight);

cleanup:
    golden_free(&in);
    golden_free(&weight);
    golden_free(&out);
}

/* Kernel 4a: MoE router top-k. Indices must match EXACTLY; weights within
 * tolerance. fc_moe_router_topk overwrites logits in place with the softmax
 * probs, so a fresh copy is uploaded. */
static void test_moe_router(void)
{
    golden_vec logits = {0}, idx_gold = {0}, wt_gold = {0};
    if (golden_load("moe_router_logits.bin", &logits) != 0 ||
        golden_load("moe_topk_indices.bin", &idx_gold) != 0 ||
        golden_load("moe_topk_weights.bin", &wt_gold) != 0) {
        fprintf(stderr, "[parity] moe_router: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    int k = PARITY_EXPERT_USED;
    fc_buffer *d_logits = fc_buffer_from_ptr(logits.data, logits.bytes);
    fc_buffer *d_idx = fc_buffer_alloc((size_t)k * sizeof(int32_t));
    fc_buffer *d_wt = fc_buffer_alloc((size_t)k * sizeof(float));

    fc_moe_router_topk(d_logits, d_idx, d_wt, PARITY_N_EXPERT, k,
                       PARITY_EXPERT_WT_SCALE);
    fc_sync();

    int32_t *got_idx = malloc((size_t)k * sizeof(int32_t));
    float *got_wt = malloc((size_t)k * sizeof(float));
    fc_buffer_to_cpu(d_idx, got_idx);
    fc_buffer_to_cpu(d_wt, got_wt);

    /* Indices: exact integer match. */
    const int32_t *gold_idx = (const int32_t *)idx_gold.data;
    int idx_mismatch = 0;
    for (int i = 0; i < k; i++)
        if (got_idx[i] != gold_idx[i]) idx_mismatch++;
    if (idx_mismatch) g_failures++;
    printf("  [%s] %-22s mismatches=%d/%d\n",
           idx_mismatch ? "FAIL" : "PASS", "moe_topk_indices",
           idx_mismatch, k);

    /* Weights: relative tolerance. */
    report_f32("moe_topk_weights", got_wt, (const float *)wt_gold.data,
               (size_t)k, PARITY_REL_TOL);

    free(got_idx);
    free(got_wt);
    fc_buffer_free(d_logits);
    fc_buffer_free(d_idx);
    fc_buffer_free(d_wt);

cleanup:
    golden_free(&logits);
    golden_free(&idx_gold);
    golden_free(&wt_gold);
}

/* Kernel 4b: shared-expert sigmoid gate. The host computes the gate logit
 * (gate_w . x) — exactly as the CPU path does — then fc_moe_shared_gate
 * applies sigmoid in place. Seeding shexp_out with 1.0 makes element 0
 * read back as sigmoid(gate_logit), the golden scalar. */
static void test_moe_shared_gate(void)
{
    golden_vec gw = {0}, gx = {0}, gscalar = {0};
    if (golden_load("moe_shared_gate_weight.bin", &gw) != 0 ||
        golden_load("moe_shared_gate_input.bin", &gx) != 0 ||
        golden_load("moe_shared_gate_scalar.bin", &gscalar) != 0) {
        fprintf(stderr, "[parity] moe_shared_gate: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    const float *w = (const float *)gw.data;
    const float *x = (const float *)gx.data;
    float gate_logit = 0.0f;
    for (size_t i = 0; i < gw.count; i++)
        gate_logit += w[i] * x[i];

    int n = PARITY_N_EMBD;
    float *ones = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        ones[i] = 1.0f;

    fc_buffer *d_out = fc_buffer_from_ptr(ones, (size_t)n * sizeof(float));
    fc_moe_shared_gate(d_out, gate_logit, n);
    fc_sync();

    float got = 0.0f;
    fc_buffer_to_cpu(d_out, ones);   /* ones now holds the gated output */
    got = ones[0];
    report_f32("moe_shared_gate", &got, (const float *)gscalar.data, 1,
               PARITY_REL_TOL);

    free(ones);
    fc_buffer_free(d_out);

cleanup:
    golden_free(&gw);
    golden_free(&gx);
    golden_free(&gscalar);
}

/* Kernel 5: SSM per-token delta-rule recurrence. Uploads the initial state,
 * q/k/v, g/beta; runs fc_ssm_recurrence (updates state in place + writes y);
 * diffs both the output y and the updated state against the goldens. */
static void test_ssm_recurrence(void)
{
    golden_vec s_in = {0}, q = {0}, k = {0}, v = {0}, g = {0}, beta = {0};
    golden_vec y_gold = {0}, s_out_gold = {0};
    if (golden_load("ssm_recur_state_in.bin", &s_in) != 0 ||
        golden_load("ssm_recur_q.bin", &q) != 0 ||
        golden_load("ssm_recur_k.bin", &k) != 0 ||
        golden_load("ssm_recur_v.bin", &v) != 0 ||
        golden_load("ssm_recur_g.bin", &g) != 0 ||
        golden_load("ssm_recur_beta.bin", &beta) != 0 ||
        golden_load("ssm_recur_y.bin", &y_gold) != 0 ||
        golden_load("ssm_recur_state_out.bin", &s_out_gold) != 0) {
        fprintf(stderr, "[parity] ssm_recurrence: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    int H = PARITY_SSM_N_V_HEADS;
    int S = PARITY_SSM_STATE_SIZE;
    float scale = 1.0f / sqrtf((float)S);

    fc_buffer *d_s = fc_buffer_from_ptr(s_in.data, s_in.bytes);
    fc_buffer *d_q = fc_buffer_from_ptr(q.data, q.bytes);
    fc_buffer *d_k = fc_buffer_from_ptr(k.data, k.bytes);
    fc_buffer *d_v = fc_buffer_from_ptr(v.data, v.bytes);
    fc_buffer *d_g = fc_buffer_from_ptr(g.data, g.bytes);
    fc_buffer *d_beta = fc_buffer_from_ptr(beta.data, beta.bytes);
    fc_buffer *d_y = fc_buffer_alloc((size_t)H * S * sizeof(float));

    fc_ssm_recurrence(d_s, d_q, d_k, d_v, d_g, d_beta, d_y, H, S, scale);
    fc_sync();

    float *got_y = malloc((size_t)H * S * sizeof(float));
    float *got_s = malloc(s_in.bytes);
    fc_buffer_to_cpu(d_y, got_y);
    fc_buffer_to_cpu(d_s, got_s);   /* state was updated in place */
    report_f32("ssm_recur_y", got_y, (const float *)y_gold.data,
               (size_t)H * S, PARITY_REL_TOL);
    report_f32("ssm_recur_state", got_s, (const float *)s_out_gold.data,
               s_out_gold.count, PARITY_REL_TOL);

    free(got_y);
    free(got_s);
    fc_buffer_free(d_s);
    fc_buffer_free(d_q);
    fc_buffer_free(d_k);
    fc_buffer_free(d_v);
    fc_buffer_free(d_g);
    fc_buffer_free(d_beta);
    fc_buffer_free(d_y);

cleanup:
    golden_free(&s_in);
    golden_free(&q);
    golden_free(&k);
    golden_free(&v);
    golden_free(&g);
    golden_free(&beta);
    golden_free(&y_gold);
    golden_free(&s_out_gold);
}

/* Kernel 6: SSM causal conv1d + SiLU. Uploads qkv, weight, conv_state; runs
 * fc_ssm_conv1d_silu (writes out, rolls conv_state in place); diffs out and
 * the rolled conv_state against the goldens. */
static void test_ssm_conv1d(void)
{
    golden_vec qkv = {0}, w = {0}, st_in = {0}, out_gold = {0}, st_out_gold = {0};
    if (golden_load("ssm_conv_qkv.bin", &qkv) != 0 ||
        golden_load("ssm_conv_weight.bin", &w) != 0 ||
        golden_load("ssm_conv_state_in.bin", &st_in) != 0 ||
        golden_load("ssm_conv_out.bin", &out_gold) != 0 ||
        golden_load("ssm_conv_state_out.bin", &st_out_gold) != 0) {
        fprintf(stderr, "[parity] ssm_conv1d: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    int C = PARITY_SSM_CONV_DIM;
    int kk = PARITY_SSM_CONV_KERNEL;

    fc_buffer *d_qkv = fc_buffer_from_ptr(qkv.data, qkv.bytes);
    fc_buffer *d_w = fc_buffer_from_ptr(w.data, w.bytes);
    fc_buffer *d_state = fc_buffer_from_ptr(st_in.data, st_in.bytes);
    fc_buffer *d_out = fc_buffer_alloc((size_t)C * sizeof(float));

    fc_ssm_conv1d_silu(d_qkv, d_w, d_state, d_out, C, kk);
    fc_sync();

    float *got_out = malloc((size_t)C * sizeof(float));
    float *got_state = malloc(st_in.bytes);
    fc_buffer_to_cpu(d_out, got_out);
    fc_buffer_to_cpu(d_state, got_state);   /* conv_state rolled in place */
    report_f32("ssm_conv_out", got_out, (const float *)out_gold.data,
               (size_t)C, PARITY_REL_TOL);
    report_f32("ssm_conv_state", got_state, (const float *)st_out_gold.data,
               st_out_gold.count, PARITY_REL_TOL);

    free(got_out);
    free(got_state);
    fc_buffer_free(d_qkv);
    fc_buffer_free(d_w);
    fc_buffer_free(d_state);
    fc_buffer_free(d_out);

cleanup:
    golden_free(&qkv);
    golden_free(&w);
    golden_free(&st_in);
    golden_free(&out_gold);
    golden_free(&st_out_gold);
}

/* Kernel 7: SSM gated RMSNorm. Uploads y, weight, z; runs fc_ssm_gated_norm
 * (writes gated); diffs gated against the golden. */
static void test_ssm_gated_norm(void)
{
    golden_vec y = {0}, w = {0}, z = {0}, out_gold = {0};
    if (golden_load("ssm_gated_y.bin", &y) != 0 ||
        golden_load("ssm_gated_weight.bin", &w) != 0 ||
        golden_load("ssm_gated_z.bin", &z) != 0 ||
        golden_load("ssm_gated_out.bin", &out_gold) != 0) {
        fprintf(stderr, "[parity] ssm_gated_norm: golden load failed\n");
        g_failures++;
        goto cleanup;
    }

    int H = PARITY_SSM_N_V_HEADS;
    int S = PARITY_SSM_STATE_SIZE;

    fc_buffer *d_y = fc_buffer_from_ptr(y.data, y.bytes);
    fc_buffer *d_w = fc_buffer_from_ptr(w.data, w.bytes);
    fc_buffer *d_z = fc_buffer_from_ptr(z.data, z.bytes);
    fc_buffer *d_gated = fc_buffer_alloc((size_t)H * S * sizeof(float));

    fc_ssm_gated_norm(d_y, d_w, d_z, d_gated, H, S, PARITY_RMS_NORM_EPS);
    fc_sync();

    float *got = malloc((size_t)H * S * sizeof(float));
    fc_buffer_to_cpu(d_gated, got);
    report_f32("ssm_gated_norm", got, (const float *)out_gold.data,
               (size_t)H * S, PARITY_REL_TOL);

    free(got);
    fc_buffer_free(d_y);
    fc_buffer_free(d_w);
    fc_buffer_free(d_z);
    fc_buffer_free(d_gated);

cleanup:
    golden_free(&y);
    golden_free(&w);
    golden_free(&z);
    golden_free(&out_gold);
}

int main(int argc, char **argv)
{
    /* Unbuffered so a hung kernel can't swallow the last line printed before
     * the wedge (a buffered run lost everything on the K80). */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== qwen35moe kernel parity harness ===\n");

    /* Args: [die_id] [stage]
     *   - a leading numeric arg = barswap die to acquire + bring nvidia up on
     *     (self-contained; no compute_persist needed to set up the die).
     *   - stage (optional) runs exactly one of: init q3k rope qknorm
     *     moe_router moe_gate, to pinpoint a hang without touching the rest. */
    int ai = 1;
    int die_id = -1;
    if (argc > ai && isdigit((unsigned char)argv[ai][0]))
        die_id = atoi(argv[ai++]);
    const char *only = (argc > ai) ? argv[ai] : NULL;

    if (die_id >= 0) {
        printf("[step] bs_acquire die %d ...\n", die_id);
        bs_context *bs = bs_init();
        if (!bs) { fprintf(stderr, "bs_init failed — is bar_swap.ko loaded?\n"); _Exit(1); }
        if (!bs_acquire(bs, die_id)) { fprintf(stderr, "bs_acquire(%d) failed\n", die_id); _Exit(1); }
        printf("[step] die %d acquired; loading nvidia ...\n", die_id);
        if (parity_load_nvidia() != 0) { fprintf(stderr, "nvidia load failed\n"); _Exit(1); }
        printf("[step] nvidia up\n");
    }

    printf("[step] fc_init ...\n");
    if (fc_init() != 0 || !fc_available()) {
        printf("[SKIP] no CUDA — host compile/link OK, GPU diffs deferred "
               "to a K80 host\n");
        return 0;
    }
    printf("[step] fc_init OK, CUDA available\n");
    /* NOTE: on the barswap K80 rig, CUDA's atexit context teardown hangs a
     * thread (zombies the process, half-unloads nvidia). So we DON'T call
     * fc_free()/return — we _Exit() to let the kernel reclaim the context on
     * process death, skipping the wedging userspace teardown. */
    if (only && strcmp(only, "init") == 0) { printf("[step] init-only done\n"); fflush(stdout); _Exit(0); }

    if (!only || !strcmp(only, "q3k"))        { printf("[step] q3k_matvec ...\n");    test_q3k_matvec(); }
    if (!only || !strcmp(only, "rope"))       { printf("[step] rope ...\n");          test_rope(); }
    if (!only || !strcmp(only, "qknorm"))     { printf("[step] qk_rms_norm ...\n");   test_qk_rms_norm(); }
    if (!only || !strcmp(only, "moe_router")) { printf("[step] moe_router ...\n");    test_moe_router(); }
    if (!only || !strcmp(only, "moe_gate"))   { printf("[step] moe_shared_gate ...\n"); test_moe_shared_gate(); }
    if (!only || !strcmp(only, "ssm"))        { printf("[step] ssm_recurrence ...\n"); test_ssm_recurrence(); }
    if (!only || !strcmp(only, "ssm_conv"))   { printf("[step] ssm_conv1d ...\n");     test_ssm_conv1d(); }
    if (!only || !strcmp(only, "ssm_gated"))  { printf("[step] ssm_gated_norm ...\n"); test_ssm_gated_norm(); }

    printf("=== %s (%d failure%s) ===\n",
           g_failures ? "FAIL" : "PASS", g_failures,
           g_failures == 1 ? "" : "s");
    /* _Exit (not fc_free/return): skip the CUDA atexit teardown that hangs a
     * thread on this barswap rig. Kernel reclaims the context on process death. */
    fflush(stdout);
    _Exit(g_failures ? 1 : 0);
}
