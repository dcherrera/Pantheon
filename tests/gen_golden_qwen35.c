/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gen_golden_qwen35.c
 * @brief CPU-reference "golden vector" generator for the qwen35moe GPU kernels.
 *        Produces deterministic input+output golden files under tests/golden/
 *        so the forthcoming K80 CUDA kernels can be diffed bit-for-bit (routing
 *        indices) / within tolerance (matvec, RoPE) against a trusted CPU
 *        reference. NO model GGUF is loaded — all inputs are synthetic,
 *        generated from a fixed-seed xorshift PRNG.
 *
 *        Kernels covered (priority order):
 *          1. Q3_K dequant + Q3_K matvec  (history of dequant bugs)
 *          2. Partial NEOX RoPE           (rope_dim 64 of head_dim 256)
 *          3. QK-RMSNorm                  (per-head RMSNorm)
 *          4. MoE routing                 (softmax + top-8/256 + norm_topk_prob
 *                                          + shared-expert sigmoid gate)
 *
 *        Where a PUBLIC Pantheon CPU function exists it is CALLED directly (the
 *        gold standard): pn_dequantize_block_q3_k / pn_qdot for Q3_K, and
 *        pn_vrms_norm_f32 for QK-RMSNorm. The partial-RoPE and MoE-routing math
 *        live in static helpers inside the model .c files; those are replicated
 *        here line-for-line with the source cited inline so the replica is
 *        auditable.
 *
 * @status Complete: generates all four golden sets, verifies finiteness, prints
 *         summary sums. CPU-only, no model, runs in well under a second.
 * @issues None
 * @todo None
 */

/* 1. (no own header — standalone tool) */

/* 2. Standard library */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 4. Project headers */
#include "../src/accel/simd.h"
#include "../src/core/qweight.h"
#include "../src/formats/gguf_dequant.h"

/* ============================================================
 * Config constants (verified Nex-N2-mini qwen35moe dimensions)
 * ============================================================ */

#define GOLDEN_SEED            0x9E3779B9u  /* fixed xorshift seed (golden ratio) */
#define GOLDEN_DIR             "tests/golden"

#define QWEN35_N_EMBD          2048
#define QWEN35_N_HEAD          16
#define QWEN35_N_HEAD_KV       2
#define QWEN35_HEAD_DIM        256
#define QWEN35_ROPE_DIM        64
#define QWEN35_N_EXPERT        256
#define QWEN35_EXPERT_USED     8
#define QWEN35_EXPERT_FF       512
#define QWEN35_RMS_NORM_EPS    1e-6f        /* attention.layer_norm_rms_epsilon */
#define QWEN35_ROPE_THETA      1e7f         /* rope.freq_base */
#define QWEN35_EXPERT_WT_SCALE 1.0f         /* expert_weights_scale (GGUF default) */

/* Q3_K matvec test shape: a synthetic weight matrix with cols a multiple of the
 * 256-element super-block and a handful of output rows. */
#define Q3K_ROWS               24
#define Q3K_COLS               1024         /* 4 Q3_K super-blocks per row */

/* Gated-Delta-Net SSM dimensions (verified Nex-N2-mini qwen35moe). */
#define SSM_N_V_HEADS          32
#define SSM_N_K_HEADS          16
#define SSM_STATE_SIZE         128          /* S == head_k_dim == head_v_dim */
#define SSM_CONV_DIM           8192
#define SSM_CONV_KERNEL        4

/* RoPE test positions. */
static const int ROPE_POSITIONS[] = {0, 1, 7, 100};
#define ROPE_N_POS (int)(sizeof(ROPE_POSITIONS) / sizeof(ROPE_POSITIONS[0]))

/* ============================================================
 * Deterministic PRNG — xorshift32 (seeded, never rand())
 * ============================================================ */

/* A tiny, fully deterministic xorshift32. Same seed -> same stream on every
 * platform, which is exactly what golden vectors need. */
typedef struct {
    uint32_t state;
} golden_rng;

static void golden_rng_init(golden_rng *rng, uint32_t seed)
{
    /* Avoid the zero fixed point; any nonzero seed produces a full-period stream. */
    rng->state = seed ? seed : 0xDEADBEEFu;
}

static uint32_t golden_rng_next_u32(golden_rng *rng)
{
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

/* Uniform byte. */
static uint8_t golden_rng_next_u8(golden_rng *rng)
{
    return (uint8_t)(golden_rng_next_u32(rng) & 0xFFu);
}

/* Uniform float in [-1, 1). */
static float golden_rng_next_signed_unit(golden_rng *rng)
{
    /* 24 mantissa bits -> [0,1), then map to [-1,1). */
    uint32_t bits = golden_rng_next_u32(rng) >> 8;
    float u01 = (float)bits / (float)(1u << 24);
    return u01 * 2.0f - 1.0f;
}

/* ============================================================
 * FP16 packing (host-side, mirrors IEEE-754 half; round-to-nearest-even)
 * ============================================================ */

/* Pack a finite, in-range float into an IEEE-754 half. Used only to fill the
 * synthetic Q3_K super-block scale field (d). Kept small and conservative — the
 * synthetic d values are O(0.01), comfortably inside the half range. The decode
 * side is the production pn_half_to_float, so any rounding here is captured in
 * the golden output (the real CPU kernel reads exactly these bytes). */
static uint16_t float_to_half(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0) {
        /* Subnormal/zero half — synthetic d is never this small; flush to 0. */
        return (uint16_t)sign;
    }
    if (exp >= 0x1F) {
        /* Overflow — clamp to max finite half. */
        return (uint16_t)(sign | 0x7BFFu);
    }
    /* Round mantissa to 10 bits, round-to-nearest-even. */
    uint32_t mant10 = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (mant10 & 1u)))
        mant10++;
    if (mant10 == 0x400u) {  /* mantissa overflow carries into exponent */
        mant10 = 0;
        exp++;
        if (exp >= 0x1F)
            return (uint16_t)(sign | 0x7BFFu);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant10);
}

/* ============================================================
 * Golden file writer — simple self-describing text-prefix format
 *
 * Format of every golden file:
 *   line 1: "PNGOLDEN1 <name> <dtype> <count> seed=<hex>"
 *   then <count> raw little-endian values of <dtype> (binary payload).
 * dtype is one of: f32, u8, i32. Shapes/semantics are in tests/golden/README.md.
 * ============================================================ */

static int write_golden(const char *filename, const char *name, const char *dtype,
                        const void *payload, size_t count, size_t elem_size)
{
    char path[512];
    int n = snprintf(path, sizeof(path), "%s/%s", GOLDEN_DIR, filename);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "write_golden: path overflow for '%s'\n", filename);
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "write_golden: cannot open '%s'\n", path);
        return -1;
    }

    int rc = -1;
    if (fprintf(fp, "PNGOLDEN1 %s %s %zu seed=0x%08X\n",
                name, dtype, count, (unsigned)GOLDEN_SEED) < 0)
        goto cleanup;
    if (count > 0 && fwrite(payload, elem_size, count, fp) != count)
        goto cleanup;
    rc = 0;

cleanup:
    if (rc != 0)
        fprintf(stderr, "write_golden: write failed for '%s'\n", path);
    fclose(fp);
    return rc;
}

/* Finiteness/sanity check + summary print for a float buffer. */
static int report_f32(const char *label, const float *data, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(data[i])) {
            fprintf(stderr, "  [FAIL] %s: non-finite value at index %zu\n", label, i);
            return -1;
        }
        sum += (double)data[i];
    }
    printf("  %-22s count=%-6zu sum=%+.6f  first=[%+.5f %+.5f %+.5f]\n",
           label, count, sum,
           count > 0 ? data[0] : 0.0f,
           count > 1 ? data[1] : 0.0f,
           count > 2 ? data[2] : 0.0f);
    return 0;
}

/* ============================================================
 * Kernel 1: Q3_K dequant + Q3_K matvec (REAL CPU functions)
 * ============================================================ */

/* Fill one Q3_K super-block with deterministic pseudo-random bytes that respect
 * the real block layout (pn_block_q3_k in gguf_dequant.h): hmask[32], qs[64],
 * scales[12], d (fp16). The decode is done by the production kernel, so every
 * bit pattern we write here is interpreted exactly as the GPU kernel must. */
static void fill_q3k_block(golden_rng *rng, pn_block_q3_k *block)
{
    for (int i = 0; i < 32; i++)
        block->hmask[i] = golden_rng_next_u8(rng);
    for (int i = 0; i < 64; i++)
        block->qs[i] = golden_rng_next_u8(rng);
    for (int i = 0; i < 12; i++)
        block->scales[i] = golden_rng_next_u8(rng);
    /* Small positive-ish super-block scale, typical of real Q3_K weights. */
    float d = 0.005f + 0.02f * (golden_rng_next_signed_unit(rng) * 0.5f + 0.5f);
    block->d = float_to_half(d);
}

/* Generate the Q3_K golden set: synthetic quantized weight matrix + F32 input
 * vector, then run the REAL pn_qmv to produce the golden output.
 *
 * Source of golden output: pn_qmv -> pn_qdot -> dot_q3_k ->
 * pn_dequantize_block_q3_k (src/core/qweight.c:46-57, 161-182;
 * src/formats/gguf_dequant.c:118-162). All public, all called directly. */
static int gen_q3k_golden(golden_rng *rng)
{
    const int rows = Q3K_ROWS;
    const int cols = Q3K_COLS;
    const int blocks_per_row = cols / 256;
    const size_t total_blocks = (size_t)rows * (size_t)blocks_per_row;

    pn_block_q3_k *weight = malloc(total_blocks * sizeof(*weight));
    float *input = malloc((size_t)cols * sizeof(*input));
    float *output = malloc((size_t)rows * sizeof(*output));
    int rc = -1;
    if (!weight || !input || !output) {
        fprintf(stderr, "gen_q3k_golden: allocation failed\n");
        goto cleanup;
    }

    for (size_t b = 0; b < total_blocks; b++)
        fill_q3k_block(rng, &weight[b]);
    for (int i = 0; i < cols; i++)
        input[i] = golden_rng_next_signed_unit(rng);

    /* REAL CPU matvec — the gold standard. row_stride = bytes per quantized row. */
    pn_qweight qw = {
        .data = weight,
        .type = GGUF_QUANT_Q3_K,
        .rows = rows,
        .cols = cols,
        .row_stride = (size_t)blocks_per_row * sizeof(pn_block_q3_k),
        .scales = NULL,
        .group_size = 0,
    };
    pn_qmv(&qw, input, output);

    if (write_golden("q3k_matvec_weight.bin", "q3k_matvec_weight", "u8",
                     weight, total_blocks * sizeof(pn_block_q3_k), 1) < 0)
        goto cleanup;
    if (write_golden("q3k_matvec_input.bin", "q3k_matvec_input", "f32",
                     input, (size_t)cols, sizeof(float)) < 0)
        goto cleanup;
    if (write_golden("q3k_matvec_output.bin", "q3k_matvec_output", "f32",
                     output, (size_t)rows, sizeof(float)) < 0)
        goto cleanup;

    /* Also emit a single dequantized super-block (block 0) as a standalone
     * dequant golden — isolates the dequant from the dot-product. */
    float dequant0[256];
    pn_dequantize_block_q3_k(&weight[0], dequant0);
    if (write_golden("q3k_dequant_block0.bin", "q3k_dequant_block0", "f32",
                     dequant0, 256, sizeof(float)) < 0)
        goto cleanup;

    printf("[1] Q3_K dequant + matvec (REAL pn_qmv / pn_dequantize_block_q3_k)\n");
    if (report_f32("q3k_dequant_block0", dequant0, 256) < 0) goto cleanup;
    if (report_f32("q3k_matvec_output", output, (size_t)rows) < 0) goto cleanup;
    rc = 0;

cleanup:
    free(weight);
    free(input);
    free(output);
    return rc;
}

/* ============================================================
 * Kernel 2: Partial NEOX RoPE (cross-checked replica)
 *
 * Replicates qwen35moe_rope_partial, src/models/qwen35moe_attn.c:57-76.
 * Only the first rope_dim dims of each head are rotated (NEOX pairing j vs
 * j+half, half = rope_dim/2); dims [rope_dim, head_dim) pass through. The math
 * below is identical to the source, term for term.
 * ============================================================ */

static void rope_partial_replica(float *data, int n_heads, int head_dim,
                                 int rope_dim, int pos, float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + (size_t)h * head_dim;
        for (int j = 0; j < half; j++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[j];
            float x1 = head[j + half];
            head[j]        = x0 * cos_a - x1 * sin_a;
            head[j + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* Generate RoPE golden: one synthetic Q tensor [n_head*head_dim] and one
 * synthetic K tensor [n_head_kv*head_dim], rotated at each test position. The
 * pre-rotation tensors are shared across positions (written once); each position
 * gets its own rotated Q and K golden. */
static int gen_rope_golden(golden_rng *rng)
{
    const int q_elems = QWEN35_N_HEAD * QWEN35_HEAD_DIM;
    const int k_elems = QWEN35_N_HEAD_KV * QWEN35_HEAD_DIM;

    float *q_in = malloc((size_t)q_elems * sizeof(float));
    float *k_in = malloc((size_t)k_elems * sizeof(float));
    float *q_work = malloc((size_t)q_elems * sizeof(float));
    float *k_work = malloc((size_t)k_elems * sizeof(float));
    int rc = -1;
    if (!q_in || !k_in || !q_work || !k_work) {
        fprintf(stderr, "gen_rope_golden: allocation failed\n");
        goto cleanup;
    }

    for (int i = 0; i < q_elems; i++)
        q_in[i] = golden_rng_next_signed_unit(rng);
    for (int i = 0; i < k_elems; i++)
        k_in[i] = golden_rng_next_signed_unit(rng);

    if (write_golden("rope_q_input.bin", "rope_q_input", "f32",
                     q_in, (size_t)q_elems, sizeof(float)) < 0) goto cleanup;
    if (write_golden("rope_k_input.bin", "rope_k_input", "f32",
                     k_in, (size_t)k_elems, sizeof(float)) < 0) goto cleanup;

    printf("[2] Partial NEOX RoPE (replica of qwen35moe_attn.c:57-76)\n");
    for (int p = 0; p < ROPE_N_POS; p++) {
        int pos = ROPE_POSITIONS[p];
        memcpy(q_work, q_in, (size_t)q_elems * sizeof(float));
        memcpy(k_work, k_in, (size_t)k_elems * sizeof(float));
        rope_partial_replica(q_work, QWEN35_N_HEAD, QWEN35_HEAD_DIM,
                             QWEN35_ROPE_DIM, pos, QWEN35_ROPE_THETA);
        rope_partial_replica(k_work, QWEN35_N_HEAD_KV, QWEN35_HEAD_DIM,
                             QWEN35_ROPE_DIM, pos, QWEN35_ROPE_THETA);

        char qname[64], kname[64], qfile[64], kfile[64];
        snprintf(qname, sizeof(qname), "rope_q_pos%d", pos);
        snprintf(kname, sizeof(kname), "rope_k_pos%d", pos);
        snprintf(qfile, sizeof(qfile), "rope_q_pos%d.bin", pos);
        snprintf(kfile, sizeof(kfile), "rope_k_pos%d.bin", pos);
        if (write_golden(qfile, qname, "f32", q_work, (size_t)q_elems,
                         sizeof(float)) < 0) goto cleanup;
        if (write_golden(kfile, kname, "f32", k_work, (size_t)k_elems,
                         sizeof(float)) < 0) goto cleanup;

        char qlabel[64];
        snprintf(qlabel, sizeof(qlabel), "rope_q_pos%d", pos);
        if (report_f32(qlabel, q_work, (size_t)q_elems) < 0) goto cleanup;
    }
    rc = 0;

cleanup:
    free(q_in);
    free(k_in);
    free(q_work);
    free(k_work);
    return rc;
}

/* ============================================================
 * Kernel 3: QK-RMSNorm (REAL CPU function pn_vrms_norm_f32)
 *
 * Mirrors qwen35moe_qk_norm_heads, src/models/qwen35moe_attn.c:39-46, which
 * applies the public pn_vrms_norm_f32 (src/accel/simd.c:198) once per head with
 * a shared norm weight. We CALL the real pn_vrms_norm_f32 — gold standard.
 * ============================================================ */

static int gen_qknorm_golden(golden_rng *rng)
{
    const int n_heads = QWEN35_N_HEAD;
    const int head_dim = QWEN35_HEAD_DIM;
    const int total = n_heads * head_dim;

    float *input = malloc((size_t)total * sizeof(float));
    float *weight = malloc((size_t)head_dim * sizeof(float));
    float *output = malloc((size_t)total * sizeof(float));
    int rc = -1;
    if (!input || !weight || !output) {
        fprintf(stderr, "gen_qknorm_golden: allocation failed\n");
        goto cleanup;
    }

    for (int i = 0; i < total; i++)
        input[i] = golden_rng_next_signed_unit(rng);
    /* Norm weight is shared across heads; GGUF bakes it as ~1.0 + small offset. */
    for (int i = 0; i < head_dim; i++)
        weight[i] = 1.0f + 0.1f * golden_rng_next_signed_unit(rng);

    /* REAL per-head RMSNorm — exactly what qwen35moe_qk_norm_heads does. */
    for (int h = 0; h < n_heads; h++) {
        const float *head_in = input + (size_t)h * head_dim;
        float *head_out = output + (size_t)h * head_dim;
        pn_vrms_norm_f32(head_in, weight, head_out, head_dim, QWEN35_RMS_NORM_EPS);
    }

    if (write_golden("qknorm_input.bin", "qknorm_input", "f32",
                     input, (size_t)total, sizeof(float)) < 0) goto cleanup;
    if (write_golden("qknorm_weight.bin", "qknorm_weight", "f32",
                     weight, (size_t)head_dim, sizeof(float)) < 0) goto cleanup;
    if (write_golden("qknorm_output.bin", "qknorm_output", "f32",
                     output, (size_t)total, sizeof(float)) < 0) goto cleanup;

    printf("[3] QK-RMSNorm (REAL pn_vrms_norm_f32, per head)\n");
    if (report_f32("qknorm_output", output, (size_t)total) < 0) goto cleanup;
    rc = 0;

cleanup:
    free(input);
    free(weight);
    free(output);
    return rc;
}

/* ============================================================
 * Kernel 4: MoE routing (cross-checked replica)
 *
 * Replicates the routed-gating path of qwen35moe_moe.c:
 *   - qwen35moe_router_softmax     (lines 47-61)  softmax over n_expert logits
 *   - qwen35moe_select_topk        (lines 65-89)  top-k by probability
 *   - norm_topk_prob + scale       (lines 133-136) renormalize selected weights
 *   - shared-expert sigmoid gate   (lines 179-183) sigmoid(gate_w . x) scalar
 * Output: top-k expert indices (i32), normalized+scaled weights (f32), and the
 * shared-expert gate scalar (f32). Routing indices must match EXACTLY on GPU.
 * ============================================================ */

/* sigmoid(x) — mirrors qwen35moe_sigmoid, qwen35moe_moe.c:37-40. */
static float moe_sigmoid_replica(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* In-place softmax — mirrors qwen35moe_router_softmax, qwen35moe_moe.c:47-61. */
static void moe_router_softmax_replica(float *logits, int n_expert)
{
    float max_gl = logits[0];
    for (int e = 1; e < n_expert; e++)
        if (logits[e] > max_gl)
            max_gl = logits[e];
    float sum_exp = 0.0f;
    for (int e = 0; e < n_expert; e++) {
        logits[e] = expf(logits[e] - max_gl);
        sum_exp += logits[e];
    }
    float inv = 1.0f / sum_exp;
    for (int e = 0; e < n_expert; e++)
        logits[e] *= inv;
}

/* Top-k selection — mirrors qwen35moe_select_topk, qwen35moe_moe.c:65-89. The
 * selection-sort + first-seen tie-break MUST match the GPU bit-for-bit. */
static float moe_select_topk_replica(const float *probs, int n_expert, int k,
                                    int *top_idx, float *top_wt)
{
    float wt_sum = 0.0f;
    for (int s = 0; s < k; s++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_expert; e++) {
            int skip = 0;
            for (int j = 0; j < s; j++)
                if (top_idx[j] == e) {
                    skip = 1;
                    break;
                }
            if (!skip && probs[e] > best_val) {
                best = e;
                best_val = probs[e];
            }
        }
        top_idx[s] = best;
        top_wt[s] = best_val;
        wt_sum += best_val;
    }
    return wt_sum;
}

static int gen_moe_routing_golden(golden_rng *rng)
{
    const int n_expert = QWEN35_N_EXPERT;
    const int k = QWEN35_EXPERT_USED;
    const int n_embd = QWEN35_N_EMBD;

    float *logits = malloc((size_t)n_expert * sizeof(float));
    float *probs = malloc((size_t)n_expert * sizeof(float));
    float *shared_gate_w = malloc((size_t)n_embd * sizeof(float));
    float *hidden_x = malloc((size_t)n_embd * sizeof(float));
    int rc = -1;
    if (!logits || !probs || !shared_gate_w || !hidden_x) {
        fprintf(stderr, "gen_moe_routing_golden: allocation failed\n");
        goto cleanup;
    }

    for (int e = 0; e < n_expert; e++)
        logits[e] = golden_rng_next_signed_unit(rng) * 3.0f;  /* router logits */
    for (int j = 0; j < n_embd; j++)
        shared_gate_w[j] = golden_rng_next_signed_unit(rng);
    for (int j = 0; j < n_embd; j++)
        hidden_x[j] = golden_rng_next_signed_unit(rng);

    /* Save raw router logits as the input golden (GPU recomputes from these). */
    if (write_golden("moe_router_logits.bin", "moe_router_logits", "f32",
                     logits, (size_t)n_expert, sizeof(float)) < 0) goto cleanup;
    if (write_golden("moe_shared_gate_weight.bin", "moe_shared_gate_weight", "f32",
                     shared_gate_w, (size_t)n_embd, sizeof(float)) < 0) goto cleanup;
    if (write_golden("moe_shared_gate_input.bin", "moe_shared_gate_input", "f32",
                     hidden_x, (size_t)n_embd, sizeof(float)) < 0) goto cleanup;

    /* Routed gating: softmax -> top-k -> norm_topk_prob -> expert_weights_scale. */
    memcpy(probs, logits, (size_t)n_expert * sizeof(float));
    moe_router_softmax_replica(probs, n_expert);

    int top_idx[QWEN35_EXPERT_USED];
    float top_wt[QWEN35_EXPERT_USED];
    float wt_sum = moe_select_topk_replica(probs, n_expert, k, top_idx, top_wt);
    float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
    float scale = QWEN35_EXPERT_WT_SCALE * inv_wt;
    for (int s = 0; s < k; s++)
        top_wt[s] *= scale;

    /* Shared-expert scalar sigmoid gate: sigmoid(gate_w . x). */
    float gate_logit = 0.0f;
    for (int j = 0; j < n_embd; j++)
        gate_logit += shared_gate_w[j] * hidden_x[j];
    float shared_gate = moe_sigmoid_replica(gate_logit);

    if (write_golden("moe_topk_indices.bin", "moe_topk_indices", "i32",
                     top_idx, (size_t)k, sizeof(int32_t)) < 0) goto cleanup;
    if (write_golden("moe_topk_weights.bin", "moe_topk_weights", "f32",
                     top_wt, (size_t)k, sizeof(float)) < 0) goto cleanup;
    if (write_golden("moe_shared_gate_scalar.bin", "moe_shared_gate_scalar", "f32",
                     &shared_gate, 1, sizeof(float)) < 0) goto cleanup;

    printf("[4] MoE routing (replica of qwen35moe_moe.c softmax/top-k/norm/gate)\n");
    printf("  top-%d indices:", k);
    for (int s = 0; s < k; s++)
        printf(" %d", top_idx[s]);
    printf("\n");
    if (report_f32("moe_topk_weights", top_wt, (size_t)k) < 0) goto cleanup;
    printf("  shared_gate_scalar = %+.6f (logit %+.6f)\n", shared_gate, gate_logit);
    {
        double wsum = 0.0;
        for (int s = 0; s < k; s++) wsum += top_wt[s];
        printf("  weights sum (should be ~%.4f) = %.6f\n",
               (double)QWEN35_EXPERT_WT_SCALE, wsum);
    }
    rc = 0;

cleanup:
    free(logits);
    free(probs);
    free(shared_gate_w);
    free(hidden_x);
    return rc;
}

/* ============================================================
 * Kernel 5: SSM per-token delta-rule recurrence (cross-checked replica)
 *
 * Replicates qwen35moe_compute_recur_heads, src/models/qwen35moe_ssm.c:163-196
 * (the static core recurrence). Per V-head h, per row j of the TRANSPOSED state
 * s[h*S*S + j*S + i] == S_h[i][j]:
 *   (a) row[i] *= exp(g[h])                       (ssm.c:180-182)
 *   (b) pred = sum_i row[i]*k[h][i];
 *       delta = (v[h][j] - pred) * beta[h]        (ssm.c:184-186)
 *   (c) row[i] += k[h][i]*delta                   (ssm.c:188-190)
 *   (d) y[h][j] = scale * sum_i row[i]*q[h][i]    (ssm.c:192-193, read-after-write)
 * q,k are taken AS-IF already L2-normed + GQA-repeated to n_v_heads (this is the
 * input contract the GPU kernel receives); g is PRE-exp; beta is already sigmoid'd.
 * Uses the public pn_dot_f32 for the two reductions, exactly as the source does.
 * ============================================================ */

static void ssm_recur_replica(float *s, const float *q, const float *k,
                              const float *v, const float *g, const float *beta,
                              float *y, int n_v_heads, int S, float scale)
{
    for (int h = 0; h < n_v_heads; h++) {
        float *sh = s + (size_t)h * S * S;
        const float *kh = k + (size_t)h * S;
        const float *qh = q + (size_t)h * S;
        const float *vh = v + (size_t)h * S;
        const float decay = expf(g[h]);
        const float beta_h = beta[h];
        float *yh = y + (size_t)h * S;

        for (int j = 0; j < S; j++) {
            float *row = sh + (size_t)j * S;     /* row j: S[i][j] over i */
            for (int i = 0; i < S; i++)          /* (a) scalar decay */
                row[i] *= decay;
            float pred = pn_dot_f32(row, kh, S); /* (b) pred + delta */
            float delta = (vh[j] - pred) * beta_h;
            for (int i = 0; i < S; i++)          /* (c) rank-1 write */
                row[i] += kh[i] * delta;
            yh[j] = scale * pn_dot_f32(row, qh, S);  /* (d) read-after-write */
        }
    }
}

static int gen_ssm_recur_golden(golden_rng *rng)
{
    const int H = SSM_N_V_HEADS;
    const int S = SSM_STATE_SIZE;
    const int hs = H * S;                 /* q/k/v/y stream length */
    const size_t state_n = (size_t)H * S * S;
    const float scale = 1.0f / sqrtf((float)S);

    float *q = malloc((size_t)hs * sizeof(float));
    float *k = malloc((size_t)hs * sizeof(float));
    float *v = malloc((size_t)hs * sizeof(float));
    float *g = malloc((size_t)H * sizeof(float));
    float *beta = malloc((size_t)H * sizeof(float));
    float *s_in = malloc(state_n * sizeof(float));
    float *s_out = malloc(state_n * sizeof(float));
    float *y = malloc((size_t)hs * sizeof(float));
    int rc = -1;
    if (!q || !k || !v || !g || !beta || !s_in || !s_out || !y) {
        fprintf(stderr, "gen_ssm_recur_golden: allocation failed\n");
        goto cleanup;
    }

    /* q/k arrive L2-normed per head; emulate that so magnitudes are realistic. */
    for (int h = 0; h < H; h++) {
        for (int d = 0; d < S; d++) {
            q[h * S + d] = golden_rng_next_signed_unit(rng);
            k[h * S + d] = golden_rng_next_signed_unit(rng);
            v[h * S + d] = golden_rng_next_signed_unit(rng);
        }
        /* L2-normalize q,k per head (mirror ssm.c:70-76 divisor form). */
        float qsq = 0.0f, ksq = 0.0f;
        for (int d = 0; d < S; d++) { qsq += q[h*S+d]*q[h*S+d]; ksq += k[h*S+d]*k[h*S+d]; }
        float qinv = 1.0f / fmaxf(sqrtf(qsq), QWEN35_RMS_NORM_EPS);
        float kinv = 1.0f / fmaxf(sqrtf(ksq), QWEN35_RMS_NORM_EPS);
        for (int d = 0; d < S; d++) { q[h*S+d] *= qinv; k[h*S+d] *= kinv; }
        /* g is PRE-exp: small negative so exp(g) in (0,1), as the decay gate yields. */
        g[h] = -0.05f - 0.1f * (golden_rng_next_signed_unit(rng) * 0.5f + 0.5f);
        beta[h] = moe_sigmoid_replica(golden_rng_next_signed_unit(rng));
    }
    for (size_t i = 0; i < state_n; i++)
        s_in[i] = 0.1f * golden_rng_next_signed_unit(rng);

    /* Run the replica on a copy of the input state (state is updated in place). */
    memcpy(s_out, s_in, state_n * sizeof(float));
    ssm_recur_replica(s_out, q, k, v, g, beta, y, H, S, scale);

    if (write_golden("ssm_recur_state_in.bin", "ssm_recur_state_in", "f32",
                     s_in, state_n, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_q.bin", "ssm_recur_q", "f32", q, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_k.bin", "ssm_recur_k", "f32", k, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_v.bin", "ssm_recur_v", "f32", v, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_g.bin", "ssm_recur_g", "f32", g, (size_t)H, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_beta.bin", "ssm_recur_beta", "f32", beta, (size_t)H, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_y.bin", "ssm_recur_y", "f32", y, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_recur_state_out.bin", "ssm_recur_state_out", "f32",
                     s_out, state_n, sizeof(float)) < 0) goto cleanup;

    printf("[5] SSM recurrence (replica of qwen35moe_ssm.c:163-196)\n");
    if (report_f32("ssm_recur_y", y, (size_t)hs) < 0) goto cleanup;
    if (report_f32("ssm_recur_state_out", s_out, state_n) < 0) goto cleanup;
    rc = 0;

cleanup:
    free(q); free(k); free(v); free(g); free(beta);
    free(s_in); free(s_out); free(y);
    return rc;
}

/* ============================================================
 * Kernel 6: SSM causal conv1d (k=4) + SiLU (cross-checked replica)
 *
 * Replicates qwen35moe_conv1d_silu, src/models/qwen35moe_ssm.c:110-135. Per
 * channel c over conv_dim: out[c] = SiLU(history . w[0..k-2] + qkv[c]*w[k-1]),
 * then roll the per-channel conv_state (drop oldest, append raw qkv[c]).
 * conv_state layout [conv_dim][k-1], weight row-major [conv_dim, k].
 * ============================================================ */

static float ssm_silu_replica(float x)  /* mirror qwen35moe_silu (ssm.c:60-63) */
{
    return x / (1.0f + expf(-x));
}

static void ssm_conv1d_replica(const float *qkv, const float *weight,
                               float *conv_state, float *out,
                               int conv_dim, int k)
{
    const int km1 = k - 1;
    for (int c = 0; c < conv_dim; c++) {
        const float *cw = weight + (size_t)c * k;
        float *cs = conv_state + (size_t)c * km1;
        float acc = qkv[c] * cw[km1];           /* ssm.c:125-127 */
        for (int t = 0; t < km1; t++)
            acc += cs[t] * cw[t];
        out[c] = ssm_silu_replica(acc);          /* ssm.c:128 */
        for (int t = 0; t < km1 - 1; t++)        /* ssm.c:131-133 roll */
            cs[t] = cs[t + 1];
        cs[km1 - 1] = qkv[c];
    }
}

static int gen_ssm_conv_golden(golden_rng *rng)
{
    const int C = SSM_CONV_DIM;
    const int k = SSM_CONV_KERNEL;
    const int km1 = k - 1;
    const size_t st_n = (size_t)C * km1;
    const size_t w_n = (size_t)C * k;

    float *qkv = malloc((size_t)C * sizeof(float));
    float *weight = malloc(w_n * sizeof(float));
    float *st_in = malloc(st_n * sizeof(float));
    float *st_out = malloc(st_n * sizeof(float));
    float *out = malloc((size_t)C * sizeof(float));
    int rc = -1;
    if (!qkv || !weight || !st_in || !st_out || !out) {
        fprintf(stderr, "gen_ssm_conv_golden: allocation failed\n");
        goto cleanup;
    }

    for (int i = 0; i < C; i++)
        qkv[i] = golden_rng_next_signed_unit(rng);
    for (size_t i = 0; i < w_n; i++)
        weight[i] = 0.5f * golden_rng_next_signed_unit(rng);
    for (size_t i = 0; i < st_n; i++)
        st_in[i] = golden_rng_next_signed_unit(rng);

    memcpy(st_out, st_in, st_n * sizeof(float));   /* state rolls in place */
    ssm_conv1d_replica(qkv, weight, st_out, out, C, k);

    if (write_golden("ssm_conv_qkv.bin", "ssm_conv_qkv", "f32", qkv, (size_t)C, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_conv_weight.bin", "ssm_conv_weight", "f32", weight, w_n, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_conv_state_in.bin", "ssm_conv_state_in", "f32", st_in, st_n, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_conv_out.bin", "ssm_conv_out", "f32", out, (size_t)C, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_conv_state_out.bin", "ssm_conv_state_out", "f32", st_out, st_n, sizeof(float)) < 0) goto cleanup;

    printf("[6] SSM conv1d+SiLU (replica of qwen35moe_ssm.c:110-135)\n");
    if (report_f32("ssm_conv_out", out, (size_t)C) < 0) goto cleanup;
    if (report_f32("ssm_conv_state_out", st_out, st_n) < 0) goto cleanup;
    rc = 0;

cleanup:
    free(qkv); free(weight); free(st_in); free(st_out); free(out);
    return rc;
}

/* ============================================================
 * Kernel 7: SSM gated RMSNorm (REAL pn_vrms_norm_f32 + silu replica)
 *
 * Mirrors qwen35moe_gated_norm, src/models/qwen35moe_ssm.c:206-223:
 *   pn_vrms_norm_f32(y_h, w, gated_h, S, eps)   then  gated_h[d] *= silu(z_h[d]).
 * Calls the REAL pn_vrms_norm_f32 per head (gold standard); silu mirrors source.
 * ============================================================ */

static int gen_ssm_gated_golden(golden_rng *rng)
{
    const int H = SSM_N_V_HEADS;
    const int S = SSM_STATE_SIZE;
    const int hs = H * S;

    float *y = malloc((size_t)hs * sizeof(float));
    float *weight = malloc((size_t)S * sizeof(float));
    float *z = malloc((size_t)hs * sizeof(float));
    float *gated = malloc((size_t)hs * sizeof(float));
    int rc = -1;
    if (!y || !weight || !z || !gated) {
        fprintf(stderr, "gen_ssm_gated_golden: allocation failed\n");
        goto cleanup;
    }

    for (int i = 0; i < hs; i++) {
        y[i] = golden_rng_next_signed_unit(rng);
        z[i] = golden_rng_next_signed_unit(rng);
    }
    for (int d = 0; d < S; d++)             /* shared norm weight, ~1.0 + offset */
        weight[d] = 1.0f + 0.1f * golden_rng_next_signed_unit(rng);

    for (int h = 0; h < H; h++) {            /* mirror ssm.c:214-222 */
        const float *yh = y + (size_t)h * S;
        const float *zh = z + (size_t)h * S;
        float *gh = gated + (size_t)h * S;
        pn_vrms_norm_f32(yh, weight, gh, S, QWEN35_RMS_NORM_EPS);
        for (int d = 0; d < S; d++)
            gh[d] *= ssm_silu_replica(zh[d]);
    }

    if (write_golden("ssm_gated_y.bin", "ssm_gated_y", "f32", y, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_gated_weight.bin", "ssm_gated_weight", "f32", weight, (size_t)S, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_gated_z.bin", "ssm_gated_z", "f32", z, (size_t)hs, sizeof(float)) < 0) goto cleanup;
    if (write_golden("ssm_gated_out.bin", "ssm_gated_out", "f32", gated, (size_t)hs, sizeof(float)) < 0) goto cleanup;

    printf("[7] SSM gated RMSNorm (REAL pn_vrms_norm_f32 * silu(z), per head)\n");
    if (report_f32("ssm_gated_out", gated, (size_t)hs) < 0) goto cleanup;
    rc = 0;

cleanup:
    free(y); free(weight); free(z); free(gated);
    return rc;
}

/* ============================================================
 * Entry point
 * ============================================================ */

int main(void)
{
    printf("qwen35moe golden-vector generator (seed=0x%08X)\n", (unsigned)GOLDEN_SEED);
    printf("output dir: %s/\n\n", GOLDEN_DIR);

    /* One RNG threaded through all kernels: the full output stream is determined
     * solely by the seed and the (fixed) order of generation calls below. */
    golden_rng rng;
    golden_rng_init(&rng, GOLDEN_SEED);

    int rc = 0;
    rc |= gen_q3k_golden(&rng);
    printf("\n");
    rc |= gen_rope_golden(&rng);
    printf("\n");
    rc |= gen_qknorm_golden(&rng);
    printf("\n");
    rc |= gen_moe_routing_golden(&rng);
    printf("\n");
    rc |= gen_ssm_recur_golden(&rng);
    printf("\n");
    rc |= gen_ssm_conv_golden(&rng);
    printf("\n");
    rc |= gen_ssm_gated_golden(&rng);

    printf("\n%s\n", rc == 0 ? "ALL GOLDEN VECTORS GENERATED OK"
                             : "GENERATION FAILED — see errors above");
    return rc == 0 ? 0 : 1;
}
