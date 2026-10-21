/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rwkvx.c — RWKV-X inference engine for Pantheon
 *
 * Copied from foundry/rwkvx/rwkvx.c with Metal, CUDA, HybridMem, Training,
 * Batch, Instrumentation, and OpenMP support stripped.
 * Wrapped with Pantheon's pn_model/pn_state adapter interface.
 */

#include "rwkvx.h"
#include "tensor/tensor.h"
#include "safetensors/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __ARM_NEON
  #include <arm_neon.h>
#endif

#ifdef USE_THREADS
  #include "threadpool/threadpool.h"
#endif

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_CUDA
  #include "cuda/fc_cuda.h"
#endif

#ifdef USE_THREADS
/* File-scope pool pointer — set during forward, used by internal
 * parallel functions. Avoids changing all static function signatures. */
static _Thread_local fw_pool *__rx_pool = NULL;
#endif

/* Aligned malloc helper — 64-byte alignment for NEON/AVX cache line ops */
static void* aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}

/* ================================================================
 * Internal: INT8 quantized weight matrix (per-row symmetric)
 * ================================================================ */

typedef struct {
    int8_t *data;       /* [rows * cols] quantized data */
    float *scales;      /* [rows] per-row scale: float = int8 * scale */
    int rows, cols;
#ifdef USE_CUDA
    fc_buffer *cuda_data;   /* Persistent GPU copy of int8 weights */
    fc_buffer *cuda_scales; /* Persistent GPU copy of per-row scales */
#endif
} rx_q8_weight;

static rx_q8_weight* rx_q8_from_f32(const float *src, int rows, int cols) {
    rx_q8_weight *q = (rx_q8_weight*)calloc(1, sizeof(rx_q8_weight));
    q->rows = rows;
    q->cols = cols;
    q->data = (int8_t*)aligned_malloc(rows * cols * sizeof(int8_t));
    q->scales = (float*)aligned_malloc(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        const float *row = &src[i * cols];
        float amax = 0.0f;
        for (int j = 0; j < cols; j++) {
            float a = row[j] < 0 ? -row[j] : row[j];
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        q->scales[i] = scale;
        float inv_scale = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        int8_t *dst = &q->data[i * cols];
        for (int j = 0; j < cols; j++) {
            float v = row[j] * inv_scale;
            int vi = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (vi > 127) vi = 127;
            if (vi < -127) vi = -127;
            dst[j] = (int8_t)vi;
        }
    }
#ifdef USE_CUDA
    if (fc_available()) {
        q->cuda_data = fc_buffer_from_ptr(q->data, rows * cols * sizeof(int8_t));
        q->cuda_scales = fc_buffer_from_ptr(q->scales, rows * sizeof(float));
    } else {
        q->cuda_data = NULL;
        q->cuda_scales = NULL;
    }
#endif
    return q;
}

static void rx_q8_free(rx_q8_weight *q) {
    if (!q) return;
#ifdef USE_CUDA
    if (q->cuda_data) fc_buffer_free(q->cuda_data);
    if (q->cuda_scales) fc_buffer_free(q->cuda_scales);
#endif
    free(q->data);
    free(q->scales);
    free(q);
}

/* Quantize a float vector to int8 in-place into a pre-allocated buffer.
 * Returns the scale factor (amax / 127). */
static float fast_quantize_input(const float *x, int8_t *x_q, int n) {
    float x_amax = 0.0f;
    for (int j = 0; j < n; j++) {
        float a = x[j] < 0 ? -x[j] : x[j];
        if (a > x_amax) x_amax = a;
    }
    float x_scale = x_amax / 127.0f;
    float x_inv_scale = (x_amax > 0.0f) ? 127.0f / x_amax : 0.0f;
    for (int j = 0; j < n; j++) {
        float v = x[j] * x_inv_scale;
        int vi = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (vi > 127) vi = 127;
        if (vi < -127) vi = -127;
        x_q[j] = (int8_t)vi;
    }
    return x_scale;
}

#ifdef USE_THREADS
typedef struct {
    const int8_t *x_q;
    float x_scale;
    const rx_q8_weight *q;
    float *out;
    int cols;
} q8_row_ctx;

static void q8_row_worker(int start, int end, void *arg);
#endif

/* Quantized matvec with pre-quantized input. */
static void fast_matvec_q8_preq(const int8_t *x_q, float x_scale,
                                 const rx_q8_weight *q, float *out,
                                 int rows, int cols) {
#ifdef USE_CUDA
    if (fc_available() && q->cuda_data && (int64_t)rows * cols >= FC_Q8_THRESHOLD) {
        fc_buffer *bxq = fc_buffer_from_ptr(x_q, cols * sizeof(int8_t));
        fc_buffer *bout = fc_buffer_alloc(rows * sizeof(float));
        fc_matvec_q8(bxq, q->cuda_data, q->cuda_scales, bout,
                     x_scale, rows, cols);
        fc_sync();
        fc_buffer_to_cpu(bout, out);
        fc_buffer_free(bxq);
        fc_buffer_free(bout);
        return;
    }
#endif

#ifdef __ARM_NEON
    int cols64 = cols & ~63;
    int cols16 = cols & ~15;
#endif

#ifdef USE_THREADS
    if (__rx_pool && rows >= 512) {
        q8_row_ctx ctx = { x_q, x_scale, q, out, cols };
        fw_for(__rx_pool, rows, q8_row_worker, &ctx);
        return;
    }
#endif
    for (int i = 0; i < rows; i++) {
        const int8_t *row = &q->data[i * cols];
        float w_scale = q->scales[i];

        if (i + 1 < rows)
            __builtin_prefetch(&q->data[(i + 1) * cols], 0, 1);

#ifdef __ARM_NEON
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (int j = 0; j < cols64; j += 64) {
            acc0 = vdotq_s32(acc0, vld1q_s8(&x_q[j]),      vld1q_s8(&row[j]));
            acc1 = vdotq_s32(acc1, vld1q_s8(&x_q[j + 16]), vld1q_s8(&row[j + 16]));
            acc2 = vdotq_s32(acc2, vld1q_s8(&x_q[j + 32]), vld1q_s8(&row[j + 32]));
            acc3 = vdotq_s32(acc3, vld1q_s8(&x_q[j + 48]), vld1q_s8(&row[j + 48]));
        }
        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        for (int j = cols64; j < cols16; j += 16)
            acc = vdotq_s32(acc, vld1q_s8(&x_q[j]), vld1q_s8(&row[j]));
        int32_t dot = vaddvq_s32(acc);
        for (int j = cols16; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#else
        int32_t dot = 0;
        for (int j = 0; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#endif
        out[i] = (float)dot * w_scale * x_scale;
    }
}

#ifdef USE_THREADS
static void q8_row_worker(int start, int end, void *arg) {
    q8_row_ctx *ctx = (q8_row_ctx *)arg;
    const int8_t *x_q = ctx->x_q;
    float x_scale = ctx->x_scale;
    const rx_q8_weight *q = ctx->q;
    float *out = ctx->out;
    int cols = ctx->cols;
    int rows = q->rows;
#ifdef __ARM_NEON
    int cols64 = cols & ~63;
    int cols16 = cols & ~15;
#endif
    for (int i = start; i < end; i++) {
        const int8_t *row = &q->data[i * cols];
        float w_scale = q->scales[i];
        if (i + 1 < rows)
            __builtin_prefetch(&q->data[(i + 1) * cols], 0, 1);
#ifdef __ARM_NEON
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (int j = 0; j < cols64; j += 64) {
            acc0 = vdotq_s32(acc0, vld1q_s8(&x_q[j]),      vld1q_s8(&row[j]));
            acc1 = vdotq_s32(acc1, vld1q_s8(&x_q[j + 16]), vld1q_s8(&row[j + 16]));
            acc2 = vdotq_s32(acc2, vld1q_s8(&x_q[j + 32]), vld1q_s8(&row[j + 32]));
            acc3 = vdotq_s32(acc3, vld1q_s8(&x_q[j + 48]), vld1q_s8(&row[j + 48]));
        }
        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        for (int j = cols64; j < cols16; j += 16)
            acc = vdotq_s32(acc, vld1q_s8(&x_q[j]), vld1q_s8(&row[j]));
        int32_t dot = vaddvq_s32(acc);
        for (int j = cols16; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#else
        int32_t dot = 0;
        for (int j = 0; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#endif
        out[i] = (float)dot * w_scale * x_scale;
    }
}
#endif

/* Quantized matvec: convenience wrapper that quantizes x on the fly. */
static void fast_matvec_q8(const float *x, const rx_q8_weight *q, float *out,
                           int rows, int cols, int8_t *q8_buf) {
    float x_scale = fast_quantize_input(x, q8_buf, cols);
    fast_matvec_q8_preq(q8_buf, x_scale, q, out, rows, cols);
}

/* ================================================================
 * Internal: RWKV-7 block weights
 * ================================================================ */

typedef struct {
    tm_tensor *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    tm_tensor *x_r, *x_w, *x_k, *x_v, *x_a, *x_g;
    tm_tensor *w0, *w1, *w2;
    tm_tensor *a0, *a1, *a2;
    tm_tensor *v0, *v1, *v2;
    tm_tensor *g1, *g2;
    tm_tensor *k_k, *k_a, *r_k;
    tm_tensor *receptance_w, *key_w, *value_w, *output_w;
    tm_tensor *ln_x_w, *ln_x_b;
    tm_tensor *ffn_x_k, *ffn_key_w, *ffn_value_w;

    rx_q8_weight *receptance_q8, *key_q8, *value_q8, *output_q8;
    rx_q8_weight *ffn_key_q8, *ffn_value_q8;
} rx_rwkv_block;

/* ================================================================
 * Internal: Sparse attention block weights
 * ================================================================ */

typedef struct {
    tm_tensor *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    tm_tensor *receptance_w, *key_w, *value_w, *output_w;
    tm_tensor *ffn_x_k, *ffn_key_w, *ffn_value_w;

    rx_q8_weight *receptance_q8, *key_q8, *value_q8, *output_q8;
    rx_q8_weight *ffn_key_q8, *ffn_value_q8;
} rx_attn_block;

/* ================================================================
 * Internal: Model configuration
 * ================================================================ */

typedef struct {
    int n_embd;
    int n_head;
    int head_size;
    int n_rwkv_layer;
    int n_attn_layer;
    int vocab_size;
} rx_config;

/* ================================================================
 * Internal: Model structure
 * ================================================================ */

typedef struct rx_model_internal rx_model_internal;

struct rx_model_internal {
    rx_config config;

    tm_tensor *emb_weight;
    tm_tensor *ln_out_w;
    tm_tensor *ln_out_b;
    tm_tensor *head_weight;
    rx_q8_weight *head_q8;

    rx_rwkv_block *rwkv_blocks;
    rx_attn_block *attn_blocks;

    int *block_order;
    int *block_index;
    int n_blocks;

    bool quantized;

#ifdef USE_THREADS
    fw_pool *pool;
#endif

    /* Tokenizer (loaded separately) */
    void *tokenizer;  /* rx_tokenizer* */
};

/* ================================================================
 * Internal: State structure
 * ================================================================ */

typedef struct {
    tm_tensor *att_x_prev;
    tm_tensor *att_kv;
    tm_tensor *ffn_x_prev;
} rx_rwkv_state;

typedef struct {
    tm_tensor *k_cache;
    tm_tensor *v_cache;
    int cache_len;
    int max_cache_len;
    tm_tensor *ffn_x_prev;
} rx_attn_state;

/* ================================================================
 * Internal: Scratch workspace
 * ================================================================ */

#define RX_NUM_SCRATCH 16

typedef struct {
    float **buf;
    float *lora_buf;
    float *ffn_buf;
    float *x_buf[2];
    int8_t *q8_buf;
    int c;
    int max_lora_dim;
    int ffn_dim;
} rx_workspace;

typedef struct {
    rx_rwkv_state *rwkv;
    rx_attn_state *attn;
    tm_tensor *v_first;
    float *v_first_buf;
    rx_config config;
    rx_workspace *workspace;
} rx_state_internal;

static rx_workspace* rx_workspace_new(int c, int max_lora_dim, int ffn_dim) {
    rx_workspace *ws = (rx_workspace*)calloc(1, sizeof(rx_workspace));
    ws->c = c;
    ws->max_lora_dim = max_lora_dim > 0 ? max_lora_dim : 256;
    ws->ffn_dim = ffn_dim > 0 ? ffn_dim : c * 4;
    ws->buf = (float**)malloc(RX_NUM_SCRATCH * sizeof(float*));
    for (int i = 0; i < RX_NUM_SCRATCH; i++)
        ws->buf[i] = (float*)aligned_malloc(c * sizeof(float));
    ws->lora_buf = (float*)aligned_malloc(ws->max_lora_dim * sizeof(float));
    ws->ffn_buf = (float*)aligned_malloc(ws->ffn_dim * sizeof(float));
    ws->x_buf[0] = (float*)aligned_malloc(c * sizeof(float));
    ws->x_buf[1] = (float*)aligned_malloc(c * sizeof(float));
    int q8_size = ws->ffn_dim > c ? ws->ffn_dim : c;
    ws->q8_buf = (int8_t*)aligned_malloc(q8_size * sizeof(int8_t));
    return ws;
}

static void rx_workspace_free(rx_workspace *ws) {
    if (!ws) return;
    for (int i = 0; i < RX_NUM_SCRATCH; i++)
        free(ws->buf[i]);
    free(ws->buf);
    free(ws->lora_buf);
    free(ws->ffn_buf);
    free(ws->x_buf[0]);
    free(ws->x_buf[1]);
    free(ws->q8_buf);
    free(ws);
}

/* ================================================================
 * Internal: Tokenizer trie
 * ================================================================ */

#define RX_VOCAB_SIZE 65536
#define RX_MAX_TOKEN_LEN 128

typedef struct rx_trie_node {
    struct rx_trie_node *children[256];
    int32_t token_id;
} rx_trie_node;

typedef struct {
    rx_trie_node *root;
    uint8_t *token_bytes[RX_VOCAB_SIZE];
    int token_lens[RX_VOCAB_SIZE];
} rx_tokenizer;

/* ================================================================
 * Zero-allocation helpers
 * ================================================================ */

/* Matrix-vector: out[D] = x[C] @ w[C, D] */
static void fast_matvec(const float *x, const float *w, float *out,
                        int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasTrans,
                rows, cols,
                1.0f, w, cols, x, 1, 0.0f, out, 1);
#else
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        float xi = x[i];
        const float *wi = &w[i * cols];
        for (int j = 0; j < cols; j++)
            out[j] += xi * wi[j];
    }
#endif
}

/* Layer norm: out = (x - mean) / sqrt(var + eps) * weight + bias */
static void fast_layer_norm(const float *x, const float *w, const float *b,
                            float *out, int c) {
#ifdef __ARM_NEON
    int c4 = c & ~3;
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    for (int i = 0; i < c4; i += 4)
        sum_v = vaddq_f32(sum_v, vld1q_f32(&x[i]));
    float mean = vaddvq_f32(sum_v);
    for (int i = c4; i < c; i++) mean += x[i];
    mean /= c;

    float32x4_t mean_v = vdupq_n_f32(mean);
    float32x4_t var_v = vdupq_n_f32(0.0f);
    for (int i = 0; i < c4; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&x[i]), mean_v);
        var_v = vfmaq_f32(var_v, d, d);
    }
    float var = vaddvq_f32(var_v);
    for (int i = c4; i < c; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= c;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    float32x4_t inv_std_v = vdupq_n_f32(inv_std);
    for (int i = 0; i < c4; i += 4) {
        float32x4_t xv = vsubq_f32(vld1q_f32(&x[i]), mean_v);
        float32x4_t normed = vmulq_f32(xv, inv_std_v);
        vst1q_f32(&out[i], vfmaq_f32(vld1q_f32(&b[i]),
                                       normed, vld1q_f32(&w[i])));
    }
    for (int i = c4; i < c; i++)
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
#else
    float mean = 0.0f;
    for (int i = 0; i < c; i++) mean += x[i];
    mean /= c;

    float var = 0.0f;
    for (int i = 0; i < c; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= c;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < c; i++)
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
#endif
}

/* Group norm: C elements in num_groups groups, eps=64e-5 per RWKV convention */
static void fast_group_norm(const float *x, int num_groups, int group_size,
                            const float *weight, const float *bias,
                            float *out) {
    for (int g = 0; g < num_groups; g++) {
        const float *gx = &x[g * group_size];
        float *go = &out[g * group_size];
        int off = g * group_size;

#ifdef __ARM_NEON
        int gs4 = group_size & ~3;
        float32x4_t sum_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < gs4; i += 4)
            sum_v = vaddq_f32(sum_v, vld1q_f32(&gx[i]));
        float mean = vaddvq_f32(sum_v);
        for (int i = gs4; i < group_size; i++) mean += gx[i];
        mean /= group_size;

        float32x4_t mean_v = vdupq_n_f32(mean);
        float32x4_t var_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < gs4; i += 4) {
            float32x4_t d = vsubq_f32(vld1q_f32(&gx[i]), mean_v);
            var_v = vfmaq_f32(var_v, d, d);
        }
        float var = vaddvq_f32(var_v);
        for (int i = gs4; i < group_size; i++) {
            float d = gx[i] - mean;
            var += d * d;
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + 64e-5f);
        float32x4_t inv_std_v = vdupq_n_f32(inv_std);
        for (int i = 0; i < gs4; i += 4) {
            float32x4_t xv = vsubq_f32(vld1q_f32(&gx[i]), mean_v);
            float32x4_t normed = vmulq_f32(xv, inv_std_v);
            vst1q_f32(&go[i], vfmaq_f32(vld1q_f32(&bias[off + i]),
                                          normed, vld1q_f32(&weight[off + i])));
        }
        for (int i = gs4; i < group_size; i++)
            go[i] = (gx[i] - mean) * inv_std * weight[off + i] + bias[off + i];
#else
        float mean = 0.0f;
        for (int i = 0; i < group_size; i++) mean += gx[i];
        mean /= group_size;

        float var = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float d = gx[i] - mean;
            var += d * d;
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + 64e-5f);
        for (int i = 0; i < group_size; i++)
            go[i] = (gx[i] - mean) * inv_std * weight[off + i] + bias[off + i];
#endif
    }
}

/* L2 normalize per head */
static void fast_l2_normalize_per_head(const float *x, float *out,
                                        int n_head, int head_size) {
    for (int h = 0; h < n_head; h++) {
        const float *hx = &x[h * head_size];
        float *ho = &out[h * head_size];
#ifdef __ARM_NEON
        int hs4 = head_size & ~3;
        float32x4_t sq_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < hs4; i += 4) {
            float32x4_t v = vld1q_f32(&hx[i]);
            sq_v = vfmaq_f32(sq_v, v, v);
        }
        float sq_sum = vaddvq_f32(sq_v);
        for (int i = hs4; i < head_size; i++)
            sq_sum += hx[i] * hx[i];

        float inv_norm = 1.0f / (sqrtf(sq_sum) + 1e-12f);
        float32x4_t inv_v = vdupq_n_f32(inv_norm);
        for (int i = 0; i < hs4; i += 4)
            vst1q_f32(&ho[i], vmulq_f32(vld1q_f32(&hx[i]), inv_v));
        for (int i = hs4; i < head_size; i++)
            ho[i] = hx[i] * inv_norm;
#else
        float sq_sum = 0.0f;
        for (int i = 0; i < head_size; i++)
            sq_sum += hx[i] * hx[i];
        float inv_norm = 1.0f / (sqrtf(sq_sum) + 1e-12f);
        for (int i = 0; i < head_size; i++)
            ho[i] = hx[i] * inv_norm;
#endif
    }
}

/* Token shift + lerp: out = xx + (prev - xx) * lerp_coeff */
static void fast_token_shift(const float *xx, const float *prev,
                              const float *lerp, float *out, int c) {
#ifdef __ARM_NEON
    int c4 = c & ~3;
    for (int i = 0; i < c4; i += 4) {
        float32x4_t xx_v = vld1q_f32(&xx[i]);
        float32x4_t diff = vsubq_f32(vld1q_f32(&prev[i]), xx_v);
        vst1q_f32(&out[i], vfmaq_f32(xx_v, diff, vld1q_f32(&lerp[i])));
    }
    for (int i = c4; i < c; i++)
        out[i] = xx[i] + (prev[i] - xx[i]) * lerp[i];
#else
    for (int i = 0; i < c; i++)
        out[i] = xx[i] + (prev[i] - xx[i]) * lerp[i];
#endif
}

/* NEON residual add helper: out[i] += src[i] */
static inline void fast_residual_add(float *out, const float *src, int c) {
#ifdef __ARM_NEON
    int c4 = c & ~3;
    for (int i = 0; i < c4; i += 4)
        vst1q_f32(&out[i], vaddq_f32(vld1q_f32(&out[i]), vld1q_f32(&src[i])));
    for (int i = c4; i < c; i++)
        out[i] += src[i];
#else
    for (int i = 0; i < c; i++)
        out[i] += src[i];
#endif
}

/* ================================================================
 * WKV-7 kernel (single timestep, zero-allocation)
 * ================================================================ */

#ifdef USE_THREADS
typedef struct {
    const float *r_d, *w_d, *k_d, *v_d, *nkk_d, *kka_d;
    float *s_d, *y_out;
    int n_head, head_size;
} wkv7_ctx;

static void wkv7_head_worker(int start, int end, void *arg);
#endif

static void rx_wkv7_one(
    const float *r_d, const float *w_d, const float *k_d, const float *v_d,
    const float *nkk_d, const float *kka_d,
    float *s_d, float *y_out,
    int n_head, int head_size)
{
    int n = head_size;

#ifdef USE_THREADS
    if (__rx_pool) {
        wkv7_ctx ctx = { r_d, w_d, k_d, v_d, nkk_d, kka_d,
                         s_d, y_out, n_head, head_size };
        fw_for(__rx_pool, n_head, wkv7_head_worker, &ctx);
        return;
    }
#endif
    for (int h = 0; h < n_head; h++) {
        int off = h * n;
        for (int i = 0; i < n; i++) {
            int sbase = (h * n + i) * n;
            if (i + 1 < n)
                __builtin_prefetch(&s_d[(h * n + (i + 1)) * n], 1, 3);

#ifdef __ARM_NEON
            int n4 = n & ~3;
            float32x4_t sa_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4)
                sa_v = vfmaq_f32(sa_v, vld1q_f32(&nkk_d[off + j]),
                                       vld1q_f32(&s_d[sbase + j]));
            float sa = vaddvq_f32(sa_v);
            for (int j = n4; j < n; j++)
                sa += nkk_d[off + j] * s_d[sbase + j];

            float vv = v_d[off + i];
            float32x4_t vv_v = vdupq_n_f32(vv);
            float32x4_t sa_bcast = vdupq_n_f32(sa);
            float32x4_t yy_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4) {
                float32x4_t s = vld1q_f32(&s_d[sbase + j]);
                float32x4_t w = vld1q_f32(&w_d[off + j]);
                float32x4_t k = vld1q_f32(&k_d[off + j]);
                float32x4_t kka = vld1q_f32(&kka_d[off + j]);
                s = vfmaq_f32(vfmaq_f32(vmulq_f32(s, w), k, vv_v), kka, sa_bcast);
                vst1q_f32(&s_d[sbase + j], s);
                yy_v = vfmaq_f32(yy_v, s, vld1q_f32(&r_d[off + j]));
            }
            float yy = vaddvq_f32(yy_v);
            for (int j = n4; j < n; j++) {
                s_d[sbase + j] = s_d[sbase + j] * w_d[off + j]
                                + k_d[off + j] * vv
                                + sa * kka_d[off + j];
                yy += s_d[sbase + j] * r_d[off + j];
            }
#else
            float sa = 0.0f;
            for (int j = 0; j < n; j++)
                sa += nkk_d[off + j] * s_d[sbase + j];

            float vv = v_d[off + i];
            float yy = 0.0f;
            for (int j = 0; j < n; j++) {
                s_d[sbase + j] = s_d[sbase + j] * w_d[off + j]
                                + k_d[off + j] * vv
                                + sa * kka_d[off + j];
                yy += s_d[sbase + j] * r_d[off + j];
            }
#endif
            y_out[off + i] = yy;
        }
    }
}

#ifdef USE_THREADS
static void wkv7_head_worker(int start, int end, void *arg) {
    wkv7_ctx *ctx = (wkv7_ctx *)arg;
    int n = ctx->head_size;
    for (int h = start; h < end; h++) {
        int off = h * n;
        for (int i = 0; i < n; i++) {
            int sbase = (h * n + i) * n;
            if (i + 1 < n)
                __builtin_prefetch(&ctx->s_d[(h * n + (i + 1)) * n], 1, 3);
#ifdef __ARM_NEON
            int n4 = n & ~3;
            float32x4_t sa_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4)
                sa_v = vfmaq_f32(sa_v, vld1q_f32(&ctx->nkk_d[off + j]),
                                       vld1q_f32(&ctx->s_d[sbase + j]));
            float sa = vaddvq_f32(sa_v);
            for (int j = n4; j < n; j++)
                sa += ctx->nkk_d[off + j] * ctx->s_d[sbase + j];

            float vv = ctx->v_d[off + i];
            float32x4_t vv_v = vdupq_n_f32(vv);
            float32x4_t sa_bcast = vdupq_n_f32(sa);
            float32x4_t yy_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4) {
                float32x4_t s = vld1q_f32(&ctx->s_d[sbase + j]);
                float32x4_t w = vld1q_f32(&ctx->w_d[off + j]);
                float32x4_t k = vld1q_f32(&ctx->k_d[off + j]);
                float32x4_t kka = vld1q_f32(&ctx->kka_d[off + j]);
                s = vfmaq_f32(vfmaq_f32(vmulq_f32(s, w), k, vv_v), kka, sa_bcast);
                vst1q_f32(&ctx->s_d[sbase + j], s);
                yy_v = vfmaq_f32(yy_v, s, vld1q_f32(&ctx->r_d[off + j]));
            }
            float yy = vaddvq_f32(yy_v);
            for (int j = n4; j < n; j++) {
                ctx->s_d[sbase + j] = ctx->s_d[sbase + j] * ctx->w_d[off + j]
                                + ctx->k_d[off + j] * vv
                                + sa * ctx->kka_d[off + j];
                yy += ctx->s_d[sbase + j] * ctx->r_d[off + j];
            }
#else
            float sa = 0.0f;
            for (int j = 0; j < n; j++)
                sa += ctx->nkk_d[off + j] * ctx->s_d[sbase + j];
            float vv = ctx->v_d[off + i];
            float yy = 0.0f;
            for (int j = 0; j < n; j++) {
                ctx->s_d[sbase + j] = ctx->s_d[sbase + j] * ctx->w_d[off + j]
                                + ctx->k_d[off + j] * vv
                                + sa * ctx->kka_d[off + j];
                yy += ctx->s_d[sbase + j] * ctx->r_d[off + j];
            }
#endif
            ctx->y_out[off + i] = yy;
        }
    }
}
#endif

/* ================================================================
 * RWKV-7 block forward (single token)
 * ================================================================ */

static void rwkv7_time_mix_fast(
    rx_rwkv_block *blk, int layer_id,
    const float *xx, rx_rwkv_state *st, float *v_first,
    int n_head, int head_size, rx_workspace *ws,
    float *output_buf)
{
    int c = n_head * head_size;
    float *prev = tm_data_f32(st->att_x_prev);

    float *shifted = ws->buf[0];
    float *r_vec   = ws->buf[1];
    float *k_vec   = ws->buf[2];
    float *v_vec   = ws->buf[3];
    float *w_vec   = ws->buf[4];
    float *a_vec   = ws->buf[5];
    float *g_vec   = ws->buf[6];
    float *kk      = ws->buf[7];
    float *k_mod   = ws->buf[8];
    float *v_final = ws->buf[9];
    float *y       = ws->buf[10];
    float *y_gn    = ws->buf[11];
    float *tmp     = ws->buf[12];
    float *tmp2    = ws->buf[13];
    float *xv_save = ws->buf[14];
    float *lora    = ws->lora_buf;

    int8_t *q8_buf = ws->q8_buf;
    #define MATVEC(x_arg, w_tensor, q8_arg, out_arg, r_arg) do { \
        if (q8_arg) fast_matvec_q8((x_arg), (q8_arg), (out_arg), (q8_arg)->rows, (q8_arg)->cols, q8_buf); \
        else fast_matvec((x_arg), tm_data_f32(w_tensor), (out_arg), (r_arg), (int)tm_size((w_tensor), 1)); \
    } while(0)

    fast_token_shift(xx, prev, tm_data_f32(blk->x_r), shifted, c);
    MATVEC(shifted, blk->receptance_w, blk->receptance_q8, r_vec, c);

    fast_token_shift(xx, prev, tm_data_f32(blk->x_k), shifted, c);
    MATVEC(shifted, blk->key_w, blk->key_q8, k_vec, c);

    fast_token_shift(xx, prev, tm_data_f32(blk->x_v), shifted, c);
    memcpy(xv_save, shifted, c * sizeof(float));
    MATVEC(shifted, blk->value_w, blk->value_q8, v_vec, c);

    /* Decay gate: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2)) */
    {
        int lora_dim = (int)tm_size(blk->w1, 1);
        float *w0_d = tm_data_f32(blk->w0);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_w), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->w1), lora, c, lora_dim);

#if defined(USE_BLAS) && defined(__APPLE__)
        vvtanhf(lora, lora, &lora_dim);
#else
        for (int i = 0; i < lora_dim; i++)
            lora[i] = tanhf(lora[i]);
#endif

        fast_matvec(lora, tm_data_f32(blk->w2), w_vec, lora_dim, c);

#if defined(USE_BLAS) && defined(__APPLE__)
        vDSP_vadd(w0_d, 1, w_vec, 1, w_vec, 1, c);
        vDSP_vneg(w_vec, 1, w_vec, 1, c);
        vvexpf(w_vec, w_vec, &c);
        float one = 1.0f;
        vDSP_vsadd(w_vec, 1, &one, w_vec, 1, c);
        vvrecf(w_vec, w_vec, &c);
        float neg_coeff = -0.606531f;
        vDSP_vsmul(w_vec, 1, &neg_coeff, w_vec, 1, c);
        vvexpf(w_vec, w_vec, &c);
#else
        for (int i = 0; i < c; i++) {
            float sig = 1.0f / (1.0f + expf(-(w0_d[i] + w_vec[i])));
            w_vec[i] = expf(-0.606531f * sig);
        }
#endif
    }

    /* Gate a: a = sigmoid(a0 + (xa @ a1) @ a2) */
    {
        int lora_dim = (int)tm_size(blk->a1, 1);
        float *a0_d = tm_data_f32(blk->a0);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_a), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->a1), lora, c, lora_dim);
        fast_matvec(lora, tm_data_f32(blk->a2), a_vec, lora_dim, c);

#if defined(USE_BLAS) && defined(__APPLE__)
        vDSP_vadd(a0_d, 1, a_vec, 1, a_vec, 1, c);
        vDSP_vneg(a_vec, 1, a_vec, 1, c);
        vvexpf(a_vec, a_vec, &c);
        float one_a = 1.0f;
        vDSP_vsadd(a_vec, 1, &one_a, a_vec, 1, c);
        vvrecf(a_vec, a_vec, &c);
#else
        for (int i = 0; i < c; i++)
            a_vec[i] = 1.0f / (1.0f + expf(-(a0_d[i] + a_vec[i])));
#endif
    }

    /* Gate g: g = sigmoid(xg @ g1) @ g2 */
    {
        int lora_dim = (int)tm_size(blk->g1, 1);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_g), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->g1), lora, c, lora_dim);

#if defined(USE_BLAS) && defined(__APPLE__)
        vDSP_vneg(lora, 1, lora, 1, lora_dim);
        vvexpf(lora, lora, &lora_dim);
        float one_g = 1.0f;
        vDSP_vsadd(lora, 1, &one_g, lora, 1, lora_dim);
        vvrecf(lora, lora, &lora_dim);
#else
        for (int i = 0; i < lora_dim; i++)
            lora[i] = 1.0f / (1.0f + expf(-lora[i]));
#endif

        fast_matvec(lora, tm_data_f32(blk->g2), g_vec, lora_dim, c);
    }

    /* Update prev for next token */
    memcpy(prev, xx, c * sizeof(float));

    /* Key normalization: kk = L2_normalize(k * k_k), per head */
    {
        float *kk_d = tm_data_f32(blk->k_k);
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&tmp[i], vmulq_f32(vld1q_f32(&k_vec[i]),
                                           vld1q_f32(&kk_d[i])));
        for (int i = c4; i < c; i++)
            tmp[i] = k_vec[i] * kk_d[i];
#else
        for (int i = 0; i < c; i++)
            tmp[i] = k_vec[i] * kk_d[i];
#endif
        fast_l2_normalize_per_head(tmp, kk, n_head, head_size);
    }

    /* Modify key: k_mod = k * (1 + (a-1) * k_a) */
    {
        float *ka_d = tm_data_f32(blk->k_a);
#ifdef __ARM_NEON
        int c4 = c & ~3;
        float32x4_t one_v = vdupq_n_f32(1.0f);
        for (int i = 0; i < c4; i += 4) {
            float32x4_t am1 = vsubq_f32(vld1q_f32(&a_vec[i]), one_v);
            float32x4_t factor = vfmaq_f32(one_v, am1, vld1q_f32(&ka_d[i]));
            vst1q_f32(&k_mod[i], vmulq_f32(vld1q_f32(&k_vec[i]), factor));
        }
        for (int i = c4; i < c; i++)
            k_mod[i] = k_vec[i] * (1.0f + (a_vec[i] - 1.0f) * ka_d[i]);
#else
        for (int i = 0; i < c; i++)
            k_mod[i] = k_vec[i] * (1.0f + (a_vec[i] - 1.0f) * ka_d[i]);
#endif
    }

    /* Value first-token gating */
    if (layer_id == 0) {
        memcpy(v_final, v_vec, c * sizeof(float));
        memcpy(v_first, v_vec, c * sizeof(float));
    } else {
        int lora_dim = (int)tm_size(blk->v1, 1);
        float *v0_d = tm_data_f32(blk->v0);

        fast_matvec(xv_save, tm_data_f32(blk->v1), lora, c, lora_dim);
        fast_matvec(lora, tm_data_f32(blk->v2), tmp, lora_dim, c);

#if defined(USE_BLAS) && defined(__APPLE__)
        vDSP_vadd(v0_d, 1, tmp, 1, tmp, 1, c);
        vDSP_vneg(tmp, 1, tmp, 1, c);
        vvexpf(tmp, tmp, &c);
        float one_v = 1.0f;
        vDSP_vsadd(tmp, 1, &one_v, tmp, 1, c);
        vvrecf(tmp, tmp, &c);
  #ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4) {
            float32x4_t vv = vld1q_f32(&v_vec[i]);
            float32x4_t diff = vsubq_f32(vld1q_f32(&v_first[i]), vv);
            vst1q_f32(&v_final[i], vfmaq_f32(vv, diff, vld1q_f32(&tmp[i])));
        }
        for (int i = c4; i < c; i++)
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * tmp[i];
  #else
        for (int i = 0; i < c; i++)
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * tmp[i];
  #endif
#else
        for (int i = 0; i < c; i++) {
            float blend = 1.0f / (1.0f + expf(-(v0_d[i] + tmp[i])));
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * blend;
        }
#endif
    }

    /* WKV-7 kernel */
    {
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4) {
            float32x4_t kk_v = vld1q_f32(&kk[i]);
            vst1q_f32(&tmp[i], vnegq_f32(kk_v));
            vst1q_f32(&tmp2[i], vmulq_f32(kk_v, vld1q_f32(&a_vec[i])));
        }
        for (int i = c4; i < c; i++) {
            tmp[i] = -kk[i];
            tmp2[i] = kk[i] * a_vec[i];
        }
#else
        for (int i = 0; i < c; i++) {
            tmp[i] = -kk[i];
            tmp2[i] = kk[i] * a_vec[i];
        }
#endif
        rx_wkv7_one(r_vec, w_vec, k_mod, v_final,
                     tmp, tmp2, tm_data_f32(st->att_kv), y,
                     n_head, head_size);
    }

    /* Group norm */
    fast_group_norm(y, n_head, head_size,
                    tm_data_f32(blk->ln_x_w), tm_data_f32(blk->ln_x_b),
                    y_gn);

    /* Bonus: per-head (r * k_mod * r_k).sum() * v_final */
    {
        float *rk_d = tm_data_f32(blk->r_k);
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t sum_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4) {
                float32x4_t rv = vld1q_f32(&r_vec[off + j]);
                float32x4_t km = vld1q_f32(&k_mod[off + j]);
                float32x4_t rk = vld1q_f32(&rk_d[off + j]);
                sum_v = vfmaq_f32(sum_v, vmulq_f32(rv, km), rk);
            }
            float sum = vaddvq_f32(sum_v);
            for (int j = hs4; j < head_size; j++)
                sum += r_vec[off + j] * k_mod[off + j] * rk_d[off + j];

            float32x4_t sum_bcast = vdupq_n_f32(sum);
            for (int j = 0; j < hs4; j += 4) {
                float32x4_t cur = vld1q_f32(&y_gn[off + j]);
                vst1q_f32(&y_gn[off + j],
                           vfmaq_f32(cur, sum_bcast, vld1q_f32(&v_final[off + j])));
            }
            for (int j = hs4; j < head_size; j++)
                y_gn[off + j] += sum * v_final[off + j];
#else
            float sum = 0.0f;
            for (int j = 0; j < head_size; j++)
                sum += r_vec[off + j] * k_mod[off + j] * rk_d[off + j];
            for (int j = 0; j < head_size; j++)
                y_gn[off + j] += sum * v_final[off + j];
#endif
        }
    }

    /* Output: (y_bonus * g) @ output_w */
    {
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&tmp[i], vmulq_f32(vld1q_f32(&y_gn[i]),
                                           vld1q_f32(&g_vec[i])));
        for (int i = c4; i < c; i++)
            tmp[i] = y_gn[i] * g_vec[i];
#else
        for (int i = 0; i < c; i++)
            tmp[i] = y_gn[i] * g_vec[i];
#endif
        MATVEC(tmp, blk->output_w, blk->output_q8, output_buf, c);
    }

    #undef MATVEC
}

/* Channel mix (ReLU^2 FFN) for a single token */
static void rwkv7_channel_mix_fast(
    rx_rwkv_block *blk, const float *xx, rx_rwkv_state *st,
    rx_workspace *ws, float *output_buf)
{
    int c = ws->c;
    float *prev = tm_data_f32(st->ffn_x_prev);
    float *shifted = ws->buf[0];
    int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows
                                  : (int)tm_size(blk->ffn_key_w, 1);

    fast_token_shift(xx, prev, tm_data_f32(blk->ffn_x_k), shifted, c);
    memcpy(prev, xx, c * sizeof(float));

    float *fk_buf = ws->ffn_buf;
    int8_t *q8_buf = ws->q8_buf;

    if (blk->ffn_key_q8)
        fast_matvec_q8(shifted, blk->ffn_key_q8, fk_buf,
                       blk->ffn_key_q8->rows, blk->ffn_key_q8->cols, q8_buf);
    else
        fast_matvec(shifted, tm_data_f32(blk->ffn_key_w), fk_buf, c, ffn_dim);

    /* relu^2 */
#ifdef __ARM_NEON
    {
        int fd4 = ffn_dim & ~3;
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (int i = 0; i < fd4; i += 4) {
            float32x4_t v = vmaxq_f32(vld1q_f32(&fk_buf[i]), zero);
            vst1q_f32(&fk_buf[i], vmulq_f32(v, v));
        }
        for (int i = fd4; i < ffn_dim; i++) {
            float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
            fk_buf[i] = v * v;
        }
    }
#else
    for (int i = 0; i < ffn_dim; i++) {
        float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
        fk_buf[i] = v * v;
    }
#endif

    if (blk->ffn_value_q8)
        fast_matvec_q8(fk_buf, blk->ffn_value_q8, output_buf,
                       blk->ffn_value_q8->rows, blk->ffn_value_q8->cols, q8_buf);
    else
        fast_matvec(fk_buf, tm_data_f32(blk->ffn_value_w), output_buf, ffn_dim, c);
}

/* Full RWKV-7 block forward */
static void rwkv7_block_forward_fast(
    rx_rwkv_block *blk, int layer_id,
    float *x_data, rx_rwkv_state *st, float *v_first,
    int n_head, int head_size, rx_workspace *ws,
    float *x_out)
{
    int c = n_head * head_size;
    float *ln_out = ws->buf[15];

    fast_layer_norm(x_data, tm_data_f32(blk->ln1_w), tm_data_f32(blk->ln1_b),
                    ln_out, c);

    rwkv7_time_mix_fast(blk, layer_id, ln_out, st, v_first,
                         n_head, head_size, ws, x_out);
    fast_residual_add(x_out, x_data, c);

    fast_layer_norm(x_out, tm_data_f32(blk->ln2_w), tm_data_f32(blk->ln2_b),
                    ln_out, c);

    float *cmix_out = ws->buf[0];
    rwkv7_channel_mix_fast(blk, ln_out, st, ws, cmix_out);
    fast_residual_add(x_out, cmix_out, c);
}

/* ================================================================
 * Sparse attention block forward (single token, decode mode)
 * Stripped: HybridMem, instrumentation, embedding capture
 * ================================================================ */

static void attn_block_forward_fast(
    rx_attn_block *blk, float *x_data, rx_attn_state *st,
    int n_head, int head_size, rx_workspace *ws,
    float *x_out)
{
    int c = n_head * head_size;
    float *ln_out  = ws->buf[15];
    float *q_vec   = ws->buf[1];
    float *k_vec   = ws->buf[2];
    float *v_vec   = ws->buf[3];
    float *y_data  = ws->buf[10];
    float *att_out = ws->buf[11];

    fast_layer_norm(x_data, tm_data_f32(blk->ln1_w), tm_data_f32(blk->ln1_b),
                    ln_out, c);

    /* Q/K/V projections */
    int8_t *q8_buf = ws->q8_buf;
    if (blk->receptance_q8) {
        float ln_q8_scale = fast_quantize_input(ln_out, q8_buf, c);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->receptance_q8, q_vec,
                            blk->receptance_q8->rows, blk->receptance_q8->cols);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->key_q8, k_vec,
                            blk->key_q8->rows, blk->key_q8->cols);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->value_q8, v_vec,
                            blk->value_q8->rows, blk->value_q8->cols);
    } else {
        fast_matvec(ln_out, tm_data_f32(blk->receptance_w), q_vec, c, c);
        fast_matvec(ln_out, tm_data_f32(blk->key_w), k_vec, c, c);
        fast_matvec(ln_out, tm_data_f32(blk->value_w), v_vec, c, c);
    }

    /* Append K/V to cache — simple eviction (discard oldest half) */
    float *k_cache = tm_data_f32(st->k_cache);
    float *v_cache = tm_data_f32(st->v_cache);

    if (st->cache_len >= st->max_cache_len) {
        int evict_count = st->max_cache_len / 2;
        if (evict_count < 1) evict_count = 1;
        int remain = st->cache_len - evict_count;
        if (remain > 0) {
            memmove(k_cache, &k_cache[evict_count * c], remain * c * sizeof(float));
            memmove(v_cache, &v_cache[evict_count * c], remain * c * sizeof(float));
        }
        st->cache_len = remain;
    }

    memcpy(&k_cache[st->cache_len * c], k_vec, c * sizeof(float));
    memcpy(&v_cache[st->cache_len * c], v_vec, c * sizeof(float));
    st->cache_len++;

    int ct = st->cache_len;
    float scale = 1.0f / sqrtf((float)head_size);

    memset(y_data, 0, c * sizeof(float));

    /* Per-head attention */
    for (int h = 0; h < n_head; h++) {
        int off = h * head_size;
        float attn_scores_local[ct];

        /* Q @ K^T */
        for (int t = 0; t < ct; t++) {
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t dot_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4)
                dot_v = vfmaq_f32(dot_v, vld1q_f32(&q_vec[off + j]),
                                         vld1q_f32(&k_cache[t * c + off + j]));
            float dot = vaddvq_f32(dot_v);
            for (int j = hs4; j < head_size; j++)
                dot += q_vec[off + j] * k_cache[t * c + off + j];
#else
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += q_vec[off + j] * k_cache[t * c + off + j];
#endif
            if (t + 1 < ct)
                __builtin_prefetch(&k_cache[(t + 1) * c + off], 0, 1);
            attn_scores_local[t] = dot * scale;
        }

        /* Softmax */
        float max_s = attn_scores_local[0];
        for (int t = 1; t < ct; t++)
            if (attn_scores_local[t] > max_s) max_s = attn_scores_local[t];

        float sum_e = 0.0f;
        for (int t = 0; t < ct; t++) {
            attn_scores_local[t] = expf(attn_scores_local[t] - max_s);
            sum_e += attn_scores_local[t];
        }
        for (int t = 0; t < ct; t++)
            attn_scores_local[t] /= sum_e;

        /* Value aggregation */
        for (int j = 0; j < head_size; j++) {
            float val = 0.0f;
            for (int t = 0; t < ct; t++)
                val += attn_scores_local[t] * v_cache[t * c + off + j];
            y_data[off + j] = val;
        }
    }

    /* Output projection */
    if (blk->output_q8)
        fast_matvec_q8(y_data, blk->output_q8, att_out,
                       blk->output_q8->rows, blk->output_q8->cols, q8_buf);
    else
        fast_matvec(y_data, tm_data_f32(blk->output_w), att_out, c, c);

    /* Residual: x1 = x + att_out */
#ifdef __ARM_NEON
    {
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&x_out[i], vaddq_f32(vld1q_f32(&x_data[i]),
                                             vld1q_f32(&att_out[i])));
        for (int i = c4; i < c; i++)
            x_out[i] = x_data[i] + att_out[i];
    }
#else
    for (int i = 0; i < c; i++)
        x_out[i] = x_data[i] + att_out[i];
#endif

    /* Pre-FFN LayerNorm */
    fast_layer_norm(x_out, tm_data_f32(blk->ln2_w), tm_data_f32(blk->ln2_b),
                    ln_out, c);

    /* Channel mix: token shift + relu^2 FFN */
    {
        float *ffn_prev = tm_data_f32(st->ffn_x_prev);
        float *shifted = ws->buf[0];
        int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows
                                      : (int)tm_size(blk->ffn_key_w, 1);
        float *fk_buf = ws->ffn_buf;

        fast_token_shift(ln_out, ffn_prev, tm_data_f32(blk->ffn_x_k), shifted, c);
        memcpy(ffn_prev, ln_out, c * sizeof(float));

        if (blk->ffn_key_q8)
            fast_matvec_q8(shifted, blk->ffn_key_q8, fk_buf,
                           blk->ffn_key_q8->rows, blk->ffn_key_q8->cols, q8_buf);
        else
            fast_matvec(shifted, tm_data_f32(blk->ffn_key_w), fk_buf, c, ffn_dim);

#ifdef __ARM_NEON
        {
            int fd4 = ffn_dim & ~3;
            float32x4_t zero = vdupq_n_f32(0.0f);
            for (int i = 0; i < fd4; i += 4) {
                float32x4_t v = vmaxq_f32(vld1q_f32(&fk_buf[i]), zero);
                vst1q_f32(&fk_buf[i], vmulq_f32(v, v));
            }
            for (int i = fd4; i < ffn_dim; i++) {
                float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
                fk_buf[i] = v * v;
            }
        }
#else
        for (int i = 0; i < ffn_dim; i++) {
            float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
            fk_buf[i] = v * v;
        }
#endif

        if (blk->ffn_value_q8)
            fast_matvec_q8(fk_buf, blk->ffn_value_q8, att_out,
                           blk->ffn_value_q8->rows, blk->ffn_value_q8->cols, q8_buf);
        else
            fast_matvec(fk_buf, tm_data_f32(blk->ffn_value_w), att_out, ffn_dim, c);

        fast_residual_add(x_out, att_out, c);
    }
}

/* ================================================================
 * Weight loading helpers
 * ================================================================ */

static float* bf16_to_f32(const void *raw, int64_t numel) {
    const uint16_t *src = (const uint16_t*)raw;
    float *dst = (float*)malloc(numel * sizeof(float));
    for (int64_t i = 0; i < numel; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, sizeof(float));
    }
    return dst;
}

static tm_tensor* load_weight(fs_file *f, const char *name) {
    if (!fs_has_tensor(f, name))
        return NULL;

    fs_tensor_info *info = fs_get_info(f, name);
    if (!info) return NULL;

    tm_tensor *t = NULL;

    if (info->dtype == FS_DTYPE_F32) {
        const float *data = fs_get_f32(f, name);
        if (!data) { fs_free_info(info); return NULL; }
        tm_tensor *t_view = tm_from_data((void*)data, info->shape, info->ndim, TM_F32);
        t = tm_clone(t_view);
        tm_free(t_view);
    } else if (info->dtype == FS_DTYPE_BF16) {
        const void *raw = fs_get_data(f, name);
        if (!raw) { fs_free_info(info); return NULL; }
        int64_t numel = 1;
        for (int d = 0; d < info->ndim; d++) numel *= info->shape[d];
        float *f32_data = bf16_to_f32(raw, numel);
        tm_tensor *t_view = tm_from_data(f32_data, info->shape, info->ndim, TM_F32);
        t = tm_clone(t_view);
        tm_free(t_view);
        free(f32_data);
    } else {
        fprintf(stderr, "rwkvx: unsupported dtype %s for weight %s\n",
                fs_dtype_name(info->dtype), name);
        fs_free_info(info);
        return NULL;
    }

    bool is_linear = strstr(name, "key.weight") || strstr(name, "value.weight") ||
                     strstr(name, "receptance.weight") || strstr(name, "output.weight") ||
                     strstr(name, "head.weight");
    if (is_linear && info->ndim == 2) {
        tm_tensor *tt = tm_t(t);
        tm_tensor *tc = tm_contiguous(tt);
        tm_free(t); tm_free(tt);
        t = tc;
    }

    for (int d = tm_ndim(t) - 1; d >= 0; d--) {
        if (tm_size(t, d) == 1 && tm_ndim(t) > 1) {
            tm_tensor *sq = tm_squeeze(t, d);
            tm_free(t);
            t = sq;
        }
    }

    if (strstr(name, "att.r_k") && tm_ndim(t) == 2) {
        int64_t flat_shape[] = {tm_numel(t)};
        tm_tensor *flat = tm_reshape(t, flat_shape, 1);
        tm_free(t);
        t = flat;
    }

    fs_free_info(info);
    return t;
}

static tm_tensor* require_weight(fs_file *f, const char *name) {
    tm_tensor *t = load_weight(f, name);
    if (!t) {
        fprintf(stderr, "rwkvx: missing required weight: %s\n", name);
    }
    return t;
}

static void pre_normalize_embeddings(tm_tensor **emb,
                                     tm_tensor *ln0_w, tm_tensor *ln0_b,
                                     int vocab_size, int n_embd)
{
    if (!ln0_w || !ln0_b) return;
    float *e = tm_data_f32(*emb);
    float *w = tm_data_f32(ln0_w);
    float *b = tm_data_f32(ln0_b);
    float eps = 1e-5f;

    for (int v = 0; v < vocab_size; v++) {
        float *row = &e[v * n_embd];
        float mean = 0.0f;
        for (int j = 0; j < n_embd; j++) mean += row[j];
        mean /= n_embd;
        float var = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= n_embd;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < n_embd; j++)
            row[j] = (row[j] - mean) * inv_std * w[j] + b[j];
    }
}

/* ================================================================
 * Internal model loading
 * ================================================================ */

static rx_model_internal* rx_model_load_internal(const char *path) {
#ifdef USE_CUDA
    fc_init();  /* Initialize CUDA if available (no-op if already init) */
#endif

    fs_file *f = fs_open(path);
    if (!f) {
        fprintf(stderr, "rwkvx: failed to open %s\n", path);
        return NULL;
    }

    /* Auto-detect weight prefix: RWKV-X uses "rwkv." prefix, pure RWKV-7 uses none */
    const char *prefix = "rwkv.";
    fs_tensor_info *emb_info = fs_get_info(f, "rwkv.emb.weight");
    if (!emb_info) {
        emb_info = fs_get_info(f, "emb.weight");
        if (!emb_info) {
            fprintf(stderr, "rwkvx: missing emb.weight (tried rwkv.emb.weight and emb.weight)\n");
            fs_close(f);
            return NULL;
        }
        prefix = "";
    }
    int vocab_size = (int)emb_info->shape[0];
    int n_embd = (int)emb_info->shape[1];
    int head_size = 64;
    int n_head = n_embd / head_size;
    fs_free_info(emb_info);

    int count;
    const char **names = fs_tensor_names(f, &count);
    int max_rwkv = -1, max_attn = -1;
    char rwkv_scan_fmt[64];
    snprintf(rwkv_scan_fmt, sizeof(rwkv_scan_fmt), "%sblocks.%%d.", prefix);
    for (int i = 0; i < count; i++) {
        int idx;
        if (sscanf(names[i], rwkv_scan_fmt, &idx) == 1) {
            if (idx > max_rwkv) max_rwkv = idx;
        } else if (sscanf(names[i], "moba.%d.", &idx) == 1) {
            if (idx > max_attn) max_attn = idx;
        }
    }
    int n_rwkv_layer = max_rwkv + 1;
    int n_attn_layer = max_attn >= 0 ? max_attn + 1 : 0;

    const char *model_type = n_attn_layer > 0 ? "RWKV-X hybrid" : "RWKV-7 pure";
    fprintf(stderr, "rwkvx: %s — n_embd=%d n_head=%d n_rwkv=%d n_attn=%d vocab=%d\n",
            model_type, n_embd, n_head, n_rwkv_layer, n_attn_layer, vocab_size);

    rx_model_internal *m = (rx_model_internal*)calloc(1, sizeof(rx_model_internal));
    m->config.n_embd = n_embd;
    m->config.n_head = n_head;
    m->config.head_size = head_size;
    m->config.n_rwkv_layer = n_rwkv_layer;
    m->config.n_attn_layer = n_attn_layer;
    m->config.vocab_size = vocab_size;

    char name_buf[256];
    snprintf(name_buf, sizeof(name_buf), "%semb.weight", prefix);
    m->emb_weight = require_weight(f, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sln_out.weight", prefix);
    m->ln_out_w = require_weight(f, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sln_out.bias", prefix);
    m->ln_out_b = require_weight(f, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%shead.weight", prefix);
    m->head_weight = require_weight(f, name_buf);

    if (!m->emb_weight || !m->ln_out_w || !m->ln_out_b || !m->head_weight) {
        /* Let pn_rwkvx_free handle cleanup */
        fs_close(f);
        return NULL;
    }

    snprintf(name_buf, sizeof(name_buf), "%sblocks.0.ln0.weight", prefix);
    tm_tensor *ln0_w = load_weight(f, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sblocks.0.ln0.bias", prefix);
    tm_tensor *ln0_b = load_weight(f, name_buf);
    pre_normalize_embeddings(&m->emb_weight, ln0_w, ln0_b, vocab_size, n_embd);
    if (ln0_w) tm_free(ln0_w);
    if (ln0_b) tm_free(ln0_b);

    /* Load RWKV-7 blocks */
    m->rwkv_blocks = (rx_rwkv_block*)calloc(n_rwkv_layer, sizeof(rx_rwkv_block));
    for (int i = 0; i < n_rwkv_layer; i++) {
        char buf[256], block_pfx[128];
        snprintf(block_pfx, sizeof(block_pfx), "%sblocks.%d", prefix, i);
        rx_rwkv_block *b = &m->rwkv_blocks[i];

        #define LOAD_RWKV(field, suffix) do { \
            snprintf(buf, sizeof(buf), "%s.%s", block_pfx, suffix); \
            b->field = require_weight(f, buf); \
        } while(0)

        LOAD_RWKV(ln1_w, "ln1.weight");
        LOAD_RWKV(ln1_b, "ln1.bias");
        LOAD_RWKV(ln2_w, "ln2.weight");
        LOAD_RWKV(ln2_b, "ln2.bias");
        LOAD_RWKV(x_r, "att.x_r");
        LOAD_RWKV(x_w, "att.x_w");
        LOAD_RWKV(x_k, "att.x_k");
        LOAD_RWKV(x_v, "att.x_v");
        LOAD_RWKV(x_a, "att.x_a");
        LOAD_RWKV(x_g, "att.x_g");
        LOAD_RWKV(w0, "att.w0");
        LOAD_RWKV(w1, "att.w1");
        LOAD_RWKV(w2, "att.w2");
        LOAD_RWKV(a0, "att.a0");
        LOAD_RWKV(a1, "att.a1");
        LOAD_RWKV(a2, "att.a2");
        LOAD_RWKV(g1, "att.g1");
        LOAD_RWKV(g2, "att.g2");
        LOAD_RWKV(k_k, "att.k_k");
        LOAD_RWKV(k_a, "att.k_a");
        LOAD_RWKV(r_k, "att.r_k");
        LOAD_RWKV(receptance_w, "att.receptance.weight");
        LOAD_RWKV(key_w, "att.key.weight");
        LOAD_RWKV(value_w, "att.value.weight");
        LOAD_RWKV(output_w, "att.output.weight");
        LOAD_RWKV(ln_x_w, "att.ln_x.weight");
        LOAD_RWKV(ln_x_b, "att.ln_x.bias");
        LOAD_RWKV(ffn_x_k, "ffn.x_k");
        LOAD_RWKV(ffn_key_w, "ffn.key.weight");
        LOAD_RWKV(ffn_value_w, "ffn.value.weight");
        if (i > 0) {
            LOAD_RWKV(v0, "att.v0");
            LOAD_RWKV(v1, "att.v1");
            LOAD_RWKV(v2, "att.v2");
        }
        #undef LOAD_RWKV
        fprintf(stderr, "rwkvx: loaded RWKV-7 block %d\n", i);
    }

    /* Load sparse attention blocks */
    m->attn_blocks = (rx_attn_block*)calloc(n_attn_layer > 0 ? n_attn_layer : 1,
                                            sizeof(rx_attn_block));
    for (int i = 0; i < n_attn_layer; i++) {
        char buf[256];
        rx_attn_block *b = &m->attn_blocks[i];

        #define LOAD_ATTN(field, fmt, ...) do { \
            snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
            b->field = require_weight(f, buf); \
        } while(0)

        LOAD_ATTN(ln1_w, "moba.%d.ln1.weight", i);
        LOAD_ATTN(ln1_b, "moba.%d.ln1.bias", i);
        LOAD_ATTN(ln2_w, "moba.%d.ln2.weight", i);
        LOAD_ATTN(ln2_b, "moba.%d.ln2.bias", i);
        LOAD_ATTN(receptance_w, "moba.%d.att.receptance.weight", i);
        LOAD_ATTN(key_w, "moba.%d.att.key.weight", i);
        LOAD_ATTN(value_w, "moba.%d.att.value.weight", i);
        LOAD_ATTN(output_w, "moba.%d.att.output.weight", i);
        LOAD_ATTN(ffn_x_k, "moba.%d.ffn.x_k", i);
        LOAD_ATTN(ffn_key_w, "moba.%d.ffn.key.weight", i);
        LOAD_ATTN(ffn_value_w, "moba.%d.ffn.value.weight", i);
        #undef LOAD_ATTN
        fprintf(stderr, "rwkvx: loaded sparse attention block %d\n", i);
    }

    /* Build block execution order */
    int n_blocks = n_rwkv_layer + n_attn_layer;
    m->n_blocks = n_blocks;
    m->block_order = (int*)calloc(n_blocks, sizeof(int));
    m->block_index = (int*)calloc(n_blocks, sizeof(int));

    if (n_attn_layer == 0) {
        for (int i = 0; i < n_rwkv_layer; i++) {
            m->block_order[i] = 0;
            m->block_index[i] = i;
        }
    } else {
        int interval = n_rwkv_layer / n_attn_layer;
        int pos = 0;
        for (int a = 0; a < n_attn_layer; a++) {
            int start = a * interval;
            int end = (a + 1) * interval;
            for (int j = start; j < end; j++) {
                m->block_order[pos] = 0;
                m->block_index[pos] = j;
                pos++;
            }
            m->block_order[pos] = 1;
            m->block_index[pos] = a;
            pos++;
        }
    }

#ifdef USE_THREADS
    m->pool = fw_pool_new(0);
    if (m->pool)
        fprintf(stderr, "rwkvx: pthread thread pool created\n");
#endif

    fs_close(f);
    fprintf(stderr, "rwkvx: model loaded successfully (%d blocks)\n", n_blocks);
    return m;
}

/* Quantize a 2D weight tensor to INT8 */
static rx_q8_weight* rx_quantize_weight(tm_tensor *t) {
    if (!t || tm_ndim(t) != 2) return NULL;
    int in_dim = (int)tm_size(t, 0);
    int out_dim = (int)tm_size(t, 1);
    const float *src = tm_data_f32(t);

    float *transposed = (float*)malloc(out_dim * in_dim * sizeof(float));
    for (int i = 0; i < in_dim; i++)
        for (int j = 0; j < out_dim; j++)
            transposed[j * in_dim + i] = src[i * out_dim + j];

    rx_q8_weight *q = rx_q8_from_f32(transposed, out_dim, in_dim);
    free(transposed);
    return q;
}

static void rx_model_quantize_weights(rx_model_internal *m) {
    if (!m || m->quantized) return;

    for (int i = 0; i < m->config.n_rwkv_layer; i++) {
        rx_rwkv_block *b = &m->rwkv_blocks[i];
        b->receptance_q8 = rx_quantize_weight(b->receptance_w);
        b->key_q8 = rx_quantize_weight(b->key_w);
        b->value_q8 = rx_quantize_weight(b->value_w);
        b->output_q8 = rx_quantize_weight(b->output_w);
        b->ffn_key_q8 = rx_quantize_weight(b->ffn_key_w);
        b->ffn_value_q8 = rx_quantize_weight(b->ffn_value_w);
        tm_free(b->receptance_w); b->receptance_w = NULL;
        tm_free(b->key_w); b->key_w = NULL;
        tm_free(b->value_w); b->value_w = NULL;
        tm_free(b->output_w); b->output_w = NULL;
        tm_free(b->ffn_key_w); b->ffn_key_w = NULL;
        tm_free(b->ffn_value_w); b->ffn_value_w = NULL;
    }

    for (int i = 0; i < m->config.n_attn_layer; i++) {
        rx_attn_block *b = &m->attn_blocks[i];
        b->receptance_q8 = rx_quantize_weight(b->receptance_w);
        b->key_q8 = rx_quantize_weight(b->key_w);
        b->value_q8 = rx_quantize_weight(b->value_w);
        b->output_q8 = rx_quantize_weight(b->output_w);
        b->ffn_key_q8 = rx_quantize_weight(b->ffn_key_w);
        b->ffn_value_q8 = rx_quantize_weight(b->ffn_value_w);
        tm_free(b->receptance_w); b->receptance_w = NULL;
        tm_free(b->key_w); b->key_w = NULL;
        tm_free(b->value_w); b->value_w = NULL;
        tm_free(b->output_w); b->output_w = NULL;
        tm_free(b->ffn_key_w); b->ffn_key_w = NULL;
        tm_free(b->ffn_value_w); b->ffn_value_w = NULL;
    }

    m->head_q8 = rx_quantize_weight(m->head_weight);
    tm_free(m->head_weight); m->head_weight = NULL;

    m->quantized = true;
    fprintf(stderr, "rwkvx: weights quantized to INT8 (per-row symmetric)\n");
}

/* Internal model free */
static void rx_model_free_internal(rx_model_internal *m) {
    if (!m) return;
    if (m->emb_weight) tm_free(m->emb_weight);
    if (m->ln_out_w) tm_free(m->ln_out_w);
    if (m->ln_out_b) tm_free(m->ln_out_b);
    if (m->head_weight) tm_free(m->head_weight);
    rx_q8_free(m->head_q8);

    for (int i = 0; i < m->config.n_rwkv_layer; i++) {
        rx_rwkv_block *b = &m->rwkv_blocks[i];
        if (b->ln1_w) tm_free(b->ln1_w); if (b->ln1_b) tm_free(b->ln1_b);
        if (b->ln2_w) tm_free(b->ln2_w); if (b->ln2_b) tm_free(b->ln2_b);
        if (b->x_r) tm_free(b->x_r); if (b->x_w) tm_free(b->x_w);
        if (b->x_k) tm_free(b->x_k); if (b->x_v) tm_free(b->x_v);
        if (b->x_a) tm_free(b->x_a); if (b->x_g) tm_free(b->x_g);
        if (b->w0) tm_free(b->w0); if (b->w1) tm_free(b->w1); if (b->w2) tm_free(b->w2);
        if (b->a0) tm_free(b->a0); if (b->a1) tm_free(b->a1); if (b->a2) tm_free(b->a2);
        if (b->v0) tm_free(b->v0); if (b->v1) tm_free(b->v1); if (b->v2) tm_free(b->v2);
        if (b->g1) tm_free(b->g1); if (b->g2) tm_free(b->g2);
        if (b->k_k) tm_free(b->k_k); if (b->k_a) tm_free(b->k_a); if (b->r_k) tm_free(b->r_k);
        if (b->receptance_w) tm_free(b->receptance_w); if (b->key_w) tm_free(b->key_w);
        if (b->value_w) tm_free(b->value_w); if (b->output_w) tm_free(b->output_w);
        if (b->ln_x_w) tm_free(b->ln_x_w); if (b->ln_x_b) tm_free(b->ln_x_b);
        if (b->ffn_x_k) tm_free(b->ffn_x_k);
        if (b->ffn_key_w) tm_free(b->ffn_key_w); if (b->ffn_value_w) tm_free(b->ffn_value_w);
        rx_q8_free(b->receptance_q8); rx_q8_free(b->key_q8);
        rx_q8_free(b->value_q8); rx_q8_free(b->output_q8);
        rx_q8_free(b->ffn_key_q8); rx_q8_free(b->ffn_value_q8);
    }
    free(m->rwkv_blocks);

    for (int i = 0; i < m->config.n_attn_layer; i++) {
        rx_attn_block *b = &m->attn_blocks[i];
        if (b->ln1_w) tm_free(b->ln1_w); if (b->ln1_b) tm_free(b->ln1_b);
        if (b->ln2_w) tm_free(b->ln2_w); if (b->ln2_b) tm_free(b->ln2_b);
        if (b->receptance_w) tm_free(b->receptance_w); if (b->key_w) tm_free(b->key_w);
        if (b->value_w) tm_free(b->value_w); if (b->output_w) tm_free(b->output_w);
        if (b->ffn_x_k) tm_free(b->ffn_x_k);
        if (b->ffn_key_w) tm_free(b->ffn_key_w); if (b->ffn_value_w) tm_free(b->ffn_value_w);
        rx_q8_free(b->receptance_q8); rx_q8_free(b->key_q8);
        rx_q8_free(b->value_q8); rx_q8_free(b->output_q8);
        rx_q8_free(b->ffn_key_q8); rx_q8_free(b->ffn_value_q8);
    }
    free(m->attn_blocks);

    free(m->block_order);
    free(m->block_index);

    /* Tokenizer is freed by pn_rwkvx_free() before calling this function */

#ifdef USE_THREADS
    if (m->pool) fw_pool_free(m->pool);
#endif

#ifdef USE_CUDA
    fc_free();
#endif

    free(m);
}

/* ================================================================
 * Internal: Forward pass
 * ================================================================ */

#define DEFAULT_MAX_CACHE_LEN 2048

static tm_tensor* rx_forward_internal(rx_model_internal *m, uint32_t token,
                                       rx_state_internal *s) {
    if (!m || !s) return NULL;
#ifdef USE_THREADS
    __rx_pool = m->pool;
#endif
    rx_config cfg = m->config;
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    rx_workspace *ws = s->workspace;

    float *emb_data = tm_data_f32(m->emb_weight);
    float *x_cur = ws->x_buf[0];
    float *x_next = ws->x_buf[1];
    memcpy(x_cur, &emb_data[token * c], c * sizeof(float));

    for (int pos = 0; pos < m->n_blocks; pos++) {
        int type = m->block_order[pos];
        int idx  = m->block_index[pos];

        if (type == 0) {
            rwkv7_block_forward_fast(
                &m->rwkv_blocks[idx], idx, x_cur,
                &s->rwkv[idx], s->v_first_buf,
                cfg.n_head, cfg.head_size, ws, x_next);
        } else {
            attn_block_forward_fast(
                &m->attn_blocks[idx], x_cur, &s->attn[idx],
                cfg.n_head, cfg.head_size, ws, x_next);
        }
        float *tmp = x_cur;
        x_cur = x_next;
        x_next = tmp;
    }

    fast_layer_norm(x_cur, tm_data_f32(m->ln_out_w), tm_data_f32(m->ln_out_b),
                    x_next, c);

    int64_t logits_shape[] = {vocab};
    tm_tensor *logits = tm_empty(logits_shape, 1, TM_F32);
    if (m->head_q8)
        fast_matvec_q8(x_next, m->head_q8, tm_data_f32(logits),
                       m->head_q8->rows, m->head_q8->cols, ws->q8_buf);
    else
        fast_matvec(x_next, tm_data_f32(m->head_weight),
                    tm_data_f32(logits), c, vocab);

    return logits;
}

/* ================================================================
 * Internal: State management
 * ================================================================ */

static rx_state_internal* rx_state_new_internal(const rx_model_internal *m) {
    if (!m) return NULL;
    rx_config cfg = m->config;
    int c = cfg.n_embd;

    rx_state_internal *s = (rx_state_internal*)calloc(1, sizeof(rx_state_internal));
    s->config = cfg;

    s->rwkv = (rx_rwkv_state*)calloc(cfg.n_rwkv_layer, sizeof(rx_rwkv_state));
    int64_t shape_c[] = {c};
    int state_size = cfg.n_head * cfg.head_size * cfg.head_size;
    int64_t shape_s[] = {state_size};
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        s->rwkv[i].att_x_prev = tm_zeros(shape_c, 1, TM_F32);
        s->rwkv[i].att_kv = tm_zeros(shape_s, 1, TM_F32);
        s->rwkv[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    s->attn = (rx_attn_state*)calloc(cfg.n_attn_layer > 0 ? cfg.n_attn_layer : 1,
                                     sizeof(rx_attn_state));
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        int64_t cache_shape[] = {DEFAULT_MAX_CACHE_LEN * c};
        s->attn[i].k_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].v_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].cache_len = 0;
        s->attn[i].max_cache_len = DEFAULT_MAX_CACHE_LEN;
        s->attn[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    s->v_first = tm_zeros(shape_c, 1, TM_F32);
    s->v_first_buf = tm_data_f32(s->v_first);

    int max_lora = 0, max_ffn = 0;
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        rx_rwkv_block *blk = &m->rwkv_blocks[i];
        int d;
        d = (int)tm_size(blk->w1, 1); if (d > max_lora) max_lora = d;
        d = (int)tm_size(blk->a1, 1); if (d > max_lora) max_lora = d;
        d = (int)tm_size(blk->g1, 1); if (d > max_lora) max_lora = d;
        if (blk->v1) { d = (int)tm_size(blk->v1, 1); if (d > max_lora) max_lora = d; }
        d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
        if (d > max_ffn) max_ffn = d;
    }
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        rx_attn_block *blk = &m->attn_blocks[i];
        int d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
        if (d > max_ffn) max_ffn = d;
    }
    s->workspace = rx_workspace_new(c, max_lora, max_ffn);

    return s;
}

static void rx_state_free_internal(rx_state_internal *s) {
    if (!s) return;
    for (int i = 0; i < s->config.n_rwkv_layer; i++) {
        tm_free(s->rwkv[i].att_x_prev);
        tm_free(s->rwkv[i].att_kv);
        tm_free(s->rwkv[i].ffn_x_prev);
    }
    free(s->rwkv);
    for (int i = 0; i < s->config.n_attn_layer; i++) {
        tm_free(s->attn[i].k_cache);
        tm_free(s->attn[i].v_cache);
        tm_free(s->attn[i].ffn_x_prev);
    }
    free(s->attn);
    tm_free(s->v_first);
    rx_workspace_free(s->workspace);
    free(s);
}

static void rx_state_reset_internal(rx_state_internal *s) {
    if (!s) return;
    rx_config cfg = s->config;
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        tm_fill_(s->rwkv[i].att_x_prev, 0.0f);
        tm_fill_(s->rwkv[i].att_kv, 0.0f);
        tm_fill_(s->rwkv[i].ffn_x_prev, 0.0f);
    }
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        tm_fill_(s->attn[i].k_cache, 0.0f);
        tm_fill_(s->attn[i].v_cache, 0.0f);
        s->attn[i].cache_len = 0;
        tm_fill_(s->attn[i].ffn_x_prev, 0.0f);
    }
    tm_fill_(s->v_first, 0.0f);
}

/* ================================================================
 * Tokenizer
 * ================================================================ */

static rx_trie_node* trie_node_new(void) {
    rx_trie_node *n = (rx_trie_node*)calloc(1, sizeof(rx_trie_node));
    n->token_id = -1;
    return n;
}

static void trie_node_free(rx_trie_node *n) {
    if (!n) return;
    for (int i = 0; i < 256; i++)
        trie_node_free(n->children[i]);
    free(n);
}

static void trie_insert(rx_trie_node *root, const uint8_t *key, int len, int32_t token_id) {
    rx_trie_node *node = root;
    for (int i = 0; i < len; i++) {
        int idx = key[i];
        if (!node->children[idx])
            node->children[idx] = trie_node_new();
        node = node->children[idx];
    }
    node->token_id = token_id;
}

static int parse_python_literal(const char *s, uint8_t *out, int max_out) {
    const char *p = s;
    bool is_bytes = false;
    int out_len = 0;

    while (*p == ' ' || *p == '\t') p++;
    if (*p == 'b') { is_bytes = true; p++; }

    char quote = *p;
    if (quote != '\'' && quote != '"') return -1;
    p++;

    const char *end = p;
    while (*end && *end != quote) {
        if (*end == '\\') end++;
        if (*end) end++;
    }
    if (*end != quote) return -1;

    while (p < end && out_len < max_out) {
        if (*p == '\\' && p + 1 < end) {
            switch (p[1]) {
                case 'x':
                    if (p + 3 < end) {
                        char hex[3] = {p[2], p[3], 0};
                        out[out_len++] = (uint8_t)strtol(hex, NULL, 16);
                        p += 4;
                    } else {
                        return -1;
                    }
                    break;
                case 'n': out[out_len++] = '\n'; p += 2; break;
                case 'r': out[out_len++] = '\r'; p += 2; break;
                case 't': out[out_len++] = '\t'; p += 2; break;
                case '\\': out[out_len++] = '\\'; p += 2; break;
                case '\'': out[out_len++] = '\''; p += 2; break;
                case '"': out[out_len++] = '"'; p += 2; break;
                case '0': out[out_len++] = 0; p += 2; break;
                default:
                    out[out_len++] = '\\';
                    p++;
                    break;
            }
        } else {
            if (is_bytes) {
                out[out_len++] = (uint8_t)*p;
            } else {
                out[out_len++] = (unsigned char)*p;
            }
            p++;
        }
    }
    return out_len;
}

static rx_tokenizer* rx_tokenizer_new(const char *vocab_path) {
    FILE *fp = fopen(vocab_path, "r");
    if (!fp) {
        fprintf(stderr, "rwkvx: failed to open vocab: %s\n", vocab_path);
        return NULL;
    }

    rx_tokenizer *tok = (rx_tokenizer*)calloc(1, sizeof(rx_tokenizer));
    tok->root = trie_node_new();

    const char *eot = "<|endoftext|>";
    int eot_len = (int)strlen(eot);
    tok->token_bytes[0] = (uint8_t*)malloc(eot_len);
    memcpy(tok->token_bytes[0], eot, eot_len);
    tok->token_lens[0] = eot_len;

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        int ll = (int)strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (ll == 0) continue;

        char *first_space = strchr(line, ' ');
        if (!first_space) continue;
        char *last_space = strrchr(line, ' ');
        if (last_space == first_space) continue;

        int idx = atoi(line);
        if (idx < 0 || idx >= RX_VOCAB_SIZE) continue;

        *last_space = 0;
        const char *literal = first_space + 1;

        uint8_t bytes[RX_MAX_TOKEN_LEN];
        int blen = parse_python_literal(literal, bytes, RX_MAX_TOKEN_LEN);
        if (blen < 0) continue;

        trie_insert(tok->root, bytes, blen, idx);
        tok->token_bytes[idx] = (uint8_t*)malloc(blen);
        memcpy(tok->token_bytes[idx], bytes, blen);
        tok->token_lens[idx] = blen;
    }

    fclose(fp);
    return tok;
}

static void rx_tokenizer_free(rx_tokenizer *tok) {
    if (!tok) return;
    trie_node_free(tok->root);
    for (int i = 0; i < RX_VOCAB_SIZE; i++)
        free(tok->token_bytes[i]);
    free(tok);
}

static int rx_tokenizer_encode(const rx_tokenizer *tok, const char *text,
                                uint32_t *tokens_out, int max_tokens) {
    if (!tok || !text) return 0;
    const uint8_t *src = (const uint8_t*)text;
    int src_len = (int)strlen(text);
    int n = 0, i = 0;

    while (i < src_len && n < max_tokens) {
        rx_trie_node *node = tok->root;
        int last_match_pos = i;
        int32_t last_match_id = -1;
        int j = i;

        while (j < src_len) {
            int byte = src[j];
            if (!node->children[byte]) break;
            node = node->children[byte];
            j++;
            if (node->token_id >= 0) {
                last_match_pos = j;
                last_match_id = node->token_id;
            }
        }

        if (last_match_id >= 0) {
            tokens_out[n++] = (uint32_t)last_match_id;
            i = last_match_pos;
        } else {
            tokens_out[n++] = (uint32_t)(src[i] + 1);
            i++;
        }
    }
    return n;
}

static char* rx_tokenizer_decode(const rx_tokenizer *tok, const uint32_t *tokens,
                                  int n_tokens) {
    if (!tok || !tokens) return NULL;

    int total = 0;
    for (int i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid < RX_VOCAB_SIZE)
            total += tok->token_lens[tid];
    }

    char *out = (char*)malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid < RX_VOCAB_SIZE && tok->token_bytes[tid]) {
            memcpy(&out[pos], tok->token_bytes[tid], tok->token_lens[tid]);
            pos += tok->token_lens[tid];
        }
    }
    out[pos] = 0;
    return out;
}

/* ================================================================
 * Pantheon adapter layer
 * ================================================================ */

void* pn_rwkvx_load(const char *safetensors_path, pn_config *config, bool quantize) {
    rx_model_internal *m = rx_model_load_internal(safetensors_path);
    if (!m) return NULL;

    if (quantize)
        rx_model_quantize_weights(m);

    /* Fill Pantheon config */
    config->arch = PN_ARCH_RWKVX;
    config->n_vocab = m->config.vocab_size;
    config->n_embd = m->config.n_embd;
    config->n_head = m->config.n_head;
    config->n_kv_head = m->config.n_head;
    config->n_layer = m->config.n_rwkv_layer + m->config.n_attn_layer;
    config->n_ff = 0;
    config->n_ctx = DEFAULT_MAX_CACHE_LEN;
    config->rope_theta = 0.0f;
    config->rms_norm_eps = 1e-5f;
    config->sliding_window = 0;
    config->rope_dim = 0;
    config->n_experts = 0;
    config->n_experts_used = 0;

    return m;
}

void pn_rwkvx_free(void *weights) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m) return;

    /* Free tokenizer if loaded */
    if (m->tokenizer)
        rx_tokenizer_free((rx_tokenizer*)m->tokenizer);
    m->tokenizer = NULL;

    rx_model_free_internal(m);
}

tm_tensor* pn_rwkvx_forward(pn_model *model, uint32_t token, pn_state *state) {
    rx_model_internal *m = (rx_model_internal*)model->weights;
    rx_state_internal *s = (rx_state_internal*)state->kv_cache;
    return rx_forward_internal(m, token, s);
}

void* pn_rwkvx_state_new(void *weights) {
    rx_model_internal *m = (rx_model_internal*)weights;
    return rx_state_new_internal(m);
}

void pn_rwkvx_state_free(void *state) {
    rx_state_free_internal((rx_state_internal*)state);
}

void pn_rwkvx_state_reset(void *state) {
    rx_state_reset_internal((rx_state_internal*)state);
}

int pn_rwkvx_encode(void *weights, const char *text, uint32_t *out, int max) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m || !m->tokenizer) return 0;
    return rx_tokenizer_encode((rx_tokenizer*)m->tokenizer, text, out, max);
}

char* pn_rwkvx_decode(void *weights, const uint32_t *tokens, int n) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m || !m->tokenizer) return NULL;
    return rx_tokenizer_decode((rx_tokenizer*)m->tokenizer, tokens, n);
}

int pn_rwkvx_eos_id(void *weights) {
    (void)weights;
    return 0;  /* <|endoftext|> */
}

int pn_rwkvx_vocab_size(void *weights) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m) return 0;
    return m->config.vocab_size;
}

int pn_rwkvx_n_attn_layers(void *weights) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m) return 0;
    return m->config.n_attn_layer;
}

int pn_rwkvx_load_tokenizer(void *weights, const char *vocab_path) {
    rx_model_internal *m = (rx_model_internal*)weights;
    if (!m || !vocab_path) return -1;
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) return -1;
    m->tokenizer = tok;
    return 0;
}
