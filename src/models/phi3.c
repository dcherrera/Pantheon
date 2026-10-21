/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "phi3.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "../memory/arena.h"
#include "../accel/simd.h"
#include "../formats/gguf_dequant.h"
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Config reading from GGUF (self-contained, uses "phi3" prefix)
 * ============================================================ */

static int phi3_read_config(pn_gguf_file *gguf, pn_config *config) {
    const char *prefix = "phi3";
    char key[128];

    snprintf(key, sizeof(key), "%s.vocab_size", prefix);
    config->n_vocab  = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.embedding_length", prefix);
    config->n_embd   = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.attention.head_count", prefix);
    config->n_head   = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.attention.head_count_kv", prefix);
    config->n_kv_head = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.block_count", prefix);
    config->n_layer  = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.feed_forward_length", prefix);
    config->n_ff     = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.context_length", prefix);
    config->n_ctx    = (int)pn_gguf_get_u32(gguf, key, 2048);
    snprintf(key, sizeof(key), "%s.rope.freq_base", prefix);
    config->rope_theta = pn_gguf_get_f32(gguf, key, 10000.0f);
    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", prefix);
    config->rms_norm_eps = pn_gguf_get_f32(gguf, key, 1e-5f);
    snprintf(key, sizeof(key), "%s.attention.sliding_window", prefix);
    config->sliding_window = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.rope.dimension_count", prefix);
    config->rope_dim = (int)pn_gguf_get_u32(gguf, key, 0);

    if (config->n_vocab == 0) {
        pn_gguf_tensor_info *embd = pn_gguf_find_tensor(gguf, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            config->n_vocab = (int)embd->dims[embd->n_dims - 1];
        }
    }

    if (config->n_embd == 0 || config->n_head == 0 || config->n_layer == 0) {
        fprintf(stderr, "phi3: invalid config (n_embd=%d, n_head=%d, n_layer=%d)\n",
                config->n_embd, config->n_head, config->n_layer);
        return -1;
    }

    if (config->n_kv_head == 0) config->n_kv_head = config->n_head;

    fprintf(stderr, "phi3: n_vocab=%d n_embd=%d n_head=%d n_kv_head=%d n_layer=%d n_ff=%d rope_dim=%d\n",
           config->n_vocab, config->n_embd, config->n_head,
           config->n_kv_head, config->n_layer, config->n_ff, config->rope_dim);
    return 0;
}

/* ============================================================
 * Helper: dequantize a GGUF tensor to F32 float array (owned)
 * ============================================================ */

static float* phi3_dequant_to_f32(pn_gguf_file *gguf, const char *name, size_t *numel_out) {
    return pn_gguf_dequantize_tensor(gguf, name, numel_out);
}

/* ============================================================
 * Check if a quant type is supported for direct qmv
 * ============================================================ */

static int phi3_is_supported_quant(pn_gguf_quant type) {
    switch (type) {
        case GGUF_QUANT_F32:
        case GGUF_QUANT_F16:
        case GGUF_QUANT_Q4_0:
        case GGUF_QUANT_Q8_0:
        case GGUF_QUANT_Q4_K:
        case GGUF_QUANT_Q5_K:
        case GGUF_QUANT_Q6_K:
            return 1;
        default:
            return 0;
    }
}

/* ============================================================
 * Quantized weight loading from GGUF
 * ============================================================ */

pn_phi3_qweights* pn_phi3_load_gguf_quantized(pn_gguf_file *gguf, pn_config *config) {
    if (!gguf || !config) return NULL;

    if (phi3_read_config(gguf, config) < 0) return NULL;

    /* Check that the first layer's fused QKV weight type is supported */
    pn_gguf_tensor_info *test_info = pn_gguf_find_tensor(gguf, "blk.0.attn_qkv.weight");
    if (!test_info || !phi3_is_supported_quant(test_info->type)) {
        fprintf(stderr, "phi3: quant type %d not supported for quantized path\n",
                test_info ? test_info->type : -1);
        return NULL;
    }

    pn_phi3_qweights *qw = calloc(1, sizeof(pn_phi3_qweights));
    if (!qw) return NULL;
    qw->is_quantized = 1;
    qw->n_layers = config->n_layer;
    qw->n_embd = config->n_embd;
    qw->n_vocab = config->n_vocab;
    qw->gguf = gguf;

    /* Token embeddings: always dequant to F32 */
    size_t embd_numel = 0;
    qw->tok_embd = phi3_dequant_to_f32(gguf, "token_embd.weight", &embd_numel);
    if (!qw->tok_embd) {
        fprintf(stderr, "phi3: failed to load token_embd.weight\n");
        pn_phi3_qweights_free(qw);
        return NULL;
    }

    /* Detect actual vocab size from embedding */
    if (config->n_vocab == 0 && embd_numel > 0 && config->n_embd > 0) {
        config->n_vocab = (int)(embd_numel / (size_t)config->n_embd);
        qw->n_vocab = config->n_vocab;
    }

    /* Output norm: always F32 */
    size_t norm_numel = 0;
    qw->output_norm = phi3_dequant_to_f32(gguf, "output_norm.weight", &norm_numel);
    if (!qw->output_norm) {
        fprintf(stderr, "phi3: failed to load output_norm.weight\n");
        pn_phi3_qweights_free(qw);
        return NULL;
    }

    /* Output head: try as qweight, fall back to tied embeddings */
    pn_gguf_tensor_info *out_info = pn_gguf_find_tensor(gguf, "output.weight");
    if (out_info && phi3_is_supported_quant(out_info->type)) {
        if (pn_qweight_from_gguf(&qw->output, gguf, "output.weight") < 0) {
            pn_phi3_qweights_free(qw);
            return NULL;
        }
    } else {
        /* Tied embeddings: use tok_embd as F32 output head */
        qw->output_f32 = malloc(embd_numel * sizeof(float));
        if (!qw->output_f32) {
            pn_phi3_qweights_free(qw);
            return NULL;
        }
        memcpy(qw->output_f32, qw->tok_embd, embd_numel * sizeof(float));
        qw->output.data = qw->output_f32;
        qw->output.type = GGUF_QUANT_F32;
        qw->output.rows = config->n_vocab;
        qw->output.cols = config->n_embd;
        qw->output.row_stride = (size_t)config->n_embd * sizeof(float);
    }

    /* Per-layer weights */
    qw->layers = calloc(config->n_layer, sizeof(pn_phi3_qlayer));
    if (!qw->layers) {
        pn_phi3_qweights_free(qw);
        return NULL;
    }

    for (int i = 0; i < config->n_layer; i++) {
        char name[128];
        pn_phi3_qlayer *l = &qw->layers[i];

        /* Norms: always dequant to F32 */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        l->attn_norm = phi3_dequant_to_f32(gguf, name, NULL);
        if (!l->attn_norm) {
            fprintf(stderr, "phi3: failed to load %s\n", name);
            pn_phi3_qweights_free(qw);
            return NULL;
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        l->ffn_norm = phi3_dequant_to_f32(gguf, name, NULL);
        if (!l->ffn_norm) {
            fprintf(stderr, "phi3: failed to load %s\n", name);
            pn_phi3_qweights_free(qw);
            return NULL;
        }

        /* Projection weights: keep quantized in mmap */
        #define LOAD_QW(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, i); \
            if (pn_qweight_from_gguf(&l->field, gguf, name) < 0) { \
                fprintf(stderr, "phi3: failed to load %s\n", name); \
                pn_phi3_qweights_free(qw); \
                return NULL; \
            } \
        } while(0)

        LOAD_QW(attn_qkv,     "blk.%d.attn_qkv.weight");
        LOAD_QW(attn_output,   "blk.%d.attn_output.weight");
        LOAD_QW(ffn_gate_up,   "blk.%d.ffn_up.weight");
        LOAD_QW(ffn_down,      "blk.%d.ffn_down.weight");

        #undef LOAD_QW

        fprintf(stderr, "\r  loaded layer %d/%d", i + 1, config->n_layer);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    return qw;
}

/* ============================================================
 * Free
 * ============================================================ */

void pn_phi3_qweights_free(pn_phi3_qweights *qw) {
    if (!qw) return;
    free(qw->tok_embd);
    free(qw->output_norm);
    free(qw->output_f32);

    if (qw->layers) {
        for (int i = 0; i < qw->n_layers; i++) {
            free(qw->layers[i].attn_norm);
            free(qw->layers[i].ffn_norm);
        }
        free(qw->layers);
    }
    free(qw);
}

/* ============================================================
 * RoPE (Rotary Position Embeddings) — NeoX style, in-place
 *
 * NeoX-style RoPE pairs elements offset by half the rotation
 * dimension: pair i = (head[i], head[i + rot/2]).
 * This matches HuggingFace's rotate_half() convention used by
 * Phi-3 (and Qwen2, Falcon, etc.), where the GGUF weights are
 * stored WITHOUT the interleaving permutation.
 * ============================================================ */

static void phi3_apply_rope(float *data, int n_heads, int head_dim,
                            int rope_dim, int pos, float theta_base) {
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[i];
            float x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* ============================================================
 * Attention head computation — for threading
 * ============================================================ */

typedef struct {
    const float *q_data;
    const float *k_cache_data;
    const float *v_cache_data;
    float *attn_out;
    float *scores_buf;
    int head_dim;
    int n_kv_groups;
    int max_seq;
    int seq_len;
    float scale;
} phi3_attn_args;

static void phi3_compute_attn_heads(int start, int end, void *arg) {
    phi3_attn_args *a = (phi3_attn_args *)arg;

    for (int qh = start; qh < end; qh++) {
        int kv_h = qh / a->n_kv_groups;
        const float *q_vec = a->q_data + qh * a->head_dim;
        const float *k_base = a->k_cache_data + kv_h * a->max_seq * a->head_dim;
        const float *v_base = a->v_cache_data + kv_h * a->max_seq * a->head_dim;
        float *scores = a->scores_buf + qh * a->max_seq;
        float *out_vec = a->attn_out + qh * a->head_dim;

        for (int s = 0; s < a->seq_len; s++) {
            scores[s] = pn_dot_f32(q_vec, k_base + s * a->head_dim, a->head_dim) * a->scale;
        }

        float max_score = scores[0];
        for (int s = 1; s < a->seq_len; s++) {
            if (scores[s] > max_score) max_score = scores[s];
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < a->seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int s = 0; s < a->seq_len; s++) {
            scores[s] *= inv_sum;
        }

        memset(out_vec, 0, a->head_dim * sizeof(float));
        for (int s = 0; s < a->seq_len; s++) {
            float w_s = scores[s];
            const float *v_vec = v_base + s * a->head_dim;
            for (int d = 0; d < a->head_dim; d++) {
                out_vec[d] += w_s * v_vec[d];
            }
        }
    }
}

/* ============================================================
 * Forward pass — quantized weights + arena
 *
 * Key differences from Llama:
 *   1. Single fused QKV matvec, then pointer split
 *   2. Single fused gate+up matvec, then pointer split
 * ============================================================ */

tm_tensor* pn_phi3_forward(pn_model *model, uint32_t token, pn_state *state) {
    if (!model || !state || !model->weights) return NULL;

    pn_phi3_qweights *qw = (pn_phi3_qweights *)model->weights;
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    pn_arena *arena = state->arena;
    int pos = state->pos;
    int n_embd = cfg->n_embd;
    int head_dim = n_embd / cfg->n_head;
    int n_kv_groups = cfg->n_head / cfg->n_kv_head;
    int max_seq = kv->max_seq;

#ifdef USE_THREADS
    fw_pool *pool = model->pool;
#endif

    if (!arena) return NULL;

    pn_arena_reset(arena);

    /* 1. Token embedding lookup */
    float *hidden = pn_arena_alloc_f32(arena, n_embd);
    if (!hidden) return NULL;
    memcpy(hidden, qw->tok_embd + (size_t)token * n_embd, n_embd * sizeof(float));

    /* Pre-allocate attention scores buffer: [n_head, max_seq] */
    float *scores_buf = pn_arena_alloc_f32(arena, cfg->n_head * max_seq);
    if (!scores_buf) return NULL;

    /* 2. Transformer layers */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_phi3_qlayer *l = &qw->layers[layer];

        /* === Attention sublayer === */

        /* RMSNorm */
        float *normed = pn_arena_alloc_f32(arena, n_embd);
        if (!normed) return NULL;
        pn_vrms_norm_f32(hidden, l->attn_norm, normed, n_embd, cfg->rms_norm_eps);

        /* Fused QKV projection: single matvec, then pointer split */
        float *qkv = pn_arena_alloc_f32(arena, l->attn_qkv.rows);  /* 3*n_embd */
        if (!qkv) return NULL;
        pn_qmv(&l->attn_qkv, normed, qkv);

        float *q = qkv;                      /* [0 : n_embd] */
        float *k = qkv + n_embd;             /* [n_embd : 2*n_embd] */
        float *v = qkv + 2 * n_embd;         /* [2*n_embd : 3*n_embd] */

        /* Apply RoPE to Q and K */
        phi3_apply_rope(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        phi3_apply_rope(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);

        /* Update KV cache */
        float *k_cache_data = (float *)kv->k_cache[layer]->data;
        float *v_cache_data = (float *)kv->v_cache[layer]->data;
        int cache_pos = pos % max_seq;

        for (int h = 0; h < cfg->n_kv_head; h++) {
            memcpy(k_cache_data + h * max_seq * head_dim + cache_pos * head_dim,
                   k + h * head_dim,
                   head_dim * sizeof(float));
            memcpy(v_cache_data + h * max_seq * head_dim + cache_pos * head_dim,
                   v + h * head_dim,
                   head_dim * sizeof(float));
        }

        /* Compute attention */
        int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
        float *attn_out = pn_arena_alloc_f32(arena, cfg->n_head * head_dim);
        if (!attn_out) return NULL;

        phi3_attn_args args = {
            .q_data = q,
            .k_cache_data = k_cache_data,
            .v_cache_data = v_cache_data,
            .attn_out = attn_out,
            .scores_buf = scores_buf,
            .head_dim = head_dim,
            .n_kv_groups = n_kv_groups,
            .max_seq = max_seq,
            .seq_len = seq_len,
            .scale = 1.0f / sqrtf((float)head_dim),
        };

#ifdef USE_THREADS
        if (pool && cfg->n_head >= 4) {
            fw_for(pool, cfg->n_head, phi3_compute_attn_heads, &args);
        } else {
            phi3_compute_attn_heads(0, cfg->n_head, &args);
        }
#else
        phi3_compute_attn_heads(0, cfg->n_head, &args);
#endif

        /* Output projection */
        float *attn_proj = pn_arena_alloc_f32(arena, n_embd);
        if (!attn_proj) return NULL;
        pn_qmv(&l->attn_output, attn_out, attn_proj);

        /* Residual connection */
        pn_vadd_f32(hidden, attn_proj, hidden, n_embd);

        /* === FFN sublayer === */

        /* RMSNorm */
        float *ffn_normed = pn_arena_alloc_f32(arena, n_embd);
        if (!ffn_normed) return NULL;
        pn_vrms_norm_f32(hidden, l->ffn_norm, ffn_normed, n_embd, cfg->rms_norm_eps);

        /* Fused gate+up projection: single matvec, then pointer split */
        float *gate_up = pn_arena_alloc_f32(arena, l->ffn_gate_up.rows);  /* 2*n_ff */
        if (!gate_up) return NULL;
        pn_qmv(&l->ffn_gate_up, ffn_normed, gate_up);

        float *gate = gate_up;               /* [0 : n_ff] */
        float *up = gate_up + cfg->n_ff;     /* [n_ff : 2*n_ff] */

        /* SiLU(gate) * up */
        float *ffn_hidden = pn_arena_alloc_f32(arena, cfg->n_ff);
        if (!ffn_hidden) return NULL;
        pn_vsilu_mul_f32(gate, up, ffn_hidden, cfg->n_ff);

        /* Down projection */
        float *ffn_out = pn_arena_alloc_f32(arena, n_embd);
        if (!ffn_out) return NULL;
        pn_qmv(&l->ffn_down, ffn_hidden, ffn_out);

        /* Residual */
        pn_vadd_f32(hidden, ffn_out, hidden, n_embd);
    }

    /* 3. Final RMSNorm */
    float *final_normed = pn_arena_alloc_f32(arena, n_embd);
    if (!final_normed) return NULL;
    pn_vrms_norm_f32(hidden, qw->output_norm, final_normed, n_embd, cfg->rms_norm_eps);

    /* 4. LM head: output @ normed -> logits */
    float *logits_raw = pn_arena_alloc_f32(arena, cfg->n_vocab);
    if (!logits_raw) return NULL;
    pn_qmv(&qw->output, final_normed, logits_raw);

    /* Create output tm_tensor (only allocation that escapes the arena) */
    int64_t logits_shape[] = { cfg->n_vocab };
    tm_tensor *logits_t = tm_from_data(logits_raw, logits_shape, 1, TM_F32);
    tm_tensor *logits = logits_t ? tm_clone(logits_t) : NULL;
    if (logits_t) tm_free(logits_t);

    state->pos++;
    return logits;
}
