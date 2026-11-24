/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "llama.h"
#include "llama_attention_trace.h"
#include "llama_residual_trace.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "../memory/arena.h"
#include "../accel/simd.h"
#include "../formats/gguf_dequant.h"
#include "../core/mem/pn_host_admission.h"
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif
#ifdef USE_METAL
  #include "metal/fm_metal.h"
#endif

/* ============================================================
 * Config reading from GGUF (shared by both loaders)
 * ============================================================ */

static int read_config(pn_gguf_file *gguf, pn_config *config, const char *prefix) {
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

    /* Qwen3 decouples head_dim from n_embd/n_head (stored as attention.key_length). */
    snprintf(key, sizeof(key), "%s.attention.key_length", prefix);
    config->head_dim = (int)pn_gguf_get_u32(gguf, key, 0);

    snprintf(key, sizeof(key), "%s.expert_count", prefix);
    config->n_experts = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.expert_used_count", prefix);
    config->n_experts_used = (int)pn_gguf_get_u32(gguf, key, 0);

    if (config->n_vocab == 0) {
        pn_gguf_tensor_info *embd = pn_gguf_find_tensor(gguf, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            config->n_vocab = (int)embd->dims[embd->n_dims - 1];
        }
    }

    if (config->n_embd == 0 || config->n_head == 0 || config->n_layer == 0) {
        fprintf(stderr, "%s: invalid config (n_embd=%d, n_head=%d, n_layer=%d)\n",
                prefix, config->n_embd, config->n_head, config->n_layer);
        return -1;
    }

    if (config->n_kv_head == 0) config->n_kv_head = config->n_head;

    /* Qwen3 adds per-head RMSNorm on Q and K (detected by the norm tensors). */
    config->qk_norm = pn_gguf_find_tensor(gguf, "blk.0.attn_q_norm.weight") != NULL;

    fprintf(stderr, "%s: n_vocab=%d n_embd=%d n_head=%d n_kv_head=%d head_dim=%d qk_norm=%d rope_dim=%d theta=%.0f n_layer=%d n_ff=%d sw=%d",
           prefix, config->n_vocab, config->n_embd, config->n_head,
           config->n_kv_head, pn_head_dim(config), config->qk_norm, config->rope_dim,
           config->rope_theta, config->n_layer, config->n_ff, config->sliding_window);
    if (config->n_experts > 0)
        fprintf(stderr, " n_experts=%d n_experts_used=%d", config->n_experts, config->n_experts_used);
    fprintf(stderr, "\n");
    return 0;
}

/* ============================================================
 * Helper: dequantize a GGUF tensor to F32 float array (owned)
 * ============================================================ */

static float* dequant_to_f32(pn_gguf_file *gguf, const char *name, size_t *numel_out) {
    return pn_gguf_dequantize_tensor(gguf, name, numel_out);
}

/* ============================================================
 * Helper: dequantize a GGUF tensor into a Foundry tm_tensor
 * (legacy, used by F32 loader)
 * ============================================================ */

static tm_tensor* gguf_to_tensor(pn_gguf_file *gguf, const char *name) {
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(gguf, name);
    if (!info) {
        fprintf(stderr, "llama: tensor '%s' not found in GGUF\n", name);
        return NULL;
    }

    int64_t shape[4];
    for (uint32_t d = 0; d < info->n_dims; d++) {
        shape[d] = (int64_t)info->dims[info->n_dims - 1 - d];
    }

    if (info->type == GGUF_QUANT_F32) {
        void *raw = (char *)gguf->data + info->offset;
        tm_tensor *t = tm_from_data(raw, shape, (int)info->n_dims, TM_F32);
        if (!t) return NULL;
        tm_tensor *owned = tm_clone(t);
        tm_free(t);
        return owned;
    }

    size_t out_numel = 0;
    float *data = pn_gguf_dequantize_tensor(gguf, name, &out_numel);
    if (!data) return NULL;

    tm_tensor *t = tm_from_data(data, shape, (int)info->n_dims, TM_F32);
    if (!t) {
        free(data);
        return NULL;
    }
    tm_tensor *owned = tm_clone(t);
    tm_free(t);
    free(data);
    return owned;
}

/* ============================================================
 * Quantized weight loading from GGUF (Phase 2)
 * ============================================================ */

/* Check if a quant type is supported for direct qmv */
static int is_supported_quant(pn_gguf_quant type) {
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

static const char* arch_prefix(pn_arch arch) {
    switch (arch) {
        case PN_ARCH_QWEN2:   return "qwen2";
        case PN_ARCH_QWEN3:   return "qwen3";
        case PN_ARCH_PHI:     return "phi3";
        case PN_ARCH_MIXTRAL: return "llama";
        default:              return "llama";
    }
}

pn_llama_qweights* pn_llama_load_gguf_quantized(pn_gguf_file *gguf, pn_config *config) {
    if (!gguf || !config) return NULL;

    if (read_config(gguf, config, arch_prefix(config->arch)) < 0) return NULL;

    /* Check that the first layer's weight type is supported */
    pn_gguf_tensor_info *test_info = pn_gguf_find_tensor(gguf, "blk.0.attn_q.weight");
    if (!test_info || !is_supported_quant(test_info->type)) {
        fprintf(stderr, "llama: quant type %d not supported for quantized path\n",
                test_info ? test_info->type : -1);
        return NULL;
    }

    pn_llama_qweights *qw = calloc(1, sizeof(pn_llama_qweights));
    if (!qw) return NULL;
    qw->is_quantized = 1;
    qw->n_layers = config->n_layer;
    qw->n_embd = config->n_embd;
    qw->n_vocab = config->n_vocab;
    qw->gguf = gguf;

    /* Token embeddings: always dequant to F32 (needed for embedding lookup) */
    size_t embd_numel = 0;
    qw->tok_embd = dequant_to_f32(gguf, "token_embd.weight", &embd_numel);
    if (!qw->tok_embd) {
        fprintf(stderr, "llama: failed to load token_embd.weight\n");
        pn_llama_qweights_free(qw);
        return NULL;
    }

    /* Detect actual vocab size from embedding */
    if (config->n_vocab == 0 && embd_numel > 0 && config->n_embd > 0) {
        config->n_vocab = (int)(embd_numel / (size_t)config->n_embd);
        qw->n_vocab = config->n_vocab;
    }

    /* Output norm: always F32 */
    size_t norm_numel = 0;
    qw->output_norm = dequant_to_f32(gguf, "output_norm.weight", &norm_numel);
    if (!qw->output_norm) {
        fprintf(stderr, "llama: failed to load output_norm.weight\n");
        pn_llama_qweights_free(qw);
        return NULL;
    }

    /* Output head: try as qweight (quantized mmap), fall back to tied embeddings */
    pn_gguf_tensor_info *out_info = pn_gguf_find_tensor(gguf, "output.weight");
    if (out_info && is_supported_quant(out_info->type)) {
        if (pn_qweight_from_gguf(&qw->output, gguf, "output.weight") < 0) {
            pn_llama_qweights_free(qw);
            return NULL;
        }
    } else {
        /* Tied embeddings: reuse the quantized token_embd tensor directly as the
         * LM head instead of materializing a ~622 MB F32 copy. Its GGUF dims
         * reverse to rows=n_vocab, cols=n_embd (matching the manual setup this
         * replaces), and pn_qweight_from_gguf points qw->output at the mmapped
         * Q6_K blocks. The decode paths take the quantized branch: single-stream
         * pn_qmv(&qw->output, ...) and batched pn_qmm(&qw->output, ...) (the F32
         * GEMM head is only taken when qw->output.type == GGUF_QUANT_F32).
         * qw->output_f32 stays NULL, so no F32 head buffer is allocated. */
        if (pn_qweight_from_gguf(&qw->output, gguf, "token_embd.weight") < 0) {
            pn_llama_qweights_free(qw);
            return NULL;
        }
    }

    /* Per-layer weights */
    qw->layers = calloc(config->n_layer, sizeof(pn_llama_qlayer));
    if (!qw->layers) {
        pn_llama_qweights_free(qw);
        return NULL;
    }

    for (int i = 0; i < config->n_layer; i++) {
        char name[128];
        pn_llama_qlayer *l = &qw->layers[i];

        /* Norms: always dequant to F32 */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        l->attn_norm = dequant_to_f32(gguf, name, NULL);
        if (!l->attn_norm) {
            fprintf(stderr, "llama: failed to load %s\n", name);
            pn_llama_qweights_free(qw);
            return NULL;
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        l->ffn_norm = dequant_to_f32(gguf, name, NULL);
        if (!l->ffn_norm) {
            fprintf(stderr, "llama: failed to load %s\n", name);
            pn_llama_qweights_free(qw);
            return NULL;
        }

        /* Projection weights: keep quantized in mmap */
        #define LOAD_QW(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, i); \
            if (pn_qweight_from_gguf(&l->field, gguf, name) < 0) { \
                fprintf(stderr, "llama: failed to load %s\n", name); \
                pn_llama_qweights_free(qw); \
                return NULL; \
            } \
        } while(0)

        /* Attention weights (shared by all architectures) */
        LOAD_QW(wq,     "blk.%d.attn_q.weight");
        LOAD_QW(wk,     "blk.%d.attn_k.weight");
        LOAD_QW(wv,     "blk.%d.attn_v.weight");
        LOAD_QW(wo,     "blk.%d.attn_output.weight");

        if (config->n_experts > 0) {
            /* MoE: load router + expert weights */
            LOAD_QW(ffn_gate_inp, "blk.%d.ffn_gate_inp.weight");

            int n_exp = config->n_experts;
            l->experts = calloc(n_exp, sizeof(pn_moe_expert));
            if (!l->experts) {
                pn_llama_qweights_free(qw);
                return NULL;
            }

            /* Try merged format first: blk.N.ffn_gate_exps.weight */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", i);
            pn_gguf_tensor_info *gate_exps_info = pn_gguf_find_tensor(gguf, name);

            if (gate_exps_info) {
                /* Merged format: 3D tensors [n_experts, n_ff/n_embd, cols] */
                pn_gguf_tensor_info *up_exps_info, *down_exps_info;
                snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", i);
                up_exps_info = pn_gguf_find_tensor(gguf, name);
                snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", i);
                down_exps_info = pn_gguf_find_tensor(gguf, name);

                if (!up_exps_info || !down_exps_info) {
                    fprintf(stderr, "llama: layer %d missing merged expert tensors\n", i);
                    pn_llama_qweights_free(qw);
                    return NULL;
                }

                /* Compute row strides for slicing */
                size_t gate_bpb = pn_gguf_bytes_per_block(gate_exps_info->type);
                int gate_bs = pn_gguf_block_size(gate_exps_info->type);
                size_t down_bpb = pn_gguf_bytes_per_block(down_exps_info->type);
                int down_bs = pn_gguf_block_size(down_exps_info->type);

                /* gate_exps/up_exps: GGUF dims [n_embd, n_ff, n_experts] innermost-first
                 * Per expert: rows=n_ff, cols=n_embd, row_stride based on n_embd */
                int gate_cols = config->n_embd;
                int gate_rows = config->n_ff;
                size_t gate_row_stride = ((size_t)gate_cols / (size_t)gate_bs) * gate_bpb;
                size_t gate_expert_stride = (size_t)gate_rows * gate_row_stride;

                /* down_exps: GGUF dims [n_ff, n_embd, n_experts] innermost-first
                 * Per expert: rows=n_embd, cols=n_ff, row_stride based on n_ff */
                int down_cols = config->n_ff;
                int down_rows = config->n_embd;
                size_t down_row_stride = ((size_t)down_cols / (size_t)down_bs) * down_bpb;
                size_t down_expert_stride = (size_t)down_rows * down_row_stride;

                const char *gate_base = (const char *)gguf->data + gate_exps_info->offset;
                const char *up_base   = (const char *)gguf->data + up_exps_info->offset;
                const char *down_base = (const char *)gguf->data + down_exps_info->offset;

                for (int e = 0; e < n_exp; e++) {
                    pn_moe_expert *exp = &l->experts[e];

                    exp->w_gate.data       = gate_base + e * gate_expert_stride;
                    exp->w_gate.type       = gate_exps_info->type;
                    exp->w_gate.rows       = gate_rows;
                    exp->w_gate.cols       = gate_cols;
                    exp->w_gate.row_stride = gate_row_stride;

                    exp->w_up.data       = up_base + e * gate_expert_stride;
                    exp->w_up.type       = up_exps_info->type;
                    exp->w_up.rows       = gate_rows;
                    exp->w_up.cols       = gate_cols;
                    exp->w_up.row_stride = gate_row_stride;

                    exp->w_down.data       = down_base + e * down_expert_stride;
                    exp->w_down.type       = down_exps_info->type;
                    exp->w_down.rows       = down_rows;
                    exp->w_down.cols       = down_cols;
                    exp->w_down.row_stride = down_row_stride;
                }
            } else {
                /* Split format: blk.N.ffn_gate.E.weight per expert */
                for (int e = 0; e < n_exp; e++) {
                    pn_moe_expert *exp = &l->experts[e];

                    snprintf(name, sizeof(name), "blk.%d.ffn_gate.%d.weight", i, e);
                    if (pn_qweight_from_gguf(&exp->w_gate, gguf, name) < 0) {
                        fprintf(stderr, "llama: failed to load %s\n", name);
                        pn_llama_qweights_free(qw);
                        return NULL;
                    }
                    snprintf(name, sizeof(name), "blk.%d.ffn_up.%d.weight", i, e);
                    if (pn_qweight_from_gguf(&exp->w_up, gguf, name) < 0) {
                        fprintf(stderr, "llama: failed to load %s\n", name);
                        pn_llama_qweights_free(qw);
                        return NULL;
                    }
                    snprintf(name, sizeof(name), "blk.%d.ffn_down.%d.weight", i, e);
                    if (pn_qweight_from_gguf(&exp->w_down, gguf, name) < 0) {
                        fprintf(stderr, "llama: failed to load %s\n", name);
                        pn_llama_qweights_free(qw);
                        return NULL;
                    }
                }
            }
        } else {
            /* Standard FFN weights (non-MoE) */
            LOAD_QW(w_gate, "blk.%d.ffn_gate.weight");
            LOAD_QW(w_up,   "blk.%d.ffn_up.weight");
            LOAD_QW(w_down, "blk.%d.ffn_down.weight");
        }

        #undef LOAD_QW

        /* Optional attention biases (Qwen2 has them, Llama doesn't) */
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->bq = dequant_to_f32(gguf, name, NULL);

        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->bk = dequant_to_f32(gguf, name, NULL);

        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->bv = dequant_to_f32(gguf, name, NULL);

        snprintf(name, sizeof(name), "blk.%d.attn_output.bias", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->bo = dequant_to_f32(gguf, name, NULL);

        /* Optional per-head Q/K RMSNorm weights (Qwen3) */
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->attn_q_norm = dequant_to_f32(gguf, name, NULL);
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
        if (pn_gguf_find_tensor(gguf, name))
            l->attn_k_norm = dequant_to_f32(gguf, name, NULL);

        fprintf(stderr, "\r  loaded layer %d/%d", i + 1, config->n_layer);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    return qw;
}

/* ============================================================
 * F32 weight loading from GGUF (Phase 1 legacy)
 * ============================================================ */

pn_llama_weights* pn_llama_load_gguf(pn_gguf_file *gguf, pn_config *config) {
    if (!gguf || !config) return NULL;

    if (read_config(gguf, config, arch_prefix(config->arch)) < 0) return NULL;

    pn_llama_weights *w = calloc(1, sizeof(pn_llama_weights));
    if (!w) return NULL;
    w->n_layers = config->n_layer;
    w->gguf = gguf;

    w->tok_embd = gguf_to_tensor(gguf, "token_embd.weight");
    w->output_norm = gguf_to_tensor(gguf, "output_norm.weight");

    w->output = gguf_to_tensor(gguf, "output.weight");
    if (!w->output && w->tok_embd) {
        w->output = tm_clone(w->tok_embd);
    }

    if (!w->tok_embd || !w->output_norm || !w->output) {
        fprintf(stderr, "llama: failed to load global tensors\n");
        pn_llama_weights_free(w);
        return NULL;
    }

    if (config->n_vocab == 0) {
        config->n_vocab = (int)w->tok_embd->shape[0];
    }

    w->layers = calloc(config->n_layer, sizeof(pn_llama_layer));
    if (!w->layers) {
        pn_llama_weights_free(w);
        return NULL;
    }

    for (int i = 0; i < config->n_layer; i++) {
        char name[128];
        pn_llama_layer *l = &w->layers[i];

        #define LOAD_LAYER(field, fmt) do { \
            snprintf(name, sizeof(name), fmt, i); \
            l->field = gguf_to_tensor(gguf, name); \
            if (!l->field) { \
                fprintf(stderr, "llama: failed to load %s\n", name); \
                pn_llama_weights_free(w); \
                return NULL; \
            } \
        } while(0)

        LOAD_LAYER(attn_norm, "blk.%d.attn_norm.weight");
        LOAD_LAYER(wq,       "blk.%d.attn_q.weight");
        LOAD_LAYER(wk,       "blk.%d.attn_k.weight");
        LOAD_LAYER(wv,       "blk.%d.attn_v.weight");
        LOAD_LAYER(wo,       "blk.%d.attn_output.weight");
        LOAD_LAYER(ffn_norm, "blk.%d.ffn_norm.weight");
        LOAD_LAYER(w_gate,   "blk.%d.ffn_gate.weight");
        LOAD_LAYER(w_up,     "blk.%d.ffn_up.weight");
        LOAD_LAYER(w_down,   "blk.%d.ffn_down.weight");

        #undef LOAD_LAYER

        fprintf(stderr, "\r  loaded layer %d/%d", i + 1, config->n_layer);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    return w;
}

/* ============================================================
 * HuggingFace safetensors loading (BF16 + ternary)
 * ============================================================ */

#include "safetensors/safetensors.h"
#include "tokenizer/tokenizer.h"

/* Read entire file into malloc'd buffer, NUL-terminated */
static char* read_file_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    return buf;
}

/* Simple JSON helpers for config.json parsing */
static const char* json_find(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static int json_int(const char *json, const char *key, int def) {
    const char *p = json_find(json, key);
    if (!p) return def;
    return atoi(p);
}

static double json_float(const char *json, const char *key, double def) {
    const char *p = json_find(json, key);
    if (!p) return def;
    return atof(p);
}

static int json_bool(const char *json, const char *key, int def) {
    const char *p = json_find(json, key);
    if (!p) return def;
    return strncmp(p, "true", 4) == 0;
}

static pn_arch detect_arch_from_json(const char *json) {
    const char *p = json_find(json, "model_type");
    if (!p || *p != '"') return PN_ARCH_LLAMA;
    p++;
    if (strncmp(p, "qwen2", 5) == 0) return PN_ARCH_QWEN2;
    if (strncmp(p, "llama", 5) == 0) return PN_ARCH_LLAMA;
    if (strncmp(p, "mistral", 7) == 0) return PN_ARCH_MISTRAL;
    if (strncmp(p, "phi3", 4) == 0 || strncmp(p, "phi", 3) == 0) return PN_ARCH_PHI;
    return PN_ARCH_LLAMA;
}

static int read_config_json(const char *path, pn_config *config) {
    char *json = read_file_text(path);
    if (!json) {
        fprintf(stderr, "llama_sf: failed to read config: %s\n", path);
        return -1;
    }

    config->arch = detect_arch_from_json(json);
    config->n_embd    = json_int(json, "hidden_size", 0);
    config->n_head    = json_int(json, "num_attention_heads", 0);
    config->n_kv_head = json_int(json, "num_key_value_heads", config->n_head);
    config->n_layer   = json_int(json, "num_hidden_layers", 0);
    config->n_ff      = json_int(json, "intermediate_size", 0);
    config->n_vocab   = json_int(json, "vocab_size", 0);
    config->n_ctx     = json_int(json, "max_position_embeddings", 32768);
    config->rope_theta    = (float)json_float(json, "rope_theta", 10000.0);
    config->rms_norm_eps  = (float)json_float(json, "rms_norm_eps", 1e-6);
    config->sliding_window = json_int(json, "sliding_window", 0);
    config->rope_dim = 0;
    config->n_experts = 0;
    config->n_experts_used = 0;

    int tied = json_bool(json, "tie_word_embeddings", 0);
    free(json);

    if (config->n_embd == 0 || config->n_head == 0 || config->n_layer == 0) {
        fprintf(stderr, "llama_sf: invalid config (n_embd=%d, n_head=%d, n_layer=%d)\n",
                config->n_embd, config->n_head, config->n_layer);
        return -1;
    }

    fprintf(stderr, "llama_sf: n_vocab=%d n_embd=%d n_head=%d n_kv=%d n_layer=%d n_ff=%d tied=%d\n",
            config->n_vocab, config->n_embd, config->n_head, config->n_kv_head,
            config->n_layer, config->n_ff, tied);
    return tied ? 1 : 0;  /* return tied flag */
}

/* BF16 -> F32 conversion */
static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

/* Dequantize a safetensors BF16 tensor to F32 (owned malloc) */
static float* sf_load_bf16(fs_file *sf, const char *name, int expected_numel) {
    fs_tensor_info *info = fs_get_info(sf, name);
    if (!info) return NULL;

    const void *raw = fs_get_data(sf, name);
    if (!raw) { fs_free_info(info); return NULL; }

    size_t numel = 1;
    for (int d = 0; d < info->ndim; d++) numel *= (size_t)info->shape[d];

    if (expected_numel > 0 && (int)numel != expected_numel) {
        fprintf(stderr, "llama_sf: %s: expected %d elements, got %zu\n", name, expected_numel, numel);
        fs_free_info(info);
        return NULL;
    }

    if (pn_host_admit_weight_copy((uint64_t)numel * sizeof(float), name) != 0) {
        fs_free_info(info);
        return NULL;
    }
    float *out = malloc(numel * sizeof(float));
    if (!out) { fs_free_info(info); return NULL; }

    if (info->dtype == FS_DTYPE_BF16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < numel; i++) out[i] = bf16_to_f32(src[i]);
    } else if (info->dtype == FS_DTYPE_F16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < numel; i++) out[i] = pn_half_to_float(src[i]);
    } else if (info->dtype == FS_DTYPE_F32) {
        memcpy(out, raw, numel * sizeof(float));
    } else {
        fprintf(stderr, "llama_sf: %s: unsupported dtype %s\n", name, fs_dtype_name(info->dtype));
        free(out);
        fs_free_info(info);
        return NULL;
    }

    fs_free_info(info);
    return out;
}

/* Load a BF16 weight matrix into a pn_qweight as F32 (owned) */
static int sf_load_weight_f32(fs_file *sf, const char *name, pn_qweight *qw, int rows, int cols) {
    float *data = sf_load_bf16(sf, name, rows * cols);
    if (!data) return -1;

    qw->data = data;
    qw->type = GGUF_QUANT_F32;
    qw->rows = rows;
    qw->cols = cols;
    qw->row_stride = (size_t)cols * sizeof(float);
    qw->scales = NULL;
    qw->group_size = 0;
    return 0;
}

/* Load a ternary weight from packed U8 + scales + shape tensors */
static int sf_load_weight_ternary(fs_file *sf, const char *base_name,
                                   pn_qweight *qw, int group_size) {
    char scale_name[256], shape_name[256];
    snprintf(scale_name, sizeof(scale_name), "%s.__scale", base_name);
    snprintf(shape_name, sizeof(shape_name), "%s.__shape", base_name);

    /* Read shape */
    const int32_t *shape_data = fs_get_i32(sf, shape_name);
    if (!shape_data) {
        fprintf(stderr, "llama_sf: missing %s\n", shape_name);
        return -1;
    }
    int rows = shape_data[0];
    int cols = shape_data[1];

    /* Read packed ternary data (U8) — copy to owned memory */
    const void *packed_raw = fs_get_data(sf, base_name);
    if (!packed_raw) {
        fprintf(stderr, "llama_sf: missing %s\n", base_name);
        return -1;
    }
    size_t packed_bytes_per_row = ((size_t)cols + 3) / 4;
    size_t total_packed = (size_t)rows * packed_bytes_per_row;
    uint8_t *packed = malloc(total_packed);
    if (!packed) return -1;
    memcpy(packed, packed_raw, total_packed);

    /* Read scales (F32) — copy to owned memory */
    int n_groups = (cols + group_size - 1) / group_size;
    size_t total_scales = (size_t)rows * n_groups;

    fs_tensor_info *scale_info = fs_get_info(sf, scale_name);
    if (!scale_info) {
        fprintf(stderr, "llama_sf: missing %s\n", scale_name);
        free(packed);
        return -1;
    }
    const void *scale_raw = fs_get_data(sf, scale_name);
    float *scales = malloc(total_scales * sizeof(float));
    if (!scales) { free(packed); fs_free_info(scale_info); return -1; }
    memcpy(scales, scale_raw, total_scales * sizeof(float));
    fs_free_info(scale_info);

    qw->data = packed;
    qw->type = PN_QUANT_TERNARY;
    qw->rows = rows;
    qw->cols = cols;
    qw->row_stride = packed_bytes_per_row;
    qw->scales = scales;
    qw->group_size = group_size;
    return 0;
}

/* Permute Q/K weight rows from HF natural order to GGUF interleaved order.
 * GGUF conversion interleaves first/second half of each head's dimensions
 * so that RoPE can rotate consecutive pairs (i, i+1).
 * Input layout per head:  [elem0, elem1, ..., elemH/2-1, elemH/2, ..., elemH-1]
 * Output layout per head: [elem0, elemH/2, elem1, elemH/2+1, ...] */
static void permute_qk_for_rope(pn_qweight *qw, int n_heads) {
    if (!qw->data || qw->type != GGUF_QUANT_F32) return;
    int head_dim = qw->rows / n_heads;
    int half = head_dim / 2;
    int cols = qw->cols;
    float *data = (float *)qw->data;
    float *temp = malloc((size_t)head_dim * cols * sizeof(float));
    if (!temp) return;

    for (int h = 0; h < n_heads; h++) {
        float *head_start = data + (size_t)h * head_dim * cols;
        memcpy(temp, head_start, (size_t)head_dim * cols * sizeof(float));
        for (int i = 0; i < half; i++) {
            memcpy(head_start + (size_t)(2 * i) * cols,
                   temp + (size_t)i * cols, (size_t)cols * sizeof(float));
            memcpy(head_start + (size_t)(2 * i + 1) * cols,
                   temp + (size_t)(i + half) * cols, (size_t)cols * sizeof(float));
        }
    }
    free(temp);
}

/* Permute Q/K bias from HF to GGUF interleaved order (same as weight row permutation). */
static void permute_qk_bias(float *bias, int n_heads, int head_dim) {
    if (!bias) return;
    int half = head_dim / 2;
    float *temp = malloc((size_t)head_dim * sizeof(float));
    if (!temp) return;
    for (int h = 0; h < n_heads; h++) {
        float *head_start = bias + h * head_dim;
        memcpy(temp, head_start, (size_t)head_dim * sizeof(float));
        for (int i = 0; i < half; i++) {
            head_start[2 * i]     = temp[i];
            head_start[2 * i + 1] = temp[i + half];
        }
    }
    free(temp);
}

/* Permute Q/K weight rows for ternary (generic: works with any row_stride).
 * Same interleaving as permute_qk_for_rope but operates on packed byte rows
 * and also permutes per-row scale arrays. */
static void permute_qk_ternary(pn_qweight *qw, int n_heads) {
    if (!qw->data) return;
    int head_dim = qw->rows / n_heads;
    int half = head_dim / 2;
    size_t rs = qw->row_stride;
    uint8_t *data = (uint8_t *)qw->data;
    int n_groups = (qw->cols + qw->group_size - 1) / qw->group_size;

    uint8_t *temp = malloc((size_t)head_dim * rs);
    float *stemp = qw->scales ? malloc((size_t)head_dim * n_groups * sizeof(float)) : NULL;
    if (!temp) return;

    for (int h = 0; h < n_heads; h++) {
        uint8_t *hd = data + (size_t)h * head_dim * rs;
        memcpy(temp, hd, (size_t)head_dim * rs);
        for (int i = 0; i < half; i++) {
            memcpy(hd + (size_t)(2 * i) * rs, temp + (size_t)i * rs, rs);
            memcpy(hd + (size_t)(2 * i + 1) * rs, temp + (size_t)(i + half) * rs, rs);
        }
        if (qw->scales && stemp) {
            float *hs = (float *)qw->scales + (size_t)h * head_dim * n_groups;
            memcpy(stemp, hs, (size_t)head_dim * n_groups * sizeof(float));
            for (int i = 0; i < half; i++) {
                memcpy(hs + (size_t)(2 * i) * n_groups,
                       stemp + (size_t)i * n_groups, (size_t)n_groups * sizeof(float));
                memcpy(hs + (size_t)(2 * i + 1) * n_groups,
                       stemp + (size_t)(i + half) * n_groups, (size_t)n_groups * sizeof(float));
            }
        }
    }
    free(temp);
    free(stemp);
}

/* Free owned data in a pn_qweight (for safetensors path) */
static void free_owned_qweight(pn_qweight *qw) {
    if (qw->data) free((void *)qw->data);
    if (qw->scales) free((void *)qw->scales);
    memset(qw, 0, sizeof(*qw));
}

pn_llama_qweights* pn_llama_load_safetensors(const char *model_path,
                                              const char *config_path,
                                              pn_config *config) {
    if (!model_path || !config_path || !config) return NULL;

    /* Parse config.json */
    int tied = read_config_json(config_path, config);
    if (tied < 0) return NULL;

    /* Open safetensors */
    fs_file *sf = fs_open(model_path);
    if (!sf) {
        fprintf(stderr, "llama_sf: failed to open %s\n", model_path);
        return NULL;
    }

    /* Detect format: HuggingFace BF16 or ternarized */
    int is_ternary = fs_has_tensor(sf, "layers.0.self_attn.q_proj.weight.__scale");
    int is_hf = fs_has_tensor(sf, "model.layers.0.self_attn.q_proj.weight");

    if (!is_ternary && !is_hf) {
        fprintf(stderr, "llama_sf: unrecognized safetensors format\n");
        fs_close(sf);
        return NULL;
    }

    fprintf(stderr, "llama_sf: detected %s format\n", is_ternary ? "ternary" : "HF BF16");

    pn_llama_qweights *qw = calloc(1, sizeof(pn_llama_qweights));
    if (!qw) { fs_close(sf); return NULL; }
    qw->is_quantized = 1;
    qw->n_layers = config->n_layer;
    qw->n_embd = config->n_embd;
    qw->n_vocab = config->n_vocab;
    qw->gguf = NULL;  /* signals owned data */

    int n_embd = config->n_embd;
    int n_head = config->n_head;
    int n_kv = config->n_kv_head;
    int head_dim = n_embd / n_head;
    int n_ff = config->n_ff;

    /* Token embeddings (always F32/BF16, never ternary) */
    if (is_hf) {
        qw->tok_embd = sf_load_bf16(sf, "model.embed_tokens.weight", config->n_vocab * n_embd);
    } else {
        qw->tok_embd = sf_load_bf16(sf, "embed_tokens.weight", config->n_vocab * n_embd);
    }
    if (!qw->tok_embd) {
        fprintf(stderr, "llama_sf: failed to load embeddings\n");
        pn_llama_qweights_free(qw);
        fs_close(sf);
        return NULL;
    }

    /* Final norm */
    if (is_hf) {
        qw->output_norm = sf_load_bf16(sf, "model.norm.weight", n_embd);
    } else {
        qw->output_norm = sf_load_bf16(sf, "norm.weight", n_embd);
    }
    if (!qw->output_norm) {
        fprintf(stderr, "llama_sf: failed to load output norm\n");
        pn_llama_qweights_free(qw);
        fs_close(sf);
        return NULL;
    }

    /* LM head: check for lm_head.weight, else tie to embeddings */
    const char *head_name = is_hf ? "lm_head.weight" : "lm_head.weight";
    if (fs_has_tensor(sf, head_name) && !tied) {
        float *head = sf_load_bf16(sf, head_name, config->n_vocab * n_embd);
        if (head) {
            qw->output_f32 = head;
            qw->output.data = head;
            qw->output.type = GGUF_QUANT_F32;
            qw->output.rows = config->n_vocab;
            qw->output.cols = n_embd;
            qw->output.row_stride = (size_t)n_embd * sizeof(float);
            qw->output.scales = NULL;
            qw->output.group_size = 0;
        }
    }
    if (!qw->output.data) {
        /* Tied embeddings */
        size_t sz = (size_t)config->n_vocab * n_embd * sizeof(float);
        qw->output_f32 = pn_host_admit_weight_copy(sz, "tied output (lm_head)") == 0
                             ? malloc(sz) : NULL;
        if (!qw->output_f32) {
            pn_llama_qweights_free(qw);
            fs_close(sf);
            return NULL;
        }
        memcpy(qw->output_f32, qw->tok_embd, sz);
        qw->output.data = qw->output_f32;
        qw->output.type = GGUF_QUANT_F32;
        qw->output.rows = config->n_vocab;
        qw->output.cols = n_embd;
        qw->output.row_stride = (size_t)n_embd * sizeof(float);
        qw->output.scales = NULL;
        qw->output.group_size = 0;
    }

    /* Per-layer weights */
    qw->layers = calloc(config->n_layer, sizeof(pn_llama_qlayer));
    if (!qw->layers) {
        pn_llama_qweights_free(qw);
        fs_close(sf);
        return NULL;
    }

    for (int i = 0; i < config->n_layer; i++) {
        pn_llama_qlayer *l = &qw->layers[i];
        char name[256];
        int ok = 1;

        if (is_hf) {
            /* HuggingFace BF16 format */

            /* Norms */
            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
            l->attn_norm = sf_load_bf16(sf, name, n_embd);
            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
            l->ffn_norm = sf_load_bf16(sf, name, n_embd);
            if (!l->attn_norm || !l->ffn_norm) { ok = 0; goto layer_fail; }

            /* Attention projections */
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->wq, n_head * head_dim, n_embd) == 0;
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->wk, n_kv * head_dim, n_embd) == 0;
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->wv, n_kv * head_dim, n_embd) == 0;
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->wo, n_embd, n_head * head_dim) == 0;

            /* FFN projections */
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->w_gate, n_ff, n_embd) == 0;
            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->w_up, n_ff, n_embd) == 0;
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", i);
            ok = ok && sf_load_weight_f32(sf, name, &l->w_down, n_embd, n_ff) == 0;

            if (!ok) goto layer_fail;

            /* Permute Q/K weights from HF to GGUF interleaved layout for RoPE */
            permute_qk_for_rope(&l->wq, n_head);
            permute_qk_for_rope(&l->wk, n_kv);

            /* Optional biases (Qwen2) */
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", i);
            if (fs_has_tensor(sf, name)) l->bq = sf_load_bf16(sf, name, n_head * head_dim);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", i);
            if (fs_has_tensor(sf, name)) l->bk = sf_load_bf16(sf, name, n_kv * head_dim);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", i);
            if (fs_has_tensor(sf, name)) l->bv = sf_load_bf16(sf, name, n_kv * head_dim);

            /* Permute Q/K biases to match permuted weights */
            if (l->bq) permute_qk_bias(l->bq, n_head, head_dim);
            if (l->bk) permute_qk_bias(l->bk, n_kv, head_dim);

        } else {
            /* Ternarized format */
            int gs = 128; /* default group_size from ternarizer */

            /* Norms (always F32) */
            snprintf(name, sizeof(name), "layers.%d.input_layernorm.weight", i);
            l->attn_norm = sf_load_bf16(sf, name, n_embd);
            snprintf(name, sizeof(name), "layers.%d.post_attention_layernorm.weight", i);
            l->ffn_norm = sf_load_bf16(sf, name, n_embd);
            if (!l->attn_norm || !l->ffn_norm) { ok = 0; goto layer_fail; }

            /* Ternary weights (HF-style names from merge output) */
            snprintf(name, sizeof(name), "layers.%d.self_attn.q_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->wq, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.self_attn.k_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->wk, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.self_attn.v_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->wv, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.self_attn.o_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->wo, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.mlp.gate_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->w_gate, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.mlp.up_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->w_up, gs) == 0;
            snprintf(name, sizeof(name), "layers.%d.mlp.down_proj.weight", i);
            ok = ok && sf_load_weight_ternary(sf, name, &l->w_down, gs) == 0;

            if (!ok) goto layer_fail;

            /* Permute ternary Q/K weights for interleaved RoPE */
            permute_qk_ternary(&l->wq, n_head);
            permute_qk_ternary(&l->wk, n_kv);

            /* Load biases from original model if available */
            snprintf(name, sizeof(name), "layers.%d.self_attn.q_proj.bias", i);
            if (fs_has_tensor(sf, name)) {
                l->bq = sf_load_bf16(sf, name, n_head * head_dim);
                if (l->bq) permute_qk_bias(l->bq, n_head, head_dim);
            }
            snprintf(name, sizeof(name), "layers.%d.self_attn.k_proj.bias", i);
            if (fs_has_tensor(sf, name)) {
                l->bk = sf_load_bf16(sf, name, n_kv * head_dim);
                if (l->bk) permute_qk_bias(l->bk, n_kv, head_dim);
            }
            snprintf(name, sizeof(name), "layers.%d.self_attn.v_proj.bias", i);
            if (fs_has_tensor(sf, name)) l->bv = sf_load_bf16(sf, name, n_kv * head_dim);
        }

        fprintf(stderr, "\r  loaded layer %d/%d", i + 1, config->n_layer);
        fflush(stderr);
        continue;

layer_fail:
        fprintf(stderr, "\nllama_sf: failed to load layer %d\n", i);
        pn_llama_qweights_free(qw);
        fs_close(sf);
        return NULL;
    }
    fprintf(stderr, "\n");

    fs_close(sf);
    return qw;
}

/* ============================================================
 * HuggingFace tokenizer loader (vocab.json + merges.txt)
 * ============================================================ */

/* Find closing quote of a JSON string, handling escape sequences.
 * p must point to the first char after the opening quote.
 * Returns pointer to the closing quote, or NULL if not found. */
static const char* json_skip_string(const char *p) {
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
            if (*p == 'u') {
                /* \uXXXX — skip 4 hex digits */
                for (int i = 0; i < 4 && p[1]; i++) p++;
            }
        }
        p++;
    }
    return *p == '"' ? p : NULL;
}

/* Unescape JSON string [start, end) into malloc'd buffer.
 * Handles: \" \\ \/ \n \t \r \b \f \uXXXX (including surrogate pairs → UTF-8).
 * Returns owned string, caller frees. */
static char* json_unescape(const char *start, const char *end) {
    size_t max_len = (size_t)(end - start);
    char *out = malloc(max_len * 4 + 1); /* \uXXXX → up to 4 UTF-8 bytes */
    if (!out) return NULL;

    char *w = out;
    const char *p = start;
    while (p < end) {
        if (*p == '\\' && p + 1 < end) {
            p++;
            switch (*p) {
                case '"':  *w++ = '"'; break;
                case '\\': *w++ = '\\'; break;
                case '/':  *w++ = '/'; break;
                case 'n':  *w++ = '\n'; break;
                case 't':  *w++ = '\t'; break;
                case 'r':  *w++ = '\r'; break;
                case 'b':  *w++ = '\b'; break;
                case 'f':  *w++ = '\f'; break;
                case 'u': {
                    if (p + 4 >= end) { p++; continue; }
                    unsigned cp = 0;
                    for (int i = 1; i <= 4; i++) {
                        char c = p[i];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                        else if (c >= 'a' && c <= 'f') cp |= 10u + (unsigned)(c - 'a');
                        else if (c >= 'A' && c <= 'F') cp |= 10u + (unsigned)(c - 'A');
                    }
                    p += 4; /* p now at last hex digit */
                    /* Handle UTF-16 surrogate pairs: \uD800-DBFF + \uDC00-DFFF */
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        p + 6 < end && p[1] == '\\' && p[2] == 'u') {
                        unsigned lo = 0;
                        for (int i = 3; i <= 6; i++) {
                            char c = p[i];
                            lo <<= 4;
                            if (c >= '0' && c <= '9') lo |= (unsigned)(c - '0');
                            else if (c >= 'a' && c <= 'f') lo |= 10u + (unsigned)(c - 'a');
                            else if (c >= 'A' && c <= 'F') lo |= 10u + (unsigned)(c - 'A');
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p += 6; /* skip \uXXXX of low surrogate */
                    }
                    /* Encode as UTF-8 */
                    if (cp < 0x80) {
                        *w++ = (char)cp;
                    } else if (cp < 0x800) {
                        *w++ = (char)(0xC0 | (cp >> 6));
                        *w++ = (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        *w++ = (char)(0xE0 | (cp >> 12));
                        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
                        *w++ = (char)(0x80 | (cp & 0x3F));
                    } else {
                        *w++ = (char)(0xF0 | (cp >> 18));
                        *w++ = (char)(0x80 | ((cp >> 12) & 0x3F));
                        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
                        *w++ = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: *w++ = *p; break;
            }
            p++;
        } else {
            *w++ = *p++;
        }
    }
    *w = '\0';
    return out;
}

ft_tokenizer* pn_llama_load_hf_tokenizer(const char *dir_path, const pn_config *config) {
    char vocab_path[1024], merges_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", dir_path);
    snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", dir_path);

    /* Read vocab.json */
    char *vocab_json = read_file_text(vocab_path);
    if (!vocab_json) {
        fprintf(stderr, "llama_sf: failed to read %s\n", vocab_path);
        return NULL;
    }

    /* Count vocab size and allocate sorted array */
    int n_vocab = config->n_vocab;
    if (n_vocab <= 0) n_vocab = 151936; /* Qwen2 default */

    char **tokens = calloc(n_vocab, sizeof(char *));
    if (!tokens) { free(vocab_json); return NULL; }

    /* Parse vocab.json: find "token": id pairs.
     * Properly handles JSON escape sequences (\", \\, \uXXXX, etc.) */
    const char *p = vocab_json;
    int n_parsed = 0;

    /* Skip to opening brace */
    while (*p && *p != '{') p++;
    if (*p == '{') p++;

    while (*p) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == '}' || *p == '\0') break;

        /* Expect opening quote of key */
        if (*p != '"') { p++; continue; }
        p++; /* skip opening quote */

        /* Find end of key string (handles \" and \\ correctly) */
        const char *key_start = p;
        const char *key_end = json_skip_string(p);
        if (!key_end) break;
        p = key_end + 1; /* past closing quote */

        /* Skip colon + whitespace */
        while (*p == ' ' || *p == '\t' || *p == ':') p++;

        /* Parse integer ID */
        if (*p >= '0' && *p <= '9') {
            int id = atoi(p);
            while (*p >= '0' && *p <= '9') p++;
            if (id >= 0 && id < n_vocab) {
                tokens[id] = json_unescape(key_start, key_end);
                if (tokens[id]) n_parsed++;
            }
        }
    }
    fprintf(stderr, "llama_sf: parsed %d/%d tokens from vocab.json\n", n_parsed, n_vocab);
    free(vocab_json);

    /* Create tokenizer and add tokens in order */
    ft_tokenizer *tok = ft_new();
    if (!tok) {
        for (int i = 0; i < n_vocab; i++) free(tokens[i]);
        free(tokens);
        return NULL;
    }

    for (int i = 0; i < n_vocab; i++) {
        if (tokens[i]) {
            ft_add_token(tok, tokens[i]);
            free(tokens[i]);
        } else {
            /* Fill gaps with placeholder */
            char placeholder[32];
            snprintf(placeholder, sizeof(placeholder), "<|reserved_%d|>", i);
            ft_add_token(tok, placeholder);
        }
    }
    free(tokens);

    /* Load merges.txt */
    char *merges = read_file_text(merges_path);
    if (merges) {
        char *line = merges;
        int n_merges = 0;
        while (*line) {
            char *eol = strchr(line, '\n');
            if (!eol) eol = line + strlen(line);
            size_t line_len = (size_t)(eol - line);

            /* Skip comment lines and empty lines */
            if (line_len > 0 && line[0] != '#') {
                /* Find space separator */
                char *space = NULL;
                for (char *s = line; s < eol; s++) {
                    if (*s == ' ') { space = s; break; }
                }
                if (space && space > line && space < eol - 1) {
                    *space = '\0';
                    char saved = *eol;
                    *eol = '\0';
                    ft_add_merge(tok, line, space + 1);
                    *eol = saved;
                    *space = ' ';
                    n_merges++;
                }
            }

            if (*eol) line = eol + 1;
            else break;
        }
        fprintf(stderr, "llama_sf: loaded %d merges\n", n_merges);
        free(merges);
    } else {
        fprintf(stderr, "llama_sf: no merges.txt found, BPE disabled\n");
    }

    /* Set BOS/EOS from config */
    int bos_id = json_int("{\"bos_token_id\": 151643}", "bos_token_id", 151643);
    int eos_id = json_int("{\"eos_token_id\": 151643}", "eos_token_id", 151643);
    (void)bos_id;

    const char *eos_tok = ft_id_to_token(tok, eos_id);
    if (eos_tok) ft_set_eos_token(tok, eos_tok);

    fprintf(stderr, "llama_sf: tokenizer loaded (%d vocab)\n", n_vocab);
    return tok;
}

/* ============================================================
 * Free
 * ============================================================ */

void pn_llama_weights_free(pn_llama_weights *w) {
    if (!w) return;
    if (w->tok_embd) tm_free(w->tok_embd);
    if (w->output_norm) tm_free(w->output_norm);
    if (w->output) tm_free(w->output);

    if (w->layers) {
        for (int i = 0; i < w->n_layers; i++) {
            pn_llama_layer *l = &w->layers[i];
            if (l->attn_norm) tm_free(l->attn_norm);
            if (l->wq)       tm_free(l->wq);
            if (l->wk)       tm_free(l->wk);
            if (l->wv)       tm_free(l->wv);
            if (l->wo)       tm_free(l->wo);
            if (l->ffn_norm) tm_free(l->ffn_norm);
            if (l->w_gate)   tm_free(l->w_gate);
            if (l->w_up)     tm_free(l->w_up);
            if (l->w_down)   tm_free(l->w_down);
        }
        free(w->layers);
    }
    free(w);
}

void pn_llama_qweights_free(pn_llama_qweights *qw) {
    if (!qw) return;
    free(qw->tok_embd);
    free(qw->output_norm);
    free(qw->output_f32);  /* NULL-safe if tied embeddings */

    if (qw->layers) {
        int owned = (qw->gguf == NULL);  /* safetensors path: data is owned */
        for (int i = 0; i < qw->n_layers; i++) {
            free(qw->layers[i].attn_norm);
            free(qw->layers[i].ffn_norm);
            free(qw->layers[i].bq);
            free(qw->layers[i].bk);
            free(qw->layers[i].bv);
            free(qw->layers[i].bo);
            free(qw->layers[i].attn_q_norm);  /* NULL-safe (Qwen3) */
            free(qw->layers[i].attn_k_norm);
            free(qw->layers[i].experts);  /* NULL-safe */

            if (owned) {
                free_owned_qweight(&qw->layers[i].wq);
                free_owned_qweight(&qw->layers[i].wk);
                free_owned_qweight(&qw->layers[i].wv);
                free_owned_qweight(&qw->layers[i].wo);
                free_owned_qweight(&qw->layers[i].w_gate);
                free_owned_qweight(&qw->layers[i].w_up);
                free_owned_qweight(&qw->layers[i].w_down);
            }
        }
        free(qw->layers);
    }
    free(qw);
}

/* ============================================================
 * KV Cache
 * ============================================================ */

pn_llama_kv_cache* pn_llama_kv_cache_new(int n_layers, int n_kv_head, int max_seq, int head_dim) {
    pn_llama_kv_cache *cache = calloc(1, sizeof(pn_llama_kv_cache));
    if (!cache) return NULL;

    cache->n_layers = n_layers;
    cache->max_seq = max_seq;
    cache->k_cache = calloc(n_layers, sizeof(tm_tensor *));
    cache->v_cache = calloc(n_layers, sizeof(tm_tensor *));
    if (!cache->k_cache || !cache->v_cache) {
        pn_llama_kv_cache_free(cache);
        return NULL;
    }

    int64_t shape[] = { n_kv_head, max_seq, head_dim };
    for (int i = 0; i < n_layers; i++) {
        cache->k_cache[i] = tm_zeros(shape, 3, TM_F32);
        cache->v_cache[i] = tm_zeros(shape, 3, TM_F32);
        if (!cache->k_cache[i] || !cache->v_cache[i]) {
            pn_llama_kv_cache_free(cache);
            return NULL;
        }
    }

    return cache;
}

void pn_llama_kv_cache_free(pn_llama_kv_cache *cache) {
    if (!cache) return;
    if (cache->k_cache) {
        for (int i = 0; i < cache->n_layers; i++) {
            if (cache->k_cache[i]) tm_free(cache->k_cache[i]);
        }
        free(cache->k_cache);
    }
    if (cache->v_cache) {
        for (int i = 0; i < cache->n_layers; i++) {
            if (cache->v_cache[i]) tm_free(cache->v_cache[i]);
        }
        free(cache->v_cache);
    }
    free(cache);
}

/* ============================================================
 * RoPE (Rotary Position Embeddings) — in-place on float*
 * ============================================================ */

static void apply_rope_raw(float *data, int n_heads, int head_dim,
                           int rope_dim, int pos, float theta_base) {
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < rot; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[i];
            float x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

void pn_llama_qkv_finish(const pn_llama_qlayer *l, const pn_config *cfg,
                         float *q, float *k, float *v)
{
    int head_dim = pn_head_dim(cfg);

    if (l->bq) pn_vadd_f32(q, l->bq, q, l->wq.rows);
    if (l->bk) pn_vadd_f32(k, l->bk, k, l->wk.rows);
    if (l->bv) pn_vadd_f32(v, l->bv, v, l->wv.rows);
    if (l->attn_q_norm)
        for (int h = 0; h < cfg->n_head; h++)
            pn_vrms_norm_f32(q + (size_t)h * head_dim, l->attn_q_norm,
                             q + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
    if (l->attn_k_norm)
        for (int h = 0; h < cfg->n_kv_head; h++)
            pn_vrms_norm_f32(k + (size_t)h * head_dim, l->attn_k_norm,
                             k + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
}

/* NEOX-style RoPE: pairs dim i with i+half (half = rot/2). Qwen3 GGUF weights
 * are stored unpermuted, so the rotation is NEOX (matching HF), unlike the
 * interleaved layout the permuted Llama/Qwen2 GGUFs above expect. */
static void apply_rope_neox(float *data, int n_heads, int head_dim,
                            int rope_dim, int pos, float theta_base) {
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
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
    const float *q_data;        /* [n_head, head_dim] */
    const float *k_cache_data;  /* [n_kv_head, max_seq, head_dim] */
    const float *v_cache_data;  /* [n_kv_head, max_seq, head_dim] */
    float *attn_out;            /* [n_head, head_dim] */
    float *scores_buf;          /* [n_head, max_seq] pre-allocated */
    int head_dim;
    int n_kv_groups;
    int max_seq;
    int seq_len;
    float scale;
} attn_args;

static void compute_attn_heads(int start, int end, void *arg) {
    attn_args *a = (attn_args *)arg;

    for (int qh = start; qh < end; qh++) {
        int kv_h = qh / a->n_kv_groups;
        const float *q_vec = a->q_data + qh * a->head_dim;
        const float *k_base = a->k_cache_data + kv_h * a->max_seq * a->head_dim;
        const float *v_base = a->v_cache_data + kv_h * a->max_seq * a->head_dim;
        float *scores = a->scores_buf + qh * a->max_seq;
        float *out_vec = a->attn_out + qh * a->head_dim;

        /* Q @ K^T for each cached position */
        for (int s = 0; s < a->seq_len; s++) {
            scores[s] = pn_dot_f32(q_vec, k_base + s * a->head_dim, a->head_dim) * a->scale;
        }

        /* Softmax */
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

        /* Weighted sum of V */
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

/* Blocked ("flash") prefill attention. The per-query path (compute_attn_heads
 * called once per position) re-reads the whole K/V cache for every query, so
 * prefill's attention is O(P^2) DRAM traffic and dominates prefill. This
 * processes a BLOCK of queries together, streaming K and V in blocks that stay
 * cache-resident and are reused across the query block, cutting K/V reads from
 * P times to ~P/BQ times. Same math as compute_attn_heads (full-precision
 * softmax; scores tile is L2-resident). Parallelized over query heads.
 * Assumes contiguous keys at positions 0..base_pos+P-1 (no sliding-window
 * wraparound), which holds for a fresh prefill. */
typedef struct {
    const float *q;             /* [P, n_head*head_dim] */
    const float *k_cache;       /* [n_kv_head, max_seq, head_dim] */
    const float *v_cache;
    float       *out;           /* [P, n_head*head_dim] */
    int P, base_pos, head_dim, n_head, n_kv_groups, max_seq;
    float scale;
} pf_attn_args;

/* Query block sized so the QK^T / A@V sgemm M dimension is AMX-friendly. At
 * PF_BQ=128 the scores tile is 128 x (base+P) x 4 bytes (~700 KB for ~1400
 * keys), which stays L2-resident per worker. BLAS tiles the K dimension
 * internally, so no explicit key-blocking is needed. */
#define PF_BQ 128

static void prefill_flash_range(int start, int end, void *arg) {
    const pf_attn_args *a = (const pf_attn_args *)arg;
    int hd = a->head_dim, P = a->P, base = a->base_pos, maxs = a->max_seq;
    int qdim = a->n_head * hd;
    int nkey_max = base + P;                       /* keys 0..base+P-1 */
    float *scores = malloc((size_t)PF_BQ * nkey_max * sizeof(float));
    if (!scores) return;

    for (int qh = start; qh < end; qh++) {
        int kv_h = qh / a->n_kv_groups;
        const float *k_base = a->k_cache + (size_t)kv_h * maxs * hd;
        const float *v_base = a->v_cache + (size_t)kv_h * maxs * hd;

        for (int qb0 = 0; qb0 < P; qb0 += PF_BQ) {
            int bq = (qb0 + PF_BQ < P) ? PF_BQ : (P - qb0);
            int kmax = base + qb0 + bq;             /* exclusive causal max in block */
            /* This head's query and output slabs: rows strided by qdim (heads
             * are interleaved), each hd wide, starting at column qh*hd. */
            const float *qblk = a->q   + (size_t)qb0 * qdim + (size_t)qh * hd;
            float       *oblk = a->out + (size_t)qb0 * qdim + (size_t)qh * hd;

#ifdef USE_BLAS
            /* 1. Full score rectangle S[bq, kmax] = scale * Qblk @ K[0:kmax]^T.
             *    NoTrans A (bq x hd, lda=qdim) times Trans B (K stored kmax x hd,
             *    ldb=hd) yields Qblk @ K^T; the softmax scale folds into alpha.
             *    Columns past each row's causal limit are computed, then masked. */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bq, kmax, hd,
                        a->scale, qblk, qdim, k_base, hd, 0.0f, scores, nkey_max);

            /* 2. Exact per-row causal softmax over [0, lim) (unchanged math),
             *    then zero the non-causal tail [lim, kmax) so it contributes
             *    exactly 0.0 to the A@V GEMM below. */
            for (int i = 0; i < bq; i++) {
                int lim = base + qb0 + i + 1;
                float *sc = scores + (size_t)i * nkey_max;
                float mx = sc[0];
                for (int s = 1; s < lim; s++) if (sc[s] > mx) mx = sc[s];
                float sum = 0.0f;
                for (int s = 0; s < lim; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
                float inv = 1.0f / sum;
                for (int s = 0; s < lim; s++) sc[s] *= inv;
                for (int s = lim; s < kmax; s++) sc[s] = 0.0f;
            }

            /* 3. out[bq, hd] = S[bq, kmax] @ V[0:kmax]. The zeroed tail makes
             *    this exactly the masked weighted sum of V. */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bq, hd, kmax,
                        1.0f, scores, nkey_max, v_base, hd, 0.0f, oblk, qdim);
#else
            (void)qblk; (void)oblk;
            /* Portable fallback: same per-row causal softmax + weighted V sum. */
            for (int i = 0; i < bq; i++) {
                int lim = base + qb0 + i + 1;
                const float *qv = a->q + (size_t)(qb0 + i) * qdim + (size_t)qh * hd;
                float *sc = scores + (size_t)i * nkey_max;
                for (int s = 0; s < lim; s++)
                    sc[s] = pn_dot_f32(qv, k_base + (size_t)s * hd, hd) * a->scale;
                float mx = sc[0];
                for (int s = 1; s < lim; s++) if (sc[s] > mx) mx = sc[s];
                float sum = 0.0f;
                for (int s = 0; s < lim; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
                float inv = 1.0f / sum;
                float *out = a->out + (size_t)(qb0 + i) * qdim + (size_t)qh * hd;
                memset(out, 0, hd * sizeof(float));
                for (int s = 0; s < lim; s++) {
                    float w = sc[s] * inv;
                    const float *vv = v_base + (size_t)s * hd;
                    for (int d = 0; d < hd; d++) out[d] += w * vv[d];
                }
            }
#endif
        }
    }
    free(scores);
}

/* ============================================================
 * Forward pass — quantized weights + arena (Phase 2)
 *
 * Uses raw float* buffers from arena instead of tm_tensor.
 * Only the final logits output is a tm_tensor (returned to caller).
 * ============================================================ */

tm_tensor* pn_llama_forward(pn_model *model, uint32_t token, pn_state *state) {
    if (!model || !state || !model->weights) return NULL;

    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    if (!qw->is_quantized) {
        return pn_llama_forward_f32(model, token, state);
    }

    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    pn_arena *arena = state->arena;
    int pos = state->pos;
    int n_embd = cfg->n_embd;
    int head_dim = pn_head_dim(cfg);
    int n_kv_groups = cfg->n_head / cfg->n_kv_head;
    int max_seq = kv->max_seq;

#ifdef USE_THREADS
    fw_pool *pool = model->pool;
    /* Register this model's pool so pn_qmv parallelizes its row loop across
     * cores for every Q/K/V/O and gate/up/down projection in this token. */
    pn_qmv_set_pool(pool);
#endif

    if (!arena) return NULL;

    /* Reset arena for this forward call */
    pn_arena_reset(arena);

    /* 1. Token embedding lookup (or injected image embedding for VLM tokens) */
    float *hidden = pn_arena_alloc_f32(arena, n_embd);
    if (!hidden) return NULL;
    if (state->embd_override)
        memcpy(hidden, state->embd_override, n_embd * sizeof(float));
    else
        memcpy(hidden, qw->tok_embd + (size_t)token * n_embd, n_embd * sizeof(float));

    /* Pre-allocate attention scores buffer: [n_head, max_seq] */
    float *scores_buf = pn_arena_alloc_f32(arena, cfg->n_head * max_seq);
    if (!scores_buf) return NULL;

    /* 2. Transformer layers */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_qlayer *l = &qw->layers[layer];

        /* === Attention sublayer === */

        /* RMSNorm */
        float *normed = pn_arena_alloc_f32(arena, n_embd);
        if (!normed) return NULL;
        pn_vrms_norm_f32(hidden, l->attn_norm, normed, n_embd, cfg->rms_norm_eps);

        /* Q/K/V projections via quantized matvec */
        float *q = pn_arena_alloc_f32(arena, l->wq.rows);
        float *k = pn_arena_alloc_f32(arena, l->wk.rows);
        float *v = pn_arena_alloc_f32(arena, l->wv.rows);
        if (!q || !k || !v) return NULL;
        /* Q/K/V all read `normed`: fuse into one parallel dispatch. */
        {
            const pn_qweight *qkv_w[3]   = { &l->wq, &l->wk, &l->wv };
            const float      *qkv_vec[3] = { normed, normed, normed };
            float            *qkv_out[3] = { q, k, v };
            pn_qmv_group(qkv_w, qkv_vec, qkv_out, 3);
        }

        /* Optional biases (Qwen2) + per-head Q/K RMSNorm (Qwen3), before RoPE */
        pn_llama_qkv_finish(l, cfg, q, k, v);

        /* Apply RoPE to Q and K (NEOX for Qwen3's unpermuted GGUF weights) */
        if (cfg->arch == PN_ARCH_QWEN3) {
            apply_rope_neox(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            apply_rope_neox(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        } else {
            apply_rope_raw(q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            apply_rope_raw(k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        }

        /* Update KV cache (circular for sliding window) */
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

        attn_args args = {
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
            fw_for(pool, cfg->n_head, compute_attn_heads, &args);
        } else {
            compute_attn_heads(0, cfg->n_head, &args);
        }
#else
        compute_attn_heads(0, cfg->n_head, &args);
#endif

        /* Attention SHAPE capture — the post-softmax weights are still in scores_buf and
         * attn_out is not yet consumed, so this reads the arithmetic that just ran rather
         * than recomputing anything. One branch on a global when nothing is armed. */
        if (pn_attention_trace_armed()) {
            pn_attention_trace_layer cap = {
                .scores = scores_buf,
                .attn_out = attn_out,
                .v_cache = v_cache_data,
                .normed = normed,
                .n_head = cfg->n_head,
                .n_kv_groups = n_kv_groups,
                .head_dim = head_dim,
                .max_seq = max_seq,
                .seq_len = seq_len,
                .n_embd = n_embd,
            };

            pn_attention_trace_record(layer, &cap);
        }

        /* Output projection */
        float *attn_proj = pn_arena_alloc_f32(arena, n_embd);
        if (!attn_proj) return NULL;
        pn_qmv(&l->wo, attn_out, attn_proj);
        if (l->bo) pn_vadd_f32(attn_proj, l->bo, attn_proj, n_embd);

        /* Residual connection */
        pn_vadd_f32(hidden, attn_proj, hidden, n_embd);

        /* === FFN sublayer === */

        /* RMSNorm */
        float *ffn_normed = pn_arena_alloc_f32(arena, n_embd);
        if (!ffn_normed) return NULL;
        pn_vrms_norm_f32(hidden, l->ffn_norm, ffn_normed, n_embd, cfg->rms_norm_eps);

        if (cfg->n_experts > 0) {
            /* MoE routing */
            float *gate_logits = pn_arena_alloc_f32(arena, cfg->n_experts);
            if (!gate_logits) return NULL;
            pn_qmv(&l->ffn_gate_inp, ffn_normed, gate_logits);

            /* Softmax over gate logits */
            float max_gl = gate_logits[0];
            for (int e = 1; e < cfg->n_experts; e++)
                if (gate_logits[e] > max_gl) max_gl = gate_logits[e];
            float sum_exp_gl = 0;
            for (int e = 0; e < cfg->n_experts; e++) {
                gate_logits[e] = expf(gate_logits[e] - max_gl);
                sum_exp_gl += gate_logits[e];
            }
            for (int e = 0; e < cfg->n_experts; e++)
                gate_logits[e] /= sum_exp_gl;

            /* Top-k selection */
            int top_idx[8];
            float top_wt[8];
            for (int k = 0; k < cfg->n_experts_used; k++) {
                int best = -1;
                float best_val = -1;
                for (int e = 0; e < cfg->n_experts; e++) {
                    int skip = 0;
                    for (int j = 0; j < k; j++)
                        if (top_idx[j] == e) skip = 1;
                    if (!skip && gate_logits[e] > best_val) {
                        best = e;
                        best_val = gate_logits[e];
                    }
                }
                top_idx[k] = best;
                top_wt[k] = best_val;
            }

            /* Renormalize selected weights */
            float wt_sum = 0;
            for (int k = 0; k < cfg->n_experts_used; k++) wt_sum += top_wt[k];
            float inv_wt = 1.0f / wt_sum;
            for (int k = 0; k < cfg->n_experts_used; k++) top_wt[k] *= inv_wt;

            /* Run selected experts and accumulate */
            float *ffn_out = pn_arena_alloc_f32(arena, n_embd);
            if (!ffn_out) return NULL;
            memset(ffn_out, 0, n_embd * sizeof(float));
            float *exp_gate = pn_arena_alloc_f32(arena, cfg->n_ff);
            float *exp_up   = pn_arena_alloc_f32(arena, cfg->n_ff);
            float *exp_hid  = pn_arena_alloc_f32(arena, cfg->n_ff);
            float *exp_out  = pn_arena_alloc_f32(arena, n_embd);
            if (!exp_gate || !exp_up || !exp_hid || !exp_out) return NULL;

            for (int k = 0; k < cfg->n_experts_used; k++) {
                pn_moe_expert *expert = &l->experts[top_idx[k]];
                pn_qmv(&expert->w_gate, ffn_normed, exp_gate);
                pn_qmv(&expert->w_up,   ffn_normed, exp_up);
                pn_vsilu_mul_f32(exp_gate, exp_up, exp_hid, cfg->n_ff);
                pn_qmv(&expert->w_down, exp_hid, exp_out);
                for (int j = 0; j < n_embd; j++)
                    ffn_out[j] += top_wt[k] * exp_out[j];
            }

            pn_vadd_f32(hidden, ffn_out, hidden, n_embd);
        } else {
            /* Standard FFN (non-MoE) */
            float *gate = pn_arena_alloc_f32(arena, cfg->n_ff);
            float *up = pn_arena_alloc_f32(arena, cfg->n_ff);
            if (!gate || !up) return NULL;
            /* gate and up both read `ffn_normed`: fuse into one dispatch. */
            {
                const pn_qweight *gu_w[2]   = { &l->w_gate, &l->w_up };
                const float      *gu_vec[2] = { ffn_normed, ffn_normed };
                float            *gu_out[2] = { gate, up };
                pn_qmv_group(gu_w, gu_vec, gu_out, 2);
            }

            float *ffn_hidden = pn_arena_alloc_f32(arena, cfg->n_ff);
            if (!ffn_hidden) return NULL;
            pn_vsilu_mul_f32(gate, up, ffn_hidden, cfg->n_ff);

            float *ffn_out = pn_arena_alloc_f32(arena, n_embd);
            if (!ffn_out) return NULL;
            pn_qmv(&l->w_down, ffn_hidden, ffn_out);

            pn_vadd_f32(hidden, ffn_out, hidden, n_embd);
        }

        /* The residual stream as this layer leaves it — the state an accelerator arm has to
         * be compared against, which a per-op score never sees. Off unless armed. */
        pn_residual_trace_record(layer, hidden, n_embd);
    }

    /* 3. Final RMSNorm */
    float *final_normed = pn_arena_alloc_f32(arena, n_embd);
    if (!final_normed) return NULL;
    pn_vrms_norm_f32(hidden, qw->output_norm, final_normed, n_embd, cfg->rms_norm_eps);
    pn_residual_trace_record(cfg->n_layer, final_normed, n_embd);

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

/* ============================================================
 * Batched multi-token prefill (Phase 3)
 *
 * Processes P tokens through each layer with GEMM projections. Each quantized
 * weight is dequantized once per layer in L2-resident panels (via pn_qmm_prefill)
 * and multiplied against all P rows on AMX, instead of once per token.
 * Numerically equivalent to P sequential single-token forwards up to BLAS
 * summation order (the same reassociation the projection GEMMs already use).
 * ============================================================ */

#ifdef USE_METAL
/* GPU F16 prefill projection: out[P,rows] = normed[P,cols] @ dequant(w)^T using
 * the GPU-resident Q4_K weight (fm_gemm_q4_k_f16). Returns 1 if it ran on the
 * GPU, 0 if the caller must use the CPU dequant+GEMM path. */
static fm_buffer *g_prefill_x = NULL; static size_t g_prefill_x_cap = 0;
static fm_buffer *g_prefill_o = NULL; static size_t g_prefill_o_cap = 0;
static fm_buffer *prefill_stage(fm_buffer **b, size_t *cap, size_t bytes) {
    if (*b && *cap >= bytes) return *b;
    if (*b) fm_buffer_free(*b);
    *b = fm_buffer_alloc(bytes); *cap = *b ? bytes : 0; return *b;
}
static int prefill_gemm_gpu(const pn_qweight *w, const float *normed, float *out,
                            int P, int rows, int cols) {
    if (!w->gpu || w->type != GGUF_QUANT_Q4_K || !fm_available()) return 0;
    fm_buffer *x = prefill_stage(&g_prefill_x, &g_prefill_x_cap, (size_t)P * cols * sizeof(float));
    fm_buffer *o = prefill_stage(&g_prefill_o, &g_prefill_o_cap, (size_t)P * rows * sizeof(float));
    if (!x || !o) return 0;
    memcpy(fm_buffer_data(x), normed, (size_t)P * cols * sizeof(float));
    fm_gemm_q4_k_f16((fm_buffer *)w->gpu, x, o, P, rows, cols, (int)w->row_stride);
    fm_sync();
    memcpy(out, fm_buffer_data(o), (size_t)P * rows * sizeof(float));
    return 1;
}
#endif

/* A prefill projection GEMM. GPU F16 path when the weight is GPU-resident (opt-in
 * PN_GPU_PREFILL upload), else the CPU panel path: pn_qmm_prefill dequantizes the
 * weight in L2-resident panels and runs a per-panel cblas_sgemm on AMX, with no
 * full-weight F32 DRAM scratch. Output layout is [P, rows] (out[p*rows + r]),
 * matching the [B, rows] contract of pn_qmm. */
static void prefill_project(const pn_qweight *w, const float *normed, float *out,
                            int P, int rows, int cols) {
#ifdef USE_METAL
    if (prefill_gemm_gpu(w, normed, out, P, rows, cols)) return;
#else
    (void)rows; (void)cols;
#endif
    pn_qmm_prefill(w, normed, P, out);
}

/* ============================================================
 * Batched prefill helpers: per-row work parallelized over the row range.
 * Each range_fn performs the identical per-element arithmetic on a disjoint
 * subset of rows, so the parallel result is bit-for-bit identical to the
 * serial loop it replaces.
 * ============================================================ */

/* NEOX RoPE applied from a precomputed [P, rot/2] cos/sin table. This computes
 * the SAME rotation as apply_rope_neox: cos_row[i] == cosf(pos*freq_i) and
 * sin_row[i] == sinf(pos*freq_i) with freq_i == 1/powf(theta,(2i)/rot), and the
 * same pairing (dim i with i+half). Reusing the cached cosf/sinf float bits
 * makes the output bit-for-bit identical to the per-row recompute. */
static void apply_rope_neox_tab(float *data, int n_heads, int head_dim,
                                int rot, const float *cos_row, const float *sin_row) {
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float cos_a = cos_row[i], sin_a = sin_row[i];
            float x0 = head[i];
            float x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* Row-parallel RMSNorm: dst[p] = rmsnorm(src[p], weight). */
typedef struct {
    const float *src; const float *weight; float *dst;
    int n_embd; float eps;
} prefill_rmsnorm_args;
static void prefill_rmsnorm_range(int start, int end, void *arg) {
    const prefill_rmsnorm_args *a = (const prefill_rmsnorm_args *)arg;
    for (int p = start; p < end; p++)
        pn_vrms_norm_f32(a->src + (size_t)p * a->n_embd, a->weight,
                         a->dst + (size_t)p * a->n_embd, a->n_embd, a->eps);
}

/* Row-parallel SiLU-multiply: out[p] = silu(gate[p]) * up[p]. */
typedef struct {
    const float *gate; const float *up; float *out; int n_ff;
} prefill_silu_args;
static void prefill_silu_range(int start, int end, void *arg) {
    const prefill_silu_args *a = (const prefill_silu_args *)arg;
    for (int p = start; p < end; p++)
        pn_vsilu_mul_f32(a->gate + (size_t)p * a->n_ff, a->up + (size_t)p * a->n_ff,
                         a->out + (size_t)p * a->n_ff, a->n_ff);
}

/* Row-parallel residual add: hidden[p] += delta[p], applying an optional bias
 * to delta first (the attention-output projection bias). */
typedef struct {
    float *hidden; float *delta; const float *bias; int n_embd;
} prefill_residual_args;
static void prefill_residual_range(int start, int end, void *arg) {
    const prefill_residual_args *a = (const prefill_residual_args *)arg;
    for (int p = start; p < end; p++) {
        float *d = a->delta + (size_t)p * a->n_embd;
        if (a->bias) pn_vadd_f32(d, a->bias, d, a->n_embd);
        pn_vadd_f32(a->hidden + (size_t)p * a->n_embd, d,
                    a->hidden + (size_t)p * a->n_embd, a->n_embd);
    }
}

/* Row-parallel attention pre-processing: QKV bias, optional QK-norm, RoPE
 * (table-driven NEOX for Qwen3, raw fallback otherwise), then KV-cache fill.
 * Rows are independent, so per-row identical arithmetic is bitwise identical. */
typedef struct {
    float *q_buf, *k_buf, *v_buf;
    const float *bq, *bk, *bv;
    const float *attn_q_norm, *attn_k_norm;
    const float *cos_tab, *sin_tab;   /* [P, rot/2]; NULL when not NEOX */
    float *k_cache_data, *v_cache_data;
    int qdim, kdim, vdim, head_dim, n_head, n_kv_head, rot, half;
    int arch, rope_dim, base_pos, max_seq;
    float rms_eps, rope_theta;
} prefill_attn_pre_args;
static void prefill_attn_pre_range(int start, int end, void *arg) {
    const prefill_attn_pre_args *a = (const prefill_attn_pre_args *)arg;
    for (int p = start; p < end; p++) {
        int pos = a->base_pos + p;
        float *q = a->q_buf + (size_t)p * a->qdim;
        float *k = a->k_buf + (size_t)p * a->kdim;
        float *v = a->v_buf + (size_t)p * a->vdim;

        if (a->bq) pn_vadd_f32(q, a->bq, q, a->qdim);
        if (a->bk) pn_vadd_f32(k, a->bk, k, a->kdim);
        if (a->bv) pn_vadd_f32(v, a->bv, v, a->vdim);

        if (a->attn_q_norm) {
            for (int h = 0; h < a->n_head; h++)
                pn_vrms_norm_f32(q + (size_t)h * a->head_dim, a->attn_q_norm,
                                 q + (size_t)h * a->head_dim, a->head_dim, a->rms_eps);
        }
        if (a->attn_k_norm) {
            for (int h = 0; h < a->n_kv_head; h++)
                pn_vrms_norm_f32(k + (size_t)h * a->head_dim, a->attn_k_norm,
                                 k + (size_t)h * a->head_dim, a->head_dim, a->rms_eps);
        }

        if (a->arch == PN_ARCH_QWEN3) {
            const float *cos_row = a->cos_tab + (size_t)p * a->half;
            const float *sin_row = a->sin_tab + (size_t)p * a->half;
            apply_rope_neox_tab(q, a->n_head, a->head_dim, a->rot, cos_row, sin_row);
            apply_rope_neox_tab(k, a->n_kv_head, a->head_dim, a->rot, cos_row, sin_row);
        } else {
            apply_rope_raw(q, a->n_head, a->head_dim, a->rope_dim, pos, a->rope_theta);
            apply_rope_raw(k, a->n_kv_head, a->head_dim, a->rope_dim, pos, a->rope_theta);
        }

        int cache_pos = pos % a->max_seq;
        for (int h = 0; h < a->n_kv_head; h++) {
            memcpy(a->k_cache_data + (size_t)h * a->max_seq * a->head_dim + (size_t)cache_pos * a->head_dim,
                   k + (size_t)h * a->head_dim, a->head_dim * sizeof(float));
            memcpy(a->v_cache_data + (size_t)h * a->max_seq * a->head_dim + (size_t)cache_pos * a->head_dim,
                   v + (size_t)h * a->head_dim, a->head_dim * sizeof(float));
        }
    }
}

/* Dispatch a per-row range fn over all P rows: use the pool when one exists and
 * P clears a small threshold, otherwise run it inline. */
#ifdef USE_THREADS
#define PN_PREFILL_ROW_PAR_MIN 8
#define PREFILL_FOR_ROWS(fn, arg) \
    do { if (pool && P >= PN_PREFILL_ROW_PAR_MIN) fw_for(pool, P, (fn), (arg)); \
         else (fn)(0, P, (arg)); } while (0)
#else
#define PREFILL_FOR_ROWS(fn, arg) (fn)(0, P, (arg))
#endif

/* Shared batched forward over P tokens of ONE sequence starting at state->pos.
 * This is the common engine behind pn_llama_prefill (prompt ingest) and
 * pn_llama_verify (speculative-decode verify): identical layer stack, identical
 * kernels, so the numerics are the same. The tail differs only in what logits
 * are produced and whether the position is committed:
 *
 *   all_logits != NULL -> compute the final RMSNorm + LM head for ALL P rows
 *                         into all_logits[P*n_vocab] (row-major). Used by verify.
 *   out_last  != NULL  -> also (or instead) return the LAST row's logits as an
 *                         owned tm_tensor via *out_last. Used by prefill.
 *   commit_pos != 0    -> advance state->pos by P (prefill). Verify passes 0 so
 *                         the K/V just written stays uncommitted (free rollback).
 *
 * Returns 0 on success, -1 on unsupported model / allocation failure (state left
 * untouched in that case). */
static int llama_batch_forward_seq(pn_model *model, const uint32_t *tokens,
                                   const float *const *embd_rows, int n_tokens,
                                   pn_state *state, tm_tensor **out_last,
                                   float *all_logits, int commit_pos) {
    if (out_last) *out_last = NULL;
    if (!model || !state || !model->weights || n_tokens <= 0) return -1;

    /* Only the quantized, non-MoE Llama/Qwen forward is supported here. */
    if (model->forward != pn_llama_forward) return -1;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    if (!qw->is_quantized) return -1;

    pn_config *cfg = &model->config;
    if (cfg->n_experts > 0) return -1;  /* MoE: fall back to per-token */

    /* Reject ternary / unsupported quant up front (no state mutation). */
    pn_qweight *probe[] = { &qw->layers[0].wq, &qw->layers[0].wk,
                            &qw->layers[0].wv, &qw->layers[0].wo,
                            &qw->layers[0].w_gate, &qw->layers[0].w_up,
                            &qw->layers[0].w_down };
    for (size_t i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
        switch (probe[i]->type) {
            case GGUF_QUANT_F32: case GGUF_QUANT_F16:
            case GGUF_QUANT_Q4_0: case GGUF_QUANT_Q8_0:
            case GGUF_QUANT_Q3_K: case GGUF_QUANT_Q4_K:
            case GGUF_QUANT_Q5_K: case GGUF_QUANT_Q6_K:
                break;
            default:
                return -1;
        }
    }

#ifdef USE_METAL
    fm_init();  /* idempotent; enables the Metal GEMM path if a device exists */
    /* Opt-in (PN_GPU_PREFILL): upload Q4_K decoder weights and run prefill's
     * projection GEMMs as F16 MPS matmuls on the GPU. OFF by default: measured
     * a net LOSS for bulk OCR, because (a) prefill is attention-bound (the O(P^2)
     * decoder attention over ~1369 image tokens dominates, not the GEMMs, so the
     * GEMM speedup is only ~1.3x) and (b) it contends with Core ML vision for the
     * single GPU, breaking the CPU-compute / GPU-vision overlap that makes bulk
     * fast. Useful only when vision runs on the CPU (GPU otherwise idle). */
    static int gpu_prefill_uploaded = 0;
    if (!gpu_prefill_uploaded && fm_available() && getenv("PN_GPU_PREFILL")) {
        for (int layer = 0; layer < model->config.n_layer; layer++) {
            pn_llama_qlayer *l = &qw->layers[layer];
            pn_qmv_gpu_upload(&l->wq); pn_qmv_gpu_upload(&l->wk); pn_qmv_gpu_upload(&l->wv);
            pn_qmv_gpu_upload(&l->wo); pn_qmv_gpu_upload(&l->w_gate);
            pn_qmv_gpu_upload(&l->w_up); pn_qmv_gpu_upload(&l->w_down);
        }
        gpu_prefill_uploaded = 1;
    }
#endif

    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_arena *arena = state->arena;
    int P = n_tokens;
    int base_pos = state->pos;
    int n_embd = cfg->n_embd;
    int head_dim = pn_head_dim(cfg);
    int n_head = cfg->n_head;
    int n_kv_head = cfg->n_kv_head;
    int n_kv_groups = n_head / n_kv_head;
    int n_ff = cfg->n_ff;
    int max_seq = kv->max_seq;
    int qdim = qw->layers[0].wq.rows;   /* n_head * head_dim */
    int kdim = qw->layers[0].wk.rows;   /* n_kv_head * head_dim */
    int vdim = qw->layers[0].wv.rows;

    /* Batched activation buffers (malloc'd; the per-token arena is sized for
     * a single token and may be too small for P rows). They scale with the
     * whole prompt, so the host must be able to hold them first. */
    uint64_t batch_bytes = ((uint64_t)P * (4ull * n_embd + 2ull * qdim + kdim + vdim +
                                           3ull * n_ff) +
                            (uint64_t)n_head * max_seq) * sizeof(float);
    if (batch_bytes >= PN_HOST_ADMIT_THRESHOLD_BYTES &&
        pn_host_admit_check(batch_bytes, "batched prefill buffers", "shorten the "
                            "prompt or free host memory", NULL, 0) != 0)
        return -1;
    float *hidden     = malloc((size_t)P * n_embd * sizeof(float));
    float *normed     = malloc((size_t)P * n_embd * sizeof(float));
    float *q_buf      = malloc((size_t)P * qdim   * sizeof(float));
    float *k_buf      = malloc((size_t)P * kdim   * sizeof(float));
    float *v_buf      = malloc((size_t)P * vdim   * sizeof(float));
    float *attn_out   = malloc((size_t)P * qdim   * sizeof(float));
    float *attn_proj  = malloc((size_t)P * n_embd * sizeof(float));
    float *gate_buf   = malloc((size_t)P * n_ff   * sizeof(float));
    float *up_buf      = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_hidden = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_out    = malloc((size_t)P * n_embd * sizeof(float));
    float *scores_buf = malloc((size_t)n_head * max_seq * sizeof(float));

    if (!hidden || !normed || !q_buf || !k_buf || !v_buf || !attn_out ||
        !attn_proj || !gate_buf || !up_buf || !ffn_hidden || !ffn_out ||
        !scores_buf) {
        free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
        free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
        free(ffn_hidden); free(ffn_out); free(scores_buf);
        return -1;
    }

    /* Precompute the NEOX RoPE cos/sin table once for this prefill (Change 1):
     * cos/sin of pos*freq_i for every row and every rotary pair. freq_i depends
     * only on i and the angle only on (pos, i), so this [P, rot/2] table is
     * reused across all layers and all heads instead of recomputing powf/cosf/
     * sinf per (layer, row, head, i). The values are computed exactly as
     * apply_rope_neox does, so the cached float bits keep the result identical. */
    int rope_rot  = (cfg->rope_dim > 0 && cfg->rope_dim <= head_dim) ? cfg->rope_dim : head_dim;
    int rope_half = rope_rot / 2;
    float *rope_cos = NULL, *rope_sin = NULL;
    if (cfg->arch == PN_ARCH_QWEN3) {
        rope_cos = malloc((size_t)P * rope_half * sizeof(float));
        rope_sin = malloc((size_t)P * rope_half * sizeof(float));
        float *rope_freq = malloc((size_t)rope_half * sizeof(float));
        if (!rope_cos || !rope_sin || !rope_freq) {
            free(rope_cos); free(rope_sin); free(rope_freq);
            free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
            free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
            free(ffn_hidden); free(ffn_out); free(scores_buf);
            return -1;
        }
        for (int i = 0; i < rope_half; i++)
            rope_freq[i] = 1.0f / powf(cfg->rope_theta, (float)(2 * i) / (float)rope_rot);
        for (int p = 0; p < P; p++) {
            int pos = base_pos + p;
            float *cr = rope_cos + (size_t)p * rope_half;
            float *sr = rope_sin + (size_t)p * rope_half;
            for (int i = 0; i < rope_half; i++) {
                float angle = (float)pos * rope_freq[i];
                cr[i] = cosf(angle);
                sr[i] = sinf(angle);
            }
        }
        free(rope_freq);
    }

#ifdef USE_THREADS
    fw_pool *pool = model->pool;
#endif

    /* 1. Build the [P, n_embd] input embedding matrix. */
    for (int p = 0; p < P; p++) {
        float *dst = hidden + (size_t)p * n_embd;
        if (embd_rows && embd_rows[p])
            memcpy(dst, embd_rows[p], n_embd * sizeof(float));
        else
            memcpy(dst, qw->tok_embd + (size_t)tokens[p] * n_embd,
                   n_embd * sizeof(float));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_qlayer *l = &qw->layers[layer];

        /* --- Attention --- */

        /* RMSNorm each row. */
        prefill_rmsnorm_args attn_rn = {
            .src = hidden, .weight = l->attn_norm, .dst = normed,
            .n_embd = n_embd, .eps = cfg->rms_norm_eps,
        };
        PREFILL_FOR_ROWS(prefill_rmsnorm_range, &attn_rn);

        /* Q/K/V projections (dequant once, then GEMM). */
        prefill_project(&l->wq, normed, q_buf, P, qdim, n_embd);
        prefill_project(&l->wk, normed, k_buf, P, kdim, n_embd);
        prefill_project(&l->wv, normed, v_buf, P, vdim, n_embd);

        /* Per-row post-processing: biases, QK-norm, RoPE, KV-cache fill. */
        float *k_cache_data = (float *)kv->k_cache[layer]->data;
        float *v_cache_data = (float *)kv->v_cache[layer]->data;

        prefill_attn_pre_args pre = {
            .q_buf = q_buf, .k_buf = k_buf, .v_buf = v_buf,
            .bq = l->bq, .bk = l->bk, .bv = l->bv,
            .attn_q_norm = l->attn_q_norm, .attn_k_norm = l->attn_k_norm,
            .cos_tab = rope_cos, .sin_tab = rope_sin,
            .k_cache_data = k_cache_data, .v_cache_data = v_cache_data,
            .qdim = qdim, .kdim = kdim, .vdim = vdim, .head_dim = head_dim,
            .n_head = n_head, .n_kv_head = n_kv_head, .rot = rope_rot, .half = rope_half,
            .arch = cfg->arch, .rope_dim = cfg->rope_dim,
            .base_pos = base_pos, .max_seq = max_seq,
            .rms_eps = cfg->rms_norm_eps, .rope_theta = cfg->rope_theta,
        };
        PREFILL_FOR_ROWS(prefill_attn_pre_range, &pre);

        /* Causal attention (KV cache now holds all P keys). Blocked "flash" path
         * when keys are contiguous (no sliding-window wraparound) — reads K/V
         * once per query block instead of once per query; else per-query. */
        if (base_pos + P <= max_seq) {
            pf_attn_args fa = {
                .q = q_buf, .k_cache = k_cache_data, .v_cache = v_cache_data,
                .out = attn_out, .P = P, .base_pos = base_pos, .head_dim = head_dim,
                .n_head = n_head, .n_kv_groups = n_kv_groups, .max_seq = max_seq,
                .scale = scale,
            };
#ifdef USE_THREADS
            if (pool && n_head >= 4) fw_for(pool, n_head, prefill_flash_range, &fa);
            else prefill_flash_range(0, n_head, &fa);
#else
            prefill_flash_range(0, n_head, &fa);
#endif
        } else {
            for (int p = 0; p < P; p++) {
                int pos = base_pos + p;
                int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
                attn_args args = {
                    .q_data = q_buf + (size_t)p * qdim,
                    .k_cache_data = k_cache_data,
                    .v_cache_data = v_cache_data,
                    .attn_out = attn_out + (size_t)p * qdim,
                    .scores_buf = scores_buf,
                    .head_dim = head_dim,
                    .n_kv_groups = n_kv_groups,
                    .max_seq = max_seq,
                    .seq_len = seq_len,
                    .scale = scale,
                };
#ifdef USE_THREADS
                if (pool && n_head >= 4) fw_for(pool, n_head, compute_attn_heads, &args);
                else compute_attn_heads(0, n_head, &args);
#else
                compute_attn_heads(0, n_head, &args);
#endif
            }
        }

        /* Output projection + residual. */
        prefill_project(&l->wo, attn_out, attn_proj, P, n_embd, qdim);
        prefill_residual_args attn_res = {
            .hidden = hidden, .delta = attn_proj, .bias = l->bo, .n_embd = n_embd,
        };
        PREFILL_FOR_ROWS(prefill_residual_range, &attn_res);

        /* --- FFN --- */
        prefill_rmsnorm_args ffn_rn = {
            .src = hidden, .weight = l->ffn_norm, .dst = normed,
            .n_embd = n_embd, .eps = cfg->rms_norm_eps,
        };
        PREFILL_FOR_ROWS(prefill_rmsnorm_range, &ffn_rn);

        prefill_project(&l->w_gate, normed, gate_buf, P, n_ff, n_embd);
        prefill_project(&l->w_up, normed, up_buf, P, n_ff, n_embd);

        prefill_silu_args silu = {
            .gate = gate_buf, .up = up_buf, .out = ffn_hidden, .n_ff = n_ff,
        };
        PREFILL_FOR_ROWS(prefill_silu_range, &silu);

        prefill_project(&l->w_down, ffn_hidden, ffn_out, P, n_embd, n_ff);

        prefill_residual_args ffn_res = {
            .hidden = hidden, .delta = ffn_out, .bias = NULL, .n_embd = n_embd,
        };
        PREFILL_FOR_ROWS(prefill_residual_range, &ffn_res);
    }

    /* 3. Final RMSNorm + LM head. Two shapes share the same normed hidden state:
     *    - all_logits: every row (verify needs a prediction per candidate). The
     *      final RMSNorm reuses the [P, n_embd] `normed` scratch (free after the
     *      layer loop), then ONE M=P LM-head GEMM, exactly as forward_batch does.
     *    - out_last: only the last row (prefill's returned logits are for the
     *      last token), matching the sequential path via a single pn_qmv. */
    int rc = 0;
    if (all_logits) {
        for (int p = 0; p < P; p++)
            pn_vrms_norm_f32(hidden + (size_t)p * n_embd, qw->output_norm,
                             normed + (size_t)p * n_embd, n_embd, cfg->rms_norm_eps);
#ifdef USE_BLAS
        if (qw->output.type == GGUF_QUANT_F32) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, P, cfg->n_vocab, n_embd,
                        1.0f, normed, n_embd, (const float *)qw->output.data, n_embd,
                        0.0f, all_logits, cfg->n_vocab);
        } else
#endif
        {
            pn_qmm(&qw->output, normed, P, all_logits);
        }
    }
    if (out_last) {
        float *last_hidden = hidden + (size_t)(P - 1) * n_embd;
        float *final_normed = arena ? pn_arena_alloc_f32(arena, n_embd) : NULL;
        float *fn = final_normed ? final_normed : normed;  /* reuse normed if no arena */
        pn_vrms_norm_f32(last_hidden, qw->output_norm, fn, n_embd, cfg->rms_norm_eps);

        float *logits_raw = malloc((size_t)cfg->n_vocab * sizeof(float));
        if (!logits_raw) {
            rc = -1;
        } else {
            pn_qmv(&qw->output, fn, logits_raw);
            int64_t logits_shape[] = { cfg->n_vocab };
            tm_tensor *logits_t = tm_from_data(logits_raw, logits_shape, 1, TM_F32);
            *out_last = logits_t ? tm_clone(logits_t) : NULL;
            if (logits_t) tm_free(logits_t);
            free(logits_raw);
        }
    }

    free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
    free(ffn_hidden); free(ffn_out); free(scores_buf);
    free(rope_cos); free(rope_sin);

    /* Commit the position only for the prefill path. Verify leaves pos untouched
     * so the K/V just written into slots pos..pos+P-1 stays uncommitted, giving
     * free rollback of rejected drafts. */
    if (commit_pos) state->pos = base_pos + P;
    return rc;
}

#undef PREFILL_FOR_ROWS
#undef PN_PREFILL_ROW_PAR_MIN

/* Batched multi-token prefill: thin wrapper over the shared core that returns
 * the last token's logits and commits state->pos (see header contract). */
tm_tensor* pn_llama_prefill(pn_model *model, const uint32_t *tokens,
                            const float *const *embd_rows, int n_tokens,
                            pn_state *state) {
    tm_tensor *last = NULL;
    if (llama_batch_forward_seq(model, tokens, embd_rows, n_tokens, state,
                                &last, NULL, /*commit_pos=*/1) != 0)
        return NULL;
    return last;
}

/* Speculative-decode verify pass: run the same batched forward over the P
 * candidate tokens, emit per-row logits into out_logits, and DO NOT advance pos
 * (see header contract). out_logits must hold n_tokens * n_vocab floats. */
int pn_llama_verify(pn_model *model, const uint32_t *tokens,
                    const float *const *embd_rows, int n_tokens,
                    pn_state *state, float *out_logits) {
    if (!out_logits) return -1;
    return llama_batch_forward_seq(model, tokens, embd_rows, n_tokens, state,
                                   NULL, out_logits, /*commit_pos=*/0);
}

int pn_llama_forward_batch(pn_model *model, const uint32_t *tokens,
                           const float *const *embd_rows, pn_state **states,
                           int batch, float *out_logits) {
    if (!model || !states || !out_logits || batch <= 0) return -1;
    if (!model->weights) return -1;

    /* Same support envelope as pn_llama_prefill. */
    if (model->forward != pn_llama_forward) return -1;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    if (!qw->is_quantized) return -1;
    pn_config *cfg = &model->config;
    if (cfg->n_experts > 0) return -1;

    pn_qweight *probe[] = { &qw->layers[0].wq, &qw->layers[0].wk,
                            &qw->layers[0].wv, &qw->layers[0].wo,
                            &qw->layers[0].w_gate, &qw->layers[0].w_up,
                            &qw->layers[0].w_down };
    for (size_t i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
        switch (probe[i]->type) {
            case GGUF_QUANT_F32: case GGUF_QUANT_F16:
            case GGUF_QUANT_Q4_0: case GGUF_QUANT_Q8_0:
            case GGUF_QUANT_Q3_K: case GGUF_QUANT_Q4_K:
            case GGUF_QUANT_Q5_K: case GGUF_QUANT_Q6_K:
                break;
            default:
                return -1;
        }
    }

    /* Every sequence must carry its own KV cache. */
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if (!states[b] || !states[b]->kv_cache) return -1;
        pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[b]->kv_cache;
        if (kvb->max_seq > max_seq) max_seq = kvb->max_seq;
    }

#ifdef USE_METAL
    fm_init();
    /* Opt-in (PN_BATCH_GPU): upload the Q4_K decoder weights to the GPU once so
     * pn_qmm runs the batched GEMM on the GPU. Non-Q4_K weights (e.g. Q6_K
     * output) stay on the CPU AMX path — pn_qmm dispatches per weight. */
    static int gpu_batch_uploaded = 0;
    if (!gpu_batch_uploaded && batch <= 32 && getenv("PN_BATCH_GPU") && fm_available()) {
        for (int layer = 0; layer < cfg->n_layer; layer++) {
            pn_llama_qlayer *l = &qw->layers[layer];
            pn_qmv_gpu_upload(&l->wq); pn_qmv_gpu_upload(&l->wk);
            pn_qmv_gpu_upload(&l->wv); pn_qmv_gpu_upload(&l->wo);
            pn_qmv_gpu_upload(&l->w_gate); pn_qmv_gpu_upload(&l->w_up);
            pn_qmv_gpu_upload(&l->w_down);
        }
        pn_qmm_set_gpu_decode(1);   /* PN_BATCH_GPU experiment: decode on GPU */
        gpu_batch_uploaded = 1;
    }
#endif

    int P = batch;                       /* one row per sequence */
    int n_embd = cfg->n_embd;
    int head_dim = pn_head_dim(cfg);
    int n_head = cfg->n_head;
    int n_kv_head = cfg->n_kv_head;
    int n_kv_groups = n_head / n_kv_head;
    int n_ff = cfg->n_ff;
    int qdim = qw->layers[0].wq.rows;
    int kdim = qw->layers[0].wk.rows;
    int vdim = qw->layers[0].wv.rows;

    float *hidden     = malloc((size_t)P * n_embd * sizeof(float));
    float *normed     = malloc((size_t)P * n_embd * sizeof(float));
    float *q_buf      = malloc((size_t)P * qdim   * sizeof(float));
    float *k_buf      = malloc((size_t)P * kdim   * sizeof(float));
    float *v_buf      = malloc((size_t)P * vdim   * sizeof(float));
    float *attn_out   = malloc((size_t)P * qdim   * sizeof(float));
    float *attn_proj  = malloc((size_t)P * n_embd * sizeof(float));
    float *gate_buf   = malloc((size_t)P * n_ff   * sizeof(float));
    float *up_buf     = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_hidden = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_out    = malloc((size_t)P * n_embd * sizeof(float));
    float *scores_buf = malloc((size_t)n_head * max_seq * sizeof(float));
    float *final_normed = malloc((size_t)P * n_embd * sizeof(float));

    if (!hidden || !normed || !q_buf || !k_buf || !v_buf || !attn_out ||
        !attn_proj || !gate_buf || !up_buf || !ffn_hidden || !ffn_out ||
        !scores_buf || !final_normed) {
        free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
        free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
        free(ffn_hidden); free(ffn_out); free(scores_buf);
        free(final_normed);
        return -1;
    }

#ifdef USE_THREADS
    fw_pool *pool = model->pool;
#endif

    /* 1. Input embedding matrix [P, n_embd], one row per sequence. */
    for (int b = 0; b < P; b++) {
        float *dst = hidden + (size_t)b * n_embd;
        if (embd_rows && embd_rows[b])
            memcpy(dst, embd_rows[b], n_embd * sizeof(float));
        else
            memcpy(dst, qw->tok_embd + (size_t)tokens[b] * n_embd,
                   n_embd * sizeof(float));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. Projections/FFN are M=P GEMMs (weight dequant'd
     *    once, applied to all P sequences); only the KV write + attention are
     *    per-sequence, since each sequence owns its cache and position. */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_qlayer *l = &qw->layers[layer];

        for (int b = 0; b < P; b++)
            pn_vrms_norm_f32(hidden + (size_t)b * n_embd, l->attn_norm,
                             normed + (size_t)b * n_embd, n_embd, cfg->rms_norm_eps);

        /* Fused M=P projections: dequant each weight row once, dot against all
         * P sequences (quantized weight read once, no F32 DRAM copy). */
        pn_qmm(&l->wq, normed, P, q_buf);
        pn_qmm(&l->wk, normed, P, k_buf);
        pn_qmm(&l->wv, normed, P, v_buf);

        /* Per-sequence post-proc: bias, QK-norm, RoPE, KV write into that
         * sequence's own cache at its own position. */
        for (int b = 0; b < P; b++) {
            pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[b]->kv_cache;
            int pos = states[b]->pos;
            int b_max_seq = kvb->max_seq;
            float *k_cache_data = (float *)kvb->k_cache[layer]->data;
            float *v_cache_data = (float *)kvb->v_cache[layer]->data;
            float *q = q_buf + (size_t)b * qdim;
            float *k = k_buf + (size_t)b * kdim;
            float *v = v_buf + (size_t)b * vdim;

            if (l->bq) pn_vadd_f32(q, l->bq, q, qdim);
            if (l->bk) pn_vadd_f32(k, l->bk, k, kdim);
            if (l->bv) pn_vadd_f32(v, l->bv, v, vdim);

            if (l->attn_q_norm) {
                for (int h = 0; h < n_head; h++)
                    pn_vrms_norm_f32(q + (size_t)h * head_dim, l->attn_q_norm,
                                     q + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
            }
            if (l->attn_k_norm) {
                for (int h = 0; h < n_kv_head; h++)
                    pn_vrms_norm_f32(k + (size_t)h * head_dim, l->attn_k_norm,
                                     k + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
            }

            if (cfg->arch == PN_ARCH_QWEN3) {
                apply_rope_neox(q, n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
                apply_rope_neox(k, n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            } else {
                apply_rope_raw(q, n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
                apply_rope_raw(k, n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            }

            int cache_pos = pos % b_max_seq;
            for (int h = 0; h < n_kv_head; h++) {
                memcpy(k_cache_data + (size_t)h * b_max_seq * head_dim + (size_t)cache_pos * head_dim,
                       k + (size_t)h * head_dim, head_dim * sizeof(float));
                memcpy(v_cache_data + (size_t)h * b_max_seq * head_dim + (size_t)cache_pos * head_dim,
                       v + (size_t)h * head_dim, head_dim * sizeof(float));
            }
        }

        /* Attention: each sequence attends only its own cache. */
        for (int b = 0; b < P; b++) {
            pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[b]->kv_cache;
            int pos = states[b]->pos;
            int b_max_seq = kvb->max_seq;
            int seq_len = (pos + 1 < b_max_seq) ? pos + 1 : b_max_seq;
            attn_args args = {
                .q_data = q_buf + (size_t)b * qdim,
                .k_cache_data = (float *)kvb->k_cache[layer]->data,
                .v_cache_data = (float *)kvb->v_cache[layer]->data,
                .attn_out = attn_out + (size_t)b * qdim,
                .scores_buf = scores_buf,
                .head_dim = head_dim,
                .n_kv_groups = n_kv_groups,
                .max_seq = b_max_seq,
                .seq_len = seq_len,
                .scale = scale,
            };
#ifdef USE_THREADS
            if (pool && n_head >= 4) fw_for(pool, n_head, compute_attn_heads, &args);
            else compute_attn_heads(0, n_head, &args);
#else
            compute_attn_heads(0, n_head, &args);
#endif
        }

        pn_qmm(&l->wo, attn_out, P, attn_proj);
        for (int b = 0; b < P; b++) {
            float *ap = attn_proj + (size_t)b * n_embd;
            if (l->bo) pn_vadd_f32(ap, l->bo, ap, n_embd);
            pn_vadd_f32(hidden + (size_t)b * n_embd, ap, hidden + (size_t)b * n_embd, n_embd);
        }

        for (int b = 0; b < P; b++)
            pn_vrms_norm_f32(hidden + (size_t)b * n_embd, l->ffn_norm,
                             normed + (size_t)b * n_embd, n_embd, cfg->rms_norm_eps);

        pn_qmm(&l->w_gate, normed, P, gate_buf);
        pn_qmm(&l->w_up, normed, P, up_buf);

        for (int b = 0; b < P; b++)
            pn_vsilu_mul_f32(gate_buf + (size_t)b * n_ff, up_buf + (size_t)b * n_ff,
                             ffn_hidden + (size_t)b * n_ff, n_ff);

        pn_qmm(&l->w_down, ffn_hidden, P, ffn_out);

        for (int b = 0; b < P; b++)
            pn_vadd_f32(hidden + (size_t)b * n_embd, ffn_out + (size_t)b * n_embd,
                        hidden + (size_t)b * n_embd, n_embd);
    }

    /* 3. Final RMSNorm (all rows), then the LM head as ONE M=P GEMM so the large
     *    output/embedding weight is read once instead of once per row (for tied
     *    embeddings it is a ~600 MB F32 matrix, so per-row was B x that traffic). */
    for (int b = 0; b < P; b++)
        pn_vrms_norm_f32(hidden + (size_t)b * n_embd, qw->output_norm,
                         final_normed + (size_t)b * n_embd, n_embd, cfg->rms_norm_eps);
#ifdef USE_BLAS
    if (qw->output.type == GGUF_QUANT_F32) {
        /* F32 tied-embedding head: GEMM straight off the weight, no dequant. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, P, cfg->n_vocab, n_embd,
                    1.0f, final_normed, n_embd, (const float *)qw->output.data, n_embd,
                    0.0f, out_logits, cfg->n_vocab);
    } else
#endif
    {
        pn_qmm(&qw->output, final_normed, P, out_logits);
    }

    for (int b = 0; b < P; b++) states[b]->pos += 1;

    free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
    free(ffn_hidden); free(ffn_out); free(scores_buf);
    free(final_normed);
    return 0;
}

/* Ragged batched speculative-decode verify. This is pn_llama_forward_batch's
 * body with P = M (M = sum of the per-sequence block lengths), specialized so a
 * row's logits are bit-identical to running that one token through
 * forward_batch on its own sequence:
 *
 *   - Projections / FFN / LM head are M=P pn_qmm calls, exactly as forward_batch.
 *     pn_qmm dequantizes each weight row once and dots it against every
 *     activation row independently, so an output row does not depend on M or the
 *     other rows (this is what makes it bitwise-equal to forward_batch, which
 *     uses the same pn_qmm). NOT pn_qmm_prefill and NOT the F32 BLAS LM head,
 *     whose reassociated summation order would diverge.
 *   - Attention pre-processing (bias, QK-norm, RoPE at each row's true position,
 *     KV write into that row's OWN cache) and per-query causal attention are the
 *     one ragged part, driven by row via row_seq[]/row_pos[]. Uses
 *     apply_rope_neox/apply_rope_raw directly (the same calls forward_batch
 *     uses) and per-query compute_attn_heads (NOT prefill_flash_range, whose
 *     cblas summation order differs from forward_batch). Row r's attention
 *     seq_len = states[b]->pos + (r-off[b]) + 1 IS the causal mask: it sees its
 *     committed prefix plus its own block's earlier rows, nothing after and
 *     nothing from another sequence.
 *
 * No state->pos is advanced: rejected drafts roll back for free. */
int pn_llama_verify_batch(pn_model *model, const uint32_t *tokens,
                          pn_state **states, const int *n_tok, int n_seqs,
                          float *out_logits) {
    if (!model || !tokens || !states || !n_tok || !out_logits || n_seqs <= 0)
        return -1;
    if (!model->weights) return -1;

    /* Same support envelope as pn_llama_forward_batch. */
    if (model->forward != pn_llama_forward) return -1;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    if (!qw->is_quantized) return -1;
    pn_config *cfg = &model->config;
    if (cfg->n_experts > 0) return -1;

    pn_qweight *probe[] = { &qw->layers[0].wq, &qw->layers[0].wk,
                            &qw->layers[0].wv, &qw->layers[0].wo,
                            &qw->layers[0].w_gate, &qw->layers[0].w_up,
                            &qw->layers[0].w_down };
    for (size_t i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
        switch (probe[i]->type) {
            case GGUF_QUANT_F32: case GGUF_QUANT_F16:
            case GGUF_QUANT_Q4_0: case GGUF_QUANT_Q8_0:
            case GGUF_QUANT_Q3_K: case GGUF_QUANT_Q4_K:
            case GGUF_QUANT_Q5_K: case GGUF_QUANT_Q6_K:
                break;
            default:
                return -1;
        }
    }

    /* Ragged layout: off[b] is block b's first row, M the total rows. Reject any
     * block with a non-positive length, a missing cache, or a cache wraparound
     * (states[b]->pos + n_tok[b] > that state's max_seq) — the no-wrap guarantee
     * is what lets cache_pos == row_pos and keeps the committed prefix contiguous
     * below the block. All checks run before any state mutation. */
    int M = 0, max_seq = 0;
    for (int b = 0; b < n_seqs; b++) {
        if (n_tok[b] <= 0) return -1;
        if (!states[b] || !states[b]->kv_cache) return -1;
        pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[b]->kv_cache;
        if (states[b]->pos + n_tok[b] > kvb->max_seq) return -1;
        if (kvb->max_seq > max_seq) max_seq = kvb->max_seq;
        M += n_tok[b];
    }
    if (M > PN_QMM_MAX_B) return -1;   /* the LM-head pn_qmm caps at PN_QMM_MAX_B */

    int *off     = malloc((size_t)n_seqs * sizeof(int));
    int *row_seq = malloc((size_t)M * sizeof(int));
    int *row_pos = malloc((size_t)M * sizeof(int));
    if (!off || !row_seq || !row_pos) { free(off); free(row_seq); free(row_pos); return -1; }
    {
        int r = 0;
        for (int b = 0; b < n_seqs; b++) {
            off[b] = r;
            for (int j = 0; j < n_tok[b]; j++) {
                row_seq[r] = b;
                row_pos[r] = states[b]->pos + j;   /* this row's true position */
                r++;
            }
        }
    }

    int P = M;                           /* one row per candidate token */
    int n_embd = cfg->n_embd;
    int head_dim = pn_head_dim(cfg);
    int n_head = cfg->n_head;
    int n_kv_head = cfg->n_kv_head;
    int n_kv_groups = n_head / n_kv_head;
    int n_ff = cfg->n_ff;
    int qdim = qw->layers[0].wq.rows;
    int kdim = qw->layers[0].wk.rows;
    int vdim = qw->layers[0].wv.rows;

    float *hidden     = malloc((size_t)P * n_embd * sizeof(float));
    float *normed     = malloc((size_t)P * n_embd * sizeof(float));
    float *q_buf      = malloc((size_t)P * qdim   * sizeof(float));
    float *k_buf      = malloc((size_t)P * kdim   * sizeof(float));
    float *v_buf      = malloc((size_t)P * vdim   * sizeof(float));
    float *attn_out   = malloc((size_t)P * qdim   * sizeof(float));
    float *attn_proj  = malloc((size_t)P * n_embd * sizeof(float));
    float *gate_buf   = malloc((size_t)P * n_ff   * sizeof(float));
    float *up_buf     = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_hidden = malloc((size_t)P * n_ff   * sizeof(float));
    float *ffn_out    = malloc((size_t)P * n_embd * sizeof(float));
    float *scores_buf = malloc((size_t)n_head * max_seq * sizeof(float));
    float *final_normed = malloc((size_t)P * n_embd * sizeof(float));

    if (!hidden || !normed || !q_buf || !k_buf || !v_buf || !attn_out ||
        !attn_proj || !gate_buf || !up_buf || !ffn_hidden || !ffn_out ||
        !scores_buf || !final_normed) {
        free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
        free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
        free(ffn_hidden); free(ffn_out); free(scores_buf);
        free(final_normed); free(off); free(row_seq); free(row_pos);
        return -1;
    }

#ifdef USE_THREADS
    fw_pool *pool = model->pool;
#endif

    /* 1. Input embedding matrix [M, n_embd], one row per candidate token. */
    for (int r = 0; r < P; r++)
        memcpy(hidden + (size_t)r * n_embd,
               qw->tok_embd + (size_t)tokens[r] * n_embd, n_embd * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. Projections/FFN are M=P pn_qmm GEMMs (identical to
     *    forward_batch); only the KV write + attention are ragged per row. */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_qlayer *l = &qw->layers[layer];

        for (int r = 0; r < P; r++)
            pn_vrms_norm_f32(hidden + (size_t)r * n_embd, l->attn_norm,
                             normed + (size_t)r * n_embd, n_embd, cfg->rms_norm_eps);

        pn_qmm(&l->wq, normed, P, q_buf);
        pn_qmm(&l->wk, normed, P, k_buf);
        pn_qmm(&l->wv, normed, P, v_buf);

        /* Per-row post-proc: bias, QK-norm, RoPE at row_pos[r], KV write into
         * row_seq[r]'s own cache at cache_pos = row_pos[r] % that cache max_seq. */
        for (int r = 0; r < P; r++) {
            pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[row_seq[r]]->kv_cache;
            int pos = row_pos[r];
            int b_max_seq = kvb->max_seq;
            float *k_cache_data = (float *)kvb->k_cache[layer]->data;
            float *v_cache_data = (float *)kvb->v_cache[layer]->data;
            float *q = q_buf + (size_t)r * qdim;
            float *k = k_buf + (size_t)r * kdim;
            float *v = v_buf + (size_t)r * vdim;

            if (l->bq) pn_vadd_f32(q, l->bq, q, qdim);
            if (l->bk) pn_vadd_f32(k, l->bk, k, kdim);
            if (l->bv) pn_vadd_f32(v, l->bv, v, vdim);

            if (l->attn_q_norm) {
                for (int h = 0; h < n_head; h++)
                    pn_vrms_norm_f32(q + (size_t)h * head_dim, l->attn_q_norm,
                                     q + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
            }
            if (l->attn_k_norm) {
                for (int h = 0; h < n_kv_head; h++)
                    pn_vrms_norm_f32(k + (size_t)h * head_dim, l->attn_k_norm,
                                     k + (size_t)h * head_dim, head_dim, cfg->rms_norm_eps);
            }

            if (cfg->arch == PN_ARCH_QWEN3) {
                apply_rope_neox(q, n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
                apply_rope_neox(k, n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            } else {
                apply_rope_raw(q, n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
                apply_rope_raw(k, n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            }

            int cache_pos = pos % b_max_seq;
            for (int h = 0; h < n_kv_head; h++) {
                memcpy(k_cache_data + (size_t)h * b_max_seq * head_dim + (size_t)cache_pos * head_dim,
                       k + (size_t)h * head_dim, head_dim * sizeof(float));
                memcpy(v_cache_data + (size_t)h * b_max_seq * head_dim + (size_t)cache_pos * head_dim,
                       v + (size_t)h * head_dim, head_dim * sizeof(float));
            }
        }

        /* Attention: row r attends its own cache with seq_len = row_pos[r] + 1
         * (committed prefix + this block's rows up to and including r). */
        for (int r = 0; r < P; r++) {
            pn_llama_kv_cache *kvb = (pn_llama_kv_cache *)states[row_seq[r]]->kv_cache;
            int b_max_seq = kvb->max_seq;
            int seq_len = row_pos[r] + 1;   /* no-wrap guaranteed: <= b_max_seq */
            attn_args args = {
                .q_data = q_buf + (size_t)r * qdim,
                .k_cache_data = (float *)kvb->k_cache[layer]->data,
                .v_cache_data = (float *)kvb->v_cache[layer]->data,
                .attn_out = attn_out + (size_t)r * qdim,
                .scores_buf = scores_buf,
                .head_dim = head_dim,
                .n_kv_groups = n_kv_groups,
                .max_seq = b_max_seq,
                .seq_len = seq_len,
                .scale = scale,
            };
#ifdef USE_THREADS
            if (pool && n_head >= 4) fw_for(pool, n_head, compute_attn_heads, &args);
            else compute_attn_heads(0, n_head, &args);
#else
            compute_attn_heads(0, n_head, &args);
#endif
        }

        pn_qmm(&l->wo, attn_out, P, attn_proj);
        for (int r = 0; r < P; r++) {
            float *ap = attn_proj + (size_t)r * n_embd;
            if (l->bo) pn_vadd_f32(ap, l->bo, ap, n_embd);
            pn_vadd_f32(hidden + (size_t)r * n_embd, ap, hidden + (size_t)r * n_embd, n_embd);
        }

        for (int r = 0; r < P; r++)
            pn_vrms_norm_f32(hidden + (size_t)r * n_embd, l->ffn_norm,
                             normed + (size_t)r * n_embd, n_embd, cfg->rms_norm_eps);

        pn_qmm(&l->w_gate, normed, P, gate_buf);
        pn_qmm(&l->w_up, normed, P, up_buf);

        for (int r = 0; r < P; r++)
            pn_vsilu_mul_f32(gate_buf + (size_t)r * n_ff, up_buf + (size_t)r * n_ff,
                             ffn_hidden + (size_t)r * n_ff, n_ff);

        pn_qmm(&l->w_down, ffn_hidden, P, ffn_out);

        for (int r = 0; r < P; r++)
            pn_vadd_f32(hidden + (size_t)r * n_embd, ffn_out + (size_t)r * n_embd,
                        hidden + (size_t)r * n_embd, n_embd);
    }

    /* 3. Final RMSNorm (all rows) + LM head as ONE M=P pn_qmm, so row r's logits
     *    equal forward_batch's for that token. Always pn_qmm (never the F32 BLAS
     *    branch): row-independence is the bit-identity guarantee. */
    for (int r = 0; r < P; r++)
        pn_vrms_norm_f32(hidden + (size_t)r * n_embd, qw->output_norm,
                         final_normed + (size_t)r * n_embd, n_embd, cfg->rms_norm_eps);
    pn_qmm(&qw->output, final_normed, P, out_logits);

    /* No pos commit: the caller advances states[b]->pos over accepted rows. */

    free(hidden); free(normed); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(attn_proj); free(gate_buf); free(up_buf);
    free(ffn_hidden); free(ffn_out); free(scores_buf);
    free(final_normed); free(off); free(row_seq); free(row_pos);
    return 0;
}

/* ============================================================
 * Forward pass — F32 weights (Phase 1 legacy)
 * ============================================================ */

static void apply_rope(tm_tensor *t, int pos, int head_dim, int rope_dim, float theta_base) {
    if (!t) return;

    int n_heads = (int)t->shape[0];
    float *data = (float *)t->data;
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;

    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < rot; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[i];
            float x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

tm_tensor* pn_llama_forward_f32(pn_model *model, uint32_t token, pn_state *state) {
    if (!model || !state || !model->weights) return NULL;

    pn_llama_weights *w = (pn_llama_weights *)model->weights;
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    int pos = state->pos;
    int head_dim = cfg->n_embd / cfg->n_head;
    int n_kv_groups = cfg->n_head / cfg->n_kv_head;

    /* 1. Token embedding lookup */
    int64_t tok_shape[] = { 1 };
    int32_t tok_val = (int32_t)token;
    tm_tensor *tok_idx = tm_from_data(&tok_val, tok_shape, 1, TM_I32);
    tm_tensor *hidden = tm_embedding(w->tok_embd, tok_idx);
    tm_free(tok_idx);

    tm_tensor *h = tm_squeeze(hidden, 0);
    tm_free(hidden);
    hidden = h;

    /* 2. Transformer layers */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        pn_llama_layer *l = &w->layers[layer];

        /* Attention */
        tm_tensor *normed = tm_rms_norm(hidden, l->attn_norm, cfg->rms_norm_eps);

        tm_tensor *q = tm_mv(l->wq, normed);
        tm_tensor *k = tm_mv(l->wk, normed);
        tm_tensor *v = tm_mv(l->wv, normed);
        tm_free(normed);

        int64_t q_shape[] = { cfg->n_head, head_dim };
        int64_t kv_shape[] = { cfg->n_kv_head, head_dim };

        tm_tensor *q_heads = tm_reshape(q, q_shape, 2);
        tm_tensor *k_heads = tm_reshape(k, kv_shape, 2);
        tm_tensor *v_heads = tm_reshape(v, kv_shape, 2);
        tm_free(q); tm_free(k); tm_free(v);

        apply_rope(q_heads, pos, head_dim, cfg->rope_dim, cfg->rope_theta);
        apply_rope(k_heads, pos, head_dim, cfg->rope_dim, cfg->rope_theta);

        float *k_cache_data = (float *)kv->k_cache[layer]->data;
        float *v_cache_data = (float *)kv->v_cache[layer]->data;
        float *k_data = (float *)k_heads->data;
        float *v_data = (float *)v_heads->data;
        int max_seq_f32 = kv->max_seq;
        int cache_pos = pos % max_seq_f32;

        for (int hh = 0; hh < cfg->n_kv_head; hh++) {
            memcpy(k_cache_data + hh * max_seq_f32 * head_dim + cache_pos * head_dim,
                   k_data + hh * head_dim,
                   head_dim * sizeof(float));
            memcpy(v_cache_data + hh * max_seq_f32 * head_dim + cache_pos * head_dim,
                   v_data + hh * head_dim,
                   head_dim * sizeof(float));
        }
        tm_free(k_heads);
        tm_free(v_heads);

        int seq_len = (pos + 1 < max_seq_f32) ? pos + 1 : max_seq_f32;
        float *attn_out = calloc(cfg->n_head * head_dim, sizeof(float));
        float *q_data_ptr = (float *)q_heads->data;

        for (int qh = 0; qh < cfg->n_head; qh++) {
            int kv_h = qh / n_kv_groups;
            float *q_vec = q_data_ptr + qh * head_dim;
            float *k_base = k_cache_data + kv_h * kv->max_seq * head_dim;
            float *v_base = v_cache_data + kv_h * kv->max_seq * head_dim;

            float *scores = calloc(seq_len, sizeof(float));
            float scale = 1.0f / sqrtf((float)head_dim);

            for (int s = 0; s < seq_len; s++) {
                float dot = 0.0f;
                float *k_vec = k_base + s * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[s] = dot * scale;
            }

            float max_score = scores[0];
            for (int s = 1; s < seq_len; s++) {
                if (scores[s] > max_score) max_score = scores[s];
            }
            float sum_exp = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                scores[s] = expf(scores[s] - max_score);
                sum_exp += scores[s];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int s = 0; s < seq_len; s++) {
                scores[s] *= inv_sum;
            }

            float *out_vec = attn_out + qh * head_dim;
            for (int s = 0; s < seq_len; s++) {
                float w_s = scores[s];
                float *v_vec = v_base + s * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    out_vec[d] += w_s * v_vec[d];
                }
            }

            free(scores);
        }
        tm_free(q_heads);

        int64_t attn_shape[] = { cfg->n_head * head_dim };
        tm_tensor *attn_vec = tm_from_data(attn_out, attn_shape, 1, TM_F32);
        tm_tensor *attn_proj = tm_mv(l->wo, attn_vec);
        tm_free(attn_vec);
        free(attn_out);

        tm_tensor *h2 = tm_add(hidden, attn_proj);
        tm_free(hidden);
        tm_free(attn_proj);
        hidden = h2;

        /* FFN */
        normed = tm_rms_norm(hidden, l->ffn_norm, cfg->rms_norm_eps);

        tm_tensor *gate = tm_mv(l->w_gate, normed);
        tm_tensor *up   = tm_mv(l->w_up, normed);
        tm_free(normed);

        tm_tensor *gate_act = tm_silu(gate);
        tm_free(gate);

        tm_tensor *ffn_hidden = tm_mul(gate_act, up);
        tm_free(gate_act);
        tm_free(up);

        tm_tensor *ffn_out = tm_mv(l->w_down, ffn_hidden);
        tm_free(ffn_hidden);

        h2 = tm_add(hidden, ffn_out);
        tm_free(hidden);
        tm_free(ffn_out);
        hidden = h2;
    }

    /* 3. Final RMSNorm */
    tm_tensor *normed = tm_rms_norm(hidden, w->output_norm, cfg->rms_norm_eps);
    tm_free(hidden);

    /* 4. LM head */
    tm_tensor *logits = tm_mv(w->output, normed);
    tm_free(normed);

    state->pos++;
    return logits;
}
