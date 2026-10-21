/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_PHI3_H
#define PN_PHI3_H

#include "../../include/pantheon.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "../formats/gguf.h"
#include "llama.h"  /* Reuses pn_llama_kv_cache */

/*
 * Phi-3 model architecture.
 *
 * Covers: Phi-3-mini (3.8B), Phi-3-small, Phi-3-medium.
 *
 * Same compute as Llama (RMSNorm, RoPE, MHA/GQA, SwiGLU) but different
 * GGUF weight layout:
 *   - Fused QKV: blk.N.attn_qkv.weight [3*n_embd, n_embd]
 *   - Fused gate+up: blk.N.ffn_up.weight [2*n_ff, n_embd]
 *   - Partial RoPE: rope.dimension_count may be < head_dim
 */

/* ============================================================
 * Quantized weight structs
 * ============================================================ */

typedef struct {
    float *attn_norm;           /* RMSNorm weight [n_embd] (F32, owned) */
    pn_qweight attn_qkv;       /* FUSED Q+K+V [3*n_embd, n_embd] (quantized, mmap) */
    pn_qweight attn_output;    /* Output projection [n_embd, n_embd] (quantized, mmap) */
    float *ffn_norm;            /* RMSNorm weight [n_embd] (F32, owned) */
    pn_qweight ffn_gate_up;    /* FUSED gate+up [2*n_ff, n_embd] (quantized, mmap) */
    pn_qweight ffn_down;       /* Down projection [n_embd, n_ff] (quantized, mmap) */
} pn_phi3_qlayer;

typedef struct {
    int is_quantized;           /* Always 1, used to distinguish weight type */
    float *tok_embd;            /* Token embeddings [n_vocab * n_embd] (F32, owned) */
    float *output_norm;         /* Final RMSNorm [n_embd] (F32, owned) */
    pn_qweight output;          /* LM head (quantized or F32) */
    float *output_f32;          /* LM head F32 if tied embeddings (owned) */
    pn_phi3_qlayer *layers;     /* Per-layer quantized weights [n_layer] */
    int n_layers;
    int n_embd;
    int n_vocab;
    pn_gguf_file *gguf;         /* Kept open (mmap backing) */
} pn_phi3_qweights;

/* ============================================================
 * Forward pass
 * ============================================================ */

tm_tensor* pn_phi3_forward(pn_model *model, uint32_t token, pn_state *state);

/* ============================================================
 * Weight loading
 * ============================================================ */

pn_phi3_qweights* pn_phi3_load_gguf_quantized(pn_gguf_file *gguf, pn_config *config);

/* Free */
void pn_phi3_qweights_free(pn_phi3_qweights *qw);

#endif /* PN_PHI3_H */
