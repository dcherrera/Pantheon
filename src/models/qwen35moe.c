/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe.c
 * @brief Qwen3.5-MoE (Qwen3-Next lineage) loader for Pantheon. Implements A2:
 *        config parsing, per-layer type map, block-struct definitions, GGUF
 *        weight loading, and a stub forward. Hybrid Gated-Delta-Net SSM +
 *        sparse full-attention + shared-expert MoE.
 *
 * @status A6 complete: the forward orchestration + dual-state lifecycle live in
 *         qwen35moe_forward.c; this file keeps the model struct + GGUF load and
 *         exposes internal accessors (config, per-type counts, index maps,
 *         per-type weight arrays, embed/norm/head) for that file. A2-A5 kernels
 *         load via the GGUF/pn_qweight path; F32 norms/biases dequantized+owned.
 * @issues None
 * @todo
 *   - [ ] A8: validate logits vs the llama.cpp oracle (numeric correctness).
 */

/* 1. This module's own header */
#include "qwen35moe.h"

/* 2. Standard library */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 4. Project headers */
#include "../core/qweight.h"
#include "../formats/gguf_dequant.h"
#include "qwen35moe_internal.h"
#include "qwen35moe_kv_q4.h"

/* The config, per-layer-type enum, and block structs (pn_qwen35moe_config,
 * pn_qwen35moe_moe, pn_qwen35moe_ssm_layer, pn_qwen35moe_attn_layer) live in
 * qwen35moe_internal.h so the per-block compute files can share them. */

/* ============================================================
 * Model (opaque)
 *
 * Mirrors rwkvx.c's dual per-type arrays: ssm_layers[] and attn_layers[] are
 * indexed by per-type index; block_type[] + block_index[] map a global layer
 * index to its type and per-type slot.
 * ============================================================ */

struct pn_qwen35moe_model {
    pn_qwen35moe_config cfg;

    /* Model-level tensors */
    float *token_embd;          /* [n_vocab * n_embd] (F32, owned) */
    float *output_norm;         /* final RMSNorm [n_embd] (F32, owned) */
    pn_qweight output;          /* LM head */
    float *output_f32;          /* LM head F32 if tied embeddings (owned) */

    /* Per-type block arrays */
    pn_qwen35moe_ssm_layer *ssm_layers;
    pn_qwen35moe_attn_layer *attn_layers;
    int n_ssm;
    int n_attn;

    /* Global layer index -> (type, per-type slot) */
    pn_qwen35moe_layer_type *block_type;  /* [n_layer] */
    int *block_index;                     /* [n_layer] per-type index */

    pn_gguf_file *gguf;         /* kept open (mmap backing) */

    /* GPU residency handle (opaque pn_qwen35moe_gpu, defined in
     * qwen35moe_forward_cuda.c). NULL when CUDA is unavailable or the model
     * was not uploaded. Held here so the single-die GPU forward can find its
     * persistent fc_buffer weights without re-uploading per token. The CPU
     * weights above stay live so the CPU path is never broken. */
    void *gpu;

    /* Explicit die placement (pn_load_on, Phase 7). n_gpu_dies == 0 means "no
     * placement given" — the GPU upload falls back to the QWEN35_GPU_DIE_A/B
     * env knobs / defaults. When set, placement wins over env. */
    int gpu_dies[8];
    int n_gpu_dies;

    /* Where a DETACHED accelerator matvec seam is parked. This family's accelerator is a
     * hook on cfg.mv rather than a separate forward, so a parity harness needs somewhere to
     * put it while it runs the CPU oracle in the same process.
     * See pn_qwen35moe_model_accel_mv_enable(). */
    pn_qwen35moe_mv mv_detached;
};

/* ============================================================
 * Config reading
 * ============================================================ */

static int qwen35moe_read_config(pn_gguf_file *gguf, pn_qwen35moe_config *qc,
                                 pn_config *config)
{
    /* Two GGUF families share this loader: Nex-N2-mini tags
     * general.architecture="qwen35moe" (routed MoE), PrismML Bonsai-27B tags
     * "qwen35" (dense SwiGLU FFN). The metadata keys are prefixed by the arch
     * string, so pick the prefix from what the file declares. */
    const char *arch = pn_gguf_get_str(gguf, "general.architecture");
    const char *p = (arch && strcmp(arch, "qwen35") == 0) ? "qwen35" : "qwen35moe";
    char key[128];

    snprintf(key, sizeof(key), "%s.block_count", p);
    qc->n_layer = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.full_attention_interval", p);
    qc->full_attention_interval = (int)pn_gguf_get_u32(gguf, key, 4);
    snprintf(key, sizeof(key), "%s.embedding_length", p);
    qc->n_embd = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.attention.head_count", p);
    qc->n_head = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.attention.head_count_kv", p);
    qc->n_head_kv = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.attention.key_length", p);
    qc->head_dim = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.rope.dimension_count", p);
    qc->rope_dim = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.rope.freq_base", p);
    qc->rope_theta = pn_gguf_get_f32(gguf, key, 10000.0f);

    /* Multi-section (IMRoPE) split. A2 didn't parse this — A3 does. For a
     * text-only model every section shares the single linear token position,
     * so the sections only matter for image/video position ids; the attention
     * kernel collapses multi-section RoPE to standard NEOX. We still parse and
     * log the array so the field is populated for A6 / multimodal variants.
     * Stored innermost-first as INT32; default split [rope_dim,0,0,0]. */
    qc->rope_sections[0] = qc->rope_dim;
    qc->rope_sections[1] = 0;
    qc->rope_sections[2] = 0;
    qc->rope_sections[3] = 0;
    snprintf(key, sizeof(key), "%s.rope.dimension_sections", p);
    pn_gguf_arr *sections = pn_gguf_get_arr(gguf, key);
    if (sections && sections->n > 0 && sections->elem_type == GGUF_TYPE_INT32) {
        const int32_t *vals = (const int32_t *)sections->data;
        int n = sections->n < 4 ? (int)sections->n : 4;
        int sum = 0;
        for (int s = 0; s < n; s++) {
            qc->rope_sections[s] = (int)vals[s];
            sum += qc->rope_sections[s];
        }
        fprintf(stderr,
                "qwen35moe: rope.dimension_sections = [%d %d %d %d] (sum=%d, "
                "rope_dim=%d) — text path uses standard NEOX\n",
                qc->rope_sections[0], qc->rope_sections[1],
                qc->rope_sections[2], qc->rope_sections[3], sum, qc->rope_dim);
    }
    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", p);
    qc->rms_norm_eps = pn_gguf_get_f32(gguf, key, 1e-6f);
    snprintf(key, sizeof(key), "%s.expert_count", p);
    qc->n_expert = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.expert_used_count", p);
    qc->n_expert_used = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.expert_feed_forward_length", p);
    qc->expert_ff = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", p);
    qc->shared_ff = (int)pn_gguf_get_u32(gguf, key, 0);
    /* Routed-expert weight scale. Absent in the Qwen3.5-MoE GGUFs seen so far
     * (gating is softmax + norm_topk_prob, no extra scale), so it defaults to
     * 1.0; honored if a future conversion emits the key. */
    snprintf(key, sizeof(key), "%s.expert_weights_scale", p);
    qc->expert_weights_scale = pn_gguf_get_f32(gguf, key, 1.0f);

    /* Dense-FFN (Bonsai) detection: no routed experts + merged expert tensors
     * absent + a plain blk.0.ffn_gate present. When dense, feed_forward_length
     * gives the SwiGLU inner dim and we drive the FFN through the shared-expert
     * slots (shared_ff := ffn_length). */
    qc->dense_ffn = 0;
    qc->ffn_length = 0;
    if (qc->n_expert == 0 &&
        pn_gguf_find_tensor(gguf, "blk.0.ffn_gate_exps.weight") == NULL &&
        pn_gguf_find_tensor(gguf, "blk.0.ffn_gate.weight") != NULL) {
        qc->dense_ffn = 1;
        snprintf(key, sizeof(key), "%s.feed_forward_length", p);
        qc->ffn_length = (int)pn_gguf_get_u32(gguf, key, 0);
        if (qc->ffn_length == 0) {
            /* Fall back to the actual tensor shape (rows = ffn dim). */
            pn_gguf_tensor_info *g0 = pn_gguf_find_tensor(gguf, "blk.0.ffn_gate.weight");
            if (g0 && g0->n_dims >= 2)
                qc->ffn_length = (int)g0->dims[g0->n_dims - 1];
        }
        qc->shared_ff = qc->ffn_length;  /* dense FFN reuses the shared-expert path */
    }

    snprintf(key, sizeof(key), "%s.ssm.state_size", p);
    qc->ssm_state = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.ssm.group_count", p);
    qc->ssm_n_group = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.ssm.time_step_rank", p);
    qc->ssm_dt_rank = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.ssm.inner_size", p);
    qc->ssm_d_inner = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.ssm.conv_kernel", p);
    qc->ssm_conv = (int)pn_gguf_get_u32(gguf, key, 0);
    snprintf(key, sizeof(key), "%s.context_length", p);
    qc->n_ctx = (int)pn_gguf_get_u32(gguf, key, 2048);

    /* Derived: conv runs over the full q|k|v stream =
     * key_dim*n_head_kv*2 (q,k) + value_dim*... — actually key*2 + value where
     * key = head_dim*n_head_kv*... ; per the design doc this is the attn_qkv
     * row count. Prefer the actual tensor shape over arithmetic. */
    pn_gguf_tensor_info *qkv0 = pn_gguf_find_tensor(gguf, "blk.0.attn_qkv.weight");
    if (qkv0 && qkv0->n_dims >= 2)
        qc->conv_dim = (int)qkv0->dims[qkv0->n_dims - 1];
    else
        qc->conv_dim = qc->n_embd * 2 + qc->ssm_d_inner;

    /* Vocab from the token embedding shape (outermost dim). */
    pn_gguf_tensor_info *embd = pn_gguf_find_tensor(gguf, "token_embd.weight");
    if (embd && embd->n_dims >= 2)
        qc->n_vocab = (int)embd->dims[embd->n_dims - 1];

    /* Multi-Token-Prediction (MTP) head skip. Qwen3.6-35B-A3B ships an extra
     * trailing block (block_count=41: 40 transformer layers + 1 MTP head). The
     * MTP block is only used for speculative draft during training and lacks the
     * full hybrid-layer tensor set (no blk.N.ssm_conv1d) - the standard
     * autoregressive forward never touches it. If the last block is missing its
     * SSM conv while an earlier block has one, treat it as an MTP head and drop
     * it from the layer count. Additive: Nex-N2 (40 layers, every block a real
     * hybrid layer) never trips this, so it is a no-op there. */
    if (qc->n_layer >= 2) {
        char mtp_key[64];
        snprintf(mtp_key, sizeof(mtp_key), "blk.%d.nextn.eh_proj.weight", qc->n_layer - 1);
        if (pn_gguf_find_tensor(gguf, mtp_key) != NULL) {
            fprintf(stderr, "qwen35moe: dropping trailing MTP/NextN head (block %d, has "
                            "nextn.eh_proj); loading %d transformer layers\n",
                            qc->n_layer - 1, qc->n_layer - 1);
            qc->n_layer -= 1;
        }
    }

    if (qc->n_embd == 0 || qc->n_head == 0 || qc->n_layer == 0) {
        fprintf(stderr, "qwen35moe: invalid config (n_embd=%d n_head=%d n_layer=%d)\n",
                qc->n_embd, qc->n_head, qc->n_layer);
        return -1;
    }
    if (qc->n_head_kv == 0)
        qc->n_head_kv = qc->n_head;

    /* KV Phase 1: carry the per-model context cap set by the engine (pn_load_on
     * from the load opts) INTO the internal config, so the KV ring is sized per
     * model. config->ctx_cap is set before this call; 0 = unset (env fallback). */
    qc->ctx_cap = config->ctx_cap;

    /* Mirror the shared subset into the engine config. */
    config->arch = PN_ARCH_QWEN35MOE;
    config->modality = PN_MODALITY_TEXT;
    config->n_vocab = qc->n_vocab;
    config->n_embd = qc->n_embd;
    config->n_head = qc->n_head;
    config->n_kv_head = qc->n_head_kv;
    config->n_layer = qc->n_layer;
    config->n_ff = qc->dense_ffn ? qc->ffn_length : qc->expert_ff;
    config->n_ctx = qc->n_ctx;
    config->rope_theta = qc->rope_theta;
    config->rms_norm_eps = qc->rms_norm_eps;
    config->rope_dim = qc->rope_dim;
    config->n_experts = qc->n_expert;
    config->n_experts_used = qc->n_expert_used;

    fprintf(stderr,
            "qwen35moe: n_layer=%d interval=%d n_embd=%d n_head=%d/%d head_dim=%d "
            "rope_dim=%d\n",
            qc->n_layer, qc->full_attention_interval, qc->n_embd, qc->n_head,
            qc->n_head_kv, qc->head_dim, qc->rope_dim);
    if (qc->dense_ffn)
        fprintf(stderr,
                "qwen35moe: DENSE FFN (Bonsai, arch=%s) ffn_length=%d conv_dim=%d "
                "n_vocab=%d\n",
                p, qc->ffn_length, qc->conv_dim, qc->n_vocab);
    else
        fprintf(stderr,
                "qwen35moe: experts=%d top-%d expert_ff=%d shared_ff=%d conv_dim=%d "
                "n_vocab=%d\n",
                qc->n_expert, qc->n_expert_used, qc->expert_ff, qc->shared_ff,
                qc->conv_dim, qc->n_vocab);
    fprintf(stderr,
            "qwen35moe: ssm state=%d n_group=%d dt_rank=%d d_inner=%d conv=%d\n",
            qc->ssm_state, qc->ssm_n_group, qc->ssm_dt_rank, qc->ssm_d_inner,
            qc->ssm_conv);
    return 0;
}

/* True when global layer i is a full-attention layer. */
static int qwen35moe_is_attn_layer(const pn_qwen35moe_config *qc, int i)
{
    int interval = qc->full_attention_interval > 0 ? qc->full_attention_interval : 4;
    return ((i + 1) % interval) == 0;
}

/* ============================================================
 * Loading helpers
 * ============================================================ */

/* Dequantize a named tensor to an owned F32 buffer. Logs + returns NULL on miss. */
static float *qwen35moe_load_f32(pn_gguf_file *gguf, const char *name)
{
    float *out = pn_gguf_dequantize_tensor(gguf, name, NULL);
    if (!out)
        fprintf(stderr, "qwen35moe: failed to load %s\n", name);
    return out;
}

/* Load the shared-expert MoE block for one layer. Routed experts are sliced
 * from the merged 3D *_exps tensors (zero-copy). Returns 0 on success. */
static int qwen35moe_load_moe(pn_gguf_file *gguf, const pn_qwen35moe_config *qc,
                              int i, pn_qwen35moe_moe *moe)
{
    char name[128];

    /* Dense-FFN (Bonsai): a single SwiGLU per layer. Load ffn_gate/up/down into
     * the shared-expert slots (the shared-expert code path IS a dense SwiGLU
     * minus the routed experts + scalar gate) and leave the router/expert
     * pointers NULL so the FFN block skips routing entirely. */
    if (qc->dense_ffn) {
        moe->ffn_gate_inp = (pn_qweight){0};
        moe->gate_exps = NULL;
        moe->up_exps = NULL;
        moe->down_exps = NULL;
        moe->ffn_gate_inp_shexp = (pn_qweight){0};
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        if (pn_qweight_from_gguf(&moe->ffn_gate_shexp, gguf, name) < 0)
            return -1;
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        if (pn_qweight_from_gguf(&moe->ffn_up_shexp, gguf, name) < 0)
            return -1;
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        if (pn_qweight_from_gguf(&moe->ffn_down_shexp, gguf, name) < 0)
            return -1;
        return 0;
    }

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", i);
    if (pn_qweight_from_gguf(&moe->ffn_gate_inp, gguf, name) < 0)
        return -1;

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", i);
    pn_gguf_tensor_info *gate = pn_gguf_find_tensor(gguf, name);
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", i);
    pn_gguf_tensor_info *up = pn_gguf_find_tensor(gguf, name);
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", i);
    pn_gguf_tensor_info *down = pn_gguf_find_tensor(gguf, name);
    if (!gate || !up || !down) {
        fprintf(stderr, "qwen35moe: layer %d missing merged expert tensors\n", i);
        return -1;
    }

    int n_exp = qc->n_expert;
    moe->gate_exps = calloc((size_t)n_exp, sizeof(*moe->gate_exps));
    moe->up_exps = calloc((size_t)n_exp, sizeof(*moe->up_exps));
    moe->down_exps = calloc((size_t)n_exp, sizeof(*moe->down_exps));
    if (!moe->gate_exps || !moe->up_exps || !moe->down_exps)
        return -1;

    /* gate/up: GGUF dims [n_embd, expert_ff, n_expert]; per expert
     * rows=expert_ff cols=n_embd. down: [expert_ff, n_embd, n_expert];
     * per expert rows=n_embd cols=expert_ff. (mirrors llama.c MoE slicing) */
    size_t gate_bpb = pn_gguf_bytes_per_block(gate->type);
    int gate_bs = pn_gguf_block_size(gate->type);
    size_t down_bpb = pn_gguf_bytes_per_block(down->type);
    int down_bs = pn_gguf_block_size(down->type);

    int gate_rows = qc->expert_ff, gate_cols = qc->n_embd;
    size_t gate_row_stride = ((size_t)gate_cols / (size_t)gate_bs) * gate_bpb;
    size_t gate_expert_stride = (size_t)gate_rows * gate_row_stride;
    int down_rows = qc->n_embd, down_cols = qc->expert_ff;
    size_t down_row_stride = ((size_t)down_cols / (size_t)down_bs) * down_bpb;
    size_t down_expert_stride = (size_t)down_rows * down_row_stride;

    const char *gate_base = (const char *)gguf->data + gate->offset;
    const char *up_base = (const char *)gguf->data + up->offset;
    const char *down_base = (const char *)gguf->data + down->offset;

    for (int e = 0; e < n_exp; e++) {
        moe->gate_exps[e] = (pn_qweight){ .data = gate_base + e * gate_expert_stride,
            .type = gate->type, .rows = gate_rows, .cols = gate_cols,
            .row_stride = gate_row_stride };
        moe->up_exps[e] = (pn_qweight){ .data = up_base + e * gate_expert_stride,
            .type = up->type, .rows = gate_rows, .cols = gate_cols,
            .row_stride = gate_row_stride };
        moe->down_exps[e] = (pn_qweight){ .data = down_base + e * down_expert_stride,
            .type = down->type, .rows = down_rows, .cols = down_cols,
            .row_stride = down_row_stride };
    }

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", i);
    if (pn_qweight_from_gguf(&moe->ffn_gate_shexp, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", i);
    if (pn_qweight_from_gguf(&moe->ffn_up_shexp, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", i);
    if (pn_qweight_from_gguf(&moe->ffn_down_shexp, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", i);
    if (pn_qweight_from_gguf(&moe->ffn_gate_inp_shexp, gguf, name) < 0)
        return -1;

    return 0;
}

/* Load one SSM (Gated-Delta-Net) layer. Returns 0 on success. */
static int qwen35moe_load_ssm_layer(pn_gguf_file *gguf, const pn_qwen35moe_config *qc,
                                    int i, pn_qwen35moe_ssm_layer *l)
{
    char name[128];

    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
    if (!(l->attn_norm = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
    if (!(l->post_attention_norm = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
    if (!(l->ssm_conv1d = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
    if (!(l->ssm_a = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
    if (!(l->ssm_dt = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
    if (!(l->ssm_norm = qwen35moe_load_f32(gguf, name)))
        return -1;

    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
    if (pn_qweight_from_gguf(&l->attn_qkv, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
    if (pn_qweight_from_gguf(&l->attn_gate, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", i);
    if (pn_qweight_from_gguf(&l->ssm_alpha, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", i);
    if (pn_qweight_from_gguf(&l->ssm_beta, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
    if (pn_qweight_from_gguf(&l->ssm_out, gguf, name) < 0)
        return -1;

    return qwen35moe_load_moe(gguf, qc, i, &l->moe);
}

/* Load one full-attention layer. Returns 0 on success. */
static int qwen35moe_load_attn_layer(pn_gguf_file *gguf, const pn_qwen35moe_config *qc,
                                     int i, pn_qwen35moe_attn_layer *l)
{
    char name[128];

    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
    if (!(l->attn_norm = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
    if (!(l->post_attention_norm = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
    if (!(l->attn_q_norm = qwen35moe_load_f32(gguf, name)))
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
    if (!(l->attn_k_norm = qwen35moe_load_f32(gguf, name)))
        return -1;

    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
    if (pn_qweight_from_gguf(&l->attn_q, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
    if (pn_qweight_from_gguf(&l->attn_k, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
    if (pn_qweight_from_gguf(&l->attn_v, gguf, name) < 0)
        return -1;
    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
    if (pn_qweight_from_gguf(&l->attn_output, gguf, name) < 0)
        return -1;

    return qwen35moe_load_moe(gguf, qc, i, &l->moe);
}

/* ============================================================
 * Public: load
 * ============================================================ */

pn_qwen35moe_model *pn_qwen35moe_load(pn_gguf_file *gguf, pn_config *config)
{
    if (!gguf || !config)
        return NULL;

    pn_qwen35moe_model *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->gguf = gguf;

    if (qwen35moe_read_config(gguf, &m->cfg, config) < 0)
        goto fail;

    const pn_qwen35moe_config *qc = &m->cfg;

    /* Final RMSNorm. */
    if (!(m->output_norm = qwen35moe_load_f32(gguf, "output_norm.weight")))
        goto fail;

    /* Embedding + LM head. A2c host-RAM reclaim (eight_card_serving_plan): for
     * UNTIED models (a separate output.weight exists) the ~2 GB F32 token_embd
     * heap is only needed for the per-token embedding lookup — do that on
     * demand from the mmap (pn_qwen35moe_model_embed_row) and skip the heap.
     * For TIED models the embedding also IS the LM head, so it's materialized. */
    pn_gguf_tensor_info *out_info = pn_gguf_find_tensor(gguf, "output.weight");
    if (out_info) {
        if (pn_qweight_from_gguf(&m->output, gguf, "output.weight") < 0)
            goto fail;
        /* Per-row embedding needs the tensor present + row-aligned; otherwise
         * fall back to materializing so no model is ever broken. */
        pn_gguf_tensor_info *ei = pn_gguf_find_tensor(gguf, "token_embd.weight");
        int bsz = ei ? pn_gguf_block_size(ei->type) : 0;
        if (ei && bsz > 0 && (qc->n_embd % bsz) == 0) {
            m->token_embd = NULL;  /* on-demand per-row dequant from the mmap */
        } else {
            size_t embd_numel = 0;
            m->token_embd = pn_gguf_dequantize_tensor(gguf, "token_embd.weight",
                                                      &embd_numel);
            if (!m->token_embd) {
                fprintf(stderr, "qwen35moe: failed to load token_embd.weight\n");
                goto fail;
            }
        }
    } else {
        /* Tied: materialize the embedding (also serves as the F32 LM head). */
        size_t embd_numel = 0;
        m->token_embd = pn_gguf_dequantize_tensor(gguf, "token_embd.weight",
                                                  &embd_numel);
        if (!m->token_embd) {
            fprintf(stderr, "qwen35moe: failed to load token_embd.weight\n");
            goto fail;
        }
        m->output_f32 = malloc(embd_numel * sizeof(float));
        if (!m->output_f32)
            goto fail;
        memcpy(m->output_f32, m->token_embd, embd_numel * sizeof(float));
        m->output = (pn_qweight){ .data = m->output_f32, .type = GGUF_QUANT_F32,
            .rows = qc->n_vocab, .cols = qc->n_embd,
            .row_stride = (size_t)qc->n_embd * sizeof(float) };
    }

    /* Count layers per type, build the index maps. */
    m->block_type = calloc((size_t)qc->n_layer, sizeof(*m->block_type));
    m->block_index = calloc((size_t)qc->n_layer, sizeof(*m->block_index));
    if (!m->block_type || !m->block_index)
        goto fail;
    for (int i = 0; i < qc->n_layer; i++) {
        if (qwen35moe_is_attn_layer(qc, i)) {
            m->block_type[i] = PN_QWEN35MOE_LAYER_ATTN;
            m->block_index[i] = m->n_attn++;
        } else {
            m->block_type[i] = PN_QWEN35MOE_LAYER_SSM;
            m->block_index[i] = m->n_ssm++;
        }
    }

    m->ssm_layers = calloc((size_t)m->n_ssm, sizeof(*m->ssm_layers));
    m->attn_layers = calloc((size_t)m->n_attn, sizeof(*m->attn_layers));
    if ((m->n_ssm && !m->ssm_layers) || (m->n_attn && !m->attn_layers))
        goto fail;

    /* Honor the MTP trap: nextn_predict_layers=0 — never touch blk.N.nextn.* . */
    for (int i = 0; i < qc->n_layer; i++) {
        int idx = m->block_index[i];
        int rc;
        if (m->block_type[i] == PN_QWEN35MOE_LAYER_ATTN)
            rc = qwen35moe_load_attn_layer(gguf, qc, i, &m->attn_layers[idx]);
        else
            rc = qwen35moe_load_ssm_layer(gguf, qc, i, &m->ssm_layers[idx]);
        if (rc < 0) {
            fprintf(stderr, "qwen35moe: failed to load layer %d\n", i);
            goto fail;
        }
        fprintf(stderr, "\r  loaded layer %d/%d", i + 1, qc->n_layer);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "qwen35moe: loaded %d SSM + %d attention layers\n",
            m->n_ssm, m->n_attn);

    return m;

fail:
    pn_qwen35moe_free(m);
    return NULL;
}

/* ============================================================
 * Public: free
 * ============================================================ */

static void qwen35moe_free_moe(pn_qwen35moe_moe *moe)
{
    free(moe->gate_exps);
    moe->gate_exps = NULL;
    free(moe->up_exps);
    moe->up_exps = NULL;
    free(moe->down_exps);
    moe->down_exps = NULL;
}

void pn_qwen35moe_free(pn_qwen35moe_model *model)
{
    if (!model)
        return;

#ifdef USE_CUDA
    /* Release any GPU-resident weight buffers before the CPU weights. */
    pn_qwen35moe_gpu_free(model);
#endif
#ifdef USE_METAL
    /* Metal buffers wrap the mmap'd weights; drop the wrappers before the
     * layer arrays that hold them go away. */
    pn_qwen35moe_metal_free(model);
#endif

    free(model->token_embd);
    free(model->output_norm);
    free(model->output_f32);

    if (model->ssm_layers) {
        for (int i = 0; i < model->n_ssm; i++) {
            pn_qwen35moe_ssm_layer *l = &model->ssm_layers[i];
            free(l->attn_norm);
            free(l->post_attention_norm);
            free(l->ssm_conv1d);
            free(l->ssm_a);
            free(l->ssm_dt);
            free(l->ssm_norm);
            qwen35moe_free_moe(&l->moe);
        }
        free(model->ssm_layers);
    }
    if (model->attn_layers) {
        for (int i = 0; i < model->n_attn; i++) {
            pn_qwen35moe_attn_layer *l = &model->attn_layers[i];
            free(l->attn_norm);
            free(l->post_attention_norm);
            free(l->attn_q_norm);
            free(l->attn_k_norm);
            qwen35moe_free_moe(&l->moe);
        }
        free(model->attn_layers);
    }

    free(model->block_type);
    free(model->block_index);
    free(model);
}

pn_gguf_file *pn_qwen35moe_gguf(const pn_qwen35moe_model *model)
{
    return model ? model->gguf : NULL;
}

/* ============================================================
 * Internal accessors for the forward/state file (qwen35moe_forward.c)
 *
 * The model struct is opaque outside this file; the forward orchestration
 * needs the config + per-type layer counts to size the dual state. It also
 * reaches the per-type block arrays + index maps via the helpers below.
 * ============================================================ */

const pn_qwen35moe_config *pn_qwen35moe_model_config(const pn_qwen35moe_model *m)
{
    return m ? &m->cfg : NULL;
}

pn_qwen35moe_config *pn_qwen35moe_model_config_mut(pn_qwen35moe_model *m)
{
    return m ? &m->cfg : NULL;
}

int pn_qwen35moe_model_accel_mv_enable(pn_qwen35moe_model *m, int enable)
{
    int was_attached;

    if (!m)
        return -1;
    was_attached = m->cfg.mv.fn != NULL;
    if (enable) {
        if (!was_attached && m->mv_detached.fn) {
            m->cfg.mv = m->mv_detached;
            m->mv_detached.fn = NULL;
            m->mv_detached.ctx = NULL;
        }
    } else if (was_attached) {
        m->mv_detached = m->cfg.mv;
        m->cfg.mv.fn = NULL;
        m->cfg.mv.ctx = NULL;
    }
    return was_attached;
}

void pn_qwen35moe_model_accel_mv_forget(pn_qwen35moe_model *m)
{
    if (!m)
        return;
    /* BOTH copies. The shadow is the whole difference between PAUSING the seam and RETIRING
     * it, and a teardown that left the shadow set would hand the freed context back the next
     * time anything re-enabled the seam. */
    m->cfg.mv.fn = NULL;
    m->cfg.mv.ctx = NULL;
    m->mv_detached.fn = NULL;
    m->mv_detached.ctx = NULL;
}

int pn_qwen35moe_model_n_ssm(const pn_qwen35moe_model *m)
{
    return m ? m->n_ssm : 0;
}

int pn_qwen35moe_model_n_attn(const pn_qwen35moe_model *m)
{
    return m ? m->n_attn : 0;
}

uint64_t pn_qwen35moe_kv_bytes_per_token(const pn_qwen35moe_model *m, bool kv_q4)
{
    if (!m)
        return 0;
    const pn_qwen35moe_config *qc = &m->cfg;
    /* Only the full-attention layers grow a per-token KV cache; the SSM layers
     * hold a fixed-size recurrent state that does NOT scale with sequence
     * length, so it is not part of the per-token admission cost. */
    uint64_t bytes_per_vec = kv_q4
        ? (uint64_t)pn_kvq4_row_bytes(qc->head_dim)
        : (uint64_t)qc->head_dim * sizeof(float);
    /* K and V, one packed vector each per KV head, per attention layer. */
    return (uint64_t)m->n_attn * (uint64_t)qc->n_head_kv * 2ULL * bytes_per_vec;
}

/* The forward loop walks layers in global order; these expose the index maps
 * and per-type weight arrays without leaking the struct layout into the header. */
pn_qwen35moe_layer_type pn_qwen35moe_model_layer_type(const pn_qwen35moe_model *m,
                                                      int layer)
{
    return m->block_type[layer];
}

int pn_qwen35moe_model_layer_index(const pn_qwen35moe_model *m, int layer)
{
    return m->block_index[layer];
}

const pn_qwen35moe_ssm_layer *pn_qwen35moe_model_ssm_layer(
    const pn_qwen35moe_model *m, int idx)
{
    return &m->ssm_layers[idx];
}

const pn_qwen35moe_attn_layer *pn_qwen35moe_model_attn_layer(
    const pn_qwen35moe_model *m, int idx)
{
    return &m->attn_layers[idx];
}

const float *pn_qwen35moe_model_token_embd(const pn_qwen35moe_model *m)
{
    return m->token_embd;
}

/* Write embedding row `token` into dst[n_embd]. Materialized F32 heap (tied
 * models / fallback) → row copy; otherwise (A2c untied path) dequantize the row
 * on demand straight from the mmap, so no ~2 GB F32 heap is held. */
void pn_qwen35moe_model_embed_row(const pn_qwen35moe_model *m, uint32_t token,
                                  float *dst)
{
    size_t c = (size_t)m->cfg.n_embd;
    /* Bound the token id against the vocab. The forward-path guards test
     * `(int)token >= n_vocab`, which a token in [INT_MAX+1, UINT32_MAX] slips
     * past (it casts negative), so this — the single embedding chokepoint —
     * must reject out-of-range ids itself: an unbounded `token * c` offset
     * would read past the embedding table (OOB read into the residual stream).
     * Out-of-range → emit a zero row rather than garbage. */
    if (token >= (uint32_t)m->cfg.n_vocab) {
        memset(dst, 0, c * sizeof(float));
        return;
    }
    if (m->token_embd) {
        memcpy(dst, m->token_embd + (size_t)token * c, c * sizeof(float));
        return;
    }
    if (pn_gguf_dequantize_row(m->gguf, "token_embd.weight", token, c, dst) != 0) {
        /* NOT defensive -- this used to memset and carry on, which is how a
         * BF16 model ran to completion emitting noise: every token embedded as
         * a zero vector and the forward dutifully generated from nothing. A
         * zero embedding is not a safe fallback, it is a silent wrong answer.
         * Say so once, loudly, so the cause is in the log instead of being
         * inferred from the output. */
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "qwen35moe: token_embd.weight row %d could not be "
                    "dequantized — the embedding is UNAVAILABLE and generation "
                    "from here is meaningless. Check the tensor's quant is "
                    "supported by pn_dequantize.\n", token);
        }
        memset(dst, 0, c * sizeof(float));
    }
}

const float *pn_qwen35moe_model_output_norm(const pn_qwen35moe_model *m)
{
    return m->output_norm;
}

const pn_qweight *pn_qwen35moe_model_output(const pn_qwen35moe_model *m)
{
    return &m->output;
}

/* Mutable views of the same three objects, for a GPU backend that must attach
 * a device-buffer handle to every weight (pn_qweight.gpu). Kept as separate
 * entry points rather than dropping const from the readers above so the
 * compute path stays unable to write a weight by accident. */
pn_qwen35moe_ssm_layer *pn_qwen35moe_model_ssm_layer_mut(pn_qwen35moe_model *m,
                                                         int idx)
{
    return &m->ssm_layers[idx];
}

pn_qwen35moe_attn_layer *pn_qwen35moe_model_attn_layer_mut(pn_qwen35moe_model *m,
                                                           int idx)
{
    return &m->attn_layers[idx];
}

pn_qweight *pn_qwen35moe_model_output_mut(pn_qwen35moe_model *m)
{
    return &m->output;
}

pn_gguf_file *pn_qwen35moe_model_gguf(const pn_qwen35moe_model *m)
{
    return m ? m->gguf : NULL;
}

/* ============================================================
 * Weight enumeration (see qwen35moe_internal.h)
 * ============================================================ */

/* Emit one weight, skipping the empty slots a variant does not carry (the
 * dense-FFN Bonsai family has no router and no expert arrays). An empty slot is
 * not a weight, and emitting it would put a phantom in every consumer's census. */
static void qwen35moe_emit_weight(pn_qwen35moe_weight_fn fn, void *user,
                                  pn_qweight *w, const char *name, int layer,
                                  int expert)
{
    if (!w || !w->data || w->rows <= 0)
        return;
    pn_qwen35moe_weight_ref ref = { w, name, layer, expert };
    fn(&ref, user);
}

/* Router + routed expert slices + shared-expert SwiGLU + its scalar gate.
 * n_expert is 0 for the dense-FFN variant, where only the shared slots exist. */
static void qwen35moe_for_each_moe_weight(pn_qwen35moe_moe *moe, int layer,
                                          int n_expert,
                                          pn_qwen35moe_weight_fn fn, void *user)
{
    char name[96];

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp", layer);
    qwen35moe_emit_weight(fn, user, &moe->ffn_gate_inp, name, layer, -1);

    for (int e = 0; e < n_expert; e++) {
        if (moe->gate_exps) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps[%d]", layer, e);
            qwen35moe_emit_weight(fn, user, &moe->gate_exps[e], name, layer, e);
        }
        if (moe->up_exps) {
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps[%d]", layer, e);
            qwen35moe_emit_weight(fn, user, &moe->up_exps[e], name, layer, e);
        }
        if (moe->down_exps) {
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps[%d]", layer, e);
            qwen35moe_emit_weight(fn, user, &moe->down_exps[e], name, layer, e);
        }
    }

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp", layer);
    qwen35moe_emit_weight(fn, user, &moe->ffn_gate_shexp, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp", layer);
    qwen35moe_emit_weight(fn, user, &moe->ffn_up_shexp, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp", layer);
    qwen35moe_emit_weight(fn, user, &moe->ffn_down_shexp, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp", layer);
    qwen35moe_emit_weight(fn, user, &moe->ffn_gate_inp_shexp, name, layer, -1);
}

/* One Gated-Delta-Net SSM layer's projections. The conv/decay/norm weights are
 * F32 arrays owned by the model, not pn_qweights, so they are not emitted. */
static void qwen35moe_for_each_ssm_weight(pn_qwen35moe_ssm_layer *l, int layer,
                                          int n_expert,
                                          pn_qwen35moe_weight_fn fn, void *user)
{
    char name[96];

    snprintf(name, sizeof(name), "blk.%d.attn_qkv", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_qkv, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.attn_gate", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_gate, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ssm_alpha", layer);
    qwen35moe_emit_weight(fn, user, &l->ssm_alpha, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ssm_beta", layer);
    qwen35moe_emit_weight(fn, user, &l->ssm_beta, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.ssm_out", layer);
    qwen35moe_emit_weight(fn, user, &l->ssm_out, name, layer, -1);

    qwen35moe_for_each_moe_weight(&l->moe, layer, n_expert, fn, user);
}

/* One full-attention layer's projections. */
static void qwen35moe_for_each_attn_weight(pn_qwen35moe_attn_layer *l, int layer,
                                           int n_expert,
                                           pn_qwen35moe_weight_fn fn, void *user)
{
    char name[96];

    snprintf(name, sizeof(name), "blk.%d.attn_q", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_q, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.attn_k", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_k, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.attn_v", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_v, name, layer, -1);
    snprintf(name, sizeof(name), "blk.%d.attn_output", layer);
    qwen35moe_emit_weight(fn, user, &l->attn_output, name, layer, -1);

    qwen35moe_for_each_moe_weight(&l->moe, layer, n_expert, fn, user);
}

void pn_qwen35moe_for_each_weight(pn_qwen35moe_model *m,
                                  pn_qwen35moe_weight_fn fn, void *user)
{
    if (!m || !fn)
        return;

    for (int i = 0; i < m->cfg.n_layer; i++) {
        int idx = m->block_index[i];
        if (m->block_type[i] == PN_QWEN35MOE_LAYER_SSM)
            qwen35moe_for_each_ssm_weight(&m->ssm_layers[idx], i,
                                          m->cfg.n_expert, fn, user);
        else
            qwen35moe_for_each_attn_weight(&m->attn_layers[idx], i,
                                           m->cfg.n_expert, fn, user);
    }
    qwen35moe_emit_weight(fn, user, &m->output, "output", -1, -1);
}

/* GPU-residency handle accessors. The handle is an opaque pn_qwen35moe_gpu
 * (defined in qwen35moe_forward_cuda.c); this file only stores the pointer so
 * the model carries its persistent fc_buffer weights across forward calls. */
void *pn_qwen35moe_model_gpu(const pn_qwen35moe_model *m)
{
    return m ? m->gpu : NULL;
}

void pn_qwen35moe_model_set_gpu(pn_qwen35moe_model *m, void *gpu)
{
    if (m)
        m->gpu = gpu;
}

/* Explicit die placement (pn_load_on). Stored before the GPU upload so
 * gpu_die_ids()/gpu_split_layer() in qwen35moe_forward_cuda.c can prefer it
 * over the QWEN35_GPU_DIE_A/B env knobs. Capped at the struct's 8 entries
 * (the forward consumes at most 2 today). */
void pn_qwen35moe_model_set_dies(pn_qwen35moe_model *m, const int *dies, int n)
{
    if (!m)
        return;
    if (!dies || n <= 0) {
        m->n_gpu_dies = 0;
        return;
    }
    if (n > (int)(sizeof(m->gpu_dies) / sizeof(m->gpu_dies[0])))
        n = (int)(sizeof(m->gpu_dies) / sizeof(m->gpu_dies[0]));
    for (int i = 0; i < n; i++)
        m->gpu_dies[i] = dies[i];
    m->n_gpu_dies = n;
}

int pn_qwen35moe_model_dies(const pn_qwen35moe_model *m, const int **dies_out)
{
    if (dies_out)
        *dies_out = m ? m->gpu_dies : NULL;
    return m ? m->n_gpu_dies : 0;
}

#ifdef USE_CUDA
int pn_qwen35moe_model_gpu_loaded(const pn_qwen35moe_model *m)
{
    return (m && m->gpu) ? 1 : 0;
}
#endif

/* See qwen35moe.h.  Lives here (not in the CUDA file) so a non-CUDA build still
 * links it; the flag is simply never read when there is no device top-k. */
/* True when `model`'s pn_state carries a pn_qwen35moe_state — the MoE arch and
 * the dense Bonsai/Ornith family, which share this state and forward. */
int pn_qwen35moe_owns_state(const pn_model *model)
{
    if (!model) return 0;
    return model->config.arch == PN_ARCH_QWEN35MOE ||
           model->config.arch == PN_ARCH_BONSAI;
}

void pn_qwen35moe_state_set_full_logits(pn_state *state, int on)
{
    if (!state || !state->kv_cache) return;
    ((pn_qwen35moe_state *)state->kv_cache)->full_logits = on ? 1 : 0;
}
