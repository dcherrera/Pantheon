/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_kv_geometry.c
 * @brief Implementation of per-model KV-cache geometry for admission.
 *        Dispatches on the model's architecture: the qwen35moe/bonsai hybrid
 *        knows its own attention-layer count (delegated to
 *        pn_qwen35moe_kv_bytes_per_token); every other supported arch is dense
 *        and grows KV on all layers as fp32.
 *
 * @status Phase 1 — used by pn_admit to price requests against real geometry.
 * @issues None
 * @todo None
 */

#include "pn_kv_geometry.h"

#include "../../models/qwen35moe.h"

/* Dense (all-layers-KV) per-token cost. Dense forwards (llama_forward_cuda and
 * the CPU llama path) store the KV cache as fp32 [n_kv_head, max_seq, head_dim]
 * per layer, so the fp32 term is the truth here regardless of a Q4 request. */
static uint64_t dense_kv_bytes_per_token(const pn_config *cfg)
{
    int head_dim = pn_head_dim(cfg);
    if (cfg->n_layer <= 0 || cfg->n_kv_head <= 0 || head_dim <= 0)
        return 0;
    return (uint64_t)cfg->n_layer * (uint64_t)cfg->n_kv_head * 2ULL
         * (uint64_t)head_dim * sizeof(float);
}

uint64_t pn_kv_bytes_per_token(const pn_model *model, bool kv_q4)
{
    if (!model)
        return 0;

    /* Hybrid stack (Nex-N2-mini MoE, Bonsai, Ornith): KV grows only on the
     * full-attention layers. Both archs store their weights as a
     * pn_qwen35moe_model, so the same accessor serves both. */
    if (model->config.arch == PN_ARCH_QWEN35MOE ||
        model->config.arch == PN_ARCH_BONSAI) {
        const pn_qwen35moe_model *qm =
            (const pn_qwen35moe_model *)model->weights;
        uint64_t per_tok = pn_qwen35moe_kv_bytes_per_token(qm, kv_q4);
        if (per_tok > 0)
            return per_tok;
    }

    /* Every other supported arch is dense — all layers hold fp32 KV. */
    return dense_kv_bytes_per_token(&model->config);
}
