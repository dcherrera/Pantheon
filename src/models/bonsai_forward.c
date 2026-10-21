/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bonsai_forward.c
 * @brief PrismML Bonsai-27B top-level forward/loader entry (PN_ARCH_BONSAI).
 *
 * This is Bonsai's OWN entry file, kept physically separate from the shipped
 * Nex-N2 forward (qwen35moe_forward*.c). It owns the pn_forward_fn/loader/state
 * seam for PN_ARCH_BONSAI and DELEGATES the heavy lifting to the shared
 * qwen35moe machinery: Bonsai is the SAME hybrid GDN + sparse-full-attention
 * family (general.architecture="qwen35"), differing only in a DENSE SwiGLU FFN
 * (no MoE) and native-ternary (Q2_0_g128 / type-142) weights.
 *
 * Why delegate instead of clone: the hybrid forward is ~7.4k lines of shared,
 * die-scratch-heavy CUDA orchestration (GDN block, flash-attn, Q4 KV, RoPE,
 * RMSNorm, the ternary fc_matvec_q2_0). Cloning it to strip MoE -> dense would
 * be a large correctness/drift risk (the exact hazard the bring-up flagged).
 * Instead, the ONE Bonsai-specific behavior — the dense SwiGLU FFN block — is a
 * config-driven branch inside the shared per-layer FFN helper, selected by
 * cfg->dense_ffn (set at load from the GGUF metadata). The shipped Nex-N2 (qwen35moe,
 * dense_ffn=0) never takes that branch and is byte-identical. Bonsai therefore
 * has its own arch identity + its own forward entry file, without a risky clone.
 */

#include "bonsai.h"
#include "qwen35moe.h"

#include <string.h>

int pn_bonsai_is_bonsai_gguf(pn_gguf_file *gguf)
{
    if (!gguf)
        return 0;
    const char *arch = pn_gguf_get_str(gguf, "general.architecture");
    if (!arch || strcmp(arch, "qwen35") != 0)
        return 0;
    /* Dense FFN (Bonsai) vs routed MoE (Nex-N2): a plain per-layer ffn_gate is
     * present and the merged expert tensor is absent. */
    if (pn_gguf_find_tensor(gguf, "blk.0.ffn_gate_exps.weight") != NULL)
        return 0;
    if (pn_gguf_find_tensor(gguf, "blk.0.ffn_gate.weight") == NULL)
        return 0;
    return 1;
}

void *pn_bonsai_load(pn_gguf_file *gguf, pn_config *config)
{
    /* The shared loader detects the dense-FFN variant from the metadata
     * (expert_count absent + ffn_gate present) and sets cfg->dense_ffn. */
    return pn_qwen35moe_load(gguf, config);
}

tm_tensor *pn_bonsai_forward(pn_model *model, uint32_t token, pn_state *state)
{
    return pn_qwen35moe_forward(model, token, state);
}

#ifdef USE_CUDA
tm_tensor *pn_bonsai_forward_cuda(pn_model *model, uint32_t token,
                                  pn_state *state)
{
    return pn_qwen35moe_forward_cuda(model, token, state);
}

int pn_bonsai_gpu_upload(void *model)
{
    return pn_qwen35moe_gpu_upload((pn_qwen35moe_model *)model);
}

int pn_bonsai_model_gpu_loaded(const void *model)
{
    return pn_qwen35moe_model_gpu_loaded((const pn_qwen35moe_model *)model);
}
#endif

void *pn_bonsai_state_new(void *weights)
{
    return pn_qwen35moe_state_new(weights);
}

void pn_bonsai_state_free(void *state)
{
    pn_qwen35moe_state_free(state);
}

void pn_bonsai_state_reset(void *state)
{
    pn_qwen35moe_state_reset(state);
}

void pn_bonsai_free(void *model)
{
    pn_qwen35moe_free((pn_qwen35moe_model *)model);
}

pn_gguf_file *pn_bonsai_gguf(const void *model)
{
    return pn_qwen35moe_gguf((const pn_qwen35moe_model *)model);
}

void pn_bonsai_model_set_dies(void *model, const int *dies, int n)
{
    pn_qwen35moe_model_set_dies((pn_qwen35moe_model *)model, dies, n);
}
