/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bonsai.h
 * @brief PrismML Bonsai-27B (native ternary) arch entry — its OWN forward/loader
 *        surface (PN_ARCH_BONSAI). Bonsai declares general.architecture="qwen35":
 *        the SAME hybrid Gated-DeltaNet + sparse-full-attention family as
 *        Nex-N2-mini (qwen35moe), but with a DENSE SwiGLU FFN (no MoE) and native
 *        ternary (Q2_0_g128 / type-142) weights.
 *
 *        Isolation model: Bonsai gets its own arch enum and its own top-level
 *        forward/loader entry points (this file + bonsai_forward.c), kept
 *        physically separate from the shipped Nex-N2 forward. The heavy,
 *        already-shared machinery (GDN block, full-attention block, RoPE,
 *        RMSNorm, Q4 KV, the ternary fc_matvec_q2_0, GPU upload, dual-state
 *        lifecycle) is REUSED from the qwen35moe module rather than cloned —
 *        the only Bonsai-specific behavior (dense SwiGLU FFN) is driven by a
 *        config property (cfg->dense_ffn) the shared block reads. The shipped Nex-N2
 *        (qwen35moe, dense_ffn=0) is byte-identical either way.
 *
 * @status Phase 2 refactor: own-files isolation of the Bonsai bring-up that was
 *         first brought up as a dense_ffn flag on the qwen35moe path.
 */

#ifndef PN_BONSAI_H
#define PN_BONSAI_H

#include "../core/engine.h"
#include "../formats/gguf.h"

/**
 * True if the open GGUF is a Bonsai-family model: general.architecture="qwen35"
 * AND a dense FFN (no routed experts / merged expert tensors — a plain
 * blk.0.ffn_gate is present). Nex-N2-mini (qwen35moe, routed MoE) is NOT Bonsai.
 * engine.c uses this to route to PN_ARCH_BONSAI vs PN_ARCH_QWEN35MOE.
 */
int pn_bonsai_is_bonsai_gguf(pn_gguf_file *gguf);

/**
 * Load a Bonsai GGUF. Reuses the qwen35moe loader (which detects the dense-FFN
 * variant from the metadata and loads ffn_gate/up/down into the shared-expert
 * slots). Returns the shared opaque qwen35moe model handle (as void*) so the
 * engine can drive it through the qwen35moe lifecycle. NULL on failure.
 */
void *pn_bonsai_load(pn_gguf_file *gguf, pn_config *config);

/**
 * Bonsai CPU forward (pn_forward_fn). Runs the hybrid stack for one token
 * (48 GDN + 16 full-attn + dense SwiGLU FFN, ternary matvecs on the CPU
 * oracle). Delegates to the shared qwen35moe CPU forward.
 */
tm_tensor *pn_bonsai_forward(pn_model *model, uint32_t token, pn_state *state);

#ifdef USE_CUDA
/**
 * Bonsai single-die GPU forward (pn_forward_fn). Delegates to the shared
 * qwen35moe CUDA forward; the dense-FFN GPU block + fc_matvec_q2_0 ternary
 * matvec run inside the shared forward, selected by cfg->dense_ffn.
 */
tm_tensor *pn_bonsai_forward_cuda(pn_model *model, uint32_t token,
                                  pn_state *state);

/** Upload Bonsai weights to persistent VRAM (delegates to qwen35moe). */
int  pn_bonsai_gpu_upload(void *model);
/** True if the Bonsai model's weights are resident on a GPU die. */
int  pn_bonsai_model_gpu_loaded(const void *model);
#endif

/** Bonsai state lifecycle (dual SSM + attn-KV) — delegates to qwen35moe. */
void *pn_bonsai_state_new(void *weights);
void  pn_bonsai_state_free(void *state);
void  pn_bonsai_state_reset(void *state);

/** Free the Bonsai model handle (delegates to qwen35moe). */
void  pn_bonsai_free(void *model);
/** GGUF backing the Bonsai model (delegates to qwen35moe). */
pn_gguf_file *pn_bonsai_gguf(const void *model);
/** Explicit die placement for the GPU upload (delegates to qwen35moe). */
void  pn_bonsai_model_set_dies(void *model, const int *dies, int n);

#endif /* PN_BONSAI_H */
