/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_kv_geometry.h
 * @brief Per-model KV-cache geometry for admission (KV Phase 1).
 *
 *        Admission must price a request against the model's REAL KV growth, not
 *        a one-size constant. A dense model grows KV on every layer; a hybrid
 *        (qwen35moe / bonsai / ornith) grows it only on its full-attention
 *        layers, so a fixed PN_ADMIT_KV_BYTES_PER_TOKEN_* mis-prices it by
 *        several-fold and either over-reserves (fewer sessions) or under-guards
 *        (silent drop / OOM risk). This module derives the honest whole-model
 *        per-token KV byte cost from the loaded model's own geometry.
 *
 * @status Phase 1 — the correctness fix that makes every downstream KV budget
 *         real (session cap, reject-backstop, KV-usage reporting).
 * @issues None
 * @todo None
 */

#ifndef PN_KV_GEOMETRY_H
#define PN_KV_GEOMETRY_H

#include <stdbool.h>
#include <stdint.h>

#include "../engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Whole-model KV bytes added per admitted token, from THIS model's geometry.
 *
 *   hybrid (qwen35moe / bonsai): n_attn * n_kv_head * 2(K+V) * bytes_per_vec
 *     — only the full-attention layers hold a growing KV cache; the SSM layers
 *       carry a fixed recurrent state that does not scale with sequence length.
 *   dense (llama / qwen / mistral / phi): n_layer * n_kv_head * 2 * bytes_per_vec
 *
 * `bytes_per_vec` is the packed Q4 row size under `kv_q4` (for the hybrid Q4 KV
 * path), else head_dim * sizeof(float). Dense forwards store fp32 KV, so their
 * cost is computed fp32 regardless of `kv_q4`.
 *
 * Returns 0 when the model / geometry is unavailable — callers fall back to the
 * conservative generic constant so admission never silently prices at zero.
 *
 * Thread-safe: yes (reads immutable model config)
 * Reentrant: yes
 * Signal-safe: yes */
uint64_t pn_kv_bytes_per_token(const pn_model *model, bool kv_q4);

#ifdef __cplusplus
}
#endif

#endif /* PN_KV_GEOMETRY_H */
