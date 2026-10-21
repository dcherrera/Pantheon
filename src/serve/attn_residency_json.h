/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file attn_residency_json.h
 * @brief Serialises a model's ATTENTION-PATH residency into the JSON both the
 *        client surface (GET /v1/models) and the operator surface
 *        (GET /admin/models) emit.
 *
 * WHY IT IS ON THE WIRE AT ALL
 *   Same argument as graph_residency_json.h, only sharper. When the on-die
 *   flash-attention path is unavailable the layer falls back to host-side
 *   causal GQA softmax on a 2-core host — the exact wall this project spent a
 *   phase escaping (87.5% of an 8k prefill; fixing it took that prefill from
 *   26 minutes to 3.82). The fallback returns the SAME tokens, logs nothing per
 *   step, and on the depth bench costs 13.02 -> 1.22 tok/s (K=2, depth 4096).
 *   A user gets correct answers an order of magnitude slower, with no error
 *   anywhere, and it gets worse the deeper the context runs.
 *
 *   Two states, and `host` is the fault:
 *     ondie — the flash kernels are running. Good.
 *     host  — this layer is on the host-softmax fallback. `reasons` names why.
 *
 *   `gave_up` is the number that should page someone: a layer whose KV-ring
 *   allocation spent its retry budget will not try again at this geometry.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_ATTN_RESIDENCY_JSON_H
#define PN_SERVE_ATTN_RESIDENCY_JSON_H

#include "../models/gpu_attn_path_tracker.h"
#include "json_writer.h"

/**
 * Append `,"attention":{...}` — the client-facing summary. Writes NOTHING when
 * the forward has never reached an attention layer (CPU-resident, an
 * architecture with no on-die attention path, or a model that has not served a
 * request yet), the same omit-when-unknown rule `context_window` follows.
 *
 * @param jw  Open json_writer positioned inside a model object
 * @param r   Residency snapshot (mr_info.attn)
 *
 * Thread-safe: yes (writes only to `jw`)
 * Reentrant: yes
 * Signal-safe: no
 */
void ar_write_summary(json_writer *jw, const pn_model_attn_residency *r);

/**
 * Append `,"attention":{...}` with the per-reason breakdown and the lifetime
 * counters — the operator view. A reason no layer is currently sitting on is
 * omitted rather than reported as a zero.
 *
 * Thread-safe: yes (writes only to `jw`)
 * Reentrant: yes
 * Signal-safe: no
 */
void ar_write_detail(json_writer *jw, const pn_model_attn_residency *r);

#endif /* PN_SERVE_ATTN_RESIDENCY_JSON_H */
