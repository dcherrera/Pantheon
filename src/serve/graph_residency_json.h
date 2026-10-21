/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file graph_residency_json.h
 * @brief Serialises a model's CUDA-graph residency into the JSON both the
 *        client surface (GET /v1/models) and the operator surface
 *        (GET /admin/models) emit.
 *
 * WHY IT IS ON THE WIRE AT ALL
 *   A CUDA-graph capture that fails degrades to the direct kernel body and
 *   produces CORRECT output — the only trace is that the layer costs its share
 *   of ~1.8 ms/step forever. Nothing in the logs, the completion `usage`, or
 *   any metric said so, which is how a serve could silently shed graph coverage
 *   layer by layer and still look healthy. This is that missing gauge.
 *
 *   Three states, and only ONE of them is a fault:
 *     live       — a captured graph is replaying. Good.
 *     failed     — a capture was attempted and did not stick. THE fault.
 *     ineligible — this model never captures (its MoE experts are not Q3_K, so
 *                  the grouped path is off). A Q5_K_M model reporting 40
 *                  ineligible layers is correct behaviour, not a problem.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_GRAPH_RESIDENCY_JSON_H
#define PN_SERVE_GRAPH_RESIDENCY_JSON_H

#include "../models/gpu_graph_capture_tracker.h"
#include "json_writer.h"

/**
 * Append `,"graphs":{...}` — the client-facing summary. Writes NOTHING when
 * the model has no graph slots at all (CPU-resident, or an architecture with no
 * graph path), the same omit-when-unknown rule `context_window` follows.
 *
 * @param jw  Open json_writer positioned inside a model object
 * @param r   Residency snapshot (mr_info.graphs)
 *
 * Thread-safe: yes (writes only to `jw`)
 * Reentrant: yes
 * Signal-safe: no
 */
void gr_write_summary(json_writer *jw, const pn_model_graph_residency *r);

/**
 * Append `,"graphs":{...}` with the per-class breakdown and the lifetime
 * capture/failure counters — the operator view. A class the model has never
 * exercised is omitted rather than reported as an empty deficit.
 *
 * Thread-safe: yes (writes only to `jw`)
 * Reentrant: yes
 * Signal-safe: no
 */
void gr_write_detail(json_writer *jw, const pn_model_graph_residency *r);

#endif /* PN_SERVE_GRAPH_RESIDENCY_JSON_H */
