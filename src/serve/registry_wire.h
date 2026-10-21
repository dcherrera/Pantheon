/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file registry_wire.h
 * @brief Shared wire-protocol contract for the dispatch layer (the client §10 / build
 *        plan Phase G). Defines the JSON shapes exchanged between a Pantheon
 *        serve instance (the G0 client) and the registry+router (G1/G2), plus a
 *        few small limits both sides agree on. This is the single source of
 *        truth for the field names so serve and router never drift.
 *
 * Dedup / load-balance semantics (CRITICAL — do not confuse the two ids):
 *   - `model_id`    — the SHARED model identity clients request. This is what
 *                     `GET /v1/models` shows: ONE entry per model_id. Two serve
 *                     processes hosting the same weights advertise the SAME
 *                     model_id (e.g. the fleet's two "ornith").
 *   - `instance_id` — UNIQUE per serve process. Multiple instances that share a
 *                     model_id are REPLICAS the router load-balances across.
 *
 * Endpoints (registry side, local position = no auth):
 *   POST   /registry/register    {model_id,instance_id,node_id,endpoint,
 *                                  transport,capacity,context_window}
 *   POST   /registry/heartbeat   {instance_id,load:{active_slots,queue_depth,
 *                                  free_slots,context_window},healthy}
 *   DELETE /registry/deregister  {instance_id}
 *
 * `context_window` is the instance's EFFECTIVE per-session context window in
 * tokens (pn_effective_ctx_window). It rides the HEARTBEAT as well as the
 * register because the memory ladder can move it at runtime — a value fixed at
 * registration goes stale exactly when it matters. 0/absent = unreported, and
 * the router then omits the field rather than advertise a guess. It reaches
 * clients unchanged as `context_window` on the router's GET /v1/models and per
 * instance in /registry/fleet.
 *   GET    /registry/fleet       -> model-level fleet view (ops/TUI)
 *   GET    /v1/models            -> deduped-by-model_id live model list
 *   POST   /v1/chat/completions  -> least-busy dispatch + SSE pass-through
 *   POST   /v1/completions       -> same
 *
 * Liveness: no heartbeat within RW_TTL_SECONDS ⇒ the registry evicts the
 * instance (crash-safe). Serve should heartbeat every RW_HEARTBEAT_SECONDS.
 *
 * @status G0/G1 contract. Field names below are load-bearing on both sides.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_REGISTRY_WIRE_H
#define PN_SERVE_REGISTRY_WIRE_H

/* Liveness TTL: an instance with no heartbeat in this many seconds is evicted. */
#define RW_TTL_SECONDS 15

/* Recommended serve heartbeat cadence (well under the TTL). */
#define RW_HEARTBEAT_SECONDS 3

/* Field-length caps for the in-memory instance table (bounded, no heap churn
 * per row). Generous but fixed so the table row is a POD struct. */
#define RW_MODEL_ID_MAX   96
#define RW_INSTANCE_ID_MAX 96
#define RW_NODE_ID_MAX    64
#define RW_ENDPOINT_MAX   256    /* e.g. "http://127.0.0.1:8201" */
#define RW_TRANSPORT_MAX  16     /* "local" | "wg" */

/* Transport values (string payload of the `transport` field). */
#define RW_TRANSPORT_LOCAL "local"
#define RW_TRANSPORT_WG    "wg"

#endif /* PN_SERVE_REGISTRY_WIRE_H */
