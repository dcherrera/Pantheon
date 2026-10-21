/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_admin.h
 * @brief Runtime load/unload admin endpoints for multi-model serving:
 *        models add/drop without restarting serve, so
 *        GET /v1/models is a live list.
 *
 *        Routes (wired in serve_command.c):
 *          POST /admin/models/load    {"path":"...", "name":"...",
 *                                      "dies":[0,1], "quantize":false}
 *          POST /admin/models/unload  {"name":"..."}
 *          GET  /admin/dies           die-manager occupancy snapshot
 *
 *        These share the serve port, so keep that port private (localhost or
 *        a private network) and put a gateway in front that forwards only
 *        /v1. Loads run on the caller's connection thread — a big model load
 *        takes a while and the client waits; serving of resident models
 *        continues meanwhile.
 *
 * @status Live on the CPU host; real die placement numbers are gated on GPU hardware.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_MODEL_ADMIN_H
#define PN_SERVE_MODEL_ADMIN_H

#include "model_registry.h"

/** POST /admin/models/load — load + register a model at runtime. */
void ma_handle_load(int client_fd, mr_registry *reg, const char *body);

/** POST /admin/models/unload — drain and unload a resident model. */
void ma_handle_unload(int client_fd, mr_registry *reg, const char *body);

/** GET /admin/dies — die-manager occupancy snapshot (empty list on CPU). */
/**
 * GET /admin/models — every LIVE instance, one row each (NOT deduped by name,
 * unlike /v1/models: two instances of the same model on different dies are two
 * rows, because an operator's question is per-instance).
 *
 * The row is the same one /admin/models/load returns, so until now its
 * operator-only fields — context_window_configured, and the CUDA-graph
 * residency `graphs` — could be read only as a side effect of LOADING a model.
 * That is why task #20 had to infer the memory ladder's runtime clamp instead
 * of reading it. This makes the row readable at any time.
 *
 * Thread-safe: yes (mr_snapshot takes the registry lock)
 * Reentrant: no (writes to a client fd)
 * Signal-safe: no
 */
void ma_handle_models(int client_fd, mr_registry *reg);

void ma_handle_dies(int client_fd);

/**
 * POST /admin/session/export {"name":"...","session_id":"..."} — session
 * warm-tier export (K80 serving-ops Phase 7). Snapshot + serialize the named
 * model's live session state for `session_id` into a portable blob and return
 * it base64 in JSON alongside its weights_version tag and the committed token
 * ids the blob covers, so the CLIENT can store it and later POST
 * /session/restore to warm-restart the session. Read-only w.r.t. the session.
 *
 * Covers BOTH session models via the model_session.h facade — the batched
 * scheduler's warm decode slot as well as the single-user prefix cache. Since
 * the host KV mirror was removed (Phase 10b / INV3) this is the only path by
 * which a session's state can leave the GPU, so it materializes the KV rows
 * from the decode-slot ring on demand rather than reading any mirror.
 *
 * 404 if the model or session is unknown (including "the session's rows are no
 * longer in any GPU ring" — the client falls back to message replay); 429 if a
 * turn is in flight on that session; 400 if the model has no session store.
 */
void ma_handle_session_export(int client_fd, mr_registry *reg, const char *body);

#endif /* PN_SERVE_MODEL_ADMIN_H */
