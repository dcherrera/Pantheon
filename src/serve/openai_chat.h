/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file openai_chat.h
 * @brief OpenAI-compatible endpoint handlers: GET /v1/models and
 *        POST /v1/chat/completions. Parses the request JSON, routes the
 *        request's `model` field to a resident registry entry, maps sampling
 *        params, drives serve_generation, and shapes chat.completion /
 *        chat.completion.chunk responses.
 *
 * @status /v1/models and /v1/chat/completions (streaming + non-streaming),
 *         including OpenAI tool calling (tools rendered into the prompt, model
 *         <tool_call> output parsed to `tool_calls`, multi-step tool loops).
 *         Multi-model: requests route by `model` name; per-model generation
 *         serialization (429 on a busy model); models on different dies run
 *         concurrently.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_OPENAI_CHAT_H
#define PN_SERVE_OPENAI_CHAT_H

#include "../../include/pantheon.h"
#include "model_registry.h"

/** Handle GET /v1/models: emit every resident model as an OpenAI model list. */
void oai_handle_models(int client_fd, mr_registry *reg);

/**
 * Handle POST /v1/chat/completions. Parses `body`, routes the request's
 * `model` field to a resident registry entry (single resident model → always
 * that model, today's behavior; several → exact name match or 404
 * model_not_found), serializes on the entry's gen_lock (429 when that model is
 * busy), runs the completion, and writes either a JSON chat.completion or an
 * SSE stream of chat.completion.chunk events. Errors are OpenAI-shaped JSON.
 */
/**
 * @param session_id  Optional prefix-cache session key (PC0.1) from the
 *        `X-Session-Id` request header; a `session_id` field in the request
 *        body overrides it. NULL/empty → anonymous (byte-identical full-prefill
 *        behaviour; no caching). Backward-compatible: existing clients that
 *        send neither are unaffected.
 * @param replica_header  X-Pantheon-Replica request header (may be NULL); a
 *        "replica" body field wins. A new session binds to that copy only;
 *        every response after routing carries X-Pantheon-Replica (replica_pin.h).
 */
void oai_handle_chat(int client_fd, mr_registry *reg, const char *body,
                     const char *session_id, const char *replica_header);

/**
 * Handle POST /session/restore — session warm-tier import (K80 serving-ops
 * Phase 7). Body {"name","session_id","blob":"<base64>","tokens":[...]}: admits
 * the session (429 at capacity), validates the blob's weights_version against
 * the model, deserializes + restores it into a live slot with the committed
 * `tokens` history, and positions the state so the next chat turn append-resumes
 * off it. A weights mismatch / corrupt blob / dim mismatch is a clean error
 * (the client falls back to replaying the message history). `session_id` is the
 * X-Session-Id header fallback when the body omits it. `replica_header` (or a
 * "replica" body field, which wins) picks the copy a new session binds to.
 */
void oai_handle_session_restore(int client_fd, mr_registry *reg,
                                const char *body, const char *session_id,
                                const char *replica_header);

/**
 * Handle POST /session/release {name, session_id} — free a session's serve slot on
 * a client's clean shutdown, returning capacity immediately rather than waiting out
 * the idle-TTL. Idempotent (an unknown session still returns released:true), best-
 * effort from the client's side. `session_id` is the X-Session-Id header fallback
 * when the body omits it.
 */
void oai_handle_session_release(int client_fd, mr_registry *reg,
                                const char *body, const char *session_id);

/**
 * Handle GET /session/estimate?name=&tokens=N[&blob_bytes=&upload_bps=] —
 * warm-restore vs cold-reprefill time estimate for the client UX (Phase 7).
 * Returns cold (tokens / measured prefill rate) and warm (blob upload + host→GPU
 * upload) seconds. `blob_bytes`/`upload_bytes_per_s` <= 0 → estimated defaults.
 */
void oai_handle_session_estimate(int client_fd, mr_registry *reg,
                                 const char *name, int tokens,
                                 double blob_bytes, double upload_bytes_per_s);

#endif /* PN_SERVE_OPENAI_CHAT_H */
