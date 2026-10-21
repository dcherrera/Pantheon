/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file router.h
 * @brief Dispatch router (Phase G2): the request-facing side of the registry.
 *        Parses the OpenAI `model` field, picks the least-busy live instance of
 *        that model from the registry, reverse-proxies the request to it with
 *        SSE streaming pass-through, and fails over to the next-least-busy
 *        instance if the chosen one errors before any bytes are relayed.
 *
 *        Endpoints handled:
 *          POST /v1/chat/completions
 *          POST /v1/completions
 *
 * Session affinity (PC0.2): a request carrying an X-Session-Id is routed to the
 * instance that last served that session (where the serve's prefix cache holds
 * its prefilled KV), forwarding the X-Session-Id header upstream. A new session
 * (or one whose bound instance is gone) falls back to least-busy and is
 * recorded. Requests with no session id route least-busy exactly as before.
 *
 * @status PC0.2: least-busy pick, failover, streaming + non-streaming relay,
 *         session affinity + X-Session-Id forwarding.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_ROUTER_H
#define PN_SERVE_ROUTER_H

#include "registry.h"
#include "session_affinity.h"

/**
 * Handle a POST /v1/chat/completions or /v1/completions request. Reads `model`
 * from the JSON body, picks a live instance (session-affinity when a session id
 * is given, else least-busy), reverse-proxies with SSE pass-through, and relays
 * the response to `client_fd`. Sends 400 on a missing/unparseable model,
 * 503 model_unavailable when no live instance exists (or all failovers fail).
 *
 * @param client_fd   the downstream client connection
 * @param reg         the live-instance registry
 * @param sa          the session -> instance affinity map (NULL disables it)
 * @param path        the upstream path to proxy to ("/v1/chat/completions" etc.)
 * @param body        the request body (JSON), NUL-terminated
 * @param session_id  the X-Session-Id header value, or NULL (anonymous)
 *
 * Thread-safe: yes (reg + sa carry their own locks)
 */
void router_handle_completion(int client_fd, reg_registry *reg,
                              session_affinity *sa, const char *path,
                              const char *body, const char *session_id);

#endif /* PN_SERVE_ROUTER_H */
