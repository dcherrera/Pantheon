/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_mode.h
 * @brief `pantheon serve --gateway`: the serve sits behind a trusted gateway
 *        that carries many customers' traffic on one key.
 *
 * What gateway mode changes (and nothing else — off, the serve behaves exactly
 * as it does for home use):
 *   - The API port answers only the OpenAI surface (pn_gateway_route_allowed);
 *     admin and session routes are on a separate loopback listener or off.
 *   - Every API request must carry the gateway's bearer key (gateway_auth.h).
 *   - Client session keys (`session_id`, `prompt_cache_key`, …) are scoped to
 *     the tenant the gateway names in `X-Pantheon-Tenant`; without one the turn
 *     runs anonymously. The `X-Session-Id` and `X-Pantheon-Replica` client
 *     headers are ignored.
 *   - A session seat is a cache, not a reservation: an idle seat is taken
 *     from its session whenever other traffic needs the slot, and idle seats
 *     expire after PN_GATEWAY_SESSION_IDLE_TTL_MS. One customer's key can
 *     never lock other customers (or anonymous turns) out.
 *   - Requests a text-only serve cannot honour (`json_schema`, image / audio /
 *     video parts) are refused with an OpenAI-style 400 instead of degraded.
 *
 * The config is process-wide and set once at startup, before any model loads
 * and before the accept loop starts.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_GATEWAY_MODE_H
#define PANTHEON_SERVE_GATEWAY_MODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gateway_auth.h"

/** Idle seat lifetime in gateway mode when PN_SESSION_IDLE_TTL_MS is unset. */
#define PN_GATEWAY_SESSION_IDLE_TTL_MS 120000u

/** Longest tenant id accepted from X-Pantheon-Tenant. */
#define PN_GATEWAY_TENANT_MAX 48

/** Room for a tenant-scoped session id (the slot tables hold 63 characters). */
#define PN_GATEWAY_SESSION_KEY_MAX 64

/**
 * Turn gateway mode on with `key` (copied; the caller may clear its copy).
 * Not thread-safe: call once at startup.
 */
void pn_gateway_enable(const pn_gateway_key *key);

/** Turn gateway mode off and wipe the key (tests). Not thread-safe. */
void pn_gateway_disable(void);

/** Is gateway mode on? Thread-safe: yes (set before serving). */
bool pn_gateway_enabled(void);

/** Does this Authorization header carry the gateway key? Thread-safe: yes. */
bool pn_gateway_authorized(const char *authorization);

/**
 * May the API listener route this request in gateway mode? Only
 * POST /v1/chat/completions, GET /v1/models, GET /health/live and
 * GET /health/ready (a query string is ignored).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
bool pn_gateway_route_allowed(const char *method, const char *path);

/**
 * Is `tenant` a usable tenant id: 1..PN_GATEWAY_TENANT_MAX characters of
 * [A-Za-z0-9._-]? (No ':' — it separates tenant from key, so two tenants can
 * never form one session family.)
 */
bool pn_gateway_tenant_valid(const char *tenant);

/**
 * The tenant-scoped session id for a client key: "<tenant>:<key>". A key that
 * would not fit the slot tables is replaced by a 64-bit hash of it
 * ("<tenant>:#<16 hex>") so two long keys never collide by truncation.
 * @return `out`, or NULL (anonymous turn) when tenant is invalid/absent or the
 *         client sent no key
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
const char *pn_gateway_session_key(const char *tenant, const char *client_key,
                                   char *out, size_t cap);

/** Set (or clear, NULL) this connection thread's gateway tenant. */
void pn_gateway_set_thread_tenant(const char *tenant);

/** This connection thread's gateway tenant, or NULL. */
const char *pn_gateway_thread_tenant(void);

#endif /* PANTHEON_SERVE_GATEWAY_MODE_H */
