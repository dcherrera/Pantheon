/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   replica_pin.h
 * @brief  A client choosing its copy of a model. A request may name a replica
 *         ("replica": N in the body, or the X-Pantheon-Replica: N header; the
 *         body wins). A new session binds to exactly that copy, a session
 *         already bound keeps its own, and a copy that is missing or full is a
 *         4xx naming the model, N and the reason, never a quiet re-route. Every
 *         response after routing carries X-Pantheon-Replica: <serving copy>.
 *         No field = the least-busy routing of mr_acquire_session.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_REPLICA_PIN_H
#define PN_SERVE_REPLICA_PIN_H

#include <stddef.h>

#include "json_reader.h"
#include "model_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The request and response header name. */
#define PN_REPLICA_HEADER "X-Pantheon-Replica"

/**
 * The copy a request names: the body's "replica" (an integer >= 1) wins over
 * the header value (decimal >= 1). Neither present = 0.
 *
 * @param body          parsed request body (may be NULL)
 * @param header_value  X-Pantheon-Replica value (may be NULL)
 * @param replica_out   the copy, 0 = none named
 * @param err           why it is invalid
 * @return 0, or -1 (invalid value) with a reason in err
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_replica_pin_parse(const jr_value *body, const char *header_value, int *replica_out,
                         char *err, size_t errlen);

/**
 * mr_acquire_session_on for a request: on success set X-Pantheon-Replica for
 * every response this thread sends (pn_replica_pin_announce). A requested copy
 * that is not loaded is answered here (404 replica_not_found, naming the model,
 * the copy and the copies loaded). An unknown model is NOT answered: NULL with
 * route->status MR_ROUTE_NO_MODEL, and the caller sends its own not-found
 * error. Pair a non-NULL return with mr_release.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
mr_entry *pn_replica_pin_acquire(int client_fd, mr_registry *reg, const char *name,
                                 const char *sid, mr_route *route);

/**
 * Set X-Pantheon-Replica: <entry's copy> on every response this thread sends
 * (hs_set_thread_response_headers). NULL clears it.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no
 */
void pn_replica_pin_announce(const mr_entry *entry);

/**
 * A new session bound by this request and then refused before it ran (its copy
 * is full or busy): drop that binding, so the client may pin another copy with
 * the same session id and the refused session leaves no routing row behind.
 * Covers a requested copy and a least-busy one alike. No-op for a session that
 * was already bound.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_pin_undo(const mr_route *route, const mr_entry *entry, const char *sid);

/**
 * The 429 for a session refused because its copy has no free session slot. A
 * pinned route names the model, the copy and the reason; an unpinned one keeps
 * the older wording. Both carry Retry-After and the re-prefill estimate.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_replica_pin_send_full(int client_fd, const mr_route *route, const mr_entry *entry,
                              int restart_s);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_REPLICA_PIN_H */
