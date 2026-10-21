/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file http_client.h
 * @brief Minimal blocking HTTP/1.1 client for the dispatch router's reverse
 *        proxy (Phase G2). Just enough to open a TCP connection to an upstream
 *        serve, send a POST with a JSON body, and stream the response back
 *        chunk-by-chunk through a sink callback so SSE passes through UNBUFFERED.
 *
 *        Only `http://host:port` endpoints are supported (the local + WG
 *        positions are plain HTTP; TLS is terminated at Caddy in the VPS
 *        position, not here). No external deps — POSIX sockets only.
 *
 * @status G2: connect + POST + streamed response relay with a connect-timeout
 *         and a first-byte distinction for failover.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_HTTP_CLIENT_H
#define PN_SERVE_HTTP_CLIENT_H

#include <stddef.h>

/* Sink for streamed response bytes. Called repeatedly with raw body bytes as
 * they arrive from the upstream (headers already stripped). Return 0 to keep
 * going, non-zero to abort the relay (e.g. the downstream client went away). */
typedef int (*hc_sink_fn)(const char *bytes, size_t len, void *user);

/* Outcome of an hc_post_stream call — lets the router decide whether to fail
 * over (safe only if NOTHING was relayed to the client yet). */
typedef enum {
    HC_OK = 0,             /* completed; body fully relayed */
    HC_CONNECT_FAILED,     /* could not connect / DNS / bad endpoint — pre-relay */
    HC_UPSTREAM_ERROR,     /* connected but no valid response before any body — pre-relay */
    HC_RELAY_ABORTED,      /* the sink returned non-zero (downstream closed) — post-relay */
    HC_MIDSTREAM_FAILED,   /* upstream dropped AFTER we began relaying body — NOT safe to retry */
} hc_result;

/**
 * POST `body` to `endpoint`+`path` and stream the response body to `sink`.
 *
 * @param endpoint     upstream base, e.g. "http://127.0.0.1:8201" (no trailing /)
 * @param path         request target, e.g. "/v1/chat/completions"
 * @param body         request body bytes (JSON), or NULL
 * @param body_len     length of body
 * @param extra_hdrs   NULL, or additional "Name: value\r\n" header lines to send
 * @param connect_ms   connect timeout in milliseconds (<=0 ⇒ a sane default)
 * @param sink         called with response body bytes as they arrive
 * @param user         opaque passed to sink
 * @param status_out   if non-NULL, receives the upstream HTTP status code (or 0)
 *
 * @return an hc_result. HC_CONNECT_FAILED / HC_UPSTREAM_ERROR mean nothing was
 *         relayed yet (safe to fail over to another instance).
 */
hc_result hc_post_stream(const char *endpoint, const char *path,
                         const char *body, size_t body_len,
                         const char *extra_hdrs, int connect_ms,
                         hc_sink_fn sink, void *user, int *status_out);

/**
 * Simple request/response helper (no streaming) used by the serve-side
 * registry client for register/heartbeat/deregister. Sends `method`
 * (e.g. "POST"/"DELETE") with a JSON body and discards the response body,
 * returning only the HTTP status via *status_out.
 *
 * @return HC_OK if a full response arrived (check *status_out for 2xx);
 *         HC_CONNECT_FAILED / HC_UPSTREAM_ERROR on transport failure.
 */
hc_result hc_request_simple(const char *method, const char *endpoint,
                            const char *path, const char *body, size_t body_len,
                            int connect_ms, int *status_out);

#endif /* PN_SERVE_HTTP_CLIENT_H */
