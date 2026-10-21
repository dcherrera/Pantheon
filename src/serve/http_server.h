/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file http_server.h
 * @brief Minimal HTTP/1.1 server over POSIX sockets — just enough to host the
 *        OpenAI-compatible endpoints. Supports a plain JSON response path and a
 *        chunked Server-Sent-Events path. Each request/response is
 *        per-connection with no shared state, so serve_command.c drives one
 *        handler per connection thread concurrently.
 *
 * JACLibc was evaluated first (global instruction / plan hint) but its socket
 * surface (include/core/jsio.h) is a Node-style JS runtime abstraction, not a
 * drop-in POSIX helper, and pulling in a whole alternate libc under Pantheon's
 * Accelerate build is a heavy dep. Per the plan, minimal POSIX sockets are used.
 *
 * @status Serves GET/POST with Content-Length bodies; SSE streaming works.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_HTTP_SERVER_H
#define PN_SERVE_HTTP_SERVER_H

#include <stddef.h>

/* A parsed HTTP request. Buffers are owned; release with hs_request_free(). */
typedef struct {
    char  *method;     /* "GET", "POST", ... (owned) */
    char  *path;       /* request target, e.g. "/v1/chat/completions" (owned) */
    char  *body;       /* request body bytes (owned, NUL-terminated), or NULL */
    size_t body_len;   /* body length in bytes */
    char  *session_id; /* optional X-Session-Id header value (owned), or NULL —
                        * the prefix-cache session key (PC0.1). */
} hs_request;

/**
 * Bind and listen on host:port (host is a dotted-quad, e.g. "127.0.0.1").
 * @return listening socket fd, or -1 on error (message logged to stderr).
 */
int hs_listen(const char *host, int port);

/** Accept one client connection. Returns the client fd, or -1 on error. */
int hs_accept(int listen_fd);

/**
 * Read and parse one request from a client fd. Reads headers then the body per
 * Content-Length. Returns 0 on success (fills *req), -1 on error/EOF.
 * On success the caller must hs_request_free(req).
 */
int hs_read_request(int client_fd, hs_request *req);

/** Free the owned buffers of a request. NULL-safe. */
void hs_request_free(hs_request *req);

/** Send a complete JSON response with the given HTTP status code. */
void hs_send_json(int client_fd, int status, const char *json_body);

/**
 * Send an OpenAI-shaped error response:
 *   {"error":{"message":..,"type":..,"code":null}}
 */
void hs_send_error(int client_fd, int status, const char *type,
                   const char *message);

/**
 * Like hs_send_error but injects `extra_headers` (may be NULL) into the
 * response before the terminal blank line.  Each header must be in the form
 * "Name: value\r\n".  Used for Retry-After on 429/503 responses.
 */
void hs_send_error_ex(int client_fd, int status, const char *type,
                      const char *message, const char *extra_headers);

/** Begin a Server-Sent-Events stream (writes 200 + SSE headers). */
void hs_begin_sse(int client_fd);

/** Send one SSE event: "data: <json>\n\n". */
void hs_send_sse(int client_fd, const char *json_data);

/** Send an SSE comment line (": <text>\n\n"). Clients ignore it as an event but
 * it is body bytes, so a stall watchdog that counts bytes sees the stream alive.
 * `text` must not contain a newline. */
void hs_send_sse_comment(int client_fd, const char *text);

/** Send the terminal "data: [DONE]\n\n" SSE event. */
void hs_end_sse(int client_fd);

/** Close a client connection. */
void hs_close(int client_fd);

/**
 * Non-blocking check that the client is still connected: 1 if open, 0 if the
 * peer has closed (an orderly shutdown seen via a zero-length MSG_PEEK) or the
 * fd is dead. Used to abort a long prefill the instant the client disconnects,
 * releasing the model slot instead of finishing a response nobody will read.
 */
int hs_client_alive(int client_fd);

#endif /* PN_SERVE_HTTP_SERVER_H */
