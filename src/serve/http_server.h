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
    char  *authorization; /* optional Authorization header value (owned), or
                           * NULL. Read only by the auth hook; never logged. */
    char  *replica;    /* optional X-Pantheon-Replica header value (owned), or
                        * NULL: the copy a new session asks to bind to. */
    char  *tenant;     /* optional X-Pantheon-Tenant header value (owned), or
                        * NULL: the customer a trusted gateway names
                        * (gateway_mode.h; read only in gateway mode). */
} hs_request;

/* Room for hs_set_thread_response_headers. */
#define HS_THREAD_HEADERS_MAX 128
/* Room for hs_set_thread_principal (a person id). */
#define HS_PRINCIPAL_MAX      32

/**
 * Auth hook, checked by the serve before routing (Pantheon Local, A3).
 * Return 0 to let the request through; non-zero means the hook has already
 * answered the client (401/403/429) and the request must not be routed.
 * NULL hook (the default) = every request is routed, today's behaviour.
 */
typedef int (*hs_auth_fn)(int client_fd, const hs_request *req, void *user);

/**
 * Install (or clear, with NULL) the process-wide auth hook. Call once before
 * the accept loop starts.
 *
 * Thread-safe: no (set before serving)
 * Reentrant: no
 * Signal-safe: no
 */
void hs_set_auth_fn(hs_auth_fn fn, void *user);

/**
 * Run the installed auth hook on a parsed request.
 * @return 0 if the request may be routed, non-zero if it was refused (the
 *         response is already sent).
 *
 * Thread-safe: yes (the hook must be)
 * Reentrant: yes
 * Signal-safe: no
 */
int hs_auth_check(int client_fd, const hs_request *req);

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
 * Send an OpenAI-shaped error with a `code` and an optional `param`:
 *   {"error":{"message":..,"type":..,"param":..|null,"code":..}}
 * The shape OpenAI SDKs and gateways key on (e.g. code "unsupported_parameter").
 */
void hs_send_openai_error(int client_fd, int status, const char *type,
                          const char *code, const char *param,
                          const char *message);

/**
 * Like hs_send_error but injects `extra_headers` (may be NULL) into the
 * response before the terminal blank line.  Each header must be in the form
 * "Name: value\r\n".  Used for Retry-After on 429/503 responses.
 */
void hs_send_error_ex(int client_fd, int status, const char *type,
                      const char *message, const char *extra_headers);

/**
 * Set headers ("Name: value\r\n" lines; NULL or "" clears) that every response
 * this thread sends from now on carries: hs_send_json, hs_send_error(_ex) and
 * hs_begin_sse. A connection thread answers one request, so a handler sets them
 * once it knows them (e.g. X-Pantheon-Replica after routing) and the connection
 * loop clears them after the request. Headers longer than
 * HS_THREAD_HEADERS_MAX - 1 are refused (cleared).
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no
 */
void hs_set_thread_response_headers(const char *headers);

/** This thread's current response headers ("" when none). Thread-safe: yes. */
const char *hs_thread_response_headers(void);

/**
 * Set who this thread's request is from (Pantheon Local: the caller's person id,
 * and whether that person is an admin); NULL or "" clears. The auth hook sets it,
 * the connection loop clears it after the request; serve sessions are bound to it
 * (model_session.h). Plain `pantheon serve` never sets one. Longer than
 * HS_PRINCIPAL_MAX - 1 is refused (cleared).
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no
 */
void hs_set_thread_principal(const char *principal, int admin);

/** This thread's principal, or NULL when none. Thread-safe: yes (thread-local). */
const char *hs_thread_principal(void);

/** 1 when this thread's principal is an admin. Thread-safe: yes (thread-local). */
int hs_thread_principal_admin(void);

/**
 * The send timeout every SSE stream gets, in ms: PN_SSE_SEND_TIMEOUT_MS when set
 * (0 = none), else 60000. Read once.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int hs_sse_send_timeout_ms(void);

/**
 * Begin a Server-Sent-Events stream: sets the stream's send timeout
 * (hs_sse_send_timeout_ms, so a peer that stops reading fails a write instead
 * of blocking it forever) and writes 200 + SSE headers.
 *
 * @return 0, or -1 when the client is gone
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int hs_begin_sse(int client_fd);

/** Send one SSE event: "data: <json>\n\n". Returns 0, or -1 when the client is
 * gone (closed, reset, or not reading within the send timeout). */
int hs_send_sse(int client_fd, const char *json_data);

/** Send an SSE comment line (": <text>\n\n"). Clients ignore it as an event but
 * it is body bytes, so a stall watchdog that counts bytes sees the stream alive.
 * `text` must not contain a newline. Returns 0, or -1 when the client is gone. */
int hs_send_sse_comment(int client_fd, const char *text);

/** Send the terminal "data: [DONE]\n\n" SSE event. Returns 0, or -1 when the
 * client is gone. */
int hs_end_sse(int client_fd);

/** Close a client connection. */
void hs_close(int client_fd);

/**
 * Non-blocking check that the client is still connected: 1 if open, 0 if the
 * peer has closed (its FIN via POLLRDHUP where the platform has it, even with
 * unread bytes queued; else a zero-length MSG_PEEK) or the fd is dead. Used to abort a long prefill the instant the client disconnects,
 * releasing the model slot instead of finishing a response nobody will read.
 */
int hs_client_alive(int client_fd);

#endif /* PN_SERVE_HTTP_SERVER_H */
