/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file http_server.c
 * @brief Implementation of the minimal HTTP/1.1 server (see http_server.h).
 *        Uses BSD sockets; SIGPIPE is ignored so a client disconnect mid-stream
 *        cannot kill the server. Each request/response is per-connection and
 *        carries no shared state, so the accept loop in serve_command.c drives
 *        one of these handlers per connection thread concurrently.
 *
 * @status Complete for the serve layer.
 * @issues None
 * @todo None
 */

#include "http_server.h"

#include <arpa/inet.h>
/* Hard ceiling on a request body (Content-Length). The largest legitimate body
 * is a /session/restore blob: the biggest session snapshot on the fleet is
 * ~130 MB (ornith), ~173 MB base64'd, plus the JSON envelope — comfortably
 * under this. The cap exists so a hostile/broken client cannot declare a
 * multi-GB Content-Length and drive the connection thread to malloc it and OOM
 * the 7.8 GB host (a DoS with no per-request memory admission on this path).
 * Anything over the cap is refused before a single body byte is allocated. */
#define HS_MAX_BODY_BYTES ((size_t)512 * 1024 * 1024)
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

/* Write all `len` bytes, retrying short/interrupted writes. Returns 0 on
 * success, -1 if the peer went away. */
static int write_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        off += (size_t)n;
    }
    return 0;
}

int hs_listen(const char *host, int port)
{
    /* A broken client connection must not raise SIGPIPE and kill the daemon. */
    signal(SIGPIPE, SIG_IGN);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "serve: socket() failed: %s\n", strerror(errno));
        return -1;
    }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "serve: bad host address '%s'\n", host);
        close(fd);
        return -1;
    }

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "serve: bind(%s:%d) failed: %s\n", host, port,
                strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 16) < 0) {
        fprintf(stderr, "serve: listen() failed: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

int hs_accept(int listen_fd)
{
    int cfd = accept(listen_fd, NULL, NULL);
    if (cfd < 0 && errno != EINTR)
        fprintf(stderr, "serve: accept() failed: %s\n", strerror(errno));
    return cfd;
}

/* Case-insensitive search for a header value; returns a pointer into `headers`
 * at the start of the value, or NULL. Value ends at CRLF. */
static const char *find_header(const char *headers, const char *name)
{
    size_t nlen = strlen(name);
    const char *p = headers;
    while (*p) {
        const char *line_end = strstr(p, "\r\n");
        if (!line_end) break;
        if ((size_t)(line_end - p) > nlen && strncasecmp(p, name, nlen) == 0 &&
            p[nlen] == ':') {
            const char *v = p + nlen + 1;
            while (*v == ' ' || *v == '\t') v++;
            return v;
        }
        p = line_end + 2;
    }
    return NULL;
}

int hs_read_request(int client_fd, hs_request *req)
{
    memset(req, 0, sizeof(*req));

    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    if (!buf) return -1;

    /* Read until we have the full header block (terminated by CRLFCRLF). */
    char *header_end = NULL;
    while (!header_end) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *grown = realloc(buf, cap);
            if (!grown) { free(buf); return -1; }
            buf = grown;
        }
        ssize_t n = read(client_fd, buf + len, cap - len - 1);
        if (n <= 0) { free(buf); return -1; }
        len += (size_t)n;
        buf[len] = '\0';
        header_end = strstr(buf, "\r\n\r\n");
    }

    /* Parse the request line: METHOD SP TARGET SP VERSION. */
    char *sp1 = strchr(buf, ' ');
    if (!sp1) { free(buf); return -1; }
    char *sp2 = strchr(sp1 + 1, ' ');
    if (!sp2) { free(buf); return -1; }
    req->method = strndup(buf, (size_t)(sp1 - buf));
    req->path = strndup(sp1 + 1, (size_t)(sp2 - sp1 - 1));

    /* Optional prefix-cache session key (PC0.1): X-Session-Id header. Copied
     * out before `buf` is freed. Absent → NULL (anonymous / full prefill). */
    const char *sid = find_header(buf, "X-Session-Id");
    if (sid) {
        const char *e = strstr(sid, "\r\n");
        size_t slen = e ? (size_t)(e - sid) : strlen(sid);
        req->session_id = strndup(sid, slen);
    }
    const char *pin = find_header(buf, "X-Pantheon-Replica");
    if (pin) {
        const char *e = strstr(pin, "\r\n");
        req->replica = strndup(pin, e ? (size_t)(e - pin) : strlen(pin));
    }
    const char *auth = find_header(buf, "Authorization");
    if (auth) {
        const char *e = strstr(auth, "\r\n");
        req->authorization = strndup(auth, e ? (size_t)(e - auth) : strlen(auth));
    }

    /* Body: read Content-Length bytes past the header terminator. */
    size_t body_offset = (size_t)(header_end - buf) + 4;
    size_t content_length = 0;
    const char *cl = find_header(buf, "Content-Length");
    if (cl) content_length = (size_t)strtoul(cl, NULL, 10);

    /* Refuse an oversized body BEFORE allocating it: a hostile Content-Length
     * would otherwise malloc gigabytes on this connection thread and OOM the
     * host. Answer a clean 413 so a legit-but-too-large client gets a reason. */
    if (content_length > HS_MAX_BODY_BYTES) {
        hs_send_error(client_fd, 413, "payload_too_large",
                      "request body exceeds the server limit");
        free(buf);
        return -1;
    }

    if (content_length > 0) {
        req->body = malloc(content_length + 1);
        if (!req->body) { hs_request_free(req); free(buf); return -1; }
        size_t have = len - body_offset;
        if (have > content_length) have = content_length;
        memcpy(req->body, buf + body_offset, have);
        size_t got = have;
        while (got < content_length) {
            ssize_t n = read(client_fd, req->body + got, content_length - got);
            if (n <= 0) { hs_request_free(req); free(buf); return -1; }
            got += (size_t)n;
        }
        req->body[content_length] = '\0';
        req->body_len = content_length;
    }

    free(buf);
    return (req->method && req->path) ? 0 : -1;
}

void hs_request_free(hs_request *req)
{
    if (!req) return;
    free(req->method);
    free(req->path);
    free(req->body);
    free(req->session_id);
    free(req->replica);
    free(req->authorization);
    memset(req, 0, sizeof(*req));
}

/* Human-readable reason phrase for the status codes the serve layer emits. */
static const char *reason_phrase(int status)
{
    switch (status) {
        case 200: return "OK";
        case 202: return "Accepted";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        case 507: return "Insufficient Storage";
        default:  return "OK";
    }
}

/* This thread's headers for every response it sends (hs_set_thread_response_headers):
 * one connection thread answers one request, so they belong to that request. */
static _Thread_local char t_response_headers[HS_THREAD_HEADERS_MAX];

void hs_set_thread_response_headers(const char *headers)
{
    if (!headers || strlen(headers) >= sizeof(t_response_headers))
        headers = "";
    snprintf(t_response_headers, sizeof(t_response_headers), "%s", headers);
}

const char *hs_thread_response_headers(void)
{
    return t_response_headers;
}

/* This thread's principal (hs_set_thread_principal): who the request is from. */
static _Thread_local char t_principal[HS_PRINCIPAL_MAX];
static _Thread_local int  t_principal_admin;

void hs_set_thread_principal(const char *principal, int admin)
{
    if (!principal || strlen(principal) >= sizeof(t_principal))
        principal = "";
    snprintf(t_principal, sizeof(t_principal), "%s", principal);
    t_principal_admin = principal[0] && admin;
}

const char *hs_thread_principal(void)
{
    return t_principal[0] ? t_principal : NULL;
}

int hs_thread_principal_admin(void)
{
    return t_principal_admin;
}

void hs_send_json(int client_fd, int status, const char *json_body)
{
    size_t blen = json_body ? strlen(json_body) : 0;
    char header[256 + HS_THREAD_HEADERS_MAX];
    int hn = snprintf(header, sizeof(header),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Connection: close\r\n%s\r\n",
                      status, reason_phrase(status), blen, t_response_headers);
    if (hn < 0 || (size_t)hn >= sizeof(header)) return;
    if (write_all(client_fd, header, (size_t)hn) != 0) return;
    if (blen) write_all(client_fd, json_body, blen);
}

void hs_send_error(int client_fd, int status, const char *type,
                   const char *message)
{
    hs_send_error_ex(client_fd, status, type, message, NULL);
}

void hs_send_error_ex(int client_fd, int status, const char *type,
                      const char *message, const char *extra_headers)
{
    /* Small, hand-built to avoid a json_writer dependency in the socket layer. */
    char body[1024];
    /* Escape the message minimally (quotes/backslashes) into a temp buffer. */
    char esc[512];
    size_t j = 0;
    for (const char *s = message; *s && j < sizeof(esc) - 2; s++) {
        if (*s == '"' || *s == '\\') esc[j++] = '\\';
        esc[j++] = *s;
    }
    esc[j] = '\0';
    int bn = snprintf(body, sizeof(body),
                      "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"code\":null}}",
                      esc, type);
    if (bn < 0) return;

    /* When extra headers are provided (e.g. "Retry-After: 5\r\n"), inject
     * them before the blank line that ends the response headers. */
    if (!extra_headers || !extra_headers[0]) {
        hs_send_json(client_fd, status, body);
        return;
    }

    size_t blen = (size_t)bn;
    char header[384 + HS_THREAD_HEADERS_MAX];
    int hn = snprintf(header, sizeof(header),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Connection: close\r\n"
                      "%s%s\r\n",        /* extra + thread headers, blank line */
                      status, reason_phrase(status), blen, extra_headers,
                      t_response_headers);
    if (hn < 0 || (size_t)hn >= sizeof(header)) return;
    if (write_all(client_fd, header, (size_t)hn) != 0) return;
    write_all(client_fd, body, blen);
}

int hs_sse_send_timeout_ms(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_SSE_SEND_TIMEOUT_MS");
        char *end = NULL;
        long v = env && *env ? strtol(env, &end, 10) : -1;
        cached = (env && *env && end && *end == '\0' && v >= 0 && v <= 86400000L)
               ? (int)v : 60000;
    }
    return cached;
}

int hs_begin_sse(int client_fd)
{
    /* A peer that stops reading without closing (a vanished host, a frozen
     * client) would otherwise block a write forever, and the turn holding a
     * decode slot with it. With a send timeout the write fails and the turn
     * stops like any other disconnect. */
    int ms = hs_sse_send_timeout_ms();
    if (ms > 0) {
        struct timeval tv = { .tv_sec = ms / 1000, .tv_usec = (ms % 1000) * 1000 };
        if (setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
            fprintf(stderr, "serve: SO_SNDTIMEO on a stream failed: %s\n", strerror(errno));
    }
    char header[256 + HS_THREAD_HEADERS_MAX];
    int hn = snprintf(header, sizeof(header),
                      "HTTP/1.1 200 OK\r\n"
                      "Content-Type: text/event-stream\r\n"
                      "Cache-Control: no-cache\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Connection: close\r\n%s\r\n", t_response_headers);
    if (hn <= 0 || (size_t)hn >= sizeof(header))
        return -1;
    return write_all(client_fd, header, (size_t)hn);
}

int hs_send_sse(int client_fd, const char *json_data)
{
    if (write_all(client_fd, "data: ", 6) != 0 ||
        write_all(client_fd, json_data, strlen(json_data)) != 0)
        return -1;
    return write_all(client_fd, "\n\n", 2);
}

int hs_send_sse_comment(int client_fd, const char *text)
{
    if (write_all(client_fd, ": ", 2) != 0 ||
        write_all(client_fd, text, strlen(text)) != 0)
        return -1;
    return write_all(client_fd, "\n\n", 2);
}

int hs_end_sse(int client_fd)
{
    /* sizeof-1, not a hand-counted 13 — "data: [DONE]\n\n" is 14 bytes; writing 13
     * dropped the terminating blank line of the final SSE event. */
    static const char done[] = "data: [DONE]\n\n";
    return write_all(client_fd, done, sizeof(done) - 1);
}

void hs_close(int client_fd)
{
    close(client_fd);
}

int hs_client_alive(int client_fd)
{
    if (client_fd < 0)
        return 0;
#ifdef POLLRDHUP
    /* The peer's FIN shows here even when unread bytes are still queued, which
     * the peek below would read as "open". */
    struct pollfd pfd = { .fd = client_fd, .events = POLLIN | POLLRDHUP };
    if (poll(&pfd, 1, 0) > 0 && (pfd.revents & (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)))
        return 0;
#endif
    /* Peek one byte without blocking: recv()==0 is an orderly peer close (the
     * client aborted / disconnected); EAGAIN is open-but-idle (the normal case
     * while we stream); >0 is buffered data (still open). */
    char probe;
    ssize_t r = recv(client_fd, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
    if (r == 0)
        return 0;
    if (r < 0)
        return (errno == EAGAIN || errno == EWOULDBLOCK) ? 1 : 0;
    return 1;
}

static hs_auth_fn g_auth_fn;
static void      *g_auth_user;

void hs_set_auth_fn(hs_auth_fn fn, void *user)
{
    g_auth_fn = fn;
    g_auth_user = user;
}

int hs_auth_check(int client_fd, const hs_request *req)
{
    return g_auth_fn ? g_auth_fn(client_fd, req, g_auth_user) : 0;
}
