/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file http_client.c
 * @brief Implementation of the minimal proxy HTTP client (see http_client.h).
 *        Blocking sockets with a non-blocking connect + select() timeout, then
 *        a straight relay of the response body to a sink. Headers are parsed
 *        only far enough to find the status code and the end of the header
 *        block; everything after CRLFCRLF is streamed through untouched, so SSE
 *        chunks (and chunked-transfer bytes) pass through unbuffered.
 *
 *        The router hands the upstream the same body it received and forwards
 *        raw bytes back, so it does not need to understand chunked framing — it
 *        relays exactly what serve emits (serve uses Connection: close, so EOF
 *        marks the end).
 *
 * @status G2.
 * @issues None
 * @todo None
 */

#include "http_client.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

/* Portable memmem (avoid relying on the GNU/BSD extension across platforms). */
static void *memmem_local(const void *hay, size_t hlen, const void *needle,
                          size_t nlen)
{
    if (nlen == 0) return (void *)hay;
    if (hlen < nlen) return NULL;
    const char *h = hay;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(h + i, needle, nlen) == 0)
            return (void *)(h + i);
    return NULL;
}

/* Parse "http://host:port" into host + port. Returns 0 on success. Port
 * defaults to 80 if absent. A trailing path on the endpoint is ignored. */
static int parse_endpoint(const char *endpoint, char *host, size_t host_cap,
                          int *port)
{
    const char *p = endpoint;
    if (strncmp(p, "http://", 7) == 0) p += 7;
    else if (strncmp(p, "https://", 8) == 0) return -1; /* TLS not handled here */

    /* host is up to ':' or '/' or end. */
    size_t i = 0;
    while (*p && *p != ':' && *p != '/') {
        if (i + 1 >= host_cap) return -1;
        host[i++] = *p++;
    }
    host[i] = '\0';
    if (i == 0) return -1;

    *port = 80;
    if (*p == ':') {
        p++;
        int v = 0;
        if (*p < '0' || *p > '9') return -1;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        *port = v;
    }
    return 0;
}

/* Connect with a timeout. Returns fd (>=0) or -1. */
static int connect_timeout(const char *host, int port, int connect_ms)
{
    if (connect_ms <= 0) connect_ms = 2000;

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res)
        return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) {
            fcntl(fd, F_SETFL, flags);           /* back to blocking */
            break;
        }
        if (errno == EINPROGRESS) {
            fd_set wf;
            FD_ZERO(&wf);
            FD_SET(fd, &wf);
            struct timeval tv = { connect_ms / 1000, (connect_ms % 1000) * 1000 };
            rc = select(fd + 1, NULL, &wf, NULL, &tv);
            if (rc > 0) {
                int soerr = 0;
                socklen_t slen = sizeof(soerr);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen);
                if (soerr == 0) {
                    fcntl(fd, F_SETFL, flags);   /* connected — blocking */
                    break;
                }
            }
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static int write_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        off += (size_t)n;
    }
    return 0;
}

/* Core: send `method path` + body to endpoint, then relay the response body to
 * `sink`. All the public entry points funnel through this. */
static hc_result hc_do(const char *method, const char *endpoint,
                       const char *path, const char *body, size_t body_len,
                       const char *extra_hdrs, int connect_ms,
                       hc_sink_fn sink, void *user, int *status_out)
{
    if (status_out) *status_out = 0;
    signal(SIGPIPE, SIG_IGN);

    char host[256];
    int port = 0;
    if (parse_endpoint(endpoint, host, sizeof(host), &port) != 0)
        return HC_CONNECT_FAILED;

    int fd = connect_timeout(host, port, connect_ms);
    if (fd < 0) return HC_CONNECT_FAILED;

    /* Build + send the request. Connection: close so the response ends at EOF
     * (no chunked-decode needed on our side; we relay raw bytes). */
    char header[1024];
    int hn = snprintf(header, sizeof(header),
                      "%s %s HTTP/1.1\r\n"
                      "Host: %s:%d\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Accept: text/event-stream\r\n"
                      "Connection: close\r\n"
                      "%s"
                      "\r\n",
                      method, path, host, port, body_len,
                      (extra_hdrs && extra_hdrs[0]) ? extra_hdrs : "");
    if (hn < 0 || (size_t)hn >= sizeof(header)) { close(fd); return HC_UPSTREAM_ERROR; }
    if (write_all(fd, header, (size_t)hn) != 0) { close(fd); return HC_UPSTREAM_ERROR; }
    if (body_len && write_all(fd, body, body_len) != 0) { close(fd); return HC_UPSTREAM_ERROR; }

    /* Read the response. Accumulate until we have the header block, parse the
     * status line, then relay everything past CRLFCRLF to the sink and keep
     * relaying subsequent reads until EOF. */
    char   buf[8192];
    char  *acc = NULL;
    size_t acc_len = 0, acc_cap = 0;
    int    header_done = 0;
    int    status = 0;
    int    relayed_any = 0;
    hc_result result = HC_UPSTREAM_ERROR;

    for (;;) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            result = relayed_any ? HC_MIDSTREAM_FAILED : HC_UPSTREAM_ERROR;
            break;
        }
        if (n == 0) {
            /* EOF. If we never finished headers, upstream error. Otherwise the
             * body ended cleanly (Connection: close). */
            result = header_done ? HC_OK : HC_UPSTREAM_ERROR;
            break;
        }

        if (!header_done) {
            /* Append to the header accumulator and look for CRLFCRLF. */
            if (acc_len + (size_t)n + 1 > acc_cap) {
                size_t ncap = (acc_cap ? acc_cap * 2 : 8192);
                while (ncap < acc_len + (size_t)n + 1) ncap *= 2;
                char *g = realloc(acc, ncap);
                if (!g) { free(acc); close(fd); return HC_UPSTREAM_ERROR; }
                acc = g;
                acc_cap = ncap;
            }
            memcpy(acc + acc_len, buf, (size_t)n);
            acc_len += (size_t)n;
            acc[acc_len] = '\0';

            char *hend = memmem_local(acc, acc_len, "\r\n\r\n", 4);
            if (!hend) continue;   /* need more header bytes */

            /* Parse status line: "HTTP/1.1 <code> ...". */
            if (acc_len >= 12 && strncmp(acc, "HTTP/", 5) == 0) {
                const char *sp = memchr(acc, ' ', acc_len);
                if (sp) status = atoi(sp + 1);
            }
            if (status_out) *status_out = status;
            header_done = 1;

            /* Relay any body bytes already read past the header terminator. */
            size_t body_off = (size_t)(hend - acc) + 4;
            size_t have = acc_len - body_off;
            if (have) {
                relayed_any = 1;
                if (sink(acc + body_off, have, user) != 0) {
                    result = HC_RELAY_ABORTED;
                    break;
                }
            }
            free(acc);
            acc = NULL;
            continue;
        }

        /* Header already done — straight relay. */
        relayed_any = 1;
        if (sink(buf, (size_t)n, user) != 0) { result = HC_RELAY_ABORTED; break; }
    }

    free(acc);
    close(fd);
    return result;
}

hc_result hc_post_stream(const char *endpoint, const char *path,
                         const char *body, size_t body_len,
                         const char *extra_hdrs, int connect_ms,
                         hc_sink_fn sink, void *user, int *status_out)
{
    return hc_do("POST", endpoint, path, body, body_len, extra_hdrs,
                 connect_ms, sink, user, status_out);
}

/* Discarding sink for hc_request_simple — we only care about the status code. */
static int discard_sink(const char *bytes, size_t len, void *user)
{
    (void)bytes; (void)len; (void)user;
    return 0;
}

hc_result hc_request_simple(const char *method, const char *endpoint,
                            const char *path, const char *body, size_t body_len,
                            int connect_ms, int *status_out)
{
    return hc_do(method, endpoint, path, body, body_len, NULL,
                 connect_ms, discard_sink, NULL, status_out);
}
