/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * local_host_wait_support.h — helpers of the registry HTTP tests that hold a
 * host's wait (GET /local/hosts/<key>/wait) open: send it without reading,
 * read its answer later, check its shape against a fixture member, and time
 * it. Needs local_test_support.h and local_fixture_replay.h first.
 */

#ifndef PN_LOCAL_HOST_WAIT_SUPPORT_H
#define PN_LOCAL_HOST_WAIT_SUPPORT_H

#include <poll.h>

/* Send a wait without reading the answer; the socket, or -1. */
static inline int wait_open(int port, const char *secret, const char *key, long long since)
{
    struct sockaddr_in a = {0};
    char head[512];
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int n = snprintf(head, sizeof(head), "GET /local/hosts/%s/wait?since=%lld HTTP/1.1\r\n"
                     "Host: x\r\nAuthorization: Bearer %s\r\n\r\n", key, since, secret);

    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        write(fd, head, (size_t)n) != n) {
        if (fd >= 0)
            close(fd);
        return -1;
    }
    return fd;
}

/* Read a wait's answer (until EOF, or timeout_ms of silence); its status, body in out. */
static inline int wait_read(int fd, char *out, size_t cap, int timeout_ms)
{
    char resp[8192];
    size_t len = 0;
    struct pollfd p = { .fd = fd, .events = POLLIN };
    int status = 0;
    const char *b = NULL;

    while (fd >= 0 && len + 1 < sizeof(resp) && poll(&p, 1, timeout_ms) > 0) {
        ssize_t n = read(fd, resp + len, sizeof(resp) - len - 1);
        if (n <= 0)
            break;
        len += (size_t)n;
    }
    resp[len] = '\0';
    if (fd >= 0)
        close(fd);
    if (sscanf(resp, "HTTP/1.1 %d", &status) != 1)
        status = 0;
    b = strstr(resp, "\r\n\r\n");
    snprintf(out, cap, "%s", b ? b + 4 : "");
    return status;
}

/* 1 when the answer `body` has the shape of the fixture's `resp_key`. */
static inline int wait_shape(const jr_value *fx, const char *resp_key, const char *body)
{
    jr_value *got = jr_parse(body);
    char where[300] = "";
    int ok = fx_same_shape(jr_object_get(fx, resp_key), got, where, sizeof(where), resp_key);

    if (!ok)
        fprintf(stderr, "  wait answer %s: %s\n    body %s\n", resp_key, where, body);
    jr_free(got);
    return ok;
}

static inline int silent_for(int fd, int ms)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    return poll(&p, 1, ms) == 0;
}

static inline uint64_t ms_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

#endif /* PN_LOCAL_HOST_WAIT_SUPPORT_H */
