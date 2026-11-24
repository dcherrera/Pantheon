/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * local_fixture_httpd.h — a tiny static HTTP/1.1 file server for the Pantheon
 * Local download tests (and tests/local/fixture_httpd.c for the on-hardware
 * scripts). GET only, `Range: bytes=N-` resume (206), and an optional cut: the
 * first GET of each file closes the connection after `cut_after` body bytes,
 * like a dropped download. One connection at a time; runs until killed.
 */

#ifndef PN_LOCAL_FIXTURE_HTTPD_H
#define PN_LOCAL_FIXTURE_HTTPD_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define FX_MAX_CUT_PATHS 32

static inline void fx_send_all(int fd, const void *buf, size_t len)
{
    const char *p = buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return;
        p += n;
        len -= (size_t)n;
    }
}

/* Serve one request on cfd. cut_seen tracks which paths were already cut. */
static inline void fx_serve_one(int cfd, const char *root, long cut_after,
                                char cut_seen[][512], int *n_seen)
{
    char req[4096], path[512], full[1100];
    ssize_t n = read(cfd, req, sizeof(req) - 1);
    if (n <= 0) return;
    req[n] = '\0';
    if (sscanf(req, "GET %511s ", path) != 1 || strstr(path, "..")) {
        const char *br = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        fx_send_all(cfd, br, strlen(br));
        return;
    }
    snprintf(full, sizeof(full), "%s%s", root, path);
    FILE *f = fopen(full, "rb");
    struct stat sb;
    if (!f || stat(full, &sb) != 0) {
        const char *nf = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        fx_send_all(cfd, nf, strlen(nf));
        if (f) fclose(f);
        return;
    }
    long start = 0;
    const char *range = strstr(req, "Range: bytes=");
    if (range) start = atol(range + 13);
    long size = (long)sb.st_size, len = size - start;
    char head[512];
    int hn = range
        ? snprintf(head, sizeof(head), "HTTP/1.1 206 Partial Content\r\nContent-Length: %ld\r\n"
                   "Content-Range: bytes %ld-%ld/%ld\r\nConnection: close\r\n\r\n",
                   len, start, size - 1, size)
        : snprintf(head, sizeof(head), "HTTP/1.1 200 OK\r\nContent-Length: %ld\r\n"
                   "Accept-Ranges: bytes\r\nConnection: close\r\n\r\n", size);
    fx_send_all(cfd, head, (size_t)hn);
    long limit = len;
    int seen = 0;
    for (int i = 0; i < *n_seen; i++) seen |= strcmp(cut_seen[i], path) == 0;
    if (cut_after > 0 && !seen && *n_seen < FX_MAX_CUT_PATHS) {
        snprintf(cut_seen[(*n_seen)++], 512, "%s", path);
        if (cut_after < limit) limit = cut_after;
    }
    fseek(f, start, SEEK_SET);
    char buf[65536];
    while (limit > 0) {
        size_t want = limit < (long)sizeof(buf) ? (size_t)limit : sizeof(buf);
        size_t got = fread(buf, 1, want, f);
        if (got == 0) break;
        fx_send_all(cfd, buf, got);
        limit -= (long)got;
    }
    fclose(f);
}

/* Serve root on 127.0.0.1:port (or 0.0.0.0 when any_addr) forever. */
static inline int fx_httpd_run(int port, const char *root, long cut_after, int any_addr)
{
    static char cut_seen[FX_MAX_CUT_PATHS][512];
    int n_seen = 0, one = 1;
    /* A client that hangs up mid-body (a cancelled download) must not kill the server. */
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(any_addr ? INADDR_ANY : INADDR_LOOPBACK);
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(fd, 8) != 0) {
        perror("fixture_httpd");
        return 1;
    }
    for (;;) {
        int cfd = accept(fd, NULL, NULL);
        if (cfd < 0) continue;
        fx_serve_one(cfd, root, cut_after, cut_seen, &n_seen);
        close(cfd);
    }
}

#endif /* PN_LOCAL_FIXTURE_HTTPD_H */
