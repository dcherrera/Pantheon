/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_hook_adopt.c — people registry P2.1: the serve's local route hook
 * may answer 2 (handled, fd adopted). A forked `serve --local` child runs with a
 * test hook: GET /test/adopt keeps its fd and signals the parent over a pipe;
 * GET /test/release answers that kept fd later. The parent checks the adopted
 * connection stays open until then, the late answer arrives, plain hook routes
 * and the serve's own routes are unchanged, and host waits are not logged.
 * Run output lives under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include <poll.h>
#include <pthread.h>

#include "../src/serve/serve_command.h"

#define HOOK_DIR "build/test_local/hook_adopt"

static int g_signal_fd = -1;          /* child: write end of the "adopted" pipe */
static int g_kept_fd = -1;            /* child: the adopted connection */
static pthread_mutex_t g_kept_lock = PTHREAD_MUTEX_INITIALIZER;

static int test_route(int cfd, mr_registry *reg, const hs_request *req, void *user)
{
    (void)reg;
    (void)user;
    if (strcmp(req->path, "/test/adopt") == 0) {
        pthread_mutex_lock(&g_kept_lock);
        g_kept_fd = cfd;
        pthread_mutex_unlock(&g_kept_lock);
        if (write(g_signal_fd, "a", 1) != 1)
            fprintf(stderr, "hook: signal write failed\n");
        return 2;
    }
    if (strcmp(req->path, "/test/release") == 0) {
        pthread_mutex_lock(&g_kept_lock);
        int fd = g_kept_fd;
        g_kept_fd = -1;
        pthread_mutex_unlock(&g_kept_lock);
        if (fd >= 0) {
            hs_send_json(fd, 200, "{\"late\":true}");
            hs_close(fd);
        }
        hs_send_json(cfd, fd >= 0 ? 200 : 409, "{\"released\":true}");
        return 1;
    }
    if (strcmp(req->path, "/test/plain") == 0 ||
        strncmp(req->path, "/local/hosts/c-0123456789ab/wait", 32) == 0) {
        hs_send_json(cfd, 200, "{\"plain\":true}");
        return 1;
    }
    return 0;
}

static pid_t spawn_serve(int port, int signal_w)
{
    char port_s[16];

    snprintf(port_s, sizeof(port_s), "%d", port);
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        pn_serve_local_hooks hooks = { .route = test_route };
        char *argv[] = { "serve", "--local", "--host", "127.0.0.1", "--port", port_s, NULL };
        FILE *f = freopen(HOOK_DIR "/serve.log", "w", stderr);
        (void)f;
        setvbuf(stderr, NULL, _IONBF, 0);   /* _exit below skips stdio flushes */
        g_signal_fd = signal_w;
        pn_serve_set_local_hooks(&hooks);
        _exit(pn_serve_main(6, argv));
    }
    if (pid > 0 && lt_wait_port(port, 20000) != 0) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        return -1;
    }
    return pid;
}

/* Connect to the box and send a body-less GET; the fd, or -1. */
static int open_get(int port, const char *path)
{
    struct sockaddr_in a = {0};
    char head[256];
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int n = snprintf(head, sizeof(head), "GET %s HTTP/1.1\r\nHost: x\r\n\r\n", path);

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

/* Read until EOF (or 5 s of silence) into out. */
static void read_all(int fd, char *out, size_t cap)
{
    size_t len = 0;
    struct pollfd p = { .fd = fd, .events = POLLIN };

    while (len + 1 < cap && poll(&p, 1, 5000) > 0) {
        ssize_t n = read(fd, out + len, cap - len - 1);
        if (n <= 0)
            break;
        len += (size_t)n;
    }
    out[len] = '\0';
}

static int file_has(const char *path, const char *needle)
{
    char buf[65536];
    FILE *f = fopen(path, "r");
    size_t n = f ? fread(buf, 1, sizeof(buf) - 1, f) : 0;

    if (f)
        fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

int main(void)
{
    int port = lt_pick_port(41), pipefd[2] = { -1, -1 };
    char got[4096], *body = NULL, sig = 0;
    pid_t pid = -1;

    lt_clean_dir(HOOK_DIR);
    lt_make_dir(HOOK_DIR);
    if (pipe(pipefd) != 0 || (pid = spawn_serve(port, pipefd[1])) <= 0) {
        fprintf(stderr, "could not start the serve\n");
        return 1;
    }

    printf("adopt: a hook answer of 2 keeps the connection open\n");
    int held = open_get(port, "/test/adopt");
    struct pollfd ps = { .fd = pipefd[0], .events = POLLIN };
    CHECK(held >= 0, "connect for /test/adopt");
    CHECK(poll(&ps, 1, 5000) == 1 && read(pipefd[0], &sig, 1) == 1, "the hook adopted the fd");
    /* The hook has returned 2; a serve that still closed the fd would hand the
     * client EOF within microseconds. 300 ms of silence = still open. */
    struct pollfd ph = { .fd = held, .events = POLLIN };
    CHECK(poll(&ph, 1, 300) == 0, "adopted connection still open after the hook returned");

    printf("adopt: the hook answers the kept fd later\n");
    CHECK(lt_http(port, "GET", "/test/release", NULL, NULL, &body) == 200, "release answers 200");
    free(body);
    read_all(held, got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 200") && strstr(got, "{\"late\":true}"),
          "the late answer reached the adopted connection");
    close(held);

    printf("adopt: answers of 1 and 0 are unchanged\n");
    CHECK(lt_http(port, "GET", "/test/plain", NULL, NULL, &body) == 200 &&
          strstr(body, "plain"), "a hook answer of 1 still answers and closes");
    free(body);
    CHECK(lt_http(port, "GET", "/", NULL, NULL, &body) == 200 && strstr(body, "pantheon"),
          "a hook answer of 0 falls through to the serve's own routes");
    free(body);
    CHECK(lt_http(port, "GET", "/local/hosts/c-0123456789ab/wait?since=3", NULL, NULL, &body) ==
          200, "the host wait route answers");
    free(body);

    lt_stop(pid);
    printf("log: host waits are not logged; other requests are\n");
    CHECK(file_has(HOOK_DIR "/serve.log", "serve: GET /test/adopt"), "a normal request is logged");
    CHECK(file_has(HOOK_DIR "/serve.log", "serve: GET /test/plain"), "a plain request is logged");
    CHECK(!file_has(HOOK_DIR "/serve.log", "/wait"), "the host wait is not logged");
    close(pipefd[0]);
    close(pipefd[1]);
    return lt_report("test_local_hook_adopt");
}
