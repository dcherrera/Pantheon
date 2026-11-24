/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * local_test_support.h — shared helpers for the tests/test_local_*.c binaries:
 * a CHECK counter, a raw-socket HTTP client (method, path, optional bearer key,
 * body → status + body), and a forked `pantheon local` child (pn_local_main in
 * the test binary itself) with wait-for-port and SIGTERM stop. State dirs live
 * under build/test_local/ (gitignored), never /tmp.
 */

#ifndef PN_LOCAL_TEST_SUPPORT_H
#define PN_LOCAL_TEST_SUPPORT_H

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int g_lt_failed = 0;
static int g_lt_checks = 0;

#define CHECK(cond, msg) do { \
    g_lt_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_lt_failed++; \
    } \
} while (0)

static inline int lt_report(const char *name)
{
    printf("%s: %d checks, %d failures\n", name, g_lt_checks, g_lt_failed);
    return g_lt_failed ? 1 : 0;
}

/* One HTTP/1.1 request to 127.0.0.1:port. Returns the status (0 on transport
 * failure). *body_out (may be NULL) receives the heap response body. */
static inline int lt_http(int port, const char *method, const char *path, const char *key,
                   const char *body, char **body_out)
{
    if (body_out) *body_out = NULL;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return 0; }

    size_t blen = body ? strlen(body) : 0;
    char auth[256] = "";
    if (key) snprintf(auth, sizeof(auth), "Authorization: Bearer %s\r\n", key);
    char head[1024];
    int hn = snprintf(head, sizeof(head), "%s %s HTTP/1.1\r\nHost: x\r\n%s"
                      "Content-Type: application/json\r\nContent-Length: %zu\r\n\r\n",
                      method, path, auth, blen);
    if (write(fd, head, (size_t)hn) != hn || (blen && write(fd, body, blen) != (ssize_t)blen)) {
        close(fd);
        return 0;
    }
    size_t cap = 65536, len = 0;
    char *resp = malloc(cap);
    for (;;) {
        if (len + 1 >= cap) { cap *= 2; resp = realloc(resp, cap); }
        ssize_t n = read(fd, resp + len, cap - len - 1);
        if (n <= 0) break;
        len += (size_t)n;
    }
    close(fd);
    resp[len] = '\0';
    int status = 0;
    if (sscanf(resp, "HTTP/1.1 %d", &status) != 1) status = 0;
    char *b = strstr(resp, "\r\n\r\n");
    if (body_out) *body_out = strdup(b ? b + 4 : "");
    free(resp);
    return status;
}

/* rm -rf a build/test_local state dir (fixed prefix: never anything else). */
static inline void lt_clean_dir(const char *dir)
{
    char cmd[1200];
    if (strncmp(dir, "build/test_local/", 17) != 0) return;
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir);
    if (system(cmd) != 0) fprintf(stderr, "  warn: could not clean %s\n", dir);
}

/* mkdir -p a build/test_local dir. */
static inline void lt_make_dir(const char *dir)
{
    char cmd[1200];
    if (strncmp(dir, "build/test_local/", 17) != 0) return;
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
    if (system(cmd) != 0) fprintf(stderr, "  warn: could not create %s\n", dir);
}

/* Pair with the box the way an app does: read the code from <state>/pair_code
 * (what `pantheon local pair-code` prints) and POST /local/pair. Returns a
 * static copy of the admin key, or NULL when the box has no pairing. */
static inline const char *lt_admin_key(int port, const char *state_dir)
{
    static char key[128];
    char path[1100], code[32] = "", body[128], *resp = NULL;
    FILE *f = NULL;

    snprintf(path, sizeof(path), "%s/pair_code", state_dir);
    f = fopen(path, "r");
    if (!f) return NULL;
    if (!fgets(code, sizeof(code), f)) code[0] = '\0';
    fclose(f);
    code[strcspn(code, "\r\n")] = '\0';
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"test\"}", code);
    if (lt_http(port, "POST", "/local/pair", NULL, body, &resp) != 200 || !resp) {
        free(resp);
        return NULL;
    }
    char *k = strstr(resp, "\"key\":\"");
    if (!k) { free(resp); return NULL; }
    k += 7;
    size_t n = strcspn(k, "\"");
    if (n >= sizeof(key)) n = sizeof(key) - 1;
    memcpy(key, k, n);
    key[n] = '\0';
    free(resp);
    return key;
}

/* Poll until GET / answers on port (any status) or ms runs out. */
static inline int lt_wait_port(int port, int ms)
{
    for (int waited = 0; waited < ms; waited += 50) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int ok = connect(fd, (struct sockaddr *)&a, sizeof(a)) == 0;
        close(fd);
        if (ok) return 0;
        usleep(50000);
    }
    return -1;
}

int pn_local_main(int argc, char **argv);

/* Fork a `pantheon local` child (stderr → log_path) and wait for its port. */
static inline pid_t lt_spawn_local(int argc, char **argv, int port, const char *log_path)
{
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        FILE *f = freopen(log_path, "w", stderr);
        (void)f;
        setvbuf(stderr, NULL, _IONBF, 0);   /* _exit skips stdio flushes: keep the log */
        _exit(pn_local_main(argc, argv));
    }
    if (pid > 0 && lt_wait_port(port, 20000) != 0) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        return -1;
    }
    return pid;
}

static inline void lt_stop(pid_t pid)
{
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 200; i++) {
        if (waitpid(pid, NULL, WNOHANG) == pid) return;
        usleep(50000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
}

/* 1 when nothing holds 127.0.0.1:port (a bind succeeds). */
static inline int lt_port_free(int port)
{
    struct sockaddr_in a;
    int fd = socket(AF_INET, SOCK_STREAM, 0), ok;

    if (fd < 0) return 0;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ok = bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0;
    close(fd);
    return ok;
}

/* A free port in 18400-18999 derived from the pid, so parallel test binaries differ.
 * Ports already held (another program's listener) are skipped: a test that connected
 * to someone else's server would fail for no reason of its own. */
static inline int lt_pick_port(int salt)
{
    int base = (int)((getpid() * 7 + salt * 31) % 600);

    for (int k = 0; k < 600; k++) {
        int port = 18400 + (base + k) % 600;
        if (lt_port_free(port)) return port;
    }
    return 18400 + base;
}

#endif /* PN_LOCAL_TEST_SUPPORT_H */
