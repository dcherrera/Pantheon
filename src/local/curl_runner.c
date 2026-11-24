/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   curl_runner.c
 * @brief  Implementation of the curl child process (see curl_runner.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "curl_runner.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <signal.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

static pthread_once_t g_curl_once = PTHREAD_ONCE_INIT;
static char           g_curl_path[512];

static void find_curl(void)
{
    const char *path = getenv("PATH");
    char dir[512];

    if (!path || !*path)
        path = "/usr/bin:/bin:/usr/local/bin";
    while (*path) {
        size_t n = strcspn(path, ":");
        if (n > 0 && n < sizeof(dir)) {
            struct stat sb;
            memcpy(dir, path, n);
            dir[n] = '\0';
            snprintf(g_curl_path, sizeof(g_curl_path), "%s/curl", dir);
            if (stat(g_curl_path, &sb) == 0 && S_ISREG(sb.st_mode) && access(g_curl_path, X_OK) == 0)
                return;
        }
        path += n + (path[n] == ':');
    }
    g_curl_path[0] = '\0';
}

const char *pn_curl_path(void)
{
    pthread_once(&g_curl_once, find_curl);
    return g_curl_path[0] ? g_curl_path : NULL;
}

/* First line of what the child wrote to stderr; the rest is drained so the
 * child never blocks on a full pipe. */
static void read_first_line(int fd, char *err, size_t errlen)
{
    char buf[512], sink[512];
    size_t len = 0;
    ssize_t n;

    while (len < sizeof(buf) - 1 && (n = read(fd, buf + len, sizeof(buf) - 1 - len)) > 0)
        len += (size_t)n;
    while (read(fd, sink, sizeof(sink)) > 0) {}
    buf[len] = '\0';
    buf[strcspn(buf, "\n")] = '\0';
    if (err && errlen)
        snprintf(err, errlen, "%s", buf);
}

int pn_curl_download(const char *url, const char *out_path, const char *protocols,
                     bool resume, char *err, size_t errlen)
{
    const char *curl = pn_curl_path();
    int pipefd[2];

    if (err && errlen)
        err[0] = '\0';
    if (!curl) {
        snprintf(err, errlen, "curl is not installed");
        return -1;
    }
    char *argv[] = {
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--proto", (char *)protocols, "--proto-redir", (char *)protocols,
        "--connect-timeout", "30", "--output", (char *)out_path,
        resume ? "--continue-at" : "--url", resume ? "-" : (char *)url,
        resume ? "--url" : NULL, resume ? (char *)url : NULL, NULL
    };
    if (pipe(pipefd) != 0)
        return -1;
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        /* Child: only async-signal-safe calls until execv. */
#ifdef __linux__
        /* Die with the appliance: no orphan curl appending to a .part file. */
        prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
        dup2(pipefd[1], 2);
        close(pipefd[0]);
        close(pipefd[1]);
        execv(curl, argv);
        _exit(127);
    }
    close(pipefd[1]);
    read_first_line(pipefd[0], err, errlen);
    close(pipefd[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status))
        return -1;
    return WEXITSTATUS(status);
}

/* pipe() with both ends close-on-exec: a curl another thread forks must not hold them. */
static int pipe_cloexec(int fds[2])
{
    if (pipe(fds) != 0)
        return -1;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    return 0;
}

static void close_pair(int fds[2])
{
    close(fds[0]);
    close(fds[1]);
}

int pn_curl_stream_open(const char *url, const char *protocols, uint64_t offset,
                        pn_curl_stream *s, char *err, size_t errlen)
{
    const char *curl = pn_curl_path();
    int body[2], errp[2];
    char from[24];

    memset(s, 0, sizeof(*s));
    s->body_fd = s->err_fd = -1;
    if (!curl) {
        snprintf(err, errlen, "curl is not installed");
        return -1;
    }
    snprintf(from, sizeof(from), "%llu", (unsigned long long)offset);
    char *argv[] = {
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--proto", (char *)protocols, "--proto-redir", (char *)protocols,
        "--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "120",
        "--output", "-", "--continue-at", from, "--url", (char *)url, NULL
    };
    if (pipe_cloexec(body) != 0) {
        snprintf(err, errlen, "pipe: %s", strerror(errno));
        return -1;
    }
    if (pipe_cloexec(errp) != 0) {
        snprintf(err, errlen, "pipe: %s", strerror(errno));
        close_pair(body);
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, errlen, "fork: %s", strerror(errno));
        close_pair(body);
        close_pair(errp);
        return -1;
    }
    if (pid == 0) {
        /* Child: only async-signal-safe calls until execv. dup2 clears close-on-exec. */
#ifdef __linux__
        prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
        dup2(body[1], 1);
        dup2(errp[1], 2);
        execv(curl, argv);
        _exit(127);
    }
    close(body[1]);
    close(errp[1]);
    s->pid = pid;
    s->body_fd = body[0];
    s->err_fd = errp[0];
    return 0;
}

int pn_curl_stream_close(pn_curl_stream *s, bool stop, char *err, size_t errlen)
{
    int status = 0;

    if (err && errlen)
        err[0] = '\0';
    if (s->pid <= 0)
        return -1;
    if (stop)
        kill(s->pid, SIGTERM);
    close(s->body_fd);
    read_first_line(s->err_fd, err, errlen);
    close(s->err_fd);
    pid_t got = -1;
    while ((got = waitpid(s->pid, &status, 0)) < 0 && errno == EINTR) {}
    memset(s, 0, sizeof(*s));
    s->body_fd = s->err_fd = -1;
    return got > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
