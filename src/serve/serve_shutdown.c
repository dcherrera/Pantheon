/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_shutdown.c
 * @brief Implementation of the deferred-shutdown state (see serve_shutdown.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "serve_shutdown.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

/* Signal-side state. Every one of these is written from a handler, so each is
 * `volatile sig_atomic_t` and nothing else — no strings, no counters wider than
 * the type the standard promises is atomic against a signal. */
static volatile sig_atomic_t g_shutdown_requested = 0;
static volatile sig_atomic_t g_shutdown_escalated = 0;
static volatile sig_atomic_t g_shutdown_signal    = 0;

/* The self-pipe. Only the write end is touched from a handler. */
static int g_wake_read_fd  = -1;
static int g_wake_write_fd = -1;

/* In-flight accelerator-driving sections. A plain int under a mutex rather than
 * an atomic, because the drain has to read the count and the shutdown flag as
 * ONE decision — an entry that slipped past the flag check must already be in
 * the count by the time the drain looks. */
static pthread_mutex_t g_generation_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_generation_inflight = 0;

/* Mark a descriptor close-on-exec and non-blocking. The write end must never
 * block: a handler that blocks on a full pipe is a hung process, and one queued
 * byte already carries the whole message. */
static int shutdown_prepare_fd(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);

    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
        return -1;
    flags = fcntl(fd, F_GETFD, 0);
    if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
        return -1;
    return 0;
}

int pn_serve_shutdown_arm(void)
{
    int fds[2];

    if (g_wake_read_fd >= 0)
        return 0;
    if (pipe(fds) != 0)
        return -1;
    if (shutdown_prepare_fd(fds[0]) != 0 || shutdown_prepare_fd(fds[1]) != 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    g_wake_read_fd = fds[0];
    g_wake_write_fd = fds[1];
    return 0;
}

void pn_serve_shutdown_note_signal(int sig)
{
    if (g_shutdown_requested)
        g_shutdown_escalated = 1;
    else
        g_shutdown_signal = sig;
    g_shutdown_requested = 1;

    /* One byte is the whole message, and a short/failed write is not worth
     * handling: if the pipe already holds a byte the accept loop is about to
     * wake anyway, and there is nothing a handler could do about EAGAIN. */
    if (g_wake_write_fd >= 0) {
        const char token = 1;
        ssize_t written = write(g_wake_write_fd, &token, 1);
        (void)written;
    }
}

int pn_serve_shutdown_wake_fd(void)
{
    return g_wake_read_fd;
}

int pn_serve_shutdown_requested(void)
{
    return g_shutdown_requested ? 1 : 0;
}

int pn_serve_shutdown_escalated(void)
{
    return g_shutdown_escalated ? 1 : 0;
}

int pn_serve_shutdown_signal_number(void)
{
    return (int)g_shutdown_signal;
}

unsigned pn_serve_shutdown_drain_budget_ms(void)
{
    const char *raw = getenv("PN_SERVE_DRAIN_MS");
    char *end = NULL;
    long value;

    if (!raw || !*raw)
        return PN_SERVE_DRAIN_DEFAULT_MS;
    /* strtol, not atoi: a typo'd budget must fall back to the default rather
     * than silently parse to 0 and skip the drain entirely. */
    value = strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0)
        return PN_SERVE_DRAIN_DEFAULT_MS;
    return (unsigned)value;
}

int pn_serve_generation_enter(void)
{
    int entered;

    pthread_mutex_lock(&g_generation_lock);
    entered = g_shutdown_requested ? 0 : 1;
    if (entered)
        g_generation_inflight++;
    pthread_mutex_unlock(&g_generation_lock);
    return entered ? 0 : -1;
}

void pn_serve_generation_leave(void)
{
    pthread_mutex_lock(&g_generation_lock);
    if (g_generation_inflight > 0)
        g_generation_inflight--;
    pthread_mutex_unlock(&g_generation_lock);
}

int pn_serve_generation_inflight(void)
{
    int count;

    pthread_mutex_lock(&g_generation_lock);
    count = g_generation_inflight;
    pthread_mutex_unlock(&g_generation_lock);
    return count;
}

void pn_serve_shutdown_reset_for_test(void)
{
    if (g_wake_read_fd >= 0) {
        close(g_wake_read_fd);
        g_wake_read_fd = -1;
    }
    if (g_wake_write_fd >= 0) {
        close(g_wake_write_fd);
        g_wake_write_fd = -1;
    }
    g_shutdown_requested = 0;
    g_shutdown_escalated = 0;
    g_shutdown_signal = 0;
    pthread_mutex_lock(&g_generation_lock);
    g_generation_inflight = 0;
    pthread_mutex_unlock(&g_generation_lock);
}

pn_shutdown_drain_decision
pn_serve_shutdown_drain_decision(int inflight_generations,
                                 unsigned elapsed_ms,
                                 unsigned budget_ms)
{
    /* Quiescence is checked FIRST and unconditionally. Ordering the deadline
     * ahead of it would abandon the retirement on a serve that had already
     * finished its last turn, which is the one case where retiring is both safe
     * and the entire point. */
    if (inflight_generations == 0)
        return PN_SHUTDOWN_DRAIN_RETIRE;
    if (elapsed_ms >= budget_ms)
        return PN_SHUTDOWN_DRAIN_ABANDON;
    return PN_SHUTDOWN_DRAIN_WAIT;
}
