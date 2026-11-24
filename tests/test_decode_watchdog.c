/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_decode_watchdog.c — B3/B4: a decode slot's owner always gets control
 * back, so the slot is released instead of held until restart.
 *
 * On a hand-built scheduler (no engine, no GPU; the test_bs_load pattern):
 *   - the watched token wait returns a token the driver deposits, drains queued
 *     tokens first, and reports a finished slot;
 *   - with no token flowing it returns ABORTED once should_abort says the client
 *     left (B3: before, the owner blocked in bs_slot_pop_token forever and the
 *     disconnect was only noticed between tokens), and STALLED after the stall
 *     limit (B4: a stream the driver stopped feeding stayed open, silent);
 *   - the legacy lockstep wait (bs_slot_await_first) gives up after the limit.
 * On a socketpair:
 *   - an SSE write to a closed peer reports the failure;
 *   - an SSE stream whose peer stops reading fails a write within the send
 *     timeout instead of blocking forever (the owner then releases the slot).
 *
 * Every wait runs under alarm(): a regression that blocks fails the test
 * instead of hanging it.
 */

#include "../src/serve/batch_sched.h"
#include "../src/serve/batch_sched_watchdog.h"
#include "../src/serve/http_server.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int g_failed = 0, g_checks = 0;
static const char *g_phase = "";

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static void on_alarm(int sig)
{
    (void)sig;
    static const char msg[] = "  FAIL: blocked past the alarm (a wait never returned)\n";
    ssize_t n = write(2, msg, sizeof(msg) - 1);
    (void)n;
    n = write(2, g_phase, strlen(g_phase));
    (void)n;
    _exit(1);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static pn_batch_sched g_sched;
static uint32_t *g_ring;   /* malloc'd: bs_slot_release frees the ring */

/* A cap-1 scheduler whose slot 0 is mid-decode with an empty ring. */
static void sched_reset(void)
{
    memset(&g_sched, 0, sizeof(g_sched));
    pthread_mutex_init(&g_sched.mu, NULL);
    pthread_cond_init(&g_sched.slots[0].cv, NULL);
    g_sched.cap = 1;
    bs_slot *sl = &g_sched.slots[0];
    sl->slot_idx = 0;
    sl->status = BS_SLOT_DECODE;
    g_ring = calloc(16, sizeof(*g_ring));   /* a released ring was freed */
    sl->ring = g_ring;
    sl->ring_cap = 16;
}

static void *deposit_later(void *arg)
{
    (void)arg;
    usleep(100 * 1000);
    pthread_mutex_lock(&g_sched.mu);
    g_sched.slots[0].ring[g_sched.slots[0].ring_w++] = 42;
    pthread_cond_signal(&g_sched.slots[0].cv);
    pthread_mutex_unlock(&g_sched.mu);
    return NULL;
}

static int g_abort_calls;
static int client_gone(void *ud) { (void)ud; g_abort_calls++; return 1; }
static int client_here(void *ud) { (void)ud; g_abort_calls++; return 0; }

static void test_token_paths(void)
{
    printf("token_paths:\n");
    g_phase = "token_paths\n";
    uint32_t tok = 0;
    bs_pop_watch w = { client_here, NULL, 50, 5000 };

    sched_reset();
    pthread_t th;
    pthread_create(&th, NULL, deposit_later, NULL);
    alarm(5);
    bs_pop_status r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    alarm(0);
    pthread_join(th, NULL);
    CHECK(r == BS_POP_TOKEN && tok == 42, "a token the driver deposits mid-wait comes back");

    g_ring[0] = 7;
    g_sched.slots[0].ring_r = 0;
    g_sched.slots[0].ring_w = 1;
    g_sched.slots[0].finished = true;
    r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    CHECK(r == BS_POP_TOKEN && tok == 7, "queued tokens drain before the finish");
    r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    CHECK(r == BS_POP_FINISHED, "then the finish");

    g_sched.slots[0].finished = false;
    g_sched.slots[0].step_error = true;
    r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    CHECK(r == BS_POP_ERROR, "a step error");
}

static void test_client_gone_releases(void)
{
    printf("client_gone_releases (B3):\n");
    g_phase = "client_gone_releases\n";
    uint32_t tok = 0;
    bs_pop_watch w = { client_gone, NULL, 50, 0 };   /* no stall limit at all */

    sched_reset();
    g_abort_calls = 0;
    double t0 = now_ms();
    alarm(5);
    bs_pop_status r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    alarm(0);
    CHECK(r == BS_POP_ABORTED, "no token flowing, client gone: the owner gets control back");
    CHECK(now_ms() - t0 < 1000.0, "within about one poll interval");
    CHECK(g_abort_calls >= 1, "the abort check ran while waiting");

    /* What the owner then does: release, and the slot is free again. */
    bs_slot_release(&g_sched, 0);
    CHECK(g_sched.slots[0].status == BS_SLOT_IDLE, "released: the slot is idle");
}

static void test_stall_watchdog(void)
{
    printf("stall_watchdog (B4):\n");
    g_phase = "stall_watchdog\n";
    uint32_t tok = 0;
    bs_pop_watch w = { client_here, NULL, 50, 300 };

    sched_reset();
    double t0 = now_ms();
    alarm(5);
    bs_pop_status r = bs_slot_pop_token_watched(&g_sched, 0, &tok, &w);
    alarm(0);
    double took = now_ms() - t0;
    CHECK(r == BS_POP_STALLED, "no token for the stall limit: STALLED, not a silent stream");
    CHECK(took >= 280.0 && took < 2000.0, "fires at the limit");

    /* The legacy lockstep exchange is bounded by the same limit (env, ms). */
    sched_reset();
    alarm(5);
    float *logits = NULL;
    int rc = bs_slot_await_first(&g_sched, 0, &logits);
    alarm(0);
    CHECK(rc == -1 && logits == NULL, "legacy wait: no logits within the limit gives up");
}

static void test_sse_write_failures(void)
{
    printf("sse_write_failures (B3):\n");
    g_phase = "sse_write_failures\n";
    int sv[2];

    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair");
    close(sv[1]);
    CHECK(hs_send_sse(sv[0], "{\"x\":1}") == -1, "a write to a closed peer reports it");
    close(sv[0]);

    /* A peer that stays connected but never reads: the stream's send timeout
     * turns the blocked write into a failure. */
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair");
    static char big[64 * 1024];
    memset(big, 'a', sizeof(big) - 1);
    alarm(10);
    double t0 = now_ms();
    int rc = hs_begin_sse(sv[0]);
    for (int i = 0; rc == 0 && i < 100000; i++)
        rc = hs_send_sse(sv[0], big);
    alarm(0);
    CHECK(rc == -1, "a peer that stops reading fails the write");
    CHECK(now_ms() - t0 < 5000.0, "within the send timeout, not forever");
    close(sv[0]);
    close(sv[1]);
}

int main(void)
{
    /* Before anything reads them (both are read once). */
    setenv("PN_DECODE_STALL_MS", "300", 1);
    setenv("PN_SSE_SEND_TIMEOUT_MS", "200", 1);
    signal(SIGALRM, on_alarm);
    signal(SIGPIPE, SIG_IGN);

    test_token_paths();
    test_client_gone_releases();
    test_stall_watchdog();
    test_sse_write_failures();

    if (g_failed) {
        fprintf(stderr, "test_decode_watchdog: %d of %d checks FAILED\n", g_failed, g_checks);
        return 1;
    }
    printf("test_decode_watchdog: all %d checks passed\n", g_checks);
    return 0;
}
