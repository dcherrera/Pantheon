/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_serve_shutdown.c — the deferred-shutdown mechanism, off-hardware.
 *
 * WHAT THIS GATES. `pantheon serve` used to answer SIGTERM from inside the
 * signal handler and never retire its Tenstorrent residencies, which is the
 * "an abort must DRAIN, not return" shape that has wedged cores. The fix moves
 * the teardown to the accept loop and gates it on a drain. The retirement
 * itself is device-facing and cannot run here, so this test covers the three
 * decisions that determine whether it happens at all:
 *
 *   1. what the HANDLER records (and that it records ONLY that);
 *   2. what the LOOP does with an in-flight count — including that quiescence
 *      beats the deadline, and that a timeout ABANDONS rather than tearing down
 *      under a live thread;
 *   3. that a normal path is unaffected — before any signal, generations enter
 *      freely and the drain decision is RETIRE.
 *
 * The signal path is exercised by RAISING A REAL SIGNAL through the process's
 * own installed handler, not by calling the recorder directly — the claim being
 * tested is that a delivered signal reaches the flag and the wake pipe.
 *
 * DELIBERATELY NOT INSTRUMENTED WITH ITSELF: the drain decision is a pure
 * function of (inflight, elapsed, budget), so it is driven with literal values
 * here rather than with a clock or a real drain loop. A test that ran the drain
 * to decide whether the drain is right would be measuring its own agreement.
 *
 * Links only serve_shutdown.c — no engine, no serve, no card.
 */

#include "../src/serve/serve_shutdown.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

/* ---------------------------------------------------------------------------
 * 1. The drain decision — the pure policy the whole mechanism turns on.
 * ------------------------------------------------------------------------- */

static void test_drain_decision_quiesced(void)
{
    printf("drain: a quiesced serve RETIREs\n");

    CHECK(pn_serve_shutdown_drain_decision(0, 0, 10000) == PN_SHUTDOWN_DRAIN_RETIRE,
          "no work, budget left → RETIRE");
    CHECK(pn_serve_shutdown_drain_decision(0, 9999, 10000) == PN_SHUTDOWN_DRAIN_RETIRE,
          "no work, just inside the budget → RETIRE");

    /* The ordering check. A decision that tested the deadline first would throw
     * away the retirement on a serve whose last turn finished in the same
     * millisecond the budget expired — the one case where retiring is both safe
     * and the entire point of the change. */
    CHECK(pn_serve_shutdown_drain_decision(0, 10000, 10000) == PN_SHUTDOWN_DRAIN_RETIRE,
          "no work AT the deadline → RETIRE (quiescence beats the deadline)");
    CHECK(pn_serve_shutdown_drain_decision(0, 999999, 10000) == PN_SHUTDOWN_DRAIN_RETIRE,
          "no work long past the deadline → RETIRE");
    CHECK(pn_serve_shutdown_drain_decision(0, 0, 0) == PN_SHUTDOWN_DRAIN_RETIRE,
          "no work with a zero budget → RETIRE");
}

static void test_drain_decision_busy(void)
{
    printf("drain: work in flight WAITs while budget remains\n");

    CHECK(pn_serve_shutdown_drain_decision(1, 0, 10000) == PN_SHUTDOWN_DRAIN_WAIT,
          "one turn, no time spent → WAIT");
    CHECK(pn_serve_shutdown_drain_decision(4, 9999, 10000) == PN_SHUTDOWN_DRAIN_WAIT,
          "four turns, one ms of budget left → WAIT");
    CHECK(pn_serve_shutdown_drain_decision(1, 1, 2) == PN_SHUTDOWN_DRAIN_WAIT,
          "WAIT holds at a tiny budget too");
}

static void test_drain_decision_timeout(void)
{
    printf("drain: a spent budget ABANDONs rather than tearing down\n");

    CHECK(pn_serve_shutdown_drain_decision(1, 10000, 10000) == PN_SHUTDOWN_DRAIN_ABANDON,
          "one turn AT the deadline → ABANDON");
    CHECK(pn_serve_shutdown_drain_decision(1, 10001, 10000) == PN_SHUTDOWN_DRAIN_ABANDON,
          "one turn past the deadline → ABANDON");
    CHECK(pn_serve_shutdown_drain_decision(8, 60000, 10000) == PN_SHUTDOWN_DRAIN_ABANDON,
          "several turns well past the deadline → ABANDON");

    /* A zero budget is the escalation path (a second SIGTERM) and the card-side
     * red proof's lever: never wait, and therefore never retire under a live
     * thread. */
    CHECK(pn_serve_shutdown_drain_decision(1, 0, 0) == PN_SHUTDOWN_DRAIN_ABANDON,
          "zero budget with work in flight → ABANDON immediately");

    /* An unknown count must not be read as quiescence. */
    CHECK(pn_serve_shutdown_drain_decision(-1, 0, 10000) == PN_SHUTDOWN_DRAIN_WAIT,
          "negative (unknown) count is NOT quiescence → WAIT");
    CHECK(pn_serve_shutdown_drain_decision(-1, 10000, 10000) == PN_SHUTDOWN_DRAIN_ABANDON,
          "negative (unknown) count past the deadline → ABANDON");
}

/* ---------------------------------------------------------------------------
 * 2. The in-flight bracket — what the drain counts.
 * ------------------------------------------------------------------------- */

static void test_generation_bracket_normal_path(void)
{
    printf("bracket: the normal path is unaffected\n");
    pn_serve_shutdown_reset_for_test();

    CHECK(pn_serve_generation_inflight() == 0, "starts empty");
    CHECK(pn_serve_generation_enter() == 0, "enter succeeds with no shutdown pending");
    CHECK(pn_serve_generation_inflight() == 1, "one turn counted");
    CHECK(pn_serve_generation_enter() == 0, "a second concurrent turn enters");
    CHECK(pn_serve_generation_inflight() == 2, "two turns counted");
    pn_serve_generation_leave();
    pn_serve_generation_leave();
    CHECK(pn_serve_generation_inflight() == 0, "back to empty after both leave");

    /* And with nothing in flight the loop would retire — the ordinary stop. */
    CHECK(pn_serve_shutdown_drain_decision(pn_serve_generation_inflight(), 0,
                                           pn_serve_shutdown_drain_budget_ms())
              == PN_SHUTDOWN_DRAIN_RETIRE,
          "an idle serve retires immediately");

    /* An unbalanced leave must not drive the count negative — a negative count
     * would read as "past quiescence" to any caller that compared it wrong. */
    pn_serve_generation_leave();
    CHECK(pn_serve_generation_inflight() == 0, "leave without enter floors at zero");
}

static void test_generation_refused_after_request(void)
{
    printf("bracket: a new turn is refused once a shutdown is requested\n");
    pn_serve_shutdown_reset_for_test();

    CHECK(pn_serve_generation_enter() == 0, "turn A enters before the signal");
    pn_serve_shutdown_note_signal(SIGTERM);

    /* The turn already inside is NOT evicted — it is drained. */
    CHECK(pn_serve_generation_inflight() == 1, "the in-flight turn is still counted");
    CHECK(pn_serve_generation_enter() == -1, "a NEW turn is refused after the request");
    CHECK(pn_serve_generation_inflight() == 1, "a refused turn is not counted");

    /* While it runs, the loop must WAIT — not retire. */
    CHECK(pn_serve_shutdown_drain_decision(pn_serve_generation_inflight(), 0, 10000)
              == PN_SHUTDOWN_DRAIN_WAIT,
          "the loop waits while the turn is on the accelerator");

    pn_serve_generation_leave();
    CHECK(pn_serve_shutdown_drain_decision(pn_serve_generation_inflight(), 5, 10000)
              == PN_SHUTDOWN_DRAIN_RETIRE,
          "once the turn stops, the loop retires");
}

/* ---------------------------------------------------------------------------
 * 3. The handler — a REAL delivered signal, through the real sigaction.
 * ------------------------------------------------------------------------- */

/* The handler under test: exactly what signal_handlers.c installs for the
 * deferred path — record and return, nothing else. Installed here rather than
 * linking serve_command.c, which would drag the whole engine in. */
static void test_handler(int sig)
{
    pn_serve_shutdown_note_signal(sig);
}

static void install_test_handler(int sig)
{
    struct sigaction sa;

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = test_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, NULL);
}

/* Drain one byte from the wake pipe. Returns the count read (0 = nothing
 * pending), or -1 on a real error. */
static int read_wake_bytes(void)
{
    char buf[8];
    ssize_t n;

    if (pn_serve_shutdown_wake_fd() < 0)
        return -1;
    n = read(pn_serve_shutdown_wake_fd(), buf, sizeof(buf));
    if (n < 0)
        return (errno == EAGAIN || errno == EWOULDBLOCK) ? 0 : -1;
    return (int)n;
}

static void test_handler_records_and_wakes(int sig, const char *label)
{
    printf("handler: %s records the request and wakes the loop\n", label);
    pn_serve_shutdown_reset_for_test();
    CHECK(pn_serve_shutdown_arm() == 0, "wake pipe arms");
    install_test_handler(sig);

    /* Before: nothing recorded, nothing pending on the pipe. Without this the
     * "after" assertions would pass on a flag that was never clear. */
    CHECK(!pn_serve_shutdown_requested(), "no request before the signal");
    CHECK(!pn_serve_shutdown_escalated(), "not escalated before the signal");
    CHECK(pn_serve_shutdown_signal_number() == 0, "no signal number before the signal");
    CHECK(read_wake_bytes() == 0, "wake pipe is empty before the signal");

    raise(sig);

    CHECK(pn_serve_shutdown_requested(), "the delivered signal set the request");
    CHECK(pn_serve_shutdown_signal_number() == sig, "the signal number is recorded");
    CHECK(!pn_serve_shutdown_escalated(), "one signal is not an escalation");
    CHECK(read_wake_bytes() == 1, "exactly one wake byte reached the pipe");

    signal(sig, SIG_DFL);
}

static void test_handler_escalates_on_second_signal(void)
{
    printf("handler: a second signal escalates instead of aborting\n");
    pn_serve_shutdown_reset_for_test();
    CHECK(pn_serve_shutdown_arm() == 0, "wake pipe arms");
    install_test_handler(SIGTERM);
    install_test_handler(SIGINT);

    raise(SIGTERM);
    CHECK(!pn_serve_shutdown_escalated(), "first signal is not an escalation");
    (void)read_wake_bytes();

    /* A different signal the second time, to prove the FIRST one is what the
     * exit status will report — an escalation must not rewrite it. */
    raise(SIGINT);
    CHECK(pn_serve_shutdown_escalated(), "the second signal escalated");
    CHECK(pn_serve_shutdown_signal_number() == SIGTERM,
          "the FIRST signal number is kept");

    /* Escalation is what collapses the budget, and a collapsed budget with work
     * in flight must ABANDON — the process gets out, and it does so without an
     * accelerator teardown under a live thread. */
    CHECK(pn_serve_shutdown_drain_decision(1, 0, 0) == PN_SHUTDOWN_DRAIN_ABANDON,
          "escalated budget with work in flight → ABANDON");

    signal(SIGTERM, SIG_DFL);
    signal(SIGINT, SIG_DFL);
}

/* The handler must be observable from a thread that is not the one that ran it:
 * the kernel may deliver SIGTERM to any thread with it unblocked, and the whole
 * point of the wake pipe is that the accept loop still learns about it. */
static void *worker_reads_flag(void *ud)
{
    int *saw = ud;

    *saw = pn_serve_shutdown_requested();
    return NULL;
}

static void test_request_is_visible_across_threads(void)
{
    printf("handler: the request is visible from another thread\n");
    pn_serve_shutdown_reset_for_test();
    CHECK(pn_serve_shutdown_arm() == 0, "wake pipe arms");
    install_test_handler(SIGTERM);

    pthread_t tid;
    int saw_before = -1, saw_after = -1;

    CHECK(pthread_create(&tid, NULL, worker_reads_flag, &saw_before) == 0,
          "worker thread starts");
    pthread_join(tid, NULL);
    CHECK(saw_before == 0, "worker sees no request before the signal");

    raise(SIGTERM);

    CHECK(pthread_create(&tid, NULL, worker_reads_flag, &saw_after) == 0,
          "second worker thread starts");
    pthread_join(tid, NULL);
    CHECK(saw_after == 1, "worker sees the request after the signal");

    signal(SIGTERM, SIG_DFL);
}

/* ---------------------------------------------------------------------------
 * 4. The drain budget.
 * ------------------------------------------------------------------------- */

static void test_drain_budget_from_environment(void)
{
    printf("budget: PN_SERVE_DRAIN_MS parses, and a bad value falls back\n");

    unsetenv("PN_SERVE_DRAIN_MS");
    CHECK(pn_serve_shutdown_drain_budget_ms() == PN_SERVE_DRAIN_DEFAULT_MS,
          "unset → the default budget");

    setenv("PN_SERVE_DRAIN_MS", "250", 1);
    CHECK(pn_serve_shutdown_drain_budget_ms() == 250u, "a number is honoured");

    setenv("PN_SERVE_DRAIN_MS", "0", 1);
    CHECK(pn_serve_shutdown_drain_budget_ms() == 0u,
          "an explicit 0 is honoured (never wait)");

    /* A typo must not silently become 0 and skip the drain — that would turn a
     * misspelled variable into the exposure this change removes. */
    setenv("PN_SERVE_DRAIN_MS", "10s", 1);
    CHECK(pn_serve_shutdown_drain_budget_ms() == PN_SERVE_DRAIN_DEFAULT_MS,
          "a trailing-garbage value falls back to the default");
    setenv("PN_SERVE_DRAIN_MS", "abc", 1);
    CHECK(pn_serve_shutdown_drain_budget_ms() == PN_SERVE_DRAIN_DEFAULT_MS,
          "a non-numeric value falls back to the default");
    setenv("PN_SERVE_DRAIN_MS", "-5", 1);
    CHECK(pn_serve_shutdown_drain_budget_ms() == PN_SERVE_DRAIN_DEFAULT_MS,
          "a negative value falls back to the default");
    unsetenv("PN_SERVE_DRAIN_MS");
}

int main(void)
{
    printf("=== serve shutdown (deferred SIGTERM/SIGINT teardown) ===\n\n");

    test_drain_decision_quiesced();
    test_drain_decision_busy();
    test_drain_decision_timeout();
    test_generation_bracket_normal_path();
    test_generation_refused_after_request();
    test_handler_records_and_wakes(SIGTERM, "SIGTERM");
    test_handler_records_and_wakes(SIGINT, "SIGINT");
    test_handler_escalates_on_second_signal();
    test_request_is_visible_across_threads();
    test_drain_budget_from_environment();

    pn_serve_shutdown_reset_for_test();

    printf("\n%d checks, %d failures\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
