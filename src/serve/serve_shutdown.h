/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_shutdown.h
 * @brief Deferred-shutdown state for `pantheon serve`: what a SIGINT/SIGTERM
 *        handler is allowed to record, how the accept loop observes it, and the
 *        drain decision that says whether an accelerator teardown may run.
 *
 *        WHY THIS EXISTS. Retiring a Tenstorrent residency means asking the
 *        resident kernel to stop and then POLLING the card until every enabled
 *        processor reports EXIT (pn_tt_residency_stop). That is device I/O, a
 *        bounded spin, and a `pn_model_free` that takes the registry lock —
 *        none of which a signal handler may do, because a handler runs at an
 *        arbitrary point in the program. Calling it from the handler would be
 *        worse than the gap it closes: it can retire a residency out from under
 *        a connection thread that is mid-descriptor, and `pn_tt_device_close`
 *        then frees the device that thread is still dereferencing.
 *
 *        So the handler does the two things that ARE async-signal-safe — store
 *        to a `volatile sig_atomic_t` and `write(2)` one byte to a self-pipe —
 *        and the accept loop performs the teardown at a point where the
 *        program's state is known. The pipe (rather than only relying on
 *        accept() returning EINTR) is what makes this correct in a threaded
 *        serve: the kernel may deliver the signal to ANY thread that has it
 *        unblocked, and a connection thread running the handler still wakes the
 *        accept loop through the pipe.
 *
 *        MID-GENERATION. A turn already on the card is not interrupted; it is
 *        ASKED to stop, through the `sg_should_abort_fn` the generation loop
 *        already polls between prefill chunks and every 8 decode tokens. The
 *        accept loop then waits for the in-flight count to reach zero before
 *        anything is retired. That is the "an abort must DRAIN, not return"
 *        rule (foundry/tt/DEV_NOTES.md §1) applied one level out, at the
 *        process boundary.
 *
 *        ON TIMEOUT. If the drain does not quiesce, the teardown is ABANDONED,
 *        not forced: tearing down under a live thread is the failure mode this
 *        module exists to avoid. What is left is exactly what a SIGKILL leaves,
 *        and that is already covered — SET_NOC_CLEANUP and recovery-on-acquire.
 *
 * @status Complete. Decision logic is pure and unit-tested off-hardware
 *         (tests/test_serve_shutdown.c); the retirement it gates is device-side.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_SHUTDOWN_H
#define PN_SERVE_SHUTDOWN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Default drain budget. Chosen against three measured numbers, not by feel:
 * the cooperative abort is polled every 8 decode tokens and every 8 prefill
 * chunks, which on the slowest backend in the fleet is ~2 s; a residency
 * descriptor's own device timeout is 60 s, so a budget below that turns a
 * WEDGED card into a bounded abandon instead of a hung shutdown; and an
 * operator stop has to feel like a stop. Override with PN_SERVE_DRAIN_MS
 * (0 = never wait — the escalation path and the card-side red proof use it). */
#define PN_SERVE_DRAIN_DEFAULT_MS 10000u

/* What the accept loop should do with an in-progress drain. */
typedef enum {
    PN_SHUTDOWN_DRAIN_WAIT = 0,   /* work in flight, budget left — keep waiting */
    PN_SHUTDOWN_DRAIN_RETIRE,     /* quiesced — the accelerator teardown may run */
    PN_SHUTDOWN_DRAIN_ABANDON,    /* budget spent with work still in flight —
                                   * do NOT retire under a live thread */
} pn_shutdown_drain_decision;

/* Create the self-pipe the handler writes and the accept loop polls. Call ONCE,
 * before pn_install_signal_handlers(). Returns 0 on success, -1 if the pipe
 * could not be created — in which case pn_serve_shutdown_wake_fd() stays -1 and
 * the handler falls back to its pre-drain behaviour rather than leaving the
 * process unkillable.
 *
 * Thread-safe: no (call once at startup)
 * Reentrant: no
 * Signal-safe: no */
int pn_serve_shutdown_arm(void);

/* Record a shutdown request. THE ONLY FUNCTION A SIGNAL HANDLER MAY CALL — it
 * stores to `volatile sig_atomic_t` and write(2)s one byte, both of which are
 * on POSIX's async-signal-safe list, and does nothing else. A second call marks
 * the request ESCALATED (a second Ctrl-C / SIGTERM), which collapses the drain
 * budget to zero at the next poll instead of aborting the process from a
 * handler.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes */
void pn_serve_shutdown_note_signal(int sig);

/* Read end of the self-pipe, or -1 if arming failed. Poll it alongside the
 * listening socket; one readable byte means "a shutdown was requested".
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes */
int pn_serve_shutdown_wake_fd(void);

/* Non-zero once a shutdown has been requested. Read by worker threads through
 * their abort predicate, and by the accept loop.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes */
int pn_serve_shutdown_requested(void);

/* Non-zero once a SECOND shutdown signal has arrived.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes */
int pn_serve_shutdown_escalated(void);

/* The signal number that requested the shutdown (0 if none). The first one
 * wins, so the exit status reflects what the operator actually sent.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes */
int pn_serve_shutdown_signal_number(void);

/* The drain budget in milliseconds: PN_SERVE_DRAIN_MS when it parses as a
 * non-negative integer, PN_SERVE_DRAIN_DEFAULT_MS otherwise.
 *
 * Thread-safe: yes
 * Reentrant: no (getenv)
 * Signal-safe: no */
unsigned pn_serve_shutdown_drain_budget_ms(void);

/* Enter the accelerator-driving section of a request. Returns 0 when entered —
 * the caller MUST pair it with pn_serve_generation_leave() — or -1 when a
 * shutdown has already been requested, in which case the route must refuse the
 * request instead of starting one. The check and the increment happen under one
 * lock, so a turn that gets in is always visible to the drain.
 *
 * Thread-safe: yes
 * Reentrant: no
 * Signal-safe: no */
int pn_serve_generation_enter(void);

/* Leave the accelerator-driving section. Pairs with a successful enter.
 *
 * Thread-safe: yes
 * Reentrant: no
 * Signal-safe: no */
void pn_serve_generation_leave(void);

/* How many accelerator-driving sections are currently in flight.
 *
 * Thread-safe: yes
 * Reentrant: no
 * Signal-safe: no */
int pn_serve_generation_inflight(void);

/* Reset every bit of module state. FOR TESTS ONLY — a serve process arms once
 * and shuts down once.
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no */
void pn_serve_shutdown_reset_for_test(void);

/**
 * Decide what an in-progress drain should do next. PURE — no clock, no globals,
 * no device — so the policy this whole mechanism turns on is testable without a
 * card, a serve or a signal.
 *
 * @param inflight_generations  accelerator-driving sections still running; a
 *                              negative value (unknown) counts as "in flight"
 * @param elapsed_ms            milliseconds since the drain began
 * @param budget_ms             the drain budget (0 = do not wait at all)
 * @return                      WAIT, RETIRE or ABANDON
 *
 * QUIESCENCE WINS OVER THE DEADLINE. `inflight == 0` returns RETIRE even at or
 * past the budget: a serve that finished its last turn in the same millisecond
 * the budget expired is safe to tear down, and answering ABANDON there would
 * throw away the retirement this mechanism exists to perform.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes
 */
pn_shutdown_drain_decision
pn_serve_shutdown_drain_decision(int inflight_generations,
                                 unsigned elapsed_ms,
                                 unsigned budget_ms);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_SHUTDOWN_H */
