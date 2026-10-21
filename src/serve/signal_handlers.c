/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file signal_handlers.c
 * @brief Implementation of the serve-process crash/shutdown handlers
 *        (MemoryAware.md §8, §11 Q4). Two tiers:
 *
 *        - SIGINT / SIGTERM (controlled shutdown): RECORD the request and
 *          return. The teardown itself belongs to the accept loop, which the
 *          record wakes through a self-pipe (serve_shutdown.h). It was fair to
 *          run the CUDA teardown inline while that was the only accelerator —
 *          an operator stop is not a crash mid-kernel — but the Tenstorrent
 *          backend made it unsafe: retiring a residency polls the card, takes
 *          the registry lock, and frees a device a connection thread may still
 *          be driving. A handler runs at an arbitrary point in the program and
 *          may do neither, so it does the two async-signal-safe things
 *          (sig_atomic_t store + write(2)) and hands the work to a point where
 *          the program's state is known. The CUDA teardown moved with it and
 *          still runs on every controlled stop, in serve_command.c.
 *
 *        - SIGSEGV / SIGABRT / SIGBUS (crash): do NOT call into libcuda from the
 *          handler (unsafe — can hang mid-fault). Write a one-line async-safe
 *          notice and _exit(); the kernel closes the fd on teardown, and the
 *          next start's stranded-VRAM detection (qwen35moe_forward_cuda die-A
 *          init) catches any reclaim race and refuses/reroutes rather than
 *          OOMing on a small alloc.
 *
 * @status Phase 3.
 * @issues None
 * @todo None
 */

#include "signal_handlers.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "serve_registry.h"
#include "serve_shutdown.h"

#ifdef USE_CUDA
#include "cuda/fc_cuda.h"
#endif

/* Guard so the handler runs its teardown at most once even if two signals race
 * (sig_atomic_t is the async-signal-safe flag type). */
static volatile sig_atomic_t g_handling = 0;

/* Async-signal-safe write of a fixed string to stderr (fd 2). fprintf/printf
 * are NOT signal-safe, so crash-path messaging goes through write(2). */
static void safe_write(const char *msg)
{
    ssize_t n = write(2, msg, strlen(msg));
    (void)n;   /* best-effort; nothing useful to do if it fails in a handler */
}

/* Controlled-shutdown handler (SIGINT/SIGTERM): record the request, wake the
 * accept loop, return. Everything the stop actually has to DO — deregister,
 * drain the in-flight turns, retire the card residencies, destroy the CUDA
 * contexts — runs in pn_serve_main's shutdown sequence. */
static void handle_shutdown(int sig)
{
    /* Async-signal-safe by construction: a sig_atomic_t store and a one-byte
     * write(2). Two signals in a row are fine — the second marks the request
     * escalated, which collapses the drain budget rather than aborting the
     * process from here. */
    pn_serve_shutdown_note_signal(sig);

    safe_write("\n[signal] shutdown requested — draining in-flight turns, then "
               "retiring accelerator residencies\n");

    if (pn_serve_shutdown_wake_fd() >= 0)
        return;   /* the accept loop owns the teardown from here */

    /* No wake pipe (pn_serve_shutdown_arm failed): nothing guarantees the
     * accept loop ever observes the flag, because the kernel may have delivered
     * this signal to a connection thread. Falling back to the pre-drain
     * behaviour is worse than the deferred path but far better than an
     * unkillable serve. It deliberately does NOT touch the accelerator
     * residencies — an unsafe teardown from a handler is worse than the gap. */
    if (g_handling)
        _exit(128 + sig);
    g_handling = 1;

    serve_registry_deregister_all();
#ifdef USE_CUDA
    if (fc_cuda_live_ctx_count() > 0)
        fc_cuda_emergency_teardown();
#endif
    signal(sig, SIG_DFL);
    raise(sig);
}

/* Crash handler (SIGSEGV/SIGABRT/SIGBUS): minimal, no libcuda calls. */
static void handle_crash(int sig)
{
    if (g_handling)
        _exit(128 + sig);
    g_handling = 1;

    safe_write("\n[signal] fatal crash — exiting; kernel will close the nvidia "
               "fd. Next start's stranded-VRAM check guards the die.\n");

    /* Do NOT touch libcuda here: a cuCtxDestroy mid-fault can hang. Restore the
     * default handler and re-raise so we get the real crash exit status (and a
     * core dump if enabled); process teardown closes the fd. */
    signal(sig, SIG_DFL);
    raise(sig);
}

void pn_install_signal_handlers(void)
{
    struct sigaction sa_shutdown;
    memset(&sa_shutdown, 0, sizeof(sa_shutdown));
    sa_shutdown.sa_handler = handle_shutdown;
    sigemptyset(&sa_shutdown.sa_mask);
    sigaction(SIGINT, &sa_shutdown, NULL);
    sigaction(SIGTERM, &sa_shutdown, NULL);

    struct sigaction sa_crash;
    memset(&sa_crash, 0, sizeof(sa_crash));
    sa_crash.sa_handler = handle_crash;
    sigemptyset(&sa_crash.sa_mask);
    /* SA_RESETHAND so a fault inside the handler falls through to the default. */
    sa_crash.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sa_crash, NULL);
    sigaction(SIGABRT, &sa_crash, NULL);
    sigaction(SIGBUS, &sa_crash, NULL);
}
