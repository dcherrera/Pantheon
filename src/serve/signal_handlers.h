/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file signal_handlers.h
 * @brief Crash / shutdown signal handling for the serve process (MemoryAware.md
 *        §8). On the Kepler nvidia stack VRAM is reclaimed only when the owning
 *        process's /dev/nvidiaN fd closes — cuCtxDestroy alone does not drop it.
 *        So the guarantee is: on ANY exit, close the fd. A controlled shutdown
 *        (SIGINT/SIGTERM) still drains + destroys the live CUDA contexts, but
 *        the handler no longer does it: it records the request through
 *        serve_shutdown.h and the accept loop performs the teardown, because
 *        the Tenstorrent residency retirement that now shares that path polls
 *        the card and takes locks a handler may not. On a crash
 *        (SIGSEGV/SIGABRT) they do the minimal async-signal-safe thing and
 *        _exit(), leaning on the kernel to close the fd and on startup
 *        stranded-VRAM detection to catch any race.
 *
 * @status Phase 3 — crash cleanup + VRAM reclaim.
 * @issues Worker-respawn reclaim (kill a wedged die's owner to force fd-close)
 *         is DEFERRED — barswap per-die workers are deprecated for this project;
 *         within the single-process k80_quirk boot-map the reclaim primitive is
 *         process exit itself. Multi-process reclaim needs design (see §8.3).
 * @todo None
 */

#ifndef PN_SIGNAL_HANDLERS_H
#define PN_SIGNAL_HANDLERS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Install the serve-process signal handlers (SIGINT, SIGTERM, SIGSEGV,
 * SIGABRT, SIGBUS). Idempotent. Safe to call even in CPU-only builds — the CUDA
 * teardown is a no-op when no contexts are live. Call once, after model load,
 * before the accept loop.
 *
 * Thread-safe: no (install once at startup)
 * Reentrant: no */
void pn_install_signal_handlers(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_SIGNAL_HANDLERS_H */
