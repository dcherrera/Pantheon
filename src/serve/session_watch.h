/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_watch.h
 * @brief Liveness-based session release — the "alive holds, dead releases"
 *        control plane (K80 serving-ops §C.4). A client opens a long-held
 *        GET /session/watch for each of its live prefix-cache sessions and
 *        keeps the connection open for the session's whole life. While the
 *        connection is held the session is ALIVE and is NEVER reaped for
 *        inactivity — an autonomous task that sits quiet in wait_agents for an
 *        hour keeps its warm slot. Only the connection actually CLOSING (client
 *        crash / drop) frees the slot, and only after a short grace so a
 *        flaky-bridge hiccup + resilient reconnect never false-releases.
 *
 * Why a manager (not a parked thread per watch): the serve host is 2-core. A
 * blocking recv() thread per watch would burn a thread and a stack for every
 * idle-but-alive session (up to ~max_sessions per model) and steal scheduling
 * from request handling. Instead ONE background thread holds every watch fd in
 * a single poll() loop (readability on a held GET body-less connection means
 * the peer closed — the same MSG_PEEK==0 signal hs_client_alive uses), plus a
 * self-pipe for wakeups and a periodic grace sweep. Cost is ~one fd per watch
 * and a handful of syscalls, independent of the watch count.
 *
 * The connection thread that accepted the GET /session/watch hands the fd to
 * the manager (pn_session_watch_add) and RETURNS — it does not loop or close
 * the fd. The manager owns the fd from then on (closes it when the watch ends).
 *
 * @status Built; CPU-testable (poll loop + grace reaper), CUDA compile-checked
 *         on the K80 test machine. NOT deployed (serve --fresh is user-gated).
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_SESSION_WATCH_H
#define PN_SERVE_SESSION_WATCH_H

#include "model_registry.h"

/* Opaque watch manager. Concrete definition in session_watch.c. */
typedef struct pn_session_watch_mgr pn_session_watch_mgr;

/**
 * Grace after a watch connection closes before the session is released, in ms.
 * A reconnecting watch for the same session within this window cancels the
 * release. Default 45 s — comfortably longer than the client's resilient-reconnect
 * backoff (RESUME_RETRY_MAX ~6, a few s each) plus link-recovery margin, so a
 * bridge hiccup never false-releases. Override: PN_SESSION_WATCH_GRACE_MS.
 */
uint64_t pn_session_watch_grace_ms(void);

/**
 * Long safety backstop, in ms: the ONLY inactivity-based reaper left, and it
 * only touches sessions with NO watch at all (a pathological/forgotten slot
 * whose client never opened a watch and never returned). A watched — or
 * within-grace — session is NEVER reaped by this. Default 12 h. Override:
 * PN_SESSION_IDLE_TTL_MS (repurposed from the old short idle-TTL).
 */
uint64_t pn_session_watch_backstop_ms(void);

/**
 * Create + start the watch manager over `reg` (borrowed; must outlive the
 * manager). Spawns the single poll/grace background thread. Returns the manager
 * or NULL on failure (self-pipe / thread). One per serve process.
 */
pn_session_watch_mgr *pn_session_watch_start(mr_registry *reg);

/**
 * Stop the manager: signal the poll thread, join it, close every held watch fd
 * and the self-pipe, free the manager. NULL-safe. Call at shutdown only.
 */
void pn_session_watch_stop(pn_session_watch_mgr *mgr);

/**
 * Register a held GET /session/watch connection. `fd` is the accepted client
 * socket (OWNED by the manager from here on: it closes it when the watch ends —
 * peer close, replacement by a reconnect for the same key, or shutdown). `entry`
 * is the resolved registry entry whose prefix_cache holds the session; the
 * caller must have taken an mr_acquire reference — the manager takes its OWN
 * reference for the fd's lifetime and releases it when the fd is dropped, so the
 * caller releases its reference as usual after this returns. `name` (model id,
 * for logging) and `sid` (the prefix-cache session key) are copied.
 *
 * A watch arriving for a (name,sid) that already has a live watch CANCELS the
 * old one's grace (if pending) and REPLACES it — the reconnect case. On success
 * the manager sends the HTTP 200 SSE-style keepalive header on `fd` itself
 * (so the client sees the watch is established) and holds it.
 *
 * @return 0 on success (fd is owned by the manager), -1 on failure (fd is NOT
 *         adopted — the caller still owns and must close it). Failure reasons:
 *         NULL args, no prefix cache on the entry, table full, OOM.
 */
int pn_session_watch_add(pn_session_watch_mgr *mgr, int fd, mr_entry *entry,
                         const char *name, const char *sid);

/**
 * Is the live prefix-cache slot `entry_sid` on model `name` covered by a
 * live-or-in-grace liveness watch? The client holds ONE watch per tab keyed on the
 * tab's BASE id `client-<pid>-<tab>`, while the live KV slots are the
 * mode-suffixed sids `<base>-orch`/`-act`/`-chat` — so a watch on the base covers
 * `entry_sid` when entry_sid == "<base>" or "<base>-<suffix>". Returns nonzero if
 * so. Bound (via pn_session_watch_install_backstop_hooks) as each cache's
 * watch-liveness predicate so the 12 h backstop reaper never reaps a watched
 * base's mode slots. Thread-safe: yes (takes the manager mutex). NULL-safe.
 */
int pn_session_watch_base_is_watched(pn_session_watch_mgr *mgr, const char *name,
                                     const char *entry_sid);

/**
 * Install the backstop watch-hook (pn_session_watch_base_is_watched) on every
 * loaded model's prefix cache, so the backstop reaper skips slots whose base is
 * still watched. Call once after the manager starts and models are loaded. Safe
 * to call again (idempotent) if models load later. NULL-safe.
 */
void pn_session_watch_install_backstop_hooks(pn_session_watch_mgr *mgr);

/**
 * Test-only: run one grace-sweep pass at wall-clock `now_ms` WITHOUT the
 * background thread (the unit test drives time deterministically). Releases any
 * session whose watch closed more than the grace ago with no reconnect. Not for
 * production use (the started manager sweeps on its own). NULL-safe.
 */
void pn_session_watch_sweep_for_test(pn_session_watch_mgr *mgr, uint64_t now_ms);

/**
 * Test-only: mark an already-registered watch's connection as closed at
 * `now_ms` (as if poll() saw the peer hang up), starting its grace WITHOUT a
 * real socket close. Used by the unit test to exercise the drop→grace→release
 * and drop→reconnect paths deterministically. Returns 0 if the watch was found,
 * -1 otherwise. NULL-safe.
 */
int pn_session_watch_mark_closed_for_test(pn_session_watch_mgr *mgr,
                                          const char *name, const char *sid,
                                          uint64_t now_ms);

#endif /* PN_SERVE_SESSION_WATCH_H */
