/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_session.h
 * @brief THE session-state facade — one entry point per session-state
 *        operation, covering BOTH of the serve's session models.
 *
 * Why this file exists
 * ====================
 * A resident model keeps its per-session state in one of two places, decided
 * by whether the batch scheduler is active (`entry->sched`):
 *
 *   - SINGLE-USER path (`sched == NULL`): one `pn_state` per session inside
 *     the model's `pn_prefix_cache`, serialized by `gen_lock`.
 *   - BATCHED path (`sched != NULL`): one `pn_state` per DECODE SLOT inside
 *     the scheduler, with the session bound to the slot it last finalized on
 *     (`bs_slot.sid` / `.hist`), and the scheduler admits sessions
 *     (max_sessions).  On this path the `pn_prefix_cache` is not used —
 *     nothing calls its acquire, so it has no entries and no session state
 *     whatsoever.
 *
 * Every session feature written against `entry->prefix_cache` alone therefore
 * silently covers only the single-user path — which is the path a K80 host
 * does not run.  `/admin/session/export` returned `session_not_found` for
 * every batched session for exactly this reason, and `/session/restore` was
 * worse: it "succeeded" into a prefix-cache slot no batched generation ever
 * reads, allocating a whole unused `pn_state` in the process.  Session
 * release had the same hole.
 *
 * The rule this file encodes: **serve code never touches `entry->prefix_cache`
 * or `entry->sched` for a session operation — it calls an `ms_*` function.**
 * Each one has both arms and takes the right lock, so a feature added here is
 * a feature both paths get.  Adding a session operation means adding it here,
 * with both arms, or not at all.
 *
 * What is NOT unified (deliberate)
 * ================================
 * The two STORES stay separate.  Merging them — letting the scheduler's slots
 * borrow the prefix cache's per-session `pn_state`s — mixes two ownership
 * models (see the @todo in prefix_cache.h) and is an architecture change well
 * beyond a session-export fix.  What is unified is the CONTRACT: the same
 * snapshot object, the same portable blob format, the same
 * `weights_version` guard, and one call site per operation.
 *
 * @status Batched arm validated on a K80 host on saber-orch + nex-n2-mini.
 * @issues The warm append-resume after a restore only fires when the client
 *         replays the committed token prefix exactly (the `tokens[]` array the
 *         export returns).  Reasoning models whose clients omit the generated
 *         <think> text from the replay fall back to the system-prefix boundary
 *         resume or a full prefill — pre-existing, identical on both paths.
 * @todo None
 */

#ifndef PN_SERVE_MODEL_SESSION_H
#define PN_SERVE_MODEL_SESSION_H

#include <stdbool.h>
#include <stdint.h>

#include "model_registry.h"

/**
 * Monotonic wall clock in milliseconds — the time source every session
 * operation uses (the idle-TTL reaper's `now_ms`). Monotonic so an NTP step
 * cannot spuriously reap or immortalize a session.
 */
uint64_t ms_now_ms(void);

/**
 * Long safety-backstop idle TTL in ms (default 12 h; `PN_SESSION_IDLE_TTL_MS`
 * overrides, 0 disables). Idleness is NOT the normal release trigger — a
 * session's slot is held by its `GET /session/watch` connection, which frees it
 * when the client dies. This TTL only reaps a forgotten slot whose client never
 * opened a watch and never returned. Read once, then cached.
 */
uint64_t ms_session_idle_ttl_ms(void);

/* Outcome of a session-state operation. Maps 1:1 onto the HTTP answers so the
 * two handlers cannot drift apart in how they report the same condition. */
typedef enum {
    MS_OK           =  0,  /* done                                           */
    MS_NO_CACHE     = -1,  /* model has no session store at all → 400        */
    MS_NOT_FOUND    = -2,  /* no live session with that id → 404             */
    MS_BUSY         = -3,  /* session mid-turn, or at capacity → 429         */
    MS_BAD_SNAPSHOT = -4,  /* snapshot did not fit this model → 400          */
    MS_ERROR        = -5,  /* allocation / GPU materialize failure → 500     */
    MS_NOT_YOURS    = -6,  /* bound to another person → 403 session_not_yours */
} ms_result;

/**
 * Export session `sid`'s state as an opaque qwen35moe snapshot plus the
 * committed token history it covers.  Read-only w.r.t. the session.
 *
 * Batched arm: the session's warm decode slot; the KV rows are materialized
 * on demand from the GPU decode-slot ring (Phase 10b / INV3) — no host KV
 * mirror is maintained or resurrected.
 * Single-user arm: the model's prefix-cache entry for `sid`.
 *
 * @param snap_out    Out: snapshot (free with pn_qwen35moe_state_snapshot_free).
 * @param tokens_out  Out: malloc'd COPY of the committed history (caller frees).
 * @param n_tok_out   Out: committed token count (== the snapshot's pos).
 *
 * Thread-safe: yes (takes the model's session lock internally).
 */
ms_result ms_session_export(mr_entry *entry, const char *sid,
                            void **snap_out, uint32_t **tokens_out,
                            int *n_tok_out);

/**
 * Restore a deserialized snapshot as session `sid`'s live state, positioned at
 * pos == n_tok with `tokens[0..n_tok)` installed as the committed history, so
 * the next chat turn on `sid` resumes off it.  The caller MUST have validated
 * the blob's `weights_version` against `mr_weights_version(entry)` first — the
 * blob format carries the tag, this function does not re-check it.
 *
 * Thread-safe: yes.
 */
ms_result ms_session_restore(mr_entry *entry, const char *sid,
                             const void *snap, const uint32_t *tokens,
                             int n_tok);

/**
 * Release session `sid`: drop its state and return its capacity immediately
 * rather than waiting out the idle TTL.  Idempotent; no-op for an unknown
 * session.  Both arms, so a client's clean shutdown actually frees a batched
 * session's warm slot instead of leaving its context bound to that slot.
 *
 * Thread-safe: yes.
 */
void ms_session_release(mr_entry *entry, const char *sid);

/**
 * Release EVERY session whose id starts with `base_prefix` — what a dead
 * liveness watch fires (the client watches the BASE tab id while the live sessions
 * are the mode-suffixed `<base>-orch` / `-act` / `-chat` ids).  Both arms.
 *
 * Thread-safe: yes.
 */
void ms_session_release_by_prefix(mr_entry *entry, const char *base_prefix);

/**
 * Session families holding state on this model now — what max_sessions counts
 * and what `sessions_used` reports.  Batched arm: families holding decode
 * slots (warm or mid-turn).  Single-user arm: families holding prefix-cache
 * entries.  A model with no session store (a CPU copy) has no seats, so its
 * sessions are its routing bindings (replica_affinity.h).
 *
 * Thread-safe: yes; lock-safe under the registry lock (takes at most
 * `sched->mu`, under which nothing reaches back into the registry).
 */
int ms_sessions_used(mr_entry *entry);

/** One consistent snapshot of a model's seats (ms_session_occupancy). */
typedef struct {
    int  sessions_used;  /* as ms_sessions_used */
    int  max_sessions;   /* the session limit admission enforces */
    int  slots_held;     /* decode slots some session holds; a family may hold
                          * more than one (== sessions_used without slots) */
    int  slots_free;     /* decode slots nobody holds */
    bool seat_free;      /* as ms_session_seat_free */
} ms_occupancy;

/**
 * Fill `out` with sessions, held and free slots and seat_free, read together.
 * sessions_used < max_sessions with seat_free false means some family holds
 * more than one slot.  All zero for NULL.
 *
 * Thread-safe: yes; lock-safe under the registry lock (as ms_sessions_used).
 */
void ms_session_occupancy(const mr_entry *entry, ms_occupancy *out);

/**
 * Could a NEW session be seated on this model now?  Batched arm: under
 * max_sessions and a decode slot nobody holds exists.  Single-user arm: under
 * max_sessions.  No session store: always (no cap is enforced there).  What
 * replica routing asks to keep a new session off a full replica.
 *
 * Thread-safe: yes; lock-safe under the registry lock (as ms_sessions_used).
 */
bool ms_session_seat_free(mr_entry *entry);

/**
 * Does `sid`'s session hold a seat on this model — state it would lose by
 * moving to another replica?  Batched arm: its family holds a decode slot
 * (warm or mid-turn).  Single-user arm and no session store: true (the cache is
 * serialized by gen_lock, which routing cannot take, so a binding there is
 * kept as before).  false for NULL / an empty sid.  A bound session that holds
 * no seat is placed like a new one (model_replica_route.c).
 *
 * Thread-safe: yes; lock-safe under the registry lock (as ms_sessions_used).
 */
bool ms_session_holds_seat(mr_entry *entry, const char *sid);

/* ── Serve sessions belong to a person (people registry spec §4.8) ──────────
 * Pantheon Local puts the caller's person in the thread principal
 * (hs_set_thread_principal). The first chat or restore of a session id binds it
 * to that person; a later use by another person is refused. One table for the
 * process, keyed by session id with the family rule (session_family.h), so a
 * binding follows a session that fails over to another copy; each binding
 * remembers the entry of its last bind. Release (and release by prefix) forgets
 * it. Plain `pantheon serve` has no principal: nothing is bound or refused. */

#define MS_PRINCIPAL_MAX          32
#define MS_PRINCIPAL_BINDINGS_MAX 512   /* table full -> the least recently used goes */
#define MS_NOT_YOURS_MESSAGE      "this session belongs to another person"

/* How a request uses a session id (ms_session_principal_check). */
typedef enum {
    MS_PRINCIPAL_BIND,    /* chat, restore: first use binds it to the principal */
    MS_PRINCIPAL_CHECK,   /* release: never binds; an unbound session passes */
    MS_PRINCIPAL_EXPORT,  /* export: never binds; an unbound session (opened before
                           * people) passes for an admin only */
} ms_principal_use;

/**
 * May `principal` use session `sid` this way? Refused when the sid, or any id of
 * its family, is bound to another person (or, for an export, when nothing of the
 * family is bound and the principal is not an admin). A BIND records the binding.
 * NULL / empty principal or sid: always MS_OK, nothing recorded.
 *
 * @return MS_OK or MS_NOT_YOURS
 *
 * Thread-safe: yes (one table mutex; a linear scan of the live bindings).
 */
ms_result ms_session_principal_check(mr_entry *entry, const char *sid, const char *principal,
                                     bool admin, ms_principal_use use);

/** One person's seats on one model (ms_seats_by_principal). */
typedef struct {
    char principal[MS_PRINCIPAL_MAX];
    int  held;   /* session families of this person holding a seat */
} ms_principal_seats;

/**
 * Seats held on `entry` per person, counting families as ms_session_occupancy
 * does. Batched arm: a bound family counts while it holds a decode slot here.
 * Single-user arm and no session store: a binding counts on the entry of its
 * last bind until it is released (the stores' idle reapers are not seen).
 * Unbound sessions are not counted.
 *
 * @return how many people `out` holds (at most `max`)
 *
 * Thread-safe: yes (copies the table, then asks the scheduler without the
 * table lock held).
 */
int ms_seats_by_principal(mr_entry *entry, ms_principal_seats *out, int max);

#endif /* PN_SERVE_MODEL_SESSION_H */
