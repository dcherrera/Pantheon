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
 *     (`bs_slot.sid` / `.hist`).  On this path the `pn_prefix_cache` holds
 *     ONLY admission bookkeeping — nothing calls its acquire, so it has no
 *     entries and no session state whatsoever.
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

#endif /* PN_SERVE_MODEL_SESSION_H */
