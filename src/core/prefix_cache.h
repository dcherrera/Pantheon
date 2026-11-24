/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file prefix_cache.h
 * @brief Session-based prefix cache — the agentic-turn latency win. Across
 *        agent turns the context is mostly unchanged (a tool result / message
 *        appended to a fixed prefix); instead of re-prefilling the WHOLE
 *        context every turn, a session's model state (attention KV + the small
 *        fixed SSM recurrent state + position) is kept keyed by a session id,
 *        so a follow-up prefills ONLY the new (delta) tokens.
 *
 * @status Increment 1: session-based caching (one cached pn_state per session
 *         id, LRU eviction). SYS1: system-prefix caching — the state at the
 *         system-prompt token boundary is snapshotted per session, so a turn
 *         whose history DIVERGES (the reasoning-model case: generated <think>
 *         text is committed but never resent by the client) still resumes at
 *         the system boundary and re-prefills only the conversation.
 *         MULTI-CONVO: an id carrying several conversations (opencode's title
 *         request + real turns) keeps an entry per conversation; the acquire
 *         picks the one the prompt continues (single-user bs_slot_reroute).
 * @issues Measured on a K80 host (2026-07-18, K80, Qwen3.6-35B Q3_K): a sys-prefix
 *         RESTORED turn is answer-correct but NOT bit-identical to a cold
 *         full prefill of the same context at temp 0 — greedy output matches
 *         for ~50 tokens then flips (reasoning cp ~180 chars), while two cold
 *         full prefills ARE byte-identical to each other.  So the capture-time
 *         host mirror (pn_qwen35moe_gpu_state_to_host) is not byte-faithful
 *         to the live GPU state it snapshots (host-side snapshot/restore
 *         round-trip IS byte-exact — unit-proven).  Applies equally to the
 *         single-user and batched paths (same primitives).  Follow-up: audit
 *         the D2H mirror fidelity (fp16/q4 shadow round-trip, SSM precision).
 * @todo
 *   - [ ] Increment 2: content-addressed prefix dedup (match arbitrary token
 *         prefixes across sessions; would let ONE system snapshot serve many
 *         sessions).  Batched-path SYS1 is done — it lives per-slot in
 *         batch_sched.[ch] (bs_prefill_resume_sys / bs_prefill_capture_sys),
 *         not here: the scheduler's slots own their pn_states, so sharing this
 *         cache's per-session states would mix two ownership models.
 *
 * @warning On the BATCHED path (`mr_entry.sched` non-NULL) this cache is NOT
 *          USED — nothing calls its acquire, so it has no entries and no
 *          session state at all; the sessions live in the scheduler's decode
 *          slots, and the scheduler is also that path's admission authority
 *          (batch_sched_admission.h).  Serve code must therefore never reach
 *          for `entry->prefix_cache` directly to find, admit or count a
 *          session: go through the model_session.h facade, which has both
 *          arms.  Session export, restore and release each shipped against
 *          this cache alone and each silently covered only the single-user
 *          path until that facade existed; admission did the same until
 *          2026-11-05 (an empty cache admits everyone, so max_sessions was
 *          never enforced on a batched model).
 */

#ifndef PN_PREFIX_CACHE_H
#define PN_PREFIX_CACHE_H

#include "engine.h"
#include "think_history.h"
#include <stdint.h>

/* Opaque session prefix cache. Concrete definition in prefix_cache.c. */
typedef struct pn_prefix_cache pn_prefix_cache;

/**
 * Watch-liveness predicate: "is the base session that owns entry `sid` currently
 * covered by a live (or within-grace) liveness watch?" The serve layer binds this
 * to the session-watch manager (see pn_prefix_cache_set_watch_hook). It exists so
 * the 12 h backstop reaper in pn_prefix_cache_admit does NOT reap a mode-suffixed
 * slot (`<base>-orch`/`-act`/`-chat`) while a watch on its base `<base>` is alive:
 * The client holds ONE watch per tab keyed on the tab's BASE id, but the live KV slots
 * are the mode-suffixed sids, so the watch must protect the whole `<base>-*`
 * prefix. Returns nonzero when `sid`'s base is watched (skip the backstop).
 * Called under the same lock as admit (a model lock); it may take the watch
 * manager's own mutex — safe because the watch manager NEVER holds its mutex
 * across a model lock (the reverse edge), so no lock cycle forms.
 */
typedef int (*pn_session_watched_fn)(const char *sid, void *ctx);

/* Session-admission verdict (K80 serving-ops Phase 5). A "session" is a live
 * prefix-cache slot. The cache holds up to max_sessions of them; a genuinely
 * NEW session id over that cap is REFUSED (not queued, not LRU-evicting a live
 * session) so "how many concurrent users this model holds" is a hard number.
 * An idle slot (untouched past the TTL) is reaped first so a waiting user can
 * take it. An EXISTING session id always resumes. */
typedef enum {
    PN_SESSION_ADMIT_OK           = 0,  /* existing sid resumes, or slot free */
    PN_SESSION_ADMIT_AT_CAPACITY  = 1,  /* new sid, all slots hold live sessions */
} pn_session_admit;

/**
 * Create a session prefix cache over `model`, holding up to `max_sessions`
 * live sessions (each owns a pn_state whose VRAM KV cache is the bulk of the
 * cost). Least-recently-used sessions are evicted when the cache is full.
 *
 * Entries are per CONVERSATION: one id may hold several (one per distinct
 * conversation it sends — see pn_prefix_cache_acquire_sys). The pool is still
 * `max_sessions` entries in all, so memory stays at most `max_sessions`
 * pn_states. A new conversation takes a free entry, else evicts the least
 * recently used among its own id's entries and every id's second-and-later
 * ones — never another id's only conversation (admission counts IDS, so a
 * side conversation never makes a new id or the main one 429).
 *
 * @param model         Loaded model (weights + forward). Not owned.
 * @param max_sessions  Max concurrent cached sessions (>= 1).
 * @return              Cache handle, or NULL on failure.
 *
 * Thread-safe: no
 */
pn_prefix_cache *pn_prefix_cache_new(pn_model *model, int max_sessions);

/**
 * Install the watch-liveness predicate the backstop reaper consults (see
 * pn_session_watched_fn). `fn`/`ctx` are borrowed (must outlive the cache); pass
 * `fn == NULL` to clear it (the default — no watch protection, backstop reaps any
 * idle slot). Set once by the serve after the watch manager starts. Thread-safe:
 * no (set before serving, then read-only under the cache's lock).
 */
void pn_prefix_cache_set_watch_hook(pn_prefix_cache *cache,
                                    pn_session_watched_fn fn, void *ctx);

/** Free the cache and every pn_state it still holds. */
void pn_prefix_cache_free(pn_prefix_cache *cache);

/**
 * Session families holding entries now (what max_sessions counts). A snapshot
 * republished on every entry change, so it may be read WITHOUT the lock that
 * serializes the cache (e.g. under the registry lock for sessions_used and
 * replica routing). 0 for a NULL cache.
 *
 * Thread-safe: yes (atomic read). Reentrant: yes. Signal-safe: no
 */
int pn_prefix_cache_session_count(const pn_prefix_cache *cache);

/**
 * The cache's max_sessions (fixed at pn_prefix_cache_new). 0 for NULL.
 *
 * Thread-safe: yes (immutable). Reentrant: yes. Signal-safe: no
 */
int pn_prefix_cache_max_sessions(const pn_prefix_cache *cache);

/**
 * Session admission (K80 serving-ops Phase 5). Decide whether session `sid` may
 * take (or already holds) one of the cache's live slots — WITHOUT allocating.
 * Call this before pn_prefix_cache_acquire on the request path.
 *
 * Steps, in order:
 *   1. Reap idle slots: any slot untouched for >= `idle_ttl_ms` (its last
 *      acquire/commit was that long ago) is freed here, so a returning-later
 *      user frees room for a waiting one. `idle_ttl_ms <= 0` disables reaping.
 *   2. If `sid`'s session family (session_family.h: the id, or "<id>-<more>")
 *      already holds a slot → PN_ADMIT_OK (it always resumes).
 *   3. Else if a slot is free (or one was just reaped), or fewer than
 *      max_sessions families hold the slots (one holds a spare
 *      conversation the acquire will evict) → PN_ADMIT_OK.
 *   4. Else (max_sessions live, non-idle families) → PN_ADMIT_AT_CAPACITY.
 * Reaping is per conversation entry (each has its own last-activity time).
 *
 * On PN_SESSION_ADMIT_AT_CAPACITY the caller should REFUSE the new session
 * (429/503) rather than let acquire LRU-evict a live one. `*restart_tokens_out`
 * (may be NULL) receives the committed context length of the session that would
 * need re-prefilling if this were an expired/returning session — 0 for a truly
 * new sid — so the caller can quote a restart estimate.
 *
 * @param cache          Cache handle (NULL → OK, no-op — caching off).
 * @param sid            Session id (NULL/empty → OK — uncached path).
 * @param now_ms         Caller's monotonic wall clock in ms (time-source
 *                       agnostic: the serve layer passes its own now_ms()).
 * @param idle_ttl_ms    Idle-session TTL in ms (<= 0 disables the reaper).
 * @param restart_tokens_out  Out (nullable): committed tokens of `sid`'s slot.
 * @return               PN_SESSION_ADMIT_OK or PN_SESSION_ADMIT_AT_CAPACITY.
 *
 * Thread-safe: no (serialize with the same lock as acquire/commit).
 */
pn_session_admit pn_prefix_cache_admit(pn_prefix_cache *cache, const char *sid,
                                       uint64_t now_ms, uint64_t idle_ttl_ms,
                                       int *restart_tokens_out);

/**
 * Mark session `sid` as active at wall-clock `now_ms` (feeds the idle-TTL
 * reaper in pn_prefix_cache_admit). Like capture, commit, reset and export it
 * addresses the id's CURRENT conversation: the entry most recently acquired,
 * committed or imported (generation is serialized, so between an acquire and
 * its commit that is the acquired one). Call after a successful acquire/generation
 * on the request path. No-op for an unknown session, a NULL cache, or an
 * empty sid. Thread-safe: no.
 */
void pn_prefix_cache_touch(pn_prefix_cache *cache, const char *sid,
                           uint64_t now_ms);

/**
 * Acquire session `sid`'s cached model state for the full context
 * `tokens[0..n_ctx)` and report where the caller must resume prefilling.
 *
 * Restores the session (first snapshotting any OTHER currently-live session's
 * GPU state back to its host, so the shared GPU SSM buffers can be reused) and
 * returns its pn_state positioned so that tokens[0..*delta_start) are already
 * processed (state->pos == *delta_start). The caller then prefills
 * tokens[*delta_start..n_ctx) through the model's batched forward (which
 * advances state->pos) and generates, then calls pn_prefix_cache_commit.
 *
 * *delta_start == n_ctx means the whole context was already cached (nothing to
 * prefill); *delta_start == 0 means a fresh or diverged session (full prefill).
 *
 * @param cache        Cache handle.
 * @param sid          Session id (copied; NUL-terminated, up to 63 chars).
 * @param tokens       Full context token ids [0..n_ctx).
 * @param n_ctx        Context length (> 0).
 * @param delta_start  Out: first position the caller must prefill.
 * @return             The session's pn_state, or NULL on failure.
 *
 * Thread-safe: no
 */
pn_state *pn_prefix_cache_acquire(pn_prefix_cache *cache, const char *sid,
                                  const uint32_t *tokens, int n_ctx,
                                  int *delta_start);

/**
 * SYS1 — system-prefix-aware acquire. As pn_prefix_cache_acquire, but with the
 * caller's system-prompt token boundary `sys_len` (tokens[0..sys_len) are the
 * per-session-stable system prefix; 0 = no system prefix). Resume preference:
 *
 *   1. Full-history append resume (exact behavior of pn_prefix_cache_acquire):
 *      *delta_start = cached history length.
 *   2. System-prefix resume: the session holds a state snapshot at its system
 *      boundary AND tokens[0..sys_len) match the snapshotted prefix exactly →
 *      the snapshot is restored and *delta_start = sys_len (only the
 *      conversation is prefilled). Fires on the divergent-history turns where
 *      (1) never can (reasoning models: <think> text committed, never resent).
 *   3. Fresh/diverged with no usable snapshot: clean reset, *delta_start = 0.
 *
 * Which of the id's conversations: the entry whose cached state the prompt
 * continues — committed history a prefix of `tokens` (1), else a boundary
 * snapshot over tokens[0..sys_len) (2), else, on arches that resume
 * mid-history (llama LCP rewind, the Tenstorrent card checkpoint), the longest
 * shared prefix covering the system prefix. None → a NEW entry for this
 * conversation (case 3 on it), leaving the id's other conversations intact.
 *
 * *capture_sys is set to 1 when the caller should CAPTURE the system-prefix
 * state this turn: a full prefill is about to run (case 3), a system prefix
 * exists (0 < sys_len < n_ctx), and no valid snapshot covers these exact
 * system tokens (none yet, or the system prompt changed — the stale snapshot
 * is invalidated by the capture). The caller then prefills [0, sys_len),
 * calls pn_prefix_cache_capture_sys, and prefills the rest.
 *
 * Correctness: any mismatch falls to case 3 (full reset + prefill) — never a
 * partial/wrong reuse. Restore is exact: SSM state + KV rows + fp16 shadows
 * are byte-restored to the boundary and GPU shadows are re-seeded from them.
 *
 * @param sys_len      System-prefix token count (0 disables SYS1 for the call).
 * @param capture_sys  Out: 1 = capture the boundary state during this prefill.
 * @return             The session's pn_state, or NULL on failure.
 *
 * Thread-safe: no
 */
pn_state *pn_prefix_cache_acquire_sys(pn_prefix_cache *cache, const char *sid,
                                      const uint32_t *tokens, int n_ctx,
                                      int sys_len, int *delta_start,
                                      int *capture_sys);

/** What pn_prefix_cache_acquire_bounded is asked to honour. */
typedef struct {
    int  sys_len;          /* as pn_prefix_cache_acquire_sys (0: none) */
    int  resume_ceiling;   /* last prompt position a resume may start from */
    bool drop_last;        /* think_history drop_last: try the turn snapshot
                            * first (pn_prefix_cache_capture_turn) */
} pn_prefix_acquire_opts;

/** What pn_prefix_cache_acquire_bounded decided. */
typedef struct {
    int            delta_start;   /* as pn_prefix_cache_acquire_sys */
    int            capture_sys;   /* as pn_prefix_cache_acquire_sys */
    pn_resume_kind kind;          /* the tier that ran (usage `resume`) */
} pn_prefix_acquire_result;

/**
 * pn_prefix_cache_acquire_sys with a resume ceiling, reporting the tier that
 * ran. With drop_last, the turn snapshot is tried first: it resumes at the
 * snapshot's position when the prompt starts with its tokens. Otherwise no
 * tier resumes past prompt position `resume_ceiling`: the append tier
 * and the Tenstorrent card checkpoint are skipped when they would, the plain-KV
 * LCP rewind is clamped to it, and the system-prefix tier needs
 * sys_len <= resume_ceiling. A ceiling >= n_ctx is pn_prefix_cache_acquire_sys
 * exactly. `think_history: "as_sent"` passes the request's first think block
 * (core/think_history.h), so no earlier think block is resumed over.
 *
 * @param opts            system prefix, resume ceiling, drop_last
 * @param out             receives delta_start, capture_sys and the tier
 * @return                The session's pn_state, or NULL on failure.
 *
 * Thread-safe: no
 */
pn_state *pn_prefix_cache_acquire_bounded(pn_prefix_cache *cache,
                                          const char *sid,
                                          const uint32_t *tokens, int n_ctx,
                                          const pn_prefix_acquire_opts *opts,
                                          pn_prefix_acquire_result *out);

/** Can this cache keep turn snapshots (the qwen35 dual state with its KV on
 *  the host side: the archs SYS1 supports)? */
bool pn_prefix_cache_turn_snapshot_supported(const pn_prefix_cache *cache);

/** Does this cache serve think_history drop_last: the turn snapshot, or on
 *  Tenstorrent (K/V on the card) the card's multi-turn prefill checkpoint? */
bool pn_prefix_cache_drop_last_supported(const pn_prefix_cache *cache);

/**
 * Save session `sid`'s TURN snapshot: the caller just prefilled tokens[0, cut)
 * (the request's own messages, sg_prompt_bounds.turn_start) and the state sits
 * there. A recurrent snapshot (SSM images + q4 shadows; the KV rows stay in
 * the state) replaces the previous one. A later acquire with drop_last whose
 * prompt starts with tokens[0, cut) resumes there. Host RAM: the SSM images
 * (Qwen3.6 ~63 MiB), admitted like the system-prefix snapshot.
 *
 * @return 0 saved, -1 not (unsupported arch, wrong position, no memory)
 *
 * Thread-safe: no (serialize with acquire/commit)
 */
int pn_prefix_cache_capture_turn(pn_prefix_cache *cache, const char *sid,
                                 const uint32_t *tokens, int cut);

/**
 * How many of this turn's prompt tokens the cache let the caller SKIP — the
 * prompt tokens NOT computed this turn. This is exactly OpenAI's
 * `usage.prompt_tokens_details.cached_tokens`, and the only supported way to
 * derive it: it is a restatement of the resume boundary the cache ITSELF chose
 * (`delta_start`), never a second opinion recomputed from the prompt. A
 * recomputed number would measure the reporter instead of the cache — the
 * client's whole reason for wanting it is that `prompt_tokens / TTFT` silently
 * inflates with cache hits, so the correction has to come from the mechanism.
 *
 * Every resume rung lands in `delta_start`, so all of them are covered by
 * construction: full-history append, think-elided append (batched), system-
 * prefix snapshot restore, and the llama-family LCP rewind. A full prefill
 * (`delta_start == 0`) reports 0 — reported, not omitted, so a client can tell
 * "the serve said nothing was reused" from "the serve does not report this".
 *
 * The clamp is what makes the wire invariant `0 <= cached_tokens <= n_ctx`
 * unconditional, including for a caller that hands over a boundary from before
 * its own fallback reset.
 *
 * @param delta_start  The cache's chosen resume position for this turn.
 * @param n_ctx        This turn's prompt length in tokens.
 * @return             Reused prompt tokens, clamped to [0, n_ctx].
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline int pn_prefix_cached_tokens(int delta_start, int n_ctx)
{
    if (delta_start <= 0 || n_ctx <= 0)
        return 0;
    return delta_start < n_ctx ? delta_start : n_ctx;
}

/**
 * SYS1 — snapshot session `sid`'s state at its system-prompt boundary. Call
 * mid-prefill, when the session's state sits EXACTLY at pos == sys_len having
 * just prefilled tokens[0..sys_len) (the acquire's *capture_sys == 1 protocol).
 * Replaces any previous snapshot for the session. On the GPU path the live GPU
 * state is synced to the host first, so the snapshot is self-contained.
 *
 * @param tokens   The full prompt (tokens[0..sys_len) are copied as the guard).
 * @param sys_len  System-prefix token count (> 0).
 * @return         0 on success, -1 on failure (unknown session / pos mismatch /
 *                 unsupported architecture / allocation). Failure is benign:
 *                 the caller just continues its full prefill uncached.
 *
 * Thread-safe: no
 */
int pn_prefix_cache_capture_sys(pn_prefix_cache *cache, const char *sid,
                                const uint32_t *tokens, int sys_len);

/**
 * Commit session `sid`'s new committed token history (prompt + everything
 * generated this turn), tokens[0..n_ctx). The session's pn_state must already
 * be at pos == n_ctx (the caller prefilled the delta and stepped decode). The
 * history drives next turn's delta computation.
 *
 * @return 0 on success, -1 on failure (unknown session / allocation).
 *
 * Thread-safe: no
 */
int pn_prefix_cache_commit(pn_prefix_cache *cache, const char *sid,
                           const uint32_t *tokens, int n_ctx);

/**
 * Session warm-tier EXPORT (K80 serving-ops Phase 7). Snapshot session `sid`'s
 * FULL current committed state (pos == its committed history length) as an
 * opaque qwen35moe snapshot the caller then serializes to a portable blob. On
 * the GPU path the live GPU state is synced to the host first, so the snapshot
 * is self-contained (same discipline as pn_prefix_cache_capture_sys). The
 * session is left untouched (read-only export). An id holding several
 * conversations exports its most recently used one.
 *
 * @param cache      Cache handle.
 * @param sid        Session id (must be a live slot).
 * @param n_tok_out  Out (nullable): the committed token count the snapshot
 *                   covers (== snapshot pos).
 * @param tokens_out Out (nullable): borrowed pointer to the session's committed
 *                   token history (valid until the next cache op on this cache);
 *                   the caller copies it if it needs to outlive that.
 * @return           Opaque snapshot (free with pn_qwen35moe_state_snapshot_free),
 *                   or NULL (unknown session / empty history / unsupported arch
 *                   / GPU sync or snapshot failure).
 *
 * Thread-safe: no (serialize with the cache's lock).
 */
void *pn_prefix_cache_export_snapshot(pn_prefix_cache *cache, const char *sid,
                                      int *n_tok_out, const uint32_t **tokens_out);

/**
 * Session warm-tier IMPORT/RESTORE (K80 serving-ops Phase 7). Take (or reuse) a
 * live slot for `sid` and restore `snap` (a deserialized qwen35moe snapshot)
 * into its state, then install `tokens[0..n_tok)` as the committed history and
 * position the state at pos == n_tok, so the very next acquire is a full-history
 * append resume (delta prefill of just what the client sends beyond n_tok). The
 * caller must have already admitted the session (capacity) and validated the
 * blob's weights_version against this model. `snap`'s pos MUST equal n_tok.
 *
 * The restore replaces the id's conversation whose committed history `tokens`
 * continues, or is a prefix of (or an empty one); otherwise it is another
 * conversation of the id and takes its own entry, leaving the others intact.
 *
 * On the GPU path the state's GPU bindings are forgotten so the next forward
 * re-seeds from the restored host mirror (same as the SYS1 restore path).
 *
 * @param cache    Cache handle (must be non-NULL — restore requires caching).
 * @param sid      Session id (created if new; reused if already live).
 * @param snap     Deserialized snapshot to restore (pos must equal n_tok).
 * @param tokens   Committed token history to install [0..n_tok).
 * @param n_tok    Committed token count (> 0, == snap pos).
 * @return         0 on success, -1 on failure (bad args / pos mismatch / dim
 *                 mismatch / allocation / no free slot).
 *
 * Thread-safe: no (serialize with the cache's lock).
 */
int pn_prefix_cache_import_snapshot(pn_prefix_cache *cache, const char *sid,
                                    const void *snap, const uint32_t *tokens,
                                    int n_tok);

/**
 * Reset session `sid` to an empty state: clears its cached KV/SSM state,
 * committed history, and system-prefix snapshot, so the NEXT acquire does a
 * clean full prefill. Used to recover after a turn was ABORTED mid-prefill —
 * the partial, uncommitted prefill left state->pos ahead of the committed
 * history, which would break append-resume; this discards it. No-op for an
 * unknown session. Thread-safe: no.
 */
void pn_prefix_cache_reset_session(pn_prefix_cache *cache, const char *sid);

/**
 * Release session `sid`: free its slots outright — every conversation it holds
 * (state + snapshot freed, sid cleared), returning capacity to the pool IMMEDIATELY so a waiting session can be
 * admitted. Unlike pn_prefix_cache_reset_session (which keeps the slot occupied),
 * this drops the whole entry. Used by POST /session/release on a client's clean
 * shutdown so a departing session doesn't hold a slot until the idle-TTL reaps it.
 * No-op for an unknown session. Thread-safe: no.
 */
void pn_prefix_cache_release_session(pn_prefix_cache *cache, const char *sid);

/**
 * Release EVERY live session whose sid begins with `base_prefix` — the
 * prefix-match complement to pn_prefix_cache_release_session. A liveness watch is
 * held on a tab's BASE id `client-<pid>-<tab>`, but the live KV slots are the
 * mode-suffixed sids `<base>-orch`/`-act`/`-chat`; when that base's watch dies
 * past grace this frees all of its mode slots at once (pass `base_prefix` =
 * "<base>-"). Each freed slot returns capacity immediately. No-op for a NULL
 * cache, a NULL/empty prefix, or no match. Thread-safe: no (serialize with the
 * same lock as acquire/release).
 */
void pn_prefix_cache_release_by_prefix(pn_prefix_cache *cache,
                                       const char *base_prefix);

#endif /* PN_PREFIX_CACHE_H */
