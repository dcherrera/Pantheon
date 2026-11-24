/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file prefix_cache.c
 * @brief Session-based prefix cache (see prefix_cache.h). Keeps a cached
 *        pn_state per conversation of a session id (see find_conversation)
 *        so an agent turn prefills only the delta
 *        tokens appended to a fixed prefix instead of re-prefilling the whole
 *        context. The heavy lifting is already in the model's batched forward
 *        (it continues from the current state at an arbitrary pos0); this
 *        module is the session bookkeeping: which pn_state belongs to which
 *        session, the longest-common-prefix delta, LRU eviction, and — the one
 *        subtlety — preserving a session's small SSM state off the shared GPU
 *        buffers before another session reuses them (the KV cache already
 *        survives in each pn_state's host mirror).
 *
 * @status Increment 1: session-based; LRU eviction. SYS1: per-session
 *         system-prefix snapshot + restore (see prefix_cache.h). LCP-REWIND:
 *         plain-KV (llama-family) arches resume at the longest common prefix
 *         on history divergence (no snapshot needed — pure causal KV rows are
 *         position-invariant, so pos is simply truncated). MULTI-CONVO: one
 *         id holds an entry per conversation it carries; max_sessions counts
 *         session families (session_family.h). Content dedup is follow-up.
 * @issues None
 * @todo Increment 2: content-addressed cross-session prefix dedup.
 */

#include "prefix_cache.h"
#include "session_family.h"
#include "session_snapshot.h"
#include "../models/qwen35moe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PN_PREFIX_SID_MAX 64

/* One cached conversation of a session (an id may hold several, each its own
 * entry). state == NULL marks a free slot. */
typedef struct {
    char      sid[PN_PREFIX_SID_MAX];
    pn_state *state;      /* owns the session's KV + SSM + pos */
    uint32_t *tokens;     /* committed token history (prompt + generated) */
    int       n_tok;
    int       cap_tok;
    uint64_t  stamp;      /* LRU: last acquire/commit tick (monotonic counter) */
    uint64_t  last_active_ms;  /* wall-clock ms of last acquire/commit; 0 = never.
                                * Drives the Phase-5 idle-TTL reaper (the `stamp`
                                * counter is only an ordering, not a duration). */
    /* SYS1 — system-prefix boundary state. sys_snap is the dual-state image at
     * pos == sys_len (pn_qwen35moe_state_snapshot); sys_tokens[0..sys_len) are
     * the exact prefix tokens it covers (the reuse guard: a changed system
     * prompt never matches). Survives history divergence resets; replaced when
     * the system prompt changes; freed with the entry. */
    void     *sys_snap;
    uint32_t *sys_tokens;
    int       sys_len;
    /* TURN — the recurrent snapshot at the start of the last turn that carried
     * think_history (pn_qwen35moe_state_snapshot_recurrent: SSM images + q4
     * shadows, no KV rows); turn_tokens[0..turn_n) are the request tokens it
     * covers. A drop_last request restores it over the KV rows the state still
     * holds, so it is valid only while no row below turn_n is rewritten. */
    void     *turn_snap;
    uint32_t *turn_tokens;
    int       turn_n;
} pn_prefix_entry;

struct pn_prefix_cache {
    pn_model        *model;   /* not owned */
    pn_prefix_entry *ents;
    int              cap;     /* == max_sessions */
    uint64_t         clock;   /* monotonic LRU tick */
    /* The session whose image currently occupies the shared GPU state
     * buffers (or NULL). Snapshotted back to its host before another session
     * overwrites those buffers. */
    pn_state        *live;
    /* Watch-liveness predicate (borrowed): the backstop reaper skips an idle slot
     * whose base session is still watched. NULL = no watch protection. */
    pn_session_watched_fn watch_fn;
    void                 *watch_ctx;
    /* Session families holding entries, republished after every change to an
     * entry's state/sid (publish_session_count) so pn_prefix_cache_session_count
     * reads it without the cache's lock. */
    int                   session_count;
};

/* Length of the longest common prefix of two token runs. */
static int lcp_len(const uint32_t *a, int na, const uint32_t *b, int nb)
{
    int n = na < nb ? na : nb;
    int i = 0;
    while (i < n && a[i] == b[i])
        i++;
    return i;
}

/* Snapshot the live session's SSM state off the shared GPU buffers into its
 * host pn_state, so it survives another session reusing those buffers. */
static void preserve_live(pn_prefix_cache *c)
{
    if (!c->live)
        return;
#ifdef USE_CUDA
    /* Bonsai shares the qwen35moe SSM/GPU state machinery, so it snapshots the
     * same way. */
    if (c->model->config.arch == PN_ARCH_QWEN35MOE ||
        c->model->config.arch == PN_ARCH_BONSAI)
        pn_qwen35moe_gpu_state_to_host(c->model, c->live);
#endif
}

pn_prefix_cache *pn_prefix_cache_new(pn_model *model, int max_sessions)
{
    if (!model || max_sessions < 1)
        return NULL;
    pn_prefix_cache *c = calloc(1, sizeof(*c));
    if (!c)
        return NULL;
    c->ents = calloc((size_t)max_sessions, sizeof(*c->ents));
    if (!c->ents) {
        free(c);
        return NULL;
    }
    c->model = model;
    c->cap = max_sessions;
    return c;
}

void pn_prefix_cache_set_watch_hook(pn_prefix_cache *c, pn_session_watched_fn fn,
                                    void *ctx)
{
    if (!c)
        return;
    c->watch_fn = fn;
    c->watch_ctx = ctx;
}

/* Does this model's attention K/V live on an ACCELERATOR rather than in the host state?
 *
 * Every snapshot in this file serialises pn_state's host K/V. On the Tenstorrent on-card
 * attention arm that mirror is deliberately never written — 32,768 B a position stops
 * crossing the link, which is the entire point of the arm — so a capture there records ZEROS
 * and a restore hands the forward an empty cache. The output stays fluent and is wrong, with
 * nothing refusing anywhere. So ask before capturing, and decline the whole mechanism rather
 * than capture something that is not the session.
 *
 * Declining costs a system-prefix re-prefill on a divergent history. Capturing the wrong
 * bytes costs a wrong answer nobody can attribute, and the second is not a trade.
 *
 * A build with no accelerator, and an accelerator arm that left attention on the host, both
 * answer 0 and keep the existing behaviour exactly. */
static int kv_is_off_host(const pn_prefix_cache *c)
{
#ifdef USE_TT
    return pn_qwen35moe_model_tt_kv_on_card(c->model);
#else
    (void)c;
    return 0;
#endif
}

/* Is this cache's model one whose dual state supports snapshot/restore?
 * (Bonsai shares the qwen35moe dual-state machinery.) */
static int sys_snapshot_supported(const pn_prefix_cache *c)
{
    if (kv_is_off_host(c))
        return 0;
    return c->model->config.arch == PN_ARCH_QWEN35MOE ||
           c->model->config.arch == PN_ARCH_BONSAI;
}

/* SESSION EXPORT/RESTORE — can this model's session be captured into a blob a
 * client can hold and hand back? A strict SUPERSET of sys_snapshot_supported(),
 * and deliberately a SEPARATE predicate.
 *
 * The SYS1 machinery above decides WHEN the serve snapshots on its own account
 * (at the system-prompt boundary, to survive a divergent history) and is
 * qwen35-only by design — the llama family rewinds instead, see
 * lcp_rewind_supported() below. Export is a different question: any arch whose
 * whole per-session state can be serialized can be exported, and every one of
 * those two properties is owned by session_snapshot.h. Merging the predicates
 * would have turned SYS1 on for the llama family as a side effect of making
 * export work, i.e. changed the resume ladder on the hot path to fix an admin
 * endpoint. */
static int export_supported(const pn_prefix_cache *c)
{
    /* Same reason as above, and it is the LOUDER failure of the two: an exported blob leaves
     * the box, and a client that hands back a session of zeroed K/V gets a coherent answer to
     * a conversation the model cannot see. */
    if (kv_is_off_host(c))
        return 0;
    return pn_session_snapshot_supported(c->model->config.arch);
}

/* LCP-REWIND — is this model's per-session state a plain llama-family KV cache
 * (pure causal attention, host-resident K/V rows, no recurrent component)?
 * For those, row [i] of the KV cache depends only on tokens [0..i], so the
 * state can be rewound to ANY l <= pos by truncating pos — the rows [0,l)
 * stay exact and the next prefill simply overwrites [l..). This gives a
 * divergent-history resume WITHOUT a snapshot: resume at the longest common
 * prefix. Excluded: the recurrent (RWKVX) and hybrid-SSM (QWEN35MOE/BONSAI)
 * arches, whose per-position state cannot be rewound — that is exactly what
 * the SYS1 boundary snapshots above exist for. */
static int lcp_rewind_supported(const pn_prefix_cache *c)
{
    switch (c->model->config.arch) {
    case PN_ARCH_LLAMA:
    case PN_ARCH_QWEN2:
    case PN_ARCH_QWEN3:
    case PN_ARCH_PHI:
    case PN_ARCH_MISTRAL:
    case PN_ARCH_MIXTRAL:
        return 1;
    default:
        return 0;
    }
}

/* Replace an entry's committed history with tokens[0, n). 0, or -1 on OOM. */
static int entry_set_tokens(pn_prefix_entry *e, const uint32_t *tokens, int n)
{
    if (e->cap_tok < n) {
        uint32_t *t = realloc(e->tokens, (size_t)n * sizeof(uint32_t));
        if (!t)
            return -1;
        e->tokens = t;
        e->cap_tok = n;
    }
    memcpy(e->tokens, tokens, (size_t)n * sizeof(uint32_t));
    e->n_tok = n;
    return 0;
}

#ifdef USE_TT
/* THE CARD TIER (Tenstorrent: K/V and the GDN state live on the card, one sequence at a time).
 * First choice: the batched prefill's checkpoint (qwen35moe_prefill_checkpoint_tt.c) — the
 * output of a resumed turn is the output of a cold prefill of the same prompt, and it covers the
 * re-rendered history (lcp < n_tok) the append tier cannot. Then the append tier, but only when
 * the card still holds this session's sequence: another session's forwards since then left its
 * own K/V and state there. The checkpoint is never used past `ceiling`. Returns 1 (resumed,
 * *delta set), 0 (fall through to the append tier), -1 (full reset). */
static int tt_card_tier(pn_prefix_cache *c, pn_prefix_entry *e, const uint32_t *tokens,
                        int n_ctx, int lcp, int ceiling, int *delta)
{
    int p;

    if (!kv_is_off_host(c))
        return 0;
    /* The checkpoint resumes only when its position is below the token count it is given, so
     * passing ceiling + 1 keeps it at or below the resume ceiling (think_history as_sent). */
    p = pn_qwen35moe_tt_checkpoint_resume(c->model, e->state, tokens,
                                          ceiling < n_ctx ? ceiling + 1 : n_ctx);
    if (p > 0) {
        fprintf(stderr, "[prefix_cache] card checkpoint sid=%s: resume at %d, prefill %d of %d "
                "tokens [lcp=%d n_tok=%d]\n", e->sid, p, n_ctx - p, n_ctx, lcp, e->n_tok);
        if (entry_set_tokens(e, tokens, p) != 0)
            return -1;
        *delta = p;
        return 1;
    }
    if (p != 0)
        return -1;
    return pn_qwen35moe_tt_card_holds(c->model, e->state) ? 0 : -1;
}
#endif

/* Drop an entry's system-prefix snapshot (keeps the tokens buffer for reuse).
 * `sys_snap` is produced ONLY by pn_prefix_cache_capture_sys, which is gated on
 * sys_snapshot_supported(), so it is always a qwen35-family snapshot and the
 * typed free below is correct. If SYS1 is ever opened to another family this
 * must go through session_snapshot.h like the export path does. */
static void sys_snap_drop(pn_prefix_entry *e)
{
    pn_qwen35moe_state_snapshot_free(e->sys_snap);
    e->sys_snap = NULL;
    e->sys_len = 0;
}

/* Drop an entry's turn snapshot (keeps the tokens buffer for reuse). */
static void turn_snap_drop(pn_prefix_entry *e)
{
    pn_qwen35moe_state_snapshot_free(e->turn_snap);
    e->turn_snap = NULL;
    e->turn_n = 0;
}

/* The entry's state was replaced or reset: neither snapshot describes it. */
static void entry_snaps_drop(pn_prefix_entry *e)
{
    sys_snap_drop(e);
    turn_snap_drop(e);
}

void pn_prefix_cache_free(pn_prefix_cache *c)
{
    if (!c)
        return;
    for (int i = 0; i < c->cap; i++) {
        if (c->ents[i].state)
            pn_state_free(c->ents[i].state);
        free(c->ents[i].tokens);
        pn_qwen35moe_state_snapshot_free(c->ents[i].sys_snap);
        free(c->ents[i].sys_tokens);
        pn_qwen35moe_state_snapshot_free(c->ents[i].turn_snap);
        free(c->ents[i].turn_tokens);
    }
    free(c->ents);
    free(c);
}

static int entry_has_sid(const pn_prefix_entry *e, const char *sid)
{
    return e->state && strncmp(e->sid, sid, PN_PREFIX_SID_MAX) == 0;
}

/* The entry `sid`'s last turn ran on: its most recently acquired, committed or
 * imported conversation.  Generation is serialized, so between an acquire and
 * its commit the acquired entry is this one; capture, commit, reset, touch and
 * export all address it. */
static pn_prefix_entry *find_entry(pn_prefix_cache *c, const char *sid)
{
    pn_prefix_entry *best = NULL;
    for (int i = 0; i < c->cap; i++)
        if (entry_has_sid(&c->ents[i], sid) &&
            (!best || c->ents[i].stamp > best->stamp))
            best = &c->ents[i];
    return best;
}

/* Does a live entry belong to `sid`'s session family? */
static int entry_in_family(const pn_prefix_entry *e, const char *sid)
{
    return e->state && pn_session_same_family(e->sid, sid);
}

/* Live entries in `sid`'s session family. */
static int family_entry_count(const pn_prefix_cache *c, const char *sid)
{
    int n = 0;
    for (int i = 0; i < c->cap; i++)
        n += entry_in_family(&c->ents[i], sid);
    return n;
}

/* Distinct session families holding at least one entry (what max_sessions
 * counts — the batched scheduler counts the same way). */
static int live_family_count(const pn_prefix_cache *c)
{
    int n = 0;
    for (int i = 0; i < c->cap; i++) {
        const pn_prefix_entry *e = &c->ents[i];
        int first = e->state != NULL;
        for (int j = 0; j < i && first; j++)
            if (entry_in_family(&c->ents[j], e->sid))
                first = 0;
        n += first;
    }
    return n;
}

static void publish_session_count(pn_prefix_cache *c)
{
    __atomic_store_n(&c->session_count, live_family_count(c), __ATOMIC_RELEASE);
}

int pn_prefix_cache_session_count(const pn_prefix_cache *c)
{
    return c ? __atomic_load_n(&c->session_count, __ATOMIC_ACQUIRE) : 0;
}

int pn_prefix_cache_max_sessions(const pn_prefix_cache *c)
{
    return c ? c->cap : 0;
}

/* Free a single live slot in place (frees its pn_state + snapshot, keeps the
 * tokens buffer for reuse), so a waiting session can take it. */
static void free_slot(pn_prefix_cache *c, pn_prefix_entry *e)
{
    if (!e->state)
        return;
    if (c->live == e->state)
        c->live = NULL;
    pn_state_free(e->state);
    e->state = NULL;
    e->n_tok = 0;
    e->sid[0] = '\0';
    e->last_active_ms = 0;
    entry_snaps_drop(e);
    publish_session_count(c);
}

pn_session_admit pn_prefix_cache_admit(pn_prefix_cache *c, const char *sid,
                                       uint64_t now_ms, uint64_t idle_ttl_ms,
                                       int *restart_tokens_out)
{
    if (restart_tokens_out)
        *restart_tokens_out = 0;
    /* No cache or no session id → uncached path: nothing to admit against. */
    if (!c || !sid || !sid[0])
        return PN_SESSION_ADMIT_OK;

    /* Step 1: reap idle slots (last activity older than the TTL). Skip the slot
     * that currently owns the shared GPU buffers only if it is `sid`'s own —
     * an idle OTHER session is fine to reap even if live-buffer-resident, since
     * free_slot drops c->live for it. A slot whose base session still holds a
     * liveness watch (this idle TTL is the 12 h backstop, which only touches
     * UNwatched slots) is skipped — the watch, not the backstop, owns its
     * release: the client watches the BASE id while the live slot is the mode-suffixed
     * sid, so the watch protects the whole `<base>-*` prefix. */
    if (idle_ttl_ms > 0) {
        for (int i = 0; i < c->cap; i++) {
            pn_prefix_entry *e = &c->ents[i];
            if (!e->state || strncmp(e->sid, sid, PN_PREFIX_SID_MAX) == 0)
                continue;
            if (c->watch_fn && c->watch_fn(e->sid, c->watch_ctx))
                continue;   /* base is watched → the watch manager owns release */
            if (e->last_active_ms != 0 &&
                now_ms >= e->last_active_ms &&
                now_ms - e->last_active_ms >= idle_ttl_ms) {
                fprintf(stderr, "[prefix_cache] reaped idle session sid=%s "
                        "(idle %llums >= %llums)\n", e->sid,
                        (unsigned long long)(now_ms - e->last_active_ms),
                        (unsigned long long)idle_ttl_ms);
                free_slot(c, e);
            }
        }
    }

    /* Step 2: an existing session always resumes — and so does a new id of
     * its family ("<id>-<more>"), which shares its seat. */
    pn_prefix_entry *mine = find_entry(c, sid);
    if (mine) {
        if (restart_tokens_out)
            *restart_tokens_out = mine->n_tok;
        return PN_SESSION_ADMIT_OK;
    }
    if (family_entry_count(c, sid) > 0)
        return PN_SESSION_ADMIT_OK;

    /* Step 3: a free slot (fresh or just-reaped) admits the new session, and
     * so does a full pool holding fewer than max_sessions families: some
     * family then holds a second conversation, which alloc_entry gives up for
     * this one. */
    for (int i = 0; i < c->cap; i++)
        if (!c->ents[i].state)
            return PN_SESSION_ADMIT_OK;
    if (live_family_count(c) < c->cap)
        return PN_SESSION_ADMIT_OK;

    /* Step 4: max_sessions live, non-idle families → refuse the new one. */
    return PN_SESSION_ADMIT_AT_CAPACITY;
}

void pn_prefix_cache_touch(pn_prefix_cache *c, const char *sid, uint64_t now_ms)
{
    if (!c || !sid || !sid[0])
        return;
    pn_prefix_entry *e = find_entry(c, sid);
    if (e)
        e->last_active_ms = now_ms;
}

/* The entry a full pool gives up for a new conversation of `sid`: the least
 * recently used among `sid`'s own family's entries and every family's
 * second-and-later conversations, so one family's extra conversation never
 * costs another family its only one.  Only when every entry is some other
 * family's only conversation (a new family at max_sessions, which admission
 * refuses before this) does it fall back to the plain LRU. */
static pn_prefix_entry *eviction_victim(pn_prefix_cache *c, const char *sid)
{
    pn_prefix_entry *lru = NULL, *any = NULL;
    for (int i = 0; i < c->cap; i++) {
        pn_prefix_entry *e = &c->ents[i];
        if (!any || e->stamp < any->stamp)
            any = e;
        if ((entry_in_family(e, sid) || family_entry_count(c, e->sid) > 1) &&
            (!lru || e->stamp < lru->stamp))
            lru = e;
    }
    return lru ? lru : any;
}

/* Return a free slot for a new conversation of `sid`, evicting per
 * eviction_victim when the pool is full. NULL only on a pn_state_new failure. */
static pn_prefix_entry *alloc_entry(pn_prefix_cache *c, const char *sid)
{
    pn_prefix_entry *slot = NULL;
    for (int i = 0; i < c->cap; i++)
        if (!c->ents[i].state) { slot = &c->ents[i]; break; }

    if (!slot) {
        pn_prefix_entry *lru = eviction_victim(c, sid);
        fprintf(stderr, "[prefix_cache] evict sid=%s (%d tokens) for a new "
                "conversation of sid=%s\n", lru->sid, lru->n_tok, sid);
        if (c->live == lru->state)
            c->live = NULL;
        pn_state_free(lru->state);
        lru->state = NULL;
        lru->n_tok = 0;                 /* keep tokens buffer for reuse */
        entry_snaps_drop(lru);
        slot = lru;
    }

    slot->state = pn_state_new(c->model);
    if (!slot->state) {
        publish_session_count(c);
        return NULL;
    }
    pn_state_reset(slot->state);
    snprintf(slot->sid, PN_PREFIX_SID_MAX, "%s", sid);
    slot->n_tok = 0;
    publish_session_count(c);
    return slot;
}

/* SYS1 — does the entry hold a snapshot covering EXACTLY this system prefix? */
static int sys_snap_matches(const pn_prefix_entry *e, const uint32_t *tokens,
                            int sys_len)
{
    return e->sys_snap && e->sys_len == sys_len &&
           memcmp(e->sys_tokens, tokens,
                  (size_t)sys_len * sizeof(uint32_t)) == 0;
}

/* What an entry's cached state is worth to a prompt (higher wins). */
enum { PC_FIT_NONE, PC_FIT_SHARED, PC_FIT_SYS, PC_FIT_WARM };

/* Mirrors the acquire ladder's token checks, as bs_slot_fit does for the
 * batched slots: the committed history is a prefix of the prompt (append), or
 * the boundary snapshot covers its system prefix, or — only where the ladder
 * can resume mid-history (the llama-family LCP rewind, the Tenstorrent card
 * checkpoint) — the prompt shares at least its whole system prefix. */
static int entry_fit(const pn_prefix_cache *c, const pn_prefix_entry *e,
                     const uint32_t *tokens, int n_ctx, int sys_len, int *lcp)
{
    *lcp = lcp_len(e->tokens, e->n_tok, tokens, n_ctx);
    if (e->n_tok > 0 && *lcp == e->n_tok)
        return PC_FIT_WARM;
    if (sys_len > 0 && sys_len < n_ctx && sys_snap_matches(e, tokens, sys_len))
        return PC_FIT_SYS;
    if (*lcp > 0 && *lcp >= sys_len &&
        (lcp_rewind_supported(c) || kv_is_off_host(c)))
        return PC_FIT_SHARED;
    return PC_FIT_NONE;
}

/* Which of `sid`'s conversations this prompt continues (one session id can
 * carry several: opencode sends a title request and the real turns under one
 * id, each with its own system prompt).  Best fit, then longest shared prefix,
 * then most recent.  NULL when none fits: the prompt is a new conversation and
 * gets its own entry instead of resetting another's. */
static pn_prefix_entry *find_conversation(pn_prefix_cache *c, const char *sid,
                                          const uint32_t *tokens, int n_ctx,
                                          int sys_len)
{
    pn_prefix_entry *best = NULL;
    int best_fit = PC_FIT_NONE, best_lcp = 0;
    for (int i = 0; i < c->cap; i++) {
        pn_prefix_entry *e = &c->ents[i];
        int lcp = 0;
        if (!entry_has_sid(e, sid))
            continue;
        int fit = entry_fit(c, e, tokens, n_ctx, sys_len, &lcp);
        if (fit == PC_FIT_NONE)
            continue;
        if (!best || fit > best_fit || (fit == best_fit && lcp > best_lcp) ||
            (fit == best_fit && lcp == best_lcp && e->stamp > best->stamp)) {
            best = e;
            best_fit = fit;
            best_lcp = lcp;
        }
    }
    return best;
}

/* SYS1 — restore the session's state to its system-prefix boundary. On the GPU
 * path the state's GPU shadows (SSM image, KV sync watermark, fp16 window
 * ring) still hold the END of the previous turn; a restore to a non-zero pos
 * never trips the pos==0 restart re-sync, so the bindings must be forgotten
 * explicitly — the next forward then re-seeds everything from the restored
 * host mirror. Returns 0 on success (state sits at pos == sys_len). */
static int sys_snap_restore(pn_prefix_cache *c, pn_prefix_entry *e, int sys_len)
{
    if (!sys_snapshot_supported(c))
        return -1;
    if (pn_qwen35moe_state_restore(e->state->kv_cache, e->sys_snap) != 0)
        return -1;
    e->state->pos = sys_len;
#ifdef USE_CUDA
    pn_qwen35moe_gpu_forget_state(c->model, e->state);
#endif
    return 0;
}

/* TURN tier (think_history drop_last): when `tokens` start with the entry's
 * turn snapshot tokens and the state still holds those KV rows (committed
 * history agrees that far and pos is at or past it), put the recurrent state
 * back at the turn start and resume there. Same re-seed protocol as SYS1; the
 * MTP head's backlog restarts at the restored position. 1 resumed (*delta). */
static int turn_snap_resume(pn_prefix_cache *c, pn_prefix_entry *e,
                            const char *sid, const uint32_t *tokens, int n_ctx,
                            int *delta)
{
    int n = e->turn_n;
    if (!e->turn_snap || n <= 0 || n >= n_ctx || e->state->pos < n ||
        lcp_len(e->tokens, e->n_tok, tokens, n_ctx) < n ||
        memcmp(e->turn_tokens, tokens, (size_t)n * sizeof(uint32_t)) != 0)
        return 0;
    if (pn_qwen35moe_state_restore(e->state->kv_cache, e->turn_snap) != 0) {
        turn_snap_drop(e);
        return 0;
    }
    e->state->pos = n;
#ifdef USE_CUDA
    pn_qwen35moe_gpu_forget_state(c->model, e->state);
    (void)pn_qwen35moe_mtp_rewind(c->model, e->state);
#else
    (void)c;
#endif
    e->n_tok = n;   /* committed history truncated to the turn start */
    *delta = n;
    fprintf(stderr, "[prefix_cache] turn snapshot sid=%s: resume at %d, "
            "prefill %d of %d tokens\n", sid, n, n_ctx - n, n_ctx);
    return 1;
}

/* DIAG — the tokens straddling a divergence at `l`: `committed` is what we
 * generated and cached, `incoming` what the client sent back for the same span.
 * Printing from their first difference shows a re-render mismatch directly
 * instead of inferring it from lengths. */
static void log_divergence(const pn_prefix_entry *e, const uint32_t *tokens,
                           int n_ctx, int l, int width)
{
    if (l >= e->n_tok)
        return;
    fprintf(stderr, "[prefix_cache]   committed[%d..%d]:", l, e->n_tok - 1);
    for (int i = l; i < e->n_tok && i < l + width; i++)
        fprintf(stderr, " %u", e->tokens[i]);
    fprintf(stderr, "\n[prefix_cache]   incoming [%d..%d]:", l, n_ctx - 1);
    for (int i = l; i < n_ctx && i < l + width; i++)
        fprintf(stderr, " %u", tokens[i]);
    fprintf(stderr, "\n");
}

/* LCP-REWIND tier: rewind a plain-KV entry to the longest common prefix `l`
 * and prefill only from there. This is what makes a warmed prefix pay off on
 * these arches — the next turn's prompt shares the (large) system prefix but
 * diverges right after it (assistant opener / No-Think think block / a fresh
 * conversation), which the exact-prefix append check can never hit. Clamps:
 * never past the state's own position (rows beyond pos were never filled),
 * never past `ceiling`, and leave at least one prompt token to prefill so the
 * next-token logits are recovered. Returns the resume position (0: reset). */
static int lcp_rewind(pn_prefix_entry *e, const char *sid, int l, int n_ctx,
                      int ceiling)
{
    int l2 = l;
    if (l2 > e->state->pos)
        l2 = e->state->pos;
    if (l2 >= n_ctx)
        l2 = n_ctx - 1;
    if (l2 > ceiling)
        l2 = ceiling;
    if (l2 <= 0) {
        pn_state_reset(e->state);
        e->n_tok = 0;
        return 0;
    }
    e->state->pos = l2;
    e->n_tok = l2;   /* committed history truncated to the LCP */
    fprintf(stderr, "[prefix_cache] lcp rewind sid=%s: resume at %d, prefill "
            "%d of %d tokens\n", sid, l2, n_ctx - l2, n_ctx);
    return l2;
}

/* The acquire ladder over an existing entry: the turn snapshot (drop_last),
 * then card checkpoint (TT), append, system prefix, LCP rewind, full reset —
 * none of those resuming past the ceiling.
 * Returns the tier that ran and sets *delta. */
static pn_resume_kind resume_entry(pn_prefix_cache *c, pn_prefix_entry *e,
                                   const char *sid, const uint32_t *tokens,
                                   int n_ctx,
                                   const pn_prefix_acquire_opts *opts,
                                   int *delta)
{
    int sys_len = opts->sys_len, ceiling = opts->resume_ceiling;
    if (opts->drop_last && turn_snap_resume(c, e, sid, tokens, n_ctx, delta))
        return PN_RESUME_TURN_CKPT;
    int sys_ok = sys_len > 0 && sys_len < n_ctx && sys_len <= ceiling;
    int l = lcp_len(e->tokens, e->n_tok, tokens, n_ctx);
    int card = 0;
#ifdef USE_TT
    card = tt_card_tier(c, e, tokens, n_ctx, l, ceiling, delta);
#endif
    if (card == 1)
        return PN_RESUME_TURN_CKPT;   /* resumed from the card's checkpoint */
    if (card == 0 && l == e->n_tok && e->state->pos == e->n_tok &&
        l <= ceiling) {
        /* Append case (the agentic turn): the whole cached history is a
         * prefix of the new context and the state sits exactly at its end. */
        *delta = l;
        return PN_RESUME_EXACT;
    }
    if (sys_ok && sys_snap_matches(e, tokens, sys_len) &&
        sys_snap_restore(c, e, sys_len) == 0) {
        /* Divergent history but identical system prefix (the reasoning-model
         * turn): resume at the system boundary, prefill only the
         * conversation; commit rebuilds the history at end of turn. DIAG:
         * `lcp` is how far the new context agrees with the committed history:
         * lcp == n_tok means `pos` disqualified the append (state left
         * mid-turn); lcp < n_tok means the client re-rendered the conversation
         * and `lcp - sys_len` is re-prefilled because this arch has no
         * lcp-rewind. Read the FIRST of these after a fresh turn — later ones
         * follow the n_tok = 0 below and only report the reset. */
        fprintf(stderr, "[prefix_cache] sys-prefix reuse sid=%s: resume at "
                "%d, prefill %d of %d tokens "
                "[diag lcp=%d n_tok=%d pos=%d wasted=%d ceiling=%d]\n",
                sid, sys_len, n_ctx - sys_len, n_ctx, l, e->n_tok,
                e->state->pos, l > sys_len ? l - sys_len : 0, ceiling);
        log_divergence(e, tokens, n_ctx, l, 32);
        e->n_tok = 0;
        *delta = sys_len;
        return PN_RESUME_SYS_PREFIX;
    }
    if (lcp_rewind_supported(c)) {
        *delta = lcp_rewind(e, sid, l, n_ctx, ceiling);
        return *delta > 0 ? PN_RESUME_PARTIAL : PN_RESUME_COLD;
    }
    /* DIAG — a session that MISSED every tier re-prefills in full; say why,
     * or a client re-rendering its history differently is invisible. */
    fprintf(stderr, "[prefix_cache] full reset sid=%s: lcp=%d n_tok=%d pos=%d "
            "ceiling=%d (no tier: append needs lcp == n_tok == pos <= "
            "ceiling)\n", sid, l, e->n_tok, e->state->pos, ceiling);
    log_divergence(e, tokens, n_ctx, l, 16);
    pn_state_reset(e->state);
    e->n_tok = 0;
    *delta = 0;
    return PN_RESUME_COLD;
}

pn_state *pn_prefix_cache_acquire_bounded(pn_prefix_cache *c, const char *sid,
                                          const uint32_t *tokens, int n_ctx,
                                          const pn_prefix_acquire_opts *opts,
                                          pn_prefix_acquire_result *out)
{
    if (!c || !sid || !tokens || n_ctx <= 0 || !opts || !out)
        return NULL;
    int sys_len = opts->sys_len;
    /* A system prefix is only usable when a conversation follows it. */
    int sys_ok = sys_len > 0 && sys_len < n_ctx;
    out->delta_start = 0;
    out->capture_sys = 0;
    out->kind = PN_RESUME_COLD;

    pn_prefix_entry *e = find_conversation(c, sid, tokens, n_ctx, sys_len);
    int delta = 0;
    pn_resume_kind kind = PN_RESUME_COLD;

    if (e) {
        kind = resume_entry(c, e, sid, tokens, n_ctx, opts, &delta);
        /* KV rows from the resume point on are rewritten by this prefill. */
        if (e->turn_snap && delta < e->turn_n)
            turn_snap_drop(e);
    } else {
        /* A new session, or a new conversation of this one: its own entry, so
         * the conversations it already holds keep their caches. */
        if (find_entry(c, sid))
            fprintf(stderr, "[prefix_cache] new conversation sid=%s (%d held)\n",
                    sid, family_entry_count(c, sid));
        e = alloc_entry(c, sid);
        if (!e)
            return NULL;
    }

    /* Full prefill ahead + no snapshot covering these exact system tokens →
     * ask the caller to capture the boundary state (replacing a stale snapshot
     * from a changed system prompt). */
    if (delta == 0 && sys_ok && sys_snapshot_supported(c) &&
        !sys_snap_matches(e, tokens, sys_len))
        out->capture_sys = 1;

    /* Any forward the caller now runs re-seeds/re-syncs this session's GPU
     * state (host_state / KV host-id mismatch), overwriting whatever session
     * last occupied the shared buffers — so preserve that one first. */
    if (c->live && c->live != e->state)
        preserve_live(c);
    c->live = e->state;

    e->stamp = ++c->clock;
    out->delta_start = delta;
    out->kind = delta > 0 ? kind : PN_RESUME_COLD;
    return e->state;
}

pn_state *pn_prefix_cache_acquire_sys(pn_prefix_cache *c, const char *sid,
                                      const uint32_t *tokens, int n_ctx,
                                      int sys_len, int *delta_start,
                                      int *capture_sys)
{
    if (!delta_start || !capture_sys)
        return NULL;
    pn_prefix_acquire_result out = { 0, 0, PN_RESUME_COLD };
    pn_prefix_acquire_opts opts = { sys_len, n_ctx, false };
    pn_state *st = pn_prefix_cache_acquire_bounded(c, sid, tokens, n_ctx,
                                                   &opts, &out);
    *capture_sys = out.capture_sys;
    if (st)
        *delta_start = out.delta_start;
    return st;
}

pn_state *pn_prefix_cache_acquire(pn_prefix_cache *c, const char *sid,
                                  const uint32_t *tokens, int n_ctx,
                                  int *delta_start)
{
    int capture_unused = 0;
    return pn_prefix_cache_acquire_sys(c, sid, tokens, n_ctx, 0, delta_start,
                                       &capture_unused);
}

int pn_prefix_cache_capture_sys(pn_prefix_cache *c, const char *sid,
                                const uint32_t *tokens, int sys_len)
{
    if (!c || !sid || !tokens || sys_len <= 0)
        return -1;
    if (!sys_snapshot_supported(c))
        return -1;
    pn_prefix_entry *e = find_entry(c, sid);
    /* The capture protocol: acquire_sys set capture_sys, the caller prefilled
     * exactly the system prefix — the state must sit at that boundary. */
    if (!e || e->state->pos != sys_len)
        return -1;

#ifdef USE_CUDA
    /* Make the host mirror self-contained (GPU SSM image D2H + stale KV
     * mirror refresh) before copying it. Continuing the prefill afterwards is
     * unaffected: the D2H reads the GPU state without invalidating it. */
    if (pn_qwen35moe_gpu_state_to_host(c->model, e->state) != 0)
        return -1;
#endif

    void *snap = pn_qwen35moe_state_snapshot(e->state->kv_cache, sys_len);
    if (!snap)
        return -1;
    uint32_t *st = realloc(e->sys_tokens, (size_t)sys_len * sizeof(uint32_t));
    if (!st) {
        pn_qwen35moe_state_snapshot_free(snap);
        return -1;
    }
    memcpy(st, tokens, (size_t)sys_len * sizeof(uint32_t));
    pn_qwen35moe_state_snapshot_free(e->sys_snap);
    e->sys_snap = snap;
    e->sys_tokens = st;
    e->sys_len = sys_len;
    fprintf(stderr, "[prefix_cache] sys-prefix captured sid=%s: %d tokens\n",
            sid, sys_len);
    return 0;
}

bool pn_prefix_cache_turn_snapshot_supported(const pn_prefix_cache *c)
{
    return c && sys_snapshot_supported(c);
}

bool pn_prefix_cache_drop_last_supported(const pn_prefix_cache *c)
{
    if (!c)
        return false;
#ifdef USE_TT
    if (kv_is_off_host(c))
        return pn_qwen35moe_tt_checkpoint_available(c->model) != 0;
#endif
    return pn_prefix_cache_turn_snapshot_supported(c);
}

int pn_prefix_cache_capture_turn(pn_prefix_cache *c, const char *sid,
                                 const uint32_t *tokens, int cut)
{
    if (!pn_prefix_cache_turn_snapshot_supported(c) || !sid || !tokens ||
        cut <= 0)
        return -1;
    pn_prefix_entry *e = find_entry(c, sid);
    if (!e || e->state->pos != cut)
        return -1;
#ifdef USE_CUDA
    /* Same as the system-prefix capture: the host mirror must hold the SSM
     * image and the KV rows [0, cut) the restore will resume over. */
    if (pn_qwen35moe_gpu_state_to_host(c->model, e->state) != 0)
        return -1;
#endif
    void *snap = pn_qwen35moe_state_snapshot_recurrent(e->state->kv_cache, cut);
    if (!snap)
        return -1;
    uint32_t *tt = realloc(e->turn_tokens, (size_t)cut * sizeof(uint32_t));
    if (!tt) {
        pn_qwen35moe_state_snapshot_free(snap);
        return -1;
    }
    memcpy(tt, tokens, (size_t)cut * sizeof(uint32_t));
    pn_qwen35moe_state_snapshot_free(e->turn_snap);
    e->turn_snap = snap;
    e->turn_tokens = tt;
    e->turn_n = cut;
    return 0;
}

int pn_prefix_cache_commit(pn_prefix_cache *c, const char *sid,
                           const uint32_t *tokens, int n_ctx)
{
    if (!c || !sid || !tokens || n_ctx < 0)
        return -1;
    pn_prefix_entry *e = find_entry(c, sid);
    if (!e)
        return -1;

    if (entry_set_tokens(e, tokens, n_ctx) != 0)
        return -1;
    e->stamp = ++c->clock;
    return 0;
}

void *pn_prefix_cache_export_snapshot(pn_prefix_cache *c, const char *sid,
                                      int *n_tok_out, const uint32_t **tokens_out)
{
    if (n_tok_out) *n_tok_out = 0;
    if (tokens_out) *tokens_out = NULL;
    if (!c || !sid || !export_supported(c))
        return NULL;
    pn_prefix_entry *e = find_entry(c, sid);
    /* Export the FULL committed history. The append path leaves the state at
     * pos == n_tok; a mid-turn/diverged state (pos != n_tok) is not a clean
     * export point — refuse rather than snapshot an inconsistent position. */
    if (!e || e->n_tok <= 0 || e->state->pos != e->n_tok)
        return NULL;

#ifdef USE_CUDA
    /* Make the host mirror self-contained before copying (GPU SSM image D2H +
     * stale-KV refresh). This session must own the shared GPU buffers for the
     * D2H to read its live state — preserve any OTHER live session first, then
     * claim `live` for this one (a read-only D2H does not disturb it). */
    if (c->live && c->live != e->state)
        preserve_live(c);
    c->live = e->state;
    if (pn_qwen35moe_gpu_state_to_host(c->model, e->state) != 0)
        return NULL;
#endif

    void *snap = pn_session_snapshot_capture(c->model->config.arch,
                                             e->state->kv_cache, e->n_tok);
    if (!snap)
        return NULL;
    if (n_tok_out) *n_tok_out = e->n_tok;
    if (tokens_out) *tokens_out = e->tokens;
    return snap;
}

/* The conversation of `sid` an imported history replaces: one whose committed
 * history it continues, or which continues it, or an empty one (most recent
 * first). NULL when the history belongs to none of them. */
static pn_prefix_entry *import_target(pn_prefix_cache *c, const char *sid,
                                      const uint32_t *tokens, int n_tok)
{
    pn_prefix_entry *best = NULL;
    for (int i = 0; i < c->cap; i++) {
        pn_prefix_entry *e = &c->ents[i];
        if (!entry_has_sid(e, sid))
            continue;
        int shorter = e->n_tok < n_tok ? e->n_tok : n_tok;
        if (lcp_len(e->tokens, e->n_tok, tokens, n_tok) == shorter &&
            (!best || e->stamp > best->stamp))
            best = e;
    }
    return best;
}

int pn_prefix_cache_import_snapshot(pn_prefix_cache *c, const char *sid,
                                    const void *snap, const uint32_t *tokens,
                                    int n_tok)
{
    if (!c || !sid || !snap || !tokens || n_tok <= 0)
        return -1;
    if (!export_supported(c))
        return -1;
    if (pn_session_snapshot_pos(c->model->config.arch, snap) != n_tok)
        return -1;

    /* Take (or reuse) a slot. The caller already ran admission, so a
     * genuinely-new sid at capacity was refused upstream. A returning sid
     * reuses the conversation this history continues (or is a prefix of);
     * anything else is another conversation of that sid and gets its own entry
     * (alloc_entry: a free slot, else per eviction_victim). */
    pn_prefix_entry *e = import_target(c, sid, tokens, n_tok);
    int is_new = (e == NULL);
    if (!e) {
        e = alloc_entry(c, sid);
        if (!e)
            return -1;
    }

    /* Restore the snapshot into the slot's live state. Validate BEFORE mutating
     * pos/history so a rejected restore (dim mismatch — only reachable if the
     * blob is from a different arch, which weights_version already guards) never
     * corrupts the slot. A brand-new entry is freed on failure so it does not
     * linger as an empty live conversation; a reused one keeps its prior state
     * intact (we have not touched it yet). */
    if (pn_session_snapshot_restore(c->model->config.arch, e->state->kv_cache,
                                    snap) != 0) {
        if (is_new)
            free_slot(c, e);
        return -1;
    }
    e->state->pos = n_tok;
#ifdef USE_CUDA
    /* A restore to a non-zero pos never trips the pos==0 restart re-sync, so
     * forget the GPU bindings: the next forward re-seeds from the restored host
     * mirror. This session now owns the shared GPU buffers. */
    pn_qwen35moe_gpu_forget_state(c->model, e->state);
    if (c->live && c->live != e->state)
        preserve_live(c);
    c->live = e->state;
#endif

    /* Install the committed history (drives next turn's delta computation). On a
     * token-buffer OOM here the state is already restored; free a brand-new slot
     * (same contract as the restore-reject above) so it does not linger as an
     * empty live session — a returning sid keeps its slot (it self-heals to a
     * full prefill next turn, its old history being harmless). */
    if (e->cap_tok < n_tok) {
        uint32_t *t = realloc(e->tokens, (size_t)n_tok * sizeof(uint32_t));
        if (!t) {
            if (is_new)
                free_slot(c, e);
            return -1;
        }
        e->tokens = t;
        e->cap_tok = n_tok;
    }
    memcpy(e->tokens, tokens, (size_t)n_tok * sizeof(uint32_t));
    e->n_tok = n_tok;
    entry_snaps_drop(e);         /* the restored state is the new truth */
    e->stamp = ++c->clock;
    return 0;
}

void pn_prefix_cache_reset_session(pn_prefix_cache *c, const char *sid)
{
    if (!c || !sid)
        return;
    pn_prefix_entry *e = find_entry(c, sid);
    if (!e)
        return;
    pn_state_reset(e->state);   /* drop the (partial) KV/SSM state */
    e->n_tok = 0;               /* no committed history → next acquire full-prefills */
    entry_snaps_drop(e);        /* invalidate the system-prefix and turn snapshots */
}

void pn_prefix_cache_release_session(pn_prefix_cache *c, const char *sid)
{
    if (!c || !sid)
        return;
    /* Free the whole session (unlike reset_session, which keeps it occupied):
     * every conversation's pn_state + snapshot, returning capacity immediately
     * so a waiting session can take it. No-op for an unknown session. */
    for (int i = 0; i < c->cap; i++)
        if (entry_has_sid(&c->ents[i], sid))
            free_slot(c, &c->ents[i]);
}

void pn_prefix_cache_release_by_prefix(pn_prefix_cache *c,
                                       const char *base_prefix)
{
    if (!c || !base_prefix || !base_prefix[0])
        return;
    /* Free every live slot whose sid begins with base_prefix (the mode-suffixed
     * `<base>-orch`/`-act`/`-chat` slots a single base watch covers). A dead
     * base's watch releases all of them at once. */
    size_t plen = strlen(base_prefix);
    for (int i = 0; i < c->cap; i++) {
        pn_prefix_entry *e = &c->ents[i];
        if (e->state && strncmp(e->sid, base_prefix, plen) == 0)
            free_slot(c, e);
    }
}
