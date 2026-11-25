/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched.h
 * @brief Per-model batch decode scheduler (C3: one card serves K users in one
 *        batched GPU forward, removing the same-model 429).
 *
 * Architecture
 * ============
 * Each `mr_entry` that carries a GPU-capable model gets a `pn_batch_sched`
 * (gated by QWEN35_BATCH_SLOTS=1; disabled → old trylock→429 behaviour is
 * unchanged).  The scheduler owns K `pn_state*` instances — one per slot —
 * allocated at model-load time (host RAM only; GPU slot state is lazily bound
 * on the first real request to avoid hanging on cuMemAlloc right after a long
 * weight upload).  A single background driver thread drives the
 * per-step `pn_qwen35moe_decode_slots` calls; request threads each prefill
 * their own prompt, bind their slot, then enter the slot's decode loop,
 * exchange tokens with the driver, and stream their own SSE.
 *
 * Concurrency model (the part three bugs came from — read this first)
 * ===================================================================
 * ONE mutex (`mu`) + a FIFO GPU-ownership turnstile built on it:
 *
 *   - GPU turnstile (`gpu_ticket_next` / `gpu_ticket_serving` / `cv_gpu`,
 *     all guarded by `mu`) — the model's single-GPU-owner "lock".  The
 *     model's single-user forward machinery (C2 async phase machine
 *     `g->as_phase`, per-die scratch, the shared live SSM buffers keyed by
 *     `d->host_state`, the shared GPU KV ring) supports exactly ONE
 *     in-flight generation; pre-C3 that was guaranteed by the per-model
 *     gen_lock, which the batched path does not take.  The turnstile
 *     restores the guarantee: a request thread owns the turn across a
 *     prefill CHUNK *and* the slot bind (so no other forward can run
 *     between the last prefill token and the bind / the chunk-boundary
 *     `state_to_host` — the "bind right after prefill" protocol the SSM
 *     D2D snapshot requires), and the driver owns it across each
 *     `decode_slots` step (batched decode shares `sc->r_logits` and the
 *     CUDA streams with the single-user path, so a step must never overlap
 *     a prefill token in flight).
 *
 *     Why a ticketed turnstile instead of a plain mutex (the pre-L2/L3
 *     `gpu_mu`): pthread mutexes allow barging.  With the driver
 *     free-running (L3: it no longer waits for owners between steps) a
 *     plain mutex lets it re-acquire the GPU ahead of a waiting or
 *     chunk-yielding prefiller indefinitely; and two alternating
 *     prefillers could likewise starve the driver.  FIFO tickets make the
 *     interleave provably fair: chunk → step → chunk → step.  A turn
 *     holder never cond-waits (chunks and steps are bounded GPU work), and
 *     every ticket is always followed by a release, so the turnstile is
 *     deadlock-free by construction.
 *
 *   - `mu` — guards slot status words, the token/logits handshake flags,
 *     the L3 per-slot output rings and the turnstile counters.  Every
 *     pthread_cond_wait predicate is re-checked in a while loop under
 *     `mu`; every transition another thread waits on is made (and
 *     signalled) under `mu`, so wakeups cannot be lost.
 *
 * Slot lifecycle per request (statuses under `mu`):
 *   IDLE -(bs_slot_acquire)-> PREFILL -(bs_slot_bind)-> BOUND
 *        -(driver step)----> DECODE  -(bs_slot_release)-> IDLE
 * Only the OWNER (the request thread) ever moves a slot back to IDLE — the
 * driver never frees a slot on its own, so a slot can never be re-acquired
 * while its previous owner might still touch it.
 *
 * Driver step:
 *   The driver steps only the active lanes (BOUND, or DECODE and not yet
 *   `finished`), each in its own block row.
 *   PREFILL lanes are simply excluded from the step — they no
 *   longer close the gate (pre-L2 they did): their pn_state is never
 *   passed to decode_slots, and GPU exclusivity against the prefill's own
 *   forwards is the turnstile's job.  This is what lets decode interleave
 *   a chunked prefill (L2).  The driver's gate is:
 *
 *     >=1 slot BOUND/DECODE (not finished)
 *         AND  every unfinished DECODE slot has token_ready.
 *
 *   With in-driver sampling (L3, default) token_ready is set by the driver
 *   itself right after sampling, so the gate is open continuously while
 *   any slot decodes — owners are OFF the step critical path entirely.
 *
 *   Liveness: every PREFILL leaves the state in finite time — prefill
 *   chunks are serialized by the turnstile and complete (bind → BOUND) or
 *   fail (release → IDLE); at most `cap` requests are admitted, and the
 *   FIFO tickets interleave decode steps between prefill chunks (L2), so
 *   decoders make progress even during a long join.  Every transition
 *   that can open the gate signals `cv_work`.
 *
 * Isolation guarantees:
 *   - Each slot has its own `pn_state*` (own KV ring, own SSM images via the
 *     slot-major conv_slots/ssm_slots arrays, own RoPE position).
 *   - Sampling: with L3 (default) the driver samples every slot itself,
 *     sequentially, with a per-slot copy of the request's sampling params +
 *     repeat window — per-slot streams are independent and, at temp 0,
 *     bit-identical to owner-side sampling (same sample_token, same recent
 *     window contents at every step).  With QWEN35_BS_DRIVER_SAMPLE=0 the
 *     legacy owner-side sampling (pn_sample_next_token, _Thread_local
 *     scratch) is restored.
 *   - The driver READS the next-token word and WRITES the logits pointer /
 *     output ring; owners read the ring (L3) or logits (legacy); all
 *     guarded by `mu` + the token_ready/logits_ready flags.
 *   - No request thread touches another slot's state, logits, or emitter.
 *
 * 1-user no-regression:
 *   With one active user the driver runs n_active=1 and calls
 *   decode_slots with K=1 (not the old fixed-cap-8 step); the turnstile is
 *   uncontended; pn_qwen35moe_decode_slots is documented bit-exact for
 *   K=1 vs. the single-user forward.  The SSE streaming path is unchanged.
 *   Measured goal: K=1 tok/s ≈ single-user non-batched (~13 tok/s), NOT
 *   the old 3.4 tok/s from 8-wide batching with 7 dummy lanes.
 *
 * Fail-soft:
 *   Any GPU failure (lazy init, slot bind, decode_slots) propagates as an
 *   error to the owning request thread(s) — logits==NULL wakeups, -1
 *   returns — which release their slots; nothing is ever left waiting on a
 *   condition nobody will signal.  All-slots-busy stays a clean 429.
 *
 * Session blocks + dynamic-width batching:
 *   Slot i owns GPU row i on the card for the scheduler's life — a ctx_cap
 *   KV ring row per attention layer and an SSM image per SSM layer, carved at
 *   bs_create (a model whose blocks do not fit is refused at load). Each step
 *   runs only the active (BOUND/DECODE) slots, passing their rows to
 *   pn_qwen35moe_decode_slots_rows, so IDLE slots pay zero compute and keep
 *   their warm state untouched: nothing is compacted, moved or copied when
 *   the active set changes. A join/leave/finish sets active_set_changed and
 *   the driver rebuilds gpu_map[] (bookkeeping only) before the next step.
 *
 * L3 — in-driver sampling + per-slot output ring (default on;
 * QWEN35_BS_DRIVER_SAMPLE=0 restores the legacy lockstep):
 *   bs_slot_bind_ex hands the driver the request's sampling params, the
 *   repeat window seeded with the prompt tail + first token, and a token
 *   budget (max_tokens - 1).  After each step the driver appends the
 *   just-stepped token to the slot's repeat window, samples the next token
 *   from the slot's (host-side, sparse top-k) logits, pushes it to the
 *   slot's output ring, and immediately re-arms token_ready — the step
 *   loop never waits on an owner again.  Owners drain the ring with
 *   bs_slot_pop_token and do detokenize + emitter + SSE entirely off the
 *   critical path: ONE STALLED CLIENT NO LONGER STALLS THE BATCH (the §3.5
 *   reliability hazard).  The driver stops stepping a slot (finished=true)
 *   on EOS/<|im_end|> or budget exhaustion; stop-STRING detection stays
 *   owner-side (text-level), so a slot may over-generate past a stop
 *   string until its owner releases — bounded by the budget, never
 *   affecting emitted output (surplus ring tokens are discarded).  The
 *   ring is sized to the full budget up front, so it can never overflow
 *   and the driver never blocks on a slow consumer.
 *
 * L2 — chunked prefill (bs_prefill_yield):
 *   The owner prefills in chunks (serve_generation), releasing the GPU
 *   turn between chunks so decode steps interleave.  At each yield the
 *   prefill so far is saved into the slot's own block on the card
 *   (decode_slot_bind; eager mode: the host mirror, its legacy protocol),
 *   so ANY forward that interleaves (a decode step, or even another
 *   prefill claiming the live SSM buffers / shared KV ring) is safe: the
 *   resumed chunk re-seeds from the block device to device via the ordinary
 *   seed-on-host-change path.  A joiner's decode stall is bounded by one
 *   chunk instead of the whole prompt.
 *
 * SYS1-batched — per-slot system-prefix caching (bs_prefill_resume_sys /
 * bs_prefill_capture_sys):
 *   The batched twin of the single-user SYS1 (prefix_cache.h): each slot may
 *   hold ONE host-side boundary snapshot (state at pos == sys_len), keyed by
 *   session id + the exact prefix tokens.  A divergent-history turn (the
 *   reasoning-model case, where the warm-history append resume can never
 *   fire) restores the snapshot into the slot's own pn_state and prefills
 *   only the conversation.  DESIGN: snapshots are per-SLOT (not a shared
 *   per-session store) — a deliberate conservative bound.  Each snapshot is
 *   a self-contained deep copy of the slot's own host mirrors, captured and
 *   restored only by the slot's owner under the GPU turnstile, so concurrent
 *   sessions in different slots can never read or clobber each other's
 *   state; the shared-GPU hazards are handled by the exact protocols the
 *   warm-history path already validated on a K80 host (capture = the L2-yield D2H
 *   that makes the host mirror authoritative; restore = host write + GPU
 *   forget + seed-on-host-change re-seed).  Cost of the conservatism: a
 *   session that lands on a foreign slot (its own busy) full-prefills and
 *   its capture replaces that slot's snapshot; the affinity scans (warm sid
 *   first, then snapshot sys_sid) make that rare.
 *
 * @status Validated on a K80 host (QWEN35_BATCH_SLOTS=1 to enable; default off).
 *         L1/L2/L3 gated by tests/test_serve_sched_qwen35 (serve-path
 *         parity vs the single-user reference, join-during-decode,
 *         slow-client isolation) + tests/test_users_qwen35 (engine gate).
 *         SYS1-batched host logic gated by tests/test_prefix_cache_sys;
 *         proven on a K80 host 2026-07-18 (K80 dies 2,3, Qwen3.6-35B Q3_K,
 *         QWEN35_BATCH_SLOTS=1, 2 concurrent sessions): per-session
 *         sys-prefix reuse on divergent turn 2 (TTFT 84.5s -> 3.5s at a
 *         ~3.1k-token system prefix), zero cross-session leakage.  Restored
 *         turns match the single-user SYS1's fidelity exactly (see the
 *         prefix_cache.h @issues note: answer-correct, not bit-identical to
 *         a cold prefill — pre-existing at 85a130e, same primitives).
 *         THINK-ELISION (Phase 11) measured on a K80 host on nex-n2-mini + saber-orch:
 *         append-resume 0/6 -> 5/6 turns, re-prefilled tokens 1848 -> 117,
 *         client TTFT flat with conversation depth instead of linear.
 *         REASONING-BEARING RESUME: the same rung now also matches a client
 *         that SENDS `reasoning_content` BACK — the convention that keeps the
 *         model reasoning at all and
 *         which, on the exact-history rung alone, re-prefilled every turn
 *         because a client cannot reproduce the ids sampled inside a think
 *         block.  Host logic gated by tests/test_prefix_cache_sys
 *         (test_reasoning_bearing_resume) + tests/test_prefix_projection; NOT
 *         yet measured on a K80 host end to end.
 * @issues A turn that reasons and emits NO visible answer cannot be
 *         projected, so it falls back to the SYS1 boundary (measured 3/6
 *         turns on saber-orch at max_tokens 200).  Tool-call turns are
 *         attempted but their client re-render was never exercised.
 *
 *         The SYS1 snapshot is a per-slot HOST copy (system-prefix KV + the
 *         full SSM image, ~62 MiB on qwen36) — the one per-session host
 *         footprint left; warm KV + SSM themselves live in the slot's block
 *         on the card (see batch_sched.c @issues).
 * @todo None
 */

#ifndef PN_SERVE_BATCH_SCHED_H
#define PN_SERVE_BATCH_SCHED_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "../../include/pantheon.h"
#include "../core/engine.h"
#include "../core/prefix_cache.h"   /* pn_session_watched_fn */
#include "../core/think_history.h"
#include "../core/host_stage_timer.h"
#include "serve_spec_mtp_cuda.h"
#include "batch_sched_turn_checkpoint.h"
#include "reasoning_budget.h"
#include "batch_sched_pd_budget.h"

/* Maximum concurrent USER slots this scheduler will manage.  8 to match the
 * K80 host die count and keep per-slot SSM images well within headroom. */
#define BS_MAX_USERS 8

/* Slot-array size. One block per user slot; the +1 is array headroom only
 * (no scratch row is carved since rows stopped moving). Must stay ≤
 * PN_QWEN35MOE_GPU_MAX_SLOTS (16). */
#define BS_MAX_SLOTS (BS_MAX_USERS + 1)

/* Per-slot state machine values (bs_slot.status). */
typedef enum {
    BS_SLOT_IDLE     = 0,   /* free, pn_state available (dummy decode lane) */
    BS_SLOT_PREFILL  = 1,   /* request thread owns the slot; prefill runs
                             * owning the GPU turnstile turn */
    BS_SLOT_BOUND    = 2,   /* prefill done, GPU SSM bound, waiting for driver */
    BS_SLOT_DECODE   = 3,   /* driver is stepping this slot */
} bs_slot_status;

/* One decode slot. */
typedef struct {
    pn_state        *state;         /* owned by sched; reset by the owner in
                                     * bs_prefill_begin (owning the GPU turn) */
    bs_slot_status   status;        /* guarded by sched->mu */
    uint32_t         next_token;    /* → driver: token for the next step (written
                                     * by the owner in legacy mode, by the driver
                                     * itself in L3 driver-sampling mode) */
    float           *logits;        /* driver → request thread (LEGACY mode only):
                                     * logits for sampling (pointer into state's
                                     * logits buffer; valid until the next step).
                                     * NULL with logits_ready set = step failed. */
    bool             logits_ready;  /* driver has deposited logits for this step */
    bool             token_ready;   /* next_token is valid for the next step */
    pthread_cond_t   cv;            /* driver → request wakeup (logits_ready /
                                     * ring push / finished / step_error) */
    int              slot_idx;      /* index in sched->slots[] */
    int              gpu_idx;       /* this slot's GPU block row (== slot_idx)
                                     * while bound this turn; -1 when the turn
                                     * bound none */
    int              req_active_min; /* fewest / most decoding slots in the */
    int              req_active_max; /* steps this request ran in (driver,
                                      * under mu; reset at bind) */

    /* ── L3 driver-sampling state (all guarded by sched->mu; allocated by
     * bs_slot_bind_ex, freed by bs_slot_release). ── */
    pn_sampling_params sp;          /* per-request sampling params (copy) */
    uint32_t        *recent;        /* repeat-penalty window (driver-owned copy;
                                     * NULL when repeat_last_n == 0) */
    int              n_recent;      /* live entries in recent */
    int              max_recent;    /* capacity of recent (sp.repeat_last_n) */
    uint32_t        *ring;          /* output ring, sized to the full bind-time
                                     * budget → can never overflow */
    int              ring_cap;      /* == bind-time budget */
    int              ring_r;        /* owner read index (absolute) */
    int              ring_w;        /* driver write index (absolute) */
    int              reason_end_at; /* ring position (1-based) of the first token
                                     * after which the live walk reported forced
                                     * reasoning ended; 0 = not yet.  Driver
                                     * writes (atomic, under mu); the owner reads
                                     * (bs_slot_reasoning_ended).  Reset at bind. */
    int              budget;        /* driver samples remaining (max_tokens-1
                                     * at bind; slot finishes at 0) */
    bool             finished;      /* driver stopped stepping this slot (EOS /
                                     * <|im_end|> / budget) — excluded from the
                                     * active set; owner drains + releases */
    bool             step_error;    /* a decode step failed; owner sees -1 from
                                     * bs_slot_pop_token after draining */

    /* ── L3 AUTO reasoning-budget force-close (guarded by sched->mu; armed at
     * bs_slot_bind_ex, driven per-step by the driver's sampler; mirrors the
     * single-user sg_run_completion cap PER SLOT).  A think-style model is
     * prefilled with "<think>" OPEN, so `reason_in_think` starts true; every
     * token the driver samples while still inside <think> counts toward
     * `reason_budget`.  Once the count reaches the budget without a self-emitted
     * "</think>" (and no tool-call skeleton is live), the driver FORCE-CLOSES:
     * it queues this slot's "</think>" token id(s) in `reason_close[0..n)` and
     * steps them (KV/SSM advances over the close) before resuming normal
     * sampling, so the model concludes and answers.  `reason_budget <= 0` (a
     * model with budget 0, or a non-think model) disables all of this — the
     * slot decodes byte-identically to the pre-fix batched path. ── */
    int              reason_budget;   /* per-turn AUTO reasoning cap, the plan's
                                       * hard point (0 = off) */
    int              reason_soft;     /* from this count on a line end closes
                                       * (pn_reasoning_plan.soft) */
    int              reason_toks;     /* tokens sampled while inside <think> */
    bool             reason_in_think; /* slot is currently inside <think> */
    bool             reason_forced;   /* force-close already fired this turn */
    bool             reason_forced_call; /* a COMPLETE tool call closed after that
                                       * force-close.  Recorded by the driver (it
                                       * already queries the grammar's `complete`
                                       * for the completion-stop) purely so
                                       * bs_slot_release can report the OUTCOME of
                                       * an abnormal turn without dereferencing the
                                       * request-owned grammar hook, which may be
                                       * freed by then. */
    uint32_t         reason_close[8]; /* precomputed "</think>" token id(s) */
    int              reason_close_n;  /* count of close ids (0 = none) */
    int              reason_close_i;  /* next queued close id to inject
                                       * (== reason_close_n → nothing pending) */

    /* ── PC-batched: session affinity + warm-state history (guarded by
     * sched->mu; set by bs_slot_finalize, read by bs_slot_acquire_biased /
     * bs_prefill_resume).  When `sid[0]` is set this slot's pn_state holds
     * session `sid`'s committed history in `hist[0..hist_n)` at pos==hist_n,
     * so the same session routed back here delta-prefills only the appended
     * tokens.  Cleared on any reset (a divergence / different session claiming
     * the slot).  No session id → sid stays empty and the slot behaves
     * byte-identically to the pre-PC batched path. ── */
    char             sid[64];       /* session bound to this warm state ("" = none) */
    char             claim_sid[64]; /* session that claimed this slot for the
                                     * turn now running ("" = none, or a request
                                     * with no session id); set at claim, cleared
                                     * when the slot goes IDLE.  sid is only set
                                     * at resume/finalize, so admission counts an
                                     * in-flight first turn by this (under mu). */
    uint32_t        *hist;          /* committed token history in this pn_state (owned) */
    int              hist_n;        /* committed length (== state->pos) */
    int              hist_cap;      /* allocation of hist */
    double           last_active_ms; /* monotonic ms of this slot's last finalize/
                                      * release; the LRU key when every idle slot
                                      * is warm for another session (under mu) */

    /* ── THINK-ELISION (the append-resume fix).  A reasoning model's assistant
     * turn opens with a forced "<think>" prefill and then generates a think
     * block the client is NEVER shown as text: the serve streams it on the
     * separate `reasoning_content` channel, and streams NOTHING at all when the
     * block is empty — the common case, since at temperature 0 nex-n2-mini
     * deterministically answers the prefill with "\n\n</think>\n\n" and goes
     * straight to the answer.  The client's re-render of that turn therefore
     * cannot reproduce those tokens, exact-prefix append-resume fails at the
     * "<think>" opener, and the WHOLE conversation is re-prefilled every turn
     * (Phase 8).
     *
     * `vis[0..vis_n)` is the CLIENT-VISIBLE PROJECTION of `hist`: the same
     * conversation with every think block this session has ever generated
     * removed — i.e. what the client is expected to resend as the history so
     * far.  When it is a strict prefix of the new context, every KV row in
     * [0, hist_n) is a correct continuation of the conversation the client is
     * describing, so the state resumes AS IS at pos == hist_n and only the new
     * tail is prefilled.  The think blocks stay in the model's context, which
     * is where they were generated and where they cannot be removed from (30 of
     * 40 layers are a sequential SSM recurrence — there is no rewind).
     *
     * A projection, not a span list, because the blocks ACCUMULATE: after two
     * elided resumes the client omits two blocks, and matching around only the
     * most recent one misses on every other turn (measured).  vis_n == 0
     * disables the rung (non-think model, or a turn with no reproducible tail).
     * ── */
    uint32_t        *vis;           /* client-visible projection of hist (owned) */
    int              vis_n;         /* its length (0 = rung disabled) */
    int              vis_cap;       /* allocation of vis */

    /* ── REASONING-BEARING projection (the collapse fix).  Eliding reasoning
     * from the history is what a client does by DEFAULT, and it is also what
     * makes the model stop reasoning: shown a history of assistant turns with
     * no visible deliberation, it copies that shape and closes its own think
     * block at token 0 — 18/18 collapse with reasoning stripped, 0/18 with it
     * present, Fisher p = 2.2e-10.
     * So a client SHOULD resend `reasoning_content`, and then `vis` — which has
     * every think block removed — matches nothing and the whole conversation
     * re-prefills every turn.
     *
     * `vis_think[0..vis_think_n)` is the same projection with this turn's think
     * block PUT BACK, rendered from the reasoning TEXT the serve streamed
     * through the builder's own sg_encode_history_think_block.  The client
     * cannot reproduce the ids the model sampled inside the block, and does not
     * have to: it echoes the string, the serve re-derives the ids.
     *
     * Both projections are kept because the two are the same length only by
     * accident and a client may switch (the client's send_reasoning is a flag).  The
     * head of each is taken from THIS turn's prompt, so a switch costs at most
     * one missed turn.  vis_think_n == 0 disables the rung — a non-think model,
     * an empty think block, or a No-Think turn, all of which `vis` covers. ── */
    uint32_t        *vis_think;     /* reasoning-bearing projection (owned) */
    int              vis_think_n;   /* its length (0 = rung disabled) */
    int              vis_think_cap; /* allocation of vis_think */

    /* The projections for the turn currently in flight, built by
     * bs_slot_note_visible and published into vis / vis_think by
     * bs_slot_finalize.  Owner-only, like `ct`. */
    uint32_t        *pend_vis;
    int              pend_vis_n;
    int              pend_vis_cap;
    uint32_t        *pend_vis_think;
    int              pend_vis_think_n;
    int              pend_vis_think_cap;

    /* ── Committed-log seeding for a turn that resumed via think-elision.  The
     * KV then holds hist[0, hist_n) followed by prompt[ct_prompt_from ..), which
     * is NOT the prompt the caller passed — so bs_slot_bind cannot simply copy
     * the prompt into `ct`.  ct_keep is how many leading tokens of this slot's
     * own `hist` the log starts with; ct_prompt_from is where the prompt joins
     * it.  Both 0 = the ordinary case (log == prompt).  Set by
     * bs_prefill_resume_sys, consumed by the next bs_slot_bind. ── */
    int              ct_keep;
    int              ct_prompt_from;
    /* ── SYS1-batched: per-slot SYSTEM-PREFIX boundary snapshot (owner-only;
     * see bs_prefill_resume_sys / bs_prefill_capture_sys).  `sys_snap` is a
     * self-contained host deep-copy of this slot's pn_state at pos == sys_len
     * (pn_qwen35moe_state_snapshot: SSM conv+recurrent, KV rows [0, sys_len),
     * fp16 sink/window shadows), captured for session `sys_sid` over the exact
     * prefix `sys_tokens[0..sys_len)` (the reuse guard).  Because it is a deep
     * copy off THIS slot's own host mirrors it references no live state: it
     * survives slot resets, other sessions using the slot, and warm-history
     * (`sid`/`hist`) invalidation — restore only ever fires on an exact
     * session-id + prefix byte-match and fully overwrites [0, sys_len).
     * Replaced when a different session (or a changed system prompt) captures
     * on this slot; freed by bs_destroy.  Written only by the slot's owner
     * while it holds the slot; `sys_sid` is read by the affinity scans under
     * sched->mu (IDLE slots only, so no owner write can race a read). ── */
    char             sys_sid[64];   /* session the snapshot belongs to ("" = none) */
    void            *sys_snap;      /* boundary snapshot (NULL = none) */
    int             *sys_snap_refs; /* Phase 10b: shared-snapshot refcount.
                                       NULL = sys_snap privately owned (the
                                       common case); non-NULL = sys_snap is
                                       SHARED between slots whose system
                                       prefixes are byte-identical (each
                                       ~68 MiB host image depends only on
                                       weights + prefix tokens), and *refs
                                       counts the sharers. */
    uint32_t        *sys_tokens;    /* exact prefix tokens covered (owned) */
    int              sys_len;       /* snapshot boundary (tokens covered) */

    /* ── PC-batched committed-token log (guarded by sched->mu).  When caching
     * is armed for this turn (armed_sid[0] set at bs_slot_bind_ex), `ct` holds
     * the EXACT tokens whose K/V+SSM are committed in `state` — i.e. the ids
     * at positions [0, state->pos).  Seeded with the prompt at bind
     * (ct_n == n_prompt == state->pos); the driver appends each token it steps
     * into the state, so ct_n == state->pos is an invariant the driver
     * maintains at the source of truth.  bs_slot_finalize copies ct → hist as
     * the committed history for next-turn delta reuse (no owner-side token
     * reconstruction).  Empty armed_sid → the log is not maintained (no
     * caching this turn). ── */
    char             armed_sid[64]; /* session this turn caches for ("" = off) */
    uint32_t        *ct;            /* committed-token log (owned) */
    int              ct_n;          /* committed length (== state->pos when armed) */
    int              ct_cap;        /* allocation of ct */

    /* ── SESSION-SUBSTITUTION detection (owner-only; set by
     * bs_prefill_resume_sys, read by bs_slot_session_conflict).  True when THIS
     * turn arrived under a session id that this slot already held a record for,
     * but carrying a DIFFERENT system prompt — i.e. the client is running a
     * second, unrelated conversation under one session id.  See
     * bs_slot_session_conflict for why the system block is the discriminator
     * and what it deliberately does NOT flag. ── */
    bool             sid_conflict;

    /* ── Turn-start checkpoint (rung 1c; batch_sched_turn_checkpoint.h).
     * Owner-only while the slot is owned; read by the reroute scan and
     * dropped by a session release only while the slot is IDLE, under mu
     * (the sys_* convention). ── */
    bs_turn_ckpt     ckpt;
    /* What an aborted prefill left in the live state (same bookkeeping as a
     * checkpoint; the state itself is in the slot's block, not a checkpoint
     * buffer). Valid from the abort to the session's next ladder pass. */
    bs_turn_ckpt     partial;
    /* Divergence checkpoint (rung 1d; batch_sched_turn_checkpoint.h): the
     * state at the first token where a re-prefilled request left the
     * session's previous history, kept for the next divergent request (the
     * client's next context compaction). Same bookkeeping and ownership as
     * `ckpt`; card checkpoint index cap + slot. */
    bs_turn_ckpt     divck;
    int              div_cut;       /* owner-only: client position this turn's
                                     * prefill should capture divck at (-1) */

#ifdef USE_CUDA
    /* ── Solo-slot MTP speculation (batch_sched_spec.h; MTP models only).
     * mtp: the request's depth controller + cycle window (re-armed at bind, kept for the
     * scheduler's life so a release mid-cycle cannot free it under the driver).
     * spec_live_pos: state->pos the shared GPU buffers held for this slot after its last
     * cycle (-1 = none); driver-only. ── */
    sg_mtp_cuda      mtp;
    bool             mtp_on;
    int              spec_live_pos;
    /* Stepped by a prefill overlap cycle (bs_pd_cycle) since the driver last
     * stepped it: the MTP head missed those rows (and the prefill's chunk feed
     * took the head over), so the slot neither speculates nor feeds it. The
     * driver clears it on its next step; the head's backlog then restarts as
     * after any other session's prefill. */
    bool             pd_stepped;
#endif
    /* While this slot's request waits for the GPU turn to prefill: about how
     * many tokens it will prefill (0 = not waiting). An overlap prefill
     * holding the turn hands it over to a clearly shorter one (bs_pd_cycle's
     * caller, shortest-prefill-first). Under mu. */
    int              pf_wait_est;
    /* PN_HOST_TIMING: this request's driver-side stage times (driver-only under mu;
     * reset at bind, copied out at release). */
    pn_host_timing   host;
} bs_slot;

/* Maximum number of requests that may wait in the per-model overflow queue
 * (D1).  When the queue is also full, callers receive 429 + Retry-After.
 * Env override BS_QUEUE_DEPTH (integer > 0). */
#define BS_QUEUE_DEPTH_DEFAULT 32

/* D3: memory-pressure levels at/above which new prefills are held in the
 * queue rather than immediately starting a prefill that could OOM.
 * Corresponds to pn_mem_level values (see pn_mem_state.h):
 *   PN_MEM_LEVEL_OK=0  WATCH=1  TIGHT=2  CRITICAL=3
 *
 * Threshold is CRITICAL (3), NOT TIGHT (2), by design:
 *   - TIGHT is the *normal resting state* when a 16GB model is resident on the
 *     8GB host ("reserve nearly reached" is baseline, not an emergency).  It is
 *     already handled where it belongs: the memory ladder degrades at TIGHT
 *     (Q4/shrink/evict, see pn_ladder.h) and pn_admit() fits every request to
 *     live headroom per-prefill.  Holding all intake at TIGHT would stall the
 *     server at its baseline (every request queues → times out → 429).
 *   - CRITICAL means genuinely at the reserve line — only then does D3 hold new
 *     prefills as a last-resort backpressure, and 503 if the queue is also full.
 * The D1 queue still absorbs pure slot-contention bursts independently of this. */
#define BS_PRESSURE_HOLD_LEVEL  3   /* PN_MEM_LEVEL_CRITICAL */

/* Batch scheduler — one per resident model. */
typedef struct pn_batch_sched_s {
    pn_model       *model;          /* borrowed (registry keeps it alive) */
    char            model_name[64]; /* registry name, for LOGGING only — never a
                                     * routing input.  The reasoning force-close
                                     * is an abnormal event and its log line has
                                     * to name the model it happened to. */

    /* ── GPU-ownership turnstile (guarded by mu) — the single-GPU-owner
     * "lock": one prefill chunk (+ the final chunk's bind), or one driver
     * decode step, at a time.  FIFO by ticket: acquire = take
     * gpu_ticket_next++ and wait on cv_gpu until gpu_ticket_serving reaches
     * it; release = gpu_ticket_serving++ + broadcast.  FIFO (not a plain
     * mutex) so the free-running L3 driver cannot barge ahead of a waiting
     * or chunk-yielding prefiller, and prefillers cannot starve the driver:
     * the interleave is provably chunk → step → chunk.  A turn owner never
     * cond-waits; every ticket is followed by a release (chunks and steps
     * are bounded) — deadlock-free by construction. ── */
    uint64_t        gpu_ticket_next;    /* next ticket to hand out */
    uint64_t        gpu_ticket_serving; /* ticket currently owning the GPU */
    pthread_cond_t  cv_gpu;             /* turnstile wakeup (serving advanced) */
    double          gpu_turn_t0_ms;     /* when the current turn owner got the
                                         * turn (bs_now_ms): a yielding
                                         * prefiller's chunk time */
    uint64_t        driver_steps;       /* decode steps run (stats only) */
    int             n_finalize_waiting; /* owners queued for the turn in
                                         * bs_slot_finalize: a finished reply
                                         * held there makes bs_prefill_yield
                                         * hand the turn over between chunks */
    bs_pd_budget    pd_budget;          /* QWEN35_BS_PD_OVERLAP pieces per
                                         * cycle, plain-step EMA (under mu) */

    pthread_mutex_t mu;             /* guards all slot status + token exchanges */
    pthread_cond_t  cv_work;        /* request → driver: gate may have opened
                                     * (bind / token posted / slot released) */
    pthread_cond_t  cv_idle;        /* → callers: a slot went IDLE (space free)
                                     * OR pressure level dropped (D3) */

    bs_slot         slots[BS_MAX_SLOTS];
    int             cap;            /* active capacity (≤ BS_MAX_SLOTS, clamped
                                     * to pn_qwen35moe_gpu_ladder_max_slots at
                                     * create time) */

    pthread_t       driver;         /* background driver thread */
    bool            driver_running;
    bool            shutdown;       /* set to true to stop driver */

    /* Lazy GPU slot init: true once pn_qwen35moe_decode_slot_bind has been
     * called for EVERY lane so that decode_slots can validate
     * g->slot_state[i] for all i.  False at bs_create time (no GPU work at
     * create — avoids the startup hang where an eager pre-bind races an
     * unsettled CUDA context right after a multi-minute weight upload); set
     * true inside the first successful bs_slot_bind.  Written under mu
     * with the GPU turn owned, read by the driver under mu. */
    bool            turn_ckpt_on;   /* turn checkpoints carved (rung 1c) */
    bool            div_ckpt_on;    /* divergence checkpoints carved (rung 1d) */
    bool            slots_gpu_init;

    /* Dynamic-width / continuous-batching state (Phase D2).
     *
     * gpu_map[0..n_active_gpu) lists the active (BOUND/DECODE) slots the
     * next step runs; each decodes in its own fixed GPU row (its slot
     * index), so the list is pure bookkeeping. active_set_changed: set by
     * bs_slot_bind (a user joined), bs_slot_release (a user left) and the
     * driver (a slot finished — L3) so the driver rebuilds the list before
     * the next step. Written and read under mu. */
    int             n_active_gpu;                /* active slots next step */
    int             gpu_map[BS_MAX_SLOTS];       /* active slots (= rows) */
    bool            active_set_changed;          /* mapping needs rebuild */

    /* ── D1: per-model overflow queue ──────────────────────────────────────
     * When all K slots are busy, incoming requests increment n_queued and
     * block on cv_idle (already signaled by bs_slot_release).  Only 429
     * when n_queued == queue_cap (the queue itself is full).  FIFO fairness
     * is guaranteed by the cond broadcast: every waiter re-races for the
     * one free slot; the first to succeed decrements n_queued and continues.
     * This is the standard bounded-queue / cond-broadcast pattern — safe
     * and no starvation in the common burst case because prefills are
     * serialized by the GPU turnstile anyway so throughput is identical to
     * N separate FIFO queues draining one at a time.
     *
     * Guarded by mu. */
    int             n_queued;        /* current waiters (D1 queue depth)    */
    int             queue_cap;       /* max waiters before 429 (D1 limit)   */

    /* ── Session admission (bs_set_session_admission; guarded by mu) ──────
     * The scheduler is the admission authority on the batched path: a slot
     * holds a session's KV, so seating a new session on another's slot would
     * erase it.  At most max_sessions session FAMILIES (session_family.h) hold
     * slots; the next new family is refused (BS_ACQUIRE_AT_CAPACITY), never
     * seated on someone else's slot.  A family whose slots sit idle past
     * session_idle_ttl_ms with no session watch (watch_fn) is reaped, so a
     * dead client cannot keep a seat forever.  Defaults: max_sessions = cap,
     * no reaping, no watch hook. */
    int                   max_sessions;
    uint64_t              session_idle_ttl_ms;   /* 0 = never reap */
    pn_session_watched_fn watch_fn;              /* borrowed; NULL = no watches */
    void                 *watch_ctx;
    bool                  sessions_evictable;    /* gateway mode: a seat is a
                                                  * cache, not a reservation —
                                                  * an idle seat goes to other
                                                  * traffic that needs its slot
                                                  * (bs_set_sessions_evictable) */

    /* ── D3: admission pressure level (snapshot) ──────────────────────────
     * A waiter in the D1 queue also holds off starting its prefill while
     * the host pressure level is >= BS_PRESSURE_HOLD_LEVEL.  The pressure
     * poller thread (or any path that calls pn_memwatch_poll_now) signals
     * cv_idle to wake waiters so they can re-check and proceed once the
     * pressure clears.  Waiters use a timed wait (abs-time deadline) so
     * they never wait longer than bs_queue_wait_ms milliseconds total.
     * Only 503 when BOTH the queue is full AND pressure is CRITICAL —
     * the last-resort hard refusal that keeps the box from OOMing. */
    int             pressure_level;  /* cached pn_mem_level, written under mu */

    /* ── L3: in-driver sampling config (immutable after bs_create) ──────── */
    bool            driver_sample;   /* QWEN35_BS_DRIVER_SAMPLE != "0" (default
                                      * true): driver samples + fills per-slot
                                      * output rings; false = legacy lockstep
                                      * (owner-side sampling via the logits
                                      * handshake) */
    int             n_vocab;         /* model->config.n_vocab (sampling width) */
    int             eos_id;          /* tokenizer EOS id (-1 if unknown) */
    int             im_end_id;       /* <|im_end|> id (-1 if unknown) */
    bool            mtp;             /* the model carries its MTP head (serve --mtp):
                                      * a solo decoding slot speculates
                                      * (batch_sched_spec.h) */

    /* ── Session storage (batch_sched_slots.h; immutable after bs_create) ── */
    int             slot_store;      /* bs_slot_store: CARD_BLOCK or HOST_KV */
    float          *host_logits[BS_MAX_SLOTS]; /* HOST_KV: each row's logits
                                      * from its last step (n_vocab floats);
                                      * NULL for CARD_BLOCK */
} pn_batch_sched;

/* ============================================================
 * Scheduler lifecycle (called from model_registry.c)
 * ============================================================ */

/**
 * Allocate and start a batch scheduler for `model`.  Allocates `cap`
 * pn_state instances (cap ≤ BS_MAX_SLOTS, clamped to the memory-pressure
 * ladder ceiling so lazy GPU init cannot be refused later).  Returns NULL
 * on failure.
 *
 * `model_name` is the registry name, copied for LOGGING only (may be NULL).
 */
pn_batch_sched *bs_create(pn_model *model, int cap, const char *model_name);

/**
 * Stop the driver thread, free all slot states, free the scheduler.
 * Must be called after all slot threads have finished (no in-flight requests).
 */
void bs_destroy(pn_batch_sched *sched);

/* ============================================================
 * Per-request slot API (called from openai_chat / serve_generation)
 * ============================================================ */

/**
 * Reserve a free slot.  Returns the slot index (0..cap-1) if a slot is free,
 * or -1 if all slots are occupied (caller should return 429).
 * Sets slot->status = BS_SLOT_PREFILL; does NOT touch the slot's pn_state
 * (that happens in bs_prefill_begin, under the GPU lock). Non-blocking.
 */
int bs_slot_acquire(pn_batch_sched *sched);

/* Return codes for bs_slot_acquire_or_wait. */
typedef enum {
    BS_ACQUIRE_OK        =  0,   /* slot acquired; *slot_out is the slot index  */
    BS_ACQUIRE_429       = -1,   /* queue full — 429 + Retry-After to client    */
    BS_ACQUIRE_503       = -2,   /* memory CRITICAL + queue full — hard 503     */
    BS_ACQUIRE_TIMEOUT   = -3,   /* waited too long without a slot freeing      */
    BS_ACQUIRE_SHUTDOWN  = -4,   /* scheduler is shutting down                  */
    BS_ACQUIRE_AT_CAPACITY = -5, /* every seat is held by another session
                                  * family: 429 model_at_session_capacity now */
    BS_ACQUIRE_CLIENT_GONE = -6, /* the waiter's client disconnected while it
                                  * queued (bs_slot_acquire_or_wait_watched) */
} bs_acquire_result;

/** A queued waiter's liveness check: non-zero while its client is still
 *  there. Called with sched->mu held, so it must not block or take mu. */
typedef int (*bs_waiter_alive_fn)(void *ud);

/** How often a watched waiter re-checks its client while queued. */
#define BS_WAITER_ALIVE_POLL_MS 250

/**
 * D1 + D3: Reserve a slot, waiting in a bounded FIFO queue if all slots are
 * busy or if host memory pressure is TIGHT/CRITICAL (D3 backpressure gate).
 *
 * Behaviour:
 *  - Session admission first (pn_batch_sched.max_sessions): a request whose
 *    session family already holds a slot always proceeds; a new family is
 *    refused with BS_ACQUIRE_AT_CAPACITY, immediately and without queueing,
 *    when max_sessions families hold slots or when no slot could ever free
 *    for it (every slot is warm for, or running, another family).  A request
 *    with no session id takes only a slot nobody holds, under the same rule.
 *    Idle families with no watch are reaped first (session_idle_ttl_ms).
 *  - If a slot is free AND pressure < BS_PRESSURE_HOLD_LEVEL: acquire
 *    immediately (identical to bs_slot_acquire) and return BS_ACQUIRE_OK.
 *  - Otherwise: enqueue this caller (increment n_queued) and block on
 *    cv_idle (signaled by bs_slot_release and the pressure poller).
 *    - 429 (BS_ACQUIRE_429) immediately if n_queued would exceed queue_cap
 *      AND pressure < CRITICAL.
 *    - 503 (BS_ACQUIRE_503) immediately if n_queued > queue_cap AND
 *      pressure level is CRITICAL — last-resort refusal.
 *    - When a slot frees AND pressure < hold level: claim the slot and
 *      return BS_ACQUIRE_OK.
 *    - If wait_ms elapses without a slot: BS_ACQUIRE_TIMEOUT.
 *    - On scheduler shutdown: BS_ACQUIRE_SHUTDOWN.
 *
 * On BS_ACQUIRE_OK, *slot_out holds the acquired slot index.
 * On any error, *slot_out is -1 and no slot was acquired.
 *
 * `retry_after_s` (if non-NULL): filled with a suggested Retry-After value
 * in seconds on BS_ACQUIRE_429/503/TIMEOUT; 0 on success.
 *
 * wait_ms: max milliseconds to queue-wait (0 → non-blocking, equivalent to
 * bs_slot_acquire with pressure check).  Typical: 30 000 (30 s).
 *
 * `sid` (PC-batched, may be NULL): the request's session.  When a free slot is
 * claimed, THIS session's warm slot is preferred (delta-prefill next turn),
 * then a slot nobody holds, then the least recently used idle slot of its own
 * family.  Another family's slot is never taken.
 *
 * Same-session retry: while a slot is still PREFILLing a request of this same
 * `sid` (a client that hung up mid-prefill and resent before the old request
 * noticed), the request waits for that slot (at most PN_SAME_SESSION_WAIT_MS,
 * default 30000, and never past wait_ms) instead of claiming another one: the
 * old request keeps its prefix on that slot (bs_prefill_abort_keep) and the
 * retry resumes there rather than re-prefilling the conversation elsewhere.
 * PN_SAME_SESSION_WAIT_MS=0 turns the wait off.
 *
 * Composes with D2: a dequeued request acquires a slot set to BS_SLOT_PREFILL,
 * then proceeds through the normal bs_prefill_begin → bs_slot_bind → decode
 * path, joining the batch at the next driver step exactly as a non-queued
 * request would.
 *
 * Thread-safe: yes (guards via sched->mu; does NOT touch the GPU turn).
 */
bs_acquire_result bs_slot_acquire_or_wait(pn_batch_sched *sched,
                                          int wait_ms,
                                          int *slot_out,
                                          int *retry_after_s,
                                          const char *sid);

/**
 * bs_slot_acquire_or_wait for a caller whose client may leave while it queues:
 * `alive(alive_ud)` is checked before each claim attempt and at least every
 * BS_WAITER_ALIVE_POLL_MS while waiting; once it returns 0 the waiter leaves
 * the queue with BS_ACQUIRE_CLIENT_GONE and never takes a slot. NULL `alive`
 * is exactly bs_slot_acquire_or_wait.
 *
 * Evictable seats (bs_set_sessions_evictable): a request that admission would
 * refuse first takes the least recently active idle seat of another family
 * (its warm state is dropped, as the idle reaper does); when every seat is
 * mid-turn it queues for one instead of getting BS_ACQUIRE_AT_CAPACITY.
 *
 * Thread-safe: yes (guards via sched->mu; does NOT touch the GPU turn).
 */
bs_acquire_result bs_slot_acquire_or_wait_watched(pn_batch_sched *sched,
                                                  int wait_ms,
                                                  int *slot_out,
                                                  int *retry_after_s,
                                                  const char *sid,
                                                  bs_waiter_alive_fn alive,
                                                  void *alive_ud);

/**
 * PC-batched — session-affinity acquire.  Like bs_slot_acquire, but when
 * `sid` is non-NULL/non-empty it FIRST tries to reclaim the IDLE slot whose
 * warm pn_state already holds this session's committed history (so its next
 * prefill deltas rather than resets), then the IDLE slot holding this
 * session's SYSTEM-PREFIX snapshot (SYS1-batched: a divergent turn still
 * resumes at the boundary), then a slot no session holds, then the least
 * recently used idle slot of its own family (session_family.h).  It never
 * blocks, never displaces an active slot and never takes a slot another
 * family holds; if the session's warm slot is busy it takes a free one (that
 * turn full-prefills).  `sid == NULL` (or empty) takes only a slot nobody
 * holds, as bs_slot_acquire.  No max_sessions check (bs_slot_acquire_or_wait
 * does that).  Returns the slot index or -1 if none fits.  Non-blocking.
 */
int bs_slot_acquire_biased(pn_batch_sched *sched, const char *sid);

/**
 * Move a just-claimed slot to the IDLE slot whose cache this prompt actually
 * continues.  The acquire runs before tokenization, so with one session id
 * carrying several conversations (opencode: a title request plus the real
 * turns, each with its own system prompt) it can only pick the FIRST slot of
 * that id.  Called once the prompt is known; preference:
 *   1. a slot of `sid` whose warm history (exact or a think projection) is a
 *      strict prefix of `tokens`;
 *   2. a slot holding `sid`'s boundary snapshot over tokens[0..sys_len);
 *   3. nothing reusable anywhere: when slot `idx` holds ANOTHER conversation
 *      of `sid`, a cold IDLE slot instead, so the new conversation does not
 *      evict it.
 * Ties keep `idx`.  On a move the new slot is claimed (PREFILL) and `idx`
 * returns to IDLE with its warm state intact.  Call only between the acquire
 * and bs_prefill_resume_sys (`idx` is PREFILL, no GPU turn held).  Empty
 * `sid` is a no-op.  Returns the slot the caller now owns.
 */
int bs_slot_reroute(pn_batch_sched *sched, int idx, const char *sid,
                    const uint32_t *tokens, int n_ctx, int sys_len);

/**
 * PC-batched — resume-aware prefill begin.  Like bs_prefill_begin (takes the
 * GPU turn), but if this slot's warm pn_state still belongs to session `sid`
 * and its committed history is an exact prefix of `tokens[0..n_ctx)` with
 * state->pos == that history length, it RESUMES: does NOT reset, and sets
 * *delta_start to the history length so the caller prefills only the appended
 * tokens.  Otherwise it resets + forgets exactly like bs_prefill_begin and
 * sets *delta_start = 0 (full prefill).  `sid == NULL`/empty always resets
 * (*delta_start = 0) — byte-identical to bs_prefill_begin.  Same GPU-turn
 * pairing contract as bs_prefill_begin (→ bind / abort).
 */
void bs_prefill_resume(pn_batch_sched *sched, int idx, const char *sid,
                       const uint32_t *tokens, int n_ctx, int *delta_start);

/**
 * SYS1-batched — system-prefix-aware resume (the batched twin of
 * pn_prefix_cache_acquire_sys).  As bs_prefill_resume, plus the caller's
 * system-prompt token boundary `sys_len` (tokens[0..sys_len) are the
 * per-session-stable system prefix; 0 disables SYS1 for the call).  Resume
 * preference, mirroring the single-user ladder:
 *
 *   1. Warm-history append resume (exact bs_prefill_resume behavior):
 *      *delta_start = this slot's committed history length.
 *   2. System-prefix resume: this slot holds a boundary snapshot for THIS
 *      session AND tokens[0..sys_len) byte-match it → the snapshot is restored
 *      into the slot's pn_state (host-side; GPU shadows forgotten so the delta
 *      prefill re-seeds from the restored mirror) and *delta_start = sys_len.
 *      Fires on the divergent-history turns (reasoning models: <think> text
 *      committed, never resent) where (1) never can.
 *   3. Fresh/diverged with no usable snapshot: full reset, *delta_start = 0.
 *
 * *capture_sys is set to 1 when the caller should CAPTURE the boundary this
 * turn: a full prefill is about to run (case 3), a system prefix exists
 * (0 < sys_len < n_ctx), a session id was supplied, and no valid snapshot on
 * this slot covers these exact system tokens.  The caller then prefills
 * [0, sys_len), calls bs_prefill_capture_sys (still owning the GPU turn), and
 * prefills the rest.
 *
 * Correctness: any mismatch (session, prefix bytes, dims) falls to case 3 —
 * never a partial/wrong reuse.  sys_len == 0 or an empty sid is byte-identical
 * to bs_prefill_resume.  Same GPU-turn pairing contract as bs_prefill_begin
 * (→ bind / abort).
 */
void bs_prefill_resume_sys(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *tokens, int n_ctx, int sys_len,
                           int *delta_start, int *capture_sys);

/** Does this scheduler serve `think_history: "drop_last"` (turn checkpoints
 *  carved and enabled)? false for NULL. Lock-free read of load-time state. */
bool bs_think_history_drop_last(const pn_batch_sched *sched);

/** What bs_prefill_resume_think decided. */
typedef struct {
    int            delta_start;   /* as bs_prefill_resume_sys */
    int            capture_sys;   /* as bs_prefill_resume_sys */
    pn_resume_kind kind;          /* the rung that ran (usage `resume`) */
} bs_resume_outcome;

/**
 * bs_prefill_resume_sys honouring the request's `think_history`
 * (core/think_history.h) and reporting the rung that ran. KEEP is
 * bs_prefill_resume_sys exactly.
 *
 *   AS_SENT    rungs 1, 1b, the aborted-prefill rung, 1c and 1d are skipped:
 *              the request resumes at the system prefix (rung 2) or cold, so
 *              the KV holds the request as rendered and nothing else. History,
 *              projections and checkpoint are rebuilt from this turn.
 *   DROP_LAST  the client dropped the last turn's think block: rungs 1, 1b
 *              and the aborted-prefill rung are skipped and rung 1c resumes
 *              when the request starts with the checkpoint's client tokens and
 *              the checkpoint's KV is exactly those tokens (no think block an
 *              earlier projection kept). Its history becomes the request as
 *              rendered; the turn saves its own checkpoint as usual. No match:
 *              as AS_SENT.
 *
 * Same GPU-turn pairing contract as bs_prefill_begin (→ bind / abort).
 */
void bs_prefill_resume_think(pn_batch_sched *sched, int idx, const char *sid,
                             const uint32_t *tokens, int n_ctx, int sys_len,
                             pn_think_history think_history,
                             bs_resume_outcome *out);

/**
 * Did the turn just resolved by bs_prefill_resume_sys arrive under a session id
 * that already identified a DIFFERENT conversation on this slot?
 *
 * A session id is meant to name ONE conversation.  When a client reuses one id
 * for two conversations (the measured case: an agent's reflect pass POSTing its
 * own system prompt under the main chat's id) the serve cannot resume either of
 * them: each turn evicts the other's warm prefix and re-prefills in full, so the
 * session silently goes quadratic.  Nothing about that is visible to the client
 * today, which is why it hid for weeks.
 *
 * THE DISCRIMINATOR IS THE SYSTEM BLOCK, and it is chosen because it is the one
 * thing every LEGITIMATE history change keeps byte-identical:
 *   - append          — the system block is the unchanged head of the prompt;
 *   - think-elision   — the client's re-render differs only after the header;
 *   - compaction      — the client rewrites the log body only, and asserts the wire
 *                       prefix ahead of it is byte-identical (its own strcmp
 *                       test, tests/test_context_summary.c).
 * So "same session id, different system block" is substitution, and a plain
 * divergence of the conversation body is NOT flagged.
 *
 * Deliberate limits (false negatives only — this never flags a legitimate turn):
 *   - a request with NO system message (sys_len == 0) carries no signal;
 *   - a session whose slot was taken by someone else has no record left to
 *     compare against, so its next substitution is missed;
 *   - a client that legitimately varies its system prompt per turn (injected
 *     timestamps, a changing tool list) is reported every turn.  That is why
 *     this is a REPORT and not a refusal.
 *
 * Valid from bs_prefill_resume_sys until the slot's next resume/reset.
 * Thread-safe: no (owner-only, like the rest of the resume state).
 */
bool bs_slot_session_conflict(const pn_batch_sched *sched, int idx);

/**
 * SYS1-batched — snapshot slot `idx`'s state at its system-prompt boundary
 * for session `sid`.  Call mid-prefill, OWNING the GPU turn (between
 * bs_prefill_resume_sys and bs_slot_bind, never inside a yield window), when
 * the slot's state sits EXACTLY at pos == sys_len having just prefilled
 * tokens[0..sys_len) (the *capture_sys == 1 protocol).  On the GPU path the
 * live GPU state is synced to the slot's host mirror first (the same D2H the
 * L2 yield protocol performs), so the snapshot is self-contained and the
 * prefill continues unaffected.  Replaces any previous snapshot on this slot
 * (possibly another session's — per-slot capacity is the documented
 * conservative bound; session affinity routes a session back to its slot, so
 * replacement is rare in practice).
 *
 * @return 0 on success, -1 on failure (pos mismatch / unsupported arch /
 *         allocation / D2H).  Failure is benign: the caller just continues
 *         its full prefill uncached.
 *
 * Thread-safe: no (owner-only, GPU turn held)
 */
int bs_prefill_capture_sys(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *tokens, int sys_len);

/**
 * Record what the client is expected to resend for this turn — BOTH ways.
 *
 * Call once per turn from the decode loop, after generation ends and BEFORE
 * bs_slot_finalize.  Builds the slot's two projections:
 *
 *   vis        = prompt[0, n_prompt - think_prefill_len) ++ rerender
 *   vis_think  = prompt[0, n_prompt - think_prefill_len) ++ think_block
 *                                                        ++ rerender
 *
 * The shared head is the history the client rendered ITSELF this turn (the
 * prompt minus the forced generation prefill), so whichever convention the
 * client follows for earlier turns is already baked in and the projections stay
 * correct as think blocks accumulate in the KV.  The two differ only in this
 * turn's reply: `vis` is the client that DROPS reasoning, `vis_think` the
 * client that sends it back.  Recording both means neither client pays a
 * re-prefill, and a client that switches mid-session pays at most one turn.
 *
 * @param prompt            the request's full prompt token array
 * @param n_prompt          its length
 * @param think_prefill_len prompt tokens the chat template forced after the
 *                          assistant header ("<think>", "<think>\n", or the
 *                          closed No-Think block) — see
 *                          sg_build_prompt_tokens().  0 for a non-think model,
 *                          whose whole prompt is then the head.
 * @param think_block       this turn's think block as the builder will render
 *                          it from the `reasoning_content` the client echoes
 *                          (sg_encode_history_think_block).  NULL / 0 leaves
 *                          vis_think empty, disabling the reasoning-bearing
 *                          rung for this turn — which is right for a No-Think
 *                          turn, an empty block, or a non-think model.
 * @param think_block_n     how many tokens.
 * @param rerender          this turn's reply AS THE CLIENT WILL RESEND IT, put
 *                          back through the serve's own render + tokenize path
 *                          (sg_project_client_rerender).  NULL / 0 disables
 *                          BOTH rungs for this turn.
 * @param rerender_n        how many tokens.
 *
 * The projection is what the client will SEND, not what the model produced.
 * Those differ more than they look: a model's sampled ids are not a function of
 * the text it produced — "<tool_call>" has both a special id and a four-token
 * byte encoding and the char-level tool grammar accepts either — so a projection
 * built from the GENERATED ids misses on whichever turns sampling picked the
 * other one, which is what made every turn re-prefill its whole conversation.
 * Building it from the re-render
 * removes the choice.  A turn whose re-render still does not match is safe to
 * record anyway — the resume path memcmps the whole projection and simply
 * misses.
 *
 * Thread-safe: no (owner-only, between decode and finalize)
 */
void bs_slot_note_visible(pn_batch_sched *sched, int idx,
                          const uint32_t *prompt, int n_prompt,
                          int think_prefill_len,
                          const uint32_t *think_block, int think_block_n,
                          const uint32_t *rerender, int rerender_n);

/**
 * Publish this turn's pending projections as the slot's live ones, against the
 * committed history `hist_n` they map onto.  Called by bs_slot_finalize once
 * the history is published; exposed so the CPU test drives the REAL publish
 * rule rather than a stand-in that is free to drift from it.
 *
 * The two candidates are NOT bounded the same way, and that asymmetry is the
 * rule: the think-elided projection must not exceed `hist_n` (it strictly drops
 * tokens, so a longer one is nonsense), while the reasoning-bearing one may —
 * The client's rendering carries the template's own think separators where the
 * KV carries what the model sampled, which lands the two lengths within a token
 * or two of each other.  Bounding it would disable the rung on a coin flip; it
 * is not a correctness gate either way (see bs_projection_resume).
 *
 * @param sl      the slot (caller holds sched->mu)
 * @param hist_n  the committed history length the projections map onto
 *
 * Thread-safe: no (call under sched->mu)
 */
void bs_slot_publish_projections(bs_slot *sl, int hist_n);

/**
 * PC-batched — commit a session's warm state to its slot.  Called by the
 * owner AFTER decode completes and BEFORE bs_slot_release, while the slot's
 * GPU decode state (gpu_idx) is still valid.  Takes the GPU turn and checks
 * that the slot's block on the card holds the committed state (KV + SSM stay
 * there between turns; eager mode instead captures them to the host mirror),
 * then publishes the driver-maintained committed-token log (ct → hist) + armed_sid
 * as this slot's warm history for next-turn delta reuse.  Releases the GPU
 * turn before returning.  Does NOT release the slot — the owner still calls
 * bs_slot_release.  No-op if the slot was never armed for caching (no session
 * id at bind).  Returns 0 on success, -1 on failure (the slot's warm state is
 * invalidated so a future affinity hit re-full-prefills rather than reusing
 * stale state).
 */
int bs_slot_finalize(pn_batch_sched *sched, int idx);

/* ============================================================
 * Session warm-tier EXPORT / RESTORE / RELEASE — batched arm
 *
 * The batched twin of prefix_cache.h's export/import.  A batched session's
 * authoritative state is NOT in the model's pn_prefix_cache (that cache holds
 * only admission bookkeeping on this path — nothing ever calls its acquire, so
 * it has no entries and no pn_states): it is the warm `bs_slot` its last turn
 * finalized on — `sl->sid` + `sl->hist[0..hist_n)` over `sl->state` at
 * pos == hist_n, with the KV rows themselves living in the GPU decode-slot
 * ring (Phase 10b / INV3: the host mirror is NOT maintained).
 *
 * Export therefore materializes [0, pos) from the ring through
 * pn_qwen35moe_gpu_state_to_host (which calls the INV3 materializer and
 * refuses loudly rather than reading a stale mirror), snapshots the now-
 * self-contained host image, and leaves the session untouched.  Restore is
 * the exact inverse of the SYS1-batched boundary restore that the warm path
 * already validated on a K80 host: host write + GPU forget + content-generation bump,
 * so the next delta prefill re-seeds from the restored mirror.
 *
 * Both reserve the target slot (IDLE → PREFILL) under `mu` so no request can
 * claim it mid-operation, take the GPU turn for the D2H/H2D bookkeeping, and
 * return the slot to IDLE.  A slot that is mid-turn is never exported from —
 * the single-user path refuses a `pos != n_tok` state for the same reason.
 * ============================================================ */

/* Result codes shared by bs_session_export / bs_session_restore. */
typedef enum {
    BS_SESSION_OK        =  0,  /* done                                      */
    BS_SESSION_NOT_FOUND = -1,  /* no warm slot holds this session (export)  */
    BS_SESSION_BUSY      = -2,  /* the session's slot is mid-turn, or (restore)
                                 * no slot can seat it → 429                 */
    BS_SESSION_ERROR     = -3,  /* materialize / snapshot / restore failed   */
} bs_session_result;

/**
 * Export session `sid`'s warm slot state as an opaque qwen35moe snapshot the
 * caller serializes to a portable blob (same snapshot type as
 * pn_prefix_cache_export_snapshot, so ONE serializer serves both paths).
 *
 * Materializes the slot's KV rows [0, pos) from the GPU decode-slot ring via
 * the INV3 materializer — no host mirror is maintained or resurrected; the
 * host bytes are transient (they live in the slot's pn_state mirror until its
 * next reset) and bounded by the session's real context, exactly the
 * occasional-consumer contract INV3 defines.  Read-only w.r.t. the session:
 * the slot's warm history, sid and position are unchanged.
 *
 * @param n_tok_out   Out: committed token count the snapshot covers.
 * @param tokens_out  Out: malloc'd COPY of the committed history (caller frees).
 * @param snap_out    Out: snapshot (free with pn_qwen35moe_state_snapshot_free).
 * @return BS_SESSION_OK / _NOT_FOUND / _BUSY / _ERROR.
 *
 * Thread-safe: yes (takes mu + the GPU turn internally).
 */
bs_session_result bs_session_export(pn_batch_sched *sched, const char *sid,
                                    void **snap_out, uint32_t **tokens_out,
                                    int *n_tok_out);

/**
 * Restore a deserialized snapshot into a slot as session `sid`'s warm state,
 * so the NEXT chat turn on `sid` routes to that slot (session affinity) and
 * delta-prefills only what the client appends beyond `tokens[0..n_tok)`.
 *
 * Seated as a chat turn is (session admission, then try_claim_idle's order):
 * the slot already holding `sid` (a re-restore overwrites it), else a slot no
 * session holds, else one of its own family's; never another family's.
 * BS_SESSION_BUSY when no such slot is idle or the session is refused at
 * max_sessions.  The caller must have validated the blob's weights_version
 * first.  `snap`'s pos MUST equal n_tok.
 *
 * Thread-safe: yes (takes mu + the GPU turn internally).
 */
bs_session_result bs_session_restore(pn_batch_sched *sched, const char *sid,
                                     const void *snap, const uint32_t *tokens,
                                     int n_tok);

/**
 * Drop session `sid`'s warm state from whatever IDLE slot holds it: clears the
 * session binding + committed history and releases the slot's system-prefix
 * boundary snapshot, so the slot full-prefills for whoever takes it next and
 * no departed session's context lingers.  The slot's pn_state buffers are left
 * alone — bs_prefill_begin resets + forgets them under the GPU turn on the next
 * acquire, which is the only safe place to do it.  A slot mid-turn is skipped
 * (its owner still needs it).  Idempotent; no-op for an unknown session.
 *
 * Thread-safe: yes (takes mu).
 */
void bs_session_release(pn_batch_sched *sched, const char *sid);

/**
 * Set this scheduler's session admission: at most `max_sessions` session
 * families hold slots (<= 0 keeps the default, the slot count), and a family
 * idle for `idle_ttl_ms` with no session watch loses its slots (0 = never).
 * Call once after bs_create, before serving.
 *
 * Thread-safe: yes (takes mu).
 */
void bs_set_session_admission(pn_batch_sched *sched, int max_sessions,
                              uint64_t idle_ttl_ms);

/**
 * Make this scheduler's session seats evictable (gateway mode): an idle seat
 * is taken from its family whenever a request needs the slot, so a session key
 * can never lock out other traffic. Off (the default) keeps home behaviour: a
 * seat is held until released, reaped or its TTL passes.
 *
 * Thread-safe: yes (takes mu).
 */
void bs_set_sessions_evictable(pn_batch_sched *sched, bool evictable);

/**
 * Install the watch-liveness predicate the idle reaper asks before reaping a
 * family (the same hook as pn_prefix_cache_set_watch_hook; called WITHOUT mu
 * held).  NULL clears it.  `fn`/`ctx` are borrowed and must outlive the hook.
 *
 * Thread-safe: yes (takes mu).
 */
void bs_set_watch_hook(pn_batch_sched *sched, pn_session_watched_fn fn,
                       void *ctx);

/**
 * Session families holding slots now, warm or mid-turn (what max_sessions
 * counts and what sessions_used reports).  0 for a NULL scheduler.
 *
 * Thread-safe: yes (takes mu).
 */
int bs_session_count(pn_batch_sched *sched);

/** Session occupancy of one scheduler (bs_session_occupancy). */
typedef struct {
    int  families;      /* session families holding slots (sessions_used) */
    int  max_sessions;  /* the family limit admission enforces */
    int  slots;         /* decode slots */
    int  slots_held;    /* slots some session holds (warm or mid-turn) */
    int  slots_free;    /* slots nobody holds */
    bool seat_free;     /* could a NEW family be seated now */
} bs_occupancy;

/**
 * Fill `out` with one consistent snapshot of who holds the slots.  All zero
 * for a NULL scheduler.
 *
 * Thread-safe: yes (takes mu).
 */
void bs_session_occupancy(pn_batch_sched *sched, bs_occupancy *out);

/**
 * Could a NEW session family be seated now (fewer than max_sessions families
 * hold slots and some slot is held by nobody)?  What replica routing asks to
 * steer a new session away from a full replica.  false for NULL.
 *
 * Thread-safe: yes (takes mu).
 */
bool bs_session_seat_free(pn_batch_sched *sched);

/**
 * Does `sid`'s session family hold a slot here (warm or mid-turn)?  false for
 * NULL / an empty sid.  What replica routing asks before keeping a session on
 * the replica it is bound to.
 *
 * Thread-safe: yes (takes mu).
 */
bool bs_session_family_holds(pn_batch_sched *sched, const char *sid);

/**
 * Prefix-match complement of bs_session_release: drop the warm state of every
 * IDLE slot whose session id begins with `base_prefix`.  A dead liveness watch
 * covers a tab's BASE id while the live slots are the mode-suffixed
 * `<base>-orch` / `-act` / `-chat` sessions, so it frees them all at once.
 *
 * Thread-safe: yes.
 */
void bs_session_release_by_prefix(pn_batch_sched *sched,
                                  const char *base_prefix);

/**
 * Take exclusive ownership of the model's GPU (a FIFO turnstile ticket) for
 * this slot's prefill and reset the slot's pn_state.  Blocks until any
 * in-flight decode step or other prefill chunk finishes (finite; FIFO).
 * MUST be paired with exactly one of bs_slot_bind / bs_slot_bind_ex (normal
 * path) or bs_prefill_abort (any prefill failure) — both release the GPU
 * turn.  bs_prefill_yield may release + re-acquire it in between (L2).
 */
void bs_prefill_begin(pn_batch_sched *sched, int idx);

/**
 * L2 — chunk-boundary yield: give the driver a decode step in the middle of
 * a long prefill.  If there is a step to run (gate open), or a finished slot
 * is queued in bs_slot_finalize (its reply waits on it), saves the prefill
 * so far into the slot's own block on the card (eager mode: the host
 * mirror), so any interleaved forward can safely claim the live GPU buffers
 * and the next chunk re-seeds from the block; releases the GPU turn, and
 * re-acquires it FIFO behind the driver (and any earlier waiter).  No-op when
 * there is nothing to interleave, or if the save fails (keeping the turn is
 * always safe).  Call ONLY between bs_prefill_begin and bs_slot_bind/abort, never
 * after the final chunk (the bind must directly follow the last prefill
 * token under the same turn).
 */
void bs_prefill_yield(pn_batch_sched *sched, int idx);

/**
 * QWEN35_BS_PD_OVERLAP: is the stage-alternating overlap on for this
 * scheduler (switch on, card-resident batched path, driver sampling)?
 */
bool bs_pd_overlap_on(const pn_batch_sched *sched);

/**
 * One overlap cycle, run by a prefiller that holds the GPU turn: the decode
 * step the driver would have run next (same snapshot, same deposit and
 * sampling), with `cursor`'s prefill pieces in each board's idle half.
 * Returns 1 when no slot is decoding (nothing ran; advance the prefill alone),
 * 0 after a cycle, -1 if the step failed (its slots are errored, as the
 * driver does). The stepped slots stop MTP speculation for this request.
 */
int bs_pd_cycle(pn_batch_sched *sched, void *cursor);

/**
 * Hand the GPU turn over at a quiet point of an overlap prefill (the cursor
 * quiesced: nothing in flight, state->pos at what cleared): saves the prefill
 * into the slot's block as bs_prefill_yield does, releases the turn and
 * re-acquires it FIFO, behind every earlier waiter (a queued finalize, the
 * driver, a waiting prefill). No stand-aside window: decode already ran
 * beside the prefill.
 */
void bs_pd_yield(pn_batch_sched *sched, int idx);

/**
 * Shortest-prefill-first: is a request waiting for the turn whose prefill is
 * at least BS_PD_SRPT_MARGIN tokens shorter than `remaining`?
 */
bool bs_pd_shorter_waiting(pn_batch_sched *sched, int remaining);

/** Owners queued in bs_slot_finalize for the GPU turn (racy read is fine). */
int bs_pd_finalize_waiting(pn_batch_sched *sched);

/**
 * Abort a prefill started with bs_prefill_begin: releases the GPU turn and
 * frees the slot (→ IDLE).  Do NOT also call bs_slot_release.
 */
void bs_prefill_abort(pn_batch_sched *sched, int idx);

/**
 * bs_prefill_abort for a prefill that was cut off PART-WAY THROUGH (client
 * disconnect, or a chunk that failed): the slot's pn_state now holds KV/SSM
 * for a prompt prefix that no committed history describes.  Resets the state
 * and drops the slot's warm session binding under the still-held GPU turn,
 * then releases turn + slot exactly like bs_prefill_abort.
 *
 * The resume ladder already refuses a half-written state (rungs 1/1b require
 * state->pos == hist_n), so this is not what makes the next turn correct — it
 * makes it CHEAP and quiet: without it the slot keeps advertising the session
 * id it can no longer serve, so try_claim_idle steers that session's next turn
 * straight back onto the poisoned slot to discover the miss, logging an
 * append-resume miss and evicting nothing useful.  Mirror of the single-user
 * path's pn_prefix_cache_reset_session on abort.
 */
void bs_prefill_abort_reset(pn_batch_sched *sched, int idx);

/**
 * bs_prefill_abort_reset for a CLIENT abort, keeping what can resume: the
 * prompt prefix the prefill already computed (bs_partial_prefill_keep) and the
 * slot's turn-start checkpoint (rung 1c). The slot keeps its session, drops
 * only the warm history the half-written state no longer matches, and the
 * session's next turn resumes from the kept prefix or the checkpoint. With
 * neither it is bs_prefill_abort_reset. Releases turn + slot either way.
 *
 * @param sid       the request's session id
 * @param prompt    the request's prompt tokens [0, n_prompt)
 * @param sys_len   its system-prefix boundary
 */
void bs_prefill_abort_keep(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *prompt, int n_prompt, int sys_len);

/**
 * Finish the prefill critical section: snapshot this slot's live SSM state
 * and top up its prefilled KV rows into its own block (GPU row `idx`, device
 * to device; eager mode also refreshes the host mirrors), mark the slot BOUND
 * and wake the driver.  `first_token` is the
 * first decode token (sampled from prefill logits by the caller).
 *
 * Must be called owning the GPU turn (from bs_prefill_begin); ALWAYS
 * releases it before returning.  Returns 0 on success; -1 on failure, in
 * which case the slot has already been released (do not call
 * bs_slot_release or bs_prefill_abort).
 */
int bs_slot_bind(pn_batch_sched *sched, int idx, uint32_t first_token);

/**
 * L3 variant of bs_slot_bind for driver-sampling mode (bs_driver_sampling()
 * true).  Same GPU-turn contract and failure semantics as bs_slot_bind,
 * plus it arms the driver's per-slot sampler:
 *   - `sp`        the request's sampling params (copied);
 *   - `recent`/`n_recent` the repeat-penalty window as of the FIRST decode
 *     token — the prompt tail; the driver appends each stepped token
 *     (starting with `first_token`) before sampling, exactly mirroring the
 *     legacy owner loop, so streams are bit-identical at temp 0;
 *   - `max_new`   the driver's sampling budget (max_tokens - 1; the owner
 *     already holds token #1 = `first_token`).  max_new <= 0 binds the slot
 *     already `finished` (nothing to step — max_tokens == 1).
 * The owner then consumes tokens via bs_slot_pop_token.
 *
 * PC-batched: when `sid` is non-NULL/non-empty the slot is ARMED for caching —
 * its committed-token log is seeded with `prompt[0..n_prompt)` (== state->pos
 * after prefill) and the driver appends every token it steps, so at
 * bs_slot_finalize the log is the exact committed history at state->pos.
 * `sid == NULL`/empty leaves caching off for this turn (byte-identical path).
 */
int bs_slot_bind_ex(pn_batch_sched *sched, int idx, uint32_t first_token,
                    const pn_sampling_params *sp,
                    const uint32_t *recent, int n_recent, int max_new,
                    const char *sid, const uint32_t *prompt, int n_prompt);

/**
 * Arm the AUTO reasoning-budget force-close for a slot the driver is sampling
 * (L3).  Mirrors sg_run_completion's single-user cap PER SLOT: the driver
 * counts tokens sampled while the slot is inside <think> and, once the count
 * reaches the plan's hard point, or its soft point on a token that ends a line
 * (reasoning_budget.h), without a self-emitted "</think>", force-decodes the
 * "</think>" close (`close_ids[0..n_close)`) so the model concludes and answers.
 *
 * Call AFTER bs_slot_bind_ex and BEFORE the owner drains any token — the slot
 * is BOUND (the driver rebuilds its GPU map before the first step), so this
 * write races nothing.  `first_token` is the owner-held token #1; it is counted
 * here iff the slot starts inside <think> (`in_think`), so the driver's per-step
 * count stays aligned with the single-user loop (which counts every token
 * emitted while em.in_think, first token included).
 *
 * @param plan           the turn's reasoning plan (NULL or hard <= 0 → no-op)
 * @param in_think       true if the assistant turn was prefilled with <think>
 *                       still OPEN (a think-style model)
 * @param close_ids      the model's "</think>" token id(s) (from the caller's
 *                       tokenizer encode); copied into the slot
 * @param n_close        count of close ids (<= 8; 0 → force-close disabled)
 * @param first_token    the owner-held first decode token (counted iff in_think)
 *
 * Thread-safe: yes (takes sched->mu).  No-op when driver-sampling is off.
 */
void bs_slot_arm_reasoning_budget(pn_batch_sched *sched, int idx,
                                  const pn_reasoning_plan *plan, bool in_think,
                                  const uint32_t *close_ids, int n_close,
                                  uint32_t first_token);

/**
 * L3 — pop the next driver-sampled token for this slot (blocks while the
 * ring is empty and the driver is still stepping).  Returns:
 *   1  = *tok holds the next token (EOS/<|im_end|> ARE delivered — the
 *        owner performs the finish check, mirroring the legacy loop);
 *   0  = clean end of stream (budget exhausted; ring fully drained);
 *  -1  = a decode step failed or the scheduler is shutting down (after the
 *        ring is drained) — the caller must still bs_slot_release.
 */
int bs_slot_pop_token(pn_batch_sched *sched, int idx, uint32_t *tok);

/**
 * L3 — has forced-mode reasoning ended at the owner's read position?  True once
 * the owner has popped the token after which the driver's live grammar walk
 * first reported it (pn_tool_grammar_hook.reasoning_ended).  The owner flips its
 * emitter out of <think> on this, not on the live walk, which runs ahead of the
 * owner by the ring backlog (an MTP deposit, a descheduled owner): reading the
 * walk flipped the channel early and filed the "</think>" close as content.
 * Owner thread only (it reads its own read index).  PN_THINK_FLIP_RED=1 drops
 * the position check (the old read-ahead behaviour, for the host test's red arm).
 */
bool bs_slot_reasoning_ended(pn_batch_sched *sched, int idx);

/* Driver, under mu: the slot's walk has just been advanced by the token at ring
 * position ring_w; record it in reason_end_at if forced reasoning first ended there. */
void bs_slot_note_reasoning_end(bs_slot *sl);

/* Deposit one driver token into the slot's ring as a decode step does.  Under
 * sched->mu.  CUDA builds: driver_emit_token itself (ring push, budget,
 * reasoning-end note, finish checks; 1 = the slot finished).  Host builds have no
 * driver: the ring push, budget and note only.  -1 when the ring is full.
 * Exposed for host tests. */
int bs_slot_deposit_token(pn_batch_sched *sched, bs_slot *sl, uint32_t next);

/* True when this scheduler runs L3 in-driver sampling (default; disable
 * with QWEN35_BS_DRIVER_SAMPLE=0). */
bool bs_driver_sampling(const pn_batch_sched *sched);

/**
 * Wait for the driver to complete the FIRST decode step after bs_slot_bind.
 * No token is deposited here — the first token was passed to bs_slot_bind.
 *
 * On return, `*logits_out` points to this slot's logits from the first decode
 * step (valid until the next call to bs_slot_step_wait or bs_slot_release).
 *
 * Returns 0 on success, -1 on error (driver failed or shutdown) — the caller
 * must still bs_slot_release.
 */
int bs_slot_await_first(pn_batch_sched *sched, int idx, float **logits_out);

/**
 * Per-step decode exchange from the request-thread side (steps 2+).
 *
 * Deposits `next_token` (sampled from previous step's logits) for the driver.
 * Then waits for the driver to complete the next decode step and deposit new
 * logits.
 *
 * On return, `*logits_out` points to this slot's logits (valid until the
 * next call to bs_slot_step_wait or bs_slot_release).
 *
 * Returns 0 on success, -1 if the scheduler is shutting down or the step
 * failed — the caller must still bs_slot_release.
 */
int bs_slot_step_wait(pn_batch_sched *sched, int idx,
                      uint32_t next_token, float **logits_out);

/**
 * Free this slot (EOS / stop / length limit / error).  Must be called
 * exactly once by the slot's owner after bs_slot_bind succeeded (on the
 * pre-bind paths use bs_prefill_abort / plain release for a slot still in
 * PREFILL that never took the GPU lock).  Marks the slot IDLE for reuse.
 */
void bs_slot_release(pn_batch_sched *sched, int idx);

/** What the scheduler saw of one request, for its [req] line
 *  (request_timing_line.h). */
typedef struct {
    int  active_min;   /* fewest / most decoding slots in the steps that */
    int  active_max;   /*   stepped this request; 0/0 = it never stepped */
    long mtp_cycles;   /* MTP cycles it ran; 0 = MTP did not run */
    pn_host_timing host; /* PN_HOST_TIMING driver stages (zero when off) */
} bs_request_stats;

/** bs_slot_release that also returns the finishing request's stats, read under
 *  the same lock (out may be NULL; zeroed when the slot was already IDLE). */
void bs_slot_release_stats(pn_batch_sched *sched, int idx, bs_request_stats *out);

/* Returns true when QWEN35_BATCH_SLOTS=1 (env check, cached). */
bool bs_enabled(void);

/**
 * D3: Notify the scheduler that the host memory pressure level has changed.
 * Wakes any waiters in the D1 queue so they can re-check whether they should
 * proceed (pressure dropped) or keep waiting (still under pressure).  Callers:
 * the serve layer or any path that polls pn_memwatch after a level change.
 *
 * `level` is the new pn_mem_level value (0=OK … 3=CRITICAL).
 * Thread-safe: yes.
 */
void bs_notify_pressure(pn_batch_sched *sched, int level);

/* ============================================================
 * Dispatch layer (Phase G0): live load snapshot for self-registration
 * ============================================================ */

/* Read-only load snapshot reported to the dispatch registry on each heartbeat.
 * `capacity` is the served concurrency (batch-slot count); `active_slots` the
 * currently-busy slots (not IDLE); `queue_depth` the D1 overflow waiters. */
typedef struct {
    int capacity;
    int active_slots;
    int queue_depth;
    int free_slots;
} bs_load;

/**
 * Snapshot the scheduler's current load under `sched->mu` (a pure read — no
 * bookkeeping added). When `sched == NULL` (the single-user, non-batched
 * QWEN35_BATCH_SLOTS-off path) this reports capacity 1 / free 1: that path
 * 429s a second concurrent chat, so it is treated as a single always-available
 * slot for dispatch purposes.  Thread-safe.
 */
void bs_get_load(pn_batch_sched *sched, bs_load *out);

#endif /* PN_SERVE_BATCH_SCHED_H */
