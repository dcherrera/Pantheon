/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched.c
 * @brief Per-model batch decode scheduler (C3).
 *        See batch_sched.h for the full design + concurrency model.
 *
 * Locking recap:
 *   GPU turnstile (gpu_ticket_next/serving + cv_gpu, under mu) — one GPU
 *            owner at a time, FIFO: a prefill chunk (+ the final chunk's
 *            bind), or one driver decode step.  Restores the
 *            one-in-flight-forward guarantee the per-model gen_lock used to
 *            give (the C2 async phase machine, the shared live SSM buffers /
 *            d->host_state, the shared GPU KV ring and sc->r_logits all
 *            assume it).  FIFO tickets — not a mutex — so the free-running
 *            L3 driver and chunk-yielding prefillers alternate fairly
 *            (chunk → step → chunk); a turn owner never cond-waits and every
 *            ticket is followed by a release, so no deadlock is possible.
 *   mu     — slot statuses, the token/logits/ring handshakes, the turnstile
 *            counters.  The only mutex; all condvars wait on it.
 *
 * @status Complete.  The resume ladder (exact-prefix append, think-elision,
 *         system-prefix snapshot, full reset) plus the PN_REQPROF divergence
 *         dumps that make a miss readable straight out of the serve log, and
 *         the session-substitution report (bs_slot_session_conflict), and
 *         per-conversation routing under one session id (bs_slot_reroute).
 * @issues A warm resume is only valid while the session's KV rows are still in
 *         a GPU ring (the host mirror is lazy — Phase 10b/INV3), so
 *         bs_warm_kv_reusable has to prove it before every warm rung.  How many
 *         sessions can hold a WARM rung at once is therefore a property of the
 *         decode-slot ring, and it is NOT one — it is however many are
 *         CONCURRENTLY ACTIVE.  rebuild_gpu_map compacts the active slots into
 *         GPU ring rows [0, n_active), so:
 *           - sessions whose turns OVERLAP occupy distinct rows and all keep
 *             their warm KV (measured on a K80 host 2026-08-02, saber-orch, two live
 *             sessions, both resuming at ~1.05 s TTFT);
 *           - sessions that run STRICTLY SEQUENTIALLY each compact alone to row
 *             0 and therefore overwrite each other's rows — every turn, so both
 *             fall to the system-prefix rung and their TTFT goes linear in
 *             conversation depth again (measured 2.2 -> 3.4 -> 4.3 s).
 *         The system-prefix rung is genuinely multi-session (its snapshots are
 *         per-slot host deep copies), so N sessions stay warm AT THAT TIER
 *         regardless of interleaving.
 * @todo None
 */

#include "batch_sched.h"

#include "request_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "../core/engine.h"          /* gpt2_bytes_decode (diagnostic dumps) */
#include "../core/mem/pn_mem_state.h"
#include "../core/mem/pn_memwatch.h"
#include "tokenizer/tokenizer.h"

/* The qwen35moe header is needed on ALL builds now: the SYS1-batched
 * system-prefix snapshot/restore primitives (pn_qwen35moe_state_snapshot &
 * co.) are pure host code, unit-tested on the Mac.  The GPU slot API it also
 * declares stays behind USE_CUDA at the call sites: on non-CUDA builds the
 * scheduler is disabled (bs_enabled() false, bs_create NULL, no driver
 * thread), so those calls never compile in. */
#include "../models/qwen35moe.h"

/* Phase 10b: forward decl — defined with the SYS1 snapshot helpers below. */
static void bs_sys_snap_unref(bs_slot *sl);

/* ============================================================
 * Feature gate
 * ============================================================ */

bool bs_enabled(void)
{
#ifndef USE_CUDA
    return false;   /* batch decode requires the CUDA backend */
#else
    /* DEFAULT ON. This is the path the fleet deploys, and the only one that
     * commits the CLIENT RE-RENDER of a turn (sg_project_client_rerender) rather
     * than the raw generated ids. The single-user path commits what the model
     * emitted, so a tool call GENERATED as ordinary text ("<","tool","_call",">")
     * but RE-RENDERED as the special <tool_call> id diverges from the committed
     * prefix — and with no lcp-rewind on this arch that collapses the resume
     * point to the system boundary on EVERY turn.
     *
     * Measured on the K80 test machine 2026-08-28, same probe both ways: single-user
     * collapsed 9 of 9 turns, wasting 1 -> 839 tokens and growing; batched
     * collapsed only turn 1 (no history yet) and appended the other 5. Defaulting
     * this OFF meant every measurement taken through it described a path we do
     * not deploy. QWEN35_BATCH_SLOTS=0 still forces the old behaviour. */
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("QWEN35_BATCH_SLOTS");
        cached = (env && env[0] == '0') ? 0 : 1;
        fprintf(stderr, "[batch_sched] multi-user batched decode %s%s\n",
                cached ? "enabled" : "DISABLED",
                env ? " (QWEN35_BATCH_SLOTS)" : " (default)");
    }
    return (bool)cached;
#endif
}

/* ============================================================
 * GPU-ownership turnstile (see batch_sched.h) — FIFO tickets under mu.
 *
 * acquire: take a ticket, wait until served.  The wait is UNCONDITIONAL on
 * shutdown by design: a taken ticket must always be served-then-released or
 * the turnstile would skip/starve later tickets.  This is safe — every turn
 * owner (a prefill chunk, a bind, a decode step) completes in bounded time
 * and releases, so the turn always arrives; shutdown is checked by callers
 * after they own (and release) the turn.
 * ============================================================ */

static void bs_gpu_acquire_locked(pn_batch_sched *sched)
{
    uint64_t my = sched->gpu_ticket_next++;
    while (sched->gpu_ticket_serving != my)
        pthread_cond_wait(&sched->cv_gpu, &sched->mu);
}

static void bs_gpu_release_locked(pn_batch_sched *sched)
{
    sched->gpu_ticket_serving++;
    pthread_cond_broadcast(&sched->cv_gpu);
}

static void bs_gpu_acquire(pn_batch_sched *sched)
{
    pthread_mutex_lock(&sched->mu);
    bs_gpu_acquire_locked(sched);
    pthread_mutex_unlock(&sched->mu);
}

static void bs_gpu_release(pn_batch_sched *sched)
{
    pthread_mutex_lock(&sched->mu);
    bs_gpu_release_locked(sched);
    pthread_mutex_unlock(&sched->mu);
}

/* ============================================================
 * Driver thread — the heart of C3
 * ============================================================ */

#ifdef USE_CUDA

/* Per-step driver timing to stderr (M6 instrumentation), QWEN35_BS_STATS=1. */
static int bs_stats_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_BS_STATS");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static double bs_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/*
 * Driver design — "dynamic-width step" (Phase D2):
 *
 * The driver compacts the active (BOUND/DECODE) lanes into a contiguous
 * GPU index range [0, n_active) so `pn_qwen35moe_decode_slots` runs over
 * only the live users.  Idle lanes pay zero GPU compute.
 *
 * gpu_map[gpu_idx] = scheduler slot index.  The mapping is built/rebuilt
 * whenever active_set_changed is set (a user joined or left); rebuild_gpu_map
 * realizes the (arbitrary) permutation that compacts the active lanes to
 * [0, n_active), using the reserved scratch slot at index `cap` to break
 * cycles (the engine is bound with cap+1 slots).  Steady state (no
 * join/leave) has zero overhead.
 *
 * Correctness:
 *   - BOUND/DECODE lanes: carry their real token; BOUND → DECODE on first
 *     step.
 *   - PREFILL lanes: excluded from the compacted set (never stepped); the
 *     prefill's own GPU work is serialized by the turnstile.
 *   - IDLE lanes: excluded from the compacted set; never passed to the GPU.
 *   - A slot released mid-step has left DECODE before the driver reads mu
 *     again; its GPU slot's logits are discarded at deposit time.
 *   - A newly-BOUND slot (continuous batching join) is included in the
 *     NEXT step after its bind completes (bind sets active_set_changed and
 *     signals cv_work).  The bind itself owns the GPU turn so no
 *     interleaving with a step is possible.
 *   - A slot that turns IDLE during the step (released by its owner before
 *     the driver re-acquires mu for deposit): the deposit loop checks
 *     status == BS_SLOT_DECODE before writing logits, so it is skipped
 *     harmlessly — the slot's next owner will find it IDLE and its state
 *     reset on the next bs_prefill_begin.
 *
 * The step runs owning the GPU turn (no prefill can overlap) with mu
 * dropped while the GPU is busy (request threads keep posting/releasing).
 */

/* Gate: may the driver run a step right now?  Under sched->mu.
 * Returns the number of active (BOUND/DECODE) slots if a step is allowed,
 * 0 if blocked.  The caller uses this count as n_active for the compacted
 * step so it does not need to scan again. */
static int gate_ok(pn_batch_sched *sched)
{
    int n_active = 0;
    if (!sched->slots_gpu_init)
        return 0;                      /* no lane registered with the GPU yet */
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        switch (sl->status) {
        case BS_SLOT_PREFILL:
            /* PREFILL no longer closes the gate (pre-L2 it did): since D2
             * the lane is never in the compacted gpu_map (never stepped),
             * and GPU exclusivity against the prefill's own forwards is the
             * turnstile's job — the prefiller owns (or is FIFO-queued for)
             * the GPU turn for each chunk.  Blocking here would defeat the
             * L2 chunk-yield windows. */
            break;
        case BS_SLOT_DECODE:
            if (sl->finished)
                break;                /* driver stopped it (EOS / budget);
                                       * owner is draining — not stepped */
            if (!sl->token_ready)
                return 0;             /* lockstep (legacy mode): wait for the
                                       * owner.  In driver-sampling mode the
                                       * driver re-arms this itself, so it
                                       * never blocks the gate. */
            n_active++;
            break;
        case BS_SLOT_BOUND:
            if (sl->finished)
                break;                /* bound with a zero budget
                                       * (max_tokens == 1): nothing to step */
            n_active++;               /* first step: no token needed yet */
            break;
        default:
            break;                    /* IDLE: excluded from active set */
        }
    }
    return n_active;                  /* 0 ↔ nothing to step */
}

/* Rebuild the GPU-index → scheduler-slot compacted mapping (GPU turn owned,
 * mu held).  Called when active_set_changed is set (a user joined, left, or
 * finished).
 *
 * Compacts all active (BOUND/DECODE) scheduler slots to GPU indices [0, n_active).
 * The new GPU-index order mirrors the scheduler slot order (stable, lowest first).
 *
 * For each active slot:
 *   - Its CURRENT GPU index = the old gpu_map entry (for DECODE slots),
 *     or the scheduler slot index (for newly-BOUND slots, whose bind call
 *     in bs_slot_bind set g->slot_state[sched_slot_idx]).
 *   - Its TARGET GPU index = its rank in the compacted list (0, 1, 2, ...).
 *   - If current == target: no move needed (slot_state already correct).
 *   - If current != target: call pn_qwen35moe_decode_slot_move(current, target,
 *     cap) to D2D copy the SSM state from conv_slots[current] to conv_slots[target]
 *     and update g->slot_state[target] — without touching the stale host mirror.
 *
 * Moves realize an ARBITRARY permutation (both downward compaction and the
 * upward moves / cycles that concurrent multi-join produces): see the body
 * for the greedy place-into-free + cycle-break algorithm.  A move that is
 * genuinely upward is not a corruption — the old "src >= dst only" refusal
 * was the K>=5 empty-output bug.
 *
 * Returns n_active (≥ 1; the gate already ensured this), or -1 on failure.
 * Precondition: GPU turn owned, mu held.
 *
 * A slot the driver marked `finished` (EOS / budget, L3) is inactive here —
 * it drops out of the map exactly like a released slot. */
static int slot_is_active(const bs_slot *sl)
{
    return (sl->status == BS_SLOT_BOUND || sl->status == BS_SLOT_DECODE) &&
           !sl->finished;
}

static int rebuild_gpu_map(pn_batch_sched *sched)
{
    /* Build the target compacted list: all active sched slots in order. */
    int target_sched[BS_MAX_SLOTS];   /* target_sched[gpu_rank] = sched_slot */
    int n_active = 0;
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (slot_is_active(sl))
            target_sched[n_active++] = i;
    }

    /* Realize the (arbitrary) permutation that maps each active slot from its
     * CURRENT GPU index (sl->gpu_idx — where its live decode state resides,
     * set at bind to a free GPU index and maintained by every move below) to
     * its TARGET GPU index (its rank in the compacted list).
     *
     * The old loop assumed every move packs DOWN (src >= dst) and refused any
     * upward move.  That invariant does NOT hold: >=5 joiners binding
     * concurrently at the highest free GPU indices (7,6,5,4,3) before the
     * driver rebuilds get compacted to ranks (0,1,2,3,4), so the slot at
     * gpu 3 must move UP to rank 4 (a gpu 3<->4 swap) — the refusal returned
     * -1 and decode_slots then produced nothing for every user (the K>=5
     * "all users emit empty content" bug).  A correct compaction must handle
     * upward moves and cycles.
     *
     * occupied[p] = the active sched slot currently living at GPU index p, or
     * -1 if free.  Realize the permutation greedily: repeatedly move any slot
     * whose target rank is currently free into place (that frees its old
     * index, unblocking others); when no such slot exists the remaining
     * unplaced slots form a pure cycle, so break it by parking one slot at a
     * free GPU index and continue.  Every move D2D-copies the SSM state and
     * re-seeds the KV ring from the (current) host mirror, so a park + final
     * move is correct.
     *
     * A free GPU index is ALWAYS available for the cycle break: users occupy
     * only [0, cap), and the engine is bound with cap+1 slots, so index `cap`
     * is a permanent scratch slot the decode step never touches (it runs over
     * [0, n_active) with n_active <= cap).  This is essential at n_active ==
     * cap: 8 joiners binding concurrently at the highest free indices produce
     * a full reversal (gpu 7..0 -> ranks 0..7 = four 2-cycles) with no free
     * index in [0, cap) — the scratch at `cap` breaks it.  Placed slots
     * (gpu_idx == rank) are never disturbed again. */
    const int scratch = sched->cap;       /* reserved cycle-break slot */
    int occupied[BS_MAX_SLOTS];
    for (int p = 0; p <= scratch; p++)
        occupied[p] = -1;
    for (int rank = 0; rank < n_active; rank++) {
        int i = target_sched[rank];
        int g = sched->slots[i].gpu_idx;
        if (g < 0 || g >= sched->cap) {
            fprintf(stderr, "[batch_sched] rebuild_gpu_map: sched slot %d has "
                    "bad GPU index %d\n", i, g);
            sched->active_set_changed = false;
            return -1;
        }
        occupied[g] = i;
    }

    for (;;) {
        int made_progress = 0;
        int all_placed    = 1;
        for (int rank = 0; rank < n_active; rank++) {
            int i   = target_sched[rank];
            int src = sched->slots[i].gpu_idx;
            if (src == rank)
                continue;                 /* already at its target */
            all_placed = 0;
            if (occupied[rank] >= 0)
                continue;                 /* target still held; try later */
            if (pn_qwen35moe_decode_slot_move(sched->model, src, rank,
                                              sched->cap + 1) != 0) {
                fprintf(stderr, "[batch_sched] rebuild_gpu_map: slot_move "
                        "failed (sched=%d gpu %d→%d cap=%d)\n", i, src, rank,
                        sched->cap);
                sched->active_set_changed = false;
                return -1;
            }
            occupied[src]           = -1;
            occupied[rank]          = i;
            sched->slots[i].gpu_idx = rank;
            made_progress           = 1;
        }
        if (all_placed)
            break;
        if (!made_progress) {
            /* Pure cycle: every unplaced slot's target is held by another
             * unplaced slot.  Park the first unplaced slot at a free GPU
             * index (the reserved scratch at `cap` is always free) to break
             * the cycle. */
            int tmp = -1;
            for (int p = 0; p <= scratch; p++)
                if (occupied[p] < 0) { tmp = p; break; }
            if (tmp < 0) {
                fprintf(stderr, "[batch_sched] rebuild_gpu_map: permutation "
                        "cycle with no free GPU slot (n_active=%d cap=%d)\n",
                        n_active, sched->cap);
                sched->active_set_changed = false;
                return -1;
            }
            for (int rank = 0; rank < n_active; rank++) {
                int i   = target_sched[rank];
                int src = sched->slots[i].gpu_idx;
                if (src == rank)
                    continue;
                if (pn_qwen35moe_decode_slot_move(sched->model, src, tmp,
                                                  sched->cap + 1) != 0) {
                    fprintf(stderr, "[batch_sched] rebuild_gpu_map: park move "
                            "failed (sched=%d gpu %d→%d cap=%d)\n", i, src, tmp,
                            sched->cap);
                    sched->active_set_changed = false;
                    return -1;
                }
                occupied[src]           = -1;
                occupied[tmp]           = i;
                sched->slots[i].gpu_idx = tmp;
                break;
            }
        }
    }

    /* Commit the new map: gpu rank -> sched slot (each slot's gpu_idx is now
     * exactly its rank, set by the moves above). */
    for (int rank = 0; rank < n_active; rank++) {
        sched->gpu_map[rank] = target_sched[rank];
        sched->slots[target_sched[rank]].gpu_idx = rank;
    }
    sched->n_active_gpu       = n_active;
    sched->active_set_changed = false;
    return n_active;
}

/* Deposit one slot's step failure and wake its owner (under mu).  Serves
 * both modes: legacy owners see logits==NULL + logits_ready; L3 owners see
 * step_error after draining the ring. */
static void driver_error_slot(pn_batch_sched *sched, bs_slot *sl)
{
    sl->logits       = NULL;
    sl->logits_ready = true;
    sl->step_error   = true;
    sl->finished     = true;
    sched->active_set_changed = true;
    pthread_cond_signal(&sl->cv);
}

/* Advance this slot's tool-grammar (if any) over a FORCE-INJECTED token id —
 * the driver-side twin of the single-user injection's sg_tg_step.  A normally
 * SAMPLED token is advanced inside pn_sample_next_token; an injected "</think>"
 * close is not sampled, so its text must be fed to the grammar here so the FSM
 * walk stays aligned (and a live forced walk sees reasoning end).  No-op when
 * the slot carries no grammar hook. */
static void driver_grammar_advance_injected(bs_slot *sl, uint32_t tok)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;
    if (tgh && tgh->advance && tgh->st && tgh->token_text) {
        const char *txt = tgh->token_text(tgh->user, tok);
        if (txt) tgh->advance(tgh->st, txt);
    }
    /* An injected token is not sampled, so the logprob sink never sees it — and
     * the consumer pairs records with tokens BY POSITION, so a missing record
     * would shift every later token's logprobs onto the wrong text.  Record the
     * documented NaN placeholder to keep the two sequences aligned (see the
     * injected-token contract on pn_logprob_sink).  No-op when the request did
     * not ask for logprobs. */
    const pn_logprob_sink *lp = sl->sp.logprobs;
    if (lp && lp->emit)
        lp->emit(lp->user, tok, NAN, NULL, NULL, 0);
}

/* ============================================================
 * Full-logits arming window, driver side
 *
 * The design, the invariant it maintains and why arming one token early or one
 * token late are both wrong are documented once, in serve_generation.c above
 * sg_grammar_arm_window_on().  Read that first.  This is the batched half.
 *
 * The batched path is the STRAIGHTFORWARD half, and it is worth saying why: the
 * driver samples a slot, the sampler advances that slot's grammar over the token
 * it drew, the reasoning force-close injects and advances in the same breath —
 * and only after all of that does the driver leave the deposit loop and issue
 * the next decode_slots.  One thread, one critical section, so arming at the end
 * of the deposit is provably before the read and provably after every mutation.
 * There is no hole here and no latch is needed.
 * ============================================================ */

/* PN_GRAMMAR_ARM_WINDOW=1 — the driver maintains the window (default OFF: the
 * owner armed the row for the whole turn and the driver must not touch it). */
static int bs_grammar_arm_window_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ARM_WINDOW");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* PN_GRAMMAR_ARM_STALE=1 — FALSIFIER (see serve_generation.c).  Arms from the
 * walk state as it stood BEFORE this step's advance, reproducing the one-token
 * -late ordering, so a gate can watch pn_sampler_grammar_sparse_rows go
 * non-zero and know its detector is not vacuous.  Never set in production. */
static int bs_grammar_arm_stale_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ARM_STALE");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/**
 * Arm this slot's full-vocabulary logits row for the next decode step.
 *
 * Call at the END of the slot's deposit, after the sampler's advance and after
 * any injected-token advance, and before the driver issues the next step.
 *
 * @param sched         owning scheduler (for the model's state discriminator)
 * @param sl            the slot just sampled
 * @param active_before the walk's mask-active value read BEFORE this step's
 *                      advance; used only by the PN_GRAMMAR_ARM_STALE falsifier
 *
 * Thread-safe: no — driver thread only, under sched->mu
 */
static void driver_arm_full_logits(pn_batch_sched *sched, bs_slot *sl,
                                   int active_before)
{
    if (!bs_grammar_arm_window_on()) return;
    if (!sl->state || !pn_qwen35moe_owns_state(sched->model)) return;
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;
    int need = ((sl->sp.logit_bias && sl->sp.n_logit_bias > 0) ||
                sl->sp.logprobs != NULL) ? 1 : 0;
    if (!need && tgh && tgh->st)
        need = bs_grammar_arm_stale_on()
                 ? active_before
                 : (tgh->refresh ? tgh->refresh(tgh->st) : 1);
    pn_qwen35moe_state_set_full_logits(sl->state, need);
}

/* AUTO reasoning-budget force-close, PER SLOT, in the driver's sampler — the
 * batched twin of sg_run_completion's cap.  Returns the token this step must
 * push to the ring + step into the state:
 *
 *   - FORCE-CLOSE injection pending (reason_close_i < reason_close_n): return
 *     the next queued "</think>" id (do NOT sample; the caller already skips
 *     sampling when *inject_out is set), advance the grammar over it, and when
 *     the queue drains leave <think> and stamp the grammar reasoning-end.
 *   - otherwise: `sampled` is the token pn_sample_next_token just produced.
 *     Count it toward the budget iff the slot is inside <think>; latch the trip
 *     off the PRE-count think state (mirrors 8680b89's fix so a self-close on
 *     the trip token can't starve the cap), and if the budget is reached while
 *     still open — with no live tool-call skeleton — queue the close for the
 *     next step(s).  If `sampled` is itself the "</think>" close, mark the block
 *     self-closed instead (no double-close).
 *
 * *inject_out is set true when the returned token is an injected close (the
 * caller then must NOT count it against the grammar again — it was fed here).
 * budget<=0 / not-think slots never enter here (guarded by the caller). */
static uint32_t driver_reason_step(bs_slot *sl, uint32_t sampled,
                                   const pn_tool_grammar_hook *tgh,
                                   bool *inject_out)
{
    *inject_out = false;
    /* Drain a pending force-close injection first. */
    if (sl->reason_close_i < sl->reason_close_n) {
        uint32_t close_tok = sl->reason_close[sl->reason_close_i++];
        driver_grammar_advance_injected(sl, close_tok);
        *inject_out = true;
        if (sl->reason_close_i >= sl->reason_close_n) {
            /* Whole close emitted — reasoning region is over.  The grammar's own
             * advance over "</think>" (driver_grammar_advance_injected above)
             * self-detects reasoning-end, so a live forced walk would compel the
             * tool-call opener on content from the very next char. */
            sl->reason_in_think = false;
            /* …and that is exactly the frame-capture defect: THIS close was
             * INJECTED by the budget, not chosen by the model, so the model is
             * still mid-sentence.  Tell the grammar so it grants a bounded
             * landing window instead of forcing a frame around the unfinished
             * thought.  Runs AFTER the advance above, whose self-detection has
             * already set reasoning_ended — this adds the window on top.  NULL
             * hook (or a dialect without a window) → pre-fix behavior. */
            if (tgh && tgh->reasoning_end_injected && tgh->st)
                tgh->reasoning_end_injected(tgh->st);
        }
        return close_tok;
    }

    if (!sl->reason_in_think)
        return sampled;

    /* A token sampled while inside <think> counts against the budget.  Latch
     * the trip HERE, off the pre-count in-think state, so a budget reached on
     * the very token that also self-closes still fires the cap (the 8680b89
     * starve bug on the single-user path).  A tool-call skeleton being emitted
     * must NOT be force-closed mid-call — the same guard sg_tg_in_call gives the
     * single-user path. */
    sl->reason_toks++;
    bool in_call = (tgh && tgh->in_call && tgh->st && tgh->in_call(tgh->st));
    bool self_closed = (sl->reason_close_n == 1 &&
                        sampled == sl->reason_close[0]);
    if (self_closed) {
        sl->reason_in_think = false;   /* model concluded on its own */
        return sampled;
    }
    if (sl->reason_budget > 0 && !sl->reason_forced &&
        sl->reason_toks >= sl->reason_budget && !in_call) {
        sl->reason_forced = true;
        sl->reason_close_i = 0;   /* arm the injection for the next step(s) */
    }
    return sampled;
}

static void *driver_thread(void *arg)
{
    pn_batch_sched *sched = arg;

    /* Compacted active-set arrays — sized to sched->cap. */
    uint32_t  tokens[BS_MAX_SLOTS];
    pn_state *states[BS_MAX_SLOTS];
    float    *lo[BS_MAX_SLOTS];
    /* Maps compacted GPU index → scheduler slot index (snapshot copy). */
    int       gpu_map_snap[BS_MAX_SLOTS];

    const int stats = bs_stats_on();
    uint64_t  step_no = 0;
    double    t_prev_done = 0.0;      /* end of the previous deposit (M6) */

    pthread_mutex_lock(&sched->mu);
    for (;;) {
        /* ── Wait for the gate (predicate re-checked under mu). ── */
        while (!sched->shutdown && !gate_ok(sched))
            pthread_cond_wait(&sched->cv_work, &sched->mu);
        if (sched->shutdown)
            break;
        double t_gate = stats ? bs_now_ms() : 0.0;

        /* ── Take the GPU turn — FIFO behind any prefiller already queued
         *    (initial join or L2 chunk re-acquire), so the driver can never
         *    barge a joiner out and a joiner can never be starved by the
         *    free-running L3 step loop.  cv_gpu waits release mu, so owners
         *    keep posting/releasing meanwhile. ── */
        bs_gpu_acquire_locked(sched);

        /* Re-check the gate: it may have closed while we waited for the
         * turn (a release, an error).  Back off by releasing the turn. */
        int n_active = gate_ok(sched);
        if (sched->shutdown || !n_active) {
            bs_gpu_release_locked(sched);
            if (sched->shutdown)
                break;
            continue;
        }

        /* ── Rebuild the compacted GPU map if the active set changed (a
         *    user joined, left, or finished since the last step).  GPU turn
         *    owned + mu held.  On failure, error out all active slots. ── */
        if (sched->active_set_changed) {
            int rn = rebuild_gpu_map(sched);
            if (rn < 0) {
                /* Hard GPU failure: error every active slot (BOUND included
                 * — its owner is parked in await_first/pop_token) so owners
                 * unblock and release; a later joiner triggers a fresh
                 * rebuild attempt. */
                for (int i = 0; i < sched->cap; i++) {
                    bs_slot *sl = &sched->slots[i];
                    if (sl->status == BS_SLOT_DECODE ||
                        sl->status == BS_SLOT_BOUND)
                        driver_error_slot(sched, sl);
                }
                bs_gpu_release_locked(sched);
                continue;
            }
            n_active = rn;
        }

        /* ── Snapshot the compacted step (mu held, turn owned).
         *    Build tokens[]/states[] over GPU indices [0, n_active). ── */
        int snap_n = 0;
        for (int g = 0; g < sched->n_active_gpu; g++) {
            int i = sched->gpu_map[g];
            if (i < 0 || i >= sched->cap)
                continue;             /* stale map entry (should not happen) */
            bs_slot *sl = &sched->slots[i];
            states[snap_n] = sl->state;
            if (sl->status == BS_SLOT_BOUND && !sl->finished) {
                sl->status      = BS_SLOT_DECODE;   /* first step */
                sl->token_ready = false;
                tokens[snap_n] = sl->next_token;
            } else if (sl->status == BS_SLOT_DECODE && !sl->finished) {
                sl->token_ready = false;
                tokens[snap_n] = sl->next_token;
            } else {
                /* Slot went IDLE/finished between gate_ok and the snapshot
                 * (owner released it); skip by not advancing snap_n. */
                continue;
            }
            gpu_map_snap[snap_n] = i;
            snap_n++;
        }
        /* snap_n may be < n_active if a release raced us; that's fine. */
        if (snap_n == 0) {
            /* Nothing left to step (all active slots were released). */
            bs_gpu_release_locked(sched);
            continue;
        }

        /* ── GPU step: turn owned (no prefill can overlap), mu dropped
         *    (owners keep draining rings / releasing meanwhile). ── */
        pthread_mutex_unlock(&sched->mu);
        double t_launch = stats ? bs_now_ms() : 0.0;
        memset(lo, 0, sizeof(lo));
        int rc = pn_qwen35moe_decode_slots(sched->model, tokens, states,
                                           snap_n, lo);
        /* Phase 0: split the step into the GPU forward and the host-side
         * per-slot deposit+sampling that follows it, so `samp_ms` can be read
         * against K directly (the per-slot sampler is sequential, so it is
         * expected to scale with snap_n). Stamped before re-taking mu so the
         * lock wait is not charged to either. */
        double t_dec = stats ? bs_now_ms() : 0.0;
        pthread_mutex_lock(&sched->mu);

        if (rc != 0)
            fprintf(stderr, "[batch_sched] decode_slots failed (n=%d); "
                    "erroring out active slots\n", snap_n);

        /* ── Deposit and wake each owner.  Use gpu_map_snap to reverse the
         *    GPU-index → sched-slot mapping.  A slot released mid-step is
         *    no longer DECODE and is skipped.
         *
         *    L3 (driver_sample): append the just-stepped token to the
         *    slot's repeat window, sample the next token from its
         *    (host-side, sparse top-k) logits — sequentially per slot, the
         *    documented deterministic order — push it to the output ring
         *    and re-arm token_ready, so the next step needs NOTHING from
         *    the owners.  EOS/<|im_end|>/budget ends the slot's stepping
         *    (finished; the owner drains and releases).
         *
         *    Legacy: deposit the logits pointer; the owner samples. ── */
        for (int g = 0; g < snap_n; g++) {
            int i = gpu_map_snap[g];
            bs_slot *sl = &sched->slots[i];
            if (sl->status != BS_SLOT_DECODE)
                continue;
            if (rc != 0) {
                driver_error_slot(sched, sl);
                continue;
            }
            /* PC-batched: tokens[g] was just stepped INTO sl->state (pos
             * advanced by 1).  If this slot is armed for caching, append it to
             * the committed-token log so ct_n == state->pos stays invariant.
             * ct is pre-sized to prompt + budget at bind, so this never
             * overflows; on the impossible overflow, disarm (finalize will
             * then no-op → this session re-full-prefills next turn). */
            if (sl->armed_sid[0]) {
                if (sl->ct && sl->ct_n < sl->ct_cap)
                    sl->ct[sl->ct_n++] = tokens[g];
                else
                    sl->armed_sid[0] = '\0';
            }
            if (!sched->driver_sample) {
                sl->logits       = lo[g];
                sl->logits_ready = true;
                pthread_cond_signal(&sl->cv);
                continue;
            }
            /* L3 in-driver sampling. */
            if (sl->max_recent > 0 && sl->recent) {
                if (sl->n_recent < sl->max_recent) {
                    sl->recent[sl->n_recent++] = tokens[g];
                } else {
                    memmove(sl->recent, sl->recent + 1,
                            (size_t)(sl->max_recent - 1) * sizeof(uint32_t));
                    sl->recent[sl->max_recent - 1] = tokens[g];
                }
            }
            const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;
            /* AUTO reasoning-budget force-close (per slot).  When a "</think>"
             * close is queued (the budget tripped on a prior step) inject it
             * WITHOUT sampling — advancing the grammar over it — so the KV/SSM
             * steps over the close exactly as if the model had emitted it.
             * Otherwise sample normally, then account the token toward the
             * budget (arming the injection when it trips).  Slots with no budget
             * are byte-identical: driver_reason_step returns `sampled` untouched
             * (reason_in_think stays false). */
            bool injected = false;
            uint32_t next;
            /* PN_GRAMMAR_ARM_STALE falsifier only: the walk's mask-active value
             * BEFORE this step's advance — the stale reading the window exists
             * to avoid.  Not read on the shipping path. */
            int active_before = (bs_grammar_arm_stale_on() && tgh &&
                                 tgh->refresh && tgh->st)
                                    ? tgh->refresh(tgh->st) : 0;
            if (sl->reason_close_i < sl->reason_close_n) {
                next = driver_reason_step(sl, 0, tgh, &injected);
            } else {
                next = pn_sample_next_token(lo[g], sched->n_vocab,
                                            &sl->sp, sl->recent,
                                            sl->n_recent);
                next = driver_reason_step(sl, next, tgh, &injected);
            }
            /* Every mutation this step can make to the grammar walk has now
             * happened (the sampler advanced over a drawn token; the force-close
             * advanced over an injected one).  Arm the next step's row from the
             * walk as it now stands — the driver issues that step itself, still
             * holding mu, so this write provably precedes the read. */
            driver_arm_full_logits(sched, sl, active_before);
            if (!sl->ring || sl->ring_w >= sl->ring_cap) {
                /* Cannot happen (ring_cap == bind budget); defensive. */
                driver_error_slot(sched, sl);
                continue;
            }
            sl->ring[sl->ring_w++] = next;
            sl->budget--;
            /* Grammar completion-stop (batched): the sampler (or the injection
             * above) just advanced this slot's grammar by `next`; if that closed
             * a whole tool call, finish the slot NOW so a forced call terminates
             * right after "</function>"/"</tool_call>" instead of running to
             * budget (the batched twin of the single-user break-on-complete).
             * No-op when the slot carries no grammar hook. */
            bool grammar_done = (tgh && tgh->complete && tgh->st &&
                                 tgh->complete(tgh->st));
            /* Record the OUTCOME of a force-closed turn while the grammar hook
             * is still live and held under mu: bs_slot_release reports it, and
             * by then the request may have freed the hook. */
            if (grammar_done && sl->reason_forced)
                sl->reason_forced_call = true;
            if ((int)next == sched->eos_id ||
                (int)next == sched->im_end_id || sl->budget <= 0 ||
                grammar_done) {
                sl->finished = true;
                sched->active_set_changed = true;
            } else {
                sl->next_token  = next;
                sl->token_ready = true;
            }
            pthread_cond_signal(&sl->cv);
        }
        bs_gpu_release_locked(sched);

        if (stats) {
            double t_done = bs_now_ms();
            fprintf(stderr, "[bs] step=%llu n=%d dead_ms=%.1f "
                    "turn_wait_ms=%.1f step_ms=%.1f dec_ms=%.1f "
                    "samp_ms=%.1f\n",
                    (unsigned long long)step_no++, snap_n,
                    t_prev_done > 0.0 ? t_gate - t_prev_done : 0.0,
                    t_launch - t_gate, t_done - t_launch,
                    t_dec - t_launch, t_done - t_dec);
            t_prev_done = t_done;
        }
    }

    /* Shutdown: wake every parked owner (their wait predicate checks
     * sched->shutdown).  bs_destroy broadcasts too — belt and braces. */
    for (int i = 0; i < sched->cap; i++)
        pthread_cond_broadcast(&sched->slots[i].cv);
    pthread_mutex_unlock(&sched->mu);
    return NULL;
}

/* ============================================================
 * Lazy GPU lane registration
 *
 * Called from bs_slot_bind with the GPU turn owned (so no other prefill,
 * bind or decode step can interleave — the double-entry race that wedged the
 * K80 is structurally impossible: the second binder queues on the turnstile
 * for its whole prefill and sees slots_gpu_init == true).  Registers every lane's
 * pn_state* with the GPU (allocates the slot-major conv/ssm buffers once and
 * seeds each lane) so decode_slots finds a valid slot_state[i] for every i.
 *
 * Fail-soft: a bind error propagates -1 immediately; nothing waits on it.
 * ============================================================ */
static int bs_lazy_init_gpu(pn_batch_sched *sched)
{
    /* Bind cap+1 engine slots: the extra top index is the rebuild_gpu_map
     * cycle-break scratch (users occupy [0, cap); decode never steps it). */
    for (int i = 0; i < sched->cap; i++) {
        if (pn_qwen35moe_decode_slot_bind(sched->model,
                                          sched->slots[i].state,
                                          i, sched->cap + 1) != 0) {
            fprintf(stderr, "[batch_sched] lazy gpu init: bind failed "
                    "(slot=%d cap=%d)\n", i, sched->cap);
            return -1;
        }
    }
    pthread_mutex_lock(&sched->mu);
    sched->slots_gpu_init = true;      /* driver reads this under mu */
    pthread_mutex_unlock(&sched->mu);
    return 0;
}
#endif /* USE_CUDA — driver_thread and helpers */

/* ============================================================
 * Lifecycle
 * ============================================================ */

pn_batch_sched *bs_create(pn_model *model, int cap, const char *model_name)
{
#ifndef USE_CUDA
    (void)model; (void)cap; (void)model_name;
    return NULL;   /* batch decode requires USE_CUDA */
#else
    if (!model || cap <= 0 || cap > BS_MAX_USERS)
        return NULL;   /* BS_MAX_SLOTS reserves the top index as scratch */

    /* Clamp to the memory-pressure concurrency ceiling up front so the
     * per-bind ladder guard can never refuse the lazy init later (which
     * would have turned every request into a 500).  We bind cap+1 engine
     * slots (the extra one is the rebuild scratch), so leave room for it
     * under the ceiling: cap ≤ ceiling-1. */
    int ceiling = pn_qwen35moe_gpu_ladder_max_slots();
    if (ceiling > 1 && cap > ceiling - 1) {
        fprintf(stderr, "[batch_sched] capping slots %d -> %d "
                "(memory-pressure ladder ceiling %d, incl. 1 scratch)\n",
                cap, ceiling - 1, ceiling);
        cap = ceiling - 1;
    }

    pn_batch_sched *sched = calloc(1, sizeof(*sched));
    /* Stamp the log name BEFORE the driver thread starts, so a force-close
     * logged by an early step already has it (calloc left it empty). */
    if (sched && model_name && model_name[0])
        snprintf(sched->model_name, sizeof(sched->model_name), "%s", model_name);
    if (!sched)
        return NULL;

    sched->model = model;
    sched->cap   = cap;

    /* GPU turnstile: ticket 0 is the first served. */
    sched->gpu_ticket_next    = 0;
    sched->gpu_ticket_serving = 0;

    /* L3 — in-driver sampling (default ON; QWEN35_BS_DRIVER_SAMPLE=0
     * restores the legacy owner-side lockstep). */
    {
        const char *env = getenv("QWEN35_BS_DRIVER_SAMPLE");
        sched->driver_sample = !(env && env[0] == '0');
    }
    sched->n_vocab   = model->config.n_vocab;
    sched->eos_id    = model->tokenizer ? ft_eos_id(model->tokenizer) : -1;
    sched->im_end_id = model->tokenizer
                           ? ft_token_to_id(model->tokenizer, "<|im_end|>")
                           : -1;

    if (pthread_cond_init(&sched->cv_gpu, NULL) != 0) {
        free(sched); return NULL;
    }
    if (pthread_mutex_init(&sched->mu, NULL) != 0) {
        pthread_cond_destroy(&sched->cv_gpu); free(sched); return NULL;
    }
    if (pthread_cond_init(&sched->cv_work, NULL) != 0) {
        pthread_mutex_destroy(&sched->mu);
        pthread_cond_destroy(&sched->cv_gpu); free(sched); return NULL;
    }
    if (pthread_cond_init(&sched->cv_idle, NULL) != 0) {
        pthread_cond_destroy(&sched->cv_work);
        pthread_mutex_destroy(&sched->mu);
        pthread_cond_destroy(&sched->cv_gpu); free(sched); return NULL;
    }

    for (int i = 0; i < cap; i++) {
        bs_slot *sl = &sched->slots[i];
        sl->slot_idx = i;
        sl->status   = BS_SLOT_IDLE;
        sl->gpu_idx  = -1;
        if (pthread_cond_init(&sl->cv, NULL) != 0) {
            /* Roll back already-initialised condvars. */
            for (int j = 0; j < i; j++)
                pthread_cond_destroy(&sched->slots[j].cv);
            pthread_cond_destroy(&sched->cv_idle);
            pthread_cond_destroy(&sched->cv_work);
            pthread_mutex_destroy(&sched->mu);
            pthread_cond_destroy(&sched->cv_gpu);
            free(sched);
            return NULL;
        }
        sl->state = pn_state_new(model);
        if (!sl->state) {
            /* Free everything allocated so far. */
            for (int j = 0; j <= i; j++) {
                if (sched->slots[j].state)
                    pn_state_free(sched->slots[j].state);
                if (j < i)
                    pthread_cond_destroy(&sched->slots[j].cv);
            }
            pthread_cond_destroy(&sched->slots[i].cv);
            pthread_cond_destroy(&sched->cv_idle);
            pthread_cond_destroy(&sched->cv_work);
            pthread_mutex_destroy(&sched->mu);
            pthread_cond_destroy(&sched->cv_gpu);
            free(sched);
            return NULL;
        }
    }

    /* GPU lane registration is LAZY: no GPU work here.  Binding all lanes
     * (cuMemAlloc for the slot-major buffers) immediately after the model
     * upload hangs on K80 because the CUDA context has not settled after a
     * multi-minute weight upload.  Deferred to the first bs_slot_bind. */
    sched->slots_gpu_init     = false;
    sched->n_active_gpu       = 0;
    sched->active_set_changed = false;
    for (int i = 0; i < BS_MAX_SLOTS; i++)
        sched->gpu_map[i] = -1;

    /* D1: per-model overflow queue.  Read queue depth from env. */
    sched->n_queued   = 0;
    sched->queue_cap  = BS_QUEUE_DEPTH_DEFAULT;
    {
        const char *env = getenv("BS_QUEUE_DEPTH");
        if (env) {
            int v = atoi(env);
            if (v > 0) sched->queue_cap = v;
        }
    }

    /* D3: pressure level cache (OK at startup until first poll). */
    sched->pressure_level = 0; /* PN_MEM_LEVEL_OK */

    sched->driver_running = true;
    sched->shutdown = false;
    if (pthread_create(&sched->driver, NULL, driver_thread, sched) != 0) {
        sched->driver_running = false;
        for (int i = 0; i < cap; i++) {
            pn_state_free(sched->slots[i].state);
            pthread_cond_destroy(&sched->slots[i].cv);
        }
        pthread_cond_destroy(&sched->cv_idle);
        pthread_cond_destroy(&sched->cv_work);
        pthread_mutex_destroy(&sched->mu);
        pthread_cond_destroy(&sched->cv_gpu);
        free(sched);
        return NULL;
    }

    fprintf(stderr, "[batch_sched] created scheduler (cap=%d) for model '%s'\n",
            cap, sched->model_name[0] ? sched->model_name : "?");
    return sched;
#endif /* USE_CUDA */
}

/* Both projections describe the SAME committed history, so they are always
 * invalidated together: a survivor would claim to be a prefix of a conversation
 * this slot no longer holds, and a resume off it would splice a request onto
 * somebody else's KV.  One function so that a future invalidation site cannot
 * clear one and forget the other. */
static void bs_slot_clear_projections(bs_slot *sl)
{
    sl->vis_n = 0;
    sl->vis_think_n = 0;
}

void bs_destroy(pn_batch_sched *sched)
{
    if (!sched)
        return;

    /* Signal the driver to stop and wake anything parked. */
    pthread_mutex_lock(&sched->mu);
    sched->shutdown = true;
    pthread_cond_broadcast(&sched->cv_work);
    pthread_cond_broadcast(&sched->cv_gpu);
    for (int i = 0; i < sched->cap; i++)
        pthread_cond_broadcast(&sched->slots[i].cv);
    pthread_mutex_unlock(&sched->mu);

    if (sched->driver_running)
        pthread_join(sched->driver, NULL);

    for (int i = 0; i < sched->cap; i++) {
        pn_state_free(sched->slots[i].state);
        pthread_cond_destroy(&sched->slots[i].cv);
        /* L3 buffers normally freed by bs_slot_release; defensive sweep. */
        free(sched->slots[i].recent);
        free(sched->slots[i].ring);
        free(sched->slots[i].hist);   /* PC-batched warm-state history */
        free(sched->slots[i].vis);        /* think-elided projection */
        free(sched->slots[i].vis_think);   /* reasoning-bearing projection */
        free(sched->slots[i].pend_vis);
        free(sched->slots[i].pend_vis_think);
        free(sched->slots[i].ct);     /* PC-batched committed-token log */
        bs_sys_snap_unref(&sched->slots[i]);
        free(sched->slots[i].sys_tokens);   /* SYS1-batched boundary snapshot */
    }
    pthread_cond_destroy(&sched->cv_idle);
    pthread_cond_destroy(&sched->cv_work);
    pthread_mutex_destroy(&sched->mu);
    pthread_cond_destroy(&sched->cv_gpu);
    free(sched);
}

/* ============================================================
 * Per-request API
 * ============================================================ */

/* Mark an IDLE slot PREFILL and clear its handshake state.  Under mu.
 * Does NOT touch sl->state (a dummy lane may still be in flight) nor the
 * PC-batched warm-state fields (sid/hist — preserved so a same-session
 * re-acquire can resume; bs_prefill_resume decides reset vs resume). */
static void bs_slot_mark_prefill(bs_slot *sl)
{
    sl->status       = BS_SLOT_PREFILL;
    sl->logits_ready = false;
    sl->token_ready  = false;
    sl->logits       = NULL;
    sl->next_token   = 0;
    sl->finished     = false;
    sl->step_error   = false;
    sl->sid_conflict = false;
    sl->gpu_idx      = -1;   /* no GPU decode state until bind */
}

/* Claim one IDLE slot (already under mu) and mark it PREFILL.  With a
 * non-empty `sid`, preference order (cheapest resume first):
 *   1. THIS session's warm IDLE slot (committed history → append delta);
 *   2. the IDLE slot holding THIS session's system-prefix snapshot
 *      (SYS1-batched: a divergent turn still resumes at the boundary —
 *      matters when the warm history was invalidated but the snapshot,
 *      which survives resets, was not);
 *   3. any free slot (this turn full-prefills).
 * Returns the slot index >= 0, or -1 if none free. */
static int try_claim_idle(pn_batch_sched *sched, const char *sid)
{
    if (sid && sid[0]) {
        for (int i = 0; i < sched->cap; i++) {
            bs_slot *sl = &sched->slots[i];
            if (sl->status == BS_SLOT_IDLE && sl->sid[0] &&
                strncmp(sl->sid, sid, sizeof(sl->sid)) == 0) {
                bs_slot_mark_prefill(sl);
                return i;
            }
        }
        for (int i = 0; i < sched->cap; i++) {
            bs_slot *sl = &sched->slots[i];
            if (sl->status == BS_SLOT_IDLE && sl->sys_sid[0] &&
                strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0) {
                bs_slot_mark_prefill(sl);
                return i;
            }
        }
    }
    /* No slot holds anything for THIS session. Prefer one that holds nothing for
     * ANYONE: taking the first idle slot evicts its warm history + system-prefix
     * snapshot, and it always picks slot 0, so a second session (a tab's reflect
     * pass, a second mode, a second tab) destroyed the first session's cache
     * while the model still had idle cold slots. */
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && !sl->sid[0] && !sl->sys_sid[0]) {
            bs_slot_mark_prefill(sl);
            return i;
        }
    }
    /* Every idle slot is warm for someone else — one of them has to lose. */
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE) {
            bs_slot_mark_prefill(sl);
            return i;
        }
    }
    return -1;
}

int bs_slot_acquire(pn_batch_sched *sched)
{
    if (!sched)
        return -1;
    pthread_mutex_lock(&sched->mu);
    int found = try_claim_idle(sched, NULL);
    pthread_mutex_unlock(&sched->mu);
    return found;
}

int bs_slot_acquire_biased(pn_batch_sched *sched, const char *sid)
{
    if (!sched)
        return -1;
    pthread_mutex_lock(&sched->mu);
    int found = try_claim_idle(sched, sid);
    pthread_mutex_unlock(&sched->mu);
    return found;
}

/* Reset a slot's pn_state to fresh (host mirror zeroed + GPU bindings
 * forgotten) and drop any PC-batched warm-state binding.  Must be called
 * owning the GPU turn (pn_qwen35moe_gpu_forget_state is host bookkeeping the
 * turn protects from a racing forward). */
static void bs_slot_reset_state(pn_batch_sched *sched, int idx)
{
    pn_state_reset(sched->slots[idx].state);
#ifdef USE_CUDA
    /* This slot's pn_state is REUSED across requests.  pn_state_reset cleared
     * only the host mirror; the GPU still thinks this state is "live" on some
     * layers (SSM d->host_state, single-user KV kv_host_id) with stale
     * contents from its previous request.  A long prefill would read that
     * residue and diverge from a fresh state (short prefills never reach the
     * stale positions, which is why the bug only showed on long prompts).
     * Forget the bindings so the upcoming prefill re-seeds every GPU shadow
     * from the freshly-zeroed host mirror.  Safe under the GPU turn: no
     * forward can race this host-side bookkeeping. */
    pn_qwen35moe_gpu_forget_state(sched->model, sched->slots[idx].state);
#endif
    /* The warm state is gone — this slot no longer holds any session. */
    sched->slots[idx].sid[0] = '\0';
    sched->slots[idx].hist_n = 0;
    bs_slot_clear_projections(&sched->slots[idx]);
    sched->slots[idx].ct_keep = 0;
    sched->slots[idx].ct_prompt_from = 0;
}

void bs_prefill_begin(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    /* From here to bs_slot_bind / bs_prefill_abort (modulo bs_prefill_yield
     * windows) this thread is the model's single GPU owner: no other
     * prefill chunk, no decode step.  FIFO ticket: a joiner waits at most
     * one step / one chunk of every earlier waiter — the free-running L3
     * driver cannot starve it. */
    bs_gpu_acquire(sched);
    bs_slot_reset_state(sched, idx);
}

void bs_prefill_resume(pn_batch_sched *sched, int idx, const char *sid,
                       const uint32_t *tokens, int n_ctx, int *delta_start)
{
    int capture_unused = 0;
    bs_prefill_resume_sys(sched, idx, sid, tokens, n_ctx, 0, delta_start,
                          &capture_unused);
}

/* ── SYS1-batched helpers (per-slot system-prefix boundary snapshot).  All
 * owner-only: the slot's owner holds the slot (PREFILL) and the GPU turn, so
 * no other thread reads or writes the sys_* fields concurrently (the affinity
 * scans read sys_sid under mu on IDLE slots only). ── */

/* Is this model one whose dual state supports snapshot/restore?  (Bonsai
 * shares the qwen35moe dual-state machinery — same rule as prefix_cache.c.) */
static int bs_sys_snapshot_supported(const pn_batch_sched *sched)
{
    return sched->model->config.arch == PN_ARCH_QWEN35MOE ||
           sched->model->config.arch == PN_ARCH_BONSAI;
}

/* Does this slot hold a boundary snapshot for EXACTLY this session + system
 * prefix?  Both guards are required: the sid pins the snapshot to its session
 * and the byte-match catches a changed system prompt. */
static int bs_sys_snap_matches(const bs_slot *sl, const char *sid,
                               const uint32_t *tokens, int sys_len)
{
    return sl->sys_snap && sl->sys_len == sys_len &&
           strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0 &&
           memcmp(sl->sys_tokens, tokens,
                  (size_t)sys_len * sizeof(uint32_t)) == 0;
}

/* ── Reroute: several conversations under one session id (bs_slot_reroute).
 * All under mu, reading IDLE slots only — their warm fields (sid, hist, the
 * projections, the boundary snapshot) change only under mu or while owned. ── */

/* What an IDLE slot's cached state is worth to this prompt. */
enum { BS_FIT_NONE, BS_FIT_SYS, BS_FIT_WARM };

/* Is p[0..n) a strict prefix of tokens[0..n_ctx)? */
static int bs_is_strict_prefix(const uint32_t *p, int n,
                               const uint32_t *tokens, int n_ctx)
{
    return p && n > 0 && n < n_ctx &&
           memcmp(p, tokens, (size_t)n * sizeof(uint32_t)) == 0;
}

/* The warm-history test mirrors the resume ladder's token checks (rungs 1 and
 * 1b); the GPU-side checks it cannot make here are re-made by the ladder,
 * which falls back to this slot's own snapshot on a miss. */
static int bs_slot_fit(const bs_slot *sl, const char *sid,
                       const uint32_t *tokens, int n_ctx, int sys_len)
{
    if (sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
        (bs_is_strict_prefix(sl->hist, sl->hist_n, tokens, n_ctx) ||
         bs_is_strict_prefix(sl->vis, sl->vis_n, tokens, n_ctx) ||
         bs_is_strict_prefix(sl->vis_think, sl->vis_think_n, tokens, n_ctx)))
        return BS_FIT_WARM;
    if (sys_len > 0 && sys_len < n_ctx &&
        bs_sys_snap_matches(sl, sid, tokens, sys_len))
        return BS_FIT_SYS;
    return BS_FIT_NONE;
}

static int bs_slot_holds_sid(const bs_slot *sl, const char *sid)
{
    return (sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0) ||
           (sl->sys_sid[0] && strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0);
}

/* The slot this prompt should run on (see bs_slot_reroute).  Under mu. */
static int bs_reroute_target_locked(const pn_batch_sched *sched, int idx,
                                    const char *sid, const uint32_t *tokens,
                                    int n_ctx, int sys_len)
{
    int best = idx;
    int best_fit = bs_slot_fit(&sched->slots[idx], sid, tokens, n_ctx, sys_len);
    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        if (i == idx || sl->status != BS_SLOT_IDLE)
            continue;
        int fit = bs_slot_fit(sl, sid, tokens, n_ctx, sys_len);
        if (fit > best_fit) {
            best = i;
            best_fit = fit;
        }
    }
    if (best_fit != BS_FIT_NONE || !bs_slot_holds_sid(&sched->slots[idx], sid))
        return best;
    /* A new conversation under this id: keep the other one's cache if a cold
     * slot can take this one instead. */
    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && !sl->sid[0] && !sl->sys_sid[0])
            return i;
    }
    return idx;
}

int bs_slot_reroute(pn_batch_sched *sched, int idx, const char *sid,
                    const uint32_t *tokens, int n_ctx, int sys_len)
{
    if (!sched || idx < 0 || idx >= sched->cap || !sid || !sid[0] ||
        !tokens || n_ctx <= 0)
        return idx;
    pthread_mutex_lock(&sched->mu);
    int to = bs_reroute_target_locked(sched, idx, sid, tokens, n_ctx, sys_len);
    int fit = BS_FIT_NONE;
    if (to != idx) {
        fit = bs_slot_fit(&sched->slots[to], sid, tokens, n_ctx, sys_len);
        bs_slot_mark_prefill(&sched->slots[to]);
        /* Claimed but never touched (no GPU turn, no state, no bind): handing
         * it back is just the status flip; its warm fields stay as they were. */
        sched->slots[idx].status = BS_SLOT_IDLE;
        pthread_cond_broadcast(&sched->cv_idle);
    }
    pthread_mutex_unlock(&sched->mu);
    if (to != idx)
        fprintf(stderr, "[batch_sched] session reroute slot=%d -> %d sid=%s: "
                "%s\n", idx, to, sid,
                fit == BS_FIT_WARM ? "warm history continues there"
                : fit == BS_FIT_SYS ? "system-prefix snapshot there"
                : "new conversation, cold slot keeps the other one's cache");
    return to;
}

/* Phase 10b: release a slot's reference on its boundary snapshot. A shared
 * snapshot (sys_snap_refs non-NULL) is freed only when the last sharer lets
 * go; a private one is freed immediately. */
static void bs_sys_snap_unref(bs_slot *sl)
{
    if (sl->sys_snap && sl->sys_snap_refs && --(*sl->sys_snap_refs) > 0) {
        sl->sys_snap = NULL;
        sl->sys_snap_refs = NULL;
        return;
    }
    pn_qwen35moe_state_snapshot_free(sl->sys_snap);
    free(sl->sys_snap_refs);
    sl->sys_snap = NULL;
    sl->sys_snap_refs = NULL;
}

/* Drop a slot's boundary snapshot (keeps the tokens buffer for realloc reuse).
 * sys_sid is cleared under mu — the affinity scans read it. */
static void bs_sys_snap_drop(pn_batch_sched *sched, bs_slot *sl)
{
    bs_sys_snap_unref(sl);
    sl->sys_len = 0;
    pthread_mutex_lock(&sched->mu);
    sl->sys_sid[0] = '\0';
    pthread_mutex_unlock(&sched->mu);
}

/* Restore the slot's boundary snapshot into its own pn_state, leaving the
 * state at pos == sys_len.  Host-side write into THIS slot's own mirrors (the
 * snapshot is a self-contained deep copy — no other slot's state is touched).
 * The GPU shadows still hold the end of some previous turn; a restore to a
 * non-zero pos never trips the pos==0 restart re-sync, so the bindings are
 * forgotten explicitly — the delta prefill then re-seeds [0, sys_len) SSM +
 * KV from the restored host mirror via the ordinary seed-on-host-change path
 * (the exact mechanism the warm-history resume validated on a K80 host).  GPU turn
 * must be held.  Returns 0 on success. */
static int bs_sys_snap_restore(pn_batch_sched *sched, bs_slot *sl, int sys_len)
{
    if (!bs_sys_snapshot_supported(sched))
        return -1;
    if (pn_qwen35moe_state_restore(sl->state->kv_cache, sl->sys_snap) != 0)
        return -1;
    sl->state->pos = sys_len;
#ifdef USE_CUDA
    pn_qwen35moe_gpu_forget_state(sched->model, sl->state);
#endif
    return 0;
}

/* How many token ids either side of the divergence the window dump prints. */
#define BS_DIVERGENCE_WINDOW 24

/* Longest decoded run the window prints per side (keeps one log line bounded). */
#define BS_DIVERGENCE_TEXT_MAX 1024

/* Decode [from, to) of `ids` into printable text, escaping newlines/tabs so the
 * whole window stays on ONE greppable line.  Owned result, caller frees.
 * Diagnostic-only (PN_REQPROF=1), so a NULL return just degrades the dump. */
static char *bs_decode_run(pn_model *model, const uint32_t *ids, int from, int to)
{
    if (!model || !model->tokenizer || to <= from)
        return NULL;
    char *out = calloc(BS_DIVERGENCE_TEXT_MAX + 8, 1);
    if (!out)
        return NULL;
    size_t used = 0;
    for (int i = from; i < to && used < BS_DIVERGENCE_TEXT_MAX; i++) {
        int id = (int)ids[i];
        char *raw = ft_decode(model->tokenizer, &id, 1);
        if (!raw)
            continue;
        char *piece = raw;
        char *decoded = NULL;
        if (ft_get_model_type(model->tokenizer) == 0) {
            decoded = gpt2_bytes_decode(raw);
            if (decoded)
                piece = decoded;
        }
        for (const char *p = piece; *p && used < BS_DIVERGENCE_TEXT_MAX; p++) {
            if (*p == '\n')      { out[used++] = '\\'; if (used < BS_DIVERGENCE_TEXT_MAX) out[used++] = 'n'; }
            else if (*p == '\t') { out[used++] = '\\'; if (used < BS_DIVERGENCE_TEXT_MAX) out[used++] = 't'; }
            else if (*p == '\r') { out[used++] = '\\'; if (used < BS_DIVERGENCE_TEXT_MAX) out[used++] = 'r'; }
            else                 { out[used++] = *p; }
        }
        free(decoded);
        free(raw);
    }
    out[used] = '\0';
    return out;
}

/* Print two token runs around the point where they diverge, as ids AND as
 * decoded text.  `first_diff` alone names WHICH token broke the prefix match;
 * the window names WHAT the client failed to reproduce after it, which is the
 * difference between "one unreproducible opener" (recoverable by resending it)
 * and "an entire span that was never streamed".  PN_REQPROF=1 only, so no
 * hot-path cost.  `a_label`/`b_label` name the two sides. */
static void bs_log_token_window(pn_model *model,
                                const char *a_label, const uint32_t *a, int a_n,
                                const char *b_label, const uint32_t *b, int b_n,
                                int first_diff)
{
    int from = first_diff - BS_DIVERGENCE_WINDOW;
    if (from < 0)
        from = 0;
    int a_to = first_diff + BS_DIVERGENCE_WINDOW;
    if (a_to > a_n) a_to = a_n;
    int b_to = first_diff + BS_DIVERGENCE_WINDOW;
    if (b_to > b_n) b_to = b_n;

    fprintf(stderr, "[reqprof]   %s[%d..] =", a_label, from);
    for (int i = from; i < a_to; i++)
        fprintf(stderr, " %u", a[i]);
    fprintf(stderr, "\n[reqprof]   %s[%d..] =", b_label, from);
    for (int i = from; i < b_to; i++)
        fprintf(stderr, " %u", b[i]);
    fprintf(stderr, "\n");

    char *a_txt = bs_decode_run(model, a, from, a_to);
    char *b_txt = bs_decode_run(model, b, from, b_to);
    fprintf(stderr, "[reqprof]   %s TEXT: |%s|\n", a_label, a_txt ? a_txt : "?");
    fprintf(stderr, "[reqprof]   %s TEXT: |%s|\n", b_label, b_txt ? b_txt : "?");
    free(a_txt);
    free(b_txt);
}

/* Phase 8 diagnostic: report WHY append-resume missed and fell back to the
 * system-prefix path. Every real multi-turn conversation measured in Phase 8
 * fell through here, and the per-turn cost is the re-prefill that fallback
 * forces — so the reason is the finding. PN_REQPROF=1 only. */
static void bs_log_append_resume_miss(pn_model *model, int idx, const char *sid,
                                      const bs_slot *sl,
                                      const uint32_t *tokens, int n_ctx)
{
    if (!pn_reqprof_enabled())
        return;
    const char *why;
    int first_diff = -1;
    if (!sid || !sid[0])
        why = "request carried no session id";
    else if (!sl->sid[0])
        why = "slot holds no committed history";
    else if (strncmp(sl->sid, sid, sizeof(sl->sid)) != 0)
        why = "slot history belongs to a different session";
    else if (sl->hist_n <= 0)
        why = "committed history empty";
    else if (sl->hist_n >= n_ctx)
        why = "committed history not shorter than the new context";
    else if ((int)sl->state->pos != sl->hist_n)
        why = "state pos != committed history length";
    else {
        why = "committed history is not a prefix of the new context";
        for (int i = 0; i < sl->hist_n; i++) {
            if (sl->hist[i] != tokens[i]) {
                first_diff = i;
                break;
            }
        }
    }
    fprintf(stderr, "[reqprof] append-resume MISS slot=%d sid=%s: %s "
            "(hist_n=%d pos=%d n_ctx=%d first_diff=%d hist_tok=%u new_tok=%u)\n",
            idx, sid ? sid : "-", why, sl->hist_n, (int)sl->state->pos, n_ctx,
            first_diff,
            first_diff >= 0 ? sl->hist[first_diff] : 0u,
            first_diff >= 0 ? tokens[first_diff] : 0u);
    if (first_diff >= 0)
        bs_log_token_window(model, "hist", sl->hist, sl->hist_n,
                            "new ", tokens, n_ctx, first_diff);
    fflush(stderr);
}

/* PROJECTION diagnostic: the projection is what actually decides the rung, and
 * it is NOT `hist` — so bs_log_append_resume_miss's first_diff (which is
 * measured against hist) points at the first think block the two sides disagree
 * about, never at whatever the client really failed to reproduce.  This dump
 * answers the only question that matters on a miss: at which index does the
 * SERVE's prediction of the client's re-render stop matching what the client
 * actually sent, and what is on each side.  `label` names WHICH projection
 * (think-elided / think-included) this run is about, so a session that misses
 * both can be told apart from one that is simply on the other convention.
 * PN_REQPROF=1 only. */
static void bs_log_vis_divergence(pn_model *model, int idx, const char *sid,
                                  const bs_slot *sl, const char *label,
                                  const uint32_t *proj, int proj_n,
                                  const uint32_t *tokens, int n_ctx)
{
    if (!pn_reqprof_enabled())
        return;
    int n = proj_n < n_ctx ? proj_n : n_ctx;
    int first_diff = -1;
    for (int i = 0; i < n; i++) {
        if (proj[i] != tokens[i]) {
            first_diff = i;
            break;
        }
    }
    fprintf(stderr, "[reqprof] %s MISS slot=%d sid=%s: projection is "
            "not a prefix of the new context (proj_n=%d hist_n=%d n_ctx=%d "
            "first_diff=%d proj_tok=%u new_tok=%u)\n", label, idx,
            sid ? sid : "-", proj_n, sl->hist_n, n_ctx, first_diff,
            first_diff >= 0 ? proj[first_diff] : 0u,
            first_diff >= 0 ? tokens[first_diff] : 0u);
    if (first_diff >= 0)
        bs_log_token_window(model, "proj", proj, proj_n,
                            "new ", tokens, n_ctx, first_diff);
    fflush(stderr);
}

/* ── SESSION-SUBSTITUTION detection ────────────────────────────────────────
 *
 * A session id names ONE conversation.  When a client posts a second, unrelated
 * conversation under the same id, every rung of the resume ladder misses, the
 * slot's boundary snapshot is replaced by the intruder's, and BOTH conversations
 * re-prefill in full from then on — silently.  Detect it and say so.
 *
 * The discriminator is the SYSTEM BLOCK: every legitimate history change (an
 * append, a think-elided re-render, a client-side compaction) keeps
 * tokens[0..sys_len) byte-identical, so a change there is the one thing that
 * cannot be a continuation.  See bs_slot_session_conflict in the header for the
 * full argument and the deliberate false-negative list.
 * ────────────────────────────────────────────────────────────────────────── */

/* The system-prefix record this slot holds for `sid`, if any.  Prefers the
 * boundary snapshot (which stores the exact prefix tokens it covers) and falls
 * back to the committed history, whose head IS the system block.  Returns 1 and
 * fills `out` and `out_n` on a hit, 0 when this slot holds nothing for `sid`. */
static int bs_recorded_sys_prefix(const bs_slot *sl, const char *sid,
                                  const uint32_t **out, int *out_n)
{
    if (sl->sys_sid[0] && strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0 &&
        sl->sys_tokens && sl->sys_len > 0) {
        *out = sl->sys_tokens;
        *out_n = sl->sys_len;
        return 1;
    }
    if (sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
        sl->hist && sl->hist_n > 0) {
        *out = sl->hist;
        *out_n = sl->hist_n;
        return 1;
    }
    return 0;
}

/* Set sl->sid_conflict and report (unconditionally — this is a client defect the
 * client can only learn about if the serve says it) when this turn's system
 * block differs from the one this session id last committed on this slot.
 * Called only after rungs 1 and 1b have missed: a hit on either proves the
 * system block matched, since both memcmp over a span that contains it. */
static void bs_note_session_substitution(pn_batch_sched *sched, int idx,
                                         const char *sid,
                                         const uint32_t *tokens, int n_ctx,
                                         int sys_len)
{
    bs_slot *sl = &sched->slots[idx];
    sl->sid_conflict = false;
    if (!sid || !sid[0] || sys_len <= 0 || sys_len > n_ctx)
        return;   /* no session id, or no system block to compare */

    const uint32_t *rec = NULL;
    int rec_n = 0;
    if (!bs_recorded_sys_prefix(sl, sid, &rec, &rec_n))
        return;   /* nothing recorded for this session here — cannot tell */

    /* A record shorter than this request's system block cannot contain it, so
     * the block changed; otherwise compare the block byte for byte. */
    if (rec_n >= sys_len &&
        memcmp(rec, tokens, (size_t)sys_len * sizeof(uint32_t)) == 0)
        return;   /* same system block → an ordinary history divergence */

    sl->sid_conflict = true;
    fprintf(stderr, "[batch_sched] WARN session-substitution slot=%d sid=%s: "
            "this session id already identified a conversation with a %d-token "
            "system prompt; this request carries a DIFFERENT %d-token one. Both "
            "conversations now re-prefill in full every turn. Use one session id "
            "per conversation.\n", idx, sid, rec_n, sys_len);
    fflush(stderr);
}

bool bs_slot_session_conflict(const pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return false;
    return sched->slots[idx].sid_conflict;
}

/* Is this slot's warm KV actually RE-USABLE, and not merely bookkept?
 *
 * Phase 10b (INV3) made the host KV mirror LAZY: between turns a session's rows
 * live in a GPU ring, not in host RAM, and bs_slot_finalize deliberately does not
 * materialize them (that is the per-session host footprint lazy mode removed).
 * A warm resume then forgets the GPU bindings and re-uploads from the host
 * mirror — which only works while a ring still holds the content.  When another
 * session's forward has taken the ring in between, nothing does, and the
 * resumed forward FAILS ITS STEP ("host mirror STALE and no GPU ring source"):
 * the turn returns with zero tokens generated.  Measured on the K80 test machine
 * 2026-08-02, four consecutive turns produced an EMPTY response this way.
 *
 * So the warm rungs must prove the rows are reachable before committing to
 * them.  Materializing the mirror here does that and makes the re-upload sound;
 * it is bounded by the session's own context (a few MiB over PCIe, once per
 * resumed turn) and it costs no host RAM the state had not already allocated.
 * A failure is not an error — it means another session won the ring, so the
 * caller falls through to a rung that re-prefills.
 *
 * GPU turn must be held. */
static bool bs_warm_kv_reusable(pn_batch_sched *sched, bs_slot *sl)
{
#ifdef USE_CUDA
    if (pn_qwen35moe_gpu_state_to_host(sched->model, sl->state) != 0) {
        if (pn_reqprof_enabled())
            fprintf(stderr, "[reqprof] warm resume refused slot=%d sid=%s: this "
                    "session's KV rows are no longer in any GPU ring\n",
                    sl->slot_idx, sl->sid);
        return false;
    }
    return true;
#else
    (void)sched; (void)sl;
    return true;
#endif
}

/* One candidate re-render of the committed history, as the serve predicted the
 * client would send it back.  Two exist because two conventions exist. */
typedef struct {
    const uint32_t *tok;
    int             n;
    const char     *label;
} bs_projection;

/* PROJECTION rung — resume a session whose committed history differs from the
 * client's re-render only in the think blocks, whichever way round.
 *
 * A projection is that re-render as the serve predicted it (see
 * bs_slot_note_visible).  When one is a strict prefix of the new context, every
 * KV row in [0, hist_n) is a correct continuation of the conversation the client
 * is describing, so the state resumes AS IS at pos == hist_n and only the
 * genuinely new tail is prefilled.  The two candidates differ in what they claim
 * about this turn's think block:
 *
 *   think-elided  — the client DROPPED it.  The block stays in the model's
 *                   context, which is where it was generated and where it
 *                   cannot be removed from (30 of 40 layers are a sequential
 *                   SSM recurrence; there is no rewind).  The KV then holds
 *                   strictly more than the client described.
 *   think-included — the client SENT IT BACK.  The KV holds the block as the
 *                   model SAMPLED it, the client describes it as re-tokenized
 *                   TEXT; same content, and the ids need not agree because the
 *                   KV is authoritative for a span both sides agree is there.
 *                   This is the strictly weaker claim of the two, and it is the
 *                   rung that lets a client send reasoning back without paying a
 *                   full re-prefill every turn.
 *
 * Neither candidate is trusted past its own memcmp: a divergence at ANY index
 * drops through to the re-prefilling rungs, which is the only safe answer —
 * reusing KV that does not correspond to the request's prefix is silently wrong
 * output, strictly worse than a slow turn.
 *
 * *delta is set in the CLIENT's index space (where the prefill reads from
 * `tokens`), which is hist_n - proj_n behind the state's position — that is the
 * whole point, and sg_prefill_range_sched positions tokens from state->pos, not
 * from the array index.  The committed log is then hist[0, hist_n) followed by
 * tokens[*delta, n_ctx), which is what ct_keep / ct_prompt_from carry to bind.
 *
 * Returns 1 on a hit (state untouched, ready to continue), 0 on a miss.
 * GPU turn must be held. */
static int bs_projection_resume(pn_batch_sched *sched, int idx,
                                const char *sid, const uint32_t *tokens,
                                int n_ctx, int *delta)
{
    bs_slot *sl = &sched->slots[idx];
    if (!sid || !sid[0] || !sl->sid[0])
        return 0;
    if (strncmp(sl->sid, sid, sizeof(sl->sid)) != 0)
        return 0;
    if (sl->hist_n <= 0 || (int)sl->state->pos != sl->hist_n)
        return 0;

    /* Reasoning-bearing first: it is the convention that keeps the model
     * reasoning, so on a session that uses it the elided candidate would only
     * log a misleading divergence before this one hit. */
    const bs_projection cands[] = {
        { sl->vis_think, sl->vis_think_n, "think-included" },
        { sl->vis,       sl->vis_n,       "think-elided"   },
    };
    const bs_projection *hit = NULL;
    for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]) && !hit; i++) {
        if (!cands[i].tok || cands[i].n <= 0)
            continue;
        /* Strictly shorter, so at least one new token remains to prefill and the
         * next-token logits are recovered (same rule as the exact-prefix rung). */
        if (cands[i].n >= n_ctx) {
            if (pn_reqprof_enabled())
                fprintf(stderr, "[reqprof] %s MISS slot=%d sid=%s: projection "
                        "not shorter than the new context (proj_n=%d hist_n=%d "
                        "n_ctx=%d)\n", cands[i].label, idx, sid, cands[i].n,
                        sl->hist_n, n_ctx);
            continue;
        }
        if (memcmp(cands[i].tok, tokens,
                   (size_t)cands[i].n * sizeof(uint32_t)) == 0)
            hit = &cands[i];
    }
    if (!hit) {
        for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); i++)
            if (cands[i].tok && cands[i].n > 0 && cands[i].n < n_ctx)
                bs_log_vis_divergence(sched->model, idx, sid, sl,
                                      cands[i].label, cands[i].tok, cands[i].n,
                                      tokens, n_ctx);
        return 0;
    }

    /* Keeping the model's own think tokens costs context: the KV ends this turn
     * at hist_n + (n_ctx - proj_n), which is hist_n - proj_n AHEAD of the n_ctx
     * that admission already cleared.  Refuse the rung rather than overrun the
     * session's window — the fallback rungs re-prefill and land exactly at
     * n_ctx. */
    int cap = pn_session_ctx_cap(sched->model);
    if (cap > 0 && sl->hist_n + (n_ctx - hit->n) > cap)
        return 0;
    if (!bs_warm_kv_reusable(sched, sl))
        return 0;

#ifdef USE_CUDA
    /* Same re-seed contract as the exact-prefix rung: another session may have
     * claimed the shared SSM/KV buffers between this session's turns, so forget
     * the GPU bindings and let the delta prefill re-upload [0, hist_n) from the
     * warm host mirror finalize left authoritative. */
    pn_qwen35moe_gpu_forget_state(sched->model, sl->state);
#endif
    sl->ct_keep = sl->hist_n;
    sl->ct_prompt_from = hit->n;
    *delta = hit->n;
    if (pn_reqprof_enabled()) {
        /* hist_n - proj_n is NOT printed as a count: it is negative whenever the
         * reasoning-bearing projection runs past the KV, which is routine. */
        fprintf(stderr, "[reqprof] append-resume HIT (%s) slot=%d sid=%s: "
                "projection %d over a %d-token KV, resume at pos %d, "
                "prefill %d of %d\n", hit->label, idx, sid, hit->n,
                sl->hist_n, sl->hist_n, n_ctx - hit->n, n_ctx);
        fflush(stderr);
    }
    return 1;
}

/* Copy a pending projection into the slot's published slot, refusing one that
 * slot.  Returns the published length, or 0 when nothing was published (OOM is
 * benign: the session just loses that rung for a turn and re-prefills).
 *
 * No length relation to the committed history is imposed here, and the
 * reasoning-bearing projection is why.  Its length is
 * hist_n +/- a couple of tokens by construction: the client's rendering carries
 * the template's own separators ("<think>\n" ... "</think>\n\n") where the KV
 * carries whatever the model sampled around the same text, and the emitter
 * trimmed the leading newline out of the reasoning string.  A
 * "must not exceed hist_n" tripwire therefore lands right on the boundary and
 * would silently disable the rung on a coin flip.  It is not a correctness gate
 * either way: the gates are the full memcmp against the request and the strict
 * proj_n < n_ctx in bs_projection_resume, and a LONGER projection is the
 * conservative direction — more tokens must match before any KV is reused. */
static int bs_publish_projection(uint32_t **dst, int *cap,
                                 const uint32_t *pend, int pend_n)
{
    if (pend_n <= 0)
        return 0;
    if (*cap < pend_n) {
        uint32_t *v = realloc(*dst, (size_t)pend_n * sizeof(uint32_t));
        if (!v)
            return 0;
        *dst = v;
        *cap = pend_n;
    }
    memcpy(*dst, pend, (size_t)pend_n * sizeof(uint32_t));
    return pend_n;
}

void bs_slot_publish_projections(bs_slot *sl, int hist_n)
{
    if (!sl)
        return;
    bs_slot_clear_projections(sl);
    /* The think-ELIDED projection drops every think block, so one longer than
     * the history it projects is nonsense — refuse it (the tripwire this rung
     * shipped with).  The reasoning-bearing one puts the block back and is
     * legitimately hist_n-sized either way, so it carries no such bound; see
     * bs_publish_projection for why a bound there would be a coin flip. */
    if (sl->pend_vis_n <= hist_n)
        sl->vis_n = bs_publish_projection(&sl->vis, &sl->vis_cap,
                                          sl->pend_vis, sl->pend_vis_n);
    sl->vis_think_n = bs_publish_projection(&sl->vis_think, &sl->vis_think_cap,
                                            sl->pend_vis_think,
                                            sl->pend_vis_think_n);
}

/* Build one pending projection into `dst` as head ++ mid ++ tail.  `mid` may be
 * empty (the think-elided candidate).  Returns the length, or 0 on OOM — benign,
 * the turn just loses that candidate. */
static int bs_build_projection(uint32_t **dst, int *cap,
                               const uint32_t *head, int head_n,
                               const uint32_t *mid, int mid_n,
                               const uint32_t *tail, int tail_n)
{
    int want = head_n + mid_n + tail_n;
    if (*cap < want) {
        uint32_t *v = realloc(*dst, (size_t)want * sizeof(uint32_t));
        if (!v)
            return 0;
        *dst = v;
        *cap = want;
    }
    memcpy(*dst, head, (size_t)head_n * sizeof(uint32_t));
    if (mid_n > 0)
        memcpy(*dst + head_n, mid, (size_t)mid_n * sizeof(uint32_t));
    memcpy(*dst + head_n + mid_n, tail, (size_t)tail_n * sizeof(uint32_t));
    return want;
}

void bs_slot_note_visible(pn_batch_sched *sched, int idx,
                          const uint32_t *prompt, int n_prompt,
                          int think_prefill_len,
                          const uint32_t *think_block, int think_block_n,
                          const uint32_t *rerender, int rerender_n)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    bs_slot *sl = &sched->slots[idx];
    sl->pend_vis_n = 0;
    sl->pend_vis_think_n = 0;
    /* The prompt minus its forced generation prefix is the client's own
     * rendering of every earlier turn — whichever think-block convention it
     * follows — so it is exactly what it will resend.  `rerender` is this turn's
     * reply put back through that same rendering, so the two concatenated ARE
     * next turn's context up to the point the client appends to it. */
    int head_n = n_prompt - think_prefill_len;
    if (!prompt || n_prompt <= 0 || think_prefill_len < 0 || head_n <= 0 ||
        !rerender || rerender_n <= 0) {
        if (pn_reqprof_enabled())
            fprintf(stderr, "[reqprof] note_visible slot=%d: NO projection "
                    "(n_prompt=%d think_prefill=%d rerender_n=%d)\n",
                    idx, n_prompt, think_prefill_len, rerender_n);
        return;
    }
    if (!think_block || think_block_n <= 0)
        think_block_n = 0;

    sl->pend_vis_n = bs_build_projection(&sl->pend_vis, &sl->pend_vis_cap,
                                         prompt, head_n, NULL, 0,
                                         rerender, rerender_n);
    if (think_block_n > 0)
        sl->pend_vis_think_n =
            bs_build_projection(&sl->pend_vis_think, &sl->pend_vis_think_cap,
                                prompt, head_n, think_block, think_block_n,
                                rerender, rerender_n);
    if (pn_reqprof_enabled()) {
        char *tail_txt = bs_decode_run(sched->model, rerender, 0, rerender_n);
        fprintf(stderr, "[reqprof] note_visible slot=%d: projections elided=%d "
                "included=%d (head %d + think %d + re-render %d, n_prompt=%d "
                "think_prefill=%d) RENDER: |%s|\n", idx, sl->pend_vis_n,
                sl->pend_vis_think_n, head_n, think_block_n, rerender_n,
                n_prompt, think_prefill_len, tail_txt ? tail_txt : "?");
        free(tail_txt);
        fflush(stderr);
    }
}

void bs_prefill_resume_sys(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *tokens, int n_ctx, int sys_len,
                           int *delta_start, int *capture_sys)
{
    if (delta_start)
        *delta_start = 0;
    if (capture_sys)
        *capture_sys = 0;
    if (!sched || idx < 0 || idx >= sched->cap || !tokens || n_ctx <= 0)
        return;
    bs_gpu_acquire(sched);   /* same GPU-owner contract as bs_prefill_begin */

    bs_slot *sl = &sched->slots[idx];
    /* A system prefix is only usable when a conversation follows it. */
    int sys_ok = sid && sid[0] && sys_len > 0 && sys_len < n_ctx;
    int delta = 0;
    /* Default committed-log seeding: the log IS the prompt (rungs 1, 2 and 3).
     * Only the think-elision rung overrides it. */
    sl->ct_keep = 0;
    sl->ct_prompt_from = 0;
    sl->sid_conflict = false;
    /* Resume ladder — mirrors pn_prefix_cache_acquire_sys.
     *
     * 1. Append resume: THIS slot's warm state belongs to `sid`, its committed
     *    history is an exact prefix of the new context, and the state sits
     *    exactly at that history's end (pos == hist_n).  The agentic append
     *    case: prefill only [hist_n, n_ctx).  hist_n < n_ctx is strict so at
     *    least one delta token remains to recover the next-token logits (the
     *    degenerate full-hit never happens in a real chat turn — an assistant
     *    opener is always appended — and falls through to a full reset). */
    if (sid && sid[0] && sl->sid[0] &&
        strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
        sl->hist_n > 0 && sl->hist_n < n_ctx &&
        (int)sl->state->pos == sl->hist_n &&
        memcmp(sl->hist, tokens,
               (size_t)sl->hist_n * sizeof(uint32_t)) == 0 &&
        bs_warm_kv_reusable(sched, sl)) {
        delta = sl->hist_n;   /* warm: resume, do NOT reset */
#ifdef USE_CUDA
        /* Force every GPU shadow of this warm state to re-seed from its HOST
         * mirror on the delta-prefill forward.  Between this session's turns
         * another session's forward may have claimed the shared SSM buffers /
         * the shared KV ring, and this state's own live decode advanced its
         * per-slot buffers (not the shared ones the prefill forward re-seeds).
         * finalize left the host mirror authoritative at pos==hist_n; forgetting
         * the GPU bindings (host-bookkeeping only — the host mirror is NOT
         * touched) makes the resumed forward re-upload [0,hist_n) SSM + KV from
         * that warm mirror, then continue the delta.  Safe under the GPU turn
         * just acquired. */
        pn_qwen35moe_gpu_forget_state(sched->model, sl->state);
#endif
    } else if (bs_projection_resume(sched, idx, sid, tokens, n_ctx, &delta)) {
        /* 1b. Projection append resume — the client's re-render of the history
         *     differs from the committed KV only in the think blocks, either
         *     because it dropped them or because it sent the reasoning back as
         *     text.  See bs_projection_resume. */
    } else {
        /* 2. System-prefix resume: divergent history but this slot holds THIS
         *    session's boundary snapshot over these exact system tokens (the
         *    reasoning-model turn).  Restore it and prefill only the
         *    conversation.  The stale committed history is dropped —
         *    bs_slot_finalize rebuilds it at end of turn. */
        /* Both warm rungs missed.  Before the fallbacks mutate this slot,
         * decide whether the miss is an ordinary history divergence or a second
         * conversation wearing this session's id — and say so out loud. */
        bs_note_session_substitution(sched, idx, sid, tokens, n_ctx, sys_len);
        bs_log_append_resume_miss(sched->model, idx, sid, sl, tokens, n_ctx);
        int resumed = 0;
        if (sys_ok && bs_sys_snap_matches(sl, sid, tokens, sys_len)) {
            if (bs_sys_snap_restore(sched, sl, sys_len) == 0) {
                resumed = 1;
                delta = sys_len;
                pthread_mutex_lock(&sched->mu);
                sl->sid[0] = '\0';   /* history no longer matches the state */
                sl->hist_n = 0;
                bs_slot_clear_projections(sl);
                pthread_mutex_unlock(&sched->mu);
                fprintf(stderr, "[batch_sched] sys-prefix reuse slot=%d "
                        "sid=%s: resume at %d, prefill %d of %d tokens\n",
                        idx, sid, sys_len, n_ctx - sys_len, n_ctx);
            } else {
                /* Restore refused (dim/KV-mode change): the snapshot can never
                 * apply again — drop it so a capture can replace it.  Nothing
                 * was written (the restore validates dims before any copy). */
                bs_sys_snap_drop(sched, sl);
            }
        }
        /* 3. Fresh/diverged with no usable snapshot: full reset + prefill. */
        if (!resumed) {
            bs_slot_reset_state(sched, idx);
            delta = 0;
        }
    }

    /* Full prefill ahead + no snapshot covering these exact system tokens →
     * ask the caller to capture the boundary state mid-prefill (replacing a
     * stale snapshot from a changed system prompt or another session). */
    if (capture_sys && delta == 0 && sys_ok && bs_sys_snapshot_supported(sched)
        && !bs_sys_snap_matches(sl, sid, tokens, sys_len))
        *capture_sys = 1;
    if (delta_start)
        *delta_start = delta;
}

int bs_prefill_capture_sys(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *tokens, int sys_len)
{
    if (!sched || idx < 0 || idx >= sched->cap || !sid || !sid[0] || !tokens ||
        sys_len <= 0)
        return -1;
    if (!bs_sys_snapshot_supported(sched))
        return -1;
    bs_slot *sl = &sched->slots[idx];
    /* The capture protocol: resume_sys set capture_sys, the caller prefilled
     * exactly the system prefix — the state must sit at that boundary. */
    if ((int)sl->state->pos != sys_len)
        return -1;

    /* Phase 10b: if another slot already holds a snapshot of this EXACT
     * system prefix, share it (refcounted) instead of materializing +
     * deep-copying another ~68 MiB host image — the content depends only on
     * the weights + the prefix tokens, so any slot's snapshot of a byte-
     * identical prefix is interchangeable. The sid pinning that guards
     * RESUME lives in bs_sys_snap_matches (per-slot sys_sid), which is
     * per-slot state and unaffected by sharing the snapshot bytes. Safe to
     * scan: sys_snap/sys_len/sys_tokens are only mutated under the GPU
     * turn, which this caller holds. */
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *o = &sched->slots[i];
        if (o == sl || !o->sys_snap || o->sys_len != sys_len ||
            memcmp(o->sys_tokens, tokens,
                   (size_t)sys_len * sizeof(uint32_t)) != 0)
            continue;
        if (!o->sys_snap_refs) {
            o->sys_snap_refs = malloc(sizeof(int));
            if (!o->sys_snap_refs)
                break;   /* OOM: fall through to a private capture */
            *o->sys_snap_refs = 1;
        }
        uint32_t *st_tok = realloc(sl->sys_tokens,
                                   (size_t)sys_len * sizeof(uint32_t));
        if (!st_tok)
            break;       /* fall through to a private capture */
        memcpy(st_tok, tokens, (size_t)sys_len * sizeof(uint32_t));
        bs_sys_snap_unref(sl);
        sl->sys_snap = o->sys_snap;
        sl->sys_snap_refs = o->sys_snap_refs;
        ++(*sl->sys_snap_refs);
        sl->sys_tokens = st_tok;
        sl->sys_len = sys_len;
        pthread_mutex_lock(&sched->mu);
        snprintf(sl->sys_sid, sizeof(sl->sys_sid), "%s", sid);
        pthread_mutex_unlock(&sched->mu);
        fprintf(stderr, "[batch_sched] sys-prefix SHARED slot=%d <- slot=%d "
                "sid=%s: %d tokens (refs=%d)\n", idx, i, sid, sys_len,
                *sl->sys_snap_refs);
        return 0;
    }

#ifdef USE_CUDA
    /* Make the host mirror self-contained (GPU SSM image D2H + stale KV
     * mirror refresh) before copying it — the same D2H the L2 yield protocol
     * performs at every chunk boundary, so continuing the prefill afterwards
     * is unaffected (the GPU state is read, not invalidated, and no other
     * forward can interleave: the GPU turn is held). */
    if (pn_qwen35moe_gpu_state_to_host(sched->model, sl->state) != 0)
        return -1;
#endif

    void *snap = pn_qwen35moe_state_snapshot(sl->state->kv_cache, sys_len);
    if (!snap)
        return -1;
    uint32_t *st = realloc(sl->sys_tokens, (size_t)sys_len * sizeof(uint32_t));
    if (!st) {
        pn_qwen35moe_state_snapshot_free(snap);
        return -1;
    }
    memcpy(st, tokens, (size_t)sys_len * sizeof(uint32_t));
    bs_sys_snap_unref(sl);
    sl->sys_snap = snap;
    sl->sys_snap_refs = NULL;   /* private until another slot shares it */
    sl->sys_tokens = st;
    sl->sys_len = sys_len;
    pthread_mutex_lock(&sched->mu);   /* sys_sid is read by the affinity scans */
    snprintf(sl->sys_sid, sizeof(sl->sys_sid), "%s", sid);
    pthread_mutex_unlock(&sched->mu);
    fprintf(stderr, "[batch_sched] sys-prefix captured slot=%d sid=%s: "
            "%d tokens\n", idx, sid, sys_len);
    return 0;
}

/* PC-batched — snapshot the slot's live SSM + KV into its pn_state host mirror
 * and publish the driver-maintained committed-token log as this slot's warm
 * history for next-turn delta reuse.  Owner-called after decode (before
 * release); takes + releases the GPU turn itself.  No-op if the slot was not
 * armed for caching (no session id at bind). */
int bs_slot_finalize(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return -1;

    bs_slot *sl = &sched->slots[idx];

    /* Is this turn armed for caching? (armed_sid set at bind; driver appends to
     * ct under mu.  The owner is the sole toucher of this slot post-decode.) */
    pthread_mutex_lock(&sched->mu);
    int armed = (sl->armed_sid[0] != '\0');
    pthread_mutex_unlock(&sched->mu);
    if (!armed)
        return 0;   /* not caching this turn */

    bs_gpu_acquire(sched);   /* exclusive: no decode step / rebuild can run */
    /* Read gpu_idx UNDER the GPU turn: rebuild_gpu_map (which relocates a
     * slot's gpu_idx via decode_slot_move) only runs inside the driver's turn,
     * so while we hold the turn gpu_idx cannot move out from under us. */
    pthread_mutex_lock(&sched->mu);
    int gpu_idx = sl->gpu_idx;
    pthread_mutex_unlock(&sched->mu);
    int rc = 0;
#ifdef USE_CUDA
    if (gpu_idx < 0) {
        rc = -1;   /* slot never bound a GPU decode state (e.g. EOS at token 1) */
    } else {
        /* Refresh the host mirror and pull this slot's per-slot SSM images
         * back into the pn_state host mirror. Phase 10b lazy-mirror mode:
         * KV stays in the GPU slot ring between turns (the warm resume path
         * re-uses it in place; a rebind seeds it D2D), so only the SSM half
         * is captured — materializing the KV here would recreate the exact
         * per-session host footprint lazy mode removes. Eager mode keeps
         * the full legacy capture. */
        if (pn_qwen35moe_host_kv_mirror_eager()) {
            if (pn_qwen35moe_gpu_state_to_host(sched->model, sl->state) != 0)
                rc = -1;
        } else if (pn_qwen35moe_gpu_state_ssm_to_host(sched->model,
                                                      sl->state) != 0) {
            rc = -1;
        }
        if (rc == 0 &&
            pn_qwen35moe_decode_slot_ssm_to_host(sched->model, gpu_idx,
                                                 sl->state) != 0)
            rc = -1;
    }
#else
    (void)gpu_idx;
#endif
    bs_gpu_release(sched);

    pthread_mutex_lock(&sched->mu);
    if (rc != 0) {
        /* Warm state is untrustworthy — disarm + invalidate so a future
         * affinity hit re-full-prefills rather than reusing a stale snapshot. */
        sl->armed_sid[0] = '\0';
        sl->sid[0] = '\0';
        sl->hist_n = 0;
        bs_slot_clear_projections(sl);
        pthread_mutex_unlock(&sched->mu);
        return -1;
    }

    /* Sanity: the committed log length must equal the state position — that is
     * the invariant the whole delta-reuse rests on.  If it ever diverges, do
     * not publish a mismatched history. */
    if ((int)sl->state->pos != sl->ct_n) {
        sl->armed_sid[0] = '\0';
        sl->sid[0] = '\0';
        sl->hist_n = 0;
        bs_slot_clear_projections(sl);
        pthread_mutex_unlock(&sched->mu);
        return -1;
    }

    /* Publish the committed log (ct → hist) + session as this slot's warm
     * history (read by bs_slot_acquire_biased / bs_prefill_resume). */
    int n_ctx = sl->ct_n;
    if (sl->hist_cap < n_ctx) {
        uint32_t *h = realloc(sl->hist, (size_t)n_ctx * sizeof(uint32_t));
        if (!h) {
            sl->armed_sid[0] = '\0';
            sl->sid[0] = '\0';
            sl->hist_n = 0;
            bs_slot_clear_projections(sl);
            pthread_mutex_unlock(&sched->mu);
            return -1;
        }
        sl->hist = h;
        sl->hist_cap = n_ctx;
    }
    memcpy(sl->hist, sl->ct, (size_t)n_ctx * sizeof(uint32_t));
    sl->hist_n = n_ctx;
    bs_slot_publish_projections(sl, n_ctx);
    snprintf(sl->sid, sizeof(sl->sid), "%s", sl->armed_sid);
    sl->armed_sid[0] = '\0';   /* consumed */
    pthread_mutex_unlock(&sched->mu);
    return 0;
}

/* ============================================================
 * Session warm-tier EXPORT / RESTORE / RELEASE — batched arm
 * (see the block comment in batch_sched.h)
 * ============================================================ */

/* Reserve an IDLE slot for an out-of-band (non-request) operation by marking
 * it PREFILL: try_claim_idle then skips it, the driver excludes PREFILL lanes
 * from its active set, and the owner-only warm-state and boundary-snapshot
 * fields become ours.  Under mu.  Paired with bs_slot_unreserve_locked. */
static void bs_slot_reserve_locked(bs_slot *sl)
{
    sl->status = BS_SLOT_PREFILL;
}

/* Release an out-of-band reservation: back to IDLE and wake anyone queued on
 * a free slot (the D1 waiters).  Under mu. */
static void bs_slot_unreserve_locked(pn_batch_sched *sched, bs_slot *sl)
{
    sl->status = BS_SLOT_IDLE;
    pthread_cond_broadcast(&sched->cv_idle);
}

bs_session_result bs_session_export(pn_batch_sched *sched, const char *sid,
                                    void **snap_out, uint32_t **tokens_out,
                                    int *n_tok_out)
{
    if (snap_out)   *snap_out = NULL;
    if (tokens_out) *tokens_out = NULL;
    if (n_tok_out)  *n_tok_out = 0;
    if (!sched || !sid || !sid[0] || !snap_out)
        return BS_SESSION_ERROR;
    if (!bs_sys_snapshot_supported(sched))
        return BS_SESSION_ERROR;

    /* Locate this session's warm slot.  A slot that is not IDLE is mid-turn:
     * its pos runs ahead of the committed history and the driver is still
     * appending, so it is not a clean export point (the single-user path
     * refuses the same shape via `state->pos != n_tok`).  Report BUSY so the
     * caller can tell "retry after the turn" from "no such session". */
    pthread_mutex_lock(&sched->mu);
    int idx = -1, busy = 0;
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *s = &sched->slots[i];
        if (!s->sid[0] || strncmp(s->sid, sid, sizeof(s->sid)) != 0)
            continue;
        if (s->status != BS_SLOT_IDLE) { busy = 1; continue; }
        if (s->hist_n <= 0)
            continue;
        idx = i;
        break;
    }
    if (idx < 0) {
        pthread_mutex_unlock(&sched->mu);
        return busy ? BS_SESSION_BUSY : BS_SESSION_NOT_FOUND;
    }
    bs_slot *sl = &sched->slots[idx];
    bs_slot_reserve_locked(sl);
    int n_tok = sl->hist_n;
    pthread_mutex_unlock(&sched->mu);

    /* The committed log and the state position must agree — that invariant is
     * what makes the snapshot a coherent resume point (bs_slot_finalize
     * enforces it at publish time; re-check because we own the slot now). */
    bs_session_result rc = BS_SESSION_OK;
    if ((int)sl->state->pos != n_tok)
        rc = BS_SESSION_ERROR;

    void *snap = NULL;
    if (rc == BS_SESSION_OK) {
        bs_gpu_acquire(sched);   /* exclusive: no decode step can move the ring */
#ifdef USE_CUDA
        /* INV3: the KV rows live in the GPU decode-slot ring, not in a host
         * mirror.  gpu_state_to_host drives qw_kv_layer_materialize for every
         * layer still flagged stale, bounded to [0, min(pos, max_seq)), and
         * fails loudly if no ring holds the content rather than serialising a
         * stale mirror.  The SSM half of the slot's mirror is already current
         * (bs_slot_finalize pulled the per-slot images down at commit time). */
        if (pn_qwen35moe_gpu_state_to_host(sched->model, sl->state) != 0) {
            /* No GPU ring still holds this session's rows — another session
             * took the ring after this slot was released.  The state is gone,
             * not broken: report NOT_FOUND so the client falls back to replay,
             * and drop the warm binding so no later turn tries to append-resume
             * onto content that no longer exists. */
            fprintf(stderr, "[batch_sched] session export slot=%d sid=%s: "
                    "KV materialize failed (no ring source) — warm state "
                    "invalidated\n", idx, sid);
            pthread_mutex_lock(&sched->mu);
            sl->sid[0] = '\0';
            sl->hist_n = 0;
            bs_slot_clear_projections(sl);
            pthread_mutex_unlock(&sched->mu);
            rc = BS_SESSION_NOT_FOUND;
        }
#endif
        if (rc == BS_SESSION_OK) {
            snap = pn_qwen35moe_state_snapshot(sl->state->kv_cache, n_tok);
            if (!snap)
                rc = BS_SESSION_ERROR;
        }
        bs_gpu_release(sched);
    }

    /* Copy the committed history out while we still hold the slot. */
    uint32_t *toks = NULL;
    if (rc == BS_SESSION_OK) {
        toks = (uint32_t *)malloc((size_t)n_tok * sizeof(uint32_t));
        if (!toks) {
            pn_qwen35moe_state_snapshot_free(snap);
            snap = NULL;
            rc = BS_SESSION_ERROR;
        } else {
            memcpy(toks, sl->hist, (size_t)n_tok * sizeof(uint32_t));
        }
    }

    pthread_mutex_lock(&sched->mu);
    bs_slot_unreserve_locked(sched, sl);
    pthread_mutex_unlock(&sched->mu);

    if (rc != BS_SESSION_OK)
        return rc;
    *snap_out = snap;
    if (tokens_out) *tokens_out = toks; else free(toks);
    if (n_tok_out)  *n_tok_out = n_tok;
    fprintf(stderr, "[batch_sched] session exported slot=%d sid=%s: %d tokens\n",
            idx, sid, n_tok);
    return BS_SESSION_OK;
}

bs_session_result bs_session_restore(pn_batch_sched *sched, const char *sid,
                                     const void *snap, const uint32_t *tokens,
                                     int n_tok)
{
    if (!sched || !sid || !sid[0] || !snap || !tokens || n_tok <= 0)
        return BS_SESSION_ERROR;
    if (!bs_sys_snapshot_supported(sched))
        return BS_SESSION_ERROR;
    if (pn_qwen35moe_state_snapshot_pos(snap) != n_tok)
        return BS_SESSION_ERROR;

    /* Prefer this session's own slot (a re-restore replaces its warm state),
     * then any IDLE slot.  Never displace an active one. */
    pthread_mutex_lock(&sched->mu);
    int idx = -1;
    for (int i = 0; i < sched->cap && idx < 0; i++) {
        bs_slot *s = &sched->slots[i];
        if (s->status == BS_SLOT_IDLE && s->sid[0] &&
            strncmp(s->sid, sid, sizeof(s->sid)) == 0)
            idx = i;
    }
    for (int i = 0; i < sched->cap && idx < 0; i++)
        if (sched->slots[i].status == BS_SLOT_IDLE)
            idx = i;
    if (idx < 0) {
        pthread_mutex_unlock(&sched->mu);
        return BS_SESSION_BUSY;
    }
    bs_slot *sl = &sched->slots[idx];
    bs_slot_reserve_locked(sl);
    pthread_mutex_unlock(&sched->mu);

    /* Grow the history buffer BEFORE mutating the state, so an OOM leaves the
     * slot's previous warm state intact rather than half-replaced. */
    bs_session_result rc = BS_SESSION_OK;
    if (sl->hist_cap < n_tok) {
        uint32_t *h = realloc(sl->hist, (size_t)n_tok * sizeof(uint32_t));
        if (!h)
            rc = BS_SESSION_ERROR;
        else { sl->hist = h; sl->hist_cap = n_tok; }
    }

    if (rc == BS_SESSION_OK) {
        bs_gpu_acquire(sched);
        /* Host write into THIS slot's own mirrors; state_restore clears
         * kv_mirror_stale (the mirror is now authoritative — this is the one
         * case INV3 keeps host-sourced) and bumps the KV content generation, so
         * every ring watermark still carrying the old generation stops matching
         * and the next prefill re-uploads [0, n_tok) from the restored mirror.
         * The explicit forget covers the SSM shadows the same way — exactly the
         * SYS1-batched boundary-restore protocol (bs_sys_snap_restore). */
        if (pn_qwen35moe_state_restore(sl->state->kv_cache, snap) != 0) {
            rc = BS_SESSION_ERROR;
        } else {
            sl->state->pos = n_tok;
#ifdef USE_CUDA
            pn_qwen35moe_gpu_forget_state(sched->model, sl->state);
#endif
        }
        bs_gpu_release(sched);
    }

    pthread_mutex_lock(&sched->mu);
    if (rc == BS_SESSION_OK) {
        memcpy(sl->hist, tokens, (size_t)n_tok * sizeof(uint32_t));
        sl->hist_n = n_tok;
        /* An imported history carries no think-elision span (export does not
         * ship one), so the restored session's next turn falls back to the
         * exact-prefix rung and establishes a span of its own. */
        bs_slot_clear_projections(sl);
        snprintf(sl->sid, sizeof(sl->sid), "%s", sid);
        sl->armed_sid[0] = '\0';
    } else {
        /* The state may have been partially written — do not advertise it. */
        sl->sid[0] = '\0';
        sl->hist_n = 0;
        bs_slot_clear_projections(sl);
    }
    bs_slot_unreserve_locked(sched, sl);
    pthread_mutex_unlock(&sched->mu);

    /* The restored image is the new truth for this session; a boundary
     * snapshot captured under its old history must not out-rank it. */
    if (rc == BS_SESSION_OK && sl->sys_sid[0] &&
        strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0)
        bs_sys_snap_drop(sched, sl);

    if (rc == BS_SESSION_OK)
        fprintf(stderr, "[batch_sched] session restored slot=%d sid=%s: "
                "%d tokens\n", idx, sid, n_tok);
    return rc;
}

/* Shared body of bs_session_release / _by_prefix.  `n` == 0 means "exact
 * match on the whole id"; n > 0 means "match the first n characters". */
static void bs_session_release_match(pn_batch_sched *sched, const char *key,
                                     size_t n)
{
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        pthread_mutex_lock(&sched->mu);
        int mine = (sl->status == BS_SLOT_IDLE && sl->sid[0] &&
                    (n ? strncmp(sl->sid, key, n) == 0
                       : strncmp(sl->sid, key, sizeof(sl->sid)) == 0));
        int snap_mine = (sl->status == BS_SLOT_IDLE && sl->sys_sid[0] &&
                         (n ? strncmp(sl->sys_sid, key, n) == 0
                            : strncmp(sl->sys_sid, key,
                                      sizeof(sl->sys_sid)) == 0));
        if (mine) {
            sl->sid[0] = '\0';
            sl->hist_n = 0;
            bs_slot_clear_projections(sl);
            sl->armed_sid[0] = '\0';
        }
        pthread_mutex_unlock(&sched->mu);
        /* bs_sys_snap_drop takes mu itself for the sys_sid clear. */
        if (snap_mine)
            bs_sys_snap_drop(sched, sl);
        if (mine || snap_mine)
            fprintf(stderr, "[batch_sched] session released slot=%d key=%s\n",
                    i, key);
    }
}

void bs_session_release(pn_batch_sched *sched, const char *sid)
{
    if (!sched || !sid || !sid[0])
        return;
    bs_session_release_match(sched, sid, 0);
}

void bs_session_release_by_prefix(pn_batch_sched *sched,
                                  const char *base_prefix)
{
    if (!sched || !base_prefix || !base_prefix[0])
        return;
    bs_session_release_match(sched, base_prefix, strlen(base_prefix));
}

void bs_prefill_yield(pn_batch_sched *sched, int idx)
{
#ifdef USE_CUDA
    if (!sched || idx < 0 || idx >= sched->cap)
        return;

    /* Only worth a handoff when the driver actually has a step to run
     * (gate open: >=1 active slot, tokens ready).  Racing a token post /
     * release here is harmless — worst case one missed (or one empty)
     * interleave window. */
    pthread_mutex_lock(&sched->mu);
    int runnable = gate_ok(sched);
    pthread_mutex_unlock(&sched->mu);
    if (!runnable)
        return;                      /* keep the turn; nothing to interleave */

    /* Phase 9b: a pipelined prefill chunk deliberately returns with die B's
     * segment still in flight.  This is the point where the GPU turn is handed
     * to somebody else, so drain both dies FIRST — explicitly, rather than
     * leaning on state_to_host's per-layer syncs, which skip any layer whose
     * GPU buffer is not currently this state's image and are therefore not a
     * guaranteed full drain.  ~1 µs on an idle stream, only on a real
     * handover, and a no-op when the pipeline is off. */
    pn_qwen35moe_prefill_drain_cuda(sched->model);

    /* Make this slot's host mirror authoritative BEFORE letting any other
     * forward at the GPU (the prefix-cache eviction protocol): the next
     * chunk then re-seeds from the mirror via the ordinary
     * seed-on-host-change path even if an interleaved forward claimed the
     * live SSM buffers / shared KV ring.  On failure keep the turn —
     * resuming without a valid mirror would be unsafe, and keeping it is
     * always safe (just no interleave). */
    if (pn_qwen35moe_gpu_state_to_host(sched->model,
                                       sched->slots[idx].state) != 0) {
        fprintf(stderr, "[batch_sched] prefill_yield: state_to_host failed "
                "(slot=%d); continuing without yield\n", idx);
        return;
    }

    pthread_mutex_lock(&sched->mu);
    bs_gpu_release_locked(sched);
    pthread_cond_signal(&sched->cv_work);  /* the driver may be parked */
    /* Re-acquire FIFO: the driver (and any earlier waiter) goes first —
     * this is the chunk → step → chunk alternation. */
    bs_gpu_acquire_locked(sched);
    pthread_mutex_unlock(&sched->mu);
#else
    (void)sched; (void)idx;
#endif
}

void bs_prefill_abort(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    bs_gpu_release(sched);
    bs_slot_release(sched, idx);
}

void bs_prefill_abort_reset(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    /* Still owning the GPU turn here (the range helper re-acquires after every
     * yield), which is bs_slot_reset_state's precondition. */
    bs_slot_reset_state(sched, idx);
    bs_gpu_release(sched);
    bs_slot_release(sched, idx);
}

/* Shared bind body.  `sp == NULL` = legacy (owner-side sampling via the
 * logits handshake); `sp != NULL` = L3 (arm the driver's per-slot sampler:
 * copy the params + repeat window, allocate the output ring sized to the
 * full budget so it can never overflow). */
static int bs_slot_bind_common(pn_batch_sched *sched, int idx,
                               uint32_t first_token,
                               const pn_sampling_params *sp,
                               const uint32_t *recent, int n_recent,
                               int max_new,
                               const char *sid, const uint32_t *prompt,
                               int n_prompt)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return -1;

#ifdef USE_CUDA
    uint32_t *rec_copy = NULL;
    uint32_t *ring     = NULL;
    int       max_recent = 0;

    /* PC-batched: cache this turn only when a session id AND the full prompt
     * are supplied AND driver-sampling is on (the committed log is driver-fed;
     * legacy lockstep is left non-caching).  Pre-size the committed-token log
     * to prompt + every possibly-stepped decode token so the driver never
     * reallocs it mid-step under mu. */
    int   cache_turn = (sp && sid && sid[0] && prompt && n_prompt > 0);
    uint32_t *ct_buf = NULL;
    int   ct_cap     = 0;

    if (sp) {
        max_recent = sp->repeat_last_n > 0 ? sp->repeat_last_n : 0;
        if (n_recent > max_recent)
            n_recent = max_recent;   /* defensive; caller passes the window */
        if (max_recent > 0) {
            rec_copy = malloc((size_t)max_recent * sizeof(uint32_t));
            if (!rec_copy)
                goto fail;
            if (n_recent > 0)
                memcpy(rec_copy, recent, (size_t)n_recent * sizeof(uint32_t));
        }
        if (max_new > 0) {
            ring = malloc((size_t)max_new * sizeof(uint32_t));
            if (!ring)
                goto fail;
        }
    }
    /* THINK-ELISION: a turn that resumed by eliding the previous think block
     * committed hist[0, ct_keep) followed by prompt[ct_prompt_from ..) — NOT
     * the prompt as given.  Seed the log to match the KV, or the ct_n ==
     * state->pos invariant finalize checks fails and the turn goes uncached.
     * Both fields are 0 on every other path, which reduces to the plain copy. */
    int ct_keep = sched->slots[idx].ct_keep;
    int ct_from = sched->slots[idx].ct_prompt_from;
    int ct_seed_n = ct_keep + (n_prompt - ct_from);
    if (cache_turn) {
        ct_cap = ct_seed_n + (max_new > 0 ? max_new : 0);
        ct_buf = malloc((size_t)ct_cap * sizeof(uint32_t));
        if (!ct_buf)
            goto fail;   /* rec_copy/ring freed at fail */
        if (ct_keep > 0)
            memcpy(ct_buf, sched->slots[idx].hist,
                   (size_t)ct_keep * sizeof(uint32_t));
        memcpy(ct_buf + ct_keep, prompt + ct_from,
               (size_t)(n_prompt - ct_from) * sizeof(uint32_t));
    }

    /* The GPU turn is owned (bs_prefill_begin) — this whole section is
     * atomic w.r.t. every other forward on this model. */

    /* The batched decode seeds each slot from this state, and a GPU prefill
     * leaves the host mirror stale (B2).  Refresh the B3 SSM host copy now,
     * while the live GPU state is still this request's (the slot SSM seed
     * in decode_slot_bind reads the host mirror).  Phase 10b lazy-mirror
     * mode skips the KV half: the prefill rows are in the single-user GPU
     * ring and the first slots step D2D-seeds the slot ring from it — no
     * host round-trip.  Eager mode keeps the full legacy capture. */
    if (pn_qwen35moe_host_kv_mirror_eager()) {
        if (pn_qwen35moe_gpu_state_to_host(sched->model,
                                           sched->slots[idx].state) != 0) {
            fprintf(stderr, "[batch_sched] state_to_host failed (slot=%d)\n",
                    idx);
            goto fail;
        }
    } else if (pn_qwen35moe_gpu_state_ssm_to_host(sched->model,
                                                  sched->slots[idx].state)
               != 0) {
        fprintf(stderr, "[batch_sched] ssm_to_host failed (slot=%d)\n", idx);
        goto fail;
    }

    /* Lazy lane registration on the first bind ever (serialized by the GPU
     * turn; runs at most once).  Registering lane idx here uses this state's
     * now-authoritative host mirror, and re-binding it below is harmless. */
    if (!sched->slots_gpu_init) {
        if (bs_lazy_init_gpu(sched) != 0)
            goto fail;
    }

    /* Pick a FREE GPU slot index for this joiner: the lowest GPU index not
     * currently claimed by ANY active slot's gpu_idx.  This is the
     * continuous-batching fix — binding at the SCHEDULER slot index (the old
     * code) could, after a compaction diverged gpu-index from sched-index,
     * land on top of an in-flight decoder and corrupt it (decode_slots then
     * failed its slot_state validation).
     *
     * Scanning claimed gpu_idx (not n_active_gpu) is required for the
     * CONCURRENT-START case: several joiners bind back-to-back, each under
     * the GPU turn, BEFORE the driver rebuilds — so n_active_gpu is still
     * stale and would hand them all the same index.  Every active slot's
     * gpu_idx is already set at its own bind, so the scan sees them.  Read
     * under mu (the driver and other binds write gpu_idx under mu). */
    pthread_mutex_lock(&sched->mu);
    bool claimed[BS_MAX_SLOTS] = { false };
    for (int i = 0; i < sched->cap; i++) {
        bs_slot *o = &sched->slots[i];
        if (i != idx && slot_is_active(o) && o->gpu_idx >= 0 &&
            o->gpu_idx < sched->cap)
            claimed[o->gpu_idx] = true;
    }
    /* Scan from the TOP down for the highest free GPU index.  Binding high
     * tends to keep a joiner's gpu_idx >= its compacted rank, so the common
     * case is a downward compaction; but this is only a hint now, not a
     * correctness requirement — rebuild_gpu_map realizes an arbitrary
     * permutation, so the >=5-concurrent-joiners layout that forces an
     * upward move / cycle is handled correctly rather than refused. */
    int gpu_slot = -1;
    for (int g = sched->cap - 1; g >= 0; g--)
        if (!claimed[g]) { gpu_slot = g; break; }
    pthread_mutex_unlock(&sched->mu);
    if (gpu_slot < 0) {
        /* Every GPU index claimed — cannot happen (at most `cap` slots exist
         * and this one is not yet active), but fail cleanly if it does. */
        fprintf(stderr, "[batch_sched] bind: no free GPU slot (slot=%d)\n",
                idx);
        goto fail;
    }

    /* Snapshot this request's post-prefill SSM state into GPU slot gpu_slot.
     * cap+1 slots are bound (top index reserved as rebuild scratch). */
    if (pn_qwen35moe_decode_slot_bind(sched->model,
                                      sched->slots[idx].state,
                                      gpu_slot, sched->cap + 1) != 0) {
        fprintf(stderr, "[batch_sched] slot_bind failed (slot=%d gpu=%d)\n",
                idx, gpu_slot);
        goto fail;
    }

    pthread_mutex_lock(&sched->mu);
    {
        bs_slot *sl = &sched->slots[idx];
        sl->gpu_idx      = gpu_slot;
        sl->next_token   = first_token;
        sl->token_ready  = false;   /* driver reads next_token directly on BOUND */
        sl->logits_ready = false;
        sl->logits       = NULL;
        sl->step_error   = false;
        if (sp) {
            sl->sp         = *sp;
            sl->recent     = rec_copy;
            sl->n_recent   = n_recent;
            sl->max_recent = max_recent;
            sl->ring       = ring;
            sl->ring_cap   = max_new > 0 ? max_new : 0;
            sl->ring_r     = 0;
            sl->ring_w     = 0;
            sl->budget     = max_new;
            /* max_tokens == 1: nothing for the driver to step. */
            sl->finished   = (max_new <= 0);
        } else {
            sl->recent     = NULL;
            sl->n_recent   = 0;
            sl->max_recent = 0;
            sl->ring       = NULL;
            sl->ring_cap   = 0;
            sl->ring_r     = 0;
            sl->ring_w     = 0;
            sl->budget     = 0;
            sl->finished   = false;
        }
        /* PC-batched: arm the committed-token log so the driver records every
         * stepped token.  ct is seeded with what the KV actually holds (== the
         * prompt, except on a think-elided resume — see ct_keep above), so the
         * invariant ct_n == state->pos holds from here.  The old ct buffer
         * (from this slot's previous request) is replaced. */
        if (cache_turn) {
            free(sl->ct);
            sl->ct     = ct_buf;
            sl->ct_cap = ct_cap;
            sl->ct_n   = ct_seed_n;
            snprintf(sl->armed_sid, sizeof(sl->armed_sid), "%s", sid);
        } else {
            sl->armed_sid[0] = '\0';   /* not caching this turn */
            sl->ct_n = 0;
        }
        sl->pend_vis_n = 0;
        sl->pend_vis_think_n = 0;
        sl->status       = BS_SLOT_BOUND;
    }
    /* A new user is joining the active set — the driver must rebuild the
     * compacted GPU map before the next step to include this slot. */
    sched->active_set_changed = true;
    pthread_cond_signal(&sched->cv_work);   /* gate may open */
    bs_gpu_release_locked(sched);
    pthread_mutex_unlock(&sched->mu);
    return 0;

fail:
    free(rec_copy);
    free(ring);
    free(ct_buf);
    bs_gpu_release(sched);
    fprintf(stderr, "[batch_sched] bind failed; slot %d released, "
            "request degrades to an error\n", idx);
    bs_slot_release(sched, idx);
    return -1;
#else
    /* Non-CUDA build: sched is always NULL so this is never reached. */
    (void)first_token; (void)sp; (void)recent; (void)n_recent; (void)max_new;
    (void)sid; (void)prompt; (void)n_prompt;
    return -1;
#endif
}

int bs_slot_bind(pn_batch_sched *sched, int idx, uint32_t first_token)
{
    return bs_slot_bind_common(sched, idx, first_token, NULL, NULL, 0, 0,
                               NULL, NULL, 0);
}

int bs_slot_bind_ex(pn_batch_sched *sched, int idx, uint32_t first_token,
                    const pn_sampling_params *sp,
                    const uint32_t *recent, int n_recent, int max_new,
                    const char *sid, const uint32_t *prompt, int n_prompt)
{
    if (!sp)
        return -1;
    return bs_slot_bind_common(sched, idx, first_token, sp, recent, n_recent,
                               max_new, sid, prompt, n_prompt);
}

void bs_slot_arm_reasoning_budget(pn_batch_sched *sched, int idx,
                                  int reason_budget, bool in_think,
                                  const uint32_t *close_ids, int n_close,
                                  uint32_t first_token)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    if (!sched->driver_sample)
        return;   /* legacy lockstep path caps in the owner loop, not here */
    int cap = (int)(sizeof(((bs_slot *)0)->reason_close) /
                    sizeof(((bs_slot *)0)->reason_close[0]));
    if (n_close > cap) n_close = cap;

    pthread_mutex_lock(&sched->mu);
    bs_slot *sl = &sched->slots[idx];
    sl->reason_budget   = reason_budget > 0 ? reason_budget : 0;
    sl->reason_in_think = (sl->reason_budget > 0) && in_think && n_close > 0;
    sl->reason_toks     = 0;
    sl->reason_forced   = false;
    sl->reason_forced_call = false;
    sl->reason_close_n  = (sl->reason_budget > 0) ? n_close : 0;
    sl->reason_close_i  = sl->reason_close_n;   /* nothing queued yet */
    for (int i = 0; i < sl->reason_close_n && close_ids; i++)
        sl->reason_close[i] = close_ids[i];
    /* The owner holds token #1 (first_token) and will em_feed it itself; count
     * it here iff it was sampled inside <think>, so the driver's per-step count
     * stays aligned with the single-user loop (which counts every <think> token
     * including the first).  A first_token that is itself the "</think>" close
     * means the model concluded immediately — leave <think>. */
    if (sl->reason_in_think) {
        sl->reason_toks = 1;
        if (sl->reason_close_n == 1 && first_token == sl->reason_close[0])
            sl->reason_in_think = false;
        else if (sl->reason_budget > 0 && sl->reason_toks >= sl->reason_budget) {
            /* Degenerate budget of 1: trip on the first token. */
            sl->reason_forced  = true;
            sl->reason_close_i = 0;
        }
    }
    pthread_mutex_unlock(&sched->mu);
}

/* Wait for logits from the driver (shared between await_first and step_wait).
 * Called with sched->mu held; returns with it released. */
static int slot_wait_logits(pn_batch_sched *sched, bs_slot *sl,
                            float **logits_out)
{
    while (!sl->logits_ready && !sched->shutdown)
        pthread_cond_wait(&sl->cv, &sched->mu);

    if (sched->shutdown || !sl->logits) {
        pthread_mutex_unlock(&sched->mu);
        return -1;
    }

    *logits_out      = sl->logits;
    sl->logits_ready = false;
    pthread_mutex_unlock(&sched->mu);
    return 0;
}

int bs_slot_await_first(pn_batch_sched *sched, int idx, float **logits_out)
{
    if (!sched || idx < 0 || idx >= sched->cap || !logits_out)
        return -1;
    pthread_mutex_lock(&sched->mu);
    /* The first token was already deposited via bs_slot_bind; just wait. */
    return slot_wait_logits(sched, &sched->slots[idx], logits_out);
}

int bs_slot_step_wait(pn_batch_sched *sched, int idx,
                      uint32_t next_token, float **logits_out)
{
    if (!sched || idx < 0 || idx >= sched->cap || !logits_out)
        return -1;

    pthread_mutex_lock(&sched->mu);

    bs_slot *sl = &sched->slots[idx];

    /* Deposit this step's token for the driver, then wait for its logits. */
    sl->next_token  = next_token;
    sl->token_ready = true;
    pthread_cond_signal(&sched->cv_work);

    return slot_wait_logits(sched, sl, logits_out);
}

int bs_slot_pop_token(pn_batch_sched *sched, int idx, uint32_t *tok)
{
    if (!sched || idx < 0 || idx >= sched->cap || !tok)
        return -1;
    pthread_mutex_lock(&sched->mu);
    bs_slot *sl = &sched->slots[idx];
    for (;;) {
        /* Drain order matters: queued tokens first (an EOS the driver
         * pushed must reach the owner), then error, then clean finish. */
        if (sl->ring && sl->ring_r < sl->ring_w) {
            *tok = sl->ring[sl->ring_r++];
            pthread_mutex_unlock(&sched->mu);
            return 1;
        }
        if (sched->shutdown || sl->step_error) {
            pthread_mutex_unlock(&sched->mu);
            return -1;
        }
        if (sl->finished) {
            pthread_mutex_unlock(&sched->mu);
            return 0;
        }
        pthread_cond_wait(&sl->cv, &sched->mu);
    }
}

bool bs_driver_sampling(const pn_batch_sched *sched)
{
    return sched && sched->driver_sample;
}

void bs_slot_release(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    pthread_mutex_lock(&sched->mu);
    bs_slot *sl = &sched->slots[idx];
    /* Only the owner calls release, and it does so exactly once, so any
     * non-IDLE status here belongs to this owner: free the slot directly.
     * The driver never retires slots on its own — a mid-step release just
     * means this lane's logits are discarded at deposit time.  (The driver
     * only touches the L3 ring/recent under mu with status == DECODE, so
     * freeing them here cannot race a deposit.) */
    bool was_active = (sl->status == BS_SLOT_BOUND ||
                       sl->status == BS_SLOT_DECODE);
    if (sl->status != BS_SLOT_IDLE) {
        sl->status       = BS_SLOT_IDLE;
        sl->logits       = NULL;
        sl->logits_ready = false;
        sl->token_ready  = false;
        free(sl->recent);
        sl->recent     = NULL;
        sl->n_recent   = 0;
        sl->max_recent = 0;
        free(sl->ring);
        sl->ring       = NULL;
        sl->ring_cap   = 0;
        sl->ring_r     = 0;
        sl->ring_w     = 0;
        sl->budget     = 0;
        sl->finished   = false;
        sl->step_error = false;
        /* OBSERVABILITY — a reasoning force-close is an ABNORMAL event, so it
         * logs unconditionally (not behind a verbose flag).  This is the BATCHED
         * path, which is what the live fleet runs, so this is the line that
         * actually answers "how often is this happening, and to which model".
         * It is the one condition under which the grammar compels the tool-call
         * opener while the model is still mid-thought — how a half-written call
         * ends up inside a parameter VALUE (tool_calling.c FRAME_MARKERS).
         * Logged from release (not from the injection point) so the OUTCOME —
         * whether a tool call actually closed afterwards — is on the same line.
         * Held under sched->mu, like every other slot-field read here. */
        if (sl->reason_forced)
            fprintf(stderr, "[reason_budget] FORCED close model=%s path=batched "
                    "slot=%d sid=%s reason_toks=%d/%d tool_call=%s\n",
                    sched->model_name[0] ? sched->model_name : "?",
                    idx, sl->sid[0] ? sl->sid : "-",
                    sl->reason_toks, sl->reason_budget,
                    sl->reason_forced_call ? "yes" : "no");
        /* Clear the AUTO reasoning-budget arm so a reused slot starts clean. */
        sl->reason_budget   = 0;
        sl->reason_toks     = 0;
        sl->reason_in_think = false;
        sl->reason_forced   = false;
        sl->reason_forced_call = false;
        sl->reason_close_n  = 0;
        sl->reason_close_i  = 0;
        /* PC-batched: clear the per-turn caching arm.  A successful
         * bs_slot_finalize already consumed armed_sid (and published sid/hist);
         * clearing here covers the error/no-finalize path so a stale arm can't
         * mislead the next request that reuses this slot.  sid/hist (the warm
         * state) are intentionally preserved for session-affinity reuse. */
        sl->armed_sid[0] = '\0';
        /* Drop our GPU index.  The GPU slot_state[gpu_idx] entry is left as-is
         * (harmless): the released slot is excluded from every future
         * compacted step (target_sched only lists active slots), and the
         * rebuild's compaction overwrites this vacated GPU index when a
         * higher-ranked active slot moves down onto it.  Clearing GPU state
         * here would race the driver's turn-held decode_slots read. */
        sl->gpu_idx    = -1;
    }
    /* A user left the active set — the driver must rebuild the compacted GPU
     * map before the next step so the vacated GPU slot is not stepped. */
    if (was_active)
        sched->active_set_changed = true;
    pthread_cond_signal(&sched->cv_work);    /* gate may open (or reshape) */
    pthread_cond_broadcast(&sched->cv_idle); /* a slot is free — wake D1 waiters */
    pthread_mutex_unlock(&sched->mu);
}

/* ============================================================
 * D1 + D3: bounded overflow queue + admission-as-backpressure
 * ============================================================ */

/* Compute absolute deadline for pthread_cond_timedwait from a relative
 * wait_ms value.  Fills *ts with CLOCK_REALTIME deadline. */
static void make_deadline(struct timespec *ts, int wait_ms)
{
    clock_gettime(CLOCK_REALTIME, ts);
    long ns = ts->tv_nsec + (long)wait_ms * 1000000L;
    ts->tv_sec  += ns / 1000000000L;
    ts->tv_nsec  = ns % 1000000000L;
}

bs_acquire_result bs_slot_acquire_or_wait(pn_batch_sched *sched,
                                          int wait_ms,
                                          int *slot_out,
                                          int *retry_after_s,
                                          const char *sid)
{
    if (slot_out)      *slot_out      = -1;
    if (retry_after_s) *retry_after_s = 0;
    if (!sched)        return BS_ACQUIRE_429;

    pthread_mutex_lock(&sched->mu);

    if (sched->shutdown) {
        pthread_mutex_unlock(&sched->mu);
        return BS_ACQUIRE_SHUTDOWN;
    }

    /* Fast path: slot free and pressure OK — claim immediately. */
    if (sched->pressure_level < BS_PRESSURE_HOLD_LEVEL) {
        int s = try_claim_idle(sched, sid);
        if (s >= 0) {
            pthread_mutex_unlock(&sched->mu);
            if (slot_out) *slot_out = s;
            return BS_ACQUIRE_OK;
        }
    }

    /* Slow path: must queue.
     *
     * Admission decision:
     *   - If queue is full + pressure CRITICAL → 503 (last resort, no wait).
     *   - If queue is full (but pressure not CRITICAL) → 429.
     *   - Otherwise enqueue and wait. */
    if (sched->n_queued >= sched->queue_cap) {
        int level = sched->pressure_level;
        pthread_mutex_unlock(&sched->mu);
        if (retry_after_s) *retry_after_s = 5;
        /* 3 = PN_MEM_LEVEL_CRITICAL */
        return (level >= 3) ? BS_ACQUIRE_503 : BS_ACQUIRE_429;
    }

    /* Enqueue this waiter. */
    sched->n_queued++;

    /* Compute absolute deadline. */
    struct timespec deadline;
    make_deadline(&deadline, wait_ms > 0 ? wait_ms : 0);

    bs_acquire_result result = BS_ACQUIRE_TIMEOUT;

    for (;;) {
        if (sched->shutdown) {
            result = BS_ACQUIRE_SHUTDOWN;
            break;
        }

        /* D3: if pressure is TIGHT/CRITICAL, stay queued — don't admit a new
         * prefill that could OOM.  But if CRITICAL and queue is now full
         * (more callers arrived while we waited), fail with 503. */
        if (sched->pressure_level >= BS_PRESSURE_HOLD_LEVEL) {
            /* If queue is now at its cap and we're still stuck at the back
             * due to pressure, we've been holding a queue slot ourselves —
             * count that as fair (we're already in); just keep waiting until
             * pressure clears or deadline. */
            if (wait_ms <= 0) {
                /* Non-blocking mode: don't wait. */
                result = (sched->pressure_level >= 3) ? BS_ACQUIRE_503
                                                      : BS_ACQUIRE_429;
                break;
            }
            /* Timed wait for pressure to clear or a slot to open. */
            int rc = pthread_cond_timedwait(&sched->cv_idle, &sched->mu,
                                            &deadline);
            if (rc != 0) {
                /* Timeout. */
                if (retry_after_s) *retry_after_s = 2;
                result = BS_ACQUIRE_TIMEOUT;
                break;
            }
            continue;   /* re-check all conditions */
        }

        /* Pressure OK: try to claim a slot. */
        int s = try_claim_idle(sched, sid);
        if (s >= 0) {
            result = BS_ACQUIRE_OK;
            sched->n_queued--;
            pthread_mutex_unlock(&sched->mu);
            if (slot_out) *slot_out = s;
            return BS_ACQUIRE_OK;
        }

        /* No slot free yet. */
        if (wait_ms <= 0) {
            result = BS_ACQUIRE_429;
            break;
        }

        int rc = pthread_cond_timedwait(&sched->cv_idle, &sched->mu,
                                        &deadline);
        if (rc != 0) {
            if (retry_after_s) *retry_after_s = 2;
            result = BS_ACQUIRE_TIMEOUT;
            break;
        }
        /* Woke up — re-check all conditions from the top. */
    }

    /* Leaving the queue without a slot. */
    sched->n_queued--;
    pthread_mutex_unlock(&sched->mu);
    return result;
}

void bs_notify_pressure(pn_batch_sched *sched, int level)
{
    if (!sched) return;
    pthread_mutex_lock(&sched->mu);
    int old = sched->pressure_level;
    sched->pressure_level = level;
    /* Wake D1 waiters whenever pressure drops (they might now be able to
     * proceed) or whenever it hits CRITICAL (so they can fail fast). */
    if (level < old || level >= 3)
        pthread_cond_broadcast(&sched->cv_idle);
    pthread_mutex_unlock(&sched->mu);
}

void bs_get_load(pn_batch_sched *sched, bs_load *out)
{
    if (!out) return;

    /* Single-user (non-batched) path: no scheduler. That path serializes on the
     * per-model gen_lock and 429s a second concurrent chat, so it presents as a
     * single always-free slot for dispatch. Exact instantaneous busyness there
     * is minor (a second request is refused, not queued). */
    if (!sched) {
        out->capacity     = 1;
        out->active_slots = 0;
        out->queue_depth  = 0;
        out->free_slots   = 1;
        return;
    }

    pthread_mutex_lock(&sched->mu);
    int busy = 0;
    for (int i = 0; i < sched->cap; i++)
        if (sched->slots[i].status != BS_SLOT_IDLE) busy++;  /* PREFILL/BOUND/DECODE */
    out->capacity     = sched->cap;
    out->active_slots = busy;
    out->free_slots   = sched->cap - busy;
    out->queue_depth  = sched->n_queued;   /* D1 overflow-queue waiters */
    pthread_mutex_unlock(&sched->mu);
}
