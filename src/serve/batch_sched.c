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
 * @issues Each slot owns a fixed block on the card (GPU row = slot index: a
 *         ctx_cap KV ring row per attention layer + an SSM image per SSM
 *         layer), carved at bs_create; a model whose blocks do not fit is
 *         refused at load. The step runs the active rows through a row map,
 *         so nothing moves when conversations take turns, and a resumed turn
 *         re-seeds from its block device to device — no host RAM. Until
 *         2026-10-21 the active set was compacted into rows [0, n_active) and
 *         sessions taking STRICT turns overwrote each other's rows (K80,
 *         qwen36-stock, A ~10k / B 10k -> 29k: A2 128 s, B2 518 s, A3 129 s
 *         TTFT); now A2/A3 resume in ~0.9 s. Block = 394 MiB per die at a
 *         131072 ctx_cap (360 KV + 2.6 q4 shadows + 31.4 SSM).
 *         Remaining per-session host footprint: the SYS1 system-prefix
 *         snapshot (a host copy by design, ~62 MiB SSM + system KV).
 *         The think-included rung resumes on the ids the model SAMPLED, so a
 *         resumed turn can differ from a cold prefill of the re-tokenized
 *         text (test_serve_sched_session_switch: B2/B3, main and here alike).
 * @todo None
 */

#include "batch_sched.h"
#include "../core/host_row_dump.h"
#include "batch_sched_admission.h"
#include "batch_sched_prefill_share.h"
#include "batch_sched_slots.h"
#include "batch_sched_spec.h"
#include "batch_sched_watchdog.h"

#include "request_profile.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "../core/engine.h"          /* gpt2_bytes_decode (diagnostic dumps) */
#include "../core/session_family.h"
#include "../core/session_snapshot.h"
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
static void bs_sys_snap_unref(const pn_batch_sched *sched, bs_slot *sl);
static void make_deadline(struct timespec *ts, int wait_ms);

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

/* Monotonic ms: the driver's step timing, the slot LRU key and
 * the turn owner's hold time. */
static double bs_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
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
    sched->gpu_turn_t0_ms = bs_now_ms();
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

/*
 * Driver design — fixed session rows (2026-10-21):
 *
 * Every scheduler slot owns ONE persistent block on the card for its whole
 * life: GPU row `i` of every attention layer's KV ring and every SSM layer's
 * conv/recurrent image (carved at bs_create). A step runs over whichever
 * slots are active, in any order, by handing pn_qwen35moe_decode_slots_rows
 * the row map (gpu_map[k] = scheduler slot = GPU row). Nothing is compacted,
 * moved or copied when the active set changes, so an idle session's KV + SSM
 * stay exactly where its last turn left them and its next turn resumes from
 * the card. (Until 2026-10-21 the active set was compacted into rows
 * [0, n_active) by D2D moves, which overwrote idle sessions' rows whenever
 * conversations took turns.)
 *
 * Correctness:
 *   - BOUND/DECODE lanes: carry their real token; BOUND → DECODE on first
 *     step.
 *   - PREFILL and IDLE lanes are never in the map (never stepped).
 *   - A slot released mid-step has left DECODE before the driver reads mu
 *     again; its logits are discarded at deposit time.
 *   - A newly-BOUND slot joins the NEXT step after its bind completes (bind
 *     sets active_set_changed and signals cv_work). The bind owns the GPU
 *     turn, so it can never interleave with a step.
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

/* The slot's walk has just been advanced by the token at ring position ring_w: note
 * where forced reasoning first ended, so the owner flips its channel at that token
 * (bs_slot_reasoning_ended).  Under mu. */
void bs_slot_note_reasoning_end(bs_slot *sl)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;

    if (!sl->reason_end_at && tgh && tgh->reasoning_ended && tgh->st &&
        tgh->reasoning_ended(tgh->st))
        __atomic_store_n(&sl->reason_end_at, sl->ring_w, __ATOMIC_RELEASE);
}

bool bs_slot_reasoning_ended(pn_batch_sched *sched, int idx)
{
    static int red = -1;

    if (!sched || idx < 0 || idx >= sched->cap)
        return false;
    int r = __atomic_load_n(&red, __ATOMIC_RELAXED);
    if (r < 0) {
        const char *e = getenv("PN_THINK_FLIP_RED");
        r = e && e[0] == '1';
        __atomic_store_n(&red, r, __ATOMIC_RELAXED);
    }
    const bs_slot *sl = &sched->slots[idx];
    const int at = __atomic_load_n(&sl->reason_end_at, __ATOMIC_ACQUIRE);
    return at > 0 && (r || sl->ring_r >= at);
}

#ifndef USE_CUDA
/* No driver without CUDA: the ring push and the reasoning-end note of
 * driver_emit_token, for the host tests. */
int bs_slot_deposit_token(pn_batch_sched *sched, bs_slot *sl, uint32_t next)
{
    (void)sched;
    if (!sl->ring || sl->ring_w >= sl->ring_cap)
        return -1;
    sl->ring[sl->ring_w++] = next;
    sl->budget--;
    bs_slot_note_reasoning_end(sl);
    return 0;
}
#endif

#ifdef USE_CUDA

/* A slot the driver marked `finished` (EOS / budget, L3) is inactive: it drops
 * out of the step exactly like a released slot. */
static int slot_is_active(const bs_slot *sl)
{
    return (sl->status == BS_SLOT_BOUND || sl->status == BS_SLOT_DECODE) &&
           !sl->finished;
}

/* Rebuild the list of active slots the next step runs (mu held). Each active
 * slot decodes in its own fixed GPU row (= its slot index), so this is pure
 * bookkeeping — no device work, nothing moves. Returns the active count. */
static int rebuild_gpu_map(pn_batch_sched *sched)
{
    int n_active = 0;
    for (int i = 0; i < sched->cap; i++)
        if (slot_is_active(&sched->slots[i]))
            sched->gpu_map[n_active++] = i;
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

/* Whether token `tok` ends a line, for the reasoning close window.  The
 * grammar hook's text when the request has one (no allocation), else the
 * tokenizer's decode (GPT-2 byte-level pieces mapped back to bytes). */
static bool driver_token_ends_line(pn_batch_sched *sched, const pn_tool_grammar_hook *tgh,
                                   uint32_t tok)
{
    if (tgh && tgh->token_text) {
        const char *txt = tgh->token_text(tgh->user, tok);
        if (txt)
            return pn_reasoning_piece_ends_line(txt);
    }
    if (!sched || !sched->model || !sched->model->tokenizer)
        return false;
    int id = (int)tok;
    char *raw = ft_decode(sched->model->tokenizer, &id, 1);
    if (!raw)
        return false;
    char *decoded = ft_get_model_type(sched->model->tokenizer) == 0
                        ? gpt2_bytes_decode(raw) : NULL;
    bool ends = pn_reasoning_piece_ends_line(decoded ? decoded : raw);
    free(decoded);
    free(raw);
    return ends;
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
static uint32_t driver_reason_step(pn_batch_sched *sched, bs_slot *sl, uint32_t sampled,
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
             * hook (or a dialect without a window) → pre-fix behavior.  A
             * "required" walk takes no window (tg_reasoning_end_injected). */
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
    /* At the cap, or on the first line end inside the window just before it
     * (reasoning_budget.h), so the thought ends on a whole line.  The token is
     * only decoded inside that window. */
    bool line_end = sl->reason_budget > 0 && !sl->reason_forced &&
                    sl->reason_toks >= sl->reason_soft &&
                    driver_token_ends_line(sched, tgh, sampled);
    if (sl->reason_budget > 0 && !sl->reason_forced &&
        pn_reasoning_should_close(&(pn_reasoning_plan){ sl->reason_budget, sl->reason_soft },
                                  sl->reason_toks, line_end) &&
        !in_call) {
        sl->reason_forced = true;
        sl->reason_close_i = 0;   /* arm the injection for the next step(s) */
    }
    return sampled;
}

/* A token just stepped INTO the slot's state: PC-batched, append it to the committed-token
 * log so ct_n == state->pos stays invariant (ct is pre-sized to prompt + budget at bind; on
 * the impossible overflow, disarm: finalize then no-ops and the session re-prefills). */
static void driver_log_committed(bs_slot *sl, uint32_t tok)
{
    if (!sl->armed_sid[0])
        return;
    if (sl->ct && sl->ct_n < sl->ct_cap)
        sl->ct[sl->ct_n++] = tok;
    else
        sl->armed_sid[0] = '\0';
}

/* L3: a stepped token joins the slot's repeat window before the next draw. */
static void driver_window_push(bs_slot *sl, uint32_t tok)
{
    if (sl->max_recent <= 0 || !sl->recent)
        return;
    if (sl->n_recent < sl->max_recent) {
        sl->recent[sl->n_recent++] = tok;
    } else {
        memmove(sl->recent, sl->recent + 1, (size_t)(sl->max_recent - 1) * sizeof(uint32_t));
        sl->recent[sl->max_recent - 1] = tok;
    }
}

/* L3: push the token the driver chose for this slot to its output ring and decide whether
 * the slot is finished (EOS / <|im_end|> / budget / a completed tool call). Under mu.
 * Returns 1 when finished, 0 when `next` is the next step's input, -1 on a full ring. */
static int driver_emit_token(pn_batch_sched *sched, bs_slot *sl, uint32_t next)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;

    if (!sl->ring || sl->ring_w >= sl->ring_cap)
        return -1;
    sl->ring[sl->ring_w++] = next;
    sl->budget--;
    bs_slot_note_reasoning_end(sl);
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
        return 1;
    }
    sl->next_token  = next;
    sl->token_ready = true;
    return 0;
}

int bs_slot_deposit_token(pn_batch_sched *sched, bs_slot *sl, uint32_t next)
{
    return driver_emit_token(sched, sl, next);
}

#ifdef USE_CUDA
/* Deposit one MTP cycle's tokens exactly as successive plain steps would have: the seed
 * and every accepted draft were stepped into the state (committed: logged and windowed),
 * each emitted token goes through the reasoning-budget accounting and the ring. Under mu. */
static void driver_spec_deposit(pn_batch_sched *sched, bs_slot *sl, uint32_t seed,
                                const uint32_t *emit, int ne)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;

    driver_log_committed(sl, seed);
    driver_window_push(sl, seed);
    for (int j = 0; j < ne; j++) {
        bool injected = false;
        /* The draws ran hook-free: advance the live walk as the sampler would have. */
        if (tgh && tgh->advance && tgh->st && tgh->token_text) {
            const char *text = tgh->token_text(tgh->user, emit[j]);
            if (text)
                tgh->advance(tgh->st, text);
        }
        uint32_t next = driver_reason_step(sched, sl, emit[j], tgh, &injected);
        driver_arm_full_logits(sched, sl, 0);
        int rc = driver_emit_token(sched, sl, next);
        if (rc < 0) {
            driver_error_slot(sched, sl);
            return;
        }
        if (rc > 0)
            break;
        if (j < ne - 1) {
            driver_log_committed(sl, next);
            driver_window_push(sl, next);
        }
    }
    pthread_cond_signal(&sl->cv);
}

/* The solo-slot step on an MTP model (mu held, GPU turn owned): one exact MTP cycle in
 * place of a decode-slots step. Returns true when it handled the step (deposited, or
 * errored the slot), false when the slot is not eligible (the caller steps it plain). */
static bool driver_spec_step(pn_batch_sched *sched, int i, uint32_t seed)
{
    bs_slot *sl = &sched->slots[i];
    uint32_t emit[PN_SPEC_DEPTH_MAX + 1];

    if (!bs_spec_eligible(sched, sl))
        return false;
    const int cap = bs_spec_draft_cap(sl), row = sl->gpu_idx;
    /* No drafts this step (the controller's cost model, or the budget): the slot's
     * block is current, so the plain decode step is the cheaper exact choice. */
    const int depth = sg_mtp_cuda_choose(&sl->mtp, sl->state->pos, cap);
    if (depth < 1 || bs_spec_stress_plain(sl))
        return false;
    const double t_cycle = sg_mtp_cuda_now_ms();
    const int n_recent = bs_spec_stage_window(sl, seed);
    pn_sampling_params sp = sl->sp;   /* the draws advance this copy's RNG */
    pthread_mutex_unlock(&sched->mu);
    int ne = bs_spec_cycle_gpu(sched, sl, row, seed, depth, cap, &sp, n_recent, emit);
    pthread_mutex_lock(&sched->mu);
    if (sl->status != BS_SLOT_DECODE)
        return true;   /* released mid-cycle: nothing to deposit */
    if (ne < 1) {
        fprintf(stderr, "[batch_sched] MTP cycle failed (slot=%d); erroring it out\n", i);
        driver_error_slot(sched, sl);
        return true;
    }
    sl->sp = sp;   /* the sampler state after this cycle's draws (RNG stream) */
    driver_spec_deposit(sched, sl, seed, emit, ne);
    sl->mtp.ms_cycle += sg_mtp_cuda_now_ms() - t_cycle;
    return true;
}
#endif

/* PN_HOST_TIMING (host_stage_timer.h): charge `ns` of `stage` to every slot of the step. */
static void driver_ht_charge(pn_batch_sched *sched, const int *map, int n, pn_ht_stage stage,
                             uint64_t ns)
{
    for (int k = 0; k < n; k++)
        pn_ht_add(&sched->slots[map[k]].host, stage, ns);
}

/* The step's shared stages and launch stamps, charged to each slot it ran (mu held). */
static void driver_ht_step(pn_batch_sched *sched, const int *map, int n, uint64_t gate0,
                           uint64_t acq0, uint64_t acq1, uint64_t launch, uint64_t stepped,
                           uint64_t cpu_ns)
{
    const uint64_t relocked = pn_ht_now();
    const uint64_t run_delay = pn_ht_run_delay();

    driver_ht_charge(sched, map, n, PN_HT_GATE, acq0 - gate0);
    driver_ht_charge(sched, map, n, PN_HT_ACQUIRE, acq1 - acq0);
    driver_ht_charge(sched, map, n, PN_HT_STEP, stepped - launch);
    driver_ht_charge(sched, map, n, PN_HT_RELOCK, relocked - stepped);
    for (int k = 0; k < n; k++) {
        pn_host_timing *h = &sched->slots[map[k]].host;
        if (!h->first_launch_ns) {
            h->first_launch_ns = launch;
            h->run_delay0_ns = run_delay;
        }
        h->last_launch_ns = launch;
        h->last_step_ns = stepped - launch;
        h->step_cpu_ns += cpu_ns;
        h->run_delay_ns = run_delay - h->run_delay0_ns;
        h->steps++;
        h->width_sum += (uint32_t)n;
    }
}

/* End of one slot's deposit. */
static void driver_ht_deposited(bs_slot *sl, uint64_t now)
{
    sl->host.last_deposit_ns = now;
}

/* Per-request active-slot range for the [req] line (mu held). */
static void driver_note_step_width(bs_slot *sl, int width)
{
    if (sl->req_active_min == 0 || width < sl->req_active_min)
        sl->req_active_min = width;
    if (width > sl->req_active_max)
        sl->req_active_max = width;
}

/* Snapshot the next step (mu held, turn owned): refresh the active list if a
 * user joined, left or finished since the last step, then tokens[] / states[]
 * for the active slots, gpu_map_snap[k] = slot = its fixed GPU row. Returns
 * the step width (0: every active slot was released meanwhile). */
static int driver_snapshot_locked(pn_batch_sched *sched, uint32_t *tokens,
                                  pn_state **states, int *gpu_map_snap)
{
    /* ── Refresh the active list if a user joined, left, or finished
     *    since the last step (mu held; bookkeeping only). ── */
    if (sched->active_set_changed)
        (void)rebuild_gpu_map(sched);

    /* ── Snapshot the step (mu held, turn owned): tokens[]/states[] for
     *    the active slots, gpu_map_snap[k] = slot = its fixed GPU row. ── */
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
    return snap_n;
}

/* Deposit a finished step (mu held): rc is the step's result, lo[k] slot
 * gpu_map_snap[k]'s logits, plain_ms the step's time. */
static void driver_deposit_locked(pn_batch_sched *sched, const uint32_t *tokens,
                                  const int *gpu_map_snap, int snap_n,
                                  float *const *lo, int rc, double plain_ms)
{
    (void)plain_ms;   /* read only by the CUDA MTP accounting */
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
        driver_log_committed(sl, tokens[g]);
        pn_ht_bind(&sl->host);
        uint64_t ht = pn_ht_now();
#ifdef USE_CUDA
        if (snap_n == 1 && sl->mtp_on && !sl->pd_stepped)   /* price it (D = 0), feed the head its row */
            bs_spec_after_plain_step(sched, sl, tokens[g], plain_ms);
        ht = pn_ht_lap(PN_HT_MTP_PUSH, ht);
#endif
        if (!sched->driver_sample) {
            sl->logits       = lo[g];
            sl->logits_ready = true;
            pthread_cond_signal(&sl->cv);
            continue;
        }
        /* L3 in-driver sampling. */
        driver_window_push(sl, tokens[g]);
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
            next = driver_reason_step(sched, sl, 0, tgh, &injected);
            ht = pn_ht_lap(PN_HT_REASON, ht);
        } else {
            const int dumped = pn_hrd_enabled() && pn_hrd_begin(lo[g], sched->n_vocab, &sl->sp, sl->recent,
                                            sl->n_recent,
                                            tgh && tgh->refresh && tgh->st
                                                ? tgh->refresh(tgh->st) : 0);
            ht = dumped ? pn_ht_now() : ht;   /* the copy is not sampler time */
            next = pn_sample_next_token(lo[g], sched->n_vocab,
                                        &sl->sp, sl->recent,
                                        sl->n_recent);
            ht = pn_ht_lap(PN_HT_SAMPLE, ht);
            if (dumped)
                pn_hrd_end(next);
            next = driver_reason_step(sched, sl, next, tgh, &injected);
            ht = pn_ht_lap(PN_HT_REASON, ht);
        }
        /* Every mutation this step can make to the grammar walk has now
         * happened (the sampler advanced over a drawn token; the force-close
         * advanced over an injected one).  Arm the next step's row from the
         * walk as it now stands — the driver issues that step itself, still
         * holding mu, so this write provably precedes the read. */
        driver_arm_full_logits(sched, sl, active_before);
        ht = pn_ht_lap(PN_HT_ARM, ht);
        int emitted = driver_emit_token(sched, sl, next);
        if (emitted < 0) {
            /* Cannot happen (ring_cap == bind budget); defensive. */
            driver_error_slot(sched, sl);
            pn_ht_bind(NULL);
            continue;
        }
#ifdef USE_CUDA
        if (emitted == 0 && snap_n == 1 && sl->mtp_on && !sl->pd_stepped && tgh &&
            bs_spec_shadow_on())
            bs_spec_shadow_step(sched, sl);
#endif
        pthread_cond_signal(&sl->cv);
        ht = pn_ht_lap(PN_HT_EMIT, ht);
        driver_ht_deposited(sl, ht);
        pn_ht_bind(NULL);
    }
    sched->driver_steps++;
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
        const uint64_t ht_gate0 = pn_ht_now();
        while (!sched->shutdown && !gate_ok(sched))
            pthread_cond_wait(&sched->cv_work, &sched->mu);
        if (sched->shutdown)
            break;
        double t_gate = stats ? bs_now_ms() : 0.0;
        const uint64_t ht_acq0 = pn_ht_now();

        /* ── Take the GPU turn — FIFO behind any prefiller already queued
         *    (initial join or L2 chunk re-acquire), so the driver can never
         *    barge a joiner out and a joiner can never be starved by the
         *    free-running L3 step loop.  cv_gpu waits release mu, so owners
         *    keep posting/releasing meanwhile. ── */
        bs_gpu_acquire_locked(sched);
        const uint64_t ht_acq1 = pn_ht_now();

        /* Re-check the gate: it may have closed while we waited for the
         * turn (a release, an error).  Back off by releasing the turn. */
        int n_active = gate_ok(sched);
        if (sched->shutdown || !n_active) {
            bs_gpu_release_locked(sched);
            if (sched->shutdown)
                break;
            continue;
        }

        int snap_n = driver_snapshot_locked(sched, tokens, states, gpu_map_snap);
        /* snap_n may be < n_active if a release raced us; that's fine. */
        for (int k = 0; k < snap_n; k++)
            driver_note_step_width(&sched->slots[gpu_map_snap[k]], snap_n);
        if (snap_n == 0) {
            /* Nothing left to step (all active slots were released). */
            bs_gpu_release_locked(sched);
            continue;
        }

#ifdef USE_CUDA
        /* Slots an overlap cycle stepped are the driver's again (pd_stepped). */
        for (int k = 0; k < snap_n; k++)
            sched->slots[gpu_map_snap[k]].pd_stepped = false;
        /* ── Solo slot on an MTP model: one exact MTP cycle instead of a
         *    one-token step (batch_sched_spec.h). ── */
        if (snap_n == 1 && driver_spec_step(sched, gpu_map_snap[0], tokens[0])) {
            sched->driver_steps++;
            bs_gpu_release_locked(sched);
            continue;
        }
#endif
        /* ── GPU step: turn owned (no prefill can overlap), mu dropped
         *    (owners keep draining rings / releasing meanwhile). ── */
        pthread_mutex_unlock(&sched->mu);
        double t_launch = bs_now_ms();
        const uint64_t ht_launch = pn_ht_now();
        const uint64_t ht_cpu0 = pn_ht_on() ? pn_ht_thread_cpu() : 0;
        memset(lo, 0, sizeof(lo));
        int rc = bs_slots_step(sched, tokens, states, gpu_map_snap, snap_n, lo);
        const double plain_ms = bs_now_ms() - t_launch;
        const uint64_t ht_stepped = pn_ht_now();
        const uint64_t ht_cpu1 = pn_ht_on() ? pn_ht_thread_cpu() : 0;
        /* Phase 0: split the step into the GPU forward and the host-side
         * per-slot deposit+sampling that follows it, so `samp_ms` can be read
         * against K directly (the per-slot sampler is sequential, so it is
         * expected to scale with snap_n). Stamped before re-taking mu so the
         * lock wait is not charged to either. */
        double t_dec = stats ? bs_now_ms() : 0.0;
        pthread_mutex_lock(&sched->mu);
        if (pn_ht_on())
            driver_ht_step(sched, gpu_map_snap, snap_n, ht_gate0, ht_acq0, ht_acq1, ht_launch,
                           ht_stepped, ht_cpu1 - ht_cpu0);

        if (rc != 0)
            fprintf(stderr, "[batch_sched] decode_slots failed (n=%d); "
                    "erroring out active slots\n", snap_n);

        driver_deposit_locked(sched, tokens, gpu_map_snap, snap_n, lo, rc, plain_ms);
        if (rc == 0)   /* what an overlap cycle of this width is held to */
            bs_pd_budget_note_plain(&sched->pd_budget, snap_n, bs_now_ms() - t_launch);
        const uint64_t ht_rel0 = pn_ht_now();
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
        if (ht_rel0)
            driver_ht_charge(sched, gpu_map_snap, snap_n, PN_HT_RELEASE, pn_ht_now() - ht_rel0);
    }

    /* Shutdown: wake every parked owner (their wait predicate checks
     * sched->shutdown).  bs_destroy broadcasts too — belt and braces. */
    for (int i = 0; i < sched->cap; i++)
        pthread_cond_broadcast(&sched->slots[i].cv);
    pthread_mutex_unlock(&sched->mu);
    return NULL;
}

/* ============================================================
 * Session blocks — carved and registered at load
 *
 * Every lane gets its own persistent block on the card (GPU row i: a ctx_cap
 * KV ring row per attention layer + an SSM image per SSM layer) for the
 * scheduler's whole life. Carving at load is what makes "it does not fit" a
 * load-time refusal instead of a request-time failure, and binding each lane
 * once registers its pn_state with its row so a step can run any subset of
 * rows. Runs before the driver thread exists, so nothing can interleave.
 * The llama family keeps its sessions in host KV instead (no card block;
 * batch_sched_slots.h), and its fit was checked before the states existed.
 * ============================================================ */
static int bs_carve_session_blocks(pn_batch_sched *sched)
{
    if (bs_slots_carve(sched) != 0) {
        fprintf(stderr, "[batch_sched] FATAL: cannot carve %d session blocks "
                "for '%s' %s - lower --max-sessions or --ctx-cap\n", sched->cap,
                sched->model_name[0] ? sched->model_name : "?",
                sched->slot_store == BS_SLOT_STORE_HOST_KV
                    ? "(host KV + logits row per session)"
                    : "on its dies (KV ring row at ctx_cap + SSM image per "
                      "session)");
        return -1;
    }
    for (int i = 0; i < sched->cap; i++) {
        if (bs_slots_bind(sched, sched->slots[i].state, i) != 0) {
            fprintf(stderr, "[batch_sched] FATAL: registering session block "
                    "%d of %d failed\n", i, sched->cap);
            return -1;
        }
    }
    sched->slots_gpu_init = true;
    fprintf(stderr, "[batch_sched] carved %d session blocks for '%s'\n",
            sched->cap, sched->model_name[0] ? sched->model_name : "?");
    return 0;
}

/* ============================================================
 * QWEN35_BS_PD_OVERLAP — the decode half of a prefill overlap cycle
 *
 * While a prefiller holds the GPU turn it runs, between its prefill pieces,
 * the very step the driver would have run next: the same snapshot
 * (driver_snapshot_locked), the same deposit and in-driver sampling
 * (driver_deposit_locked), so each slot's tokens are exactly the driver's.
 * The driver itself stays parked on its turnstile ticket meanwhile, and
 * re-reads the active set when it next gets the turn.
 * ============================================================ */

int bs_pd_cycle(pn_batch_sched *sched, void *cursor)
{
    uint32_t  tokens[BS_MAX_SLOTS];
    pn_state *states[BS_MAX_SLOTS];
    float    *lo[BS_MAX_SLOTS];
    int       gpu_map_snap[BS_MAX_SLOTS];

    pthread_mutex_lock(&sched->mu);
    int snap_n = gate_ok(sched) ? driver_snapshot_locked(sched, tokens, states, gpu_map_snap)
                                : 0;
    if (snap_n == 0) {
        pthread_mutex_unlock(&sched->mu);
        return 1;
    }
    for (int k = 0; k < snap_n; k++) {
        sched->slots[gpu_map_snap[k]].pd_stepped = true;
        driver_note_step_width(&sched->slots[gpu_map_snap[k]], snap_n);
    }
    const int pieces = sched->pd_budget.pieces;
    pthread_mutex_unlock(&sched->mu);

    const double t_launch = bs_now_ms();
    memset(lo, 0, sizeof(lo));
    int rc = bs_slots_pd_cycle(sched, cursor, tokens, states, gpu_map_snap, snap_n, lo,
                               pieces);
    const double step_ms = bs_now_ms() - t_launch;

    pthread_mutex_lock(&sched->mu);
    if (rc != 0)
        fprintf(stderr, "[batch_sched] overlap cycle failed (n=%d); erroring out active "
                "slots\n", snap_n);
    driver_deposit_locked(sched, tokens, gpu_map_snap, snap_n, lo, rc, step_ms);
    const double cycle_ms = bs_now_ms() - t_launch;
    if (bs_stats_on()) {
        double ph[PN_QWEN35MOE_PD_PHASES];
        bs_slots_pd_phase_ms(cursor, ph);
        fprintf(stderr, "[bs] pd n=%d cycle_ms=%.1f plain_ms=%.1f pieces=%d "
                "ph=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f\n", snap_n, cycle_ms,
                snap_n <= BS_PD_MAX_WIDTH ? sched->pd_budget.plain_ms[snap_n] : 0.0,
                pieces, ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]);
    }
    pthread_mutex_unlock(&sched->mu);
    return rc ? -1 : 0;
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
        return NULL;
    if (!bs_slots_supported(model)) {
        fprintf(stderr, "[batch_sched] '%s': this architecture has no batched "
                "session storage\n", model_name && model_name[0] ? model_name
                                                            : "?");
        return NULL;
    }

    /* Clamp to the memory-pressure concurrency ceiling up front so the
     * per-bind ladder guard can never refuse a bind later (which would turn
     * every request into a 500). One block per slot, no scratch row. */
    int ceiling = pn_qwen35moe_gpu_ladder_max_slots();
    if (cap > ceiling) {
        fprintf(stderr, "[batch_sched] capping slots %d -> %d "
                "(memory-pressure ladder ceiling)\n", cap, ceiling);
        cap = ceiling;
    }
    /* A HOST_KV session is an eager host allocation (pn_state_new below):
     * refuse a set that does not fit BEFORE allocating it. */
    if (bs_slots_precheck(model, cap, model_name) != 0)
        return NULL;

    pn_batch_sched *sched = calloc(1, sizeof(*sched));
    /* Stamp the log name BEFORE the driver thread starts, so a force-close
     * logged by an early step already has it (calloc left it empty). */
    if (sched && model_name && model_name[0])
        snprintf(sched->model_name, sizeof(sched->model_name), "%s", model_name);
    if (!sched)
        return NULL;

    sched->model = model;
    sched->cap   = cap;
    sched->slot_store = bs_slot_store_for(model);

    /* GPU turnstile: ticket 0 is the first served. */
    sched->gpu_ticket_next    = 0;
    sched->gpu_ticket_serving = 0;
    bs_pd_budget_init(&sched->pd_budget, bs_pd_pieces_parse(getenv("QWEN35_BS_PD_PIECES")));

    /* L3 — in-driver sampling (default ON; QWEN35_BS_DRIVER_SAMPLE=0
     * restores the legacy owner-side lockstep). */
    {
        const char *env = getenv("QWEN35_BS_DRIVER_SAMPLE");
        sched->driver_sample = !(env && env[0] == '0');
    }
    sched->n_vocab   = model->config.n_vocab;
    sched->mtp       = sg_mtp_cuda_available(model);
    if (sched->mtp)
        fprintf(stderr, "[batch_sched] MTP head loaded: a solo decoding slot speculates "
                "(exact; PN_MTP_SERVE_SPEC=0 off)\n");
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

    sched->n_active_gpu       = 0;
    sched->active_set_changed = false;
    for (int i = 0; i < BS_MAX_SLOTS; i++)
        sched->gpu_map[i] = -1;

    /* Carve every session's block on the card NOW, at load: a model whose
     * max_sessions blocks do not fit is refused here, loudly, rather than
     * failing a request later. */
    if (bs_carve_session_blocks(sched) != 0) {
        bs_slots_free(sched);
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
        bs_slots_free(sched);
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
        bs_turn_ckpt_free(&sched->slots[i].ckpt);
        bs_turn_ckpt_free(&sched->slots[i].partial);
        bs_turn_ckpt_free(&sched->slots[i].divck);
#ifdef USE_CUDA
        sg_mtp_cuda_end(&sched->slots[i].mtp);
#endif
        bs_sys_snap_unref(sched, &sched->slots[i]);
        free(sched->slots[i].sys_tokens);   /* SYS1-batched boundary snapshot */
    }
#ifdef USE_CUDA
    bs_slots_free(sched);
#endif
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

/* The least recently active IDLE slot of `sid`'s family, or -1: the
 * conversation the user touched last keeps its warm KV.  Under mu. */
static int family_lru_idle(const pn_batch_sched *sched, const char *sid)
{
    int lru = -1;

    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        const char *owner = bs_slot_session_owner(sl);
        if (sl->status == BS_SLOT_IDLE && owner &&
            pn_session_same_family(owner, sid) &&
            (lru < 0 || sl->last_active_ms < sched->slots[lru].last_active_ms))
            lru = i;
    }
    return lru;
}

/* Claim one IDLE slot (already under mu), mark it PREFILL and record `sid` as
 * the session it runs for.  Preference order (cheapest resume first):
 *   1. THIS session's warm IDLE slot (committed history → append delta);
 *   2. the IDLE slot holding THIS session's system-prefix snapshot
 *      (SYS1-batched: a divergent turn still resumes at the boundary —
 *      matters when the warm history was invalidated but the snapshot,
 *      which survives resets, was not);
 *   3. an IDLE slot no session holds (this turn full-prefills) — for a
 *      family that already holds a slot, only while that leaves a seat for
 *      every family admission could still let in
 *      (bs_family_may_take_extra_locked); otherwise it waits for its own;
 *   4. the least recently active IDLE slot of THIS session's family (another
 *      conversation of the same user gives way).
 * Never a slot another family holds: its KV would be erased to seat this one
 * (admission refuses instead — batch_sched_admission.h).  No `sid` takes only
 * step 3.  Returns the slot index >= 0, or -1 if none fits. */
static int try_claim_idle(pn_batch_sched *sched, const char *sid)
{
    int found = -1;
    bool has_sid = sid && sid[0];

    for (int i = 0; has_sid && found < 0 && i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && sl->sid[0] &&
            strncmp(sl->sid, sid, sizeof(sl->sid)) == 0)
            found = i;
    }
    for (int i = 0; has_sid && found < 0 && i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && sl->sys_sid[0] &&
            strncmp(sl->sys_sid, sid, sizeof(sl->sys_sid)) == 0)
            found = i;
    }
    bool cold_ok = found >= 0 || !has_sid ||
                   !bs_family_holds_slot_locked(sched, sid) ||
                   bs_family_may_take_extra_locked(sched);
    for (int i = 0; cold_ok && found < 0 && i < sched->cap; i++) {
        bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && !bs_slot_session_owner(sl))
            found = i;
    }
    if (has_sid && found < 0)
        found = family_lru_idle(sched, sid);
    if (found >= 0) {
        bs_slot_mark_prefill(&sched->slots[found]);
        snprintf(sched->slots[found].claim_sid,
                 sizeof(sched->slots[found].claim_sid), "%s", has_sid ? sid : "");
    }
    return found;
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
    bs_slots_forget(sched, sched->slots[idx].state);
#endif
    /* The warm state is gone — this slot no longer holds any session. */
    sched->slots[idx].sid[0] = '\0';
    sched->slots[idx].hist_n = 0;
    bs_slot_clear_projections(&sched->slots[idx]);
    sched->slots[idx].ct_keep = 0;
    sched->slots[idx].ct_prompt_from = 0;
    bs_turn_ckpt_forget(&sched->slots[idx].ckpt);
    bs_turn_ckpt_forget(&sched->slots[idx].partial);
    bs_turn_ckpt_forget(&sched->slots[idx].divck);
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
    return pn_session_snapshot_supported(sched->model->config.arch);
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
enum { BS_FIT_NONE, BS_FIT_SYS, BS_FIT_CKPT, BS_FIT_WARM };

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
    if (bs_turn_ckpt_fits(&sl->partial, sid, tokens, n_ctx) ||
        bs_turn_ckpt_fits(&sl->ckpt, sid, tokens, n_ctx))
        return BS_FIT_CKPT;
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
     * slot can take this one instead — and that slot is not a seat another
     * family could still need (the family holds `idx` already). */
    if (!bs_family_may_take_extra_locked(sched))
        return idx;
    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_IDLE && !bs_slot_session_owner(sl))
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
        memcpy(sched->slots[to].claim_sid, sched->slots[idx].claim_sid,
               sizeof(sched->slots[to].claim_sid));
        /* Claimed but never touched (no GPU turn, no state, no bind): handing
         * it back is just the status flip; its warm fields stay as they were. */
        sched->slots[idx].status = BS_SLOT_IDLE;
        sched->slots[idx].claim_sid[0] = '\0';
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
static void bs_sys_snap_unref(const pn_batch_sched *sched, bs_slot *sl)
{
    if (sl->sys_snap && sl->sys_snap_refs && --(*sl->sys_snap_refs) > 0) {
        sl->sys_snap = NULL;
        sl->sys_snap_refs = NULL;
        return;
    }
    pn_session_snapshot_free(sched->model->config.arch, sl->sys_snap);
    free(sl->sys_snap_refs);
    sl->sys_snap = NULL;
    sl->sys_snap_refs = NULL;
}

/* Drop a slot's boundary snapshot (keeps the tokens buffer for realloc reuse).
 * sys_sid is cleared under mu — the affinity scans read it. */
static void bs_sys_snap_drop(pn_batch_sched *sched, bs_slot *sl)
{
    bs_sys_snap_unref(sched, sl);
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
    if (pn_session_snapshot_restore(sched->model->config.arch,
                                    sl->state->kv_cache, sl->sys_snap) != 0)
        return -1;
    sl->state->pos = sys_len;
#ifdef USE_CUDA
    bs_slots_forget(sched, sl->state);
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

/* Is this slot's warm state actually RE-USABLE, and not merely bookkept?
 *
 * A session's KV + SSM live in its own block on the card (GPU row = slot
 * index, carved at load), never in host RAM between turns. The warm rungs
 * resume from that block — the delta prefill re-seeds the shared buffers from
 * it device to device — so they must prove the block still holds exactly
 * this session's committed state (SSM image at pos, KV rows [0, pos)).
 * Nothing in the normal flow can take the block away; this guards the
 * edges (a reset, an aborted turn, eager mode's different bookkeeping), and
 * a failure just means the caller falls through to a re-prefilling rung.
 *
 * GPU turn must be held. */
static bool bs_warm_kv_reusable(pn_batch_sched *sched, bs_slot *sl)
{
#ifdef USE_CUDA
    if (!bs_slots_live_resumable(sched, sl->state, sl->slot_idx)) {
        if (pn_reqprof_enabled())
            fprintf(stderr, "[reqprof] warm resume refused slot=%d sid=%s: "
                    "the session block no longer holds this state\n",
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
    bs_slots_forget(sched, sl->state);
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

/* First index where p[0, n) and tokens[0, n_ctx) differ (min(n, n_ctx) when
 * one is a prefix of the other). */
static int bs_first_diff(const uint32_t *p, int n, const uint32_t *tokens,
                         int n_ctx)
{
    int m = n < n_ctx ? n : n_ctx;
    for (int i = 0; i < m; i++)
        if (p[i] != tokens[i])
            return i;
    return m;
}

/* Why a request falls to the system-prefix rung, in one short phrase for the
 * `sys-prefix reuse` line: where it left the slot's committed history for this
 * session (a history edit, e.g. a client rewriting old messages), or that the
 * slot held none of it. Cheap: one pass over at most hist_n tokens. */
static void bs_reread_cause(const bs_slot *sl, const char *sid,
                            const uint32_t *tokens, int n_ctx, char *out,
                            size_t cap)
{
    bool own = sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0;
    if (own && sl->hist_n > 0)
        snprintf(out, cap, "history diverged at %d of %d",
                 bs_first_diff(sl->hist, sl->hist_n, tokens, n_ctx),
                 sl->hist_n);
    else if (sl->ckpt.valid &&
             strncmp(sl->ckpt.sid, sid, sizeof(sl->ckpt.sid)) == 0)
        snprintf(out, cap, "checkpoint diverged at %d of %d",
                 bs_first_diff(sl->ckpt.client, sl->ckpt.client_n, tokens,
                               n_ctx), sl->ckpt.client_n);
    else
        snprintf(out, cap, "slot held no history of this session");
}

/* Rung 1, append resume: THIS slot's warm state belongs to `sid`, its
 * committed history is an exact prefix of the new context, and the state sits
 * exactly at that history's end (pos == hist_n).  The agentic append case:
 * prefill only [hist_n, n_ctx).  hist_n < n_ctx is strict so at least one delta
 * token remains to recover the next-token logits (the degenerate full-hit never
 * happens in a real chat turn — an assistant opener is always appended — and
 * falls through to a full reset).  1 resumed (*delta set), 0 not applicable. */
static int bs_exact_append_resume(pn_batch_sched *sched, int idx,
                                  const char *sid, const uint32_t *tokens,
                                  int n_ctx, int *delta)
{
    bs_slot *sl = &sched->slots[idx];
    if (!(sid && sid[0] && sl->sid[0] &&
          strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
          sl->hist_n > 0 && sl->hist_n < n_ctx &&
          (int)sl->state->pos == sl->hist_n &&
          memcmp(sl->hist, tokens,
                 (size_t)sl->hist_n * sizeof(uint32_t)) == 0 &&
          bs_warm_kv_reusable(sched, sl)))
        return 0;
    *delta = sl->hist_n;   /* warm: resume, do NOT reset */
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
    bs_slots_forget(sched, sl->state);
#endif
    return 1;
}

/* The warm rungs (1 to 1d) that `think_history` allows, in ladder order.
 * Returns the rung that resumed (*delta set), or PN_RESUME_COLD when none did
 * and the caller falls to the system prefix. */
static pn_resume_kind bs_resume_warm_rung(pn_batch_sched *sched, int idx,
                                          const char *sid,
                                          const uint32_t *tokens, int n_ctx,
                                          pn_think_history think_history,
                                          int *delta)
{
    /* as_sent: the KV must be the request as rendered; every warm rung can
     * leave think blocks in it that the request does not carry. */
    if (think_history == PN_THINK_HISTORY_AS_SENT)
        return PN_RESUME_COLD;
    /* drop_last: the request is the last turn's start plus that turn's
     * content without its think block; resume at the turn checkpoint when its
     * KV is exactly the request's prefix, else fall to the system prefix. */
    if (think_history == PN_THINK_HISTORY_DROP_LAST)
        return bs_turn_ckpt_resume_exact(sched, idx, sid, tokens, n_ctx, delta)
                   ? PN_RESUME_TURN_CKPT : PN_RESUME_COLD;
    if (bs_exact_append_resume(sched, idx, sid, tokens, n_ctx, delta))
        return PN_RESUME_EXACT;
    /* 1b. Projection append resume — the client's re-render of the history
     *     differs from the committed KV only in the think blocks, either
     *     because it dropped them or because it sent the reasoning back as
     *     text.  See bs_projection_resume. */
    if (bs_projection_resume(sched, idx, sid, tokens, n_ctx, delta))
        return PN_RESUME_PROJECTION;
    /* 1c, aborted prefill: the client hung up mid-prefill and resent (or
     * extended) the request; the slot kept what was already computed. */
    if (bs_partial_prefill_resume(sched, idx, sid, tokens, n_ctx, delta))
        return PN_RESUME_PARTIAL;
    /* 1c. Turn-start checkpoint — the client dropped or rewrote the last
     *     turn's reply (a reasoning-only turn, an interrupt, an aborted
     *     prefill), but everything before it is unchanged.  See
     *     batch_sched_turn_checkpoint.h. */
    if (bs_turn_ckpt_resume(sched, idx, sid, tokens, n_ctx, delta))
        return PN_RESUME_TURN_CKPT;
    /* 1d. Divergence checkpoint — the client rewrote history after a head it
     *     keeps (a context compaction); resume where the previous rewrite left
     *     the history.  See batch_sched_turn_checkpoint.h. */
    if (bs_div_ckpt_resume(sched, idx, sid, tokens, n_ctx, delta))
        return PN_RESUME_DIV_CKPT;
    return PN_RESUME_COLD;
}

/* Rungs 2 and 3, after every allowed warm rung missed.
 *
 * 2. System-prefix resume: divergent history but this slot holds THIS
 *    session's boundary snapshot over these exact system tokens (the
 *    reasoning-model turn).  Restore it and prefill only the conversation.
 *    The stale committed history is dropped — bs_slot_finalize rebuilds it at
 *    end of turn.
 * 3. Fresh/diverged with no usable snapshot: full reset + prefill.
 *
 * Returns PN_RESUME_SYS_PREFIX or PN_RESUME_COLD and sets *delta. */
static pn_resume_kind bs_resume_sys_or_cold(pn_batch_sched *sched, int idx,
                                            const char *sid,
                                            const uint32_t *tokens, int n_ctx,
                                            int sys_len, int sys_ok, int *delta)
{
    bs_slot *sl = &sched->slots[idx];
    /* Before the fallbacks mutate this slot, decide whether the miss is an
     * ordinary history divergence or a second conversation wearing this
     * session's id — and say so out loud. */
    bs_note_session_substitution(sched, idx, sid, tokens, n_ctx, sys_len);
    bs_log_append_resume_miss(sched->model, idx, sid, sl, tokens, n_ctx);
    if (sys_ok && bs_sys_snap_matches(sl, sid, tokens, sys_len)) {
        /* Where the conversation left what this slot holds (for the re-read
         * census; read before the history is dropped). */
        char why[96];
        bs_reread_cause(sl, sid, tokens, n_ctx, why, sizeof(why));
        if (bs_sys_snap_restore(sched, sl, sys_len) == 0) {
            pthread_mutex_lock(&sched->mu);
            sl->sid[0] = '\0';   /* history no longer matches the state */
            sl->hist_n = 0;
            bs_slot_clear_projections(sl);
            pthread_mutex_unlock(&sched->mu);
            bs_turn_ckpt_forget(&sl->ckpt);
            fprintf(stderr, "[batch_sched] sys-prefix reuse slot=%d "
                    "sid=%s: resume at %d, prefill %d of %d tokens (%s)\n",
                    idx, sid, sys_len, n_ctx - sys_len, n_ctx, why);
            *delta = sys_len;
            return PN_RESUME_SYS_PREFIX;
        }
        /* Restore refused (dim/KV-mode change): the snapshot can never apply
         * again — drop it so a capture can replace it.  Nothing was written
         * (the restore validates dims before any copy). */
        bs_sys_snap_drop(sched, sl);
    }
    bs_slot_reset_state(sched, idx);
    *delta = 0;
    return PN_RESUME_COLD;
}

/* About how many tokens a request will prefill: the prompt past where it
 * leaves the history slot idx holds for its session (the whole prompt when
 * the slot holds another one). Read before the turn: the slot is this
 * request's own and nothing else writes its history meanwhile. >= 1. */
static int bs_pd_wait_estimate(const pn_batch_sched *sched, int idx, const char *sid,
                               const uint32_t *tokens, int n_ctx)
{
    const bs_slot *sl = &sched->slots[idx];
    int same = 0;
    if (sid && sid[0] && sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
        sl->hist && sl->hist_n > 0) {
        const int n = sl->hist_n < n_ctx ? sl->hist_n : n_ctx;
        while (same < n && sl->hist[same] == tokens[same])
            same++;
    }
    return n_ctx - same > 0 ? n_ctx - same : 1;
}

void bs_prefill_resume_think(pn_batch_sched *sched, int idx, const char *sid,
                             const uint32_t *tokens, int n_ctx, int sys_len,
                             pn_think_history think_history,
                             bs_resume_outcome *out)
{
    bs_resume_outcome none = { 0, 0, PN_RESUME_COLD };
    if (out)
        *out = none;
    if (!sched || idx < 0 || idx >= sched->cap || !tokens || n_ctx <= 0)
        return;
    /* Same GPU-owner contract as bs_prefill_begin; the wait advertises this
     * prefill's size for shortest-prefill-first (bs_pd_shorter_waiting). */
    const int est = bs_pd_wait_estimate(sched, idx, sid, tokens, n_ctx);
    pthread_mutex_lock(&sched->mu);
    sched->slots[idx].pf_wait_est = est;
    bs_gpu_acquire_locked(sched);
    sched->slots[idx].pf_wait_est = 0;
    pthread_mutex_unlock(&sched->mu);

    bs_slot *sl = &sched->slots[idx];
    /* A system prefix is only usable when a conversation follows it. */
    int sys_ok = sid && sid[0] && sys_len > 0 && sys_len < n_ctx;
    int delta = 0;
    /* Default committed-log seeding: the log IS the prompt (rungs 1, 2 and 3).
     * Only the think-elision rung overrides it. */
    sl->ct_keep = 0;
    sl->ct_prompt_from = 0;
    sl->sid_conflict = false;
    /* Where this request leaves what the slot holds for the session (rung 1d
     * plans a checkpoint there), read before any rung rewrites the history. */
    const int div_point = bs_div_point(sched, idx, sid, tokens, n_ctx);
    /* Resume ladder — mirrors pn_prefix_cache_acquire_sys. */
    pn_resume_kind kind = bs_resume_warm_rung(sched, idx, sid, tokens, n_ctx,
                                              think_history, &delta);
    if (kind == PN_RESUME_COLD)
        kind = bs_resume_sys_or_cold(sched, idx, sid, tokens, n_ctx, sys_len,
                                     sys_ok, &delta);
    if (think_history != PN_THINK_HISTORY_KEEP)
        fprintf(stderr, "[batch_sched] think_history=%s slot=%d sid=%s: "
                "resume %s at %d, prefill %d of %d tokens\n",
                think_history == PN_THINK_HISTORY_AS_SENT ? "as_sent"
                                                          : "drop_last",
                idx, sid ? sid : "", pn_resume_kind_name(kind), delta,
                n_ctx - delta, n_ctx);

    /* An aborted prefill's record is single-use: it was either resumed (its
     * tokens are the history now) or the rung that ran replaced the state. */
    bs_turn_ckpt_forget(&sl->partial);
    bs_div_ckpt_plan(sched, idx, div_point, delta, n_ctx);

    if (!out)
        return;
    out->delta_start = delta;
    out->kind = kind;
    /* Full prefill ahead + no snapshot covering these exact system tokens →
     * ask the caller to capture the boundary state mid-prefill (replacing a
     * stale snapshot from a changed system prompt or another session). */
    out->capture_sys = delta == 0 && sys_ok &&
                       bs_sys_snapshot_supported(sched) &&
                       !bs_sys_snap_matches(sl, sid, tokens, sys_len);
}

void bs_prefill_resume_sys(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *tokens, int n_ctx, int sys_len,
                           int *delta_start, int *capture_sys)
{
    bs_resume_outcome out;
    bs_prefill_resume_think(sched, idx, sid, tokens, n_ctx, sys_len,
                            PN_THINK_HISTORY_KEEP, &out);
    if (delta_start)
        *delta_start = out.delta_start;
    if (capture_sys)
        *capture_sys = out.capture_sys;
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
        bs_sys_snap_unref(sched, sl);
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
    if (bs_slots_state_to_host(sched, sl->state) != 0)
        return -1;
#endif

    const pn_arch arch = sched->model->config.arch;
    void *snap = pn_session_snapshot_capture(arch, sl->state->kv_cache, sys_len);
    if (!snap)
        return -1;
    uint32_t *st = realloc(sl->sys_tokens, (size_t)sys_len * sizeof(uint32_t));
    if (!st) {
        pn_session_snapshot_free(arch, snap);
        return -1;
    }
    memcpy(st, tokens, (size_t)sys_len * sizeof(uint32_t));
    bs_sys_snap_unref(sched, sl);
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

    /* Exclusive: no decode step / rebuild can run.  Counted while queued so a
     * prefiller holding the turn yields at its next chunk boundary instead of
     * holding this finished reply behind its whole prompt. */
    pthread_mutex_lock(&sched->mu);
    sched->n_finalize_waiting++;
    bs_gpu_acquire_locked(sched);
    sched->n_finalize_waiting--;
    /* gpu_idx is this slot's fixed block row (== idx) once bound this turn,
     * -1 if the turn never bound one. */
    int gpu_idx = sl->gpu_idx;
    pthread_mutex_unlock(&sched->mu);
    int rc = 0;
#ifdef USE_CUDA
    if (gpu_idx < 0) {
        rc = -1;   /* slot never bound a GPU decode state (e.g. EOS at token 1) */
    } else {
        /* Lazy mode: the session's KV + SSM stay in its block on the card
         * (GPU row gpu_idx == idx) — nothing to capture; just confirm the
         * block holds the committed state. Eager mode keeps the full legacy
         * host capture. */
        if (pn_qwen35moe_host_kv_mirror_eager()) {
            if (bs_slots_state_to_host(sched, sl->state) != 0 ||
                bs_slots_row_ssm_to_host(sched, gpu_idx, sl->state) != 0)
                rc = -1;
        } else if (!bs_slots_row_holds_state(sched, gpu_idx, sl->state)) {
            rc = -1;
        }
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
    sl->claim_sid[0] = '\0';
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
        /* The session lives in its block on the card (GPU row idx). Export is
         * the one deliberate copy to host: the KV rows [0, min(pos, max_seq))
         * via gpu_state_to_host (fails loudly if no ring holds them) and the
         * SSM image from the block, so the blob can go to the client. */
        bool lazy = !pn_qwen35moe_host_kv_mirror_eager();
        if ((lazy &&
             (!bs_slots_row_holds_state(sched, idx, sl->state) ||
              bs_slots_row_ssm_to_host(sched, idx, sl->state) != 0)) ||
            bs_slots_state_to_host(sched, sl->state) != 0) {
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
            snap = pn_session_snapshot_capture(sched->model->config.arch,
                                               sl->state->kv_cache, n_tok);
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
            pn_session_snapshot_free(sched->model->config.arch, snap);
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
    if (pn_session_snapshot_pos(sched->model->config.arch, snap) != n_tok)
        return BS_SESSION_ERROR;

    /* Seat it as a chat turn would be (session admission, then this session's
     * own slot — a re-restore replaces its warm state — else a slot nobody
     * holds, else its own family's).  Never displace an active slot or
     * another family's warm one. */
    pthread_mutex_lock(&sched->mu);
    int idx = bs_admission_check_locked(sched, sid) == BS_ACQUIRE_OK
            ? try_claim_idle(sched, sid) : -1;
    pthread_mutex_unlock(&sched->mu);
    if (idx < 0)
        return BS_SESSION_BUSY;
    bs_slot *sl = &sched->slots[idx];

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
        if (pn_session_snapshot_restore(sched->model->config.arch,
                                        sl->state->kv_cache, snap) != 0) {
            rc = BS_SESSION_ERROR;
        } else {
            sl->state->pos = n_tok;
#ifdef USE_CUDA
            bs_slots_forget(sched, sl->state);
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
        sl->last_active_ms = bs_now_ms();   /* the idle reaper's clock */
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
        int ckpt_mine = (sl->status == BS_SLOT_IDLE && sl->ckpt.valid &&
                         (n ? strncmp(sl->ckpt.sid, key, n) == 0
                            : strncmp(sl->ckpt.sid, key,
                                      sizeof(sl->ckpt.sid)) == 0));
        int partial_mine = (sl->status == BS_SLOT_IDLE && sl->partial.valid &&
                            (n ? strncmp(sl->partial.sid, key, n) == 0
                               : strncmp(sl->partial.sid, key,
                                         sizeof(sl->partial.sid)) == 0));
        if (mine) {
            sl->sid[0] = '\0';
            sl->hist_n = 0;
            bs_slot_clear_projections(sl);
            sl->armed_sid[0] = '\0';
        }
        if (ckpt_mine)
            bs_turn_ckpt_forget(&sl->ckpt);
        if (sl->status == BS_SLOT_IDLE && sl->divck.valid &&
            (n ? strncmp(sl->divck.sid, key, n) == 0
               : strncmp(sl->divck.sid, key, sizeof(sl->divck.sid)) == 0))
            bs_turn_ckpt_forget(&sl->divck);
        if (partial_mine)
            bs_turn_ckpt_forget(&sl->partial);
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

void bs_slot_drop_session(pn_batch_sched *sched, bs_slot *sl)
{
    pthread_mutex_lock(&sched->mu);
    sl->sid[0] = '\0';
    sl->hist_n = 0;
    bs_slot_clear_projections(sl);
    sl->armed_sid[0] = '\0';
    bs_turn_ckpt_forget(&sl->ckpt);
    bs_turn_ckpt_forget(&sl->partial);
    bs_turn_ckpt_forget(&sl->divck);
    pthread_mutex_unlock(&sched->mu);
    bs_sys_snap_drop(sched, sl);   /* takes mu itself for the sys_sid clear */
}

void bs_set_session_admission(pn_batch_sched *sched, int max_sessions,
                              uint64_t idle_ttl_ms)
{
    if (!sched)
        return;
    pthread_mutex_lock(&sched->mu);
    sched->max_sessions = max_sessions > 0 ? max_sessions : 0;
    sched->session_idle_ttl_ms = idle_ttl_ms;
    pthread_mutex_unlock(&sched->mu);
}

void bs_set_watch_hook(pn_batch_sched *sched, pn_session_watched_fn fn,
                       void *ctx)
{
    if (!sched)
        return;
    pthread_mutex_lock(&sched->mu);
    sched->watch_fn = fn;
    sched->watch_ctx = fn ? ctx : NULL;
    pthread_mutex_unlock(&sched->mu);
}

bool bs_session_seat_free(pn_batch_sched *sched)
{
    if (!sched)
        return false;
    pthread_mutex_lock(&sched->mu);
    bool free_seat = bs_admission_seat_free_locked(sched);
    pthread_mutex_unlock(&sched->mu);
    return free_seat;
}

int bs_session_count(pn_batch_sched *sched)
{
    if (!sched)
        return 0;
    pthread_mutex_lock(&sched->mu);
    int n = bs_family_count_locked(sched);
    pthread_mutex_unlock(&sched->mu);
    return n;
}

void bs_session_occupancy(pn_batch_sched *sched, bs_occupancy *out)
{
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    if (!sched)
        return;
    pthread_mutex_lock(&sched->mu);
    out->families = bs_family_count_locked(sched);
    out->max_sessions = sched->max_sessions > 0 ? sched->max_sessions
                                                : sched->cap;
    out->slots = sched->cap;
    out->slots_free = bs_unheld_count_locked(sched);
    out->slots_held = sched->cap - out->slots_free;
    out->seat_free = bs_admission_seat_free_locked(sched);
    pthread_mutex_unlock(&sched->mu);
}

/* QWEN35_BS_PREFILL_DECODE_SHARE, read once (batch_sched_prefill_share.h). */
static int bs_prefill_decode_share(void)
{
    static int v = -1;
    if (v < 0)
        v = bs_prefill_decode_share_parse(
            getenv("QWEN35_BS_PREFILL_DECODE_SHARE"));
    return v;
}

/* Under mu: is any slot still decoding (BOUND/DECODE, not finished)? Unlike
 * gate_ok this ignores token_ready, which is false while a step is in flight. */
static bool bs_any_decoding_locked(const pn_batch_sched *sched)
{
    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        if ((sl->status == BS_SLOT_DECODE || sl->status == BS_SLOT_BOUND) &&
            !sl->finished)
            return true;
    }
    return false;
}

/* Under mu, the GPU turn just released: let the driver step the decoders for
 * up to `window_ms` before the caller re-queues for the turn. Wakes on every
 * turn release (cv_gpu) and stops early when nothing is left to decode or the
 * scheduler shuts down. Holds no ticket meanwhile, so the free-running driver
 * (and any other prefiller) takes turns. */
static void bs_prefill_stand_aside_locked(pn_batch_sched *sched,
                                          double window_ms)
{
    if (window_ms <= 0.0)
        return;
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    const long long ns = (long long)(window_ms * 1e6);
    deadline.tv_sec += (time_t)(ns / 1000000000LL);
    deadline.tv_nsec += (long)(ns % 1000000000LL);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    while (!sched->shutdown && bs_any_decoding_locked(sched)) {
        if (pthread_cond_timedwait(&sched->cv_gpu, &sched->mu, &deadline) ==
            ETIMEDOUT)
            break;
    }
}
void bs_prefill_yield(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;

    /* Only worth a handoff when a turn waiter would put tokens in front of a
     * client: the driver with a step to run (gate open: >=1 active slot,
     * tokens ready), or a finished slot queued in bs_slot_finalize, whose
     * reply is sent only after it.  A queued prefiller alone is no reason —
     * alternating two prompts chunk by chunk delays both first tokens.
     * Racing a token post / release / finalize here is harmless — worst case
     * one missed (or one empty) interleave window. */
    pthread_mutex_lock(&sched->mu);
    const int runnable = gate_ok(sched);
    const int finalizers = sched->n_finalize_waiting;
    pthread_mutex_unlock(&sched->mu);
    if (!runnable && !finalizers)
        return;                      /* keep the turn; nothing to interleave */

#ifdef USE_CUDA
    /* Phase 9b: a pipelined prefill chunk deliberately returns with die B's
     * segment still in flight.  This is the point where the GPU turn is handed
     * to somebody else, so drain both dies FIRST — explicitly, rather than
     * leaning on state_to_host's per-layer syncs, which skip any layer whose
     * GPU buffer is not currently this state's image and are therefore not a
     * guaranteed full drain.  ~1 µs on an idle stream, only on a real
     * handover, and a no-op when the pipeline is off (bs_slots_save_live
     * drains before it saves).
     *
     * Save the prefill so far into this session's OWN block on the card
     * BEFORE letting any other forward at the GPU: an interleaved prefill
     * reuses the shared SSM buffers and the single-user KV ring, and the
     * next chunk then re-seeds from the block device to device (eager mode:
     * from the host mirror, its legacy protocol). The slot is in PREFILL, so
     * no decode step touches its row meanwhile. On failure keep the turn —
     * keeping it is always safe (just no interleave). */
    if (bs_slots_save_live(sched, sched->slots[idx].state, idx) != 0) {
        fprintf(stderr, "[batch_sched] prefill_yield: saving the prefill "
                "failed (slot=%d); continuing without yield\n", idx);
        return;
    }
#endif

    pthread_mutex_lock(&sched->mu);
    const double held_ms = bs_now_ms() - sched->gpu_turn_t0_ms;
    const double window_ms =
        bs_prefill_decode_window_ms(held_ms, bs_prefill_decode_share());
    const uint64_t steps0 = sched->driver_steps;
    bs_gpu_release_locked(sched);
    pthread_cond_signal(&sched->cv_work);  /* the driver may be parked */
    bs_prefill_stand_aside_locked(sched, window_ms);
    const uint64_t steps = sched->driver_steps - steps0;
    /* Re-acquire FIFO: the driver, a queued finalize (and any earlier waiter)
     * go first — this is the chunk → step(s) → chunk alternation. */
    bs_gpu_acquire_locked(sched);
    pthread_mutex_unlock(&sched->mu);
    if (bs_stats_on())
        fprintf(stderr, "[bs] prefill_yield slot=%d chunk_ms=%.1f "
                "window_ms=%.1f decode_steps=%llu finalizers=%d\n", idx,
                held_ms, window_ms, (unsigned long long)steps, finalizers);
}

bool bs_pd_overlap_on(const pn_batch_sched *sched)
{
#ifdef USE_CUDA
    return sched && sched->driver_sample && pn_qwen35moe_pd_overlap_enabled();
#else
    (void)sched;
    return false;
#endif
}

bool bs_pd_shorter_waiting(pn_batch_sched *sched, int remaining)
{
    if (!sched)
        return false;
    bool shorter = false;
    pthread_mutex_lock(&sched->mu);
    for (int i = 0; i < sched->cap && !shorter; i++) {
        const int est = sched->slots[i].pf_wait_est;
        shorter = est > 0 && est + BS_PD_SRPT_MARGIN < remaining;
    }
    pthread_mutex_unlock(&sched->mu);
    return shorter;
}

int bs_pd_finalize_waiting(pn_batch_sched *sched)
{
    if (!sched)
        return 0;
    pthread_mutex_lock(&sched->mu);
    const int n = sched->n_finalize_waiting;
    pthread_mutex_unlock(&sched->mu);
    return n;
}

void bs_pd_yield(pn_batch_sched *sched, int idx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
#ifdef USE_CUDA
    /* Same save as bs_prefill_yield: any forward that runs before this turn
     * comes back may claim the live buffers; the next piece re-seeds from the
     * slot's own block. On failure keep the turn (always safe). */
    if (bs_slots_save_live(sched, sched->slots[idx].state, idx) != 0) {
        fprintf(stderr, "[batch_sched] pd_yield: saving the prefill failed (slot=%d); "
                "continuing without yield\n", idx);
        return;
    }
#endif
    pthread_mutex_lock(&sched->mu);
    bs_gpu_release_locked(sched);
    pthread_cond_signal(&sched->cv_work);   /* the driver may be parked */
    bs_gpu_acquire_locked(sched);           /* FIFO: every earlier waiter goes first */
    pthread_mutex_unlock(&sched->mu);
}

#ifndef USE_CUDA
int bs_pd_cycle(pn_batch_sched *sched, void *cursor)
{
    (void)sched; (void)cursor;
    return 1;
}
#endif

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

void bs_prefill_abort_keep(pn_batch_sched *sched, int idx, const char *sid,
                           const uint32_t *prompt, int n_prompt, int sys_len)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    bs_slot *sl = &sched->slots[idx];
    int kept = bs_partial_prefill_keep(sched, idx, sid, prompt, n_prompt,
                                       sys_len) == 0;
    if (!kept && (!bs_turn_ckpt_enabled() || !sl->ckpt.valid)) {
        bs_prefill_abort_reset(sched, idx);
        return;
    }
    /* No committed history describes the state (the prefill stopped
     * part-way); the kept prefix and the checkpoint do. Keep the session id
     * so the next turn is routed here. */
    pthread_mutex_lock(&sched->mu);
    snprintf(sl->sid, sizeof(sl->sid), "%s",
             kept ? sl->partial.sid : sl->ckpt.sid);
    sl->hist_n = 0;
    bs_slot_clear_projections(sl);
    pthread_mutex_unlock(&sched->mu);
    sl->ct_keep = 0;
    sl->ct_prompt_from = 0;
    if (sl->ckpt.valid)
        fprintf(stderr, "[batch_sched] slot %d: prefill aborted; keeping the "
                "turn checkpoint of sid=%s (%d tokens)\n", idx, sl->ckpt.sid,
                sl->ckpt.client_n);
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

    /* Eager mode (QWEN35_HOST_KV_MIRROR=1) keeps its legacy host capture.
     * Lazy mode needs none: the bind below snapshots the live SSM and tops up
     * the prefilled KV rows device-to-device into this slot's OWN block (GPU
     * row idx, carved at bs_create), so a session's state never visits host
     * RAM between its turns. */
    if (pn_qwen35moe_host_kv_mirror_eager() &&
        bs_slots_state_to_host(sched, sched->slots[idx].state) != 0) {
        fprintf(stderr, "[batch_sched] state_to_host failed (slot=%d)\n", idx);
        goto fail;
    }

    /* The slot's block is GPU row idx for its whole life. */
    const int gpu_slot = idx;
    if (bs_slots_bind(sched, sched->slots[idx].state, gpu_slot) != 0) {
        fprintf(stderr, "[batch_sched] slot_bind failed (slot=%d)\n", idx);
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
            sl->reason_end_at = 0;
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
            sl->reason_end_at = 0;
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
        sl->req_active_min = 0;
        sl->req_active_max = 0;
        memset(&sl->host, 0, sizeof(sl->host));
        /* MTP speculation for this request: a fresh depth controller and window. */
        sl->pd_stepped = false;
        sl->mtp_on = sched->mtp && sp &&
                     sg_mtp_cuda_begin(&sl->mtp, sched->model, max_recent) == 0;
        if (sl->mtp_on)
            bs_spec_arm_controller(sl);
        const char *plain_why = sp ? bs_spec_plain_reason(sched, sp) : NULL;
        if (plain_why)
            fprintf(stderr, "[spec] census: mode=%s path=batched slot=%d\n", plain_why, idx);
#ifdef USE_CUDA
        else if (sl->mtp_on && sp->tool_grammar && bs_spec_grammar_draws(sp->tool_grammar))
            fprintf(stderr, "[spec] census: mode=mtp-grammar path=batched slot=%d\n", idx);
#endif
        sl->spec_live_pos = -1;
        sl->status       = BS_SLOT_BOUND;
    }
    /* A new user is joining the active set — the driver must rebuild the
     * compacted GPU map before the next step to include this slot. */
    sched->active_set_changed = true;
    pthread_cond_signal(&sched->cv_work);   /* gate may open */
    /* The slot left PREFILL: a same-session resend waiting on it (see
     * bs_slot_acquire_or_wait) stops waiting. */
    pthread_cond_broadcast(&sched->cv_idle);
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
                                  const pn_reasoning_plan *plan, bool in_think,
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
    sl->reason_budget   = plan && plan->hard > 0 ? plan->hard : 0;
    sl->reason_soft     = sl->reason_budget > 0 ? plan->soft : 0;
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
    /* Bounded by the decode watchdog (batch_sched_watchdog.h): a step that
     * never comes back must not hold the slot and its stream forever. */
    int stall_ms = bs_decode_stall_ms();
    struct timespec until;
    make_deadline(&until, stall_ms);
    while (!sl->logits_ready && !sched->shutdown) {
        if (stall_ms <= 0) {
            pthread_cond_wait(&sl->cv, &sched->mu);
        } else if (pthread_cond_timedwait(&sl->cv, &sched->mu, &until) == ETIMEDOUT &&
                   !sl->logits_ready) {
            fprintf(stderr, "[batch_sched] slot %d: no logits for %d ms - step stalled, "
                    "the owner releases the slot\n", sl->slot_idx, stall_ms);
            pthread_mutex_unlock(&sched->mu);
            return -1;
        }
    }

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

/* PN_SERVE_TOKEN_HASH=1 (exactness gates): one line per released slot with an FNV-1a over
 * the token ids it emitted, so two serve runs compare token for token, not by text. */
static bool bs_token_hash_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_SERVE_TOKEN_HASH");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

static void bs_report_token_hash(const bs_slot *sl, int idx)
{
    uint64_t h = 1469598103934665603ULL;

    for (int i = 0; sl->ring && i < sl->ring_w; i++)
        for (int b = 0; b < 4; b++)
            h = (h ^ ((sl->ring[i] >> (8 * b)) & 0xffu)) * 1099511628211ULL;
    fprintf(stderr, "[serve] token hash slot=%d sid=%s n=%d fnv1a=%016llx\n", idx,
            sl->sid[0] ? sl->sid : "-", sl->ring_w, (unsigned long long)h);
}

void bs_slot_release(pn_batch_sched *sched, int idx)
{
    bs_slot_release_stats(sched, idx, NULL);
}

void bs_slot_release_stats(pn_batch_sched *sched, int idx, bs_request_stats *out)
{
    if (out)
        memset(out, 0, sizeof(*out));
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    pthread_mutex_lock(&sched->mu);
    bs_slot *sl = &sched->slots[idx];
    if (out && sl->status != BS_SLOT_IDLE) {
        out->active_min = sl->req_active_min;
        out->active_max = sl->req_active_max;
        out->host = sl->host;
#ifdef USE_CUDA
        out->mtp_cycles = sl->mtp_on ? sl->mtp.cycles : 0;
#endif
    }
    /* Only the owner calls release, and it does so exactly once, so any
     * non-IDLE status here belongs to this owner: free the slot directly.
     * The driver never retires slots on its own — a mid-step release just
     * means this lane's logits are discarded at deposit time.  (The driver
     * only touches the L3 ring/recent under mu with status == DECODE, so
     * freeing them here cannot race a deposit.) */
    bool was_active = (sl->status == BS_SLOT_BOUND ||
                       sl->status == BS_SLOT_DECODE);
    if (sl->status != BS_SLOT_IDLE) {
        if (bs_token_hash_on())
            bs_report_token_hash(sl, idx);
#ifdef USE_CUDA
        if (sl->mtp_on && bs_spec_shadow_on())
            bs_spec_shadow_report(sl, idx);
        if (sl->mtp_on)
            sg_mtp_cuda_report(&sl->mtp, sl->ring_w);
        sl->mtp_on = false;
#endif
        sl->status       = BS_SLOT_IDLE;
        sl->claim_sid[0] = '\0';   /* the slot's warm fields say who holds it now */
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
        sl->reason_end_at = 0;
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
    sl->last_active_ms = bs_now_ms();   /* LRU key for try_claim_idle */
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

/* Is a slot still prefilling a request of `sid`?  Under mu. */
static bool same_session_prefilling_locked(const pn_batch_sched *sched,
                                           const char *sid)
{
    if (!sid || !sid[0])
        return false;
    for (int i = 0; i < sched->cap; i++) {
        const bs_slot *sl = &sched->slots[i];
        if (sl->status == BS_SLOT_PREFILL &&
            strncmp(sl->claim_sid, sid, sizeof(sl->claim_sid)) == 0)
            return true;
    }
    return false;
}

/* PN_SAME_SESSION_WAIT_MS (default 30000; 0 = off), capped by wait_ms. */
static int same_session_wait_ms(int wait_ms)
{
    const char *env = getenv("PN_SAME_SESSION_WAIT_MS");
    int ms = (env && env[0]) ? atoi(env) : 30000;
    if (ms < 0)
        ms = 0;
    return ms < wait_ms ? ms : (wait_ms > 0 ? wait_ms : 0);
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

    /* A dead client's idle seat goes back before the seat count is read. */
    bs_admission_reap_idle(sched, sid, bs_now_ms());

    pthread_mutex_lock(&sched->mu);

    if (sched->shutdown) {
        pthread_mutex_unlock(&sched->mu);
        return BS_ACQUIRE_SHUTDOWN;
    }

    /* Session admission, atomically with the claim below (both under mu): a
     * refused session is told now, never queued behind seats that will not
     * free for it. */
    if (bs_admission_check_locked(sched, sid) != BS_ACQUIRE_OK) {
        pthread_mutex_unlock(&sched->mu);
        if (retry_after_s) *retry_after_s = 30;
        return BS_ACQUIRE_AT_CAPACITY;
    }

    /* A resend of a request this session is still prefilling waits for that
     * slot (see batch_sched.h): its kept prefix will be there. */
    int hold_ms = same_session_wait_ms(wait_ms);
    bool hold = hold_ms > 0 && sched->n_queued < sched->queue_cap &&
                same_session_prefilling_locked(sched, sid);
    struct timespec hold_until;
    make_deadline(&hold_until, hold_ms);

    /* Fast path: slot free and pressure OK — claim immediately. */
    if (!hold && sched->pressure_level < BS_PRESSURE_HOLD_LEVEL) {
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

        /* Other families may have taken the last seats while we waited. */
        if (bs_admission_check_locked(sched, sid) != BS_ACQUIRE_OK) {
            if (retry_after_s) *retry_after_s = 30;
            result = BS_ACQUIRE_AT_CAPACITY;
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

        /* Same-session resend: wait for the slot still prefilling it. */
        if (hold && same_session_prefilling_locked(sched, sid)) {
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            if (now.tv_sec < hold_until.tv_sec ||
                (now.tv_sec == hold_until.tv_sec &&
                 now.tv_nsec < hold_until.tv_nsec)) {
                (void)pthread_cond_timedwait(&sched->cv_idle, &sched->mu,
                                             &hold_until);
                continue;   /* re-check all conditions */
            }
            hold = false;   /* waited long enough: claim as before */
            fprintf(stderr, "[batch_sched] sid=%s: its earlier request is "
                    "still prefilling after %d ms; claiming another slot\n",
                    sid, hold_ms);
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
