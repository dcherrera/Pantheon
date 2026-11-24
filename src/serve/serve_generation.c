/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_generation.c
 * @brief Implementation of chat-completion generation (see serve_generation.h).
 *        Reuses the engine's forward pass (model->forward), the shared sampler
 *        (pn_sample_next_token), the qwen35 BPE encoder (pn_bpe_encode) and the
 *        GPT-2 byte decode used by the CLI/chat paths.
 *
 * @status Complete for CPU host serving. Splits <think> reasoning, answer
 *         content, and a <tool_call> block into separate result channels; tool
 *         results (role:"tool") are re-fed inside <tool_response> wrappers.
 *         GPU serving steps through sg_forward_step (C2): non-blocking
 *         launch + event-poll when the model supports it, so concurrent
 *         generations on different cards overlap from the 2-core host.
 *         Reasoning-budget force-close: BOTH paths cap AUTO-path reasoning at
 *         params->reasoning_budget tokens — once a think-style model spends that
 *         many tokens inside <think> without a self-emitted "</think>", the
 *         "</think>" close is force-decoded so the model concludes and answers
 *         (independent of the tool-forcing grammar).  The single-user path
 *         (sg_run_completion) does it inline in its decode loop; the batched
 *         path (sg_run_completion_batched) arms it PER SLOT in the scheduler's
 *         driver sampler via bs_slot_arm_reasoning_budget — each batched session
 *         tracks its own reasoning count + close-injection independently.
 *         THINK-ELISION (Phase 11): build_prompt reports the forced generation
 *         prefill length and the BATCHED decode loop counts the generated
 *         tokens that produce no client-visible byte, so the scheduler can
 *         resume a turn the client re-renders without its think block.
 *         REASONING-BEARING RESUME: the same loop also hands the scheduler this
 *         turn's think block re-rendered from the `reasoning_content` it
 *         streamed, so a client that sends reasoning BACK — the one that keeps
 *         the model reasoning — resumes too instead of re-prefilling.
 *         TOOL-CALL WRAPPER FIDELITY: an assistant history turn's "<tool_call>"
 *         / "</tool_call>" go back in as the SPECIAL ids the model emits, not as
 *         byte-encoded text (see TOOL_WRAPPER_MARKERS).
 *         REPETITION-LOOP GUARD: all THREE decode loops (single-user, batched
 *         driver-sampling, batched legacy lockstep) feed every generated token
 *         id to a per-turn cycle detector (repetition_loop_detector.h); a
 *         confirmed loop ends the turn with finish_reason
 *         SG_FINISH_REPETITION_LOOP and one "[repetition] STOP" log line.
 *         ON by default, PANTHEON_REPETITION_STOP=0 disables.
 * @issues The single-user path (sg_run_completion) has no think-elision — the
 *         two paths have diverged again.  The deployed fleet runs batched.
 *         A model that emits a BARE "<function=" with no wrapper (possible in
 *         AUTO mode, i.e. without the tool grammar) still gets a "<tool_call>"
 *         wrapper on re-render that it never generated, so that turn cannot
 *         append-resume — it fails safe to a full prefill, never a wrong one.
 * @todo None
 */

#include "serve_generation.h"
#include "context_over_budget.h"

#include "request_profile.h"
#include "request_timing_line.h"
#include "../core/host_stage_timer.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/bench.h"
#include "../core/bpe_pretokenize.h"
#include "../core/chat.h"
#include "../core/engine.h"
#include "../core/sampling_modifiers.h"
#include "../core/sampler_topn_surgery.h"
#include "../core/mem/pn_host_admission.h"
#include "../core/mem/pn_memwatch.h"
#include "batch_sched.h"
#include "batch_sched_watchdog.h"
#include "reasoning_budget.h"
#include "repetition_loop_detector.h"
#include "serve_dump_reqlog.h"
#include "serve_emitter.h"
#include "serve_prompt_tokens.h"
#include "serve_spec_lookup.h"
#include "serve_spec_census.h"
#include "serve_spec_mtp_cuda.h"
#include "grammar_spec_chooser.h"
#include "../core/spec_sampling_chooser.h"
#include "tool_grammar.h"
#include "tool_grammar_json.h"
#include "response_format_grammar.h"
#include "json_reader.h"
#include "tensor/tensor.h"
#include "tokenizer/tokenizer.h"

/* Unconditional: the constrained-decoding full-logits seam
 * (pn_qwen35moe_state_set_full_logits) is needed in every build, not only the
 * CUDA one — the flag is simply never read when there is no device top-k. */
#include "../models/qwen35moe.h"   /* pn_qwen35moe_forward_batch_cuda (L1) */
#include "../models/llama.h"       /* pn_llama_prefill_tt (Phase 7.8b) */

/* ============================================================
 * Prefill-throughput counter (K80 serving-ops Phase 7)
 *
 * A cheap running measure of prefill tokens/sec, fed from every real prefill
 * this file runs (delta tokens / wall-time around sg_prefill_range). The
 * restart-estimate endpoint reads it to quote a cold re-prefill time. Kept as
 * an exponential moving average behind a small mutex — prefills are already
 * serialized per model but different models share this file, so the lock keeps
 * the EMA update atomic. A tiny prefill (< 8 tokens, e.g. an append turn) is
 * ignored: too small to time meaningfully and it would skew the average.
 * ============================================================ */

static pthread_mutex_t sg_prefill_rate_lock = PTHREAD_MUTEX_INITIALIZER;
static double sg_prefill_rate_ema = 0.0;   /* tokens/sec; 0 = no sample yet */

static double sg_prefill_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Fold one prefill measurement (tokens over elapsed ms) into the EMA. */
static void sg_prefill_rate_record(int tokens, double elapsed_ms)
{
    if (tokens < 8 || elapsed_ms <= 0.0)
        return;
    double rate = (double)tokens / (elapsed_ms / 1000.0);
    pthread_mutex_lock(&sg_prefill_rate_lock);
    sg_prefill_rate_ema = sg_prefill_rate_ema <= 0.0
                        ? rate
                        : 0.7 * sg_prefill_rate_ema + 0.3 * rate;
    pthread_mutex_unlock(&sg_prefill_rate_lock);
}

double sg_prefill_tok_per_s_measured(void)
{
    pthread_mutex_lock(&sg_prefill_rate_lock);
    double r = sg_prefill_rate_ema;
    pthread_mutex_unlock(&sg_prefill_rate_lock);
    return r;
}

/* ============================================================
 * Reasoning / content / tool-call stream splitter
 *
 * The emitter (em_init/em_feed/em_flush/em_free) lives in serve_emitter.c —
 * it depends only on libc, so it is unit-tested in isolation
 * (tests/test_serve_emitter.c) without the engine/CUDA forward path.  It routes
 * <think> reasoning vs. answer content into separate channels and diverts a
 * tool call (opened by "<tool_call>" always, or a bare "<function=" when the
 * request carried tools) into a raw `tool` channel for structured parsing.
 * ============================================================ */

/* ============================================================
 * Detokenize one token id to UTF-8 text (BPE byte-decode when needed).
 * ============================================================ */

static char *decode_token(pn_model *model, uint32_t token_id)
{
    int id = (int)token_id;
    char *raw = ft_decode(model->tokenizer, &id, 1);
    if (!raw) return NULL;
    if (ft_get_model_type(model->tokenizer) == 0) {
        char *decoded = gpt2_bytes_decode(raw);
        free(raw);
        return decoded;
    }
    return raw;
}

/* Is this model's reasoning close ONE token whose text is "</think>"?  Then that
 * token is the close, and the emitter and grammars run in token mode
 * (think_close_rule.h): the same text spelled out of ordinary tokens — a model
 * writing a C string with "</think>" in it — stays reasoning. */
static bool sg_think_close_is_token(pn_model *model)
{
    int id = ft_token_to_id(model->tokenizer, "</think>");
    if (id < 0) return false;
    char *text = decode_token(model, (uint32_t)id);
    bool whole = text && strcmp(text, "</think>") == 0;
    free(text);
    return whole;
}

/* ============================================================
 * Reasoning-budget force-close (AUTO / non-forced path)
 *
 * A think-style model prefilled with "<think>" reasons until it emits "</think>"
 * on its own.  A reasoning model can run away — filling the whole generation cap
 * with reasoning and never producing an answer (the saber runaway: 1024 tokens of
 * reasoning_content, finish=length, empty content).  The grammar's forced-mode
 * TG_REASON_BUDGET solves this ONLY on the tool_choice REQUIRED/FUNCTION path;
 * an orchestration turn arrives as tool_choice AUTO, so that budget never fires.
 *
 * This is the general, tool-forcing-independent budget.  The decode loop counts
 * the tokens generated while INSIDE the <think> block (em.in_think); once they
 * exceed the per-request/per-model budget and the model has NOT self-closed, it
 * FORCE-DECODES the "</think>" close: each of its token ids is fed through
 * sg_forward_step (so the KV/SSM state advances over the close exactly as if the
 * model had emitted it) and through em_feed (so the emitter's own THINK_CLOSE
 * matching flips it into the content channel — no flag-poking).  Normal sampling
 * then resumes and the model answers.  Disabled (budget <= 0) → no-op.
 * ============================================================ */

/* Tokenize the "</think>" close for this model into `out` (capacity `cap`),
 * returning the count (0 on failure).  Mirrors build_prompt's tokenization: a
 * think-style model has "</think>" as a single special token, but fall back to
 * text-encoding it so the primitive is robust to a tokenizer that lacks it. */
static int sg_encode_think_close(pn_model *model, uint32_t *out, int cap)
{
    ft_tokenizer *tok = model->tokenizer;
    int special = ft_token_to_id(tok, "</think>");
    if (special >= 0) {
        if (cap < 1) return 0;
        out[0] = (uint32_t)special;
        return 1;
    }
    int n = 0;
    int *ids = (ft_get_model_type(tok) == 0)
                   ? pn_bpe_encode(tok, "</think>", &n)
                   : ft_encode(tok, "</think>", &n);
    if (!ids) return 0;
    int m = 0;
    for (int i = 0; i < n && m < cap; i++)
        if (ids[i] >= 0) out[m++] = (uint32_t)ids[i];
    free(ids);
    return m;
}

/* The single-user resume ceiling for the request's `think_history`
 * (core/think_history.h): with as_sent the prefix cache must not resume past
 * the request's first history think block (the forced generation prefill
 * excluded). drop_last tries the turn snapshot first (pn_prefix_acquire_opts)
 * and otherwise takes the same ceiling. KEEP: no ceiling. */
static int sg_think_history_ceiling(pn_model *model, const uint32_t *prompt,
                                    int n_prompt, int think_prefill_len,
                                    pn_think_history think_history)
{
    if (think_history == PN_THINK_HISTORY_KEEP)
        return n_prompt;
    int n_history = n_prompt - think_prefill_len;
    return pn_think_history_first_think(prompt, n_history,
                                        ft_token_to_id(model->tokenizer,
                                                       "<think>"));
}

/* Push one force-decoded token into the repeat-penalty window (same order the
 * decode loop appends a sampled token). */
static void sg_recent_push(uint32_t *recent, int max_recent, int *n_recent,
                           uint32_t tok)
{
    if (max_recent <= 0) return;
    if (*n_recent < max_recent) {
        recent[(*n_recent)++] = tok;
    } else {
        memmove(recent, recent + 1, (size_t)(max_recent - 1) * sizeof(uint32_t));
        recent[max_recent - 1] = tok;
    }
}

/* ============================================================
 * Repetition-loop guard
 *
 * A heavily quantised model at long context can stop producing new text and
 * re-emit one paragraph until the token budget is gone — measured on saber-orch
 * (Q3_K_M) at ~40k context as a 14,535-byte answer whose distinct content was
 * ~470 bytes, cycled 31 times, reported to the caller as a normal finish.
 *
 * The detector itself is libc-only and lives in repetition_loop_detector.c (see
 * that header for the thresholds and the false-positive reasoning); this file
 * only feeds it the generated token ids and, on a trip, ends the turn with
 * SG_FINISH_REPETITION_LOOP.  It is armed the same way in ALL THREE decode
 * loops — single-user, batched driver-sampling (what the fleet runs), and the
 * batched legacy lockstep path — because a guard on one path and not the others
 * is how this file has diverged before.
 * ============================================================ */

/* Arm the per-turn detector.  Left INERT when PANTHEON_REPETITION_STOP=0 or the
 * ring allocation fails: an unguarded turn is the pre-existing behaviour, so a
 * detector that cannot start must never be able to fail a completion. */
static void sg_reploop_arm(rld_detector *rd)
{
    memset(rd, 0, sizeof(*rd));
    if (rld_enabled())
        rld_init(rd, NULL);
}

/* OBSERVABILITY — a repetition collapse is an ABNORMAL event, so it logs
 * unconditionally (same rule as the reasoning force-close above), naming the
 * model, the decode path, and the shape of the cycle.  One greppable line, so a
 * run can be counted: `grep -c '\[repetition\] STOP' serve.log`. */
static void sg_reploop_log(const rld_detector *rd, const sg_gen_params *params,
                           const char *path, int generated)
{
    fprintf(stderr, "[repetition] STOP model=%s path=%s cycle_tokens=%d "
            "repeats=%d generated=%d\n",
            (params->model_name && params->model_name[0])
                ? params->model_name : "?",
            path, rd->cycle_tokens, rd->repeats, generated);
}

/* ============================================================
 * Phase 5 — grammar-constrained tool decoding glue
 *
 * Enabled when sg_tool_grammar_arms() says so (tool_choice "required" by
 * default; every tool request with PANTHEON_TOOL_GRAMMAR=1; nothing with =0);
 * otherwise sg_tg_setup returns a zeroed (inert) context and the sampler hook
 * pointer stays NULL, so generation is byte-identical to before.
 *
 * The hook needs each candidate token's DECODED UTF-8 text.  Decoding per-token
 * per-step (≈248k × a malloc) would be far too costly, so we PRECOMPUTE a
 * per-vocab piece table ONCE per request (n_vocab decoded strings) and the
 * hook's token_text() is then an O(1) array lookup with no allocation in the
 * hot sampling loop.
 *
 * Advance vs. mask split.  The serve loop ADVANCES the grammar itself every
 * step (with the already-decoded piece), so the FSM tracks the opener while in
 * the free reasoning/prose region.  The sampler hook only MASKS (advance is
 * left NULL), and its `active` flag is refreshed each step from tg_mask_active():
 * 0 while FREE (no mask applied, so a legit prose answer is never forced into a
 * call AND the 248k scan is skipped entirely), flipping to 1 the instant the
 * grammar has consumed a tool-call opener and entered the skeleton.
 *
 * FORCED-CALL mode (tool_choice "required" / a named function).  The request's
 * resolved tool_choice picks tg_state_new_forced instead of tg_state_new; the
 * grammar then REQUIRES the opener once reasoning ends.  Reasoning-end is driven
 * by the grammar itself, which SELF-DETECTS "</think>" in the raw token stream it
 * already advances over (BUG 1 fix) — it no longer depends on the emitter's
 * <think> flag, which the Q3 model does not reliably trigger (it sometimes
 * reasons in the content channel with no clean </think>, so forcing never fired).
 * sg_tg_reasoning_ended() is still called off the emitter's <think>-close as a
 * harmless SECONDARY trigger (whichever fires first engages the constraint).
 * In forced mode tg_mask_active() is also true in the post-reasoning FREE state,
 * so prose/JSON are masked out.  AUTO mode is unchanged (and byte-identical
 * unless PANTHEON_TOOL_GRAMMAR=1 — the hook pointer stays NULL).
 * ============================================================ */

/* Does this request arm the tool-call grammar?  PANTHEON_TOOL_GRAMMAR:
 *   "0"            off for every request (explicit opt-out);
 *   "1"            every request that carries tools (AUTO engages on an opener,
 *                  "required"/named function compel a call) — the old opt-in;
 *   unset / other  only tool_choice "required": the client asked for a call
 *                  every turn, and without the grammar a model that ends in
 *                  narration hands it a turn with no call at all.
 * A named function stays behind "1", unchanged. */
bool sg_tool_grammar_arms(const sg_gen_params *params)
{
    if (!params || !params->tools) return false;
    const char *env = getenv("PANTHEON_TOOL_GRAMMAR");
    if (env && env[0] == '0') return false;
    if (env && env[0] == '1') return true;
    return params->tool_choice == SG_TOOL_CHOICE_REQUIRED;
}

/* Which armed grammars speculate on the Tenstorrent loop (grammar_spec_chooser.h):
 * the turn keeps the top-N rows and the drafts, every verify row is chosen as plain
 * constrained decoding chooses it, and a masked step the top-N cannot decide refills
 * the dense row (row 0) or ends the cycle (a later row).  "required" since
 * 2026-10-13; `PN_TT_SPEC_GRAMMAR` (default 1) extends it to every armed shape — an
 * AUTO or named-function tool grammar (PANTHEON_TOOL_GRAMMAR=1) and response_format
 * json_object.  `PN_TT_SPEC_GRAMMAR=0`: only "required"; the others keep the dense row
 * and plain decode, exactly as before. */
#ifdef USE_TT
static bool sg_spec_grammar_all(void)
{
    const char *env = getenv("PN_TT_SPEC_GRAMMAR");
    return !(env && atoi(env) == 0);
}

static bool sg_tool_grammar_speculates(const sg_gen_params *params)
{
    return sg_tool_grammar_arms(params) &&
           (params->tool_choice == SG_TOOL_CHOICE_REQUIRED || sg_spec_grammar_all());
}

static bool sg_response_format_speculates(const sg_gen_params *params)
{
    return params->response_format == SG_RESPONSE_FORMAT_JSON_OBJECT && sg_spec_grammar_all();
}
#endif

/* The think-gated "required" walk exists for every dialect a grammar models
 * (tgj_state_new_required for ORNITH_JSON, tg_state_new_required for QWEN_XML);
 * MiniCPM's attribute form is not modelled, so it cannot claim the guarantee. */
bool sg_tool_choice_required_with_thinking(tc_dialect dialect)
{
    const char *env = getenv("PANTHEON_TOOL_GRAMMAR");
    return tc_dialect_grammar_models(dialect) && !(env && env[0] == '0');
}

/* Which grammar is driving this request's constrained decode.  EXACTLY ONE is
 * ever live: the tool grammars constrain a tool CALL, the response-format
 * grammar constrains the ANSWER, and a single sampler hook cannot serve two
 * masks.  The HTTP layer resolves the precedence (tools win) before we get here,
 * so this is a plain three-way selector, not a conflict. */
typedef enum {
    SG_GRAMMAR_NONE = 0,
    SG_GRAMMAR_TOOL_XML,       /* tool_grammar.c      — Qwen/N2/MiniCPM dialects */
    SG_GRAMMAR_TOOL_JSON,      /* tool_grammar_json.c — ornith's JSON dialect    */
    SG_GRAMMAR_RESPONSE_JSON,  /* response_format_grammar.c — json_object answer */
} sg_grammar_kind;

/* The constrained-decoding context: the live grammar, its walk state, the
 * per-vocab decoded-piece table the mask needs, and the sampler hook the engine
 * calls.  All the sg_tg_* helpers below dispatch on `kind` so the serve loops
 * (single-user + batched) drive any grammar unchanged. */
typedef struct {
    sg_grammar_kind kind;
    tg_grammar *grammar;      /* XML dialect */
    tg_state   *state;
    tgj_grammar *grammar_json;/* JSON dialect (ornith) */
    tgj_state   *state_json;
    rfg_state  *state_rf;     /* response_format json_object (no grammar object:
                               * the shape is fixed, nothing to compile) */
    char      **pieces;       /* [n_vocab] decoded UTF-8 text per token id */
    int         n_vocab;
    pn_tool_grammar_hook hook;
    bool        enabled;
    bool        forced;       /* FORCED walk (tool_choice required/function) —
                               * reported alongside a DEAD verdict, because AUTO
                               * and FORCED fail in opposite directions and the
                               * log has to say which one this turn was */
    bool        verbose;      /* PANTHEON_TOOL_GRAMMAR_DEBUG=1 → per-step trace */
    /* ── Full-logits arming window (see the block comment above
     * sg_arm_full_logits).  `fl_always` is a turn-lifetime requirement that has
     * nothing to do with the grammar walk (logit_bias / logprobs); `fl_pending`
     * is the one-shot latch that covers the reasoning force-close, the only
     * place a grammar mutation lands after the forward it should have preceded. */
    bool        fl_always;
    bool        fl_pending;
    /* Engine mask-density counters as they stood when this turn STARTED, so
     * the turn can report its own delta rather than a process running total
     * (see sg_tg_report_mask_density). */
    unsigned long fl_mask_steps0;
    unsigned long fl_sparse_rows0;
    unsigned long fl_shortlist_steps0;
    unsigned long fl_bound_steps0;
    unsigned long fl_chunk_steps0;
    unsigned long fl_fullscan_steps0;
    double fl_mask_seconds0;
} sg_tg_ctx;

/* True when the live grammar is one of the tool-call grammars. */
static bool sg_tg_is_tool(const sg_tg_ctx *c) {
    return c->kind == SG_GRAMMAR_TOOL_XML || c->kind == SG_GRAMMAR_TOOL_JSON;
}

static const char *sg_tg_token_text(void *user, uint32_t id) {
    sg_tg_ctx *c = (sg_tg_ctx *)user;
    if ((int)id >= c->n_vocab) return NULL;
    return c->pieces[id];
}

static int sg_tg_allowed(void *st, const char *text) {
    return tg_token_allowed((const tg_state *)st, text) ? 1 : 0;
}
static int sg_tg_allowed_json(void *st, const char *text) {
    return tgj_token_allowed((const tgj_state *)st, text) ? 1 : 0;
}
/* pn_tool_grammar_hook.steer, armed only when the request sent field_patterns:
 * a refused close on a patterned value becomes its forced continuation
 * (tg_steer / tgj_steer). */
static int sg_tg_steer(void *st, const char *text) {
    return tg_steer((tg_state *)st, text) ? 1 : 0;
}
static int sg_tg_steer_json(void *st, const char *text) {
    return tgj_steer((tgj_state *)st, text) ? 1 : 0;
}

/* Driver-side callbacks (batched path).  The batched driver samples per-slot in
 * its own thread from the slot's copied sampling params, so the grammar must
 * advance + gate + report completion FROM the sampler itself (the single-user
 * loop does these externally with advance == NULL).  These three make the hook
 * self-contained for the driver. */
static void sg_tg_advance_cb(void *st, const char *text) {
    tg_advance((tg_state *)st, text);
}
static int sg_tg_refresh_cb(void *st) {
    return tg_mask_active((const tg_state *)st) ? 1 : 0;
}
static int sg_tg_complete_cb(void *st) {
    return tg_complete((const tg_state *)st) ? 1 : 0;
}
static void sg_tg_advance_cb_json(void *st, const char *text) {
    tgj_advance((tgj_state *)st, text);
}
static int sg_tg_refresh_cb_json(void *st) {
    return tgj_mask_active((const tgj_state *)st) ? 1 : 0;
}
static int sg_tg_complete_cb_json(void *st) {
    return tgj_complete((const tgj_state *)st) ? 1 : 0;
}
static int sg_tg_in_call_cb(void *st) {
    return tg_in_call((const tg_state *)st) ? 1 : 0;
}
/* pn_tool_grammar_hook.clone / clone_free (the batched MTP driver's look-ahead). On the
 * batched path only the sampler (and the driver's injected close) moves a tool walk, so a
 * clone advanced over the drafts predicts the mask exactly, FORCED walks included. */
static void *sg_tg_clone_cb(void *st) { return tg_state_clone((const tg_state *)st); }
static void sg_tg_clone_free_cb(void *st) { tg_state_free((tg_state *)st); }
static void *sg_tg_clone_cb_json(void *st) { return tgj_state_clone((const tgj_state *)st); }
static void sg_tg_clone_free_cb_json(void *st) { tgj_state_free((tgj_state *)st); }
/* Batched-driver callback (pn_tool_grammar_hook.reasoning_end_injected): the
 * driver finished decoding the reasoning-budget's injected "</think>" and must
 * tell the grammar that close was NOT the model's, so the opener is not compelled
 * around an unfinished thought.  XML dialect only — see
 * sg_tg_reasoning_ended_injected for why the JSON dialect has no window. */
static void sg_tg_reasoning_end_injected_cb(void *st) {
    tg_reasoning_end_injected((tg_state *)st);
}
/* pn_tool_grammar_hook.reasoning_ended: sg_tg_forced_ended for the batched driver, which
 * records where it first holds (bs_slot_reasoning_ended). */
static int sg_tg_reasoning_ended_cb(void *st) {
    return tg_forced_reasoning_ended((const tg_state *)st) ? 1 : 0;
}
static int sg_tg_reasoning_ended_cb_json(void *st) {
    return tgj_forced_reasoning_ended((const tgj_state *)st) ? 1 : 0;
}
static int sg_tg_end_allowed_json(void *st) {
    return tgj_end_allowed((const tgj_state *)st) ? 1 : 0;
}
static int sg_tg_end_allowed_xml(void *st) {
    return tg_end_allowed((const tg_state *)st) ? 1 : 0;
}
static int sg_tg_in_call_cb_json(void *st) {
    return tgj_in_call((const tgj_state *)st) ? 1 : 0;
}
static int sg_tg_allowed_rf(void *st, const char *text) {
    return rfg_token_allowed((const rfg_state *)st, text) ? 1 : 0;
}
static void sg_tg_advance_cb_rf(void *st, const char *text) {
    rfg_advance((rfg_state *)st, text);
}
static int sg_tg_refresh_cb_rf(void *st) {
    return rfg_mask_active((const rfg_state *)st) ? 1 : 0;
}
static int sg_tg_complete_cb_rf(void *st) {
    return rfg_complete((const rfg_state *)st) ? 1 : 0;
}
static int sg_tg_in_call_cb_rf(void *st) {
    return rfg_in_json((const rfg_state *)st) ? 1 : 0;
}

/* Build the ornith JSON grammar + walk state (dialect TC_DIALECT_ORNITH_JSON).
 * Sets c->grammar_json / c->state_json and points the hook at the JSON callbacks.
 * Returns true if the grammar built, false to leave the feature off (inert). */
static bool sg_tg_build_json(sg_tg_ctx *c, pn_model *model, const sg_gen_params *params,
                             bool *forced_out) {
    c->grammar_json = tgj_build((const jr_value *)params->tools);
    if (!c->grammar_json) return false;
    if (!tgj_require_if_reasoned(c->grammar_json,
                                 (const jr_value *)params->required_if_reasoned))
        fprintf(stderr, "[tool_grammar] required_if_reasoned: OOM, keys left optional\n");
    if (!tgj_apply_field_patterns(c->grammar_json, (const jr_value *)params->field_patterns))
        fprintf(stderr, "[tool_grammar] field_patterns: a pattern did not build, "
                "some fields left unconstrained\n");
    bool forced = false;
    bool think_open = sg_chat_style_is_think(model->chat_style) &&
                      !sg_thinking_suppressed(params->thinking);
    if (params->tool_choice == SG_TOOL_CHOICE_REQUIRED) {
        /* THINK-GATED: the think block the prompt opened is never constrained;
         * the call is compelled from the token after the model's "</think>"
         * (or from token 0 when no block was opened). */
        c->state_json = tgj_state_new_required(c->grammar_json, think_open);
        forced = (c->state_json != NULL);
    } else if (params->tool_choice == SG_TOOL_CHOICE_FUNCTION) {
        c->state_json = tgj_state_new_forced(c->grammar_json, params->forced_name);
        forced = (c->state_json != NULL);
    } else {
        c->state_json = tgj_state_new(c->grammar_json);
    }
    if (!c->state_json && params->tool_choice == SG_TOOL_CHOICE_FUNCTION)
        c->state_json = tgj_state_new(c->grammar_json);
    if (!c->state_json) {
        tgj_free(c->grammar_json); c->grammar_json = NULL; return false;
    }
    /* AUTO / named function: watch the open think block too, so `reasoned` is
     * known for required_if_reasoned (the walk is otherwise unchanged). */
    if (think_open && params->tool_choice != SG_TOOL_CHOICE_REQUIRED)
        tgj_watch_reasoning(c->state_json);
    c->kind = SG_GRAMMAR_TOOL_JSON;
    c->hook.st      = c->state_json;
    c->hook.allowed = sg_tg_allowed_json;
    if (params->field_patterns) c->hook.steer = sg_tg_steer_json;
    *forced_out = forced;
    return true;
}

/* Build the XML grammar + walk state (Qwen/N2/MiniCPM dialects).  Same shape as
 * the JSON builder above, "required" think-gated the same way. */
static bool sg_tg_build_xml(sg_tg_ctx *c, pn_model *model, const sg_gen_params *params,
                            bool *forced_out) {
    c->grammar = tg_build((const jr_value *)params->tools);
    if (!c->grammar) return false;
    tg_require_if_reasoned(c->grammar, (const jr_value *)params->required_if_reasoned);
    if (!tg_apply_field_patterns(c->grammar, (const jr_value *)params->field_patterns))
        fprintf(stderr, "[tool_grammar] field_patterns: a pattern did not build, "
                "some fields left unconstrained\n");
    bool forced = false;
    bool think_open = sg_chat_style_is_think(model->chat_style) &&
                      !sg_thinking_suppressed(params->thinking);
    if (params->tool_choice == SG_TOOL_CHOICE_REQUIRED) {
        c->state = tg_state_new_required(c->grammar, think_open);
        forced = (c->state != NULL);
    } else if (params->tool_choice == SG_TOOL_CHOICE_FUNCTION) {
        c->state = tg_state_new_forced(c->grammar, params->forced_name);
        forced = (c->state != NULL);
    } else {
        c->state = tg_state_new(c->grammar);
    }
    if (!c->state && params->tool_choice == SG_TOOL_CHOICE_FUNCTION)
        c->state = tg_state_new(c->grammar);
    if (!c->state) { tg_free(c->grammar); c->grammar = NULL; return false; }
    if (think_open && params->tool_choice != SG_TOOL_CHOICE_REQUIRED)
        tg_watch_reasoning(c->state);
    c->kind = SG_GRAMMAR_TOOL_XML;
    c->hook.st      = c->state;
    c->hook.allowed = sg_tg_allowed;
    if (params->field_patterns) c->hook.steer = sg_tg_steer;
    *forced_out = forced;
    return true;
}

/* Build the response-format grammar: the ANSWER must be one valid JSON object.
 *
 * `reasoning_grace` mirrors what the prompt actually did — a think-style model
 * is prefilled with an open <think>, so its first tokens are reasoning and must
 * stay unconstrained until "</think>".  With No-Think on, the block was prefilled
 * CLOSED and the very first token is already the answer, so the constraint
 * engages immediately. */
static bool sg_tg_build_response_format(sg_tg_ctx *c, pn_model *model,
                                        const sg_gen_params *params) {
    bool reasoning_grace = sg_chat_style_is_think(model->chat_style) &&
                           !sg_thinking_suppressed(params->thinking);
    c->state_rf = rfg_state_new(RFG_FORMAT_JSON_OBJECT, reasoning_grace);
    if (!c->state_rf) return false;
    c->kind         = SG_GRAMMAR_RESPONSE_JSON;
    c->hook.st      = c->state_rf;
    c->hook.allowed = sg_tg_allowed_rf;
    return true;
}

/* The end-of-turn ids must not be sampleable while the JSON object is still
 * open, or the model can stop mid-object and hand the client truncated,
 * un-parseable content — the exact silent breakage response_format exists to
 * prevent.  The mask only ever sees a token's TEXT, and these decode to ""
 * (which the sampler treats as always-allowed), so their piece-table entries are
 * replaced with a sentinel the JSON grammar rejects.  The mask is inactive
 * during reasoning and once the object closes, so EOS is sampleable exactly
 * where it should be.  Tool grammars keep the untouched table (their forced mode
 * relies on being able to force EOS as a clean exit); the "required" JSON walk
 * governs its end tokens through hook.end_allowed instead. */
static void sg_tg_mask_end_tokens(sg_tg_ctx *c, pn_model *model) {
    static const char SENTINEL[] = "\x01";   /* never a valid JSON continuation */
    const int end_ids[2] = { ft_eos_id(model->tokenizer),
                             ft_token_to_id(model->tokenizer, "<|im_end|>") };
    for (int i = 0; i < 2; i++) {
        int id = end_ids[i];
        if (id < 0 || id >= c->n_vocab) continue;
        char *sentinel = strdup(SENTINEL);
        if (!sentinel) continue;
        free(c->pieces[id]);
        c->pieces[id] = sentinel;
    }
}

/* "required": the end ids must reach the mask as tokens WITH NO TEXT, the only
 * shape hook.end_allowed governs.  The tokenizer decodes them to their literal
 * text ("<|im_end|>"), which a free-text value accepts as content: the end rule
 * then never fired inside a value, and the batched driver advanced the walk
 * over "<|im_end|>" as value bytes — so a floored value the client never saw
 * filled counted as filled and the serve closed `bash {"command":""}` (N2,
 * K80, closer probe).  Blank both; advancing over "" is a no-op. */
static void sg_tg_blank_end_tokens(sg_tg_ctx *c, pn_model *model) {
    const int end_ids[2] = { ft_eos_id(model->tokenizer),
                             ft_token_to_id(model->tokenizer, "<|im_end|>") };
    for (int i = 0; i < 2; i++) {
        int id = end_ids[i];
        if (id < 0 || id >= c->n_vocab) continue;
        char *blank = strdup("");
        if (!blank) continue;
        free(c->pieces[id]);
        c->pieces[id] = blank;
    }
}

/* Build the grammar context for this request, or leave it inert. Returns 0
 * always (a build failure just leaves the feature off — never regress serve).
 * Selects the grammar by dialect: TC_DIALECT_ORNITH_JSON → the ornith JSON
 * grammar (which honors `required` natively, so ornith runs FORCED like saber);
 * every XML dialect → the trained XML skeleton grammar. */
static void sg_tg_setup(sg_tg_ctx *c, pn_model *model,
                        const sg_gen_params *params, int n_vocab) {
    memset(c, 0, sizeof(*c));

    /* Tool-call constraint — see sg_tool_grammar_arms for the policy. */
    const char *env = getenv("PANTHEON_TOOL_GRAMMAR");
    bool tool_grammar_on = sg_tool_grammar_arms(params);
    /* Response-format constraint: NOT env-gated.  `response_format` is a
     * standard OpenAI field a client sends expecting a guarantee; honouring it
     * only when an operator happens to have set an env var would be the same
     * silent degradation as ignoring it outright.  The HTTP layer has already
     * resolved tools-vs-response_format, so both are never on at once. */
    bool response_format_on =
        (params->response_format == SG_RESPONSE_FORMAT_JSON_OBJECT);

    bool forced = false;
    bool built = false;
    if (tool_grammar_on) {
        built = (params->dialect == TC_DIALECT_ORNITH_JSON)
                    ? sg_tg_build_json(c, model, params, &forced)
                    : sg_tg_build_xml(c, model, params, &forced);
    } else if (response_format_on) {
        built = sg_tg_build_response_format(c, model, params);
    }
    if (!built) {
        /* SAY SO, ONCE.  A request that carried tools and armed no constraint
         * decodes its tool call on the model's own habits, and until now that
         * fact left NO trace: the "ENABLED" line below never printed, the mask
         * never ran, and a model improvising a near-miss dialect
         * ("<tool_calls><tool_name>bash</tool_name>") produced a turn with no
         * call and a silent log.  Chasing that silence costs a session, because
         * an absent line is evidence of nothing — it reads the same whether the
         * grammar was off, armed-and-inert, or armed-and-broken.
         *
         * Logged on a decade schedule (1st, 10th, 100th, …) rather than once:
         * once answers "is it armed", the running total answers "how much of
         * this serve's traffic decoded that way", and a decade schedule gives
         * both without a line per request on a fleet that never turns it on.
         * The counter is a plain unsigned long touched from several request
         * threads — a lost increment costs a log line, nothing else, which is
         * why it does not carry a lock. */
        if (params->tools) {
            static unsigned long unconstrained_turns = 0;
            unsigned long n = ++unconstrained_turns;
            bool decade = true;
            for (unsigned long d = n; d > 1; d /= 10)
                if (d % 10 != 0) { decade = false; break; }
            if (decade)
                fprintf(stderr, "[tool_grammar] UNCONSTRAINED: this request "
                        "carries tools but no decoding constraint is armed "
                        "(PANTHEON_TOOL_GRAMMAR=%s, %s) — tool-call syntax is "
                        "whatever the model emits [turns=%lu]\n",
                        env ? env : "unset",
                        !tool_grammar_on ? "feature off"
                                         : "grammar build failed", n);
        }
        return;
    }

    c->pieces = calloc((size_t)n_vocab, sizeof(char *));
    if (!c->pieces) {
        tg_state_free(c->state); tg_free(c->grammar);
        tgj_state_free(c->state_json); tgj_free(c->grammar_json);
        rfg_state_free(c->state_rf);
        memset(c, 0, sizeof(*c));
        return;
    }
    for (int i = 0; i < n_vocab; i++)
        c->pieces[i] = decode_token(model, (uint32_t)i);   /* may be NULL */
    c->n_vocab = n_vocab;
    if (c->kind == SG_GRAMMAR_RESPONSE_JSON)
        sg_tg_mask_end_tokens(c, model);
    /* tool_choice "required" (think-gated walk, either dialect): the turn may not
     * end before its call opens, and may end inside it only where the serve can
     * close it (tgj_end_allowed / tg_end_allowed; the loops append the closing
     * text, sg_tg_close_required_call). */
    if (sg_tg_is_tool(c) && params->tool_choice == SG_TOOL_CHOICE_REQUIRED) {
        c->hook.end_allowed = (c->kind == SG_GRAMMAR_TOOL_JSON) ? sg_tg_end_allowed_json
                                                                : sg_tg_end_allowed_xml;
        sg_tg_blank_end_tokens(c, model);
    }

    if (sg_think_close_is_token(model)) {
        tg_set_think_close_token(c->state);
        tgj_set_think_close_token(c->state_json);
        rfg_set_think_close_token(c->state_rf);
    }

    c->hook.token_text = sg_tg_token_text;
    c->hook.advance    = NULL;   /* serve loop advances; hook masks only */
    c->hook.user       = c;
    c->hook.n_vocab    = n_vocab;
    c->hook.active     = 0;   /* refreshed each step from the grammar */
    /* Fail-CLOSED only in forced mode: on a 0-survivor step force EOS instead of
     * falling open to the model's out-of-grammar top token.  AUTO mode keeps the
     * legacy fail-open (a spurious in-call dead-end there must not truncate a real
     * generation, and AUTO never COMPELS a call). */
    if (forced) {
        int eos = ft_eos_id(model->tokenizer);
        if (eos >= 0) {
            c->hook.fail_closed  = 1;
            c->hook.forced_token = (uint32_t)eos;
        }
    }
    c->enabled = true;
    c->forced  = forced;
    const char *dbg = getenv("PANTHEON_TOOL_GRAMMAR_DEBUG");
    c->verbose = (dbg && dbg[0] == '1');
    if (c->kind == SG_GRAMMAR_RESPONSE_JSON) {
        fprintf(stderr, "[response_format] constrained decoding ENABLED "
                "(%d-vocab mask, json_object)\n", n_vocab);
    } else {
        fprintf(stderr, "[tool_grammar] constrained decoding ENABLED (%d-vocab mask, "
                "dialect=%s, mode=%s%s%s)\n", n_vocab,
                c->kind == SG_GRAMMAR_TOOL_JSON ? "ORNITH_JSON" : "XML",
                (params->tool_choice == SG_TOOL_CHOICE_REQUIRED) ? "REQUIRED"
              : (params->tool_choice == SG_TOOL_CHOICE_FUNCTION) ? "FUNCTION"
              : (params->tool_choice == SG_TOOL_CHOICE_NONE)     ? "NONE" : "AUTO",
                params->forced_name ? " fn=" : "",
                params->forced_name ? params->forced_name : "");
    }
}

/* ============================================================
 * OpenAI `logprobs` accumulation
 *
 * The sampler reports each token's logprobs through a pn_logprob_sink.  On the
 * BATCHED path that sink fires inside the shared decode driver — holding the
 * scheduler lock, on every other session's critical path — so it must not
 * detokenize there.  It therefore records ids + floats only, and the request's
 * OWN thread resolves the text as it consumes each token.
 *
 * Both arrays are PREALLOCATED for the whole turn (max_tokens + the force-close
 * headroom).  Entries are handed to the stream emitter by pointer so they can
 * ride the content delta that carries their bytes, and a realloc would leave
 * those pointers dangling.
 * ============================================================ */

typedef struct {
    uint32_t chosen;
    float    chosen_logprob;
    int      n_top;
    uint32_t top_ids[PN_LOGPROBS_MAX_TOP];
    float    top_logprob[PN_LOGPROBS_MAX_TOP];
    bool     injected;     /* a token the serve force-decoded rather than sampled;
                            * recorded so the sequence stays aligned with the
                            * emitted token stream (see the pn_logprob_sink
                            * injected-token contract) */
} sg_raw_logprob;

typedef struct {
    sg_raw_logprob *raw;
    int             n;        /* records written by the sampler */
    int             taken;    /* records resolved by the request thread */
    int             cap;
    pn_logprob_sink sink;
} sg_logprob_acc;

/* pn_logprob_sink callback: record one step.  Runs on whichever thread sampled. */
static void sg_logprob_emit(void *user, uint32_t chosen, float chosen_logprob,
                            const uint32_t *top_ids, const float *top_logprob,
                            int n_top)
{
    sg_logprob_acc *acc = (sg_logprob_acc *)user;
    if (!acc || !acc->raw || acc->n >= acc->cap) return;
    sg_raw_logprob *r = &acc->raw[acc->n++];
    /* NaN is the documented placeholder for a force-injected token. isnan(),
     * not the `v == v` self-compare: this TU builds with -ffast-math, which
     * folds the self-compare to constant true (proven — the whole check
     * compiles to `return 0` unconditionally), so injected tokens would
     * never be flagged and the raw NaN sentinel would leak out as
     * chosen_logprob. */
    r->injected       = isnan(chosen_logprob);
    r->chosen         = chosen;
    r->chosen_logprob = r->injected ? 0.0f : chosen_logprob;
    r->n_top = 0;
    if (!r->injected && top_ids && top_logprob) {
        int k = n_top > PN_LOGPROBS_MAX_TOP ? PN_LOGPROBS_MAX_TOP : n_top;
        for (int i = 0; i < k; i++) {
            r->top_ids[i]     = top_ids[i];
            r->top_logprob[i] = top_logprob[i];
        }
        r->n_top = k > 0 ? k : 0;
    }
}

/* Arm the accumulator + the result's reported array for a turn of at most
 * `cap` tokens.  Returns false (and leaves everything inert) when the request
 * did not ask for logprobs or an allocation failed — logprobs are a reporting
 * feature and must never fail a generation. */
static bool sg_logprobs_init(sg_logprob_acc *acc, sg_result *result,
                             const sg_gen_params *params, int cap)
{
    memset(acc, 0, sizeof(*acc));
    if (!params->logprobs || cap <= 0) return false;
    acc->raw = calloc((size_t)cap, sizeof(*acc->raw));
    if (!acc->raw) return false;
    result->logprobs = calloc((size_t)cap, sizeof(*result->logprobs));
    if (!result->logprobs) { free(acc->raw); acc->raw = NULL; return false; }
    acc->cap        = cap;
    acc->sink.n_top = params->logprobs_top;
    acc->sink.emit  = sg_logprob_emit;
    acc->sink.user  = acc;
    return true;
}

static void sg_logprobs_attach(sg_logprob_acc *acc, pn_sampling_params *sp)
{
    sp->logprobs = acc->raw ? &acc->sink : NULL;
}

static void sg_logprobs_free(sg_logprob_acc *acc)
{
    free(acc->raw);
    memset(acc, 0, sizeof(*acc));
}

/*
 * Resolve the NEXT recorded step into the result's reported array, decoding the
 * alternative tokens' text.  `piece` is the chosen token's already-decoded text
 * (the decode loop has it, so it is not decoded twice).
 *
 * Returns the entry (borrowed and stable for the whole turn — the array is
 * preallocated) so the caller can hand it to the stream emitter, or NULL when
 * logprobs are off, the sampler recorded nothing for this token, or the token
 * was force-injected.
 *
 * Exactly ONE of this and sg_logprobs_skip must be called per token the decode
 * loop consumes, or the record sequence stops being paired with the token
 * stream and every later token's logprobs land on the wrong text.
 */
/* Consume the next record WITHOUT reporting it — for a token that went to the
 * reasoning channel.  OpenAI's `logprobs.content[]` describes the ANSWER, and
 * our reasoning is a separate channel, so a think-block token must not appear
 * there.  Skipping (rather than simply not calling take) is what keeps the
 * record sequence aligned for the tokens that follow. */
static void sg_logprobs_skip(sg_logprob_acc *acc)
{
    if (acc->raw && acc->taken < acc->n) acc->taken++;
}

static const sg_token_logprob *sg_logprobs_take(sg_logprob_acc *acc,
                                                sg_result *result,
                                                pn_model *model,
                                                const char *piece)
{
    if (!acc->raw || acc->taken >= acc->n) return NULL;
    const sg_raw_logprob *r = &acc->raw[acc->taken++];
    if (r->injected) return NULL;
    if (result->n_logprobs >= acc->cap) return NULL;

    sg_token_logprob *out = &result->logprobs[result->n_logprobs];
    out->token = strdup(piece ? piece : "");
    if (!out->token) return NULL;
    out->logprob = r->chosen_logprob;
    out->n_top   = 0;
    for (int i = 0; i < r->n_top; i++) {
        char *txt = decode_token(model, r->top_ids[i]);
        out->top_token[out->n_top]   = txt ? txt : strdup("");
        out->top_logprob[out->n_top] = r->top_logprob[i];
        out->n_top++;
    }
    result->n_logprobs++;
    return out;
}

/*
 * Copy the request's OpenAI sampling fields onto the engine's sampling params.
 *
 * Every field is a no-op at its default (0 / NULL / seed unset), so a request
 * that sends none of them decodes byte-identically to before.  Shared by the
 * single-user and batched paths so the two cannot drift — the last two bugs in
 * this area were a field that worked on one path only.
 */
static void sg_apply_openai_sampling(pn_sampling_params *sp,
                                     const sg_gen_params *params)
{
    sp->presence_penalty  = params->presence_penalty;
    sp->frequency_penalty = params->frequency_penalty;
    sp->logit_bias        = params->logit_bias;
    sp->n_logit_bias      = params->n_logit_bias;
    if (params->seed_set) {
        sp->rng_seeded = 1;
        sp->rng_state  = pn_seed_to_rng_state(params->seed);
    }
}

/*
 * Which requests need the WHOLE logits row.
 *
 * The K80 forward reduces the LM head ON THE DIE to the top-k (id, logit) pairs
 * so only k pairs cross the 0.4 GB/s host link — the host row is -inf everywhere
 * else.  That is invisible and harmless for ordinary sampling, but any feature
 * that must reason about a token the model did NOT rank highly is silently
 * defeated by it:
 *
 *   - a GRAMMAR mask can only choose among logits it can see, so when none of
 *     the legal tokens is in the model's top-k the step has zero survivors, the
 *     sampler fails open, and the walk dies — the response_format defect;
 *   - `logit_bias` cannot lift (or veto) a pruned token at all: -inf + 100 is
 *     still -inf, so a bias on anything outside the top-k is a no-op;
 *   - `logprobs` are normalised over whatever survived, so a pruned row yields a
 *     log-softmax taken over ~k candidates instead of the vocabulary.
 *
 * `logit_bias` and `logprobs` ask for the full row for the turn's lifetime; a
 * GRAMMAR asks for it only while its mask is live (see sg_arm_full_logits).
 * Everything else keeps the fast top-k path untouched.
 */
/* The turn-lifetime half of the requirement: features that need every logit on
 * EVERY step, with no window to narrow. */
static bool sg_full_logits_always(const sg_gen_params *params)
{
    return (params->logit_bias && params->n_logit_bias > 0) ||
           params->logprobs;
}

#ifdef USE_TT
/* A logit bias on the TT top-N rows: the sampler decides from the row when no biased outside
 * token can reach the pick and refills the dense row when one can (sampler_topn_surgery.h).
 * A SAMPLED request with a positive bias keeps the dense row (it would refill every row). */
static bool sg_tt_bias_on_topn(const sg_gen_params *params)
{
    pn_sampling_params bias = { 0 };

    bias.logit_bias = params->logit_bias;
    bias.n_logit_bias = params->n_logit_bias;
    return params->logit_bias && params->n_logit_bias > 0 && pn_topn_surgery_enabled() &&
           (params->temperature <= 0.0f || !pn_topn_surgery_raising_bias(&bias));
}

/* Before sg_tg_setup has run: could this request arm a grammar, or need the whole row? */
static bool sg_may_need_full_logits(const sg_gen_params *params)
{
    return params->logprobs ||
           (params->logit_bias && params->n_logit_bias > 0 && !sg_tt_bias_on_topn(params)) ||
           (sg_tool_grammar_arms(params) && !sg_tool_grammar_speculates(params)) ||
           (params->response_format == SG_RESPONSE_FORMAT_JSON_OBJECT &&
            !sg_response_format_speculates(params));
}
#endif

static bool sg_needs_full_logits(const sg_tg_ctx *c, const sg_gen_params *params)
{
    return c->enabled || sg_full_logits_always(params);
}

static void sg_require_full_logits(const sg_tg_ctx *c,
                                   const sg_gen_params *params,
                                   pn_model *model, pn_state *state, int on)
{
    if (!state) return;
    if (on && !sg_needs_full_logits(c, params)) return;
    if (!pn_qwen35moe_owns_state(model)) return;
    pn_qwen35moe_state_set_full_logits(state, on);
}

/* ============================================================
 * Full-logits ARMING WINDOW (PN_GRAMMAR_ARM_WINDOW=1, default OFF)
 *
 * WHY A WINDOW.  The K80 head reduces the 248320-float logits row to k=40
 * (id, logit) pairs on the die.  A request that must reason about a token the
 * model did NOT rank highly needs the whole row instead, and that costs the
 * constrained slot ~4 ms of D2H per token PLUS a dense host sampler (~7.35 ms
 * at temp 0.7 — a dense row falls out of PN_SPARSE_SAMPLER's fast path).
 * `logit_bias` and `logprobs` need it on every step of the turn.  A tool
 * GRAMMAR does not: in AUTO mode the mask is inactive for the whole reasoning
 * region — up to the 2000-token reasoning budget — and engages only once the
 * model has actually emitted a tool-call opener.  Arming for the TURN therefore
 * pays the full-row price through the entire reasoning budget for a mask that
 * is not being applied, on what the client makes the common turn shape.
 *
 * THE INVARIANT THIS MUST MAINTAIN
 *
 *     On every step whose sampler evaluates the grammar mask, the row it reads
 *     is the FULL vocabulary row.
 *
 * A missed mask is not a slow step, it is a correctness bug: the mask can only
 * choose among logits it can see, so a sparse row holding none of the legal
 * tokens has zero survivors and the walk fails open (AUTO) or force-terminates
 * (FORCED) — the response_format defect.  engine.c counts violations
 * (pn_sampler_grammar_sparse_rows) so the invariant is observable at runtime
 * rather than argued about.
 *
 * WHY THE OBVIOUS FIX BREAKS IT — THE TRANSITION STEP.  `hook.active` is
 * refreshed at the TOP of a step, from the grammar state as it stood after the
 * PREVIOUS token.  The forward that fetches the NEXT row runs at the BOTTOM of
 * that same step, after this step's token has already advanced the grammar.
 * Arming off `hook.active` (or, equivalently, off "is the grammar active right
 * now?" asked before the sampler runs) is therefore stale by exactly one token
 * — and the token it is stale on is precisely the opener that engages the
 * grammar.  The transition step gets a sparse row under an active mask.
 * PN_GRAMMAR_ARM_STALE=1 reproduces that ordering on purpose, so a gate can
 * prove its detector fires instead of assuming it would.
 *
 * THE MECHANISM.  Arm from the grammar state AFTER the advance and immediately
 * BEFORE the forward.  mask_active() is a pure function of the walk state, and
 * the walk state is mutated only by advance() (over a sampled or an injected
 * token) and by sg_tg_reasoning_ended().  The forward for step T+1 consumes
 * exactly the token the sampler at step T advanced over, so mask_active() read
 * between those two events is bit-for-bit the value refresh() will return at
 * step T+1's sampler.  This is not a prediction carrying a lookahead error — it
 * is the same read, moved to the other side of the advance.
 *
 * THE ONE ORDERING HOLE, AND ITS LATCH.  The single-user reasoning-budget
 * force-close is the only place a grammar mutation can land AFTER the forward
 * that fetched the row the next step samples: the block runs at the bottom of
 * the loop and ends with sg_tg_reasoning_ended().  `fl_pending` is latched the
 * moment the budget TRIPS — which happens before that forward — and cleared
 * once the whole close has run, so every forward in the window is armed
 * unconditionally.  The batched driver has no such hole: it samples, advances,
 * injects and arms inside one critical section and only then issues the next
 * step (see driver_arm_full_logits in batch_sched.c).
 *
 * KNOWN, PRE-EXISTING, NOT CHANGED HERE.  On the single-user path sg_tg_setup
 * runs AFTER the prefill, so the FIRST sampled token of a grammar turn already
 * reads a sparse row today, window or no window.  It only bites a FORCED
 * grammar whose reasoning has already ended at token 0; the batched path (the
 * production shape) arms before its prefill and is unaffected.  Left alone
 * deliberately so this change stays one mechanism, and now detected at runtime
 * by the engine counter above.
 * ============================================================ */

/* PN_GRAMMAR_ARM_WINDOW=1 — arm the full row only while the grammar mask is
 * live, instead of for the whole turn.  Default OFF: this changes which rows
 * cross the host link on the live tool-calling path and has not been through
 * the agent-client gate.  Process-constant. */
static int sg_grammar_arm_window_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ARM_WINDOW");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* PN_GRAMMAR_ARM_STALE=1 — FALSIFIER, only meaningful with the window on.
 * Arms from the grammar state as it stood BEFORE this step's advance, i.e.
 * reintroduces the one-token-late ordering the window exists to avoid.  It
 * exists so the gate can watch pn_sampler_grammar_sparse_rows go non-zero and
 * know the detector is not vacuous.  Never set this in production. */
static int sg_grammar_arm_stale_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ARM_STALE");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* Would the mask be applied if a step sampled RIGHT NOW?  A pure read of the
 * live grammar walk — the same three-way dispatch the sampler's refresh()
 * callbacks make, and the same function `sg_tg_refresh` publishes into
 * `hook.active`.  Keeping one predicate is the point: the arming window is only
 * correct because what it reads is literally what the sampler will read. */
static bool sg_tg_mask_active(const sg_tg_ctx *c) {
    if (!c->enabled) return false;
    return (c->kind == SG_GRAMMAR_TOOL_JSON)     ? tgj_mask_active(c->state_json)
         : (c->kind == SG_GRAMMAR_RESPONSE_JSON) ? rfg_mask_active(c->state_rf)
                                                 : tg_mask_active(c->state);
}

/**
 * Arm (or disarm) the full-vocabulary logits row for the NEXT forward.
 *
 * Call immediately before every forward and after every grammar mutation that
 * precedes it — that ordering is the whole mechanism; see the block comment
 * above sg_grammar_arm_window_on for why one token either way matters.
 *
 * @param c      the request's grammar context (may be inert)
 * @param params the request's generation params (logit_bias / logprobs)
 * @param model  the model owning `state` (non-qwen35 states are left alone)
 * @param state  the decode state whose logits row is being armed
 *
 * Thread-safe: no — called by whichever thread owns `state`'s next forward
 */
static void sg_arm_full_logits(const sg_tg_ctx *c, const sg_gen_params *params,
                               pn_model *model, pn_state *state)
{
    if (!state || !pn_qwen35moe_owns_state(model)) return;
    int need;
    if (c->fl_always || sg_full_logits_always(params)) need = 1;
    else if (!c->enabled)                              need = 0;
    else if (!sg_grammar_arm_window_on())              need = 1;
    else if (sg_grammar_arm_stale_on())                need = c->hook.active ? 1 : 0;
    else need = (c->fl_pending || sg_tg_mask_active(c)) ? 1 : 0;
    pn_qwen35moe_state_set_full_logits(state, need);
}

/* Snapshot the engine's mask-density counters at the start of a constrained
 * turn, so the turn can report its own delta. */
static void sg_tg_mark_mask_density(sg_tg_ctx *c) {
    c->fl_mask_steps0  = pn_sampler_grammar_mask_steps();
    c->fl_sparse_rows0 = pn_sampler_grammar_sparse_rows();
    c->fl_shortlist_steps0 = pn_sampler_grammar_shortlist_steps();
    c->fl_bound_steps0 = pn_sampler_grammar_bound_steps();
    c->fl_chunk_steps0 = pn_sampler_grammar_chunk_steps();
    c->fl_fullscan_steps0 = pn_sampler_grammar_fullscan_steps();
    c->fl_mask_seconds0 = pn_sampler_grammar_mask_seconds();
}

/* Report this turn's mask-density result: how many of its steps evaluated the
 * grammar mask, and how many of those read a SPARSE row.
 *
 * `sparse=0` is the invariant.  `masked=0` is the other thing worth seeing: it
 * means the turn never engaged the grammar at all, so a clean `sparse=0` on its
 * own would be proving nothing — which is exactly the failure mode a
 * throughput-only gate would sail past.
 *
 * The engine counters are PROCESS-WIDE, so this delta is exactly this turn's
 * only while one turn is decoding; with K concurrent constrained slots each
 * line carries some of its neighbours'.  Fine for what it is — a tripwire whose
 * only interesting value is zero — and the gate runs at K=1, where the
 * attribution is exact. */
static void sg_tg_report_mask_density(const sg_tg_ctx *c) {
    if (!c->enabled) return;
    unsigned long masked = pn_sampler_grammar_mask_steps() - c->fl_mask_steps0;
    unsigned long sparse = pn_sampler_grammar_sparse_rows() - c->fl_sparse_rows0;
    fprintf(stderr, "[tool_grammar] turn mask density: masked_steps=%lu "
            "sparse_rows=%lu shortlist=%lu chunks=%lu fullscan=%lu bound=%lu mask_ms=%.1f%s\n",
            masked, sparse, pn_sampler_grammar_shortlist_steps() - c->fl_shortlist_steps0,
            pn_sampler_grammar_chunk_steps() - c->fl_chunk_steps0,
            pn_sampler_grammar_fullscan_steps() - c->fl_fullscan_steps0,
            pn_sampler_grammar_bound_steps() - c->fl_bound_steps0,
            (pn_sampler_grammar_mask_seconds() - c->fl_mask_seconds0) * 1000.0,
            sparse ? "  <-- INVARIANT VIOLATED" : "");
}

/* The reverse coupling's one observable moment: the GRAMMAR decided reasoning
 * was over while the emitter was still routing to <think>, so the serve moves the
 * channel under the model.
 *
 * WHY THIS LOGS.  Everything after the flip is filed as ANSWER content, and the
 * client cannot tell an answer the model wrote from a think block the serve
 * re-labelled.  When the flip is right — the grammar is compelling an opener and
 * a tool call follows — the line is one entry beside the call.  When it is wrong
 * the line is the ONLY evidence: 37,721 chars of think-block prose reached a live
 * transcript as three `answer` entries beginning mid-sentence, and nothing in
 * either process said anything (qwen36-bf16, one recorded session,
 * 2026-08-26).  The `reasoning_leak` defect code does NOT cover this: it fires
 * only when a parsed tool call's PARAMETER VALUE begins with "</think>", and here
 * there was no call and no tool region to parse.
 *
 * A walk that has gone DEAD no longer reports reasoning-ended at all
 * (tg_forced_reasoning_ended), so the runaway case this line was written for is
 * now left in the reasoning channel where it belongs — and the walk's own
 * "went DEAD" line reports it.  This stays for the flips that do happen. */
static void sg_log_reasoning_channel_flip(const sg_tg_ctx *c, int generated)
{
    static unsigned long flips = 0;
    fprintf(stderr, "[reasoning_channel] the grammar ended reasoning while the "
            "emitter was still in <think> — routing to CONTENT from token %d so "
            "the compelled call is parsed (%s mode, flips=%lu)\n",
            generated, c->forced ? "FORCED" : "AUTO", ++flips);
}

/* pn_sampling_params.dense_row for a row that is ALREADY dense: nothing to
 * fetch.  Setting it lets the sampler decide the mask from a shortlist
 * (grammar_mask_shortlist.h), which it does only for rows a backend can refill. */
static int sg_gs_row_already_dense(void *ctx, float *logits, int n_vocab)
{
    (void)ctx; (void)logits; (void)n_vocab;
    return 0;
}

/* The batched (CUDA) path hands a grammar slot the FULL logits row on every step
 * (sg_require_full_logits arms it for the turn), so the row is already dense and the
 * sampler may judge a shortlist of its top GMS_MAX instead of all 248k tokens: the
 * same masked row, hence the same token (tests/test_grammar_dense_shortlist.c).
 * No dense_chunks / tail_counts: nothing to fetch, no device counts.
 * Not with PN_GRAMMAR_ARM_WINDOW=1: unmasked steps then carry the sparse top-k row,
 * which a refill callback would make the sampler treat as a top-N row.
 * TT builds wire their own callbacks (sg_serve TT block). */
static void sg_tg_wire_dense_row(const sg_tg_ctx *c, pn_sampling_params *sp)
{
#ifndef USE_TT
    if (!c->enabled || sg_grammar_arm_window_on()) return;
    sp->dense_row = sg_gs_row_already_dense;
    sp->dense_ctx = NULL;
    sp->tail_counts = NULL;
    sp->dense_chunks = NULL;
#else
    (void)c; (void)sp;
#endif
}

/* Point the sampling params at the hook (no-op when disabled). */
static void sg_tg_attach(sg_tg_ctx *c, pn_sampling_params *sp) {
    sp->tool_grammar = c->enabled ? &c->hook : NULL;
}

/* Batched variant: arm the driver-side callbacks so the hook advances, gates
 * (self-refresh) and reports completion from WITHIN the sampler — the batched
 * driver samples per-slot in its own thread and never touches the hook
 * externally.  Then point sp at the hook.  (Single-user leaves these NULL and
 * drives advance/refresh/completion from its own decode loop.) */
static void sg_tg_attach_batched(sg_tg_ctx *c, pn_sampling_params *sp) {
    switch (c->kind) {
    case SG_GRAMMAR_TOOL_JSON:
        c->hook.advance  = sg_tg_advance_cb_json;
        c->hook.refresh  = sg_tg_refresh_cb_json;
        c->hook.complete = sg_tg_complete_cb_json;
        c->hook.in_call  = sg_tg_in_call_cb_json;
        c->hook.reasoning_ended = sg_tg_reasoning_ended_cb_json;
        c->hook.clone      = sg_tg_clone_cb_json;
        c->hook.clone_free = sg_tg_clone_free_cb_json;
        break;
    case SG_GRAMMAR_TOOL_XML:
        c->hook.advance  = sg_tg_advance_cb;
        c->hook.refresh  = sg_tg_refresh_cb;
        c->hook.complete = sg_tg_complete_cb;
        c->hook.in_call  = sg_tg_in_call_cb;
        /* Only the XML dialect gets a landing window (the JSON dialect's model
         * runs reasoning_budget 0, so its close is never injected). */
        c->hook.reasoning_end_injected = sg_tg_reasoning_end_injected_cb;
        c->hook.reasoning_ended = sg_tg_reasoning_ended_cb;
        c->hook.clone      = sg_tg_clone_cb;
        c->hook.clone_free = sg_tg_clone_free_cb;
        break;
    case SG_GRAMMAR_RESPONSE_JSON:
        c->hook.advance  = sg_tg_advance_cb_rf;
        c->hook.refresh  = sg_tg_refresh_cb_rf;
        c->hook.complete = sg_tg_complete_cb_rf;
        c->hook.in_call  = sg_tg_in_call_cb_rf;
        break;
    case SG_GRAMMAR_NONE:
    default:
        break;
    }
    sp->tool_grammar = c->enabled ? &c->hook : NULL;
    sg_tg_wire_dense_row(c, sp);
}

/* Refresh the mask's active flag before each sampling step.  AUTO: mask only
 * inside a tool call (FREE region unconstrained → scan skipped).  FORCED: also
 * mask the post-reasoning FREE state so the opener is compelled. */
static void sg_tg_refresh(sg_tg_ctx *c) {
    if (!c->enabled) return;
    c->hook.active = sg_tg_mask_active(c) ? 1 : 0;
}

/* Has the grammar reached a COMPLETE tool call?  Used to STOP generation the
 * instant "</function>"/"</tool_call>" closes, so a constrained decode
 * terminates right after the call instead of running to max_tokens (the ~80 s
 * "hang" on forced requests — the old loop had no completion-stop). */
static bool sg_tg_done(const sg_tg_ctx *c) {
    if (!c->enabled) return false;
    if (c->kind == SG_GRAMMAR_TOOL_JSON)     return tgj_complete(c->state_json);
    if (c->kind == SG_GRAMMAR_RESPONSE_JSON) return rfg_complete(c->state_rf);
    return tg_complete(c->state);
}

/* Is the grammar constraint live for this request (forced or auto+tools)?  A
 * DEAD grammar (mask left the walk with no valid continuation) must NOT stop
 * generation — only a genuine tg_complete does. */
static bool sg_tg_live(const sg_tg_ctx *c) { return c->enabled; }

/* True when a tool-call skeleton is currently being emitted under the grammar.
 * The AUTO reasoning-budget force-close must NOT fire mid-skeleton: injecting
 * "</think>" then would mismatch the live grammar walk → TG_DEAD → the mask
 * silently disengages and a forced request strands a half-call in reasoning. */
static bool sg_tg_in_call(const sg_tg_ctx *c) {
    if (!c->enabled) return false;
    if (c->kind == SG_GRAMMAR_TOOL_JSON)     return tgj_in_call(c->state_json);
    if (c->kind == SG_GRAMMAR_RESPONSE_JSON) return rfg_in_json(c->state_rf);
    return tg_in_call(c->state);
}

/* Did this turn's tool walk leave the grammar (DEAD)?  Only the tool grammars
 * answer — the response-format grammar reports its own death separately. */
static bool sg_tg_dead(const sg_tg_ctx *c) {
    if (!c->enabled) return false;
    if (c->kind == SG_GRAMMAR_TOOL_JSON) return tgj_is_dead(c->state_json);
    if (c->kind == SG_GRAMMAR_TOOL_XML)  return tg_is_dead(c->state);
    return false;
}

/*
 * End-of-turn verdict for a `response_format` request: did the JSON constraint
 * actually hold?
 *
 * Sets result->response_format_ok, and logs the grammar's final state so a turn
 * that did NOT end in `done` says why on the fleet log — phase `reason` means the
 * grace never ended (no "</think>" was seen), `pre` means the object never
 * opened, `dead` means the walk left the grammar, and advances == 0 means the
 * sampler never fed the grammar at all.  Those are indistinguishable from the
 * response body alone, which is how a non-constraining build was able to keep
 * reporting "applied".
 */
static void sg_tg_report_response_format(const sg_tg_ctx *c, sg_result *result)
{
    if (!c->enabled || c->kind != SG_GRAMMAR_RESPONSE_JSON) return;
    bool ok = rfg_complete(c->state_rf);
    result->response_format_ok = ok ? 1 : 0;
    fprintf(stderr, "[response_format] turn end: enforced=%d phase=%s "
            "advances=%d depth=%d reasoning_ended=%d died_in=%s on=\'%s\'\n",
            ok ? 1 : 0, rfg_debug_phase_name(c->state_rf),
            rfg_debug_advances(c->state_rf), rfg_debug_depth(c->state_rf),
            rfg_reasoning_ended(c->state_rf) ? 1 : 0,
            rfg_debug_death_phase(c->state_rf),
            rfg_debug_death_piece(c->state_rf));
}

/*
 * End-of-turn verdict for a TOOL-CALL request: did the call the model started
 * ever close?  The sibling of sg_tg_report_response_format above, and it exists
 * for the same reason — the bit is computed inside the grammar and dies with it
 * unless something carries it out.
 *
 * WHY THE HTTP LAYER CANNOT WORK THIS OUT ITSELF.  A truncated call's text is a
 * legal prefix of a good call, so the parser reads it as a complete call minus an
 * argument and reports no error at all.  Only the walk knows it was still inside
 * the skeleton.  Live: 9/9 truncated calls were accepted char-for-char by this
 * grammar with tg_complete false, and shipped as `read {}` / `write {"path":…}`.
 *
 * WHY "entered a call" IS PART OF THE TEST.  A walk still in FREE never started a
 * call — the model answered in prose, or emitted a dialect this grammar does not
 * model (MiniCPM's `<function name="…">` never matches the `<function=` opener, so
 * an armed grammar sits in FREE for the whole turn).  Reporting those INCOMPLETE
 * would suppress perfectly good calls.  A DEAD walk is excluded for the same
 * reason in the other direction: the mask disengaged, so the text after that point
 * was never constrained and the grammar has no standing to judge it.
 *
 * Logged only on the failing verdict — healthy traffic costs nothing, and the raw
 * text is already dumped by openai_chat.c's [tool_call_dropped] line.
 */
static void sg_tg_report_tool_call(const sg_tg_ctx *c, sg_result *result)
{
    result->tool_grammar_verdict = SG_TOOL_GRAMMAR_ABSENT;
    if (!c->enabled || !sg_tg_is_tool(c)) return;
    if (sg_tg_done(c)) {
        result->tool_grammar_verdict = SG_TOOL_GRAMMAR_COMPLETE;
        return;
    }
    /* THE SILENT SWITCH-OFF.  A DEAD walk reports its mask INACTIVE
     * (tg_mask_active), so from the step that killed it every remaining token of
     * this turn was sampled with NO tool constraint — and until now that left the
     * same trace as a turn that simply answered in prose, because both are
     * "!in_call".  Named here, with a running count, so "how often does the
     * grammar switch itself off" is a question the fleet log can answer.
     * Deliberately does NOT set INCOMPLETE: the text after the death was never
     * constrained, so the grammar has no standing to judge it (see the note
     * above), and refusing the call would turn an observability gap into a
     * behaviour change. */
    if (sg_tg_dead(c)) {
        static unsigned long dead_turns = 0;
        fprintf(stderr, "[tool_grammar] turn end: the walk went DEAD in %s mode "
                "— the mask reported inactive from that point and the rest of "
                "this turn decoded UNCONSTRAINED (dead_turns=%lu)\n",
                c->forced ? "FORCED" : "AUTO", ++dead_turns);
        return;
    }
    if (!sg_tg_in_call(c)) return;      /* never entered the skeleton */
    result->tool_grammar_verdict = SG_TOOL_GRAMMAR_INCOMPLETE;
    fprintf(stderr, "[tool_grammar] turn end: the call never closed "
                    "(grammar still inside the skeleton) — refusing to emit it\n");
}

/* Per-step trace (PANTHEON_TOOL_GRAMMAR_DEBUG=1): dump the sampled piece and the
 * grammar's live gating flags so the forced-decode can be watched on the K80 test machine. */
static void sg_tg_trace(const sg_tg_ctx *c, int step, uint32_t tok,
                        const char *piece) {
    if (!c->verbose) return;
    char esc[64]; int j = 0;
    const char *p = piece ? piece : "";
    for (int i = 0; p[i] && j < (int)sizeof(esc) - 4; i++) {
        unsigned char ch = (unsigned char)p[i];
        if (ch == '\n') { esc[j++] = '\\'; esc[j++] = 'n'; }
        else if (ch == '\t') { esc[j++] = '\\'; esc[j++] = 't'; }
        else if (ch < 32 || ch > 126) { esc[j++] = '?'; }
        else esc[j++] = (char)ch;
    }
    esc[j] = '\0';
    int in_call = sg_tg_in_call(c) ? 1 : 0;
    int done    = sg_tg_done(c) ? 1 : 0;
    int rchars  = (c->kind == SG_GRAMMAR_TOOL_JSON)
                      ? tgj_debug_reason_chars(c->state_json)
                : (c->kind == SG_GRAMMAR_RESPONSE_JSON)
                      ? rfg_debug_depth(c->state_rf)
                      : tg_debug_reason_chars(c->state);
    fprintf(stderr, "[tg] step=%d tok=%u piece='%s' mask=%d in_call=%d "
            "complete=%d rchars=%d\n", step, tok, esc, c->hook.active,
            in_call, done, rchars);
}

/* Advance the grammar by the chosen token's decoded text (every step). */
static void sg_tg_step(sg_tg_ctx *c, const char *piece) {
    if (!c->enabled || !piece) return;
    if (c->kind == SG_GRAMMAR_TOOL_JSON)          tgj_advance(c->state_json, piece);
    else if (c->kind == SG_GRAMMAR_RESPONSE_JSON) rfg_advance(c->state_rf, piece);
    else                                          tg_advance(c->state, piece);
}

/* Secondary reasoning-end trigger (forced mode): the emitter's <think> block
 * closed.  The grammar ALSO self-detects "</think>" in its own stream (BUG 1), so
 * this is belt-and-suspenders — whichever fires first compels the opener.  Kept
 * because it costs nothing and covers callers/streams the emitter splits cleanly. */
static void sg_tg_reasoning_ended(sg_tg_ctx *c) {
    if (!c->enabled) return;
    if (c->kind == SG_GRAMMAR_TOOL_JSON)          tgj_reasoning_end(c->state_json);
    else if (c->kind == SG_GRAMMAR_RESPONSE_JSON) rfg_reasoning_end(c->state_rf);
    else                                          tg_reasoning_end(c->state);
}

/* Reasoning-end trigger for an INJECTED close — the reasoning-budget force-close
 * decoded "</think>" itself rather than the model choosing it.  Grants the XML
 * grammar's landing window so the opener is not compelled around an unfinished
 * thought (see tg_reasoning_end_injected).  A "required" walk gets no window:
 * its opener is compelled from the next token, as after the model's own close.
 *
 * The JSON dialect (ornith) is deliberately NOT given a window here: it runs
 * reasoning_budget 0, so the serve never injects a close for it and this path is
 * unreachable — see the note in tool_grammar_json.h.  It falls back to the plain
 * reasoning-end so behavior is unchanged rather than silently different. */
static void sg_tg_reasoning_ended_injected(sg_tg_ctx *c) {
    if (!c->enabled) return;
    if (c->kind == SG_GRAMMAR_TOOL_XML) tg_reasoning_end_injected(c->state);
    else                                sg_tg_reasoning_ended(c);
}


/* Has the grammar itself decided forced-mode reasoning is over (saw "</think>" OR
 * the reasoning budget tripped)?  The serve loop uses this to flip the emitter out
 * of its <think> channel so a compelled call routes to CONTENT and is parsed as a
 * tool_call, even when the model never emitted a clean "</think>". */
static bool sg_tg_forced_ended(const sg_tg_ctx *c) {
    if (!c->enabled) return false;
    if (c->kind == SG_GRAMMAR_TOOL_JSON) return tgj_forced_reasoning_ended(c->state_json);
    /* The response-format grammar has no "forced" mode: reasoning ending is what
     * ARMS its constraint, and the emitter tracks the same "</think>" itself, so
     * there is nothing extra to tell it.  Reporting true here would only flip the
     * emitter's channel, which it has already done. */
    if (c->kind == SG_GRAMMAR_RESPONSE_JSON) return false;
    return tg_forced_reasoning_ended(c->state);
}

/* The model ended its turn INSIDE a "required" call — the mask let the end
 * token through only because the call can still be closed (tgj_end_allowed /
 * tg_end_allowed).  Append the closing text through the grammar and the
 * emitter, so the turn ends in the call the model meant: the same call the
 * parser reads from the unclosed text, but checked by the grammar.  Returns true
 * when the call is now complete.  The closing bytes are not forwarded through
 * the model (the turn is over); a later turn re-prefills from the divergence.
 * Both loops call it on an end token; on the batched path the driver has
 * finished the slot by then, so the walk is this thread's alone. */
static bool sg_tg_close_required_call(sg_tg_ctx *c, const sg_gen_params *params, emitter *em)
{
    static unsigned long closes = 0;
    char closing[TG_CLOSING_MAX], shown[2 * TG_CLOSING_MAX];
    int n, j = 0;
    bool json;

    if (!c->enabled || !sg_tg_is_tool(c) || params->tool_choice != SG_TOOL_CHOICE_REQUIRED)
        return false;
    json = (c->kind == SG_GRAMMAR_TOOL_JSON);
    if (json ? (!tgj_in_call(c->state_json) || tgj_complete(c->state_json))
             : (!tg_in_call(c->state) || tg_complete(c->state)))
        return false;
    n = json ? tgj_closing_text(c->state_json, closing, (int)sizeof(closing))
             : tg_closing_text(c->state, closing, (int)sizeof(closing));
    if (n < 0)
        return false;
    if (json) tgj_advance(c->state_json, closing);
    else      tg_advance(c->state, closing);
    em_feed(em, closing);
    for (int i = 0; i < n && j < (int)sizeof(shown) - 3; i++) {
        if (closing[i] == '\n') { shown[j++] = '\\'; shown[j++] = 'n'; }
        else shown[j++] = closing[i];
    }
    shown[j] = '\0';
    fprintf(stderr, "[tool_grammar] the model ended inside its call — closed it with "
            "'%s' (closes=%lu)\n", shown, ++closes);
    return json ? tgj_complete(c->state_json) : tg_complete(c->state);
}

static void sg_tg_free(sg_tg_ctx *c) {
    if (c->pieces) {
        for (int i = 0; i < c->n_vocab; i++) free(c->pieces[i]);
        free(c->pieces);
    }
    tg_state_free(c->state);
    tg_free(c->grammar);
    tgj_state_free(c->state_json);
    tgj_free(c->grammar_json);
    rfg_state_free(c->state_rf);
    memset(c, 0, sizeof(*c));
}

/* ============================================================
 * Forward step — async when the model supports it (C2)
 *
 * One token's forward. When the engine exposes a non-blocking step for
 * this model (qwen35moe on GPU), LAUNCH it and sleep-poll the completion
 * event instead of blocking inside a CUDA stream sync: a blocked sync
 * spins/yields a host core, and with up to 8 cards generating from a
 * 2-core host the spinning waiters starve each other's launch turns so
 * the cards serialize. Sleeping between event polls costs ~no CPU, so
 * every connection thread's card stays in flight simultaneously — this
 * per-connection thread IS its card's worker, and it never blocks the
 * others. Poll cadence backs off 50us -> ~500us (worst-case added
 * latency well under 1 ms/step, negligible against a K80 decode step);
 * override the cap with PN_ASYNC_POLL_US. Falls back to the ordinary
 * blocking model->forward whenever the async step is unsupported or the
 * token is not eligible — bit-identical behavior.
 * ============================================================ */

static tm_tensor *sg_forward_step(pn_model *model, pn_state *state,
                                  uint32_t token)
{
    int rc = pn_forward_step_begin(model, token, state);
    if (rc > 0)
        return model->forward(model, token, state);
    if (rc < 0)
        return NULL;

    static long cap_us = -1;
    if (cap_us < 0) {
        const char *env = getenv("PN_ASYNC_POLL_US");
        cap_us = env ? atol(env) : 500;
        if (cap_us < 50) cap_us = 50;
        if (cap_us > 20000) cap_us = 20000;
    }

    long us = 50;
    for (;;) {
        tm_tensor *out = NULL;
        rc = pn_forward_step_poll(model, state, &out);
        if (rc == 0)
            return out;
        if (rc < 0)
            return NULL;
        struct timespec ts = { 0, us * 1000L };
        nanosleep(&ts, NULL);
        if (us < cap_us) {
            us *= 2;
            if (us > cap_us) us = cap_us;
        }
    }
}

/* ============================================================
 * Prefill (L1 — batched, L2 — chunked)
 *
 * One chunk of prompt tokens through the fastest available path: the
 * BATCHED forward (pn_qwen35moe_forward_batch_cuda — one pass for the whole
 * chunk, ~2× the token-by-token loop; internally falls back to sequential
 * single-token forwards when the batched path cannot run, so correctness
 * never depends on it) when the model runs the qwen35moe CUDA forward,
 * else the ordinary sg_forward_step loop.  Bit-exact either way: the
 * batched prefill's KV/SSM state and last-position logits are verified
 * identical to n sequential single-token forwards (SpeedOptimization §2).
 *
 * QWEN35_BS_PREFILL_BATCH=0 restores the token-by-token loop (A/B lever).
 * QWEN35_BS_PREFILL_CHUNK sets the chunk size in tokens (default 128;
 * 0 = whole prompt in one pass).  Chunking serves two masters: it bounds
 * the batch scratch for long prompts, and in the batch-scheduled path each
 * chunk boundary is a bs_prefill_yield window where the decode driver gets
 * a step in (L2) — a joiner freezes concurrent decoders for one chunk,
 * not one prompt.
 * ============================================================ */

#ifdef USE_CUDA
static int sg_prefill_batch_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_BS_PREFILL_BATCH");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}
#endif

static int sg_prefill_chunk(void)
{
    static int v = -2;
    if (v == -2) {
        const char *e = getenv("QWEN35_BS_PREFILL_CHUNK");
        v = e ? atoi(e) : 128;
        if (v < 0) v = 0;
    }
    return v;
}

/* ── Phase 9b: prefill chunk pipelining (QWEN35_PREFILL_PIPELINE, default OFF).
 *
 * Every chunk but the LAST of a range has its logits freed three lines later,
 * and computing them is what forces the host to drain die B — which is what
 * makes the two dies run strictly sequentially. Running the intermediate
 * chunks through the no-head entry leaves die B's segment in flight so the
 * next chunk's die-A segment overlaps it.
 *
 * This is a lever on ISSUE ORDER only: identical kernels, identical per-die
 * order, identical inputs, no reassociation. Every early exit MUST drain
 * first — an in-flight die-B segment is still writing this session's KV rings
 * and SSM state, and the abort path resets exactly that.
 *
 * Only engaged for the qwen35moe CUDA batched prefill; any refusal from the
 * model falls back to the ordinary chunk call, which is always correct. ── */
#ifdef USE_CUDA
/* Does this model run the qwen35moe CUDA batched prefill? The MoE forward
 * always does; a dense-FFN model (Ornith, Qwen3.8) runs pn_bonsai_forward_cuda
 * and does when every weight has a batched kernel (Bonsai's ternary Q2_0 has
 * none). The 9b/9c pipelines below run that same batched layer path, so they
 * take the same gate: keying them on the MoE forward pointer alone left Qwen3.8
 * prefilling one die at a time (8,404 tokens: 900 s). */
static int sg_qwen35_batch_prefill_model(pn_model *model)
{
    return model && (model->forward == pn_qwen35moe_forward_cuda ||
                     pn_qwen35moe_dense_batch_prefill_ok(model));
}
#endif

static int sg_prefill_pipeline_ready(pn_model *model)
{
#ifdef USE_CUDA
    return pn_qwen35moe_prefill_pipeline_enabled() &&
           sg_prefill_batch_enabled() && sg_qwen35_batch_prefill_model(model);
#else
    (void)model;
    return 0;
#endif
}

/* Run one intermediate (logits-discarded) chunk. 0 = the pipelined path took
 * it and advanced state->pos; non-zero = nothing happened, run the ordinary
 * chunk instead. */
static int sg_prompt_chunk_nologits(pn_model *model, pn_state *state,
                                    const uint32_t *toks, int n)
{
#ifdef USE_CUDA
    return pn_qwen35moe_prefill_chunk_cuda(model, toks, n, state);
#else
    (void)model; (void)state; (void)toks; (void)n;
    return -1;
#endif
}

static void sg_prefill_pipeline_drain(pn_model *model)
{
#ifdef USE_CUDA
    pn_qwen35moe_prefill_drain_cuda(model);
#else
    (void)model;
#endif
}

/* ── Phase 9c: WIDE pipelining (QWEN35_PREFILL_PIPELINE_WIDE, default OFF).
 *
 * 9b's chunk loop lives here and hands the model ONE chunk at a time, which
 * caps the overlap at two dies. A pipeline that keeps all n_die dies busy has
 * to see the whole range at once, so this hands the range over wholesale and
 * the model owns the chunking. The abort check the loop below does between
 * chunks is preserved by passing it down: the model polls it once per pipeline
 * step and drains what is in flight before reporting the abort.
 *
 * Refusal (-1) is always safe — nothing has been advanced and the ordinary
 * chunked loop runs instead. ── */
static int sg_prefill_wide_ready(pn_model *model)
{
#ifdef USE_CUDA
    return pn_qwen35moe_prefill_pipeline_wide_enabled() &&
           sg_prefill_batch_enabled() && sg_qwen35_batch_prefill_model(model);
#else
    (void)model;
    return 0;
#endif
}

/* Adapter: the model layer polls abort through a plain int(*)(void*), which is
 * what sg_should_abort_fn already is — but only when the caller supplied one. */
typedef struct {
    sg_should_abort_fn fn;
    void              *ud;
} sg_wide_abort;

static int sg_wide_abort_poll(void *p)
{
    sg_wide_abort *a = (sg_wide_abort *)p;
    return (a && a->fn) ? a->fn(a->ud) : 0;
}

/* Returns 1 if the wide pipeline ran the range (out_logits set to the last
 * position's logits, or NULL on abort), 0 if it refused and the caller must run
 * the ordinary chunked loop. */
static int sg_prefill_range_wide(pn_model *model, pn_state *state,
                                 const uint32_t *prompt, int from, int to,
                                 sg_should_abort_fn should_abort, void *ud,
                                 tm_tensor **out_logits)
{
#ifdef USE_CUDA
    *out_logits = NULL;
    if (!sg_prefill_wide_ready(model))
        return 0;
    int chunk = sg_prefill_chunk();
    if (chunk <= 0) chunk = to - from;
    sg_wide_abort abort_ctx = { should_abort, ud };
    int rc = pn_qwen35moe_prefill_range_pipelined_cuda(
        model, prompt + from, to - from, state, chunk,
        should_abort ? sg_wide_abort_poll : NULL, &abort_ctx, out_logits);
    if (rc < 0)
        return 0;                        /* refused — nothing advanced */
    if (rc == 1) {
        fprintf(stderr, "[prefill_pipeline_wide] aborted and drained\n");
        return 1;                        /* *out_logits stays NULL */
    }
    /* The sentinel a wide run - and only a wide run - prints. It is emitted
     * from the branch that consumed the pipelined result, so a binary that
     * ignored the flag, or a model that refused the entry, cannot print it. */
    fprintf(stderr, "[prefill_pipeline_wide] ON: %d tokens in %d-token chunks\n",
            to - from, chunk);
    return 1;
#else
    (void)model; (void)state; (void)prompt; (void)from; (void)to;
    (void)should_abort; (void)ud;
    *out_logits = NULL;
    return 0;
#endif
}

/* ── Partial overlap: QWEN35_PREFILL_PIPELINE_DUTY, default 100 (= today).
 *
 * A K80 is two GK210 dies on ONE board sharing ONE 300 W budget. Running both
 * at once — which is exactly what the pipeline above does — puts prefill at
 * ~287 W of that 300 and the board answers by taking the clock away: the
 * downstream die averages 584 MHz of its 875 MHz application clock through a
 * 14k ingest. Throughput is roughly
 * `dies_busy x clock`, and only the two extremes have ever been run: one die at
 * ~875, or two dies at ~584. The middle is unmeasured, and if a duty cycle
 * exists that holds the board just under its cap it could beat both.
 *
 * The duty is that knob. It is the percentage of a range's intermediate chunks
 * whose die-B segment is left IN FLIGHT when the next chunk's die-A segment is
 * issued — i.e. the fraction of chunk boundaries that overlap the two dies.
 * 100 is the shipped pipeline (every boundary overlaps), 0 serialises every
 * boundary. Boundaries are picked by a Bresenham accumulator so the overlapped
 * ones are spread evenly through the range instead of bunched at the front,
 * which is what makes the board's response a duty cycle rather than a ramp.
 *
 * A non-overlapped chunk still takes the no-head entry and is then drained
 * explicitly, so the ONLY difference between duty points is issue order — the
 * discarded LM head of an intermediate chunk is not computed at any duty. That
 * keeps the knob a pure overlap knob; comparing duty=0 against
 * QWEN35_PREFILL_PIPELINE=0 prices the head separately.
 *
 * DUTY_FILE is a bench-only escape hatch: the duty is re-read from that file at
 * the start of every prefill range, so a sweep can change it under a serve that
 * already holds 23 GB of weights instead of paying a reload per arm. Unset by
 * default, and the file is only consulted when the pipeline is on. ── */
typedef struct {
    int percent;      /* 0..100 — share of boundaries left overlapping */
    int accumulator;  /* Bresenham state, in percent-units */
    int overlapped;   /* boundaries left in flight */
    int serialised;   /* boundaries drained on purpose */
} sg_prefill_duty;

static int sg_prefill_duty_percent(void)
{
    static int env_percent = -1;
    static const char *duty_path = NULL;
    if (env_percent < 0) {
        const char *e = getenv("QWEN35_PREFILL_PIPELINE_DUTY");
        env_percent = e ? atoi(e) : 100;
        if (env_percent < 0) env_percent = 0;
        if (env_percent > 100) env_percent = 100;
        duty_path = getenv("QWEN35_PREFILL_PIPELINE_DUTY_FILE");
        if (duty_path && !duty_path[0])
            duty_path = NULL;
    }
    if (!duty_path)
        return env_percent;
    FILE *f = fopen(duty_path, "r");
    if (!f)
        return env_percent;
    int from_file = -1;
    if (fscanf(f, "%d", &from_file) != 1)
        from_file = -1;
    fclose(f);
    if (from_file < 0)
        return env_percent;
    return from_file > 100 ? 100 : from_file;
}

static void sg_prefill_duty_begin(sg_prefill_duty *duty, int pipeline_on)
{
    duty->percent = pipeline_on ? sg_prefill_duty_percent() : 100;
    duty->accumulator = 0;
    duty->overlapped = 0;
    duty->serialised = 0;
}

/* 1 = leave die B in flight across this chunk boundary, 0 = drain it first. */
static int sg_prefill_duty_overlap(sg_prefill_duty *duty)
{
    duty->accumulator += duty->percent;
    if (duty->accumulator >= 100) {
        duty->accumulator -= 100;
        duty->overlapped++;
        return 1;
    }
    duty->serialised++;
    return 0;
}

/* PN_PREFILL_LOGITS_HASH=1 — the bit-exactness instrument for Phase 9b.
 *
 * FNV-1a over the raw bytes of the logits vector the prefill produced, i.e.
 * over the exact float32 pattern the whole KV + SSM state funnelled into.  A
 * one-ULP divergence anywhere in the prefill changes it.  Off by default and
 * free when off; it is a diagnostic, not a code path.
 *
 * Also prints prefilled-token count + n_vocab so a hash can never be compared
 * across two different amounts of work by accident. */
static void sg_prefill_logits_hash(const tm_tensor *logits, int prefill_tokens,
                                   int pos)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PN_PREFILL_LOGITS_HASH");
        on = (e && e[0] == '1') ? 1 : 0;
    }
    if (!on || !logits)
        return;
    const unsigned char *p = (const unsigned char *)tm_data_f32((tm_tensor *)logits);
    size_t n = (size_t)tm_numel((tm_tensor *)logits) * sizeof(float);
    if (!p || n == 0)
        return;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    fprintf(stderr, "[prefill_hash] tokens=%d pos=%d nfloat=%zu fnv1a=%016llx\n",
            prefill_tokens, pos, n / sizeof(float), (unsigned long long)h);
}

/* The 9b sentinel.  A gate that reads green because the flag was never picked
 * up is this project's most repeated failure, so the pipelined path — and ONLY
 * the pipelined path — prints proof of itself: the count is incremented inside
 * the loop by a successful no-head chunk, so it cannot be printed by a binary
 * that ignored QWEN35_PREFILL_PIPELINE or by a model that refused the entry.
 *
 * ⚠ What it does NOT prove: that die B was actually left IN FLIGHT.  It counts
 * the code path this loop chose, and QWEN35_PROF=1 drains die B inside every
 * chunk regardless (see qw_warn_prof_serialises_pipeline in
 * qwen35moe_forward_cuda.c), so a fully serialised profiled run still prints
 * `duty=100% serialised=0`.  Overlap is only observable from outside the
 * process — both dies at high utilisation in the same 200 ms sample. */
static void sg_prefill_pipeline_report(const sg_prefill_duty *duty, int n_tok)
{
    if (duty->overlapped <= 0 && duty->serialised <= 0)
        return;
    fprintf(stderr, "[prefill_pipeline] ON: %d of the range's chunks ran "
                    "no-head + die-B-in-flight (%d tokens) duty=%d%% "
                    "serialised=%d\n",
            duty->overlapped, n_tok, duty->percent, duty->serialised);
}

static tm_tensor *sg_prompt_chunk(pn_model *model, pn_state *state,
                                  const uint32_t *toks, int n)
{
#ifdef USE_CUDA
    /* A dense-FFN model (Ornith, Bonsai) runs pn_bonsai_forward_cuda, not this
     * pointer. It takes the batched prefill when every weight has a batched
     * kernel (Ornith Q8_0); Bonsai's native ternary Q2_0 has none and keeps the
     * per-token loop below — byte-identical to sequential forwards. */
    if (n > 1 && sg_prefill_batch_enabled() &&
        sg_qwen35_batch_prefill_model(model))
        return pn_qwen35moe_forward_batch_cuda(model, toks, n, state, NULL);
#endif
#ifdef USE_TT
    /* The Tenstorrent route's batched prefill. A tile is 32 rows and a matvec is Mt=1, so
     * the card already multiplies 32 rows against every weight tile it reads and 31 of them
     * are zero padding — this fills them with prompt positions at no card cost. It REFUSES
     * (returns NULL without touching state) for on-card attention, the diagnostic arms and
     * a prompt that would wrap the ring, and the loop below is what runs then. */
    if (n > 1 && model->forward == pn_llama_forward_tt) {
        tm_tensor *batched = pn_llama_prefill_tt(model, toks, NULL, n, state);

        if (batched)
            return batched;
    }
    /* AND ORNITH'S, which is a DIFFERENT function because it is a different trade — both of
     * its mixers run whole on the card and are single-position there, so its batch reaches
     * each row through the link and its win comes from the chunkwise GDN mixers.
     *
     * IT WAS MISSING AND THAT IS WHY THE SERVE NEVER SAW THE 24x. `pn_qwen35moe_prefill_tt`
     * shipped on by default on 2026-08-25 and this file only ever asked the llama family,
     * so every Ornith prefill the fleet ran went through the per-token loop below — no
     * error, no refusal line, just ~20x the wall. */
    if (n > 1 && model->forward == pn_qwen35moe_forward_tt) {
        tm_tensor *batched = pn_qwen35moe_prefill_tt(model, toks, n, state);

        /* An ABORT invalidated the session (its card state is half written): fail, never
         * replay per token over it. */
        if (batched || pn_qwen35moe_prefill_tt_aborted(model))
            return batched;
    }
#endif
    tm_tensor *logits = NULL;
    for (int i = 0; i < n; i++) {
        if (logits) tm_free(logits);
        logits = sg_forward_step(model, state, toks[i]);
        if (!logits)
            return NULL;
    }
    return logits;
}

/* Prefill prompt[from, to) through the chunked batched path, returning the
 * LAST position's logits (owned; caller frees), or NULL on failure. Chunk
 * boundaries never change the result (the batched prefill is bit-exact vs
 * sequential forwards), so splitting a prefill at the system boundary (SYS1)
 * is bit-faithful to one uninterrupted pass. Returns NULL when from >= to. */
/* The CUDA MTP head's backlog follows every trunk batch of the prompt (serve_spec_mtp_cuda.h).
 * A failure only costs drafts (logged once): the verify still decides every token. */
static void sg_mtp_feed_chunk(pn_model *model, pn_state *state, const uint32_t *toks, int n)
{
#ifdef USE_CUDA
    static int warned;

    if (!sg_mtp_cuda_available(model) ||
        pn_qwen35moe_mtp_push_rows(model, state, toks, n) == 0 || warned)
        return;
    warned = 1;
    fprintf(stderr, "serve: MTP head backlog push failed (chunk %d tokens at pos %d); drafts "
                    "restart after the next verify\n", n, state->pos);
#else
    (void)model; (void)state; (void)toks; (void)n;
#endif
}

static tm_tensor *sg_prefill_range(pn_model *model, pn_state *state,
                                   const uint32_t *prompt, int from, int to,
                                   sg_should_abort_fn should_abort, void *ud)
{
    tm_tensor *logits = NULL;
    /* 9c first: it owns the whole range when it takes it, and refuses cleanly
     * (nothing advanced) when it cannot, leaving the 9b loop below untouched.
     * With the MTP head on, 9c feeds the head's backlog itself. */
    if (sg_prefill_range_wide(model, state, prompt, from, to, should_abort, ud,
                              &logits))
        return logits;
    int chunk = sg_prefill_chunk();
    if (chunk <= 0) chunk = to - from;
#ifdef USE_TT
    /* The Tenstorrent qwen35 batched prefill starts at position 0 or at a restored checkpoint
     * and declines any other continuation: a prompt goes in ONE range, or every chunk after the
     * first ran per token (1,630 tokens: 78 s = 20.8 tok/s against the batched ~400). */
    if (model->forward == pn_qwen35moe_forward_tt)
        chunk = to - from;
#endif
    int pipeline = sg_prefill_pipeline_ready(model);
    sg_prefill_duty duty;
    sg_prefill_duty_begin(&duty, pipeline);
    for (int off = from; off < to; off += chunk) {
        /* Bail between chunks the instant the client disconnects — a cold
         * prefill is ~100s on a big model, and finishing it just wedges the
         * single slot for a response nobody will read. Returns NULL; the caller
         * classifies abort vs error via the same should_abort. */
        if (should_abort && should_abort(ud)) {
            if (pipeline) sg_prefill_pipeline_drain(model);
            if (logits) tm_free(logits);
            return NULL;
        }
        int m = to - off < chunk ? to - off : chunk;
        if (logits) tm_free(logits);
        logits = NULL;
        /* 9b: intermediate chunks skip the head and leave die B in flight,
         * except on the share of boundaries the duty serialises (drain). */
        if (pipeline && m > 1 && off + m < to &&
            sg_prompt_chunk_nologits(model, state, prompt + off, m) == 0) {
            sg_mtp_feed_chunk(model, state, prompt + off, m);
            if (!sg_prefill_duty_overlap(&duty))
                sg_prefill_pipeline_drain(model);
            continue;
        }
        logits = sg_prompt_chunk(model, state, prompt + off, m);
        if (!logits) {
            if (pipeline) sg_prefill_pipeline_drain(model);
            return NULL;
        }
        sg_mtp_feed_chunk(model, state, prompt + off, m);
    }
    /* The last chunk always runs the ordinary path, which syncs die B, so the
     * pipeline is drained on the success return by construction. */
    sg_prefill_pipeline_report(&duty, to - from);
    return logits;
}

/* QWEN35_BS_PREFILL_WIDE_SUB=<tokens> (default 1024 since 2026-11-02; 0 = off):
 * while other sessions are decoding, prefill in sub-ranges of this many tokens
 * through the N-die wide pipeline (bit-exact with the chunk loop; every die
 * busy instead of one) and yield to the decode driver between sub-ranges,
 * instead of one sequential 128-token chunk per yield. Process-constant. */
#define SG_PREFILL_WIDE_SUB_DEFAULT 1024

static int sg_prefill_wide_sub(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN35_BS_PREFILL_WIDE_SUB");
        v = e ? atoi(e) : SG_PREFILL_WIDE_SUB_DEFAULT;
        if (v < 0)
            v = 0;
    }
    return v;
}

/* Prefill prompt[*from_io, to) in wide-pipelined sub-ranges of `sub` tokens,
 * yielding the GPU turn to the decode driver between sub-ranges (never after
 * the last, for the same reason as sg_prefill_range_sched). Returns 1 when it
 * handled the range: *out holds the last position's logits, or NULL on an abort
 * (the caller classifies via should_abort). Returns 0 when the wide path
 * refused; *from_io then says how far it got, and the caller's chunk loop
 * prefills the rest. */
static int sg_prefill_range_wide_yielding(pn_model *model, pn_batch_sched *sched,
                                          int slot, pn_state *state,
                                          const uint32_t *prompt, int *from_io,
                                          int to, int sub,
                                          sg_should_abort_fn should_abort, void *ud,
                                          tm_tensor **out)
{
    *out = NULL;
    if (!sg_prefill_wide_ready(model))
        return 0;
    for (int off = *from_io; off < to; off += sub) {
        const int end = to - off <= sub ? to : off + sub;
        tm_tensor *logits = NULL;
        *from_io = off;
        if (!sg_prefill_range_wide(model, state, prompt, off, end, should_abort, ud,
                                   &logits))
            return 0;
        if (!logits)
            return 1;                    /* aborted and drained */
        if (end == to) {
            *out = logits;
            return 1;
        }
        tm_free(logits);
        bs_prefill_yield(sched, slot);
    }
    return 1;
}

/* Ranges shorter than this skip the overlap: the cursor would spread their two
 * stages' 40 layers over several decode steps, where the ordinary path runs a
 * 5-token checkpoint tail or a short tool reply in a fraction of a second. */
#define SG_PD_OVERLAP_MIN_TOKENS 256

/* QWEN35_BS_PD_OVERLAP: prefill prompt[from, to) through the prefill cursor
 * while other sessions decode — each decode step (bs_pd_cycle, the driver's
 * very step run by this thread) carries prefill pieces in the boards' idle
 * halves; with nobody decoding the cursor runs alone, a chunk at a time. A
 * queued finalize, or a waiting prefill clearly shorter than what is left
 * (shortest-prefill-first), gets the turn at a quiet point (cursor quiesced,
 * prefill saved) while chunks are left to inject; after the last one the range
 * ends under this turn (bs_pd_may_yield). Returns 1 when it handled the range: *out holds the last position's
 * logits, or NULL on an abort or an error (the caller classifies via
 * should_abort; on an error the state is undefined and the caller resets).
 * Returns 0 when the cursor refused (nothing touched). Bit-exact with the wide
 * path. */
static int sg_prefill_range_overlap(pn_model *model, pn_batch_sched *sched, int slot,
                                    pn_state *state, const uint32_t *prompt, int from,
                                    int to, sg_should_abort_fn should_abort, void *ud,
                                    tm_tensor **out)
{
    *out = NULL;
#ifdef USE_CUDA
    int chunk = sg_prefill_chunk();
    if (chunk <= 0)
        chunk = to - from;
    pn_qwen35moe_pd_cursor *cur =
        pn_qwen35moe_pd_begin_cuda(model, state, prompt + from, to - from, chunk);
    if (!cur)
        return 0;
    int cycles = 0, alone = 0, yields = 0, held = 0;
    while (!pn_qwen35moe_pd_issued_all(cur)) {
        if (should_abort && should_abort(ud)) {
            (void)pn_qwen35moe_pd_abort_cuda(cur);
            fprintf(stderr, "[prefill_overlap] aborted and drained\n");
            return 1;
        }
        const int finalizers = bs_pd_finalize_waiting(sched);
        const bool shorter = finalizers == 0 &&
            bs_pd_shorter_waiting(sched, pn_qwen35moe_pd_remaining(cur));
        if (bs_pd_may_yield(pn_qwen35moe_pd_injected_all(cur), finalizers, shorter)) {
            if (pn_qwen35moe_pd_quiesce_cuda(cur) < 0)
                break;
            bs_pd_yield(sched, slot);
            yields++;
        } else if (finalizers > 0 || shorter) {
            held++;   /* last chunk in: the range ends under this turn */
        }
        int r = bs_pd_cycle(sched, cur);
        if (r < 0)
            break;
        if (r == 1 && pn_qwen35moe_pd_advance_cuda(cur, 1) < 0)
            break;
        cycles += r == 0;
        alone += r == 1;
    }
    if (!pn_qwen35moe_pd_issued_all(cur)) {
        (void)pn_qwen35moe_pd_abort_cuda(cur);
        fprintf(stderr, "[prefill_overlap] failed after %d cycles\n", cycles);
        return 1;
    }
    if (pn_qwen35moe_pd_finish_cuda(cur, out) < 0) {
        fprintf(stderr, "[prefill_overlap] head failed\n");
        return 1;
    }
    fprintf(stderr, "[prefill_overlap] ON: %d tokens in %d-token chunks, %d overlap "
            "cycles, %d chunks alone, %d yields, %d tail cycles held\n", to - from, chunk,
            cycles, alone, yields, held);
    return 1;
#else
    (void)model; (void)sched; (void)slot; (void)state; (void)prompt; (void)from; (void)to;
    (void)should_abort; (void)ud;
    return 0;
#endif
}

/* Batch-scheduled twin of sg_prefill_range: prefill prompt[from, to) in
 * chunks, yielding the GPU turn to the decode driver BETWEEN chunks (L2) but
 * never after the last one — the caller's next action (a boundary capture or
 * the slot bind) must directly follow the final token under the same turn.
 * Stage boundaries (SYS1's split at sys_len) therefore behave exactly like
 * chunk boundaries without a yield, which is bit-faithful to one
 * uninterrupted pass.  Returns the LAST position's logits (owned; caller
 * frees), or NULL on failure / when from >= to. */
static tm_tensor *sg_prefill_range_sched(pn_model *model,
                                         pn_batch_sched *sched, int slot,
                                         pn_state *state,
                                         const uint32_t *prompt,
                                         int from, int to,
                                         sg_should_abort_fn should_abort,
                                         void *ud)
{
    tm_tensor *logits = NULL;
    /* 9c owns the whole range when it takes it, and therefore skips the L2
     * yield windows the loop below opens between chunks. That is only
     * acceptable when this slot is the ONLY active one — with anyone else on
     * the model, a long range would freeze their decode for its whole
     * duration, which is exactly what L2 exists to prevent. */
    bs_load load = {0};
    if (sched)
        bs_get_load(sched, &load);
    if (load.active_slots <= 1 &&
        sg_prefill_range_wide(model, state, prompt, from, to, should_abort, ud,
                              &logits))
        return logits;
    if (sched && load.active_slots > 1 && to - from >= SG_PD_OVERLAP_MIN_TOKENS &&
        bs_pd_overlap_on(sched) &&
        sg_prefill_range_overlap(model, sched, slot, state, prompt, from, to, should_abort,
                                 ud, &logits))
        return logits;
    if (sched && sg_prefill_wide_sub() > 0 &&
        sg_prefill_range_wide_yielding(model, sched, slot, state, prompt, &from, to,
                                       sg_prefill_wide_sub(), should_abort, ud,
                                       &logits))
        return logits;
    int chunk = sg_prefill_chunk();
    if (chunk <= 0) chunk = to - from;
    int pipeline = sg_prefill_pipeline_ready(model);
    sg_prefill_duty duty;
    sg_prefill_duty_begin(&duty, pipeline);
    for (int off = from; off < to; off += chunk) {
        /* Bail between chunks the instant the client disconnects — the same
         * check the single-user twin has had, and it matters MORE here: this is
         * the path the fleet runs, so a cold prefill nobody will read (~205 s at
         * 8k) pins one of the few decode slots for its whole duration while
         * every other user shares what is left.  One chunk is ~4-5 s, so an
         * abandoned request costs seconds instead of minutes.  Returns NULL; the
         * caller classifies abort vs error via the same should_abort. */
        if (should_abort && should_abort(ud)) {
            if (pipeline) sg_prefill_pipeline_drain(model);
            if (logits) tm_free(logits);
            return NULL;
        }
        int m = to - off < chunk ? to - off : chunk;
        if (logits) tm_free(logits);
        logits = NULL;
        /* 9b: intermediate chunks skip the head and leave die B in flight.
         * bs_prefill_yield below is safe on top of that — it only hands the
         * GPU over after pn_qwen35moe_gpu_state_to_host, which walks both dies
         * with fc_sync_ctx and therefore drains the pipeline itself. It also
         * only fires when another slot is actually runnable, so the
         * single-session case this optimises never pays for it. */
        if (pipeline && m > 1 && off + m < to) {
            if (sg_prompt_chunk_nologits(model, state, prompt + off, m) == 0) {
                sg_mtp_feed_chunk(model, state, prompt + off, m);
                if (!sg_prefill_duty_overlap(&duty))
                    sg_prefill_pipeline_drain(model);
                bs_prefill_yield(sched, slot);
                continue;
            }
        }
        logits = sg_prompt_chunk(model, state, prompt + off, m);
        if (!logits) {
            if (pipeline) sg_prefill_pipeline_drain(model);
            return NULL;
        }
        sg_mtp_feed_chunk(model, state, prompt + off, m);
        if (off + m < to)
            bs_prefill_yield(sched, slot);   /* L2: let decode interleave */
    }
    sg_prefill_pipeline_report(&duty, to - from);
    return logits;
}

/* Prefill prompt[from, to) on the batched path, stopping at up to three cuts
 * to save the state there:
 *   sys_cut   the SYS1 system-prefix boundary snapshot (-1: none);
 *   ckpt_cut  the turn-start checkpoint, rung 1c (-1: none);
 *   div_cut   the divergence checkpoint, rung 1d (-1: none).
 * At a cut the GPU turn is still held (the range helper never yields after its
 * last chunk), so the state sits exactly there. Splitting is bit-faithful to
 * one uninterrupted pass (the batched prefill is bit-exact vs sequential
 * forwards at any chunking). A failed save is benign: the turn continues
 * without it. Returns the last position's logits, or NULL on abort/error. */
static tm_tensor *sg_prefill_turn_sched(pn_model *model, pn_batch_sched *sched,
                                        int slot, pn_state *state,
                                        const uint32_t *prompt, int from,
                                        int to, int sys_cut, int ckpt_cut,
                                        int div_cut, const char *sid,
                                        const sg_gen_params *params)
{
    int cuts[3];
    int n_cuts = 0;
    const int wanted[3] = { sys_cut, ckpt_cut, div_cut };
    for (int w = 0; w < 3; w++) {
        int c = wanted[w];
        bool dup = false;
        for (int k = 0; k < n_cuts; k++)
            dup = dup || cuts[k] == c;
        if (c <= from || c >= to || dup)
            continue;
        int k = n_cuts++;                    /* insertion sort, ascending */
        while (k > 0 && cuts[k - 1] > c) {
            cuts[k] = cuts[k - 1];
            k--;
        }
        cuts[k] = c;
    }
    tm_tensor *logits = NULL;
    int at = from;
    for (int i = 0; i <= n_cuts; i++) {
        int end = i < n_cuts ? cuts[i] : to;
        if (logits)
            tm_free(logits);
        logits = sg_prefill_range_sched(model, sched, slot, state, prompt, at,
                                        end, params->should_abort,
                                        params->abort_ud);
        if (!logits)
            return NULL;
        if (end == sys_cut &&
            bs_prefill_capture_sys(sched, slot, sid, prompt, sys_cut) != 0)
            fprintf(stderr, "[batch_sched] sys-prefix capture failed slot=%d "
                    "sid=%s (continuing uncached)\n", slot, sid);
        if (end == ckpt_cut && end < to)
            (void)bs_turn_ckpt_capture(sched, slot, sid, prompt, ckpt_cut);
        if (end == div_cut && end < to)
            (void)bs_div_ckpt_capture(sched, slot, sid, prompt, div_cut);
        at = end;
    }
    return logits;
}

/* Fill the repeat-penalty window with the prompt tail — identical result to
 * the historical per-token append loop (the window ends holding the last
 * max_recent prompt tokens, in order). */
static int sg_seed_recent(uint32_t *recent, int max_recent,
                          const uint32_t *prompt, int n_prompt)
{
    if (max_recent <= 0)
        return 0;
    int take = n_prompt < max_recent ? n_prompt : max_recent;
    memcpy(recent, prompt + (n_prompt - take),
           (size_t)take * sizeof(uint32_t));
    return take;
}

/* ============================================================
 * Completion driver
 * ============================================================ */

/* The single-user speculative loop runs on Tenstorrent (the TT step) and CUDA (the MTP step in
 * serve_spec_mtp_cuda.h); everything else decodes plain. */
#if defined(USE_TT) || defined(USE_CUDA)
#define SG_SPEC 1
#endif

#ifdef SG_SPEC
/* ---------------------------------------------------------------------------
 * MTP speculative decode on the single-user completion (Tenstorrent qwen35, and CUDA
 * qwen35moe with the MTP head: serve_spec_mtp_cuda.h, exact for sampled requests too).
 *
 * The loop's shape is kept: every token is emitted by the same code; only the
 * forward that follows an emitted token changes. After the seed (the token just
 * emitted, not yet in the state) the verify commits the seed + the accepted drafts
 * and returns them + the model's next token; the committed ones are then emitted
 * with NO forward, the last (uncommitted) one becomes the next seed. Lossless only
 * when the request's pick IS the plain argmax (temperature 0, no bias, penalties,
 * grammar or logprobs). Drafts are cut so a cycle can never commit past a stop
 * token, the token budget or the reasoning budget's trip: the state then stays
 * level with the emitted history except on mid-cycle stops (stop strings, repeat
 * loop, abort), where the committed extras are recorded in the history commit.
 * ------------------------------------------------------------------------- */
typedef struct sg_spec {
    int k;                 /* drafts a cycle; 0 = off */
    sg_spec_lookup lookup; /* prompt-lookup drafts, tried ahead of the MTP head */
    uint32_t emit[16];     /* accepted drafts, then the model's next token */
    int n, at;             /* emit[at..n-2] committed, emit[n-1] the next seed */
    bool sampling;         /* SPECULATIVE SAMPLING: rows drawn by pn_spec_sample_choose */
    pn_spec_sample_ctx choose;
    /* UNDER THE TOOL GRAMMAR (tool_choice "required", grammar_spec_chooser.h):
     * drafts are cut where the grammar refuses them and every verify row is
     * chosen through gsc_choose, so the output is plain constrained decoding's. */
    bool grammar;
    pn_sampling_params plain;  /* the request's params without the grammar hook */
    pn_sampling_params *loop;  /* the decode loop's params: `plain` borrows its seeded
                                * RNG state for a cycle and hands it back after */
    gsc_chooser gsc;
    void *gsc_walk;            /* the chooser's scratch walk (a clone of the live one) */
    sg_grammar_kind gsc_kind;  /* which grammar gsc_walk is */
    const sg_tg_ctx *tg;
    pn_model *model;
    float *dense;              /* n_vocab: the row a masked pick samples */
    sg_spec_census census;     /* where the request's tokens came from (serve_spec_census.h) */
    sg_plain_reason why_plain; /* why the last sg_spec_cycle returned false */
#ifdef USE_CUDA
    bool cuda;                 /* the CUDA MTP backend runs this request */
    sg_mtp_cuda mtp;
#endif
} sg_spec;
#endif /* SG_SPEC */

#ifdef USE_TT

/* The grammar walk behind gsc_grammar_ops, per dialect. */
static void sg_gs_copy_json(void *d, const void *s) { tgj_state_copy(d, s); }
static bool sg_gs_allowed_json(const void *st, const char *t) { return tgj_token_allowed(st, t); }
static void sg_gs_advance_json(void *st, const char *t) { tgj_advance(st, t); }
static bool sg_gs_mask_json(const void *st) { return tgj_mask_active(st); }
static bool sg_gs_complete_json(const void *st) { return tgj_complete(st); }
static void sg_gs_copy_xml(void *d, const void *s) { tg_state_copy(d, s); }
static bool sg_gs_allowed_xml(const void *st, const char *t) { return tg_token_allowed(st, t); }
static void sg_gs_advance_xml(void *st, const char *t) { tg_advance(st, t); }
static bool sg_gs_mask_xml(const void *st) { return tg_mask_active(st); }
static bool sg_gs_complete_xml(const void *st) { return tg_complete(st); }
static const gsc_grammar_ops SG_GS_OPS_JSON = {
    sg_gs_copy_json, sg_gs_allowed_json, sg_gs_advance_json, sg_gs_mask_json, sg_gs_complete_json,
};
static const gsc_grammar_ops SG_GS_OPS_XML = {
    sg_gs_copy_xml, sg_gs_allowed_xml, sg_gs_advance_xml, sg_gs_mask_xml, sg_gs_complete_xml,
};
/* RED `gspec_ignore_mask` (PN_TT_RED): the chooser sees the mask as never
 * active, so speculative output is unconstrained while plain decode is
 * constrained — the spec-on vs spec-off comparison MUST then differ. */
static bool sg_gs_mask_never(const void *st) { (void)st; return false; }
static const gsc_grammar_ops SG_GS_OPS_JSON_RED = {
    sg_gs_copy_json, sg_gs_allowed_json, sg_gs_advance_json, sg_gs_mask_never, sg_gs_complete_json,
};
static const gsc_grammar_ops SG_GS_OPS_XML_RED = {
    sg_gs_copy_xml, sg_gs_allowed_xml, sg_gs_advance_xml, sg_gs_mask_never, sg_gs_complete_xml,
};
static void sg_gs_copy_rf(void *d, const void *s) { rfg_state_copy(d, s); }
static bool sg_gs_allowed_rf(const void *st, const char *t) { return rfg_token_allowed(st, t); }
static void sg_gs_advance_rf(void *st, const char *t) { rfg_advance(st, t); }
static bool sg_gs_mask_rf(const void *st) { return rfg_mask_active(st); }
static bool sg_gs_complete_rf(const void *st) { return rfg_complete(st); }
static const gsc_grammar_ops SG_GS_OPS_RF = {
    sg_gs_copy_rf, sg_gs_allowed_rf, sg_gs_advance_rf, sg_gs_mask_rf, sg_gs_complete_rf,
};
static const gsc_grammar_ops SG_GS_OPS_RF_RED = {
    sg_gs_copy_rf, sg_gs_allowed_rf, sg_gs_advance_rf, sg_gs_mask_never, sg_gs_complete_rf,
};

/* A verify row the mask does not touch: the plain speculative-sampling draw. */
static uint32_t sg_gs_base_sample(void *ctx, int row, const uint32_t *index, const float *value,
                                  uint32_t count)
{
    pn_spec_sample_ctx *choose = (pn_spec_sample_ctx *)ctx;

    choose->row_index = row;
    return pn_spec_sample_choose(choose, index, value, count);
}

/* Push a draw onto the cycle's repeat window, as pn_spec_sample_choose does. */
static void sg_gs_push_recent(pn_spec_sample_ctx *choose, uint32_t token)
{
    if (choose->max_recent <= 0) return;
    if (choose->n_recent < choose->max_recent) {
        choose->recent[choose->n_recent++] = token;
    } else {
        memmove(choose->recent, choose->recent + 1,
                (size_t)(choose->max_recent - 1) * sizeof(uint32_t));
        choose->recent[choose->max_recent - 1] = token;
    }
}

/* A verify row under an active mask: the token plain constrained decoding draws
 * at this position, or GSC_DEFER (rows > 0 only).  The row is handed over as
 * plain decode's sampler receives it — the top-N pairs, -inf elsewhere.
 *   row 0  : sample_token itself, with the real dense callbacks — row 0 of the
 *            verify IS the card's logits row 0, so the shortlist / chunk fetch /
 *            dense refill it falls through are the plain step's own.
 *   row > 0: only the shortlist decision (the card's logits buffer holds row 0,
 *            not this row); decided → the same draw on the same rewritten row
 *            (pn_sample_decided_row: final, as row 0's own mask leaves it), else
 *            defer — the next cycle meets this position as its row 0. */
static uint32_t sg_gs_masked_pick(void *ctx, void *walk, int row, const uint32_t *index,
                                  const float *value, uint32_t count)
{
    sg_spec *sp = (sg_spec *)ctx;
    const int n_vocab = sp->model->config.n_vocab;
    const uint32_t neg_inf_bits = 0xff800000u;
    pn_tool_grammar_hook hook = sp->tg->hook;
    pn_sampling_params params = sp->plain;
    pn_spec_sample_ctx *choose = &sp->choose;
    uint32_t i, token;
    float neg_inf;

    hook.st = walk;
    hook.active = 1;
    hook.refresh = NULL;
    hook.advance = NULL;
    params.tool_grammar = &hook;
    /* A masked row: no repetition penalty, as in plain decode's sample_token.
     * Set here too because row > 0 is decided with the hook detached. */
    params.repeat_penalty = 1.0f;
    memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));
    for (i = 0; i < (uint32_t)n_vocab; i++) sp->dense[i] = neg_inf;
    for (i = 0; i < count; i++)
        if (index[i] < (uint32_t)n_vocab) sp->dense[index[i]] = value[i];
    if (row > 0) {
        params.dense_row = sg_gs_row_already_dense;   /* never row 0's logits here */
        params.tail_counts = NULL;
        params.dense_chunks = NULL;
        params.dense_ctx = NULL;
        if (!pn_sampler_grammar_shortlist(sp->dense, n_vocab, &params, choose->recent,
                                          choose->n_recent))
            return GSC_DEFER;
        params.tool_grammar = NULL;
    } else {
        params.dense_row = pn_qwen35moe_tt_dense_row;
        params.tail_counts = pn_qwen35moe_tt_tail_counts;
        params.dense_chunks = pn_qwen35moe_tt_dense_chunks;
        params.dense_ctx = sp->model;
    }
    token = row > 0 ? pn_sample_decided_row(sp->dense, n_vocab, &params, choose->recent,
                                            choose->n_recent)
                    : pn_sample_next_token(sp->dense, n_vocab, &params, choose->recent,
                                           choose->n_recent);
    sp->plain.rng_state = params.rng_state;
    sp->plain.carried_uniform_set = params.carried_uniform_set;
    sg_gs_push_recent(choose, token);
    return token;
}

/* Arm speculation under the tool grammar: the chooser, its scratch walk and the
 * dense-row buffer.  Returns false (spec stays off) when any piece is missing. */
static bool sg_spec_init_grammar(sg_spec *sp, pn_model *model, const sg_tg_ctx *tg,
                                 int max_recent)
{
    const bool red = getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "gspec_ignore_mask") == 0;

    if (!pn_qwen35moe_tt_spec_can_sample(model))
        return false;
    /* A field_patterns steer moves the walk from inside the sampler, which the
     * verify rows > 0 (decided by the shortlist alone) would not see: plain
     * constrained decoding only. */
    if (tg->hook.steer)
        return false;
    sp->gsc_kind = tg->kind;   /* tells sg_spec_free the walk's type */
    switch (tg->kind) {
    case SG_GRAMMAR_TOOL_JSON:
        sp->gsc.ops = red ? &SG_GS_OPS_JSON_RED : &SG_GS_OPS_JSON;
        sp->gsc_walk = tgj_state_clone(tg->state_json);
        sp->gsc.live = tg->state_json;
        break;
    case SG_GRAMMAR_TOOL_XML:
        sp->gsc.ops = red ? &SG_GS_OPS_XML_RED : &SG_GS_OPS_XML;
        sp->gsc_walk = tg_state_clone(tg->state);
        sp->gsc.live = tg->state;
        break;
    case SG_GRAMMAR_RESPONSE_JSON:
        sp->gsc.ops = red ? &SG_GS_OPS_RF_RED : &SG_GS_OPS_RF;
        sp->gsc_walk = rfg_state_clone(tg->state_rf);
        sp->gsc.live = tg->state_rf;
        break;
    default:
        return false;
    }
    sp->dense = malloc((size_t)model->config.n_vocab * sizeof(float));
    if (!sp->choose.recent) {
        sp->choose.recent = malloc((size_t)(max_recent > 0 ? max_recent : 1) * sizeof(uint32_t));
        sp->choose.max_recent = max_recent;
    }
    if (!sp->gsc_walk || !sp->dense || !sp->choose.recent)
        return false;
    sp->grammar = true;
    sp->tg = tg;
    sp->model = model;
    if (red)
        fprintf(stderr, "serve: RED gspec_ignore_mask — drafts are NOT checked against the grammar\n");
    sp->gsc.walk = sp->gsc_walk;
    sp->gsc.text = sg_tg_token_text;
    sp->gsc.text_ctx = (void *)tg;
    sp->gsc.base = sp->sampling ? sg_gs_base_sample : NULL;
    sp->gsc.base_ctx = &sp->choose;
    sp->gsc.masked = sg_gs_masked_pick;
    sp->gsc.masked_ctx = sp;
    return true;
}

/* One line per sampled request that drew on top-N rows past the top-k gate: how the tail
 * planner decided (sampler_topn_tail.h) and how many dense rows the request downloaded. */
static void sg_report_sampled_rows(const unsigned long tail0[4], unsigned long refills0)
{
    unsigned long t[3];

    pn_sampler_tail_counts(&t[0], &t[1], &t[2]);
    if (t[0] == tail0[0] && t[1] == tail0[1] && t[2] == tail0[2])
        return;
    fprintf(stderr, "[sampled rows] tail planner: from top-N %lu, residual %lu, dense %lu; "
                    "card tail counts %lu; dense refills %lu\n", t[0] - tail0[0],
            t[1] - tail0[1], t[2] - tail0[2], pn_sampler_tail_count_calls() - tail0[3],
            pn_sampler_topn_refills() - refills0);
}

#endif /* USE_TT */

#ifdef SG_SPEC
static void sg_spec_free(sg_spec *sp)
{
#ifdef USE_CUDA
    if (sp->cuda)
        sg_mtp_cuda_end(&sp->mtp);
    sp->cuda = false;
#endif
    pn_spec_sample_rule_free(&sp->choose);
    sg_spec_lookup_free(&sp->lookup);
    free(sp->choose.row);
    free(sp->choose.recent);
    sp->choose.row = NULL;
    sp->choose.recent = NULL;
    if (sp->gsc_walk) {
        if (sp->gsc_kind == SG_GRAMMAR_TOOL_JSON)      tgj_state_free(sp->gsc_walk);
        else if (sp->gsc_kind == SG_GRAMMAR_TOOL_XML)  tg_state_free(sp->gsc_walk);
        else                                           rfg_state_free(sp->gsc_walk);
    }
    sp->gsc_walk = NULL;
    free(sp->dense);
    sp->dense = NULL;
}

/* One line per constrained turn that speculated: how the grammar shaped it. */
static void sg_spec_report_grammar(const sg_spec *sp)
{
    if (!sp->grammar) return;
    fprintf(stderr, "[tool_grammar] speculation under the grammar: rows_masked=%lu "
            "deferred=%lu drafts_cut=%lu\n", sp->gsc.rows_masked, sp->gsc.rows_deferred,
            sp->gsc.drafts_cut);
}
#endif /* SG_SPEC */

#ifdef USE_TT
/* Why this request needs the dense row on every step (and so cannot speculate: a verify's
 * rows are top-N), or NULL. Same conditions sg_run_completion always used, now named. */
static const char *sg_spec_full_row_reason(const sg_tg_ctx *tg, const sg_gen_params *params,
                                           bool spec_under_grammar)
{
    if (params->logprobs)
        return "off:logprobs";
    if (params->logit_bias && params->n_logit_bias > 0 && !sg_tt_bias_on_topn(params))
        return "off:logit-bias";
    if (!spec_under_grammar && tg->enabled)
        return sg_tg_is_tool(tg) ? "off:tool-grammar" : "off:response-format";
    return NULL;
}

/* Greedy speculation when the request's pick IS the argmax; speculative SAMPLING when the
 * rows are top-N (pn_qwen35moe_tt_select_rows chose them) and nothing needs the dense row. */
static void sg_spec_init_tt(sg_spec *sp, pn_model *model, pn_sampling_params *samp,
                            const char *full_row_reason, int max_recent,
                            const uint32_t *prompt, int n_prompt, int max_tokens,
                            const sg_tg_ctx *tg, bool grammar_speculates)
{
    const char *env = getenv("PN_TT_SPEC");
    const bool grammar = grammar_speculates && samp->tool_grammar;

    memset(sp, 0, sizeof(*sp));
    sg_spec_census_begin(&sp->census, "off:not-tt");
    if (model->forward != pn_qwen35moe_forward_tt)
        return;
    if (full_row_reason) {
        sp->census.mode = full_row_reason;
        return;
    }
    if (getenv("PN_TT_SERVE_SPEC") && atoi(getenv("PN_TT_SERVE_SPEC")) == 0) {
        sp->census.mode = "off:env";
        return;
    }
    /* Under the grammar the rows are judged by the chooser, so exactness is
     * asked of the request's params WITHOUT the hook. */
    sp->plain = *samp;
    sp->loop = samp;
    if (grammar)
        sp->plain.tool_grammar = NULL;
    sp->census.mode = grammar ? "grammar-greedy" : "greedy";
    if (!pn_qwen35moe_tt_greedy_exact_on_topn(&sp->plain)) {
        /* Greedy under client surgery takes the chooser too: each verify row's pick is the
         * sampler's own (surgery over the cycle's window, dense refill / deferral when the
         * row cannot decide it). `PN_TT_TOPN_SURGERY=0`: plain decode, as before. */
        const bool greedy_surgery = sp->plain.temperature <= 0.0f &&
                                    pn_topn_surgery_enabled() &&
                                    pn_topn_surgery_present(&sp->plain);

        sp->census.mode = sp->plain.temperature <= 0.0f && !greedy_surgery ? "off:greedy-penalized"
                        : sp->plain.tool_grammar        ? "off:grammar"
                        : !pn_qwen35moe_tt_spec_can_sample(model) ? "off:dense-rows"
                        : "off:env-sample";
        if ((sp->plain.temperature <= 0.0f && !greedy_surgery) || sp->plain.tool_grammar ||
            !pn_qwen35moe_tt_spec_can_sample(model) ||
            (getenv("PN_TT_SPEC_SAMPLE") && atoi(getenv("PN_TT_SPEC_SAMPLE")) == 0))
            return;
        sp->choose.params = grammar ? &sp->plain : samp;
        sp->choose.n_vocab = model->config.n_vocab;
        sp->choose.max_recent = max_recent;
        sp->choose.row = malloc((size_t)model->config.n_vocab * sizeof(float));
        sp->choose.recent = malloc((size_t)(max_recent > 0 ? max_recent : 1) * sizeof(uint32_t));
        if (!sp->choose.row || !sp->choose.recent) {
            free(sp->choose.row);
            free(sp->choose.recent);
            memset(sp, 0, sizeof(*sp));
            sg_spec_census_begin(&sp->census, "off:no-memory");
            return;
        }
        sp->sampling = true;
        sp->census.mode = greedy_surgery ? (grammar ? "grammar-greedy-surgery" : "greedy-surgery")
                        : (grammar ? "grammar-sampled" : "sampled");
        /* TRUE SPECULATIVE SAMPLING outside the tool grammar (spec_sampling_chooser.h):
         * drafts drawn from the MTP head's q, rows decided by min(1, p/q) + residual.
         * Opt-in (`PN_TT_SPEC_SAMPLE_RULE=1`/`2`); off (or a grammar): argmax drafts, the
         * shared-randomness rule. */
        if (!grammar && pn_spec_rule_applies(sp->choose.params) &&
            pn_spec_sample_rule_arm(&sp->choose) != 0)
            fprintf(stderr, "serve: true speculative sampling off (no memory)\n");
    }
    if (grammar && !sg_spec_init_grammar(sp, model, tg, max_recent)) {
        fprintf(stderr, "serve: speculation under the tool grammar unavailable; plain decode\n");
        sg_spec_free(sp);
        memset(sp, 0, sizeof(*sp));
        sg_spec_census_begin(&sp->census, "off:grammar-unavailable");
        return;
    }
    sp->k = env ? atoi(env) : 3;
    if (sp->k > 15) sp->k = 15;
    if (sp->k < 0) sp->k = 0;
    if (sp->k == 0)
        sp->census.mode = "off:k0";
    if (sp->k > 0 && sg_spec_lookup_init(&sp->lookup, prompt, n_prompt, max_tokens) != 0)
        fprintf(stderr, "serve: prompt-lookup drafts off (no memory for the history)\n");
}


/* The Tenstorrent cycle behind sg_spec_cycle. */
static bool sg_spec_cycle_tt(sg_spec *sp, pn_model *model, pn_state *state, uint32_t seed,
                             int cap, int eos_id, int im_end, const uint32_t *recent,
                             int n_recent)
{
    uint32_t drafts[16], looked[16];
    int nd, nl, i, ne;

    sp->why_plain = sp->k > 0 ? SG_PLAIN_CAP
                  : sp->census.plain[SG_PLAIN_FAILED] ? SG_PLAIN_FAILED : SG_PLAIN_OFF;
    if (sp->k <= 0 || cap <= 0)
        return false;
    sp->why_plain = SG_PLAIN_NODRAFT;
    /* Prompt lookup: when the context's suffix recurs, its continuation replaces the MTP
     * head's drafts — in mode 2 only when both agree on the first draft. The head always
     * drafts first (one streamed segment, as without lookup): a cycle the lookup does not
     * take costs exactly what it costs without it. */
    nl = sg_spec_lookup_draft(&sp->lookup, cap, looked);
    if (sp->choose.q) {
        /* SAMPLED drafts: the window they are drawn under starts at the seed; stop tokens are
         * never drafted (a drawn draft cut by its value would bias the output). */
        memcpy(sp->choose.recent, recent, (size_t)n_recent * sizeof(uint32_t));
        sp->choose.n_recent = n_recent;
        sp->choose.n_no_draft = 0;
        if (eos_id >= 0) sp->choose.no_draft[sp->choose.n_no_draft++] = (uint32_t)eos_id;
        if (im_end >= 0) sp->choose.no_draft[sp->choose.n_no_draft++] = (uint32_t)im_end;
        pn_spec_sample_begin_drafts(&sp->choose);
        nd = pn_qwen35moe_tt_mtp_draft_ex(model, state, seed, sp->k < cap ? sp->k : cap, drafts,
                                          pn_spec_sample_draft, &sp->choose);
    } else {
        nd = pn_qwen35moe_tt_mtp_draft(model, state, seed, sp->k < cap ? sp->k : cap, drafts);
    }
    if (nl > 0 && nd >= 1 && sg_spec_lookup_agrees(&sp->lookup, looked[0], drafts[0])) {
        memcpy(drafts, looked, (size_t)nl * sizeof(*drafts));
        nd = nl;
        /* Draft 0 is the same token either way (drawn from q); the rest are point masses. */
        pn_spec_sample_point_drafts(&sp->choose, 1, drafts, nd);
        if (getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "spec_lookup_unverified") == 0)
            pn_qwen35moe_tt_red_accept_next_cycle();
    } else {
        nl = 0;
    }
    for (i = 0; i < nd; i++)
        if ((int)drafts[i] == eos_id || (int)drafts[i] == im_end)
            break;
    nd = i;
    if (sp->grammar)
        nd = gsc_truncate_drafts(&sp->gsc, drafts, nd);
    pn_spec_sample_cut_drafts(&sp->choose, nd);
    if (nd < 1)
        return false;
    if (sp->grammar) {
        memcpy(sp->choose.recent, recent, (size_t)n_recent * sizeof(uint32_t));
        sp->choose.n_recent = n_recent;
        sp->choose.row_index = 0;
        gsc_begin_cycle(&sp->gsc);
        /* One RNG stream: the cycle's draws continue the loop's (OpenAI `seed`). */
        sp->plain.rng_state = sp->loop->rng_state;
        sp->plain.carried_uniform_set = sp->loop->carried_uniform_set;
        sp->plain.carried_uniform = sp->loop->carried_uniform;
        ne = pn_qwen35moe_tt_spec_step_ex(model, state, seed, drafts, nd, sp->emit, gsc_choose,
                                          &sp->gsc);
        sp->loop->rng_state = sp->plain.rng_state;
        sp->loop->carried_uniform_set = sp->plain.carried_uniform_set;
        sp->loop->carried_uniform = sp->plain.carried_uniform;
    } else if (sp->sampling) {
        memcpy(sp->choose.recent, recent, (size_t)n_recent * sizeof(uint32_t));
        sp->choose.n_recent = n_recent;
        sp->choose.row_index = 0;
        ne = pn_qwen35moe_tt_spec_step_ex(model, state, seed, drafts, nd, sp->emit,
                                          pn_spec_sample_choose, &sp->choose);
    } else {
        ne = pn_qwen35moe_tt_spec_step(model, state, seed, drafts, nd, sp->emit);
    }
    if (ne < 1) {
        fprintf(stderr, "serve: speculative step failed; plain decode from here\n");
        sp->k = 0;
        sp->why_plain = SG_PLAIN_FAILED;
        return false;
    }
    if (nl > 0)
        sg_spec_lookup_report(&sp->lookup, nd, ne - 1);
    sp->n = ne;
    sp->at = 0;
    return true;
}
#endif /* USE_TT */

#ifdef SG_SPEC
#ifdef USE_CUDA
/* An AUTO tool grammar under the arming window: its mask is live inside a call only and the
 * rows outside it are the plain top-k rows, so a cycle may speculate up to the first row the
 * mask would touch (sg_spec_cuda_grammar_cut). FORCED walks and response_format stay plain. */
static bool sg_spec_cuda_grammar_lookahead(const sg_tg_ctx *tg)
{
    return sg_tg_is_tool(tg) && !tg->forced && sg_grammar_arm_window_on();
}

/* sg_mtp_cut_fn for the single-user loop: advance a clone of the live walk over the drafts'
 * text (the same decode_token text sg_tg_step feeds it) and keep those after which the next
 * row is still unmasked. */
static int sg_spec_cuda_grammar_cut(void *ctx, const uint32_t *drafts, int nd)
{
    const sg_spec *sp = (const sg_spec *)ctx;
    const sg_tg_ctx *tg = sp->tg;
    const bool json = tg->kind == SG_GRAMMAR_TOOL_JSON;
    void *walk = json ? (void *)tgj_state_clone(tg->state_json) : (void *)tg_state_clone(tg->state);
    int keep = 0;

    if (!walk)
        return 0;
    for (; keep < nd; keep++) {
        char *piece = decode_token(sp->model, drafts[keep]);
        if (piece) {
            if (json) tgj_advance((tgj_state *)walk, piece);
            else      tg_advance((tg_state *)walk, piece);
            free(piece);
        }
        if (json ? tgj_mask_active((const tgj_state *)walk) : tg_mask_active((const tg_state *)walk))
            break;
    }
    if (json) tgj_state_free((tgj_state *)walk);
    else      tg_state_free((tg_state *)walk);
    return keep;
}

/* Why a CUDA request decodes plain (spec §3.5): anything that needs the dense row — the
 * verify rows carry the plain step's top-k pairs only — or no MTP head. NULL = speculate.
 * Penalties need no fallback: a verify row is the plain step's row byte for byte, so any
 * sampler surgery over it draws the same token. */
static const char *sg_spec_cuda_plain_reason(pn_model *model, const sg_tg_ctx *tg,
                                             const sg_gen_params *params)
{
    if (params->logprobs)
        return "off:logprobs";
    if (params->logit_bias && params->n_logit_bias > 0)
        return "off:logit-bias";
    if (tg->enabled && !sg_spec_cuda_grammar_lookahead(tg))
        return sg_tg_is_tool(tg) ? "off:tool-grammar" : "off:response-format";
    if (!pn_qwen35moe_mtp_ready(model))
        return "off:no-mtp-head";
    if (getenv("PN_MTP_SERVE_SPEC") && atoi(getenv("PN_MTP_SERVE_SPEC")) == 0)
        return "off:env";
    return NULL;
}

static void sg_spec_init_cuda(sg_spec *sp, pn_model *model, pn_sampling_params *samp,
                              const sg_tg_ctx *tg, const sg_gen_params *params, int max_recent,
                              const uint32_t *prompt, int n_prompt, int max_tokens)
{
    const char *why = sg_spec_cuda_plain_reason(model, tg, params);

    sg_spec_census_begin(&sp->census, why ? why : samp->temperature > 0.0f ? "mtp-sampled"
                                                                             : "mtp-greedy");
    if (why)
        return;
    if (sg_mtp_cuda_begin(&sp->mtp, model, max_recent) != 0) {
        sp->census.mode = "off:no-memory";
        return;
    }
    sp->cuda = true;
    sp->k = PN_SPEC_DEPTH_MAX;
    sp->tg = tg;
    sp->model = model;
    if (sg_spec_lookup_init_cuda(&sp->lookup, prompt, n_prompt, max_tokens) != 0)
        fprintf(stderr, "serve: prompt-lookup drafts off (no memory for the history)\n");
    sp->mtp.lookup = sp->lookup.on ? &sp->lookup : NULL;
}

/* One CUDA MTP cycle (sg_spec_cycle's contract). A failed verify ends speculation for the
 * request: the state is then exactly where it was (nothing committed). */
static bool sg_spec_cycle_cuda(sg_spec *sp, pn_model *model, pn_state *state, uint32_t seed,
                               int cap, int eos_id, int im_end, pn_sampling_params *samp,
                               const uint32_t *recent, int n_recent)
{
    const bool grammar = sp->tg && sp->tg->enabled;
    if (grammar && (sg_tg_mask_active(sp->tg) || sp->tg->fl_pending)) {
        sp->why_plain = SG_PLAIN_NODRAFT;   /* the mask is live for this row */
        return false;
    }
    /* Every row of the cycle lies outside the mask, where the sampler skips the hook. */
    pn_sampling_params draw = *samp;
    draw.tool_grammar = NULL;
    int ne = sg_mtp_cuda_cycle(&sp->mtp, model, state, seed, -1, cap, eos_id, im_end,
                               grammar ? sg_spec_cuda_grammar_cut : NULL, sp, &draw, recent,
                               n_recent, sp->emit);
    draw.tool_grammar = samp->tool_grammar;
    *samp = draw;

    if (ne < 1) {
        fprintf(stderr, "serve: MTP speculative step failed; plain decode from here\n");
        sg_mtp_cuda_end(&sp->mtp);
        sp->cuda = false;
        sp->k = 0;
        sp->why_plain = SG_PLAIN_FAILED;
        return false;
    }
    sp->n = ne;
    sp->at = 0;
    return true;
}
#endif /* USE_CUDA */

/* Arm speculation for one request on whichever backend runs `model`. */
static void sg_spec_init(sg_spec *sp, pn_model *model, pn_sampling_params *samp,
                         const sg_tg_ctx *tg, const sg_gen_params *params, int max_recent,
                         const uint32_t *prompt, int n_prompt, int max_tokens,
                         bool spec_under_grammar)
{
    memset(sp, 0, sizeof(*sp));
#ifdef USE_CUDA
    if (sg_qwen35_cuda_forward(model)) {
        sg_spec_init_cuda(sp, model, samp, tg, params, max_recent, prompt, n_prompt,
                          max_tokens);
        return;
    }
#endif
#ifdef USE_TT
    sg_spec_init_tt(sp, model, samp, sg_spec_full_row_reason(tg, params, spec_under_grammar),
                    max_recent, prompt, n_prompt, max_tokens, tg, spec_under_grammar);
#else
    (void)prompt; (void)n_prompt; (void)max_tokens; (void)spec_under_grammar;
    sg_spec_census_begin(&sp->census, "off:no-backend");
#endif
}

/* One cycle with `seed` as the next token. `cap` bounds the drafts (token / reasoning
 * budget). @return true when the verify committed the seed (no plain forward owed). */
static bool sg_spec_cycle(sg_spec *sp, pn_model *model, pn_state *state, uint32_t seed,
                          int cap, int eos_id, int im_end, pn_sampling_params *samp,
                          const uint32_t *recent, int n_recent)
{
#ifdef USE_CUDA
    if (sp->cuda)
        return sg_spec_cycle_cuda(sp, model, state, seed, cap, eos_id, im_end, samp, recent,
                                  n_recent);
    sp->why_plain = sp->census.plain[SG_PLAIN_FAILED] ? SG_PLAIN_FAILED : SG_PLAIN_OFF;
#endif
#ifdef USE_TT
    (void)samp;
    return sg_spec_cycle_tt(sp, model, state, seed, cap, eos_id, im_end, recent, n_recent);
#else
    (void)model; (void)state; (void)seed; (void)cap; (void)eos_id; (void)im_end; (void)samp;
    (void)recent; (void)n_recent;
    return false;
#endif
}
#endif /* SG_SPEC */

/* Where this single-user turn saves its turn snapshot (think_history
 * drop_last), or -1. Only a request that carries think_history pays for it
 * (an SSM-image D2H + host copy per turn): a client that sends drop_last sends
 * the flag on the turns before it too. QWEN35_TURN_CHECKPOINT=0 turns it off
 * with the batched checkpoints. */
static int sg_turn_snapshot_cut(const sg_prefix_ctx *pc,
                                const sg_gen_params *params, int turn_start)
{
    if (params->think_history == PN_THINK_HISTORY_KEEP ||
        !bs_turn_ckpt_enabled() ||
        !pn_prefix_cache_turn_snapshot_supported(pc->cache))
        return -1;
    return turn_start;
}

/* Single-user prefill of prompt[from, to), stopping at up to two cuts to save
 * the state there: the SYS1 system-prefix snapshot (sys_cut) and the turn
 * snapshot (turn_cut); -1 or a cut outside (from, to) is skipped. Splitting is
 * bit-faithful to one pass. A failed save is benign (logged; the turn just
 * goes without it). Returns the last position's logits, or NULL. */
static tm_tensor *sg_prefill_cuts(pn_model *model, pn_state *state,
                                  const uint32_t *prompt, int from, int to,
                                  int sys_cut, int turn_cut,
                                  const sg_prefix_ctx *pc,
                                  const sg_gen_params *params)
{
    int cuts[2] = { sys_cut, turn_cut };
    if (cuts[0] > cuts[1]) {
        cuts[0] = turn_cut;
        cuts[1] = sys_cut;
    }
    tm_tensor *logits = NULL;
    int at = from;
    for (int i = 0; i <= 2; i++) {
        int end = i < 2 ? cuts[i] : to;
        if (i < 2 && (end <= at || end >= to))
            continue;
        if (logits)
            tm_free(logits);
        logits = sg_prefill_range(model, state, prompt, at, end,
                                  params->should_abort, params->abort_ud);
        if (!logits)
            return NULL;
        if (end == sys_cut &&
            pn_prefix_cache_capture_sys(pc->cache, pc->sid, prompt, end) != 0)
            fprintf(stderr, "[prefix_cache] sys-prefix capture failed sid=%s "
                    "(continuing uncached)\n", pc->sid);
        if (end == turn_cut && end < to &&
            pn_prefix_cache_capture_turn(pc->cache, pc->sid, prompt, end) != 0)
            fprintf(stderr, "[prefix_cache] turn snapshot failed sid=%s at %d "
                    "(drop_last will resume at the system prefix)\n", pc->sid,
                    end);
        at = end;
    }
    return logits;
}

int sg_run_completion(pn_model *model, pn_state *state,
                      const sg_message *messages, int n_messages,
                      const sg_gen_params *params,
                      sg_delta_fn delta_fn, void *delta_ud,
                      sg_result *result, const sg_prefix_ctx *pc)
{
    if (!model || !messages || !params || !result) return -1;
    memset(result, 0, sizeof(*result));

    /* PC2: caching is active only when a cache + session id were supplied. */
    bool use_cache = pc && pc->cache && pc->sid && pc->sid[0];
    if (!use_cache && !state) return -1;  /* legacy path needs a state */

    int n_prompt = 0;
    int sys_len = 0;
    pn_reqprof_phase_start(PN_REQ_TOKENIZE);
    sg_prompt_bounds bounds;
    uint32_t *prompt = sg_build_prompt_tokens_bounded(model, messages,
                                                      n_messages,
                                                      params->thinking,
                                                      &n_prompt, &bounds);
    sys_len = bounds.sys_len;
    int think_prefill_len = bounds.think_prefill;
    pn_reqprof_phase_end(PN_REQ_TOKENIZE);
    if (!prompt || n_prompt == 0) { free(prompt); return -1; }

    /* Pre-flight admission (MemoryAware §6.2): compute this prompt's projected
     * host + die cost and refuse/cap BEFORE the prefill loop below — the gate
     * that would have prevented the 32k full-attention OOM-reboot. Engine-level
     * pn_admit reads the live pn_mem_state; fail-open when memwatch is off. */
    int fit_tokens = n_prompt;
    pn_admit_status admit = pn_admit(model, n_prompt, &fit_tokens);
    int ctx_cap = pn_session_ctx_cap(model);
    /* The CLIENT gets the effective window, never the configured cap: kv_cap is
     * what the client re-plans its budget against mid-session, so it has to be the
     * number admission will actually enforce on the next turn — the same one
     * /v1/models advertised. `ctx_cap` below stays the configured value purely
     * so the operator's log line can show both. */
    result->kv_cap_tokens = pn_effective_ctx_window(model);
    if (admit == PN_ADMIT_DENIED) {
        free(prompt);
        return SG_ADMIT_DENIED;
    }
    /* KV Phase 1: over the per-session context budget. The old behavior silently
     * dropped the oldest tokens — which broke the prefix-cache match past the
     * system prefix, forced a full re-prefill every turn, and stalled the
     * client. Instead REJECT with a distinct, machine-readable signal so the
     * client compacts its history and retries. Never allocate past budget,
     * never OOM, never drop-oldest. */
    if (sg_context_over_budget(admit, n_prompt, fit_tokens, &result->kv_cap_tokens)) {
        result->kv_used_tokens = n_prompt;
        /* Report the EFFECTIVE cap that actually refused this prompt, not the
         * configured one (Phase 10 — see the batched twin of this branch). The
         * 422 quotes the same number (H2c). */
        fprintf(stderr, "[admit] REJECT context_over_budget: prompt %d tokens > "
                "effective cap %d (configured %d) — refusing (client must "
                "compact + retry)\n",
                n_prompt, fit_tokens, ctx_cap > 0 ? ctx_cap : fit_tokens);
        free(prompt);
        return SG_CONTEXT_OVER_BUDGET;
    }

    result->prompt_tokens = n_prompt;
    result->kv_used_tokens = n_prompt;

    /* PC2: acquire the session's cached state and the delta boundary.  On any
     * failure fall back to the passed-in `state` + full prefill (never regress
     * the working serve).  `delta_start` is the first prompt position that
     * still needs prefilling (== n_prompt means the whole prompt was cached).
     * SYS1: the acquire also gets the system-prefix boundary — a session whose
     * history diverged (reasoning turns) but whose system prefix matches its
     * cached boundary snapshot resumes at delta_start == sys_len; capture_sys
     * asks this turn's full prefill to snapshot the boundary state. */
    int delta_start = 0;
    int capture_sys = 0;
    pn_resume_kind resume_kind = PN_RESUME_COLD;
    pn_reqprof_phase_start(PN_REQ_CACHE_LOOKUP);
    if (use_cache) {
        if (pc->pc_lock) pthread_mutex_lock(pc->pc_lock);
        uint64_t refusals = pn_host_admit_thread_refusals();
        pn_prefix_acquire_result acq;
        pn_prefix_acquire_opts opts = {
            sys_len,
            sg_think_history_ceiling(model, prompt, n_prompt, think_prefill_len,
                                     params->think_history),
            params->think_history == PN_THINK_HISTORY_DROP_LAST,
        };
        pn_state *cs = pn_prefix_cache_acquire_bounded(
            pc->cache, pc->sid, prompt, n_prompt, &opts, &acq);
        delta_start = acq.delta_start;
        capture_sys = acq.capture_sys;
        resume_kind = acq.kind;
        if (cs) {
            state = cs;   /* use the session's state; ignore the passed-in one */
        } else if (pn_host_admit_thread_refusals() != refusals) {
            /* The new session's state was refused by host-RAM admission: answer
             * with the reason (503) rather than borrow the shared state, which
             * would silently serve a session the host said it cannot hold. */
            if (pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
            pn_reqprof_phase_end(PN_REQ_CACHE_LOOKUP);
            snprintf(result->admit_reason, sizeof(result->admit_reason), "%s",
                     pn_host_admit_last_refusal());
            free(prompt);
            return SG_ADMIT_DENIED;
        } else {
            if (pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
            use_cache = false;
            delta_start = 0;
            capture_sys = 0;
        }
    }
    if (!use_cache) {
        pn_state_reset(state);
        delta_start = 0;
        resume_kind = PN_RESUME_COLD;
    }

    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = params->temperature;
    sp.top_p = params->top_p;
    sp.top_k = params->top_k;
    sg_apply_openai_sampling(&sp, params);
#ifdef USE_TT
    /* Greedy requests take the card's exact top-N row unless a grammar, logprobs or logit
     * surgery needs the dense one — decided BEFORE the prompt (its last row is token 1). */
    unsigned long tail0[4], refills0 = pn_sampler_topn_refills();
    pn_sampler_tail_counts(&tail0[0], &tail0[1], &tail0[2]);
    tail0[3] = pn_sampler_tail_count_calls();
    if (pn_qwen35moe_tt_select_rows(model, &sp, sg_may_need_full_logits(params)) > 0u) {
        sp.dense_row = pn_qwen35moe_tt_dense_row;
        sp.tail_counts = pn_qwen35moe_tt_tail_counts;
        sp.dense_chunks = pn_qwen35moe_tt_dense_chunks;
        sp.dense_ctx = model;
    }
#endif

    int max_recent = sp.repeat_last_n;
    uint32_t *recent = calloc((size_t)(max_recent > 0 ? max_recent : 1),
                              sizeof(uint32_t));
    if (!recent) {   /* OOM: bail before sg_seed_recent memcpy's into NULL */
        if (use_cache && pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
        free(prompt);
        return -1;
    }
    int n_recent = 0;

    /* Track this turn's generated tokens so PC2 can commit the exact committed
     * history (prompt + generated) back to the session. +8 headroom: the
     * reasoning-budget force-close injects up to sizeof(close_toks)=8 "</think>"
     * tokens beyond the sampled ones, so `generated` can reach max_tokens+n_close
     * while the main loop's unguarded gen_toks[generated] write still runs. */
    int max_tokens = params->max_tokens > 0 ? params->max_tokens : 512;
    uint32_t *gen_toks = use_cache
        ? malloc((size_t)(max_tokens + 8) * sizeof(uint32_t)) : NULL;
    if (use_cache && !gen_toks) {
        if (pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
        free(prompt); free(recent);
        return -1;
    }

    /* Prefill (L1): run the DELTA (tokens [delta_start, n_prompt)) through the
     * batched forward in chunks, keeping the last logits to sample the first
     * generated token from.  With caching off delta_start==0 (full prefill).
     *
     * A real chat turn ALWAYS appends new tokens (at minimum the assistant
     * opener), so delta_start < n_prompt in practice.  The degenerate full-hit
     * case (new prompt == prior committed history exactly) would leave nothing
     * to prefill and no way to recover the next-token logits without advancing
     * pos past n_prompt — so fall back to a clean full reset+prefill for it
     * (correctness over the microscopic caching win). */
    if (use_cache && delta_start >= n_prompt) {
        pn_state_reset(state);
        delta_start = 0;
        resume_kind = PN_RESUME_COLD;
    }
    pn_reqprof_phase_end(PN_REQ_CACHE_LOOKUP);
    pn_reqprof_note_cache(use_cache, n_prompt, delta_start, sys_len);
    /* Report the reuse on the wire, not just in the serve log: read AFTER the
     * degenerate-full-hit fallback above, so what the client is told is the
     * boundary this turn actually prefills from. */
    result->cached_prompt_tokens = pn_prefix_cached_tokens(delta_start, n_prompt);
    result->resume_kind = resume_kind;
    result->prefill_tokens = n_prompt - delta_start;
    tm_tensor *logits = NULL;
    int prefill_tokens = n_prompt - delta_start;   /* Phase 7 rate counter */
    pn_reqprof_phase_start(PN_REQ_PREFILL);
    double prefill_t0 = sg_prefill_now_ms();
    bool tt_one_pass = false;
#ifdef USE_TT
    /* SYS1's split would start the conversation at pos sys_len, which the TT batched
     * prefill declines (per token from there): one pass, no system snapshot, until the
     * TT prefill can continue from a carried state (plan: multi-turn state caching). */
    tt_one_pass = model->forward == pn_qwen35moe_forward_tt;
    /* Checkpoint no higher than the history the next turn resends: the generation prompt
     * after the assistant header ("<think>\n") is not in it, and a checkpoint covering it
     * cannot be resumed (the whole conversation re-prefills). A think_history request keeps
     * it at or below turn_start, so a drop_last request that rewrites this reply resumes it. */
    if (tt_one_pass && use_cache)
        pn_qwen35moe_tt_checkpoint_ceiling(
            model, state,
            pn_think_history_tt_ceiling(params->think_history, bounds.turn_start,
                                        n_prompt - think_prefill_len));
#endif
    if (use_cache && !tt_one_pass) {
        int sys_cut = (capture_sys && delta_start == 0 && sys_len < n_prompt)
                          ? sys_len : -1;
        int turn_cut = sg_turn_snapshot_cut(pc, params, bounds.turn_start);
        logits = sg_prefill_cuts(model, state, prompt, delta_start, n_prompt,
                                 sys_cut, turn_cut, pc, params);
    } else {
        logits = sg_prefill_range(model, state, prompt, delta_start, n_prompt,
                                  params->should_abort, params->abort_ud);
    }
    if (logits)
        sg_prefill_rate_record(prefill_tokens,
                               sg_prefill_now_ms() - prefill_t0);
    pn_reqprof_phase_end(PN_REQ_PREFILL);
    sg_prefill_logits_hash(logits, prefill_tokens, state->pos);
    pn_reqprof_phase_start(PN_REQ_DECODE);
    if (!logits) {
        /* Distinguish a client-disconnect abort from a real prefill error: the
         * check is idempotent (a closed socket stays closed). Either way, reset the
         * session so the partial, uncommitted prefill can't corrupt next turn's
         * append-resume (a card prefill that aborted part way left its state half
         * written). The session-watch owns slot release when the client's watch
         * dies past grace; this path only cleans up the partial prefill. Then
         * return a clean SG_ABORTED, or -1. */
        int aborted = params->should_abort && params->should_abort(params->abort_ud);
        if (use_cache) {
            pn_prefix_cache_reset_session(pc->cache, pc->sid);
            if (pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
        }
        free(prompt); free(recent); free(gen_toks);
        return aborted ? SG_ABORTED : -1;
    }
    n_recent = sg_seed_recent(recent, max_recent, prompt, n_prompt);

    emitter em;
    em_init(&em, delta_fn, delta_ud, params->stop, params->n_stop,
            params->has_tools, params->dialect);
    em_set_tool_tap(&em, params->tool_tap, params->tool_tap_ud);
    if (sg_think_close_is_token(model))
        em_set_think_close_token(&em);
    /* The emitter starts inside a think block only when the prompt left one OPEN:
     * a think-style template with thinking on.  No-Think prefills the block CLOSED,
     * and a non-think template (plain ChatML, Mistral, Phi-3) never opens one, so
     * the model never emits </think> — start in CONTENT mode, else all output is
     * misrouted to reasoning_content and the answer/tool-call channel stays empty. */
    if (sg_thinking_suppressed(params->thinking) ||
        !sg_chat_style_is_think(model->chat_style))
        em.in_think = 0;

    int eos_id = ft_eos_id(model->tokenizer);
    int im_end = ft_token_to_id(model->tokenizer, "<|im_end|>");
    int n_vocab = model->config.n_vocab;

    /* Phase 5: grammar-constrained tool decoding (sg_tool_grammar_arms).  Inert
     * otherwise → sp.tool_grammar stays NULL and the sampler is byte-identical. */
    sg_tg_ctx tg; sg_tg_setup(&tg, model, params, n_vocab);
    sg_tg_attach(&tg, &sp);
    tg.fl_always = sg_full_logits_always(params);
    sg_tg_mark_mask_density(&tg);
    /* Window OFF: arm the whole turn, exactly here, exactly as before.
     * Window ON: sg_arm_full_logits owns the flag from here on and is called
     * immediately before every forward this loop runs. */
    if (!sg_grammar_arm_window_on())
        sg_require_full_logits(&tg, params, model, state, 1);
    else
        sg_arm_full_logits(&tg, params, model, state);

    /* OpenAI `logprobs` (inert unless the request asked).  Sized like gen_toks:
     * max_tokens plus the reasoning force-close headroom. */
    sg_logprob_acc lpacc;
    sg_logprobs_init(&lpacc, result, params, max_tokens + 8);
    sg_logprobs_attach(&lpacc, &sp);

    /* AUTO-path reasoning budget (independent of the tool-forcing grammar): count
     * tokens generated while inside <think>; once over budget without a self-emitted
     * "</think>", force-close reasoning so the model answers.  Disabled when
     * budget <= 0 (byte-identical to today) or the model is not think-style. */
    pn_reasoning_plan reason_plan =
        pn_reasoning_plan_for_turn((params->reasoning_budget > 0 &&
                                    sg_chat_style_is_think(model->chat_style))
                                       ? params->reasoning_budget : 0,
                                   max_tokens, params->reasoning_reserve,
                                   params->model_name);
    int reason_budget = reason_plan.hard; /* the cap run: budget less the answer
                                           * reserve (reasoning_budget.h) */
    int reason_tokens = 0;      /* tokens routed to the reasoning channel so far */
    bool reason_forced = false; /* force-close already applied this turn */
    bool reason_over_budget = false; /* latched: budget tripped this iteration
                                      * (set off the pre-feed think state so a
                                      * same-token </think> close can't starve it) */

    /* Repetition-loop guard: fed EVERY generated token id of this turn, in
     * order, reasoning and answer alike — a model looping inside <think> is just
     * as stuck as one looping in the answer, and the reasoning budget only
     * bounds the think block's LENGTH, not whether it is saying anything new. */
    rld_detector reploop;
    sg_reploop_arm(&reploop);

    const char *finish = "length";
    int generated = 0;
#ifdef SG_SPEC
    sg_spec spec;
    bool spec_under_grammar = false;
#ifdef USE_TT
    spec_under_grammar = tg.enabled &&
                         (sg_tg_is_tool(&tg) ? sg_tool_grammar_speculates(params)
                                             : sg_response_format_speculates(params));
    uint64_t spec_c0 = 0, spec_d0 = 0, spec_a0 = 0;
    (void)pn_qwen35moe_tt_spec_counts(model, &spec_c0, &spec_d0, &spec_a0);
#endif
    sg_spec_init(&spec, model, &sp, &tg, params, max_recent, prompt, n_prompt, max_tokens,
                 spec_under_grammar);
    double step_t0 = 0.0;   /* census: the decode-loop step clock */
    int step_masked = 0;
#endif
    for (int i = 0; i < max_tokens; i++) {
#ifdef SG_SPEC
        if (!logits && spec.at >= spec.n) break;
#else
        if (!logits) break;
#endif
        /* Stop decoding if the client disconnected (checked every 8 tokens to
         * bound the syscall). Tokens generated so far are valid — the commit
         * below keeps them, and finish="stop" ends the turn cleanly. */
        if ((i & 7) == 0 && params->should_abort &&
            params->should_abort(params->abort_ud)) {
            finish = "stop";
            break;
        }
        sg_tg_refresh(&tg);   /* mask only inside a tool call; FREE unconstrained */
        uint32_t next;
#ifdef SG_SPEC
        {
            /* The previous step's wall time, booked to the region it ran in. */
            const double step_now = sg_prefill_now_ms();

            if (i > 0)
                sg_spec_census_step(&spec.census, step_masked, step_now - step_t0);
            step_t0 = step_now;
            step_masked = tg.enabled && tg.hook.active;
        }
        if (spec.at < spec.n) {
            next = spec.emit[spec.at++];
        } else
#endif
        {
            pn_sampler_row_hint(logits); /* a backend's survivor list, this draw only */
            next = pn_sample_next_token((float *)logits->data, n_vocab,
                                        &sp, recent, n_recent);
            tm_free(logits);
            logits = NULL;
        }

        if ((int)next == eos_id || (int)next == im_end) {
            finish = sg_tg_close_required_call(&tg, params, &em) ? "tool_calls" : "stop";
            break;
        }

        char *piece = decode_token(model, next);
        /* Resolve this token's logprobs BEFORE feeding the emitter, so the entry
         * can ride the content delta that carries its bytes.  A token generated
         * inside <think> is REASONING, not answer content, so its record is
         * consumed but NOT reported — that is what keeps this path's reported set
         * identical to the streaming path's, which only ever sees content deltas.
         * Every consumed token takes or skips exactly one record, so the sequence
         * stays paired with the token stream. */
        const sg_token_logprob *lp = NULL;
        if (em.in_think) sg_logprobs_skip(&lpacc);
        else lp = sg_logprobs_take(&lpacc, result, model, piece);
        if (lp) em_note_token_logprob(&em, lp);
        if (piece) {
            sg_tg_step(&tg, piece);   /* advance the grammar by the chosen text */
            sg_tg_trace(&tg, i, next, piece);
            int was_think = em.in_think;
            em_feed(&em, piece);
            /* A token generated while inside <think> counts against the reasoning
             * budget (whether or not this particular token closed the block).
             * Latch the budget trip HERE, off the pre-feed think state (was_think):
             * the model can reach the budget on the very token that also closes
             * <think>, at which point em.in_think is already 0 by the force-close
             * check below and the (em.in_think && …) guard would starve the trip —
             * so the reasoning cap silently never fired and the model ran on to
             * max_tokens (the observed saber/ornith runaway).  Arm the trip while
             * we still know we WERE reasoning; the force-close below then closes
             * cleanly whether or not this token already ended the block. */
            if (was_think) {
                reason_tokens++;
                /* At the cap, or on the first line end inside the window
                 * just before it, so the thought ends on a whole line. */
                if (reason_budget && !reason_forced &&
                    pn_reasoning_should_close(&reason_plan, reason_tokens,
                                              pn_reasoning_piece_ends_line(piece)) &&
                    !sg_tg_in_call(&tg)) {
                    reason_over_budget = true;
                    /* Arming-window latch.  The force-close block below runs
                     * AFTER the forward that fetches the row the next step
                     * samples, and it ends in sg_tg_reasoning_ended() — the one
                     * grammar mutation that can land too late to be read.
                     * Latch here, while we are still on the near side of that
                     * forward, so the whole close window is armed. */
                    tg.fl_pending = true;
                }
            }
            /* Forced mode: when the emitter's <think> block closes ("</think>"
             * consumed), reasoning is over → the grammar now REQUIRES the opener
             * on the content channel, compelling the call. */
            if (was_think && !em.in_think) sg_tg_reasoning_ended(&tg);
            /* Reverse coupling: when the GRAMMAR has decided reasoning is over
             * (self-detected "</think>" or the budget tripped) but the emitter is
             * still routing to <think>, force it to CONTENT — otherwise the
             * grammar-compelled call lands in reasoning_content and is never parsed
             * as a tool_call (the empty-prompt degeneration: budget forces a valid
             * <function=…> call, but it was mis-routed to the reasoning channel). */
            if (em.in_think && sg_tg_forced_ended(&tg)) {
                em.in_think = 0;
                sg_log_reasoning_channel_flip(&tg, generated);
            }
            free(piece);
        }
        if (gen_toks) gen_toks[generated] = next;
        generated++;
        (void)rld_feed(&reploop, next);
#ifdef SG_SPEC
        sg_spec_lookup_push(&spec.lookup, next);
#endif

        if (max_recent > 0) {
            if (n_recent < max_recent) {
                recent[n_recent++] = next;
            } else {
                memmove(recent, recent + 1, (size_t)(max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = next;
            }
        }
        /* Forward `next` into the state BEFORE the stop/tool_calls breaks below.
         * `next` is real output (already streamed + counted in `generated`), so it
         * must be stepped into the KV/SSM before we break — otherwise state->pos
         * falls one behind the committed history and next turn pays a full
         * re-prefill on arches without lcp-rewind (qwen35moe). `logits` then holds
         * the post-`next` distribution for the next iteration (or the force-close).
         *
         * The grammar has already advanced over `next` (sg_tg_step above), so
         * arming here reads exactly the walk state the NEXT step's sampler will
         * refresh from — the whole point of the window. */
        sg_arm_full_logits(&tg, params, model, state);
#ifdef SG_SPEC
        if (spec.at < spec.n) {
            /* `next` was an accepted draft: the verify already committed it. */
            sg_spec_census_verified(&spec.census);
        } else {
            /* Drafts this cycle may emit: they and the next token must fit the token
             * budget, and must not reach the reasoning budget's trip (the trip token
             * takes a plain forward: the force-close needs its logits). */
            int cap = max_tokens - i - 2;
            if (reason_budget && !reason_forced && em.in_think &&
                pn_reasoning_draft_room(&reason_plan, reason_tokens) < cap)
                cap = pn_reasoning_draft_room(&reason_plan, reason_tokens);
            if (reason_over_budget) {
                sg_spec_census_plain(&spec.census, SG_PLAIN_TRIP);
                logits = sg_forward_step(model, state, next);
            } else if (!sg_spec_cycle(&spec, model, state, next, cap, eos_id, im_end, &sp,
                                      recent, n_recent)) {
                sg_spec_census_plain(&spec.census, spec.why_plain);
                logits = sg_forward_step(model, state, next);
            } else {
                sg_spec_census_verified(&spec.census);
            }
        }
#else
        logits = sg_forward_step(model, state, next);
#endif

        if (em.stopped) { finish = "stop"; break; }

        /* Repetition collapse: the model has re-emitted the same block of tokens
         * back to back, byte-identical, with nothing novel in between.  Checked
         * HERE, after sg_forward_step, so `next` is committed to the KV/SSM like
         * every other generated token and state->pos stays level with the
         * committed history — the same rule the stop/tool_calls breaks follow.
         * Read off the LATCH, not off this token's rld_feed return, so a trip
         * that lands on a force-close injection below is still acted on (one
         * token later) instead of being swallowed by the latch. */
        if (reploop.tripped) { finish = SG_FINISH_REPETITION_LOOP; break; }

        /* Grammar completion-stop: once a whole tool call has been emitted
         * ("</function>" / "</tool_call>" consumed) the constrained request is
         * DONE — stop immediately instead of running to max_tokens.  Without
         * this a forced request generated 1024 tokens (~80 s) past the call and
         * looked like a hang.  Only fires for a live constrained grammar; auto
         * requests that never engaged a call are unaffected. */
        if (sg_tg_live(&tg) && sg_tg_done(&tg)) {
            /* A TOOL grammar closing means a call was emitted; the
             * response-format grammar closing just means the JSON object is
             * finished, which is an ordinary "stop". */
            finish = sg_tg_is_tool(&tg) ? "tool_calls" : "stop";
            break;
        }

        /* AUTO reasoning budget: the model has spent its whole reasoning grace
         * inside <think> without concluding.  Force-close it so it concludes and
         * answers.  `reason_over_budget` was latched above off the PRE-feed think
         * state, so the trip survives even when this same token also closed the
         * block (the starvation bug: gating on the post-feed em.in_think here let
         * a budget reached on the closing token slip through, and the model ran on
         * to max_tokens).  Force-decode the "</think>" tokens through the state
         * (KV/SSM advances over the close) AND the emitter (its own THINK_CLOSE
         * match flips it to the content channel), accounting each like a generated
         * token; `logits` ends holding the post-close next-token distribution, so
         * the next loop iteration samples the answer's first token.  Fires at most
         * once per turn.  If the emitter already left <think> on the trip token
         * (self-close), the injection loop below is a no-op and we only stamp the
         * grammar reasoning-ended below. */
        if (reason_over_budget && !reason_forced && logits) {
            reason_over_budget = false;
            reason_forced = true;
            /* Only force-decode the close when the block is still OPEN.  When the
             * trip token also self-closed <think> (em.in_think already 0), the
             * reasoning region ended on its own — injecting another "</think>"
             * would double-close (a stray marker in the KV state + content), so
             * skip straight to stamping the grammar reasoning-ended below. */
            if (em.in_think) {
                uint32_t close_toks[8];
                int n_close = sg_encode_think_close(model, close_toks,
                                                    (int)(sizeof(close_toks) /
                                                          sizeof(close_toks[0])));
                /* Bound the injection by the gen_toks buffer (max_tokens + its +8
                 * headroom), NOT by max_tokens: a multi-token fallback close firing
                 * near the cap must complete, not truncate mid-sequence (which would
                 * leave a half-closed think block in the KV state). */
                int inj_cap = max_tokens + (int)(sizeof(close_toks) /
                                                 sizeof(close_toks[0]));
                for (int c = 0; c < n_close && generated < inj_cap && !em.stopped;
                     c++) {
                    char *cp = decode_token(model, close_toks[c]);
                    if (cp) {
                        sg_tg_step(&tg, cp);
                        em_feed(&em, cp);
                        free(cp);
                    }
                    if (gen_toks) gen_toks[generated] = close_toks[c];
                    generated++;
#ifdef SG_SPEC
                    sg_spec_lookup_push(&spec.lookup, close_toks[c]);
                    sg_spec_census_plain(&spec.census, SG_PLAIN_TRIP);
#endif
                    /* Keep the detector's view of the stream faithful — these
                     * ids really are in the committed history.  A trip here is
                     * not acted on mid-injection (that would leave a half-closed
                     * think block in the KV state); the loop's latch check picks
                     * it up on the next iteration. */
                    (void)rld_feed(&reploop, close_toks[c]);
                    sg_recent_push(recent, max_recent, &n_recent, close_toks[c]);
                    tm_free(logits);
                    /* fl_pending is latched across this whole block, so these
                     * forwards are armed whatever the walk currently says. */
                    sg_arm_full_logits(&tg, params, model, state);
                    logits = sg_forward_step(model, state, close_toks[c]);
                    if (!logits) break;
                }
                /* If the emitter never saw a clean "</think>" (n_close==0, or a
                 * tokenizer whose close piece doesn't decode to the literal), feed
                 * the LITERAL through the emitter's own THINK_CLOSE matcher so any
                 * held carry resolves into the right channel — poke the flag only as
                 * a last resort. Without this, a later model-emitted "</think>"
                 * leaks verbatim into content. */
                if (em.in_think) {
                    if (n_close == 0)
                        fprintf(stderr, "[reason_budget] no </think> token for this "
                                "model's tokenizer — forcing a literal emitter "
                                "close\n");
                    em_feed(&em, "</think>");
                    if (em.in_think) em.in_think = 0;
                }
            }
            /* Symmetry with the self-close path above (line ~933): tell the grammar
             * reasoning ended so a live forced walk compels the opener on content.
             *
             * INJECTED variant: this close was force-decoded by the budget, not
             * chosen by the model, so the grammar gets a bounded landing window
             * rather than compelling the opener from the very next char around an
             * unfinished thought.  The self-close path above deliberately keeps
             * the plain sg_tg_reasoning_ended — a model that concluded on its own
             * needs no grace, and its behavior must stay byte-identical. */
            sg_tg_reasoning_ended_injected(&tg);
            /* The late mutation has landed.  Every forward taken while the latch
             * was up was armed, so the row the next step samples is dense, and
             * the walk is now authoritative again — drop the latch and let the
             * window narrow. */
            tg.fl_pending = false;
        }
    }
    if (logits) tm_free(logits);
    free(recent);

    if (reploop.tripped)
        sg_reploop_log(&reploop, params, "single-user", generated);
    rld_free(&reploop);

    /* OBSERVABILITY — a reasoning force-close is an ABNORMAL event, so it logs
     * unconditionally (not behind --verbose; the only prior trace of any of this
     * was the [tg] step dump gated on c->verbose).  It is the one condition under
     * which the grammar compels the tool-call opener while the model is still
     * mid-thought, which is how a half-written call ends up inside a parameter
     * VALUE (tool_calling.c FRAME_MARKERS).  The BATCHED twin lives in
     * bs_slot_release; the fleet runs batched, so both must be present or the
     * production rate is invisible. */
    if (reason_forced)
        fprintf(stderr, "[reason_budget] FORCED close model=%s path=single-user "
                "reason_toks=%d/%d generated=%d tool_call=%s\n",
                (params->model_name && params->model_name[0])
                    ? params->model_name : "?",
                reason_tokens, reason_budget, generated,
                !sg_tg_live(&tg) ? "n/a" : (sg_tg_done(&tg) ? "yes" : "no"));

    /* PC2: commit the session's committed history (prompt + generated) so next turn
     * deltas from it. Every generated token is forwarded into `state` before any
     * break — sg_forward_step(next) now precedes the stop/tool_calls checks, and
     * eos/im_end is excluded from BOTH the count and the forward — so
     * state->pos == n_prompt + generated exactly on every exit, and next turn's
     * append-resume hits with no re-prefill (matters on arches without lcp-rewind,
     * e.g. qwen35moe). */
    if (use_cache) {
        /* Speculation: a mid-cycle stop leaves accepted drafts COMMITTED but not
         * emitted (emit[at..n-2]); the history names what the state really holds,
         * so next turn's LCP sees the divergence instead of trusting a stale state. */
        int extra = 0;
#ifdef SG_SPEC
        extra = spec.n - 1 - spec.at;
        if (extra < 0) extra = 0;
        /* RED `spec_stop_noextra` (PN_TT_RED): forget the committed-but-unemitted drafts of a
         * mid-cycle stop — the resumed next turn then runs on a state the history misnames
         * and MUST differ from the cold one. */
        if (getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "spec_stop_noextra") == 0)
            extra = 0;
#endif
        int hist_n = n_prompt + generated + extra;
        uint32_t *hist = malloc((size_t)hist_n * sizeof(uint32_t));
        if (hist) {
            memcpy(hist, prompt, (size_t)n_prompt * sizeof(uint32_t));
            if (generated > 0)
                memcpy(hist + n_prompt, gen_toks,
                       (size_t)generated * sizeof(uint32_t));
#ifdef SG_SPEC
            if (extra > 0)
                memcpy(hist + n_prompt + generated, spec.emit + spec.at,
                       (size_t)extra * sizeof(uint32_t));
#endif
            pn_prefix_cache_commit(pc->cache, pc->sid, hist, hist_n);
            free(hist);
        }
        if (pc->pc_lock) pthread_mutex_unlock(pc->pc_lock);
    }
#ifdef SG_SPEC
    sg_spec_report_grammar(&spec);
    sg_spec_census_report(&spec.census, generated);
#ifdef USE_CUDA
    if (spec.cuda)
        sg_mtp_cuda_report(&spec.mtp, generated);
#endif
#endif
#ifdef USE_TT
    pn_spec_rule_report(&spec.choose, "serve");
    {
        /* One line a speculating request: its own cycles and acceptance (both rules). */
        uint64_t c1, d1, a1;

        if (pn_qwen35moe_tt_spec_counts(model, &c1, &d1, &a1) == 0 && c1 > spec_c0)
            fprintf(stderr, "[spec] request: %llu cycles, %llu drafts, %llu accepted (%.3f), "
                            "%.2f tokens a cycle\n", (unsigned long long)(c1 - spec_c0),
                    (unsigned long long)(d1 - spec_d0), (unsigned long long)(a1 - spec_a0),
                    d1 > spec_d0 ? (double)(a1 - spec_a0) / (double)(d1 - spec_d0) : 0.0,
                    (double)(c1 - spec_c0 + a1 - spec_a0) / (double)(c1 - spec_c0));
    }
    sg_report_sampled_rows(tail0, refills0);
#endif
    if (pn_dump_reqlog_enabled()) {
        /* `PN_TT_DUMP_REQLOG`: this request's line, after its last dump record was flushed. */
        pn_dump_reqlog_request rl = {
            .prompt_tokens = n_prompt, .completion_tokens = generated, .max_tokens = max_tokens,
            .top_k = sp.top_k, .temperature = sp.temperature, .top_p = sp.top_p,
            .seed_set = params->seed_set, .seed = params->seed, .finish = finish,
        };
#ifdef USE_TT
        uint64_t rc1, rd1, ra1;

        if (pn_qwen35moe_tt_spec_counts(model, &rc1, &rd1, &ra1) == 0) {
            rl.has_spec = true;
            rl.spec_cycles = rc1 - spec_c0;
            rl.spec_drafts = rd1 - spec_d0;
            rl.spec_accepted = ra1 - spec_a0;
        }
#endif
        (void)sg_dump_reqlog_request(&rl, prompt, n_prompt);
    }
#ifdef SG_SPEC
    sg_spec_free(&spec);
#endif
    free(gen_toks);
    free(prompt);
    sg_tg_report_response_format(&tg, result);
    sg_tg_report_tool_call(&tg, result);
    sg_tg_report_mask_density(&tg);
    sg_require_full_logits(&tg, params, model, state, 0);
    sg_tg_free(&tg);
    /* Frees only the raw record buffer — the RESOLVED entries live on the result
     * and are freed by sg_result_free. */
    sg_logprobs_free(&lpacc);

    em_flush(&em);

    pn_reqprof_phase_end(PN_REQ_DECODE);
    pn_reqprof_note_generated(generated);
    result->completion_tokens = generated;
    result->finish_reason = finish;
    result->content = em.content.data ? em.content.data : strdup("");
    result->reasoning = em.reasoning.data;  /* may be NULL */
    result->tool_raw = em.tool.data;        /* may be NULL (no tool call) */
    /* Transfer ownership out of the emitter, then free its remaining buffers. */
    em.content.data = NULL;
    em.reasoning.data = NULL;
    em.tool.data = NULL;
    em_free(&em);
    return 0;
}

void sg_result_free(sg_result *result)
{
    if (!result) return;
    free(result->content);
    free(result->reasoning);
    free(result->tool_raw);
    for (int i = 0; i < result->n_logprobs; i++) {
        free(result->logprobs[i].token);
        for (int j = 0; j < result->logprobs[i].n_top; j++)
            free(result->logprobs[i].top_token[j]);
    }
    free(result->logprobs);
    result->content = NULL;
    result->reasoning = NULL;
    result->tool_raw = NULL;
    result->logprobs = NULL;
    result->n_logprobs = 0;
}

/* ============================================================
 * C3 — batched completion (sg_run_completion_batched)
 *
 * Prefill is single-user but BATCHED (L1: whole chunks through
 * pn_qwen35moe_forward_batch_cuda) and CHUNKED (L2: the GPU turn is yielded
 * to the decode driver between chunks, so a join stalls concurrent decoders
 * for one chunk, not one prompt).  Decode is batched: the scheduler's
 * driver calls decode_slots with all active slots in one GPU pass and (L3,
 * default) samples every slot in-driver, handing this thread finished
 * tokens through a per-slot ring — detokenize/emitter/SSE run entirely off
 * the step's critical path.  QWEN35_BS_DRIVER_SAMPLE=0 restores the legacy
 * lockstep logits/token exchange.
 * ============================================================ */

/* PN_HOST_TIMING: the delta callback timed into the owner's accumulator (O_SSE). */
typedef struct {
    sg_delta_fn      fn;
    void            *ud;
    pn_host_timing  *acc;
} sg_ht_delta;

static void sg_ht_delta_shim(const char *text, int is_reasoning,
                             const sg_token_logprob *const *logprobs, int n_logprobs,
                             void *user_data)
{
    sg_ht_delta *d = user_data;
    const uint64_t t0 = pn_ht_now();

    if (d->fn)
        d->fn(text, is_reasoning, logprobs, n_logprobs, d->ud);
    pn_ht_add(d->acc, PN_HT_O_SSE, pn_ht_now() - t0);
}

int sg_run_completion_batched(pn_model *model, pn_batch_sched *sched,
                              int slot,
                              const sg_message *messages, int n_messages,
                              const sg_gen_params *params,
                              sg_delta_fn delta_fn, void *delta_ud,
                              sg_result *result, const char *sid)
{
    if (!model || !sched || slot < 0 || !messages || !params || !result)
        return -1;
    memset(result, 0, sizeof(*result));

    /* The [req] line (request_timing_line.h): TTFT runs from the request's
     * arrival (slot wait included); a second choice of the same request, or a
     * caller that never marked arrival, starts here. */
    double t_req0 = sg_request_arrival_take();
    if (t_req0 <= 0.0)
        t_req0 = sg_request_timing_now_ms();
    sg_request_timing rt = { .slot = slot };
    bs_request_stats rstats = { 0 };
    double t_first = 0.0, t_dec_end = 0.0;
    /* PN_HOST_TIMING: owner-side stages and the decode window's ns stamps. */
    pn_host_timing ht_own;
    memset(&ht_own, 0, sizeof(ht_own));
    uint64_t ht_first = 0, ht_dec_end = 0, ht_cpu0 = 0, ht_rd0 = 0, ht_cpu1 = 0, ht_rd1 = 0;
    sg_ht_delta ht_delta = { delta_fn, delta_ud, &ht_own };

    /* PC-batched: caching is active only with a session id.  NULL/empty sid
     * → full-prefill on every turn, byte-identical to the pre-PC batched
     * path (safe-by-default). */
    int use_cache = (sid && sid[0]);

    /* Slot was pre-acquired by the caller (bs_slot_acquire).  The slot's
     * pn_state is reset inside bs_prefill_begin below (under the GPU-owner
     * lock — it may still be in an in-flight driver step as a dummy lane
     * right now); until then we only take the pointer. */
    pn_state *state = sched->slots[slot].state;

    /* ── 2. Build prompt ── (with the SYS1 system-prefix boundary: sys_len
     * leading tokens cover the first message's system turn, stable across a
     * session's turns — the batched per-slot snapshot point.) */
    int n_prompt = 0;
    sg_prompt_bounds bounds;
    pn_reqprof_phase_start(PN_REQ_TOKENIZE);
    uint32_t *prompt = sg_build_prompt_tokens_bounded(model, messages,
                                                      n_messages,
                                                      params->thinking,
                                                      &n_prompt, &bounds);
    pn_reqprof_phase_end(PN_REQ_TOKENIZE);
    int sys_len = bounds.sys_len;
    int think_prefill_len = bounds.think_prefill;
    if (!prompt || n_prompt == 0) {
        free(prompt);
        bs_slot_release(sched, slot);
        return -1;
    }
    /* The acquire could not see the prompt; now that it is known, run on the
     * slot whose cache it continues (one session id, several conversations). */
    if (use_cache) {
        slot = bs_slot_reroute(sched, slot, sid, prompt, n_prompt, sys_len);
        state = sched->slots[slot].state;
    }

    /* ── 3. Admission (same logic as sg_run_completion) ── */
    int fit_tokens = n_prompt;
    pn_admit_status admit = pn_admit(model, n_prompt, &fit_tokens);
    int ctx_cap = pn_session_ctx_cap(model);
    /* Effective, not configured — see the single-user twin in sg_run_completion.
     * The BATCHED path is the one the live fleet runs, so a gap here is the gap
     * that ships. */
    result->kv_cap_tokens = pn_effective_ctx_window(model);
    if (admit == PN_ADMIT_DENIED) {
        free(prompt);
        bs_slot_release(sched, slot);
        /* D3: propagate the pressure signal so queued waiters back off.
         * Read the live level and push it into the scheduler. */
        if (pn_memwatch_running()) {
            pn_mem_state ms;
            pn_memwatch_read(&ms);
            bs_notify_pressure(sched, (int)ms.level);
        }
        return SG_ADMIT_DENIED;
    }
    /* KV Phase 1: over the per-session context budget — REJECT with a distinct
     * signal instead of the old silent drop-oldest that shattered the prefix
     * cache and stalled the client. The client compacts + retries. */
    if (sg_context_over_budget(admit, n_prompt, fit_tokens, &result->kv_cap_tokens)) {
        result->kv_used_tokens = n_prompt;
        /* Report the EFFECTIVE cap that actually refused this prompt, not the
         * configured one; the 422 quotes the same number (H2c). Printing ctx_cap alone produced self-contradictory
         * lines ("prompt 70000 > session cap 131072") whenever the memory
         * ladder had clamped the window underneath it — Phase 10. */
        fprintf(stderr, "[batch_sched] slot %d: REJECT context_over_budget: "
                "prompt %d tokens > effective cap %d (configured %d) — "
                "refusing (compact + retry)\n",
                slot, n_prompt, fit_tokens, ctx_cap > 0 ? ctx_cap : fit_tokens);
        free(prompt);
        bs_slot_release(sched, slot);
        return SG_CONTEXT_OVER_BUDGET;
    }
    result->prompt_tokens = n_prompt;
    result->kv_used_tokens = n_prompt;

    /* ── 4. Sampling params ── */
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = params->temperature;
    sp.top_p       = params->top_p;
    sp.top_k       = params->top_k;
    sg_apply_openai_sampling(&sp, params);

    /* Phase 5: grammar-constrained tool decoding on the BATCHED path
     * (sg_tool_grammar_arms).  The driver samples per-slot in
     * its own thread from the slot's copied sampling params, so the hook carries
     * the driver-side advance/refresh/complete callbacks (sg_tg_attach_batched)
     * — it advances, gates and reports completion FROM the sampler.  The ctx
     * (grammar + state + piece table + hook) must OUTLIVE the whole decode: the
     * driver dereferences sl->sp.tool_grammar (== &tg.hook) every step until the
     * slot is released.  It is freed at `out` after bs_slot_release. */
    sg_tg_ctx tg; sg_tg_setup(&tg, model, params, model->config.n_vocab);
    sg_tg_attach_batched(&tg, &sp);
    tg.fl_always = sg_full_logits_always(params);
    sg_tg_mark_mask_density(&tg);
    /* The arming window needs a per-step re-arm to stay correct, and only the
     * L3 in-driver sampler has one (driver_arm_full_logits).  The legacy
     * lockstep exchange samples on THIS thread with no such seam, so it keeps
     * the whole-turn arm — the window is simply not offered there. */
    const bool arm_window = sg_grammar_arm_window_on() &&
                            bs_driver_sampling(sched);
    /* The slot's state owns the logits buffer the driver samples from.  This
     * arms the PREFILL row, whose logits sample the turn's FIRST token. */
    if (!arm_window)
        sg_require_full_logits(&tg, params, model, state, 1);
    else
        sg_arm_full_logits(&tg, params, model, state);

    /* OpenAI `logprobs` (inert unless the request asked).  The sink pointer is
     * copied into the slot's sampling params by bs_slot_bind, so the DRIVER
     * records every step it samples; this thread resolves the text as it drains
     * the ring.  The accumulator must outlive the decode — it is freed at `out`,
     * after the slot is released. */
    sg_logprob_acc lpacc;
    {
        int lp_cap = (params->max_tokens > 0 ? params->max_tokens : 512) + 8;
        sg_logprobs_init(&lpacc, result, params, lp_cap);
        sg_logprobs_attach(&lpacc, &sp);
    }

    int max_recent = sp.repeat_last_n;
    uint32_t *recent = calloc((size_t)(max_recent > 0 ? max_recent : 1),
                              sizeof(uint32_t));
    int n_recent = 0;

    /* ── 5. Prefill (L1: batched, L2: chunked with yield windows).
     * bs_prefill_begin takes the model's GPU turn: the C2 async phase
     * machine, the shared per-die scratch and the live SSM/KV GPU state all
     * support exactly ONE in-flight generation per model, so concurrent
     * prefill chunks must be serialized — and the turn is held through the
     * slot bind so no other forward can run between the last prefill token
     * and the SSM snapshot (the bind-right-after-prefill protocol).
     * Between chunks bs_prefill_yield hands the GPU to the decode driver
     * for one step (making the host mirror authoritative first), bounding
     * concurrent decoders' stall to one chunk instead of one prompt.  Any
     * failure before the bind must exit via bs_prefill_abort (releases the
     * turn and the slot). ── */
    /* PC-batched: resume-aware begin.  With a session id, if this slot's warm
     * pn_state still holds this session's committed history as a STRICT prefix
     * of `prompt` (at least one appended token), delta_start is set past it so
     * only the delta is prefilled.  SYS1-batched: a divergent history whose
     * system prefix matches this slot's boundary snapshot resumes at
     * delta_start == sys_len instead; capture_sys asks this turn's full
     * prefill to snapshot the boundary.  Otherwise the slot resets and
     * delta_start == 0 (full prefill).  Both resume cases guarantee
     * delta_start < n_prompt, so the prefill below always runs >=1 token and
     * recovers the next-token logits. */
    bs_resume_outcome resume = { 0, 0, PN_RESUME_COLD };
    pn_reqprof_phase_start(PN_REQ_CACHE_LOOKUP);
    if (use_cache)
        bs_prefill_resume_think(sched, slot, sid, prompt, n_prompt, sys_len,
                                params->think_history, &resume);
    else
        bs_prefill_begin(sched, slot);
    int delta_start = resume.delta_start;
    int capture_sys = resume.capture_sys;
    pn_reqprof_phase_end(PN_REQ_CACHE_LOOKUP);
    pn_reqprof_note_cache(use_cache, n_prompt, delta_start, sys_len);
    /* Same wire report as the single-user twin.  bs_prefill_resume_sys funnels
     * EVERY batched rung (append, projection, checkpoints, sys-prefix restore,
     * full reset) through this one delta_start, so the batched path reports the
     * number honestly rather than guessing at it — and this is the path the
     * fleet runs, so a gap here is the gap that ships. */
    result->cached_prompt_tokens = pn_prefix_cached_tokens(delta_start, n_prompt);
    result->resume_kind = resume.kind;
    result->prefill_tokens = n_prompt - delta_start;
    rt.prompt_tok = n_prompt;
    rt.prefill_tok = n_prompt - delta_start;
    /* Report a session id that now carries a SECOND conversation: every rung
     * missed because the system block itself changed, so this turn and the
     * displaced conversation's next turn both re-prefill in full. */
    result->session_conflict = (use_cache && bs_slot_session_conflict(sched, slot))
                             ? 1 : 0;

    pn_reqprof_phase_start(PN_REQ_PREFILL);
    int sys_cut = (capture_sys && delta_start == 0 && sys_len < n_prompt)
                      ? sys_len : -1;
    int ckpt_cut = (use_cache && bs_turn_ckpt_enabled()) ? bounds.turn_start
                                                          : -1;
    int div_cut = use_cache ? bs_div_ckpt_cut(sched, slot) : -1;
    tm_tensor *logits = sg_prefill_turn_sched(model, sched, slot, state, prompt,
                                              delta_start, n_prompt, sys_cut,
                                              ckpt_cut, div_cut, sid, params);
    pn_reqprof_phase_end(PN_REQ_PREFILL);
    sg_prefill_logits_hash(logits, n_prompt - delta_start, state->pos);
    if (!logits) {
        /* Distinguish a client-disconnect abort from a real prefill error (the
         * check is idempotent — a closed socket stays closed), exactly as the
         * single-user twin does.  Either way the slot's KV now covers a prompt
         * prefix nothing describes, so release through the resetting abort so
         * the next turn does not get steered back onto it. */
        int aborted = params->should_abort &&
                      params->should_abort(params->abort_ud);
        /* A client abort keeps what can resume — the prefix already prefilled
         * and the turn checkpoint (rung 1c); a real prefill error resets. */
        if (aborted)
            bs_prefill_abort_keep(sched, slot, use_cache ? sid : NULL, prompt,
                                  n_prompt, sys_len);
        else
            bs_prefill_abort_reset(sched, slot);
        free(prompt);
        free(recent);
        sg_tg_free(&tg);
        sg_logprobs_free(&lpacc);
        if (aborted)
            fprintf(stderr, "[batch_sched] slot %d: client disconnected during "
                    "prefill — turn aborted, slot released\n", slot);
        return aborted ? SG_ABORTED : -1;
    }
    /* Report the KV the session ACTUALLY occupies — the state's own position,
     * not n_prompt.  A think-elided resume keeps the previous turn's think block
     * in the KV, so the committed length runs ahead of the prompt the client
     * sent, and the client's compaction trigger reads this field. */
    result->kv_used_tokens = (int)state->pos;
    n_recent = sg_seed_recent(recent, max_recent, prompt, n_prompt);
    /* PC-batched: keep `prompt` (the full context) alive for bs_slot_finalize,
     * which needs prompt+generated as the committed history.  Freed at `out`. */
    if (!use_cache) {
        free(prompt);
        prompt = NULL;
    }

    /* ── 6. Sample first decode token from prefill logits ── */
    pn_reqprof_phase_start(PN_REQ_FIRST_SAMPLE);
    uint32_t next = pn_sample_next_token((float *)logits->data,
                                         model->config.n_vocab,
                                         &sp, recent, n_recent);
    pn_reqprof_phase_end(PN_REQ_FIRST_SAMPLE);
    t_first = sg_request_timing_now_ms();
    if (pn_ht_on()) {
        ht_first = pn_ht_now();
        ht_cpu0 = pn_ht_thread_cpu();
        ht_rd0 = pn_ht_run_delay();
    }
    /* Token #1 is sampled HERE, on the owner thread, and that sample advanced
     * the grammar (the batched hook carries advance).  The driver's first step
     * forwards `next` and its row is read before the driver has re-armed
     * anything — so this is the one arming point the driver cannot own.  Same
     * rule as everywhere else: arm after the advance, before the forward. */
    if (arm_window)
        sg_arm_full_logits(&tg, params, model, state);
    pn_reqprof_phase_start(PN_REQ_DECODE);
    tm_free(logits);
    logits = NULL;

    /* ── 7/8. Decode.  Two drivers:
     *
     *   L3 (default): bs_slot_bind_ex hands the scheduler's driver this
     *   request's sampler (params + repeat window + budget); the driver
     *   samples every step itself and never waits on this thread again.
     *   This thread only drains tokens (bs_slot_pop_token) and does
     *   detokenize + emitter + SSE — entirely OFF the batch's critical
     *   path, so a slow/stalled client cannot stall the other users.
     *
     *   Legacy (QWEN35_BS_DRIVER_SAMPLE=0): the original lockstep
     *   logits/token exchange. ── */
    emitter em;
    if (pn_ht_on())
        em_init(&em, sg_ht_delta_shim, &ht_delta, params->stop, params->n_stop,
                params->has_tools, params->dialect);
    else
        em_init(&em, delta_fn, delta_ud, params->stop, params->n_stop,
                params->has_tools, params->dialect);
    em_set_tool_tap(&em, params->tool_tap, params->tool_tap_ud);
    if (sg_think_close_is_token(model))
        em_set_think_close_token(&em);
    /* No open think block (No-Think, or a non-think template) → start the emitter
     * in content mode (see the mirror at the other em_init site). */
    if (sg_thinking_suppressed(params->thinking) ||
        !sg_chat_style_is_think(model->chat_style))
        em.in_think = 0;

    int eos_id    = ft_eos_id(model->tokenizer);
    int im_end    = ft_token_to_id(model->tokenizer, "<|im_end|>");
    int n_vocab   = model->config.n_vocab;
    int max_tokens = params->max_tokens > 0 ? params->max_tokens : 512;

    /* Repetition-loop guard — the batched twin of the single-user arm.  This is
     * the path the deployed fleet runs, so it is the one that matters most; the
     * detector sits on THIS thread (which already drains every token to
     * detokenize and stream it), never in the scheduler's driver, so it adds
     * nothing at all to the batch's critical path. */
    rld_detector reploop;
    sg_reploop_arm(&reploop);

    const char *finish = "length";
    bool reploop_stop = false;   /* keeps the post-loop em.stopped fixup from
                                  * overwriting the degeneration finish_reason */
    bool stalled = false;        /* the decode watchdog fired (SG_STALLED) */
    int generated = 0;

    if (bs_driver_sampling(sched)) {
        /* EOS as the very first token: nothing to decode — release the GPU
         * turn + slot without binding (the legacy path burned one step
         * here; the emitted stream is identical). */
        if ((int)next == eos_id || (int)next == im_end) {
            free(recent);
            bs_prefill_abort(sched, slot);
            finish = "stop";
            em_flush(&em);
            goto out;
        }
        /* Arm the driver's sampler.  `recent` currently holds the prompt
         * tail; the driver appends each stepped token (starting with
         * `next`) before sampling — exactly the legacy owner order, so
         * streams are bit-identical at temp 0.  Budget = max_tokens - 1
         * (this thread already holds token #1 = `next`). */
        if (bs_slot_bind_ex(sched, slot, next, &sp, recent, n_recent,
                            max_tokens - 1,
                            use_cache ? sid : NULL,
                            use_cache ? prompt : NULL,
                            use_cache ? n_prompt : 0) != 0) {
            free(recent);
            free(prompt);
            em_free(&em);
            sg_tg_free(&tg);
            rld_free(&reploop);
            /* slot + GPU turn were already released inside bind */
            return -1;
        }
        free(recent);   /* the driver owns its own copy */

        /* AUTO reasoning-budget force-close (batched, per slot): mirror the
         * single-user cap.  A think-style model was prefilled with "<think>"
         * OPEN (unless No-Think closed it), so the slot starts inside <think>;
         * the driver counts tokens spent reasoning and force-decodes "</think>"
         * once the per-model budget is hit without a self-close.  Arm it now,
         * while the slot is BOUND (the driver rebuilds its map before the first
         * step, and the arm takes sched->mu, so this races nothing).  Budget
         * <= 0 or a non-think model → no-op (byte-identical decode). */
        pn_reasoning_plan reason_plan =
            pn_reasoning_plan_for_turn((params->reasoning_budget > 0 &&
                                        sg_chat_style_is_think(model->chat_style))
                                           ? params->reasoning_budget : 0,
                                       max_tokens, params->reasoning_reserve,
                                       params->model_name);
        int reason_budget = reason_plan.hard;
        bool reason_in_think = reason_budget > 0 &&
                               !(sg_thinking_suppressed(params->thinking) &&
                                 sg_chat_style_is_think(model->chat_style));
        if (reason_budget > 0) {
            uint32_t close_toks[8];
            int n_close = sg_encode_think_close(model, close_toks,
                                                (int)(sizeof(close_toks) /
                                                      sizeof(close_toks[0])));
            bs_slot_arm_reasoning_budget(sched, slot, &reason_plan,
                                         reason_in_think, close_toks, n_close,
                                         next);
        }

        /* Emit token #1, then drain the driver's ring. */
        {
            char *piece = decode_token(model, next);
            /* Token #1 was sampled on THIS thread (from the prefill logits), so
             * it is the accumulator's first record; the driver's records follow
             * it in ring order. */
            const sg_token_logprob *lp = NULL;
            if (em.in_think) sg_logprobs_skip(&lpacc);
            else lp = sg_logprobs_take(&lpacc, result, model, piece);
            if (lp) em_note_token_logprob(&em, lp);
            if (piece) { em_feed(&em, piece); }
            free(piece);
            generated++;
            (void)rld_feed(&reploop, next);
        }
        while (!em.stopped) {
            /* Stop decoding once the client is gone (every 8 tokens, to bound
             * the syscall — the single-user loop's cadence).  The tokens
             * generated so far are valid: the finalize + commit below keeps
             * them and finish="stop" ends the turn cleanly, so the session can
             * still append-resume if the client comes back. */
            if ((generated & 7) == 0 && params->should_abort &&
                params->should_abort(params->abort_ud)) {
                finish = "stop";
                break;
            }
            /* Watched wait: a client that leaves while no token is flowing, or
             * a slot the driver stopped feeding, hands control back here so
             * the slot is released instead of held until restart. */
            uint32_t tok;
            bs_pop_watch watch = { params->should_abort, params->abort_ud, 0,
                                   bs_decode_stall_ms() };
            pn_ht_bind(&ht_own);
            uint64_t ht = pn_ht_now();
            bs_pop_status prc = bs_slot_pop_token_watched(sched, slot, &tok, &watch);
            ht = pn_ht_lap(PN_HT_O_POP_WAIT, ht);
            if (prc == BS_POP_STALLED) {
                fprintf(stderr, "[batch_sched] slot %d sid=%s: no token for %d ms after %d "
                        "generated - stream stalled, slot released\n", slot,
                        use_cache ? sid : "-", watch.stall_ms, generated);
                stalled = true;
                break;
            }
            if (prc < 0) { finish = "stop"; break; }  /* gone, step error, shutdown */
            if (prc == BS_POP_FINISHED) {
                /* The driver finishes a slot on EOS, on budget exhaustion, AND
                 * on grammar completion.  Only the middle one is "length" — a
                 * turn that stopped because the constrained object CLOSED is a
                 * clean stop, and reporting "length" would tell the client its
                 * perfectly complete JSON had been truncated. */
                if (sg_tg_live(&tg) && sg_tg_done(&tg)) finish = "stop";
                break;
            }
            if ((int)tok == eos_id || (int)tok == im_end) {
                /* The driver finished the slot on this end token, so the walk
                 * is no longer touched there: close a "required" call the
                 * model stopped inside (the single-user loop's rule). */
                finish = sg_tg_close_required_call(&tg, params, &em) ? "tool_calls" : "stop";
                break;
            }
            char *piece = decode_token(model, tok);
            ht = pn_ht_lap(PN_HT_O_DETOK, ht);
            const sg_token_logprob *lp = NULL;
            if (em.in_think) sg_logprobs_skip(&lpacc);
            else lp = sg_logprobs_take(&lpacc, result, model, piece);
            if (lp) em_note_token_logprob(&em, lp);
            ht = pn_ht_lap(PN_HT_O_LOGPROB, ht);
            if (piece) { em_feed(&em, piece); }
            free(piece);
            ht = pn_ht_lap(PN_HT_O_EMIT, ht);
            /* Reverse coupling (batched): the driver's sampler advances the grammar
             * in-thread; if it has decided forced-mode reasoning is over (</think>
             * or budget) but the emitter still routes to <think>, flip it to CONTENT
             * so the compelled call is parsed as a tool_call (mirror of the
             * single-user loop).  Asked at THIS token's ring position: the live walk
             * runs ahead by the ring backlog, and reading it flipped the channel a
             * few tokens early, filing "\n</think>\n\n" as content. */
            if (em.in_think && bs_slot_reasoning_ended(sched, slot)) {
                em.in_think = 0;
                sg_log_reasoning_channel_flip(&tg, generated);
            }
            generated++;
            /* Repetition collapse.  Breaking here releases the slot exactly the
             * way the client-disconnect check above does — the driver drops it
             * from the next step's compaction and stops generating for it — so
             * the remaining budget is not spent re-emitting the same block. */
            const bool rld_tripped = rld_feed(&reploop, tok);
            pn_ht_lap(PN_HT_O_RLD, ht);
            if (rld_tripped) {
                finish = SG_FINISH_REPETITION_LOOP;
                reploop_stop = true;
                break;
            }
        }
        pn_ht_bind(NULL);
        t_dec_end = sg_request_timing_now_ms();
        ht_dec_end = pn_ht_now();
        if (ht_dec_end) {
            ht_cpu1 = pn_ht_thread_cpu();
            ht_rd1 = pn_ht_run_delay();
        }
        if (em.stopped && !reploop_stop)
            finish = "stop";
        em_flush(&em);
        if (stalled) {
            /* No finalize: it waits for the GPU turn, which a wedged step may
             * never give back, and the turn did not complete. The next turn's
             * resume checks the slot's block before trusting it. */
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }
        /* Hand the slot this turn's CLIENT RE-RENDER — the reply put back
         * through the same render+tokenize path the next request will take —
         * BEFORE finalize publishes the committed log, so next turn's resume can
         * match around the think block whichever way the client handles it: the
         * think BLOCK for a client that resends `reasoning_content` (the one
         * that keeps the model reasoning), the reply alone for one that drops
         * it.  A turn with no reproducible reply (a pure-reasoning runaway, or a
         * tool block that will not parse back) projects NULL and disables both
         * rungs; an empty think block projects no block and disables only the
         * reasoning-bearing one. */
        if (use_cache) {
            int rerender_n = 0;
            uint32_t *rerender =
                sg_project_client_rerender(model, em.content.data, em.tool.data,
                                           params->tools, em.dialect,
                                           &rerender_n);
            int think_block_n = 0;
            uint32_t *think_block =
                sg_encode_history_think_block(model, em.reasoning.data,
                                              &think_block_n);
            bs_slot_note_visible(sched, slot, prompt, n_prompt,
                                 think_prefill_len, think_block, think_block_n,
                                 rerender, rerender_n);
            free(think_block);
            free(rerender);
        }
        /* PC-batched: snapshot the slot's warm SSM+KV into its pn_state and
         * publish this session's committed history BEFORE releasing, while the
         * slot's GPU decode state is still valid.  No-op when not caching. */
        if (use_cache) {
            const double t_fin = sg_request_timing_now_ms();
            bs_slot_finalize(sched, slot);
            rt.finalize_ms = sg_request_timing_now_ms() - t_fin;
        }
        bs_slot_release_stats(sched, slot, &rstats);   /* driver drops the slot from the
                                         * next step's compaction */
        goto out;
    }

    /* ── Legacy lockstep path.  bs_slot_bind releases the GPU turn taken by
     * bs_prefill_begin (success or failure) and releases the slot itself on
     * failure. ── */
    if (bs_slot_bind(sched, slot, next) != 0) {
        free(recent);
        free(prompt);
        em_free(&em);
        sg_tg_free(&tg);
        rld_free(&reploop);
        /* slot + GPU turn were already released inside bs_slot_bind */
        return -1;
    }

    /* Wait for first batch step's logits (driver already has `next`). */
    float *slot_logits = NULL;
    if (bs_slot_await_first(sched, slot, &slot_logits) != 0) {
        em_flush(&em);
        free(recent);
        bs_slot_release_stats(sched, slot, &rstats);
        goto out;
    }

    /* The token we deposited via bs_slot_bind was `next` — it is the first
     * generated token.  We need to emit it and then exchange the second
     * token with the driver. */
    {
        /* Emit `next` (the one we just deposited): decode and stream it. */
        if ((int)next != eos_id && (int)next != im_end) {
            char *piece = decode_token(model, next);
            const sg_token_logprob *lp = NULL;
            if (em.in_think) sg_logprobs_skip(&lpacc);
            else lp = sg_logprobs_take(&lpacc, result, model, piece);
            if (lp) em_note_token_logprob(&em, lp);
            if (piece) { em_feed(&em, piece); free(piece); }
            generated++;
            (void)rld_feed(&reploop, next);
            if (max_recent > 0) {
                if (n_recent < max_recent) recent[n_recent++] = next;
                else {
                    memmove(recent, recent + 1,
                            (size_t)(max_recent - 1) * sizeof(uint32_t));
                    recent[max_recent - 1] = next;
                }
            }
        } else {
            finish = "stop";
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }
        if (em.stopped) {
            finish = "stop";
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }
    }

    /* Exchange loop: sample from slot_logits → check EOS/stop → emit →
     * post next token → wait for next logits.
     *
     * EOS check BEFORE posting to driver: once we know we're done, call
     * bs_slot_release (→ RETIRE) instead of posting, so the driver does
     * not waste a GPU step on an EOS/stopped token and does not deadlock
     * waiting for a token we'll never post. */
    for (int i = 0; i < max_tokens - 1; i++) {
        /* Client gone → stop (every 8 tokens; same cadence and same
         * keep-what-was-generated semantics as the L3 loop above). */
        if ((i & 7) == 0 && params->should_abort &&
            params->should_abort(params->abort_ud)) {
            finish = "stop";
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }
        /* Sample from slot logits deposited by driver. */
        uint32_t sampled = pn_sample_next_token(slot_logits, n_vocab,
                                                &sp, recent, n_recent);

        /* EOS check BEFORE posting. */
        if ((int)sampled == eos_id || (int)sampled == im_end) {
            finish = sg_tg_close_required_call(&tg, params, &em) ? "tool_calls" : "stop";
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }

        /* Emit sampled token. */
        char *piece = decode_token(model, sampled);
        const sg_token_logprob *lp = NULL;
        if (em.in_think) sg_logprobs_skip(&lpacc);
        else lp = sg_logprobs_take(&lpacc, result, model, piece);
        if (lp) em_note_token_logprob(&em, lp);
        if (piece) { em_feed(&em, piece); free(piece); }
        generated++;

        if (max_recent > 0) {
            if (n_recent < max_recent) recent[n_recent++] = sampled;
            else {
                memmove(recent, recent + 1,
                        (size_t)(max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = sampled;
            }
        }

        /* Repetition collapse (legacy lockstep twin): stop exactly the way an
         * em.stopped stop does — the token is already emitted and counted, so
         * releasing the slot here ends the turn cleanly. */
        if (rld_feed(&reploop, sampled)) {
            finish = SG_FINISH_REPETITION_LOOP;
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }

        if (em.stopped) {
            finish = "stop";
            em_flush(&em);
            free(recent);
            bs_slot_release_stats(sched, slot, &rstats);
            goto out;
        }

        /* Post sampled token to driver and wait for the next step's logits. */
        slot_logits = NULL;
        if (bs_slot_step_wait(sched, slot, sampled, &slot_logits) != 0) {
            finish = "stop";
            break;
        }
    }

    em_flush(&em);
    free(recent);
    bs_slot_release_stats(sched, slot, &rstats);

out:
    /* The batched twin of the single-user [repetition] line.  The fleet runs
     * batched, so this is the copy that makes the production rate countable —
     * `path=` tells the two decoders apart. */
    if (reploop.tripped)
        sg_reploop_log(&reploop, params,
                       bs_driver_sampling(sched) ? "batched" : "batched-legacy",
                       generated);
    rld_free(&reploop);
    /* PC-batched: `prompt` is kept alive past prefill (for bind_ex + finalize)
     * only when caching; free it here.  NULL when not caching (freed earlier). */
    free(prompt);
    /* Free the grammar ctx.  Safe here: the slot was already released
     * (bs_slot_release, above every path reaching `out`) under sched->mu, so the
     * driver — which reads sl->sp.tool_grammar (== &tg.hook) only under mu with
     * status == DECODE — will never touch it again after this point. */
    sg_tg_report_response_format(&tg, result);
    sg_tg_report_tool_call(&tg, result);
    sg_tg_report_mask_density(&tg);
    sg_require_full_logits(&tg, params, model, state, 0);
    sg_tg_free(&tg);
    /* Same reasoning for the logprob sink (also reached through sl->sp): the
     * slot is released, so the driver can no longer write records.  Frees only
     * the raw buffer — the RESOLVED entries live on the result. */
    sg_logprobs_free(&lpacc);
    pn_reqprof_phase_end(PN_REQ_DECODE);
    pn_reqprof_note_generated(generated);
    result->completion_tokens = generated;
    result->finish_reason     = finish;
    if (t_dec_end <= 0.0)
        t_dec_end = sg_request_timing_now_ms();   /* EOS first, legacy path */
    rt.ttft_ms    = t_first - t_req0;
    rt.generated  = generated;
    rt.decode_ms  = t_dec_end - t_first;
    rt.active_min = rstats.active_min;
    rt.active_max = rstats.active_max;
    rt.mtp_cycles = rstats.mtp_cycles;
    rt.finish     = finish;
    sg_request_timing_log(&rt);
    if (ht_first && ht_dec_end) {
        const pn_host_timing *hd = &rstats.host;
        pn_ht_report(hd, &ht_own, slot, generated, rt.decode_ms,
                     hd->first_launch_ns > ht_first
                         ? (double)(hd->first_launch_ns - ht_first) / 1e6 : 0.0,
                     ht_dec_end > hd->last_deposit_ns
                         ? (double)(ht_dec_end - hd->last_deposit_ns) / 1e6 : 0.0,
                     ht_cpu1 - ht_cpu0, ht_rd1 - ht_rd0);
    }
    result->content    = em.content.data  ? em.content.data  : strdup("");
    result->reasoning  = em.reasoning.data;
    result->tool_raw   = em.tool.data;
    em.content.data    = NULL;
    em.reasoning.data  = NULL;
    em.tool.data       = NULL;
    em_free(&em);
    return stalled ? SG_STALLED : 0;
}
