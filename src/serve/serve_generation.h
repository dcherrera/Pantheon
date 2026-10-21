/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_generation.h
 * @brief Chat-completion generation for the serve layer. Builds the engine's
 *        ChatML(+think) prompt from an OpenAI messages[] array, prefills and
 *        decodes via the existing engine forward + sampler paths (no generation
 *        logic is reinvented here), and splits the model's <think>…</think>
 *        reasoning from the answer into separate channels.
 *
 * @status Non-streaming and streaming completions work on the CPU host path;
 *         reasoning is surfaced as a deepseek-style separate channel.  PC2:
 *         the single-user path delta-prefills via pn_prefix_cache when the
 *         client sends a session id (sg_prefix_ctx); no session id → full
 *         prefill (byte-identical to before).  PC3: the batched path
 *         (sg_run_completion_batched) also delta-prefills when a session id is
 *         sent — session→slot affinity + bs_prefill_resume delta-prefill +
 *         bs_slot_finalize snapshots the slot's warm SSM/KV back into its
 *         pn_state (via pn_qwen35moe_decode_slot_ssm_to_host).  No session id →
 *         full prefill (byte-identical to the pre-PC batched path).  SYS1:
 *         the single-user path additionally caches the SYSTEM-PREFIX boundary
 *         state per session (two-stage prefill + snapshot), so a turn whose
 *         history diverges (reasoning models) still skips the system prefill.
 *         OpenAI decoding fields: `response_format` (json_object) is enforced by
 *         a third grammar on the SAME constrained-decoding hook the tool
 *         grammars use; `seed`, `logit_bias`, `presence_penalty`,
 *         `frequency_penalty` and `logprobs` are applied in the sampler
 *         (src/core/sampling_modifiers.c) on BOTH the single-user and batched
 *         paths.  Every one of them is a no-op at its default.
 * @issues None
 * @todo
 *   - [ ] SYS1 on the batched path (bs_prefill_resume system-prefix restore).
 */

#ifndef PN_SERVE_GENERATION_H
#define PN_SERVE_GENERATION_H

#include <stdbool.h>

#include "../../include/pantheon.h"
#include "../core/prefix_cache.h"
#include "batch_sched.h"
#include "tool_calling.h"   /* tc_dialect */

/* sg_run_completion return codes. 0 = success; -1 = generic error; the value
 * below is a distinct code the HTTP layer maps to a clean 503 (Retry-After)
 * instead of a 500 — the request was refused pre-flight to protect the host
 * (MemoryAware §6.2), nothing failed. */
#define SG_ADMIT_DENIED (-2)

/* Returned when the prompt exceeds the model's per-session context budget (the
 * hard KV cap / die-or-host KV budget). KV Phase 1: the serve NEVER silently
 * drops the oldest tokens (which shattered the prefix cache and stalled the
 * client) — it refuses the turn and hands back a distinct, machine-readable
 * signal so the client can compact its history and retry. The HTTP layer maps
 * this to a `context_over_budget` error body. `sg_result.kv_used_tokens` /
 * `kv_cap_tokens` carry the numbers the client needs to decide how much to
 * shed. Nothing failed — the request was refused pre-flight to protect the KV
 * reservation. */
#define SG_CONTEXT_OVER_BUDGET (-4)

/* Returned when the caller's should_abort() fired mid-turn — the client
 * disconnected, so the turn stopped early and the model slot was released. The
 * cache session (if any) is reset so the next turn full-prefills cleanly.
 * Nothing failed; the HTTP layer just drops the already-gone connection. */
#define SG_ABORTED (-3)

/* finish_reason for a turn ended by the repetition-loop detector
 * (repetition_loop_detector.h): the model stopped producing new text and was
 * re-emitting one block of tokens over and over.  It is a DISTINCT value on
 * purpose — a collapse reported as "stop" reads to the caller as a model that
 * chose to finish, which is exactly how the measured 31×-repeated-paragraph
 * turn was scored as a tidy stop.  A caller tells degeneration from a normal
 * finish by this string and nothing else: it never appears on a healthy turn,
 * and it OUTRANKS "tool_calls" in the response body (a turn that degenerated
 * must not be handed back as a clean call to execute). */
#define SG_FINISH_REPETITION_LOOP "repetition_loop"

/* Cooperative abort check: sg_run_completion calls this between prefill chunks
 * and periodically during decode; a non-zero return aborts the turn. Used to
 * free the model slot the instant the client disconnects instead of finishing a
 * ~100s cold prefill nobody is reading. NULL disables it (unchanged behavior). */
typedef int (*sg_should_abort_fn)(void *user_data);

/* One chat message from the request (borrowed pointers). */
typedef struct {
    const char *role;     /* "system" | "user" | "assistant" | "tool" */
    const char *content;  /* message text (may be empty, never NULL) */
    /* Assistant turns only: the `reasoning_content` the serve streamed for that
     * turn, resent verbatim by a client that wants the think block replayed into
     * the prompt.  NULL/empty = the classic behaviour (reasoning dropped).
     * When present, the assistant turn renders as
     * "<think>{reasoning}</think>{content}" instead of bare "{content}" — which
     * is what the serve itself committed to the KV, so the prefix cache's
     * append-resume path can match the previous turn exactly instead of
     * diverging at the forced <think> opener.  See build_prompt(). */
    const char *reasoning;
} sg_message;

/* Resolved OpenAI `tool_choice` (n2 tool-calling Phase 5 — forced-call mode).
 *   AUTO      : default — the model decides; grammar only cleans up format once a
 *               call starts (the original Phase-5 behavior; FREE prose allowed).
 *   NONE      : never call a tool — no grammar is built (req->tools is NULLed).
 *   REQUIRED  : MUST emit some tool call — the grammar's FREE state is replaced by
 *               "optional <think> reasoning, then REQUIRE the opener".
 *   FUNCTION  : MUST call the named function (`forced_name`) — as REQUIRED, and the
 *               NAME is pinned to that one tool. */
typedef enum {
    SG_TOOL_CHOICE_AUTO = 0,
    SG_TOOL_CHOICE_NONE,
    SG_TOOL_CHOICE_REQUIRED,
    SG_TOOL_CHOICE_FUNCTION,
} sg_tool_choice_kind;

/* Resolved OpenAI `response_format` for one completion.
 *   TEXT        : the default — no decode constraint at all.
 *   JSON_OBJECT : the content channel is constrained to ONE syntactically valid
 *                 JSON object (response_format_grammar.c), so a client that sent
 *                 `{"type":"json_object"}` really can JSON.parse the reply.
 * A `json_schema` request is resolved to JSON_OBJECT by the HTTP layer (the
 * schema's SHAPE is not enforced) and reported as downgraded on the wire — the
 * serve layer only ever sees TEXT or JSON_OBJECT. */
typedef enum {
    SG_RESPONSE_FORMAT_TEXT = 0,
    SG_RESPONSE_FORMAT_JSON_OBJECT,
} sg_response_format;

/* Resolved per-request thinking control — the request's `enable_thinking`.
 *
 * This is the ONE reasoning control a Qwen3.x-family chat template actually
 * exposes, and it is BINARY: the template's generation prompt is either
 * "<think>\n" (reason) or "<think>\n\n</think>\n\n" (do not).  There is no
 * graded "reasoning effort" in the template, so there is no graded mode here.
 *
 *   DEFAULT : the request said nothing → the process-wide PANTHEON_NO_THINK
 *             env decides, exactly as before.  Zero value, so a memset-0
 *             sg_gen_params (every non-HTTP caller) is byte-identical to today.
 *   ON      : `"enable_thinking": true`  → reason, even under PANTHEON_NO_THINK.
 *   OFF     : `"enable_thinking": false` → prefill the CLOSED empty think block.
 *
 * Resolve with sg_thinking_suppressed(); never test the enum directly, or the
 * prompt builder and the stream emitter can disagree about which channel the
 * first generated token belongs to. */
typedef enum {
    SG_THINKING_DEFAULT = 0,
    SG_THINKING_ON,
    SG_THINKING_OFF,
} sg_thinking_mode;

/* Sampling + limits for one completion. */
typedef struct {
    float  temperature;
    float  top_p;
    int    top_k;
    int    max_tokens;
    char **stop;          /* borrowed array of stop strings (may be NULL) */
    int    n_stop;
    bool   has_tools;     /* request carried tools → arm bare "<function="
                           * detection in the stream emitter (a wrapped
                           * "<tool_call>" is detected regardless). */
    tc_dialect dialect;   /* resident model's tool-call dialect, resolved
                           * post-route (chat_apply_tool_dialect).  Drives the
                           * stream emitter's opener set and the grammar setup:
                           * TC_DIALECT_ORNITH_JSON diverts the JSON block and
                           * builds NO XML grammar (Phase 1).  Default 0 =
                           * TC_DIALECT_QWEN_XML → byte-identical XML behavior. */
    const void *tools;    /* borrowed request `tools` JSON (jr_value*), or NULL.
                           * Only consulted for Phase-5 grammar-constrained
                           * decoding, and only when PANTHEON_TOOL_GRAMMAR=1;
                           * unset/NULL → the sampler is byte-identical. */
    sg_tool_choice_kind tool_choice;  /* resolved `tool_choice` (default AUTO).
                           * Phase-5 forced-call mode: REQUIRED/FUNCTION compel a
                           * call (grammar's FREE state forced to the opener);
                           * only acts when PANTHEON_TOOL_GRAMMAR=1 and tools are
                           * present.  AUTO leaves the pre-forced behavior. */
    const char *forced_name;  /* target function name for SG_TOOL_CHOICE_FUNCTION;
                           * NULL otherwise.  Owned by the caller's request (must
                           * outlive the completion); NOT aliased into a JSON DOM
                           * that could be freed independently. */
    sg_should_abort_fn should_abort;  /* optional; non-zero return aborts the turn
                           * (client disconnect → release the slot). NULL = never. */
    void *abort_ud;       /* passed to should_abort (e.g. the client fd). */
    const char *model_name;  /* registry name of the model serving this turn, for
                           * LOGGING only — never a routing/behavior input.  The
                           * reasoning force-close is an abnormal event and its log
                           * line must name the model it happened to.  Borrowed
                           * from the caller (must outlive the completion); NULL is
                           * fine and logs as "?". */
    int   reasoning_budget;  /* AUTO-path reasoning cap (tokens).  On a think-style
                           * model, once this many tokens have been generated INSIDE
                           * the model's <think> block without a self-emitted
                           * "</think>", the decode loop force-closes reasoning
                           * (emits the "</think>" sequence into the state + stream)
                           * so the model concludes and answers instead of running
                           * away mid-thought (saber runaway).  Resolved as: request
                           * override (if the body carried it) ELSE the per-model
                           * default ELSE 0.  0 (or absent) = DISABLED — today's
                           * unbounded behavior, byte-identical.  Independent of the
                           * tool-forcing grammar (that only fires on tool_choice
                           * REQUIRED/FUNCTION; this fires on the plain AUTO turn). */

    sg_thinking_mode thinking; /* request `enable_thinking` (default 0 = the env
                           * decides).  Unlike reasoning_budget — which lets the
                           * model OPEN a think block and then guillotines it
                           * mid-sentence — this decides BEFORE generation whether
                           * the block is opened at all, so nothing is ever cut
                           * off mid-word.  See sg_thinking_mode. */

    /* ── OpenAI sampling/decoding fields (all default to "off", so a request
     * that sends none of them behaves byte-identically to before) ── */

    sg_response_format response_format;  /* constrain the answer to JSON.  Only
                           * ever JSON_OBJECT when the request carried NO tools —
                           * the two constraints cannot both drive one decode, so
                           * the HTTP layer resolves the precedence (tools win) and
                           * tells the client which it got. */

    float presence_penalty;   /* OpenAI `presence_penalty`  (0 = off) */
    float frequency_penalty;  /* OpenAI `frequency_penalty` (0 = off) */

    const pn_logit_bias_entry *logit_bias;  /* OpenAI `logit_bias`, borrowed
                           * (owned by the request; must outlive the completion).
                           * NULL / n_logit_bias 0 = off. */
    int   n_logit_bias;

    bool     seed_set;    /* request carried `seed` → sample from a per-request
                           * RNG stream instead of the process-global rand(), so
                           * the same request + seed replays the same tokens. */
    uint64_t seed;

    bool  logprobs;       /* OpenAI `logprobs`: report per-token logprobs on the
                           * response.  false (the default) = off, nothing is
                           * computed and the sampler is untouched. */
    int   logprobs_top;   /* OpenAI `top_logprobs`: how many ALTERNATIVES to
                           * report per token (0..PN_LOGPROBS_MAX_TOP).  Only
                           * consulted when `logprobs` is true. */

    void (*tool_tap)(const char *bytes, size_t n, void *ud);
                          /* streaming only: receives every byte the emitter
                           * diverts into the raw tool channel, as it is
                           * generated (em_set_tool_tap), so tool-call deltas
                           * reach the client while the call is written.
                           * NULL (the default) = no tap. */
    void *tool_tap_ud;
} sg_gen_params;

/*
 * One token's reported logprobs (OpenAI `logprobs.content[]`).
 *
 * `token` and `top_token[]` are the DECODED text of the ids, since that is what
 * the wire format carries; the numbers are log-probabilities of the model's own
 * distribution (before temperature / top-k / top-p / any grammar mask).
 *
 * Only ANSWER tokens are reported.  Tokens generated inside the model's <think>
 * block belong to the separate `reasoning_content` channel, which OpenAI's
 * `logprobs.content[]` does not describe — and reporting them on one path but
 * not the other is exactly the kind of divergence that has bitten this file.
 */
typedef struct {
    char  *token;                            /* owned decoded UTF-8 */
    float  logprob;
    int    n_top;
    char  *top_token[PN_LOGPROBS_MAX_TOP];   /* owned */
    float  top_logprob[PN_LOGPROBS_MAX_TOP];
} sg_token_logprob;

/*
 * Streaming delta sink.  `is_reasoning` is 1 for reasoning_content deltas
 * (inside <think>), 0 for answer content deltas.
 *
 * `logprobs`/`n_logprobs` carry the logprobs of the tokens that produced THIS
 * delta (borrowed; valid for the duration of the call, and stable for the whole
 * completion since the array is preallocated).  They are only ever non-empty for
 * content deltas of a request that asked for logprobs — the emitter holds a
 * partial tail back to catch markers, so one delta can cover several tokens and
 * a token can produce no delta at all, which is why they ride the delta instead
 * of being counted by the client.
 */
typedef void (*sg_delta_fn)(const char *text, int is_reasoning,
                            const sg_token_logprob *const *logprobs,
                            int n_logprobs, void *user_data);

/* End-of-turn verdict from the TOOL-CALL decoding grammar: did the call the model
 * started ever CLOSE?
 *
 * This exists because a truncated tool call is not detectable from its text.  The
 * parser is a string scanner, and the bytes a truncated call leaves behind are a
 * legal PREFIX of a good call — every one of them scans into a well-formed-looking
 * call with an argument quietly missing (`read {}`, `write` with no `content`).
 * The grammar is the only component that knows it was still mid-skeleton when the
 * turn ended, and it computes that bit already; this carries it to the HTTP layer,
 * which refuses to hand such a call to the client (TC_PARSE_CALL_INCOMPLETE).
 *
 * ABSENT is not a soft "no": it means no tool grammar ran (PANTHEON_TOOL_GRAMMAR
 * off, no tools on the request, a response_format turn, or a build that fell back
 * to inert), so there is no verdict and the caller must not gate on one.  A
 * grammar that never entered a call is also ABSENT — the model answered in prose,
 * or emitted a dialect this grammar does not model, and neither is a truncation.
 *
 * Same discipline as response_format_ok below: a status that claims a guarantee it
 * did not deliver is the defect the field exists to stop. */
enum {
    SG_TOOL_GRAMMAR_ABSENT     =  0,  /* no tool grammar ran / never entered a call */
    SG_TOOL_GRAMMAR_COMPLETE   =  1,  /* the grammar closed a whole call */
    SG_TOOL_GRAMMAR_INCOMPLETE = -1,  /* entered a call, turn ended before it closed */
};

/* Result of a completion (owned strings; free with sg_result_free). */
typedef struct {
    char       *content;           /* answer text (owned, never NULL on success) */
    char       *reasoning;         /* reasoning text (owned; NULL if none) */
    char       *tool_raw;          /* raw model text from "<tool_call>" onward
                                    * (owned; NULL if the model emitted no call).
                                    * The caller parses it with tc_parse_tool_calls. */
    int         prompt_tokens;
    int         completion_tokens;
    int         cached_prompt_tokens; /* of `prompt_tokens`, how many the prefix
                                    * cache let this turn SKIP — the prompt
                                    * tokens NOT computed here (OpenAI's
                                    * usage.prompt_tokens_details.cached_tokens).
                                    * Always the cache's own resume boundary
                                    * (pn_prefix_cached_tokens), never a count
                                    * recomputed from the prompt. 0 on a cold
                                    * turn, and always <= prompt_tokens.
                                    *
                                    * Without it a client measuring prefill rate
                                    * divides prompt_tokens by TTFT and gets a
                                    * number that RISES the more the serve
                                    * caches — a 10x inflation that reads as
                                    * progress. This is the divisor correction. */
    int         context_capped;    /* 1 if admission clamped the prompt window
                                    * (MemoryAware §6.2 rung 3) — the client is
                                    * told its context was shortened. */
    int         kv_used_tokens;    /* KV Phase 1: this session's KV occupancy in
                                    * tokens after this turn (prompt + committed
                                    * generation), for the client's compaction
                                    * trigger. 0 = unknown / not reported. */
    int         kv_cap_tokens;     /* the EFFECTIVE per-session context window in
                                    * tokens (pn_effective_ctx_window: the
                                    * configured cap AFTER the memory ladder's
                                    * runtime clamp). 0 = unknown.
                                    *
                                    * The same number GET /v1/models advertises
                                    * as `context_window` — deliberately: the
                                    * client treats the model list as its INITIAL
                                    * plan and this as the LIVE truth, so a
                                    * mid-session clamp reaches it with no
                                    * polling and no extra endpoint. */
    int         session_conflict;  /* 1 when this request's session id already
                                    * identified a DIFFERENT conversation (a
                                    * different system prompt): the client is
                                    * sharing one session id across two
                                    * conversations, which costs BOTH of them
                                    * their warm prefix on every turn. Batched
                                    * path only — see bs_slot_session_conflict.
                                    * 0 = no conflict / not detectable. */
    const char *finish_reason;     /* "stop" | "length" | "tool_calls" |
                                    * SG_FINISH_REPETITION_LOOP (static string) */
    int         response_format_ok; /* 1 when a requested `response_format` was
                                    * actually ENFORCED — the JSON grammar ran and
                                    * closed a complete object.  0 means the
                                    * constraint did not hold, and the HTTP layer
                                    * must NOT tell the client "applied": a status
                                    * that claims a guarantee it did not deliver is
                                    * the exact defect this field exists to stop.
                                    * Meaningless (0) when no response_format was
                                    * requested. */
    int         tool_grammar_verdict; /* SG_TOOL_GRAMMAR_* — whether the tool-call
                                    * grammar closed the call it entered.  The
                                    * HTTP layer gates on this before emitting any
                                    * call parsed out of `tool_raw`. */
    sg_token_logprob *logprobs;    /* OpenAI `logprobs`: one entry per generated
                                    * token, owned (freed by sg_result_free).
                                    * NULL unless the request asked for them. */
    int         n_logprobs;
} sg_result;

/**
 * PC2 — optional prefix-cache context for a completion.  When non-NULL and
 * `cache`/`sid` are set, sg_run_completion acquires the session's cached model
 * state, prefills ONLY the delta tokens (the LCP of the cached history and this
 * prompt), and commits the full prompt+reply back to the session — so an
 * agentic follow-up prefills the new tokens instead of the whole context.
 * `pc_lock` (if non-NULL) serializes cache access (the cache is thread-unsafe).
 *
 * When the whole struct pointer is NULL (or cache/sid NULL), sg_run_completion
 * behaves EXACTLY as before: it resets the passed-in `state` and full-prefills
 * — byte-identical to the pre-PC serve path (safe-by-default).
 */
typedef struct {
    pn_prefix_cache *cache;    /* session cache (NULL → no caching) */
    const char      *sid;      /* session id (NULL → no caching) */
    pthread_mutex_t *pc_lock;  /* serializes cache ops (may be NULL) */
} sg_prefix_ctx;

/**
 * Does this request arm the tool-call grammar?  The PANTHEON_TOOL_GRAMMAR
 * policy: "0" never; "1" every request carrying tools; unset (or any other
 * value) only tool_choice "required" — the client asked for a call every turn.
 *
 * Thread-safe: yes (reads the environment)
 */
bool sg_tool_grammar_arms(const sg_gen_params *params);

/**
 * Can a client rely on tool_choice "required" with thinking ON for a model
 * speaking `dialect`?  True when a grammar models the dialect
 * (tc_dialect_grammar_models: ORNITH_JSON, QWEN_XML) and "required" arms it (the
 * env does not opt out): the think block is free and the turn ends in a parsed
 * call.  False for MINICPM_XML.  Advertised on /v1/models as
 * `tool_choice_required_with_thinking`.
 *
 * Thread-safe: yes
 */
bool sg_tool_choice_required_with_thinking(tc_dialect dialect);

/**
 * Run one chat completion. Resets `state`, builds the ChatML prompt, prefills
 * and decodes. If `delta_fn` is non-NULL, streams deltas as tokens arrive;
 * regardless, the full reasoning + answer are accumulated into *result.
 *
 * @param pc  Optional prefix-cache context (PC2).  NULL → full prefill on the
 *            passed-in `state` (legacy behaviour).  When set, the session's
 *            cached state is used and `state` is ignored.
 *
 * @return 0 on success, -1 on error.
 *
 * Thread-safe: no (drives a single model/state; uses the shared sampler scratch)
 */
int sg_run_completion(pn_model *model, pn_state *state,
                      const sg_message *messages, int n_messages,
                      const sg_gen_params *params,
                      sg_delta_fn delta_fn, void *delta_ud,
                      sg_result *result, const sg_prefix_ctx *pc);

/** Free the owned strings of a result. NULL-safe. */
void sg_result_free(sg_result *result);

/**
 * C3 — batched completion via a slot scheduler.
 *
 * Same semantics as sg_run_completion but runs inside the batch scheduler:
 * prefill is single-user, BATCHED (L1: chunk-wise
 * pn_qwen35moe_forward_batch_cuda) and CHUNKED (L2: the GPU turn is yielded
 * to the decode driver between chunks); decode is batched with any other
 * users concurrently generating on the same model, sampled in-driver (L3,
 * default — this thread only drains a token ring and streams SSE, off the
 * batch's critical path; QWEN35_BS_DRIVER_SAMPLE=0 restores the legacy
 * lockstep exchange).
 *
 * `slot` must be a valid slot index previously acquired with bs_slot_acquire
 * (the caller checks slot availability BEFORE beginning SSE so a 429 can be
 * returned before the SSE stream is opened).  The function takes ownership of
 * the slot and releases it on completion (or error); the caller must NOT call
 * bs_slot_release separately.
 *
 * PC-batched: `sid` (may be NULL) is the prefix-cache session id.  When
 * non-NULL/non-empty the slot's warm pn_state is delta-prefilled (only the
 * tokens appended since the session's last turn on this slot) and re-committed
 * for next-turn reuse; the session→slot affinity is a soft hint applied at
 * acquire time.  NULL/empty → full prefill every turn, byte-identical to the
 * pre-PC batched path (safe-by-default).
 *
 * Thread-safe: YES — each call runs on its own request thread; shared state
 * is only the scheduler (its own mutex) and _Thread_local sampler scratch.
 */
int sg_run_completion_batched(pn_model *model, pn_batch_sched *sched,
                              int slot,
                              const sg_message *messages, int n_messages,
                              const sg_gen_params *params,
                              sg_delta_fn delta_fn, void *delta_ud,
                              sg_result *result, const char *sid);

/**
 * Measured prefill throughput in tokens/sec (K80 serving-ops Phase 7): an
 * exponential moving average over the real prefills this process has run
 * (fed from sg_run_completion). Returns 0.0 before the first measurable
 * prefill — callers fall back to a static constant (PN_PREFILL_TOK_PER_S) for
 * the restart estimate until a real sample lands. Thread-safe.
 */
double sg_prefill_tok_per_s_measured(void);

#endif /* PN_SERVE_GENERATION_H */
