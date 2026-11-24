/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_grammar.h
 * @brief Grammar / state-machine that constrains tool-call decoding to the
 *        model's TRAINED XML emission (n2 toolcalling Phase 5).
 *
 * The Q3 nex-n2-mini drifts off its trained XML tool-call format (emits the
 * base model's JSON `{"tool_call":…}` habit, or prose). This module builds a
 * char-level FSM from the request's `tools` JSON that recognizes exactly the
 * trained emission:
 *
 *     <tool_call>\n
 *     <function=NAME>\n
 *     (<parameter=KEY>\nVALUE\n</parameter>\n)+
 *     </function>\n
 *     </tool_call>
 *
 * with NAME drawn from the tool names and KEY drawn from that tool's parameter
 * names.  The bare (no-wrapper) `<function=NAME>…</function>` form the Q3 model
 * favors is ALSO accepted.  VALUE is free text up to `</parameter>` — the same
 * bytes the parser ends a value on, so grammar and parser cannot disagree about
 * where a value ended.
 *
 * EACH KEY AT MOST ONCE.  A key this call has already written cannot be opened
 * again, and once every key is written "</function>" is the only way on, so a
 * call cannot loop on its own keys until max_tokens.  Same in the JSON dialect.
 *
 * REQUIRED ARGUMENTS.  Every key in that tool's schema `required` list must be
 * emitted before `</function>` becomes a legal continuation — so a call that
 * OMITS a required argument (the bare `<function=NAME>\n</function>`, which the
 * parser turns into `{}`) is structurally unreachable.  Same guarantee the JSON
 * dialect (tool_grammar_json.h) gives.  A tool with no `required` list is
 * unconstrained, exactly as before.
 *
 * AN EMPTY VALUE IS REJECTED WHERE THE SCHEMA SAYS SO, AND NOWHERE ELSE.
 * "</parameter>" is not a legal continuation of a value that holds no
 * non-whitespace char, IF that parameter's schema forbids an empty value —
 * a non-string `type`, or `minLength` ≥ 1 (tool_param_empty_value_rule.h, the
 * same predicate the client enforces on arrival, compiled into both dialects from one
 * function).  So `bash {"command":""}` and `read {"path":""}` are unreachable,
 * while `edit`'s `new_string:""` — where empty IS the deletion — still is not.
 * Requiredness is not the test: both of those keys are `required`.
 *
 * THE GUARD DIES; IT DOES NOT CONSUME.  An earlier attempt refused the
 * terminator by taking its first char as VALUE CONTENT instead, which is an
 * ACCEPT: tg_token_allowed said yes, the model emitted the close anyway, and the
 * walk then sat inside VALUE for the rest of the turn while the parser had
 * already closed the parameter at those same bytes — a visible empty argument
 * turned into a silently missing one (`read {}`), 9/9 truncated live calls.
 * Reaching TG_DEAD instead is what makes the token carrying that close MASKED,
 * so it is never drawn and the walk never leaves the parser's frame.  Cost of
 * having no guard at all, measured: qwen36-bf16 with the grammar armed emitted
 * `bash {"command":""}` seven times running before the run was stopped.
 *
 * The PARSER is unchanged and still reports an empty value it receives as
 * `{"path":""}` rather than `{}` — the grammar prevents emission, the parser
 * reports, the client judges.  Whether the model closed the CALL at all is a
 * separate question, answered by `tg_complete()`.
 *
 * ENUM VALUES.  A parameter carrying a string `enum` does NOT get a free-text
 * VALUE: it is a closed alternation over the listed literals, so an off-list
 * value is unreachable.  Enums that cannot be represented as a string
 * alternation (non-string members, a contradicting `type`, more members than the
 * cap) DEGRADE to free text rather than reject — see build_key_enum in the .c
 * for the exact admission rules.
 *
 * CONTAINER-TYPED VALUES.  A parameter whose schema `type` is "array" or
 * "object" does not get a free-text VALUE either: its bytes must be ONE
 * well-formed JSON value of that container kind, walked incrementally by
 * json_value_grammar.h, and "</parameter>" only becomes a legal continuation
 * once that value has CLOSED.  This is the gap the frame grammar could not
 * cover: the call skeleton was masked correctly and the bytes inside
 * `<parameter=todos>` were still free text — so a model that writes one
 * malformed nested object breaks not that argument but the whole `arguments`
 * string, because tool_calling.c's write_typed_value SPLICES a container-typed
 * value in RAW.  Scope is exactly that raw-splice set; every other type is
 * JSON-escaped or already re-validated there.  The schema INSIDE the value
 * (`items`, nested `properties`/`required`, member types) is still dropped:
 * `[{"wrong_key":1}]` is well-formed, reaches the tool, and is refused there.
 * Every other schema keyword (minLength, pattern, const, numeric bounds) is
 * still dropped, and `type` still does not constrain a SCALAR VALUE.
 *
 * FIELD PATTERNS.  The request field `field_patterns` (tool -> property ->
 * pattern, field_pattern.h syntax; tg_apply_field_patterns) puts a whole-value
 * regex on one free-text string value.  It is a request field, not the schema
 * `pattern` keyword, because the schema is rendered into the prompt.  The value
 * judged is the one the parser hands the client: the bytes between
 * "<parameter=KEY>" and "</parameter>", whitespace-trimmed at both ends
 * (tool_calling.c dup_trim) — so the trained "\nVALUE\n" framing newlines are
 * not part of it.  Every byte that leaves no way to a matching value is masked
 * (whitespace is allowed while a close could still follow it, since trailing
 * whitespace is trimmed), and "</parameter>" closes only on a match.  It never
 * forces the key present.  A value wrapped in "<value>…</value>" (MiniCPM; the
 * parser peels it) is judged with the wrapper, the same unmodelled shape as the
 * empty-value floor's known gap.
 *
 * A constrained sampler asks, for each candidate token, `tg_token_allowed()`
 * ("does appending this token's text keep a valid grammar prefix?") and masks
 * (logit = -INFINITY) the ones that don't; after a token is drawn it calls
 * `tg_advance()` with the chosen token's text to walk the FSM forward.
 *
 * REASONING / PROSE composition.  The grammar starts in a FREE state that
 * accepts arbitrary text (the model reasons and may answer in prose with NO
 * tool call at all).  It only ENGAGES the skeleton once a tool-call opener
 * ("<tool_call>" or a bare "<function=") actually appears — mirroring the
 * serve_emitter's opener rule.  So grammar-on never forces a spurious call: a
 * prose answer stays entirely in FREE and is unconstrained.  (The sampler's
 * "arm the mask" trigger is the same opener signal; see engine.c.)
 *
 * FORCED-CALL mode (Phase-5 refinement; tool_choice "required" / a named
 * function).  In AUTO mode above the grammar can only clean up format once the
 * model *chooses* to start a call — it cannot COMPEL one, so the Q3 model's
 * prose-refusal / JSON-drift (which happen in the FREE state, before any opener)
 * slip past.  Forced mode fixes that: the FREE state is replaced by "optional
 * reasoning, then the opener is REQUIRED" — the model may reason first (its
 * trained <think>…</think> block is allowed), but the moment reasoning ends the
 * ONLY tokens left allowed are the ones that lead toward the "<tool_call>" /
 * "<function=" opener, so prose and JSON are impossible and the response *is* a
 * valid XML call.  With a specific function pinned, NAME is fixed to it (other
 * tool names are rejected).  The reasoning phase ends when "</think>" appears in
 * the model's raw output — the grammar DETECTS THIS ITSELF (it watches its own
 * token stream) so forcing is self-contained and does NOT depend on the serve
 * emitter's <think> flag, which the Q3 model does not reliably trigger (it
 * sometimes reasons in the content channel with no clean </think>).  The serve
 * layer still calls tg_reasoning_end() as a harmless secondary trigger.
 *
 * Pure C, libc-only — NO engine/CUDA deps — so it is unit-tested in isolation
 * (tests/test_tool_grammar.c) without the forward path.
 *
 * @status Phase 5 module — the mask is wired behind an OFF-by-default flag
 *         (PANTHEON_TOOL_GRAMMAR=1); grammar-off generation is byte-identical.
 *         The free-text VALUE terminator is now the parser's own "</parameter>",
 *         so the two cannot disagree about where a value ended, and tg_complete()
 *         is the serve's truncation verdict.
 * @issues max_tokens and the repetition latch finish a turn without consulting
 *         the grammar, so a call truncated by them is DETECTED (tg_complete
 *         false → refused on the wire) rather than prevented.  End tokens are
 *         governed for "required" only (tg_end_allowed + tg_closing_text: the
 *         serve closes a call the model stopped inside); AUTO and a named
 *         function keep the old rule.
 * @todo Per-state allowed-token cache for the fixed-skeleton states (see the
 *       perf note in tool_grammar.c) to shrink the per-step mask cost.
 */

#ifndef PN_TOOL_GRAMMAR_H
#define PN_TOOL_GRAMMAR_H

#include <stdbool.h>

#include "json_reader.h"   /* jr_value */

/* Opaque compiled grammar (tool + parameter name sets + the fixed skeleton). */
typedef struct tg_grammar tg_grammar;

/* Decoding mode for a walk state (see the header note).
 *   TG_MODE_AUTO   : engage the skeleton only after an opener appears; FREE prose
 *                    is unconstrained (the original Phase-5 behavior).
 *   TG_MODE_FORCED : compel a call — after optional reasoning the opener is
 *                    REQUIRED, so prose/JSON are masked out and the response is a
 *                    valid XML call.  Pin a specific function with tg_state_new_forced. */
typedef enum {
    TG_MODE_AUTO = 0,
    TG_MODE_FORCED,
} tg_mode;

/* Mutable per-request walk position over the grammar. Cheap to copy by value
 * (it is a small POD), which the allowed-check exploits: it clones the state,
 * walks the candidate text on the clone, and reports whether the clone stayed
 * on a valid prefix — the caller's live state is untouched until tg_advance. */
typedef struct tg_state tg_state;

/**
 * Build a grammar from the request's `tools` JSON array (OpenAI shape:
 * `[{"type":"function","function":{"name":…,"parameters":{"properties":{…}}}}]`).
 * Returns NULL if `tools` is not a non-empty array or on OOM. Free with
 * tg_free.
 */
tg_grammar *tg_build(const jr_value *tools);

/** Free a grammar (NULL-safe). */
void tg_free(tg_grammar *g);

/** Allocate a fresh AUTO-mode walk state at the start (FREE region). NULL on OOM. */
tg_state *tg_state_new(const tg_grammar *g);

/**
 * Allocate a FORCED-mode walk state (Phase-5 forced-call).  After optional
 * reasoning the opener is REQUIRED (no prose/JSON escape).  If `forced_name` is
 * non-NULL it must be one of the grammar's tool names and the call's NAME is
 * pinned to it (any other name is rejected); NULL forces *some* tool ("required").
 * Returns NULL on OOM or if `forced_name` is not a known tool.
 */
tg_state *tg_state_new_forced(const tg_grammar *g, const char *forced_name);

/**
 * Allocate the walk state for `tool_choice:"required"` (forced, any tool) with
 * THINK-GATED reasoning — the XML twin of tgj_state_new_required.
 *
 * `think_open` says what the prompt did: true when it left a <think> block OPEN
 * (a think-style template with thinking on), false when there is no block.
 *   - think_open: everything before the model's own "</think>" is reasoning and
 *     is never judged — no opener tracking and no char budget (max_tokens and
 *     the serve's reasoning_budget force-close bound it; an injected close
 *     gets NO landing window, tg_reasoning_end_injected).  From the token
 *     after "</think>" the opener is compelled as in tg_state_new_forced.
 *   - !think_open: compelled from the first token.
 * The legacy tg_state_new_forced grace (opener tracking + TG_REASON_BUDGET) is
 * kept for the named-function form.  NULL on OOM / NULL grammar.
 */
tg_state *tg_state_new_required(const tg_grammar *g, bool think_open);

/**
 * Apply the request field `required_if_reasoned` (a JSON array of property
 * names) to a built grammar.  For each listed name, on every tool whose
 * properties have it, the key is REQUIRED (like a schema `required` key, so
 * "</function>" is unreachable without it) on the reply's FIRST call when the
 * reply REASONED — its think block held a non-whitespace char before its close
 * (reasoned_reply_tracker.h).  Otherwise the key stays optional: allowed, never
 * forced, never blocked.  Unknown names, non-string members and a non-array
 * value are ignored; NULL or an empty array leaves the grammar unchanged.  Call
 * once, after tg_build and before any state walks.
 */
void tg_require_if_reasoned(tg_grammar *g, const jr_value *names);

/**
 * Apply the request field `field_patterns` (a JSON object: tool name -> object
 * of property name -> pattern string, field_pattern.h syntax) to a built
 * grammar.  Each pattern binds the free-text value of that property of that
 * tool: see "FIELD PATTERNS" above.  Unknown tools and properties, non-string
 * patterns, and properties that are not free-text strings (a non-string `type`,
 * an `enum`, a container) are ignored; NULL or an empty object leaves the
 * grammar unchanged.  Call once, after tg_build and before any state walks.
 * Returns false when a pattern does not compile or on OOM (the request parse
 * refuses a bad pattern first: field_patterns_request.h).
 */
bool tg_apply_field_patterns(tg_grammar *g, const jr_value *field);

/**
 * STEER a patterned value into its pattern instead of only refusing its close.
 * `token_text` is the token the model itself ranks first (the sampler's
 * unmasked argmax).  When it is refused because it tries to END a non-empty
 * free-text value whose `field_patterns` rule does not match yet — the
 * whitespace of the trained "\n</parameter>" tail, or "</parameter>" whole —
 * the walk is put on the pattern's forced continuation
 * (fp_forced_continuation: " from: " for Argus's finding rule): until those
 * bytes are spelled, a token off them is not allowed, so the mask leaves only
 * that continuation.  Returns true when it did so; false (walk untouched) for
 * anything else — an allowed token, any other refusal, no pattern, an empty
 * value, a value already mid-"</parameter>", a steer still pending, or a
 * pattern that forces no bytes there.
 */
bool tg_steer(tg_state *st, const char *token_text);

/**
 * The prompt left a think block open: watch it so the walk knows whether the
 * reply reasoned.  tg_state_new_required(g, true) does this itself; the AUTO and
 * named-function walks need this call (before the first tg_advance).  It changes
 * nothing else about the walk.  Without it a walk is never "reasoned".
 */
void tg_watch_reasoning(tg_state *st);

/** Has the walk seen reasoning (non-whitespace think text outside the close)? */
bool tg_reply_reasoned(const tg_state *st);

/** How many calls of this reply have reached a NAME (0 before the first). */
int tg_calls_named(const tg_state *st);

/** Largest text tg_closing_text can produce, NUL included. */
#define TG_CLOSING_MAX 96

/**
 * The text that takes this walk to a COMPLETE call, if the call can still be
 * closed as it stands: the trained tail ("\n</parameter>\n</function>\n
 * </tool_call>" and its suffixes), shortest first, optionally after the rest of
 * a literal the walk is part-way through — each checked on a copy of the walk.
 * None (-1) when a required key is still missing, a schema-floored value is
 * empty, an enum/JSON value is unfinished, the NAME/KEY is incomplete, or the
 * walk is not inside a call.
 *
 * Used when the model ends a "required" turn INSIDE its call: the serve appends
 * this so the turn ends in the call the model meant — the value the parser
 * reads from the unclosed call — instead of a refused incomplete call.
 *
 * @return the length written to `out` (NUL-terminated), or -1
 */
int tg_closing_text(const tg_state *st, char *out, int cap);

/**
 * May the turn END (an end token, which has no text) here?  Always once the
 * mask is off — reasoning, a landing window, AUTO prose, a completed call.  With
 * the mask on: never before the call opens, and inside the call only when
 * tg_closing_text can close it.  The serve wires this for "required" only.
 */
bool tg_end_allowed(const tg_state *st);

/** Heap copy of a walk state (NULL on OOM / NULL input).  Free with
 *  tg_state_free.  Used to look ahead over speculative drafts without touching
 *  the live walk. */
tg_state *tg_state_clone(const tg_state *st);

/** Overwrite `dst` with `src` (both from this module; same grammar). */
void tg_state_copy(tg_state *dst, const tg_state *src);

/** Free a walk state (NULL-safe). */
void tg_state_free(tg_state *st);

/**
 * Close reasoning only on the "</think>" TOKEN: a tg_advance whose whole text is
 * "</think>".  The same text spelled out of ordinary tokens stays reasoning
 * (think_close_rule.h).  Set it when the model's close is one token; leave it off
 * otherwise.  Call before the first tg_advance.
 *
 * Thread-safe: no (mutates st)
 * Reentrant: yes
 * Signal-safe: no
 */
void tg_set_think_close_token(tg_state *st);

/**
 * Signal that the reasoning phase has ended.  In FORCED mode this releases the
 * reasoning grace and REQUIRES the opener from here on (prose/JSON masked out).
 * A no-op in AUTO mode, and idempotent.
 *
 * NOTE: forcing no longer DEPENDS on this call — the grammar self-detects
 * "</think>" in its own token stream (tg_advance), so a request forces reliably
 * even if the serve emitter never flips its <think> flag (BUG 1).  This remains
 * as a harmless secondary trigger for callers that track reasoning externally.
 */
void tg_reasoning_end(tg_state *st);

/**
 * Signal that the reasoning phase was ended by an INJECTED "</think>" — one the
 * SERVE force-decoded because the turn spent its `reasoning_budget`, NOT one the
 * model chose (batch_sched.c driver_reason_step / serve_generation.c
 * sg_run_completion).
 *
 * Same effect as tg_reasoning_end, PLUS a bounded LANDING WINDOW: for the next
 * TG_LANDING_BUDGET chars the FREE state behaves like AUTO (prose unconstrained,
 * an opener still engages the skeleton), and only then is the opener compelled.
 *
 * WHY.  A force-closed model is mid-sentence.  Compelling the opener from the
 * very next char makes it emit "<function=NAME>\n<parameter=KEY>\n" around an
 * unfinished thought, and everything it still had to say — often a complete tool
 * call — lands in the free-text VALUE instead of being a call.  That is the
 * frame-capture defect recorded on the live fleet (a recorded native-XML capture,
 * three recorded Hermes-shape captures).  The window lets the model land: finish its
 * sentence, emit its own close, and open the call itself.
 *
 * `tool_choice=required` is preserved — the opener is still compelled, just up
 * to TG_LANDING_BUDGET chars later.  The window ends EARLY the moment the model
 * emits its own "</think>".  A no-op in AUTO mode, and idempotent (a second call
 * simply re-grants the window).
 *
 * A "required" walk (tg_state_new_required) gets NO window: the opener is
 * compelled from the next token, as after the model's own close.  The serve's
 * close lands on a line end (reasoning_budget.h), and the window let a model
 * write its thought on as prose and run to max_tokens with no call.
 */
void tg_reasoning_end_injected(tg_state *st);

/**
 * Would appending `token_text` keep the walk on a valid grammar prefix?
 * Does NOT mutate `st` (walks a clone). While in the FREE region (before an
 * opener) every text is allowed. Returns false for empty/NULL text only if it
 * would be invalid (empty text is always allowed — it is a no-op).
 */
bool tg_token_allowed(const tg_state *st, const char *token_text);

/** Walk the state forward by `token_text` (must have been allowed). */
void tg_advance(tg_state *st, const char *token_text);

/**
 * Has the walk reached a COMPLETE tool call (a full `</function>` closed, with
 * or without the `</tool_call>` wrapper)? Used to let the sampler release the
 * constraint / stop once a whole call has been emitted.
 *
 * ALSO THE TRUNCATION VERDICT.  False while inside a call means the model stopped
 * part-way through emitting it — the turn ended on EOS, `max_tokens` or a latch
 * that never consults the grammar.  That is the one bit that separates a call the
 * model finished from one it abandoned, and the serve gates on it before handing
 * anything to the client (SG_TOOL_GRAMMAR_INCOMPLETE in serve_generation.h,
 * TC_PARSE_CALL_INCOMPLETE on the wire).  It is only meaningful for a walk that
 * actually entered the skeleton — pair it with tg_in_call().
 */
bool tg_complete(const tg_state *st);

/**
 * Is the walk currently INSIDE a tool call (past an opener, not yet FREE-only)?
 * The sampler uses this together with the opener detection to decide whether to
 * apply the mask this step.
 */
bool tg_in_call(const tg_state *st);

/**
 * Has the walk left the grammar entirely (TG_DEAD)?
 *
 * NOT the same question as !tg_in_call(): a walk still in TG_FREE has also never
 * entered a call, and the two mean opposite things.  FREE = the model has not
 * started one (prose, or a dialect this grammar does not model).  DEAD = it
 * started one and went off-grammar, at which point tg_mask_active() reports
 * INACTIVE and every remaining token of the turn is sampled with NO constraint.
 * Collapsing the two is why "the AUTO grammar silently switched itself off" had
 * no observable answer; the serve reports the DEAD case on its own line.
 */
bool tg_is_dead(const tg_state *st);

/**
 * Should the sampler MASK this step?  This is the gating policy the serve layer
 * feeds into pn_tool_grammar_hook.active:
 *   - AUTO mode   : true only once inside a call (== tg_in_call) — FREE prose is
 *                   unconstrained, byte-identical to the pre-forced behavior.
 *   - FORCED mode : true once reasoning has ended (tg_reasoning_end) AND the call
 *                   is not yet complete — so the post-reasoning FREE state is
 *                   constrained to opener-leading tokens (compels the call), and
 *                   the skeleton is constrained thereafter.  Before reasoning ends
 *                   it stays false so the <think> block is unconstrained.
 * Always false once the call is complete (tg_complete) or DEAD.
 */
bool tg_mask_active(const tg_state *st);

/* DEBUG: current forced-mode reasoning-grace char count (for tracing the budget).
 * Returns 0 in AUTO mode or when st is NULL. */
int tg_debug_reason_chars(const tg_state *st);

/* DEBUG: chars of landing-window grace still owed after an injected reasoning
 * close (tg_reasoning_end_injected).  0 = no window live, i.e. the opener is
 * compelled normally.  Returns 0 when st is NULL. */
int tg_debug_landing_chars(const tg_state *st);

/* Has forced-mode reasoning ended (the opener is now compelled)?  True once the
 * grammar has seen "</think>" OR the reasoning budget tripped.  The serve loop
 * uses this to flip the emitter out of its <think> channel so the compelled call
 * routes to CONTENT (and is parsed as a tool_call) even when the model never
 * emitted a clean "</think>" itself.  Always false in AUTO mode / when NULL. */
bool tg_forced_reasoning_ended(const tg_state *st);

#endif /* PN_TOOL_GRAMMAR_H */
