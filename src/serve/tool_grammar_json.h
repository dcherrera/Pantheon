/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_grammar_json.h
 * @brief Grammar / state-machine that constrains tool-call decoding to ornith's
 *        RL-learned JSON emission (per-model tool-call dialect, Phase 2).
 *
 * The XML sibling (tool_grammar.h) constrains the Qwen/N2/MiniCPM XML skeleton.
 * Ornith-1.0-9B instead emits a markdown-fenced JSON object in the content
 * channel:
 *
 *     ```json
 *     {"function": "NAME", "arguments": {"KEY": <value>, …}}
 *     ```
 *
 * (the fence is optional, "function" may be spelled "name", "arguments" may be
 * spelled "parameters").  This module builds a char-level FSM from the request's
 * `tools` JSON that recognizes exactly that shape, with NAME drawn from the tool
 * names, KEY drawn from that tool's parameter names, EVERY `required` key
 * present, and each value typed to its schema (string → a quoted string,
 * integer/number → a numeric literal, boolean → true/false, object/array → a
 * balanced {…}/[…]).  An empty `{}` / a missing required key is structurally
 * impossible to complete — the direct analog of the XML grammar's missing-key
 * guard, and the reason ornith gets the same `required` guarantee saber gets
 * from the XML grammar.
 *
 * A string value may be EMPTY only where the schema allows it: the closing quote
 * is off-grammar on a value holding no non-whitespace char IF that parameter has
 * a length floor (a non-string `type`, or `minLength` ≥ 1 —
 * tool_param_empty_value_rule.h, shared with the XML dialect so one schema
 * cannot mean two things depending on which model answered).  `read`'s `path`
 * cannot be `""`; `edit`'s `new_string` still can, because empty is what a
 * deletion IS.
 *
 * A parameter carrying a string `enum` is tighter still: its value is a closed
 * alternation over the listed literals, so an off-list value is unreachable.
 * Enums that cannot be represented as a string alternation (non-string members,
 * a contradicting `type`, more members than the cap) DEGRADE to the plain type
 * rather than reject — see build_key_enum in the .c for the exact admission
 * rules.  Every other schema keyword (minLength, pattern, const, numeric
 * bounds, nested shape) is still dropped.
 *
 * FIELD PATTERNS.  The request field `field_patterns` (tgj_apply_field_patterns;
 * the XML twin is tg_apply_field_patterns) puts a whole-value regex
 * (field_pattern.h) on one string value.  The value judged is the DECODED JSON
 * string — `\"` is `"`, `\n` a newline, `\uXXXX` its UTF-8 bytes (a surrogate
 * pair as one code point, a lone surrogate as its 3-byte form) — exactly what
 * the client gets, since ornith's parser forwards the value verbatim.  Every
 * char that leaves no way to a matching value is masked (with a pattern, an
 * invalid escape or a non-hex \u digit is off-grammar too), and the closing
 * quote is legal only on a match.  It never forces the key present.
 *
 * The public contract MIRRORS tool_grammar.h exactly (tgj_* == tg_*), so the
 * serve sampler hook (pn_tool_grammar_hook / sg_tg_* glue in serve_generation.c)
 * is reused unchanged — only the grammar object it drives differs by dialect.
 *
 * REASONING / AUTO / FORCED composition is identical to the XML grammar: FREE
 * reasoning/prose is unconstrained in AUTO and engages the skeleton only after
 * an opener; FORCED mode replaces the FREE grace with "optional reasoning, then
 * the opener is REQUIRED" (prose/garbage masked out), with the reasoning close
 * self-detected from the raw "</think>" in the stream and a reasoning budget as
 * a backstop.  Ornith runs reasoning_budget 0, so the grace is minimal — the
 * budget + self-detected close both arm the opener cleanly (0-budget path).
 *
 * NO LANDING WINDOW HERE, deliberately.  The XML grammar has
 * tg_reasoning_end_injected: a bounded grace after a "</think>" the SERVE
 * force-decoded, so the opener is not compelled around a thought the model had
 * not finished (the frame-capture defect — tool_grammar.c "FRAME CAPTURE", and
 * tool_calling.c recover_captured_call).  This grammar has the same structure
 * and would have the same defect, but it is UNREACHABLE today: reasoning_budget
 * 0 means the serve never injects a close for an ornith turn, so there is
 * nothing to land from.  That is a CONFIG fact, not a structural one — if ornith
 * (or any future ORNITH_JSON model) is ever given a non-zero reasoning_budget,
 * add tgj_reasoning_end_injected here and wire it in
 * sg_tg_reasoning_ended_injected / sg_tg_attach_batched alongside the XML one.
 *
 * Pure C, libc-only — NO engine/CUDA deps — unit-tested in isolation
 * (tests/test_tool_grammar_json.c) without the forward path.
 *
 * @status Phase 2 module — armed per sg_tool_grammar_arms (tool_choice
 *         "required" by default, every tool request with PANTHEON_TOOL_GRAMMAR=1)
 *         and selected by dialect (TC_DIALECT_ORNITH_JSON) in sg_tg_setup.
 *         "required" uses the think-gated walk (tgj_state_new_required).
 * @issues None
 * @todo None
 */

#ifndef PN_TOOL_GRAMMAR_JSON_H
#define PN_TOOL_GRAMMAR_JSON_H

#include <stdbool.h>

#include "json_reader.h"   /* jr_value */

/* Opaque compiled grammar (tool names + per-tool param keys/types/required). */
typedef struct tgj_grammar tgj_grammar;

/* Decoding mode for a walk state (same semantics as tg_mode).
 *   TGJ_MODE_AUTO   : engage the skeleton only after an opener; FREE prose free.
 *   TGJ_MODE_FORCED : compel a call — after optional reasoning the opener is
 *                     REQUIRED, so prose/garbage is masked out. */
typedef enum {
    TGJ_MODE_AUTO = 0,
    TGJ_MODE_FORCED,
} tgj_mode;

/* Mutable per-request walk position (small POD, cheap to copy by value — the
 * allowed-check clones it and walks the candidate on the clone). */
typedef struct tgj_state tgj_state;

/**
 * Build a grammar from the request's `tools` JSON array (OpenAI shape). Returns
 * NULL if `tools` is not a non-empty array or on OOM. Free with tgj_free.
 *
 * Thread-safe: yes (no shared state)
 */
tgj_grammar *tgj_build(const jr_value *tools);

/** Free a grammar (NULL-safe). */
void tgj_free(tgj_grammar *g);

/** Allocate a fresh AUTO-mode walk state at the start (FREE region). NULL OOM. */
tgj_state *tgj_state_new(const tgj_grammar *g);

/**
 * Allocate a FORCED-mode walk state.  After optional reasoning the opener is
 * REQUIRED.  `forced_name` (if non-NULL) pins NAME to that tool; NULL forces
 * SOME tool ("required").  Returns NULL on OOM or if `forced_name` is not a
 * known tool.
 */
tgj_state *tgj_state_new_forced(const tgj_grammar *g, const char *forced_name);

/**
 * Allocate the walk state for `tool_choice:"required"` (forced, any tool) with
 * THINK-GATED reasoning.
 *
 * `think_open` says what the prompt did: true when it left a <think> block OPEN
 * (a think-style template with thinking on), false when there is no block to
 * wait for (thinking off, or a non-think template).
 *   - think_open: everything before the model's own "</think>" is reasoning and
 *     is never constrained or judged — no opener tracking inside it (a '{' the
 *     model writes while thinking is thought, not a call) and no char budget
 *     (the turn is bounded by max_tokens and, when set, the serve's
 *     reasoning_budget force-close, which ends reasoning via
 *     tgj_reasoning_end).  From the token after "</think>" the call is
 *     compelled exactly as in tgj_state_new_forced.
 *   - !think_open: reasoning has already ended; the call is compelled from the
 *     first token.
 * The legacy tgj_state_new_forced grace (opener tracking + an 8192-char
 * budget inside reasoning) is kept for the named-function form.
 *
 * Returns NULL on OOM / NULL grammar.
 */
tgj_state *tgj_state_new_required(const tgj_grammar *g, bool think_open);

/**
 * Apply the request field `required_if_reasoned` — the JSON twin of
 * tg_require_if_reasoned (tool_grammar.h): each listed name, on every tool whose
 * properties have it, is required on the reply's FIRST call when the reply
 * REASONED, and optional otherwise.  Unknown names, non-string members and a
 * non-array value are ignored; NULL or an empty array leaves the grammar
 * unchanged.  Call once, after tgj_build and before any state walks.
 * Returns false only on OOM (some names may then be left optional).
 */
bool tgj_require_if_reasoned(tgj_grammar *g, const jr_value *names);

/**
 * Apply the request field `field_patterns` (tool name -> object of property
 * name -> pattern string) — the JSON twin of tg_apply_field_patterns
 * (tool_grammar.h).  Only plain string values (schema `type` absent or
 * "string", no `enum`) take a pattern; unknown tools/properties and non-string
 * patterns are ignored; NULL or an empty object leaves the grammar unchanged.
 * Call once, after tgj_build and before any state walks.  Returns false when a
 * pattern does not compile or on OOM.
 */
bool tgj_apply_field_patterns(tgj_grammar *g, const jr_value *field);

/**
 * The JSON twin of tg_steer (tool_grammar.h): when `token_text` (the model's
 * own top token) is refused at the unescaped closing quote of a non-empty
 * string value whose field pattern does not match yet, put the walk on the
 * pattern's forced continuation (fp_forced_continuation, cut before the first
 * byte a JSON string would have to escape).  Returns true when it did so;
 * false (walk untouched) otherwise.
 */
bool tgj_steer(tgj_state *st, const char *token_text);

/**
 * The prompt left a think block open: watch it so the walk knows whether the
 * reply reasoned (tgj_state_new_required(g, true) does this itself).  Changes
 * nothing else about the walk.  Call before the first tgj_advance.
 */
void tgj_watch_reasoning(tgj_state *st);

/** Has the walk seen reasoning (non-whitespace think text outside the close)? */
bool tgj_reply_reasoned(const tgj_state *st);

/** How many calls of this reply have reached a NAME (0 before the first). */
int tgj_calls_named(const tgj_state *st);

/** Heap copy of a walk state (NULL on OOM / NULL input).  Free with
 *  tgj_state_free.  Used to look ahead over speculative drafts without
 *  touching the live walk. */
tgj_state *tgj_state_clone(const tgj_state *st);

/** Overwrite `dst` with `src` (both from this module; same grammar). */
void tgj_state_copy(tgj_state *dst, const tgj_state *src);

/** Free a walk state (NULL-safe). */
void tgj_state_free(tgj_state *st);

/**
 * Close reasoning only on the "</think>" TOKEN: a tgj_advance whose whole text is
 * "</think>".  The same text spelled out of ordinary tokens stays reasoning
 * (think_close_rule.h).  Set it when the model's close is one token; leave it off
 * otherwise.  Call before the first tgj_advance.
 *
 * Thread-safe: no (mutates st)
 * Reentrant: yes
 * Signal-safe: no
 */
void tgj_set_think_close_token(tgj_state *st);

/**
 * Signal that the reasoning phase has ended.  In FORCED mode this releases the
 * grace and REQUIRES the opener from here on.  A no-op in AUTO mode, idempotent.
 * The grammar ALSO self-detects "</think>" in its own stream, so forcing does
 * not depend on this call (it is a harmless secondary trigger).
 */
void tgj_reasoning_end(tgj_state *st);

/**
 * Would appending `token_text` keep the walk on a valid grammar prefix? Does
 * NOT mutate `st` (walks a clone). Empty/NULL text is always allowed (no-op).
 */
bool tgj_token_allowed(const tgj_state *st, const char *token_text);

/** Walk the state forward by `token_text` (must have been allowed). */
void tgj_advance(tgj_state *st, const char *token_text);

/**
 * The shortest text that takes this walk to a COMPLETE call, if the call can
 * still be closed as it stands: the open string value's quote, the arguments'
 * and the object's braces, the fence close — tried in that order of length and
 * checked by walking a copy.  None (-1) when a required key is still missing, a
 * value is unfinished in a way no closer fixes (a half literal, an open nested
 * container), or the walk is not inside a call.
 *
 * Used when the model ends its turn INSIDE the call (ornith closes the markdown
 * fence inside a long `write` string and stops): the serve appends this so the
 * turn ends in the call the model meant — what the parser's truncation repair
 * does for an unconstrained turn — instead of a refused incomplete call.
 *
 * @return the length written to `out` (NUL-terminated), or -1
 */
int tgj_closing_text(const tgj_state *st, char *out, int cap);

/**
 * May the turn END (an end token, which has no text) here?  Always once the
 * mask is off — reasoning, AUTO prose, a completed call.  For a "required" walk
 * with the mask on: never before the call opens (the turn must make a call),
 * and inside the call only when tgj_closing_text can close it.
 */
bool tgj_end_allowed(const tgj_state *st);

/** Has the walk reached a COMPLETE tool call (outer '}' + optional fence close)? */
bool tgj_complete(const tgj_state *st);

/** Is the walk currently INSIDE a call (past an opener, not FREE / DEAD)? */
bool tgj_in_call(const tgj_state *st);

/** Has the walk left the grammar entirely (TGJ_DEAD)?  See tg_is_dead() in
 *  tool_grammar.h for why this is a different question from !tgj_in_call(). */
bool tgj_is_dead(const tgj_state *st);

/**
 * Should the sampler MASK this step?  Same gating policy as tg_mask_active:
 *   AUTO   : true only once inside a call (FREE prose unconstrained).
 *   FORCED : also true once reasoning ended (or an opener is underway during
 *            reasoning), so the opener is compelled.  Always false once complete
 *            or DEAD.
 */
bool tgj_mask_active(const tgj_state *st);

/* DEBUG: current forced-mode reasoning-grace char count (0 in AUTO / NULL). */
int tgj_debug_reason_chars(const tgj_state *st);

/* Has forced-mode reasoning ended (opener now compelled)?  True once "</think>"
 * was seen OR the reasoning budget tripped.  False in AUTO / when NULL. */
bool tgj_forced_reasoning_ended(const tgj_state *st);

#endif /* PN_TOOL_GRAMMAR_JSON_H */
