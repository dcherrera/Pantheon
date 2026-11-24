/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file response_format_grammar.h
 * @brief Grammar / state-machine that constrains a whole completion to valid
 *        JSON — the decode-side implementation of OpenAI `response_format`.
 *
 * The two tool-call grammars (tool_grammar.h / tool_grammar_json.h) constrain a
 * TOOL CALL's skeleton.  This one constrains the ANSWER: when a request carries
 * `response_format: {"type":"json_object"}` the model's content channel must be
 * a single, syntactically valid JSON object — which is exactly what the OpenAI
 * contract promises and what a client that sends the field is already assuming.
 * Ignoring the field (the pre-existing behaviour) hands that client prose it
 * will try to `JSON.parse` and fail on, with nothing on the wire saying why.
 *
 * The public contract MIRRORS tool_grammar.h (rfg_* == tg_*), so the serve
 * sampler hook (pn_tool_grammar_hook / the sg_tg_* glue in serve_generation.c)
 * is reused unchanged — only the grammar object it drives differs.  There is
 * exactly ONE constrained-decoding path in the serve; this is a third grammar
 * on it, not a second mechanism.
 *
 * The constrained emission:
 *
 *     (optional reasoning, unconstrained) </think>
 *     (whitespace) { …one balanced, syntactically valid JSON object… }
 *
 * REASONING GRACE.  A think-style model is prefilled with an open `<think>`, so
 * its first tokens are reasoning and must NOT be constrained.  `reasoning_grace`
 * starts the walk in a FREE region that ends when "</think>" appears in the raw
 * token stream (self-detected here, exactly like the tool grammars — the serve
 * emitter's think flag is not reliable on every model) or when the caller calls
 * rfg_reasoning_end().  With `reasoning_grace` false the walk starts constrained
 * immediately (non-think models, or No-Think mode).
 *
 * JSON validated: objects, arrays, strings (with \\ escapes and \\uXXXX), numbers
 * (at least one digit, and a number may not end on '.', 'e', '+' or '-'), the
 * literals true/false/null, and whitespace between tokens.  Nesting is capped at
 * RFG_MAX_DEPTH; a deeper open is simply not an allowed continuation.
 *
 * Pure C, libc-only — NO engine/CUDA deps — so it is unit-tested in isolation
 * (tests/test_response_format_grammar.c) without the forward path.
 *
 * @status json_object works and is wired into both the single-user and the
 *         batched decode paths.  `text` is a no-op (the default).
 * @issues `json_schema` is NOT enforced: a request carrying one is decoded under
 *         the json_object constraint (valid JSON, arbitrary shape) and the serve
 *         reports "schema_downgraded" on the wire.  Per-schema grammar
 *         compilation is unimplemented.
 * @todo
 *   - [ ] Compile a `json_schema` into a typed FSM (keys/required/types), the
 *         way tool_grammar_json.c compiles a tool's parameter schema.
 */

#ifndef PN_RESPONSE_FORMAT_GRAMMAR_H
#define PN_RESPONSE_FORMAT_GRAMMAR_H

#include <stdbool.h>

/* Maximum JSON container nesting the grammar will allow the model to open. */
#define RFG_MAX_DEPTH 64

/* Requested response format, resolved from the request's `response_format`. */
typedef enum {
    RFG_FORMAT_TEXT = 0,      /* {"type":"text"} or absent — no constraint */
    RFG_FORMAT_JSON_OBJECT,   /* {"type":"json_object"} — content must be one
                               * syntactically valid JSON object */
    RFG_FORMAT_JSON_SCHEMA,   /* {"type":"json_schema", …} — SHAPE NOT ENFORCED;
                               * decoded as JSON_OBJECT and reported as
                               * downgraded (see @issues) */
} rfg_format;

/* Mutable per-request walk position (a small POD — the allowed-check clones it
 * and walks the candidate on the clone, leaving the live state untouched). */
typedef struct rfg_state rfg_state;

/**
 * Allocate a walk state for `format`.
 *
 * @param format          RFG_FORMAT_JSON_OBJECT / RFG_FORMAT_JSON_SCHEMA build a
 *                        state; RFG_FORMAT_TEXT returns NULL (nothing to
 *                        constrain — the caller leaves the feature off).
 * @param reasoning_grace true for a think-style model whose prompt opens
 *                        `<think>`: the leading reasoning region is
 *                        unconstrained until "</think>".  false constrains from
 *                        the first token.
 * @return  a fresh state, or NULL on OOM / RFG_FORMAT_TEXT.
 *
 * Thread-safe: yes (no shared state)
 */
rfg_state *rfg_state_new(rfg_format format, bool reasoning_grace);

/** Free a walk state (NULL-safe). */
void rfg_state_free(rfg_state *st);

/**
 * End reasoning only on the "</think>" TOKEN: an rfg_advance whose whole text is
 * "</think>".  The same text spelled out of ordinary tokens stays reasoning
 * (think_close_rule.h).  Set it when the model's close is one token; leave it off
 * otherwise.  Call before the first rfg_advance.
 *
 * Thread-safe: no (mutates st)
 * Reentrant: yes
 * Signal-safe: no
 */
void rfg_set_think_close_token(rfg_state *st);

/** A new walk at the same position as `st` (NULL on OOM / NULL input). */
rfg_state *rfg_state_clone(const rfg_state *st);

/** Overwrite `dst` with `src`'s position (both non-NULL). */
void rfg_state_copy(rfg_state *dst, const rfg_state *src);

/**
 * Signal that the reasoning phase has ended — the JSON object is required from
 * here on.  Idempotent, and a no-op once reasoning has already ended.  The
 * grammar ALSO self-detects "</think>" in its own token stream, so a caller that
 * does not track reasoning externally still gets correct behaviour.
 */
void rfg_reasoning_end(rfg_state *st);

/**
 * Would appending `token_text` keep the walk on a valid prefix of the format?
 * Does NOT mutate `st` (walks a clone).  Empty/NULL text is always allowed (it
 * is a no-op).  While in the reasoning region every text is allowed.
 */
bool rfg_token_allowed(const rfg_state *st, const char *token_text);

/** Walk the state forward by `token_text` (must have been allowed). */
void rfg_advance(rfg_state *st, const char *token_text);

/**
 * Has the walk emitted a COMPLETE top-level JSON object?  The serve stops
 * decoding here — everything the format promises has been produced, and any
 * further token would make the content un-parseable.
 */
bool rfg_complete(const rfg_state *st);

/** Is the walk currently INSIDE the JSON object (past the opening '{')? */
bool rfg_in_json(const rfg_state *st);

/**
 * Should the sampler MASK this step?  False in the reasoning region (so the
 * think block is unconstrained), false once complete, and false if the walk went
 * DEAD (fail open rather than wedge a generation).  True otherwise.
 */
bool rfg_mask_active(const rfg_state *st);

/** Has the reasoning region ended (saw "</think>", or never had a grace)? */
bool rfg_reasoning_ended(const rfg_state *st);

/* DEBUG: current JSON nesting depth (0 before the object opens). */
int rfg_debug_depth(const rfg_state *st);

/* DEBUG: how many times rfg_advance has been called on this state.  Zero means
 * the sampler never fed the grammar at all — which distinguishes "the constraint
 * was wired but never engaged" from "the constraint was never wired", the two
 * failure modes that look identical from the response body. */
int rfg_debug_advances(const rfg_state *st);

/* DEBUG: the token text that took the walk to DEAD, and the phase it was in
 * when that happened ("" if it never died).  A DEAD walk means the sampler
 * emitted a token the grammar rejects, which cannot happen while the mask is
 * working — so these name the defect directly. */
const char *rfg_debug_death_piece(const rfg_state *st);
const char *rfg_debug_death_phase(const rfg_state *st);

/* DEBUG: short name of the current phase ("reason", "pre", "value", "done",
 * "dead", …).  Static string, never NULL — the serve logs it at end of turn so a
 * completion that did not end in `done` says WHY on the fleet log. */
const char *rfg_debug_phase_name(const rfg_state *st);

#endif /* PN_RESPONSE_FORMAT_GRAMMAR_H */
