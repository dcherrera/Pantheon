/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_value_grammar.h
 * @brief Char-level, incremental "is this text still a valid JSON VALUE prefix?"
 *        walk, small enough to embed BY VALUE inside another grammar's state.
 *
 * WHAT THIS IS FOR.  The tool grammars constrain a tool call's FRAME — the
 * `<function=NAME>` / `<parameter=KEY>` skeleton, the NAME and KEY sets, the
 * `required` mask, a string `enum`.  Inside `<parameter=KEY>…</parameter>` the
 * bytes were free text, so a parameter whose schema says `"type":"array"` or
 * `"type":"object"` — whose value the parser SPLICES RAW into the call's
 * `arguments` JSON (tool_calling.c write_typed_value) — was completely
 * unconstrained.  A model that writes one malformed nested object there does not
 * merely get that argument wrong: the raw splice makes the WHOLE `arguments`
 * string un-parseable, so every other argument of the call is lost with it.
 * Live, full-weight qwen36-bf16, grammar armed and masking the frame correctly:
 * three consecutive `todowrite` calls died on
 *
 *     [{"content":"…","status":"in_progress"},{"content":"",""}]
 *                                                            ^ a bare value,
 *                                                              no key
 *
 * This walk is what makes those bytes off-grammar.
 *
 * WHY A SEPARATE, EMBEDDABLE MODULE.  The constrained sampler asks
 * "does appending this token's text keep a valid prefix?" once per vocabulary
 * entry per step (~248k times), and it answers by COPYING the walk state onto
 * the stack and walking the candidate on the copy.  So the state must be
 * (a) a POD with no pointers to own and (b) small.  This one is: the container
 * nesting stack is ONE BIT per open level in a `uint64_t` rather than a byte
 * array, which is the whole reason it costs 16 bytes instead of 80 and can be
 * embedded in `tg_state` without measurably changing the per-step mask cost.
 *
 * NOT A PARSER.  It answers only "could this still become a valid JSON value",
 * never "what value is it" — there is no tree, no allocation, and it never sees
 * the whole text at once.  Use json_reader.h (`jr_parse`) when the bytes are all
 * in hand and the answer needs to be a value; use this when the bytes arrive one
 * token at a time and the answer has to come BEFORE the next token is drawn.
 *
 * WHAT IS VALIDATED.  Objects, arrays, strings (with `\\` escapes, `\\uXXXX`, and
 * raw control characters rejected), numbers (at least one digit; a number may not
 * end on `.`, `e`, `+` or `-`), the literals true/false/null, and whitespace
 * between tokens.  Trailing commas are structurally unreachable, and so is a
 * bare value in an object slot — the two shapes the live model actually emitted.
 * Nesting deeper than JVG_MAX_DEPTH is simply not an allowed continuation.
 *
 * WHAT IS NOT.  Nothing about the SCHEMA: `items`, nested `properties`,
 * `required`, per-member `type` and `enum` are all still dropped.  A well-formed
 * `[{"wrong_key":1}]` is accepted here and rejected later by the tool.  See the
 * spec note in tool_grammar.h for why well-formedness was taken first.
 *
 * @note DUPLICATION, NAMED.  response_format_grammar.c carries its own copy of
 *       this FSM (it predates this file) with a top-level-object rule and a
 *       reasoning grace wrapped around it.  This module was written to be that
 *       core — same phases, same `reconsume` protocol — so the migration is
 *       mechanical, but it is NOT done here: `response_format` is a live feature
 *       and refactoring it inside a tool-grammar fix would widen the blast
 *       radius for no functional gain.  Follow-up, not yet filed: migrate
 *       response_format_grammar.c onto this walk and delete its copy.
 *
 * Pure C, libc-only — no engine/CUDA deps, unit-tested standalone
 * (tests/test_json_value_grammar.c).
 */

#ifndef PN_JSON_VALUE_GRAMMAR_H
#define PN_JSON_VALUE_GRAMMAR_H

#include <stdbool.h>
#include <stdint.h>

/* Deepest container nesting the walk will allow to be opened.  One BIT of
 * `container_is_array` per level, so this is exactly the width of that word. */
#define JVG_MAX_DEPTH 64

/* Walk phases.  Exposed only so `jvg_walk` can be a POD the embedder copies by
 * value — treat the enum as opaque and use the predicates below. */
typedef enum {
    JVG_P_START = 0,   /* whitespace, then the value begins */
    JVG_P_VALUE_REQ,   /* a value is REQUIRED here (after ':' or ',') */
    JVG_P_FIRST_ELEM,  /* an array's first slot: a value, or ']' ("[]") */
    JVG_P_FIRST_KEY,   /* an object's first slot: a key, or '}' ("{}") */
    JVG_P_KEY_REQ,     /* a key is REQUIRED here (after ',') */
    JVG_P_COLON,       /* expect ':' after a key (whitespace ok) */
    JVG_P_STRING,      /* inside a "…" */
    JVG_P_STR_ESC,     /* the previous char was a backslash */
    JVG_P_STR_UNI,     /* consuming the 4 hex digits of a \uXXXX escape */
    JVG_P_NUMBER,      /* inside a numeric literal */
    JVG_P_LITERAL,     /* inside true / false / null */
    JVG_P_AFTER,       /* after a value: ',' or this container's close */
    JVG_P_COMPLETE,    /* the TOP-LEVEL value closed — nothing more is legal */
    JVG_P_DEAD,        /* off-grammar */
} jvg_phase;

/**
 * A walk position.  A POD by construction: no pointers, no owned memory, safe to
 * copy with `=` and to throw the copy away (which is exactly what the mask's
 * per-candidate check does).  16 bytes.
 */
typedef struct {
    uint64_t      container_is_array;  /* bit i: level i is an array, else object */
    unsigned char phase;               /* jvg_phase */
    unsigned char depth;               /* open containers (0 before the first) */
    unsigned char uni_left;            /* \uXXXX hex digits still required */
    unsigned char lit_pos;             /* chars of true/false/null matched */
    unsigned char lit_which;           /* 0 none, 1 true, 2 false, 3 null */
    bool          str_is_key;          /* the string being read is an object key */
    bool          num_can_end;         /* the number so far may legally end here */
    unsigned char required_container;  /* '[' , '{' , or 0 = any JSON value */
} jvg_walk;

/**
 * Start a fresh walk.
 *
 * @param required_container  '[' to require the value be an ARRAY, '{' to
 *                            require an OBJECT, 0 to accept any JSON value.
 *                            The schema-typed callers pass the container their
 *                            `"type"` names, which is what lets the consumer
 *                            splice the bytes in raw and know they will parse.
 *                            Leading whitespace is allowed either way.
 */
void jvg_begin(jvg_walk *w, char required_container);

/** Consume one char, moving the walk (to JVG_P_DEAD on any violation). */
void jvg_step(jvg_walk *w, char c);

/** Has the walk left the grammar?  Nothing can revive it. */
bool jvg_is_dead(const jvg_walk *w);

/**
 * Has a COMPLETE top-level JSON value been emitted?  True the instant the value
 * closes; from that point every further char is off-grammar, so the embedder
 * takes over (a tool grammar goes looking for its `</parameter>`).
 */
bool jvg_is_complete(const jvg_walk *w);

/** Open container nesting depth (0 before the first '{'/'[' and after the last). */
int jvg_depth(const jvg_walk *w);

/** Phase name, for diagnostics only ("string", "after", "dead", …). */
const char *jvg_phase_name(const jvg_walk *w);

/**
 * Whole-text convenience: is `text` (NUL-terminated) exactly one well-formed
 * JSON value, with nothing but whitespace after it?
 *
 * For callers that hold all the bytes and want the same verdict the incremental
 * walk would give — so the decoding constraint and any post-parse re-check
 * cannot disagree about what "well-formed" means.
 */
bool jvg_text_is_json_value(const char *text, char required_container);

#endif /* PN_JSON_VALUE_GRAMMAR_H */
