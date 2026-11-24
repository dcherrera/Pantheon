/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file field_pattern.h
 * @brief A small byte-level regex, compiled to a bounded NFA and stepped one
 *        byte at a time over a fixed bitset — the matcher behind the request
 *        field `field_patterns` (a pattern rule on one tool field), shared by
 *        both tool-call grammars (tool_grammar.c, tool_grammar_json.c).
 *
 * WHOLE-VALUE MATCH.  A pattern matches the WHOLE value: the anchors are
 * implicit.  A leading `^` and a trailing `$` are accepted and ignored; an
 * anchor anywhere else is a compile error.
 *
 * SYNTAX (matched on BYTES, not characters):
 *   literal        any byte that is not a metachar; non-ASCII bytes are literals
 *   .              any byte except '\n'
 *   [...] [^...]   a byte class: single bytes, ranges `a-z`, the escapes below
 *                  (`\d \s \w \D \S \W` add their set); a `]` right after `[` or
 *                  `[^` and a `-` first or last are literal.  A class that
 *                  matches no byte is an error.
 *   \d \D          [0-9] and its complement
 *   \s \S          [ \t\n\r\f\v] and its complement
 *   \w \W          [A-Za-z0-9_] and its complement
 *   \n \t \r \f \v the control byte
 *   \<punct>       the punctuation byte itself (`\.` `\\` `\[` `\-` ...)
 *   ( )            a group (no captures)
 *   |              alternation
 *   * + ?          zero-or-more, one-or-more, optional
 *   {m} {m,} {m,n} counted repetition, 0 <= m <= n <= FP_MAX_REPEAT
 * Not supported (compile error): lazy or stacked quantifiers, backreferences,
 * lookaround, \b, \x / \u escapes, flags.  A multi-byte UTF-8 character inside a
 * positive class adds its bytes one by one; negated classes and `.` pass every
 * byte >= 0x80, so UTF-8 text goes through them.
 *
 * THE AUTOMATON.  The pattern compiles to its position (Glushkov) automaton —
 * Thompson's NFA with the epsilon moves folded away, one state per byte-class
 * atom plus a start state.  A walk is a set of states (fp_set, a fixed bitset),
 * stepped per byte with one table lookup per live state.  Every state can still
 * reach acceptance (classes are never empty), so a NON-EMPTY set means "this
 * prefix can still be completed to a match"; an empty one means it cannot.
 * Counted repetition is expanded, so `x{3,5}` costs five copies of `x`.
 *
 * LIMITS: at most FP_MAX_STATES states (atoms + start, after expansion), a
 * pattern of at most FP_MAX_PATTERN_LEN bytes, and repeat counts of at most
 * FP_MAX_REPEAT.  Past any of them the pattern does not compile (the serve then
 * answers HTTP 400).  128 states keep fp_set at 16 bytes, which matters because
 * the grammars copy their whole walk state for every candidate token; the
 * pattern the client sends today needs 22.
 *
 * Pure C, libc-only.  Unit-tested in tests/test_field_pattern.c.
 *
 * @status Used by tool_grammar.c / tool_grammar_json.c (`field_patterns`).
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_FIELD_PATTERN_H
#define PANTHEON_FIELD_PATTERN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FP_MAX_STATES      128
#define FP_SET_WORDS       (FP_MAX_STATES / 64)
#define FP_MAX_PATTERN_LEN 1024
#define FP_MAX_REPEAT      100

/* A set of automaton states.  A plain POD: grammar walk states embed it by value,
 * so clone/copy of a walk carries it. */
typedef struct {
    uint64_t w[FP_SET_WORDS];
} fp_set;

/* A compiled pattern (immutable once compiled; shared by every walk). */
typedef struct field_pattern field_pattern;

/**
 * Compile `src` (NUL-terminated).  On success returns the pattern (free with
 * fp_free).  On failure returns NULL and writes a one-line reason, naming the
 * byte offset where it applies, into `err` (when `err_cap` > 0); the reason for
 * an allocation failure is "out of memory".
 */
field_pattern *fp_compile(const char *src, char *err, size_t err_cap);

/** Free a compiled pattern (NULL-safe). */
void fp_free(field_pattern *p);

/** The state set before any byte (the empty prefix). */
fp_set fp_start(const field_pattern *p);

/** Step `s` over one byte.  Returns false when the set became empty (no
 *  completion of this prefix can match). */
bool fp_step(const field_pattern *p, fp_set *s, unsigned char byte);

/** Does the prefix that led to `s` match the whole pattern? */
bool fp_accepting(const field_pattern *p, const fp_set *s);

/** Is `s` empty (a dead prefix)? */
bool fp_set_empty(const fp_set *s);

/** Number of states the compiled pattern uses (start included). */
int fp_state_count(const field_pattern *p);

/**
 * The bytes every SHORTEST completion of the prefix that led to `s` begins
 * with: at each step the one byte that brings the walk a byte closer to a
 * match, until no byte or more than one does, or the walk accepts.  For
 * `[^\n]+ from: [^\s,:]+:[0-9]+` after "a fact" that is " from: " (the
 * path class next offers many bytes).  Writes them NUL-terminated into
 * `out[0..cap)` (at most cap-1 bytes) and returns how many; 0 when none is
 * forced, `s` is empty or already accepting.  The steer behind field_patterns
 * (tg_steer / tgj_steer) forces exactly these bytes.
 */
int fp_forced_continuation(const field_pattern *p, const fp_set *s, char *out, int cap);

/** Does `text[0..len)` match the whole pattern? */
bool fp_full_match(const field_pattern *p, const char *text, size_t len);

#endif /* PANTHEON_FIELD_PATTERN_H */
