/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Incremental JSON-value well-formedness walk.  See json_value_grammar.h for
 * the contract, the size argument, and the named duplication with
 * response_format_grammar.c.
 *
 * Pure C, libc-only.  No engine/CUDA deps → unit-tested standalone.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * The one non-obvious mechanic: RECONSUME
 * ────────────────────────────────────────────────────────────────────────────
 * Every JSON token is self-delimiting except numbers and the bare literals: `1`
 * is finished only by the char AFTER it, and that char is itself structural
 * ("1,", "1]", "1 ").  So step_char reports `reconsume` when it ended a number
 * or a literal without consuming the char, and the caller re-feeds it into the
 * new phase.  It can happen at most once per char — the reconsumed char is
 * always structural and structural phases never reconsume — which is what the
 * guard in jvg_step asserts rather than assumes.
 */

#include "json_value_grammar.h"

#include <string.h>

/* ── small helpers ── */

static bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

static bool is_hex(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static void die(jvg_walk *w) {
    w->phase = JVG_P_DEAD;
}

/* The literal a `lit_which` names, so the walk can match it char by char
 * without carrying the string itself in the (copied-per-candidate) state. */
static const char *literal_text(unsigned char which) {
    switch (which) {
    case 1:  return "true";
    case 2:  return "false";
    case 3:  return "null";
    default: return "";
    }
}

/* Is the innermost open container an array?  Undefined at depth 0 — every
 * caller checks depth first. */
static bool innermost_is_array(const jvg_walk *w) {
    return (w->container_is_array >> (unsigned)(w->depth - 1)) & 1u;
}

/* Open a container and sit the walk in its FIRST slot.  The first slot is the
 * only one where the matching close is legal, which is what makes `{}` / `[]`
 * valid while `{"a":1,}` / `[1,]` are structurally unreachable. */
static void open_container(jvg_walk *w, bool is_array) {
    if (w->depth >= JVG_MAX_DEPTH) { die(w); return; }
    if (is_array) w->container_is_array |=  ((uint64_t)1 << w->depth);
    else          w->container_is_array &= ~((uint64_t)1 << w->depth);
    w->depth++;
    w->phase = is_array ? JVG_P_FIRST_ELEM : JVG_P_FIRST_KEY;
}

/* Close the innermost container.  Closing the LAST one completes the walk. */
static void close_container(jvg_walk *w) {
    if (w->depth == 0) { die(w); return; }
    w->depth--;
    w->phase = (w->depth == 0) ? JVG_P_COMPLETE : JVG_P_AFTER;
}

/* A value just finished: the innermost container decides what may follow, and
 * at depth 0 the whole walk is done. */
static void value_finished(jvg_walk *w) {
    w->phase = (w->depth == 0) ? JVG_P_COMPLETE : JVG_P_AFTER;
}

/* Consume the first char of a value.  Kills the walk when `c` cannot begin one. */
static void begin_value(jvg_walk *w, char c) {
    if (c == '{') { open_container(w, false); return; }
    if (c == '[') { open_container(w, true);  return; }
    if (c == '"') { w->phase = JVG_P_STRING; w->str_is_key = false; return; }
    if (c == '-' || is_digit(c)) {
        w->phase       = JVG_P_NUMBER;
        w->num_can_end = is_digit(c);
        return;
    }
    if (c == 't' || c == 'f' || c == 'n') {
        w->lit_which = (c == 't') ? 1 : (c == 'f') ? 2 : 3;
        w->lit_pos   = 1;
        w->phase     = JVG_P_LITERAL;
        return;
    }
    die(w);
}

/* ── the char walk ── */

static void step_char(jvg_walk *w, char c, bool *reconsume) {
    *reconsume = false;

    switch (w->phase) {

    case JVG_P_START:
        if (is_ws(c)) return;
        /* A schema-typed parameter names the container its value must be, so
         * anything else — a bare string, a number, the prose the model writes
         * when it forgets the format — is off-grammar at the very first char
         * rather than at the tool. */
        if (w->required_container && c != (char)w->required_container) {
            die(w);
            return;
        }
        begin_value(w, c);
        return;

    case JVG_P_VALUE_REQ:
        if (is_ws(c)) return;
        begin_value(w, c);
        return;

    case JVG_P_FIRST_ELEM:
        if (is_ws(c)) return;
        if (c == ']' && w->depth > 0 && innermost_is_array(w)) {
            close_container(w);   /* "[]" */
            return;
        }
        begin_value(w, c);
        return;

    case JVG_P_FIRST_KEY:
        if (is_ws(c)) return;
        if (c == '"') { w->phase = JVG_P_STRING; w->str_is_key = true; return; }
        if (c == '}' && w->depth > 0 && !innermost_is_array(w)) {
            close_container(w);   /* "{}" */
            return;
        }
        die(w);
        return;

    case JVG_P_KEY_REQ:
        /* After a ',' inside an object the next thing MUST be a key.  This is
         * the state that makes the live failure unreachable: the model wrote
         * `{"content":"",""}` — a second, bare STRING where a key/value pair
         * was owed.  The string parses as a key, and then the ':' this phase's
         * successor requires never arrives, so '}' dies. */
        if (is_ws(c)) return;
        if (c == '"') { w->phase = JVG_P_STRING; w->str_is_key = true; return; }
        die(w);   /* a trailing comma cannot be followed by '}' either */
        return;

    case JVG_P_COLON:
        if (is_ws(c)) return;
        if (c == ':') { w->phase = JVG_P_VALUE_REQ; return; }
        die(w);
        return;

    case JVG_P_STRING:
        if (c == '\\') { w->phase = JVG_P_STR_ESC; return; }
        if (c == '"') {
            if (w->str_is_key) w->phase = JVG_P_COLON;
            else               value_finished(w);
            return;
        }
        if ((unsigned char)c < 0x20) { die(w); return; }  /* raw control char */
        return;

    case JVG_P_STR_ESC:
        if (c == 'u') { w->phase = JVG_P_STR_UNI; w->uni_left = 4; return; }
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
            c == 'n' || c == 'r' || c == 't') {
            w->phase = JVG_P_STRING;
            return;
        }
        die(w);
        return;

    case JVG_P_STR_UNI:
        if (!is_hex(c)) { die(w); return; }
        if (--w->uni_left == 0) w->phase = JVG_P_STRING;
        return;

    case JVG_P_NUMBER:
        if (is_digit(c)) { w->num_can_end = true; return; }
        if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
            w->num_can_end = false;
            return;
        }
        /* Anything else ends the number — but a bare "-" or a trailing
         * '.'/'e'/'+'/'-' is not a number, so it may only end on a digit. */
        if (!w->num_can_end) { die(w); return; }
        value_finished(w);
        *reconsume = true;
        return;

    case JVG_P_LITERAL: {
        const char *lit = literal_text(w->lit_which);
        if (lit[w->lit_pos] != c) { die(w); return; }
        w->lit_pos++;
        if (lit[w->lit_pos] == '\0') value_finished(w);
        return;
    }

    case JVG_P_AFTER:
        if (is_ws(c)) return;
        if (w->depth == 0) { die(w); return; }
        if (c == ',') {
            w->phase = innermost_is_array(w) ? JVG_P_VALUE_REQ : JVG_P_KEY_REQ;
            return;
        }
        if (c == ']' &&  innermost_is_array(w)) { close_container(w); return; }
        if (c == '}' && !innermost_is_array(w)) { close_container(w); return; }
        die(w);
        return;

    case JVG_P_COMPLETE:
        /* Trailing whitespace is padding, not a second value.  Tolerating it
         * here is what lets a scalar at depth 0 be ENDED by the space after it
         * ("3 ") rather than dying on its own terminator. */
        if (is_ws(c)) return;
        die(w);
        return;

    case JVG_P_DEAD:
    default:
        die(w);
        return;
    }
}

/* ── public API ── */

void jvg_begin(jvg_walk *w, char required_container) {
    if (!w) return;
    memset(w, 0, sizeof(*w));
    w->phase = JVG_P_START;
    w->required_container =
        (required_container == '[' || required_container == '{')
            ? (unsigned char)required_container : 0;
}

void jvg_step(jvg_walk *w, char c) {
    if (!w || w->phase == JVG_P_DEAD) return;
    bool reconsume = true;
    /* At most two passes — see the RECONSUME note at the top.  The guard is the
     * assertion, not an allowance: a third pass would be a bug in step_char. */
    for (int pass = 0; reconsume && w->phase != JVG_P_DEAD && pass < 2; pass++)
        step_char(w, c, &reconsume);
}

bool jvg_is_dead(const jvg_walk *w) {
    return w && w->phase == JVG_P_DEAD;
}

bool jvg_is_complete(const jvg_walk *w) {
    return w && w->phase == JVG_P_COMPLETE;
}

int jvg_depth(const jvg_walk *w) {
    return w ? (int)w->depth : 0;
}

const char *jvg_phase_name(const jvg_walk *w) {
    if (!w) return "none";
    switch (w->phase) {
    case JVG_P_START:      return "start";
    case JVG_P_VALUE_REQ:  return "value_req";
    case JVG_P_FIRST_ELEM: return "first_elem";
    case JVG_P_FIRST_KEY:  return "first_key";
    case JVG_P_KEY_REQ:    return "key_req";
    case JVG_P_COLON:      return "colon";
    case JVG_P_STRING:     return "string";
    case JVG_P_STR_ESC:    return "str_esc";
    case JVG_P_STR_UNI:    return "str_uni";
    case JVG_P_NUMBER:     return "number";
    case JVG_P_LITERAL:    return "literal";
    case JVG_P_AFTER:      return "after";
    case JVG_P_COMPLETE:   return "complete";
    case JVG_P_DEAD:       return "dead";
    default:               return "?";
    }
}

bool jvg_text_is_json_value(const char *text, char required_container) {
    if (!text) return false;
    jvg_walk w;
    jvg_begin(&w, required_container);
    for (const char *p = text; *p; p++) {
        jvg_step(&w, *p);
        if (w.phase == JVG_P_DEAD) return false;
        /* Past the closing brace/bracket only whitespace may follow — a second
         * value, or any trailing junk, is not "one well-formed value". */
        if (w.phase == JVG_P_COMPLETE) {
            for (const char *q = p + 1; *q; q++)
                if (!is_ws(*q)) return false;
            return true;
        }
    }
    /* END OF TEXT.  A top-level NUMBER is the one value with no terminator of
     * its own — "3" is finished only by the char after it, and here there is no
     * char after it.  Running out of text IS that boundary, so settle it.  Every
     * other phase means the value is genuinely unfinished, and a number still
     * inside a container is unfinished too (its ']' never arrived). */
    return w.phase == JVG_P_NUMBER && w.num_can_end && w.depth == 0;
}
