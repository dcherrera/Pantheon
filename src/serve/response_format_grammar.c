/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Response-format decoding grammar — a char-level FSM that constrains a whole
 * completion to one syntactically valid JSON object.  See
 * response_format_grammar.h for the contract; this is the answer-channel analog
 * of tool_grammar.c / tool_grammar_json.c, which constrain a TOOL CALL.
 *
 * Pure C, libc-only.  No engine/CUDA deps → unit-tested standalone.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * The constrained emission
 * ────────────────────────────────────────────────────────────────────────────
 *   (reasoning, unconstrained)</think>   ← only when reasoning_grace is set
 *   (whitespace) { …a balanced, valid JSON object… }
 *
 * Everything after the closing brace is out of grammar: the walk reports
 * COMPLETE, the serve stops decoding, and the content the client receives parses.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Why a pushdown FSM and not a JSON parser
 * ────────────────────────────────────────────────────────────────────────────
 * The sampler asks "would this candidate token keep the output on a valid JSON
 * prefix?" ~150k times per constrained step, so the check must be (a) cheap and
 * (b) non-destructive.  A by-value POD state satisfies both: the allowed-check
 * copies the struct onto the stack, walks the candidate's chars on the copy, and
 * throws it away.  That rules out a heap parse tree, so container nesting is a
 * fixed-size stack (RFG_MAX_DEPTH) of one byte per open container.
 */

#include "response_format_grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "think_close_rule.h"

/* Longest literal the walk ever matches char-by-char ("false"). */
#define RFG_LIT_MAX 8

/* Phases of the walk (char-consuming). */
typedef enum {
    RFG_P_REASON = 0,   /* unconstrained reasoning; watching for "</think>" */
    RFG_P_PRE,          /* whitespace, then the opening '{' is REQUIRED */
    RFG_P_VALUE,        /* an array's FIRST element slot: a value, or ']' ("[]") */
    RFG_P_VALUE_REQ,    /* a value is REQUIRED here (after ':' or a ',') */
    RFG_P_STRING,       /* inside a "…" */
    RFG_P_STR_ESC,      /* the previous char was a backslash */
    RFG_P_STR_UNI,      /* consuming the 4 hex digits of a \uXXXX escape */
    RFG_P_NUMBER,       /* inside a numeric literal */
    RFG_P_LITERAL,      /* inside true / false / null */
    RFG_P_KEY,          /* an object's FIRST key slot: a key, or '}' ("{}") */
    RFG_P_KEY_REQ,      /* a key is REQUIRED here (after a ',') */
    RFG_P_COLON,        /* expect ':' after a key (whitespace ok) */
    RFG_P_AFTER,        /* after a value: expect ',' or this container's close */
    RFG_P_DONE,         /* the top-level object closed — nothing more is legal */
    RFG_P_DEAD,         /* off-grammar (only reachable via a fail-open advance) */
} rfg_phase;

/* One open container on the nesting stack. */
#define RFG_C_OBJECT 0
#define RFG_C_ARRAY  1

struct rfg_state {
    rfg_phase phase;
    bool      reasoning_ended;
    int       think_term;              /* rolling prefix of "</think>" matched */
    bool      think_close_token;       /* only the "</think>" TOKEN ends reasoning */

    unsigned char stack[RFG_MAX_DEPTH]; /* RFG_C_OBJECT / RFG_C_ARRAY per level */
    int           depth;                /* open containers (0 before '{') */

    bool str_is_key;                    /* the string being read is an object key */
    int  uni_left;                      /* \uXXXX hex digits still required */

    /* Number sub-walk: JSON forbids a bare "-" or a trailing 'e'/'.'/'+'/'-',
     * so a number may only END on a digit. */
    bool num_can_end;

    /* Literal sub-walk (true / false / null). */
    char lit[RFG_LIT_MAX];
    int  lit_pos;

    /* Diagnostics only (never affects the walk). */
    int  advances;
    char death_piece[48];   /* the token text that took the walk to DEAD */
    char death_phase[16];   /* the phase it was in when that happened */
};

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

static void die(rfg_state *st) {
    st->phase = RFG_P_DEAD;
}

/* Push a container; DEAD when the nesting cap is reached (so a token that would
 * exceed RFG_MAX_DEPTH is simply not an allowed continuation). */
static void push_container(rfg_state *st, unsigned char kind) {
    if (st->depth >= RFG_MAX_DEPTH) { die(st); return; }
    st->stack[st->depth++] = kind;
}

/* Close the innermost container.  Closing the LAST one completes the walk. */
static void pop_container(rfg_state *st) {
    if (st->depth <= 0) { die(st); return; }
    st->depth--;
    st->phase = (st->depth == 0) ? RFG_P_DONE : RFG_P_AFTER;
}

/* Begin a container and position the walk at its first element/key slot.  The
 * FIRST slot is the only one where the matching close is legal — that is what
 * makes `{}` / `[]` valid and `{"a":1,}` / `[1,]` structurally impossible. */
static void open_container(rfg_state *st, unsigned char kind) {
    push_container(st, kind);
    if (st->phase == RFG_P_DEAD) return;
    st->phase = (kind == RFG_C_OBJECT) ? RFG_P_KEY : RFG_P_VALUE;
}

/* Consume the first char of a value.  Returns false (and kills the walk) when
 * `c` cannot begin any JSON value. */
static bool begin_value_char(rfg_state *st, char c) {
    if (c == '{') { open_container(st, RFG_C_OBJECT); return true; }
    if (c == '[') { open_container(st, RFG_C_ARRAY);  return true; }
    if (c == '"') { st->phase = RFG_P_STRING; st->str_is_key = false; return true; }
    if (c == '-' || is_digit(c)) {
        st->phase       = RFG_P_NUMBER;
        st->num_can_end = is_digit(c);
        return true;
    }
    if (c == 't' || c == 'f' || c == 'n') {
        const char *lit = (c == 't') ? "true" : (c == 'f') ? "false" : "null";
        memcpy(st->lit, lit, strlen(lit) + 1);
        st->lit_pos = 1;
        st->phase   = RFG_P_LITERAL;
        return true;
    }
    die(st);
    return false;
}

/* A value just finished: an array element or object member is complete, so the
 * innermost container decides what may follow. */
static void value_finished(rfg_state *st) {
    st->phase = (st->depth == 0) ? RFG_P_DONE : RFG_P_AFTER;
}

/* ── the char walk ──
 *
 * `reconsume` is set when the char that ENDED a variable-length token (a number
 * or a literal) is itself the next structural char and must be re-fed. */
static void step_char(rfg_state *st, char c, bool *reconsume);

static void step_reason(rfg_state *st, char c) {
    /* Self-detect "</think>" in the raw stream (the serve emitter's think flag
     * is not reliable on every model — see the tool grammars' BUG 1 note). */
    if (think_close_text_step(&st->think_term, c, st->think_close_token)) {
        st->reasoning_ended = true;
        st->phase           = RFG_P_PRE;
    }
}

static void step_char(rfg_state *st, char c, bool *reconsume) {
    *reconsume = false;

    switch (st->phase) {

    case RFG_P_REASON:
        step_reason(st, c);
        return;

    case RFG_P_PRE:
        if (is_ws(c)) return;
        if (c == '{') { open_container(st, RFG_C_OBJECT); return; }
        die(st);
        return;

    case RFG_P_VALUE:
        if (is_ws(c)) return;
        /* An array's first-element slot doubles as its close: `[]` is valid. */
        if (c == ']' && st->depth > 0 &&
            st->stack[st->depth - 1] == RFG_C_ARRAY) {
            pop_container(st);
            return;
        }
        begin_value_char(st, c);
        return;

    case RFG_P_VALUE_REQ:
        if (is_ws(c)) return;
        begin_value_char(st, c);
        return;

    case RFG_P_STRING:
        if (c == '\\') { st->phase = RFG_P_STR_ESC; return; }
        if (c == '"') {
            if (st->str_is_key) st->phase = RFG_P_COLON;
            else                value_finished(st);
            return;
        }
        /* Raw control characters are illegal inside a JSON string. */
        if ((unsigned char)c < 0x20) { die(st); return; }
        return;

    case RFG_P_STR_ESC:
        if (c == 'u') { st->phase = RFG_P_STR_UNI; st->uni_left = 4; return; }
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
            c == 'n' || c == 'r' || c == 't') {
            st->phase = RFG_P_STRING;
            return;
        }
        die(st);
        return;

    case RFG_P_STR_UNI:
        if (!is_hex(c)) { die(st); return; }
        if (--st->uni_left == 0) st->phase = RFG_P_STRING;
        return;

    case RFG_P_NUMBER:
        if (is_digit(c)) { st->num_can_end = true; return; }
        if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
            st->num_can_end = false;
            return;
        }
        /* Anything else ends the number — but only if it was well-formed. */
        if (!st->num_can_end) { die(st); return; }
        value_finished(st);
        *reconsume = true;
        return;

    case RFG_P_LITERAL:
        if (st->lit[st->lit_pos] != c) { die(st); return; }
        st->lit_pos++;
        if (st->lit[st->lit_pos] == '\0') value_finished(st);
        return;

    case RFG_P_KEY:
        if (is_ws(c)) return;
        if (c == '"') { st->phase = RFG_P_STRING; st->str_is_key = true; return; }
        /* An object's first-key slot doubles as its close: `{}` is valid. */
        if (c == '}' && st->depth > 0 &&
            st->stack[st->depth - 1] == RFG_C_OBJECT) {
            pop_container(st);
            return;
        }
        die(st);
        return;

    case RFG_P_KEY_REQ:
        if (is_ws(c)) return;
        if (c == '"') { st->phase = RFG_P_STRING; st->str_is_key = true; return; }
        die(st);   /* a trailing comma cannot be followed by '}' */
        return;

    case RFG_P_COLON:
        if (is_ws(c)) return;
        if (c == ':') { st->phase = RFG_P_VALUE_REQ; return; }
        die(st);
        return;

    case RFG_P_AFTER:
        if (is_ws(c)) return;
        if (st->depth <= 0) { die(st); return; }
        if (c == ',') {
            /* A trailing comma is illegal, so the slot after it REQUIRES a key
             * (object) or an element (array) — never the container's close. */
            st->phase = (st->stack[st->depth - 1] == RFG_C_OBJECT)
                            ? RFG_P_KEY_REQ : RFG_P_VALUE_REQ;
            return;
        }
        if (c == '}' && st->stack[st->depth - 1] == RFG_C_OBJECT) {
            pop_container(st);
            return;
        }
        if (c == ']' && st->stack[st->depth - 1] == RFG_C_ARRAY) {
            pop_container(st);
            return;
        }
        die(st);
        return;

    case RFG_P_DONE:
    case RFG_P_DEAD:
    default:
        die(st);
        return;
    }
}

/* Walk `text` char by char, honouring the reconsume signal. */
static void walk(rfg_state *st, const char *text) {
    if (st->phase == RFG_P_REASON && think_close_token_piece(text, st->think_close_token)) {
        st->reasoning_ended = true;
        st->phase           = RFG_P_PRE;
        return;
    }
    for (const char *p = text; *p; p++) {
        bool reconsume = true;
        /* At most two passes: a number/literal end reconsumes exactly once, and
         * the reconsumed char is always structural (it cannot reconsume again). */
        int guard = 0;
        while (reconsume && st->phase != RFG_P_DEAD && guard++ < 2)
            step_char(st, *p, &reconsume);
        if (st->phase == RFG_P_DEAD) return;
    }
}

/* ── public API ── */

rfg_state *rfg_state_new(rfg_format format, bool reasoning_grace) {
    if (format == RFG_FORMAT_TEXT) return NULL;
    rfg_state *st = calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->phase           = reasoning_grace ? RFG_P_REASON : RFG_P_PRE;
    st->reasoning_ended = !reasoning_grace;
    return st;
}

void rfg_state_free(rfg_state *st) { free(st); }

rfg_state *rfg_state_clone(const rfg_state *st) {
    if (!st) return NULL;
    rfg_state *copy = malloc(sizeof(*copy));
    if (copy) *copy = *st;
    return copy;
}

void rfg_state_copy(rfg_state *dst, const rfg_state *src) { *dst = *src; }

void rfg_set_think_close_token(rfg_state *st) {
    if (st) st->think_close_token = true;
}

void rfg_reasoning_end(rfg_state *st) {
    if (!st || st->reasoning_ended) return;
    st->reasoning_ended = true;
    if (st->phase == RFG_P_REASON) {
        st->phase      = RFG_P_PRE;
        st->think_term = 0;
    }
}

bool rfg_token_allowed(const rfg_state *st, const char *token_text) {
    if (!st || !token_text || !*token_text) return true;
    if (st->phase == RFG_P_REASON) return true;   /* unconstrained grace */
    rfg_state clone = *st;                        /* POD copy — see the header */
    walk(&clone, token_text);
    return clone.phase != RFG_P_DEAD;
}

void rfg_advance(rfg_state *st, const char *token_text) {
    if (!st || !token_text || !*token_text) return;
    st->advances++;
    /* Record what killed the walk.  A DEAD walk means the sampler emitted a
     * token the grammar rejects — which should be impossible while the mask is
     * active, so the offending piece names the real defect. */
    rfg_phase before = st->phase;
    walk(st, token_text);
    if (st->phase == RFG_P_DEAD && before != RFG_P_DEAD &&
        st->death_piece[0] == '\0') {
        size_t n = strlen(token_text);
        if (n > sizeof(st->death_piece) - 1) n = sizeof(st->death_piece) - 1;
        memcpy(st->death_piece, token_text, n);
        st->death_piece[n] = '\0';
        const char *pn = rfg_debug_phase_name(st);   /* now DEAD; want `before` */
        (void)pn;
        rfg_state probe = *st;
        probe.phase = before;
        snprintf(st->death_phase, sizeof(st->death_phase), "%s",
                 rfg_debug_phase_name(&probe));
    }
}

bool rfg_complete(const rfg_state *st) {
    return st && st->phase == RFG_P_DONE;
}

bool rfg_in_json(const rfg_state *st) {
    if (!st) return false;
    return st->phase != RFG_P_REASON && st->phase != RFG_P_PRE &&
           st->phase != RFG_P_DONE   && st->phase != RFG_P_DEAD;
}

bool rfg_mask_active(const rfg_state *st) {
    if (!st) return false;
    return st->phase != RFG_P_REASON && st->phase != RFG_P_DONE &&
           st->phase != RFG_P_DEAD;
}

bool rfg_reasoning_ended(const rfg_state *st) {
    return st && st->reasoning_ended;
}

int rfg_debug_depth(const rfg_state *st) {
    return st ? st->depth : 0;
}

int rfg_debug_advances(const rfg_state *st) {
    return st ? st->advances : 0;
}

const char *rfg_debug_death_piece(const rfg_state *st) {
    return (st && st->death_piece[0]) ? st->death_piece : "";
}

const char *rfg_debug_death_phase(const rfg_state *st) {
    return (st && st->death_phase[0]) ? st->death_phase : "";
}

const char *rfg_debug_phase_name(const rfg_state *st) {
    if (!st) return "none";
    switch (st->phase) {
    case RFG_P_REASON:    return "reason";
    case RFG_P_PRE:       return "pre";
    case RFG_P_VALUE:     return "value";
    case RFG_P_VALUE_REQ: return "value_req";
    case RFG_P_STRING:    return "string";
    case RFG_P_STR_ESC:   return "str_esc";
    case RFG_P_STR_UNI:   return "str_uni";
    case RFG_P_NUMBER:    return "number";
    case RFG_P_LITERAL:   return "literal";
    case RFG_P_KEY:       return "key";
    case RFG_P_KEY_REQ:   return "key_req";
    case RFG_P_COLON:     return "colon";
    case RFG_P_AFTER:     return "after";
    case RFG_P_DONE:      return "done";
    case RFG_P_DEAD:      return "dead";
    default:              return "?";
    }
}
