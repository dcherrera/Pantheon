/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_canonical_stream.h
 * @brief Incremental JSON re-serializer: fed raw JSON text in arbitrary
 *        pieces, it writes — as early as each byte is DECIDED — exactly the
 *        bytes `jr_to_json(jr_parse(text))` would produce for the whole text.
 *
 * Why it exists.  The serve's tool-call parser builds a call's `arguments`
 * string by parsing the model's JSON into a DOM (json_reader) and
 * re-serializing it compactly (jr_to_json).  Streaming those arguments while the
 * model is still writing them is only correct if the bytes streamed so far are
 * a PREFIX of that final re-serialization.  This module is the incremental half
 * of that contract: everything it writes to `out` is final, and when the input
 * is complete `out` is byte-identical to jr_to_json's output.
 *
 * Acceptance mirrors jr_parse, not RFC 8259: the same whitespace set
 * (space/tab/LF/CR), the same string escapes (including the unpaired-surrogate
 * and "\u0000 truncates the C string" behaviour), numbers read by strtod and
 * written by jr_to_json, literals true/false/null.  A text jr_parse rejects puts
 * the parser in JCS_FAILED at the first byte that decides the rejection.
 *
 * What is held back (not yet in `out`): a number or literal until the byte that
 * ends it (its value — and so its re-serialized form — is unknown until then),
 * an object key until its closing quote, a partial escape sequence, and a high
 * surrogate until the next bytes show whether a low surrogate follows.  String
 * CONTENT is written as it arrives, which is what makes a long string value
 * stream.
 *
 * Root-object member events (jcs_events) let a caller find one member's value
 * inside `out` — the tool-call streamer uses them to slice out "arguments".
 *
 * Depends only on libc + json_writer/json_reader.
 *
 * @status Complete; byte-equivalence with jr_parse+jr_to_json is property-tested
 *         over generated documents fed in random splits
 *         (tests/test_tool_call_stream.c).
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_JSON_CANONICAL_STREAM_H
#define PN_SERVE_JSON_CANONICAL_STREAM_H

#include <stdbool.h>
#include <stddef.h>

#include "json_writer.h"

typedef enum {
    JCS_RUNNING = 0,   /* more input wanted */
    JCS_DONE,          /* the root value is complete; later bytes are ignored */
    JCS_FAILED,        /* jr_parse would reject this text */
} jcs_status;

/* Callbacks for the members of a ROOT OBJECT (depth 1).  Every pointer may be
 * NULL.  Offsets are into the parser's `out`. */
typedef struct {
    /* A member key completed (decoded; cut at an embedded NUL as the DOM does). */
    void (*root_key)(void *ud, const char *key);
    /* That member's value begins; its first canonical byte will be at `at`.
     * `first` is the value's first raw char ('{', '[', '"', or a scalar's). */
    void (*root_value_begin)(void *ud, size_t at, char first);
    /* That member's value is complete; its canonical bytes end at `end`.
     * `decoded` is the decoded string for a string value, else NULL. */
    void (*root_value_end)(void *ud, size_t end, const char *decoded);
    void *ud;
} jcs_events;

/* One frame of the container stack: 'o' object, 'a' array. */
typedef struct {
    char  *kinds;
    int    depth;
    int    cap;
} jcs_stack;

typedef struct {
    jcs_status  status;
    json_writer out;          /* canonical bytes so far (all final) */

    /* -- private -- */
    jcs_events  ev;
    jcs_stack   stack;
    int         state;        /* jcs_state (json_canonical_stream.c) */
    json_writer token;        /* pending number/literal text */
    json_writer decoded;      /* decoded bytes of the string being read */
    bool        str_is_key;
    bool        str_nul;      /* a \u0000 ended the C string: drop the rest */
    int         esc;          /* escape sub-state */
    int         hex_n;        /* hex digits read in the current \u */
    int         hex_val;
    int         high;         /* pending high surrogate */
} jcs_parser;

/** Initialize. `ev` may be NULL. Returns false on allocation failure. */
bool jcs_init(jcs_parser *p, const jcs_events *ev);

/**
 * Feed `n` bytes. Returns how many were consumed: all of them while RUNNING;
 * fewer once the root value completes (the byte that completed it is consumed,
 * the rest are not). A FAILED or DONE parser consumes nothing.
 */
size_t jcs_feed(jcs_parser *p, const char *bytes, size_t n);

/** Release owned buffers. */
void jcs_free(jcs_parser *p);

#endif /* PN_SERVE_JSON_CANONICAL_STREAM_H */
