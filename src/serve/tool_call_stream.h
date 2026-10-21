/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_call_stream.h
 * @brief Streams OpenAI tool-call deltas WHILE the model writes the call,
 *        instead of all at once when generation ends.
 *
 * The stream emitter diverts a tool call into a raw channel (serve_emitter.h);
 * before this module, nothing of it reached the client until the turn ended and
 * tc_parse_tool_calls ran.  A long call (a `write` of a whole file) is minutes
 * of silence on the wire, and a client watchdog reads silence as a dead stream.
 *
 * The streamer is fed every byte the emitter diverts (tcs_feed) and, through a
 * sink, puts three things on the wire:
 *   - call_begin      the OpenAI opening delta (index, id, name) — as soon as
 *                     the call's name is DECIDED;
 *   - call_arguments  argument fragments as the model writes them;
 *   - progress        "tool bytes arrived" — the caller's keepalive hook for
 *                     the stretches where nothing can be sent yet.
 * At end of turn tcs_finish reconciles with the authoritative parse and sends
 * only what is still owed.
 *
 * THE CONTRACT: for every call that ends up in the parse, the argument
 * fragments sent under its index concatenate to EXACTLY that call's
 * `arguments` (the string the non-streaming response carries).  Arguments are
 * the DOM re-serialization of the model's JSON (jr_to_json), so they are
 * streamed through json_canonical_stream, which writes only bytes that are
 * already final.  What cannot be decided early is held, not guessed:
 *   - Only the ORNITH_JSON dialect streams early, and only an object whose name
 *     comes from a "function" key (the spelling that outranks "name"/"func"),
 *     whose arguments are the first "arguments" member and are an object.
 *     Anything else (other spellings, the {"tool_calls":[…]} wrapper, the XML
 *     dialects, whose values are typed and may be rewritten by recovery) is
 *     sent whole the moment its object closes — or at end of turn for the XML
 *     dialects — with the caller's keepalive covering the gap.
 *   - The arguments' final '}' is held until the object closes and the parse
 *     confirms the call.  A call opened on the wire whose object then fails to
 *     parse cannot be retracted (OpenAI has no such delta), but it is left with
 *     arguments that are not valid JSON, so no client can run it as if it were
 *     complete; when it was the turn's only call, finish_reason is not
 *     "tool_calls" either — exactly as the non-streaming response reports it.
 *   - When an object closes, its calls are parsed by the SAME per-object code
 *     the end-of-turn parse uses (tc_parse_ornith_object) and anything not yet
 *     sent is sent then.  tcs_finish re-checks every call against the final
 *     parse (which also applies the truncation repair and the grammar gate).
 *
 * Depends on libc + tool_calling / json_canonical_stream / json_reader /
 * json_writer; no engine deps (unit-tested in tests/test_tool_call_stream.c).
 *
 * @status Complete.
 * @issues A call opened on the wire whose object then fails to parse cannot be
 *         retracted (there is no OpenAI delta for it).  It is logged
 *         ([tool_stream] ... abandoned), its arguments are left unparseable,
 *         and the turn's finish_reason/tool_call_error are exactly what the
 *         non-streaming path reports.
 * @todo None
 */

#ifndef PN_SERVE_TOOL_CALL_STREAM_H
#define PN_SERVE_TOOL_CALL_STREAM_H

#include <stdbool.h>
#include <stddef.h>

#include "json_canonical_stream.h"
#include "json_reader.h"
#include "json_writer.h"
#include "tool_calling.h"

/* Where the streamer's output goes.  Every pointer may be NULL. */
typedef struct {
    void (*call_begin)(void *ud, int index, const char *id, const char *name);
    void (*call_arguments)(void *ud, int index, const char *bytes, size_t n);
    void (*progress)(void *ud);
    void *ud;
} tcs_sink;

/* One call as the client has seen it. */
typedef struct {
    int         index;      /* wire tool_calls[].index */
    char       *id;
    char       *name;
    json_writer sent;       /* argument bytes already sent */
    bool        committed;  /* its object closed and parsed into this call */
    bool        abandoned;  /* opened on the wire, then the parse dropped it */
} tcs_wire_call;

typedef struct tcs_streamer {
    tc_dialect      dialect;
    const jr_value *tools;      /* borrowed: the request's tools */
    tcs_sink        sink;
    bool            early;      /* this dialect/request streams early at all */

    /* outer scan (mirrors parse_ornith_json: '{'-rooted objects in order) */
    bool        in_object;
    json_writer raw;            /* raw text of the current object */
    int         brace_depth;
    bool        brace_in_str;
    bool        brace_escape;

    /* current object */
    jcs_parser  jcs;
    bool        jcs_live;
    char       *member_key;     /* key of the root member being read */
    bool        fn_seen, fn_pending;
    bool        args_seen, args_live, args_done;
    size_t      args_begin, args_end;
    int         spec;           /* slot in `calls` streamed early, -1 none */

    tcs_wire_call *calls;
    int            n_calls, cap_calls;
    int            next_index;
    bool           any_committed;
} tcs_streamer;

/** Set up a streamer. `tools` is borrowed; NULL means the request has no tools
 *  (nothing is streamed early). Returns false on allocation failure. */
bool tcs_init(tcs_streamer *ts, tc_dialect dialect, const jr_value *tools,
              const tcs_sink *sink);

/** Feed bytes the emitter appended to its raw tool channel (in order). */
void tcs_feed(tcs_streamer *ts, const char *bytes, size_t n);

/**
 * End of turn: make the wire carry exactly `calls` (the final parse, after
 * repair and the grammar gate).  Sends what each call still owes; a call the
 * wire has not seen is sent whole (opening delta + arguments) under a fresh
 * index.  A call already on the wire whose sent bytes are not a prefix of the
 * final arguments is logged and re-sent under a fresh index — the contract
 * makes that unreachable, the check makes a violation loud.
 */
void tcs_finish(tcs_streamer *ts, const tc_tool_call *calls, int n);

/** True once any call delta has been sent (early or at an object's close). */
bool tcs_sent_any(const tcs_streamer *ts);

/** Free owned state. */
void tcs_free(tcs_streamer *ts);

#endif /* PN_SERVE_TOOL_CALL_STREAM_H */
