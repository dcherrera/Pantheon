/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_call_stream.c
 * @brief Implementation of the incremental tool-call streamer
 *        (see tool_call_stream.h).
 *
 * @status Complete.
 * @issues None beyond the header's (an abandoned early call cannot be
 *         retracted).
 * @todo None
 */

#include "tool_call_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- wire calls ---------- */

static tcs_wire_call *open_wire_call(tcs_streamer *ts, const char *id,
                                     const char *name)
{
    if (ts->n_calls >= ts->cap_calls) {
        int cap = ts->cap_calls ? ts->cap_calls * 2 : 4;
        tcs_wire_call *grown = realloc(ts->calls, (size_t)cap * sizeof(*grown));
        if (!grown) return NULL;
        ts->calls = grown;
        ts->cap_calls = cap;
    }
    tcs_wire_call *wc = &ts->calls[ts->n_calls];
    memset(wc, 0, sizeof(*wc));
    wc->index = ts->next_index;
    wc->id = strdup(id);
    wc->name = strdup(name);
    if (!wc->id || !wc->name || !jw_init(&wc->sent)) {
        free(wc->id);
        free(wc->name);
        jw_free(&wc->sent);
        return NULL;
    }
    ts->n_calls++;
    ts->next_index++;
    if (ts->sink.call_begin)
        ts->sink.call_begin(ts->sink.ud, wc->index, wc->id, wc->name);
    return wc;
}

static void send_arguments(tcs_streamer *ts, tcs_wire_call *wc,
                           const char *bytes, size_t n)
{
    if (n == 0) return;
    jw_raw_n(&wc->sent, bytes, n);
    if (ts->sink.call_arguments)
        ts->sink.call_arguments(ts->sink.ud, wc->index, bytes, n);
}

/* Send what `wc` still owes of `final_args`, if what it has sent is a prefix of
 * them.  Returns false (sending nothing) when it is not. */
static bool complete_wire_call(tcs_streamer *ts, tcs_wire_call *wc,
                               const char *name, const char *final_args)
{
    size_t total = strlen(final_args);
    if (strcmp(wc->name, name) != 0 || wc->sent.len > total ||
        memcmp(wc->sent.data, final_args, wc->sent.len) != 0)
        return false;
    send_arguments(ts, wc, final_args + wc->sent.len, total - wc->sent.len);
    wc->committed = true;
    ts->any_committed = true;
    return true;
}

/* A call the wire has not seen: opening delta, then all of its arguments. */
static void send_whole_call(tcs_streamer *ts, const tc_tool_call *call)
{
    tcs_wire_call *wc = open_wire_call(ts, call->id, call->name);
    if (!wc) return;
    send_arguments(ts, wc, call->arguments, strlen(call->arguments));
    wc->committed = true;
    ts->any_committed = true;
}

/* ---------- the object being read: root-member events from jcs ---------- */

static void on_root_key(void *ud, const char *key)
{
    tcs_streamer *ts = ud;
    free(ts->member_key);
    ts->member_key = strdup(key);
}

static void on_root_value_begin(void *ud, size_t at, char first)
{
    tcs_streamer *ts = ud;
    const char *key = ts->member_key ? ts->member_key : "";
    /* Only the FIRST member of each spelling counts: jr_object_get returns the
     * first match, so a later duplicate never decides anything. */
    if (strcmp(key, "function") == 0 && !ts->fn_seen) {
        ts->fn_seen = true;
        ts->fn_pending = true;
    }
    if (strcmp(key, "arguments") == 0 && !ts->args_seen) {
        ts->args_seen = true;
        if (first == '{') {             /* a non-object falls to "parameters" */
            ts->args_live = true;
            ts->args_begin = at;
        }
    }
}

static void on_root_value_end(void *ud, size_t end, const char *decoded)
{
    tcs_streamer *ts = ud;
    if (ts->args_live && !ts->args_done && ts->member_key &&
        strcmp(ts->member_key, "arguments") == 0 && end > ts->args_begin) {
        ts->args_done = true;
        ts->args_end = end;
    }
    if (!ts->fn_pending) return;
    ts->fn_pending = false;
    /* The first "function" member is a string: that IS the call's name (it
     * outranks "name"/"func"), so the call can open now — if the parse would
     * accept the name at all. */
    if (!decoded || !*decoded || !tc_tool_name_known(ts->tools, decoded)) return;
    char id[24];
    tc_new_call_id(id, sizeof(id));
    tcs_wire_call *wc = open_wire_call(ts, id, decoded);
    if (wc) ts->spec = ts->n_calls - 1;
}

/* Send the early call's newly decided argument bytes.  The arguments' closing
 * '}' is held until the object's parse confirms the call. */
static void pump_early_arguments(tcs_streamer *ts)
{
    if (ts->spec < 0 || !ts->args_live || ts->jcs.status == JCS_FAILED) return;
    tcs_wire_call *wc = &ts->calls[ts->spec];
    size_t limit = ts->args_done ? ts->args_end - 1 : ts->jcs.out.len;
    size_t from = ts->args_begin + wc->sent.len;
    if (limit > from)
        send_arguments(ts, wc, ts->jcs.out.data + from, limit - from);
}

static void reset_object(tcs_streamer *ts)
{
    if (ts->jcs_live) jcs_free(&ts->jcs);
    ts->jcs_live = false;
    free(ts->member_key);
    ts->member_key = NULL;
    ts->in_object = false;
    ts->raw.len = 0;
    ts->brace_depth = 0;
    ts->brace_in_str = ts->brace_escape = false;
    ts->fn_seen = ts->fn_pending = false;
    ts->args_seen = ts->args_live = ts->args_done = false;
    ts->args_begin = ts->args_end = 0;
    ts->spec = -1;
}

static void begin_object(tcs_streamer *ts)
{
    reset_object(ts);
    ts->in_object = true;
    jcs_events ev = { on_root_key, on_root_value_begin, on_root_value_end, ts };
    ts->jcs_live = jcs_init(&ts->jcs, &ev);
}

static void feed_object(tcs_streamer *ts, const char *bytes, size_t n)
{
    jw_raw_n(&ts->raw, bytes, n);
    if (!ts->jcs_live) return;
    jcs_feed(&ts->jcs, bytes, n);
    pump_early_arguments(ts);
}

/* The object closed: parse it exactly as the end-of-turn parse will, and put
 * on the wire whatever of its calls is not there yet. */
static void close_object(tcs_streamer *ts)
{
    tc_tool_call *calls = NULL;
    int n = 0;
    const char *open = ts->raw.data;
    if (tc_parse_ornith_object(open, open + ts->raw.len, ts->tools, &calls, &n) != 0)
        n = 0;
    int first = 0;
    if (ts->spec >= 0) {
        tcs_wire_call *wc = &ts->calls[ts->spec];
        if (n > 0 && complete_wire_call(ts, wc, calls[0].name, calls[0].arguments)) {
            first = 1;
        } else {
            wc->abandoned = true;
            fprintf(stderr, "[tool_stream] call %d (%s) was streamed early but its "
                    "object parsed to %s — abandoned on the wire with "
                    "unterminated arguments (%zu bytes sent)\n", wc->index,
                    wc->name, n > 0 ? "a different call" : "no call",
                    wc->sent.len);
        }
    }
    for (int i = first; i < n; i++)
        send_whole_call(ts, &calls[i]);
    tc_tool_calls_free(calls, n);
    reset_object(ts);
}

/* Mirror parse_ornith_json's scan: each '{' starts an object that ends at its
 * matching '}' (braces inside strings ignored, as json_object_end does). */
static void scan_objects(tcs_streamer *ts, const char *bytes, size_t n)
{
    size_t i = 0;
    while (i < n) {
        if (!ts->in_object) {
            const char *brace = memchr(bytes + i, '{', n - i);
            if (!brace) return;
            i = (size_t)(brace - bytes);
            begin_object(ts);
        }
        size_t start = i;
        bool closed = false;
        for (; i < n && !closed; i++) {
            char c = bytes[i];
            if (ts->brace_in_str) {
                if (ts->brace_escape) ts->brace_escape = false;
                else if (c == '\\') ts->brace_escape = true;
                else if (c == '"') ts->brace_in_str = false;
                continue;
            }
            if (c == '"') ts->brace_in_str = true;
            else if (c == '{') ts->brace_depth++;
            else if (c == '}' && --ts->brace_depth == 0) closed = true;
        }
        feed_object(ts, bytes + start, i - start);
        if (closed) close_object(ts);
    }
}

/* ---------- public API ---------- */

bool tcs_init(tcs_streamer *ts, tc_dialect dialect, const jr_value *tools,
              const tcs_sink *sink)
{
    memset(ts, 0, sizeof(*ts));
    ts->dialect = dialect;
    ts->tools = tools;
    if (sink) ts->sink = *sink;
    ts->spec = -1;
    ts->early = (dialect == TC_DIALECT_ORNITH_JSON && tools != NULL);
    return jw_init(&ts->raw);
}

void tcs_feed(tcs_streamer *ts, const char *bytes, size_t n)
{
    if (ts->early && n > 0) scan_objects(ts, bytes, n);
    if (ts->sink.progress) ts->sink.progress(ts->sink.ud);
}

void tcs_finish(tcs_streamer *ts, const tc_tool_call *calls, int n)
{
    int w = 0;   /* next wire call to match, skipping abandoned ones */
    for (int j = 0; j < n; j++) {
        while (w < ts->n_calls && ts->calls[w].abandoned) w++;
        tcs_wire_call *wc = (w < ts->n_calls) ? &ts->calls[w] : NULL;
        if (wc) {
            w++;
            if (complete_wire_call(ts, wc, calls[j].name, calls[j].arguments))
                continue;
            wc->abandoned = true;
            fprintf(stderr, "[tool_stream] BUG: wire call %d (%s, %zu bytes sent) "
                    "is not a prefix of final call %d (%s) — re-sending it whole\n",
                    wc->index, wc->name, wc->sent.len, j, calls[j].name);
        }
        send_whole_call(ts, &calls[j]);
    }
}

bool tcs_sent_any(const tcs_streamer *ts)
{
    return ts->n_calls > 0;
}

void tcs_free(tcs_streamer *ts)
{
    reset_object(ts);
    jw_free(&ts->raw);
    for (int i = 0; i < ts->n_calls; i++) {
        free(ts->calls[i].id);
        free(ts->calls[i].name);
        jw_free(&ts->calls[i].sent);
    }
    free(ts->calls);
    ts->calls = NULL;
    ts->n_calls = ts->cap_calls = 0;
}
