/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_canonical_stream.c
 * @brief Implementation of the incremental JSON re-serializer
 *        (see json_canonical_stream.h).  A pushdown state machine fed one byte
 *        at a time (plain string bytes in bulk), mirroring json_reader.c's
 *        acceptance rule for rule.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "json_canonical_stream.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "json_reader.h"

typedef enum {
    ST_VALUE = 0,   /* a value is owed (whitespace skipped) */
    ST_OBJ_FIRST,   /* just after '{': a key or '}' */
    ST_OBJ_KEY,     /* just after ',' in an object: a key */
    ST_COLON,       /* after a key: ':' */
    ST_ARR_FIRST,   /* just after '[': a value or ']' */
    ST_AFTER,       /* after a value inside a container: ',' or the closer */
    ST_STRING,      /* inside a string (key or value) */
    ST_TOKEN,       /* inside a number or literal */
    ST_END,         /* DONE or FAILED */
} jcs_state;

typedef enum {
    ESC_NONE = 0,
    ESC_BACKSLASH,       /* read '\' */
    ESC_HEX,             /* reading the 4 digits of \uXXXX */
    ESC_HIGH,            /* a high surrogate was read: is "\u" next? */
    ESC_HIGH_BACKSLASH,  /* high surrogate, then '\' */
    ESC_LOW,             /* reading the 4 digits of the low half */
} jcs_escape;

static bool is_json_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/* Bytes that end a number/literal token.  jr_parse reads a scalar and then
 * requires whitespace, ',' or the closer; any other structural byte right after
 * a scalar is a rejection, which the state after the token reports. */
static bool ends_token(char c)
{
    return is_json_ws(c) || c == ',' || c == '}' || c == ']' || c == '{' ||
           c == '[' || c == '"' || c == ':';
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void fail(jcs_parser *p)
{
    p->status = JCS_FAILED;
    p->state = ST_END;
}

static bool stack_push(jcs_parser *p, char kind)
{
    jcs_stack *s = &p->stack;
    if (s->depth >= s->cap) {
        int cap = s->cap ? s->cap * 2 : 16;
        char *grown = realloc(s->kinds, (size_t)cap);
        if (!grown) return false;
        s->kinds = grown;
        s->cap = cap;
    }
    s->kinds[s->depth++] = kind;
    return true;
}

static char stack_top(const jcs_parser *p)
{
    return p->stack.depth ? p->stack.kinds[p->stack.depth - 1] : 0;
}

/* Inside the root object, at member level (where the events fire). */
static bool at_root_member(const jcs_parser *p)
{
    return p->stack.depth == 1 && p->stack.kinds[0] == 'o';
}

bool jcs_init(jcs_parser *p, const jcs_events *ev)
{
    memset(p, 0, sizeof(*p));
    if (ev) p->ev = *ev;
    if (!jw_init(&p->out) || !jw_init(&p->token) || !jw_init(&p->decoded)) {
        jcs_free(p);
        return false;
    }
    p->state = ST_VALUE;
    return true;
}

void jcs_free(jcs_parser *p)
{
    jw_free(&p->out);
    jw_free(&p->token);
    jw_free(&p->decoded);
    free(p->stack.kinds);
    p->stack.kinds = NULL;
    p->stack.depth = p->stack.cap = 0;
}

/* A value just finished at the current depth. */
static void value_complete(jcs_parser *p, const char *decoded)
{
    if (p->stack.depth == 0) {
        p->status = JCS_DONE;
        p->state = ST_END;
        return;
    }
    if (at_root_member(p) && p->ev.root_value_end)
        p->ev.root_value_end(p->ev.ud, p->out.len, decoded);
    p->state = ST_AFTER;
}

/* Append decoded string bytes: to the decoded copy, and (for a value) escaped
 * to the output.  Nothing after a decoded NUL survives — the DOM keeps strings
 * as C strings, so jr_to_json stops there. */
static void emit_decoded(jcs_parser *p, const char *bytes, size_t n)
{
    if (p->str_nul || n == 0) return;
    jw_raw_n(&p->decoded, bytes, n);
    if (!p->str_is_key) jw_escaped_n(&p->out, bytes, n);
}

/* One code point, encoded exactly as json_reader.c's append_utf8 does (the
 * same truncating char casts, so even out-of-range surrogate arithmetic matches
 * byte for byte). */
static void emit_code_point(jcs_parser *p, uint32_t cp)
{
    char buf[4];
    size_t len = 0;
    if (cp == 0) { p->str_nul = true; return; }
    if (cp <= 0x7F) {
        buf[len++] = (char)cp;
    } else if (cp <= 0x7FF) {
        buf[len++] = (char)(0xC0 | (cp >> 6));
        buf[len++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        buf[len++] = (char)(0xE0 | (cp >> 12));
        buf[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[len++] = (char)(0x80 | (cp & 0x3F));
    } else {
        buf[len++] = (char)(0xF0 | (cp >> 18));
        buf[len++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[len++] = (char)(0x80 | (cp & 0x3F));
    }
    emit_decoded(p, buf, len);
}

static void begin_string(jcs_parser *p, bool is_key)
{
    p->str_is_key = is_key;
    p->str_nul = false;
    p->esc = ESC_NONE;
    p->decoded.len = 0;
    p->decoded.data[0] = '\0';
    if (!is_key) jw_raw_n(&p->out, "\"", 1);
    p->state = ST_STRING;
}

static void end_string(jcs_parser *p)
{
    if (p->str_is_key) {
        jw_string_n(&p->out, p->decoded.data, p->decoded.len);
        if (at_root_member(p) && p->ev.root_key)
            p->ev.root_key(p->ev.ud, p->decoded.data);
        p->state = ST_COLON;
        return;
    }
    jw_raw_n(&p->out, "\"", 1);
    value_complete(p, p->decoded.data);
}

/* Write a finished number/literal, or fail if jr_parse would reject it. */
static void finish_token(jcs_parser *p)
{
    const char *t = p->token.data;
    size_t len = p->token.len;
    if (t[0] == 't' || t[0] == 'f' || t[0] == 'n') {
        if (strcmp(t, "true") != 0 && strcmp(t, "false") != 0 &&
            strcmp(t, "null") != 0) { fail(p); return; }
        jw_raw_n(&p->out, t, len);
    } else {
        char *end = NULL;
        strtod(t, &end);
        if (end == t || end != t + len) { fail(p); return; }
        jr_value *v = jr_parse(t);        /* the DOM's own number → text path */
        if (!v) { fail(p); return; }
        jr_to_json(v, &p->out);
        jr_free(v);
    }
    value_complete(p, NULL);
}

/* The first byte of a value (whitespace already skipped). */
static void begin_value(jcs_parser *p, char c)
{
    if (c == ',' || c == '}' || c == ']' || c == ':') { fail(p); return; }
    if (at_root_member(p) && p->ev.root_value_begin)
        p->ev.root_value_begin(p->ev.ud, p->out.len, c);
    if (c == '"') { begin_string(p, false); return; }
    if (c == '{' || c == '[') {
        if (!stack_push(p, c == '{' ? 'o' : 'a')) { fail(p); return; }
        jw_raw_n(&p->out, &c, 1);
        p->state = (c == '{') ? ST_OBJ_FIRST : ST_ARR_FIRST;
        return;
    }
    p->token.len = 0;
    jw_raw_n(&p->token, &c, 1);
    p->state = ST_TOKEN;
}

static void close_container(jcs_parser *p, char closer)
{
    jw_raw_n(&p->out, &closer, 1);
    p->stack.depth--;
    value_complete(p, NULL);
}

/* One escape-sequence byte.  Returns true when `c` must be re-read in the
 * (new) state — the byte after a high surrogate that turned out not to start a
 * low one. */
static bool step_escape(jcs_parser *p, char c)
{
    switch (p->esc) {
    case ESC_BACKSLASH: {
        char out;
        switch (c) {
        case '"': case '\\': case '/': out = c; break;
        case 'b': out = '\b'; break;
        case 'f': out = '\f'; break;
        case 'n': out = '\n'; break;
        case 'r': out = '\r'; break;
        case 't': out = '\t'; break;
        case 'u': p->esc = ESC_HEX; p->hex_n = 0; p->hex_val = 0; return false;
        default:  fail(p); return false;
        }
        emit_decoded(p, &out, 1);
        p->esc = ESC_NONE;
        return false;
    }
    case ESC_HEX:
    case ESC_LOW: {
        int nib = hex_nibble(c);
        if (nib < 0) { fail(p); return false; }
        p->hex_val = (p->hex_val << 4) | nib;
        if (++p->hex_n < 4) return false;
        if (p->esc == ESC_LOW) {
            /* json_reader.c's arithmetic verbatim (int, unchecked low half). */
            int cp = 0x10000 + (((p->high - 0xD800) << 10) | (p->hex_val - 0xDC00));
            emit_code_point(p, (uint32_t)cp);
            p->esc = ESC_NONE;
        } else if (p->hex_val >= 0xD800 && p->hex_val <= 0xDBFF) {
            p->high = p->hex_val;
            p->esc = ESC_HIGH;
        } else {
            emit_code_point(p, (uint32_t)p->hex_val);
            p->esc = ESC_NONE;
        }
        return false;
    }
    case ESC_HIGH:
        if (c == '\\') { p->esc = ESC_HIGH_BACKSLASH; return false; }
        emit_code_point(p, (uint32_t)p->high);
        p->esc = ESC_NONE;
        return true;
    case ESC_HIGH_BACKSLASH:
        if (c == 'u') { p->esc = ESC_LOW; p->hex_n = 0; p->hex_val = 0; return false; }
        emit_code_point(p, (uint32_t)p->high);
        p->esc = ESC_BACKSLASH;          /* the '\' starts an ordinary escape */
        return true;
    default:
        return false;
    }
}

/* One byte in a structural (non-string, non-token) state. */
static void step_structural(jcs_parser *p, char c)
{
    if (is_json_ws(c)) return;
    switch (p->state) {
    case ST_VALUE:
        begin_value(p, c);
        return;
    case ST_OBJ_FIRST:
        if (c == '}') { close_container(p, '}'); return; }
        if (c == '"') { begin_string(p, true); return; }
        fail(p);
        return;
    case ST_OBJ_KEY:
        if (c == '"') { begin_string(p, true); return; }
        fail(p);
        return;
    case ST_COLON:
        if (c == ':') { jw_raw_n(&p->out, ":", 1); p->state = ST_VALUE; return; }
        fail(p);
        return;
    case ST_ARR_FIRST:
        if (c == ']') { close_container(p, ']'); return; }
        p->state = ST_VALUE;
        begin_value(p, c);
        return;
    case ST_AFTER: {
        char top = stack_top(p);
        if (c == ',') {
            jw_raw_n(&p->out, ",", 1);
            p->state = (top == 'o') ? ST_OBJ_KEY : ST_VALUE;
            return;
        }
        if ((top == 'o' && c == '}') || (top == 'a' && c == ']')) {
            close_container(p, c);
            return;
        }
        fail(p);
        return;
    }
    default:
        return;
    }
}

/* Run of plain string bytes starting at bytes[i]: neither '"' nor '\'. */
static size_t plain_run(const char *bytes, size_t i, size_t n)
{
    size_t j = i;
    while (j < n && bytes[j] != '"' && bytes[j] != '\\') j++;
    return j - i;
}

size_t jcs_feed(jcs_parser *p, const char *bytes, size_t n)
{
    size_t i = 0;
    while (i < n && p->status == JCS_RUNNING) {
        char c = bytes[i];
        if (p->state == ST_STRING) {
            if (p->esc == ESC_NONE) {
                size_t run = plain_run(bytes, i, n);
                if (run) { emit_decoded(p, bytes + i, run); i += run; continue; }
                if (c == '"') end_string(p);
                else p->esc = ESC_BACKSLASH;
                i++;
                continue;
            }
            if (!step_escape(p, c)) i++;
            continue;
        }
        if (p->state == ST_TOKEN) {
            if (!ends_token(c)) { jw_raw_n(&p->token, &c, 1); i++; continue; }
            finish_token(p);            /* then re-read c after the value */
            continue;
        }
        step_structural(p, c);
        i++;
    }
    if (p->out.oom || p->token.oom || p->decoded.oom) fail(p);
    return i;
}
