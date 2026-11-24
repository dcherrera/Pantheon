/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_reader.c
 * @brief Implementation of the minimal JSON DOM parser (see json_reader.h).
 *
 * @status Handles the OpenAI request grammar; UTF-16 surrogate pairs in \u
 *         escapes are decoded to UTF-8.
 * @issues None
 * @todo None
 */

#include "json_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One object member: key + value. */
typedef struct {
    char *key;
    jr_value *value;
} jr_member;

struct jr_value {
    jr_type type;
    union {
        bool boolean;
        double number;
        char *string;            /* JR_STRING (owned) */
        struct {                 /* JR_ARRAY */
            jr_value **items;
            int count;
        } array;
        struct {                 /* JR_OBJECT */
            jr_member *members;
            int count;
        } object;
    } as;
};

/* Cursor over the source text. */
typedef struct {
    const char *p;
    bool error;
} jr_cursor;

static jr_value *parse_value(jr_cursor *cur);

static void skip_whitespace(jr_cursor *cur)
{
    while (*cur->p == ' ' || *cur->p == '\t' || *cur->p == '\n' || *cur->p == '\r')
        cur->p++;
}

/* Append one Unicode code point as UTF-8 to buf[*len], growing via caller cap. */
static void append_utf8(char *buf, size_t *len, uint32_t cp)
{
    if (cp <= 0x7F) {
        buf[(*len)++] = (char)cp;
    } else if (cp <= 0x7FF) {
        buf[(*len)++] = (char)(0xC0 | (cp >> 6));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        buf[(*len)++] = (char)(0xE0 | (cp >> 12));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        buf[(*len)++] = (char)(0xF0 | (cp >> 18));
        buf[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Read 4 hex digits at cur->p (which points just past the 'u'); -1 on error. */
static int read_hex4(jr_cursor *cur)
{
    int v = 0;
    for (int i = 0; i < 4; i++) {
        int n = hex_nibble(cur->p[i]);
        if (n < 0) return -1;
        v = (v << 4) | n;
    }
    cur->p += 4;
    return v;
}

/* Parse a JSON string literal (cur->p at the opening quote). Returns a malloc'd
 * unescaped NUL-terminated string, or NULL on error. */
static char *parse_string_raw(jr_cursor *cur)
{
    cur->p++; /* opening quote */
    const char *start = cur->p;
    size_t max = 1;
    for (const char *s = start; *s && *s != '"'; s++) {
        max += (*s == '\\') ? 4 : 1; /* worst case: \uXXXX -> up to 4 UTF-8 bytes */
        if (*s == '\\' && s[1]) s++;
    }
    char *out = malloc(max + 1);
    if (!out) { cur->error = true; return NULL; }

    size_t len = 0;
    while (*cur->p && *cur->p != '"') {
        char c = *cur->p;
        if (c != '\\') {
            out[len++] = c;
            cur->p++;
            continue;
        }
        cur->p++; /* consume backslash */
        char e = *cur->p++;
        switch (e) {
            case '"':  out[len++] = '"';  break;
            case '\\': out[len++] = '\\'; break;
            case '/':  out[len++] = '/';  break;
            case 'b':  out[len++] = '\b'; break;
            case 'f':  out[len++] = '\f'; break;
            case 'n':  out[len++] = '\n'; break;
            case 'r':  out[len++] = '\r'; break;
            case 't':  out[len++] = '\t'; break;
            case 'u': {
                int cp = read_hex4(cur);
                if (cp < 0) { free(out); cur->error = true; return NULL; }
                if (cp >= 0xD800 && cp <= 0xDBFF &&
                    cur->p[0] == '\\' && cur->p[1] == 'u') {
                    cur->p += 2;
                    int lo = read_hex4(cur);
                    if (lo < 0) { free(out); cur->error = true; return NULL; }
                    cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                }
                append_utf8(out, &len, (uint32_t)cp);
                break;
            }
            default:
                free(out);
                cur->error = true;
                return NULL;
        }
    }
    if (*cur->p != '"') { free(out); cur->error = true; return NULL; }
    cur->p++; /* closing quote */
    out[len] = '\0';
    return out;
}

static jr_value *make_value(jr_type type)
{
    jr_value *v = calloc(1, sizeof(*v));
    if (v) v->type = type;
    return v;
}

static jr_value *parse_string(jr_cursor *cur)
{
    char *s = parse_string_raw(cur);
    if (!s) return NULL;
    jr_value *v = make_value(JR_STRING);
    if (!v) { free(s); cur->error = true; return NULL; }
    v->as.string = s;
    return v;
}

static jr_value *parse_number(jr_cursor *cur)
{
    char *end = NULL;
    double d = strtod(cur->p, &end);
    if (end == cur->p) { cur->error = true; return NULL; }
    cur->p = end;
    jr_value *v = make_value(JR_NUMBER);
    if (!v) { cur->error = true; return NULL; }
    v->as.number = d;
    return v;
}

static jr_value *parse_literal(jr_cursor *cur, const char *word, jr_type type,
                               bool bool_value)
{
    size_t n = strlen(word);
    if (strncmp(cur->p, word, n) != 0) { cur->error = true; return NULL; }
    cur->p += n;
    jr_value *v = make_value(type);
    if (!v) { cur->error = true; return NULL; }
    if (type == JR_BOOL) v->as.boolean = bool_value;
    return v;
}

static jr_value *parse_array(jr_cursor *cur)
{
    cur->p++; /* '[' */
    jr_value *v = make_value(JR_ARRAY);
    if (!v) { cur->error = true; return NULL; }
    skip_whitespace(cur);
    if (*cur->p == ']') { cur->p++; return v; }

    int cap = 4;
    v->as.array.items = malloc((size_t)cap * sizeof(jr_value *));
    if (!v->as.array.items) { jr_free(v); cur->error = true; return NULL; }

    while (1) {
        jr_value *item = parse_value(cur);
        if (cur->error) { jr_free(v); return NULL; }
        if (v->as.array.count >= cap) {
            cap *= 2;
            jr_value **grown = realloc(v->as.array.items,
                                       (size_t)cap * sizeof(jr_value *));
            if (!grown) { jr_free(item); jr_free(v); cur->error = true; return NULL; }
            v->as.array.items = grown;
        }
        v->as.array.items[v->as.array.count++] = item;
        skip_whitespace(cur);
        if (*cur->p == ',') { cur->p++; skip_whitespace(cur); continue; }
        if (*cur->p == ']') { cur->p++; break; }
        jr_free(v);
        cur->error = true;
        return NULL;
    }
    return v;
}

static jr_value *parse_object(jr_cursor *cur)
{
    cur->p++; /* '{' */
    jr_value *v = make_value(JR_OBJECT);
    if (!v) { cur->error = true; return NULL; }
    skip_whitespace(cur);
    if (*cur->p == '}') { cur->p++; return v; }

    int cap = 4;
    v->as.object.members = malloc((size_t)cap * sizeof(jr_member));
    if (!v->as.object.members) { jr_free(v); cur->error = true; return NULL; }

    while (1) {
        skip_whitespace(cur);
        if (*cur->p != '"') { jr_free(v); cur->error = true; return NULL; }
        char *key = parse_string_raw(cur);
        if (!key) { jr_free(v); return NULL; }
        skip_whitespace(cur);
        if (*cur->p != ':') { free(key); jr_free(v); cur->error = true; return NULL; }
        cur->p++;
        jr_value *val = parse_value(cur);
        if (cur->error) { free(key); jr_free(v); return NULL; }
        if (v->as.object.count >= cap) {
            cap *= 2;
            jr_member *grown = realloc(v->as.object.members,
                                       (size_t)cap * sizeof(jr_member));
            if (!grown) { free(key); jr_free(val); jr_free(v); cur->error = true; return NULL; }
            v->as.object.members = grown;
        }
        v->as.object.members[v->as.object.count].key = key;
        v->as.object.members[v->as.object.count].value = val;
        v->as.object.count++;
        skip_whitespace(cur);
        if (*cur->p == ',') { cur->p++; continue; }
        if (*cur->p == '}') { cur->p++; break; }
        jr_free(v);
        cur->error = true;
        return NULL;
    }
    return v;
}

static jr_value *parse_value(jr_cursor *cur)
{
    skip_whitespace(cur);
    switch (*cur->p) {
        case '"': return parse_string(cur);
        case '{': return parse_object(cur);
        case '[': return parse_array(cur);
        case 't': return parse_literal(cur, "true", JR_BOOL, true);
        case 'f': return parse_literal(cur, "false", JR_BOOL, false);
        case 'n': return parse_literal(cur, "null", JR_NULL, false);
        default:  return parse_number(cur);
    }
}

jr_value *jr_parse(const char *text)
{
    if (!text) return NULL;
    jr_cursor cur = { .p = text, .error = false };
    jr_value *root = parse_value(&cur);
    if (cur.error) { jr_free(root); return NULL; }
    return root;
}

void jr_free(jr_value *value)
{
    if (!value) return;
    switch (value->type) {
        case JR_STRING:
            free(value->as.string);
            break;
        case JR_ARRAY:
            for (int i = 0; i < value->as.array.count; i++)
                jr_free(value->as.array.items[i]);
            free(value->as.array.items);
            break;
        case JR_OBJECT:
            for (int i = 0; i < value->as.object.count; i++) {
                free(value->as.object.members[i].key);
                jr_free(value->as.object.members[i].value);
            }
            free(value->as.object.members);
            break;
        default:
            break;
    }
    free(value);
}

jr_type jr_kind(const jr_value *value)
{
    return value ? value->type : JR_NULL;
}

const jr_value *jr_object_get(const jr_value *value, const char *key)
{
    if (!value || value->type != JR_OBJECT || !key) return NULL;
    for (int i = 0; i < value->as.object.count; i++)
        if (strcmp(value->as.object.members[i].key, key) == 0)
            return value->as.object.members[i].value;
    return NULL;
}

int jr_array_len(const jr_value *value)
{
    return (value && value->type == JR_ARRAY) ? value->as.array.count : 0;
}

int jr_object_member_count(const jr_value *value)
{
    return (value && value->type == JR_OBJECT) ? value->as.object.count : 0;
}

const char *jr_object_key_at(const jr_value *value, int index)
{
    if (!value || value->type != JR_OBJECT) return NULL;
    if (index < 0 || index >= value->as.object.count) return NULL;
    return value->as.object.members[index].key;
}

const jr_value *jr_object_value_at(const jr_value *value, int index)
{
    if (!value || value->type != JR_OBJECT) return NULL;
    if (index < 0 || index >= value->as.object.count) return NULL;
    return value->as.object.members[index].value;
}

const jr_value *jr_array_get(const jr_value *value, int index)
{
    if (!value || value->type != JR_ARRAY) return NULL;
    if (index < 0 || index >= value->as.array.count) return NULL;
    return value->as.array.items[index];
}

const char *jr_as_string(const jr_value *value)
{
    return (value && value->type == JR_STRING) ? value->as.string : NULL;
}

double jr_as_number(const jr_value *value, double fallback)
{
    return (value && value->type == JR_NUMBER) ? value->as.number : fallback;
}

bool jr_as_bool(const jr_value *value, bool fallback)
{
    return (value && value->type == JR_BOOL) ? value->as.boolean : fallback;
}

/* Render a number without a trailing ".0" when it is integral, so a schema
 * like {"minimum": 0} round-trips as 0 rather than 0.0.  Anything else gets the
 * SHORTEST digits that read back to the same double, as Python's json.dumps
 * (float repr) writes it: 0.1 is "0.1", not %.17g's "0.10000000000000001".
 * For a non-integral double %g picks the same plain/exponent form repr does. */
static void jr_write_number(json_writer *jw, double n)
{
    if (isfinite(n) && n == floor(n) && fabs(n) < 1e15) {
        jw_printf(jw, "%lld", (long long)n);
        return;
    }
    char digits[32];
    for (int precision = 1; precision <= 17; precision++) {
        snprintf(digits, sizeof(digits), "%.*g", precision, n);
        if (strtod(digits, NULL) == n) break;
    }
    jw_raw(jw, digits);
}

static void jr_write_value(const jr_value *value, json_writer *jw,
                           const char *item_sep, const char *key_sep)
{
    if (!value) { jw_raw(jw, "null"); return; }
    switch (value->type) {
        case JR_NULL:   jw_raw(jw, "null"); break;
        case JR_BOOL:   jw_raw(jw, value->as.boolean ? "true" : "false"); break;
        case JR_NUMBER: jr_write_number(jw, value->as.number); break;
        case JR_STRING: jw_string(jw, value->as.string); break;
        case JR_ARRAY:
            jw_raw(jw, "[");
            for (int i = 0; i < value->as.array.count; i++) {
                if (i) jw_raw(jw, item_sep);
                jr_write_value(value->as.array.items[i], jw, item_sep, key_sep);
            }
            jw_raw(jw, "]");
            break;
        case JR_OBJECT:
            jw_raw(jw, "{");
            for (int i = 0; i < value->as.object.count; i++) {
                if (i) jw_raw(jw, item_sep);
                jw_string(jw, value->as.object.members[i].key);
                jw_raw(jw, key_sep);
                jr_write_value(value->as.object.members[i].value, jw, item_sep, key_sep);
            }
            jw_raw(jw, "}");
            break;
    }
}

void jr_to_json(const jr_value *value, json_writer *jw)
{
    jr_write_value(value, jw, ",", ":");
}

void jr_to_json_tojson(const jr_value *value, json_writer *jw)
{
    jr_write_value(value, jw, ", ", ": ");
}
