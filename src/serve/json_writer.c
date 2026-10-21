/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_writer.c
 * @brief Implementation of the growable JSON string builder (see json_writer.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "json_writer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Ensure at least `extra` more bytes (plus the trailing NUL) fit. */
static bool jw_reserve(json_writer *jw, size_t extra)
{
    if (jw->oom) return false;
    size_t need = jw->len + extra + 1;
    if (need <= jw->cap) return true;
    size_t cap = jw->cap ? jw->cap : 256;
    while (cap < need) cap *= 2;
    char *grown = realloc(jw->data, cap);
    if (!grown) { jw->oom = true; return false; }
    jw->data = grown;
    jw->cap = cap;
    return true;
}

bool jw_init(json_writer *jw)
{
    jw->data = malloc(256);
    if (!jw->data) { jw->oom = true; jw->len = jw->cap = 0; return false; }
    jw->data[0] = '\0';
    jw->len = 0;
    jw->cap = 256;
    jw->oom = false;
    return true;
}

void jw_free(json_writer *jw)
{
    if (!jw) return;
    free(jw->data);
    jw->data = NULL;
    jw->len = jw->cap = 0;
    jw->oom = false;
}

void jw_raw_n(json_writer *jw, const char *bytes, size_t len)
{
    if (!jw_reserve(jw, len)) return;
    memcpy(jw->data + jw->len, bytes, len);
    jw->len += len;
    jw->data[jw->len] = '\0';
}

void jw_raw(json_writer *jw, const char *text)
{
    if (text) jw_raw_n(jw, text, strlen(text));
}

void jw_printf(json_writer *jw, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0 || !jw_reserve(jw, (size_t)n)) { va_end(ap2); return; }
    vsnprintf(jw->data + jw->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    jw->len += (size_t)n;
}

void jw_escaped_n(json_writer *jw, const char *text, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        switch (c) {
            case '"':  jw_raw_n(jw, "\\\"", 2); break;
            case '\\': jw_raw_n(jw, "\\\\", 2); break;
            case '\b': jw_raw_n(jw, "\\b", 2);  break;
            case '\f': jw_raw_n(jw, "\\f", 2);  break;
            case '\n': jw_raw_n(jw, "\\n", 2);  break;
            case '\r': jw_raw_n(jw, "\\r", 2);  break;
            case '\t': jw_raw_n(jw, "\\t", 2);  break;
            default:
                if (c < 0x20) {
                    jw_printf(jw, "\\u%04x", c);
                } else {
                    jw_raw_n(jw, (const char *)&c, 1);
                }
                break;
        }
    }
}

void jw_string_n(json_writer *jw, const char *text, size_t len)
{
    jw_raw_n(jw, "\"", 1);
    jw_escaped_n(jw, text, len);
    jw_raw_n(jw, "\"", 1);
}

void jw_string(json_writer *jw, const char *text)
{
    jw_string_n(jw, text ? text : "", text ? strlen(text) : 0);
}
