/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_writer.h
 * @brief Growable string builder with JSON string escaping. Used to shape
 *        OpenAI response bodies and SSE chunks without a JSON library.
 *
 * @status Complete for the serve layer's output needs.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_JSON_WRITER_H
#define PN_SERVE_JSON_WRITER_H

#include <stdbool.h>
#include <stddef.h>

/* Growable byte buffer. Zero-initialize with jw_init(). */
typedef struct {
    char  *data;   /* NUL-terminated buffer (owned) */
    size_t len;    /* bytes used, excluding the NUL */
    size_t cap;    /* allocated capacity */
    bool   oom;    /* set if any append failed (allocation) */
} json_writer;

/** Initialize an empty writer. Returns false on allocation failure. */
bool jw_init(json_writer *jw);

/** Release the writer's buffer. NULL-safe; leaves the struct zeroed. */
void jw_free(json_writer *jw);

/** Append a raw (already-escaped / structural) NUL-terminated fragment. */
void jw_raw(json_writer *jw, const char *text);

/** Append `len` raw bytes. */
void jw_raw_n(json_writer *jw, const char *bytes, size_t len);

/** Append a formatted fragment (printf-style). */
void jw_printf(json_writer *jw, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/**
 * Append a JSON string literal: a leading '"', the escaped contents of `text`,
 * and a trailing '"'. Escapes control chars, quotes and backslashes per RFC
 * 8259. `text` may be NULL (emits an empty string literal).
 */
void jw_string(json_writer *jw, const char *text);

/**
 * Append the ESCAPED contents of `len` bytes with no surrounding quotes — the
 * body jw_string_n writes between its two '"'.  Escaping is per byte, so a
 * string written in pieces produces the same bytes as one jw_string_n call.
 */
void jw_escaped_n(json_writer *jw, const char *text, size_t len);

/** As jw_string but for a bounded (not necessarily NUL-terminated) buffer. */
void jw_string_n(json_writer *jw, const char *text, size_t len);

#endif /* PN_SERVE_JSON_WRITER_H */
