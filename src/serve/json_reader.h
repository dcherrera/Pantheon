/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file json_reader.h
 * @brief Minimal recursive-descent JSON parser to a small in-memory DOM.
 *        Just enough to read OpenAI chat-completion requests (objects, arrays,
 *        strings with escapes, numbers, booleans, null). No external deps.
 *
 * @status Parses the request shapes the serve layer needs; not a general-purpose
 *         validator (duplicate keys keep the first, trailing garbage ignored).
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_JSON_READER_H
#define PN_SERVE_JSON_READER_H

#include <stdbool.h>
#include <stddef.h>

#include "json_writer.h"

/* JSON value kinds. */
typedef enum {
    JR_NULL,
    JR_BOOL,
    JR_NUMBER,
    JR_STRING,
    JR_ARRAY,
    JR_OBJECT,
} jr_type;

/* Opaque JSON value node. */
typedef struct jr_value jr_value;

/**
 * Parse a NUL-terminated JSON document into a DOM tree.
 *
 * @param text  UTF-8 JSON document (not modified, not retained).
 * @return      Root value (free with jr_free), or NULL on parse error / OOM.
 *
 * Ownership: caller frees the returned tree via jr_free().
 * Thread-safe: yes (no shared state)
 */
jr_value *jr_parse(const char *text);

/** Recursively free a value tree returned by jr_parse(). NULL-safe. */
void jr_free(jr_value *value);

/** Kind of a value. Returns JR_NULL for a NULL pointer. */
jr_type jr_kind(const jr_value *value);

/**
 * Look up a key in an object. Returns the child value (borrowed, owned by the
 * tree) or NULL if `value` is not an object or the key is absent.
 */
const jr_value *jr_object_get(const jr_value *value, const char *key);

/** Number of elements in an array (0 if not an array). */
int jr_array_len(const jr_value *value);

/** Number of members in an object (0 if not an object). */
int jr_object_member_count(const jr_value *value);

/** Key of member `index` (borrowed), preserving source order, or NULL. */
const char *jr_object_key_at(const jr_value *value, int index);

/** Value of member `index` (borrowed), preserving source order, or NULL. */
const jr_value *jr_object_value_at(const jr_value *value, int index);

/** Element `index` of an array (borrowed), or NULL if out of range. */
const jr_value *jr_array_get(const jr_value *value, int index);

/**
 * String payload of a JR_STRING value (unescaped, NUL-terminated, borrowed),
 * or NULL if `value` is not a string.
 */
const char *jr_as_string(const jr_value *value);

/** Numeric payload; returns `fallback` if `value` is not a number. */
double jr_as_number(const jr_value *value, double fallback);

/** Boolean payload; returns `fallback` if `value` is not a bool. */
bool jr_as_bool(const jr_value *value, bool fallback);

/**
 * Serialize a value tree back to compact JSON, appending to `jw`. Integral
 * numbers render without a decimal point (so a tool schema round-trips cleanly).
 * This is the C equivalent of Jinja's `value | tojson`, used to render tool
 * definitions into the prompt.
 */
void jr_to_json(const jr_value *value, json_writer *jw);

#endif /* PN_SERVE_JSON_READER_H */
