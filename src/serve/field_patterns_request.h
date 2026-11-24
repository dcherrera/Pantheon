/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file field_patterns_request.h
 * @brief Shape check of the request field `field_patterns` (a pattern rule on
 *        one tool field), run at request parse so a bad field is an HTTP 400
 *        before any generation starts.
 *
 * The field: an object of tool name -> object of property name -> pattern
 * string (field_pattern.h syntax), e.g.
 *     "field_patterns": {"read": {"finding": "-|[^\\n]+ from: ..."}}
 * Tool and property names are NOT checked against the request's tools: an
 * unknown one is ignored by the grammar build (tg_apply_field_patterns /
 * tgj_apply_field_patterns).  What is checked here is everything that would
 * otherwise be silently dropped: the field (when present and not null) must be
 * an object, each tool entry an object, each pattern a string that compiles.
 *
 * @status Called from openai_chat.c parse_chat_request.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_FIELD_PATTERNS_REQUEST_H
#define PANTHEON_FIELD_PATTERNS_REQUEST_H

#include <stddef.h>

#include "json_reader.h"

/**
 * Check `field` (the request's `field_patterns` value; NULL or JSON null = absent).
 * Returns 0 when it is absent or well-formed; -1 with a one-line message naming
 * the tool, the property and the reason in `err` otherwise.
 */
int field_patterns_validate(const jr_value *field, char *err, size_t err_cap);

#endif /* PANTHEON_FIELD_PATTERNS_REQUEST_H */
