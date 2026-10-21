/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_param_empty_value_rule.c
 * @brief Implementation of the shared may-this-value-be-empty schema rule.
 *
 * Pure C, libc-only, one function.  See the header for the rule and why it is
 * shared rather than duplicated per dialect.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#include "tool_param_empty_value_rule.h"

#include <string.h>

/* Sentinel distinct from any legal minLength, so "absent" and "0" stay
 * distinguishable through jr_as_number's fallback. */
#define MIN_LENGTH_ABSENT (-1.0)

bool tool_param_forbids_empty_value(const jr_value *prop)
{
    const char *type = jr_as_string(jr_object_get(prop, "type"));

    /* Nothing advertised as non-string has an empty form to permit. */
    if (!type || strcmp(type, "string") != 0)
        return true;

    const jr_value *min = jr_object_get(prop, "minLength");
    if (jr_kind(min) != JR_NUMBER)
        return false;

    /* ONLY THE FLOOR'S EXISTENCE IS COMPILED, never its magnitude.  A walk that
     * enforced minLength: 12 would have to count characters BEFORE the parser
     * trims them, so an interior-space value like "a b" would be measured
     * differently by the grammar and by the tool — and being STRICTER than the
     * schema makes a legal call impossible to emit, which is the one outcome
     * this module's degrade-don't-reject precedent (TG_REQ_MAX_KEYS,
     * TGJ_ENUM_MAX_MEMBERS) forbids.  minLength ≥ 2 therefore degrades to ≥ 1;
     * the remainder is the client's to check, where the whole value is in hand.
     * Every floor in the client's schemas is 1, so nothing is lost today. */
    return jr_as_number(min, MIN_LENGTH_ABSENT) >= 1.0;
}
