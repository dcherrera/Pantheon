/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_param_empty_value_rule.h
 * @brief The one question both tool-call grammars ask of a parameter's JSON
 *        Schema: may this parameter's value be EMPTY?
 *
 * WHY THIS IS ITS OWN MODULE.  The XML dialect (tool_grammar.c) and the JSON
 * dialect (tool_grammar_json.c) each own a value walk, and each needs the same
 * verdict at the same moment — the instant the model tries to CLOSE a value that
 * holds nothing.  Every time this pair has been given two copies of one rule the
 * copies have drifted, and the drift is the defect: the whitespace guard was
 * fixed on one dialect and not the other, and the empty guard was later removed
 * from XML while JSON kept an unconditional one, so the same schema meant two
 * different things depending on which model answered.  One function, compiled
 * into both, makes that class of bug unrepresentable.
 *
 * @status Complete.  Mirrors the agent client's own rule byte for byte.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_TOOL_PARAM_EMPTY_VALUE_RULE_H
#define PANTHEON_TOOL_PARAM_EMPTY_VALUE_RULE_H

#include <stdbool.h>

#include "json_reader.h"

/**
 * Does this parameter's schema forbid an empty value?
 *
 * The rule, and it is DERIVED not invented — it is the same predicate the
 * client enforces on arrival (the client's schema code,
 * param_has_length_floor), read off the same two keywords:
 *
 *   floored  ⇔  the declared `type` is absent or is not "string"
 *               OR  `minLength` is a number ≥ 1
 *
 * So `{"type":"string","minLength":1}` is floored and `{"type":"string"}` is
 * not, which is exactly the distinction between `read`'s `path` (empty is the
 * empty-path wedge) and `edit`'s `new_string` (empty IS the deletion).  Both
 * are `required`; requiredness is therefore NOT the predicate and is not
 * consulted here.  Emptiness is a question about what the parameter MEANS, and
 * `minLength` is the schema's own way of saying it.
 *
 * A non-string type is floored regardless of `minLength` because it has no
 * empty form to permit: `""` arriving where an integer belongs is a malformed
 * call, not a permitted one.  An untyped parameter is floored for the same
 * reason the client floors it — nothing advertised it as able to mean empty.
 *
 * @param prop  The parameter's schema object (properties[key]); NULL/non-object
 *              is treated as untyped, i.e. floored.
 * @return      true if an empty value must be refused for this parameter.
 *
 * Thread-safe: yes (pure read of a caller-owned tree)
 * Reentrant: yes
 */
bool tool_param_forbids_empty_value(const jr_value *prop);

#endif /* PANTHEON_TOOL_PARAM_EMPTY_VALUE_RULE_H */
