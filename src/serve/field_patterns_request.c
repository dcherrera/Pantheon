/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Shape check of the request field `field_patterns` — see
 * field_patterns_request.h.  Every pattern is compiled once here (and thrown
 * away) so a pattern the grammar could not build is refused with a reason
 * instead of being dropped mid-generation.
 *
 * Pure C, libc-only.
 */

#include "field_patterns_request.h"

#include <stdio.h>

#include "field_pattern.h"

int field_patterns_validate(const jr_value *field, char *err, size_t err_cap) {
    if (jr_kind(field) == JR_NULL) return 0;
    if (jr_kind(field) != JR_OBJECT) {
        snprintf(err, err_cap, "'field_patterns' must be an object of tool name -> "
                 "{property: pattern}");
        return -1;
    }
    int n_tools = jr_object_member_count(field);
    for (int t = 0; t < n_tools; t++) {
        const char *tool = jr_object_key_at(field, t);
        const jr_value *props = jr_object_value_at(field, t);
        if (jr_kind(props) != JR_OBJECT) {
            snprintf(err, err_cap, "'field_patterns.%s' must be an object of property "
                     "name -> pattern", tool ? tool : "");
            return -1;
        }
        int n_props = jr_object_member_count(props);
        for (int k = 0; k < n_props; k++) {
            const char *prop = jr_object_key_at(props, k);
            const char *pattern = jr_as_string(jr_object_value_at(props, k));
            char why[128] = "pattern is not a string";
            field_pattern *compiled = pattern ? fp_compile(pattern, why, sizeof(why)) : NULL;
            if (!compiled) {
                snprintf(err, err_cap, "'field_patterns.%s.%s': %s", tool ? tool : "",
                         prop ? prop : "", why);
                return -1;
            }
            fp_free(compiled);
        }
    }
    return 0;
}
