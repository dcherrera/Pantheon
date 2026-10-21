/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_name_list.c
 * @brief Implementation of exact comma-separated-list membership. See pn_tt_name_list.h,
 *        which carries why a filter that matches nothing is indistinguishable from a filter
 *        that is off.
 *
 *        No allocation, no I/O, no device — deliberately, so it is provable with no card.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "pn_tt_name_list.h"

#include <string.h>

bool pn_tt_name_list_contains(const char *list, const char *name)
{
    size_t length;

    /* An UNSET knob does not restrict. This is the asymmetry the header names, and it is why
     * a caller can never tell an empty list from a matching one — nor should need to. */
    if (!list || !*list)
        return true;
    if (!name || !*name)
        return false;
    length = strlen(name);
    for (;;) {
        const char *comma = strchr(list, ',');
        size_t span = comma ? (size_t)(comma - list) : strlen(list);

        /* Compared WHOLE, so `render` does not take `render-widen`. */
        if (span == length && strncmp(list, name, span) == 0)
            return true;
        if (!comma)
            return false;
        list = comma + 1;
    }
}

bool pn_tt_name_list_selects(const char *list, const char *name)
{
    /* THE ONE LINE THAT IS THE WHOLE POINT. An unset selector selects NOTHING, where an unset
     * filter admits everything — see the header for the silicon this cost. Everything after
     * this guard is deliberately the SAME walk, so the two can only ever disagree about the
     * empty list and never about a membership. */
    if (!list || !*list)
        return false;
    return pn_tt_name_list_contains(list, name);
}
