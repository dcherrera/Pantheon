/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_family.h
 * @brief Which session ids belong to one user session: an id, and every
 *        "<id>-<more>" id under it. The rule max_sessions counts by, replica
 *        routing binds by, and a session watch covers.
 *
 * @status None
 * @issues None
 * @todo None
 */
/* WHY A FAMILY
 *   A client holds one session but may send turns under derived ids: Argus
 *   watches the tab's base id while a side request runs on "<base>-new2", and
 *   older clients ran each mode on "<base>-<mode>". Those ids are the same user,
 *   so they share one seat and one replica. Two ids are one family when they are
 *   equal or one is the other followed by '-' and more. Siblings ("x-a", "x-b")
 *   only meet through their base.
 */

#ifndef PN_CORE_SESSION_FAMILY_H
#define PN_CORE_SESSION_FAMILY_H

#include <stdbool.h>
#include <string.h>

/** Is `b` the id `a` followed by '-' and more? */
static inline bool pn_session_id_extends(const char *a, const char *b)
{
    size_t n = strlen(a);

    return n > 0 && strncmp(a, b, n) == 0 && b[n] == '-';
}

/** Are session ids `a` and `b` one family (equal, or one extends the other)? */
static inline bool pn_session_same_family(const char *a, const char *b)
{
    return strcmp(a, b) == 0 || pn_session_id_extends(a, b) ||
           pn_session_id_extends(b, a);
}

#endif
