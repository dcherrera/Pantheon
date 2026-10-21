/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_name_list.h
 * @brief Exact membership in a comma-separated name list — the shape every `PN_TT_*` env
 *        knob that names families, cores or arms is parsed with.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS ITS OWN FILE
 * ---------------------------------------------------------------------------
 * A list filter that matches NOTHING and a filter that is simply off look identical from the
 * outside: both leave the default behaviour in place. So a knob whose whole job is to
 * restrict something — `PN_TT_ONCARD_BATCH_FAMILY` restricts the queue-depth instrument to
 * the families whose descriptors are idempotent — fails SILENTLY into "the instrument did
 * nothing", and the reading is "the lever is worth nothing" rather than "the filter is
 * broken". That is the shape of every vacuous gate recorded on this project.
 *
 * Pulling the four lines out means they are provable with no card, no device and no
 * `USE_TT`, which is the only way that failure gets caught before it produces a number.
 *
 * ---------------------------------------------------------------------------
 * WHAT "EXACT" MEANS, AND WHY IT IS NOT A SUBSTRING TEST
 * ---------------------------------------------------------------------------
 * The names it matches are residency names, and two of them are `render-widen` and
 * `render-narrow`. A substring test on `render` would take both — which is sometimes what a
 * caller wants and never what it SAID. Each comma-separated element must equal the whole
 * name, so a caller that wants both writes both.
 *
 * No allocation, no I/O, no device.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_NAME_LIST_H
#define PN_TT_NAME_LIST_H

#include <stdbool.h>

/**
 * Is `name` one of `list`'s comma-separated elements, compared whole?
 *
 * @param list  e.g. "eltwise,render-widen"; NULL or "" means EVERY name (an unset knob does
 *              not restrict), which is the one asymmetry in here and the reason it is stated
 *              in the signature rather than left to a caller's `if`
 * @param name  the candidate; NULL or "" is never a member of a non-empty list
 * @return      true when `list` is empty, or when some element equals `name` exactly
 *
 * Thread-safe: yes (pure)   Reentrant: yes   Signal-safe: yes
 */
bool pn_tt_name_list_contains(const char *list, const char *name);

/**
 * Does `list` SELECT `name`? The exact dual of the above: **an unset list selects NOTHING.**
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS, AND IT IS NOT A STYLE PREFERENCE
 * ---------------------------------------------------------------------------
 * `pn_tt_name_list_contains()` is a FILTER — "unset means every family" — and it answers
 * **true** for a NULL list by design. A RED ARM is a SELECTOR, and arming one with the
 * filter INVERTS it: with the knob unset the arm is armed, on every run, for everyone.
 *
 * That is not hypothetical. `PN_TT_RED=residency_await_early` was armed with the filter, so
 * every unnamed run checked completion against `expected + 1` and **every descriptor of
 * every family was refused at token 0**, with "completed_count does not match the number
 * enqueued". It cost four card sessions and a cross-agent bisect, and was mis-attributed to
 * an unrelated commit on the way. Measured on silicon 2026-08-25.
 *
 * So the two live side by side with OPPOSITE unset behaviour, each named for what it does,
 * and `tests/test_tt_name_list.c` asserts the asymmetry in both directions — because the
 * defect is invisible at the call site: the two calls read identically.
 *
 * @param list  NULL or "" selects NOTHING and returns false
 * @param name  NULL or "" is never selected
 * @return      true only when some element of a NON-EMPTY `list` equals `name` exactly
 *
 * Thread-safe: yes (pure)   Reentrant: yes   Signal-safe: yes
 */
bool pn_tt_name_list_selects(const char *list, const char *name);

#endif /* PN_TT_NAME_LIST_H */
