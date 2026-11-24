/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog_match.h
 * @brief  Match catalog profiles against the box (spec §5.5): every model ×
 *         profile becomes one entry that either fits (free units of the right
 *         type, card rule, memory, host RAM) or carries the reason it does not.
 *         Fitting entries come first, each group in the catalog's rank order.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CATALOG_MATCH_H
#define PN_LOCAL_CATALOG_MATCH_H

#include <stdbool.h>
#include <stdint.h>

#include "catalog_manifest.h"
#include "hw_report.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_CAT_MAX_ENTRIES (PN_CAT_MAX_MODELS * PN_CAT_MAX_PROFILES)

typedef struct {
    const pn_cat_model   *model;
    const pn_cat_profile *profile;
    bool                  fits;
    char                  reason[128];   /* "" when it fits */
} pn_cat_match;

/**
 * Rank every profile of the manifest for this box.
 * @return number of entries written to out (<= max)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_cat_match_profiles(const pn_catalog_manifest *m, const pn_hw_report *hw,
                          uint64_t host_ram, pn_cat_match *out, int max);

/**
 * Whether one profile fits now; reason filled when not.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
bool pn_cat_profile_fits(const pn_cat_profile *p, const pn_hw_report *hw, uint64_t host_ram,
                         char *reason, size_t cap);

/**
 * The same check for a bare hardware request (a remembered load re-checked at
 * boot): enough units of the type on this box, the host RAM floor (0 = none),
 * and free units to place it on now. reason filled when not.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
bool pn_cat_request_fits(const pn_unit_request *r, uint64_t min_host_ram, const pn_hw_report *hw,
                         uint64_t host_ram, char *reason, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CATALOG_MATCH_H */
