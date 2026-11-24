/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_options.h
 * @brief  Smart load options (spec "Load options", plan Phase L): the context x
 *         users choices a catalog profile offers, which of them fit the units
 *         free NOW, the recommended one, and the check a load request must pass.
 *
 *         need(ctx, users) = fixed + ctx * ctx_per_token
 *                            + users * (per_session + ctx * kv_per_token),
 *         device bytes per unit, from the profile's measured memory model. An
 *         option fits when free units of the profile's type (and card rule) each
 *         have need + PN_LOAD_FIT_MARGIN_BYTES of device memory free. A unit's
 *         device free = its reported mem_free + hw->die_reserve (mem_free is the
 *         die budget, total - reserve, minus placements).
 *
 *         Grid: context on power-of-two steps from 4096 inside [ctx_min, ctx_max],
 *         users 1, 2, 4, 8, ... up to sessions_max; plus the profile's own load
 *         settings (the recommended default). Pure functions: no I/O, no locks.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_LOAD_OPTIONS_H
#define PN_LOCAL_LOAD_OPTIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../core/engine.h"
#include "catalog_manifest.h"
#include "hw_report.h"
#include "unit_place.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_LOAD_OPTIONS_MAX      64
/* Device headroom kept above the measured model on every unit. */
#define PN_LOAD_FIT_MARGIN_BYTES ((uint64_t)256 << 20)

typedef struct {
    int      ctx_cap;          /* tokens per user */
    int      max_sessions;     /* users at once */
    uint64_t need_per_unit;    /* device bytes per unit, from the memory model */
    bool     fits;             /* on the units free now */
    bool     recommended;      /* the one to preselect (at most one) */
} pn_load_option;

/**
 * Device bytes one unit needs for (ctx, sessions).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
uint64_t pn_load_option_need(const pn_cat_load_options *o, int ctx, int sessions);

/**
 * The profile's choices (fits/recommended false), sorted by context then users.
 * A profile without options has one: its fixed load settings.
 * @return how many (<= max)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_load_options_grid(const pn_cat_profile *p, pn_load_option *out, int max);

/**
 * The unit request for one choice: the profile's hardware, min_mem_per_unit
 * raised to need + margin - reserve (budget terms, what hw units report).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
pn_unit_request pn_load_option_request(const pn_cat_profile *p, const pn_load_option *o,
                                       const pn_hw_report *hw);

/**
 * The grid with `fits` set against `hw` and one `recommended`: the profile's
 * own settings when they fit, else the biggest fitting choice not above them,
 * else the smallest fitting one. A profile without options fits as today
 * (pn_unit_place on its hardware request).
 * @return how many (<= max)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_load_options_fit(const pn_cat_profile *p, const pn_hw_report *hw, pn_load_option *out,
                        int max);

/**
 * Resolve a load request's ctx_cap / max_sessions (0 = absent). Both absent:
 * the recommended choice. One absent: the profile's own value for it. The
 * result must be a grid choice that fits now; else -1 with a plain reason
 * ("65536 tokens x 2 users needs 11.9 GB on each k80-gpu; the free ones have
 * at most 11.2 GB", "16000 tokens is not an offered context (8192, 16384)").
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_load_options_choose(const pn_cat_profile *p, const pn_hw_report *hw, int ctx,
                           int sessions, pn_load_option *chosen, char *why, size_t cap);

/**
 * The serve's KV format (pn_kv_format, engine.h) for a profile's options.kv_type:
 * "q4" and "fp32" are explicit choices the serve keeps (the memory ladder never
 * switches them); anything else (bf16 on the p150, no options) is the serve default.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_local_kv_format_of(const char *kv_type);

/**
 * The PN_TT_KV_TYPE a load on `unit_type` sets for its profile's options.kv_type: the
 * name of a block-float KV ("k8v4", "bfp8", ...) on a p150; NULL for bf16, any other
 * value and any other unit type (the process's PN_TT_KV_TYPE, unset = bf16, as before).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
const char *pn_local_tt_kv_type_of(const char *unit_type, const char *kv_type);

/**
 * The PN_TT_SPEC a load on `unit_type` sets for its profile's MTP fields, written to
 * `out`: "0" for mtp false (no MTP head, plain decode), "<n>" for mtp_drafts n, on a
 * p150 only. NULL (out untouched) for mtp true without a draft count and any other unit
 * type: the process's PN_TT_SPEC / the build's default, as before.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
const char *pn_local_tt_spec_of(const char *unit_type, bool mtp, int mtp_drafts, char *out,
                                size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_LOAD_OPTIONS_H */
