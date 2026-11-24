/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_conv_share.h
 * @brief ONE conv weight for both GDN families (option D of the Qwen3.8 fit): the decode CONV reads the chunkwise family's conv
 *        constant copy 0 instead of its own 5 MiB a layer (0.234 GiB on Qwen3.8-27B).
 *
 * WHY IT IS BIT-EXACT
 *   - Same layout: both are `taps` tap-major planes of `conv_dim/32` fp32 tiles
 *     (`ftg_conv_plane_tile()`); the chunkwise constant only adds band tiles AFTER the planes.
 *   - Same row 0: the chunkwise planes are row-replicated and their row 0 is the decode plane
 *     byte for byte (`pn_tt_gdnc_stage_weight_column()` vs `pn_tt_gdn_stage_plane()`).
 *   - The decode CONV never reads rows 1..31: under `PN_TT_GDN_CONV_STRIP` (default on) every
 *     CONV read is row 0 only (`conv_read()` in foundry `resident_gdn_reader.c`). With the
 *     strip off it reads whole tiles, so sharing is NOT taken then.
 *
 * WHEN IT ENGAGES
 *   Only under the qwen35 FIT POLICY (main's placement does not fit the card), the strip on,
 *   the chunkwise family wanted and armed, and `PN_TT_GDN_CONV_SHARE` not `0`. Every other
 *   load (Ornith, Qwen3.6, Bonsai) keeps main's placement byte for byte.
 *
 * PLACEMENT
 *   The decode arm skips each layer's weight but still advances the round-robin cursor, so the
 *   window and state (and everything allocated after them) land in the same banks as without
 *   sharing: each bank simply keeps the weights it would have held free (30 MiB a bank on
 *   Qwen3.8). If the chunkwise family does not arm, the weights are allocated after all.
 *
 * RED: `PN_TT_RED=gdn_conv_share_shift` points the decode weight one tap plane into copy 0;
 * decode output must DIFFER (prefill, which reads its own operand, does not move).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GDN_CONV_SHARE_H
#define PN_TT_GDN_CONV_SHARE_H

#ifdef USE_TT

#include <stdbool.h>

#include "pn_tt_device.h"

/** `0` declines sharing even under the fit policy (the A/B and parity arm). Default on. */
#define PN_TT_GDN_CONV_SHARE_ENV "PN_TT_GDN_CONV_SHARE"

/** What `gdn_region[].weight` is. */
typedef enum pn_tt_gdn_conv_share_state {
    PN_TT_GDN_CONV_OWN = 0,     /**< main: each layer allocated its own weight */
    PN_TT_GDN_CONV_PENDING = 1, /**< sharing asked for; weights not yet bound */
    PN_TT_GDN_CONV_SHARED = 2,  /**< every layer's weight is the chunkwise copy 0 */
} pn_tt_gdn_conv_share_state;

/**
 * Should this load share the conv weight? `fit_policy` AND the strip on AND the chunkwise
 * family not declined AND `PN_TT_GDN_CONV_SHARE` not `0`. Reads the environment only.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
bool pn_tt_gdn_conv_share_wanted(bool fit_policy);

/** Ask (or not) for sharing. Call BEFORE `pn_tt_device_arm_gdn()`, which reads it. */
void pn_tt_gdn_conv_share_request(pn_tt_device *device, bool share);

/**
 * Bind the decode weights once the chunkwise family has armed (or failed to): alias every
 * layer to its copy 0, or allocate the decode family's own weights after all. A no-op unless
 * sharing is pending. Call before the weights are uploaded.
 *
 * @return 0, or -1 when the fallback allocation fails (the GDN block then cannot stage)
 */
int pn_tt_gdn_conv_share_bind(pn_tt_device *device);

/** Does the decode CONV read the chunkwise constant? The uploader then writes it ONCE, through
 *  `pn_tt_gdnc_conv_weight_upload()`, and a failure there is fatal to the GDN block. */
bool pn_tt_gdn_conv_weight_is_shared(const pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_GDN_CONV_SHARE_H */
