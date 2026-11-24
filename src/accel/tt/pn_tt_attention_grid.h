/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_grid.h
 * @brief PREFILL ATTENTION ON THE GRID CORES: the
 *        48 prefill-grid cores sit idle while an attention layer's band loop runs on 12. Using
 *        them means parking the grid, arming the attention program on its cores, retiring it,
 *        and booting the grid again — once per attention layer per pass.
 *
 * This file is the PROBE of that swap: what one arm + retire + park of the attention program
 * over the grid's cores costs, and whether it wedges. It runs inside the grid's own wedge gate
 * (`PN_TT_PREFILL_GRID_CYCLES=N` with `PN_TT_GRID_CYCLE_ATTN=1`), before the first prefill,
 * and serves nothing.
 *
 * @status Probe and C (2026-11-18). C is off by default (`PN_TT_ATTN_GRID_DEPTH=0`).
 * @issues Re-booting parked grid cores is a standing wedge hazard (pn_tt_matmul_group.c
 *         pn_tt_grid_park); this probe exists to count it.
 * @todo None
 */
#ifndef PN_TT_ATTENTION_GRID_H
#define PN_TT_ATTENTION_GRID_H

#ifdef USE_TT

#include <stdint.h>

#include "pn_tt_device.h"
#include "pn_tt_residency.h"

/** The most grid cores C borrows (the grid's own cap). */
#define PN_TT_ATTN_GRID_MAX_CORES 64u

/** Seconds spent in each half of one attention swap over the grid's cores. */
typedef struct pn_tt_attention_grid_cycle_time {
    double arm_seconds;  /**< load + bind + start the attention program on every core */
    double park_seconds; /**< retire every residency, then hold its RISCs in reset */
} pn_tt_attention_grid_cycle_time;

/** `PN_TT_GRID_CYCLE_ATTN=1`: the grid's wedge gate also cycles the attention program. */
bool pn_tt_attention_grid_cycle_wanted(void);

/**
 * Arm the attention program on each of `count` PARKED cores, retire it, and park the cores
 * again (every RISC held in reset). Needs on-card attention armed (its L1 shape and KV formats).
 *
 * @return 0 when every core armed, retired and parked; -1 at the first failure, with every
 *         core it armed retired and, when it exited, parked (a core whose triad did not exit
 *         is left alone and named)
 */
int pn_tt_attention_grid_cycle(pn_tt_device *device, const uint32_t *cores, uint32_t count,
                               pn_tt_attention_grid_cycle_time *out_time);

/**
 * C (plan 3.2) and L1a: arm the attention program on each of `count` lent cores, as extra prefill
 * attention members after the head-split ones and after any already borrowed (it appends).
 *
 * @return members armed by this call (it stops at the first core that refuses; those armed
 *         still serve)
 */
uint32_t pn_tt_attention_grid_arm(pn_tt_device *device, const uint32_t *cores, uint32_t count);

/** Retire every borrowed member and hold its RISCs in reset, so the grid may boot again.
 *  `stuck` (NULL: not wanted) receives each core whose member did not exit, `*stuck_count` how
 *  many; `stuck` holds PN_TT_ATTN_GRID_MAX_MEMBERS.
 *  @return 0, or -1 when a member did not exit (its core is named and must not be booted) */
int pn_tt_attention_grid_park_all(pn_tt_device *device, uint32_t *stuck, uint32_t *stuck_count);

/** Prefill attention members serving now: the head-split members plus the borrowed ones. */
uint32_t pn_tt_attention_queue_members(const pn_tt_device *device);

/** Member `m` of pn_tt_attention_queue_members()' set: 0 the attention residency, then the
 *  head-split members, then the borrowed grid members. */
pn_tt_residency *pn_tt_attention_queue_member(pn_tt_device *device, uint32_t m);

#endif /* USE_TT */

#endif /* PN_TT_ATTENTION_GRID_H */
