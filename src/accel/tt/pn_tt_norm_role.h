/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_role.h
 * @brief PERSISTENT NORM ROLE:
 *        one resident program on M cores owns every residual add and RMS norm of a decode
 *        token, with the residual stream h resident in L1. Arm, step, teardown, report.
 *
 * WHAT CHANGES FOR THE FORWARD. With the role armed, a token's 129 norms and 128 residual adds
 * become 129 role steps (foundry kernels/resident_norm_role_protocol.h): step 0 LOADs the
 * embedding row into h, every later step adds the projection the forward just wrote into
 * `project` to h and norms it into `normed`. The residual step itself issues nothing (no ADD,
 * no slot flip) and the out-projection runs unfused (no S3a): the role does the add.
 *
 * BRIDGE MODE: each step is one descriptor a member on the role's own resident queues, issued
 * like any split family's (S1 group pull, S2 precondition), so the dispatch recorder replays it
 * and every producer and consumer keeps today's precondition and completion format.
 *
 * EXACT: every tile sees today's verbs, operands, order and pack points (foundry
 * tests/test_norm_role.c); the card gate compares md5 and full logits.
 *
 * KNOBS
 *   `PN_TT_PERSIST_NORM`        0 off (default), 1 on (plain single-row decode only: no MTP,
 *                               no attention compare, a dense FFN)
 *   `PN_TT_PERSIST_NORM_CORES`  the M member cores (default 0-3,7-10: grid cores, so the
 *                               prefill grid must start at 16 — PN_TT_PREFILL_GRID_FIRST=16)
 *   `PN_TT_PERSIST_NORM_EXCHANGE`  the exchange (foundry spec §11.13): whole (default,
 *                               §11.12's whole pages), faces, stagger, or faces,stagger
 *   RED `PN_TT_RED=norm_role_gamma_next`: every step's table entry names the NEXT step's
 *   constants (legal addresses, wrong gamma): the output MUST differ.
 *
 * @status P0 (plan P0.3). Off by default. Card gate pending.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_NORM_ROLE_H
#define PN_TT_NORM_ROLE_H

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_norm_role_plan.h"

#ifdef USE_TT

#include "tt/ft_tt_op_norm_role.h"

typedef struct pn_tt_device pn_tt_device;
typedef struct pn_tt_activation pn_tt_activation;
typedef struct pn_tt_norm_constants pn_tt_norm_constants;

/** Is the role armed (and not broken) on `device`? */
bool pn_tt_norm_role_armed(const pn_tt_device *device);

/**
 * Arm the role: boot `norm_role` on every member core, place its CBs, start it, write each
 * member's schedule table and publish the replica map to it. Idempotent for the same schedule.
 * A refusal leaves nothing running and the role off for the process (logged).
 *
 * @param schedule  the token's steps (pn_tt_norm_role_build_schedule)
 * @return 0 armed, -1 refused
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_norm_role_arm(pn_tt_device *device, const ft_tt_op_norm_role_schedule *schedule);

/**
 * Issue step `step` and await it (member 0's group pull covers the rest). At step 0 a member
 * the idle retire stopped re-arms the whole role first.
 *
 * @return 0, or -1 (the role is torn down and stays off; the caller's forward fails)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_norm_role_step(pn_tt_device *device, uint32_t step);

/** Stop and release every member and leave the role table; logs the busy clock and the arm /
 *  teardown wall time. */
void pn_tt_norm_role_teardown(pn_tt_device *device);

/**
 * Re-arm a role that was armed and torn down (a mode switch, pn_tt_role_table.h) from its
 * saved schedule.
 *
 * @return 0 armed, -1 never armed, broken, or the arm refused
 */
int pn_tt_norm_role_resume(pn_tt_device *device);

/** Members' residencies for the replica map and core-holder walks (NULL past the count). */
const struct pn_tt_residency *pn_tt_norm_role_member(const pn_tt_device *device, uint32_t m);

/** Mutable member (the replica map publishes to it). */
struct pn_tt_residency *pn_tt_norm_role_member_mut(pn_tt_device *device, uint32_t m);

/**
 * One schedule step from the forward's strips and norm constants (the model code builds the
 * table, this resolves banks to NoC coordinates).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_norm_role_fill_step(const pn_tt_device *device, ft_tt_op_norm_role_step *step,
                               uint32_t kind, const pn_tt_activation *input,
                               const pn_tt_norm_constants *constants,
                               const pn_tt_activation *output);

#endif /* USE_TT */
#endif /* PN_TT_NORM_ROLE_H */
