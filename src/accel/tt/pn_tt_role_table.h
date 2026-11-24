/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_role_table.h
 * @brief THE ROLE TABLE: which persistent role holds which core in which mode (decode,
 *        prefill), and the mode switch — tear down the roles the next mode does not run, then
 *        re-arm the ones it does, each timed ("reload
 *        only the cores whose role changes, once per turn"). Pure C, no card: a role registers
 *        its cores and two callbacks; tests/test_tt_role_table.c proves the order.
 *
 * ORDER OF A SWITCH (decode -> prefill, and back): every role whose modes exclude the target
 * is torn down FIRST, in reverse registration order, and only then is every role whose modes
 * include it re-armed, in registration order — so a core moving from a decode role to a
 * prefill role is never claimed twice. A claim that overlaps another role's core in a shared
 * mode is refused.
 *
 * Owners: the persistent norm role (decode) registers here; the prefill rework adds its
 * prefill core map and calls pn_tt_role_table_enter() at the prefill entry and exit.
 *
 * @status P0 (plan P0.3), first user the norm role.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ROLE_TABLE_H
#define PN_TT_ROLE_TABLE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_TT_ROLE_MODE_DECODE 1u
#define PN_TT_ROLE_MODE_PREFILL 2u
#define PN_TT_ROLE_TABLE_MAX_ROLES 8u
#define PN_TT_ROLE_TABLE_MAX_CORES 64u

/** A role's callbacks: tear down / re-arm on `user`, with the switch's per-call `context`
 *  (NULL when the caller has none). Arm returns 0 armed, 1 DECLINED (the role stays down and
 *  it is not a failure: e.g. a prompt too short for it), -1 failed. */
typedef void (*pn_tt_role_teardown_fn)(void *user);
typedef int (*pn_tt_role_arm_fn)(void *user, void *context);

/**
 * Register role `name` on `cores` for `modes`. Re-registering the same name replaces it.
 *
 * @return 0, or -1 (table full, a core outside 0..139, or a core another role holds in a mode
 *         both run)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_role_table_claim(const char *name, const uint32_t *cores, uint32_t count,
                           uint32_t modes, pn_tt_role_teardown_fn teardown, pn_tt_role_arm_fn arm,
                           void *user);

/** Forget role `name` (its cores become free). */
void pn_tt_role_table_release(const char *name);

/** The role holding `core` in `mode`, or NULL. */
const char *pn_tt_role_table_holder(uint32_t core, uint32_t mode);

/**
 * Switch to `mode`: tear down the roles that do not run in it (reverse order), then re-arm the
 * ones that do and are not armed (registration order), each arm given `context`. A declined
 * arm leaves its role down (it is asked again at the next switch). Seconds spent go to the
 * out pointers.
 *
 * @return 0, or -1 when a re-arm failed (that role stays down; the others are switched)
 */
int pn_tt_role_table_enter(uint32_t mode, void *context, double *out_teardown_seconds,
                           double *out_arm_seconds);

/** The mode last entered (0 = none yet). */
uint32_t pn_tt_role_table_mode(void);

/** Reset the table (tests). */
void pn_tt_role_table_clear(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_ROLE_TABLE_H */
