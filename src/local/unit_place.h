/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   unit_place.h
 * @brief  Unit placement (spec §4.5): turn a profile's request ("1 k80-gpu",
 *         "2 k80-gpu same card", "1 p150", "1 cpu") into free units of the box,
 *         i.e. the `dies` of the load — or a reason it does not fit.
 *
 *         A free unit has no model on it and is not held by another program
 *         (pn_hw_unit_available). One-unit K80 requests prefer a unit
 *         whose card mate is also free: two busy GPUs on one K80 board throttle
 *         to ~398 MHz, and speed is never traded for capacity.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_UNIT_PLACE_H
#define PN_LOCAL_UNIT_PLACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hw_report.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_UNIT_PLACE_MAX 8

typedef struct {
    char     unit_type[PN_HW_TYPE_MAX];  /* "k80-gpu", "p150", "cpu" */
    int      count;                      /* units wanted (>= 1) */
    bool     same_card;                  /* all units on one card */
    int      per_card;                   /* > 0: whole groups of this many units per card
                                          * (count 4, per_card 2 = two K80 boards, both
                                          * GPUs of each); 0 = no rule */
    uint64_t min_mem_per_unit;           /* bytes; 0 = no minimum */
} pn_unit_request;

/**
 * Count the free units of `type` with at least `min_mem` free.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int pn_unit_count_free(const pn_hw_report *hw, const char *type, uint64_t min_mem);

/**
 * Pick free units for `req`.
 *
 * @param hw          the box (pn_hw_report_build)
 * @param req         what the profile asks for
 * @param units_out   receives the chosen unit ids (die ids; the cpu unit is
 *                    returned as count 0 — a CPU load has no dies)
 * @param n_out       receives how many ids were written
 * @param reason      on refusal: "needs 2 free k80-gpu on one card, you have 1"
 * @return 0 on success, -1 when it does not fit (reason set)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int pn_unit_place(const pn_hw_report *hw, const pn_unit_request *req, int *units_out,
                  int *n_out, char *reason, size_t reason_cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_UNIT_PLACE_H */
