/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_role_plan.h
 * @brief The persistent norm role's host planning, with no card and no `USE_TT`: the knobs,
 *        the step <-> (layer, norm) map, and the forward's STEP CURSOR, which turns the
 *        forward's norm and residual calls into role steps and refuses any order the role's
 *        schedule would not run. tests/test_tt_norm_role.c proves it off-card.
 *
 * THE SCHEDULE OF A TOKEN (n layers, 2n + 1 steps):
 *   step 0          LOAD  the embedding row, attn_norm of layer 0
 *   step 2l + 1     ADD   the mixer's out-projection, post_norm of layer l
 *   step 2l + 2     ADD   the FFN's down projection, attn_norm of layer l + 1 (output_norm
 *                         after the last layer)
 *
 * THE CURSOR. The forward calls norm, residual, norm, residual, ... norm. A residual folds into
 * the NEXT norm (that step's ADD), so the cursor demands: a norm with no residual pending only
 * at step 0, a norm with one pending at every other step, never two residuals in a row, never
 * more steps than the token has, and exactly all of them by the token's end. Any other order
 * would add the wrong projection to h, so it is refused and the forward fails, never computed.
 *
 * @status P0.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_NORM_ROLE_PLAN_H
#define PN_TT_NORM_ROLE_PLAN_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_TT_PERSIST_NORM_ENV "PN_TT_PERSIST_NORM"
#define PN_TT_PERSIST_NORM_CORES_ENV "PN_TT_PERSIST_NORM_CORES"
#define PN_TT_PERSIST_NORM_EXCHANGE_ENV "PN_TT_PERSIST_NORM_EXCHANGE"
/** Grid cores 0-3 and 7-10: NoC (1..4, 2) and (10..13, 2), two multicast rows of four. */
#define PN_TT_PERSIST_NORM_CORES_DEFAULT "0-3,7-10"
#define PN_TT_NORM_ROLE_MAX_MEMBERS 8u

/** Which norm a step applies. */
typedef enum pn_tt_norm_role_norm {
    PN_TT_NORM_ROLE_ATTN = 0,
    PN_TT_NORM_ROLE_POST = 1,
    PN_TT_NORM_ROLE_OUTPUT = 2
} pn_tt_norm_role_norm;

/** The forward's step cursor. */
typedef struct pn_tt_norm_role_cursor {
    uint32_t steps;        /**< steps a token */
    uint32_t next;         /**< the step the next norm issues */
    bool residual_pending; /**< a residual folded into the next step */
} pn_tt_norm_role_cursor;

/**
 * `PN_TT_PERSIST_NORM` from `text` (NULL = unset): 0 off, 1 on; anything else is off.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_norm_role_mode_from(const char *text);

/* Exchange mode bits (foundry kNrExchangeFaces / kNrExchangeStagger, by value). */
#define PN_TT_NORM_ROLE_EXCHANGE_FACES 1u
#define PN_TT_NORM_ROLE_EXCHANGE_STAGGER 2u

/**
 * `PN_TT_PERSIST_NORM_EXCHANGE` from `text` (NULL or empty = the default, whole; the card
 * measured faces 1.7x slower and stagger a wash, foundry spec §11.13):
 * "whole" 0 (whole pages, table order), "faces" 1, "stagger" 2, "faces,stagger" 3.
 *
 * @return the mode bits, or -1 for anything else (the role refuses to arm)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_norm_role_exchange_from(const char *text);

/** Steps a token of `layers` layers (2 * layers + 1). */
uint32_t pn_tt_norm_role_steps(uint32_t layers);

/**
 * Step `step`'s norm and layer in a model of `layers` layers.
 *
 * @return false when the step is outside the token
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_norm_role_step_norm(uint32_t step, uint32_t layers, pn_tt_norm_role_norm *out_norm,
                               uint32_t *out_layer);

/** A token begins: the next norm is step 0, nothing pending. */
void pn_tt_norm_role_cursor_begin(pn_tt_norm_role_cursor *cursor, uint32_t steps);

/**
 * The forward's residual call: folds into the next step.
 *
 * @return 0, or -1 (a residual already pending, or the token's steps are spent)
 */
int pn_tt_norm_role_cursor_residual(pn_tt_norm_role_cursor *cursor);

/**
 * The forward's norm call: the step it issues.
 *
 * @param out_step  receives the step index
 * @return 0, or -1 (the residual state does not match the step, or the token is spent)
 */
int pn_tt_norm_role_cursor_norm(pn_tt_norm_role_cursor *cursor, uint32_t *out_step);

/** The token ends: 0 when every step was issued and nothing is pending, else -1. */
int pn_tt_norm_role_cursor_end(const pn_tt_norm_role_cursor *cursor);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_NORM_ROLE_PLAN_H */
