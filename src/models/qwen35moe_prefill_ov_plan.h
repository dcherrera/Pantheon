/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_prefill_ov_plan.h
 * @brief  Where each super-band step of the PHASE W overlap lands in an SSM layer's band loop.
 *
 * The streamed band loop is ONE in-order puller program: a step that waits for the grid
 * stalls every chain band queued behind it. So each super-band job's steps are placed at band
 * offsets after the job's last band, far enough apart that the grid step a wait names has had
 * chain bands to run under:
 *
 *   OUT     the job's `gated` rows settled (the PIPE leaves its last bands' gates `pipe_depth`
 *           bands behind), out-proj issued detached
 *   UP      grid waited, residual add, post-norm, gate/up issued detached
 *   DOWN    grid waited, SwiGLU, down issued detached
 *   FINISH  grid waited, residual add, the next layer's attn-norm, q/k/v/z/alpha/beta issued
 *           detached (the wavefront) — at the next job's OUT, just after it (G.2)
 *
 * Pure: no device, no model. The whole layer's action list is computed once and handed out
 * band by band; the layer's tail (`band == bands`) gets everything left, in order.
 *
 * @status O.1 of the prefill chain work.
 * @issues None
 * @todo   None
 */
#ifndef QWEN35MOE_PREFILL_OV_PLAN_H
#define QWEN35MOE_PREFILL_OV_PLAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PN_OV_STAGE_OUT = 0,
    PN_OV_STAGE_UP = 1,
    PN_OV_STAGE_DOWN = 2,
    PN_OV_STAGE_FINISH = 3,
    PN_OV_STAGE_COUNT = 4
} pn_ov_stage;

typedef struct {
    uint32_t job;
    pn_ov_stage stage;
    uint32_t tick; /**< the band after which it runs; `bands` = the layer's tail */
} pn_ov_action;

#define PN_OV_PLAN_MAX_JOBS 64u

typedef struct {
    uint32_t bands, span, jobs;
    uint32_t at[3]; /**< OUT / UP / DOWN offsets after a job's last band */
    pn_ov_action actions[PN_OV_PLAN_MAX_JOBS * PN_OV_STAGE_COUNT];
    size_t count;
} pn_ov_plan;

/**
 * Build a layer's plan.
 *
 * @param bands       bands in the layer's pass (1..PN_OV_PLAN_MAX_JOBS * span)
 * @param span        bands a super-band job covers (>= 2)
 * @param pipe_depth  bands the chain's gates trail their band (PIPE 2: 2, PIPE 1: 1, serial 0)
 * @param at          OUT / UP / DOWN offsets, or NULL for the defaults (pn_ov_plan_default_at)
 * @return 0, or -1 on a shape it cannot plan (the plan is then empty)
 *
 * OUT is never earlier than `pipe_depth` bands after the job's last band; the stages of a job
 * are at strictly later bands; a job's OUT is never before the previous job's DOWN + 1.
 *
 * Thread-safe: yes (own plan)   Reentrant: yes   Signal-safe: yes
 */
int pn_ov_plan_build(pn_ov_plan *plan, uint32_t bands, uint32_t span, uint32_t pipe_depth,
                     const uint32_t at[3]);

/** The defaults for `span` / `pipe_depth`: OUT `pipe_depth`, UP + span/4, DOWN + 5*span/13
 *  (13 bands, PIPE 2: 2, 5, 10). */
void pn_ov_plan_default_at(uint32_t span, uint32_t pipe_depth, uint32_t out_at[3]);

/** Parse "a,b,c" into `out_at`. @return 0, or -1 (unset, malformed, or not increasing). */
int pn_ov_plan_parse_at(const char *text, uint32_t out_at[3]);

/** Job `job`'s bands: [first, first + count) (pn_tt_superband.h: a last job under 2 bands is
 *  folded into the one before). */
void pn_ov_plan_job_bands(const pn_ov_plan *plan, uint32_t job, uint32_t *first,
                          uint32_t *count);

/**
 * The actions due after band `band` ran (`band == bands`: the tail), in order.
 *
 * @return how many were written to `out` (at most `max`)
 */
size_t pn_ov_plan_due(const pn_ov_plan *plan, uint32_t band, pn_ov_action *out, size_t max);

#ifdef __cplusplus
}
#endif

#endif /* QWEN35MOE_PREFILL_OV_PLAN_H */
