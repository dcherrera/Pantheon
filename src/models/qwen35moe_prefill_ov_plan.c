/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_prefill_ov_plan.c
 * @brief  The overlap's step placement — see qwen35moe_prefill_ov_plan.h.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "qwen35moe_prefill_ov_plan.h"

#include "../accel/tt/pn_tt_superband.h"

#include <stdlib.h>
#include <string.h>

void pn_ov_plan_default_at(uint32_t span, uint32_t pipe_depth, uint32_t out_at[3])
{
    const uint32_t up = span / 4u ? span / 4u : 1u;
    const uint32_t down = (5u * span) / 13u ? (5u * span) / 13u : 1u;

    out_at[0] = pipe_depth;
    out_at[1] = out_at[0] + up;
    out_at[2] = out_at[1] + down;
}

int pn_ov_plan_parse_at(const char *text, uint32_t out_at[3])
{
    uint32_t value[3];
    char *end;

    if (!text || !*text)
        return -1;
    for (int i = 0; i < 3; i++) {
        const long v = strtol(text, &end, 10);

        if (end == text || v < 0 || v > 1000 || (i < 2 && *end != ',') || (i == 2 && *end))
            return -1;
        value[i] = (uint32_t)v;
        text = end + (i < 2 ? 1 : 0);
    }
    if (!(value[0] < value[1] && value[1] < value[2]))
        return -1;
    memcpy(out_at, value, sizeof(value));
    return 0;
}

void pn_ov_plan_job_bands(const pn_ov_plan *plan, uint32_t job, uint32_t *first,
                          uint32_t *count)
{
    pn_tt_superband_job(plan->bands, plan->span, job, first, count);
}

static uint32_t ov_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

/** A job's FINISH sorts between the next job's OUT and UP: the next out-projection is queued on
 *  the grid before FINISH waits for this job's down, so the grid is never left empty there. */
static uint32_t ov_action_key(const pn_ov_action *a)
{
    static const uint32_t rank[PN_OV_STAGE_COUNT] = { 0u, 2u, 3u, 1u }; /* OUT FINISH UP DOWN */

    return (a->job + (a->stage == PN_OV_STAGE_FINISH ? 1u : 0u)) * 4u + rank[a->stage];
}

static int ov_action_order(const void *a, const void *b)
{
    const pn_ov_action *x = a, *y = b;
    const uint32_t kx = ov_action_key(x), ky = ov_action_key(y);

    if (x->tick != y->tick)
        return x->tick < y->tick ? -1 : 1;
    return kx < ky ? -1 : kx > ky ? 1 : 0;
}

int pn_ov_plan_build(pn_ov_plan *plan, uint32_t bands, uint32_t span, uint32_t pipe_depth,
                     const uint32_t at[3])
{
    uint32_t tick[PN_OV_PLAN_MAX_JOBS][PN_OV_STAGE_COUNT];
    uint32_t use[3];

    memset(plan, 0, sizeof(*plan));
    /* Span 1 is all 1-band jobs (decode-shaped grid projections, G.3): refused. */
    if (bands == 0u || span < 2u || pn_tt_superband_jobs(bands, span) > PN_OV_PLAN_MAX_JOBS)
        return -1;
    if (at)
        memcpy(use, at, sizeof(use));
    else
        pn_ov_plan_default_at(span, pipe_depth, use);
    /* A job's gates trail its last band by the pipe's depth: OUT cannot read `gated` sooner. */
#ifndef PN_OV_PLAN_RED
    use[0] = ov_max(use[0], pipe_depth);
#endif
    plan->bands = bands;
    plan->span = span;
    plan->jobs = pn_tt_superband_jobs(bands, span);
    memcpy(plan->at, use, sizeof(use));
    for (uint32_t k = 0; k < plan->jobs; k++) {
        uint32_t first, count, last;

        pn_ov_plan_job_bands(plan, k, &first, &count);
        last = first + count - 1u;
        tick[k][PN_OV_STAGE_OUT] = last + use[0];
        if (k > 0u)
            tick[k][PN_OV_STAGE_OUT] =
                ov_max(tick[k][PN_OV_STAGE_OUT], tick[k - 1u][PN_OV_STAGE_DOWN] + 1u);
        tick[k][PN_OV_STAGE_UP] = ov_max(last + use[1], tick[k][PN_OV_STAGE_OUT] + 1u);
        tick[k][PN_OV_STAGE_DOWN] = ov_max(last + use[2], tick[k][PN_OV_STAGE_UP] + 1u);
        /* FINISH waits for this job's down and issues the next layer's projections: it runs
         * at the next job's OUT, just after it (G.2: that out-proj keeps the grid busy while
         * FINISH's synchronous steps run). */
        if (k > 0u)
            tick[k - 1u][PN_OV_STAGE_FINISH] = tick[k][PN_OV_STAGE_OUT];
    }
    tick[plan->jobs - 1u][PN_OV_STAGE_FINISH] = bands;
    for (uint32_t k = 0; k < plan->jobs; k++)
        for (uint32_t s = 0; s < PN_OV_STAGE_COUNT; s++) {
            pn_ov_action *a = &plan->actions[plan->count++];

            a->job = k;
            a->stage = (pn_ov_stage)s;
            a->tick = tick[k][s] < bands ? tick[k][s] : bands;
        }
    qsort(plan->actions, plan->count, sizeof(plan->actions[0]), ov_action_order);
    return 0;
}

size_t pn_ov_plan_due(const pn_ov_plan *plan, uint32_t band, pn_ov_action *out, size_t max)
{
    size_t n = 0;

    for (size_t i = 0; i < plan->count && n < max; i++)
        if (plan->actions[i].tick == band)
            out[n++] = plan->actions[i];
    return n;
}
