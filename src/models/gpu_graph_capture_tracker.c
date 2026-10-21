/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_graph_capture_tracker.c
 * @brief Implementation of the per-layer CUDA-graph capture tracker — bounded
 *        retry, permanent give-up, and the three-state residency roll-up.
 *
 * Pure host state machine: no CUDA, no allocation, no locks. That is deliberate
 * — it is the piece that has to be testable without a GPU, and the gate that
 * proves the retry policy is a unit test over this file alone.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "gpu_graph_capture_tracker.h"

#include <stdlib.h>

/* Eligible-step gap before retry attempt number N (1-based on
 * consecutive_failures). The last entry is used for any further failure, but
 * the budget gives up before it can be reached twice. */
static const unsigned short PN_GRAPH_RETRY_BACKOFF_STEPS[] = { 8, 32, 128, 512 };

#define PN_GRAPH_RETRY_BACKOFF_COUNT \
    ((int)(sizeof(PN_GRAPH_RETRY_BACKOFF_STEPS) / \
           sizeof(PN_GRAPH_RETRY_BACKOFF_STEPS[0])))

int pn_graph_capture_should_attempt(pn_graph_capture_tracker *t)
{
    if (!t)
        return 0;

    /* A slot that spent its budget never attempts again — this is the bound.
     * It is also the ONLY permanent no: a gated-out slot never reaches this
     * function at all (the caller tests eligibility first), which is what keeps
     * INELIGIBLE structurally incapable of becoming a retry loop. */
    if (t->state == PN_GRAPH_CAPTURE_GAVE_UP)
        return 0;

    /* Reaching here means the gate is open, so a stale INELIGIBLE from a
     * previously-closed gate is out of date. */
    if (t->state == PN_GRAPH_CAPTURE_INELIGIBLE) {
        t->state = PN_GRAPH_CAPTURE_PENDING;
        t->ineligible_reason = PN_GRAPH_INELIGIBLE_NONE;
    }

    t->evaluated = 1;
    if (t->retry_cooldown_steps > 0) {
        t->retry_cooldown_steps--;
        return 0;
    }
    return 1;
}

void pn_graph_capture_succeeded(pn_graph_capture_tracker *t)
{
    if (!t)
        return;
    t->evaluated = 1;
    t->state = PN_GRAPH_CAPTURE_LIVE;
    t->ineligible_reason = PN_GRAPH_INELIGIBLE_NONE;
    t->consecutive_failures = 0;
    t->retry_cooldown_steps = 0;
    t->lifetime_captures++;
}

void pn_graph_capture_failed(pn_graph_capture_tracker *t)
{
    if (!t)
        return;
    t->evaluated = 1;
    t->lifetime_failures++;
    if (t->consecutive_failures < 255)
        t->consecutive_failures++;

    if (t->consecutive_failures >= PN_GRAPH_CAPTURE_MAX_FAILURES) {
        t->state = PN_GRAPH_CAPTURE_GAVE_UP;
        t->retry_cooldown_steps = 0;
        return;
    }

    int idx = t->consecutive_failures - 1;
    if (idx >= PN_GRAPH_RETRY_BACKOFF_COUNT)
        idx = PN_GRAPH_RETRY_BACKOFF_COUNT - 1;
    t->state = PN_GRAPH_CAPTURE_RETRYING;
    t->retry_cooldown_steps = PN_GRAPH_RETRY_BACKOFF_STEPS[idx];
}

void pn_graph_capture_dropped(pn_graph_capture_tracker *t)
{
    if (!t)
        return;
    t->evaluated = 1;
    t->state = PN_GRAPH_CAPTURE_PENDING;
    t->ineligible_reason = PN_GRAPH_INELIGIBLE_NONE;
    t->consecutive_failures = 0;
    t->retry_cooldown_steps = 0;
}

void pn_graph_capture_superseded(pn_graph_capture_tracker *t)
{
    if (!t)
        return;
    t->evaluated = 0;
    t->state = PN_GRAPH_CAPTURE_PENDING;
    t->ineligible_reason = PN_GRAPH_INELIGIBLE_NONE;
    t->consecutive_failures = 0;
    t->retry_cooldown_steps = 0;
}

void pn_graph_capture_ineligible(pn_graph_capture_tracker *t,
                                 pn_graph_ineligible_reason reason)
{
    if (!t)
        return;
    /* A live graph is a live graph regardless of this step's gate, and a slot
     * that exhausted its retry budget really did FAIL — neither should be
     * relabelled as "this layer never captures". */
    if (t->state == PN_GRAPH_CAPTURE_LIVE || t->state == PN_GRAPH_CAPTURE_GAVE_UP)
        return;
    t->evaluated = 1;
    t->state = PN_GRAPH_CAPTURE_INELIGIBLE;
    t->ineligible_reason = (unsigned char)reason;
}

void pn_graph_residency_add(pn_graph_residency *out,
                            const pn_graph_capture_tracker *t)
{
    if (!out || !t || !t->evaluated)
        return;
    out->total++;
    out->lifetime_captures += t->lifetime_captures;
    out->lifetime_failures += t->lifetime_failures;
    switch (t->state) {
    case PN_GRAPH_CAPTURE_LIVE:
        out->live++;
        break;
    case PN_GRAPH_CAPTURE_RETRYING:
        out->failed++;
        break;
    case PN_GRAPH_CAPTURE_GAVE_UP:
        out->failed++;
        out->gave_up++;
        break;
    case PN_GRAPH_CAPTURE_INELIGIBLE:
        out->ineligible++;
        break;
    default:
        out->pending++;
        break;
    }
}

const char *pn_graph_capture_state_name(unsigned char state)
{
    switch (state) {
    case PN_GRAPH_CAPTURE_LIVE:        return "live";
    case PN_GRAPH_CAPTURE_RETRYING:    return "retrying";
    case PN_GRAPH_CAPTURE_GAVE_UP:     return "gave_up";
    case PN_GRAPH_CAPTURE_INELIGIBLE:  return "ineligible";
    default:                           return "pending";
    }
}

const char *pn_graph_ineligible_reason_name(unsigned char reason)
{
    switch (reason) {
    case PN_GRAPH_INELIGIBLE_GRAPH_OFF:       return "QWEN35_GRAPH=0";
    case PN_GRAPH_INELIGIBLE_SLOTS_OFF:       return "QWEN35_GRAPH_SLOTS=0";
    case PN_GRAPH_INELIGIBLE_ATTN_OFF:        return "QWEN35_GRAPH_SLOTS_ATTN=0";
    case PN_GRAPH_INELIGIBLE_PROFILING:       return "profiling active";
    case PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED: return "MoE not grouped";
    case PN_GRAPH_INELIGIBLE_SLOT_BASE:       return "slot_base>0";
    default:                                  return "none";
    }
}

int pn_graph_capture_inject_failures(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("QWEN35_GRAPH_CAPTURE_FAIL_INJECT");
        cached = env ? atoi(env) : 0;
        if (cached < 0)
            cached = 0;
    }
    return cached;
}

int pn_graph_capture_inject_now(const pn_graph_capture_tracker *t)
{
    int n = pn_graph_capture_inject_failures();
    if (n <= 0 || !t)
        return 0;
    return (int)t->lifetime_failures < n;
}
