/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_attn_path_tracker.c
 * @brief Implementation of the per-attention-layer path tracker — which
 *        attention path ran, why the on-die one was refused, and the bounded
 *        retry for a failed per-slot KV-ring allocation.
 *
 * Pure host state machine: no CUDA, no allocation, no locks. Deliberate — the
 * retry policy has to be provable by a unit test over this file alone, on any
 * machine, with no GPU and no model.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "gpu_attn_path_tracker.h"

#include <stdlib.h>
#include <string.h>

/* Eligible-step gap before retry attempt number N (1-based on
 * alloc_failures). The last entry covers any further failure, but the budget
 * gives up before it can be reached twice. */
static const unsigned short PN_ATTN_RING_BACKOFF_STEPS[] = { 8, 32, 128, 512 };

#define PN_ATTN_RING_BACKOFF_COUNT \
    ((int)(sizeof(PN_ATTN_RING_BACKOFF_STEPS) / \
           sizeof(PN_ATTN_RING_BACKOFF_STEPS[0])))

void pn_attn_path_ondie(pn_attn_path_tracker *t)
{
    if (!t)
        return;
    t->state = PN_ATTN_PATH_ONDIE;
    t->reason = PN_ATTN_HOST_NONE;
}

void pn_attn_path_host(pn_attn_path_tracker *t, pn_attn_host_reason reason)
{
    if (!t)
        return;
    /* Count the ENGAGEMENT, not the step: a layer already parked on the host
     * path would otherwise add one per token and the counter would read as a
     * token rate rather than an incident count. */
    if (t->state != PN_ATTN_PATH_HOST)
        t->lifetime_host_engagements++;
    t->state = PN_ATTN_PATH_HOST;
    t->reason = (unsigned char)reason;
}

int pn_attn_ring_should_attempt(pn_attn_path_tracker *t)
{
    if (!t)
        return 1;   /* no tracker: never block the allocation */

    /* Budget spent (transiently exhausted, or the geometry proven impossible on
     * this die) — this is the bound, and the only permanent no. A strictly
     * smaller geometry clears it via pn_attn_ring_geometry_shrank before this
     * function is reached, which is what keeps "give up" from meaning "this
     * layer is off until a reload". */
    if (t->alloc_gave_up)
        return 0;

    if (t->retry_cooldown_steps > 0) {
        t->retry_cooldown_steps--;
        return 0;
    }
    return 1;
}

void pn_attn_ring_alloc_ok(pn_attn_path_tracker *t)
{
    /* Called on every steady-state step (the geometry is already satisfied), so
     * the common case must touch nothing. */
    if (!t || (!t->alloc_failures && !t->alloc_gave_up &&
               !t->retry_cooldown_steps))
        return;
    t->alloc_failures = 0;
    t->alloc_gave_up = 0;
    t->alloc_impossible = 0;
    t->retry_cooldown_steps = 0;
}

void pn_attn_ring_alloc_failed(pn_attn_path_tracker *t, int impossible)
{
    if (!t)
        return;
    t->lifetime_alloc_failures++;
    if (t->alloc_failures < 255)
        t->alloc_failures++;

    /* Measured, not guessed: the caller compared the request against this die's
     * TOTAL usable VRAM. A request that would fail on an empty die has nothing
     * transient to wait for, so waiting out four backoffs would be four rounds
     * of pointless multi-hundred-MiB cuMemAllocs. Stop now. */
    if (impossible) {
        t->alloc_impossible = 1;
        t->alloc_gave_up = 1;
        t->retry_cooldown_steps = 0;
        return;
    }

    if (t->alloc_failures >= PN_ATTN_RING_MAX_FAILURES) {
        t->alloc_gave_up = 1;
        t->retry_cooldown_steps = 0;
        return;
    }

    int idx = t->alloc_failures - 1;
    if (idx >= PN_ATTN_RING_BACKOFF_COUNT)
        idx = PN_ATTN_RING_BACKOFF_COUNT - 1;
    t->retry_cooldown_steps = PN_ATTN_RING_BACKOFF_STEPS[idx];
}

void pn_attn_ring_geometry_shrank(pn_attn_path_tracker *t)
{
    if (!t)
        return;
    t->alloc_failures = 0;
    t->alloc_gave_up = 0;
    t->alloc_impossible = 0;
    t->retry_cooldown_steps = 0;
}

void pn_attn_residency_add(pn_attn_residency *out,
                           const pn_attn_path_tracker *t)
{
    if (!out || !t || t->state == PN_ATTN_PATH_UNKNOWN)
        return;
    out->total++;
    out->lifetime_alloc_failures += t->lifetime_alloc_failures;
    out->lifetime_host_engagements += t->lifetime_host_engagements;
    if (t->state == PN_ATTN_PATH_HOST) {
        out->host++;
        if (t->reason < PN_ATTN_HOST_REASON_COUNT)
            out->by_reason[t->reason]++;
        if (t->alloc_gave_up)
            out->gave_up++;
        if (t->alloc_impossible)
            out->impossible++;
    } else {
        out->ondie++;
    }
}

const char *pn_attn_path_state_name(unsigned char state)
{
    switch (state) {
    case PN_ATTN_PATH_ONDIE: return "ondie";
    case PN_ATTN_PATH_HOST:  return "host";
    default:                 return "unknown";
    }
}

const char *pn_attn_host_reason_name(unsigned char reason)
{
    switch (reason) {
    case PN_ATTN_HOST_FLASH_OFF:      return "flash_slots_off";
    case PN_ATTN_HOST_NO_KERNELS:     return "no_flash_kernels";
    case PN_ATTN_HOST_NO_SCRATCH:     return "no_batch_scratch";
    case PN_ATTN_HOST_NO_SLOT_KERNEL: return "no_slot_decode_kernel";
    case PN_ATTN_HOST_RING_WRAP:      return "kv_geometry_or_ring_wrap";
    case PN_ATTN_HOST_RING_ALLOC:     return "ring_alloc_failed";
    case PN_ATTN_HOST_DEC_SCRATCH:    return "dec_scratch_alloc_failed";
    default:                          return "none";
    }
}

/* QWEN35_RING_ALLOC_FAIL_INJECT=N or N:impossible. Parsed once. */
static void ring_inject_parse(int *n_out, int *imp_out)
{
    static int cached_n = -1;
    static int cached_imp = 0;
    if (cached_n < 0) {
        cached_n = 0;
        const char *env = getenv("QWEN35_RING_ALLOC_FAIL_INJECT");
        if (env && env[0]) {
            cached_n = atoi(env);
            if (cached_n < 0)
                cached_n = 0;
            cached_imp = (strstr(env, "impossible") != NULL);
        }
    }
    *n_out = cached_n;
    *imp_out = cached_imp;
}

int pn_attn_ring_inject_failures(void)
{
    int n = 0, imp = 0;
    ring_inject_parse(&n, &imp);
    return n;
}

int pn_attn_ring_inject_impossible(void)
{
    int n = 0, imp = 0;
    ring_inject_parse(&n, &imp);
    return imp;
}

int pn_attn_ring_inject_now(const pn_attn_path_tracker *t)
{
    int n = pn_attn_ring_inject_failures();
    if (n <= 0 || !t)
        return 0;
    return (int)t->lifetime_alloc_failures < n;
}
