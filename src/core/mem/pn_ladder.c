/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_ladder.c
 * @brief Implementation of the reactive degradation ladder (MemoryAware.md §7,
 *        §10 — Phase 4). A pure level->plan policy: given the pressure level
 *        pn_memwatch publishes and the operator config, decide how far to pull
 *        each mitigation rung. Monotonic (a higher level never relaxes a rung a
 *        lower one engaged) and reduce-only (a rung never grows a configured
 *        value), so the worst a mis-tuned ladder can do is serve conservatively.
 *
 * @status Phase 4 — reactive ladder rungs 2–4.
 * @issues None
 * @todo None
 */

#include "pn_ladder.h"

#include <stdio.h>
#include <stdlib.h>

bool pn_ladder_disabled(void)
{
    const char *v = getenv("PN_LADDER_DISABLE");
    return v && v[0] && v[0] != '0';
}

bool pn_ladder_kv_q4_allowed(void)
{
    const char *v = getenv("PN_LADDER_KV_Q4");
    return !(v && v[0] == '0');
}

pn_ladder_kv_action pn_ladder_kv_action_for(pn_mem_level level, bool explicit_fp32)
{
    pn_ladder_plan plan;

    pn_ladder_plan_for(level, 1, 0, false, &plan);
    if (!plan.force_kv_q4)
        return PN_LADDER_KV_KEEP;
    return explicit_fp32 ? PN_LADDER_KV_REFUSE : PN_LADDER_KV_TO_Q4;
}

static __thread char g_kv_refusal[192];

void pn_ladder_kv_refusal_set(const char *why)
{
    snprintf(g_kv_refusal, sizeof(g_kv_refusal), "%s", why ? why : "");
}

void pn_ladder_kv_refusal_clear(void)
{
    g_kv_refusal[0] = '\0';
}

const char *pn_ladder_kv_refusal(void)
{
    return g_kv_refusal[0] ? g_kv_refusal : NULL;
}

/* Rung 2: the effective slot ceiling for a level. Reduce-only, floor of 1. */
static int slots_for(pn_mem_level level, int cfg_max_slots)
{
    int cfg = cfg_max_slots > 0 ? cfg_max_slots : 1;
    if (level >= PN_MEM_LEVEL_CRITICAL)
        return PN_LADDER_CRITICAL_SLOTS;
    if (level >= PN_LADDER_SHRINK_LEVEL) {
        int s = cfg / PN_LADDER_TIGHT_SLOTS_DIV;
        return s > 0 ? s : 1;
    }
    return cfg;
}

/* Rung 3: the effective prefill window for a level. Reduce-only; 0 base means
 * "no ladder clamp" and stays 0. */
static int window_for(pn_mem_level level, int cfg_prefill_window)
{
    if (cfg_prefill_window <= 0)
        return 0;
    if (level >= PN_MEM_LEVEL_CRITICAL)
        return cfg_prefill_window / PN_LADDER_CRITICAL_WINDOW_DIV;
    if (level >= PN_LADDER_CLAMP_LEVEL)
        return cfg_prefill_window / PN_LADDER_TIGHT_WINDOW_DIV;
    return cfg_prefill_window;
}

void pn_ladder_plan_for(pn_mem_level level, int cfg_max_slots,
                        int cfg_prefill_window, bool cfg_kv_q4,
                        pn_ladder_plan *out)
{
    if (!out)
        return;

    int cfg_slots = cfg_max_slots > 0 ? cfg_max_slots : 1;

    /* Disabled, or ample headroom: mirror the config exactly. Keeping OK
     * byte-identical to today is what makes wiring the rungs low-risk. */
    if (pn_ladder_disabled() || level <= PN_MEM_LEVEL_OK) {
        out->level          = level;
        out->max_slots      = cfg_slots;
        out->force_kv_q4    = cfg_kv_q4;
        out->kv_evict       = false;
        out->prefill_window = cfg_prefill_window > 0 ? cfg_prefill_window : 0;
        out->degraded       = false;
        out->note           = pn_ladder_disabled() ? "ladder disabled"
                                                    : "ample headroom";
        return;
    }

    out->level          = level;
    out->max_slots      = slots_for(level, cfg_slots);
    out->force_kv_q4    = cfg_kv_q4 || (level >= PN_LADDER_FORCE_Q4_LEVEL &&
                                         pn_ladder_kv_q4_allowed());
    out->kv_evict       = (level >= PN_LADDER_EVICT_LEVEL);
    out->prefill_window = window_for(level, cfg_prefill_window);

    out->degraded = out->max_slots < cfg_slots ||
                    (out->force_kv_q4 && !cfg_kv_q4) ||
                    out->kv_evict ||
                    (cfg_prefill_window > 0 &&
                     out->prefill_window < cfg_prefill_window);

    /* One client/log line per level. Ordered worst-first so the message names
     * the strongest rung engaged. */
    switch (level) {
    case PN_MEM_LEVEL_CRITICAL:
        out->note = "server memory critical — single-slot, tightened KV, "
                    "context window reduced";
        break;
    case PN_MEM_LEVEL_TIGHT:
        out->note = "server under memory pressure — reduced concurrency, "
                    "tightened KV, context window may be reduced";
        break;
    case PN_MEM_LEVEL_WATCH:
        out->note = "server headroom shrinking — KV cache tightened";
        break;
    default:
        out->note = "ample headroom";
        break;
    }
}

void pn_ladder_plan_from_state(const pn_mem_state *state, int cfg_max_slots,
                               int cfg_prefill_window, bool cfg_kv_q4,
                               pn_ladder_plan *out)
{
    pn_mem_level level = state ? state->level : PN_MEM_LEVEL_OK;
    pn_ladder_plan_for(level, cfg_max_slots, cfg_prefill_window, cfg_kv_q4, out);
}
