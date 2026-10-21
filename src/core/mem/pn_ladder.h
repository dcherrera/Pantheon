/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_ladder.h
 * @brief The reactive degradation ladder (MemoryAware.md §7, §10 — Phase 4).
 *        Maps the pressure level pn_memwatch already publishes (pn_mem_level)
 *        onto concrete, monotonic mitigation "rungs" the engine engages BEFORE
 *        the box tips over: shrink concurrency, force a tighter KV cache, enable
 *        low-attention KV eviction, and clamp an over-window prefill. Admission
 *        (pn_admission) is the per-request fit gate; the ladder is the
 *        pressure-driven policy that keeps the whole box off the OOM-reboot line
 *        as headroom shrinks.
 *
 *        Safe-by-construction: at PN_MEM_LEVEL_OK the plan is byte-identical to
 *        the operator config (full slots, no clamp, configured KV format) — the
 *        rungs only ever REDUCE from config, and only under real pressure. The
 *        pure planner has no I/O and no allocation, so it is trivially
 *        unit-testable off the K80 test machine; PN_LADDER_DISABLE bypasses it entirely for
 *        the portable / bare-Mac path (MemoryAware.md §11 Q2 — the layer must
 *        still work launched with no cgroup).
 *
 * @status Phase 4 — reactive ladder rungs 2–4. Pure planner + wiring hooks.
 * @issues Rung-4b eviction exposes a policy flag + hook point; the SnapKV-style
 *         row selection itself lands with the KV path work.
 * @todo None
 */

#ifndef PN_LADDER_H
#define PN_LADDER_H

#include <stdbool.h>

#include "pn_mem_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rung thresholds (which level first engages each rung). Monotonic and coarse
 * on purpose — the levels are already the tuned signal (pn_memwatch host_level /
 * die_level). Rationale for the ordering, least-harmful first:
 *   - Rung 4a force-q4 KV (near-zero quality cost, ~4–7x VRAM win) engages first,
 *     at WATCH.
 *   - Rung 2 shrink-concurrency (throughput only, no correctness hit) and Rung 3
 *     clamp-prefill and Rung 4b evict engage together at TIGHT and harden at
 *     CRITICAL. */
#define PN_LADDER_FORCE_Q4_LEVEL   PN_MEM_LEVEL_WATCH   /* rung 4a */
#define PN_LADDER_SHRINK_LEVEL     PN_MEM_LEVEL_TIGHT   /* rung 2  */
#define PN_LADDER_CLAMP_LEVEL      PN_MEM_LEVEL_TIGHT   /* rung 3  */
#define PN_LADDER_EVICT_LEVEL      PN_MEM_LEVEL_TIGHT   /* rung 4b */

/* Rung 2 concurrency reduction: TIGHT halves the configured slot ceiling,
 * CRITICAL floors it. Always at least one slot so a single request can still be
 * served (shedding new *work* is admission's job, not the ladder's). */
#define PN_LADDER_TIGHT_SLOTS_DIV     2
#define PN_LADDER_CRITICAL_SLOTS      1

/* Rung 3 prefill-window clamp: TIGHT clamps an over-window prompt to half the
 * configured window, CRITICAL to a quarter. Applied only when a base window is
 * known (>0) — 0 means "no ladder clamp" (the caller's own max still applies). */
#define PN_LADDER_TIGHT_WINDOW_DIV    2
#define PN_LADDER_CRITICAL_WINDOW_DIV 4

/* The mitigation plan for one pressure level. Every field is an absolute value
 * the caller applies directly (not a delta), so a caller wires one read without
 * having to know the ladder's internal policy. */
typedef struct {
    pn_mem_level level;          /* the level this plan was computed for        */
    int  max_slots;              /* rung 2: effective concurrency ceiling (>=1) */
    bool force_kv_q4;            /* rung 4a: force Q4 KV regardless of config    */
    bool kv_evict;               /* rung 4b: enable low-attention KV eviction    */
    int  prefill_window;         /* rung 3: max prefill tokens (0 = no clamp)   */
    bool degraded;               /* true if any rung reduced a configured value */
    const char *note;            /* short client/log note (never NULL)          */
} pn_ladder_plan;

/* True when the ladder is disabled via PN_LADDER_DISABLE (the plan will mirror
 * the operator config at every level). The portable / bare-Mac escape hatch. */
bool pn_ladder_disabled(void);

/* Compute the rung plan for a pressure `level` given the operator config:
 *   cfg_max_slots      — configured concurrency ceiling (>=1; clamped up if <1)
 *   cfg_prefill_window — configured max prefill window in tokens (0 = unknown /
 *                        no base to clamp against; the rung 3 clamp is then a
 *                        no-op and prefill_window stays 0)
 *   cfg_kv_q4          — whether q4 KV is already the operator default
 * Fills *out. Pure — no I/O, no allocation, always succeeds. At level OK (or
 * when disabled) the plan equals the config with degraded=false. */
void pn_ladder_plan_for(pn_mem_level level, int cfg_max_slots,
                        int cfg_prefill_window, bool cfg_kv_q4,
                        pn_ladder_plan *out);

/* Convenience wrapper: derive the plan from a live pn_mem_state snapshot's
 * level. `state` may be NULL (treated as level OK). Pure. */
void pn_ladder_plan_from_state(const pn_mem_state *state, int cfg_max_slots,
                               int cfg_prefill_window, bool cfg_kv_q4,
                               pn_ladder_plan *out);

#ifdef __cplusplus
}
#endif

#endif /* PN_LADDER_H */
