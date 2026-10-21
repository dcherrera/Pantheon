/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_attn_path_tracker.h
 * @brief Per-attention-layer record of WHICH attention path is running — the
 *        on-die flash kernels or the host-softmax fallback — plus the bounded
 *        retry policy for a per-slot KV-ring allocation that failed.
 *
 * WHY THIS EXISTS
 *   Host-side causal GQA softmax on a 2-core host is the wall this project
 *   spent a phase escaping: it was 87.5% of an 8k prefill, and moving attention
 *   on-die took that prefill from 26 minutes to 3.82 minutes. Falling back to it
 *   is not graceful degradation. A measured cliff on the depth bench is
 *   13.13 / 13.02 tok/s at depth 1024 / 2048 and 1.22 tok/s at 4096 — 10.7x,
 *   K=2, two dies of one K80, 2026-08-08 — with correct output, no error, and nothing on
 *   any status surface saying which path the layer took. (A widely-quoted 158x
 *   for the same cliff came off an ASAN build; ASAN inflates the host path by
 *   roughly another order of magnitude. 10.7x is the real number, and it grows
 *   with context because host softmax is O(context) per layer per token.)
 *
 *   Worse, one of the ways in LATCHED. afcad12 made a failed
 *   gpu_attn_slot_ring_ensure sticky, for a good reason: without it the layer
 *   re-freed and re-attempted a 144 MiB cuMemAlloc on EVERY decode step and
 *   bumped slots_ring_gen each time, which invalidates every layer's captured
 *   CUDA graph every step — a 62% collapse, far worse than the ~10-15% the
 *   fallback intends. But the cure made ONE transient allocation failure pin
 *   that layer to the slow path for the life of the process. That is the same
 *   one-way ratchet the CUDA-graph capture latch had
 *   (gpu_graph_capture_tracker.h), and it gets the same treatment.
 *
 * THE TWO CASES, KEPT APART
 *   TRANSIENT   — the rings do not fit RIGHT NOW. Another model is loading, a
 *                 sibling instance holds the die, a bigger geometry has not been
 *                 released yet. Retried on a backoff of 8/32/128/512 eligible
 *                 steps and then given up on. A success resets the budget.
 *   IMPOSSIBLE  — the request exceeds the die's whole usable VRAM
 *                 (total - reserve), so it would fail on an empty die. Nothing
 *                 to wait for: give up immediately at this geometry. The caller
 *                 decides this from fc_mem_info_ctx, so it is a measurement, not
 *                 a heuristic.
 *
 *   Either way a STRICTLY SMALLER geometry (fewer slots, shorter window, a
 *   cheaper cache mode) resets the budget at once — dropping K or ctx_cap must
 *   recover the on-die path without waiting out a backoff and without a reload.
 *
 * WHY THE RETRY IS CHEAP NOW
 *   The 62% collapse afcad12 measured was not the failed cuMemAlloc; it was the
 *   unconditional slots_ring_gen bump beside it. A retry that frees nothing and
 *   allocates nothing must not claim the ring addresses moved. With that fixed
 *   and the backoff above, a permanently-unfittable layer spends at most
 *   PN_ATTN_RING_MAX_FAILURES rounds of a handful of failed cuMemAllocs over
 *   ~680 steps and then costs nothing at all.
 *
 * Thread-safe: no — one tracker belongs to one attention layer on one die,
 * touched only by that die's driver thread. The residency roll-up reads them
 * without a lock and may observe a step-boundary-torn snapshot; every field is
 * a single word and the counts are advisory.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_GPU_ATTN_PATH_TRACKER_H
#define PN_GPU_ATTN_PATH_TRACKER_H

/* Which attention path this layer ran on its most recent batched-decode step. */
typedef enum {
    PN_ATTN_PATH_UNKNOWN = 0,  /* the forward has not reached this layer yet */
    PN_ATTN_PATH_ONDIE,        /* on-die flash attention */
    PN_ATTN_PATH_HOST          /* host-softmax fallback */
} pn_attn_path_state;

/* Why the on-die path was refused. Mirrors the guards in
 * attn_slots_flash_prep, in the order they are evaluated. */
typedef enum {
    PN_ATTN_HOST_NONE = 0,
    PN_ATTN_HOST_FLASH_OFF,       /* QWEN35_FLASH_SLOTS=0 */
    PN_ATTN_HOST_NO_KERNELS,      /* no flash kernels on this ctx */
    PN_ATTN_HOST_NO_SCRATCH,      /* batch scratch a_z / pos_g missing */
    PN_ATTN_HOST_NO_SLOT_KERNEL,  /* no fp32 / q4 slot-decode kernel */
    PN_ATTN_HOST_RING_WRAP,       /* non-uniform KV geometry, or pos+1>max_seq */
    PN_ATTN_HOST_RING_ALLOC,      /* per-slot KV ring allocation failed */
    PN_ATTN_HOST_DEC_SCRATCH,     /* decode partial-scratch allocation failed */
    PN_ATTN_HOST_REASON_COUNT
} pn_attn_host_reason;

/* Retry budget for a failed ring allocation: 4 retries after the first
 * failure, then stop. Counted in ELIGIBLE STEPS (steps where this layer ran
 * and a geometry change asked for an allocation), so the cost is fixed
 * regardless of token rate. */
#define PN_ATTN_RING_MAX_FAILURES 5

/* Per-attention-layer path + ring-allocation tracker. Zero-initialised =
 * UNKNOWN, which is what calloc of the layer array already gives. */
typedef struct {
    unsigned char  state;            /* pn_attn_path_state */
    unsigned char  reason;           /* pn_attn_host_reason (HOST only) */
    unsigned char  alloc_failures;   /* consecutive ring-alloc failures */
    unsigned char  alloc_gave_up;    /* budget spent, or proven impossible */
    unsigned char  alloc_impossible; /* need > this die's whole usable VRAM */
    unsigned short retry_cooldown_steps; /* eligible steps left before retry */
    unsigned int   lifetime_alloc_failures;
    unsigned int   lifetime_host_engagements; /* on-die -> host transitions */
} pn_attn_path_tracker;

/* Roll-up over a set of trackers. `total` counts only layers the forward has
 * actually REACHED — a model that has never decoded batched reports total 0
 * rather than a fabricated deficit. ondie + host == total. */
typedef struct {
    int total;
    int ondie;
    int host;
    int gave_up;      /* subset of host: ring-alloc retry budget spent */
    int impossible;   /* subset of gave_up: cannot fit an empty die */
    int by_reason[PN_ATTN_HOST_REASON_COUNT];
    unsigned int lifetime_alloc_failures;
    unsigned int lifetime_host_engagements;
} pn_attn_residency;

/* One model's attention-path residency, split by code path. `all` is the sum.
 * Only the batched-decode slots path is instrumented today — it is the path
 * the serve actually takes (QWEN35_BATCH_SLOTS=1 routes even a lone request
 * through the batch scheduler) and the only one carrying the ring latch. */
typedef struct {
    pn_attn_residency all;
    pn_attn_residency batched_attn;
} pn_model_attn_residency;

/* This layer ran on-die this step. Clears any host reason; does NOT clear the
 * ring-allocation budget (that is pn_attn_ring_alloc_ok's job) so a layer that
 * is on-die for an unrelated reason cannot silently refund a failing ring. */
void pn_attn_path_ondie(pn_attn_path_tracker *t);

/* This layer took the host-softmax fallback this step, for `reason`. Counts a
 * lifetime engagement only on an on-die -> host transition, so a layer parked
 * on the host path does not inflate the counter every token. */
void pn_attn_path_host(pn_attn_path_tracker *t, pn_attn_host_reason reason);

/**
 * Should a per-slot KV-ring allocation be attempted on this step?
 *
 * Call ONCE per step on a layer whose ring geometry changed and therefore needs
 * an allocation. It owns the whole backoff: on a cooldown step it decrements
 * and answers no.
 *
 * @param t  Tracker for this attention layer
 * @return   1 to attempt the allocation now, 0 to take the host path this step
 *
 * Thread-safe: no (single owning driver thread)
 * Reentrant: no (mutates the tracker)
 */
int pn_attn_ring_should_attempt(pn_attn_path_tracker *t);

/* The rings allocated. Clears the failure budget: a geometry that has ever
 * fitted is proven fittable, so a later failure starts fresh. */
void pn_attn_ring_alloc_ok(pn_attn_path_tracker *t);

/* A ring allocation attempt failed. `impossible` != 0 means the request
 * exceeds the die's entire usable VRAM and would fail on an empty die, so
 * there is nothing transient to wait for and the budget is spent at once. */
void pn_attn_ring_alloc_failed(pn_attn_path_tracker *t, int impossible);

/* The requested geometry is strictly SMALLER than the one that failed, so the
 * old verdict does not describe it. Clears the budget and any cooldown, which
 * is what makes dropping K or ctx_cap recover the on-die path immediately. */
void pn_attn_ring_geometry_shrank(pn_attn_path_tracker *t);

/* Fold ONE tracker into a residency roll-up. Zero the roll-up first. */
void pn_attn_residency_add(pn_attn_residency *out,
                           const pn_attn_path_tracker *t);

/* Short stable names for logs and JSON. */
const char *pn_attn_path_state_name(unsigned char state);
const char *pn_attn_host_reason_name(unsigned char reason);

/**
 * Test-only fault injection: force the first N ring-allocation attempts of
 * EVERY attention layer to be treated as failures, so the retry path can be
 * exercised end to end on real hardware without having to genuinely exhaust a
 * die. Read once from QWEN35_RING_ALLOC_FAIL_INJECT; 0 (unset) is the
 * production path and costs one cached compare.
 *
 * A value of the form "N:impossible" injects PERMANENT failures instead of
 * transient ones, which is how the give-up-at-once path is exercised.
 *
 * @return  How many attempts per layer to fail, 0 = never inject
 */
int pn_attn_ring_inject_failures(void);

/* Is the injected failure a permanent (impossible) one? */
int pn_attn_ring_inject_impossible(void);

/* Should THIS layer's current allocation attempt be forced to fail? Keyed off
 * the layer's own lifetime failure count, so each layer fails exactly N times
 * and then behaves normally. Injecting N < PN_ATTN_RING_MAX_FAILURES exercises
 * recovery; injecting more exercises the give-up path. */
int pn_attn_ring_inject_now(const pn_attn_path_tracker *t);

#endif /* PN_GPU_ATTN_PATH_TRACKER_H */
