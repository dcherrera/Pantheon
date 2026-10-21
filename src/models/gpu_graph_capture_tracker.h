/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_graph_capture_tracker.h
 * @brief Per-layer CUDA-graph capture state: bounded retry after a FAILED
 *        capture, a hard stop for a capture that will never work, and the
 *        three-state residency counters that make graph loss observable.
 *
 * WHY THIS EXISTS
 *   A captured CUDA graph collapses a layer's ~13-30 kernel launches into one
 *   replay and is worth ~1.8 ms/step across the 40 layers (+3.8% at K=1,
 *   +2.4% at K=2 aggregate). Capture can fail for reasons that have nothing to
 *   do with this layer: until foundry e6af3d4 any other host thread's
 *   cuMemAlloc/cuStreamSynchronize could invalidate an in-flight capture
 *   process-wide, and an /admin/models/unload during another model's decode
 *   still legitimately can.
 *
 *   The forward used to latch a single `graph_tried` flag, cleared only in the
 *   drop branches — every one of which requires an EXISTING graph. So a capture
 *   that FAILED could never re-arm: one transient collision cost that layer its
 *   graph for the life of the process. A long-lived serve ratcheted downward
 *   and never recovered, with correct output and no metric saying so.
 *
 * THE TWO CASES, KEPT APART
 *   FAILED     — a capture was attempted and did not produce a live graph.
 *                Might be a transient collision, so it is retried on an
 *                exponential backoff (8, 32, 128, 512 eligible steps) and then
 *                given up on permanently. A capture that succeeds resets the
 *                budget, so recurring collisions never exhaust it.
 *   INELIGIBLE — a gate says this layer does not capture at all (MoE not
 *                grouped, QWEN35_GRAPH=0, sprof active, slot_base>0, ...).
 *                NO capture is ever attempted, so there is nothing to retry
 *                and nothing to back off. A Q5_K_M model reporting 40
 *                ineligible layers is correct behaviour, not a fault.
 *
 * Thread-safe: no — one tracker belongs to one layer on one die, touched only
 * by that die's driver thread. The residency roll-up reads them without a lock
 * and may therefore observe a step-boundary-torn snapshot; every field is a
 * single word and the counts are advisory, so a torn read costs an off-by-one
 * gauge reading and nothing else.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_GPU_GRAPH_CAPTURE_TRACKER_H
#define PN_GPU_GRAPH_CAPTURE_TRACKER_H

/* Lifecycle state of ONE graph slot (one layer, one code path). */
typedef enum {
    PN_GRAPH_CAPTURE_PENDING = 0,   /* eligible; no capture attempted yet */
    PN_GRAPH_CAPTURE_LIVE,          /* a captured graph is replaying */
    PN_GRAPH_CAPTURE_RETRYING,      /* capture failed; backing off, will retry */
    PN_GRAPH_CAPTURE_GAVE_UP,       /* capture failed the retry budget; stopped */
    PN_GRAPH_CAPTURE_INELIGIBLE     /* gated out; no capture is ever attempted */
} pn_graph_capture_state;

/* Why a slot is INELIGIBLE. Ordered so the FIRST failing gate wins, which is
 * how the caller evaluates them. */
typedef enum {
    PN_GRAPH_INELIGIBLE_NONE = 0,
    PN_GRAPH_INELIGIBLE_GRAPH_OFF,      /* QWEN35_GRAPH=0 */
    PN_GRAPH_INELIGIBLE_SLOTS_OFF,      /* QWEN35_GRAPH_SLOTS=0 */
    PN_GRAPH_INELIGIBLE_ATTN_OFF,       /* QWEN35_GRAPH_SLOTS_ATTN=0 */
    PN_GRAPH_INELIGIBLE_PROFILING,      /* sprof/prof/debug syncs mid-body */
    PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED,/* expert quant is not Q3_K */
    PN_GRAPH_INELIGIBLE_SLOT_BASE       /* slot_base>0 group runs direct */
} pn_graph_ineligible_reason;

/* Warm-up (pos<1, body not yet run directly) is deliberately NOT an
 * ineligibility reason: it clears by itself within a step or two, and counting
 * it as "will never capture" would make a healthy serve's first readings lie.
 * A warming slot simply stays PENDING. */

/* Retry budget: 4 retries after the first failure, then GAVE_UP. The backoff
 * is counted in ELIGIBLE STEPS (steps where this layer ran and could have
 * captured), so it is a fixed amortised cost regardless of token rate: a
 * genuinely uncapturable layer spends 5 wasted begin/end pairs over ~680 steps
 * and then costs nothing at all. */
#define PN_GRAPH_CAPTURE_MAX_FAILURES 5

/* Per-layer, per-path capture tracker. Zero-initialised = PENDING, which is
 * what calloc of the layer array already gives. */
typedef struct {
    unsigned char  state;              /* pn_graph_capture_state */
    unsigned char  ineligible_reason;  /* pn_graph_ineligible_reason */
    unsigned char  consecutive_failures;
    unsigned char  evaluated;          /* the forward has reached this slot */
    unsigned short retry_cooldown_steps; /* eligible steps left before retry */
    unsigned int   lifetime_captures;    /* successful captures */
    unsigned int   lifetime_failures;    /* failed capture attempts */
} pn_graph_capture_tracker;

/* Three-state residency roll-up over a set of trackers. `total` counts only
 * slots the forward has actually REACHED (`evaluated`) — a model that decodes
 * batched never touches its single-token slots, and reporting those as pending
 * graphs would be an invented deficit. live + failed + ineligible + pending ==
 * total. */
typedef struct {
    int total;
    int live;         /* holding a captured graph right now */
    int failed;       /* attempted, no graph (RETRYING or GAVE_UP) */
    int gave_up;      /* subset of `failed`: retry budget exhausted */
    int ineligible;   /* gated out; will never attempt a capture */
    int pending;      /* eligible, not yet attempted */
    unsigned int lifetime_captures;
    unsigned int lifetime_failures;
} pn_graph_residency;

/* One model's graph residency, split by the code path each slot belongs to.
 * `all` is the sum; a class the model has never exercised reports total 0
 * rather than a fabricated deficit (see `evaluated` above). */
typedef struct {
    pn_graph_residency all;
    pn_graph_residency batched_attn;   /* per-layer graph, batched decode */
    pn_graph_residency batched_ssm;    /* per-layer graph, batched decode */
    pn_graph_residency single_ssm;     /* per-layer graph, single-token decode */
    pn_graph_residency segment;        /* per-die full-step graph (GRAPH_FULL) */
} pn_model_graph_residency;

/**
 * Should a capture be attempted on this eligible step?
 *
 * Call ONCE per step on a slot that is eligible and holds no live graph. It
 * owns the whole backoff: on a cooldown step it decrements and answers no.
 *
 * @param t  Tracker for this layer's graph slot
 * @return   1 to attempt a capture now, 0 to run the body directly this step
 *
 * Thread-safe: no (single owning driver thread)
 * Reentrant: no (mutates the tracker)
 */
int pn_graph_capture_should_attempt(pn_graph_capture_tracker *t);

/* A capture succeeded and the graph is live. Clears the failure budget: a slot
 * that has ever captured is proven capturable, so a later drop starts fresh. */
void pn_graph_capture_succeeded(pn_graph_capture_tracker *t);

/* A capture attempt did not produce a live, replayable graph. Arms the next
 * backoff, or gives up once the budget is spent. */
void pn_graph_capture_failed(pn_graph_capture_tracker *t);

/* A LIVE graph was dropped for a legitimate reason (a capture key changed, or a
 * replay failed). Re-arms an immediate attempt with a fresh budget — this is
 * not a capture failure. */
void pn_graph_capture_dropped(pn_graph_capture_tracker *t);

/* This slot's eligibility gate failed. Records the reason and attempts
 * nothing; safe to call every step. A LIVE graph or a spent retry budget is
 * left alone — both are a truer description of the slot than "ineligible" —
 * and the failure budget is preserved, so a slot that is only TEMPORARILY
 * gated out (sprof active, this step's slot_base>0) resumes where it left off
 * when the gate reopens. */
void pn_graph_capture_ineligible(pn_graph_capture_tracker *t,
                                 pn_graph_ineligible_reason reason);

/**
 * This slot has been SUPERSEDED by a graph at a coarser granularity — the
 * per-die full-step segment graph swallowed the per-layer path, so those layer
 * slots are no longer part of any decode step.
 *
 * Without this a slot frozen mid-retry keeps reporting a deficit it can never
 * work off, because the code that would retry it is never reached again. The
 * measured symptom: a serve whose segment graphs were all live still read
 * `live 2 / total 32 / failed 30`. The lifetime capture/failure counters are
 * deliberately KEPT — what happened, happened — but the slot drops out of the
 * live/total headline until the coarser graph is dropped and the per-layer path
 * runs again, which re-evaluates it.
 */
void pn_graph_capture_superseded(pn_graph_capture_tracker *t);

/* Fold ONE tracker into a residency roll-up. Zero the roll-up first. */
void pn_graph_residency_add(pn_graph_residency *out,
                            const pn_graph_capture_tracker *t);

/* Short stable names for logs and JSON. */
const char *pn_graph_capture_state_name(unsigned char state);
const char *pn_graph_ineligible_reason_name(unsigned char reason);

/**
 * Test-only fault injection: force the first N capture attempts of EVERY slot
 * to be treated as failures, so the retry path can be exercised end to end on
 * real hardware. Read once from QWEN35_GRAPH_CAPTURE_FAIL_INJECT; 0 (unset) is
 * the production path and costs one cached compare.
 *
 * @return  How many attempts per slot to fail, 0 = never inject
 */
int pn_graph_capture_inject_failures(void);

/* Should THIS slot's current capture attempt be forced to fail? Keyed off the
 * slot's own lifetime failure count, so each slot fails exactly N times and
 * then behaves normally. Injecting N < PN_GRAPH_CAPTURE_MAX_FAILURES exercises
 * recovery; injecting more exercises the give-up path. */
int pn_graph_capture_inject_now(const pn_graph_capture_tracker *t);

#endif /* PN_GPU_GRAPH_CAPTURE_TRACKER_H */
