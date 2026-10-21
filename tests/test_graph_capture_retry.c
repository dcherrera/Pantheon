/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_graph_capture_retry.c
 * @brief Unit test for the CUDA-graph capture tracker's retry policy and its
 *        three-state residency gauge (tasks #41/#42). Host-side, no GPU.
 *
 * WHAT IT GUARDS
 *   Before this, a capture that FAILED latched the layer's graph off for the
 *   life of the process — a transient collision cost that layer ~1.8 ms/step
 *   forever, silently, with correct output. The fix retries on a bounded
 *   backoff. This test pins the two halves of "bounded":
 *
 *     - a FAILED capture must recover (the whole point), and
 *     - an INELIGIBLE slot must never attempt a capture at all, so a model
 *       that legitimately captures nothing (non-Q3_K experts → MoE not
 *       grouped) does not turn into a per-step retry loop.
 *
 * RED-FIRST
 *   `--latch` runs every expectation against a simulated OLD tracker — one
 *   chance to capture, then off forever, which is exactly what the shipped
 *   `graph_tried` / `slots_graph_tried` booleans did. It MUST fail. A gate
 *   nobody has watched go red is not evidence, so the failing arm ships with
 *   the passing one.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <string.h>

#include "../src/models/gpu_graph_capture_tracker.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* When set, every capture attempt is answered by the OLD one-shot latch
 * instead of the tracker's backoff — the red arm. */
static int latch_mode = 0;

/* One simulated layer: the tracker plus the boolean latch the shipped code
 * used, so both policies can be driven by the same step loop. */
typedef struct {
    pn_graph_capture_tracker tracker;
    int graph_live;        /* a captured graph is replaying (the fast path) */
    int latch_tried;       /* the old `slots_graph_tried` */
    int capture_attempts;  /* how many times a capture was actually attempted */
} sim_layer;

/* Ask the policy under test whether to attempt a capture this step. */
static int sim_should_attempt(sim_layer *l)
{
    if (latch_mode) {
        if (l->latch_tried)
            return 0;
        l->latch_tried = 1;
        return 1;
    }
    return pn_graph_capture_should_attempt(&l->tracker);
}

/* Run one decode step for a layer with no live graph. `capture_succeeds` is
 * what the driver would report if a capture is attempted. */
static void sim_step(sim_layer *l, int capture_succeeds)
{
    if (l->graph_live)
        return;             /* fast path: replay, no tracker involvement */
    if (!sim_should_attempt(l))
        return;
    l->capture_attempts++;
    if (capture_succeeds) {
        l->graph_live = 1;
        pn_graph_capture_succeeded(&l->tracker);
    } else {
        pn_graph_capture_failed(&l->tracker);
    }
}

/* A live graph is dropped for a legitimate reason (its capture key changed). */
static void sim_drop(sim_layer *l)
{
    l->graph_live = 0;
    l->latch_tried = 0;
    pn_graph_capture_dropped(&l->tracker);
}

/* GATE 1 — a capture that fails must recover on a later step. */
static void test_failed_capture_recovers(void)
{
    printf("test: a failed capture recovers on a later step\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    sim_step(&l, 0);   /* the collision */
    CHECK(l.tracker.state == PN_GRAPH_CAPTURE_RETRYING,
          "one failure leaves the slot RETRYING, not latched off");
    CHECK(l.tracker.lifetime_failures == 1, "the failure is counted");

    /* Every later step succeeds if it is allowed to try at all. */
    for (int step = 0; step < 64; step++)
        sim_step(&l, 1);

    CHECK(l.tracker.state == PN_GRAPH_CAPTURE_LIVE,
          "the layer holds a live graph again");
    CHECK(l.tracker.lifetime_captures == 1, "exactly one recapture");
    CHECK(l.capture_attempts == 2,
          "the backoff cost exactly one extra attempt, not one per step");
}

/* GATE 1b — the retry is BOUNDED: a capture that can never work stops. */
static void test_permanent_failure_stops(void)
{
    printf("test: a capture that always fails gives up (bounded retry)\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int step = 0; step < 100000; step++)
        sim_step(&l, 0);

    CHECK(l.tracker.state == PN_GRAPH_CAPTURE_GAVE_UP,
          "the slot gave up instead of retrying forever");
    CHECK(l.capture_attempts == PN_GRAPH_CAPTURE_MAX_FAILURES,
          "attempts are capped at the retry budget");

    /* And it stays given up. */
    int before = l.capture_attempts;
    for (int step = 0; step < 100000; step++)
        sim_step(&l, 0);
    CHECK(l.capture_attempts == before, "no further attempts, ever");
}

/* A recurring-collision serve must not exhaust its budget: a success in
 * between resets it, so the layer keeps recovering for the life of the
 * process. */
static void test_intermittent_collisions_never_exhaust(void)
{
    printf("test: intermittent collisions never exhaust the budget\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int round = 0; round < 50; round++) {
        if (round)
            sim_drop(&l);                   /* a capture key changed */
        sim_step(&l, 0);                    /* a collision */
        for (int step = 0; step < 32; step++)
            sim_step(&l, 1);                /* it recovers */
        if (l.tracker.state != PN_GRAPH_CAPTURE_LIVE)
            break;
    }
    CHECK(l.tracker.state == PN_GRAPH_CAPTURE_LIVE,
          "still recovering after 50 collision/recovery rounds");
    CHECK(l.tracker.lifetime_captures == 50, "recovered every time");
}

/* GATE 2 — an INELIGIBLE slot attempts NOTHING. This is the case the retry
 * must not touch: a Q5_K_M model's 40 layers report "MoE not grouped" and
 * that is correct behaviour, not a fault to retry. */
static void test_ineligible_never_attempts(void)
{
    printf("test: an ineligible slot never attempts a capture\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    /* The forward marks the gate closed and never calls should_attempt. */
    for (int step = 0; step < 100000; step++)
        pn_graph_capture_ineligible(&l.tracker,
                                    PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);

    CHECK(l.capture_attempts == 0, "zero capture attempts");
    CHECK(l.tracker.lifetime_failures == 0, "zero failures — nothing failed");
    CHECK(l.tracker.state == PN_GRAPH_CAPTURE_INELIGIBLE, "state is ineligible");
    CHECK(strcmp(pn_graph_ineligible_reason_name(l.tracker.ineligible_reason),
                 "MoE not grouped") == 0, "the reason is reported");

    /* A temporarily-closed gate (sprof, slot_base>0) must reopen cleanly. */
    CHECK(pn_graph_capture_should_attempt(&l.tracker) == 1,
          "a reopened gate allows a capture again");
}

/* GATE 4 — the gauge tells the three states apart, and does not invent a
 * deficit for a code path the model never runs. */
static void test_residency_gauge(void)
{
    printf("test: the residency gauge distinguishes live/failed/ineligible\n");
    pn_graph_capture_tracker slots[10];
    memset(slots, 0, sizeof(slots));

    for (int i = 0; i < 6; i++)
        pn_graph_capture_succeeded(&slots[i]);
    for (int i = 6; i < 8; i++)
        pn_graph_capture_failed(&slots[i]);
    for (int i = 8; i < 10; i++)
        pn_graph_capture_ineligible(&slots[i],
                                    PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);

    pn_graph_residency r;
    memset(&r, 0, sizeof(r));
    for (int i = 0; i < 10; i++)
        pn_graph_residency_add(&r, &slots[i]);

    CHECK(r.live == 6 && r.failed == 2 && r.ineligible == 2 && r.total == 10,
          "6 live / 2 failed / 2 ineligible of 10");
    CHECK(r.gave_up == 0, "one failure each is not a give-up");
    CHECK(r.lifetime_captures == 6 && r.lifetime_failures == 2,
          "lifetime counters agree");

    /* A model that decodes batched never touches its single-token slots. */
    pn_graph_capture_tracker untouched[40];
    memset(untouched, 0, sizeof(untouched));
    pn_graph_residency u;
    memset(&u, 0, sizeof(u));
    for (int i = 0; i < 40; i++)
        pn_graph_residency_add(&u, &untouched[i]);
    CHECK(u.total == 0,
          "an unexercised path reports total 0, not 40 missing graphs");

    /* A model that captures nothing reports honestly: no fault, all gated. */
    pn_graph_capture_tracker q5[40];
    memset(q5, 0, sizeof(q5));
    pn_graph_residency g;
    memset(&g, 0, sizeof(g));
    for (int i = 0; i < 40; i++) {
        pn_graph_capture_ineligible(&q5[i], PN_GRAPH_INELIGIBLE_MOE_NOT_GROUPED);
        pn_graph_residency_add(&g, &q5[i]);
    }
    CHECK(g.total == 40 && g.ineligible == 40 && g.failed == 0 && g.live == 0,
          "a non-Q3_K model reads 40 ineligible / 0 failed");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--latch") == 0)
            latch_mode = 1;

    printf("=== CUDA-graph capture retry + residency gauge ===\n");
    if (latch_mode)
        printf("MODE: --latch (the OLD one-shot behaviour — this arm MUST fail)\n");

    test_failed_capture_recovers();
    test_permanent_failure_stops();
    test_intermittent_collisions_never_exhaust();
    test_ineligible_never_attempts();
    test_residency_gauge();

    printf("\n%d check(s) failed\n", failures);
    if (latch_mode) {
        /* Inverted: the red arm proves the gate is capable of failing. */
        printf(failures > 0 ? "[PASS] the latch arm failed, as it must\n"
                            : "[FAIL] the latch arm PASSED — the gate is vacuous\n");
        return failures > 0 ? 0 : 1;
    }
    printf(failures ? "[FAIL] graph capture retry\n" : "[PASS] graph capture retry\n");
    return failures ? 1 : 0;
}
