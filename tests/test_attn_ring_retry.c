/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attn_ring_retry.c
 * @brief Unit test for the attention-path tracker: the bounded retry after a
 *        failed per-slot KV-ring allocation, and the ondie/host residency
 *        gauge. Host-side, no GPU, no model.
 *
 * WHAT IT GUARDS
 *   afcad12 made a failed gpu_attn_slot_ring_ensure STICKY: the layer recorded
 *   the geometry and never attempted it again. The reason was sound — an
 *   unbraked retry re-attempted a 144 MiB cuMemAlloc every decode step and
 *   bumped slots_ring_gen with it, dropping every layer's captured graph, for
 *   a measured 62% collapse. The cure was worse in a different direction: ONE
 *   transient allocation failure pinned that layer to the host-softmax
 *   fallback for the life of the process. On the depth bench that fallback is
 *   14.58 -> 0.12 tok/s, with correct output and no error.
 *
 *   This test pins the three halves of "bounded, and not a latch":
 *
 *     - a TRANSIENT failure must recover (the whole point),
 *     - the retry must STOP (PN_ATTN_RING_MAX_FAILURES), and a success must
 *       reset the budget so recurring contention never exhausts it,
 *     - an IMPOSSIBLE geometry — one that could not fit an EMPTY die — must
 *       give up on the first failure, because there is nothing to wait for.
 *       That is the distinction that keeps a doomed layer from spending four
 *       rounds of pointless multi-hundred-MiB allocations.
 *
 *   Plus the gauge itself: ondie + host == total, layers the forward never
 *   reached are excluded rather than reported as a deficit, and the host
 *   reason is carried so an operator learns WHY, not just that.
 *
 * RED-FIRST
 *   `--latch` answers every allocation question with the OLD sticky rule — one
 *   attempt at a geometry, then never again — which is exactly what
 *   kv_slots_fail_{seq,n,q4} did. It MUST fail. A gate nobody has watched go
 *   red is not evidence, so the failing arm ships with the passing one.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <string.h>

#include "../src/models/gpu_attn_path_tracker.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* When set, the allocation question is answered by the OLD sticky latch
 * instead of the tracker's backoff — the red arm. */
static int latch_mode = 0;

/* One simulated attention layer: the tracker plus the sticky geometry record
 * the shipped code used, so both policies can be driven by the same step loop. */
typedef struct {
    pn_attn_path_tracker tracker;
    int rings_live;      /* the rings are allocated at the current geometry */
    int latch_failed;    /* the old `kv_slots_fail_n` != 0 */
    int alloc_attempts;  /* how many times an allocation was actually tried */
} sim_layer;

/* Ask the policy under test whether to attempt an allocation this step. */
static int sim_should_attempt(sim_layer *l)
{
    if (latch_mode) {
        if (l->latch_failed)
            return 0;
        return 1;
    }
    return pn_attn_ring_should_attempt(&l->tracker);
}

/* Run one decode step against a layer whose rings are missing.
 * `alloc_ok`      — would the allocation succeed if attempted this step?
 * `impossible`    — if it fails, is the geometry unfittable on an empty die? */
static void sim_step(sim_layer *l, int alloc_ok, int impossible)
{
    if (l->rings_live) {
        pn_attn_path_ondie(&l->tracker);
        return;
    }
    if (!sim_should_attempt(l)) {
        pn_attn_path_host(&l->tracker, PN_ATTN_HOST_RING_ALLOC);
        return;
    }
    l->alloc_attempts++;
    if (alloc_ok) {
        l->rings_live = 1;
        l->latch_failed = 0;
        pn_attn_ring_alloc_ok(&l->tracker);
        pn_attn_path_ondie(&l->tracker);
    } else {
        l->latch_failed = 1;
        pn_attn_ring_alloc_failed(&l->tracker, impossible);
        pn_attn_path_host(&l->tracker, PN_ATTN_HOST_RING_ALLOC);
    }
}

/* Drop the rings (a geometry change) without touching the failure budget —
 * what a slot-count or ctx_cap change at the SAME-or-larger geometry does. */
static void sim_geometry_changed(sim_layer *l)
{
    l->rings_live = 0;
}

/* A strictly SMALLER geometry: a different question, so both policies get to
 * forget the old verdict. */
static void sim_geometry_shrank(sim_layer *l)
{
    l->rings_live = 0;
    l->latch_failed = 0;
    pn_attn_ring_geometry_shrank(&l->tracker);
}

/* --------------------------------------------------------------------- */

static void test_transient_recovers(void)
{
    printf("\n1. a transient ring-allocation failure recovers on a later step\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    sim_step(&l, 0, 0);   /* the die is momentarily full */
    CHECK(l.tracker.state == PN_ATTN_PATH_HOST,
          "one failure puts the layer on the host path");
    CHECK(l.tracker.alloc_gave_up == 0,
          "one failure does NOT give up — it is a backoff, not a latch");
    CHECK(l.tracker.reason == PN_ATTN_HOST_RING_ALLOC,
          "the host reason is recorded as ring_alloc_failed");

    /* The memory frees up. Walk enough steps to clear the first backoff. */
    int recovered = 0;
    for (int step = 0; step < 64 && !recovered; step++) {
        sim_step(&l, 1, 0);
        recovered = l.rings_live;
    }
    CHECK(recovered, "the layer retried and got its rings back");
    CHECK(l.tracker.state == PN_ATTN_PATH_ONDIE,
          "and is reported ON-DIE again");
    CHECK(l.alloc_attempts == 2,
          "exactly one retry was spent, not one per step");
    CHECK(l.tracker.lifetime_alloc_failures == 1,
          "the failure is still on the lifetime counter — what happened, happened");
}

static void test_retry_is_bounded(void)
{
    printf("\n2. the retry STOPS: a permanently squeezed die costs a fixed budget\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int step = 0; step < 5000; step++)
        sim_step(&l, 0, 0);   /* never succeeds, never "impossible" */

    CHECK(l.alloc_attempts == PN_ATTN_RING_MAX_FAILURES,
          "exactly PN_ATTN_RING_MAX_FAILURES allocations were attempted");
    CHECK(l.tracker.alloc_gave_up == 1, "the layer gave up");
    CHECK(l.tracker.alloc_impossible == 0,
          "gave up on the BUDGET, not because the geometry is impossible");
    CHECK(l.tracker.state == PN_ATTN_PATH_HOST,
          "and still honestly reports the host path");

    /* 5000 steps at 5 attempts is the whole point: the un-latched version this
     * replaces would have attempted 5000. */
    CHECK(l.alloc_attempts < 10,
          "5000 steps cost fewer than 10 allocation attempts");
}

static void test_impossible_gives_up_at_once(void)
{
    printf("\n3. an IMPOSSIBLE geometry gives up on the FIRST failure\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int step = 0; step < 5000; step++)
        sim_step(&l, 0, 1);   /* would not fit an empty die */

    CHECK(l.alloc_attempts == 1,
          "one attempt only — there is nothing transient to wait for");
    CHECK(l.tracker.alloc_gave_up == 1, "gave up immediately");
    CHECK(l.tracker.alloc_impossible == 1,
          "and says WHY: the request exceeds the die's usable VRAM");
}

static void test_success_resets_the_budget(void)
{
    printf("\n4. recurring contention never exhausts the budget\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int round = 0; round < 50; round++) {
        sim_geometry_changed(&l);
        sim_step(&l, 0, 0);            /* collide once */
        int recovered = 0;
        for (int step = 0; step < 64 && !recovered; step++) {
            sim_step(&l, 1, 0);
            recovered = l.rings_live;
        }
        if (!recovered) {
            CHECK(0, "round recovered");
            break;
        }
    }
    CHECK(l.rings_live, "still on-die after 50 collide/recover rounds");
    CHECK(l.tracker.alloc_gave_up == 0, "never gave up");
    CHECK(l.tracker.lifetime_alloc_failures == 50,
          "all 50 collisions are counted");
    CHECK(l.alloc_attempts == 100,
          "each round cost exactly one failure plus one success");
}

static void test_shrinking_geometry_recovers_now(void)
{
    printf("\n5. a SMALLER geometry recovers immediately, with no backoff\n");
    sim_layer l;
    memset(&l, 0, sizeof(l));

    for (int step = 0; step < 5000; step++)
        sim_step(&l, 0, 1);   /* impossible at this geometry: gave up */
    CHECK(l.tracker.alloc_gave_up == 1, "gave up at the large geometry");

    /* The operator drops max_sessions, or the ladder drops ctx_cap. */
    sim_geometry_shrank(&l);
    int before = l.alloc_attempts;
    sim_step(&l, 1, 0);
    CHECK(l.alloc_attempts == before + 1,
          "the very next step attempts the smaller geometry");
    CHECK(l.rings_live, "and it lands");
    CHECK(l.tracker.state == PN_ATTN_PATH_ONDIE,
          "back on-die without a reload");
}

static void test_residency_gauge(void)
{
    printf("\n6. the gauge: ondie + host == total, and unreached layers are not a deficit\n");
    pn_attn_path_tracker layers[10];
    memset(layers, 0, sizeof(layers));

    /* Seven layers ran on-die, two fell back for different reasons, one was
     * never reached by the forward at all. */
    for (int i = 0; i < 7; i++)
        pn_attn_path_ondie(&layers[i]);
    pn_attn_path_host(&layers[7], PN_ATTN_HOST_RING_ALLOC);
    pn_attn_ring_alloc_failed(&layers[7], 1);
    pn_attn_path_host(&layers[8], PN_ATTN_HOST_RING_WRAP);
    /* layers[9] untouched — UNKNOWN */

    pn_attn_residency r;
    memset(&r, 0, sizeof(r));
    for (int i = 0; i < 10; i++)
        pn_attn_residency_add(&r, &layers[i]);

    CHECK(r.total == 9, "the layer the forward never reached is not counted");
    CHECK(r.ondie == 7 && r.host == 2, "seven on-die, two on the host path");
    CHECK(r.ondie + r.host == r.total, "the states sum to the total");
    CHECK(r.by_reason[PN_ATTN_HOST_RING_ALLOC] == 1 &&
          r.by_reason[PN_ATTN_HOST_RING_WRAP] == 1,
          "both host reasons are reported separately");
    CHECK(r.gave_up == 1 && r.impossible == 1,
          "the impossible allocation is flagged gave_up AND impossible");
    CHECK(strcmp(pn_attn_host_reason_name(PN_ATTN_HOST_RING_WRAP),
                 "kv_geometry_or_ring_wrap") == 0,
          "the reason has a stable name for the JSON");

    /* A layer that comes back on-die must LEAVE the host count — a gauge that
     * cannot go back down is a log line, not a gauge. */
    pn_attn_path_ondie(&layers[8]);
    memset(&r, 0, sizeof(r));
    for (int i = 0; i < 10; i++)
        pn_attn_residency_add(&r, &layers[i]);
    CHECK(r.ondie == 8 && r.host == 1, "recovery is visible on the gauge");
    CHECK(r.lifetime_host_engagements == 2,
          "but the lifetime engagement count keeps the history");
}

static void test_host_engagements_count_incidents(void)
{
    printf("\n7. host engagements count INCIDENTS, not tokens\n");
    pn_attn_path_tracker t;
    memset(&t, 0, sizeof(t));

    for (int step = 0; step < 1000; step++)
        pn_attn_path_host(&t, PN_ATTN_HOST_RING_WRAP);
    CHECK(t.lifetime_host_engagements == 1,
          "1000 steps parked on the host path is ONE engagement");

    pn_attn_path_ondie(&t);
    pn_attn_path_host(&t, PN_ATTN_HOST_RING_WRAP);
    CHECK(t.lifetime_host_engagements == 2,
          "leaving and re-entering the host path is a second engagement");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--latch") == 0)
            latch_mode = 1;

    printf("=== attention ring-allocation retry + residency ===\n");
    if (latch_mode)
        printf("MODE: --latch (the OLD sticky behaviour — this arm MUST fail)\n");

    test_transient_recovers();
    test_retry_is_bounded();
    test_impossible_gives_up_at_once();
    test_success_resets_the_budget();
    test_shrinking_geometry_recovers_now();
    test_residency_gauge();
    test_host_engagements_count_incidents();

    printf("\n%d check(s) failed\n", failures);
    if (latch_mode) {
        /* Inverted: the red arm exists to be seen failing. */
        printf(failures > 0 ? "[PASS] the latch arm failed, as it must\n"
                            : "[FAIL] the latch arm PASSED — the gate is vacuous\n");
        return failures > 0 ? 0 : 1;
    }
    printf(failures == 0 ? "[PASS]\n" : "[FAIL]\n");
    return failures == 0 ? 0 : 1;
}
