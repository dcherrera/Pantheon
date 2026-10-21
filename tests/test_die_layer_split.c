/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_die_layer_split.c — the N-die layer partitioner, and the one property
 * that has to hold for the deployed config: a 2-die plan is byte-identical to the hardcoded
 * `n_layer / 2` boundary it replaces.
 *
 * The bug this partitioner exists to kill: qwen35moe_forward_cuda.c split at
 * `n_layer / 2` for ANY placement of two or more dies, while model_registry.c
 * projected `n_layer * idx / n_placed` per die. The two agree at exactly one
 * width. A `--dies 5,6,7` placement therefore bound three dies, uploaded a
 * TWO-way split onto the first two, priced each die at a third of the model,
 * and OOM'd on the die that got half. Both sides now call this module, so the
 * disagreement is not expressible.
 *
 * The N=2 identity is asserted against the OLD expression, computed inline
 * here, over every layer count from 2 to 200 — even and odd — rather than
 * spot-checked at 40. Odd counts are where a "remainder to the earlier dies"
 * partitioner would silently move the deployed config's boundary by one layer.
 */

#include "../src/accel/die_layer_split.h"

#include <stdio.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

/* Qwen3.6-35B-A3B, the model this work is for. */
#define QWEN36_N_LAYER 40

/* Every plan must satisfy these, whatever N is: the ranges tile [0,n_layer)
 * with no gap, no overlap and no empty die, and no two dies differ by more
 * than one layer. Returns the widest range length seen minus the narrowest. */
static int check_plan_is_a_partition(const pn_die_layer_split *p, int n_layer,
                                     const char *what)
{
    int lo_min = p->layer_hi[0] - p->layer_lo[0];
    int lo_max = lo_min;
    CHECK(p->layer_lo[0] == 0, what);
    CHECK(p->layer_hi[p->n_die - 1] == n_layer, what);
    for (int i = 0; i < p->n_die; i++) {
        int len = p->layer_hi[i] - p->layer_lo[i];
        CHECK(len >= 1, what);
        if (i > 0)
            CHECK(p->layer_lo[i] == p->layer_hi[i - 1], what);
        if (len < lo_min) lo_min = len;
        if (len > lo_max) lo_max = len;
    }
    return lo_max - lo_min;
}

/* The 2-die boundary the CUDA forward hardcoded before this module existed. */
static int legacy_two_die_boundary(int n_layer)
{
    return n_layer / 2;
}

static void test_two_die_matches_the_hardcoded_split(void)
{
    printf("two_die_matches_the_hardcoded_split:\n");
    int mismatches = 0;
    for (int n_layer = 2; n_layer <= 200; n_layer++) {
        pn_die_layer_split p;
        if (pn_plan_die_layer_split(n_layer, 2, 0, &p) != 0) {
            mismatches++;
            continue;
        }
        int want = legacy_two_die_boundary(n_layer);
        if (p.layer_lo[0] != 0 || p.layer_hi[0] != want ||
            p.layer_lo[1] != want || p.layer_hi[1] != n_layer)
            mismatches++;
    }
    CHECK(mismatches == 0,
          "every 2-die plan from 2 to 200 layers splits at exactly n_layer/2");
    if (mismatches)
        fprintf(stderr, "  (%d layer counts disagreed with n_layer/2)\n",
                mismatches);

    /* Odd counts are the interesting half: the remainder must land on the LAST
     * die, matching integer n_layer/2, not on the first. */
    pn_die_layer_split odd;
    CHECK(pn_plan_die_layer_split(41, 2, 0, &odd) == 0, "41 layers, 2 dies");
    CHECK(odd.layer_hi[0] == 20 && odd.layer_hi[1] == 41,
          "41 layers is 20 + 21, the way n_layer/2 has always cut it");
}

static void test_one_die_is_the_whole_model(void)
{
    printf("one_die_is_the_whole_model:\n");
    pn_die_layer_split p;
    CHECK(pn_plan_die_layer_split(QWEN36_N_LAYER, 1, 0, &p) == 0, "plans");
    CHECK(p.n_die == 1, "one die");
    CHECK(p.layer_lo[0] == 0 && p.layer_hi[0] == QWEN36_N_LAYER,
          "which holds every layer");
    /* A single-die placement is authoritative: the boundary knob cannot turn
     * it into a split. */
    CHECK(pn_plan_die_layer_split(QWEN36_N_LAYER, 1, 20, &p) == 0,
          "plans with an override");
    CHECK(p.layer_hi[0] == QWEN36_N_LAYER,
          "and the override does not carve up a single die");
}

static void test_wide_splits_are_flat_and_contiguous(void)
{
    printf("wide_splits_are_flat_and_contiguous:\n");
    for (int n_die = 1; n_die <= PN_DIE_LAYER_SPLIT_MAX_DIES; n_die++) {
        pn_die_layer_split p;
        CHECK(pn_plan_die_layer_split(QWEN36_N_LAYER, n_die, 0, &p) == 0,
              "Qwen3.6's 40 layers plan at every width up to the ceiling");
        CHECK(p.n_die == n_die, "at the requested width");
        int spread = check_plan_is_a_partition(&p, QWEN36_N_LAYER,
                                               "40 layers tile exactly");
        CHECK(spread <= 1, "and no two dies differ by more than one layer");
    }
    /* The widths this work is actually for. */
    pn_die_layer_split p3, p4;
    CHECK(pn_plan_die_layer_split(40, 3, 0, &p3) == 0, "40 over 3");
    CHECK(p3.layer_hi[0] == 13 && p3.layer_hi[1] == 26 && p3.layer_hi[2] == 40,
          "40 over 3 dies is 13 + 13 + 14");
    CHECK(pn_plan_die_layer_split(40, 4, 0, &p4) == 0, "40 over 4");
    CHECK(p4.layer_hi[0] == 10 && p4.layer_hi[1] == 20 &&
          p4.layer_hi[2] == 30 && p4.layer_hi[3] == 40,
          "40 over 4 dies is 10 + 10 + 10 + 10");
}

/* An unsupported width must REFUSE, never mis-split. Silently narrowing is
 * precisely what produced the three-dies-bound / two-dies-uploaded state. */
static void test_unsupported_widths_refuse(void)
{
    printf("unsupported_widths_refuse:\n");
    pn_die_layer_split p;
    p.n_die = -7;
    CHECK(pn_plan_die_layer_split(40, 0, 0, &p) == -1, "zero dies refuses");
    CHECK(pn_plan_die_layer_split(40, -1, 0, &p) == -1, "negative refuses");
    CHECK(pn_plan_die_layer_split(40, PN_DIE_LAYER_SPLIT_MAX_DIES + 1, 0, &p)
              == -1,
          "one die past the ceiling refuses");
    CHECK(pn_plan_die_layer_split(40, 64, 0, &p) == -1, "far past refuses");
    CHECK(pn_plan_die_layer_split(0, 2, 0, &p) == -1, "no layers refuses");
    CHECK(pn_plan_die_layer_split(3, 4, 0, &p) == -1,
          "more dies than layers refuses -- an empty die is a bug, not a split");
    CHECK(pn_plan_die_layer_split(40, 2, 0, NULL) == -1, "NULL out refuses");
    CHECK(p.n_die == -7, "and a refusal leaves the caller's plan untouched");
}

/* QWEN35_GPU_SPLIT_LAYER tunes ONE boundary. That describes a 2-die split
 * completely and nothing wider, so it applies at exactly that width and is
 * ignored elsewhere -- documented, not silent. */
static void test_boundary_override_is_two_die_only(void)
{
    printf("boundary_override_is_two_die_only:\n");
    pn_die_layer_split p;
    CHECK(pn_plan_die_layer_split(40, 2, 12, &p) == 0, "2-die with override");
    CHECK(p.layer_hi[0] == 12 && p.layer_lo[1] == 12,
          "the override moves the single boundary");
    CHECK(check_plan_is_a_partition(&p, 40, "an overridden plan still tiles")
              >= 0, "and still tiles [0,40)");

    CHECK(pn_plan_die_layer_split(40, 3, 12, &p) == 0, "3-die with override");
    CHECK(p.layer_hi[0] == 13 && p.layer_hi[1] == 26,
          "at 3 dies the override is ignored, not applied to boundary 0");

    /* Out-of-range knobs narrow the split; they never empty a die. */
    CHECK(pn_plan_die_layer_split(40, 2, 40, &p) == 0, "override == n_layer");
    CHECK(p.layer_hi[0] == 39 && p.layer_hi[1] == 40,
          "clamped so die 1 keeps at least one layer");
    CHECK(pn_plan_die_layer_split(40, 2, 999, &p) == 0, "override past the end");
    CHECK(p.layer_hi[0] == 39, "clamped the same way");
}

/* The single-index view has to be the plan viewed through one index, or the
 * two derivations drift apart exactly the way the registry and the forward
 * did. Asserted across every width and every index. */
static void test_single_range_matches_the_full_plan(void)
{
    printf("single_range_matches_the_full_plan:\n");
    int mismatches = 0;
    for (int n_layer = 1; n_layer <= 128; n_layer++) {
        for (int n_die = 1;
             n_die <= PN_DIE_LAYER_SPLIT_MAX_DIES && n_die <= n_layer;
             n_die++) {
            pn_die_layer_split p;
            if (pn_plan_die_layer_split(n_layer, n_die, 0, &p) != 0) {
                mismatches++;
                continue;
            }
            for (int i = 0; i < n_die; i++) {
                int lo = -1, hi = -1;
                if (pn_die_layer_range(n_layer, n_die, i, &lo, &hi) != 0 ||
                    lo != p.layer_lo[i] || hi != p.layer_hi[i])
                    mismatches++;
            }
        }
    }
    CHECK(mismatches == 0,
          "pn_die_layer_range agrees with the plan at every width and index");
    if (mismatches)
        fprintf(stderr, "  (%d disagreements)\n", mismatches);

    int lo = -1, hi = -1;
    CHECK(pn_die_layer_range(40, 3, 3, &lo, &hi) == -1, "index == n_die refuses");
    CHECK(pn_die_layer_range(40, 3, -1, &lo, &hi) == -1, "negative index refuses");
    CHECK(lo == -1 && hi == -1, "and a refusal writes nothing");
}

int main(void)
{
    test_two_die_matches_the_hardcoded_split();
    test_one_die_is_the_whole_model();
    test_wide_splits_are_flat_and_contiguous();
    test_unsupported_widths_refuse();
    test_boundary_override_is_two_die_only();
    test_single_range_matches_the_full_plan();

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("test_die_layer_split: all checks passed\n");
    return 0;
}
