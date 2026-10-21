/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_qwen35moe_metal_coverage.c — the Metal kernel-coverage gate.
 *
 * The gate asserts that every weight the qwen35moe forward actually dispatches
 * a kernel at has a Metal matvec kernel registered for its quant type. Without
 * it, a missing kernel for one of the six quant types in the stock Q3_K_M file
 * surfaces as garbage logits at end-to-end bring-up (Phase 5), where it is easy
 * to blame the SSM or attention instead.
 *
 * This file is compiled TWICE, and that is what proves the gate measures
 * something:
 *
 *   tests/test_qwen35moe_metal_coverage        — plain. Reports the real
 *       uncovered set, and (with PN_METAL_COVERAGE_DROP_TYPE) shows that
 *       hiding a type from the dispatch derivation is caught as UNACCOUNTED
 *       rather than passing silently.
 *
 *   tests/test_qwen35moe_metal_coverage_fake   — built with
 *       -DPN_COVERAGE_FAKE_KERNEL, which links a stub `fm_matvec_q5k`. Q5_K
 *       must flip to covered while Q3_K/Q6_K/Q8_0 stay uncovered. A gate that
 *       printed a constant could not tell those two binaries apart.
 *
 * The model-gated tests load the real ~15 GiB GGUF, so they need
 * PN_QWEN35_MODEL=<path to the .gguf>; `make test` skips them otherwise. They
 * run ONE forward token (~0.25 s on CPU) and never generate.
 */

#include "lib/test.h"
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe_internal.h"
#include "../src/models/qwen35moe_metal_coverage.h"
#include "../src/models/qwen35moe_metal_kernel_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef PN_COVERAGE_FAKE_KERNEL
/*
 * A stub standing in for the Phase-1 Q5_K matvec that foundry does not have
 * yet. Never called — the registry only asks the loader whether the symbol
 * resolves. Its whole job is to move exactly one type from uncovered to
 * covered so the gate's output can be shown to track linkage.
 */
void fm_matvec_q5k(void);
void fm_matvec_q5k(void) { }
#endif

/* Default search paths, tried when PN_QWEN35_MODEL is unset.
 *
 * A skip that has to be opted OUT of is the safe default here. The model is
 * large but it is normally present on a dev box, and the failure this gate
 * exists to catch is a missing kernel — exactly the thing a silent skip would
 * hide behind a green suite. So look for the model before giving up. */
/* Relative to the working directory, then to $HOME. */
static const char *const k_default_models[] = {
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf",
};

static const char *model_path(void)
{
    const char *p = getenv("PN_QWEN35_MODEL");
    if (p && *p && access(p, F_OK) == 0) return p;
    if (p && *p) return NULL;   /* explicitly pointed somewhere missing — do not second-guess */

    static char home_path[4096];
    const char *home = getenv("HOME");
    for (size_t i = 0; i < sizeof(k_default_models)/sizeof(*k_default_models); i++) {
        if (access(k_default_models[i], F_OK) == 0) return k_default_models[i];
        if (home && snprintf(home_path, sizeof(home_path), "%s/%s", home, k_default_models[i])
                        < (int)sizeof(home_path) &&
            access(home_path, F_OK) == 0)
            return home_path;
    }
    return NULL;
}

/* Load the model and hand back its qwen35moe weights, or NULL when skipping. */
static pn_qwen35moe_model *load_qwen35moe(pn_model **model_out)
{
    const char *path = model_path();
    if (!path) {
        printf("    [SKIP] set PN_QWEN35_MODEL to the GGUF to run\n");
        return NULL;
    }
    pn_model *m = pn_load(path);
    if (!m) return NULL;
    *model_out = m;
    return (pn_qwen35moe_model *)m->weights;
}

static const pn_qwen35moe_metal_coverage_row *
row_for(const pn_qwen35moe_metal_coverage *cov, pn_gguf_quant type)
{
    for (int i = 0; i < cov->n_rows; i++)
        if (cov->rows[i].type == type)
            return &cov->rows[i];
    return NULL;
}

/* ============================================================
 * Registry — no model needed
 * ============================================================ */

/*
 * The registry answers by symbol lookup, so its answer must track what is
 * actually linked. Q4_K's matvec ships in fm_metal today, so it resolves in a
 * USE_METAL build and cannot resolve without one. Q2_K has no planned kernel
 * at all and must never resolve.
 */
static void test_registry_tracks_linkage(void)
{
    const char *entry = NULL;

    int q4k = pn_qwen35moe_metal_kernel_for_type(GGUF_QUANT_Q4_K, &entry);
#ifdef USE_METAL
    ASSERT_INT_EQ(1, q4k);
    ASSERT_NOT_NULL((void *)entry);
#else
    ASSERT_INT_EQ(0, q4k);
    ASSERT_NULL((void *)entry);
#endif

    ASSERT_INT_EQ(0, pn_qwen35moe_metal_kernel_for_type(GGUF_QUANT_Q2_K, &entry));
    ASSERT_NULL((void *)entry);
}

/*
 * The Q5_K answer is the difference between the two binaries this file builds.
 * Same source, same model, same gate — only a linked stub differs.
 */
static void test_registry_q5k_matches_build(void)
{
    const char *entry = NULL;
    int q5k = pn_qwen35moe_metal_kernel_for_type(GGUF_QUANT_Q5_K, &entry);
    /* Off Apple the registry reports nothing linked by design (no Metal), so the fake
     * kernel can only move the answer on macOS. */
#if defined(PN_COVERAGE_FAKE_KERNEL) && defined(__APPLE__)
    ASSERT_INT_EQ(1, q5k);
    ASSERT_STR_EQ("fm_matvec_q5k", entry);
#else
    ASSERT_INT_EQ(0, q5k);
    ASSERT_NULL((void *)entry);
#endif
}

/* ============================================================
 * The gate itself — model-gated
 * ============================================================ */

/*
 * The headline run: trace the dispatch set off a real forward, print the
 * per-type table, and hold the classification to a conservation law — every
 * tensor in the file is dispatched, excluded, or unaccounted for, and nothing
 * falls off the edge. That law is what makes "no false alarm" and "nothing
 * silently skipped" checkable rather than asserted.
 */
static void test_coverage_gate_reports_real_state(void)
{
    pn_model *model = NULL;
    pn_qwen35moe_model *qw = load_qwen35moe(&model);
    if (!qw) return;

    pn_qwen35moe_metal_coverage cov = {0};
    int rc = pn_qwen35moe_metal_coverage_run(qw, &cov);
    pn_qwen35moe_metal_coverage_report(&cov, stdout);

    ASSERT_INT_EQ(0, rc);
    ASSERT_INT_EQ(1, cov.traced);

    /* Conservation: nothing is dropped on the floor. */
    ASSERT_INT_EQ(cov.file_tensors,
                  cov.dispatched_tensors + cov.excluded_tensors +
                  cov.unaccounted_tensors);

    /* The derivation is complete: every bound weight was seen dispatched. */
    ASSERT_INT_EQ(0, cov.unaccounted_tensors);
    ASSERT_INT_EQ(0, cov.unresolved_weights);

    /* Real work happened, and far more weights than tensors — the merged
     * expert tensors are reached through thousands of slices. */
    ASSERT_TRUE(cov.dispatched_tensors > 0);
    ASSERT_TRUE(cov.dispatched_weights > cov.dispatched_tensors);

    /* No false alarm: the file's norms and embeddings are never dispatched and
     * must not be demanding a kernel. */
    ASSERT_TRUE(cov.excluded_tensors > 0);

    /* Verdict follows the findings, not a constant. */
    ASSERT_INT_EQ(cov.uncovered_tensors == 0,
                  pn_qwen35moe_metal_coverage_passed(&cov));

    pn_model_free(model);
}

/*
 * Per-type expectations. Q3_K, Q6_K and Q8_0 have no kernel in either binary
 * (note Q8_0 in particular: fm_matvec_q8 exists but is a single-scale int8 dot,
 * the wrong layout for block-scaled Q8_0, and the registry deliberately refuses
 * to count it). Q5_K is the one that moves between builds.
 */
static void test_coverage_rows_track_registration(void)
{
    pn_model *model = NULL;
    pn_qwen35moe_model *qw = load_qwen35moe(&model);
    if (!qw) return;

    pn_qwen35moe_metal_coverage cov = {0};
    ASSERT_INT_EQ(0, pn_qwen35moe_metal_coverage_run(qw, &cov));

    const pn_gguf_quant never[] = { GGUF_QUANT_Q3_K, GGUF_QUANT_Q6_K,
                                    GGUF_QUANT_Q8_0 };
    for (size_t i = 0; i < sizeof(never) / sizeof(never[0]); i++) {
        const pn_qwen35moe_metal_coverage_row *r = row_for(&cov, never[i]);
        ASSERT_NOT_NULL((void *)r);
        ASSERT_INT_EQ(0, r->covered);
        ASSERT_TRUE(r->tensors > 0);
        ASSERT_TRUE(r->example[0] != '\0');
    }

    const pn_qwen35moe_metal_coverage_row *q5k = row_for(&cov, GGUF_QUANT_Q5_K);
    ASSERT_NOT_NULL((void *)q5k);
#ifdef PN_COVERAGE_FAKE_KERNEL
    ASSERT_INT_EQ(1, q5k->covered);
#else
    ASSERT_INT_EQ(0, q5k->covered);
#endif

    pn_model_free(model);
}

/*
 * Prove the gate can fail for the OTHER reason: hide one dispatched quant type
 * from the derivation and its tensors must resurface as UNACCOUNTED, named,
 * with the verdict flipped — not quietly vanish from the table. This is the
 * check that a hardcoded dispatch list would fail.
 */
static void test_dropping_a_type_is_caught(void)
{
    pn_model *model = NULL;
    pn_qwen35moe_model *qw = load_qwen35moe(&model);
    if (!qw) return;

    char drop[16];
    snprintf(drop, sizeof(drop), "%d", (int)GGUF_QUANT_Q5_K);
    setenv("PN_METAL_COVERAGE_DROP_TYPE", drop, 1);

    pn_qwen35moe_metal_coverage cov = {0};
    ASSERT_INT_EQ(0, pn_qwen35moe_metal_coverage_run(qw, &cov));
    pn_qwen35moe_metal_coverage_report(&cov, stdout);

    unsetenv("PN_METAL_COVERAGE_DROP_TYPE");

    ASSERT_TRUE(cov.unaccounted_tensors > 0);
    ASSERT_TRUE(cov.n_unaccounted_named > 0);
    ASSERT_INT_EQ(0, pn_qwen35moe_metal_coverage_passed(&cov));
    ASSERT_NULL((void *)row_for(&cov, GGUF_QUANT_Q5_K));

    /* Still conserved — the tensors moved class, they did not disappear. */
    ASSERT_INT_EQ(cov.file_tensors,
                  cov.dispatched_tensors + cov.excluded_tensors +
                  cov.unaccounted_tensors);

    pn_model_free(model);
}

int main(void)
{
#ifdef PN_COVERAGE_FAKE_KERNEL
    printf("Pantheon qwen35moe Metal coverage gate (FAKE Q5_K kernel linked)\n");
#else
    printf("Pantheon qwen35moe Metal coverage gate\n");
#endif
    printf("======================================\n\n");

    RUN_TEST(registry_tracks_linkage);
    RUN_TEST(registry_q5k_matches_build);
    RUN_TEST(coverage_gate_reports_real_state);
    RUN_TEST(coverage_rows_track_registration);
    RUN_TEST(dropping_a_type_is_caught);

    TEST_REPORT();

    /* Three of the five tests do nothing without a model, and the shared harness
     * prints "ALL TESTS PASSED" either way — a green that means nothing. Say so
     * loudly, so a skipped run is never mistaken for a covered one. */
    if (!model_path())
        printf("\n  *** WARNING: no model found — the 3 tests that actually exercise the\n"
               "      gate DID NOT RUN. This result proves nothing about coverage.\n"
               "      Set PN_QWEN35_MODEL=<path to the qwen35moe .gguf> to run them. ***\n");

    return __ft_failed > 0 ? 1 : 0;
}
