/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_context_budget.c — the per-session context cap on the CPU path, and the
 * number the 422 `context_over_budget` quotes.
 *
 *   H2d: a CPU-resident model admitted a prompt over its --ctx-cap whenever host
 *        memory held it (the Mac: 5,000 tokens at cap 4096 were answered). Both
 *        pn_admit_request's CPU path and pn_admit with no memory watcher must CAP
 *        at the window.
 *   H2c: the 422 quoted the advertised window (21139) instead of the cap that
 *        refused the prompt (1581, host memory on a CPU-fallback serve).
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/core/mem/pn_admission.h"
#include "../src/core/mem/pn_memwatch.h"
#include "../src/serve/context_over_budget.h"

#include <stdio.h>
#include <stdlib.h>

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

/* H2d, the engine-level decision: ample host RAM, CPU path, a prompt past the window. */
static void test_cpu_path_caps_at_window(void)
{
    pn_mem_state st = {0};
    pn_admit_result r;

    printf("cpu_path_caps_at_window:\n");
    unsetenv("PN_HOST_RESERVE_BYTES");
    unsetenv("PN_PSI_FULL_REJECT");
    st.host_total_bytes = 64ULL << 30;
    st.host_avail_bytes = 48ULL << 30;
    pn_admit_request(5000, true, 0, false, -1, 4096, &st, &r);
    CHECK(r.verdict == PN_ADMIT_CAP, "cpu path: 5000 tokens at window 4096 is capped");
    CHECK(r.fit_tokens == 4096, "cpu path: the cap is the window");
    pn_admit_request(4000, true, 0, false, -1, 4096, &st, &r);
    CHECK(r.verdict == PN_ADMIT_OK, "cpu path: a prompt inside the window is admitted");
}

/* A dense Qwen3 the size of the Mac test model, --ctx-cap 4096, CPU forward. */
static pn_model *fake_dense_model(void)
{
    pn_model *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->config.arch = PN_ARCH_QWEN3;
    m->config.n_ctx = 40960;
    m->config.ctx_cap = 4096;
    m->config.n_layer = 28;
    m->config.n_embd = 1024;
    m->config.n_head = 16;
    m->config.n_kv_head = 8;
    m->config.head_dim = 128;
    return m;
}

/* H2d, through pn_admit (what sg_run_completion calls), with and without the watcher. */
static void test_admit_caps_dense_cpu_model(void)
{
    pn_model *m = fake_dense_model();
    int fit = 0;

    printf("admit_caps_dense_cpu_model:\n");
    CHECK(m != NULL, "model");
    if (!m)
        return;
    CHECK(pn_session_ctx_cap(m) == 4096, "the configured window is the ctx cap");
    CHECK(pn_admit(m, 5000, &fit) == PN_ADMIT_CAPPED && fit == 4096,
          "no memory watcher: 5000 tokens at cap 4096 is capped to 4096");
    CHECK(pn_admit(m, 3000, &fit) == PN_ADMIT_FIT, "no memory watcher: 3000 fits");
    if (pn_memwatch_start() == 0) {
        fit = 0;
        CHECK(pn_admit(m, 5000, &fit) == PN_ADMIT_CAPPED && fit == 4096,
              "memory watcher on, host holds it: 5000 tokens is still capped to 4096");
        pn_memwatch_stop();
    }
    free(m);
}

/* H2c: the 422's kv_cap is the cap that refused the prompt. */
static void test_over_budget_quotes_refusing_cap(void)
{
    int kv_cap = 21139;   /* what the client was told before: the advertised window */

    printf("over_budget_quotes_refusing_cap:\n");
    CHECK(sg_context_over_budget(PN_ADMIT_CAPPED, 5000, 1581, &kv_cap),
          "a capped prompt is over budget");
    CHECK(kv_cap == 1581, "the 422 quotes 1581, the cap that refused it, not 21139");
    kv_cap = 7;
    CHECK(!sg_context_over_budget(PN_ADMIT_FIT, 5000, 5000, &kv_cap) && kv_cap == 7,
          "an admitted prompt is not over budget and leaves kv_cap alone");
    CHECK(!sg_context_over_budget(PN_ADMIT_CAPPED, 5000, 5000, &kv_cap),
          "a cap at the prompt length is not over budget");
    CHECK(!sg_context_over_budget(PN_ADMIT_CAPPED, 5000, 0, &kv_cap),
          "a zero window is a denial, not over budget");
}

int main(void)
{
    printf("test_context_budget\n");
    test_cpu_path_caps_at_window();
    test_admit_caps_dense_cpu_model();
    test_over_budget_quotes_refusing_cap();
    printf("test_context_budget: %d checks, %d failures\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
