/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_family_guard.c
 * @brief The qwen35 Tenstorrent accessors must not read a NON-qwen35 model's weights as a
 *        pn_qwen35moe_model.
 *
 * The bug: the prefix cache asks EVERY model `pn_qwen35moe_model_tt_kv_on_card()` before a
 * system-prompt capture. On a TT build serving a dense Qwen3 (llama family, CPU fallback), that
 * cast the model's pn_llama_qweights to pn_qwen35moe_model, read a garbage `gpu` pointer and
 * dereferenced it — SIGSEGV on the first chat that carried a system prompt (Argus CHAT, 1,891
 * tokens). This hands each accessor a llama-arch model whose weights are a buffer of non-NULL
 * garbage: the broken accessor dereferences it and dies, the fixed one answers "not mine".
 *
 * Opens no device. Build with `make USE_TT=1 tests/test_tt_family_guard` and run it anywhere,
 * including the Mac.
 *
 * @status Regression test for the tt-long-prompt-crash fix.
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"

#ifdef USE_TT

/* Larger than any weight struct the accessors could read through, every byte 0x41 — so a
 * pointer read out of it is 0x4141414141414141, non-NULL and unmapped. */
static unsigned char g_llama_weights[1 << 16];

static int g_failures;

static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_failures++;
}

int main(void)
{
    pn_model model;
    uint64_t on_card = 1, on_host = 1, failed = 1;
    uint32_t drafts[4];
    float row[8];

    memset(g_llama_weights, 0x41, sizeof(g_llama_weights));
    memset(&model, 0, sizeof(model));
    model.config.arch = PN_ARCH_QWEN3;
    model.config.n_vocab = 8;
    model.weights = g_llama_weights;

    check(pn_qwen35moe_model_tt_kv_on_card(&model) == 0, "kv_on_card: a llama model is not on card");
    check(pn_qwen35moe_model_tt_loaded(&model) == 0, "tt_loaded: a llama model has no qwen35 residency");
    check(!pn_qwen35moe_tt_spec_can_sample(&model), "spec_can_sample: false for a llama model");
    pn_qwen35moe_tt_matmul_counts(&model, &on_card, &on_host, &failed);
    check(on_card == 0 && on_host == 0 && failed == 0, "matmul_counts: zeros for a llama model");
    check(pn_qwen35moe_tt_dense_row(&model, row, 8) == -1, "dense_row: refused for a llama model");
    check(pn_qwen35moe_tt_mtp_draft(&model, NULL, 0u, 1, drafts) == -1,
          "mtp_draft: refused for a llama model");

    printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("SKIP test_tt_family_guard: needs USE_TT=1\n");
    return 0;
}

#endif
