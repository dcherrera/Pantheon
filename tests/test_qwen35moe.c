/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_qwen35moe.c — Tests for the qwen35moe hybrid architecture.
 *
 * Locks in the A8 oracle validation as a regression guard: the hybrid
 * (30 Gated-Delta SSM + 10 full-attn + shared-expert MoE) forward must
 * load, produce finite logits, and predict the oracle-validated argmax.
 *
 * The forward tests load the real ~16 GB Q3_K_M GGUF, so they are gated
 * behind the PN_QWEN35_MODEL env var (path to the GGUF) — `make test`
 * never loads a model unless you opt in:
 *
 *   PN_QWEN35_MODEL=../models/nex-agi_Nex-N2-mini-Q3_K_M.gguf make test
 *
 * Run it through run_model_safe.sh on memory-constrained machines.
 */

#include "lib/test.h"
#include "tensor/tensor.h"
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Verified config, read from the GGUF metadata. */
#define Q35_N_LAYER 40
#define Q35_N_EMBD  2048
#define Q35_N_VOCAB 248320
/* A8 oracle result: token "Hello" (9419) at pos 0 → argmax 11. */
#define Q35_HELLO_TOKEN 9419
#define Q35_HELLO_ARGMAX 11

/* ============================================================
 * Architecture registry (no model needed)
 * ============================================================ */

static void test_qwen35moe_forward_fn_registered(void) {
    /* The arch must resolve to a forward fn, else dispatch silently no-ops. */
    ASSERT_NOT_NULL((void *)pn_get_forward_fn(PN_ARCH_QWEN35MOE));
}

/* ============================================================
 * Model-gated integration tests (only run if PN_QWEN35_MODEL is set)
 * ============================================================ */

static const char *model_path(void) {
    const char *p = getenv("PN_QWEN35_MODEL");
    if (p && *p && access(p, F_OK) == 0) return p;
    return NULL;
}

static void test_qwen35moe_load_model(void) {
    const char *path = model_path();
    if (!path) { printf("    [SKIP] set PN_QWEN35_MODEL to the GGUF to run\n"); return; }

    pn_model *model = pn_load(path);
    ASSERT_NOT_NULL(model);
    ASSERT_INT_EQ(PN_ARCH_QWEN35MOE, (int)model->config.arch);
    ASSERT_INT_EQ(Q35_N_LAYER, model->config.n_layer);
    ASSERT_INT_EQ(Q35_N_EMBD,  model->config.n_embd);
    ASSERT_INT_EQ(Q35_N_VOCAB, model->config.n_vocab);
    ASSERT_TRUE(model->forward != NULL);
    ASSERT_TRUE(model->tokenizer != NULL);   /* BPE + ChatML */

    pn_model_free(model);
}

static void test_qwen35moe_state_lifecycle(void) {
    const char *path = model_path();
    if (!path) { printf("    [SKIP] set PN_QWEN35_MODEL to the GGUF to run\n"); return; }

    pn_model *model = pn_load(path);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);
    ASSERT_INT_EQ(0, state->pos);

    pn_state_reset(state);
    ASSERT_INT_EQ(0, state->pos);

    pn_state_free(state);
    pn_model_free(model);
}

static void test_qwen35moe_forward_matches_oracle(void) {
    const char *path = model_path();
    if (!path) { printf("    [SKIP] set PN_QWEN35_MODEL to the GGUF to run\n"); return; }

    pn_model *model = pn_load(path);
    ASSERT_NOT_NULL(model);
    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    tm_tensor *out = model->forward(model, Q35_HELLO_TOKEN, state);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(Q35_N_VOCAB, (int)out->shape[0]);

    float *logits = tm_data_f32(out);
    ASSERT_NOT_NULL(logits);

    /* Logits finite and argmax matches the llama.cpp oracle (A8). */
    int best = 0;
    int finite = 1;
    for (int i = 0; i < Q35_N_VOCAB; i++) {
        if (!isfinite(logits[i])) { finite = 0; break; }
        if (logits[i] > logits[best]) best = i;
    }
    ASSERT_TRUE(finite);
    ASSERT_INT_EQ(Q35_HELLO_ARGMAX, best);

    tm_free(out);
    pn_state_free(state);
    pn_model_free(model);
}

int main(void) {
    printf("Pantheon qwen35moe Tests\n");
    printf("========================\n\n");

    /* Registry (no model needed) */
    RUN_TEST(qwen35moe_forward_fn_registered);

    /* Model-gated integration (PN_QWEN35_MODEL) */
    RUN_TEST(qwen35moe_load_model);
    RUN_TEST(qwen35moe_state_lifecycle);
    RUN_TEST(qwen35moe_forward_matches_oracle);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
