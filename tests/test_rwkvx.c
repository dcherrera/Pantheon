/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_rwkvx.c — Tests for RWKV-X integration into Pantheon
 *
 * Tests safetensors format detection, RWKV-X adapter functions,
 * and (when model is available) full forward pass.
 */

#include "lib/test.h"
#include "tensor/tensor.h"
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/rwkvx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ============================================================
 * Format detection tests
 * ============================================================ */

static void test_is_safetensors_detects_gguf_as_false(void) {
    /* Create a temp file with GGUF magic */
    const char *path = "/tmp/test_pn_gguf.gguf";
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    fwrite("GGUF", 1, 4, f);
    /* Pad to 9 bytes */
    char pad[5] = {0};
    fwrite(pad, 1, 5, f);
    fclose(f);

    ASSERT_TRUE(pn_is_gguf(path));
    ASSERT_TRUE(!pn_is_safetensors(path));
    unlink(path);
}

static void test_is_safetensors_detects_safetensors(void) {
    /* Create a minimal safetensors-like file:
     * 8 bytes LE header length + '{' JSON start */
    const char *path = "/tmp/test_pn_st.safetensors";
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    uint64_t header_len = 42;
    fwrite(&header_len, 8, 1, f);
    fwrite("{\"key\": \"value\"}", 1, 16, f);
    fclose(f);

    ASSERT_TRUE(!pn_is_gguf(path));
    ASSERT_TRUE(pn_is_safetensors(path));
    unlink(path);
}

static void test_is_safetensors_rejects_garbage(void) {
    const char *path = "/tmp/test_pn_garbage.bin";
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    /* Write garbage that doesn't match either format */
    fwrite("JUNK DATA HERE!!", 1, 16, f);
    fclose(f);

    ASSERT_TRUE(!pn_is_gguf(path));
    ASSERT_TRUE(!pn_is_safetensors(path));
    unlink(path);
}

static void test_is_safetensors_nonexistent(void) {
    ASSERT_TRUE(!pn_is_safetensors("/nonexistent/file.safetensors"));
}

/* ============================================================
 * pn_load_ex tests
 * ============================================================ */

static void test_load_ex_null_path(void) {
    pn_model *m = pn_load_ex(NULL, NULL);
    ASSERT_TRUE(m == NULL);
}

static void test_load_ex_nonexistent_safetensors(void) {
    pn_load_opts opts = { .vocab_path = NULL, .quantize = false };
    pn_model *m = pn_load_ex("/nonexistent/model.safetensors", &opts);
    ASSERT_TRUE(m == NULL);
}

/* ============================================================
 * Model-gated integration tests (only run if model file exists)
 * ============================================================ */

/* Check if a model file exists */
static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static const char *MODEL_PATH = "../models/base/RWKV-X-0.2B-64k-Base.safetensors";
static const char *VOCAB_PATH = "../foundry/rwkvx/data/rwkv_vocab_v20230424.txt";

static void test_rwkvx_load_model(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);
    ASSERT_INT_EQ(PN_ARCH_RWKVX, (int)model->config.arch);
    ASSERT_TRUE(model->config.n_embd > 0);
    ASSERT_TRUE(model->config.n_head > 0);
    ASSERT_TRUE(model->config.n_layer > 0);
    ASSERT_TRUE(model->config.n_vocab == 65536);
    ASSERT_TRUE(model->forward != NULL);
    ASSERT_TRUE(model->tokenizer == NULL);  /* RWKV-X uses internal tokenizer */

    pn_model_free(model);
}

static void test_rwkvx_state_lifecycle(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);
    ASSERT_INT_EQ(0, state->pos);
    ASSERT_TRUE(state->arena == NULL);  /* RWKV-X doesn't use arena */
    ASSERT_NOT_NULL(state->kv_cache);   /* RWKV-X state stored here */

    pn_state_reset(state);
    ASSERT_INT_EQ(0, state->pos);

    pn_state_free(state);
    pn_model_free(model);
}

static void test_rwkvx_forward_produces_logits(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    /* Forward a single token (e.g., token 1 = common token) */
    tm_tensor *logits = model->forward(model, 1, state);
    ASSERT_NOT_NULL(logits);
    ASSERT_INT_EQ(1, logits->ndim);
    ASSERT_INT_EQ(65536, (int)logits->shape[0]);

    /* Logits should not be all zeros */
    float *data = (float *)logits->data;
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) sum += data[i] * data[i];
    ASSERT_TRUE(sum > 0.0f);

    tm_free(logits);
    pn_state_free(state);
    pn_model_free(model);
}

static void test_rwkvx_tokenizer_roundtrip(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    /* Encode */
    uint32_t tokens[64];
    int n = pn_rwkvx_encode(model->weights, "Hello world", tokens, 64);
    ASSERT_TRUE(n > 0);

    /* Decode */
    char *decoded = pn_rwkvx_decode(model->weights, tokens, n);
    ASSERT_NOT_NULL(decoded);
    ASSERT_STR_EQ("Hello world", decoded);
    free(decoded);

    /* EOS and vocab size */
    ASSERT_INT_EQ(0, pn_rwkvx_eos_id(model->weights));
    ASSERT_INT_EQ(65536, pn_rwkvx_vocab_size(model->weights));

    pn_model_free(model);
}

static void test_rwkvx_generate(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    pn_sampling_params params = pn_sampling_defaults();
    params.temperature = 0.0f;  /* Greedy for determinism */

    char *output = pn_generate(model, state, "Hello", 8, &params);
    ASSERT_NOT_NULL(output);
    ASSERT_TRUE(strlen(output) > 0);
    printf("    [INFO] generated: \"%s\"\n", output);
    free(output);

    pn_state_free(state);
    pn_model_free(model);
}

static void test_rwkvx_quantized_load(void) {
    if (!file_exists(MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = true };
    pn_model *model = pn_load_ex(MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    /* Forward pass should work with quantized weights */
    tm_tensor *logits = model->forward(model, 1, state);
    ASSERT_NOT_NULL(logits);
    ASSERT_INT_EQ(65536, (int)logits->shape[0]);
    tm_free(logits);

    pn_state_free(state);
    pn_model_free(model);
}

/* ============================================================
 * RWKV-7 pure model tests (model-gated)
 * ============================================================ */

static const char *RWKV7_MODEL_PATH = "../models/base/RWKV-7-0.1B.safetensors";

static void test_rwkv7_load(void) {
    if (!file_exists(RWKV7_MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] RWKV-7 model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(RWKV7_MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);
    ASSERT_INT_EQ(PN_ARCH_RWKVX, (int)model->config.arch);
    ASSERT_TRUE(model->config.n_embd > 0);
    ASSERT_TRUE(model->config.n_head > 0);
    ASSERT_TRUE(model->config.n_layer > 0);
    /* Pure RWKV-7: no attention layers */
    ASSERT_INT_EQ(0, pn_rwkvx_n_attn_layers(model->weights));
    printf("    [INFO] RWKV-7 config: n_embd=%d n_layer=%d vocab=%d\n",
           model->config.n_embd, model->config.n_layer, model->config.n_vocab);

    pn_model_free(model);
}

static void test_rwkv7_forward(void) {
    if (!file_exists(RWKV7_MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] RWKV-7 model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(RWKV7_MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    tm_tensor *logits = model->forward(model, 1, state);
    ASSERT_NOT_NULL(logits);
    ASSERT_INT_EQ(1, logits->ndim);
    ASSERT_TRUE((int)logits->shape[0] > 0);

    /* Logits should not be all zeros */
    float *data = (float *)logits->data;
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) sum += data[i] * data[i];
    ASSERT_TRUE(sum > 0.0f);

    tm_free(logits);
    pn_state_free(state);
    pn_model_free(model);
}

static void test_rwkv7_generate(void) {
    if (!file_exists(RWKV7_MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] RWKV-7 model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = false };
    pn_model *model = pn_load_ex(RWKV7_MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    pn_sampling_params params = pn_sampling_defaults();
    params.temperature = 0.0f;

    char *output = pn_generate(model, state, "Hello", 8, &params);
    ASSERT_NOT_NULL(output);
    ASSERT_TRUE(strlen(output) > 0);
    printf("    [INFO] RWKV-7 generated: \"%s\"\n", output);
    free(output);

    pn_state_free(state);
    pn_model_free(model);
}

static void test_rwkv7_quantized(void) {
    if (!file_exists(RWKV7_MODEL_PATH) || !file_exists(VOCAB_PATH)) {
        printf("    [SKIP] RWKV-7 model or vocab not found\n");
        return;
    }

    pn_load_opts opts = { .vocab_path = VOCAB_PATH, .quantize = true };
    pn_model *model = pn_load_ex(RWKV7_MODEL_PATH, &opts);
    ASSERT_NOT_NULL(model);

    pn_state *state = pn_state_new(model);
    ASSERT_NOT_NULL(state);

    tm_tensor *logits = model->forward(model, 1, state);
    ASSERT_NOT_NULL(logits);
    ASSERT_TRUE((int)logits->shape[0] > 0);
    tm_free(logits);

    pn_state_free(state);
    pn_model_free(model);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("Pantheon RWKV-X Tests\n");
    printf("=====================\n\n");

    /* Format detection (no model needed) */
    RUN_TEST(is_safetensors_detects_gguf_as_false);
    RUN_TEST(is_safetensors_detects_safetensors);
    RUN_TEST(is_safetensors_rejects_garbage);
    RUN_TEST(is_safetensors_nonexistent);

    /* pn_load_ex edge cases */
    RUN_TEST(load_ex_null_path);
    RUN_TEST(load_ex_nonexistent_safetensors);

    /* Model-gated RWKV-X integration tests */
    RUN_TEST(rwkvx_load_model);
    RUN_TEST(rwkvx_state_lifecycle);
    RUN_TEST(rwkvx_forward_produces_logits);
    RUN_TEST(rwkvx_tokenizer_roundtrip);
    RUN_TEST(rwkvx_generate);
    RUN_TEST(rwkvx_quantized_load);

    /* Model-gated RWKV-7 pure model tests */
    RUN_TEST(rwkv7_load);
    RUN_TEST(rwkv7_forward);
    RUN_TEST(rwkv7_generate);
    RUN_TEST(rwkv7_quantized);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
