/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Build sanity test — verifies Pantheon links against Foundry correctly.
 */

#include "lib/test.h"
#include "tensor/tensor.h"
#include "../include/pantheon.h"

/* Test: can create a Foundry tensor */
static void test_foundry_tensor_creation(void) {
    int64_t shape[] = {4, 4};
    tm_tensor *t = tm_zeros(shape, 2, TM_F32);
    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(2, t->ndim);
    ASSERT_INT_EQ(4, (int)t->shape[0]);
    ASSERT_INT_EQ(4, (int)t->shape[1]);
    ASSERT_INT_EQ(16, (int)t->numel);
    tm_free(t);
}

/* Test: can create tensors and do matmul via Foundry */
static void test_foundry_matmul(void) {
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 4};
    tm_tensor *a = tm_ones(shape_a, 2, TM_F32);
    tm_tensor *b = tm_ones(shape_b, 2, TM_F32);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    tm_tensor *c = tm_matmul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, c->ndim);
    ASSERT_INT_EQ(2, (int)c->shape[0]);
    ASSERT_INT_EQ(4, (int)c->shape[1]);

    /* ones(2,3) @ ones(3,4) = 3.0 everywhere */
    float *data = (float *)c->data;
    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(data[i] > 2.99f && data[i] < 3.01f);
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

/* Test: Pantheon sampling defaults are sane */
static void test_sampling_defaults(void) {
    pn_sampling_params p = pn_sampling_defaults();
    ASSERT_TRUE(p.temperature > 0.0f);
    ASSERT_TRUE(p.top_p > 0.0f && p.top_p <= 1.0f);
    ASSERT_TRUE(p.top_k > 0);
}

/* Test: pn_load returns NULL for non-existent file */
static void test_load_nonexistent(void) {
    pn_model *m = pn_load("/nonexistent/model.gguf");
    ASSERT_TRUE(m == NULL);
}

/* Test: pn_state_new returns NULL for NULL model */
static void test_state_null_model(void) {
    pn_state *s = pn_state_new(NULL);
    ASSERT_TRUE(s == NULL);
}

int main(void) {
    printf("Pantheon Build Tests\n");
    printf("====================\n\n");

    RUN_TEST(foundry_tensor_creation);
    RUN_TEST(foundry_matmul);
    RUN_TEST(sampling_defaults);
    RUN_TEST(load_nonexistent);
    RUN_TEST(state_null_model);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
