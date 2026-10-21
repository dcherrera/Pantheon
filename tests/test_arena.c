/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Arena allocator tests.
 */

#include "lib/test.h"
#include "../src/memory/arena.h"
#include <stdint.h>
#include <string.h>

/* Test basic allocation */
static void test_arena_alloc(void) {
    pn_arena *a = pn_arena_new(4096);
    ASSERT_NOT_NULL(a);

    float *p1 = pn_arena_alloc_f32(a, 64);
    ASSERT_NOT_NULL(p1);

    float *p2 = pn_arena_alloc_f32(a, 128);
    ASSERT_NOT_NULL(p2);

    /* Pointers should not overlap */
    ASSERT_TRUE(p2 >= p1 + 64);

    /* Should be able to write */
    for (int i = 0; i < 64; i++) p1[i] = (float)i;
    for (int i = 0; i < 128; i++) p2[i] = (float)(i + 100);

    ASSERT_FLT_NEAR(0.0f, p1[0], 1e-6f);
    ASSERT_FLT_NEAR(63.0f, p1[63], 1e-6f);
    ASSERT_FLT_NEAR(100.0f, p2[0], 1e-6f);

    pn_arena_free(a);
}

/* Test reset */
static void test_arena_reset(void) {
    pn_arena *a = pn_arena_new(1024);
    ASSERT_NOT_NULL(a);

    float *p1 = pn_arena_alloc_f32(a, 64);
    ASSERT_NOT_NULL(p1);
    ASSERT_TRUE(pn_arena_used(a) > 0);

    pn_arena_reset(a);
    ASSERT_INT_EQ(0, (int)pn_arena_used(a));

    /* Allocate again after reset — should get same base pointer */
    float *p2 = pn_arena_alloc_f32(a, 64);
    ASSERT_NOT_NULL(p2);
    ASSERT_TRUE(p1 == p2);

    pn_arena_free(a);
}

/* Test overflow returns NULL */
static void test_arena_overflow(void) {
    pn_arena *a = pn_arena_new(256);
    ASSERT_NOT_NULL(a);

    /* 256 bytes / 4 bytes per float = 64 floats max */
    float *p1 = pn_arena_alloc_f32(a, 60);
    ASSERT_NOT_NULL(p1);

    /* This should fail — not enough room */
    float *p2 = pn_arena_alloc_f32(a, 60);
    ASSERT_NULL(p2);

    pn_arena_free(a);
}

/* Test 16-byte alignment */
static void test_arena_alignment(void) {
    pn_arena *a = pn_arena_new(4096);
    ASSERT_NOT_NULL(a);

    /* Allocate odd-sized chunks */
    void *p1 = pn_arena_alloc(a, 7);
    ASSERT_NOT_NULL(p1);
    ASSERT_TRUE(((uintptr_t)p1 & 15) == 0);

    void *p2 = pn_arena_alloc(a, 13);
    ASSERT_NOT_NULL(p2);
    ASSERT_TRUE(((uintptr_t)p2 & 15) == 0);

    void *p3 = pn_arena_alloc(a, 1);
    ASSERT_NOT_NULL(p3);
    ASSERT_TRUE(((uintptr_t)p3 & 15) == 0);

    pn_arena_free(a);
}

int main(void) {
    printf("Arena Allocator Tests\n");
    printf("=====================\n\n");

    RUN_TEST(arena_alloc);
    RUN_TEST(arena_reset);
    RUN_TEST(arena_overflow);
    RUN_TEST(arena_alignment);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
