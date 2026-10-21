/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "arena.h"
#include <stdlib.h>
#include <string.h>

#define ARENA_ALIGN 16

pn_arena* pn_arena_new(size_t capacity) {
    pn_arena *a = malloc(sizeof(pn_arena));
    if (!a) return NULL;

    /* Allocate aligned backing buffer */
    a->base = NULL;
    if (posix_memalign((void **)&a->base, ARENA_ALIGN, capacity) != 0) {
        free(a);
        return NULL;
    }

    a->capacity = capacity;
    a->used = 0;
    return a;
}

void pn_arena_free(pn_arena *arena) {
    if (!arena) return;
    free(arena->base);
    free(arena);
}

void* pn_arena_alloc(pn_arena *arena, size_t n) {
    if (!arena || n == 0) return NULL;

    /* Align up to 16 bytes */
    size_t aligned = (arena->used + (ARENA_ALIGN - 1)) & ~(size_t)(ARENA_ALIGN - 1);
    if (aligned + n > arena->capacity) return NULL;

    void *ptr = arena->base + aligned;
    arena->used = aligned + n;
    return ptr;
}

float* pn_arena_alloc_f32(pn_arena *arena, int n) {
    return (float *)pn_arena_alloc(arena, (size_t)n * sizeof(float));
}

void pn_arena_reset(pn_arena *arena) {
    if (arena) arena->used = 0;
}

size_t pn_arena_used(const pn_arena *arena) {
    return arena ? arena->used : 0;
}
