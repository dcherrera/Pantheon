/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_ARENA_H
#define PN_ARENA_H

#include <stddef.h>

/*
 * Bump allocator for per-token scratch memory.
 * O(1) alloc, O(1) reset. 16-byte aligned.
 */

typedef struct pn_arena {
    char  *base;
    size_t capacity;
    size_t used;
} pn_arena;

/* Create arena with given capacity (bytes). Returns NULL on failure. */
pn_arena* pn_arena_new(size_t capacity);

/* Free arena and its backing buffer. */
void pn_arena_free(pn_arena *arena);

/* Allocate n bytes from arena (16-byte aligned). Returns NULL if full. */
void* pn_arena_alloc(pn_arena *arena, size_t n);

/* Allocate n floats from arena. Returns NULL if full. */
float* pn_arena_alloc_f32(pn_arena *arena, int n);

/* Reset arena to empty (O(1), no deallocation). */
void pn_arena_reset(pn_arena *arena);

/* Current bytes used. */
size_t pn_arena_used(const pn_arena *arena);

#endif /* PN_ARENA_H */
