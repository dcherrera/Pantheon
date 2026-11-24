/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* pn_host_heap — see pn_host_heap.h (B5). */
#include "pn_host_heap.h"

#include <stdlib.h>
#include <string.h>

#if defined(__GLIBC__)
#include <malloc.h>
#include <stdatomic.h>
#define PN_HEAP_GLIBC 1
#endif

/* >= this many bytes: a private mapping, unmapped on free. Fixing it (any
 * explicit M_MMAP_THRESHOLD) also stops glibc's sliding threshold, which would
 * otherwise climb to 32 MiB after the first large free. */
#define PN_HEAP_MMAP_THRESHOLD (2 * 1024 * 1024)

static bool heap_tune_enabled(void)
{
    const char *e = getenv("PANTHEON_HEAP_TUNE");
    return !(e && strcmp(e, "0") == 0);
}

#ifdef PN_HEAP_GLIBC
static atomic_int g_enabled = 0;
static atomic_flag g_trimming = ATOMIC_FLAG_INIT;
#endif

bool pn_host_heap_init(void)
{
#ifdef PN_HEAP_GLIBC
    if (!heap_tune_enabled())
        return false;
    mallopt(M_MMAP_THRESHOLD, PN_HEAP_MMAP_THRESHOLD);
    atomic_store(&g_enabled, 1);
    return true;
#else
    (void)heap_tune_enabled;
    return false;
#endif
}

void pn_host_heap_release(void)
{
#ifdef PN_HEAP_GLIBC
    if (!atomic_load(&g_enabled))
        return;
    /* One trim at a time; a caller arriving mid-trim is covered by it. */
    if (atomic_flag_test_and_set(&g_trimming))
        return;
    malloc_trim(0);
    atomic_flag_clear(&g_trimming);
#endif
}
