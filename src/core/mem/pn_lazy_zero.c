/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* pn_lazy_zero — mmap-backed, lazily-committed, zero-filled host buffers.
 * See pn_lazy_zero.h for the design rationale (Phase 10b / INV3). */
#include "pn_lazy_zero.h"

#include <string.h>
#include <sys/mman.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

/* MAP_NORESERVE keeps a mostly-untouched ctx_cap-sized mirror from charging
 * the overcommit accounting on Linux; harmless elsewhere. */
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

void *pn_lazy_zero_alloc(size_t bytes)
{
    if (bytes == 0)
        return NULL;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

void pn_lazy_zero_free(void *ptr, size_t bytes)
{
    if (ptr && bytes)
        munmap(ptr, bytes);
}

void pn_lazy_zero_reset(void *ptr, size_t bytes)
{
    if (!ptr || !bytes)
        return;
#ifdef __linux__
    /* On private anonymous mappings MADV_DONTNEED is not advisory on Linux:
     * it drops the pages and guarantees zero-fill on the next access. */
    if (madvise(ptr, bytes, MADV_DONTNEED) == 0)
        return;
#else
    /* Portable equivalent: atomically replace the region with a fresh
     * zero-fill mapping at the same address. */
    if (mmap(ptr, bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != MAP_FAILED)
        return;
#endif
    memset(ptr, 0, bytes); /* last-resort correctness fallback */
}
