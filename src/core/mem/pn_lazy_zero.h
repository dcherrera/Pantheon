/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* pn_lazy_zero — mmap-backed, lazily-committed, zero-filled host buffers.
 *
 * Why this exists (Phase 10b / INV3): the per-session host KV mirrors and SSM
 * states were calloc'd at full ctx_cap size and then fully COMMITTED by the
 * first pn_state_reset (a full-buffer memset) — ~789 MiB/slot for saber-orch
 * at ctx_cap 131072, ~245 MiB/slot for nex-n2-mini at 32000 — even for a
 * session holding 200 tokens. glibc makes it worse: after a large free the
 * sliding mmap threshold moves these buffers into 64 MiB arenas, where calloc
 * memsets them (full commit at allocation) and free() never returns the pages
 * to the OS.
 *
 * These helpers replace calloc/free/memset for such buffers with anonymous
 * mmap / munmap / MADV_DONTNEED:
 *   - alloc: pages are committed only when WRITTEN, so RSS tracks the bytes a
 *     session actually uses, not its ctx_cap ceiling.
 *   - free: munmap returns the pages to the OS immediately (no arena
 *     retention).
 *   - reset: logically zero-fills the whole buffer AND returns its committed
 *     pages, in O(pages present): on Linux, MADV_DONTNEED on private
 *     anonymous memory guarantees the next read sees zeros; elsewhere the
 *     region is atomically replaced with a fresh zero mapping (MAP_FIXED).
 *     The pointer is unchanged in both cases.
 *
 * Use for large (>= ~1 MiB), zero-primed, size-fixed buffers whose committed
 * footprint should track usage. Not a general allocator: the caller must
 * remember the allocation size and pass it to free/reset.
 *
 * Thread-safe: yes (no shared state).
 */
#ifndef PN_LAZY_ZERO_H
#define PN_LAZY_ZERO_H

#include <stddef.h>

/* Allocate `bytes` of zero-filled, lazily-committed memory (page-granular).
 * Returns NULL on failure or if bytes == 0. */
void *pn_lazy_zero_alloc(size_t bytes);

/* Release a pn_lazy_zero_alloc'd buffer. `bytes` must be the allocation size.
 * NULL is a no-op. */
void pn_lazy_zero_free(void *ptr, size_t bytes);

/* Zero the buffer's contents and return its committed pages to the OS.
 * Equivalent to memset(ptr, 0, bytes) + "give the RAM back". The pointer
 * stays valid and unchanged. Falls back to memset if the kernel call fails.
 * NULL is a no-op. */
void pn_lazy_zero_reset(void *ptr, size_t bytes);

#endif /* PN_LAZY_ZERO_H */
