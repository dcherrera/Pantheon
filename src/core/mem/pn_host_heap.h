/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* pn_host_heap — keep per-request host memory returnable to the OS.
 *
 * Why this exists (B5): a long-running serve on glibc lost ~4 GB of host RAM
 * in a day. Every request runs on its own connection thread, so glibc spreads
 * the per-request churn (tool-grammar token tables, sys-prefix snapshots,
 * prompt/JSON buffers) over up to 8 x cores arenas. Freed memory inside a
 * non-main arena goes back to the OS only from the very top of a heap, and
 * the sliding mmap threshold moves multi-MiB buffers into those arenas after
 * the first large free. The live process held 56 arena heaps of 64 MiB, all
 * free-but-retained (mostly swapped out), until a restart.
 *
 *   pn_host_heap_init()    once at startup: pins the mmap threshold (buffers
 *                          >= 256 KiB are their own mapping, unmapped on free).
 *   pn_host_heap_release() after a request ends: malloc_trim(0), which
 *                          returns the free pages inside every arena.
 *
 * Capping the arena count (M_ARENA_MAX) was measured too and does not help.
 * No-ops off glibc. PANTHEON_HEAP_TUNE=0 turns both off (A/B escape hatch).
 * Thread-safe: yes. */
#ifndef PN_HOST_HEAP_H
#define PN_HOST_HEAP_H

#include <stdbool.h>

/* Apply the allocator settings. Call once, before worker threads start.
 * Returns true when the settings were applied (glibc, not disabled). */
bool pn_host_heap_init(void);

/* Return free arena pages to the OS. Cheap enough to call after every request
 * (one walk of the free lists); concurrent callers collapse into one trim. */
void pn_host_heap_release(void);

#endif /* PN_HOST_HEAP_H */
