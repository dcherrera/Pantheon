/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_host_signals.h
 * @brief Cheap host memory-pressure readers (MemoryAware.md §2.1). Reads
 *        /proc/meminfo MemAvailable/MemTotal, /proc/pressure/memory PSI, and
 *        cgroup-v2 memory.{current,high,max} on Linux; on other platforms fills
 *        the portable total/available figure and reports the Linux-only signals
 *        as unobserved (0). Split out from pn_memwatch so the OS-specific /proc
 *        parsing is isolated and independently testable.
 *
 * @status Phase 1 — read-only. Mac-testable (portable total/avail path).
 * @issues None
 * @todo None
 */

#ifndef PN_HOST_SIGNALS_H
#define PN_HOST_SIGNALS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One sample of host memory truth. Zeroed fields mean "not observed" — on
 * macOS the PSI/cgroup fields are always 0 (no such interface). */
typedef struct {
    uint64_t total_bytes;       /* physical RAM total                          */
    uint64_t avail_bytes;       /* allocatable-without-swap estimate           */
    double   psi_some_avg10;    /* memory PSI "some avg10" %, 0 if unobserved   */
    double   psi_full_avg10;    /* memory PSI "full avg10" %, 0 if unobserved   */
    uint64_t cgroup_current;    /* cgroup-v2 memory.current, 0 if not in cgroup */
    uint64_t cgroup_high;       /* cgroup-v2 memory.high (0 == "max"/unset)     */
    uint64_t cgroup_max;        /* cgroup-v2 memory.max  (0 == "max"/unset)     */
} pn_host_sample;

/* Take one full host sample (all sources). Never fails destructively: sources
 * that can't be read leave their field 0. Cheap (a few small reads, ~1 ms).
 *
 * Thread-safe: yes (no shared state)
 * Reentrant: yes
 * Signal-safe: no (uses stdio/open) */
void pn_host_signals_sample(pn_host_sample *out);

/* --- Individually testable readers (exposed for unit tests) --- */

/* Parse total + available from /proc/meminfo. Returns true on success (Linux),
 * false if /proc/meminfo is absent (then the caller uses the portable path). */
bool pn_read_proc_meminfo(uint64_t *total_bytes, uint64_t *avail_bytes);

/* Portable RAM total + available fallback for non-Linux (macOS sysctl +
 * host_statistics64). Always fills something usable. */
void pn_read_portable_memory(uint64_t *total_bytes, uint64_t *avail_bytes);

/* Parse "some"/"full" avg10 from /proc/pressure/memory. Returns true on Linux
 * with PSI enabled; leaves outputs 0 and returns false otherwise. */
bool pn_read_proc_pressure(double *some_avg10, double *full_avg10);

/* Read this process's cgroup-v2 memory.{current,high,max}. Returns true if the
 * process is under a cgroup-v2 memory controller; 0-fills and returns false
 * otherwise. A "max" literal maps to 0 (meaning "unset"). */
bool pn_read_cgroup_v2_memory(uint64_t *current, uint64_t *high, uint64_t *max);

#ifdef __cplusplus
}
#endif

#endif /* PN_HOST_SIGNALS_H */
