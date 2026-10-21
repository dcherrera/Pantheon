/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_mem_state.h
 * @brief Published memory-pressure truth for the MemoryAware layer. A single
 *        lock-free snapshot (host RAM + PSI + per-die VRAM + a derived pressure
 *        level) written by pn_memwatch and read by admission / the degradation
 *        ladder. Transient runtime state only — never persisted (no-auto-save).
 *
 * @status Phase 1 — accounting + read-only signals. Publishes truth; no
 *         behavior change. Host signals are Mac/Metal-portable (Linux-only
 *         fields read 0 off-Linux); per-die VRAM is fed in from the GPU side.
 * @issues None
 * @todo None
 */

#ifndef PN_MEM_STATE_H
#define PN_MEM_STATE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hard ceiling on tracked dies (the K80 test machine has 16 GK210 dies). The state carries a
 * fixed array so it stays a plain value type with no allocation. */
#define PN_MEM_MAX_DIES 16

/* Derived pressure level. Ordered: each higher level engages the next rung of
 * the degradation ladder (MemoryAware.md §7). Phase 1 only *computes* and logs
 * it; later phases react to it. */
typedef enum {
    PN_MEM_LEVEL_OK       = 0,  /* ample headroom — normal operation */
    PN_MEM_LEVEL_WATCH    = 1,  /* headroom shrinking — pre-emptive caution */
    PN_MEM_LEVEL_TIGHT    = 2,  /* reserve nearly reached / PSI rising */
    PN_MEM_LEVEL_CRITICAL = 3,  /* at the reserve line — refuse/shed */
} pn_mem_level;

/* One immutable snapshot of memory truth. Copied out by readers under a seqlock
 * (pn_memwatch_read), so readers always see a self-consistent set of numbers.
 *
 * Host fields come from /proc on Linux; on macOS the sysctl-backed available
 * figure is filled and the Linux-only PSI/cgroup fields read 0 (meaning "not
 * observed", treated as no-pressure). Per-die fields are populated from the GPU
 * side (pn_memwatch_set_die) and are 0 until a die reports in. */
typedef struct {
    /* --- Host RAM (system-wide) --- */
    uint64_t host_total_bytes;      /* MemTotal (or sysctl hw.memsize on macOS) */
    uint64_t host_avail_bytes;      /* MemAvailable — allocatable without swap  */

    /* --- Host memory stall (PSI, Linux only; 0 elsewhere) --- */
    double   host_psi_some_avg10;   /* /proc/pressure/memory "some avg10" (%)   */
    double   host_psi_full_avg10;   /* "full avg10" (%) — whole-box thrash      */

    /* --- Our cgroup slice (Linux cgroup-v2 only; 0 if not under a cgroup) --- */
    uint64_t cgroup_current_bytes;  /* memory.current                            */
    uint64_t cgroup_high_bytes;     /* memory.high  (0 == "max"/unset)           */
    uint64_t cgroup_max_bytes;      /* memory.max   (0 == "max"/unset)           */

    /* --- Per-die VRAM (fed in from the GPU side) --- */
    int      n_dies;                /* number of dies with a reported number     */
    uint64_t per_die_free[PN_MEM_MAX_DIES];    /* cuMemGetInfo free, bytes       */
    uint64_t per_die_budget[PN_MEM_MAX_DIES];  /* usable budget (total - reserve)*/
    uint64_t per_die_used[PN_MEM_MAX_DIES];    /* our own alloc tally, bytes     */

    /* --- Derived --- */
    pn_mem_level level;             /* worst-of host + any die                   */
    uint64_t     seq;               /* snapshot sequence number (monotonic)      */
} pn_mem_state;

#ifdef __cplusplus
}
#endif

#endif /* PN_MEM_STATE_H */
