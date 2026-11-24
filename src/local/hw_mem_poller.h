/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hw_mem_poller.h
 * @brief  Device memory, read in the background (fast /local/hardware). One
 *         thread reads every unit's device memory every PN_HW_POLL_INTERVAL_MS
 *         (and at once after a load or unload: pn_hw_mem_poller_kick) and
 *         keeps the latest reading per unit with its time; requests answer from
 *         that snapshot instantly and report its age. The fit / load paths ask
 *         for a fresh-enough one (pn_hw_mem_poller_ensure_fresh).
 *         The thread holds nothing on a device between reads: each read is
 *         whatever the reader does (NVML, a transient context, the TT
 *         allocator), and it never runs while no one started it.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_HW_MEM_POLLER_H
#define PN_LOCAL_HW_MEM_POLLER_H

#include <stdint.h>

#include "hw_report.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Between two background reads of every unit. */
#define PN_HW_POLL_INTERVAL_MS 3000
/* A pass never starts sooner than this many times its own length after the
 * last one began, so a slow reader (transient contexts: ~285 ms a K80 die)
 * stays a small share of the 2-core host. */
#define PN_HW_POLL_DUTY_DIVISOR 4
/* The fit / load paths use a reading at most this old, taken after the last
 * load or unload. */
#define PN_HW_FIT_MAX_AGE_MS 10000
/* How long they wait for one before using what there is. */
#define PN_HW_FIT_WAIT_MS 15000

/**
 * Start the background reads of dies 0..n_dies-1 of `family` through `reader`.
 * Returns once the first pass is done, so the first request has a reading.
 * @return 0, or -1 (already running, bad args, no thread)
 *
 * Thread-safe: no (start once). Reentrant: no. Signal-safe: no
 */
int pn_hw_mem_poller_start(pn_hw_family family, int n_dies, pn_hw_mem_reader reader,
                           int interval_ms);

/**
 * Stop and join the thread (no-op when not running).
 *
 * Thread-safe: no. Reentrant: no. Signal-safe: no
 */
void pn_hw_mem_poller_stop(void);

/** Whether the poller is running. Thread-safe: yes. */
int pn_hw_mem_poller_running(void);

/**
 * The latest reading of `die` and its age in ms.
 * @return 0, or -1 when there is none (not running, the die could not be read)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hw_mem_poller_get(int die, pn_hw_dev_mem *out, uint64_t *age_ms);

/**
 * Ask for a pass now (after a load or unload). Returns at once.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_hw_mem_poller_kick(void);

/**
 * Make sure the snapshot is at most max_age_ms old and was read after the last
 * kick; otherwise kick and wait (at most timeout_ms) for that pass.
 * @return 0 when fresh, -1 on timeout (the snapshot is then whatever is there)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hw_mem_poller_ensure_fresh(int max_age_ms, int timeout_ms);

/**
 * Passes done since start (tests).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
uint64_t pn_hw_mem_poller_passes(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HW_MEM_POLLER_H */
