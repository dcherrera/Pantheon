/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_memwatch.h
 * @brief The MemoryAware poll loop + accounting hub (MemoryAware.md §2, §3).
 *        One low-priority thread reads the cheap host signals (/proc/meminfo
 *        MemAvailable, /proc/pressure/memory PSI, cgroup-v2 memory.*) every
 *        ~250 ms and publishes a pn_mem_state snapshot. The GPU side feeds in
 *        per-die VRAM numbers; admission and the degradation ladder read the
 *        published truth. Unified in src/core/mem/ (§11 Q3) so host pressure is
 *        tracked on Mac/Metal too, not just a K80 host.
 *
 *        Library-first: the reserve config and the die numbers are passed in,
 *        not hard-coded here. Boot-measured per-die budget (§11 Q6) is applied
 *        by the caller before feeding pn_memwatch_set_die.
 *
 * @status Phase 1 — publishes truth, logs on level change, no behavior change.
 * @issues None
 * @todo None
 */

#ifndef PN_MEMWATCH_H
#define PN_MEMWATCH_H

#include <stdbool.h>
#include <stdint.h>

#include "pn_mem_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Host-headroom reserve (MemoryAware.md §6.3). Admission treats this many bytes
 * of host RAM as off-limits so the kernel, sshd, the WG tunnel and the serve
 * loop always have room. Default policy: max(PN_HOST_RESERVE_MIN_BYTES,
 * PN_HOST_RESERVE_FRACTION * MemTotal). Env override PN_HOST_RESERVE_BYTES. */
#define PN_HOST_RESERVE_MIN_BYTES   (1024ULL * 1024 * 1024)  /* 1 GiB          */
#define PN_HOST_RESERVE_FRACTION    0.15                     /* 15% of MemTotal */

/* Per-die VRAM reserve (MemoryAware.md §6.3): driver overhead + fragmentation
 * slack held back from each die's measured total. Env override
 * PN_DIE_RESERVE_BYTES. The budget the GPU side reports is total - reserve. */
#define PN_DIE_RESERVE_BYTES        (1024ULL * 1024 * 1024)  /* ~1 GiB          */

/* Fragmentation slack the qwen35moe CUDA model-load gate holds back on top of a
 * projection that already counts the die's runtime scratch
 * (qwen35moe_die_projection.h). Holding the whole PN_DIE_RESERVE_BYTES there
 * charged that scratch twice: a K80 die of the shipped 65,536-token q8 profile
 * really serves with ~660 MiB free. Env override PN_DIE_LOAD_MARGIN_BYTES. */
#define PN_DIE_LOAD_MARGIN_BYTES    (256ULL * 1024 * 1024)   /* 256 MiB         */

/* Poll cadence for the host-signal loop. */
#define PN_MEMWATCH_POLL_MS         250

/* PSI-full REJECT threshold (percent of wall time the WHOLE box was stalled on
 * memory, avg10). Below this, PSI-full is treated as a transient blip — e.g.
 * the page-cache churn right after a ~90 s model load leaves avg10 slightly
 * above 0 for a full decay minute — and requests are still admitted on the
 * byte math. At or above it the box is genuinely thrashing and new work is
 * rejected. Env override PN_PSI_FULL_REJECT (percent, float). */
#define PN_PSI_FULL_REJECT_DEFAULT  15.0

/* --- PSI escalation thresholds (Phase 10) -----------------------------------
 *
 * PSI avg10 is a *decaying 10-second average*, so it reads slightly above zero
 * for ~a minute after any brief reclaim — a model load, page-cache churn, a
 * log rotation. Treating "> 0" as pressure is what tripped the ladder to TIGHT
 * on a healthy K80 host (log 2026-07-27: `OK -> TIGHT | host avail 5578 MiB /
 * 7845 MiB psi(some 0.2 full 0.2)`, then `TIGHT -> OK | host avail 5071 MiB …
 * psi(some 0.0 full 0.0)` — it relaxed at a LOWER avail, so PSI, not bytes, was
 * the trigger). full avg10 = 0.2 means the whole box stalled for 20 ms out of
 * the last 10 s. That is noise, not pressure.
 *
 * So PSI now escalates PROPORTIONATELY, with a noise floor, instead of snapping
 * to TIGHT on any non-zero reading. All four are percentages, env-overridable.
 * The REJECT threshold above is unchanged and still maps to CRITICAL. */
#define PN_PSI_FULL_TIGHT_DEFAULT    5.0   /* PN_PSI_FULL_TIGHT  -> TIGHT      */
#define PN_PSI_FULL_WATCH_DEFAULT    1.0   /* PN_PSI_FULL_WATCH  -> WATCH      */
#define PN_PSI_SOME_TIGHT_DEFAULT   10.0   /* PN_PSI_SOME_TIGHT  -> TIGHT      */
#define PN_PSI_SOME_WATCH_DEFAULT    5.0   /* PN_PSI_SOME_WATCH  -> WATCH      */

/* Consecutive polls a sub-CRITICAL PSI escalation must HOLD before it is
 * allowed to raise the published level (~1 s at a 250 ms cadence). PSI is a
 * leading indicator sampled every 250 ms; a single elevated sample must not be
 * able to engage a degradation rung. De-escalation is immediate (relax fast) and
 * a CRITICAL PSI reading bypasses the dwell entirely — real thrash is never
 * delayed. Hard signals (byte headroom, per-die free VRAM) are never delayed:
 * they are facts, not leading indicators. Env override PN_PSI_TRIP_POLLS. */
#define PN_PSI_TRIP_POLLS_DEFAULT   4

/* Compute the effective host reserve in bytes for a given MemTotal, honoring
 * the PN_HOST_RESERVE_BYTES env override. Pure — safe to call anywhere. */
uint64_t pn_host_reserve_bytes(uint64_t host_total_bytes);

/* Compute the effective per-die reserve in bytes, honoring the
 * PN_DIE_RESERVE_BYTES env override. Pure. */
uint64_t pn_die_reserve_bytes(void);

/* The effective model-load margin per die (PN_DIE_LOAD_MARGIN_BYTES env
 * override). Pure. */
uint64_t pn_die_load_margin_bytes(void);

/* The effective PSI-full avg10 percentage at/above which admission rejects new
 * work, honoring the PN_PSI_FULL_REJECT env override (values <= 0 or garbage
 * fall back to the default — a zero threshold would re-arm the transient-blip
 * hypersensitivity). Pure — safe to call anywhere. */
double pn_psi_full_reject_pct(void);

/* The effective PSI escalation thresholds, honoring their env overrides
 * (PN_PSI_FULL_TIGHT / PN_PSI_FULL_WATCH / PN_PSI_SOME_TIGHT /
 * PN_PSI_SOME_WATCH; values <= 0 or garbage fall back to the default — a zero
 * threshold would re-arm the any-non-zero-is-pressure hypersensitivity this
 * replaced). Pure. */
double pn_psi_full_tight_pct(void);
double pn_psi_full_watch_pct(void);
double pn_psi_some_tight_pct(void);
double pn_psi_some_watch_pct(void);

/* Consecutive polls a sub-CRITICAL PSI escalation must hold before it counts,
 * honoring PN_PSI_TRIP_POLLS (values < 1 fall back to the default). Pure. */
int pn_psi_trip_polls(void);

/* --- Level derivation, split by evidence class (pure; exposed for tests) ---
 *
 * HARD signals are facts about bytes we can measure right now: host
 * MemAvailable against the reserve line, and each die's free VRAM against its
 * budget. They escalate immediately.
 *
 * The PSI level is what the stall counters *predict*. It is noisy and decays
 * slowly, so it is passed through the dwell filter below before it is allowed
 * to raise the published level. */
pn_mem_level pn_mem_hard_level(const pn_mem_state *s);
pn_mem_level pn_mem_psi_level(const pn_mem_state *s);

/* Dwell filter for the PSI level: a sub-CRITICAL escalation must be observed on
 * pn_psi_trip_polls() consecutive samples before `effective` follows it; any
 * sample back at OK clears the counter and relaxes immediately. CRITICAL is
 * published on the first sample. Caller-owned state so it is unit-testable
 * without a poll loop; pn_memwatch owns one instance internally. */
typedef struct {
    int          trips;      /* consecutive elevated samples so far */
    pn_mem_level effective;  /* the level the filter currently publishes */
} pn_psi_dwell;

void pn_psi_dwell_update(pn_psi_dwell *d, pn_mem_level raw);

/* Combine the hard level with the (dwell-filtered) PSI level into the level
 * that gets published. Sub-CRITICAL PSI is capped at WATCH while the hard
 * signals say OK: a stall with ample measured headroom is reclaim churn, not
 * exhaustion, so it may engage only the free rung. Once the byte math or a die
 * corroborates, PSI escalates in full. PSI-CRITICAL is never capped. Pure. */
pn_mem_level pn_mem_combine_level(pn_mem_level hard, pn_mem_level psi);

/* Start the background poll thread. Idempotent — a second call is a no-op while
 * one is already running. Returns 0 on success, negative on failure. The thread
 * reads only host signals; die numbers arrive via pn_memwatch_set_die.
 *
 * Thread-safe: yes (idempotent start guard)
 * Reentrant: no
 * Signal-safe: no */
int pn_memwatch_start(void);

/* Stop and join the poll thread. Safe to call if never started. */
void pn_memwatch_stop(void);

/* True once pn_memwatch_start has succeeded and the loop is live. */
bool pn_memwatch_running(void);

/* Read a self-consistent copy of the current published state into *out. Always
 * succeeds (returns a zeroed-but-valid snapshot before the first poll). Uses a
 * seqlock, so it never blocks the writer.
 *
 * Thread-safe: yes (lock-free reader)
 * Reentrant: yes
 * Signal-safe: no (loops on a torn read) */
void pn_memwatch_read(pn_mem_state *out);

/* Feed one die's VRAM numbers in from the GPU side (MemoryAware.md §2.2, §11
 * Q3). `free_bytes` is the latest cuMemGetInfo free; `budget_bytes` is the
 * boot-measured usable budget (total - reserve); `used_bytes` is our own
 * per-die alloc tally. Republishes the level. die_id must be < PN_MEM_MAX_DIES.
 *
 * Thread-safe: yes (single-writer publish is serialized internally)
 * Reentrant: no */
void pn_memwatch_set_die(int die_id, uint64_t free_bytes,
                         uint64_t budget_bytes, uint64_t used_bytes);

/* Force one immediate host-signal poll + publish, independent of the loop. Lets
 * a caller get fresh truth at a decision point (e.g. just before admission)
 * without waiting for the next tick. Safe to call whether or not the loop is
 * running. */
void pn_memwatch_poll_now(void);

/* Human-readable level name, for logging. */
const char *pn_mem_level_name(pn_mem_level level);

#ifdef __cplusplus
}
#endif

#endif /* PN_MEMWATCH_H */
