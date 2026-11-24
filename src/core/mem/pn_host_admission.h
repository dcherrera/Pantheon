/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_host_admission.h
 * @brief Host-RAM admission control: every large host allocation asks here
 *        first, and a request that would push the box past its reserve is
 *        REFUSED with a human message instead of being attempted and OOM-ing
 *        the host (the 2026-10-27 K80 bank outages: a 40,000-position dense
 *        KV ring in a CLI run, and 8 x 3.5 GB of scheduler host KV).
 *
 *        The decision is
 *
 *            bytes <= MemAvailable - outstanding - reserve
 *
 *        where `outstanding` is what this process has admitted but not yet
 *        touched (admitted minus the growth of its anonymous RSS since the
 *        first admission). An eager allocation (malloc + memset) shows up in
 *        MemAvailable as soon as it is touched, so it is not charged twice; a
 *        lazy one (calloc of a big block) is still owed and is charged, so a
 *        run of allocations in one process adds up either way.
 *
 *        mmap'd weights are page cache and never come through here; anonymous
 *        copies of weights do.
 *
 *        Environment:
 *          PN_HOST_ADMIT=0          disable (escape hatch; everything admits)
 *          PN_HOST_RESERVE_MB=N     reserve in MiB (default 1536)
 *          PN_HOST_ADMIT_TEST_AVAIL_MB=N  TEST HOOK: pretend MemAvailable is N
 *                                   MiB and ignore RSS growth, so a test can
 *                                   drive a refusal on any machine.
 *
 * @status Complete
 * @issues Cross-process: two processes that each pass admission before either
 *         touches its memory can still oversubscribe together. Eager sites
 *         close that window quickly; the systemd-run MemoryMax backstop in the
 *         README covers ad-hoc jobs.
 * @todo None
 */

#ifndef PANTHEON_HOST_ADMISSION_H
#define PANTHEON_HOST_ADMISSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_HOST_ADMIT_RESERVE_DEFAULT_BYTES (1536ULL * 1024 * 1024) /* 1.5 GiB */

/* Allocations below this are not worth a meminfo read; callers may skip them. */
#define PN_HOST_ADMIT_THRESHOLD_BYTES (64ULL * 1024 * 1024)

/* A reason buffer this large always holds the full refusal message. */
#define PN_HOST_ADMIT_REASON_LEN 320

/* The hint used by pn_host_admit when the caller gives none. */
#define PN_HOST_ADMIT_DEFAULT_HINT "lower --ctx-cap or --max-sessions"

/**
 * Is admission control on? False only when PN_HOST_ADMIT=0.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no (getenv).
 */
bool pn_host_admit_enabled(void);

/**
 * The host reserve in bytes: PN_HOST_RESERVE_MB when set to a positive number,
 * else PN_HOST_ADMIT_RESERVE_DEFAULT_BYTES.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no.
 */
uint64_t pn_host_admit_reserve_bytes(void);

/**
 * Admit `bytes` of host memory for `what` (e.g. "host KV ring"), or refuse.
 * On admission the bytes are charged to this process until pn_host_release.
 * On refusal nothing is charged, `reason` receives
 *   "<what> needs 4.6 GiB, 2.1 GiB available (reserve 1.5 GiB); <hint>"
 * and the same line is logged to stderr. Disabled, or MemAvailable
 * unobservable: admits without checking (and charges, so a later reading
 * still accounts for it).
 *
 * @param bytes       Size of the allocation about to be made
 * @param what        Short noun phrase naming it (non-NULL)
 * @param hint        What the operator can change; NULL = default hint
 * @param reason      Out: refusal message (may be NULL)
 * @param reason_len  Size of `reason` (PN_HOST_ADMIT_REASON_LEN suffices)
 * @return            0 admitted, -1 refused
 *
 * Thread-safe: yes (check + charge under one lock). Reentrant: no.
 * Signal-safe: no.
 */
int pn_host_admit_hint(uint64_t bytes, const char *what, const char *hint,
                       char *reason, size_t reason_len);

/** pn_host_admit_hint with the default hint. */
int pn_host_admit(uint64_t bytes, const char *what, char *reason,
                  size_t reason_len);

/**
 * Check-only admission for a TRANSIENT buffer that is written as soon as it is
 * allocated and freed by the same caller (a serialize buffer, a base64 blob,
 * an upload staging copy): admits, then immediately releases, so nothing is
 * left charged. Same return and reason contract as pn_host_admit_hint.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
int pn_host_admit_check(uint64_t bytes, const char *what, const char *hint,
                        char *reason, size_t reason_len);

/**
 * Admit an anonymous host copy of a weight tensor named `name` (dequantized,
 * converted or tied-head copy) when it is at least
 * PN_HOST_ADMIT_THRESHOLD_BYTES; smaller copies pass unchecked. The charge is
 * the loading thread's (pn_load_on hands it to the model). 0 or -1.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
int pn_host_admit_weight_copy(uint64_t bytes, const char *name);

/**
 * Return `bytes` previously admitted (call when the allocation is freed).
 * Never drops the charge below zero.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
void pn_host_release(uint64_t bytes);

/**
 * The last refusal message produced on THIS thread ("" if none), so a caller
 * several frames above a refused allocation (pn_state_new returning NULL) can
 * still hand the operator the exact reason.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no.
 */
const char *pn_host_admit_last_refusal(void);

/**
 * Count of refusals made ON THIS THREAD (never decreases). Read it before an
 * operation and again after it fails: a change means the failure was a host
 * refusal (answer 503 with pn_host_admit_last_refusal), not a bug.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: yes.
 */
uint64_t pn_host_admit_thread_refusals(void);

/**
 * Running total of bytes admitted ON THIS THREAD (never decreases). A loader
 * brackets itself with two readings to learn what it admitted for the object
 * it built, so the object's free can release exactly that.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: yes.
 */
uint64_t pn_host_admit_thread_total(void);

/** Bytes admitted and not yet released by this process. Thread-safe: yes. */
uint64_t pn_host_admitted_bytes(void);

/**
 * Bytes a new admission could take right now (MemAvailable - outstanding -
 * reserve, floored at 0). UINT64_MAX when disabled or unobservable.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
uint64_t pn_host_admit_headroom(void);

/**
 * Largest unit count n in [min_units, want_units] for which
 * `fixed_bytes + n * per_unit_bytes` fits the current headroom — how a default
 * that does not fit (a 40,000-position ring) is capped to one that does.
 * Returns want_units when it fits, 0 when even min_units does not.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
long pn_host_admit_fit_units(uint64_t fixed_bytes, uint64_t per_unit_bytes,
                             long want_units, long min_units);

/**
 * Format `bytes` as "4.6 GiB" (>= 1 GiB) or "512 MiB" into `out`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no.
 */
void pn_host_admit_format_bytes(uint64_t bytes, char *out, size_t out_len);

/* --- Test hooks (unit tests only) --- */

/**
 * Pretend MemAvailable is `avail_bytes` and this process's anonymous RSS is
 * `anon_bytes`. avail_bytes == 0 removes the hook (real readings again).
 */
void pn_host_admit_test_set_memory(uint64_t avail_bytes, uint64_t anon_bytes);

/** Clear the hook, the admitted total and the RSS baseline. */
void pn_host_admit_test_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* PANTHEON_HOST_ADMISSION_H */
