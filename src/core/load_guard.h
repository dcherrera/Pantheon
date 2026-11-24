/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_guard.h
 * @brief  Where a model load may run: no silent CPU fallback, and no CPU load
 *         the host cannot hold.
 *
 *         A load in a build that places models on accelerator dies (CUDA,
 *         Tenstorrent) runs on an accelerator or FAILS. When the upload refuses
 *         (die admission, an allocation, no device visible, no accelerator
 *         forward for the architecture) the load is refused with the reason;
 *         it never keeps the host CPU forward. On the K80 bank (8 GB, 2 cores)
 *         that fallback put Ornith's 9.8 GB of weights on the host, which
 *         thrashed until killed, and the error then read "no MTP head".
 *
 *         The CPU forward runs only when asked for: a build with no die
 *         accelerator (CPU, Metal), or `cpu_only` (`--cpu`, a Local `cpu`
 *         profile). Even then the weights the CPU forward reads every token
 *         must fit host RAM, or the load is refused before it starts.
 *
 *         Refusal reasons are thread-local strings: an upload several frames
 *         down notes WHY (pn_accel_refusal_note); pn_load_on turns it into the
 *         operator message (pn_load_refusal_last), which the registry reports.
 *
 * @status Complete
 * @issues The CPU working set of a MoE model counts the routed experts at
 *         used/total; over many tokens every expert is read, so a model whose
 *         full file does not fit pages its cold experts from disk (slow, not
 *         a swap thrash: they are clean file pages).
 * @todo   None
 */

#ifndef PN_LOAD_GUARD_H
#define PN_LOAD_GUARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A buffer this large holds any refusal message in full. */
#define PN_LOAD_REFUSAL_LEN 320

/**
 * Note why the accelerator refused this load ("die 6 needs ~11945 MiB, has
 * 11467 MiB usable"). Called by an upload path when it gives up; the last note
 * on this thread wins. printf-style.
 *
 * Thread-safe: yes (thread-local). Reentrant: no. Signal-safe: no.
 */
void pn_accel_refusal_note(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));

/** Forget this thread's accelerator note. Thread-safe: yes. */
void pn_accel_refusal_clear(void);

/** This thread's accelerator note, "" if none. Thread-safe: yes. */
const char *pn_accel_refusal_last(void);

/**
 * The last load refusal made on this thread ("" if none): the full operator
 * message, e.g. "Ornith.gguf did not fit GPU die 6: ... Not run on the CPU".
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no.
 */
const char *pn_load_refusal_last(void);

/**
 * Count of load refusals made on THIS thread (never decreases). Read it before
 * a load and again after it fails: a change means the guard refused it and
 * pn_load_refusal_last says why.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: yes.
 */
uint64_t pn_load_thread_refusals(void);

/**
 * Does this build place models on accelerator dies (CUDA or Tenstorrent, or
 * the test fake)? Metal is not counted: its forward computes on the CPU.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no.
 */
bool pn_load_guard_accel_build(void);

/**
 * Will a load with these options run the host CPU forward by request? True for
 * `cpu_only`, or a build with no die accelerator.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no.
 */
bool pn_load_guard_cpu_planned(bool cpu_only);

/**
 * Bytes the CPU forward of the model at `path` reads per token, which must stay
 * resident: a GGUF's tensors with the routed experts (`*_exps.*`) counted at
 * expert_used_count / expert_count; any other file, its size. 0 = unknown.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no.
 */
uint64_t pn_load_guard_cpu_working_set(const char *path);

/**
 * Refuse a CPU-forward load whose working set (pn_load_guard_cpu_working_set)
 * does not fit host RAM (MemAvailable less the host reserve, the same rule as
 * pn_host_admission; PN_HOST_ADMIT=0 turns it off). The weights are mmap'd, so
 * nothing stays charged. On refusal the message is the thread's load refusal.
 *
 * @return 0 fits (or unknown), -1 refused
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
int pn_load_guard_cpu_host_fit(const char *path);

/**
 * Record the refusal of a load that was asked to run on an accelerator and did
 * not land on one: which dies, the accelerator's note (or a generic reason), and
 * what to change. Logs it and makes it the thread's load refusal.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
void pn_load_guard_refuse_accel(const char *path, const int *dies, int n_dies);

/* --- Test hook (unit tests only) --- */

/**
 * Pretend this build places models on an accelerator whose upload refuses
 * every load with `reason` (the fake runs where the upload would). NULL turns
 * the fake off.
 */
void pn_load_guard_test_fake_accel(const char *reason);

/** The fake's reason, or NULL when it is off. */
const char *pn_load_guard_test_fake_reason(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOAD_GUARD_H */
