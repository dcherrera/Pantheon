/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_backend_guard.h
 * @brief The one decision that answers "was this serve asked to use dies it
 *        cannot possibly use?", plus the operator-facing text for each way the
 *        answer can be yes.
 *
 *        The incident: a CPU-only binary accepted `--dies 0,1`, loaded a 35B
 *        MoE, and served it on a 2-core no-AVX Pentium with all 8 K80 dies at
 *        0 MiB. Nothing refused, and the only trace in the log was
 *        `placed on dies 0,1 (0.0 GiB projected)` — which reads like a rounding
 *        artifact, not like "there is no GPU here". bs_enabled()/bs_create()
 *        are #ifndef USE_CUDA stubs, so the batch scheduler was NULL too and
 *        every second chat 429'd, at an advertised max_concurrent of 2.
 *
 *        A warning would not have helped: the serve had already been running
 *        for an unknown period and measurements had already been taken off it.
 *        The only thing that catches this is refusing to start.
 *
 *        Both callers use this: pn_serve_main pre-flights every `--model` spec
 *        BEFORE loading anything (so the operator gets the message in a second,
 *        not after a 55 s weight upload), and mr_load re-checks so a runtime
 *        POST /admin/models/load with a `dies` array is refused by the same
 *        rule rather than by a second, drifting copy of it.
 *
 *        n_dies == 0 is always OK here, on every backend: where such a load
 *        runs is the load guard's call (core/load_guard.h) — the CPU forward in
 *        a CPU build or with `--cpu`, never as a fallback in an accelerator one.
 *
 *        Note the first arm is COMPILE-TIME, deliberately, and is checked
 *        before the die count. pn_dm_init_logical() can hand a CPU build a
 *        non-zero pn_dm_count() (it is placement bookkeeping for the Mac/test
 *        path — see die_manager.h), and trusting the count alone would let
 *        exactly the incident's shape through again on any host that ever calls
 *        it. Logical dies are not silicon, so an explicit --dies against them is
 *        refused too.
 *
 * @status Complete; unit-tested (tests/test_backend_guard.c). Pure logic — no
 *         probing, no I/O.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_BACKEND_GUARD_H
#define PN_SERVE_BACKEND_GUARD_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Why a die request cannot be honoured — or that it can. */
typedef enum {
    /* Either no dies were requested (a deliberate CPU serve) or they can be
     * honoured. Nothing to say. */
    PN_BACKEND_GUARD_OK = 0,
    /* Dies were named and this binary has NO die-capable backend compiled in.
     * The `make` that produced it was a CPU build (or a Metal one). This is the
     * incident's failure mode. */
    PN_BACKEND_GUARD_NO_DIE_BACKEND,
    /* Dies were named, the backend IS compiled in, but the die manager probed
     * zero dies — no driver, no card, an empty CUDA_VISIBLE_DEVICES. The
     * process would still serve, on the CPU, exactly as silently. */
    PN_BACKEND_GUARD_NO_DIES_VISIBLE,
} pn_backend_guard_verdict;

/**
 * Decide whether a die request can be honoured.
 *
 * @param n_dies_requested    How many dies this model asked for (0 = none, the
 *                            deliberate CPU serve — always OK).
 * @param die_backend_compiled  pn_backend_supports_die_placement().
 * @param dies_visible        pn_dm_count() — dies the manager actually holds.
 * @return                    The verdict; PN_BACKEND_GUARD_OK to proceed.
 *
 * Pure: no globals read, no I/O, so the caller supplies both facts and the test
 * can supply any combination of them.
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: yes
 */
pn_backend_guard_verdict pn_backend_guard_check(int n_dies_requested,
                                                bool die_backend_compiled,
                                                int dies_visible);

/**
 * The full operator message for a non-OK verdict: what was asked, what this
 * binary is, why continuing would be worse than stopping, and the two ways
 * forward (deploy an accelerator build, or drop --dies on purpose).
 *
 * Written as ONE line so it greps out of a fleet log, and it names the concrete
 * binary to build — the deploy path being wrong is the whole point, so "use the
 * right build" without the command is not a fix.
 *
 * @param verdict     From pn_backend_guard_check (PN_BACKEND_GUARD_OK writes an
 *                    empty string and returns 0).
 * @param model_name  Served name for the model that asked, or NULL.
 * @param dies        The requested die indices, or NULL.
 * @param n_dies      How many entries in `dies`.
 * @param out         Destination buffer.
 * @param out_len     Its size; the message is truncated, never overrun.
 * @return            Number of bytes that WOULD have been written (snprintf
 *                    semantics), or 0 for PN_BACKEND_GUARD_OK.
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: no (snprintf)
 */
int pn_backend_guard_message(pn_backend_guard_verdict verdict,
                             const char *model_name,
                             const int *dies, int n_dies,
                             char *out, size_t out_len);

/**
 * The SHORT form of the same refusal, for callers that carry a reason string
 * rather than print (mr_load's err[]). Stable text — model_admin.c's
 * status_for_reason matches "no accelerator" on it to answer 400 instead of the
 * catch-all 500.
 *
 * @return A string literal, or NULL for PN_BACKEND_GUARD_OK.
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: yes
 */
const char *pn_backend_guard_short_reason(pn_backend_guard_verdict verdict);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_BACKEND_GUARD_H */
