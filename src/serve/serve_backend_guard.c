/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_backend_guard.c
 * @brief Implementation of the degraded-start refusal (see
 *        serve_backend_guard.h).
 *
 * The message text is here rather than at the two call sites on purpose: an
 * operator who hits this at startup and an operator who hits it through
 * /admin/models/load are looking at the same mistake, and a second copy of the
 * wording is a second thing to let drift.
 *
 * @status Complete; unit-tested (tests/test_backend_guard.c).
 * @issues None
 * @todo None
 */

#include "serve_backend_guard.h"

#include <stdio.h>
#include <string.h>

#include "../accel/backend_identity.h"

pn_backend_guard_verdict pn_backend_guard_check(int n_dies_requested,
                                                bool die_backend_compiled,
                                                int dies_visible)
{
    /* No dies asked for = a deliberate CPU serve. Always allowed, on every
     * backend: this is the supported way to run without an accelerator, and it
     * has to keep working or the guard just becomes a thing people disable. */
    if (n_dies_requested <= 0)
        return PN_BACKEND_GUARD_OK;

    if (!die_backend_compiled)
        return PN_BACKEND_GUARD_NO_DIE_BACKEND;

    if (dies_visible <= 0)
        return PN_BACKEND_GUARD_NO_DIES_VISIBLE;

    return PN_BACKEND_GUARD_OK;
}

const char *pn_backend_guard_short_reason(pn_backend_guard_verdict verdict)
{
    switch (verdict) {
    case PN_BACKEND_GUARD_NO_DIE_BACKEND:
        return "dies requested but this binary has no accelerator backend "
               "compiled in (see the startup backend= line)";
    case PN_BACKEND_GUARD_NO_DIES_VISIBLE:
        return "dies requested but no accelerator dies are visible to this "
               "process (see GET /admin/dies)";
    case PN_BACKEND_GUARD_OK:
    default:
        return NULL;
    }
}

/* Render "0,1" (or "-" when the list is empty) into out[]. Bounded; a list too
 * long for the buffer is truncated rather than overrunning it. */
static void format_die_list(const int *dies, int n_dies, char *out,
                            size_t out_len)
{
    size_t used = 0;

    if (!out || out_len == 0)
        return;
    out[0] = '\0';
    if (!dies || n_dies <= 0) {
        snprintf(out, out_len, "-");
        return;
    }
    for (int i = 0; i < n_dies && used + 1 < out_len; i++) {
        int n = snprintf(out + used, out_len - used, "%s%d", i ? "," : "",
                         dies[i]);
        if (n < 0 || (size_t)n >= out_len - used) {
            out[out_len - 1] = '\0';
            return;
        }
        used += (size_t)n;
    }
}

int pn_backend_guard_message(pn_backend_guard_verdict verdict,
                             const char *model_name,
                             const int *dies, int n_dies,
                             char *out, size_t out_len)
{
    char die_list[128];
    const char *name = (model_name && *model_name) ? model_name : "(unnamed)";

    if (!out || out_len == 0)
        return 0;
    out[0] = '\0';
    if (verdict == PN_BACKEND_GUARD_OK)
        return 0;

    format_die_list(dies, n_dies, die_list, sizeof(die_list));

    if (verdict == PN_BACKEND_GUARD_NO_DIE_BACKEND) {
        /* The exact case that shipped: `make` (CPU) writing the same output
         * path a `make USE_CUDA=1` had written, then serving a MoE off a
         * Pentium. Name the binary to build — the deploy path being wrong is
         * the failure, so "use the right build" without the command is not a
         * fix. */
        return snprintf(out, out_len,
            "serve: REFUSED to start — model '%s' asks for die(s) %s but this binary "
            "has NO accelerator backend compiled in (backend=%s build=%s). It would "
            "load the model and answer every request on the host CPU with every die "
            "idle, and nothing in the log would say so. Fix: deploy an accelerator "
            "build — `make USE_CUDA=1` produces ./pantheon-cuda, `make USE_TT=1` "
            "produces ./pantheon-tt — or drop --dies to serve on the CPU "
            "deliberately.",
            name, die_list, pn_backend_name(), pn_backend_build_commit());
    }

    return snprintf(out, out_len,
        "serve: REFUSED to start — model '%s' asks for die(s) %s and this binary has "
        "the %s backend, but the die manager probed 0 dies (build=%s). It would load "
        "the model and answer every request on the host CPU, as silently as if the "
        "dies had been honoured. Fix: make the dies visible (driver loaded, card "
        "present, CUDA_VISIBLE_DEVICES not empty) — or replace --dies with --cpu to "
        "serve on the CPU deliberately.",
        name, die_list, pn_backend_name(), pn_backend_build_commit());
}
