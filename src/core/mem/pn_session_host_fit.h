/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_session_host_fit.h
 * @brief What one session costs in host RAM, and fitting a model's per-session
 *        context window to what the host can admit (pn_host_admission.h).
 *
 *        The dense (llama-family) session is an EAGER host allocation — the
 *        fp32 KV ring for every layer at the session window plus its scratch
 *        arena — so it is admitted up front. A default window that does not
 *        fit (the 40,000-position ring) is capped to one that does, with a log
 *        line; a window the operator chose (--ctx-cap, PN_SESSION_MAX_CTX) is
 *        refused instead of silently shrunk.
 *
 *        Hybrid (qwen35moe/bonsai) and RWKV states are not priced here: their
 *        KV/SSM mirrors are lazy (pn_lazy_zero), committed per row actually
 *        written, and per-request growth is gated by pn_admit_request.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#ifndef PANTHEON_SESSION_HOST_FIT_H
#define PANTHEON_SESSION_HOST_FIT_H

#include <stddef.h>
#include <stdint.h>

#include "../engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The smallest window a default may be capped to before the load is refused. */
#define PN_SESSION_HOST_FIT_MIN_CTX 512

/**
 * Eager host bytes ONE session (pn_state_new) of `cfg` allocates: the dense KV
 * ring + scratch arena. 0 for architectures whose state is lazy (hybrid, RWKV).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no (getenv).
 */
uint64_t pn_session_host_bytes(const pn_config *cfg);

/**
 * Make `n_sessions` sessions of `model` fit the host, BEFORE any is allocated.
 *   - fits as configured: 0, nothing changes;
 *   - default window does not fit: model->config.ctx_cap is lowered to the
 *     largest window that does (>= PN_SESSION_HOST_FIT_MIN_CTX), logged; 0;
 *   - an operator-chosen window (ctx_cap, PN_SESSION_MAX_CTX) does not fit,
 *     or not even the minimum does: -1, `reason` filled and logged.
 * Does not charge anything; pn_state_new charges each session as it is made.
 *
 * Thread-safe: no (writes model->config). Reentrant: no. Signal-safe: no.
 */
int pn_session_host_fit(pn_model *model, int n_sessions, char *reason,
                        size_t reason_len);

/**
 * Check, without changing anything, that `n_sessions` MORE sessions of `model`
 * at its current window fit the host — the precheck before a batch of eager
 * slot states is allocated (the scheduler's slots). 0 fits; -1 refused with
 * `reason` filled and logged ("... lower --max-sessions or --ctx-cap").
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no.
 */
int pn_session_host_check(const pn_model *model, int n_sessions, char *reason,
                          size_t reason_len);

#ifdef __cplusplus
}
#endif

#endif /* PANTHEON_SESSION_HOST_FIT_H */
