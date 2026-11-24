/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_session_host_fit.c
 * @brief Implementation of per-session host sizing and the context-window fit.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "pn_session_host_fit.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pn_host_admission.h"

static bool arch_state_is_dense(pn_arch arch)
{
    return arch != PN_ARCH_RWKVX && arch != PN_ARCH_QWEN35MOE &&
           arch != PN_ARCH_BONSAI;
}

uint64_t pn_session_host_bytes(const pn_config *cfg)
{
    if (!cfg || !arch_state_is_dense(cfg->arch))
        return 0;
    uint64_t ctx = (uint64_t)pn_dense_session_max_seq(cfg);
    uint64_t kv = 2ull * (uint64_t)cfg->n_layer * (uint64_t)cfg->n_kv_head *
                  ctx * (uint64_t)pn_head_dim(cfg) * sizeof(float);
    return kv + (uint64_t)pn_dense_arena_bytes(cfg);
}

/* Session bytes for `cfg` with its window forced to `ctx`. */
static uint64_t bytes_at_ctx(const pn_config *cfg, int ctx)
{
    pn_config probe = *cfg;
    probe.ctx_cap = ctx;
    return pn_session_host_bytes(&probe);
}

/* Largest window in [lo, hi] whose n sessions fit `room` (bytes are monotone
 * in the window), or 0 when even `lo` does not. */
static int largest_fitting_ctx(const pn_config *cfg, int n, int lo, int hi,
                               uint64_t room)
{
    if (bytes_at_ctx(cfg, lo) * (uint64_t)n > room)
        return 0;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (bytes_at_ctx(cfg, mid) * (uint64_t)n <= room)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

static bool window_is_operator_chosen(const pn_config *cfg)
{
    const char *env = getenv("PN_SESSION_MAX_CTX");
    return cfg->ctx_cap > 0 || (env && atoi(env) > 0);
}

static void refuse(char *reason, size_t reason_len, int n_sessions, int ctx,
                   uint64_t need, const char *hint)
{
    char what[96];
    snprintf(what, sizeof(what), "host KV ring (%d session%s x %d positions)",
             n_sessions, n_sessions == 1 ? "" : "s", ctx);
    /* Re-run through the module so the refusal text and log line are the one
     * format every site uses; it refuses again because nothing changed. */
    if (pn_host_admit_hint(need, what, hint, reason, reason_len) == 0)
        pn_host_release(need);   /* raced into fitting: give the charge back */
}

int pn_session_host_fit(pn_model *model, int n_sessions, char *reason,
                        size_t reason_len)
{
    if (reason && reason_len)
        reason[0] = '\0';
    if (!model || !pn_host_admit_enabled())
        return 0;
    pn_config *cfg = &model->config;
    if (n_sessions < 1)
        n_sessions = 1;
    uint64_t per = pn_session_host_bytes(cfg);
    uint64_t need = per * (uint64_t)n_sessions;
    uint64_t room = pn_host_admit_headroom();
    if (per == 0 || need <= room)
        return 0;

    int ctx = pn_dense_session_max_seq(cfg);
    if (window_is_operator_chosen(cfg)) {
        refuse(reason, reason_len, n_sessions, ctx, need,
               "lower --ctx-cap / PN_SESSION_MAX_CTX or --max-sessions");
        return -1;
    }
    int lo = PN_SESSION_HOST_FIT_MIN_CTX < ctx ? PN_SESSION_HOST_FIT_MIN_CTX : ctx;
    int fit = largest_fitting_ctx(cfg, n_sessions, lo, ctx, room);
    if (fit == 0) {
        refuse(reason, reason_len, n_sessions, lo, bytes_at_ctx(cfg, lo) *
               (uint64_t)n_sessions, "lower --max-sessions or free host memory");
        return -1;
    }
    char s_room[32];
    pn_host_admit_format_bytes(room, s_room, sizeof(s_room));
    fprintf(stderr, "engine: host RAM caps the default context %d -> %d "
            "positions (%d session%s, %s admissible above the reserve; set "
            "--ctx-cap to choose)\n", ctx, fit, n_sessions,
            n_sessions == 1 ? "" : "s", s_room);
    cfg->ctx_cap = fit;
    return 0;
}

int pn_session_host_check(const pn_model *model, int n_sessions, char *reason,
                          size_t reason_len)
{
    if (reason && reason_len)
        reason[0] = '\0';
    if (!model || n_sessions < 1)
        return 0;
    uint64_t need = pn_session_host_bytes(&model->config) * (uint64_t)n_sessions;
    char what[96];
    snprintf(what, sizeof(what), "host KV ring (%d session%s x %d positions)",
             n_sessions, n_sessions == 1 ? "" : "s",
             pn_dense_session_max_seq(&model->config));
    return pn_host_admit_check(need, what, PN_HOST_ADMIT_DEFAULT_HINT, reason,
                               reason_len);
}
