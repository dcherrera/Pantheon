/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_profile.c
 * @brief Implementation of the per-request wall-clock phase profiler
 *        (see request_profile.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "request_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PN_REQPROF_PATH_MAX 96

/* One request's accumulated timings. Thread-local: the serve owns one detached
 * pthread per connection, so this needs no lock and no plumbing. */
typedef struct {
    int    active;
    double t_begin;
    double open_at[PN_REQ_PHASE_COUNT];   /* 0.0 == no span open */
    double total_ms[PN_REQ_PHASE_COUNT];
    char   method[8];
    char   path[PN_REQPROF_PATH_MAX];
    int    has_sid;
    int    prompt_tok;
    int    delta_start;
    int    sys_len;
    int    generated_tok;
} pn_reqprof_record;

static _Thread_local pn_reqprof_record g_record;

static const char *const PHASE_LABELS[PN_REQ_PHASE_COUNT] = {
    "http", "json", "route", "admit", "slot_wait",
    "tokenize", "cache", "prefill", "first_sample", "decode"
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

int pn_reqprof_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("PN_REQPROF");
        cached = (v && v[0] == '1') ? 1 : 0;
    }
    return cached;
}

void pn_reqprof_begin(void)
{
    if (!pn_reqprof_enabled())
        return;
    memset(&g_record, 0, sizeof(g_record));
    g_record.active = 1;
    g_record.delta_start = -1;      /* -1 == never reached the cache decision */
    g_record.t_begin = now_ms();
}

void pn_reqprof_phase_start(pn_req_phase phase)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;
    if (phase < 0 || phase >= PN_REQ_PHASE_COUNT)
        return;
    /* A span already open means a re-entrant/duplicated start: keep the first
     * so the phase measures the outermost span rather than silently drifting. */
    if (g_record.open_at[phase] == 0.0)
        g_record.open_at[phase] = now_ms();
}

void pn_reqprof_phase_end(pn_req_phase phase)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;
    if (phase < 0 || phase >= PN_REQ_PHASE_COUNT)
        return;
    if (g_record.open_at[phase] == 0.0)
        return;
    g_record.total_ms[phase] += now_ms() - g_record.open_at[phase];
    g_record.open_at[phase] = 0.0;
}

void pn_reqprof_note_route(const char *method, const char *path)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;
    if (method)
        snprintf(g_record.method, sizeof(g_record.method), "%s", method);
    if (path)
        snprintf(g_record.path, sizeof(g_record.path), "%s", path);
}

void pn_reqprof_note_cache(int has_sid, int prompt_tok, int delta_start,
                           int sys_len)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;
    g_record.has_sid     = has_sid;
    g_record.prompt_tok  = prompt_tok;
    g_record.delta_start = delta_start;
    g_record.sys_len     = sys_len;
}

void pn_reqprof_note_generated(int generated_tok)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;
    g_record.generated_tok = generated_tok;
}

void pn_reqprof_end(void)
{
    if (!pn_reqprof_enabled() || !g_record.active)
        return;

    double total = now_ms() - g_record.t_begin;
    double attributed = 0.0;
    char phases[512];
    size_t off = 0;
    for (int i = 0; i < PN_REQ_PHASE_COUNT; i++) {
        /* Close anything still open so an early-return path (429, error) still
         * reports the span it was inside rather than dropping it. */
        if (g_record.open_at[i] != 0.0)
            pn_reqprof_phase_end((pn_req_phase)i);
        attributed += g_record.total_ms[i];
        int n = snprintf(phases + off, sizeof(phases) - off, " %s=%.1f",
                         PHASE_LABELS[i], g_record.total_ms[i]);
        if (n < 0 || (size_t)n >= sizeof(phases) - off)
            break;
        off += (size_t)n;
    }

    int prefilled = (g_record.delta_start >= 0)
                        ? g_record.prompt_tok - g_record.delta_start
                        : g_record.prompt_tok;

    fprintf(stderr,
            "[reqprof] %s %s total=%.1fms%s unattributed=%.1f | sid=%d "
            "prompt_tok=%d delta_start=%d prefilled_tok=%d sys_len=%d gen=%d\n",
            g_record.method[0] ? g_record.method : "?",
            g_record.path[0] ? g_record.path : "?",
            total, phases, total - attributed,
            g_record.has_sid, g_record.prompt_tok, g_record.delta_start,
            prefilled, g_record.sys_len, g_record.generated_tok);
    fflush(stderr);
    g_record.active = 0;
}
