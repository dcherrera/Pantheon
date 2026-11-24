/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file host_stage_timer.c
 * @brief Per-request host-time accounting by stage (host_stage_timer.h).
 *
 * @status Instrument only.
 * @issues schedstat is Linux-only (0 elsewhere).
 * @todo None
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "host_stage_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

int pn_ht_state = -1;

static _Thread_local pn_host_timing *t_ht_acc;

void pn_ht_init(void)
{
    const char *e = getenv("PN_HOST_TIMING");
    pn_ht_state = (e && e[0] == '1') ? 1 : 0;
}

uint64_t pn_ht_now(void)
{
    struct timespec ts;

    if (!pn_ht_on())
        return 0;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t pn_ht_thread_cpu(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t pn_ht_run_delay(void)
{
#ifdef __linux__
    /* One fd per thread, re-read with pread: a proc seq file restarts at offset 0. */
    static _Thread_local int fd = -1;
    unsigned long long on_cpu = 0, delay = 0;
    char buf[128];

    if (fd < 0)
        fd = open("/proc/thread-self/schedstat", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return 0;
    ssize_t n = pread(fd, buf, sizeof buf - 1, 0);
    if (n <= 0)
        return 0;
    buf[n] = '\0';
    if (sscanf(buf, "%llu %llu", &on_cpu, &delay) != 2)
        return 0;
    return (uint64_t)delay;
#else
    return 0;
#endif
}

void pn_ht_bind(pn_host_timing *acc) { t_ht_acc = acc; }

pn_host_timing *pn_ht_bound(void) { return t_ht_acc; }

uint64_t pn_ht_lap(pn_ht_stage stage, uint64_t t0)
{
    if (!pn_ht_on())
        return 0;
    const uint64_t now = pn_ht_now();
    if (t_ht_acc && t0 && now > t0) {
        t_ht_acc->ns[stage] += now - t0;
        t_ht_acc->n[stage]++;
    }
    return now;
}

void pn_ht_add(pn_host_timing *acc, pn_ht_stage stage, uint64_t ns)
{
    if (!acc)
        return;
    acc->ns[stage] += ns;
    acc->n[stage]++;
}

void pn_ht_merge(pn_host_timing *dst, const pn_host_timing *src)
{
    if (!dst || !src)
        return;
    for (int i = 0; i < PN_HT_N; i++) {
        dst->ns[i] += src->ns[i];
        dst->n[i] += src->n[i];
    }
    dst->step_cpu_ns += src->step_cpu_ns;
    dst->steps += src->steps;
    dst->width_sum += src->width_sum;
    dst->masked_rows += src->masked_rows;
    dst->full_rows += src->full_rows;
}

void pn_ht_note_row(int masked, int full)
{
    if (!pn_ht_on() || !t_ht_acc)
        return;
    t_ht_acc->masked_rows += masked ? 1u : 0u;
    t_ht_acc->full_rows += full ? 1u : 0u;
}

static const char *const k_ht_names[PN_HT_N] = {
    "gate", "acquire", "step", "relock", "mtp_push", "release",
    "sample", "reason", "arm", "emit",
    "s_mask_select", "s_mask_decide", "s_mask_full", "s_topn", "s_surgery", "s_greedy",
    "s_sparse", "s_full_probs", "s_full_draw", "s_margin", "s_advance",
    "o_pop_wait", "o_detok", "o_logprob", "o_emit", "o_sse", "o_rld",
};

const char *pn_ht_stage_name(pn_ht_stage stage)
{
    return (stage >= 0 && stage < PN_HT_N) ? k_ht_names[stage] : "?";
}

/* "name=ms_per_token" for every stage with time; tokens = the divisor. */
static size_t ht_format_stages(char *buf, size_t cap, const pn_host_timing *t, int from,
                               int to, double tokens)
{
    size_t w = 0;

    for (int i = from; i < to && w < cap; i++) {
        if (!t->n[i])
            continue;
        int k = snprintf(buf + w, cap - w, " %s=%.3f", k_ht_names[i],
                         (double)t->ns[i] / 1e6 / tokens);
        if (k < 0)
            break;
        w += (size_t)k;
    }
    return w < cap ? w : cap;
}

void pn_ht_report(const pn_host_timing *drv, const pn_host_timing *own, int slot,
                  int generated, double decode_ms, double first_gap_ms, double tail_gap_ms,
                  uint64_t owner_cpu_ns, uint64_t owner_run_delay_ns)
{
    char line[2048];
    size_t w;

    if (!drv || !own || drv->steps < 2)
        return;
    /* Driver off-step: launch-to-launch span minus the steps inside it (the last step is
     * outside the span). Per token = per span interval. */
    const double span_ms = (double)(drv->last_launch_ns - drv->first_launch_ns) / 1e6;
    const double steps_in_span_ms =
        (double)(drv->ns[PN_HT_STEP] - drv->last_step_ns) / 1e6;
    const double intervals = (double)(drv->steps - 1);
    const double off_ms = (span_ms - steps_in_span_ms) / intervals;
    const double toks = (double)drv->steps;
    w = (size_t)snprintf(line, sizeof line,
                         "[host] slot=%d gen=%d steps=%u width=%.2f masked=%u full=%u "
                         "decode_ms=%.1f step_ms=%.2f off_ms=%.3f first_gap_ms=%.1f "
                         "tail_gap_ms=%.1f drv_cpu_in_step_ms=%.3f drv_run_delay_ms=%.3f "
                         "own_cpu_ms=%.3f own_run_delay_ms=%.3f |",
                         slot, generated, drv->steps, (double)drv->width_sum / toks,
                         drv->masked_rows, drv->full_rows, decode_ms,
                         (double)drv->ns[PN_HT_STEP] / 1e6 / toks, off_ms, first_gap_ms,
                         tail_gap_ms, (double)drv->step_cpu_ns / 1e6 / toks,
                         (double)drv->run_delay_ns / 1e6 / toks,
                         generated > 0 ? (double)owner_cpu_ns / 1e6 / generated : 0.0,
                         generated > 0 ? (double)owner_run_delay_ns / 1e6 / generated : 0.0);
    w += ht_format_stages(line + w, sizeof line - w, drv, 0, PN_HT_O_POP_WAIT, toks);
    if (w < sizeof line)
        w += (size_t)snprintf(line + w, sizeof line - w, " |");
    if (w < sizeof line)
        ht_format_stages(line + w, sizeof line - w, own, PN_HT_O_POP_WAIT, PN_HT_N,
                         generated > 0 ? (double)generated : 1.0);
    fprintf(stderr, "%s\n", line);
}
