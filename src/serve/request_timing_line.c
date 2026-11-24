/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_timing_line.c
 * @brief The per-request `[req]` timing line (request_timing_line.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#include "request_timing_line.h"

#include <stdio.h>

/* One detached pthread per connection (serve_command.c), so a thread-local
 * mark needs no plumbing and cannot be raced by another request. */
static _Thread_local double g_arrival_ms;

double sg_request_timing_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

double sg_request_timing_tok_s(const sg_request_timing *t)
{
    if (!t || t->generated < 2 || t->decode_ms <= 0.0)
        return 0.0;
    return (double)(t->generated - 1) * 1000.0 / t->decode_ms;
}

int sg_request_timing_format(const sg_request_timing *t, const struct timespec *wall,
                             char *buf, size_t cap)
{
    struct tm lt;
    time_t secs = wall->tv_sec;
    localtime_r(&secs, &lt);
    char stamp[32], zone[8];
    strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", &lt);
    strftime(zone, sizeof zone, "%z", &lt);
    return snprintf(buf, cap,
                    "[req] %s.%03ld%s slot=%d prompt=%d prefill=%d ttft_ms=%.1f gen=%d "
                    "decode_ms=%.1f tok_s=%.1f active=%d-%d mtp=%s fin_ms=%.1f finish=%s",
                    stamp, wall->tv_nsec / 1000000L, zone, t->slot, t->prompt_tok,
                    t->prefill_tok, t->ttft_ms, t->generated, t->decode_ms,
                    sg_request_timing_tok_s(t), t->active_min, t->active_max,
                    t->mtp_cycles > 0 ? "yes" : "no", t->finalize_ms,
                    t->finish ? t->finish : "-");
}

void sg_request_timing_log(const sg_request_timing *t)
{
    struct timespec wall;
    clock_gettime(CLOCK_REALTIME, &wall);
    char line[384];
    sg_request_timing_format(t, &wall, line, sizeof line);
    fprintf(stderr, "%s\n", line);
}

void sg_request_arrival_mark(void)
{
    g_arrival_ms = sg_request_timing_now_ms();
}

double sg_request_arrival_take(void)
{
    double t = g_arrival_ms;
    g_arrival_ms = 0.0;
    return t;
}
