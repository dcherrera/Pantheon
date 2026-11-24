/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_host_gap.c
 * @brief HOST GAP clock — see pn_tt_host_gap.h.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* RUSAGE_THREAD */
#endif
#include "pn_tt_host_gap.h"

#include "../../core/sample_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* 160, not 48: the launch lane's `live:<residency>` split filled 48 with prompt/warm-up labels
 * and every later label (verify:*, mtp:*, engine:*) was dropped (TTFT lane, 2026-10-22). */
#define HOST_GAP_BUCKETS 160

typedef struct {
    const char *phase;
    const char *why;
    unsigned long long count;
    double ms;
} host_gap_bucket;

static struct {
    int enabled; /* -1 unread */
    const char *phase;
    const char *open_phase; /* the phase the open gap is booked under */
    const char *why;
    double idle_since; /* 0 = no gap open */
    host_gap_bucket bucket[HOST_GAP_BUCKETS];
    unsigned buckets;
} g_gap = { -1, "?", "?", "?", 0.0, { { 0 } }, 0u };

/* `PN_TT_HOST_GAP_PROFILE=<phase>:<why>` (with `PN_PROFILE=<file>`): the sample profiler
 * records only while a gap under that label is open — where the host's time in it goes. */
static void host_gap_profile(const char *phase, const char *why, bool open)
{
    static const char *target = (const char *)1;
    static bool inside;
    char label[96];
    bool want;

    if (target == (const char *)1)
        target = getenv("PN_TT_HOST_GAP_PROFILE");
    if (!target)
        return;
    snprintf(label, sizeof(label), "%s:%s", phase ? phase : "?", why ? why : "?");
    want = open && strcmp(label, target) == 0;
    if (want != inside) {
        inside = want;
        pn_sample_profiler_window(want ? 1 : 0);
    }
}

static double host_gap_now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

bool pn_tt_host_gap_enabled(void)
{
    if (g_gap.enabled < 0) {
        const char *env = getenv("PN_TT_HOST_GAP");

        g_gap.enabled = env && atoi(env) != 0;
    }
    return g_gap.enabled != 0;
}

void pn_tt_host_gap_phase(const char *phase)
{
    g_gap.phase = phase ? phase : "?";
}

void pn_tt_host_gap_card_idle(const char *why)
{
    if (!pn_tt_host_gap_enabled())
        return;
    g_gap.why = why ? why : "?";
    g_gap.open_phase = g_gap.phase;
    g_gap.idle_since = host_gap_now_ms();
    host_gap_profile(g_gap.open_phase, g_gap.why, true);
}

/* Exact pointer match first (string literals), else by content. */
static host_gap_bucket *host_gap_bucket_of(const char *phase, const char *why)
{
    unsigned i;

    for (i = 0; i < g_gap.buckets; i++)
        if ((g_gap.bucket[i].phase == phase || strcmp(g_gap.bucket[i].phase, phase) == 0) &&
            (g_gap.bucket[i].why == why || strcmp(g_gap.bucket[i].why, why) == 0))
            return &g_gap.bucket[i];
    if (g_gap.buckets == HOST_GAP_BUCKETS)
        return NULL;
    g_gap.bucket[g_gap.buckets].phase = phase;
    g_gap.bucket[g_gap.buckets].why = why;
    return &g_gap.bucket[g_gap.buckets++];
}

/* Books the open gap's time since `idle_since` and returns now. */
static double host_gap_book(void)
{
    host_gap_bucket *b = host_gap_bucket_of(g_gap.open_phase, g_gap.why);
    const double now = host_gap_now_ms();

    if (b) {
        b->count++;
        b->ms += now - g_gap.idle_since;
    }
    return now;
}

void pn_tt_host_gap_card_busy(void)
{
    if (!pn_tt_host_gap_enabled() || g_gap.idle_since == 0.0)
        return;
    (void)host_gap_book();
    g_gap.idle_since = 0.0;
    host_gap_profile(NULL, NULL, false);
}

void pn_tt_host_gap_mark(const char *phase, const char *why)
{
    if (!pn_tt_host_gap_enabled() || g_gap.idle_since == 0.0)
        return;
    g_gap.idle_since = host_gap_book();
    g_gap.open_phase = phase ? phase : "?";
    g_gap.why = why ? why : "?";
    host_gap_profile(g_gap.open_phase, g_gap.why, true);
}

void pn_tt_host_gap_split(const char *why)
{
    pn_tt_host_gap_mark(g_gap.phase, why);
}

void pn_tt_host_gap_report(const char *label, double divisor)
{
    unsigned i;
    double total = 0.0;
    unsigned long long n = 0u;

    if (!pn_tt_host_gap_enabled() || divisor <= 0.0)
        return;
    for (i = 0; i < g_gap.buckets; i++) {
        total += g_gap.bucket[i].ms;
        n += g_gap.bucket[i].count;
    }
    fprintf(stderr, "pn_tt: HOST GAP a %s: %.2f ms over %.1f gaps (card idle, host deciding) =",
            label, total / divisor, (double)n / divisor);
    for (i = 0; i < g_gap.buckets; i++)
        if (g_gap.bucket[i].ms / divisor >= 0.005)
            fprintf(stderr, " %s:%s %.2f (%.1f x %.1f us)", g_gap.bucket[i].phase,
                    g_gap.bucket[i].why, g_gap.bucket[i].ms / divisor,
                    (double)g_gap.bucket[i].count / divisor,
                    g_gap.bucket[i].count ? 1e3 * g_gap.bucket[i].ms /
                                                (double)g_gap.bucket[i].count : 0.0);
    fprintf(stderr, "\n");
    {
        /* Was the host thread switched out? (a gap with little CPU in it) */
        struct rusage use;

        #ifdef RUSAGE_THREAD
        if (getrusage(RUSAGE_THREAD, &use) == 0)
#else
        if (getrusage(RUSAGE_SELF, &use) == 0)
#endif
            fprintf(stderr, "pn_tt: HOST GAP context switches a %s: %.2f voluntary, %.2f "
                            "involuntary; %.2f major faults\n", label,
                    (double)use.ru_nvcsw / divisor, (double)use.ru_nivcsw / divisor,
                    (double)use.ru_majflt / divisor);
    }
}

void pn_tt_host_gap_red_delay(void)
{
    static long us = -1;
    double until;

    if (us < 0) {
        const char *env = getenv("PN_TT_HOST_GAP_RED_US");

        us = env ? atol(env) : 0;
        if (us > 0)
            fprintf(stderr, "pn_tt: HOST GAP RED armed — %ld us of host spin after each verify\n",
                    us);
    }
    if (us <= 0)
        return;
    until = host_gap_now_ms() + (double)us / 1e3;
    while (host_gap_now_ms() < until)
        ;
}
