/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sample_profiler.c
 * @brief Implementation of the SIGPROF sampling profiler. See sample_profiler.h.
 *
 * @status Instrument only.
 * @issues Linux/x86-64 only.
 * @todo None
 */

#define _GNU_SOURCE
#include "sample_profiler.h"

#if defined(__linux__) && defined(__x86_64__)

#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <ucontext.h>
#include <execinfo.h>
#include <unistd.h>

/** Samples kept: a 60 s run at 1 kHz on two cores. */
#define PROF_MAX_SAMPLES (1u << 18)

/** Frames kept a sample: the interrupted IP, then up to PROF_CALLERS callers (glibc
 *  backtrace through the signal frame; .eh_frame, no frame pointers needed). */
#define PROF_CALLERS 4

typedef struct prof_sample {
    uintptr_t ip;
    uintptr_t caller[PROF_CALLERS];
    int32_t tid;
} prof_sample;

static prof_sample g_samples[PROF_MAX_SAMPLES];
static _Atomic uint32_t g_count;
static const char *g_path;
static uintptr_t g_base;
static _Atomic int g_window = -1; /**< -1 always record; 0 outside / 1 inside a marked window */

static void prof_on_signal(int sig, siginfo_t *info, void *context)
{
    const ucontext_t *uc = (const ucontext_t *)context;
    uint32_t slot;

    (void)sig;
    (void)info;
    if (atomic_load_explicit(&g_window, memory_order_relaxed) == 0)
        return;
    slot = atomic_fetch_add_explicit(&g_count, 1u, memory_order_relaxed);
    if (slot >= PROF_MAX_SAMPLES)
        return;
    g_samples[slot].ip = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
    g_samples[slot].tid = (int32_t)syscall(SYS_gettid);
    {
        void *frames[PROF_CALLERS + 4];
        const int n = backtrace(frames, PROF_CALLERS + 4);
        int i, at = -1, k;

        /* Find the interrupted IP in the walk; its successors are the callers. */
        for (i = 0; i < n; i++)
            if ((uintptr_t)frames[i] == g_samples[slot].ip) {
                at = i;
                break;
            }
        for (k = 0; k < PROF_CALLERS; k++)
            g_samples[slot].caller[k] =
                at >= 0 && at + 1 + k < n ? (uintptr_t)frames[at + 1 + k] - 1u : 0u;
    }
}

/** The executable's load base, from the first r-xp line naming it (PIE). */
static uintptr_t prof_image_base(void)
{
    char self[512], line[1024];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1u);
    FILE *maps;
    uintptr_t base = 0u;

    if (n <= 0)
        return 0u;
    self[n] = '\0';
    maps = fopen("/proc/self/maps", "r");
    if (!maps)
        return 0u;
    while (fgets(line, sizeof(line), maps))
        if (strstr(line, self)) {
            base = (uintptr_t)strtoull(line, NULL, 16);
            break;
        }
    fclose(maps);
    return base;
}

static int prof_compare(const void *a, const void *b)
{
    const prof_sample *x = a, *y = b;
    int k;

    if (x->tid != y->tid)
        return x->tid < y->tid ? -1 : 1;
    if (x->ip != y->ip)
        return x->ip < y->ip ? -1 : 1;
    for (k = 0; k < PROF_CALLERS; k++)
        if (x->caller[k] != y->caller[k])
            return x->caller[k] < y->caller[k] ? -1 : 1;
    return 0;
}

static uintptr_t prof_rel(uintptr_t address)
{
    return address >= g_base ? address - g_base : 0u;
}

static void prof_dump(void)
{
    struct itimerval off;
    uint32_t count = atomic_load(&g_count), i, run;
    FILE *out;

    memset(&off, 0, sizeof(off));
    setitimer(ITIMER_PROF, &off, NULL);
    if (count > PROF_MAX_SAMPLES)
        count = PROF_MAX_SAMPLES;
    out = fopen(g_path, "w");
    if (!out)
        return;
    qsort(g_samples, count, sizeof(g_samples[0]), prof_compare);
    fprintf(out, "# base 0x%lx samples %u (1 ms of process CPU each)\n", (unsigned long)g_base,
            count);
    for (i = 0; i < count; i = run) {
        for (run = i; run < count && prof_compare(&g_samples[run], &g_samples[i]) == 0; run++)
            ;
        fprintf(out, "0x%lx %d %u 0x%lx 0x%lx 0x%lx 0x%lx\n",
                (unsigned long)prof_rel(g_samples[i].ip), g_samples[i].tid, run - i,
                (unsigned long)prof_rel(g_samples[i].caller[0]),
                (unsigned long)prof_rel(g_samples[i].caller[1]),
                (unsigned long)prof_rel(g_samples[i].caller[2]),
                (unsigned long)prof_rel(g_samples[i].caller[3]));
    }
    fclose(out);
}

void pn_sample_profiler_start(void)
{
    static int started;
    struct sigaction action;
    struct itimerval every;

    g_path = getenv("PN_PROFILE");
    if (started || !g_path || !*g_path)
        return;
    started = 1;
    g_base = prof_image_base();
    {
        void *warm[4];

        (void)backtrace(warm, 4); /* loads libgcc's unwinder now, not inside a handler */
    }
    /* `PN_PROFILE_ALL=1` records the whole process; otherwise only marked windows. */
    atomic_store(&g_window, getenv("PN_PROFILE_ALL") ? -1 : 0);
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = prof_on_signal;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGPROF, &action, NULL);
    memset(&every, 0, sizeof(every));
    every.it_interval.tv_usec = 1000;
    every.it_value.tv_usec = 1000;
    /* Windowed: the timer runs only inside a window, so nothing else is ever interrupted
     * (round 1 sampled the whole process and its text differed — an EINTR somewhere). */
    if (atomic_load(&g_window) == -1)
        setitimer(ITIMER_PROF, &every, NULL);
    atexit(prof_dump);
    fprintf(stderr, "profile: sampling every 1 ms of CPU into %s\n", g_path);
}

void pn_sample_profiler_window(int inside)
{
    struct itimerval every;

    if (!g_path || atomic_load(&g_window) == -1)
        return;
    memset(&every, 0, sizeof(every));
    if (inside) {
        every.it_interval.tv_usec = 1000;
        every.it_value.tv_usec = 1000;
    }
    atomic_store(&g_window, inside ? 1 : 0);
    setitimer(ITIMER_PROF, &every, NULL);
}

#else

void pn_sample_profiler_start(void)
{
}

void pn_sample_profiler_window(int inside)
{
    (void)inside;
}

#endif
