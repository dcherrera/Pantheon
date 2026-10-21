/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "bench.h"
#include <stdio.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

double pn_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

size_t pn_memory_usage(void) {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
        return (size_t)info.resident_size;
    return 0;
#elif defined(__linux__)
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0;
    if (fscanf(f, "%*ld %ld", &pages) != 1) pages = 0;
    fclose(f);
    return (size_t)pages * (size_t)sysconf(_SC_PAGESIZE);
#else
    return 0;
#endif
}

void pn_bench_report(const pn_bench_result *result) {
    if (!result) return;

    fprintf(stderr, "\n=== Benchmark ===\n");

    if (result->n_prefill > 0 && result->prefill_ms > 0) {
        double prefill_tps = (double)result->n_prefill / (result->prefill_ms / 1000.0);
        fprintf(stderr, "  prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
                result->n_prefill, result->prefill_ms, prefill_tps);
    }

    if (result->n_decode > 0 && result->decode_ms > 0) {
        double decode_tps = (double)result->n_decode / (result->decode_ms / 1000.0);
        fprintf(stderr, "  decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
                result->n_decode, result->decode_ms, decode_tps);
    }

    if (result->memory_bytes > 0) {
        double mb = (double)result->memory_bytes / (1024.0 * 1024.0);
        fprintf(stderr, "  memory:  %.1f MB RSS\n", mb);
    }

    fprintf(stderr, "=================\n");
}
