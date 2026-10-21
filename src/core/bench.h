/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_BENCH_H
#define PN_BENCH_H

#include <stdint.h>
#include <stddef.h>

/*
 * Benchmark timing for Pantheon inference.
 * Reports tok/s for prefill and decode phases.
 */

typedef struct {
    double prefill_ms;
    double decode_ms;
    int    n_prefill;
    int    n_decode;
    size_t memory_bytes;
} pn_bench_result;

/* Get current time in milliseconds (monotonic clock) */
double pn_time_ms(void);

/* Print benchmark results to stderr */
void pn_bench_report(const pn_bench_result *result);

/* Get approximate RSS memory usage in bytes */
size_t pn_memory_usage(void);

#endif /* PN_BENCH_H */
