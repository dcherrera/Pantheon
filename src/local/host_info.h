/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   host_info.h
 * @brief  Host facts for the hardware report (spec §4.2): CPU model and cores,
 *         RAM total/free, disk free on the model store, ISA flags (no-AVX hosts
 *         exist), OS, Pantheon build and backend, uptime. Linux reads /proc;
 *         macOS uses sysctl. Detection only — nothing here executes AVX.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_HOST_INFO_H
#define PN_LOCAL_HOST_INFO_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "../serve/json_writer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     hostname[128];
    char     cpu_model[128];
    int      cpu_cores;          /* online logical CPUs */
    uint64_t ram_total;
    uint64_t ram_free;           /* available (Linux MemAvailable) */
    uint64_t disk_free;          /* bytes free on the store's filesystem */
    char     arch[32];           /* uname machine */
    char     os[128];            /* "Linux 6.8.12-k80strap" */
    bool     isa_x86;            /* the flags below are meaningful */
    bool     avx, avx2, sse4_2;
    char     build[64];          /* Pantheon build commit */
    char     backend[16];        /* cpu / cuda / tt / metal */
    long     uptime_s;           /* since `pantheon local` started */
} pn_host_info;

/**
 * Collect the host facts.
 * @param store_dir  model store (for disk free); may be NULL
 * @param started    process start time (uptime)
 * @return 0 (missing facts are left empty / 0)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_host_info_collect(const char *store_dir, time_t started, pn_host_info *out);

/**
 * Append `"host":{...}` (no surrounding braces) to jw.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_host_info_write_json(json_writer *jw, const pn_host_info *h);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HOST_INFO_H */
