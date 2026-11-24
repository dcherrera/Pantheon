/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_host_admission.c
 * @brief Implementation of host-RAM admission control (see the header for the
 *        decision rule and the environment knobs).
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "pn_host_admission.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "pn_host_signals.h"

#define MIB_BYTES (1024ULL * 1024)
#define GIB_BYTES (1024ULL * 1024 * 1024)

static pthread_mutex_t g_admit_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_admitted_bytes = 0;
static uint64_t g_anon_baseline = 0;     /* anon RSS at the first admission */
static bool     g_have_baseline = false;
static uint64_t g_test_avail = 0;        /* unit-test hook (0 = off) */
static uint64_t g_test_anon = 0;
static _Thread_local char g_last_refusal[PN_HOST_ADMIT_REASON_LEN];
static _Thread_local uint64_t g_thread_admitted = 0;
static _Thread_local uint64_t g_thread_refusals = 0;

/* A positive integer from the environment, or 0. */
static uint64_t env_positive_u64(const char *name)
{
    const char *v = getenv(name);
    if (!v || !*v)
        return 0;
    char *end = NULL;
    long long parsed = strtoll(v, &end, 10);
    return (end != v && parsed > 0) ? (uint64_t)parsed : 0;
}

bool pn_host_admit_enabled(void)
{
    const char *v = getenv("PN_HOST_ADMIT");
    return !(v && v[0] == '0' && v[1] == '\0');
}

uint64_t pn_host_admit_reserve_bytes(void)
{
    uint64_t mb = env_positive_u64("PN_HOST_RESERVE_MB");
    return mb ? mb * MIB_BYTES : PN_HOST_ADMIT_RESERVE_DEFAULT_BYTES;
}

/* This process's anonymous resident bytes: RssAnon on Linux (file-backed
 * pages — the mmap'd weights — are excluded on purpose), the phys footprint
 * on macOS. 0 when unobservable. */
static uint64_t read_process_anon_bytes(void)
{
#if defined(__linux__)
    FILE *f = fopen("/proc/self/status", "re");
    if (!f)
        return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "RssAnon:", 8) == 0) {
            kb = strtoull(line + 8, NULL, 10);
            break;
        }
    }
    fclose(f);
    return kb * 1024ULL;
#elif defined(__APPLE__)
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS)
        return 0;
    return (uint64_t)info.phys_footprint;
#else
    return 0;
#endif
}

/* One reading of (MemAvailable, anon RSS). The env hook pins avail and zeroes
 * anon so accounting is the plain admitted sum. Caller holds the lock. */
static void read_memory_locked(uint64_t *avail, uint64_t *anon)
{
    if (g_test_avail) {
        *avail = g_test_avail;
        *anon = g_test_anon;
        return;
    }
    uint64_t env_mb = env_positive_u64("PN_HOST_ADMIT_TEST_AVAIL_MB");
    if (env_mb) {
        *avail = env_mb * MIB_BYTES;
        *anon = 0;
        return;
    }
    uint64_t total = 0;
    *avail = 0;
    if (!pn_read_proc_meminfo(&total, avail))
        pn_read_portable_memory(&total, avail);
    *anon = read_process_anon_bytes();
}

/* Admitted bytes not yet visible in MemAvailable: admitted minus the anon RSS
 * this process has grown by since its first admission. Caller holds the lock. */
static uint64_t outstanding_locked(uint64_t anon)
{
    if (!g_have_baseline) {
        g_anon_baseline = anon;
        g_have_baseline = true;
    }
    uint64_t grown = anon > g_anon_baseline ? anon - g_anon_baseline : 0;
    return g_admitted_bytes > grown ? g_admitted_bytes - grown : 0;
}

/* Effective available (MemAvailable less what is owed). Caller holds lock.
 * Returns false when MemAvailable is unobservable. */
static bool effective_avail_locked(uint64_t *eff_avail, uint64_t *owed)
{
    uint64_t avail = 0, anon = 0;
    read_memory_locked(&avail, &anon);
    *owed = outstanding_locked(anon);
    if (avail == 0)
        return false;
    *eff_avail = avail > *owed ? avail - *owed : 0;
    return true;
}

void pn_host_admit_format_bytes(uint64_t bytes, char *out, size_t out_len)
{
    if (bytes >= GIB_BYTES)
        snprintf(out, out_len, "%.1f GiB", (double)bytes / (double)GIB_BYTES);
    else
        snprintf(out, out_len, "%llu MiB",
                 (unsigned long long)((bytes + MIB_BYTES - 1) / MIB_BYTES));
}

static void format_refusal(char *reason, size_t reason_len, const char *what,
                           uint64_t need, uint64_t eff_avail, uint64_t reserve,
                           const char *hint)
{
    char s_need[32], s_avail[32], s_reserve[32];
    pn_host_admit_format_bytes(need, s_need, sizeof(s_need));
    pn_host_admit_format_bytes(eff_avail, s_avail, sizeof(s_avail));
    pn_host_admit_format_bytes(reserve, s_reserve, sizeof(s_reserve));
    snprintf(reason, reason_len, "%s needs %s, %s available (reserve %s); %s",
             what, s_need, s_avail, s_reserve, hint);
}

int pn_host_admit_hint(uint64_t bytes, const char *what, const char *hint,
                       char *reason, size_t reason_len)
{
    if (reason && reason_len)
        reason[0] = '\0';
    if (bytes == 0 || !pn_host_admit_enabled())
        return 0;

    uint64_t reserve = pn_host_admit_reserve_bytes();
    uint64_t eff_avail = 0, owed = 0;
    pthread_mutex_lock(&g_admit_lock);
    bool observed = effective_avail_locked(&eff_avail, &owed);
    bool fits = !observed || (eff_avail > reserve && bytes <= eff_avail - reserve);
    if (fits)
        g_admitted_bytes += bytes;
    pthread_mutex_unlock(&g_admit_lock);
    if (fits) {
        g_thread_admitted += bytes;
        return 0;
    }

    char msg[PN_HOST_ADMIT_REASON_LEN];
    format_refusal(msg, sizeof(msg), what ? what : "host allocation", bytes,
                   eff_avail, reserve, hint ? hint : PN_HOST_ADMIT_DEFAULT_HINT);
    fprintf(stderr, "[host_admit] REFUSED: %s\n", msg);
    snprintf(g_last_refusal, sizeof(g_last_refusal), "%s", msg);
    g_thread_refusals++;
    if (reason && reason_len)
        snprintf(reason, reason_len, "%s", msg);
    return -1;
}

int pn_host_admit(uint64_t bytes, const char *what, char *reason,
                  size_t reason_len)
{
    return pn_host_admit_hint(bytes, what, NULL, reason, reason_len);
}

uint64_t pn_host_admit_thread_refusals(void)
{
    return g_thread_refusals;
}

uint64_t pn_host_admit_thread_total(void)
{
    return g_thread_admitted;
}

const char *pn_host_admit_last_refusal(void)
{
    return g_last_refusal;
}

int pn_host_admit_check(uint64_t bytes, const char *what, const char *hint,
                        char *reason, size_t reason_len)
{
    if (pn_host_admit_hint(bytes, what, hint, reason, reason_len) != 0)
        return -1;
    if (pn_host_admit_enabled() && bytes) {
        pn_host_release(bytes);
        g_thread_admitted -= bytes;   /* nothing stays charged to a loader */
    }
    return 0;
}

int pn_host_admit_weight_copy(uint64_t bytes, const char *name)
{
    if (bytes < PN_HOST_ADMIT_THRESHOLD_BYTES)
        return 0;
    char what[160];
    snprintf(what, sizeof(what), "host copy of weight '%s'", name ? name : "?");
    return pn_host_admit_hint(bytes, what, "use a smaller quant/model or free "
                              "host memory", NULL, 0);
}

void pn_host_release(uint64_t bytes)
{
    pthread_mutex_lock(&g_admit_lock);
    g_admitted_bytes = g_admitted_bytes > bytes ? g_admitted_bytes - bytes : 0;
    pthread_mutex_unlock(&g_admit_lock);
}

uint64_t pn_host_admitted_bytes(void)
{
    pthread_mutex_lock(&g_admit_lock);
    uint64_t v = g_admitted_bytes;
    pthread_mutex_unlock(&g_admit_lock);
    return v;
}

uint64_t pn_host_admit_headroom(void)
{
    if (!pn_host_admit_enabled())
        return UINT64_MAX;
    uint64_t reserve = pn_host_admit_reserve_bytes();
    uint64_t eff_avail = 0, owed = 0;
    pthread_mutex_lock(&g_admit_lock);
    bool observed = effective_avail_locked(&eff_avail, &owed);
    pthread_mutex_unlock(&g_admit_lock);
    if (!observed)
        return UINT64_MAX;
    return eff_avail > reserve ? eff_avail - reserve : 0;
}

long pn_host_admit_fit_units(uint64_t fixed_bytes, uint64_t per_unit_bytes,
                             long want_units, long min_units)
{
    if (want_units <= 0)
        return 0;
    uint64_t room = pn_host_admit_headroom();
    if (fixed_bytes + (uint64_t)want_units * per_unit_bytes <= room)
        return want_units;
    if (fixed_bytes > room || per_unit_bytes == 0)
        return 0;
    uint64_t n = (room - fixed_bytes) / per_unit_bytes;
    if (n < (uint64_t)(min_units > 0 ? min_units : 1))
        return 0;
    return (long)n;
}

void pn_host_admit_test_set_memory(uint64_t avail_bytes, uint64_t anon_bytes)
{
    pthread_mutex_lock(&g_admit_lock);
    g_test_avail = avail_bytes;
    g_test_anon = anon_bytes;
    pthread_mutex_unlock(&g_admit_lock);
}

void pn_host_admit_test_reset(void)
{
    pthread_mutex_lock(&g_admit_lock);
    g_admitted_bytes = 0;
    g_anon_baseline = 0;
    g_have_baseline = false;
    g_test_avail = 0;
    g_test_anon = 0;
    pthread_mutex_unlock(&g_admit_lock);
}
