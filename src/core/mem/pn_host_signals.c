/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_host_signals.c
 * @brief Implementation of the cheap host memory-pressure readers
 *        (MemoryAware.md §2.1). All parsing is defensive: a missing or
 *        malformed source leaves its output field 0 ("not observed") rather
 *        than failing the whole sample.
 *
 * @status Phase 1 — read-only.
 * @issues None
 * @todo None
 */

#include "pn_host_signals.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

/* Read a whole small pseudo-file into buf (NUL-terminated). Returns the byte
 * count, or -1 if the file could not be opened. Pseudo-files under /proc and
 * the cgroup fs have no stable size, so we read until EOF into a fixed buffer. */
static long read_small_file(const char *path, char *buf, size_t buf_len)
{
    FILE *f = fopen(path, "re");
    if (!f)
        return -1;

    size_t n = fread(buf, 1, buf_len - 1, f);
    fclose(f);
    buf[n] = '\0';
    return (long)n;
}

/* Find "key" at the start of a line in a /proc/meminfo-style buffer and return
 * its value in kB (meminfo reports kB). Returns true if found. */
static bool meminfo_field_kb(const char *buf, const char *key, uint64_t *out_kb)
{
    size_t key_len = strlen(key);
    const char *p = buf;

    while (p && *p) {
        if (strncmp(p, key, key_len) == 0 && p[key_len] == ':') {
            const char *v = p + key_len + 1;
            while (*v == ' ' || *v == '\t')
                v++;
            *out_kb = strtoull(v, NULL, 10);
            return true;
        }
        p = strchr(p, '\n');
        if (p)
            p++;
    }
    return false;
}

bool pn_read_proc_meminfo(uint64_t *total_bytes, uint64_t *avail_bytes)
{
    char buf[4096];
    if (read_small_file("/proc/meminfo", buf, sizeof(buf)) < 0)
        return false;

    uint64_t total_kb = 0, avail_kb = 0;
    bool have_total = meminfo_field_kb(buf, "MemTotal", &total_kb);
    bool have_avail = meminfo_field_kb(buf, "MemAvailable", &avail_kb);

    /* Older kernels lack MemAvailable — approximate with MemFree + Cached so
     * the number is never wildly pessimistic. */
    if (!have_avail) {
        uint64_t free_kb = 0, cached_kb = 0;
        meminfo_field_kb(buf, "MemFree", &free_kb);
        meminfo_field_kb(buf, "Cached", &cached_kb);
        avail_kb = free_kb + cached_kb;
        have_avail = (free_kb != 0 || cached_kb != 0);
    }

    if (!have_total)
        return false;

    *total_bytes = total_kb * 1024ULL;
    *avail_bytes = avail_kb * 1024ULL;
    return have_avail;
}

void pn_read_portable_memory(uint64_t *total_bytes, uint64_t *avail_bytes)
{
    *total_bytes = 0;
    *avail_bytes = 0;

#if defined(__APPLE__)
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0 && memsize > 0)
        *total_bytes = (uint64_t)memsize;

    /* Available ~= (free + inactive + speculative) pages: pages the VM can hand
     * out without swapping. Mirrors what MemAvailable estimates on Linux. */
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS)
        page_size = 4096;

    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vmstat, &count) == KERN_SUCCESS) {
        uint64_t avail_pages = (uint64_t)vmstat.free_count +
                               (uint64_t)vmstat.inactive_count +
                               (uint64_t)vmstat.speculative_count;
        *avail_bytes = avail_pages * (uint64_t)page_size;
    }
#endif
}

bool pn_read_proc_pressure(double *some_avg10, double *full_avg10)
{
    *some_avg10 = 0.0;
    *full_avg10 = 0.0;

    char buf[512];
    if (read_small_file("/proc/pressure/memory", buf, sizeof(buf)) < 0)
        return false;

    /* Lines look like: "some avg10=1.23 avg60=... total=...". Scan for the two
     * prefixes and pull the avg10 field. */
    const char *some = strstr(buf, "some ");
    const char *full = strstr(buf, "full ");
    bool any = false;

    if (some) {
        const char *a = strstr(some, "avg10=");
        if (a) { *some_avg10 = strtod(a + 6, NULL); any = true; }
    }
    if (full) {
        const char *a = strstr(full, "avg10=");
        if (a) { *full_avg10 = strtod(a + 6, NULL); any = true; }
    }
    return any;
}

/* Resolve this process's cgroup-v2 memory dir from /proc/self/cgroup. On pure
 * cgroup-v2 the file has a single "0::<path>" line; <path> is relative to
 * /sys/fs/cgroup. Writes the joined dir into out. Returns true on success. */
static bool cgroup_v2_dir(char *out, size_t out_len)
{
    char buf[512];
    if (read_small_file("/proc/self/cgroup", buf, sizeof(buf)) < 0)
        return false;

    const char *marker = strstr(buf, "0::");
    if (!marker)
        return false;
    const char *rel = marker + 3;

    /* Trim trailing newline. */
    char rel_path[256];
    size_t i = 0;
    while (rel[i] && rel[i] != '\n' && i + 1 < sizeof(rel_path)) {
        rel_path[i] = rel[i];
        i++;
    }
    rel_path[i] = '\0';

    int n = snprintf(out, out_len, "/sys/fs/cgroup%s", rel_path);
    return n > 0 && (size_t)n < out_len;
}

/* Read one cgroup memory file as a u64; a "max" literal maps to 0 ("unset"). */
static uint64_t read_cgroup_u64(const char *dir, const char *leaf)
{
    char path[512];
    if (snprintf(path, sizeof(path), "%s/%s", dir, leaf) >= (int)sizeof(path))
        return 0;

    char buf[64];
    if (read_small_file(path, buf, sizeof(buf)) < 0)
        return 0;
    if (strncmp(buf, "max", 3) == 0)
        return 0;
    return strtoull(buf, NULL, 10);
}

bool pn_read_cgroup_v2_memory(uint64_t *current, uint64_t *high, uint64_t *max)
{
    *current = 0;
    *high = 0;
    *max = 0;

    char dir[512];
    if (!cgroup_v2_dir(dir, sizeof(dir)))
        return false;

    /* memory.current is the tell that a v2 memory controller is present. */
    char probe[576];
    if (snprintf(probe, sizeof(probe), "%s/memory.current", dir) >= (int)sizeof(probe))
        return false;
    char tmp[64];
    if (read_small_file(probe, tmp, sizeof(tmp)) < 0)
        return false;

    *current = strtoull(tmp, NULL, 10);
    *high = read_cgroup_u64(dir, "memory.high");
    *max = read_cgroup_u64(dir, "memory.max");
    return true;
}

void pn_host_signals_sample(pn_host_sample *out)
{
    memset(out, 0, sizeof(*out));

    if (!pn_read_proc_meminfo(&out->total_bytes, &out->avail_bytes))
        pn_read_portable_memory(&out->total_bytes, &out->avail_bytes);

    pn_read_proc_pressure(&out->psi_some_avg10, &out->psi_full_avg10);
    pn_read_cgroup_v2_memory(&out->cgroup_current, &out->cgroup_high,
                             &out->cgroup_max);
}
