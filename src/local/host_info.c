/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   host_info.c
 * @brief  Implementation of the host facts (see host_info.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "host_info.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include "../accel/backend_identity.h"

#ifdef __linux__
/* Value of `key` in a "Key: value" /proc file, as a string. */
static int proc_field(const char *path, const char *key, char *out, size_t cap)
{
    FILE *f = fopen(path, "r");
    char line[512];
    size_t klen = strlen(key);
    int found = -1;

    if (!f)
        return -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, klen) != 0)
            continue;
        const char *v = strchr(line + klen, ':');
        if (!v)
            continue;
        v++;
        while (*v == ' ' || *v == '\t')
            v++;
        snprintf(out, cap, "%s", v);
        out[strcspn(out, "\n")] = '\0';
        found = 0;
        break;
    }
    fclose(f);
    return found;
}

static void collect_platform(pn_host_info *h)
{
    char buf[256];

    if (proc_field("/proc/cpuinfo", "model name", buf, sizeof(buf)) == 0)
        snprintf(h->cpu_model, sizeof(h->cpu_model), "%s", buf);
    if (proc_field("/proc/meminfo", "MemTotal", buf, sizeof(buf)) == 0)
        h->ram_total = strtoull(buf, NULL, 10) * 1024ull;
    if (proc_field("/proc/meminfo", "MemAvailable", buf, sizeof(buf)) == 0)
        h->ram_free = strtoull(buf, NULL, 10) * 1024ull;
}
#elif defined(__APPLE__)
static void collect_platform(pn_host_info *h)
{
    size_t len = sizeof(h->cpu_model);
    uint64_t mem = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (sysctlbyname("machdep.cpu.brand_string", h->cpu_model, &len, NULL, 0) != 0)
        h->cpu_model[0] = '\0';
    len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0)
        h->ram_total = mem;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) ==
        KERN_SUCCESS)
        h->ram_free = ((uint64_t)vm.free_count + vm.inactive_count) * (uint64_t)vm_page_size;
}
#else
static void collect_platform(pn_host_info *h)
{
    (void)h;
}
#endif

static void collect_isa(pn_host_info *h)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    h->isa_x86 = true;
    h->avx = __builtin_cpu_supports("avx");
    h->avx2 = __builtin_cpu_supports("avx2");
    h->sse4_2 = __builtin_cpu_supports("sse4.2");
#else
    (void)h;
#endif
}

int pn_host_info_collect(const char *store_dir, time_t started, pn_host_info *out)
{
    struct utsname u;
    struct statvfs vfs;
    long cores = sysconf(_SC_NPROCESSORS_ONLN);

    memset(out, 0, sizeof(*out));
    if (gethostname(out->hostname, sizeof(out->hostname) - 1) != 0)
        out->hostname[0] = '\0';
    out->cpu_cores = cores > 0 ? (int)cores : 0;
    collect_platform(out);
    collect_isa(out);
    if (uname(&u) == 0) {
        snprintf(out->arch, sizeof(out->arch), "%s", u.machine);
        snprintf(out->os, sizeof(out->os), "%s %s", u.sysname, u.release);
    }
    if (store_dir && statvfs(store_dir, &vfs) == 0)
        out->disk_free = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
    snprintf(out->build, sizeof(out->build), "%s", pn_backend_build_commit());
    snprintf(out->backend, sizeof(out->backend), "%s", pn_backend_token());
    out->uptime_s = started > 0 ? (long)(time(NULL) - started) : 0;
    return 0;
}

void pn_host_info_write_json(json_writer *jw, const pn_host_info *h)
{
    jw_raw(jw, "\"host\":{\"hostname\":");
    jw_string(jw, h->hostname);
    jw_raw(jw, ",\"cpu\":{\"model\":");
    jw_string(jw, h->cpu_model);
    jw_printf(jw, ",\"cores\":%d},\"ram\":{\"total\":%llu,\"free\":%llu},\"disk_free\":%llu",
              h->cpu_cores, (unsigned long long)h->ram_total,
              (unsigned long long)h->ram_free, (unsigned long long)h->disk_free);
    jw_raw(jw, ",\"arch\":");
    jw_string(jw, h->arch);
    jw_raw(jw, ",\"os\":");
    jw_string(jw, h->os);
    if (h->isa_x86)
        jw_printf(jw, ",\"isa\":{\"avx\":%s,\"avx2\":%s,\"sse4_2\":%s}",
                  h->avx ? "true" : "false", h->avx2 ? "true" : "false",
                  h->sse4_2 ? "true" : "false");
    else
        jw_raw(jw, ",\"isa\":{}");
    jw_raw(jw, ",\"pantheon\":{\"build\":");
    jw_string(jw, h->build);
    jw_raw(jw, ",\"backend\":");
    jw_string(jw, h->backend);
    jw_printf(jw, "},\"uptime_s\":%ld}", h->uptime_s);
}
