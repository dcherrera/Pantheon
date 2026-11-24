/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   nvml_device_mem.c
 * @brief  Implementation of the NVML device memory read (see
 *         nvml_device_mem.h). The NVML and CUDA driver entry points are looked
 *         up with dlsym; only their stable C ABI is declared here.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "nvml_device_mem.h"

#include <string.h>

#ifdef USE_CUDA

#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

typedef void *nvml_device;
typedef struct { unsigned long long total, free, used; } nvml_memory;
/* nvmlProcessInfo_v1_t: what the unversioned nvmlDeviceGetComputeRunningProcesses
 * fills in every driver that has it. */
typedef struct { unsigned int pid; unsigned long long used_gpu_memory; } nvml_process_v1;

#define NVML_PCI_BUS_ID_LEN  32
#define NVML_MAX_PROCESSES   64
#define NVML_VALUE_NA        0xFFFFFFFFFFFFFFFFULL

static struct {
    pthread_mutex_t lock;
    bool            tried, ok;
    int  (*init)(void);
    int  (*by_pci)(const char *, nvml_device *);
    int  (*mem_info)(nvml_device, nvml_memory *);
    int  (*processes)(nvml_device, unsigned int *, nvml_process_v1 *);
    int  (*cu_init)(unsigned int);
    int  (*cu_device_get)(int *, int);
    int  (*cu_pci_bus_id)(char *, int, int);
    nvml_device devices[PN_HW_MAX_UNITS];
} g = { .lock = PTHREAD_MUTEX_INITIALIZER };

/* dlopen libnvidia-ml and find libcuda's PCI lookup. g.lock held. */
static void load_nvml(void)
{
    void *nvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);

    g.tried = true;
    if (!nvml || !cuda)
        return;
    *(void **)&g.init = dlsym(nvml, "nvmlInit_v2");
    *(void **)&g.by_pci = dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
    *(void **)&g.mem_info = dlsym(nvml, "nvmlDeviceGetMemoryInfo");
    *(void **)&g.processes = dlsym(nvml, "nvmlDeviceGetComputeRunningProcesses");
    *(void **)&g.cu_init = dlsym(cuda, "cuInit");
    *(void **)&g.cu_device_get = dlsym(cuda, "cuDeviceGet");
    *(void **)&g.cu_pci_bus_id = dlsym(cuda, "cuDeviceGetPCIBusId");
    g.ok = g.init && g.by_pci && g.mem_info && g.cu_init && g.cu_device_get &&
           g.cu_pci_bus_id && g.init() == 0 && g.cu_init(0) == 0;
}

/* The NVML device of CUDA ordinal `die`, looked up once. g.lock held. */
static nvml_device device_of(int die)
{
    char bus[NVML_PCI_BUS_ID_LEN];
    int dev = 0;

    if (die < 0 || die >= PN_HW_MAX_UNITS)
        return NULL;
    if (!g.devices[die] && g.cu_device_get(&dev, die) == 0 &&
        g.cu_pci_bus_id(bus, (int)sizeof(bus), dev) == 0)
        g.by_pci(bus, &g.devices[die]);
    return g.devices[die];
}

/* This process's bytes on `d`, or -1 when NVML does not say. g.lock held. */
static int own_bytes(nvml_device d, uint64_t *out)
{
    nvml_process_v1 procs[NVML_MAX_PROCESSES];
    unsigned int n = NVML_MAX_PROCESSES;
    unsigned int me = (unsigned int)getpid();

    *out = 0;
    if (!g.processes || g.processes(d, &n, procs) != 0)
        return -1;
    for (unsigned int i = 0; i < n && i < NVML_MAX_PROCESSES; i++) {
        if (procs[i].pid != me)
            continue;
        if (procs[i].used_gpu_memory == NVML_VALUE_NA)
            return -1;
        *out += procs[i].used_gpu_memory;
    }
    return 0;
}

int pn_nvml_available(void)
{
    pthread_mutex_lock(&g.lock);
    if (!g.tried)
        load_nvml();
    int ok = g.ok;
    pthread_mutex_unlock(&g.lock);
    return ok;
}

int pn_nvml_read_die(int die, pn_hw_dev_mem *out)
{
    nvml_memory mem;
    int rc = -1;

    memset(out, 0, sizeof(*out));
    pthread_mutex_lock(&g.lock);
    if (!g.tried)
        load_nvml();
    nvml_device d = g.ok ? device_of(die) : NULL;
    if (d && g.mem_info(d, &mem) == 0 && mem.total > 0) {
        out->total = mem.total;
        out->used = mem.used;
        out->self_known = own_bytes(d, &out->used_self) == 0;
        rc = 0;
    }
    pthread_mutex_unlock(&g.lock);
    return rc;
}

#else /* !USE_CUDA */

int pn_nvml_read_die(int die, pn_hw_dev_mem *out)
{
    (void)die;
    memset(out, 0, sizeof(*out));
    return -1;
}

int pn_nvml_available(void)
{
    return 0;
}

#endif
