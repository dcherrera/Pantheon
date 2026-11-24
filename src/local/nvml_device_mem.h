/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   nvml_device_mem.h
 * @brief  NVIDIA device memory without a CUDA context: NVML (libnvidia-ml,
 *         shipped with the driver), loaded at run time. One read is a few
 *         driver calls (no context, nothing on the device), so the background
 *         poll can read every K80 die every few seconds beside running models.
 *         A CUDA ordinal is matched to its NVML device by PCI bus id
 *         (CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER renumber CUDA, not NVML).
 *         Our own share is this process's entry in NVML's compute process list.
 *         Builds without USE_CUDA, or hosts without libnvidia-ml, report
 *         "unavailable" and the caller falls back.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_NVML_DEVICE_MEM_H
#define PN_LOCAL_NVML_DEVICE_MEM_H

#include "hw_report.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Read CUDA ordinal `die`'s memory: total, used by anyone, and (when NVML
 * lists this process on it) our own bytes (out->self_known).
 * @return 0, or -1 when NVML is unavailable or the die cannot be matched
 *
 * Thread-safe: yes (serialized). Reentrant: no. Signal-safe: no
 */
int pn_nvml_read_die(int die, pn_hw_dev_mem *out);

/**
 * Whether NVML loaded (loads it on first use).
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int pn_nvml_available(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_NVML_DEVICE_MEM_H */
