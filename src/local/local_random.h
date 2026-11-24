/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_random.h
 * @brief  Cryptographic random bytes for keys and pairing codes (/dev/urandom).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_RANDOM_H
#define PN_LOCAL_RANDOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fill buf[n] from the kernel CSPRNG.
 * @return 0, or -1 (never returns weak bytes)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_random_bytes(void *buf, size_t n);

/**
 * Uniform integer in [0, bound) by rejection sampling.
 * @return 0, or -1 on an RNG failure
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_random_below(uint32_t bound, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_RANDOM_H */
