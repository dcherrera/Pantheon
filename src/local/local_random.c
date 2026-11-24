/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_random.c
 * @brief  Implementation of the CSPRNG helpers (see local_random.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_random.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

int pn_local_random_bytes(void *buf, size_t n)
{
    unsigned char *p = buf;
    size_t off = 0;
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);

    if (fd < 0)
        return -1;
    while (off < n) {
        ssize_t r = read(fd, p + off, n - off);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0) {
            close(fd);
            return -1;
        }
        off += (size_t)r;
    }
    close(fd);
    return 0;
}

int pn_local_random_below(uint32_t bound, uint32_t *out)
{
    /* Largest multiple of bound that fits in 32 bits: values above it are
     * rejected so every residue is equally likely. */
    uint32_t limit = bound ? (uint32_t)(0x100000000ull - (0x100000000ull % bound)) : 0;
    uint32_t v = 0;

    if (bound == 0)
        return -1;
    do {
        if (pn_local_random_bytes(&v, sizeof(v)) != 0)
            return -1;
    } while (limit != 0 && v >= limit);
    *out = v % bound;
    return 0;
}
