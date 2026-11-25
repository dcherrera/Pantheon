/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_id.c
 * @brief Implementation of request_id.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "request_id.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#define PN_HAVE_ARC4RANDOM 1
#elif defined(__linux__)
#include <sys/random.h>
#endif

#ifndef PN_HAVE_ARC4RANDOM
/* /dev/urandom, the source when the platform call is missing or failed. */
static int read_urandom(void *buf, size_t n)
{
    int fd = open("/dev/urandom", O_RDONLY);
    size_t got = 0;

    if (fd < 0)
        return -1;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r <= 0)
            break;
        got += (size_t)r;
    }
    close(fd);
    return got == n ? 0 : -1;
}
#endif

int pn_os_random_bytes(void *buf, size_t n)
{
#if defined(PN_HAVE_ARC4RANDOM)
    arc4random_buf(buf, n);
    return 0;
#elif defined(__linux__)
    size_t got = 0;
    while (got < n) {
        ssize_t r = getrandom((char *)buf + got, n - got, 0);
        if (r <= 0)
            break;
        got += (size_t)r;
    }
    return got == n ? 0 : read_urandom(buf, n);
#else
    return read_urandom(buf, n);
#endif
}

/* Last-resort bytes when no entropy source answered: clock and address mixing
 * through splitmix64. Never expected on a supported host; it keeps ids from
 * collapsing to one value rather than claiming to be strong. */
static void fallback_bytes(unsigned char *buf, size_t n)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    static uint64_t counter;
    uint64_t x = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^
                 (uint64_t)(uintptr_t)buf ^
                 __atomic_add_fetch(&counter, 0x9E3779B97F4A7C15ull, __ATOMIC_RELAXED);
    for (size_t i = 0; i < n; i++) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        buf[i] = (unsigned char)(z ^ (z >> 31));
    }
}

void pn_request_id_new(const char *prefix, char *buf, size_t cap)
{
    static const char base62[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    char tail[PN_REQUEST_ID_RANDOM_CHARS + 1];
    int filled = 0;

    if (!buf || cap == 0)
        return;
    /* Rejection sampling keeps every character uniform: a byte >= 248 (the
     * largest multiple of 62 that fits) would bias the low characters. */
    while (filled < PN_REQUEST_ID_RANDOM_CHARS) {
        unsigned char raw[64];
        if (pn_os_random_bytes(raw, sizeof(raw)) != 0)
            fallback_bytes(raw, sizeof(raw));
        for (size_t i = 0; i < sizeof(raw) && filled < PN_REQUEST_ID_RANDOM_CHARS; i++)
            if (raw[i] < 248)
                tail[filled++] = base62[raw[i] % 62];
    }
    tail[filled] = '\0';
    snprintf(buf, cap, "%s%s", prefix ? prefix : "", tail);
}

uint64_t pn_os_random_seed(void)
{
    uint64_t seed = 0;

    if (pn_os_random_bytes(&seed, sizeof(seed)) != 0)
        fallback_bytes((unsigned char *)&seed, sizeof(seed));
    return seed;
}
