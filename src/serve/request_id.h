/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_id.h
 * @brief Response ids that never repeat: completion ids (`chatcmpl-…`) and
 *        tool-call ids (`call_…`), each a prefix plus 24 base62 characters
 *        drawn from the operating system's entropy source.
 *
 * A gateway that bills per request joins its records on these ids, so two
 * requests must never share one, across restarts and across replicas. The ids
 * used to be 8 hex characters from the unseeded process rand(), which replays
 * the same sequence after every start.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_REQUEST_ID_H
#define PANTHEON_SERVE_REQUEST_ID_H

#include <stddef.h>
#include <stdint.h>

/** Random characters after the prefix (24 base62 ≈ 143 bits). */
#define PN_REQUEST_ID_RANDOM_CHARS 24

/**
 * Fill `buf` with `n` bytes from the OS entropy source (arc4random_buf on
 * macOS/BSD, getrandom on Linux, /dev/urandom as the fallback).
 * @return 0, or -1 when no source answered (buf is then unspecified)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_os_random_bytes(void *buf, size_t n);

/**
 * Write `<prefix><24 base62 characters>` into buf (NUL-terminated). With too
 * small a buffer the id is truncated, still NUL-terminated.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_request_id_new(const char *prefix, char *buf, size_t cap);

/** A 64-bit seed from the OS entropy source (for seeding process RNGs). */
uint64_t pn_os_random_seed(void);

#endif /* PANTHEON_SERVE_REQUEST_ID_H */
