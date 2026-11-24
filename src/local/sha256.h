/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   sha256.h
 * @brief  Pure-C SHA-256 (FIPS 180-4), streaming and one-shot, HMAC-SHA256
 *         (RFC 2104), plus hex helpers. Ported from GiTeam app/src/util/hash.c
 *         (same author); scalar code, no SIMD, so it runs on a no-AVX host.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_SHA256_H
#define PN_LOCAL_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_SHA256_LEN     32
#define PN_SHA256_HEX_LEN 64

typedef struct {
    uint32_t state[8];
    uint64_t count;          /* bytes hashed so far */
    uint8_t  buffer[64];
} pn_sha256_ctx;

/** Start a hash. Thread-safe: yes (per ctx). Reentrant: yes. Signal-safe: yes */
void pn_sha256_init(pn_sha256_ctx *ctx);

/** Add bytes. Thread-safe: yes (per ctx). Reentrant: yes. Signal-safe: yes */
void pn_sha256_update(pn_sha256_ctx *ctx, const void *data, size_t len);

/** Finish into digest[32]. Thread-safe: yes (per ctx). Reentrant: yes. Signal-safe: yes */
void pn_sha256_final(pn_sha256_ctx *ctx, uint8_t digest[PN_SHA256_LEN]);

/** One-shot hash. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
void pn_sha256(const void *data, size_t len, uint8_t digest[PN_SHA256_LEN]);

/**
 * HMAC-SHA256 (RFC 2104) of `msg` under `key` into mac[32]. A key longer than
 * the 64-byte block is hashed first. The key material on the stack is wiped.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
void pn_hmac_sha256(const void *key, size_t klen, const void *msg, size_t mlen,
                    uint8_t mac[PN_SHA256_LEN]);

/**
 * Lowercase hex of `n` bytes into hex[2n+1].
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
void pn_hex_encode(const uint8_t *bytes, size_t n, char *hex);

/**
 * Parse exactly 2n hex chars (either case) into bytes[n].
 * @return 0, or -1 on a bad char / short string
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_hex_decode(const char *hex, uint8_t *bytes, size_t n);

/**
 * SHA-256 of a whole file, streamed in 1 MiB reads.
 * @return 0, or -1 when the file cannot be read
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_sha256_file(const char *path, uint8_t digest[PN_SHA256_LEN]);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_SHA256_H */
