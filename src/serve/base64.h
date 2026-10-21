/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file base64.h
 * @brief Standard Base64 (RFC 4648, '+' '/' alphabet, '=' padding) encode /
 *        decode over byte buffers. Used by the session warm-tier (K80
 *        serving-ops Phase 7) to carry a binary session snapshot blob inside a
 *        JSON body: /admin/session/export returns base64, /session/restore
 *        accepts it. Kept a tiny standalone module (no deps) so both the serve
 *        and any test can link it.
 *
 * @status Complete; unit-tested (tests/test_base64.c).
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_BASE64_H
#define PN_SERVE_BASE64_H

#include <stddef.h>

/**
 * Base64-encode `len` bytes of `src`. Returns a malloc'd NUL-terminated string
 * (caller frees), or NULL on allocation failure. A zero-length input encodes to
 * an empty string.
 */
char *pn_base64_encode(const unsigned char *src, size_t len);

/**
 * Base64-decode NUL-terminated `src` into a freshly malloc'd byte buffer.
 * Rejects any non-alphabet character (whitespace included) and malformed
 * padding — a corrupt/tampered blob is a clean failure, not silent truncation.
 *
 * @param src      NUL-terminated base64 text.
 * @param out_len  Out: decoded byte count.
 * @return         malloc'd bytes (caller frees), or NULL on invalid input /
 *                 allocation failure. A zero-length input decodes to a 1-byte
 *                 allocation with *out_len == 0.
 */
unsigned char *pn_base64_decode(const char *src, size_t *out_len);

#endif /* PN_SERVE_BASE64_H */
