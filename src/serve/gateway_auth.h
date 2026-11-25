/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_auth.h
 * @brief The shared bearer key a trusted gateway presents to `pantheon serve
 *        --gateway`: read once from a key file, compared in constant time.
 *
 * The key never leaves this module: it is not logged, not echoed in errors and
 * not exposed through any getter.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_GATEWAY_AUTH_H
#define PANTHEON_SERVE_GATEWAY_AUTH_H

#include <stdbool.h>
#include <stddef.h>

/** Longest key accepted, in bytes. */
#define PN_GATEWAY_KEY_MAX 256

/** A loaded key. Zero-initialise; pn_gateway_key_clear wipes it. */
typedef struct {
    char   key[PN_GATEWAY_KEY_MAX + 1];
    size_t len;          /* 0 = no key loaded */
} pn_gateway_key;

/**
 * Parse a key from the bytes of a key file: surrounding whitespace is trimmed;
 * an empty key, one with inner whitespace or control bytes, or one longer than
 * PN_GATEWAY_KEY_MAX is refused.
 * @return 0, or -1 with a reason in err (never containing key bytes)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_gateway_key_parse(pn_gateway_key *out, const char *text, size_t n,
                         char *err, size_t errlen);

/**
 * Read and parse the key file at `path` (at most 4 KiB is read). Warns on
 * stderr when the file is readable by group or others.
 * @return 0, or -1 with a reason in err
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_gateway_key_load(pn_gateway_key *out, const char *path, char *err,
                        size_t errlen);

/**
 * Does an `Authorization` header value carry this key ("Bearer <key>", the
 * scheme case-insensitive)? The comparison takes the same time for every
 * presented key of a given length, so timing does not reveal a prefix match.
 * A NULL/empty header or an unloaded key never matches.
 *
 * Thread-safe: yes (read-only). Reentrant: yes. Signal-safe: no
 */
bool pn_gateway_key_matches(const pn_gateway_key *key, const char *authorization);

/** Wipe the key bytes. NULL-safe. */
void pn_gateway_key_clear(pn_gateway_key *key);

#endif /* PANTHEON_SERVE_GATEWAY_AUTH_H */
