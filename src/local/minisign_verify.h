/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   minisign_verify.h
 * @brief  Verify a minisign signature (Ed25519) over a buffer with one of a set
 *         of trusted public keys (spec §6, §11.2). Accepts the prehashed `ED`
 *         form (BLAKE2b-512, minisign's default) and the legacy `Ed` form, and
 *         checks the global signature over the trusted comment. Crypto is
 *         Monocypher (lib/monocypher).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_MINISIGN_VERIFY_H
#define PN_LOCAL_MINISIGN_VERIFY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_MINISIGN_KEY_ID_LEN 8
#define PN_MINISIGN_COMMENT_MAX 256

typedef struct {
    uint8_t key_id[PN_MINISIGN_KEY_ID_LEN];
    uint8_t pk[32];
} pn_minisign_pubkey;

/**
 * Parse a public key: the base64 line of a minisign.pub ("RWQ..."), or the
 * whole .pub file (the comment line is skipped).
 * @return 0, or -1 if malformed
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_minisign_parse_pubkey(const char *text, pn_minisign_pubkey *out);

/**
 * Verify `sig_text` (a whole .minisig file) over msg[len] with any of keys[n].
 *
 * @param trusted_comment  receives the trusted comment (may be NULL)
 * @param which_key        receives the index of the key that verified (may be NULL)
 * @param err              short reason on failure
 * @return 0 when valid, -1 otherwise
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_minisign_verify(const uint8_t *msg, size_t len, const char *sig_text,
                       const pn_minisign_pubkey *keys, int n_keys,
                       char trusted_comment[PN_MINISIGN_COMMENT_MAX], int *which_key,
                       char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MINISIGN_VERIFY_H */
