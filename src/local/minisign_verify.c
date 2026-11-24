/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   minisign_verify.c
 * @brief  Implementation of minisign verification (see minisign_verify.h).
 *         Format: https://jedisct1.github.io/minisign/ — a .minisig is
 *           untrusted comment: ...
 *           base64( alg[2] "ED"|"Ed" ‖ key_id[8] ‖ signature[64] )
 *           trusted comment: ...
 *           base64( global_signature[64] over signature ‖ trusted comment )
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "minisign_verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../lib/monocypher/monocypher-ed25519.h"
#include "../../lib/monocypher/monocypher.h"
#include "../serve/base64.h"

#define SIG_BLOB_LEN 74   /* alg 2 + key id 8 + signature 64 */
#define PK_BLOB_LEN  42   /* alg 2 + key id 8 + public key 32 */

/* Copy line `index` (0-based) of text into out, without the newline. */
static int get_line(const char *text, int index, char *out, size_t cap)
{
    const char *p = text;

    for (int i = 0; i < index; i++) {
        p = strchr(p, '\n');
        if (!p)
            return -1;
        p++;
    }
    size_t n = strcspn(p, "\r\n");
    if (n >= cap)
        return -1;
    memcpy(out, p, n);
    out[n] = '\0';
    return 0;
}

/* Base64-decode exactly `want` bytes into out. */
static int decode_exact(const char *b64, uint8_t *out, size_t want)
{
    size_t n = 0;
    unsigned char *raw = pn_base64_decode(b64, &n);
    int rc = (raw && n == want) ? 0 : -1;

    if (rc == 0)
        memcpy(out, raw, want);
    free(raw);
    return rc;
}

int pn_minisign_parse_pubkey(const char *text, pn_minisign_pubkey *out)
{
    char line[256];
    uint8_t blob[PK_BLOB_LEN];
    int idx = strncmp(text, "untrusted comment:", 18) == 0 ? 1 : 0;

    if (get_line(text, idx, line, sizeof(line)) != 0 || decode_exact(line, blob, sizeof(blob)) != 0)
        return -1;
    if (blob[0] != 'E' || blob[1] != 'd')
        return -1;
    memcpy(out->key_id, blob + 2, PN_MINISIGN_KEY_ID_LEN);
    memcpy(out->pk, blob + 10, 32);
    return 0;
}

static int fail(char *err, size_t errlen, const char *why)
{
    snprintf(err, errlen, "%s", why);
    return -1;
}

int pn_minisign_verify(const uint8_t *msg, size_t len, const char *sig_text,
                       const pn_minisign_pubkey *keys, int n_keys,
                       char trusted_comment[PN_MINISIGN_COMMENT_MAX], int *which_key,
                       char *err, size_t errlen)
{
    char line[512], comment[PN_MINISIGN_COMMENT_MAX];
    uint8_t blob[SIG_BLOB_LEN], global[64], prehash[64];
    const pn_minisign_pubkey *key = NULL;
    int ki = -1;

    if (!sig_text || strncmp(sig_text, "untrusted comment:", 18) != 0)
        return fail(err, errlen, "no signature");
    if (get_line(sig_text, 1, line, sizeof(line)) != 0 || decode_exact(line, blob, sizeof(blob)) != 0)
        return fail(err, errlen, "malformed signature line");
    if (get_line(sig_text, 2, line, sizeof(line)) != 0 ||
        strncmp(line, "trusted comment: ", 17) != 0 ||
        snprintf(comment, sizeof(comment), "%s", line + 17) >= (int)sizeof(comment))
        return fail(err, errlen, "malformed trusted comment");
    if (get_line(sig_text, 3, line, sizeof(line)) != 0 || decode_exact(line, global, 64) != 0)
        return fail(err, errlen, "malformed global signature");
    for (int i = 0; i < n_keys && !key; i++)
        if (memcmp(keys[i].key_id, blob + 2, PN_MINISIGN_KEY_ID_LEN) == 0)
            key = &keys[ki = i];
    if (!key)
        return fail(err, errlen, "signed with a key this build does not trust");

    const uint8_t *sig = blob + 10;
    int ok = -1;
    if (blob[0] == 'E' && blob[1] == 'D') {
        crypto_blake2b(prehash, sizeof(prehash), msg, len);
        ok = crypto_ed25519_check(sig, key->pk, prehash, sizeof(prehash));
    } else if (blob[0] == 'E' && blob[1] == 'd') {
        ok = crypto_ed25519_check(sig, key->pk, msg, len);
    } else {
        return fail(err, errlen, "unknown signature algorithm");
    }
    if (ok != 0)
        return fail(err, errlen, "signature does not match the manifest");

    /* The global signature binds the trusted comment to this signature. */
    size_t clen = strlen(comment);
    uint8_t *gmsg = malloc(64 + clen);
    if (!gmsg)
        return fail(err, errlen, "oom");
    memcpy(gmsg, sig, 64);
    memcpy(gmsg + 64, comment, clen);
    ok = crypto_ed25519_check(global, key->pk, gmsg, 64 + clen);
    free(gmsg);
    if (ok != 0)
        return fail(err, errlen, "trusted comment signature does not match");
    if (trusted_comment)
        memcpy(trusted_comment, comment, PN_MINISIGN_COMMENT_MAX);
    if (which_key)
        *which_key = ki;
    return 0;
}
