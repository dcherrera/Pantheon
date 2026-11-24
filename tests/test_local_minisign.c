/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_minisign.c — Pantheon Local E2/E3/E7: SHA-256 (FIPS 180-4 vectors)
 * and minisign verification against signatures made by the real minisign CLI
 * (tests/local/fixtures, TEST ONLY keys): good signature, prehashed and legacy
 * forms, backup key accepted, flipped byte refused, wrong key refused,
 * unsigned refused, tampered trusted comment refused, and the red arm — a
 * manifest with one sha256 changed must fail to verify.
 */

#include "local_test_support.h"

#include "../src/local/minisign_verify.h"
#include "../src/local/sha256.h"

#define FX "tests/local/fixtures/"

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) n = 0;
    b[n] = '\0';
    fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

static void test_sha256_vectors(void)
{
    uint8_t d[32];
    char hex[65];

    printf("sha256_vectors:\n");
    pn_sha256("", 0, d);
    pn_hex_encode(d, 32, hex);
    CHECK(strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0, "empty");
    pn_sha256("abc", 3, d);
    pn_hex_encode(d, 32, hex);
    CHECK(strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0, "abc");
    const char *two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    pn_sha256(two, strlen(two), d);
    pn_hex_encode(d, 32, hex);
    CHECK(strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0,
          "448-bit message");
    pn_sha256_ctx c;
    pn_sha256_init(&c);
    char block[1000];
    memset(block, 'a', sizeof(block));
    for (int i = 0; i < 1000; i++)
        pn_sha256_update(&c, block, sizeof(block));
    pn_sha256_final(&c, d);
    pn_hex_encode(d, 32, hex);
    CHECK(strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") == 0,
          "one million 'a' (streamed)");
    uint8_t back[32];
    CHECK(pn_hex_decode(hex, back, 32) == 0 && memcmp(back, d, 32) == 0, "hex round trip");
    CHECK(pn_hex_decode("zz", back, 1) == -1, "bad hex refused");
}

static int keys_from(const char **files, int n, pn_minisign_pubkey *out)
{
    int k = 0;
    for (int i = 0; i < n; i++) {
        char *t = slurp(files[i], NULL);
        if (t && pn_minisign_parse_pubkey(t, &out[k]) == 0) k++;
        free(t);
    }
    return k;
}

static int verify(const char *manifest, const char *sigfile, const pn_minisign_pubkey *keys,
                  int n, int *which, char *err)
{
    size_t len = 0;
    char *m = slurp(manifest, &len), *s = sigfile ? slurp(sigfile, NULL) : NULL;
    char tc[PN_MINISIGN_COMMENT_MAX];
    int rc = (m) ? pn_minisign_verify((uint8_t *)m, len, s, keys, n, tc, which, err, 128) : -9;
    free(m);
    free(s);
    return rc;
}

static void test_minisign(void)
{
    const char *trusted[] = { FX "test_catalog.pub", FX "test_catalog_backup.pub" };
    pn_minisign_pubkey keys[2], only_current[1];
    char err[128];
    int which = -1;

    printf("minisign:\n");
    CHECK(keys_from(trusted, 2, keys) == 2, "parse both public keys");
    CHECK(keys_from(trusted, 1, only_current) == 1, "parse current key");
    CHECK(verify(FX "catalog_v1.json", FX "catalog_v1.json.minisig", keys, 2, &which, err) == 0
          && which == 0, "good prehashed signature, current key");
    CHECK(verify(FX "catalog_v1.json", FX "catalog_v1.legacy.minisig", keys, 2, &which, err) == 0,
          "good legacy signature");
    CHECK(verify(FX "catalog_v2.json", FX "catalog_v2.backup.minisig", keys, 2, &which, err) == 0
          && which == 1, "backup key accepted");
    CHECK(verify(FX "catalog_v2.json", FX "catalog_v2.backup.minisig", only_current, 1, &which,
                 err) == -1, "backup signature refused when only the current key is trusted");
    CHECK(verify(FX "catalog_v1.json", FX "catalog_v1.untrusted.minisig", keys, 2, NULL, err) == -1
          && strstr(err, "does not trust"), "wrong key refused");
    CHECK(verify(FX "catalog_v1.json", NULL, keys, 2, NULL, err) == -1, "unsigned refused");
    CHECK(verify(FX "catalog_v2_tampered.json", FX "catalog_v2_tampered.json.minisig", keys, 2,
                 NULL, err) == -1 && strstr(err, "does not match"),
          "red arm: one changed sha256 fails to verify");

    size_t len = 0;
    char *m = slurp(FX "catalog_v1.json", &len), *s = slurp(FX "catalog_v1.json.minisig", NULL);
    m[len / 2] ^= 1;
    CHECK(pn_minisign_verify((uint8_t *)m, len, s, keys, 2, NULL, NULL, err, sizeof(err)) == -1,
          "flipped byte refused");
    m[len / 2] ^= 1;
    char *tc = strstr(s, "\ntrusted comment: ");
    if (tc) tc[21] ^= 1;
    CHECK(pn_minisign_verify((uint8_t *)m, len, s, keys, 2, NULL, NULL, err, sizeof(err)) == -1,
          "edited trusted comment refused");
    free(m);
    free(s);
}

int main(void)
{
    test_sha256_vectors();
    test_minisign();
    return lt_report("test_local_minisign");
}
