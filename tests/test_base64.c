/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_base64.c
 * @brief Unit test for the Base64 encode/decode used by the session warm-tier
 *        (K80 serving-ops Phase 7). Covers RFC 4648 test vectors, a
 *        random-bytes round-trip across every length residue (padding cases),
 *        and rejection of malformed input (bad char, bad length, misplaced
 *        padding). Host-side, no deps.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/serve/base64.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* RFC 4648 §10 test vectors. */
static void test_vectors(void)
{
    printf("test: RFC 4648 vectors\n");
    struct { const char *in; const char *out; } v[] = {
        { "",       ""         },
        { "f",      "Zg=="     },
        { "fo",     "Zm8="     },
        { "foo",    "Zm9v"     },
        { "foob",   "Zm9vYg==" },
        { "fooba",  "Zm9vYmE=" },
        { "foobar", "Zm9vYmFy" },
    };
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
        char *enc = pn_base64_encode((const unsigned char *)v[i].in,
                                     strlen(v[i].in));
        CHECK(enc && strcmp(enc, v[i].out) == 0, v[i].out[0] ? v[i].out : "(empty)");
        size_t dl = 0;
        unsigned char *dec = pn_base64_decode(v[i].out, &dl);
        CHECK(dec && dl == strlen(v[i].in) &&
              memcmp(dec, v[i].in, dl) == 0, "decode inverts");
        free(enc);
        free(dec);
    }
}

/* Random-bytes round-trip across all length residues mod 3. */
static void test_roundtrip(void)
{
    printf("test: random round-trip (all padding residues)\n");
    srand(1234);
    int ok = 1;
    for (int len = 0; len <= 300 && ok; len++) {
        unsigned char *buf = malloc(len ? len : 1);
        for (int i = 0; i < len; i++) buf[i] = (unsigned char)(rand() & 0xFF);
        char *enc = pn_base64_encode(buf, (size_t)len);
        size_t dl = 0;
        unsigned char *dec = pn_base64_decode(enc, &dl);
        if (!enc || !dec || dl != (size_t)len ||
            (len && memcmp(dec, buf, len) != 0))
            ok = 0;
        free(buf); free(enc); free(dec);
    }
    CHECK(ok, "every length 0..300 round-trips byte-exact");
}

static void test_reject(void)
{
    printf("test: malformed input rejected\n");
    size_t dl = 0;
    CHECK(pn_base64_decode("Zm9vYg=", &dl) == NULL, "non-multiple-of-4 rejected");
    CHECK(pn_base64_decode("Zm9!", &dl) == NULL, "invalid char rejected");
    CHECK(pn_base64_decode("Z=m8", &dl) == NULL, "misplaced padding rejected");
    CHECK(pn_base64_decode(NULL, &dl) == NULL, "NULL rejected");
}

int main(void)
{
    printf("=== test_base64 ===\n");
    test_vectors();
    test_roundtrip();
    test_reject();
    if (failures) {
        printf("=== %d FAILURE(S) ===\n", failures);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
