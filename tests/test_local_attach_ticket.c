/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_attach_ticket.c — people registry P3.1: HMAC-SHA256 and attach
 * tickets (spec §4.6), no sockets:
 *   - pn_hmac_sha256 against the seven RFC 4231 vectors (test 5 truncated to
 *     128 bits; 6 and 7 hash a 131-byte key first);
 *   - the golden ticket (tests/local/fixtures/people_registry/ticket_golden.json,
 *     made by Python's hmac + base64): encode reproduces it byte for byte, check
 *     accepts it until exp + 30 and gives back every field;
 *   - mint: v 1, exp = now + 60, fresh 32-hex nonces, the cpub carried;
 *   - refused: expired, bad MAC (wrong secret, a flipped MAC digit), wrong
 *     session key, forged role (payload re-encoded with the old MAC), truncated
 *     (either part), no dot, padding, a bad field to mint.
 */

#include "local_test_support.h"
#include "local_fixture_replay.h"

#include "../src/local/attach_ticket.h"
#include "../src/local/sha256.h"

typedef struct {
    const char *key_spec;   /* "x*N" = byte 0xx repeated N times, else raw text */
    const char *msg_spec;
    const char *mac_hex;    /* a prefix compare when shorter than 64 (test 5) */
} rfc4231_case;

static const rfc4231_case k_cases[] = {
    { "0b*20", "Hi There",
      "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7" },
    { "Jefe", "what do ya want for nothing?",
      "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843" },
    { "aa*20", "dd*50",
      "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe" },
    { "seq25", "cd*50",
      "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b" },
    { "0c*20", "Test With Truncation", "a3b6167473100ee06e0c796c2955552b" },
    { "aa*131", "Test Using Larger Than Block-Size Key - Hash Key First",
      "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54" },
    { "aa*131", "This is a test using a larger than block-size key and a larger than "
      "block-size data. The key needs to be hashed before being used by the HMAC algorithm.",
      "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2" },
};

/* Expand a spec into bytes; the length. */
static size_t expand(const char *spec, uint8_t *out, size_t cap)
{
    unsigned byte = 0, count = 0;

    if (strcmp(spec, "seq25") == 0) {
        for (size_t i = 0; i < 25; i++)
            out[i] = (uint8_t)(i + 1);
        return 25;
    }
    if (sscanf(spec, "%2x*%u", &byte, &count) == 2 && strchr(spec, '*') && count <= cap) {
        memset(out, (int)byte, count);
        return count;
    }
    snprintf((char *)out, cap, "%s", spec);
    return strlen(spec);
}

static void test_hmac_rfc4231(void)
{
    printf("hmac_rfc4231:\n");
    for (size_t i = 0; i < sizeof(k_cases) / sizeof(k_cases[0]); i++) {
        uint8_t key[256], msg[256], mac[PN_SHA256_LEN];
        char hex[PN_SHA256_HEX_LEN + 1], what[64];
        size_t kl = expand(k_cases[i].key_spec, key, sizeof(key));
        size_t ml = expand(k_cases[i].msg_spec, msg, sizeof(msg));

        pn_hmac_sha256(key, kl, msg, ml, mac);
        pn_hex_encode(mac, PN_SHA256_LEN, hex);
        snprintf(what, sizeof(what), "RFC 4231 test case %zu", i + 1);
        CHECK(strncmp(hex, k_cases[i].mac_hex, strlen(k_cases[i].mac_hex)) == 0, what);
    }
}

typedef struct {
    uint8_t          secret[PN_TICKET_SECRET_LEN];
    pn_ticket_fields fields;
    char             ticket[PN_TICKET_MAX];
    long long        accept_until;
} golden;

static int copy_str(const jr_value *obj, const char *name, char *out, size_t cap)
{
    const char *s = jr_as_string(jr_object_get(obj, name));
    return s && snprintf(out, cap, "%s", s) < (int)cap ? 0 : -1;
}

static int load_golden(golden *g)
{
    jr_value *fx = fx_load("ticket_golden.json");
    const jr_value *f = jr_object_get(fx, "fields");
    const char *secret = jr_as_string(jr_object_get(fx, "secret"));
    const char *role = jr_as_string(jr_object_get(f, "role"));
    int ok = fx && secret && strlen(secret) == 64 &&
             pn_hex_decode(secret, g->secret, sizeof(g->secret)) == 0 &&
             copy_str(f, "key", g->fields.key, sizeof(g->fields.key)) == 0 &&
             copy_str(f, "person", g->fields.person, sizeof(g->fields.person)) == 0 &&
             copy_str(f, "device_id", g->fields.device_id, sizeof(g->fields.device_id)) == 0 &&
             copy_str(f, "cpub", g->fields.cpub, sizeof(g->fields.cpub)) == 0 &&
             copy_str(f, "nonce", g->fields.nonce, sizeof(g->fields.nonce)) == 0 &&
             copy_str(fx, "ticket", g->ticket, sizeof(g->ticket)) == 0 && role;

    g->fields.version = (int)jr_as_number(jr_object_get(f, "v"), 0);
    g->fields.exp = (long long)jr_as_number(jr_object_get(f, "exp"), 0);
    g->accept_until = (long long)jr_as_number(jr_object_get(fx, "accept_until"), 0);
    for (int r = PN_TICKET_ROLE_VIEW; role && r <= PN_TICKET_ROLE_OWNER; r++)
        if (strcmp(role, pn_ticket_role_name((pn_ticket_role)r)) == 0)
            g->fields.role = (pn_ticket_role)r;
    jr_free(fx);
    return ok && g->fields.role ? 0 : -1;
}

static void test_golden(const golden *g)
{
    char out[PN_TICKET_MAX];
    pn_ticket_fields got = {0};

    printf("golden:\n");
    CHECK(pn_ticket_encode(g->secret, &g->fields, out, sizeof(out)) == PN_TICKET_OK &&
          strcmp(out, g->ticket) == 0, "encode reproduces the golden ticket byte for byte");
    CHECK(pn_ticket_check(g->secret, g->fields.key, g->ticket, g->fields.exp - 60, &got) ==
          PN_TICKET_OK, "check accepts it at issue time");
    CHECK(got.version == 1 && strcmp(got.key, g->fields.key) == 0 &&
          strcmp(got.person, g->fields.person) == 0 &&
          strcmp(got.device_id, g->fields.device_id) == 0 &&
          strcmp(got.cpub, g->fields.cpub) == 0 && got.role == g->fields.role &&
          got.exp == g->fields.exp && strcmp(got.nonce, g->fields.nonce) == 0,
          "and gives back every field");
    CHECK(g->accept_until == g->fields.exp + PN_TICKET_SKEW_S &&
          pn_ticket_check(g->secret, g->fields.key, g->ticket, g->accept_until, NULL) ==
          PN_TICKET_OK, "accepted at exp + 30 (skew)");
    CHECK(pn_ticket_check(g->secret, g->fields.key, g->ticket, g->accept_until + 1, NULL) ==
          PN_TICKET_E_EXPIRED, "expired at exp + 31");
    CHECK(pn_ticket_encode(g->secret, &g->fields, out, 100) == PN_TICKET_E_INVALID,
          "a short out buffer is refused, not cut");
}

static void test_mint(const golden *g)
{
    char a[PN_TICKET_MAX], b[PN_TICKET_MAX];
    pn_ticket_fields fa = {0}, fb = {0}, got = {0};
    const long long now = 1793000000;

    printf("mint:\n");
    CHECK(pn_ticket_mint(g->secret, "c-00000000000a", "p00000002", "d00000000000000bb",
                         g->fields.cpub, PN_TICKET_ROLE_VIEW, now, &fa, a, sizeof(a)) ==
          PN_TICKET_OK, "mint a view ticket");
    CHECK(pn_ticket_mint(g->secret, "c-00000000000a", "p00000002", "d00000000000000bb",
                         g->fields.cpub, PN_TICKET_ROLE_VIEW, now, &fb, b, sizeof(b)) ==
          PN_TICKET_OK, "mint another");
    CHECK(fa.version == 1 && fa.exp == now + 60 && strlen(fa.nonce) == 32 &&
          strcmp(fa.nonce, fb.nonce) != 0 && strcmp(a, b) != 0,
          "v 1, exp = now + 60, a fresh nonce each time");
    CHECK(pn_ticket_check(g->secret, "c-00000000000a", a, now, &got) == PN_TICKET_OK &&
          got.role == PN_TICKET_ROLE_VIEW && strcmp(got.cpub, g->fields.cpub) == 0 &&
          strcmp(got.nonce, fa.nonce) == 0, "it checks, with its role, cpub and nonce");
    CHECK(pn_ticket_mint(g->secret, "c-00000000000a", "p00000002", "d00000000000000bb",
                         "8F40", PN_TICKET_ROLE_VIEW, now, NULL, a, sizeof(a)) ==
          PN_TICKET_E_INVALID, "a bad cpub is refused at mint");
    CHECK(pn_ticket_mint(g->secret, "c-XYZ", "p00000002", "d00000000000000bb", g->fields.cpub,
                         PN_TICKET_ROLE_VIEW, now, NULL, a, sizeof(a)) == PN_TICKET_E_INVALID,
          "a bad key is refused at mint");
}

/* The golden fields with role "owner", encoded under a throwaway secret, with
 * the golden MAC spliced on: a payload the box never signed. */
static void forge_role(const golden *g, char *out, size_t cap)
{
    const uint8_t *throwaway = (const uint8_t *)"0123456789abcdef0123456789abcdef";
    const char *mac = strchr(g->ticket, '.');
    pn_ticket_fields f = g->fields;
    char tmp[PN_TICKET_MAX];

    f.role = PN_TICKET_ROLE_OWNER;
    out[0] = '\0';
    if (!mac || pn_ticket_encode(throwaway, &f, tmp, sizeof(tmp)) != PN_TICKET_OK)
        return;
    *strchr(tmp, '.') = '\0';
    snprintf(out, cap, "%s%s", tmp, mac);
}

static void test_refused(const golden *g)
{
    char t[PN_TICKET_MAX * 2];
    uint8_t other[PN_TICKET_SECRET_LEN];
    const long long now = g->fields.exp - 60;
    const size_t len = strlen(g->ticket);

    printf("refused:\n");
    memcpy(other, g->secret, sizeof(other));
    other[0] ^= 1;
    CHECK(pn_ticket_check(other, g->fields.key, g->ticket, now, NULL) == PN_TICKET_E_MAC,
          "another secret: bad MAC");
    snprintf(t, sizeof(t), "%s", g->ticket);
    t[len - 1] = t[len - 1] == '0' ? '1' : '0';
    CHECK(pn_ticket_check(g->secret, g->fields.key, t, now, NULL) == PN_TICKET_E_MAC,
          "a flipped MAC digit: bad MAC");
    CHECK(pn_ticket_check(g->secret, "c-0123456789ac", g->ticket, now, NULL) == PN_TICKET_E_KEY,
          "another session key: refused");
    forge_role(g, t, sizeof(t));
    CHECK(t[0] && strstr(t, ".") && pn_ticket_check(g->secret, g->fields.key, t, now, NULL) ==
          PN_TICKET_E_MAC, "a forged role (payload changed, MAC kept): bad MAC");
    snprintf(t, sizeof(t), "%.*s", (int)(len - 1), g->ticket);
    CHECK(pn_ticket_check(g->secret, g->fields.key, t, now, NULL) == PN_TICKET_E_FORMAT,
          "truncated MAC: refused");
    snprintf(t, sizeof(t), "%s", g->ticket + 4);
    CHECK(pn_ticket_check(g->secret, g->fields.key, t, now, NULL) == PN_TICKET_E_MAC,
          "truncated payload: bad MAC");
    snprintf(t, sizeof(t), "%.*s", (int)(strchr(g->ticket, '.') - g->ticket), g->ticket);
    CHECK(pn_ticket_check(g->secret, g->fields.key, t, now, NULL) == PN_TICKET_E_FORMAT,
          "no dot: refused");
    CHECK(pn_ticket_check(g->secret, g->fields.key, "", now, NULL) == PN_TICKET_E_FORMAT &&
          pn_ticket_check(g->secret, g->fields.key, NULL, now, NULL) == PN_TICKET_E_FORMAT,
          "empty and NULL: refused");
    snprintf(t, sizeof(t), "%.*s=%s", (int)(strchr(g->ticket, '.') - g->ticket), g->ticket,
             strchr(g->ticket, '.'));
    CHECK(pn_ticket_check(g->secret, g->fields.key, t, now, NULL) != PN_TICKET_OK,
          "padding: refused");
}

int main(void)
{
    golden g = {0};

    test_hmac_rfc4231();
    if (load_golden(&g) != 0) {
        fprintf(stderr, "could not load ticket_golden.json\n");
        return 1;
    }
    test_golden(&g);
    test_mint(&g);
    test_refused(&g);
    return lt_report("test_local_attach_ticket");
}
