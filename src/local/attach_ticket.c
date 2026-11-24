/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   attach_ticket.c
 * @brief  Implementation of attach tickets (see attach_ticket.h): base64url,
 *         the payload writer and parser, the MAC. Never logs a ticket.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "attach_ticket.h"

#include <stdio.h>
#include <string.h>

#include "../../lib/monocypher/monocypher.h"
#include "../serve/json_reader.h"
#include "local_random.h"
#include "sha256.h"

#define PAYLOAD_MAX 384   /* the decoded payload (about 235 bytes used) */

static const char k_b64url[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

static const char *const k_role_names[] = { "", "view", "control", "owner" };

const char *pn_ticket_role_name(pn_ticket_role role)
{
    return role >= PN_TICKET_ROLE_VIEW && role <= PN_TICKET_ROLE_OWNER ? k_role_names[role]
                                                                       : "unknown";
}

const char *pn_ticket_result_name(int result)
{
    static const char *const names[] = { "ok", "invalid", "format", "mac", "version", "key",
                                         "expired", "random" };
    return result <= 0 && result >= PN_TICKET_E_RANDOM ? names[-result] : "unknown";
}

/* Unpadded base64url of n bytes into out; -1 when out is too small. */
static int b64url_encode(const uint8_t *in, size_t n, char *out, size_t cap)
{
    size_t o = 0;

    if (cap < (n * 4 + 2) / 3 + 1)
        return -1;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        size_t left = n - i;
        if (left > 1)
            v |= (uint32_t)in[i + 1] << 8;
        if (left > 2)
            v |= in[i + 2];
        out[o++] = k_b64url[(v >> 18) & 63];
        out[o++] = k_b64url[(v >> 12) & 63];
        if (left > 1)
            out[o++] = k_b64url[(v >> 6) & 63];
        if (left > 2)
            out[o++] = k_b64url[v & 63];
    }
    out[o] = '\0';
    return 0;
}

static int b64url_value(char c)
{
    const char *p = c ? strchr(k_b64url, c) : NULL;
    return p ? (int)(p - k_b64url) : -1;
}

/* Strict decode of n chars (no padding, canonical trailing bits); the byte
 * count, or -1. */
static int b64url_decode(const char *in, size_t n, uint8_t *out, size_t cap)
{
    size_t o = 0;
    uint32_t acc = 0;
    int bits = 0;

    if (n % 4 == 1 || n / 4 * 3 + 2 > cap)
        return -1;
    for (size_t i = 0; i < n; i++) {
        int v = b64url_value(in[i]);
        if (v < 0)
            return -1;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = (uint8_t)(acc >> bits);
        }
    }
    if (bits && (acc & ((1u << bits) - 1)) != 0)
        return -1;
    return (int)o;
}

/* `s` is `prefix` + exactly n lowercase hex digits. */
static int prefixed_hex(const char *s, const char *prefix, size_t n)
{
    size_t p = strlen(prefix);

    if (!s || strncmp(s, prefix, p) != 0 || strlen(s) != p + n)
        return 0;
    for (size_t i = p; i < p + n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return 0;
    return 1;
}

static int fields_valid(const pn_ticket_fields *f)
{
    return f && f->version == PN_TICKET_VERSION && prefixed_hex(f->key, "c-", 12) &&
           prefixed_hex(f->person, "p", 8) && prefixed_hex(f->device_id, "d", 16) &&
           prefixed_hex(f->cpub, "", 64) && prefixed_hex(f->nonce, "", 32) &&
           f->role >= PN_TICKET_ROLE_VIEW && f->role <= PN_TICKET_ROLE_OWNER && f->exp > 0;
}

/* hex(HMAC-SHA256(secret, part[0..n))) into mac_hex[65]. */
static void mac_hex_of(const uint8_t secret[PN_TICKET_SECRET_LEN], const char *part, size_t n,
                       char mac_hex[PN_SHA256_HEX_LEN + 1])
{
    uint8_t mac[PN_SHA256_LEN];

    pn_hmac_sha256(secret, PN_TICKET_SECRET_LEN, part, n, mac);
    pn_hex_encode(mac, PN_SHA256_LEN, mac_hex);
    crypto_wipe(mac, sizeof(mac));
}

int pn_ticket_encode(const uint8_t secret[PN_TICKET_SECRET_LEN], const pn_ticket_fields *f,
                     char *out, size_t cap)
{
    char payload[PAYLOAD_MAX], mac_hex[PN_SHA256_HEX_LEN + 1];
    int n = 0;
    size_t b = 0;

    if (!secret || !out || !fields_valid(f))
        return PN_TICKET_E_INVALID;
    n = snprintf(payload, sizeof(payload),
                 "{\"v\":%d,\"key\":\"%s\",\"person\":\"%s\",\"device_id\":\"%s\","
                 "\"cpub\":\"%s\",\"role\":\"%s\",\"exp\":%lld,\"nonce\":\"%s\"}",
                 f->version, f->key, f->person, f->device_id, f->cpub,
                 pn_ticket_role_name(f->role), f->exp, f->nonce);
    if (n < 0 || (size_t)n >= sizeof(payload) ||
        b64url_encode((const uint8_t *)payload, (size_t)n, out, cap) != 0)
        return PN_TICKET_E_INVALID;
    b = strlen(out);
    if (b + 1 + PN_SHA256_HEX_LEN + 1 > cap)
        return PN_TICKET_E_INVALID;
    mac_hex_of(secret, out, b, mac_hex);
    out[b] = '.';
    memcpy(out + b + 1, mac_hex, PN_SHA256_HEX_LEN + 1);
    return PN_TICKET_OK;
}

int pn_ticket_mint(const uint8_t secret[PN_TICKET_SECRET_LEN], const char *key,
                   const char *person, const char *device_id, const char *cpub,
                   pn_ticket_role role, long long now, pn_ticket_fields *fields_out,
                   char *out, size_t cap)
{
    pn_ticket_fields f = { .version = PN_TICKET_VERSION, .role = role,
                           .exp = now + PN_TICKET_TTL_S };
    uint8_t nonce[16];
    int rc = 0;

    if (!key || !person || !device_id || !cpub ||
        snprintf(f.key, sizeof(f.key), "%s", key) >= (int)sizeof(f.key) ||
        snprintf(f.person, sizeof(f.person), "%s", person) >= (int)sizeof(f.person) ||
        snprintf(f.device_id, sizeof(f.device_id), "%s", device_id) >= (int)sizeof(f.device_id) ||
        snprintf(f.cpub, sizeof(f.cpub), "%s", cpub) >= (int)sizeof(f.cpub))
        return PN_TICKET_E_INVALID;
    if (pn_local_random_bytes(nonce, sizeof(nonce)) != 0)
        return PN_TICKET_E_RANDOM;
    pn_hex_encode(nonce, sizeof(nonce), f.nonce);
    rc = pn_ticket_encode(secret, &f, out, cap);
    if (rc == PN_TICKET_OK && fields_out)
        *fields_out = f;
    return rc;
}

/* A string member matching prefix + n lowercase hex, copied into out. */
static int read_hex_field(const jr_value *obj, const char *name, const char *prefix, size_t n,
                          char *out, size_t cap)
{
    const char *s = jr_as_string(jr_object_get(obj, name));

    if (!prefixed_hex(s, prefix, n))
        return -1;
    return snprintf(out, cap, "%s", s) < (int)cap ? 0 : -1;
}

/* An integral number member in [lo, hi]. */
static int read_int_field(const jr_value *obj, const char *name, double lo, double hi,
                          long long *out)
{
    const jr_value *v = jr_object_get(obj, name);
    double d = 0;

    if (jr_kind(v) != JR_NUMBER)
        return -1;
    d = jr_as_number(v, 0);
    if (d < lo || d > hi || d != (double)(long long)d)
        return -1;
    *out = (long long)d;
    return 0;
}

/* Parse the decoded payload into f: exactly the eight members, each to rule. */
static int parse_payload(const char *json, pn_ticket_fields *f)
{
    jr_value *v = jr_parse(json);
    const char *role = NULL;
    long long version = 0;
    int rc = PN_TICKET_E_FORMAT;

    if (jr_kind(v) != JR_OBJECT || jr_object_member_count(v) != 8 ||
        read_int_field(v, "v", 0, 1e9, &version) != 0 ||
        read_int_field(v, "exp", 1, 9007199254740991.0, &f->exp) != 0 ||
        read_hex_field(v, "key", "c-", 12, f->key, sizeof(f->key)) != 0 ||
        read_hex_field(v, "person", "p", 8, f->person, sizeof(f->person)) != 0 ||
        read_hex_field(v, "device_id", "d", 16, f->device_id, sizeof(f->device_id)) != 0 ||
        read_hex_field(v, "cpub", "", 64, f->cpub, sizeof(f->cpub)) != 0 ||
        read_hex_field(v, "nonce", "", 32, f->nonce, sizeof(f->nonce)) != 0)
        goto done;
    role = jr_as_string(jr_object_get(v, "role"));
    f->role = (pn_ticket_role)0;
    for (int r = PN_TICKET_ROLE_VIEW; role && r <= PN_TICKET_ROLE_OWNER; r++)
        if (strcmp(role, k_role_names[r]) == 0)
            f->role = (pn_ticket_role)r;
    if (!f->role)
        goto done;
    f->version = (int)version;
    rc = version == PN_TICKET_VERSION ? PN_TICKET_OK : PN_TICKET_E_VERSION;
done:
    jr_free(v);
    return rc;
}

int pn_ticket_check(const uint8_t secret[PN_TICKET_SECRET_LEN], const char *key,
                    const char *ticket, long long now, pn_ticket_fields *fields_out)
{
    char mac_hex[PN_SHA256_HEX_LEN + 1], payload[PAYLOAD_MAX];
    uint8_t want[PN_SHA256_LEN], got[PN_SHA256_LEN];
    const char *dot = ticket ? strrchr(ticket, '.') : NULL;
    pn_ticket_fields f = {0};
    size_t plen = dot ? (size_t)(dot - ticket) : 0;
    int n = 0, rc = PN_TICKET_OK;

    if (!secret || !key || !dot || plen == 0 || strlen(ticket) >= PN_TICKET_MAX ||
        !prefixed_hex(dot + 1, "", PN_SHA256_HEX_LEN))
        return PN_TICKET_E_FORMAT;
    mac_hex_of(secret, ticket, plen, mac_hex);
    if (pn_hex_decode(mac_hex, want, PN_SHA256_LEN) != 0 ||
        pn_hex_decode(dot + 1, got, PN_SHA256_LEN) != 0)
        return PN_TICKET_E_FORMAT;
    rc = crypto_verify32(want, got) == 0 ? PN_TICKET_OK : PN_TICKET_E_MAC;
    crypto_wipe(want, sizeof(want));
    crypto_wipe(mac_hex, sizeof(mac_hex));
    if (rc != PN_TICKET_OK)
        return rc;
    n = b64url_decode(ticket, plen, (uint8_t *)payload, sizeof(payload) - 1);
    if (n < 0 || memchr(payload, '\0', (size_t)n))
        return PN_TICKET_E_FORMAT;
    payload[n] = '\0';
    if ((rc = parse_payload(payload, &f)) != PN_TICKET_OK)
        return rc;
    if (strcmp(f.key, key) != 0)
        return PN_TICKET_E_KEY;
    if (now > f.exp + PN_TICKET_SKEW_S)
        return PN_TICKET_E_EXPIRED;
    if (fields_out)
        *fields_out = f;
    return PN_TICKET_OK;
}
