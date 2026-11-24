/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   attach_ticket.h
 * @brief  Attach tickets (people registry spec §4.6): the box mints one per
 *         attach, the host (and the box's relay) checks it. A ticket is
 *           b64url(payload) "." hex(HMAC-SHA256(host_secret, b64url(payload)))
 *         with base64url unpadded, the MAC over the ASCII of the first part, and
 *         the payload compact JSON in this field order:
 *           {"v":1,"key":"c-…","person":"p…","device_id":"d…","cpub":"<64 hex>",
 *            "role":"owner|control|view","exp":<unix s>,"nonce":"<32 hex>"}
 *         The HMAC key is the host_secret's 32 raw bytes (not its hex).
 *         Argus D7 matches this byte for byte (fixture ticket_golden.json).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_ATTACH_TICKET_H
#define PN_LOCAL_ATTACH_TICKET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_TICKET_VERSION     1
#define PN_TICKET_TTL_S       60    /* exp = issue + 60 */
#define PN_TICKET_SKEW_S      30    /* accepted until exp + 30 (LAN clock skew) */
#define PN_TICKET_SECRET_LEN  32
#define PN_TICKET_MAX         512   /* a ticket's bytes + NUL fit (about 380 used) */
#define PN_TICKET_KEY_MAX     16    /* "c-" + 12 hex + NUL fits */
#define PN_TICKET_PERSON_MAX  16    /* "p" + 8 hex + NUL fits */
#define PN_TICKET_DEVICE_MAX  18    /* "d" + 16 hex + NUL */
#define PN_TICKET_CPUB_MAX    65    /* 64 lowercase hex + NUL */
#define PN_TICKET_NONCE_MAX   33    /* 32 lowercase hex + NUL */

typedef enum {
    PN_TICKET_ROLE_VIEW = 1,
    PN_TICKET_ROLE_CONTROL,
    PN_TICKET_ROLE_OWNER,
} pn_ticket_role;

/* Typed results (0 = success). */
typedef enum {
    PN_TICKET_OK        = 0,
    PN_TICKET_E_INVALID = -1,   /* a field to mint breaks the rules, or out is too small */
    PN_TICKET_E_FORMAT  = -2,   /* not <b64url>.<64 hex>, or the payload is not a ticket */
    PN_TICKET_E_MAC     = -3,   /* the MAC does not match under this secret */
    PN_TICKET_E_VERSION = -4,   /* v is not 1 */
    PN_TICKET_E_KEY     = -5,   /* minted for another session key */
    PN_TICKET_E_EXPIRED = -6,   /* now > exp + 30 */
    PN_TICKET_E_RANDOM  = -7,   /* no randomness for the nonce */
} pn_ticket_result;

/* The payload's fields. */
typedef struct {
    int            version;
    char           key[PN_TICKET_KEY_MAX];
    char           person[PN_TICKET_PERSON_MAX];
    char           device_id[PN_TICKET_DEVICE_MAX];
    char           cpub[PN_TICKET_CPUB_MAX];
    pn_ticket_role role;
    long long      exp;
    char           nonce[PN_TICKET_NONCE_MAX];
} pn_ticket_fields;

/**
 * Write the ticket for `fields` exactly (no clock, no randomness): the golden
 * fixture's path. Every field must match its pattern (lowercase hex), v 1,
 * exp > 0.
 *
 * @param secret  The host_secret's 32 bytes
 * @param fields  What the payload carries
 * @param out     Receives the NUL-terminated ticket
 * @param cap     Size of out (PN_TICKET_MAX is always enough)
 * @return        PN_TICKET_OK or PN_TICKET_E_INVALID
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_ticket_encode(const uint8_t secret[PN_TICKET_SECRET_LEN], const pn_ticket_fields *fields,
                     char *out, size_t cap);

/**
 * Mint a ticket: v 1, exp = now + 60, a fresh random nonce.
 *
 * @param fields_out  May be NULL; receives the fields minted (the nonce included)
 * @return            PN_TICKET_OK, PN_TICKET_E_INVALID or PN_TICKET_E_RANDOM
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_ticket_mint(const uint8_t secret[PN_TICKET_SECRET_LEN], const char *key,
                   const char *person, const char *device_id, const char *cpub,
                   pn_ticket_role role, long long now, pn_ticket_fields *fields_out,
                   char *out, size_t cap);

/**
 * Check a ticket for session `key` at `now`: split at the last '.', the MAC in
 * constant time, then the payload (every field and its pattern), v == 1, the
 * key, now <= exp + 30. The nonce (seen once) and the handshake's cpub are the
 * caller's to check against `fields_out`.
 *
 * @param fields_out  May be NULL; filled only on PN_TICKET_OK
 * @return            PN_TICKET_OK or a negative pn_ticket_result
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_ticket_check(const uint8_t secret[PN_TICKET_SECRET_LEN], const char *key,
                    const char *ticket, long long now, pn_ticket_fields *fields_out);

/** Wire name of a role ("view", "control", "owner"). Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const char *pn_ticket_role_name(pn_ticket_role role);

/** Name of a pn_ticket_result, for logs. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const char *pn_ticket_result_name(int result);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_ATTACH_TICKET_H */
