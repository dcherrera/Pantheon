/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   mdns_packet.h
 * @brief  mDNS / DNS-SD wire format for the appliance (RFC 6762 / 6763), pure and
 *         socket-free so it is unit-tested: parse a message (questions + the
 *         names of its answer records), build the probe, the response to a
 *         query (PTR / SRV / TXT / A for `_pantheon._tcp`), the announcement and
 *         the goodbye, detect a name conflict, and pick the name for the next attempt (`-2`).
 *
 * @status None
 * @issues Names are written without compression (responses stay well under
 *         one MTU). Simultaneous-probe tie-breaking (RFC 6762 §8.2) is not done.
 * @todo   None
 */

#ifndef PN_LOCAL_MDNS_PACKET_H
#define PN_LOCAL_MDNS_PACKET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_MDNS_PORT         5353
#define PN_MDNS_GROUP        "224.0.0.251"
#define PN_MDNS_SERVICE      "_pantheon._tcp.local"
#define PN_MDNS_ENUM         "_services._dns-sd._udp.local"
#define PN_MDNS_NAME_MAX     256
#define PN_MDNS_LABEL_MAX    63
#define PN_MDNS_MAX_QUESTIONS 16
#define PN_MDNS_MAX_ANSWERS   32
#define PN_MDNS_MAX_TXT       12
#define PN_MDNS_TTL          120

enum {
    PN_DNS_TYPE_A    = 1,
    PN_DNS_TYPE_PTR  = 12,
    PN_DNS_TYPE_TXT  = 16,
    PN_DNS_TYPE_AAAA = 28,
    PN_DNS_TYPE_SRV  = 33,
    PN_DNS_TYPE_ANY  = 255,
};

typedef struct {
    char     name[PN_MDNS_NAME_MAX];   /* dotted, no trailing dot */
    uint16_t type;
    bool     unicast_response;         /* QU bit */
} pn_mdns_question;

typedef struct {
    uint16_t         id;
    bool             is_response;      /* QR bit */
    pn_mdns_question questions[PN_MDNS_MAX_QUESTIONS];
    int              n_questions;
    char             answer_names[PN_MDNS_MAX_ANSWERS][PN_MDNS_NAME_MAX]; /* answers + authority */
    uint16_t         answer_types[PN_MDNS_MAX_ANSWERS];
    int              n_answers;
} pn_mdns_message;

/* What this box announces. */
typedef struct {
    char        label[PN_MDNS_LABEL_MAX + 1];  /* "pantheon-lab-node-1": host and instance */
    uint16_t    port;
    uint32_t    ipv4;                          /* network byte order */
    const char *txt[PN_MDNS_MAX_TXT];          /* "key=value" */
    int         n_txt;
} pn_mdns_service;

/**
 * Parse a DNS message (follows compression pointers, bounded).
 * @return 0, or -1 on a malformed packet
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_mdns_parse(const uint8_t *pkt, size_t len, pn_mdns_message *out);

/**
 * Build the response to `q`'s questions that this service can answer.
 * `legacy` (a querier not on port 5353) echoes the id and questions and sets no
 * cache-flush bits. Returns 0 when nothing matched.
 *
 * @return bytes written, 0 if no question matched (or out too small)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
size_t pn_mdns_build_answer(const pn_mdns_service *s, const pn_mdns_message *q, bool legacy,
                            uint8_t *out, size_t cap);

/**
 * Unsolicited announcement of every record (ttl = PN_MDNS_TTL), or the goodbye
 * (ttl 0) when `goodbye`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
size_t pn_mdns_build_announce(const pn_mdns_service *s, bool goodbye, uint8_t *out, size_t cap);

/**
 * Probe query (RFC 6762 §8.1): ANY questions for the host and instance names,
 * proposed records in the authority section.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
size_t pn_mdns_build_probe(const pn_mdns_service *s, uint8_t *out, size_t cap);

/**
 * Whether a RESPONSE from someone else claims one of our names.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
bool pn_mdns_conflicts(const pn_mdns_message *m, const pn_mdns_service *s);

/**
 * The label to try on attempt `n` (1-based): 1 → base, 2 → "base-2", ... The
 * suffix is added to the base, never parsed back out of a name, so a host
 * named "lab-node-1" does not become "lab-node-2" (another box's name).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
void pn_mdns_conflict_label(const char *base, int attempt, char *out, size_t cap);

/**
 * "pantheon-" + the hostname's first label, lowercased, non [a-z0-9-] → '-'.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
void pn_mdns_label_for_host(const char *hostname, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MDNS_PACKET_H */
