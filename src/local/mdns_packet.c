/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   mdns_packet.c
 * @brief  Implementation of the mDNS wire format (see mdns_packet.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "mdns_packet.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#define DNS_CLASS_IN     1
#define DNS_CACHE_FLUSH  0x8000
#define DNS_QU_BIT       0x8000

/* Records this box owns, as a bitmask. */
enum {
    REC_PTR  = 1 << 0,     /* _pantheon._tcp.local → instance */
    REC_SRV  = 1 << 1,
    REC_TXT  = 1 << 2,
    REC_A    = 1 << 3,
    REC_ENUM = 1 << 4,     /* _services._dns-sd._udp.local → _pantheon._tcp.local */
    REC_ALL  = REC_PTR | REC_SRV | REC_TXT | REC_A | REC_ENUM,
};

typedef struct {
    uint8_t *buf;
    size_t   cap, len;
    bool     overflow;
} wire;

static void put_u8(wire *w, uint8_t v)
{
    if (w->len + 1 > w->cap) { w->overflow = true; return; }
    w->buf[w->len++] = v;
}

static void put_u16(wire *w, uint16_t v)
{
    put_u8(w, (uint8_t)(v >> 8));
    put_u8(w, (uint8_t)v);
}

static void put_u32(wire *w, uint32_t v)
{
    put_u16(w, (uint16_t)(v >> 16));
    put_u16(w, (uint16_t)v);
}

/* Dotted name → length-prefixed labels + root. */
static void put_name(wire *w, const char *name)
{
    const char *p = name;

    while (*p) {
        const char *dot = strchr(p, '.');
        size_t n = dot ? (size_t)(dot - p) : strlen(p);
        if (n == 0 || n > PN_MDNS_LABEL_MAX) { w->overflow = true; return; }
        put_u8(w, (uint8_t)n);
        for (size_t i = 0; i < n; i++)
            put_u8(w, (uint8_t)p[i]);
        p += n + (dot ? 1 : 0);
        if (!dot)
            break;
    }
    put_u8(w, 0);
}

static void instance_name(const pn_mdns_service *s, char *out, size_t cap)
{
    snprintf(out, cap, "%s." PN_MDNS_SERVICE, s->label);
}

static void host_name(const pn_mdns_service *s, char *out, size_t cap)
{
    snprintf(out, cap, "%s.local", s->label);
}

/* Write one record. rdlength is patched after the rdata. */
static void put_record(wire *w, const pn_mdns_service *s, int rec, uint32_t ttl, bool flush)
{
    char inst[PN_MDNS_NAME_MAX], host[PN_MDNS_NAME_MAX];
    uint16_t type = rec == REC_SRV ? PN_DNS_TYPE_SRV : rec == REC_TXT ? PN_DNS_TYPE_TXT
                  : rec == REC_A   ? PN_DNS_TYPE_A   : PN_DNS_TYPE_PTR;

    instance_name(s, inst, sizeof(inst));
    host_name(s, host, sizeof(host));
    put_name(w, rec == REC_PTR ? PN_MDNS_SERVICE : rec == REC_ENUM ? PN_MDNS_ENUM
              : rec == REC_A   ? host            : inst);
    put_u16(w, type);
    put_u16(w, (uint16_t)(DNS_CLASS_IN | (flush ? DNS_CACHE_FLUSH : 0)));
    put_u32(w, ttl);
    size_t rdlen_at = w->len;
    put_u16(w, 0);
    size_t start = w->len;
    if (rec == REC_PTR) {
        put_name(w, inst);
    } else if (rec == REC_ENUM) {
        put_name(w, PN_MDNS_SERVICE);
    } else if (rec == REC_SRV) {
        put_u16(w, 0);
        put_u16(w, 0);
        put_u16(w, s->port);
        put_name(w, host);
    } else if (rec == REC_TXT) {
        for (int i = 0; i < s->n_txt; i++) {
            size_t n = strlen(s->txt[i]);
            put_u8(w, (uint8_t)(n > 255 ? 255 : n));
            for (size_t k = 0; k < n && k < 255; k++)
                put_u8(w, (uint8_t)s->txt[i][k]);
        }
        if (s->n_txt == 0)
            put_u8(w, 0);
    } else {
        const uint8_t *ip = (const uint8_t *)&s->ipv4;
        for (int i = 0; i < 4; i++)
            put_u8(w, ip[i]);
    }
    if (!w->overflow) {
        size_t rdlen = w->len - start;
        w->buf[rdlen_at] = (uint8_t)(rdlen >> 8);
        w->buf[rdlen_at + 1] = (uint8_t)rdlen;
    }
}

/* Records in mask, in a fixed order. Returns how many. */
static int put_records(wire *w, const pn_mdns_service *s, int mask, uint32_t ttl, bool legacy)
{
    static const int order[] = { REC_PTR, REC_ENUM, REC_SRV, REC_TXT, REC_A };
    int n = 0;

    for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
        if (!(mask & order[i]))
            continue;
        bool unique = order[i] == REC_SRV || order[i] == REC_TXT || order[i] == REC_A;
        put_record(w, s, order[i], ttl, unique && !legacy);
        n++;
    }
    return n;
}

static void put_header(wire *w, uint16_t id, uint16_t flags, int qd, int an, int ns, int ar)
{
    put_u16(w, id);
    put_u16(w, flags);
    put_u16(w, (uint16_t)qd);
    put_u16(w, (uint16_t)an);
    put_u16(w, (uint16_t)ns);
    put_u16(w, (uint16_t)ar);
}

static void patch_u16(wire *w, size_t at, int v)
{
    if (w->overflow || at + 1 >= w->len)
        return;
    w->buf[at] = (uint8_t)(v >> 8);
    w->buf[at + 1] = (uint8_t)v;
}

/* Which of our records answer (name, type). */
static int records_for(const pn_mdns_service *s, const char *name, uint16_t type)
{
    char inst[PN_MDNS_NAME_MAX], host[PN_MDNS_NAME_MAX];
    bool any = type == PN_DNS_TYPE_ANY;
    int mask = 0;

    instance_name(s, inst, sizeof(inst));
    host_name(s, host, sizeof(host));
    if (strcasecmp(name, PN_MDNS_SERVICE) == 0 && (any || type == PN_DNS_TYPE_PTR))
        mask |= REC_PTR;
    if (strcasecmp(name, PN_MDNS_ENUM) == 0 && (any || type == PN_DNS_TYPE_PTR))
        mask |= REC_ENUM;
    if (strcasecmp(name, inst) == 0) {
        if (any || type == PN_DNS_TYPE_SRV) mask |= REC_SRV;
        if (any || type == PN_DNS_TYPE_TXT) mask |= REC_TXT;
    }
    if (strcasecmp(name, host) == 0 && (any || type == PN_DNS_TYPE_A))
        mask |= REC_A;
    return mask;
}

size_t pn_mdns_build_answer(const pn_mdns_service *s, const pn_mdns_message *q, bool legacy,
                            uint8_t *out, size_t cap)
{
    wire w = { out, cap, 0, false };
    int answers = 0, extra = 0;

    if (q->is_response)
        return 0;
    for (int i = 0; i < q->n_questions; i++)
        answers |= records_for(s, q->questions[i].name, q->questions[i].type);
    if (!answers)
        return 0;
    /* RFC 6763 §12: a PTR answer carries SRV/TXT/A; an SRV carries A. */
    if (answers & REC_PTR) extra |= REC_SRV | REC_TXT | REC_A;
    if (answers & REC_SRV) extra |= REC_A;
    extra &= ~answers;

    put_header(&w, legacy ? q->id : 0, 0x8400, legacy ? q->n_questions : 0, 0, 0, 0);
    if (legacy) {
        for (int i = 0; i < q->n_questions; i++) {
            put_name(&w, q->questions[i].name);
            put_u16(&w, q->questions[i].type);
            put_u16(&w, DNS_CLASS_IN);
        }
    }
    uint32_t ttl = legacy ? 10 : PN_MDNS_TTL;   /* RFC 6762 §6.7: legacy TTL ≤ 10 s */
    int an = put_records(&w, s, answers, ttl, legacy);
    int ar = put_records(&w, s, extra, ttl, legacy);
    patch_u16(&w, 6, an);
    patch_u16(&w, 10, ar);
    return w.overflow ? 0 : w.len;
}

size_t pn_mdns_build_announce(const pn_mdns_service *s, bool goodbye, uint8_t *out, size_t cap)
{
    wire w = { out, cap, 0, false };

    put_header(&w, 0, 0x8400, 0, 0, 0, 0);
    int an = put_records(&w, s, REC_ALL, goodbye ? 0 : PN_MDNS_TTL, false);
    patch_u16(&w, 6, an);
    return w.overflow ? 0 : w.len;
}

size_t pn_mdns_build_probe(const pn_mdns_service *s, uint8_t *out, size_t cap)
{
    wire w = { out, cap, 0, false };
    char inst[PN_MDNS_NAME_MAX], host[PN_MDNS_NAME_MAX];

    instance_name(s, inst, sizeof(inst));
    host_name(s, host, sizeof(host));
    put_header(&w, 0, 0, 2, 0, 0, 0);
    put_name(&w, host);
    put_u16(&w, PN_DNS_TYPE_ANY);
    put_u16(&w, DNS_CLASS_IN | DNS_QU_BIT);
    put_name(&w, inst);
    put_u16(&w, PN_DNS_TYPE_ANY);
    put_u16(&w, DNS_CLASS_IN | DNS_QU_BIT);
    int ns = put_records(&w, s, REC_SRV | REC_TXT | REC_A, PN_MDNS_TTL, true);
    patch_u16(&w, 8, ns);
    return w.overflow ? 0 : w.len;
}

/* Read a possibly-compressed name at *off into out. Advances *off past it. */
static int read_name(const uint8_t *pkt, size_t len, size_t *off, char *out, size_t cap)
{
    size_t pos = *off, n = 0;
    int jumps = 0;
    bool jumped = false;

    for (;;) {
        if (pos >= len)
            return -1;
        uint8_t l = pkt[pos];
        if ((l & 0xC0) == 0xC0) {
            if (pos + 1 >= len || ++jumps > 16)
                return -1;
            if (!jumped)
                *off = pos + 2;
            jumped = true;
            pos = ((size_t)(l & 0x3F) << 8) | pkt[pos + 1];
            continue;
        }
        if (l & 0xC0)
            return -1;
        pos++;
        if (l == 0)
            break;
        if (pos + l > len || n + l + 2 > cap)
            return -1;
        if (n)
            out[n++] = '.';
        memcpy(out + n, pkt + pos, l);
        n += l;
        pos += l;
    }
    out[n] = '\0';
    if (!jumped)
        *off = pos;
    return 0;
}

static uint16_t get_u16(const uint8_t *p)
{
    return (uint16_t)((p[0] << 8) | p[1]);
}

int pn_mdns_parse(const uint8_t *pkt, size_t len, pn_mdns_message *out)
{
    size_t off = 12;

    memset(out, 0, sizeof(*out));
    if (len < 12)
        return -1;
    out->id = get_u16(pkt);
    out->is_response = (pkt[2] & 0x80) != 0;
    int qd = get_u16(pkt + 4);
    int rr = get_u16(pkt + 6) + get_u16(pkt + 8) + get_u16(pkt + 10);
    for (int i = 0; i < qd; i++) {
        pn_mdns_question tmp, *q = out->n_questions < PN_MDNS_MAX_QUESTIONS
                                       ? &out->questions[out->n_questions] : &tmp;
        if (read_name(pkt, len, &off, q->name, sizeof(q->name)) != 0 || off + 4 > len)
            return -1;
        q->type = get_u16(pkt + off);
        q->unicast_response = (get_u16(pkt + off + 2) & DNS_QU_BIT) != 0;
        off += 4;
        if (q != &tmp)
            out->n_questions++;
    }
    for (int i = 0; i < rr; i++) {
        char name[PN_MDNS_NAME_MAX];
        if (read_name(pkt, len, &off, name, sizeof(name)) != 0 || off + 10 > len)
            return -1;
        uint16_t type = get_u16(pkt + off);
        size_t rdlen = get_u16(pkt + off + 8);
        off += 10 + rdlen;
        if (off > len)
            return -1;
        if (out->n_answers < PN_MDNS_MAX_ANSWERS) {
            snprintf(out->answer_names[out->n_answers], PN_MDNS_NAME_MAX, "%s", name);
            out->answer_types[out->n_answers++] = type;
        }
    }
    return 0;
}

bool pn_mdns_conflicts(const pn_mdns_message *m, const pn_mdns_service *s)
{
    char inst[PN_MDNS_NAME_MAX], host[PN_MDNS_NAME_MAX];

    if (!m->is_response)
        return false;
    instance_name(s, inst, sizeof(inst));
    host_name(s, host, sizeof(host));
    for (int i = 0; i < m->n_answers; i++)
        if (strcasecmp(m->answer_names[i], inst) == 0 || strcasecmp(m->answer_names[i], host) == 0)
            return true;
    return false;
}

void pn_mdns_conflict_label(const char *base, int attempt, char *out, size_t cap)
{
    if (attempt <= 1)
        snprintf(out, cap, "%.*s", PN_MDNS_LABEL_MAX, base);
    else
        snprintf(out, cap, "%.*s-%d", PN_MDNS_LABEL_MAX - 4, base, attempt);
}

void pn_mdns_label_for_host(const char *hostname, char *out, size_t cap)
{
    size_t n = (size_t)snprintf(out, cap, "pantheon-");

    for (const char *p = hostname; *p && *p != '.' && n + 1 < cap && n < PN_MDNS_LABEL_MAX; p++) {
        char c = (char)tolower((unsigned char)*p);
        out[n++] = (isalnum((unsigned char)c) || c == '-') ? c : '-';
    }
    out[n < cap ? n : cap - 1] = '\0';
}
