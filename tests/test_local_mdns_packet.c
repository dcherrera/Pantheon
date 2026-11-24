/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_mdns_packet.c — Pantheon Local D1/D4: mDNS wire format.
 * A PTR query for _pantheon._tcp is answered with PTR + SRV/TXT/A; an A query
 * only with A; unrelated names get nothing; a legacy (non-5353) query echoes
 * the id and question; compressed names parse; a probe carries our names; a
 * response that claims our name is a conflict and the next label is "-2".
 */

#include "local_test_support.h"

#include "../src/local/mdns_packet.h"

static pn_mdns_service service(void)
{
    static const char *txt[] = { "txtvers=1", "port=8400", "cards=k80x8", "pair=yes" };
    pn_mdns_service s;
    memset(&s, 0, sizeof(s));
    snprintf(s.label, sizeof(s.label), "pantheon-lab-node-1");
    s.port = 8400;
    s.ipv4 = inet_addr("192.0.2.41");
    for (int i = 0; i < 4; i++)
        s.txt[i] = txt[i];
    s.n_txt = 4;
    return s;
}

/* A one-question query, hand-assembled (uncompressed name). */
static size_t make_query(uint8_t *out, uint16_t id, const char *name, uint16_t type)
{
    size_t n = 0;
    const uint8_t hdr[12] = { (uint8_t)(id >> 8), (uint8_t)id, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    memcpy(out, hdr, 12);
    n = 12;
    for (const char *p = name; *p;) {
        const char *dot = strchr(p, '.');
        size_t l = dot ? (size_t)(dot - p) : strlen(p);
        out[n++] = (uint8_t)l;
        memcpy(out + n, p, l);
        n += l;
        p += l + (dot ? 1 : 0);
    }
    out[n++] = 0;
    out[n++] = (uint8_t)(type >> 8);
    out[n++] = (uint8_t)type;
    out[n++] = 0;
    out[n++] = 1;
    return n;
}

static int has_answer(const pn_mdns_message *m, const char *name, uint16_t type)
{
    for (int i = 0; i < m->n_answers; i++)
        if (strcasecmp(m->answer_names[i], name) == 0 && m->answer_types[i] == type)
            return 1;
    return 0;
}

static void test_ptr_query(void)
{
    pn_mdns_service s = service();
    uint8_t q[512], out[1500];
    pn_mdns_message qm, am;

    printf("ptr_query:\n");
    size_t qn = make_query(q, 0, "_pantheon._tcp.local", PN_DNS_TYPE_PTR);
    CHECK(pn_mdns_parse(q, qn, &qm) == 0 && qm.n_questions == 1 && !qm.is_response, "parse query");
    size_t n = pn_mdns_build_answer(&s, &qm, false, out, sizeof(out));
    CHECK(n > 12, "answered");
    CHECK(pn_mdns_parse(out, n, &am) == 0 && am.is_response, "answer parses as a response");
    CHECK(has_answer(&am, "_pantheon._tcp.local", PN_DNS_TYPE_PTR), "PTR");
    CHECK(has_answer(&am, "pantheon-lab-node-1._pantheon._tcp.local", PN_DNS_TYPE_SRV), "SRV");
    CHECK(has_answer(&am, "pantheon-lab-node-1._pantheon._tcp.local", PN_DNS_TYPE_TXT), "TXT");
    CHECK(has_answer(&am, "pantheon-lab-node-1.local", PN_DNS_TYPE_A), "A");
    CHECK(am.id == 0 && (out[2] & 0x04), "multicast id 0, authoritative");
    int an = (out[6] << 8) | out[7];
    CHECK(an == 1, "PTR in answers, the rest additional");
    CHECK(memmem(out, n, "cards=k80x8", 11) != NULL, "TXT carries cards=k80x8");
    const uint8_t ip[4] = { 192, 0, 2, 41 };
    CHECK(memmem(out, n, ip, 4) != NULL, "A carries the address");
}

static void test_a_query_and_miss(void)
{
    pn_mdns_service s = service();
    uint8_t q[512], out[1500];
    pn_mdns_message qm, am;

    printf("a_query_and_miss:\n");
    size_t qn = make_query(q, 0, "PANTHEON-LAB-NODE-1.local", PN_DNS_TYPE_A);
    pn_mdns_parse(q, qn, &qm);
    size_t n = pn_mdns_build_answer(&s, &qm, false, out, sizeof(out));
    CHECK(n > 0 && pn_mdns_parse(out, n, &am) == 0, "A answered (case-insensitive)");
    CHECK(am.n_answers == 1 && am.answer_types[0] == PN_DNS_TYPE_A, "only A");
    qn = make_query(q, 0, "other-box.local", PN_DNS_TYPE_A);
    pn_mdns_parse(q, qn, &qm);
    CHECK(pn_mdns_build_answer(&s, &qm, false, out, sizeof(out)) == 0, "other names ignored");
    qn = make_query(q, 0, "_pantheon._tcp.local", PN_DNS_TYPE_PTR);
    pn_mdns_parse(q, qn, &qm);
    qm.is_response = true;
    CHECK(pn_mdns_build_answer(&s, &qm, false, out, sizeof(out)) == 0, "responses ignored");
}

static void test_legacy_query(void)
{
    pn_mdns_service s = service();
    uint8_t q[512], out[1500];
    pn_mdns_message qm, am;

    printf("legacy_query:\n");
    size_t qn = make_query(q, 0x4242, "pantheon-lab-node-1._pantheon._tcp.local", PN_DNS_TYPE_SRV);
    pn_mdns_parse(q, qn, &qm);
    size_t n = pn_mdns_build_answer(&s, &qm, true, out, sizeof(out));
    CHECK(n > 0 && pn_mdns_parse(out, n, &am) == 0, "legacy answered");
    CHECK(am.id == 0x4242 && am.n_questions == 1, "id and question echoed");
    CHECK(has_answer(&am, "pantheon-lab-node-1.local", PN_DNS_TYPE_A), "SRV brings A");
}

static void test_compressed_name(void)
{
    /* Query for _pantheon._tcp.local PTR + a second question using a pointer
     * back to "_tcp.local" at offset 12+10. */
    uint8_t q[512];
    pn_mdns_message m;

    printf("compressed_name:\n");
    size_t n = make_query(q, 0, "_pantheon._tcp.local", PN_DNS_TYPE_PTR);
    q[5] = 2;
    const uint8_t second[] = { 3, 'f', 'o', 'o', 0xC0, 22, 0, 12, 0, 1 };
    memcpy(q + n, second, sizeof(second));
    n += sizeof(second);
    CHECK(pn_mdns_parse(q, n, &m) == 0 && m.n_questions == 2, "two questions");
    CHECK(strcmp(m.questions[1].name, "foo._tcp.local") == 0, "pointer followed");
    q[n - 6] = 0xC0;
    q[n - 5] = (uint8_t)(n - 6);   /* points at itself: a loop */
    CHECK(pn_mdns_parse(q, n, &m) == -1, "pointer loop refused");
    CHECK(pn_mdns_parse(q, 5, &m) == -1, "short packet refused");
}

static void test_probe_and_conflict(void)
{
    pn_mdns_service s = service();
    uint8_t out[1500];
    pn_mdns_message m;
    char label[80];

    printf("probe_and_conflict:\n");
    size_t n = pn_mdns_build_probe(&s, out, sizeof(out));
    CHECK(n > 0 && pn_mdns_parse(out, n, &m) == 0, "probe parses");
    CHECK(!m.is_response && m.n_questions == 2, "probe = 2 questions");
    CHECK(m.questions[0].type == PN_DNS_TYPE_ANY && m.questions[0].unicast_response, "ANY + QU");
    CHECK(!pn_mdns_conflicts(&m, &s), "our own probe is no conflict");
    n = pn_mdns_build_announce(&s, false, out, sizeof(out));
    pn_mdns_parse(out, n, &m);
    CHECK(pn_mdns_conflicts(&m, &s), "a response claiming our name conflicts");
    pn_mdns_service other = s;
    snprintf(other.label, sizeof(other.label), "pantheon-lab-node-2");
    CHECK(!pn_mdns_conflicts(&m, &other), "different name no conflict");
    pn_mdns_conflict_label("pantheon-lab-node-1", 1, label, sizeof(label));
    CHECK(strcmp(label, "pantheon-lab-node-1") == 0, "attempt 1 = base");
    pn_mdns_conflict_label("pantheon-lab-node-1", 2, label, sizeof(label));
    CHECK(strcmp(label, "pantheon-lab-node-1-2") == 0, "attempt 2 = base-2 (not node-2)");
    pn_mdns_label_for_host("Lab_Node-1.local", label, sizeof(label));
    CHECK(strcmp(label, "pantheon-lab-node-1") == 0, "label from hostname");
    n = pn_mdns_build_announce(&s, true, out, sizeof(out));
    CHECK(n > 0 && memmem(out, n, "\x00\x00\x00\x00", 4) != NULL, "goodbye has ttl 0");
}

int main(void)
{
    test_ptr_query();
    test_a_query_and_miss();
    test_legacy_query();
    test_compressed_name();
    test_probe_and_conflict();
    return lt_report("test_local_mdns_packet");
}
