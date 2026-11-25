/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for `pantheon serve --gateway`'s pure pieces: the API route
 * allow-list, the gateway key (parse + constant-time match), tenant-scoped
 * session keys, response ids that never repeat, and the `[usage]` line.
 * Pure C, libc-only; no model.
 */

#include "lib/test.h"

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/core/session_family.h"
#include "../src/serve/gateway_auth.h"
#include "../src/serve/gateway_mode.h"
#include "../src/serve/gateway_request_gate.h"
#include "../src/serve/request_id.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/usage_record.h"

static void test_routes_allow_only_the_openai_surface(void)
{
    ASSERT_TRUE(pn_gateway_route_allowed("POST", "/v1/chat/completions"));
    ASSERT_TRUE(pn_gateway_route_allowed("GET", "/v1/models"));
    ASSERT_TRUE(pn_gateway_route_allowed("GET", "/health/live"));
    ASSERT_TRUE(pn_gateway_route_allowed("GET", "/health/ready?x=1"));
    static const char *const refused[][2] = {
        { "POST", "/admin/models/load" }, { "POST", "/admin/models/unload" },
        { "GET", "/admin/models" },       { "GET", "/admin/dies" },
        { "POST", "/admin/session/export" }, { "POST", "/session/restore" },
        { "POST", "/session/release" },   { "POST", "/v1/session/release" },
        { "GET", "/session/watch?session_id=a" }, { "GET", "/session/estimate" },
        { "GET", "/" }, { "GET", "/v1/chat/completions" },
        { "POST", "/v1/models" }, { "OPTIONS", "/v1/chat/completions" },
        { "POST", "/v1/chat/completions/x" }, { "GET", "/local/hosts" },
    };
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); i++)
        ASSERT_FALSE(pn_gateway_route_allowed(refused[i][0], refused[i][1]));
}

static void test_key_parse(void)
{
    pn_gateway_key k;
    char err[128];
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "  s3cret-key\n", 13, err, sizeof(err)), 0);
    ASSERT_INT_EQ((int)k.len, 10);
    ASSERT_TRUE(memcmp(k.key, "s3cret-key", 10) == 0);
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "\n \t", 3, err, sizeof(err)), -1);
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "a b", 3, err, sizeof(err)), -1);
    char big[PN_GATEWAY_KEY_MAX + 2];
    memset(big, 'k', sizeof(big));
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, big, sizeof(big), err, sizeof(err)), -1);
    ASSERT_TRUE(strstr(err, "kkkk") == NULL);   /* never echo key bytes */
}

static void test_key_matches_bearer_exactly(void)
{
    pn_gateway_key k;
    char err[128];
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "abc123", 6, err, sizeof(err)), 0);
    ASSERT_TRUE(pn_gateway_key_matches(&k, "Bearer abc123"));
    ASSERT_TRUE(pn_gateway_key_matches(&k, "bearer abc123"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Bearer abc12"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Bearer abc1234"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Bearer abd123"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Basic abc123"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "abc123"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, ""));
    ASSERT_FALSE(pn_gateway_key_matches(&k, NULL));
    pn_gateway_key_clear(&k);
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Bearer abc123"));
    ASSERT_FALSE(pn_gateway_key_matches(&k, "Bearer "));
}

static void test_gateway_authorized_only_when_enabled(void)
{
    pn_gateway_key k;
    char err[128];
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "zz9", 3, err, sizeof(err)), 0);
    ASSERT_FALSE(pn_gateway_enabled());
    ASSERT_FALSE(pn_gateway_authorized("Bearer zz9"));
    pn_gateway_enable(&k);
    ASSERT_TRUE(pn_gateway_enabled());
    ASSERT_TRUE(pn_gateway_authorized("Bearer zz9"));
    ASSERT_FALSE(pn_gateway_authorized("Bearer zz8"));
    pn_gateway_disable();
    ASSERT_FALSE(pn_gateway_enabled());
    ASSERT_FALSE(pn_gateway_authorized("Bearer zz9"));
}

static void test_tenant_validation(void)
{
    ASSERT_TRUE(pn_gateway_tenant_valid("acme"));
    ASSERT_TRUE(pn_gateway_tenant_valid("org_1.prod-eu"));
    ASSERT_TRUE(pn_gateway_tenant_valid("123e4567-e89b-12d3-a456-426614174000"));
    ASSERT_FALSE(pn_gateway_tenant_valid(NULL));
    ASSERT_FALSE(pn_gateway_tenant_valid(""));
    ASSERT_FALSE(pn_gateway_tenant_valid("a:b"));
    ASSERT_FALSE(pn_gateway_tenant_valid("a b"));
    char longt[PN_GATEWAY_TENANT_MAX + 2];
    memset(longt, 't', sizeof(longt) - 1);
    longt[sizeof(longt) - 1] = '\0';
    ASSERT_FALSE(pn_gateway_tenant_valid(longt));
}

/* The same client key under two tenants is two sessions, never one family;
 * related keys of ONE tenant still form a family as they do for home use. */
static void test_session_key_scoped_to_tenant(void)
{
    char a[PN_GATEWAY_SESSION_KEY_MAX], b[PN_GATEWAY_SESSION_KEY_MAX],
         c[PN_GATEWAY_SESSION_KEY_MAX];
    ASSERT_NOT_NULL(pn_gateway_session_key("tenantA", "chat-1", a, sizeof(a)));
    ASSERT_NOT_NULL(pn_gateway_session_key("tenantB", "chat-1", b, sizeof(b)));
    ASSERT_STR_EQ(a, "tenantA:chat-1");
    ASSERT_FALSE(pn_session_same_family(a, b));
    /* Related tenant ids (tenantA-1, tenantA-1-x) never form one family. */
    ASSERT_NOT_NULL(pn_gateway_session_key("tenantA-1", "k", a, sizeof(a)));
    ASSERT_NOT_NULL(pn_gateway_session_key("tenantA-1-x", "k", b, sizeof(b)));
    ASSERT_FALSE(pn_session_same_family(a, b));
    ASSERT_NOT_NULL(pn_gateway_session_key("t", "k", a, sizeof(a)));
    ASSERT_NOT_NULL(pn_gateway_session_key("t", "k-x", b, sizeof(b)));
    ASSERT_TRUE(pn_session_same_family(a, b));
    /* No tenant, a bad tenant, or no key: anonymous. */
    ASSERT_NULL(pn_gateway_session_key(NULL, "k", a, sizeof(a)));
    ASSERT_NULL(pn_gateway_session_key("a:b", "k", a, sizeof(a)));
    ASSERT_NULL(pn_gateway_session_key("t", "", a, sizeof(a)));
    ASSERT_NULL(pn_gateway_session_key("t", NULL, a, sizeof(a)));
    /* Two long keys sharing a 63-byte prefix never collide by truncation. */
    char k1[128], k2[128];
    memset(k1, 'q', 100); k1[100] = '\0';
    memcpy(k2, k1, sizeof(k1)); k2[99] = 'r';
    ASSERT_NOT_NULL(pn_gateway_session_key("t", k1, a, sizeof(a)));
    ASSERT_NOT_NULL(pn_gateway_session_key("t", k2, c, sizeof(c)));
    ASSERT_TRUE(strcmp(a, c) != 0);
    ASSERT_TRUE(strncmp(a, "t:#", 3) == 0);
    ASSERT_TRUE(strlen(a) < PN_GATEWAY_SESSION_KEY_MAX);
}

static int is_base62(const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        char ch = s[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') ||
              (ch >= 'A' && ch <= 'Z')))
            return 0;
    }
    return 1;
}

static void test_request_ids_are_long_and_distinct(void)
{
    char ids[64][48];
    for (int i = 0; i < 64; i++) {
        pn_request_id_new("chatcmpl-", ids[i], sizeof(ids[i]));
        ASSERT_INT_EQ((int)strlen(ids[i]), 9 + PN_REQUEST_ID_RANDOM_CHARS);
        ASSERT_TRUE(strncmp(ids[i], "chatcmpl-", 9) == 0);
        ASSERT_TRUE(is_base62(ids[i] + 9, PN_REQUEST_ID_RANDOM_CHARS));
        for (int j = 0; j < i; j++)
            ASSERT_TRUE(strcmp(ids[i], ids[j]) != 0);
    }
    char small[12];
    pn_request_id_new("chatcmpl-", small, sizeof(small));
    ASSERT_INT_EQ((int)strlen(small), 11);
}

/* Tool-call ids come from the same source: `call_` + 24 base62, distinct. On
 * main they were `call_` + 8 hex from rand(), replayed after every restart. */
static void test_tool_call_ids_are_long_and_distinct(void)
{
    char a[TC_CALL_ID_MAX], b[TC_CALL_ID_MAX];
    tc_new_call_id(a, sizeof(a));
    tc_new_call_id(b, sizeof(b));
    ASSERT_INT_EQ((int)strlen(a), 5 + PN_REQUEST_ID_RANDOM_CHARS);
    ASSERT_TRUE(strncmp(a, "call_", 5) == 0);
    ASSERT_TRUE(is_base62(a + 5, PN_REQUEST_ID_RANDOM_CHARS));
    ASSERT_TRUE(strcmp(a, b) != 0);
}

static void test_usage_line_format(void)
{
    char line[512];
    pn_usage_counts u = { .prompt = 812, .cached = 640, .completion = 96, .choices = 2 };
    pn_usage_line_format(line, sizeof(line), "chatcmpl-X", "acme", "qwen", &u,
                         "client_gone");
    ASSERT_STR_EQ(line, "[usage] id=chatcmpl-X tenant=acme model=qwen prompt=812 "
                        "cached=640 completion=96 choices=2 finish=client_gone");
    pn_usage_line_format(line, sizeof(line), "chatcmpl-Y", NULL, NULL, NULL, NULL);
    ASSERT_STR_EQ(line, "[usage] id=chatcmpl-Y tenant=- model=- prompt=0 cached=0 "
                        "completion=0 choices=0 finish=-");
}

/* Gate one request over a socket pair; the reply's status line lands in `out`. */
static int gate(const char *method, const char *path, const char *auth,
                const char *tenant, char *out, size_t cap)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return -2;
    hs_request req = { .method = (char *)method, .path = (char *)path,
                       .authorization = (char *)auth, .tenant = (char *)tenant };
    int rc = pn_gateway_gate_request(sv[1], &req);
    close(sv[1]);
    ssize_t n = read(sv[0], out, cap - 1);
    out[n > 0 ? n : 0] = '\0';
    close(sv[0]);
    return rc;
}

/* The API listener in gateway mode: unknown routes look absent (404, checked
 * before the key), then the bearer key (401), then the tenant (400). */
static void test_gate_route_key_tenant(void)
{
    pn_gateway_key k;
    char err[128], out[1024];
    ASSERT_INT_EQ(pn_gateway_key_parse(&k, "gw-key", 6, err, sizeof(err)), 0);
    pn_gateway_enable(&k);

    ASSERT_TRUE(gate("POST", "/admin/models/load", "Bearer gw-key", NULL, out, sizeof(out)) != 0);
    ASSERT_TRUE(strncmp(out, "HTTP/1.1 404", 12) == 0);
    ASSERT_TRUE(gate("POST", "/v1/session/release", NULL, NULL, out, sizeof(out)) != 0);
    ASSERT_TRUE(strncmp(out, "HTTP/1.1 404", 12) == 0);
    ASSERT_TRUE(gate("POST", "/v1/chat/completions", NULL, NULL, out, sizeof(out)) != 0);
    ASSERT_TRUE(strncmp(out, "HTTP/1.1 401", 12) == 0);
    ASSERT_TRUE(strstr(out, "\"code\":\"invalid_api_key\"") != NULL);
    ASSERT_TRUE(strstr(out, "gw-key") == NULL);
    ASSERT_TRUE(gate("GET", "/v1/models", "Bearer wrong", NULL, out, sizeof(out)) != 0);
    ASSERT_TRUE(strncmp(out, "HTTP/1.1 401", 12) == 0);
    ASSERT_TRUE(gate("POST", "/v1/chat/completions", "Bearer gw-key", "a:b", out,
                     sizeof(out)) != 0);
    ASSERT_TRUE(strncmp(out, "HTTP/1.1 400", 12) == 0);
    ASSERT_INT_EQ(gate("POST", "/v1/chat/completions", "Bearer gw-key", "acme", out,
                       sizeof(out)), 0);
    ASSERT_STR_EQ(out, "");                       /* nothing sent: routed */
    ASSERT_STR_EQ(pn_gateway_thread_tenant(), "acme");
    ASSERT_INT_EQ(gate("GET", "/health/ready", "Bearer gw-key", NULL, out, sizeof(out)), 0);
    ASSERT_NULL(pn_gateway_thread_tenant());
    pn_gateway_disable();
}

int main(void)
{
    printf("=== gateway mode ===\n\n");
    RUN_TEST(routes_allow_only_the_openai_surface);
    RUN_TEST(key_parse);
    RUN_TEST(key_matches_bearer_exactly);
    RUN_TEST(gateway_authorized_only_when_enabled);
    RUN_TEST(tenant_validation);
    RUN_TEST(session_key_scoped_to_tenant);
    RUN_TEST(request_ids_are_long_and_distinct);
    RUN_TEST(tool_call_ids_are_long_and_distinct);
    RUN_TEST(usage_line_format);
    RUN_TEST(gate_route_key_tenant);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
