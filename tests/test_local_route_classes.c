/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_route_classes.c — people registry P1.3: route classes and the
 * caller (spec §4.1, §6.1).
 *   - every route of §4.1 has its class;
 *   - the full route x caller matrix through the real auth hook: no key,
 *     inference key, non-admin device (may_load on), non-admin device with
 *     may_load off, admin device (may_load off stored: admins always load) —
 *     each gets the spec's status and error `type`;
 *   - the caller thread-local: set on a pass, cleared at the next request
 *     (also when it is refused or needs no key), not visible on other threads;
 *     the serve principal (person + admin, P5.2) follows it;
 *   - a key whose person is gone gets 401; the per-key bucket still applies.
 * `POST /local/people/<id>` is DEVICE at the hook: the route refuses a
 * non-admin changing anyone but their own name (test_local_people_routes).
 */

#include "local_test_support.h"

#include <fcntl.h>
#include <pthread.h>

#include "../src/local/key_store.h"
#include "../src/local/local_auth.h"
#include "../src/local/local_caller.h"
#include "../src/local/people_store.h"
#include "../src/local/rate_limit.h"

#define STATE "build/test_local/route_classes"

typedef struct {
    const char          *method;
    const char          *path;
    pn_local_route_class cls;
} route_case;

static const route_case k_routes[] = {
    { "OPTIONS", "/v1/models", PN_ROUTE_NONE },
    { "OPTIONS", "/local/people", PN_ROUTE_NONE },
    { "POST", "/local/pair", PN_ROUTE_NONE },
    { "GET", "/", PN_ROUTE_INFERENCE },
    { "GET", "/v1/models", PN_ROUTE_INFERENCE },
    { "POST", "/v1/chat/completions", PN_ROUTE_INFERENCE },
    { "POST", "/v1/session/release", PN_ROUTE_INFERENCE },
    { "POST", "/session/restore", PN_ROUTE_INFERENCE },
    { "POST", "/session/release", PN_ROUTE_INFERENCE },
    { "GET", "/session/estimate?name=x&tokens=10", PN_ROUTE_INFERENCE },
    { "GET", "/session/watch", PN_ROUTE_INFERENCE },
    { "GET", "/local/hardware", PN_ROUTE_DEVICE },
    { "GET", "/local/catalog", PN_ROUTE_DEVICE },
    { "GET", "/local/models/loaded", PN_ROUTE_DEVICE },
    { "GET", "/local/models/install/qwen3-0.6b", PN_ROUTE_DEVICE },
    { "GET", "/local/me", PN_ROUTE_DEVICE },
    { "POST", "/local/me/device", PN_ROUTE_DEVICE },
    { "GET", "/local/devices", PN_ROUTE_DEVICE },
    { "GET", "/local/devices?all=1", PN_ROUTE_DEVICE },
    { "GET", "/local/people", PN_ROUTE_DEVICE },
    { "POST", "/local/people/p00000002", PN_ROUTE_DEVICE },
    { "POST", "/local/invites", PN_ROUTE_DEVICE },
    { "GET", "/local/keys", PN_ROUTE_DEVICE },
    { "GET", "/local/keys?all=1", PN_ROUTE_DEVICE },
    { "POST", "/local/keys", PN_ROUTE_DEVICE },
    { "POST", "/local/keys/revoke", PN_ROUTE_DEVICE },
    { "POST", "/local/hosts", PN_ROUTE_DEVICE },
    { "GET", "/local/hosts/c-0123456789ab/wait?since=0", PN_ROUTE_DEVICE },
    { "POST", "/local/hosts/c-0123456789ab/delete", PN_ROUTE_DEVICE },
    { "GET", "/local/sessions", PN_ROUTE_DEVICE },
    { "GET", "/local/sessions?rev=4", PN_ROUTE_DEVICE },
    { "POST", "/local/sessions/c-0123456789ab/share", PN_ROUTE_DEVICE },
    { "POST", "/local/sessions/c-0123456789ab/attach", PN_ROUTE_DEVICE },
    { "POST", "/local/relay/c-0123456789ab", PN_ROUTE_DEVICE },
    { "POST", "/local/relay/r-0011223344556677/host", PN_ROUTE_DEVICE },
    { "POST", "/local/models/install", PN_ROUTE_LOADER },
    { "POST", "/local/models/load", PN_ROUTE_LOADER },
    { "POST", "/local/people", PN_ROUTE_ADMIN },
    { "POST", "/local/people/remove", PN_ROUTE_ADMIN },
    { "GET", "/local/seats/people", PN_ROUTE_ADMIN },
    { "POST", "/local/models/uninstall", PN_ROUTE_ADMIN },
    { "POST", "/local/catalog/refresh", PN_ROUTE_ADMIN },
    { "POST", "/admin/models/load", PN_ROUTE_ADMIN },
    { "GET", "/admin/models", PN_ROUTE_ADMIN },
    { "GET", "/admin/dies", PN_ROUTE_ADMIN },
    { "POST", "/admin/models/unload", PN_ROUTE_LOADER },
    { "POST", "/admin/session/export", PN_ROUTE_INFERENCE },
    { "GET", "/local/pair", PN_ROUTE_ADMIN },
    { "GET", "/local/people/p00000002", PN_ROUTE_ADMIN },
    { "GET", "/local/unknown", PN_ROUTE_ADMIN },
    { "GET", "/v1x", PN_ROUTE_ADMIN },
    { "GET", "/local/meadow", PN_ROUTE_ADMIN },
};

typedef enum { C_NONE, C_INFERENCE, C_DEVICE, C_DEVICE_NO_LOAD, C_ADMIN, C_COUNT } caller_kind;

static const char *k_caller_names[C_COUNT] = { "no key", "inference key", "device",
                                               "device (may_load off)", "admin device" };

/* What the spec says caller `c` gets on a route of class `cls`: 0 = passes,
 * else the status, with the error type in *type. */
static int expected(caller_kind c, pn_local_route_class cls, const char **type)
{
    *type = "";
    if (cls == PN_ROUTE_NONE)
        return 0;
    if (c == C_NONE) {
        *type = "unauthorized";
        return 401;
    }
    *type = "forbidden";
    if (c == C_INFERENCE)
        return cls == PN_ROUTE_INFERENCE ? 0 : 403;
    if (cls == PN_ROUTE_ADMIN)
        return c == C_ADMIN ? 0 : 403;
    if (cls == PN_ROUTE_LOADER && c == C_DEVICE_NO_LOAD) {
        *type = "may_load_off";
        return 403;
    }
    return 0;
}

/* Run the hook on a socketpair; returns 0 when it passed, else the status
 * sent, with the error type copied to type_out. */
static int run_hook(pn_local_ctx *ctx, const char *method, const char *path, const char *key,
                    char *type_out, size_t type_cap)
{
    int sv[2] = { -1, -1 }, status = 0;
    char auth[128] = "", buf[4096] = "";
    hs_request req = {0};

    type_out[0] = '\0';
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return -1;
    if (key)
        snprintf(auth, sizeof(auth), "Bearer %s", key);
    req.method = (char *)method;
    req.path = (char *)path;
    req.authorization = key ? auth : NULL;
    if (pn_local_auth_check(sv[0], &req, ctx) != 0) {
        close(sv[0]);
        sv[0] = -1;
        ssize_t n = read(sv[1], buf, sizeof(buf) - 1);
        buf[n > 0 ? n : 0] = '\0';
        if (sscanf(buf, "HTTP/1.1 %d", &status) != 1)
            status = -1;
        const char *t = strstr(buf, "\"type\":\"");
        if (t)
            sscanf(t + 8, "%63[^\"]", type_out);
        (void)type_cap;
    }
    if (sv[0] >= 0)
        close(sv[0]);
    close(sv[1]);
    return status;
}

static pn_people *g_people;
static pn_key_store *g_keys;
static char g_secret[C_COUNT][PN_KEY_SECRET_MAX];
static pn_person g_admin, g_dev, g_noload;

static void setup(pn_local_ctx *ctx)
{
    pn_person_change off = { NULL, NULL, 0, -1 };

    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    g_people = pn_people_open(STATE "/people.json");
    g_keys = pn_key_store_open(STATE "/keys.json");
    pn_people_add(g_people, "admin", "Admin", 1, NULL, 0, &g_admin);   /* may_load off stored */
    pn_people_add(g_people, "dev", "Dev", 0, NULL, 1, &g_dev);
    pn_people_add(g_people, "noload", "No Load", 0, NULL, 1, &g_noload);
    pn_people_update(g_people, g_admin.id, g_noload.id, &off, &g_noload);
    pn_key_store_mint_inference(g_keys, g_dev.id, "scripts", g_secret[C_INFERENCE], NULL);
    pn_key_store_mint_device(g_keys, g_dev.id, "d0000000000000001", "dev-mac", "argus",
                             g_secret[C_DEVICE], NULL);
    pn_key_store_mint_device(g_keys, g_noload.id, "d0000000000000002", "noload-mac", "argus",
                             g_secret[C_DEVICE_NO_LOAD], NULL);
    pn_key_store_mint_device(g_keys, g_admin.id, "d0000000000000003", "admin-mac", "argus",
                             g_secret[C_ADMIN], NULL);
    memset(ctx, 0, sizeof(*ctx));
    ctx->people = g_people;
    ctx->keys = g_keys;
}

static void test_class_table(void)
{
    int wrong = 0;

    printf("class_table:\n");
    for (size_t i = 0; i < sizeof(k_routes) / sizeof(k_routes[0]); i++) {
        pn_local_route_class got = pn_local_route_class_of(k_routes[i].method, k_routes[i].path);
        if (got != k_routes[i].cls) {
            fprintf(stderr, "  %s %s: %s, spec %s\n", k_routes[i].method, k_routes[i].path,
                    pn_local_route_class_name(got), pn_local_route_class_name(k_routes[i].cls));
            wrong++;
        }
    }
    CHECK(wrong == 0, "every route has its spec class");
}

static void test_matrix(pn_local_ctx *ctx)
{
    int cells = 0, wrong = 0;
    char type[64];

    printf("matrix:\n");
    for (size_t i = 0; i < sizeof(k_routes) / sizeof(k_routes[0]); i++) {
        for (int c = 0; c < C_COUNT; c++) {
            const char *want_type = NULL;
            int want = expected((caller_kind)c, k_routes[i].cls, &want_type);
            int got = run_hook(ctx, k_routes[i].method, k_routes[i].path,
                               c == C_NONE ? NULL : g_secret[c], type, sizeof(type));
            cells++;
            if (got != want || (want && strcmp(type, want_type) != 0)) {
                fprintf(stderr, "  %s %s as %s: %d %s, spec %d %s\n", k_routes[i].method,
                        k_routes[i].path, k_caller_names[c], got, type, want, want_type);
                wrong++;
            }
        }
    }
    printf("  %d cells\n", cells);
    CHECK(cells == (int)(sizeof(k_routes) / sizeof(k_routes[0])) * C_COUNT, "all cells ran");
    CHECK(wrong == 0, "every cell matches the spec");
}

static void *other_thread(void *arg)
{
    *(int *)arg = pn_local_caller_get() == NULL;
    return NULL;
}

static void test_caller(pn_local_ctx *ctx)
{
    char type[64];
    const pn_local_caller *c = NULL;
    pthread_t t;
    int clean = 0;

    printf("caller:\n");
    CHECK(run_hook(ctx, "GET", "/local/me", g_secret[C_ADMIN], type, sizeof(type)) == 0,
          "admin passes");
    c = pn_local_caller_get();
    CHECK(c && strcmp(c->person, g_admin.id) == 0 && c->admin && c->may_load &&
          c->kind == PN_KEY_KIND_DEVICE && strcmp(c->device_id, "d0000000000000003") == 0,
          "caller = the admin's person and device (admin may load)");
    CHECK(hs_thread_principal() && strcmp(hs_thread_principal(), g_admin.id) == 0 &&
          hs_thread_principal_admin(), "the serve principal = the admin's person, admin");
    pthread_create(&t, NULL, other_thread, &clean);
    pthread_join(t, NULL);
    CHECK(clean, "another thread sees no caller");
    CHECK(run_hook(ctx, "GET", "/local/me", NULL, type, sizeof(type)) == 401, "no key 401");
    CHECK(pn_local_caller_get() == NULL && hs_thread_principal() == NULL &&
          !hs_thread_principal_admin(), "a refused request clears the caller and principal");
    run_hook(ctx, "GET", "/v1/models", g_secret[C_INFERENCE], type, sizeof(type));
    c = pn_local_caller_get();
    CHECK(c && c->kind == PN_KEY_KIND_INFERENCE && strcmp(c->person, g_dev.id) == 0 &&
          !c->admin && c->device_id[0] == '\0', "inference caller");
    CHECK(hs_thread_principal() && strcmp(hs_thread_principal(), g_dev.id) == 0 &&
          !hs_thread_principal_admin(), "an inference key's person is the principal");
    run_hook(ctx, "POST", "/local/pair", NULL, type, sizeof(type));
    CHECK(pn_local_caller_get() == NULL, "a no-key route clears the caller");
    run_hook(ctx, "GET", "/local/me", g_secret[C_DEVICE_NO_LOAD], type, sizeof(type));
    c = pn_local_caller_get();
    CHECK(c && !c->may_load && !c->admin, "may_load off is carried");
}

static void test_gone_person_and_bucket(pn_local_ctx *ctx)
{
    char type[64];
    int ok = 0;

    printf("gone_person_and_bucket:\n");
    CHECK(pn_people_remove(g_people, g_admin.id, g_noload.id) == PN_PEOPLE_OK, "remove person");
    CHECK(run_hook(ctx, "GET", "/local/me", g_secret[C_DEVICE_NO_LOAD], type, sizeof(type)) ==
          401 && strcmp(type, "unauthorized") == 0, "a removed person's key gets 401");
    ctx->limiter = pn_rate_limiter_new(1.0, 3.0);
    for (int i = 0; i < 4; i++)
        ok += run_hook(ctx, "GET", "/local/me", g_secret[C_DEVICE], type, sizeof(type)) == 0;
    CHECK(ok == 3 && strcmp(type, "rate_limited") == 0, "the per-key bucket applies");
    pn_rate_limiter_free(ctx->limiter);
    ctx->limiter = NULL;
}

int main(void)
{
    pn_local_ctx ctx;

    signal(SIGPIPE, SIG_IGN);
    setup(&ctx);
    test_class_table();
    test_matrix(&ctx);
    test_caller(&ctx);
    test_gone_person_and_bucket(&ctx);
    pn_key_store_close(g_keys);
    pn_people_close(g_people);
    return lt_report("test_local_route_classes");
}
