/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_keys.c — Pantheon Local C1-C5: keys and pairing.
 *   - key store keeps only SHA-256 hashes, file mode 0600, survives reopen;
 *   - revoke, kinds, route → class table (full matrix: test_local_route_classes);
 *   - pairing code rotates after use, after expiry, after 5 wrong tries;
 *   - per-key token bucket;
 *   - over HTTP against a forked `pantheon local`: no key / wrong key → 401,
 *     inference key on an admin route → 403, pairing mints an admin key,
 *     admin mints an inference key, and a REVOKED key fails (red arm).
 */

#include "local_test_support.h"

#include "../src/local/key_store.h"
#include "../src/local/local_auth.h"
#include "../src/local/pairing.h"
#include "../src/local/rate_limit.h"

#define STATE "build/test_local/keys"

static void test_key_store(void)
{
    char secret[PN_KEY_SECRET_MAX], admin[PN_KEY_SECRET_MAX], *text = NULL;
    pn_key_info info, got;
    struct stat sb;

    printf("key_store:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    pn_key_store *ks = pn_key_store_open(STATE "/keys.json");
    CHECK(ks != NULL, "open empty store");
    CHECK(pn_key_store_mint_device(ks, "p00000001", NULL, "app", "app", admin, &info) == 0,
          "mint device");
    CHECK(strncmp(admin, "pnk_", 4) == 0 && strlen(admin) == 68, "secret is pnk_ + 64 hex");
    CHECK(pn_key_store_mint_inference(ks, "p00000001", "editor", secret, &info) == 0,
          "mint inference");
    CHECK(stat(STATE "/keys.json", &sb) == 0 && (sb.st_mode & 0777) == 0600, "keys.json 0600");
    FILE *f = fopen(STATE "/keys.json", "r");
    if (f) {
        char buf[8192];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = '\0';
        fclose(f);
        text = strdup(buf);
    }
    CHECK(text && !strstr(text, secret + 4) && !strstr(text, admin + 4), "no secret on disk");
    CHECK(text && strstr(text, "\"sha256\""), "hashes on disk");
    free(text);
    pn_key_store_close(ks);

    ks = pn_key_store_open(STATE "/keys.json");
    CHECK(pn_key_store_verify(ks, secret, &got) == 0 && got.kind == PN_KEY_KIND_INFERENCE,
          "inference key verifies after reopen");
    CHECK(pn_key_store_verify(ks, "pnk_0000", &got) != 0, "unknown key refused");
    CHECK(pn_key_store_revoke(ks, info.id) == 0, "revoke");
    CHECK(pn_key_store_verify(ks, secret, &got) != 0, "revoked key refused");
    CHECK(pn_key_store_verify(ks, admin, &got) == 0 && got.kind == PN_KEY_KIND_DEVICE,
          "device key still valid");
    pn_key_info rows[8];
    CHECK(pn_key_store_list(ks, rows, 8) == 1, "one key left");
    pn_key_store_close(ks);
}

static void test_route_roles(void)
{
    printf("route_roles:\n");
    CHECK(pn_local_route_class_of("POST", "/local/pair") == PN_ROUTE_NONE, "pair needs no key");
    CHECK(pn_local_route_class_of("GET", "/local/pair") == PN_ROUTE_ADMIN, "GET pair is admin");
    CHECK(pn_local_route_class_of("OPTIONS", "/v1/models") == PN_ROUTE_NONE, "preflight");
    CHECK(pn_local_route_class_of("POST", "/v1/chat/completions") == PN_ROUTE_INFERENCE, "chat");
    CHECK(pn_local_route_class_of("GET", "/session/estimate?x=1") == PN_ROUTE_INFERENCE,
          "session routes are inference");
    CHECK(pn_local_route_class_of("GET", "/v1x") == PN_ROUTE_ADMIN, "/v1x is not /v1");
    CHECK(pn_local_route_class_of("POST", "/admin/models/load") == PN_ROUTE_ADMIN, "admin");
    CHECK(pn_local_route_class_of("GET", "/local/hardware") == PN_ROUTE_DEVICE, "hardware");
}

static void test_pairing_rotation(void)
{
    char code[16], wrong[16];

    printf("pairing_rotation:\n");
    pn_pairing *p = pn_pairing_start(STATE "/pair_code", 600, 0);
    CHECK(p != NULL, "start");
    pn_pairing_current(p, code);
    CHECK(strlen(code) == 8, "8 digits");
    snprintf(wrong, sizeof(wrong), "%08d", (atoi(code) + 1) % 100000000);
    CHECK(pn_pairing_redeem(p, wrong) != 0, "wrong code refused");
    char spaced[16];
    snprintf(spaced, sizeof(spaced), "%.4s %.4s", code, code + 4);
    CHECK(pn_pairing_redeem(p, spaced) == 0, "right code (with a space) accepted");

    CHECK(pn_pairing_redeem(p, code) != 0, "used code no longer works");

    pn_pairing_current(p, code);
    CHECK(pn_pairing_tick(p, time(NULL) + 599) == 0, "not expired at 599 s");
    CHECK(pn_pairing_tick(p, time(NULL) + 601) == 1, "rotates after 600 s");

    pn_pairing_current(p, code);
    for (int i = 0; i < PN_PAIR_MAX_FAILURES; i++)
        pn_pairing_redeem(p, "00000000x");
    CHECK(pn_pairing_redeem(p, code) != 0, "5 wrong tries rotated the code");
    struct stat sb;
    CHECK(stat(STATE "/pair_code", &sb) == 0 && (sb.st_mode & 0777) == 0600, "pair_code 0600");
    pn_pairing_stop(p);
    CHECK(stat(STATE "/pair_code", &sb) != 0, "pair_code removed at stop");
}

static void test_rate_limit(void)
{
    printf("rate_limit:\n");
    pn_rate_limiter *rl = pn_rate_limiter_new(1.0, 3.0);
    int ok = 0;
    for (int i = 0; i < 5; i++)
        ok += pn_rate_limiter_take(rl, "k1", 100.0);
    CHECK(ok == 3, "burst of 3 then refused");
    CHECK(pn_rate_limiter_take(rl, "k2", 100.0), "other key unaffected");
    CHECK(pn_rate_limiter_take(rl, "k1", 101.1), "refills at 1/s");
    pn_rate_limiter_free(rl);
}

static void test_http(void)
{
    int port = lt_pick_port(2);
    char port_s[16], *body = NULL, inf[128] = "", inf_id[32] = "";

    printf("http:\n");
    lt_clean_dir(STATE "/srv");
    lt_make_dir(STATE "/srv");
    snprintf(port_s, sizeof(port_s), "%d", port);
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", port_s, "--state-dir",
                     STATE "/srv/state", "--store", STATE "/srv/store", "--no-mdns", "--catalog-url", "", NULL };
    pid_t pid = lt_spawn_local(12, argv, port, STATE "/srv/serve.log");
    CHECK(pid > 0, "server up");
    if (pid <= 0)
        return;
    CHECK(lt_http(port, "GET", "/v1/models", NULL, NULL, NULL) == 401, "no key -> 401");
    CHECK(lt_http(port, "GET", "/v1/models", "pnk_bad", NULL, NULL) == 401, "bad key -> 401");
    CHECK(lt_http(port, "POST", "/local/pair", NULL, "{\"code\":\"1\"}", NULL) == 403,
          "wrong pairing code -> 403");
    const char *admin = lt_admin_key(port, STATE "/srv/state");
    CHECK(admin != NULL, "paired with the code from pair_code");
    char admin_copy[128];
    snprintf(admin_copy, sizeof(admin_copy), "%s", admin ? admin : "");
    CHECK(lt_http(port, "GET", "/local/hardware", admin_copy, NULL, NULL) == 200,
          "admin key reads hardware");

    int st = lt_http(port, "POST", "/local/keys", admin_copy, "{\"label\":\"editor\"}", &body);
    CHECK(st == 200 && body && strstr(body, "\"role\":\"inference\""), "mint inference key");
    if (body) {
        sscanf(strstr(body, "\"key\":\"") ? strstr(body, "\"key\":\"") + 7 : "", "%127[^\"]", inf);
        sscanf(strstr(body, "\"id\":\"") ? strstr(body, "\"id\":\"") + 6 : "", "%31[^\"]", inf_id);
    }
    free(body);
    CHECK(lt_http(port, "GET", "/v1/models", inf, NULL, NULL) == 200, "inference key on /v1");
    CHECK(lt_http(port, "GET", "/local/hardware", inf, NULL, NULL) == 403,
          "inference key on admin route -> 403");
    st = lt_http(port, "GET", "/local/keys", admin_copy, NULL, &body);
    CHECK(st == 200 && body && strstr(body, inf_id) && !strstr(body, inf + 4),
          "list shows the id, never the secret");
    free(body);
    char rv[96];
    snprintf(rv, sizeof(rv), "{\"id\":\"%s\"}", inf_id);
    CHECK(lt_http(port, "POST", "/local/keys/revoke", admin_copy, rv, NULL) == 200, "revoke");
    CHECK(lt_http(port, "GET", "/v1/models", inf, NULL, NULL) == 401, "revoked key -> 401");
    CHECK(lt_http(port, "POST", "/local/keys", admin_copy, "{\"role\":\"admin\"}", NULL) == 400,
          "admin keys cannot be minted over the API");
    FILE *cf = fopen(STATE "/srv/state/pair_code", "r");
    char code[16] = "";
    if (cf) {
        if (!fgets(code, sizeof(code), cf)) code[0] = '\0';
        fclose(cf);
    }
    code[strcspn(code, "\n")] = '\0';
    const char *paths[] = { "/", "/local/hardware", "/local/keys", "/v1/models", "/admin/dies" };
    int leaked = 0;
    for (int i = 0; i < 5; i++) {
        lt_http(port, "GET", paths[i], admin_copy, NULL, &body);
        leaked += body && code[0] && strstr(body, code) != NULL;
        free(body);
    }
    CHECK(strlen(code) == 8 && leaked == 0, "no route returns the pairing code");
    lt_stop(pid);
}

int main(void)
{
    test_key_store();
    test_route_roles();
    test_pairing_rotation();
    test_rate_limit();
    test_http();
    return lt_report("test_local_keys");
}
