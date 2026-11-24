/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_pairing_devices.c — people registry P1.4 over HTTP against a
 * forked `pantheon local`, replaying the contract fixtures (spec §4.2-4.3):
 *   - a fresh box: the console code makes person 1 (admin) from handle/name,
 *     first_owner true, role "admin" for older clients;
 *   - a second console pair binds the same person (recovery), first_owner false;
 *   - GET /local/me; POST /local/me/device claims a device_id once (409 on a
 *     different id), renames freely; GET /local/devices marks `this`;
 *   - keys per person on a seeded box (admin david, non-admin bob): bob lists,
 *     mints and revokes only his own keys; all=1 is admin only; david sees and
 *     revokes everyone's.
 */

#include "local_test_support.h"

#include "local_fixture_replay.h"
#include "local_people_test_support.h"

#define STATE "build/test_local/pairing"

static void test_pairing_and_devices(void)
{
    jr_value *fx_pair = fx_load("pair.json"), *fx_me = fx_load("me.json");
    jr_value *fx_dev = fx_load("me_device.json"), *fx_devs = fx_load("devices.json");
    char code[32], quoted[48], k1[128], k2[128], k3[128], p1[32], p2[32], type[64];
    char *resp = NULL, *body = NULL;
    int port = lt_pick_port(41);

    printf("pairing_and_devices:\n");
    CHECK(fx_pair && fx_me && fx_dev && fx_devs, "fixtures load");
    lt_clean_dir(STATE "/fresh");
    lt_make_dir(STATE "/fresh");
    pid_t pid = spawn_box(STATE "/fresh", port);
    CHECK(pid > 0, "fresh box up");
    if (pid <= 0)
        return;
    CHECK(fx_has_error(fx_pair, 403, "pairing_refused") &&
          fx_error(port, "POST", "/local/pair", NULL, "{\"code\":\"00000000\"}", type,
                   sizeof(type)) == 403 && strcmp(type, "pairing_refused") == 0,
          "wrong code -> 403 pairing_refused (fixture error)");

    read_code(STATE "/fresh/state", code);
    snprintf(quoted, sizeof(quoted), "\"%s\"", code);
    const char *subs1[] = { "code", quoted, NULL };
    body = fx_request(fx_pair, subs1);
    CHECK(fx_replay(port, fx_pair, "POST", "/local/pair", NULL, body, &resp),
          "first pair matches pair.json");
    fx_field(resp, "key", NULL, k1, sizeof(k1));
    fx_field(resp, "person", "id", p1, sizeof(p1));
    CHECK(resp && strstr(resp, "\"first_owner\":true") && strstr(resp, "\"handle\":\"david\"") &&
          strstr(resp, "\"admin\":true") && strstr(resp, "\"role\":\"admin\""),
          "fresh box -> person 1 david (admin), first owner");
    free(resp);
    free(body);

    read_code(STATE "/fresh/state", code);
    snprintf(quoted, sizeof(quoted), "\"%s\"", code);
    const char *subs2[] = { "code", quoted, "device_id", "\"d1111111111111111\"", "handle",
                            "\"someone\"", "name", "\"Someone\"", NULL };
    body = fx_request(fx_pair, subs2);
    CHECK(fx_replay(port, fx_pair, "POST", "/local/pair", NULL, body, &resp),
          "second pair matches pair.json");
    fx_field(resp, "key", NULL, k2, sizeof(k2));
    fx_field(resp, "person", "id", p2, sizeof(p2));
    CHECK(strcmp(p1, p2) == 0 && resp && strstr(resp, "\"first_owner\":false") &&
          strstr(resp, "\"handle\":\"david\""), "second console pair -> the same person");
    free(resp);
    free(body);

    /* An older Argus: {code, label} only — unclaimed device, claimed later. */
    read_code(STATE "/fresh/state", code);
    char old_body[96];
    snprintf(old_body, sizeof(old_body), "{\"code\":\"%s\",\"label\":\"argus\"}", code);
    CHECK(lt_http(port, "POST", "/local/pair", NULL, old_body, &resp) == 200 && resp &&
          strstr(resp, "\"role\":\"admin\""), "an older client pairs ({code,label})");
    fx_field(resp, "key", NULL, k3, sizeof(k3));
    free(resp);

    CHECK(fx_replay(port, fx_me, "GET", "/local/me", k1, NULL, &resp) && resp &&
          strstr(resp, p1) && strstr(resp, "\"device_id\":\"d0123456789abcdef\""),
          "GET /local/me matches me.json");
    free(resp);
    CHECK(lt_http(port, "GET", "/local/me", k3, NULL, &resp) == 200 && resp &&
          strstr(resp, "\"device_id\":\"\""), "unclaimed key reads device_id \"\"");
    free(resp);
    const char *claim[] = { "device_id", "\"d2222222222222222\"", "device_name", "\"laptop\"",
                            NULL };
    body = fx_request(fx_dev, claim);
    CHECK(fx_replay(port, fx_dev, "POST", "/local/me/device", k3, body, &resp) && resp &&
          strstr(resp, "\"device_id\":\"d2222222222222222\""), "claim matches me_device.json");
    free(resp);
    free(body);
    CHECK(lt_http(port, "POST", "/local/me/device", k3,
                  "{\"device_id\":\"d2222222222222222\",\"device_name\":\"Laptop 2\"}", &resp) ==
          200 && resp && strstr(resp, "Laptop 2"), "same id again renames");
    free(resp);
    CHECK(fx_has_error(fx_dev, 409, "device_id_mismatch") &&
          fx_error(port, "POST", "/local/me/device", k3, "{\"device_id\":\"d3333333333333333\"}",
                   type, sizeof(type)) == 409 && strcmp(type, "device_id_mismatch") == 0,
          "a different device_id -> 409 device_id_mismatch");

    CHECK(fx_replay(port, fx_devs, "GET", "/local/devices", k2, NULL, &resp) && resp &&
          count_of(resp, "\"key_id\"") == 3 && count_of(resp, "\"this\":true") == 1,
          "devices match devices.json; three devices, one is this");
    free(resp);
    lt_stop(pid);
    jr_free(fx_pair);
    jr_free(fx_me);
    jr_free(fx_dev);
    jr_free(fx_devs);
}

static void test_keys_per_person(void)
{
    jr_value *fx_list = fx_load("keys_list.json"), *fx_mint = fx_load("keys_mint.json");
    jr_value *fx_rev = fx_load("keys_revoke.json");
    char david[80], bob[80], bob_inf[80], type[64], id[32], *resp = NULL, *body = NULL;
    int port = lt_pick_port(43);

    printf("keys_per_person:\n");
    CHECK(fx_list && fx_mint && fx_rev, "fixtures load");
    lt_clean_dir(STATE "/seed");
    lt_make_dir(STATE "/seed/state");
    seed_state(STATE "/seed/state", 1);
    secret_of('a', david);
    secret_of('b', bob);
    secret_of('c', bob_inf);
    pid_t pid = spawn_box(STATE "/seed", port);
    CHECK(pid > 0, "seeded box up");
    if (pid <= 0)
        return;
    CHECK(fx_replay(port, fx_list, "GET", "/local/keys", bob, NULL, &resp) && resp &&
          count_of(resp, "\"id\":\"k") == 2 && strstr(resp, "k000000bb") &&
          strstr(resp, "k000000cc") && !strstr(resp, "k000000aa") &&
          strstr(resp, "\"role\":\"device\""), "bob lists only his keys (keys_list.json)");
    free(resp);
    CHECK(fx_has_error(fx_list, 403, "forbidden") &&
          fx_error(port, "GET", "/local/keys?all=1", bob, NULL, type, sizeof(type)) == 403 &&
          strcmp(type, "forbidden") == 0, "bob all=1 -> 403");
    CHECK(fx_error(port, "GET", "/local/devices?all=1", bob, NULL, type, sizeof(type)) == 403,
          "bob devices all=1 -> 403");
    CHECK(lt_http(port, "GET", "/local/keys?all=1", david, NULL, &resp) == 200 &&
          count_of(resp, "\"id\":\"k") == 4, "david all=1 sees every key");
    free(resp);
    CHECK(lt_http(port, "GET", "/local/keys", bob_inf, NULL, NULL) == 403,
          "an inference key cannot list keys");

    body = fx_request(fx_mint, NULL);
    CHECK(fx_replay(port, fx_mint, "POST", "/local/keys", bob, body, &resp) && resp &&
          strstr(resp, "\"id\":\"p00000002\""), "bob mints for himself (keys_mint.json)");
    fx_field(resp, "id", NULL, id, sizeof(id));
    free(resp);
    free(body);
    CHECK(fx_error(port, "POST", "/local/keys", bob, "{\"label\":\"x\",\"person\":\"p00000001\"}",
                   type, sizeof(type)) == 403 && fx_has_error(fx_mint, 403, type),
          "bob minting for david -> 403");
    CHECK(fx_error(port, "POST", "/local/keys", david, "{\"label\":\"x\",\"person\":\"p0000000f\"}",
                   type, sizeof(type)) == 404 && fx_has_error(fx_mint, 404, type),
          "unknown person -> 404");
    CHECK(fx_error(port, "POST", "/local/keys", david, "{\"role\":\"admin\"}", type,
                   sizeof(type)) == 400 && fx_has_error(fx_mint, 400, type), "role admin -> 400");
    CHECK(lt_http(port, "POST", "/local/keys", david,
                  "{\"label\":\"for-bob\",\"person\":\"p00000002\"}", &resp) == 200 && resp &&
          strstr(resp, "\"handle\":\"bob\""), "an admin mints for bob");
    free(resp);

    CHECK(fx_error(port, "POST", "/local/keys/revoke", bob, "{\"id\":\"k000000dd\"}", type,
                   sizeof(type)) == 403 && fx_has_error(fx_rev, 403, type),
          "bob revoking david's key -> 403");
    CHECK(fx_error(port, "POST", "/local/keys/revoke", bob, "{\"id\":\"k00000000\"}", type,
                   sizeof(type)) == 404 && fx_has_error(fx_rev, 404, type), "unknown key -> 404");
    char rv[64];
    snprintf(rv, sizeof(rv), "{\"id\":\"%s\"}", id);
    CHECK(fx_replay(port, fx_rev, "POST", "/local/keys/revoke", bob, rv, NULL),
          "bob revokes his own key (keys_revoke.json)");
    CHECK(lt_http(port, "POST", "/local/keys/revoke", david, "{\"id\":\"k000000cc\"}", NULL) ==
          200 && lt_http(port, "GET", "/v1/models", bob_inf, NULL, NULL) == 401,
          "david revokes bob's key; it gets 401");
    CHECK(lt_http(port, "GET", "/local/me", bob, NULL, &resp) == 200 && resp &&
          strstr(resp, "\"admin\":false"), "bob's /local/me says not admin");
    free(resp);
    lt_stop(pid);
    jr_free(fx_list);
    jr_free(fx_mint);
    jr_free(fx_rev);
}

int main(void)
{
    signal(SIGPIPE, SIG_IGN);
    test_pairing_and_devices();
    test_keys_per_person();
    return lt_report("test_local_pairing_devices");
}
