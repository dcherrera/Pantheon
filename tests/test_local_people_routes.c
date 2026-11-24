/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_people_routes.c — people registry P1.5: invites and people routes.
 *   - invites (unit): one use, 10 min expiry, 16 live max, voided per person
 *     and all; the console code's 5th miss returns -2 (misses 1-4 return -1);
 *   - over HTTP against a forked box, replaying the contract fixtures: an admin
 *     adds a person and invites them; the invite pairs a device of that person
 *     once; the people list is full for an admin, short otherwise; a non-admin
 *     invites only themselves (403 for another) and renames only themselves;
 *     `admin` follows the admin tree (the child admin cannot demote or remove
 *     its maker); removal revokes the keys (401 after) and voids the invites;
 *     the last admin stays; 16 live invites max; on a second box, the 5th
 *     wrong pairing code voids a live invite.
 */

#include "local_test_support.h"

#include "local_fixture_replay.h"
#include "local_people_test_support.h"

#include "../src/local/invites.h"
#include "../src/local/pairing.h"

#define STATE "build/test_local/people_routes"

static void test_invites_unit(void)
{
    char code[16], code2[16], person[PN_INVITE_PERSON_MAX];
    time_t t0 = 1793000000;
    int issued = 0;

    printf("invites_unit:\n");
    pn_invites *inv = pn_invites_new(PN_INVITE_TTL_S);
    CHECK(pn_invites_issue(inv, "p00000002", t0, code) == 0 && strlen(code) == 8, "issue");
    CHECK(pn_invites_redeem(inv, code, t0 + 1, person) == 0 && strcmp(person, "p00000002") == 0,
          "redeem binds the person");
    CHECK(pn_invites_redeem(inv, code, t0 + 2, person) != 0, "one use");
    pn_invites_issue(inv, "p00000002", t0, code);
    pn_invites_issue(inv, "p00000003", t0, code2);
    char spaced[16];
    snprintf(spaced, sizeof(spaced), "%.4s-%.4s", code2, code2 + 4);
    CHECK(pn_invites_redeem(inv, spaced, t0 + PN_INVITE_TTL_S - 1, person) == 0,
          "valid at 599 s (dashes ignored)");
    CHECK(pn_invites_redeem(inv, code, t0 + PN_INVITE_TTL_S, person) != 0, "expired at 600 s");
    for (int i = 0; i < PN_INVITES_MAX + 1; i++)
        issued += pn_invites_issue(inv, "p00000002", t0, code) == 0;
    CHECK(issued == 16 && pn_invites_live(inv, t0) == 16, "16 live max");
    CHECK(pn_invites_issue(inv, "p00000002", t0 + PN_INVITE_TTL_S, code) == 0,
          "room again once they expire");
    pn_invites_void_all(inv);
    pn_invites_issue(inv, "p00000002", t0, code);
    pn_invites_issue(inv, "p00000003", t0, code2);
    CHECK(pn_invites_void_person(inv, "p00000002") == 1 &&
          pn_invites_redeem(inv, code, t0, person) != 0 &&
          pn_invites_redeem(inv, code2, t0, person) == 0, "void one person's invites");
    pn_invites_free(inv);

    lt_make_dir(STATE);
    pn_pairing *p = pn_pairing_start(STATE "/pair_code", 600, 0);
    int minus_one = 0;
    for (int i = 0; i < PN_PAIR_MAX_FAILURES - 1; i++)
        minus_one += pn_pairing_redeem(p, "00000000x") == -1;
    CHECK(minus_one == 4 && pn_pairing_redeem(p, "00000000x") == -2,
          "misses 1-4 -> -1, the 5th -> -2 (rotated)");
    for (int i = 0; i < PN_PAIR_MAX_FAILURES - 1; i++)
        pn_pairing_redeem(p, "00000000x");
    pn_pairing_clear_misses(p);
    CHECK(pn_pairing_redeem(p, "00000000x") == -1, "cleared misses start over");
    pn_pairing_stop(p);
}

typedef struct {
    int  port;
    char david[80];
    char bob_id[32];
    char bob[128];
} box_a;

/* An admin adds bob and invites him; bob pairs with the invite. */
static void add_and_invite(box_a *a)
{
    jr_value *fx_add = fx_load("people_add.json"), *fx_inv = fx_load("invites.json");
    char *resp = NULL, *body = NULL, type[64], code[32], sub[48], pair[256];

    CHECK(fx_add && fx_inv, "fixtures load");
    body = fx_request(fx_add, NULL);
    CHECK(fx_replay(a->port, fx_add, "POST", "/local/people", a->david, body, &resp) && resp &&
          strstr(resp, "\"admin\":false"), "add bob (people_add.json)");
    fx_field(resp, "person", "id", a->bob_id, sizeof(a->bob_id));
    free(resp);
    free(body);
    CHECK(fx_error(a->port, "POST", "/local/people", a->david, "{\"handle\":\"BOB\",\"name\":\"B\"}",
                   type, sizeof(type)) == 409 && fx_has_error(fx_add, 409, type),
          "handle taken ignoring case -> 409");
    CHECK(fx_error(a->port, "POST", "/local/people", a->david,
                   "{\"handle\":\"abcdefghijklmnopqrstuvwxy\",\"name\":\"B\"}", type,
                   sizeof(type)) == 400 && strcmp(type, "bad_handle") == 0,
          "25-byte handle -> 400 bad_handle");
    CHECK(fx_error(a->port, "POST", "/local/people", a->david,
                   "{\"handle\":\"carl\",\"name\":\"C\\u0001\"}", type, sizeof(type)) == 400 &&
          fx_has_error(fx_add, 400, type) && strcmp(type, "bad_name") == 0,
          "control char in name -> 400 bad_name");

    snprintf(sub, sizeof(sub), "\"%s\"", a->bob_id);
    const char *subs[] = { "person", sub, NULL };
    body = fx_request(fx_inv, subs);
    CHECK(fx_replay(a->port, fx_inv, "POST", "/local/invites", a->david, body, &resp) && resp &&
          strstr(resp, "\"expires_in\":600"), "invite bob (invites.json)");
    fx_field(resp, "code", NULL, code, sizeof(code));
    free(resp);
    free(body);
    snprintf(pair, sizeof(pair), "{\"code\":\"%s\",\"label\":\"argus\",\"device_id\":"
             "\"d00000000000000b0\",\"device_name\":\"bob-mac\"}", code);
    CHECK(lt_http(a->port, "POST", "/local/pair", NULL, pair, &resp) == 200 && resp &&
          strstr(resp, a->bob_id) && strstr(resp, "\"role\":\"device\"") &&
          strstr(resp, "\"first_owner\":false"), "the invite pairs a device of bob");
    fx_field(resp, "key", NULL, a->bob, sizeof(a->bob));
    free(resp);
    CHECK(fx_error(a->port, "POST", "/local/pair", NULL, pair, type, sizeof(type)) == 403 &&
          strcmp(type, "pairing_refused") == 0, "an invite works once");
    jr_free(fx_add);
    jr_free(fx_inv);
}

static void list_and_self_service(box_a *a)
{
    jr_value *fx_list = fx_load("people_list.json"), *fx_inv = fx_load("invites.json");
    char *resp = NULL, type[64], path[64];

    CHECK(fx_replay(a->port, fx_list, "GET", "/local/people", a->david, NULL, &resp) && resp &&
          strstr(resp, "\"admin\""), "admin sees full objects (people_list.json)");
    free(resp);
    CHECK(fx_replay_as(a->port, fx_list, "response_short", "GET", "/local/people", a->bob, NULL,
                       &resp) && resp && !strstr(resp, "\"admin\""),
          "non-admin sees short objects");
    free(resp);
    CHECK(lt_http(a->port, "POST", "/local/invites", a->bob, "{}", &resp) == 200 && resp &&
          strstr(resp, a->bob_id), "bob invites himself");
    free(resp);
    CHECK(fx_error(a->port, "POST", "/local/invites", a->bob, "{\"person\":\"p00000001\"}", type,
                   sizeof(type)) == 403 && fx_has_error(fx_inv, 403, type),
          "bob inviting for david -> 403");
    CHECK(fx_error(a->port, "POST", "/local/invites", a->david, "{\"person\":\"p0000000f\"}",
                   type, sizeof(type)) == 404 && fx_has_error(fx_inv, 404, type),
          "unknown person -> 404");
    snprintf(path, sizeof(path), "/local/people/%s", a->bob_id);
    CHECK(lt_http(a->port, "POST", path, a->bob, "{\"name\":\"Bobby\"}", &resp) == 200 && resp &&
          strstr(resp, "Bobby"), "bob renames himself");
    free(resp);
    CHECK(fx_error(a->port, "POST", path, a->bob, "{\"may_load\":false}", type, sizeof(type)) ==
          403 && strcmp(type, "forbidden") == 0, "bob cannot change his may_load");
    CHECK(fx_error(a->port, "POST", "/local/people/p00000001", a->bob, "{\"name\":\"x\"}", type,
                   sizeof(type)) == 403 && strcmp(type, "forbidden") == 0,
          "bob cannot rename david");
    CHECK(fx_error(a->port, "POST", "/local/people", a->bob, "{\"handle\":\"x\",\"name\":\"X\"}",
                   type, sizeof(type)) == 403, "bob cannot add people");
    jr_free(fx_list);
    jr_free(fx_inv);
}

static void admin_tree(box_a *a)
{
    jr_value *fx_up = fx_load("people_update.json");
    char *resp = NULL, *body = NULL, type[64], path[64];

    snprintf(path, sizeof(path), "/local/people/%s", a->bob_id);
    body = fx_request(fx_up, NULL);
    CHECK(fx_replay(a->port, fx_up, "POST", path, a->david, body, &resp) && resp &&
          strstr(resp, "\"admin\":true"), "david makes bob admin (people_update.json)");
    free(resp);
    free(body);
    CHECK(fx_error(a->port, "POST", "/local/people/p00000001", a->bob, "{\"admin\":false}", type,
                   sizeof(type)) == 403 && strcmp(type, "admin_above") == 0 &&
          fx_has_error(fx_up, 403, type), "bob cannot demote his maker");
    CHECK(fx_error(a->port, "POST", "/local/people/remove", a->bob, "{\"id\":\"p00000001\"}",
                   type, sizeof(type)) == 403 && strcmp(type, "admin_above") == 0,
          "bob cannot remove his maker");
    CHECK(lt_http(a->port, "POST", path, a->david, "{\"admin\":false}", &resp) == 200 && resp &&
          strstr(resp, "\"admin\":false"), "david demotes bob");
    free(resp);
    jr_free(fx_up);
}

static void remove_and_limits(box_a *a)
{
    jr_value *fx_rm = fx_load("people_remove.json");
    char *resp = NULL, type[64], code[32], body[160];
    int issued = 0;

    snprintf(body, sizeof(body), "{\"person\":\"%s\"}", a->bob_id);
    lt_http(a->port, "POST", "/local/invites", a->david, body, &resp);
    fx_field(resp, "code", NULL, code, sizeof(code));
    free(resp);
    snprintf(body, sizeof(body), "{\"id\":\"%s\"}", a->bob_id);
    CHECK(fx_replay(a->port, fx_rm, "POST", "/local/people/remove", a->david, body, &resp) &&
          resp && strstr(resp, "\"keys_revoked\":1"), "remove bob (people_remove.json)");
    free(resp);
    CHECK(lt_http(a->port, "GET", "/local/me", a->bob, NULL, NULL) == 401,
          "a removed person's key gets 401");
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"argus\"}", code);
    CHECK(fx_error(a->port, "POST", "/local/pair", NULL, body, type, sizeof(type)) == 403,
          "a removed person's invite is void");
    CHECK(fx_error(a->port, "POST", "/local/people/remove", a->david, "{\"id\":\"p00000001\"}",
                   type, sizeof(type)) == 409 && fx_has_error(fx_rm, 409, type),
          "the last admin stays");
    for (int i = 0; i < PN_INVITES_MAX; i++)
        issued += lt_http(a->port, "POST", "/local/invites", a->david, "{}", NULL) == 200;
    CHECK(issued == 16 && fx_error(a->port, "POST", "/local/invites", a->david, "{}", type,
                                   sizeof(type)) == 429 && strcmp(type, "invites_full") == 0,
          "16 live invites, then 429 invites_full");
    jr_free(fx_rm);
}

static void test_people_http(void)
{
    box_a a = { .port = lt_pick_port(47) };

    printf("people_http:\n");
    lt_clean_dir(STATE "/a");
    lt_make_dir(STATE "/a/state");
    seed_state(STATE "/a/state", 0);
    secret_of('a', a.david);
    pid_t pid = spawn_box(STATE "/a", a.port);
    CHECK(pid > 0, "box up");
    if (pid <= 0)
        return;
    add_and_invite(&a);
    list_and_self_service(&a);
    admin_tree(&a);
    remove_and_limits(&a);
    lt_stop(pid);
}

/* The 5th wrong pairing code rotates the console code and voids live invites. */
static void test_fifth_miss_voids_invites(void)
{
    char david[80], code[32], body[128], type[64], *resp = NULL;
    int port = lt_pick_port(53), refused = 0;

    printf("fifth_miss_voids_invites:\n");
    lt_clean_dir(STATE "/b");
    lt_make_dir(STATE "/b/state");
    seed_state(STATE "/b/state", 0);
    secret_of('a', david);
    pid_t pid = spawn_box(STATE "/b", port);
    CHECK(pid > 0, "box up");
    if (pid <= 0)
        return;
    lt_http(port, "POST", "/local/invites", david, "{}", &resp);
    fx_field(resp, "code", NULL, code, sizeof(code));
    free(resp);
    for (int i = 0; i < PN_PAIR_MAX_FAILURES; i++)
        refused += lt_http(port, "POST", "/local/pair", NULL, "{\"code\":\"00000000x\"}", NULL) ==
                   403;
    CHECK(refused == 5, "five wrong codes refused");
    usleep(5300000);   /* the box-wide pair bucket refills one try per 5 s */
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"argus\"}", code);
    CHECK(strlen(code) == 8 && fx_error(port, "POST", "/local/pair", NULL, body, type,
                                        sizeof(type)) == 403 &&
          strcmp(type, "pairing_refused") == 0, "the live invite was voided");
    lt_stop(pid);
}

int main(void)
{
    signal(SIGPIPE, SIG_IGN);
    test_invites_unit();
    test_people_http();
    test_fifth_miss_voids_invites();
    return lt_report("test_local_people_routes");
}
