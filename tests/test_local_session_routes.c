/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_session_routes.c — people registry P3.2: visibility, sharing and
 * attach tickets on a forked `pantheon local` (spec §4.5-4.6; Argus §10.8 test 2,
 * box side), replaying sessions_list / sessions_share / sessions_attach and
 * hosts_wait's response_shares:
 *   - bob sees nothing until david shares; a stranger's attach is 403;
 *   - list: own rows with shares, shared rows with role and an empty share list,
 *     never host_addr / host_pub; ?rev=<current> -> unchanged, a change moves rev;
 *   - a view share lists with role view and attach gives a view ticket that
 *     checks under the host's secret (person, device_id, cpub lowercased, exp);
 *     control and owner likewise;
 *   - share and unshare queue `shares` on the host's held wait within one notify;
 *   - errors: not the owner, yourself, unknown person / key, desktop row
 *     (not_shareable; owner-only attach), bad cpub, device_unclaimed, and a
 *     disconnected row after a box restart -> 409;
 *   - no ticket, host_secret or host_pub in the log; none in sessions.json.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"
#include "local_fixture_replay.h"
#include "local_people_test_support.h"
#include "local_host_wait_support.h"

#include "../src/local/attach_ticket.h"

#define DIR    "build/test_local/session_routes"
#define STATE  DIR "/state"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define CPUB   "8F40C5ADB68F25624AE5B214EA767A6EC94D829D3D7B5E1AD1BA6F3E2138285F"
#define K1     "c-0123456789ab"
#define KD     "c-00000000000d"   /* david's desktop row */
#define DAVID  "p00000001"
#define BOB    "p00000002"
#define DEV_A  "d00000000000000aa"
#define DEV_B  "d00000000000000bb"

static int g_port;
static char g_key_a[80], g_key_b[80];
static char g_macs[8][PN_SHA256_HEX_LEN + 1];   /* tickets' MACs, searched for in the log */
static int g_n_macs;

static int post_host(const char *secret, const char *key, const char *kind, const char *origin)
{
    char body[1024];

    snprintf(body, sizeof(body), "{\"key\":\"%s\",\"kind\":\"%s\",\"origin_device\":\"%s\","
             "\"host_port\":53211,\"title\":\"Fix the relay\",\"model\":\"m\",\"instance\":2,"
             "\"state\":\"idle\",\"host_secret\":\"" SECRET "\",\"host_pub\":\"" PUB "\"}",
             key, kind, origin);
    return lt_http(g_port, "POST", "/local/hosts", secret, body, NULL);
}

static int share(const char *secret, const char *key, const char *body, char *type, size_t cap)
{
    char path[128];

    snprintf(path, sizeof(path), "/local/sessions/%s/share", key);
    return fx_error(g_port, "POST", path, secret, body, type, cap);
}

static int attach_error(const char *secret, const char *key, const char *body, char *type,
                        size_t cap)
{
    char path[128];

    snprintf(path, sizeof(path), "/local/sessions/%s/attach", key);
    return fx_error(g_port, "POST", path, secret, body, type, cap);
}

/* Attach with the fixture; check the ticket under the host's secret. */
static int attach_ok(const jr_value *fx, const char *secret, const char *key, const char *person,
                     const char *device, pn_ticket_role want)
{
    char path[128], ticket[PN_TICKET_MAX], v[128], *resp = NULL, cpub_lc[65];
    const char *subs[] = { "cpub", "\"" CPUB "\"", NULL };
    char *req = fx_request(fx, subs);
    uint8_t host_secret[PN_TICKET_SECRET_LEN];
    pn_ticket_fields f = {0};
    long long now = (long long)time(NULL);
    int ok = 0;

    snprintf(path, sizeof(path), "/local/sessions/%s/attach", key);
    ok = req && fx_replay(g_port, fx, "POST", path, secret, req, &resp);
    fx_field(resp, "ticket", NULL, ticket, sizeof(ticket));
    pn_hex_decode(SECRET, host_secret, sizeof(host_secret));
    ok = ok && pn_ticket_check(host_secret, key, ticket, now, &f) == PN_TICKET_OK;
    for (int i = 0; i < 64; i++)
        cpub_lc[i] = (char)(CPUB[i] >= 'A' && CPUB[i] <= 'F' ? CPUB[i] - 'A' + 'a' : CPUB[i]);
    cpub_lc[64] = '\0';
    ok = ok && f.role == want && strcmp(f.person, person) == 0 &&
         strcmp(f.device_id, device) == 0 && strcmp(f.cpub, cpub_lc) == 0 &&
         f.exp >= now + 59 && f.exp <= now + 61;
    fx_field(resp, "role", NULL, v, sizeof(v));
    ok = ok && strcmp(v, pn_ticket_role_name(want)) == 0;
    fx_field(resp, "host_pub", NULL, v, sizeof(v));
    ok = ok && strcmp(v, PUB) == 0;
    fx_field(resp, "host_addr", NULL, v, sizeof(v));
    ok = ok && strcmp(v, "127.0.0.1:53211") == 0 && strstr(resp, "\"expires_in\":60");
    if (g_n_macs < 8 && strchr(ticket, '.'))
        snprintf(g_macs[g_n_macs++], sizeof(g_macs[0]), "%s", strchr(ticket, '.') + 1);
    if (!ok)
        fprintf(stderr, "  attach %s: %s\n", key, resp ? resp : "(none)");
    free(req);
    free(resp);
    return ok;
}

static char *list_of(const char *secret, const char *query)
{
    char path[64], *resp = NULL;

    snprintf(path, sizeof(path), "/local/sessions%s", query ? query : "");
    if (lt_http(g_port, "GET", path, secret, NULL, &resp) != 200) {
        free(resp);
        return strdup("");
    }
    return resp;
}

static void test_visibility(void)
{
    jr_value *fl = fx_load("sessions_list.json"), *fa = fx_load("sessions_attach.json");
    char *resp = NULL, path[64], rev[32] = "", type[64];
    jr_value *got = NULL;

    printf("visibility:\n");
    CHECK(post_host(g_key_a, K1, "session", DEV_A) == 200, "david registers a session");
    CHECK(fl && fx_replay(g_port, fl, "GET", "/local/sessions", g_key_b, NULL, &resp) &&
          strstr(resp, "\"sessions\":[]"), "bob sees nothing before a share");
    free(resp);
    resp = NULL;
    CHECK(attach_error(g_key_b, K1, "{\"cpub\":\"" CPUB "\"}", type, sizeof(type)) == 403 &&
          strcmp(type, "forbidden") == 0 && fx_has_error(fa, 403, "forbidden"),
          "a stranger's attach: 403 forbidden");
    CHECK(fx_replay(g_port, fl, "GET", "/local/sessions", g_key_a, NULL, &resp) &&
          count_of(resp, "\"key\":") == 1 && strstr(resp, "\"role\":\"owner\"") &&
          strstr(resp, "\"shares\":[]") && strstr(resp, "\"name\":\"seed\""),
          "david's list replays the fixture: his row, owner, origin device name");
    CHECK(resp && !strstr(resp, "host_addr") && !strstr(resp, PUB) && !strstr(resp, SECRET),
          "no host_addr, host_pub or secret in the list");
    got = jr_parse(resp);
    snprintf(rev, sizeof(rev), "%lld", (long long)jr_as_number(jr_object_get(got, "rev"), -1));
    jr_free(got);
    free(resp);
    resp = NULL;
    snprintf(path, sizeof(path), "/local/sessions?rev=%s", rev);
    CHECK(fx_replay_as(g_port, fl, "response_unchanged", "GET", path, g_key_a, NULL, NULL),
          "?rev=<current>: unchanged");
    CHECK(fx_error(g_port, "GET", "/local/sessions?rev=x", g_key_a, NULL, type,
                   sizeof(type)) == 400 && fx_has_error(fl, 400, "invalid_request_error"),
          "?rev=x: 400");
    CHECK(attach_ok(fa, g_key_a, K1, DAVID, DEV_A, PN_TICKET_ROLE_OWNER),
          "the owner's attach replays the fixture with an owner ticket");
    jr_free(fl);
    jr_free(fa);
}

/* Share as `body` while a wait from `since` is held; 1 when the wait hears
 * `shares` holding `needle` within 1 s (the wait itself lasts 5 s). */
static int share_heard(const jr_value *fs, const jr_value *fw, const char *body,
                       long long since, const char *needle)
{
    int w = wait_open(g_port, g_key_a, K1, since);
    char got[4096], path[96], *resp = NULL;
    uint64_t t0 = 0;
    int ok = silent_for(w, 100);

    snprintf(path, sizeof(path), "/local/sessions/%s/share", K1);
    t0 = ms_now();
    ok = ok && fx_replay(g_port, fs, "POST", path, g_key_a, body, &resp);
    ok = ok && wait_read(w, got, sizeof(got), 3000) == 200 && ms_now() - t0 < 1000 &&
         wait_shape(fw, "response_shares", got) && strstr(got, needle);
    free(resp);
    return ok;
}

static void test_sharing(void)
{
    jr_value *fs = fx_load("sessions_share.json"), *fw = fx_load("hosts_wait.json");
    jr_value *fa = fx_load("sessions_attach.json"), *fl = fx_load("sessions_list.json");
    char *resp = NULL, *before = NULL, type[64];

    printf("sharing:\n");
    before = list_of(g_key_a, NULL);
    CHECK(fs && share_heard(fs, fw, "{\"handle\":\"bob\",\"role\":\"view\"}", 0,
                            "{\"person\":\"" BOB "\",\"handle\":\"bob\",\"role\":\"view\"}"),
          "share by handle replays the fixture; the held wait hears shares at once");
    resp = list_of(g_key_a, NULL);
    CHECK(strcmp(resp, before) != 0 && strstr(resp, "\"handle\":\"bob\",\"role\":\"view\""),
          "rev moved; david's row lists bob's share");
    free(resp);
    resp = NULL;
    CHECK(fx_replay(g_port, fl, "GET", "/local/sessions", g_key_b, NULL, &resp) &&
          count_of(resp, "\"key\":") == 1 && strstr(resp, "\"role\":\"view\"") &&
          strstr(resp, "\"shares\":[]") && strstr(resp, "\"handle\":\"david\""),
          "bob lists it with role view and no share list");
    free(resp);
    CHECK(attach_ok(fa, g_key_b, K1, BOB, DEV_B, PN_TICKET_ROLE_VIEW),
          "bob's attach gives a view ticket");
    CHECK(share(g_key_b, K1, "{\"person\":\"" DAVID "\",\"role\":\"view\"}", type,
                sizeof(type)) == 403 && strcmp(type, "forbidden") == 0 &&
          fx_has_error(fs, 403, "forbidden"), "bob cannot share david's session");
    CHECK(share_heard(fs, fw, "{\"person\":\"" BOB "\",\"role\":\"control\"}", 1,
                      "\"role\":\"control\""), "upgrade to control by person id is heard");
    CHECK(attach_ok(fa, g_key_b, K1, BOB, DEV_B, PN_TICKET_ROLE_CONTROL),
          "bob's attach now gives a control ticket");
    CHECK(share_heard(fs, fw, "{\"handle\":\"bob\",\"role\":\"none\"}", 2, "\"shares\":[]"),
          "unshare: the held wait hears an empty share list within one notify");
    resp = list_of(g_key_b, NULL);
    CHECK(strstr(resp, "\"sessions\":[]") != NULL, "bob sees nothing again");
    free(resp);
    CHECK(attach_error(g_key_b, K1, "{\"cpub\":\"" CPUB "\"}", type, sizeof(type)) == 403 &&
          strcmp(type, "forbidden") == 0, "and his attach is 403");
    jr_free(fs);
    jr_free(fw);
    jr_free(fa);
    jr_free(fl);
}

static void test_errors(void)
{
    jr_value *fs = fx_load("sessions_share.json"), *fa = fx_load("sessions_attach.json");
    char type[64], code[32] = "", body[160], key_old[80] = "", *resp = NULL;

    printf("errors:\n");
    CHECK(share(g_key_a, K1, "{\"person\":\"" DAVID "\",\"role\":\"view\"}", type,
                sizeof(type)) == 400 && strcmp(type, "invalid_request_error") == 0 &&
          fx_has_error(fs, 400, "invalid_request_error"), "sharing with yourself: 400");
    CHECK(share(g_key_a, K1, "{\"handle\":\"bob\",\"role\":\"owner\"}", type,
                sizeof(type)) == 400, "role owner: 400");
    CHECK(share(g_key_a, K1, "{\"handle\":\"nobody\",\"role\":\"view\"}", type,
                sizeof(type)) == 404 && strcmp(type, "not_found") == 0 &&
          fx_has_error(fs, 404, "not_found"), "unknown person: 404");
    CHECK(share(g_key_a, "c-ffffffffffff", "{\"handle\":\"bob\",\"role\":\"view\"}", type,
                sizeof(type)) == 404 && strcmp(type, "not_found") == 0, "unknown key: 404");
    CHECK(post_host(g_key_a, KD, "desktop", DEV_A) == 200, "david registers a desktop row");
    CHECK(share(g_key_a, KD, "{\"handle\":\"bob\",\"role\":\"view\"}", type,
                sizeof(type)) == 400 && strcmp(type, "not_shareable") == 0 &&
          fx_has_error(fs, 400, "not_shareable"), "a desktop row: 400 not_shareable");
    CHECK(attach_error(g_key_b, KD, "{\"cpub\":\"" CPUB "\"}", type, sizeof(type)) == 403,
          "another person's desktop row: 403");
    CHECK(attach_ok(fa, g_key_a, KD, DAVID, DEV_A, PN_TICKET_ROLE_OWNER),
          "the owner attaches to his desktop row");
    CHECK(attach_error(g_key_a, K1, "{\"cpub\":\"abc\"}", type, sizeof(type)) == 400 &&
          strcmp(type, "bad_cpub") == 0 && fx_has_error(fa, 400, "bad_cpub") &&
          attach_error(g_key_a, K1, "{}", type, sizeof(type)) == 400 &&
          strcmp(type, "bad_cpub") == 0, "bad or missing cpub: 400 bad_cpub");
    CHECK(attach_error(g_key_a, "c-ffffffffffff", "{\"cpub\":\"" CPUB "\"}", type,
                       sizeof(type)) == 404 && strcmp(type, "not_found") == 0 &&
          fx_has_error(fa, 404, "not_found"), "unknown key: 404");
    read_code(STATE, code);
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"old\"}", code);
    if (lt_http(g_port, "POST", "/local/pair", NULL, body, &resp) == 200)
        fx_field(resp, "key", NULL, key_old, sizeof(key_old));
    free(resp);
    CHECK(key_old[0] && attach_error(key_old, K1, "{\"cpub\":\"" CPUB "\"}", type,
                                     sizeof(type)) == 409 &&
          strcmp(type, "device_unclaimed") == 0 && fx_has_error(fa, 409, "device_unclaimed"),
          "a key with no device_id: 409 device_unclaimed");
    secret_of('d', key_old);
    CHECK(fx_error(g_port, "GET", "/local/sessions", key_old, NULL, type, sizeof(type)) == 403,
          "an inference key: 403");
    jr_free(fs);
    jr_free(fa);
}

static pid_t test_disconnected(pid_t pid)
{
    jr_value *fa = fx_load("sessions_attach.json");
    char type[64], *resp = NULL;

    printf("disconnected:\n");
    lt_stop(pid);
    if (rename(DIR "/serve.log", DIR "/serve.run1.log") != 0)
        fprintf(stderr, "  warn: could not keep the first log\n");
    pid = spawn_box(DIR, g_port);
    CHECK(pid > 0, "the box restarted");
    resp = list_of(g_key_a, NULL);
    CHECK(strstr(resp, "\"state\":\"disconnected\"") != NULL, "rows read disconnected");
    free(resp);
    CHECK(attach_error(g_key_a, K1, "{\"cpub\":\"" CPUB "\"}", type, sizeof(type)) == 409 &&
          strcmp(type, "disconnected") == 0 && fx_has_error(fa, 409, "disconnected"),
          "attach to a disconnected row: 409 disconnected");
    CHECK(post_host(g_key_a, K1, "session", DEV_A) == 200 &&
          attach_ok(fa, g_key_a, K1, DAVID, DEV_A, PN_TICKET_ROLE_OWNER),
          "re-registered: attach works again");
    jr_free(fa);
    return pid;
}

static int file_has(const char *path, const char *needle)
{
    static char buf[1 << 20];
    FILE *f = fopen(path, "r");
    size_t n = f ? fread(buf, 1, sizeof(buf) - 1, f) : 0;

    if (f)
        fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

int main(void)
{
    pid_t pid = -1;
    int clean = 1;

    signal(SIGPIPE, SIG_IGN);
    setenv("PN_LOCAL_HOST_WAIT_MS", "5000", 1);
    g_port = lt_pick_port(54);
    lt_clean_dir(DIR);
    lt_make_dir(STATE);
    seed_state(STATE, 1);
    secret_of('a', g_key_a);
    secret_of('b', g_key_b);
    pid = spawn_box(DIR, g_port);
    if (pid <= 0) {
        fprintf(stderr, "could not start the box\n");
        return 1;
    }
    test_visibility();
    test_sharing();
    test_errors();
    pid = test_disconnected(pid);
    lt_stop(pid);

    printf("traces:\n");
    for (int run = 0; run < 2; run++) {
        const char *log = run ? DIR "/serve.log" : DIR "/serve.run1.log";
        CHECK(file_has(log, "serve: POST /local/sessions"), "the log is being written");
        CHECK(!file_has(log, SECRET) && !file_has(log, PUB), "no host_secret / host_pub in the log");
        for (int i = 0; i < g_n_macs; i++)
            clean &= !file_has(log, g_macs[i]);
    }
    CHECK(g_n_macs >= 5 && clean, "no ticket in either log");
    CHECK(file_has(STATE "/sessions.json", K1) && !file_has(STATE "/sessions.json", SECRET) &&
          !file_has(STATE "/sessions.json", PUB), "sessions.json: rows, no secret or pub");
    return lt_report("test_local_session_routes");
}
