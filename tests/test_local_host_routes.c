/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_host_routes.c — people registry P2.4: the host registry routes on a
 * forked `pantheon local` (spec §4.4), replaying the contract fixtures
 * hosts_register / hosts_wait / hosts_delete. The box runs with short test
 * timings (wait 1.5 s, disconnect 2 s, sweep 0.2 s) from the environment:
 *   - register: owner from the caller, host_addr from the peer IP + host_port or
 *     explicit; device_mismatch, device_unclaimed, key_taken, state disconnected;
 *   - wait: held to its deadline; delete from another device -> close removed;
 *     unknown key 404 not_registered; another person's row 403;
 *   - box stop -> close shutdown; restart -> reregister -> re-POST -> held (live);
 *   - no wait for the disconnect time -> reregister; waits skip the token bucket;
 *   - a revoked device key -> close revoked; a removed person -> close removed;
 *   - no wait in the log, no host_secret / host_pub in the log or sessions.json.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"
#include "local_fixture_replay.h"
#include "local_people_test_support.h"
#include "local_host_wait_support.h"

#define DIR    "build/test_local/host_routes"
#define STATE  DIR "/state"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define K1     "c-0123456789ab"
#define DEV_A  "d00000000000000aa"
#define DEV_B  "d00000000000000bb"
#define DEV_A2 "d000000000000a2a2"

static int g_port;
static char g_key_a[80], g_key_b[80];

static void reg_body(char *out, size_t cap, const char *key, const char *origin,
                     const char *state, const char *extra)
{
    snprintf(out, cap, "{\"key\":\"%s\",\"kind\":\"session\",\"origin_device\":\"%s\","
             "\"host_port\":53211,\"title\":\"Fix the relay\",\"model\":\"m\",\"instance\":2,"
             "\"state\":\"%s\",\"host_secret\":\"" SECRET "\",\"host_pub\":\"" PUB "\"%s}",
             key, origin, state, extra ? extra : "");
}

static int post_host(const char *secret, const char *key, const char *origin, const char *state,
                     char *type, size_t cap)
{
    char body[1024];

    reg_body(body, sizeof(body), key, origin, state, NULL);
    return fx_error(g_port, "POST", "/local/hosts", secret, body, type, cap);
}

/* Pair a device for person `person_key`'s person through an invite. */
static void pair_device(const char *inviter, const char *device_id, char *key_out, char *id_out)
{
    char *resp = NULL, code[32] = "", body[256];

    key_out[0] = id_out[0] = '\0';
    if (lt_http(g_port, "POST", "/local/invites", inviter, "{}", &resp) == 200)
        fx_field(resp, "code", NULL, code, sizeof(code));
    free(resp);
    resp = NULL;
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"argus\",\"device_id\":\"%s\","
             "\"device_name\":\"laptop\"}", code, device_id);
    if (lt_http(g_port, "POST", "/local/pair", NULL, body, &resp) == 200) {
        fx_field(resp, "key", NULL, key_out, 80);
        fx_field(resp, "id", NULL, id_out, 32);
    }
    free(resp);
}

static void test_register(void)
{
    jr_value *fx = fx_load("hosts_register.json");
    const char *subs[] = { "origin_device", "\"" DEV_A "\"", "host_secret", "\"" SECRET "\"",
                           "host_pub", "\"" PUB "\"", NULL };
    char *req = fx ? fx_request(fx, subs) : NULL, *resp = NULL, v[128], type[64], body[1024];
    char key_old[80] = "", code[32] = "";

    printf("register:\n");
    CHECK(req && fx_replay(g_port, fx, "POST", "/local/hosts", g_key_a, req, &resp),
          "POST /local/hosts replays the fixture");
    fx_field(resp, "host_addr", NULL, v, sizeof(v));
    CHECK(strcmp(v, "127.0.0.1:53211") == 0, "host_addr = peer IP + host_port");
    fx_field(resp, "owner", "handle", v, sizeof(v));
    CHECK(strcmp(v, "david") == 0, "owner = the caller's person");
    CHECK(strstr(resp, "\"seq\":0") && strstr(resp, "\"beat_s\":2") &&
          strstr(resp, "\"disconnect_s\":2"), "seq 0 and the box's timings");
    free(resp);
    resp = NULL;
    reg_body(body, sizeof(body), "c-00000000000e", DEV_A, "idle", ",\"host_addr\":\"10.0.0.5:4000\"");
    CHECK(lt_http(g_port, "POST", "/local/hosts", g_key_a, body, &resp) == 200 &&
          strstr(resp, "\"host_addr\":\"10.0.0.5:4000\""), "an explicit host_addr is kept");
    free(resp);

    CHECK(post_host(g_key_a, K1, DEV_B, "idle", type, sizeof(type)) == 409 &&
          strcmp(type, "device_mismatch") == 0 && fx_has_error(fx, 409, "device_mismatch"),
          "origin_device of another device: 409 device_mismatch");
    CHECK(post_host(g_key_a, K1, DEV_A, "disconnected", type, sizeof(type)) == 400 &&
          strcmp(type, "invalid_request_error") == 0, "state disconnected: 400");
    CHECK(post_host(g_key_b, K1, DEV_B, "idle", type, sizeof(type)) == 409 &&
          strcmp(type, "key_taken") == 0 && fx_has_error(fx, 409, "key_taken"),
          "another person's key: 409 key_taken");
    read_code(STATE, code);
    snprintf(body, sizeof(body), "{\"code\":\"%s\",\"label\":\"old\"}", code);
    resp = NULL;
    if (lt_http(g_port, "POST", "/local/pair", NULL, body, &resp) == 200)
        fx_field(resp, "key", NULL, key_old, sizeof(key_old));
    free(resp);
    CHECK(key_old[0] && post_host(key_old, "c-00000000000f", DEV_A, "idle", type,
                                  sizeof(type)) == 409 && strcmp(type, "device_unclaimed") == 0 &&
          fx_has_error(fx, 409, "device_unclaimed"), "a key with no device_id: device_unclaimed");
    free(req);
    jr_free(fx);
}

static void test_wait_and_delete(void)
{
    jr_value *fx = fx_load("hosts_wait.json"), *fd = fx_load("hosts_delete.json");
    char got[4096], key_a2[80], id_a2[32], type[64];
    uint64_t t0 = ms_now();
    int w = wait_open(g_port, g_key_a, K1, 0);

    printf("wait_and_delete:\n");
    CHECK(wait_read(w, got, sizeof(got), 5000) == 200 && wait_shape(fx, "response_timeout", got),
          "a wait with nothing to say returns at its deadline");
    CHECK(ms_now() - t0 >= 1400, "it was held for the wait time");

    w = wait_open(g_port, g_key_a, K1, 0);
    pair_device(g_key_a, DEV_A2, key_a2, id_a2);
    CHECK(key_a2[0] != '\0', "a second device of david");
    CHECK(silent_for(w, 100), "the wait is held");
    CHECK(fx_error(g_port, "POST", "/local/hosts/c-00000000000e/delete", g_key_b, "{}", type,
                   sizeof(type)) == 403 && strcmp(type, "forbidden") == 0 &&
          fx_has_error(fd, 403, "forbidden"), "another person cannot delete");
    CHECK(fx_replay(g_port, fd, "POST", "/local/hosts/" K1 "/delete", key_a2, "{}", NULL),
          "delete from the owner's other device replays the fixture");
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && wait_shape(fx, "response_close", got) &&
          strstr(got, "\"reason\":\"removed\""), "the held wait hears close removed");
    w = wait_open(g_port, g_key_a, K1, 0);
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && strstr(got, "\"reason\":\"removed\""),
          "the host's next wait hears it too");
    CHECK(fx_error(g_port, "POST", "/local/hosts/" K1 "/delete", g_key_a, "{}", type,
                   sizeof(type)) == 404 && strcmp(type, "not_found") == 0, "deleted: 404");
    CHECK(fx_error(g_port, "GET", "/local/hosts/c-ffffffffffff/wait?since=0", g_key_a, NULL,
                   type, sizeof(type)) == 404 && strcmp(type, "not_registered") == 0 &&
          fx_has_error(fx, 404, "not_registered"), "unknown key: 404 not_registered");
    CHECK(fx_error(g_port, "GET", "/local/hosts/c-00000000000e/wait?since=0", g_key_b, NULL,
                   type, sizeof(type)) == 403 && strcmp(type, "forbidden") == 0,
          "another person's row: 403");
    jr_free(fx);
    jr_free(fd);
}

/* Stop and start the box on the same state dir; the first run's log is kept. */
static pid_t restart_box(pid_t pid)
{
    lt_stop(pid);
    if (rename(DIR "/serve.log", DIR "/serve.run1.log") != 0)
        fprintf(stderr, "  warn: could not keep the first log\n");
    return spawn_box(DIR, g_port);
}

static pid_t test_restart_and_disconnect(pid_t pid)
{
    jr_value *fx = fx_load("hosts_wait.json");
    char got[4096], type[64];
    int w = -1, refused = 0;

    printf("restart_and_disconnect:\n");
    CHECK(post_host(g_key_a, "c-0000000000c3", DEV_A, "thinking", type, sizeof(type)) == 200,
          "register c3");
    w = wait_open(g_port, g_key_a, "c-0000000000c3", 0);
    CHECK(silent_for(w, 100), "held before the stop");
    pid = restart_box(pid);
    CHECK(pid > 0, "the box restarted");
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && strstr(got, "\"reason\":\"shutdown\""),
          "a box stop answers close shutdown");
    w = wait_open(g_port, g_key_a, "c-0000000000c3", 1);
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 &&
          wait_shape(fx, "response_reregister", got) && strstr(got, "\"seq\":2"),
          "after a restart the wait hears reregister, numbered above since");
    CHECK(post_host(g_key_a, "c-0000000000c3", DEV_A, "idle", type, sizeof(type)) == 200,
          "re-POST");
    w = wait_open(g_port, g_key_a, "c-0000000000c3", 2);
    CHECK(silent_for(w, 300), "live again: the wait is held");
    CHECK(wait_read(w, got, sizeof(got), 5000) == 200 && strstr(got, "\"events\":[]"),
          "and returns at its deadline");

    usleep(2600 * 1000);   /* the timing under test: no wait for > disconnect_s */
    w = wait_open(g_port, g_key_a, "c-0000000000c3", 2);
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && strstr(got, "\"type\":\"reregister\""),
          "no wait for the disconnect time: reregister");
    for (int i = 0; i < 70; i++) {
        w = wait_open(g_port, g_key_a, "c-0000000000c3", 3);
        refused += wait_read(w, got, sizeof(got), 3000) != 200;
    }
    CHECK(refused == 0, "70 waits in a row: none spends the token bucket (burst 60)");
    jr_free(fx);
    return pid;
}

static void test_revoke_and_remove(void)
{
    char got[4096], key_a2[80], id_a2[32], body[128], *resp = NULL, type[64];
    int w = -1;

    printf("revoke_and_remove:\n");
    pair_device(g_key_a, "d000000000000a3a3", key_a2, id_a2);
    CHECK(post_host(key_a2, "c-0000000000a3", "d000000000000a3a3", "idle", type,
                    sizeof(type)) == 200, "a host on the new device");
    w = wait_open(g_port, key_a2, "c-0000000000a3", 0);
    CHECK(silent_for(w, 100), "held");
    snprintf(body, sizeof(body), "{\"id\":\"%s\"}", id_a2);
    CHECK(lt_http(g_port, "POST", "/local/keys/revoke", g_key_a, body, NULL) == 200,
          "david revokes that device key");
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && strstr(got, "\"reason\":\"revoked\""),
          "its held wait hears close revoked");

    CHECK(post_host(g_key_b, "c-0000000000b6", DEV_B, "idle", type, sizeof(type)) == 200,
          "bob registers");
    w = wait_open(g_port, g_key_b, "c-0000000000b6", 0);
    CHECK(silent_for(w, 100), "bob's wait held");
    CHECK(lt_http(g_port, "POST", "/local/people/remove", g_key_a, "{\"id\":\"p00000002\"}",
                  &resp) == 200 && strstr(resp, "\"sessions_removed\":1"),
          "removing bob deletes his row");
    free(resp);
    CHECK(wait_read(w, got, sizeof(got), 3000) == 200 && strstr(got, "\"reason\":\"removed\""),
          "bob's held wait hears close removed");
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

    signal(SIGPIPE, SIG_IGN);
    setenv("PN_LOCAL_HOST_WAIT_MS", "1500", 1);
    setenv("PN_LOCAL_HOST_DISCONNECT_S", "2", 1);
    setenv("PN_LOCAL_HOST_SWEEP_MS", "200", 1);
    g_port = lt_pick_port(53);
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
    test_register();
    test_wait_and_delete();
    pid = test_restart_and_disconnect(pid);
    test_revoke_and_remove();
    lt_stop(pid);

    printf("traces:\n");
    for (int run = 0; run < 2; run++) {
        const char *log = run ? DIR "/serve.log" : DIR "/serve.run1.log";
        CHECK(!file_has(log, "/wait"), "no wait in the log");
        CHECK(file_has(log, "serve: POST /local/hosts"), "the log is being written");
        CHECK(!file_has(log, SECRET) && !file_has(log, PUB), "no host_secret / host_pub in the log");
    }
    CHECK(file_has(STATE "/sessions.json", "c-0000000000c3") &&
          !file_has(STATE "/sessions.json", SECRET) && !file_has(STATE "/sessions.json", PUB),
          "sessions.json holds rows but no host_secret / host_pub");
    return lt_report("test_local_host_routes");
}
