/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_hosts_hub.c — people registry P2.3: the hub's held waits (spec
 * §4.4, §6.3), driven over socketpairs (no network, no serve):
 *   - a held wait returns on an event (shares, with handles), at once when
 *     events are already queued, and at its deadline with no events;
 *   - a second wait for the same key answers the first `replaced`;
 *   - a peer that closes frees its slot; a deleted row answers close removed;
 *     a revoked device key answers close revoked; a sweep answers reregister;
 *   - the poll timeout follows the deadlines (no fast timer); the cap refuses;
 *   - stop answers every held wait close shutdown and refuses new ones.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include <poll.h>

#include "../src/local/hosts_hub.h"

#define STATE  "build/test_local/hosts_hub"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define DAVID  "p00000001"
#define BOB    "p00000002"
#define KEY    "c-0123456789ab"

static int handle_of(const char *person, char *out, size_t cap, void *user)
{
    (void)user;
    snprintf(out, cap, "%s", strcmp(person, BOB) == 0 ? "bob" : "someone");
    return 0;
}

static void add_row(pn_reg *reg, const char *key, long long now)
{
    pn_reg_host h = { .key = key, .kind = PN_REG_KIND_SESSION, .owner = DAVID,
                      .origin_device = "d0123456789abcdef", .host_addr = "192.0.2.20:53211",
                      .title = "t", .model = "m", .instance = 0, .state = PN_REG_STATE_IDLE,
                      .host_secret = SECRET, .host_pub = PUB };
    if (pn_reg_upsert(reg, &h, now, NULL) != PN_REG_OK)
        fprintf(stderr, "  setup: upsert %s failed\n", key);
}

static pn_hub *hub_with(pn_reg *reg, int wait_ms, int sweep_ms, int max_waits)
{
    pn_hub_cfg cfg = { .wait_ms = wait_ms, .sweep_ms = sweep_ms, .max_waits = max_waits,
                       .handle_of = handle_of };
    return pn_hub_start(reg, &cfg);
}

/* A wait: a socketpair whose [0] the hub takes; [1] is the client's end. */
static int add(pn_hub *hub, int sp[2], const char *key, const char *key_id, long long since)
{
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0)
        return -1;
    if (pn_hub_add_wait(hub, sp[0], key, key_id, since) != 0) {
        close(sp[0]);
        return -1;
    }
    return 0;
}

/* Read the client's end until EOF or timeout_ms of silence. */
static void read_answer(int fd, char *out, size_t cap, int timeout_ms)
{
    size_t len = 0;
    struct pollfd p = { .fd = fd, .events = POLLIN };

    while (len + 1 < cap && poll(&p, 1, timeout_ms) > 0) {
        ssize_t n = read(fd, out + len, cap - len - 1);
        if (n <= 0)
            break;
        len += (size_t)n;
    }
    out[len] = '\0';
    close(fd);
}

/* 1 when the client's end has nothing to read for ms (the wait is held). */
static int silent_for(int fd, int ms)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    return poll(&p, 1, ms) == 0;
}

/* Bounded wait for the held count to reach n (the hub thread acts async). */
static int held_reaches(pn_hub *hub, int n)
{
    for (int i = 0; i < 400; i++) {
        if (pn_hub_waits_held(hub) == n)
            return 1;
        usleep(5000);
    }
    return 0;
}

static uint64_t ms_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static pn_reg *fresh(void)
{
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    return pn_reg_open(STATE "/sessions.json");
}

static void test_event_and_queued(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = NULL;
    pn_reg_event relay = { .type = PN_REG_EV_RELAY, .relay_expires = (long long)time(NULL) + 10 };
    char got[4096];
    int sp[2];

    printf("event_and_queued:\n");
    add_row(reg, KEY, (long long)time(NULL));
    hub = hub_with(reg, 5000, 0, 0);
    CHECK(hub && add(hub, sp, KEY, "k1", 0) == 0 && pn_hub_waits_held(hub) == 1, "wait held");
    CHECK(silent_for(sp[1], 100), "no answer without an event");
    pn_reg_share(reg, KEY, DAVID, BOB, PN_REG_ROLE_VIEW);
    pn_hub_notify(hub, KEY);
    read_answer(sp[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "HTTP/1.1 200") && strstr(got, "{\"seq\":1,\"events\":[{\"seq\":1,"
          "\"type\":\"shares\",\"shares\":[{\"person\":\"p00000002\",\"handle\":\"bob\","
          "\"role\":\"view\"}]}]}"), "returns on the shares event, with handles");
    CHECK(held_reaches(hub, 0), "slot freed after the answer");

    snprintf(relay.relay_id, sizeof(relay.relay_id), "r-0011223344556677");
    pn_reg_push_event(reg, KEY, &relay);
    CHECK(add(hub, sp, KEY, "k1", 1) == 0, "wait since 1");
    read_answer(sp[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "\"seq\":2,\"events\":[{\"seq\":2,\"type\":\"relay\","
                      "\"id\":\"r-0011223344556677\",\"expires_in\":"),
          "an already-queued event answers at once");
    pn_hub_free(hub);
    pn_reg_close(reg);
}

static void test_timeout_replace_peer_left(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = NULL;
    char got[4096];
    int a[2], b[2];
    uint64_t t0 = 0, waited = 0;

    printf("timeout_replace_peer_left:\n");
    add_row(reg, KEY, (long long)time(NULL));
    hub = hub_with(reg, 250, 0, 0);
    t0 = ms_now();
    CHECK(hub && add(hub, a, KEY, "k1", 7) == 0, "wait held");
    read_answer(a[1], got, sizeof(got), 3000);
    waited = ms_now() - t0;
    CHECK(waited >= 240 && waited < 2000, "answered at its deadline");
    CHECK(strstr(got, "{\"seq\":7,\"events\":[]}") != NULL, "a timeout carries no events");
    pn_hub_free(hub);

    hub = hub_with(reg, 5000, 0, 0);
    CHECK(add(hub, a, KEY, "k1", 7) == 0 && add(hub, b, KEY, "k1", 7) == 0, "two waits");
    read_answer(a[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "{\"seq\":7,\"events\":[],\"replaced\":true}") != NULL,
          "the first wait is answered replaced");
    CHECK(pn_hub_waits_held(hub) == 1 && silent_for(b[1], 100), "the second is held");
    close(b[1]);
    CHECK(held_reaches(hub, 0), "a peer that left frees its slot");
    pn_hub_free(hub);
    pn_reg_close(reg);
}

static void test_close_events(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = NULL;
    char got[4096];
    int a[2], b[2], c[2];

    printf("close_events:\n");
    add_row(reg, KEY, (long long)time(NULL));
    add_row(reg, "c-00000000000b", (long long)time(NULL));
    hub = hub_with(reg, 5000, 0, 0);
    CHECK(hub && add(hub, a, KEY, "k1", 0) == 0, "wait held");
    pn_reg_delete(reg, KEY, DAVID, (long long)time(NULL));
    pn_hub_notify(hub, KEY);
    read_answer(a[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "{\"seq\":1,\"events\":[{\"seq\":1,\"type\":\"close\","
                      "\"reason\":\"removed\"}]}"), "a deleted row: close removed");

    CHECK(add(hub, b, "c-00000000000b", "k9", 3) == 0, "a wait of device k9");
    CHECK(pn_hub_close_device(hub, "k1", PN_REG_CLOSE_REVOKED) == 0, "another key: untouched");
    CHECK(pn_hub_close_device(hub, "k9", PN_REG_CLOSE_REVOKED) == 1, "k9 revoked");
    read_answer(b[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "{\"seq\":4,\"events\":[{\"seq\":4,\"type\":\"close\","
                      "\"reason\":\"revoked\"}]}"), "close revoked, numbered above since");

    CHECK(add(hub, c, "c-00000000000b", "k9", 0) == 0, "a wait before stop");
    pn_hub_stop(hub);
    read_answer(c[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "\"type\":\"close\",\"reason\":\"shutdown\"") != NULL,
          "stop answers close shutdown");
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, c) == 0 &&
          pn_hub_add_wait(hub, c[0], "c-00000000000b", "k9", 0) == -1,
          "a stopped hub refuses new waits");
    close(c[0]);
    close(c[1]);
    pn_hub_free(hub);
    pn_reg_close(reg);
}

static void test_sweep_cap_timer(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = NULL;
    char got[4096];
    int a[2], b[2], c[2];
    unsigned long wakeups = 0;

    printf("sweep_cap_timer:\n");
    add_row(reg, KEY, (long long)time(NULL) - 200);   /* silent for 200 s */
    hub = hub_with(reg, 5000, 100, 0);
    CHECK(hub && add(hub, a, KEY, "k1", 0) == 0, "wait on a silent row");
    read_answer(a[1], got, sizeof(got), 3000);
    CHECK(strstr(got, "\"type\":\"reregister\"") != NULL, "the sweep answers reregister");
    pn_hub_free(hub);

    add_row(reg, KEY, (long long)time(NULL));
    add_row(reg, "c-00000000000b", (long long)time(NULL));
    add_row(reg, "c-00000000000c", (long long)time(NULL));
    hub = hub_with(reg, 1500, 0, 2);
    CHECK(add(hub, a, KEY, "k1", 1) == 0 && add(hub, b, "c-00000000000b", "k1", 0) == 0,
          "two waits fit a cap of 2");
    CHECK(add(hub, c, "c-00000000000c", "k1", 0) == -1, "the third is refused");
    close(c[1]);
    close(b[1]);
    CHECK(held_reaches(hub, 1), "one wait left");
    wakeups = pn_hub_wakeups(hub);
    read_answer(a[1], got, sizeof(got), 4000);
    CHECK(strstr(got, "\"events\":[]") != NULL, "the held wait timed out");
    CHECK(pn_hub_wakeups(hub) - wakeups <= 3, "1.5 s held: the hub woke only for its deadline");
    pn_hub_free(hub);
    pn_reg_close(reg);
}

int main(void)
{
    signal(SIGPIPE, SIG_IGN);
    test_event_and_queued();
    test_timeout_replace_peer_left();
    test_close_events();
    test_sweep_cap_timer();
    return lt_report("test_local_hosts_hub");
}
