/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_hub_relays.c — people registry P4.1: relay pairs in the hub
 * (spec §4.6, §6.3), driven over socketpairs (no network, no serve):
 *   - open queues `relay {id}` on the row; only the row's owner adds the host
 *     leg; both legs read the 200 header block first;
 *   - 5 MiB each way arrives intact through a stalled, then slow, reader: the
 *     writers stall at a bounded amount and the relay never holds more than
 *     one 16 KiB buffer per direction (backpressure, no growth);
 *   - either side closing closes both; a pending client that sends early or
 *     leaves is dropped;
 *   - no host leg in time -> 504 relay_timeout, then the id is gone;
 *   - caps (box, per person) refuse; one relay per nonce; a refused open does
 *     not use up its nonce; stop closes every relay and refuses new ones.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include <poll.h>
#include <pthread.h>

#include "../src/local/hosts_hub.h"

#define STATE  "build/test_local/hub_relays"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define DAVID  "p00000001"
#define BOB    "p00000002"
#define CAROL  "p00000003"
#define KEY    "c-0123456789ab"
#define BULK   (5u * 1024u * 1024u)
#define HEAD_FMT "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n" \
                 "X-Pantheon-Relay: %s\r\nConnection: close\r\n\r\n"

static pn_reg *fresh(void)
{
    pn_reg *reg = NULL;
    pn_reg_host h = { .key = KEY, .kind = PN_REG_KIND_SESSION, .owner = DAVID,
                      .origin_device = "d0123456789abcdef", .host_addr = "192.0.2.20:53211",
                      .title = "t", .model = "m", .instance = 0, .state = PN_REG_STATE_IDLE,
                      .host_secret = SECRET, .host_pub = PUB };

    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    reg = pn_reg_open(STATE "/sessions.json");
    if (!reg || pn_reg_upsert(reg, &h, (long long)time(NULL), NULL) != PN_REG_OK)
        fprintf(stderr, "  setup: registry\n");
    return reg;
}

static pn_hub *hub_with(pn_reg *reg, int host_ms, int max_relays, int per_person)
{
    pn_hub_cfg cfg = { .wait_ms = 5000,
                       .relays = { .max_relays = max_relays, .per_person = per_person,
                                   .host_ms = host_ms } };
    return pn_hub_start(reg, &cfg);
}

/* Open a relay for `person` with `nonce` over a fresh socketpair; [1] is the
 * client's end. Returns the hub's answer; on refusal both ends are closed. */
static int open_as(pn_hub *hub, int sp[2], const char *person, const char *nonce, char *id)
{
    pn_hub_relay_open_args a = { .key = KEY, .client_person = person, .owner = DAVID,
                                 .nonce = nonce, .nonce_until = (long long)time(NULL) + 90 };
    int rc = 0;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0)
        return -100;
    rc = pn_hub_relay_open(hub, sp[0], &a, id);
    if (rc != PN_HUB_RELAY_OK) {
        close(sp[0]);
        close(sp[1]);
        sp[0] = sp[1] = -1;
    }
    return rc;
}

/* Read exactly the header block (up to the blank line); 1 when it matches. */
static int read_header(int fd, const char *id)
{
    char want[256], got[256];
    size_t len = 0;
    struct pollfd p = { .fd = fd, .events = POLLIN };

    snprintf(want, sizeof(want), HEAD_FMT, id);
    while (len < strlen(want) && poll(&p, 1, 3000) > 0) {
        ssize_t n = read(fd, got + len, 1);
        if (n <= 0)
            break;
        len++;
    }
    got[len] = '\0';
    return strcmp(got, want) == 0;
}

/* 1 when fd reads EOF (or an error) within ms, discarding any bytes first. */
static int sees_eof(int fd, int ms)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    char buf[4096];

    while (poll(&p, 1, ms) > 0) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            return 1;
    }
    return 0;
}

static int relays_reach(pn_hub *hub, int n)
{
    for (int i = 0; i < 400; i++) {
        if (pn_hub_relays_held(hub) == n)
            return 1;
        usleep(5000);
    }
    return 0;
}

static uint8_t pattern(int dir, size_t i)
{
    return (uint8_t)((i * (dir ? 131u : 31u) + (i >> 13) + (unsigned)dir * 7u) & 0xffu);
}

typedef struct {
    int    fd;
    int    dir;
    size_t sent;   /* read by main under __atomic */
} bulk_writer;

static void *write_bulk(void *arg)
{
    bulk_writer *w = arg;
    uint8_t chunk[8192];

    while (__atomic_load_n(&w->sent, __ATOMIC_RELAXED) < BULK) {
        size_t at = __atomic_load_n(&w->sent, __ATOMIC_RELAXED);
        size_t n = BULK - at < sizeof(chunk) ? BULK - at : sizeof(chunk);
        ssize_t k = 0;
        for (size_t i = 0; i < n; i++)
            chunk[i] = pattern(w->dir, at + i);
        k = write(w->fd, chunk, n);
        if (k <= 0)
            break;
        __atomic_store_n(&w->sent, at + (size_t)k, __ATOMIC_RELAXED);
    }
    return NULL;
}

static void cap_buffers(int fd)
{
    int size = 65536;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
}

/* Wait until both writers stop advancing (they are blocked on a full pipe). */
static void until_stalled(bulk_writer *a, bulk_writer *b)
{
    size_t last_a = (size_t)-1, last_b = (size_t)-1;

    for (int i = 0; i < 50; i++) {
        size_t sa = __atomic_load_n(&a->sent, __ATOMIC_RELAXED);
        size_t sb = __atomic_load_n(&b->sent, __ATOMIC_RELAXED);
        if (sa == last_a && sb == last_b)
            return;
        last_a = sa;
        last_b = sb;
        poll(NULL, 0, 200);
    }
}

/* Read both directions slowly to the end, checking every byte. */
static int read_both_slowly(int c, int h, size_t *got_c, size_t *got_h)
{
    uint8_t buf[4096];
    int bad = 0;

    while (*got_c < BULK || *got_h < BULK) {
        struct pollfd p[2] = { { .fd = c, .events = POLLIN }, { .fd = h, .events = POLLIN } };
        if (poll(p, 2, 5000) <= 0)
            break;
        for (int s = 0; s < 2; s++) {
            size_t *got = s ? got_h : got_c;
            ssize_t n = 0;
            if (!(p[s].revents & POLLIN))
                continue;
            n = read(p[s].fd, buf, sizeof(buf));
            if (n <= 0)
                return 0;
            for (ssize_t i = 0; i < n; i++)   /* the client reads what the host wrote */
                bad |= buf[i] != pattern(s ? 0 : 1, *got + (size_t)i);
            *got += (size_t)n;
            if ((*got & 0xffff) < (size_t)n)
                usleep(500);   /* a slow reader: ~4 KiB at a time, a pause per 64 KiB */
        }
    }
    return !bad;
}

static void test_bulk_backpressure(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = hub_with(reg, 5000, 0, 0);
    pn_reg_event ev[PN_REG_QUEUE_MAX];
    bulk_writer up = { .dir = 0 }, down = { .dir = 1 };
    pthread_t tu, td;
    char id[PN_HUB_RELAY_ID_MAX] = "";
    size_t got_c = 0, got_h = 0;
    long long seq = 0;
    int c[2], h[2], n = 0;

    printf("bulk_backpressure:\n");
    CHECK(hub && open_as(hub, c, BOB, "00112233445566778899aabbccddeeff", id) == 0 &&
          strncmp(id, "r-", 2) == 0 && strlen(id) == 18, "open: held, id r-<16 hex>");
    n = pn_reg_events_since(reg, KEY, 0, ev, PN_REG_QUEUE_MAX, &seq, NULL, NULL);
    CHECK(n == 1 && ev[0].type == PN_REG_EV_RELAY && strcmp(ev[0].relay_id, id) == 0 &&
          ev[0].relay_expires >= (long long)time(NULL) + 4, "the row has relay {id}");
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, h) == 0, "host pair");
    CHECK(pn_hub_relay_host(hub, h[0], id, BOB) == PN_HUB_RELAY_E_NOT_OWNER,
          "a host leg from another person: refused");
    CHECK(pn_hub_relay_host(hub, h[0], "r-0000000000000000", DAVID) == PN_HUB_RELAY_E_NOT_FOUND,
          "an unknown id: not found");
    CHECK(pn_hub_relay_host(hub, h[0], id, DAVID) == PN_HUB_RELAY_OK, "the owner's host leg");
    CHECK(read_header(c[1], id) && read_header(h[1], id), "both legs read the header block");
    cap_buffers(c[1]);
    cap_buffers(h[1]);
    up.fd = c[1];
    down.fd = h[1];
    pthread_create(&tu, NULL, write_bulk, &up);
    pthread_create(&td, NULL, write_bulk, &down);
    until_stalled(&up, &down);
    CHECK(up.sent < 1024 * 1024 && down.sent < 1024 * 1024,
          "nobody reads: both writers stall under 1 MiB");
    CHECK(pn_hub_relay_buffered_peak(hub) <= 2 * PN_HUB_RELAY_BUF,
          "stalled: at most one 16 KiB buffer per direction");
    CHECK(read_both_slowly(c[1], h[1], &got_c, &got_h) && got_c == BULK && got_h == BULK,
          "5 MiB each way arrives intact through a slow reader");
    pthread_join(tu, NULL);
    pthread_join(td, NULL);
    CHECK(pn_hub_relay_buffered_peak(hub) <= 2 * PN_HUB_RELAY_BUF, "no growth over the run");
    close(c[1]);
    CHECK(sees_eof(h[1], 3000), "the client closing closes the host leg");
    close(h[1]);
    CHECK(relays_reach(hub, 0), "the pair is gone");

    CHECK(open_as(hub, c, BOB, "10112233445566778899aabbccddeeff", id) == 0 &&
          socketpair(AF_UNIX, SOCK_STREAM, 0, h) == 0 &&
          pn_hub_relay_host(hub, h[0], id, DAVID) == 0 && read_header(c[1], id) &&
          read_header(h[1], id), "a second pair");
    close(h[1]);
    CHECK(sees_eof(c[1], 3000), "the host closing closes the client leg");
    close(c[1]);
    pn_hub_free(hub);
    pn_reg_close(reg);
}

static void test_timeout_and_early(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = hub_with(reg, 300, 0, 0);
    char id[PN_HUB_RELAY_ID_MAX] = "", got[512] = "";
    struct pollfd p;
    uint64_t t0 = pn_hub_relays_now_ms();
    ssize_t n = 0;
    int c[2], h[2];

    printf("timeout_and_early:\n");
    CHECK(hub && open_as(hub, c, BOB, "20112233445566778899aabbccddeeff", id) == 0, "open");
    p.fd = c[1];
    p.events = POLLIN;
    if (poll(&p, 1, 3000) > 0 && (n = read(c[1], got, sizeof(got) - 1)) > 0)
        got[n] = '\0';
    CHECK(strstr(got, "HTTP/1.1 504") && strstr(got, "\"type\":\"relay_timeout\""),
          "no host leg in time: 504 relay_timeout");
    CHECK(pn_hub_relays_now_ms() - t0 >= 280, "answered at the deadline, not before");
    close(c[1]);
    CHECK(relays_reach(hub, 0) && socketpair(AF_UNIX, SOCK_STREAM, 0, h) == 0 &&
          pn_hub_relay_host(hub, h[0], id, DAVID) == PN_HUB_RELAY_E_NOT_FOUND,
          "the timed-out id is gone");
    close(h[0]);
    close(h[1]);
    pn_hub_free(hub);

    hub = hub_with(reg, 5000, 0, 0);
    CHECK(open_as(hub, c, BOB, "30112233445566778899aabbccddeeff", id) == 0 &&
          write(c[1], "x", 1) == 1, "a pending client sends before the header");
    CHECK(sees_eof(c[1], 3000) && relays_reach(hub, 0), "it is dropped");
    close(c[1]);
    CHECK(open_as(hub, c, BOB, "40112233445566778899aabbccddeeff", id) == 0, "another pending");
    close(c[1]);
    CHECK(relays_reach(hub, 0), "a pending client that leaves frees its slot");
    pn_hub_free(hub);
    pn_reg_close(reg);
}

static void test_caps_and_nonces(void)
{
    pn_reg *reg = fresh();
    pn_hub *hub = hub_with(reg, 5000, 3, 2);
    char id[PN_HUB_RELAY_ID_MAX] = "";
    int a[2], b[2], c[2], d[2];
    pn_hub_relay_open_args gone = { .key = "c-ffffffffffff", .client_person = BOB,
                                    .owner = DAVID, .nonce = "50112233445566778899aabbccddeeff",
                                    .nonce_until = (long long)time(NULL) + 90 };

    printf("caps_and_nonces:\n");
    CHECK(hub && open_as(hub, a, BOB, "a0112233445566778899aabbccddeeff", id) == 0,
          "bob's first");
    CHECK(open_as(hub, d, CAROL, "a0112233445566778899aabbccddeeff", id) == PN_HUB_RELAY_E_NONCE,
          "the same nonce again: refused");
    CHECK(open_as(hub, b, BOB, "b0112233445566778899aabbccddeeff", id) == 0, "bob's second");
    CHECK(open_as(hub, d, BOB, "c0112233445566778899aabbccddeeff", id) == PN_HUB_RELAY_E_FULL,
          "bob's third: over the per-person cap");
    CHECK(open_as(hub, c, CAROL, "d0112233445566778899aabbccddeeff", id) == 0, "carol's first");
    CHECK(open_as(hub, d, CAROL, "e0112233445566778899aabbccddeeff", id) == PN_HUB_RELAY_E_FULL,
          "the box is at its cap");
    close(a[1]);
    CHECK(relays_reach(hub, 2), "bob's first left");
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, d) == 0 &&
          pn_hub_relay_open(hub, d[0], &gone, id) == PN_HUB_RELAY_E_INVALID,
          "a row that is gone: refused, fd still ours");
    close(d[0]);
    close(d[1]);
    CHECK(open_as(hub, d, BOB, "c0112233445566778899aabbccddeeff", id) == 0,
          "the nonce of a refused open still opens");
    pn_hub_stop(hub);
    CHECK(sees_eof(b[1], 3000) && sees_eof(c[1], 3000) && sees_eof(d[1], 3000),
          "stop closes every relay");
    CHECK(open_as(hub, a, CAROL, "f0112233445566778899aabbccddeeff", id) ==
          PN_HUB_RELAY_E_STOPPED, "a stopped hub refuses relays");
    close(b[1]);
    close(c[1]);
    close(d[1]);
    pn_hub_free(hub);
    pn_reg_close(reg);
}

int main(void)
{
    signal(SIGPIPE, SIG_IGN);
    test_bulk_backpressure();
    test_timeout_and_early();
    test_caps_and_nonces();
    return lt_report("test_local_hub_relays");
}
