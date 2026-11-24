/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_relay_routes.c — people registry P4.2: the relay routes on a
 * forked `pantheon local` (spec §4.6; Argus §10.8 test 5, box side), replaying
 * relay_open / relay_host (head and error types) and hosts_wait's relay event:
 *   - direct is blocked (nothing listens on host_addr) -> bob opens a relay with
 *     his control ticket, david's host hears `relay {id}` on its wait, joins,
 *     both legs read the 200 head, and a §5-shaped exchange (hello with the
 *     ticket, which the host checks; a submit; an entry_add) crosses it, a
 *     marker string each way; closing one leg closes the other;
 *   - one relay per nonce (409 ticket_used); bad MAC, another person's ticket,
 *     another session's ticket -> 403 bad_ticket; no ticket 400; unknown key
 *     404; device_unclaimed 409; a disconnected row 409 after a restart;
 *   - host leg: unknown id 404, not the owner 403; no host in time -> 504
 *     relay_timeout and the id is gone; the per-person cap -> 503 relay_full
 *     with Retry-After: 1;
 *   - the marker appears nowhere under the test dir (state dir, sessions.json,
 *     store) or in either log; no ticket in the logs.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"
#include "local_fixture_replay.h"
#include "local_people_test_support.h"
#include "local_host_wait_support.h"

#include <dirent.h>

#include "../src/local/attach_ticket.h"
#include "../src/local/hub_relays.h"

#define TEST_DIR "build/test_local/relay_routes"
#define STATE   TEST_DIR "/state"
#define SECRET  "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define SECRET2 "6ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB     "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define CPUB    "8f40c5adb68f25624ae5b214ea767a6ec94d829d3d7b5e1ad1ba6f3e2138285f"
#define K1      "c-0123456789ab"
#define K2      "c-00000000000e"
#define DEV_A   "d00000000000000aa"
#define MARKER  "PN-RELAY-MARKER-4f1c9e2a7b"
#define HOST_MS "1500"

static int g_port, g_closed_port;
static char g_key_a[80], g_key_b[80];
static long long g_since;
static char g_macs[24][PN_SHA256_HEX_LEN + 1];   /* tickets' MACs, searched for in the logs */
static int g_n_macs;

/* A loopback port nothing listens on: direct attaches to it are refused. */
static int closed_port(void)
{
    struct sockaddr_in a = { .sin_family = AF_INET };
    socklen_t len = sizeof(a);
    int fd = socket(AF_INET, SOCK_STREAM, 0), port = 0;

    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd >= 0 && bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0 &&
        getsockname(fd, (struct sockaddr *)&a, &len) == 0)
        port = ntohs(a.sin_port);
    if (fd >= 0)
        close(fd);
    return port;
}

static int post_host(const char *key, const char *secret)
{
    char body[1024];

    snprintf(body, sizeof(body), "{\"key\":\"%s\",\"kind\":\"session\",\"origin_device\":\""
             DEV_A "\",\"host_port\":%d,\"title\":\"Fix the relay\",\"model\":\"m\","
             "\"instance\":2,\"state\":\"idle\",\"host_secret\":\"%s\",\"host_pub\":\"" PUB
             "\"}", key, g_closed_port, secret);
    return lt_http(g_port, "POST", "/local/hosts", g_key_a, body, NULL);
}

/* An attach ticket for `secret_key` on `key` into ticket (empty on failure). */
static void attach(const char *secret_key, const char *key, char ticket[PN_TICKET_MAX])
{
    char path[96], *resp = NULL;

    snprintf(path, sizeof(path), "/local/sessions/%s/attach", key);
    ticket[0] = '\0';
    if (lt_http(g_port, "POST", path, secret_key, "{\"cpub\":\"" CPUB "\"}", &resp) == 200)
        fx_field(resp, "ticket", NULL, ticket, PN_TICKET_MAX);
    if (g_n_macs < 24 && strchr(ticket, '.'))
        snprintf(g_macs[g_n_macs++], sizeof(g_macs[0]), "%s", strchr(ticket, '.') + 1);
    free(resp);
}

/* Send a POST without reading the answer; the socket, or -1. */
static int raw_post(const char *path, const char *secret_key, const char *body)
{
    struct sockaddr_in a = { .sin_family = AF_INET };
    char head[768];
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int n = snprintf(head, sizeof(head), "POST %s HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
                     "%s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
                     path, secret_key, strlen(body), body);

    a.sin_port = htons((uint16_t)g_port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        write(fd, head, (size_t)n) != n) {
        if (fd >= 0)
            close(fd);
        return -1;
    }
    return fd;
}

static int open_relay(const char *secret_key, const char *key, const char *ticket)
{
    char path[96], body[PN_TICKET_MAX + 32];

    snprintf(path, sizeof(path), "/local/relay/%s", key);
    snprintf(body, sizeof(body), "{\"ticket\":\"%s\"}", ticket);
    return raw_post(path, secret_key, body);
}

/* Read up to the blank line (byte by byte: nothing after the head is consumed). */
static void read_head(int fd, char *out, size_t cap)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    size_t len = 0;

    while (len + 1 < cap && poll(&p, 1, 5000) > 0 && read(fd, out + len, 1) == 1) {
        len++;
        if (len >= 4 && memcmp(out + len - 4, "\r\n\r\n", 4) == 0)
            break;
    }
    out[len] = '\0';
}

/* Read until `needle` shows (or 5 s of silence); 1 when it did. */
static int read_until(int fd, const char *needle, char *out, size_t cap)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    size_t len = 0;

    out[0] = '\0';
    while (len + 1 < cap && !strstr(out, needle) && poll(&p, 1, 5000) > 0) {
        ssize_t n = read(fd, out + len, cap - len - 1);
        if (n <= 0)
            break;
        len += (size_t)n;
        out[len] = '\0';
    }
    return strstr(out, needle) != NULL;
}

/* 1 when fd reads EOF within 5 s (bytes before it are discarded). */
static int sees_eof(int fd)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    char buf[1024];

    while (poll(&p, 1, 5000) > 0) {
        if (read(fd, buf, sizeof(buf)) <= 0)
            return 1;
    }
    return 0;
}

/* The fixture's head with its example id replaced by `id`. */
static void fixture_head(const jr_value *fx, const char *id, char *out, size_t cap)
{
    const char *h = jr_as_string(jr_object_get(fx, "response_head"));
    const char *at = h ? strstr(h, "r-0011223344556677") : NULL;

    if (!at) {
        snprintf(out, cap, "(no head in the fixture)");
        return;
    }
    snprintf(out, cap, "%.*s%s%s", (int)(at - h), h, id, at + 18);
}

/* Hold david's wait from g_since; the last relay id heard into id. */
static int wait_for_relay(const jr_value *fw, char id[24])
{
    char got[4096], *at = NULL;
    int st = wait_read(wait_open(g_port, g_key_a, K1, g_since), got, sizeof(got), 7000);
    jr_value *v = jr_parse(got);

    g_since = (long long)jr_as_number(jr_object_get(v, "seq"), (double)g_since);
    jr_free(v);
    id[0] = '\0';
    for (char *p = strstr(got, "\"type\":\"relay\",\"id\":\""); p;
         p = strstr(p + 1, "\"type\":\"relay\",\"id\":\""))
        at = p;
    if (at)
        snprintf(id, 24, "%.18s", at + strlen("\"type\":\"relay\",\"id\":\""));
    return st == 200 && id[0] && wait_shape(fw, "response", got);
}

/* Drain david's queued events (the share) so the next answer holds relays only. */
static void drain_wait(void)
{
    char got[4096];
    int w = wait_open(g_port, g_key_a, K1, g_since);
    jr_value *v = NULL;

    if (wait_read(w, got, sizeof(got), 7000) == 200) {
        v = jr_parse(got);
        g_since = (long long)jr_as_number(jr_object_get(v, "seq"), (double)g_since);
        jr_free(v);
    }
}

/* The host's side of hello: the ticket checks under its secret, as Argus D7 does. */
static int host_accepts_hello(const char *line, const char *want_ticket)
{
    jr_value *v = jr_parse(line);
    const char *t = jr_as_string(jr_object_get(v, "ticket"));
    uint8_t secret[PN_TICKET_SECRET_LEN];
    pn_ticket_fields f = {0};
    int ok = 0;

    pn_hex_decode(SECRET, secret, sizeof(secret));
    ok = t && strcmp(t, want_ticket) == 0 &&
         pn_ticket_check(secret, K1, t, (long long)time(NULL), &f) == PN_TICKET_OK &&
         f.role == PN_TICKET_ROLE_CONTROL && strcmp(f.cpub, CPUB) == 0;
    jr_free(v);
    return ok;
}

static void test_relay_exchange(void)
{
    jr_value *fo = fx_load("relay_open.json"), *fh = fx_load("relay_host.json");
    jr_value *fw = fx_load("hosts_wait.json");
    char ticket[PN_TICKET_MAX], id[24] = "", want[256], head[256], buf[8192], line[1024];
    char path[96], type[64];
    struct sockaddr_in a = { .sin_family = AF_INET };
    int c = -1, h = -1, direct = socket(AF_INET, SOCK_STREAM, 0);

    printf("relay_exchange:\n");
    a.sin_port = htons((uint16_t)g_closed_port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(direct >= 0 && connect(direct, (struct sockaddr *)&a, sizeof(a)) != 0,
          "direct to host_addr is blocked");
    if (direct >= 0)
        close(direct);
    attach(g_key_b, K1, ticket);
    CHECK(ticket[0] != '\0', "bob's control ticket");
    c = open_relay(g_key_b, K1, ticket);
    CHECK(c >= 0 && silent_for(c, 100), "bob's relay is held");
    CHECK(fw && wait_for_relay(fw, id) && strncmp(id, "r-", 2) == 0,
          "david's wait hears relay {id} (hosts_wait shape)");
    snprintf(path, sizeof(path), "/local/relay/%s/host", id);
    CHECK(fx_error(g_port, "POST", path, g_key_b, "{}", type, sizeof(type)) == 403 &&
          strcmp(type, "forbidden") == 0 && fh && fx_has_error(fh, 403, "forbidden"),
          "bob cannot join as the host: 403");
    h = raw_post(path, g_key_a, "{}");
    fixture_head(fo, id, want, sizeof(want));
    read_head(c, head, sizeof(head));
    CHECK(fo && strcmp(head, want) == 0, "the client reads relay_open's head");
    fixture_head(fh, id, want, sizeof(want));
    read_head(h, head, sizeof(head));
    CHECK(strcmp(head, want) == 0, "the host reads relay_host's head");

    snprintf(line, sizeof(line), "{\"v\":1,\"type\":\"hello\",\"client\":\"bob-phone\","
             "\"ticket\":\"%s\"}\n{\"v\":1,\"type\":\"submit\",\"text\":\"" MARKER "\"}\n",
             ticket);
    CHECK(write(c, line, strlen(line)) == (ssize_t)strlen(line) &&
          read_until(h, MARKER "\"}\n", buf, sizeof(buf)), "hello + submit reach the host");
    if (strchr(buf, '\n'))
        *strchr(buf, '\n') = '\0';
    CHECK(host_accepts_hello(buf, ticket), "the host checks the hello's ticket");
    snprintf(line, sizeof(line), "{\"v\":1,\"type\":\"entry_add\",\"seq\":1,\"index\":0,"
             "\"entry\":{\"kind\":\"answer\",\"text\":\"" MARKER "\"}}\n");
    CHECK(write(h, line, strlen(line)) == (ssize_t)strlen(line) &&
          read_until(c, MARKER, buf, sizeof(buf)) && strcmp(buf, line) == 0,
          "the host's entry_add reaches the client intact");
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_b, "{\"ticket\":\"x\"}", type,
                   sizeof(type)) == 403, "(a bad ticket while the relay is open: 403)");
    snprintf(line, sizeof(line), "{\"ticket\":\"%s\"}", ticket);
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_b, line, type, sizeof(type)) ==
          409 && strcmp(type, "ticket_used") == 0 && fx_has_error(fo, 409, "ticket_used"),
          "the same ticket again: 409 ticket_used");
    close(c);
    CHECK(sees_eof(h), "the client closing closes the host leg");
    close(h);
    jr_free(fo);
    jr_free(fh);
    jr_free(fw);
}

static void test_open_errors(void)
{
    jr_value *fo = fx_load("relay_open.json");
    char ticket[PN_TICKET_MAX], body[PN_TICKET_MAX + 32], type[64], code[32] = "";
    char key_old[80] = "", pair[160], *resp = NULL;

    printf("open_errors:\n");
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_b, "{}", type, sizeof(type)) ==
          400 && strcmp(type, "invalid_request_error") == 0 &&
          fx_has_error(fo, 400, "invalid_request_error"), "no ticket: 400");
    attach(g_key_b, K1, ticket);
    snprintf(body, sizeof(body), "{\"ticket\":\"%s\"}", ticket);
    body[strlen(body) - 3] = body[strlen(body) - 3] == '0' ? '1' : '0';
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_b, body, type, sizeof(type)) ==
          403 && strcmp(type, "bad_ticket") == 0 && fx_has_error(fo, 403, "bad_ticket"),
          "a bad MAC: 403 bad_ticket");
    snprintf(body, sizeof(body), "{\"ticket\":\"%s\"}", ticket);
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_a, body, type, sizeof(type)) ==
          403 && strcmp(type, "bad_ticket") == 0, "bob's ticket from david's device: 403");
    CHECK(post_host(K2, SECRET2) == 200 &&
          fx_error(g_port, "POST", "/local/relay/" K2, g_key_b, body, type, sizeof(type)) ==
          403 && strcmp(type, "bad_ticket") == 0, "a ticket of another session: 403");
    CHECK(fx_error(g_port, "POST", "/local/relay/c-ffffffffffff", g_key_b, body, type,
                   sizeof(type)) == 404 && strcmp(type, "not_found") == 0 &&
          fx_has_error(fo, 404, "not_found"), "an unknown key: 404");
    read_code(STATE, code);
    snprintf(pair, sizeof(pair), "{\"code\":\"%s\",\"label\":\"old\"}", code);
    if (lt_http(g_port, "POST", "/local/pair", NULL, pair, &resp) == 200)
        fx_field(resp, "key", NULL, key_old, sizeof(key_old));
    free(resp);
    CHECK(key_old[0] && fx_error(g_port, "POST", "/local/relay/" K1, key_old, body, type,
                                 sizeof(type)) == 409 &&
          strcmp(type, "device_unclaimed") == 0 && fx_has_error(fo, 409, "device_unclaimed"),
          "a key with no device_id: 409");
    jr_free(fo);
}

/* Read a whole answer (head + body) until the box closes. */
static void read_all(int fd, char *out, size_t cap)
{
    struct pollfd p = { .fd = fd, .events = POLLIN };
    size_t len = 0;

    while (len + 1 < cap && poll(&p, 1, 5000) > 0) {
        ssize_t n = read(fd, out + len, cap - len - 1);
        if (n <= 0)
            break;
        len += (size_t)n;
    }
    out[len] = '\0';
    close(fd);
}

static void test_timeout_and_cap(void)
{
    jr_value *fo = fx_load("relay_open.json"), *fh = fx_load("relay_host.json");
    jr_value *fw = fx_load("hosts_wait.json");
    char ticket[PN_TICKET_MAX], id[24] = "", path[96], type[64], got[2048];
    int fds[PN_HUB_RELAYS_PER_PERSON], c = -1, ok = 1;
    uint64_t t0 = 0;

    printf("timeout_and_cap:\n");
    CHECK(fx_error(g_port, "POST", "/local/relay/r-0000000000000000/host", g_key_a, "{}", type,
                   sizeof(type)) == 404 && strcmp(type, "not_found") == 0 && fh &&
          fx_has_error(fh, 404, "not_found"), "an unknown id: 404");
    attach(g_key_b, K1, ticket);
    t0 = ms_now();
    c = open_relay(g_key_b, K1, ticket);
    CHECK(wait_for_relay(fw, id), "a relay the host never joins");
    read_all(c, got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 504") && strstr(got, "\"type\":\"relay_timeout\"") &&
          fx_has_error(fo, 504, "relay_timeout"), "no host in time: 504 relay_timeout");
    CHECK(ms_now() - t0 >= 1400, "not before the host-leg deadline");
    snprintf(path, sizeof(path), "/local/relay/%s/host", id);
    CHECK(fx_error(g_port, "POST", path, g_key_a, "{}", type, sizeof(type)) == 404,
          "the expired id: 404");

    for (int i = 0; i < PN_HUB_RELAYS_PER_PERSON; i++) {
        attach(g_key_b, K1, ticket);
        fds[i] = open_relay(g_key_b, K1, ticket);
        ok &= fds[i] >= 0 && silent_for(fds[i], 50);
    }
    CHECK(ok, "bob holds his 8 relays");
    attach(g_key_b, K1, ticket);
    read_all(open_relay(g_key_b, K1, ticket), got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 503") && strstr(got, "Retry-After: 1\r\n") &&
          strstr(got, "\"type\":\"relay_full\"") && fx_has_error(fo, 503, "relay_full"),
          "the 9th: 503 relay_full, Retry-After: 1");
    for (int i = 0; i < PN_HUB_RELAYS_PER_PERSON; i++)
        if (fds[i] >= 0)
            close(fds[i]);
    drain_wait();
    jr_free(fo);
    jr_free(fh);
    jr_free(fw);
}

static pid_t test_disconnected(pid_t pid)
{
    jr_value *fo = fx_load("relay_open.json");
    char ticket[PN_TICKET_MAX], body[PN_TICKET_MAX + 32], type[64];

    printf("disconnected:\n");
    attach(g_key_b, K1, ticket);
    lt_stop(pid);
    if (rename(TEST_DIR "/serve.log", TEST_DIR "/serve.run1.log") != 0)
        fprintf(stderr, "  warn: could not keep the first log\n");
    pid = spawn_box(TEST_DIR, g_port);
    CHECK(pid > 0, "the box restarted");
    snprintf(body, sizeof(body), "{\"ticket\":\"%s\"}", ticket);
    CHECK(fx_error(g_port, "POST", "/local/relay/" K1, g_key_b, body, type, sizeof(type)) ==
          409 && strcmp(type, "disconnected") == 0 && fx_has_error(fo, 409, "disconnected"),
          "a disconnected row: 409");
    jr_free(fo);
    return pid;
}

static int file_has(const char *path, const char *needle)
{
    static char buf[1 << 22];
    FILE *f = fopen(path, "rb");
    size_t n = f ? fread(buf, 1, sizeof(buf) - 1, f) : 0;

    if (f)
        fclose(f);
    for (size_t i = 0, k = strlen(needle); k && i + k <= n; i++)
        if (memcmp(buf + i, needle, k) == 0)
            return 1;
    return 0;
}

/* Files under dir holding needle (recursive); *files_out counts the files read. */
static int tree_hits(const char *dir, const char *needle, int *files_out)
{
    DIR *d = opendir(dir);
    struct dirent *e = NULL;
    int hits = 0;

    while (d && (e = readdir(d)) != NULL) {
        char path[1024];
        struct stat st;
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
            continue;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (lstat(path, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode)) {
            hits += tree_hits(path, needle, files_out);
        } else if (S_ISREG(st.st_mode)) {
            (*files_out)++;
            hits += file_has(path, needle);
        }
    }
    if (d)
        closedir(d);
    return hits;
}

int main(void)
{
    pid_t pid = -1;
    int clean = 1, files = 0;

    signal(SIGPIPE, SIG_IGN);
    setenv("PN_LOCAL_HOST_WAIT_MS", "5000", 1);
    setenv("PN_LOCAL_RELAY_HOST_MS", HOST_MS, 1);
    g_port = lt_pick_port(64);
    g_closed_port = closed_port();
    lt_clean_dir(TEST_DIR);
    lt_make_dir(STATE);
    seed_state(STATE, 1);
    secret_of('a', g_key_a);
    secret_of('b', g_key_b);
    pid = spawn_box(TEST_DIR, g_port);
    if (pid <= 0 || g_closed_port <= 0) {
        fprintf(stderr, "could not start the box\n");
        lt_stop(pid);
        return 1;
    }
    CHECK(post_host(K1, SECRET) == 200 &&
          lt_http(g_port, "POST", "/local/sessions/" K1 "/share", g_key_a,
                  "{\"handle\":\"bob\",\"role\":\"control\"}", NULL) == 200,
          "david registers a session and shares it with bob (control)");
    drain_wait();
    test_relay_exchange();
    test_open_errors();
    test_timeout_and_cap();
    pid = test_disconnected(pid);
    lt_stop(pid);

    printf("traces:\n");
    CHECK(file_has(TEST_DIR "/serve.run1.log", "serve: POST /local/relay/") &&
          file_has(TEST_DIR "/serve.run1.log", "local: relay r-"), "the log is being written");
    CHECK(file_has(STATE "/sessions.json", K1), "sessions.json holds the row");
    CHECK(tree_hits(TEST_DIR, MARKER, &files) == 0 && files >= 5,
          "the marker is nowhere under the test dir: state, sessions.json, logs");
    for (int i = 0; i < g_n_macs; i++)
        clean &= !file_has(TEST_DIR "/serve.run1.log", g_macs[i]) &&
                 !file_has(TEST_DIR "/serve.log", g_macs[i]);
    CHECK(g_n_macs >= 10 && clean, "no ticket in either log");
    return lt_report("test_local_relay_routes");
}
