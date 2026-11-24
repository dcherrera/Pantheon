/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_replica_pin.c — a client picks its copy of a model, end to end
 * through Pantheon Local's front door (its port, its keys, its /v1 routes).
 *
 * Two CPU copies of the test model under one name. The box remembers replica 2
 * (4096 tokens x 1 user) and replica 1 (2048 x 3) in that order; its boot
 * reload brings back replica 2 only (one cpu unit) and numbers it 2 as
 * remembered, not 1 as the first load. A runtime /admin/models/load adds the
 * second copy, which takes the free number 1. Then, with a paired key:
 *   - "replica": 2 binds a new session to copy 2 (X-Pantheon-Replica: 2 on the
 *     non-streamed reply); the header alone pins a streamed one (SSE head);
 *   - the body wins over the header;
 *   - a bound session keeps its copy when a later request names another;
 *   - a copy that is not loaded: 404 replica_not_found naming model and copy;
 *   - no field: today's least-busy routing, still announced;
 *   - /admin/models numbers the rows as the loaded list does.
 * A full copy needs a session cache, which a CPU copy behind the box does not
 * get: test_replica_pin_serve covers it in process.
 * Uses models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL); skipped when absent.
 */

#include "local_test_support.h"

#include "../src/local/loaded_set.h"

#define DIR "build/test_local/replica_pin"
#define NAME "tiny"

static int g_port;
static char g_port_s[16];
static const char *g_model;

/* One request to the box; the response head and body come back separately
 * (heap). Returns the status, 0 on a transport failure. */
static int pin_http(const char *key, const char *extra_headers, const char *body,
                    char **head_out, char **body_out)
{
    *head_out = *body_out = NULL;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)g_port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        if (fd >= 0) close(fd);
        return 0;
    }
    char req[2048];
    int n = snprintf(req, sizeof(req), "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                     "Authorization: Bearer %s\r\n%sContent-Type: application/json\r\n"
                     "Content-Length: %zu\r\n\r\n%s", key, extra_headers ? extra_headers : "",
                     strlen(body), body);
    if (n <= 0 || (size_t)n >= sizeof(req) || write(fd, req, (size_t)n) != n) {
        close(fd);
        return 0;
    }
    size_t cap = 65536, len = 0;
    char *resp = malloc(cap);
    for (;;) {
        if (len + 1 >= cap) { cap *= 2; resp = realloc(resp, cap); }
        ssize_t r = read(fd, resp + len, cap - len - 1);
        if (r <= 0) break;
        len += (size_t)r;
    }
    close(fd);
    resp[len] = '\0';
    int status = 0;
    if (sscanf(resp, "HTTP/1.1 %d", &status) != 1) status = 0;
    char *split = strstr(resp, "\r\n\r\n");
    *body_out = strdup(split ? split + 4 : "");
    if (split) split[2] = '\0';
    *head_out = strdup(resp);
    free(resp);
    return status;
}

/* The X-Pantheon-Replica value in a response head, or -1. */
static int served_replica(const char *head)
{
    const char *h = head ? strstr(head, "X-Pantheon-Replica: ") : NULL;
    return h ? atoi(h + 20) : -1;
}

/* A chat body: session sid, "replica" when replica > 0, streamed or not. */
static void chat_body(char *out, size_t cap, const char *sid, int replica, int stream)
{
    char pin[32] = "";
    char session[96] = "";

    if (replica > 0)
        snprintf(pin, sizeof(pin), "\"replica\":%d,", replica);
    if (sid)
        snprintf(session, sizeof(session), "\"session_id\":\"%s\",", sid);
    snprintf(out, cap, "{\"model\":\"" NAME "\",%s%s\"stream\":%s,\"max_tokens\":2,"
             "\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}", session, pin,
             stream ? "true" : "false");
}

/* One chat; returns the status and sets *replica to the announced copy. */
static int chat(const char *key, const char *sid, int replica, const char *header, int stream,
                int *served, char **body_out)
{
    char body[512], *head = NULL, *resp = NULL;

    chat_body(body, sizeof(body), sid, replica, stream);
    int st = pin_http(key, header, body, &head, &resp);
    *served = served_replica(head);
    if (stream && head && !strstr(head, "text/event-stream"))
        st = -st;   /* a streamed request must get an SSE head */
    free(head);
    if (body_out) *body_out = resp; else free(resp);
    return st;
}

/* Remember the two copies, replica 2 first. */
static int remember_copies(void)
{
    pn_loaded_set *set = pn_loaded_set_open(DIR "/state/loaded.json");
    int rc = set ? 0 : -1;
    const int replicas[2] = { 2, 1 }, ctx[2] = { 4096, 2048 }, users[2] = { 1, 3 };

    for (int i = 0; set && i < 2; i++) {
        pn_loaded_entry e;
        memset(&e, 0, sizeof(e));
        snprintf(e.catalog_id, sizeof(e.catalog_id), NAME);
        snprintf(e.profile_id, sizeof(e.profile_id), "cpu");
        snprintf(e.path, sizeof(e.path), "%s", g_model);
        snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), "cpu");
        e.hardware.count = 1;
        e.replica = replicas[i];
        e.ctx_cap = ctx[i];
        e.max_sessions = users[i];
        rc |= pn_loaded_set_add(set, &e);
    }
    pn_loaded_set_close(set);
    return rc;
}

/* Poll /admin/models until its body has needle. */
static int wait_models(const char *key, const char *needle)
{
    for (int i = 0; i < 360; i++) {
        char *body = NULL;
        lt_http(g_port, "GET", "/admin/models", key, NULL, &body);
        int hit = body && strstr(body, needle);
        free(body);
        if (hit) return 1;
        usleep(250000);
    }
    return 0;
}

/* The boot reload brings back replica 2; load the second copy at runtime. */
static int load_both(const char *key)
{
    char body[1400], *resp = NULL;

    if (!wait_models(key, "\"replica\":2"))
        return 0;
    snprintf(body, sizeof(body), "{\"path\":\"%s\",\"name\":\"" NAME "\",\"cpu\":true,"
             "\"ctx_cap\":2048,\"max_sessions\":3}", g_model);
    int st = lt_http(g_port, "POST", "/admin/models/load", key, body, &resp);
    CHECK(st == 200, resp ? resp : "runtime load of the second copy");
    free(resp);
    return st == 200 && wait_models(key, "\"replica\":1");
}

static void test_numbering(const char *key)
{
    char *body = NULL;

    printf("numbering:\n");
    lt_http(g_port, "GET", "/admin/models", key, NULL, &body);
    const char *r2 = body ? strstr(body, "\"replica\":2") : NULL;
    const char *next = r2 ? strstr(r2 + 1, "\"replica\":") : NULL;
    const char *w = r2 ? strstr(r2, "\"context_window_configured\":4096") : NULL;
    CHECK(w && (!next || w < next),
          "replica 2 is the 4096-token boot copy (numbered as remembered, not 1)");
    free(body);
}

static void test_pin_and_header(const char *key)
{
    char *body = NULL;
    int served = 0;

    printf("pin_and_header:\n");
    CHECK(chat(key, "pin-a", 2, NULL, 0, &served, NULL) == 200 && served == 2,
          "body replica 2: non-streamed reply from copy 2, header says 2");
    CHECK(chat(key, "pin-b", 0, "X-Pantheon-Replica: 1\r\n", 1, &served, &body) == 200 &&
              served == 1 && body && strstr(body, "data: [DONE]"),
          "header replica 1: streamed reply (SSE head) from copy 1, header says 1");
    free(body);
    CHECK(chat(key, "pin-e", 1, "X-Pantheon-Replica: 2\r\n", 0, &served, NULL) == 200 &&
              served == 1, "body wins over the header");
    CHECK(chat(key, "pin-a", 1, NULL, 0, &served, NULL) == 200 && served == 2,
          "a bound session keeps its copy and the header reports it");
}

static void test_refusals(const char *key)
{
    char *body = NULL;
    int served = 0;

    printf("refusals:\n");
    int st = chat(key, "pin-x", 3, NULL, 0, &served, &body);
    CHECK(st == 404 && body && strstr(body, "replica_not_found") &&
              strstr(body, "model 'tiny' has no replica 3"),
          "unknown replica: 404 naming the model and the copy");
    free(body);
    CHECK(chat(key, "pin-y", 0, "X-Pantheon-Replica: zero\r\n", 0, &served, NULL) == 400,
          "a malformed header: 400");
}

static void test_no_field(const char *key)
{
    int served = 0;

    printf("no_field:\n");
    /* Bound now: copy 2 holds pin-a (1), copy 1 holds pin-b and pin-e (2). */
    CHECK(chat(key, NULL, 0, NULL, 0, &served, NULL) == 200 && served == 2,
          "no field: the least busy copy (2), announced");
}

int main(void)
{
    char admin[128];

    printf("test_local_replica_pin\n");
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP: %s not found\n", g_model);
        return 0;
    }
    lt_clean_dir(DIR);
    lt_make_dir(DIR "/state");
    lt_make_dir(DIR "/store");
    CHECK(remember_copies() == 0, "two copies remembered");
    g_port = lt_pick_port(14);
    snprintf(g_port_s, sizeof(g_port_s), "%d", g_port);
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", g_port_s, "--state-dir",
                     DIR "/state", "--store", DIR "/store", "--no-mdns", "--catalog-url", "",
                     NULL };
    pid_t pid = lt_spawn_local(12, argv, g_port, DIR "/serve.log");
    CHECK(pid > 0, "box up");
    const char *k = lt_admin_key(g_port, DIR "/state");
    snprintf(admin, sizeof(admin), "%s", k ? k : "");
    CHECK(k != NULL, "paired");
    if (pid > 0 && k && load_both(admin)) {
        test_numbering(admin);
        test_pin_and_header(admin);
        test_refusals(admin);
        test_no_field(admin);
    } else {
        CHECK(0, "both copies loaded (see " DIR "/serve.log)");
    }
    lt_stop(pid);
    return lt_report("test_local_replica_pin");
}
