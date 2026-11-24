/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_replica_pin_serve.c — a client picks its copy of a model, through the
 * real chat handler (oai_handle_chat over a socket pair) on two CPU copies of
 * the test model, each with a session cache attached so a copy can be full:
 *   - copies are numbered as asked (mr_load_replica), else the lowest free;
 *   - "replica": N binds a new session to copy N (X-Pantheon-Replica: N on the
 *     non-streamed reply); the header alone pins a streamed reply;
 *   - a full copy is a 429 naming model, copy and reason, from that copy, and
 *     never a re-route; the refused session may then pin another copy;
 *   - a bound session keeps its copy and the header reports it;
 *   - a copy that is not loaded is a 404 naming model and copy;
 *   - no field = least-busy routing (binding counts) among copies with a free
 *     session seat; every copy full = the least busy refuses, unpinned
 *     wording, and the refused session leaves no routing row;
 *   - one copy left: the single-model rule still holds, a missing copy is 404.
 * Uses models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL); skipped when absent.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/core/prefix_cache.h"
#include "../src/serve/model_registry.h"
#include "../src/serve/openai_chat.h"
#include "../src/serve/replica_affinity.h"

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static const char *g_model;

/* One chat through oai_handle_chat; the whole response comes back (heap). */
static char *chat(mr_registry *reg, const char *sid, int replica, const char *header, int stream)
{
    char body[512], pin[32] = "";
    int sv[2];
    size_t cap = 1 << 16, len = 0;

    if (replica > 0)
        snprintf(pin, sizeof(pin), "\"replica\":%d,", replica);
    snprintf(body, sizeof(body), "{\"model\":\"tiny\",%s\"stream\":%s,\"max_tokens\":2,"
             "\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}", pin,
             stream ? "true" : "false");
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return NULL;
    oai_handle_chat(sv[1], reg, body, sid, header);
    close(sv[1]);
    char *out = malloc(cap);
    for (ssize_t r; out && len + 1 < cap && (r = read(sv[0], out + len, cap - len - 1)) > 0;)
        len += (size_t)r;
    close(sv[0]);
    if (out) out[len] = '\0';
    return out;
}

static int status_of(const char *resp)
{
    int st = 0;
    return resp && sscanf(resp, "HTTP/1.1 %d", &st) == 1 ? st : 0;
}

static int served_of(const char *resp)
{
    const char *h = resp ? strstr(resp, "X-Pantheon-Replica: ") : NULL;
    const char *end = resp ? strstr(resp, "\r\n\r\n") : NULL;
    return h && end && h < end ? atoi(h + 20) : -1;
}

/* Status + serving copy of one chat; *text (may be NULL) keeps the response. */
static int run(mr_registry *reg, const char *sid, int replica, const char *header, int stream,
               int *served, char **text)
{
    char *resp = chat(reg, sid, replica, header, stream);
    int st = status_of(resp);

    *served = served_of(resp);
    if (stream && st == 200 && (!resp || !strstr(resp, "text/event-stream")))
        st = -1;
    if (text) *text = resp; else free(resp);
    return st;
}

static mr_entry *entry_numbered(mr_registry *reg, int replica)
{
    for (int i = 0; i < MR_MAX_MODELS; i++)
        if (reg->entries[i].used && reg->entries[i].replica == replica)
            return &reg->entries[i];
    return NULL;
}

/* Two CPU copies: replica 2 (1 session) loaded first, then the auto-numbered one. */
static int load_copies(mr_registry *reg)
{
    pn_load_opts opts = { .ctx_cap = 512 };
    char err[256] = "";

    if (mr_load_replica(reg, g_model, "tiny", &opts, NULL, 0, 0, 1, 1, 2, err,
                        sizeof(err)) != 0 ||
        mr_load(reg, g_model, "tiny", &opts, NULL, 0, 0, 2, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "  load error: %s\n", err);
        return -1;
    }
    CHECK(mr_load_replica(reg, g_model, "tiny", &opts, NULL, 0, 0, 1, 1, 2, err,
                          sizeof(err)) != 0 && strstr(err, "replica 2 of 'tiny' is already"),
          "a taken copy number is refused");
    mr_entry *two = entry_numbered(reg, 2), *one = entry_numbered(reg, 1);
    CHECK(two && one && two->serial < one->serial,
          "numbering: the first load is copy 2 as asked, the next takes the free 1");
    if (!two || !one)
        return -1;
    /* The registry builds a session cache only for die-placed models; attach
     * one so a copy has session slots to run out of. */
    two->prefix_cache = pn_prefix_cache_new(two->model, 1);
    one->prefix_cache = pn_prefix_cache_new(one->model, 3);
    return two->prefix_cache && one->prefix_cache ? 0 : -1;
}

static void test_pins(mr_registry *reg)
{
    char *text = NULL;
    int served = 0;

    printf("pins:\n");
    CHECK(run(reg, "a", 2, NULL, 0, &served, NULL) == 200 && served == 2,
          "body replica 2: non-streamed reply from copy 2, header says 2");
    int st = run(reg, "b", 0, "2", 1, &served, &text);
    CHECK(st == 429 && served == 2 && text &&
              strstr(text, "replica 2 of model 'tiny' has no free session slot") &&
              strstr(text, "Retry-After: 30"),
          "copy 2 is full: 429 naming model, copy and reason, from copy 2 (no re-route)");
    free(text);
    CHECK(run(reg, "b", 0, "1", 1, &served, &text) == 200 && served == 1 && text &&
              strstr(text, "data: [DONE]"),
          "the refused session pins copy 1: streamed reply, SSE head says 1");
    free(text);
    CHECK(run(reg, "a", 1, NULL, 0, &served, NULL) == 200 && served == 2,
          "a bound session keeps its copy and the header reports it");
    CHECK(run(reg, "c", 1, "2", 0, &served, NULL) == 200 && served == 1,
          "the body wins over the header");
    st = run(reg, "x", 3, NULL, 0, &served, &text);
    CHECK(st == 404 && text && strstr(text, "replica_not_found") &&
              strstr(text, "model 'tiny' has no replica 3 (loaded replicas: 2, 1)"),
          "a copy that is not loaded: 404 naming model, copy and the loaded copies");
    free(text);
    CHECK(run(reg, "y", 0, "two", 0, &served, NULL) == 400, "a malformed header: 400");
}

static void test_no_field(mr_registry *reg)
{
    char *text = NULL;
    int served = 0;

    printf("no_field:\n");
    /* Bound now: copy 2 holds a (its 1 seat), copy 1 holds b and c (2 of 3):
     * copy 2 is less busy but full, so a new session goes to copy 1. */
    CHECK(run(reg, "d", 0, NULL, 0, &served, NULL) == 200 && served == 1,
          "no field: a full copy is passed over for one with a free seat");
    /* Both full now (copy 1 holds b, c, d): the least busy copy refuses. */
    int st = run(reg, "f", 0, NULL, 0, &served, &text);
    CHECK(st == 429 && served == 2 && text &&
              strstr(text, "model at session capacity (its one session is in use)"),
          "no field, every copy full: least busy (copy 2) refuses, unpinned wording");
    free(text);
    CHECK(pn_replica_affinity_count(entry_numbered(reg, 2)->serial) == 1,
          "the refused unpinned session left no routing row on copy 2");
    CHECK(run(reg, NULL, 0, NULL, 0, &served, NULL) == 200 && served == 2,
          "no field, no session: least busy copy, announced");
}

static void test_one_copy_left(mr_registry *reg)
{
    char err[256] = "";
    int served = 0;

    printf("one_copy_left:\n");
    mr_entry *two = entry_numbered(reg, 2);
    CHECK(two && mr_unload_replica(reg, "tiny", -1, err, sizeof(err)) == 0 &&
              !entry_numbered(reg, 2), "copy 2 unloaded (the first entry of the name)");
    char *text = NULL;
    CHECK(run(reg, "e", 2, NULL, 0, &served, &text) == 404 && text &&
              strstr(text, "model 'tiny' has no replica 2 (loaded replicas: 1)"),
          "a missing copy is still a 404, naming the model the single-model rule picked");
    free(text);
    /* Copy 1's three seats hold b, c and d, so name it with one of them. */
    CHECK(run(reg, "b", 1, NULL, 0, &served, NULL) == 200 && served == 1,
          "the remaining copy serves when named");
    CHECK(run(reg, NULL, 0, NULL, 0, &served, NULL) == 200 && served == 1,
          "single model, no field: it serves, announced");
}

int main(void)
{
    mr_registry reg;

    printf("test_replica_pin_serve\n");
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP: %s not found\n", g_model);
        return 0;
    }
    if (mr_init(&reg) != 0)
        return 1;
    pn_replica_affinity_reset();
    if (load_copies(&reg) == 0) {
        test_pins(&reg);
        test_no_field(&reg);
        test_one_copy_left(&reg);
    } else {
        CHECK(0, "two copies with session caches");
    }
    mr_free(&reg);
    printf("test_replica_pin_serve: %d checks, %d failures\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
