/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_seats.c — people registry P5.3: GET /local/seats/people (spec §4.5;
 * Argus §10.8 test 3, box side), through the real route with a real people store
 * and session registry and hand-built serve entries (a batched copy on two fake
 * schedulers, a single-user copy with a prefix cache):
 *   - every person is listed with their own session rows by state (rows shared
 *     with them and desktop rows do not count; a silent row reads disconnected);
 *   - seats per model and instance from the serve-session binding: families
 *     holding a slot (batched) or bound on the copy (single-user); an unbound
 *     session counts for nobody;
 *   - the answer has seats_people.json's shape;
 *   - no title, host_addr, key, session id or secret anywhere in the body.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"
#include "local_fixture_replay.h"

#include "../src/core/prefix_cache.h"
#include "../src/local/local_caller.h"
#include "../src/local/local_routes.h"
#include "../src/local/people_store.h"
#include "../src/local/session_registry.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/model_session.h"
#include "../src/serve/replica_affinity.h"

#define DIR    "build/test_local/seats"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define TITLE  "TITLE-MARKER fix the relay"
#define ADDR   "10.9.8.7"

static pn_person g_david, g_bob, g_carol;

static int upsert(pn_reg *reg, const char *key, pn_reg_kind kind, const pn_person *owner,
                  pn_reg_state state, long long now)
{
    pn_reg_host h = {
        .key = key, .kind = kind, .owner = owner->id, .origin_device = "d00000000000000aa",
        .host_addr = ADDR ":53211", .title = TITLE, .model = "qwen-a", .instance = 1,
        .state = state, .host_secret = SECRET, .host_pub = PUB,
    };
    return pn_reg_upsert(reg, &h, now, NULL);
}

static void seed_registry(pn_reg *reg)
{
    long long now = (long long)time(NULL);

    upsert(reg, "c-00000000000a", PN_REG_KIND_SESSION, &g_david, PN_REG_STATE_THINKING, now);
    upsert(reg, "c-00000000000b", PN_REG_KIND_SESSION, &g_david, PN_REG_STATE_NEEDS_YOU, now);
    upsert(reg, "c-00000000000c", PN_REG_KIND_SESSION, &g_david, PN_REG_STATE_IDLE, now);
    upsert(reg, "c-00000000000d", PN_REG_KIND_SESSION, &g_david, PN_REG_STATE_IDLE, now);
    upsert(reg, "c-0000000000de", PN_REG_KIND_DESKTOP, &g_david, PN_REG_STATE_IDLE, now);
    upsert(reg, "c-0000000000b1", PN_REG_KIND_SESSION, &g_bob, PN_REG_STATE_IDLE, now);
    upsert(reg, "c-0000000000b2", PN_REG_KIND_SESSION, &g_bob, PN_REG_STATE_THINKING, now - 200);
    pn_reg_sweep(reg, now);   /* b2 has been silent 200 s: disconnected */
    pn_reg_share(reg, "c-00000000000a", g_david.id, g_bob.id, PN_REG_ROLE_VIEW);
}

/* A zeroed pn_model: mr_snapshot reads only config fields and gauges off it. */
static pn_model g_fake_model;

static void seed_row(mr_registry *reg, int slot, const char *name, int replica, int die)
{
    mr_entry *e = &reg->entries[slot];

    memset(e, 0, sizeof(*e));
    e->used = true;
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->model = &g_fake_model;
    e->serial = 700 + (uint64_t)slot;
    e->replica = replica;
    e->max_sessions = 2;
    e->max_concurrent = 2;
    if (die >= 0) {
        e->dies[0] = die;
        e->n_dies = 1;
    }
    pthread_mutex_init(&e->gen_lock, NULL);
    reg->n++;
}

static pn_batch_sched *fake_sched(int cap)
{
    pn_batch_sched *s = calloc(1, sizeof(*s));

    s->cap = cap;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv_idle, NULL);
    for (int i = 0; i < cap; i++)
        s->slots[i].status = BS_SLOT_IDLE;
    return s;
}

static void hold(mr_entry *e, int slot, const char *sid)
{
    snprintf(e->sched->slots[slot].sid, sizeof(e->sched->slots[slot].sid), "%s", sid);
}

static void bind_to(mr_entry *e, const char *sid, const pn_person *who)
{
    ms_session_principal_check(e, sid, who->id, who->admin, MS_PRINCIPAL_BIND);
}

static char *get_seats(pn_local_ctx *ctx)
{
    hs_request req = {0};
    pn_local_caller c;
    int sv[2];
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);

    memset(&c, 0, sizeof(c));
    snprintf(c.person, sizeof(c.person), "%s", g_david.id);
    c.kind = PN_KEY_KIND_DEVICE;
    c.admin = c.may_load = 1;
    pn_local_caller_set(&c);
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    req.method = "GET";
    req.path = "/local/seats/people";
    pn_local_routes_dispatch(sv[0], ctx, &req);
    close(sv[0]);
    for (ssize_t r; buf && len + 1 < cap && (r = read(sv[1], buf + len, cap - len - 1)) > 0;)
        len += (size_t)r;
    if (buf)
        buf[len] = '\0';
    close(sv[1]);
    pn_local_caller_clear();
    return buf;
}

static const jr_value *person_row(const jr_value *root, const char *id)
{
    const jr_value *people = jr_object_get(root, "people");

    for (int i = 0; i < jr_array_len(people); i++)
        if (strcmp(jr_as_string(jr_object_get(jr_array_get(people, i), "id")), id) == 0)
            return jr_array_get(people, i);
    return NULL;
}

static int count(const jr_value *person, const char *state)
{
    return (int)jr_as_number(jr_object_get(jr_object_get(person, "sessions"), state), -1);
}

/* held for (model, instance) in the person's seats, -1 when absent. */
static int held(const jr_value *person, const char *model, int instance)
{
    const jr_value *seats = jr_object_get(person, "seats");

    for (int i = 0; i < jr_array_len(seats); i++) {
        const jr_value *s = jr_array_get(seats, i);
        if (strcmp(jr_as_string(jr_object_get(s, "model")), model) == 0 &&
            (int)jr_as_number(jr_object_get(s, "instance"), 0) == instance)
            return (int)jr_as_number(jr_object_get(s, "held"), -1);
    }
    return -1;
}

static void check_body(const char *resp, int with_models)
{
    const char *json = resp ? strstr(resp, "\r\n\r\n") : NULL;
    jr_value *root = json ? jr_parse(json + 4) : NULL, *fx = fx_load("seats_people.json");
    const jr_value *d = person_row(root, g_david.id), *b = person_row(root, g_bob.id);
    const jr_value *c = person_row(root, g_carol.id);
    char where[300] = "";

    CHECK(resp && strstr(resp, "HTTP/1.1 200"), "200");
    CHECK(fx && fx_same_shape(jr_object_get(fx, "response"), root, where, sizeof(where),
                              "response"), "the fixture's shape");
    if (where[0])
        fprintf(stderr, "  shape: %s\n", where);
    CHECK(jr_array_len(jr_object_get(root, "people")) == 3 && d && b && c, "every person listed");
    CHECK(count(d, "registered") == 4 && count(d, "thinking") == 1 && count(d, "needs_you") == 1 &&
          count(d, "idle") == 2 && count(d, "disconnected") == 0,
          "david: 4 sessions by state (the desktop row does not count)");
    CHECK(count(b, "registered") == 2 && count(b, "idle") == 1 && count(b, "disconnected") == 1 &&
          count(b, "thinking") == 0, "bob: his own 2 (a row shared with him does not count)");
    CHECK(count(c, "registered") == 0, "carol: none");
    if (with_models) {
        CHECK(held(d, "qwen-a", 1) == 2 && jr_array_len(jr_object_get(d, "seats")) == 1,
              "david: 2 families on instance 1 (one with two ids)");
        CHECK(held(b, "qwen-a", 2) == 1 && jr_array_len(jr_object_get(b, "seats")) == 1,
              "bob: 1 on instance 2 (the unbound slot there counts for nobody)");
        CHECK(held(c, "solo", 1) == 1 && jr_array_len(jr_object_get(c, "seats")) == 1,
              "carol: 1 on the single-user copy");
    } else {
        CHECK(jr_array_len(jr_object_get(d, "seats")) == 0, "no models: no seats");
    }
    CHECK(resp && !strstr(resp, "TITLE-MARKER") && !strstr(resp, "\"title\"") &&
          !strstr(resp, ADDR) && !strstr(resp, "host_addr") && !strstr(resp, "\"key\"") &&
          !strstr(resp, "c-0000") && !strstr(resp, SECRET) && !strstr(resp, PUB) &&
          !strstr(resp, "s-d1") && !strstr(resp, "s-legacy"),
          "no title, address, key, session id or secret in the body (Argus 10.8 test 3)");
    jr_free(fx);
    jr_free(root);
}

static void test_seats(pn_local_ctx *ctx, mr_registry *reg)
{
    mr_entry *a1 = &reg->entries[0], *a2 = &reg->entries[1], *one = &reg->entries[2];
    char *resp = NULL;

    printf("no_models:\n");
    resp = get_seats(ctx);
    check_body(resp, 0);
    free(resp);

    printf("seats:\n");
    pn_replica_affinity_reset();
    seed_row(reg, 0, "qwen-a", 1, 4);
    seed_row(reg, 1, "qwen-a", 2, 5);
    seed_row(reg, 2, "solo", 1, -1);
    a1->sched = fake_sched(3);
    a2->sched = fake_sched(2);
    one->prefix_cache = pn_prefix_cache_new(&g_fake_model, 2);
    bind_to(a1, "s-d1", &g_david);
    bind_to(a1, "s-d1-orch", &g_david);
    bind_to(a1, "s-d2", &g_david);
    bind_to(a1, "s-d3", &g_david);   /* bound, but its slot was reaped: no seat */
    bind_to(a2, "s-b1", &g_bob);
    bind_to(one, "s-c1", &g_carol);
    hold(a1, 0, "s-d1-orch");
    hold(a1, 1, "s-d2");
    hold(a2, 0, "s-b1");
    hold(a2, 1, "s-legacy");      /* opened before people: unbound */
    resp = get_seats(ctx);
    check_body(resp, 1);
    free(resp);

    ms_session_release_by_prefix(a1, "s-");
    ms_session_release_by_prefix(one, "s-");
    pn_replica_affinity_reset();
    pn_prefix_cache_free(one->prefix_cache);
    for (int i = 0; i < 2; i++) {
        pthread_cond_destroy(&reg->entries[i].sched->cv_idle);
        pthread_mutex_destroy(&reg->entries[i].sched->mu);
        free(reg->entries[i].sched);
    }
    memset(reg->entries, 0, sizeof(reg->entries));   /* fake models: never pn_model_free */
    reg->n = 0;
}

int main(void)
{
    mr_registry reg;
    pn_local_ctx ctx;

    printf("test_local_seats\n");
    signal(SIGPIPE, SIG_IGN);
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    memset(&ctx, 0, sizeof(ctx));
    ctx.people = pn_people_open(DIR "/people.json");
    pn_people_add(ctx.people, "david", "David", 1, NULL, 1, &g_david);
    pn_people_add(ctx.people, "bob", "Bob", 0, NULL, 1, &g_bob);
    pn_people_add(ctx.people, "carol", "Carol", 0, NULL, 1, &g_carol);
    ctx.registry = pn_reg_open(DIR "/sessions.json");
    CHECK(ctx.people && ctx.registry && mr_init(&reg) == 0, "people, registry, serve registry");
    ctx.reg = &reg;
    seed_registry(ctx.registry);
    test_seats(&ctx, &reg);
    mr_free(&reg);
    pn_reg_close(ctx.registry);
    pn_people_close(ctx.people);
    return lt_report("test_local_seats");
}
