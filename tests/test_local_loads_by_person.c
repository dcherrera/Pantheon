/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_loads_by_person.c — people registry P5.1: loads by person (spec §4.7).
 *   - loaded.json keeps `loaded_by`; entries from before people read "" and are
 *     given to person 1 (the oldest admin) at open, once, and saved;
 *   - a `loaded_by` that is not a short string makes the file malformed;
 *   - LOADER covers install, load and unload: `may_load` off -> 403 may_load_off;
 *   - unload by a non-admin (through the real route, hand-built registry rows): someone
 *     else's replica -> 403 unload_refused "was loaded by <handle>"; their own with a
 *     session -> 403 "has 1 session on it"; {"name"} needs that for every replica and
 *     every row under the name (a hand-loaded one is refused); nothing is unloaded on
 *     a refusal; their own idle replica unloads; an admin unloads anything;
 *   - GET /local/models/loaded rows carry `loaded_by` {id,handle,name} (a removed
 *     person keeps its id);
 *   - the boot reload keeps `loaded_by` (real load of models/qwen3-0.6b-Q8_0.gguf,
 *     PN_TEST_MODEL; skipped when absent).
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include "../src/local/key_store.h"
#include "../src/local/local_auth.h"
#include "../src/local/local_caller.h"
#include "../src/local/local_load.h"
#include "../src/local/local_model_routes.h"
#include "../src/local/people_store.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/replica_affinity.h"

#define DIR "build/test_local/loads_by_person"

static pn_person g_david, g_bob, g_alice;

static void write_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");

    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static pn_loaded_entry entry(const char *id, int replica, int unit, const char *by)
{
    pn_loaded_entry e;

    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "%s", id);
    snprintf(e.profile_id, sizeof(e.profile_id), unit >= 0 ? "k80" : "cpu");
    snprintf(e.path, sizeof(e.path), "%s.gguf", id);
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), unit >= 0 ? "k80" : "cpu");
    snprintf(e.loaded_by, sizeof(e.loaded_by), "%s", by);
    e.hardware.count = 1;
    e.replica = replica;
    e.ctx_cap = 2048;
    e.max_sessions = 2;
    if (unit >= 0) {
        e.units[0] = unit;
        e.n_units = 1;
    }
    return e;
}

static const char *k_old_loaded =
    "{\"version\":1,\"models\":["
    "{\"catalog_id\":\"a\",\"replica\":1,\"profile_id\":\"cpu\",\"path\":\"a.gguf\",\"units\":[],"
    "\"hardware\":{\"unit_type\":\"cpu\",\"count\":1,\"same_card\":false,\"per_card\":0,"
    "\"min_mem_per_unit\":0},\"ctx_cap\":2048,\"max_sessions\":1,\"max_concurrent\":0,"
    "\"reasoning_budget\":0,\"mtp\":false,\"kv_type\":\"\",\"sampling\":{}},"
    "{\"catalog_id\":\"b\",\"replica\":1,\"profile_id\":\"cpu\",\"path\":\"b.gguf\",\"units\":[],"
    "\"hardware\":{\"unit_type\":\"cpu\",\"count\":1,\"same_card\":false,\"per_card\":0,"
    "\"min_mem_per_unit\":0},\"ctx_cap\":2048,\"max_sessions\":1,\"max_concurrent\":0,"
    "\"reasoning_budget\":0,\"mtp\":false,\"kv_type\":\"\",\"sampling\":{},"
    "\"loaded_by\":\"p00000002\"}]}\n";

static void test_loaded_set_file(pn_people *people)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    pn_local_ctx ctx;
    char text[4096];
    pn_loaded_set *s = NULL;

    printf("loaded_set_file:\n");
    write_text(DIR "/old.json", k_old_loaded);
    s = pn_loaded_set_open(DIR "/old.json");
    CHECK(s && pn_loaded_set_list(s, list, PN_LOADED_MAX) == 2 && list[0].loaded_by[0] == '\0' &&
          strcmp(list[1].loaded_by, "p00000002") == 0,
          "an old entry reads no loaded_by; a new one keeps it");
    memset(&ctx, 0, sizeof(ctx));
    ctx.loaded = s;
    ctx.people = people;
    CHECK(pn_local_loads_claim_unowned(&ctx) == 0, "claim at open");
    pn_loaded_set_list(s, list, PN_LOADED_MAX);
    CHECK(strcmp(list[0].loaded_by, g_david.id) == 0 && strcmp(list[1].loaded_by, "p00000002") == 0,
          "the old entry goes to person 1 (the oldest admin); the other is kept");
    pn_loaded_set_close(s);
    s = pn_loaded_set_open(DIR "/old.json");
    pn_loaded_set_list(s, list, PN_LOADED_MAX);
    CHECK(s && strcmp(list[0].loaded_by, g_david.id) == 0, "saved: a reopen keeps it");
    CHECK(pn_loaded_set_claim_unowned(s, g_bob.id) == 0, "nothing left to claim");
    pn_loaded_set_close(s);

    snprintf(text, sizeof(text), "%s", k_old_loaded);
    char *at = strstr(text, "\"p00000002\"");
    memcpy(at, "12345678", 8);   /* a number where the person id was */
    memset(at + 8, ' ', 3);
    write_text(DIR "/bad.json", text);
    CHECK(pn_loaded_set_open(DIR "/bad.json") == NULL, "a loaded_by that is not a string: malformed");
    write_text(DIR "/bad.json", "{\"version\":1,\"models\":[{\"catalog_id\":\"a\",\"profile_id\":"
               "\"cpu\",\"path\":\"a\",\"units\":[],\"hardware\":{\"unit_type\":\"cpu\"},"
               "\"loaded_by\":\"p0123456789abcdef0\"}]}");
    CHECK(pn_loaded_set_open(DIR "/bad.json") == NULL, "a loaded_by too long: malformed");
}

/* The status and error type of `fn` answering on a socketpair. */
static int answer(char *buf, size_t cap, int sv1)
{
    int status = 0;
    ssize_t n = read(sv1, buf, cap - 1);

    buf[n > 0 ? n : 0] = '\0';
    if (sscanf(buf, "HTTP/1.1 %d", &status) != 1)
        status = -1;
    return status;
}

static int auth_status(pn_local_ctx *ctx, const char *path, const char *secret, char *type)
{
    int sv[2], status = 0;
    char auth[128], buf[2048];
    hs_request req = {0};

    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    snprintf(auth, sizeof(auth), "Bearer %s", secret);
    req.method = "POST";
    req.path = (char *)path;
    req.authorization = auth;
    type[0] = '\0';
    if (pn_local_auth_check(sv[0], &req, ctx) != 0) {
        close(sv[0]);
        sv[0] = -1;
        status = answer(buf, sizeof(buf), sv[1]);
        const char *t = strstr(buf, "\"type\":\"");
        if (t)
            sscanf(t + 8, "%63[^\"]", type);
    }
    if (sv[0] >= 0)
        close(sv[0]);
    close(sv[1]);
    return status;
}

static void test_loader_class(pn_people *people)
{
    pn_key_store *keys = pn_key_store_open(DIR "/keys.json");
    pn_person_change off = { NULL, NULL, 0, -1 };
    const char *paths[] = { "/local/models/install", "/local/models/load", "/admin/models/unload" };
    char secret[PN_KEY_SECRET_MAX], type[64];
    pn_local_ctx ctx;
    pn_person carol;
    int wrong = 0;

    printf("loader_class:\n");
    pn_people_add(people, "carol", "Carol", 0, NULL, 1, &carol);
    pn_people_update(people, g_david.id, carol.id, &off, &carol);
    pn_key_store_mint_device(keys, carol.id, "d00000000000000cc", "carol-mac", "argus", secret,
                             NULL);
    memset(&ctx, 0, sizeof(ctx));
    ctx.people = people;
    ctx.keys = keys;
    for (int i = 0; i < 3; i++)
        wrong += auth_status(&ctx, paths[i], secret, type) != 403 ||
                 strcmp(type, "may_load_off") != 0;
    CHECK(wrong == 0, "may_load off: install, load and unload -> 403 may_load_off");
    pn_people_remove(people, g_david.id, carol.id);
    pn_key_store_close(keys);
    pn_local_caller_clear();
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
    e->serial = 900 + (uint64_t)slot;
    e->replica = replica;
    e->max_sessions = 2;
    e->max_concurrent = 2;
    if (die >= 0) {
        e->dies[0] = die;
        e->n_dies = 1;
    }
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

/* POST /admin/models/unload as `who` (admin or not); the body goes to resp. */
static int unload_as(pn_local_ctx *ctx, const pn_person *who, const char *body, char *resp,
                     size_t cap)
{
    pn_local_caller c;
    hs_request req = {0};
    int sv[2];

    memset(&c, 0, sizeof(c));
    c.kind = PN_KEY_KIND_DEVICE;
    snprintf(c.person, sizeof(c.person), "%s", who->id);
    c.admin = who->admin;
    c.may_load = 1;
    pn_local_caller_set(&c);
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    req.method = "POST";
    req.path = "/admin/models/unload";
    req.body = (char *)body;
    pn_local_model_routes(sv[0], ctx, &req);
    close(sv[0]);
    int st = answer(resp, cap, sv[1]);
    close(sv[1]);
    pn_local_caller_clear();
    return st;
}

static int remembered(pn_local_ctx *ctx, const char *id, int replica)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    int n = pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX);

    for (int i = 0; i < n; i++)
        if (strcmp(list[i].catalog_id, id) == 0 && list[i].replica == replica)
            return 1;
    return 0;
}

static void test_loaded_rows(pn_local_ctx *ctx)
{
    json_writer jw;

    jw_init(&jw);
    pn_local_write_loaded(&jw, ctx);
    jr_value *root = jr_parse(jw.data);
    const jr_value *m = jr_object_get(root, "models");
    const jr_value *by0 = jr_object_get(jr_array_get(m, 0), "loaded_by");
    int all = jr_array_len(m) > 0;
    for (int i = 0; i < jr_array_len(m); i++) {
        const jr_value *by = jr_object_get(jr_array_get(m, i), "loaded_by");
        all &= jr_kind(by) == JR_OBJECT && jr_object_member_count(by) == 3 &&
               jr_as_string(jr_object_get(by, "id")) && jr_as_string(jr_object_get(by, "handle")) &&
               jr_as_string(jr_object_get(by, "name"));
    }
    CHECK(all, "loaded rows: loaded_by is {id,handle,name} on every row");
    CHECK(by0 && strcmp(jr_as_string(jr_object_get(by0, "id")), g_bob.id) == 0 &&
          strcmp(jr_as_string(jr_object_get(by0, "handle")), "bob") == 0 &&
          strcmp(jr_as_string(jr_object_get(by0, "name")), "Bob") == 0,
          "loaded rows: replica 1 of tiny shows bob");
    CHECK(strstr(jw.data, "\"loaded_by\":{\"id\":\"p0000dead\",\"handle\":\"\",\"name\":\"\"}") != NULL,
          "loaded rows: a removed person keeps its id");
    jr_free(root);
    jw_free(&jw);
}

static void test_unload_rule(pn_people *people)
{
    mr_registry reg;
    pn_local_ctx ctx;
    char resp[2048];
    pn_loaded_entry rows[] = {
        entry("tiny", 1, 6, g_bob.id), entry("tiny", 2, -1, g_alice.id),
        entry("solo", 1, -1, g_bob.id), entry("other", 1, -1, g_bob.id),
        entry("gone", 1, -1, "p0000dead"),
    };

    printf("unload_rule:\n");
    CHECK(mr_init(&reg) == 0, "registry");
    memset(&ctx, 0, sizeof(ctx));
    ctx.reg = &reg;
    ctx.people = people;
    ctx.loaded = pn_loaded_set_open(DIR "/rule.json");
    pn_replica_affinity_reset();
    seed_row(&reg, 0, "tiny", 1, 6);        /* bob's, batched, one session on it */
    seed_row(&reg, 1, "hand", 1, -1);       /* loaded by hand: not remembered */
    pn_batch_sched *sched = fake_sched(2);
    reg.entries[0].sched = sched;
    snprintf(sched->slots[0].sid, sizeof(sched->slots[0].sid), "s1");
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        pn_loaded_set_add(ctx.loaded, &rows[i]);
    test_loaded_rows(&ctx);

    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"tiny\",\"replica\":2}", resp, sizeof(resp)) ==
          403 && strstr(resp, "\"unload_refused\"") &&
          strstr(resp, "instance 2 of tiny was loaded by alice"),
          "someone else's replica: 403 unload_refused, loaded by alice");
    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"tiny\",\"replica\":1}", resp, sizeof(resp)) ==
          403 && strstr(resp, "\"unload_refused\"") &&
          strstr(resp, "instance 1 of tiny has 1 session on it"),
          "their own replica with a session: 403 with the count");
    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"tiny\"}", resp, sizeof(resp)) == 403 &&
          remembered(&ctx, "tiny", 1) && remembered(&ctx, "tiny", 2),
          "{name}: every replica must pass; nothing unloaded");
    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"hand\"}", resp, sizeof(resp)) == 403 &&
          strstr(resp, "not loaded through the box"), "a hand-loaded row: refused");
    CHECK(unload_as(&ctx, &g_alice, "{\"name\":\"other\"}", resp, sizeof(resp)) == 403 &&
          strstr(resp, "loaded by bob"), "alice cannot unload bob's");
    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"gone\"}", resp, sizeof(resp)) == 403 &&
          strstr(resp, "a person no longer on this box"), "a removed person's: refused");
    CHECK(unload_as(&ctx, &g_bob, "{\"name\":\"solo\"}", resp, sizeof(resp)) == 200 &&
          !remembered(&ctx, "solo", 1), "their own idle replica: unloaded");
    CHECK(unload_as(&ctx, &g_david, "{\"name\":\"tiny\",\"replica\":2}", resp, sizeof(resp)) ==
          200 && !remembered(&ctx, "tiny", 2), "the admin unloads alice's");
    CHECK(unload_as(&ctx, &g_david, "{\"name\":\"other\"}", resp, sizeof(resp)) == 200 &&
          !remembered(&ctx, "other", 1), "the admin unloads bob's");

    pn_replica_affinity_reset();
    pn_loaded_set_close(ctx.loaded);
    pthread_cond_destroy(&sched->cv_idle);
    pthread_mutex_destroy(&sched->mu);
    free(sched);
    memset(reg.entries, 0, sizeof(reg.entries));   /* fake models: never pn_model_free */
    reg.n = 0;
    mr_free(&reg);
}

/* The boot reload loads the remembered replica and keeps who loaded it. */
static void test_reload_keeps_loaded_by(const char *model)
{
    pn_loaded_entry e = entry("tiny-cpu", 1, -1, g_bob.id), list[PN_LOADED_MAX];
    mr_registry reg;
    pn_local_ctx ctx;
    char err[256];
    int n = 0, status = 0;

    printf("reload_keeps_loaded_by:\n");
    snprintf(e.path, sizeof(e.path), "%s", model);
    e.max_sessions = 1;
    CHECK(mr_init(&reg) == 0, "registry");
    memset(&ctx, 0, sizeof(ctx));
    snprintf(ctx.state_dir, sizeof(ctx.state_dir), "%s", DIR);
    snprintf(ctx.store_dir, sizeof(ctx.store_dir), "%s", DIR);
    ctx.started = time(NULL);
    ctx.reg = &reg;
    ctx.loaded = pn_loaded_set_open(DIR "/reload.json");
    CHECK(ctx.loaded && pn_loaded_set_add(ctx.loaded, &e) == 0, "remembered with loaded_by bob");
    pn_loaded_set_close(ctx.loaded);
    ctx.loaded = pn_loaded_set_open(DIR "/reload.json");   /* a restart */
    pn_local_reload_now(&ctx);
    CHECK(pn_local_reload_state("tiny-cpu", 1, NULL, 0) == PN_RELOAD_LOADED, "reloaded");
    CHECK(pn_loaded_set_list(ctx.loaded, list, PN_LOADED_MAX) == 1 &&
          strcmp(list[0].loaded_by, g_bob.id) == 0, "the reload keeps loaded_by");
    pn_loaded_set_close(ctx.loaded);
    ctx.loaded = pn_loaded_set_open(DIR "/reload.json");
    CHECK(pn_loaded_set_list(ctx.loaded, list, PN_LOADED_MAX) == 1 &&
          strcmp(list[0].loaded_by, g_bob.id) == 0, "and in the saved file");
    CHECK(pn_local_unload(&ctx, "tiny-cpu", 0, &n, &status, err, sizeof(err)) == 0 && n == 1,
          "unloaded again");
    pn_loaded_set_close(ctx.loaded);
    mr_free(&reg);
}

int main(void)
{
    const char *model = getenv("PN_TEST_MODEL");
    pn_people *people = NULL;

    printf("test_local_loads_by_person\n");
    signal(SIGPIPE, SIG_IGN);
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    people = pn_people_open(DIR "/people.json");
    pn_people_add(people, "david", "David", 1, NULL, 1, &g_david);
    pn_people_add(people, "bob", "Bob", 0, NULL, 1, &g_bob);
    pn_people_add(people, "alice", "Alice", 0, NULL, 1, &g_alice);
    test_loaded_set_file(people);
    test_loader_class(people);
    test_unload_rule(people);
    if (!model || !*model)
        model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(model, R_OK) == 0)
        test_reload_keeps_loaded_by(model);
    else
        printf("  SKIP reload_keeps_loaded_by: %s not found\n", model);
    pn_people_close(people);
    return lt_report("test_local_loads_by_person");
}
