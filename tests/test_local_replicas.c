/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_replicas.c — Pantheon Local L4: unload one replica or all.
 *
 * Two copies of one catalog model under one served name (CPU-resident here: a
 * CPU build takes no dies; the die-set side is test_die_manager's
 * release_one_replica and tests/local/replicas_bank.sh on the K80 bank):
 *   - the loaded list shows each replica with its number and load option;
 *   - unloading replica 2 leaves replica 1 serving and remembered;
 *   - unloading the name takes every replica down and out of the set;
 *   - a replica that does not exist is a 404 with a plain reason.
 * Uses models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL); skipped when absent.
 *
 * sessions_used (no model file): hand-built registry rows (a zeroed pn_model,
 * as test_backend_guard): a batched copy on a die (a hand-built scheduler) and
 * two CPU copies (no session store). The batched copy reports the session
 * families holding its decode slots — not its routing rows, which outlive a
 * session's seat; a CPU copy, having no seats, reports its routing rows. Both
 * show per replica in the loaded list and in /admin/models.
 */

#include "local_test_support.h"

#include "../src/local/local_load.h"
#include "../src/local/local_model_routes.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/model_admin.h"
#include "../src/serve/replica_affinity.h"
#include "../src/core/engine.h"

#define DIR "build/test_local/replicas"

static const char *g_model;

static int copies(mr_registry *reg, const char *name)
{
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS), k = 0;

    for (int i = 0; i < n; i++)
        k += strcmp(info[i].name, name) == 0;
    return k;
}

/* Load one CPU copy of the test model as replica `replica` and remember it. */
static int load_copy(mr_registry *reg, pn_local_ctx *ctx, int replica, int ctx_cap)
{
    ma_load_request lr;
    pn_loaded_entry e;
    char err[256];

    memset(&lr, 0, sizeof(lr));
    lr.path = g_model;
    lr.name = "tiny";
    lr.opts.ctx_cap = ctx_cap;
    lr.opts.cpu_only = true;
    lr.max_sessions = 1;
    if (ma_load_model(reg, &lr, err, sizeof(err)) != 0)
        return -1;
    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "tiny");
    snprintf(e.profile_id, sizeof(e.profile_id), "cpu");
    snprintf(e.path, sizeof(e.path), "%s", g_model);
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), "cpu");
    e.hardware.count = 1;
    e.replica = replica;
    e.ctx_cap = ctx_cap;
    e.max_sessions = 1;
    return pn_loaded_set_add(ctx->loaded, &e);
}

static void test_loaded_rows(pn_local_ctx *ctx)
{
    json_writer jw;

    jw_init(&jw);
    pn_local_write_loaded(&jw, ctx);
    jr_value *root = jr_parse(jw.data);
    const jr_value *m = jr_object_get(root, "models");
    CHECK(jr_array_len(m) == 2, "loaded API: one row per replica");
    const jr_value *r2 = jr_array_get(m, 1);
    CHECK(jr_as_number(jr_object_get(r2, "replica"), 0) == 2 &&
              jr_as_number(jr_object_get(r2, "ctx_cap"), 0) == 4096 &&
              jr_as_number(jr_object_get(r2, "max_sessions"), 0) == 1,
          "loaded API: replica number and load option");
    CHECK(strcmp(jr_as_string(jr_object_get(r2, "state")), "loaded") == 0,
          "loaded API: the second copy is loaded");
    jr_free(root);
    jw_free(&jw);
}

static void test_unload_one_then_all(mr_registry *reg, pn_local_ctx *ctx)
{
    pn_loaded_entry list[PN_LOADED_MAX];
    char err[256];
    int n = 0, status = 0;

    printf("unload_one_then_all:\n");
    CHECK(load_copy(reg, ctx, 1, 2048) == 0 && load_copy(reg, ctx, 2, 4096) == 0,
          "two copies under one name");
    CHECK(copies(reg, "tiny") == 2, "the registry holds both");
    test_loaded_rows(ctx);

    CHECK(pn_local_unload(ctx, "tiny", 3, &n, &status, err, sizeof(err)) != 0 &&
              status == 404 && strstr(err, "no replica 3"),
          "an unknown replica: 404 with the reason");
    CHECK(pn_local_unload(ctx, "tiny", 2, &n, &status, err, sizeof(err)) == 0 && n == 1,
          "unload replica 2");
    CHECK(copies(reg, "tiny") == 1, "one copy still serves");
    int k = pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX);
    CHECK(k == 1 && list[0].replica == 1, "replica 1 stays remembered");
    mr_entry *e = mr_acquire_session(reg, "tiny", "s1");
    CHECK(e != NULL, "the name still routes");
    if (e)
        mr_release(reg, e);

    CHECK(load_copy(reg, ctx, 2, 4096) == 0, "load copy 2 again");
    CHECK(pn_local_unload(ctx, "tiny", 0, &n, &status, err, sizeof(err)) == 0 && n == 2,
          "unload all: both copies");
    CHECK(copies(reg, "tiny") == 0 && pn_loaded_set_list(ctx->loaded, list, PN_LOADED_MAX) == 0,
          "nothing loaded, nothing remembered");
    CHECK(pn_local_unload(ctx, "tiny", 0, &n, &status, err, sizeof(err)) != 0 && status == 404,
          "unloading it again: 404");
}

/* A zeroed pn_model: mr_snapshot reads only config fields and gauges off it. */
static pn_model g_fake_model;

/* Hand-build registry row `slot`: replica `replica` of "tiny" on `die` (-1 = CPU). */
static void seed_row(mr_registry *reg, int slot, int replica, uint64_t serial, int die)
{
    mr_entry *e = &reg->entries[slot];

    memset(e, 0, sizeof(*e));
    e->used = true;
    snprintf(e->name, sizeof(e->name), "tiny");
    e->model = &g_fake_model;
    e->serial = serial;
    e->replica = replica;
    e->max_sessions = 2;
    e->max_concurrent = 2;
    if (die >= 0) {
        e->dies[0] = die;
        e->n_dies = 1;
    }
    reg->n++;
}

/* A scheduler with `cap` idle decode slots and nothing else: enough for
 * session counting and release (no driver, no model, no slot state). */
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

static void fake_sched_free(pn_batch_sched *s)
{
    pthread_cond_destroy(&s->cv_idle);
    pthread_mutex_destroy(&s->mu);
    free(s);
}

/* Remember replica `replica` of "tiny" (unit `unit`, -1 = CPU). */
static int remember(pn_local_ctx *ctx, int replica, int unit)
{
    pn_loaded_entry e;

    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "tiny");
    snprintf(e.profile_id, sizeof(e.profile_id), unit >= 0 ? "k80" : "cpu");
    snprintf(e.path, sizeof(e.path), "tiny.gguf");
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), unit >= 0 ? "k80" : "cpu");
    e.hardware.count = 1;
    e.replica = replica;
    e.ctx_cap = 2048;
    e.max_sessions = 2;
    if (unit >= 0) {
        e.units[0] = unit;
        e.n_units = 1;
    }
    return pn_loaded_set_add(ctx->loaded, &e);
}

/* A session's turn: route it (binding a new one) and finish. */
static int turn(mr_registry *reg, const char *sid)
{
    mr_entry *e = mr_acquire_session(reg, "tiny", sid);
    int replica = e ? e->replica : 0;

    mr_release(reg, e);
    return replica;
}

/* sessions_used of each loaded row, by replica (1..3); -1 = absent. */
static void loaded_used(pn_local_ctx *ctx, int used[4])
{
    json_writer jw;

    for (int i = 0; i < 4; i++)
        used[i] = -1;
    jw_init(&jw);
    pn_local_write_loaded(&jw, ctx);
    jr_value *root = jr_parse(jw.data);
    const jr_value *m = jr_object_get(root, "models");
    for (int i = 0; i < jr_array_len(m); i++) {
        const jr_value *r = jr_array_get(m, i);
        int replica = (int)jr_as_number(jr_object_get(r, "replica"), 0);
        const jr_value *u = jr_object_get(r, "sessions_used");
        if (replica >= 1 && replica <= 3 && u &&
            strcmp(jr_as_string(jr_object_get(r, "state")), "loaded") == 0)
            used[replica] = (int)jr_as_number(u, -1);
    }
    jr_free(root);
    jw_free(&jw);
}

/* GET /admin/models over a socketpair: the whole response. Caller frees. */
static char *admin_models(mr_registry *reg)
{
    int sv[2];
    size_t cap = 65536, len = 0;
    char *buf;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return NULL;
    ma_handle_models(sv[1], reg);
    close(sv[1]);
    buf = malloc(cap);
    for (ssize_t r; buf && len + 1 < cap && (r = read(sv[0], buf + len, cap - len - 1)) > 0;)
        len += (size_t)r;
    if (buf)
        buf[len] = '\0';
    close(sv[0]);
    return buf;
}

static void test_sessions_used(void)
{
    mr_registry reg;
    pn_local_ctx ctx;
    int used[4];

    printf("sessions_used:\n");
    CHECK(mr_init(&reg) == 0, "registry");
    memset(&ctx, 0, sizeof(ctx));
    ctx.reg = &reg;
    ctx.loaded = pn_loaded_set_open(DIR "/sessions.json");
    CHECK(ctx.loaded != NULL, "loaded set");
    if (!ctx.loaded)
        return;
    pn_replica_affinity_reset();
    seed_row(&reg, 0, 1, 101, 6);    /* replica 1 on die 6, batched */
    seed_row(&reg, 1, 2, 102, -1);   /* replicas 2 and 3 on the CPU */
    seed_row(&reg, 2, 3, 103, -1);
    pn_batch_sched *sched = fake_sched(2);
    reg.entries[0].sched = sched;
    CHECK(remember(&ctx, 1, 6) == 0 && remember(&ctx, 2, -1) == 0 && remember(&ctx, 3, -1) == 0,
          "three replicas remembered");

    loaded_used(&ctx, used);
    CHECK(used[1] == 0 && used[2] == 0 && used[3] == 0, "no sessions: 0 used on every replica");

    /* Routing counts seats on the batched copy, so s1's turn takes its seat. */
    CHECK(turn(&reg, "s1") == 1, "s1 on the first replica");
    snprintf(sched->slots[0].sid, sizeof(sched->slots[0].sid), "s1");   /* s1 ran */
    CHECK(turn(&reg, "s2") == 2 && turn(&reg, "s3") == 3 && turn(&reg, "s4") == 1,
          "four sessions spread over the replicas");
    CHECK(turn(&reg, "s1") == 1, "a later turn of s1 binds nothing new");
    loaded_used(&ctx, used);
    CHECK(used[1] == 1 && used[2] == 1 && used[3] == 1,
          "batched copy: seats held (s1), not routing rows (s1, s4); CPU copies: rows");

    char *body = admin_models(&reg);
    CHECK(body && strstr(body, "\"replica\":1,") &&
              strstr(body, "\"max_sessions\":2,\"sessions_used\":1,") &&
              !strstr(body, "\"sessions_used\":2,"),
          "/admin/models carries sessions_used per replica");
    free(body);

    pn_replica_affinity_forget("s4");   /* /session/release (or its watch closed) */
    pn_replica_affinity_forget("s2");
    loaded_used(&ctx, used);
    CHECK(used[1] == 1 && used[2] == 0 && used[3] == 1, "released sessions no longer count");

    /* A tab: its watch binds the base id, its turns run on "<base>-<mode>". */
    CHECK(turn(&reg, "argus-7-1") == 2 && turn(&reg, "argus-7-1-chat") == 2,
          "a tab lands on the idle replica 2 and stays there");
    loaded_used(&ctx, used);
    CHECK(used[2] == 1, "a tab counts once");

    /* The seat goes (released or reaped) while the routing row stays. */
    snprintf(sched->slots[1].sid, sizeof(sched->slots[1].sid), "s1-side");
    loaded_used(&ctx, used);
    CHECK(used[1] == 1, "s1 and s1-side are one session family");
    {
        json_writer jw;
        jw_init(&jw);
        pn_local_write_loaded(&jw, &ctx);
        CHECK(jw.data && strstr(jw.data, "\"sessions_used\":1,\"slots_held\":2,"
                                         "\"seat_free\":false"),
              "one family on both slots: 1 session, 2 slots held, no seat free");
        jw_free(&jw);
        char *admin = admin_models(&reg);
        CHECK(admin && strstr(admin, "\"sessions_used\":1,\"slots_held\":2,"
                                     "\"seat_free\":false"),
              "/admin/models carries slots_held and seat_free");
        free(admin);
    }
    bs_session_release(sched, "s1");
    bs_session_release(sched, "s1-side");
    loaded_used(&ctx, used);
    CHECK(used[1] == 0, "a released seat stops counting though s1's routing row remains");

    pn_replica_affinity_forget_serial(103);   /* replica 3 unloaded */
    loaded_used(&ctx, used);
    CHECK(used[2] == 1 && used[3] == 0, "an unload drops only that replica's sessions");

    pn_replica_affinity_reset();
    pn_loaded_set_close(ctx.loaded);
    fake_sched_free(sched);
    memset(reg.entries, 0, sizeof(reg.entries));   /* fake models: never pn_model_free */
    reg.n = 0;
    mr_free(&reg);
}

int main(void)
{
    mr_registry reg;
    pn_local_ctx ctx;

    printf("test_local_replicas\n");
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    test_sessions_used();
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP: %s not found\n", g_model);
        return lt_report("test_local_replicas");
    }
    CHECK(mr_init(&reg) == 0, "registry");
    memset(&ctx, 0, sizeof(ctx));
    snprintf(ctx.state_dir, sizeof(ctx.state_dir), "%s", DIR);
    snprintf(ctx.store_dir, sizeof(ctx.store_dir), "%s", DIR);
    ctx.started = time(NULL);
    ctx.reg = &reg;
    ctx.loaded = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(ctx.loaded != NULL, "loaded set");
    if (ctx.loaded) {
        test_unload_one_then_all(&reg, &ctx);
        pn_loaded_set_close(ctx.loaded);
    }
    mr_free(&reg);
    return lt_report("test_local_replicas");
}
