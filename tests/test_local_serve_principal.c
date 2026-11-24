/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_serve_principal.c — people registry P5.2: serve sessions belong to a
 * person (spec §4.8), on BOTH session paths.
 *   A. The table (ms_session_principal_check, no model): first use binds; another
 *      person -> MS_NOT_YOURS, for the id and every id of its family; a check or
 *      an export never binds; an unbound session exports for an admin only;
 *      release and release-by-prefix forget bindings on the batched arm (a fake
 *      scheduler) and the single-user arm (a prefix cache); seats per person
 *      count families (batched: holding a decode slot; single-user: bound on that
 *      entry); a full table forgets the least recently used binding; no principal
 *      (plain `pantheon serve`) binds and refuses nothing.
 *   B. The real handlers on both arms (hand-built entries: the refusal comes
 *      before any model work): chat, restore, release and /admin/session/export by
 *      another person -> 403 session_not_yours; the owner's release frees the
 *      binding; the owner's and an admin's export pass the check; with no
 *      principal every handler answers as before.
 *   C. The single-user path end to end (models/qwen3-0.6b-Q8_0.gguf on the CPU with
 *      a prefix cache attached, PN_TEST_MODEL; skipped when absent): chat, export,
 *      restore and release by owner and by another person; release frees the id;
 *      plain (no principal) chat / export / release unchanged; a session opened with
 *      no principal exports for an admin only; an export with "release":true frees
 *      the slot and binding as it returns (another person's frees nothing).
 *   D. GET /session/watch on both arms (the manager the route calls, hand-built
 *      entries): a watch on another person's session (or its family) is refused
 *      and holds nothing; a watch opened on an unbound id that another person then
 *      binds does not release it when it dies; the owner's watch holds and its
 *      death releases; with no principal a watch releases as before.
 *   E. The route over HTTP: a forked `pantheon serve` on the CPU with the same
 *      model (a test auth hook sets the person from the bearer, as Pantheon Local's
 *      does; a session cache attached at listen): another person's watch -> 403
 *      session_not_yours and the session survives; a watch on an unbound id that
 *      the owner then opens does not release it on close; the owner's watch
 *      releases on close; plain serve unchanged.
 * The batched arm's generation needs a GPU scheduler: on the bank, not here.
 */

#include "local_test_support.h"

#include <pthread.h>

#include "../src/core/prefix_cache.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/http_server.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/json_writer.h"
#include "../src/serve/model_admin.h"
#include "../src/serve/model_session.h"
#include "../src/serve/openai_chat.h"
#include "../src/serve/replica_affinity.h"
#include "../src/serve/serve_command.h"
#include "../src/serve/session_watch.h"

/* The manual (no-thread) watch manager (session_watch.c; not in the header). */
pn_session_watch_mgr *pn_session_watch_new_manual(mr_registry *reg);

#define DAVID "p00000001"
#define BOB   "p00000002"
#define CAROL "p00000003"

/* ── socketpair calls with a concurrent reader (an export can be megabytes) ── */

typedef struct {
    int    fd;
    char  *buf;
    size_t len;
} drain_job;

static void *drain_main(void *arg)
{
    drain_job *j = arg;
    size_t cap = 1 << 16;

    j->buf = malloc(cap);
    for (ssize_t r; j->buf && (r = read(j->fd, j->buf + j->len, cap - j->len - 1)) > 0;) {
        j->len += (size_t)r;
        if (j->len + 1 >= cap) {
            char *bigger = realloc(j->buf, cap * 2);
            if (!bigger)
                break;
            j->buf = bigger;
            cap *= 2;
        }
    }
    if (j->buf)
        j->buf[j->len] = '\0';
    return NULL;
}

typedef enum { CALL_CHAT, CALL_RESTORE, CALL_RELEASE, CALL_EXPORT } call_kind;

/* Run one handler as `principal` (NULL = plain serve). Returns the status; the
 * whole response goes to *resp (caller frees; may be NULL). */
static int call_as(const char *principal, int admin, call_kind kind, mr_registry *reg,
                   const char *body, char **resp)
{
    drain_job job = {0};
    pthread_t t;
    int sv[2], status = -1;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return -1;
    job.fd = sv[0];
    pthread_create(&t, NULL, drain_main, &job);
    hs_set_thread_principal(principal, admin);
    if (kind == CALL_CHAT)
        oai_handle_chat(sv[1], reg, body, NULL, NULL);
    else if (kind == CALL_RESTORE)
        oai_handle_session_restore(sv[1], reg, body, NULL, NULL);
    else if (kind == CALL_RELEASE)
        oai_handle_session_release(sv[1], reg, body, NULL);
    else
        ma_handle_session_export(sv[1], reg, body);
    hs_set_thread_principal(NULL, 0);
    close(sv[1]);
    pthread_join(t, NULL);
    close(sv[0]);
    if (job.buf && sscanf(job.buf, "HTTP/1.1 %d", &status) != 1)
        status = -1;
    if (resp)
        *resp = job.buf;
    else
        free(job.buf);
    return status;
}

static int not_yours(int status, const char *resp)
{
    return status == 403 && resp && strstr(resp, "\"session_not_yours\"");
}

/* ── A. the table ─────────────────────────────────────────────────────────── */

static pn_model g_fake_model;

static void seed_row(mr_registry *reg, int slot, const char *name, uint64_t serial, int die)
{
    mr_entry *e = &reg->entries[slot];

    memset(e, 0, sizeof(*e));
    e->used = true;
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->model = &g_fake_model;
    e->serial = serial;
    e->replica = 1;
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

static ms_result chk(mr_entry *e, const char *sid, const char *who, int admin, ms_principal_use u)
{
    return ms_session_principal_check(e, sid, who, admin, u);
}

/* Seats of `who` on `e` (0 when absent). */
static int seats_of(mr_entry *e, const char *who)
{
    ms_principal_seats s[8];
    int n = ms_seats_by_principal(e, s, 8);

    for (int i = 0; i < n; i++)
        if (strcmp(s[i].principal, who) == 0)
            return s[i].held;
    return 0;
}

static void test_table(mr_entry *bat, mr_entry *one)
{
    printf("table:\n");
    CHECK(chk(bat, "t-p", NULL, 0, MS_PRINCIPAL_BIND) == MS_OK &&
          chk(bat, "t-p", NULL, 0, MS_PRINCIPAL_EXPORT) == MS_OK &&
          chk(bat, "t-p", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "no principal: passes and binds nothing (bob may then take the id)");
    ms_session_release(bat, "t-p");
    CHECK(chk(bat, "t-d", DAVID, 1, MS_PRINCIPAL_BIND) == MS_OK, "david binds t-d");
    CHECK(chk(bat, "t-d", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS &&
          chk(bat, "t-d", BOB, 0, MS_PRINCIPAL_CHECK) == MS_NOT_YOURS &&
          chk(bat, "t-d", BOB, 0, MS_PRINCIPAL_EXPORT) == MS_NOT_YOURS,
          "bob: chat, release and export of t-d refused");
    CHECK(chk(bat, "t-d-orch", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS &&
          chk(one, "t-d", BOB, 1, MS_PRINCIPAL_BIND) == MS_NOT_YOURS,
          "the family and every copy too (even for an admin)");
    CHECK(chk(bat, "t-d", DAVID, 1, MS_PRINCIPAL_EXPORT) == MS_OK &&
          chk(bat, "t-d-orch", DAVID, 1, MS_PRINCIPAL_BIND) == MS_OK, "david: export and family");
    CHECK(chk(bat, "t-legacy", BOB, 0, MS_PRINCIPAL_EXPORT) == MS_NOT_YOURS &&
          chk(bat, "t-legacy", DAVID, 1, MS_PRINCIPAL_EXPORT) == MS_OK &&
          chk(bat, "t-legacy", BOB, 0, MS_PRINCIPAL_CHECK) == MS_OK,
          "unbound: export for an admin only; a release passes");
    CHECK(chk(bat, "t-legacy", CAROL, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "an export or a check bound nothing (carol takes it)");
    CHECK(chk(bat, "t-x-a", DAVID, 1, MS_PRINCIPAL_BIND) == MS_OK &&
          chk(bat, "t-x-b", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK &&
          chk(bat, "t-x", DAVID, 1, MS_PRINCIPAL_BIND) == MS_NOT_YOURS,
          "siblings are separate; their base meets both");

    ms_session_release(bat, "t-d");
    CHECK(chk(bat, "t-d", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS,
          "batched release of t-d: t-d-orch still david's");
    ms_session_release_by_prefix(bat, "t-d");
    CHECK(chk(bat, "t-d", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "batched release by prefix frees the family (bob binds it)");
    CHECK(chk(one, "t-u", DAVID, 1, MS_PRINCIPAL_BIND) == MS_OK &&
          chk(one, "t-u", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS, "single-user: bound");
    ms_session_release(one, "t-u");
    CHECK(chk(one, "t-u", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "single-user release frees it (bob binds it)");
    ms_session_release_by_prefix(bat, "t-");
    ms_session_release_by_prefix(one, "t-");
}

static void test_seats(mr_entry *bat, mr_entry *one)
{
    printf("seats:\n");
    chk(bat, "t-a", DAVID, 1, MS_PRINCIPAL_BIND);
    chk(bat, "t-a-orch", DAVID, 1, MS_PRINCIPAL_BIND);
    chk(bat, "t-b", BOB, 0, MS_PRINCIPAL_BIND);
    chk(bat, "t-c", CAROL, 0, MS_PRINCIPAL_BIND);
    snprintf(bat->sched->slots[0].sid, sizeof(bat->sched->slots[0].sid), "t-a-orch");
    snprintf(bat->sched->slots[1].sid, sizeof(bat->sched->slots[1].sid), "t-b");
    CHECK(seats_of(bat, DAVID) == 1 && seats_of(bat, BOB) == 1 && seats_of(bat, CAROL) == 0,
          "batched: one family each holding a slot; carol holds none");
    chk(one, "t-u1", DAVID, 1, MS_PRINCIPAL_BIND);
    chk(one, "t-u2", DAVID, 1, MS_PRINCIPAL_BIND);
    CHECK(seats_of(one, DAVID) == 2 && seats_of(one, BOB) == 0,
          "single-user: david's two families bound there; bob's are elsewhere");
    ms_session_release(bat, "t-b");
    ms_session_release(one, "t-u2");
    CHECK(seats_of(bat, BOB) == 0 && seats_of(one, DAVID) == 1, "released seats stop counting");
    bs_session_release(bat->sched, "t-a-orch");   /* the idle reaper took the slot */
    CHECK(seats_of(bat, DAVID) == 0, "batched: a reaped slot stops counting");
    ms_session_release_by_prefix(bat, "t-");
    ms_session_release_by_prefix(one, "t-");
}

static void test_table_full(mr_entry *bat)
{
    char sid[32];
    int ok = 1;

    printf("table_full:\n");
    for (int i = 0; i < MS_PRINCIPAL_BINDINGS_MAX; i++) {
        snprintf(sid, sizeof(sid), "t-e%d", i);
        ok &= chk(bat, sid, CAROL, 0, MS_PRINCIPAL_BIND) == MS_OK;
    }
    CHECK(ok && chk(bat, "t-e0", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS, "512 bound");
    usleep(2000);
    chk(bat, "t-e1", CAROL, 0, MS_PRINCIPAL_BIND);   /* used again: not the oldest */
    CHECK(chk(bat, "t-new", CAROL, 0, MS_PRINCIPAL_BIND) == MS_OK, "the 513th binds");
    CHECK(chk(bat, "t-e0", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK &&
          chk(bat, "t-e1", BOB, 0, MS_PRINCIPAL_BIND) == MS_NOT_YOURS,
          "the least recently used binding was forgotten");
    ms_session_release_by_prefix(bat, "t-");
}

/* ── B. the real handlers on hand-built entries ───────────────────────────── */

static void test_handlers_arm(mr_registry *reg, const char *model)
{
    char body[512], *resp = NULL;
    int st = 0;

    printf("handlers (%s):\n", model);
    mr_entry *e = mr_acquire(reg, model);
    chk(e, "t-h", DAVID, 1, MS_PRINCIPAL_BIND);
    mr_release(reg, e);
    snprintf(body, sizeof(body), "{\"model\":\"%s\",\"session_id\":\"t-h\",\"max_tokens\":2,"
             "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}", model);
    st = call_as(BOB, 0, CALL_CHAT, reg, body, &resp);
    CHECK(not_yours(st, resp), "chat by another person: 403 session_not_yours");
    free(resp);
    snprintf(body, sizeof(body), "{\"name\":\"%s\",\"session_id\":\"t-h\",\"blob\":\"AAAA\","
             "\"tokens\":[1,2]}", model);
    st = call_as(BOB, 0, CALL_RESTORE, reg, body, &resp);
    CHECK(not_yours(st, resp), "restore by another person: 403");
    free(resp);
    snprintf(body, sizeof(body), "{\"name\":\"%s\",\"session_id\":\"t-h\"}", model);
    st = call_as(BOB, 1, CALL_EXPORT, reg, body, &resp);
    CHECK(not_yours(st, resp), "export by another person (an admin too): 403");
    free(resp);
    st = call_as(DAVID, 1, CALL_EXPORT, reg, body, &resp);
    CHECK(st == 400 && strstr(resp, "session_export_unsupported"),
          "export by the owner passes the check (the fake arch then refuses)");
    free(resp);
    st = call_as(BOB, 0, CALL_RELEASE, reg, body, &resp);
    CHECK(not_yours(st, resp), "release by another person: 403");
    free(resp);
    st = call_as(DAVID, 1, CALL_RELEASE, reg, body, NULL);
    e = mr_acquire(reg, model);
    CHECK(st == 200 && chk(e, "t-h", BOB, 0, MS_PRINCIPAL_CHECK) == MS_OK &&
          chk(e, "t-h", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "release by the owner: 200, and the id is free");
    ms_session_release(e, "t-h");
    mr_release(reg, e);

    snprintf(body, sizeof(body), "{\"name\":\"%s\",\"session_id\":\"t-old\"}", model);
    st = call_as(BOB, 0, CALL_EXPORT, reg, body, &resp);
    CHECK(not_yours(st, resp), "an unbound session: export by a non-admin refused");
    free(resp);
    st = call_as(DAVID, 1, CALL_EXPORT, reg, body, &resp);
    CHECK(st == 400 && strstr(resp, "session_export_unsupported"), "an admin passes");
    free(resp);
    st = call_as(NULL, 0, CALL_EXPORT, reg, body, &resp);
    CHECK(st == 400 && strstr(resp, "session_export_unsupported"),
          "no principal (plain serve): export as before");
    free(resp);
    st = call_as(NULL, 0, CALL_RELEASE, reg, body, NULL);
    CHECK(st == 200, "no principal: release as before");
    snprintf(body, sizeof(body), "{\"name\":\"%s\",\"session_id\":\"t-old\",\"blob\":\"!!\","
             "\"tokens\":[1]}", model);
    st = call_as(NULL, 0, CALL_RESTORE, reg, body, &resp);
    CHECK(st == 400 && strstr(resp, "bad_session_blob"), "no principal: restore as before");
    free(resp);
}

/* ── D. the watch on both arms ───────────────────────────────────────────── */

/* Open a watch on `sid` as `who` (NULL = plain serve) through the manager, as the
 * route does. Returns pn_session_watch_add's code; *client is the client end of a
 * held watch, else -1. */
static int watch_as(pn_session_watch_mgr *mgr, mr_entry *e, const char *who, const char *sid,
                    int *client)
{
    int sv[2], rc = -1;

    *client = -1;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return -1;
    hs_set_thread_principal(who, 0);
    rc = pn_session_watch_add(mgr, sv[1], e, e->name, sid);
    hs_set_thread_principal(NULL, 0);
    if (rc == 0) {
        *client = sv[0];
    } else {
        close(sv[1]);
        close(sv[0]);
    }
    return rc;
}

/* The client hangs up on `sid`'s watch and the grace runs out. */
static void watch_dies(pn_session_watch_mgr *mgr, mr_entry *e, const char *sid, int client)
{
    close(client);
    pn_session_watch_mark_closed_for_test(mgr, e->name, sid, 1000);
    pn_session_watch_sweep_for_test(mgr, 1000 + pn_session_watch_grace_ms() + 1);
}

/* Batched: the session holds a decode slot (slot `i`). Single-user: nothing to do. */
static void hold_slot(mr_entry *e, int i, const char *sid)
{
    if (e->sched)
        snprintf(e->sched->slots[i].sid, sizeof(e->sched->slots[i].sid), "%s", sid);
}

/* Is `sid` still alive: bound to someone else than carol, and (batched) in its slot? */
static int session_alive(mr_entry *e, const char *sid)
{
    return chk(e, sid, CAROL, 0, MS_PRINCIPAL_CHECK) == MS_NOT_YOURS &&
           (!e->sched || bs_session_family_holds(e->sched, sid));
}

static void test_watch_arm(mr_registry *reg, const char *model)
{
    mr_entry *e = mr_acquire(reg, model);
    pn_session_watch_mgr *mgr = pn_session_watch_new_manual(reg);
    int c = -1, cb = -1;

    printf("watch (%s):\n", model);
    chk(e, "t-w", DAVID, 1, MS_PRINCIPAL_BIND);
    hold_slot(e, 0, "t-w");
    CHECK(watch_as(mgr, e, BOB, "t-w", &c) == MS_NOT_YOURS && c < 0 &&
          watch_as(mgr, e, BOB, "t-w-new2", &c) == MS_NOT_YOURS &&
          !pn_session_watch_base_is_watched(mgr, model, "t-w"),
          "bob's watch on david's session (and its family): refused, nothing held");
    CHECK(session_alive(e, "t-w"), "david's session survives");
    CHECK(watch_as(mgr, e, BOB, "t-v", &cb) == 0, "bob may watch an unbound id");
    chk(e, "t-v", DAVID, 1, MS_PRINCIPAL_BIND);   /* then david opens it */
    hold_slot(e, 1, "t-v");
    watch_dies(mgr, e, "t-v", cb);
    CHECK(session_alive(e, "t-v"), "bob's watch dying does not release david's session");
    CHECK(watch_as(mgr, e, DAVID, "t-w", &c) == 0 &&
          pn_session_watch_base_is_watched(mgr, model, "t-w"), "david's own watch: held");
    watch_dies(mgr, e, "t-w", c);
    CHECK(!session_alive(e, "t-w") && chk(e, "t-w", BOB, 0, MS_PRINCIPAL_BIND) == MS_OK,
          "david's watch dying releases his session (bob may take the id)");
    ms_session_release(e, "t-w");
    hold_slot(e, 0, "t-p");
    CHECK(watch_as(mgr, e, NULL, "t-p", &c) == 0, "no principal: a watch as before");
    watch_dies(mgr, e, "t-p", c);
    CHECK(!e->sched || !bs_session_family_holds(e->sched, "t-p"),
          "no principal: its death releases as before");
    pn_session_watch_stop(mgr);
    ms_session_release_by_prefix(e, "t-");
    mr_release(reg, e);
}

static void test_fake_entries(void)
{
    mr_registry reg;

    CHECK(mr_init(&reg) == 0, "registry");
    pn_replica_affinity_reset();
    g_fake_model.config.arch = PN_ARCH_RWKVX;   /* no snapshot: export stops after the check */
    seed_row(&reg, 0, "bat", 501, 6);           /* batched arm */
    seed_row(&reg, 1, "one", 502, -1);          /* single-user arm */
    reg.entries[0].sched = fake_sched(2);
    reg.entries[1].prefix_cache = pn_prefix_cache_new(&g_fake_model, 2);
    test_table(&reg.entries[0], &reg.entries[1]);
    test_seats(&reg.entries[0], &reg.entries[1]);
    test_table_full(&reg.entries[0]);
    test_handlers_arm(&reg, "bat");
    test_handlers_arm(&reg, "one");
    reg.entries[0].prefix_cache = pn_prefix_cache_new(&g_fake_model, 2);   /* a watch needs one */
    test_watch_arm(&reg, "bat");
    test_watch_arm(&reg, "one");

    pn_replica_affinity_reset();
    pn_prefix_cache_free(reg.entries[0].prefix_cache);
    pn_prefix_cache_free(reg.entries[1].prefix_cache);
    pthread_cond_destroy(&reg.entries[0].sched->cv_idle);
    pthread_mutex_destroy(&reg.entries[0].sched->mu);
    free(reg.entries[0].sched);
    memset(reg.entries, 0, sizeof(reg.entries));   /* fake models: never pn_model_free */
    reg.n = 0;
    mr_free(&reg);
}

/* ── C. the single-user path with a real model ────────────────────────────── */

static int chat_as(mr_registry *reg, const char *who, int admin, const char *sid)
{
    char body[512];

    snprintf(body, sizeof(body), "{\"model\":\"small\",\"session_id\":\"%s\",\"max_tokens\":2,"
             "\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}", sid);
    return call_as(who, admin, CALL_CHAT, reg, body, NULL);
}

static int sid_call(mr_registry *reg, const char *who, int admin, call_kind kind,
                    const char *sid, char **resp)
{
    char body[256];

    snprintf(body, sizeof(body), "{\"name\":\"small\",\"session_id\":\"%s\"}", sid);
    return call_as(who, admin, kind, reg, body, resp);
}

/* The restore body for an export answer, into session `sid`. Caller frees. */
static char *restore_body(const char *export_resp, const char *sid)
{
    const char *json = export_resp ? strstr(export_resp, "\r\n\r\n") : NULL;
    jr_value *v = json ? jr_parse(json + 4) : NULL;
    const char *blob = jr_as_string(jr_object_get(v, "blob_b64"));
    json_writer jw;
    char *out = NULL;

    if (blob && jw_init(&jw)) {
        jw_printf(&jw, "{\"name\":\"small\",\"session_id\":\"%s\",\"blob\":", sid);
        jw_string(&jw, blob);
        jw_raw(&jw, ",\"tokens\":");
        jr_to_json(jr_object_get(v, "tokens"), &jw);
        jw_raw(&jw, "}");
        out = strdup(jw.data);
        jw_free(&jw);
    }
    jr_free(v);
    return out;
}

/* Export with "release":true (a closing client): the owner's answer still carries
 * the whole blob, but the slot and the binding are gone when it returns, so a new
 * session that was refused (429, every slot held) is admitted right away; another
 * person's is refused and frees nothing. */
static void test_export_with_release(mr_registry *reg, mr_entry *e)
{
    const char *body = "{\"name\":\"small\",\"session_id\":\"t-xr\",\"release\":true}";
    char *resp = NULL, *rb = NULL;

    printf("export_with_release:\n");
    CHECK(chat_as(reg, DAVID, 1, "t-xr") == 200 && chat_as(reg, DAVID, 1, "t-x1") == 200 &&
          chat_as(reg, DAVID, 1, "t-x2") == 200 && chat_as(reg, DAVID, 1, "t-x3") == 200 &&
          seats_of(e, DAVID) == 4, "david opens 4 sessions: every slot held");
    CHECK(chat_as(reg, DAVID, 1, "t-new") == 429, "a new session: 429 session capacity");
    CHECK(not_yours(call_as(BOB, 0, CALL_EXPORT, reg, body, &resp), resp),
          "bob's export with release: 403");
    free(resp);
    CHECK(sid_call(reg, DAVID, 1, CALL_EXPORT, "t-xr", NULL) == 200, "and t-xr is still live");
    int seats = seats_of(e, DAVID);   /* the refused t-new stays bound: not 4 */
    CHECK(call_as(DAVID, 1, CALL_EXPORT, reg, body, &resp) == 200 && strstr(resp, "blob_b64"),
          "david's export with release: 200 with the blob");
    CHECK(seats_of(e, DAVID) == seats - 1, "t-xr's seat is free when the export returns");
    CHECK(sid_call(reg, DAVID, 1, CALL_EXPORT, "t-xr", NULL) == 404, "t-xr is gone: export 404");
    CHECK(chat_as(reg, DAVID, 1, "t-new") == 200, "the new session is admitted right away");
    sid_call(reg, DAVID, 1, CALL_RELEASE, "t-new", NULL);
    rb = restore_body(resp, "t-xr2");
    free(resp);
    CHECK(rb && call_as(DAVID, 1, CALL_RESTORE, reg, rb, NULL) == 200,
          "the released export's blob restores as t-xr2");
    free(rb);
    sid_call(reg, DAVID, 1, CALL_RELEASE, "t-xr2", NULL);
    CHECK(chat_as(reg, BOB, 0, "t-xr") == 200, "the binding is gone: bob may open t-xr");
    sid_call(reg, BOB, 0, CALL_RELEASE, "t-xr", NULL);
    for (int i = 1; i <= 3; i++) {
        char sid[16];
        snprintf(sid, sizeof(sid), "t-x%d", i);
        sid_call(reg, DAVID, 1, CALL_RELEASE, sid, NULL);
    }
}

static void test_single_user_path(const char *model)
{
    pn_load_opts opts = { .ctx_cap = 2048 };
    char err[512] = "", *resp = NULL, *rb = NULL;
    mr_registry reg;

    printf("single_user_path:\n");
    CHECK(mr_init(&reg) == 0, "registry");
    CHECK(mr_load(&reg, model, "small", &opts, NULL, 0, 0, 4, 1, err, sizeof(err)) == 0,
          "small model loads on the CPU");
    mr_entry *e = mr_acquire(&reg, "small");
    if (!e) {
        mr_free(&reg);
        return;
    }
    e->prefix_cache = pn_prefix_cache_new(e->model, 4);   /* the single-user session store */
    CHECK(e->prefix_cache != NULL && !e->sched, "single-user arm: a prefix cache, no scheduler");

    CHECK(chat_as(&reg, DAVID, 1, "t-sd") == 200, "david's chat opens t-sd");
    CHECK(seats_of(e, DAVID) == 1, "david holds a seat on it");
    CHECK(chat_as(&reg, BOB, 0, "t-sd") == 403, "bob's chat on t-sd: 403");
    CHECK(not_yours(sid_call(&reg, BOB, 0, CALL_EXPORT, "t-sd", &resp), resp), "bob's export: 403");
    free(resp);
    CHECK(sid_call(&reg, DAVID, 1, CALL_EXPORT, "t-sd", &resp) == 200 && strstr(resp, "blob_b64"),
          "david's export: 200 with the blob");
    rb = restore_body(resp, "t-sd");
    free(resp);
    CHECK(rb && not_yours(call_as(BOB, 0, CALL_RESTORE, &reg, rb, &resp), resp),
          "bob's restore into t-sd: 403");
    free(resp);
    free(rb);
    CHECK(sid_call(&reg, DAVID, 1, CALL_EXPORT, "t-sd", &resp) == 200, "export again");
    rb = restore_body(resp, "t-sd2");
    free(resp);
    CHECK(rb && call_as(DAVID, 1, CALL_RESTORE, &reg, rb, NULL) == 200,
          "david restores it as t-sd2: 200 (binds t-sd2)");
    free(rb);
    CHECK(chat_as(&reg, BOB, 0, "t-sd2") == 403, "t-sd2 is david's");
    CHECK(not_yours(sid_call(&reg, BOB, 0, CALL_RELEASE, "t-sd", &resp), resp),
          "bob's release of t-sd: 403");
    free(resp);
    CHECK(sid_call(&reg, DAVID, 1, CALL_RELEASE, "t-sd", NULL) == 200 &&
          sid_call(&reg, DAVID, 1, CALL_RELEASE, "t-sd2", NULL) == 200, "david releases both");
    CHECK(chat_as(&reg, BOB, 0, "t-sd") == 200, "released: bob may open t-sd now");
    sid_call(&reg, BOB, 0, CALL_RELEASE, "t-sd", NULL);

    CHECK(chat_as(&reg, NULL, 0, "t-plain") == 200 &&
          sid_call(&reg, NULL, 0, CALL_EXPORT, "t-plain", NULL) == 200,
          "no principal (plain serve): chat and export as before");
    CHECK(not_yours(sid_call(&reg, BOB, 0, CALL_EXPORT, "t-plain", &resp), resp),
          "a session opened with no principal: a non-admin's export refused");
    free(resp);
    CHECK(sid_call(&reg, DAVID, 1, CALL_EXPORT, "t-plain", NULL) == 200, "an admin's export: 200");
    CHECK(sid_call(&reg, NULL, 0, CALL_RELEASE, "t-plain", NULL) == 200,
          "no principal: release as before");
    test_export_with_release(&reg, e);
    mr_release(&reg, e);
    mr_free(&reg);
}

/* ── E. the route over HTTP (a forked `pantheon serve`) ─────────────────── */

#define E2E_DIR "build/test_local/serve_principal"

/* Test auth hook: "Bearer david" / "Bearer bob" set that person, as Pantheon
 * Local's hook sets the caller's; no header = plain serve. */
static int e2e_auth(int fd, const hs_request *req, void *user)
{
    (void)fd;
    (void)user;
    if (req->authorization && strcmp(req->authorization, "Bearer david") == 0)
        hs_set_thread_principal(DAVID, 1);
    else if (req->authorization && strcmp(req->authorization, "Bearer bob") == 0)
        hs_set_thread_principal(BOB, 0);
    return 0;
}

/* A CPU load gets no session cache (GPU only): attach one, as section C does. */
static int e2e_attach_cache(mr_registry *reg, const char *host, int port, void *user)
{
    (void)host;
    (void)port;
    (void)user;
    pthread_mutex_lock(&reg->lock);
    for (int i = 0; i < MR_MAX_MODELS; i++)
        if (reg->entries[i].used && !reg->entries[i].prefix_cache)
            reg->entries[i].prefix_cache = pn_prefix_cache_new(reg->entries[i].model, 4);
    pthread_mutex_unlock(&reg->lock);
    return 0;
}

static pid_t e2e_spawn_serve(const char *model, int port)
{
    char port_s[16];
    char *argv[] = { "serve", "--model", (char *)model, "--name", "small", "--cpu", "--ctx-cap",
                     "2048", "--host", "127.0.0.1", "--port", port_s, NULL };
    pn_serve_local_hooks hooks = { .on_listening = e2e_attach_cache };

    snprintf(port_s, sizeof(port_s), "%d", port);
    lt_make_dir(E2E_DIR);
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        FILE *f = freopen(E2E_DIR "/serve.log", "w", stderr);
        (void)f;
        setvbuf(stderr, NULL, _IONBF, 0);
        setenv("PN_SESSION_WATCH_GRACE_MS", "300", 1);
        hs_set_auth_fn(e2e_auth, NULL);
        pn_serve_set_local_hooks(&hooks);
        _exit(pn_serve_main(12, argv));
    }
    if (pid > 0 && lt_wait_port(port, 120000) != 0) {
        lt_stop(pid);
        return -1;
    }
    return pid;
}

/* GET /session/watch as `key` (NULL = no principal). Returns the held fd on 200 (close
 * it to drop the watch); else -1, the status in *status and the answer in resp. */
static int e2e_watch(int port, const char *key, const char *sid, int *status, char *resp,
                     size_t cap)
{
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                             .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
    char head[512], auth[64] = "";
    size_t len = 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);

    *status = 0;
    resp[0] = '\0';
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0)
        goto fail;
    if (key)
        snprintf(auth, sizeof(auth), "Authorization: Bearer %s\r\n", key);
    int hn = snprintf(head, sizeof(head), "GET /session/watch?name=small&session_id=%s HTTP/1.1"
                      "\r\nHost: x\r\n%s\r\n", sid, auth);
    if (write(fd, head, (size_t)hn) != hn)
        goto fail;
    /* A held watch never closes: read until its keepalive line, a refusal to EOF. */
    for (ssize_t n; len + 1 < cap && (n = read(fd, resp + len, cap - len - 1)) > 0;) {
        len += (size_t)n;
        resp[len] = '\0';
        if (strstr(resp, ": watch established"))
            break;
    }
    if (sscanf(resp, "HTTP/1.1 %d", status) == 1 && *status == 200)
        return fd;
fail:
    if (fd >= 0)
        close(fd);
    return -1;
}

static int e2e_chat(int port, const char *key, const char *sid)
{
    char body[512];

    snprintf(body, sizeof(body), "{\"model\":\"small\",\"session_id\":\"%s\",\"max_tokens\":2,"
             "\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}", sid);
    return lt_http(port, "POST", "/v1/chat/completions", key, body, NULL);
}

static int e2e_export(int port, const char *key, const char *sid)
{
    char body[256];

    snprintf(body, sizeof(body), "{\"name\":\"small\",\"session_id\":\"%s\"}", sid);
    return lt_http(port, "POST", "/admin/session/export", key, body, NULL);
}

/* Past the 300 ms grace and the watch thread's 250 ms sweep. */
static void e2e_wait_grace(void)
{
    usleep(1200 * 1000);
}

static void test_watch_over_http(const char *model)
{
    char resp[4096];
    int port = lt_pick_port(9), st = 0, fd = -1;

    printf("watch_over_http:\n");
    pid_t pid = e2e_spawn_serve(model, port);
    CHECK(pid > 0, "serve up (" E2E_DIR "/serve.log)");
    if (pid <= 0)
        return;
    CHECK(e2e_chat(port, "david", "t-e1") == 200, "david's chat opens t-e1");
    fd = e2e_watch(port, "bob", "t-e1", &st, resp, sizeof(resp));
    CHECK(fd < 0 && st == 403 && strstr(resp, "\"session_not_yours\""),
          "bob's watch on t-e1: 403 session_not_yours");
    if (fd >= 0)
        close(fd);
    e2e_wait_grace();
    CHECK(e2e_export(port, "david", "t-e1") == 200 && e2e_chat(port, "bob", "t-e1") == 403,
          "t-e1 survives bob's watch closing (david exports it; still david's)");

    fd = e2e_watch(port, "bob", "t-e2", &st, resp, sizeof(resp));
    CHECK(fd >= 0, "bob may watch an unbound id t-e2");
    CHECK(e2e_chat(port, "david", "t-e2") == 200, "then david's chat opens t-e2");
    if (fd >= 0)
        close(fd);
    e2e_wait_grace();
    CHECK(e2e_export(port, "david", "t-e2") == 200 && e2e_chat(port, "bob", "t-e2") == 403,
          "bob's watch closing does not release david's t-e2");

    fd = e2e_watch(port, "david", "t-e1", &st, resp, sizeof(resp));
    CHECK(fd >= 0, "david's own watch on t-e1: 200, held");
    if (fd >= 0)
        close(fd);
    e2e_wait_grace();
    CHECK(e2e_export(port, "david", "t-e1") == 404 && e2e_chat(port, "bob", "t-e1") == 200,
          "david's watch closing releases t-e1 (bob may take the id)");

    CHECK(e2e_chat(port, NULL, "t-e3") == 200, "no principal: chat");
    fd = e2e_watch(port, NULL, "t-e3", &st, resp, sizeof(resp));
    CHECK(fd >= 0, "no principal: watch as before");
    if (fd >= 0)
        close(fd);
    e2e_wait_grace();
    CHECK(e2e_export(port, NULL, "t-e3") == 404, "no principal: its close releases as before");
    lt_stop(pid);
}

int main(void)
{
    const char *model = getenv("PN_TEST_MODEL");

    printf("test_local_serve_principal\n");
    signal(SIGPIPE, SIG_IGN);
    if (!model || !*model)
        model = "models/qwen3-0.6b-Q8_0.gguf";
    /* First: it forks, before any model threads, and before section D reads the
     * watch grace (cached per process), so the child serve's 300 ms grace holds. */
    if (access(model, R_OK) == 0)
        test_watch_over_http(model);
    test_fake_entries();
    if (access(model, R_OK) == 0)
        test_single_user_path(model);
    else
        printf("  SKIP watch_over_http, single_user_path: %s not found\n", model);
    return lt_report("test_local_serve_principal");
}
