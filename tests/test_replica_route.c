/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_replica_route.c — Pantheon Local L4: one served name on several replicas.
 *
 * mr_acquire_session on a hand-built registry (entries marked resident with a
 * dummy model pointer; no model file, no GPU):
 *   - new sessions spread: the least busy replica (live turns + seated sessions;
 *     untracked seats here = every binding holds one);
 *   - a session goes back to its replica on every later turn, its watch (the
 *     base tab id) and its "<base>-<mode>" turns land together;
 *   - released sessions and an unloaded replica free their bindings;
 *   - a single replica, an unknown name and the default name behave as mr_acquire;
 *   - all of the above again through mr_acquire_session_on with no replica named
 *     (no field = exactly the least-busy routing);
 *   - a named replica (replica_pin.h): a new session binds to exactly that copy, a
 *     missing copy is NO_REPLICA with nothing bound, a bound session keeps its
 *     copy, a refused new session's binding (pinned or not) can be undone;
 *   - a full replica (no free session seat) never gets a new session while
 *     another replica has a seat; with every replica full, the least busy;
 *   - X-Pantheon-Replica on JSON, SSE and error responses (socketpair), and the
 *     request value parse (body wins over the header);
 *   - B1: a session bound to a copy where it holds no seat (its watch bound it
 *     while only that copy was loaded, as after a restart) is placed by free
 *     seats on its next turn, like a new session; a seated session never moves;
 *     a refused seatless session fails over to another copy with a seat;
 *   - idle watches (K80 host, 11-15): seatless bindings are not load, so 8 idle
 *     watches bound to replica 1 during a restart do not push every new session
 *     onto replica 2 (sequential and concurrent arrivals split 2/2).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/serve/http_server.h"
#include "../src/serve/model_registry.h"
#include "../src/serve/model_session.h"
#include "../src/serve/replica_affinity.h"
#include "../src/serve/replica_pin.h"

static int g_failed = 0, g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static int g_dummy_model;

/* Seat state the router sees (model_session.c and the session stores behind
 * it are not linked here): a serial listed full has no free session seat. */
static uint64_t g_full_serials[4];

bool ms_session_seat_free(mr_entry *entry)
{
    for (int i = 0; i < 4; i++)
        if (g_full_serials[i] && g_full_serials[i] == entry->serial)
            return false;
    return true;
}

/* Which sessions hold a seat where. Untracked (the default): every binding
 * holds one, the routing the tests above were written against. */
static bool g_seats_tracked;
static struct { const char *sid; uint64_t serial; } g_seats[8];

bool ms_session_holds_seat(mr_entry *entry, const char *sid)
{
    if (!sid || !*sid)
        return false;
    if (!g_seats_tracked)
        return true;
    for (int i = 0; i < 8; i++)
        if (g_seats[i].sid && strcmp(g_seats[i].sid, sid) == 0 &&
            g_seats[i].serial == entry->serial)
            return true;
    return false;
}

/* Sessions seated on a copy, what least-busy routing counts. Untracked: every
 * binding holds a seat, as above. */
int ms_sessions_used(mr_entry *entry)
{
    int n = 0;

    if (!g_seats_tracked)
        return pn_replica_affinity_count(entry->serial);
    for (int i = 0; i < 8; i++)
        n += g_seats[i].sid && g_seats[i].serial == entry->serial;
    return n;
}

/* Slot `slot` resident as `name` on dies [die, die+1], load order `serial`. */
static mr_entry *put(mr_registry *reg, int slot, const char *name, int die, uint64_t serial)
{
    mr_entry *e = &reg->entries[slot];

    memset(e, 0, sizeof(*e));
    e->used = true;
    e->model = (pn_model *)(void *)&g_dummy_model;
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->dies[0] = die;
    e->dies[1] = die + 1;
    e->n_dies = 2;
    e->serial = serial;
    return e;
}

/* 1: route every turn through mr_acquire_session_on with no replica named. */
static int g_route_on;

static mr_entry *acquire(mr_registry *reg, const char *name, const char *sid)
{
    mr_route route = {0};
    mr_entry *e = NULL;

    if (!g_route_on)
        return mr_acquire_session(reg, name, sid);
    e = mr_acquire_session_on(reg, name, sid, &route);
    CHECK(!route.kept_binding, "no replica named: never 'kept another binding'");
    CHECK((e != NULL) == (route.status == MR_ROUTE_OK), "status matches the result");
    return e;
}

static mr_entry *turn(mr_registry *reg, const char *name, const char *sid)
{
    mr_entry *e = acquire(reg, name, sid);
    if (e)
        e->refs--;   /* the turn ended (mr_release without the cond signal) */
    return e;
}

/* A turn naming `replica`; *route_out gets the decision. */
static mr_entry *pinned(mr_registry *reg, const char *name, const char *sid, int replica,
                        mr_route *route_out)
{
    memset(route_out, 0, sizeof(*route_out));
    route_out->replica = replica;
    mr_entry *e = mr_acquire_session_on(reg, name, sid, route_out);
    if (e)
        e->refs--;
    return e;
}

static void test_spread_and_affinity(mr_registry *reg)
{
    printf("spread_and_affinity:\n");
    pn_replica_affinity_reset();
    mr_entry *a = put(reg, 3, "qwen", 0, 1);   /* slot order differs from load order */
    mr_entry *b = put(reg, 1, "qwen", 2, 2);

    CHECK(turn(reg, "qwen", "s1") == a, "first session: the first loaded replica");
    CHECK(turn(reg, "qwen", "s2") == b, "second session: the other replica (least busy)");
    CHECK(turn(reg, "qwen", "s1") == a, "s1 comes back to its replica");
    CHECK(turn(reg, "qwen", "s2") == b, "s2 comes back to its replica");
    CHECK(pn_replica_affinity_count(a->serial) == 1 && pn_replica_affinity_count(b->serial) == 1,
          "one session bound to each");

    /* A live turn counts as busy: a new session avoids it. */
    mr_entry *held = acquire(reg, "qwen", "s1");
    CHECK(held == a, "s1 mid-turn on its replica");
    CHECK(turn(reg, "qwen", "s3") == b, "s3: a is busier (bound + live turn), so b");
    held->refs--;

    /* Watch on the base id + turns on "<base>-<mode>" stay together. */
    CHECK(turn(reg, "qwen", "argus-1-2-chat") == a, "a tab watch lands on a (1 bound vs 2)");
    CHECK(turn(reg, "qwen", "argus-1-2-chat-think") == a, "its turns follow the watch");
    CHECK(turn(reg, "qwen", "argus-1-2-chat-plain") == a, "every mode of that tab");

    /* Release frees the binding: the session may go anywhere next time. */
    pn_replica_affinity_forget("argus-1-2-chat");
    CHECK(pn_replica_affinity_count(a->serial) == 1, "the tab's binding is gone (s1 left)");
    pn_replica_affinity_forget("s3");
    pn_replica_affinity_forget("s2");
    CHECK(turn(reg, "qwen", "s4") == b, "s4 goes to the now-idle b");
}

static void test_unloaded_replica(mr_registry *reg)
{
    printf("unloaded_replica:\n");
    pn_replica_affinity_reset();
    mr_entry *a = put(reg, 0, "orn", 4, 10);
    mr_entry *b = put(reg, 2, "orn", 6, 11);

    CHECK(turn(reg, "orn", "x") == a && turn(reg, "orn", "y") == b, "x on a, y on b");
    /* b unloads: its bindings go, y moves to the replica that is left. */
    pn_replica_affinity_forget_serial(b->serial);
    memset(b, 0, sizeof(*b));
    CHECK(turn(reg, "orn", "y") == a, "y routes to the remaining replica");
    /* A replica that is unloading takes no new sessions. */
    b = put(reg, 2, "orn", 6, 12);
    b->unloading = true;
    CHECK(turn(reg, "orn", "z") == a, "an unloading replica is never picked");
    memset(a, 0, sizeof(*a));
    memset(b, 0, sizeof(*b));
}

static void test_single_and_default(mr_registry *reg)
{
    printf("single_and_default:\n");
    pn_replica_affinity_reset();
    for (int i = 0; i < MR_MAX_MODELS; i++)
        memset(&reg->entries[i], 0, sizeof(reg->entries[i]));
    mr_entry *only = put(reg, 5, "solo", 0, 20);

    CHECK(turn(reg, "solo", NULL) == only, "one replica, no session id");
    CHECK(turn(reg, NULL, "s") == only, "no name: the default model");
    CHECK(turn(reg, "", NULL) == only, "empty name: the default model");
    CHECK(acquire(reg, "nope", "s") == NULL, "unknown name: NULL");
    CHECK(turn(reg, "solo", "") == only && pn_replica_affinity_count(only->serial) == 1,
          "an empty session id binds nothing (only \"s\" is bound)");
}

static void clear_entries(mr_registry *reg)
{
    for (int i = 0; i < MR_MAX_MODELS; i++)
        memset(&reg->entries[i], 0, sizeof(reg->entries[i]));
}

static void test_pinned(mr_registry *reg)
{
    mr_route r;

    printf("pinned:\n");
    pn_replica_affinity_reset();
    clear_entries(reg);
    mr_entry *a = put(reg, 4, "qwen", 0, 30);
    mr_entry *b = put(reg, 2, "qwen", 2, 31);
    a->replica = 1;
    b->replica = 2;

    CHECK(pinned(reg, "qwen", "p1", 2, &r) == b && r.status == MR_ROUTE_OK && r.bound_now,
          "a new session naming 2 binds to copy 2 (least busy would pick 1)");
    CHECK(pn_replica_affinity_count(b->serial) == 1 && b->refs == 0, "bound once, ref paired");
    CHECK(turn(reg, "qwen", "p1") == b, "its next turn without a field goes back to 2");
    CHECK(pinned(reg, "qwen", "p1", 1, &r) == b && r.kept_binding && !r.bound_now,
          "a bound session naming another copy keeps its own (and says so)");
    CHECK(pinned(reg, "qwen", "p1", 2, &r) == b && !r.kept_binding,
          "naming its own copy is not a mismatch");

    CHECK(pinned(reg, "qwen", "p2", 3, &r) == NULL && r.status == MR_ROUTE_NO_REPLICA,
          "a copy that is not loaded: NO_REPLICA, no entry");
    CHECK(r.n_replicas == 2 && r.replicas[0] == 1 && r.replicas[1] == 2,
          "the route lists the loaded copies in load order");
    CHECK(pn_replica_affinity_lookup("p2", &a->serial, 1) == 0 &&
              pn_replica_affinity_count(a->serial) == 0 && a->refs == 0 && b->refs == 0,
          "nothing bound and no reference taken for the missing copy");
    CHECK(pinned(reg, "nope", "p2", 1, &r) == NULL && r.status == MR_ROUTE_NO_MODEL,
          "an unknown name stays NO_MODEL");

    CHECK(pinned(reg, "qwen", NULL, 2, &r) == b && !r.bound_now &&
              pn_replica_affinity_count(b->serial) == 1,
          "no session id: served by the named copy, nothing bound");
    b->unloading = true;
    CHECK(pinned(reg, "qwen", "p3", 2, &r) == NULL && r.status == MR_ROUTE_NO_REPLICA,
          "an unloading copy is not there to pin");
    b->unloading = false;

    /* A pinned new session refused before it ran: the binding goes. */
    CHECK(pinned(reg, "qwen", "p4", 2, &r) == b && r.bound_now, "p4 binds to 2");
    pn_replica_pin_undo(&r, b, "p4");
    CHECK(pinned(reg, "qwen", "p4", 1, &r) == a && r.bound_now,
          "after the undo p4 may pin copy 1 (no stale binding to 2)");
    mr_route least = { .replica = 0, .bound_now = true };
    pn_replica_pin_undo(&least, a, "p4");
    CHECK(pn_replica_affinity_lookup("p4", &a->serial, 1) == 0,
          "an unpinned route's fresh binding is undone too (no stale row)");
    CHECK(turn(reg, "qwen", "p4") == a, "p4 binds afresh on its next turn");
    pn_replica_pin_undo(&r, a, "p4-other");
    CHECK(turn(reg, "qwen", "p4") == a, "undo drops only the exact key");
}

/* A new session goes to a replica with a free seat, not to a full one that
 * merely looks less busy; an unpinned session refused there leaves no row. */
static void test_full_replica(mr_registry *reg)
{
    uint64_t serials[2];

    printf("full_replica:\n");
    pn_replica_affinity_reset();
    clear_entries(reg);
    mr_entry *a = put(reg, 0, "q", 0, 40);
    mr_entry *b = put(reg, 1, "q", 2, 41);
    serials[0] = a->serial;
    serials[1] = b->serial;
    pn_replica_affinity_bind("b-user", b->serial);   /* b is the busier copy */

    g_full_serials[0] = a->serial;                    /* a's one seat is taken */
    CHECK(turn(reg, "q", "new1") == b,
          "replica 1 full, replica 2 has a seat: the new session lands on 2");
    pn_replica_affinity_forget("new1");

    g_full_serials[1] = b->serial;                    /* now both are full */
    mr_route r = {0};
    mr_entry *e = mr_acquire_session_on(reg, "q", "new2", &r);
    CHECK(e == a && r.bound_now, "every replica full: the least busy one (it refuses)");
    pn_replica_pin_undo(&r, e, "new2");               /* the unpinned 429 */
    if (e)
        e->refs--;
    CHECK(pn_replica_affinity_lookup("new2", serials, 2) == 0 &&
              pn_replica_affinity_count(a->serial) == 0,
          "the refused unpinned session leaves no routing row");
    memset(g_full_serials, 0, sizeof(g_full_serials));
}

/* Everything one response wrote to the socket pair's far end. */
static void read_response(int fd, char *out, size_t cap)
{
    ssize_t n = recv(fd, out, cap - 1, MSG_DONTWAIT);
    out[n > 0 ? n : 0] = '\0';
}

static void test_response_header(mr_registry *reg)
{
    int sv[2];
    char got[2048];
    mr_route r = { .replica = 2 };

    printf("response_header:\n");
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socket pair");
    mr_entry *b = &reg->entries[2];
    pn_replica_pin_announce(b);
    hs_send_json(sv[0], 200, "{}");
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 200") && strstr(got, "X-Pantheon-Replica: 2\r\n") &&
              strstr(got, "\r\n\r\n{}"), "JSON reply carries the serving copy");
    hs_begin_sse(sv[0]);
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "text/event-stream") && strstr(got, "X-Pantheon-Replica: 2\r\n"),
          "SSE head carries the serving copy");
    hs_send_error(sv[0], 500, "server_error", "x");
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 500") && strstr(got, "X-Pantheon-Replica: 2\r\n"),
          "an error after routing carries it");
    b->max_sessions = 1;
    pn_replica_pin_send_full(sv[0], &r, b, 7);
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 429") && strstr(got, "Retry-After: 30\r\n") &&
              strstr(got, "X-Pantheon-Replica: 2\r\n") &&
              strstr(got, "replica 2 of model 'qwen' has no free session slot "
                          "(its one session is in use)"),
          "full copy: 429 naming model, copy and reason, with both headers");
    b->max_sessions = 3;
    pn_replica_pin_send_full(sv[0], &r, b, 7);
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "(all 3 sessions are in use)") && !strstr(got, "live sessions"),
          "the count is the seat total, not called live sessions");
    pn_replica_pin_announce(NULL);
    hs_send_json(sv[0], 200, "{}");
    read_response(sv[1], got, sizeof(got));
    CHECK(strstr(got, "HTTP/1.1 200") && !strstr(got, "X-Pantheon-Replica"),
          "cleared: no header on a response outside a model route");
    close(sv[0]);
    close(sv[1]);
}

/* pn_replica_pin_parse on a body (JSON text, may be NULL) and a header. */
static int parse(const char *json, const char *header, int *out)
{
    char err[96];
    jr_value *root = json ? jr_parse(json) : NULL;
    int rc = pn_replica_pin_parse(root, header, out, err, sizeof(err));
    jr_free(root);
    return rc;
}

static void test_parse(void)
{
    int n = -1;

    printf("parse:\n");
    CHECK(parse("{\"replica\":2}", "1", &n) == 0 && n == 2, "the body wins over the header");
    CHECK(parse("{}", " 3 ", &n) == 0 && n == 3, "the header alone");
    CHECK(parse("{\"replica\":null}", "4", &n) == 0 && n == 4, "a null field falls to the header");
    CHECK(parse("{}", NULL, &n) == 0 && n == 0, "neither: 0 (least busy)");
    CHECK(parse(NULL, NULL, &n) == 0 && n == 0, "no body: 0");
    CHECK(parse("{\"replica\":0}", NULL, &n) != 0, "0 is refused");
    CHECK(parse("{\"replica\":-1}", NULL, &n) != 0, "negative is refused");
    CHECK(parse("{\"replica\":2.5}", NULL, &n) != 0, "a fraction is refused");
    CHECK(parse("{\"replica\":\"2\"}", NULL, &n) != 0, "a string is refused");
    CHECK(parse("{}", "abc", &n) != 0 && parse("{}", "0", &n) != 0 &&
              parse("{}", "2x", &n) != 0, "a malformed header is refused");
}


/* A generation turn (place_unseated) with no replica named. */
static mr_entry *gen_turn(mr_registry *reg, const char *sid, mr_route *r)
{
    memset(r, 0, sizeof(*r));
    r->place_unseated = true;
    mr_entry *e = mr_acquire_session_on(reg, "q", sid, r);
    if (e)
        e->refs--;
    return e;
}

static void test_stale_binding(mr_registry *reg)
{
    static const char *sids[6] = { "argus-79316-1", "argus-13372-1", "argus-45473-1",
                                   "argus-45454-1", "argus-58722-1", "argus-58717-1" };
    uint64_t serials[2];
    mr_route r;

    printf("stale_binding (B1):\n");
    pn_replica_affinity_reset();
    clear_entries(reg);
    memset(g_seats, 0, sizeof(g_seats));
    g_seats_tracked = true;

    /* The 11-07 restart log: replica 2 came up first and the old clients'
     * watches reconnected before replica 1 finished loading. */
    mr_entry *r2 = put(reg, 0, "q", 4, 50);
    r2->replica = 2;
    for (int i = 0; i < 6; i++)
        CHECK(turn(reg, "q", sids[i]) == r2, "a watch while only replica 2 is up binds there");
    mr_entry *r1 = put(reg, 1, "q", 0, 51);
    r1->replica = 1;
    serials[0] = r2->serial;
    serials[1] = r1->serial;

    /* The watch path never moves a binding. */
    CHECK(turn(reg, "q", sids[0]) == r2, "a watch reconnect keeps the binding");

    /* Turns: each session holds no seat on replica 2, so it is placed by
     * load like a new session and takes its seat where it lands. */
    int on1 = 0, on2 = 0;
    for (int i = 0; i < 6; i++) {
        mr_entry *e = gen_turn(reg, sids[i], &r);
        CHECK(e != NULL && r.status == MR_ROUTE_OK, "the turn routes");
        if (!e)
            continue;
        CHECK(pn_replica_affinity_lookup(sids[i], serials, 2) == e->serial,
              "the binding follows the turn");
        g_seats[i].sid = sids[i];
        g_seats[i].serial = e->serial;
        on1 += e == r1;
        on2 += e == r2;
    }
    CHECK(on1 == 3 && on2 == 3, "six stale sessions split 3/3, not 6 on replica 2");
    CHECK(gen_turn(reg, sids[1], &r) == r1 && !r.moved, "a moved session stays");

    /* A seated session never moves, however busy its copy. */
    mr_entry *seated_on = NULL;
    for (int i = 0; i < 6; i++)
        if (g_seats[i].serial == r2->serial)
            seated_on = r2;
    pn_replica_affinity_bind("extra-1", r1->serial);
    pn_replica_affinity_bind("extra-2", r1->serial);
    for (int i = 0; i < 6; i++)
        if (g_seats[i].serial == r2->serial)
            CHECK(gen_turn(reg, sids[i], &r) == r2 && !r.moved, "seated on 2: stays on 2");
    CHECK(seated_on == r2, "some session is seated on replica 2");

    /* A bound session holding no seat whose copy is full moves to a seat even
     * when the other copy is busier. */
    g_full_serials[0] = r1->serial;
    pn_replica_affinity_bind("argus-1-1", r1->serial);
    mr_entry *e = gen_turn(reg, "argus-1-1", &r);
    CHECK(e == r2 && r.moved, "seatless on a full copy: placed on the copy with a seat");

    /* A pinned turn is never re-placed. */
    pn_replica_affinity_bind("argus-2-1", r1->serial);
    memset(&r, 0, sizeof(r));
    r.place_unseated = true;
    r.replica = 1;
    e = mr_acquire_session_on(reg, "q", "argus-2-1", &r);
    CHECK(e == r1 && !r.moved, "pinned: stays on its copy");
    if (e)
        e->refs--;

    /* Failover: refused on replica 1, holding no seat there -> replica 2. */
    pn_replica_affinity_bind("argus-3-1", r1->serial);
    memset(&r, 0, sizeof(r));
    int refs_before = r2->refs;
    e = mr_failover_session(reg, r1, "argus-3-1", &r);
    CHECK(e == r2 && r.moved && r2->refs == refs_before + 1,
          "failover takes the other copy with a seat (with a reference)");
    CHECK(pn_replica_affinity_lookup("argus-3-1", serials, 2) == r2->serial,
          "and moves the binding there");
    if (e)
        e->refs--;
    g_full_serials[1] = r2->serial;
    CHECK(mr_failover_session(reg, r1, "argus-3-1", &r) == NULL,
          "every other copy full: no failover (the caller answers 429)");

    mr_entry *all[4];
    CHECK(mr_acquire_replicas(reg, "q", all, 4) == 2 && all[0] == r2 && all[1] == r1,
          "mr_acquire_replicas: every copy, load order");
    r1->refs--;
    r2->refs--;

    memset(g_full_serials, 0, sizeof(g_full_serials));
    g_seats_tracked = false;
}

/* Bind the 8 idle watches of the K80 host's 11-15 restart to `r1`, the only copy up. */
static void bind_idle_watches(mr_registry *reg, mr_entry *r1)
{
    static const char *idle[8] = { "c-cf3b5efeab24-s3", "c-da301f57e136-s3",
                                   "c-b1ea01803530-s3", "c-095e58818182-s2",
                                   "c-c7cc871b92fa-s3", "c-9a91efe970f1-s3",
                                   "c-cc7ed05679c4-s3", "c-7f14739c22d0-s3" };

    for (int i = 0; i < 8; i++)
        CHECK(turn(reg, "q", idle[i]) == r1, "an idle watch while only replica 1 is up");
}

static void test_idle_watches(mr_registry *reg)
{
    static const char *evals[4] = { "c-e3b24386b01c", "c-2151eab0609c",
                                    "c-1d6d28de54d4", "c-4063cb9df208" };
    mr_entry *held[4] = { NULL };
    mr_route r;
    int on1 = 0, on2 = 0;

    printf("idle_watches (K80 host, 11-15):\n");
    pn_replica_affinity_reset();
    clear_entries(reg);
    memset(g_seats, 0, sizeof(g_seats));
    g_seats_tracked = true;

    /* Replica 1 came up first; 8 idle agent hosts' watches reconnected before
     * replica 2 finished loading. They hold no seat and take no turn. */
    mr_entry *r1 = put(reg, 0, "q", 0, 60);
    r1->replica = 1;
    bind_idle_watches(reg, r1);
    mr_entry *r2 = put(reg, 1, "q", 4, 61);
    r2->replica = 2;

    /* 4 new sessions, each seated where its first turn lands. */
    for (int i = 0; i < 4; i++) {
        mr_entry *e = gen_turn(reg, evals[i], &r);
        CHECK(e != NULL, "the turn routes");
        if (!e)
            continue;
        g_seats[i].sid = evals[i];
        g_seats[i].serial = e->serial;
        on1 += e == r1;
        on2 += e == r2;
    }
    CHECK(on1 == 2 && on2 == 2, "4 new sessions split 2/2, not 4 on replica 2");

    /* The same 4 arriving together: first turns in flight, no seat yet. */
    pn_replica_affinity_reset();
    memset(g_seats, 0, sizeof(g_seats));
    r2->model = NULL;   /* replica 2 still loading */
    bind_idle_watches(reg, r1);
    r2->model = (pn_model *)(void *)&g_dummy_model;
    on1 = on2 = 0;
    for (int i = 0; i < 4; i++) {
        memset(&r, 0, sizeof(r));
        r.place_unseated = true;
        held[i] = mr_acquire_session_on(reg, "q", evals[i], &r);
        on1 += held[i] == r1;
        on2 += held[i] == r2;
    }
    CHECK(on1 == 2 && on2 == 2, "4 concurrent first turns split 2/2");
    for (int i = 0; i < 4; i++)
        if (held[i])
            held[i]->refs--;

    g_seats_tracked = false;
}

int main(void)
{
    mr_registry reg;

    printf("test_replica_route\n");
    memset(&reg, 0, sizeof(reg));
    pthread_mutex_init(&reg.lock, NULL);
    test_spread_and_affinity(&reg);
    for (int i = 0; i < MR_MAX_MODELS; i++)
        memset(&reg.entries[i], 0, sizeof(reg.entries[i]));
    test_unloaded_replica(&reg);
    test_single_and_default(&reg);
    printf("again, through mr_acquire_session_on with no replica named:\n");
    g_route_on = 1;
    clear_entries(&reg);
    test_spread_and_affinity(&reg);
    clear_entries(&reg);
    test_unloaded_replica(&reg);
    test_single_and_default(&reg);
    g_route_on = 0;
    test_pinned(&reg);
    test_response_header(&reg);
    test_full_replica(&reg);
    test_stale_binding(&reg);
    test_idle_watches(&reg);
    test_parse();
    pthread_mutex_destroy(&reg.lock);
    printf("test_replica_route: %d checks, %d failures\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
