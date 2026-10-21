/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_session_affinity.c — dispatch-layer session-affinity unit tests (PC0.2).
 *
 * Exercises the session -> instance map in-process (no sockets): record/lookup,
 * refresh-on-hit, LRU eviction when full, forget-on-deregister/evict, plus the
 * registry helpers affinity depends on (reg_find_live model+liveness gating,
 * reg_sweep_collect / reg_deregister_json_collect reporting evicted ids).
 *
 * Links only session_affinity.c + registry.c + json_reader.c + json_writer.c —
 * no engine/CUDA. Host-side, deterministic.
 */

#include "../src/serve/session_affinity.h"
#include "../src/serve/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static void reg_one(reg_registry *r, const char *model, const char *inst,
                    const char *endpoint, int cap)
{
    char body[512];
    snprintf(body, sizeof(body),
             "{\"model_id\":\"%s\",\"instance_id\":\"%s\",\"node_id\":\"b\","
             "\"endpoint\":\"%s\",\"transport\":\"local\",\"capacity\":%d}",
             model, inst, endpoint, cap);
    reg_status st = reg_register_json(r, body);
    CHECK(st == REG_OK, "register should succeed");
}

/* --- session_affinity map: record / lookup / refresh / LRU / forget --------- */

static void test_map_basic(void)
{
    printf("map: record + lookup + miss\n");
    session_affinity sa;
    CHECK(sa_init(&sa) == 0, "sa_init");

    char out[RW_INSTANCE_ID_MAX];
    CHECK(!sa_lookup(&sa, "sess-1", out, sizeof(out)), "unknown session → miss");
    CHECK(!sa_lookup(&sa, NULL, out, sizeof(out)), "NULL session → miss");
    CHECK(!sa_lookup(&sa, "", out, sizeof(out)), "empty session → miss");

    sa_record(&sa, "sess-1", "ornith-A");
    CHECK(sa_lookup(&sa, "sess-1", out, sizeof(out)), "recorded session → hit");
    CHECK(strcmp(out, "ornith-A") == 0, "hit returns the bound instance");

    /* Re-record overwrites the binding for the same session. */
    sa_record(&sa, "sess-1", "ornith-B");
    CHECK(sa_lookup(&sa, "sess-1", out, sizeof(out)), "still a hit after rebind");
    CHECK(strcmp(out, "ornith-B") == 0, "rebind overwrites the instance");

    /* No-ops: empty ids must not create bindings. */
    sa_record(&sa, "", "ornith-A");
    sa_record(&sa, "sess-x", "");
    CHECK(!sa_lookup(&sa, "sess-x", out, sizeof(out)),
          "record with empty instance is a no-op");

    sa_destroy(&sa);
}

static void test_map_lru_eviction(void)
{
    printf("map: LRU eviction when full\n");
    session_affinity sa;
    CHECK(sa_init(&sa) == 0, "sa_init");

    char sid[32], out[RW_INSTANCE_ID_MAX];
    /* Fill the table exactly. */
    for (int i = 0; i < SA_MAX_SESSIONS; i++) {
        snprintf(sid, sizeof(sid), "s%d", i);
        sa_record(&sa, sid, "inst-A");
    }
    /* Touch s1 so it is NOT the LRU, then insert one more → s0 (LRU) is evicted. */
    CHECK(sa_lookup(&sa, "s1", out, sizeof(out)), "s1 present before overflow");
    sa_record(&sa, "overflow", "inst-B");

    CHECK(!sa_lookup(&sa, "s0", out, sizeof(out)), "LRU (s0) evicted on overflow");
    CHECK(sa_lookup(&sa, "s1", out, sizeof(out)), "recently-used s1 survives");
    CHECK(sa_lookup(&sa, "overflow", out, sizeof(out)), "new session admitted");

    sa_destroy(&sa);
}

static void test_map_forget_instance(void)
{
    printf("map: forget every binding for a dead instance\n");
    session_affinity sa;
    CHECK(sa_init(&sa) == 0, "sa_init");

    char out[RW_INSTANCE_ID_MAX];
    sa_record(&sa, "s-a1", "inst-A");
    sa_record(&sa, "s-a2", "inst-A");
    sa_record(&sa, "s-b1", "inst-B");

    int removed = sa_forget_instance(&sa, "inst-A");
    CHECK(removed == 2, "forget drops both bindings on inst-A");
    CHECK(!sa_lookup(&sa, "s-a1", out, sizeof(out)), "s-a1 gone");
    CHECK(!sa_lookup(&sa, "s-a2", out, sizeof(out)), "s-a2 gone");
    CHECK(sa_lookup(&sa, "s-b1", out, sizeof(out)), "inst-B binding untouched");

    CHECK(sa_forget_instance(&sa, "nonesuch") == 0, "forget unknown → 0");
    CHECK(sa_forget_instance(&sa, NULL) == 0, "forget NULL → 0");

    sa_destroy(&sa);
}

/* --- registry helpers affinity relies on ----------------------------------- */

static void hb(reg_registry *r, const char *inst, int freeS, int healthy)
{
    char body[256];
    snprintf(body, sizeof(body),
             "{\"instance_id\":\"%s\",\"load\":{\"active_slots\":0,"
             "\"queue_depth\":0,\"free_slots\":%d},\"healthy\":%s}",
             inst, freeS, healthy ? "true" : "false");
    reg_heartbeat_json(r, body);
}

static void test_reg_find_live(void)
{
    printf("registry: reg_find_live model + liveness gating\n");
    reg_registry r;
    CHECK(reg_init(&r) == 0, "reg_init");
    reg_one(&r, "ornith", "ornith-A", "http://127.0.0.1:8201", 4);
    reg_one(&r, "ornith", "ornith-B", "http://127.0.0.1:8202", 4);

    reg_instance out;
    CHECK(reg_find_live(&r, "ornith", "ornith-A", &out), "A is live for ornith");
    CHECK(strcmp(out.endpoint, "http://127.0.0.1:8201") == 0, "returns A's endpoint");
    CHECK(!reg_find_live(&r, "ornith", "ghost", &out), "unknown instance → false");
    CHECK(!reg_find_live(&r, "other", "ornith-A", &out),
          "right instance wrong model → false");

    hb(&r, "ornith-A", 4, 0);   /* A unhealthy */
    CHECK(!reg_find_live(&r, "ornith", "ornith-A", &out),
          "unhealthy instance → false (falls back to least-busy)");

    reg_destroy(&r);
}

static void test_reg_collect_variants(void)
{
    printf("registry: deregister/sweep report evicted ids\n");
    reg_registry r;
    CHECK(reg_init(&r) == 0, "reg_init");
    reg_one(&r, "ornith", "ornith-A", "http://127.0.0.1:8201", 4);
    reg_one(&r, "ornith", "ornith-B", "http://127.0.0.1:8202", 4);

    char removed[RW_INSTANCE_ID_MAX];
    reg_status st = reg_deregister_json_collect(
        &r, "{\"instance_id\":\"ornith-A\"}", removed, sizeof(removed));
    CHECK(st == REG_OK, "deregister A ok");
    CHECK(strcmp(removed, "ornith-A") == 0, "collect reports the removed id");

    /* Sweep the remaining one with a future clock; it must be reported. */
    char evicted[8][RW_INSTANCE_ID_MAX];
    int reported = 0;
    time_t now = time(NULL);
    int ev = reg_sweep_collect(&r, now + RW_TTL_SECONDS + 1, evicted, 8, &reported);
    CHECK(ev == 1, "one instance swept");
    CHECK(reported == 1, "one evicted id reported");
    CHECK(strcmp(evicted[0], "ornith-B") == 0, "reported id is ornith-B");

    reg_destroy(&r);
}

/* --- integration: affinity across the pick path ---------------------------- */

/* Mirror the router's pick logic: session bound + live → reuse; else least-busy
 * and record. Proves same-session stickiness and dead-instance re-route. */
static void test_affinity_pick_flow(void)
{
    printf("affinity: same session sticks; dead instance re-routes\n");
    reg_registry r;
    session_affinity sa;
    CHECK(reg_init(&r) == 0, "reg_init");
    CHECK(sa_init(&sa) == 0, "sa_init");
    reg_one(&r, "ornith", "ornith-A", "http://127.0.0.1:8201", 4);
    reg_one(&r, "ornith", "ornith-B", "http://127.0.0.1:8202", 4);

    char bound[RW_INSTANCE_ID_MAX];
    reg_instance pick;

    /* Turn 1: new session → least-busy, recorded. */
    CHECK(!sa_lookup(&sa, "chat-1", bound, sizeof(bound)), "turn1 no binding yet");
    CHECK(reg_pick_least_busy(&r, "ornith", NULL, 0, &pick), "turn1 least-busy");
    sa_record(&sa, "chat-1", pick.instance_id);
    char first_inst[RW_INSTANCE_ID_MAX];
    snprintf(first_inst, sizeof(first_inst), "%s", pick.instance_id);

    /* Turn 2..5: same session → SAME instance (KV reuse). */
    for (int t = 0; t < 4; t++) {
        CHECK(sa_lookup(&sa, "chat-1", bound, sizeof(bound)), "turn2+ has a binding");
        CHECK(reg_find_live(&r, "ornith", bound, &pick), "bound instance live");
        CHECK(strcmp(pick.instance_id, first_inst) == 0,
              "same session lands on the same instance");
    }

    /* The bound instance dies (deregister) and its binding is forgotten. */
    char removed[RW_INSTANCE_ID_MAX];
    char body[128];
    snprintf(body, sizeof(body), "{\"instance_id\":\"%s\"}", first_inst);
    reg_deregister_json_collect(&r, body, removed, sizeof(removed));
    sa_forget_instance(&sa, removed);

    /* Next turn: no binding → least-busy picks the surviving instance, recorded. */
    CHECK(!sa_lookup(&sa, "chat-1", bound, sizeof(bound)),
          "binding cleared after its instance died");
    CHECK(reg_pick_least_busy(&r, "ornith", NULL, 0, &pick), "re-route least-busy");
    CHECK(strcmp(pick.instance_id, first_inst) != 0, "re-routed to a live instance");
    sa_record(&sa, "chat-1", pick.instance_id);
    CHECK(sa_lookup(&sa, "chat-1", bound, sizeof(bound)), "re-route recorded");

    sa_destroy(&sa);
    reg_destroy(&r);
}

int main(void)
{
    test_map_basic();
    test_map_lru_eviction();
    test_map_forget_instance();
    test_reg_find_live();
    test_reg_collect_variants();
    test_affinity_pick_flow();

    if (g_failed) {
        fprintf(stderr, "\n%d CHECK(s) FAILED\n", g_failed);
        return 1;
    }
    printf("ALL SESSION-AFFINITY UNIT TESTS PASSED\n");
    return 0;
}
