/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_registry.c — dispatch-layer registry unit tests (Phase G1).
 *
 * Exercises the in-memory instance table in-process (no sockets): register,
 * heartbeat load updates, deduped /v1/models, model-level /registry/fleet,
 * TTL eviction (via reg_sweep with an injected clock), and least-busy pick
 * (max free_slots, min queue_depth, round-robin tiebreak + failover exclude).
 *
 * Links only registry.c + json_reader.c + json_writer.c — no engine/CUDA.
 */

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

/* substring presence helper for JSON assertions. */
static int contains(const char *hay, const char *needle)
{
    return hay && strstr(hay, needle) != NULL;
}

/* count occurrences of a substring. */
static int count_occurrences(const char *hay, const char *needle)
{
    int n = 0;
    const char *p = hay;
    size_t nl = strlen(needle);
    while ((p = strstr(p, needle)) != NULL) { n++; p += nl; }
    return n;
}

static void reg_one(reg_registry *r, const char *model, const char *inst,
                    const char *node, const char *endpoint, int cap)
{
    char body[512];
    snprintf(body, sizeof(body),
             "{\"model_id\":\"%s\",\"instance_id\":\"%s\",\"node_id\":\"%s\","
             "\"endpoint\":\"%s\",\"transport\":\"local\",\"capacity\":%d}",
             model, inst, node, endpoint, cap);
    reg_status st = reg_register_json(r, body);
    CHECK(st == REG_OK, "register should succeed");
}

static void heartbeat(reg_registry *r, const char *inst, int active, int queue,
                      int freeS, int healthy)
{
    char body[256];
    snprintf(body, sizeof(body),
             "{\"instance_id\":\"%s\",\"load\":{\"active_slots\":%d,"
             "\"queue_depth\":%d,\"free_slots\":%d},\"healthy\":%s}",
             inst, active, queue, freeS, healthy ? "true" : "false");
    reg_status st = reg_heartbeat_json(r, body);
    CHECK(st == REG_OK, "heartbeat should succeed");
}

int main(void)
{
    printf("== registry unit tests ==\n");
    reg_registry r;
    CHECK(reg_init(&r) == 0, "reg_init");

    /* --- Register the fleet: 4 model_ids, ornith has TWO instances (replicas). */
    reg_one(&r, "saber-orch",  "saber-1",  "node0", "http://127.0.0.1:8201", 4);
    reg_one(&r, "qwen36-base", "qwen36-1", "node0", "http://127.0.0.1:8202", 4);
    reg_one(&r, "nex-n2-mini", "nex-1",    "node0", "http://127.0.0.1:8203", 4);
    reg_one(&r, "ornith",      "ornith-A", "node0", "http://127.0.0.1:8204", 4);
    reg_one(&r, "ornith",      "ornith-B", "node0", "http://127.0.0.1:8205", 4);

    /* --- /v1/models is deduped by model_id: 4 entries, ornith once, live=2. */
    {
        char *models = reg_models_json(&r);
        CHECK(models != NULL, "models json");
        CHECK(count_occurrences(models, "\"object\":\"model\"") == 4,
              "4 deduped models");
        CHECK(count_occurrences(models, "\"id\":\"ornith\"") == 1,
              "ornith appears once");
        CHECK(contains(models, "\"live\":2"), "ornith live count = 2");
        free(models);
    }

    /* --- /registry/fleet: ornith model shows 2 instances + total_free_slots. */
    heartbeat(&r, "ornith-A", 1, 0, 3, 1);   /* 3 free */
    heartbeat(&r, "ornith-B", 3, 0, 1, 1);   /* 1 free */
    {
        char *fleet = reg_fleet_json(&r);
        CHECK(fleet != NULL, "fleet json");
        CHECK(contains(fleet, "\"model_id\":\"ornith\""), "fleet has ornith");
        CHECK(contains(fleet, "\"instance_id\":\"ornith-A\"") &&
              contains(fleet, "\"instance_id\":\"ornith-B\""),
              "fleet lists both ornith instances");
        CHECK(contains(fleet, "\"total_free_slots\":4"),
              "ornith total_free_slots = 3+1 = 4");
        free(fleet);
    }

    /* --- Least-busy: ornith-A (3 free) beats ornith-B (1 free). */
    {
        reg_instance pick;
        int hits_A = 0, hits_B = 0;
        for (int i = 0; i < 10; i++) {
            CHECK(reg_pick_least_busy(&r, "ornith", NULL, 0, &pick),
                  "pick ornith");
            if (strcmp(pick.instance_id, "ornith-A") == 0) hits_A++;
            else if (strcmp(pick.instance_id, "ornith-B") == 0) hits_B++;
        }
        CHECK(hits_A == 10 && hits_B == 0, "least-busy always picks freer A");
    }

    /* --- Round-robin on ties: make both equal, expect alternation. */
    heartbeat(&r, "ornith-A", 0, 0, 4, 1);
    heartbeat(&r, "ornith-B", 0, 0, 4, 1);
    {
        reg_instance pick;
        int hits_A = 0, hits_B = 0;
        for (int i = 0; i < 10; i++) {
            reg_pick_least_busy(&r, "ornith", NULL, 0, &pick);
            if (strcmp(pick.instance_id, "ornith-A") == 0) hits_A++; else hits_B++;
        }
        CHECK(hits_A > 0 && hits_B > 0, "round-robin spreads across tied replicas");
        CHECK(hits_A + hits_B == 10, "all picks accounted");
    }

    /* --- Failover exclude: excluding A yields B (or vice versa). */
    {
        reg_instance pick;
        const char *ex[1] = { "ornith-A" };
        CHECK(reg_pick_least_busy(&r, "ornith", ex, 1, &pick),
              "pick with A excluded");
        CHECK(strcmp(pick.instance_id, "ornith-B") == 0,
              "excluding A gives B");
        /* Exclude both → no candidate. */
        const char *ex2[2] = { "ornith-A", "ornith-B" };
        CHECK(!reg_pick_least_busy(&r, "ornith", ex2, 2, &pick),
              "excluding both ornith gives no pick");
    }

    /* --- Unknown model → no pick. */
    {
        reg_instance pick;
        CHECK(!reg_pick_least_busy(&r, "nonesuch", NULL, 0, &pick),
              "unknown model → no pick");
    }

    /* --- Unhealthy instances are skipped. */
    {
        heartbeat(&r, "ornith-A", 0, 0, 4, 0);   /* A now unhealthy */
        reg_instance pick;
        int all_B = 1;
        for (int i = 0; i < 5; i++) {
            reg_pick_least_busy(&r, "ornith", NULL, 0, &pick);
            if (strcmp(pick.instance_id, "ornith-B") != 0) all_B = 0;
        }
        CHECK(all_B, "unhealthy A skipped, all picks go to B");
        heartbeat(&r, "ornith-A", 0, 0, 4, 1);   /* restore */
    }

    /* --- TTL eviction: sweep with a future clock evicts all; models empties. */
    {
        /* Set all last_seen to "now"; sweeping at now+TTL+1 should evict all. */
        time_t now = time(NULL);
        int ev = reg_sweep(&r, now);   /* nothing stale yet */
        CHECK(ev == 0, "no eviction when fresh");

        int total = 5;
        ev = reg_sweep(&r, now + RW_TTL_SECONDS + 1);
        CHECK(ev == total, "sweep evicts all stale instances");

        char *models = reg_models_json(&r);
        CHECK(models != NULL, "models json after evict");
        CHECK(count_occurrences(models, "\"object\":\"model\"") == 0,
              "no models after TTL eviction");
        free(models);
    }

    /* --- context_window: advertised, MIN across replicas, and suppressed when
     *     any live replica hasn't reported one. The router may dispatch to ANY
     *     replica, so promising more than the smallest window on offer hands the
     *     client a prefill some replica would then reject. */
    {
        reg_registry w;
        CHECK(reg_init(&w) == 0, "reg_init (window)");

        /* Two replicas of one model_id at DIFFERENT windows. */
        reg_status st = reg_register_json(&w,
            "{\"model_id\":\"twin\",\"instance_id\":\"t-A\",\"node_id\":\"b\","
            "\"endpoint\":\"http://127.0.0.1:9001\",\"capacity\":4,"
            "\"context_window\":131072}");
        CHECK(st == REG_OK, "register with context_window");
        st = reg_register_json(&w,
            "{\"model_id\":\"twin\",\"instance_id\":\"t-B\",\"node_id\":\"b\","
            "\"endpoint\":\"http://127.0.0.1:9002\",\"capacity\":4,"
            "\"context_window\":32000}");
        CHECK(st == REG_OK, "register second replica");

        char *models = reg_models_json(&w);
        CHECK(contains(models, "\"context_window\":32000"),
              "models advertises the SMALLEST replica window");
        CHECK(!contains(models, "\"context_window\":131072"),
              "models never advertises the larger replica window");
        free(models);

        /* A heartbeat carrying a clamped window MOVES the advertised number —
         * the ladder-clamp case reaching the client through the router. */
        st = reg_heartbeat_json(&w,
            "{\"instance_id\":\"t-B\",\"load\":{\"active_slots\":0,"
            "\"queue_depth\":0,\"free_slots\":4,\"context_window\":16000},"
            "\"healthy\":true}");
        CHECK(st == REG_OK, "heartbeat with context_window");
        models = reg_models_json(&w);
        CHECK(contains(models, "\"context_window\":16000"),
              "a heartbeat clamp moves the advertised window");
        free(models);

        /* A heartbeat WITHOUT the field must not blank the known value — an
         * older serve build heartbeating cannot erase what it reported before. */
        heartbeat(&w, "t-B", 0, 0, 4, 1);
        models = reg_models_json(&w);
        CHECK(contains(models, "\"context_window\":16000"),
              "a heartbeat with no window keeps the last known one");
        free(models);

        /* A replica that has NEVER reported one suppresses the field entirely:
         * "undetected" is a state the client handles gracefully, a window it can
         * be denied is not. */
        st = reg_register_json(&w,
            "{\"model_id\":\"twin\",\"instance_id\":\"t-C\",\"node_id\":\"b\","
            "\"endpoint\":\"http://127.0.0.1:9003\",\"capacity\":4}");
        CHECK(st == REG_OK, "register replica with no window");
        models = reg_models_json(&w);
        CHECK(!contains(models, "\"context_window\""),
              "one silent replica suppresses the model-level window");
        free(models);

        char *fleet = reg_fleet_json(&w);
        CHECK(contains(fleet, "\"context_window\":16000") &&
              contains(fleet, "\"context_window\":131072"),
              "fleet reports each instance's own window");
        free(fleet);

        reg_destroy(&w);
    }

    /* --- Bad requests. */
    {
        CHECK(reg_register_json(&r, "{\"instance_id\":\"x\"}") == REG_BAD_REQUEST,
              "register missing model_id/endpoint → 400");
        CHECK(reg_heartbeat_json(&r, "{\"instance_id\":\"ghost\"}") == REG_NOT_FOUND,
              "heartbeat unknown instance → not found");
        CHECK(reg_register_json(&r, "not json") == REG_BAD_REQUEST,
              "malformed JSON → 400");
    }

    reg_destroy(&r);

    if (g_failed) {
        fprintf(stderr, "\n%d CHECK(s) FAILED\n", g_failed);
        return 1;
    }
    printf("ALL REGISTRY UNIT TESTS PASSED\n");
    return 0;
}
