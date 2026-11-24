/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_load_progress.c — Pantheon Local H2n: live load / unload progress.
 *   - a progress slot belongs to the thread that opened it: the engine's
 *     reports on that thread land in it, another thread's are no-ops;
 *   - totals grow to the bytes reported; an unload starts from its totals;
 *   - GET /local/models/loaded shows a model being loaded with
 *     {phase, bytes, total, layer, n_layers, units:[{id, bytes, total}]},
 *     and progress null once the load is over.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include <pthread.h>

#include "../src/core/load_progress.h"
#include "../src/local/loaded_set.h"
#include "../src/local/local_load.h"
#include "../src/local/local_model_routes.h"
#include "../src/serve/json_reader.h"

#define DIR "build/test_local/load_progress"
#define GIB(n) ((uint64_t)(n) << 30)

static void *report_from_other_thread(void *arg)
{
    (void)arg;
    pn_load_progress_set_bytes(0, GIB(9));   /* this thread has no slot: no-op */
    pn_load_progress_set_layer(39, 40);
    return NULL;
}

static void test_slots(void)
{
    pn_load_progress p;
    pthread_t t;
    int dies[2] = { 0, 1 }, active = 0;
    uint64_t held[2] = { GIB(5), GIB(6) };

    printf("slots:\n");
    CHECK(pn_load_progress_find("qwen", 1, &p) != 0, "nothing open");
    CHECK(pn_load_progress_begin("qwen", 1, PN_LOAD_PHASE_LOADING, dies, NULL, 2) == 0, "open");
    pn_load_progress_set_total(0, GIB(5));
    pn_load_progress_set_bytes(0, GIB(2));
    pn_load_progress_set_bytes(1, GIB(7));      /* more than its total (0): total follows */
    pn_load_progress_set_layer(12, 40);
    pthread_create(&t, NULL, report_from_other_thread, NULL);
    pthread_join(t, NULL);
    CHECK(pn_load_progress_find("qwen", 1, &p) == 0 && p.phase == PN_LOAD_PHASE_LOADING,
          "found by name + replica");
    CHECK(p.n_units == 2 && p.units[0].bytes == GIB(2) && p.units[0].total == GIB(5),
          "die 0: 2 of 5 GiB");
    CHECK(p.units[1].total == GIB(7), "die 1: total raised to its bytes");
    CHECK(p.layer == 12 && p.n_layers == 40, "layer 12 / 40 (the other thread's report ignored)");
    CHECK(pn_load_progress_die_bytes(0, &active) == GIB(2) && active, "die 0 bytes for the tiles");
    CHECK(pn_load_progress_die_bytes(3, &active) == 0 && !active, "die 3 untouched");
    pn_load_progress_end();
    CHECK(pn_load_progress_find("qwen", 1, &p) != 0, "closed");

    CHECK(pn_load_progress_begin("qwen", 2, PN_LOAD_PHASE_UNLOADING, dies, held, 2) == 0,
          "unload slot");
    pn_load_progress_set_bytes(0, GIB(1));
    CHECK(pn_load_progress_find("qwen", 2, &p) == 0 && p.units[0].bytes == GIB(1) &&
          p.units[0].total == GIB(5) && p.units[1].total == GIB(6),
          "unload drains from what each die held");
    pn_load_progress_end();
}

static void test_loaded_api(void)
{
    pn_local_ctx ctx;
    pn_loaded_entry e;
    json_writer jw;
    int dies[2] = { 0, 1 };

    printf("loaded_api:\n");
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    memset(&ctx, 0, sizeof(ctx));
    ctx.loaded = pn_loaded_set_open(DIR "/loaded.json");
    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "qwen");
    snprintf(e.profile_id, sizeof(e.profile_id), "k80-pair");
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), "k80-gpu");
    e.hardware.count = 2;
    e.units[0] = 0;
    e.units[1] = 1;
    e.n_units = 2;
    e.replica = 1;
    e.ctx_cap = 40960;
    CHECK(ctx.loaded && pn_loaded_set_add(ctx.loaded, &e) == 0, "remembered (a reload's turn)");

    pn_load_progress_begin("qwen", 1, PN_LOAD_PHASE_LOADING, dies, NULL, 2);
    pn_load_progress_set_total(0, GIB(9));
    pn_load_progress_set_total(1, GIB(9));
    pn_load_progress_set_bytes(0, GIB(9));
    pn_load_progress_set_bytes(1, GIB(3));
    pn_load_progress_set_layer(26, 40);
    jw_init(&jw);
    pn_local_write_loaded(&jw, &ctx);
    jr_value *root = jr_parse(jw.data);
    const jr_value *row = jr_array_get(jr_object_get(root, "models"), 0);
    const jr_value *pr = row ? jr_object_get(row, "progress") : NULL;
    if (pr && jr_kind(pr) != JR_OBJECT)
        pr = NULL;
    CHECK(pr != NULL, "a progress object while loading");
    CHECK(row && strcmp(jr_as_string(jr_object_get(row, "state")), "loading") == 0,
          "state loading");
    CHECK(pr && strcmp(jr_as_string(jr_object_get(pr, "phase")), "loading") == 0 &&
          jr_as_number(jr_object_get(pr, "bytes"), 0) == (double)GIB(12) &&
          jr_as_number(jr_object_get(pr, "total"), 0) == (double)GIB(18) &&
          jr_as_number(jr_object_get(pr, "layer"), 0) == 26 &&
          jr_as_number(jr_object_get(pr, "n_layers"), 0) == 40,
          "progress: 12 / 18 GiB, layer 26 / 40");
    const jr_value *u1 = pr ? jr_array_get(jr_object_get(pr, "units"), 1) : NULL;
    CHECK(u1 && jr_as_number(jr_object_get(u1, "id"), -1) == 1 &&
          jr_as_number(jr_object_get(u1, "bytes"), 0) == (double)GIB(3),
          "per unit: unit 1 at 3 GiB");
    jr_free(root);
    jw_free(&jw);
    pn_load_progress_end();

    /* An unload: the registry row is gone before the model is freed. */
    uint64_t held[2] = { GIB(9), GIB(9) };
    pn_load_progress_begin("qwen", 1, PN_LOAD_PHASE_UNLOADING, dies, held, 2);
    pn_load_progress_set_bytes(0, GIB(2));
    jw_init(&jw);
    pn_local_write_loaded(&jw, &ctx);
    root = jr_parse(jw.data);
    row = jr_array_get(jr_object_get(root, "models"), 0);
    pr = row ? jr_object_get(row, "progress") : NULL;
    CHECK(row && strcmp(jr_as_string(jr_object_get(row, "state")), "unloading") == 0 &&
          pr && jr_kind(pr) == JR_OBJECT &&
          strcmp(jr_as_string(jr_object_get(pr, "phase")), "unloading") == 0,
          "an unload in progress reads unloading");
    jr_free(root);
    jw_free(&jw);
    pn_load_progress_end();

    jw_init(&jw);
    pn_local_write_loaded(&jw, &ctx);
    root = jr_parse(jw.data);
    row = jr_array_get(jr_object_get(root, "models"), 0);
    CHECK(row && jr_kind(jr_object_get(row, "progress")) == JR_NULL &&
          strcmp(jr_as_string(jr_object_get(row, "state")), "not_loaded") == 0,
          "after: progress null");
    jr_free(root);
    jw_free(&jw);
    pn_loaded_set_close(ctx.loaded);
}

int main(void)
{
    test_slots();
    test_loaded_api();
    return lt_report("test_local_load_progress");
}
