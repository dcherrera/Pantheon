/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_hw_report.c — Pantheon Local B1/B5: the card → unit tree.
 * Logical dies stand in for hardware: 16 dies as K80 = 8 cards x 2 GPU units;
 * 1 die as p150 = 1 card x 1 unit; no dies = 1 cpu card. A model placed on a
 * die shows on that unit, and the JSON has the documented shape.
 * A fake device reader stands in for cuMemGetInfo / the TT allocator: memory
 * another program holds shows as theirs, such a unit is "in_use_other" (not
 * free), and the fit refuses it.
 * Fast reads: with the background poller running, a report reads no device
 * (instant, from the snapshot, with its age); a kick + ensure_fresh brings a
 * new reading. A load in progress counts its live bytes as ours.
 */

#include "local_test_support.h"

#include <sys/time.h>

#include "../src/accel/die_manager.h"
#include "../src/core/load_progress.h"
#include "../src/local/hw_mem_poller.h"
#include "../src/local/hw_report.h"
#include "../src/local/unit_place.h"

#define GIB(n) ((uint64_t)(n) << 30)

static void test_k80_bank(void)
{
    pn_hw_report hw;
    mr_info models[1];

    printf("k80_bank:\n");
    CHECK(pn_dm_init_logical(16, GIB(11)) == 0, "16 logical dies");
    memset(models, 0, sizeof(models));
    snprintf(models[0].name, sizeof(models[0].name), "qwen");
    models[0].dies[0] = 6;
    models[0].dies[1] = 7;
    models[0].n_dies = 2;
    CHECK(pn_dm_bind(models[0].dies, 2, "qwen", GIB(16)) == 0, "bind qwen on 6,7");
    pn_hw_report_build(PN_HW_FAMILY_K80, models, 1, GIB(8), &hw);
    CHECK(hw.n_cards == 8, "8 cards");
    CHECK(hw.n_units == 16, "16 units");
    CHECK(strcmp(hw.cards[0].type, "k80") == 0, "card type k80");
    CHECK(hw.cards[3].n_units == 2 && hw.cards[3].first_unit == 6, "card 3 = units 6,7");
    CHECK(strcmp(hw.units[6].type, "k80-gpu") == 0, "unit type k80-gpu");
    CHECK(strcmp(hw.units[6].model, "qwen") == 0 && hw.units[7].n_models == 1,
          "qwen shows on units 6 and 7");
    CHECK(hw.units[6].mem_free == GIB(3), "unit 6 free = 11 - 8 GiB");
    CHECK(hw.units[0].n_models == 0 && hw.units[0].mem_free == GIB(11), "unit 0 free");

    json_writer jw;
    jw_init(&jw);
    jw_raw(&jw, "{");
    pn_hw_report_write_cards(&jw, &hw);
    jw_raw(&jw, "}");
    jr_value *root = jr_parse(jw.data);
    const jr_value *cards = jr_object_get(root, "cards");
    CHECK(jr_array_len(cards) == 8, "json: 8 cards");
    const jr_value *u6 = jr_array_get(jr_object_get(jr_array_get(cards, 3), "units"), 0);
    CHECK(u6 && strcmp(jr_as_string(jr_object_get(u6, "model")), "qwen") == 0,
          "json: card 3 unit 0 runs qwen");
    CHECK(jr_kind(jr_object_get(u6, "temperature_c")) == JR_NULL, "json: temperature null");
    jr_free(root);
    jw_free(&jw);

    char summary[32];
    pn_hw_report_card_summary(&hw, summary, sizeof(summary));
    CHECK(strcmp(summary, "k80x8") == 0, "summary k80x8");
    pn_dm_shutdown();
}

static void test_p150(void)
{
    pn_hw_report hw;

    printf("p150:\n");
    CHECK(pn_dm_init_logical(1, GIB(31)) == 0, "1 logical die");
    pn_hw_report_build(PN_HW_FAMILY_P150, NULL, 0, GIB(8), &hw);
    CHECK(hw.n_cards == 1 && hw.n_units == 1, "1 card, 1 unit");
    CHECK(strcmp(hw.cards[0].type, "p150") == 0 && strcmp(hw.units[0].type, "p150") == 0,
          "types p150");
    CHECK(hw.units[0].mem_total == GIB(31), "unit memory = card DRAM budget");
    pn_dm_shutdown();
}

static void test_cpu_only(void)
{
    pn_hw_report hw;
    mr_info m;

    printf("cpu_only:\n");
    memset(&m, 0, sizeof(m));
    snprintf(m.name, sizeof(m.name), "small");
    pn_hw_report_build(pn_hw_family_detect(), &m, 1, GIB(8), &hw);
    CHECK(hw.family == PN_HW_FAMILY_CPU, "no dies -> cpu family");
    CHECK(hw.n_cards == 1 && hw.n_units == 1 && strcmp(hw.units[0].type, "cpu") == 0,
          "one cpu unit");
    CHECK(strcmp(hw.units[0].model, "small") == 0, "CPU-resident model on the cpu unit");
    CHECK(hw.units[0].mem_total == GIB(8), "cpu unit memory = host RAM");
}

/* The K80 bank as David saw it: our model on GPU 0, GPU 1 idle, GPUs 2 and 3
 * held by another process (9.2 GB each, as nvidia-smi showed). */
static int fake_k80(pn_hw_family family, int die, pn_hw_dev_mem *out)
{
    (void)family;
    memset(out, 0, sizeof(*out));
    out->total = GIB(11);
    if (die == 0)
        out->used = GIB(6) + (100ULL << 20);     /* our 6 GiB + our context */
    else if (die == 1)
        out->used = 60ULL << 20;                 /* the reading's own context */
    else
        out->used = 9200000000ULL;               /* another program */
    return 0;
}

static void test_other_program(void)
{
    pn_hw_report hw;
    mr_info models[1];

    printf("other_program:\n");
    CHECK(pn_dm_init_logical(4, GIB(10)) == 0, "4 logical dies, 10 GiB budget");
    memset(models, 0, sizeof(models));
    snprintf(models[0].name, sizeof(models[0].name), "qwen");
    models[0].dies[0] = 0;
    models[0].n_dies = 1;
    CHECK(pn_dm_bind(models[0].dies, 1, "qwen", GIB(6)) == 0, "bind qwen on 0");
    pn_hw_report_set_mem_reader(fake_k80);
    pn_hw_report_build(PN_HW_FAMILY_K80, models, 1, GIB(8), &hw);

    const pn_hw_unit *u0 = &hw.units[0], *u1 = &hw.units[1], *u2 = &hw.units[2];
    CHECK(u2->mem_live && u2->mem_device_total == GIB(11), "unit 2 read from the device");
    CHECK(u2->mem_used_other > 8500000000ULL && u2->mem_used_self == 0,
          "unit 2: the other program's 9.2 GB is theirs");
    CHECK(u2->in_use_other && hw.units[3].in_use_other, "units 2, 3: in use (another program)");
    CHECK(u2->mem_free == u2->mem_total - u2->mem_used_other, "unit 2 free = budget - theirs");
    CHECK(!pn_hw_unit_available(u2), "unit 2 is not free");
    CHECK(u0->mem_used_self == GIB(6) && u0->mem_used_other == 0 && !u0->in_use_other,
          "unit 0: our model is ours, no other program");
    CHECK(u1->mem_used_other == 0 && pn_hw_unit_available(u1) && u1->mem_free == GIB(10),
          "unit 1: idle, free");

    pn_unit_request r;
    int ids[4], n = 0;
    char why[128];
    memset(&r, 0, sizeof(r));
    snprintf(r.unit_type, sizeof(r.unit_type), "k80-gpu");
    r.count = 2;
    r.same_card = true;
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1 &&
          strcmp(why, "needs 2 free k80-gpu on one card, you have 1") == 0,
          "fit refuses card 1: another program holds it");
    r.count = 1;
    r.same_card = false;
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 1 && ids[0] == 1,
          "a one-GPU load goes to unit 1, not 2 or 3");
    pn_dm_die d;
    CHECK(pn_dm_get(2, &d) == 0 && d.vram_used > GIB(8) && d.vram_bound == 0,
          "die manager sees the other program on die 2 too");

    json_writer jw;
    jw_init(&jw);
    jw_raw(&jw, "{");
    pn_hw_report_write_cards(&jw, &hw);
    jw_raw(&jw, "}");
    jr_value *root = jr_parse(jw.data);
    const jr_value *cards = jr_object_get(root, "cards");
    const jr_value *j2 = jr_array_get(jr_object_get(jr_array_get(cards, 1), "units"), 0);
    const jr_value *j0 = jr_array_get(jr_object_get(jr_array_get(cards, 0), "units"), 0);
    const jr_value *st2 = j2 ? jr_object_get(j2, "status") : NULL;
    const jr_value *st0 = j0 ? jr_object_get(j0, "status") : NULL;
    CHECK(st2 && strcmp(jr_as_string(st2), "in_use_other") == 0, "json: unit 2 in_use_other");
    CHECK(st0 && strcmp(jr_as_string(st0), "loaded") == 0, "json: unit 0 loaded");
    CHECK(j2 && jr_as_number(jr_object_get(j2, "mem_used_other"), 0) > 8.5e9 &&
          jr_as_number(jr_object_get(j2, "mem_used_self"), -1) == 0 &&
          jr_as_number(jr_object_get(j2, "mem_device_total"), 0) == (double)GIB(11) &&
          jr_as_bool(jr_object_get(j2, "mem_live"), false),
          "json: total, ours, theirs, live");
    jr_free(root);
    jw_free(&jw);
    pn_hw_report_set_mem_reader(NULL);
    pn_dm_shutdown();
}

/* A p150 as the TT allocator reports it: our 20 GiB, no cross-process view. */
static int fake_p150(pn_hw_family family, int die, pn_hw_dev_mem *out)
{
    (void)family;
    (void)die;
    memset(out, 0, sizeof(*out));
    out->total = GIB(32);
    out->used = GIB(20);
    out->used_self = GIB(20);
    out->self_known = true;
    return 0;
}

static void test_p150_allocator(void)
{
    pn_hw_report hw;

    printf("p150_allocator:\n");
    CHECK(pn_dm_init_logical(1, GIB(31)) == 0, "1 logical die");
    pn_hw_report_set_mem_reader(fake_p150);
    pn_hw_report_build(PN_HW_FAMILY_P150, NULL, 0, GIB(8), &hw);
    const pn_hw_unit *u = &hw.units[0];
    CHECK(u->mem_live && u->mem_used_self == GIB(20) && u->mem_used_other == 0,
          "allocator bytes are ours");
    CHECK(u->mem_free == GIB(11) && !u->in_use_other, "free = budget - ours");
    pn_hw_report_set_mem_reader(NULL);
    pn_dm_shutdown();
}

/* A slow device: 100 ms a read (a K80 transient context is ~285 ms), counted. */
static int g_slow_reads;
static uint64_t g_slow_used = GIB(1);

static int fake_slow(pn_hw_family family, int die, pn_hw_dev_mem *out)
{
    (void)family;
    (void)die;
    usleep(100000);
    __atomic_add_fetch(&g_slow_reads, 1, __ATOMIC_SEQ_CST);
    memset(out, 0, sizeof(*out));
    out->total = GIB(11);
    out->used = __atomic_load_n(&g_slow_used, __ATOMIC_SEQ_CST);
    return 0;
}

static double ms_since(const struct timeval *t0)
{
    struct timeval t1;
    gettimeofday(&t1, NULL);
    return (double)(t1.tv_sec - t0->tv_sec) * 1e3 + (double)(t1.tv_usec - t0->tv_usec) / 1e3;
}

static void test_snapshot_reads_fast(void)
{
    pn_hw_report hw;
    struct timeval t0;

    printf("snapshot_reads_fast:\n");
    CHECK(pn_dm_init_logical(8, GIB(10)) == 0, "8 logical dies");
    pn_hw_report_set_mem_reader(fake_slow);
    gettimeofday(&t0, NULL);
    pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    CHECK(ms_since(&t0) >= 700, "without the poller a report reads all 8 dies (~800 ms)");
    CHECK(pn_hw_mem_poller_start(PN_HW_FAMILY_K80, 8, pn_hw_report_read_device, 3000) == 0,
          "poller started (first pass done)");
    int reads = __atomic_load_n(&g_slow_reads, __ATOMIC_SEQ_CST);
    gettimeofday(&t0, NULL);
    for (int i = 0; i < 20; i++)
        pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    double ms = ms_since(&t0);
    CHECK(ms < 50, "20 reports from the snapshot in < 50 ms");
    CHECK(__atomic_load_n(&g_slow_reads, __ATOMIC_SEQ_CST) == reads, "no device read per report");
    CHECK(hw.units[3].mem_live && hw.units[3].mem_used_other > 0 && hw.units[3].mem_age_ms < 3000,
          "unit read from the snapshot, with its age");

    __atomic_store_n(&g_slow_used, GIB(9), __ATOMIC_SEQ_CST);   /* another program arrives */
    pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    CHECK(!hw.units[3].in_use_other, "the snapshot is not re-read per request");
    gettimeofday(&t0, NULL);
    CHECK(pn_hw_mem_poller_ensure_fresh(10000, 5000) == 0 && ms_since(&t0) < 20,
          "fresh enough and no load since: no wait");
    pn_hw_mem_poller_kick();   /* what a load / unload does */
    CHECK(pn_hw_mem_poller_ensure_fresh(10000, 5000) == 0, "after a kick: waits for a new pass");
    pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    CHECK(hw.units[3].in_use_other, "the new pass shows the other program");
    uint64_t passes = pn_hw_mem_poller_passes();
    usleep(1200000);
    CHECK(pn_hw_mem_poller_passes() == passes, "no pass sooner than the interval");
    pn_hw_mem_poller_stop();
    CHECK(!pn_hw_mem_poller_running(), "stopped");
    __atomic_store_n(&g_slow_used, GIB(1), __ATOMIC_SEQ_CST);
    pn_hw_report_set_mem_reader(NULL);
    pn_dm_shutdown();
}

/* H2n: the bars fill live — a load in progress reports its bytes on its dies,
 * and the report counts them as ours before any device reading shows them. */
static void test_live_load_bytes(void)
{
    pn_hw_report hw;
    int dies[2] = { 0, 1 };

    printf("live_load_bytes:\n");
    CHECK(pn_dm_init_logical(4, GIB(10)) == 0, "4 logical dies");
    pn_hw_report_set_mem_reader(fake_k80);   /* die 1 reads idle */
    CHECK(pn_load_progress_begin("qwen", 1, PN_LOAD_PHASE_LOADING, dies, NULL, 2) == 0,
          "progress slot open");
    pn_load_progress_set_bytes(1, GIB(4));
    pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    CHECK(hw.units[1].mem_used_self == GIB(4) && hw.units[1].mem_used_other == 0 &&
          hw.units[1].mem_free == GIB(6), "unit 1: 4 GiB uploaded so far count as ours");
    pn_load_progress_end();
    pn_hw_report_build(PN_HW_FAMILY_K80, NULL, 0, GIB(8), &hw);
    CHECK(hw.units[1].mem_used_self == 0, "done: back to the reading");
    pn_hw_report_set_mem_reader(NULL);
    pn_dm_shutdown();
}

int main(void)
{
    test_k80_bank();
    test_p150();
    test_cpu_only();
    test_other_program();
    test_p150_allocator();
    test_snapshot_reads_fast();
    test_live_load_bytes();
    return lt_report("test_local_hw_report");
}
