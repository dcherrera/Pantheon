/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_load_guard.c — no silent CPU fallback (core/load_guard.h).
 *
 *   K80 bank 2026-10-29: Ornith 9B Q8_0 on one GPU at a context the die refused
 *   fell back to the CPU forward; its 9.8 GB of weights thrashed the 8 GB host
 *   until killed, and --mtp then said "no MTP head on this model".
 *
 *   - A fake accelerator that refuses the upload makes the load FAIL with its
 *     reason: no model comes back, so no CPU forward can start; nothing stays
 *     charged to the host.
 *   - The same through the registry (auto-placed on a die, --mtp asked): the
 *     error is the accelerator's reason, not "no MTP head", and the die is free.
 *   - No die with room is refused, not placed on the CPU.
 *   - An explicit CPU load still runs; one whose weights do not fit host RAM is
 *     refused before it starts.
 *   - --mtp on a CPU load says so, and leaves nothing loaded.
 *
 * Uses models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL); skipped when absent.
 */

#include "local_test_support.h"

#include "../src/accel/die_manager.h"
#include "../src/core/load_guard.h"
#include "../src/core/mem/pn_host_admission.h"
#include "../src/serve/model_admin.h"

#define GIB(n) ((uint64_t)(n) << 30)
#define FAKE_REASON "GPU die 6 needs ~11945 MiB, has 11467 MiB usable " \
                    "(32/32 layers + KV for 262144 tokens + scratch)"

static const char *g_model;

static ma_load_request request(const char *name)
{
    ma_load_request lr;

    memset(&lr, 0, sizeof(lr));
    lr.path = g_model;
    lr.name = name;
    lr.opts.ctx_cap = 1024;
    lr.max_sessions = 1;
    return lr;
}

static int die_models(int die)
{
    pn_dm_die d;
    return pn_dm_get(die, &d) == 0 ? d.n_models : -1;
}

/* The accelerator refused: the load fails with the reason, no model, no charge. */
static void test_engine_refuses_fallback(void)
{
    pn_load_opts opts = { .ctx_cap = 1024 };
    int dies[1] = { 6 };

    printf("engine_refuses_fallback:\n");
    uint64_t refusals = pn_load_thread_refusals();
    uint64_t held = pn_host_admitted_bytes();
    pn_load_guard_test_fake_accel(FAKE_REASON);
    pn_model *m = pn_load_on(g_model, &opts, dies, 1);
    CHECK(m == NULL, "the load fails: no model, so no CPU forward can run");
    pn_model_free(m);
    CHECK(pn_load_thread_refusals() == refusals + 1, "counted as a load refusal");
    const char *why = pn_load_refusal_last();
    CHECK(strstr(why, "qwen3-0.6b") || strstr(why, ".gguf"), "names the model");
    CHECK(strstr(why, FAKE_REASON) != NULL, "carries the accelerator's reason (needed vs free)");
    CHECK(strstr(why, "No CPU fallback") != NULL, "says it did not fall back");
    CHECK(strstr(why, "smaller context") && strstr(why, "fewer users") &&
              strstr(why, "smaller quant"), "suggests what to change");
    CHECK(strlen(why) < 256, "fits the 256-byte reason buffers of Local / reload");
    CHECK(pn_host_admitted_bytes() == held, "nothing stays charged to the host");

    opts.cpu_only = true;
    m = pn_load_on(g_model, &opts, NULL, 0);
    CHECK(m != NULL && !pn_model_on_accelerator(m), "a cpu_only load still runs on the CPU");
    pn_model_free(m);
    pn_load_guard_test_fake_accel(NULL);
}

/* Registry + --mtp: the real cause is reported, nothing stays loaded or bound. */
static void test_registry_reports_cause(mr_registry *reg)
{
    ma_load_request lr = request("refused-gpu");
    char err[PN_LOAD_REFUSAL_LEN] = "";

    printf("registry_reports_cause:\n");
    lr.mtp = true;
    pn_load_guard_test_fake_accel(FAKE_REASON);
    CHECK(ma_load_model(reg, &lr, err, sizeof(err)) != 0, "the load fails");
    CHECK(strstr(err, FAKE_REASON) != NULL, "the error is the accelerator's reason");
    CHECK(strstr(err, "MTP") == NULL, "not \"no MTP head\"");
    CHECK(mr_count(reg) == 0, "nothing is loaded");
    CHECK(die_models(0) == 0 && die_models(1) == 0, "no die stays bound");
    pn_load_guard_test_fake_accel(NULL);
}

/* No die has room: refused, not placed on the host CPU. */
static void test_no_room_is_refused(mr_registry *reg)
{
    ma_load_request lr = request("no-room");
    char err[PN_LOAD_REFUSAL_LEN] = "";

    printf("no_room_is_refused:\n");
    pn_dm_shutdown();
    CHECK(pn_dm_init_logical(2, 64ULL << 20) == 0, "two 64 MiB dies");
    pn_load_guard_test_fake_accel(FAKE_REASON);
    CHECK(ma_load_model(reg, &lr, err, sizeof(err)) != 0, "the load fails");
    CHECK(strstr(err, "no die has room") && strstr(err, "no CPU fallback"),
          "says no die has room and that it did not fall back");
    CHECK(mr_count(reg) == 0, "nothing is loaded");
    pn_load_guard_test_fake_accel(NULL);
    pn_dm_shutdown();
    CHECK(pn_dm_init_logical(2, GIB(11)) == 0, "two 11 GiB dies again");
}

/* An explicit CPU load must fit host RAM (mmap'd weights read every token). */
static void test_cpu_load_host_fit(void)
{
    pn_load_opts opts = { .ctx_cap = 1024, .cpu_only = true };
    uint64_t ws = pn_load_guard_cpu_working_set(g_model);
    uint64_t reserve = pn_host_admit_reserve_bytes();

    printf("cpu_load_host_fit:\n");
    CHECK(ws > 500ULL << 20 && ws < 700ULL << 20, "working set of a dense 0.6B Q8_0 ~ its file");
    pn_host_admit_test_reset();
    pn_host_admit_test_set_memory(reserve + ws / 2, 0);
    pn_model *m = pn_load_on(g_model, &opts, NULL, 0);
    CHECK(m == NULL, "refused when half the weights fit");
    CHECK(strstr(pn_load_refusal_last(), "host RAM") &&
              strstr(pn_load_refusal_last(), "on the CPU"), "says host RAM, on the CPU");
    pn_model_free(m);

    pn_host_admit_test_set_memory(reserve + GIB(8), 0);
    m = pn_load_on(g_model, &opts, NULL, 0);
    CHECK(m != NULL, "loads when they fit");
    pn_model_free(m);
    pn_host_admit_test_reset();
}

/* --mtp on a CPU load: says why, leaves nothing loaded. */
static void test_mtp_on_cpu_load(mr_registry *reg)
{
    ma_load_request lr = request("cpu-mtp");
    char err[PN_LOAD_REFUSAL_LEN] = "";

    printf("mtp_on_cpu_load:\n");
    lr.opts.cpu_only = true;
    lr.mtp = true;
    CHECK(ma_load_model(reg, &lr, err, sizeof(err)) != 0, "refused");
    CHECK(strstr(err, "--mtp needs the model on a GPU") != NULL, "says the load is on the CPU");
    CHECK(mr_count(reg) == 0, "and unloads it");
}

int main(void)
{
    mr_registry reg;

    printf("test_load_guard\n");
#if defined(USE_CUDA) || defined(USE_TT)
    /* The fake stands in for a device; with a real one the upload would run. */
    printf("  SKIP: accelerator build (the real refusal is checked on hardware)\n");
    return 0;
#endif
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP: %s not found\n", g_model);
        return 0;
    }
    CHECK(pn_dm_init_logical(2, GIB(11)) == 0, "two logical dies");
    CHECK(mr_init(&reg) == 0, "registry");
    test_engine_refuses_fallback();
    test_registry_reports_cause(&reg);
    test_no_room_is_refused(&reg);
    test_cpu_load_host_fit();
    test_mtp_on_cpu_load(&reg);
    mr_free(&reg);
    pn_dm_shutdown();
    return lt_report("test_load_guard");
}
