/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_load_budget_release.c — a load that fails after its die reservation
 * gives back exactly the reservation it took (model_registry.c, die_manager.c).
 *
 *   K80 bank 2026-11-03, 12 dies: copies of qwen3.6-35b-a3b-q8 share one
 *   registry name. A third copy on units 8-11 was refused by the host-RAM gate;
 *   the failure path released the FIRST binding of that name (copy 1's), so die
 *   8 stayed budgeted ("die 8 has 792 MiB free, needs 10261 MiB" with the die
 *   empty) until Pantheon Local restarted.
 *
 *   - A binding is released by its own handle: two same-name bindings on one
 *     die, the right one goes; a stale handle (slot reused, or from before a
 *     manager restart) releases nothing.
 *   - Same-name replicas resident; a further replica fails at each stage after
 *     the reservation (upload refused, host RAM refused before the load, host
 *     RAM refused for its sessions after the load): every die's budget is back
 *     to its prior value and the resident replicas keep theirs.
 *   - A load that fits then succeeds on the dies the refused one had.
 *   - Unloading a replica releases its own binding, not its namesake's.
 *
 * Logical dies + the load guard's fake accelerator, so it runs on the Mac.
 * Registry parts use models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL); skipped when
 * absent.
 */

#include "local_test_support.h"

#include "../src/accel/die_manager.h"
#include "../src/core/load_guard.h"
#include "../src/core/mem/pn_host_admission.h"
#include "../src/serve/model_admin.h"

#define GIB(n)  ((uint64_t)(n) << 30)
#define N_DIES  4
#define NAME    "qwen-copy"
#define FAKE_REASON "host RAM for the GPU upload: needs ~2400 MiB beyond the reserve, " \
                    "2670 MiB available"

static const char *g_model;

/* What the die manager holds on every die. */
typedef struct {
    uint64_t used[N_DIES];
    int      models[N_DIES];
} die_budget_snapshot;

static die_budget_snapshot snapshot_dies(void)
{
    die_budget_snapshot s;

    memset(&s, 0, sizeof(s));
    for (int i = 0; i < N_DIES; i++) {
        pn_dm_die d;
        if (pn_dm_get(i, &d) == 0) {
            s.used[i] = d.vram_used;
            s.models[i] = d.n_models;
        }
    }
    return s;
}

static int snapshots_equal(const die_budget_snapshot *a, const die_budget_snapshot *b)
{
    return memcmp(a, b, sizeof(*a)) == 0;
}

/* One more copy of NAME, auto-placed (a non-accelerator build refuses an
 * explicit die list; auto-placement still binds a die). */
static ma_load_request replica_request(void)
{
    ma_load_request lr;

    memset(&lr, 0, sizeof(lr));
    lr.path = g_model;
    lr.name = NAME;
    lr.opts.ctx_cap = 1024;
    lr.max_sessions = 1;
    return lr;
}

/* Bytes one copy reserves on its die: load one on a roomy die and look. */
static uint64_t measure_copy_reservation(void)
{
    mr_registry reg;
    ma_load_request lr = replica_request();
    char err[PN_LOAD_REFUSAL_LEN] = "";
    pn_dm_die d;
    uint64_t used = 0;

    CHECK(pn_dm_init_logical(1, GIB(11)) == 0, "one roomy logical die");
    CHECK(mr_init(&reg) == 0, "registry");
    CHECK(ma_load_model(&reg, &lr, err, sizeof(err)) == 0, "a probe copy loads");
    if (pn_dm_get(0, &d) == 0)
        used = d.vram_used;
    mr_free(&reg);
    pn_dm_shutdown();
    CHECK(used > 0, "the probe copy reserved its die");
    return used;
}

/* Two bindings of one name on ONE die: each handle releases its own. */
static void test_release_by_handle(void)
{
    int die0 = 0;
    pn_dm_binding_ref small = PN_DM_BINDING_NONE, big = PN_DM_BINDING_NONE;
    pn_dm_die d;

    printf("release_by_handle:\n");
    CHECK(pn_dm_init_logical(N_DIES, GIB(11)) == 0, "logical dies");
    CHECK(pn_dm_bind_ref(&die0, 1, NAME, GIB(1), &small) == 0, "bind 1 GiB");
    CHECK(pn_dm_bind_ref(&die0, 1, NAME, GIB(3), &big) == 0, "bind 3 GiB, same name, same die");
    CHECK(pn_dm_release_ref(&big) == 0, "release the 3 GiB binding by its handle");
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == GIB(1) && d.n_models == 1,
          "the 1 GiB binding is the one left");
    CHECK(pn_dm_release_ref(&big) == -1, "the released handle releases nothing again");
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == GIB(1), "and leaves the die alone");
    pn_dm_binding_ref stale = small;
    CHECK(pn_dm_release_ref(&small) == 0 && pn_dm_get(0, &d) == 0 && d.vram_used == 0,
          "releasing the other empties the die");
    pn_dm_binding_ref reuse = PN_DM_BINDING_NONE;
    CHECK(pn_dm_bind_ref(&die0, 1, NAME, GIB(2), &reuse) == 0 && reuse.slot == stale.slot,
          "a new binding reuses the freed slot");
    CHECK(pn_dm_release_ref(&stale) == -1, "a stale copy of the old handle is refused");
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == GIB(2), "and the new binding keeps its bytes");
    pn_dm_shutdown();
    CHECK(pn_dm_init_logical(N_DIES, GIB(11)) == 0 && pn_dm_bind_ref(&die0, 1, NAME, GIB(1),
          &small) == 0 && pn_dm_release_ref(&reuse) == -1,
          "a handle from before a manager restart releases nothing");
    pn_dm_shutdown();
}

/* One failed replica load: refused, and every die back where it was. */
static void expect_refused_and_restored(mr_registry *reg, const ma_load_request *lr,
                                        const char *stage, const char *why)
{
    char err[PN_LOAD_REFUSAL_LEN] = "", msg[160];
    die_budget_snapshot before = snapshot_dies();
    int count = mr_count(reg);

    snprintf(msg, sizeof(msg), "%s: the load fails", stage);
    CHECK(ma_load_model(reg, lr, err, sizeof(err)) != 0, msg);
    snprintf(msg, sizeof(msg), "%s: refused at that stage (\"%s\")", stage, why);
    CHECK(strstr(err, why) != NULL, msg);
    if (!strstr(err, why))
        fprintf(stderr, "    error was: %s\n", err);
    snprintf(msg, sizeof(msg), "%s: nothing new is loaded", stage);
    CHECK(mr_count(reg) == count, msg);
    die_budget_snapshot after = snapshot_dies();
    snprintf(msg, sizeof(msg), "%s: every die's budget is back to its prior value", stage);
    CHECK(snapshots_equal(&before, &after), msg);
    if (!snapshots_equal(&before, &after))
        for (int i = 0; i < N_DIES; i++)
            fprintf(stderr, "    die %d: %llu MiB / %d model(s) before, %llu MiB / %d after\n",
                    i, (unsigned long long)(before.used[i] >> 20), before.models[i],
                    (unsigned long long)(after.used[i] >> 20), after.models[i]);
}

/* Where a replica load fails, each one after the die reservation. */
typedef enum {
    STAGE_UPLOAD_REFUSED,        /* the accelerator upload refuses (fake) */
    STAGE_HOST_RAM_BEFORE_LOAD,  /* host RAM: the weights do not fit */
    STAGE_HOST_RAM_SESSIONS,     /* host RAM: the model loaded, its session KV does not fit */
    STAGE_COUNT
} failure_stage;

static const char *const k_stage_names[STAGE_COUNT] = {
    "upload refused", "host RAM refused before the load", "host RAM refused for the sessions",
};

/* The refusal text each stage must produce, so a stage that fails EARLIER
 * (before the reservation) cannot pass for it. */
static const char *const k_stage_reasons[STAGE_COUNT] = {
    FAKE_REASON, "on the CPU (weights read every token)", "host RAM admission refused",
};

static void arm_failure_stage(failure_stage stage)
{
    uint64_t reserve = pn_host_admit_reserve_bytes();
    uint64_t ws = pn_load_guard_cpu_working_set(g_model);

    if (stage == STAGE_UPLOAD_REFUSED)
        pn_load_guard_test_fake_accel(FAKE_REASON);
    else if (stage == STAGE_HOST_RAM_BEFORE_LOAD)
        pn_host_admit_test_set_memory(reserve + ws / 2, 0);
    else   /* room for the weights beside what is already admitted, not the KV */
        pn_host_admit_test_set_memory(reserve + pn_host_admitted_bytes() + ws + (8ULL << 20), 0);
}

static void disarm_failure_stages(void)
{
    pn_load_guard_test_fake_accel(NULL);
    pn_host_admit_test_reset();
}

/* From a fresh start (dies that hold one copy each; copies 1 and 2 on dies 0
 * and 1), a third copy fails at `stage`, then a load that fits succeeds on the
 * die it had. The last stage also unloads copy 2 by its replica. */
static void test_failed_replica_stage(failure_stage stage, uint64_t one_copy)
{
    mr_registry reg;
    char err[PN_LOAD_REFUSAL_LEN] = "";
    ma_load_request lr = replica_request();

    printf("failed_replica_releases_its_own_dies (%s):\n", k_stage_names[stage]);
    CHECK(pn_dm_init_logical(N_DIES, one_copy + one_copy / 2) == 0, "one-copy logical dies");
    CHECK(mr_init(&reg) == 0, "registry");
    CHECK(ma_load_model(&reg, &lr, err, sizeof(err)) == 0, "copy 1");
    CHECK(ma_load_model(&reg, &lr, err, sizeof(err)) == 0, "copy 2, same name");
    die_budget_snapshot resident = snapshot_dies();
    CHECK(resident.models[0] == 1 && resident.models[1] == 1 && resident.models[2] == 0,
          "dies 0 and 1 bound, die 2 free");

    arm_failure_stage(stage);
    expect_refused_and_restored(&reg, &lr, k_stage_names[stage], k_stage_reasons[stage]);
    disarm_failure_stages();

    CHECK(ma_load_model(&reg, &lr, err, sizeof(err)) == 0,
          "a load that fits then succeeds on the refused load's die");
    die_budget_snapshot three = snapshot_dies();
    CHECK(three.models[2] == 1 && three.used[0] == resident.used[0] &&
              three.used[1] == resident.used[1], "die 2 bound; dies 0 and 1 unchanged");

    if (stage == STAGE_COUNT - 1) {
        CHECK(mr_unload_replica(&reg, NAME, 1, err, sizeof(err)) == 0, "unload copy 2");
        die_budget_snapshot two = snapshot_dies();
        CHECK(two.models[1] == 0 && two.used[1] == 0, "die 1 freed");
        CHECK(two.used[0] == three.used[0] && two.used[2] == three.used[2],
              "copies 1 and 3 keep their budget");
    }
    mr_free(&reg);
    pn_dm_shutdown();
}

int main(void)
{
    printf("test_load_budget_release\n");
    test_release_by_handle();
#if defined(USE_CUDA) || defined(USE_TT)
    /* The fake stands in for a device; with a real one the upload would run. */
    printf("  SKIP registry part: accelerator build\n");
    return lt_report("test_load_budget_release");
#endif
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP registry part: %s not found\n", g_model);
        return lt_report("test_load_budget_release");
    }
    uint64_t one_copy = measure_copy_reservation();
    for (int stage = 0; stage < STAGE_COUNT; stage++)
        test_failed_replica_stage((failure_stage)stage, one_copy);
    return lt_report("test_load_budget_release");
}
