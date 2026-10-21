/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_model_registry.c — Phase 7 model-registry unit tests.
 *
 * Part A (always runs): registry bookkeeping with no model — empty-registry
 * behavior, bad-path load refusal, unknown unload refusal.
 *
 * Part B (model-backed; skips with a note if the GGUF is absent): two names
 * resident from one file, duplicate rejection, default resolution,
 * acquire/release refcounts, pn_load_on placement recorded on the model,
 * generation smoke on an acquired entry, and unload-drain blocking until a
 * concurrent holder releases. Model: models/qwen3-0.6b-Q8_0.gguf (override
 * with PN_TEST_MODEL). CPU path only — no GPU needed.
 */

#include "../src/serve/model_registry.h"
#include "../src/accel/die_manager.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define GIB(n) ((uint64_t)(n) << 30)

static const char *test_model_path(void)
{
    const char *env = getenv("PN_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

/* --- Part A: bookkeeping without a model --------------------------------- */

static void test_empty_registry(void)
{
    printf("empty_registry:\n");
    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "init");
    CHECK(mr_count(&reg) == 0, "empty count");
    CHECK(mr_acquire(&reg, NULL) == NULL, "no default on empty registry");
    CHECK(mr_acquire(&reg, "nope") == NULL, "unknown name on empty registry");

    char err[128] = "";
    CHECK(mr_unload(&reg, "nope", err, sizeof(err)) == -1, "unknown unload refused");
    CHECK(err[0] != '\0', "unload refusal carries a reason");

    err[0] = '\0';
    CHECK(mr_load(&reg, "/no/such/file.gguf", NULL, NULL, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == -1, "missing file refused");
    CHECK(err[0] != '\0', "load refusal carries a reason");
    CHECK(mr_count(&reg) == 0, "failed load leaves the registry empty");

    mr_free(&reg);
}

/* --- Part B: model-backed ------------------------------------------------- */

typedef struct {
    mr_registry *reg;
    mr_entry    *held;
    int          released;   /* set right before mr_release */
} holder_args;

static void *holder_thread(void *arg)
{
    holder_args *h = arg;
    struct timespec ts = { 0, 300 * 1000 * 1000 };   /* hold 300 ms */
    nanosleep(&ts, NULL);
    h->released = 1;
    mr_release(h->reg, h->held);
    return NULL;
}

/* The pre-load gate must size the window to the model's --ctx-cap, not to its
 * 8k guess.
 *
 * A cap is not advisory: the KV ring is allocated to it, and the per-die scratch
 * reserve is sized from it, so gating at 8k charges a --ctx-cap 2048 model for
 * headroom it will never allocate. On a 2-die Qwen3.6-35B-A3B Q4_K_M that alone
 * is the difference between refusing and loading, which is why this is asserted
 * on a budget that straddles the two rather than on a spacious one where either
 * answer passes.
 *
 * Numbers for the 0.6b test model (1039 MiB resident -- see the projection test
 * below for why that exceeds its 609 MiB file), as a GPU load: weights + KV at
 * the generic 5898 B/token + pn_admit_die_scratch_bytes(cap).
 *
 *     8192: 1039 + 46 + 768  = 1853 MiB
 *     2048: 1039 + 12 + 576  = 1627 MiB
 *
 * 1740 MiB is their midpoint and straddles them with ~113 MiB either side, and
 * clears the 1039 MiB placement check both ways so it is the ADMISSION gate
 * being measured, not placement.
 *
 * (These were 2285 / 1210 MiB against a 1700 MiB budget while this path still
 * priced a GPU load through the CPU-resident gate. Dropping that gate's host
 * activation LUT -- 1200 MiB at 8192, 160 MiB at 2048 -- is what moved them,
 * and is what stops a --ctx-cap 131072 serve being refused outright.)
 *
 * The budget is a die-manager budget, already net of the per-die reserve; the
 * gate must not hold that back a second time. */
static void test_ctx_cap_gates_admission(void)
{
    const char *path = test_model_path();
    if (access(path, R_OK) != 0) {
        printf("ctx_cap_gates_admission: [SKIP] %s not found\n", path);
        return;
    }
    printf("ctx_cap_gates_admission:\n");
    CHECK(pn_dm_init_logical(1, 1740ull << 20) == 0, "one tight logical die");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "init");
    char err[256] = "";
    pn_dm_die d;

    /* No cap -> the 8k guess applies -> refused. The 1039 MiB share PLACES on
     * this die, so the refusal comes from the admission gate rather than from
     * placement falling through to a CPU-resident load — which is the gate this
     * test is about. */
    CHECK(mr_load(&reg, path, "uncapped", NULL, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == -1,
          "no --ctx-cap: gated at the 8k guess, and this die cannot hold that");
    CHECK(err[0] != '\0', "and the refusal carries its numbers");
    /* And they are GPU numbers. 1040 weights + 46 KV + 768 scratch = 1854 MiB;
     * the CPU-resident gate would have said 2286 here, and 56363 at the deployed config's cap.
     * Asserting the figure is what ties this test to the projection rather than
     * merely to the verdict. */
    {
        const char *n = strstr(err, "needs ");
        long needs_mib = n ? strtol(n + 6, NULL, 10) : -1;
        CHECK(needs_mib >= 1840 && needs_mib <= 1870,
              "the refusal quotes the GPU projection, not the host LUT");
        if (!(needs_mib >= 1840 && needs_mib <= 1870))
            fprintf(stderr, "  (needs was %ld MiB; err: %s)\n", needs_mib, err);
    }
    CHECK(mr_count(&reg) == 0, "the refusal left no entry behind");
    CHECK(pn_dm_get(0, &d) == 0 && d.n_models == 0 && d.vram_used == 0,
          "and no die binding either");

    /* Same model, same die, --ctx-cap 2048 -> admitted onto the die. */
    pn_load_opts opts = { 0 };
    opts.ctx_cap = 2048;
    CHECK(mr_load(&reg, path, "capped", &opts, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == 0, "capped loads");
    CHECK(pn_dm_get(0, &d) == 0 && d.n_models == 1,
          "--ctx-cap 2048: gated at the cap the ring is really sized to, "
          "so the die IS bound");
    CHECK(mr_unload(&reg, "capped", err, sizeof(err)) == 0, "unload");

    mr_free(&reg);
    pn_dm_shutdown();
}

/* The pre-load gate must project a die's weights from the TENSOR TABLE, not
 * from a slice of the file — and this test uses the under-projection direction
 * because that is the one that hurts.
 *
 * The 0.6b test model is TIED: it has no output.weight, so its LM head is
 * token_embd materialized to F32 at load. Its real single-die residency is
 * 1039 MiB (446 MiB of layers + a 593 MiB F32 head) against a 609 MiB file.
 * Prorating the file UNDER-projects it by 430 MiB — it puts a model on a die
 * that cannot hold it, which is worse than the over-projection that refuses
 * one that can.
 *
 * 900 MiB is the straddle: the real 1039 MiB share does not even place, while
 * the prorated 609 MiB share does.
 *
 * What proration then costs has CHANGED, and honestly so rather than by tuning
 * the budget. It used to place AND admit (780 MiB needed at 2048), binding a die
 * that cannot hold the model -- the bad-bind failure this test was written for.
 * Now the GPU projection charges 609 + 12 KV + 576 scratch = 1197 MiB, so a
 * prorated share places and is then REFUSED. The test still detects proration --
 * mr_load returns -1 where it must return 0, and the die stays unbound -- but
 * via a refusal, not a bad bind.
 *
 * The old mechanism is no longer reachable at any budget, and that is a fact
 * about the fix rather than about this test: admitting a prorated share would
 * need budget >= 1197 while not placing the real share needs budget < 1039. The
 * 512 MiB scratch floor is larger than the 430 MiB proration gap, so an
 * under-projected share can no longer sneak onto a die too small for it. */
static void test_weight_projection_is_not_prorated(void)
{
    const char *path = test_model_path();
    if (access(path, R_OK) != 0) {
        printf("weight_projection_is_not_prorated: [SKIP] %s not found\n", path);
        return;
    }
    printf("weight_projection_is_not_prorated:\n");
    CHECK(pn_dm_init_logical(1, 900ull << 20) == 0, "one straddling die");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "init");
    char err[256] = "";

    pn_load_opts opts = { 0 };
    opts.ctx_cap = 2048;
    CHECK(mr_load(&reg, path, "tied", &opts, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == 0, "it loads (CPU-resident)");
    pn_dm_die d;
    CHECK(pn_dm_get(0, &d) == 0 && d.n_models == 0 && d.vram_used == 0,
          "but NO die is bound: the F32 head this tied model really uploads "
          "does not fit, and the file size does not say so");
    CHECK(mr_unload(&reg, "tied", err, sizeof(err)) == 0, "unload");

    mr_free(&reg);
    pn_dm_shutdown();
}

static void test_model_backed(void)
{
    const char *path = test_model_path();
    if (access(path, R_OK) != 0) {
        printf("model_backed: [SKIP] %s not found (set PN_TEST_MODEL)\n", path);
        return;
    }
    printf("model_backed: (%s)\n", path);

    /* Logical dies so auto-placement engages (the 0.6b fits one die). */
    CHECK(pn_dm_init_logical(4, GIB(11)) == 0, "logical die manager up");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "init");

    char err[128] = "";
    CHECK(mr_load(&reg, path, "alpha", NULL, NULL, 0, 0, 0, 0, err, sizeof(err)) == 0,
          "load 'alpha'");
    /* A repeated name is NOT refused — see the same-name-placement block at the
     * end of this test for what it does instead (model_registry.c mr_load). */
    CHECK(mr_load(&reg, path, "beta", NULL, NULL, 0, 0, 0, 0, err, sizeof(err)) == 0,
          "load 'beta' (same file, second name)");
    CHECK(mr_count(&reg) == 2, "two resident models");

    mr_info info[4];
    CHECK(mr_snapshot(&reg, info, 4) == 2, "snapshot lists both");

    /* Default = the first (lowest-slot) resident model. */
    mr_entry *def = mr_acquire(&reg, NULL);
    CHECK(def && strcmp(def->name, "alpha") == 0, "default resolves to alpha");

    /* Placement bookkeeping: die manager bound a die-set and pn_load_on
     * recorded the same set on the model. */
    CHECK(def->n_dies >= 1, "alpha auto-placed on a die-set");
    CHECK(def->model && def->model->n_gpu_dies == def->n_dies &&
          def->model->gpu_dies[0] == def->dies[0],
          "pn_load_on recorded the placement on the model");
    pn_dm_die d;
    CHECK(pn_dm_get(def->dies[0], &d) == 0 && d.n_models >= 1,
          "die manager shows the binding");

    /* refs are visible while held. */
    CHECK(mr_snapshot(&reg, info, 4) == 2, "snapshot again");
    int alpha_refs = -1;
    for (int i = 0; i < 2; i++)
        if (strcmp(info[i].name, "alpha") == 0)
            alpha_refs = info[i].refs;
    CHECK(alpha_refs == 1, "held entry shows one ref");

    /* Generation smoke on the acquired entry (CPU forward). */
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = 0.0f;
    char *text = pn_generate(def->model, def->state, "Hello", 4, &sp);
    CHECK(text != NULL && *text, "acquired entry generates");
    if (text) {
        printf("    alpha says: %.40s\n", text);
        free(text);
    }
    mr_release(&reg, def);

    /* Unload-drain: a holder keeps beta for ~300 ms; unload must block until
     * the release, never freeing under the holder. */
    mr_entry *beta = mr_acquire(&reg, "beta");
    CHECK(beta != NULL, "acquire beta");
    holder_args h = { .reg = &reg, .held = beta, .released = 0 };
    pthread_t tid;
    CHECK(pthread_create(&tid, NULL, holder_thread, &h) == 0, "holder thread");

    CHECK(mr_unload(&reg, "beta", err, sizeof(err)) == 0, "unload beta");
    CHECK(h.released == 1, "unload blocked until the holder released");
    pthread_join(tid, NULL);

    CHECK(mr_count(&reg) == 1, "one model left");
    CHECK(mr_acquire(&reg, "beta") == NULL, "beta gone after unload");

    /* Unloading alpha returns its die to the manager. */
    int alpha_die = -1;
    if (mr_snapshot(&reg, info, 4) == 1 && info[0].n_dies > 0)
        alpha_die = info[0].dies[0];
    CHECK(mr_unload(&reg, "alpha", err, sizeof(err)) == 0, "unload alpha");
    CHECK(mr_count(&reg) == 0, "registry empty again");
    if (alpha_die >= 0) {
        CHECK(pn_dm_get(alpha_die, &d) == 0 && d.n_models == 0 &&
              d.vram_used == 0, "die freed after unload");
    }

    /* --- same-name placements ------------------------------------------- */
    /* An operator places ONE model id on several cards (--dies 0,1 and
     * --dies 2,3 under one --name) and mr_acquire load-balances across them, so
     * a repeated name reserves ANOTHER slot rather than being refused. Clients
     * never see the duplication: oai_handle_models dedupes by name, and
     * load/unload is operator-only. Registry starts empty here. */
    CHECK(mr_count(&reg) == 0, "starting from an empty registry");
    CHECK(mr_load(&reg, path, "dup", NULL, NULL, 0, 0, 0, 0, err, sizeof(err)) == 0,
          "load 'dup'");
    CHECK(mr_load(&reg, path, "dup", NULL, NULL, 0, 0, 0, 0, err, sizeof(err)) == 0,
          "the same name loads again as a second placement");
    CHECK(mr_count(&reg) == 2, "two entries under one id");
    CHECK(mr_snapshot(&reg, info, 4) == 2, "snapshot lists both placements");
    CHECK(strcmp(info[0].name, "dup") == 0 && strcmp(info[1].name, "dup") == 0,
          "both entries carry the same id");

    /* Least-loaded wins: the second acquire must land on the OTHER placement,
     * which is the whole point of allowing the duplicate. */
    mr_entry *p1 = mr_acquire(&reg, "dup");
    mr_entry *p2 = mr_acquire(&reg, "dup");
    CHECK(p1 && p2, "two acquires under one id");
    CHECK(p1 != p2, "the second acquire load-balances onto the other placement");
    mr_release(&reg, p1);
    mr_release(&reg, p2);

    /* One unload per placement — first-match, like pn_dm_release. */
    CHECK(mr_unload(&reg, "dup", err, sizeof(err)) == 0, "unload one placement");
    CHECK(mr_count(&reg) == 1, "the other placement is still resident");
    mr_entry *survivor = mr_acquire(&reg, "dup");
    CHECK(survivor != NULL, "the id still serves from the survivor");
    mr_release(&reg, survivor);
    CHECK(mr_unload(&reg, "dup", err, sizeof(err)) == 0, "unload the last one");
    CHECK(mr_count(&reg) == 0, "id gone once every placement is unloaded");
    CHECK(mr_unload(&reg, "dup", err, sizeof(err)) == -1,
          "a further unload is refused");

    mr_free(&reg);
    pn_dm_shutdown();
}

int main(void)
{
    printf("test_model_registry\n");
    test_empty_registry();
    test_model_backed();
    test_ctx_cap_gates_admission();
    test_weight_projection_is_not_prorated();

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
