/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_model_unload_reclaim.c — an unload must give the dies BACK, or refuse.
 *
 * The defect this covers, observed live on the K80 test machine: POST /admin/models/unload
 * answered success, the model vanished from /v1/models, and the two dies it had
 * been placed on stayed budgeted AND resident. The very next load onto those
 * dies came back "die VRAM budget refused the model (admission)", and the only
 * recovery was a full serve restart, which drops every other model with it.
 *
 * Root cause: mr_unload marked the entry `unloading` and decremented the count
 * FIRST, then waited for in-flight references with an UNBOUNDED
 * pthread_cond_wait. A reference that never drains — pn_session_watch_add holds
 * one for the whole lifetime of an open /session/watch — parked the unload
 * there forever, so pn_dm_release and the model free below it never ran. One
 * cause, both symptoms: stale die accounting AND resident VRAM.
 *
 * Three behaviours, each with its own arm:
 *   1. reclaim  — unload returns the die budget, and a DIFFERENT model then
 *                 loads onto the SAME die in the SAME process. The die is
 *                 sized so exactly one model fits, so a leaked budget refuses
 *                 the second load. (Loading into an already-free die proves
 *                 nothing — that case always passed.)
 *   2. refuse   — an unload that cannot drain refuses, names the reason, and
 *                 rolls the model back to fully resident. Never 200-and-leak.
 *   3. drain    — a transient holder is still WAITED for; the bounded drain
 *                 must not have turned into an instant refusal.
 *
 * Model: models/qwen3-0.6b-Q8_0.gguf (override with PN_TEST_MODEL). Logical
 * dies only — the accounting under test is host-side, no GPU needed.
 */

#include "../src/serve/model_registry.h"
#include "../src/accel/die_manager.h"
#include "../src/core/mem/pn_memwatch.h"

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

#define MIB(n) ((uint64_t)(n) << 20)

static const char *test_model_path(void)
{
    const char *env = getenv("PN_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

static void nap_ms(int ms)
{
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static uint64_t die0_used(void)
{
    pn_dm_die d;
    return pn_dm_get(0, &d) == 0 ? d.vram_used : 0;
}

static int die0_models(void)
{
    pn_dm_die d;
    return pn_dm_get(0, &d) == 0 ? d.n_models : -1;
}

/* One logical die sized so EXACTLY ONE copy of the test model fits: probe the
 * admission cost by loading once against a generous budget, then re-init the
 * die manager at that cost plus the per-die reserve and a little slack. A die
 * this tight is the whole point — it is what makes a leaked budget refuse the
 * next load instead of quietly fitting beside the leak. Returns 0 on success. */
static int size_one_model_die(const char *path, uint64_t *budget_out)
{
    if (pn_dm_init_logical(1, MIB(32768)) != 0)
        return -1;
    mr_registry probe;
    if (mr_init(&probe) != 0)
        return -1;
    char err[160] = "";
    if (mr_load(&probe, path, "probe", NULL, NULL, 0, 0, 0, 0,
                err, sizeof(err)) != 0) {
        fprintf(stderr, "  probe load failed: %s\n", err);
        return -1;
    }
    uint64_t need = die0_used();
    mr_free(&probe);
    pn_dm_shutdown();
    if (need == 0)
        return -1;
    *budget_out = need + pn_die_reserve_bytes() + MIB(32);
    return 0;
}

/* --- 1. reclaim: unload gives the die back, a different model takes it ---- */

static void test_unload_reclaims_the_die(const char *path, uint64_t budget)
{
    printf("unload_reclaims_the_die:\n");
    CHECK(pn_dm_init_logical(1, budget) == 0, "one-model-sized logical die up");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "registry init");

    char err[160] = "";
    CHECK(mr_load(&reg, path, "alpha", NULL, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == 0, "load 'alpha'");
    uint64_t held = die0_used();
    CHECK(held > 0, "alpha's placement is charged to the die");
    CHECK(die0_models() == 1, "die shows one bound model");

    err[0] = '\0';
    CHECK(mr_unload(&reg, "alpha", err, sizeof(err)) == 0, "unload 'alpha'");
    /* The host-side mirror of "nvidia-smi is back at its idle floor". */
    CHECK(die0_used() == 0, "die budget is back to zero after unload");
    CHECK(die0_models() == 0, "die has no bound model after unload");

    /* THE acceptance criterion: a DIFFERENT model onto the SAME die, in the
     * SAME process. The die only fits one, so a leaked budget refuses here. */
    err[0] = '\0';
    int rc = mr_load(&reg, path, "beta", NULL, NULL, 0, 0, 0, 0,
                     err, sizeof(err));
    if (rc != 0)
        fprintf(stderr, "  load-after-unload refused: %s\n", err);
    CHECK(rc == 0, "a different model loads onto the same die after unload");
    CHECK(die0_used() == held, "beta is charged the same die budget alpha was");

    mr_free(&reg);
    pn_dm_shutdown();
}

/* --- 2. refuse: an unload that cannot drain must not report success ------- */

typedef struct {
    mr_registry *reg;
    int          done;      /* set after mr_unload returns */
    int          rc;
    char         err[160];
} unload_call;

static void *unload_thread(void *arg)
{
    unload_call *u = arg;
    u->rc = mr_unload(u->reg, "alpha", u->err, sizeof(u->err));
    __atomic_store_n(&u->done, 1, __ATOMIC_RELEASE);
    return NULL;
}

/* Poll for the unload to return, up to `budget_ms`. */
static int wait_for_unload(unload_call *u, int budget_ms)
{
    for (int waited = 0; waited < budget_ms; waited += 20) {
        if (__atomic_load_n(&u->done, __ATOMIC_ACQUIRE))
            return 1;
        nap_ms(20);
    }
    return __atomic_load_n(&u->done, __ATOMIC_ACQUIRE);
}

static void test_pinned_unload_refuses(const char *path, uint64_t budget)
{
    printf("pinned_unload_refuses:\n");
    /* The registry outlives this function on the failure path: if the unload
     * never returns, its thread is left parked pointing at it. */
    static mr_registry reg;

    CHECK(pn_dm_init_logical(1, budget) == 0, "one-model-sized logical die up");
    CHECK(mr_init(&reg) == 0, "registry init");

    char err[160] = "";
    CHECK(mr_load(&reg, path, "alpha", NULL, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == 0, "load 'alpha'");
    uint64_t held = die0_used();

    /* A reference that does NOT go away on its own — what an open
     * /session/watch holds for the lifetime of its connection. */
    mr_entry *pin = mr_acquire(&reg, "alpha");
    CHECK(pin != NULL, "acquire pins alpha");

    /* Refuse as soon as anything is in flight, so the arm is fast and exact. */
    setenv("PN_UNLOAD_DRAIN_MS", "0", 1);

    unload_call u = { .reg = &reg, .done = 0, .rc = 0, .err = "" };
    pthread_t tid;
    CHECK(pthread_create(&tid, NULL, unload_thread, &u) == 0, "unload thread");

    if (!wait_for_unload(&u, 3000)) {
        /* This is the pre-fix behaviour: parked in an unbounded drain with the
         * entry already deregistered. Report every consequence and stop — the
         * parked thread makes anything after this non-deterministic. */
        CHECK(0, "unload returned within 3s instead of parking forever");
        CHECK(0, "pinned unload refused (never returned)");
        CHECK(mr_count(&reg) == 1, "model still resident while unload is stuck");
        CHECK(die0_used() == 0, "die budget released by the stuck unload");
        pthread_detach(tid);
        unsetenv("PN_UNLOAD_DRAIN_MS");
        return;
    }
    pthread_join(tid, NULL);
    unsetenv("PN_UNLOAD_DRAIN_MS");

    CHECK(u.rc != 0, "pinned unload REFUSES rather than reporting success");
    CHECK(strstr(u.err, "busy") != NULL, "the refusal says the model is busy");

    /* Rolled back: nothing was freed, so nothing may have been deregistered. */
    CHECK(mr_count(&reg) == 1, "refused unload leaves the model resident");
    mr_entry *again = mr_acquire(&reg, "alpha");
    CHECK(again != NULL, "refused unload leaves the model acquirable");
    if (again)
        mr_release(&reg, again);
    CHECK(die0_used() == held, "refused unload leaves the die budget intact");
    CHECK(die0_models() == 1, "refused unload leaves the die binding intact");

    /* Drop the pin and the same unload now succeeds and returns the die. */
    mr_release(&reg, pin);
    err[0] = '\0';
    CHECK(mr_unload(&reg, "alpha", err, sizeof(err)) == 0,
          "unload succeeds once the pin is gone");
    CHECK(die0_used() == 0, "die budget is back after the successful unload");

    err[0] = '\0';
    int rc = mr_load(&reg, path, "beta", NULL, NULL, 0, 0, 0, 0,
                     err, sizeof(err));
    if (rc != 0)
        fprintf(stderr, "  load-after-unload refused: %s\n", err);
    CHECK(rc == 0, "a different model loads onto the same die afterwards");

    mr_free(&reg);
    pn_dm_shutdown();
}

/* --- 3. drain: a transient holder is still waited for -------------------- */

typedef struct {
    mr_registry *reg;
    mr_entry    *held;
    int          released;
} transient_holder;

static void *transient_holder_thread(void *arg)
{
    transient_holder *h = arg;
    nap_ms(300);
    __atomic_store_n(&h->released, 1, __ATOMIC_RELEASE);
    mr_release(h->reg, h->held);
    return NULL;
}

static void test_transient_holder_is_drained(const char *path, uint64_t budget)
{
    printf("transient_holder_is_drained:\n");
    CHECK(pn_dm_init_logical(1, budget) == 0, "one-model-sized logical die up");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "registry init");

    char err[160] = "";
    CHECK(mr_load(&reg, path, "alpha", NULL, NULL, 0, 0, 0, 0,
                  err, sizeof(err)) == 0, "load 'alpha'");

    mr_entry *held = mr_acquire(&reg, "alpha");
    CHECK(held != NULL, "acquire alpha");
    transient_holder h = { .reg = &reg, .held = held, .released = 0 };
    pthread_t tid;
    CHECK(pthread_create(&tid, NULL, transient_holder_thread, &h) == 0,
          "holder thread");

    err[0] = '\0';
    CHECK(mr_unload(&reg, "alpha", err, sizeof(err)) == 0,
          "unload waits out a transient holder and succeeds");
    CHECK(__atomic_load_n(&h.released, __ATOMIC_ACQUIRE) == 1,
          "unload blocked until the holder released");
    pthread_join(tid, NULL);
    CHECK(die0_used() == 0, "die budget is back after the drained unload");

    mr_free(&reg);
    pn_dm_shutdown();
}

int main(void)
{
    printf("test_model_unload_reclaim\n");
    const char *path = test_model_path();
    if (access(path, R_OK) != 0) {
        printf("[SKIP] %s not found (set PN_TEST_MODEL)\n", path);
        return 0;
    }

    uint64_t budget = 0;
    if (size_one_model_die(path, &budget) != 0) {
        printf("FAILED: could not size a one-model die from %s\n", path);
        return 1;
    }
    printf("(one-model die budget: %llu MiB)\n",
           (unsigned long long)(budget >> 20));

    /* The pinned arm runs LAST on purpose: if the unload under test parks
     * instead of refusing (the pre-fix behaviour), it leaves a thread wedged on
     * the registry and the die still bound, which would poison anything after
     * it and turn one real failure into a cascade. */
    test_unload_reclaims_the_die(path, budget);
    test_transient_holder_is_drained(path, budget);
    test_pinned_unload_refuses(path, budget);

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
