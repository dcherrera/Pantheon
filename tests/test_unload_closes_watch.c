/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_unload_closes_watch.c — an open /session/watch must not block an unload.
 *
 * The defect: Argus Desktop's Unload was refused whenever a chat tab to that
 * model was open — "model is busy: 2 reference(s) in flight after 15000 ms".
 * Each open GET /session/watch held an mr_acquire reference for the life of its
 * connection, the unload waited 15 s for references to drain, and a watch only
 * let go 45 s after it closed. So the client's own idle watches pinned the model.
 *
 * The contract under test (real registry + real watch manager + real poll
 * thread, a socketpair standing in for the client's watch connection):
 *   1. idle watch   — unload succeeds, the card is freed, the watch gets
 *                     `event: model_unloaded` and is closed; an open watch
 *                     never counts as a live reference.
 *   2. in-grace     — a watch the client already closed (45 s grace running)
 *                     does not block an unload either.
 *   3. live turn    — finishes within the wait: unload waits for it, no cancel.
 *   4. overstaying  — cancelled after the wait; unload then succeeds.
 *   5. stuck turn   — ignores the cancel: unload refuses, the model and its
 *                     watch are untouched, refs return to 0, a later unload
 *                     succeeds.
 *
 * Model: models/qwen3-0.6b-Q8_0.gguf (override with PN_TEST_MODEL). Logical
 * dies only — no GPU needed.
 */

#include "../src/serve/model_registry.h"
#include "../src/serve/session_watch.h"
#include "../src/accel/die_manager.h"

#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } else { \
        printf("  PASS: %s\n", (msg)); \
    } \
} while (0)

#define MIB(n) ((uint64_t)(n) << 20)
#define WATCH_SID "argus-86384-2-chat"

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

/* The registry row for `name` (refs + watch_refs), or -1s when absent. */
static void model_refs(mr_registry *reg, const char *name, int *refs, int *pins)
{
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);

    *refs = *pins = -1;
    for (int i = 0; i < n; i++)
        if (strcmp(info[i].name, name) == 0) {
            *refs = info[i].refs;
            *pins = info[i].watch_refs;
        }
}

/* Read what the server sends on the client end of a watch until EOF or
 * `budget_ms`. Returns 1 on EOF (server closed), 0 on timeout. */
static int read_watch_until_close(int fd, char *buf, size_t cap, int budget_ms)
{
    size_t used = 0;

    buf[0] = '\0';
    for (int waited = 0; waited < budget_ms; waited += 20) {
        struct pollfd p = { .fd = fd, .events = POLLIN };
        if (poll(&p, 1, 20) <= 0)
            continue;
        ssize_t n = read(fd, buf + used, cap - 1 - used);
        if (n <= 0)
            return 1;
        used += (size_t)n;
        buf[used] = '\0';
    }
    return 0;
}

/* Drain the `: watch established` header so later reads see only new bytes. */
static void read_established(int fd)
{
    char buf[512];
    struct pollfd p = { .fd = fd, .events = POLLIN };

    if (poll(&p, 1, 1000) > 0) {
        ssize_t n = read(fd, buf, sizeof(buf));
        (void)n;
    }
}

/* Open a watch the way Argus does: a held connection for one chat tab. Returns
 * the client end (the server end is adopted by the manager), or -1. */
static int open_watch(mr_registry *reg, pn_session_watch_mgr *mgr,
                      const char *name)
{
    int sv[2];
    mr_entry *e = mr_acquire(reg, name);

    if (!e || socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        if (e)
            mr_release(reg, e);
        return -1;
    }
    int rc = pn_session_watch_add(mgr, sv[0], e, e->name, WATCH_SID);
    mr_release(reg, e);
    if (rc != 0) {
        close(sv[0]);
        close(sv[1]);
        return -1;
    }
    read_established(sv[1]);
    return sv[1];
}

static int load_alpha(mr_registry *reg)
{
    char err[160] = "";
    int rc = mr_load(reg, getenv("PN_TEST_MODEL") ? getenv("PN_TEST_MODEL")
                     : "models/qwen3-0.6b-Q8_0.gguf", "alpha", NULL, NULL, 0,
                     0, 0, 0, err, sizeof(err));
    if (rc != 0)
        fprintf(stderr, "  load failed: %s\n", err);
    return rc;
}

static void set_budgets(const char *drain_ms, const char *cancel_ms)
{
    setenv("PN_UNLOAD_DRAIN_MS", drain_ms, 1);
    setenv("PN_UNLOAD_CANCEL_MS", cancel_ms, 1);
}

/* --- 1 + 2: watches never block an unload -------------------------------- */

static void test_idle_watch_unloads(mr_registry *reg, pn_session_watch_mgr *mgr)
{
    char buf[512], err[160] = "";
    int refs, pins;

    printf("idle_watch_unloads:\n");
    CHECK(load_alpha(reg) == 0, "load 'alpha'");
    int cfd = open_watch(reg, mgr, "alpha");
    CHECK(cfd >= 0, "watch opened on alpha");
    model_refs(reg, "alpha", &refs, &pins);
    CHECK(refs == 0, "an idle watch is not a live reference (refs == 0)");
    CHECK(pins == 1, "the watch is counted as a pin");

    set_budgets("500", "500");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0,
          "unload with an idle watch open succeeds");
    if (err[0])
        fprintf(stderr, "  unload said: %s\n", err);
    CHECK(die0_used() == 0, "the card (die budget) is freed");
    int closed = cfd >= 0 && read_watch_until_close(cfd, buf, sizeof(buf), 2000);
    CHECK(strstr(buf, "event: model_unloaded") != NULL,
          "the watch got the terminal model_unloaded event");
    CHECK(strstr(buf, "\"model\":\"alpha\"") != NULL, "the event names the model");
    CHECK(closed, "the server closed the watch");
    if (cfd >= 0)
        close(cfd);

    CHECK(load_alpha(reg) == 0, "alpha loads again on the freed card");
    model_refs(reg, "alpha", &refs, &pins);
    CHECK(refs == 0 && pins == 0, "the reloaded model starts at refs 0, pins 0");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0, "clean unload");
}

static void test_in_grace_watch_unloads(mr_registry *reg,
                                        pn_session_watch_mgr *mgr)
{
    char err[160] = "";
    int refs, pins;

    printf("in_grace_watch_unloads:\n");
    CHECK(load_alpha(reg) == 0, "load 'alpha'");
    int cfd = open_watch(reg, mgr, "alpha");
    CHECK(cfd >= 0, "watch opened on alpha");
    if (cfd >= 0)
        close(cfd);                     /* client closed the tab: grace starts */
    nap_ms(400);                        /* poll thread sees the hang-up */
    model_refs(reg, "alpha", &refs, &pins);
    CHECK(refs == 0, "a closed watch in grace holds no live reference");

    set_budgets("500", "500");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0,
          "unload with a watch in its 45 s grace succeeds at once");
    CHECK(die0_used() == 0, "the card is freed");
}

/* --- 3-5: live turns ------------------------------------------------------ */

typedef struct {
    mr_registry *reg;
    mr_entry    *entry;
    int          max_ms;        /* longest the turn runs on its own */
    int          obeys_cancel;  /* polls mr_turn_cancel_requested like a turn */
    int          saw_cancel;
    int          done;
} live_turn;

static void *live_turn_thread(void *arg)
{
    live_turn *t = arg;

    for (int ms = 0; ms < t->max_ms; ms += 10) {
        if (t->obeys_cancel && mr_turn_cancel_requested(t->entry)) {
            t->saw_cancel = 1;
            break;
        }
        nap_ms(10);
    }
    __atomic_store_n(&t->done, 1, __ATOMIC_RELEASE);
    mr_release(t->reg, t->entry);
    return NULL;
}

static int start_turn(live_turn *t, pthread_t *tid, mr_registry *reg,
                      int max_ms, int obeys_cancel)
{
    memset(t, 0, sizeof(*t));
    t->reg = reg;
    t->entry = mr_acquire(reg, "alpha");
    t->max_ms = max_ms;
    t->obeys_cancel = obeys_cancel;
    return t->entry && pthread_create(tid, NULL, live_turn_thread, t) == 0 ? 0 : -1;
}

static void test_turn_finishes_within_wait(mr_registry *reg,
                                           pn_session_watch_mgr *mgr)
{
    (void)mgr;
    live_turn t;
    pthread_t tid;
    char err[160] = "";

    printf("turn_finishes_within_wait:\n");
    CHECK(load_alpha(reg) == 0, "load 'alpha'");
    CHECK(start_turn(&t, &tid, reg, 300, 1) == 0, "live turn running");
    set_budgets("3000", "3000");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0,
          "unload waits for the turn and succeeds");
    pthread_join(tid, NULL);
    CHECK(t.done, "the turn finished before the model was freed");
    CHECK(!t.saw_cancel, "a turn that finishes in time is not cancelled");
}

static void test_overstaying_turn_cancelled(mr_registry *reg,
                                            pn_session_watch_mgr *mgr)
{
    live_turn t;
    pthread_t tid;
    char buf[512], err[160] = "";

    printf("overstaying_turn_cancelled:\n");
    CHECK(load_alpha(reg) == 0, "load 'alpha'");
    int cfd = open_watch(reg, mgr, "alpha");
    CHECK(start_turn(&t, &tid, reg, 20000, 1) == 0, "long live turn running");
    set_budgets("200", "3000");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0,
          "unload cancels the overstaying turn and succeeds");
    pthread_join(tid, NULL);
    CHECK(t.saw_cancel, "the turn saw the cancel");
    CHECK(die0_used() == 0, "the card is freed");
    CHECK(cfd >= 0 && read_watch_until_close(cfd, buf, sizeof(buf), 2000) &&
          strstr(buf, "event: model_unloaded") != NULL,
          "the watch got model_unloaded and was closed");
    if (cfd >= 0)
        close(cfd);
}

static void test_stuck_turn_refuses(mr_registry *reg, pn_session_watch_mgr *mgr)
{
    live_turn t;
    pthread_t tid;
    char buf[512], err[160] = "";
    int refs, pins;

    printf("stuck_turn_refuses:\n");
    CHECK(load_alpha(reg) == 0, "load 'alpha'");
    uint64_t held = die0_used();
    int cfd = open_watch(reg, mgr, "alpha");
    CHECK(start_turn(&t, &tid, reg, 1500, 0) == 0, "turn that ignores cancel");
    set_budgets("100", "200");
    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) != 0,
          "unload refuses when the turn will not stop");
    CHECK(strstr(err, "busy") != NULL, "the refusal says the model is busy");
    CHECK(die0_used() == held, "the refused unload leaves the card intact");
    struct pollfd p = { .fd = cfd, .events = POLLIN };
    CHECK(cfd >= 0 && poll(&p, 1, 100) == 0,
          "the refused unload leaves the watch open and silent");

    pthread_join(tid, NULL);
    model_refs(reg, "alpha", &refs, &pins);
    CHECK(refs == 0, "refs return to 0 once the turn ends");
    CHECK(pins == 1, "the watch pin is still held");
    mr_entry *e = mr_acquire(reg, "alpha");
    CHECK(e && !mr_turn_cancel_requested(e), "the cancel flag was rolled back");
    if (e)
        mr_release(reg, e);

    CHECK(mr_unload(reg, "alpha", err, sizeof(err)) == 0,
          "the next unload succeeds");
    CHECK(cfd >= 0 && read_watch_until_close(cfd, buf, sizeof(buf), 2000) &&
          strstr(buf, "event: model_unloaded") != NULL,
          "and only then the watch gets model_unloaded");
    if (cfd >= 0)
        close(cfd);
}

typedef void (*watch_arm_fn)(mr_registry *reg, pn_session_watch_mgr *mgr);

static void run_arm(watch_arm_fn arm)
{
    static mr_registry reg;

    if (mr_init(&reg) != 0) {
        CHECK(0, "registry init");
        return;
    }
    pn_session_watch_mgr *mgr = pn_session_watch_start(&reg);
    CHECK(mgr != NULL, "watch manager started");
    if (mgr)
        arm(&reg, mgr);
    pn_session_watch_stop(mgr);
    mr_free(&reg);
}

int main(void)
{
    printf("test_unload_closes_watch\n");
    const char *path = getenv("PN_TEST_MODEL") ? getenv("PN_TEST_MODEL")
                                               : "models/qwen3-0.6b-Q8_0.gguf";
    if (access(path, R_OK) != 0) {
        printf("[SKIP] %s not found (set PN_TEST_MODEL)\n", path);
        return 0;
    }
    setenv("PN_SESSION_WATCH_GRACE_MS", "45000", 1);
    if (pn_dm_init_logical(1, MIB(32768)) != 0) {
        printf("FAILED: logical die init\n");
        return 1;
    }
    /* Each arm gets a fresh registry + watch manager, so one arm's failure
     * (a model left resident) cannot make the next arm pass or fail. */
    run_arm(test_idle_watch_unloads);
    run_arm(test_in_grace_watch_unloads);
    run_arm(test_turn_finishes_within_wait);
    run_arm(test_overstaying_turn_cancelled);
    run_arm(test_stuck_turn_refuses);

    pn_dm_shutdown();
    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
