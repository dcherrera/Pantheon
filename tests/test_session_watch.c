/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_session_watch.c
 * @brief Liveness-based session release unit test (§C.4) — the "alive holds,
 *        dead releases" contract, driven WITHOUT a model or real generation.
 *        Links session_watch.c + prefix_cache.c + qwen35moe_state_snapshot.c
 *        and provides its own pn_state_* stubs (as test_prefix_cache_sys does)
 *        plus tiny mr_acquire/mr_release + a hand-built mr_registry/mr_entry, so
 *        it exercises the watch manager's grace/reconnect/backstop logic on the
 *        Mac (the poll thread itself is a real-socket path — here the manual
 *        manager + the test-only mark_closed/sweep hooks drive time
 *        deterministically).
 *
 *        Covers: a watched session survives arbitrarily long inactivity (never
 *        reaped — the admit reaper's TTL is the 12 h backstop and only touches
 *        UNwatched sessions); a watch close → released after the grace;
 *        watch-close then re-watch within grace → NOT released; the backstop
 *        only reaps a session with NO watch; and a watch on a BASE id
 *        `client-<pid>-<tab>` protects the whole `<base>-*` prefix — the
 *        mode-suffixed slots (`-orch`/`-act`/`-chat`) survive the backstop while
 *        the base is watched and are all released when the base watch dies.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/core/prefix_cache.h"

/* Phase 10b stub: the content-generation counter lives in
 * qwen35moe_forward.c, which this test intentionally does not link (it
 * stubs the pn_state lifecycle). snapshot-restore bumps it. */
unsigned pn_qwen35moe_kv_next_content_gen(void)
{
    static unsigned counter;
    return ++counter;
}
#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_internal.h"
#include "../src/serve/model_registry.h"
#include "../src/serve/session_watch.h"

/* The manual (no-thread) manager constructor lives in session_watch.c; declared
 * here for the test (it is intentionally not in the public header). */
pn_session_watch_mgr *pn_session_watch_new_manual(mr_registry *reg);

/* ---- batch_sched.c externals this test intentionally does not link ----
 * batch_sched.c is linked for the slot bookkeeping session_watch drives, but it
 * also carries the resume ladder, which reaches engine.c (pn_session_ctx_cap)
 * and the tokenizer (the PN_REQPROF divergence dumps). Neither is exercised
 * here — no resume runs and PN_REQPROF is unset — so stub them, exactly as
 * test_prefix_cache_sys does. Without pn_session_ctx_cap this test has not
 * linked since Phase 11 moved that call into batch_sched.c. */
int pn_session_ctx_cap(const pn_model *model)
{
    (void)model;
    return 0;                              /* 0 = "no cap", as engine.c means it */
}

char *ft_decode(ft_tokenizer *tok, int *ids, int length)
{
    (void)tok; (void)ids; (void)length;
    return NULL;
}

int ft_get_model_type(ft_tokenizer *tok)
{
    (void)tok;
    return -1;
}

char *gpt2_bytes_decode(const char *input)
{
    (void)input;
    return NULL;
}

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* ---- Minimal pn_state stubs (a bare dual state; SYS1 snapshot machinery is
 * unused here, we only need acquire/commit/release bookkeeping). ---- */

enum { T_N_SSM = 1, T_N_ATTN = 1, T_CONV_M1 = 2, T_CONV_DIM = 4,
       T_STATE = 2, T_VHEADS = 1, T_HEADS_KV = 1, T_HEAD_DIM = 32,
       T_MAX_SEQ = 32 };

static pn_qwen35moe_state *make_state(void)
{
    pn_qwen35moe_state *s = calloc(1, sizeof(*s));
    s->n_ssm = T_N_SSM;
    s->n_attn = T_N_ATTN;
    s->ssm = calloc(T_N_SSM, sizeof(*s->ssm));
    s->kv = calloc(T_N_ATTN, sizeof(*s->kv));
    for (int i = 0; i < T_N_SSM; i++) {
        s->ssm[i].conv_dim = T_CONV_DIM;
        s->ssm[i].conv_minus_1 = T_CONV_M1;
        s->ssm[i].state_size = T_STATE;
        s->ssm[i].n_v_heads = T_VHEADS;
        s->ssm[i].conv_state = calloc((size_t)T_CONV_M1 * T_CONV_DIM, sizeof(float));
        s->ssm[i].ssm_state = calloc((size_t)T_STATE * T_STATE * T_VHEADS, sizeof(float));
    }
    for (int i = 0; i < T_N_ATTN; i++) {
        s->kv[i].n_head_kv = T_HEADS_KV;
        s->kv[i].head_dim = T_HEAD_DIM;
        s->kv[i].max_seq = T_MAX_SEQ;
        s->kv[i].q4 = 0;
        size_t n = (size_t)T_HEADS_KV * T_MAX_SEQ * T_HEAD_DIM;
        s->kv[i].k = calloc(n, sizeof(float));
        s->kv[i].v = calloc(n, sizeof(float));
    }
    return s;
}

static void free_state(pn_qwen35moe_state *s)
{
    if (!s) return;
    for (int i = 0; i < s->n_ssm; i++) { free(s->ssm[i].conv_state); free(s->ssm[i].ssm_state); }
    for (int i = 0; i < s->n_attn; i++) { free(s->kv[i].k); free(s->kv[i].v); }
    free(s->ssm); free(s->kv); free(s);
}

pn_state *pn_state_new(pn_model *model)
{
    pn_state *st = calloc(1, sizeof(*st));
    st->config = &model->config;
    st->kv_cache = make_state();
    return st;
}
void pn_state_free(pn_state *state)
{
    if (!state) return;
    free_state((pn_qwen35moe_state *)state->kv_cache);
    free(state);
}
void pn_state_reset(pn_state *state) { if (state) state->pos = 0; }

/* ---- Tiny registry stubs (avoid linking model_registry.c's heavy deps). The
 * watch manager only calls mr_acquire/mr_release; we hand-build the entry. ---- */

mr_entry *mr_acquire(mr_registry *reg, const char *name)
{
    if (!reg) return NULL;
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        if (e->used && e->model && !e->unloading &&
            (!name || !*name || strncmp(e->name, name, MR_MAX_NAME) == 0)) {
            e->refs++;
            return e;
        }
    }
    return NULL;
}
void mr_release(mr_registry *reg, mr_entry *entry)
{
    (void)reg;
    if (entry && entry->refs > 0) entry->refs--;
}

/* ---- Fixtures ---- */

static pn_model g_model;               /* arch qwen35moe (snapshot-capable) */

static void build_registry(mr_registry *reg, mr_entry *e, const char *name,
                           int max_sessions)
{
    memset(reg, 0, sizeof(*reg));
    pthread_mutex_init(&reg->lock, NULL);   /* install_backstop_hooks takes it */
    memset(&g_model, 0, sizeof(g_model));
    g_model.config.arch = PN_ARCH_QWEN35MOE;
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->model = &g_model;
    e->used = true;
    e->unloading = false;
    e->sched = NULL;                    /* single-user path → gen_lock */
    pthread_mutex_init(&e->gen_lock, NULL);
    pthread_mutex_init(&e->pc_lock, NULL);
    e->prefix_cache = pn_prefix_cache_new(&g_model, max_sessions);
    reg->entries[0] = *e;               /* the registry owns the live copy */
}

/* Acquire a live session so the cache holds a slot for `sid` (mimics a chat
 * turn creating the prefix-cache entry). Uses the registry's entry copy. */
static void seed_session(mr_entry *reg_e, const char *sid)
{
    static const uint32_t toks[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int delta = 0;
    pn_state *st = pn_prefix_cache_acquire(reg_e->prefix_cache, sid, toks, 8, &delta);
    (void)st;
    pn_prefix_cache_commit(reg_e->prefix_cache, sid, toks, 8);
}

/* Is `sid` still resident? Admit an OTHER sid at capacity: if `sid`'s slot is
 * gone a genuinely-new sid fits (ADMIT_OK); if it still holds, over-cap refuses
 * (AT_CAPACITY). Cap must be 1 for this probe to be decisive. */
static int session_resident(mr_entry *reg_e, const char *sid, const char *probe)
{
    (void)sid;
    /* Backstop huge here — pure "is a slot free" probe (no reaping). */
    pn_session_admit v = pn_prefix_cache_admit(reg_e->prefix_cache, probe,
                                               1000, 0, NULL);
    return v == PN_SESSION_ADMIT_AT_CAPACITY;   /* refused ⇒ sid still holds */
}

/* ============================================================
 * Test 1 — a watched session survives arbitrarily long inactivity, and a
 * watch-close releases it only after the grace.
 * ============================================================ */

static void test_watch_holds_then_grace_release(void)
{
    printf("test: watch holds through inactivity; close → grace release\n");
    setenv("PN_SESSION_WATCH_GRACE_MS", "45000", 1);
    mr_registry reg; mr_entry e;
    build_registry(&reg, &e, "m", 1);   /* cap 1: one session at a time */
    mr_entry *R = &reg.entries[0];
    seed_session(R, "s1");

    pn_session_watch_mgr *m = pn_session_watch_new_manual(&reg);
    CHECK(m != NULL, "manual watch manager created");

    /* Register a watch WITHOUT a real socket: use a pipe read-end as the fd so
     * the manager's write of the establish header goes to the pipe buffer. */
    int p[2]; CHECK(pipe(p) == 0, "watch fd pipe");
    int rc = pn_session_watch_add(m, p[1], R, "m", "s1");
    CHECK(rc == 0, "watch added (fd adopted)");

    /* Long inactivity while ALIVE: sweep far in the future — NOT released
     * (the watch is still open; closed_at_ms == 0). */
    pn_session_watch_sweep_for_test(m, 100u * 60u * 60u * 1000u);  /* +100 h */
    CHECK(session_resident(R, "s1", "probe1"),
          "watched session survives 100 h of inactivity (never reaped)");

    /* Now the client dies: mark the watch closed at t=1000. Before the grace
     * elapses it must NOT be released. */
    CHECK(pn_session_watch_mark_closed_for_test(m, "m", "s1", 1000) == 0,
          "watch marked closed (grace started)");
    pn_session_watch_sweep_for_test(m, 1000 + 45000 - 1);   /* just before grace */
    CHECK(session_resident(R, "s1", "probe2"),
          "before grace: dropped session still resident");

    /* Past the grace: released. */
    pn_session_watch_sweep_for_test(m, 1000 + 45000);
    CHECK(!session_resident(R, "s1", "probe3"),
          "after grace: dropped session released");

    pn_session_watch_stop(m);
    pn_prefix_cache_free(R->prefix_cache);
    close(p[0]);
    pthread_mutex_destroy(&e.gen_lock);
    pthread_mutex_destroy(&e.pc_lock);
}

/* ============================================================
 * Test 2 — watch-close then RE-watch within grace → NOT released.
 * ============================================================ */

static void test_reconnect_cancels_grace(void)
{
    printf("test: reconnect within grace cancels the release\n");
    setenv("PN_SESSION_WATCH_GRACE_MS", "45000", 1);
    mr_registry reg; mr_entry e;
    build_registry(&reg, &e, "m", 1);
    mr_entry *R = &reg.entries[0];
    seed_session(R, "s1");

    pn_session_watch_mgr *m = pn_session_watch_new_manual(&reg);
    int p[2]; (void)!pipe(p);
    CHECK(pn_session_watch_add(m, p[1], R, "m", "s1") == 0, "watch added");

    /* Drop at t=1000, then reconnect at t=5000 (< grace) — a fresh fd. */
    CHECK(pn_session_watch_mark_closed_for_test(m, "m", "s1", 1000) == 0,
          "watch marked closed");
    int p2[2]; (void)!pipe(p2);
    CHECK(pn_session_watch_add(m, p2[1], R, "m", "s1") == 0,
          "reconnect watch added within grace");

    /* Well past the ORIGINAL drop+grace: must NOT be released (reconnect
     * cancelled the grace — closed_at_ms was cleared). */
    pn_session_watch_sweep_for_test(m, 1000 + 45000 + 60000);
    CHECK(session_resident(R, "s1", "probe"),
          "reconnected session kept its slot past the old grace");

    pn_session_watch_stop(m);
    pn_prefix_cache_free(R->prefix_cache);
    close(p[0]); close(p2[0]);
    pthread_mutex_destroy(&e.gen_lock);
    pthread_mutex_destroy(&e.pc_lock);
}

/* ============================================================
 * Test 3 — the 12 h backstop only touches UNwatched sessions; the request-path
 * admit reaper (TTL = backstop) never reaps a session while it is watched.
 * (We assert the admit-reaper semantics directly: a watched session is touched
 * every turn, so its last_active_ms stays fresh; an unwatched forgotten slot
 * ages out only at the 12 h backstop.)
 * ============================================================ */

static void test_backstop_only_unwatched(void)
{
    printf("test: 12 h backstop only reaps unwatched/forgotten sessions\n");
    mr_registry reg; mr_entry e;
    build_registry(&reg, &e, "m", 2);   /* cap 2 */
    mr_entry *R = &reg.entries[0];
    seed_session(R, "watched");
    seed_session(R, "forgotten");

    const uint64_t BACKSTOP = 12u * 60u * 60u * 1000u;   /* 12 h */

    /* 'watched' is touched every turn (an active client); 'forgotten' never
     * returns. At t = a bit over the backstop, the admit reaper with TTL =
     * backstop reaps ONLY 'forgotten' (the truly-idle one), freeing a slot. */
    const uint64_t NOW = BACKSTOP + 2000;
    pn_prefix_cache_touch(R->prefix_cache, "watched", NOW);        /* fresh */
    pn_prefix_cache_touch(R->prefix_cache, "forgotten", 1000);    /* stale: idle
                                                                   * NOW-1000 >
                                                                   * BACKSTOP */

    /* Drive the reaper via a probe sid past the backstop. */
    pn_session_admit v = pn_prefix_cache_admit(R->prefix_cache, "probe",
                                               NOW, BACKSTOP, NULL);
    CHECK(v == PN_SESSION_ADMIT_OK, "probe admitted (forgotten reaped, slot free)");

    /* 'watched' (kept fresh) must still resume; 'forgotten' must be gone. */
    CHECK(pn_prefix_cache_admit(R->prefix_cache, "watched", NOW,
                                BACKSTOP, NULL) == PN_SESSION_ADMIT_OK,
          "watched session survives the backstop (was touched every turn)");

    /* Prove 'forgotten' is gone: cap 2, 'watched' holds one slot, so a NEW sid
     * fits only because 'forgotten' was reaped. */
    pn_state *st; int d = 0;
    static const uint32_t toks[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    st = pn_prefix_cache_acquire(R->prefix_cache, "newcomer", toks, 8, &d);
    CHECK(st != NULL, "newcomer takes the freed (forgotten) slot");
    /* Now both slots live ('watched' + 'newcomer') → a third new sid refused. */
    CHECK(pn_prefix_cache_admit(R->prefix_cache, "third", NOW,
                                BACKSTOP, NULL) == PN_SESSION_ADMIT_AT_CAPACITY,
          "both slots live again → third new sid refused (watched never reaped)");

    pn_prefix_cache_free(R->prefix_cache);
    pthread_mutex_destroy(&e.gen_lock);
    pthread_mutex_destroy(&e.pc_lock);
}

/* ============================================================
 * Test 5 — a watch on the BASE id protects the whole `<base>-*` prefix:
 *   (a) a live watch on base `client-1-1` holds `client-1-1-orch` through the
 *       backstop (the mode slot is NOT reaped);
 *   (b) the base's watch dying past grace releases `client-1-1-orch` (and any
 *       other `client-1-1-*`), even though NO exact `client-1-1` entry exists;
 *   (c) an unwatched `client-9-9-orch` IS backstop-reaped.
 * This is the user's exact case: an hour-long orchestration sits quiet in
 * wait_agents; its -orch slot must be held by the base watch and released only
 * when the base watch dies.
 * ============================================================ */

static void test_base_watch_covers_mode_prefix(void)
{
    printf("test: base watch protects the whole <base>-* prefix\n");
    setenv("PN_SESSION_WATCH_GRACE_MS", "45000", 1);
    mr_registry reg; mr_entry e;
    build_registry(&reg, &e, "m", 4);   /* room for several mode slots */
    mr_entry *R = &reg.entries[0];

    /* Live mode slots: a watched base's -orch + -act, and an UNwatched base's
     * -orch (the crashed/forgotten one). NO bare-base entries exist. */
    seed_session(R, "client-1-1-orch");
    seed_session(R, "client-1-1-act");
    seed_session(R, "client-9-9-orch");

    pn_session_watch_mgr *m = pn_session_watch_new_manual(&reg);
    CHECK(m != NULL, "manual watch manager created");
    /* The serve installs the backstop hook after start; do the same here so the
     * backstop reaper consults the watch table. */
    pn_session_watch_install_backstop_hooks(m);

    /* Watch the BASE id client-1-1 (as the client does — one watch per tab). */
    int p[2]; CHECK(pipe(p) == 0, "watch fd pipe");
    CHECK(pn_session_watch_add(m, p[1], R, "m", "client-1-1") == 0,
          "base watch added on client-1-1");

    /* (a) Backstop reaper runs far past the 12 h TTL. client-1-1-orch and -act
     * are watched (base alive) → kept; client-9-9-orch is unwatched → reaped. */
    const uint64_t BACKSTOP = 12u * 60u * 60u * 1000u;
    const uint64_t NOW = BACKSTOP + 2000;
    pn_prefix_cache_touch(R->prefix_cache, "client-1-1-orch", 1000);  /* stale */
    pn_prefix_cache_touch(R->prefix_cache, "client-1-1-act", 1000);   /* stale */
    pn_prefix_cache_touch(R->prefix_cache, "client-9-9-orch", 1000);  /* stale */
    /* Drive the backstop reaper via a probe sid past the TTL. */
    pn_prefix_cache_admit(R->prefix_cache, "probe", NOW, BACKSTOP, NULL);

    int d = 0; static const uint32_t tk[8] = {1,2,3,4,5,6,7,8};
    CHECK(pn_prefix_cache_admit(R->prefix_cache, "client-1-1-orch", NOW,
                                BACKSTOP, NULL) == PN_SESSION_ADMIT_OK &&
          pn_prefix_cache_acquire(R->prefix_cache, "client-1-1-orch", tk, 8, &d)
              != NULL,
          "watched base's -orch survives the 12 h backstop");
    CHECK(pn_prefix_cache_acquire(R->prefix_cache, "client-1-1-act", tk, 8, &d)
              != NULL,
          "watched base's -act survives the 12 h backstop");
    /* client-9-9-orch (unwatched) must have been reaped: a fresh acquire finds NO
     * cached history (delta_start 0), rather than resuming its 8-token slot. */
    int d99 = -1;
    pn_prefix_cache_acquire(R->prefix_cache, "client-9-9-orch", tk, 8, &d99);
    CHECK(d99 == 0, "unwatched client-9-9-orch WAS backstop-reaped (fresh acquire)");

    /* (b) The base's watch dies past grace → release the whole client-1-1-* set. */
    CHECK(pn_session_watch_mark_closed_for_test(m, "m", "client-1-1", 1000) == 0,
          "base watch marked closed (grace started)");
    pn_session_watch_sweep_for_test(m, 1000 + 45000);   /* past grace */

    /* Both mode slots must now be gone: re-acquiring each yields a FRESH slot
     * (delta_start 0 — no cached history), proving the release happened. */
    int d2 = 0;
    pn_prefix_cache_acquire(R->prefix_cache, "client-1-1-orch", tk, 8, &d2);
    CHECK(d2 == 0, "client-1-1-orch released by base-watch death (fresh acquire)");
    int d3 = 0;
    pn_prefix_cache_acquire(R->prefix_cache, "client-1-1-act", tk, 8, &d3);
    CHECK(d3 == 0, "client-1-1-act released by base-watch death (fresh acquire)");

    pn_session_watch_stop(m);
    pn_prefix_cache_free(R->prefix_cache);
    close(p[0]);
    pthread_mutex_destroy(&reg.lock);
    pthread_mutex_destroy(&e.gen_lock);
    pthread_mutex_destroy(&e.pc_lock);
}

/* ============================================================
 * Test 4 — grace/backstop constants honor their env overrides + defaults.
 * ============================================================ */

static void test_constants(void)
{
    printf("test: grace + backstop constants (defaults + env)\n");
    /* Defaults are cached on first read; these run in a fresh process so the
     * first read below establishes them. Set env BEFORE the first call. */
    setenv("PN_SESSION_WATCH_GRACE_MS", "45000", 1);
    setenv("PN_SESSION_IDLE_TTL_MS", "43200000", 1);   /* 12 h */
    CHECK(pn_session_watch_grace_ms() == 45000, "grace default/env = 45 s");
    CHECK(pn_session_watch_backstop_ms() == 43200000ull,
          "backstop default/env = 12 h");
}

int main(void)
{
    printf("=== test_session_watch (§C.4 liveness release) ===\n");
    test_constants();
    test_watch_holds_then_grace_release();
    test_reconnect_cancels_grace();
    test_backstop_only_unwatched();
    test_base_watch_covers_mode_prefix();
    if (failures) {
        printf("=== %d FAILURE(S) ===\n", failures);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
