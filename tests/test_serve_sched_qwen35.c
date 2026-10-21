/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Serve-scheduler gate for the L1/L2/L3 serving package (multi-tenant plan
 * Phase 1) — exercises sg_run_completion_batched through a real
 * pn_batch_sched with concurrent request threads, against the single-user
 * sg_run_completion reference.  Greedy (temp 0) → every batched user's TEXT
 * must be IDENTICAL to its single-user reference.
 *
 * Phases:
 *   1. PARITY   — K users with different prompts run CONCURRENTLY through
 *      the scheduler (batched+chunked prefill, in-driver sampling, dynamic
 *      join/leave as threads start/finish).  Content == reference, per user.
 *   2. JOIN     — 3 users decode; mid-decode a 4th joins with a LONG prompt
 *      (exercises the L2 chunk-yield path).  All 4 contents == reference;
 *      reports the 3 decoders' max inter-token gap during the join (M5-like,
 *      in-process).
 *   3. SLOW     — 4 users; user 0's delta callback sleeps SLOW_MS per delta
 *      (a stalled SSE client).  All contents == reference; reports fast
 *      users' max gap.  With L3 (default) a stalled owner must NOT stall
 *      the batch: fast max gap ≪ SLOW_MS stall; hard-FAIL if the fast gap
 *      exceeds SLOW_GATE_MS (generous; lockstep would exceed it hugely).
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_serve_sched_qwen35
 * Run:  QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *         QWEN35_FUSE=0x37 ./tests/test_serve_sched_qwen35 <model.gguf>
 * QWEN35_FUSE=0x37 makes the single-token decode path float-identical to
 * the batched kernels (same convention as test_users_qwen35; the strict
 * text gate runs under it).  NOT part of the CPU test suite. */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/serve_generation.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define MAX_USERS 8

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static const char *user_text(int u)
{
    static const char *texts[MAX_USERS] = {
        "Name three primary colors.",
        "What is the capital of France?",
        "Count from one to five.",
        "Name a planet in the solar system.",
        "What sound does a cat make?",
        "Name one ocean.",
        "What is two plus two?",
        "Name a season of the year.",
    };
    return texts[u % MAX_USERS];
}

/* Per-request context: runs one batched completion, records per-delta
 * timestamps (max inter-delta gap in a window) and optionally stalls. */
typedef struct {
    pn_model       *model;
    pn_batch_sched *sched;
    const char     *text;       /* user message */
    int             max_tokens;
    int             stall_ms;   /* sleep per delta (slow-client simulation) */
    double          t_start;    /* first-delta reference */
    double          t_last;     /* last delta time */
    double          max_gap_ms; /* max inter-delta gap */
    int             n_deltas;
    sg_result       res;
    int             rc;
} req_ctx;

static void delta_cb(const char *text, int is_reasoning, void *ud)
{
    (void)text; (void)is_reasoning;
    req_ctx *rc = ud;
    double t = now_ms();
    if (rc->n_deltas > 0) {
        double gap = t - rc->t_last;
        if (gap > rc->max_gap_ms) rc->max_gap_ms = gap;
    } else {
        rc->t_start = t;
    }
    rc->t_last = t;
    rc->n_deltas++;
    if (rc->stall_ms > 0) {
        struct timespec ts = { rc->stall_ms / 1000,
                               (long)(rc->stall_ms % 1000) * 1000000L };
        nanosleep(&ts, NULL);
    }
}

static void *req_thread(void *arg)
{
    req_ctx *rc = arg;
    sg_message msg = { "user", rc->text };
    sg_gen_params gp;
    memset(&gp, 0, sizeof(gp));
    gp.temperature = 0.0f;          /* greedy — the bit-exact gate regime */
    gp.top_p       = 1.0f;
    gp.top_k       = 0;
    gp.max_tokens  = rc->max_tokens;

    int slot = bs_slot_acquire(rc->sched);
    if (slot < 0) { rc->rc = -100; return NULL; }
    rc->rc = sg_run_completion_batched(rc->model, rc->sched, slot, &msg, 1,
                                       &gp, delta_cb, rc, &rc->res, NULL);
    return NULL;
}

/* Single-user reference for one prompt (fresh state, greedy). */
static int reference(pn_model *model, const char *text, int max_tokens,
                     sg_result *res)
{
    sg_message msg = { "user", text };
    sg_gen_params gp;
    memset(&gp, 0, sizeof(gp));
    gp.temperature = 0.0f;
    gp.top_p       = 1.0f;
    gp.top_k       = 0;
    gp.max_tokens  = max_tokens;
    pn_state *st = pn_state_new(model);
    if (!st) return -1;
    int rc = sg_run_completion(model, st, &msg, 1, &gp, NULL, NULL, res, NULL);
    pn_state_free(st);
    return rc;
}

static int check_match(const char *what, int u, const sg_result *got,
                       const sg_result *ref)
{
    const char *g = got->content ? got->content : "";
    const char *r = ref->content ? ref->content : "";
    const char *gr = got->reasoning ? got->reasoning : "";
    const char *rr = ref->reasoning ? ref->reasoning : "";
    if (strcmp(g, r) == 0 && strcmp(gr, rr) == 0) {
        fprintf(stderr, "%s u=%d: IDENTICAL (%d tokens)\n", what, u,
                got->completion_tokens);
        return 0;
    }
    fprintf(stderr, "%s u=%d: DIVERGED\n  ref reasoning: %.120s\n  got "
            "reasoning: %.120s\n  ref content: %.120s\n  got content: %.120s\n",
            what, u, rr, gr, r, g);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [K] [max_tokens]\n", argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    int K = argc > 2 ? atoi(argv[2]) : 4;
    if (K < 2) K = 2;
    if (K > MAX_USERS) K = MAX_USERS;
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;
    if (max_tokens < 4) max_tokens = 4;

    int failed = 0;

    pn_batch_sched *sched = bs_create(model, MAX_USERS);
    if (!sched) { fprintf(stderr, "bs_create failed\n"); return 1; }
    fprintf(stderr, "serve-sched gate: K=%d max_tokens=%d driver_sample=%d "
            "chunk=%s\n", K, max_tokens, bs_driver_sampling(sched) ? 1 : 0,
            getenv("QWEN35_BS_PREFILL_CHUNK") ?
                getenv("QWEN35_BS_PREFILL_CHUNK") : "(default 128)");

    /* Long prompt for the JOIN phase: filler + question (exercises several
     * prefill chunks). */
    char *long_text = malloc(8192);
    if (!long_text) return 1;
    long_text[0] = '\0';
    for (int i = 0; i < 40; i++)
        strcat(long_text, "The quick brown fox jumps over the lazy dog "
                          "while the calm river flows east past the mill. ");
    strcat(long_text, "After all of that: what is the capital of France?");

    /* ---- References (single-user path, sequential). ---- */
    sg_result refs[MAX_USERS];
    memset(refs, 0, sizeof(refs));
    for (int u = 0; u < K; u++) {
        if (reference(model, user_text(u), max_tokens, &refs[u]) != 0) {
            fprintf(stderr, "reference %d failed\n", u);
            return 1;
        }
        fprintf(stderr, "ref u=%d: %d tokens\n", u, refs[u].completion_tokens);
    }
    sg_result ref_long;
    memset(&ref_long, 0, sizeof(ref_long));
    if (reference(model, long_text, max_tokens, &ref_long) != 0) {
        fprintf(stderr, "long reference failed\n");
        return 1;
    }
    fprintf(stderr, "ref long-prompt: %d prompt tokens, %d out tokens\n",
            ref_long.prompt_tokens, ref_long.completion_tokens);

    /* ---- Phase 1: PARITY — K concurrent users through the scheduler. ---- */
    {
        fprintf(stderr, "\n-- phase 1: parity (K=%d concurrent) --\n", K);
        pthread_t th[MAX_USERS];
        req_ctx rq[MAX_USERS];
        memset(rq, 0, sizeof(rq));
        for (int u = 0; u < K; u++) {
            rq[u].model = model; rq[u].sched = sched;
            rq[u].text = user_text(u); rq[u].max_tokens = max_tokens;
            pthread_create(&th[u], NULL, req_thread, &rq[u]);
        }
        for (int u = 0; u < K; u++) pthread_join(th[u], NULL);
        for (int u = 0; u < K; u++) {
            if (rq[u].rc != 0) { fprintf(stderr, "parity u=%d rc=%d\n", u, rq[u].rc); failed = 1; }
            else failed |= check_match("parity", u, &rq[u].res, &refs[u]);
            sg_result_free(&rq[u].res);
        }
    }

    /* ---- Phase 2: JOIN — 3 decode, a 4th joins with the long prompt. ---- */
    {
        fprintf(stderr, "\n-- phase 2: join-during-decode (long prompt) --\n");
        pthread_t th[4];
        req_ctx rq[4];
        memset(rq, 0, sizeof(rq));
        for (int u = 0; u < 3; u++) {
            rq[u].model = model; rq[u].sched = sched;
            rq[u].text = user_text(u); rq[u].max_tokens = max_tokens;
            pthread_create(&th[u], NULL, req_thread, &rq[u]);
        }
        /* Let the 3 reach steady decode, then join the long prompt. */
        usleep(2500 * 1000);
        rq[3].model = model; rq[3].sched = sched;
        rq[3].text = long_text; rq[3].max_tokens = max_tokens;
        pthread_create(&th[3], NULL, req_thread, &rq[3]);
        for (int u = 0; u < 4; u++) pthread_join(th[u], NULL);
        double worst = 0.0;
        for (int u = 0; u < 3; u++)
            if (rq[u].max_gap_ms > worst) worst = rq[u].max_gap_ms;
        fprintf(stderr, "join: decoders' max inter-token gap %.0f ms "
                "(prompt %d tokens)\n", worst, rq[3].res.prompt_tokens);
        for (int u = 0; u < 4; u++) {
            const sg_result *ref = (u < 3) ? &refs[u] : &ref_long;
            if (rq[u].rc != 0) { fprintf(stderr, "join u=%d rc=%d\n", u, rq[u].rc); failed = 1; }
            else failed |= check_match("join", u, &rq[u].res, ref);
            sg_result_free(&rq[u].res);
        }
    }

    /* ---- Phase 3: SLOW — one stalled client must not stall the rest. ---- */
    {
        enum { SLOW_MS = 1500, SLOW_GATE_MS = 1200 };
        fprintf(stderr, "\n-- phase 3: slow-client isolation (stall %d ms) --\n",
                (int)SLOW_MS);
        pthread_t th[4];
        req_ctx rq[4];
        memset(rq, 0, sizeof(rq));
        for (int u = 0; u < 4; u++) {
            rq[u].model = model; rq[u].sched = sched;
            rq[u].text = user_text(u);
            rq[u].max_tokens = max_tokens < 24 ? max_tokens : 24;
            if (u == 0) rq[u].stall_ms = SLOW_MS;
            pthread_create(&th[u], NULL, req_thread, &rq[u]);
        }
        for (int u = 0; u < 4; u++) pthread_join(th[u], NULL);
        double worst = 0.0;
        for (int u = 1; u < 4; u++)
            if (rq[u].max_gap_ms > worst) worst = rq[u].max_gap_ms;
        fprintf(stderr, "slow: fast users' max inter-token gap %.0f ms "
                "(stalled user emitted %d deltas)\n", worst, rq[0].n_deltas);
        for (int u = 0; u < 4; u++) {
            sg_result ref_slow;
            memset(&ref_slow, 0, sizeof(ref_slow));
            if (reference(model, user_text(u),
                          rq[u].max_tokens, &ref_slow) != 0) { failed = 1; }
            if (rq[u].rc != 0) { fprintf(stderr, "slow u=%d rc=%d\n", u, rq[u].rc); failed = 1; }
            else failed |= check_match("slow", u, &rq[u].res, &ref_slow);
            sg_result_free(&rq[u].res);
            sg_result_free(&ref_slow);
        }
        if (bs_driver_sampling(sched) && worst > (double)SLOW_GATE_MS) {
            fprintf(stderr, "slow: FAIL — fast users stalled (%.0f ms > %d ms "
                    "gate): the stalled client is on the critical path\n",
                    worst, (int)SLOW_GATE_MS);
            failed = 1;
        }
    }

    for (int u = 0; u < K; u++) sg_result_free(&refs[u]);
    sg_result_free(&ref_long);
    free(long_text);
    bs_destroy(sched);

    fprintf(stderr, "\nGATE: %s\n", failed ? "FAIL" : "PASS (batched serve "
            "text ≡ single-user reference; slow client isolated)");
    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(failed);
    return failed;
}
#endif
