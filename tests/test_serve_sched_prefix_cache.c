/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* PC-batched gate (multi-tenant plan Phase 4) — prefix caching in the batched
 * scheduler must be TOKEN-IDENTICAL to a non-cached reference.
 *
 * Setup: 4 concurrent USERS, each a distinct session_id, each running 3 TURNS
 * of a GROWING conversation (a fixed system prompt + the accumulated
 * user/assistant history + a new user line per turn).  All 4 users' turns are
 * interleaved through the batch scheduler concurrently.  With a session_id the
 * batched path delta-prefills the turn's appended tokens against the session's
 * warm slot (bs_slot_acquire_biased → bs_prefill_resume → bs_slot_finalize);
 * the whole point of the campaign.
 *
 * Correctness bar (argmax-exact, QWEN35_FUSE=0x37): for EVERY user and EVERY
 * turn, the cached batched reply must be byte-identical to a NON-CACHED
 * single-user reference (fresh pn_state, full prefill of the same accumulated
 * conversation).  Greedy (temp 0).  If any turn diverges the gate FAILS — a
 * silent-drift ship is exactly what this bars.
 *
 * The conversation grows turn over turn so the delta-prefill / warm-cache path
 * is genuinely exercised (turn 2/3 reuse turn 1/2's committed SSM+KV), which is
 * also the realistic steady-state the capacity benchmark measures.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_serve_sched_prefix_cache
 * Run:  QWEN35_BATCH_SLOTS=1 QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 \
 *         QWEN35_GPU_DIE_B=1 QWEN35_FUSE=0x37 \
 *         ./tests/test_serve_sched_prefix_cache <model.gguf> [max_tokens]
 * NOT part of the CPU test suite. */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/serve_generation.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define N_USERS 4
#define N_TURNS 3
#define SYS_PROMPT "You are a concise, helpful assistant."

/* The per-turn user lines for each user's growing conversation. */
static const char *user_line(int u, int t)
{
    static const char *lines[N_USERS][N_TURNS] = {
        { "Name three primary colors.",
          "Now name three secondary colors.",
          "Which of all those is your favorite, and why in one sentence?" },
        { "What is the capital of France?",
          "What is its most famous landmark?",
          "Roughly how tall is that landmark?" },
        { "Count from one to five.",
          "Now continue to ten.",
          "What is the sum of all those numbers?" },
        { "Name a planet in the solar system.",
          "Name the planet just inside its orbit.",
          "Which of the two is larger?" },
    };
    return lines[u][t];
}

/* A conversation: system + accumulated (user/assistant) turns.  Grows as
 * replies are appended.  Held as an sg_message array. */
typedef struct {
    sg_message msgs[1 + 2 * N_TURNS];   /* system + up to N_TURNS (user+asst) */
    int        n;
    char      *owned[1 + 2 * N_TURNS];  /* owned assistant-reply strings to free */
    int        n_owned;
} convo;

static void convo_init(convo *c)
{
    memset(c, 0, sizeof(*c));
    c->msgs[0].role = "system";
    c->msgs[0].content = SYS_PROMPT;
    c->n = 1;
}

static void convo_add_user(convo *c, const char *line)
{
    c->msgs[c->n].role = "user";
    c->msgs[c->n].content = line;
    c->n++;
}

/* Append an assistant reply (copied, owned by the convo). */
static void convo_add_assistant(convo *c, const char *reply)
{
    char *copy = strdup(reply ? reply : "");
    c->owned[c->n_owned++] = copy;
    c->msgs[c->n].role = "assistant";
    c->msgs[c->n].content = copy;
    c->n++;
}

static void convo_free(convo *c)
{
    for (int i = 0; i < c->n_owned; i++)
        free(c->owned[i]);
}

/* ── Cached batched worker: one user runs all N_TURNS through the scheduler
 * with its session_id, appending each reply so the next turn's prompt grows.
 * The replies are recorded for the gate compare. ── */
typedef struct {
    pn_model       *model;
    pn_batch_sched *sched;
    int             user;
    int             max_tokens;
    char            sid[32];
    char           *reply[N_TURNS];   /* owned copies of each turn's content */
    int             rc;
} worker_ctx;

static void run_turns(pn_model *model, pn_batch_sched *sched, int user,
                      const char *sid, int max_tokens, char *out_reply[N_TURNS],
                      int *out_rc)
{
    convo c;
    convo_init(&c);
    sg_gen_params gp;
    memset(&gp, 0, sizeof(gp));
    gp.temperature = 0.0f;   /* greedy — argmax-exact regime */
    gp.top_p = 1.0f;
    gp.top_k = 0;
    gp.max_tokens = max_tokens;

    for (int t = 0; t < N_TURNS; t++) {
        convo_add_user(&c, user_line(user, t));
        sg_result res;
        memset(&res, 0, sizeof(res));
        int slot = sid ? bs_slot_acquire_biased(sched, sid)
                       : bs_slot_acquire(sched);
        if (slot < 0) { *out_rc = -100; convo_free(&c); return; }
        int rc = sg_run_completion_batched(model, sched, slot, c.msgs, c.n,
                                           &gp, NULL, NULL, &res, sid);
        if (rc != 0) {
            *out_rc = rc;
            sg_result_free(&res);
            convo_free(&c);
            return;
        }
        out_reply[t] = strdup(res.content ? res.content : "");
        convo_add_assistant(&c, res.content ? res.content : "");
        sg_result_free(&res);
    }
    convo_free(&c);
    *out_rc = 0;
}

static void *worker(void *arg)
{
    worker_ctx *w = arg;
    run_turns(w->model, w->sched, w->user, w->sid, w->max_tokens,
              w->reply, &w->rc);
    return NULL;
}

/* Non-cached single-user reference: the SAME growing conversation, fresh
 * pn_state per turn, full prefill, no session id.  Uses sg_run_completion (the
 * established single-user reference path). */
static int reference_turns(pn_model *model, int user, int max_tokens,
                           char *out_reply[N_TURNS])
{
    convo c;
    convo_init(&c);
    sg_gen_params gp;
    memset(&gp, 0, sizeof(gp));
    gp.temperature = 0.0f;
    gp.top_p = 1.0f;
    gp.top_k = 0;
    gp.max_tokens = max_tokens;

    for (int t = 0; t < N_TURNS; t++) {
        convo_add_user(&c, user_line(user, t));
        pn_state *st = pn_state_new(model);
        if (!st) { convo_free(&c); return -1; }
        sg_result res;
        memset(&res, 0, sizeof(res));
        int rc = sg_run_completion(model, st, c.msgs, c.n, &gp,
                                   NULL, NULL, &res, NULL);
        pn_state_free(st);
        if (rc != 0) { sg_result_free(&res); convo_free(&c); return rc; }
        out_reply[t] = strdup(res.content ? res.content : "");
        convo_add_assistant(&c, res.content ? res.content : "");
        sg_result_free(&res);
    }
    convo_free(&c);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [max_tokens]\n", argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    int max_tokens = argc > 2 ? atoi(argv[2]) : 48;
    if (max_tokens < 4) max_tokens = 4;

    int failed = 0;

    /* ── References first (sequential, non-cached, fresh state per turn). ── */
    char *ref[N_USERS][N_TURNS];
    memset(ref, 0, sizeof(ref));
    for (int u = 0; u < N_USERS; u++) {
        if (reference_turns(model, u, max_tokens, ref[u]) != 0) {
            fprintf(stderr, "reference user %d failed\n", u);
            return 1;
        }
        for (int t = 0; t < N_TURNS; t++)
            fprintf(stderr, "ref u=%d t=%d: %.80s\n", u, t, ref[u][t]);
    }

    /* ── Cached batched run: 4 users concurrent, 3 growing turns each. ── */
    pn_batch_sched *sched = bs_create(model, N_USERS, NULL);
    if (!sched) { fprintf(stderr, "bs_create failed\n"); return 1; }
    fprintf(stderr, "\nPC-batched gate: %d users x %d turns, driver_sample=%d\n",
            N_USERS, N_TURNS, bs_driver_sampling(sched) ? 1 : 0);

    pthread_t th[N_USERS];
    worker_ctx w[N_USERS];
    memset(w, 0, sizeof(w));
    for (int u = 0; u < N_USERS; u++) {
        w[u].model = model; w[u].sched = sched;
        w[u].user = u; w[u].max_tokens = max_tokens;
        snprintf(w[u].sid, sizeof(w[u].sid), "sess-%d", u);
        pthread_create(&th[u], NULL, worker, &w[u]);
    }
    for (int u = 0; u < N_USERS; u++) pthread_join(th[u], NULL);

    /* ── Compare: every user, every turn, byte-identical to the reference. ── */
    for (int u = 0; u < N_USERS; u++) {
        if (w[u].rc != 0) {
            fprintf(stderr, "cached user %d rc=%d\n", u, w[u].rc);
            failed = 1;
            continue;
        }
        for (int t = 0; t < N_TURNS; t++) {
            const char *g = w[u].reply[t] ? w[u].reply[t] : "";
            const char *r = ref[u][t] ? ref[u][t] : "";
            if (strcmp(g, r) == 0) {
                fprintf(stderr, "PC u=%d t=%d: IDENTICAL (%zu chars)\n",
                        u, t, strlen(g));
            } else {
                fprintf(stderr, "PC u=%d t=%d: DIVERGED\n  ref: %.160s\n"
                        "  got: %.160s\n", u, t, r, g);
                failed = 1;
            }
        }
    }

    for (int u = 0; u < N_USERS; u++)
        for (int t = 0; t < N_TURNS; t++) {
            free(w[u].reply[t]);
            free(ref[u][t]);
        }
    bs_destroy(sched);

    fprintf(stderr, "\nGATE: %s\n", failed ? "FAIL" :
            "PASS (batched prefix-cached replies ≡ non-cached reference, "
            "argmax-exact across 4 users x 3 growing turns)");
    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(failed);
    return failed;
}
#endif
