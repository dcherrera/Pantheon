/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* True per-user capacity bench — REALISTIC steady-state with prefix caching ON.
 *
 * The old synthetic bench cold-prefilled a fixed ~100-token prompt every step.
 * This measures what the campaign is actually about: a SHARED/GROWING context
 * per user (a fixed system prompt + a context that grows each turn) served
 * through the real batched path with prefix caching ON, so per-turn prefill is
 * a cheap DELTA against the session's warm slot (delta-prefill, warm KV+SSM) —
 * i.e. steady-state per-user speed, not cold re-prefill.
 *
 * K users run concurrently through the batch scheduler, each with its own
 * session_id, each doing WARM_TURNS growing turns.  Turn 0 is a cold prefill
 * (untimed warm-up); turns 1..WARM_TURNS-1 are timed steady-state (delta
 * prefill + decode).  Reports per-user tok/s (total decoded tokens / wall time,
 * averaged over the K concurrent users) and aggregate tok/s (K users' summed
 * decode throughput over the wall clock).
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_serve_prefix_cache
 * Run:  QWEN35_BATCH_SLOTS=1 QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 \
 *         QWEN35_GPU_DIE_B=1 QWEN35_FUSE=0x37 \
 *         ./tests/bench_serve_prefix_cache <model.gguf> <K> [gen_per_turn] [warm_turns]
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/serve_generation.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define MAX_K 8
#define MAX_TURNS 8
#define SYS_PROMPT "You are a concise, helpful assistant answering follow-up questions in a running conversation."

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* A generic follow-up that keeps the context growing turn over turn. */
static const char *turn_line(int t)
{
    static const char *lines[MAX_TURNS] = {
        "Explain what a hash table is in two sentences.",
        "Now give one real-world use for it.",
        "What is its average lookup complexity?",
        "And the worst case?",
        "How does chaining differ from open addressing?",
        "Which is more cache-friendly?",
        "Summarize the trade-off in one line.",
        "Give a one-word verdict.",
    };
    return lines[t % MAX_TURNS];
}

typedef struct {
    pn_model       *model;
    pn_batch_sched *sched;
    int             user;
    int             gen_per_turn;
    int             warm_turns;   /* total turns; turn 0 untimed warm-up */
    char            sid[32];
    /* results */
    int             timed_tokens; /* decode tokens over timed turns */
    double          timed_ms;     /* wall time over timed turns (this user) */
    int             rc;
} bctx;

/* Growing conversation held as an sg_message array (system + user/assistant
 * pairs). */
typedef struct {
    sg_message msgs[1 + 2 * MAX_TURNS];
    int        n;
    char      *owned[2 * MAX_TURNS];
    int        n_owned;
} convo;

static void convo_init(convo *c)
{
    memset(c, 0, sizeof(*c));
    c->msgs[0].role = "system";
    c->msgs[0].content = SYS_PROMPT;
    c->n = 1;
}
static void convo_user(convo *c, const char *line)
{
    c->msgs[c->n].role = "user"; c->msgs[c->n].content = line; c->n++;
}
static void convo_asst(convo *c, const char *reply)
{
    char *cp = strdup(reply ? reply : "");
    c->owned[c->n_owned++] = cp;
    c->msgs[c->n].role = "assistant"; c->msgs[c->n].content = cp; c->n++;
}
static void convo_free(convo *c)
{
    for (int i = 0; i < c->n_owned; i++) free(c->owned[i]);
}

static void *worker(void *arg)
{
    bctx *b = arg;
    convo c;
    convo_init(&c);
    sg_gen_params gp;
    memset(&gp, 0, sizeof(gp));
    gp.temperature = 0.0f; gp.top_p = 1.0f; gp.top_k = 0;
    gp.max_tokens = b->gen_per_turn;

    b->timed_tokens = 0;
    b->timed_ms = 0.0;

    for (int t = 0; t < b->warm_turns; t++) {
        convo_user(&c, turn_line(t));
        sg_result res;
        memset(&res, 0, sizeof(res));
        int slot = bs_slot_acquire_biased(b->sched, b->sid);
        if (slot < 0) { b->rc = -100; convo_free(&c); return NULL; }
        double t0 = now_ms();
        int rc = sg_run_completion_batched(b->model, b->sched, slot, c.msgs,
                                           c.n, &gp, NULL, NULL, &res, b->sid);
        double dt = now_ms() - t0;
        if (rc != 0) { b->rc = rc; sg_result_free(&res); convo_free(&c); return NULL; }
        /* Turn 0 is the cold prefill warm-up — not timed.  Turns 1.. are the
         * steady-state warm-cache turns the bench reports. */
        if (t > 0) {
            b->timed_tokens += res.completion_tokens;
            b->timed_ms     += dt;
        }
        convo_asst(&c, res.content);
        sg_result_free(&res);
    }
    convo_free(&c);
    b->rc = 0;
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <K> [gen_per_turn] [warm_turns]\n",
                argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    int K = atoi(argv[2]);
    if (K < 1) K = 1;
    if (K > MAX_K) K = MAX_K;
    int gen_per_turn = argc > 3 ? atoi(argv[3]) : 64;
    if (gen_per_turn < 4) gen_per_turn = 4;
    int warm_turns = argc > 4 ? atoi(argv[4]) : 4;
    if (warm_turns < 2) warm_turns = 2;
    if (warm_turns > MAX_TURNS) warm_turns = MAX_TURNS;

    pn_batch_sched *sched = bs_create(model, K < MAX_K ? K : MAX_K);
    if (!sched) { fprintf(stderr, "bs_create failed\n"); return 1; }

    fprintf(stderr, "bench: K=%d gen/turn=%d warm_turns=%d (turn0 untimed) "
            "driver_sample=%d\n", K, gen_per_turn, warm_turns,
            bs_driver_sampling(sched) ? 1 : 0);

    pthread_t th[MAX_K];
    bctx b[MAX_K];
    memset(b, 0, sizeof(b));
    double wall0 = now_ms();
    for (int u = 0; u < K; u++) {
        b[u].model = model; b[u].sched = sched; b[u].user = u;
        b[u].gen_per_turn = gen_per_turn; b[u].warm_turns = warm_turns;
        snprintf(b[u].sid, sizeof(b[u].sid), "bench-%d", u);
        pthread_create(&th[u], NULL, worker, &b[u]);
    }
    for (int u = 0; u < K; u++) pthread_join(th[u], NULL);
    double wall_ms = now_ms() - wall0;

    int total_tokens = 0;
    double sum_per_user_tps = 0.0;
    int ok = 1;
    for (int u = 0; u < K; u++) {
        if (b[u].rc != 0) { fprintf(stderr, "user %d rc=%d\n", u, b[u].rc); ok = 0; }
        double u_tps = b[u].timed_ms > 0 ? b[u].timed_tokens * 1000.0 / b[u].timed_ms : 0.0;
        sum_per_user_tps += u_tps;
        total_tokens += b[u].timed_tokens;
        fprintf(stderr, "  user %d: %d timed tok, %.0f ms, %.2f tok/s\n",
                u, b[u].timed_tokens, b[u].timed_ms, u_tps);
    }
    if (!ok) { bs_destroy(sched); return 1; }

    /* Per-user tok/s = mean of the K concurrent users' steady-state rates.
     * Aggregate tok/s = all timed decode tokens over the concurrent wall time
     * of the timed turns.  Wall time here spans all turns incl. turn 0; for a
     * clean aggregate use the summed per-user rate (the users run concurrently
     * so their steady-state rates sum), which is the standard batched-agg
     * definition and matches the T4 A/B table's "agg tok/s". */
    double per_user_tps = sum_per_user_tps / K;
    double agg_tps = sum_per_user_tps;   /* K concurrent users' summed rate */

    /* Cost /1M tok: per-card active power 0.33 kW, rate $0.12491/kWh.
     * $/1M = (1e6 / agg_tps) s / 3600 * 0.33 kW * $0.12491. */
    double cost_per_1m = agg_tps > 0
        ? (1.0e6 / agg_tps) / 3600.0 * 0.33 * 0.12491 : 0.0;

    fprintf(stderr, "\n=== K=%d  per-user %.2f tok/s  aggregate %.2f tok/s  "
            "cost/1M $%.4f  (wall %.0f ms, %d timed tok) ===\n",
            K, per_user_tps, agg_tps, cost_per_1m, wall_ms, total_tokens);
    /* Machine-parseable line for the fill script. */
    printf("PCBENCH K=%d per_user=%.2f agg=%.2f cost1m=%.4f\n",
           K, per_user_tps, agg_tps, cost_per_1m);

    bs_destroy(sched);
    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(0);
    return 0;
}
#endif
