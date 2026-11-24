/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_bs_finalize_yield.c — a finished reply is not held behind another
 * session's prefill.
 *
 * The defect: bs_slot_finalize needs the GPU turn, and the owner sends its
 * reply (tool call included) only after finalize. bs_prefill_yield handed the
 * turn over between chunks only when a slot was still decoding, so a slot that
 * had hit EOS waited out the other session's WHOLE prefill (live: a ~6 s decode
 * returned after 32.8 s).
 *
 * Hand-built scheduler (no driver thread, no GPU): a prefiller thread holds the
 * turn across N chunks and calls the real bs_prefill_yield between them; a
 * finished, cache-armed slot calls the real bs_slot_finalize. The test records
 * how many chunks had run when finalize returned. Also checks the two cases
 * that must keep the turn: nobody waiting, and only another prefiller waiting.
 */

#include "../src/serve/batch_sched.h"
#include "../src/core/engine.h"

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
    } else { \
        printf("  ok: %s\n", (msg)); \
    } \
} while (0)

#define N_CHUNKS     6
#define CHUNK_US     20000
#define FIN_SLOT     1
#define COMMITTED_N  4

static pn_batch_sched g_sched;
static struct pn_state g_fin_state;
static volatile int g_chunks_done;

/* The turnstile is static in batch_sched.c; the prefiller and the queued
 * prefiller take/return a ticket the same way bs_prefill_begin / bind do. */
static void take_turn(void)
{
    pthread_mutex_lock(&g_sched.mu);
    uint64_t my = g_sched.gpu_ticket_next++;
    while (g_sched.gpu_ticket_serving != my)
        pthread_cond_wait(&g_sched.cv_gpu, &g_sched.mu);
    pthread_mutex_unlock(&g_sched.mu);
}

static void return_turn(void)
{
    pthread_mutex_lock(&g_sched.mu);
    g_sched.gpu_ticket_serving++;
    pthread_cond_broadcast(&g_sched.cv_gpu);
    pthread_mutex_unlock(&g_sched.mu);
}

/* Tickets handed out but not yet served, excluding the current holder. */
static int turn_waiters(void)
{
    pthread_mutex_lock(&g_sched.mu);
    int n = (int)(g_sched.gpu_ticket_next - g_sched.gpu_ticket_serving) - 1;
    pthread_mutex_unlock(&g_sched.mu);
    return n;
}

static void wait_for_turn_waiter(void)
{
    for (int i = 0; i < 2000 && turn_waiters() < 1; i++)
        usleep(1000);
}

static void sched_reset(void)
{
    memset(&g_sched, 0, sizeof g_sched);
    pthread_mutex_init(&g_sched.mu, NULL);
    pthread_cond_init(&g_sched.cv_gpu, NULL);
    pthread_cond_init(&g_sched.cv_work, NULL);
    pthread_cond_init(&g_sched.cv_idle, NULL);
    g_sched.cap = 2;
    for (int i = 0; i < g_sched.cap; i++) {
        g_sched.slots[i].slot_idx = i;
        g_sched.slots[i].gpu_idx = -1;
        pthread_cond_init(&g_sched.slots[i].cv, NULL);
    }
    g_sched.slots[0].status = BS_SLOT_PREFILL;

    /* Slot 1 hit EOS: the driver marked it finished; its owner is about to
     * finalize a cache-armed turn (committed log == state position). */
    bs_slot *fin = &g_sched.slots[FIN_SLOT];
    fin->status = BS_SLOT_DECODE;
    fin->finished = true;
    fin->gpu_idx = FIN_SLOT;
    memset(&g_fin_state, 0, sizeof g_fin_state);
    g_fin_state.pos = COMMITTED_N;
    fin->state = &g_fin_state;
    fin->ct = calloc(COMMITTED_N, sizeof(uint32_t));
    for (int i = 0; i < COMMITTED_N; i++)
        fin->ct[i] = (uint32_t)(100 + i);
    fin->ct_cap = COMMITTED_N;
    fin->ct_n = COMMITTED_N;
    snprintf(fin->armed_sid, sizeof fin->armed_sid, "%s", "agent-a");
    g_chunks_done = 0;
}

static void sched_free(void)
{
    bs_slot *fin = &g_sched.slots[FIN_SLOT];
    free(fin->ct);
    free(fin->hist);
    free(fin->vis);
    free(fin->vis_think);
}

/* Slot 0's prefill: N chunks under one turn, yielding between chunks (never
 * after the last — the bind follows it under the same turn). After chunk 0 it
 * waits for the other thread to queue, so the first yield sees it. */
static void *prefill_thread(void *arg)
{
    (void)arg;
    take_turn();
    for (int c = 0; c < N_CHUNKS; c++) {
        usleep(CHUNK_US);
        __atomic_store_n(&g_chunks_done, c + 1, __ATOMIC_SEQ_CST);
        if (c == 0)
            wait_for_turn_waiter();
        if (c < N_CHUNKS - 1)
            bs_prefill_yield(&g_sched, 0);
    }
    return_turn();   /* stands in for bs_slot_bind */
    return NULL;
}

typedef struct {
    int rc;
    int chunks_at_done;
} waiter_result;

static void *finalize_thread(void *arg)
{
    waiter_result *r = arg;
    r->rc = bs_slot_finalize(&g_sched, FIN_SLOT);
    r->chunks_at_done = __atomic_load_n(&g_chunks_done, __ATOMIC_SEQ_CST);
    return NULL;
}

/* Another session's prefill queued for the turn (bs_prefill_begin). */
static void *queued_prefill_thread(void *arg)
{
    waiter_result *r = arg;
    take_turn();
    r->chunks_at_done = __atomic_load_n(&g_chunks_done, __ATOMIC_SEQ_CST);
    return_turn();
    return NULL;
}

static void wait_prefill_started(void)
{
    for (int i = 0; i < 2000; i++) {
        pthread_mutex_lock(&g_sched.mu);
        int held = g_sched.gpu_ticket_next > 0;
        pthread_mutex_unlock(&g_sched.mu);
        if (held)
            return;
        usleep(1000);
    }
}

static void test_finished_reply_not_held(void)
{
    printf("finished_reply_not_held:\n");
    sched_reset();
    pthread_t pf, fz;
    waiter_result r = { -2, -1 };
    pthread_create(&pf, NULL, prefill_thread, NULL);
    wait_prefill_started();
    pthread_create(&fz, NULL, finalize_thread, &r);
    pthread_join(fz, NULL);
    pthread_join(pf, NULL);
    printf("  finalize returned after %d of %d chunks\n", r.chunks_at_done,
           N_CHUNKS);
    CHECK(r.rc == 0, "finalize publishes the turn");
    CHECK(r.chunks_at_done == 1,
          "finalize runs at the first chunk boundary, not after the prefill");
    CHECK(g_sched.slots[FIN_SLOT].hist_n == COMMITTED_N &&
          strcmp(g_sched.slots[FIN_SLOT].sid, "agent-a") == 0,
          "the committed history is published for the next turn");
    CHECK(g_sched.n_finalize_waiting == 0, "no finalizer left counted");
    CHECK(turn_waiters() == -1, "turnstile drained");
    sched_free();
}

static void test_queued_prefill_does_not_split(void)
{
    printf("queued_prefill_does_not_split:\n");
    sched_reset();
    pthread_t pf, qp;
    waiter_result r = { 0, -1 };
    pthread_create(&pf, NULL, prefill_thread, NULL);
    wait_prefill_started();
    pthread_create(&qp, NULL, queued_prefill_thread, &r);
    pthread_join(qp, NULL);
    pthread_join(pf, NULL);
    CHECK(r.chunks_at_done == N_CHUNKS,
          "a queued prefill waits for the whole prefill (no chunk ping-pong)");
    sched_free();
}

static void test_no_waiter_keeps_turn(void)
{
    printf("no_waiter_keeps_turn:\n");
    sched_reset();
    take_turn();
    for (int c = 0; c < 3; c++)
        bs_prefill_yield(&g_sched, 0);
    pthread_mutex_lock(&g_sched.mu);
    uint64_t next = g_sched.gpu_ticket_next;
    uint64_t serving = g_sched.gpu_ticket_serving;
    pthread_mutex_unlock(&g_sched.mu);
    CHECK(next == 1 && serving == 0, "nobody waiting: the turn is kept");
    return_turn();
    sched_free();
}

int main(void)
{
    printf("test_bs_finalize_yield: finished reply vs a long prefill\n");
    test_finished_reply_not_held();
    test_queued_prefill_does_not_split();
    test_no_waiter_keeps_turn();
    if (g_failed) {
        fprintf(stderr, "test_bs_finalize_yield: %d FAILED\n", g_failed);
        return 1;
    }
    printf("test_bs_finalize_yield: all passed\n");
    return 0;
}
