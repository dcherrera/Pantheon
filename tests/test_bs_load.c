/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_bs_load.c — bs_get_load() unit test (dispatch layer Phase G0).
 *
 * bs_get_load() is a pure read of a pn_batch_sched's slot states + queue depth
 * under sched->mu (no engine, no GPU). This test hand-builds a pn_batch_sched
 * on the stack with known slot statuses / n_queued and asserts the reported
 * {capacity, active_slots, free_slots, queue_depth}, plus the sched==NULL
 * single-user case (capacity 1 / free 1). It links batch_sched.c (which needs
 * the engine for its OTHER functions), but exercises only this pure accessor.
 */

#include "../src/serve/batch_sched.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

int main(void)
{
    fprintf(stderr, "test_bs_load: bs_get_load slot/queue snapshot\n");

    /* --- sched == NULL: single-user path presents 1 always-free slot --- */
    {
        bs_load l;
        memset(&l, 0xAA, sizeof l);   /* poison to prove every field is written */
        bs_get_load(NULL, &l);
        CHECK(l.capacity == 1,     "NULL sched: capacity == 1");
        CHECK(l.active_slots == 0, "NULL sched: active_slots == 0");
        CHECK(l.free_slots == 1,   "NULL sched: free_slots == 1");
        CHECK(l.queue_depth == 0,  "NULL sched: queue_depth == 0");
    }

    /* --- hand-built scheduler, cap=4, mixed slot states, 2 queued --- */
    {
        pn_batch_sched sched;
        memset(&sched, 0, sizeof sched);
        pthread_mutex_init(&sched.mu, NULL);
        sched.cap      = 4;
        sched.n_queued = 2;
        /* 3 busy (PREFILL/BOUND/DECODE all count), 1 idle. */
        sched.slots[0].status = BS_SLOT_PREFILL;
        sched.slots[1].status = BS_SLOT_BOUND;
        sched.slots[2].status = BS_SLOT_DECODE;
        sched.slots[3].status = BS_SLOT_IDLE;

        bs_load l;
        memset(&l, 0xAA, sizeof l);
        bs_get_load(&sched, &l);
        CHECK(l.capacity == 4,     "cap=4: capacity == 4");
        CHECK(l.active_slots == 3, "cap=4: 3 non-idle slots busy");
        CHECK(l.free_slots == 1,   "cap=4: 1 free slot");
        CHECK(l.queue_depth == 2,  "cap=4: queue_depth reflects n_queued");

        /* all idle → fully free, no queue */
        for (int i = 0; i < 4; i++) sched.slots[i].status = BS_SLOT_IDLE;
        sched.n_queued = 0;
        bs_get_load(&sched, &l);
        CHECK(l.active_slots == 0, "all idle: active_slots == 0");
        CHECK(l.free_slots == 4,   "all idle: free_slots == cap");
        CHECK(l.queue_depth == 0,  "all idle: queue_depth == 0");

        /* all busy → zero free */
        for (int i = 0; i < 4; i++) sched.slots[i].status = BS_SLOT_DECODE;
        bs_get_load(&sched, &l);
        CHECK(l.active_slots == 4, "all busy: active_slots == cap");
        CHECK(l.free_slots == 0,   "all busy: free_slots == 0");

        pthread_mutex_destroy(&sched.mu);
    }

    if (g_failed) {
        fprintf(stderr, "test_bs_load: %d CHECK(s) FAILED\n", g_failed);
        return 1;
    }
    fprintf(stderr, "test_bs_load: OK\n");
    return 0;
}
