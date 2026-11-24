/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_request_timing_line.c — the per-request [req] line carries every field
 * (timestamp, prompt / prefill tokens, TTFT, decode rate over the request's own
 * decode time, active-slot range, MTP, finalize wait, finish), the arrival mark
 * is consumed once, and bs_slot_release_stats hands back the slot's stats.
 */

#include "../src/serve/request_timing_line.h"
#include "../src/serve/batch_sched.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } else { \
        printf("  ok: %s\n", (msg)); \
    } \
} while (0)

static void test_line_fields(void)
{
    printf("line_fields:\n");
    setenv("TZ", "UTC", 1);
    tzset();
    sg_request_timing t = {
        .slot = 2, .prompt_tok = 12345, .prefill_tok = 1830, .ttft_ms = 10234.54,
        .generated = 201, .decode_ms = 5000.0, .active_min = 1, .active_max = 3,
        .mtp_cycles = 0, .finalize_ms = 0.42, .finish = "tool_calls",
    };
    struct timespec wall = { 1795097002, 123456789 };   /* 2026-11-19T14:03:22.123Z */
    char buf[384];
    int n = sg_request_timing_format(&t, &wall, buf, sizeof buf);
    printf("  %s\n", buf);
    const char *want =
        "[req] 2026-11-19T14:03:22.123+0000 slot=2 prompt=12345 prefill=1830 "
        "ttft_ms=10234.5 gen=201 decode_ms=5000.0 tok_s=40.0 active=1-3 mtp=no "
        "fin_ms=0.4 finish=tool_calls";
    CHECK(n > 0 && (size_t)n < sizeof buf, "fits the buffer");
    CHECK(strcmp(buf, want) == 0, "every field, in order, with the wall stamp");

    t.mtp_cycles = 57;
    sg_request_timing_format(&t, &wall, buf, sizeof buf);
    CHECK(strstr(buf, " mtp=yes ") != NULL, "MTP cycles -> mtp=yes");
}

static void test_tok_s(void)
{
    printf("tok_s:\n");
    sg_request_timing t = { .generated = 101, .decode_ms = 2500.0 };
    CHECK(fabs(sg_request_timing_tok_s(&t) - 40.0) < 1e-9,
          "tokens after the first over the request's decode time");
    t.generated = 1;
    CHECK(sg_request_timing_tok_s(&t) == 0.0, "one token: no decode to rate");
    t.generated = 10;
    t.decode_ms = 0.0;
    CHECK(sg_request_timing_tok_s(&t) == 0.0, "zero decode time: 0, not inf");
}

static void test_arrival_once(void)
{
    printf("arrival_once:\n");
    CHECK(sg_request_arrival_take() == 0.0, "no mark: 0");
    double before = sg_request_timing_now_ms();
    sg_request_arrival_mark();
    double t = sg_request_arrival_take();
    CHECK(t >= before && t <= sg_request_timing_now_ms(), "the mark is the arrival time");
    CHECK(sg_request_arrival_take() == 0.0, "taken once (a second choice starts fresh)");
}

static void test_release_stats(void)
{
    printf("release_stats:\n");
    pn_batch_sched sched;
    memset(&sched, 0, sizeof sched);
    pthread_mutex_init(&sched.mu, NULL);
    pthread_cond_init(&sched.cv_idle, NULL);
    pthread_cond_init(&sched.cv_work, NULL);
    sched.cap = 2;
    sched.slots[1].status = BS_SLOT_DECODE;
    sched.slots[1].finished = true;
    sched.slots[1].req_active_min = 2;
    sched.slots[1].req_active_max = 4;
    bs_request_stats st;
    memset(&st, 0xAA, sizeof st);
    bs_slot_release_stats(&sched, 1, &st);
    CHECK(st.active_min == 2 && st.active_max == 4, "active range read at release");
    CHECK(st.mtp_cycles == 0, "no MTP on this build/slot");
    CHECK(sched.slots[1].status == BS_SLOT_IDLE, "and the slot is released");
    memset(&st, 0xAA, sizeof st);
    bs_slot_release_stats(&sched, 1, &st);
    CHECK(st.active_min == 0 && st.active_max == 0, "already idle: zeroed");
    pthread_mutex_destroy(&sched.mu);
}

int main(void)
{
    printf("test_request_timing_line: the per-request [req] line\n");
    test_line_fields();
    test_tok_s();
    test_arrival_once();
    test_release_stats();
    if (g_failed) {
        fprintf(stderr, "test_request_timing_line: %d FAILED\n", g_failed);
        return 1;
    }
    printf("test_request_timing_line: all passed\n");
    return 0;
}
