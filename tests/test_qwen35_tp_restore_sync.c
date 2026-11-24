/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_qwen35_tp_restore_sync.c
 * @brief Under QWEN35_TP, a host restore (sys-prefix snapshot, prefix-cache turn, warm tier)
 *        followed by a resume at pos0 > 0 syncs BOTH rank halves' KV rings from the mirror.
 *
 * The two halves read one shared host kv. The restore bumps content_gen, so no GPU ring is a
 * D2D source and both halves must upload from the (fresh) mirror. The TP walk used to run one
 * rank at a time: rank 0 synced, attended and marked the mirror stale (lazy mode), then rank 1
 * refused the "stale" mirror ("refusing stale upload") and the request failed with HTTP 500
 * after every Argus compaction (seen on a K80 host, 2026-11-20). CUDA build with one die; skips otherwise.
 * Red build: CFLAGS+=-DQW_TP_SYNC_RED (no both-rank pre-sync) must fail.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../src/models/qwen35moe.h"

#include <stdio.h>

int main(void)
{
#ifndef USE_CUDA
    printf("test_qwen35_tp_restore_sync: skipped (built without USE_CUDA)\n");
    return 0;
#else
    int bad = pn_qwen35moe_tp_restore_sync_selftest();

    if (bad == -2) {
        printf("test_qwen35_tp_restore_sync: skipped (no CUDA device)\n");
        return 0;
    }
    if (bad != 0) {
        printf("FAIL: %d (refused syncs + ring bytes off the host mirror) after a TP restore\n",
               bad);
        return 1;
    }
    printf("test_qwen35_tp_restore_sync: all passed (both TP halves synced after a restore)\n");
    return 0;
#endif
}
