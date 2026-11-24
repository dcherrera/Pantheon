/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_qwen35_tp_forget.c
 * @brief pn_qwen35moe_gpu_forget_state under QWEN35_TP drops BOTH rank halves' bindings to
 *        the host state (SSM host_state, KV host id + synced watermark).
 *
 * Under TP the layer arrays are doubled (rank 1's halves at [i + tp_n_*]). Dropping rank 0's
 * alone (the code before this test) left rank 1 seeding its GDN heads from the shared
 * buffers' stale prompt-end state on a session's turn 2: the greedy reply diverged from the
 * layer split (k80-q38-mtp-accept, two-turn serve check). CUDA build only; no GPU is touched
 * (pn_qwen35moe_tp_forget_selftest builds a fake layout on the host).
 * Red build: CFLAGS+=-DQW_FORGET_RED (rank 0's half only) must fail.
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
    printf("test_qwen35_tp_forget: skipped (built without USE_CUDA)\n");
    return 0;
#else
    int left = pn_qwen35moe_tp_forget_selftest();

    if (left != 0) {
        printf("FAIL: %d bindings left after forget_state on a TP layout (rank 1 kept)\n", left);
        return 1;
    }
    printf("test_qwen35_tp_forget: all passed (both TP halves dropped)\n");
    return 0;
#endif
}
