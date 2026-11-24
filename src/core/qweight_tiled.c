/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qweight_tiled.c
 * @brief  Implementation of the tiled matvec / matmul (decode a tile column, dot its rows).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "qweight_tiled.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/simd.h"
#include "../formats/gguf_tiled.h"

#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

typedef struct {
    const pn_qweight *w;
    const float *acts;
    int B;
    float *out;
} tiled_task;

/**
 * `PN_TILED_ACT_BF16=1`: round the INPUT rows to bf16 (nearest-even) before the dots — the card
 * feeds activations to the FPU as bf16, so this oracle arm predicts the routing / argmax noise
 * floor card arithmetic alone produces, before any card run (the bar of plan 4.4).
 */
static int tiled_act_bf16(void)
{
    static int on = -1;

    if (on < 0)
        on = getenv("PN_TILED_ACT_BF16") && atoi(getenv("PN_TILED_ACT_BF16")) == 1;
    return on;
}

static float tiled_round_bf16(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

/** Tile columns [start, end): decode each, then every row of it against every input row. */
static void tiled_columns(int start, int end, void *arg)
{
    const tiled_task *t = arg;
    const pn_qweight *w = t->w;
    float dots[PN_TILED_DIM];
    float *panel, *scratch;

    /* One input row on a fused type: dot each column straight from its tiles (no panel). */
    if (t->B == 1 && start < end &&
        pn_tiled_dot_column(w->type, (const unsigned char *)w->data + (size_t)start * PN_TILED_DIM * w->row_stride,
                            w->cols, t->acts, dots) == 0) {
        for (int nt = start; nt < end; nt++) {
            const int r0 = nt * PN_TILED_DIM;
            const int valid = w->rows - r0 < PN_TILED_DIM ? w->rows - r0 : PN_TILED_DIM;

            if (nt > start)
                pn_tiled_dot_column(w->type, (const unsigned char *)w->data + (size_t)r0 * w->row_stride, w->cols,
                                    t->acts, dots);
            memcpy(t->out + r0, dots, (size_t)valid * sizeof(float));
        }
        return;
    }
    panel = malloc((size_t)PN_TILED_DIM * (size_t)w->cols * sizeof(float));
    scratch = malloc((size_t)PN_TILED_DIM * (size_t)w->cols * sizeof(float));
    if (!panel || !scratch) {
        fprintf(stderr, "qweight_tiled: out of memory for a %d-wide tile column\n", w->cols);
        free(panel);
        free(scratch);
        return;
    }
    for (int nt = start; nt < end; nt++) {
        const int r0 = nt * PN_TILED_DIM;
        const unsigned char *column = (const unsigned char *)w->data + (size_t)r0 * w->row_stride;
        /* An N that is not whole tiles: the last column's rows past w->rows are the
         * converter's zero padding (gguf_fttw_overlay.c). Decode it whole, keep the valid rows. */
        const int valid = w->rows - r0 < PN_TILED_DIM ? w->rows - r0 : PN_TILED_DIM;

        if (pn_tiled_decode_column(w->type, column, w->cols, panel, scratch) != 0)
            break;
        for (int i = 0; i < valid; i++)
            for (int b = 0; b < t->B; b++)
                t->out[(size_t)b * (size_t)w->rows + (size_t)(r0 + i)] =
                    pn_dot_f32(panel + (size_t)i * (size_t)w->cols, t->acts + (size_t)b * (size_t)w->cols, w->cols);
    }
    free(panel);
    free(scratch);
}

void pn_qmm_tiled(const pn_qweight *w, const float *acts, int B, float *out, struct fw_pool *pool)
{
    tiled_task task = { w, acts, B, out };
    float *rounded = NULL;
    int columns;

    if (!w || !acts || !out || B <= 0)
        return;
    if (w->rows <= 0 || w->cols % PN_TILED_DIM != 0) {
        fprintf(stderr, "qweight_tiled: [%d x %d]: K is not whole tiles — nothing computed\n", w->rows,
                w->cols);
        return;
    }
    columns = (w->rows + PN_TILED_DIM - 1) / PN_TILED_DIM;
    if (tiled_act_bf16()) {
        rounded = malloc((size_t)B * (size_t)w->cols * sizeof(float));
        if (!rounded)
            return;
        for (size_t i = 0; i < (size_t)B * (size_t)w->cols; i++)
            rounded[i] = tiled_round_bf16(acts[i]);
        task.acts = rounded;
    }
#ifdef USE_THREADS
    if (pool && columns > 1)
        fw_for((fw_pool *)pool, columns, tiled_columns, &task);
    else
        tiled_columns(0, columns, &task);
#else
    (void)pool;
    tiled_columns(0, columns, &task);
#endif
    free(rounded);
}
