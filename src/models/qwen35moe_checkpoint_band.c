/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_checkpoint_band.c
 * @brief Checkpoint position arithmetic (see qwen35moe_checkpoint_band.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "qwen35moe_checkpoint_band.h"

int pn_qwen35_ckpt_band_at(int base, int batch_end, int ceiling)
{
    int limit = batch_end;
    int at;

    if (ceiling > 0 && ceiling < limit)
        limit = ceiling;
    if (limit <= base)
        return 0;
    at = base + ((limit - base) / PN_QWEN35_CKPT_BAND_ROWS) * PN_QWEN35_CKPT_BAND_ROWS;
    return at > base ? at : 0;
}
