/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_topk_tail.c
 * @brief Tail counts on the card (see pn_tt_topk_tail.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_topk_tail.h"

#include <string.h>

#include "ft_tt_op_sample.h"
#include "pn_tt_dispatch.h"
#include "pn_tt_device_internal.h"

bool pn_tt_topk_tail_prepare(pn_tt_device *device, uint32_t x, uint32_t y,
                             uint32_t result_address, struct ft_tt_op_topk_request *request)
{
    uint32_t words[FT_TT_OP_SAMPLE_TAIL_MAX_BINS];

    if (device->tail_bins == 0u)
        return true;
    if (ft_tt_op_sample_tail_keys_encode(device->tail_keys, device->tail_bins, words) != FT_TT_OK ||
        device->runtime.noc.write(device->runtime.noc.user, (uint16_t)x, (uint16_t)y,
                                  result_address + FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET * 4u, words,
                                  sizeof(words)) != FT_TT_OK) {
        device->tail_ok = false;
        return false;
    }
    request->flags |= FT_TT_OP_SAMPLE_FLAG_TAIL_STATS;
    request->tail_bins = device->tail_bins;
    return true;
}

void pn_tt_topk_tail_collect(pn_tt_device *device, uint32_t x, uint32_t y,
                             uint32_t result_address)
{
    uint32_t area[FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS];
    uint32_t counts[FT_TT_OP_SAMPLE_TAIL_MAX_BINS + 1u];
    uint32_t j;

    if (device->tail_bins == 0u)
        return;
    if (device->runtime.noc.read(device->runtime.noc.user, (uint16_t)x, (uint16_t)y,
                                 result_address + FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET * 4u, area,
                                 sizeof(area)) != FT_TT_OK ||
        ft_tt_op_sample_tail_counts_decode(area, device->tail_bins, counts) != FT_TT_OK) {
        device->tail_ok = false;
        return;
    }
    device->bytes_read += sizeof(area);
    for (j = 0u; j <= device->tail_bins; j++)
        device->tail_counts[j] += counts[j];
}

int pn_tt_topk_tail_counts(pn_tt_device *device, const pn_tt_activation *logits, uint32_t row,
                           const uint32_t *keys, uint32_t bins, uint32_t *counts)
{
    uint32_t index[1], produced = 0u, saved_row;
    float value[1];
    int rc;

    if (!device || !logits || !keys || !counts || bins == 0u || bins > PN_TT_TAIL_MAX_BINS ||
        row >= 32u)
        return -1;
    saved_row = device->topk_row;
    device->topk_row = row;
    device->tail_bins = bins;
    memcpy(device->tail_keys, keys, bins * sizeof(keys[0]));
    memset(device->tail_counts, 0, sizeof(device->tail_counts));
    device->tail_ok = true;
    rc = pn_tt_topk(device, logits, 1u, index, value, &produced);
    device->tail_bins = 0u;
    device->topk_row = saved_row;
    /* An extra descriptor on the top-k queues between two forwards: the dispatch replay's
     * learned per-token queue advance no longer holds — re-learn it (a replayed plain decode
     * otherwise DIVERGES on the next token's top-k). */
    pn_tt_dispatch_sequence_reset();
    if (rc != 0 || !device->tail_ok)
        return -1;
    memcpy(counts, device->tail_counts, (bins + 1u) * sizeof(counts[0]));
    /* RED `tail_counts_shift`: every count one bin LOWER (the mass looks further down the
     * tail than it is) — the crosscheck must see it and the bound is no longer one. */
    if (pn_tt_red("tail_counts_shift")) {
        uint32_t j;

        for (j = bins; j > 0u; j--) counts[j] += counts[j - 1u], counts[j - 1u] = 0u;
    }
    return 0;
}

#endif /* USE_TT */
