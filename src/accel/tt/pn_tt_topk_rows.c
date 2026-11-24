/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_topk_rows.c
 * @brief Multi-row top-k and the split's result read + exact merge. See pn_tt_topk_rows.h.
 * @status None
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_topk_rows.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_op_sample.h"
#include "pn_tt_device_internal.h"

uint32_t pn_tt_topk_helper_area(uint32_t result_address)
{
    static int helpers = -1;

    if (helpers < 0)
        helpers = (int)pn_tt_env_u32("PN_TT_TOPK_HELPERS", 1u);
    if (!helpers || result_address == 0u)
        return 0u;
    return result_address + FT_TT_OP_SAMPLE_MAX_ROWS * FT_TT_OP_SAMPLE_RESULT_BYTES;
}

uint32_t pn_tt_topk_order_key(float value)
{
    uint32_t bits;
    uint32_t u;

    memcpy(&bits, &value, sizeof(bits));
    u = bits >> 16;
    return (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
}

int pn_tt_topk_read_result(pn_tt_device *device, uint32_t x, uint32_t y, uint32_t address,
                           uint32_t *block, size_t bytes)
{
    static int bulk = -1;
    uint32_t staged[(FT_TT_OP_SAMPLE_RESULT_BYTES + 15u) / 4u];
    const size_t rounded = (bytes + 15u) & ~(size_t)15u;
    int rc;

    if (bulk < 0)
        bulk = (int)pn_tt_env_u32("PN_TT_TOPK_BULK", 1u);
    if (!bulk || (address & 15u) != 0u || rounded > sizeof(staged) || !device->runtime.bulk_l1.read)
        return device->runtime.noc.read(device->runtime.noc.user, (uint16_t)x, (uint16_t)y,
                                        address, block, bytes);
    rc = device->runtime.bulk_l1.read(device->runtime.bulk_l1.user, (uint16_t)x, (uint16_t)y,
                                      address, staged, rounded);
    if (rc == FT_TT_OK)
        memcpy(block, staged, bytes);
    return rc;
}

void pn_tt_topk_dump_failure(pn_tt_device *device)
{
    const pn_tt_residency *r = &device->sample;
    uint32_t words[8], p;

    for (p = 0u; p < 5u; p++) {
        if (device->runtime.noc.read(device->runtime.noc.user, r->queue.x, r->queue.y,
                                     r->queue.control_address + 0x40u + p * 32u, words,
                                     sizeof(words)) != FT_TT_OK)
            return;
        fprintf(stderr, "pn_tt: topk FAILURE proc %u: state %u cursor %u error %u lastop %u "
                        "spins %u\n", p, words[0], words[1], words[3], words[4], words[5]);
    }
    if (device->sample_result_address != 0u &&
        device->runtime.noc.read(device->runtime.noc.user, r->queue.x, r->queue.y,
                                 pn_tt_topk_helper_area(device->sample_result_address), words,
                                 sizeof(words)) == FT_TT_OK)
        fprintf(stderr, "pn_tt: topk FAILURE helper area: page %#x stamp %#x tile %u depth %u "
                        "done %#x %#x %#x\n", words[0], words[1], words[2], words[3], words[4],
                words[5], words[6]);
}

/** Member m's residency and result address (0 = the main sampling residency). */
static pn_tt_residency *topk_member(pn_tt_device *device, uint32_t m, uint32_t *out_result)
{
    *out_result = m == 0u ? device->sample_result_address : device->sample_member_result[m - 1u];
    return m == 0u ? &device->sample : &device->sample_member[m - 1u];
}

/** Insertion sort in the kernel's order: key descending, then index ascending. */
static void topk_sort_pairs(uint32_t *index, float *value, uint32_t total)
{
    uint32_t k;

    for (k = 1; k < total; k++) {
        const uint32_t ix = index[k];
        const float v = value[k];
        const uint32_t key = pn_tt_topk_order_key(v);
        uint32_t j = k;

        while (j > 0u && (pn_tt_topk_order_key(value[j - 1u]) < key ||
                          (pn_tt_topk_order_key(value[j - 1u]) == key && index[j - 1u] > ix))) {
            index[j] = index[j - 1u];
            value[j] = value[j - 1u];
            j--;
        }
        index[j] = ix;
        value[j] = v;
    }
}

int pn_tt_topk_merge_members(pn_tt_device *device, uint32_t count, uint32_t block_offset,
                             bool slice_offset_red, uint32_t *out_indices, float *out_values,
                             uint32_t *out_count)
{
    const uint32_t members = device->sample_members;
    const uint32_t slice = device->sample_tile_count / members;
    uint32_t all_index[PN_TT_TOPK_MAX_MEMBERS * PN_TT_TOPK_SELECT_MAX];
    float all_value[PN_TT_TOPK_MAX_MEMBERS * PN_TT_TOPK_SELECT_MAX];
    const size_t bytes = (FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                          count * FT_TT_OP_SAMPLE_PAIR_WORDS) * sizeof(uint32_t);
    uint32_t total = 0u, m, k;

    for (m = 0; m < members; m++) {
        uint32_t result;
        const pn_tt_residency *r = topk_member(device, m, &result);
        uint32_t block[FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                       PN_TT_TOPK_SELECT_MAX * FT_TT_OP_SAMPLE_PAIR_WORDS];
        uint32_t indices[PN_TT_TOPK_SELECT_MAX];
        float values[PN_TT_TOPK_SELECT_MAX];
        uint32_t produced = 0u;

        if (pn_tt_topk_read_result(device, r->core_x, r->core_y, result + block_offset, block,
                                   bytes) != FT_TT_OK ||
            ft_tt_op_sample_result_decode(block, count, indices, values, &produced,
                                          NULL) != FT_TT_OK)
            return -1;
        device->bytes_read += bytes;
        for (k = 0; k < produced; k++) {
            all_index[total] = indices[k] + (slice_offset_red ? 0u : m * slice * 32u);
            all_value[total] = values[k];
            total++;
        }
    }
    topk_sort_pairs(all_index, all_value, total);
    if (total > count)
        total = count;
    for (k = 0; k < total; k++) {
        out_indices[k] = all_index[k];
        out_values[k] = all_value[k];
    }
    *out_count = total;
    return 0;
}

/** Issue one `rows`-row descriptor to every member and await them all. @return 0 or -1 */
static int topk_rows_issue(pn_tt_device *device, const pn_tt_activation *logits, uint32_t count,
                           uint32_t rows)
{
    const uint32_t members = device->sample_members;
    const uint32_t slice = device->sample_tile_count / members;
    int failed = 0;
    uint32_t m;

    for (m = 0; m < members; m++) {
        uint32_t result;
        pn_tt_residency *r = topk_member(device, m, &result);
        ft_tt_op_topk_request request;
        uint32_t words[FT_TT_OP_SAMPLE_WORDS];

        memset(&request, 0, sizeof(request));
        request.tile_count = slice;
        request.row = device->topk_row;
        request.rows = rows;
        request.select_count = count;
        request.prefetch_tiles = device->sample_depth;
        request.logits = pn_tt_operand(device, logits);
        request.logits.address += m * slice * PN_TT_TILE_BF16;
        request.result_address = result;
        request.helper_area = pn_tt_topk_helper_area(result);
        if (pn_tt_red("topk_helper_drop"))
            request.flags |= FT_TT_OP_SAMPLE_FLAG_RED_DROP_HELPER;
        request.state_address =
            m == 0u ? device->sample_state_address : device->sample_member_state[m - 1u];
        if (ft_tt_op_sample_parameters(&request, words) != FT_TT_OK ||
            pn_tt_residency_enqueue(r, FT_TT_QUEUE_OP_TOPK, words, 1u) != 0)
            failed = 1;
    }
    for (m = 0; m < members; m++) {
        uint32_t result;

        if (pn_tt_residency_await(topk_member(device, m, &result), PN_TT_OP_TIMEOUT_MS) != 0)
            failed = 1;
    }
    return failed ? -1 : 0;
}

int pn_tt_topk_rows(pn_tt_device *device, const pn_tt_activation *logits, uint32_t count,
                    uint32_t rows, uint32_t stride, uint32_t *out_indices, float *out_values,
                    uint32_t *out_counts)
{
    const double started = pn_tt_now_seconds();
    uint32_t k;

    if (!device || !logits || !out_indices || !out_values || !out_counts || stride < count)
        return -1;
    if (device->sample_members < 2u || device->tail_bins != 0u || rows < 2u ||
        rows > FT_TT_OP_SAMPLE_MAX_ROWS || device->topk_row + rows > 32u)
        return 1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (count == 0u || count > PN_TT_TOPK_SELECT_MAX || !device->sample.running ||
        logits->tile_count < device->sample_tile_count)
        return -1;
    if (topk_rows_issue(device, logits, count, rows) != 0)
        return -1;
    for (k = 0; k < rows; k++)
        /* RED `topk_rows_block0`: every row reads row 0's block — must DIFFER. */
        if (pn_tt_topk_merge_members(device, count,
                                     pn_tt_red("topk_rows_block0")
                                         ? 0u : k * FT_TT_OP_SAMPLE_RESULT_BYTES,
                                     false, out_indices + k * stride,
                                     out_values + k * stride, &out_counts[k]) != 0)
            return -1;
    pn_tt_oncard_account(device, "topk", "select_rows", device->sample_tile_count * rows,
                         pn_tt_now_seconds() - started);
    return 0;
}

#endif /* USE_TT */
