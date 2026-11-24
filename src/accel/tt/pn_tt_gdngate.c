/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdngate.c
 * @brief The GDN GATE family host half. See pn_tt_gdngate.h for what and why.
 *
 * @status Shipped: det gates on every TT path. FLAT retry after an idle retire fixed 2026-11-13.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdngate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"

#include "tt/ft_tt_op_dispatch.h"
#include "tt/kernels/resident_gdngate_protocol.h"

#define PN_TT_GDNGATE_PROGRAM "gdngate"
#define PN_TT_GDNGATE_CORE_ENV "PN_TT_GDNGATE_CORE_LIST"
/** The cores no family holds with every decode family and the 48-core prefill grid (0-47)
 *  armed — read off a full arm's `residency up on core` census (PM 7, 2026-10-08). */
#define PN_TT_GDNGATE_CORE_DEFAULT "121,126,127,128"
/** The two CBs: the reader's stage (two tiles + the constants, addressed) and the blocks. */
#define PN_TT_GDNGATE_CB_STAGE 0u
#define PN_TT_GDNGATE_CB_OUT 16u
#define PN_TT_GDNGATE_CB_OUT_PAGES 2u

/** `PN_TT_GDNGATE_CORE_LIST` (comma-separated core indices) into `cores`. @return count. */
static uint32_t gdngate_core_list(uint32_t *cores)
{
    const char *env = getenv(PN_TT_GDNGATE_CORE_ENV);
    const char *at = (env && *env) ? env : PN_TT_GDNGATE_CORE_DEFAULT;
    uint32_t count = 0u;
    char *end;

    while (*at && count < PN_TT_GDNGATE_MAX_MEMBERS) {
        const unsigned long core = strtoul(at, &end, 10);

        if (end == at)
            break;
        cores[count++] = (uint32_t)core;
        at = *end == ',' ? end + 1 : end;
    }
    return count;
}

static uint32_t gdngate_xy(const pn_tt_device *device, uint32_t bank)
{
    return (uint32_t)device->dram[bank].x | ((uint32_t)device->dram[bank].y << 16);
}

static uint32_t gdngate_align64(uint32_t bytes)
{
    return (bytes + 63u) & ~63u;
}

/** Band 0's DRAM address of a strip, whatever band it is seeked to. */
static uint32_t gdngate_band0_address(const pn_tt_activation *strip)
{
    return strip->address - strip->row_offset * strip->tile_count * PN_TT_TILE_BF16;
}

/** Prepare, place and start member `index` on `core`. */
static bool gdngate_arm_member(pn_tt_device *device, uint32_t index, uint32_t core)
{
    pn_tt_residency *member = &device->gdngate[index];
    ft_tt_launch_cb_entry placed[2];
    ft_tt_op_cb_request requests[2];

    if (!pn_tt_prepare_family_on_core(device, member, PN_TT_GDNGATE_PROGRAM, "gdngate",
                                      PN_TT_GDNGATE_CORE_ENV, core, PN_TT_ONCARD_DM_MASK))
        return false;
    requests[0].index = PN_TT_GDNGATE_CB_STAGE;
    requests[0].page_bytes = 2u * PN_TT_TILE_BF16 + kGgConstWords * 4u;
    requests[0].page_count = 1u;
    requests[1].index = PN_TT_GDNGATE_CB_OUT;
    requests[1].page_bytes = gdngate_align64(ftgg_block_bytes(device->gdngate_heads));
    requests[1].page_count = PN_TT_GDNGATE_CB_OUT_PAGES;
    if (ft_tt_op_cb_place(requests, 2u, placed, member->cb_buffer, sizeof(member->cb_buffer),
                          &member->cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdngate circular buffers do not fit\n");
        pn_tt_residency_release(member);
        return false;
    }
    return pn_tt_start_family(member);
}

/** Upload every layer's constants block: ssm_a[32] | ssm_dt[32] (zero past `heads`). */
static int gdngate_upload_constants(pn_tt_device *device, const float *const *ssm_a,
                                    const float *const *ssm_dt)
{
    const size_t bytes = (size_t)device->gdngate_layers * kGgConstWords * 4u;
    uint32_t *table = calloc(1, bytes);
    ft_tt_buffer buffer;
    int rc = -1;

    if (!table)
        return -1;
    for (uint32_t layer = 0; layer < device->gdngate_layers; layer++) {
        uint32_t *row = table + (size_t)layer * kGgConstWords;

        memcpy(row, ssm_a[layer], device->gdngate_heads * sizeof(float));
        memcpy(row + kGgMaxHeads, ssm_dt[layer], device->gdngate_heads * sizeof(float));
    }
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 0u, bytes, &buffer) ==
        FT_TT_OK) {
        device->gdngate_constants.bank = buffer.logical_bank;
        device->gdngate_constants.address = (uint32_t)buffer.address;
        rc = pn_tt_write_dram(device, buffer.logical_bank, (uint32_t)buffer.address, table,
                              bytes);
    }
    free(table);
    return rc;
}

int pn_tt_gdngate_arm(pn_tt_device *device, uint32_t layers, uint32_t heads,
                      const float *const *ssm_a, const float *const *ssm_dt, uint32_t max_rows,
                      uint32_t scale_bits)
{
    const char *env = getenv("PN_TT_GDNGATE");
    uint32_t want, index;
    ft_tt_buffer buffer;
    size_t array_bytes;

    if (!device || (env && *env && atoi(env) == 0))
        return -1;
    if (layers == 0u || heads == 0u || heads > kGgMaxHeads || !ssm_a || !ssm_dt ||
        max_rows == 0u || !pn_tt_program_is_present(PN_TT_GDNGATE_PROGRAM)) {
        fprintf(stderr, "pn_tt: gdngate refused — %u layer(s), %u head(s), program %s\n",
                layers, heads,
                pn_tt_program_is_present(PN_TT_GDNGATE_PROGRAM) ? "present" : "ABSENT");
        return -1;
    }
    want = gdngate_core_list(device->gdngate_cores);
    if (want == 0u)
        return -1;
    device->gdngate_layers = layers;
    device->gdngate_heads = heads;
    device->gdngate_scale_bits = scale_bits;
    device->gdngate_max_bands = (max_rows + kGgTileRows - 1u) / kGgTileRows;
    device->gdngate_block_stride = gdngate_align64(ftgg_block_bytes(heads));
    /* THE DELTA DESCRIPTOR READS ITS GATE AT ITS CONSTANT'S xy: the card-gate DELTA always
     * uses constant slot 0 (pn_tt_gdnc_delta_oncard_card_gate), so the blocks go in its bank. */
    array_bytes = (size_t)layers * 2u * device->gdngate_max_bands * device->gdngate_block_stride;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM,
                           device->gdnc_delta_constant[0].bank, array_bytes, &buffer) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: gdngate — no room for %zu bytes of gate blocks\n", array_bytes);
        return -1;
    }
    device->gdngate_blocks.bank = buffer.logical_bank;
    device->gdngate_blocks.address = (uint32_t)buffer.address;
    if (gdngate_upload_constants(device, ssm_a, ssm_dt) != 0) {
        fprintf(stderr, "pn_tt: gdngate — the constants table did not upload\n");
        return -1;
    }
    for (index = 0; index < want; index++) {
        if (!gdngate_arm_member(device, index, device->gdngate_cores[index]))
            break;
        device->gdngate_members = index + 1u;
    }
    if (device->gdngate_members == 0u) {
        fprintf(stderr, "pn_tt: gdngate — no member armed\n");
        return -1;
    }
    fprintf(stderr, "pn_tt: gdngate — %u member(s), %u layer(s) x %u band(s) of gate blocks "
                    "(%zu bytes)\n",
            device->gdngate_members, layers, device->gdngate_max_bands, array_bytes);
    return 0;
}

bool pn_tt_gdngate_enabled(const pn_tt_device *device)
{
    return device && device->gdngate_members > 0u;
}

uint32_t pn_tt_gdngate_block_address(const pn_tt_device *device, uint32_t layer, uint32_t band,
                                     uint32_t sub)
{
    const uint32_t per_layer = 2u * device->gdngate_max_bands * device->gdngate_block_stride;

    return device->gdngate_blocks.address + layer * per_layer +
           (2u * band + sub) * device->gdngate_block_stride;
}

/** The parameter words of one descriptor: bands [first, first + count) of `layer`. */
static void gdngate_words(const pn_tt_device *device, uint32_t layer,
                          const pn_tt_activation *alpha, const pn_tt_activation *beta,
                          uint32_t rows, uint32_t sub_rows, uint32_t first, uint32_t count,
                          uint32_t *words)
{
    const uint32_t band_bytes = alpha->tile_count * PN_TT_TILE_BF16;
    const uint32_t done = first * kGgTileRows;
    const uint32_t take = rows - done < count * kGgTileRows ? rows - done : count * kGgTileRows;

    memset(words, 0, kGgWords * sizeof(uint32_t));
    words[kGgFlags] = pn_tt_red("gdngate_beta_as_decay") ? kGgFlagRedBetaAsDecay : 0u;
    words[kGgRows] = take;
    words[kGgBandCount] = count;
    words[kGgSubRows] = sub_rows;
    words[kGgHeads] = device->gdngate_heads;
    words[kGgAlphaXY] = gdngate_xy(device, alpha->bank);
    words[kGgAlphaBase] = gdngate_band0_address(alpha) + first * band_bytes;
    words[kGgBetaXY] = gdngate_xy(device, beta->bank);
    words[kGgBetaBase] = gdngate_band0_address(beta) + first * band_bytes;
    words[kGgBandStride] = band_bytes;
    words[kGgConstXY] = gdngate_xy(device, device->gdngate_constants.bank);
    words[kGgConstBase] = device->gdngate_constants.address + layer * kGgConstWords * 4u;
    words[kGgDstXY] = gdngate_xy(device, device->gdngate_blocks.bank);
    words[kGgDstBase] = pn_tt_gdngate_block_address(device, layer, first, 0u);
    words[kGgBlockStride] = device->gdngate_block_stride;
    words[kGgScaleBits] = device->gdngate_scale_bits;
}

int pn_tt_gdngate_issue_bands(pn_tt_device *device, uint32_t layer,
                              const pn_tt_activation *alpha, const pn_tt_activation *beta,
                              uint32_t rows, uint32_t sub_rows, uint32_t first_band,
                              uint32_t band_count)
{
    const uint32_t bands = (rows + kGgTileRows - 1u) / kGgTileRows;
    uint32_t members, per_member, first = first_band, index;
    uint32_t words[kGgWords];

    if (!pn_tt_gdngate_enabled(device) || !alpha || !beta || rows == 0u || band_count == 0u ||
        layer >= device->gdngate_layers || bands > device->gdngate_max_bands ||
        first_band + band_count > bands || alpha->tile_count != 1u || beta->tile_count != 1u)
        return -1;
    members = device->gdngate_members < band_count ? device->gdngate_members : band_count;
    per_member = (band_count + members - 1u) / members;
    device->gdngate_last.layer = layer;
    device->gdngate_last.alpha = *alpha;
    device->gdngate_last.beta = *beta;
    device->gdngate_last.rows = rows;
    device->gdngate_last.sub_rows = sub_rows;
    for (index = 0; index < members && first < first_band + band_count; index++) {
        const uint32_t end = first_band + band_count;
        const uint32_t count = end - first < per_member ? end - first : per_member;
        /* Round-robin over issues too, so a super-band's few bands do not all land on 0. */
        const uint32_t member = (device->gdngate_next_member + index) % device->gdngate_members;

        gdngate_words(device, layer, alpha, beta, rows, sub_rows, first, count, words);
        if (pn_tt_residency_enqueue(&device->gdngate[member], FT_TT_QUEUE_OP_GDN_GATE, words,
                                    1u) != 0)
            device->gdngate_failed = true;
        device->gdngate_pending[member]++;
        first += count;
    }
    device->gdngate_next_member = (device->gdngate_next_member + members) %
                                  device->gdngate_members;
    device->gdngate_issues++;
    return 0;
}

int pn_tt_gdngate_issue(pn_tt_device *device, uint32_t layer, const pn_tt_activation *alpha,
                        const pn_tt_activation *beta, uint32_t rows, uint32_t sub_rows)
{
    if (device)
        device->gdngate_layer_issues++;
    return pn_tt_gdngate_issue_bands(device, layer, alpha, beta, rows, sub_rows, 0u,
                                     (rows + kGgTileRows - 1u) / kGgTileRows);
}

/** Re-arm every member and replay the last issue, awaited. The strips are untouched (the
 *  caller settles before its band loop rewrites anything), so the replay is the same op. */
static int gdngate_replay(pn_tt_device *device)
{
    uint32_t index;

    for (index = 0; index < device->gdngate_members; index++) {
        const uint32_t core = device->gdngate_cores[index];

        fprintf(stderr, "pn_tt: gdngate member %u refused a descriptor — retiring it by ASKING "
                        "and arming a fresh one on core %u\n", index, core);
        pn_tt_residency_stop(&device->gdngate[index]);
        pn_tt_residency_release(&device->gdngate[index]);
        if (!gdngate_arm_member(device, index, core))
            return -1;
        device->gdngate_pending[index] = 0u;
    }
    device->gdngate_failed = false;
    if (pn_tt_gdngate_issue(device, device->gdngate_last.layer, &device->gdngate_last.alpha,
                            &device->gdngate_last.beta, device->gdngate_last.rows,
                            device->gdngate_last.sub_rows) != 0)
        return -1;
    for (index = 0; index < device->gdngate_members; index++) {
        if (device->gdngate_pending[index] != 0u &&
            pn_tt_residency_await(&device->gdngate[index], PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
        device->gdngate_pending[index] = 0u;
    }
    return device->gdngate_failed ? -1 : 0;
}

int pn_tt_gdngate_settle(pn_tt_device *device)
{
    const double started = pn_tt_now_seconds();
    bool failed;
    uint32_t index;
    int rc;

    if (!pn_tt_gdngate_enabled(device))
        return -1;
    failed = device->gdngate_failed;
    for (index = 0; index < device->gdngate_members; index++) {
        if (device->gdngate_pending[index] == 0u)
            continue;
        if (pn_tt_residency_await(&device->gdngate[index], PN_TT_OP_TIMEOUT_MS) != 0)
            failed = true;
        device->gdngate_pending[index] = 0u;
    }
    rc = failed ? gdngate_replay(device) : 0;
    device->gdngate_settle_seconds += pn_tt_now_seconds() - started;
    device->gdngate_settles++;
    return rc;
}

void pn_tt_gdngate_report(const pn_tt_device *device)
{
    if (!pn_tt_gdngate_enabled(device))
        return;
    fprintf(stderr, "pn_tt: gdngate — %llu issue(s) (%llu at a layer start), %llu settle(s) "
                    "waiting %.4f s, %llu FLAT (cumulative)\n",
            (unsigned long long)device->gdngate_issues,
            (unsigned long long)device->gdngate_layer_issues,
            (unsigned long long)device->gdngate_settles, device->gdngate_settle_seconds,
            (unsigned long long)device->gdngate_flats);
}

/** FLAT words for heads [first, first + count) of `heads_total`. */
static void gdngate_flat_words(const pn_tt_device *device, uint32_t layer,
                               const pn_tt_activation *alpha, const pn_tt_activation *beta,
                               uint32_t rows, uint32_t dst_bank, uint32_t dst_address,
                               uint32_t row_stride, uint32_t first, uint32_t count,
                               uint32_t *words)
{
    gdngate_words(device, layer, alpha, beta, rows, 16u, 0u, 1u, words);
    words[kGgFlags] |= kGgFlagFlat;
    words[kGgSubRows] = first;
    words[kGgHeads] = count;
    words[kGgBandStride] = device->gdngate_heads;
    words[kGgDstXY] = gdngate_xy(device, dst_bank);
    words[kGgDstBase] = dst_address;
    words[kGgBlockStride] = row_stride;
}

int pn_tt_gdngate_flat(pn_tt_device *device, uint32_t layer, const pn_tt_activation *alpha,
                       const pn_tt_activation *beta, uint32_t rows, uint32_t dst_bank,
                       uint32_t dst_address, uint32_t row_stride)
{
    uint32_t words[PN_TT_GDNGATE_MAX_MEMBERS][kGgWords];
    bool enqueued[PN_TT_GDNGATE_MAX_MEMBERS];
    uint32_t members, per, index;
    int failed = 0;

    if (!pn_tt_gdngate_enabled(device) || !alpha || !beta || rows == 0u || rows > kGgTileRows ||
        layer >= device->gdngate_layers || alpha->tile_count != 1u || beta->tile_count != 1u)
        return -1;
    /* THE HEADS SPLIT OVER THE MEMBERS in slices of a multiple of 4 (16-byte runs), each
     * member writing only its slice's two runs a row; `PN_TT_GDNGATE_FLAT_SPLIT=0` = one. */
    members = pn_tt_env_u32("PN_TT_GDNGATE_FLAT_SPLIT", 1u) ? device->gdngate_members : 1u;
    while (members > 1u && (device->gdngate_heads % (members * 4u)) != 0u)
        members--;
    per = device->gdngate_heads / members;
    for (index = 0; index < members; index++) {
        gdngate_flat_words(device, layer, alpha, beta, rows, dst_bank, dst_address, row_stride,
                           index * per, per, words[index]);
        /* A refused enqueue is not a failure yet: an idle-retired member refuses by design
         * (pn_tt_residency.c, THE IDLE RETIRE) and the loop below re-arms it and runs its
         * slice. Counting it here failed the forward after a clean retry (mtpft selftest2,
         * 2026-11-13: the first per-token spec request after > 5 s idle, "generation failed"). */
        enqueued[index] = pn_tt_residency_enqueue(&device->gdngate[index],
                                                  FT_TT_QUEUE_OP_GDN_GATE, words[index],
                                                  1u) == 0;
    }
    device->gdngate_flats++;
    /* Awaited here: the DELTA after it reads what they write (streamed: puller AWAITs). */
    for (index = 0; index < members; index++) {
        pn_tt_residency *member = &device->gdngate[index];

        if (enqueued[index] && pn_tt_residency_await(member, PN_TT_OP_TIMEOUT_MS) == 0)
            continue;
        /* A refused member (idle retire): re-armed and its slice run again, awaited. */
        fprintf(stderr, "pn_tt: gdngate member %u refused a FLAT descriptor — retiring it by "
                        "ASKING and arming a fresh one on core %u\n",
                index, device->gdngate_cores[index]);
        pn_tt_residency_stop(member);
        pn_tt_residency_release(member);
        if (!gdngate_arm_member(device, index, device->gdngate_cores[index]) ||
            pn_tt_residency_submit(member, FT_TT_QUEUE_OP_GDN_GATE, words[index],
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            failed = 1;
    }
    return failed ? -1 : 0;
}

long pn_tt_gdngate_verify_layer(pn_tt_device *device, uint32_t layer,
                                const pn_tt_activation *alpha, const pn_tt_activation *beta,
                                uint32_t rows, uint32_t sub_rows)
{
    const uint32_t bands = (rows + kGgTileRows - 1u) / kGgTileRows;
    const uint32_t words = ftgg_block_bytes(device->gdngate_heads) / 4u;
    uint32_t constants[kGgConstWords], expect[kGgBlockBytes / 4u], got[kGgBlockBytes / 4u];
    uint16_t tiles[2][PN_TT_TILE_BF16 / 2u];
    long bad = 0;

    if (pn_tt_read_dram(device, device->gdngate_constants.bank,
                        device->gdngate_constants.address + layer * kGgConstWords * 4u,
                        constants, sizeof(constants)) != 0)
        return -1;
    for (uint32_t band = 0; band < bands; band++) {
        const uint32_t band_rows = rows - band * kGgTileRows < kGgTileRows
                                       ? rows - band * kGgTileRows : kGgTileRows;
        const uint32_t offset = band * alpha->tile_count * PN_TT_TILE_BF16;

        if (pn_tt_read_dram(device, alpha->bank, gdngate_band0_address(alpha) + offset,
                            tiles[0], sizeof(tiles[0])) != 0 ||
            pn_tt_read_dram(device, beta->bank, gdngate_band0_address(beta) + offset, tiles[1],
                            sizeof(tiles[1])) != 0)
            return -1;
        for (uint32_t sub = 0; sub < 2u; sub++) {
            const uint32_t live = ftgg_sub_live_rows(band_rows, sub_rows, sub);

            if (live == 0u)
                continue;
            ftgg_build_block(expect, device->gdngate_heads, device->gdngate_scale_bits,
                             tiles[0], tiles[1], constants, sub * sub_rows, live, 0);
            if (pn_tt_read_dram(device, device->gdngate_blocks.bank,
                                pn_tt_gdngate_block_address(device, layer, band, sub), got,
                                words * 4u) != 0)
                return -1;
            for (uint32_t word = 0; word < words; word++)
                bad += expect[word] != got[word];
        }
    }
    return bad;
}

/** Fill `count` face-ordered bf16 tiles so that, over all of them, every bf16 bit pattern
 *  appears exactly once at each head column after rotation `rotation` (value v lands in
 *  band v / 1024, row (v / 32) % 32, head (v + rotation) % 32). Beta takes the reverse. */
static void gdngate_sweep_tiles(uint16_t *alpha, uint16_t *beta, uint32_t count,
                                uint32_t rotation)
{
    const uint32_t per_tile = PN_TT_TILE_BF16 / 2u;

    for (uint32_t value = 0; value < count * per_tile; value++) {
        const uint32_t band = value / per_tile, row = (value / 32u) % 32u;
        const uint32_t head = (value + rotation) % 32u;
        const size_t at = (size_t)band * per_tile + ftgg_face_index(row, head);

        alpha[at] = (uint16_t)value;
        beta[at] = (uint16_t)(0xFFFFu - value);
    }
}

long pn_tt_gdngate_sweep(pn_tt_device *device, uint32_t rotations)
{
    const uint32_t bands = 64u, rows = bands * kGgTileRows;
    const size_t bytes = (size_t)bands * PN_TT_TILE_BF16;
    uint16_t *host = malloc(2u * bytes);
    pn_tt_activation alpha, beta;
    ft_tt_buffer a_buffer, b_buffer;
    double compute_seconds = 0.0;
    long bad = 0, part;

    if (!host || bands > device->gdngate_max_bands ||
        ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 1u, bytes, &a_buffer) !=
            FT_TT_OK ||
        ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 2u, bytes, &b_buffer) !=
            FT_TT_OK) {
        free(host);
        return -1;
    }
    memset(&alpha, 0, sizeof(alpha));
    alpha.bank = a_buffer.logical_bank;
    alpha.address = (uint32_t)a_buffer.address;
    alpha.tile_count = 1u;
    alpha.replica_count = 1u;
    alpha.tile_rows = bands;
    alpha.active_rows = 1u;
    beta = alpha;
    beta.bank = b_buffer.logical_bank;
    beta.address = (uint32_t)b_buffer.address;
    for (uint32_t rotation = 0; rotation < rotations && bad >= 0; rotation++) {
        gdngate_sweep_tiles(host, host + bytes / 2u, bands, rotation);
        if (pn_tt_write_dram(device, alpha.bank, alpha.address, host, bytes) != 0 ||
            pn_tt_write_dram(device, beta.bank, beta.address, host + bytes / 2u, bytes) != 0)
            bad = -1;
        for (uint32_t layer = 0; bad >= 0 && layer < device->gdngate_layers; layer++) {
            const double started = pn_tt_now_seconds();
            int rc = pn_tt_gdngate_issue(device, layer, &alpha, &beta, rows, 16u);

            if (rc == 0)
                rc = pn_tt_gdngate_settle(device);
            compute_seconds += pn_tt_now_seconds() - started;
            if (rc != 0 ||
                (part = pn_tt_gdngate_verify_layer(device, layer, &alpha, &beta, rows, 16u)) < 0)
                bad = -1;
            else
                bad += part;
        }
    }
    free(host);
    fprintf(stderr, "pn_tt: gdngate SWEEP — every bf16 alpha and beta x %u head rotation(s) x "
                    "%u layers' constants: %ld word(s) differ from the host build; card "
                    "%.3f s over %u member(s) = %.2f us a (beta, decay) pair a member\n",
            rotations, device->gdngate_layers, bad, compute_seconds, device->gdngate_members,
            compute_seconds * 1e6 * device->gdngate_members /
                ((double)rotations * device->gdngate_layers * rows * device->gdngate_heads));
    return bad;
}

void pn_tt_gdngate_release(pn_tt_device *device)
{
    uint32_t index;

    if (!device)
        return;
    for (index = 0; index < device->gdngate_members; index++)
        pn_tt_residency_release(&device->gdngate[index]);
    device->gdngate_members = 0u;
}

#endif /* USE_TT */
