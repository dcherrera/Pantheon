/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_head_norm.c
 * @brief Implementation of the per-head reduce residencies. See pn_tt_head_norm.h — the
 *        layout argument that makes this file short lives there, as does the reason there
 *        is a SET of these keyed by width rather than one.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_head_norm.h"

#include <stdio.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_fp32_strip.h"

/**
 * Which enabled Tensix each slot drives.
 *
 * Defaults, not constants, and each must be a core no other residency holds AND that the
 * matmul group cannot grow into: members run upward from `PN_TT_MATMUL_CORE_FIRST` = 96, so
 * the shipped `PN_TT_MATMUL_CORES=16` stops at 110. The rest of the map, which is why the second
 * slot is not 92:
 *
 *     71        matmul member 0            72/73/74   reduce / eltwise / sample
 *     96..110   matmul members 1..15       82/83/84   rope / memory / attention
 *     85/86     render-widen / -narrow     90         gdn member 0
 *     91        head norm, slot 0          92..94     gdn head-split members (4)
 *     111       head norm, slot 1          112        eltwise split member
 *     114..120  gdnc members
 *
 * SLOT 0 IS WHOEVER ARMS FIRST, not whoever needs which width. With both families up the GDN
 * block's Wt=4 takes core 91 and attention's Wt=8 takes 103; with attention alone, attention
 * takes 91. Nothing downstream depends on the number — routing is by WIDTH — but a probe
 * reading a core map does, which is why the order is written down rather than inferred.
 *
 * 85 and 86 are the strip bridge's two render residencies and they are inside the range the
 * matmul group grows into: at `PN_TT_MATMUL_CORES=16` members reach 90 and would collide with
 * both of them and with the GDN block. The refusal below NAMES that, but the shipped count is
 * 8 (members 75..81) and there is headroom to 84 only.
 *
 * `pn_tt_prepare_family_on_core()` refuses a collision BY NAME against every live residency,
 * and that refusal is the guard — a contested core does not fail, it computes badly.
 */
#define PN_TT_CORE_HEAD_NORM_ENV       "PN_TT_CORE_HEAD_NORM"
#define PN_TT_CORE_HEAD_NORM_DEFAULT   91u
#define PN_TT_CORE_HEAD_NORM_2_ENV     "PN_TT_CORE_HEAD_NORM_2"
/* 111 since the PX7 placement shipped (was 103, inside the 16-member matmul's 96..110). */
#define PN_TT_CORE_HEAD_NORM_2_DEFAULT 111u

/** THE HEAD SPLIT, per slot. Members 1..N-1 of slot 0 run on 130..136 and of slot 1 on
 *  123..128 — the two free blocks above every other family's default (render members 122 and
 *  129 sit between them). Each slot's count is capped to its block so no member index can
 *  leave the grid. A call whose heads the count does not divide runs whole on member 0. */
#define PN_TT_HEAD_NORM_CORES_ENV          "PN_TT_HEAD_NORM_CORES"
#define PN_TT_HEAD_NORM_CORES_DEFAULT      4u /* 0.345 -> 0.150 s, prefill +3.2%, decode +3.3% */
#define PN_TT_HEAD_NORM_MEMBER_CORE_ENV    "PN_TT_HEAD_NORM_MEMBER_CORE_FIRST"
static const uint32_t g_head_norm_member_first[PN_TT_HEAD_NORM_MAX_WIDTHS] = { 130u, 123u };
static const uint32_t g_head_norm_member_room[PN_TT_HEAD_NORM_MAX_WIDTHS] = { 7u, 6u };

/** Both members every slot serves. The placement is the UNION: RMS norm carries gamma,
 *  L2 norm does not, and a stream the residency does not carry is placed at one page — so
 *  arming for the union is what lets one core run both. */
static const ft_tt_op_reduce_function g_head_norm_members[] = { FT_TT_OP_REDUCE_FN_RMS_NORM,
                                                                FT_TT_OP_REDUCE_FN_L2_NORM };
#define PN_TT_HEAD_NORM_MEMBER_COUNT \
    (sizeof(g_head_norm_members) / sizeof(g_head_norm_members[0]))

/** The core environment name and default for slot `index`. */
static void pn_tt_head_norm_core_for_slot(uint32_t index, const char **out_env,
                                          uint32_t *out_default)
{
    if (index == 0u) {
        *out_env = PN_TT_CORE_HEAD_NORM_ENV;
        *out_default = PN_TT_CORE_HEAD_NORM_DEFAULT;
        return;
    }
    *out_env = PN_TT_CORE_HEAD_NORM_2_ENV;
    *out_default = PN_TT_CORE_HEAD_NORM_2_DEFAULT;
}

/** The live slot placed for `tile_cols`, or NULL. Const and non-const forms, because the
 *  routing question is asked from both. */
static const pn_tt_head_norm_residency *pn_tt_head_norm_slot_const(const pn_tt_device *device,
                                                                   uint32_t tile_cols)
{
    uint32_t index;

    for (index = 0; index < device->head_norm_count; index++) {
        if (device->head_norm[index].tile_cols == tile_cols &&
            device->head_norm[index].residency.running)
            return &device->head_norm[index];
    }
    return NULL;
}

static pn_tt_head_norm_residency *pn_tt_head_norm_slot(pn_tt_device *device, uint32_t tile_cols)
{
    return (pn_tt_head_norm_residency *)pn_tt_head_norm_slot_const(device, tile_cols);
}

/** Which reduce PROGRAM serves a page size. The two are the same choice: `reduce_fp32` IS
 *  the `TILE_BYTES=4096` build of the same three kernel sources. */
static const char *pn_tt_head_norm_program_for(uint32_t tile_bytes)
{
    return tile_bytes == PN_TT_TILE_FP32 ? PN_TT_REDUCE_FP32_PROGRAM_NAME
                                         : PN_TT_REDUCE_PROGRAM_NAME;
}

/** Arm, or re-arm after a refusal, the slot at `index` for the width and page it records. */
static bool pn_tt_arm_head_norm_slot(pn_tt_device *device, uint32_t index)
{
    pn_tt_head_norm_residency *slot = &device->head_norm[index];
    ft_tt_op_reduce_residency residency;
    ft_tt_launch_cb_entry placed[FT_TT_OP_REDUCE_CB_COUNT];
    const char *core_env;
    uint32_t core_default;

    pn_tt_head_norm_core_for_slot(index, &core_env, &core_default);
    if (ft_tt_op_reduce_residency_for(g_head_norm_members, PN_TT_HEAD_NORM_MEMBER_COUNT,
                                      slot->tile_cols, slot->tile_bytes, &residency) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: no reduce residency serves the head width Wt=%u — those "
                        "per-head norms stay on the host\n", slot->tile_cols);
        return false;
    }
    if (!pn_tt_prepare_family(device, &slot->residency,
                              pn_tt_head_norm_program_for(slot->tile_bytes), "head-norm",
                              core_env, core_default, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_reduce_place_cbs(&residency, placed, slot->residency.cb_buffer,
                                  sizeof(slot->residency.cb_buffer),
                                  &slot->residency.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the per-head reduce circular buffers do not fit at Wt=%u\n",
                slot->tile_cols);
        pn_tt_residency_release(&slot->residency);
        return false;
    }
    return pn_tt_start_family(&slot->residency);
}

/** Arm head-split member `member` (1-based) of slot `index` on `core`. */
static bool pn_tt_arm_head_norm_member(pn_tt_device *device, uint32_t index, uint32_t member,
                                       uint32_t core)
{
    pn_tt_head_norm_residency *slot = &device->head_norm[index];
    pn_tt_residency *residency = &slot->member[member - 1u];
    ft_tt_op_reduce_residency shape;
    ft_tt_launch_cb_entry placed[FT_TT_OP_REDUCE_CB_COUNT];

    if (ft_tt_op_reduce_residency_for(g_head_norm_members, PN_TT_HEAD_NORM_MEMBER_COUNT,
                                      slot->tile_cols, slot->tile_bytes, &shape) != FT_TT_OK)
        return false;
    if (!pn_tt_prepare_family_on_core(device, residency,
                                      pn_tt_head_norm_program_for(slot->tile_bytes),
                                      "head-norm-member", PN_TT_HEAD_NORM_MEMBER_CORE_ENV,
                                      core, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_reduce_place_cbs(&shape, placed, residency->cb_buffer,
                                  sizeof(residency->cb_buffer), &residency->cb_mask) !=
        FT_TT_OK) {
        pn_tt_residency_release(residency);
        return false;
    }
    return pn_tt_start_family(residency);
}

/** Arm slot `index`'s head-split members; a member that will not come up shrinks the split. */
static void pn_tt_arm_head_norm_members(pn_tt_device *device, uint32_t index)
{
    pn_tt_head_norm_residency *slot = &device->head_norm[index];
    const uint32_t asked = pn_tt_env_u32(PN_TT_HEAD_NORM_CORES_ENV,
                                         PN_TT_HEAD_NORM_CORES_DEFAULT);
    const uint32_t first = g_head_norm_member_first[index];
    const uint32_t room = g_head_norm_member_room[index] + 1u;
    uint32_t want = asked < 1u ? 1u : (asked > room ? room : asked);
    uint32_t up = 1u, member;

    slot->members = 1u;
    for (member = 1u; member < want; member++) {
        if (!pn_tt_arm_head_norm_member(device, index, member, first + member - 1u))
            break;
        up++;
    }
    slot->members = up;
    if (asked > 1u)
        fprintf(stderr, "pn_tt: head-norm Wt=%u HEAD SPLIT — asked %u, running %u member(s) "
                        "from core %u (members 1.. on %u..)\n",
                slot->tile_cols, asked, up, slot->residency.core_index, first);
}

/** Retire and re-arm member `member` (1-based) of slot `index` on its own core. */
static bool pn_tt_head_norm_member_rearm(pn_tt_device *device, uint32_t index, uint32_t member)
{
    pn_tt_residency *residency = &device->head_norm[index].member[member - 1u];
    const uint32_t core = residency->core_index;

    fprintf(stderr, "pn_tt: head-norm member %u refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", member, core);
    pn_tt_residency_stop(residency);
    pn_tt_residency_release(residency);
    device->oncard_restarts++;
    if (!pn_tt_residency_retired(residency))
        return false;
    return pn_tt_arm_head_norm_member(device, index, member, core);
}

int pn_tt_device_arm_head_norm(pn_tt_device *device, uint32_t head_width)
{
    return pn_tt_device_arm_head_norm_at(device, head_width, PN_TT_TILE_FP32);
}

int pn_tt_device_arm_head_norm_at(pn_tt_device *device, uint32_t head_width,
                                  uint32_t tile_bytes)
{
    const char *program = pn_tt_head_norm_program_for(tile_bytes);
    uint32_t tile_cols;
    uint32_t index;

    if (!device || head_width == 0u || (head_width % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (tile_bytes != PN_TT_TILE_FP32 && tile_bytes != PN_TT_TILE_BF16)
        return -1;
    tile_cols = head_width / PN_TT_TILE_DIM;
    /* Idempotent per width: the second GDN layer to ask must not take a second core. A width
     * already placed at ANOTHER page is a refusal rather than a second slot, because the
     * routing downstream is by width and two slots at one width would be a coin toss. */
    if (pn_tt_head_norm_slot(device, tile_cols) != NULL) {
        if (pn_tt_head_norm_slot(device, tile_cols)->tile_bytes != tile_bytes) {
            fprintf(stderr, "pn_tt: Wt=%u is already placed at a %u-byte page and was asked "
                            "for at %u — one width, one residency\n", tile_cols,
                    pn_tt_head_norm_slot(device, tile_cols)->tile_bytes, tile_bytes);
            return -1;
        }
        return 0;
    }
    if (device->head_norm_count >= PN_TT_HEAD_NORM_MAX_WIDTHS) {
        fprintf(stderr, "pn_tt: %u per-head reduce residencies are already placed (Wt",
                device->head_norm_count);
        for (index = 0; index < device->head_norm_count; index++)
            fprintf(stderr, " %u", device->head_norm[index].tile_cols);
        fprintf(stderr, ") — the Wt=%u norms stay on the host. Raise "
                        "PN_TT_HEAD_NORM_MAX_WIDTHS.\n", tile_cols);
        return 0;
    }
    if (!pn_tt_program_is_present(program)) {
        fprintf(stderr, "pn_tt: no `%s` program — the Wt=%u per-head norms stay on the "
                        "host\n", program, tile_cols);
        return 0;
    }

    index = device->head_norm_count;
    device->head_norm[index].tile_cols = tile_cols;
    device->head_norm[index].tile_bytes = tile_bytes;
    if (!pn_tt_arm_head_norm_slot(device, index)) {
        device->head_norm[index].tile_cols = 0u;
        device->head_norm[index].tile_bytes = 0u;
        return 0;
    }
    device->head_norm_count++;
    pn_tt_arm_head_norm_members(device, index);
    return 0;
}

bool pn_tt_head_norm_is_armed(const pn_tt_device *device, uint32_t head_width)
{
    if (!device)
        return false;
    if (head_width == 0u)
        return device->head_norm_count != 0u &&
               device->head_norm[0].residency.running;
    if ((head_width % PN_TT_TILE_DIM) != 0u)
        return false;
    return pn_tt_head_norm_slot_const(device, head_width / PN_TT_TILE_DIM) != NULL;
}

int pn_tt_head_norm_constants_upload(pn_tt_device *device, ft_tt_op_reduce_function function,
                                     const float *gamma, uint32_t width, float eps,
                                     pn_tt_norm_constants *out_constants)
{
    uint32_t page;

    if (!device || !out_constants || width == 0u || (width % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (pn_tt_head_norm_slot(device, width / PN_TT_TILE_DIM) == NULL) {
        fprintf(stderr, "pn_tt: head-norm constants at Wt=%u, which no per-head reduce "
                        "residency was placed for\n", width / PN_TT_TILE_DIM);
        return -1;
    }
    /* RED PROOF, on the card: build a Float32 residency's constants at the bf16 page under a
     * program placed at 4096. Nothing refuses it — the reader simply reads 4096 bytes where
     * 2048 were written — so the arm exists to show that the page size is load-bearing and
     * that the probe can see it. `PN_TT_RED=head_norm_bf16_constants`. It is a NO-OP on a
     * slot already at the bf16 page, which is stated rather than left to be discovered. */
    page = pn_tt_head_norm_slot(device, width / PN_TT_TILE_DIM)->tile_bytes;
    if (page == PN_TT_TILE_FP32 && pn_tt_red("head_norm_bf16_constants"))
        page = PN_TT_TILE_BF16;
    return pn_tt_reduce_constants_upload(device, function, gamma, width, eps, page,
                                         out_constants);
}

/**
 * THE HEAD SPLIT: member m reduces heads [m*per, (m+1)*per) — the same descriptor with
 * `row_blocks = per` and the input and output bases moved by `m*per*Wt` tiles. Gamma is
 * shared by every head, so the constants operand is not moved. Enqueue all, then await all;
 * a member that fails is re-armed and its slice submitted alone, after every member was
 * awaited (an enqueued descriptor is always awaited, or that member's cursor desynchronises).
 */
static int pn_tt_head_norm_submit_split(pn_tt_device *device, pn_tt_head_norm_residency *slot,
                                        uint32_t index, const ft_tt_op_reduce_request *whole)
{
    const uint32_t members = slot->members;
    const uint32_t per = whole->row_blocks / members;
    const uint32_t step = per * whole->tile_cols * slot->tile_bytes;
    uint32_t words[PN_TT_HEAD_NORM_MAX_MEMBERS][FT_TT_OP_VECTOR_WORDS];
    pn_tt_residency *set[PN_TT_HEAD_NORM_MAX_MEMBERS];
    bool bad[PN_TT_HEAD_NORM_MAX_MEMBERS];
    double started = pn_tt_now_seconds();
    int failed = 0;
    uint32_t m;

    memset(bad, 0, sizeof(bad));
    for (m = 0; m < members; m++) {
        ft_tt_op_reduce_request sliced = *whole;

        sliced.row_blocks = per;
        sliced.input.address += m * step;
        sliced.output.address += m * step;
        if (ft_tt_op_reduce_parameters(&sliced, words[m]) != FT_TT_OK)
            return -1;
        pn_tt_vec_probe_apply(words[m]);
        set[m] = m == 0u ? &slot->residency : &slot->member[m - 1u];
    }
    for (m = 0; m < members; m++)
        if (pn_tt_residency_enqueue(set[m], FT_TT_QUEUE_OP_REDUCE, words[m], 1u) != 0)
            failed = bad[m] = 1;
    set[0]->seconds += pn_tt_now_seconds() - started;
    for (m = 0; m < members; m++) {
        const double at = pn_tt_now_seconds();

        if (pn_tt_residency_await(set[m], PN_TT_OP_TIMEOUT_MS) != 0)
            failed = bad[m] = 1;
        set[m]->seconds += pn_tt_now_seconds() - at;
    }
    if (!failed)
        return 0;
    for (m = 0; m < members; m++) {
        if (!bad[m])
            continue;
        if (m == 0u) {
            pn_tt_residency_stop(&slot->residency);
            pn_tt_residency_release(&slot->residency);
            device->oncard_restarts++;
            if (!pn_tt_arm_head_norm_slot(device, index))
                return -1;
        } else if (!pn_tt_head_norm_member_rearm(device, index, m)) {
            return -1;
        }
        if (pn_tt_residency_submit(set[m], FT_TT_QUEUE_OP_REDUCE, words[m],
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    return 0;
}

/**
 * The whole of a per-head reduce, over operands the caller has already described.
 *
 * Taking OPERANDS rather than a strip type is what lets one body serve both pages: an fp32
 * strip and a bf16 activation differ in the page size and in nothing else this function
 * needs, and the page it must agree with is the SLOT's — checked here, because a 2048-byte
 * operand under a 4096-byte program is the failure with no card-side refusal anywhere.
 */
static int pn_tt_head_norm_submit(pn_tt_device *device, ft_tt_op_reduce_function function,
                                  const pn_tt_norm_constants *constants, uint32_t heads,
                                  ft_tt_op_operand input, ft_tt_op_operand output,
                                  uint32_t input_tiles, uint32_t output_tiles)
{
    ft_tt_op_reduce_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    pn_tt_head_norm_residency *slot;
    uint32_t tiles;
    uint32_t index;
    double started;

    if (!device || !constants || heads == 0u)
        return -1;
    /* THE CONSTANTS CHOOSE THE CORE. They were built for one width and carry it, so routing
     * on them is what makes "the wrong residency" unrepresentable rather than merely
     * unlikely — a reduce at another width would read this buffer's gamma at the wrong
     * stride and return a smoothly wrong norm with nothing refusing. */
    slot = pn_tt_head_norm_slot(device, constants->tile_cols);
    if (!slot) {
        fprintf(stderr, "pn_tt: no per-head reduce residency is placed for Wt=%u\n",
                constants->tile_cols);
        return -1;
    }
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    tiles = heads * constants->tile_cols;
    if (input_tiles < tiles || output_tiles < tiles) {
        fprintf(stderr, "pn_tt: a %u-head norm at Wt=%u needs %u tiles; the strips hold %u "
                        "and %u\n", heads, constants->tile_cols, tiles, input_tiles,
                output_tiles);
        return -1;
    }
    if (input.tile_bytes != slot->tile_bytes || output.tile_bytes != slot->tile_bytes) {
        fprintf(stderr, "pn_tt: a Wt=%u head norm was handed %u/%u-byte operands under a "
                        "%u-byte program — the page size is a three-sided contract with no "
                        "card-side check\n", constants->tile_cols, input.tile_bytes,
                output.tile_bytes, slot->tile_bytes);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    request.function = function;
    /* ONE ROW BLOCK PER HEAD. The reader reads row block `b` at tiles `b*tile_cols + wt`,
     * so over a [1, heads*W] strip block `h` is head `h`'s W elements — all of them in
     * ROW 0, where the projection that produced them left them. pn_tt_head_norm.h has the
     * argument; this line is the whole of the implementation of it. */
    request.row_blocks = heads;
    request.tile_cols = constants->tile_cols;
    request.prefetch_tiles = constants->tile_cols;
    request.input = input;
    request.constants.x = (uint16_t)device->dram[constants->bank].x;
    request.constants.y = (uint16_t)device->dram[constants->bank].y;
    request.constants.address = constants->address;
    request.constants.tile_bytes = slot->tile_bytes;
    request.output = output;
    if (ft_tt_op_reduce_parameters(&request, words) != FT_TT_OK)
        return -1;
    pn_tt_vec_probe_apply(words);

    started = pn_tt_now_seconds();
    index = (uint32_t)(slot - &device->head_norm[0]);
    if (slot->members > 1u && (heads % slot->members) == 0u) {
        if (pn_tt_head_norm_submit_split(device, slot, index, &request) != 0)
            return -1;
        pn_tt_oncard_account(device, "head-norm",
                             function == FT_TT_OP_REDUCE_FN_L2_NORM ? "l2_norm" : "rms_norm",
                             slot->tile_cols, pn_tt_now_seconds() - started);
        return 0;
    }
    if (pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_REDUCE, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* The words address DRAM only, so the same descriptor is valid for a fresh
         * residency — the re-arm argument pn_tt_oncard.c makes for the other three. */
        fprintf(stderr, "pn_tt: the Wt=%u head-norm residency refused a descriptor — "
                        "retiring it by ASKING and arming a fresh one on core %u\n",
                slot->tile_cols, slot->residency.core_index);
        pn_tt_residency_stop(&slot->residency);
        pn_tt_residency_release(&slot->residency);
        device->oncard_restarts++;
        if (!pn_tt_arm_head_norm_slot(device, index) ||
            pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_REDUCE, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    pn_tt_oncard_account(device, "head-norm",
                         function == FT_TT_OP_REDUCE_FN_L2_NORM ? "l2_norm" : "rms_norm",
                         slot->tile_cols, pn_tt_now_seconds() - started);
    return 0;
}

int pn_tt_head_norm(pn_tt_device *device, ft_tt_op_reduce_function function,
                    const pn_tt_norm_constants *constants, uint32_t heads,
                    const pn_tt_fp32_strip *input, const pn_tt_fp32_strip *output)
{
    if (!device || !input || !output)
        return -1;
    return pn_tt_head_norm_submit(device, function, constants, heads,
                                  pn_tt_fp32_strip_operand(device, input),
                                  pn_tt_fp32_strip_operand(device, output),
                                  input->tile_count, output->tile_count);
}

int pn_tt_head_norm_bf16(pn_tt_device *device, ft_tt_op_reduce_function function,
                         const pn_tt_norm_constants *constants, uint32_t heads,
                         const pn_tt_activation *input, const pn_tt_activation *output)
{
    if (!device || !input || !output)
        return -1;
    return pn_tt_head_norm_submit(device, function, constants, heads,
                                  pn_tt_operand(device, input),
                                  pn_tt_operand(device, output), input->tile_count,
                                  output->tile_count);
}

int pn_tt_head_norm_operands(pn_tt_device *device, ft_tt_op_reduce_function function,
                             const pn_tt_norm_constants *constants, uint32_t heads,
                             ft_tt_op_operand input, ft_tt_op_operand output,
                             uint32_t input_tiles, uint32_t output_tiles)
{
    return pn_tt_head_norm_submit(device, function, constants, heads, input, output,
                                  input_tiles, output_tiles);
}

void pn_tt_head_norm_defer(pn_tt_device *device, uint32_t tile_cols, bool on)
{
    pn_tt_head_norm_residency *slot = device ? pn_tt_head_norm_slot(device, tile_cols) : NULL;

    if (!slot)
        return;
    slot->residency.defer_await = on;
    for (uint32_t member = 1u; member < slot->members; member++)
        slot->member[member - 1u].defer_await = on;
}

int pn_tt_head_norm_settle(pn_tt_device *device, uint32_t tile_cols)
{
    pn_tt_head_norm_residency *slot = device ? pn_tt_head_norm_slot(device, tile_cols) : NULL;
    int rc;

    if (!slot)
        return 0;
    rc = pn_tt_residency_settle(&slot->residency, PN_TT_OP_TIMEOUT_MS);
    for (uint32_t member = 1u; member < slot->members; member++)
        if (pn_tt_residency_settle(&slot->member[member - 1u], PN_TT_OP_TIMEOUT_MS) != 0)
            rc = -1;
    return rc;
}

#endif /* USE_TT */
