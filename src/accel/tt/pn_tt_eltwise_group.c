/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_eltwise_group.c
 * @brief Implementation of the elementwise core split. See pn_tt_eltwise_group.h — the core
 *        map, the knobs and the argument for why this family splits cleanly live there, and
 *        the tile arithmetic lives in pn_tt_eltwise_split.c where a test can reach it.
 *
 * @status Behind `PN_TT_ELTWISE_CORES`, default 1.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_eltwise_group.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_name_list.h"

/** Members, and where they start. See the header for why 106 cannot collide.
 *
 *  **DEFAULT 2, AND 2 IS THE KNEE — MEASURED 2026-08-24 ON THE SHIPPED BLOB.** The wall over
 *  the ladder is 5.01 / **5.17** / 4.20 / 3.18 tok/s at N = 1 / 2 / 8 / 16, against a control
 *  re-taken the same hour on the same binary. It is not a shape anyone predicted, and the
 *  census fit says exactly why:
 *
 *      N        1        2        8       16
 *      us/tile  1.999    0.998    0.000   -0.001
 *      fixed   -100.5   +42.2   +563.0  +1125.2   us/descriptor
 *
 *  **THE PER-TILE TERM SCALES PERFECTLY** — 1.999 -> 0.998 is 2.00x at two members, and by
 *  eight there is no per-tile term left at all. The cores are not the bind. What stops it is
 *  the INTERCEPT, which grows ~+75-85 us for every member added, flat, whatever the width.
 *  So a member pays for itself only on a descriptor wide enough: saving is
 *  `1.999 * W * (1/N - 1/(N+1))` against ~80 us, i.e. **W >= ~143 tiles at the first
 *  doubling** — and Ornith's elementwise widths are 128 and 384. Measured per width at two
 *  members: SwiGLU Wt=384 **667.2 -> 425.6 us (1.57x)**, and the residual ADD at Wt=128
 *  **110.0 -> 147.0 us, WORSE**, exactly as that line predicts. At four members the
 *  break-even is 572 tiles and nothing in the family is that wide.
 *
 *  **THE INTERCEPT IS THE HOST'S SERIAL ENQUEUE, AND THE PER-MEMBER AWAITS PROVE IT.** Every
 *  member above 0 reports **0.0079-0.0080 s/token** — identical to three digits at N=2 and at
 *  N=16, where its slice is eight times smaller. A cost that does not move with the work is
 *  not the card. It is 58 us a descriptor of parameter write, ring write and head bump that
 *  the host pays for member `m` before member `m+1` is told anything. Member 0's own await
 *  falls 0.0202 -> 0.0081 across that same ladder, which is its work halving away.
 *
 *  So the lever above this one is NOT more cores: it is the ISSUE path — the parameter cache
 *  `pn_tt_matmul_group.c` already uses to make a repeated projection cost zero device
 *  accesses, applied to a family that re-writes 64 bytes for all 136 descriptors of a token.
 *
 *  **THE BLOCK DEPTH IS NOT THE CONFOUND, AND THAT WAS MEASURED RATHER THAN ASSUMED.** The
 *  depth moves with the member count (32 unsplit, 8 at eight members), so the ladder walks
 *  two knobs. `PN_TT_ELTWISE_DEPTH=8` at ONE member is 5.02 tok/s and fits -103.5 + 2.003,
 *  against the control's 5.01 and -100.5 + 1.999. The depth is worth **0.0%**. */
#define PN_TT_ELTWISE_CORES_ENV           "PN_TT_ELTWISE_CORES"
#define PN_TT_ELTWISE_CORES_DEFAULT       2u
#define PN_TT_ELTWISE_MEMBER_CORE_ENV     "PN_TT_ELTWISE_MEMBER_CORE_FIRST"
/* 112 since the PX7 placement shipped (was 106, inside the 16-member matmul's 96..110). */
#define PN_TT_ELTWISE_MEMBER_CORE_DEFAULT 112u

/**
 * PHASE W1a — WIDE MEMBERS for the prefill's whole-prompt descriptors.
 *
 * A 1,629-row prefill's SwiGLU is ONE descriptor of 51 x 384 = 19,584 tiles: 15.65 ms on the
 * two decode members, 33 of them a prompt = 0.52 s of the 1.57 s `batched` line, while the
 * free Tensix of the 0-70 region sit idle (2026-10-07). The fit above says the per-tile term
 * scales perfectly and the cost of a member is its ~45 us serial enqueue, so a descriptor that
 * long pays for many members where a decode one pays for two.
 *
 * `PN_TT_ELTWISE_WIDE_CORES=N` arms N MORE eltwise residencies from
 * `PN_TT_ELTWISE_WIDE_CORE_FIRST` (58: 58..70 is clear of every family and of the 48-core
 * prefill grid on 0..47), at member 0's placement. An SFPU descriptor of at least
 * `PN_TT_ELTWISE_WIDE_MIN_TILES` (2048; 16x that for ADD/MUL) goes over EVERY member, dealt out in whole blocks of
 * `2 * depth` (`pn_tt_eltwise_split_slice_blocks()`), so no width has to divide; every
 * narrower descriptor takes the decode split exactly as before. Same per-element arithmetic
 * on a different core: bit-exact by construction. The mirrors of each wide slice are keyed in
 * that member's own replica table (`pn_tt_replica_map.c`).
 */
#define PN_TT_ELTWISE_WIDE_CORES_ENV      "PN_TT_ELTWISE_WIDE_CORES"
#define PN_TT_ELTWISE_WIDE_CORE_ENV       "PN_TT_ELTWISE_WIDE_CORE_FIRST"
#define PN_TT_ELTWISE_WIDE_CORE_DEFAULT   58u
#define PN_TT_ELTWISE_WIDE_MIN_TILES_ENV  "PN_TT_ELTWISE_WIDE_MIN_TILES"
#define PN_TT_ELTWISE_WIDE_MIN_TILES_DEFAULT 2048u

/** CAP on the block depth, so a member-count ladder can be run at a FIXED depth. The header
 *  has the confound this exists to separate. */
#define PN_TT_ELTWISE_DEPTH_ENV "PN_TT_ELTWISE_DEPTH"

/**
 * WHICH FUNCTIONS THE SPLIT IS APPLIED TO — because THE KNEE WAS NEVER A WIDTH, AND IT MOVES
 * WITH THE COMPLETION SHAPE.
 *
 * RE-MEASURED ON SILICON 2026-08-26, per SHAPE, at one member and at two, on TWO completion
 * shapes. `K` is the break-even recovered from each pair by `T(2) = T(1)/2 + K/2`:
 *
 *                          ---- window, 5 settle reads ----   ---- line, 0 ----
 *      shape                 N=1      N=2      K                N=1     N=2     K
 *      SILU_MUL Wt=384     634.5    361.0    87.5             619.8   332.6   45.4
 *      SILU_MUL Wt=128     235.3    161.7    88.1             220.1   132.6   45.1
 *      SIGMOID  Wt=128     203.6    144.6    85.6             189.0   115.7   42.4
 *      ADD      Wt=128      77.5     82.5    87.5 (LOSES)      62.3    53.5   44.7 (WINS)
 *      render-widen Wt=256 125.4    105.1    84.6             110.3    77.1   43.9
 *      -------------------------------------------------------------------------------
 *                                   **87.2 +- 1.3**                   **44.3 +- 1.1**
 *
 * **TWO SHAPES AT Wt=128 WIN AND TWO AT Wt=128 LOSE on the window, so the discriminator is
 * not the width** — the recorded "W >= 143 tiles" was this statement seen through the
 * family's pooled line (which re-measures at W >= 110). A descriptor is worth splitting in
 * two iff its UNSPLIT TIME is above K.
 *
 * **AND K IS A PROPERTY OF THE AWAIT, NOT OF THE FAMILY.** 87.2 -> 44.3 when the completion
 * shape went `window` -> `line`: 42.9 us, i.e. 21.5 us off each of the two members, against
 * the 5 x 3.31 = 16.6 us of settle reads the line removes. Five shapes across TWO families
 * (`eltwise` and `render-widen`, different programs at different page sizes) recover the same
 * constant to +-2.5%, twice.
 *
 * **SO ON THE SHIPPED `line` DEFAULT EVERY SHAPE IN THIS FAMILY IS ABOVE THE BAR** — the
 * cheapest is the residual ADD at 62.3 us against K = 44.3 — and the filter DEFAULTS TO
 * UNRESTRICTED. It is kept, with the table, because it is the only way to run the family
 * under `window` or `control` without paying 8.8 us on each of 64 residual adds, and because
 * a knee that moves with another knob is exactly the thing a later round will re-derive
 * wrongly from one row of this table.
 *
 * A LIST AND NOT A THRESHOLD, deliberately. A threshold read out of the live census would
 * make some descriptors of one key split and others not, and the key's mean is what the
 * `fixed + per_tile * tiles` fit is solved from — the instrument this whole finding came out
 * of would start reporting a blend.
 *
 * CONTAINS, NOT SELECTS: this is a FILTER, so an unset or empty value does not restrict.
 * `PN_TT_ELTWISE_SPLIT_FUNCTIONS=NOTHING_MATCHES_THIS` is the arm that proves it is live —
 * measured, it puts member 1 on **0.0 descriptors/token** and every census row back on the
 * one-member line (per_tile 0.977 -> 2.000).
 */
#define PN_TT_ELTWISE_SPLIT_FUNCTIONS_ENV "PN_TT_ELTWISE_SPLIT_FUNCTIONS"

/** Is `function` one the split pays for? See the table above.
 *
 *  READ ONCE. A `getenv` here is 136 calls a token on this family alone, and the knob cannot
 *  change inside a run. The names are compared WHOLE against
 *  `ft_tt_op_vector_function_name()`, which returns "SILU_MUL"/"SIGMOID"/"ADD"/"MUL" —
 *  UPPERCASE, so a lowercase list matches nothing, splits nothing, and reads exactly like a
 *  knob that is off. */
static bool eltwise_split_covers(const char *function)
{
    static const char *list;
    static bool cached;

    if (!cached) {
        cached = true;
        list = getenv(PN_TT_ELTWISE_SPLIT_FUNCTIONS_ENV);
    }
    return pn_tt_name_list_contains(list, function);
}

uint32_t pn_tt_eltwise_group_members(const pn_tt_device *device)
{
    if (!device || device->eltwise_members == 0u)
        return 1u;
    return device->eltwise_members;
}

bool pn_tt_device_eltwise_member_stat(const pn_tt_device *device, unsigned index,
                                      uint32_t *out_core, uint64_t *out_ops, double *out_seconds)
{
    const pn_tt_residency *member;

    if (!device || index >= pn_tt_eltwise_group_members(device))
        return false;
    member = index == 0u ? &device->eltwise : &device->eltwise_member[index - 1u];
    if (out_core)
        *out_core = member->core_index;
    if (out_ops)
        *out_ops = member->ops;
    if (out_seconds)
        *out_seconds = device->eltwise_member_seconds[index];
    return true;
}

static void eltwise_group_arm_wide(pn_tt_device *device,
                                   const ft_tt_op_eltwise_request *placement);

/** The widest width any descriptor of this residency carries. */
static uint32_t eltwise_widest(const uint32_t *tile_counts, uint32_t count)
{
    uint32_t widest = 0u, index;

    for (index = 0; index < count; index++) {
        if (tile_counts[index] > widest)
            widest = tile_counts[index];
    }
    return widest;
}

void pn_tt_eltwise_group_plan(const pn_tt_device *device, const uint32_t *tile_counts,
                              uint32_t count, uint32_t *out_members, uint32_t *out_widest,
                              uint32_t *out_depth)
{
    uint32_t sliced[PN_TT_ONCARD_MAX_ELTWISE_SHAPES];
    uint32_t requested, members, cap, depth = 0u, index;

    if (!out_members || !out_widest || !out_depth)
        return;
    *out_members = 1u;
    *out_widest = 0u;
    *out_depth = 0u;
    if (!device || !tile_counts || count == 0u || count > PN_TT_ONCARD_MAX_ELTWISE_SHAPES)
        return;

    cap = pn_tt_env_u32(PN_TT_ELTWISE_DEPTH_ENV, PN_TT_ONCARD_MAX_DEPTH);
    if (cap == 0u || cap > PN_TT_ONCARD_MAX_DEPTH)
        cap = PN_TT_ONCARD_MAX_DEPTH;
    requested = pn_tt_env_u32(PN_TT_ELTWISE_CORES_ENV, PN_TT_ELTWISE_CORES_DEFAULT);
    /* THE TWO INSTRUMENTS DO NOT COMPOSE, AND THE SPLIT IS THE ONE THAT YIELDS. The
     * queue-depth ladder counts K copies of one descriptor through
     * `pn_tt_residency_submit()`; the split enqueues ONE descriptor per member directly, so
     * `PN_TT_ONCARD_BATCH` would be inert here and its ladder would print a flat line
     * produced by this file. Refusing names that. */
    if (requested > 1u && pn_tt_env_u32("PN_TT_ONCARD_BATCH", 0u) > 1u &&
        (!getenv("PN_TT_ONCARD_BATCH_FAMILY") ||
         pn_tt_name_list_contains(getenv("PN_TT_ONCARD_BATCH_FAMILY"), "eltwise"))) {
        fprintf(stderr, "pn_tt: REFUSING the eltwise split — PN_TT_ONCARD_BATCH is armed for "
                        "this family and the split does not go through the batched submit, so "
                        "the ladder would measure nothing. Run one instrument at a time.\n");
        requested = 1u;
    }
    members = pn_tt_eltwise_split_member_limit(requested, tile_counts, count,
                                               PN_TT_ELTWISE_MAX_MEMBERS);
    /* THE DEPTH IS PART OF WHETHER A MEMBER COUNT IS LEGAL, not a consequence of it. A split
     * whose slices no block depth serves is a descriptor the kernel refuses, so it shrinks
     * here rather than on the card. */
    while (members > 1u) {
        for (index = 0; index < count; index++)
            sliced[index] = tile_counts[index] / members;
        depth = pn_tt_common_depth(sliced, count, cap);
        if (depth != 0u)
            break;
        members = pn_tt_eltwise_split_member_limit(members - 1u, tile_counts, count,
                                                   PN_TT_ELTWISE_MAX_MEMBERS);
    }
    if (members > 1u) {
        *out_widest = eltwise_widest(sliced, count);
    } else {
        depth = pn_tt_common_depth(tile_counts, count, cap);
        *out_widest = eltwise_widest(tile_counts, count);
    }
    if (requested > 1u && members != requested)
        fprintf(stderr, "pn_tt: eltwise SPLIT — %u members was asked for; %u is the largest "
                        "that divides every width at a legal block depth\n",
                requested, members);
    *out_members = members;
    *out_depth = depth;
}

/** The placement every member of this device was armed at, rebuilt from what the device kept.
 *  It is DERIVED rather than stored because a stored copy is a second thing that must agree
 *  with `eltwise_depth` and the shape table, and this backend has already paid for a field
 *  that could disagree with the one beside it. */
static ft_tt_op_eltwise_request eltwise_placement(const pn_tt_device *device)
{
    ft_tt_op_eltwise_request placement;

    memset(&placement, 0, sizeof(placement));
    placement.function = FT_TT_OP_VECTOR_FN_SILU_MUL; /* the widest of the six */
    /* THE DECODE SPLIT'S slice, not the whole set's: the arm placed every member (wide ones
     * included) for that width, and the block depth is derived from it — dividing by the
     * W1a set's 15 placed a re-armed member at another depth, and every descriptor after an
     * idle re-arm was refused (run_w1a_eltwise_wide.sh round 1). */
    placement.tile_count = eltwise_widest(device->oncard_eltwise_tiles,
                                          device->oncard_eltwise_count) /
                           (device->eltwise_decode_members ? device->eltwise_decode_members
                                                           : pn_tt_eltwise_group_members(device));
    placement.prefetch_tiles = device->eltwise_depth;
    placement.input_a.tile_bytes = PN_TT_TILE_BF16;
    placement.input_b.tile_bytes = PN_TT_TILE_BF16;
    placement.output.tile_bytes = PN_TT_TILE_BF16;
    return placement;
}

void pn_tt_eltwise_group_arm(pn_tt_device *device, uint32_t members,
                             const ft_tt_op_eltwise_request *placement)
{
    uint32_t first, index;

    if (!device || !placement)
        return;
    pn_tt_eltwise_group_release(device);
    memset(device->eltwise_member_seconds, 0, sizeof(device->eltwise_member_seconds));
    device->eltwise_members = 1u;
    device->eltwise_decode_members = 1u;
    if (members < 2u) {
        eltwise_group_arm_wide(device, placement);
        return;
    }
    first = pn_tt_env_u32(PN_TT_ELTWISE_MEMBER_CORE_ENV, PN_TT_ELTWISE_MEMBER_CORE_DEFAULT);
    for (index = 0; index + 1u < members; index++) {
        pn_tt_residency *member = &device->eltwise_member[index];
        ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];

        if (!pn_tt_prepare_family_on_core(device, member, PN_TT_ELTWISE_PROGRAM,
                                          "eltwise-member", PN_TT_ELTWISE_MEMBER_CORE_ENV,
                                          first + index, PN_TT_ONCARD_TRIAD_MASK))
            break;
        if (ft_tt_op_eltwise_place_cbs(placement, placed, member->cb_buffer,
                                       sizeof(member->cb_buffer), &member->cb_mask) !=
                FT_TT_OK ||
            !pn_tt_start_family(member)) {
            pn_tt_residency_release(member);
            break;
        }
        device->eltwise_members++;
    }
    /* COLLAPSE, DO NOT SHRINK. The header has the argument: a partial set is a member count
     * the widths no longer divide, which is a ragged slice and a refused descriptor. */
    if (device->eltwise_members != members) {
        fprintf(stderr, "pn_tt: eltwise SPLIT — asked for %u members and %u armed; COLLAPSING "
                        "to one core, because a partial split is a width no member divides\n",
                members, device->eltwise_members);
        pn_tt_eltwise_group_release(device);
        eltwise_group_arm_wide(device, placement);
        return;
    }
    fprintf(stderr, "pn_tt: eltwise SPLIT — every descriptor over %u members, core %u then "
                    "%u..%u, at block depth %u\n",
            device->eltwise_members, device->eltwise.core_index, first,
            first + device->eltwise_members - 2u, device->eltwise_depth);
    eltwise_group_arm_wide(device, placement);
}

/** W1a: arm the WIDE members after the decode split. A member that fails to arm stops the
 *  set where it is — the wide split is ragged by blocks, so any count is legal. */
static void eltwise_group_arm_wide(pn_tt_device *device, const ft_tt_op_eltwise_request *placement)
{
    /* DEFAULT 13 (W1a, run_w1a_eltwise_wide.sh 2026-10-08): serve 1,630 TTFT 4.02 -> 3.25 s
     * with the reader thread, decode flat; `PN_TT_ELTWISE_WIDE_CORES=0` off. */
    const uint32_t want = pn_tt_env_u32(PN_TT_ELTWISE_WIDE_CORES_ENV, 13u);
    const uint32_t first = pn_tt_env_u32(PN_TT_ELTWISE_WIDE_CORE_ENV,
                                         PN_TT_ELTWISE_WIDE_CORE_DEFAULT);
    uint32_t index;

    device->eltwise_decode_members = device->eltwise_members;
    for (index = 0; index < want && device->eltwise_members < PN_TT_ELTWISE_MAX_MEMBERS;
         index++) {
        pn_tt_residency *member = &device->eltwise_member[device->eltwise_members - 1u];
        ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];

        if (!pn_tt_prepare_family_on_core(device, member, PN_TT_ELTWISE_PROGRAM,
                                          "eltwise-member", PN_TT_ELTWISE_WIDE_CORE_ENV,
                                          first + index, PN_TT_ONCARD_TRIAD_MASK))
            break;
        if (ft_tt_op_eltwise_place_cbs(placement, placed, member->cb_buffer,
                                       sizeof(member->cb_buffer), &member->cb_mask) !=
                FT_TT_OK ||
            !pn_tt_start_family(member)) {
            pn_tt_residency_release(member);
            break;
        }
        device->eltwise_members++;
    }
    if (want)
        fprintf(stderr, "pn_tt: eltwise WIDE — %u of %u extra members armed from core %u; "
                        "descriptors of %u+ tiles run over all %u (block %u)\n",
                device->eltwise_members - device->eltwise_decode_members, want, first,
                pn_tt_env_u32(PN_TT_ELTWISE_WIDE_MIN_TILES_ENV,
                              PN_TT_ELTWISE_WIDE_MIN_TILES_DEFAULT),
                device->eltwise_members, 2u * device->eltwise_depth);
}

uint32_t pn_tt_eltwise_group_members_for(const pn_tt_device *device, uint32_t tile_count,
                                         uint32_t function)
{
    static uint32_t min_wide;
    uint32_t min_tiles;

    if (!device)
        return 1u;
    if (min_wide == 0u)
        min_wide = pn_tt_env_u32(PN_TT_ELTWISE_WIDE_MIN_TILES_ENV,
                                 PN_TT_ELTWISE_WIDE_MIN_TILES_DEFAULT);
    /* THE SFPU FUNCTIONS cost ~0.8 us a tile a member; ADD/MUL/SUB ~0.06 — a wide split of
     * those pays its ~45 us a member enqueue only 13x wider (the residual ADD at Wt=6528 is
     * 390 us on two). */
    min_tiles = (function == FT_TT_OP_VECTOR_FN_SILU_MUL || function == FT_TT_OP_VECTOR_FN_SILU ||
                 function == FT_TT_OP_VECTOR_FN_SIGMOID)
                    ? min_wide : 16u * min_wide;
    if (device->eltwise_decode_members == 0u ||
        device->eltwise_members <= device->eltwise_decode_members || tile_count < min_tiles ||
        device->eltwise_depth == 0u ||
        tile_count / (2u * device->eltwise_depth) < device->eltwise_members ||
        (tile_count % (2u * device->eltwise_depth)) != 0u)
        return device->eltwise_decode_members ? device->eltwise_decode_members
                                              : pn_tt_eltwise_group_members(device);
    return device->eltwise_members;
}

bool pn_tt_eltwise_group_slice(const pn_tt_device *device, uint32_t tile_count,
                               uint32_t members, uint32_t member, uint32_t tile_bytes,
                               pn_tt_eltwise_slice *out_slice)
{
    if (members > 1u && members == device->eltwise_members &&
        members > device->eltwise_decode_members && device->eltwise_decode_members != 0u) {
        /* RED `eltwise_wide_shift`: a wide member's slice starts one block late — the slices
         * then overlap and leave the last block unwritten; must DIFFER. */
        if (!pn_tt_eltwise_split_slice_blocks(tile_count, members, member,
                                              2u * device->eltwise_depth, tile_bytes, out_slice))
            return false;
        if (member > 0u && out_slice->tile_count && pn_tt_red("eltwise_wide_shift") &&
            out_slice->first_tile + out_slice->tile_count < tile_count) {
            out_slice->first_tile += 2u * device->eltwise_depth;
            out_slice->byte_offset = out_slice->first_tile * tile_bytes;
        }
        return true;
    }
    return pn_tt_eltwise_split_slice(tile_count, members, member, tile_bytes, out_slice);
}

void pn_tt_eltwise_group_release(pn_tt_device *device)
{
    uint32_t index;

    if (!device)
        return;
    for (index = 1u; index < device->eltwise_members; index++)
        pn_tt_residency_release(&device->eltwise_member[index - 1u]);
    device->eltwise_members = device->eltwise.running ? 1u : 0u;
    device->eltwise_decode_members = device->eltwise_members;
}

/**
 * Retire ONE member of the set and arm a fresh one ON THE CORE IT ALREADY HELD.
 *
 * The core is captured BEFORE the release, because re-reading `core_index` off a released
 * handle is how a member gets armed on core 0. Member 0 goes through the same code against
 * its own env name — the alternative, `pn_tt_oncard_rearm()`, re-places the WHOLE set from
 * scratch, which is the wrong operation in the middle of a descriptor whose other members
 * have already retired.
 *
 * @return true when a fresh residency is live on that core and the slice may be replayed
 */
static bool eltwise_member_rearm(pn_tt_device *device, uint32_t member_index,
                                 const ft_tt_op_eltwise_request *placement)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
    const char *core_env = member_index == 0u ? PN_TT_CORE_ELTWISE_ENV
                                              : PN_TT_ELTWISE_MEMBER_CORE_ENV;
    const char *name = member_index == 0u ? "eltwise" : "eltwise-member";
    pn_tt_residency *member;
    uint32_t core;

    if (member_index >= pn_tt_eltwise_group_members(device))
        return false;
    member = member_index == 0u ? &device->eltwise : &device->eltwise_member[member_index - 1u];
    core = member->core_index;
    fprintf(stderr, "pn_tt: eltwise member %u refused a descriptor — retiring it by ASKING and "
                    "arming a fresh one on core %u\n", member_index, core);
    pn_tt_residency_stop(member);
    pn_tt_residency_release(member);
    device->oncard_restarts++;
    /* A core whose compute thread is still spinning out its bound is a WEDGE, not a re-arm:
     * booting firmware onto it leaves `instrn_buf_status = 0x66` and needs a rung-3 recovery
     * and sudo. Every other family here refuses on the same reading. */
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: eltwise member %u did not retire — REFUSING to arm a fresh one "
                        "on core %u\n", member_index, core);
        return false;
    }
    if (!pn_tt_prepare_family_on_core(device, member, PN_TT_ELTWISE_PROGRAM, name, core_env,
                                      core, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_eltwise_place_cbs(placement, placed, member->cb_buffer,
                                   sizeof(member->cb_buffer), &member->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(member);
        return false;
    }
    return pn_tt_start_family(member);
}

/** Advance one operand's base to `slice`. An UNSET operand — a unary descriptor's B, which
 *  the encoder zeroes and the kernel never reads — is left alone, because offsetting it
 *  would build an address out of nothing. */
static void eltwise_slice_operand(ft_tt_op_operand *operand, const pn_tt_eltwise_slice *slice)
{
    if (operand->tile_bytes == 0u)
        return;
    operand->address += slice->first_tile * operand->tile_bytes;
}

/** Encode member `index`'s slice of `request`. @return false with the reason on stderr. */
static bool eltwise_slice_words(const pn_tt_device *device,
                                const ft_tt_op_eltwise_request *request, uint32_t members,
                                uint32_t index, uint32_t *out_words)
{
    ft_tt_op_eltwise_request sliced = *request;
    pn_tt_eltwise_slice slice;

    if (!pn_tt_eltwise_group_slice(device, request->tile_count, members, index,
                                   PN_TT_TILE_BF16, &slice) || slice.tile_count == 0u) {
        fprintf(stderr, "pn_tt: %u eltwise tiles do not divide over %u members\n",
                request->tile_count, members);
        return false;
    }
    sliced.tile_count = slice.tile_count;
    sliced.prefetch_tiles = device->eltwise_depth;
    eltwise_slice_operand(&sliced.input_a, &slice);
    if (ft_tt_op_vector_function_is_binary(request->function))
        eltwise_slice_operand(&sliced.input_b, &slice);
    eltwise_slice_operand(&sliced.output, &slice);
    if (ft_tt_op_eltwise_parameters(&sliced, out_words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the eltwise slice [%u, %u) is one the family refuses\n",
                slice.first_tile, slice.first_tile + slice.tile_count);
        return false;
    }
    pn_tt_vec_probe_apply(out_words);
    return true;
}

/** The member residency `index` of the set (member 0 is `eltwise` itself). */
static pn_tt_residency *eltwise_member_residency(pn_tt_device *device, uint32_t index)
{
    return index == 0u ? &device->eltwise : &device->eltwise_member[index - 1u];
}

/**
 * Await every member of an enqueued descriptor, then re-arm and replay any member that was
 * refused, then charge the set. `words[index]` is member `index`'s slice.
 *
 * PER-MEMBER RE-ARM AND REPLAY. Replaying one member's slice is the SAME COMPUTATION and not
 * merely a safe one: an elementwise descriptor is a pure function of operands it does not
 * modify, and the slices are disjoint, so a member that succeeded is neither re-run nor read.
 * A refused descriptor also leaves no circular-buffer traffic behind it — the reader aborts
 * before the rebase. (A deferred descriptor keeps this property only while its operands are
 * untouched: the caller settles it before anything rewrites them.)
 */
static int eltwise_group_complete(pn_tt_device *device, const ft_tt_op_eltwise_request *request,
                                  uint32_t (*words)[FT_TT_OP_VECTOR_WORDS], uint32_t members,
                                  bool *bad, double started)
{
    uint32_t index;
    int failed = 0;

    /* EVERY member is awaited even after one has failed. An enqueue that got as far as the
     * ring has a descriptor the card will run whatever this host does next, and leaving it
     * un-awaited desynchronises that member's cursor for the rest of the run. */
    for (index = 0; index < members; index++) {
        pn_tt_residency *member = eltwise_member_residency(device, index);
        const double member_started = pn_tt_now_seconds();
        double member_elapsed;

        if (pn_tt_residency_await(member, PN_TT_OP_TIMEOUT_MS) != 0)
            bad[index] = true;
        failed |= bad[index];
        member_elapsed = pn_tt_now_seconds() - member_started;
        device->eltwise_member_seconds[index] += member_elapsed;
        member->seconds += member_elapsed;
    }
    if (failed) {
        const ft_tt_op_eltwise_request placement = eltwise_placement(device);

        failed = 0;
        for (index = 0; index < members; index++) {
            if (!bad[index])
                continue;
            if (!eltwise_member_rearm(device, index, &placement) ||
                pn_tt_residency_submit(eltwise_member_residency(device, index),
                                       FT_TT_QUEUE_OP_ELTWISE, words[index],
                                       PN_TT_OP_TIMEOUT_MS) != 0)
                failed = 1;
        }
    }
    /* Charged to the SET, once, at the WHOLE descriptor's tile count — so the census key is
     * the same row at every member count and its fit is comparable across the ladder. */
    pn_tt_oncard_account(device, "eltwise", ft_tt_op_vector_function_name(request->function),
                         request->tile_count, pn_tt_now_seconds() - started);
    return failed ? -1 : 0;
}

void pn_tt_eltwise_group_defer_next(pn_tt_device *device)
{
    const char *env = getenv("PN_TT_ELTWISE_DEFER");

    if (device && !(env && *env && atoi(env) == 0))
        device->eltwise_defer_next = true;
}

uint64_t pn_tt_eltwise_group_deferred(const pn_tt_device *device)
{
    return device ? device->eltwise_deferred : 0u;
}

int pn_tt_eltwise_group_settle(pn_tt_device *device)
{
    bool bad[PN_TT_ELTWISE_MAX_MEMBERS];
    uint32_t members;

    if (!device || device->eltwise_pending_members == 0u)
        return 0;
    members = device->eltwise_pending_members;
    device->eltwise_pending_members = 0u;
    memset(bad, 0, sizeof(bad));
    return eltwise_group_complete(device, &device->eltwise_pending_request,
                                  device->eltwise_pending_words, members, bad,
                                  device->eltwise_pending_started);
}

int pn_tt_eltwise_group_submit(pn_tt_device *device, const ft_tt_op_eltwise_request *request)
{
    uint32_t words[PN_TT_ELTWISE_MAX_MEMBERS][FT_TT_OP_VECTOR_WORDS];
    bool bad[PN_TT_ELTWISE_MAX_MEMBERS];
    uint32_t members, index;
    double started;
    bool wide, defer, enqueue_failed = false;

    if (!device || !request)
        return -1;
    defer = device->eltwise_defer_next;
    device->eltwise_defer_next = false;
    /* The ring is FIFO, but the bookkeeping is not: a pending descriptor is awaited first. */
    if (pn_tt_eltwise_group_settle(device) != 0)
        return -1;
    members = pn_tt_eltwise_group_members_for(device, request->tile_count,
                                              (uint32_t)request->function);
    wide = members > device->eltwise_decode_members && device->eltwise_decode_members != 0u;
    /* A WIDTH THE SET DOES NOT DIVIDE RUNS WHOLE ON MEMBER 0 rather than failing. The arm
     * planned the split over every width the residency DECLARED, and a caller that reaches
     * here with another one has already passed the depth guard — so member 0's placement
     * serves it, and one core is slower rather than wrong. Failing instead would turn a new
     * call site into a dropped descriptor, which on this family is a wrong answer with a
     * return code nobody reads. Said once, because 136 descriptors a token is a flood. */
    if (!wide && (request->tile_count % members) != 0u) {
        static bool narrowed;

        if (!narrowed) {
            narrowed = true;
            fprintf(stderr, "pn_tt: %u eltwise tiles do not divide over %u members — this "
                            "width runs WHOLE on member 0; the split was planned for the "
                            "widths the residency was armed with\n",
                    request->tile_count, members);
        }
        members = 1u;
    }
    /* AND A SHAPE BELOW THE BREAK-EVEN RUNS WHOLE ON MEMBER 0 TOO. The knee is a descriptor
     * TIME of ~87 us and not a width — the residual ADD (77.5 us) and the attention MUL
     * (69.0 us) are made SLOWER by a member, at the same Wt=128 where SILU_MUL and SIGMOID
     * are made 1.46x and 1.41x faster. See the table beside
     * PN_TT_ELTWISE_SPLIT_FUNCTIONS_ENV. Member 0's placement serves the whole width, so
     * this is one core rather than wrong. */
    if (members > 1u && !eltwise_split_covers(ft_tt_op_vector_function_name(request->function)))
        members = 1u;
    memset(bad, 0, sizeof(bad));
    for (index = 0; index < members; index++)
        if (!eltwise_slice_words(device, request, members, index, words[index]))
            return -1;

    /* RED `eltwise_defer_drop`: a deferred descriptor is never enqueued — its output is
     * whatever the strip held. Must DIFFER: proves the gate scores the deferred op. */
    if (defer && pn_tt_red("eltwise_defer_drop"))
        return 0;
    started = pn_tt_now_seconds();
    for (index = 0; index < members; index++)
        if (pn_tt_residency_enqueue(eltwise_member_residency(device, index),
                                    FT_TT_QUEUE_OP_ELTWISE, words[index], 1u) != 0)
            enqueue_failed = bad[index] = true;
    /* THE RESIDENCY CLOCKS RUN ON THIS PATH TOO (class 28: `ops != 0 && seconds == 0` read
     * as a free family). The enqueue is booked on member 0 and each member's serial await on
     * that member, so the rows SUM to the set's wall and no time is booked twice. */
    device->eltwise.seconds += pn_tt_now_seconds() - started;
    /* PM 7 DEFERRED: the awaits wait for pn_tt_eltwise_group_settle(). A failed enqueue is
     * completed (replayed) now instead, while the caller is still at this descriptor. */
    if (defer && !enqueue_failed) {
        memcpy(device->eltwise_pending_words, words, sizeof(words[0]) * members);
        device->eltwise_pending_request = *request;
        device->eltwise_pending_started = started;
        device->eltwise_pending_members = members;
        device->eltwise_deferred++;
        return 0;
    }
    return eltwise_group_complete(device, request, words, members, bad, started);
}

#endif /* USE_TT */
