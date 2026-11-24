/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_render.c
 * @brief Implementation of the activation strip bridge. See pn_tt_render.h — the argument for
 *        why this is two programs rather than one mixed CB table lives there.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_render.h"

#include <stdio.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_eltwise_split.h"
#include "pn_tt_group_pull.h"
#include "pn_tt_replica_map.h"

/**
 * Which enabled Tensix each direction drives.
 *
 * Defaults, not constants, and each must be a core no other residency holds AND that the
 * matmul group cannot grow into: members run upward from `PN_TT_MATMUL_CORE_FIRST` = 75, so
 * even `PN_TT_MATMUL_CORES=16` stops at 90. The rest of the map, which is why these two are
 * 85 and 86:
 *
 *     71        matmul member 0            72/73/74   reduce / eltwise / sample
 *     75..90    matmul members 1..15       82/83/84   rope / memory / attention
 *     85/86     render widen / narrow      90         gdn member 0
 *     91        head norm, slot 0          92..99     gdn head-split members
 *     103       head norm, slot 1          100..102   used by the rectangle arm's control
 *
 * 85 and 86 are inside the range sixteen matmul members would reach, and that is not an
 * oversight: `pn_tt_prepare_family()` refuses a collision BY NAME against every live
 * residency, so a run that wants both would be told rather than silently sharing a core. The
 * shipped setting is 8 members, which stops at 82.
 */
#define PN_TT_CORE_RENDER_WIDEN_ENV      "PN_TT_CORE_RENDER_WIDEN"
#define PN_TT_CORE_RENDER_WIDEN_DEFAULT  85u
#define PN_TT_CORE_RENDER_NARROW_ENV     "PN_TT_CORE_RENDER_NARROW"
#define PN_TT_CORE_RENDER_NARROW_DEFAULT 86u

#define PN_TT_RENDER_WIDEN_PROGRAM  "render_widen"
#define PN_TT_RENDER_NARROW_PROGRAM "render_narrow"

/**
 * THE SPLIT'S KNOBS, and where its members live.
 *
 * `PN_TT_RENDER_CORES=N` runs every render descriptor over N Tensix, member `m` taking a
 * contiguous slice `[m*W/N, (m+1)*W/N)` of A, of the zero B and of the output. It is the
 * SAME pattern the elementwise split runs and it needs no kernel change and no new opcode,
 * because `render_widen`/`render_narrow` ARE the elementwise program at two page sizes:
 * `resident_eltwise_reader.c` reads tile `t` at `base + t * TILE_BYTES`, unconditionally and
 * in order. `pn_tt_eltwise_split.h` holds the arithmetic and is shared rather than copied.
 *
 * THE MEMBERS RUN ABOVE EVERY CORE ANY OTHER FAMILY CAN REACH AT ANY LEGAL KNOB. The
 * elementwise split's map ends its own members at 121 and calls 122..139 free; the two
 * directions are split INDEPENDENTLY, so they take one block each:
 *
 *     122..128  render-widen members 1..7    129..135  render-narrow members 1..7
 *
 * That is a claim about the whole map and not about today's defaults, which is what
 * `pn_tt_core_holder()` turns into a refusal BY NAME — a contested core does not fail, it
 * computes badly (1.0e-2 against 6.9e-4 on the same descriptor).
 *
 * WHY THIS FAMILY AND NOT `reduce`: both `reduce` and `head_norm` are a row-wise reduction
 * into DST column 0, so a tile slice there needs a cross-core partial-sum combine — a kernel
 * and protocol change, not a host-side address offset. Render needs neither.
 */
#define PN_TT_RENDER_CORES_ENV             "PN_TT_RENDER_CORES"
#define PN_TT_RENDER_MEMBER_CORE_ENV       "PN_TT_RENDER_MEMBER_CORE_FIRST"
#define PN_TT_RENDER_MEMBER_CORE_DEFAULT   122u
/** CAP on the block depth, so a member-count ladder can be run at a FIXED depth. The depth
 *  MOVES WITH THE MEMBER COUNT — the slice shrinks and `2 * depth` must divide it — so a
 *  ladder walks two knobs at once and "it stopped at N" would not say which one stopped. */
#define PN_TT_RENDER_DEPTH_ENV             "PN_TT_RENDER_DEPTH"

/** Everything that differs between the two directions, in one row each. */
typedef struct render_direction {
    const char *program;
    const char *name;
    const char *member_name;
    const char *core_env;
    uint32_t core_default;
    uint32_t in_tile_bytes;
    uint32_t out_tile_bytes;
} render_direction;

static const render_direction g_render_directions[PN_TT_RENDER_DIRECTIONS] = {
    { PN_TT_RENDER_WIDEN_PROGRAM, "render-widen", "render-widen-member",
      PN_TT_CORE_RENDER_WIDEN_ENV, PN_TT_CORE_RENDER_WIDEN_DEFAULT,
      PN_TT_TILE_BF16, PN_TT_TILE_FP32 },
    { PN_TT_RENDER_NARROW_PROGRAM, "render-narrow", "render-narrow-member",
      PN_TT_CORE_RENDER_NARROW_ENV, PN_TT_CORE_RENDER_NARROW_DEFAULT,
      PN_TT_TILE_FP32, PN_TT_TILE_BF16 },
};

/**
 * THE DEFAULT MEMBER COUNT IS PER DIRECTION, AND IT IS MEASURED, NOT CHOSEN.
 *
 * Silicon 2026-08-26, on the WINDOW await, five arms on one binary in one hour, every one
 * `cmp` IDENTICAL:
 *
 *      direction        Wt    N=1      N=2      N=4       K = T(1) - 2*(T(1)-T(2))
 *      render-widen    256   125.4    105.1    152.7      84.6 us
 *      render-narrow   128    77.9     82.6    153.3      87.1 us
 *
 * The elementwise family's own break-even, measured the same day over five shapes, is
 * **87.2 +- 1.3 us** — and render's two directions recover 84.6 and 87.1 from completely
 * different programs at different page sizes. **The host's per-member cost does not care
 * about the page size**, which is the falsifier for this lever not firing. At four members
 * both sit on the host floor — 152.7 and 153.3 against 4 x 38.2 us — the same 38.1-38.4 us a
 * member the elementwise ladder measured flat at N = 4, 8 and 16.
 *
 * **BUT K IS A PROPERTY OF THE AWAIT, AND THE SHIPPED AWAIT IS THE `line`.** On the line the
 * same five shapes give **K = 44.3 +- 1.1 us** (render-widen 110.3 -> 77.1 recovers 43.9),
 * and BOTH directions are above it: 110.3 and 62.8. So **{2, 2}**. Deciding this on the
 * window numbers alone would have left the narrow direction unsplit for a bar that no longer
 * applies — which is the same mistake as reading the width knee off a family average.
 *
 * `PN_TT_RENDER_CORES` overrides BOTH, which is what a ladder wants; a per-direction knob
 * would make the two arms of a sweep disagree about what was being swept.
 */
static const uint32_t g_render_cores_default[PN_TT_RENDER_DIRECTIONS] = { 2u, 2u };

static void render_release_direction(pn_tt_device *device, uint32_t index);

/** Member `m` of direction `d`, member 0 being the direction's own residency. */
static pn_tt_residency *render_member(pn_tt_device *device, uint32_t index, uint32_t member)
{
    return member == 0u ? &device->render[index].residency
                        : &device->render[index].member[member - 1u];
}

/** The first core direction `index`'s members 1..N-1 run upward from. The two directions take
 *  disjoint blocks of `PN_TT_RENDER_SPLIT_MAX_MEMBERS - 1` cores so a ladder on one does not
 *  have to know what the other was set to. */
static uint32_t render_member_core_first(uint32_t index)
{
    return pn_tt_env_u32(PN_TT_RENDER_MEMBER_CORE_ENV, PN_TT_RENDER_MEMBER_CORE_DEFAULT) +
           index * (PN_TT_RENDER_SPLIT_MAX_MEMBERS - 1u);
}

/** Fill `request` with the placement one member of `direction` runs. Placement reads the
 *  depth and the three page sizes and nothing else, so the tile count carried here is the
 *  SLICE and every member places identically. */
static void render_placement(const render_direction *direction, uint32_t tiles, uint32_t depth,
                             ft_tt_op_eltwise_request *request)
{
    memset(request, 0, sizeof(*request));
    request->function = FT_TT_OP_VECTOR_FN_ADD;
    request->tile_count = tiles;
    request->prefetch_tiles = depth;
    request->input_a.tile_bytes = direction->in_tile_bytes;
    request->input_b.tile_bytes = direction->in_tile_bytes;
    request->output.tile_bytes = direction->out_tile_bytes;
}

/**
 * How many members this direction may run, and the ONE block depth every one of them takes.
 *
 * IT RUNS BEFORE ANYTHING IS PLACED, for the reason `pn_tt_eltwise_group_plan()` gives: the
 * placement is sized from the SLICED width, because the input CBs are two blocks of the
 * effective depth and the depth has to divide what a descriptor actually carries. Deciding
 * the member count afterwards places member 0 for a width no descriptor has.
 *
 * A member count whose slice admits no depth SHRINKS the split rather than refusing it —
 * unsplit is always correct and always available.
 *
 * @param out_depth  the depth every member is placed at; 0 when none serves even at one
 * @return           1..PN_TT_RENDER_SPLIT_MAX_MEMBERS
 */
static uint32_t render_plan(uint32_t index, uint32_t tiles, uint32_t *out_depth)
{
    const render_direction *direction = &g_render_directions[index];
    const uint32_t cap = pn_tt_env_u32(PN_TT_RENDER_DEPTH_ENV, PN_TT_ONCARD_MAX_DEPTH);
    uint32_t requested = pn_tt_env_u32(PN_TT_RENDER_CORES_ENV, g_render_cores_default[index]);
    uint32_t members;

    *out_depth = 0u;
    members = pn_tt_eltwise_split_member_limit(requested, &tiles, 1u,
                                               PN_TT_RENDER_SPLIT_MAX_MEMBERS);
    for (;;) {
        const uint32_t slice = tiles / members;
        uint32_t depth;

        /* THE DEEPEST BLOCK THAT BOTH DIVIDES THE RUN AND FITS THE ARENA, and the two are not
         * the same question. `pn_tt_common_depth()` answers the first; `place_cbs` answers the
         * second, and at 4096-byte pages it is the binding one — two input streams of
         * `2*depth` pages is `4 * depth * tile_bytes` of a 1.29 MiB arena, so the narrow
         * direction's ceiling is half the widen direction's. Walking down from the divisor
         * rather than refusing is what stops a page size from silently costing a residency. */
        for (depth = pn_tt_common_depth(&slice, 1u, cap); depth > 0u;
             depth = pn_tt_common_depth(&slice, 1u, depth - 1u)) {
            ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
            ft_tt_op_eltwise_request request;
            /* The depth search must not write into any residency's own CB buffer:
             * `prepare_family` has not run yet and it zeroes the whole handle. */
            unsigned char scratch_cb[FT_TT_NUM_CIRCULAR_BUFFERS *
                                     FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
            uint64_t scratch_mask;

            render_placement(direction, slice, depth, &request);
            if (ft_tt_op_eltwise_place_cbs(&request, placed, scratch_cb, sizeof(scratch_cb),
                                           &scratch_mask) == FT_TT_OK) {
                *out_depth = depth;
                if (requested > 1u && members != requested)
                    fprintf(stderr, "pn_tt: %s SPLIT — %u members was asked for; %u is the "
                                    "largest that divides %u tiles into slices a block depth "
                                    "serves\n", direction->name, requested, members, tiles);
                return members;
            }
        }
        if (members == 1u)
            return 1u;
        members = pn_tt_eltwise_split_member_limit(members - 1u, &tiles, 1u,
                                                   PN_TT_RENDER_SPLIT_MAX_MEMBERS);
    }
}

/**
 * Arm one direction for `tiles`, ZERO OPERAND INCLUDED.
 *
 * The zero buffer is `tiles` INPUT-format tiles and it is never written after the allocation
 * zeroes it — that is the whole of the "no kernel change" argument, and it is also why the
 * buffer must be allocated by something that zeroes: a B operand full of whatever the
 * allocator handed back would make a copy into a wrong answer with nothing refusing.
 */
static bool render_arm_direction(pn_tt_device *device, uint32_t index, uint32_t tiles)
{
    const render_direction *direction = &g_render_directions[index];
    pn_tt_render_residency *slot = &device->render[index];
    ft_tt_op_eltwise_request request;
    ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
    ft_tt_buffer zeros;
    uint32_t depth;
    uint32_t members;
    uint32_t member;
    uint32_t first;

    if (!pn_tt_program_is_present(direction->program)) {
        fprintf(stderr, "pn_tt: no `%s` program — that half of the activation strip bridge "
                        "stays on the host\n", direction->program);
        return false;
    }
    members = render_plan(index, tiles, &depth);
    if (depth == 0u) {
        fprintf(stderr, "pn_tt: no block depth both divides a %u-tile %s and fits the L1 "
                        "arena at %u B -> %u B pages\n", tiles, direction->name,
                direction->in_tile_bytes, direction->out_tile_bytes);
        return false;
    }
    if (pn_tt_dram_alloc_any(device, (size_t)tiles * direction->in_tile_bytes, &zeros) != 0) {
        fprintf(stderr, "pn_tt: DRAM full allocating the %s zero operand\n", direction->name);
        return false;
    }
    if (pn_tt_zero_dram(device, zeros.logical_bank, (uint32_t)zeros.address,
                        (size_t)tiles * direction->in_tile_bytes) != 0)
        return false;

    if (!pn_tt_prepare_family(device, &slot->residency, direction->program, direction->name,
                              direction->core_env, direction->core_default,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    render_placement(direction, tiles / members, depth, &request);
    if (ft_tt_op_eltwise_place_cbs(&request, placed, slot->residency.cb_buffer,
                                   sizeof(slot->residency.cb_buffer),
                                   &slot->residency.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the %s circular buffers do not fit at %u tiles, depth %u\n",
                direction->name, tiles / members, depth);
        pn_tt_residency_release(&slot->residency);
        return false;
    }
    if (!pn_tt_start_family(&slot->residency))
        return false;
    slot->members = 1u;
    memset(slot->member_seconds, 0, sizeof(slot->member_seconds));
    slot->in_tile_bytes = direction->in_tile_bytes;
    slot->out_tile_bytes = direction->out_tile_bytes;
    slot->tiles = tiles;
    slot->depth = depth;
    slot->zero_bank = zeros.logical_bank;
    slot->zero_address = (uint32_t)zeros.address;

    /* MEMBERS 1..N-1, EACH ON ITS OWN TENSIX, at member 0's identical placement.
     * A MEMBER THAT FAILS TO ARM COLLAPSES THE SPLIT TO ONE, loudly, rather than shrinking
     * it: the width must divide the member count exactly, and 8 shrinking to 5 is a ragged
     * slice the kernel refuses. Unsplit is always correct — member 0's placement serves the
     * whole width too, since `2 * depth` divides a slice and the slice divides the width. */
    first = render_member_core_first(index);
    for (member = 1u; member < members; member++) {
        pn_tt_residency *handle = &slot->member[member - 1u];

        if (!pn_tt_prepare_family_on_core(device, handle, direction->program,
                                          direction->member_name, PN_TT_RENDER_MEMBER_CORE_ENV,
                                          first + member - 1u, PN_TT_ONCARD_TRIAD_MASK))
            break;
        if (ft_tt_op_eltwise_place_cbs(&request, placed, handle->cb_buffer,
                                       sizeof(handle->cb_buffer), &handle->cb_mask) !=
                FT_TT_OK ||
            !pn_tt_start_family(handle)) {
            pn_tt_residency_release(handle);
            break;
        }
        slot->members++;
    }
    if (slot->members != members) {
        fprintf(stderr, "pn_tt: %s SPLIT — asked for %u members and %u armed; COLLAPSING to "
                        "one core, because a partial split is a width no member divides\n",
                direction->name, members, slot->members);
        for (member = 1u; member < slot->members; member++)
            pn_tt_residency_release(&slot->member[member - 1u]);
        slot->members = 1u;
    }
    /* THE TABLE EVERY ONE OF THESE CORES READS MUST BE A WRITTEN FACT. They run
     * `resident_vector_writer.c` and they arm on whichever side of
     * `pn_tt_publish_replica_map()` the model's setup order puts them, and that function does
     * not write to this family at all — so without this they read whatever is at the fixed L1
     * replica-table address, which is not "no mirroring" but "whatever a previous process
     * left there". That is exactly the state that cost the elementwise split its `cmp`.
     * Neither render output is a mirrored strip (`gdn_stream_in` is a GDN family buffer,
     * `gdn_y_bridge` feeds the elementwise gate, and only `normed` and `ffn_swiglu` are
     * replicated), so EMPTY is the CORRECT table here and not a stub. */
    for (member = 0u; member < slot->members; member++) {
        if (pn_tt_publish_empty_replica_map(device, render_member(device, index, member),
                                            direction->name) != 0) {
            fprintf(stderr, "pn_tt: %s could not be given an empty replica table — refusing "
                            "the residency rather than running it on an unknown one\n",
                    direction->name);
            render_release_direction(device, index);
            return false;
        }
    }
    if (slot->members > 1u)
        fprintf(stderr, "pn_tt: %s SPLIT — every descriptor over %u members, core %u then "
                        "%u..%u, %u tiles each at block depth %u\n",
                direction->name, slot->members, slot->residency.core_index, first,
                first + slot->members - 2u, tiles / slot->members, depth);
    fprintf(stderr, "pn_tt: %s is up on core %u — %u tiles, %u B -> %u B, depth %u\n",
            direction->name, slot->residency.core_index, tiles, slot->in_tile_bytes,
            slot->out_tile_bytes, depth);
    return true;
}

int pn_tt_device_arm_render(pn_tt_device *device, uint32_t widen_tiles, uint32_t narrow_tiles)
{
    if (!device)
        return -1;
    if (widen_tiles != 0u && !device->render[PN_TT_RENDER_WIDEN].residency.running)
        (void)render_arm_direction(device, PN_TT_RENDER_WIDEN, widen_tiles);
    if (narrow_tiles != 0u && !device->render[PN_TT_RENDER_NARROW].residency.running)
        (void)render_arm_direction(device, PN_TT_RENDER_NARROW, narrow_tiles);
    return 0;
}

bool pn_tt_render_widen_is_armed(const pn_tt_device *device)
{
    return device && device->render[PN_TT_RENDER_WIDEN].residency.running;
}

bool pn_tt_render_narrow_is_armed(const pn_tt_device *device)
{
    return device && device->render[PN_TT_RENDER_NARROW].residency.running;
}

/** Advance one operand's base to `slice`, in ITS OWN page size — the two sides of a render
 *  are at different page sizes and an offset built from the wrong one is an address with
 *  nothing on either end of the link able to refuse it. An UNSET operand is left alone. */
static void render_slice_operand(ft_tt_op_operand *operand, uint32_t first_tile)
{
    if (operand->tile_bytes == 0u)
        return;
    operand->address += first_tile * operand->tile_bytes;
}

/**
 * Retire ONE member and arm a fresh one ON THE CORE IT ALREADY HELD.
 *
 * The core is captured BEFORE the release, because re-reading `core_index` off a released
 * handle is how a member gets armed on core 0.
 */
static bool render_member_rearm(pn_tt_device *device, uint32_t index, uint32_t member,
                                const ft_tt_op_eltwise_request *placement)
{
    const render_direction *direction = &g_render_directions[index];
    ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
    pn_tt_residency *handle = render_member(device, index, member);
    const char *core_env = member == 0u ? direction->core_env : PN_TT_RENDER_MEMBER_CORE_ENV;
    const char *name = member == 0u ? direction->name : direction->member_name;
    uint32_t core = handle->core_index;

    fprintf(stderr, "pn_tt: %s member %u refused a descriptor — retiring it by ASKING and "
                    "arming a fresh one on core %u\n", direction->name, member, core);
    pn_tt_residency_stop(handle);
    pn_tt_residency_release(handle);
    device->oncard_restarts++;
    /* A core whose compute thread is still spinning out its bound is a WEDGE, not a re-arm:
     * booting firmware onto it leaves `instrn_buf_status = 0x66` and needs a rung-3 recovery
     * and sudo. Every other family here refuses on the same reading. */
    if (!pn_tt_residency_retired(handle)) {
        fprintf(stderr, "pn_tt: %s member %u did not retire — REFUSING to arm a fresh one on "
                        "core %u\n", direction->name, member, core);
        return false;
    }
    if (!pn_tt_prepare_family_on_core(device, handle, direction->program, name, core_env, core,
                                      PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_eltwise_place_cbs(placement, placed, handle->cb_buffer,
                                   sizeof(handle->cb_buffer), &handle->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(handle);
        return false;
    }
    if (!pn_tt_start_family(handle))
        return false;
    /* Same argument as the arm: a fresh residency's replica table is whatever that L1 region
     * held, and empty is a written fact. */
    return pn_tt_publish_empty_replica_map(device, handle, direction->name) == 0;
}

/**
 * Run ONE render descriptor across the whole set: ALL ENQUEUED BEFORE ANY IS AWAITED.
 *
 * The two loops being separate is the measurement and not a style. Enqueue-wait per member
 * would run `members` SERIAL descriptors and print the flat line "it does not parallelise",
 * produced by this function rather than by the card.
 *
 * Time is charged to the SET, once, at the WHOLE descriptor's tile count, so the census key
 * `(render-widen, add, tiles)` is the same row at every member count and the ladder is
 * directly comparable.
 */
static int render_submit_split(pn_tt_device *device, uint32_t index,
                               const ft_tt_op_eltwise_request *whole, uint32_t members)
{
    uint32_t words[PN_TT_RENDER_SPLIT_MAX_MEMBERS][FT_TT_OP_VECTOR_WORDS];
    pn_tt_residency *set[PN_TT_RENDER_SPLIT_MAX_MEMBERS];
    bool bad[PN_TT_RENDER_SPLIT_MAX_MEMBERS];
    pn_tt_render_residency *slot = &device->render[index];
    double enqueue_started;
    uint32_t member;
    bool pulled;
    int failed = 0;

    memset(bad, 0, sizeof(bad));
    for (member = 0u; member < members; member++) {
        ft_tt_op_eltwise_request sliced = *whole;
        pn_tt_eltwise_slice slice;

        if (!pn_tt_eltwise_split_slice(whole->tile_count, members, member,
                                       slot->in_tile_bytes, &slice)) {
            fprintf(stderr, "pn_tt: %u %s tiles do not divide over %u members\n",
                    whole->tile_count, g_render_directions[index].name, members);
            return -1;
        }
        sliced.tile_count = slice.tile_count;
        sliced.prefetch_tiles = slot->depth;
        render_slice_operand(&sliced.input_a, slice.first_tile);
        render_slice_operand(&sliced.output, slice.first_tile);
        /* THE ZERO OPERAND IS NOT OFFSET, AND THAT IS DELIBERATE. Every tile of it is the
         * same tile — zeros, staged by the allocator and never written — so every member
         * reading from the base reads exactly what an offset would have given it, and the
         * read is provably inside the allocation even when a caller submits fewer tiles than
         * the direction was armed for. Offsetting it would be arithmetic with no
         * corresponding fact. */
        if (ft_tt_op_eltwise_parameters(&sliced, words[member]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the %s slice [%u, %u) is one the family refuses\n",
                    g_render_directions[index].name, slice.first_tile,
                    slice.first_tile + slice.tile_count);
            return -1;
        }
        pn_tt_vec_probe_apply(words[member]);
    }
    for (member = 0u; member < members; member++)
        set[member] = render_member(device, index, member);
    /* S1: member 0 posts only once every member has (pn_tt_group_pull.h); one await. */
    pulled = pn_tt_group_pull_decide(&slot->pull, pn_tt_group_pull_decode(device), set, members,
                                     members, g_render_directions[index].name);
    enqueue_started = pn_tt_now_seconds();
    (void)pn_tt_group_pull_enqueue(&slot->pull, pulled, set, members, FT_TT_QUEUE_OP_ELTWISE,
                                   words[0], bad);
    /* THE RESIDENCY CLOCKS RUN ON THIS PATH TOO (class 28). Enqueue on member 0, each serial
     * await on its member, so the rows SUM to the split's wall. */
    set[0]->seconds += pn_tt_now_seconds() - enqueue_started;
    failed = pn_tt_group_pull_await(&slot->pull, pulled, set, members, bad,
                                    slot->member_seconds) != 0;
    /* PER-MEMBER RE-ARM AND REPLAY. Replaying one member's slice is the SAME COMPUTATION and
     * not merely a safe one: `a + 0` is a pure function of operands it does not modify and
     * the slices are disjoint, so a member that succeeded is neither re-run nor read. */
    if (failed) {
        ft_tt_op_eltwise_request placement;

        render_placement(&g_render_directions[index], whole->tile_count / members, slot->depth,
                         &placement);
        failed = 0;
        for (member = 0u; member < members; member++) {
            if (!bad[member])
                continue;
            if (!render_member_rearm(device, index, member, &placement) ||
                pn_tt_residency_submit(render_member(device, index, member),
                                       FT_TT_QUEUE_OP_ELTWISE, words[member],
                                       PN_TT_OP_TIMEOUT_MS) != 0)
                failed = 1;
        }
    }
    return failed ? -1 : 0;
}

/** One render descriptor, in whichever direction the caller's operands name. */
static int render_submit(pn_tt_device *device, uint32_t index, ft_tt_op_operand in,
                         ft_tt_op_operand out, uint32_t tiles)
{
    pn_tt_render_residency *slot;
    ft_tt_op_eltwise_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t members;
    double started;

    if (!device)
        return -1;
    slot = &device->render[index];
    if (!slot->residency.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. A render
     * whose input a queued matvec is still writing reads the PREVIOUS token's activation and
     * nothing anywhere refuses it. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    /* THE PAGE SIZE IS A THREE-SIDED CONTRACT WITH NO CARD-SIDE CHECK: the DRAM operand, the
     * L1 circular buffer and the data-movement kernel's compile-time TILE_BYTES all have to
     * agree, and the device can see none of it. Measured on the reduce family — constants
     * built at 2048 under a 4096 program score 1.0e+00 with nothing refusing anywhere. */
    if (in.tile_bytes != slot->in_tile_bytes || out.tile_bytes != slot->out_tile_bytes) {
        fprintf(stderr, "pn_tt: a %u B -> %u B render was handed %u B -> %u B operands\n",
                slot->in_tile_bytes, slot->out_tile_bytes, in.tile_bytes, out.tile_bytes);
        return -1;
    }
    if (tiles == 0u || tiles > slot->tiles) {
        fprintf(stderr, "pn_tt: %u tiles is not a strip the %s residency was placed for "
                        "(%u)\n", tiles, g_render_directions[index].name, slot->tiles);
        return -1;
    }
    /* THE DEPTH INVARIANT IS PER DESCRIPTOR AND THE DESCRIPTOR A MEMBER RUNS IS THE SLICE.
     * A width the set does not divide, or one whose slice is off the `2 * depth` grid, RUNS
     * WHOLE ON MEMBER 0 rather than failing: member 0's placement serves it (the slice
     * divides the width and `2 * depth` divides the slice), so one core is slower rather
     * than wrong. Failing instead would turn a new call site into a dropped descriptor,
     * which on this family is a wrong answer with a return code nobody reads. Said ONCE,
     * because 48 descriptors a token is a flood. */
    members = slot->members < 2u ? 1u : slot->members;
    if (members > 1u && ((tiles % members) != 0u ||
                         ((tiles / members) % (2u * slot->depth)) != 0u ||
                         ft_tt_op_block_depth(tiles / members, slot->depth) != slot->depth)) {
        static bool narrowed;

        if (!narrowed) {
            narrowed = true;
            fprintf(stderr, "pn_tt: %u %s tiles do not slice over %u members at depth %u — "
                            "this width runs WHOLE on member 0\n", tiles,
                    g_render_directions[index].name, members, slot->depth);
        }
        members = 1u;
    }
    if (ft_tt_op_block_depth(tiles / members, slot->depth) != slot->depth ||
        ((tiles / members) % (2u * slot->depth)) != 0u) {
        fprintf(stderr, "pn_tt: %u tiles is not a whole number of depth-%u blocks the render "
                        "residency was placed for (%u)\n", tiles, slot->depth, slot->tiles);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    /* `a + 0`. See pn_tt_render.h: there is no identity in the family and adding one would
     * make every eltwise-family program stale by content provenance. */
    request.function = FT_TT_OP_VECTOR_FN_ADD;
    request.tile_count = tiles;
    request.prefetch_tiles = slot->depth;
    request.input_a = in;
    request.input_b.x = (uint16_t)device->dram[slot->zero_bank].x;
    request.input_b.y = (uint16_t)device->dram[slot->zero_bank].y;
    request.input_b.address = slot->zero_address;
    request.input_b.tile_bytes = slot->in_tile_bytes;
    request.output = out;

    if (members > 1u) {
        int rc;

        started = pn_tt_now_seconds();
        rc = render_submit_split(device, index, &request, members);
        pn_tt_oncard_account(device, g_render_directions[index].name, "add", tiles,
                             pn_tt_now_seconds() - started);
        return rc;
    }
    if (ft_tt_op_eltwise_parameters(&request, words) != FT_TT_OK)
        return -1;
    pn_tt_vec_probe_apply(words);

    started = pn_tt_now_seconds();
    if (pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_ELTWISE, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* The words address DRAM only, so the same descriptor is valid for a fresh residency
         * — the re-arm argument every other family here makes. */
        fprintf(stderr, "pn_tt: the %s residency refused a descriptor — retiring it by ASKING "
                        "and arming a fresh one\n", g_render_directions[index].name);
        pn_tt_residency_stop(&slot->residency);
        pn_tt_residency_release(&slot->residency);
        device->oncard_restarts++;
        if (!pn_tt_prepare_family(device, &slot->residency, g_render_directions[index].program,
                                  g_render_directions[index].name,
                                  g_render_directions[index].core_env,
                                  g_render_directions[index].core_default,
                                  PN_TT_ONCARD_TRIAD_MASK) ||
            !pn_tt_start_family(&slot->residency) ||
            pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_ELTWISE, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    pn_tt_oncard_account(device, g_render_directions[index].name, "add", tiles,
                         pn_tt_now_seconds() - started);
    return 0;
}

int pn_tt_render_widen(pn_tt_device *device, ft_tt_op_operand in, ft_tt_op_operand out,
                       uint32_t tiles)
{
    return render_submit(device, PN_TT_RENDER_WIDEN, in, out, tiles);
}

int pn_tt_render_narrow(pn_tt_device *device, ft_tt_op_operand in, ft_tt_op_operand out,
                        uint32_t tiles)
{
    return render_submit(device, PN_TT_RENDER_NARROW, in, out, tiles);
}

/** Retire one direction WHOLE — member 0 and every member. A LEAKED TRIAD IS A WEDGED TENSIX
 *  for the next process to ask for that core, which is what the head-norm slots cost when
 *  `pn_tt_device_close()` released named fields and missed a set. */
static void render_release_direction(pn_tt_device *device, uint32_t index)
{
    pn_tt_render_residency *slot = &device->render[index];
    uint32_t member;

    for (member = slot->members; member > 0u; member--) {
        pn_tt_residency *handle = render_member(device, index, member - 1u);

        if (handle->running) {
            pn_tt_residency_stop(handle);
            pn_tt_residency_release(handle);
        }
    }
    if (slot->residency.running) {
        pn_tt_residency_stop(&slot->residency);
        pn_tt_residency_release(&slot->residency);
    }
    memset(slot, 0, sizeof(*slot));
}

void pn_tt_render_release(pn_tt_device *device)
{
    uint32_t index;

    if (!device)
        return;
    for (index = 0; index < (uint32_t)PN_TT_RENDER_DIRECTIONS; index++)
        render_release_direction(device, index);
}

uint32_t pn_tt_render_members(const pn_tt_device *device, uint32_t index)
{
    if (!device || index >= (uint32_t)PN_TT_RENDER_DIRECTIONS)
        return 0u;
    /* `members` is set by the arm, so a direction that never armed reads 0 and one that armed
     * unsplit reads 1 — the caller has to be able to tell those two apart. */
    return device->render[index].members;
}

bool pn_tt_render_member_stat(const pn_tt_device *device, uint32_t index, uint32_t member,
                              const char **out_name, uint32_t *out_core, uint64_t *out_ops,
                              double *out_seconds)
{
    const pn_tt_residency *handle;

    if (!device || index >= (uint32_t)PN_TT_RENDER_DIRECTIONS ||
        member >= pn_tt_render_members(device, index))
        return false;
    handle = member == 0u ? &device->render[index].residency
                          : &device->render[index].member[member - 1u];
    if (out_name)
        *out_name = g_render_directions[index].name;
    if (out_core)
        *out_core = handle->core_index;
    if (out_ops)
        *out_ops = handle->ops;
    if (out_seconds)
        *out_seconds = device->render[index].member_seconds[member];
    return true;
}

void pn_tt_render_defer(pn_tt_device *device, bool narrow, bool on)
{
    pn_tt_render_residency *slot;

    if (!device)
        return;
    slot = &device->render[narrow ? PN_TT_RENDER_NARROW : PN_TT_RENDER_WIDEN];
    slot->residency.defer_await = on;
    for (uint32_t member = 1u; member < slot->members; member++)
        slot->member[member - 1u].defer_await = on;
}

int pn_tt_render_settle(pn_tt_device *device, bool narrow)
{
    pn_tt_residency *set[PN_TT_RENDER_SPLIT_MAX_MEMBERS];
    pn_tt_render_residency *slot;
    uint32_t members;

    if (!device)
        return -1;
    slot = &device->render[narrow ? PN_TT_RENDER_NARROW : PN_TT_RENDER_WIDEN];
    members = slot->members < 1u ? 1u : slot->members;
    for (uint32_t member = 0u; member < members; member++)
        set[member] = render_member(device, narrow ? PN_TT_RENDER_NARROW : PN_TT_RENDER_WIDEN,
                                    member);
    /* A member a pulled split covered is settled by member 0's completion (S1). */
    return pn_tt_group_pull_settle(&slot->pull, set, members, PN_TT_OP_TIMEOUT_MS);
}

#endif /* USE_TT */
