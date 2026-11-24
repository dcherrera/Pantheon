/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn.c
 * @brief The host stager for the GATED-DELTA-NET op: DRAM allocation, the layouts
 *        `foundry/tt/kernels/resident_gdn_protocol.h` settles, and two descriptors a layer.
 *        The API and the cost model are in pn_tt_gdn.h.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FILE INCLUDES A KERNEL HEADER
 * ---------------------------------------------------------------------------
 * The stager and the two dataflow kernels are the two ends of a contract that NOTHING checks
 * at run time: a stager that disagrees about where a plane starts produces plausible numbers
 * from the first token instead of an error. So every offset below comes from the `ftg_*`
 * accessors in `resident_gdn_protocol.h` — the same functions the reader and the writer
 * compute their addresses with — rather than from arithmetic respelled here. The header
 * guards its one SFPI attribute so a host compiler can read it, which is the sanctioned
 * escape `foundry/tt/tests/test_gdn_ops.c` already uses.
 *
 * ---------------------------------------------------------------------------
 * THE FOUR LAYOUTS THIS FILE IS RESPONSIBLE FOR GETTING RIGHT
 * ---------------------------------------------------------------------------
 *  1. **The conv WEIGHT is a TRANSPOSE.** The oracle stores `w[c*taps + t]`; the card wants
 *     `taps` planes, plane `t` holding `w[.][t]` laid out like the stream. A reshape is the
 *     plausible wrong answer and it is silent. `PN_TT_RED=gdn_conv_reshape` is that mutation,
 *     kept as a knob so the on-card gate can be shown to catch it.
 *  2. **The conv stream is a ONE-ROW tilized strip.** `qkv[c]` lands at row 0, column `c%32`
 *     of tile `c/32`, so the weight plane and the window plane put channel `c` at the same
 *     place and the compute thread's whole-tile multiply lines them up. Rows 1-31 are the
 *     tile quantum, and `SiLU(0) = 0` keeps them zero through the convolution — which is what
 *     lets this write only row 0 per token, 128 B a tile instead of 4096.
 *  3. **`q` and `k` are per K-HEAD and NOT repeated.** They are tilized `[k_heads, S]`
 *     matrices; the reader's `vh % k_heads` is the whole of the GQA repeat. Materialising it
 *     doubles them.
 *  4. **The gate block is not tiles.** `decay[v_heads]` then `beta[v_heads]`, flat fp32, in an
 *     allocation rounded up to the 64 B read granule the reader rounds its read up to.
 *
 * @status EXECUTED ON SILICON 2026-08-18. Every layout above is now checked BY READBACK as
 *         well as off hardware: `PN_TT_GDN_SELFCHECK` echoes the staged qkv row and all four
 *         conv weight tap planes out of card DRAM and both come back with 0 of 8192 channels
 *         differing, so the transpose, the one-row strip and the row-0 offsets are confirmed
 *         rather than inferred.
 * @issues Two defects were found here on the first silicon run and BOTH were invisible to
 *         every off-hardware check, because both produce a descriptor that retires cleanly:
 *
 *         1. `kGdnActivationXy` is ONE packed coordinate for the activation IN and OUT, and
 *            this file round-robined DRAM banks, so the writer put `conv_out` and `y` at the
 *            output's offset inside the INPUT's bank. Fixed — the outputs are pinned to their
 *            inputs' bank and `ft_tt_op_gdn_parameters()` refuses a mismatch.
 *         2. The core default was 75, which `PN_TT_MATMUL_CORES=4` hands to matmul member 1.
 *            1560 mutual re-arms over 65 forwards. Fixed — 90, plus a refusal by name in
 *            `pn_tt_prepare_family()`.
 *
 *         A third was found on 2026-08-18 and it was NOT in this file: the family's circular
 *         buffers were sized for capacity and for the PRODUCER's push grid but never for the
 *         CONSUMER's indexed read, so CB_VECTOR (8 pages, 7 consumed a head) and CB_STATE
 *         (8 pages, 3 indexed a CONV block) both walked off the end. See
 *         `foundry/tt/kernels/resident_gdn_protocol.h`.
 *
 *         A FOURTH was found on 2026-08-18 BY THE PROBES IN THIS FILE and it was not in this
 *         file either: the rolled conv window's NEWEST plane was wrong on every CONV
 *         descriptor in a run EXCEPT the first. It read as "wrong from token 1", and the
 *         reframing is the whole diagnosis — SSM layer 0 of position 0 is the only CONV never
 *         preceded by a PHASE DELTA. `PN_TT_GDN_RAMP` stages an INJECTIVE stream
 *         (`qkv[c] = c`) so the plane names its own source, and the map read `s = t (mod 8)`
 *         for source tile `s` of output tile `t` at all 256 tiles: a free-running consumer on
 *         an 8-page buffer. CB_IN was pushed by the READER in PHASE CONV and by the COMPUTE
 *         thread in PHASE DELTA, and the two push primitives keep incompatible books for one
 *         stream register — see `foundry/tt/kernels/resident_gdn_protocol.h`. PHASE DELTA has
 *         its own CB now and the newest plane is byte-exact at every layer of every token.
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdn.h"
#include "tt/ft_tt_launch.h"
#include "pn_tt_dispatch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"

#include "pn_tt_fp32_strip.h"
#include "pn_tt_gdn_conv_share.h"
#include "pn_tt_gdn_conv_split.h"
#include "pn_tt_gdn_core_scaling.h"
#include "pn_tt_gdn_verify_slots.h"
#include "pn_tt_gdn_layout.h"
#include "pn_tt_group_pull.h"

#include "tt/ft_tt_op_gdn.h"

/* See the file header: the L1 pointer attribute is the SFPI fork's, not the contract's. */
#define FTQ_L1_PTR
#include "tt/kernels/resident_gdn_protocol.h"

/** Which enabled Tensix it drives. A default, not a constant — and it must be a core neither
 *  the other six residencies (reduce 72, eltwise 73, topk 74, rope 82, memory 83, attention
 *  84) NOR THE MATMUL GROUP holds.
 *
 *  **75 was wrong and the card said so.** `PN_TT_MATMUL_CORES=4` is the shipped knee and its
 *  members are 71, 75, 76, 77, so the paper check against the other residencies missed the
 *  one set that MOVES with a configuration knob. Measured 2026-08-18: 1560 mutual re-arms
 *  over 65 forwards, the GDN and the matmul residency retiring each other every token, both
 *  logging a healthy "residency up" each time. 90 clears the group out past
 *  `PN_TT_MATMUL_CORES=16`, and `pn_tt_prepare_family()` now refuses a collision by name. */
#define PN_TT_CORE_GDN_ENV     "PN_TT_CORE_GDN"
#define PN_TT_CORE_GDN_DEFAULT 90u

/** THE HEAD SPLIT. `PN_TT_GDN_HEAD_CORES=N` arms N-1 EXTRA GDN residencies and gives each
 *  member `v_heads/N` of the 32 v-heads; PHASE CONV splits by tile column over as many of
 *  them as its slices fit (`gdn_conv_members()`).
 *
 *  **DEFAULT 4, MEASURED ON THE SHIPPED BLOB** — 2026-08-19, `PN_TT_PROGRAM_DIR` UNSET, so
 *  the embedded program blob a serve runs rather than an arm tree in /tmp. PHASE DELTA
 *  **1835.0 -> 679.3 us, 2.70x**; the GDN block **0.0734 -> 0.0456 s/token**; taken twice
 *  to the digit, `cmp` IDENTICAL against the 48-token oracle on every arm, `company=0`.
 *  The gate this default used to carry was that the 7.12x justifying it came from a PROBE
 *  which replicated a whole descriptor rather than splitting one. The split has now been
 *  measured AS a split, on the artifact, and it is worth 2.70x — which is the number this
 *  default is set from.
 *
 *  **4 IS THE KNEE AND 8 IS WORSE**: 1.83 / 2.70 / 2.65x at N = 2 / 4 / 8, N=8
 *  reproducibly behind N=4 (692.0 us against 679.3). Phase 1's knee, which is why this is
 *  a number and not "as many as fit".
 *
 *  **IT COSTS THE SHIPPED FORWARD NOTHING, BECAUSE THE GDN FAMILY IS OPT-IN.** An unset
 *  `PN_TT_ONCARD` does not arm gdn (`pn_tt_read_oncard_selection()` makes gdn and attention
 *  opt-in-only), so none of this runs and no extra Tensix is claimed unless a run names
 *  `gdn`. Measured beside the flip: the shipped wall is 0.5116 s/token either way.
 *  `PN_TT_GDN_HEAD_CORES=1` is still the way back to a single residency.
 *
 *  92 is where the members start, the same first core the core-scaling probe used —
 *  `pn_tt_prepare_family_on_core()` still refuses a matmul group member BY NAME, and the
 *  group MOVES with `PN_TT_MATMUL_CORES`, so the refusal is the guard and this default is
 *  only the first place to look. 90 is member 0 and 91 is the head norm's. */
#define PN_TT_GDN_HEAD_CORES_ENV        "PN_TT_GDN_HEAD_CORES"
/* EIGHT on 133..139 since 2026-10-04 (was 4 on 92..94): CONV 309 -> 267 us, DELTA 434 -> 242 us,
 * decode 10.69 -> 11.34 tok/s, IDENTICAL — once the reader stopped refusing a CONV after an
 * 8-member DELTA. 133..139 is above every other family's default member block. */
#define PN_TT_GDN_HEAD_CORES_DEFAULT    16u /* CONV over 16; DELTA stays 8 (run_d_ceilings.sh) */
#define PN_TT_GDN_MEMBER_CORE_FIRST_ENV "PN_TT_GDN_MEMBER_CORE_FIRST"
#define PN_TT_GDN_MEMBER_CORE_DEFAULT   133u

/** Elements of one tile row, and bytes of one fp32 tile. Both are the family's, not this
 *  file's: every circular buffer of the GDN program is Float32, and pn_tt_gdn_layout.h holds
 *  the same two numbers for the arithmetic that was split out of here to be provable. */
#define PN_TT_GDN_TILE_DIM   FT_TT_OP_TILE_DIM
#define PN_TT_GDN_TILE_BYTES FT_TT_OP_GDN_FP32_TILE_BYTES

/** The v-head ceiling this family admits — `ft_tt_op_gdn_shape_is_supported()` refuses more,
 *  because the count is a BYTE of the shape word. The gate block is the only operand small
 *  enough to stage on the stack, and this is what bounds it. */
#define PN_TT_GDN_MAX_V_HEADS 255u

/* ================================================================
 * Shape, operands and allocation
 * ================================================================ */

/** The device's own view of this residency's shape, for the `ftg_*` accessors. */
static ftg_shape gdn_shape(const pn_tt_device *device, uint32_t phase)
{
    ftg_shape shape;

    shape.phase = phase;
    shape.state_tiles = device->gdn_shape.state_size / PN_TT_GDN_TILE_DIM;
    shape.v_heads = device->gdn_shape.v_heads;
    shape.k_heads = device->gdn_shape.k_heads;
    shape.conv_tile_cols = device->gdn_shape.conv_dim / PN_TT_GDN_TILE_DIM;
    shape.conv_taps = device->gdn_shape.conv_taps;
    shape.tile_bytes = PN_TT_GDN_TILE_BYTES;
    shape.tile_count = 0u;
    return shape;
}

/** The request describing this residency, with the operands a caller then fills in. Carries
 *  the CONV taps whatever the phase, because ONE CB placement serves both phases and
 *  `ft_tt_op_gdn_place_cbs()` sizes the weight buffers from them. */
static ft_tt_op_gdn_request gdn_request(const pn_tt_device *device, uint32_t phase)
{
    ft_tt_op_gdn_request request;

    memset(&request, 0, sizeof(request));
    request.phase = phase;
    request.state_size = device->gdn_shape.state_size;
    request.v_heads = device->gdn_shape.v_heads;
    request.k_heads = device->gdn_shape.k_heads;
    request.conv_dim = device->gdn_shape.conv_dim;
    request.conv_taps = device->gdn_shape.conv_taps;
    return request;
}

/** An fp32 operand descriptor for one of this family's DRAM buffers. */
static ft_tt_op_operand gdn_operand(const pn_tt_device *device, const pn_tt_gdn_buffer *buffer)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[buffer->bank].x;
    operand.y = (uint16_t)device->dram[buffer->bank].y;
    operand.address = buffer->address;
    operand.tile_bytes = PN_TT_GDN_TILE_BYTES;
    return operand;
}

/** "Any bank will do" — the round-robin every buffer takes unless it is PINNED to another
 *  buffer's endpoint. */
#define PN_TT_GDN_ANY_BANK UINT32_MAX

/**
 * Carve one DRAM buffer in `bank`, and when asked stage it ZEROED.
 *
 * The zeroing is not tidiness for three of the callers. The conv window and the recurrent
 * state ARE the recurrence's initial condition and the oracle's are calloc'd; the qkv stream
 * is written row 0 at a time forever after, so rows 1-31 have to start zero and stay zero.
 *
 * **THE BANK IS AN ARGUMENT BECAUSE TWO PAIRS OF THESE BUFFERS MUST SHARE ONE.** The
 * descriptor has a single packed x/y word for the activation IN and the activation OUT
 * (`kGdnActivationXy`), so the two are one NoC endpoint by contract, and a round-robin that
 * split them made the writer put every answer at the output's offset inside the input's bank.
 * `ft_tt_op_gdn_parameters()` now refuses that descriptor; this is the side that stops
 * building it.
 */
static int gdn_alloc_in(pn_tt_device *device, uint32_t bank, size_t bytes, bool zero,
                        pn_tt_gdn_buffer *out)
{
    ft_tt_buffer buffer;

    if (bytes == 0u)
        return -1;
    if (bank == PN_TT_GDN_ANY_BANK ? pn_tt_dram_alloc_any(device, bytes, &buffer) != 0
                                   : ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM,
                                                        bank, bytes, &buffer) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full allocating a %zu-byte GDN buffer\n", bytes);
        return -1;
    }
    out->bank = buffer.logical_bank;
    out->address = (uint32_t)buffer.address;
    if (zero && pn_tt_zero_dram(device, out->bank, out->address, bytes) != 0)
        return -1;
    return 0;
}

/** Grow the host tile-staging buffer to `bytes`. Kept on the device so a per-layer call is
 *  not a malloc on a two-core host. */
static int gdn_stage_grow(pn_tt_device *device, size_t bytes)
{
    if (bytes > device->gdn_stage_bytes) {
        unsigned char *grown = realloc(device->gdn_stage, bytes);

        if (!grown)
            return -1;
        device->gdn_stage = grown;
        device->gdn_stage_bytes = bytes;
    }
    return 0;
}

/* ================================================================
 * Logical row 0 of a strip of fp32 tiles
 * ================================================================ */

/** Move row 0 of `tiles` consecutive fp32 tiles, in either direction. The arithmetic and the
 *  transfer are `pn_tt_fp32_strip.c`'s — the same primitive the per-head reduce residency's
 *  Float32 strips use, and there is no reason for this family to own a second copy. */
static int gdn_row0_move(pn_tt_device *device, const pn_tt_gdn_buffer *buffer, uint32_t tiles,
                         float *values, bool reading)
{
    return pn_tt_fp32_row0_move(device, buffer->bank, buffer->address, tiles, values, reading);
}

/* ================================================================
 * Arming
 * ================================================================ */

/** Every per-layer buffer, and the shared per-token staging set. */
static int gdn_alloc_regions(pn_tt_device *device, const ft_tt_op_gdn_request *request)
{
    const size_t state = ft_tt_op_gdn_state_bytes(request);
    const size_t window = ft_tt_op_gdn_conv_state_bytes(request);
    const size_t weight = ft_tt_op_gdn_conv_weight_bytes(request);
    const size_t stream = ft_tt_op_gdn_conv_stream_bytes(request);
    uint32_t layer;

    device->gdn_region = calloc(device->gdn_shape.layer_count, sizeof(*device->gdn_region));
    if (!device->gdn_region)
        return -1;
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++) {
        pn_tt_gdn_region *region = &device->gdn_region[layer];

        /* The weight is a CONSTANT the caller writes whole, plane by plane, so it needs no
         * staging; the two pieces of carried state are the recurrence's initial condition.
         * SHARED (pn_tt_gdn_conv_share.h): the weight is not allocated, but its round-robin
         * turn is still taken, so the window and state keep main's banks. */
        if (device->gdn_conv_share == PN_TT_GDN_CONV_PENDING)
            device->strip_bank_cursor++;
        else if (gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, weight, false, &region->weight) != 0)
            return -1;
        if (gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, window, true, &region->window) != 0 ||
            gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, state, true, &region->state) != 0)
            return -1;
    }
    /* ONE set of staging buffers for the whole model: a token runs one layer at a time and
     * nothing in them carries across a descriptor. Every one of them is staged ZEROED, because
     * every one is written ROW 0 AT A TIME forever after and the strip invariant is that rows
     * 1-31 start zero and stay zero.
     *
     * **THERE ARE THREE OF THEM, NOT FOUR, AND THAT IS THE STRIP BRIDGE.** `stream_out` is
     * CONV's output AND DELTA's input: both are `conv_dim/32` row-0 tiles and conv_out's q, k
     * and v slices land exactly where `ftg_delta_stream_base()` reads them, so the delta rule
     * takes the convolution's answer in place and the separate `delta_in` staging buffer is
     * gone. What used to sit between them — de-tilize on the host, L2-normalise, re-tilize,
     * upload — is 49,408 B a layer of PCIe traffic that no longer exists.
     *
     * **EVERY OUTPUT IS PINNED TO ITS INPUT'S BANK.** `conv_out` shares CONV's
     * `kGdnActivationXy` with the qkv stream and `y` shares DELTA's with `stream_out` — ONE
     * packed coordinate word for two buffers — so a round-robin that split a pair sent the
     * writer to the input's endpoint with the output's offset. It completed, wrote over
     * whatever was at that offset in the input's bank, and left the output buffer exactly as
     * allocated: `conv_out` unzeroed garbage and `y` zeros. All four now share `stream_in`'s
     * bank, which is what makes both pairs satisfy the contract at once. */
    if (gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, stream, true, &device->gdn_stream_in) != 0 ||
        gdn_alloc_in(device, device->gdn_stream_in.bank, stream, true,
                     &device->gdn_stream_out) != 0 ||
        gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, ft_tt_op_gdn_gate_bytes(request), true,
                     &device->gdn_gate) != 0 ||
        gdn_alloc_in(device, device->gdn_stream_in.bank,
                     ft_tt_op_gdn_delta_output_bytes(request), true, &device->gdn_y) != 0)
        return -1;

    /* THE MAP, so an overlap is readable instead of inferred. A buffer this family writes
     * that lands inside another one is invisible from either end — the writer retires clean
     * and the reader gets a plausible tensor — and the only place the two addresses are ever
     * both in view is here. Under SELFCHECK only: 77 lines a run is a probe, not a log. */
    if (getenv("PN_TT_GDN_SELFCHECK")) {
        for (layer = 0; layer < device->gdn_shape.layer_count; layer++) {
            const pn_tt_gdn_region *region = &device->gdn_region[layer];

            fprintf(stderr, "pn_tt: gdn MAP layer %2u  weight bank %u @ %08x +%zu  window "
                            "bank %u @ %08x +%zu  state bank %u @ %08x +%zu\n",
                    layer, region->weight.bank, region->weight.address, weight,
                    region->window.bank, region->window.address, window, region->state.bank,
                    region->state.address, state);
        }
        fprintf(stderr, "pn_tt: gdn MAP stream_in bank %u @ %08x +%zu  stream_out/delta_in "
                        "bank %u @ %08x +%zu  gate bank %u @ %08x  y bank %u @ %08x\n",
                device->gdn_stream_in.bank, device->gdn_stream_in.address, stream,
                device->gdn_stream_out.bank, device->gdn_stream_out.address, stream,
                device->gdn_gate.bank, device->gdn_gate.address, device->gdn_y.bank,
                device->gdn_y.address);
    }
    {
        const size_t own_weight = device->gdn_conv_share == PN_TT_GDN_CONV_PENDING ? 0u : weight;

        fprintf(stderr, "pn_tt: gdn — %u layers x (%.2f MiB state + %.2f MiB window + %.2f MiB "
                        "weight) = %.2f MiB of card DRAM\n",
                device->gdn_shape.layer_count, (double)state / (1024.0 * 1024.0),
                (double)window / (1024.0 * 1024.0), (double)own_weight / (1024.0 * 1024.0),
                (double)((state + window + own_weight) * device->gdn_shape.layer_count) /
                    (1024.0 * 1024.0));
    }
    return 0;
}

int pn_tt_gdn_alloc_conv_weight(pn_tt_device *device, pn_tt_gdn_buffer *out)
{
    const ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);

    return gdn_alloc_in(device, PN_TT_GDN_ANY_BANK, ft_tt_op_gdn_conv_weight_bytes(&request),
                        false, out);
}

int pn_tt_gdn_place_regions(pn_tt_device *device)
{
    ft_tt_op_gdn_request request;

    if (!device || device->gdn_shape.layer_count == 0u)
        return -1;
    request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);
    request.activation_in.tile_bytes = PN_TT_GDN_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDN_TILE_BYTES;
    return gdn_alloc_regions(device, &request);
}

/** A positive env integer, or `fallback`. */
static uint32_t gdn_env_u32(const char *name, uint32_t fallback)
{
    const char *value = getenv(name);
    long parsed;

    if (!value || !*value)
        return fallback;
    parsed = strtol(value, NULL, 10);
    return parsed > 0 ? (uint32_t)parsed : fallback;
}

/**
 * How many GDN residencies this run will split PHASE DELTA across.
 *
 * Clamped to the array, to `PN_TT_GDN_MAX_MEMBERS`, and to a DIVISOR of `v_heads` — an
 * uneven split would give one member a different trip count, and a member is a whole
 * residency whose descriptor cost is `59.8 us + heads x 103.4`: the set is only as fast as
 * its longest member, so an uneven split pays for the biggest slice and wastes the rest.
 */
static uint32_t gdn_head_member_count(const pn_tt_device *device)
{
    uint32_t members = gdn_env_u32(PN_TT_GDN_HEAD_CORES_ENV, PN_TT_GDN_HEAD_CORES_DEFAULT);

    if (members > PN_TT_GDN_MAX_MEMBERS)
        members = PN_TT_GDN_MAX_MEMBERS;
    if (members > device->gdn_shape.v_heads)
        members = device->gdn_shape.v_heads;
    while (members > 1u && (device->gdn_shape.v_heads % members) != 0u)
        members--;
    return members;
}

/**
 * Arm the EXTRA GDN residencies of a head split, if this run asked for one.
 *
 * They take the SAME circular-buffer placement member 0 has, and that is a property of the
 * family rather than a convenience: `resolve_cb_pages()` reads `state_tiles` and the CONV
 * depth and nothing else, so a member owning four heads places exactly what a member owning
 * thirty-two does. Sizing them any other way would be measuring a different residency from
 * the one the model runs.
 *
 * A member that fails to arm is not fatal — the split shrinks to what came up and the run
 * continues, because every member's descriptors are complete on their own. It IS reported,
 * because a silent shrink would show up only as a number that did not move.
 */
static bool gdn_conv_split_enabled(void);
static uint32_t gdn_conv_members(const pn_tt_device *device);
static uint32_t gdn_delta_members(const pn_tt_device *device);

static void gdn_arm_head_members(pn_tt_device *device, const ft_tt_op_gdn_request *shape)
{
    const uint32_t members = gdn_head_member_count(device);
    const uint32_t first = gdn_env_u32(PN_TT_GDN_MEMBER_CORE_FIRST_ENV,
                                       PN_TT_GDN_MEMBER_CORE_DEFAULT);
    uint32_t index;

    device->gdn_members = 1u;
    if (members < 2u)
        return;
    for (index = 0; index + 1u < members; index++) {
        pn_tt_residency *member = &device->gdn_member[index];
        ft_tt_launch_cb_entry placed[FT_TT_OP_GDN_CB_COUNT];
        ft_tt_op_gdn_request request = *shape;

        request.phase = FT_TT_OP_GDN_PHASE_CONV;
        request.head_first = 0u;
        request.head_count = 0u;
        request.activation_in.tile_bytes = PN_TT_GDN_TILE_BYTES;
        request.activation_out.tile_bytes = PN_TT_GDN_TILE_BYTES;
        /* L5: members 1..7 on `first` upward (133..139); members 8..15 on a second block
         * (`PN_TT_GDN_MEMBER_CORE_SECOND`, default 50 — cores 0..70 probed healthy
         * 2026-10-05 and used by nothing else). */
        const uint32_t core = index < 7u ? first + index
                                         : gdn_env_u32("PN_TT_GDN_MEMBER_CORE_SECOND", 50u) +
                                               (index - 7u);

        if (!pn_tt_prepare_family_on_core(device, member, PN_TT_GDN_PROGRAM, "gdn-member",
                                          PN_TT_GDN_MEMBER_CORE_FIRST_ENV, core,
                                          PN_TT_ONCARD_TRIAD_MASK))
            break;
        if (ft_tt_op_gdn_place_cbs(&request, placed, member->cb_buffer,
                                   sizeof(member->cb_buffer), &member->cb_mask) != FT_TT_OK ||
            !pn_tt_start_family(member)) {
            pn_tt_residency_release(member);
            break;
        }
        device->gdn_members++;
    }
    if (device->gdn_members != members)
        fprintf(stderr, "pn_tt: gdn HEAD SPLIT — asked for %u members, %u armed; the split "
                        "runs at %u\n", members, device->gdn_members, device->gdn_members);
    fprintf(stderr, "pn_tt: gdn HEAD SPLIT — PHASE DELTA over %u members from core %u, "
                    "%u v-heads each; PHASE CONV over %u member(s)\n",
            gdn_delta_members(device), device->gdn.core_index,
            device->gdn_shape.v_heads / gdn_delta_members(device),
            gdn_conv_split_enabled() ? gdn_conv_members(device) : 1u);
}

int pn_tt_device_arm_gdn(pn_tt_device *device, const pn_tt_gdn_shape *shape)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDN_CB_COUNT];
    ft_tt_op_gdn_request request;

    if (!device || !shape || shape->layer_count == 0u)
        return -1;
    device->gdn_shape = *shape;
    request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);
    /* The placement needs operands only to pass the encoder's "is this set" test; the ones
     * the descriptors carry are per call. Any fp32 operand answers it. */
    request.activation_in.tile_bytes = PN_TT_GDN_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDN_TILE_BYTES;
    if (!ft_tt_op_gdn_shape_is_supported(&request)) {
        fprintf(stderr, "pn_tt: the gdn family does not serve S=%u, %u v-heads over %u "
                        "k-heads, conv_dim %u, %u taps — the recurrence stays on the host\n",
                shape->state_size, shape->v_heads, shape->k_heads, shape->conv_dim,
                shape->conv_taps);
        return 0;
    }

    if (!pn_tt_prepare_family(device, &device->gdn, PN_TT_GDN_PROGRAM, "gdn",
                              PN_TT_CORE_GDN_ENV, PN_TT_CORE_GDN_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return 0;
    if (ft_tt_op_gdn_place_cbs(&request, placed, device->gdn.cb_buffer,
                               sizeof(device->gdn.cb_buffer), &device->gdn.cb_mask) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdn circular buffers do not fit at S=%u, %u taps\n",
                shape->state_size, shape->conv_taps);
        pn_tt_residency_release(&device->gdn);
        return 0;
    }
    if (pn_tt_gdn_place_regions(device) != 0 ||
        gdn_stage_grow(device, ft_tt_op_gdn_conv_stream_bytes(&request)) != 0) {
        fprintf(stderr, "pn_tt: the gdn family could not be staged in card DRAM\n");
        pn_tt_residency_release(&device->gdn);
        /* A HALF-ALLOCATED region table is worse than none: `pn_tt_gdn_is_armed()` would
         * still refuse on `running`, but a later reader of the table would find addresses for
         * some layers and zeros for the rest. The DRAM itself is not reclaimed — the
         * allocator frees LIFO by design — and a caller that reaches here is not continuing. */
        free(device->gdn_region);
        device->gdn_region = NULL;
        return 0;
    }
    if (!pn_tt_start_family(&device->gdn))
        return 0;
    device->oncard[PN_TT_ONCARD_GDN] = true;
    gdn_arm_head_members(device, &request);
    return 0;
}

bool pn_tt_gdn_is_armed(const pn_tt_device *device)
{
    return device && device->gdn.running && device->gdn_region != NULL;
}

/**
 * Zero one carried-state buffer: ON CARD by the puller's FILL (`PN_TT_ONCARD_FILL=0` off; the
 * ~120 MB of a sequence start was 0.43 s over the link), else over the link. RED `fill_short`
 * leaves the last 4 KiB of each buffer unzeroed — the state of the previous sequence leaks
 * into the next one and must DIFFER.
 */
static int gdn_zero_buffer(pn_tt_device *device, uint32_t bank, uint32_t address, size_t bytes)
{
    const size_t want = pn_tt_red("fill_short") && bytes > 4096u ? bytes - 4096u : bytes;

    if (pn_tt_env_u32("PN_TT_ONCARD_FILL", 1u) != 0u &&
        pn_tt_fill_dram_oncard(device, bank, address, want, 0u) == 0)
        return 0;
    return pn_tt_zero_dram(device, bank, address, want);
}

/** `PN_TT_FILL_VERIFY=1`: read every zeroed buffer back and count the 32-bit words that are
 *  not zero — the gate of the on-card fill is bytes, not an argmax. */
static void gdn_state_reset_verify(pn_tt_device *device, const ft_tt_op_gdn_request *request)
{
    const size_t sizes[2] = { ft_tt_op_gdn_conv_state_bytes(request),
                              ft_tt_op_gdn_state_bytes(request) };
    uint64_t nonzero = 0u, words = 0u;
    uint32_t *buffer = malloc(sizes[0] > sizes[1] ? sizes[0] : sizes[1]);
    uint32_t layer, which;
    size_t i;

    if (!buffer)
        return;
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++)
        for (which = 0; which < 2u; which++) {
            const pn_tt_gdn_buffer *b = which ? &device->gdn_region[layer].state
                                              : &device->gdn_region[layer].window;

            if (pn_tt_read_dram(device, b->bank, b->address, buffer, sizes[which]) != 0) {
                fprintf(stderr, "pn_tt: gdn — FILL VERIFY could not read layer %u\n", layer);
                continue;
            }
            for (i = 0; i < sizes[which] / 4u; i++)
                nonzero += buffer[i] != 0u;
            words += sizes[which] / 4u;
        }
    free(buffer);
    fprintf(stderr, "pn_tt: gdn — FILL VERIFY %llu of %llu state words NOT zero%s\n",
            (unsigned long long)nonzero, (unsigned long long)words,
            pn_tt_red("fill_short") ? " [RED fill_short]" : "");
}

int pn_tt_gdn_state_reset(pn_tt_device *device)
{
    ft_tt_op_gdn_request request;
    uint32_t layer;

    if (!pn_tt_gdn_is_armed(device))
        return -1;
    /* STEP 3 (A): a new sequence — the on-card dispatch model re-learns its positions. */
    pn_tt_dispatch_sequence_reset();
    request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++) {
        pn_tt_gdn_region *region = &device->gdn_region[layer];

        if (gdn_zero_buffer(device, region->window.bank, region->window.address,
                            ft_tt_op_gdn_conv_state_bytes(&request)) != 0 ||
            gdn_zero_buffer(device, region->state.bank, region->state.address,
                            ft_tt_op_gdn_state_bytes(&request)) != 0)
            return -1;
    }
    if (pn_tt_env_u32("PN_TT_FILL_VERIFY", 0u) != 0u)
        gdn_state_reset_verify(device, &request);
    /* Announced because "position 0 is the sequence boundary" is an ASSUMPTION about the
     * caller, not something this file can check: a caller resuming from a prefix cache never
     * reaches here and would inherit the previous sequence's state silently. One line a
     * sequence makes the assumption observable in a log instead of only in a comment. */
    fprintf(stderr, "pn_tt: gdn — carried state zeroed for %u layers at position 0\n",
            device->gdn_shape.layer_count);
    return 0;
}

/* ================================================================
 * STEP 4 S2: snapshots of the carried state (the verify's rollback)
 * ================================================================ */

/* Every row of a verify writes its new state to a SNAPSHOT SLOT instead of over the state it
 * read (FT_TT_OP_GDN_FLAG_STATE_OUT), and the slot and the live region then trade places: no
 * copy anywhere. After rows 0..k-1 the live region holds the state after row k-1 and slot r
 * the state BEFORE row r, so a rollback to "r rows kept" is one more trade. */

int pn_tt_gdn_snapshot_reserve(pn_tt_device *device, uint32_t slots)
{
    const uint32_t layers = device ? device->gdn_shape.layer_count : 0u;
    pn_tt_gdn_region *more;
    uint32_t s, l;

    if (!device || !pn_tt_gdn_is_armed(device) || slots > 16u)
        return -1;
    if (slots <= device->gdn_snapshot_slots)
        return 0;
    more = realloc(device->gdn_snapshot, (size_t)slots * layers * sizeof(*more));
    if (!more)
        return -1;
    device->gdn_snapshot = more;
    for (s = device->gdn_snapshot_slots; s < slots; s++)
        for (l = 0; l < layers; l++)
            if (pn_tt_gdn_alloc_twin(device, l, &more[s * layers + l]) != 0)
                return -1;
    device->gdn_snapshot_slots = slots;
    return 0;
}

int pn_tt_gdn_project_bank_free(pn_tt_device *device, uint32_t slots, bool checkpoint,
                                uint32_t stacks, uint64_t stack_bytes, uint64_t *out_min_free)
{
    int64_t free_bytes[FT_TT_MAX_DRAM_BANKS];
    uint32_t sets, b, l;
    size_t window, state;
    int64_t lowest;

    if (!device || !out_min_free || !pn_tt_gdn_is_armed(device) || device->dram_bank_count == 0u)
        return -1;
    for (b = 0; b < device->dram_bank_count; b++)
        free_bytes[b] = (int64_t)ft_tt_allocator_free_bytes(&device->allocator,
                                                            FT_TT_MEM_SPACE_DRAM, b);
    pn_tt_gdn_carried_bytes(device, &window, &state);
    window = (window + 63u) & ~(size_t)63u;
    state = (state + 63u) & ~(size_t)63u;
    sets = (slots > device->gdn_snapshot_slots ? slots - device->gdn_snapshot_slots : 0u) +
           (checkpoint && !device->gdn_checkpoint ? 1u : 0u);
    for (l = 0; l < device->gdn_shape.layer_count; l++) {
        free_bytes[device->gdn_region[l].window.bank] -= (int64_t)(sets * window);
        free_bytes[device->gdn_region[l].state.bank] -= (int64_t)(sets * state);
    }
    /* The stacks go where pn_tt_moe_place_stack puts them: the bank with the most free. */
    for (uint32_t i = 0; i < stacks; i++) {
        uint32_t best = 0u;

        for (b = 1; b < device->dram_bank_count; b++)
            if (free_bytes[b] > free_bytes[best])
                best = b;
        free_bytes[best] -= (int64_t)stack_bytes;
    }
    lowest = free_bytes[0];
    for (b = 1; b < device->dram_bank_count; b++)
        if (free_bytes[b] < lowest)
            lowest = free_bytes[b];
    *out_min_free = lowest > 0 ? (uint64_t)lowest : 0u;
    return 0;
}

void pn_tt_gdn_carried_bytes(const pn_tt_device *device, size_t *window, size_t *state)
{
    const ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);

    *window = ft_tt_op_gdn_conv_state_bytes(&request);
    *state = ft_tt_op_gdn_state_bytes(&request);
}

int pn_tt_gdn_alloc_twin(pn_tt_device *device, uint32_t layer, pn_tt_gdn_region *out)
{
    const ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);
    const size_t window = ft_tt_op_gdn_conv_state_bytes(&request);
    const size_t state = ft_tt_op_gdn_state_bytes(&request);

    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count || !out)
        return -1;
    memset(out, 0, sizeof(*out));
    /* THE LIVE REGION'S BANK: one coordinate word carries both (ft_tt_op_gdn.c). */
    if (gdn_alloc_in(device, device->gdn_region[layer].window.bank,
                     (window + 63u) & ~(size_t)63u, false, &out->window) != 0 ||
        gdn_alloc_in(device, device->gdn_region[layer].state.bank,
                     (state + 63u) & ~(size_t)63u, false, &out->state) != 0)
        return -1;
    return 0;
}

int pn_tt_gdn_state_out(pn_tt_device *device, int slot)
{
    if (!device || slot >= (int)device->gdn_snapshot_slots)
        return -1;
    device->gdn_state_out_plus1 = slot < 0 ? 0u : (uint32_t)slot + 1u;
    return 0;
}

void pn_tt_gdn_verify_compact_set(pn_tt_device *device, bool compact)
{
    if (device)
        device->gdn_verify_compact = compact;
}

uint32_t pn_tt_gdn_verify_slots(const pn_tt_device *device, uint32_t rows)
{
    return pn_gdn_verify_slot_count(device && device->gdn_verify_compact, rows);
}

int pn_tt_gdn_verify_row_out(pn_tt_device *device, uint32_t rows, uint32_t row)
{
    if (!device)
        return -1;
    return pn_tt_gdn_state_out(device, pn_gdn_verify_row_slot(device->gdn_verify_compact, rows,
                                                              row));
}

int pn_tt_gdn_verify_row_trade(pn_tt_device *device, uint32_t layer, uint32_t rows,
                               uint32_t row, bool window, bool state)
{
    int slot;

    if (!device)
        return -1;
    slot = pn_gdn_verify_row_slot(device->gdn_verify_compact, rows, row);
    return slot < 0 ? 0 : pn_tt_gdn_layer_swap_part(device, layer, (uint32_t)slot, window, state);
}

uint32_t pn_tt_gdn_verify_rollback_slot(const pn_tt_device *device, uint32_t rows,
                                        uint32_t last_kept)
{
    return pn_gdn_verify_rollback_slot(device && device->gdn_verify_compact, rows, last_kept);
}

/** Multi-row descriptor, row `row` of `rows`: the buffer its state goes to — the slot the map
 *  names, or the live part itself (compact's last row, into S_0's buffer). */
static const pn_tt_gdn_buffer *gdn_rows_out_buffer(const pn_tt_device *device, uint32_t layer,
                                                   uint32_t rows, uint32_t row, bool window)
{
    const int slot = pn_gdn_verify_rows_out_slot(device->gdn_verify_compact, rows, row);
    const pn_tt_gdn_region *r = slot < 0 ? &device->gdn_region[layer]
                                         : &device->gdn_snapshot[(uint32_t)slot *
                                                                     device->gdn_shape.layer_count +
                                                                 layer];

    return window ? &r->window : &r->state;
}

int pn_tt_gdn_layer_swap_part(pn_tt_device *device, uint32_t layer, uint32_t slot, bool window,
                              bool state)
{
    pn_tt_gdn_region *live, *snap;
    pn_tt_gdn_buffer t;

    if (!device || layer >= device->gdn_shape.layer_count || slot >= device->gdn_snapshot_slots)
        return -1;
    live = &device->gdn_region[layer];
    snap = &device->gdn_snapshot[slot * device->gdn_shape.layer_count + layer];
    if (window) {
        t = live->window;
        live->window = snap->window;
        snap->window = t;
    }
    if (state) {
        t = live->state;
        live->state = snap->state;
        snap->state = t;
    }
    return 0;
}

int pn_tt_gdn_layer_swap(pn_tt_device *device, uint32_t layer, uint32_t slot)
{
    return pn_tt_gdn_layer_swap_part(device, layer, slot, true, true);
}

int pn_tt_gdn_defer(pn_tt_device *device, bool defer)
{
    if (!device)
        return -1;
    device->gdn_defer = defer;
    return 0;
}

int pn_tt_gdn_await_deferred(pn_tt_device *device)
{
    pn_tt_residency *set[PN_TT_GDN_MAX_MEMBERS];
    bool bad[PN_TT_GDN_MAX_MEMBERS];
    uint32_t index;
    int failed = 0;

    if (!device)
        return -1;
    memset(bad, 0, sizeof(bad));
    for (index = 0; index < device->gdn_pending_members && index < PN_TT_GDN_MAX_MEMBERS; index++)
        set[index] = index == 0u ? &device->gdn : &device->gdn_member[index - 1u];
    /* S1: when every split of the window pulled, `gdn`'s last completion covers each member's
     * (member 0 runs its rows in order, and each one waited for the members it split over). */
    if (device->gdn_pending_pulled && device->gdn_pending_members > 1u) {
        failed = pn_tt_group_pull_await(&device->gdn_pull, true, set,
                                        device->gdn_pending_members, bad, NULL);
    } else {
        for (index = 0; index < device->gdn_pending_members; index++)
            if (pn_tt_residency_await(set[index], PN_TT_OP_TIMEOUT_MS) != 0)
                failed = 1;
    }
    device->gdn_pending_pulled = false;
    device->gdn_pending_members = 0u;
    device->gdn_pending_depth = 0u;
    return failed ? -1 : 0;
}

int pn_tt_gdn_set_row(pn_tt_device *device, uint32_t row)
{
    if (!device || row > 31u)
        return -1;
    /* RED `gdn_row_zero`: every row reads and writes row 0 — must DIFFER. */
    device->gdn_row = pn_tt_red("gdn_row_zero") ? 0u : row;
    return 0;
}

/** The STATE_OUT slot's buffer for this layer, or NULL (the in-place walk). */
static const pn_tt_gdn_buffer *gdn_state_out_buffer(const pn_tt_device *device, uint32_t layer,
                                                    bool window)
{
    const pn_tt_gdn_region *snap;

    if (device->gdn_state_out_plus1 == 0u)
        return NULL;
    snap = &device->gdn_snapshot[(device->gdn_state_out_plus1 - 1u) *
                                     device->gdn_shape.layer_count + layer];
    return window ? &snap->window : &snap->state;
}

/* ================================================================
 * The conv weight — the TRANSPOSE
 * ================================================================ */

int pn_tt_gdn_conv_weight_upload(pn_tt_device *device, uint32_t layer, const float *weight)
{
    const ftg_shape shape = gdn_shape(device, FT_TT_OP_GDN_PHASE_CONV);
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t taps = device->gdn_shape.conv_taps;
    float *plane;
    uint32_t tap;
    int rc = 0;

    if (!pn_tt_gdn_is_armed(device) || !weight || layer >= device->gdn_shape.layer_count)
        return -1;
    plane = malloc((size_t)conv_dim * sizeof(*plane));
    if (!plane)
        return -1;
    if (layer == 0u)
        device->gdn_weight_oracle = weight;
    /* SHARED: the planes are the chunkwise constant's row 0, which its own upload writes. */
    if (pn_tt_gdn_conv_weight_is_shared(device)) {
        free(plane);
        return 0;
    }

    for (tap = 0; tap < taps && rc == 0; tap++) {
        const uint32_t first = ftg_conv_plane_tile(&shape, tap, 0u);
        /* THE TRANSPOSE, and the whole of it: plane `t` is the STRIDE-`taps` gather of the
         * oracle's row-major `[conv_dim, taps]` starting at `t`.
         * RED PROOF: `(tap * conv_dim, 1)` is the RESHAPE — a contiguous slice, the same
         * bytes, the same length, a different answer, and nothing on the card can see the
         * difference. It is the single most likely thing for a stager to get silently wrong,
         * which is why it is a named mutation the on-card gate can be shown to catch. */
        const uint32_t gather_first = pn_tt_red("gdn_conv_reshape") ? tap * conv_dim : tap;
        const uint32_t gather_stride = pn_tt_red("gdn_conv_reshape") ? 1u : taps;

        if (pn_tt_gdn_stage_plane(weight, gather_first, gather_stride, conv_dim, plane,
                                  device->gdn_stage) != 0)
            rc = -1;
        else
            rc = pn_tt_write_dram(device, device->gdn_region[layer].weight.bank,
                                  device->gdn_region[layer].weight.address +
                                      first * PN_TT_GDN_TILE_BYTES,
                                  device->gdn_stage,
                                  (size_t)shape.conv_tile_cols * PN_TT_GDN_TILE_BYTES);
    }
    free(plane);
    return rc;
}

/* ================================================================
 * Submission
 * ================================================================ */

/**
 * Retire a residency that refused a descriptor, and arm a FRESH one on the same core.
 *
 * The DRAM is untouched: the regions are allocated once and a retired core owns none of them,
 * so the carried state a half-finished token left behind is still there. What is re-placed is
 * the circular-buffer geometry, which is a property of the shape and not of the descriptor.
 *
 * A residency is BOUNDED on purpose — `idle_spin_limit` retires a kernel whose host has
 * stopped feeding it — so this fires for a `serve` that idles between requests, not only for
 * a malformed descriptor. Same mechanism as `pn_tt_oncard_rearm()`.
 */
static bool pn_tt_gdn_rearm(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDN_CB_COUNT];
    ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);

    fprintf(stderr, "pn_tt: the gdn residency refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", device->gdn.core_index);
    pn_tt_residency_stop(&device->gdn);
    pn_tt_residency_release(&device->gdn);
    device->oncard_restarts++;
    /* NOTHING BOOTS FIRMWARE ONTO A CORE THAT DID NOT RETIRE. Measured on silicon
     * 2026-08-21: a refused descriptor leaves the compute thread spinning out its bound, the
     * stop reported `processors still running: 0x4`, and the boot that followed WEDGED core 90
     * at `instrn_buf_status = 0x66` — a state only `ft_tt_core recover --rung 3` and sudo can
     * clear. A failed descriptor is a bad token; a wedged Tensix takes the process AND the next
     * one, because the next boot lands on the same core. */
    if (!pn_tt_residency_retired(&device->gdn)) {
        fprintf(stderr, "pn_tt: the gdn residency did not retire — REFUSING to arm a fresh "
                        "one on core %u. Booting firmware onto a core whose compute thread is "
                        "still running is the wedge, and it needs sudo to clear.\n",
                device->gdn.core_index);
        return false;
    }

    request.activation_in.tile_bytes = PN_TT_GDN_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDN_TILE_BYTES;
    if (!pn_tt_prepare_family(device, &device->gdn, PN_TT_GDN_PROGRAM, "gdn",
                              PN_TT_CORE_GDN_ENV, PN_TT_CORE_GDN_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_gdn_place_cbs(&request, placed, device->gdn.cb_buffer,
                               sizeof(device->gdn.cb_buffer), &device->gdn.cb_mask) !=
        FT_TT_OK) {
        pn_tt_residency_release(&device->gdn);
        return false;
    }
    return pn_tt_start_family(&device->gdn);
}

/**
 * One descriptor, with the same re-arm-once recovery the other on-card families use.
 *
 * A GDN re-arm is safe in a way the sampler's is not: every address in the parameter block is
 * a DRAM one, and the carried state lives in DRAM and is untouched by retiring the core. So
 * the SAME words are valid against a fresh residency.
 */
/**
 * A positive integer measurement knob, or 0 when it is unset or not a number.
 *
 * Read at every descriptor rather than cached, for the same reason `pn_tt_red()` is: what
 * these change is the WHOLE run, and a knob read once at arm time cannot be turned on for a
 * probe that runs after the residency is up.
 */
static uint32_t gdn_probe_knob(const char *name)
{
    const char *value = getenv(name);
    long parsed;

    if (!value || !*value)
        return 0u;
    parsed = strtol(value, NULL, 10);
    return parsed > 0 ? (uint32_t)parsed : 0u;
}

/**
 * Retire ONE head-split member and arm a fresh one ON THE CORE IT ALREADY HELD.
 *
 * The core is captured BEFORE the release, because `pn_tt_residency_release()` zeroes the
 * handle — re-reading `core_index` afterwards would arm member 3 on core 0. Member 0 is the
 * `gdn` residency itself and goes through `pn_tt_gdn_rearm()`, which is the same operation
 * spelled against the environment default.
 */
/** Arm a fresh `gdn` (member 0) or head-split member residency on `core`: load, boot its
 *  firmware, place the CBs for the shape, GO. The handle must be released (or never armed). */
static bool gdn_arm_on_core(pn_tt_device *device, uint32_t member_index, uint32_t core)
{
    pn_tt_residency *member = member_index == 0u ? &device->gdn : &device->gdn_member[member_index - 1u];
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDN_CB_COUNT];
    ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);

    request.activation_in.tile_bytes = PN_TT_GDN_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDN_TILE_BYTES;
    if (!pn_tt_prepare_family_on_core(device, member, PN_TT_GDN_PROGRAM,
                                      member_index == 0u ? "gdn" : "gdn-member",
                                      member_index == 0u ? PN_TT_CORE_GDN_ENV
                                                         : PN_TT_GDN_MEMBER_CORE_FIRST_ENV,
                                      core, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_gdn_place_cbs(&request, placed, member->cb_buffer, sizeof(member->cb_buffer),
                               &member->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(member);
        return false;
    }
    return pn_tt_start_family(member);
}

static bool gdn_member_rearm(pn_tt_device *device, uint32_t member_index)
{
    pn_tt_residency *member;
    uint32_t core;

    if (member_index == 0u)
        return pn_tt_gdn_rearm(device);
    member = &device->gdn_member[member_index - 1u];
    core = member->core_index;
    fprintf(stderr, "pn_tt: gdn member %u refused a descriptor — retiring it by ASKING and "
                    "arming a fresh one on core %u\n", member_index, core);
    pn_tt_residency_stop(member);
    pn_tt_residency_release(member);
    device->oncard_restarts++;
    /* The same refusal as `pn_tt_gdn_rearm()`'s, for the same reason. */
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: gdn member %u did not retire — REFUSING to arm a fresh one on "
                        "core %u\n", member_index, core);
        return false;
    }
    return gdn_arm_on_core(device, member_index, core);
}

uint32_t pn_tt_gdn_lend_count(const pn_tt_device *device)
{
    return pn_tt_gdn_is_armed(device) ? device->gdn_members : 0u;
}

uint32_t pn_tt_gdn_lend(pn_tt_device *device, uint32_t member_index)
{
    pn_tt_residency *member = member_index == 0u ? &device->gdn : &device->gdn_member[member_index - 1u];
    const uint32_t core = member->core_index;

    /* Running, or idle-retired with its firmware still on the core: both lend (stop is safe
     * on a residency that is not running). */
    if (!member->core_booted)
        return UINT32_MAX;
    pn_tt_residency_stop(member);
    /* The retire is asked BEFORE the release: release zeroes the handle and its answer. */
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: gdn member %u (core %u) did not retire — NOT lent\n",
                member_index, core);
        return UINT32_MAX;
    }
    pn_tt_residency_release(member);
    /* Release keeps `core_booted`, and the core-holder check would then refuse the borrower:
     * park the exited triad and drop the claim, as pn_tt_attention_grid.c does after its own
     * retire. The next boot on this core (attention, or the restore) asserts reset again. */
    if (ft_tt_launch_assert_reset(&device->runtime.noc, member->core_x, member->core_y,
                                  FT_TT_SOFT_RESET_ALL_RISCS) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: gdn member %u (core %u) could not be parked — NOT lent\n",
                member_index, core);
        return UINT32_MAX;
    }
    member->core_booted = false;
    return core;
}

bool pn_tt_gdn_restore(pn_tt_device *device, uint32_t member_index, uint32_t core)
{
    return gdn_arm_on_core(device, member_index, core);
}

/**
 * PHASE DELTA over a head split: one descriptor per member, ALL ENQUEUED BEFORE ANY IS
 * WAITED ON.
 *
 * The two loops being separate is the measurement and not a style. Enqueue-wait per member
 * would run `members` SERIAL descriptors and print the flat line "it does not parallelise",
 * produced by this function rather than by the card — the exact failure the core-scaling
 * probe's header calls out, and the reason its own loops are split the same way.
 *
 * NO TWO MEMBERS TOUCH THE SAME DRAM BYTE. The carried state is head-major, so a head slice
 * is a contiguous, disjoint region. `y` is the one operand they share, and they share it
 * SAFELY: it is a tilized [v_heads, S] matrix with one ROW per head, and the writer scatters
 * each head's answer at row granularity into `head % 32`, so two members write disjoint byte
 * ranges of the same tiles. The gate block is READ whole by every member, which is a read.
 *
 * Time is charged to the SET, once, so `us each` stays the wall a layer's DELTA costs rather
 * than the sum of what its members cost in parallel.
 *
 * @return 0, or -1 with the run's DELTA left incomplete — there is no partial retry here,
 *   because a member that failed has left its slice of the recurrence un-advanced and the
 *   next token's state would be a mixture of two positions.
 */
void pn_tt_gdn_report_phases(const pn_tt_device *device)
{
    static const char *const name[3] = { "?", "CONV", "DELTA" };
    uint32_t phase;

    if (!device)
        return;
    for (phase = 1u; phase < 3u; phase++) {
        const uint64_t ops = device->gdn_phase_ops[phase];
        const uint64_t recovered = device->gdn_phase_recovered_ops[phase];

        if (ops == 0u && recovered == 0u)
            continue;
        fprintf(stderr, "pn_tt: gdn — %-5s %.4f s over %llu clean descriptors (%.1f us "
                        "each), %llu replayed over %llu re-arms (%.4f s)\n",
                name[phase], device->gdn_phase_seconds[phase], (unsigned long long)ops,
                ops ? 1e6 * device->gdn_phase_seconds[phase] / (double)ops : 0.0,
                (unsigned long long)recovered,
                (unsigned long long)device->gdn_phase_rearms[phase],
                device->gdn_phase_recovered_seconds[phase]);
    }
}

/**
 * Charge one submit bracket to its phase, CLEAN AND RECOVERED KEPT APART.
 *
 * The stopwatch every caller here holds spans the refusal, the re-arm — a residency teardown
 * and relaunch — and the resubmit. Summing that into `seconds / ops` prices the teardown as
 * the op's arithmetic, and it has already produced a wrong headline number: PHASE CONV read
 * **4,502 us a descriptor against 738.2** when a circular-buffer write pointer made every CONV
 * following a DELTA refuse, and the unsourced "CONV is DATA-DEPENDENT, 4,571 us on a
 * degenerate stream" has the same shape. So a bracket that re-armed anything is charged
 * elsewhere and the reader prints the count beside every figure it quotes.
 *
 * @param phase        `FT_TT_OP_GDN_PHASE_*`; anything else is dropped rather than aliased
 * @param seconds      the whole bracket
 * @param descriptors  descriptors it covered (`PN_TT_GDN_BATCH` submits several)
 * @param rearms       re-arms inside it; 0 makes the bracket CLEAN
 */
static void gdn_account_phase(pn_tt_device *device, uint32_t phase, double seconds,
                              uint64_t descriptors, uint64_t rearms)
{
    if (phase >= 3u)
        return;
    if (rearms == 0u) {
        device->gdn_phase_seconds[phase] += seconds;
        device->gdn_phase_ops[phase] += descriptors;
        return;
    }
    device->gdn_phase_recovered_seconds[phase] += seconds;
    device->gdn_phase_recovered_ops[phase] += descriptors;
    device->gdn_phase_rearms[phase] += rearms;
}

static bool gdn_conv_split_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_GDN_CONV_SPLIT");

        /* ON since 2026-10-04: CONV 941 -> 309 us at 4 members, decode 9.18 -> 10.7 tok/s,
         * IDENTICAL; `gdn_conv_col_dup` red DIFFERS. `=0` is the whole-core CONV. */
        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        if (cached)
            fprintf(stderr, "pn_tt: gdn PHASE CONV splits by tile column over the members\n");
    }
    return cached != 0;
}

/** L5: PHASE DELTA may run over FEWER members than CONV — measured 2026-10-05 at 16: CONV
 *  153 -> 120 us but DELTA 147 -> 162 us, so DELTA stayed at 8. L10 D7 (2026-10-21, under the
 *  puller): DELTA over all 16 is +1.2% on shipped spec decode, IDENTICAL — DEFAULT every member.
 *  `PN_TT_GDN_DELTA_CORES=N` caps DELTA's set at the first N members; it must divide v_heads or
 *  DELTA uses them all. */
static uint32_t gdn_delta_members(const pn_tt_device *device)
{
    const uint32_t asked = pn_tt_env_u32("PN_TT_GDN_DELTA_CORES", device->gdn_members);

    if (asked >= 1u && asked < device->gdn_members && device->gdn_shape.v_heads % asked == 0u)
        return asked;
    return device->gdn_members;
}

/** `PN_TT_GDN_CONV_WIDE` (DEFAULT ON): CONV slices may start and span past column 255
 *  (foundry's wide column encoding, `kGdnByteColFirstHigh`), so Qwen3.8's 320 columns split
 *  over all 16 members. Measured 2026-11-17 (p150): k1 11.53 -> 13.22 tok/s, plain 6.98 ->
 *  7.54, md5-exact on every arm. `=0` keeps the narrow cap (4 members of 80). */
static uint32_t gdn_conv_col_max(void)
{
    static int wide = -1;

    if (wide < 0) {
        wide = pn_tt_gdn_conv_wide_enabled(getenv(PN_TT_GDN_CONV_WIDE_ENV));
        fprintf(stderr, "pn_tt: gdn PHASE CONV slices may address %s (%s=%s)\n",
                wide ? "every column" : "columns 0-255 only", PN_TT_GDN_CONV_WIDE_ENV,
                wide ? "1, the default; =0 for the narrow cap" : "0");
    }
    return wide ? FT_TT_OP_GDN_COL_MAX : PN_TT_GDN_COL_BYTE_MAX;
}

/** The column count every CONV slice is a multiple of (foundry's slice granule: 4 at
 *  Qwen3.8's placement). An 18-column slice ran its CONV and broke the DELTA after it on the
 *  card; the encoder refuses it, so the split never asks for one. */
static uint32_t gdn_conv_granule(const pn_tt_device *device)
{
    return ft_tt_op_gdn_conv_slice_granule(device->gdn_shape.state_size,
                                           device->gdn_shape.conv_taps);
}

/** CHARACTERIZATION (2026-10-05): `PN_TT_GDN_CONV_CORES` caps CONV's set like DELTA's, so
 *  armed-but-idle members can be told apart from members that work.
 *
 *  A stream wider than 255 tile columns (Qwen3.8: conv_dim 10240 = 320) cannot be split 16
 *  ways in the narrow encoding — the 14th slice starts at column 260, which failed the first
 *  decode. Every member count is capped to the largest that divides the stream into slices
 *  the encoding addresses (narrow: 320 -> 4 members of 80; `PN_TT_GDN_CONV_WIDE=1`: every
 *  armed member); 1 = the whole-core CONV. */
static uint32_t gdn_conv_members(const pn_tt_device *device)
{
    const uint32_t asked = pn_tt_env_u32("PN_TT_GDN_CONV_CORES", device->gdn_members);
    const uint32_t cols = device->gdn_shape.conv_dim / FT_TT_OP_TILE_DIM;
    const uint32_t members = pn_tt_gdn_conv_member_count(cols, device->gdn_members, asked,
                                                         gdn_conv_granule(device),
                                                         gdn_conv_col_max());

    if (members < device->gdn_members && members != asked) {
        static bool said;

        if (!said)
            fprintf(stderr, "pn_tt: gdn PHASE CONV over %u member(s), not %u — %u tile columns "
                            "do not split %u ways into slices the descriptor's column bytes "
                            "encode (%s=0 set: the 255-column cap)\n", members,
                    device->gdn_members, cols, device->gdn_members, PN_TT_GDN_CONV_WIDE_ENV);
        said = true;
    }
    return members;
}

/** MULTI-ROW DELTA: the per-row state addresses into the slot after the descriptor's own,
 *  on this residency, before its enqueue (and again before a replay after a re-arm). */
static int gdn_write_row_extension(pn_tt_residency *residency,
                                   const ft_tt_op_gdn_request *request)
{
    uint32_t extension[FT_TT_OP_GDN_WORDS];

    if (request->row_count < 2u)
        return 0;
    if (ft_tt_op_gdn_row_extension(request, extension) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdn multi-row extension is one the family refuses\n");
        return -1;
    }
    return pn_tt_residency_write_param_extension(residency, 1u, extension);
}

static int gdn_submit_delta_split(pn_tt_device *device, const ft_tt_op_gdn_request *request)
{
    const uint32_t members = request->phase == FT_TT_OP_GDN_PHASE_CONV
                                 ? gdn_conv_members(device)
                                 : gdn_delta_members(device);
    const bool conv = request->phase == FT_TT_OP_GDN_PHASE_CONV;
    /* RED PROOFS on the CONV slices (pn_tt_gdn_conv_split.h): `gdn_conv_col_dup` (every member
     * takes member 0's columns), `gdn_conv_col_drop` (the last block is never written),
     * `gdn_conv_col_wrap` (first/count lose their high byte) — each must DIFFER. */
    const pn_tt_gdn_slice_red red = pn_tt_red("gdn_conv_col_dup")
                                        ? PN_TT_GDN_SLICE_RED_COL_DUP
                                    : pn_tt_red("gdn_conv_col_drop")
                                        ? PN_TT_GDN_SLICE_RED_COL_DROP
                                    : pn_tt_red("gdn_conv_col_wrap") ? PN_TT_GDN_SLICE_RED_COL_WRAP
                                                                     : PN_TT_GDN_SLICE_EXACT;
    uint32_t words[PN_TT_GDN_MAX_MEMBERS][FT_TT_OP_GDN_WORDS];
    pn_tt_residency *set[PN_TT_GDN_MAX_MEMBERS];
    pn_tt_residency *table[PN_TT_GDN_MAX_MEMBERS];
    bool bad[PN_TT_GDN_MAX_MEMBERS];
    double started;
    uint32_t index, sequence = 0u;
    uint64_t rearms = 0;
    bool pulled;
    int failed = 0;

    memset(bad, 0, sizeof(bad));
    for (index = 0; index < members; index++) {
        /* THIS DESCRIPTOR'S head count, not the shape's — `PN_TT_GDN_HEADS` narrows the
         * request and the two knobs have to compose or refuse, not silently address heads the
         * descriptor does not claim. The caller has already checked that the count divides. */
        ft_tt_op_gdn_request sliced;

        if (pn_tt_gdn_slice(request, members, index, ft_tt_op_gdn_conv_depth(),
                            conv ? red : PN_TT_GDN_SLICE_EXACT, &sliced) != 0 ||
            ft_tt_op_gdn_parameters(&sliced, words[index]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the gdn %s slice %u of %u is one the family refuses\n",
                    conv ? "column" : "head", index, members);
            return -1;
        }
        set[index] = index == 0u ? &device->gdn : &device->gdn_member[index - 1u];
    }

    /* RING FLOW CONTROL. A deferred member holds one descriptor a row, unawaited, and
     * ft_tt_queue_stage() does not check occupancy: past `capacity` the host rewrote ring
     * slots the card had not read yet (10+ verify rows DIFFERED on an 8-slot ring). A full
     * ring drains first. RED `gdn_defer_overrun` skips the drain — must DIFFER when the rows
     * exceed the ring. */
    if (device->gdn_defer && device->gdn_pending_depth >= set[0]->queue.capacity &&
        !pn_tt_red("gdn_defer_overrun")) {
        if (pn_tt_gdn_await_deferred(device) != 0)
            return -1;
        if (device->gdn_defer_drains++ == 0u)
            fprintf(stderr, "pn_tt: gdn — deferred rows fill the %u-slot ring; draining it "
                            "(flow control)\n", set[0]->queue.capacity);
    }
    /* S1: `gdn` posts only once every member has (pn_tt_group_pull.h). The table names the
     * whole set, so CONV and DELTA may split over different counts. */
    for (index = 0; index < device->gdn_members && index < PN_TT_GDN_MAX_MEMBERS; index++)
        table[index] = index == 0u ? &device->gdn : &device->gdn_member[index - 1u];
    pulled = pn_tt_group_pull_decide(&device->gdn_pull, pn_tt_group_pull_decode(device), table,
                                     device->gdn_members, members, "gdn");
    if (pulled)
        sequence = pn_tt_group_pull_next(&device->gdn_pull);
    started = pn_tt_now_seconds();
    for (index = 0; index < members; index++) {
        const pn_tt_group_pull_words pull = pn_tt_group_pull_words_for(
            index, members, sequence, pulled && pn_tt_red("group_pull_short"));

        /* S2c DEFERRED: a queued descriptor keeps its own parameter block until it runs —
         * the shipped single slot would be overwritten by the next row's. */
        set[index]->param_slot = device->gdn_defer ? 1u + device->gdn_row : 0u;
        if (gdn_write_row_extension(set[index], request) != 0 ||
            (pulled ? pn_tt_residency_enqueue_pull(set[index], FT_TT_QUEUE_OP_GDN, words[index],
                                                   pull.a3, pull.a4, pull.a5)
                    : pn_tt_residency_enqueue(set[index], FT_TT_QUEUE_OP_GDN, words[index],
                                              1u)) != 0)
            failed = bad[index] = 1;
        set[index]->param_slot = 0u;
    }
    /* S2c DEFERRED: queued behind each other on every member (in-order per core is the only
     * ordering a row of the verify needs from the previous row); awaited together later. */
    if (device->gdn_defer && !failed) {
        if (members > device->gdn_pending_members)
            device->gdn_pending_members = members;
        device->gdn_pending_pulled = (device->gdn_pending_depth == 0u ||
                                      device->gdn_pending_pulled) && pulled;
        device->gdn_pending_depth++;
        device->oncard_ops += 1u;
        return 0;
    }
    /* EVERY member is awaited even after one has failed (unless the pull vouches for it). An
     * enqueue that got as far as the ring has a descriptor the card will run whatever this
     * host does next, and leaving it un-awaited desynchronises that member's cursor for the
     * rest of the run. */
    if (pulled)
        failed = pn_tt_group_pull_await(&device->gdn_pull, true, set, members, bad, NULL) != 0;
    else
        for (index = 0; index < members; index++)
            if (pn_tt_residency_await(set[index], PN_TT_OP_TIMEOUT_MS) != 0)
                failed = bad[index] = 1;

    /* PER-MEMBER RE-ARM AND REPLAY, AND IT RUNS ONLY WHERE THIS FUNCTION USED TO RETURN -1.
     *
     * A resident triad is BOUNDED: ~40 s with no descriptor retires it. Until the chunkwise
     * prefill landed, this family ran two descriptors per SSM layer per POSITION and could
     * never idle that long inside a request — a batched prefill running the chunkwise mixers
     * leaves it idle for the WHOLE PROMPT, which at 8k is over a minute, so the first decode
     * token after one would have found four retired residencies and this function had no
     * recovery at all while `gdn_submit()` beside it has had one since it was written.
     *
     * Replaying ONE member's slice is the same computation because a refused descriptor
     * leaves no circular-buffer traffic behind it — the reader aborts before the rebase — and
     * because the head slices are disjoint in DRAM: the carried state is head-major, and `y`
     * is scattered at row granularity. A member that succeeded is not touched.
     */
    if (failed) {
        failed = 0;
        for (index = 0; index < members; index++) {
            if (!bad[index])
                continue;
            rearms++;
            if (!gdn_member_rearm(device, index) ||
                gdn_write_row_extension(set[index], request) != 0 ||
                pn_tt_residency_submit(set[index], FT_TT_QUEUE_OP_GDN, words[index],
                                       PN_TT_OP_TIMEOUT_MS) != 0)
                failed = 1;
        }
    }
    {
        const double elapsed = pn_tt_now_seconds() - started;

        device->oncard_seconds += elapsed;
        device->oncard_ops += 1u;
        gdn_account_phase(device, conv ? FT_TT_OP_GDN_PHASE_CONV : FT_TT_OP_GDN_PHASE_DELTA,
                          elapsed, 1u, rearms);
    }
    return failed ? -1 : 0;
}

static int gdn_submit(pn_tt_device *device, const ft_tt_op_gdn_request *request)
{
    /* THE QUEUE-DEPTH INSTRUMENT. `PN_TT_GDN_BATCH=K` enqueues K copies of this descriptor
     * and waits ONCE, so the host's parameter write, ring write, head bump and first poll are
     * paid once for K descriptors instead of K times. `seconds(K) = host_fixed +
     * K * on_card_serial`, and two values of K solve both terms — which is the whole question
     * "is the 2.3 ms a round trip or the card's own work".
     *
     * IT BREAKS THE RECURRENCE and is not a fallback: every copy runs the same parameter
     * block, so copies 2..K decay and update a state the model did not ask them to. The
     * TIMING is what this arm produces; the text it generates is not evidence of anything. */
    const uint32_t repeat = gdn_probe_knob("PN_TT_GDN_BATCH");
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint64_t rearms = 0;
    double started;
    double elapsed;
    ft_tt_op_gdn_request flagged;

    /* M6: `PN_TT_GDN_NOC_REPLICATE` — the DELTA reader builds its replicated k/q tiles with
     * local NoC copies (kGdnFlagNocReplicate). Same bytes; a reader-side cost only. */
    {
        static int noc_replicate = -1;

        static int conv_strip = -1;

        if (noc_replicate < 0)
            noc_replicate = (int)pn_tt_env_u32("PN_TT_GDN_NOC_REPLICATE", 1u);
        /* M9: `PN_TT_GDN_CONV_STRIP` — PHASE CONV moves only ROW 0 of every tile it reads
         * and writes (2 x 64 B instead of 4 KiB); all its arithmetic is elementwise, and
         * rows 1-31 of its DRAM buffers are left as they are. ON since 2026-10-05: CONV
         * 268 -> 153.5 us, decode 15.47 -> 16.00 tok/s, IDENTICAL; red DIFFERS. =0 is off. */
        if (conv_strip < 0)
            conv_strip = (int)pn_tt_env_u32("PN_TT_GDN_CONV_STRIP", 1u);
        flagged = *request;
        if (conv_strip != 0) {
            flagged.flags |= FT_TT_OP_GDN_FLAG_CONV_STRIP;
            /* RED: columns 16-31 of every CONV operand stay stale — must DIFFER. */
            if (pn_tt_red("gdn_conv_strip_half"))
                flagged.flags |= FT_TT_OP_GDN_FLAG_RED_STRIP_HALF;
        }
        request = &flagged;
        if (noc_replicate != 0) {
            flagged.flags |= FT_TT_OP_GDN_FLAG_NOC_REPLICATE;
            if (pn_tt_red("gdn_noc_replicate_no_copy"))
                flagged.flags |= FT_TT_OP_GDN_FLAG_RED_NO_COPY;
        }
    }

    /* THE HEAD SPLIT TAKES PHASE DELTA AND NOTHING ELSE, and only when this run armed
     * members for it. `PN_TT_GDN_BATCH` is refused alongside it rather than combined: the
     * batch instrument deliberately re-runs one parameter block K times and would give every
     * member a different number of repeats of a broken recurrence. */
    if (gdn_delta_members(device) > 1u && request->phase == FT_TT_OP_GDN_PHASE_DELTA &&
        !repeat && request->v_heads >= gdn_delta_members(device) &&
        (request->v_heads % gdn_delta_members(device)) == 0u)
        return gdn_submit_delta_split(device, request);
    /* AND PHASE CONV, by tile column, behind `PN_TT_GDN_CONV_SPLIT` until it is measured: it
     * needs the gdn programs that honour the column range (regenerated 2026-10). The count
     * must divide the stream into slices the CONV slice granule also divides. */
    if (gdn_conv_members(device) > 1u && request->phase == FT_TT_OP_GDN_PHASE_CONV &&
        !repeat && gdn_conv_split_enabled()) {
        const uint32_t cols = request->conv_dim / FT_TT_OP_TILE_DIM;
        const uint32_t per = cols / gdn_conv_members(device);

        if (per > 0u && (cols % gdn_conv_members(device)) == 0u && gdn_conv_granule(device) &&
            (per % gdn_conv_granule(device)) == 0u)
            return gdn_submit_delta_split(device, request);
    }

    if (ft_tt_op_gdn_parameters(request, words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdn phase-%u descriptor is one the family refuses\n",
                request->phase);
        return -1;
    }
    started = pn_tt_now_seconds();
    if (gdn_write_row_extension(&device->gdn, request) != 0 ||
        pn_tt_residency_submit_repeat(&device->gdn, FT_TT_QUEUE_OP_GDN, words,
                                      repeat ? repeat : 1u, PN_TT_OP_TIMEOUT_MS) != 0) {
        rearms = 1u;
        if (!pn_tt_gdn_rearm(device) || gdn_write_row_extension(&device->gdn, request) != 0 ||
            pn_tt_residency_submit_repeat(&device->gdn, FT_TT_QUEUE_OP_GDN, words,
                                          repeat ? repeat : 1u, PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    elapsed = pn_tt_now_seconds() - started;
    device->oncard_seconds += elapsed;
    /* DESCRIPTORS, not host waits — so "seconds per descriptor" stays the same quantity at
     * every batch depth and the two can be divided against each other. */
    device->oncard_ops += repeat ? repeat : 1u;
    gdn_account_phase(device, request->phase, elapsed, repeat ? repeat : 1u, rearms);
    return 0;
}

/**
 * PROVE THE HOST STAGING BEFORE BLAMING THE CARD: write the qkv row back out of DRAM and
 * compare it with what went in, and read the conv weight's tap-0 plane back and compare it
 * with the oracle's `w[c*taps]`.
 *
 * Runs once, on the first CONV of a run, under `PN_TT_GDN_SELFCHECK`. It exists because a
 * wrong answer out of this op has two halves — the host's DRAM layout and the card's
 * arithmetic — and nothing else in the stack can tell them apart. A clean self-check moves
 * the whole search onto the card; a dirty one ends it here.
 */
static void gdn_self_check(pn_tt_device *device, const float *qkv, const float *weight,
                           const pn_tt_gdn_buffer *weight_buffer)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t taps = device->gdn_shape.conv_taps;
    const uint32_t tiles = conv_dim / PN_TT_GDN_TILE_DIM;
    float *echo = calloc(conv_dim, sizeof(*echo));
    uint32_t index;
    uint32_t tap;
    uint32_t stream_bad = 0;
    uint32_t weight_bad = 0;

    if (!echo)
        return;
    if (gdn_row0_move(device, &device->gdn_stream_in, tiles, echo, true) == 0) {
        for (index = 0; index < conv_dim; index++)
            if (echo[index] != qkv[index])
                stream_bad++;
    } else {
        stream_bad = conv_dim;
    }
    fprintf(stderr, "pn_tt: gdn SELFCHECK stream row 0 — %u of %u channels differ; "
                    "in[0..3] %g %g %g %g  echo[0..3] %g %g %g %g\n",
            stream_bad, conv_dim, (double)qkv[0], (double)qkv[1], (double)qkv[2],
            (double)qkv[3], (double)echo[0], (double)echo[1], (double)echo[2],
            (double)echo[3]);

    /* EVERY tap plane, not just tap 0. At position 0 the window is zero and the only weight
     * the card touches is tap `taps-1`, so a check that stops at plane 0 leaves the one plane
     * that matters unexamined. */
    for (tap = 0; weight && tap < taps; tap++) {
        const pn_tt_gdn_buffer plane = {
            weight_buffer->bank,
            weight_buffer->address + tap * tiles * PN_TT_GDN_TILE_BYTES
        };

        weight_bad = 0;
        if (gdn_row0_move(device, &plane, tiles, echo, true) != 0) {
            weight_bad = conv_dim;
        } else {
            for (index = 0; index < conv_dim; index++)
                if (echo[index] != weight[(size_t)index * taps + tap])
                    weight_bad++;
        }
        fprintf(stderr, "pn_tt: gdn SELFCHECK conv weight tap-%u plane — %u of %u channels "
                        "differ; oracle[0..2] %g %g %g  card[0..2] %g %g %g\n",
                tap, weight_bad, conv_dim, (double)weight[tap], (double)weight[taps + tap],
                (double)weight[2u * taps + tap], (double)echo[0], (double)echo[1],
                (double)echo[2]);
    }
    free(echo);
}

/**
 * Read tile 0 of a CONV operand back WHOLE — all 32 logical rows — and report row 0 against
 * the 31 pad rows.
 *
 * The whole conv path rests on "row 0 is live and rows 1-31 stay zero through the
 * convolution and the roll, because SiLU(0) = 0". Nothing on the card enforces that, and if
 * it is false the operand the NEXT token convolves is not the one this one wrote. Reading a
 * whole tile is the only way to see it: the row-0 mover cannot, by construction.
 */
static void gdn_self_check_out(pn_tt_device *device, const pn_tt_gdn_buffer *buffer,
                               uint32_t first_tile, const char *name)
{
    float tile[PN_TT_GDN_TILE_DIM * PN_TT_GDN_TILE_DIM];
    uint32_t face;
    uint32_t nonzero[4] = { 0u, 0u, 0u, 0u };
    uint32_t negative_zero = 0;
    uint32_t worst_bits = 0;

    if (pn_tt_read_dram(device, buffer->bank,
                        buffer->address + first_tile * PN_TT_GDN_TILE_BYTES, tile,
                        sizeof(tile)) != 0)
        return;
    /* Face-ordered 2x2 grid of 16x16: face f is elements [256*f, +256). Row 0 lives in the
     * first 16 of face 0 and the first 16 of face 1; everything else is pad.
     *
     * ASKED OF THE BITS, NOT OF `!= 0.0f`. Two reasons, and the second is the one that
     * matters: this tree compiles with `-ffast-math`, so the compiler is licensed to reason
     * about float comparisons; and `-0.0f != 0.0f` is FALSE, so a pad row of negative zeros
     * would read as clean under a value test. Any proposal to SKIP work on the pad rows —
     * a row-0-only SFPU pass over the CONV SiLU is the live one — rests on this being an
     * exact bit pattern, so it is measured as one. `-0.0` is reported separately rather than
     * folded in: SiLU(-0.0) is -0.0, so it is bit-safe for that lever but is still not the
     * zero the strip invariant claims. */
    for (face = 0; face < 4u; face++) {
        uint32_t index;

        for (index = (face < 2u) ? 16u : 0u; index < 256u; index++) {
            uint32_t bits;

            memcpy(&bits, &tile[face * 256u + index], sizeof(bits));
            if (bits == 0u)
                continue;
            if (bits == 0x80000000u) {
                negative_zero++;
                continue;
            }
            nonzero[face]++;
            if (bits > worst_bits)
                worst_bits = bits;
        }
    }
    fprintf(stderr, "pn_tt: gdn SELFCHECK %s tile %u — pad non-zero BITS by face %u/240 "
                    "%u/240 %u/256 %u/256, %u negative zeros, worst bits %08x; row 0 [0..3] "
                    "%g %g %g %g; pad [16..18] %g %g %g\n",
            name, first_tile, nonzero[0], nonzero[1], nonzero[2], nonzero[3], negative_zero,
            worst_bits, (double)tile[0], (double)tile[1], (double)tile[2], (double)tile[3],
            (double)tile[16], (double)tile[17], (double)tile[18]);
}

/**
 * At position 0 the window is zero, so `conv_out = SiLU(qkv * w[.][taps-1])` — ONE product
 * and one SFPU call, with three additions of zero between them. Scoring the card against BOTH
 * the raw product and the SiLU of it says which of the two the card actually did, and how
 * well: a card that matches the product is a kernel missing its SiLU, and a card that matches
 * neither has an arithmetic problem in the multiply.
 */
static void gdn_self_check_arithmetic(pn_tt_device *device, const float *qkv,
                                      const float *weight)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t taps = device->gdn_shape.conv_taps;
    float *echo = calloc(conv_dim, sizeof(*echo));
    double worst_product = 0.0;
    double worst_silu = 0.0;
    double scale = 0.0;
    double worst_x = 0.0;
    double worst_want = 0.0;
    double worst_got = 0.0;
    uint32_t worst_at = 0;
    uint32_t index;

    if (!echo || !weight)
        goto done;
    if (gdn_row0_move(device, &device->gdn_stream_out, conv_dim / PN_TT_GDN_TILE_DIM, echo,
                      true) != 0)
        goto done;
    for (index = 0; index < conv_dim; index++) {
        const double product = (double)qkv[index] * (double)weight[(size_t)index * taps +
                                                                   (taps - 1u)];
        const double silu = product / (1.0 + exp(-product));
        const double product_gap = fabs((double)echo[index] - product);
        const double silu_gap = fabs((double)echo[index] - silu);

        if (fabs(silu) > scale)
            scale = fabs(silu);
        if (product_gap > worst_product)
            worst_product = product_gap;
        if (silu_gap > worst_silu) {
            worst_silu = silu_gap;
            worst_at = index;
            worst_x = product;
            worst_want = silu;
            worst_got = (double)echo[index];
        }
    }
    fprintf(stderr, "pn_tt: gdn SELFCHECK position-0 arithmetic — max-norm relative against "
                    "the RAW PRODUCT %.3e, against SiLU(product) %.3e (tensor max %.4g)\n",
            scale > 0.0 ? worst_product / scale : 0.0, scale > 0.0 ? worst_silu / scale : 0.0,
            scale);
    fprintf(stderr, "pn_tt: gdn SELFCHECK   worst SiLU channel %u: x %.6g  want %.6g  got "
                    "%.6g  (sigmoid want %.6g got %.6g)\n",
            worst_at, worst_x, worst_want, worst_got,
            worst_x != 0.0 ? worst_want / worst_x : 0.0,
            worst_x != 0.0 ? worst_got / worst_x : 0.0);
done:
    free(echo);
}

/** Read the row-0 strip of a CONV plane back and compare it against a host array. */
static void gdn_self_check_plane(pn_tt_device *device, const pn_tt_gdn_buffer *buffer,
                                 uint32_t first_tile, const float *expect, const char *name)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const pn_tt_gdn_buffer plane = { buffer->bank,
                                     buffer->address + first_tile * PN_TT_GDN_TILE_BYTES };
    float *echo = calloc(conv_dim, sizeof(*echo));
    uint32_t index;
    uint32_t bad = 0;

    if (!echo)
        return;
    if (gdn_row0_move(device, &plane, conv_dim / PN_TT_GDN_TILE_DIM, echo, true) == 0) {
        for (index = 0; index < conv_dim; index++)
            if (echo[index] != expect[index])
                bad++;
    } else {
        bad = conv_dim;
    }
    fprintf(stderr, "pn_tt: gdn SELFCHECK %s — %u of %u channels differ; expect[0..3] %g %g "
                    "%g %g  card[0..3] %g %g %g %g\n",
            name, bad, conv_dim, (double)expect[0], (double)expect[1], (double)expect[2],
            (double)expect[3], (double)echo[0], (double)echo[1], (double)echo[2],
            (double)echo[3]);
    free(echo);
}

/**
 * NAME THE TENSOR THE NEWEST WINDOW PLANE HOLDS, by searching CARD DRAM rather than the host.
 *
 * The earlier searches asked the host's own arrays — this token's qkv, every SSM layer's qkv
 * from the PREVIOUS token, every tile rotation of the stream — and all three came back
 * negative. What they could not ask is "which BUFFER ON THE CARD does this plane equal", and
 * that is the question a wrong DRAM address answers. Every candidate below is read back off
 * the card at the moment of the miss, so a match names a buffer and an offset rather than a
 * host tensor that may never have been on the card at all.
 *
 * The ELEMENT shift is the second half. `gdn_self_check_window()` searches TILE rotations, 32
 * channels at a step, because a circular buffer out of step by whole pages produces exactly
 * that. A face-row or a within-tile offset is a different bug and lands between those steps.
 */
static void gdn_name_plane(pn_tt_device *device, const float *plane, const float *stream)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t taps = device->gdn_shape.conv_taps;
    const uint32_t tiles = conv_dim / PN_TT_GDN_TILE_DIM;
    float *echo = calloc(conv_dim, sizeof(*echo));
    uint32_t best_shift = 0;
    uint32_t best_off = conv_dim + 1u;
    uint32_t layer;
    uint32_t index;
    uint32_t shift;

    if (!echo)
        return;

    /* EVERY element shift, not every tile shift. */
    for (shift = 0; shift < conv_dim; shift++) {
        uint32_t off = 0;

        for (index = 0; index < conv_dim; index++)
            if (plane[index] != stream[(index + shift) % conv_dim])
                off++;
        if (off < best_off) {
            best_off = off;
            best_shift = shift;
        }
    }
    fprintf(stderr, "pn_tt: gdn NAME   best ELEMENT shift against the stream: %u channels, "
                    "%u of %u differ\n", best_shift, best_off, conv_dim);

    /* Every OTHER buffer this family owns, at the offset a wrong address would land on. Each
     * is one row-0 strip of `tiles` tiles, the same shape the plane is. */
    {
        struct {
            const char *name;
            const pn_tt_gdn_buffer *buffer;
            uint32_t first_tile;
        } candidate[6];
        unsigned count = 0;
        unsigned which;

        candidate[count].name = "conv_out (activation out)";
        candidate[count].buffer = &device->gdn_stream_out;
        candidate[count++].first_tile = 0u;
        candidate[count].name = "delta input q|k|v";
        candidate[count].buffer = &device->gdn_stream_out;
        candidate[count++].first_tile = 0u;
        candidate[count].name = "layer 0 recurrent state, tiles 0..";
        candidate[count].buffer = &device->gdn_region[0].state;
        candidate[count++].first_tile = 0u;
        candidate[count].name = "layer 0 recurrent state, tiles 256..";
        candidate[count].buffer = &device->gdn_region[0].state;
        candidate[count++].first_tile = tiles;
        for (which = 0; which < count; which++) {
            uint32_t off = 0;

            if (gdn_row0_move(device, candidate[which].buffer, tiles, echo, true) != 0)
                continue;
            for (index = 0; index < conv_dim; index++)
                if (plane[index] != echo[index])
                    off++;
            fprintf(stderr, "pn_tt: gdn NAME   vs %s (tile %u): %u of %u differ; card[0..2] "
                            "%g %g %g\n",
                    candidate[which].name, candidate[which].first_tile, off, conv_dim,
                    (double)echo[0], (double)echo[1], (double)echo[2]);
        }
    }

    /* Every layer's conv weight tap plane and every layer's window plane. A wrong plane index
     * or a wrong layer stride lands in one of these and nowhere else. */
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++) {
        uint32_t tap;

        for (tap = 0; tap < taps; tap++) {
            const pn_tt_gdn_buffer view = {
                device->gdn_region[layer].weight.bank,
                device->gdn_region[layer].weight.address + tap * tiles * PN_TT_GDN_TILE_BYTES
            };
            uint32_t off = 0;

            if (gdn_row0_move(device, &view, tiles, echo, true) != 0)
                continue;
            for (index = 0; index < conv_dim; index++)
                if (plane[index] != echo[index])
                    off++;
            if (off * 100u < conv_dim)
                fprintf(stderr, "pn_tt: gdn NAME   IS layer %u conv WEIGHT tap plane %u "
                                "(%u of %u differ)\n", layer, tap, off, conv_dim);
        }
        for (tap = 0; tap + 1u < taps; tap++) {
            const pn_tt_gdn_buffer view = {
                device->gdn_region[layer].window.bank,
                device->gdn_region[layer].window.address + tap * tiles * PN_TT_GDN_TILE_BYTES
            };
            uint32_t off = 0;

            if (layer == 0u && tap + 2u == taps)
                continue; /* the plane itself */
            if (gdn_row0_move(device, &view, tiles, echo, true) != 0)
                continue;
            for (index = 0; index < conv_dim; index++)
                if (plane[index] != echo[index])
                    off++;
            if (off * 100u < conv_dim)
                fprintf(stderr, "pn_tt: gdn NAME   IS layer %u WINDOW plane %u (%u of %u "
                                "differ)\n", layer, tap, off, conv_dim);
        }
    }

    /* The bit patterns, because a decimal cannot tell a shifted exponent from a wrong tile. */
    {
        union { float f; uint32_t u; } value;

        fprintf(stderr, "pn_tt: gdn NAME   plane[0..31] hex");
        for (index = 0; index < 32u; index++) {
            value.f = plane[index];
            fprintf(stderr, " %08x", value.u);
        }
        fprintf(stderr, "\npn_tt: gdn NAME   plane[8160..8191] hex");
        for (index = conv_dim - 32u; index < conv_dim; index++) {
            value.f = plane[index];
            fprintf(stderr, " %08x", value.u);
        }
        fprintf(stderr, "\npn_tt: gdn NAME   stream[0..31] hex");
        for (index = 0; index < 32u; index++) {
            value.f = stream[index];
            fprintf(stderr, " %08x", value.u);
        }
        fprintf(stderr, "\n");
        /* THE CHANNELS THAT DID MATCH. Five of 8192 agreeing with the roll is not noise at
         * bf16 spacing; WHERE they are says whether the plane is partly right (a short
         * transfer) or right by coincidence (a different tensor that happens to collide). */
        {
            unsigned found = 0;

            fprintf(stderr, "pn_tt: gdn NAME   channels that DO match the stream:");
            for (index = 0; index < conv_dim && found < 16u; index++)
                if (plane[index] == stream[index]) {
                    fprintf(stderr, " %u", index);
                    found++;
                }
            fprintf(stderr, "%s\n", found ? "" : " none");
        }
    }
    free(echo);
}

/**
 * SCORE THE ROLLED WINDOW ACROSS TOKENS, which position 0 cannot.
 *
 * `gdn_self_check_plane()` compares the newest plane against this token's qkv, and at
 * position 0 that is the whole of the window — every other plane is zero, so a clean result
 * says nothing about the ROLL. The roll is `window_new[t] = window_old[t+1]`, and it is the
 * only part of PHASE CONV that carries state from one token to the next.
 *
 * The shadow here is rolled the way the contract says, from the SAME qkv the card was given,
 * so a disagreement is the card's roll and not a drifting input. Layer 0 only: one layer's
 * roll is the same code as every layer's, and 24 shadows would be 786 KiB of host memory for
 * no extra statement.
 */
static void gdn_self_check_window(pn_tt_device *device, const pn_tt_gdn_buffer *window,
                                  const float *qkv)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t taps = device->gdn_shape.conv_taps;
    const uint32_t tiles = conv_dim / PN_TT_GDN_TILE_DIM;
    float *echo = calloc(conv_dim, sizeof(*echo));
    uint32_t plane;

    if (!echo)
        return;
    if (!device->gdn_window_shadow)
        device->gdn_window_shadow = calloc((size_t)(taps - 1u) * conv_dim, sizeof(float));
    if (!device->gdn_window_shadow) {
        free(echo);
        return;
    }
    /* THE NEWEST PLANE IS READ BEFORE ANY OTHER TRAFFIC THIS FUNCTION MAKES. Every readback
     * here goes through `pn_tt_matvec_flush()`, so a plane read after three other reads cannot
     * say whether the descriptor wrote it wrong or something after the descriptor wrote over
     * it. Read first, and the answer is the descriptor's. The whole TILE goes with it: rows
     * 1-31 of a conv operand are zero by construction, so a plane whose pad is non-zero did
     * not come from the stream at all and the search below is looking for a dense tensor. */
    gdn_self_check_out(device, window, (taps - 2u) * tiles, "window newest plane, FIRST read");
    for (plane = 0; plane + 2u < taps; plane++)
        memcpy(device->gdn_window_shadow + (size_t)plane * conv_dim,
               device->gdn_window_shadow + (size_t)(plane + 1u) * conv_dim,
               (size_t)conv_dim * sizeof(float));
    memcpy(device->gdn_window_shadow + (size_t)(taps - 2u) * conv_dim, qkv,
           (size_t)conv_dim * sizeof(float));

    /* NAME the tensor, do not describe it. The stream and the window are the two operands the
     * copy is between, and if the plane is neither then the only other thing that has been
     * through this buffer is ANOTHER LAYER's qkv — so every layer's is kept and searched. */
    {
        float *stream = calloc(conv_dim, sizeof(*stream));

        if (stream && gdn_row0_move(device, &device->gdn_stream_in, tiles, stream, true) == 0) {
            for (plane = 0; plane + 1u < taps; plane++) {
                const pn_tt_gdn_buffer view = {
                    window->bank, window->address + plane * tiles * PN_TT_GDN_TILE_BYTES
                };
                const float *want = device->gdn_window_shadow + (size_t)plane * conv_dim;
                uint32_t bad = 0;
                uint32_t vs_stream = 0;
                uint32_t index;

                if (gdn_row0_move(device, &view, tiles, echo, true) != 0) {
                    bad = conv_dim;
                    vs_stream = conv_dim;
                } else {
                    for (index = 0; index < conv_dim; index++) {
                        if (echo[index] != want[index])
                            bad++;
                        if (echo[index] != stream[index])
                            vs_stream++;
                    }
                }
                fprintf(stderr, "pn_tt: gdn SELFCHECK window token %u plane %u — %u of %u "
                                "differ from the roll, %u from the card's own qkv stream; "
                                "want[0..2] %g %g %g  card[0..2] %g %g %g\n",
                        device->gdn_window_checks, plane, bad, conv_dim, vs_stream,
                        (double)want[0], (double)want[1], (double)want[2], (double)echo[0],
                        (double)echo[1], (double)echo[2]);
                if (bad != 0u) {
                    /* IS IT THE SAME STREAM, ROTATED? A circular buffer whose producer and
                     * consumer are out of step by `s` pages delivers exactly that — the right
                     * tiles in the wrong order — and it is indistinguishable from garbage in a
                     * per-channel difference count. One tile is 32 channels. */
                    const uint32_t tiles_of = conv_dim / PN_TT_GDN_TILE_DIM;
                    uint32_t shift;

                    for (shift = 1u; shift < tiles_of; shift++) {
                        uint32_t off = 0;

                        for (index = 0; index < conv_dim; index++) {
                            const uint32_t from = (index + shift * PN_TT_GDN_TILE_DIM) % conv_dim;

                            if (echo[index] != stream[from])
                                off++;
                        }
                        if (off * 100u < conv_dim) {
                            fprintf(stderr, "pn_tt: gdn SELFCHECK   plane %u IS the qkv stream "
                                            "ROTATED BY %u TILES (%u of %u channels differ)\n",
                                    plane, shift, off, conv_dim);
                            break;
                        }
                    }
                    if (shift >= tiles_of)
                        fprintf(stderr, "pn_tt: gdn SELFCHECK   plane %u is not the qkv stream "
                                        "at ANY tile rotation\n", plane);
                    /* The NEWEST plane is the one this defect is about: the older planes are
                     * byte-exact and a search of them would only re-report the roll. */
                    if (plane + 2u == taps)
                        gdn_name_plane(device, echo, stream);
                }
            }
        }
        free(stream);
    }
    device->gdn_window_checks++;
    free(echo);
}

/**
 * STAGE AN INJECTIVE STREAM AND READ THE NEWEST PLANE BACK, so the transform is a MAP rather
 * than a verdict.
 *
 * Every probe so far can only say the plane is not qkv. With `qkv[c] = c` the plane names its
 * own source: `plane[c] == c` is a copy, `plane[c] == c + k` is a shift, `plane[c] == c % 256`
 * is a buffer of 8 pages read 32 times over, and a value outside `[0, conv_dim)` did not come
 * from the stream at all. It costs one extra CONV descriptor and it POISONS the window and
 * `conv_out` for that layer, which is why it is behind its own variable and runs once.
 */
static void gdn_ramp_probe(pn_tt_device *device, uint32_t layer,
                           const ft_tt_op_gdn_request *template_request)
{
    const uint32_t conv_dim = device->gdn_shape.conv_dim;
    const uint32_t tiles = conv_dim / PN_TT_GDN_TILE_DIM;
    const uint32_t plane_tile = (device->gdn_shape.conv_taps - 2u) * tiles;
    const pn_tt_gdn_buffer plane = {
        device->gdn_region[layer].window.bank,
        device->gdn_region[layer].window.address + plane_tile * PN_TT_GDN_TILE_BYTES
    };
    ft_tt_op_gdn_request request = *template_request;
    float *ramp = calloc(conv_dim, sizeof(*ramp));
    float *echo = calloc(conv_dim, sizeof(*echo));
    uint32_t identity = 0;
    uint32_t in_range = 0;
    uint32_t index;

    if (!ramp || !echo)
        goto done;
    for (index = 0; index < conv_dim; index++)
        ramp[index] = (float)index;
    if (gdn_row0_move(device, &device->gdn_stream_in, tiles, ramp, false) != 0)
        goto done;
    if (gdn_submit(device, &request) != 0)
        goto done;
    if (gdn_row0_move(device, &plane, tiles, echo, true) != 0)
        goto done;

    for (index = 0; index < conv_dim; index++) {
        if (echo[index] == (float)index)
            identity++;
        if (echo[index] >= 0.0f && echo[index] < (float)conv_dim &&
            echo[index] == (float)(int)echo[index])
            in_range++;
    }
    fprintf(stderr, "pn_tt: gdn RAMP layer %u — %u of %u channels are the IDENTITY, %u are a "
                    "whole number inside [0, %u)\n",
            layer, identity, conv_dim, in_range, conv_dim);
    for (index = 0; index < 64u; index += 16u) {
        unsigned step;

        fprintf(stderr, "pn_tt: gdn RAMP   plane[%u..%u]", index, index + 15u);
        for (step = 0; step < 16u; step++)
            fprintf(stderr, " %g", (double)echo[index + step]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "pn_tt: gdn RAMP   plane[8160..8175]");
    for (index = conv_dim - 32u; index < conv_dim - 16u; index++)
        fprintf(stderr, " %g", (double)echo[index]);
    fprintf(stderr, "\n");
    /* THE WHOLE TILE MAP. Output tile `t` holds column 0 of source tile `s`, so one number a
     * tile says where every tile came from — and the shape of `s(t) - t` is the difference
     * between a constant page offset, a drift and a wrap. */
    for (index = 0; index < tiles; index += 32u) {
        unsigned step;

        fprintf(stderr, "pn_tt: gdn RAMP   source tile for out tiles %3u..%3u:", index,
                index + 31u);
        for (step = 0; step < 32u; step++)
            fprintf(stderr, " %d", (int)(echo[(index + step) * PN_TT_GDN_TILE_DIM] / 32.0f));
        fprintf(stderr, "\n");
    }
done:
    free(ramp);
    free(echo);
}

/* ---------------------------------------------------------------------------
 * THE CONV DATA ARM — three stopwatches, no gate
 * ---------------------------------------------------------------------------
 * "CONV is DATA-DEPENDENT — 1222.7 us on real activations, 4571 on a degenerate stream,
 * suspect an SFPU slow path on out-of-range SiLU input" stood as an open item with no
 * harness behind either half. The suspicion is refuted at the SOURCE — the SiLU stream is
 * straight-line, every conditional in it is SFPU condition-code-stack predication
 * (`ft_tensix_sfpu.h`'s `sfppushc`/`sfpxvif`/`sfpcompc`/`sfppopc`), so disabled lanes still
 * ISSUE and only write-back is masked; `exp` is a fixed degree-5 Horner and `recip` is
 * exactly two Newton steps with the second PREDICATED, not branched. These arms are what
 * settle it on silicon.
 *
 * THREE NAMES, NOT ONE KNOB WITH A VALUE, because `pn_tt_red()` is an exact match on one
 * name: an arm therefore cannot be selected by a value nobody parses, and each arm names
 * itself in the log.
 *
 *   gdn_conv_row0_uniform   the shipped OCCUPANCY (row 0 live, rows 1-31 zero), benign
 *                           magnitudes — the control for the fill machinery itself
 *   gdn_conv_full_uniform   THE SHARP ONE. The same row-0 values, plus all 31 pad rows
 *                           filled: **32x the non-zero SFPU operands at identical shape,
 *                           identical traffic and an identical descriptor.**
 *   gdn_conv_out_of_range   row-0 occupancy, values the hypothesis names: +-1e30 (SiLU
 *                           saturates), +-1e-40 (DENORMAL), and the 0..8191 ramp
 *
 * Deliberately NOT infinities or NaNs: those would propagate into PHASE DELTA's recurrence
 * and the arms would differ in what the DELTA descriptor beside them does, which is a second
 * variable. Saturating and denormal operands are what the suspicion is actually about.
 *
 * Each produces a WRONG ANSWER by construction. The timing is what they are for.
 */
typedef struct pn_tt_gdn_synthetic {
    const char *name; /**< NULL when no arm is selected, which is the control */
    bool full;        /**< fill all 31 pad rows as well as logical row 0 */
    bool out_of_range; /**< +-1e30 and +-1e-40 instead of uniform [-1, 1] */
} pn_tt_gdn_synthetic;

/**
 * Which arm, as FLAGS rather than a name to sniff.
 *
 * The first version keyed `full` off `arm[0] == 'f'` and `out_of_range` off `'o'`, which is
 * correct for exactly the three names it had and silently wrong for the fourth: a
 * `full_out_of_range` arm begins with 'f', so it would have filled the pad and then drawn
 * every value from the BENIGN generator — a maximal-stress arm that ran no stress and read
 * as a null result. The flags are what the fill actually needs, so they are what it is given.
 */
static pn_tt_gdn_synthetic gdn_synthetic_arm(void)
{
    static const pn_tt_gdn_synthetic none = { NULL, false, false };
    static const pn_tt_gdn_synthetic row0 = { "row0_uniform", false, false };
    static const pn_tt_gdn_synthetic full = { "full_uniform", true, false };
    static const pn_tt_gdn_synthetic wild = { "out_of_range", false, true };
    /* THE MAXIMAL STRESS: 992 out-of-range operands a tile instead of 32, so the SFPU sees
     * a saturating or denormal value in nearly every lane of every face. */
    static const pn_tt_gdn_synthetic full_wild = { "full_out_of_range", true, true };

    if (pn_tt_red("gdn_conv_row0_uniform"))
        return row0;
    if (pn_tt_red("gdn_conv_full_uniform"))
        return full;
    if (pn_tt_red("gdn_conv_out_of_range"))
        return wild;
    if (pn_tt_red("gdn_conv_full_out_of_range"))
        return full_wild;
    return none;
}

/**
 * One 32x32 fp32 tile of the arm's stream, FACE-ORDERED.
 *
 * A tile is four 16x16 faces in row-major order of the 2x2 face grid, so LOGICAL ROW 0 IS
 * NOT CONTIGUOUS: columns 0-15 are elements 0-15 of face 0 and columns 16-31 are elements
 * 256-271 (`ft_tt_tile_row0.h`). A contiguous 32-float write at offset 0 would land on tile
 * ROWS 0 AND 1, columns 0-15 — a stream that is neither the shipped occupancy nor a full
 * tile, and every figure taken against it would be of a third thing nobody asked about.
 *
 * THE THIRTY-TWO LIVE VALUES ARE DRAWN FIRST AND ARE THE SAME IN EVERY ARM AT THE SAME
 * `full`, so `row0_uniform` and `full_uniform` differ in the PAD AND NOTHING ELSE.
 */
static float gdn_synthetic_value(const pn_tt_gdn_synthetic *arm, uint32_t *state,
                                 uint32_t channel)
{
    /* +-1e30 (SiLU saturates), +-1e-40 (DENORMAL in fp32). No infinity, no NaN — see the
     * block above `gdn_synthetic_arm()` for why those would add a second variable. */
    static const float wild[4] = { 1e30f, -1e30f, 1e-40f, -1e-40f };

    *state = *state * 1664525u + 1013904223u;
    if (arm->out_of_range)
        return ((*state >> 8) & 4u) ? (float)channel : wild[(*state >> 8) & 3u];
    return (float)((double)(*state >> 8) / 8388608.0 - 1.0);
}

static void gdn_synthetic_tile(const pn_tt_gdn_synthetic *arm, uint32_t tile_index,
                               float *tile)
{
    uint32_t state = 0x1234abcdu + tile_index * 2654435761u;
    uint32_t column;
    uint32_t index;

    memset(tile, 0, PN_TT_GDN_TILE_BYTES);
    for (column = 0; column < PN_TT_GDN_TILE_DIM; column++) {
        const uint32_t at = (column < 16u) ? column : (256u + column - 16u);

        tile[at] = gdn_synthetic_value(arm, &state,
                                       tile_index * PN_TT_GDN_TILE_DIM + column);
    }
    if (!arm->full)
        return;
    /* Every element that is NOT logical row 0, from the SAME generator the live row used —
     * so `full_out_of_range` really is 992 pathological operands a tile and not 992 benign
     * ones wearing the arm's name. */
    for (index = 0; index < PN_TT_GDN_TILE_DIM * PN_TT_GDN_TILE_DIM; index++) {
        const bool live = (index < 16u) || (index >= 256u && index < 272u);

        if (live)
            continue;
        tile[index] = gdn_synthetic_value(arm, &state, index);
    }
}

/**
 * Overwrite the qkv stream in place, before the CONV descriptor reads it.
 *
 * READ BACK AND SAY SO, ONCE. An arm that silently did not fire reads exactly like an arm
 * that fired and cost nothing — the whole point of these three — so the first descriptor of
 * a run echoes tile 0 out of card DRAM and prints the live row AND the pad, and the control
 * prints `arm=none` beside the real activations. A reader can then see WHICH stream produced
 * the microsecond figure rather than trusting the environment.
 *
 * Writes only into `gdn_stream_in`, a buffer this file allocated and this family reads every
 * token, through the ordinary bulk-write path. No new endpoint, no window aim, no tile.
 */
static int gdn_conv_synthetic(pn_tt_device *device, const pn_tt_gdn_synthetic *arm)
{
    const uint32_t tiles = device->gdn_shape.conv_dim / PN_TT_GDN_TILE_DIM;
    float *tile = malloc(PN_TT_GDN_TILE_BYTES);
    uint32_t index;
    int rc = 0;

    if (!tile)
        return -1;
    /* THE WHOLE TILE IS WRITTEN IN EVERY ARM, INCLUDING THE ROW-0 ONES. Writing only the two
     * 64 B row-0 pieces would leave the pad holding the previous descriptor's values, so
     * `row0_uniform` and `full_uniform` would differ in the number of host writes AND in
     * whether the pad was defined — two variables where the arm is asking about one. Same
     * bytes, same call count, same descriptor; only the CONTENT of the pad moves. */
    for (index = 0; index < tiles && rc == 0; index++) {
        gdn_synthetic_tile(arm, index, tile);
        rc = pn_tt_write_dram(device, device->gdn_stream_in.bank,
                              device->gdn_stream_in.address +
                                  (size_t)index * PN_TT_GDN_TILE_BYTES,
                              tile, PN_TT_GDN_TILE_BYTES);
    }
    free(tile);
    return rc;
}

/** Echo tile 0 of the stream the CONV descriptor is about to read, live row and pad. */
static void gdn_conv_stream_echo(pn_tt_device *device, const char *arm)
{
    float tile[PN_TT_GDN_TILE_DIM * PN_TT_GDN_TILE_DIM];
    uint32_t pad_nonzero = 0;
    uint32_t denormals = 0;
    uint32_t huge = 0;
    double smallest = 0.0;
    double largest = 0.0;
    uint32_t column;
    uint32_t index;

    if (device->gdn_synthetic_reported)
        return;
    device->gdn_synthetic_reported = true;
    if (pn_tt_read_dram(device, device->gdn_stream_in.bank, device->gdn_stream_in.address,
                        tile, sizeof(tile)) != 0) {
        fprintf(stderr, "pn_tt: gdn CONV STREAM arm=%s — readback FAILED\n",
                arm ? arm : "none");
        return;
    }
    /* Face-ordered: row 0 is the first 16 of face 0 and the first 16 of face 1. */
    for (index = 0; index < PN_TT_GDN_TILE_DIM * PN_TT_GDN_TILE_DIM; index++) {
        const uint32_t face = index / 256u;
        const uint32_t within = index % 256u;

        if (face < 2u && within < 16u)
            continue;
        if (tile[index] != 0.0f)
            pad_nonzero++;
    }
    /* THE RANGE OF THE LIVE ROW, ASKED OF THE EXPONENT BITS. Four sample values cannot say
     * whether an out-of-range arm's +-1e30 and +-1e-40 reached the card at all — the first
     * four of the ramp branch are 0 1 2 3 and read exactly like a benign stream. A DENORMAL
     * is what the SFPU hypothesis is about, so it is counted rather than inferred. */
    for (column = 0; column < PN_TT_GDN_TILE_DIM; column++) {
        const uint32_t at = (column < 16u) ? column : (256u + column - 16u);
        const double magnitude = tile[at] < 0.0f ? -(double)tile[at] : (double)tile[at];
        uint32_t bits;

        memcpy(&bits, &tile[at], sizeof(bits));
        if (((bits >> 23) & 0xFFu) == 0u && (bits & 0x7FFFFFu) != 0u)
            denormals++;
        if (((bits >> 23) & 0xFFu) >= 0xC0u)
            huge++;
        if (magnitude > largest)
            largest = magnitude;
        if (magnitude != 0.0 && (smallest == 0.0 || magnitude < smallest))
            smallest = magnitude;
    }
    fprintf(stderr, "pn_tt: gdn CONV STREAM arm=%s — tile 0 row 0 [0..3] %g %g %g %g; "
                    "live |x| in [%g, %g], %u denormals, %u over 2^65; "
                    "pad %u of 992 non-zero; pad[16..18] %g %g %g\n",
            arm ? arm : "none", (double)tile[0], (double)tile[1], (double)tile[2],
            (double)tile[3], smallest, largest, denormals, huge, pad_nonzero,
            (double)tile[16], (double)tile[17], (double)tile[18]);
}

/** The PHASE CONV descriptor, whatever put the qkv stream where it is. */
static int gdn_conv_descriptor(pn_tt_device *device, uint32_t layer,
                               ft_tt_op_gdn_request *out_request)
{
    const pn_tt_gdn_region *region = &device->gdn_region[layer];
    const pn_tt_gdn_synthetic arm = gdn_synthetic_arm();
    ft_tt_op_gdn_request request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);

    if (arm.name && gdn_conv_synthetic(device, &arm) != 0)
        return -1;
    gdn_conv_stream_echo(device, arm.name);
    /* THE STRIP INVARIANT, ON THE SHIPPED ON-CARD ROUTE, AT MORE THAN ONE LAYER AND MORE
     * THAN ONE TOKEN.
     *
     * `gdn_self_check_out()` above runs only on the HOST-STAGED path, where this file wrote
     * the stream itself and rows 1-31 are zero because it left them zero. The shipped route
     * is different in exactly the way that matters: `render_widen` converts WHOLE TILES from
     * bf16, so rows 1-31 of `gdn_stream_in` are whatever the matmul left in `ssm_qkv` — not
     * anything this file controls. Every proposal to skip work on the pad rows depends on
     * them being an exact zero bit pattern THERE, so it is measured there. Three layers and
     * four tokens: layer 0 of token 0 is the only CONV in a run never preceded by a DELTA,
     * and a defect that only appears once the window has rolled is invisible in token 0. */
    if (getenv("PN_TT_GDN_SELFCHECK") && device->gdn_strip_checks < 12u &&
        (layer == 0u || layer == 11u || layer == 23u)) {
        char label[64];

        device->gdn_strip_checks++;
        snprintf(label, sizeof(label), "STRIP stream_in L%u #%u", layer,
                 device->gdn_strip_checks);
        gdn_self_check_out(device, &device->gdn_stream_in, 0u, label);
    }

    request.activation_in = gdn_operand(device, &device->gdn_stream_in);
    request.activation_out = gdn_operand(device, &device->gdn_stream_out);
    request.conv_weight = gdn_operand(device, &region->weight);
    request.state_conv = gdn_operand(device, &region->window);
    request.row = device->gdn_row;
    if (gdn_state_out_buffer(device, layer, true))
        request.state_out = gdn_operand(device, gdn_state_out_buffer(device, layer, true));
    /* The CONV half of the work knob. `PN_TT_GDN_CONV_DIM=D` convolves the first D channels
     * instead of all 8,192 — fewer tile columns per tap, fewer output tiles, every address
     * still inside the buffers this layer owns because `ftg_conv_plane_tile()` is
     * `tap * conv_tile_cols + column` and both shrink together. A stopwatch, not a gate: the
     * stream past D is left holding the previous token's values. */
    {
        const uint32_t conv_dim = gdn_probe_knob("PN_TT_GDN_CONV_DIM");

        if (conv_dim >= PN_TT_GDN_TILE_DIM && conv_dim < request.conv_dim &&
            (conv_dim % PN_TT_GDN_TILE_DIM) == 0u)
            request.conv_dim = conv_dim;
    }
    *out_request = request;
    return gdn_submit(device, &request);
}

int pn_tt_gdn_conv_oncard(pn_tt_device *device, uint32_t layer)
{
    ft_tt_op_gdn_request request;

    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count)
        return -1;
    return gdn_conv_descriptor(device, layer, &request);
}

int pn_tt_gdn_conv(pn_tt_device *device, uint32_t layer, const float *qkv, float *conv_out)
{
    const uint32_t tiles = device->gdn_shape.conv_dim / PN_TT_GDN_TILE_DIM;
    ft_tt_op_gdn_request request;
    pn_tt_gdn_region *region;

    if (!pn_tt_gdn_is_armed(device) || !qkv || !conv_out ||
        layer >= device->gdn_shape.layer_count)
        return -1;
    region = &device->gdn_region[layer];

    /* The cast is the const the row-0 mover cannot express in one signature; the write arm
     * never touches the buffer. */
    if (gdn_row0_move(device, &device->gdn_stream_in, tiles, (float *)(uintptr_t)qkv, false) != 0)
        return -1;
    if (!device->gdn_self_checked && getenv("PN_TT_GDN_SELFCHECK")) {
        device->gdn_self_checked = true;
        gdn_self_check(device, qkv, device->gdn_weight_oracle, &region->weight);
    }

    if (gdn_conv_descriptor(device, layer, &request) != 0)
        return -1;
    /* WHAT RAN BEFORE THIS DESCRIPTOR, as a knob. The newest window plane is correct at
     * SSM layer 0 of position 0 and wrong at every other CONV in the run, and the one thing
     * that distinguishes that descriptor is that it is the first this residency ever ran —
     * every other CONV follows the previous layer's PHASE DELTA. `PN_TT_GDN_DOUBLE_CONV=1`
     * issues a SECOND, identical CONV so the phase that precedes one is a CONV instead. It
     * rolls the window a second time and therefore breaks the recurrence, which is why it is
     * a probe and not a fallback; `window_new[taps-2] = qkv` is unchanged by the repeat, so
     * the newest plane is still scoreable against the same qkv. */
    if (getenv("PN_TT_GDN_DOUBLE_CONV") && gdn_submit(device, &request) != 0)
        return -1;
    /* The injective-stream map, on the FIRST layer that follows a PHASE DELTA and on that one
     * only — layer 0 of position 0 is the descriptor that already works. */
    if (layer == 1u && !device->gdn_ramp_probed && getenv("PN_TT_GDN_RAMP")) {
        device->gdn_ramp_probed = true;
        gdn_ramp_probe(device, layer, &request);
    }
    /* THE FIRST DRAM ACCESS AFTER THE DESCRIPTOR RETIRES, before any other probe. Every other
     * readback in this file runs `pn_tt_matvec_flush()` first, so a plane read after one of
     * them cannot separate "the descriptor wrote it wrong" from "something the host did next
     * wrote over it". This one can: nothing runs between the retire and this read.
     *
     * **EVERY LAYER, not only layer 0, and that is the point.** The defect reads as
     * "token 0 is clean and token 1 is not", but the ONE thing that distinguishes layer 0 of
     * token 0 from every other CONV in the run is that it is the FIRST DESCRIPTOR THIS
     * RESIDENCY EVER RAN — every other CONV is preceded by the previous layer's PHASE DELTA.
     * A per-layer figure at token 0 separates a token-carried defect from a phase-carried
     * one, and the two have nothing in common. */
    if (device->gdn_window_checks < 3u && getenv("PN_TT_GDN_SELFCHECK")) {
        char label[64];

        snprintf(label, sizeof(label), "newest = qkv, layer %u", layer);
        gdn_self_check_plane(device, &region->window,
                             (device->gdn_shape.conv_taps - 2u) * tiles, qkv, label);
    }
    if (device->gdn_self_checked && !device->gdn_out_checked && getenv("PN_TT_GDN_SELFCHECK")) {
        const uint32_t last_plane = (device->gdn_shape.conv_taps - 2u) * tiles;

        device->gdn_out_checked = true;
        gdn_self_check_out(device, &device->gdn_stream_out, 0u, "conv_out");
        gdn_self_check_out(device, &region->window, last_plane, "window newest plane");
        /* PASS B's newest window plane IS the qkv stream, copied tile for tile through
         * unpack -> datacopy -> pack. It is the ONLY thing in this descriptor whose right
         * answer is known without doing any arithmetic, so it separates a broken Float32
         * DATAPATH from broken Float32 ARITHMETIC. */
        gdn_self_check_plane(device, &region->window, last_plane, qkv, "window newest = qkv");
        gdn_self_check_arithmetic(device, qkv, device->gdn_weight_oracle);
    }
    /* Every layer's qkv, kept so a wrong tensor coming back can be NAMED. Written before the
     * layer-0 check so the history is the PREVIOUS token's for every layer but this one. */
    if (getenv("PN_TT_GDN_SELFCHECK") && device->gdn_window_checks < 3u) {
        if (!device->gdn_qkv_history)
            device->gdn_qkv_history = calloc((size_t)device->gdn_shape.layer_count *
                                                 device->gdn_shape.conv_dim,
                                             sizeof(float));
        if (device->gdn_qkv_history && layer != 0u)
            memcpy(device->gdn_qkv_history + (size_t)layer * device->gdn_shape.conv_dim, qkv,
                   (size_t)device->gdn_shape.conv_dim * sizeof(float));
    }
    /* SSM layer 0's window, for the first few tokens — the ROLL, which position 0 cannot see.
     *
     * The qkv STREAM is echoed at the same moment, because the newest window plane is a copy
     * of it made at the END of the descriptor while the convolution read it at the START. If
     * the stream no longer holds what the host wrote, the roll is innocent and something
     * inside the descriptor is writing over its own input. */
    if (layer == 0u && device->gdn_window_checks < 3u && getenv("PN_TT_GDN_SELFCHECK")) {
        gdn_self_check_plane(device, &device->gdn_stream_in, 0u, qkv,
                             "qkv stream AFTER the descriptor");
        gdn_self_check_window(device, &region->window, qkv);
    }

    return gdn_row0_move(device, &device->gdn_stream_out, tiles, conv_out, true);
}

/** Row 0 of `tiles` tiles starting at `first_tile` of `buffer`, in either direction. */
static int gdn_row0_move_at(pn_tt_device *device, const pn_tt_gdn_buffer *buffer,
                            uint32_t first_tile, uint32_t tiles, float *values, bool reading)
{
    return pn_tt_fp32_row0_move(device, buffer->bank,
                                buffer->address + first_tile * PN_TT_GDN_TILE_BYTES, tiles,
                                values, reading);
}

static int gdn_delta_descriptor(pn_tt_device *device, uint32_t layer);

/** S2c: sixteen per-row gate blocks, CONTIGUOUS at a 64 B stride, allocated once. */
static int gdn_gate_rows_ensure(pn_tt_device *device)
{
    const ft_tt_op_gdn_request delta = gdn_request(device, FT_TT_OP_GDN_PHASE_DELTA);
    const uint32_t stride = (uint32_t)((ft_tt_op_gdn_gate_bytes(&delta) + 63u) & ~(size_t)63u);
    pn_tt_gdn_buffer all;
    uint32_t r;

    if (device->gdn_gate_rows == 16u)
        return 0;
    if (gdn_alloc_in(device, device->gdn_gate.bank, (size_t)stride * 16u, true, &all) != 0)
        return -1;
    for (r = 0; r < 16u; r++) {
        device->gdn_gate_row[r].bank = all.bank;
        device->gdn_gate_row[r].address = all.address + r * stride;
    }
    device->gdn_gate_stride = stride;
    device->gdn_gate_rows = 16u;
    return 0;
}

int pn_tt_gdn_gate_target(pn_tt_device *device, uint32_t rows, uint32_t *bank,
                          uint32_t *address, uint32_t *row_stride)
{
    const pn_tt_gdn_buffer *to = &device->gdn_gate;

    if (!pn_tt_gdn_is_armed(device) || rows == 0u || rows > 16u || !bank || !address ||
        !row_stride)
        return -1;
    /* Where pn_tt_gdn_gate_upload{,_rows}() would have written: the row blocks for a deferred
     * row or for several rows, else the single gate. */
    if (rows > 1u || device->gdn_defer) {
        if (gdn_gate_rows_ensure(device) != 0 || (rows == 1u && device->gdn_row >= 16u))
            return -1;
        to = rows > 1u ? &device->gdn_gate_row[0] : &device->gdn_gate_row[device->gdn_row];
    }
    *bank = to->bank;
    *address = to->address;
    *row_stride = device->gdn_gate_rows ? device->gdn_gate_stride : 256u;
    return 0;
}

int pn_tt_gdn_gate_upload(pn_tt_device *device, const float *decay, const float *beta)
{
    const uint32_t heads = device->gdn_shape.v_heads;
    float gate[2u * PN_TT_GDN_MAX_V_HEADS];
    uint32_t head;

    if (!pn_tt_gdn_is_armed(device) || !decay || !beta)
        return -1;
    /* `decay[v_heads]` then `beta[v_heads]`, flat fp32 — the one operand that is not tiles,
     * and on the on-card path THE ONLY THING THAT STILL CROSSES THE LINK for this block:
     * 256 B up a layer against the 49,408 B the activations used to cost. `decay = exp(g)`
     * is applied by the caller because the SFPU has no `log` and therefore no `pow`.
     *
     * Only the live prefix is written; the rounding the reader's 64 B read needs was staged
     * as zero at allocation.
     *
     * RED PROOF: `gdn_decay_raw` sends the oracle's `g` where the contract says `exp(g)`.
     * NOTHING checks which of the two the card was handed — at position 0 the carried state
     * is zero, so `(a)` multiplies zero by it and the two are INDISTINGUISHABLE. The mutation
     * is therefore only observable from position 1, which is exactly what makes it worth
     * having: it is the one arm that proves the decay word is consumed at all. */
    for (head = 0; head < heads; head++) {
        gate[head] = pn_tt_red("gdn_decay_raw") ? logf(decay[head]) : decay[head];
        gate[heads + head] = beta[head];
    }
    {
        const pn_tt_gdn_buffer *to = &device->gdn_gate;

        /* S2c: a deferred row writes its own block (a queued DELTA still reads its row's). */
        if (device->gdn_defer) {
            if (gdn_gate_rows_ensure(device) != 0 || device->gdn_row >= 16u)
                return -1;
            to = &device->gdn_gate_row[device->gdn_row];
        }
        return pn_tt_write_dram(device, to->bank, to->address, gate,
                                (size_t)2u * heads * sizeof(gate[0]));
    }
}

int pn_tt_gdn_gate_upload_rows(pn_tt_device *device, uint32_t rows, const float *decay,
                               const float *beta)
{
    const uint32_t heads = device ? device->gdn_shape.v_heads : 0u;
    float *block;
    uint32_t r, head, stride_words;
    int rc;

    if (!pn_tt_gdn_is_armed(device) || !decay || !beta || rows == 0u || rows > 16u ||
        gdn_gate_rows_ensure(device) != 0)
        return -1;
    stride_words = device->gdn_gate_stride / 4u;
    block = calloc((size_t)rows * stride_words, sizeof(float));
    if (!block)
        return -1;
    for (r = 0; r < rows; r++)
        for (head = 0; head < heads; head++) {
            block[r * stride_words + head] = pn_tt_red("gdn_decay_raw")
                                                 ? logf(decay[r * heads + head])
                                                 : decay[r * heads + head];
            block[r * stride_words + heads + head] = beta[r * heads + head];
        }
    /* ONE write for every row's block (they are contiguous at the gate stride). */
    rc = pn_tt_write_dram(device, device->gdn_gate_row[0].bank, device->gdn_gate_row[0].address,
                          block, (size_t)rows * device->gdn_gate_stride);
    free(block);
    return rc;
}

void pn_tt_gdn_table_gate(pn_tt_device *device, uint32_t bank, uint32_t address)
{
    if (!device)
        return;
    device->gdn_table_gate.bank = bank;
    device->gdn_table_gate.address = address;
    device->gdn_table_on = address != 0u;
}

/** The table gate, when on, in place of the request's staged gate block. */
static void gdn_table_gate_apply(const pn_tt_device *device, ft_tt_op_gdn_request *request)
{
    if (!device->gdn_table_on)
        return;
    request->gate = gdn_operand(device, &device->gdn_table_gate);
    request->flags |= FT_TT_OP_GDN_FLAG_TABLE;
}

int pn_tt_gdn_delta_descriptor_only(pn_tt_device *device, uint32_t layer)
{
    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count)
        return -1;
    return gdn_delta_descriptor(device, layer);
}

/**
 * MULTI-ROW DELTA (PM 5): verify rows 0..rows-1 of `layer` in ONE descriptor per member, on the
 * gate blocks pn_tt_gdn_gate_upload_rows() staged. Legacy: row r's state goes to snapshot slot r
 * and the slots then trade places with the live state in row order — exactly the bookkeeping
 * the one-row-a-descriptor loop leaves: live = after the last row, slot r = before row r.
 * Compact (pn_tt_gdn_verify_slots.h): rows 0..rows-2 go straight to the slots the per-row walk
 * leaves them in and the last row into the live buffer (S_0's, which the reader reads per head
 * at row 0 only, before any write of that head), with no trades.
 *
 * RED `gdn_rows_state_once`: every row's state goes to slot 0's buffer, so the slots the swaps
 * bring live hold stale state — must DIFFER. RED `gdn_rows_row0`: see below.
 */
int pn_tt_gdn_delta_rows_oncard(pn_tt_device *device, uint32_t layer, uint32_t rows)
{
    ft_tt_op_gdn_request request;
    uint32_t r;

    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count || rows < 2u ||
        rows > FT_TT_OP_GDN_MULTI_ROW_MAX ||
        pn_tt_gdn_verify_slots(device, rows) > device->gdn_snapshot_slots ||
        device->gdn_defer || gdn_gate_rows_ensure(device) != 0)
        return -1;
    request = gdn_request(device, FT_TT_OP_GDN_PHASE_DELTA);
    request.activation_in = gdn_operand(device, &device->gdn_stream_out);
    request.activation_out = gdn_operand(device, &device->gdn_y);
    request.gate = gdn_operand(device, &device->gdn_gate_row[0]);
    gdn_table_gate_apply(device, &request);
    request.state_recurrent = gdn_operand(device, &device->gdn_region[layer].state);
    request.row = 0u;
    request.row_count = rows;
    /* RED `gdn_rows_row0`: the reader stages row 0's q/k/v for every row — must DIFFER. */
    if (pn_tt_red("gdn_rows_row0"))
        request.flags |= FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0;
    for (r = 0; r < rows; r++)
        request.state_rows_out[r] = gdn_operand(
            device, gdn_rows_out_buffer(device, layer, rows,
                                        pn_tt_red("gdn_rows_state_once") ? 0u : r, false));
    request.state_out = request.state_rows_out[0];
    if (gdn_submit(device, &request) != 0)
        return -1;
    for (r = 0; pn_gdn_verify_rows_trade(device->gdn_verify_compact) && r < rows; r++)
        if (pn_tt_gdn_layer_swap_part(device, layer, r, false, true) != 0)
            return -1;
    return 0;
}

int pn_tt_gdn_conv_rows_oncard(pn_tt_device *device, uint32_t layer, uint32_t rows)
{
    const pn_tt_gdn_region *region;
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t r;

    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count || rows < 2u ||
        rows > FT_TT_OP_GDN_MULTI_ROW_MAX ||
        pn_tt_gdn_verify_slots(device, rows) > device->gdn_snapshot_slots ||
        device->gdn_defer || pn_tt_env_u32("PN_TT_GDN_CONV_STRIP", 1u) == 0u)
        return -2;
    region = &device->gdn_region[layer];
    request = gdn_request(device, FT_TT_OP_GDN_PHASE_CONV);
    request.activation_in = gdn_operand(device, &device->gdn_stream_in);
    request.activation_out = gdn_operand(device, &device->gdn_stream_out);
    request.conv_weight = gdn_operand(device, &region->weight);
    request.state_conv = gdn_operand(device, &region->window);
    request.row = 0u;
    request.row_count = rows;
    request.flags |= FT_TT_OP_GDN_FLAG_CONV_STRIP;
    /* RED `gdn_conv_rows_row0`: every row convolves row 0's stream — must DIFFER. */
    if (pn_tt_red("gdn_conv_rows_row0"))
        request.flags |= FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0;
    /* RED `gdn_conv_rows_window_once`: every row's window into row 0's buffer (the rollback
     * then restores the wrong window) — must DIFFER. */
    for (r = 0; r < rows; r++)
        request.state_rows_out[r] = gdn_operand(
            device, gdn_rows_out_buffer(device, layer, rows,
                                        pn_tt_red("gdn_conv_rows_window_once") ? 0u : r, true));
    request.state_out = request.state_rows_out[0];
    /* Refused by the encoder (a shape the family does not run): nothing ran. */
    if (ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK)
        return -2;
    if (gdn_submit(device, &request) != 0)
        return -1;
    for (r = 0; pn_gdn_verify_rows_trade(device->gdn_verify_compact) && r < rows; r++)
        if (pn_tt_gdn_layer_swap_part(device, layer, r, true, false) != 0)
            return -1;
    return 0;
}

/**
 * `q | k | v` and the flat gate block, into the two DRAM buffers the DELTA reader takes.
 *
 * THE THREE STREAMS ARE ROW-0 STRIPS, so this is three `pn_tt_fp32_row0_move()` writes and no
 * tilize buffer at all. It used to build three `[heads, state]` matrices in host scratch and
 * push the whole thing as one bulk write; the strip geometry is what lets PHASE CONV's output
 * BE this buffer, so on the on-card path none of these three writes happens at all.
 */
static int gdn_stage_delta_input(pn_tt_device *device, const float *q, const float *k,
                                 const float *v, const float *decay, const float *beta)
{
    const ftg_shape shape = gdn_shape(device, FT_TT_OP_GDN_PHASE_DELTA);
    /* RED PROOF: q and k swapped. Same shape, same length — (b) and (d) then use q where the
     * oracle uses k and (e) uses k where it uses q, which no shape check can see. */
    const float *first_stream = pn_tt_red("gdn_qk_swap") ? k : q;
    const float *second_stream = pn_tt_red("gdn_qk_swap") ? q : k;

    if (gdn_row0_move_at(device, &device->gdn_stream_out,
                         ftg_delta_stream_base(&shape, (uint32_t)kGdnStreamQ),
                         ftg_delta_stream_tiles(&shape, (uint32_t)kGdnStreamQ),
                         (float *)(uintptr_t)first_stream, false) != 0 ||
        gdn_row0_move_at(device, &device->gdn_stream_out,
                         ftg_delta_stream_base(&shape, (uint32_t)kGdnStreamK),
                         ftg_delta_stream_tiles(&shape, (uint32_t)kGdnStreamK),
                         (float *)(uintptr_t)second_stream, false) != 0 ||
        gdn_row0_move_at(device, &device->gdn_stream_out,
                         ftg_delta_stream_base(&shape, (uint32_t)kGdnStreamV),
                         ftg_delta_stream_tiles(&shape, (uint32_t)kGdnStreamV),
                         (float *)(uintptr_t)v, false) != 0)
        return -1;

    return pn_tt_gdn_gate_upload(device, decay, beta);
}

/**
 * SCORE PHASE DELTA AGAINST ITS CLOSED FORM, at the one position where it has one.
 *
 * At position 0 the carried state is identically zero, so (a) decays nothing, (b)'s `pred` is
 * zero, (c) is `delta_j = v_j * beta`, (d) writes `S[j][i] = k[i]*delta_j` and (e) reads it
 * back as `y_j = scale * delta_j * sum_i k[i]*q[i]`. The whole of the four passes collapses to
 *
 *     y[h][j] = scale * beta[h] * v[h][j] * (k[kh] . q[kh]),   kh = h % k_heads
 *
 * and every term on the right is an operand this function was handed. So a disagreement is
 * localised to a HEAD rather than to a tensor, which is what a per-tensor max-norm figure
 * cannot do: the failure this op actually had was per-head, and it was invisible in a number
 * that maximised over all 32 of them.
 *
 * The dot product is reported for head 0 beside the ratio, because `k . q` is the single
 * scalar the whole head's answer is proportional to — a card that has the right `v` and the
 * wrong dot is a reduce or a staging fault, and a card whose `y` is proportional to the right
 * answer is a scale fault.
 */
static void gdn_self_check_delta(pn_tt_device *device, uint32_t layer, const float *q,
                                 const float *k, const float *v, const float *beta,
                                 const float *y)
{
    const uint32_t state = device->gdn_shape.state_size;
    const uint32_t v_heads = device->gdn_shape.v_heads;
    const uint32_t k_heads = device->gdn_shape.k_heads;
    const double scale = 1.0 / sqrt((double)state);
    /* The per-head detail once, then one line a layer: 24 layers x 32 heads is 768 lines of
     * detail nobody reads, and the question the later layers answer is whether the error is
     * the OP's or one layer's data. */
    const bool detail = device->gdn_delta_checks == 0u;
    double worst_all = 0.0;
    double ratio_sum = 0.0;
    uint32_t ratio_count = 0;
    uint32_t bad_heads = 0;
    uint32_t head;

    if (detail)
        fprintf(stderr, "pn_tt: gdn SELFCHECK DELTA closed form at position 0 — "
                        "y[h][j] = scale * beta[h] * v[h][j] * (k.q), scale %.6g\n", scale);
    for (head = 0; head < v_heads; head++) {
        const float *qh = q + (size_t)(head % k_heads) * state;
        const float *kh = k + (size_t)(head % k_heads) * state;
        const float *vh = v + (size_t)head * state;
        const float *yh = y + (size_t)head * state;
        double dot = 0.0;
        double reference = 0.0;
        double worst = 0.0;
        uint32_t index;

        for (index = 0; index < state; index++)
            dot += (double)kh[index] * (double)qh[index];
        for (index = 0; index < state; index++) {
            const double want = scale * (double)beta[head] * (double)vh[index] * dot;
            const double gap = fabs((double)yh[index] - want);

            if (fabs(want) > reference)
                reference = fabs(want);
            if (gap > worst)
                worst = gap;
        }
        worst = reference > 0.0 ? worst / reference : (worst > 0.0 ? 1.0 : 0.0);
        if (worst > worst_all)
            worst_all = worst;
        if (worst > 1.0e-3)
            bad_heads++;
        {
            /* THE SIGN OF THE ERROR, not just its size. A Tf32 unpack TRUNCATES, so every
             * multiply stage biases the magnitude DOWN by a fixed fraction and the ratio
             * lands below 1 on every head — which is a different finding from rounding noise
             * and points at the stage COUNT rather than at any one stage. */
            const double want = scale * (double)beta[head] * (double)vh[0] * dot;

            if (want != 0.0) {
                ratio_sum += (double)yh[0] / want;
                ratio_count++;
            }
            if (detail)
                fprintf(stderr, "pn_tt: gdn SELFCHECK DELTA head %2u  rel %.3e  k.q %+.6g  "
                                "beta %.6g  want[0] %+.6g  card[0] %+.6g\n",
                        head, worst, dot, (double)beta[head], want, (double)yh[0]);
        }
    }
    fprintf(stderr, "pn_tt: gdn SELFCHECK DELTA layer %2u — %u of %u heads over 1e-3, worst "
                    "%.3e, mean card/want %.6f\n",
            layer, bad_heads, v_heads, worst_all,
            ratio_count ? ratio_sum / ratio_count : 0.0);
}

/** The PHASE DELTA descriptor, whatever put q|k|v and the gate block where they are. */
static int gdn_delta_descriptor(pn_tt_device *device, uint32_t layer)
{
    ft_tt_op_gdn_request request;

    request = gdn_request(device, FT_TT_OP_GDN_PHASE_DELTA);
    request.activation_in = gdn_operand(device, &device->gdn_stream_out);
    request.activation_out = gdn_operand(device, &device->gdn_y);
    request.gate = gdn_operand(device, device->gdn_defer && device->gdn_row < device->gdn_gate_rows
                                           ? &device->gdn_gate_row[device->gdn_row]
                                           : &device->gdn_gate);
    gdn_table_gate_apply(device, &request);
    request.state_recurrent = gdn_operand(device, &device->gdn_region[layer].state);
    request.row = device->gdn_row;
    if (gdn_state_out_buffer(device, layer, false))
        request.state_out = gdn_operand(device, gdn_state_out_buffer(device, layer, false));
    /* THE WORK KNOB. `PN_TT_GDN_HEADS=N` runs the descriptor over the first N v-heads instead
     * of all 32 — the compute kernel's outer loop and nothing else — so the descriptor's time
     * against N says whether that time is the card's arithmetic or a fixed cost around it.
     *
     * THE TRAFFIC IS THE SAME SHAPE AT EVERY N, WHICH IS WHAT MAKES THIS A STOPWATCH — but
     * since the streams became head-major row-0 strips the ADDRESSES move with N as well:
     * `ftg_delta_stream_base()` is `k_heads * state_tiles`, so a short run reads q|k|v out of
     * a shorter strip. Same tiles per head, same reads per head, different values. K-HEADS
     * MOVES WITH IT because the family refuses `v_heads % k_heads != 0`. It was a
     * byte-identical layout under the old dense `[heads, S]` packing; it is not now, and this
     * arm was never a gate. */
    {
        const uint32_t heads = gdn_probe_knob("PN_TT_GDN_HEADS");

        if (heads > 0u && heads < request.v_heads) {
            request.v_heads = heads;
            request.k_heads = heads;
        }
    }
    /* BEFORE the descriptor, at the first DELTA of the run, where the carried state it
     * borrows is still the zero the sequence starts from. Does nothing unless asked. */
    pn_tt_gdn_core_scaling_probe(device, &request);
    return gdn_submit(device, &request);
}

int pn_tt_gdn_delta_oncard(pn_tt_device *device, uint32_t layer, const float *decay,
                           const float *beta)
{
    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count)
        return -1;
    /* The gate block is the ONLY operand this path uploads — 256 B a layer. q, k and v are
     * already in `gdn_stream_out`: PHASE CONV wrote them there and the per-head L2 norm wrote
     * its answer back over the first `2*k_heads*state_tiles` tiles of the same buffer. */
    if (pn_tt_gdn_gate_upload(device, decay, beta) != 0)
        return -1;
    return gdn_delta_descriptor(device, layer);
}

int pn_tt_gdn_delta(pn_tt_device *device, uint32_t layer, const float *q, const float *k,
                    const float *v, const float *decay, const float *beta, float *y)
{
    const ftg_shape shape = gdn_shape(device, FT_TT_OP_GDN_PHASE_DELTA);

    if (!pn_tt_gdn_is_armed(device) || !q || !k || !v || !decay || !beta || !y ||
        layer >= device->gdn_shape.layer_count)
        return -1;
    if (gdn_stage_delta_input(device, q, k, v, decay, beta) != 0)
        return -1;
    if (gdn_delta_descriptor(device, layer) != 0)
        return -1;

    /* y is a ROW-0 STRIP, head-major, so it comes back as row 0 of `v_heads * state_tiles`
     * tiles — the same 16,384 B the [v_heads, S] matrix cost, in the same number of 64-byte
     * transactions, because a whole-tile read of the dense form moved exactly the rows the
     * strip skips. On the on-card path this read does not happen at all. */
    if (gdn_row0_move_at(device, &device->gdn_y, 0u,
                         shape.v_heads * shape.state_tiles, y, true) != 0)
        return -1;
    /* The FIRST TOKEN only: the closed form is a statement about a ZERO carried state, and
     * every descriptor after the first token's has one this function cannot see. */
    if (device->gdn_delta_checks < device->gdn_shape.layer_count &&
        getenv("PN_TT_GDN_SELFCHECK")) {
        gdn_self_check_delta(device, layer, q, k, v, beta, y);
        device->gdn_delta_checks++;
    }
    return 0;
}

/* ================================================================
 * The on-card chain's operands
 *
 * The strip bridge, the two per-head reduces and the elementwise gate all address these three
 * buffers directly, and none of them knows what a `pn_tt_gdn_buffer` is. Handing out the
 * `ft_tt_op_operand` rather than the buffer is what keeps that boundary: every one of them is
 * an fp32 4096-byte-page endpoint and nothing else about this family reaches them.
 * ================================================================ */

ft_tt_op_operand pn_tt_gdn_stream_in_operand(const pn_tt_device *device)
{
    return gdn_operand(device, &device->gdn_stream_in);
}

ft_tt_op_operand pn_tt_gdn_stream_operand(const pn_tt_device *device)
{
    return gdn_operand(device, &device->gdn_stream_out);
}

ft_tt_op_operand pn_tt_gdn_y_operand(const pn_tt_device *device)
{
    return gdn_operand(device, &device->gdn_y);
}

int pn_tt_gdn_stream_read(pn_tt_device *device, uint32_t first_tile, uint32_t tiles,
                          float *out)
{
    if (!pn_tt_gdn_is_armed(device) || !out || tiles == 0u ||
        first_tile + tiles > device->gdn_shape.conv_dim / PN_TT_GDN_TILE_DIM)
        return -1;
    return gdn_row0_move_at(device, &device->gdn_stream_out, first_tile, tiles, out, true);
}

/* ================================================================
 * The phase-isolation seam. See pn_tt_gdn.h for why it exists.
 * ================================================================ */

/** The buffer and its tile count, or NULL for an unarmed family or an unknown strip. */
static const pn_tt_gdn_buffer *gdn_strip_buffer(const pn_tt_device *device,
                                                pn_tt_gdn_strip_id strip, uint32_t *out_tiles)
{
    switch (strip) {
    case PN_TT_GDN_STRIP_STREAM_IN:
        *out_tiles = pn_tt_gdn_stream_tiles(device);
        return &device->gdn_stream_in;
    case PN_TT_GDN_STRIP_STREAM_OUT:
        *out_tiles = pn_tt_gdn_stream_tiles(device);
        return &device->gdn_stream_out;
    case PN_TT_GDN_STRIP_Y:
        *out_tiles = pn_tt_gdn_value_tiles(device);
        return &device->gdn_y;
    }
    return NULL;
}

int pn_tt_gdn_strip_row_move(pn_tt_device *device, pn_tt_gdn_strip_id strip, uint32_t row,
                             uint32_t tiles, float *values, bool reading)
{
    uint32_t capacity = 0u;
    const pn_tt_gdn_buffer *buffer;

    if (!pn_tt_gdn_is_armed(device) || !values || tiles == 0u)
        return -1;
    buffer = gdn_strip_buffer(device, strip, &capacity);
    if (!buffer || tiles > capacity)
        return -1;
    return pn_tt_fp32_row_move(device, buffer->bank, buffer->address, row, tiles, values,
                               reading);
}

int pn_tt_gdn_strip_zero(pn_tt_device *device, pn_tt_gdn_strip_id strip)
{
    uint32_t tiles = 0u;
    const pn_tt_gdn_buffer *buffer;

    if (!pn_tt_gdn_is_armed(device))
        return -1;
    buffer = gdn_strip_buffer(device, strip, &tiles);
    if (!buffer || tiles == 0u)
        return -1;
    return pn_tt_zero_dram(device, buffer->bank, buffer->address,
                           (size_t)tiles * PN_TT_GDN_TILE_BYTES);
}

uint32_t pn_tt_gdn_stream_tiles(const pn_tt_device *device)
{
    return device ? device->gdn_shape.conv_dim / PN_TT_GDN_TILE_DIM : 0u;
}

uint32_t pn_tt_gdn_key_tiles(const pn_tt_device *device)
{
    if (!device)
        return 0u;
    return 2u * device->gdn_shape.k_heads * (device->gdn_shape.state_size / PN_TT_GDN_TILE_DIM);
}

uint32_t pn_tt_gdn_v_heads(const pn_tt_device *device)
{
    return device ? device->gdn_shape.v_heads : 0u;
}

uint32_t pn_tt_gdn_value_tiles(const pn_tt_device *device)
{
    if (!device)
        return 0u;
    return device->gdn_shape.v_heads * (device->gdn_shape.state_size / PN_TT_GDN_TILE_DIM);
}

#endif /* USE_TT */
