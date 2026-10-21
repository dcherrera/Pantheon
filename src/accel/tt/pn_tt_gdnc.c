/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdnc.c
 * @brief The host stager for the CHUNKWISE Gated-Delta-Net op. The API, the ownership split
 *        and the cost model are in pn_tt_gdnc.h.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FILE INCLUDES A KERNEL HEADER
 * ---------------------------------------------------------------------------
 * The same reason `pn_tt_gdn.c` does: the stager and the two dataflow kernels are the two
 * ends of a contract nothing checks at run time, so every offset comes from the `ftgc_*`
 * accessors in `resident_gdnc_protocol.h` — the functions the reader and the writer compute
 * their addresses with — rather than from arithmetic respelled here. It also lets THIS file
 * state, in code, that the window and the state tile maps are the decode family's.
 *
 * ---------------------------------------------------------------------------
 * THE RED ARMS, AND WHAT EACH ONE PROVES
 * ---------------------------------------------------------------------------
 * Every one mutates ONLY WHAT IS UPLOADED — the ninth recorded vacuous-gate class in this
 * project is a red arm that mutated an input the oracle also read and scored 0.000e+00.
 *
 *   gdnc_weight_row0   stage the conv weight the DECODE family's way, live in row 0 and zero
 *                      below. It is the plausible transcription error, and it is invisible at
 *                      row 0 of every chunk and wrong at rows 1..C-1 — so it fires from a
 *                      chunk's SECOND position and not from its first.
 *   gdnc_gate_first    build the decay tile and the gate run from position 0's decay and beta
 *                      repeated over the chunk. That is the shape of a caller who forgot the
 *                      gate is per POSITION now, and it is likewise invisible at row 0.
 *   gdnc_window_wipe   zero the shared conv window before every CONV descriptor. **INVISIBLE
 *                      IN CHUNK 0**, where the window starts zero anyway, and wrong from
 *                      chunk 1 — the boundary carry a single-chunk test cannot see.
 *   gdnc_state_wipe    zero the shared recurrent state before every DELTA descriptor. Same
 *                      structure, same invisibility in chunk 0.
 *
 * The last two are the host-level form of the two arms `foundry/tt/tests/test_phase6_gdnc.c`
 * fires on the card, and they are here because the CARRY IS THIS FILE'S CLAIM: the kernels
 * carry whatever they are pointed at, and what says the chunks are one sequence is that this
 * file points every chunk at the SAME buffer and never touches it in between.
 *
 * @status SHIPPED ON 2026-08-25 at C = 16, behind `PN_TT_PREFILL_GDNC` (default ON, `0`
 *   declines). Scored 6/6 against an fp32-KV oracle at 339 positions.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdnc.h"
#include "pn_tt_dispatch.h"
#include "pn_tt_stage_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_gdn.h"

#include "tt/ft_tt_op_gdnc.h"
#include "tt/ft_tt_tilize.h"

/* See the file header: the L1 pointer attribute is the SFPI fork's, not the contract's. */
#define FTQ_L1_PTR
#include "tt/kernels/ftgc_compact.h"
#include "tt/kernels/resident_gdnc_protocol.h"

/**
 * **ON BY DEFAULT as of 2026-08-25, at C = 16.** Only an explicit `0` turns it off.
 *
 * THE CHUNKWISE MIXERS ARE A DIFFERENT NUMERICAL PATH from the decode family, so this default
 * waited on an output gate rather than on an argument, and the first gate FAILED it: at
 * C = 32 the arm misses worst-document. The chunk length is the whole of that (see
 * `kGdncFirstRow`), and at C = 16 — one PHASE CONV a pass, `ceil(rows/16)` PHASE DELTA
 * descriptors IN PLACE — the arm clears all six criteria against an fp32-KV oracle at 339
 * positions and is BETTER THAN THE SEQUENTIAL ARM IT REPLACES on three of them:
 *
 *     criterion (limit)    C = 32                C = 16              sequential
 *     ppl_ratio (1.05)     1.000515              0.987406            0.978511
 *     paired_t  (4)        0.027                 -1.866              -2.358
 *     disagree  (0.08)     0.079646 (+0.000354)  0.038348 (+0.041652) 0.029499
 *     worst_doc (1.05)     1.048610 (+0.00139)   1.010782 (+0.039218) 1.020988
 *     dnll_sd              0.355599              0.125078            0.169621
 *     kl_mean              0.0440231             0.00251879          0.00424355
 *
 * **THE MARGIN IS WHY C = 16 SHIPS AND C = 32 DOES NOT.** C = 32 passes disagreement by
 * 0.000354 and worst-document by 0.00139, both on ONE 113-position document; C = 16 passes the
 * same two by 118x and 28x those margins.
 */
#define PN_TT_GDNC_ENV "PN_TT_PREFILL_GDNC"

/** Which Tensix the chunkwise triad runs on. A default, not a constant: it must be a core no
 *  other residency and no matmul group member holds, and the group MOVES with
 *  `PN_TT_MATMUL_CORES`. `pn_tt_prepare_family()` refuses a collision BY NAME, so this is
 *  only the first place to look. 95 is above the shipped four-way GDN head split (90 member 0,
 *  91 the narrow head norm, 92-94 the members) and below attention's 103. */
#define PN_TT_CORE_GDNC_ENV     "PN_TT_CORE_GDNC"
#define PN_TT_CORE_GDNC_DEFAULT 95u

/**
 * THE CORE SPLIT. `PN_TT_GDNC_CORES=N` arms N-1 EXTRA chunkwise residencies and gives each a
 * slice of BOTH phases: PHASE CONV `conv_tile_cols/N` tile columns, PHASE DELTA `v_heads/N`
 * v-heads.
 *
 * **ON BY DEFAULT AT 8 since 2026-08-28**, after its own confirmation round on a fresh binary:
 * 168.97 -> 195.91 tok/s (+15.9%), reproduced as a later session's control at 195.06/194.66
 * against a 0.024% drift, `cmp` IDENTICAL at every rung, decode wall untouched. Two independent
 * binaries. The ladder is 169.0 / 186.6 / 195.5 / **196.0** / 187.8 at N = 1/2/4/8/16.
 *
 * WHY IT IS LEGAL, AND WHY THE WIRE ALREADY CARRIED IT. `ft_tt_op_gdnc_request.first_col` and
 * `.col_count` (`kGdncColSlice`) have been on the wire since this family shipped, refused out
 * of range by all four sides, and `foundry/tt/tests/test_phase6_gdnc.c` already proves ON
 * SILICON that four column slices are BYTE-FOR-BYTE the one whole descriptor. Nothing was
 * missing but the host: `pn_tt_gdnc.c` armed exactly one residency and never set the two
 * fields. PHASE CONV divides exactly by channel — column `c` reads only column `c`'s input
 * tile, window planes and weight planes, and writes only column `c`'s conv_out tile and window
 * rows — and PHASE DELTA divides by v-head the same way the decode family's does.
 *
 * WHY THIS IS THE LEVER. The CONV descriptor costs **680.2 / 681.8 / 682.5 us at 32 / 16 / 8
 * LIVE ROWS** — 0.34% over a 4x range — because none of its cost depends on live rows: it is
 * `conv_tile_cols = 256` iterations of `read_one_column()`, each ending in ONE NoC read barrier
 * (`resident_gdnc_reader.c:292`, loop `:643-644`), and the work is per COLUMN, not per row.
 *
 * IT IS COMPUTE, NOT BARRIER LATENCY — corrected 2026-08-28, and the correction matters.
 * The first reading here was "2.66 us a column of barrier-serialized latency", which predicts
 * that batching columns behind one barrier pays. **Measured single-core, the CONV depth knob is
 * FLAT: 679.1 / 680.9 / 683.8 us at D = 1/2/4.** So at 256 columns on one core the descriptor is
 * bound by the ~18 tile ops a column, not by the barriers between them. The depth only becomes a
 * lever ONCE THIS SPLIT HAS CUT THE COMPUTE 8x and the reader is finally the bind: at N=8 it is
 * 177.5 / 163.6 / 157.9 us at D = 1/2/4. **Composition, not independence.** What the split
 * itself buys is parallel compute — CONV 680.2 -> 177.1 (3.84x), DELTA 439.1 -> 143.7 (3.06x).
 *
 * AND THE KNEE IS 8 BECAUSE OF THE HOST, NOT THE CARD. At N=8 member 0 takes 113.1 us a
 * descriptor while members 1..7 take 3.3-10.6 — the set genuinely runs concurrently — but the
 * host awaits every member IN SERIES, so a marginal member costs ~3.5 us of host against a card
 * saving already down to ~4. At 16 the fifteen trailing awaits add ~52 us and CONV goes back to
 * 266.9. That is `wall(N) = 0.3885 + 0.00199*N + card(N)` in a second family, and it is what a
 * device-side command fan-out removes.
 *
 * THE MEMBERS START AT 107 and that is a first place to look, not a guarantee.
 * `pn_tt_prepare_family_on_core()` refuses a collision BY NAME against every live residency and
 * against the matmul group, which MOVES with `PN_TT_MATMUL_CORES`. 107..121 is fifteen
 * consecutive Tensix free at every shipped default of every knob in this backend (72 reduce /
 * 73 eltwise / 74 sample / 75-81 the matmul group / 82-84 attention / 85,86 render / 87 the
 * merged gate / 90,92-94 gdn / 91,103 head norms / 95 gdnc itself / 106 the eltwise split's
 * member / 122,129 the render splits' members). The ONE family that can reach into it is the
 * eltwise split at its maximum of 16 members (106..120), which is eight times its shipped
 * default of two — and that is a refusal by name, not a corruption.
 */
#define PN_TT_GDNC_CORES_ENV             "PN_TT_GDNC_CORES"
#define PN_TT_GDNC_CORES_DEFAULT         8u
#define PN_TT_GDNC_MEMBER_CORE_FIRST_ENV "PN_TT_GDNC_MEMBER_CORE_FIRST"
/* 114 since the PX7 placement shipped (was 107, inside the 16-member matmul's 96..110). */
#define PN_TT_GDNC_MEMBER_CORE_DEFAULT   114u
/* PM 8 ITEM 7 step 2: members 1..N-1 on an explicit comma-separated core list (any order,
 * not necessarily consecutive) instead of `first + index`. Unset = the consecutive block. */
#define PN_TT_GDNC_MEMBER_CORE_LIST_ENV  "PN_TT_GDNC_MEMBER_CORE_LIST"

/**
 * THE COMPLETION SHAPE OF THE SPLIT, AND IT IS A HAZARD BEHIND ITS OWN SPELLING.
 * `PN_TT_GDNC_GROUP=i-know-this-wedges-the-posting-core` makes every member post into ONE
 * group completion block and replaces the per-member await loop with member 0's count plus a
 * single BLOCK16 read of the whole slot array. **`1` IS REFUSED**, because a `1` is what
 * someone types who has not read the paragraph below.
 *
 * ---------------------------------------------------------------------------
 * WHY THE SPELLING, AND IT IS THE SAME EVIDENCE `PN_TT_GROUP_COMPLETION` CARRIES
 * ---------------------------------------------------------------------------
 * `ftq_group_post()` is an inline cross-core NoC write from DM1, and the matmul group already
 * paid for it: two armed arms WEDGED a core each — 75, then 71 — at
 * `instrn_buf_status = 0x66`, which is the state where BRISC firmware cannot complete a boot
 * and only a whole-card reset clears it (`pn_tt_matmul_group.c`, PN_TT_GROUP_COMPLETION_ENV).
 * What survives that account: the post is **an ACCELERANT of the resident queue's own
 * "retired clean, will not boot next" defect** (`DEV_NOTES.md` §1) and not the cause, because
 * a DISARMED N=4 arm took all four cores after ~46x the work. Neither remoteness nor member
 * index predicts which core goes.
 *
 * **AND ON THE MATMUL THE TRADE WAS ALREADY JUDGED NOT WORTH IT** — there the post buys
 * TRANSACTION COUNT, `1 + ceil(8N/16)` for the whole group against `PN_TT_FAST_COMPLETION`'s
 * one per member: 5 against 8 at N=8, better and not enough better to be worth a core.
 *
 * **THIS FAMILY'S CASE IS A DIFFERENT QUANTITY, WHICH IS THE ONLY REASON THE CODE EXISTS.**
 * It is not transactions, it is SERIALIZATION: N independent blocking poll loops, run one
 * after another. Measured at N = 8 on 2026-08-28, member 0 at 113.2 us a descriptor and
 * members 1..7 at 3.3-10.6 — member 0's 0.4158 s is the card genuinely working and members
 * 1..7's **0.1164 s** is the host learning, one at a time, something the card already
 * finished. Against an 8.35 s prefill that is **1.4%**, which does not buy a wedge.
 * The case is at N = 16, where the record has 187.8 tok/s against N = 8's 196.0 with the
 * cause named as the trailing awaits — a 4.2% question, and the one arm that can say whether
 * the 16-member regression is the host's or the card's.
 *
 * RECOVERY, KNOWN BEFORE ARMING: `ft_tt_core recover <core> --rung 3 --driver udrv --write
 * --then-probe` under sudo. Whole card, in place, no vendor driver and no PCI remove/rescan;
 * measured at ~10 s wall on 2026-08-28 recovering core 90 from exactly the 0x66 state.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS, MECHANICALLY
 *
 * WHY IT EXISTS, MEASURED RATHER THAN ARGUED. At N = 8 the per-member awaits are member 0 at
 * **113.2 us a descriptor** and members 1..7 at **3.3-10.6** — the set genuinely runs
 * concurrently, and the host still pays every one of them IN SERIES. That is ~38 us of host a
 * descriptor that the card is not spending, over 3,672 descriptors a prompt, and it is why
 * the member ladder peaks at 8 and goes BACKWARDS at 16: a marginal member costs ~3.5 us of
 * host against a card saving already down to ~4.
 *
 * NOTHING HERE IS NEW MACHINERY. `resident_gdnc_writer.c:404` already calls
 * `ftq_group_post()`; the block layout, its magic, the slot read and the two-assertion check
 * are `ft_tt_queue.c`'s; and `pn_tt_matmul_group.c` has run exactly this shape as its default
 * since 2026-08-20 over millions of poll reads. The group binding rides in the kernel's
 * RUNTIME ARGS, so this is host-only: **no kernel change and no program-blob regeneration.**
 *
 * IT IS NOT FREE, WHICH IS WHY IT SHIPS OFF UNTIL IT IS MEASURED. `ftq_group_post()`'s refused
 * inline write is the ONE fault the protocol raises after `completed_count` is published, so
 * `pn_tt_residency_start()` demotes these residencies from the `line` completion shape to
 * `window` — five extra reads on the settle of the ONE await that remains. The trade is N poll
 * loops for one, against five reads; it has to be read off the card, not predicted.
 */
#define PN_TT_GDNC_GROUP_ENV "PN_TT_GDNC_GROUP"
#define PN_TT_GDNC_GROUP_ACK "i-know-this-wedges-the-posting-core"
/** The block's L1 address on member 0's core. The SAME derivation `pn_tt_matmul_group.c:1177`
 *  makes, so the two cannot drift into two spellings of one address. */
#define PN_TT_GDNC_GROUP_ADDRESS (FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET)

#define PN_TT_GDNC_TILE_DIM   FT_TT_OP_TILE_DIM
#define PN_TT_GDNC_TILE_BYTES FT_TT_OP_GDNC_FP32_TILE_BYTES

/** Diff granularity: a tile FACE (16x16 fp32) for the decay tiles, one 32-float run for the
 *  gate block. `PN_TT_GDNC_STAGE_DIFF` defaults ON: 213.56 -> 235.5 tok/s, IDENTICAL (spec). */
#define PN_TT_GDNC_STAGE_DECAY_UNIT   1024u
#define PN_TT_GDNC_STAGE_GATE_UNIT    gdnc_stage_gate_unit()
#define PN_TT_GDNC_STAGE_DIFF_DEFAULT 1

/* ================================================================
 * Shape and operands
 * ================================================================ */

/**
 * The request describing this residency, with the operands a caller then fills in.
 *
 * IT CARRIES BOTH PHASES' SHAPES WHATEVER THE PHASE, because ONE circular-buffer placement
 * serves both — a residency is armed once and then answers whatever the queue hands it, so a
 * request naming only its own phase would place buffers the OTHER phase reads out of.
 * `ft_tt_op_gdnc_place_cbs()` refuses one that does not.
 */
static ft_tt_op_gdnc_request gdnc_request(const pn_tt_device *device, uint32_t phase,
                                          uint32_t chunk_rows)
{
    ft_tt_op_gdnc_request request;

    memset(&request, 0, sizeof(request));
    request.phase = phase;
    request.chunk_rows = chunk_rows;
    request.conv_dim = device->gdn_shape.conv_dim;
    request.conv_taps = device->gdn_shape.conv_taps;
    request.state_tiles = device->gdn_shape.state_size / PN_TT_GDNC_TILE_DIM;
    request.v_heads = device->gdn_shape.v_heads;
    request.k_heads = device->gdn_shape.k_heads;
    return request;
}

/** The device's own view of a DELTA shape, for the `ftgc_*` accessors. */
static ftgc_shape gdnc_delta_shape(const pn_tt_device *device)
{
    ftgc_shape shape;

    memset(&shape, 0, sizeof(shape));
    shape.phase = FT_TT_OP_GDNC_PHASE_DELTA;
    shape.conv_taps = device->gdn_shape.conv_taps;
    shape.conv_tile_cols = device->gdn_shape.conv_dim / PN_TT_GDNC_TILE_DIM;
    shape.state_tiles = device->gdn_shape.state_size / PN_TT_GDNC_TILE_DIM;
    shape.v_heads = device->gdn_shape.v_heads;
    shape.k_heads = device->gdn_shape.k_heads;
    shape.head_count = device->gdn_shape.v_heads;
    shape.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    return shape;
}

/** An fp32 operand descriptor for one of this backend's DRAM buffers. */
static ft_tt_op_operand gdnc_operand(const pn_tt_device *device,
                                     const pn_tt_gdn_buffer *buffer)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[buffer->bank].x;
    operand.y = (uint16_t)device->dram[buffer->bank].y;
    operand.address = buffer->address;
    operand.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    return operand;
}

/** Carve one DRAM buffer. `PN_TT_GDNC_ANY_BANK` round-robins; anything else PINS. */
#define PN_TT_GDNC_ANY_BANK UINT32_MAX

static int gdnc_alloc_in(pn_tt_device *device, uint32_t bank, size_t bytes, bool zero,
                         pn_tt_gdn_buffer *out)
{
    ft_tt_buffer buffer;

    if (bytes == 0u)
        return -1;
    if (bank == PN_TT_GDNC_ANY_BANK)
        bank = device->strip_bank_cursor++ % device->dram_bank_count;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes, &buffer) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full allocating a %zu-byte GDNC buffer\n", bytes);
        return -1;
    }
    out->bank = buffer.logical_bank;
    out->address = (uint32_t)buffer.address;
    if (zero && pn_tt_zero_dram(device, out->bank, out->address, bytes) != 0)
        return -1;
    return 0;
}

/* ================================================================
 * Arming
 * ================================================================ */

bool pn_tt_gdnc_is_armed(const pn_tt_device *device)
{
    return device && device->gdnc.running && device->gdnc_constant != NULL;
}

uint32_t pn_tt_gdnc_min_chunk_rows(const pn_tt_device *device)
{
    if (!pn_tt_gdnc_is_armed(device))
        return 0u;
    return device->gdn_shape.conv_taps - 1u;
}

/** The four PHASE DELTA program constants — `I`, `2I`, the strictly-lower mask and the
 *  `scale` column. Identical in every layer and every chunk, so they go up once. */
static int gdnc_upload_delta_constants(pn_tt_device *device)
{
    unsigned char *tiles = malloc(ft_tt_op_gdnc_delta_constant_tile_bytes());
    uint32_t slot;
    int rc = -1;

    if (!tiles)
        return -1;
    if (ft_tt_op_gdnc_build_delta_constants(device->gdn_shape.state_size, tiles,
                                            ft_tt_op_gdnc_delta_constant_tile_bytes()) ==
        FT_TT_OK) {
        rc = 0;
        for (slot = 0; slot < PN_TT_GDNC_STAGE_SLOTS && rc == 0; slot++)
            if (pn_tt_write_dram(device, device->gdnc_delta_constant[slot].bank,
                                 device->gdnc_delta_constant[slot].address, tiles,
                                 ft_tt_op_gdnc_delta_constant_tile_bytes()) != 0)
                rc = -1;
    }
    free(tiles);
    return rc;
}

/**
 * PM 8: `PN_TT_GDNC_CONST_REPLICAS` copies of every layer's conv constant, copy r in bank
 * `(bank of copy 0 + r) % banks`. Every CONV member reads its columns' weight planes — 16 KiB
 * a column a band, 4 MiB a band over the set — and with one copy all of it comes from ONE
 * bank. Same bytes in every copy, so the answer cannot move.
 */
#define PN_TT_GDNC_CONST_REPLICAS_ENV "PN_TT_GDNC_CONST_REPLICAS"

static int gdnc_alloc_constant_replicas(pn_tt_device *device, size_t constant)
{
    /* DEFAULT 8 (run_pm8_conv_replicas.sh: serve warm 1,630 TTFT 2,547/2,537 ->
     * 2,472/2,458 ms, CLI sequential 1.026 -> 0.967 s, IDENTICAL). `=1` off. */
    uint32_t replicas = pn_tt_env_u32(PN_TT_GDNC_CONST_REPLICAS_ENV, 8u);
    const uint32_t layers = device->gdn_shape.layer_count;
    uint32_t layer;
    uint32_t copy;

    if (replicas > device->dram_bank_count)
        replicas = device->dram_bank_count;
    device->gdnc_constant_replicas = replicas < 1u ? 1u : replicas;
    if (device->gdnc_constant_replicas == 1u)
        return 0;
    device->gdnc_constant_replica = calloc((size_t)layers * (device->gdnc_constant_replicas - 1u),
                                           sizeof(*device->gdnc_constant_replica));
    if (!device->gdnc_constant_replica)
        return -1;
    for (layer = 0; layer < layers; layer++)
        for (copy = 1; copy < device->gdnc_constant_replicas; copy++)
            if (gdnc_alloc_in(device,
                              (device->gdnc_constant[layer].bank + copy) %
                                  device->dram_bank_count,
                              constant, false,
                              &device->gdnc_constant_replica[(size_t)layer *
                                                                 (device->gdnc_constant_replicas -
                                                                  1u) +
                                                             copy - 1u]) != 0)
                return -1;
    fprintf(stderr, "pn_tt: gdnc — conv constant in %u banks (%.1f MiB of copies)\n",
            device->gdnc_constant_replicas,
            (double)constant * layers * (device->gdnc_constant_replicas - 1u) /
                (1024.0 * 1024.0));
    return 0;
}

/** Copy `copy` of `layer`'s conv constant (0 = the original). */
static const pn_tt_gdn_buffer *gdnc_constant_copy(const pn_tt_device *device, uint32_t layer,
                                                  uint32_t copy)
{
    if (copy == 0u || device->gdnc_constant_replicas <= 1u)
        return &device->gdnc_constant[layer];
    return &device->gdnc_constant_replica[(size_t)layer * (device->gdnc_constant_replicas - 1u) +
                                          copy - 1u];
}

static int gdnc_alloc_regions(pn_tt_device *device, const ft_tt_op_gdnc_request *conv,
                              const ft_tt_op_gdnc_request *delta)
{
    const size_t constant = ft_tt_op_gdnc_constant_bytes(conv);
    uint32_t layer;

    device->gdnc_constant = calloc(device->gdn_shape.layer_count,
                                   sizeof(*device->gdnc_constant));
    if (!device->gdnc_constant)
        return -1;
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++)
        if (gdnc_alloc_in(device, PN_TT_GDNC_ANY_BANK, constant, true,
                          &device->gdnc_constant[layer]) != 0)
            return -1;
    if (gdnc_alloc_constant_replicas(device, constant) != 0)
        return -1;

    /* THE GATE BLOCK IS PINNED TO THE DELTA CONSTANT'S BANK. They share that operand's packed
     * NoC coordinate word, exactly as the activation pair shares its own, and they fail the
     * same way: the reader would pull 640 bytes from the right bank at an address belonging
     * to nothing, the descriptor would retire clean, and every gate column would be whatever
     * was there. Nothing on the device can catch it. */
    for (uint32_t slot = 0; slot < PN_TT_GDNC_STAGE_SLOTS; slot++) {
        if (gdnc_alloc_in(device, PN_TT_GDNC_ANY_BANK,
                          ft_tt_op_gdnc_delta_constant_bytes(delta), true,
                          &device->gdnc_delta_constant[slot]) != 0 ||
            gdnc_alloc_in(device, device->gdnc_delta_constant[slot].bank,
                          ft_tt_op_gdnc_gate_bytes(delta), true, &device->gdnc_gate[slot]) != 0)
            return -1;
        device->gdnc_shadow_decay[slot] = calloc((size_t)device->gdn_shape.v_heads,
                                                 PN_TT_GDNC_TILE_BYTES);
        device->gdnc_shadow_gate[slot] = calloc((size_t)device->gdn_shape.v_heads *
                                                    FT_TT_OP_GDNC_GATE_STRIDE,
                                                sizeof(float));
        device->gdnc_shadow_valid[slot] = false;
        device->gdnc_prestaged[slot].valid = false;
        if (!device->gdnc_shadow_decay[slot] || !device->gdnc_shadow_gate[slot])
            return -1;
    }

    /* The per-chunk host scratch: one decay tile per v-head, contiguous so the whole 128 KiB
     * block is ONE write rather than `v_heads` of them, and one gate run per v-head. */
    device->gdnc_decay_stage = calloc((size_t)device->gdn_shape.v_heads,
                                      PN_TT_GDNC_TILE_BYTES);
    device->gdnc_gate_stage = calloc((size_t)device->gdn_shape.v_heads *
                                         FT_TT_OP_GDNC_GATE_STRIDE,
                                     sizeof(*device->gdnc_gate_stage));
    if (!device->gdnc_decay_stage || !device->gdnc_gate_stage)
        return -1;
    if (gdnc_upload_delta_constants(device) != 0)
        return -1;

    fprintf(stderr, "pn_tt: gdnc — %u layers x %.2f MiB of row-replicated conv weight + band "
                    "= %.2f MiB, plus %.0f KiB of delta constants. The window and the "
                    "recurrent state are the gdn family's, byte for byte.\n",
            device->gdn_shape.layer_count, (double)constant / (1024.0 * 1024.0),
            (double)(constant * device->gdn_shape.layer_count) / (1024.0 * 1024.0),
            (double)PN_TT_GDNC_STAGE_SLOTS *
                (double)(ft_tt_op_gdnc_delta_constant_bytes(delta) +
                         ft_tt_op_gdnc_gate_bytes(delta)) / 1024.0);
    return 0;
}

/**
 * How many members this run may split across.
 *
 * Clamped to the array and then shrunk until it divides BOTH quantities exactly — CONV's tile
 * columns and DELTA's v-heads. Both, not either: one placement and one member set serve both
 * phases, and a count that divided only one would leave the other running a ragged split whose
 * longest member sets the descriptor's cost while the short ones idle. At Ornith that is
 * `gcd(256, 32) = 32`, so 1/2/4/8/16 all survive and 3 collapses to 2.
 */
static uint32_t gdnc_member_count(const pn_tt_device *device)
{
    const uint32_t columns = device->gdn_shape.conv_dim / PN_TT_GDNC_TILE_DIM;
    const uint32_t heads = device->gdn_shape.v_heads;
    uint32_t members = pn_tt_env_u32(PN_TT_GDNC_CORES_ENV, PN_TT_GDNC_CORES_DEFAULT);

    if (members > PN_TT_GDNC_MAX_MEMBERS)
        members = PN_TT_GDNC_MAX_MEMBERS;
    if (columns == 0u || heads == 0u)
        return 1u;
    if (members > columns)
        members = columns;
    if (members > heads)
        members = heads;
    while (members > 1u && ((columns % members) != 0u || (heads % members) != 0u))
        members--;
    return members;
}

/**
 * The core member `index + 1` arms on: entry `index` of `PN_TT_GDNC_MEMBER_CORE_LIST` when the
 * list is set, else `first + index`. A list too short for the count returns UINT32_MAX, which
 * stops the set there (the split then shrinks to a count that still divides).
 */
static uint32_t gdnc_member_core(uint32_t index, uint32_t first)
{
    const char *at = getenv(PN_TT_GDNC_MEMBER_CORE_LIST_ENV);
    uint32_t position = 0u;
    char *end;

    if (!at || !*at)
        return first + index;
    while (*at) {
        const unsigned long core = strtoul(at, &end, 10);

        if (end == at)
            break;
        if (position++ == index)
            return (uint32_t)core;
        at = *end == ',' ? end + 1 : end;
    }
    return UINT32_MAX;
}

/**
 * Arm the EXTRA chunkwise residencies, if this run asked for a split.
 *
 * They take the SAME circular-buffer placement member 0 has, and that is a property of the
 * family and not a convenience: `resolve_cb_pages()` reads `state_tiles` and `conv_taps` and
 * nothing else, so a member owning sixteen columns places exactly what a member owning two
 * hundred and fifty-six does. Sizing them any other way would be measuring a different
 * residency from the one the model runs.
 *
 * A member that fails to arm is not fatal — the split shrinks to what came up — but the
 * shrunken count must STILL DIVIDE BOTH quantities, so it walks back down to one that does
 * rather than leaving a ragged set. Both facts are reported, because a silent shrink shows up
 * only as a number that did not move.
 */
static void gdnc_arm_members(pn_tt_device *device, const ft_tt_op_gdnc_request *shape)
{
    const uint32_t columns = device->gdn_shape.conv_dim / PN_TT_GDNC_TILE_DIM;
    const uint32_t heads = device->gdn_shape.v_heads;
    const uint32_t asked = pn_tt_env_u32(PN_TT_GDNC_CORES_ENV, PN_TT_GDNC_CORES_DEFAULT);
    const uint32_t members = gdnc_member_count(device);
    const uint32_t first = pn_tt_env_u32(PN_TT_GDNC_MEMBER_CORE_FIRST_ENV,
                                         PN_TT_GDNC_MEMBER_CORE_DEFAULT);
    uint32_t up = 1u;
    uint32_t armed;
    uint32_t index;

    device->gdnc_members = 1u;
    /* A COUNT THAT WAS SILENTLY REDUCED IS A KNOB THAT DID NOT ARM. `PN_TT_GDNC_CORES=3` runs
     * a 2-way split and `=99` runs a 16-way one, and a sweep that did not print this would
     * report two rungs as three. It prints only when the two differ, so the shipped default
     * adds no line. */
    if (asked != members)
        fprintf(stderr, "pn_tt: gdnc CORE SPLIT — requested %u members, running %u: the count "
                        "is clamped to %u and then shrunk until it divides BOTH %u tile "
                        "columns and %u v-heads\n",
                asked, members, PN_TT_GDNC_MAX_MEMBERS, columns, heads);
    if (members < 2u)
        return;
    for (index = 0; index + 1u < members; index++) {
        pn_tt_residency *member = &device->gdnc_member[index];
        ft_tt_launch_cb_entry placed[FT_TT_OP_GDNC_CB_COUNT];
        ft_tt_op_gdnc_request request = *shape;

        request.phase = FT_TT_OP_GDNC_PHASE_CONV;
        request.first_col = 0u;
        request.col_count = 0u;
        request.activation_in.tile_bytes = PN_TT_GDNC_TILE_BYTES;
        request.activation_out.tile_bytes = PN_TT_GDNC_TILE_BYTES;
        if (gdnc_member_core(index, first) == UINT32_MAX ||
            !pn_tt_prepare_family_on_core(device, member, PN_TT_GDNC_PROGRAM, "gdnc-member",
                                          PN_TT_GDNC_MEMBER_CORE_FIRST_ENV,
                                          gdnc_member_core(index, first),
                                          PN_TT_ONCARD_TRIAD_MASK))
            break;
        /* BEFORE THE START, because after it the binding is already in the kernel's runtime
         * args. The block lives on MEMBER 0's core, which is the one the aggregated read
         * addresses; a member's own slot is its index.
         *
         * RED PROOF: `gdnc_group_mute` leaves the LAST member unbound, so it runs its
         * descriptors correctly and posts NOTHING. It is the arm that proves the group check
         * reads every slot rather than only member 0's — the recorded twenty-eighth class one
         * level in, where a shape that aggregated nothing would report every figure a working
         * one reports. The await must TIME OUT naming that member, the shape must disarm, and
         * the run must then finish `cmp` IDENTICAL at the per-member cost. */
        if (device->gdnc_group &&
            !(pn_tt_red("gdnc_group_mute") && index + 2u == members))
            pn_tt_residency_join_group(member, device->gdnc.core_x, device->gdnc.core_y,
                                       PN_TT_GDNC_GROUP_ADDRESS, index + 1u);
        if (ft_tt_op_gdnc_place_cbs(&request, placed, member->cb_buffer,
                                    sizeof(member->cb_buffer), &member->cb_mask) != FT_TT_OK ||
            !pn_tt_start_family(member)) {
            pn_tt_residency_release(member);
            break;
        }
        up++;
    }
    /* A PARTIAL SET IS NOT A SMALLER SPLIT unless the smaller number still divides both. The
     * members that came up past the legal count are STOPPED rather than left resident, so a
     * later token cannot find a residency nothing addresses and no core is held by a triad
     * this host will never feed. */
    armed = up;
    while (armed > 1u && ((columns % armed) != 0u || (heads % armed) != 0u))
        armed--;
    for (index = armed; index < up; index++) {
        pn_tt_residency *member = &device->gdnc_member[index - 1u];

        if (member->running)
            pn_tt_residency_stop(member);
        pn_tt_residency_release(member);
    }
    device->gdnc_members = armed;
    if (armed != members)
        fprintf(stderr, "pn_tt: gdnc CORE SPLIT — asked for %u members, %u came up, %u divide "
                        "both %u columns and %u v-heads; the split runs at %u\n",
                members, up, armed, columns, heads, armed);
    if (armed < 2u) {
        fprintf(stderr, "pn_tt: gdnc CORE SPLIT — NOT SPLIT; the family runs whole on core "
                        "%u\n", device->gdnc.core_index);
        /* Member 0 armed the post on a count that then did not come up, so it has already
         * paid the `line` -> `window` demotion for a group that does not exist. Nothing can
         * undo that this run — the binding is in the kernel — but the SHAPE is cleared so
         * `gdnc_group_ops` cannot read as armed-and-idle when the truth is not-a-group. */
        device->gdnc_group = false;
        return;
    }
    fprintf(stderr, "pn_tt: gdnc CORE SPLIT — every descriptor over %u members, core %u then "
                    "%u ... %u; PHASE CONV %u of %u tile columns each, PHASE DELTA %u of %u "
                    "v-heads each\n",
            armed, device->gdnc.core_index, gdnc_member_core(0u, first),
            gdnc_member_core(armed - 2u, first), columns / armed,
            columns, heads / armed, heads);

    /* THE GROUP BLOCK, WRITTEN AFTER THE MEMBER COUNT IS FINAL AND BEFORE ANY DESCRIPTOR IS
     * ISSUED. The matmul group writes its block before any member is kicked; here the count
     * is not known until the last member has either come up or failed, and the ordering is
     * still safe for the two reasons that mattered there: no descriptor is in flight during
     * arming, so nothing can post into an un-magicked block, and `ft_tt_queue_group_init()`
     * ZEROES every slot, so a sequence a previous residency left cannot be read as this
     * run's.
     *
     * AND THE MAGIC IS CHECKED ONCE, HERE. What it catches is the WRONG ADDRESS, which is a
     * constant of the run and not a per-poll risk — re-reading the header on every poll is
     * exactly the cost the slots-only read exists to remove. A failure DISARMS the shape
     * rather than aborting: the per-member await is still correct, just slower, and a family
     * that cannot read its own block must not then spin on it for a timeout. */
    if (device->gdnc_group) {
        ft_tt_queue_group_slot slots[PN_TT_GDNC_MAX_MEMBERS];

        if (ft_tt_queue_group_init(&device->runtime.noc, device->gdnc.core_x,
                                   device->gdnc.core_y, PN_TT_GDNC_GROUP_ADDRESS,
                                   armed) != FT_TT_OK ||
            ft_tt_queue_group_read(&device->runtime.noc, device->gdnc.core_x,
                                   device->gdnc.core_y, PN_TT_GDNC_GROUP_ADDRESS, armed,
                                   slots) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: gdnc GROUP COMPLETION — the block at 0x%08x on core %u "
                            "did not write and read back with its magic. DISARMED; the split "
                            "keeps the per-member await.\n",
                    (unsigned)PN_TT_GDNC_GROUP_ADDRESS, device->gdnc.core_index);
            device->gdnc_group = false;
            return;
        }
        fprintf(stderr, "pn_tt: gdnc GROUP COMPLETION — %u members post into one block at "
                        "0x%08x on core %u; the await is member 0's count plus ONE %u-byte "
                        "read, not %u poll loops\n",
                armed, (unsigned)PN_TT_GDNC_GROUP_ADDRESS, device->gdnc.core_index,
                (unsigned)((armed * 8u + 15u) & ~15u), armed);
    }
}

int pn_tt_device_arm_gdnc(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDNC_CB_COUNT];
    ft_tt_op_gdnc_request conv;
    ft_tt_op_gdnc_request delta;
    const char *want;

    if (!device)
        return -1;
    /* ON BY DEFAULT: only an explicit `0` declines. See PN_TT_GDNC_ENV for the gate that
     * bought this default and for why the chunk length is 16 and not 32. */
    want = getenv(PN_TT_GDNC_ENV);
    if (want && *want && atoi(want) == 0)
        return 0;
    /* IT OWNS NO CARRIED STATE. The window and the recurrent state it advances belong to the
     * decode family, which is what makes the prefill->decode seam free — so without that
     * family there is nothing for this one to carry. */
    if (!pn_tt_gdn_is_armed(device)) {
        fprintf(stderr, "pn_tt: %s needs the gdn family armed — the chunkwise mixers share "
                        "its conv window and recurrent state byte for byte. Add `gdn` to "
                        "PN_TT_ONCARD.\n", PN_TT_GDNC_ENV);
        return 0;
    }

    conv = gdnc_request(device, FT_TT_OP_GDNC_PHASE_CONV, PN_TT_GDNC_TILE_DIM);
    delta = gdnc_request(device, FT_TT_OP_GDNC_PHASE_DELTA, PN_TT_GDNC_TILE_DIM);
    /* The placement needs operands only to pass the encoder's "is this set" test. */
    conv.activation_in.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    conv.activation_out.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    if (!ft_tt_op_gdnc_shape_is_supported(&conv) || !ft_tt_op_gdnc_shape_is_supported(&delta)) {
        fprintf(stderr, "pn_tt: the gdnc family does not serve S=%u, %u v-heads over %u "
                        "k-heads, conv_dim %u, %u taps\n",
                device->gdn_shape.state_size, device->gdn_shape.v_heads,
                device->gdn_shape.k_heads, device->gdn_shape.conv_dim,
                device->gdn_shape.conv_taps);
        return 0;
    }

    if (!pn_tt_prepare_family(device, &device->gdnc, PN_TT_GDNC_PROGRAM, "gdnc",
                              PN_TT_CORE_GDNC_ENV, PN_TT_CORE_GDNC_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return 0;
    if (ft_tt_op_gdnc_place_cbs(&conv, placed, device->gdnc.cb_buffer,
                                sizeof(device->gdnc.cb_buffer),
                                &device->gdnc.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdnc circular buffers do not fit at S=%u, %u taps\n",
                device->gdn_shape.state_size, device->gdn_shape.conv_taps);
        pn_tt_residency_release(&device->gdnc);
        return 0;
    }
    if (gdnc_alloc_regions(device, &conv, &delta) != 0) {
        fprintf(stderr, "pn_tt: the gdnc family could not be staged in card DRAM\n");
        pn_tt_residency_release(&device->gdnc);
        /* A HALF-ALLOCATED table is worse than none: a later reader would find addresses for
         * some layers and zeros for the rest. The DRAM is not reclaimed — the allocator frees
         * LIFO by design — and a caller that reaches here is not continuing on this path. */
        free(device->gdnc_constant);
        device->gdnc_constant = NULL;
        return 0;
    }
    /* THE COMPLETION SHAPE IS DECIDED BEFORE MEMBER 0 STARTS, because the binding rides in
     * the runtime args and there is no second chance to set it. It is decided on the count
     * this run WILL arm rather than on the knob alone: a family that does not split has no
     * group, and arming the post anyway would cost it the `line` await shape for nothing. */
    {
        const char *ask = getenv(PN_TT_GDNC_GROUP_ENV);

        device->gdnc_group = ask && strcmp(ask, PN_TT_GDNC_GROUP_ACK) == 0 &&
                             gdnc_member_count(device) >= 2u;
        /* A `1` IS REFUSED AND SAYS SO. Silently declining would make an arm that thought it
         * measured the group shape measure the per-member one and report it as the group's —
         * which is the shape of every vacuous gate this backend has paid for. */
        if (ask && *ask && strcmp(ask, PN_TT_GDNC_GROUP_ACK) != 0 && strcmp(ask, "0") != 0)
            fprintf(stderr, "pn_tt: %s=%s is REFUSED. The group post is an accelerant of a "
                            "defect that has wedged two cores at instrn_buf_status 0x66 and "
                            "costs a whole-card `ft_tt_core recover --rung 3`. Spell it "
                            "`%s` if that is what you mean. The split keeps the per-member "
                            "await.\n",
                    PN_TT_GDNC_GROUP_ENV, ask, PN_TT_GDNC_GROUP_ACK);
    }
    device->gdnc_sequence = 0u;
    if (device->gdnc_group)
        pn_tt_residency_join_group(&device->gdnc, device->gdnc.core_x, device->gdnc.core_y,
                                   PN_TT_GDNC_GROUP_ADDRESS, 0u);
    if (!pn_tt_start_family(&device->gdnc))
        return 0;
    /* AFTER member 0 is up, so a member cannot be armed onto a core member 0 is about to take
     * and so `pn_tt_core_holder()` sees the whole set as it grows. */
    gdnc_arm_members(device, &conv);
    return 0;
}

/**
 * One phase's line: the CLEAN per-descriptor figure, and the recovery that is not in it.
 *
 * `rearms` prints even at 0 — a contaminated figure that looks like a clean one is the whole
 * defect this split exists for, and an absent count reads as a clean run. Silent when the
 * phase ran nothing at all.
 */
static void gdnc_report_phase(const pn_tt_device *device, const char *name, uint32_t phase)
{
    const uint64_t ops = device->gdnc_phase_ops[phase];
    const uint64_t recovered = device->gdnc_phase_recovered_ops[phase];

    if (ops == 0u && recovered == 0u)
        return;
    fprintf(stderr, "pn_tt: gdnc — %-5s %.4f s over %llu clean descriptors (%.1f us each), "
                    "%llu replayed over %llu re-arms (%.4f s)\n",
            name, device->gdnc_phase_seconds[phase], (unsigned long long)ops,
            ops ? 1e6 * device->gdnc_phase_seconds[phase] / (double)ops : 0.0,
            (unsigned long long)recovered,
            (unsigned long long)device->gdnc_phase_rearms[phase],
            device->gdnc_phase_recovered_seconds[phase]);
}

/**
 * THE STAGING LINE — the term that was inside `sequential` and outside every phase counter.
 *
 * `pn_tt_gdnc_delta_oncard_at()` transposes `v_heads` decay runs into ratio tiles and gate
 * runs into the flat gate block, then writes both to DRAM, all BEFORE `gdnc_submit()` marks
 * its start. So the DELTA line's "us each" has never contained it, and the prefill report's
 * `sequential` term has always contained it un-named. This prints it.
 *
 * BUILD AND UPLOAD ARE SEPARATE CLOCKS ON PURPOSE. They answer different questions: build is
 * host arithmetic on a two-core box and upload is the x1 Gen2 link, and a smaller operand
 * moves the second and not necessarily the first. Reporting one pooled number would make a
 * lever that halves the bytes look like it half-worked.
 *
 * Silent when nothing staged, for the same reason `gdnc_report_phase()` is — except that
 * silence here is itself readable, because a run that issued DELTA descriptors and printed
 * no STAGE line is a run where this bracket did not arm. The runner scores on the token this
 * function EMITS (the recorded thirtieth vacuous-gate class), so the DELTA descriptor count
 * is printed BESIDE the stage count and the two must agree.
 */
static void gdnc_report_stage(const pn_tt_device *device)
{
    const uint64_t ops = device->gdnc_stage_ops;
    const double total = device->gdnc_stage_build_seconds + device->gdnc_stage_upload_seconds;
    /* DECIMAL MB, because every link rate this program has ever recorded (200-390 MB/s, the
     * 286.3 PIO figure) is decimal and a MiB here would read as a 4.9% link regression. */
    const double megabytes = (double)device->gdnc_stage_bytes / 1e6;

    if (ops == 0u)
        return;
    fprintf(stderr, "pn_tt: gdnc — STAGE %.4f s over %llu stagings (%.1f us each) = "
                    "%.4f s build (%.1f us) + %.4f s upload (%.1f us); %.1f MB at %.1f MB/s; "
                    "DELTA descriptors %llu clean + %llu replayed\n",
            total, (unsigned long long)ops, 1e6 * total / (double)ops,
            device->gdnc_stage_build_seconds,
            1e6 * device->gdnc_stage_build_seconds / (double)ops,
            device->gdnc_stage_upload_seconds,
            1e6 * device->gdnc_stage_upload_seconds / (double)ops,
            megabytes,
            device->gdnc_stage_upload_seconds > 0.0
                ? megabytes / device->gdnc_stage_upload_seconds : 0.0,
            (unsigned long long)device->gdnc_phase_ops[FT_TT_OP_GDNC_PHASE_DELTA],
            (unsigned long long)device->gdnc_phase_recovered_ops[FT_TT_OP_GDNC_PHASE_DELTA]);
    /* THE TWO P3 STAGING LEVERS, as counts the arm cannot fake: how many DRAM writes the
     * staging made (2 a staging when full; more, smaller ones when diffed) and how many
     * stagings ran inside a split's card time. */
    fprintf(stderr, "pn_tt: gdnc — STAGE SHAPE %llu DRAM writes (%.2f a staging), %llu of %llu "
                    "stagings hidden under a descriptor\n",
            (unsigned long long)device->gdnc_stage_writes,
            (double)device->gdnc_stage_writes / (double)ops,
            (unsigned long long)device->gdnc_stage_hidden_ops, (unsigned long long)ops);
    if (device->gdnc_stage_verify_checks != 0u)
        fprintf(stderr, "pn_tt: gdnc — STAGE VERIFY %llu units differ from the builders over "
                        "%llu read-backs\n",
                (unsigned long long)device->gdnc_stage_verify_mismatches,
                (unsigned long long)device->gdnc_stage_verify_checks);
}

void pn_tt_gdnc_release(pn_tt_device *device)
{
    if (!device)
        return;
    /* WHAT THE TWO PHASES COST, PER DESCRIPTOR, measured directly rather than subtracted
     * between two runs. `oncard_seconds` is a sum over every family and a subtraction cannot
     * tell CONV from DELTA — which are the two things a re-pricing has to separate, because
     * one is flat in the chunk size and the other divides by head. */
    gdnc_report_phase(device, "CONV", FT_TT_OP_GDNC_PHASE_CONV);
    gdnc_report_phase(device, "DELTA", FT_TT_OP_GDNC_PHASE_DELTA);
    gdnc_report_stage(device);
    /* THE SPLIT'S OWN NUMBERS, one line per member, printed only when a split ran. `ops`
     * prints beside `seconds` so a member whose clock never started is `ops != 0 &&
     * seconds == 0` and reads as the recorded twenty-eighth vacuous-gate class rather than as
     * a free member. */
    /* THE COMPLETION SHAPE AS A NUMBER, NOT AS AN ABSENCE. A run that armed the group prints
     * one await for the whole set; a run that did not prints N. Printing the shape's own
     * clock beside the member rows is what stops the member table from being read as if both
     * shapes produced it: on the group shape, member 0's row IS the group await and members
     * 1..N-1 are silent by construction. */
    if (device->gdnc_group_ops != 0u)
        fprintf(stderr, "pn_tt: gdnc — GROUP %.4f s over %llu group awaits (%.1f us each) "
                        "across %u members; the per-member await did not run\n",
                device->gdnc_group_seconds, (unsigned long long)device->gdnc_group_ops,
                1e6 * device->gdnc_group_seconds / (double)device->gdnc_group_ops,
                device->gdnc_members);
    else if (device->gdnc_group)
        fprintf(stderr, "pn_tt: gdnc — GROUP armed and never used: 0 group awaits\n");
    if (device->gdnc_members > 1u) {
        uint32_t index;

        for (index = 0; index < device->gdnc_members; index++) {
            const uint64_t ops = device->gdnc_member_ops[index];
            const double seconds = device->gdnc_member_seconds[index];
            const uint32_t core = index == 0u ? device->gdnc.core_index
                                              : device->gdnc_member[index - 1u].core_index;

            fprintf(stderr, "pn_tt: gdnc split member %u core %3u — %.4f s of await over "
                            "%llu descriptors (%.1f us each)\n",
                    index, core, seconds, (unsigned long long)ops,
                    ops ? seconds * 1e6 / (double)ops : 0.0);
        }
    }
    {
        uint32_t index;

        for (index = 0; index + 1u < device->gdnc_members; index++) {
            pn_tt_residency *member = &device->gdnc_member[index];

            if (member->running)
                pn_tt_residency_stop(member);
            pn_tt_residency_release(member);
        }
        device->gdnc_members = 0u;
    }
    if (device->gdnc.running)
        pn_tt_residency_stop(&device->gdnc);
    pn_tt_residency_release(&device->gdnc);
    free(device->gdnc_constant);
    free(device->gdnc_constant_replica);
    device->gdnc_constant_replica = NULL;
    device->gdnc_constant_replicas = 1u;
    free(device->gdnc_decay_stage);
    free(device->gdnc_gate_stage);
    pn_tt_stage_pipeline_stop(&device->gdnc_stage_pipeline);
    for (uint32_t slot = 0; slot < PN_TT_GDNC_STAGE_SLOTS; slot++) {
        free(device->gdnc_shadow_decay[slot]);
        free(device->gdnc_shadow_gate[slot]);
        device->gdnc_shadow_decay[slot] = NULL;
        device->gdnc_shadow_gate[slot] = NULL;
        device->gdnc_shadow_valid[slot] = false;
        device->gdnc_prestaged[slot].valid = false;
    }
    device->gdnc_overlap_fn = NULL;
    device->gdnc_overlap_aux_fn = NULL;
    device->gdnc_constant = NULL;
    device->gdnc_decay_stage = NULL;
    device->gdnc_gate_stage = NULL;
}

/* ================================================================
 * The conv constant — the TRANSPOSE, ROW-REPLICATED, and the band behind it
 * ================================================================ */

int pn_tt_gdnc_conv_weight_upload(pn_tt_device *device, uint32_t layer, const float *weight)
{
    const uint32_t taps = device->gdn_shape.conv_taps;
    const uint32_t cols = device->gdn_shape.conv_dim / PN_TT_GDNC_TILE_DIM;
    /* ONE TILE COLUMN AT A TIME, not one `[32, conv_dim]` strip: every row of a plane carries
     * the same `taps`-strided gather, so a tile is a 32x32 plane and the whole layer never
     * needs a 1 MiB row-major staging buffer on a two-core, 7 GiB host. */
    const bool row0 = pn_tt_red("gdnc_weight_row0");
    float plane[PN_TT_GDNC_TILE_DIM * PN_TT_GDNC_TILE_DIM];
    unsigned char *tiles;
    unsigned char *band;
    uint32_t tap;
    uint32_t copy;
    int rc = 0;

    if (!pn_tt_gdnc_is_armed(device) || !weight || layer >= device->gdn_shape.layer_count)
        return -1;
    tiles = malloc((size_t)cols * PN_TT_GDNC_TILE_BYTES);
    band = malloc(ft_tt_op_gdnc_band_bytes(taps));
    if (!tiles || !band) {
        free(tiles);
        free(band);
        return -1;
    }

    for (tap = 0; tap < taps && rc == 0; tap++) {
        uint32_t column;

        for (column = 0; column < cols; column++) {
            uint32_t row;
            uint32_t lane;

            memset(plane, 0, sizeof(plane));
            /* THE ROW REPLICATION, and the whole of it: `w[c][tap]` at EVERY row, because the
             * tap coefficient is constant across the chunk's positions and the broadcast
             * surface is BROADCAST_COL only — there is no row broadcast.
             * RED PROOF: `gdnc_weight_row0` writes row 0 alone, which is the DECODE family's
             * plane. It is right at row 0 of every chunk and zero at rows 1..C-1, so it is a
             * mutation a single-POSITION comparison cannot see. */
            for (row = 0; row < (row0 ? 1u : PN_TT_GDNC_TILE_DIM); row++)
                for (lane = 0; lane < PN_TT_GDNC_TILE_DIM; lane++)
                    plane[row * PN_TT_GDNC_TILE_DIM + lane] =
                        weight[((size_t)column * PN_TT_GDNC_TILE_DIM + lane) * taps + tap];
            if (ft_tt_tilize_strip(plane, sizeof(float), PN_TT_GDNC_TILE_DIM,
                                   PN_TT_GDNC_TILE_DIM,
                                   (size_t)PN_TT_GDNC_TILE_DIM * sizeof(float),
                                   tiles + (size_t)column * PN_TT_GDNC_TILE_BYTES) !=
                FT_TT_TILIZE_OK) {
                rc = -1;
                break;
            }
        }
        /* RED `gdnc_replica_zero`: the copies' weight planes are ZEROED, not written (not
         * merely left unwritten: DRAM keeps the previous process's copies, which made an
         * "unwritten" red VACUOUS on silicon) — must DIFFER: the members read their copy. */
        for (copy = 0; rc == 0 && copy < device->gdnc_constant_replicas; copy++)
            rc = copy != 0u && pn_tt_red("gdnc_replica_zero")
                     ? pn_tt_zero_dram(device, gdnc_constant_copy(device, layer, copy)->bank,
                                       gdnc_constant_copy(device, layer, copy)->address +
                                           (size_t)tap * cols * PN_TT_GDNC_TILE_BYTES,
                                       (size_t)cols * PN_TT_GDNC_TILE_BYTES)
                     : pn_tt_write_dram(device, gdnc_constant_copy(device, layer, copy)->bank,
                                        gdnc_constant_copy(device, layer, copy)->address +
                                            (size_t)tap * cols * PN_TT_GDNC_TILE_BYTES,
                                        tiles, (size_t)cols * PN_TT_GDNC_TILE_BYTES);
    }

    /* THE BAND RIDES AT THE END OF THE WEIGHT PLANES, at tile `taps * conv_tile_cols`. It is a
     * `taps`-dependent CONSTANT — the same eight tiles in every layer of every model with this
     * tap count — and it rides here rather than in a sixth DRAM endpoint because
     * `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family. */
    if (rc == 0 && ft_tt_op_gdnc_build_band(taps, band, ft_tt_op_gdnc_band_bytes(taps)) !=
                       FT_TT_OK)
        rc = -1;
    for (copy = 0; rc == 0 && copy < device->gdnc_constant_replicas; copy++)
        rc = pn_tt_write_dram(device, gdnc_constant_copy(device, layer, copy)->bank,
                              gdnc_constant_copy(device, layer, copy)->address +
                                  (size_t)taps * cols * PN_TT_GDNC_TILE_BYTES,
                              band, ft_tt_op_gdnc_band_bytes(taps));
    free(tiles);
    free(band);
    return rc;
}

/* ================================================================
 * Submission
 * ================================================================ */

/**
 * Retire a residency that refused a descriptor, and arm a FRESH one on the same core.
 *
 * The DRAM is untouched: the constants are allocated once and a retired core owns none of
 * them, and the carried state belongs to the gdn family. What is re-placed is the
 * circular-buffer geometry, which is a property of the shape rather than of the descriptor.
 * A residency is BOUNDED on purpose — `idle_spin_limit` retires a kernel whose host has
 * stopped feeding it — so this fires for a serve that idles between requests as well as for a
 * malformed descriptor.
 */
static bool pn_tt_gdnc_rearm(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDNC_CB_COUNT];
    ft_tt_op_gdnc_request request = gdnc_request(device, FT_TT_OP_GDNC_PHASE_CONV,
                                                 PN_TT_GDNC_TILE_DIM);

    fprintf(stderr, "pn_tt: the gdnc residency refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", device->gdnc.core_index);
    pn_tt_residency_stop(&device->gdnc);
    pn_tt_residency_release(&device->gdnc);
    device->oncard_restarts++;
    /* NOTHING BOOTS FIRMWARE ONTO A CORE THAT DID NOT RETIRE. Measured on silicon
     * 2026-08-21: a refused descriptor leaves the compute thread spinning out its bound, the
     * stop reported `processors still running: 0x4`, and the boot that followed WEDGED core 90
     * at `instrn_buf_status = 0x66` — a state only `ft_tt_core recover --rung 3` and sudo can
     * clear. A failed descriptor is a bad token; a wedged Tensix takes the process AND the next
     * one, because the next boot lands on the same core. */
    if (!pn_tt_residency_retired(&device->gdnc)) {
        fprintf(stderr, "pn_tt: the gdnc residency did not retire — REFUSING to arm a fresh "
                        "one on core %u\n", device->gdnc.core_index);
        return false;
    }

    request.activation_in.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    if (!pn_tt_prepare_family(device, &device->gdnc, PN_TT_GDNC_PROGRAM, "gdnc",
                              PN_TT_CORE_GDNC_ENV, PN_TT_CORE_GDNC_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_gdnc_place_cbs(&request, placed, device->gdnc.cb_buffer,
                                sizeof(device->gdnc.cb_buffer),
                                &device->gdnc.cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(&device->gdnc);
        return false;
    }
    /* THE GROUP BINDING IS RE-APPLIED, because `pn_tt_residency_release()` zeroes the handle
     * and a fresh residency posts nothing. A member that came back silently un-bound would
     * make every LATER group await time out — a whole run lost to one recovered descriptor,
     * and the symptom would point at the group block rather than at the re-arm. */
    if (device->gdnc_group)
        pn_tt_residency_join_group(&device->gdnc, device->gdnc.core_x, device->gdnc.core_y,
                                   PN_TT_GDNC_GROUP_ADDRESS, 0u);
    return pn_tt_start_family(&device->gdnc);
}

/**
 * Retire ONE split member and arm a fresh one ON THE CORE IT ALREADY HELD.
 *
 * The core is captured BEFORE the release, because `pn_tt_residency_release()` zeroes the
 * handle — re-reading `core_index` afterwards would arm member 3 on core 0. Member 0 is the
 * `gdnc` residency itself and goes through `pn_tt_gdnc_rearm()`, which is the same operation
 * spelled against the environment default.
 */
static bool gdnc_member_rearm(pn_tt_device *device, uint32_t member_index)
{
    pn_tt_residency *member;
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDNC_CB_COUNT];
    ft_tt_op_gdnc_request request;
    uint32_t core;

    if (member_index == 0u)
        return pn_tt_gdnc_rearm(device);
    member = &device->gdnc_member[member_index - 1u];
    core = member->core_index;
    fprintf(stderr, "pn_tt: gdnc member %u refused a descriptor — retiring it by ASKING and "
                    "arming a fresh one on core %u\n", member_index, core);
    pn_tt_residency_stop(member);
    pn_tt_residency_release(member);
    device->oncard_restarts++;
    /* The same refusal as `pn_tt_gdnc_rearm()`'s, for the same reason. */
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: gdnc member %u did not retire — REFUSING to arm a fresh one "
                        "on core %u\n", member_index, core);
        return false;
    }

    request = gdnc_request(device, FT_TT_OP_GDNC_PHASE_CONV, PN_TT_GDNC_TILE_DIM);
    request.activation_in.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    request.activation_out.tile_bytes = PN_TT_GDNC_TILE_BYTES;
    if (!pn_tt_prepare_family_on_core(device, member, PN_TT_GDNC_PROGRAM, "gdnc-member",
                                      PN_TT_GDNC_MEMBER_CORE_FIRST_ENV, core,
                                      PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_gdnc_place_cbs(&request, placed, member->cb_buffer, sizeof(member->cb_buffer),
                                &member->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(member);
        return false;
    }
    /* The same re-binding, for the same reason as `pn_tt_gdnc_rearm()`'s, and on the SLOT
     * this member already held — the index, like the core, is captured before the release. */
    if (device->gdnc_group)
        pn_tt_residency_join_group(member, device->gdnc.core_x, device->gdnc.core_y,
                                   PN_TT_GDNC_GROUP_ADDRESS, member_index);
    return pn_tt_start_family(member);
}

/** The whole descriptor's slice count for a phase: CONV owns tile columns, DELTA owns
 *  v-heads. Both are addressed through the SAME two request words, which is why one function
 *  answers for both and why the split count has to divide them both. */
static uint32_t gdnc_split_total(const ft_tt_op_gdnc_request *request)
{
    if (request->phase == FT_TT_OP_GDNC_PHASE_DELTA)
        return request->v_heads;
    return request->conv_dim / PN_TT_GDNC_TILE_DIM;
}

/**
 * ONE descriptor over a core split: one slice per member, ALL ENQUEUED BEFORE ANY IS AWAITED.
 *
 * The two loops being separate is the measurement and not a style. Enqueue-wait per member
 * would run `members` SERIAL descriptors and print the flat line "it does not parallelise",
 * produced by this function rather than by the card — the failure the decode family's core
 * scaling probe names, and the reason its loops are split the same way.
 *
 * NO TWO MEMBERS TOUCH THE SAME DRAM BYTE. PHASE CONV: column `c` reads only the input tile,
 * window planes and weight planes AT COLUMN `c` (`ftgc_plane_tile()` is `tap*conv_tile_cols +
 * column`) and writes only column `c`'s conv_out tile and its own window rows. The band tiles
 * are the one shared operand and they are read-only constants. PHASE DELTA: the carried state
 * is head-major and `y` is scattered at row granularity, which is the decode family's own
 * argument. `foundry/tt/tests/test_phase6_gdnc.c` proves the CONV case on silicon — four
 * column slices byte-for-byte against one whole descriptor.
 */
/**
 * Run the one-shot overlap hook, if any, and return the seconds it took.
 *
 * Called between a split's enqueue and its await, so host staging runs while the card works.
 * Its seconds are RETURNED so the caller can keep them OUT of the phase clock — the stage line
 * already books them, and the G0 attribution must not count one second twice. A failed hook
 * is remembered in `*failed` and the await still runs: an enqueued descriptor is always
 * awaited, or that member's cursor desynchronises for the rest of the run.
 */
static double gdnc_run_overlap(pn_tt_device *device, int *failed)
{
    int (*fn)(void *) = device->gdnc_overlap_fn;
    void *context = device->gdnc_overlap_context;
    int (*aux)(void *) = device->gdnc_overlap_aux_fn;
    void *aux_context = device->gdnc_overlap_aux_context;
    double at;

    if (!fn && !aux)
        return 0.0;
    device->gdnc_overlap_fn = NULL;
    device->gdnc_overlap_context = NULL;
    device->gdnc_overlap_aux_fn = NULL;
    device->gdnc_overlap_aux_context = NULL;
    at = pn_tt_now_seconds();
    /* A FAILED HOOK IS NOT A FAILED DESCRIPTOR. The hook only prestages; a slot it did not
     * stage is staged inline by its own descriptor, exactly as with the overlap off. */
    if (fn) {
        if (fn(context) == 0)
            device->gdnc_stage_hidden_ops++;
        else
            fprintf(stderr, "pn_tt: a gdnc overlap hook failed — that slot stages inline\n");
    }
    /* The aux slot's work is a READ its owner redoes inline when it did not happen. */
    if (aux)
        (void)aux(aux_context);
    (void)failed;
    return pn_tt_now_seconds() - at;
}

void pn_tt_gdnc_set_overlap(pn_tt_device *device, int (*fn)(void *context), void *context)
{
    if (!device)
        return;
    device->gdnc_overlap_fn = fn;
    device->gdnc_overlap_context = context;
}

void pn_tt_gdnc_set_overlap_aux(pn_tt_device *device, int (*fn)(void *context), void *context)
{
    if (!device)
        return;
    device->gdnc_overlap_aux_fn = fn;
    device->gdnc_overlap_aux_context = context;
}

static int gdnc_submit_split(pn_tt_device *device, const ft_tt_op_gdnc_request *request)
{
    const uint32_t members = device->gdnc_members;
    const uint32_t per_member = gdnc_split_total(request) / members;
    uint32_t words[PN_TT_GDNC_MAX_MEMBERS][FT_TT_OP_GDNC_WORDS];
    pn_tt_residency *set[PN_TT_GDNC_MAX_MEMBERS];
    bool bad[PN_TT_GDNC_MAX_MEMBERS];
    uint64_t rearms = 0;
    double started;
    double hook_seconds = 0.0;
    int hook_failed = 0;
    uint32_t index;
    int failed = 0;

    memset(bad, 0, sizeof(bad));
    for (index = 0; index < members; index++) {
        ft_tt_op_gdnc_request sliced = *request;

        sliced.first_col = index * per_member;
        sliced.col_count = per_member;
        if (request->phase == FT_TT_OP_GDNC_PHASE_CONV && device->gdnc_constant_replicas > 1u)
            sliced.constant = gdnc_operand(
                device, gdnc_constant_copy(device, device->gdnc_conv_layer,
                                           index % device->gdnc_constant_replicas));
        if (ft_tt_op_gdnc_parameters(&sliced, words[index]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the gdnc phase-%u slice [%u, %u) is one the family "
                            "refuses\n", sliced.phase, sliced.first_col,
                    sliced.first_col + sliced.col_count);
            return -1;
        }
        set[index] = index == 0u ? &device->gdnc : &device->gdnc_member[index - 1u];
    }

    started = pn_tt_now_seconds();
    if (device->gdnc_group) {
        uint32_t token[PN_TT_GDNC_MAX_MEMBERS];
        const uint32_t sequence = ++device->gdnc_sequence;
        double at;

        memset(token, 0, sizeof(token));
        for (index = 0; index < members; index++)
            if (pn_tt_residency_enqueue_group(set[index], FT_TT_QUEUE_OP_GDN_CHUNK,
                                              words[index], 1u, sequence, &token[index]) != 0)
                failed = bad[index] = 1;
        hook_seconds = gdnc_run_overlap(device, &hook_failed);
        /* ONE AWAIT FOR THE WHOLE SET. It is booked against MEMBER 0's row so `us each` stays
         * the per-descriptor figure it has always been on both shapes — and the group's own
         * clock is printed separately, so a reader can tell which shape produced it rather
         * than inferring it from a number that moved. */
        at = pn_tt_now_seconds();
        if (!failed && pn_tt_residency_await_group(set, token, members, sequence,
                                                   PN_TT_OP_TIMEOUT_MS) != 0) {
            failed = 1;
            /* FAIL-SAFE, NOT FAIL-SLOW. A group await that did not satisfy has cost one
             * timeout; leaving the shape armed would cost one PER DESCRIPTOR for the rest of
             * the prompt — 3,672 of them — and the per-member await is still CORRECT, just
             * slower. So the shape retires itself after the first failure and says so. This
             * is also what bounds `PN_TT_RED=gdnc_group_mute`, which is otherwise a red arm
             * that takes hours to report. */
            fprintf(stderr, "pn_tt: gdnc GROUP COMPLETION FAILED at sequence %u — DISARMING "
                            "the shape. The rest of this run awaits per member, which is the "
                            "same answer at the pre-group cost.\n", sequence);
            device->gdnc_group = false;
        }
        device->gdnc_member_seconds[0] += pn_tt_now_seconds() - at;
        device->gdnc_member_ops[0] += 1u;
        device->gdnc_group_seconds += pn_tt_now_seconds() - at;
        device->gdnc_group_ops += 1u;
        /* WHICH MEMBER, ASKED ONLY ONCE THE SET HAS ALREADY FAILED. The group read says the
         * set is not satisfied and names the first member that is not; the re-arm below needs
         * to know about ALL of them, and one per-member await each is free on a path that is
         * already returning an error. This is also what keeps every member's cursor in step:
         * an enqueue that reached the ring has a descriptor the card will run whatever this
         * host does next, and leaving it un-awaited desynchronises that member for the rest
         * of the run. */
        if (failed)
            for (index = 0; index < members; index++)
                if (pn_tt_residency_await(set[index], PN_TT_OP_TIMEOUT_MS) != 0)
                    bad[index] = 1;
    } else {
        for (index = 0; index < members; index++)
            if (pn_tt_residency_enqueue(set[index], FT_TT_QUEUE_OP_GDN_CHUNK, words[index],
                                        1u) != 0)
                failed = bad[index] = 1;
        hook_seconds = gdnc_run_overlap(device, &hook_failed);
        /* EVERY member is awaited even after one has failed. An enqueue that got as far as the
         * ring has a descriptor the card will run whatever this host does next, and leaving it
         * un-awaited desynchronises that member's cursor for the rest of the run. */
        for (index = 0; index < members; index++) {
            const double at = pn_tt_now_seconds();

            if (pn_tt_residency_await(set[index], PN_TT_OP_TIMEOUT_MS) != 0)
                failed = bad[index] = 1;
            device->gdnc_member_seconds[index] += pn_tt_now_seconds() - at;
            device->gdnc_member_ops[index] += 1u;
        }
    }

    /* PER-MEMBER RE-ARM AND REPLAY, only where this function would otherwise return -1.
     * Replaying ONE member's slice is the same computation because a refused descriptor leaves
     * no circular-buffer traffic behind it — the reader aborts before the rebase — and because
     * the slices are disjoint in DRAM. A member that succeeded is not touched. */
    if (failed) {
        failed = 0;
        for (index = 0; index < members; index++) {
            if (!bad[index])
                continue;
            rearms++;
            if (!gdnc_member_rearm(device, index) ||
                pn_tt_residency_submit(set[index], FT_TT_QUEUE_OP_GDN_CHUNK, words[index],
                                       PN_TT_OP_TIMEOUT_MS) != 0)
                failed = 1;
        }
    }
    {
        /* THE HOOK'S SECONDS ARE THE STAGE LINE'S, not this phase's. */
        const double elapsed = pn_tt_now_seconds() - started - hook_seconds;

        device->oncard_seconds += elapsed;
        device->oncard_ops += 1u;
        /* ONE descriptor's worth of accounting for the whole set, so `us each` stays the
         * per-DESCRIPTOR figure it has always been and a split is visible as that number
         * falling rather than as the op count multiplying. */
        if (request->phase < 3u) {
            if (rearms == 0u) {
                device->gdnc_phase_seconds[request->phase] += elapsed;
                device->gdnc_phase_ops[request->phase] += 1u;
            } else {
                device->gdnc_phase_recovered_seconds[request->phase] += elapsed;
                device->gdnc_phase_recovered_ops[request->phase] += 1u;
                device->gdnc_phase_rearms[request->phase] += rearms;
            }
        }
    }
    return (failed || hook_failed) ? -1 : 0;
}

/** One descriptor, with the same re-arm-once recovery the other on-card families use. Safe
 *  for the same reason the decode family's is: every address in the parameter block is a DRAM
 *  one and the carried state is untouched by retiring the core. */
static int gdnc_submit(pn_tt_device *device, const ft_tt_op_gdnc_request *request)
{
    uint32_t words[FT_TT_OP_GDNC_WORDS];
    uint64_t rearms = 0;
    double started;
    double elapsed;

    /* THE CORE SPLIT TAKES EVERY DESCRIPTOR THIS FAMILY ISSUES, and only when this run armed
     * members for it, the count divides this DESCRIPTOR's own quantity, and the caller has not
     * already sliced. An ALREADY-SLICED request is refused rather than re-sliced: two
     * independent slicings of one stream is a subset of a subset with no error anywhere. */
    if (device->gdnc_members > 1u && request->col_count == 0u && request->first_col == 0u &&
        gdnc_split_total(request) >= device->gdnc_members &&
        (gdnc_split_total(request) % device->gdnc_members) == 0u)
        return gdnc_submit_split(device, request);

    /* No split, no window to hide it in: the hook runs first, so its work happens exactly once
     * whichever path takes the descriptor. */
    {
        int hook_failed = 0;

        (void)gdnc_run_overlap(device, &hook_failed);
        if (hook_failed)
            return -1;
    }
    if (ft_tt_op_gdnc_parameters(request, words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gdnc phase-%u descriptor at %u rows is one the family "
                        "refuses\n", request->phase, request->chunk_rows);
        return -1;
    }
    started = pn_tt_now_seconds();
    if (pn_tt_residency_submit(&device->gdnc, FT_TT_QUEUE_OP_GDN_CHUNK, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        rearms = 1u;
        if (!pn_tt_gdnc_rearm(device) ||
            pn_tt_residency_submit(&device->gdnc, FT_TT_QUEUE_OP_GDN_CHUNK, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    elapsed = pn_tt_now_seconds() - started;
    device->oncard_seconds += elapsed;
    device->oncard_ops += 1u;
    /* CLEAN AND RECOVERED KEPT APART. This bracket spans the refusal, the re-arm — a residency
     * teardown and relaunch — and the resubmit, and summing that into `seconds / ops` is
     * exactly how this phase once read **4,502 us a descriptor against 738.2**. */
    if (request->phase < 3u) {
        if (rearms == 0u) {
            device->gdnc_phase_seconds[request->phase] += elapsed;
            device->gdnc_phase_ops[request->phase] += 1u;
        } else {
            device->gdnc_phase_recovered_seconds[request->phase] += elapsed;
            device->gdnc_phase_recovered_ops[request->phase] += 1u;
            device->gdnc_phase_rearms[request->phase] += rearms;
        }
    }
    return 0;
}

/**
 * A STOPWATCH, NOT A GATE: overwrite the widened stream with the SAME uniform `[-1, 1]`
 * xorshift the kernel bench feeds, so the descriptor's cost can be measured on the bench's
 * data through the shipping path.
 *
 * It exists because the two disagree by 6.6x. `foundry/tt/tests/test_phase6_gdnc.c` measured
 * PHASE CONV at **681.4 us a descriptor** at Ornith's own conv_dim = 8192 on uniform random
 * activations; the shipping path measures **~4,480 us** on Ornith's real ones. The decode
 * family recorded exactly this shape of dependence in the other direction — 1222.7 us on real
 * activations against 4571 on a degenerate stream, suspected an SFPU slow path on
 * out-of-range SiLU input — so DATA is the named hypothesis and this is the one arm that can
 * confirm or refute it: same kernel, same shape, same traffic, same host path, different
 * numbers in the buffer.
 *
 * It produces a WRONG ANSWER by construction. The timing is what it is for.
 */
static int gdnc_conv_synthetic(pn_tt_device *device)
{
    const uint32_t tiles = device->gdn_shape.conv_dim / PN_TT_GDNC_TILE_DIM;
    float *tile = malloc(PN_TT_GDNC_TILE_BYTES);
    uint32_t state = 0x1234abcdu;
    uint32_t index;
    int rc = 0;

    if (!tile)
        return -1;
    for (index = 0; index < PN_TT_GDNC_TILE_BYTES / sizeof(float); index++) {
        state = state * 1664525u + 1013904223u;
        tile[index] = (float)((double)(state >> 8) / 8388608.0 - 1.0);
    }
    /* The same tile at every column: the question is the VALUE RANGE the SFPU sees, and a
     * per-column stream would cost 256 host writes to answer the same question. */
    for (index = 0; index < tiles && rc == 0; index++)
        rc = pn_tt_write_dram(device, device->gdn_stream_in.bank,
                              device->gdn_stream_in.address +
                                  (size_t)index * PN_TT_GDNC_TILE_BYTES,
                              tile, PN_TT_GDNC_TILE_BYTES);
    free(tile);
    return rc;
}

int pn_tt_gdnc_conv_oncard(pn_tt_device *device, uint32_t layer, uint32_t chunk_rows)
{
    ft_tt_op_gdnc_request request;

    if (!pn_tt_gdnc_is_armed(device) || layer >= device->gdn_shape.layer_count)
        return -1;
    if (pn_tt_red("gdnc_conv_synthetic") && gdnc_conv_synthetic(device) != 0)
        return -1;
    request = gdnc_request(device, FT_TT_OP_GDNC_PHASE_CONV, chunk_rows);
    request.activation_in = gdnc_operand(device, &device->gdn_stream_in);
    request.activation_out = gdnc_operand(device, &device->gdn_stream_out);
    request.constant = gdnc_operand(device, &device->gdnc_constant[layer]);
    device->gdnc_conv_layer = layer;
    /* THE WINDOW IS THE DECODE FAMILY'S BUFFER, AT ITS ADDRESS. `ftgc_plane_tile()` and
     * `ftg_conv_plane_tile()` are the same expression, which is what makes the seam free. */
    request.state_conv = gdnc_operand(device, &device->gdn_region[layer].window);

    /* RED PROOF: the carry, at the host end. Zeroing the window before every descriptor is
     * INVISIBLE IN CHUNK 0 — it starts zero at position 0 — and wrong from chunk 1 onward.
     * It mutates only card state this file owns the pointer to; no oracle reads it. */
    if (pn_tt_red("gdnc_window_wipe") &&
        pn_tt_zero_dram(device, device->gdn_region[layer].window.bank,
                        device->gdn_region[layer].window.address,
                        ft_tt_op_gdnc_conv_state_bytes(&request)) != 0)
        return -1;
    return gdnc_submit(device, &request);
}

int pn_tt_gdnc_delta_oncard(pn_tt_device *device, uint32_t layer, uint32_t chunk_rows,
                            const float *decay, const float *beta)
{
    return pn_tt_gdnc_delta_oncard_at(device, layer, 0u, chunk_rows, decay, beta);
}

/** `PN_TT_GDNC_STAGE_GATE_UNIT` (64 or 128, default 64): the gate block's diff granularity.
 *  A gate run is 32 floats; a 16-row sub-chunk changes half of it, which 64 B units can skip. */
static size_t gdnc_stage_gate_unit(void)
{
    static size_t cached = 0;

    if (cached == 0u) {
        const char *value = getenv("PN_TT_GDNC_STAGE_GATE_UNIT");

        /* 64 since 2026-10-04: 155.6 MB instead of 170.6 a prefill, 314.2 -> 317.0 tok/s,
         * VERIFY 0. `=128` is the previous unit. */
        cached = (value && strcmp(value, "128") == 0) ? 128u : 64u;
    }
    return cached;
}

/** The staging slot a sub-chunk uses: one per 16-row sub-chunk index. */
static uint32_t gdnc_stage_slot(uint32_t first_row)
{
    return (first_row / FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) % PN_TT_GDNC_STAGE_SLOTS;
}

/** `PN_TT_GDNC_STAGE_DIFF`: send only the units that differ from what the slot holds. */
static bool gdnc_stage_diff_enabled(void);

/** Diff the staging only OUTSIDE a stream: under the streamed band loop (PHASE W) every
 *  write is a puller entry, and 64 one-unit writes a staging cost more than the 8 KiB block
 *  (run_w_sigprof.sh: CLI wall 2.922 -> 2.891 s with the block). Same bytes land either way. */
static bool gdnc_stage_diff_now(void)
{
    static int env_set = -1;

    if (env_set < 0)
        env_set = getenv("PN_TT_GDNC_STAGE_DIFF") && *getenv("PN_TT_GDNC_STAGE_DIFF") ? 1 : 0;
    return gdnc_stage_diff_enabled() && (env_set || pn_tt_dispatch_quiet());
}

static bool gdnc_stage_diff_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_GDNC_STAGE_DIFF");

        cached = (value && *value) ? (strcmp(value, "0") != 0) : PN_TT_GDNC_STAGE_DIFF_DEFAULT;
        fprintf(stderr, "pn_tt: gdnc staging uploads %s\n",
                cached ? "only the units that CHANGED against a shadow of the slot (diff)"
                       : "every byte (full)");
    }
    return cached != 0;
}

/**
 * Write `bytes` of `stage` to the card at `address`, sending only the `unit`-byte runs that
 * differ from `shadow` — and then make the shadow equal the stage.
 *
 * BIT-EXACT BY CONSTRUCTION, NOT BY A FACE MAP. The shadow is what this slot holds ON THE CARD:
 * nothing else writes these regions and the kernels only read them, so after this call the
 * card's bytes equal `stage` whether a unit was sent or skipped. Which faces of a decay tile
 * are data, constant or zero at a given `first_row` is never assumed here; it is discovered by
 * the comparison, so a C = 32 arm or a ragged last pass needs no special case.
 *
 * Adjacent changed units go up as ONE write — the link is transaction-count bound.
 * `PN_TT_RED=gdnc_stage_shadow_lie` marks the first changed unit of each upload as already
 * present, which must make the answer DIFFER: the proof the skipped bytes are load-bearing.
 */
static int gdnc_upload_diff(pn_tt_device *device, uint32_t bank, uint32_t address,
                            const unsigned char *stage, unsigned char *shadow, size_t bytes,
                            size_t unit, bool shadow_valid, size_t *moved)
{
    const bool lie = pn_tt_red("gdnc_stage_shadow_lie");
    bool lied = false;
    size_t offset = 0;

    if (!shadow_valid) {
        device->gdnc_stage_writes++;
        if (pn_tt_write_dram(device, bank, address, stage, bytes) != 0)
            return -1;
        memcpy(shadow, stage, bytes);
        *moved += bytes;
        return 0;
    }
    while (offset < bytes) {
        size_t start, end;

        while (offset < bytes && memcmp(stage + offset, shadow + offset, unit) == 0)
            offset += unit;
        if (offset >= bytes)
            break;
        if (lie && !lied) {
            lied = true;
            offset += unit;
            continue;
        }
        start = offset;
        while (offset < bytes && memcmp(stage + offset, shadow + offset, unit) != 0)
            offset += unit;
        end = offset;
        device->gdnc_stage_writes++;
        if (pn_tt_write_dram(device, bank, address + (uint32_t)start, stage + start,
                             end - start) != 0)
            return -1;
        memcpy(shadow + start, stage + start, end - start);
        *moved += end - start;
    }
    return 0;
}

/**
 * `PN_TT_GDNC_STAGE_VERIFY=1`: read the slot back after every upload and compare it to what
 * the builders produced — the DIRECT proof the diff upload is bit-exact, because a corrupted
 * decay value can leave a short greedy completion unchanged (MEASURED: the shadow-lie red
 * skipped 2,448 x 1 KiB and the 8-token answer stayed `cmp` IDENTICAL). Counts units that
 * differ; prints the total at release. A timing arm must NOT carry it.
 */
static void gdnc_stage_verify(pn_tt_device *device, uint32_t slot, size_t decay_bytes,
                              size_t gate_bytes)
{
    static int armed = -1;
    unsigned char *back;
    size_t offset;

    if (armed < 0) {
        const char *value = getenv("PN_TT_GDNC_STAGE_VERIFY");

        armed = (value && *value && strcmp(value, "0") != 0) ? 1 : 0;
        if (armed)
            fprintf(stderr, "pn_tt: gdnc STAGE VERIFY armed — every slot read back and "
                            "compared (NOT a timing arm)\n");
    }
    if (!armed)
        return;
    back = malloc(decay_bytes > gate_bytes ? decay_bytes : gate_bytes);
    if (!back)
        return;
    if (pn_tt_read_dram(device, device->gdnc_delta_constant[slot].bank,
                        device->gdnc_delta_constant[slot].address +
                            ftgc_delta_decay_tile(0u) * PN_TT_GDNC_TILE_BYTES,
                        back, decay_bytes) == 0) {
        for (offset = 0; offset < decay_bytes; offset += PN_TT_GDNC_STAGE_DECAY_UNIT)
            if (memcmp(back + offset, device->gdnc_decay_stage + offset,
                       PN_TT_GDNC_STAGE_DECAY_UNIT) != 0)
                device->gdnc_stage_verify_mismatches++;
    }
    if (pn_tt_read_dram(device, device->gdnc_gate[slot].bank, device->gdnc_gate[slot].address,
                        back, gate_bytes) == 0) {
        for (offset = 0; offset < gate_bytes; offset += PN_TT_GDNC_STAGE_GATE_UNIT)
            if (memcmp(back + offset, (const unsigned char *)device->gdnc_gate_stage + offset,
                       PN_TT_GDNC_STAGE_GATE_UNIT) != 0)
                device->gdnc_stage_verify_mismatches++;
    }
    device->gdnc_stage_verify_checks++;
    free(back);
}

/** Upload the staged decay tiles and gate block into `slot`, full or diffed. */
static int gdnc_upload_slot(pn_tt_device *device, uint32_t slot, size_t decay_bytes,
                            size_t gate_bytes, size_t *moved)
{
    const bool diff = gdnc_stage_diff_enabled();
    const bool valid = diff && device->gdnc_shadow_valid[slot];

    if (gdnc_upload_diff(device, device->gdnc_delta_constant[slot].bank,
                         device->gdnc_delta_constant[slot].address +
                             ftgc_delta_decay_tile(0u) * PN_TT_GDNC_TILE_BYTES,
                         device->gdnc_decay_stage, device->gdnc_shadow_decay[slot],
                         decay_bytes, PN_TT_GDNC_STAGE_DECAY_UNIT, valid, moved) != 0 ||
        gdnc_upload_diff(device, device->gdnc_gate[slot].bank, device->gdnc_gate[slot].address,
                         (const unsigned char *)device->gdnc_gate_stage,
                         (unsigned char *)device->gdnc_shadow_gate[slot], gate_bytes,
                         PN_TT_GDNC_STAGE_GATE_UNIT, valid, moved) != 0) {
        /* A FAILED WRITE LEAVES THE CARD UNKNOWN, so the shadow can no longer stand for it. */
        device->gdnc_shadow_valid[slot] = false;
        return -1;
    }
    device->gdnc_shadow_valid[slot] = true;
    gdnc_stage_verify(device, slot, decay_bytes, gate_bytes);
    return 0;
}

static int gdnc_stage_gate_finite(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                  uint32_t chunk_rows);

/** One staging's inputs, for the per-head builder (inline or on the pipeline's worker). */
typedef struct {
    pn_tt_device *device;
    uint32_t first_row, chunk_rows;
    const float *decay, *beta;
    bool gate_first; /**< `PN_TT_RED=gdnc_gate_first` */
} gdnc_stage_job;

/** Build heads [first, first + count) into the stage buffers. Each head writes only its own
 *  4 KiB decay tile and its own gate run, which is what lets a worker build ahead of the
 *  upload of the heads before it. */
static int gdnc_stage_build_heads(void *context, uint32_t first, uint32_t count)
{
    const gdnc_stage_job *job = (const gdnc_stage_job *)context;
    pn_tt_device *device = job->device;
    const uint32_t heads = device->gdn_shape.v_heads;
    uint32_t head;

    for (head = first; head < first + count; head++) {
        float head_decay[PN_TT_GDNC_TILE_DIM];
        float head_beta[PN_TT_GDNC_TILE_DIM];
        uint32_t row;

        for (row = 0; row < job->chunk_rows; row++) {
            /* RED PROOF: `gdnc_gate_first` gives every position of the chunk the FIRST
             * position's gate. That is the shape of a caller who kept the decode family's
             * per-token gate, and it is right at row 0 and wrong at rows 1..C-1. */
            const uint32_t take = job->gate_first ? 0u : row;

            head_decay[row] = job->decay[(size_t)take * heads + head];
            head_beta[row] = job->beta[(size_t)take * heads + head];
        }
        /* THE SUB-CHUNK IS ENTIRELY IN THESE TWO BUILDERS AND IN THE WRITER'S ROW RANGE.
         * `first_row = 0` routes straight back into the shipping builders, so the whole-pass
         * descriptor stages byte-identical operands. */
        if (ft_tt_op_gdnc_build_decay_at(head_decay, job->first_row, job->chunk_rows,
                                         device->gdnc_decay_stage +
                                             (size_t)head * PN_TT_GDNC_TILE_BYTES,
                                         PN_TT_GDNC_TILE_BYTES) != FT_TT_OK ||
            ft_tt_op_gdnc_build_gate_at(head_decay, head_beta, job->first_row, job->chunk_rows,
                                        device->gdn_shape.state_size,
                                        device->gdnc_gate_stage +
                                            (size_t)head * FT_TT_OP_GDNC_GATE_STRIDE,
                                        FT_TT_OP_GDNC_GATE_STRIDE) != FT_TT_OK)
            return -1;
    }
    return 0;
}

/** `PN_TT_GDNC_STAGE_THREAD`: build the heads on a worker while the decay tiles of the heads
 *  already built go up — staging becomes max(build, upload) instead of their sum. */
static bool gdnc_stage_thread_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_GDNC_STAGE_THREAD");

        /* ON since 2026-10-04: 308.8 -> 314.4 tok/s, staging 356 -> 317 us, VERIFY 0. */
        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        if (cached)
            fprintf(stderr, "pn_tt: gdnc staging BUILDS on a worker thread while it uploads\n");
    }
    return cached != 0;
}

/** Heads per pipeline batch: 4 heads = 16 KiB of decay a batch, a few us of build. */
#define PN_TT_GDNC_STAGE_BATCH_HEADS 4u

/**
 * The pipelined staging: the worker builds batches of heads; this thread diff-uploads each
 * batch's decay tiles as soon as it is released, then the gate block once all are built. The
 * bytes on the card are those of the inline path by construction — same builders, same diff,
 * the only change is WHEN each head's upload starts.
 */
static int gdnc_stage_pipelined_checked(pn_tt_device *device, gdnc_stage_job *job,
                                        uint32_t slot, uint32_t layer, size_t gate_bytes,
                                        size_t *moved, double *build_wait)
{
    const uint32_t heads = device->gdn_shape.v_heads;
    const bool valid = gdnc_stage_diff_now() && device->gdnc_shadow_valid[slot];
    const uint32_t decay_base = device->gdnc_delta_constant[slot].address +
                                ftgc_delta_decay_tile(0u) * PN_TT_GDNC_TILE_BYTES;
    uint32_t done = 0u;
    int rc = 0;

    if (pn_tt_stage_pipeline_start(&device->gdnc_stage_pipeline, gdnc_stage_build_heads, job,
                                   heads, PN_TT_GDNC_STAGE_BATCH_HEADS) != 0) {
        /* No worker: build inline, then the same uploads — correct, just not overlapped. */
        if (gdnc_stage_build_heads(job, 0u, heads) != 0 ||
            gdnc_stage_gate_finite(device, layer, job->first_row, job->chunk_rows) != 0)
            return -1;
        return gdnc_upload_slot(device, slot, (size_t)heads * PN_TT_GDNC_TILE_BYTES,
                                gate_bytes, moved);
    }
    while (done < heads && rc == 0) {
        const uint32_t next = (heads - done < PN_TT_GDNC_STAGE_BATCH_HEADS)
                                  ? heads : done + PN_TT_GDNC_STAGE_BATCH_HEADS;
        const size_t offset = (size_t)done * PN_TT_GDNC_TILE_BYTES;
        const double at = pn_tt_now_seconds();

        if (pn_tt_stage_pipeline_wait(&device->gdnc_stage_pipeline, next) != 0) {
            rc = -1;
            break;
        }
        *build_wait += pn_tt_now_seconds() - at;
        rc = gdnc_upload_diff(device, device->gdnc_delta_constant[slot].bank,
                              decay_base + (uint32_t)offset, device->gdnc_decay_stage + offset,
                              device->gdnc_shadow_decay[slot] + offset,
                              (size_t)(next - done) * PN_TT_GDNC_TILE_BYTES,
                              PN_TT_GDNC_STAGE_DECAY_UNIT, valid, moved);
        done = next;
    }
    /* THE WHOLE JOB IS ALWAYS AWAITED, on success and on failure: a worker still writing the
     * stage buffers when the next staging starts would corrupt it. */
    (void)pn_tt_stage_pipeline_wait(&device->gdnc_stage_pipeline, heads);
    if (rc == 0)
        rc = gdnc_stage_gate_finite(device, layer, job->first_row, job->chunk_rows);
    if (rc == 0)
        rc = gdnc_upload_diff(device, device->gdnc_gate[slot].bank,
                              device->gdnc_gate[slot].address,
                              (const unsigned char *)device->gdnc_gate_stage,
                              (unsigned char *)device->gdnc_shadow_gate[slot], gate_bytes,
                              PN_TT_GDNC_STAGE_GATE_UNIT, valid, moved);
    device->gdnc_shadow_valid[slot] = rc == 0;
    return rc;
}

/** NOTHING NON-FINITE REACHES THE CARD — the gate scan, shared by both staging paths. */
static int gdnc_stage_gate_finite(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                  uint32_t chunk_rows)
{
    const uint32_t heads = device->gdn_shape.v_heads;

    /* NOTHING NON-FINITE REACHES THE CARD, AND THIS IS A REFUSAL RATHER THAN A CLAMP.
     *
     * A NaN in one gate column propagates into the carried state and every position after it
     * in the sequence, and the only symptom is a completion that reads as a different model —
     * which is exactly how the `last / P_t` underflow presented before it was found. There is
     * no arithmetic that can produce a NaN here now, so this is a check that a change to the
     * builders cannot quietly break the recurrence rather than a guard against known inputs.
     * It costs a linear scan of 5,120 floats against a 148 KiB DRAM write. */
    {
        const size_t count = (size_t)heads * FT_TT_OP_GDNC_GATE_STRIDE;
        size_t bad = 0;
        size_t index;

        for (index = 0; index < count; index++) {
            const float value = device->gdnc_gate_stage[index];

            if (!(value > -1e30f && value < 1e30f))
                bad++;
        }
        if (bad != 0u) {
            fprintf(stderr, "pn_tt: gdnc REFUSED layer %u — %zu of %zu gate floats are not "
                            "finite at %u rows from row %u. A NaN here reaches the carried "
                            "state and every position after it.\n",
                    layer, bad, count, chunk_rows, first_row);
            return -1;
        }
    }
    return 0;
}


/** L11b, ON by default (`PN_TT_GDNC_COMPACT=0` is off): stage decay|beta a head (8 KiB a
 *  staging) and let the reader build the decay-ratio tiles and gate runs
 *  (kernels/ftgc_compact.h, bit-identical to the builders) instead of uploading 4 KiB + 640 B
 *  a head. 1,629-row prompt: STAGE 0.72 -> 0.05 s, prefill 366 -> 404 tok/s, IDENTICAL. */
static bool gdnc_compact_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_GDNC_COMPACT");

        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        if (cached)
            fprintf(stderr, "pn_tt: gdnc staging is COMPACT — decay|beta up, the reader "
                            "builds the decay tiles and gate runs\n");
    }
    return cached != 0;
}

/** The host's own `1/sqrt(S)` exactly as the gate builder computes it: PS at P = 1. */
static uint32_t gdnc_compact_scale_bits(const pn_tt_device *device)
{
    float one = 1.0f, zero = 0.0f, out[FT_TT_OP_GDNC_GATE_STRIDE];
    uint32_t bits = 0u;

    if (ft_tt_op_gdnc_build_gate(&one, &zero, 1u, device->gdn_shape.state_size, out,
                                 FT_TT_OP_GDNC_GATE_STRIDE) == FT_TT_OK)
        memcpy(&bits, &out[FT_TT_OP_GDNC_GATE_PS * PN_TT_GDNC_TILE_DIM], sizeof(bits));
    return bits;
}

/** The compact staging: header (scale bits) then decay[32] | beta[32] a head, diffed up into
 *  the slot's gate block. The slot's decay tiles are not touched (nothing reads them). */
static int gdnc_compact_stage(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                              uint32_t chunk_rows, const float *decay, const float *beta,
                              uint32_t slot)
{
    const uint32_t heads = device->gdn_shape.v_heads;
    const size_t words = FTGC_COMPACT_HEADER_WORDS + (size_t)heads * FTGC_COMPACT_HEAD_WORDS;
    uint32_t *block = (uint32_t *)device->gdnc_gate_stage;
    const bool gate_first = pn_tt_red("gdnc_gate_first");
    /* RED PROOF: every beta zero — no position writes the carried state. Must DIFFER. */
    const bool beta0 = pn_tt_red("gdnc_compact_beta0");
    double mark = pn_tt_now_seconds();
    size_t moved = 0, bad = 0;
    uint32_t head, row;

    if (words > (size_t)heads * FT_TT_OP_GDNC_GATE_STRIDE)
        return -1;
    memset(block, 0, words * sizeof(uint32_t));
    block[0] = gdnc_compact_scale_bits(device);
    for (head = 0; head < heads; head++) {
        uint32_t *run = block + FTGC_COMPACT_HEADER_WORDS + (size_t)head * FTGC_COMPACT_HEAD_WORDS;

        for (row = 0; row < chunk_rows; row++) {
            const uint32_t take = gate_first ? 0u : row;
            const float d = decay[(size_t)take * heads + head];
            const float b = beta0 ? 0.0f : beta[(size_t)take * heads + head];

            if (!(d > -1e30f && d < 1e30f) || !(b > -1e30f && b < 1e30f))
                bad++;
            memcpy(&run[row], &d, sizeof(d));
            memcpy(&run[32u + row], &b, sizeof(b));
        }
    }
    if (bad != 0u) {
        fprintf(stderr, "pn_tt: gdnc REFUSED layer %u — %zu non-finite decay/beta values at %u "
                        "rows from row %u\n", layer, bad, chunk_rows, first_row);
        return -1;
    }
    device->gdnc_stage_build_seconds += pn_tt_now_seconds() - mark;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    mark = pn_tt_now_seconds();
    if (gdnc_upload_diff(device, device->gdnc_gate[slot].bank, device->gdnc_gate[slot].address,
                         (const unsigned char *)block,
                         (unsigned char *)device->gdnc_shadow_gate[slot],
                         words * sizeof(uint32_t), 64u,
                         gdnc_stage_diff_now() && device->gdnc_shadow_valid[slot],
                         &moved) != 0) {
        device->gdnc_shadow_valid[slot] = false;
        return -1;
    }
    device->gdnc_shadow_valid[slot] = true;
    device->gdnc_stage_upload_seconds += pn_tt_now_seconds() - mark;
    device->gdnc_stage_bytes += moved;
    device->gdnc_stage_ops += 1u;
    return 0;
}

static int gdnc_delta_stage(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                            uint32_t chunk_rows, const float *decay, const float *beta,
                            uint32_t slot)
{
    const uint32_t heads = device ? device->gdn_shape.v_heads : 0u;
    gdnc_stage_job job;
    /* P0's BRACKET. Started after the refusals so a rejected call books nothing, and closed
     * before `gdnc_submit()` so the two clocks never overlap. */
    double stage_mark;

    if (!device || !device->gdnc_constant || !decay || !beta || chunk_rows == 0u ||
        chunk_rows > PN_TT_GDNC_TILE_DIM || layer >= device->gdn_shape.layer_count)
        return -1;
    /* REFUSED HERE AS WELL AS IN `ft_tt_op_gdnc_parameters()` AND ON THE DEVICE, because this
     * is the only one of the three that sees the STAGING: a misaligned `first_row` would build
     * `heads` decay tiles and gate runs against a window the encoder then rejects, and the
     * failure would read as an encode error at a call site that had already written 148 KiB. */
    if ((first_row % FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) != 0u ||
        first_row + chunk_rows > PN_TT_GDNC_TILE_DIM)
        return -1;

    if (gdnc_compact_enabled())
        return gdnc_compact_stage(device, layer, first_row, chunk_rows, decay, beta, slot);

    job.device = device;
    job.first_row = first_row;
    job.chunk_rows = chunk_rows;
    job.decay = decay;
    job.beta = beta;
    job.gate_first = pn_tt_red("gdnc_gate_first");

    /* THE PIPELINED PATH: build on the worker, upload here, the gate after. It is not taken
     * under the stage-double red (a clock arm that must resend whole blocks). */
    if (gdnc_stage_thread_enabled() && !pn_tt_red("gdnc_stage_double")) {
        const size_t gate_bytes = (size_t)heads * FT_TT_OP_GDNC_GATE_STRIDE *
                                  sizeof(*device->gdnc_gate_stage);
        double build_wait = 0.0;
        size_t moved = 0;
        int rc;

        if (pn_tt_matvec_flush(device) != 0)
            return -1;
        stage_mark = pn_tt_now_seconds();
        rc = gdnc_stage_pipelined_checked(device, &job, slot, layer, gate_bytes, &moved,
                                          &build_wait);
        if (rc < 0)
            return -1;
        /* BUILD = the time this thread WAITED on the worker; UPLOAD = the rest. Their sum is
         * the staging's wall, which is what the stage line has always reported. */
        device->gdnc_stage_build_seconds += build_wait;
        device->gdnc_stage_upload_seconds += pn_tt_now_seconds() - stage_mark - build_wait;
        device->gdnc_stage_bytes += moved;
        device->gdnc_stage_ops += 1u;
        gdnc_stage_verify(device, slot, (size_t)heads * PN_TT_GDNC_TILE_BYTES, gate_bytes);
        return 0;
    }

    /* ONE DECAY TILE AND ONE GATE RUN PER V-HEAD, built by the SHIPPING builders. `decay` and
     * `beta` arrive ROW-MAJOR BY POSITION — which is how a batched projection produces them —
     * and a head's run is the column, so this is the transpose and nothing else. */
    stage_mark = pn_tt_now_seconds();
    if (gdnc_stage_build_heads(&job, 0u, heads) != 0)
        return -1;
    if (gdnc_stage_gate_finite(device, layer, first_row, chunk_rows) != 0)
        return -1;

    /* THE BUILD CLOCK CLOSES HERE and the UPLOAD clock opens, with the chained-projection
     * flush BETWEEN them rather than inside either. `pn_tt_write_dram()` flushes on entry, so
     * without this the first write's bracket would contain a CARD wait and the "MB/s" it
     * derives would not be the link's. Hoisting it changes nothing about what runs — the very
     * next statement would flush — it only decides which clock pays for it. */
    {
        const size_t decay_bytes = (size_t)heads * PN_TT_GDNC_TILE_BYTES;
        const size_t gate_bytes = (size_t)heads * FT_TT_OP_GDNC_GATE_STRIDE *
                                  sizeof(*device->gdnc_gate_stage);
        const double now = pn_tt_now_seconds();

        device->gdnc_stage_build_seconds += now - stage_mark;
        if (pn_tt_matvec_flush(device) != 0)
            return -1;
        stage_mark = pn_tt_now_seconds();

        /* RED PROOF: `gdnc_stage_double` sends the SAME blocks TWICE. It must NOT change an
         * answer — what is under test is a CLOCK: the STAGE line's upload term must roughly
         * double and `sequential` must rise by the same seconds. */
        {
            const unsigned repeats = pn_tt_red("gdnc_stage_double") ? 2u : 1u;
            size_t moved = 0;
            unsigned pass;

            for (pass = 0; pass < repeats; pass++) {
                /* A second pass of the red arm must resend, not diff against what the first
                 * pass just wrote — otherwise the arm would move zero bytes. */
                if (pass > 0u)
                    device->gdnc_shadow_valid[slot] = false;
                if (gdnc_upload_slot(device, slot, decay_bytes, gate_bytes, &moved) != 0)
                    return -1;
            }
            device->gdnc_stage_upload_seconds += pn_tt_now_seconds() - stage_mark;
            device->gdnc_stage_bytes += moved;
            device->gdnc_stage_ops += 1u;
        }
    }
    return 0;
}

int pn_tt_gdnc_delta_prestage_at(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                 uint32_t chunk_rows, const float *decay, const float *beta)
{
    const uint32_t slot = gdnc_stage_slot(first_row);

    /* THE REGIONS, NOT THE RESIDENCY. A prestage runs inside a split's enqueue/await window,
     * and the idle retire may have just stopped `gdnc` there (it is re-armed by the refusal
     * path a moment later). Staging writes DRAM the residency does not own, so it needs only
     * that the regions exist. MEASURED: requiring `gdnc.running` here aborted the first pass
     * after every load gap. */
    if (!device || !device->gdnc_constant || !device->gdnc_shadow_decay[slot])
        return -1;
    device->gdnc_prestaged[slot].valid = false;
    if (gdnc_delta_stage(device, layer, first_row, chunk_rows, decay, beta, slot) != 0)
        return -1;
    device->gdnc_prestaged[slot].valid = true;
    device->gdnc_prestaged[slot].layer = layer;
    device->gdnc_prestaged[slot].first_row = first_row;
    device->gdnc_prestaged[slot].chunk_rows = chunk_rows;
    return 0;
}

int pn_tt_gdnc_delta_oncard_at(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                               uint32_t chunk_rows, const float *decay, const float *beta)
{
    const ftgc_shape shape = gdnc_delta_shape(device);
    const uint32_t slot = gdnc_stage_slot(first_row);
    ft_tt_op_gdnc_request request;

    if (!pn_tt_gdnc_is_armed(device))
        return -1;
    /* A PRESTAGED SLOT IS USED ONLY FOR THE DESCRIPTOR IT WAS STAGED FOR, and is consumed by
     * it. Anything else stages here, exactly as before the overlap existed. */
    if (device->gdnc_prestaged[slot].valid && device->gdnc_prestaged[slot].layer == layer &&
        device->gdnc_prestaged[slot].first_row == first_row &&
        device->gdnc_prestaged[slot].chunk_rows == chunk_rows) {
        device->gdnc_prestaged[slot].valid = false;
    } else {
        device->gdnc_prestaged[slot].valid = false;
        if (gdnc_delta_stage(device, layer, first_row, chunk_rows, decay, beta, slot) != 0)
            return -1;
    }

    request = gdnc_request(device, FT_TT_OP_GDNC_PHASE_DELTA, chunk_rows);
    request.first_row = first_row;
    request.activation_in = gdnc_operand(device, &device->gdn_stream_out);
    request.activation_out = gdnc_operand(device, &device->gdn_y);
    request.constant = gdnc_operand(device, &device->gdnc_delta_constant[slot]);
    request.gate_address = device->gdnc_gate[slot].address;
    if (gdnc_compact_enabled())
        request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_COMPACT;
    /* THE RECURRENT STATE IS THE DECODE FAMILY'S BUFFER, AT ITS ADDRESS —
     * `ftgc_state_tile()` is `ftg_state_tile()`. */
    request.state_conv = gdnc_operand(device, &device->gdn_region[layer].state);

    /* RED PROOF: the other half of the carry, and invisible in chunk 0 for the same
     * structural reason — at chunk 0 the incoming state is zero, so `K_hat M_0^T` and
     * `Q_hat M_0^T` both vanish and half the transform is not exercised. */
    if (pn_tt_red("gdnc_state_wipe") &&
        pn_tt_zero_dram(device, device->gdn_region[layer].state.bank,
                        device->gdn_region[layer].state.address,
                        (size_t)ftgc_delta_state_tiles(&shape) * PN_TT_GDNC_TILE_BYTES) != 0)
        return -1;
    return gdnc_submit(device, &request);
}

uint32_t pn_tt_gdnc_compact_scale_bits(const pn_tt_device *device)
{
    return device ? gdnc_compact_scale_bits(device) : 0u;
}

int pn_tt_gdnc_delta_oncard_card_gate(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                      uint32_t chunk_rows, uint32_t gate_address)
{
    /* SLOT 0 FOR EVERY SUB-CHUNK: the two slots exist to double-buffer the host STAGING, and
     * nothing is staged here; their DELTA constants are the same bytes (one upload, both
     * slots), and the card gate blocks live in slot 0's bank (the gate shares its xy). */
    const uint32_t slot = 0u;
    ft_tt_op_gdnc_request request;

    if (!pn_tt_gdnc_is_armed(device) || !gdnc_compact_enabled() || chunk_rows == 0u ||
        chunk_rows > PN_TT_GDNC_TILE_DIM || layer >= device->gdn_shape.layer_count ||
        (first_row % FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) != 0u ||
        first_row + chunk_rows > PN_TT_GDNC_TILE_DIM)
        return -1;
    device->gdnc_prestaged[gdnc_stage_slot(first_row)].valid = false;
    request = gdnc_request(device, FT_TT_OP_GDNC_PHASE_DELTA, chunk_rows);
    request.first_row = first_row;
    request.activation_in = gdnc_operand(device, &device->gdn_stream_out);
    request.activation_out = gdnc_operand(device, &device->gdn_y);
    request.constant = gdnc_operand(device, &device->gdnc_delta_constant[slot]);
    request.gate_address = gate_address;
    request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_COMPACT;
    request.state_conv = gdnc_operand(device, &device->gdn_region[layer].state);
    return gdnc_submit(device, &request);
}

#endif /* USE_TT */
