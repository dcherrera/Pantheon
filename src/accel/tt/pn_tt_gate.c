/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gate.c
 * @brief Implementation of the merged GDN output gate. See pn_tt_gate.h — the argument for why
 *        a MIXED CB table is correct HERE and wrong in `gdn_float32.cbtable` lives there.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_replica_map.h"

/**
 * Which enabled Tensix the merged gate drives.
 *
 * 87 is free in the map `pn_tt_render.c` documents: the render pair holds 85/86, the GDN
 * residency 90, rope/memory/attention 82/83/84, and eight matmul members stop at 82. It sits
 * inside the range sixteen matmul members would reach, which is not an oversight —
 * `pn_tt_prepare_family()` refuses a collision BY NAME against every live residency, so a run
 * that wants both is told rather than silently sharing a core.
 */
#define PN_TT_CORE_GATE_NARROW_ENV     "PN_TT_CORE_GATE_NARROW"
#define PN_TT_CORE_GATE_NARROW_DEFAULT 87u

#define PN_TT_GATE_NARROW_PROGRAM "gate_narrow"
#define PN_TT_GATE_NARROW_NAME    "gate-narrow"

/**
 * Whether the merged gate runs. **DEFAULT OFF — it was flipped ON on 2026-08-25 and flipped
 * back the same day, because it is right in DECODE and it BREAKS THE CHUNKWISE PREFILL.**
 *
 * WHAT IT IS WORTH, MEASURED. Seven decode arms, one model load each, `company_after = 0`
 * throughout: `cmp` **IDENTICAL** (189 B, md5 80fc6259), **20,448 -> 19,296 descriptors a
 * token** (exactly 24 a token over 48 tokens, and -25,165,824 B), on-card **0.114 -> 0.112
 * s/token** against a control that read 0.114 TWICE, wall 5.17 / 5.20 control against 5.22.
 * The predicted ~7.5 ms was 6x optimistic and the census says why: the merged `silu_mul` costs
 * **267.9 us a descriptor against the plain one's ~170** (2.097 us/tile against 0.998), because
 * the fp32 unpack of a 4,096-byte operand is real work the bf16 unpack it replaces was not
 * doing. Removing `render_narrow` is worth 0.0026 s/token and the heavier multiply gives back
 * ~0.0024. Red arm `PN_TT_RED=gate_narrow_swap` DIFFERS at 190 B; both collision refusals fire.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS OFF: IT BREAKS `gdnc`, ON TWO ARMS, ABSENT FROM THE THIRD
 * ---------------------------------------------------------------------------
 * With the chunkwise prefill armed (`PN_TT_PREFILL=1 PN_TT_PREFILL_GDNC=1`) and this default
 * ON, both a C = 32 arm and a C = 16 arm produce:
 *
 *     pn_tt: gdnc descriptor 4 did not complete: error
 *     pn_tt: render-narrow descriptor 1 did not complete: error
 *     pn_tt: the gdnc residency refused a descriptor — retiring it and arming a fresh one on 95
 *
 * after which documents 1 and 2 **DECLINE the batched prefill and run per token**. The same
 * two arms with `PN_TT_GATE_NARROW=0` are clean, and one of them came back BYTE-IDENTICAL to a
 * dump recorded before this residency existed — so the variable is this default and nothing
 * else in the session. **The two-pass gate's `--require-prefill` refused the dump rather than
 * scoring a sequential arm under a chunkwise label; that guard is the only reason this was not
 * 339 records of the wrong thing.**
 *
 * NOT DIAGNOSED, AND DO NOT ASSUME THE CORE MAP. `gate_narrow` sits on 87 and `gdnc` on 95, so
 * it is not a core collision, and `pn_tt_core_holder()` refuses those by name anyway. The open
 * suspicion is arming ORDER: this residency is armed lazily, and the one thing already on
 * record about lazy arming in this family is that **a residency armed after
 * `pn_tt_publish_replica_map()` reads whatever was at that fixed L1 address** — the note beside
 * the render pair says both of those still have that gap, and `render_narrow` is the other
 * family erroring here. That is a hypothesis with an obvious first arm and no measurement yet.
 */
static bool gate_narrow_requested(void)
{
    const char *want = getenv(PN_TT_GATE_NARROW_ENV);

    return want && *want && atoi(want) != 0;
}

int pn_tt_device_arm_gate_narrow(pn_tt_device *device, uint32_t tiles)
{
    pn_tt_gate_residency *slot;
    ft_tt_op_eltwise_request request;
    ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
    /* The depth search must not write into the residency's own CB buffer: `prepare_family` has
     * not run yet, and it zeroes the whole handle. */
    unsigned char scratch_cb[sizeof(device->gate.residency.cb_buffer)];
    uint64_t scratch_mask;
    uint32_t depth;

    if (!device)
        return -1;
    if (tiles == 0u || device->gate.residency.running)
        return 0;
    if (!gate_narrow_requested())
        return 0;
    if (!pn_tt_program_is_present(PN_TT_GATE_NARROW_PROGRAM)) {
        fprintf(stderr, "pn_tt: no `%s` program — the GDN gate keeps the render_narrow + "
                        "silu_mul pair\n", PN_TT_GATE_NARROW_PROGRAM);
        return 0;
    }

    /* THE DEEPEST BLOCK THAT BOTH DIVIDES THE RUN AND FITS THE ARENA. The B stream is 4096 B
     * a page here while A, the scratch and the output are 2048, so the arena question is not
     * the shipped elementwise one and `place_cbs` is the only thing that can answer it.
     * Walking down from the divisor rather than refusing is what stops a page size from
     * silently costing a whole residency. */
    for (depth = pn_tt_common_depth(&tiles, 1u, PN_TT_ONCARD_MAX_DEPTH); depth > 0u;
         depth = pn_tt_common_depth(&tiles, 1u, depth - 1u)) {
        memset(&request, 0, sizeof(request));
        request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
        request.tile_count = tiles;
        request.prefetch_tiles = depth;
        request.input_a.tile_bytes = PN_TT_TILE_BF16;
        request.input_b.tile_bytes = PN_TT_TILE_FP32;
        request.output.tile_bytes = PN_TT_TILE_BF16;
        if (ft_tt_op_eltwise_place_cbs(&request, placed, scratch_cb, sizeof(scratch_cb),
                                       &scratch_mask) == FT_TT_OK)
            break;
    }
    if (depth == 0u) {
        fprintf(stderr, "pn_tt: no block depth both divides a %u-tile gate and fits the L1 "
                        "arena at %u B / %u B input pages\n", tiles, PN_TT_TILE_BF16,
                PN_TT_TILE_FP32);
        return 0;
    }

    slot = &device->gate;
    if (!pn_tt_prepare_family(device, &slot->residency, PN_TT_GATE_NARROW_PROGRAM,
                              PN_TT_GATE_NARROW_NAME, PN_TT_CORE_GATE_NARROW_ENV,
                              PN_TT_CORE_GATE_NARROW_DEFAULT, PN_TT_ONCARD_TRIAD_MASK))
        return 0;
    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
    request.tile_count = tiles;
    request.prefetch_tiles = depth;
    request.input_a.tile_bytes = PN_TT_TILE_BF16;
    request.input_b.tile_bytes = PN_TT_TILE_FP32;
    request.output.tile_bytes = PN_TT_TILE_BF16;
    if (ft_tt_op_eltwise_place_cbs(&request, placed, slot->residency.cb_buffer,
                                   sizeof(slot->residency.cb_buffer),
                                   &slot->residency.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the gate-narrow circular buffers do not fit at %u tiles, "
                        "depth %u\n", tiles, depth);
        pn_tt_residency_release(&slot->residency);
        return 0;
    }
    if (!pn_tt_start_family(&slot->residency))
        return 0;
    /* THE TABLE THIS CORE READS MUST BE A WRITTEN FACT. This residency runs
     * `resident_vector_writer.c` and it arms AFTER `pn_tt_publish_replica_map()` has already
     * gone out, so without this line it reads whatever is at the fixed L1 replica-table
     * address — which is not "no mirroring" but "whatever a previous process left there", the
     * state that cost the elementwise split its `cmp`. `ssm_gated` is not a mirrored strip, so
     * EMPTY is the correct table and not a stub. */
    if (pn_tt_publish_empty_replica_map(device, &slot->residency, "the gate-narrow residency")
        != 0) {
        pn_tt_residency_stop(&slot->residency);
        pn_tt_residency_release(&slot->residency);
        return 0;
    }
    slot->tiles = tiles;
    slot->depth = depth;
    fprintf(stderr, "pn_tt: gate-narrow is up on core %u — %u tiles, silu(bf16) * fp32 -> "
                    "bf16, depth %u. render_narrow leaves the decode token.\n",
            slot->residency.core_index, tiles, depth);
    return 0;
}

bool pn_tt_gate_narrow_is_armed(const pn_tt_device *device)
{
    return device && device->gate.residency.running;
}

int pn_tt_gate_narrow(pn_tt_device *device, ft_tt_op_operand a_bf16, ft_tt_op_operand b_fp32,
                      ft_tt_op_operand out_bf16, uint32_t tiles)
{
    pn_tt_gate_residency *slot;
    ft_tt_op_eltwise_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    double started;

    if (!device)
        return -1;
    slot = &device->gate;
    if (!slot->residency.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. A gate whose
     * `z` a queued matvec is still writing reads the PREVIOUS token's activation and nothing
     * anywhere refuses it. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    /* THE PAGE SIZE IS A THREE-SIDED CONTRACT WITH NO CARD-SIDE CHECK: the DRAM operand, the L1
     * circular buffer and the reader's compile-time TILE_BYTES / TILE_BYTES_B all have to
     * agree, and the device can see none of it. This is the only place that can. */
    if (a_bf16.tile_bytes != PN_TT_TILE_BF16 || b_fp32.tile_bytes != PN_TT_TILE_FP32 ||
        out_bf16.tile_bytes != PN_TT_TILE_BF16) {
        fprintf(stderr, "pn_tt: gate-narrow wants %u/%u -> %u B operands and was handed "
                        "%u/%u -> %u\n", PN_TT_TILE_BF16, PN_TT_TILE_FP32, PN_TT_TILE_BF16,
                a_bf16.tile_bytes, b_fp32.tile_bytes, out_bf16.tile_bytes);
        return -1;
    }
    if (tiles == 0u || tiles > slot->tiles ||
        ft_tt_op_block_depth(tiles, slot->depth) != slot->depth ||
        (tiles % (2u * slot->depth)) != 0u) {
        fprintf(stderr, "pn_tt: %u tiles is not a whole number of depth-%u blocks the "
                        "gate-narrow residency was placed for (%u)\n", tiles, slot->depth,
                slot->tiles);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    /* RED PROOF: the two operands of the gate swap. `silu(y) * z` is not `silu(z) * y`, both
     * strips are the same width, and no shape check anywhere can see the difference — the same
     * mutation `pn_tt_silu_mul()` carries, on the descriptor that replaces it. */
    if (pn_tt_red("gate_narrow_swap")) {
        request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
        request.input_a = b_fp32;
        request.input_b = a_bf16;
    } else {
        request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
        request.input_a = a_bf16;
        request.input_b = b_fp32;
    }
    request.tile_count = tiles;
    request.prefetch_tiles = slot->depth;
    request.output = out_bf16;
    if (ft_tt_op_eltwise_parameters(&request, words) != FT_TT_OK)
        return -1;
    pn_tt_vec_probe_apply(words);

    started = pn_tt_now_seconds();
    if (pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_ELTWISE, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* The words address DRAM only, so the same descriptor is valid for a fresh residency —
         * the re-arm argument every other family here makes. */
        fprintf(stderr, "pn_tt: the gate-narrow residency refused a descriptor — retiring it "
                        "by ASKING and arming a fresh one\n");
        pn_tt_residency_stop(&slot->residency);
        pn_tt_residency_release(&slot->residency);
        device->oncard_restarts++;
        if (!pn_tt_prepare_family(device, &slot->residency, PN_TT_GATE_NARROW_PROGRAM,
                                  PN_TT_GATE_NARROW_NAME, PN_TT_CORE_GATE_NARROW_ENV,
                                  PN_TT_CORE_GATE_NARROW_DEFAULT, PN_TT_ONCARD_TRIAD_MASK) ||
            !pn_tt_start_family(&slot->residency) ||
            pn_tt_residency_submit(&slot->residency, FT_TT_QUEUE_OP_ELTWISE, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    pn_tt_oncard_account(device, PN_TT_GATE_NARROW_NAME, "silu_mul", tiles,
                         pn_tt_now_seconds() - started);
    return 0;
}

void pn_tt_gate_release(pn_tt_device *device)
{
    if (!device)
        return;
    if (device->gate.residency.running) {
        pn_tt_residency_stop(&device->gate.residency);
        pn_tt_residency_release(&device->gate.residency);
    }
    memset(&device->gate, 0, sizeof(device->gate));
}

#endif /* USE_TT */
