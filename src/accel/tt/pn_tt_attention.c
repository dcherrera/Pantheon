/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention.c
 * @brief The three ops that have to move together — NeoX RoPE, the KV-cache append and
 *        flash-decode attention — on three more resident cores, with the KV cache itself
 *        living in card DRAM.
 *
 *        pn_tt_device.c owns the card and the matmul, pn_tt_oncard.c owns the norm /
 *        elementwise / sampling residencies, and this file owns the attention triple and the
 *        card-resident KV regions. The public API is in pn_tt_device.h.
 *
 * ---------------------------------------------------------------------------
 * WHY ALL THREE, AS A NUMBER
 * ---------------------------------------------------------------------------
 * With the norms, the SwiGLU, the residual adds and the LM head's top-N already on the card,
 * the ONLY activations still crossing the link are q, k and v — because the KV append and
 * the GQA softmax are host arithmetic and need all three in host memory. For MiniCPM5-1B
 * that is 16x128 + 2x128 + 2x128 floats a layer, moved as logical row 0 of 64 + 8 + 8 tiles
 * at 128 B a tile (the 64-byte card->host read alignment makes row 0 cost 2 x 64 B, of which
 * half is discarded — ft_tt_tile_row0.h):
 *
 *     (64 + 8 + 8) tiles x 128 B = 10,240 B a layer x 24 layers = 245,760 B a token
 *     245,760 B / 4.7 MB/s = 52.3 ms
 *
 * against a measured readback phase of 0.052 s. **The readback IS q/k/v and nothing else.**
 * Moving any ONE of the three removes nothing, because whichever is left still needs them
 * host-side. Moving all three removes the phase.
 *
 * ---------------------------------------------------------------------------
 * THE ROW-PHASE PROBLEM, AND WHY THIS FORWARD STAYS AT TILE ROW 0
 * ---------------------------------------------------------------------------
 * foundry's RoPE kernel multiplies activation tile row `r` by table tile row `r`, and the
 * table is built for positions `first_position + 32*b + r`. The contract those ops were
 * written against therefore puts a decode token at tile row `position % 32` and carries that
 * phase through the whole forward (ft_tt_op_memory.h §1.5b).
 *
 * **Pantheon's TT forward puts every token at row 0 and this file keeps it there.** Adopting
 * the row phase would mean threading a row index through the activation upload, the download,
 * the top-N selection and every strip's zero-padding invariant — a change to five ops to
 * serve one. The alternative is arithmetically identical and costs one small write:
 *
 *   **ONE row block of the rotary table lives in DRAM and its ROW 0 is rebuilt per token.**
 *   The block is `2 * rot_tiles` tiles (16 KiB for this model) and only row 0 is ever read,
 *   so the per-token write is `2 * rot_tiles * 32` bf16 values through the row-0 writer —
 *   **512 B, ~1.8 us at 287 MB/s** — plus 2 x rot host trig evaluations in double.
 *   `first_position` is then passed as 0, because the operand IS this call's row block; the
 *   driver's own `ft_tt_op_rope_table_offset()` must not add a second offset on top.
 *
 * The 67.1 MB whole-context table foundry's header describes is the PREFILL answer and is
 * still the right one there. It is not the decode answer, because decode needs one row.
 *
 * Rows 1-31 of the table stay zero, so RoPE drives rows 1-31 of its output to zero
 * (`out = x*COS + x_partner*SIN` with both factors zero). That is the same strip invariant
 * pn_tt_oncard.c states, preserved rather than assumed.
 *
 * ---------------------------------------------------------------------------
 * WHERE THE KV CACHE LIVES, AND WHAT IT DOES TO SESSION EXPORT
 * ---------------------------------------------------------------------------
 * See pn_tt_device.h for the decision and the arithmetic. In one line: the mirror costs 2x
 * the bytes of the export it would make free, and it charges every session — so the cache
 * moves to the card and export pays at export time.
 *
 * @status RoPE, KV append and flash decode wired into the forward behind
 *         `PN_TT_ONCARD=...,attention`. Opt-in only: `all` does not select it, because it
 *         changes the accumulation order and the home of the KV cache rather than the speed
 *         of an op.
 * @issues The card holds ONE K/V region set. A new sequence rebinds it (stale contents are
 *         harmless — every position a sequence reads is one it wrote); two sequences
 *         ALTERNATING would overwrite each other, and that is detected to a depth of
 *         PN_TT_KV_DISPLACED_KEYS and refused by name beyond nothing deeper.
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_device_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_op_attention.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_memory.h"
#include "tt/ft_tt_op_rope.h"
#include "tt/ft_tt_tile_row0.h"
#include "tt/ft_tt_tilize.h"

/** The three program directories this file needs beside the matmul and on-card ones. */
#define PN_TT_ROPE_PROGRAM      "rope"
#define PN_TT_MEMORY_PROGRAM    "memory"
#define PN_TT_ATTENTION_PROGRAM "attention"

/**
 * Which enabled Tensix each of the three drives.
 *
 * **82-84 AND NOT 75-77, WHICH IS WHERE THE OBVIOUS CHOICE COLLIDES.** The matmul group puts
 * members 1..N-1 at `PN_TT_MATMUL_CORE_FIRST` = 75 upward, so a group of 8 already owns
 * 75-81; core 71 is member 0 and 72-74 are the reduce, elementwise and sampling residencies.
 * Booting a second program's firmware onto a core that already holds a resident triad is a
 * wedged core, not a configuration — so these start past the widest group the device can be
 * asked for, and `pn_tt_attention_cores_are_free()` checks it rather than trusting the
 * arithmetic.
 */
#define PN_TT_CORE_ROPE_ENV          "PN_TT_CORE_ROPE"
#define PN_TT_CORE_MEMORY_ENV        "PN_TT_CORE_MEMORY"
#define PN_TT_CORE_ATTENTION_ENV     "PN_TT_CORE_ATTENTION"
#define PN_TT_CORE_ROPE_DEFAULT      82u
#define PN_TT_CORE_MEMORY_DEFAULT    83u
#define PN_TT_CORE_ATTENTION_DEFAULT 84u

/** THE PREFILL CHUNK MIXER'S HEAD SPLIT. `PN_TT_ATTN_CORES` members, member 0 the `attention`
 *  residency itself, the rest from `PN_TT_ATTN_MEMBER_CORE_FIRST` upward. 75..81 are free
 *  since the matmul group moved to 96. The count shrinks until it divides the Q heads. */
#define PN_TT_ATTN_CORES_ENV              "PN_TT_ATTN_CORES"
/* EIGHT since 2026-10-04: attention 0.78 -> 0.41 -> 0.23 -> 0.15 s of a prefill at
 * 1/2/4/8 cores, 246.7 -> 273.7 tok/s, every arm IDENTICAL, `attn_group` red DIFFERS. */
#define PN_TT_ATTN_CORES_DEFAULT          8u
#define PN_TT_ATTN_MEMBER_CORE_FIRST_ENV  "PN_TT_ATTN_MEMBER_CORE_FIRST"
#define PN_TT_ATTN_MEMBER_CORE_DEFAULT    75u

/** Position tiles per K/V block — the flash-decode prefetch depth, fixed by the RESIDENCY.
 *
 *  16 is the point every recorded attention figure was measured at (0.840 ms a descriptor at
 *  8k KV, L1 flat in context at 413,696 B). It is a knob because the PADDED read window is
 *  `2 * block_tiles` position tiles whatever the live length is, so at short context a large
 *  block is work the answer does not need — and that trade is a measurement, not a guess. */
#define PN_TT_ATTN_BLOCK_TILES_ENV     "PN_TT_ATTN_BLOCK_TILES"
#define PN_TT_ATTN_BLOCK_TILES_DEFAULT 16u

/** The rotary table's row block is rebuilt per token; this says none is current yet. */
#define PN_TT_ROPE_NO_POSITION UINT32_MAX

/* ================================================================
 * Arming
 * ================================================================ */

/** The rotary width this model rotates, resolved exactly as the CPU oracle resolves it. */
static uint32_t pn_tt_rotary_width(const pn_tt_device *device)
{
    return ft_tt_op_rope_rotary_width(device->attn_head_dim, device->attn_rope_dim);
}

/** Tiles across one rotary table row block: `rot_tiles` cos tiles then `rot_tiles` sin. */
static uint32_t pn_tt_rope_table_tiles(const pn_tt_device *device)
{
    return 2u * (pn_tt_rotary_width(device) / PN_TT_TILE_DIM);
}

/**
 * Allocate `bytes` of card DRAM, zeroed, round-robin across banks.
 *
 * Zeroing is not tidiness here either: an attention descriptor's PADDED read window reaches
 * past the live sequence by up to `2*block_tiles - 1` position tiles, and those tiles are fed
 * to the FPU before the tail mask kills their scores. Zero is a value the mask makes
 * harmless; whatever DRAM happened to hold is not.
 */
static int pn_tt_dram_alloc_zeroed(pn_tt_device *device, size_t bytes, uint32_t *out_bank,
                                   uint32_t *out_address)
{
    ft_tt_buffer buffer;
    uint32_t bank;

    if (device->dram_bank_count == 0u)
        return -1;
    bank = device->strip_bank_cursor % device->dram_bank_count;
    device->strip_bank_cursor++;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes, &buffer) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full allocating %zu bytes for the attention family\n",
                bytes);
        return -1;
    }
    *out_bank = buffer.logical_bank;
    *out_address = (uint32_t)buffer.address;
    return pn_tt_zero_dram(device, *out_bank, *out_address, bytes);
}

/** The flash-decode constants — reduce scaler, 1/sqrt(head_dim) and the 32 tail masks —
 *  built once per model and held. */
static bool pn_tt_upload_attention_constants(pn_tt_device *device)
{
    size_t bytes = ft_tt_op_attention_constants_bytes(PN_TT_TILE_BF16);
    unsigned char *constants = malloc(bytes);
    bool ok;

    if (!constants)
        return false;
    if (ft_tt_op_attention_build_constants(device->attn_head_dim, PN_TT_TILE_BF16, constants,
                                           bytes) != FT_TT_OK) {
        free(constants);
        return false;
    }
    ok = pn_tt_dram_alloc_zeroed(device, bytes, &device->attn_const_bank,
                                 &device->attn_const_address) == 0 &&
         pn_tt_write_dram(device, device->attn_const_bank, device->attn_const_address,
                          constants, bytes) == 0;
    free(constants);
    return ok;
}

/** One layer's K and V regions, identical geometry, on two banks so the append's write and
 *  the next layer's read are not usually the same endpoint. */
static bool pn_tt_alloc_kv_regions(pn_tt_device *device, uint32_t layer_count)
{
    size_t region_bytes = (size_t)device->kv_layout.region_bytes;
    uint32_t layer;

    device->kv_region = calloc(layer_count, sizeof(*device->kv_region));
    if (!device->kv_region)
        return false;
    device->kv_layer_count = layer_count;
    for (layer = 0; layer < layer_count; layer++) {
        pn_tt_kv_region *region = &device->kv_region[layer];

        if (pn_tt_dram_alloc_zeroed(device, region_bytes, &region->k_bank,
                                    &region->k_address) != 0 ||
            pn_tt_dram_alloc_zeroed(device, region_bytes, &region->v_bank,
                                    &region->v_address) != 0)
            return false;
    }
    return true;
}

/** The RoPE residency. Its circular buffers are placed for the DEPTH, which has to divide
 *  both head_tiles and rot_tiles — and both q and k run the same shape, so one placement
 *  serves the whole forward. */
static bool pn_tt_arm_rope(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_ROPE_CB_COUNT];
    ft_tt_op_rope_request request;

    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.heads = device->attn_q_heads;
    request.head_dim = device->attn_head_dim;
    request.rope_dim = device->attn_rope_dim;
    request.first_position = 0u;
    request.prefetch_tiles = device->attn_dim_tiles;
    request.input.tile_bytes = PN_TT_TILE_BF16;
    request.table.tile_bytes = PN_TT_TILE_BF16;
    request.output.tile_bytes = PN_TT_TILE_BF16;
    request.input.address = 1u; /* placement only reads the tile sizes; a set operand is a
                                 * precondition of the encoder, so give it one */
    request.table.address = 1u;
    request.output.address = 1u;

    if (!pn_tt_prepare_family(device, &device->rope, PN_TT_ROPE_PROGRAM, "rope",
                              PN_TT_CORE_ROPE_ENV, PN_TT_CORE_ROPE_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_rope_place_cbs(&request, placed, device->rope.cb_buffer,
                                sizeof(device->rope.cb_buffer), &device->rope.cb_mask) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: the RoPE circular buffers do not fit for head_dim %u\n",
                device->attn_head_dim);
        pn_tt_residency_release(&device->rope);
        return false;
    }
    return pn_tt_start_family(&device->rope);
}

/** The memory residency, for KV_APPEND only. DM0|DM1: this family has NO compute thread. */
static bool pn_tt_arm_memory(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_MEMORY_CB_COUNT];

    if (!pn_tt_prepare_family(device, &device->memory, PN_TT_MEMORY_PROGRAM, "memory",
                              PN_TT_CORE_MEMORY_ENV, PN_TT_CORE_MEMORY_DEFAULT,
                              PN_TT_ONCARD_DM_MASK))
        return false;
    /* The embedding group size is required to be non-zero even though this residency never
     * runs an EMBEDDING descriptor: a zero places no CB and the kernel's grid guard would
     * then refuse the opcode at run time for a build-time reason. It is `dim_tiles` and not
     * the model's embedding width because nothing will ever push through it. */
    if (ft_tt_op_memory_place_cbs(device->attn_embed_tiles, device->attn_dim_tiles,
                                  PN_TT_TILE_BF16, placed, device->memory.cb_buffer,
                                  sizeof(device->memory.cb_buffer), &device->memory.cb_mask) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: the KV-append circular buffers do not fit at %u tiles a "
                        "group\n", device->attn_dim_tiles);
        pn_tt_residency_release(&device->memory);
        return false;
    }
    return pn_tt_start_family(&device->memory);
}

/** The flash-decode residency. `block_tiles` is fixed here for its whole life — the kernel
 *  REFUSES a descriptor that disagrees with the buffers it was staged with. */
static bool pn_tt_arm_flash_decode(pn_tt_device *device)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_ATTENTION_CB_COUNT];
    ft_tt_op_attention_residency residency;

    residency.q_rows = device->attn_q_rows;
    residency.dim_tiles = device->attn_dim_tiles;
    residency.block_tiles = device->attn_block_tiles;
    residency.tile_bytes = PN_TT_TILE_BF16;

    if (!pn_tt_prepare_family(device, &device->attention, PN_TT_ATTENTION_PROGRAM, "attention",
                              PN_TT_CORE_ATTENTION_ENV, PN_TT_CORE_ATTENTION_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_attention_place_cbs(&residency, placed, device->attention.cb_buffer,
                                     sizeof(device->attention.cb_buffer),
                                     &device->attention.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the flash-decode circular buffers do not fit at block_tiles "
                        "%u (%zu B of arena)\n", device->attn_block_tiles,
                ft_tt_op_attention_l1_bytes(&residency));
        pn_tt_residency_release(&device->attention);
        return false;
    }
    return pn_tt_start_family(&device->attention);
}

/** Arm ONE head-split member on `core`, with the flash-decode residency's own shape. */
static bool pn_tt_arm_attention_member(pn_tt_device *device, pn_tt_residency *member,
                                       uint32_t core)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_ATTENTION_CB_COUNT];
    ft_tt_op_attention_residency residency;

    residency.q_rows = device->attn_q_rows;
    residency.dim_tiles = device->attn_dim_tiles;
    residency.block_tiles = device->attn_block_tiles;
    residency.tile_bytes = PN_TT_TILE_BF16;
    if (!pn_tt_prepare_family_on_core(device, member, PN_TT_ATTENTION_PROGRAM,
                                      "attention-member", PN_TT_ATTN_MEMBER_CORE_FIRST_ENV, core,
                                      PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_attention_place_cbs(&residency, placed, member->cb_buffer,
                                     sizeof(member->cb_buffer), &member->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(member);
        return false;
    }
    return pn_tt_start_family(member);
}

/**
 * Arm the head-split members. A member that will not come up shrinks the split rather than
 * failing attention — the split is a speed lever, member 0 alone is the shipped answer — and
 * the count then shrinks further until it divides the Q heads, so every member owns the same
 * number. Members past the legal count are STOPPED, not left resident.
 */
static void pn_tt_arm_attention_members(pn_tt_device *device)
{
    const uint32_t asked = pn_tt_env_u32(PN_TT_ATTN_CORES_ENV, PN_TT_ATTN_CORES_DEFAULT);
    const uint32_t first = pn_tt_env_u32(PN_TT_ATTN_MEMBER_CORE_FIRST_ENV,
                                         PN_TT_ATTN_MEMBER_CORE_DEFAULT);
    uint32_t want = asked < 1u ? 1u : (asked > PN_TT_ATTN_MAX_MEMBERS ? PN_TT_ATTN_MAX_MEMBERS
                                                                      : asked);
    uint32_t up = 1u, armed, index;

    device->attention_members = 1u;
    while (want > 1u && (device->attn_q_heads % want) != 0u)
        want--;
    for (index = 1u; index < want; index++) {
        if (!pn_tt_arm_attention_member(device, &device->attention_member[index - 1u],
                                        first + index - 1u))
            break;
        up++;
    }
    armed = up;
    while (armed > 1u && (device->attn_q_heads % armed) != 0u)
        armed--;
    for (index = armed; index < up; index++)
        pn_tt_residency_release(&device->attention_member[index - 1u]);
    device->attention_members = armed;
    if (asked > 1u)
        fprintf(stderr, "pn_tt: attention HEAD SPLIT — asked %u, running %u member(s) from core "
                        "%u (members 1.. on %u..), %u Q heads each\n",
                asked, armed, device->attention.core_index, first,
                device->attn_q_heads / armed);
}

/** Retire and re-arm head-split member `index` (1-based) on its own core. */
static bool pn_tt_attention_member_rearm(pn_tt_device *device, uint32_t index)
{
    pn_tt_residency *member = &device->attention_member[index - 1u];
    const uint32_t core = member->core_index;

    fprintf(stderr, "pn_tt: attention member %u refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", index, core);
    pn_tt_residency_stop(member);
    pn_tt_residency_release(member);
    device->oncard_restarts++;
    if (!pn_tt_residency_retired(member)) {
        fprintf(stderr, "pn_tt: attention member %u did not retire — REFUSING to arm a fresh "
                        "one on core %u\n", index, core);
        return false;
    }
    return pn_tt_arm_attention_member(device, member, core);
}

/** The largest block depth that both fits L1 and leaves the padded read window inside the
 *  cache — `2 * block_tiles` position tiles are fetched however short the sequence is. */
static uint32_t pn_tt_choose_block_tiles(const pn_tt_device *device, uint32_t requested)
{
    uint32_t fits_l1 = ft_tt_op_attention_max_block_tiles(device->attn_q_rows,
                                                          device->attn_dim_tiles,
                                                          PN_TT_TILE_BF16);
    uint32_t fits_cache = device->kv_layout.capacity_tiles / 2u;
    uint32_t chosen = requested;

    if (fits_l1 != 0u && chosen > fits_l1)
        chosen = fits_l1;
    if (fits_cache != 0u && chosen > fits_cache)
        chosen = fits_cache;
    return chosen;
}

/**
 * Are the three cores this family wants held by nobody else?
 *
 * A core already running a resident triad is not merely busy: `pn_tt_residency_bind_core()`
 * BOOTS firmware on it, and booting over five running processors is the wedge signature this
 * project has paid for repeatedly. The claim is checked against the live handles rather than
 * against the default arithmetic, because the defaults are all environment-overridable and an
 * arithmetic check would agree with itself.
 *
 * @return true when all three are free; false with the clash named on stderr
 */
static bool pn_tt_attention_cores_are_free(const pn_tt_device *device)
{
    uint32_t wanted[3] = { pn_tt_env_u32(PN_TT_CORE_ROPE_ENV, PN_TT_CORE_ROPE_DEFAULT),
                           pn_tt_env_u32(PN_TT_CORE_MEMORY_ENV, PN_TT_CORE_MEMORY_DEFAULT),
                           pn_tt_env_u32(PN_TT_CORE_ATTENTION_ENV,
                                         PN_TT_CORE_ATTENTION_DEFAULT) };
    static const char *const k_name[3] = { "rope", "memory", "attention" };
    uint32_t taken[PN_TT_MATMUL_MAX_CORES + 4u];
    uint32_t taken_count = 0u, index, member;
    bool free_of_clashes = true;

    taken[taken_count++] = device->core_index;
    for (member = 0; member < device->group.member_count; member++)
        taken[taken_count++] = device->group.member[member].core_index;
    if (device->reduce.core_booted)
        taken[taken_count++] = device->reduce.core_index;
    if (device->eltwise.core_booted)
        taken[taken_count++] = device->eltwise.core_index;
    if (device->sample.core_booted)
        taken[taken_count++] = device->sample.core_index;

    for (index = 0; index < 3u; index++) {
        uint32_t other;

        for (other = 0; other < taken_count; other++) {
            if (wanted[index] != taken[other])
                continue;
            fprintf(stderr, "pn_tt: REFUSED — the %s residency wants core %u and another "
                            "resident program already holds it\n", k_name[index],
                    wanted[index]);
            free_of_clashes = false;
        }
        for (other = 0; other < index; other++) {
            if (wanted[index] != wanted[other])
                continue;
            fprintf(stderr, "pn_tt: REFUSED — the %s and %s residencies were both pointed at "
                            "core %u\n", k_name[other], k_name[index], wanted[index]);
            free_of_clashes = false;
        }
    }
    return free_of_clashes;
}

int pn_tt_device_arm_attention(pn_tt_device *device, uint32_t q_heads, uint32_t kv_heads,
                               uint32_t head_dim, uint32_t rope_dim, double rope_theta,
                               uint32_t capacity_positions, uint32_t layer_count)
{
    uint32_t requested_block_tiles;

    if (!device || q_heads == 0u || kv_heads == 0u || head_dim == 0u || layer_count == 0u)
        return -1;
    if ((q_heads % kv_heads) != 0u || (head_dim % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (!pn_tt_device_oncard_requested(device, PN_TT_ONCARD_ATTENTION))
        return 0;

    /* A REFUSAL, not a fallback. Attention was asked for by name; a missing program would
     * otherwise leave it on the host with a different numerics and nothing saying so — the
     * defect that cost a whole round on the matmul residency. */
    if (!pn_tt_program_is_present(PN_TT_ROPE_PROGRAM) ||
        !pn_tt_program_is_present(PN_TT_MEMORY_PROGRAM) ||
        !pn_tt_program_is_present(PN_TT_ATTENTION_PROGRAM)) {
        fprintf(stderr, "pn_tt: REFUSED — on-card attention needs the `rope`, `memory` and "
                        "`attention` programs and at least one is missing from the program "
                        "root\n");
        return -1;
    }
    if (!pn_tt_attention_cores_are_free(device))
        return -1;

    device->attn_q_heads = q_heads;
    device->attn_kv_heads = kv_heads;
    device->attn_head_dim = head_dim;
    device->attn_rope_dim = rope_dim;
    device->attn_rope_theta = rope_theta;
    device->attn_q_rows = q_heads / kv_heads;
    device->attn_dim_tiles = head_dim / PN_TT_TILE_DIM;
    device->attn_embed_tiles = device->attn_dim_tiles;
    device->rope_table_position = PN_TT_ROPE_NO_POSITION;
    device->rope_chunk1_first = UINT32_MAX;

    if (ft_tt_kv_cache_layout_init(kv_heads, head_dim, capacity_positions, PN_TT_TILE_BF16,
                                   &device->kv_layout) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: no KV cache layout for %u heads x %u dim x %u positions\n",
                kv_heads, head_dim, capacity_positions);
        return -1;
    }
    requested_block_tiles = pn_tt_env_u32(PN_TT_ATTN_BLOCK_TILES_ENV,
                                          PN_TT_ATTN_BLOCK_TILES_DEFAULT);
    device->attn_block_tiles = pn_tt_choose_block_tiles(device, requested_block_tiles);
    if (device->attn_block_tiles == 0u) {
        fprintf(stderr, "pn_tt: REFUSED — a %u-position cache cannot hold one flash-decode "
                        "read window\n", device->kv_layout.capacity_positions);
        return -1;
    }
    if (device->attn_block_tiles != requested_block_tiles)
        fprintf(stderr, "pn_tt: flash-decode block_tiles %u -> %u (L1 and the %u-tile cache "
                        "capacity both bound it)\n", requested_block_tiles,
                device->attn_block_tiles, device->kv_layout.capacity_tiles);

    if (!pn_tt_upload_attention_constants(device) ||
        pn_tt_dram_alloc_zeroed(device, (size_t)pn_tt_rope_table_tiles(device) * PN_TT_TILE_BF16,
                                &device->rope_table_bank, &device->rope_table_address) != 0 ||
        pn_tt_dram_alloc_zeroed(device, (size_t)pn_tt_rope_table_tiles(device) * PN_TT_TILE_BF16,
                                &device->rope_chunk1_bank, &device->rope_chunk1_address) != 0 ||
        !pn_tt_alloc_kv_regions(device, layer_count)) {
        fprintf(stderr, "pn_tt: REFUSED — the attention family's DRAM did not allocate\n");
        return -1;
    }

    device->oncard[PN_TT_ONCARD_ATTENTION] =
        pn_tt_arm_rope(device) && pn_tt_arm_memory(device) && pn_tt_arm_flash_decode(device);
    if (device->oncard[PN_TT_ONCARD_ATTENTION])
        pn_tt_arm_attention_members(device);
    if (!device->oncard[PN_TT_ONCARD_ATTENTION]) {
        fprintf(stderr, "pn_tt: REFUSED — on-card attention was asked for and one of its "
                        "three residencies did not arm\n");
        return -1;
    }
    fprintf(stderr, "pn_tt: attention on the card — %u Q heads / %u KV groups of %u rows, "
                    "head_dim %u, block_tiles %u, KV cache %u positions x %u layers "
                    "(%.1f MiB)\n",
            q_heads, kv_heads, device->attn_q_rows, head_dim, device->attn_block_tiles,
            device->kv_layout.capacity_positions, layer_count,
            (double)device->kv_layout.region_bytes * 2.0 * (double)layer_count /
                (1024.0 * 1024.0));
    return 0;
}

/* ================================================================
 * Binding the ONE region set to a sequence
 * ================================================================ */

int pn_tt_kv_bind(pn_tt_device *device, const void *sequence_key)
{
    uint32_t index;

    if (!device || !sequence_key)
        return -1;
    if (device->kv_key == sequence_key)
        return 0;
    for (index = 0; index < device->kv_displaced_count; index++) {
        if (device->kv_displaced[index] != sequence_key)
            continue;
        /* This key held the region set, lost it, and is back. That is two sequences
         * alternating on ONE cache, which would overwrite each other's positions and produce
         * a plausible wrong answer with no error anywhere. */
        fprintf(stderr, "pn_tt: REFUSED — two sequences are alternating on the card's one KV "
                        "region set. On-card attention serves ONE sequence at a time.\n");
        return -1;
    }
    if (device->kv_key) {
        if (device->kv_displaced_count < PN_TT_KV_DISPLACED_KEYS)
            device->kv_displaced[device->kv_displaced_count++] = device->kv_key;
        else
            device->kv_displaced[device->kv_rebinds % PN_TT_KV_DISPLACED_KEYS] = device->kv_key;
        device->kv_rebinds++;
        fprintf(stderr, "pn_tt: the card KV region set rebound to a new sequence (%llu so "
                        "far); the previous one's positions are now stale\n",
                (unsigned long long)device->kv_rebinds);
    }
    device->kv_key = sequence_key;
    return 0;
}

/* ================================================================
 * Re-arming, and one submit that survives an idle retirement
 * ================================================================ */

/** Retire a stranded attention-family residency by ASKING and arm a fresh one on the same
 *  core from the shapes it was armed with. Same mechanism, same reason, as
 *  pn_tt_oncard.c's: a residency is BOUNDED and a serve idles between requests. */
static bool pn_tt_attention_rearm(pn_tt_device *device, pn_tt_residency *residency)
{
    fprintf(stderr, "pn_tt: the %s residency refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", residency->name,
            residency->core_index);
    pn_tt_residency_stop(residency);
    pn_tt_residency_release(residency);
    device->oncard_restarts++;

    if (residency == &device->rope)
        return pn_tt_arm_rope(device);
    if (residency == &device->memory)
        return pn_tt_arm_memory(device);
    if (residency == &device->attention)
        return pn_tt_arm_flash_decode(device);
    return false;
}

/**
 * Submit one descriptor, re-arming once if the residency had retired underneath it.
 *
 * Every parameter block in this file addresses DRAM only — none of the three families writes
 * an L1 address into its words — so the SAME words are valid against a fresh residency and
 * are reused rather than re-encoded. (The sampling family is the exception, and it re-encodes
 * for exactly that reason.)
 */
static int pn_tt_attention_submit(pn_tt_device *device, pn_tt_residency *residency,
                                  uint32_t opcode, const uint32_t *words)
{
    double started = pn_tt_now_seconds();

    if (pn_tt_residency_submit(residency, opcode, words, PN_TT_OP_TIMEOUT_MS) != 0) {
        if (!pn_tt_attention_rearm(device, residency) ||
            pn_tt_residency_submit(residency, opcode, words, PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    device->oncard_seconds += pn_tt_now_seconds() - started;
    device->oncard_ops++;
    return 0;
}

/* ================================================================
 * RoPE
 * ================================================================ */

/**
 * Rebuild row 0 of the rotary table's one row block for `position`.
 *
 * Per row block the layout is `rot_tiles` cos tiles then `rot_tiles` SIGNED-sin tiles: the
 * sign is in the table, which is what makes every rotated tile the same three FPU passes
 * with no branch (ft_tt_op_rope.h). Column c below the half carries -sin, at or above it
 * +sin, and both halves share the angle of `c mod half`.
 *
 * The angle is computed in DOUBLE by ft_tt_op_rope_angle() — `position * freq` reaches ~1e5
 * radians at this model's context and neither bf16 nor the SFPU can reduce that.
 */
static int pn_tt_rope_table_for_position(pn_tt_device *device, uint32_t position)
{
    const uint32_t rot = pn_tt_rotary_width(device);
    const uint32_t half = rot / 2u;
    const uint32_t tiles = pn_tt_rope_table_tiles(device);
    uint16_t *row;
    uint32_t column;
    int rc;

    if (device->rope_table_position == position)
        return 0;
    /* A CHUNK LEFT REAL ANGLES IN ROWS 1..31 AND THIS WRITE ONLY TOUCHES ROW 0. Decode's
     * activations have those rows zero, so the OUTPUT would be zero either way — but the
     * invariant this file states is about the TABLE, and an invariant that is true for a
     * different reason than the one written down is one nobody can rely on next time. */
    if (device->rope_table_is_chunk) {
        if (pn_tt_zero_dram(device, device->rope_table_bank, device->rope_table_address,
                            (size_t)tiles * PN_TT_TILE_BF16) != 0)
            return -1;
        device->rope_table_is_chunk = false;
    }
    row = calloc((size_t)tiles * PN_TT_TILE_DIM, sizeof(*row));
    if (!row)
        return -1;
    for (column = 0; column < rot; column++) {
        /* RED PROOF: every token is rotated as if it were position 0 — the failure a table
         * that did not follow the token would produce, and one no length or address check
         * can see. */
        uint32_t angle_position = pn_tt_red("rope_position") ? 0u : position;
        /* RED PROOF: the frequency ladder is REFLECTED, so slot p is rotated at the model's
         * pair `half - 1 - p` while the partner still sits at `+half`. This is the exact shape
         * of the "the on-card rotary re-pairs what the artifact un-paired" hypothesis, and it
         * is here so `PN_TT_ROPE_CHECK` is proved able to SEE that bug rather than assumed to.
         * `Q.K` is invariant under it, so the attention output barely moves — which is why an
         * attn-check cannot be the instrument for this question. */
        uint32_t angle_column = pn_tt_red("rope_freq_reflect")
                                    ? (half - 1u - (column % half)) + (column >= half ? half : 0u)
                                    : column;
        double angle = ft_tt_op_rope_angle(angle_position, angle_column, rot,
                                           device->attn_rope_theta);
        double sine = sin(angle);

        row[column] = ft_bf16_from_float((float)cos(angle));
        row[rot + column] = ft_bf16_from_float((float)(column < half ? -sine : sine));
    }
    rc = ft_tt_tile_row0_write(&device->runtime.bulk_dram,
                               (uint16_t)device->dram[device->rope_table_bank].x,
                               (uint16_t)device->dram[device->rope_table_bank].y,
                               device->rope_table_address, tiles, row);
    free(row);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += (size_t)tiles * FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_HALF_BYTES;
    device->rope_table_position = position;
    return 0;
}

/**
 * Rebuild the WHOLE 32-row rotary block for positions `first .. first+31`.
 *
 * THE SHIPPED TABLE BUILDER, ASKED FOR ONE ROW BLOCK. `ft_tt_op_rope_build_table()` is the
 * prefill answer the RoPE header describes — it tilizes with `ft_tt_tilize_strip()`, the
 * same face map the activations use, which is what a table whose value varies along BOTH
 * axes needs. Calling it for 32 positions rather than 131,072 costs 16 KiB of upload a
 * chunk (~50 us) and reuses code that is already gated, instead of a second copy of the
 * angle arithmetic beside the row-0 one.
 *
 * `first` must be a multiple of 32 — the ROW-PHASE CONTRACT: row r of the block carries
 * position `first + r`, and the chunk's row r carries the same. The builder refuses
 * anything else.
 */
static int pn_tt_rope_table_build_chunk(pn_tt_device *device, uint32_t first);
static int pn_tt_rope_table_build_chunk1(pn_tt_device *device, uint32_t first);

/** The chunk's rotary table, with its host time booked (G0's "unclocked" split). */
static int pn_tt_rope_table_for_chunk(pn_tt_device *device, uint32_t first,
                                      uint32_t *out_bank, uint32_t *out_address)
{
    const double started = pn_tt_now_seconds();
    int rc;

    /* ODD CHUNKS USE THE CHUNK-ONLY TABLE. With two bands a pass rotates positions p and p+32
     * in every layer, alternately, and one cached block evicted the other each time (0.156 s at
     * two bands, measured). Slot 1 is never touched by decode, so its key alone decides. */
    {
        const uint32_t chunk = first / PN_TT_TILE_DIM;
        const uint32_t slot = chunk % 64u;

        /* PHASE G: chunks 2.. of a tall pass — one cached block each. Chunk 0/1 keep their
         * shipped slots, so a 2-band prefill and decode are untouched. */
        if (chunk >= 2u && (device->strip_tile_rows > 4u || device->prefill_tile_rows > 4u) &&
            !pn_tt_red("rope_one_slot")) {
            if (device->rope_many_address[slot] == 0u &&
                pn_tt_dram_alloc_zeroed(device,
                                        ft_tt_op_rope_table_bytes(
                                            PN_TT_TILE_DIM,
                                            pn_tt_rotary_width(device) / PN_TT_TILE_DIM,
                                            PN_TT_TILE_BF16),
                                        &device->rope_many_bank[slot],
                                        &device->rope_many_address[slot]) != 0)
                return -1;
            *out_bank = device->rope_many_bank[slot];
            *out_address = device->rope_many_address[slot];
            if (device->rope_many_first[slot] == first + 1u && !pn_tt_red("rope_nocache"))
                return 0;
            {
                const size_t bytes = ft_tt_op_rope_table_bytes(
                    PN_TT_TILE_DIM, pn_tt_rotary_width(device) / PN_TT_TILE_DIM,
                    PN_TT_TILE_BF16);
                void *block = malloc(bytes);

                rc = block && ft_tt_op_rope_build_table(first, PN_TT_TILE_DIM,
                                                        device->attn_head_dim,
                                                        device->attn_rope_dim,
                                                        device->attn_rope_theta, block,
                                                        bytes) == FT_TT_OK
                         ? pn_tt_write_dram(device, *out_bank, *out_address, block, bytes)
                         : -1;
                free(block);
            }
            /* first + 1 so a zeroed field (0) never matches chunk 0's key. */
            device->rope_many_first[slot] = rc == 0 ? first + 1u : 0u;
            pn_tt_device_host_clock_add(device, PN_TT_HOST_CLOCK_ROPE_TABLE,
                                        pn_tt_now_seconds() - started);
            return rc;
        }
    }
    if (((first / PN_TT_TILE_DIM) & 1u) != 0u && !pn_tt_red("rope_one_slot")) {
        *out_bank = device->rope_chunk1_bank;
        *out_address = device->rope_chunk1_address;
        if (device->rope_chunk1_first == first && !pn_tt_red("rope_nocache"))
            return 0;
        rc = pn_tt_rope_table_build_chunk1(device, first);
        device->rope_chunk1_first = rc == 0 ? first : UINT32_MAX;
        pn_tt_device_host_clock_add(device, PN_TT_HOST_CLOCK_ROPE_TABLE,
                                    pn_tt_now_seconds() - started);
        return rc;
    }
    *out_bank = device->rope_table_bank;
    *out_address = device->rope_table_address;

    /* THE CHUNK BLOCK IS CACHED BY ITS FIRST POSITION. Every attention layer of a pass — and
     * both its q and its k — rotate the SAME 32 positions, so the block was being built in
     * double precision and re-uploaded 16 times a pass for identical bytes: 0.316 s of a
     * prefill, measured by the host clock below. A decode token (or `PN_TT_RED=rope_nocache`)
     * clears `rope_table_is_chunk`, so a stale block cannot survive a mode change. */
    /* RED PROOF, `PN_TT_RED=rope_cache_any`: the key ignores the position, so passes 1.. reuse
     * pass 0's angles — must DIFFER, the proof the key is load-bearing. */
    if (device->rope_table_is_chunk &&
        (device->rope_table_chunk_first == first || pn_tt_red("rope_cache_any")) &&
        !pn_tt_red("rope_nocache"))
        return 0;
    rc = pn_tt_rope_table_build_chunk(device, first);
    if (rc == 0)
        device->rope_table_chunk_first = first;

    pn_tt_device_host_clock_add(device, PN_TT_HOST_CLOCK_ROPE_TABLE,
                                pn_tt_now_seconds() - started);
    return rc;
}

static int pn_tt_rope_table_build_chunk(pn_tt_device *device, uint32_t first)
{
    const uint32_t rot_tiles = pn_tt_rotary_width(device) / PN_TT_TILE_DIM;
    const size_t bytes =
        ft_tt_op_rope_table_bytes(PN_TT_TILE_DIM, rot_tiles, PN_TT_TILE_BF16);
    void *block;
    int rc;

    if (bytes == 0u || bytes != (size_t)pn_tt_rope_table_tiles(device) * PN_TT_TILE_BF16)
        return -1;
    block = malloc(bytes);
    if (!block)
        return -1;
    if (ft_tt_op_rope_build_table(first, PN_TT_TILE_DIM, device->attn_head_dim,
                                  device->attn_rope_dim, device->attn_rope_theta, block,
                                  bytes) != FT_TT_OK) {
        free(block);
        fprintf(stderr, "pn_tt: no rotary block for positions %u..%u\n", first,
                first + PN_TT_TILE_DIM - 1u);
        return -1;
    }
    rc = pn_tt_write_dram(device, device->rope_table_bank, device->rope_table_address, block,
                          bytes);
    free(block);
    if (rc != 0)
        return -1;
    /* THE ROW-0 CACHE IS NOW VOID, AND SO IS THE ROWS-1..31-ARE-ZERO INVARIANT. Decode's
     * next token has to rebuild the block, not merely its row 0 — see
     * pn_tt_rope_table_for_position(). */
    device->rope_table_position = PN_TT_ROPE_NO_POSITION;
    device->rope_table_is_chunk = true;
    return 0;
}

/** Build the chunk block for `first` and upload it to the chunk-only table (slot 1). */
static int pn_tt_rope_table_build_chunk1(pn_tt_device *device, uint32_t first)
{
    const uint32_t rot_tiles = pn_tt_rotary_width(device) / PN_TT_TILE_DIM;
    const size_t bytes =
        ft_tt_op_rope_table_bytes(PN_TT_TILE_DIM, rot_tiles, PN_TT_TILE_BF16);
    void *block = malloc(bytes);
    int rc;

    if (!block)
        return -1;
    rc = ft_tt_op_rope_build_table(first, PN_TT_TILE_DIM, device->attn_head_dim,
                                   device->attn_rope_dim, device->attn_rope_theta, block,
                                   bytes) == FT_TT_OK
             ? pn_tt_write_dram(device, device->rope_chunk1_bank, device->rope_chunk1_address,
                                block, bytes)
             : -1;
    free(block);
    return rc;
}

int pn_tt_rope_chunk(pn_tt_device *device, const pn_tt_activation *input,
                     const pn_tt_activation *output, uint32_t heads,
                     uint32_t first_position, uint32_t rows)
{
    ft_tt_op_rope_request request;
    uint32_t words[FT_TT_OP_ROPE_WORDS];

    if (!device || !input || !output || heads == 0u || !device->rope.running)
        return -1;
    if (rows == 0u || rows > PN_TT_TILE_DIM || (first_position % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (input->tile_count < heads * device->attn_dim_tiles ||
        output->tile_count < heads * device->attn_dim_tiles)
        return -1;
    uint32_t table_bank = 0u, table_address = 0u;

    if (pn_tt_rope_table_for_chunk(device, first_position, &table_bank, &table_address) != 0)
        return -1;

    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.heads = heads;
    request.head_dim = device->attn_head_dim;
    request.rope_dim = device->attn_rope_dim;
    /* ZERO for the same reason as decode's: the operand IS this call's row block, whose
     * row r already carries `first_position + r`. `rows` is not on the wire at all — the
     * kernel rotates all 32 and the strip's rows `rows`..31 are zero, so it rotates zero. */
    request.first_position = 0u;
    request.prefetch_tiles = device->attn_dim_tiles;
    request.input = pn_tt_operand(device, input);
    request.table.x = (uint16_t)device->dram[table_bank].x;
    request.table.y = (uint16_t)device->dram[table_bank].y;
    request.table.address = table_address;
    request.table.tile_bytes = PN_TT_TILE_BF16;
    request.output = pn_tt_operand(device, output);
    if (ft_tt_op_rope_parameters(&request, words) != FT_TT_OK)
        return -1;
    return pn_tt_attention_submit(device, &device->rope, FT_TT_QUEUE_OP_ROPE, words);
}

int pn_tt_rope(pn_tt_device *device, const pn_tt_activation *input,
               const pn_tt_activation *output, uint32_t heads, uint32_t position)
{
    ft_tt_op_rope_request request;
    uint32_t words[FT_TT_OP_ROPE_WORDS];

    if (!device || !input || !output || heads == 0u || !device->rope.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (input->tile_count < heads * device->attn_dim_tiles ||
        output->tile_count < heads * device->attn_dim_tiles)
        return -1;
    if (pn_tt_rope_table_for_position(device, position) != 0)
        return -1;

    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.heads = heads;
    request.head_dim = device->attn_head_dim;
    request.rope_dim = device->attn_rope_dim;
    /* ZERO, and it is not a lie: the operand below IS this call's row block, whose row 0
     * already carries `position`. A real first_position would make the driver add
     * ft_tt_op_rope_table_offset() to an address that is already the right one. */
    request.first_position = 0u;
    request.prefetch_tiles = device->attn_dim_tiles;
    request.input = pn_tt_operand(device, input);
    request.table.x = (uint16_t)device->dram[device->rope_table_bank].x;
    request.table.y = (uint16_t)device->dram[device->rope_table_bank].y;
    request.table.address = device->rope_table_address;
    request.table.tile_bytes = PN_TT_TILE_BF16;
    request.output = pn_tt_operand(device, output);
    if (ft_tt_op_rope_parameters(&request, words) != FT_TT_OK)
        return -1;
    return pn_tt_attention_submit(device, &device->rope, FT_TT_QUEUE_OP_ROPE, words);
}

/* ================================================================
 * KV append
 * ================================================================ */

/** One region as an operand. K and V share a geometry and differ only in base. */
static ft_tt_op_operand pn_tt_region_operand(const pn_tt_device *device, uint32_t layer,
                                             bool values)
{
    const pn_tt_kv_region *region = &device->kv_region[layer];
    uint32_t bank = values ? region->v_bank : region->k_bank;
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[bank].x;
    operand.y = (uint16_t)device->dram[bank].y;
    operand.address = values ? region->v_address : region->k_address;
    operand.tile_bytes = PN_TT_TILE_BF16;
    return operand;
}

int pn_tt_kv_append(pn_tt_device *device, uint32_t layer, bool values,
                    const pn_tt_activation *source, uint32_t position)
{
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];

    if (!device || !source || !device->memory.running || layer >= device->kv_layer_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;

    memset(&request, 0, sizeof(request));
    request.cache = &device->kv_layout;
    /* RED PROOF: the append lands one position late, so attention reads a K row that belongs
     * to no token and misses this one. It stays INSIDE the region — a mutation that shortens
     * or moves a transfer out of range can take the card down instead of going red
     * (foundry/tt/DEV_NOTES.md §1). */
    request.position = position;
    if (pn_tt_red("kv_position") && position + 1u < device->kv_layout.capacity_positions)
        request.position = position + 1u;
    request.token_count = 1u;
    request.source = pn_tt_operand(device, source);
    /* The projection's answer is logical row 0 of every tile — this forward does not carry
     * the position's row phase through the activations (see the file note). */
    request.source_row0 = 0u;
    request.source_tiles = device->attn_kv_heads * device->attn_dim_tiles;
    request.region = pn_tt_region_operand(device, layer, values);
    if (source->tile_count < request.source_tiles)
        return -1;
    if (ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the KV append at position %u of layer %u is not encodable\n",
                position, layer);
        return -1;
    }
    return pn_tt_attention_submit(device, &device->memory, FT_TT_QUEUE_OP_KV_APPEND, words);
}

/* ================================================================
 * STEP 4 S2c: the verify's rows, unaligned
 * ================================================================ */

int pn_tt_rope_rows(pn_tt_device *device, const pn_tt_activation *input,
                    const pn_tt_activation *output, uint32_t heads, uint32_t first_position,
                    uint32_t rows)
{
    const uint32_t rot = pn_tt_rotary_width(device);
    const uint32_t half = rot / 2u;
    const uint32_t tiles = pn_tt_rope_table_tiles(device);
    const size_t bytes = (size_t)tiles * PN_TT_TILE_BF16;
    ft_tt_op_rope_request request;
    uint32_t words[FT_TT_OP_ROPE_WORDS];
    uint16_t *flat, *tiled;
    uint32_t r, column;
    int rc;

    if (!device || !input || !output || heads == 0u || !device->rope.running || rows == 0u ||
        rows > PN_TT_TILE_DIM)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (input->tile_count < heads * device->attn_dim_tiles ||
        output->tile_count < heads * device->attn_dim_tiles)
        return -1;
    if (device->rope_rows_address == 0u &&
        pn_tt_dram_alloc_zeroed(device, bytes, &device->rope_rows_bank,
                                &device->rope_rows_address) != 0)
        return -1;
    /* Row r = position first+r, by the DECODE row's expression term for term (the decode
     * table is what a single-position forward rotates with), rows past `rows` zero. Built
     * ONCE a verify (every attention layer and both Q and K rotate at the same positions). */
    if (device->rope_rows_key_rows == rows && device->rope_rows_key_first == first_position)
        goto submit;
    flat = calloc((size_t)PN_TT_TILE_DIM * tiles * PN_TT_TILE_DIM, sizeof(*flat));
    tiled = calloc(bytes / sizeof(uint16_t), sizeof(*tiled));
    if (!flat || !tiled) {
        free(flat);
        free(tiled);
        return -1;
    }
    for (r = 0; r < rows; r++) {
        uint16_t *row = flat + (size_t)r * tiles * PN_TT_TILE_DIM;

        for (column = 0; column < rot; column++) {
            double angle = ft_tt_op_rope_angle(first_position + r, column, rot,
                                               device->attn_rope_theta);
            double sine = sin(angle);

            row[column] = ft_bf16_from_float((float)cos(angle));
            row[rot + column] = ft_bf16_from_float((float)(column < half ? -sine : sine));
        }
    }
    rc = ft_tt_tilize_strip(flat, sizeof(*flat), rows, tiles * PN_TT_TILE_DIM,
                            (size_t)tiles * PN_TT_TILE_DIM * sizeof(*flat), tiled) == 0
             ? pn_tt_write_dram(device, device->rope_rows_bank, device->rope_rows_address,
                                tiled, bytes)
             : -1;
    free(flat);
    free(tiled);
    if (rc != 0)
        return -1;
    device->rope_rows_key_first = first_position;
    device->rope_rows_key_rows = rows;

submit:
    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.heads = heads;
    request.head_dim = device->attn_head_dim;
    request.rope_dim = device->attn_rope_dim;
    request.first_position = 0u;
    request.prefetch_tiles = device->attn_dim_tiles;
    request.input = pn_tt_operand(device, input);
    request.table.x = (uint16_t)device->dram[device->rope_rows_bank].x;
    request.table.y = (uint16_t)device->dram[device->rope_rows_bank].y;
    request.table.address = device->rope_rows_address;
    request.table.tile_bytes = PN_TT_TILE_BF16;
    request.output = pn_tt_operand(device, output);
    if (ft_tt_op_rope_parameters(&request, words) != FT_TT_OK)
        return -1;
    return pn_tt_attention_submit(device, &device->rope, FT_TT_QUEUE_OP_ROPE, words);
}

/** S2f: the staircase tile — row p*q_rows+h takes the DECODE tail-mask row for position
 *  first+p (the 32 tail masks' own bytes; a full last tile takes zeros), padding rows zero. */
static int pn_tt_attn_rows_mask(pn_tt_device *device, uint32_t first_position, uint32_t rows,
                                uint32_t chunk)
{
    const uint32_t key = (first_position % PN_TT_TILE_DIM) | (rows << 8) | (1u << 16);
    const size_t tile = PN_TT_TILE_BF16;
    uint16_t out[PN_TT_TILE_BF16 / 2u];
    uint32_t p, h, c;

    if (chunk >= PN_TT_ATTN_ROWS_CHUNKS)
        return -1;
    if (device->attn_rows_mask_key[chunk] == key)
        return 0;
    if (!device->attn_constants_host) {
        const size_t bytes = ft_tt_op_attention_constants_bytes(PN_TT_TILE_BF16);

        device->attn_constants_host = malloc(bytes);
        if (!device->attn_constants_host ||
            ft_tt_op_attention_build_constants(device->attn_head_dim, PN_TT_TILE_BF16,
                                               device->attn_constants_host, bytes) != FT_TT_OK)
            return -1;
    }
    if (device->attn_rows_mask_address == 0u) {
        /* ON THE CONSTANTS' BANK: the descriptor carries one coordinate for both. */
        ft_tt_buffer buffer;

        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, device->attn_const_bank,
                               tile * PN_TT_ATTN_ROWS_CHUNKS, &buffer) != FT_TT_OK)
            return -1;
        device->attn_rows_mask_address = (uint32_t)buffer.address;
    }
    memset(out, 0, sizeof(out));
    for (p = 0; p < rows; p++) {
        /* RED `attn_rows_flat`: every row takes the LAST row's length — the earlier rows see
         * the later drafts' K/V. Must DIFFER. */
        const uint32_t live = (first_position + (pn_tt_red("attn_rows_flat") ? rows - 1u : p)) %
                                  PN_TT_TILE_DIM + 1u; /* in the last tile */
        const uint16_t *mask = live == PN_TT_TILE_DIM
                                   ? NULL
                                   : (const uint16_t *)(const void *)(device->attn_constants_host +
                                                                      (FT_TT_OP_ATTENTION_CONST_MASK_TILE +
                                                                       live) * tile);

        for (h = 0; h < device->attn_q_rows; h++) {
            const uint32_t r = p * device->attn_q_rows + h;

            for (c = 0; c < PN_TT_TILE_DIM && mask; c++)
                out[ft_tt_tile_face_index(r, c)] = mask[ft_tt_tile_face_index(r, c)];
        }
    }
    if (pn_tt_write_dram(device, device->attn_const_bank,
                         device->attn_rows_mask_address + chunk * (uint32_t)tile, out, tile) != 0)
        return -1;
    device->attn_rows_mask_key[chunk] = key;
    return 0;
}

/** One ROWS descriptor a KV group for strip rows [query_row, query_row + rows), which sit at
 *  positions first_position.. (rows * q_rows <= 32), with mask tile `chunk`. */
static int pn_tt_attention_rows_chunk(pn_tt_device *device, uint32_t layer,
                                      const pn_tt_activation *query,
                                      const pn_tt_activation *output, uint32_t first_position,
                                      uint32_t query_row, uint32_t rows, uint32_t chunk)
{
    const uint32_t group_tiles = device->attn_q_rows * device->attn_dim_tiles;
    const uint32_t kv_positions = first_position + rows;
    uint32_t group;

    if (pn_tt_attn_rows_mask(device, first_position, rows, chunk) != 0)
        return -1;
    for (group = 0; group < device->attn_kv_heads; group++) {
        ft_tt_op_attention_request request;
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];
        uint64_t run_offset = 0;
        uint32_t run_bytes = 0;

        if (ft_tt_kv_cache_run(&device->kv_layout, group, 0u, PN_TT_TILE_DIM, &run_offset,
                               &run_bytes) != FT_TT_OK)
            return -1;
        memset(&request, 0, sizeof(request));
        request.q_rows = device->attn_q_rows;
        request.dim_tiles = device->attn_dim_tiles;
        request.block_tiles = device->attn_block_tiles;
        request.kv_positions = kv_positions;
        request.capacity_tiles = device->kv_layout.capacity_tiles;
        request.query_row = query_row;
        request.flags = FT_TT_OP_ATTENTION_FLAG_ROWS;
        request.positions = rows;
        request.rows_mask_address = device->attn_rows_mask_address +
                                    chunk * (uint32_t)PN_TT_TILE_BF16;
        request.query = pn_tt_operand(device, query);
        request.query.address += group * group_tiles * PN_TT_TILE_BF16;
        request.keys = pn_tt_region_operand(device, layer, false);
        request.keys.address += (uint32_t)run_offset;
        request.values = pn_tt_region_operand(device, layer, true);
        request.values.address += (uint32_t)run_offset;
        request.constants.x = (uint16_t)device->dram[device->attn_const_bank].x;
        request.constants.y = (uint16_t)device->dram[device->attn_const_bank].y;
        request.constants.address = device->attn_const_address;
        request.constants.tile_bytes = PN_TT_TILE_BF16;
        request.output = pn_tt_operand(device, output);
        request.output.address += group * group_tiles * PN_TT_TILE_BF16;
        if (ft_tt_op_attention_parameters(&request, words) != FT_TT_OK ||
            pn_tt_attention_submit(device, &device->attention, FT_TT_QUEUE_OP_ATTENTION,
                                   words) != 0)
            return -1;
    }
    return 0;
}

int pn_tt_attention_rows(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                         const pn_tt_activation *output, uint32_t first_position, uint32_t rows)
{
    const uint32_t group_tiles = device ? device->attn_q_rows * device->attn_dim_tiles : 0u;
    const uint32_t kv_positions = first_position + rows;
    uint32_t per, done, chunk;

    if (!device || !query || !output || !device->attention.running || rows == 0u ||
        device->attn_q_rows == 0u ||
        first_position / PN_TT_TILE_DIM != (kv_positions - 1u) / PN_TT_TILE_DIM)
        return -1;
    /* A DECODE TILE PACKS 32 / q_rows POSITIONS (8 at Ornith's 4 heads a group), so a wider
     * verify (DFlash's 16) runs as chunks of that many rows, each its own descriptor and mask
     * tile; the kernels' `query_row` places a chunk's rows in the strip. */
    per = PN_TT_TILE_DIM / device->attn_q_rows;
    if ((rows + per - 1u) / per > PN_TT_ATTN_ROWS_CHUNKS)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (layer >= device->kv_layer_count ||
        query->tile_count < device->attn_kv_heads * group_tiles ||
        output->tile_count < device->attn_kv_heads * group_tiles)
        return -1;
    for (done = 0, chunk = 0; done < rows; done += per, chunk++) {
        const uint32_t n = rows - done < per ? rows - done : per;

        if (pn_tt_attention_rows_chunk(device, layer, query, output, first_position + done,
                                       done, n, chunk) != 0)
            return -1;
    }
    return 0;
}

int pn_tt_kv_append_row(pn_tt_device *device, uint32_t layer, bool values,
                        const pn_tt_activation *source, uint32_t source_row, uint32_t position)
{
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];

    if (!device || !source || !device->memory.running || layer >= device->kv_layer_count ||
        source_row >= PN_TT_TILE_DIM)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    memset(&request, 0, sizeof(request));
    request.cache = &device->kv_layout;
    request.position = position;
    request.token_count = 1u;
    request.source = pn_tt_operand(device, source);
    /* RED `spec_kv_row0`: every verify row appends row 0's K/V — must DIFFER. */
    request.source_row0 = pn_tt_red("spec_kv_row0") ? 0u : source_row;
    request.source_tiles = device->attn_kv_heads * device->attn_dim_tiles;
    request.region = pn_tt_region_operand(device, layer, values);
    if (source->tile_count < request.source_tiles ||
        ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK)
        return -1;
    return pn_tt_attention_submit(device, &device->memory, FT_TT_QUEUE_OP_KV_APPEND, words);
}

int pn_tt_kv_append_chunk(pn_tt_device *device, uint32_t layer, bool values,
                          const pn_tt_activation *source, uint32_t first_position,
                          uint32_t rows)
{
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];

    if (!device || !source || !device->memory.running || layer >= device->kv_layer_count)
        return -1;
    if (rows == 0u || rows > PN_TT_TILE_DIM || (first_position % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;

    memset(&request, 0, sizeof(request));
    request.cache = &device->kv_layout;
    request.position = first_position;
    /* RED PROOF: the whole chunk lands one position late, so every row is a K row that
     * belongs to the token after it. It stays INSIDE the region for the same reason the
     * per-position arm's does. */
    if (pn_tt_red("kv_position") && first_position + rows < device->kv_layout.capacity_positions)
        request.position = first_position + 1u;
    request.token_count = rows;
    request.source = pn_tt_operand(device, source);
    /* Row t of the strip IS position `first_position + t`, and the cache wants position p at
     * row `p % 32` — identical when the first position is a multiple of 32, which is why the
     * alignment is a refusal above and not a convenience. */
    request.source_row0 = 0u;
    request.source_tiles = device->attn_kv_heads * device->attn_dim_tiles;
    request.region = pn_tt_region_operand(device, layer, values);
    if (source->tile_count < request.source_tiles)
        return -1;
    if (ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the %u-position KV append at %u of layer %u is not "
                        "encodable\n", rows, first_position, layer);
        return -1;
    }
    return pn_tt_attention_submit(device, &device->memory, FT_TT_QUEUE_OP_KV_APPEND, words);
}

/* ================================================================
 * Flash decode
 * ================================================================ */

int pn_tt_attention(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                    const pn_tt_activation *output, uint32_t kv_positions)
{
    const uint32_t group_tiles = device->attn_q_rows * device->attn_dim_tiles;
    uint32_t group;

    if (!device || !query || !output || !device->attention.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (layer >= device->kv_layer_count || kv_positions == 0u)
        return -1;
    if (query->tile_count < device->attn_kv_heads * group_tiles ||
        output->tile_count < device->attn_kv_heads * group_tiles)
        return -1;

    /* ONE DESCRIPTOR PER KV GROUP, because the unit of work is the group and not the Q head:
     * the eight Q heads sharing a KV head are the eight ROWS of one tile, so the cache is
     * walked once per layer per token instead of eight times (ft_tt_op_attention.h). */
    for (group = 0; group < device->attn_kv_heads; group++) {
        ft_tt_op_attention_request request;
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];
        uint64_t run_offset = 0;
        uint32_t run_bytes = 0;

        /* RED PROOF: every group reads KV head 0. Both regions are the same size and both
         * are live, so nothing about the transfer changes — only which head's cache the
         * second group attends to, which is exactly the class of mistake a correlation
         * metric cannot see. */
        uint32_t kv_head = pn_tt_red("attn_group") ? 0u : group;

        if (ft_tt_kv_cache_run(&device->kv_layout, kv_head, 0u, PN_TT_TILE_DIM, &run_offset,
                               &run_bytes) != FT_TT_OK)
            return -1;

        memset(&request, 0, sizeof(request));
        request.q_rows = device->attn_q_rows;
        request.dim_tiles = device->attn_dim_tiles;
        request.block_tiles = device->attn_block_tiles;
        request.kv_positions = kv_positions;
        request.capacity_tiles = device->kv_layout.capacity_tiles;
        request.query_row = 0u;
        request.query = pn_tt_operand(device, query);
        request.query.address += group * group_tiles * PN_TT_TILE_BF16;
        request.keys = pn_tt_region_operand(device, layer, false);
        request.keys.address += (uint32_t)run_offset;
        request.values = pn_tt_region_operand(device, layer, true);
        request.values.address += (uint32_t)run_offset;
        request.constants.x = (uint16_t)device->dram[device->attn_const_bank].x;
        request.constants.y = (uint16_t)device->dram[device->attn_const_bank].y;
        request.constants.address = device->attn_const_address;
        request.constants.tile_bytes = PN_TT_TILE_BF16;
        request.output = pn_tt_operand(device, output);
        request.output.address += group * group_tiles * PN_TT_TILE_BF16;
        if (ft_tt_op_attention_parameters(&request, words) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: attention over %u positions is not encodable at "
                            "block_tiles %u against a %u-tile capacity\n",
                    kv_positions, device->attn_block_tiles, device->kv_layout.capacity_tiles);
            return -1;
        }
        if (pn_tt_attention_submit(device, &device->attention, FT_TT_QUEUE_OP_ATTENTION,
                                   words) != 0)
            return -1;
    }
    return 0;
}

/** Encode the chunk descriptor for Q head `head`. All words are DRAM addresses. */
static int attention_chunk_head_words(pn_tt_device *device, uint32_t layer,
                                      const pn_tt_activation *query,
                                      const pn_tt_activation *output, uint32_t first_position,
                                      uint32_t rows, uint32_t head,
                                      uint32_t words[FT_TT_OP_ATTENTION_WORDS])
{
    const uint32_t head_tiles = device->attn_dim_tiles;
    ft_tt_op_attention_request request;
    uint64_t run_offset = 0;
    uint32_t run_bytes = 0;

    /* CONTIGUOUS GQA, which is attention's rule and NOT the GDN block's interleaved one
     * (spec §4). Head h reads group `h / q_rows`, the same grouping the per-position arm
     * walks a group at a time. */
    uint32_t kv_head = head / device->attn_q_rows;

    /* RED PROOF: every head reads KV head 0 — the same mutation the per-position arm
     * carries, and the class of mistake no correlation metric can see. */
    if (pn_tt_red("attn_group"))
        kv_head = 0u;

    if (ft_tt_kv_cache_run(&device->kv_layout, kv_head, 0u, PN_TT_TILE_DIM, &run_offset,
                           &run_bytes) != FT_TT_OK)
        return -1;

    memset(&request, 0, sizeof(request));
    request.q_rows = rows;
    request.dim_tiles = head_tiles;
    request.block_tiles = device->attn_block_tiles;
    request.kv_positions = first_position + rows;
    request.capacity_tiles = device->kv_layout.capacity_tiles;
    request.query_row = 0u;
    request.flags = FT_TT_OP_ATTENTION_FLAG_CHUNK;
    request.query = pn_tt_operand(device, query);
    request.query.address += head * head_tiles * PN_TT_TILE_BF16;
    request.keys = pn_tt_region_operand(device, layer, false);
    request.keys.address += (uint32_t)run_offset;
    request.values = pn_tt_region_operand(device, layer, true);
    request.values.address += (uint32_t)run_offset;
    request.constants.x = (uint16_t)device->dram[device->attn_const_bank].x;
    request.constants.y = (uint16_t)device->dram[device->attn_const_bank].y;
    request.constants.address = device->attn_const_address;
    request.constants.tile_bytes = PN_TT_TILE_BF16;
    request.output = pn_tt_operand(device, output);
    request.output.address += head * head_tiles * PN_TT_TILE_BF16;
    if (ft_tt_op_attention_parameters(&request, words) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: a %u-position chunk at %u over %u positions is not "
                        "encodable at block_tiles %u against a %u-tile capacity\n",
                rows, first_position, first_position + rows, device->attn_block_tiles,
                device->kv_layout.capacity_tiles);
        return -1;
    }
    return 0;
}

/**
 * THE HEAD SPLIT: `attention_members` cores, each owning a contiguous range of Q heads —
 * contiguous so a member's heads share KV groups. Round r enqueues one head on EVERY member
 * and then awaits them all, so the members run concurrently while each still holds ONE
 * descriptor's parameters at a time. A member that fails is re-armed and its head submitted
 * alone; every enqueued member is awaited first, or its cursor desynchronises.
 */
static int attention_chunk_split(pn_tt_device *device, uint32_t layer,
                                 const pn_tt_activation *query, const pn_tt_activation *output,
                                 uint32_t first_position, uint32_t rows)
{
    const uint32_t members = device->attention_members;
    const uint32_t per = device->attn_q_heads / members;
    uint32_t words[PN_TT_ATTN_MAX_MEMBERS][FT_TT_OP_ATTENTION_WORDS];
    pn_tt_residency *set[PN_TT_ATTN_MAX_MEMBERS];
    bool bad[PN_TT_ATTN_MAX_MEMBERS];
    uint32_t round, m;

    for (m = 0; m < members; m++)
        set[m] = m == 0u ? &device->attention : &device->attention_member[m - 1u];
    for (round = 0; round < per; round++) {
        const double started = pn_tt_now_seconds();
        int failed = 0;

        memset(bad, 0, sizeof(bad));
        for (m = 0; m < members; m++)
            if (attention_chunk_head_words(device, layer, query, output, first_position, rows,
                                           m * per + round, words[m]) != 0)
                return -1;
        for (m = 0; m < members; m++)
            if (pn_tt_residency_enqueue(set[m], FT_TT_QUEUE_OP_ATTENTION, words[m], 1u) != 0)
                failed = bad[m] = 1;
        set[0]->seconds += pn_tt_now_seconds() - started;
        for (m = 0; m < members; m++) {
            const double at = pn_tt_now_seconds();
            double elapsed;

            if (pn_tt_residency_await(set[m], PN_TT_OP_TIMEOUT_MS) != 0)
                failed = bad[m] = 1;
            elapsed = pn_tt_now_seconds() - at;
            device->attention_member_seconds[m] += elapsed;
            set[m]->seconds += elapsed;
        }
        if (failed) {
            for (m = 0; m < members; m++) {
                if (!bad[m])
                    continue;
                if (!(m == 0u ? pn_tt_attention_rearm(device, set[0])
                              : pn_tt_attention_member_rearm(device, m)) ||
                    pn_tt_residency_submit(set[m], FT_TT_QUEUE_OP_ATTENTION, words[m],
                                           PN_TT_OP_TIMEOUT_MS) != 0)
                    return -1;
            }
        }
        device->oncard_seconds += pn_tt_now_seconds() - started;
        device->oncard_ops += members;
    }
    return 0;
}

int pn_tt_attention_chunk(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                          const pn_tt_activation *output, uint32_t first_position,
                          uint32_t rows)
{
    const uint32_t head_tiles = device ? device->attn_dim_tiles : 0u;
    uint32_t head;

    if (!device || !query || !output || !device->attention.running)
        return -1;
    if (rows == 0u || rows > PN_TT_TILE_DIM || (first_position % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (layer >= device->kv_layer_count)
        return -1;
    if (query->tile_count < device->attn_q_heads * head_tiles ||
        output->tile_count < device->attn_q_heads * head_tiles)
        return -1;

    /* ONE DESCRIPTOR PER Q HEAD, where decode issues one per KV GROUP — because the row axis
     * that decode fills with the group's Q heads is the axis a chunk fills with POSITIONS,
     * and a score matmul contracts over the head dimension, so two heads cannot share it.
     * The trade is measured rather than argued: `q_heads` descriptors replace
     * `kv_heads * rows` (16 against 128 here) at essentially the same cost each, so both the
     * descriptor count and the number of KV walks fall by `32 * kv_heads / q_heads`. */
    if (device->attention_members > 1u)
        return attention_chunk_split(device, layer, query, output, first_position, rows);
    for (head = 0; head < device->attn_q_heads; head++) {
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];

        if (attention_chunk_head_words(device, layer, query, output, first_position, rows,
                                       head, words) != 0)
            return -1;
        if (pn_tt_attention_submit(device, &device->attention, FT_TT_QUEUE_OP_ATTENTION,
                                   words) != 0)
            return -1;
    }
    return 0;
}

/* ================================================================
 * The export path — the only place the card cache crosses the link
 * ================================================================ */

/** Bytes one head's `tiles` position tiles occupy, and the offset they start at. */
static int pn_tt_kv_head_run(const pn_tt_device *device, uint32_t head, uint32_t tiles,
                             uint64_t *out_offset, uint32_t *out_bytes)
{
    return ft_tt_kv_cache_run(&device->kv_layout, head, 0u, tiles * PN_TT_TILE_DIM, out_offset,
                              out_bytes) == FT_TT_OK
               ? 0
               : -1;
}

int pn_tt_kv_download(pn_tt_device *device, uint32_t layer, bool values,
                      uint32_t position_count, float *out)
{
    const uint32_t dim_tiles = device ? device->attn_dim_tiles : 0u;
    const uint32_t head_dim = device ? device->attn_head_dim : 0u;
    uint32_t tiles, head;
    uint16_t *staged;
    double started;

    if (!device || !out || position_count == 0u || layer >= device->kv_layer_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (position_count > device->kv_layout.capacity_positions)
        return -1;
    tiles = (position_count + PN_TT_TILE_DIM - 1u) / PN_TT_TILE_DIM;
    staged = malloc((size_t)tiles * dim_tiles * PN_TT_TILE_BF16);
    if (!staged)
        return -1;
    started = pn_tt_now_seconds();

    for (head = 0; head < device->attn_kv_heads; head++) {
        ft_tt_op_operand region = pn_tt_region_operand(device, layer, values);
        uint32_t bank = values ? device->kv_region[layer].v_bank
                               : device->kv_region[layer].k_bank;
        uint64_t offset = 0;
        uint32_t bytes = 0, position;

        if (pn_tt_kv_head_run(device, head, tiles, &offset, &bytes) != 0 ||
            pn_tt_read_dram(device, bank, region.address + (uint32_t)offset, staged, bytes) !=
                0) {
            free(staged);
            return -1;
        }
        /* WHOLE TILES, so there is no read-alignment waste at all — which is the arithmetic
         * that makes exporting at export time cheaper than mirroring per token. */
        for (position = 0; position < position_count; position++) {
            const uint16_t *tile_row = staged + (size_t)(position / PN_TT_TILE_DIM) *
                                                    dim_tiles * (PN_TT_TILE_BF16 / 2u);
            float *destination = out + ((size_t)head * position_count + position) * head_dim;
            uint32_t element;

            for (element = 0; element < head_dim; element++) {
                const uint16_t *tile = tile_row + (size_t)(element / PN_TT_TILE_DIM) *
                                                      (PN_TT_TILE_BF16 / 2u);

                destination[element] = ft_float_from_bf16(
                    tile[ft_tt_tile_face_index(position % PN_TT_TILE_DIM,
                                               element % PN_TT_TILE_DIM)]);
            }
        }
    }
    free(staged);
    device->readback_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_kv_upload(pn_tt_device *device, uint32_t layer, bool values,
                    uint32_t position_count, const float *in)
{
    const uint32_t dim_tiles = device ? device->attn_dim_tiles : 0u;
    const uint32_t head_dim = device ? device->attn_head_dim : 0u;
    uint32_t tiles, head;
    uint16_t *staged;
    double started;

    if (!device || !in || position_count == 0u || layer >= device->kv_layer_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (position_count > device->kv_layout.capacity_positions)
        return -1;
    tiles = (position_count + PN_TT_TILE_DIM - 1u) / PN_TT_TILE_DIM;
    staged = calloc((size_t)tiles * dim_tiles, PN_TT_TILE_BF16);
    if (!staged)
        return -1;
    started = pn_tt_now_seconds();

    for (head = 0; head < device->attn_kv_heads; head++) {
        ft_tt_op_operand region = pn_tt_region_operand(device, layer, values);
        uint32_t bank = values ? device->kv_region[layer].v_bank
                               : device->kv_region[layer].k_bank;
        uint64_t offset = 0;
        uint32_t bytes = 0, position;

        memset(staged, 0, (size_t)tiles * dim_tiles * PN_TT_TILE_BF16);
        for (position = 0; position < position_count; position++) {
            uint16_t *tile_row = staged + (size_t)(position / PN_TT_TILE_DIM) * dim_tiles *
                                              (PN_TT_TILE_BF16 / 2u);
            const float *source = in + ((size_t)head * position_count + position) * head_dim;
            uint32_t element;

            for (element = 0; element < head_dim; element++) {
                uint16_t *tile = tile_row + (size_t)(element / PN_TT_TILE_DIM) *
                                                (PN_TT_TILE_BF16 / 2u);

                tile[ft_tt_tile_face_index(position % PN_TT_TILE_DIM,
                                           element % PN_TT_TILE_DIM)] =
                    ft_bf16_from_float(source[element]);
            }
        }
        if (pn_tt_kv_head_run(device, head, tiles, &offset, &bytes) != 0 ||
            pn_tt_write_dram(device, bank, region.address + (uint32_t)offset, staged, bytes) !=
                0) {
            free(staged);
            return -1;
        }
    }
    free(staged);
    device->upload_seconds += pn_tt_now_seconds() - started;
    return 0;
}

#endif /* USE_TT */
