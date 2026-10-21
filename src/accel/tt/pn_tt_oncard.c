/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_oncard.c
 * @brief The ops that used to be host arithmetic between two PCIe transfers: the RMS norms,
 *        the SwiGLU gate, the residual adds and the LM head's top-N — each on its own
 *        resident core, passing activations to the matmul through card DRAM.
 *
 *        pn_tt_device.c owns the card and the matmul; this file owns the other three
 *        residencies and the DRAM strips they share. The public API is in pn_tt_device.h.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS THE WHOLE REMAINING WALL, AS A NUMBER
 * ---------------------------------------------------------------------------
 * A token was 0.568 s of which readback was 0.412 — 73% — because every projection's result
 * crossed a link that does ~4.7 MB/s card->host with no device->host DMA. Nothing about
 * that transfer can be made faster; the only move left is not to make it. Each op moved here
 * removes the readback of what it consumes AND the upload of what it produces, and the ones
 * that remain (q/k/v, for host RoPE) are remaining because of a named capability gap and not
 * because of a preference.
 *
 * ---------------------------------------------------------------------------
 * THE THREE INVARIANTS A CALLER OF THESE OPS IS RESPONSIBLE FOR
 * ---------------------------------------------------------------------------
 *  1. **Rows 1-31 of every strip are zero and stay zero.** A matvec is Mt=1; the other 31
 *     rows are the tile quantum and the FPU sees all of them. pn_tt_activation_alloc() zeroes
 *     once, and every op here preserves it for free — a matmul of a zero row is zero,
 *     silu(0)*0 is zero, 0+0 is zero, and an RMS norm of a zero row is 0*gamma*rsqrt(eps).
 *     An op that did NOT preserve it would corrupt every later token, silently.
 *     - **MEASURED ON THE ON-CARD ATTENTION ARM 2026-08-16, and the invariant HOLDS — it was
 *       never weakened.** The worry was that flash decode gathers a KV group's eight Q heads
 *       into the eight ROWS of a staging tile, so rows 1-31 of `attn` — and through the
 *       residual add, of `hidden` — might not be zero there. That was recorded as INFERRED
 *       harmless and NOT measured. It is measured now, with the whole-tile reader
 *       `pn_tt_activation_download_tile_row()` over every one of rows 1-31 of both strips,
 *       layers rotating with the position: **max|v| = 0 exactly, 0 non-finite.** The gather is
 *       internal staging; the writeback lands each of the group's eight Q heads at ROW 0 of
 *       its own `dim_tiles` tiles, which is why the invariant survives.
 *       - **The census carries its own control**, because a reader that returned zeros would
 *         have reported the invariant holding. Row 0 read through the SAME whole-tile path
 *         comes back non-zero (0.68 on `attn`, 3.97 on `hidden`) and agrees with the shipping
 *         row-0 reader — different bytes, different path — to **exactly 0**.
 *  2. **ONE block depth for every descriptor of one residency.** Circular buffers are placed
 *     at launch. A descriptor whose depth or tile count leaves the CB write pointer off the
 *     block grid makes the NEXT descriptor step OVER the buffer limit instead of wrapping,
 *     after which every block walks further up L1 with no fault and no error — measured, and
 *     it landed a read on the queue control block (foundry/tt/ft_tt_op_sample.h). So the
 *     depth is chosen ONCE, at arm time, against every tile count the run will use, and a
 *     shape that does not fit is refused rather than served.
 *  3. **A residency that refuses a descriptor is STRANDED, not wedged.** The reader leaves
 *     its loop and `tail` never advances, so the other four processors wait forever on work
 *     that will not arrive. pn_tt_residency_submit() sees it in ~1 ms rather than at the
 *     timeout, and the recovery is to retire by ASKING and re-arm — never a reset, never a
 *     park.
 *
 * ---------------------------------------------------------------------------
 * THE THREE PROGRAMS, AND HOW THEY ARE BUILT
 * ---------------------------------------------------------------------------
 * `reduce`, `eltwise` and `topk` ride in the binary's embedded blob beside the matmul ones,
 * and are rebuilt into it with `cd foundry/tt/tools && make regen-programs`. The recipes
 * below are what the blob was generated FROM; `PN_TT_PROGRAM_DIR` reads a directory built
 * this way instead, for a developer testing a program they just built. Every
 * compile-time argument below is read off the kernels' own `ft_tensix_compile_time_arg()`
 * indices rather than carried over from another op — carrying them over is what produced the
 * Makefile.memory bug. From foundry/tt, with RELOCATE_KERNEL_TEXT=1 on all three:
 *
 *   eltwise  READER_CTARGS=0,1,2048,1  COMPUTE_CTARGS=0,1,2,16,1  WRITER_CTARGS=16,2048,1
 *            CB_TABLE=kernels/cb_tables/eltwise_float16b.cbtable
 *            resident_eltwise_reader.c resident_eltwise_compute.c resident_vector_writer.c
 *   reduce   READER_CTARGS=0,1,5,6,7,2048,1  COMPUTE_CTARGS=0,1,2,3,4,5,6,7,8,9,16,1
 *            WRITER_CTARGS=16,2048,1  CB_TABLE=kernels/cb_tables/reduce_float16b.cbtable
 *            resident_reduce_reader.c resident_reduce_compute.c resident_vector_writer.c
 *   topk     resident_topk_reader.c resident_topk_compute.c resident_topk_select.c
 *
 * `tools/check_tensix_program_sources.sh <dir>` is not optional afterwards: all seven
 * directories once carried `.src` sidecars naming sources deleted weeks earlier, and every
 * green they produced came from images nobody could rebuild.
 *
 * @status Norm, SwiGLU, residual add and top-N green on silicon, and each residency is
 *         re-armed when it refuses a descriptor (pn_tt_oncard_rearm) — which is what a
 *         serve needs, because its residencies idle between requests and retire. The GDN
 *         family is selected here and armed in pn_tt_gdn.c; it has never run on silicon.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_device_internal.h"
#include "pn_tt_core_boot_guard.h"
#include "pn_tt_eltwise_group.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/ft_tt_op_sample.h"
#include "tt/ft_tt_op_vector.h"
#include "tt/ft_tt_program_resolve.h"
#include "tt/ft_tt_tile_row0.h"
#include "tt/ft_tt_tilize.h"

/** The three programs this file arms, by NAME. Each is a whole program — five kernel images
 *  plus the firmware they were linked against — and each gets its own core. They come out of
 *  the binary's own blob unless PN_TT_PROGRAM_DIR names a directory instead.
 *
 *  `PN_TT_ELTWISE_PROGRAM` and `PN_TT_CORE_ELTWISE_*` are in `pn_tt_eltwise_group.h`, because
 *  that module arms members of the same family on other cores and re-arms member 0 on a
 *  refusal — one spelling, one default, one place. */
#define PN_TT_REDUCE_PROGRAM "reduce"
#define PN_TT_SAMPLE_PROGRAM "topk"

/** Which enabled Tensix each residency drives. Defaults, not constants — a serve with a
 *  placement should pass its own, and these must be three cores nobody else holds. */
#define PN_TT_CORE_REDUCE_ENV     "PN_TT_CORE_REDUCE"
#define PN_TT_CORE_SAMPLE_ENV     "PN_TT_CORE_SAMPLE"
#define PN_TT_CORE_REDUCE_DEFAULT 72u
#define PN_TT_CORE_SAMPLE_DEFAULT 74u

/** Which host ops this run moves onto the card: a comma list of the names below, or `all`
 *  (the default) or `none`. It exists so the A/B is across ONE binary running the same
 *  kernels on the same weights — the same reason PN_TT_DISPATCH does. */
#define PN_TT_ONCARD_ENV "PN_TT_ONCARD"

/** Pairs the sampling op selects. The forward needs the argmax; a handful more costs
 *  nothing on the card and makes the result block readable as evidence rather than as one
 *  number that could be anything. */
#define PN_TT_TOPK_SELECT 8u
/** Per-call N ceiling (the kernel's kTopkMaxSelect): sampled requests ask for 64. */
#define PN_TT_TOPK_SELECT_MAX 64u

/** The greedy sampler state this file arms: temperature <= 0 is a pure argmax on the RAW
 *  values, which is exactly what a host serial scan over the same row computes — that is
 *  what makes EXACT INDEX AGREEMENT the honest gate rather than a correlation. */
#define PN_TT_TOPK_RNG_SEED 0x9E3779B9u

static const char *const pn_tt_oncard_names[PN_TT_ONCARD_COUNT] = { "norm", "swiglu",
                                                                    "residual", "topk",
                                                                    "attention", "gdn" };

/**
 * RED PROOFS. `PN_TT_RED=<name>` deliberately breaks ONE on-card op so the acceptance gate
 * can be shown to CATCH it.
 *
 * It is a knob and not a test-only build because what is being proven is a property of the
 * SHIPPING path — that the 20-probe argmax set is still sensitive to each op after that op
 * stopped producing a number the host ever looks at. A separate binary would prove it about
 * a different program. Every mutation below changes the op's ARITHMETIC and none of them
 * changes a length, an address or a tile count: a mutation that SHORTENS a transfer can
 * reach zero and take the card down instead of going red (DEV_NOTES.md §1).
 *
 *   norm_eps       the reduce's eps is 1e6x too large — a wrong norm, smoothly
 *   swiglu_swap    silu(up) * gate instead of silu(gate) * up — a swapped operand
 *   residual_sub   hidden - project instead of hidden + project
 *   topk_row       select over logical row 1 of each tile, which is the zero padding
 *   rope_position  the rotary table is built for position 0 instead of the token's
 *   attn_group     every attention descriptor reads KV group 0
 *   kv_position    the append lands one position late
 */
#define PN_TT_RED_ENV "PN_TT_RED"

/** Is `name` the mutation this process was asked to run? Read every call; it fires for the
 *  whole run rather than once, because these break an op and not a single descriptor. */
bool pn_tt_red(const char *name)
{
    const char *value = getenv(PN_TT_RED_ENV);

    return value && *value && strcmp(value, name) == 0;
}

/* ================================================================
 * Selection
 * ================================================================ */

/** Does `list` (a comma-separated PN_TT_ONCARD value) name `word`? */
static bool pn_tt_list_contains(const char *list, const char *word)
{
    size_t length = strlen(word);
    const char *cursor = list;

    while (*cursor) {
        const char *comma = strchr(cursor, ',');
        size_t span = comma ? (size_t)(comma - cursor) : strlen(cursor);

        if (span == length && strncmp(cursor, word, length) == 0)
            return true;
        if (!comma)
            break;
        cursor = comma + 1;
    }
    return false;
}

/**
 * Read PN_TT_ONCARD into `device->oncard_requested`. Unset means all of them.
 *
 * **THE EXCEPTION TO `all` IS NOW PER MODEL FAMILY, NOT A CONSTANT.** There were two
 * globally opt-in families; GDN is measured away entirely, and attention is on for the family
 * that has the evidence and off for the one that does not — `device->oncard_default[]`, set
 * by `pn_tt_device_oncard_default()` from the backend that owns the model.
 *
 * GDN was opt-in because it was slower than the host recurrence until the two per-head norms
 * moved with it. They moved: the strip bridge runs the block whole on the card at readback
 * 326.09 KiB/token against 1478.09 and 3.79 tok/s against 2.20, `cmp` IDENTICAL.
 *
 * ATTENTION was opt-in because it moves WHERE THE KV CACHE LIVES, changes the order the
 * attention sum accumulates in, and changes what session export costs — a different forward
 * rather than a faster one. Each of those three is now settled rather than argued:
 *
 *   - the FORWARD passes a calibrated chat-parity gate — ratio 0.986080, t -1.608, disagree
 *     2.76%, worst document 0.989, all five criteria, against a gate proven to FAIL at
 *     sigma 0.2 (9.09% disagree) and 0.4. It costs +0.4% perplexity, agreed by two
 *     instruments on two corpora, and the 1e-3 intermediate bar it misses is the bar an arm
 *     whose 48-token output is BYTE-IDENTICAL to the oracle also misses;
 *   - the SPEED is better at every length measured — with the GDN block armed beside it,
 *     3.79 -> 5.02 tok/s at bench, and readback 326.09 -> 6.09 KiB/token, which is the top-N
 *     and the 24 gate blocks and nothing else;
 *   - SESSION EXPORT is the one real cost and it is now REFUSED rather than served wrong:
 *     `prefix_cache.c` asks `pn_qwen35moe_model_tt_kv_on_card()` before capturing, because
 *     the host K/V mirror is not written on this arm and a capture would record zeros.
 *
 * And it is the only path to long context on a 7.1 GB host: the host fp32 mirror is 65,536 B
 * a position, so a 256k session is 16 GiB on the host arm and nothing on this one.
 *
 * **NONE OF THAT EVIDENCE TRANSFERS TO THE LLAMA FAMILY, AND A GLOBAL DEFAULT WOULD HAVE
 * SHIPPED IT ANYWAY.** `llama_forward_tt.c` reads the same flag and arms the same op family
 * for MiniCPM5-1B, where on-card attention is recorded as FAILING the chat-parity gate — and
 * it arms LAZILY, after `pn_tt_device_arm_oncard()`, so nothing on that path would have
 * mentioned the change. That is why the default is a device field a backend sets for its own
 * model rather than a constant in this function.
 *
 * **A DEFAULT AND A NAMED FAMILY WANT OPPOSITE BEHAVIOUR ON A FAILED ARM, WHICH IS WHY
 * `oncard_named` EXISTS AND WHY IT IS NOW LOAD-BEARING FOR TWO FAMILIES.** GDN without the
 * bridge is 1.43 tok/s, so `qwen35moe_resident_tt.c` gives the block back to the host when it
 * armed by DEFAULT and the bridge did not come up. Attention that fails to arm by DEFAULT
 * runs the HOST attention block — correct, O(context), not the card arm — whereas one asked
 * for BY NAME refuses, because `return -1` there reaches `goto fail` in
 * pn_qwen35moe_tt_upload and drops the whole model to the CPU forward at ~0.05 tok/s. That is
 * the right price for an explicit request and the wrong one for a default.
 */
static void pn_tt_read_oncard_selection(pn_tt_device *device)
{
    const char *value = getenv(PN_TT_ONCARD_ENV);
    unsigned index;

    for (index = 0; index < PN_TT_ONCARD_COUNT; index++) {
        /* NAMED is tracked apart from REQUESTED because the two want OPPOSITE behaviour when
         * a family fails to arm, and one flag cannot express both. See the header above. */
        device->oncard_named[index] =
            value && *value && strcmp(value, "all") != 0 && strcmp(value, "none") != 0 &&
            pn_tt_list_contains(value, pn_tt_oncard_names[index]);
        if (!value || !*value || strcmp(value, "all") == 0)
            device->oncard_requested[index] = device->oncard_default[index];
        else if (strcmp(value, "none") == 0)
            device->oncard_requested[index] = false;
        else
            device->oncard_requested[index] = device->oncard_named[index];
    }
}

void pn_tt_device_oncard_default(pn_tt_device *device, pn_tt_oncard_op op, bool on)
{
    if (!device || (unsigned)op >= PN_TT_ONCARD_COUNT)
        return;
    device->oncard_default[op] = on;
    /* RE-READ, so the caller may ask `..._oncard_requested()` before anything is armed. It is
     * a getenv and a few strcmps, and `pn_tt_device_arm_oncard()` reads it again — but
     * without this the answer before the arm is whatever calloc left, and one caller already
     * relies on it: the attention sigmoid gate is a THIRD elementwise width that has to be
     * declared to `place_cbs`, and it is chosen from this question. */
    pn_tt_read_oncard_selection(device);
}

bool pn_tt_device_oncard_requested(const pn_tt_device *device, pn_tt_oncard_op op)
{
    if (!device || (unsigned)op >= PN_TT_ONCARD_COUNT)
        return false;
    return device->oncard_requested[op];
}

bool pn_tt_device_oncard_named(const pn_tt_device *device, pn_tt_oncard_op op)
{
    if (!device || (unsigned)op >= PN_TT_ONCARD_COUNT)
        return false;
    return device->oncard_named[op];
}

bool pn_tt_device_oncard(const pn_tt_device *device, pn_tt_oncard_op op)
{
    if (!device || (unsigned)op >= PN_TT_ONCARD_COUNT)
        return false;
    return device->oncard[op];
}

/* ================================================================
 * DRAM activation strips
 * ================================================================ */

int pn_tt_zero_dram(pn_tt_device *device, uint32_t bank, uint32_t address, size_t bytes)
{
    return pn_tt_fill_dram(device, bank, address, bytes, 0u);
}

int pn_tt_fill_dram(pn_tt_device *device, uint32_t bank, uint32_t address, size_t bytes,
                    unsigned char value)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    unsigned char *zeros = calloc(1, PN_TT_ZERO_CHUNK_BYTES);
    size_t offset = 0;
    int rc = FT_TT_OK;

    if (!zeros)
        return -1;
    if (value != 0u)
        memset(zeros, value, PN_TT_ZERO_CHUNK_BYTES);
    /* EVERY DEVICE OPERATION BUT THE ENQUEUE ITSELF WAITS FIRST. A chained projection
     * may still be writing the strip this one touches, and the members run
     * concurrently — so the rule lives at ONE seam instead of at every call site
     * (pn_tt_device.h, pn_tt_matvec_enqueue). It is a branch when nothing is pending. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    while (offset < bytes && rc == FT_TT_OK) {
        size_t chunk = bytes - offset;

        if (chunk > PN_TT_ZERO_CHUNK_BYTES)
            chunk = PN_TT_ZERO_CHUNK_BYTES;
        rc = bulk->write(bulk->user, (uint16_t)device->dram[bank].x,
                         (uint16_t)device->dram[bank].y, (uint64_t)address + offset, zeros,
                         chunk);
        offset += chunk;
    }
    free(zeros);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += bytes;
    return 0;
}

int pn_tt_write_dram(pn_tt_device *device, uint32_t bank, uint32_t address, const void *data,
                     size_t bytes)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    const unsigned char *source = (const unsigned char *)data;
    size_t offset = 0;
    int rc = FT_TT_OK;

    if (!device || !data || bank >= device->dram_bank_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    while (offset < bytes && rc == FT_TT_OK) {
        size_t chunk = bytes - offset;

        if (chunk > PN_TT_ZERO_CHUNK_BYTES)
            chunk = PN_TT_ZERO_CHUNK_BYTES;
        rc = bulk->write(bulk->user, (uint16_t)device->dram[bank].x,
                         (uint16_t)device->dram[bank].y, (uint64_t)address + offset,
                         source + offset, chunk);
        offset += chunk;
    }
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += bytes;
    return 0;
}

int pn_tt_read_dram(pn_tt_device *device, uint32_t bank, uint32_t address, void *out,
                    size_t bytes)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    unsigned char *destination = (unsigned char *)out;
    size_t offset = 0;
    int rc = FT_TT_OK;

    if (!device || !out || bank >= device->dram_bank_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    while (offset < bytes && rc == FT_TT_OK) {
        size_t chunk = bytes - offset;

        if (chunk > PN_TT_ZERO_CHUNK_BYTES)
            chunk = PN_TT_ZERO_CHUNK_BYTES;
        rc = bulk->read(bulk->user, (uint16_t)device->dram[bank].x,
                        (uint16_t)device->dram[bank].y, (uint64_t)address + offset,
                        destination + offset, chunk);
        offset += chunk;
    }
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_read += bytes;
    return 0;
}

int pn_tt_activation_alloc(pn_tt_device *device, uint32_t tile_count,
                           pn_tt_activation *out_activation)
{
    /* EVERY STRIP GETS THE SAME HEIGHT, resolved once. A strip one band tall beside strips
     * two bands tall is a matmul reading band 1 of A and writing band 1 of a C that has none
     * — off the end of the buffer, with every kernel retiring clean. (The one exception is a
     * whole SET of another height used only on its own — the prefill-only tall set.) */
    return pn_tt_activation_alloc_rows(device, tile_count, pn_tt_device_strip_tile_rows(device),
                                       out_activation);
}

int pn_tt_activation_alloc_rows(pn_tt_device *device, uint32_t tile_count, uint32_t tile_rows,
                                pn_tt_activation *out_activation)
{
    ft_tt_buffer buffer;
    uint32_t bank;

    if (!device || !out_activation || tile_count == 0u || tile_rows == 0u ||
        tile_rows > PN_TT_STRIP_MAX_TILE_ROWS || device->dram_bank_count == 0u)
        return -1;
    /* Round-robin the strips across banks so the reader pulling an operand and the writer
     * pushing a result are not usually the same endpoint — Phase 6.7 measured two endpoints
     * on one NoC row sharing a ceiling. */
    bank = device->strip_bank_cursor % device->dram_bank_count;
    device->strip_bank_cursor++;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank,
                           (size_t)tile_count * tile_rows * PN_TT_TILE_BF16,
                           &buffer) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full allocating a %u-tile x %u-band activation strip\n",
                tile_count, tile_rows);
        return -1;
    }
    memset(out_activation, 0, sizeof(*out_activation));
    out_activation->bank = buffer.logical_bank;
    out_activation->address = (uint32_t)buffer.address;
    out_activation->tile_count = tile_count;
    out_activation->tile_rows = tile_rows;
    out_activation->active_rows = 1u;
    out_activation->row_offset = 0u;
    /* Mirror 0 IS the strip, always, so every caller of every accessor below can index
     * replica 0 without asking whether this strip was replicated. */
    out_activation->replica_count = 1u;
    out_activation->replica_bank[0] = out_activation->bank;
    out_activation->replica_address[0] = out_activation->address;
    /* THE PRECONDITION every row-0 write depends on and none of them can check. */
    if (pn_tt_zero_dram(device, out_activation->bank, out_activation->address,
                        (size_t)tile_count * tile_rows * PN_TT_TILE_BF16) != 0) {
        fprintf(stderr, "pn_tt: could not stage a %u-tile strip zeroed\n", tile_count);
        return -1;
    }
    return 0;
}

int pn_tt_activation_alloc_replicated(pn_tt_device *device, uint32_t tile_count,
                                      uint32_t replicas, pn_tt_activation *out_activation)
{
    return pn_tt_activation_alloc_replicated_rows(device, tile_count, replicas,
                                                  pn_tt_device_strip_tile_rows(device),
                                                  out_activation);
}

int pn_tt_activation_alloc_replicated_rows(pn_tt_device *device, uint32_t tile_count,
                                           uint32_t replicas, uint32_t tile_rows,
                                           pn_tt_activation *out_activation)
{
    uint32_t index;

    if (!device || !out_activation)
        return -1;
    if (pn_tt_activation_alloc_rows(device, tile_count, tile_rows, out_activation) != 0)
        return -1;
    if (replicas <= 1u)
        return 0;
    /* MORE MIRRORS THAN BANKS IS A REFUSAL, NOT A WRAP. Wrapping would put two mirrors in
     * one bank, which is the exact condition this mechanism exists to remove, arriving
     * under the name of the fix and reporting the higher replica count. */
    if (replicas > PN_TT_ACTIVATION_MAX_REPLICAS || replicas > device->dram_bank_count) {
        fprintf(stderr, "pn_tt: %u replicas of a %u-tile strip is more than the %u banks "
                        "this card has\n", replicas, tile_count, device->dram_bank_count);
        return -1;
    }

    for (index = 1u; index < replicas; index++) {
        ft_tt_buffer buffer;
        /* STEP ONE BANK PER MIRROR from the strip's own. The strips themselves are already
         * round-robined, so consecutive strips start in consecutive banks and a stride of
         * one visits every bank whatever the replica count is. */
        const uint32_t bank = (out_activation->bank + index) % device->dram_bank_count;

        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank,
                               (size_t)tile_count * out_activation->tile_rows * PN_TT_TILE_BF16,
                               &buffer) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: DRAM full allocating mirror %u of a %u-tile strip "
                            "(bank %u)\n", index, tile_count, bank);
            return -1;
        }
        out_activation->replica_bank[index] = buffer.logical_bank;
        out_activation->replica_address[index] = (uint32_t)buffer.address;
        /* Every mirror carries the strip invariant too: rows 1-31 zero. A mirror the
         * producer has not written yet is read by a matmul member from token 0. */
        if (pn_tt_zero_dram(device, buffer.logical_bank, (uint32_t)buffer.address,
                            (size_t)tile_count * out_activation->tile_rows *
                                PN_TT_TILE_BF16) != 0) {
            fprintf(stderr, "pn_tt: could not stage mirror %u zeroed\n", index);
            return -1;
        }
        out_activation->replica_count = index + 1u;
    }
    return 0;
}

int pn_tt_activation_upload(pn_tt_device *device, const pn_tt_activation *activation,
                            const float *values, uint32_t count)
{
    const ft_tt_noc_access *bulk;
    uint32_t tiles;
    double started;
    int rc;

    if (!device || !activation || !values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count || count > device->max_row_features)
        return -1;
    bulk = &device->runtime.bulk_dram;
    started = pn_tt_now_seconds();

    ft_bf16_from_float_array(values, device->row_in, count);
    /* EVERY MIRROR, or the ones this call did not write go stale and a matmul member reads
     * the previous token's value. The rounding is done once and the same bf16 row is sent
     * to each, so the mirrors are bit-identical by construction rather than by two
     * conversions agreeing. */
    for (uint32_t replica = 0u; replica < activation->replica_count; replica++) {
        const uint32_t bank = activation->replica_bank[replica];

        rc = ft_tt_tile_row0_write(bulk, (uint16_t)device->dram[bank].x,
                                   (uint16_t)device->dram[bank].y,
                                   activation->replica_address[replica], tiles, device->row_in);
        if (rc != FT_TT_OK)
            return -1;
        device->bytes_written +=
            (size_t)tiles * FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_HALF_BYTES;
    }
    device->upload_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_activation_download(pn_tt_device *device, const pn_tt_activation *activation,
                              float *out_values, uint32_t count)
{
    const ft_tt_noc_access *bulk;
    uint32_t tiles;
    double started;
    int rc;

    if (!device || !activation || !out_values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count || count > device->max_row_features)
        return -1;
    bulk = &device->runtime.bulk_dram;
    started = pn_tt_now_seconds();

    rc = ft_tt_tile_row0_read(bulk, (uint16_t)device->dram[activation->bank].x,
                              (uint16_t)device->dram[activation->bank].y, activation->address,
                              tiles, device->row_out);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_read += (size_t)tiles * FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_READ_BYTES;
    ft_float_from_bf16_array(device->row_out, out_values, count);
    device->readback_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_activation_download_chunks(pn_tt_device *device, const pn_tt_activation *activation,
                                     const unsigned char *need, float *out_values,
                                     uint32_t count)
{
    const ft_tt_noc_access *bulk;
    uint16_t chunk_buf[FT_TT_TILE_ROW0_READ_BYTES / sizeof(uint16_t)];
    uint32_t tiles, tile, chunk, chunks_read = 0u;
    double started;

    if (!device || !activation || !need || !out_values || count == 0u ||
        (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count || count > device->max_row_features)
        return -1;
    bulk = &device->runtime.bulk_dram;
    started = pn_tt_now_seconds();
    for (tile = 0u; tile < tiles; tile++) {
        for (chunk = 0u; chunk < FT_TT_TILE_ROW0_CHUNKS; chunk++) {
            const uint32_t at = tile * FT_TT_TILE_ROW0_CHUNKS + chunk;
            int rc;

            if (!need[at])
                continue;
            rc = bulk->read(bulk->user, (uint16_t)device->dram[activation->bank].x,
                            (uint16_t)device->dram[activation->bank].y,
                            (uint64_t)activation->address +
                                ft_tt_tile_row0_chunk_offset(tile, chunk),
                            chunk_buf, sizeof(chunk_buf));
            if (rc != FT_TT_OK)
                return -1;
            ft_float_from_bf16_array(chunk_buf, out_values + (size_t)at * FT_TT_TILE_ROW0_HALF_ELEMS,
                                     FT_TT_TILE_ROW0_HALF_ELEMS);
            chunks_read++;
        }
    }
    device->bytes_read += (size_t)chunks_read * FT_TT_TILE_ROW0_READ_BYTES;
    device->readback_seconds += pn_tt_now_seconds() - started;
    return 0;
}

/* ================================================================
 * The batched pair — rows 0..M-1 of a strip
 *
 * See pn_tt_device.h for why the 31 padding rows are the batch dimension and why moving
 * whole tiles is CHEAPER per token than the row-0 reader, not more expensive.
 * ================================================================ */

/** Grow the two staging buffers to hold `tiles` face-ordered tiles and `rows * count`
 *  bf16 elements. Kept on the device so a per-layer call is not a malloc. */
static int pn_tt_rows_stage(pn_tt_device *device, uint32_t tiles, uint32_t rows,
                            uint32_t count)
{
    size_t need_tiles = (size_t)tiles * PN_TT_TILE_BF16;
    size_t need_rows = (size_t)rows * count * sizeof(uint16_t);

    if (need_tiles > device->tile_stage_bytes) {
        unsigned char *grown = realloc(device->tile_stage, need_tiles);

        if (!grown)
            return -1;
        device->tile_stage = grown;
        device->tile_stage_bytes = need_tiles;
    }
    if (need_rows > device->rows_stage_bytes) {
        uint16_t *grown = realloc(device->rows_stage, need_rows);

        if (!grown)
            return -1;
        device->rows_stage = grown;
        device->rows_stage_bytes = need_rows;
    }
    return 0;
}

int pn_tt_activation_upload_rows(pn_tt_device *device, const pn_tt_activation *activation,
                                 const float *values, uint32_t count, uint32_t rows)
{
    uint32_t tiles, row;
    double started;

    if (!device || !activation || !values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    /* A tile has 32 rows and nothing here wraps to a second tile row: asking for more is a
     * refusal, never a silent truncation. */
    if (rows == 0u || rows > PN_TT_TILE_DIM)
        return -1;
    if (rows == 1u)
        return pn_tt_activation_upload(device, activation, values, count);
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count)
        return -1;
    if (pn_tt_rows_stage(device, tiles, rows, count) != 0)
        return -1;

    started = pn_tt_now_seconds();
    for (row = 0; row < rows; row++)
        ft_bf16_from_float_array(values + (size_t)row * count,
                                 device->rows_stage + (size_t)row * count, count);
    /* valid_rows = rows: the tilizer zeroes rows `rows`..31, which is exactly the strip
     * invariant a later Mt=1 matvec depends on. The batched write RESTORES it rather than
     * relying on the allocation-time staging surviving. */
    if (ft_tt_tilize_strip(device->rows_stage, sizeof(uint16_t), rows, count,
                           (size_t)count * sizeof(uint16_t),
                           device->tile_stage) != FT_TT_TILIZE_OK)
        return -1;
    for (uint32_t replica = 0u; replica < activation->replica_count; replica++)
        if (pn_tt_write_dram(device, activation->replica_bank[replica],
                             activation->replica_address[replica], device->tile_stage,
                             (size_t)tiles * PN_TT_TILE_BF16) != 0)
            return -1;
    device->upload_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_activation_download_rows(pn_tt_device *device, const pn_tt_activation *activation,
                                   float *out_values, uint32_t count, uint32_t rows)
{
    uint32_t tiles, tile, column, row;
    double started;

    if (!device || !activation || !out_values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (rows == 0u || rows > PN_TT_TILE_DIM)
        return -1;
    if (rows == 1u)
        return pn_tt_activation_download(device, activation, out_values, count);
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count)
        return -1;
    if (pn_tt_rows_stage(device, tiles, rows, count) != 0)
        return -1;

    started = pn_tt_now_seconds();
    /* S2e (`PN_TT_ROWS_FACES`, default 1): rows <= 16 live in faces 0-1 only — read those
     * face rows (two runs a tile, 64 B-rounded) instead of the whole tile. */
    if (rows <= 16u && tiles <= 4u && pn_tt_env_u32("PN_TT_ROWS_FACES", 1u) != 0u) {
        const size_t run = ((size_t)rows * 32u + 63u) & ~(size_t)63u;

        for (tile = 0; tile < tiles; tile++)
            if (pn_tt_read_dram(device, activation->bank,
                                activation->address + tile * PN_TT_TILE_BF16,
                                (unsigned char *)device->tile_stage + (size_t)tile * PN_TT_TILE_BF16,
                                run) != 0 ||
                pn_tt_read_dram(device, activation->bank,
                                activation->address + tile * PN_TT_TILE_BF16 + 512u,
                                (unsigned char *)device->tile_stage + (size_t)tile * PN_TT_TILE_BF16 +
                                    512u,
                                run) != 0)
                return -1;
    } else
    /* ONE read of the whole strip, then the de-tilize on the host. Reading it row by row
     * would be `rows` times the transactions for the same bytes, and this link is
     * transaction-count bound (Phase 7.8a). */
    if (pn_tt_read_dram(device, activation->bank, activation->address, device->tile_stage,
                        (size_t)tiles * PN_TT_TILE_BF16) != 0)
        return -1;
    for (tile = 0; tile < tiles; tile++) {
        const uint16_t *tile_base =
            (const uint16_t *)device->tile_stage + (size_t)tile * (PN_TT_TILE_BF16 / 2u);

        for (row = 0; row < rows; row++)
            for (column = 0; column < PN_TT_TILE_DIM; column++)
                out_values[(size_t)row * count + tile * PN_TT_TILE_DIM + column] =
                    ft_float_from_bf16(tile_base[ft_tt_tile_face_index(row, column)]);
    }
    device->readback_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_activation_zero(pn_tt_device *device, const pn_tt_activation *activation)
{
    return pn_tt_activation_zero_bands(device, activation, 0u);
}

int pn_tt_activation_zero_bands(pn_tt_device *device, const pn_tt_activation *activation,
                                uint32_t bands)
{
    return pn_tt_activation_fill_bands(device, activation, bands, 0u);
}

int pn_tt_activation_fill_bands(pn_tt_device *device, const pn_tt_activation *activation,
                                uint32_t bands, unsigned char value)
{
    if (!device || !activation)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. Zeroing
     * a strip a queued descriptor is still writing would be a race that reads as a wrong
     * answer one op later. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    /* THE WHOLE ALLOCATION, NOT THE BAND THE STRIP IS CURRENTLY POINTED AT. A seek slides
     * `address`, so zeroing from there would leave the bands before it carrying the last
     * pass's positions — which is exactly the residue the restage exists to remove. Backing
     * the address out by `row_offset` bands is the only way to say "all of it" from a strip
     * that may be mid-seek, and at `row_offset = 0` it is the pre-Mt call byte for byte. */
    for (uint32_t replica = 0u; replica < activation->replica_count; replica++) {
        const size_t band = (size_t)activation->tile_count * PN_TT_TILE_BF16;
        uint32_t rows = activation->tile_rows ? activation->tile_rows : 1u;

        /* PHASE G: a tall strip is zeroed only over the bands a pass actually wrote — the
         * whole 64-band allocation of every strip was ~2 GB over PCIe (10 s). */
        if (bands != 0u && bands < rows)
            rows = bands;

        if (pn_tt_fill_dram(device, activation->replica_bank[replica],
                            activation->replica_address[replica] -
                                (uint32_t)((size_t)activation->row_offset * band),
                            band * rows, value) != 0)
            return -1;
    }
    return 0;
}

void pn_tt_vec_probe_apply(uint32_t *words)
{
    static int probe = -1;

    if (probe < 0)
        probe = (int)(pn_tt_env_u32("PN_TT_VEC_PROBE", 0u) & 0xFFu);
    if (probe != 0 && words)
        words[FT_TT_OP_VECTOR_DEPTH] |= (uint32_t)probe << 24;
}

void pn_tt_device_prefer_prefetch_tiles(pn_tt_device *device, uint32_t tiles)
{
    if (device)
        device->prefetch_tiles_preferred = tiles;
}

void pn_tt_device_set_single_row(pn_tt_device *device, bool single_row)
{
    if (device)
        device->single_row = single_row;
}

void pn_tt_device_set_topk_row(pn_tt_device *device, uint32_t row)
{
    if (device && row < 32u)
        device->topk_row = row;
}

void pn_tt_device_set_verify_rows(pn_tt_device *device, uint32_t rows)
{
    if (device)
        device->verify_rows = rows;
}

void pn_tt_device_set_prefill_tile_rows(pn_tt_device *device, uint32_t rows)
{
    if (device)
        device->prefill_tile_rows = rows;
}

int pn_tt_device_prefer_strip_tile_rows(pn_tt_device *device, uint32_t rows)
{
    if (!device || device->strip_tile_rows != 0u)
        return -1;
    device->strip_tile_rows_preferred = rows;
    return 0;
}

uint32_t pn_tt_device_strip_tile_rows(pn_tt_device *device)
{
    if (!device)
        return 1u;
    if (device->strip_tile_rows == 0u) {
        uint32_t rows = pn_tt_env_u32(PN_TT_STRIP_TILE_ROWS_ENV,
                                      device->strip_tile_rows_preferred
                                          ? device->strip_tile_rows_preferred : 1u);

        /* A POWER OF TWO AND NO WIDER THAN THE PASS THE MIXERS CAN COVER. Every band costs a
         * separate pass of both chunk mixers (they refuse >32 rows by name in five places),
         * so an illegal value collapses to 1 rather than half-arming: a strip taller than the
         * pass would leave bands nobody wrote and a matmul would read them. */
        if (rows == 0u || (rows & (rows - 1u)) != 0u || rows > PN_TT_STRIP_MAX_TILE_ROWS) {
            fprintf(stderr, "pn_tt: %s=%u is not a power of two in 1..%u — using 1\n",
                    PN_TT_STRIP_TILE_ROWS_ENV, rows, PN_TT_STRIP_MAX_TILE_ROWS);
            rows = 1u;
        }
        if (rows > 1u)
            fprintf(stderr, "pn_tt: strip tile-rows %u — every activation strip is %u bands of "
                            "32 rows, so a batched prefill pass carries %u positions and the "
                            "matmul sweeps the weights once for all of them\n",
                    rows, rows, rows * PN_TT_TILE_DIM);
        device->strip_tile_rows = rows;
    }
    return device->strip_tile_rows;
}

int pn_tt_activation_seek(pn_tt_activation *activation, uint32_t row)
{
    const size_t band = activation ? (size_t)activation->tile_count * PN_TT_TILE_BF16 : 0u;
    int64_t delta;

    if (!activation || activation->tile_rows == 0u || row >= activation->tile_rows)
        return -1;
    if (row == activation->row_offset)
        return 0;
    delta = ((int64_t)row - (int64_t)activation->row_offset) * (int64_t)band;
    activation->address = (uint32_t)((int64_t)activation->address + delta);
    for (uint32_t replica = 0u; replica < activation->replica_count; replica++)
        activation->replica_address[replica] =
            (uint32_t)((int64_t)activation->replica_address[replica] + delta);
    activation->row_offset = row;
    return 0;
}

int pn_tt_activation_span(pn_tt_activation *activation, uint32_t rows)
{
    if (!activation || rows == 0u)
        return -1;
    /* THE SPAN IS BOUNDED BY WHAT IS LEFT ABOVE THE SEEK, not by the allocation: a strip
     * pointed at band 1 of two has one band above it, and an op told to span two would run
     * off the end of the buffer into whatever the allocator put there next. */
    if (activation->row_offset + rows > activation->tile_rows)
        return -1;
    activation->active_rows = rows;
    return 0;
}

int pn_tt_activation_download_tile_row(pn_tt_device *device, const pn_tt_activation *activation,
                                       uint32_t row, float *out_values, uint32_t count)
{
    uint32_t tiles, tile, column;
    uint16_t *staged;

    if (!device || !activation || !out_values || count == 0u || (count % PN_TT_TILE_DIM) != 0u)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (row >= PN_TT_TILE_DIM)
        return -1;
    tiles = count / PN_TT_TILE_DIM;
    if (tiles > activation->tile_count)
        return -1;
    staged = malloc((size_t)tiles * PN_TT_TILE_BF16);
    if (!staged)
        return -1;
    /* WHOLE tiles, not the row-0 reader: rows 1-31 are exactly what this exists to see, and
     * ft_tt_tile_row0_read() reads only the two 64-byte chunks row 0 lives in. */
    if (pn_tt_read_dram(device, activation->bank, activation->address, staged,
                        (size_t)tiles * PN_TT_TILE_BF16) != 0) {
        free(staged);
        return -1;
    }
    for (tile = 0; tile < tiles; tile++) {
        const uint16_t *tile_base = staged + (size_t)tile * (PN_TT_TILE_BF16 / 2u);

        for (column = 0; column < PN_TT_TILE_DIM; column++)
            out_values[tile * PN_TT_TILE_DIM + column] =
                ft_float_from_bf16(tile_base[ft_tt_tile_face_index(row, column)]);
    }
    free(staged);
    return 0;
}

ft_tt_op_operand pn_tt_operand(const pn_tt_device *device, const pn_tt_activation *activation)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[activation->bank].x;
    operand.y = (uint16_t)device->dram[activation->bank].y;
    operand.address = activation->address;
    operand.tile_bytes = PN_TT_TILE_BF16;
    return operand;
}

/* ================================================================
 * Arming the three residencies
 * ================================================================ */

/**
 * The ONE block depth a residency can run for every tile count it will see.
 *
 * Both conditions are the CB-cycle invariant and neither is optional: `depth` must divide
 * each run so a block is contiguous, and `2 * depth` — the page count the buffers are placed
 * at — must divide it too, so each descriptor returns the write pointer to the base. A
 * descriptor that leaves it elsewhere walks the next one up L1 with no error.
 *
 * @return the depth, or 0 when no depth serves every count
 */
uint32_t pn_tt_common_depth(const uint32_t *tile_counts, uint32_t count, uint32_t cap)
{
    uint32_t depth;

    if (!tile_counts || count == 0u || cap == 0u)
        return 0u;
    for (depth = cap; depth >= 1u; depth--) {
        uint32_t index;
        bool serves = true;

        for (index = 0; index < count && serves; index++) {
            if (tile_counts[index] == 0u || (tile_counts[index] % (2u * depth)) != 0u)
                serves = false;
            else if (ft_tt_op_block_depth(tile_counts[index], depth) != depth)
                serves = false;
        }
        if (serves)
            return depth;
    }
    return 0u;
}

bool pn_tt_prepare_family(pn_tt_device *device, pn_tt_residency *residency, const char *program,
                          const char *name, const char *core_env, uint32_t core_default,
                          uint32_t processor_mask)
{
    return pn_tt_prepare_family_on_core(device, residency, program, name, core_env,
                                        pn_tt_env_u32(core_env, core_default), processor_mask);
}

/**
 * Who, if anyone, already holds `core` — by NAME, so the refusal can say it.
 *
 * `except` is the residency being prepared, which is already in the table it walks and would
 * otherwise report itself on a re-arm.
 *
 * @return the holder's name, or NULL when the core is free
 */
static const char *pn_tt_core_holder(const pn_tt_device *device, uint32_t core,
                                     const pn_tt_residency *except)
{
    /* EIGHT named residencies are pushed below, not six. At the shipped four-way head split
     * with both head-norm widths live the table holds 12 of the 15 the old sizing gave, so it
     * never overflowed — but a split of 8 with two widths needs 16, and the failure would be a
     * stack write past the array on exactly the hybrid configuration this check exists for.
     * `pn_tt_device_residency_stat()` sizes the same list at 7 and is the one that was right. */
    const pn_tt_residency *others[10u + (PN_TT_TOPK_MAX_MEMBERS - 1u) +
                                  PN_TT_RENDER_DIRECTIONS * PN_TT_RENDER_SPLIT_MAX_MEMBERS +
                                  (PN_TT_GDN_MAX_MEMBERS - 1u) +
                                  (PN_TT_GDNC_MAX_MEMBERS - 1u) +
                                  (PN_TT_ELTWISE_MAX_MEMBERS - 1u) +
                                  (PN_TT_ATTN_MAX_MEMBERS - 1u) +
                                  PN_TT_HEAD_NORM_MAX_WIDTHS * PN_TT_HEAD_NORM_MAX_MEMBERS];
    uint32_t count = 0u;
    uint32_t index;

    /* Member 0 of the matmul group is the DEVICE's own core, and the group holds the rest.
     * The group is checked first because it is the one set that MOVES with a knob. */
    if (device->core_index == core)
        return "matmul member 0";
    for (index = 0; index < device->group.member_count; index++) {
        if (device->group.member[index].core_index == core)
            return "a matmul group member";
    }

    others[count++] = &device->reduce;
    others[count++] = &device->reduce_member;
    others[count++] = &device->eltwise;
    others[count++] = &device->sample;
    for (index = 0; index + 1u < device->sample_members; index++)
        others[count++] = &device->sample_member[index];
    others[count++] = &device->rope;
    others[count++] = &device->memory;
    others[count++] = &device->attention;
    others[count++] = &device->gdn;
    /* EIGHT. The chunkwise family is a second Tensix running a second program against the
     * SAME carried state, so a collision between it and anything else is the 1560-mutual-
     * re-arm failure with a correct answer at the far end of it. */
    others[count++] = &device->gdnc;
    /* THE STRIP BRIDGE AND THE MERGED GATE WERE NOT IN THIS TABLE AT ALL, and `pn_tt_render.h`
     * says in terms that they are refused by name against every live residency — which was
     * not true of them. They sit at 85/86/87, inside the range sixteen matmul members reach,
     * and any of the three core knobs can put one of them on top of another. The group half of
     * the check covered them (it runs above, before this table), so the gap was render-vs-
     * render, render-vs-gate and either-vs-a-family; a collision there is not a failure but a
     * pair of residencies retiring each other with healthy "residency up" lines. */
    for (index = 0; index < (uint32_t)PN_TT_RENDER_DIRECTIONS; index++) {
        uint32_t member;

        others[count++] = &device->render[index].residency;
        /* AND THE STRIP BRIDGE'S OWN SPLIT MEMBERS. They default to 122 and upward, above
         * every core any other family can reach at any legal knob — but that is an argument
         * about defaults, and this table is what makes it a refusal. */
        for (member = 1u; member < device->render[index].members; member++)
            others[count++] = &device->render[index].member[member - 1u];
    }
    others[count++] = &device->gate.residency;
    for (index = 0; index + 1u < device->gdn_members; index++)
        others[count++] = &device->gdn_member[index];
    /* AND THE CHUNKWISE SPLIT'S OWN MEMBERS. They default to 107 and upward, above every core
     * any other family reaches at its shipped default — but that is an argument about defaults,
     * and this table is what makes it a refusal. Sixteen of them at the maximum, which is why
     * the array above grew by `PN_TT_GDNC_MAX_MEMBERS - 1`. */
    for (index = 0; index + 1u < device->gdnc_members; index++)
        others[count++] = &device->gdnc_member[index];
    /* THE ELEMENTWISE SPLIT'S EXTRA MEMBERS. They default to 106 and upward, above every core
     * any other family can reach at any legal setting of any knob (pn_tt_eltwise_group.h) —
     * but that is an argument about defaults, and this table is what makes it a refusal. */
    for (index = 0; index + 1u < device->eltwise_members; index++)
        others[count++] = &device->eltwise_member[index];
    for (index = 0; index < device->head_norm_count; index++) {
        uint32_t member;

        others[count++] = &device->head_norm[index].residency;
        for (member = 1u; member < device->head_norm[index].members; member++)
            others[count++] = &device->head_norm[index].member[member - 1u];
    }
    /* AND THE ATTENTION CHUNK MIXER'S HEAD-SPLIT MEMBERS. */
    for (index = 0; index + 1u < device->attention_members; index++)
        others[count++] = &device->attention_member[index];
    for (index = 0; index < count; index++) {
        if (others[index] == except || !others[index]->core_booted)
            continue;
        if (others[index]->core_index == core)
            return others[index]->name;
    }
    return NULL;
}

bool pn_tt_prepare_family_on_core(pn_tt_device *device, pn_tt_residency *residency,
                                  const char *program, const char *name, const char *core_env,
                                  uint32_t core, uint32_t processor_mask)
{
    const char *root = pn_tt_program_override_root();
    const char *holder;

    if (pn_tt_residency_load(residency, &device->runtime, program, root, name,
                             processor_mask) != 0)
        return false;
    /* A RESIDENCY AND A MATMUL MEMBER ON ONE TENSIX RETIRE EACH OTHER FOREVER, and the group's
     * members are not a fixed set: `PN_TT_MATMUL_CORES=4`, the shipped knee, puts member 1 on
     * core 75. So a family's default cannot be checked against the other FAMILIES alone.
     * MEASURED ON SILICON 2026-08-18: the GDN default of 75 collided with member 1 and the run
     * logged 1560 mutual re-arms over 65 forwards, each one a correct-looking "residency up"
     * line. Refusing NAMES it; the alternative is a run that works and is inexplicably slow.
     *
     * IT CHECKS EVERY LIVE RESIDENCY, NOT ONLY THE GROUP. It used to check the group alone,
     * which was enough while the families' defaults were a hand-kept list of distinct
     * numbers — but that list is now eleven cores over five knobs, and one of them
     * (`PN_TT_GDN_HEAD_CORES`) grows a run of them. Booting a second program's firmware onto
     * a core that already holds a resident triad is a WEDGED core, not a configuration. */
    holder = pn_tt_core_holder(device, core, residency);
    if (holder) {
        fprintf(stderr, "pn_tt: the %s residency was asked for core %u, which %s already "
                        "holds — the two would retire each other every token. Set %s to a "
                        "free core.\n", name, core, holder, core_env);
        pn_tt_residency_release(residency);
        return false;
    }
    /* AND THE OTHER HALF OF THE SAME QUESTION, WHICH ONLY THE CARD CAN ANSWER. The check
     * above walks THIS PROCESS's residency table, and the failure that costs a card reset
     * crosses a process boundary: a spinning compute thread left by an earlier `serve`
     * outlives it, and this process's table is empty. `pn_tt_core_boot_guard.h` has the
     * whole argument; `pn_tt_residency_bind_core()` below is the boot it sits in front of. */
    if (!pn_tt_core_boot_is_permitted(&device->runtime, core, name, core_env)) {
        pn_tt_residency_release(residency);
        return false;
    }
    if (pn_tt_residency_bind_core(residency, core) != 0) {
        pn_tt_residency_release(residency);
        return false;
    }
    return true;
}

bool pn_tt_program_is_present(const char *program)
{
    ft_tt_runtime_images images;

    if (ft_tt_program_open_firmware(program, pn_tt_program_override_root(), &images) != FT_TT_OK)
        return false;
    ft_tt_runtime_images_release(&images);
    return true;
}

/**
 * Refuse before arming anything if a family this run ASKED FOR has no program to arm.
 *
 * The same defect the resident matmul had, one file over: a family whose program is absent
 * silently leaves its op on the host, prints `norm 0` in a line nobody reads, and
 * produces a token that is correct, slower, and numerically different — with nothing saying
 * the run was not the run that was configured. A missing program is a CONFIGURATION error
 * and the answer to one is a refusal.
 *
 * Deliberately narrow: it checks only that the program RESOLVES. Arming can still fail
 * for reasons that belong to the card and to this moment (a core in use, buffers that do not
 * fit), and those stay non-fatal because the run can honestly continue without them.
 *
 * @param device  open device, for its runtime
 * @return 0 when every requested family has a program, -1 with the list on stderr otherwise
 */
static int pn_tt_refuse_missing_oncard_programs(pn_tt_device *device)
{
    /* ATTENTION is NULL here because it is THREE programs (rope, memory, attention) and they
     * are checked where they are armed, in pn_tt_attention.c. A NULL is skipped rather than
     * defaulted to a neighbour's name — a family checked against the wrong directory would
     * report the wrong thing green. */
    static const char *const k_program[PN_TT_ONCARD_COUNT] = {
        PN_TT_REDUCE_PROGRAM, PN_TT_ELTWISE_PROGRAM, PN_TT_ELTWISE_PROGRAM,
        PN_TT_SAMPLE_PROGRAM, NULL, PN_TT_GDN_PROGRAM
    };
    const char *root = pn_tt_program_override_root();
    unsigned family;
    int missing = 0;

    for (family = 0; family < PN_TT_ONCARD_COUNT; family++) {
        ft_tt_runtime_images images;

        if (!device->oncard_requested[family] || !k_program[family])
            continue;
        if (ft_tt_program_open_firmware(k_program[family], root, &images) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: REFUSED — %s was asked for on the card and there is no "
                            "such program in %s\n", k_program[family],
                    pn_tt_program_where(root));
            missing++;
            continue;
        }
        ft_tt_runtime_images_release(&images);
    }
    if (missing != 0)
        fprintf(stderr, "pn_tt: %d on-card program%s missing. Build them into the blob, or "
                        "name only the families you have with %s.\n",
                missing, missing == 1 ? " is" : "s are", PN_TT_ONCARD_ENV);
    return missing == 0 ? 0 : -1;
}

/* PM 8: which family a core holds, by NoC coordinate, for reports keyed by queue (the
 * puller's AWAIT census). The last family started on a core wins. */
#define PN_TT_CORE_LABELS 256u
static struct { uint16_t x, y; char name[32]; } g_core_labels[PN_TT_CORE_LABELS];
static uint32_t g_core_label_count;

static void pn_tt_core_label_note(uint16_t x, uint16_t y, const char *name)
{
    uint32_t i;

    for (i = 0; i < g_core_label_count; i++)
        if (g_core_labels[i].x == x && g_core_labels[i].y == y)
            break;
    if (i == PN_TT_CORE_LABELS)
        return;
    g_core_labels[i].x = x;
    g_core_labels[i].y = y;
    snprintf(g_core_labels[i].name, sizeof(g_core_labels[i].name), "%s", name);
    if (i == g_core_label_count)
        g_core_label_count++;
}

const char *pn_tt_core_label(uint16_t x, uint16_t y)
{
    uint32_t i;

    for (i = 0; i < g_core_label_count; i++)
        if (g_core_labels[i].x == x && g_core_labels[i].y == y)
            return g_core_labels[i].name;
    return "?";
}

bool pn_tt_start_family(pn_tt_residency *residency)
{
    if (pn_tt_residency_start(residency) != 0) {
        pn_tt_residency_release(residency);
        return false;
    }
    pn_tt_core_label_note((uint16_t)residency->core_x, (uint16_t)residency->core_y,
                          residency->name);
    fprintf(stderr, "pn_tt: %s residency up on core %u\n", residency->name,
            residency->core_index);
    return true;
}

/** The reduce residency, sized for RMS norm at ONE width. */
static bool pn_tt_arm_reduce(pn_tt_device *device, uint32_t norm_tile_cols)
{
    static const ft_tt_op_reduce_function members[] = { FT_TT_OP_REDUCE_FN_RMS_NORM };
    ft_tt_op_reduce_residency residency;
    ft_tt_launch_cb_entry placed[FT_TT_OP_REDUCE_CB_COUNT];

    if (ft_tt_op_reduce_residency_for(members, 1u, norm_tile_cols, PN_TT_TILE_BF16,
                                      &residency) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: no reduce residency serves Wt=%u — norms stay on the host\n",
                norm_tile_cols);
        return false;
    }
    if (!pn_tt_prepare_family(device, &device->reduce, PN_TT_REDUCE_PROGRAM, "reduce",
                              PN_TT_CORE_REDUCE_ENV, PN_TT_CORE_REDUCE_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_reduce_place_cbs(&residency, placed, device->reduce.cb_buffer,
                                  sizeof(device->reduce.cb_buffer),
                                  &device->reduce.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the reduce circular buffers do not fit at Wt=%u\n",
                norm_tile_cols);
        pn_tt_residency_release(&device->reduce);
        return false;
    }
    device->reduce_tile_cols = norm_tile_cols;
    if (!pn_tt_start_family(&device->reduce))
        return false;
    {
        const uint32_t fill = pn_tt_env_u32("PN_TT_IDLE_FILL", 0u);
        uint32_t f;

        for (f = device->idle_fill_count; f < fill && f < 64u; f++) {
            pn_tt_residency *r = &device->idle_fill[f];

            if (!pn_tt_prepare_family_on_core(device, r, PN_TT_REDUCE_PROGRAM, "idle-fill",
                                              "PN_TT_IDLE_FILL_CORE", f,
                                              PN_TT_ONCARD_TRIAD_MASK) ||
                ft_tt_op_reduce_place_cbs(&residency, placed, r->cb_buffer,
                                          sizeof(r->cb_buffer), &r->cb_mask) != FT_TT_OK ||
                !pn_tt_start_family(r)) {
                pn_tt_residency_release(r);
                break;
            }
            device->idle_fill_count = f + 1u;
        }
        if (fill)
            fprintf(stderr, "pn_tt: IDLE FILL — %u idle reduce residencies on cores 0..%u\n",
                    device->idle_fill_count, device->idle_fill_count - 1u);
    }
    /* L4: a second member over the same placement. Its failure is not fatal — the norm then
     * runs whole on member 0. */
    if (device->reduce_members != 1u && pn_tt_env_u32("PN_TT_REDUCE_CORES", 2u) >= 2u /* L4 ON since 2026-10-05: decode 16.10 -> 16.28, prefill 335.3 -> 339.5, IDENTICAL */ &&
        (norm_tile_cols % 2u) == 0u && !device->reduce_member.running) {
        device->reduce_members = 1u;
        if (pn_tt_prepare_family(device, &device->reduce_member, PN_TT_REDUCE_PROGRAM,
                                 "reduce-member", "PN_TT_CORE_REDUCE_MEMBER", 88u,
                                 PN_TT_ONCARD_TRIAD_MASK)) {
            if (ft_tt_op_reduce_place_cbs(&residency, placed, device->reduce_member.cb_buffer,
                                          sizeof(device->reduce_member.cb_buffer),
                                          &device->reduce_member.cb_mask) == FT_TT_OK &&
                pn_tt_start_family(&device->reduce_member))
                device->reduce_members = 2u;
            else
                pn_tt_residency_release(&device->reduce_member);
        }
        fprintf(stderr, "pn_tt: reduce COLUMN SPLIT — %u member(s)\n", device->reduce_members);
    }
    return true;
}

/** L4: re-arm the reduce member alone (same placement as member 0). */
static bool pn_tt_rearm_reduce_member(pn_tt_device *device)
{
    static const ft_tt_op_reduce_function members[] = { FT_TT_OP_REDUCE_FN_RMS_NORM };
    ft_tt_op_reduce_residency residency;
    ft_tt_launch_cb_entry placed[FT_TT_OP_REDUCE_CB_COUNT];

    pn_tt_residency_stop(&device->reduce_member);
    pn_tt_residency_release(&device->reduce_member);
    device->oncard_restarts++;
    return ft_tt_op_reduce_residency_for(members, 1u, device->reduce_tile_cols,
                                         PN_TT_TILE_BF16, &residency) == FT_TT_OK &&
           pn_tt_prepare_family(device, &device->reduce_member, PN_TT_REDUCE_PROGRAM,
                                "reduce-member", "PN_TT_CORE_REDUCE_MEMBER", 88u,
                                PN_TT_ONCARD_TRIAD_MASK) &&
           ft_tt_op_reduce_place_cbs(&residency, placed, device->reduce_member.cb_buffer,
                                     sizeof(device->reduce_member.cb_buffer),
                                     &device->reduce_member.cb_mask) == FT_TT_OK &&
           pn_tt_start_family(&device->reduce_member);
}

/**
 * The elementwise residency SET, sized for the widest SLICE and ONE depth for all.
 *
 * The split is planned BEFORE the placement, not after, because the placement is sized from
 * the width a descriptor will actually carry — see `pn_tt_eltwise_group_plan()`. At the
 * shipped `PN_TT_ELTWISE_CORES=1` the plan returns one member and the widest width, which is
 * this function's pre-split behaviour word for word.
 */
static bool pn_tt_arm_eltwise(pn_tt_device *device, const uint32_t *tile_counts, uint32_t count)
{
    ft_tt_op_eltwise_request request;
    ft_tt_launch_cb_entry placed[FT_TT_OP_ELTWISE_CB_COUNT];
    uint32_t widest, depth, members;

    pn_tt_eltwise_group_plan(device, tile_counts, count, &members, &widest, &depth);
    if (depth == 0u) {
        fprintf(stderr, "pn_tt: no single block depth serves every elementwise shape — "
                        "SwiGLU and the residual adds stay on the host\n");
        return false;
    }

    if (!pn_tt_prepare_family(device, &device->eltwise, PN_TT_ELTWISE_PROGRAM, "eltwise",
                              PN_TT_CORE_ELTWISE_ENV, PN_TT_CORE_ELTWISE_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_VECTOR_FN_SILU_MUL; /* the widest of the six */
    request.tile_count = widest;
    request.prefetch_tiles = depth;
    request.input_a.tile_bytes = PN_TT_TILE_BF16;
    request.input_b.tile_bytes = PN_TT_TILE_BF16;
    request.output.tile_bytes = PN_TT_TILE_BF16;
    if (ft_tt_op_eltwise_place_cbs(&request, placed, device->eltwise.cb_buffer,
                                   sizeof(device->eltwise.cb_buffer),
                                   &device->eltwise.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the elementwise circular buffers do not fit at depth %u\n",
                depth);
        pn_tt_residency_release(&device->eltwise);
        return false;
    }
    device->eltwise_depth = depth;
    if (!pn_tt_start_family(&device->eltwise))
        return false;
    pn_tt_eltwise_group_arm(device, members, &request);
    return true;
}

/** L3: arm (or re-arm) top-k split member `m` (1-based) on `core` for a `slice`-tile row at
 *  `depth`, with the same greedy state block member 0 carries. */
static bool pn_tt_arm_topk_member(pn_tt_device *device, uint32_t m, uint32_t core,
                                  uint32_t slice, uint32_t depth)
{
    pn_tt_residency *r = &device->sample_member[m - 1u];
    ft_tt_op_topk_request request;
    ft_tt_launch_cb_entry placed[FT_TT_OP_SAMPLE_CB_COUNT];
    ft_tt_op_sample_params params;
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];

    memset(&request, 0, sizeof(request));
    request.tile_count = slice;
    request.row = 0u;
    request.select_count = PN_TT_TOPK_SELECT;
    request.prefetch_tiles = depth;
    request.logits.tile_bytes = PN_TT_TILE_BF16;
    memset(&params, 0, sizeof(params));
    params.temperature = 0.0f;
    params.top_p = 1.0f;
    params.repeat_penalty = 1.0f;
    params.rng_seed = PN_TT_TOPK_RNG_SEED;
    if (ft_tt_op_sample_state_build(&params, state, FT_TT_OP_SAMPLE_STATE_WORDS) != FT_TT_OK ||
        !pn_tt_prepare_family_on_core(device, r, PN_TT_SAMPLE_PROGRAM, "topk-member",
                                      "PN_TT_CORE_TOPK_MEMBER", core, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_sample_place_cbs(&request, placed, r->cb_buffer, sizeof(r->cb_buffer),
                                  &r->cb_mask) != FT_TT_OK ||
        !pn_tt_start_family(r)) {
        pn_tt_residency_release(r);
        return false;
    }
    device->sample_member_result[m - 1u] = placed[1].address;
    device->sample_member_state[m - 1u] = placed[2].address;
    if (device->runtime.noc.write(device->runtime.noc.user, r->core_x, r->core_y,
                                  placed[2].address, state, sizeof(state)) != FT_TT_OK) {
        pn_tt_residency_release(r);
        return false;
    }
    return true;
}

/** The sampling residency, plus the greedy state block it reads every descriptor. */
static bool pn_tt_arm_sample(pn_tt_device *device, uint32_t vocab_tiles)
{
    ft_tt_op_topk_request request;
    ft_tt_launch_cb_entry placed[FT_TT_OP_SAMPLE_CB_COUNT];
    ft_tt_op_sample_params params;
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];
    uint32_t depth;
    int rc;

    {
        /* L3: a split runs every member at ONE depth, so with `PN_TT_TOPK_CORES=N` the depth
         * is the slice's (member 0 included); the whole row at that depth still divides. */
        const uint32_t split = pn_tt_env_u32("PN_TT_TOPK_CORES", 5u) /* L3 ON 2026-10-05: decode 16.43 -> 16.78, IDENTICAL */;
        uint32_t basis = vocab_tiles;

        if (split >= 2u && split <= PN_TT_TOPK_MAX_MEMBERS && (vocab_tiles % split) == 0u)
            basis = vocab_tiles / split;
        depth = pn_tt_common_depth(&basis, 1u, PN_TT_ONCARD_MAX_DEPTH);
    }
    if (depth == 0u) {
        fprintf(stderr, "pn_tt: no block depth serves a %u-tile logit row — top-N stays on "
                        "the host\n", vocab_tiles);
        return false;
    }
    if (!pn_tt_prepare_family(device, &device->sample, PN_TT_SAMPLE_PROGRAM, "topk",
                              PN_TT_CORE_SAMPLE_ENV, PN_TT_CORE_SAMPLE_DEFAULT,
                              PN_TT_ONCARD_TRIAD_MASK))
        return false;
    memset(&request, 0, sizeof(request));
    request.tile_count = vocab_tiles;
    request.row = 0u;
    request.select_count = PN_TT_TOPK_SELECT;
    request.prefetch_tiles = depth;
    request.logits.tile_bytes = PN_TT_TILE_BF16;
    if (ft_tt_op_sample_place_cbs(&request, placed, device->sample.cb_buffer,
                                  sizeof(device->sample.cb_buffer),
                                  &device->sample.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the sampling circular buffer does not fit at depth %u\n", depth);
        pn_tt_residency_release(&device->sample);
        return false;
    }
    device->sample_tile_count = vocab_tiles;
    device->sample_depth = depth;
    device->sample_result_address = placed[1].address;
    device->sample_state_address = placed[2].address;
    if (!pn_tt_start_family(&device->sample))
        return false;

    memset(&params, 0, sizeof(params));
    params.temperature = 0.0f; /* greedy: argmax on RAW values, no penalty */
    params.top_p = 1.0f;
    params.repeat_penalty = 1.0f;
    params.rng_seed = PN_TT_TOPK_RNG_SEED;
    rc = ft_tt_op_sample_state_build(&params, state, FT_TT_OP_SAMPLE_STATE_WORDS);
    if (rc == FT_TT_OK)
        rc = device->runtime.noc.write(device->runtime.noc.user, device->sample.core_x,
                                       device->sample.core_y, device->sample_state_address,
                                       state, sizeof(state));
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the sampler state block did not upload: %s\n",
                ft_tt_error_string(rc));
        pn_tt_residency_release(&device->sample);
        return false;
    }
    /* L3: the split members, each placed for its slice at member 0's depth. Not fatal. */
    {
        static const uint32_t member_core[PN_TT_TOPK_MAX_MEMBERS - 1u] = { 92u, 93u, 94u, 113u,
                                                                         121u, 126u, 127u };
        const uint32_t asked = pn_tt_env_u32("PN_TT_TOPK_CORES", 5u) /* L3 ON 2026-10-05: decode 16.43 -> 16.78, IDENTICAL */;
        uint32_t m;

        if (device->sample_members == 0u)
            device->sample_members = 1u;
        if (asked >= 2u && asked <= PN_TT_TOPK_MAX_MEMBERS && device->sample_members == 1u &&
            (vocab_tiles % asked) == 0u) {
            uint32_t slice = vocab_tiles / asked;
            uint32_t up = 1u;

            if (pn_tt_common_depth(&slice, 1u, PN_TT_ONCARD_MAX_DEPTH) < depth ||
                (slice % (2u * depth)) != 0u) {
                fprintf(stderr, "pn_tt: top-k split over %u refused — a %u-tile slice does not "
                                "take depth %u\n", asked, slice, depth);
            } else {
                for (m = 1u; m < asked; m++) {
                    if (!pn_tt_arm_topk_member(device, m, member_core[m - 1u], slice, depth))
                        break;
                    up++;
                }
                if (up == asked) {
                    device->sample_members = asked;
                } else {
                    for (m = 1u; m < up; m++)
                        pn_tt_residency_release(&device->sample_member[m - 1u]);
                }
                fprintf(stderr, "pn_tt: top-k SPLIT — %u member(s), %u tiles each\n",
                        device->sample_members, vocab_tiles / device->sample_members);
            }
        }
    }
    return true;
}

int pn_tt_device_arm_oncard(pn_tt_device *device, uint32_t norm_tile_cols,
                            const uint32_t *eltwise_tile_counts, uint32_t eltwise_count,
                            uint32_t vocab_tiles)
{
    bool eltwise_up;

    if (!device || norm_tile_cols == 0u || !eltwise_tile_counts || eltwise_count == 0u)
        return -1;
    if (eltwise_count > PN_TT_ONCARD_MAX_ELTWISE_SHAPES) {
        fprintf(stderr, "pn_tt: %u elementwise shapes is more than the %u this device "
                        "records\n", eltwise_count, PN_TT_ONCARD_MAX_ELTWISE_SHAPES);
        return -1;
    }
    pn_tt_read_oncard_selection(device);
    if (pn_tt_refuse_missing_oncard_programs(device) != 0)
        return -1;
    /* KEPT, not consumed: pn_tt_oncard_rearm() arms a fresh residency from these. */
    memcpy(device->oncard_eltwise_tiles, eltwise_tile_counts,
           (size_t)eltwise_count * sizeof(*eltwise_tile_counts));
    device->oncard_eltwise_count = eltwise_count;

    if (device->oncard_requested[PN_TT_ONCARD_NORM])
        device->oncard[PN_TT_ONCARD_NORM] = pn_tt_arm_reduce(device, norm_tile_cols);
    /* ONE residency serves both elementwise members, so they arm together and are still
     * selected apart: a run may move the SwiGLU and leave the residual adds on the host,
     * which is what makes each a separately provable step. */
    if (device->oncard_requested[PN_TT_ONCARD_SWIGLU] ||
        device->oncard_requested[PN_TT_ONCARD_RESIDUAL]) {
        eltwise_up = pn_tt_arm_eltwise(device, eltwise_tile_counts, eltwise_count);
        device->oncard[PN_TT_ONCARD_SWIGLU] =
            eltwise_up && device->oncard_requested[PN_TT_ONCARD_SWIGLU];
        device->oncard[PN_TT_ONCARD_RESIDUAL] =
            eltwise_up && device->oncard_requested[PN_TT_ONCARD_RESIDUAL];
    }
    if (device->oncard_requested[PN_TT_ONCARD_TOPK] && vocab_tiles != 0u)
        device->oncard[PN_TT_ONCARD_TOPK] = pn_tt_arm_sample(device, vocab_tiles);

    fprintf(stderr, "pn_tt: on-card ops — norm %d, swiglu %d, residual %d, topk %d, gdn %d\n",
            (int)device->oncard[PN_TT_ONCARD_NORM], (int)device->oncard[PN_TT_ONCARD_SWIGLU],
            (int)device->oncard[PN_TT_ONCARD_RESIDUAL], (int)device->oncard[PN_TT_ONCARD_TOPK],
            (int)device->oncard[PN_TT_ONCARD_GDN]);
    return 0;
}

/**
 * Retire a residency that refused a descriptor, and arm a FRESH one on the same core from
 * the shapes it was armed with.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS, AND WHY ONLY THE MATMUL HAD IT
 * ---------------------------------------------------------------------------
 * A residency is BOUNDED on purpose (ft_tt_queue.h rule 3): `idle_spin_limit` retires a
 * kernel whose host has stopped feeding it, because the alternative is a core held forever
 * by a process that died. The matmul residency has re-armed on refusal since the resident
 * queue landed (pn_tt_device.c); these three never did, because every caller so far was a
 * one-shot harness that ran its tokens back to back and then exited.
 *
 * A `pantheon serve` is the first caller that IDLES — it arms at model load and then waits
 * for a request. MEASURED on this card: the first request after a load succeeds, and one
 * fired ~18 s later fails with `reduce descriptor 981 did not complete: error`, 981 being
 * exactly one past the 980 reduces the first request issued. So the refusal is not a wrong
 * descriptor, it is a correct descriptor arriving at a kernel that has already retired —
 * and the recovery is the same ASK-and-re-arm the matmul already proved on silicon.
 *
 * The elementwise and reduce parameter words address DRAM only, so the caller may retry
 * with the SAME words. The sampling words carry two L1 addresses on the sampling core and
 * must be re-encoded after a re-arm; pn_tt_topk() does that rather than assume the fresh
 * placement lands where the old one did.
 *
 * @return true when a fresh residency is live and the caller may retry
 */
static bool pn_tt_oncard_rearm(pn_tt_device *device, pn_tt_residency *residency)
{
    fprintf(stderr, "pn_tt: the %s residency refused a descriptor — retiring it by ASKING "
                    "and arming a fresh one on core %u\n", residency->name,
            residency->core_index);
    pn_tt_residency_stop(residency);
    pn_tt_residency_release(residency);
    device->oncard_restarts++;

    if (residency == &device->reduce)
        return pn_tt_arm_reduce(device, device->reduce_tile_cols);
    if (residency == &device->eltwise)
        return pn_tt_arm_eltwise(device, device->oncard_eltwise_tiles,
                                 device->oncard_eltwise_count);
    if (residency == &device->sample)
        return pn_tt_arm_sample(device, device->sample_tile_count);
    return false;
}

/* ================================================================
 * The ops
 * ================================================================ */

int pn_tt_reduce_constants_upload(pn_tt_device *device, ft_tt_op_reduce_function function,
                                  const float *gamma, uint32_t width, float eps,
                                  uint32_t tile_bytes, pn_tt_norm_constants *out_constants)
{
    const bool wants_gamma = ft_tt_op_reduce_uses_gamma(function);
    const ft_tt_noc_access *bulk;
    ft_tt_buffer buffer;
    unsigned char *constants;
    size_t bytes;
    size_t offset = 0;
    int rc = FT_TT_OK;

    if (!device || !out_constants || width == 0u || (width % PN_TT_TILE_DIM) != 0u)
        return -1;
    if (wants_gamma && !gamma)
        return -1;
    if (pn_tt_red("norm_eps"))
        eps *= 1000000.0f; /* RED PROOF: a wrong norm, everywhere, smoothly */
    bytes = ft_tt_op_reduce_constants_bytes(function, width / PN_TT_TILE_DIM, tile_bytes);
    if (bytes == 0)
        return -1;
    constants = malloc(bytes);
    if (!constants)
        return -1;
    /* The scaler tile, the eps tile, then gamma expanded to FULL 32x32 tiles — a norm weight
     * is 1-D and the artifact stores 1-D tensors untiled, and expanding here turns the
     * multiply into a plain mul_tiles instead of a broadcast op. */
    if (ft_tt_op_reduce_build_constants(function, width, eps, wants_gamma ? gamma : NULL, NULL,
                                        tile_bytes, constants, bytes) != FT_TT_OK) {
        free(constants);
        return -1;
    }
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM,
                           device->strip_bank_cursor++ % device->dram_bank_count, bytes,
                           &buffer) != FT_TT_OK) {
        free(constants);
        fprintf(stderr, "pn_tt: DRAM full uploading norm constants\n");
        return -1;
    }
    bulk = &device->runtime.bulk_dram;
    while (offset < bytes && rc == FT_TT_OK) {
        size_t chunk = bytes - offset;

        if (chunk > PN_TT_ZERO_CHUNK_BYTES)
            chunk = PN_TT_ZERO_CHUNK_BYTES;
        rc = bulk->write(bulk->user, (uint16_t)device->dram[buffer.logical_bank].x,
                         (uint16_t)device->dram[buffer.logical_bank].y,
                         buffer.address + offset, constants + offset, chunk);
        offset += chunk;
    }
    free(constants);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += bytes;
    out_constants->bank = buffer.logical_bank;
    out_constants->address = (uint32_t)buffer.address;
    out_constants->tile_cols = width / PN_TT_TILE_DIM;
    return 0;
}

int pn_tt_norm_constants_upload(pn_tt_device *device, const float *gamma, uint32_t width,
                                float eps, pn_tt_norm_constants *out_constants)
{
    return pn_tt_reduce_constants_upload(device, FT_TT_OP_REDUCE_FN_RMS_NORM, gamma, width, eps,
                                         PN_TT_TILE_BF16, out_constants);
}

int pn_tt_rms_norm(pn_tt_device *device, const pn_tt_norm_constants *constants,
                   const pn_tt_activation *input, const pn_tt_activation *output)
{
    ft_tt_op_reduce_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    double started;

    if (!device || !constants || !input || !output || !device->reduce.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (constants->tile_cols != device->reduce_tile_cols ||
        input->tile_count < constants->tile_cols || output->tile_count < constants->tile_cols) {
        fprintf(stderr, "pn_tt: rms_norm Wt=%u is not the Wt=%u this residency was placed "
                        "for\n", constants->tile_cols, device->reduce_tile_cols);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_REDUCE_FN_RMS_NORM;
    /* ONE 32-ROW BLOCK PER BAND. A decode token is one block with 31 rows of zero padding and
     * that is the shipped `1`; an Mt-band prefill strip is Mt of them, and the reduce family
     * has carried the axis since it shipped — the resident kernels derive `row_blocks =
     * tile_count / tile_cols` (resident_reduce_reader.c:192, _compute.c:459) and the encoder
     * writes `TILE_COUNT = row_blocks * tile_cols` (ft_tt_op_vector.c:394). So the second band
     * is a wider descriptor and NOT a second one, and each band is reduced on its own row —
     * which is the whole correctness question, because a reduce that spanned bands would
     * normalise 64 positions against one denominator.
     *
     * THE SPAN IS THE INPUT'S AND IT IS CLAMPED BY THE OUTPUT'S, because a norm that read two
     * bands and wrote one walks off the end of the destination buffer. */
    request.row_blocks = input->active_rows ? input->active_rows : 1u;
    if (output->active_rows && output->active_rows < request.row_blocks)
        request.row_blocks = output->active_rows;
    request.tile_cols = constants->tile_cols;
    request.prefetch_tiles = constants->tile_cols;
    request.input = pn_tt_operand(device, input);
    request.constants.x = (uint16_t)device->dram[constants->bank].x;
    request.constants.y = (uint16_t)device->dram[constants->bank].y;
    request.constants.address = constants->address;
    request.constants.tile_bytes = PN_TT_TILE_BF16;
    request.output = pn_tt_operand(device, output);
    if (ft_tt_op_reduce_parameters(&request, words) != FT_TT_OK)
        return -1;
    pn_tt_vec_probe_apply(words);
    /* M3 HALF TILE (word 3 bit 23, resident_vector_protocol.h ftv_half_tile): a single-token
     * forward's elementwise FPU passes run on faces 0-1. `PN_TT_REDUCE_HALF` 0 off, 1 (default)
     * honour the single-row promise, 2 RED: every descriptor, multi-row prefill too. */
    {
        static int half_mode = -1;

        if (half_mode < 0)
            half_mode = (int)pn_tt_env_u32("PN_TT_REDUCE_HALF", 1u);
        if (half_mode == 2 || (half_mode == 1 && (device->single_row ||
                                         (device->verify_rows && device->verify_rows <= 16u))))
            words[FT_TT_OP_VECTOR_DEPTH] |= 1u << 23;
    }

    started = pn_tt_now_seconds();
    if (device->reduce_members == 2u && device->reduce_member.running &&
        (constants->tile_cols % 2u) == 0u) {
        /* L4 COLUMN SPLIT: both members reduce the whole row; member m finishes half m. RED
         * `reduce_slice_dup`: member 1 also writes half 0, so half 1 is never written. */
        uint32_t second[FT_TT_OP_VECTOR_WORDS];
        int rc0, rc1 = 0;

        memcpy(second, words, sizeof(second));
        words[FT_TT_OP_REDUCE_FINAL] = (words[FT_TT_OP_REDUCE_FINAL] & 0x00FFFFFFu) |
                                       (0x20u << 24);
        second[FT_TT_OP_REDUCE_FINAL] =
            (second[FT_TT_OP_REDUCE_FINAL] & 0x00FFFFFFu) |
            ((pn_tt_red("reduce_slice_dup") ? 0x20u : 0x21u) << 24);
        rc0 = pn_tt_residency_enqueue(&device->reduce, FT_TT_QUEUE_OP_REDUCE, words, 1u);
        if (pn_tt_residency_enqueue(&device->reduce_member, FT_TT_QUEUE_OP_REDUCE, second, 1u) !=
            0)
            rc1 = -1;
        if (rc0 == 0 && pn_tt_residency_await(&device->reduce, PN_TT_OP_TIMEOUT_MS) != 0)
            rc0 = -1;
        if (rc1 == 0 && pn_tt_residency_await(&device->reduce_member, PN_TT_OP_TIMEOUT_MS) != 0)
            rc1 = -1;
        if (rc0 != 0 && (!pn_tt_oncard_rearm(device, &device->reduce) ||
                         pn_tt_residency_submit(&device->reduce, FT_TT_QUEUE_OP_REDUCE, words,
                                                PN_TT_OP_TIMEOUT_MS) != 0))
            return -1;
        if (rc1 != 0 && (!pn_tt_rearm_reduce_member(device) ||
                         pn_tt_residency_submit(&device->reduce_member, FT_TT_QUEUE_OP_REDUCE,
                                                second, PN_TT_OP_TIMEOUT_MS) != 0))
            return -1;
        pn_tt_oncard_account(device, "reduce", "rms_norm", constants->tile_cols,
                             pn_tt_now_seconds() - started);
        return 0;
    }
    if (pn_tt_residency_submit(&device->reduce, FT_TT_QUEUE_OP_REDUCE, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* The words address DRAM only, so the same descriptor is valid for a fresh
         * residency; nothing about it depended on the retired kernel. */
        if (!pn_tt_oncard_rearm(device, &device->reduce) ||
            pn_tt_residency_submit(&device->reduce, FT_TT_QUEUE_OP_REDUCE, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    pn_tt_oncard_account(device, "reduce", "rms_norm", constants->tile_cols,
                         pn_tt_now_seconds() - started);
    return 0;
}

/** One elementwise descriptor. `b` is NULL for a unary function. */
static int pn_tt_eltwise(pn_tt_device *device, ft_tt_op_vector_function function,
                         const pn_tt_activation *a, const pn_tt_activation *b,
                         const pn_tt_activation *output)
{
    ft_tt_op_eltwise_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    double started;
    uint32_t tiles;

    if (!device || !a || !output || !device->eltwise.running)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (b && b->tile_count < a->tile_count)
        return -1;
    if (output->tile_count < a->tile_count)
        return -1;
    /* AN ELEMENTWISE OP HAS NO ROW AXIS AT ALL, so Mt bands are one descriptor of Mt times the
     * tiles and the tile INDEX arithmetic is the same arithmetic. THE SPAN IS THE NARROWEST OF
     * THE THREE OPERANDS: one of them is often a single-band scratch — the GDN block's
     * `gdn_y_bridge` is written by a render inside one mixer pass — and widening the
     * descriptor past it would read tiles nothing wrote.
     *
     * The CB placement does not move with this: the elementwise buffers are `depth *
     * ELTWISE_CB_BLOCKS` pages (ft_tt_op_vector.c), sized from the block depth and never from
     * the tile count, so a wider descriptor is more blocks through the same buffer. The depth
     * condition below is preserved by a multiple: `n % 2d == 0` implies `k*n % 2d == 0`. */
    tiles = a->active_rows ? a->active_rows : 1u;
    if (b && b->active_rows && b->active_rows < tiles)
        tiles = b->active_rows;
    if (output->active_rows && output->active_rows < tiles)
        tiles = output->active_rows;
    tiles *= a->tile_count;
    /* The residency's ONE depth has to serve this descriptor too. Refusing here is a return
     * value; discovering it on the card costs the residency. */
    if ((tiles % (2u * device->eltwise_depth)) != 0u ||
        ft_tt_op_block_depth(tiles, device->eltwise_depth) != device->eltwise_depth) {
        fprintf(stderr, "pn_tt: %u tiles is not a whole number of depth-%u blocks\n",
                tiles, device->eltwise_depth);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    request.function = function;
    request.tile_count = tiles;
    request.prefetch_tiles = device->eltwise_depth;
    {
        static int row0_mode = -1;

        if (row0_mode < 0)
            row0_mode = (int)pn_tt_env_u32("PN_TT_ELTWISE_ROW0", 1u);
        request.row0 = row0_mode == 2 ||
                               (row0_mode == 1 &&
                                (device->single_row ||
                                 (device->verify_rows && device->verify_rows <= 4u)))
                           ? 1u : 0u;
    }
    request.input_a = pn_tt_operand(device, a);
    if (b)
        request.input_b = pn_tt_operand(device, b);
    request.output = pn_tt_operand(device, output);
    /* THE SPLIT TAKES EVERY DESCRIPTOR OR NONE. The residency was placed for the sliced
     * width, so there is no per-descriptor choice to make here — one member is the whole
     * pre-split path and N members is this same request over N disjoint tile ranges. */
    /* AND THE SPLIT TAKES A BAND-WIDE DESCRIPTOR WITHOUT KNOWING ABOUT BANDS, because the
     * slice is `[member * tiles/members, ...)` address arithmetic over an operation with no
     * row axis (pn_tt_eltwise_split.c). At `Mt` bands over `Mt` members each member gets
     * EXACTLY the single-band width the residency was armed for, and the CBs were sized from
     * the block DEPTH and never from the width, so a wider descriptor is more blocks through
     * the same buffer. A width the set does not divide already runs whole on member 0 by its
     * own rule. */
    if (pn_tt_eltwise_group_members(device) > 1u)
        return pn_tt_eltwise_group_submit(device, &request);
    /* PM 7: deferral is the group's; unsplit, every descriptor is awaited here. */
    device->eltwise_defer_next = false;
    if (pn_tt_eltwise_group_settle(device) != 0 ||
        ft_tt_op_eltwise_parameters(&request, words) != FT_TT_OK)
        return -1;
    pn_tt_vec_probe_apply(words);

    started = pn_tt_now_seconds();
    if (pn_tt_residency_submit(&device->eltwise, FT_TT_QUEUE_OP_ELTWISE, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        if (!pn_tt_oncard_rearm(device, &device->eltwise) ||
            pn_tt_residency_submit(&device->eltwise, FT_TT_QUEUE_OP_ELTWISE, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    pn_tt_oncard_account(device, "eltwise", ft_tt_op_vector_function_name(function), tiles,
                         pn_tt_now_seconds() - started);
    return 0;
}

int pn_tt_add(pn_tt_device *device, const pn_tt_activation *a, const pn_tt_activation *b,
              const pn_tt_activation *output)
{
    /* RED PROOF: the residual becomes a subtraction. */
    return pn_tt_eltwise(device, pn_tt_red("residual_sub") ? FT_TT_OP_VECTOR_FN_SUB
                                                           : FT_TT_OP_VECTOR_FN_ADD,
                         a, b, output);
}

int pn_tt_silu_mul(pn_tt_device *device, const pn_tt_activation *a, const pn_tt_activation *b,
                   const pn_tt_activation *output)
{
    /* RED PROOF: the two operands of the gate swap, which is the mutation a SwiGLU is most
     * likely to suffer for real and the one a shape check cannot see — both are n_ff. */
    if (pn_tt_red("swiglu_swap"))
        return pn_tt_eltwise(device, FT_TT_OP_VECTOR_FN_SILU_MUL, b, a, output);
    return pn_tt_eltwise(device, FT_TT_OP_VECTOR_FN_SILU_MUL, a, b, output);
}

int pn_tt_sigmoid_mul(pn_tt_device *device, const pn_tt_activation *gate,
                      const pn_tt_activation *b, const pn_tt_activation *scratch,
                      const pn_tt_activation *output)
{
    if (!scratch)
        return -1;
    /* TWO DESCRIPTORS, NOT A SEVENTH TABLE ROW. `sigmoid(a)*b` is NOT `silu(a)*b`
     * (`silu(a) = a*sigmoid(a)`), so the SwiGLU member cannot serve Ornith's attention gate —
     * but SIGMOID and MUL are both already in the elementwise kernel's switch, so the pair
     * costs one extra descriptor per attention layer (8 a token, ~2.4 ms) and NO kernel
     * source change, no program rebuild and no blob regeneration. A fused
     * `FT_TT_OP_VECTOR_FN_SIGMOID_MUL` would save that 2.4 ms and cost the whole provenance
     * chain; it is worth doing when something else is already regenerating the blob.
     *
     * RED PROOF: run the gate as SILU instead of SIGMOID. Same shapes, same descriptor
     * count, same operands — only the function differs, which is exactly the mistake the
     * SwiGLU next door invites and which no shape check can see. */
    if (pn_tt_eltwise(device, pn_tt_red("attn_gate_silu") ? FT_TT_OP_VECTOR_FN_SILU
                                                          : FT_TT_OP_VECTOR_FN_SIGMOID,
                      gate, NULL, scratch) != 0)
        return -1;
    return pn_tt_eltwise(device, FT_TT_OP_VECTOR_FN_MUL, scratch, b, output);
}

/** The kernel's order: key(bf16) descending, then index ascending (resident_topk_protocol.h). */
static uint32_t pn_tt_topk_key(float value)
{
    uint32_t bits;
    uint32_t u;

    memcpy(&bits, &value, sizeof(bits));
    u = bits >> 16;
    return (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
}

/** L3: every member selects over its slice, the host merges exactly. */
/* S2e (`PN_TT_TOPK_BULK`, default 1): the result block in ONE bulk L1 read (16 B-rounded)
 * instead of the register accessor's word by word. */
static int pn_tt_topk_read_block(pn_tt_device *device, uint32_t x, uint32_t y, uint32_t address,
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

static int pn_tt_topk_split(pn_tt_device *device, const pn_tt_activation *logits,
                            uint32_t count, uint32_t *out_indices, float *out_values,
                            uint32_t *out_count)
{
    const uint32_t members = device->sample_members;
    const uint32_t slice = device->sample_tile_count / members;
    uint32_t all_index[PN_TT_TOPK_MAX_MEMBERS * PN_TT_TOPK_SELECT_MAX];
    float all_value[PN_TT_TOPK_MAX_MEMBERS * PN_TT_TOPK_SELECT_MAX];
    uint32_t total = 0u, m, k;
    const bool red = pn_tt_red("topk_split_no_offset");
    double started = pn_tt_now_seconds();
    int failed = 0;

    for (m = 0; m < members; m++) {
        pn_tt_residency *r = m == 0u ? &device->sample : &device->sample_member[m - 1u];
        ft_tt_op_topk_request request;
        uint32_t words[FT_TT_OP_SAMPLE_WORDS];

        memset(&request, 0, sizeof(request));
        request.tile_count = slice;
        request.row = device->topk_row;
        request.select_count = count;
        request.prefetch_tiles = device->sample_depth;
        request.logits = pn_tt_operand(device, logits);
        request.logits.address += m * slice * PN_TT_TILE_BF16;
        request.result_address =
            m == 0u ? device->sample_result_address : device->sample_member_result[m - 1u];
        request.state_address =
            m == 0u ? device->sample_state_address : device->sample_member_state[m - 1u];
        if (ft_tt_op_sample_parameters(&request, words) != FT_TT_OK ||
            pn_tt_residency_enqueue(r, FT_TT_QUEUE_OP_TOPK, words, 1u) != 0)
            failed = 1;
    }
    for (m = 0; m < members; m++) {
        pn_tt_residency *r = m == 0u ? &device->sample : &device->sample_member[m - 1u];

        if (pn_tt_residency_await(r, PN_TT_OP_TIMEOUT_MS) != 0)
            failed = 1;
    }
    if (failed) {
        /* A member that idled out (a long prefill) or refused: re-arm the set and run the
         * whole split once more — TOPK has no side effect but its result blocks. */
        static int retrying;

        if (retrying)
            return -1;
        if (!pn_tt_oncard_rearm(device, &device->sample))
            return -1;
        for (m = 1u; m < members; m++) {
            const uint32_t core = device->sample_member[m - 1u].core_index;

            pn_tt_residency_stop(&device->sample_member[m - 1u]);
            pn_tt_residency_release(&device->sample_member[m - 1u]);
            device->oncard_restarts++;
            if (!pn_tt_arm_topk_member(device, m, core, slice, device->sample_depth))
                return -1;
        }
        retrying = 1;
        failed = pn_tt_topk_split(device, logits, count, out_indices, out_values, out_count);
        retrying = 0;
        return failed;
    }
    for (m = 0; m < members; m++) {
        const pn_tt_residency *r = m == 0u ? &device->sample : &device->sample_member[m - 1u];
        uint32_t block[FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                       PN_TT_TOPK_SELECT_MAX * FT_TT_OP_SAMPLE_PAIR_WORDS];
        uint32_t indices[PN_TT_TOPK_SELECT_MAX];
        float values[PN_TT_TOPK_SELECT_MAX];
        const size_t bytes = (FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                              count * FT_TT_OP_SAMPLE_PAIR_WORDS) * sizeof(uint32_t);
        uint32_t produced = 0u;

        if (pn_tt_topk_read_block(device, r->core_x, r->core_y,
                                  m == 0u ? device->sample_result_address
                                          : device->sample_member_result[m - 1u],
                                  block, bytes) != FT_TT_OK ||
            ft_tt_op_sample_result_decode(block, count, indices, values, &produced,
                                          NULL) != FT_TT_OK)
            return -1;
        device->bytes_read += bytes;
        for (k = 0; k < produced; k++) {
            all_index[total] = indices[k] + (red ? 0u : m * slice * 32u);
            all_value[total] = values[k];
            total++;
        }
    }
    /* Insertion sort, the kernel's comparator: key desc, then index asc. */
    for (k = 1; k < total; k++) {
        const uint32_t ix = all_index[k];
        const float v = all_value[k];
        const uint32_t key = pn_tt_topk_key(v);
        uint32_t j = k;

        while (j > 0u && (pn_tt_topk_key(all_value[j - 1u]) < key ||
                          (pn_tt_topk_key(all_value[j - 1u]) == key && all_index[j - 1u] > ix))) {
            all_index[j] = all_index[j - 1u];
            all_value[j] = all_value[j - 1u];
            j--;
        }
        all_index[j] = ix;
        all_value[j] = v;
    }
    if (total > count)
        total = count;
    for (k = 0; k < total; k++) {
        out_indices[k] = all_index[k];
        out_values[k] = all_value[k];
    }
    *out_count = total;
    pn_tt_oncard_account(device, "topk", "select", device->sample_tile_count,
                         pn_tt_now_seconds() - started);
    return 0;
}

int pn_tt_topk(pn_tt_device *device, const pn_tt_activation *logits, uint32_t count,
               uint32_t *out_indices, float *out_values, uint32_t *out_count)
{
    uint32_t block[FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                   PN_TT_TOPK_SELECT_MAX * FT_TT_OP_SAMPLE_PAIR_WORDS];
    const size_t bytes = (FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                          count * FT_TT_OP_SAMPLE_PAIR_WORDS) * sizeof(uint32_t);
    ft_tt_op_topk_request request;
    uint32_t words[FT_TT_OP_SAMPLE_WORDS];
    double started;
    int rc;

    if (!device || !logits || !out_indices || !out_values || !out_count)
        return -1;
    /* Chained projections are awaited HERE — pn_tt_device.h, pn_tt_matvec_enqueue. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    if (count == 0u || count > PN_TT_TOPK_SELECT_MAX || !device->sample.running)
        return -1;
    if (logits->tile_count < device->sample_tile_count)
        return -1;
    if (device->sample_members >= 2u)
        return pn_tt_topk_split(device, logits, count, out_indices, out_values, out_count);

    memset(&request, 0, sizeof(request));
    request.tile_count = device->sample_tile_count;
    /* a matvec is Mt=1; the token's logits are logical row 0. RED PROOF: row 1 is the tile
     * quantum's zero padding, so selecting over it returns an index of nothing. */
    request.row = pn_tt_red("topk_row") ? 1u : device->topk_row;
    request.select_count = count;
    request.prefetch_tiles = device->sample_depth;
    request.logits = pn_tt_operand(device, logits);
    request.result_address = device->sample_result_address;
    request.state_address = device->sample_state_address;
    if (ft_tt_op_sample_parameters(&request, words) != FT_TT_OK)
        return -1;

    started = pn_tt_now_seconds();
    if (pn_tt_residency_submit(&device->sample, FT_TT_QUEUE_OP_TOPK, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* Unlike the other two families, these words carry L1 addresses on the sampling
         * core, so they are RE-ENCODED against the fresh placement rather than reused. */
        if (!pn_tt_oncard_rearm(device, &device->sample))
            return -1;
        request.prefetch_tiles = device->sample_depth;
        request.result_address = device->sample_result_address;
        request.state_address = device->sample_state_address;
        if (ft_tt_op_sample_parameters(&request, words) != FT_TT_OK ||
            pn_tt_residency_submit(&device->sample, FT_TT_QUEUE_OP_TOPK, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0)
            return -1;
    }
    /* The result block is in L1 on the sampling core, so it takes the REGISTER accessor —
     * the bulk one refuses a non-DRAM address by design. It is ~96 bytes against the
     * 261,120 the logit row would have been. */
    rc = pn_tt_topk_read_block(device, device->sample.core_x, device->sample.core_y,
                               device->sample_result_address, block, bytes);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the top-N result block did not read back: %s\n",
                ft_tt_error_string(rc));
        return -1;
    }
    device->bytes_read += bytes;
    {
        uint32_t indices[PN_TT_TOPK_SELECT_MAX];
        float values[PN_TT_TOPK_SELECT_MAX];
        uint32_t produced = 0u, index;

        if (ft_tt_op_sample_result_decode(block, count, indices, values, &produced,
                                          NULL) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the top-N result block is not one (bad magic)\n");
            return -1;
        }
        if (produced > count)
            produced = count;
        for (index = 0; index < produced; index++) {
            out_indices[index] = indices[index];
            out_values[index] = values[index];
        }
        *out_count = produced;
    }
    pn_tt_oncard_account(device, "topk", "select", device->sample_tile_count,
                         pn_tt_now_seconds() - started);
    return 0;
}

#endif /* USE_TT */
