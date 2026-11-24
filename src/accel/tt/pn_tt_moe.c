/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_moe.c
 * @brief Implementation of the routed-expert block on the card (see pn_tt_moe.h).
 *
 * @status Decode (one row): router, routed gate/up, routed down + weighted sum. Phase 5.4.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_replica_map.h"

#include "tt/convert/ft_bfp.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/ft_tt_queue.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_matmul_resident_multicast.h"
#include "tt/kernels/resident_matmul_moe_protocol.h"

/** The router writer's red bits (foundry tt/kernels/ft_moe_route.h FTMR_RED_*; that header is
 *  a kernel header and is not included on the host). */
#define PN_TT_MOE_RED_TIE_FLIP 1u
#define PN_TT_MOE_RED_UNNORM 2u
#define PN_TT_MOE_RED_RANK_PERMUTE 4u

/** The two programs (foundry tt/tests/run_q36_phase5_*.sh build lines). */
#define PN_TT_MOE_EXPERT_PROGRAM "rtriad_bfp6"
#define PN_TT_MOE_EXPERT_PROGRAM_RED "rtriad_bfp6_red"
#define PN_TT_MOE_ROUTER_PROGRAM "rtriad_router"

/** Default cores: the router on the one core no Ornith family holds; the experts on the
 *  prefill grid's 48, which a MoE serve does not boot (the grid refuses held cores). */
#define PN_TT_MOE_ROUTER_CORE_DEFAULT 87u
#define PN_TT_MOE_CORE_FIRST_DEFAULT 0u
#define PN_TT_MOE_CORES_DEFAULT 48u

/** K-tiles a reader block carries: one down expert's Kt, which the K-concatenated
 *  descriptor requires (resident_matmul_moe_protocol.h), and a divisor of gate/up's 64. */
#define PN_TT_MOE_DEPTH 16u
#define PN_TT_MOE_BFP6_TILE 832u
#define PN_TT_MOE_BFP8_TILE 1088u
#define PN_TT_MOE_BF16_TILE 2048u
#define PN_TT_MOE_FP32_TILE 4096u
#define PN_TT_MOE_TILE_VALUES 1024u
/** CB indices the resident matmul triad is built for (COMPUTE_CTARGS 0,1,16,1). */
#define PN_TT_MOE_CB_IN0 0u
#define PN_TT_MOE_CB_IN1 1u
#define PN_TT_MOE_CB_OUT 16u
#define PN_TT_MOE_CB_SCRATCH 31u
/** Word 13: N-major B + the pipelined reader (the only path the MoE reader takes). */
#define PN_TT_MOE_B_FLAGS (1u | (1u << 8))

/** The matmul writer's PULL completion (foundry resident_queue_protocol.h kPull*): member 0's
 *  last descriptor of a phase posts only after every member's done-sequence reached the
 *  phase's, so the host (and a replayed token) awaits ONE queue, not M. */
#define PN_TT_MOE_PULL_TABLE (FT_TT_OP_QUEUE_BASE + 0xF900u)
#define PN_TT_MOE_PULL_MAGIC 0x46545150u
#define PN_TT_MOE_PULL_RED_SKIP 0x80000000u
/** Parameter-cache buckets a member hashes its blocks into (the region holds
 *  FT_TT_QUEUE_PARAM_CACHE_SLOTS blocks; a MoE member uses ~3 a layer). */
#define PN_TT_MOE_CACHE_BUCKETS 1024u
#define PN_TT_MOE_CACHE_EMPTY 0xFFFFu
/** Most descriptors one member takes in one phase (the ring holds the batch). */
#define PN_TT_MOE_PHASE_MAX 16u

/**
 * One member's PARAMETER CACHE, content-addressed (pn_tt_matmul_group.c's design): a MoE
 * descriptor's sixteen words are constants of the run (stack bases, strips, ranks), so each
 * block is written once into the member's cache region and every later descriptor names it.
 * The ring slot and the head bump are then a descriptor's only writes, host-issued or
 * replayed. Cleared whenever the member is (re-)armed; a full region falls back to the
 * rotating slot.
 */
typedef struct {
    uint32_t words[FT_TT_QUEUE_PARAM_CACHE_SLOTS][FT_TT_QUEUE_PARAM_WORDS];
    uint16_t bucket[PN_TT_MOE_CACHE_BUCKETS];
    uint32_t used;
} pn_tt_moe_param_cache;

struct pn_tt_moe {
    pn_tt_moe_shape shape;
    pn_tt_residency router;
    pn_tt_residency member[PN_TT_MOE_MAX_MEMBERS];
    uint32_t member_count;
    uint32_t member_core[PN_TT_MOE_MAX_MEMBERS];
    uint32_t member_scratch[PN_TT_MOE_MAX_MEMBERS];
    const char *member_program;
    uint64_t rearms;
    /** Descriptors issued to each member since the last await. */
    uint32_t member_issued[PN_TT_MOE_MAX_MEMBERS];
    uint32_t slot_bank, slot_address;
    pn_tt_fp32_strip logits;
    pn_tt_activation scale;        /**< decode's view of `scale_buffer` */
    pn_tt_activation scale_buffer; /**< the rows-mode scale strip (plan S1) */
    uint32_t red_bits;    /**< router writer red (PN_TT_MOE_RED_*) */
    bool red_idx_shift;
    bool red_stride_off;
    bool red_pull_skip;   /**< RED moe_pull_skip: member 0 posts without pulling */
    bool red_list_shift;  /**< RED moe_list_shift: every rows-mode list entry + 1 */
    pn_tt_moe_param_cache *cache[PN_TT_MOE_MAX_MEMBERS]; /**< NULL = PN_TT_MOE_PARAM_CACHE=0 */
    bool pull;            /**< the pull table is armed on member 0 */
    bool phase_pulled;    /**< this phase's member 0 carries the PULL: await it alone */
    uint32_t sequence;    /**< the phase counter the pull compares (never 0) */
    uint64_t cache_hits, cache_misses, cache_full;
    bool pull_verify;               /**< PN_TT_MOE_PULL_VERIFY=1 */
    uint32_t pull_delay_us, pull_delay_phase, phase_kind; /**< diagnostics (1 gate/up, 2 down) */
    uint64_t pull_checked, pull_early;
    uint64_t routes, expert_descriptors, awaits;
    double route_seconds, issue_seconds, await_seconds;
};

/* ============================================================
 * Weights
 * ============================================================ */

void pn_tt_moe_defer_weights(pn_tt_device *device, bool defer)
{
    if (device)
        device->moe_defer_weights = defer;
}

bool pn_tt_moe_weights_deferred(const pn_tt_device *device)
{
    return device && device->moe_defer_weights;
}

/**
 * One buffer of `bytes` in the bank with the MOST free DRAM. A stack is 218 MB and lives whole
 * in one bank, so a rotation that ignores what the dense load left uneven packs some banks to
 * the brim (measured: banks 0 and 1 at 229 KB free, the KV groups and GDNC refused) while
 * others keep room. Most-free keeps every bank within one stack of the others.
 */
static int moe_alloc(pn_tt_device *device, uint64_t bytes, const char *what, uint32_t *bank,
                     uint32_t *address)
{
    ft_tt_buffer buffer;
    uint32_t best = device->dram_bank_count;
    size_t best_free = 0u;

    for (uint32_t b = 0; b < device->dram_bank_count; b++) {
        const size_t free_bytes = ft_tt_allocator_free_bytes(&device->allocator,
                                                             FT_TT_MEM_SPACE_DRAM, b);

        if (free_bytes >= bytes && free_bytes > best_free) {
            best = b;
            best_free = free_bytes;
        }
    }
    if (best < device->dram_bank_count &&
        ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, best, (size_t)bytes,
                           &buffer) == FT_TT_OK) {
        *bank = buffer.logical_bank;
        *address = (uint32_t)buffer.address;
        return 0;
    }
    fprintf(stderr, "pn_tt: DRAM full placing %s (%.1f MiB)\n", what,
            (double)bytes / (1024.0 * 1024.0));
    return -1;
}

/** Tiles folded and written per step: one expert of Qwen3.6's stacks (832 KiB). */
#define PN_TT_MOE_FOLD_CHUNK_TILES 1024u

/**
 * Upload an expert stack in the card's FOLD layout (plan S3): the artifact holds the stream
 * layout, so each chunk is converted on the host (ft_bfp6_fold_from_stream, byte-exact to the
 * datum-level relayout) and written; the expander on the card then spends eleven ALU ops a
 * 16-datum group instead of ~46. Same values, same expansion bytes.
 */
static int moe_upload_folded(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                             uint32_t bank, uint32_t address)
{
    const unsigned char *source = device->artifact.base + entry->data_offset;
    const uint64_t tiles = entry->byte_count / PN_TT_MOE_BFP6_TILE;
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    unsigned char *folded = malloc((size_t)PN_TT_MOE_FOLD_CHUNK_TILES * PN_TT_MOE_BFP6_TILE);
    const double started = pn_tt_now_seconds();
    int rc = 0;

    if (!folded || tiles * PN_TT_MOE_BFP6_TILE != entry->byte_count ||
        entry->data_offset + entry->byte_count > (uint64_t)device->artifact.bytes) {
        free(folded);
        return -1;
    }
    for (uint64_t t = 0; t < tiles && rc == 0; t += PN_TT_MOE_FOLD_CHUNK_TILES) {
        const uint64_t count = tiles - t < PN_TT_MOE_FOLD_CHUNK_TILES ? tiles - t
                                                                      : PN_TT_MOE_FOLD_CHUNK_TILES;
        const uint64_t offset = t * PN_TT_MOE_BFP6_TILE;

        if (ft_bfp6_fold_from_stream(source + offset, (size_t)count, folded) != FT_BFP_OK ||
            bulk->write(bulk->user, (uint16_t)device->dram[bank].x, (uint16_t)device->dram[bank].y,
                        (uint64_t)address + offset, folded,
                        (uint32_t)(count * PN_TT_MOE_BFP6_TILE)) != FT_TT_OK)
            rc = -1;
    }
    free(folded);
    device->moe_fold_seconds += pn_tt_now_seconds() - started;
    return rc;
}

int pn_tt_moe_place_stack(pn_tt_device *device, const char *name, pn_tt_moe_stack *out)
{
    const ft_tt_artifact_entry *entry;
    uint64_t stride;
    uint32_t bank = 0u, address = 0u;

    if (!device || !name || !out || !device->artifact_open)
        return -1;
    entry = ft_tt_fttw_find(&device->artifact, name);
    if (!entry) {
        fprintf(stderr, "pn_tt: the artifact has no %s\n", name);
        return -1;
    }
    stride = ft_tt_fttw_expert_stride(entry);
    if (ft_tt_fttw_expert_count(entry) <= 1u || entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR ||
        entry->format != FT_TT_FORMAT_BFP6_B || stride == 0u || (stride & 63u) != 0u ||
        stride > 0xFFFFFFFFu || stride * ft_tt_fttw_expert_count(entry) != entry->byte_count) {
        fprintf(stderr, "pn_tt: %s is not a Bfp6_b N-major expert stack the MoE reader serves "
                        "(format %u, experts %u, stride %llu, %llu B)\n",
                name, entry->format, ft_tt_fttw_expert_count(entry),
                (unsigned long long)stride, (unsigned long long)entry->byte_count);
        return -1;
    }
    if (moe_alloc(device, entry->byte_count, name, &bank, &address) != 0)
        return -1;
    if (moe_upload_folded(device, entry, bank, address) != 0) {
        fprintf(stderr, "pn_tt: upload failed for %s\n", name);
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->bank = bank;
    out->address = address;
    out->x = (uint16_t)device->dram[bank].x;
    out->y = (uint16_t)device->dram[bank].y;
    out->tile_rows = entry->tile_rows;
    out->tile_cols = entry->tile_cols;
    out->experts = ft_tt_fttw_expert_count(entry);
    out->stride = (uint32_t)stride;
    device->resident_bytes += entry->byte_count;
    device->bytes_written += entry->byte_count;
    return 0;
}

/** bf16 bits of `value`, or false when it is not exactly representable. */
static bool moe_bf16_exact(float value, uint16_t *out)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    *out = (uint16_t)(bits >> 16);
    return (bits & 0xFFFFu) == 0u;
}

/** Face-ordered index of (row, column) inside one 32x32 tile. */
static uint32_t moe_face_index(uint32_t row, uint32_t column)
{
    return ((row / 16u) * 2u + column / 16u) * 256u + (row % 16u) * 16u + column % 16u;
}

/** Decode router tile `t` of `entry` (fp32 or Bfp8_b) into 1024 face-ordered floats. */
static int moe_router_tile(const pn_tt_device *device, const ft_tt_artifact_entry *entry,
                           uint32_t t, float *values)
{
    const unsigned char *source = device->artifact.base + entry->data_offset;

    if (entry->format == FT_TT_FORMAT_FP32) {
        memcpy(values, source + (size_t)t * PN_TT_MOE_FP32_TILE, PN_TT_MOE_FP32_TILE);
        return 0;
    }
    if (entry->format == FT_TT_FORMAT_BFP8_B)
        return ft_bfp_unpack_tiles(FT_BFP_BFP8_B, source + (size_t)t * PN_TT_MOE_BFP8_TILE, 1u,
                                   values) == FT_BFP_OK ? 0 : -1;
    return -1;
}

int pn_tt_moe_place_router(pn_tt_device *device, const char *name, const float *shexp_gate,
                           pn_tt_moe_router *out)
{
    const ft_tt_artifact_entry *entry;
    uint16_t *tiles = NULL;
    float values[PN_TT_MOE_TILE_VALUES];
    uint32_t kt, nt, kts, nts, bank = 0u, address = 0u;
    uint64_t inexact = 0u;
    size_t bytes;
    int rc = -1;

    if (!device || !name || !shexp_gate || !out || !device->artifact_open)
        return -1;
    entry = ft_tt_fttw_find(&device->artifact, name);
    if (!entry || entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR ||
        (entry->format != FT_TT_FORMAT_FP32 && entry->format != FT_TT_FORMAT_BFP8_B)) {
        fprintf(stderr, "pn_tt: %s is missing or not an fp32 / Bfp8_b N-major router\n", name);
        return -1;
    }
    kts = entry->tile_rows;
    nts = entry->tile_cols + 1u; /* + the shared-expert gate's column tile */
    bytes = (size_t)kts * nts * PN_TT_MOE_BF16_TILE;
    tiles = calloc(1, bytes);
    if (!tiles)
        return -1;
    for (nt = 0; nt < entry->tile_cols; nt++) {
        for (kt = 0; kt < kts; kt++) {
            uint16_t *tile = tiles + ((size_t)nt * kts + kt) * PN_TT_MOE_TILE_VALUES;

            if (moe_router_tile(device, entry, nt * kts + kt, values) != 0)
                goto done;
            for (uint32_t i = 0; i < PN_TT_MOE_TILE_VALUES; i++)
                inexact += !moe_bf16_exact(values[i], &tile[i]);
        }
    }
    for (kt = 0; kt < kts; kt++) {
        uint16_t *tile = tiles + ((size_t)entry->tile_cols * kts + kt) * PN_TT_MOE_TILE_VALUES;

        for (uint32_t row = 0; row < 32u; row++)
            inexact += !moe_bf16_exact(shexp_gate[kt * 32u + row],
                                       &tile[moe_face_index(row, 0u)]);
    }
    if (inexact) {
        fprintf(stderr, "pn_tt: %s: %llu router / shared-gate values are not exactly bf16 — the "
                        "card would round them (spec §13 P5.3 measured 0); refused\n",
                name, (unsigned long long)inexact);
        goto done;
    }
    if (moe_alloc(device, bytes, name, &bank, &address) != 0 ||
        pn_tt_write_dram(device, bank, address, tiles, bytes) != 0)
        goto done;
    out->bank = bank;
    out->address = address;
    out->x = (uint16_t)device->dram[bank].x;
    out->y = (uint16_t)device->dram[bank].y;
    out->tile_rows = kts;
    out->tile_cols = nts;
    device->resident_bytes += bytes;
    rc = 0;
done:
    free(tiles);
    return rc;
}

/* ============================================================
 * Residencies
 * ============================================================ */

/** Place the matmul CBs (and the scratch) and return the scratch's L1 address. */
static int moe_place_cbs(pn_tt_residency *r, uint32_t in1_page, uint32_t out_page,
                         uint32_t scratch_bytes, uint32_t in0_rows, uint32_t *scratch)
{
    const ft_tt_op_cb_request requests[4] = {
        { PN_TT_MOE_CB_IN0, PN_TT_MOE_BF16_TILE, PN_TT_MOE_DEPTH * 2u * in0_rows },
        { PN_TT_MOE_CB_IN1, in1_page, PN_TT_MOE_DEPTH * 2u },
        { PN_TT_MOE_CB_OUT, out_page, 2u },
        { PN_TT_MOE_CB_SCRATCH, scratch_bytes, 1u },
    };
    ft_tt_launch_cb_entry placed[4];

    if (ft_tt_op_cb_place(requests, 4u, placed, r->cb_buffer, sizeof(r->cb_buffer),
                          &r->cb_mask) != FT_TT_OK || (placed[3].address & 63u) != 0u)
        return -1;
    *scratch = placed[3].address;
    return 0;
}

/** Copy runtime args [ten, `count`) of `args` into data mover `mover`'s tail. */
static void moe_set_tail(pn_tt_residency *r, uint32_t mover, const uint32_t *args,
                         uint32_t count)
{
    r->tail_count[mover] = count - FT_TT_QUEUE_KERNEL_ARG_COUNT;
    memcpy(r->tail_args[mover], args + FT_TT_QUEUE_KERNEL_ARG_COUNT,
           r->tail_count[mover] * sizeof(uint32_t));
}

static int moe_arm_router(pn_tt_device *device, struct pn_tt_moe *moe)
{
    const uint32_t core = pn_tt_env_u32("PN_TT_MOE_ROUTER_CORE", PN_TT_MOE_ROUTER_CORE_DEFAULT);
    const pn_tt_moe_shape *shape = &moe->shape;
    pn_tt_residency *r = &moe->router;
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT + PN_TT_RESIDENCY_TAIL_ARGS];
    uint32_t scratch = 0u;

    if (!pn_tt_prepare_family_on_core(device, r, PN_TT_MOE_ROUTER_PROGRAM, "moe-router",
                                      "PN_TT_MOE_ROUTER_CORE", core, PN_TT_ONCARD_TRIAD_MASK))
        return -1;
    /* Rows mode (plan S1) routes up to FT_TT_MOE_ROWS_BANDS_MAX tile rows a descriptor, one
     * row block at a time (MB = 1): CB_IN0 holds two chunks of one tile row. */
    if (moe_place_cbs(r, PN_TT_MOE_BF16_TILE, PN_TT_MOE_FP32_TILE, FT_TT_MOE_ROUTER_SCRATCH_BYTES,
                      1u, &scratch) != 0) {
        pn_tt_residency_release(r);
        return -1;
    }
    /* The reader's cooperative tail is planted as SOLO zeros: L1 is not cleared between
     * residencies (pn_tt_matmul_group.c says what a stale role costs). */
    memset(args, 0, sizeof(args));
    moe_set_tail(r, 0u, args, kArgCoopCount);
    args[kArgRouteLogits] = shape->experts;
    args[kArgRouteShexpColumn] = shape->experts;
    args[kArgRouteK] = shape->k;
    args[kArgRouteSlotX] = device->dram[moe->slot_bank].x;
    args[kArgRouteSlotY] = device->dram[moe->slot_bank].y;
    args[kArgRouteSlotAddress] = moe->slot_address;
    args[kArgRouteScaleX] = device->dram[moe->scale.bank].x;
    args[kArgRouteScaleY] = device->dram[moe->scale.bank].y;
    args[kArgRouteScaleAddress] = moe->scale.address;
    args[kArgRouteExpertTiles] = shape->expert_tiles;
    args[kArgRouteShexpTiles] = shape->shexp_tiles;
    args[kArgRouteScratch] = scratch;
    args[kArgRouteRed] = moe->red_bits;
    moe_set_tail(r, 1u, args, kArgRouteCount);
    return pn_tt_start_family(r) ? 0 : -1;
}

/** Forget member `m`'s cached blocks (its residency was (re-)armed: the region is new). */
static void moe_cache_reset(struct pn_tt_moe *moe, uint32_t m)
{
    pn_tt_moe_param_cache *c = moe->cache[m];

    if (!c)
        return;
    c->used = 0u;
    for (uint32_t b = 0; b < PN_TT_MOE_CACHE_BUCKETS; b++)
        c->bucket[b] = PN_TT_MOE_CACHE_EMPTY;
}

/** FNV-1a over the sixteen words: a bucket index, never the claim (the memcmp is). */
static uint32_t moe_cache_hash(const uint32_t *words)
{
    uint32_t h = 2166136261u;

    for (uint32_t w = 0; w < FT_TT_QUEUE_PARAM_WORDS; w++)
        h = (h ^ words[w]) * 16777619u;
    return h;
}

/**
 * The cached block holding exactly `words` on member `m`, written now on a miss. 0 = not
 * cacheable (no cache, region full, or the write failed): the caller uses the rotating slot.
 */
static uint32_t moe_cache_address(struct pn_tt_moe *moe, uint32_t m,
                                  const uint32_t *words)
{
    pn_tt_moe_param_cache *c = moe->cache[m];
    pn_tt_residency *r = &moe->member[m];
    uint32_t bucket, address = 0u;

    if (!c)
        return 0u;
    bucket = moe_cache_hash(words) & (PN_TT_MOE_CACHE_BUCKETS - 1u);
    for (uint32_t probe = 0; probe < PN_TT_MOE_CACHE_BUCKETS; probe++) {
        const uint32_t index = (bucket + probe) & (PN_TT_MOE_CACHE_BUCKETS - 1u);
        const uint16_t entry = c->bucket[index];

        if (entry == PN_TT_MOE_CACHE_EMPTY) {
            if (c->used >= FT_TT_QUEUE_PARAM_CACHE_SLOTS) {
                moe->cache_full++;
                return 0u;
            }
            if (ft_tt_queue_write_param_cache(pn_tt_issue_access(r->runtime, r->issue_shape),
                                              &r->queue, c->used, words, &address) != FT_TT_OK)
                return 0u;
            memcpy(c->words[c->used], words, sizeof(c->words[0]));
            c->bucket[index] = (uint16_t)c->used;
            c->used++;
            moe->cache_misses++;
            return address;
        }
        if (memcmp(c->words[entry], words, sizeof(c->words[0])) == 0) {
            moe->cache_hits++;
            return FT_TT_QUEUE_PARAM_CACHE_ADDRESS(r->queue.control_address, entry);
        }
    }
    return 0u;
}

/**
 * Plant the PULL table in member 0's queue region (magic, count, x | y << 16 per member) and
 * read it back; `moe->pull` says whether it took. Re-planted whenever member 0 is armed.
 */
static void moe_plant_pull_table(pn_tt_device *device, struct pn_tt_moe *moe)
{
    uint32_t table[2u + PN_TT_MOE_MAX_MEMBERS], back[2u + PN_TT_MOE_MAX_MEMBERS];
    const uint32_t words = 2u + moe->member_count;
    const pn_tt_residency *leader = &moe->member[0];

    /* OFF BY DEFAULT (`PN_TT_MOE_PULL=1` arms it). Measured 2026-11-02: with the five-RISC-V
     * expander the pulled phases change logits (the per-position card dump of agentic_0 moved
     * at 5 of 63 argmaxes against the same code with the pull off, which is byte-identical to
     * Phase 5), while a host delay of ~30 us after the pulled gate/up await, or a single-RISC-V
     * expander, restores identity, and every member's completion was verified done when the
     * leader posted. Cause not found (qwen36 spec §14); the pull bought no decode speed
     * (12.32 vs 12.29 tok/s), so it stays off. */
    moe->pull = false;
    if (pn_tt_env_u32("PN_TT_MOE_PULL", 0u) == 0u || moe->member_count < 2u || !moe->cache[0])
        return;
    table[0] = PN_TT_MOE_PULL_MAGIC;
    table[1] = moe->member_count;
    for (uint32_t m = 0; m < moe->member_count; m++)
        table[2u + m] = (uint32_t)moe->member[m].core_x | ((uint32_t)moe->member[m].core_y << 16);
    if (device->runtime.noc.write(device->runtime.noc.user, leader->core_x, leader->core_y,
                                  PN_TT_MOE_PULL_TABLE, table, words * 4u) == FT_TT_OK &&
        device->runtime.noc.read(device->runtime.noc.user, leader->core_x, leader->core_y,
                                 PN_TT_MOE_PULL_TABLE, back, words * 4u) == FT_TT_OK &&
        memcmp(table, back, words * 4u) == 0)
        moe->pull = true;
}

/** Zero the expansion mailbox before the residency boots: job ids restart with it
 *  (foundry resident_matmul_moe_helpers.h). */
static int moe_clear_mailbox(pn_tt_device *device, const pn_tt_residency *r)
{
    const uint32_t zeros[FT_TT_MOE_MAILBOX_BYTES / 4u] = { 0u };

    return device->runtime.noc.write(device->runtime.noc.user, r->core_x, r->core_y,
                                     FT_TT_OP_QUEUE_BASE + FT_TT_MOE_MAILBOX_OFFSET, zeros,
                                     sizeof(zeros)) == FT_TT_OK ? 0 : -1;
}

/** Tile rows CB_IN0 holds two chunks of: the batched prefill's group height when it is on
 *  (`PN_TT_MOE_PREFILL`, `PN_TT_MOE_PREFILL_BANDS`, read the same way the model reads them),
 *  else one (decode alone). */
static uint32_t moe_in0_rows(void)
{
    const char *on = getenv("PN_TT_MOE_PREFILL");
    const char *bands = getenv("PN_TT_MOE_PREFILL_BANDS");
    long rows = bands && *bands ? atol(bands) : (long)FT_TT_MOE_ROWS_BANDS_MAX;

    if (on && *on && atoi(on) == 0)
        return 1u;
    return rows == 1 || rows == 2 ? (uint32_t)rows : FT_TT_MOE_ROWS_BANDS_MAX;
}

static int moe_arm_member(pn_tt_device *device, struct pn_tt_moe *moe, uint32_t m,
                          uint32_t core, const char *program)
{
    pn_tt_residency *r = &moe->member[m];
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT + PN_TT_RESIDENCY_TAIL_ARGS];
    uint32_t scratch = 0u;

    if (!pn_tt_prepare_family_on_core(device, r, program, "moe-expert", "PN_TT_MOE_CORE_FIRST",
                                      core, PN_TT_ONCARD_TRIAD_MASK))
        return -1;
    /* A prefill group's descriptors carry its bands as ONE row block (MB = bands, plan S1), so
     * every weight column is read and expanded once a group: CB_IN0 holds two chunks of that. */
    if (moe_place_cbs(r, PN_TT_MOE_BFP8_TILE, PN_TT_MOE_BF16_TILE, FT_TT_MOE_SCRATCH_BYTES,
                      moe_in0_rows(), &scratch) != 0 ||
        moe_clear_mailbox(device, r) != 0) {
        pn_tt_residency_release(r);
        return -1;
    }
    memset(args, 0, sizeof(args));
    args[kArgCoopRole] = FT_TT_RESIDENT_COOP_ROLE_SOLO;
    args[kArgMoeSlotX] = device->dram[moe->slot_bank].x;
    args[kArgMoeSlotY] = device->dram[moe->slot_bank].y;
    args[kArgMoeSlotAddress] = moe->slot_address;
    /* RED moe_stride_off: every expert read one tile past its panel. */
    args[kArgMoeExpertStride] =
        moe->shape.stride + (moe->red_stride_off ? PN_TT_MOE_BFP6_TILE : 0u);
    args[kArgMoeExpertCount] = moe->shape.experts;
    args[kArgMoeScratch] = scratch;
    moe_set_tail(r, 0u, args, kArgMoeCount);
    moe->member_scratch[m] = scratch;
    moe_cache_reset(moe, m);
    return pn_tt_start_family(r) ? 0 : -1;
}

/** The expert cores: `PN_TT_MOE_CORE_LIST` (comma list) or a run from the first core. */
static uint32_t moe_member_cores(uint32_t *cores)
{
    const char *list = getenv("PN_TT_MOE_CORE_LIST");
    uint32_t count = 0u;

    if (list && *list) {
        const char *at = list;

        while (*at && count < PN_TT_MOE_MAX_MEMBERS) {
            char *end = NULL;
            const unsigned long core = strtoul(at, &end, 10);

            if (end == at)
                return 0u;
            cores[count++] = (uint32_t)core;
            at = *end == ',' ? end + 1 : end;
        }
        return count;
    }
    count = pn_tt_env_u32("PN_TT_MOE_CORES", PN_TT_MOE_CORES_DEFAULT);
    if (count == 0u || count > PN_TT_MOE_MAX_MEMBERS)
        return 0u;
    for (uint32_t m = 0; m < count; m++)
        cores[m] = pn_tt_env_u32("PN_TT_MOE_CORE_FIRST", PN_TT_MOE_CORE_FIRST_DEFAULT) + m;
    return count;
}

static int moe_alloc_state(pn_tt_device *device, struct pn_tt_moe *moe)
{
    const uint32_t scale_tiles = moe->shape.k * moe->shape.expert_tiles + moe->shape.shexp_tiles;
    const uint32_t logit_tiles = moe->shape.experts / 32u + 1u;

    /* The logits strip holds a rows-mode router's bands (plan S1); decode uses its first
     * tile row. The scale buffer is the rows-mode layout (resident_matmul_moe_protocol.h
     * "BATCHED PREFILL"), zeroed once so entries past a list only ever hold finite scales;
     * decode's strip is a view of its first k*E + S tiles. */
    if (moe->shape.expert_tiles > FT_TT_MOE_EXPERT_TILES_MAX ||
        moe->shape.experts > FT_TT_MOE_LIST_MAX ||
        pn_tt_fp32_strip_alloc(device, logit_tiles * FT_TT_MOE_ROWS_BANDS_MAX, &moe->logits) != 0 ||
        pn_tt_activation_alloc_rows(device, FT_TT_MOE_SCALE_STRIP_TILES, 1u, &moe->scale_buffer) != 0)
        return -1;
    moe->logits.tile_count = logit_tiles;
    moe->scale = moe->scale_buffer;
    moe->scale.tile_count = scale_tiles;

    if (moe_alloc(device, FT_TT_MOE_SLOT_ENTRIES * 4u, "the routing slot", &moe->slot_bank,
                  &moe->slot_address) != 0 ||
        pn_tt_zero_dram(device, moe->slot_bank, moe->slot_address,
                        FT_TT_MOE_SLOT_ENTRIES * 4u) != 0)
        return -1;
    return 0;
}

static void moe_read_reds(struct pn_tt_moe *moe)
{
    if (pn_tt_red("moe_tie_flip"))
        moe->red_bits |= PN_TT_MOE_RED_TIE_FLIP;
    if (pn_tt_red("moe_weights_unnorm"))
        moe->red_bits |= PN_TT_MOE_RED_UNNORM;
    if (pn_tt_red("moe_rank_permute"))
        moe->red_bits |= PN_TT_MOE_RED_RANK_PERMUTE;
    moe->red_idx_shift = pn_tt_red("moe_idx_shift");
    moe->red_stride_off = pn_tt_red("moe_stride_off");
    moe->red_pull_skip = pn_tt_red("moe_pull_skip");
    moe->red_list_shift = pn_tt_red("moe_list_shift");
    if (moe->red_bits || moe->red_idx_shift || moe->red_stride_off || moe->red_pull_skip ||
        moe->red_list_shift ||
        pn_tt_red("bfp6_expand_bit"))
        fprintf(stderr, "pn_tt: MoE RED ARM armed — this run's tokens are WRONG on purpose\n");
}

int pn_tt_moe_arm(pn_tt_device *device, const pn_tt_moe_shape *shape)
{
    uint32_t cores[PN_TT_MOE_MAX_MEMBERS];
    struct pn_tt_moe *moe;
    const char *program;
    uint32_t count, m;

    if (!device || !shape || device->moe || shape->k == 0u || shape->k > PN_TT_MOE_MAX_K ||
        shape->experts == 0u || shape->experts % 32u != 0u || shape->expert_tiles == 0u ||
        shape->expert_tiles % PN_TT_MOE_DEPTH != 0u || shape->stride == 0u)
        return -1;
    count = moe_member_cores(cores);
    if (count == 0u) {
        fprintf(stderr, "pn_tt: PN_TT_MOE_CORES / PN_TT_MOE_CORE_LIST name no usable cores "
                        "(1..%u)\n", PN_TT_MOE_MAX_MEMBERS);
        return -1;
    }
    moe = calloc(1, sizeof(*moe));
    if (!moe)
        return -1;
    moe->shape = *shape;
    moe_read_reds(moe);
    device->moe = moe;
    if (moe_alloc_state(device, moe) != 0 || moe_arm_router(device, moe) != 0)
        goto fail;
    /* PN_TT_MOE_EXPERT_PROGRAM names another build of the family (an A/B arm: the expander
     * on fewer workers, rtriad_bfp6_w1 / _w2); the red always takes the red build. */
    program = pn_tt_red("bfp6_expand_bit") ? PN_TT_MOE_EXPERT_PROGRAM_RED
              : (getenv("PN_TT_MOE_EXPERT_PROGRAM") && *getenv("PN_TT_MOE_EXPERT_PROGRAM"))
                  ? getenv("PN_TT_MOE_EXPERT_PROGRAM")
                  : PN_TT_MOE_EXPERT_PROGRAM;
    moe->member_program = program;
    if (pn_tt_env_u32("PN_TT_MOE_PARAM_CACHE", 1u) != 0u)
        for (m = 0; m < count; m++) {
            moe->cache[m] = malloc(sizeof(*moe->cache[m]));
            if (!moe->cache[m])
                goto fail;
            moe_cache_reset(moe, m);
        }
    for (m = 0; m < count; m++) {
        moe->member_core[m] = cores[m];
        if (moe_arm_member(device, moe, m, cores[m], program) != 0)
            goto fail;
        moe->member_count = m + 1u;
    }
    moe_plant_pull_table(device, moe);
    moe->pull_verify = pn_tt_env_u32("PN_TT_MOE_PULL_VERIFY", 0u) != 0u;
    moe->pull_delay_us = pn_tt_env_u32("PN_TT_MOE_PULL_DELAY_US", 0u);
    moe->pull_delay_phase = pn_tt_env_u32("PN_TT_MOE_PULL_DELAY_PHASE", 0u);
    fprintf(stderr, "pn_tt: MoE block ARMED — router on core %u, %u expert residencies (%s) "
                    "from core %u; top-%u of %u, expert ff %u, shared ff %u; parameter cache "
                    "%s, PULL completion %s%s\n",
            moe->router.core_index, count, program, cores[0], shape->k, shape->experts,
            shape->expert_tiles * 32u, shape->shexp_tiles * 32u, moe->cache[0] ? "on" : "OFF",
            moe->pull ? "on" : "OFF", moe->red_pull_skip ? " (RED: member 0 SKIPS the pull)" : "");
    return 0;
fail:
    fprintf(stderr, "pn_tt: the MoE block could not be armed\n");
    pn_tt_moe_release(device);
    return -1;
}

void pn_tt_moe_release(pn_tt_device *device)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;

    if (!moe)
        return;
    for (uint32_t m = 0; m < moe->member_count; m++)
        pn_tt_residency_release(&moe->member[m]);
    pn_tt_residency_release(&moe->router);
    for (uint32_t m = 0; m < PN_TT_MOE_MAX_MEMBERS; m++)
        free(moe->cache[m]);
    free(moe);
    device->moe = NULL;
}

double pn_tt_moe_fold_seconds(const pn_tt_device *device)
{
    return device ? device->moe_fold_seconds : 0.0;
}

bool pn_tt_moe_armed(const pn_tt_device *device)
{
    return device && device->moe && device->moe->member_count > 0u;
}

const pn_tt_activation *pn_tt_moe_scale_strip(const pn_tt_device *device)
{
    return pn_tt_moe_armed(device) ? &device->moe->scale : NULL;
}

const char *pn_tt_moe_core_holder(const pn_tt_device *device, uint32_t core,
                                  const void *except)
{
    const struct pn_tt_moe *moe = device ? device->moe : NULL;

    if (!moe)
        return NULL;
    if ((const void *)&moe->router != except && moe->router.core_booted &&
        moe->router.core_index == core)
        return "the MoE router";
    for (uint32_t m = 0; m < moe->member_count; m++)
        if ((const void *)&moe->member[m] != except && moe->member[m].core_booted &&
            moe->member[m].core_index == core)
            return "a MoE expert residency";
    return NULL;
}

/* ============================================================
 * Descriptors
 * ============================================================ */

/** The sixteen words of one Mt = 1 resident matmul. */
static void moe_words(uint32_t *w, uint32_t kt, uint32_t nt, uint16_t ax, uint16_t ay,
                      uint32_t a, uint16_t bx, uint16_t by, uint32_t b, uint16_t cx, uint16_t cy,
                      uint32_t c, uint32_t flags)
{
    memset(w, 0, FT_TT_QUEUE_PARAM_WORDS * sizeof(uint32_t));
    w[FT_TT_QUEUE_MATMUL_TILE_ROWS] = 1u;
    w[FT_TT_QUEUE_MATMUL_TILE_INNER] = kt;
    w[FT_TT_QUEUE_MATMUL_TILE_COLS] = nt;
    w[FT_TT_QUEUE_MATMUL_A_X] = ax;
    w[FT_TT_QUEUE_MATMUL_A_Y] = ay;
    w[FT_TT_QUEUE_MATMUL_A_BASE] = a;
    w[FT_TT_QUEUE_MATMUL_B_X] = bx;
    w[FT_TT_QUEUE_MATMUL_B_Y] = by;
    w[FT_TT_QUEUE_MATMUL_B_BASE] = b;
    w[FT_TT_QUEUE_MATMUL_C_X] = cx;
    w[FT_TT_QUEUE_MATMUL_C_Y] = cy;
    w[FT_TT_QUEUE_MATMUL_C_BASE] = c;
    w[FT_TT_QUEUE_MATMUL_DEPTH] = PN_TT_MOE_DEPTH;
    w[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = PN_TT_MOE_B_FLAGS | flags;
}

/**
 * A residency the idle retire (or a refusal) stopped is armed again on its own core before
 * the next descriptor: the descriptors address DRAM only, so a fresh residency runs them the
 * same. Only between batches (nothing outstanding on it).
 */
static int moe_rearm(pn_tt_device *device, struct pn_tt_moe *moe, pn_tt_residency *r)
{
    const bool router = r == &moe->router;
    const uint32_t m = router ? 0u : (uint32_t)(r - moe->member);

    fprintf(stderr, "pn_tt: MoE %s on core %u is not running — re-arming it\n",
            router ? "router" : "expert residency", r->core_index);
    pn_tt_residency_stop(r);
    pn_tt_residency_release(r);
    moe->rearms++;
    if (router)
        return moe_arm_router(device, moe);
    if (moe_arm_member(device, moe, m, moe->member_core[m], moe->member_program) != 0)
        return -1;
    if (m == 0u && moe->pull)
        moe_plant_pull_table(device, moe);
    return 0;
}

/**
 * Enqueue one matmul on member `m`: from its parameter cache when the block is cacheable
 * (ring slot + head only), else in its own rotating parameter slot (the ring holds the batch).
 * `a3`/`a4` are member 0's PULL words on its last descriptor of a phase, `a5` the phase's
 * done-sequence on every member's last descriptor (0 elsewhere).
 * @return 0, 1 when the PULL words could not be carried (rotating-slot fallback), or -1
 */
static int moe_issue(pn_tt_device *device, struct pn_tt_moe *moe, uint32_t m,
                     const uint32_t *words, uint32_t a3, uint32_t a4, uint32_t a5)
{
    pn_tt_residency *r = &moe->member[m];
    uint32_t address;
    int rc;

    if (moe->member_issued[m] == 0u && !r->running && moe_rearm(device, moe, r) != 0)
        return -1;
    if (moe->member_issued[m] + 1u > r->queue.capacity)
        return -1;
    address = moe_cache_address(moe, m, words);
    if (address != 0u) {
        rc = pn_tt_residency_enqueue_cached(r, FT_TT_QUEUE_OP_MATMUL, address, 0u, 0u, a3, a4, a5);
        /* The idle retire fires INSIDE the enqueue (it stops the residency and refuses). */
        if (rc != 0 && moe->member_issued[m] == 0u && !r->running &&
            moe_rearm(device, moe, r) == 0) {
            address = moe_cache_address(moe, m, words);
            rc = address ? pn_tt_residency_enqueue_cached(r, FT_TT_QUEUE_OP_MATMUL, address, 0u,
                                                          0u, a3, a4, a5)
                         : -1;
        }
        if (rc != 0)
            return -1;
        moe->member_issued[m]++;
        moe->expert_descriptors++;
        return 0;
    }
    r->param_slot = moe->member_issued[m];
    rc = pn_tt_residency_enqueue_group(r, FT_TT_QUEUE_OP_MATMUL, words, 1u, a5, NULL);
    if (rc != 0 && moe->member_issued[m] == 0u && !r->running && moe_rearm(device, moe, r) == 0) {
        r->param_slot = 0u;
        rc = pn_tt_residency_enqueue_group(r, FT_TT_QUEUE_OP_MATMUL, words, 1u, a5, NULL);
    }
    r->param_slot = 0u;
    if (rc != 0)
        return -1;
    moe->member_issued[m]++;
    moe->expert_descriptors++;
    return a3 != 0u ? 1 : 0;
}

/** A phase's next done-sequence (never 0: a5 = 0 is "post nothing"). */
static uint32_t moe_next_sequence(struct pn_tt_moe *moe)
{
    if (++moe->sequence == 0u)
        moe->sequence = 1u;
    return moe->sequence;
}

/**
 * Issue member `m`'s `count` descriptors of a phase: the last carries the phase's
 * done-sequence, and member 0's last also the PULL over every member. Clears
 * `moe->phase_pulled` if the pull could not be carried.
 */
static int moe_issue_phase(pn_tt_device *device, struct pn_tt_moe *moe, uint32_t m,
                           uint32_t (*words)[FT_TT_QUEUE_PARAM_WORDS], uint32_t count,
                           uint32_t sequence)
{
    for (uint32_t d = 0; d < count; d++) {
        const bool last = d + 1u == count;
        const bool pull = last && m == 0u && moe->phase_pulled;
        const uint32_t a3 =
            pull ? moe->member_count | (moe->red_pull_skip ? PN_TT_MOE_PULL_RED_SKIP : 0u) : 0u;
        const int rc = moe_issue(device, moe, m, words[d], a3, pull ? sequence : 0u,
                                 last && moe->phase_pulled ? sequence : 0u);

        if (rc < 0)
            return -1;
        if (rc == 1)
            moe->phase_pulled = false;
    }
    return 0;
}

/** Member `m`'s view of a strip: mirror `m % replicas`, so the members spread over banks. */
static void moe_operand(const pn_tt_device *device, const pn_tt_activation *a, uint32_t m,
                        uint16_t *x, uint16_t *y, uint32_t *address)
{
    const uint32_t replica = a->replica_count > 1u ? m % a->replica_count : 0u;
    const uint32_t bank = replica ? a->replica_bank[replica] : a->bank;

    *x = (uint16_t)device->dram[bank].x;
    *y = (uint16_t)device->dram[bank].y;
    *address = replica ? a->replica_address[replica] : a->address;
}

int pn_tt_moe_route(pn_tt_device *device, const pn_tt_moe_router *router,
                    const pn_tt_activation *normed)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
    uint16_t ax, ay;
    uint32_t a;

    if (!moe || !router || !normed || normed->tile_count != router->tile_rows)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    moe_operand(device, normed, 0u, &ax, &ay, &a);
    moe_words(words, router->tile_rows, router->tile_cols, ax, ay, a, router->x, router->y,
              router->address, (uint16_t)device->dram[moe->logits.bank].x,
              (uint16_t)device->dram[moe->logits.bank].y, moe->logits.address, 0u);
    if ((!moe->router.running && moe_rearm(device, moe, &moe->router) != 0) ||
        pn_tt_residency_submit(&moe->router, FT_TT_QUEUE_OP_MATMUL, words,
                               PN_TT_OP_TIMEOUT_MS) != 0) {
        /* Idle-retired inside the enqueue, or refused: one fresh residency, one retry. */
        if (moe_rearm(device, moe, &moe->router) != 0 ||
            pn_tt_residency_submit(&moe->router, FT_TT_QUEUE_OP_MATMUL, words,
                                   PN_TT_OP_TIMEOUT_MS) != 0) {
            fprintf(stderr, "pn_tt: the MoE router descriptor failed\n");
            return -1;
        }
    }
    /* RED moe_idx_shift: every routed index moves to the next expert — a wrong routing
     * decision reaching the readers, written where the router wrote it. */
    if (moe->red_idx_shift) {
        /* One whole 64 B segment: DRAM reads and writes are 64 B multiples. */
        uint32_t slot[FT_TT_MOE_SLOT_SEGMENT_BYTES / 4u];

        if (pn_tt_read_dram(device, moe->slot_bank, moe->slot_address, slot, sizeof(slot)) != 0)
            return -1;
        for (uint32_t r = 0; r < moe->shape.k; r++)
            slot[r] = (slot[r] + 1u) % moe->shape.experts;
        if (pn_tt_write_dram(device, moe->slot_bank, moe->slot_address, slot, sizeof(slot)) != 0)
            return -1;
    }
    moe->routes++;
    moe->route_seconds += pn_tt_now_seconds() - started;
    return 0;
}

/** Start a phase: the pull is carried only if every member gets work (`work[m]` > 0). */
static uint32_t moe_begin_phase(struct pn_tt_moe *moe, const uint32_t *work)
{
    moe->phase_pulled = moe->pull;
    for (uint32_t m = 0; m < moe->member_count && moe->phase_pulled; m++)
        if (work[m] == 0u)
            moe->phase_pulled = false;
    return moe_next_sequence(moe);
}

int pn_tt_moe_gate_up(pn_tt_device *device, const pn_tt_moe_stack *gate,
                      const pn_tt_moe_stack *up, const pn_tt_activation *normed,
                      const pn_tt_activation *gate_out, const pn_tt_activation *up_out)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    uint32_t words[PN_TT_MOE_MAX_MEMBERS][PN_TT_MOE_PHASE_MAX][FT_TT_QUEUE_PARAM_WORDS];
    uint32_t count[PN_TT_MOE_MAX_MEMBERS];
    uint32_t cells, columns, m, sequence;

    if (!moe || !gate || !up || !normed || !gate_out || !up_out)
        return -1;
    columns = moe->shape.expert_tiles;
    cells = moe->shape.k * columns;
    if (gate->tile_cols != columns || up->tile_cols != columns ||
        gate->tile_rows != normed->tile_count || up->tile_rows != normed->tile_count ||
        gate->stride != moe->shape.stride || up->stride != moe->shape.stride ||
        gate_out->tile_count < cells || up_out->tile_count < cells)
        return -1;
    for (m = 0; m < moe->member_count; m++) {
        const uint32_t first = (uint32_t)((uint64_t)cells * m / moe->member_count);
        const uint32_t end = (uint32_t)((uint64_t)cells * (m + 1u) / moe->member_count);
        uint16_t ax, ay;
        uint32_t a, cell = first;

        count[m] = 0u;
        moe_operand(device, normed, m, &ax, &ay, &a);
        while (cell < end) {
            const uint32_t rank = cell / columns;
            const uint32_t column = cell % columns;
            const uint32_t run = (end - cell) < (columns - column) ? (end - cell) : (columns - column);
            const uint32_t offset = column * gate->tile_rows * PN_TT_MOE_BFP6_TILE;
            const uint32_t c = (rank * columns + column) * PN_TT_MOE_BF16_TILE;

            if (count[m] + 2u > PN_TT_MOE_PHASE_MAX)
                return -1;
            moe_words(words[m][count[m]++], gate->tile_rows, run, ax, ay, a, gate->x, gate->y,
                      gate->address + offset, (uint16_t)device->dram[gate_out->bank].x,
                      (uint16_t)device->dram[gate_out->bank].y, gate_out->address + c,
                      ftmoe_flags(rank));
            moe_words(words[m][count[m]++], up->tile_rows, run, ax, ay, a, up->x, up->y,
                      up->address + offset, (uint16_t)device->dram[up_out->bank].x,
                      (uint16_t)device->dram[up_out->bank].y, up_out->address + c,
                      ftmoe_flags(rank));
            cell += run;
        }
    }
    sequence = moe_begin_phase(moe, count);
    moe->phase_kind = 1u;
    for (m = 0; m < moe->member_count; m++)
        if (moe_issue_phase(device, moe, m, words[m], count[m], sequence) != 0)
            return -1;
    moe->issue_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_moe_down(pn_tt_device *device, const pn_tt_moe_stack *down,
                   const pn_tt_activation *scaled, const pn_tt_activation *out)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    uint32_t words[PN_TT_MOE_MAX_MEMBERS][1][FT_TT_QUEUE_PARAM_WORDS];
    uint32_t count[PN_TT_MOE_MAX_MEMBERS];
    uint32_t columns, m, sequence;

    if (!moe || !down || !scaled || !out)
        return -1;
    columns = down->tile_cols;
    if (down->tile_rows != moe->shape.expert_tiles || down->stride != moe->shape.stride ||
        scaled->tile_count < moe->shape.k * moe->shape.expert_tiles ||
        out->tile_count < columns || moe->shape.k > FT_TT_MOE_K_EXPERTS_MAX)
        return -1;
    for (m = 0; m < moe->member_count; m++) {
        const uint32_t first = (uint32_t)((uint64_t)columns * m / moe->member_count);
        const uint32_t end = (uint32_t)((uint64_t)columns * (m + 1u) / moe->member_count);
        uint16_t ax, ay;
        uint32_t a;

        count[m] = end > first ? 1u : 0u;
        if (!count[m])
            continue;
        moe_operand(device, scaled, m, &ax, &ay, &a);
        moe_words(words[m][0], moe->shape.k * down->tile_rows, end - first, ax, ay, a, down->x,
                  down->y, down->address + first * down->tile_rows * PN_TT_MOE_BFP6_TILE,
                  (uint16_t)device->dram[out->bank].x, (uint16_t)device->dram[out->bank].y,
                  out->address + first * PN_TT_MOE_BF16_TILE,
                  ftmoe_flags(0u) | FT_TT_MOE_FLAG_K_EXPERTS);
    }
    sequence = moe_begin_phase(moe, count);
    moe->phase_kind = 2u;
    for (m = 0; m < moe->member_count; m++)
        if (moe_issue_phase(device, moe, m, words[m], count[m], sequence) != 0)
            return -1;
    moe->issue_seconds += pn_tt_now_seconds() - started;
    return 0;
}

/* ============================================================
 * Batched prefill (plan S1): ROWS-mode routing, LIST-mode experts
 * ============================================================ */

/** Word 15 with `bands` as the row block (resident_matmul_block_protocol.h: high half). */
static uint32_t moe_row_block_word(uint32_t bands)
{
    return bands << 16;
}

/** One ROWS-mode router descriptor: `a4` = first band | CONCAT rows << 16. */
static int moe_route_rows_a4(pn_tt_device *device, const pn_tt_moe_router *router,
                             const pn_tt_activation *normed, uint32_t bands, uint32_t a4);

int pn_tt_moe_route_rows(pn_tt_device *device, const pn_tt_moe_router *router,
                         const pn_tt_activation *normed, uint32_t bands, uint32_t first_band)
{
    if (bands == 0u || bands > FT_TT_MOE_ROWS_BANDS_MAX ||
        first_band + bands > FT_TT_MOE_SHEXP_BANDS_MAX)
        return -1;
    return moe_route_rows_a4(device, router, normed, bands, ftmoe_route_a4(first_band, 0u));
}

int pn_tt_moe_route_concat(pn_tt_device *device, const pn_tt_moe_router *router,
                           const pn_tt_activation *normed, uint32_t rows)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;

    if (!moe || rows == 0u || rows > FT_TT_MOE_CONCAT_ROWS_MAX ||
        rows * moe->shape.k > FT_TT_MOE_LIST_MAX)
        return -1;
    return moe_route_rows_a4(device, router, normed, 1u, ftmoe_route_a4(0u, rows));
}

static int moe_route_rows_a4(pn_tt_device *device, const pn_tt_moe_router *router,
                             const pn_tt_activation *normed, uint32_t bands, uint32_t a4)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    const uint32_t first_band = a4 & 0xFFFFu;
    uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
    uint16_t ax, ay;
    uint32_t a;

    if (!moe || !router || !normed || normed->tile_count != router->tile_rows)
        return -1;
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    moe_operand(device, normed, 0u, &ax, &ay, &a);
    moe_words(words, router->tile_rows, router->tile_cols, ax, ay, a, router->x, router->y,
              router->address, (uint16_t)device->dram[moe->logits.bank].x,
              (uint16_t)device->dram[moe->logits.bank].y, moe->logits.address,
              FT_TT_MOE_FLAG_ROUTE_ROWS);
    words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = bands;
    if (!moe->router.running && moe_rearm(device, moe, &moe->router) != 0)
        return -1;
    moe->router.param_slot = 0u;
    /* Idle-retired inside the enqueue (the router idles through a prompt's first layers),
     * or exited on its own idle bound: one fresh residency, one retry — the decode route's
     * rule. */
    if ((pn_tt_residency_enqueue_words(&moe->router, FT_TT_QUEUE_OP_MATMUL, words, 0u, 0u, 0u,
                                       a4, 0u) != 0 ||
         pn_tt_residency_await(&moe->router, PN_TT_OP_TIMEOUT_MS) != 0) &&
        (moe_rearm(device, moe, &moe->router) != 0 ||
         pn_tt_residency_enqueue_words(&moe->router, FT_TT_QUEUE_OP_MATMUL, words, 0u, 0u, 0u,
                                       a4, 0u) != 0 ||
         pn_tt_residency_await(&moe->router, PN_TT_OP_TIMEOUT_MS) != 0)) {
        uint32_t error = 0u, state = 0u;

        (void)ft_tt_queue_read_field(&device->runtime.noc, &moe->router.queue, FT_TT_QUEUE_FIELD_ERROR,
                                     &error);
        (void)ft_tt_queue_read_field(&device->runtime.noc, &moe->router.queue, FT_TT_QUEUE_FIELD_STATE,
                                     &state);
        fprintf(stderr, "pn_tt: the MoE rows-mode router descriptor failed (%u bands from band %u, "
                        "concat rows %u): error %u (%s), state %s\n", bands, first_band, a4 >> 16, error,
                ft_tt_queue_kernel_error_string(error), ft_tt_queue_state_string(state));
        return -1;
    }
    /* RED moe_list_shift: every list entry moves to the next expert — a wrong union reaching
     * the expert readers, written where the router wrote it. */
    if (moe->red_list_shift) {
        uint32_t slot[FT_TT_MOE_SLOT_COPY_BYTES / 4u];

        if (pn_tt_read_dram(device, moe->slot_bank, moe->slot_address, slot, sizeof(slot)) != 0)
            return -1;
        for (uint32_t i = 0; i < slot[FT_TT_MOE_LIST_COUNT_ENTRY] && i < FT_TT_MOE_LIST_MAX; i++)
            slot[i] = (slot[i] + 1u) % moe->shape.experts;
        if (pn_tt_write_dram(device, moe->slot_bank, moe->slot_address, slot, sizeof(slot)) != 0)
            return -1;
    }
    moe->routes++;
    moe->route_seconds += pn_tt_now_seconds() - started;
    return 0;
}

/** Verify-shaped rows descriptors (one band, at most 4 rows' experts) repeat every cycle with
 *  the same words: they take the parameter cache. A prefill group's words move with its band,
 *  and the cache is first come and never evicts, so those keep the rotating slot and leave the
 *  cache to decode (whose dispatch replay needs its own blocks cached, plan S2). */
#define PN_TT_MOE_ROWS_CACHE_ENTRIES 32u

/** Issue member `m`'s descriptor of a prefill / verify phase (the pull on the last). */
static int moe_issue_rows(pn_tt_device *device, struct pn_tt_moe *moe, uint32_t m,
                          const uint32_t *words, uint32_t a1, uint32_t a2, bool last,
                          uint32_t sequence, bool cacheable)
{
    pn_tt_residency *r = &moe->member[m];
    const bool pull = last && m == 0u && moe->phase_pulled;
    int rc;

    if (moe->member_issued[m] == 0u && !r->running && moe_rearm(device, moe, r) != 0)
        return -1;
    if (moe->member_issued[m] + 1u > r->queue.capacity)
        return -1;
    if (cacheable) {
        const uint32_t a3 = pull ? moe->member_count | (moe->red_pull_skip ? PN_TT_MOE_PULL_RED_SKIP : 0u) : 0u;
        const uint32_t a4 = pull ? sequence : 0u, a5 = last && moe->phase_pulled ? sequence : 0u;
        uint32_t address = moe_cache_address(moe, m, words);

        if (address != 0u) {
            rc = pn_tt_residency_enqueue_cached(r, FT_TT_QUEUE_OP_MATMUL, address, a1, a2, a3, a4, a5);
            if (rc != 0 && moe->member_issued[m] == 0u && !r->running && moe_rearm(device, moe, r) == 0) {
                address = moe_cache_address(moe, m, words);
                rc = address ? pn_tt_residency_enqueue_cached(r, FT_TT_QUEUE_OP_MATMUL, address, a1, a2,
                                                              a3, a4, a5)
                             : -1;
            }
            if (rc != 0)
                return -1;
            moe->member_issued[m]++;
            moe->expert_descriptors++;
            return 0;
        }
    }
    r->param_slot = moe->member_issued[m];
    rc = pn_tt_residency_enqueue_words(
        r, FT_TT_QUEUE_OP_MATMUL, words, a1, a2,
        pull ? moe->member_count | (moe->red_pull_skip ? PN_TT_MOE_PULL_RED_SKIP : 0u) : 0u,
        pull ? sequence : 0u, last && moe->phase_pulled ? sequence : 0u);
    /* The idle retire fires INSIDE the enqueue: re-arm once and retry (moe_issue's rule). */
    if (rc != 0 && moe->member_issued[m] == 0u && !r->running && moe_rearm(device, moe, r) == 0) {
        r->param_slot = 0u;
        rc = pn_tt_residency_enqueue_words(
            r, FT_TT_QUEUE_OP_MATMUL, words, a1, a2,
            pull ? moe->member_count | (moe->red_pull_skip ? PN_TT_MOE_PULL_RED_SKIP : 0u) : 0u,
            pull ? sequence : 0u, last && moe->phase_pulled ? sequence : 0u);
    }
    r->param_slot = 0u;
    if (rc != 0)
        return -1;
    moe->member_issued[m]++;
    moe->expert_descriptors++;
    return 0;
}

int pn_tt_moe_gate_up_rows(pn_tt_device *device, const pn_tt_moe_stack *gate,
                           const pn_tt_moe_stack *up, const pn_tt_activation *normed,
                           uint32_t bands, uint32_t entries, const pn_tt_activation *gate_out,
                           const pn_tt_activation *up_out)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    uint32_t work[PN_TT_MOE_MAX_MEMBERS];
    uint32_t cells, columns, sequence;

    if (!moe || !gate || !up || !normed || !gate_out || !up_out || bands == 0u ||
        bands > FT_TT_MOE_ROWS_BANDS_MAX || bands == 3u || entries == 0u ||
        entries > FT_TT_MOE_LIST_MAX)
        return -1;
    columns = moe->shape.expert_tiles;
    cells = entries * columns;
    const bool cacheable = bands == 1u && entries <= PN_TT_MOE_ROWS_CACHE_ENTRIES;
    if (gate->tile_cols != columns || up->tile_cols != columns ||
        gate->tile_rows != normed->tile_count || up->tile_rows != normed->tile_count ||
        gate_out->tile_count < cells || up_out->tile_count < cells)
        return -1;
    for (uint32_t m = 0; m < moe->member_count; m++)
        work[m] = (uint32_t)((uint64_t)cells * (m + 1u) / moe->member_count) -
                  (uint32_t)((uint64_t)cells * m / moe->member_count);
    sequence = moe_begin_phase(moe, work);
    for (uint32_t m = 0; m < moe->member_count; m++) {
        const uint32_t first = (uint32_t)((uint64_t)cells * m / moe->member_count);
        const uint32_t a1 = ftmoe_list_a1(first / columns, first % columns, columns);
        uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
        uint16_t ax, ay;
        uint32_t a;

        if (work[m] == 0u)
            continue;
        moe_operand(device, normed, m, &ax, &ay, &a);
        moe_words(words, gate->tile_rows, work[m], ax, ay, a, gate->x, gate->y, gate->address,
                  (uint16_t)device->dram[gate_out->bank].x, (uint16_t)device->dram[gate_out->bank].y,
                  gate_out->address + first * PN_TT_MOE_BF16_TILE, 0u);
        words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = bands;
        words[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] = moe_row_block_word(bands);
        if (moe_issue_rows(device, moe, m, words, a1, gate_out->tile_count, false, sequence,
                           cacheable) != 0)
            return -1;
        moe_words(words, up->tile_rows, work[m], ax, ay, a, up->x, up->y, up->address,
                  (uint16_t)device->dram[up_out->bank].x, (uint16_t)device->dram[up_out->bank].y,
                  up_out->address + first * PN_TT_MOE_BF16_TILE, 0u);
        words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = bands;
        words[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] = moe_row_block_word(bands);
        if (moe_issue_rows(device, moe, m, words, a1, up_out->tile_count, true, sequence,
                           cacheable) != 0)
            return -1;
    }
    moe->issue_seconds += pn_tt_now_seconds() - started;
    return 0;
}

int pn_tt_moe_down_rows(pn_tt_device *device, const pn_tt_moe_stack *down,
                        const pn_tt_activation *scaled, uint32_t bands, uint32_t entries,
                        const pn_tt_activation *out)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    uint32_t work[PN_TT_MOE_MAX_MEMBERS];
    uint32_t columns, sequence;
    const uint32_t a1 = ftmoe_list_a1(0u, 0u, moe ? moe->shape.expert_tiles : 0u);

    if (!moe || !down || !scaled || !out || bands == 0u || bands > FT_TT_MOE_ROWS_BANDS_MAX ||
        bands == 3u || entries == 0u || entries > FT_TT_MOE_LIST_MAX)
        return -1;
    columns = down->tile_cols;
    if (down->tile_rows != moe->shape.expert_tiles ||
        scaled->tile_count != entries * moe->shape.expert_tiles || out->tile_count < columns)
        return -1;
    const bool cacheable = bands == 1u && entries <= PN_TT_MOE_ROWS_CACHE_ENTRIES;
    for (uint32_t m = 0; m < moe->member_count; m++)
        work[m] = (uint32_t)((uint64_t)columns * (m + 1u) / moe->member_count) -
                  (uint32_t)((uint64_t)columns * m / moe->member_count);
    sequence = moe_begin_phase(moe, work);
    for (uint32_t m = 0; m < moe->member_count; m++) {
        const uint32_t first = (uint32_t)((uint64_t)columns * m / moe->member_count);
        uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
        uint16_t ax, ay;
        uint32_t a;

        if (work[m] == 0u)
            continue;
        moe_operand(device, scaled, m, &ax, &ay, &a);
        moe_words(words, entries * down->tile_rows, work[m], ax, ay, a, down->x,
                  down->y, down->address + first * down->tile_rows * PN_TT_MOE_BFP6_TILE,
                  (uint16_t)device->dram[out->bank].x, (uint16_t)device->dram[out->bank].y,
                  out->address + first * PN_TT_MOE_BF16_TILE, FT_TT_MOE_FLAG_K_EXPERTS);
        words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = bands;
        words[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] = moe_row_block_word(bands);
        if (moe_issue_rows(device, moe, m, words, a1, out->tile_count, true, sequence,
                           cacheable) != 0)
            return -1;
    }
    moe->issue_seconds += pn_tt_now_seconds() - started;
    return 0;
}

const pn_tt_activation *pn_tt_moe_scale_buffer(const pn_tt_device *device)
{
    return pn_tt_moe_armed(device) ? &device->moe->scale_buffer : NULL;
}

/** After a failed await: every member whose kernel reports an error, with its state and
 *  counts (diagnostics only; reads the control words of each queue). */
static void moe_explain_failure(pn_tt_device *device, struct pn_tt_moe *moe)
{
    for (uint32_t m = 0; m < moe->member_count; m++) {
        uint32_t error = 0u, state = 0u, completed = 0u, tail = 0u;
        pn_tt_residency *r = &moe->member[m];

        (void)ft_tt_queue_read_field(&device->runtime.noc, &r->queue, FT_TT_QUEUE_FIELD_ERROR, &error);
        (void)ft_tt_queue_read_field(&device->runtime.noc, &r->queue, FT_TT_QUEUE_FIELD_STATE, &state);
        (void)ft_tt_queue_read_field(&device->runtime.noc, &r->queue, FT_TT_QUEUE_FIELD_COMPLETED_COUNT,
                                     &completed);
        (void)ft_tt_queue_read_field(&device->runtime.noc, &r->queue, FT_TT_QUEUE_FIELD_TAIL, &tail);
        uint32_t why[4] = { 0u, 0u, 0u, 0u };

        (void)device->runtime.noc.read(device->runtime.noc.user, r->core_x, r->core_y,
                                       FT_TT_OP_QUEUE_BASE + FT_TT_MOE_MAILBOX_OFFSET + 32u, why,
                                       sizeof(why));
        if (error != 0u || completed != r->queue.issued)
            fprintf(stderr, "pn_tt: MoE member %u (core %u): error %u (%s), state %s, completed %u "
                            "tail %u of %u issued; reader refusal reason %u (%u, %u) x%u\n", m,
                    r->core_index, error, ft_tt_queue_kernel_error_string(error),
                    ft_tt_queue_state_string(state), completed, tail, r->queue.issued, why[0],
                    why[1], why[2], why[3]);
    }
}

int pn_tt_moe_await(pn_tt_device *device)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    const double started = pn_tt_now_seconds();
    int failed = 0;

    if (!moe)
        return -1;
    /* PULL: member 0 posted only after every member's done-sequence reached this phase's, and
     * every member runs its descriptors in order, so member 0's count covers them all. */
    if (moe->phase_pulled && moe->member_issued[0] != 0u) {
        if (pn_tt_residency_await(&moe->member[0], PN_TT_OP_TIMEOUT_MS) != 0) {
            fprintf(stderr, "pn_tt: MoE expert residency on core %u (the pull leader) failed\n",
                    moe->member[0].core_index);
            moe_explain_failure(device, moe);
            failed = 1;
        }
        for (uint32_t m = 0; m < moe->member_count; m++) {
            /* `PN_TT_MOE_PULL_VERIFY=1` (diagnostic): read every member's completion count
             * right after the leader posted; a member still short of what it was issued
             * means the leader posted early (the pull did not hold it). Then await it. */
            if (!failed && m != 0u && moe->member_issued[m] != 0u && moe->pull_verify) {
                uint32_t completed = 0u;

                (void)ft_tt_queue_read_field(&device->runtime.noc, &moe->member[m].queue,
                                             FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &completed);
                moe->pull_checked++;
                if (completed != moe->member[m].queue.issued)
                    moe->pull_early++;
                if (pn_tt_residency_await(&moe->member[m], PN_TT_OP_TIMEOUT_MS) != 0)
                    failed = 1;
            } else if (!failed && m != 0u && moe->member_issued[m] != 0u)
                pn_tt_residency_mark_settled(&moe->member[m]);
            moe->member_issued[m] = 0u;
        }
        /* `PN_TT_MOE_PULL_DELAY_US=N` (diagnostic): the host waits N us after the leader
         * posted (`PN_TT_MOE_PULL_DELAY_PHASE` 1 = gate/up phases only, 2 = down only). */
        if (moe->pull_delay_us &&
            (moe->pull_delay_phase == 0u || moe->pull_delay_phase == moe->phase_kind)) {
            const double until = pn_tt_now_seconds() + moe->pull_delay_us * 1e-6;

            while (pn_tt_now_seconds() < until) {
            }
        }
        moe->phase_pulled = false;
        moe->awaits++;
        moe->await_seconds += pn_tt_now_seconds() - started;
        return failed ? -1 : 0;
    }
    for (uint32_t m = 0; m < moe->member_count; m++) {
        if (moe->member_issued[m] == 0u)
            continue;
        if (pn_tt_residency_await(&moe->member[m], PN_TT_OP_TIMEOUT_MS) != 0) {
            fprintf(stderr, "pn_tt: MoE expert residency on core %u failed\n",
                    moe->member[m].core_index);
            failed = 1;
        }
        moe->member_issued[m] = 0u;
    }
    moe->awaits++;
    moe->await_seconds += pn_tt_now_seconds() - started;
    return failed ? -1 : 0;
}

int pn_tt_moe_read_slot(pn_tt_device *device, uint32_t *out_index, float *out_weight)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;
    uint32_t slot[2u * PN_TT_MOE_MAX_K];

    if (!moe || !out_index || !out_weight ||
        pn_tt_read_dram(device, moe->slot_bank, moe->slot_address, slot, sizeof(slot)) != 0)
        return -1;
    for (uint32_t r = 0; r < moe->shape.k; r++) {
        out_index[r] = slot[r];
        memcpy(&out_weight[r], &slot[FT_TT_MOE_SLOT_WEIGHT0 + r], sizeof(float));
    }
    return 0;
}

int pn_tt_moe_read_logits(pn_tt_device *device, float *out, uint32_t count)
{
    struct pn_tt_moe *moe = device ? device->moe : NULL;

    if (!moe || !out || count != moe->logits.tile_count * 32u)
        return -1;
    return pn_tt_fp32_strip_download(device, &moe->logits, out, count);
}

void pn_tt_moe_report(const pn_tt_device *device, double tokens)
{
    const struct pn_tt_moe *moe = device ? device->moe : NULL;

    if (!moe || tokens <= 0.0)
        return;
    fprintf(stderr, "pn_tt: MoE — %llu routes, %llu expert descriptors over %u members, %llu "
                    "re-arms; per token: route %.2f ms, expert issue %.2f ms, expert await "
                    "%.2f ms; parameter cache %llu hits / %llu misses / %llu full\n",
            (unsigned long long)moe->routes, (unsigned long long)moe->expert_descriptors,
            moe->member_count, (unsigned long long)moe->rearms, moe->route_seconds * 1000.0 / tokens,
            moe->issue_seconds * 1000.0 / tokens, moe->await_seconds * 1000.0 / tokens,
            (unsigned long long)moe->cache_hits, (unsigned long long)moe->cache_misses,
            (unsigned long long)moe->cache_full);
    if (moe->pull_verify)
        fprintf(stderr, "pn_tt: MoE PULL VERIFY — %llu member completions read right after the "
                        "leader posted, %llu of them EARLY\n",
                (unsigned long long)moe->pull_checked, (unsigned long long)moe->pull_early);
}

#endif /* USE_TT */
