/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_device_internal.h
 * @brief The layout of `struct pn_tt_device`, shared by the two source files that own it.
 *
 *        pn_tt_device.c owns the card, the artifact residency and the matmul; pn_tt_oncard.c
 *        owns the reduce/elementwise/sampling residencies and the DRAM activation buffers
 *        they pass to each other. They are one object and two files because they are two
 *        subjects, and the struct is here rather than in the public header so a caller
 *        still cannot reach into it.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_DEVICE_INTERNAL_H
#define PN_TT_DEVICE_INTERNAL_H

/** Mask tiles for verify attention in ROWS mode: up to 4 x (32 / q_rows) rows. */
#define PN_TT_ATTN_ROWS_CHUNKS 4u

#ifdef USE_TT

#include <stdio.h>

#include "pn_tt_descriptor_census.h"
#include "pn_tt_device.h"
#include "pn_tt_eltwise_split.h"
#include "pn_tt_fp32_strip.h"
#include "pn_tt_gdn.h"
#include "pn_tt_group_pull.h"
#include "pn_tt_kv_format.h"
#include "pn_tt_matmul_group.h"
#include "pn_tt_norm_schedule.h"
#include "pn_tt_attn_deep.h"
#include "pn_tt_lend_ledger.h"
#include "pn_tt_residency.h"
#include "pn_tt_swiglu_fuse.h"
#include "pn_tt_resadd_fuse.h"

#include "tt/ft_tt_buffer.h"
#include "tt/ft_tt_fttw.h"
#include "tt/ft_tt_op_memory.h"
#include "tt/ft_tt_op_vector.h"
#include "tt/ft_tt_queue.h"
#include "tt/ft_tt_runtime.h"
#include "pn_tt_stage_pipeline.h"
#include "pn_tt_weight_split.h"

/** Staging slots for PHASE DELTA — one per 16-row sub-chunk of a 32-row pass. */
#define PN_TT_GDNC_STAGE_SLOTS 2u

/** Most cores the prefill attention chunk mixer may split its Q heads across (16 = one Q
 *  head a core for Ornith, the default since 2026-10-26; `PN_TT_ATTN_CORES`). */
#define PN_TT_ATTN_MAX_MEMBERS 16u
/** Attention members C may borrow onto lent grid cores (the grid's cap, = PN_TT_ATTN_GRID_MAX_CORES). */
#define PN_TT_ATTN_GRID_MAX_MEMBERS 64u
/** Split-K decode attention's split-only cores past the head-split members
 *  (`PN_TT_ATTN_SPLITK_CORE_LIST`, pn_tt_attention_splitk.h): at most 64 split cores. */
#define PN_TT_ATTN_SPLITK_MAX_EXTRA 48u

/** Most cores one per-head norm width may split its heads across. */
#define PN_TT_HEAD_NORM_MAX_MEMBERS 8u

/** Most cores a prefill KV append pair may pack on (core 83 + 7; pn_tt_kv_append_split.h). */
#define PN_TT_KV_APPEND_MAX_MEMBERS 8u

/** Bounds every wait. There is no untimed path to the card (spec §7.3). */
#define PN_TT_OP_TIMEOUT_MS 60000u

#define PN_TT_TILE_DIM   FT_TT_OP_TILE_DIM
#define PN_TT_TILE_BF16  FT_TT_OP_TILE_BYTES_BF16
#define PN_TT_TILE_FP32  FT_TT_OP_TILE_BYTES_FP32

/** THE CEILING IS THE DST SECTION, NOT L1. A band only pays if `MB` grows with it, and
 *  `MB * NB <= 8` (resident_matmul_block_protocol.h) makes MB=2/NB=4 the only point on the
 *  lattice that beats the shipped MB=1/NB=8 — so a third band buys nothing the second did not
 *  and costs another pass of both chunk mixers. 4 is slack for the arm that proves that. */
#define PN_TT_STRIP_MAX_TILE_ROWS 64u

/** How many distinct elementwise tile counts ONE arming may be asked to serve. The forward
 *  asks for two (n_embd and n_ff) and the slack is so a third width needs no header change.
 *  The shapes are kept rather than consumed because a residency is BOUNDED — an idle kernel
 *  retires itself — and re-arming one needs the shapes it was armed with. */
#define PN_TT_ONCARD_MAX_ELTWISE_SHAPES 8u

/** How many displaced KV sequence keys are remembered, i.e. how deep an interleave this
 *  detects. A key that comes back after being displaced is two sequences alternating on ONE
 *  region set; beyond this depth the detector goes blind, which is why it is a stated bound
 *  and not a guarantee. */
#define PN_TT_KV_DISPLACED_KEYS 8u

/** One resident weight plus the name it answers to. `part` is -1 for the whole tensor, or the
 *  column part a load-time split rule uploaded instead of it (pn_tt_weight_split.h). */
typedef struct {
    char name[FT_TT_ARTIFACT_NAME_BYTES];
    int32_t part;
    pn_tt_weight weight;
} pn_tt_resident;

/** One fp32 buffer of the GDN family in card DRAM. A bank and an offset, which is everything
 *  `ft_tt_op_operand` needs — the tile size is the family's, not the buffer's, because every
 *  circular buffer of the `gdn` program is Float32. */
/** PM 7 ITEM 5: most GDN GATE members one device arms. */
#define PN_TT_GDNGATE_MAX_MEMBERS 8u

typedef struct pn_tt_gdn_buffer {
    uint32_t bank;
    uint32_t address;
} pn_tt_gdn_buffer;

/** GDN residencies one head-split may hold, member 0 included.
 *
 *  EIGHT because that is where the measurement stops, not where the card does: the core
 *  scaling probe reached 7.12x on 8 and the next doubling was never taken. Raising it is a
 *  measurement, not an edit — every extra member is a Tensix `pn_tt_prepare_family_on_core()`
 *  has to be able to refuse against the matmul group by name. */
/** L3: most top-k split members, member 0 included. */
#define PN_TT_TOPK_MAX_MEMBERS 8u
#define PN_TT_GDN_MAX_MEMBERS 16u

/** Most RMS-norm (reduce) column-split members, member 0 included. The card's slice code is a
 *  4-bit member count (resident_reduce_protocol.h ftr_slice), so 15 is its ceiling; 8 is the
 *  widest that divides Qwen3.8's 160 tile columns into equal slices after 5 and 4. */
#define PN_TT_REDUCE_MAX_MEMBERS 8u

/** CHUNKWISE residencies one core split may hold, member 0 included.
 *
 *  SIXTEEN rather than the decode family's eight because the two phases divide by different
 *  quantities and the smaller one still admits it: PHASE CONV has **256 tile columns** at
 *  Ornith's conv_dim of 8192 and PHASE DELTA has **32 v-heads**, so 16 members own 16 columns
 *  or 2 heads each and both divide exactly. It is a ceiling on the LADDER, not a claim about
 *  the knee — the decode family's knee was 4 with 8 reproducibly behind it. Every extra member
 *  is a Tensix `pn_tt_prepare_family_on_core()` has to be able to refuse by name. */
#define PN_TT_GDNC_MAX_MEMBERS 16u

/** One SSM layer's GDN buffers. `weight` is a constant staged once; the other two are the
 *  carried state, READ AND WRITTEN by their descriptors and never downloaded. */
typedef struct pn_tt_gdn_region {
    pn_tt_gdn_buffer weight; /**< taps TAP-MAJOR planes, transposed at load */
    pn_tt_gdn_buffer window; /**< taps-1 planes, the same layout, rolled on the card */
    pn_tt_gdn_buffer state;  /**< [v_heads, S, S] fp32, oracle-transposed */
} pn_tt_gdn_region;

/** One layer's K and V regions in card DRAM. Identical geometry and two separate bases —
 *  contract F1 and N4 (two regions rather than one with a fixed offset, so each is a plain
 *  ft_tt_op_operand). Internal because no public call takes or returns one. */
/** Most KV head GROUPS a layer's K (and V) region may be split into, each its own bank
 *  (pn_tt_kv_banks.h). */
#define PN_TT_KV_MAX_BANK_GROUPS 8u

typedef struct pn_tt_kv_region {
    uint32_t k_bank, k_address;
    uint32_t v_bank, v_address;
    /* KV BANK GROUPS (pn_tt_kv_banks.h): group g's heads at [g] (group 0 = the fields above). */
    uint32_t k_group_bank[PN_TT_KV_MAX_BANK_GROUPS], k_group_address[PN_TT_KV_MAX_BANK_GROUPS];
    uint32_t v_group_bank[PN_TT_KV_MAX_BANK_GROUPS], v_group_address[PN_TT_KV_MAX_BANK_GROUPS];
} pn_tt_kv_region;

/** Distinct HEAD widths one model may need per-head reduces at.
 *
 *  TWO, because Ornith needs exactly two and they are not the same number: its
 *  Gated-Delta-Net per-head norms are over S = 128 (Wt 4) and its attention QK-RMSNorms are
 *  over head_dim = 256 (Wt 8). A reduce residency is PLACED for one Wt and refuses any
 *  other, so these cannot share a core — see pn_tt_head_norm.h. Raising this is one more
 *  Tensix, which the card has; it is bounded so the collision refusal has a fixed set to
 *  check. */
#define PN_TT_HEAD_NORM_MAX_WIDTHS 2u

/** One per-head reduce residency, the ONE width it was placed for, and the ONE page size —
 *  which is also which reduce PROGRAM it runs, since the two are the same choice. */
typedef struct pn_tt_head_norm_residency {
    pn_tt_residency residency;
    uint32_t tile_cols;  /**< width/32; 0 for a slot that never armed */
    uint32_t tile_bytes; /**< PN_TT_TILE_FP32 (`reduce_fp32`) or PN_TT_TILE_BF16 (`reduce`) */
    /** THE HEAD SPLIT (`PN_TT_HEAD_NORM_CORES`): members 1..N-1 run the same program and take
     *  disjoint HEAD ranges. Heads are independent row blocks and the wire carries no absolute
     *  block index, so a member is the same descriptor at an offset base — host-only. */
    pn_tt_residency member[PN_TT_HEAD_NORM_MAX_MEMBERS - 1u];
    uint32_t members; /**< 1 = no split */
    pn_tt_group_pull pull; /**< S1: member 0 posts for the split (pn_tt_group_pull.h) */
    /** P3: fixed at arm time with the placement it needs (pn_tt_norm_schedule.h). */
    pn_tt_norm_schedule schedule;
} pn_tt_head_norm_residency;

/** The two directions of the activation strip bridge, in the order `render[]` holds them. */
enum { PN_TT_RENDER_WIDEN = 0, PN_TT_RENDER_NARROW = 1, PN_TT_RENDER_DIRECTIONS = 2 };

/** Residencies ONE DIRECTION of the bridge may hold, member 0 included.
 *
 *  EIGHT and not sixteen, because the two directions are split independently and both sets
 *  have to fit above every core any other family can reach: at eight apiece they occupy
 *  122..128 and 129..135 of the 122..139 the elementwise split's map leaves free. Eight also
 *  divides both of Ornith's render widths (256 widen, 128 narrow) and leaves the narrow slice
 *  at 16 tiles, still two blocks of depth 8. Raising it is more Tensix and a re-check of that
 *  map, not an edit here. */
#define PN_TT_RENDER_SPLIT_MAX_MEMBERS 8u

/** One direction of the bridge: a residency, the two page sizes it converts between, and the
 *  ZERO operand that turns `kVecFnAdd` into a copy. See pn_tt_render.h. */
typedef struct pn_tt_render_residency {
    pn_tt_residency residency;
    /** Members 1..N-1 of the split; member 0 is `residency` above. See pn_tt_render.h. */
    pn_tt_residency member[PN_TT_RENDER_SPLIT_MAX_MEMBERS - 1u];
    uint32_t members; /**< live residencies, member 0 included; 1 when unsplit */
    /** Per-member AWAIT seconds. Kept apart from the census, which keys on the WHOLE
     *  descriptor's width so its `fixed + per_tile * tiles` fit stays comparable across a
     *  member ladder — what that cannot show is the SHAPE of the set, and "it stopped at N"
     *  is exactly the finding that needs it. */
    double member_seconds[PN_TT_RENDER_SPLIT_MAX_MEMBERS];
    pn_tt_group_pull pull; /**< S1: member 0 posts for the split (pn_tt_group_pull.h) */
    uint32_t in_tile_bytes;
    uint32_t out_tile_bytes;
    uint32_t tiles; /**< the widest strip placed for; 0 when this direction never armed */
    uint32_t depth; /**< the ONE block depth every descriptor of it carries */
    uint32_t zero_bank;
    uint32_t zero_address; /**< `tiles` input-format tiles, allocated zeroed, never written */
} pn_tt_render_residency;

/** THE MERGED GDN OUTPUT GATE — `silu(bf16 z) * fp32 y_norm -> bf16 gated` under a MIXED CB
 *  table, which is what lets `render_narrow` leave the decode token. No zero operand: this one
 *  is a real binary op, not a copy dressed as one. See pn_tt_gate.h. */
typedef struct pn_tt_gate_residency {
    pn_tt_residency residency;
    uint32_t tiles; /**< the widest gate placed for; 0 when it never armed */
    uint32_t depth; /**< the ONE block depth every descriptor of it carries */
} pn_tt_gate_residency;

struct pn_tt_device {
    ft_tt_runtime runtime;
    ft_tt_runtime_images firmware;
    ft_tt_runtime_images program_bfp8;
    ft_tt_runtime_images program_bfp4;
    bool have_bfp4;

    /* The resident triad: the same three kernels, built to not return between ops. */
    ft_tt_runtime_images program_resident;
    bool have_resident;      /**< the images loaded and the firmware matches */
    bool resident_requested; /**< PN_TT_DISPATCH did not say `oneshot` */
    bool resident_running;   /**< a residency is live on the core RIGHT NOW */
    /** Member 0's work queue. It lives here rather than in the group because the device
     *  still owns exactly one residency's lifetime; the group holds the rest. */
    ft_tt_queue queue;
    /** The cores one projection is split across. One member is the pre-group path. */
    pn_tt_matmul_group group;
    pn_tt_red_split red_split;    /**< RED PROOF: break member 1 in one named way */
    uint32_t poll_backoff_us;     /**< sleep between completion polls; 0 = tight loop */
    uint32_t resident_tile_bytes; /**< the ONE weight tile size the residency serves */
    uint32_t resident_depth;      /**< the ONE prefetch depth every descriptor carries */
    uint32_t artifact_tile_bytes; /**< the artifact's single matmul tile size, 0 if mixed */
    uint32_t artifact_kt_gcd;     /**< every matmul Kt divides by this */
    uint64_t red_bad_descriptor;  /**< RED PROOF: malform this matvec's parameters, 0 = never */
    /** RED PROOF `PN_TT_RED=fast_check`: ask the one-line completion check about one
     *  descriptor further on than the poll waited for. It must REFUSE; green would mean the
     *  fast path's check is inert and the whole member-count ladder measures nothing. */
    bool red_fast_check;
    /** RED PROOF `PN_TT_RED=param_cache_hash`: collapse the cache's hash to a constant. The
     *  memcmp still decides, so this arm must be `cmp` IDENTICAL — which is the claim that
     *  the hash is a bucket index and nothing more. */
    bool red_param_cache_hash;
    /** RED PROOF `PN_TT_RED=param_cache_blind`: take the first bucket occupant WITHOUT the
     *  memcmp. Paired with the constant hash every projection reuses block 0, so this arm
     *  must produce a `cmp` that DIFFERS. Green would mean the words never varied. */
    bool red_param_cache_blind;
    uint64_t resident_ops;        /**< matvecs executed through the queue */
    uint64_t resident_restarts;   /**< times the residency had to be re-armed */

    uint16_t core_x, core_y;
    uint32_t core_index;

    ft_tt_allocator allocator;
    ft_tt_noc_coord dram[FT_TT_MAX_DRAM_BANKS];
    uint32_t dram_bank_count;

    ft_tt_fttw artifact;
    bool artifact_open;
    /** STEP 4 S2: a second artifact (the MTP draft head) whose weights are placed on demand. */
    ft_tt_fttw extra_artifact;
    bool extra_open;
    pn_tt_resident *resident;
    uint32_t resident_count;
    uint64_t resident_bytes;

    /* ------------------------------------------------------------
     * The ON-CARD residencies — one core per op family
     * ------------------------------------------------------------
     * A resident binary is linked at a fixed slab, so one program per core. The forward
     * needs three families in sequence per layer, so it gets three cores and they pass
     * activations through DRAM instead of through the host.
     */
    pn_tt_residency reduce;  /**< RMS norm (the reduce family) */
    /** L4: the RMS-norm column split's extra members — member m owns slice m of every row's
     *  final chain and output (kRedFinal byte 3); every member reduces the whole row.
     *  `PN_TT_REDUCE_CORES` (default 2, up to PN_TT_REDUCE_MAX_MEMBERS since 2026-11-19). */
    pn_tt_residency reduce_member[PN_TT_REDUCE_MAX_MEMBERS - 1u];
    /** MEASURE ONLY (2026-10-05, PHASE G pre-work): `PN_TT_IDLE_FILL=N` arms N extra reduce
     *  residencies on cores 0..N-1 that never receive work — the price of RESIDENT-BUT-IDLE
     *  cores, which a full-grid prefill would add. */
    pn_tt_residency idle_fill[64];
    uint32_t idle_fill_count;
    uint32_t reduce_members;
    pn_tt_group_pull reduce_pull; /**< S1: `reduce` posts for the L4 split */
    /** P0: the persistent norm role (pn_tt_norm_role.h); NULL until first armed. */
    struct pn_tt_norm_role *norm_role;
    pn_tt_residency eltwise; /**< SwiGLU gate and residual add */
    pn_tt_residency sample;  /**< exact top-N over the logit row */
    pn_tt_residency rope;      /**< NeoX rotary, on q and k */
    pn_tt_residency memory;    /**< the KV append (no compute thread: DM0|DM1 only) */
    /** THE PREFILL KV APPEND SPLIT (pn_tt_kv_append_split.h): members 1..N-1 run the `memory`
     *  program on their own cores and pack one head of K or V each; decode never uses them. */
    pn_tt_residency memory_member[PN_TT_KV_APPEND_MAX_MEMBERS - 1u];
    uint32_t memory_members;   /**< 1 = no split */
    pn_tt_group_pull memory_pull; /**< S1: `memory` posts for a split append */
    /** STEP 3 (A): the on-card puller (BRISC only, `PN_TT_PULLER_CORE`, default 89). */
    pn_tt_residency puller;
    pn_tt_residency attention; /**< flash decode, one descriptor per KV group */
    /** THE CHUNK MIXER'S HEAD SPLIT (`PN_TT_ATTN_CORES`). Members 1..N-1 run the SAME program
     *  on their own cores and take disjoint Q-head ranges of a prefill chunk; decode never
     *  uses them. Every word of an attention descriptor is a DRAM address with the head folded
     *  in by the host, so the split needs no kernel change. */
    pn_tt_residency attention_member[PN_TT_ATTN_MAX_MEMBERS - 1u];
    uint32_t attention_members;  /**< 1 = no split */
    pn_tt_group_pull attention_pull; /**< S1: lane 0 posts for a decode lane round */
    double attention_member_seconds[PN_TT_ATTN_MAX_MEMBERS];
    /** Deep-queued prefill attention: the layer-pass's issue flow (pn_tt_attn_deep.h);
     *  `members == 0` between layer-passes. */
    pn_tt_attn_deep_flow attention_deep_flow;
    /** C: attention residencies borrowed onto the parked grid cores for a layer-pass
     *  (pn_tt_attention_grid.h); calloc'd on first use, `attention_grid_count` live. */
    pn_tt_residency *attention_grid;
    uint32_t attention_grid_count;
    /** L1a: the decode residencies whose cores the current prompt lent (pn_tt_prefill_lend.h). */
    pn_tt_lend_ledger prefill_lend;
    /** L1a.3: the lent GDN member whose core is the grid's 64th (UINT32_MAX core: none). */
    uint32_t prefill_lend_grid_member, prefill_lend_grid_core;
    /** SPLIT-K (pn_tt_attention_splitk.h): set at arm when `PN_TT_ATTN_SPLITK=1` found the
     *  `_split` program, which every attention core then runs. The extras are split-only
     *  cores past the members; the partial buffer holds PN_TT_ATTN_SPLITK_MAX_PARTIALS
     *  partials; the zero mask (on the constants' bank) is a non-final ROWS chunk's tail. */
    bool attn_splitk;
    /** FAST-MATH (pn_tt_attn_fast.h): the `PN_TT_ATTN_FAST` variant every attention core runs
     *  in place of the shipped program, set at arm; NULL = the shipped program. */
    const char *attn_fast_program;
    pn_tt_residency attention_splitk_extra[PN_TT_ATTN_SPLITK_MAX_EXTRA];
    uint32_t attention_splitk_extras;
    uint32_t attn_partial_bank, attn_partial_address;
    uint32_t attn_zero_mask_address;
    pn_tt_residency gdn;       /**< the Gated-Delta-Net block, two descriptors per SSM layer */
    /** THE CHUNKWISE Gated-Delta-Net block — the same two phases over C <= 32 POSITIONS in one
     *  descriptor, for prefill. It carries NO STATE OF ITS OWN: the conv window and the
     *  recurrent state it advances are `gdn_region[]`'s, at their addresses, in their layout,
     *  which is what makes the prefill->decode seam free. See pn_tt_gdnc.h. */
    pn_tt_residency gdnc;
    /** EXTRA CHUNKWISE residencies, behind `PN_TT_GDNC_CORES`. The work divides the same two
     *  ways the decode family's does and the WIRE HAS CARRIED THE SLICE SINCE THIS FAMILY
     *  SHIPPED (`ft_tt_op_gdnc_request.first_col/.col_count`, `kGdncColSlice`): PHASE CONV
     *  splits TILE COLUMNS — 256 of them at Ornith's conv_dim of 8192, and no column reads
     *  another column's operands or writes another column's tile — and PHASE DELTA splits
     *  V-HEADS, exactly as `gdn_member[]` does. Every member carries the IDENTICAL
     *  circular-buffer placement, because `resolve_cb_pages()` is sized from `state_tiles` and
     *  `conv_taps` and from neither slice. `gdnc` itself is member 0, so this array holds one
     *  fewer. See pn_tt_gdnc.c's PN_TT_GDNC_CORES block. */
    pn_tt_residency gdnc_member[PN_TT_GDNC_MAX_MEMBERS - 1u];
    uint32_t gdnc_members; /**< 1 when unsplit — `gdnc` alone */
    /** EXTRA GDN residencies for the PHASE DELTA head split, ON BY DEFAULT AT 4 behind
     *  `PN_TT_GDN_HEAD_CORES`. PHASE DELTA divides EXACTLY by v-head — measured
     *  `59.8 us + N x 103.4` — so N members each owning `v_heads/N` heads is the whole
     *  lever. **It parallelises 1.83 / 2.70 / 2.65x on 2 / 4 / 8, not the 1.96 / 3.79 /
     *  7.12x the core-scaling probe reported**: that probe gave every core a DIFFERENT
     *  layer's state, so no two descriptors shared DRAM, where a real split shares the
     *  staged q|k|v and the gate block and scatters into the same four `y` tiles. Every
     *  member carries the IDENTICAL circular-buffer placement, because nothing this family
     *  places is sized from `v_heads`. `gdn` itself is member 0, so this array holds one
     *  fewer. */
    pn_tt_residency gdn_member[PN_TT_GDN_MAX_MEMBERS - 1u];
    uint32_t gdn_members; /**< 1 when unsplit — `gdn` alone */
    pn_tt_group_pull gdn_pull; /**< S1: `gdn` posts for a CONV / DELTA split */
    /** EXTRA ELTWISE residencies, behind `PN_TT_ELTWISE_CORES`. Every descriptor of the
     *  family is split into `eltwise_members` contiguous, DISJOINT tile ranges — no shared
     *  buffer, no shared DRAM byte, no reduction — and the set is enqueued before any member
     *  is awaited. `eltwise` itself is member 0, so this array holds one fewer. See
     *  pn_tt_eltwise_group.h for the core map and why 106 upward cannot collide. */
    pn_tt_residency eltwise_member[PN_TT_ELTWISE_SPLIT_MAX_MEMBERS - 1u];
    uint32_t eltwise_members; /**< 1 when unsplit — `eltwise` alone; 0 before it arms */
    /** PHASE W1a: how many of `eltwise_members` serve EVERY descriptor (the decode split).
     *  Members past it are WIDE members (`PN_TT_ELTWISE_WIDE_CORES`), which only descriptors
     *  of `PN_TT_ELTWISE_WIDE_MIN_TILES` or more use — the prefill SwiGLU. */
    uint32_t eltwise_decode_members;
    /** S1: member 0 posts for every split descriptor; the table names all `eltwise_members`
     *  (decode then wide), and a decode split pulls the first `eltwise_decode_members`. */
    pn_tt_group_pull eltwise_pull;
    /** PM 7: the NEXT eltwise descriptor is enqueued and NOT awaited
     *  (`pn_tt_eltwise_group_defer_next`); it stays pending until
     *  `pn_tt_eltwise_group_settle()` (or the next eltwise descriptor) awaits it. Streamed,
     *  that moves its puller AWAIT from straight after it to where a consumer needs it. */
    bool eltwise_defer_next;
    uint32_t eltwise_pending_members; /**< members of the pending descriptor, 0 = none */
    bool eltwise_pending_pulled;      /**< it went out with the S1 pull */
    uint32_t eltwise_pending_words[PN_TT_ELTWISE_SPLIT_MAX_MEMBERS][FT_TT_OP_VECTOR_WORDS];
    ft_tt_op_eltwise_request eltwise_pending_request;
    double eltwise_pending_started;
    uint64_t eltwise_deferred; /**< descriptors deferred over the process */
    /** PM 7 ITEM 5: the GDN GATE family (pn_tt_gdngate.h) — members, the per-layer constants
     *  table, the per-layer gate-block arrays, and the last issue (for a refused member's
     *  replay). 0 members = not armed: the host gate path stays. */
    pn_tt_residency gdngate[PN_TT_GDNGATE_MAX_MEMBERS];
    uint32_t gdngate_members, gdngate_layers, gdngate_heads, gdngate_max_bands;
    uint32_t gdngate_scale_bits, gdngate_block_stride;
    pn_tt_gdn_buffer gdngate_constants, gdngate_blocks;
    uint32_t gdngate_pending[PN_TT_GDNGATE_MAX_MEMBERS];
    uint32_t gdngate_next_member;
    uint32_t gdngate_cores[PN_TT_GDNGATE_MAX_MEMBERS];
    bool gdngate_failed;
    uint64_t gdngate_issues;
    uint64_t gdngate_flats; /**< ITEM 6: decode/verify FLAT gate descriptors */
    uint64_t gdngate_layer_issues, gdngate_settles;
    double gdngate_settle_seconds;
    struct {
        uint32_t layer, rows, sub_rows;
        pn_tt_activation alpha, beta;
    } gdngate_last;
    /** PHASE W: the reader thread (pn_tt_reader.h), started on first use; `reader_failed`
     *  once it could not be (no reader window bank), so nothing retries per call. */
    struct pn_tt_reader *reader;
    bool reader_failed;
    /** Each member's own AWAIT time, for the report that diagnoses the knee. The awaits run
     *  in member order after every enqueue, so a balanced set puts nearly all of it on member
     *  0 and a STRAGGLER is the one member with time of its own. */
    double eltwise_member_seconds[PN_TT_ELTWISE_SPLIT_MAX_MEMBERS];
    /** EXTRA reduce residencies, one per distinct HEAD width, placed for that width rather
     *  than the model's. The GDN's per-head norms are Wt = S/32 = 4 and attention's
     *  QK-RMSNorms are Wt = head_dim/32 = 8, against the model's 128; a reduce residency is
     *  placed for one Wt — see pn_tt_head_norm.h for why they must not share a core rather
     *  than merely need not. */
    pn_tt_head_norm_residency head_norm[PN_TT_HEAD_NORM_MAX_WIDTHS];
    uint32_t head_norm_count; /**< live slots, in the order they armed */
    /** THE ACTIVATION STRIP BRIDGE — one residency per DIRECTION, `pn_tt_render.h`. They are
     *  two programs and not one because a writer is built for exactly one output page size,
     *  so a single triad cannot both widen and narrow. */
    pn_tt_render_residency render[PN_TT_RENDER_DIRECTIONS];
    /** THE MERGED GATE that deletes the narrow half of that bridge, `pn_tt_gate.h`. Default
     *  off; when it is up, `render[PN_TT_RENDER_NARROW]` issues no decode descriptors. */
    pn_tt_gate_residency gate;
    bool oncard[PN_TT_ONCARD_COUNT];
    bool oncard_requested[PN_TT_ONCARD_COUNT];
    /** What `PN_TT_ONCARD` unset — or `all` — selects, PER MODEL FAMILY. It is not a
     *  constant because on-card attention is a different forward on each family and the
     *  evidence for it is per family: it is measured good on the qwen35/bonsai hybrid
     *  (chat parity 0.986080, +0.4% perplexity, faster at every length) and measured to FAIL
     *  the same gate on MiniCPM5-1B through `llama_forward_tt.c`. One global default would
     *  have shipped the second as a side effect of shipping the first.
     *  pn_tt_device_oncard_default() sets it; pn_tt_device_open() seeds it. */
    bool oncard_default[PN_TT_ONCARD_COUNT];
    /** Requested because the OPERATOR NAMED IT, as opposed to because it is the default.
     *  The two want opposite failure behaviour and `oncard_requested` cannot tell them
     *  apart — see pn_tt_device_oncard_named(). */
    bool oncard_named[PN_TT_ONCARD_COUNT];
    uint32_t reduce_tile_cols;   /**< the ONE Wt every reduce descriptor carries */
    /** P3: the model norm's schedule, fixed at arm time with the placement it needs. */
    pn_tt_norm_schedule reduce_schedule;
    uint32_t eltwise_depth;      /**< the ONE block depth every eltwise descriptor carries */
    /** The caller's promise that every strip holds ONE live row (row 0) right now — a
     *  single-token forward. pn_tt_device_set_single_row(). */
    bool single_row;
    /** S2c: a verify's live rows are rows 0..verify_rows-1 (0 = no promise). <= 16: the
     *  half-tile levers (matmul M5 faces 0-1, reduce M3) hold; <= 4: eltwise E1 too (its SFPU
     *  row-0 body covers rows 0-3). */
    uint32_t verify_rows;
    uint32_t topk_row; /**< S2c: the logits row the next top-k selects over (0 = decode) */
    bool topk_drain_only; /**< FUSE PROBE: the next top-k pops its blocks, selects nothing */
    /** TAIL COUNTS (pn_tt_topk_tail.h): when tail_bins > 0 the next top-k also counts the row
     *  into the bins tail_keys define; every member's counts are summed into tail_counts and
     *  tail_ok says whether all of them counted. */
    uint32_t tail_bins;
    uint32_t tail_keys[32];
    uint32_t tail_counts[33];
    bool tail_ok;
    /** The matmul K-block depth THIS MODEL wants when `PN_TT_PREFETCH_TILES` is unset
     *  (0 = the global default). pn_tt_device_prefer_prefetch_tiles(). */
    uint32_t prefetch_tiles_preferred;
    uint32_t sample_tile_count;  /**< the vocabulary row's tile count */
    uint32_t sample_depth;
    uint32_t sample_result_address; /**< L1 on the sampling core */
    uint32_t sample_state_address;  /**< L1 on the sampling core */
    /** L3 TOP-K SPLIT (`PN_TT_TOPK_CORES`): members 1..n-1 each select over a contiguous
     *  vocabulary slice; the host merges by (key desc, index asc), the kernel's own order. */
    pn_tt_residency sample_member[PN_TT_TOPK_MAX_MEMBERS - 1u];
    uint32_t sample_members;
    uint32_t sample_member_result[PN_TT_TOPK_MAX_MEMBERS - 1u];
    uint32_t sample_member_state[PN_TT_TOPK_MAX_MEMBERS - 1u];
    /* The shapes each family was armed with, kept so a retired residency can be re-armed
     * from them. reduce_tile_cols and sample_tile_count above serve the same purpose for
     * their families; the elementwise counts have nowhere else to live. */
    uint32_t oncard_eltwise_tiles[PN_TT_ONCARD_MAX_ELTWISE_SHAPES];
    uint32_t oncard_eltwise_count;
    uint64_t oncard_restarts;       /**< times an on-card residency had to be re-armed */
    uint32_t strip_bank_cursor;     /**< round-robins DRAM activation strips across banks */
    /** `PN_TT_STRIP_TILE_ROWS` — bands of 32 rows every activation strip is allocated with,
     *  resolved ONCE (0 means "not read yet") so a strip allocated late cannot disagree with
     *  one allocated early. 1 is the shipped height and the pre-Mt allocation byte for byte. */
    uint32_t strip_tile_rows;
    /** The height used when `PN_TT_STRIP_TILE_ROWS` is unset; 0 = 1. Set by a MODEL whose
     *  batched prefill can run more than one band (pn_tt_device_prefer_strip_tile_rows()). */
    uint32_t strip_tile_rows_preferred;
    /** PHASE G option 3: the height of the PREFILL-ONLY strip set (`PN_TT_PREFILL_TILE_ROWS`),
     *  0 when prefill shares the decode strips. Decode strips stay `strip_tile_rows`. */
    uint32_t prefill_tile_rows;
    /** `PN_TT_ACTIVATION_REPLICAS` — how many DRAM banks the HOT activation strips are
     *  mirrored into, so N matmul members do not all read one ~59 GB/s bank. 1 is off and
     *  is the pre-replication forward byte for byte. See pn_tt_replica_map.h. */
    uint32_t activation_replicas;
    /** Round-robins WEIGHTS across banks, one step per tensor. It lives on the device rather
     *  than in `pn_tt_device_load_artifact()` because a weight re-staged after load — a
     *  de-interleaved half of a joint projection — must continue the same rotation rather
     *  than restart it and pile every restaged tensor onto one bank. Starts at 2. */
    uint32_t weight_bank_cursor;
    /** A tensor family uploaded as its column parts instead of whole, set before the artifact
     *  loads (`pn_tt_device_split_weights_at_load`). `weight_split_set` false = none. */
    pn_tt_weight_split_rule weight_split;
    bool weight_split_set;

    /* ------------------------------------------------------------
     * The attention triple — RoPE, KV append, flash decode
     * ------------------------------------------------------------
     * The shapes are kept, not consumed: every one of the three residencies is BOUNDED and
     * retires on idle, so re-arming one needs what it was armed with.
     */
    /** K's layout. V's has the same geometry (contract F1) and its own tile when the cache
     *  is mixed (pn_tt_kv_format.h): `kv_v_layout`. Pick by `pn_tt_kv_layout_of()`. */
    ft_tt_kv_cache_layout kv_layout;
    ft_tt_kv_cache_layout kv_v_layout;
    ft_tt_kv_cache_layout kv_group_layout;   /**< one bank group's K heads (pn_tt_kv_banks.h) */
    ft_tt_kv_cache_layout kv_v_group_layout; /**< ... and V heads */
    uint32_t kv_bank_groups;                 /**< 1 = one region a layer per K / V */
    bool kv_append_banked;                   /**< one BANKED append descriptor a region */
    pn_tt_kv_type kv_type;                   /**< K and V formats (pn_tt_kv_format.h) */
    /** The KV type the fit policy chose for this load, over `PN_TT_KV_TYPE`
     *  (pn_tt_device_set_kv_type); `kv_type_set` false = the knob's. */
    pn_tt_kv_type kv_type_override;
    bool kv_type_set;
    /** K4H: H as a resident matmul weight (pn_tt_kv_rotate.h); `kv_rotate_armed` once
     *  uploaded. Only when `kv_type.rotate`. */
    pn_tt_weight kv_rotate_weight;
    bool kv_rotate_armed;
    pn_tt_kv_region *kv_region;      /**< one per layer */
    uint32_t kv_layer_count;
    uint32_t attn_q_heads, attn_kv_heads, attn_head_dim, attn_rope_dim;
    uint32_t attn_q_rows;      /**< Q heads per KV group = q_heads / kv_heads */
    uint32_t attn_dim_tiles;   /**< head_dim / 32 */
    uint32_t attn_block_tiles; /**< the ONE prefetch depth the residency was placed for */
    uint32_t attn_embed_tiles; /**< the memory family's OTHER group size, required non-zero */
    double attn_rope_theta;
    /** The constants operand: the reduce scaler, 1/sqrt(head_dim) and the 32 tail masks.
     *  Built once per model. */
    uint32_t attn_const_bank, attn_const_address;
    /** ONE row block of the rotary table, rebuilt per token so its ROW 0 carries this
     *  token's position — which is what lets the whole forward stay at tile row 0 instead of
     *  carrying the position's row phase through every op (pn_tt_attention.c). */
    uint32_t rope_table_bank, rope_table_address;
    /** A SECOND chunk-only rotary table, so two bands of a pass (first positions p and p+32)
     *  each keep their block cached instead of evicting each other. Odd chunk indices use it;
     *  decode never does. */
    uint32_t rope_chunk1_bank, rope_chunk1_address;
    uint32_t rope_chunk1_first; /**< first position it holds; UINT32_MAX = none */
    /** PHASE G: a whole-prompt pass rotates up to 64 chunks per attention layer, and two slots
     *  evicted each other 51 times a layer (+0.14 s). Chunks 2.. get a block each, built once
     *  per prefill and keyed by first position. Allocated lazily. */
    uint32_t rope_many_bank[64], rope_many_address[64], rope_many_first[64];
    /** S2c: the verify's rotary block (rows 0..k-1 = positions first..), decode-formula. */
    uint32_t rope_rows_bank, rope_rows_address;
    uint32_t rope_rows_key_first, rope_rows_key_rows; /**< the block it holds (rows 0 = none) */
    /** S2f: the attention ROWS-mode staircase mask tile (on the constants' bank) and the
     *  host copy of the constants it is assembled from. */
    uint32_t attn_rows_mask_address;
    /** (first % 32) | rows << 8 | 1 << 16 when built, one a CHUNK: a verify wider than
     *  32 / q_rows rows runs as several ROWS descriptors, each with its own mask tile. */
    uint32_t attn_rows_mask_key[PN_TT_ATTN_ROWS_CHUNKS];
    unsigned char *attn_constants_host;
    uint32_t rope_table_position;  /**< the position row 0 currently holds; UINT32_MAX = none */
    /** True while the rotary block holds a CHUNK's 32 rows rather than a decode token's row
     *  0 plus 31 zeros. Decode's next rebuild has to re-zero before it writes row 0, or the
     *  stated rows-1..31-are-zero invariant is only true by accident (of the ACTIVATION's
     *  padding, not the table's). */
    bool rope_table_is_chunk;
    uint32_t rope_table_chunk_first; /**< first position the chunk block holds, when is_chunk */

    /* ------------------------------------------------------------
     * The GATED-DELTA-NET family — one core, and the recurrence's state
     * ------------------------------------------------------------
     * The per-layer regions are the only DRAM in this backend that is READ AND WRITTEN by the
     * card and never downloaded. The five staging buffers below them are SHARED across every
     * layer, because a token runs one layer at a time and nothing in them carries.
     */
    pn_tt_gdn_shape gdn_shape;
    pn_tt_gdn_region *gdn_region;    /**< one per SSM layer */
    /** Option D: does `gdn_region[].weight` alias the chunkwise family's conv constant copy 0
     *  (pn_tt_gdn_conv_share.h)? A pn_tt_gdn_conv_share_state. */
    uint8_t gdn_conv_share;
    /** STEP 4 S2: snapshot slots of every layer's window + state for the speculative verify's
     *  rollback, `gdn_snapshot[slot * layers + layer]`, allocated on first use. */
    pn_tt_gdn_region *gdn_snapshot;
    uint32_t gdn_snapshot_slots;
    /** The verify's COMPACT slot map (rows - 1 slots, pn_tt_gdn_verify_slots.h); off = legacy. */
    bool gdn_verify_compact;
    /** Multi-turn: every layer's window + state copied at a prefill pass boundary
     *  (pn_tt_gdn_checkpoint.c), one region a layer, allocated on the first capture. */
    pn_tt_gdn_region *gdn_checkpoint;
    uint32_t gdn_state_out_plus1; /**< 0 = in place; else slot + 1 takes the state writes */
    uint32_t gdn_row;             /**< S2c: the CONV/DELTA row select, 0 = decode */
    /** S2c: deferred awaits — split CONV/DELTA descriptors are enqueued and awaited together
     *  by pn_tt_gdn_await_deferred(); one gate block a row so queued DELTAs never share one. */
    bool gdn_defer;
    uint32_t gdn_pending_members;
    /** S1: every split the deferred window holds went out pulled, so `gdn` alone is awaited. */
    bool gdn_pending_pulled;
    uint64_t keepalive_nops; /**< PM 5: NOPs the keepalive has sent */
    uint32_t gdn_pending_depth; /**< deferred descriptors a member holds since its last await */
    uint64_t gdn_defer_drains;  /**< full-ring drains the deferred rows took (flow control) */
    pn_tt_gdn_buffer gdn_gate_row[16];
    uint32_t gdn_gate_rows, gdn_gate_stride;
    pn_tt_gdn_buffer gdn_stream_in; /**< CONV: the qkv projection, a row-0 strip */
    /** CONV's OUTPUT **and** DELTA's INPUT — ONE buffer, because they are one geometry. Both
     *  are `conv_dim/32` row-0 tiles and `conv_out`'s q, k and v slices sit exactly where
     *  `ftg_delta_stream_base()` looks for them, so the delta rule reads the convolution's
     *  answer in place. All the per-head L2 norm has to do is write its answer back over the
     *  first `2*k_heads*state_tiles` tiles of it. */
    pn_tt_gdn_buffer gdn_stream_out;
    pn_tt_gdn_buffer gdn_gate; /**< DELTA: decay[H] then beta[H], flat fp32 */
    /** THE EXACT CARD GATES in decode (pn_tt_gdn_table_gate): while `gdn_table_on`, DELTA
     *  descriptors name this table block and the reader builds the gate block itself. */
    pn_tt_gdn_buffer gdn_table_gate;
    bool gdn_table_on;
    pn_tt_gdn_buffer gdn_y;    /**< DELTA: y, a row-0 strip, head-major */
    unsigned char *gdn_stage;        /**< host tile staging; grown on demand */
    size_t gdn_stage_bytes;
    /** `PN_TT_GDN_SELFCHECK`: read the staged qkv row and conv weight back out of DRAM once
     *  and compare, so a wrong GDN answer can be attributed to the host layout or to the
     *  card rather than to both. `gdn_weight_oracle` is layer 0's `ssm_conv1d`, borrowed. */
    bool gdn_self_checked;
    bool gdn_out_checked;
    /** `PN_TT_GDN_SELFCHECK`: DELTA descriptors scored against the closed form so far. It is
     *  a COUNT and not a flag because position 0 is the one place the carried state is
     *  provably zero — so every SSM layer of the FIRST token has a closed form, and one layer
     *  cannot say whether an error is the op's or that layer's data. */
    uint32_t gdn_delta_checks;
    /** A HOST SHADOW of SSM layer 0's rolling conv window, plus the tokens scored against it.
     *  Position 0 is the ONE position where the window is trivially zero, so a check that
     *  stops there says nothing about the ROLL — and the roll is the only part of PHASE CONV
     *  that carries state across tokens. */
    float *gdn_window_shadow;
    uint32_t gdn_window_checks;
    /** Has the injective-stream map already run? `PN_TT_GDN_RAMP` poisons the layer it probes,
     *  so it runs once. */
    bool gdn_ramp_probed;
    /** Has the CONV data arm echoed the stream it is about to convolve? Once a run — an arm
     *  that silently did not fire reads exactly like an arm that fired and cost nothing, and
     *  those two are the whole question the arm exists to separate. */
    bool gdn_synthetic_reported;
    /** Whole-tile strip checks taken on the SHIPPED on-card route, capped so three layers
     *  over four tokens is a probe and not a log. */
    uint32_t gdn_strip_checks;
    /** Has `PN_TT_GDN_CORES`' multi-core sweep already run? It borrows the carried state of
     *  the layers it uses, so it runs ONCE per device and at position 0. */
    bool gdn_probe_done;
    /** PER-PHASE on-card accounting, indexed by `FT_TT_OP_GDN_PHASE_*` (1 and 2; slot 0 is
     *  never written). `oncard_seconds` is a SUM over every family, so the plan's "+0.110 s
     *  over 48 GDN descriptors" was a SUBTRACTION between two runs — and a subtraction cannot
     *  tell CONV from DELTA, which do different work and are the two things a re-pricing has
     *  to separate. These are measured directly, in one run. */
    double gdn_phase_seconds[3];
    uint64_t gdn_phase_ops[3];
    /** THE SAME PHASES, FOR DESCRIPTORS THAT REFUSED AND WERE REPLAYED. The submit stopwatch
     *  spans the refusal, the re-arm — a residency teardown and relaunch — and the resubmit,
     *  so a refusing run prices a teardown as the op's arithmetic. That is exactly how PHASE
     *  CONV once read **4,502 us a descriptor against 738.2**, and it is the shape the
     *  unsourced "CONV is DATA-DEPENDENT, 4,571 us on a degenerate stream" also has. Kept
     *  apart so `gdn_phase_seconds / gdn_phase_ops` is the CLEAN figure and a reader that
     *  quotes it prints `gdn_phase_rearms` beside it. */
    double gdn_phase_recovered_seconds[3];
    uint64_t gdn_phase_recovered_ops[3];
    uint64_t gdn_phase_rearms[3];
    /** Every SSM layer's qkv, as it was uploaded, so a wrong tensor coming back out of the
     *  card can be NAMED rather than described. `layer_count * conv_dim` floats. */
    float *gdn_qkv_history;
    const float *gdn_weight_oracle;

    /* ------------------------------------------------------------
     * The CHUNKWISE family's own operands — see pn_tt_gdnc.h
     * ------------------------------------------------------------
     * Three buffers, and every one of them is a CONSTANT or a per-chunk constant. The chunk's
     * activations are the gdn family's staging strips and its carried state is the gdn
     * family's regions, so nothing here is read back and nothing here carries.
     */
    pn_tt_gdn_buffer *gdnc_constant;      /**< one per SSM layer: ROW-REPLICATED tap planes,
                                           *   then the 2*taps band tiles at the end of them */
    /** PM 8: copies of each layer's conv constant in OTHER DRAM banks, so the CONV members
     *  do not all read one bank (`PN_TT_GDNC_CONST_REPLICAS`, 1 = none). Member m reads copy
     *  `m % replicas` (copy 0 = `gdnc_constant`). `layer_count * (replicas - 1)` entries. */
    pn_tt_gdn_buffer *gdnc_constant_replica;
    uint32_t gdnc_constant_replicas;
    /** The copy count the model's fit asked for (pn_tt_gdnc_request_replicas); 0 = the
     *  default. `PN_TT_GDNC_CONST_REPLICAS` still wins. */
    uint32_t gdnc_constant_replicas_requested;
    uint32_t gdnc_conv_layer; /**< the layer of the CONV descriptor being submitted */
    uint32_t gdnc_delta_layer; /**< the layer of the DELTA descriptor being submitted */
    /** Prefill lane (`PN_TT_GDNC_STATE_BANKS`): each DELTA member's heads of a layer's
     *  recurrent state in that member's OWN bank while the layer's band loop runs, copied in
     *  from / out to `gdn_region[layer].state` around it. `layer_count * gdnc_members`. */
    pn_tt_gdn_buffer *gdnc_state_split;
    bool *gdnc_state_split_live;
    uint32_t gdnc_state_split_members;
    /** Prefill lane PIPE 2 (`PN_TT_GDN_PIPE=2`): a second stream_in / stream_out / y triple
     *  (one bank, the two activation pairs' contract) so band b+1's widen + CONV and band b's
     *  DELTA / RMS never share a buffer. `gdnc_parity` 1 = the alternates are swapped into
     *  `gdn_stream_in` / `gdn_stream_out` / `gdn_y`. */
    pn_tt_gdn_buffer gdnc_alt_stream_in, gdnc_alt_stream_out, gdnc_alt_y;
    bool gdnc_alt_ready;
    uint32_t gdnc_parity;
    /** The next PHASE DELTA descriptor carries kGdncFlagSerialize (its reader waits for the
     *  previous descriptor's published completion: the state carries through DRAM). */
    bool gdnc_serialize;
    /** THE DELTA STAGING, IN TWO SLOTS — one per sub-chunk index (`first_row / 16`), so the
     *  host can stage sub-chunk 1 while the card is still reading sub-chunk 0's. Each slot
     *  is a whole constant region (I, 2I, the strict mask, scale, then one decay-ratio tile
     *  per v-head) plus its gate block PINNED to the same bank, which they share a
     *  coordinate word with. */
    pn_tt_gdn_buffer gdnc_delta_constant[PN_TT_GDNC_STAGE_SLOTS];
    pn_tt_gdn_buffer gdnc_gate[PN_TT_GDNC_STAGE_SLOTS];
    /** What each slot holds ON THE CARD, byte for byte, so an upload can send only the units
     *  that changed and the card's bytes still equal the builders' by construction. */
    unsigned char *gdnc_shadow_decay[PN_TT_GDNC_STAGE_SLOTS];
    float *gdnc_shadow_gate[PN_TT_GDNC_STAGE_SLOTS];
    bool gdnc_shadow_valid[PN_TT_GDNC_STAGE_SLOTS];
    /** A slot staged ahead of its descriptor (`pn_tt_gdnc_delta_prestage_at()`), consumed by
     *  the descriptor whose (layer, first_row, chunk_rows) it matches. */
    struct {
        bool valid;
        uint32_t layer, first_row, chunk_rows;
    } gdnc_prestaged[PN_TT_GDNC_STAGE_SLOTS];
    /** The worker that builds a staging's heads while this thread uploads them
     *  (`PN_TT_GDNC_STAGE_THREAD`). Stopped in pn_tt_gdnc_release(). */
    pn_tt_stage_pipeline gdnc_stage_pipeline;
    /** The exact card gates (pn_tt_gdnc_table_arm): the host's lookup tables and each layer's
     *  table block. NULL until armed. */
    struct pn_tt_gdnc_table_state *gdnc_table;
    /** ONE-SHOT host work to run between the next split's enqueue and its await. */
    int (*gdnc_overlap_fn)(void *context);
    void *gdnc_overlap_context;
    /** A SECOND one-shot slot, run after the first in the same window (the prefill's
     *  alpha/beta prefetch; the first belongs to the DELTA staging). It may re-arm itself. */
    int (*gdnc_overlap_aux_fn)(void *context);
    void *gdnc_overlap_aux_context;
    uint64_t gdnc_stage_writes;       /**< DRAM write calls the staging made */
    uint64_t gdnc_stage_hidden_ops;   /**< stagings that ran under a split's card time */
    uint64_t gdnc_stage_verify_checks;     /**< PN_TT_GDNC_STAGE_VERIFY read-backs */
    double host_clock_seconds[PN_TT_HOST_CLOCK_COUNT]; /**< pn_tt_device_host_clock_add() */
    uint64_t gdnc_stage_verify_mismatches; /**< units that differed from the builders */
    unsigned char *gdnc_decay_stage;      /**< v_heads fp32 tiles, contiguous, so the 128 KiB
                                           *   a layer a chunk is ONE write */
    float *gdnc_gate_stage;               /**< v_heads * FT_TT_OP_GDNC_GATE_STRIDE */
    /** PER-PHASE accounting, indexed by `FT_TT_OP_GDNC_PHASE_*`. Separate from the decode
     *  family's because the two run different work on different cores and a prefill that
     *  mixes chunks with a `1..taps-2` tail runs BOTH. */
    double gdnc_phase_seconds[3];
    uint64_t gdnc_phase_ops[3];
    /** PER-MEMBER AWAIT, so a core split is visible as `members` numbers and not only as one
     *  that moved. The awaits are issued in order after every enqueue, so a BALANCED set puts
     *  nearly all of the wall on member 0 and leaves the rest near zero — the same signature
     *  the matmul group's per-member wait line carries, and the discriminator between a set
     *  that ran concurrently and one this host serialised. */
    double gdnc_member_seconds[PN_TT_GDNC_MAX_MEMBERS];
    uint64_t gdnc_member_ops[PN_TT_GDNC_MAX_MEMBERS];
    /** And the replayed ones, for the same reason the decode family keeps them apart — this
     *  is the family the 4,502 us reading was actually taken on. */
    double gdnc_phase_recovered_seconds[3];
    uint64_t gdnc_phase_recovered_ops[3];
    uint64_t gdnc_phase_rearms[3];
    /** THE DELTA STAGING, WHICH NO PHASE LINE COULD SEE. Every DELTA descriptor transposes
     *  `v_heads` decay runs and gate runs on the host and writes 151,552 B of them to DRAM
     *  BEFORE `gdnc_submit()` starts its stopwatch, so the cost is inside the caller's
     *  `sequential` bracket and attributed to nothing. Two clocks, not one: BUILD is host
     *  arithmetic and UPLOAD is the x1 Gen2 link, and only the second is what a smaller
     *  operand would move. `bytes` is carried so the rate is derived here rather than from an
     *  assumed operand size. */
    double gdnc_stage_build_seconds;
    double gdnc_stage_upload_seconds;
    uint64_t gdnc_stage_ops;
    uint64_t gdnc_stage_bytes;
    /** THE SPLIT'S COMPLETION SHAPE, and the clock that says what it cost. `gdnc_group` is
     *  true only when every armed member carries the group binding AND the block read back
     *  with its magic — a partial arm is not a shape, it is a set of members some of which
     *  post and some of which do not, and it falls back rather than running. The sequence is
     *  the GROUP's notion of "item N" and is stamped identically on every member of one
     *  issue; it never resets, so a slot holding a previous issue's number can never satisfy
     *  this one. */
    bool gdnc_group;
    uint32_t gdnc_sequence;
    double gdnc_group_seconds;
    uint64_t gdnc_group_ops;

    /** Which host sequence owns the ONE region set, and the keys it displaced. A displaced
     *  key coming back is two sequences alternating, and it is refused by name. */
    const void *kv_key;
    const void *kv_displaced[PN_TT_KV_DISPLACED_KEYS];
    uint32_t kv_displaced_count;
    uint64_t kv_rebinds;

    /* Scratch, on the card: one activation strip in, one result strip out. Used only by
     * the host-in/host-out pn_tt_matvec(); the on-card forward names its own strips. */
    ft_tt_buffer activation;
    ft_tt_buffer result;

    /* Scratch, on the host. Only the two ROWS: a matvec moves logical row 0 of each tile
     * and nothing else, so the full tile staging buffers this used to keep are gone. */
    uint16_t *row_in;  /**< in_features bf16 */
    uint16_t *row_out; /**< out_features bf16 */
    uint32_t max_in_features, max_out_features, max_tile_rows, max_tile_cols;
    uint32_t max_row_features; /**< the widest of the two; both host rows are this */

    /* Scratch for the MULTI-ROW strip transfers (pn_tt_activation_{upload,download}_rows).
     * A batched prefill fills rows 0..M-1 of the same tiles a matvec leaves as zero padding,
     * so those two move WHOLE tiles and need the tile-shaped staging buffer the row-0 path
     * deliberately does not keep. Grown on demand and freed with the device; NULL until a
     * batched call asks, so a decode-only run pays nothing for them. */
    unsigned char *tile_stage;  /**< tiles * 2048 B, face-ordered */
    size_t tile_stage_bytes;
    uint16_t *rows_stage;       /**< rows * count bf16, row-major — the tilizer's input */
    size_t rows_stage_bytes;

    /* THE MULTI-CORE CORRECTNESS GATE. `PN_TT_MATVEC_DUMP=<file>` appends logical row 0 of
     * EVERY projection's output, in dispatch order, as raw bf16 — so one run at N=1 and one
     * at N=4 are two files that must be byte-identical. It is a byte comparison and not a
     * score on purpose: two members overwriting each other completed every kernel, kept
     * every status word clean and left the aggregate GB/s unchanged, and only bytes saw it
     * (foundry/tt/DEV_NOTES.md §3). Off by default; it costs a readback per op. */
    FILE *matvec_dump;
    uint64_t matvec_dump_bytes;
    uint64_t matvec_dump_rows;

    /** WHAT HAS BEEN ISSUED AND NOT AWAITED, so a card refusal can be REPLAYED onto a
     *  re-armed residency rather than costing the token. Chained projections are
     *  independent by contract, so replaying them is the same computation in the same
     *  order — which is why the recovery can be the plain synchronous path. */
    struct {
        const pn_tt_weight *weight;
        pn_tt_activation input;
        pn_tt_activation output;
        /** S3b: non-NULL = a FUSED SwiGLU projection (`weight` is the gate, this the up). */
        const pn_tt_weight *swiglu_up;
        /** S3a: true = a RESADD projection (`output` = `residual` + bf16(input . weight)). */
        bool resadd;
        pn_tt_activation residual;
    } pending[PN_TT_MATVEC_CHAIN_CEILING];
    uint32_t pending_count;

    uint64_t matvec_count;
    double matvec_seconds;
    double upload_seconds;   /**< bf16 convert + the row-0 host->card write */
    double dispatch_seconds; /**< ft_tt_op_matmul_run: image load, CB config, go, poll */
    /** `dispatch` split at the seam that matters: ISSUE is the host writing parameter words
     *  and ring entries over MMIO, WAIT is the completion poll and everything the card does
     *  under it. Phase 3 removes issue outright and one poll per wait; whatever is left of
     *  wait is the card's own work and survives it. */
    double dispatch_issue_seconds;
    double dispatch_wait_seconds;
    uint64_t dispatch_issues;
    uint64_t dispatch_waits;
    uint64_t dispatch_wait_descriptors;
    double readback_seconds; /**< the row-0 card->host read + bf16 convert */
    double oncard_seconds;   /**< enqueue + wait for the reduce/eltwise/sample residencies */
    uint64_t oncard_ops;
    /** The SAME descriptors as `oncard_seconds`, keyed by (family, function, tile count).
     *  `oncard_seconds` is one number over every family and the per-residency report is one
     *  number over every SHAPE — the elementwise residency runs six functions over three
     *  widths and reports a single 293.8 us mean, and the two render residencies are not in
     *  that report at all. See pn_tt_descriptor_census.h. */
    pn_tt_descriptor_census census;
    uint64_t bytes_read, bytes_written;

    /** The routed-expert block (pn_tt_moe.c), NULL until armed; `moe_defer_weights` keeps
     *  MoE-block weights out of the dense load so that module places them. */
    struct pn_tt_moe *moe;
    bool moe_defer_weights;
    /** Host wall seconds spent folding + writing expert stacks (plan S3). */
    double moe_fold_seconds;
};

/** Seconds since an arbitrary epoch. Shared by the two files that time this device. */
double pn_tt_now_seconds(void);

/**
 * Charge ONE retired on-card descriptor to the aggregate counters AND to the census.
 *
 * Every family's submit path ended in the same three lines, and the census would have been
 * the fourth — copied five times, which is five places for a family to be counted in one
 * instrument and not the other. It is one call now, and the two can no longer disagree.
 *
 * @param family   residency-level name, e.g. "eltwise", "render-widen"
 * @param function what the descriptor computed, e.g. "silu_mul"
 * @param tiles    tiles the descriptor covers — the census fit's x axis
 * @param seconds  measured wall time inside submit
 */
void pn_tt_oncard_account(pn_tt_device *device, const char *family, const char *function,
                          uint32_t tiles, double seconds);

/** `name`'s value as an unsigned, or `fallback` when unset or unparsable. */
uint32_t pn_tt_env_u32(const char *name, uint32_t fallback);

/** E3 SPLIT PROBE: OR `PN_TT_VEC_PROBE` (kVecProbe* bits) into the top byte of word 3 of an
 *  ELTWISE or REDUCE parameter block. Timing only — an armed run's text is not evidence. */
void pn_tt_vec_probe_apply(uint32_t *words);

/**
 * One matmul with BOTH operands already resident in card DRAM — no host transfer at all.
 *
 * This is the dispatch half of pn_tt_matvec(), split out so the on-card forward can chain
 * projections through DRAM strips instead of through the host.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_matvec_dispatch(pn_tt_device *device, const pn_tt_weight *weight,
                          const pn_tt_activation *input, const pn_tt_activation *output);

/* ================================================================
 * Shared by the two residency owners — pn_tt_oncard.c and pn_tt_attention.c
 * ================================================================ */

/** Every processor of a triad. The memory family is the exception: it has no compute
 *  thread and enables DM0|DM1 only (foundry/tt/ft_tt_op_memory.h PART 2). */
#define PN_TT_ONCARD_TRIAD_MASK                                                           \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |         \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** The GDN triad's program name. Here rather than in pn_tt_gdn.c because TWO files need it:
 *  that one arms the family and pn_tt_oncard.c owns the refusal when a family asked for by
 *  name has no program to arm. */
/** The RESIDENT matmul triad, built for a Bfp8_b weight. One program, not two: the
 *  production ledger puts every matmul weight of an artifact at Bfp8_b, so a run needs one
 *  resident binary and the Bfp4_b variant has no tensor to serve. That is a property of the
 *  ARTIFACT and is checked at load, never assumed — see pn_tt_matmul_group_arm().
 *
 *  HERE RATHER THAN IN pn_tt_device.c since 2026-08-27: the matmul group's DST-section
 *  interlock reads the ARMED image's recorded `defines=` to decide whether eight DST tiles
 *  are legal, so the name of the program it will arm is no longer private to the file that
 *  loads it. Two files naming the same program in two string literals is the shape this
 *  backend has already been bitten by. */
/*  S3b: the NAME is now chosen once per process — `rtriad_bfp8`, or `rtriad_bfp8_swiglu` under
 *  `PN_TT_FUSE_SWIGLU=1` (pn_tt_swiglu_fuse.h) — still through this one macro, so the device
 *  open and the DST interlock below can never ask for two different programs. */
#define PN_TT_RESIDENT_PROGRAM (pn_tt_resident_program_name())

#define PN_TT_GDN_PROGRAM "gdn"

/** The CHUNKWISE triad's program name. Same reason it is here: pn_tt_gdnc.c arms the family
 *  and pn_tt_oncard.c owns the core-collision refusal that has to name it. */
#define PN_TT_GDNC_PROGRAM "gdnc"

/** Just the two data-movement processors. */
#define PN_TT_ONCARD_DM_MASK ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1))

/** Bytes written per host->card call while zeroing or filling a DRAM region. */
#define PN_TT_ZERO_CHUNK_BYTES 65536u

/** The reduce family's program name. Two files arm a residency of it now — pn_tt_oncard.c
 *  at the model's width and pn_tt_head_norm.c at the head width. */
#define PN_TT_REDUCE_PROGRAM_NAME "reduce"

/** THE SAME TRIAD PLACED AT Float32 — `kernels/cb_tables/reduce_float32.cbtable`, a
 *  `TILE_BYTES=4096` build of the same three sources, no kernel change.
 *
 *  It is a SECOND program and not a replacement for two reasons. The bf16 one is 9x past the
 *  1e-3 gate on a NARROW norm and inside it on the model's — the error is per reduce, so the
 *  members that matter are the ones with few elements to average it over. And doubling the
 *  page halves the widest residency the arena serves, 107 tile columns against 217, so the
 *  model's `n_embd/32` = 128 RMS norms DO NOT FIT at Float32 and stay on `reduce`. */
#define PN_TT_REDUCE_FP32_PROGRAM_NAME "reduce_fp32"

/**
 * Build and upload ONE reduce member's constants buffer.
 *
 * The scaler tile, the eps tile and gamma all depend on WHICH MEMBER, so a buffer built for
 * one is wrong for another — which is why the function is a parameter rather than baked in.
 * `pn_tt_norm_constants_upload()` is this at FT_TT_OP_REDUCE_FN_RMS_NORM.
 *
 * THE PAGE SIZE IS THE RESIDENCY'S, NOT THIS FUNCTION'S. The model's reduce residency is
 * placed by `reduce_float16b.cbtable` at 2048 bytes and the per-head one by
 * `reduce_float32.cbtable` at 4096; the reader issues `TILE_BYTES` reads and nothing on the
 * card can catch a constants buffer built at the other one.
 *
 * @param gamma       `width` values, or NULL for a member whose chain has no gamma step
 * @param eps         what the member's reduce should see added to its accumulator
 * @param tile_bytes  PN_TT_TILE_BF16 or PN_TT_TILE_FP32 — the page the residency was placed at
 * @return            0, or -1 with the reason on stderr
 */
int pn_tt_reduce_constants_upload(pn_tt_device *device, ft_tt_op_reduce_function function,
                                  const float *gamma, uint32_t width, float eps,
                                  uint32_t tile_bytes, pn_tt_norm_constants *out_constants);

/**
 * P3: put `schedule`'s CB extents on `residency` (already filled for its Wt). When the arena
 * cannot hold them the schedule falls back to the pre-P3 one, logged, rather than the norm
 * falling back to the host. `what` names the residency in that line.
 */
void pn_tt_norm_schedule_fit(pn_tt_norm_schedule *schedule, ft_tt_op_reduce_residency *residency,
                             const char *what);

/**
 * P3: finish a norm descriptor's words for `schedule` — the RED face-0 arm
 * (`PN_TT_RED=norm_fast_face0`: a half read lands face 0 only, so the output MUST differ).
 */
void pn_tt_norm_schedule_words(const pn_tt_norm_schedule *schedule, uint32_t *words);

/** Is `name` the PN_TT_RED mutation this process was asked to run? Shared so the attention
 *  family's red proofs read the same knob as the on-card ones. */
bool pn_tt_red(const char *name);

/**
 * Allocate `bytes` of DRAM in the next round-robin bank (`strip_bank_cursor`, advanced once,
 * exactly as every any-bank site did), or — ONLY when that bank cannot fit it — in another
 * bank, most free first. A model whose expert stacks fill some banks (Qwen3.6 + its MTP head:
 * one bank down to 1.7 MB) then still places a lazily allocated buffer while total DRAM has
 * room; an allocation that fit before lands exactly where it did.
 * @return 0, or -1 when no bank can fit it
 */
int pn_tt_dram_alloc_any(pn_tt_device *device, size_t bytes, ft_tt_buffer *out);

/**
 * Allocate `bytes` of DRAM in the bank with the most free bytes that can hold it
 * (pn_tt_strip_placement_emptiest_bank). For the unmirrored strips of a set whose mirrors
 * were placed first, so they never take room a pinned mirror needed.
 * @return 0, or -1 when no bank can fit it
 */
int pn_tt_dram_alloc_emptiest(pn_tt_device *device, size_t bytes, ft_tt_buffer *out);

/** The DRAM half of arming the GDN family: every layer's regions and the shared staging
 *  buffers, no residency. `pn_tt_device_arm_gdn()` calls it once its circular buffers fit;
 *  a host device with only an allocator can run it (tests/test_tt_q38_conv_share.c). 0/-1. */
int pn_tt_gdn_place_regions(pn_tt_device *device);
/** Allocate one layer's decode conv weight (any bank), as the GDN arm does. 0/-1. */
int pn_tt_gdn_alloc_conv_weight(pn_tt_device *device, pn_tt_gdn_buffer *out);
/** The DRAM half of arming the chunkwise family: conv constants (and their copies), delta
 *  constants, gate blocks. `pn_tt_device_arm_gdnc()` calls it. 0/-1. */
int pn_tt_gdnc_place_regions(pn_tt_device *device);

/** Allocate a region shaped like SSM layer `layer`'s live window + state, on the live
 *  region's banks (one coordinate word carries both). Left uninitialised. 0 or -1. */
int pn_tt_gdn_alloc_twin(pn_tt_device *device, uint32_t layer, pn_tt_gdn_region *out);
/** The bytes of one layer's carried conv window and recurrent state (unrounded). */
void pn_tt_gdn_carried_bytes(const pn_tt_device *device, size_t *window, size_t *state);

/** Write `bytes` of zeros at a DRAM address, in bounded chunks. */
int pn_tt_zero_dram(pn_tt_device *device, uint32_t bank, uint32_t address, size_t bytes);
/** `PN_TT_IDLE_RETIRE_ONLY` (diagnosis): may the residency `name` be retired for idleness? */
bool pn_tt_idle_retire_allowed(const char *name);
/** `pn_tt_zero_dram` with every byte `value` (a test poison; 0 is the zero). */
int pn_tt_fill_dram(pn_tt_device *device, uint32_t bank, uint32_t address, size_t bytes,
                    unsigned char value);

/** Write `bytes` from host memory to a DRAM address, in bounded chunks. */
int pn_tt_write_dram(pn_tt_device *device, uint32_t bank, uint32_t address, const void *data,
                     size_t bytes);

/** Read `bytes` from a DRAM address into host memory, in bounded chunks. */
int pn_tt_read_dram(pn_tt_device *device, uint32_t bank, uint32_t address, void *out,
                    size_t bytes);

/** An explicitly-named program directory to read INSTEAD of the embedded blob. */
#define PN_TT_PROGRAM_ROOT_ENV "PN_TT_PROGRAM_DIR"

/**
 * The program root this process was told to use, or NULL for the embedded blob.
 *
 * NULL IS THE DEFAULT AND THE DEFAULT IS THE BINARY'S OWN PROGRAMS. There is no fallback
 * path in either direction: an unset variable means embedded, and a set one means that
 * directory or a refusal (`ft_tt_program_resolve.h`). The hardcoded
 * absolute programs directory this replaced made Pantheon a one-machine program and
 * turned a program built into the wrong root into a silent 1.085x — the `rtriad_bfp8`
 * incident.
 *
 * One reader for the whole backend so the answer cannot differ between the matmul programs
 * and the on-card families of the same run.
 *
 * @return the directory, or NULL when the variable is unset or empty
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
const char *pn_tt_program_override_root(void);

/** Name the place programs are coming from, for a message that has to say one or the other.
 *  Takes the root so a caller cannot describe one arm while resolving through another. */
const char *pn_tt_program_where(const char *root);

/**
 * Resolve `program` by NAME and boot its firmware on this family's core.
 *
 * SEPARATE FROM THE START, and the order is load-bearing: pn_tt_residency_load() zeroes the
 * whole handle, so a caller that placed its circular buffers first would have them silently
 * erased. Prepare, THEN place, THEN start.
 */
bool pn_tt_prepare_family(pn_tt_device *device, pn_tt_residency *residency, const char *program,
                          const char *name, const char *core_env, uint32_t core_default,
                          uint32_t processor_mask);

/** The same, on a core the CALLER names — for arming several residencies of one family at
 *  once, where a single environment variable cannot say a different core for each. `core_env`
 *  is still taken, but only so the refusal message can name the knob that would move it. */
bool pn_tt_prepare_family_on_core(pn_tt_device *device, pn_tt_residency *residency,
                                  const char *program, const char *name, const char *core_env,
                                  uint32_t core, uint32_t processor_mask);

/** Who, if anyone, holds `core` in THIS process — matmul member 0, a group member, a booted
 *  prefill grid core or any live family residency — BY NAME, so a refusal can say it.
 *  `except` (may be NULL) is skipped. @return the holder's name, or NULL when free */
const char *pn_tt_core_holder(const pn_tt_device *device, uint32_t core,
                              const pn_tt_residency *except);

/** Kick a prepared residency whose circular buffers the caller has now placed. */
bool pn_tt_start_family(pn_tt_residency *residency);
/** The family last started on the core at NoC (x, y), or "?" (PM 8, AWAIT census). */
const char *pn_tt_core_label(uint16_t x, uint16_t y);

/** Does this binary (or the named override root) carry the program `program`? */
bool pn_tt_program_is_present(const char *program);

/**
 * The ONE block depth a residency can run for every tile count it will see.
 *
 * Both conditions are the CB-cycle invariant and neither is optional: `depth` must divide each
 * run so a block is contiguous, and `2 * depth` — the page count the buffers are placed at —
 * must divide it too, so each descriptor returns the write pointer to the base. A descriptor
 * that leaves it elsewhere walks the next one up L1 with no error.
 *
 * Shared because TWO files place elementwise-shaped buffers now: pn_tt_oncard.c for the
 * model's eltwise and sampling residencies, pn_tt_render.c for the two strip-bridge ones.
 *
 * @return the depth, or 0 when no depth serves every count
 */
uint32_t pn_tt_common_depth(const uint32_t *tile_counts, uint32_t count, uint32_t cap);

/** Ceiling on a resident reader's block depth. Not a tuning knob: two blocks of `depth` pages
 *  on each of two input streams is 4 * depth * 2048 bytes of a 1.29 MiB arena, and a deeper
 *  block buys nothing once the reader is no longer transaction-bound. */
#define PN_TT_ONCARD_MAX_DEPTH 32u

#endif /* USE_TT */
#endif /* PN_TT_DEVICE_INTERNAL_H */
