/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_moe.h
 * @brief The routed-expert (MoE) block on the card: the router residency (`rtriad_router`),
 *        the expert residencies (`rtriad_bfp6`), the routing slot and the scale strip, and the
 *        expert stacks resident in DRAM.
 *
 * ---------------------------------------------------------------------------
 * ONE DECODE ROW, HOST-SYNCHRONOUS ORDERING
 * ---------------------------------------------------------------------------
 * The router (one core) runs the router matmul at fp32 output and the routing step in its
 * writer: top-k with the lowest-index tie rule, det-exp renormalised weights, sigmoid of the
 * shared-expert gate. It writes the ROUTING SLOT (k expert indices, then k weights) and the
 * SCALE STRIP (bf16, row 0: `expert_tiles` tiles of w_r per rank, then `shexp_tiles` of the
 * shared gate). The expert residencies read the slot on the card (indirect B operand: the
 * descriptor carries the stack base, the index is data), so no descriptor word moves with
 * the routing and dispatch replay stays valid.
 *
 * Ordering between the families is the host's: route, await; gate/up, await; the caller's
 * SwiGLU and scale multiply; down, await. Nothing here crosses the link but descriptors.
 *
 * Work split over the M expert members (any M up to PN_TT_MOE_MAX_MEMBERS):
 *   gate/up — the k x expert_tiles (rank, output column) cells, contiguous runs per member,
 *             one gate and one up descriptor per rank a run touches;
 *   down    — the hidden_tiles output columns, each descriptor K-concatenating the k experts
 *             (FT_TT_MOE_FLAG_K_EXPERTS), so the weighted sum is accumulated in rank order in
 *             DST: A is the rank-ordered scaled SwiGLU output.
 * Every expert of a stack sits whole in one bank at `base + e * stride`; every stack of a
 * model shares one stride (the reader's stride is per residency).
 *
 * Red arms (`PN_TT_RED=<name>`, each must DIFFER): moe_idx_shift (slot index + 1, host
 * write), moe_tie_flip / moe_weights_unnorm / moe_rank_permute (router writer red bits),
 * moe_stride_off (member stride one tile long), bfp6_expand_bit (the `rtriad_bfp6_red`
 * program: expander off by one bit).
 *
 * @status Decode (one row) built: Phase 5.4.
 * @issues The expander is the reader's scalar loop (~3.4 us a tile a core, spec §13 P5.1).
 * @todo None
 */

#ifndef PN_TT_MOE_H
#define PN_TT_MOE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** Most expert residencies one block may arm. */
#define PN_TT_MOE_MAX_MEMBERS 64u
/** Most experts a token routes to (the slot and the writer's ceiling). */
#define PN_TT_MOE_MAX_K 8u

/** A routed-expert stack resident whole in one DRAM bank: expert e's N-major K x N panel at
 *  `address + e * stride`. */
typedef struct pn_tt_moe_stack {
    uint32_t bank;
    uint32_t address;
    uint16_t x, y;       /**< the bank's DRAM endpoint */
    uint32_t tile_rows;  /**< one expert's Kt */
    uint32_t tile_cols;  /**< one expert's Nt */
    uint32_t experts;
    uint32_t stride;     /**< bytes from one expert to the next */
} pn_tt_moe_stack;

/** The router weight as the card holds it: bf16 tiles of [hidden x (experts + 32)], N-major,
 *  the shared-expert gate vector at column `experts`, zero padding after it. */
typedef struct pn_tt_moe_router {
    uint32_t bank;
    uint32_t address;
    uint16_t x, y;
    uint32_t tile_rows; /**< hidden / 32 */
    uint32_t tile_cols; /**< experts / 32 + 1 */
} pn_tt_moe_router;

/** The block's geometry, fixed at arm. */
typedef struct pn_tt_moe_shape {
    uint32_t hidden_tiles; /**< n_embd / 32 */
    uint32_t experts;      /**< routed experts (256) */
    uint32_t k;            /**< experts a token uses (8) */
    uint32_t expert_tiles; /**< expert ff / 32 (16) */
    uint32_t shexp_tiles;  /**< shared-expert ff / 32 (16) */
    uint32_t stride;       /**< the expert stride every stack shares */
} pn_tt_moe_shape;

/**
 * Hold MoE-block weights out of the dense load (`pn_tt_device_load_artifact` skips them,
 * this module places them). Call before the load.
 */
void pn_tt_moe_defer_weights(pn_tt_device *device, bool defer);

/** Was the dense load told to leave MoE-block weights to this module? */
bool pn_tt_moe_weights_deferred(const pn_tt_device *device);

/**
 * Upload the routed-expert stack `name` of the device's artifact whole into one bank (banks
 * taken in rotation).
 * @return 0, or -1 with the reason on stderr (not a stack, DRAM full, upload failed)
 */
int pn_tt_moe_place_stack(pn_tt_device *device, const char *name, pn_tt_moe_stack *out);

/**
 * Build the router's bf16 tiles from the artifact entry `name` (fp32 or Bfp8_b tiles) plus
 * the shared-expert gate vector, and upload them. Every value must be exactly
 * bf16-representable (spec §13 P5.3); a value that is not is a refusal, not a rounding.
 * @param shexp_gate  `hidden_tiles * 32` floats
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_moe_place_router(pn_tt_device *device, const char *name, const float *shexp_gate,
                           pn_tt_moe_router *out);

/**
 * Boot the router residency (core `PN_TT_MOE_ROUTER_CORE`, default 87) and the expert
 * residencies (`PN_TT_MOE_CORES` of them, default 48, from `PN_TT_MOE_CORE_FIRST`, default
 * 0), and allocate the routing slot, the logits and the scale strip.
 * @return 0, or -1 with the reason on stderr (nothing left armed)
 */
int pn_tt_moe_arm(pn_tt_device *device, const pn_tt_moe_shape *shape);

/** Retire every MoE residency by asking, and free the module's host state. */
void pn_tt_moe_release(pn_tt_device *device);

/** Host wall seconds the expert stacks' fold + upload took (plan S3). */
double pn_tt_moe_fold_seconds(const pn_tt_device *device);

/** Is the block armed? */
bool pn_tt_moe_armed(const pn_tt_device *device);

/** The scale strip the router writes (k * expert_tiles + shexp_tiles bf16 tiles). */
const pn_tt_activation *pn_tt_moe_scale_strip(const pn_tt_device *device);

/**
 * Route one row: the router descriptor over `normed`, awaited. Afterwards the slot and the
 * scale strip hold this row's routing.
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_moe_route(pn_tt_device *device, const pn_tt_moe_router *router,
                    const pn_tt_activation *normed);

/**
 * ISSUE the routed gate and up projections over `normed` into tiles [0, k*expert_tiles) of
 * `gate_out` / `up_out` (rank r at tile r * expert_tiles). Not awaited: pn_tt_moe_await().
 */
int pn_tt_moe_gate_up(pn_tt_device *device, const pn_tt_moe_stack *gate,
                      const pn_tt_moe_stack *up, const pn_tt_activation *normed,
                      const pn_tt_activation *gate_out, const pn_tt_activation *up_out);

/**
 * ISSUE the routed down projection + weighted sum: `scaled` holds the rank-ordered
 * w_r * SwiGLU_r (tiles [0, k*expert_tiles)); `out` receives sum_r over hidden_tiles. Not
 * awaited: pn_tt_moe_await().
 */
int pn_tt_moe_down(pn_tt_device *device, const pn_tt_moe_stack *down,
                   const pn_tt_activation *scaled, const pn_tt_activation *out);

/**
 * BATCHED PREFILL (plan S1): route `bands` (<= FT_TT_MOE_ROWS_BANDS_MAX) tile rows of `normed`
 * (a view at the group's first band) in ROWS mode, awaited: the routing slot then holds the
 * group's LIST (union of chosen experts, ascending) and the scale buffer the group's per-row
 * scale tiles; the shared-expert scale tiles land at pass bands first_band.. .
 */
/** pn_tt_moe_place_stack / _router from the EXTRA artifact (the MTP head's MoE layer). */
int pn_tt_moe_place_extra_stack(pn_tt_device *device, const char *name, pn_tt_moe_stack *out);
int pn_tt_moe_place_extra_router(pn_tt_device *device, const char *name, const float *shexp_gate,
                                 pn_tt_moe_router *out);

int pn_tt_moe_route_rows(pn_tt_device *device, const pn_tt_moe_router *router,
                         const pn_tt_activation *normed, uint32_t bands, uint32_t first_band);

/**
 * Route verify rows 0..rows-1 of ONE band (CONCAT mode, plan S6): the slot gets each row's own
 * top-k in rank order, concatenated (n = rows * k), and the scale tiles of entry r*k + i carry
 * row r's weight in row r only, so each row's routed sum is the decode route's. The shared
 * scale tiles land at pass band 0.
 */
int pn_tt_moe_route_concat(pn_tt_device *device, const pn_tt_moe_router *router,
                           const pn_tt_activation *normed, uint32_t rows);

/**
 * ISSUE the LIST-mode gate and up projections of a routed group: every (list entry, expert
 * column) cell of the `entries` x E grid (256 for a prefill group, rows * k for a CONCAT
 * verify), entries past the list as exact zeros, over `bands` rows as one row block.
 * `gate_out` / `up_out` are at least entries*E tiles wide. Not awaited.
 */
int pn_tt_moe_gate_up_rows(pn_tt_device *device, const pn_tt_moe_stack *gate,
                           const pn_tt_moe_stack *up, const pn_tt_activation *normed,
                           uint32_t bands, uint32_t entries, const pn_tt_activation *gate_out,
                           const pn_tt_activation *up_out);

/**
 * ISSUE the LIST-mode routed down + sum: `scaled` (entries*E tiles wide, the group's scaled
 * SwiGLU outputs) K-concatenated over the `entries` list entries into `out` (a view at the
 * group's first band, n_embd/32 tiles wide). Not awaited.
 */
int pn_tt_moe_down_rows(pn_tt_device *device, const pn_tt_moe_stack *down,
                        const pn_tt_activation *scaled, uint32_t bands, uint32_t entries,
                        const pn_tt_activation *out);

/** The rows-mode scale buffer (resident_matmul_moe_protocol.h layout). */
const pn_tt_activation *pn_tt_moe_scale_buffer(const pn_tt_device *device);

/** Wait for every issued expert descriptor. @return 0, or -1 (the token is failed). */
int pn_tt_moe_await(pn_tt_device *device);

/**
 * Read the routing slot back (gates and diagnostics only; the forward never does).
 * @param out_index   k entries
 * @param out_weight  k entries
 */
int pn_tt_moe_read_slot(pn_tt_device *device, uint32_t *out_index, float *out_weight);

/** Read the router's fp32 logits row back (experts + 32 values; gates only). */
int pn_tt_moe_read_logits(pn_tt_device *device, float *out, uint32_t count);

/** Which MoE residency holds `core`, by name, or NULL. `except` may be NULL. */
const char *pn_tt_moe_core_holder(const pn_tt_device *device, uint32_t core,
                                  const void *except);

/** One line: routes, expert descriptors, host wall in each phase. */
void pn_tt_moe_report(const pn_tt_device *device, double tokens);

#endif /* USE_TT */
#endif /* PN_TT_MOE_H */
