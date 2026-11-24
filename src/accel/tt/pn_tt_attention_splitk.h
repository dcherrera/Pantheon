/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_splitk.h
 * @brief SPLIT-K DECODE ATTENTION on the card: one decode
 *        or verify-rows call's per-KV-group flash-decode cut into position chunks over every
 *        split core, then one MERGE a group.
 *
 * The plan (which chunks, which core, which partial slot) is pn_tt_attn_splitk.h, pure and
 * host-tested. This file arms the cores and issues it:
 *
 *   1. round(s) of PARTIAL descriptors, chunk d of group g on split core d % cores: the lanes
 *      request for that group with the chunk's first tile folded into the K and V bases, its
 *      chunk-relative live length and capacity, the output pointed at partial slot d, and
 *      (ROWS) the staircase mask only on the last chunk, the all-zero tile on the others;
 *   2. every partial awaited, then one MERGE per group (group g on core g % cores) reading its
 *      chunks' contiguous slots and writing the ordinary output — layout, and so the sigmoid
 *      gate after it, unchanged.
 *
 * THE CORES: the attention residency, its head-split members, and the split-only extras from
 * `PN_TT_ATTN_SPLITK_CORE_LIST` (a comma list, default `87,128`; empty = none; a held core
 * is refused by name and skipped).
 * With split-K armed EVERY one of them runs `attention_k8v4_split`: its decode, ROWS and
 * CHUNK descriptors are the shipped kernel's source (the split code is all under
 * -DATTN_SPLITK), so lanes below the switch-over and prefill run on it too.
 *
 * NOT BIT-IDENTICAL to lanes when a call splits (the sums are reordered; foundry
 * test_attention_splitk.c bounds the model error): gated on the card by KL / flips (plan
 * 3.3). `PN_TT_ATTN_SPLITK=0` arms nothing here, and every call is the lanes path.
 *
 * @status On by default; K8V4 caches only (`attention_k8v4_split`). Card gate plan 3.3
 *         passed: KL max 4.4e-3, argmax 100%, needle 3/3 at 8k/16k/28k, +12% decode at 28k.
 *         Red `attn_splitk_drop_chunk`: each merge folds one chunk fewer — must DIFFER.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ATTENTION_SPLITK_H
#define PN_TT_ATTENTION_SPLITK_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_attn_splitk.h"
#include "pn_tt_device.h"
#include "pn_tt_residency.h"
#include "tt/ft_tt_op_attention.h"

/** The program every attention core runs: the KV type's, or its `_split` build when
 *  split-K is armed, or that program's fast-math variant when `PN_TT_ATTN_FAST` chose one. */
const char *pn_tt_attention_program_name(const pn_tt_device *device);

/**
 * At arm, after pn_tt_attention_splitk_choose and before any attention core: set
 * `device->attn_fast_program` to the `PN_TT_ATTN_FAST` variant of the program the run would
 * otherwise arm (pn_tt_attn_fast.h). Unset / 0 leaves it NULL (the shipped program). A
 * refusal (unknown value, no variant, variant not in the program root) leaves it NULL too and
 * says why on stderr. Applies to decode attention as well: one program per load.
 */
void pn_tt_attention_fast_choose(pn_tt_device *device);

/** At arm, before any attention core: set `device->attn_splitk` unless `PN_TT_ATTN_SPLITK=0`,
 *  when the cache is K8V4 and `attention_k8v4_split` is in the program root; says why if not. */
void pn_tt_attention_splitk_choose(pn_tt_device *device);

/**
 * After the head-split members: arm the extras and allocate the partial buffer and the zero
 * mask tile. A failure leaves the split program armed (it serves lanes) but issues nothing
 * split, and says so.
 */
void pn_tt_attention_splitk_arm(pn_tt_device *device);

/** Release the extras. */
void pn_tt_attention_splitk_release(pn_tt_device *device);

/** Split cores armed: the members plus the extras; 0 when split-K is not issuing. */
uint32_t pn_tt_attention_splitk_cores(const pn_tt_device *device);

/**
 * Issue one call split, when the plan says so.
 *
 * @param requests       the lanes path's per-group requests (decode or ROWS), unencoded
 * @param groups         KV groups (requests)
 * @param shortest_live  the shortest row's causal limit (= kv_positions for one decode row)
 * @return 1 when the call is not split (the caller sends the lanes descriptors), 0 when it
 *         ran split, -1 on a failure
 */
int pn_tt_attention_splitk_issue(pn_tt_device *device, const ft_tt_op_attention_request *requests,
                                 uint32_t groups, uint32_t shortest_live);

/**
 * The split-K plan's shape key (pn_tt_attn_splitk_plan_key) for ONE decode row at
 * `live_positions` (the token's position + 1), as pn_tt_attention_decode would plan it: 0
 * when split-K is not issuing or the depth stays on lanes. The decode forward hands it to
 * pn_tt_dispatch_shape_key before each token, so a replay never crosses a plan change.
 */
uint64_t pn_tt_attention_splitk_shape_key(const pn_tt_device *device, uint32_t live_positions);

/* Defined in pn_tt_attention.c: arm one attention residency on `core` with the flash-decode
 * residency's shape and program. */
bool pn_tt_arm_attention_member(pn_tt_device *device, pn_tt_residency *member, uint32_t core);

#endif /* USE_TT */
#endif /* PN_TT_ATTENTION_SPLITK_H */
