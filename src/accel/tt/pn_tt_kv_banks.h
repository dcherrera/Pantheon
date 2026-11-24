/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_banks.h
 * @brief KV BANK GROUPS (2026-10-26): a layer's K region (and V region) split by KV head
 *        into `G` groups, each group in its own DRAM bank, instead of all heads in one bank.
 *
 * WHY. The prefill attention chunk mixer runs one descriptor per Q head over 8 cores, and
 * every descriptor of a layer reads that layer's K from ONE bank and V from ONE other: at
 * 28.7k the band loop fell 36.3 -> 22.5 s from 4 to 8 cores and ROSE to 23.3 s at 16 —
 * a shared ceiling, not per-core work. With G = 4 (Ornith's 4 KV heads) the K and V reads
 * of a layer spread over 8 banks.
 *
 * WHAT CHANGES. Only addresses: each head's position tiles are the same bytes in the same
 * order (ft_tt_kv_cache layout, head-major), so every reader and writer moves the same data.
 * Readers address a head as `region_operand(layer, values, head) + kv_cache_run(head, ...)`
 * exactly as before: the operand of head h carries `base_of_group(h) + (h - first(h)) *
 * stride - h * stride` (uint32 wrap), so the full layout's `h * stride` lands in the group.
 * The append stays ONE descriptor over every head: each layer's K (and V) groups are
 * allocated at one common address in their banks and the memory kernel's BANKED
 * destination (foundry FT_TT_OP_MEMORY_FLAG_DST_BANKED) sends group g to its own bank.
 * `PN_TT_KV_APPEND_BANKED=0` (or G > 4) falls back to one descriptor a group.
 *
 * @status Default on since 2026-10-26: `PN_TT_KV_BANK_GROUPS=<G>` (default 4, or the
 *         largest of 2 / 1 that divides the KV heads; `=1` one bank a region; G divides the
 *         KV heads, <= PN_TT_KV_MAX_BANK_GROUPS). Red `kv_bank_group_shift`.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_KV_BANKS_H
#define PN_TT_KV_BANKS_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device_internal.h"
#include "tt/ft_tt_op_memory.h"

/** K's layout, or V's (`values`): the same geometry, V's own tile in a mixed cache. */
const ft_tt_kv_cache_layout *pn_tt_kv_layout_of(const pn_tt_device *device, bool values);

/** Choose G from the env and build the group layouts. Call after `kv_layout` and
 *  `kv_v_layout` are set. */
int pn_tt_kv_banks_configure(pn_tt_device *device);

/**
 * Allocate (zeroed) every layer's K and V groups, bank by bank in rotation. A MIXED cache (K
 * and V tiles differ) puts each layer's larger region on the less-used bank set, so the
 * banks stay level instead of the K banks carrying every layer's K.
 */
bool pn_tt_kv_banks_alloc(pn_tt_device *device, uint32_t layer_count);

/**
 * Pure: does a layer allocate V's groups first? `first_set_top` / `second_set_top` are the
 * fullest bank's cursor of the next bank set and of the one after it (K's and V's in K-then-V
 * order). Only a MIXED cache (group sizes differ) ever takes V first: when that puts the
 * larger region on the emptier set. A uniform cache keeps K then V, as every earlier build.
 */
bool pn_tt_kv_layer_v_first(uint64_t k_group_bytes, uint64_t v_group_bytes,
                            uint64_t first_set_top, uint64_t second_set_top);

/** Head `head`'s region operand: add `ft_tt_kv_cache_run(pn_tt_kv_layout_of(values), head,
 *  ...)`'s offset, or use pn_tt_kv_head_operand(). */
ft_tt_op_operand pn_tt_kv_region_operand(const pn_tt_device *device, uint32_t layer,
                                         bool values, uint32_t head);

/** Head `head`'s K (or V) cache of `layer` from position 0, as an attention operand. */
int pn_tt_kv_head_operand(const pn_tt_device *device, uint32_t layer, bool values,
                          uint32_t head, ft_tt_op_operand *out);

/** The bank head `head`'s K (or V) of `layer` lives in. */
uint32_t pn_tt_kv_head_bank(const pn_tt_device *device, uint32_t layer, bool values,
                            uint32_t head);

/**
 * Encode `request` (built against the full layout, region unset) into `words`: ONE banked
 * descriptor when `kv_append_banked`, else one per bank group; `*out_count` says how many.
 */
int pn_tt_kv_append_words(const pn_tt_device *device, const ft_tt_op_kv_append_request *request,
                          uint32_t layer, bool values,
                          uint32_t words[PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS],
                          uint32_t *out_count);

/** The UNBANKED form whatever `kv_append_banked` says: one descriptor a bank group, each its
 *  own group's source tiles and region (the prefill KV append split runs one a core). */
int pn_tt_kv_append_words_per_group(const pn_tt_device *device,
                                    const ft_tt_op_kv_append_request *request, uint32_t layer,
                                    bool values,
                                    uint32_t words[PN_TT_KV_MAX_BANK_GROUPS][FT_TT_OP_MEMORY_WORDS],
                                    uint32_t *out_count);

#endif /* USE_TT */
#endif /* PN_TT_KV_BANKS_H */
