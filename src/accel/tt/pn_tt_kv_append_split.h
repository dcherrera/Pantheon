/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_append_split.h
 * @brief THE PREFILL KV APPEND SPLIT: a band's K and V appends packed on up to eight cores at
 *        once instead of one after the other on the memory core.
 *
 * WHY. A block-float cache (`PN_TT_KV_TYPE` bfp8 / bfp4 / k8v4 ...) is packed by the memory
 * writer itself (NCRISC integer code, foundry `ft_kv_block_float.h`): ~865 us of the memory
 * core a 32-tile append, so a 2,845-token K8V4 prefill spends 2.49 s on that one core and its
 * TTFT is +15% over bf16. MEASURED on the card's idle clock (`PN_TT_IDLE_CLOCK=1`, 2026-11-13):
 * the memory core's busy grew 0.02 -> 2.49 s and no other family's moved. The host's own
 * per-family seconds said 133 us an append and ~5x that "after" it — a streamed prefill answers
 * the host's polls at once and the host blocks only when a stream region frees, inside
 * whatever call it is capturing, so those seconds are not card time.
 *
 * HOW. `pn_tt_kv_append_words_per_group()` already encodes one descriptor a KV bank group
 * (one KV head at the default G = 4), each with its own group's source tiles and region —
 * the unbanked fallback, the same words that path sends. K's G descriptors and V's G go to
 * the members round-robin (`pn_tt_kv_append_member_of()`), all are enqueued, then each member
 * is awaited once. The writer code and the bytes it writes are unchanged; only which core
 * runs which head changes. Nothing else writes a band's tiles, and the next reader (the
 * band's attention) is issued after every member is awaited.
 *
 * Decode (`pn_tt_kv_append`, `_row`) and a bf16 cache never use it: member 0 is the memory
 * residency itself and decode stays there. Members claim L1 on their own cores only; no DRAM.
 *
 * Knobs: `PN_TT_KV_APPEND_CORES` (total cores incl. the memory core; default 8 for a
 * block-float cache, 1 = off; a bf16 cache is always 1). `PN_TT_KV_APPEND_CORE_LIST=a,b,...`
 * (the members' cores in order; default 67,68,69,70,121,126,127 — free in the Qwen3.8 map
 * with and without MTP / prefix checkpoint; 87 is left out because MoE's router and the gate
 * default to it). Members arm after every attention-family residency; a held core is refused
 * by name (`pn_tt_prepare_family_on_core()`) and the next listed core is tried.
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifndef PN_TT_KV_APPEND_SPLIT_H
#define PN_TT_KV_APPEND_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

#define PN_TT_KV_APPEND_CORES_ENV          "PN_TT_KV_APPEND_CORES"
#define PN_TT_KV_APPEND_CORE_LIST_ENV      "PN_TT_KV_APPEND_CORE_LIST"
#define PN_TT_KV_APPEND_CORES_DEFAULT      8u
#define PN_TT_KV_APPEND_CORE_LIST_DEFAULT  "67,68,69,70,121,126,127"

/** Total cores (memory core included) the split asks for: 1 for a bf16 cache, else
 *  `env_value` when it is a positive number, else the default — clamped to `max_members`. */
uint32_t pn_tt_kv_append_members_wanted(bool block_float, const char *env_value,
                                        uint32_t max_members);

/** The `position`-th (0-based) core of a comma list (`list` NULL or empty: the default list);
 *  UINT32_MAX past its end or at a malformed entry. */
uint32_t pn_tt_kv_append_list_core(const char *list, uint32_t position);

/** The member that runs descriptor `descriptor` of a pair (K's groups then V's): round-robin,
 *  so a pair of 2G descriptors over 2G members puts every head of K and V on its own core. */
uint32_t pn_tt_kv_append_member_of(uint32_t descriptor, uint32_t members);

#ifdef USE_TT

#include "pn_tt_device_internal.h"
#include "pn_tt_group_pull.h"
#include "pn_tt_kv_banks.h"
#include "tt/ft_tt_op_memory.h"

/* From pn_tt_attention.c, which owns the memory family. */
/** Load, place and start the `memory` program on `core` as split member `member`. */
bool pn_tt_arm_memory_member(pn_tt_device *device, pn_tt_residency *member, uint32_t core);
/** Retire and re-arm the memory family (member 0) on its own core. */
bool pn_tt_attention_memory_rearm(pn_tt_device *device);
/** The request `pn_tt_kv_append_chunk()` encodes (chained projections awaited first). */
int pn_tt_kv_append_chunk_request(pn_tt_device *device, uint32_t layer, bool values,
                                  const pn_tt_activation *source, uint32_t first_position,
                                  uint32_t rows, ft_tt_op_kv_append_request *out);

/** Arm members 1.. on the listed cores (the memory family must be up). Decides the count from
 *  the cache's formats; a member that does not come up is skipped, never fatal. */
void pn_tt_kv_append_members_arm(pn_tt_device *device);

/** Release every member (device teardown). */
void pn_tt_kv_append_members_release(pn_tt_device *device);

/** The descriptors a split pair sends: K's per-group words, then V's (production's encoder,
 *  `pn_tt_kv_append_words_per_group()`). @return their count; 0 on a refusal. */
uint32_t pn_tt_kv_append_pair_words(pn_tt_device *device, uint32_t layer,
                                    const pn_tt_activation *k_source,
                                    const pn_tt_activation *v_source, uint32_t first_position,
                                    uint32_t rows,
                                    uint32_t words[2u * PN_TT_KV_MAX_BANK_GROUPS]
                                                  [FT_TT_OP_MEMORY_WORDS]);

/**
 * S1: the a3/a4/a5 descriptor `index` of a pulled `count`-descriptor split append carries
 * (pn_tt_group_pull.h; dealt by pn_tt_kv_append_member_of()). Only each member's LAST
 * descriptor posts `sequence`; an earlier one posts `sequence - 1`, which this split's pull
 * does not accept. Member 0's last descriptor pulls every member that received one.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
pn_tt_group_pull_words pn_tt_kv_append_pull_words(uint32_t index, uint32_t count,
                                                  uint32_t members, uint32_t sequence);

/**
 * Append `rows` positions of K (`k_source`) AND V (`v_source`) of `layer` at `first_position`
 * (a multiple of 32), split over the members; returns once every member has finished. With
 * one member it is `pn_tt_kv_append_chunk()` for K then for V, exactly as before.
 */
int pn_tt_kv_append_chunk_pair(pn_tt_device *device, uint32_t layer,
                               const pn_tt_activation *k_source,
                               const pn_tt_activation *v_source, uint32_t first_position,
                               uint32_t rows);

#endif /* USE_TT */
#endif /* PN_TT_KV_APPEND_SPLIT_H */
