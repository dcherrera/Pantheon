/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_strip_placement.h
 * @brief Bank placement for a SET of activation strips on a bump allocator: mirrored strips
 *        first, then each unmirrored strip in the emptiest bank. Plus a dry run of exactly
 *        that policy against per-bank free bytes, so a fit is settled with no card.
 *
 * WHY THE ORDER MATTERS. A strip mirrored into every bank has one copy pinned to each bank,
 * with no fallback (pn_tt_oncard.c). The DRAM allocator is a bump allocator that frees LIFO
 * only, so nothing placed earlier can be moved out of the way. Qwen3.8-27B at 16k failed its
 * prefill tall set on exactly that: unmirrored strips that overflowed went to other banks,
 * filled bank 1, and the attention `gated` mirror 6 that must live in bank 1 found 20.8 MiB
 * free against 24 MiB, with 1.870 GiB free on the card. Placing every mirror first and then
 * steering the unmirrored strips to the banks with room removes that failure mode
 * (option B of the Qwen3.8 fit).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_STRIP_PLACEMENT_H
#define PN_TT_STRIP_PLACEMENT_H

#include <stddef.h>
#include <stdint.h>

/** Where a set's copy-0 allocations go (pn_tt_strip_placement_simulate). */
typedef enum pn_tt_strip_policy {
    /** main's order: every strip's copy 0 at the round-robin bank, else the emptiest that fits
     *  (`pn_tt_dram_alloc_any()`), in the set's input order. */
    PN_TT_STRIP_POLICY_ROUND_ROBIN = 0,
    /** Option B: mirrored strips (in `order`, mirrors first) round-robin as above; each
     *  unmirrored strip in the emptiest bank that fits. */
    PN_TT_STRIP_POLICY_MIRRORS_FIRST = 1,
} pn_tt_strip_policy;

/** One strip of a set: its width and how many banks hold a copy (1 = unmirrored). */
typedef struct pn_tt_strip_request {
    uint32_t tile_count;
    uint32_t replicas;
} pn_tt_strip_request;

/**
 * The allocation order for a strip set: every mirrored strip (replicas > 1), then every
 * unmirrored one, each group in its input order.
 *
 * @param out_order  receives `count` indices into `requests`
 * @return           indices written (`count`), or 0 on bad arguments
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
size_t pn_tt_strip_placement_order(const pn_tt_strip_request *requests, size_t count,
                                   size_t *out_order);

/**
 * The bank with the most free bytes that can still hold `bytes`; the lowest index wins a tie.
 *
 * @return the bank, or `bank_count` when no bank can hold it
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_strip_placement_emptiest_bank(const uint64_t *free_bytes, uint32_t bank_count,
                                             uint64_t bytes);

/**
 * Dry-run the policy the card uses, in `order`, against `free_bytes` (updated in place).
 *
 * A mirrored strip's copy 0 goes to the round-robin bank `*cursor` (advanced once per strip),
 * or the emptiest bank that fits when that one is full — `pn_tt_dram_alloc_any()`. Copy k
 * goes to `(copy 0's bank + k) % bank_count` with no fallback. An unmirrored strip goes to the
 * emptiest bank that fits under MIRRORS_FIRST, and round-robins like a mirror's copy 0 under
 * ROUND_ROBIN. A strip of 0 tiles is skipped (it gets no copy).
 *
 * The card's sizes are whole 64 B multiples, so with the allocator's real per-bank free bytes
 * and strip cursor this is EXACT, not an estimate: it is how the load decides whether main's
 * placement fits before allocating anything.
 *
 * @param bytes_per_tile  card bytes of one tile column of a strip (tile rows x 2048)
 * @param cursor          the round-robin strip cursor, advanced as the card's is
 * @param out_failed      on failure, the index into `requests` that did not fit
 * @return                0 when every strip fits, -1 otherwise
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_strip_placement_simulate(const pn_tt_strip_request *requests, const size_t *order,
                                   size_t count, pn_tt_strip_policy policy,
                                   uint64_t bytes_per_tile, uint64_t *free_bytes,
                                   uint32_t bank_count, uint32_t *cursor, size_t *out_failed);

#endif /* PN_TT_STRIP_PLACEMENT_H */
