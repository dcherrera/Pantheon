/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_prefill_reserve.h
 * @brief Per-bank DRAM need of the buffers a prefill used to allocate AFTER the load: counted
 *        at load against the allocator's real free bytes, so a context that cannot run refuses
 *        at load instead of failing its first prefill.
 *
 * WHY. The DRAM allocator is a bump allocator and most of these buffers are PINNED to a bank
 * (they share a packed NoC coordinate word with a buffer already placed). On Qwen3.8-27B at
 * 34,816 the load succeeded with 0.004 GiB left in its fullest bank, the first prefill then
 * failed a 393,216-byte GDNC state-split buffer in that bank, and the serve kept answering
 * over a half-written carried state. Counting
 * them per bank at load turns that into a refusal with the shortfall named.
 *
 * Two classes:
 *   - MANDATORY: what every batched prefill takes (the GDNC state-split slices, the PIPE 2
 *     alternate streams, the tall-pass rope tables). Short -> the load refuses.
 *   - OPTIONAL: the multi-turn prefix checkpoint's twins of the carried state. Short -> the
 *     load serves with the checkpoint OFF (resumed turns re-prefill), said once, with the
 *     shortfall; nothing is half-allocated and no prefill retries it.
 *
 * No device here: every function is arithmetic, so the policy is tested on any host
 * (`tests/test_tt_q38_reserve.c`).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_PREFILL_RESERVE_H
#define PN_TT_PREFILL_RESERVE_H

#include <stdbool.h>
#include <stdint.h>

/** Most DRAM banks a plan counts (a p150 has 8). */
#define PN_TT_FIT_MAX_BANKS 16u

/** The allocator's DRAM alignment: every reservation is rounded up to it. */
#define PN_TT_FIT_DRAM_ALIGN 64u

/** Bytes needed in each bank, plus bytes that may go in any bank (round-robin allocations). */
typedef struct pn_tt_bank_need {
    uint32_t bank_count;
    uint64_t bank[PN_TT_FIT_MAX_BANKS];
    uint64_t any;      /**< total of the any-bank allocations */
    uint64_t any_max;  /**< the largest single any-bank allocation */
} pn_tt_bank_need;

/** An empty need over `bank_count` banks (clamped to PN_TT_FIT_MAX_BANKS). */
void pn_tt_bank_need_init(pn_tt_bank_need *need, uint32_t bank_count);

/** Add `bytes` (rounded up to PN_TT_FIT_DRAM_ALIGN) pinned to `bank`. */
void pn_tt_bank_need_pin(pn_tt_bank_need *need, uint32_t bank, uint64_t bytes);

/** Add one round-robin allocation of `bytes` (rounded up) that may land in any bank. */
void pn_tt_bank_need_any(pn_tt_bank_need *need, uint64_t bytes);

/** Total bytes of `need`. */
uint64_t pn_tt_bank_need_total(const pn_tt_bank_need *need);

/**
 * Does `need` fit `free_bytes`? Every bank must hold its pinned bytes; the any-bank bytes must
 * fit in what is left over all banks, and the largest one must fit in one bank's remainder.
 *
 * @param short_bank  on failure, the first bank whose pinned need exceeds its free bytes, or
 *                    `bank_count` when the any-bank part is what does not fit
 * @param short_bytes on failure, by how much it is short
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_bank_need_fits(const pn_tt_bank_need *need, const uint64_t *free_bytes,
                          uint32_t *short_bank, uint64_t *short_bytes);

/** Subtract a need that fits from `free_bytes` (any-bank bytes from the emptiest banks). */
void pn_tt_bank_need_take(const pn_tt_bank_need *need, uint64_t *free_bytes);

/**
 * The GDNC state-split slices (`pn_tt_gdnc.c` gdnc_state_split_alloc): for every layer, one
 * slice of `slice_bytes` per DELTA member m in bank `(state_bank[layer] + 1 + m) % banks`.
 */
void pn_tt_bank_need_state_split(pn_tt_bank_need *need, const uint32_t *state_bank,
                                 uint32_t layers, uint32_t members, uint64_t slice_bytes);

/**
 * The prefix checkpoint's twins (`pn_tt_gdn_alloc_twin`): per layer, a window of
 * `window_bytes` in `window_bank[layer]` and a state of `state_bytes` in `state_bank[layer]`.
 */
void pn_tt_bank_need_twins(pn_tt_bank_need *need, const uint32_t *window_bank,
                           const uint32_t *state_bank, uint32_t layers, uint64_t window_bytes,
                           uint64_t state_bytes);

#endif /* PN_TT_PREFILL_RESERVE_H */
