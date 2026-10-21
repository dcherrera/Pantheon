/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_eltwise_split.h
 * @brief The ARITHMETIC of splitting one elementwise descriptor across N Tensix: how many
 *        members a set of widths admits, and which contiguous tile range each member owns.
 *
 * It is a separate module from `pn_tt_eltwise_group.h` for the reason `pn_tt_gdn_layout.h`
 * is separate from `pn_tt_gdn.c`: everything here is pure integer arithmetic over a tile
 * count, so it is provable with no card, no device and no `USE_TT` — and the one thing a
 * split can get wrong that no error will ever report is an address.
 *
 * ---------------------------------------------------------------------------
 * WHY AN ELEMENTWISE OP IS THE CLEANEST SPLIT ON THE CARD
 * ---------------------------------------------------------------------------
 * `resident_eltwise_reader.c` reads tile `t` of a stream at `base + t * TILE_BYTES`,
 * unconditionally and in order — its own header calls this "the easy case", against the
 * matmul reader, which had to argue that a block's B tiles are consecutive only under an
 * N-major weight grid. So member `m` of an N-way split runs the SAME descriptor with
 * `tile_count / N` and every base advanced by `m * (tile_count / N) * tile_bytes`, and the
 * members touch pairwise-disjoint byte ranges of A, of B and of the output. There is no
 * shared operand, no reduction, no ordering: `y = f(a, b)` per tile.
 *
 * ---------------------------------------------------------------------------
 * WHY THE SPLIT MUST DIVIDE EXACTLY, WHERE THE MATMUL GROUP'S MAY BE RAGGED
 * ---------------------------------------------------------------------------
 * `pn_tt_matmul_group_place_weight_columns()` hands the remainder out one column at a time
 * and carries a RUNNING SUM, because its members' widths only have to concatenate. This
 * family cannot do that, and the reason is the circular buffers rather than the answer:
 *
 *  1. **The block-depth invariant is per descriptor.** A residency is placed for ONE depth,
 *     and every descriptor it will ever run must satisfy `2 * depth | tile_count` — the FIFO
 *     write pointer wraps only on an exact hit of the limit, so a descriptor that leaves it
 *     elsewhere walks the NEXT one up L1 with no error (`pn_tt_common_depth()`, and the top-k
 *     reader that landed on the queue control block at 0x170000 without faulting). A ragged
 *     slice is a tile count off that grid on exactly one member.
 *  2. **A set is as slow as its longest member**, and every member is a whole residency
 *     paying a whole descriptor's fixed cost. An uneven split buys the biggest slice's time
 *     and wastes the rest — the same argument `gdn_head_member_count()` makes about v-heads.
 *
 * So the member count is clamped to a COMMON DIVISOR of every width the residency was armed
 * for, and a width that does not divide shrinks the split for all of them rather than
 * splitting some descriptors and not others. One depth, one placement, every member.
 *
 * @status Off-hardware only so far: `tests/test_tt_eltwise_split.c`. The card arm that
 *         consumes it is `pn_tt_eltwise_group.h`.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_ELTWISE_SPLIT_H
#define PN_TT_ELTWISE_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

/** Elementwise residencies one split may hold, member 0 included.
 *
 *  SIXTEEN because that is what Ornith's narrowest elementwise width admits and not because
 *  the card stops there: the residual adds are 128 tiles, and 128/16 = 8 tiles is still two
 *  blocks of depth 4. Thirty-two would leave 4 tiles a member at depth 2, which is a
 *  descriptor that is nearly all fixed cost. Raising it is a measurement and sixteen more
 *  Tensix, not an edit. */
#define PN_TT_ELTWISE_SPLIT_MAX_MEMBERS 16u

/** One member's share of a descriptor: a contiguous tile range and the byte offset every
 *  operand base advances by to reach it. */
typedef struct pn_tt_eltwise_slice {
    uint32_t first_tile;
    uint32_t tile_count;
    uint32_t byte_offset; /**< `first_tile * tile_bytes`, for the operand asked about */
} pn_tt_eltwise_slice;

/**
 * The largest member count at or below `requested` that divides EVERY width exactly.
 *
 * Walks down from the clamp rather than taking a GCD, because the answer wanted is "the
 * biggest split this residency can run", not "the biggest split that exists": `requested` is
 * an operator's knob and the clamp is the array's size.
 *
 * @param requested    what the run asked for; 0 or 1 means no split
 * @param tile_counts  every width a descriptor of this residency will carry
 * @param count        how many of them; 0 refuses
 * @param max_members  the caller's array bound, itself clamped to the module's
 * @return             1..max_members; 1 whenever no larger count divides every width
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_eltwise_split_member_limit(uint32_t requested, const uint32_t *tile_counts,
                                          uint32_t count, uint32_t max_members);

/**
 * Which tiles member `member` of an `members`-way split owns, and at what byte offset.
 *
 * REFUSES rather than rounding when the split does not divide, because the caller that
 * rounds here writes one member's answer over another's — a plausible wrong answer with no
 * error anywhere, which is the failure this whole module exists to make unreachable.
 *
 * @param tile_count  the WHOLE descriptor's tile count
 * @param members     1..PN_TT_ELTWISE_SPLIT_MAX_MEMBERS
 * @param member      0..members-1
 * @param tile_bytes  the operand's page size, for the byte offset
 * @param out_slice   receives the range; untouched on a refusal
 * @return            true, or false when an argument is out of range or the split is ragged
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_eltwise_split_slice(uint32_t tile_count, uint32_t members, uint32_t member,
                               uint32_t tile_bytes, pn_tt_eltwise_slice *out_slice);

/**
 * PHASE W1a — the WIDE split: member `member`'s share of `tile_count` dealt out in whole
 * BLOCKS of `block_tiles` (the residency's `2 * depth`), the first `blocks % members` members
 * one block more. Ragged by blocks and never by tiles, so every slice stays on the CB grid the
 * depth rule protects whatever the member count — which lets a prefill-wide descriptor
 * (Mt x 384 tiles, Mt odd) spread over members the width does not divide.
 *
 * A member past the last block gets an EMPTY slice (tile_count 0) at the end of the
 * descriptor; the caller skips it. The slices tile [0, tile_count) exactly.
 *
 * @return true, or false when an argument is out of range or `tile_count` is not whole blocks
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_eltwise_split_slice_blocks(uint32_t tile_count, uint32_t members, uint32_t member,
                                      uint32_t block_tiles, uint32_t tile_bytes,
                                      pn_tt_eltwise_slice *out_slice);

#endif /* PN_TT_ELTWISE_SPLIT_H */
