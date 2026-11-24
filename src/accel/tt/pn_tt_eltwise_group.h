/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_eltwise_group.h
 * @brief SPLIT ONE ELEMENTWISE DESCRIPTOR ACROSS N TENSIX. Arms `PN_TT_ELTWISE_CORES - 1`
 *        extra `eltwise` residencies and gives each a contiguous, disjoint slice of every
 *        descriptor's tile stream. `pn_tt_eltwise_split.h` holds the arithmetic.
 *
 * ---------------------------------------------------------------------------
 * WHY: FOUR FAMILIES, ONE CEILING, AND IT IS ONE CORE
 * ---------------------------------------------------------------------------
 * Measured on silicon 2026-08-24, with the host out of the token: eltwise **4.43 GB/s**,
 * render 3.65, reduce ~3.8, topk 6.09 — against the **28.3 GB/s a single core's own
 * drain-only arm measured**, and 510.8 for the card. Four independent families sitting 7x
 * below ONE core's transaction wall and 120x below the part. The bind is the per-tile path
 * on one Tensix; it is neither the byte count nor the descriptor count.
 *
 * Meanwhile the matmul runs 8 members and the GDN head split runs 4 (measured 2.70x at 4).
 * The elementwise family ran ONE — 136 descriptors a token, 27,648 tiles, 0.0374 s, **19.7%
 * of a 0.197 s token** — and nothing had ever been split across cores here. It is the
 * cleanest parallelism on the card: `y = f(a, b)` per tile, no reduction, no ordering.
 *
 * ---------------------------------------------------------------------------
 * WHAT A MEMBER IS, AND WHAT IT IS NOT
 * ---------------------------------------------------------------------------
 * A member is a WHOLE `eltwise` residency on its own Tensix, running the same program at the
 * same circular-buffer placement, and its descriptor is the caller's with three bases
 * advanced and the tile count divided. That means:
 *
 *  - **NO BUFFER IS SHARED.** Every member has its own L1 and its own CBs. The one-producer
 *    rule (`resident_queue_protocol.h` rule 1) is about a CB with two writers; there is no
 *    such CB here, because there is no such CB between any two cores.
 *  - **NO DRAM BYTE IS SHARED.** Member `m` reads `[m*W, (m+1)*W)` of A and of B and writes
 *    the same range of the output, for `W = tile_count / N`. Pairwise disjoint by
 *    construction and asserted off-hardware in `tests/test_tt_eltwise_split.c`.
 *  - **NO KERNEL CHANGES AND NO NEW OPCODE.** The descriptor is the shipped
 *    `FT_TT_QUEUE_OP_ELTWISE` with different words. `resident_queue_protocol.h` is a
 *    content-provenance input to EVERY embedded image, so an opcode registered there would
 *    make all 23 programs stale; nothing is registered there.
 *
 * ---------------------------------------------------------------------------
 * THE CORES, AND WHY THIS SET CANNOT COLLIDE
 * ---------------------------------------------------------------------------
 * Member 0 is `device->eltwise` on `PN_TT_CORE_ELTWISE` (73). Members 1..N-1 run upward from
 * `PN_TT_ELTWISE_MEMBER_CORE_FIRST`, default **106**, and 106 is chosen as the first index
 * ABOVE EVERY CORE ANY FAMILY CAN REACH AT ANY LEGAL SETTING OF ANY KNOB:
 *
 *     71        matmul member 0          72/73/74   reduce / eltwise / topk
 *     75..105   matmul members 1..31     82/83/84   rope / memory / attention
 *                (PN_TT_MATMUL_MAX_CORES = 32, first = 75)
 *     85/86     render widen / narrow    90         gdn member 0
 *     91, 103   the two head-norm slots  92..99     gdn head-split members (max 8)
 *     95        gdnc (chunkwise)         100..102   the rectangle arm's control
 *     ---------------------------------------------------------------------
 *     106..121  ELTWISE members 1..15    122..139   free
 *
 * That is a claim about the WHOLE map and not about today's defaults, which is the lesson
 * the GDN family paid for: its 75 was clear of every other FAMILY and collided with matmul
 * member 1 the day `PN_TT_MATMUL_CORES` moved, for 1560 mutual re-arms over 65 forwards with
 * a healthy "residency up" logged each time. **A contested core does not fail; it computes
 * badly** — 1.0e-2 contested against 6.9e-4 clean on the same descriptor.
 * `pn_tt_prepare_family_on_core()` refuses a collision BY NAME and is still the guard; this
 * default is only the first place to look, and `pn_tt_core_holder()` now walks these members
 * too, so a later family that lands on one is told.
 *
 * ---------------------------------------------------------------------------
 * THE KNOBS
 * ---------------------------------------------------------------------------
 *   `PN_TT_ELTWISE_CORES=N`              members, 1 = the pre-split path byte for byte (4)
 *   `PN_TT_ELTWISE_MEMBER_CORE_LIST=a,b` the cores of members 1..N-1 (112,48,49)
 *   `PN_TT_ELTWISE_MEMBER_CORE_FIRST=C`  members 1..N-1 on C, C+1, ... (used when no list)
 *   `PN_TT_ELTWISE_DEPTH=D`              CAP on the block depth, for the knee control below
 *
 * `PN_TT_ELTWISE_DEPTH` exists because **the depth MOVES WITH THE MEMBER COUNT and would
 * otherwise be a confound in the sweep.** A residency is placed for one depth that must
 * satisfy `2 * depth | tile_count` for every width it serves; Ornith's widths are 128 / 256 /
 * 384, so unsplit it is 32 and at 8 members the slices are 16 / 32 / 48 and it is 8. A
 * member-count ladder therefore also walks a depth ladder, and "it stopped at N" would not
 * say which one stopped. `PN_TT_ELTWISE_DEPTH=8` at `PN_TT_ELTWISE_CORES=1` is the
 * discriminating arm: one core, the split's depth, no split.
 *
 * ---------------------------------------------------------------------------
 * THE ONE THING THE SPLIT BREAKS, AND IT IS NOT THE ARITHMETIC
 * ---------------------------------------------------------------------------
 * **A SLICE MOVES `C_BASE`, AND THE RESIDENT REPLICA TABLE IS KEYED ON THE EXACT TRIPLE.**
 * `resident_vector_writer.c` mirrors an output strip by looking `(kVecCX, kVecCY, kVecCBase)`
 * up in a table, and `ft_tt_op_replicas.h` says a miss "is the shipped path word for word".
 * So member 0 hit the table and mirrored its slice, and every member above it MISSED and
 * left its slice stale in seven banks — which the matmul members read. Every kernel retired
 * clean and nothing refused, exactly as that header predicts a bad map behaves.
 *
 * Measured, and settled by a DISCRIMINATING NULL rather than by reading the code: with
 * mirrors ON the split is `cmp` IDENTICAL at one member and 204 B / 8 B / 119 B / 57 B at
 * 2 / 4 / 8 / 16 — wrong in proportion to the `(N-1)/N` of the strip that misses. With
 * `PN_TT_ACTIVATION_REPLICAS=1` and no table to miss it is **IDENTICAL at 1, 2 AND 4**. The
 * slice arithmetic was never the problem.
 *
 * `pn_tt_publish_replica_map()` now takes the split family's output strip and registers ONE
 * SET PER SLICE, and publishes to the members as well — they were being armed with no table
 * at all, which is not "no mirroring" but "whatever that L1 region held".
 *
 * @status SHIPPED ON at `PN_TT_ELTWISE_CORES=4` (2026-10-21, under the on-card puller: MTP
 *         decode +2.3%, IDENTICAL), members on 112, 48, 49 — the map above predates PX7 and
 *         the prefill grid; the live map is the "residency up" lines. Was 2 while the host
 *         issued every descriptor. `pn_tt_eltwise_group.c` carries the ladder and the fit.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_ELTWISE_GROUP_H
#define PN_TT_ELTWISE_GROUP_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "tt/ft_tt_op_vector.h"

#include "pn_tt_eltwise_split.h"

typedef struct pn_tt_device pn_tt_device;

/** Elementwise residencies one split may hold, member 0 included. The device's array holds
 *  one fewer, exactly as the GDN head split's does. */
#define PN_TT_ELTWISE_MAX_MEMBERS PN_TT_ELTWISE_SPLIT_MAX_MEMBERS

/** THE FAMILY'S PLACEMENT, in one place because two files now arm it. `pn_tt_oncard.c` arms
 *  member 0 and this module arms the rest, and both re-arm on a refusal — a second spelling
 *  of the program name or the core default would be two things that must agree. */
#define PN_TT_ELTWISE_PROGRAM      "eltwise"
#define PN_TT_CORE_ELTWISE_ENV     "PN_TT_CORE_ELTWISE"
#define PN_TT_CORE_ELTWISE_DEFAULT 73u

/** W1a WIDE members: `PN_TT_ELTWISE_WIDE_CORES` more eltwise residencies from
 *  `PN_TT_ELTWISE_WIDE_CORE_FIRST` (58) that only the prefill SwiGLU's wide descriptors use.
 *  DEFAULT 5 (58..62) since 2026-10-26 (was 13, W1a 2026-10-08; 12 armed, 58..69, at the
 *  16-member cap with 4 decode members): 63..70 go to the prefill attention members 8..15
 *  (pn_tt_attention.c). `=13` restores the old layout, and the attention split then defaults
 *  back to 8. `=0` off. */
#define PN_TT_ELTWISE_WIDE_CORES_ENV      "PN_TT_ELTWISE_WIDE_CORES"
#define PN_TT_ELTWISE_WIDE_CORES_DEFAULT  5u
#define PN_TT_ELTWISE_WIDE_CORE_ENV       "PN_TT_ELTWISE_WIDE_CORE_FIRST"
#define PN_TT_ELTWISE_WIDE_CORE_DEFAULT   58u
/** The first core NOT in the wide eltwise block (58..70 is the clear region it draws from). */
#define PN_TT_ELTWISE_WIDE_REGION_END     71u

/** How many WIDE members the env asks for (default `PN_TT_ELTWISE_WIDE_CORES_DEFAULT`). */
uint32_t pn_tt_eltwise_wide_core_count(void);

/** The first WIDE member's core (default `PN_TT_ELTWISE_WIDE_CORE_DEFAULT`). */
uint32_t pn_tt_eltwise_wide_core_first(void);

/**
 * Decide the split for `tile_counts` BEFORE anything is placed, and report it.
 *
 * It runs before the placement rather than after because **the placement is sized from the
 * SLICED width**: the input CBs are two blocks of the effective depth, and the depth is
 * chosen from the tile counts descriptors will actually carry. Deciding the member count
 * afterwards would place member 0 for a width no descriptor has.
 *
 * The member count is shrunk until it divides every width AND the sliced widths admit a
 * common block depth. A shrink is reported, because a silent one shows up only as a number
 * that did not move.
 *
 * @param device       for the env knobs only; nothing is armed here
 * @param tile_counts  every width a descriptor of this residency will carry
 * @param count        how many of them
 * @param out_members  receives 1..PN_TT_ELTWISE_MAX_MEMBERS; never NULL
 * @param out_widest   receives the widest SLICE, for `place_cbs`; never NULL
 * @param out_depth    receives the ONE block depth every member runs; 0 when none serves
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_tt_eltwise_group_plan(const pn_tt_device *device, const uint32_t *tile_counts,
                              uint32_t count, uint32_t *out_members, uint32_t *out_widest,
                              uint32_t *out_depth);

/**
 * Arm members 1..`members`-1 on their own cores, with member 0's placement.
 *
 * They take the IDENTICAL circular-buffer placement member 0 has, and that is a property of
 * the family rather than a convenience: `ft_tt_op_eltwise_place_cbs()` reads the effective
 * depth and the three page sizes and nothing else, so a member owning 16 tiles places
 * exactly what a member owning 128 does.
 *
 * **A MEMBER THAT FAILS TO ARM COLLAPSES THE SPLIT TO ONE**, loudly, rather than shrinking
 * it the way the GDN head split does. A shrink there is safe because any head count divides;
 * here the widths must divide the member count exactly, and 8 members shrinking to 5 is a
 * ragged slice on a 128-tile residual — a descriptor the kernel refuses. Unsplit is always
 * correct and always available: member 0's placement serves the whole width too, since
 * `2 * depth` divides a slice and the slice divides the width.
 *
 * @param device     the device, with `device->eltwise` already started
 * @param members    what `pn_tt_eltwise_group_plan()` returned
 * @param placement  the request member 0 was placed with
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_tt_eltwise_group_arm(pn_tt_device *device, uint32_t members,
                             const ft_tt_op_eltwise_request *placement);

/**
 * Run ONE elementwise descriptor across the whole set: ALL ENQUEUED BEFORE ANY IS AWAITED.
 *
 * The two loops being separate is the measurement and not a style. Enqueue-wait per member
 * would run `members` SERIAL descriptors and print the flat line "it does not parallelise",
 * produced by this function rather than by the card — the same trap `gdn_submit_delta_split()`
 * and the core-scaling probe both call out.
 *
 * Time is charged to the SET, once, at the WHOLE descriptor's tile count, so the census key
 * `(eltwise, function, tile_count)` is the same row at every member count and the
 * `fixed + per_tile * tiles` fit is directly comparable across the ladder.
 *
 * @param device   the device
 * @param request  the WHOLE descriptor, already validated by the caller
 * @return         0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_eltwise_group_submit(pn_tt_device *device, const ft_tt_op_eltwise_request *request);

/**
 * PM 7: the NEXT pn_tt_eltwise_group_submit() enqueues and returns WITHOUT awaiting; the
 * descriptor stays pending until pn_tt_eltwise_group_settle() or the next eltwise descriptor.
 * Streamed (the prefill band loop), the puller's AWAIT then lands where the settle is, so the
 * chain's next ops run under it. The caller owns the hazard: settle before anything reads the
 * output or rewrites an operand. `PN_TT_ELTWISE_DEFER=0` makes this a no-op.
 */
void pn_tt_eltwise_group_defer_next(pn_tt_device *device);

/** Descriptors deferred so far (the report's proof the deferral is live). */
uint64_t pn_tt_eltwise_group_deferred(const pn_tt_device *device);

/** Await (and replay, if refused) the pending deferred descriptor. @return 0 when none. */
int pn_tt_eltwise_group_settle(pn_tt_device *device);

/** Release members 1..N-1 and set the count back to 1. Safe on a device that never split.
 *
 *  A LEAKED TRIAD IS A WEDGED TENSIX for the next process to ask for that core, which is
 *  what the head-norm slots cost when `pn_tt_device_close()` released named fields and
 *  missed a set. This is a set; it is released by name from that function. */
void pn_tt_eltwise_group_release(pn_tt_device *device);

/** Live members, member 0 included. 1 when unsplit.
 *
 *  The per-member ledger the report reads is `pn_tt_device_eltwise_member_stat()` in
 *  `pn_tt_device.h` — public, because the model's report is what consumes it. */
uint32_t pn_tt_eltwise_group_members(const pn_tt_device *device);

/** W1a: how many members a descriptor of `tile_count` tiles runs over — every member (decode
 *  + wide) for a wide descriptor, the decode split otherwise. */
uint32_t pn_tt_eltwise_group_members_for(const pn_tt_device *device, uint32_t tile_count,
                                         uint32_t function);

/** Member `member`'s slice of a `tile_count`-tile descriptor run over `members` — the block
 *  split when `members` is the full wide set, the equal split otherwise. The replica map
 *  keys each member's mirrors from this, so the writer and the map cannot disagree. */
bool pn_tt_eltwise_group_slice(const pn_tt_device *device, uint32_t tile_count,
                               uint32_t members, uint32_t member, uint32_t tile_bytes,
                               pn_tt_eltwise_slice *out_slice);

#endif /* USE_TT */
#endif /* PN_TT_ELTWISE_GROUP_H */
