/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_render.h
 * @brief THE ACTIVATION STRIP BRIDGE: two resident cores that convert a strip of tiles from
 *        bf16 to fp32 and back, ON THE CARD, so the GDN block's activations stop crossing
 *        PCIe.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS, IN ONE NUMBER
 * ---------------------------------------------------------------------------
 * A decode token is 0.390 s of HOST and 0.050 s of card, and 0.29 s of the host half is
 * TRANSFER. 78% of that transfer is the Gated-Delta-Net block: `ssm_qkv` and `ssm_z` come
 * down and the answer goes back up, **49,408 B a layer over 24 layers**. Nothing about the
 * arithmetic needs the host — what needs it is a FORMAT SEAM. Every projection produces a
 * bf16 strip; the GDN op and the per-head norms are Float32 programs, because the recurrence
 * carries state across tokens and bf16 mantissas compound. So the activation crossed the link
 * purely to change format.
 *
 * These two residencies are that conversion, in card DRAM.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS A SEPARATE PROGRAM AND NOT A MIXED CB TABLE
 * ---------------------------------------------------------------------------
 * The obvious move — give the GDN op a Float16_b CB_IN and read `ssm_qkv` in place — is the
 * one thing the hardware will not do. `emit_tensix_cb_descriptors.py`'s
 * `conditional_destination_format()` is a WHOLE-PROGRAM decision: a Float32 CB unpacks into
 * Tf32 source registers only when `dst_accum_mode` is on AND *every* CB of the program is
 * Float32. One bf16 buffer anywhere in `gdn_float32.cbtable` demotes the recurrent state and
 * the conv weights to Float16_b in SrcA. So the format change has to happen in a program
 * whose arithmetic does not care, and a copy is exactly that program.
 *
 * ---------------------------------------------------------------------------
 * THE OP IS `a + b` WITH b A ZERO STRIP, AND THAT IS WHY NO KERNEL CHANGED
 * ---------------------------------------------------------------------------
 * `ft_tt_op_vector_function` has no identity and adding one would edit
 * `resident_eltwise_compute.c`, which by content provenance makes the shipped `eltwise`
 * binary stale and costs a re-validation of an op that is already correct. `x + 0` is exact
 * in every format involved, and the zero operand is free: every allocator here stages its
 * DRAM zeroed, so the B buffer is allocated once and never written again.
 *
 * The two programs are `render_widen` (bf16 -> fp32) and `render_narrow` (fp32 -> bf16),
 * built from the SAME three kernel sources as `eltwise` with a different `.cbtable` and one
 * different `TILE_BYTES` on each of the two data-movement processors — the `reduce_fp32`
 * precedent exactly. This is the first program in the tree whose reader and writer are built
 * at DIFFERENT page sizes on purpose; each still agrees with its own circular buffer.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE NARROW DIRECTION COSTS, STATED RATHER THAN HIDDEN
 * ---------------------------------------------------------------------------
 * The narrowing happens at the UNPACK, and that unpack TRUNCATES toward zero where the host's
 * `ft_bf16_from_float_array()` rounded to nearest — up to 2x the error on one value per
 * element, on the last conversion before a consumer that would have rounded anyway. It is
 * priced against the OUTPUT: the whole per-head-norm miss, of which this render is the
 * largest term, costs 0.158% perplexity, and a 1e-3 relative bar on an intermediate tensor is
 * refuted here — it fails an arm whose 48-token output is byte-identical to the oracle.
 *
 * ---------------------------------------------------------------------------
 * SPLITTING IT: `PN_TT_RENDER_CORES=N`, AND WHY THE ARITHMETIC IS BORROWED
 * ---------------------------------------------------------------------------
 * These two programs ARE the elementwise program at two page sizes, so they take the
 * elementwise split unchanged: `resident_eltwise_reader.c` reads tile `t` at
 * `base + t * TILE_BYTES`, in order, so member `m` of N runs the SAME descriptor with
 * `tile_count / N` and every operand base advanced by `m * (tile_count/N) * tile_bytes` —
 * pairwise disjoint in A and in the output, with no shared operand, no reduction and no
 * ordering. `pn_tt_eltwise_split.h` holds that arithmetic and is SHARED rather than copied,
 * because two spellings of "which tiles does member m own" is the one thing here that fails
 * with no error anywhere. The two directions are split INDEPENDENTLY: different programs at
 * different page sizes have no reason to share a member count.
 *
 * `reduce` and `head_norm` do NOT take this pattern — both are a row-wise reduction into DST
 * column 0, so a tile slice there needs a cross-core partial-sum combine, a kernel and
 * protocol change rather than a host-side address offset.
 *
 * WHAT A MEMBER COSTS, MEASURED ON SILICON 2026-08-26. A descriptor is worth splitting in two
 * iff its UNSPLIT TIME is above the break-even `K`, and "the knee is a width" was that
 * statement seen through one family's pooled line. Render's two directions, on the two
 * completion shapes:
 *
 *      direction        Wt    window: N=1  N=2   N=4     K       line: N=1   N=2    K
 *      render-widen    256          125.4 105.1 152.7   84.6          110.3  77.1  43.9
 *      render-narrow   128           77.9  82.6 153.3   87.1           62.8    -      -
 *
 * The elementwise family recovers **87.2 +- 1.3 us** on the window and **44.3 +- 1.1** on the
 * line from five shapes; render's directions land on the same numbers from a different
 * program at different page sizes, so **the page size is not in the host term**. At four
 * members both sit on the host's per-member floor (152.7 and 153.3 against 4 x 38.2 us) — the
 * same 38.1-38.4 us a member the elementwise ladder measured flat at N = 4, 8 and 16.
 *
 * On the shipped `line` await BOTH directions are above K, so **the default is {2, 2}**. On
 * the window the narrow direction is below it and loses; the default would be {2, 1} there.
 * K is a property of the AWAIT, not of the family.
 *
 * THE ZERO OPERAND IS NOT SLICED. Every tile of it is the same tile, so a member reading from
 * the base reads exactly what an offset would have given it, and the read is provably inside
 * the allocation even when a caller submits fewer tiles than the direction was armed for.
 *
 * AND EVERY MEMBER IS GIVEN AN EXPLICITLY EMPTY REPLICA TABLE. These cores run
 * `resident_vector_writer.c`, `pn_tt_publish_replica_map()` does not write to this family at
 * all, and the table lives at a FIXED L1 address — so an unpublished residency reads whatever
 * a previous process left there, which is not "no mirroring" but "somebody else's addresses".
 * That is the state that cost the elementwise split its `cmp`. Neither render output is a
 * mirrored strip, so EMPTY is the correct table and not a stub.
 *
 * @status None
 * @issues None
 * @todo
 *   - [ ] An all-Float32 ELTWISE program would let `silu(z) *` run in fp32 so only the final
 *         `gated` is narrowed — modelled at 1.909e-03 against 1.023e-02, a 6.2x. It changes
 *         no bytes over PCIe, so it is an accuracy rung and not part of this bridge.
 */

#ifndef PN_TT_RENDER_H
#define PN_TT_RENDER_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "tt/ft_tt_op_dispatch.h"

/**
 * Arm both directions of the bridge, each placed for the WIDEST strip it will be asked for.
 *
 * Never fatal: a card without the two programs leaves the caller on whatever host path it
 * had. The two cores are `PN_TT_CORE_RENDER_WIDEN` / `PN_TT_CORE_RENDER_NARROW`, and like
 * every other residency here they are refused BY NAME against a live family or a matmul
 * member rather than being allowed to contend — a contested core does not fail, it computes
 * badly.
 *
 * @param widen_tiles   the widest bf16 -> fp32 strip, in tiles; 0 skips that direction
 * @param narrow_tiles  the widest fp32 -> bf16 strip, in tiles; 0 skips that direction
 * @return 0 (even when nothing armed), -1 on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_render(pn_tt_device *device, uint32_t widen_tiles, uint32_t narrow_tiles);

/** Is the bf16 -> fp32 direction live? */
bool pn_tt_render_widen_is_armed(const pn_tt_device *device);

/** Is the fp32 -> bf16 direction live? */
bool pn_tt_render_narrow_is_armed(const pn_tt_device *device);

/**
 * `out = in`, bf16 tiles to fp32 tiles, `tiles` of them, in card DRAM.
 *
 * The operands are taken RAW rather than as a strip type because the three things this bridge
 * connects are three different C types — a `pn_tt_activation`, a `pn_tt_fp32_strip` and a GDN
 * family buffer — and they differ in the page size and in nothing else. The page each side
 * must have is checked here, because a 2048-byte operand under a 4096-byte program is the
 * failure with no card-side refusal anywhere.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_render_widen(pn_tt_device *device, ft_tt_op_operand in, ft_tt_op_operand out,
                       uint32_t tiles);

/** `out = in`, fp32 tiles to bf16 tiles. See the header note on what the narrowing costs. */
int pn_tt_render_narrow(pn_tt_device *device, ft_tt_op_operand in, ft_tt_op_operand out,
                        uint32_t tiles);

/** Release both directions WHOLE — every member of each — and the two zero operands.
 *
 *  A LEAKED TRIAD IS A WEDGED TENSIX for the next process to ask for that core, which is what
 *  the head-norm slots cost when `pn_tt_device_close()` released named fields and missed a
 *  set. This is now two sets. */
void pn_tt_render_release(pn_tt_device *device);

/** The two directions, in the order the device holds them — so a caller outside the backend
 *  can walk them without reaching into `pn_tt_device_internal.h`. */
#define PN_TT_RENDER_WIDEN_INDEX  0u
#define PN_TT_RENDER_NARROW_INDEX 1u
#define PN_TT_RENDER_INDEX_COUNT  2u

/** Live members of direction `index`, member 0 included. 1 when unsplit, **0 when that
 *  direction never armed** — and the caller has to be able to tell those two apart, because
 *  an arm that did not happen and a split that did not happen print the same silence. */
uint32_t pn_tt_render_members(const pn_tt_device *device, uint32_t index);

/**
 * One member's core, descriptor count and AWAIT seconds — the shape of the set.
 *
 * Reported apart from the census because the census keys on the WHOLE descriptor's width by
 * design, which keeps its `fixed + per_tile * tiles` fit comparable across a member ladder.
 * What that therefore cannot show is whether the members are balanced, and "it stopped at N"
 * is exactly the finding that needs it: a knee where every member costs the same is the
 * cores; a knee where one member costs more than the others is that member being starved.
 *
 * @return false when the direction or the member does not exist
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_render_member_stat(const pn_tt_device *device, uint32_t index, uint32_t member,
                              const char **out_name, uint32_t *out_core, uint64_t *out_ops,
                              double *out_seconds);

/** PM 7 ITEM 7: defer the awaits of one bridge direction's descriptors (`on`), and settle
 *  them. See pn_tt_residency_settle(). */
void pn_tt_render_defer(pn_tt_device *device, bool narrow, bool on);
int pn_tt_render_settle(pn_tt_device *device, bool narrow);

#endif /* USE_TT */
#endif /* PN_TT_RENDER_H */
