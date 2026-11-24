/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_swiglu_fuse.h
 * @brief S3b, SwiGLU in the gate/up members: the knob, the program it arms, and the
 *        descriptor words of a fused gate/up member descriptor
 *        (foundry tt/kernels/resident_matmul_swiglu_protocol.h).
 *
 * `PN_TT_FUSE_SWIGLU=1` arms the matmul group with `rtriad_bfp8_swiglu` instead of
 * `rtriad_bfp8` and turns a decode / verify-row FFN's gate, up and silu_mul (16 + 16 member
 * descriptors and 4 eltwise ones) into 16 member descriptors that each read the member's gate
 * AND up panel and write silu(gate) * up into `ffn_swiglu` and its mirrors. ON by default
 * (card-proven exact on the L2 base: decode, k1 verify, 28k, 60-min soak); 0 is today's
 * program, placement and descriptors, byte for byte.
 *
 * Exactness: the fused compute reproduces gate -> bf16, up -> bf16, eltwise SILU_MUL's own
 * sequence (silu in fp32 DST, staged bf16, MUL at HiFi4, bf16 out), so `ffn_swiglu`'s live
 * rows are the unfused bytes.
 *
 * REDS (PN_TT_RED): `fuse_swiglu_no_stage` keeps silu fp32 into the multiply (must DIFFER by
 * bytes); `fuse_swiglu_mirror_skip` leaves one mirror out of the members' replica table
 * (pn_tt_replica_map.h) so the down projection reads a stale copy (must DIFFER).
 *
 * @status Added 2026-11-17; card gate passed on the
 *         L2 base. On by default with q38dec P5.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_SWIGLU_FUSE_H
#define PN_TT_SWIGLU_FUSE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** `PN_TT_FUSE_SWIGLU`: 0 today's path, 1 (default) fuse decode and verify-row SwiGLU. */
#define PN_TT_FUSE_SWIGLU_ENV "PN_TT_FUSE_SWIGLU"

/** The matmul group's resident program when the knob is off. */
#define PN_TT_RESIDENT_PROGRAM_PLAIN "rtriad_bfp8"
/** ... and when it (or S3a's PN_TT_FUSE_RESADD, pn_tt_resadd_fuse.h) is on: rtriad_bfp8's
 *  recipe plus -DFT_RTRIAD_SWIGLU=1 -DFT_RTRIAD_RESADD=1. */
#define PN_TT_RESIDENT_PROGRAM_SWIGLU "rtriad_bfp8_swiglu"

/**
 * The knob, read once per process (the residency is armed once per process shape).
 *
 * @return 0 off, 1 on
 *
 * Thread-safe: no (first call caches)    Reentrant: no    Signal-safe: no
 */
int pn_tt_fuse_swiglu_mode(void);

/**
 * True when this process arms the fused program: PN_TT_FUSE_SWIGLU or PN_TT_FUSE_RESADD on.
 * The fused program's plain matmuls are the shipped matmul's code, so a knob that is on but
 * never fires changes nothing but the program name and the extra CBs.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_fused_program_armed(void);

/**
 * The resident matmul program this process arms: PN_TT_RESIDENT_PROGRAM_SWIGLU when
 * pn_tt_fused_program_armed(), PN_TT_RESIDENT_PROGRAM_PLAIN otherwise. One function, so the
 * program the device opens and the one the DST-section interlock reads are the same name.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
const char *pn_tt_resident_program_name(void);

/**
 * The eltwise family's silu row-0 rule (`PN_TT_ELTWISE_ROW0`: 0 never, 1 a single-token
 * forward or a verify of <= 4 rows, 2 always), shared by pn_tt_eltwise() and the fused
 * descriptor so the two silu passes cannot disagree.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_eltwise_silu_row0(const pn_tt_device *device);

/**
 * True when `gate` and `up` are both split over the same members with the same shard map
 * (member m owns the same columns of both), so one descriptor per member can carry both.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_tt_swiglu_shards_pair(const pn_tt_weight *gate, const pn_tt_weight *up);

/**
 * The word-15 bits of a fused descriptor (ORed over the block word).
 *
 * @param silu_row0     silu on row 0 only (pn_tt_eltwise_silu_row0)
 * @param red_no_stage  RED fuse_swiglu_no_stage
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_swiglu_block_bits(bool silu_row0, bool red_no_stage);

/**
 * Descriptor a1/a2 of member `member`'s fused descriptor: the up shard's base and bank door.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_swiglu_up_words(const pn_tt_weight *up, uint32_t member, uint32_t *a1,
                           uint32_t *a2);

#endif /* USE_TT */
#endif /* PN_TT_SWIGLU_FUSE_H */
