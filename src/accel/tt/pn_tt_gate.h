/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gate.h
 * @brief THE GDN OUTPUT GATE WITH THE fp32->bf16 SEAM FOLDED INTO IT — one descriptor that
 *        replaces `render_narrow` + `silu_mul`, reading the fp32 `y_norm` strip DIRECTLY.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT DELETES, AND WHY THAT IS FREE
 * ---------------------------------------------------------------------------
 * Today the GDN block's last two steps are:
 *
 *     render_narrow(y_norm_fp32) -> y_bridge_bf16        24 descriptors/token
 *     silu_mul(z_bf16, y_bridge_bf16) -> gated_bf16      24 descriptors/token
 *
 * The first exists ONLY to change format. It reads 128 fp32 tiles, reads 128 MORE fp32 tiles
 * of ZEROS (the family has no identity, so a copy is `x + 0`), and writes 128 bf16 tiles —
 * 1,310,720 B a layer, 31,457,280 B a token, to move no information at all.
 *
 * `gate_narrow` is the SECOND descriptor with a MIXED CB table: cb 1 is Float32 while cb 0,
 * cb 2 and cb 16 stay Float16_b. `conditional_destination_format()` then gives cb 1
 * `unpack_src Float32 / unpack_dst Float16_b / tile_size 4096` — verified off-hardware to be
 * the SAME descriptor triple `render_narrow.cbtable` gives its own fp32 input. So the
 * unpacker performs the identical truncating fp32->bf16 conversion, on the identical bits, one
 * descriptor and one DRAM round trip earlier than the multiply that consumes it. **The
 * arithmetic is the same and the operand the multiplier sees is the same bits.** That is why
 * this is expected to be `cmp` IDENTICAL and why anything else is a finding.
 *
 * The one thing that had to change is `resident_eltwise_reader.c`, which carried ONE
 * `TILE_BYTES` for both input streams. It now takes a fifth compile-time arg for B, refuses a
 * zero on a binary descriptor rather than reading 0 bytes a tile, and every pre-existing
 * recipe passes its own A size — so `eltwise`, `render_widen` and `render_narrow` rebuild
 * byte-identical, which is the control that says the new argument is inert when unused.
 *
 * ---------------------------------------------------------------------------
 * WHY NOT ON THE SHARED ELEMENTWISE RESIDENCY
 * ---------------------------------------------------------------------------
 * A residency runs ONE program, and a program's CB formats are baked into its compute images.
 * The shipped `eltwise` residency serves the residual ADDs, the SwiGLU and attention's
 * SIGMOID+MUL, all of which are bf16 on both inputs; giving it a Float32 cb 1 would make every
 * one of them read a bf16 strip as fp32. So the gate gets its own core, and the descriptor
 * COUNT does not change by moving it — 24 gate descriptors were already being issued, they
 * are simply issued to a different residency. What goes is the 24 renders.
 *
 * Its width is Wt=128, which is BELOW the measured elementwise split break-even (~143 tiles at
 * the first doubling; a 128-tile MUL measured 101.7 -> 146.9 us, WORSE, at two members). So a
 * solo residency is not a compromise here, it is the right side of that knee.
 *
 * ---------------------------------------------------------------------------
 * WHY NOT THE GDN WRITER — REFUTED, DO NOT RETRY
 * ---------------------------------------------------------------------------
 * `gdn_float32.cbtable`'s cb 16 carries the UPDATED RECURRENT STATE as well as `y`, interleaved
 * per tile-row. One CB, one page size, two sinks: narrowing it to bf16 narrows the state the
 * next token reads. The seam cannot move upstream of the gated RMS norm.
 *
 * @status DEFAULT OFF. Flipped ON 2026-08-25 and flipped BACK the same day. In DECODE it is
 *   right and cheap — `cmp` IDENTICAL, 20,448 -> 19,296 descriptors a token, -25,165,824 B,
 *   on-card 0.114 -> 0.112 s/token against a control that read 0.114 twice. With the CHUNKWISE
 *   PREFILL armed it puts `gdnc` and `render-narrow` into `did not complete: error` and the
 *   prefill silently declines to the per-token path. See pn_tt_gate.c for the two arms that
 *   show it and the one that does not.
 * @issues Breaks `gdnc` when the chunkwise prefill is armed. Undiagnosed; not a core collision.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GATE_H
#define PN_TT_GATE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "tt/ft_tt_op_dispatch.h"

/** The flag. Default OFF; `1` arms the merged gate and takes `render_narrow` out of decode.
 *  DO NOT flip the default again without a chunkwise-prefill arm beside the decode one. */
#define PN_TT_GATE_NARROW_ENV "PN_TT_GATE_NARROW"

/**
 * Arm the merged gate for the widest `gated` strip it will be asked for.
 *
 * Never fatal: without the `gate_narrow` program, or with the flag unset, the caller keeps the
 * render + silu_mul pair it had. The core is `PN_TT_CORE_GATE_NARROW` (87 by default — free
 * between the render pair at 85/86 and the GDN residency at 90) and is refused BY NAME against
 * any live family, exactly like every other residency here.
 *
 * @param tiles  the widest bf16 output strip, in tiles; 0 skips
 * @return 0 (even when nothing armed), -1 on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_gate_narrow(pn_tt_device *device, uint32_t tiles);

/** Is the merged gate live? A caller that gets `false` must use render_narrow + silu_mul. */
bool pn_tt_gate_narrow_is_armed(const pn_tt_device *device);

/**
 * `out = silu(a) * b`, with `a` and `out` bf16 tiles and `b` FP32 tiles.
 *
 * The operands are raw for the same reason `pn_tt_render`'s are: the three buffers this joins
 * are three different C types differing only in page size. Each side's page size is checked
 * here, because a 2048-byte operand under a 4096-byte stream is the failure mode with no
 * card-side refusal anywhere (measured on the reduce family: constants built at 2048 under a
 * 4096 program score 1.0e+00 with nothing refusing).
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_gate_narrow(pn_tt_device *device, ft_tt_op_operand a_bf16, ft_tt_op_operand b_fp32,
                      ft_tt_op_operand out_bf16, uint32_t tiles);

/** Release the residency. */
void pn_tt_gate_release(pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_GATE_H */
