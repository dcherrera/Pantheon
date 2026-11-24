/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_head_norm.h
 * @brief PER-HEAD norms on EXTRA reduce residencies placed for the HEAD width — the norms
 *        the Gated-Delta-Net block and full attention have and the model's RMS norms do not.
 *
 *        Model-agnostic: it knows a head count and a head width. It does not know what a
 *        Qwen3.5 layer is.
 *
 * ---------------------------------------------------------------------------
 * ONE RESIDENCY PER WIDTH, AND A HYBRID MODEL NEEDS TWO
 * ---------------------------------------------------------------------------
 * A reduce residency is placed for ONE Wt and refuses any other (next section), so "the head
 * width" is not one number for a GDN+attention hybrid. Ornith:
 *
 *     GDN per-head norms   S = 128          Wt 4     PN_TT_CORE_HEAD_NORM     default 91
 *     attention QK-norms   head_dim = 256   Wt 8     PN_TT_CORE_HEAD_NORM_2   default 103
 *     the model's own RMS  n_embd = 4096    Wt 128   the `reduce` residency   default 72
 *
 * So this is a SET keyed by width, not a singleton. `pn_tt_device_arm_head_norm()` is
 * idempotent per width and takes one more Tensix per new one; every other entry point routes
 * on the width, and the width comes from the CONSTANTS the caller passes rather than from an
 * argument — a constants buffer is built FOR one width, and a reduce run at another width
 * would read the wrong gamma out of it and return a smoothly wrong norm.
 *
 * ---------------------------------------------------------------------------
 * WHY A SEPARATE RESIDENCY, AND WHY THAT IS THE WHOLE OF THE COST
 * ---------------------------------------------------------------------------
 * A reduce residency is PLACED for one Wt: `ft_tt_op_reduce_place_cbs()` sizes CB_IN,
 * CB_MID, CB_GAMMA and CB_BETA at `max_tile_cols` pages, and `pn_tt_rms_norm()` refuses a
 * descriptor whose Wt is not the one it was placed for. The model's norms are
 * `n_embd/32 = 128` tile columns; the GDN's per-head norms are `S/32 = 4`. So they get
 * their own core, and there are 140.
 *
 * The two widths are not merely different, they must not SHARE: `CB_GAMMA` is
 * `max_tile_cols` pages and every descriptor pushes `tile_cols` of them in ONE go, so a
 * Wt=4 descriptor leaves the write pointer at page 4 of 128 and the next Wt=128 descriptor
 * straddles the limit. `ft_tensix_cb_push_grid_is_safe()` catches it and REFUSES — a
 * re-arm, not a wrong answer — but the refusal would land on one descriptor in two.
 *
 * ---------------------------------------------------------------------------
 * THE LAYOUT RESULT THIS FILE EXISTS TO USE, AND IT COST NOTHING
 * ---------------------------------------------------------------------------
 * **A per-head norm over a ROW-VECTOR activation strip needs no reshape at all.** The
 * scoping for this item assumed the head norms wanted a `[heads, S]` matrix — one row per
 * head — which every tensor around them is not: `pn_tt_activation_alloc()` strips are
 * `[1, 32*tile_count]`, row 0 of each tile and 31 rows of padding.
 *
 * `resident_reduce_reader.c` reads row block `b` at tiles `b*tile_cols + wt`, consecutive,
 * and `row_blocks = tile_count / tile_cols`. So over a `[1, heads*S]` strip with
 * `tile_cols = S/32`:
 *
 *     row block h  ==  tiles [h*S/32, +S/32)  ==  head h's S elements, all in ROW 0
 *
 * and each row block is reduced PER ROW. One descriptor, `heads` independent norms, the
 * data exactly where the projection that produced it left it. The `[heads, S]` matrix
 * layout would have needed a transpose of 32-element runs out of row 0 of `4*heads` tiles;
 * this needs nothing.
 *
 * The consequence is that `gamma` is shared across heads, which is what the GDN's output
 * gated norm wants — `ssm_norm` is `S` values applied identically to every v-head — and
 * the q/k L2 norm has no gamma at all.
 *
 * ---------------------------------------------------------------------------
 * THE TWO MEMBERS
 * ---------------------------------------------------------------------------
 *   q/k L2 norm    `FT_TT_OP_REDUCE_FN_L2_NORM`, no gamma, eps passed as `eps*eps`
 *                  (foundry `ft_tt_op_vector.h` has why). Pantheon's oracle is
 *                  `pn_qwen35moe_l2_norm()`.
 *   output gated   `FT_TT_OP_REDUCE_FN_RMS_NORM` with gamma = `ssm_norm`, then
 *                  `pn_tt_silu_mul(z, normed)` on the ELEMENTWISE residency. Pantheon's
 *                  oracle is `pn_qwen35moe_gated_norm()`, which is exactly those two steps.
 *   attention QK   `FT_TT_OP_REDUCE_FN_RMS_NORM` with gamma = `attn_q_norm` / `attn_k_norm`,
 *                  at the ATTENTION head width, on the second slot. Pantheon's oracle is
 *                  `qwen35moe_qk_norm_heads()` in `qwen35moe_attn.c`, which is
 *                  `pn_vrms_norm_f32` per head with one shared weight — the same "gamma is
 *                  shared across heads" the row-block layout gives for free.
 *
 * **THE QK MEMBER IS THE RMS MEMBER, AND THE RMS MEMBER IS THE ONE THAT MISSES THE GATE.**
 * At Wt 4 it measured 3.165e-03 against 1e-3 (below), and the cause is the chain STEP COUNT
 * rather than the width, so Wt 8 inherits it. That is a numerics change to attention's
 * scores, not a rounding detail, and it is why on-card attention stays opt-in.
 *
 * ---------------------------------------------------------------------------
 * THE RESIDENCY IS Float32, AND THAT IS WHY IT WORKS AT ALL
 * ---------------------------------------------------------------------------
 * The FIRST arming of this residency ran the shipped bf16 `reduce` program and MISSED THE
 * GATE BY 9x: `PN_TT_GDN_NORMCHECK=1` on Ornith's real per-layer data scored the q/k L2 norm
 * at **8.599e-03** max-norm relative per head and the output gated norm at **1.283e-02**
 * against 1e-3, with 146,844 of 196,608 q/k elements out.
 *
 * **The error is per REDUCE, so it is the narrow norms that cannot afford it.** The model's
 * own RMS norms average the same per-buffer rounding over 4096 elements and pass; a per-head
 * norm has 128. And the lever is the CB TABLE and not the activation strip: priced apart
 * off-hardware in `foundry/tt/tests/test_vector_ops.c` on this exact shape, the bf16 REDUCE
 * alone is 6.126e-03 while rendering an fp32 operand into a bf16 strip adds only 1.0e-03, so
 * an fp32 strip in front of a bf16 reduce would have bought nothing.
 *
 * So this residency arms `reduce_fp32` — `reduce_float32.cbtable` at `dst_accum_mode 1`, a
 * `TILE_BYTES=4096` build of the SAME three kernel sources — and everything it touches is
 * fp32 by consequence: `pn_tt_fp32_strip` in and out, and a constants buffer built at
 * `PN_TT_TILE_FP32`. There is no fallback to the bf16 program: at this width that would be a
 * wrong answer rather than a slow one.
 *
 * WHAT THAT DOES NOT BUY, AND IT IS NOT ENOUGH FOR ONE OF THE TWO MEMBERS. A Float32 CB is
 * not fp32 arithmetic: the generated `chlkc_descriptors.h` for this program carries
 * `unpack_src_format = 0` (Float32) beside `unpack_dst_format = 4`, and 4 is **Tf32** — E8M10,
 * ten stored mantissa bits, TRUNCATED rather than rounded, ~4.9e-4 an operand and biased low.
 * `math_fidelity` cannot recover it; the bits are gone before the FPU sees the operand.
 *
 * Modelled off-hardware at this shape (`foundry/tt/tests/test_vector_ops.c`) the two members
 * land on OPPOSITE SIDES of the gate:
 *
 *     q/k L2   bf16 7.143e-03  ->  Float32 8.406e-04   CLEARS 1e-3
 *     RMS      bf16 8.101e-03  ->  Float32 1.566e-03   DOES NOT
 *
 * **The difference is the CHAIN LENGTH, not the member.** The L2 member's final chain is one
 * step; the RMS member's is two — gamma, then scale — and every chain step is a circular-
 * buffer round trip, so every step is one more Tf32 unpack of the value. The same program run
 * with an fp32 unpack the hardware does not have scores 1.0e-07 and 2.1e-07, so the ENTIRE
 * Float32 residue is truncation: no CB table moves it and the next lever is the chain's step
 * count, which is `resident_reduce_compute.c` and not a table.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT MEASURED ON SILICON, AND NEITHER NORM CLEARS
 * ---------------------------------------------------------------------------
 * `PN_TT_GDN_NORMCHECK=1` on Ornith's real per-layer data, 48 SSM layers, twice, identical
 * both times, 0 refusals:
 *
 *                    bf16 `reduce`   Float32 `reduce_fp32`   gate
 *     q/k L2           8.599e-03           1.737e-03         1e-3    4.95x better, 1.7x out
 *     RMS alone            -               3.165e-03         1e-3    3.2x out
 *     gated            1.283e-02           1.023e-02         1e-3    1.25x better, 10x out
 *     q/k elements out  146,844/196,608     25,718/196,608            5.7x fewer
 *
 * **The Float32 program is a real 4.95x on the q/k norm and it is not enough.** Three things
 * are named by those three rows and they are different problems:
 *
 *  1. **Tf32 is the floor on the reduce**, and the modelled 8.406e-04 / 1.566e-03 land about
 *     2x optimistic against the card. The model uses libm for the `rsqrt`; the card uses the
 *     SFPU's polynomial, which is a SECOND approximation the model cannot have and the
 *     leading candidate for that 2x. At bf16 it was invisible under a rounding 8x bigger.
 *  2. **The RMS member pays two more Tf32 unpacks than the L2 member** — 3.165e-03 against
 *     1.737e-03 — because its chain has two steps and every step is a CB round trip.
 *  3. **`gated` barely moved because the reduce is not what bounds it.** It is `silu(z) *`
 *     the RMS answer on the **bf16** ELEMENTWISE residency, so the fp32 answer is rendered
 *     into a bf16 strip before the gate sees it. Fixing the reduce cannot help `gated` until
 *     that bridge goes, which is the qkv/z strip item and not this one.
 *
 * @status RUN ON SILICON 2026-08-19, p150a, core 91 — armed, 96 descriptors over two
 *         tokens, 0 refusals, 0 restarts, core `instrn_buf_status` 0x77 after. **THE LAYOUT
 *         RESULT ABOVE IS CONFIRMED**: `row_blocks = heads` over a `[1, heads*S]` strip
 *         scores 8.6e-03 against the oracle, not the O(1) a wrong axis would give.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_HEAD_NORM_H
#define PN_TT_HEAD_NORM_H

#ifdef USE_TT

#include <stdint.h>

#include "pn_tt_device.h"
#include "pn_tt_fp32_strip.h"
#include "tt/ft_tt_op_vector.h"

/**
 * Arm a per-head reduce residency (one more core) placed for `head_width`.
 *
 * IDEMPOTENT PER WIDTH: asking twice for the same width is a no-op, and asking for a second
 * width takes a second core. A model with two head widths calls this twice.
 *
 * Never fatal: a card without the `reduce_fp32` program, a core already held, or no slot
 * left leaves that width's per-head norms on the host and
 * `pn_tt_head_norm_is_armed(device, width)` false.
 *
 * @param head_width  the head width in ELEMENTS, a multiple of 32
 * @return            0 (even when nothing armed), -1 on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_head_norm(pn_tt_device *device, uint32_t head_width);

/**
 * The same, naming the PAGE SIZE — which is also which reduce program arms.
 *
 * `FT_TT_OP_TILE_BYTES_FP32` is `reduce_fp32` and what `pn_tt_device_arm_head_norm()` asks
 * for; `FT_TT_OP_TILE_BYTES_BF16` is the shipped `reduce`.
 *
 * **bf16 IS THE RIGHT ANSWER FOR ATTENTION'S QK-NORM AND THE WRONG ONE FOR THE GDN'S.** The
 * GDN norms sit in the middle of an fp32 recurrence, so the Tf32 floor is the best available
 * and bf16 misses the 1e-3 gate by 9x. Flash decode is bf16 END TO END by construction — the
 * KV cache is bf16 (`ft_tt_op_memory.h` §1.6), the constants are bf16, the RoPE table is
 * bf16 — so a Float32 QK-norm would need a bf16->fp32 render before it and an fp32->bf16
 * render after, NEITHER OF WHICH EXISTS on the card, and the answer would then be rounded
 * straight back to bf16 by the very next op. It would buy precision the next descriptor
 * discards, at the cost of the strip bridge the GDN norms are already blocked on.
 *
 * A width already placed at the other page is a REFUSAL, not a second slot: routing is by
 * width, and two slots at one width would be a coin toss.
 *
 * @param tile_bytes  `FT_TT_OP_TILE_BYTES_FP32` or `FT_TT_OP_TILE_BYTES_BF16` — foundry's
 *                    own names, because the page size is foundry's contract and a second
 *                    spelling of it here would be a second thing to keep in step
 * @return            0 (even when nothing armed), -1 on a bad argument or a page clash
 */
int pn_tt_device_arm_head_norm_at(pn_tt_device *device, uint32_t head_width,
                                  uint32_t tile_bytes);

/** Did a per-head reduce residency for THIS width arm for this run?
 *  `head_width` in ELEMENTS; 0 asks whether ANY did. */
bool pn_tt_head_norm_is_armed(const pn_tt_device *device, uint32_t head_width);

/**
 * Upload one member's constants buffer: the scaler tile, the eps tile, then gamma expanded
 * to full tiles when the member takes one.
 *
 * @param function  which member the constants are FOR — the scaler and the eps tile both
 *                  depend on it, so a buffer built for one member is wrong for another
 * @param gamma     `width` values, or NULL for a member with no gamma
 * @param width     the head width in ELEMENTS, which must be the armed one
 * @param eps       what the member's reduce should see added; **L2 norm takes `eps*eps`**
 * @return          0, or -1 with the reason on stderr
 */
int pn_tt_head_norm_constants_upload(pn_tt_device *device, ft_tt_op_reduce_function function,
                                     const float *gamma, uint32_t width, float eps,
                                     pn_tt_norm_constants *out_constants);

/**
 * One per-head norm over a row-vector strip: `heads` independent reduces of `width`
 * elements each, in ONE descriptor.
 *
 * `input` and `output` are **Float32** strips — this residency is placed by
 * `reduce_float32.cbtable` and its reader issues 4096-byte reads. Each must hold at least
 * `heads * width / 32` tiles, and they MAY be the same strip — the reader has read a row block before the writer can write it, and the
 * blocks do not alias.
 *
 * Which residency runs it is chosen by `constants->tile_cols`, so a caller cannot point a
 * width at the wrong core: the constants and the placement are the same width by
 * construction, and a width nothing armed is a refusal rather than a wrong reduce.
 *
 * @param heads  row blocks, i.e. how many heads this descriptor norms
 * @return       0, or -1 with the reason on stderr
 */
int pn_tt_head_norm(pn_tt_device *device, ft_tt_op_reduce_function function,
                    const pn_tt_norm_constants *constants, uint32_t heads,
                    const pn_tt_fp32_strip *input, const pn_tt_fp32_strip *output);

/** The same over ordinary bf16 activation strips, for a slot armed at `PN_TT_TILE_BF16`.
 *  Handing either entry point the other page's operands is REFUSED — the page size is a
 *  three-sided contract and the card cannot see a violation of it. */
int pn_tt_head_norm_bf16(pn_tt_device *device, ft_tt_op_reduce_function function,
                         const pn_tt_norm_constants *constants, uint32_t heads,
                         const pn_tt_activation *input, const pn_tt_activation *output);

/**
 * The same over RAW OPERANDS, for a caller whose buffers are neither of the two strip types.
 *
 * The GDN block's are: `conv_out` and `y` are allocated by the GDN family and addressed by its
 * own `ft_tt_op_operand` accessors, and the whole point of the on-card chain is that the norms
 * read them WHERE THEY LIE instead of through a host round trip. An fp32 strip and a GDN
 * buffer differ in nothing this function needs except the address, and the page each must
 * carry is still checked against the SLOT's.
 *
 * `input_tiles` / `output_tiles` are what the caller is willing to have read and written; the
 * `heads * tile_cols` this descriptor actually touches must fit inside both.
 *
 * **INPUT AND OUTPUT MAY BE THE SAME BUFFER.** The GDN q/k norm runs IN PLACE over the first
 * `2*k_heads*state_tiles` tiles of the conv output, and that is safe by pipeline order rather
 * than by luck: block `b` has been read before the writer can write block `b`, and the reader
 * only ever reads AHEAD — under P3 up to four blocks ahead (pn_tt_norm_schedule.h), all of
 * them not yet written. Nothing here enforces it — it is stated so that an in-place caller is
 * a decision rather than an accident.
 */
int pn_tt_head_norm_operands(pn_tt_device *device, ft_tt_op_reduce_function function,
                             const pn_tt_norm_constants *constants, uint32_t heads,
                             ft_tt_op_operand input, ft_tt_op_operand output,
                             uint32_t input_tiles, uint32_t output_tiles);

/** PM 7 ITEM 7: defer the awaits of the `tile_cols`-wide head-norm slot (`on`), and settle
 *  them. See pn_tt_residency_settle(). */
void pn_tt_head_norm_defer(pn_tt_device *device, uint32_t tile_cols, bool on);
int pn_tt_head_norm_settle(pn_tt_device *device, uint32_t tile_cols);

#endif /* USE_TT */
#endif /* PN_TT_HEAD_NORM_H */
