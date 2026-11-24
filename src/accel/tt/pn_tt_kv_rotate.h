/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_kv_rotate.h
 * @brief K4H: an online Hadamard rotation of Q and K after RoPE, so a Bfp4 K cache quantizes
 *        rotated rows (`PN_TT_KV_TYPE=k4h`: K Bfp4_b rotated, V Bfp4_b).
 *
 * WHY. q.k = (qH).(kH) for any orthogonal H. A Bfp4 face row shares one power-of-two exponent
 * over 16 channels of one position, and a value below ~1/8 of the group max rounds to zero;
 * H spreads every channel over the whole head, so a group's values are near-Gaussian with an
 * even scale (QuaRot / SpinQuant R3). Nothing can fold it into the weights on this model: the
 * QK-norm gamma and the partial RoPE sit between the projection and the cache.
 *
 * HOW. H is the 256x256 Sylvester Hadamard over sqrt(256): entries +-1/16, exact in bf16 and
 * in every block-float weight format (a face row of +-1/16 is one exponent, mantissa 1.0).
 * It is applied by the EXISTING resident matmul, with no new kernel: a q_rot / k_rot strip is
 * `heads * head_tiles` consecutive tiles in one bank, which is exactly the tile-row-major
 * matrix `[heads x head_tiles]` (Mt = heads, Kt = head_tiles), so one matmul by H rotates
 * every head of every row at once. A decode strip's row 0 is the token in every view tile, so
 * the matmul's row-0 mode stays valid. Attention runs the unchanged `attention_bfp4` program.
 *
 * H is symmetric and H*H = I, so the inverse is H: session export un-rotates K and import
 * re-rotates it, and the state file stays in the canonical frame of every other KV type.
 *
 * @status Opt-in (`PN_TT_KV_TYPE=k4h`).
 * @issues None
 * @todo Fuse into the RoPE program if the two matmuls a layer cost more than they should.
 */
#ifndef PN_TT_KV_ROTATE_H
#define PN_TT_KV_ROTATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** The one head width the rotation is built for (Qwen3.8: 256 = 8 tiles). A power of two. */
#define PN_TT_KV_ROTATE_DIM 256u

/** H[i][j] of the width-`dim` Sylvester Hadamard over sqrt(dim): (-1)^popcount(i&j)/sqrt(dim).
 *  0 for a `dim` that is not a power of two of at least 2, or an index out of range. */
float pn_tt_kv_rotate_entry(uint32_t dim, uint32_t i, uint32_t j);

/**
 * Rotate `rows` rows of `dim` floats in place: row <- row * H (= H * row; H is symmetric).
 * A fast Walsh-Hadamard transform in double, rounded to float once. Applying it twice is the
 * identity up to that rounding.
 *
 * @return 0, or -1 for a `dim` that is not a power of two of at least 2
 */
int pn_tt_kv_rotate_rows(float *rows, size_t row_count, uint32_t dim);

/**
 * H as a matmul weight: `dim/32` x `dim/32` tiles in the N-major grid the resident matmul
 * reads (tile (k, n) at n * Kt + k), each tile `tile_bytes` long: 2048 bf16 (face-ordered),
 * 1088 Bfp8_b or 576 Bfp4_b. `out` holds (dim/32)^2 * tile_bytes bytes.
 *
 * @return 0, or -1 for an unsupported `dim` or `tile_bytes`
 */
int pn_tt_kv_rotate_weight_tiles(uint32_t dim, uint32_t tile_bytes, uint8_t *out);

/**
 * The matmul view of one band of a rotated strip: `heads` tile rows of `head_tiles` tiles at
 * the strip's current address, one mirror (the primary; the view is read by one matmul, not
 * the group's mirrored A path). Pure.
 *
 * @return 0, or -1 when the strip holds fewer than `heads * head_tiles` tiles
 */
int pn_tt_kv_rotate_view(const pn_tt_activation *strip, uint32_t heads, uint32_t head_tiles,
                         pn_tt_activation *view);

/** Upload H for the device's attention head width, once, when the KV type rotates (a no-op
 *  for any other type). Call after the matmul residency is up. @return 0, or -1 */
int pn_tt_kv_rotate_arm(pn_tt_device *device);

/** Does this device's attention rotate Q and K (KV type k4h, H uploaded)? */
bool pn_tt_kv_rotating(const pn_tt_device *device);

/**
 * output <- input * H per head, on the card: one waiting matmul over the `[heads x head_tiles]`
 * view of each strip's current band. `input` and `output` must be distinct strips of at least
 * `heads * head_tiles` tiles.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_kv_rotate(pn_tt_device *device, const pn_tt_activation *input,
                    const pn_tt_activation *output, uint32_t heads);

#endif /* PN_TT_KV_ROTATE_H */
