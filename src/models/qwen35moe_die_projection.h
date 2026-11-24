/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_die_projection.h
 * @brief  What one die of a qwen35moe CUDA split holds once the model serves: its
 *         weights at their on-die size, its session blocks, and the runtime scratch.
 *         Pure arithmetic, no device calls, so the shipped profiles are checked on
 *         any host (tests/test_qwen35_die_projection.c).
 *
 * The model-load gate (gpu_admit_die_share) compares this against the die's free
 * VRAM minus PN_DIE_LOAD_MARGIN_BYTES. Every term mirrors an allocation the build
 * makes:
 *   weights   the die's blk.* tensors at the repacked size, the LM head on the last
 *             die, and the MTP draft head when it is kept (also the last die)
 *   sessions  session_rows x one decode block (pn_qwen35moe_decode_slots_carve):
 *             per attention layer a K and V ring at max_seq plus the q4 fp16 sink
 *             and window shadows; per SSM layer a conv + recurrent image. At least
 *             one row is charged: a single-user serve still grows one ring.
 *   scratch   the runtime scratch (activations, prefill buffers, MTP ring), allocated
 *             lazily after load.
 * Turn / divergence checkpoints are optional and are NOT charged: their carve
 * checks the die itself (pn_qwen35_checkpoint_row_bytes) and falls back.
 *
 * @status complete
 * @issues scratch is a measured constant, not a formula.
 * @todo   None
 */

#ifndef QWEN35MOE_DIE_PROJECTION_H
#define QWEN35MOE_DIE_PROJECTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One die's share of the split, in the units the allocators use. */
typedef struct {
    uint64_t layer_weight_bytes;   /* blk.* tensors of this die's layers, on-die size */
    uint64_t extra_weight_bytes;   /* non-layer weight held here (LM head, vocab half) */
    uint64_t mtp_weight_bytes;     /* MTP draft head when kept and placed here, else 0 */
    int n_attn;                    /* attention layers on this die */
    int n_ssm;                     /* SSM layers on this die */
    int session_rows;              /* decode blocks carved on the card (< 1: one) */
    int max_seq;                   /* KV ring window (the load's ctx cap) */
    int kv_q4;                     /* 1: q4 rows + fp16 sink/window shadows */
    int n_head_kv, head_dim;
    int ssm_conv, conv_dim, ssm_state, ssm_dt_rank;
    uint64_t scratch_bytes;        /* runtime scratch on this die */
} pn_qwen35_die_shape;

typedef struct {
    uint64_t weights;    /* layer + extra + MTP weights */
    uint64_t sessions;   /* session rows x session block */
    uint64_t scratch;    /* runtime scratch */
    uint64_t total;      /* the sum: what the die holds while it serves */
} pn_qwen35_die_need;

/* One SSM layer's conv + recurrent image (one row). */
uint64_t pn_qwen35_ssm_image_bytes(const pn_qwen35_die_shape *s);

/* One decode block (one session row) on this die: every attention layer's K/V ring
 * at max_seq (+ q4 shadows) and every SSM layer's image. */
uint64_t pn_qwen35_session_block_bytes(const pn_qwen35_die_shape *s);

/* One checkpoint row on this die (pn_qwen35moe_slot_checkpoints_carve): the SSM
 * images plus, under q4, the attention layers' fp16 sink/window shadows. */
uint64_t pn_qwen35_checkpoint_row_bytes(const pn_qwen35_die_shape *s);

/* Fill `out` with what the die holds while it serves. */
void pn_qwen35_die_project(const pn_qwen35_die_shape *s, pn_qwen35_die_need *out);

#ifdef __cplusplus
}
#endif

#endif /* QWEN35MOE_DIE_PROJECTION_H */
