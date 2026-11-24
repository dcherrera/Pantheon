/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* qwen35moe_die_projection.c — see qwen35moe_die_projection.h. Each formula is the
 * byte count of the matching allocation in qwen35moe_forward_cuda.c
 * (gpu_attn_slot_ring_ensure, pn_qwen35moe_decode_slots_carve,
 * pn_qwen35moe_slot_checkpoints_carve). */

#include "qwen35moe_die_projection.h"

#include "qwen35moe_kv_q4.h"

/* One row's K or V ring entry width for one KV head. */
static uint64_t kv_row_bytes(const pn_qwen35_die_shape *s)
{
    return s->kv_q4 ? (uint64_t)pn_kvq4_row_bytes(s->head_dim)
                    : (uint64_t)s->head_dim * sizeof(float);
}

/* One attention layer's fp16 sink + window shadow, K or V, one row (q4 only). */
static uint64_t kv_shadow_bytes(const pn_qwen35_die_shape *s)
{
    if (!s->kv_q4)
        return 0;
    return (uint64_t)s->n_head_kv * (PN_KVQ4_SINKMAX + PN_KVQ4_WMAX) *
           (uint64_t)s->head_dim * sizeof(uint16_t);
}

uint64_t pn_qwen35_ssm_image_bytes(const pn_qwen35_die_shape *s)
{
    uint64_t conv = (uint64_t)(s->ssm_conv - 1) * (uint64_t)s->conv_dim * sizeof(float);
    uint64_t ssm = (uint64_t)s->ssm_state * (uint64_t)s->ssm_state *
                   (uint64_t)s->ssm_dt_rank * sizeof(float);
    return conv + ssm;
}

uint64_t pn_qwen35_session_block_bytes(const pn_qwen35_die_shape *s)
{
    uint64_t ring = (uint64_t)s->n_head_kv * (uint64_t)s->max_seq * kv_row_bytes(s);
    uint64_t attn = 2 * ring + 2 * kv_shadow_bytes(s);   /* K and V */
    return (uint64_t)s->n_attn * attn + (uint64_t)s->n_ssm * pn_qwen35_ssm_image_bytes(s);
}

uint64_t pn_qwen35_checkpoint_row_bytes(const pn_qwen35_die_shape *s)
{
    return (uint64_t)s->n_ssm * pn_qwen35_ssm_image_bytes(s) +
           (uint64_t)s->n_attn * 2 * kv_shadow_bytes(s);
}

void pn_qwen35_die_project(const pn_qwen35_die_shape *s, pn_qwen35_die_need *out)
{
    int rows = s->session_rows > 0 ? s->session_rows : 1;

    out->weights = s->layer_weight_bytes + s->extra_weight_bytes + s->mtp_weight_bytes;
    out->sessions = (uint64_t)rows * pn_qwen35_session_block_bytes(s);
    out->scratch = s->scratch_bytes;
    out->total = out->weights + out->sessions + out->scratch;
}
