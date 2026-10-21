/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_attn.c
 * @brief  Full-attention block for Qwen3.5-MoE, one token at a time.
 *
 * Written from a behaviour specification (plans/cleanroom_r2_qwen35_spec.md); no ggml or llama.cpp source was consulted.
 *
 * @status None
 * @issues A failure at the scores allocation leaves the KV appended and k mutated. Negative
 *         pos is undefined. Q4 mode with head_dim > 256 overflows the worker's stack scratch.
 * @todo   None
 */

#include "qwen35moe_internal.h"
#include "qwen35moe_kv_q4.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/simd.h"
#include "../core/qweight.h"
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

/* Per-head RMSNorm in place; the weight is shared by every head. */
static void attn_qk_norm(float *data, int n_heads, int head_dim, const float *weight,
                         float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *head = data + (size_t)h * head_dim;
        pn_vrms_norm_f32(head, weight, head, head_dim, eps);
    }
}

/* Partial rotary embedding over the first rot dims of each head, pairing dim j with
 * dim j + rot/2. rope_dim outside (0, head_dim) means full rotation. */
static void attn_rope(float *data, int n_heads, int head_dim, int rope_dim, int pos,
                      float theta) {
    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + (size_t)h * head_dim;
        for (int j = 0; j < half; j++) {
            float freq = 1.0f / powf(theta, (float)(2 * j) / (float)rot);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = head[j];
            float x1 = head[j + half];
            head[j] = x0 * c - x1 * s;
            head[j + half] = x0 * s + x1 * c;
        }
    }
}

/* Inputs of the per-query-head attention worker. */
typedef struct {
    const float *q;
    const float *k_cache;
    const float *v_cache;
    const uint8_t *kq;
    const uint8_t *vq;
    const uint16_t *k_sink;
    const uint16_t *v_sink;
    const uint16_t *k_win;
    const uint16_t *v_win;
    int q4;
    int sink;
    int win;
    int cur_pos;
    float *attn_out;
    float *scores_buf;
    int hd;
    int n_kv_groups;
    int max_seq;
    int seq_len;
    float scale;
} attn_heads_args;

/* Scores, softmax and weighted V sum for query heads [start, end). */
static void attn_heads(int start, int end, void *arg) {
    const attn_heads_args *a = (const attn_heads_args *)arg;
    int hd = a->hd;
    int max_seq = a->max_seq;
    int seq_len = a->seq_len;
    float kbuf[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
    float vbuf[PN_QWEN35MOE_ATTN_HEAD_DIM_MAX];
    for (int qh = start; qh < end; qh++) {
        int kv_h = qh / a->n_kv_groups;
        const float *q_vec = a->q + (size_t)qh * hd;
        float *scores = a->scores_buf + (size_t)qh * max_seq;
        float *out = a->attn_out + (size_t)qh * hd;
        const float *k_base = a->q4 ? NULL : a->k_cache + (size_t)kv_h * max_seq * hd;
        const float *v_base = a->q4 ? NULL : a->v_cache + (size_t)kv_h * max_seq * hd;

        for (int s = 0; s < seq_len; s++) {
            const float *k_row;
            if (a->q4) {
                pn_kvq4_load_row(a->kq, a->k_sink, a->k_win, kv_h, max_seq, s, a->cur_pos,
                                 kbuf, hd, a->sink, a->win);
                k_row = kbuf;
            } else {
                k_row = k_base + (size_t)s * hd;
            }
            scores[s] = pn_dot_f32(q_vec, k_row, hd) * a->scale;
        }

        float max = scores[0];
        for (int s = 1; s < seq_len; s++)
            if (scores[s] > max)
                max = scores[s];
        double sum = 0.0;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max);
            sum += (double)scores[s];
        }
        float inv = (float)(1.0 / sum);

        memset(out, 0, (size_t)hd * sizeof(*out));
        for (int s = 0; s < seq_len; s++) {
            float w = scores[s] * inv;
            const float *v_row;
            if (a->q4) {
                pn_kvq4_load_row(a->vq, a->v_sink, a->v_win, kv_h, max_seq, s, a->cur_pos,
                                 vbuf, hd, a->sink, a->win);
                v_row = vbuf;
            } else {
                v_row = v_base + (size_t)s * hd;
            }
            for (int d = 0; d < hd; d++)
                out[d] += w * v_row[d];
        }
    }
}

/* Append this token's K and V (per KV head) at the cache slot. */
static void attn_kv_append(pn_qwen35moe_kv *kv, const float *k, const float *v, int nkv,
                           int hd, int max_seq, int cache_pos, int sink, int win) {
    for (int h = 0; h < nkv; h++) {
        if (kv->q4) {
            pn_kvq4_store_row(kv->kq, kv->k_sink, kv->k_win, h, max_seq, cache_pos,
                              k + (size_t)h * hd, hd, sink, win);
            pn_kvq4_store_row(kv->vq, kv->v_sink, kv->v_win, h, max_seq, cache_pos,
                              v + (size_t)h * hd, hd, sink, win);
        } else {
            size_t slot = ((size_t)h * max_seq + cache_pos) * hd;
            memcpy(kv->k + slot, k + (size_t)h * hd, (size_t)hd * sizeof(*kv->k));
            memcpy(kv->v + slot, v + (size_t)h * hd, (size_t)hd * sizeof(*kv->v));
        }
    }
}

int pn_qwen35moe_attn_core(const pn_qwen35moe_attn_layer *layer,
                           const pn_qwen35moe_config *cfg,
                           const float *qg, float *k, float *v,
                           pn_qwen35moe_kv *kv, int pos, pn_arena *arena,
                           float *attn_out, struct fw_pool *pool) {
    if (!layer || !cfg || !qg || !k || !v || !kv || !arena || !attn_out)
        return -1;
    int nh = cfg->n_head;
    int nkv = cfg->n_head_kv;
    int hd = cfg->head_dim;
    int groups = nh / nkv;
    int max_seq = kv->max_seq;
    int qg_stride = 2 * hd;

    float *q = pn_arena_alloc_f32(arena, nh * hd);
    float *z = pn_arena_alloc_f32(arena, nh * hd);
    if (!q || !z)
        return -1;
    for (int h = 0; h < nh; h++) {
        const float *src = qg + (size_t)h * qg_stride;
        memcpy(q + (size_t)h * hd, src, (size_t)hd * sizeof(*q));
        memcpy(z + (size_t)h * hd, src + hd, (size_t)hd * sizeof(*z));
    }

    attn_qk_norm(q, nh, hd, layer->attn_q_norm, cfg->rms_norm_eps);
    attn_qk_norm(k, nkv, hd, layer->attn_k_norm, cfg->rms_norm_eps);
    attn_rope(q, nh, hd, cfg->rope_dim, pos, cfg->rope_theta);
    attn_rope(k, nkv, hd, cfg->rope_dim, pos, cfg->rope_theta);

    int cache_pos = pos % max_seq;
    int sink = kv->q4 ? pn_kvq4_sink() : 0;
    int win = kv->q4 ? pn_kvq4_win() : 0;
    attn_kv_append(kv, k, v, nkv, hd, max_seq, cache_pos, sink, win);

    int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
    float *scores_buf = pn_arena_alloc_f32(arena, nh * max_seq);
    if (!scores_buf)
        return -1;

    attn_heads_args args = {
        .q = q, .k_cache = kv->k, .v_cache = kv->v,
        .kq = kv->kq, .vq = kv->vq,
        .k_sink = kv->k_sink, .v_sink = kv->v_sink,
        .k_win = kv->k_win, .v_win = kv->v_win,
        .q4 = kv->q4, .sink = sink, .win = win, .cur_pos = pos,
        .attn_out = attn_out, .scores_buf = scores_buf,
        .hd = hd, .n_kv_groups = groups, .max_seq = max_seq, .seq_len = seq_len,
        .scale = 1.0f / sqrtf((float)hd),
    };
#ifdef USE_THREADS
    if (pool && nh >= 4)
        fw_for(pool, nh, attn_heads, &args);
    else
        attn_heads(0, nh, &args);
#else
    (void)pool;
    attn_heads(0, nh, &args);
#endif

    for (int i = 0; i < nh * hd; i++) {
        float g = 1.0f / (1.0f + expf(-z[i]));
        attn_out[i] *= g;
    }
    return 0;
}

int pn_qwen35moe_attn_block(const pn_qwen35moe_attn_layer *layer,
                            const pn_qwen35moe_config *cfg, const float *x,
                            pn_qwen35moe_kv *kv, int pos, pn_arena *arena,
                            float *out, struct fw_pool *pool) {
    if (!layer || !cfg || !x || !kv || !arena || !out)
        return -1;
    float *qg = pn_arena_alloc_f32(arena, layer->attn_q.rows);
    float *k = pn_arena_alloc_f32(arena, layer->attn_k.rows);
    float *v = pn_arena_alloc_f32(arena, layer->attn_v.rows);
    float *attn_out = pn_arena_alloc_f32(arena, cfg->n_head * cfg->head_dim);
    if (!qg || !k || !v || !attn_out)
        return -1;
    qwen35moe_mv(cfg, &layer->attn_q, x, qg);
    qwen35moe_mv(cfg, &layer->attn_k, x, k);
    qwen35moe_mv(cfg, &layer->attn_v, x, v);
    if (pn_qwen35moe_attn_core(layer, cfg, qg, k, v, kv, pos, arena, attn_out, pool) < 0)
        return -1;
    qwen35moe_mv(cfg, &layer->attn_output, attn_out, out);
    return 0;
}
