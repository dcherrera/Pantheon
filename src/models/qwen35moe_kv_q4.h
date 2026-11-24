/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_kv_q4.h
 * @brief  Q4 KV-cache row helpers with an fp16 sink and recent-window shadow.
 *
 * Written from a behaviour specification (cleanroom_r2_qwen35_spec.md, recorded in docs/cleanroom/README.md); no ggml or llama.cpp source was consulted.
 *
 * @status None
 * @issues Env getters cache per translation unit and are not thread-safe. The window
 *         predicate compares a cache slot with the absolute position, so after the ring
 *         wraps almost every slot reads Q4. The fp16 window ring laps at PN_KVQ4_WMAX, not
 *         at the runtime window. head_dim % 32 tail elements are neither quantised nor
 *         dequantised.
 * @todo   None
 */

#ifndef QWEN35MOE_KV_Q4_H
#define QWEN35MOE_KV_Q4_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../formats/gguf_dequant.h"

/* Layout constants are ABI: device kernels, snapshot files and allocators use them. */
#define PN_KVQ4_BLK       32
#define PN_KVQ4_BLK_BYTES ((int)sizeof(pn_block_q4_0))
#define PN_KVQ4_WMAX      256
#define PN_KVQ4_SINKMAX   8

/** Runtime fp16 window length (QWEN35_KV_Q4_WIN, default 128, clamped to [0, WMAX]). */
static inline int pn_kvq4_win(void) {
    static int cached_win = -1;
    if (cached_win < 0) {
        const char *env_value = getenv("QWEN35_KV_Q4_WIN");
        int win = env_value ? atoi(env_value) : 128;
        if (win < 0)
            win = 0;
        if (win > PN_KVQ4_WMAX)
            win = PN_KVQ4_WMAX;
        cached_win = win;
    }
    return cached_win;
}

/** Runtime fp16 sink length (QWEN35_KV_Q4_SINK, default 4, clamped to [0, SINKMAX]). */
static inline int pn_kvq4_sink(void) {
    static int cached_sink = -1;
    if (cached_sink < 0) {
        const char *env_value = getenv("QWEN35_KV_Q4_SINK");
        int sink = env_value ? atoi(env_value) : 4;
        if (sink < 0)
            sink = 0;
        if (sink > PN_KVQ4_SINKMAX)
            sink = PN_KVQ4_SINKMAX;
        cached_sink = sink;
    }
    return cached_sink;
}

/** 1 when slot s is read from an fp16 shadow (sink or recent window), else 0. */
static inline int pn_kvq4_is_fp16(int s, int cur_pos, int sink, int win) {
    return (s < sink) || (win > 0 && s > cur_pos - win);
}

static inline void pn_kvq4_f32_to_half_row(const float *src, uint16_t *dst, int head_dim) {
    for (int d = 0; d < head_dim; d++)
        dst[d] = pn_float_to_half(src[d]);
}

static inline void pn_kvq4_half_to_f32_row(const uint16_t *src, float *dst, int head_dim) {
    for (int d = 0; d < head_dim; d++)
        dst[d] = pn_half_to_float(src[d]);
}

/** Bytes of one quantised row: whole 32-element blocks only. */
static inline int pn_kvq4_row_bytes(int head_dim) {
    return (head_dim / PN_KVQ4_BLK) * PN_KVQ4_BLK_BYTES;
}

/** Quantise the whole 32-element blocks of src into dst; the tail is ignored. */
static inline void pn_kvq4_quant_row(const float *src, uint8_t *dst, int head_dim) {
    int n_blocks = head_dim / PN_KVQ4_BLK;
    for (int b = 0; b < n_blocks; b++) {
        const float *x = src + (size_t)b * PN_KVQ4_BLK;
        pn_block_q4_0 *blk = (pn_block_q4_0 *)(dst + (size_t)b * PN_KVQ4_BLK_BYTES);
        float amax = 0.0f;
        for (int j = 0; j < PN_KVQ4_BLK; j++) {
            float a = fabsf(x[j]);
            if (a > amax)
                amax = a;
        }
        float d = amax / -8.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        blk->d = pn_float_to_half(d);
        for (int j = 0; j < PN_KVQ4_BLK / 2; j++) {
            int q0 = (int)lrintf(x[j] * id) + 8;
            int q1 = (int)lrintf(x[j + PN_KVQ4_BLK / 2] * id) + 8;
            if (q0 < 0)
                q0 = 0;
            else if (q0 > 15)
                q0 = 15;
            if (q1 < 0)
                q1 = 0;
            else if (q1 > 15)
                q1 = 15;
            blk->qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

/** Dequantise the whole blocks of a row; tail floats of dst are left untouched. */
static inline void pn_kvq4_dequant_row(const uint8_t *src, float *dst, int head_dim) {
    int n_blocks = head_dim / PN_KVQ4_BLK;
    for (int b = 0; b < n_blocks; b++)
        pn_dequantize_block_q4_0(src + (size_t)b * PN_KVQ4_BLK_BYTES,
                                 dst + (size_t)b * PN_KVQ4_BLK);
}

/**
 * Append one row at cache slot cache_pos: always Q4, plus the fp16 sink shadow when the
 * slot is a sink slot, plus the fp16 window ring (indexed by slot % WMAX) when win > 0.
 */
static inline void pn_kvq4_store_row(uint8_t *q4_buf, uint16_t *sink_buf,
                                     uint16_t *win_buf, int h, int max_seq,
                                     int cache_pos, const float *src,
                                     int head_dim, int sink, int win) {
    int rb = pn_kvq4_row_bytes(head_dim);
    pn_kvq4_quant_row(src, q4_buf + ((size_t)h * max_seq + cache_pos) * rb, head_dim);
    if (cache_pos < sink)
        pn_kvq4_f32_to_half_row(src,
                                sink_buf + ((size_t)h * PN_KVQ4_SINKMAX + cache_pos) * head_dim,
                                head_dim);
    if (win > 0)
        pn_kvq4_f32_to_half_row(src,
                                win_buf + ((size_t)h * PN_KVQ4_WMAX
                                           + cache_pos % PN_KVQ4_WMAX) * head_dim,
                                head_dim);
}

/** Reconstruct slot s: sink shadow, else window ring, else the Q4 row. */
static inline void pn_kvq4_load_row(const uint8_t *q4_buf,
                                    const uint16_t *sink_buf,
                                    const uint16_t *win_buf, int h, int max_seq,
                                    int s, int cur_pos, float *dst,
                                    int head_dim, int sink, int win) {
    if (s < sink) {
        pn_kvq4_half_to_f32_row(sink_buf + ((size_t)h * PN_KVQ4_SINKMAX + s) * head_dim,
                                dst, head_dim);
        return;
    }
    if (win > 0 && s > cur_pos - win) {
        pn_kvq4_half_to_f32_row(win_buf + ((size_t)h * PN_KVQ4_WMAX + s % PN_KVQ4_WMAX)
                                          * head_dim,
                                dst, head_dim);
        return;
    }
    int rb = pn_kvq4_row_bytes(head_dim);
    pn_kvq4_dequant_row(q4_buf + ((size_t)h * max_seq + s) * rb, dst, head_dim);
}

#endif /* QWEN35MOE_KV_Q4_H */
