/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_mtp_ring.c
 * @brief  MTP head ring sizes and zero fill (see qwen35moe_mtp_ring.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "qwen35moe_mtp_ring.h"

#include "qwen35moe_kv_q4.h"

pn_mtp_ring_sizes pn_mtp_ring_sizes_for(int n_head_kv, int head_dim, int max_seq) {
    pn_mtp_ring_sizes sizes = {0, 0, 0};
    size_t kvd;

    if (n_head_kv <= 0 || head_dim <= 0 || max_seq <= 0)
        return sizes;
    kvd = (size_t)n_head_kv * (size_t)head_dim;
    sizes.cache_bytes = (size_t)n_head_kv * (size_t)max_seq * (size_t)pn_kvq4_row_bytes(head_dim);
    sizes.sink_bytes = kvd * PN_KVQ4_SINKMAX * 2;
    sizes.win_bytes = kvd * PN_KVQ4_WMAX * 2;
    return sizes;
}

int pn_mtp_ring_zero_fill(void *buf, size_t bytes, const pn_mtp_ring_fill_ops *ops) {
    static const unsigned char zeros[PN_MTP_RING_ZERO_CHUNK];
    size_t done;

    if (!buf || !ops || !ops->put || !ops->copy)
        return -1;
    if (bytes == 0)
        return 0;
    done = bytes < PN_MTP_RING_ZERO_CHUNK ? bytes : PN_MTP_RING_ZERO_CHUNK;
    if (ops->put(ops->ops_ctx, buf, zeros, done) != 0)
        return -1;
    while (done < bytes) {
        size_t n = bytes - done < done ? bytes - done : done;

        if (ops->copy(ops->ops_ctx, buf, done, 0, n) != 0)
            return -1;
        done += n;
    }
    return 0;
}

int pn_mtp_ring_clear(void *const bufs[PN_MTP_RING_BUFFERS], const pn_mtp_ring_sizes *sizes,
                      const pn_mtp_ring_fill_ops *ops) {
    if (!bufs || !sizes)
        return -1;
    for (int i = 0; i < PN_MTP_RING_BUFFERS; i++) {
        size_t bytes = i < 2 ? sizes->cache_bytes : i < 4 ? sizes->sink_bytes : sizes->win_bytes;

        if (pn_mtp_ring_zero_fill(bufs[i], bytes, ops) != 0)
            return -1;
    }
    return 0;
}
