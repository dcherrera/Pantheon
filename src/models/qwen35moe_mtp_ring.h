/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_mtp_ring.h
 * @brief  The MTP draft head's q4 KV ring: buffer sizes for a geometry, and the zero fill a
 *         freshly allocated ring gets before any row is read.
 *
 * @status None
 * @issues None
 * @todo   None
 */

/* Why the fill:
 *   The head keeps ONE ring per replica and attends over [0, pos), but its backlog resets on
 *   every session switch or broken history, so rows below the reset hold either another
 *   session's rows (finite) or memory nothing has written since the ring was allocated.
 *   cuMemAlloc hands back recycled bytes; a q4 block whose fp16 scale decodes as inf/NaN
 *   makes one key NaN, the softmax carries it, and every draft logit comes out NaN (the
 *   2026-11-14 Xid 31 on a K80 host). Zeroed rows decode to K = V = 0: finite.
 *
 *   The fill is one small zero upload followed by doubling self-copies, so a 38 MB ring
 *   costs a 64 KiB host buffer and a handful of device copies, not a ring-sized upload. */

#ifndef QWEN35MOE_MTP_RING_H
#define QWEN35MOE_MTP_RING_H

#include <stddef.h>

/** Bytes of the zero chunk uploaded once per buffer; the rest is copied on the device. */
#define PN_MTP_RING_ZERO_CHUNK ((size_t)64 * 1024)

/** The ring's six buffers, in this order: K q4, V q4, K sink, V sink, K window, V window. */
#define PN_MTP_RING_BUFFERS 6

/** Per-buffer byte sizes of the head's ring (each size applies to both K and V). */
typedef struct {
    size_t cache_bytes;   /* q4 rows: n_head_kv * max_seq * pn_kvq4_row_bytes(head_dim) */
    size_t sink_bytes;    /* fp16 sink shadow: n_head_kv * head_dim * PN_KVQ4_SINKMAX * 2 */
    size_t win_bytes;     /* fp16 window shadow: n_head_kv * head_dim * PN_KVQ4_WMAX * 2 */
} pn_mtp_ring_sizes;

/** Device-side writes the fill needs; `buf` is one of the ring's buffers. */
typedef struct {
    /* Copy `bytes` host bytes to the start of buf. Returns 0, or -1. */
    int (*put)(void *ops_ctx, void *buf, const void *src, size_t bytes);
    /* Copy buf[src_off, +bytes) to buf[dst_off, +bytes) (never overlapping). Returns 0/-1. */
    int (*copy)(void *ops_ctx, void *buf, size_t dst_off, size_t src_off, size_t bytes);
    void *ops_ctx;
} pn_mtp_ring_fill_ops;

/**
 * Sizes of the head's ring buffers for a geometry.
 *
 * @param n_head_kv  KV heads (> 0)
 * @param head_dim   Head width (a multiple of PN_KVQ4_BLK)
 * @param max_seq    Ring rows per head (> 0)
 * @return           The sizes; all zero for a non-positive argument
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
pn_mtp_ring_sizes pn_mtp_ring_sizes_for(int n_head_kv, int head_dim, int max_seq);

/**
 * Zero the first `bytes` of buf: one put of at most PN_MTP_RING_ZERO_CHUNK zeros, then
 * copies that double the zeroed prefix until it covers `bytes`.
 *
 * @param buf    Destination buffer (opaque to this module)
 * @param bytes  Bytes to zero (0 = nothing to do)
 * @param ops    put/copy implementation
 * @return       0, or -1 on a NULL argument or a failed put/copy
 *
 * Thread-safe: yes (as far as ops is)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_mtp_ring_zero_fill(void *buf, size_t bytes, const pn_mtp_ring_fill_ops *ops);

/**
 * Zero all six ring buffers (order: PN_MTP_RING_BUFFERS) at the sizes in `sizes`.
 *
 * @param bufs   The six buffers
 * @param sizes  From pn_mtp_ring_sizes_for, the same sizes the buffers were allocated at
 * @param ops    put/copy implementation
 * @return       0, or -1 when any buffer could not be zeroed
 *
 * Thread-safe: yes (as far as ops is)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_mtp_ring_clear(void *const bufs[PN_MTP_RING_BUFFERS], const pn_mtp_ring_sizes *sizes,
                      const pn_mtp_ring_fill_ops *ops);

#endif /* QWEN35MOE_MTP_RING_H */
