/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_state_snapshot.c
 * @brief Host-side snapshot/restore of the qwen35moe dual state at a token
 *        position — the primitive behind SYSTEM-PREFIX caching. A snapshot
 *        deep-copies everything the forward needs to continue from position
 *        `pos`: every SSM layer's conv + recurrent state (fixed size) and every
 *        attention layer's KV rows [0, pos) plus the full fp16 sink/window
 *        shadows (q4 mode). Restore copies it back into a live state, making
 *        the host mirror authoritative again (kv_mirror_stale cleared); the
 *        caller must then reset pn_state->pos and (on the GPU path) call
 *        pn_qwen35moe_gpu_forget_state so the next forward re-seeds the GPU
 *        shadows from the restored host mirror.
 *
 * @status Complete: capture/restore round-trip is unit-tested host-side
 *         (tests/test_prefix_cache_sys.c) for both fp32 and q4 KV modes. Adds
 *         (K80 serving-ops Phase 7) pn_qwen35moe_snap_serialize/_deserialize —
 *         a contiguous portable blob for the session warm-tier; the snap→bytes
 *         →snap round-trip is byte-identical (unit-tested, same suite).
 *         A RECURRENT snapshot (pn_qwen35moe_state_snapshot_recurrent) holds
 *         no KV rows: the single-user turn-start snapshot (think_history
 *         drop_last), restored over KV rows the mirror still holds.
 * @issues None
 * @todo None
 */

#include "qwen35moe_internal.h"
#include "qwen35moe_kv_q4.h"
#include "../core/mem/pn_host_admission.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* One SSM layer's captured recurrent state (full copy — fixed size). */
typedef struct {
    size_t conv_n;      /* floats in conv_state */
    size_t ssm_n;       /* floats in ssm_state */
    float *conv;
    float *ssm;
} snap_ssm_layer;

/* One attention layer's captured KV: rows [0, pos) per head, packed
 * [n_head_kv][pos][row_b] (the live cache strides by max_seq), plus the full
 * fp16 sink/window shadows in q4 mode (the window is a RING — later positions
 * overwrite earlier slots, so it cannot be reconstructed from the q4 body). */
typedef struct {
    int q4;
    int n_head_kv;
    int head_dim;
    int max_seq;
    size_t row_b;       /* bytes per row: q4 packed or head_dim * 4 */
    unsigned char *k_rows;
    unsigned char *v_rows;
    uint16_t *k_sink;   /* q4 only, else NULL */
    uint16_t *v_sink;
    uint16_t *k_win;
    uint16_t *v_win;
} snap_kv_layer;

/* Whole-state snapshot at one position (opaque to callers). */
typedef struct {
    int pos;
    int kv_rows;              /* KV rows held: pos, or 0 (recurrent snapshot) */
    int n_ssm;
    int n_attn;
    snap_ssm_layer *ssm;
    snap_kv_layer *kv;
    uint64_t host_admitted;   /* host bytes admitted (released on free) */
} qwen35moe_state_snap;

#define SNAP_ADMIT_HINT "shorten the session or free host memory"

/* Host bytes a capture of `s` copies: every SSM image, rows [0, pos) of every
 * attention layer's K and V, and the q4 fp16 shadows. */
static uint64_t snap_capture_bytes(const pn_qwen35moe_state *s, int pos)
{
    uint64_t total = 0;
    for (int i = 0; i < s->n_ssm; i++) {
        const pn_qwen35moe_ssm_state *st = &s->ssm[i];
        total += ((uint64_t)st->conv_minus_1 * st->conv_dim +
                  (uint64_t)st->state_size * st->state_size * st->n_v_heads) *
                 sizeof(float);
    }
    for (int i = 0; i < s->n_attn; i++) {
        const pn_qwen35moe_kv *kv = &s->kv[i];
        uint64_t row_b = kv->q4 ? (uint64_t)pn_kvq4_row_bytes(kv->head_dim)
                                : (uint64_t)kv->head_dim * sizeof(float);
        total += 2ull * kv->n_head_kv * (uint64_t)pos * row_b;
        if (kv->q4)
            total += 2ull * kv->n_head_kv * (PN_KVQ4_SINKMAX + PN_KVQ4_WMAX) *
                     (uint64_t)kv->head_dim * sizeof(uint16_t);
    }
    return total;
}

/* Admit `bytes` for `sn`; the charge rides on the snapshot until it is freed. */
static int snap_admit(qwen35moe_state_snap *sn, uint64_t bytes)
{
    if (pn_host_admit_hint(bytes, "hybrid session snapshot", SNAP_ADMIT_HINT,
                           NULL, 0) != 0)
        return -1;
    sn->host_admitted = bytes;
    return 0;
}

/* Copy `pos` rows per head between packed ([pos] stride) and live ([max_seq]
 * stride) layouts. dir_capture != 0: live -> packed; else packed -> live. */
static void kv_rows_copy(unsigned char *packed, unsigned char *live,
                         int n_head_kv, int max_seq, int pos, size_t row_b,
                         int dir_capture)
{
    for (int h = 0; h < n_head_kv; h++) {
        unsigned char *p = packed + (size_t)h * pos * row_b;
        unsigned char *l = live + (size_t)h * max_seq * row_b;
        if (dir_capture)
            memcpy(p, l, (size_t)pos * row_b);
        else
            memcpy(l, p, (size_t)pos * row_b);
    }
}

/* Capture one attention layer's KV into sk. Returns 0 on success. */
static int snap_kv_capture(const pn_qwen35moe_kv *kv, snap_kv_layer *sk,
                           int pos)
{
    sk->q4 = kv->q4;
    sk->n_head_kv = kv->n_head_kv;
    sk->head_dim = kv->head_dim;
    sk->max_seq = kv->max_seq;
    sk->row_b = kv->q4 ? (size_t)pn_kvq4_row_bytes(kv->head_dim)
                       : (size_t)kv->head_dim * sizeof(float);

    size_t rows_b = (size_t)kv->n_head_kv * pos * sk->row_b;
    sk->k_rows = malloc(rows_b ? rows_b : 1);
    sk->v_rows = malloc(rows_b ? rows_b : 1);
    if (!sk->k_rows || !sk->v_rows)
        return -1;
    unsigned char *ksrc = kv->q4 ? kv->kq : (unsigned char *)kv->k;
    unsigned char *vsrc = kv->q4 ? kv->vq : (unsigned char *)kv->v;
    kv_rows_copy(sk->k_rows, ksrc, kv->n_head_kv, kv->max_seq, pos, sk->row_b, 1);
    kv_rows_copy(sk->v_rows, vsrc, kv->n_head_kv, kv->max_seq, pos, sk->row_b, 1);

    if (kv->q4) {
        size_t sink_n = (size_t)kv->n_head_kv * PN_KVQ4_SINKMAX * kv->head_dim;
        size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
        sk->k_sink = malloc(sink_n * sizeof(uint16_t));
        sk->v_sink = malloc(sink_n * sizeof(uint16_t));
        sk->k_win = malloc(win_n * sizeof(uint16_t));
        sk->v_win = malloc(win_n * sizeof(uint16_t));
        if (!sk->k_sink || !sk->v_sink || !sk->k_win || !sk->v_win)
            return -1;
        memcpy(sk->k_sink, kv->k_sink, sink_n * sizeof(uint16_t));
        memcpy(sk->v_sink, kv->v_sink, sink_n * sizeof(uint16_t));
        memcpy(sk->k_win, kv->k_win, win_n * sizeof(uint16_t));
        memcpy(sk->v_win, kv->v_win, win_n * sizeof(uint16_t));
    }
    return 0;
}

/* Restore one attention layer's KV from sk (dims already validated). */
static void snap_kv_restore(const snap_kv_layer *sk, pn_qwen35moe_kv *kv,
                            int pos)
{
    unsigned char *kdst = kv->q4 ? kv->kq : (unsigned char *)kv->k;
    unsigned char *vdst = kv->q4 ? kv->vq : (unsigned char *)kv->v;
    kv_rows_copy(sk->k_rows, kdst, kv->n_head_kv, kv->max_seq, pos, sk->row_b, 0);
    kv_rows_copy(sk->v_rows, vdst, kv->n_head_kv, kv->max_seq, pos, sk->row_b, 0);
    if (kv->q4) {
        size_t sink_n = (size_t)kv->n_head_kv * PN_KVQ4_SINKMAX * kv->head_dim;
        size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
        memcpy(kv->k_sink, sk->k_sink, sink_n * sizeof(uint16_t));
        memcpy(kv->v_sink, sk->v_sink, sink_n * sizeof(uint16_t));
        memcpy(kv->k_win, sk->k_win, win_n * sizeof(uint16_t));
        memcpy(kv->v_win, sk->v_win, win_n * sizeof(uint16_t));
    }
    /* The restored host mirror is now the single source of truth. */
    kv->kv_mirror_stale = 0;
    /* Phase 10b: content replaced under the same pointer — invalidate every
     * GPU ring watermark keyed on (pointer, generation) so the next forward
     * re-uploads the restored rows instead of trusting stale ring content. */
    kv->content_gen = pn_qwen35moe_kv_next_content_gen();
}

void pn_qwen35moe_state_snapshot_free(void *snap_v)
{
    qwen35moe_state_snap *sn = (qwen35moe_state_snap *)snap_v;
    if (!sn)
        return;
    if (sn->ssm) {
        for (int i = 0; i < sn->n_ssm; i++) {
            free(sn->ssm[i].conv);
            free(sn->ssm[i].ssm);
        }
        free(sn->ssm);
    }
    if (sn->kv) {
        for (int i = 0; i < sn->n_attn; i++) {
            free(sn->kv[i].k_rows);
            free(sn->kv[i].v_rows);
            free(sn->kv[i].k_sink);
            free(sn->kv[i].v_sink);
            free(sn->kv[i].k_win);
            free(sn->kv[i].v_win);
        }
        free(sn->kv);
    }
    pn_host_release(sn->host_admitted);
    free(sn);
}

/* Capture `s` at `pos` holding KV rows [0, kv_rows) (kv_rows is pos or 0). */
static void *snap_capture(const pn_qwen35moe_state *s, int pos, int kv_rows)
{
    if (!s || pos <= 0)
        return NULL;
    for (int i = 0; i < s->n_attn; i++)
        if (pos > s->kv[i].max_seq)
            return NULL;

    qwen35moe_state_snap *sn = calloc(1, sizeof(*sn));
    if (!sn)
        return NULL;
    sn->pos = pos;
    sn->kv_rows = kv_rows;
    sn->n_ssm = s->n_ssm;
    sn->n_attn = s->n_attn;
    sn->ssm = s->n_ssm ? calloc((size_t)s->n_ssm, sizeof(*sn->ssm)) : NULL;
    sn->kv = s->n_attn ? calloc((size_t)s->n_attn, sizeof(*sn->kv)) : NULL;
    if ((s->n_ssm && !sn->ssm) || (s->n_attn && !sn->kv))
        goto fail;
    if (snap_admit(sn, snap_capture_bytes(s, kv_rows)) != 0)
        goto fail;

    for (int i = 0; i < s->n_ssm; i++) {
        const pn_qwen35moe_ssm_state *st = &s->ssm[i];
        snap_ssm_layer *sl = &sn->ssm[i];
        sl->conv_n = (size_t)st->conv_minus_1 * st->conv_dim;
        sl->ssm_n = (size_t)st->state_size * st->state_size * st->n_v_heads;
        sl->conv = malloc(sl->conv_n * sizeof(float));
        sl->ssm = malloc(sl->ssm_n * sizeof(float));
        if (!sl->conv || !sl->ssm)
            goto fail;
        memcpy(sl->conv, st->conv_state, sl->conv_n * sizeof(float));
        memcpy(sl->ssm, st->ssm_state, sl->ssm_n * sizeof(float));
    }
    for (int i = 0; i < s->n_attn; i++)
        if (snap_kv_capture(&s->kv[i], &sn->kv[i], kv_rows) < 0)
            goto fail;
    return sn;

fail:
    pn_qwen35moe_state_snapshot_free(sn);
    return NULL;
}

void *pn_qwen35moe_state_snapshot(const void *state_v, int pos)
{
    return snap_capture((const pn_qwen35moe_state *)state_v, pos, pos);
}

void *pn_qwen35moe_state_snapshot_recurrent(const void *state_v, int pos)
{
    return snap_capture((const pn_qwen35moe_state *)state_v, pos, 0);
}

/* Do the snapshot's dimensions match the target state exactly? A mismatch
 * (different model / KV mode / window) makes a restore meaningless. */
static int snap_matches_state(const qwen35moe_state_snap *sn,
                              const pn_qwen35moe_state *s)
{
    if (sn->n_ssm != s->n_ssm || sn->n_attn != s->n_attn)
        return 0;
    /* pos strides the live KV buffer on restore: snap_kv_restore copies
     * pos*row_b bytes per head into a buffer whose per-head span is
     * max_seq*row_b (kv_rows_copy). pos is an attacker-controlled header field
     * (deserialize only checks pos>0), so a blob with pos>max_seq — all other
     * dims matching this live model — drives an out-of-bounds write past the
     * live cache. The capture path guards pos<=max_seq (pn_qwen35moe_state_
     * snapshot); the restore path MUST re-check it here since the blob is
     * untrusted. Every attention layer shares the snapshot's single pos. */
    for (int i = 0; i < s->n_attn; i++)
        if (sn->pos > s->kv[i].max_seq)
            return 0;
    for (int i = 0; i < s->n_ssm; i++) {
        const pn_qwen35moe_ssm_state *st = &s->ssm[i];
        if (sn->ssm[i].conv_n != (size_t)st->conv_minus_1 * st->conv_dim ||
            sn->ssm[i].ssm_n !=
                (size_t)st->state_size * st->state_size * st->n_v_heads)
            return 0;
    }
    for (int i = 0; i < s->n_attn; i++) {
        const snap_kv_layer *sk = &sn->kv[i];
        const pn_qwen35moe_kv *kv = &s->kv[i];
        if (sk->q4 != kv->q4 || sk->n_head_kv != kv->n_head_kv ||
            sk->head_dim != kv->head_dim || sk->max_seq != kv->max_seq)
            return 0;
    }
    return 1;
}

int pn_qwen35moe_state_restore(void *state_v, const void *snap_v)
{
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state_v;
    const qwen35moe_state_snap *sn = (const qwen35moe_state_snap *)snap_v;
    if (!s || !sn)
        return -1;
    assert(sn->pos > 0);
    if (!snap_matches_state(sn, s))
        return -1;

    for (int i = 0; i < s->n_ssm; i++) {
        pn_qwen35moe_ssm_state *st = &s->ssm[i];
        memcpy(st->conv_state, sn->ssm[i].conv,
               sn->ssm[i].conv_n * sizeof(float));
        memcpy(st->ssm_state, sn->ssm[i].ssm,
               sn->ssm[i].ssm_n * sizeof(float));
    }
    for (int i = 0; i < s->n_attn; i++)
        snap_kv_restore(&sn->kv[i], &s->kv[i], sn->kv_rows);
    return 0;
}

int pn_qwen35moe_state_snapshot_pos(const void *snap_v)
{
    const qwen35moe_state_snap *sn = (const qwen35moe_state_snap *)snap_v;
    return sn ? sn->pos : -1;
}

/* ============================================================
 * Serialize / deserialize — the session warm-tier blob (Phase 7)
 *
 * One contiguous little-endian blob. All integers are fixed-width and written
 * LE regardless of host endianness (put_u32/64), so a blob is portable across
 * machines. Layout:
 *
 *   [ header ]
 *     u32  magic            = PN_SNAP_MAGIC ("PNS1" as LE bytes)
 *     u32  format_version   = PN_SNAP_FORMAT_VERSION
 *     u64  weights_version  (opaque model-identity tag; validated by caller)
 *     i32  pos
 *     i32  n_ssm
 *     i32  n_attn
 *   [ n_ssm × SSM layer ]
 *     u64  conv_n           (floats)
 *     u64  ssm_n            (floats)
 *     f32  conv[conv_n]     (raw little-endian float32)
 *     f32  ssm [ssm_n]
 *   [ n_attn × attention layer ]
 *     i32  q4               (0 = fp32 KV, 1 = q4 packed + fp16 shadows)
 *     i32  n_head_kv
 *     i32  head_dim
 *     i32  max_seq
 *     u64  row_b            (bytes per row)
 *     bytes k_rows[n_head_kv * pos * row_b]
 *     bytes v_rows[...]
 *     (q4 only:)
 *     u16  k_sink[n_head_kv * PN_KVQ4_SINKMAX * head_dim]
 *     u16  v_sink[...]
 *     u16  k_win [n_head_kv * PN_KVQ4_WMAX * head_dim]
 *     u16  v_win [...]
 *
 * f32/u16 payloads are copied as raw bytes: the K80 fleet and any client are
 * little-endian IEEE-754, so no per-element byte-swap is done (documented). The
 * header/count fields ARE explicitly LE so the framing is unambiguous. On a
 * hypothetical big-endian host the float payload would need swapping — out of
 * scope for this fleet; the format_version gates a future change.
 * ============================================================ */

#define PN_SNAP_MAGIC          0x31534E50u   /* 'P''N''S''1' little-endian */
#define PN_SNAP_FORMAT_VERSION 1u

/* Bounded cursor over a byte buffer (write or read). `err` latches on the first
 * overflow so callers check it once at the end instead of per-field. */
typedef struct {
    unsigned char *buf;   /* NULL in sizing pass (writer) */
    size_t         cap;
    size_t         off;
    int            err;
} snap_cursor;

static void put_bytes(snap_cursor *c, const void *src, size_t n)
{
    if (c->err)
        return;
    if (c->off + n > c->cap) { c->err = 1; return; }
    if (c->buf && n)
        memcpy(c->buf + c->off, src, n);
    c->off += n;
}

static void put_u32(snap_cursor *c, uint32_t v)
{
    unsigned char b[4] = { (unsigned char)(v), (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    put_bytes(c, b, 4);
}

static void put_i32(snap_cursor *c, int32_t v) { put_u32(c, (uint32_t)v); }

static void put_u64(snap_cursor *c, uint64_t v)
{
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    put_bytes(c, b, 8);
}

static void get_bytes(snap_cursor *c, void *dst, size_t n)
{
    if (c->err)
        return;
    if (c->off + n > c->cap) { c->err = 1; return; }
    if (n)
        memcpy(dst, c->buf + c->off, n);
    c->off += n;
}

static uint32_t get_u32(snap_cursor *c)
{
    unsigned char b[4] = {0};
    get_bytes(c, b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static int32_t get_i32(snap_cursor *c) { return (int32_t)get_u32(c); }

static uint64_t get_u64(snap_cursor *c)
{
    unsigned char b[8] = {0};
    get_bytes(c, b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

/* Total q4 sink/window element counts for one attention layer. */
static void snap_kv_shadow_counts(const snap_kv_layer *sk, size_t *sink_n,
                                  size_t *win_n)
{
    *sink_n = (size_t)sk->n_head_kv * PN_KVQ4_SINKMAX * sk->head_dim;
    *win_n = (size_t)sk->n_head_kv * PN_KVQ4_WMAX * sk->head_dim;
}

/* Walk the whole snapshot through a cursor: write (buf != NULL) or size-only
 * (buf == NULL, cursor->cap = SIZE_MAX). One walk shared by both so the sizing
 * pass and the write pass can never disagree on the layout. */
static void snap_walk_write(snap_cursor *c, const qwen35moe_state_snap *sn,
                            uint64_t weights_version)
{
    put_u32(c, PN_SNAP_MAGIC);
    put_u32(c, PN_SNAP_FORMAT_VERSION);
    put_u64(c, weights_version);
    put_i32(c, sn->pos);
    put_i32(c, sn->n_ssm);
    put_i32(c, sn->n_attn);

    for (int i = 0; i < sn->n_ssm; i++) {
        const snap_ssm_layer *sl = &sn->ssm[i];
        put_u64(c, (uint64_t)sl->conv_n);
        put_u64(c, (uint64_t)sl->ssm_n);
        put_bytes(c, sl->conv, sl->conv_n * sizeof(float));
        put_bytes(c, sl->ssm, sl->ssm_n * sizeof(float));
    }
    for (int i = 0; i < sn->n_attn; i++) {
        const snap_kv_layer *sk = &sn->kv[i];
        put_i32(c, sk->q4);
        put_i32(c, sk->n_head_kv);
        put_i32(c, sk->head_dim);
        put_i32(c, sk->max_seq);
        put_u64(c, (uint64_t)sk->row_b);
        size_t rows_b = (size_t)sk->n_head_kv * sn->pos * sk->row_b;
        put_bytes(c, sk->k_rows, rows_b);
        put_bytes(c, sk->v_rows, rows_b);
        if (sk->q4) {
            size_t sink_n, win_n;
            snap_kv_shadow_counts(sk, &sink_n, &win_n);
            put_bytes(c, sk->k_sink, sink_n * sizeof(uint16_t));
            put_bytes(c, sk->v_sink, sink_n * sizeof(uint16_t));
            put_bytes(c, sk->k_win, win_n * sizeof(uint16_t));
            put_bytes(c, sk->v_win, win_n * sizeof(uint16_t));
        }
    }
}

int pn_qwen35moe_snap_serialize(const void *snap_v, uint64_t weights_version,
                                unsigned char **bytes_out, size_t *len_out)
{
    const qwen35moe_state_snap *sn = (const qwen35moe_state_snap *)snap_v;
    /* A recurrent snapshot holds no KV rows: not a self-contained session. */
    if (!sn || !bytes_out || !len_out || sn->kv_rows != sn->pos)
        return -1;
    *bytes_out = NULL;
    *len_out = 0;

    /* Pass 1: size (buf == NULL, unbounded cap). */
    snap_cursor sz = { .buf = NULL, .cap = SIZE_MAX, .off = 0, .err = 0 };
    snap_walk_write(&sz, sn, weights_version);
    if (sz.err)
        return -1;

    if (pn_host_admit_check(sz.off, "hybrid session snapshot blob",
                            SNAP_ADMIT_HINT, NULL, 0) != 0)
        return -1;
    unsigned char *buf = malloc(sz.off ? sz.off : 1);
    if (!buf)
        return -1;

    /* Pass 2: write into the exact-size buffer. */
    snap_cursor wr = { .buf = buf, .cap = sz.off, .off = 0, .err = 0 };
    snap_walk_write(&wr, sn, weights_version);
    if (wr.err || wr.off != sz.off) {
        free(buf);
        return -1;
    }
    *bytes_out = buf;
    *len_out = sz.off;
    return 0;
}

/* Read one SSM layer from the cursor into sl (allocates conv/ssm). */
static int snap_ssm_read(snap_cursor *c, snap_ssm_layer *sl)
{
    sl->conv_n = (size_t)get_u64(c);
    sl->ssm_n = (size_t)get_u64(c);
    if (c->err)
        return -1;
    /* Guard against an absurd count that would overflow the malloc. */
    if (sl->conv_n > (SIZE_MAX / sizeof(float)) ||
        sl->ssm_n > (SIZE_MAX / sizeof(float)))
        return -1;
    sl->conv = malloc(sl->conv_n * sizeof(float) + 1);
    sl->ssm = malloc(sl->ssm_n * sizeof(float) + 1);
    if (!sl->conv || !sl->ssm)
        return -1;
    get_bytes(c, sl->conv, sl->conv_n * sizeof(float));
    get_bytes(c, sl->ssm, sl->ssm_n * sizeof(float));
    return c->err ? -1 : 0;
}

/* Read one attention layer from the cursor into sk (allocates rows + shadows). */
static int snap_kv_read(snap_cursor *c, snap_kv_layer *sk, int pos)
{
    sk->q4 = get_i32(c);
    sk->n_head_kv = get_i32(c);
    sk->head_dim = get_i32(c);
    sk->max_seq = get_i32(c);
    sk->row_b = (size_t)get_u64(c);
    if (c->err || sk->n_head_kv < 0 || sk->head_dim < 0 || sk->max_seq < 0 ||
        (sk->q4 != 0 && sk->q4 != 1))
        return -1;

    /* row_b MUST be exactly what the arch derives from (q4, head_dim). A hostile
     * blob that keeps the validated dims (q4/n_head_kv/head_dim/max_seq — the only
     * fields snap_matches_state checks) but inflates row_b would drive an
     * out-of-bounds write on restore: snap_kv_restore strides the live KV buffer by
     * this row_b (kv_rows_copy: live + h*max_seq*row_b), so a larger-than-real row_b
     * writes past the buffer. Pin row_b to the derived value here so it can never
     * diverge from head_dim. */
    size_t expect_row_b = sk->q4 ? (size_t)pn_kvq4_row_bytes(sk->head_dim)
                                 : (size_t)sk->head_dim * sizeof(float);
    if (sk->row_b != expect_row_b)
        return -1;
    /* Guard the rows_b product against size_t overflow (a wrap to a small value
     * would under-allocate k_rows). */
    if (sk->row_b != 0 && sk->n_head_kv != 0 &&
        (size_t)pos > SIZE_MAX / sk->row_b / (size_t)sk->n_head_kv)
        return -1;

    size_t rows_b = (size_t)sk->n_head_kv * (size_t)pos * sk->row_b;
    sk->k_rows = malloc(rows_b ? rows_b : 1);
    sk->v_rows = malloc(rows_b ? rows_b : 1);
    if (!sk->k_rows || !sk->v_rows)
        return -1;
    get_bytes(c, sk->k_rows, rows_b);
    get_bytes(c, sk->v_rows, rows_b);

    if (sk->q4) {
        size_t sink_n, win_n;
        snap_kv_shadow_counts(sk, &sink_n, &win_n);
        sk->k_sink = malloc(sink_n * sizeof(uint16_t));
        sk->v_sink = malloc(sink_n * sizeof(uint16_t));
        sk->k_win = malloc(win_n * sizeof(uint16_t));
        sk->v_win = malloc(win_n * sizeof(uint16_t));
        if (!sk->k_sink || !sk->v_sink || !sk->k_win || !sk->v_win)
            return -1;
        get_bytes(c, sk->k_sink, sink_n * sizeof(uint16_t));
        get_bytes(c, sk->v_sink, sink_n * sizeof(uint16_t));
        get_bytes(c, sk->k_win, win_n * sizeof(uint16_t));
        get_bytes(c, sk->v_win, win_n * sizeof(uint16_t));
    }
    return c->err ? -1 : 0;
}

void *pn_qwen35moe_snap_deserialize(const unsigned char *bytes, size_t len,
                                    uint64_t *weights_version_out)
{
    if (weights_version_out)
        *weights_version_out = 0;
    if (!bytes || len < 24)          /* header is 24 bytes minimum */
        return NULL;

    snap_cursor c = { .buf = (unsigned char *)bytes, .cap = len,
                      .off = 0, .err = 0 };
    if (get_u32(&c) != PN_SNAP_MAGIC)
        return NULL;
    if (get_u32(&c) != PN_SNAP_FORMAT_VERSION)
        return NULL;
    uint64_t wv = get_u64(&c);
    if (weights_version_out)
        *weights_version_out = wv;
    int pos = get_i32(&c);
    int n_ssm = get_i32(&c);
    int n_attn = get_i32(&c);
    if (c.err || pos <= 0 || n_ssm < 0 || n_attn < 0)
        return NULL;

    qwen35moe_state_snap *sn = calloc(1, sizeof(*sn));
    if (!sn)
        return NULL;
    sn->pos = pos;
    sn->kv_rows = pos;
    sn->n_ssm = n_ssm;
    sn->n_attn = n_attn;
    sn->ssm = n_ssm ? calloc((size_t)n_ssm, sizeof(*sn->ssm)) : NULL;
    sn->kv = n_attn ? calloc((size_t)n_attn, sizeof(*sn->kv)) : NULL;
    if ((n_ssm && !sn->ssm) || (n_attn && !sn->kv))
        goto fail;
    /* The decoded snapshot is the blob's payload, so its length bounds it. */
    if (snap_admit(sn, len) != 0)
        goto fail;

    for (int i = 0; i < n_ssm; i++)
        if (snap_ssm_read(&c, &sn->ssm[i]) != 0)
            goto fail;
    for (int i = 0; i < n_attn; i++)
        if (snap_kv_read(&c, &sn->kv[i], pos) != 0)
            goto fail;

    /* Reject trailing garbage: a valid blob is consumed exactly. */
    if (c.off != len)
        goto fail;
    return sn;

fail:
    pn_qwen35moe_state_snapshot_free(sn);
    return NULL;
}
