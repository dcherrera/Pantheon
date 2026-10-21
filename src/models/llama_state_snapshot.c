/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_state_snapshot.c
 * @brief Host-side snapshot/restore of a LLAMA-FAMILY KV cache at a token
 *        position — the session export/restore primitive for every architecture
 *        whose entire per-session state is pure causal attention (llama, qwen2,
 *        qwen3, phi, mistral, mixtral). It is the counterpart of
 *        qwen35moe_state_snapshot.c, which carries a recurrent SSM state as
 *        well; this family has none, so a snapshot is exactly the KV rows
 *        [0, pos) of every layer plus the dims needed to validate a restore.
 *
 *        This is the family the Tenstorrent Blackhole backend serves, and THERE
 *        THE CACHE MAY BE IN ONE OF TWO PLACES. With attention on the host it is
 *        host memory (`tm_zeros(..., TM_F32)`, see llama.c) and a snapshot is a
 *        memcpy that touches no device at all. With `PN_TT_ONCARD=...,attention`
 *        it is CARD DRAM in bf16, the host tensors are never written, and a
 *        snapshot that read them would export a page of zeros and call it a
 *        session. `pn_llama_kv_cache::tt_device` is which, and `kv_layer_move()`
 *        is the one place that asks.
 *
 *        Exact restore is still reachable here and still is not on CUDA (see
 *        prefix_cache.h @issues, where the D2H mirror is not byte-faithful to
 *        the live ring) — because on the card the ROUND TRIP is exact: bf16 is a
 *        strict subset of fp32, so widening out and narrowing back loses nothing.
 *
 * @status Complete: capture/restore and serialize/deserialize, unit-tested
 *         off-hardware (tests/test_llama_state_snapshot.c) and end-to-end
 *         through `pantheon serve` on the card, on both the host-cache and the
 *         card-cache arm.
 * @issues None
 * @todo None
 */

#include "llama.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* One layer's captured KV: rows [0, pos) per head, packed [n_kv_head][pos]
 * [head_dim] (the live cache strides by max_seq). fp32 throughout — this family
 * has no quantized-KV mode. */
typedef struct {
    float *k_rows;
    float *v_rows;
} llama_snap_layer;

/* Whole-cache snapshot at one position (opaque to callers). */
typedef struct {
    int pos;
    int n_layers;
    int n_kv_head;
    int head_dim;
    int max_seq;            /* the SOURCE cache's ring length (diagnostic) */
    llama_snap_layer *lay;
} llama_kv_snap;

/* Elements per layer in the packed representation. */
static size_t snap_layer_elems(const llama_kv_snap *sn)
{
    return (size_t)sn->n_kv_head * (size_t)sn->pos * (size_t)sn->head_dim;
}

/* Copy `pos` rows per head between packed ([pos] stride) and live ([max_seq]
 * stride) layouts. dir_capture != 0: live -> packed; else packed -> live. */
static void kv_rows_copy(float *packed, float *live, int n_kv_head, int max_seq,
                         int pos, int head_dim, int dir_capture)
{
    for (int h = 0; h < n_kv_head; h++) {
        float *p = packed + (size_t)h * (size_t)pos * (size_t)head_dim;
        float *l = live + (size_t)h * (size_t)max_seq * (size_t)head_dim;
        size_t n = (size_t)pos * (size_t)head_dim * sizeof(float);
        if (dir_capture)
            memcpy(p, l, n);
        else
            memcpy(l, p, n);
    }
}

/* ============================================================
 * WHERE THE ROWS COME FROM — host memory, or the card
 * ============================================================
 *
 * The trade was decided in pn_tt_device.h and the arithmetic is not close. A host MIRROR —
 * downloading k and v every layer to keep the fp32 tensors current — costs 49,152 B a token
 * of ROW-0 reads EVERY token, because a 64-byte card->host read alignment makes row 0 cost
 * twice its payload. Downloading the whole cache once at export reads WHOLE tiles with no
 * waste: 24,576 B a position. **The mirror therefore moves exactly 2x the bytes of the export
 * it would make free, and it charges every session including the ones nobody exports.**
 * Export is an event; decode is a loop; the cost goes on the event.
 *
 * Measured consequence at 4.7 MB/s card->host: ~10.2 s to export a 2048-token session, and
 * ~0.18 s to restore one (host->card is 287 MB/s, ~61x faster).
 *
 * **THE BLOB FORMAT DOES NOT CHANGE, deliberately.** The card holds bf16, so an fp32 payload
 * stores twice the bytes the state actually has — but bf16 is a strict subset of fp32, so
 * widening out and narrowing back is EXACT, and holding PN_LSNAP_FORMAT_VERSION at 1 means
 * every blob a client already holds still restores. Halving a file that lives on the user's
 * machine is not worth forking the format for.
 */

/* Move one layer's K and V between wherever they live and the packed [head][pos][dim]
 * arrays. `dir_capture` non-zero is live -> packed; zero is packed -> live.
 *
 * Returns 0, or -1 if a device transfer failed — which must NOT be absorbed. A snapshot that
 * quietly keeps whatever the buffer held is the zero-page export in a different disguise. */
static int kv_layer_move(const pn_llama_kv_cache *cache, int layer, float *k_packed,
                         float *v_packed, int n_kv_head, int max_seq, int pos, int head_dim,
                         int dir_capture)
{
    if (cache->kv_move) {
        if (cache->kv_move(cache->kv_device, layer, 0, pos, k_packed, dir_capture) != 0)
            return -1;
        return cache->kv_move(cache->kv_device, layer, 1, pos, v_packed, dir_capture) == 0
                   ? 0 : -1;
    }
    kv_rows_copy(k_packed, (float *)cache->k_cache[layer]->data, n_kv_head, max_seq, pos,
                 head_dim, dir_capture);
    kv_rows_copy(v_packed, (float *)cache->v_cache[layer]->data, n_kv_head, max_seq, pos,
                 head_dim, dir_capture);
    return 0;
}

/* Read the cache geometry out of layer 0's tensors and check every other layer
 * agrees. Returns 0 on success. A cache whose layers disagree is not a shape
 * this format can describe, and silently snapshotting layer 0's dims for all of
 * them would corrupt the restore. */
static int kv_cache_geometry(const pn_llama_kv_cache *cache, int *n_kv_head,
                             int *max_seq, int *head_dim)
{
    if (!cache || cache->n_layers <= 0 || !cache->k_cache || !cache->v_cache)
        return -1;
    const tm_tensor *k0 = cache->k_cache[0];
    if (!k0 || k0->ndim != 3)
        return -1;
    *n_kv_head = (int)k0->shape[0];
    *max_seq   = (int)k0->shape[1];
    *head_dim  = (int)k0->shape[2];
    if (*n_kv_head <= 0 || *max_seq <= 0 || *head_dim <= 0)
        return -1;
    for (int i = 0; i < cache->n_layers; i++) {
        const tm_tensor *k = cache->k_cache[i];
        const tm_tensor *v = cache->v_cache[i];
        if (!k || !v || !k->data || !v->data || k->ndim != 3 || v->ndim != 3)
            return -1;
        if (k->shape[0] != *n_kv_head || k->shape[1] != *max_seq ||
            k->shape[2] != *head_dim ||
            v->shape[0] != *n_kv_head || v->shape[1] != *max_seq ||
            v->shape[2] != *head_dim)
            return -1;
    }
    return 0;
}

void pn_llama_kv_snapshot_free(void *snap_v)
{
    llama_kv_snap *sn = (llama_kv_snap *)snap_v;
    if (!sn)
        return;
    if (sn->lay) {
        for (int i = 0; i < sn->n_layers; i++) {
            free(sn->lay[i].k_rows);
            free(sn->lay[i].v_rows);
        }
        free(sn->lay);
    }
    free(sn);
}

void *pn_llama_kv_snapshot(const void *cache_v, int pos)
{
    const pn_llama_kv_cache *cache = (const pn_llama_kv_cache *)cache_v;
    int n_kv_head = 0, max_seq = 0, head_dim = 0;

    if (pos <= 0)
        return NULL;
    if (kv_cache_geometry(cache, &n_kv_head, &max_seq, &head_dim) != 0)
        return NULL;
    /* The live cache is a RING (`cache_pos = pos % max_seq` in every forward),
     * so once pos exceeds max_seq the rows [0, pos) no longer exist anywhere —
     * they were overwritten. Refuse rather than export a wrapped cache as if it
     * were linear. */
    if (pos > max_seq)
        return NULL;

    llama_kv_snap *sn = calloc(1, sizeof(*sn));
    if (!sn)
        return NULL;
    sn->pos = pos;
    sn->n_layers = cache->n_layers;
    sn->n_kv_head = n_kv_head;
    sn->head_dim = head_dim;
    sn->max_seq = max_seq;
    sn->lay = calloc((size_t)cache->n_layers, sizeof(*sn->lay));
    if (!sn->lay)
        goto fail;

    size_t elems = snap_layer_elems(sn);
    for (int i = 0; i < cache->n_layers; i++) {
        sn->lay[i].k_rows = malloc(elems * sizeof(float));
        sn->lay[i].v_rows = malloc(elems * sizeof(float));
        if (!sn->lay[i].k_rows || !sn->lay[i].v_rows)
            goto fail;
        if (kv_layer_move(cache, i, sn->lay[i].k_rows, sn->lay[i].v_rows, n_kv_head,
                          max_seq, pos, head_dim, 1) != 0)
            goto fail;
    }
    return sn;

fail:
    pn_llama_kv_snapshot_free(sn);
    return NULL;
}

int pn_llama_kv_restore(void *cache_v, const void *snap_v)
{
    pn_llama_kv_cache *cache = (pn_llama_kv_cache *)cache_v;
    const llama_kv_snap *sn = (const llama_kv_snap *)snap_v;
    int n_kv_head = 0, max_seq = 0, head_dim = 0;

    if (!sn)
        return -1;
    if (kv_cache_geometry(cache, &n_kv_head, &max_seq, &head_dim) != 0)
        return -1;
    if (sn->n_layers != cache->n_layers || sn->n_kv_head != n_kv_head ||
        sn->head_dim != head_dim)
        return -1;
    /* pos is attacker-controlled (it arrives inside a client-held blob), so
     * bound it against THIS cache's ring rather than the one it was captured
     * on. max_seq itself is deliberately NOT required to match: the rows are
     * repacked on the way in, so a cache configured with a longer ring accepts
     * a shorter capture. */
    if (sn->pos <= 0 || sn->pos > max_seq)
        return -1;

    for (int i = 0; i < cache->n_layers; i++) {
        if (kv_layer_move(cache, i, sn->lay[i].k_rows, sn->lay[i].v_rows, n_kv_head, max_seq,
                          sn->pos, head_dim, 0) != 0)
            return -1;
    }
    return 0;
}

int pn_llama_kv_snapshot_pos(const void *snap_v)
{
    const llama_kv_snap *sn = (const llama_kv_snap *)snap_v;
    return sn ? sn->pos : -1;
}

/* ============================================================
 * Serialize / deserialize — the session warm-tier blob
 *
 * One contiguous little-endian blob, framed exactly like the qwen35moe one
 * (put_u32/put_u64 write LE regardless of host endianness) so the two are
 * distinguishable by MAGIC alone and never by length or by luck. Layout:
 *
 *   [ header, 36 bytes ]
 *     u32  magic            = PN_LSNAP_MAGIC ("PNLK" as LE bytes)
 *     u32  format_version   = PN_LSNAP_FORMAT_VERSION
 *     u64  weights_version  (opaque model-identity tag; validated by caller)
 *     i32  pos
 *     i32  n_layers
 *     i32  n_kv_head
 *     i32  head_dim
 *     i32  max_seq          (source ring length; diagnostic, see restore)
 *   [ n_layers × ]
 *     f32  k_rows[n_kv_head * pos * head_dim]
 *     f32  v_rows[...]
 *
 * f32 payloads are copied as raw bytes: this fleet and every client on it are
 * little-endian IEEE-754, so no per-element swap is done. The framing fields
 * ARE explicitly LE, and format_version gates any future change.
 * ============================================================ */

#define PN_LSNAP_MAGIC          0x4B4C4E50u   /* 'P''N''L''K' little-endian */
#define PN_LSNAP_FORMAT_VERSION 1u
#define PN_LSNAP_HEADER_BYTES   36u

/* Bounded cursor over a byte buffer (write or read). `err` latches on the first
 * overflow so callers check it once at the end instead of per-field. */
typedef struct {
    unsigned char *buf;   /* NULL in the sizing pass (writer) */
    size_t         cap;
    size_t         off;
    int            err;
} lsnap_cursor;

static void put_bytes(lsnap_cursor *c, const void *src, size_t n)
{
    if (c->err)
        return;
    if (c->off + n > c->cap) { c->err = 1; return; }
    if (c->buf && n)
        memcpy(c->buf + c->off, src, n);
    c->off += n;
}

static void put_u32(lsnap_cursor *c, uint32_t v)
{
    unsigned char b[4] = { (unsigned char)(v), (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    put_bytes(c, b, 4);
}

static void put_i32(lsnap_cursor *c, int32_t v) { put_u32(c, (uint32_t)v); }

static void put_u64(lsnap_cursor *c, uint64_t v)
{
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    put_bytes(c, b, 8);
}

static void get_bytes(lsnap_cursor *c, void *dst, size_t n)
{
    if (c->err)
        return;
    if (c->off + n > c->cap) { c->err = 1; return; }
    if (n)
        memcpy(dst, c->buf + c->off, n);
    c->off += n;
}

static uint32_t get_u32(lsnap_cursor *c)
{
    unsigned char b[4] = {0};
    get_bytes(c, b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static int32_t get_i32(lsnap_cursor *c) { return (int32_t)get_u32(c); }

static uint64_t get_u64(lsnap_cursor *c)
{
    unsigned char b[8] = {0};
    get_bytes(c, b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

/* Walk the whole snapshot through a cursor: write (buf != NULL) or size-only
 * (buf == NULL, cap = SIZE_MAX). ONE walk shared by both passes so the sizing
 * pass and the write pass can never disagree on the layout. */
static void lsnap_walk_write(lsnap_cursor *c, const llama_kv_snap *sn,
                             uint64_t weights_version)
{
    put_u32(c, PN_LSNAP_MAGIC);
    put_u32(c, PN_LSNAP_FORMAT_VERSION);
    put_u64(c, weights_version);
    put_i32(c, sn->pos);
    put_i32(c, sn->n_layers);
    put_i32(c, sn->n_kv_head);
    put_i32(c, sn->head_dim);
    put_i32(c, sn->max_seq);

    size_t rows_b = snap_layer_elems(sn) * sizeof(float);
    for (int i = 0; i < sn->n_layers; i++) {
        put_bytes(c, sn->lay[i].k_rows, rows_b);
        put_bytes(c, sn->lay[i].v_rows, rows_b);
    }
}

int pn_llama_kv_snap_serialize(const void *snap_v, uint64_t weights_version,
                               unsigned char **bytes_out, size_t *len_out)
{
    const llama_kv_snap *sn = (const llama_kv_snap *)snap_v;
    if (!sn || !bytes_out || !len_out)
        return -1;
    *bytes_out = NULL;
    *len_out = 0;

    /* Pass 1: size (buf == NULL, unbounded cap). */
    lsnap_cursor sz = { .buf = NULL, .cap = SIZE_MAX, .off = 0, .err = 0 };
    lsnap_walk_write(&sz, sn, weights_version);
    if (sz.err)
        return -1;

    unsigned char *buf = malloc(sz.off ? sz.off : 1);
    if (!buf)
        return -1;

    /* Pass 2: write into the exact-size buffer. */
    lsnap_cursor wr = { .buf = buf, .cap = sz.off, .off = 0, .err = 0 };
    lsnap_walk_write(&wr, sn, weights_version);
    if (wr.err || wr.off != sz.off) {
        free(buf);
        return -1;
    }
    *bytes_out = buf;
    *len_out = sz.off;
    return 0;
}

void *pn_llama_kv_snap_deserialize(const unsigned char *bytes, size_t len,
                                   uint64_t *weights_version_out)
{
    if (weights_version_out)
        *weights_version_out = 0;
    if (!bytes || len < PN_LSNAP_HEADER_BYTES)
        return NULL;

    lsnap_cursor c = { .buf = (unsigned char *)bytes, .cap = len,
                       .off = 0, .err = 0 };
    if (get_u32(&c) != PN_LSNAP_MAGIC)
        return NULL;
    if (get_u32(&c) != PN_LSNAP_FORMAT_VERSION)
        return NULL;
    uint64_t wv = get_u64(&c);
    if (weights_version_out)
        *weights_version_out = wv;
    int pos = get_i32(&c);
    int n_layers = get_i32(&c);
    int n_kv_head = get_i32(&c);
    int head_dim = get_i32(&c);
    int max_seq = get_i32(&c);
    if (c.err || pos <= 0 || n_layers <= 0 || n_kv_head <= 0 || head_dim <= 0 ||
        max_seq <= 0 || pos > max_seq)
        return NULL;

    llama_kv_snap *sn = calloc(1, sizeof(*sn));
    if (!sn)
        return NULL;
    sn->pos = pos;
    sn->n_layers = n_layers;
    sn->n_kv_head = n_kv_head;
    sn->head_dim = head_dim;
    sn->max_seq = max_seq;
    sn->lay = calloc((size_t)n_layers, sizeof(*sn->lay));
    if (!sn->lay)
        goto fail;

    /* Guard the element product against size_t overflow before it becomes a
     * malloc size: a wrap to a small value would under-allocate the rows. */
    size_t elems = snap_layer_elems(sn);
    if ((size_t)pos > SIZE_MAX / (size_t)n_kv_head / (size_t)head_dim ||
        elems > SIZE_MAX / sizeof(float))
        goto fail;
    size_t rows_b = elems * sizeof(float);

    for (int i = 0; i < n_layers; i++) {
        sn->lay[i].k_rows = malloc(rows_b);
        sn->lay[i].v_rows = malloc(rows_b);
        if (!sn->lay[i].k_rows || !sn->lay[i].v_rows)
            goto fail;
        get_bytes(&c, sn->lay[i].k_rows, rows_b);
        get_bytes(&c, sn->lay[i].v_rows, rows_b);
        if (c.err)
            goto fail;
    }

    /* Reject trailing garbage: a valid blob is consumed exactly. */
    if (c.off != len)
        goto fail;
    return sn;

fail:
    pn_llama_kv_snapshot_free(sn);
    return NULL;
}
