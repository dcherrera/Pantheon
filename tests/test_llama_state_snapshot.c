/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_llama_state_snapshot.c
 * @brief Off-hardware gate for the llama-family session snapshot — the
 *        primitive session export/restore rides on for every architecture whose
 *        per-session state is a plain causal KV cache, which is the family the
 *        Tenstorrent backend serves.
 *
 * What it is built to catch, and why each choice is the way it is:
 *
 *  - **Byte comparison, never a correlation score.** A layout change scored by
 *    PCC is not scored at all (DEV_NOTES §3): PCC is blind to a consistent
 *    permutation, and every failure this format can have — wrong stride, wrong
 *    layer, wrong head, a dropped row — is a permutation or a truncation.
 *  - **The fill carries low mantissa bits.** Small-integer test data makes two
 *    deliberately-different builds byte-exact (DEV_NOTES §3); an irrational
 *    step does not.
 *  - **The per-cell seed is an OFFSET, not a modulus.** Seeding
 *    `(element + index * 7) % 16` made cores 1 and 17 byte-identical once. Here
 *    every (layer, head, row, col) gets a distinct value by construction, and
 *    PAIRWISE DISTINCTNESS is asserted independently of the reference — so a
 *    collapse onto one layer's or one head's data cannot be covered for by a
 *    reference that is wrong in the same way.
 *  - **The destination is POISONED before every restore**, and the rows OUTSIDE
 *    [0, pos) are checked to still hold the poison. "It returns the right
 *    bytes" does not distinguish a restore from a cache that was already right.
 *  - **The other family's blob is fed in on purpose.** The two snapshot formats
 *    are told apart by MAGIC; that has to be a check, not a hope.
 *
 * @status Complete. Six mutations proved red — see the plan.
 * @issues None
 * @todo None
 */

#include "../src/models/llama.h"
#include "../src/core/session_snapshot.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The qwen35 snapshot unit is linked in only so the dispatcher's OTHER arm is a
 * real call and not a stub — that is what makes "a llama blob is refused by the
 * qwen35 parser" a statement about the shipping parser. Its one external is the
 * forward's content-generation counter, which no test here reaches. */
unsigned pn_qwen35moe_kv_next_content_gen(void);
unsigned pn_qwen35moe_kv_next_content_gen(void) { return 1u; }

static int checks = 0;
static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            printf("  FAIL: ");                                                \
            printf(__VA_ARGS__);                                               \
            printf("   (%s:%d)\n", __FILE__, __LINE__);                        \
        }                                                                      \
    } while (0)

/* ============================================================
 * A hand-built KV cache. Deliberately NOT pn_llama_kv_cache_new(): that lives
 * in llama.c and would drag the whole model in, and this file must stay a
 * host-only unit with no engine and no accelerator.
 * ============================================================ */

#define POISON 12345.5f

static tm_tensor *make_kv_tensor(int n_kv_head, int max_seq, int head_dim)
{
    tm_tensor *t = calloc(1, sizeof(*t));
    if (!t)
        return NULL;
    t->ndim = 3;
    t->shape[0] = n_kv_head;
    t->shape[1] = max_seq;
    t->shape[2] = head_dim;
    t->numel = (int64_t)n_kv_head * max_seq * head_dim;
    t->dtype = TM_F32;
    t->data = malloc((size_t)t->numel * sizeof(float));
    if (!t->data) { free(t); return NULL; }
    t->owns_data = true;
    return t;
}

static pn_llama_kv_cache *make_cache(int n_layers, int n_kv_head, int max_seq,
                                     int head_dim)
{
    pn_llama_kv_cache *c = calloc(1, sizeof(*c));
    if (!c)
        return NULL;
    c->n_layers = n_layers;
    c->max_seq = max_seq;
    c->k_cache = calloc((size_t)n_layers, sizeof(tm_tensor *));
    c->v_cache = calloc((size_t)n_layers, sizeof(tm_tensor *));
    if (!c->k_cache || !c->v_cache)
        return NULL;
    for (int i = 0; i < n_layers; i++) {
        c->k_cache[i] = make_kv_tensor(n_kv_head, max_seq, head_dim);
        c->v_cache[i] = make_kv_tensor(n_kv_head, max_seq, head_dim);
        if (!c->k_cache[i] || !c->v_cache[i])
            return NULL;
    }
    return c;
}

static void free_cache(pn_llama_kv_cache *c)
{
    if (!c)
        return;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->k_cache && c->k_cache[i]) { free(c->k_cache[i]->data); free(c->k_cache[i]); }
        if (c->v_cache && c->v_cache[i]) { free(c->v_cache[i]->data); free(c->v_cache[i]); }
    }
    free(c->k_cache);
    free(c->v_cache);
    free(c);
}

/* A value unique to (kv, layer, head, row, col), carrying low mantissa bits.
 * The identifying part is an OFFSET on a base with no modulus anywhere, so no
 * two cells of the whole cache can collide. */
static float cell_value(int kv, int layer, int head, int row, int col,
                        int n_kv_head, int max_seq, int head_dim)
{
    long idx = (((long)kv * 64 + layer) * n_kv_head + head);
    idx = (idx * max_seq + row) * head_dim + col;
    return (float)idx * 0.31830988618379f + 0.0009765625f * (float)(idx & 1023);
}

static void fill_cache(pn_llama_kv_cache *c, int n_kv_head, int max_seq,
                       int head_dim)
{
    for (int l = 0; l < c->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            float *d = (float *)(kv ? c->v_cache[l]->data : c->k_cache[l]->data);
            for (int h = 0; h < n_kv_head; h++)
                for (int r = 0; r < max_seq; r++)
                    for (int col = 0; col < head_dim; col++)
                        d[((size_t)h * max_seq + r) * head_dim + col] =
                            cell_value(kv, l, h, r, col, n_kv_head, max_seq, head_dim);
        }
    }
}

static void poison_cache(pn_llama_kv_cache *c, int n_kv_head, int max_seq,
                         int head_dim)
{
    size_t n = (size_t)n_kv_head * max_seq * head_dim;
    for (int l = 0; l < c->n_layers; l++) {
        float *k = (float *)c->k_cache[l]->data;
        float *v = (float *)c->v_cache[l]->data;
        for (size_t i = 0; i < n; i++) { k[i] = POISON; v[i] = POISON; }
    }
}

/* ============================================================
 * Stage 1 — capture / restore round trip, byte-exact
 * ============================================================ */

static void test_capture_restore(void)
{
    const int L = 4, H = 3, S = 16, D = 8, POS = 11;
    printf("--- capture/restore round trip (L=%d H=%d S=%d D=%d pos=%d) ---\n",
           L, H, S, D, POS);

    pn_llama_kv_cache *src = make_cache(L, H, S, D);
    pn_llama_kv_cache *dst = make_cache(L, H, S, D);
    CHECK(src && dst, "caches allocated");
    if (!src || !dst) return;
    fill_cache(src, H, S, D);
    poison_cache(dst, H, S, D);

    void *snap = pn_llama_kv_snapshot(src, POS);
    CHECK(snap != NULL, "capture at pos=%d succeeded", POS);
    CHECK(pn_llama_kv_snapshot_pos(snap) == POS, "snapshot reports pos=%d", POS);
    CHECK(pn_llama_kv_restore(dst, snap) == 0, "restore into a poisoned cache");

    /* Rows [0, POS) must be byte-identical to the source; rows [POS, S) must
     * still be poison — a restore that writes the whole ring, or one that
     * strides by max_seq instead of pos, fails one of the two. */
    int rows_in = 0, rows_out = 0;
    for (int l = 0; l < L; l++) {
        for (int kv = 0; kv < 2; kv++) {
            const float *s = (const float *)(kv ? src->v_cache[l]->data : src->k_cache[l]->data);
            const float *d = (const float *)(kv ? dst->v_cache[l]->data : dst->k_cache[l]->data);
            for (int h = 0; h < H; h++) {
                const float *sr = s + (size_t)h * S * D;
                const float *dr = d + (size_t)h * S * D;
                if (memcmp(sr, dr, (size_t)POS * D * sizeof(float)) != 0)
                    rows_in++;
                for (int r = POS; r < S; r++)
                    for (int col = 0; col < D; col++)
                        if (dr[(size_t)r * D + col] != POISON)
                            rows_out++;
            }
        }
    }
    CHECK(rows_in == 0, "every (layer,kv,head) rows [0,%d) byte-identical "
                        "(%d differed)", POS, rows_in);
    CHECK(rows_out == 0, "rows [%d,%d) still poison (%d overwritten)",
          POS, S, rows_out);

    /* Pairwise distinctness — independent of the reference. If the capture
     * collapsed every layer onto layer 0 (or every head onto head 0) the
     * comparison above would still pass whenever the reference were wrong the
     * same way; this cannot be. */
    int dup = 0;
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++) {
            const float *a = (const float *)dst->k_cache[l]->data + (size_t)h * S * D;
            for (int l2 = 0; l2 < L; l2++)
                for (int h2 = 0; h2 < H; h2++) {
                    if (l2 < l || (l2 == l && h2 <= h)) continue;
                    const float *b = (const float *)dst->k_cache[l2]->data + (size_t)h2 * S * D;
                    if (memcmp(a, b, (size_t)POS * D * sizeof(float)) == 0)
                        dup++;
                }
        }
    CHECK(dup == 0, "all %d (layer,head) restored panels differ from each other "
                    "(%d collisions)", L * H, dup);

    /* Refusals that protect the ring. */
    CHECK(pn_llama_kv_snapshot(src, 0) == NULL, "capture at pos=0 refused");
    CHECK(pn_llama_kv_snapshot(src, -1) == NULL, "capture at pos=-1 refused");
    CHECK(pn_llama_kv_snapshot(src, S) != NULL, "capture at pos==max_seq allowed");
    void *over = pn_llama_kv_snapshot(src, S + 1);
    CHECK(over == NULL, "capture at pos>max_seq refused (the cache is a RING)");
    pn_llama_kv_snapshot_free(over);

    /* Restore into a cache of a different shape must refuse, not truncate. */
    pn_llama_kv_cache *narrow = make_cache(L, H, S, D / 2);
    pn_llama_kv_cache *shallow = make_cache(L - 1, H, S, D);
    pn_llama_kv_cache *fewheads = make_cache(L, H - 1, S, D);
    CHECK(pn_llama_kv_restore(narrow, snap) != 0, "restore refuses a head_dim mismatch");
    CHECK(pn_llama_kv_restore(shallow, snap) != 0, "restore refuses an n_layers mismatch");
    CHECK(pn_llama_kv_restore(fewheads, snap) != 0, "restore refuses an n_kv_head mismatch");

    /* A SHORTER ring must refuse: `pos` arrives inside a client-held blob, and
     * writing pos rows into a ring of max_seq < pos walks off the end of the
     * live cache. This is the one bound that is a property of the DESTINATION
     * and not of the blob, so nothing the parser checks can cover it. */
    pn_llama_kv_cache *shortring = make_cache(L, H, POS - 1, D);
    CHECK(pn_llama_kv_restore(shortring, snap) != 0,
          "restore REFUSES a ring shorter than the snapshot's pos");
    free_cache(shortring);

    /* A LONGER ring is fine — the rows are repacked on the way in. */
    pn_llama_kv_cache *longer = make_cache(L, H, S * 2, D);
    poison_cache(longer, H, S * 2, D);
    CHECK(pn_llama_kv_restore(longer, snap) == 0, "restore accepts a longer ring");
    int diff_long = 0;
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++) {
            const float *s = (const float *)src->k_cache[l]->data + (size_t)h * S * D;
            const float *d = (const float *)longer->k_cache[l]->data + (size_t)h * (S * 2) * D;
            if (memcmp(s, d, (size_t)POS * D * sizeof(float)) != 0)
                diff_long++;
        }
    CHECK(diff_long == 0, "longer-ring restore is byte-identical over [0,%d)", POS);

    free_cache(narrow); free_cache(shallow); free_cache(fewheads); free_cache(longer);
    pn_llama_kv_snapshot_free(snap);
    free_cache(src);
    free_cache(dst);
}

/* ============================================================
 * Stage 2 — serialize / deserialize
 * ============================================================ */

static void test_serialize(void)
{
    const int L = 3, H = 2, S = 12, D = 8, POS = 7;
    const uint64_t WV = 0xA1B2C3D4E5F60718ull;
    printf("--- serialize / deserialize ---\n");

    pn_llama_kv_cache *src = make_cache(L, H, S, D);
    CHECK(src != NULL, "cache allocated");
    if (!src) return;
    fill_cache(src, H, S, D);

    void *snap = pn_llama_kv_snapshot(src, POS);
    CHECK(snap != NULL, "capture");

    unsigned char *b1 = NULL, *b2 = NULL;
    size_t l1 = 0, l2 = 0;
    CHECK(pn_llama_kv_snap_serialize(snap, WV, &b1, &l1) == 0, "serialize");
    size_t expect = 36 + (size_t)L * 2 * (size_t)H * POS * D * sizeof(float);
    CHECK(l1 == expect, "blob is %zu bytes (header 36 + payload), got %zu",
          expect, l1);

    uint64_t wv_out = 0;
    void *snap2 = pn_llama_kv_snap_deserialize(b1, l1, &wv_out);
    CHECK(snap2 != NULL, "deserialize");
    CHECK(wv_out == WV, "weights_version round-trips");
    CHECK(pn_llama_kv_snapshot_pos(snap2) == POS, "pos round-trips");

    CHECK(pn_llama_kv_snap_serialize(snap2, WV, &b2, &l2) == 0, "re-serialize");
    CHECK(l1 == l2 && b1 && b2 && memcmp(b1, b2, l1) == 0,
          "serialize -> deserialize -> serialize is BYTE-IDENTICAL");

    /* And the deserialized snapshot restores the same bytes the live one does. */
    pn_llama_kv_cache *dst = make_cache(L, H, S, D);
    poison_cache(dst, H, S, D);
    CHECK(pn_llama_kv_restore(dst, snap2) == 0, "restore from the parsed blob");
    int diff = 0;
    for (int l = 0; l < L; l++)
        for (int kv = 0; kv < 2; kv++) {
            const float *s = (const float *)(kv ? src->v_cache[l]->data : src->k_cache[l]->data);
            const float *d = (const float *)(kv ? dst->v_cache[l]->data : dst->k_cache[l]->data);
            for (int h = 0; h < H; h++)
                if (memcmp(s + (size_t)h * S * D, d + (size_t)h * S * D,
                           (size_t)POS * D * sizeof(float)) != 0)
                    diff++;
        }
    CHECK(diff == 0, "blob round-trip restores byte-identical rows (%d differed)",
          diff);

    /* --- hostile / malformed blobs --- */
    unsigned char *bad = malloc(l1);
    CHECK(bad != NULL, "scratch");
    if (bad) {
        memcpy(bad, b1, l1);
        bad[0] ^= 0xFF;
        CHECK(pn_llama_kv_snap_deserialize(bad, l1, NULL) == NULL,
              "corrupt magic refused");
        memcpy(bad, b1, l1);
        bad[4] = 0x02;                       /* format_version = 2 */
        CHECK(pn_llama_kv_snap_deserialize(bad, l1, NULL) == NULL,
              "unknown format_version refused");
        memcpy(bad, b1, l1);
        bad[16] = 0xFF; bad[17] = 0xFF; bad[18] = 0xFF; bad[19] = 0x7F; /* pos huge */
        CHECK(pn_llama_kv_snap_deserialize(bad, l1, NULL) == NULL,
              "absurd pos refused");
        memcpy(bad, b1, l1);
        bad[16] = 0; bad[17] = 0; bad[18] = 0; bad[19] = 0;             /* pos = 0 */
        CHECK(pn_llama_kv_snap_deserialize(bad, l1, NULL) == NULL,
              "pos=0 refused");
        free(bad);
    }
    CHECK(pn_llama_kv_snap_deserialize(b1, l1 - 1, NULL) == NULL,
          "truncated blob refused");
    CHECK(pn_llama_kv_snap_deserialize(b1, 8, NULL) == NULL,
          "sub-header blob refused");
    CHECK(pn_llama_kv_snap_deserialize(NULL, l1, NULL) == NULL, "NULL refused");

    unsigned char *longer = malloc(l1 + 4);
    if (longer) {
        memcpy(longer, b1, l1);
        memset(longer + l1, 0, 4);
        CHECK(pn_llama_kv_snap_deserialize(longer, l1 + 4, NULL) == NULL,
              "TRAILING GARBAGE refused (a valid blob is consumed exactly)");
        free(longer);
    }

    /* The other family's blob: "PNS1" must not parse as "PNLK". Built by hand
     * so this test needs no qwen35 state to produce one. */
    unsigned char other[64];
    memcpy(other, b1, sizeof(other) < l1 ? sizeof(other) : l1);
    other[0] = 'P'; other[1] = 'N'; other[2] = 'S'; other[3] = '1';
    CHECK(pn_llama_kv_snap_deserialize(other, sizeof(other), NULL) == NULL,
          "a qwen35-family 'PNS1' blob is refused by the llama parser");

    pn_llama_kv_snapshot_free(snap);
    pn_llama_kv_snapshot_free(snap2);
    free(b1);
    free(b2);
    free_cache(src);
    free_cache(dst);
}

/* ============================================================
 * Stage 3 — the arch dispatcher routes by name and refuses the rest
 * ============================================================ */

static void test_dispatcher(void)
{
    const int L = 2, H = 2, S = 8, D = 4, POS = 5;
    printf("--- session_snapshot dispatcher ---\n");

    CHECK(pn_session_snapshot_supported(PN_ARCH_LLAMA), "llama supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_QWEN2), "qwen2 supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_QWEN3), "qwen3 supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_PHI), "phi supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_MISTRAL), "mistral supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_MIXTRAL), "mixtral supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_QWEN35MOE), "qwen35moe supported");
    CHECK(pn_session_snapshot_supported(PN_ARCH_BONSAI), "bonsai supported");
    CHECK(!pn_session_snapshot_supported(PN_ARCH_RWKVX),
          "RWKVX REFUSED — a recurrent state is not a KV cache");
    CHECK(!pn_session_snapshot_supported(PN_ARCH_UNKNOWN), "unknown refused");

    pn_llama_kv_cache *src = make_cache(L, H, S, D);
    pn_llama_kv_cache *dst = make_cache(L, H, S, D);
    CHECK(src && dst, "caches allocated");
    if (!src || !dst) return;
    fill_cache(src, H, S, D);
    poison_cache(dst, H, S, D);

    void *snap = pn_session_snapshot_capture(PN_ARCH_LLAMA, src, POS);
    CHECK(snap != NULL, "dispatcher captured a llama session");
    CHECK(pn_session_snapshot_pos(PN_ARCH_LLAMA, snap) == POS,
          "dispatcher reports pos");
    CHECK(pn_session_snapshot_restore(PN_ARCH_LLAMA, dst, snap) == 0,
          "dispatcher restored it");
    int diff = 0;
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++)
            if (memcmp((const float *)src->k_cache[l]->data + (size_t)h * S * D,
                       (const float *)dst->k_cache[l]->data + (size_t)h * S * D,
                       (size_t)POS * D * sizeof(float)) != 0)
                diff++;
    CHECK(diff == 0, "dispatcher round trip is byte-identical");

    unsigned char *blob = NULL;
    size_t blen = 0;
    CHECK(pn_session_snapshot_serialize(PN_ARCH_LLAMA, snap, 7ull, &blob, &blen) == 0,
          "dispatcher serialized");
    uint64_t wv = 0;
    void *reparsed = pn_session_snapshot_deserialize(PN_ARCH_LLAMA, blob, blen, &wv);
    CHECK(reparsed != NULL && wv == 7ull, "dispatcher deserialized");

    /* A llama blob offered to the qwen35 arm must be refused by magic, and an
     * unsupported arch must refuse everything rather than guess a family. */
    CHECK(pn_session_snapshot_deserialize(PN_ARCH_QWEN35MOE, blob, blen, NULL) == NULL,
          "a llama 'PNLK' blob is refused by the qwen35 parser");
    CHECK(pn_session_snapshot_capture(PN_ARCH_RWKVX, src, POS) == NULL,
          "capture on an unsupported arch refused");
    CHECK(pn_session_snapshot_restore(PN_ARCH_RWKVX, dst, snap) != 0,
          "restore on an unsupported arch refused");
    CHECK(pn_session_snapshot_deserialize(PN_ARCH_RWKVX, blob, blen, NULL) == NULL,
          "deserialize on an unsupported arch refused");

    pn_session_snapshot_free(PN_ARCH_LLAMA, reparsed);
    pn_session_snapshot_free(PN_ARCH_LLAMA, snap);
    pn_session_snapshot_free(PN_ARCH_RWKVX, NULL);   /* NULL-safe on every arm */
    free(blob);
    free_cache(src);
    free_cache(dst);
}

/* ============================================================
 * Stage 4 — the DEVICE-RESIDENT cache, through the kv_move hook
 *
 * When the Tenstorrent backend runs attention on the card, the KV cache is card DRAM in bf16
 * and `k_cache`/`v_cache` are NEVER WRITTEN. A snapshot that read them would export a page of
 * zeros and call it a session, so `pn_llama_kv_cache::kv_move` redirects the rows to the
 * device.
 *
 * A fake device stands in for the card, and that is the point of making the hook a function
 * pointer rather than an `#ifdef`: this branch is covered by a test that can go red on any
 * machine, instead of by code no off-hardware suite ever compiles.
 *
 * The fake stores bf16, because the card does. That is what makes the round-trip check a
 * claim about the real system: bf16 is a strict subset of fp32, so widening on the way out
 * and narrowing on the way back is EXACT and the blob format needs no change.
 * ============================================================ */

/* Round to bf16 the way the card's cache does. Truncation rather than round-to-nearest — the
 * direction does not matter here, only that a value which has BEEN through it survives a
 * second pass unchanged, which is the property the round trip rests on. */
static float to_bf16(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits &= 0xFFFF0000u;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

typedef struct {
    int n_layers, n_kv_head, capacity, head_dim;
    float *rows;    /* [layer][kv][head][position][dim], bf16-rounded */
    int fail_after; /* transfers to allow before failing; -1 = never fail */
    int moves;
} fake_device;

static float *fake_slot(fake_device *d, int layer, int values)
{
    size_t per = (size_t)d->n_kv_head * d->capacity * d->head_dim;
    return d->rows + ((size_t)layer * 2 + values) * per;
}

static int fake_move(void *device, int layer, int values, int position_count, float *rows,
                     int capture)
{
    fake_device *d = (fake_device *)device;
    float *slot = fake_slot(d, layer, values);

    if (d->fail_after >= 0 && d->moves >= d->fail_after)
        return -1;
    d->moves++;
    for (int h = 0; h < d->n_kv_head; h++) {
        for (int p = 0; p < position_count; p++) {
            float *host = rows + ((size_t)h * position_count + p) * d->head_dim;
            float *dev = slot + ((size_t)h * d->capacity + p) * d->head_dim;

            for (int e = 0; e < d->head_dim; e++) {
                if (capture)
                    host[e] = dev[e];
                else
                    dev[e] = to_bf16(host[e]);
            }
        }
    }
    return 0;
}

static void test_device_resident_cache(void)
{
    const int L = 4, H = 3, S = 16, D = 8, POS = 11;
    printf("--- device-resident cache via kv_move (L=%d H=%d S=%d D=%d pos=%d) ---\n",
           L, H, S, D, POS);

    pn_llama_kv_cache *src = make_cache(L, H, S, D);
    pn_llama_kv_cache *dst = make_cache(L, H, S, D);
    fake_device from = { L, H, S, D, NULL, -1, 0 };
    fake_device into = { L, H, S, D, NULL, -1, 0 };
    size_t cells = (size_t)L * 2 * H * S * D;
    int wrong = 0, host_touched = 0;
    void *snap;

    from.rows = calloc(cells, sizeof(float));
    into.rows = calloc(cells, sizeof(float));
    CHECK(src && dst && from.rows && into.rows, "caches and fake devices allocated");
    if (!src || !dst || !from.rows || !into.rows)
        return;

    /* The DEVICE holds the real state, bf16-rounded; the host tensors hold poison. If the
     * snapshot ever reads the tensors instead of the device, every value it carries is POISON
     * and the comparison below cannot pass by accident. */
    for (int l = 0; l < L; l++) {
        for (int kv = 0; kv < 2; kv++) {
            float *slot = fake_slot(&from, l, kv);

            for (int h = 0; h < H; h++)
                for (int r = 0; r < S; r++)
                    for (int c = 0; c < D; c++)
                        slot[((size_t)h * S + r) * D + c] =
                            to_bf16(cell_value(kv, l, h, r, c, H, S, D));
        }
    }
    poison_cache(src, H, S, D);
    poison_cache(dst, H, S, D);
    src->kv_device = &from;
    src->kv_move = fake_move;
    dst->kv_device = &into;
    dst->kv_move = fake_move;

    snap = pn_llama_kv_snapshot(src, POS);
    CHECK(snap != NULL, "capture from a device-resident cache");
    CHECK(from.moves == L * 2, "capture asked the DEVICE for all %d (layer,kv) panels rather "
          "than reading the host tensors (%d moves)", L * 2, from.moves);
    CHECK(pn_llama_kv_restore(dst, snap) == 0, "restore into a device-resident cache");

    for (int l = 0; l < L; l++) {
        for (int kv = 0; kv < 2; kv++) {
            const float *a = fake_slot(&from, l, kv);
            const float *b = fake_slot(&into, l, kv);
            const float *tensor =
                (const float *)(kv ? dst->v_cache[l]->data : dst->k_cache[l]->data);

            for (int h = 0; h < H; h++)
                for (int r = 0; r < POS; r++)
                    for (int c = 0; c < D; c++) {
                        size_t i = ((size_t)h * S + r) * D + c;

                        if (a[i] != b[i])
                            wrong++;
                    }
            for (size_t i = 0; i < (size_t)H * S * D; i++)
                if (tensor[i] != POISON)
                    host_touched++;
        }
    }
    CHECK(wrong == 0, "device -> blob -> device is byte-identical over [0,%d) (%d wrong)",
          POS, wrong);
    CHECK(host_touched == 0, "the HOST tensors were never written on the device arm "
          "(%d cells touched)", host_touched);

    /* A FAILED transfer must not be absorbed: a snapshot that keeps whatever the buffer held
     * is the zero-page export wearing a different hat. */
    {
        fake_device broken = { L, H, S, D, from.rows, 2, 0 };
        pn_llama_kv_cache *flaky = make_cache(L, H, S, D);

        CHECK(flaky != NULL, "flaky cache allocated");
        if (flaky) {
            flaky->kv_device = &broken;
            flaky->kv_move = fake_move;
            CHECK(pn_llama_kv_snapshot(flaky, POS) == NULL,
                  "capture refuses when a device transfer fails");
            broken.moves = 0;
            CHECK(pn_llama_kv_restore(flaky, snap) != 0,
                  "restore refuses when a device transfer fails");
            free_cache(flaky);
        }
    }

    pn_llama_kv_snapshot_free(snap);
    free(from.rows);
    free(into.rows);
    free_cache(src);
    free_cache(dst);
}

int main(void)
{
    printf("=== llama session snapshot ===\n");
    test_capture_restore();
    test_serialize();
    test_dispatcher();
    test_device_resident_cache();
    printf("=== %d checks, %d failures ===\n", checks, failures);
    return failures ? 1 : 0;
}
