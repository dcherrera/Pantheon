/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_prefix_cache_sys.c
 * @brief SYS1 unit test — system-prefix caching in pn_prefix_cache plus the
 *        underlying qwen35moe state snapshot/restore, WITHOUT a model: the
 *        test provides its own pn_state_new/reset/free stubs over hand-built
 *        pn_qwen35moe_state structs (small dims), so it links only
 *        prefix_cache.c + qwen35moe_state_snapshot.c + batch_sched.c and runs
 *        on the Mac.
 *
 *        Covers: snapshot/restore round-trip (fp32 + q4 incl. sink/window
 *        shadows), dim-mismatch rejection, the acquire_sys decision ladder
 *        (full-history append resume preferred; sys-prefix resume on divergent
 *        history; changed system prompt -> full reset + re-capture), the
 *        capture protocol (boundary pos guard), LRU eviction of snapshots, and
 *        the SYS1-BATCHED per-slot twin (bs_prefill_resume_sys /
 *        bs_prefill_capture_sys over a hand-assembled scheduler): the batched
 *        resume ladder, per-slot capture/restore, cross-slot isolation, slot
 *        takeover by another session, and the snapshot-affinity acquire pass.
 *        (Batched DECODE itself needs CUDA — that is the K80 hardware gate.)
 *
 *        Also covers the projection rung in BOTH client conventions: reasoning
 *        elided (test_think_elided_projection) and reasoning RESENT
 *        (test_reasoning_bearing_resume), including the fallback-to-full-prefill
 *        that a divergent visible prefix must produce — and what it costs to
 *        CHANGE convention mid-session (test_reasoning_convention_switch):
 *        forward switches are free, a summarised block costs the turn it is
 *        introduced on, and retroactively dropping a block already sent costs
 *        every turn.
 *
 *        And two conversations under ONE session id (opencode's title request
 *        + its real turns): each resumes from its own slot's cache
 *        (test_two_convos_*, bs_slot_reroute), and the single-user cache's
 *        twin: an entry per conversation, admission still counting ids,
 *        export/import picking the right conversation (test_single_user_*).
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "../src/core/prefix_cache.h"
#include "../src/core/session_snapshot.h"

/* Phase 10b stub: the content-generation counter lives in
 * qwen35moe_forward.c, which this test intentionally does not link (it
 * stubs the pn_state lifecycle). snapshot-restore bumps it. */
unsigned pn_qwen35moe_kv_next_content_gen(void)
{
    static unsigned counter;
    return ++counter;
}
#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_internal.h"
#include "../src/models/qwen35moe_kv_q4.h"
#include "../src/serve/batch_sched.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* ============================================================
 * Hand-built dual state (small dims) + pn_state lifecycle stubs
 * ============================================================ */

enum {
    T_N_SSM = 2,
    T_N_ATTN = 1,
    T_CONV_M1 = 3,
    T_CONV_DIM = 8,
    T_STATE = 4,
    T_VHEADS = 2,
    T_HEADS_KV = 2,
    T_HEAD_DIM = 32,   /* multiple of 32 so q4 rows pack cleanly */
    T_MAX_SEQ = 64,
};

static int g_make_q4 = 0;   /* KV mode for the next make_state() */

static pn_qwen35moe_state *make_state(void)
{
    pn_qwen35moe_state *s = calloc(1, sizeof(*s));
    s->n_ssm = T_N_SSM;
    s->n_attn = T_N_ATTN;
    s->ssm = calloc(T_N_SSM, sizeof(*s->ssm));
    s->kv = calloc(T_N_ATTN, sizeof(*s->kv));
    for (int i = 0; i < T_N_SSM; i++) {
        pn_qwen35moe_ssm_state *st = &s->ssm[i];
        st->conv_dim = T_CONV_DIM;
        st->conv_minus_1 = T_CONV_M1;
        st->state_size = T_STATE;
        st->n_v_heads = T_VHEADS;
        st->conv_state = calloc((size_t)T_CONV_M1 * T_CONV_DIM, sizeof(float));
        st->ssm_state = calloc((size_t)T_STATE * T_STATE * T_VHEADS,
                               sizeof(float));
    }
    for (int i = 0; i < T_N_ATTN; i++) {
        pn_qwen35moe_kv *kv = &s->kv[i];
        kv->n_head_kv = T_HEADS_KV;
        kv->head_dim = T_HEAD_DIM;
        kv->max_seq = T_MAX_SEQ;
        kv->q4 = g_make_q4;
        if (kv->q4) {
            size_t bytes = (size_t)T_HEADS_KV * T_MAX_SEQ *
                           (size_t)pn_kvq4_row_bytes(T_HEAD_DIM);
            kv->kq = calloc(bytes, 1);
            kv->vq = calloc(bytes, 1);
            size_t sink_n = (size_t)T_HEADS_KV * PN_KVQ4_SINKMAX * T_HEAD_DIM;
            size_t win_n = (size_t)T_HEADS_KV * PN_KVQ4_WMAX * T_HEAD_DIM;
            kv->k_sink = calloc(sink_n, sizeof(uint16_t));
            kv->v_sink = calloc(sink_n, sizeof(uint16_t));
            kv->k_win = calloc(win_n, sizeof(uint16_t));
            kv->v_win = calloc(win_n, sizeof(uint16_t));
        } else {
            size_t n = (size_t)T_HEADS_KV * T_MAX_SEQ * T_HEAD_DIM;
            kv->k = calloc(n, sizeof(float));
            kv->v = calloc(n, sizeof(float));
        }
    }
    return s;
}

static void free_state(pn_qwen35moe_state *s)
{
    if (!s) return;
    for (int i = 0; i < s->n_ssm; i++) {
        free(s->ssm[i].conv_state);
        free(s->ssm[i].ssm_state);
    }
    for (int i = 0; i < s->n_attn; i++) {
        free(s->kv[i].k); free(s->kv[i].v);
        free(s->kv[i].kq); free(s->kv[i].vq);
        free(s->kv[i].k_sink); free(s->kv[i].v_sink);
        free(s->kv[i].k_win); free(s->kv[i].v_win);
    }
    free(s->ssm);
    free(s->kv);
    free(s);
}

/* ---- pn_state lifecycle stubs (replace engine.c for this test) ---- */

/* Phase 11: bs_think_elided_resume refuses itself above the session context cap.
 * engine.c is not linked here, so mirror its shape -- the model's own trained
 * context when set, otherwise unbounded -- which leaves the rung ENABLED for
 * these hand-built states and so keeps the think-elision paths under test. */
int pn_session_ctx_cap(const pn_model *model)
{
    if (model && model->config.n_ctx > 0)
        return model->config.n_ctx;
    return INT_MAX;
}

/* The PN_REQPROF divergence dumps decode token ids to text so a miss can be read
 * straight out of the serve log. Neither the tokenizer nor engine.c is linked
 * here, and the dumps are diagnostic-only (unreachable with PN_REQPROF unset),
 * so stub them rather than pull a tokenizer into a host-only unit test. */
char *ft_decode(ft_tokenizer *tok, int *ids, int length)
{
    (void)tok; (void)ids; (void)length;
    return NULL;
}

int ft_get_model_type(ft_tokenizer *tok)
{
    (void)tok;
    return -1;
}

char *gpt2_bytes_decode(const char *input)
{
    (void)input;
    return NULL;
}

pn_state *pn_state_new(pn_model *model)
{
    pn_state *st = calloc(1, sizeof(*st));
    st->config = &model->config;
    st->kv_cache = make_state();
    return st;
}

void pn_state_free(pn_state *state)
{
    if (!state) return;
    free_state((pn_qwen35moe_state *)state->kv_cache);
    free(state);
}

void pn_state_reset(pn_state *state)
{
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    state->pos = 0;
    for (int i = 0; i < s->n_ssm; i++) {
        memset(s->ssm[i].conv_state, 0,
               (size_t)T_CONV_M1 * T_CONV_DIM * sizeof(float));
        memset(s->ssm[i].ssm_state, 0,
               (size_t)T_STATE * T_STATE * T_VHEADS * sizeof(float));
    }
    for (int i = 0; i < s->n_attn; i++) {
        pn_qwen35moe_kv *kv = &s->kv[i];
        if (kv->q4) {
            size_t bytes = (size_t)T_HEADS_KV * T_MAX_SEQ *
                           (size_t)pn_kvq4_row_bytes(T_HEAD_DIM);
            memset(kv->kq, 0, bytes);
            memset(kv->vq, 0, bytes);
        } else {
            size_t n = (size_t)T_HEADS_KV * T_MAX_SEQ * T_HEAD_DIM;
            memset(kv->k, 0, n * sizeof(float));
            memset(kv->v, 0, n * sizeof(float));
        }
        kv->kv_mirror_stale = 0;
    }
}

/* ============================================================
 * Pattern fill / verify helpers ("simulated prefill")
 * ============================================================ */

static void fill_ssm(pn_qwen35moe_state *s, float seed)
{
    for (int i = 0; i < s->n_ssm; i++) {
        for (int j = 0; j < T_CONV_M1 * T_CONV_DIM; j++)
            s->ssm[i].conv_state[j] = seed + (float)(i * 100 + j);
        for (int j = 0; j < T_STATE * T_STATE * T_VHEADS; j++)
            s->ssm[i].ssm_state[j] = seed * 2.0f + (float)(i * 100 + j);
    }
}

static int check_ssm(const pn_qwen35moe_state *s, float seed)
{
    for (int i = 0; i < s->n_ssm; i++) {
        for (int j = 0; j < T_CONV_M1 * T_CONV_DIM; j++)
            if (s->ssm[i].conv_state[j] != seed + (float)(i * 100 + j))
                return 0;
        for (int j = 0; j < T_STATE * T_STATE * T_VHEADS; j++)
            if (s->ssm[i].ssm_state[j] != seed * 2.0f + (float)(i * 100 + j))
                return 0;
    }
    return 1;
}

/* Fill fp32 KV rows [from, to) with a per-(head,row,dim) pattern. */
static void fill_kv_rows(pn_qwen35moe_state *s, int from, int to, float seed)
{
    for (int i = 0; i < s->n_attn; i++) {
        pn_qwen35moe_kv *kv = &s->kv[i];
        for (int h = 0; h < kv->n_head_kv; h++)
            for (int r = from; r < to; r++)
                for (int d = 0; d < kv->head_dim; d++) {
                    size_t at = ((size_t)h * kv->max_seq + r) * kv->head_dim + d;
                    kv->k[at] = seed + (float)(h * 10000 + r * 100 + d);
                    kv->v[at] = -seed - (float)(h * 10000 + r * 100 + d);
                }
    }
}

static int check_kv_rows(const pn_qwen35moe_state *s, int from, int to,
                         float seed)
{
    for (int i = 0; i < s->n_attn; i++) {
        const pn_qwen35moe_kv *kv = &s->kv[i];
        for (int h = 0; h < kv->n_head_kv; h++)
            for (int r = from; r < to; r++)
                for (int d = 0; d < kv->head_dim; d++) {
                    size_t at = ((size_t)h * kv->max_seq + r) * kv->head_dim + d;
                    if (kv->k[at] != seed + (float)(h * 10000 + r * 100 + d))
                        return 0;
                    if (kv->v[at] != -seed - (float)(h * 10000 + r * 100 + d))
                        return 0;
                }
    }
    return 1;
}

/* Byte-pattern fill for q4 buffers + shadows. */
static void fill_q4(pn_qwen35moe_state *s, unsigned seed)
{
    pn_qwen35moe_kv *kv = &s->kv[0];
    size_t bytes = (size_t)kv->n_head_kv * kv->max_seq *
                   (size_t)pn_kvq4_row_bytes(kv->head_dim);
    for (size_t j = 0; j < bytes; j++) {
        kv->kq[j] = (unsigned char)((seed + j) & 0xFF);
        kv->vq[j] = (unsigned char)((seed + j + 7) & 0xFF);
    }
    size_t sink_n = (size_t)kv->n_head_kv * PN_KVQ4_SINKMAX * kv->head_dim;
    size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
    for (size_t j = 0; j < sink_n; j++) {
        kv->k_sink[j] = (uint16_t)(seed + j);
        kv->v_sink[j] = (uint16_t)(seed + j + 3);
    }
    for (size_t j = 0; j < win_n; j++) {
        kv->k_win[j] = (uint16_t)(seed + j + 11);
        kv->v_win[j] = (uint16_t)(seed + j + 13);
    }
}

/* ============================================================
 * Test 1 — snapshot/restore round-trip (fp32)
 * ============================================================ */

static void test_snapshot_roundtrip_fp32(void)
{
    printf("test: snapshot/restore round-trip (fp32)\n");
    g_make_q4 = 0;
    pn_qwen35moe_state *s = make_state();
    fill_ssm(s, 1.0f);
    fill_kv_rows(s, 0, 8, 1.0f);
    s->kv[0].kv_mirror_stale = 1;

    void *snap = pn_qwen35moe_state_snapshot(s, 8);
    CHECK(snap != NULL, "snapshot at pos 8 succeeds");
    CHECK(pn_qwen35moe_state_snapshot_pos(snap) == 8, "snapshot pos recorded");

    /* Scribble everything (a later turn), then restore the boundary. */
    fill_ssm(s, 9.0f);
    fill_kv_rows(s, 0, 24, 9.0f);
    CHECK(pn_qwen35moe_state_restore(s, snap) == 0, "restore succeeds");
    CHECK(check_ssm(s, 1.0f), "SSM state byte-restored");
    CHECK(check_kv_rows(s, 0, 8, 1.0f), "KV rows [0,8) byte-restored");
    CHECK(check_kv_rows(s, 8, 24, 9.0f), "KV rows past the boundary untouched");
    CHECK(s->kv[0].kv_mirror_stale == 0, "restore marks host mirror current");

    CHECK(pn_qwen35moe_state_snapshot(s, 0) == NULL, "pos 0 snapshot rejected");
    CHECK(pn_qwen35moe_state_snapshot(s, T_MAX_SEQ + 1) == NULL,
          "pos > max_seq snapshot rejected");
    pn_qwen35moe_state_snapshot_free(snap);
    free_state(s);
}

/* ============================================================
 * Test 2 — snapshot/restore round-trip (q4 + fp16 shadows)
 * ============================================================ */

static void test_snapshot_roundtrip_q4(void)
{
    printf("test: snapshot/restore round-trip (q4 + shadows)\n");
    g_make_q4 = 1;
    pn_qwen35moe_state *s = make_state();
    fill_ssm(s, 3.0f);
    fill_q4(s, 100);

    void *snap = pn_qwen35moe_state_snapshot(s, 8);
    CHECK(snap != NULL, "q4 snapshot succeeds");

    /* Keep a reference copy of the captured region, scribble, restore. */
    pn_qwen35moe_kv *kv = &s->kv[0];
    size_t row_b = (size_t)pn_kvq4_row_bytes(kv->head_dim);
    size_t bytes = (size_t)kv->n_head_kv * kv->max_seq * row_b;
    unsigned char *kq_ref = malloc(bytes);
    memcpy(kq_ref, kv->kq, bytes);
    size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
    uint16_t *win_ref = malloc(win_n * sizeof(uint16_t));
    memcpy(win_ref, kv->k_win, win_n * sizeof(uint16_t));

    fill_ssm(s, 8.0f);
    fill_q4(s, 200);
    CHECK(pn_qwen35moe_state_restore(s, snap) == 0, "q4 restore succeeds");
    CHECK(check_ssm(s, 3.0f), "q4: SSM state byte-restored");
    int rows_ok = 1;
    for (int h = 0; h < kv->n_head_kv && rows_ok; h++)
        if (memcmp(kv->kq + (size_t)h * kv->max_seq * row_b,
                   kq_ref + (size_t)h * kv->max_seq * row_b, 8 * row_b) != 0)
            rows_ok = 0;
    CHECK(rows_ok, "q4: KV rows [0,8) byte-restored");
    CHECK(memcmp(kv->k_win, win_ref, win_n * sizeof(uint16_t)) == 0,
          "q4: fp16 window ring byte-restored");

    /* Mode mismatch: a q4 snapshot cannot restore into an fp32 state. */
    g_make_q4 = 0;
    pn_qwen35moe_state *s32 = make_state();
    CHECK(pn_qwen35moe_state_restore(s32, snap) == -1,
          "q4 snapshot rejected by fp32 state (dim guard)");
    free_state(s32);

    free(kq_ref);
    free(win_ref);
    pn_qwen35moe_state_snapshot_free(snap);
    free_state(s);
}

/* ============================================================
 * Test 2b — serialize/deserialize blob round-trip (Phase 7)
 * The snap→bytes→snap round-trip MUST be byte-identical (it is pure
 * packing/unpacking). Verified two ways: (1) re-serializing the deserialized
 * snap yields byte-identical blob bytes; (2) restoring the deserialized snap
 * into a fresh state reproduces the original state buffers exactly.
 * ============================================================ */

static void test_snap_serialize_roundtrip(int q4)
{
    printf("test: snap serialize/deserialize round-trip (%s)\n",
           q4 ? "q4" : "fp32");
    g_make_q4 = q4;
    const uint64_t WV = 0xABCDEF0123456789ull;

    pn_qwen35moe_state *s = make_state();
    fill_ssm(s, 4.0f);
    if (q4) fill_q4(s, 300); else fill_kv_rows(s, 0, 16, 4.0f);

    void *snap = pn_qwen35moe_state_snapshot(s, 16);
    CHECK(snap != NULL, "snapshot succeeds");

    /* Serialize. */
    unsigned char *b1 = NULL; size_t l1 = 0;
    CHECK(pn_qwen35moe_snap_serialize(snap, WV, &b1, &l1) == 0 && b1 && l1 > 0,
          "serialize succeeds");

    /* Deserialize → re-serialize → byte-identical blob. */
    uint64_t wv_out = 0;
    void *snap2 = pn_qwen35moe_snap_deserialize(b1, l1, &wv_out);
    CHECK(snap2 != NULL, "deserialize succeeds");
    CHECK(wv_out == WV, "weights_version round-trips through the header");
    CHECK(pn_qwen35moe_state_snapshot_pos(snap2) == 16, "pos round-trips");

    unsigned char *b2 = NULL; size_t l2 = 0;
    CHECK(pn_qwen35moe_snap_serialize(snap2, WV, &b2, &l2) == 0, "re-serialize");
    CHECK(l1 == l2 && b1 && b2 && memcmp(b1, b2, l1) == 0,
          "snap->bytes->snap->bytes is BYTE-IDENTICAL");

    /* Restore the deserialized snap into a fresh state → equals the original. */
    pn_qwen35moe_state *s2 = make_state();
    CHECK(pn_qwen35moe_state_restore(s2, snap2) == 0,
          "restore deserialized snap into fresh state");
    CHECK(check_ssm(s2, 4.0f), "restored SSM equals original");
    if (!q4) {
        CHECK(check_kv_rows(s2, 0, 16, 4.0f), "restored fp32 KV equals original");
    } else {
        pn_qwen35moe_kv *a = &s->kv[0], *r = &s2->kv[0];
        size_t rb = (size_t)pn_kvq4_row_bytes(a->head_dim);
        int ok = 1;
        for (int h = 0; h < a->n_head_kv && ok; h++)
            if (memcmp(a->kq + (size_t)h * a->max_seq * rb,
                       r->kq + (size_t)h * r->max_seq * rb, 16 * rb) != 0)
                ok = 0;
        size_t win_n = (size_t)a->n_head_kv * PN_KVQ4_WMAX * a->head_dim;
        CHECK(ok && memcmp(a->k_win, r->k_win, win_n * sizeof(uint16_t)) == 0,
              "restored q4 rows + window shadow equal original");
    }

    /* Tamper / robustness: bad magic, truncation, and a NULL blob all reject. */
    unsigned char save0 = b1[0];
    b1[0] ^= 0xFF;
    CHECK(pn_qwen35moe_snap_deserialize(b1, l1, NULL) == NULL,
          "bad magic rejected");
    b1[0] = save0;
    CHECK(pn_qwen35moe_snap_deserialize(b1, l1 - 1, NULL) == NULL,
          "truncated blob rejected");
    CHECK(pn_qwen35moe_snap_deserialize(NULL, 0, NULL) == NULL,
          "NULL blob rejected");

    /* Hostile row_b: a blob keeping the dims snap_matches_state validates but
     * inflating the KV layer's row_b would drive an OOB write on restore. The
     * deserializer must reject any row_b inconsistent with (q4, head_dim).
     * Blob layout for this fixed test model: header (28 B: magic4 + fmt4 + wv8 +
     * pos4 + nssm4 + nattn4) + T_N_SSM SSM layers, each = 16 B (2×u64 counts) +
     * conv_n*4 + ssm_n*4, then the attention layer's 4×i32 (16 B), then the u64
     * row_b. conv_n = T_CONV_M1*T_CONV_DIM, ssm_n = T_STATE*T_STATE*T_VHEADS
     * (identical in both KV modes — SSM is fp32). */
    {
        size_t conv_n = (size_t)T_CONV_M1 * T_CONV_DIM;
        size_t ssm_n = (size_t)T_STATE * T_STATE * T_VHEADS;
        size_t per_ssm = 16 + conv_n * sizeof(float) + ssm_n * sizeof(float);
        size_t rb_off = 28 + (size_t)T_N_SSM * per_ssm + 16;
        unsigned char save_rb[8];
        memcpy(save_rb, b1 + rb_off, 8);
        b1[rb_off] = (unsigned char)(save_rb[0] + 4);   /* inflate row_b by 4 */
        CHECK(pn_qwen35moe_snap_deserialize(b1, l1, NULL) == NULL,
              "hostile row_b (inconsistent with head_dim) rejected");
        memcpy(b1 + rb_off, save_rb, 8);
        CHECK(pn_qwen35moe_snap_deserialize(b1, l1, NULL) != NULL,
              "restored row_b deserializes again (tamper was reversible)");
    }

    /* Hostile pos: pos strides the live KV buffer on restore (pos*row_b bytes
     * per head into a max_seq*row_b span). pos is an attacker-controlled header
     * field the deserializer only checks pos>0, so a blob with pos>max_seq — all
     * other dims matching the live model — drives an OOB write. Reproduce it by
     * shrinking the blob's per-layer max_seq below its pos and restoring into a
     * matching (smaller-max_seq) live state: snap_matches_state must reject on
     * the pos>max_seq guard even though every OTHER field agrees. The attn
     * layer's max_seq i32 sits after the SSM block + q4/n_head_kv/head_dim (28 B
     * header + T_N_SSM SSM layers + 3×i32). */
    {
        size_t conv_n = (size_t)T_CONV_M1 * T_CONV_DIM;
        size_t ssm_n = (size_t)T_STATE * T_STATE * T_VHEADS;
        size_t per_ssm = 16 + conv_n * sizeof(float) + ssm_n * sizeof(float);
        size_t ms_off = 28 + (size_t)T_N_SSM * per_ssm + 12;
        unsigned char save_ms[4];
        memcpy(save_ms, b1 + ms_off, 4);
        int32_t small_ms = 8;   /* < blob pos (16) — makes pos > max_seq */
        memcpy(b1 + ms_off, &small_ms, 4);
        void *hs = pn_qwen35moe_snap_deserialize(b1, l1, NULL);
        CHECK(hs != NULL, "pos-hostile blob still deserializes (pos strides ok)");
        pn_qwen35moe_state *sm = make_state();
        sm->kv[0].max_seq = small_ms;   /* match the tampered max_seq field */
        CHECK(pn_qwen35moe_state_restore(sm, hs) != 0,
              "restore rejects pos > live max_seq (OOB-write hardening)");
        pn_qwen35moe_state_snapshot_free(hs);
        free_state(sm);
        memcpy(b1 + ms_off, save_ms, 4);
    }

    free(b1); free(b2);
    pn_qwen35moe_state_snapshot_free(snap);
    pn_qwen35moe_state_snapshot_free(snap2);
    free_state(s);
    free_state(s2);
}

/* ============================================================
 * Test 3 — the acquire_sys decision ladder + capture protocol
 * ============================================================ */

static const uint32_t SYS_A[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
static const uint32_t SYS_B[8] = { 1, 2, 3, 4, 5, 6, 7, 99 };

/* Compose sys prefix + a filler conversation [base, base+n). */
static void mk_prompt(uint32_t *out, const uint32_t *sys, int conv_base,
                      int n_total)
{
    memcpy(out, sys, 8 * sizeof(uint32_t));
    for (int i = 8; i < n_total; i++)
        out[i] = (uint32_t)(conv_base + i);
}

static void test_acquire_sys_flow(void)
{
    printf("test: acquire_sys decision ladder\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);
    CHECK(c != NULL, "cache created");

    /* Turn 1: fresh session -> full prefill + capture request. */
    uint32_t p1[20];
    mk_prompt(p1, SYS_A, 1000, 20);
    int delta = -1, cap = -1;
    pn_state *st = pn_prefix_cache_acquire_sys(c, "s1", p1, 20, 8, &delta, &cap);
    CHECK(st != NULL, "turn 1 acquire returns a state");
    CHECK(delta == 0 && cap == 1, "turn 1: full prefill + capture requested");

    /* Simulate prefilling the system prefix, then capture the boundary. */
    pn_qwen35moe_state *qs = (pn_qwen35moe_state *)st->kv_cache;
    fill_ssm(qs, 1.0f);
    fill_kv_rows(qs, 0, 8, 1.0f);
    st->pos = 8;
    CHECK(pn_prefix_cache_capture_sys(c, "s1", p1, 8) == 0,
          "boundary capture succeeds at pos == sys_len");
    CHECK(pn_prefix_cache_capture_sys(c, "s1", p1, 5) == -1,
          "capture at wrong pos rejected");

    /* Simulate the rest of the turn (conversation + 4 generated tokens). */
    fill_ssm(qs, 5.0f);
    fill_kv_rows(qs, 8, 24, 5.0f);
    st->pos = 24;
    uint32_t hist1[24];
    memcpy(hist1, p1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) hist1[i] = (uint32_t)(9000 + i);
    CHECK(pn_prefix_cache_commit(c, "s1", hist1, 24) == 0, "turn 1 commit");

    /* Turn 2: same system prefix, DIVERGENT conversation (the reasoning-model
     * case) -> sys-prefix resume at 8, boundary state restored. */
    uint32_t p2[30];
    mk_prompt(p2, SYS_A, 4000, 30);
    st = pn_prefix_cache_acquire_sys(c, "s1", p2, 30, 8, &delta, &cap);
    CHECK(st != NULL && delta == 8 && cap == 0,
          "turn 2: divergent history resumes at sys boundary");
    CHECK(st->pos == 8, "turn 2: state rewound to sys_len");
    qs = (pn_qwen35moe_state *)st->kv_cache;
    CHECK(check_ssm(qs, 1.0f) && check_kv_rows(qs, 0, 8, 1.0f),
          "turn 2: boundary state byte-restored");

    /* Finish turn 2 and commit; turn 3 is a pure APPEND of the committed
     * history -> the full-history resume must win over the sys resume. */
    fill_kv_rows(qs, 8, 34, 6.0f);
    st->pos = 34;
    uint32_t hist2[34];
    memcpy(hist2, p2, 30 * sizeof(uint32_t));
    for (int i = 30; i < 34; i++) hist2[i] = (uint32_t)(9100 + i);
    CHECK(pn_prefix_cache_commit(c, "s1", hist2, 34) == 0, "turn 2 commit");
    uint32_t p3[36];
    memcpy(p3, hist2, 34 * sizeof(uint32_t));
    p3[34] = 7001; p3[35] = 7002;
    st = pn_prefix_cache_acquire_sys(c, "s1", p3, 36, 8, &delta, &cap);
    CHECK(st != NULL && delta == 34 && cap == 0,
          "turn 3: append history -> full-history resume preferred");

    /* Turn 4: CHANGED system prompt -> no reuse, full reset + re-capture. */
    uint32_t p4[20];
    mk_prompt(p4, SYS_B, 2000, 20);
    st = pn_prefix_cache_acquire_sys(c, "s1", p4, 20, 8, &delta, &cap);
    CHECK(st != NULL && delta == 0 && cap == 1,
          "turn 4: changed system prompt -> full reset + capture");
    CHECK(st->pos == 0, "turn 4: state was reset");

    /* No system prefix (sys_len 0) -> legacy behavior via the old API. */
    st = pn_prefix_cache_acquire(c, "s2", p1, 20, &delta);
    CHECK(st != NULL && delta == 0, "no sys prefix: legacy full prefill");

    /* Eviction: cap == 2, a third session evicts the LRU ("s1") and its
     * snapshot; re-acquiring "s1" is a fresh session (capture again). */
    st = pn_prefix_cache_acquire_sys(c, "s3", p1, 20, 8, &delta, &cap);
    CHECK(st != NULL, "third session (evicts LRU)");
    st = pn_prefix_cache_acquire_sys(c, "s1", p1, 20, 8, &delta, &cap);
    CHECK(st != NULL && delta == 0 && cap == 1,
          "evicted session returns fresh (snapshot gone, capture again)");

    pn_prefix_cache_free(c);
}

/* ============================================================
 * Test 4 — unsupported architecture never engages SYS1
 * ============================================================ */

static void test_unsupported_arch(void)
{
    printf("test: unsupported arch falls back cleanly\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_LLAMA;   /* stubs still build a qwen state,
                                          * but SYS1 must refuse the arch */
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);
    uint32_t p1[20];
    mk_prompt(p1, SYS_A, 1000, 20);
    int delta = -1, cap = -1;
    pn_state *st = pn_prefix_cache_acquire_sys(c, "u1", p1, 20, 8, &delta, &cap);
    CHECK(st != NULL && delta == 0 && cap == 0,
          "unsupported arch: full prefill, no capture requested");
    st->pos = 8;
    CHECK(pn_prefix_cache_capture_sys(c, "u1", p1, 8) == -1,
          "unsupported arch: capture refused");
    pn_prefix_cache_free(c);
}

/* ============================================================
 * Test 4b — LCP rewind on plain-KV (llama-family) arches
 *
 * No snapshots: a divergent history resumes at the longest common
 * prefix by truncating pos (KV rows [0,l) are position-invariant for
 * pure causal attention). Covers: full prefill on the fresh session,
 * the rewind on divergence, exact-append still preferred, pos-clamp,
 * and full reset when nothing is shared.
 * ============================================================ */

static void test_lcp_rewind(void)
{
    printf("test: llama-family LCP rewind resume\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_LLAMA;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);
    CHECK(c != NULL, "cache created");

    /* Turn 1: fresh session -> full prefill (and no SYS1 capture). */
    uint32_t p1[20];
    mk_prompt(p1, SYS_A, 1000, 20);
    int delta = -1, cap = -1;
    pn_state *st = pn_prefix_cache_acquire_sys(c, "r1", p1, 20, 8, &delta, &cap);
    CHECK(st != NULL && delta == 0 && cap == 0,
          "turn 1: fresh session full prefill, no capture");
    st->pos = 24;   /* prompt + 4 generated */
    uint32_t hist1[24];
    memcpy(hist1, p1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) hist1[i] = (uint32_t)(9000 + i);
    CHECK(pn_prefix_cache_commit(c, "r1", hist1, 24) == 0, "turn 1 commit");

    /* Turn 2: DIVERGENT conversation, same 8-token system prefix ->
     * rewind to the LCP (8) instead of a full reset. */
    uint32_t p2[30];
    mk_prompt(p2, SYS_A, 4000, 30);
    st = pn_prefix_cache_acquire_sys(c, "r1", p2, 30, 8, &delta, &cap);
    CHECK(st != NULL && delta == 8 && cap == 0,
          "turn 2: divergent history rewinds to the LCP");
    CHECK(st->pos == 8, "turn 2: pos truncated to the LCP");

    /* Turn 3: pure append of the committed history -> exact append resume
     * still wins (no rewind, no truncation). */
    st->pos = 34;
    uint32_t hist2[34];
    memcpy(hist2, p2, 30 * sizeof(uint32_t));
    for (int i = 30; i < 34; i++) hist2[i] = (uint32_t)(9100 + i);
    CHECK(pn_prefix_cache_commit(c, "r1", hist2, 34) == 0, "turn 2 commit");
    uint32_t p3[36];
    memcpy(p3, hist2, 34 * sizeof(uint32_t));
    p3[34] = 7001; p3[35] = 7002;
    st = pn_prefix_cache_acquire_sys(c, "r1", p3, 36, 8, &delta, &cap);
    CHECK(st != NULL && delta == 34, "turn 3: exact append still preferred");

    /* Rewind never past the state's own position: the committed history says
     * 36 tokens but the state only sits at 10 (fell behind) -> the LCP (36)
     * clamps to pos, resuming at 10. */
    CHECK(pn_prefix_cache_commit(c, "r1", p3, 36) == 0, "turn 3 commit");
    st->pos = 10;   /* simulate a state that fell behind its history */
    st = pn_prefix_cache_acquire_sys(c, "r1", p3, 36, 8, &delta, &cap);
    CHECK(st != NULL && delta == 10 && st->pos == 10,
          "rewind clamped to the state's own pos");

    /* Nothing shared at all (different first token) -> full reset. */
    uint32_t p5[20];
    mk_prompt(p5, SYS_A, 1000, 20);
    p5[0] = 42;
    st = pn_prefix_cache_acquire_sys(c, "r1", p5, 20, 8, &delta, &cap);
    CHECK(st != NULL && delta == 0 && st->pos == 0,
          "no shared prefix: full reset");

    pn_prefix_cache_free(c);
}

/* ============================================================
 * Test 5 — SYS1-BATCHED: per-slot resume/capture ladder + isolation
 *
 * Drives bs_prefill_resume_sys / bs_prefill_capture_sys / the affinity
 * acquires over a hand-assembled pn_batch_sched (no bs_create — that is
 * CUDA-gated).  Only the pieces these paths touch are initialized: the
 * mutex + condvars, cap, model, and per-slot pn_states (the same stubs the
 * prefix-cache tests use).  The driver thread never runs (driver_running
 * false), matching the non-CUDA reality these host paths are tested in.
 * bs_destroy is used for teardown, so it also exercises the sys-snapshot
 * frees.  Every resume takes the GPU turn; bs_prefill_abort releases turn +
 * slot, standing in for the CUDA-only bind.
 * ============================================================ */

static pn_batch_sched *mk_sched(pn_model *model, int cap)
{
    pn_batch_sched *s = calloc(1, sizeof(*s));
    s->model = model;
    s->cap = cap;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv_gpu, NULL);
    pthread_cond_init(&s->cv_work, NULL);
    pthread_cond_init(&s->cv_idle, NULL);
    for (int i = 0; i < cap; i++) {
        bs_slot *sl = &s->slots[i];
        sl->slot_idx = i;
        sl->status = BS_SLOT_IDLE;
        sl->gpu_idx = -1;
        pthread_cond_init(&sl->cv, NULL);
        sl->state = pn_state_new(model);
    }
    return s;
}

/* Simulate finalize's warm-history publication (the real bs_slot_finalize
 * needs a live GPU decode slot): mark the slot's state as holding session
 * `sid`'s committed history `hist[0..n)` at pos == n. */
static void publish_warm(bs_slot *sl, const char *sid, const uint32_t *hist,
                         int n)
{
    snprintf(sl->sid, sizeof(sl->sid), "%s", sid);
    if (sl->hist_cap < n) {
        sl->hist = realloc(sl->hist, (size_t)n * sizeof(uint32_t));
        sl->hist_cap = n;
    }
    memcpy(sl->hist, hist, (size_t)n * sizeof(uint32_t));
    sl->hist_n = n;
    sl->state->pos = n;
}

static void test_batched_sys_flow(void)
{
    printf("test: SYS1-batched per-slot resume/capture ladder\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);

    uint32_t pa1[20], pb1[20];
    mk_prompt(pa1, SYS_A, 1000, 20);   /* session A, sys prefix SYS_A */
    mk_prompt(pb1, SYS_B, 3000, 20);   /* session B, sys prefix SYS_B */
    int delta = -1, cap = -1;

    /* ── A turn 1 (slot 0): cold -> full prefill + capture request. ── */
    int sa = bs_slot_acquire_biased(bs, "A");
    CHECK(sa == 0, "A turn 1 claims slot 0");
    bs_prefill_resume_sys(bs, sa, "A", pa1, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 1, "A turn 1: full prefill + capture requested");
    pn_qwen35moe_state *qa = (pn_qwen35moe_state *)bs->slots[sa].state->kv_cache;
    fill_ssm(qa, 1.0f);                     /* "prefill" the system prefix */
    fill_kv_rows(qa, 0, 8, 1.0f);
    bs->slots[sa].state->pos = 8;
    CHECK(bs_prefill_capture_sys(bs, sa, "A", pa1, 5) == -1,
          "capture at wrong pos rejected");
    CHECK(bs_prefill_capture_sys(bs, sa, "A", pa1, 8) == 0,
          "A boundary capture succeeds at pos == sys_len");
    fill_ssm(qa, 5.0f);                     /* rest of the turn */
    fill_kv_rows(qa, 8, 24, 5.0f);
    uint32_t ha[24];
    memcpy(ha, pa1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) ha[i] = (uint32_t)(9000 + i);
    publish_warm(&bs->slots[sa], "A", ha, 24);
    bs_prefill_abort(bs, sa);   /* release turn + slot (stands in for bind) */

    /* ── B turn 1 (slot 1): concurrent session, own snapshot.  Pretend A is
     * mid-decode on slot 0 (the concurrent case this test single-threads) so
     * B's acquire takes slot 1. ── */
    bs->slots[0].status = BS_SLOT_DECODE;
    int sb = bs_slot_acquire_biased(bs, "B");
    CHECK(sb == 1, "B turn 1 claims slot 1 (A's slot busy)");
    bs->slots[0].status = BS_SLOT_IDLE;
    bs_prefill_resume_sys(bs, sb, "B", pb1, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 1, "B turn 1: full prefill + capture requested");
    pn_qwen35moe_state *qb = (pn_qwen35moe_state *)bs->slots[sb].state->kv_cache;
    fill_ssm(qb, 2.0f);
    fill_kv_rows(qb, 0, 8, 2.0f);
    bs->slots[sb].state->pos = 8;
    CHECK(bs_prefill_capture_sys(bs, sb, "B", pb1, 8) == 0,
          "B boundary capture succeeds");
    fill_ssm(qb, 6.0f);
    fill_kv_rows(qb, 8, 24, 6.0f);
    uint32_t hb[24];
    memcpy(hb, pb1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) hb[i] = (uint32_t)(9500 + i);
    publish_warm(&bs->slots[sb], "B", hb, 24);
    bs_prefill_abort(bs, sb);

    /* ── A turn 2: APPEND of the committed history -> warm resume wins. ── */
    uint32_t pa2[26];
    memcpy(pa2, ha, 24 * sizeof(uint32_t));
    pa2[24] = 7001; pa2[25] = 7002;
    sa = bs_slot_acquire_biased(bs, "A");
    CHECK(sa == 0, "A routed back to its warm slot 0");
    bs_prefill_resume_sys(bs, sa, "A", pa2, 26, 8, &delta, &cap);
    CHECK(delta == 24 && cap == 0,
          "A turn 2: append history -> warm-history resume preferred");
    publish_warm(&bs->slots[sa], "A", ha, 24);   /* keep prior warm state */
    bs_prefill_abort(bs, sa);

    /* ── A turn 3: DIVERGENT history, same system prefix -> sys resume; and
     * the restore must not touch B's slot-1 state (isolation). ── */
    uint32_t pa3[30];
    mk_prompt(pa3, SYS_A, 4000, 30);
    sa = bs_slot_acquire_biased(bs, "A");
    CHECK(sa == 0, "A turn 3 routed to slot 0 again");
    bs_prefill_resume_sys(bs, sa, "A", pa3, 30, 8, &delta, &cap);
    CHECK(delta == 8 && cap == 0,
          "A turn 3: divergent history resumes at sys boundary");
    CHECK(bs->slots[sa].state->pos == 8, "A turn 3: state rewound to sys_len");
    CHECK(check_ssm(qa, 1.0f) && check_kv_rows(qa, 0, 8, 1.0f),
          "A turn 3: boundary state byte-restored");
    CHECK(bs->slots[sa].sid[0] == '\0' && bs->slots[sa].hist_n == 0,
          "A turn 3: stale committed history dropped");
    CHECK(check_ssm(qb, 6.0f) && check_kv_rows(qb, 0, 8, 2.0f) &&
          check_kv_rows(qb, 8, 24, 6.0f),
          "isolation: B's slot-1 state untouched by A's restore");
    bs_prefill_abort(bs, sa);

    /* ── Snapshot-affinity pass: A's warm sid is gone (dropped above) but its
     * snapshot remains -> acquire must still route A to slot 0.  Plain
     * release (no resume ran, so this slot never took the GPU turn — abort
     * would release an unowned turn and desync the turnstile). ── */
    sa = bs_slot_acquire_biased(bs, "A");
    CHECK(sa == 0, "snapshot affinity routes A back to slot 0 (sid empty)");
    bs_slot_release(bs, sa);

    /* ── B turn 2 (divergent): B's own snapshot on slot 1 still restores. ── */
    uint32_t pb2[30];
    mk_prompt(pb2, SYS_B, 5000, 30);
    sb = bs_slot_acquire_biased(bs, "B");
    CHECK(sb == 1, "B routed back to its slot 1");
    bs_prefill_resume_sys(bs, sb, "B", pb2, 30, 8, &delta, &cap);
    CHECK(delta == 8 && cap == 0 && check_ssm(qb, 2.0f) &&
          check_kv_rows(qb, 0, 8, 2.0f),
          "B turn 2: own boundary snapshot restored (not A's)");
    bs_prefill_abort(bs, sb);

    /* ── Changed system prompt for A -> no reuse, full reset + re-capture,
     * and the slot REPORTS the substitution (same sid, second conversation). ── */
    uint32_t pa4[20];
    mk_prompt(pa4, SYS_B, 6000, 20);   /* A now arrives with SYS_B */
    sa = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, sa, "A", pa4, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 1 && bs->slots[sa].state->pos == 0,
          "changed system prompt: full reset + capture requested");
    CHECK(bs_slot_session_conflict(bs, sa),
          "changed system prompt under a live sid is REPORTED as a conflict");
    bs_prefill_abort(bs, sa);

    /* ── A new session never takes a slot another session holds: both are
     * held (A's snapshot, B), so C gets none and nothing of A's or B's is
     * touched.  (This used to evict the least recently active one.) ── */
    CHECK(bs_slot_acquire_biased(bs, "C") == -1,
          "C gets no slot: every slot is held by another session");
    CHECK(strcmp(bs->slots[0].sys_sid, "A") == 0 &&
          strcmp(bs->slots[1].sys_sid, "B") == 0,
          "...and A's and B's snapshots stay where they were");

    /* ── Takeover within one family: "A-side" may take A's slot — append-miss
     * + sid-guarded sys-miss -> full prefill; its capture then replaces the
     * resident snapshot. ── */
    uint32_t pc1[20];
    mk_prompt(pc1, SYS_A, 7000, 20);   /* same SYS_A bytes as A's snapshot! */
    int sc = bs_slot_acquire_biased(bs, "A-side");
    CHECK(sc == 0, "takeover: A-side (A's family) takes A's slot");
    bs_prefill_resume_sys(bs, sc, "A-side", pc1, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 1,
          "takeover: A-side never reuses A's snapshot (sid guard) even on "
          "identical prefix bytes");
    pn_qwen35moe_state *qc = (pn_qwen35moe_state *)bs->slots[sc].state->kv_cache;
    fill_ssm(qc, 3.0f);
    fill_kv_rows(qc, 0, 8, 3.0f);
    bs->slots[sc].state->pos = 8;
    CHECK(bs_prefill_capture_sys(bs, sc, "A-side", pc1, 8) == 0,
          "takeover: A-side's capture replaces the slot's snapshot");
    CHECK(strcmp(bs->slots[sc].sys_sid, "A-side") == 0,
          "takeover: snapshot ownership moved to A-side");
    bs_prefill_abort(bs, sc);

    /* ── No session id / no sys prefix -> byte-identical legacy behavior, on
     * a slot nobody holds (none while A-side and B hold both). ── */
    CHECK(bs_slot_acquire(bs) == -1, "NULL sid: no slot while all are held");
    bs_session_release(bs, "B");
    int sn = bs_slot_acquire(bs);
    CHECK(sn == 1, "NULL sid takes the slot B's release freed");
    bs_prefill_resume_sys(bs, sn, NULL, pc1, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 0, "NULL sid: full prefill, no capture");
    bs_prefill_abort(bs, sn);
    sn = bs_slot_acquire(bs);
    bs_prefill_resume(bs, sn, "C", pc1, 20, &delta);
    CHECK(delta == 0, "bs_prefill_resume wrapper: sys disabled (sys_len 0)");
    bs_prefill_abort(bs, sn);

    bs_destroy(bs);   /* also frees the per-slot snapshots (leak gate) */
}

/* ============================================================
 * Test 5b — NO DUPLICATE SLOTS: a session whose own slot is busy never takes a
 * second slot that a new session needs.
 *
 * The incident (K80 bank, 2026-11-10): 4 slots, max_sessions 4.  An agent's
 * next request arrived while its own slot was busy and took the free slot;
 * once its warm state landed there it held two.  New sessions then got 429
 * while sessions_used read 3/4.  The same happened through bs_slot_reroute
 * (a new conversation under the id moves to a cold slot).
 *
 * Red: drop bs_family_may_take_extra_locked from try_claim_idle or
 * bs_reroute_target_locked and the matching checks fail.
 * ============================================================ */

static const uint32_t DUP_HIST[4] = { 11, 12, 13, 14 };

/* Seat `sid` on a slot, give it warm state there, end its turn.  The slot. */
static int dup_seat_warm(pn_batch_sched *bs, const char *sid)
{
    int s = -1;

    if (bs_slot_acquire_or_wait(bs, 0, &s, NULL, sid) != BS_ACQUIRE_OK || s < 0)
        return -1;
    publish_warm(&bs->slots[s], sid, DUP_HIST, 4);
    bs_slot_release(bs, s);
    return s;
}

static void test_no_duplicate_slots(void)
{
    printf("test: a busy session never takes a seat a new session needs\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    bs_occupancy occ;

    /* ── Busy turn, then another request of the same session. ── */
    pn_batch_sched *bs = mk_sched(&model, 4);
    bs_set_session_admission(bs, 4, 0);
    CHECK(dup_seat_warm(bs, "agent-a") == 0 && dup_seat_warm(bs, "agent-b") == 1 &&
          dup_seat_warm(bs, "agent-c") == 2, "three sessions warm on slots 0-2");
    int a1 = -1, a2 = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 0, &a1, NULL, "agent-a") == BS_ACQUIRE_OK &&
          a1 == 0, "agent-a's turn runs on its own slot 0");
    bs_acquire_result r = bs_slot_acquire_or_wait(bs, 0, &a2, NULL, "agent-a");
    CHECK(r != BS_ACQUIRE_OK && a2 < 0,
          "agent-a's next request, slot 0 busy: waits, does not take slot 3");
    if (a2 >= 0) {   /* the old rule: its warm state lands on slot 3 too */
        publish_warm(&bs->slots[a2], "agent-a", DUP_HIST, 4);
        bs_slot_release(bs, a2);
    }
    bs_slot_release(bs, a1);
    bs_session_occupancy(bs, &occ);
    CHECK(occ.families == 3 && occ.slots_held == 3 && occ.slots_free == 1 &&
          occ.seat_free, "3 sessions hold 3 slots; a seat is free");
    int d = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 0, &d, NULL, "agent-d") == BS_ACQUIRE_OK &&
          d == 3, "a 4th session is admitted on slot 3");
    if (d >= 0)
        bs_slot_release(bs, d);
    bs_destroy(bs);

    /* ── Reroute: a new conversation under a held id stays on its slot. ── */
    static const uint32_t other[6] = { 90, 91, 92, 93, 94, 95 };
    bs = mk_sched(&model, 4);
    bs_set_session_admission(bs, 4, 0);
    dup_seat_warm(bs, "agent-a");
    dup_seat_warm(bs, "agent-b");
    dup_seat_warm(bs, "agent-c");
    a1 = -1;
    bs_slot_acquire_or_wait(bs, 0, &a1, NULL, "agent-a");
    int to = bs_slot_reroute(bs, a1, "agent-a", other, 6, 0);
    CHECK(a1 == 0 && to == 0,
          "reroute of a new conversation keeps agent-a on slot 0 (slot 3 is "
          "the 4th session's seat)");
    publish_warm(&bs->slots[to], "agent-a", other, 5);
    bs_slot_release(bs, to);
    d = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 0, &d, NULL, "agent-d") == BS_ACQUIRE_OK &&
          d == 3, "...and the 4th session is admitted after it");
    if (d >= 0)
        bs_slot_release(bs, d);
    bs_destroy(bs);

    /* ── With room to spare (max_sessions 2 of 4 slots) a second copy is still
     * allowed: it costs no session a seat. ── */
    bs = mk_sched(&model, 4);
    bs_set_session_admission(bs, 2, 0);
    dup_seat_warm(bs, "agent-a");
    a1 = a2 = -1;
    bs_slot_acquire_or_wait(bs, 0, &a1, NULL, "agent-a");
    CHECK(bs_slot_acquire_or_wait(bs, 0, &a2, NULL, "agent-a") == BS_ACQUIRE_OK &&
          a2 > 0, "room to spare: agent-a's second request takes a free slot");
    to = bs_slot_reroute(bs, a1, "agent-a", other, 6, 0);
    CHECK(to != a1 && to != a2,
          "room to spare: reroute still moves a new conversation to a cold slot");
    bs_slot_release(bs, a2);
    bs_slot_release(bs, to);
    bs_destroy(bs);
}

/* ============================================================
 * Test 6b — THINK-ELISION: the projection is what the CLIENT will resend
 *
 * bs_slot_note_visible builds the projection; bs_slot_finalize publishes it
 * (GPU-only, so publish_vis stands in); bs_prefill_resume_sys consumes it.
 * The cases that matter are the two the K80 test machine found: a plain turn, and a TOOL
 * turn where the model emitted a BARE call but the client re-renders it inside
 * a "<tool_call>" wrapper it never generated.
 * ============================================================ */

/* bs_slot_finalize needs a GPU state, so the test drives its publish STEP
 * directly — the production function, not a copy of it: the two projections'
 * asymmetric length rule is exactly the kind of thing a stand-in gets wrong. */
static void publish_vis(bs_slot *sl)
{
    bs_slot_publish_projections(sl, sl->ct_n);
}

/* Seed the slot's committed-token log as the driver would: the prompt followed
 * by `gen` generated tokens. */
static void seed_committed_log(bs_slot *sl, const uint32_t *prompt, int n_prompt,
                               const uint32_t *gen, int n_gen)
{
    int n = n_prompt + n_gen;
    if (sl->ct_cap < n) {
        sl->ct = realloc(sl->ct, (size_t)n * sizeof(uint32_t));
        sl->ct_cap = n;
    }
    memcpy(sl->ct, prompt, (size_t)n_prompt * sizeof(uint32_t));
    memcpy(sl->ct + n_prompt, gen, (size_t)n_gen * sizeof(uint32_t));
    sl->ct_n = n;
}

/* ============================================================
 * Test 6c — SESSION SUBSTITUTION: one session id, two conversations
 *
 * The defect this guards: an agent's reflect pass POSTed a second conversation
 * (its own system prompt) under the main chat's session id.  Every rung missed,
 * the slot's boundary snapshot was replaced by the intruder's, and BOTH
 * conversations re-prefilled in full from then on — with nothing said.
 *
 * The rule under test: report iff the SYSTEM BLOCK changed under a session id
 * this slot already holds a record for.  The three legitimate ways a history
 * changes (append, think-elided re-render, client-side compaction of the body)
 * all keep that block byte-identical and must NOT be reported.
 * ============================================================ */
static void test_session_substitution(void)
{
    printf("test: session substitution (one sid, two conversations)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    int delta = -1, cap = -1;

    /* Conversation A, turn 1: cold, captures its boundary. */
    uint32_t a1[20];
    mk_prompt(a1, SYS_A, 1000, 20);
    int s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", a1, 20, 8, &delta, &cap);
    CHECK(!bs_slot_session_conflict(bs, s),
          "a brand-new session id is never a conflict");
    pn_qwen35moe_state *q = (pn_qwen35moe_state *)bs->slots[s].state->kv_cache;
    fill_ssm(q, 1.0f);
    fill_kv_rows(q, 0, 8, 1.0f);
    bs->slots[s].state->pos = 8;
    CHECK(bs_prefill_capture_sys(bs, s, "S", a1, 8) == 0, "A captures at 8");
    uint32_t ha[24];
    memcpy(ha, a1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) ha[i] = (uint32_t)(9000 + i);
    publish_warm(&bs->slots[s], "S", ha, 24);
    bs_prefill_abort(bs, s);

    /* A turn 2 — a plain APPEND. Same system block: not a conflict. */
    uint32_t a2[26];
    memcpy(a2, ha, 24 * sizeof(uint32_t));
    a2[24] = 7001; a2[25] = 7002;
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", a2, 26, 8, &delta, &cap);
    CHECK(delta == 24 && !bs_slot_session_conflict(bs, s),
          "an appended history is a continuation, not a conflict");
    publish_warm(&bs->slots[s], "S", ha, 24);
    bs_prefill_abort(bs, s);

    /* A turn 3 — the history BODY is rewritten (what a client-side compaction
     * does) but the system block is untouched.  This is the case a naive
     * reject-on-divergence would break, so it must stay silent. */
    uint32_t a3[30];
    mk_prompt(a3, SYS_A, 4000, 30);
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", a3, 30, 8, &delta, &cap);
    CHECK(delta == 8, "rewritten body resumes at the system boundary");
    CHECK(!bs_slot_session_conflict(bs, s),
          "a rewritten history body (compaction) is NOT a conflict");
    bs_prefill_abort(bs, s);

    /* THE DEFECT — a SECOND conversation arrives under the same session id,
     * carrying a different system prompt. */
    uint32_t b1[20];
    mk_prompt(b1, SYS_B, 5000, 20);
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", b1, 20, 8, &delta, &cap);
    CHECK(delta == 0 && cap == 1, "the substituted conversation full-prefills");
    CHECK(bs_slot_session_conflict(bs, s),
          "a different system prompt under a live sid IS a conflict");
    fill_ssm(q, 2.0f);
    fill_kv_rows(q, 0, 8, 2.0f);
    bs->slots[s].state->pos = 8;
    CHECK(bs_prefill_capture_sys(bs, s, "S", b1, 8) == 0,
          "and it replaces the displaced conversation's snapshot");
    bs_prefill_abort(bs, s);

    /* Conversation A comes back: flagged again, and this is the turn that
     * actually pays — a full re-prefill of the whole conversation. */
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", a3, 30, 8, &delta, &cap);
    CHECK(delta == 0, "the displaced conversation re-prefills in full");
    CHECK(bs_slot_session_conflict(bs, s),
          "the displaced conversation's return is flagged too");
    publish_warm(&bs->slots[s], "S", a3, 30);
    bs_prefill_abort(bs, s);

    /* The flag is PER TURN, not per session: A's next turn is an ordinary
     * append, so it must come back clean.  A latched flag would keep telling a
     * client to renumber a session that is already behaving, and — always true
     * after one incident — would stop carrying information. */
    uint32_t a4[32];
    memcpy(a4, a3, 30 * sizeof(uint32_t));
    a4[30] = 7101; a4[31] = 7102;
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", a4, 32, 8, &delta, &cap);
    CHECK(delta == 30, "the recovered conversation resumes warm");
    CHECK(!bs_slot_session_conflict(bs, s),
          "the conflict flag does not latch: the next clean turn is clean");
    bs_prefill_abort(bs, s);

    /* A request with NO system block carries no signal — never flagged. */
    s = bs_slot_acquire_biased(bs, "S");
    bs_prefill_resume_sys(bs, s, "S", b1, 20, 0, &delta, &cap);
    CHECK(!bs_slot_session_conflict(bs, s),
          "sys_len 0: no system block, so nothing is claimed");
    bs_prefill_abort(bs, s);

    /* A DIFFERENT session id on a slot warm for someone else is an ordinary
     * takeover, not a substitution. */
    int t = bs_slot_acquire_biased(bs, "OTHER");
    bs_prefill_resume_sys(bs, t, "OTHER", a1, 20, 8, &delta, &cap);
    CHECK(!bs_slot_session_conflict(bs, t),
          "a different session id claiming a slot is not a conflict");
    bs_prefill_abort(bs, t);

    /* No session id at all: the uncached path is untouched. */
    int u = bs_slot_acquire(bs);
    bs_prefill_resume_sys(bs, u, NULL, b1, 20, 8, &delta, &cap);
    CHECK(!bs_slot_session_conflict(bs, u), "NULL sid: never a conflict");
    bs_prefill_abort(bs, u);

    bs_destroy(bs);
}

/* ============================================================
 * Test 6d — TWO CONVERSATIONS, ONE SESSION ID, each keeps its own cache
 *
 * opencode sends its session id on every request: a title-generation call
 * (its own short system prompt) and the real turns (a long one).  The acquire
 * runs before tokenization and took the FIRST slot of the id — the title's —
 * so the real turn 2 landed on the wrong slot and re-prefilled everything
 * (measured on the K80 fleet: 14,081 tokens, 170 s, for a 274-token delta).
 * bs_slot_reroute moves the turn to the slot its prompt continues.
 * ============================================================ */

/* One turn as the serve runs it: acquire, reroute on the known prompt, resume
 * (capturing the boundary when asked).  Returns the slot; *delta = resume. */
static int convo_turn(pn_batch_sched *bs, const char *sid,
                      const uint32_t *prompt, int n, int *delta)
{
    int s = bs_slot_acquire_biased(bs, sid);
    s = bs_slot_reroute(bs, s, sid, prompt, n, 8);
    int cap = 0;
    bs_prefill_resume_sys(bs, s, sid, prompt, n, 8, delta, &cap);
    if (cap) {
        bs->slots[s].state->pos = 8;
        bs_prefill_capture_sys(bs, s, sid, prompt, 8);
    }
    return s;
}

/* End the turn: publish prompt + 4 generated tokens as the warm history. */
static int convo_commit(pn_batch_sched *bs, int s, const char *sid,
                        const uint32_t *prompt, int n, uint32_t *hist)
{
    memcpy(hist, prompt, (size_t)n * sizeof(uint32_t));
    for (int i = n; i < n + 4; i++)
        hist[i] = (uint32_t)(9000 + 10 * s + i);
    publish_warm(&bs->slots[s], sid, hist, n + 4);
    bs_prefill_abort(bs, s);
    return n + 4;
}

/* The next turn's prompt: the committed history plus two new tokens. */
static int convo_append(uint32_t *out, const uint32_t *hist, int n, int tag)
{
    memcpy(out, hist, (size_t)n * sizeof(uint32_t));
    out[n] = (uint32_t)(7000 + tag);
    out[n + 1] = (uint32_t)(7100 + tag);
    return n + 2;
}

/* The measured order: the title request is still decoding when turn 1 arrives,
 * so turn 1 lands on slot 1; turn 2 must come back to slot 1. */
static void test_two_convos_concurrent_title(void)
{
    printf("test: one sid, two conversations (title still running at turn 1)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    /* One session on both slots: a second copy is allowed only while no
     * other session could need that seat (max_sessions 1 of 2 slots). */
    bs_set_session_admission(bs, 1, 0);
    uint32_t t1[16], ht[20], m1[24], hm[28], m2[30];
    int delta = -1;

    mk_prompt(t1, SYS_B, 5000, 16);                    /* title conversation */
    int st = convo_turn(bs, "S", t1, 16, &delta);
    convo_commit(bs, st, "S", t1, 16, ht);
    bs->slots[st].status = BS_SLOT_DECODE;             /* still in flight */
    mk_prompt(m1, SYS_A, 1000, 24);                    /* real turn 1 */
    int sm = convo_turn(bs, "S", m1, 24, &delta);
    CHECK(sm != st && delta == 0, "turn 1 full-prefills on the other slot");
    int hm_n = convo_commit(bs, sm, "S", m1, 24, hm);
    bs->slots[st].status = BS_SLOT_IDLE;               /* title finished */

    int n2 = convo_append(m2, hm, hm_n, 2);
    int s2 = convo_turn(bs, "S", m2, n2, &delta);
    CHECK(s2 == sm, "turn 2 routed to turn 1's slot, not the title's");
    CHECK(delta == hm_n, "turn 2 prefills only its delta");
    CHECK(!bs_slot_session_conflict(bs, s2), "turn 2 is not a conflict");
    bs_prefill_abort(bs, s2);
    bs_destroy(bs);
}

/* Strictly sequential, interleaved turns: neither conversation evicts the
 * other, and each resumes from its own warm history / snapshot. */
static void test_two_convos_interleaved(void)
{
    printf("test: one sid, two conversations, interleaved turns\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    /* One session on both slots: a second copy is allowed only while no
     * other session could need that seat (max_sessions 1 of 2 slots). */
    bs_set_session_admission(bs, 1, 0);
    uint32_t t1[16], ht[20], t2[22], ht2[26];
    uint32_t m1[24], hm[28], m2[30], hm2[34], m3[40];
    int delta = -1;

    mk_prompt(t1, SYS_B, 5000, 16);
    int st = convo_turn(bs, "S", t1, 16, &delta);
    int ht_n = convo_commit(bs, st, "S", t1, 16, ht);

    mk_prompt(m1, SYS_A, 1000, 24);
    int sm = convo_turn(bs, "S", m1, 24, &delta);
    CHECK(sm != st, "a new conversation takes the cold slot, not the title's");
    CHECK(!bs_slot_session_conflict(bs, sm), "...and is not reported");
    int hm_n = convo_commit(bs, sm, "S", m1, 24, hm);
    CHECK(strcmp(bs->slots[st].sid, "S") == 0 && bs->slots[st].hist_n == ht_n,
          "the title conversation's warm history survived");

    int n = convo_append(t2, ht, ht_n, 1);
    int s = convo_turn(bs, "S", t2, n, &delta);
    CHECK(s == st && delta == ht_n, "title turn 2 resumes warm on its slot");
    convo_commit(bs, s, "S", t2, n, ht2);

    n = convo_append(m2, hm, hm_n, 2);
    s = convo_turn(bs, "S", m2, n, &delta);
    CHECK(s == sm && delta == hm_n, "main turn 2 resumes warm on its slot");
    convo_commit(bs, s, "S", m2, n, hm2);

    mk_prompt(m3, SYS_A, 4000, 40);   /* body rewritten, same system block */
    s = convo_turn(bs, "S", m3, 40, &delta);
    CHECK(s == sm && delta == 8,
          "main body rewrite resumes from its own system-prefix snapshot");
    bs_prefill_abort(bs, s);

    CHECK(bs_slot_reroute(bs, 0, NULL, m3, 40, 8) == 0, "NULL sid: no reroute");
    bs_destroy(bs);
}

/* ============================================================
 * Test 6a2 — the SINGLE-USER twin: two conversations under one id in
 * pn_prefix_cache (the path without a batch scheduler — every Tenstorrent
 * serve, and a CUDA serve with QWEN35_BATCH_SLOTS=0).  It kept one entry per
 * id, so the title request and the real turns reset each other's state and
 * snapshot on every alternation.
 * ============================================================ */

/* One single-user turn: acquire, capture the boundary when asked (SSM seeded
 * with `seed`, so a later restore can tell whose boundary came back). */
static pn_state *su_turn(pn_prefix_cache *c, const char *sid,
                         const uint32_t *prompt, int n, float seed, int *delta)
{
    int cap = 0;
    pn_state *st = pn_prefix_cache_acquire_sys(c, sid, prompt, n, 8, delta, &cap);
    if (st && cap) {
        fill_ssm((pn_qwen35moe_state *)st->kv_cache, seed);
        st->pos = 8;
        pn_prefix_cache_capture_sys(c, sid, prompt, 8);
    }
    return st;
}

/* End the turn: prompt + 4 generated tokens committed, state at their end. */
static int su_commit(pn_prefix_cache *c, pn_state *st, const char *sid,
                     const uint32_t *prompt, int n, uint32_t *hist, int tag)
{
    memcpy(hist, prompt, (size_t)n * sizeof(uint32_t));
    for (int i = n; i < n + 4; i++)
        hist[i] = (uint32_t)(9000 + 100 * tag + i);
    st->pos = n + 4;
    pn_prefix_cache_commit(c, sid, hist, n + 4);
    return n + 4;
}

static void test_single_user_two_convos_interleaved(void)
{
    printf("test: single-user cache, one sid, two conversations interleaved\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);
    uint32_t t1[16], ht[20], t2[22], ht2[26], t3[28];
    uint32_t m1[24], hm[28], m2[30], hm2[34], m3[40], hm3[44];
    int delta = -1;

    mk_prompt(t1, SYS_B, 5000, 16);                    /* title conversation */
    pn_state *st = su_turn(c, "S", t1, 16, 2.0f, &delta);
    CHECK(st && delta == 0, "title turn 1 full-prefills");
    int ht_n = su_commit(c, st, "S", t1, 16, ht, 1);

    mk_prompt(m1, SYS_A, 1000, 24);                    /* real turn 1 */
    pn_state *sm = su_turn(c, "S", m1, 24, 3.0f, &delta);
    CHECK(sm && delta == 0, "main turn 1 full-prefills");
    CHECK(sm != st, "main turn 1 takes its own entry, not the title's");
    int hm_n = su_commit(c, sm, "S", m1, 24, hm, 2);

    int n = convo_append(t2, ht, ht_n, 1);
    pn_state *s = su_turn(c, "S", t2, n, 0.0f, &delta);
    CHECK(s == st && delta == ht_n, "title turn 2 resumes its own history");
    int ht2_n = su_commit(c, s, "S", t2, n, ht2, 3);

    n = convo_append(m2, hm, hm_n, 2);
    s = su_turn(c, "S", m2, n, 0.0f, &delta);
    CHECK(s == sm && delta == hm_n, "main turn 2 resumes its own history");
    su_commit(c, s, "S", m2, n, hm2, 4);

    mk_prompt(m3, SYS_A, 4000, 40);   /* body rewritten, same system block */
    s = su_turn(c, "S", m3, 40, 0.0f, &delta);
    CHECK(s == sm && delta == 8 &&
          check_ssm((pn_qwen35moe_state *)s->kv_cache, 3.0f),
          "main body rewrite restores ITS OWN system-prefix snapshot");
    su_commit(c, s, "S", m3, 40, hm3, 5);

    n = convo_append(t3, ht2, ht2_n, 3);
    s = su_turn(c, "S", t3, n, 0.0f, &delta);
    CHECK(s == st && delta == ht2_n, "title turn 3 still resumes after that");
    pn_prefix_cache_free(c);
}

/* max_sessions counts IDS: a side conversation neither 429s a new id nor
 * costs another id its only conversation; memory stays max_sessions entries. */
static void test_single_user_side_convo_admission(void)
{
    printf("test: single-user cache, side conversations vs max_sessions\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);
    uint32_t t1[16], ht[20], m1[24], hm[28], m2[30], hm2[34];
    uint32_t u1[20], hu[24], u2[26];
    int delta = -1;

    mk_prompt(t1, SYS_B, 5000, 16);
    pn_state *st = su_turn(c, "S", t1, 16, 2.0f, &delta);
    int ht_n = su_commit(c, st, "S", t1, 16, ht, 1);
    mk_prompt(m1, SYS_A, 1000, 24);
    pn_state *sm = su_turn(c, "S", m1, 24, 3.0f, &delta);
    int hm_n = su_commit(c, sm, "S", m1, 24, hm, 2);

    CHECK(pn_prefix_cache_admit(c, "T", 1000, 0, NULL) == PN_SESSION_ADMIT_OK,
          "pool full of S's two conversations: new id T still admitted");
    mk_prompt(u1, SYS_A, 7000, 20);
    pn_state *tt = su_turn(c, "T", u1, 20, 4.0f, &delta);
    CHECK(tt && delta == 0, "T acquires a fresh entry");
    int hu_n = su_commit(c, tt, "T", u1, 20, hu, 6);
    CHECK(pn_prefix_cache_admit(c, "U", 1000, 0, NULL) ==
          PN_SESSION_ADMIT_AT_CAPACITY, "third id over max_sessions=2 refused");
    CHECK(pn_prefix_cache_admit(c, "S", 1000, 0, NULL) == PN_SESSION_ADMIT_OK,
          "S itself still admitted");
    int n = convo_append(m2, hm, hm_n, 2);
    pn_state *s = su_turn(c, "S", m2, n, 0.0f, &delta);
    CHECK(s == sm && delta == hm_n,
          "T took S's least recently used conversation; main survived");
    su_commit(c, s, "S", m2, n, hm2, 4);

    /* S's title comes back: S and T hold one entry each, so S's new
     * conversation gives up S's own entry, never T's only one. */
    uint32_t t2[22];
    n = convo_append(t2, ht, ht_n, 1);
    s = su_turn(c, "S", t2, n, 2.0f, &delta);
    CHECK(s && delta == 0 && s != tt, "S's title was evicted; new entry is not T's");
    su_commit(c, s, "S", t2, n, hm2, 7);
    n = convo_append(u2, hu, hu_n, 6);
    s = su_turn(c, "T", u2, n, 0.0f, &delta);
    CHECK(s == tt && delta == hu_n, "T's only conversation survived S's churn");
    pn_prefix_cache_release_session(c, "S");
    CHECK(pn_prefix_cache_admit(c, "U", 1000, 0, NULL) == PN_SESSION_ADMIT_OK,
          "release frees S's conversations");
    pn_prefix_cache_free(c);
}

/* Export takes an id's most recently used conversation; import replaces the
 * conversation that history continues and leaves the other one alone. */
static void test_single_user_export_import_two_convos(void)
{
    printf("test: single-user export/import with two conversations\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 3);
    uint32_t t1[16], ht[20], m1[24], hm[28], m2[30], t2[22];
    int delta = -1;

    mk_prompt(t1, SYS_B, 5000, 16);
    pn_state *st = su_turn(c, "S", t1, 16, 2.0f, &delta);
    int ht_n = su_commit(c, st, "S", t1, 16, ht, 1);
    mk_prompt(m1, SYS_A, 1000, 24);
    pn_state *sm = su_turn(c, "S", m1, 24, 3.0f, &delta);
    fill_kv_rows((pn_qwen35moe_state *)sm->kv_cache, 0, 28, 6.0f);
    int hm_n = su_commit(c, sm, "S", m1, 24, hm, 2);

    int n_tok = 0;
    const uint32_t *tok = NULL;
    void *snap = pn_prefix_cache_export_snapshot(c, "S", &n_tok, &tok);
    CHECK(snap && n_tok == hm_n && tok &&
          memcmp(tok, hm, (size_t)hm_n * sizeof(uint32_t)) == 0,
          "export takes the most recently used conversation (main)");

    /* Round trip into a fresh cache: the next turn appends to the import. */
    pn_prefix_cache *c2 = pn_prefix_cache_new(&model, 3);
    CHECK(pn_prefix_cache_import_snapshot(c2, "S", snap, hm, hm_n) == 0,
          "import into a fresh cache");
    int n = convo_append(m2, hm, hm_n, 2);
    pn_state *s = su_turn(c2, "S", m2, n, 0.0f, &delta);
    CHECK(s && delta == hm_n &&
          check_kv_rows((pn_qwen35moe_state *)s->kv_cache, 0, 28, 6.0f),
          "imported conversation resumes with its rows");
    pn_prefix_cache_free(c2);

    /* Back into the source cache: replaces main, title untouched. */
    fill_kv_rows((pn_qwen35moe_state *)sm->kv_cache, 0, 28, 0.0f);
    CHECK(pn_prefix_cache_import_snapshot(c, "S", snap, hm, hm_n) == 0,
          "import over the id's existing conversations");
    s = su_turn(c, "S", m2, n, 0.0f, &delta);
    CHECK(s == sm && delta == hm_n &&
          check_kv_rows((pn_qwen35moe_state *)s->kv_cache, 0, 28, 6.0f),
          "import replaced the main conversation it continues");
    int n2 = convo_append(t2, ht, ht_n, 1);
    s = su_turn(c, "S", t2, n2, 0.0f, &delta);
    CHECK(s == st && delta == ht_n, "the title conversation survived the import");
    pn_session_snapshot_free(model.config.arch, snap);
    pn_prefix_cache_free(c);
}

static void test_think_elided_projection(void)
{
    printf("test: think-elision projection (plain turn + bare tool call)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);

    /* Turn 1: prompt = 10 client tokens + a 1-token forced "<think>" prefill.
     * Generation = 3 invisible tokens (the think block) then 4 visible ones. */
    uint32_t prompt[11];
    for (int i = 0; i < 11; i++) prompt[i] = (uint32_t)(100 + i);
    uint32_t gen[7] = { 900, 901, 902, 500, 501, 502, 503 };
    int delta = -1, cap_req = -1;
    int slot = bs_slot_acquire_biased(bs, "A");
    CHECK(slot == 0, "cold session claims slot 0");
    bs_prefill_resume_sys(bs, slot, "A", prompt, 11, 4, &delta, &cap_req);
    CHECK(delta == 0, "turn 1 is a cold full prefill");
    seed_committed_log(&bs->slots[slot], prompt, 11, gen, 7);
    /* The serve's re-render of the reply: the 4 visible tokens.  The 3 think
     * tokens are never in it — the client was never shown them. */
    uint32_t rerender[4] = { 500, 501, 502, 503 };
    bs_slot_note_visible(bs, slot, prompt, 11, 1, NULL, 0, rerender, 4);
    CHECK(bs->slots[slot].pend_vis_n == 10 + 4,
          "projection = client head (10) + re-rendered reply (4)");
    publish_vis(&bs->slots[slot]);
    publish_warm(&bs->slots[slot], "A", bs->slots[slot].ct, 18);
    bs_prefill_abort(bs, slot);

    /* Turn 2: the client resends exactly the projection + a new user turn. */
    uint32_t next[18];
    memcpy(next, bs->slots[0].vis, 14 * sizeof(uint32_t));
    for (int i = 14; i < 18; i++) next[i] = (uint32_t)(700 + i);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", next, 18, 4, &delta, &cap_req);
    CHECK(delta == 14, "think-elided resume: prefill only the new tail");
    CHECK(!bs_slot_session_conflict(bs, slot),
          "a think-elided re-render is a continuation, not a conflict");
    CHECK(bs->slots[slot].ct_keep == 18 && bs->slots[slot].ct_prompt_from == 14,
          "committed-log seeding records the elided KV length");
    bs_prefill_abort(bs, slot);

    /* A BARE tool call: the model generated no wrapper, the client's re-render
     * adds "<tool_call>\n" (two ids). Without them the projection is short by
     * the wrapper and every such turn misses. */
    bs->slots[0].sid[0] = '\0';
    bs->slots[0].hist_n = 0;
    bs->slots[0].vis_n = 0;
    uint32_t wrapped_rerender[6] = { 248058, 198, 500, 501, 502, 503 };
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", prompt, 11, 4, &delta, &cap_req);
    seed_committed_log(&bs->slots[slot], prompt, 11, gen, 7);
    bs_slot_note_visible(bs, slot, prompt, 11, 1, NULL, 0,
                         wrapped_rerender, 6);
    CHECK(bs->slots[slot].pend_vis_n == 10 + 6,
          "tool projection = head + the WRAPPED re-render of the bare call");
    publish_vis(&bs->slots[slot]);
    CHECK(bs->slots[slot].vis[10] == 248058 && bs->slots[slot].vis[11] == 198,
          "the wrapper sits between the head and the generated call");
    publish_warm(&bs->slots[slot], "A", bs->slots[slot].ct, 18);
    bs_prefill_abort(bs, slot);

    uint32_t next2[20];
    memcpy(next2, bs->slots[0].vis, 16 * sizeof(uint32_t));
    for (int i = 16; i < 20; i++) next2[i] = (uint32_t)(800 + i);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", next2, 20, 4, &delta, &cap_req);
    CHECK(delta == 16, "a wrapped re-render of a bare call still resumes");
    bs_prefill_abort(bs, slot);

    /* A cold, unrelated session must NOT evict slot 0's warm state while an
     * idle slot with nothing in it exists. */
    int other = bs_slot_acquire_biased(bs, "B");
    CHECK(other == 1, "a cold session takes the empty slot, not the warm one");
    CHECK(bs->slots[0].sid[0] != '\0', "slot 0 keeps its warm session");
    bs_prefill_resume_sys(bs, other, "B", prompt, 11, 4, &delta, &cap_req);
    bs_prefill_abort(bs, other);

    bs_destroy(bs);
}

/* ============================================================
 * Test 6b2 — THE REASONING-BEARING RUNG: a client that SENDS reasoning back
 *
 * A client must resend `reasoning_content` or the model stops reasoning (18/18
 * collapse without it, 0/18 with it).
 * But that history no longer matches the think-ELIDED projection, so before this
 * rung a reasoning-bearing client re-prefilled its whole conversation every
 * turn — minutes per turn at ~22 ms/token, which is what made the correct client
 * behaviour unusable.
 *
 * What must hold, and each is asserted below:
 *   1. A reasoning-bearing request RESUMES (delta == the projection length), so
 *      the reasoning span costs nothing.
 *   2. A request whose visible prefix DIVERGES falls back to a FULL prefill —
 *      the non-negotiable half.  Reusing KV that does not correspond to the
 *      request's prefix is silently wrong output, worse than a slow turn.
 *   3. The elided rung still fires for a client that drops reasoning, off the
 *      SAME committed turn: the two candidates are decided by the request.
 *   4. The first turn of a session is a cold full prefill (nothing to resume).
 * ============================================================ */

/* Commit one turn on slot 0 for session "A" with BOTH projections recorded: a
 * 10-token client head, a 3-token think block, a 4-token reply.  The KV holds 18
 * tokens (prompt 11 incl. a 1-token forced "<think>", plus 7 generated). */
static void commit_turn_with_reasoning(pn_batch_sched *bs,
                                       const uint32_t *prompt,
                                       const uint32_t *think_block,
                                       int think_block_n,
                                       const uint32_t *rerender)
{
    uint32_t gen[7] = { 900, 901, 902, 500, 501, 502, 503 };
    int delta = -1, cap_req = -1;
    int slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", prompt, 11, 4, &delta, &cap_req);
    CHECK(delta == 0, "first turn of a session is a cold full prefill");
    seed_committed_log(&bs->slots[slot], prompt, 11, gen, 7);
    bs_slot_note_visible(bs, slot, prompt, 11, 1, think_block, think_block_n,
                         rerender, 4);
    CHECK(bs->slots[slot].pend_vis_n == 10 + 4 &&
          bs->slots[slot].pend_vis_think_n == 10 + think_block_n + 4,
          "both projections recorded off one turn");
    publish_vis(&bs->slots[slot]);
    publish_warm(&bs->slots[slot], "A", bs->slots[slot].ct, 18);
    bs_prefill_abort(bs, slot);
}

static void test_reasoning_bearing_resume(void)
{
    printf("test: reasoning-bearing resume (the client sends reasoning back)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    int delta = -1, cap_req = -1;

    uint32_t prompt[11];
    for (int i = 0; i < 11; i++) prompt[i] = (uint32_t)(100 + i);
    /* The think block as the BUILDER re-renders it from the reasoning TEXT —
     * deliberately NOT the ids the model sampled (900,901,902, which are what
     * the KV holds).  That is the whole design: the client echoes the string,
     * the serve re-derives the ids, and the KV stays authoritative for the
     * span.  A rung that required the ids to agree would never fire. */
    uint32_t think_block[3] = { 248068, 310, 248069 };
    uint32_t rerender[4] = { 500, 501, 502, 503 };
    uint32_t sampled[3] = { 900, 901, 902 };
    commit_turn_with_reasoning(bs, prompt, think_block, 3, rerender);
    CHECK(bs->slots[0].vis_n == 14 && bs->slots[0].vis_think_n == 17,
          "finalize publishes both projections against the 18-token KV");
    CHECK(bs->slots[0].vis_think_n == 17 &&
          memcmp(bs->slots[0].vis_think + 10, think_block,
                 3 * sizeof(uint32_t)) == 0 &&
          memcmp(bs->slots[0].hist + 11, sampled, 3 * sizeof(uint32_t)) == 0,
          "the projection holds the RE-RENDERED block, the KV the sampled one");

    /* 1. Turn 2, reasoning resent: the client posts the reasoning-bearing
     *    projection plus a new user turn.  It must RESUME. */
    uint32_t with_reasoning[21];
    memcpy(with_reasoning, prompt, 10 * sizeof(uint32_t));         /* head */
    memcpy(with_reasoning + 10, think_block, 3 * sizeof(uint32_t));
    memcpy(with_reasoning + 13, rerender, 4 * sizeof(uint32_t));
    for (int i = 17; i < 21; i++) with_reasoning[i] = (uint32_t)(700 + i);
    int slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", with_reasoning, 21, 4, &delta, &cap_req);
    CHECK(delta == 17,
          "a reasoning-bearing history RESUMES: prefill only the new tail");
    CHECK(bs->slots[slot].ct_keep == 18 && bs->slots[slot].ct_prompt_from == 17,
          "the committed log is seeded from the KV, not the client's rendering");
    CHECK(!bs_slot_session_conflict(bs, slot),
          "resending reasoning is a continuation, not a session substitution");
    CHECK(pn_prefix_cached_tokens(delta, 21) == 17,
          "and the wire reports the reuse it actually performed");
    bs_prefill_abort(bs, slot);

    /* 2. THE TRAP.  Same length, same shape, one token different INSIDE the
     *    reasoning span — a client that re-rendered the reasoning differently.
     *    The KV does not correspond to this prefix, so the only correct answer
     *    is to throw it away and prefill from scratch. */
    uint32_t diverged[21];
    memcpy(diverged, with_reasoning, 21 * sizeof(uint32_t));
    diverged[11] = 999;   /* inside the re-rendered think block */
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", diverged, 21, 4, &delta, &cap_req);
    CHECK(delta == 0,
          "a divergence inside the reasoning span FALLS BACK to a full prefill");
    bs_prefill_abort(bs, slot);

    /* A divergence in the client HEAD — the span both projections share — must
     * fall back too, and it is the case where a wrong resume would splice one
     * conversation onto another's KV. */
    memcpy(diverged, with_reasoning, 21 * sizeof(uint32_t));
    diverged[9] = 999;
    bs->slots[0].sid[0] = '\0';
    bs->slots[0].hist_n = 0;
    commit_turn_with_reasoning(bs, prompt, think_block, 3, rerender);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", diverged, 21, 4, &delta, &cap_req);
    CHECK(delta == 0, "a divergence in the client head falls back too");
    bs_prefill_abort(bs, slot);

    /* 3. The SAME committed turn, asked by a client that DROPS reasoning: the
     *    elided rung must still fire, unchanged.  Adding a candidate must not
     *    take one away. */
    bs->slots[0].sid[0] = '\0';
    bs->slots[0].hist_n = 0;
    commit_turn_with_reasoning(bs, prompt, think_block, 3, rerender);
    uint32_t without_reasoning[18];
    memcpy(without_reasoning, prompt, 10 * sizeof(uint32_t));       /* head */
    memcpy(without_reasoning + 10, rerender, 4 * sizeof(uint32_t));
    for (int i = 14; i < 18; i++) without_reasoning[i] = (uint32_t)(700 + i);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", without_reasoning, 18, 4, &delta,
                          &cap_req);
    CHECK(delta == 14,
          "the same turn still resumes for a client that OMITS reasoning");
    CHECK(bs->slots[slot].ct_keep == 18 && bs->slots[slot].ct_prompt_from == 14,
          "the elided rung's committed-log seeding is unchanged");
    bs_prefill_abort(bs, slot);

    /* 4. COLD MUST STILL BE CORRECT, and the sharp version of cold is a NEW
     *    session id landing on the slot another one left warm — its tokens then
     *    byte-MATCH a projection that describes somebody else's KV.  Only A's
     *    own family may take A's slot now, so force it with "A-new": reserve
     *    the spare slot so it has to take slot 0. */
    int busy = bs_slot_acquire(bs);
    CHECK(busy == 1, "the spare slot is reserved so the cold session must take 0");
    int cold = bs_slot_acquire_biased(bs, "A-new");
    CHECK(cold == 0 && bs->slots[0].vis_think_n == 17,
          "the cold session takes the slot still holding A's projection");
    bs_prefill_resume_sys(bs, cold, "A-new", with_reasoning, 21, 4, &delta,
                          &cap_req);
    CHECK(delta == 0,
          "a first turn carrying reasoning is a cold full prefill even when it "
          "byte-matches another session's projection");
    CHECK(cap_req == 1, "and it captures a system boundary for next time");
    bs_prefill_abort(bs, cold);
    /* `busy` was only RESERVED — it never took a GPU turn, so releasing the slot
     * is the whole undo.  bs_prefill_abort would also release a turn nobody
     * holds and desync the turnstile. */
    bs_slot_release(bs, busy);

    /* 5. A reasoning-bearing projection LONGER than the committed KV must still
     *    resume.  This is the normal case, not a corner: the client's rendering
     *    carries the template's own "<think>\n" / "</think>\n\n" separators where
     *    the KV carries whatever the model sampled around the same text, and the
     *    emitter trimmed the leading newline out of the reasoning string — so
     *    the two lengths land within a token or two of each other and the
     *    projection routinely comes out the longer one.  A "must not exceed
     *    hist_n" tripwire (which the think-ELIDED projection does carry, and
     *    rightly) would silently disable this rung on a coin flip.  Here the KV
     *    is 18 and a 5-token block puts the projection at 19. */
    bs->slots[0].sid[0] = '\0';
    bs->slots[0].hist_n = 0;
    uint32_t long_block[5] = { 248068, 310, 311, 312, 248069 };
    commit_turn_with_reasoning(bs, prompt, long_block, 5, rerender);
    CHECK(bs->slots[0].hist_n == 18 && bs->slots[0].vis_think_n == 19,
          "the reasoning-bearing projection (19) outruns the KV it maps (18)");
    uint32_t long_req[23];
    memcpy(long_req, prompt, 10 * sizeof(uint32_t));
    memcpy(long_req + 10, long_block, 5 * sizeof(uint32_t));
    memcpy(long_req + 15, rerender, 4 * sizeof(uint32_t));
    for (int i = 19; i < 23; i++) long_req[i] = (uint32_t)(700 + i);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", long_req, 23, 4, &delta, &cap_req);
    CHECK(delta == 19,
          "and it resumes anyway: the gate is the memcmp, not the length");
    CHECK(bs->slots[slot].ct_keep == 18 && bs->slots[slot].ct_prompt_from == 19,
          "the committed log still starts from the KV the state actually holds");
    bs_prefill_abort(bs, slot);

    /* 6. The other side of that asymmetry: the think-ELIDED projection strictly
     *    DROPS tokens from the committed conversation, so one longer than the
     *    history it maps is nonsense and must be refused.  Here a 9-token reply
     *    re-render puts it at 19 against an 18-token KV. */
    bs->slots[0].sid[0] = '\0';
    bs->slots[0].hist_n = 0;
    uint32_t gen[7] = { 900, 901, 902, 500, 501, 502, 503 };
    uint32_t fat_rerender[9] = { 500, 501, 502, 503, 504, 505, 506, 507, 508 };
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", prompt, 11, 4, &delta, &cap_req);
    seed_committed_log(&bs->slots[slot], prompt, 11, gen, 7);
    bs_slot_note_visible(bs, slot, prompt, 11, 1, think_block, 3,
                         fat_rerender, 9);
    bs_slot_publish_projections(&bs->slots[slot], bs->slots[slot].ct_n);
    CHECK(bs->slots[slot].vis_n == 0,
          "an elided projection longer than its own KV (19 > 18) is refused");
    CHECK(bs->slots[slot].vis_think_n == 22,
          "and refusing it does not take the reasoning-bearing one down too");
    bs_prefill_abort(bs, slot);

    bs_destroy(bs);
}

/* ============================================================
 * Test 6b3 — CHANGING THE REASONING CONVENTION MID-SESSION
 *
 * The wanted capability is a harness that picks OFF / SUMMARY / FULL per turn
 * from the situation (circling -> SUMMARY, hard diagnosis -> FULL, context
 * tight -> step down).  That is only worth building if a switch is free, so
 * this test drives the real ladder across switches and asserts what each one
 * costs.  The three questions it answers, and the answers it asserts:
 *
 *   1. WHAT IS `head`?  The tokens the client ACTUALLY SENT for this turn,
 *      minus the forced generation prefix (bs_slot_note_visible takes
 *      prompt[0, n_prompt - think_prefill_len)) — never anything reconstructed
 *      from the KV.  So every earlier turn is frozen at whatever rendering the
 *      client used when it sent it, and the two candidates decide ONLY the
 *      newest assistant turn's think block.
 *   2. A FORWARD SWITCH IS FREE.  Turn 2 renders turn 1 with its reasoning
 *      (think-included hits); turn 3 renders turn 2 WITHOUT (think-elided hits
 *      off the very same committed turn).  Neither costs a token.
 *   3. REWRITING THE PAST IS NOT.  A client that drops a block it already sent
 *      — the "reasoning on the last turn only" sliding window — changes `head`
 *      and defeats both candidates on every turn it slides.
 *   4. A THIRD CONVENTION IS NOT FREE EITHER, ONCE.  A SUMMARY is neither
 *      candidate: the block is present with different text, so it misses on the
 *      turn it is introduced.  It costs exactly that turn — by the next one the
 *      summary is part of `head` and the ladder resumes again.
 * ============================================================ */

static int cat_tokens(uint32_t *out, int at, const uint32_t *src, int n)
{
    memcpy(out + at, src, (size_t)n * sizeof(uint32_t));
    return at + n;
}

/* The conversation's pieces.  A think block is 3 tokens whichever convention
 * renders it, so LENGTH is never what tells the candidates apart — only the
 * reasoning token in the middle differs (310/311 = the model's own reasoning,
 * 321 = a summary of it).  A rung that discriminated by length would pass the
 * forward-switch cases and silently accept the summary. */
static const uint32_t CS_SYS[4]  = { 100, 101, 102, 103 };   /* sys_len = 4 */
static const uint32_t CS_U1[3]   = { 110, 111, 112 };
static const uint32_t CS_U2[3]   = { 120, 121, 122 };
static const uint32_t CS_U3[3]   = { 130, 131, 132 };
static const uint32_t CS_U4[3]   = { 140, 141, 142 };
static const uint32_t CS_TP[1]   = { 248068 };               /* forced <think> */
static const uint32_t CS_B1[3]   = { 248068, 310, 248069 };  /* turn 1, FULL   */
static const uint32_t CS_B2[3]   = { 248068, 311, 248069 };  /* turn 2, FULL   */
static const uint32_t CS_S2[3]   = { 248068, 321, 248069 };  /* turn 2, SUMMARY*/
static const uint32_t CS_B3[3]   = { 248068, 312, 248069 };  /* turn 3, FULL   */
static const uint32_t CS_R1[4]   = { 500, 501, 502, 503 };   /* re-rendered reply */
static const uint32_t CS_R2[4]   = { 510, 511, 512, 513 };
static const uint32_t CS_R3[4]   = { 520, 521, 522, 523 };
/* Generated: 3 sampled think ids (NOT the client's rendering) + the reply. */
static const uint32_t CS_GEN1[7] = { 900, 901, 902, 500, 501, 502, 503 };
static const uint32_t CS_GEN2[7] = { 910, 911, 912, 510, 511, 512, 513 };
static const uint32_t CS_GEN3[7] = { 920, 921, 922, 520, 521, 522, 523 };

enum { CS_SYS_LEN = 4 };

/* Run one turn through the production entry points and commit it.
 *
 * The committed log is seeded by bs_slot_bind's own rule (batch_sched.c,
 * ct_keep / ct_prompt_from): hist[0, ct_keep) ++ req[ct_prompt_from, n_req) ++
 * the generated tokens.  Copying that rule rather than "prompt ++ gen" is what
 * makes the chain faithful across a projection resume, where the KV keeps the
 * ids the model SAMPLED inside its think block and the client's rendering only
 * covers the tail.  Returns the ladder's delta. */
static int commit_convention_turn(pn_batch_sched *bs, const char *sid,
                                  const uint32_t *req, int n_req,
                                  const uint32_t *gen, int n_gen,
                                  const uint32_t *block, int block_n,
                                  const uint32_t *rerender, int rerender_n)
{
    int delta = -1, cap_req = -1;
    int slot = bs_slot_acquire_biased(bs, sid);
    bs_prefill_resume_sys(bs, slot, sid, req, n_req, CS_SYS_LEN, &delta,
                          &cap_req);
    bs_slot *sl = &bs->slots[slot];
    if (cap_req) {
        /* The ladder asked for a boundary snapshot: take it mid-prefill exactly
         * as the caller does, so a later MISS falls to the sys-prefix rung and
         * the test measures the cost the fleet would actually pay. */
        pn_qwen35moe_state *q = (pn_qwen35moe_state *)sl->state->kv_cache;
        fill_ssm(q, 1.0f);
        fill_kv_rows(q, 0, CS_SYS_LEN, 1.0f);
        sl->state->pos = CS_SYS_LEN;
        bs_prefill_capture_sys(bs, slot, sid, req, CS_SYS_LEN);
    }
    int keep = sl->ct_keep, from = sl->ct_prompt_from;
    int n = keep + (n_req - from) + n_gen;
    if (sl->ct_cap < n) {
        sl->ct = realloc(sl->ct, (size_t)n * sizeof(uint32_t));
        sl->ct_cap = n;
    }
    if (keep > 0)
        memcpy(sl->ct, sl->hist, (size_t)keep * sizeof(uint32_t));
    memcpy(sl->ct + keep, req + from, (size_t)(n_req - from) * sizeof(uint32_t));
    memcpy(sl->ct + keep + (n_req - from), gen, (size_t)n_gen * sizeof(uint32_t));
    sl->ct_n = n;
    bs_slot_note_visible(bs, slot, req, n_req, 1, block, block_n,
                         rerender, rerender_n);
    publish_vis(sl);
    publish_warm(sl, sid, sl->ct, n);
    bs_prefill_abort(bs, slot);
    return delta;
}

/* Ask the ladder what a request would cost WITHOUT committing it.  A hit leaves
 * the slot untouched; a miss burns the warm state, which is the point of
 * re-seeding before each miss probe. */
static int probe_turn(pn_batch_sched *bs, const char *sid,
                      const uint32_t *req, int n_req)
{
    int delta = -1, cap_req = -1;
    int slot = bs_slot_acquire_biased(bs, sid);
    bs_prefill_resume_sys(bs, slot, sid, req, n_req, CS_SYS_LEN, &delta,
                          &cap_req);
    bs_prefill_abort(bs, slot);
    return delta;
}

/* Turns 1 and 2 of the conversation, committed.  Turn 1 is cold and captures
 * the system boundary; turn 2 is the client's FIRST convention choice — it
 * renders turn 1 WITH the reasoning, which is the OFF->FULL forward switch. */
static void seed_two_turns(pn_batch_sched *bs, uint32_t *req2, int *n2_out)
{
    uint32_t req1[8];
    int n1 = cat_tokens(req1, 0, CS_SYS, 4);
    n1 = cat_tokens(req1, n1, CS_U1, 3);
    n1 = cat_tokens(req1, n1, CS_TP, 1);
    CHECK(commit_convention_turn(bs, "A", req1, n1, CS_GEN1, 7, CS_B1, 3,
                                 CS_R1, 4) == 0,
          "turn 1 is a cold full prefill and captures the system boundary");
    CHECK(bs->slots[0].vis_n == 11 && bs->slots[0].vis_think_n == 14,
          "turn 1 records both conventions: elided 11, reasoning-bearing 14");

    int n2 = cat_tokens(req2, 0, CS_SYS, 4);
    n2 = cat_tokens(req2, n2, CS_U1, 3);
    n2 = cat_tokens(req2, n2, CS_B1, 3);      /* turn 1 rendered WITH reasoning */
    n2 = cat_tokens(req2, n2, CS_R1, 4);
    n2 = cat_tokens(req2, n2, CS_U2, 3);
    n2 = cat_tokens(req2, n2, CS_TP, 1);
    *n2_out = n2;
    CHECK(commit_convention_turn(bs, "A", req2, n2, CS_GEN2, 7, CS_B2, 3,
                                 CS_R2, 4) == 14,
          "turn 2 sends reasoning for the first time and RESUMES (delta 14)");
}

static void test_reasoning_convention_switch(void)
{
    printf("test: switching the reasoning convention mid-session\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    /* ONE slot: affinity cannot quietly hand a turn a cold slot and make a
     * resume assertion pass for the wrong reason. */
    pn_batch_sched *bs = mk_sched(&model, 1);

    uint32_t req2[18];
    int n2 = 0;
    seed_two_turns(bs, req2, &n2);
    CHECK(n2 == 18 && bs->slots[0].vis_n == 21 && bs->slots[0].vis_think_n == 24,
          "turn 2's projections are built on the head the client SENT (17), "
          "which already carries turn 1's block");

    /* 1. FORWARD SWITCH, FULL -> OFF.  Turn 3 keeps turn 1 exactly as it sent
     *    it and drops the block for turn 2 only.  The reasoning-bearing
     *    candidate is tried FIRST and must decline; the elided one hits. */
    uint32_t off[25];
    int n_off = cat_tokens(off, 0, req2, 17);   /* the head, byte-for-byte */
    n_off = cat_tokens(off, n_off, CS_R2, 4);
    n_off = cat_tokens(off, n_off, CS_U3, 3);
    n_off = cat_tokens(off, n_off, CS_TP, 1);
    CHECK(n_off == 25 && probe_turn(bs, "A", off, n_off) == 21,
          "FULL -> OFF on the newest turn RESUMES (delta 21): a forward switch "
          "costs nothing");

    /* 2. The other direction off the SAME committed turn: stay FULL. */
    uint32_t full[28];
    int n_full = cat_tokens(full, 0, req2, 17);
    n_full = cat_tokens(full, n_full, CS_B2, 3);
    n_full = cat_tokens(full, n_full, CS_R2, 4);
    n_full = cat_tokens(full, n_full, CS_U3, 3);
    n_full = cat_tokens(full, n_full, CS_TP, 1);
    CHECK(n_full == 28 && probe_turn(bs, "A", full, n_full) == 24,
          "staying FULL resumes off the same turn (delta 24): the request "
          "picks the candidate, the commit never did");

    /* 3. THE THIRD CONVENTION.  A SUMMARY renders the block with different text,
     *    so it is neither recorded candidate and the ladder drops to the
     *    system-prefix rung: 24 of 28 tokens re-prefill.  This is where the
     *    per-turn OFF/SUMMARY/FULL idea costs something. */
    uint32_t summ[28];
    int n_summ = cat_tokens(summ, 0, req2, 17);
    n_summ = cat_tokens(summ, n_summ, CS_S2, 3);   /* same length as CS_B2 */
    n_summ = cat_tokens(summ, n_summ, CS_R2, 4);
    n_summ = cat_tokens(summ, n_summ, CS_U3, 3);
    n_summ = cat_tokens(summ, n_summ, CS_TP, 1);
    CHECK(n_summ == n_full,
          "the summary is the same LENGTH as the reasoning it replaces, so only "
          "its content can be what the ladder rejects");
    CHECK(commit_convention_turn(bs, "A", summ, n_summ, CS_GEN3, 7, CS_B3, 3,
                                 CS_R3, 4) == CS_SYS_LEN,
          "introducing a SUMMARY MISSES both candidates: it falls to the "
          "system-prefix rung and re-prefills the conversation");

    /* 4. ...but only once.  The summary is now part of the head the client
     *    sent, so it is frozen into turn 3's projections like any other text
     *    and the next turn resumes again. */
    uint32_t next[35];
    int n_next = cat_tokens(next, 0, summ, 27);   /* head as sent, incl. S2 */
    n_next = cat_tokens(next, n_next, CS_R3, 4);
    n_next = cat_tokens(next, n_next, CS_U4, 3);
    n_next = cat_tokens(next, n_next, CS_TP, 1);
    CHECK(n_next == 35 && probe_turn(bs, "A", next, n_next) == 31,
          "the turn AFTER a summary resumes (delta 31): the cost of a new "
          "convention is ONE turn, paid where it is introduced");

    /* 5. REWRITING THE PAST.  "Reasoning on the newest turn only" is the
     *    obvious harness policy and it is the one that cannot work: at turn 3
     *    it drops the block it already sent for turn 1, which changes the head
     *    both candidates were built on.  Nothing about this turn is new — same
     *    conversation, same tokens for turn 2 — and it still misses. */
    bs_destroy(bs);
    bs = mk_sched(&model, 1);
    seed_two_turns(bs, req2, &n2);
    uint32_t slide[25];
    int n_slide = cat_tokens(slide, 0, CS_SYS, 4);
    n_slide = cat_tokens(slide, n_slide, CS_U1, 3);
    n_slide = cat_tokens(slide, n_slide, CS_R1, 4);   /* turn 1's block DROPPED */
    n_slide = cat_tokens(slide, n_slide, CS_U2, 3);
    n_slide = cat_tokens(slide, n_slide, CS_B2, 3);
    n_slide = cat_tokens(slide, n_slide, CS_R2, 4);
    n_slide = cat_tokens(slide, n_slide, CS_U3, 3);
    n_slide = cat_tokens(slide, n_slide, CS_TP, 1);
    CHECK(n_slide == 25 && probe_turn(bs, "A", slide, n_slide) == CS_SYS_LEN,
          "retroactively DROPPING a block the client already sent misses both "
          "candidates: a sliding reasoning window re-prefills every turn");

    bs_destroy(bs);
}

/* ============================================================
 * Test 6d — CACHED-TOKENS REPORTING: pn_prefix_cached_tokens must equal the
 * reuse each resume rung ACTUALLY performed.
 *
 * The number the serve puts on the wire as
 * usage.prompt_tokens_details.cached_tokens is derived from the cache's own
 * chosen resume boundary — never recomputed from the prompt, which would
 * instrument the cache with a second implementation of itself and agree with it
 * for the wrong reason. So this test drives the REAL ladder (both the
 * single-user acquire_sys and the batched bs_prefill_resume_sys) and asserts
 * the reported count against what each rung reused: 0 on a cold turn, the
 * history length on an append, the projection length on a think-elided append,
 * sys_len on a boundary restore, the LCP on a llama-family rewind.
 * ============================================================ */

/* Report the cached count for a rung and check it against BOTH the cache's own
 * delta (the derivation) and the length that rung was supposed to reuse (so a
 * ladder that silently stopped reusing cannot pass by reporting its own zero). */
static void check_cached(int delta, int n_ctx, int want, const char *msg)
{
    int cached = pn_prefix_cached_tokens(delta, n_ctx);
    CHECK(cached == want && cached == delta && cached <= n_ctx, msg);
}

static void test_cached_tokens_tracks_resume_rungs(void)
{
    printf("test: cached_tokens tracks every resume rung\n");
    g_make_q4 = 0;
    int delta = -1, cap = -1;

    /* ── Single-user cache, qwen35 (snapshot family). ── */
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);

    uint32_t p1[20];
    mk_prompt(p1, SYS_A, 1000, 20);
    pn_state *st = pn_prefix_cache_acquire_sys(c, "s1", p1, 20, 8, &delta, &cap);
    check_cached(delta, 20, 0, "cold turn reports 0 cached tokens");

    pn_qwen35moe_state *qs = (pn_qwen35moe_state *)st->kv_cache;
    fill_ssm(qs, 1.0f);
    fill_kv_rows(qs, 0, 8, 1.0f);
    st->pos = 8;
    CHECK(pn_prefix_cache_capture_sys(c, "s1", p1, 8) == 0, "boundary captured");
    fill_kv_rows(qs, 8, 24, 5.0f);
    st->pos = 24;
    uint32_t hist1[24];
    memcpy(hist1, p1, 20 * sizeof(uint32_t));
    for (int i = 20; i < 24; i++) hist1[i] = (uint32_t)(9000 + i);
    CHECK(pn_prefix_cache_commit(c, "s1", hist1, 24) == 0, "turn 1 commit");

    /* Append rung: the whole committed history is reused. */
    uint32_t p2[26];
    memcpy(p2, hist1, 24 * sizeof(uint32_t));
    p2[24] = 7001; p2[25] = 7002;
    pn_prefix_cache_acquire_sys(c, "s1", p2, 26, 8, &delta, &cap);
    check_cached(delta, 26, 24, "append resume reports the reused history");

    /* Sys-prefix restore rung: only the system prefix is reused. */
    uint32_t p3[30];
    mk_prompt(p3, SYS_A, 4000, 30);
    pn_prefix_cache_acquire_sys(c, "s1", p3, 30, 8, &delta, &cap);
    check_cached(delta, 30, 8, "sys-prefix restore reports sys_len cached");
    pn_prefix_cache_free(c);

    /* ── LCP-rewind rung (llama family — no snapshots). ── */
    pn_model lmodel = {0};
    lmodel.config.arch = PN_ARCH_LLAMA;
    pn_prefix_cache *lc = pn_prefix_cache_new(&lmodel, 2);
    st = pn_prefix_cache_acquire_sys(lc, "r1", p1, 20, 8, &delta, &cap);
    check_cached(delta, 20, 0, "llama cold turn reports 0 cached tokens");
    st->pos = 24;
    CHECK(pn_prefix_cache_commit(lc, "r1", hist1, 24) == 0, "llama turn 1 commit");
    pn_prefix_cache_acquire_sys(lc, "r1", p3, 30, 8, &delta, &cap);
    check_cached(delta, 30, 8, "lcp rewind reports the common prefix cached");
    pn_prefix_cache_free(lc);

    /* ── Batched twin: the path the fleet actually runs. ── */
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t prompt[11];
    for (int i = 0; i < 11; i++) prompt[i] = (uint32_t)(100 + i);
    uint32_t gen[7] = { 900, 901, 902, 500, 501, 502, 503 };
    int slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", prompt, 11, 4, &delta, &cap);
    check_cached(delta, 11, 0, "batched cold turn reports 0 cached tokens");

    /* Think-elided append: the client resends its own projection, so the reuse
     * is the PROJECTION length (14), not the slot's longer elided KV (18). */
    seed_committed_log(&bs->slots[slot], prompt, 11, gen, 7);
    uint32_t rerender[4] = { 500, 501, 502, 503 };
    bs_slot_note_visible(bs, slot, prompt, 11, 1, NULL, 0, rerender, 4);
    publish_vis(&bs->slots[slot]);
    publish_warm(&bs->slots[slot], "A", bs->slots[slot].ct, 18);
    bs_prefill_abort(bs, slot);
    uint32_t next[18];
    memcpy(next, bs->slots[0].vis, 14 * sizeof(uint32_t));
    for (int i = 14; i < 18; i++) next[i] = (uint32_t)(700 + i);
    slot = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, slot, "A", next, 18, 4, &delta, &cap);
    check_cached(delta, 18, 14,
                 "think-elided append reports the projection, not the KV length");
    bs_prefill_abort(bs, slot);
    bs_destroy(bs);

    /* Degenerate boundary: a resume position at or past the prompt still reports
     * a count a client can divide by (computed = prompt - cached >= 0). */
    CHECK(pn_prefix_cached_tokens(40, 30) == 30, "cached clamps to prompt length");
    CHECK(pn_prefix_cached_tokens(-1, 30) == 0, "negative boundary reports 0");
}

/* ============================================================
 * Test 7 — Phase 5 session admission: REFUSE at capacity + idle TTL
 * ============================================================ */

static void test_session_admission(void)
{
    printf("test: session admission (refuse at capacity + idle TTL)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 2);   /* cap = 2 */
    CHECK(c != NULL, "cache created (cap 2)");

    int rt = -1;
    /* Empty cache: any new sid is admitted (a free slot exists). */
    CHECK(pn_prefix_cache_admit(c, "a", 1000, 0, &rt) == PN_SESSION_ADMIT_OK &&
          rt == 0, "empty cache admits new sid 'a' (restart_tokens 0)");

    /* Fill both slots with two live sessions via acquire. */
    uint32_t p[20];
    mk_prompt(p, SYS_A, 1000, 20);
    int delta = -1;
    pn_state *sa = pn_prefix_cache_acquire(c, "a", p, 20, &delta);
    pn_state *sb = pn_prefix_cache_acquire(c, "b", p, 20, &delta);
    CHECK(sa && sb, "two sessions 'a','b' acquired (both slots live)");
    pn_prefix_cache_touch(c, "a", 1000);
    pn_prefix_cache_touch(c, "b", 1000);
    /* Give 'a' a committed history so its restart estimate is non-zero. */
    CHECK(pn_prefix_cache_commit(c, "a", p, 20) == 0, "commit 'a' history (20 tok)");

    /* Existing sid always resumes, and reports its committed length. */
    rt = -1;
    CHECK(pn_prefix_cache_admit(c, "a", 2000, 0, &rt) == PN_SESSION_ADMIT_OK &&
          rt == 20, "existing sid 'a' admitted; restart_tokens = 20");

    /* Genuinely-new sid over the cap with NO idle TTL -> REFUSE. */
    rt = 99;
    CHECK(pn_prefix_cache_admit(c, "c", 2000, 0, &rt) ==
          PN_SESSION_ADMIT_AT_CAPACITY && rt == 0,
          "new sid 'c' over cap refused (no reaping; restart_tokens 0)");

    /* Idle TTL frees a slot: 'b' last active at t=1000; at t=2_000_000 with a
     * 60_000 ms TTL it is idle (idle 1_999_000 ms >= 60_000) and reaped, so 'c'
     * gets in. 'a' was touched more recently (t=... we touch it now to keep it
     * live) — verify only ONE slot is freed and the active one survives. */
    pn_prefix_cache_touch(c, "a", 1999000);   /* keep 'a' fresh */
    rt = -1;
    CHECK(pn_prefix_cache_admit(c, "c", 2000000, 60000, &rt) ==
          PN_SESSION_ADMIT_OK,
          "idle 'b' reaped past TTL -> new sid 'c' admitted");
    /* 'a' was kept fresh, so it must still resume (was not the reaped one). */
    CHECK(pn_prefix_cache_admit(c, "a", 2000000, 60000, NULL) ==
          PN_SESSION_ADMIT_OK, "fresh 'a' still resident after reaping 'b'");

    /* NULL cache / empty sid -> OK (uncached path, no-op). */
    CHECK(pn_prefix_cache_admit(NULL, "x", 0, 0, NULL) == PN_SESSION_ADMIT_OK,
          "NULL cache -> OK");
    CHECK(pn_prefix_cache_admit(c, "", 0, 0, NULL) == PN_SESSION_ADMIT_OK,
          "empty sid -> OK");

    pn_prefix_cache_free(c);
}

/* ============================================================
 * Test 7b — BATCHED session admission: the scheduler is the authority
 *
 * The 2026-11-05 bug (K80 bank): the batched path admitted against the prefix
 * cache, which holds nothing there, so max_sessions was never enforced, and a
 * new session took the least recently active idle slot — bs_prefill_begin then
 * erased that user's KV.  Now an instance with max_sessions N seats at most N
 * session families, the next new one is refused at once, and no slot another
 * family holds is ever handed out.
 * ============================================================ */

/* Session `sid` finished a turn on slot `idx`: 24 committed tokens, its KV rows
 * stamped with `seed`, the slot idle and recently active. */
static void seat_warm(pn_batch_sched *bs, int idx, const char *sid, float seed)
{
    uint32_t h[24];
    mk_prompt(h, SYS_A, 1000 + 100 * (int)seed, 24);
    publish_warm(&bs->slots[idx], sid, h, 24);
    fill_kv_rows((pn_qwen35moe_state *)bs->slots[idx].state->kv_cache, 0, 24,
                 seed);
    bs->slots[idx].status = BS_SLOT_IDLE;
    bs->slots[idx].last_active_ms = 1e15;   /* never idle past any TTL */
}

/* Does slot `idx` still hold `sid`'s warm turn exactly as seat_warm left it? */
static int still_warm(pn_batch_sched *bs, int idx, const char *sid, float seed)
{
    const bs_slot *sl = &bs->slots[idx];
    return strcmp(sl->sid, sid) == 0 && sl->hist_n == 24 &&
           check_kv_rows((const pn_qwen35moe_state *)sl->state->kv_cache, 0, 24,
                         seed);
}

/* The serve's non-blocking acquire (what a 0 ms queue wait does). */
static bs_acquire_result acquire_now(pn_batch_sched *bs, const char *sid,
                                     int *slot)
{
    *slot = -1;
    return bs_slot_acquire_or_wait(bs, 0, slot, NULL, sid);
}

static void test_batched_admission_one_seat(void)
{
    printf("test: batched admission, max_sessions 1\n");
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 1);
    bs_set_session_admission(bs, 1, 0);
    int s = -1;

    seat_warm(bs, 0, "user-a", 3.0f);
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_AT_CAPACITY && s < 0,
          "a second session is refused at once (429 model_at_session_capacity)");
    CHECK(still_warm(bs, 0, "user-a", 3.0f),
          "A keeps its slot, its sid and its KV");
    CHECK(acquire_now(bs, "user-a-act", &s) == BS_ACQUIRE_OK && s == 0,
          "A's own family (user-a-act) is admitted");
    bs_slot_release(bs, s);
    CHECK(bs_session_count(bs) == 1, "one session family holds a seat");
    bs_session_release(bs, "user-a");
    CHECK(bs_session_count(bs) == 0, "A's release frees its seat");
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_OK && s == 0,
          "after A's release B is admitted");
    bs_slot_release(bs, s);
    bs_destroy(bs);

    /* More decode slots than seats (max_concurrent > max_sessions): a cold
     * slot is no seat. */
    bs = mk_sched(&model, 2);
    bs_set_session_admission(bs, 1, 0);
    seat_warm(bs, 0, "user-a", 3.0f);
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_AT_CAPACITY,
          "2 slots, 1 seat: B is refused although slot 1 is cold");
    CHECK(bs->slots[1].status == BS_SLOT_IDLE, "...and slot 1 was not claimed");
    bs_session_release(bs, "user-a");
    CHECK(acquire_now(bs, "user-a2", &s) == BS_ACQUIRE_OK, "A2 claims a slot");
    CHECK(bs->slots[s].sid[0] == '\0' &&
          strcmp(bs->slots[s].claim_sid, "user-a2") == 0,
          "mid first turn: no warm sid yet, the claim names A2");
    int s2 = -1;
    CHECK(acquire_now(bs, "user-b", &s2) == BS_ACQUIRE_AT_CAPACITY,
          "an in-flight first turn holds its seat: B refused");
    bs_slot_release(bs, s);
    bs_destroy(bs);
}

static void test_batched_admission_four_seats(void)
{
    printf("test: batched admission, max_sessions 4 (the default: one per slot)\n");
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 4);   /* no setter: max = slots */
    static const char *const sids[4] = { "t-1", "t-2", "t-3", "t-4" };
    int s = -1;

    for (int i = 0; i < 4; i++)
        seat_warm(bs, i, sids[i], 1.0f + (float)i);
    CHECK(acquire_now(bs, "t-5", &s) == BS_ACQUIRE_AT_CAPACITY && s < 0,
          "a 5th session family is refused");
    int kept = 1;
    for (int i = 0; i < 4; i++)
        kept &= still_warm(bs, i, sids[i], 1.0f + (float)i);
    CHECK(kept, "the four keep their slots and KV");
    CHECK(bs_session_count(bs) == 4, "four families hold seats");
    CHECK(acquire_now(bs, "t-3", &s) == BS_ACQUIRE_OK && s == 2,
          "a seated session still resumes on its own slot");
    bs_slot_release(bs, s);
    CHECK(bs_slot_acquire_biased(bs, "t-5") == -1,
          "the non-blocking acquire never hands out another family's slot");
    bs_destroy(bs);
}

static void test_batched_admission_no_session_id(void)
{
    printf("test: batched admission, request without a session id\n");
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    int s = -1;

    seat_warm(bs, 0, "user-a", 3.0f);
    CHECK(acquire_now(bs, NULL, &s) == BS_ACQUIRE_OK && s == 1,
          "no session id: takes the cold slot");
    bs_slot_release(bs, s);
    seat_warm(bs, 1, "user-b", 4.0f);
    CHECK(acquire_now(bs, NULL, &s) == BS_ACQUIRE_AT_CAPACITY && s < 0,
          "no session id and no cold slot: refused");
    CHECK(still_warm(bs, 0, "user-a", 3.0f) && still_warm(bs, 1, "user-b", 4.0f),
          "...and neither session lost its KV");
    CHECK(bs_slot_acquire(bs) == -1, "bs_slot_acquire: no cold slot, none");
    bs_destroy(bs);
}

/* Watch hook for the reaper test: the base id `ctx` (a client's watch) covers
 * its own id and every "<base>-<more>". */
static int watched_base(const char *sid, void *ctx)
{
    const char *base = ctx;
    size_t n = strlen(base);
    return n > 0 && strncmp(sid, base, n) == 0 &&
           (sid[n] == '\0' || sid[n] == '-');
}

static void test_batched_admission_idle_reaper(void)
{
    printf("test: batched admission, idle reaper for unwatched seats\n");
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 1);
    char watch[32] = "user-a";
    int s = -1;

    bs_set_session_admission(bs, 1, 1000);   /* 1 s idle TTL */
    bs_set_watch_hook(bs, watched_base, watch);
    seat_warm(bs, 0, "user-a-act", 3.0f);
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_AT_CAPACITY,
          "a recently active seat is not reaped");
    bs->slots[0].last_active_ms = 1.0;        /* idle far past the TTL */
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_AT_CAPACITY,
          "a watched idle seat is not reaped (its watch owns the release)");
    CHECK(still_warm(bs, 0, "user-a-act", 3.0f), "...and keeps its KV");
    watch[0] = '\0';                          /* the client died, no watch */
    CHECK(acquire_now(bs, "user-b", &s) == BS_ACQUIRE_OK && s == 0,
          "an unwatched seat idle past the TTL is reaped; B is admitted");
    CHECK(bs->slots[0].sid[0] == '\0', "the reaped session's warm state is gone");
    bs_slot_release(bs, s);
    bs_destroy(bs);
}

static void test_batched_admission_restore(void)
{
    printf("test: batched admission, session restore\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 1);
    pn_qwen35moe_state *src = make_state();
    uint32_t toks[8];

    bs_set_session_admission(bs, 1, 0);
    fill_ssm(src, 7.0f);
    fill_kv_rows(src, 0, 8, 7.0f);
    void *snap = pn_qwen35moe_state_snapshot(src, 8);
    mk_prompt(toks, SYS_A, 1000, 8);
    seat_warm(bs, 0, "user-a", 3.0f);
    CHECK(bs_session_restore(bs, "user-b", snap, toks, 8) == BS_SESSION_BUSY,
          "restoring a new session at capacity is refused");
    CHECK(still_warm(bs, 0, "user-a", 3.0f), "...and does not overwrite A");
    bs_session_release(bs, "user-a");
    CHECK(bs_session_restore(bs, "user-b", snap, toks, 8) == BS_SESSION_OK &&
          strcmp(bs->slots[0].sid, "user-b") == 0,
          "with A released the restore seats B");
    pn_qwen35moe_state_snapshot_free(snap);
    free_state(src);
    bs_destroy(bs);
}

/* The single-user path counts families too: a side id shares its seat, and a
 * full pool gives up a family's second conversation, never another family's
 * only one. */
static void test_single_user_family_admission(void)
{
    printf("test: single-user cache, max_sessions counts session families\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_prefix_cache *c = pn_prefix_cache_new(&model, 3);
    uint32_t p[20], h[24], w2[26], hw[24];
    int delta = -1;

    mk_prompt(p, SYS_A, 1000, 20);
    pn_state *st = su_turn(c, "u", p, 20, 1.0f, &delta);
    su_commit(c, st, "u", p, 20, h, 1);
    mk_prompt(p, SYS_B, 2000, 20);
    st = su_turn(c, "u-side", p, 20, 2.0f, &delta);
    su_commit(c, st, "u-side", p, 20, h, 2);
    mk_prompt(p, SYS_A, 3000, 20);
    pn_state *sw = su_turn(c, "w", p, 20, 3.0f, &delta);
    int hw_n = su_commit(c, sw, "w", p, 20, hw, 3);
    CHECK(pn_prefix_cache_session_count(c) == 2,
          "u, u-side and w hold 3 entries but 2 session families");
    CHECK(pn_prefix_cache_admit(c, "v", 1000, 0, NULL) == PN_SESSION_ADMIT_OK,
          "2 families < max_sessions 3: new session v is admitted");
    mk_prompt(p, SYS_B, 4000, 20);
    st = su_turn(c, "v", p, 20, 4.0f, &delta);
    su_commit(c, st, "v", p, 20, h, 4);
    int n = convo_append(w2, hw, hw_n, 3);
    st = su_turn(c, "w", w2, n, 0.0f, &delta);
    CHECK(st == sw && delta == hw_n,
          "v took one of u's two entries; w's only one survived");
    CHECK(pn_prefix_cache_admit(c, "x", 1000, 0, NULL) ==
          PN_SESSION_ADMIT_AT_CAPACITY, "3 families: a 4th is refused");
    CHECK(pn_prefix_cache_admit(c, "u-side-x", 1000, 0, NULL) ==
          PN_SESSION_ADMIT_OK, "a new id of a seated family is admitted");
    pn_prefix_cache_free(c);
}

/* ============================================================
 * Test 9 — TURN-START CHECKPOINT (rung 1c)
 *
 * The three cases that used to fall to the system-prefix rung and re-prefill
 * the whole conversation: a reasoning-only turn (nothing projected, Argus
 * drops the turn), an interrupted turn (a partial reply the client never
 * resends) and an aborted prefill (the slot was reset). Each must now resume
 * at the turn start. Card work is CUDA-only; these are the ladder's decisions
 * and bookkeeping (the non-CUDA arm only moves state->pos).
 * ============================================================ */

/* A 20-token turn-1 prompt: 8 system + 9 conversation + a 3-token assistant
 * opener (turn_start 17). */
#define TC_N1   20
#define TC_CUT1 17

/* Turn 1 on slot `slot`: cold prefill to the cut, checkpoint, finish the
 * prompt, decode `n_gen` tokens with NO projection (a reasoning-only or
 * interrupted reply), publish the warm history. */
static void tc_turn1(pn_batch_sched *bs, int slot, const char *sid,
                     const uint32_t *p1)
{
    int delta = -1, cap = -1;
    bs_prefill_resume_sys(bs, slot, sid, p1, TC_N1, 8, &delta, &cap);
    CHECK(delta == 0, "turn 1: cold full prefill");
    bs_slot *sl = &bs->slots[slot];
    sl->state->pos = TC_CUT1;
    CHECK(bs_turn_ckpt_capture(bs, slot, sid, p1, TC_CUT1) == 0,
          "turn 1: checkpoint saved at the turn start");
    sl->state->pos = TC_N1;
    uint32_t gen[5] = { 900, 901, 902, 903, 904 };   /* reasoning only */
    seed_committed_log(sl, p1, TC_N1, gen, 5);
    bs_slot_note_visible(bs, slot, p1, TC_N1, 2, NULL, 0, NULL, 0);
    publish_vis(sl);
    publish_warm(sl, sid, sl->ct, TC_N1 + 5);
    bs_prefill_abort(bs, slot);
}

/* The next request after a dropped reply: turn 1's messages + a 6-token note
 * + the 3-token opener. */
static void tc_next(uint32_t *out, const uint32_t *p1)
{
    memcpy(out, p1, TC_CUT1 * sizeof(uint32_t));
    for (int i = TC_CUT1; i < TC_CUT1 + 9; i++)
        out[i] = (uint32_t)(5000 + i);
}

static void test_turn_checkpoint_resume(void)
{
    printf("test: turn-start checkpoint (rung 1c)\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t p1[TC_N1], p2[TC_CUT1 + 9];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    tc_next(p2, p1);
    int delta = -1, cap = -1;

    /* Reasoning-only turn, then the rescue note: resumes at the turn start. */
    int s = bs_slot_acquire_biased(bs, "A");
    tc_turn1(bs, s, "A", p1);
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p2, TC_CUT1 + 9, 8, &delta, &cap);
    bs_slot *sl = &bs->slots[s];
    CHECK(delta == TC_CUT1, "dropped reply: rung 1c resumes at the turn start");
    CHECK((int)sl->state->pos == TC_CUT1 && sl->hist_n == TC_CUT1,
          "the state and history are back at the checkpoint");
    CHECK(sl->ct_keep == TC_CUT1 && sl->ct_prompt_from == TC_CUT1,
          "committed-log seeding joins the request at the cut");
    CHECK(cap == 0, "no system-prefix capture requested");
    bs_prefill_abort(bs, s);

    /* NEGATIVE CONTROL: an earlier message changed — no checkpoint resume. */
    uint32_t p3[TC_CUT1 + 9];
    memcpy(p3, p2, sizeof(p3));
    p3[12] ^= 1u;
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p3, TC_CUT1 + 9, 8, &delta, &cap);
    CHECK(delta == 0, "a rewritten earlier message re-prefills (no rung 1c)");
    CHECK(!bs->slots[s].ckpt.valid, "the full reset dropped the checkpoint");
    bs_prefill_abort(bs, s);
    bs_destroy(bs);
}

static void test_turn_checkpoint_think_elided_kv(void)
{
    printf("test: turn-start checkpoint over a think-elided KV\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    /* The slot's KV holds 4 more tokens than the client sends (an earlier
     * think block rung 1b kept): hist = client[0,12) + think(4) + ... */
    uint32_t client[TC_N1];
    mk_prompt(client, SYS_A, 1000, TC_N1);
    uint32_t hist[16];
    memcpy(hist, client, 12 * sizeof(uint32_t));
    for (int i = 12; i < 16; i++) hist[i] = (uint32_t)(7000 + i);
    int s = bs_slot_acquire_biased(bs, "A");
    bs_slot *sl = &bs->slots[s];
    publish_warm(sl, "A", hist, 16);
    /* This turn resumed by projection at client 12: KV = hist ++ client[12..). */
    sl->ct_keep = 16;
    sl->ct_prompt_from = 12;
    sl->state->pos = 16 + (TC_CUT1 - 12);
    CHECK(bs_turn_ckpt_capture(bs, s, "A", client, TC_CUT1) == 0,
          "checkpoint saved over the elided KV");
    CHECK(sl->ckpt.kv_n == 21 && sl->ckpt.client_n == TC_CUT1,
          "it records the KV position (21) and the client position (17)");
    CHECK(sl->ckpt.kv[13] == 7013 && sl->ckpt.kv[16] == client[12],
          "its KV tokens are hist[0,16) then the request from 12");
    /* A wrong position is refused, and leaves the checkpoint alone. */
    sl->state->pos = 20;
    CHECK(bs_turn_ckpt_capture(bs, s, "A", client, TC_CUT1) == -1,
          "capture refuses a state that is not at the cut");
    CHECK(sl->ckpt.valid && sl->ckpt.kv_n == 21, "the old checkpoint survives");
    bs_slot_release(bs, s);   /* no GPU turn was taken in this part */

    uint32_t next[TC_CUT1 + 9];
    tc_next(next, client);
    int delta = -1, cap = -1;
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", next, TC_CUT1 + 9, 8, &delta, &cap);
    CHECK(delta == TC_CUT1 && (int)sl->state->pos == 21,
          "resume: prefill from client 17 at KV position 21");
    CHECK(sl->ct_keep == 21 && sl->ct_prompt_from == TC_CUT1 &&
          sl->hist_n == 21 && sl->hist[14] == 7014,
          "history is the checkpoint's KV tokens");
    bs_prefill_abort(bs, s);
    bs_destroy(bs);
}

static void test_turn_checkpoint_prefill_abort(void)
{
    printf("test: an aborted prefill keeps the turn checkpoint\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t p1[TC_N1], p2[TC_CUT1 + 9];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    tc_next(p2, p1);
    int delta = -1, cap = -1;
    int s = bs_slot_acquire_biased(bs, "A");
    tc_turn1(bs, s, "A", p1);

    /* Turn 2 starts, resumes at the checkpoint, and is aborted mid-prefill. */
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p2, TC_CUT1 + 9, 8, &delta, &cap);
    bs->slots[s].state->pos = TC_CUT1 + 4;   /* stopped part-way */
    bs_prefill_abort_keep(bs, s, "A", p2, TC_CUT1 + 9, 8);
    bs_slot *sl = &bs->slots[s];
    CHECK(sl->status == BS_SLOT_IDLE, "the slot is released");
    CHECK(sl->ckpt.valid && strcmp(sl->sid, "A") == 0,
          "it keeps the session and the checkpoint");
    CHECK(sl->partial.valid && sl->partial.client_n == TC_CUT1 + 4,
          "and the 4 prompt tokens prefilled past the checkpoint");
    CHECK(sl->hist_n == 0, "the half-written state has no history");

    /* The retry (same request) is routed back and resumes where it stopped. */
    s = bs_slot_acquire_biased(bs, "A");
    CHECK(s == 0, "the retry lands on the same slot");
    bs_prefill_resume_sys(bs, s, "A", p2, TC_CUT1 + 9, 8, &delta, &cap);
    CHECK(delta == TC_CUT1 + 4 && (int)bs->slots[s].state->pos == TC_CUT1 + 4,
          "the retry resumes where the abort stopped, not cold");
    CHECK(!bs->slots[s].partial.valid && bs->slots[s].ckpt.valid,
          "the kept prefix is used up; the checkpoint stays");
    /* Aborted again; this time the retry differs inside the kept prefix
     * (but after the turn start): back to the checkpoint. */
    bs->slots[s].state->pos = TC_CUT1 + 6;
    bs_prefill_abort_keep(bs, s, "A", p2, TC_CUT1 + 9, 8);
    uint32_t p2b[TC_CUT1 + 9];
    memcpy(p2b, p2, sizeof(p2b));
    p2b[TC_CUT1 + 1] ^= 1u;
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p2b, TC_CUT1 + 9, 8, &delta, &cap);
    CHECK(delta == TC_CUT1 && (int)bs->slots[s].state->pos == TC_CUT1,
          "a retry that differs past the turn start resumes at the checkpoint");
    CHECK(!bs->slots[s].partial.valid, "the stale kept prefix is dropped");
    bs_prefill_abort(bs, s);

    /* Without a checkpoint, an abort inside the system prefix still resets. */
    int t = bs_slot_acquire_biased(bs, "B");
    uint32_t pb[TC_N1];
    mk_prompt(pb, SYS_B, 3000, TC_N1);
    bs_prefill_resume_sys(bs, t, "B", pb, TC_N1, 8, &delta, &cap);
    bs->slots[t].state->pos = 5;
    bs_prefill_abort_keep(bs, t, "B", pb, TC_N1, 8);
    CHECK(bs->slots[t].sid[0] == '\0' && !bs->slots[t].ckpt.valid &&
          !bs->slots[t].partial.valid,
          "abort below the system prefix, no checkpoint: the slot resets");

    /* A session release drops the checkpoint with the rest. */
    bs_session_release(bs, "A");
    CHECK(!bs->slots[0].ckpt.valid, "session release drops the checkpoint");
    bs_destroy(bs);
}

/* A session's FIRST prefill is aborted (no checkpoint exists yet): the slot
 * keeps the computed prefix and the resent request resumes there. */
static void test_first_prefill_abort_keeps_prefix(void)
{
    printf("test: an aborted first prefill keeps its prefix\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t p1[TC_N1];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    int delta = -1, cap = -1;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    CHECK(delta == 0, "first turn: cold");
    bs_slot *sl = &bs->slots[s];
    sl->state->pos = 12;                     /* past the system prefix */
    bs_prefill_abort_keep(bs, s, "A", p1, TC_N1, 8);
    CHECK(sl->status == BS_SLOT_IDLE && strcmp(sl->sid, "A") == 0 &&
          !sl->ckpt.valid && sl->partial.valid && sl->partial.client_n == 12,
          "no checkpoint, but the slot keeps the session and 12 tokens");

    /* NEGATIVE CONTROL: a resend that differs inside those 12 tokens. */
    uint32_t px[TC_N1];
    memcpy(px, p1, sizeof(px));
    px[10] ^= 1u;
    int r = bs_slot_reroute(bs, bs_slot_acquire_biased(bs, "A"), "A", px,
                            TC_N1, 8);
    s = r >= 0 ? r : 0;
    bs_prefill_resume_sys(bs, s, "A", px, TC_N1, 8, &delta, &cap);
    CHECK(delta != 12, "a changed prefix does not resume at the kept prefix");
    CHECK(!bs->slots[s].partial.valid, "and drops it");
    bs_prefill_abort(bs, s);
    bs_destroy(bs);

    /* The resend of the same request (plus a new message) resumes at 12. */
    bs = mk_sched(&model, 2);
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    bs->slots[s].state->pos = 12;
    bs_prefill_abort_keep(bs, s, "A", p1, TC_N1, 8);
    uint32_t p2[TC_N1 + 4];
    memcpy(p2, p1, TC_CUT1 * sizeof(uint32_t));
    for (int i = TC_CUT1; i < TC_N1 + 4; i++)
        p2[i] = (uint32_t)(6000 + i);
    s = bs_slot_acquire_biased(bs, "A");
    r = bs_slot_reroute(bs, s, "A", p2, TC_N1 + 4, 8);
    CHECK(r == s, "the resend is routed to the slot that kept the prefix");
    bs_prefill_resume_sys(bs, s, "A", p2, TC_N1 + 4, 8, &delta, &cap);
    sl = &bs->slots[s];
    CHECK(delta == 12 && (int)sl->state->pos == 12,
          "the resend prefills only from token 12");
    CHECK(sl->hist_n == 12 && sl->hist[11] == p1[11] &&
          sl->ct_keep == 12 && sl->ct_prompt_from == 12,
          "history and committed-log seeding describe the kept prefix");
    CHECK(cap == 0, "no system-prefix capture requested");
    bs_prefill_abort(bs, s);

    /* A session release drops a kept prefix. */
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    bs->slots[s].state->pos = 13;
    bs_prefill_abort_keep(bs, s, "A", p1, TC_N1, 8);
    bs_session_release(bs, "A");
    CHECK(!bs->slots[s].partial.valid, "session release drops the kept prefix");
    bs_destroy(bs);
}

static void test_turn_checkpoint_reroute(void)
{
    printf("test: reroute prefers the slot holding the checkpoint\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t p1[TC_N1], p2[TC_CUT1 + 9];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    tc_next(p2, p1);
    /* Slot 1 holds the checkpoint; the session's sid points nowhere else. */
    bs->slots[0].status = BS_SLOT_PREFILL;   /* keep slot 0 out of the claim */
    int s = bs_slot_acquire_biased(bs, "A");
    CHECK(s == 1, "turn 1 runs on slot 1");
    tc_turn1(bs, s, "A", p1);
    bs->slots[0].status = BS_SLOT_IDLE;
    /* Hide slot 1's warm sid so only the checkpoint can attract the turn. */
    bs->slots[1].sid[0] = '\0';
    int idx = bs_slot_acquire_biased(bs, "A");
    int r = bs_slot_reroute(bs, idx, "A", p2, TC_CUT1 + 9, 8);
    CHECK(r == 1, "the reroute picks the slot whose checkpoint fits");
    pthread_mutex_lock(&bs->mu);
    bs->slots[0].status = BS_SLOT_IDLE;
    bs->slots[1].status = BS_SLOT_IDLE;
    pthread_mutex_unlock(&bs->mu);
    bs_destroy(bs);
}

/* ============================================================
 * SAME-SESSION RESEND — a client hangs up mid-prefill and resends before the
 * old request notices (K80 bank, 2026-11-11: `prefill aborted at 25326 of
 * 25333 ... keeping that prefix` on slot 0, then the resend re-prefilled 22044
 * tokens on slot 2). The resend must wait for the slot still prefilling its
 * session and resume the prefix kept there, even with free slots to spare.
 *
 * Red: drop the hold in bs_slot_acquire_or_wait and the resend claims slot 1
 * at once ("waits for its own slot" and "resumes at 12" fail).
 * ============================================================ */

typedef struct {
    pn_batch_sched *bs;
    int wait_ms;
    int slot;
    volatile int done;
} resend_arg;

static void *resend_thread(void *p)
{
    resend_arg *a = p;
    a->slot = -1;
    bs_slot_acquire_or_wait(a->bs, a->wait_ms, &a->slot, NULL, "A");
    a->done = 1;
    return NULL;
}

/* Slot 0 prefills session A's first request up to 12 tokens. */
static pn_batch_sched *resend_setup(pn_model *model, const uint32_t *p1)
{
    pn_batch_sched *bs = mk_sched(model, 4);
    bs_set_session_admission(bs, 1, 0);   /* room to spare: an extra is allowed */
    bs->queue_cap = BS_QUEUE_DEPTH_DEFAULT;   /* as bs_create sets it */
    int s = -1, delta = -1, cap = -1;
    bs_slot_acquire_or_wait(bs, 0, &s, NULL, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    bs->slots[s].state->pos = 12;
    return bs;
}

static void test_same_session_resend_waits(void)
{
    printf("test: a same-session resend waits for its prefilling slot\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    uint32_t p1[TC_N1], p2[TC_N1 + 4];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    memcpy(p2, p1, sizeof(p1));
    for (int i = TC_N1; i < TC_N1 + 4; i++)
        p2[i] = (uint32_t)(7000 + i);
    int delta = -1, cap = -1;

    /* ── The resend arrives while slot 0 is still prefilling A. ── */
    unsetenv("PN_SAME_SESSION_WAIT_MS");
    pn_batch_sched *bs = resend_setup(&model, p1);
    resend_arg a = { bs, 5000, -1, 0 };
    pthread_t th;
    pthread_create(&th, NULL, resend_thread, &a);
    usleep(150 * 1000);
    CHECK(!a.done, "the resend waits for its own slot (slots 1-3 are free)");
    bs_prefill_abort_keep(bs, 0, "A", p1, TC_N1, 8);   /* old request notices */
    pthread_join(th, NULL);
    CHECK(a.slot == 0, "the resend lands on the slot that kept the prefix");
    int s = a.slot >= 0 ? bs_slot_reroute(bs, a.slot, "A", p2, TC_N1 + 4, 8) : 0;
    bs_prefill_resume_sys(bs, s, "A", p2, TC_N1 + 4, 8, &delta, &cap);
    CHECK(delta == 12, "the resend resumes at 12, not cold");
    bs_prefill_abort(bs, s);
    bs_destroy(bs);

    /* ── Forced miss (red arm): the wait off, the resend runs elsewhere and
     * re-prefills — the harness can fail. ── */
    setenv("PN_SAME_SESSION_WAIT_MS", "0", 1);
    bs = resend_setup(&model, p1);
    int s2 = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 5000, &s2, NULL, "A") == BS_ACQUIRE_OK &&
          s2 > 0, "wait off: the resend takes a free slot at once");
    bs_prefill_abort_keep(bs, 0, "A", p1, TC_N1, 8);
    if (s2 >= 0) {
        bs_prefill_resume_sys(bs, s2, "A", p2, TC_N1 + 4, 8, &delta, &cap);
        CHECK(delta < 12, "wait off: the resend re-prefills (the miss)");
        bs_prefill_abort(bs, s2);
    }
    bs_destroy(bs);

    /* ── Bounded: an old request that never ends holds the resend only for
     * PN_SAME_SESSION_WAIT_MS, then it claims as before. ── */
    setenv("PN_SAME_SESSION_WAIT_MS", "200", 1);
    bs = resend_setup(&model, p1);
    s2 = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 5000, &s2, NULL, "A") == BS_ACQUIRE_OK &&
          s2 > 0, "after the bounded wait the resend claims a free slot");
    if (s2 >= 0)
        bs_slot_release(bs, s2);
    bs_prefill_abort(bs, 0);
    bs_destroy(bs);
    unsetenv("PN_SAME_SESSION_WAIT_MS");

    /* ── Another session never waits on A. ── */
    bs = resend_setup(&model, p1);
    bs_set_session_admission(bs, 4, 0);
    s2 = -1;
    CHECK(bs_slot_acquire_or_wait(bs, 5000, &s2, NULL, "B") == BS_ACQUIRE_OK &&
          s2 > 0, "session B is seated at once while A prefills");
    if (s2 >= 0)
        bs_slot_release(bs, s2);
    bs_prefill_abort(bs, 0);
    bs_destroy(bs);
}

/* An abort exactly at the system boundary keeps nothing (the SYS1 snapshot
 * resumes there; the live serve logged `prefill aborted at 3320 of 15871`
 * then `partial-prefill resume refused ... the block no longer holds it`). */
static void test_abort_at_sys_boundary_keeps_nothing(void)
{
    printf("test: an abort at the system boundary keeps no prefix\n");
    g_make_q4 = 0;
    pn_model model = {0};
    model.config.arch = PN_ARCH_QWEN35MOE;
    pn_batch_sched *bs = mk_sched(&model, 2);
    uint32_t p1[TC_N1];
    mk_prompt(p1, SYS_A, 1000, TC_N1);
    int delta = -1, cap = -1;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    bs->slots[s].state->pos = 8;   /* exactly the system prefix */
    bs_prefill_abort_keep(bs, s, "A", p1, TC_N1, 8);
    CHECK(!bs->slots[s].partial.valid, "nothing kept at the boundary");
    /* One token past it is kept as before. */
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_sys(bs, s, "A", p1, TC_N1, 8, &delta, &cap);
    bs->slots[s].state->pos = 9;
    bs_prefill_abort_keep(bs, s, "A", p1, TC_N1, 8);
    CHECK(bs->slots[s].partial.valid && bs->slots[s].partial.client_n == 9,
          "one token past the boundary is kept");
    bs_destroy(bs);
}

int main(void)
{
    /* Rung 1c is gated; the gate is read once, so set it before any test.
     * Nothing else here captures a checkpoint, so the other tests run the
     * ladder exactly as before. */
    setenv("QWEN35_TURN_CHECKPOINT", "1", 1);
    printf("=== test_prefix_cache_sys (SYS1) ===\n");
    test_snapshot_roundtrip_fp32();
    test_snapshot_roundtrip_q4();
    test_snap_serialize_roundtrip(0);
    test_snap_serialize_roundtrip(1);
    test_acquire_sys_flow();
    test_unsupported_arch();
    test_lcp_rewind();
    test_batched_sys_flow();
    test_no_duplicate_slots();
    test_session_substitution();
    test_two_convos_concurrent_title();
    test_two_convos_interleaved();
    test_single_user_two_convos_interleaved();
    test_single_user_side_convo_admission();
    test_single_user_export_import_two_convos();
    test_think_elided_projection();
    test_reasoning_bearing_resume();
    test_reasoning_convention_switch();
    test_cached_tokens_tracks_resume_rungs();
    test_session_admission();
    test_batched_admission_one_seat();
    test_batched_admission_four_seats();
    test_batched_admission_no_session_id();
    test_batched_admission_idle_reaper();
    test_batched_admission_restore();
    test_single_user_family_admission();
    test_turn_checkpoint_resume();
    test_turn_checkpoint_think_elided_kv();
    test_turn_checkpoint_prefill_abort();
    test_first_prefill_abort_keeps_prefix();
    test_turn_checkpoint_reroute();
    test_same_session_resend_waits();
    test_abort_at_sys_boundary_keeps_nothing();
    if (failures) {
        printf("=== %d FAILURE(S) ===\n", failures);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
