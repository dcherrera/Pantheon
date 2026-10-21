/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_forward.c
 * @brief Qwen3.5-MoE hybrid forward-pass assembly (A6). Wires the four
 *        isolation-verified kernels (loader A2, attention A3, SSM A4, MoE A5)
 *        into an end-to-end single-token forward with dual state, plus the
 *        dual-state lifecycle (new/free/reset) the engine hooks call.
 *
 * @status A6 complete: embed -> per-layer (attn_norm -> SSM|attn block ->
 *         +residual -> post_attention_norm -> shared-MoE -> +residual) ->
 *         output_norm -> LM head. Runs end-to-end; numeric correctness vs the
 *         oracle is A8.
 * @issues None
 * @todo
 *   - [ ] A8: validate logits vs the llama.cpp oracle (the (1+w) norm watch-
 *         list items in docs/architecture/qwen35moe.md may surface here).
 */

/* 1. This module's own header */
#include "qwen35moe.h"

/* 2. Standard library */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3. System / platform headers */
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

/* 4. Project headers */
#include "../accel/simd.h"
#include "../core/qweight.h"
#include "../memory/arena.h"
#include "qwen35moe_internal.h"
#include "qwen35moe_kv_q4.h"

#include "../core/mem/pn_lazy_zero.h"
#include "../core/mem/pn_memwatch.h"
#include "../core/mem/pn_ladder.h"

/* QWEN35_KV_Q4 gate: controls Q4_0 KV-cache quantization (~4-7x smaller
 * VRAM/host, 256K-context lever). Quality validated — PPL delta 0.024%.
 *
 * Default (B1): ON. Opt-out with QWEN35_KV_Q4=0 to restore fp32 KV for
 * diagnostic or A/B work. The old opt-in (QWEN35_KV_Q4=1) still forces ON
 * explicitly and is unchanged. Read once. */
int pn_qwen35moe_kv_q4_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_KV_Q4");
        /* Default ON; only explicit "0" disables. */
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* QWEN35_HOST_KV_MIRROR gate (Phase 10b / INV3). Default LAZY (0): the
 * batched slots path stops maintaining the per-token host KV mirror — the
 * GPU slot ring is authoritative while a session is bound, rebinds are
 * seeded device-to-device, and the mirror is materialized on demand
 * (export, sys-prefix capture, prefill yield, host-softmax fallback). Set
 * QWEN35_HOST_KV_MIRROR=1 to restore the legacy eager mirror (every decode
 * token D2H-appended to host RAM). Read once. */
int pn_qwen35moe_host_kv_mirror_eager(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_HOST_KV_MIRROR");
        v = (env && env[0] == '1') ? 1 : 0;
    }
    return v;
}

/* Phase 10b: global content-generation counter for pn_qwen35moe_kv (see
 * content_gen in qwen35moe_internal.h). Seeding each kv from a process-wide
 * counter guarantees a recycled buffer address can never alias a stale GPU
 * ring watermark. */
unsigned pn_qwen35moe_kv_next_content_gen(void)
{
    static unsigned counter;
    return __atomic_add_fetch(&counter, 1u, __ATOMIC_RELAXED);
}

/* Per-token scratch arena size. The hot consumers are the attention block
 * (an [n_head * max_seq] scores buffer) and the SSM block (several conv_dim /
 * val_dim buffers). 16 MB leaves generous headroom for every block's working
 * set with the mini's dims, and is reset to empty before each block. */
#define PN_QWEN35MOE_ARENA_BYTES (16u * 1024u * 1024u)

/* KV-cache window per attention layer (default PN_QWEN35MOE_MAX_SEQ, matching
 * rwkvx's decode budget). QWEN35_MAX_SEQ overrides for long-context runs, and a
 * per-model ctx_cap overrides that. Memory scales linearly: n_head_kv * max_seq
 * * head_dim * 8 bytes per attention layer on the host, the same again on the
 * die for the flash-attention cache. Both PN_QWEN35MOE_MAX_SEQ (default) and
 * PN_QWEN35MOE_MAX_SEQ_CAP (ceiling) live in qwen35moe.h so the engine admission
 * path and the CUDA admit gate share the exact numbers the ring is sized to. */

/* The configured KV window (env-overridable, parsed once). */
static int qwen35moe_clamp_max_seq(int v)
{
    if (v < 16)
        return 16;
    if (v > PN_QWEN35MOE_MAX_SEQ_CAP)
        return PN_QWEN35MOE_MAX_SEQ_CAP;
    return v;
}

static int qwen35moe_max_seq(void)
{
    static int v = -1;
    if (v < 0) {
        v = PN_QWEN35MOE_MAX_SEQ;
        const char *env = getenv("QWEN35_MAX_SEQ");
        if (env) {
            int e = atoi(env);
            if (e > 0)
                v = qwen35moe_clamp_max_seq(e);
        }
    }
    return v;
}

/* KV Phase 1: the KV window this MODEL's ring is sized to. A per-model
 * cfg->ctx_cap (set at load) wins so ornith can run 40K without the global
 * QWEN35_MAX_SEQ moving saber's window; 0 falls back to the global/default. */
static int qwen35moe_max_seq_cfg(const pn_qwen35moe_config *cfg)
{
    if (cfg && cfg->ctx_cap > 0)
        return qwen35moe_clamp_max_seq(cfg->ctx_cap);
    return qwen35moe_max_seq();
}

int pn_qwen35moe_max_seq(const pn_qwen35moe_config *cfg)
{
    return qwen35moe_max_seq_cfg(cfg);
}

/* Add src into dst, element-wise: dst[i] += src[i]. */
static void qwen35moe_residual_add(float *dst, const float *src, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
}

/* Debug dump: first-3, last-3 + sum, matching llama-eval-callback's format so
 * activations can be bisected against the oracle. Gated by QWEN35_DEBUG. */
static int qwen35moe_dbg_enabled(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("QWEN35_DEBUG") ? 1 : 0;
    return v;
}

static void qwen35moe_dbg(const char *tag, int layer, const float *x, int n)
{
    if (!qwen35moe_dbg_enabled())
        return;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    fprintf(stderr, "PN %-22s L%-2d [%12.6f %12.6f %12.6f ... %12.6f %12.6f "
            "%12.6f] sum=%.6f\n",
            tag, layer, x[0], x[1], x[2], x[n - 3], x[n - 2], x[n - 1], sum);
}

/* ============================================================
 * Dual-state lifecycle
 * ============================================================ */

/* ============================================================
 * Lazily-committed host-state buffers (Phase 10b / INV3)
 *
 * The KV mirror ring and the SSM recurrent state are the two per-session
 * allocations that made host RAM scale with ctx_cap instead of usage: they
 * were calloc'd at full max_seq and fully committed by the first
 * pn_state_reset memset (~789 MiB/slot at ctx_cap 131072). They now come
 * from pn_lazy_zero_alloc (anonymous mmap), so committed pages track the
 * rows actually written, reset returns the pages to the OS, and free
 * bypasses glibc arena retention. The byte-size helpers below are shared by
 * init/free/reset so allocator and deallocator can never disagree.
 * ============================================================ */

/* Bytes of ONE of the kq/vq (q4) or k/v (fp32) host mirror rings. */
static size_t qwen35moe_kv_ring_bytes(const pn_qwen35moe_kv *kv)
{
    if (kv->q4)
        return (size_t)kv->n_head_kv * kv->max_seq *
               (size_t)pn_kvq4_row_bytes(kv->head_dim);
    return (size_t)kv->n_head_kv * kv->max_seq * kv->head_dim *
           sizeof(float);
}

static size_t qwen35moe_ssm_conv_bytes(const pn_qwen35moe_ssm_state *st)
{
    return (size_t)st->conv_minus_1 * st->conv_dim * sizeof(float);
}

static size_t qwen35moe_ssm_state_bytes(const pn_qwen35moe_ssm_state *st)
{
    return (size_t)st->state_size * st->state_size * st->n_v_heads *
           sizeof(float);
}

/* Allocate one attention layer's KV cache, zero-primed. Returns 0 on success.
 * Under QWEN35_KV_Q4 the cache is stored as ggml-Q4_0 rows (~4-7x smaller);
 * otherwise fp32 (the A/B baseline). */
static int qwen35moe_kv_init(pn_qwen35moe_kv *kv, const pn_qwen35moe_config *cfg)
{
    kv->n_head_kv = cfg->n_head_kv;
    kv->head_dim = cfg->head_dim;
    kv->max_seq = qwen35moe_max_seq_cfg(cfg);
    kv->q4 = pn_qwen35moe_kv_q4_on();

    /* Rung 4a of the reactive ladder (MemoryAware.md §7): a KV cache allocated
     * while the box is under memory pressure defaults to the ~4-7x-smaller Q4
     * format even if the operator did not set QWEN35_KV_Q4 — the cheapest VRAM
     * win, engaged from WATCH up. KV format is fixed at allocation, so this acts
     * at init time (not retroactively on a live fp32 cache). At level OK the
     * plan mirrors the config, so an unpressured init is unchanged. */
    if (!kv->q4 && pn_memwatch_running()) {
        pn_mem_state st;
        pn_memwatch_read(&st);
        pn_ladder_plan lp;
        pn_ladder_plan_from_state(&st, 1, 0, false, &lp);
        if (lp.force_kv_q4) {
            kv->q4 = 1;
            fprintf(stderr, "[ladder] %s -> forcing Q4 KV cache at init (%s)\n",
                    pn_mem_level_name(st.level), lp.note);
        }
    }
    kv->k = kv->v = NULL;
    kv->kq = kv->vq = NULL;
    kv->k_sink = kv->v_sink = NULL;
    kv->k_win = kv->v_win = NULL;
    kv->content_gen = pn_qwen35moe_kv_next_content_gen();
    size_t ring_bytes = qwen35moe_kv_ring_bytes(kv);
    if (kv->q4) {
        kv->kq = pn_lazy_zero_alloc(ring_bytes);
        kv->vq = pn_lazy_zero_alloc(ring_bytes);
        /* fp16 sink + recent-window shadow (kept at fp16 to stop argmax flips) */
        size_t sink_n = (size_t)kv->n_head_kv * PN_KVQ4_SINKMAX * kv->head_dim;
        size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
        kv->k_sink = calloc(sink_n, sizeof(uint16_t));
        kv->v_sink = calloc(sink_n, sizeof(uint16_t));
        kv->k_win = calloc(win_n, sizeof(uint16_t));
        kv->v_win = calloc(win_n, sizeof(uint16_t));
        return (kv->kq && kv->vq && kv->k_sink && kv->v_sink &&
                kv->k_win && kv->v_win) ? 0 : -1;
    }
    kv->k = pn_lazy_zero_alloc(ring_bytes);
    kv->v = pn_lazy_zero_alloc(ring_bytes);
    return (kv->k && kv->v) ? 0 : -1;
}

/* Allocate one SSM layer's recurrent state, zero-primed (the SSM kernel assumes
 * a zero conv_state + ssm_state at sequence start). Returns 0 on success. */
static int qwen35moe_ssm_init(pn_qwen35moe_ssm_state *st,
                              const pn_qwen35moe_config *cfg)
{
    st->conv_dim = cfg->conv_dim;
    st->conv_minus_1 = cfg->ssm_conv - 1;
    st->state_size = cfg->ssm_state;
    st->n_v_heads = cfg->ssm_dt_rank;
    st->conv_state = pn_lazy_zero_alloc(qwen35moe_ssm_conv_bytes(st));
    st->ssm_state = pn_lazy_zero_alloc(qwen35moe_ssm_state_bytes(st));
    return (st->conv_state && st->ssm_state) ? 0 : -1;
}

void *pn_qwen35moe_state_new(void *weights)
{
    pn_qwen35moe_model *m = (pn_qwen35moe_model *)weights;
    if (!m)
        return NULL;

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    pn_qwen35moe_state *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;

    s->n_ssm = pn_qwen35moe_model_n_ssm(m);
    s->n_attn = pn_qwen35moe_model_n_attn(m);
    s->n_embd = cfg->n_embd;
    s->n_vocab = cfg->n_vocab;

    s->ssm = s->n_ssm ? calloc((size_t)s->n_ssm, sizeof(*s->ssm)) : NULL;
    s->kv = s->n_attn ? calloc((size_t)s->n_attn, sizeof(*s->kv)) : NULL;
    if ((s->n_ssm && !s->ssm) || (s->n_attn && !s->kv))
        goto fail;

    for (int i = 0; i < s->n_ssm; i++)
        if (qwen35moe_ssm_init(&s->ssm[i], cfg) < 0)
            goto fail;
    for (int i = 0; i < s->n_attn; i++)
        if (qwen35moe_kv_init(&s->kv[i], cfg) < 0)
            goto fail;

    s->arena = pn_arena_new(PN_QWEN35MOE_ARENA_BYTES);
    s->x_norm = calloc((size_t)s->n_embd, sizeof(float));
    s->block_out = calloc((size_t)s->n_embd, sizeof(float));
    s->logits = calloc((size_t)s->n_vocab, sizeof(float));
    if (!s->arena || !s->x_norm || !s->block_out || !s->logits)
        goto fail;

    fprintf(stderr,
            "qwen35moe: dual state allocated (%d SSM + %d attn layers, "
            "arena %.1f MB)\n",
            s->n_ssm, s->n_attn,
            (double)PN_QWEN35MOE_ARENA_BYTES / (1024.0 * 1024.0));
    return s;

fail:
    pn_qwen35moe_state_free(s);
    return NULL;
}

void pn_qwen35moe_state_free(void *state)
{
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state;
    if (!s)
        return;

#ifdef USE_CUDA
    /* FIRST, while every buffer below is still allocated: make the GPU layers
     * forget this session. They keep raw per-slot back-pointers into `s`
     * (kv_slot_kv, kv_slot_host_id, host_state) and nothing else clears them,
     * so without this the next pn_qwen35moe_decode_slot_bind_cuda into that
     * slot dereferences freed memory in its lazy-mirror park copy. The serve
     * does exactly that on every session release and idle reap. */
    pn_qwen35moe_gpu_forget_state_cuda(s);
#endif

    if (s->ssm) {
        for (int i = 0; i < s->n_ssm; i++) {
            pn_lazy_zero_free(s->ssm[i].conv_state,
                              qwen35moe_ssm_conv_bytes(&s->ssm[i]));
            pn_lazy_zero_free(s->ssm[i].ssm_state,
                              qwen35moe_ssm_state_bytes(&s->ssm[i]));
        }
        free(s->ssm);
    }
    if (s->kv) {
        for (int i = 0; i < s->n_attn; i++) {
            size_t ring_bytes = qwen35moe_kv_ring_bytes(&s->kv[i]);
            pn_lazy_zero_free(s->kv[i].k, ring_bytes);
            pn_lazy_zero_free(s->kv[i].v, ring_bytes);
            pn_lazy_zero_free(s->kv[i].kq, ring_bytes);
            pn_lazy_zero_free(s->kv[i].vq, ring_bytes);
            free(s->kv[i].k_sink);
            free(s->kv[i].v_sink);
            free(s->kv[i].k_win);
            free(s->kv[i].v_win);
        }
        free(s->kv);
    }
    pn_arena_free(s->arena);
    free(s->x_norm);
    free(s->block_out);
    free(s->logits);
    free(s);
}

void pn_qwen35moe_state_reset(void *state)
{
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state;
    if (!s)
        return;

    for (int i = 0; i < s->n_ssm; i++) {
        pn_qwen35moe_ssm_state *st = &s->ssm[i];
        pn_lazy_zero_reset(st->conv_state, qwen35moe_ssm_conv_bytes(st));
        pn_lazy_zero_reset(st->ssm_state, qwen35moe_ssm_state_bytes(st));
    }
    for (int i = 0; i < s->n_attn; i++) {
        pn_qwen35moe_kv *kv = &s->kv[i];
        size_t ring_bytes = qwen35moe_kv_ring_bytes(kv);
        /* Phase 10b: lazy reset zeroes the mirror AND returns its committed
         * pages — a slot reused for a new request stops carrying the previous
         * session's ctx_cap-sized host high-water mark. */
        if (kv->q4) {
            pn_lazy_zero_reset(kv->kq, ring_bytes);
            pn_lazy_zero_reset(kv->vq, ring_bytes);
            size_t sink_n = (size_t)kv->n_head_kv * PN_KVQ4_SINKMAX *
                            kv->head_dim;
            size_t win_n = (size_t)kv->n_head_kv * PN_KVQ4_WMAX * kv->head_dim;
            memset(kv->k_sink, 0, sink_n * sizeof(uint16_t));
            memset(kv->v_sink, 0, sink_n * sizeof(uint16_t));
            memset(kv->k_win, 0, win_n * sizeof(uint16_t));
            memset(kv->v_win, 0, win_n * sizeof(uint16_t));
        } else {
            pn_lazy_zero_reset(kv->k, ring_bytes);
            pn_lazy_zero_reset(kv->v, ring_bytes);
        }
        /* B2: host mirror is now zeroed (empty context) — mark current so
         * preserve_live doesn't D2H stale GPU ring content over it. */
        kv->kv_mirror_stale = 0;
        /* Phase 10b: the buffer pointer survives the reset but its content
         * identity does not — invalidate every GPU ring watermark keyed on
         * (pointer, generation). */
        kv->content_gen = pn_qwen35moe_kv_next_content_gen();
    }
}

/* ============================================================
 * Forward (single token)
 * ============================================================ */

/* Run one layer's hybrid block: attn_norm -> (SSM|attn) -> +residual ->
 * post_attention_norm -> shared-MoE -> +residual. Mutates x in place. */
static int qwen35moe_layer_forward(pn_qwen35moe_model *m, int layer, float *x,
                                   pn_qwen35moe_state *s, int pos,
                                   struct fw_pool *pool)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    const float *attn_norm_w, *post_norm_w;
    const pn_qwen35moe_moe *moe;
    int rc;

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        const pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer(m, idx);
        attn_norm_w = l->attn_norm;
        post_norm_w = l->post_attention_norm;
        moe = &l->moe;
        pn_vrms_norm_f32(x, attn_norm_w, s->x_norm, c, cfg->rms_norm_eps);
        qwen35moe_dbg("attn_norm", layer, s->x_norm, c);
        pn_arena_reset(s->arena);
        rc = pn_qwen35moe_attn_block(l, cfg, s->x_norm, &s->kv[idx], pos,
                                     s->arena, s->block_out, pool);
    } else {
        const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, idx);
        attn_norm_w = l->attn_norm;
        post_norm_w = l->post_attention_norm;
        moe = &l->moe;
        pn_vrms_norm_f32(x, attn_norm_w, s->x_norm, c, cfg->rms_norm_eps);
        qwen35moe_dbg("attn_norm", layer, s->x_norm, c);
        pn_arena_reset(s->arena);
        rc = pn_qwen35moe_ssm_block(l, cfg, s->x_norm, &s->ssm[idx],
                                    s->arena, s->block_out, pool);
    }
    if (rc < 0)
        return rc;

    qwen35moe_dbg("block_out", layer, s->block_out, c);
    /* Residual after the SSM/attention block (block kernels skip it). */
    qwen35moe_residual_add(x, s->block_out, c);
    qwen35moe_dbg("attn_residual", layer, x, c);

    /* Shared-expert MoE FFN on the post_attention_norm'd hidden state. */
    pn_vrms_norm_f32(x, post_norm_w, s->x_norm, c, cfg->rms_norm_eps);
    qwen35moe_dbg("attn_post_norm", layer, s->x_norm, c);
    pn_arena_reset(s->arena);
    rc = pn_qwen35moe_moe_block(moe, cfg, s->x_norm, s->arena, s->block_out,
                                pool);
    if (rc < 0)
        return rc;
    qwen35moe_dbg("ffn_out", layer, s->block_out, c);
    qwen35moe_residual_add(x, s->block_out, c);
    qwen35moe_dbg("l_out", layer, x, c);
    return 0;
}

/* Internal forward with an explicit thread pool (the public entry point's pool
 * comes from the model; the per-block kernels accept it for parallel sections). */
static float *qwen35moe_forward_with_pool(pn_qwen35moe_model *m, uint32_t token,
                                          pn_qwen35moe_state *s, int pos,
                                          struct fw_pool *pool)
{
    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(m);
    int c = cfg->n_embd;
    int n_layer = cfg->n_layer;

    if ((int)token >= cfg->n_vocab)
        return NULL;

    /* Embed: copy the token's row from the dequantized embedding table into a
     * working hidden state that carries the residual stream through the stack. */
    float *hidden = calloc((size_t)c, sizeof(float));
    if (!hidden)
        return NULL;
    pn_qwen35moe_model_embed_row(m, token, hidden);
    qwen35moe_dbg("input_embed", -1, hidden, c);

    for (int layer = 0; layer < n_layer; layer++) {
        if (qwen35moe_layer_forward(m, layer, hidden, s, pos, pool) < 0) {
            free(hidden);
            return NULL;
        }
    }

    /* Final norm + LM head. */
    pn_vrms_norm_f32(hidden, pn_qwen35moe_model_output_norm(m), s->x_norm, c,
                     cfg->rms_norm_eps);
    qwen35moe_dbg("result_norm", -1, s->x_norm, c);
    qwen35moe_mv(cfg, pn_qwen35moe_model_output(m), s->x_norm, s->logits);
    qwen35moe_dbg("result_output", -1, s->logits, cfg->n_vocab);

    free(hidden);
    return s->logits;
}

float *pn_qwen35moe_forward_token(pn_qwen35moe_model *m, uint32_t token,
                                  pn_qwen35moe_state *s, int pos)
{
    if (!m || !s)
        return NULL;
    return qwen35moe_forward_with_pool(m, token, s, pos, NULL);
}

/* ============================================================
 * pn_forward_fn adapter
 * ============================================================ */

tm_tensor *pn_qwen35moe_forward(pn_model *model, uint32_t token, pn_state *state)
{
    if (!model || !state)
        return NULL;

    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return NULL;

    struct fw_pool *pool = NULL;
#ifdef USE_THREADS
    pool = model->pool;
#endif

    /* Thread pool lives on the model; thread it through to the per-block
     * kernels for their parallel sections. pos drives RoPE + the KV caches. */
    float *logits = qwen35moe_forward_with_pool(m, token, s, state->pos, pool);
    if (!logits)
        return NULL;

    int64_t shape[1] = { s->n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);
    if (!out)
        return NULL;
    memcpy(tm_data_f32(out), logits, (size_t)s->n_vocab * sizeof(float));

    state->pos++;
    return out;
}
