/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_ssm.c
 * @brief  Gated-Delta-Net SSM block for Qwen3.5-MoE, one token at a time.
 *
 * Written from a behaviour specification (plans/cleanroom_r2_qwen35_spec.md); no ggml or llama.cpp source was consulted.
 *
 * @status None
 * @issues The state struct's own dims are ignored (the config's are used). ssm_conv == 1
 *         writes one float before conv_state. groups = nv / nk is computed before the
 *         asserts. A failure after the conv step leaves conv_state rolled. QWEN35_DEBUG is
 *         honoured only on the first recurrence call of the process; the trace statics are
 *         not thread-safe.
 * @todo   None
 */

#include "qwen35moe_internal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/simd.h"
#include "../core/qweight.h"
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

/* Stack scratch per K-head in the q/k normalise step; bounds ssm_state. */
#define SSM_HEAD_SCRATCH 256

static float ssm_softplus(float x) {
    if (x > 20.0f)
        return x;
    return log1pf(expf(x));
}

static float ssm_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* QWEN35_DEBUG trace line: first three, last three, and the double sum of x[0..n). */
static void ssm_dbg(const char *tag, const float *x, int n) {
    static int debug_enabled = -1;
    if (debug_enabled < 0)
        debug_enabled = getenv("QWEN35_DEBUG") != NULL ? 1 : 0;
    if (!debug_enabled)
        return;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    fprintf(stderr, "SSM %-18s [%12.6f %12.6f %12.6f ... %12.6f %12.6f %12.6f] sum=%.6f\n",
            tag, x[0], x[1], x[2], x[n - 3], x[n - 2], x[n - 1], sum);
}

void pn_qwen35moe_l2_norm(float *data, int head_dim, float eps) {
    double sumsq = 0.0;
    for (int d = 0; d < head_dim; d++)
        sumsq += (double)data[d] * (double)data[d];
    float inv = 1.0f / fmaxf((float)sqrt(sumsq), eps);
    for (int d = 0; d < head_dim; d++)
        data[d] *= inv;
}

void pn_qwen35moe_beta_sigmoid(const float *beta_raw, int n_v_heads, float *beta_out) {
    for (int h = 0; h < n_v_heads; h++)
        beta_out[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
}

void pn_qwen35moe_decay_gate(const pn_qwen35moe_ssm_layer *layer,
                             const float *alpha, int n_v_heads, float *g_out) {
    for (int h = 0; h < n_v_heads; h++) {
        float sp = ssm_softplus(alpha[h] + layer->ssm_dt[h]);
        g_out[h] = layer->ssm_a[h] * sp;
    }
}

void pn_qwen35moe_conv1d_silu(const pn_qwen35moe_ssm_layer *layer,
                              const pn_qwen35moe_config *cfg, const float *qkv,
                              pn_qwen35moe_ssm_state *state, float *conv_out) {
    int channels = cfg->conv_dim;
    int taps = cfg->ssm_conv;
    int m = taps - 1;
    const float *w = layer->ssm_conv1d;
    for (int c = 0; c < channels; c++) {
        const float *cw = w + (size_t)c * taps;
        float *cs = state->conv_state + (size_t)c * m;
        float acc = qkv[c] * cw[m];
        for (int t = 0; t < m; t++)
            acc += cs[t] * cw[t];
        conv_out[c] = ssm_silu(acc);
        for (int t = 0; t < m - 1; t++)
            cs[t] = cs[t + 1];
        cs[m - 1] = qkv[c];
    }
}

void pn_qwen35moe_gated_norm(const pn_qwen35moe_ssm_layer *layer,
                             const pn_qwen35moe_config *cfg, const float *y,
                             const float *z, float *gated) {
    int S = cfg->ssm_state;
    int n_heads = cfg->ssm_dt_rank;
    float eps = cfg->rms_norm_eps;
    for (int h = 0; h < n_heads; h++) {
        const float *yh = y + (size_t)h * S;
        const float *zh = z + (size_t)h * S;
        float *gh = gated + (size_t)h * S;
        pn_vrms_norm_f32(yh, layer->ssm_norm, gh, S, eps);
        for (int d = 0; d < S; d++)
            gh[d] *= ssm_silu(zh[d]);
    }
}

/* Inputs of the per-V-head delta-rule worker. */
typedef struct {
    const float *q;
    const float *k;
    const float *v;
    const float *g;
    const float *beta;
    float *state;
    float *y;
    int S;
    float scale;
} ssm_recurrence_args;

/* Delta rule for V-heads [start, end): decay, predict, write, then read per value row. */
static void ssm_recurrence_heads(int start, int end, void *arg) {
    const ssm_recurrence_args *a = (const ssm_recurrence_args *)arg;
    int S = a->S;
    for (int h = start; h < end; h++) {
        float *sh = a->state + (size_t)h * S * S;
        const float *kh = a->k + (size_t)h * S;
        const float *qh = a->q + (size_t)h * S;
        const float *vh = a->v + (size_t)h * S;
        float *yh = a->y + (size_t)h * S;
        float decay = expf(a->g[h]);
        float beta = a->beta[h];
        for (int j = 0; j < S; j++) {
            float *row = sh + (size_t)j * S;
            for (int i = 0; i < S; i++)
                row[i] *= decay;
            double predd = 0.0;
            for (int i = 0; i < S; i++)
                predd += (double)row[i] * (double)kh[i];
            float delta = (vh[j] - (float)predd) * beta;
            for (int i = 0; i < S; i++)
                row[i] += kh[i] * delta;
            double yd = 0.0;
            for (int i = 0; i < S; i++)
                yd += (double)row[i] * (double)qh[i];
            yh[j] = a->scale * (float)yd;
        }
    }
}

/* L2-normalise each K-head of the q and k sources and repeat them across V-heads, tiled:
 * V-head r*nk + kh takes K-head kh. */
static void ssm_split_normalise_repeat(const float *conv_out, int S, int nk, int groups,
                                       int key_dim, int val_dim, float eps,
                                       float *q, float *k, float *v) {
    const float *q_src = conv_out;
    const float *k_src = conv_out + key_dim;
    const float *v_src = conv_out + 2 * (size_t)key_dim;
    for (int kh = 0; kh < nk; kh++) {
        float q_head[SSM_HEAD_SCRATCH];
        float k_head[SSM_HEAD_SCRATCH];
        memcpy(q_head, q_src + (size_t)kh * S, (size_t)S * sizeof(*q_head));
        memcpy(k_head, k_src + (size_t)kh * S, (size_t)S * sizeof(*k_head));
        pn_qwen35moe_l2_norm(q_head, S, eps);
        pn_qwen35moe_l2_norm(k_head, S, eps);
        for (int r = 0; r < groups; r++) {
            int vh = r * nk + kh;
            memcpy(q + (size_t)vh * S, q_head, (size_t)S * sizeof(*q));
            memcpy(k + (size_t)vh * S, k_head, (size_t)S * sizeof(*k));
        }
    }
    memcpy(v, v_src, (size_t)val_dim * sizeof(*v));
}

int pn_qwen35moe_ssm_recurrence(const pn_qwen35moe_ssm_layer *layer,
                                const pn_qwen35moe_config *cfg,
                                const float *qkv, const float *z,
                                const float *beta_raw, const float *alpha,
                                pn_qwen35moe_ssm_state *state, pn_arena *arena,
                                float *gated, struct fw_pool *pool) {
    if (!layer || !cfg || !qkv || !z || !beta_raw || !alpha || !state || !arena || !gated)
        return -1;
    int C = cfg->conv_dim;
    int S = cfg->ssm_state;
    int nk = cfg->ssm_n_group;
    int nv = cfg->ssm_dt_rank;
    int groups = nv / nk;
    int key_dim = S * nk;
    int val_dim = S * nv;
    assert(S <= SSM_HEAD_SCRATCH && "ssm_state exceeds the per-head stack scratch");
    assert(nv % nk == 0 && "V-heads must be a multiple of K-heads");

    float *beta = pn_arena_alloc_f32(arena, nv);
    if (!beta)
        return -1;

    /* Trace only the first recurrence call of the process. */
    static int traced_before = 0;
    int trace = !traced_before;
    traced_before = 1;

    if (trace) {
        ssm_dbg("qkv_mixed", qkv, C);
        ssm_dbg("z", z, val_dim);
    }
    pn_qwen35moe_beta_sigmoid(beta_raw, nv, beta);
    if (trace)
        ssm_dbg("beta_sigmoid", beta, nv);

    float *g = pn_arena_alloc_f32(arena, nv);
    if (!g)
        return -1;
    pn_qwen35moe_decay_gate(layer, alpha, nv, g);
    if (trace)
        ssm_dbg("gate", g, nv);

    float *conv_out = pn_arena_alloc_f32(arena, C);
    if (!conv_out)
        return -1;
    pn_qwen35moe_conv1d_silu(layer, cfg, qkv, state, conv_out);
    if (trace)
        ssm_dbg("conv_silu", conv_out, C);

    float *q = pn_arena_alloc_f32(arena, val_dim);
    float *k = pn_arena_alloc_f32(arena, val_dim);
    float *v = pn_arena_alloc_f32(arena, val_dim);
    if (!q || !k || !v)
        return -1;
    ssm_split_normalise_repeat(conv_out, S, nk, groups, key_dim, val_dim, cfg->rms_norm_eps,
                               q, k, v);
    if (trace) {
        ssm_dbg("q_predelta", q, val_dim);
        ssm_dbg("k_predelta", k, val_dim);
        ssm_dbg("v_predelta", v, val_dim);
    }

    float *y = pn_arena_alloc_f32(arena, val_dim);
    if (!y)
        return -1;
    ssm_recurrence_args args = {
        .q = q, .k = k, .v = v, .g = g, .beta = beta,
        .state = state->ssm_state, .y = y, .S = S,
        .scale = 1.0f / sqrtf((float)S),
    };
#ifdef USE_THREADS
    if (pool && nv >= 4)
        fw_for(pool, nv, ssm_recurrence_heads, &args);
    else
        ssm_recurrence_heads(0, nv, &args);
#else
    (void)pool;
    ssm_recurrence_heads(0, nv, &args);
#endif
    if (trace)
        ssm_dbg("recur_y", y, val_dim);
    pn_qwen35moe_gated_norm(layer, cfg, y, z, gated);
    if (trace)
        ssm_dbg("final_output", gated, val_dim);
    return 0;
}

int pn_qwen35moe_ssm_block(const pn_qwen35moe_ssm_layer *layer,
                           const pn_qwen35moe_config *cfg, const float *x,
                           pn_qwen35moe_ssm_state *state, pn_arena *arena,
                           float *out, struct fw_pool *pool) {
    if (!layer || !cfg || !x || !state || !arena || !out)
        return -1;
    int val_dim = cfg->ssm_state * cfg->ssm_dt_rank;
    float *qkv = pn_arena_alloc_f32(arena, cfg->conv_dim);
    float *z = pn_arena_alloc_f32(arena, val_dim);
    float *beta = pn_arena_alloc_f32(arena, cfg->ssm_dt_rank);
    float *alpha = pn_arena_alloc_f32(arena, cfg->ssm_dt_rank);
    float *gated = pn_arena_alloc_f32(arena, val_dim);
    if (!qkv || !z || !beta || !alpha || !gated)
        return -1;
    qwen35moe_mv(cfg, &layer->attn_qkv, x, qkv);
    qwen35moe_mv(cfg, &layer->attn_gate, x, z);
    qwen35moe_mv(cfg, &layer->ssm_beta, x, beta);
    qwen35moe_mv(cfg, &layer->ssm_alpha, x, alpha);
    if (pn_qwen35moe_ssm_recurrence(layer, cfg, qkv, z, beta, alpha, state, arena, gated,
                                    pool) < 0)
        return -1;
    qwen35moe_mv(cfg, &layer->ssm_out, gated, out);
    return 0;
}
