/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pixtral_vision.c
 * @brief  Pixtral-style vision tower and LightOnOCR-style projector on F32 weights.
 *
 * Written from a behaviour specification (plans/cleanroom_r2_pixtral_spec.md); no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues - Kept as-is pending a separate fix: use_silu and image_size are loaded but
 *           unused (long edge fixed at 1024); image_mean/std arrays are not type- or
 *           length-checked and their descriptors leak; tensor shapes are not validated;
 *           allocations are unchecked; out_embd == NULL crashes; n_head == 0 divides by
 *           zero; head_dim % 4 != 0 leaves RoPE slots uninitialised; non-flash paths
 *           allocate N*N scores; on macOS the projector sgemms run with the caller's
 *           BLAS threading.
 *         - pn_vision_encode is far past ~100 lines: the attention fallbacks are inline
 *           on purpose, because release output bits depend on where loops live.
 * @todo   None
 */
#include "pixtral_vision.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
  #ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif

#ifdef USE_BLAS
#define PN_VISION_HAVE_CBLAS 1
#endif

/* Tiled online-softmax attention: never materialises the N x N score matrix. */
#if defined(USE_THREADS) && defined(PN_VISION_HAVE_CBLAS) && !defined(PN_NO_VISION_FLASH)
#define PN_VISION_FLASH 1
#define PN_VISION_FLASH_QUERY_TILE 128
#define PN_VISION_FLASH_KEY_TILE 512
#endif

/* Runtime long edge of the resized image (independent of cfg.image_size). */
#define PN_VISION_TARGET_LONG_EDGE 1024

/* ------------------------------------------------------------------------- */
/* Elementwise / linear helpers                                               */
/* ------------------------------------------------------------------------- */

/* Y[M,N] = X[M,K] * W[N,K]^T, all row-major. */
static void vision_linear_forward(const float *X, const float *W, float *Y, int M, int N, int K) {
#ifdef PN_VISION_HAVE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, X, K, W, K, 0.0f, Y, N);
#else
    for (int row = 0; row < M; row++) {
        const float *x = X + (size_t)row * K;
        for (int col = 0; col < N; col++) {
            const float *w = W + (size_t)col * K;
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += x[k] * w[k];
            Y[(size_t)row * N + col] = acc;
        }
    }
#endif
}

/* In-place RMSNorm of one row; the sum of squares is accumulated in double. */
static void vision_rms_norm_row(float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++)
        x[i] = x[i] * inv * w[i];
}

static inline float vision_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static inline float vision_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Bilinear RGB8 resize with half-pixel centres and edge clamping. */
static void vision_resize_bilinear_rgb8(const uint8_t *src, int sw, int sh,
                                        uint8_t *dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        float sy = (y + 0.5f) * sh / dh - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y1 > sh - 1) y1 = sh - 1;
        if (y0 > sh - 1) y0 = sh - 1;
        for (int x = 0; x < dw; x++) {
            float sx = (x + 0.5f) * sw / dw - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x1 > sw - 1) x1 = sw - 1;
            if (x0 > sw - 1) x0 = sw - 1;
            for (int ch = 0; ch < 3; ch++) {
                float a = src[(y0 * sw + x0) * 3 + ch];
                float b = src[(y0 * sw + x1) * 3 + ch];
                float cc = src[(y1 * sw + x0) * 3 + ch];
                float d = src[(y1 * sw + x1) * 3 + ch];
                float top = a * (1 - fx) + b * fx;
                float bot = cc * (1 - fx) + d * fx;
                dst[(y * dw + x) * 3 + ch] = (uint8_t)lroundf(top * (1 - fy) + bot * fy);
            }
        }
    }
}

/*
 * Per-patch 2D RoPE tables [gh*gw, head_dim]. The first half of each head rotates by
 * the patch row, the second half by the patch column; each pair's two slots hold the
 * same angle.
 */
static void vision_rope_2d_tables(float *cosb, float *sinb, int gh, int gw, int head_dim,
                                  float theta) {
    int half = head_dim / 2;
    int npairs = half / 2;
    for (int r = 0; r < gh; r++) {
        for (int c = 0; c < gw; c++) {
            float *cs = cosb + (size_t)(r * gw + c) * head_dim;
            float *sn = sinb + (size_t)(r * gw + c) * head_dim;
            for (int j = 0; j < npairs; j++) {
                float fe = powf(theta, -(float)(2 * j) / (float)half);
                float fo = powf(theta, -(float)(2 * j + 1) / (float)half);
                float ar = r * fe;
                float ac = c * fo;
                cs[2 * j] = cs[2 * j + 1] = cosf(ar);
                sn[2 * j] = sn[2 * j + 1] = sinf(ar);
                cs[half + 2 * j] = cs[half + 2 * j + 1] = cosf(ac);
                sn[half + 2 * j] = sn[half + 2 * j + 1] = sinf(ac);
            }
        }
    }
}

/* Rotate adjacent element pairs of one head; even slot uses table[2k], odd table[2k+1]. */
static void vision_rope_rotate_head(float *v, const float *cp, const float *sp, int hd) {
    for (int k = 0; k < hd / 2; k++) {
        float x0 = v[2 * k];
        float x1 = v[2 * k + 1];
        v[2 * k] = x0 * cp[2 * k] - x1 * sp[2 * k];
        v[2 * k + 1] = x0 * sp[2 * k + 1] + x1 * cp[2 * k + 1];
    }
}

/* ------------------------------------------------------------------------- */
/* Attention workers (one per build path)                                     */
/* ------------------------------------------------------------------------- */

#ifdef PN_VISION_FLASH
typedef struct {
    const float *Q, *K, *V;
    float *AO;
    int N, D, off, hd;
    float scaleq;
} vision_flash_attention_job;

/*
 * Query tiles [start, end) of one head. Each tile is computed whole by one thread and
 * written to its own rows, so results do not depend on the worker count.
 */
static void vision_flash_attention_tiles(int start, int end, void *arg) {
    const vision_flash_attention_job *job = (const vision_flash_attention_job *)arg;
    const int BQ = PN_VISION_FLASH_QUERY_TILE;
    const int BK = PN_VISION_FLASH_KEY_TILE;
    const int N = job->N, D = job->D, off = job->off, hd = job->hd;
#ifdef __APPLE__
    /* Tile sgemms are small; threading them inside a worker would oversubscribe. */
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
#endif
    float *S = malloc((size_t)BQ * BK * sizeof(float));
    float *O = malloc((size_t)BQ * hd * sizeof(float));
    float *mrun = malloc((size_t)BQ * sizeof(float));
    float *lrun = malloc((size_t)BQ * sizeof(float));

    for (int qt = start; qt < end; qt++) {
        int q0 = qt * BQ;
        int qn = N - q0 < BQ ? N - q0 : BQ;
        const float *Qp = job->Q + (size_t)q0 * D + off;
        for (int i = 0; i < qn; i++) {
            mrun[i] = -1e30f;
            lrun[i] = 0.0f;
        }
        memset(O, 0, (size_t)qn * hd * sizeof(float));

        for (int k0 = 0; k0 < N; k0 += BK) {
            int kn = N - k0 < BK ? N - k0 : BK;
            const float *Kp = job->K + (size_t)k0 * D + off;
            const float *Vp = job->V + (size_t)k0 * D + off;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, qn, kn, hd, job->scaleq,
                        Qp, D, Kp, D, 0.0f, S, kn);
            for (int i = 0; i < qn; i++) {
                float *srow = S + (size_t)i * kn;
                float mprev = mrun[i];
                float mcur = mprev;
                for (int j = 0; j < kn; j++)
                    if (srow[j] > mcur) mcur = srow[j];
                float corr = expf(mprev - mcur);
                float sum = 0.0f;
                for (int j = 0; j < kn; j++) {
                    float p = expf(srow[j] - mcur);
                    srow[j] = p;
                    sum += p;
                }
                lrun[i] = lrun[i] * corr + sum;
                mrun[i] = mcur;
                for (int d = 0; d < hd; d++)
                    O[i * hd + d] *= corr;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, qn, hd, kn, 1.0f,
                        S, kn, Vp, D, 1.0f, O, hd);
        }

        for (int i = 0; i < qn; i++) {
            float inv = 1.0f / lrun[i];
            for (int d = 0; d < hd; d++)
                job->AO[(size_t)(q0 + i) * D + off + d] = O[i * hd + d] * inv;
        }
    }

    free(S);
    free(O);
    free(mrun);
    free(lrun);
}
#endif /* PN_VISION_FLASH */

#if defined(USE_THREADS) && defined(PN_VISION_HAVE_CBLAS) && !defined(PN_VISION_FLASH)
typedef struct {
    float *scores;
    int N;
} vision_softmax_job;

/* In-place softmax of score rows [start, end). */
static void vision_softmax_rows(int start, int end, void *arg) {
    const vision_softmax_job *job = (const vision_softmax_job *)arg;
    int N = job->N;
    for (int qi = start; qi < end; qi++) {
        float *row = job->scores + (size_t)qi * N;
        float mx = -1e30f;
        for (int j = 0; j < N; j++)
            if (row[j] > mx) mx = row[j];
        float sum = 0;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }
}
#endif

#if defined(USE_THREADS) && !defined(PN_VISION_HAVE_CBLAS)
typedef struct {
    const float *Q, *K, *V;
    float *AO, *scores;
    int N, D, off, hd;
    float scaleq;
} vision_attention_job;

/* Scalar attention for query rows [start, end) of one head; row qi uses scores + qi*N. */
static void vision_attention_rows(int start, int end, void *arg) {
    const vision_attention_job *job = (const vision_attention_job *)arg;
    int N = job->N, D = job->D, off = job->off, hd = job->hd;
    float scaleq = job->scaleq;
    for (int qi = start; qi < end; qi++) {
        float *scores = job->scores + (size_t)qi * N;
        const float *q = job->Q + (size_t)qi * D + off;
        float maxs = -1e30f;
        for (int ki = 0; ki < N; ki++) {
            const float *k = job->K + (size_t)ki * D + off;
            float s = 0;
            for (int d = 0; d < hd; d++)
                s += q[d] * k[d];
            s *= scaleq;
            scores[ki] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0;
        for (int ki = 0; ki < N; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        float inv = 1.0f / sum;
        float *o = job->AO + (size_t)qi * D + off;
        for (int d = 0; d < hd; d++)
            o[d] = 0;
        for (int ki = 0; ki < N; ki++) {
            float wgt = scores[ki] * inv;
            const float *v = job->V + (size_t)ki * D + off;
            for (int d = 0; d < hd; d++)
                o[d] += wgt * v[d];
        }
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Load / free                                                                */
/* ------------------------------------------------------------------------- */

pn_vision_model* pn_vision_load(pn_gguf_file *gguf) {
    if (!gguf)
        return NULL;
    const char *arch = pn_gguf_get_str(gguf, "general.architecture");
    if (!arch || strcmp(arch, "clip") != 0) {
        fprintf(stderr, "vision: not a clip mmproj GGUF\n");
        return NULL;
    }

    pn_vision_model *m = calloc(1, sizeof(*m));
    pn_vision_config *c = &m->cfg;
    c->image_size = (int)pn_gguf_get_u32(gguf, "clip.vision.image_size", 1540);
    c->patch_size = (int)pn_gguf_get_u32(gguf, "clip.vision.patch_size", 14);
    c->hidden = (int)pn_gguf_get_u32(gguf, "clip.vision.embedding_length", 1024);
    c->n_layer = (int)pn_gguf_get_u32(gguf, "clip.vision.block_count", 24);
    c->n_head = (int)pn_gguf_get_u32(gguf, "clip.vision.attention.head_count", 16);
    c->n_ff = (int)pn_gguf_get_u32(gguf, "clip.vision.feed_forward_length", 4096);
    c->merge = (int)pn_gguf_get_u32(gguf, "clip.vision.spatial_merge_size", 2);
    c->proj_dim = (int)pn_gguf_get_u32(gguf, "clip.vision.projection_dim", 1024);
    c->eps = pn_gguf_get_f32(gguf, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
    c->rope_theta = pn_gguf_get_f32(gguf, "clip.vision.rope_theta", 10000.0f);
    c->use_silu = (int)pn_gguf_get_u32(gguf, "clip.use_silu", 1);
    c->head_dim = c->hidden / c->n_head;

    /* Array type and length are trusted as-is (see @issues). */
    pn_gguf_arr *mean = pn_gguf_get_arr(gguf, "clip.vision.image_mean");
    pn_gguf_arr *std = pn_gguf_get_arr(gguf, "clip.vision.image_std");
    for (int ch = 0; ch < 3; ch++) {
        c->image_mean[ch] = mean ? ((float *)mean->data)[ch] : 0.5f;
        c->image_std[ch] = std ? ((float *)std->data)[ch] : 0.5f;
    }

    /* Every top-level tensor is attempted so each missing one is reported. */
    m->patch_embd = pn_gguf_dequantize_tensor(gguf, "v.patch_embd.weight", NULL);
    m->pre_ln = pn_gguf_dequantize_tensor(gguf, "v.pre_ln.weight", NULL);
    m->mm_input_norm = pn_gguf_dequantize_tensor(gguf, "mm.input_norm.weight", NULL);
    m->mm_merger = pn_gguf_dequantize_tensor(gguf, "mm.patch_merger.weight", NULL);
    m->mm1 = pn_gguf_dequantize_tensor(gguf, "mm.1.weight", NULL);
    m->mm2 = pn_gguf_dequantize_tensor(gguf, "mm.2.weight", NULL);

    m->layers = calloc(c->n_layer, sizeof(*m->layers));
    int ok = m->patch_embd && m->pre_ln && m->mm_input_norm && m->mm_merger && m->mm1 && m->mm2;
    static const char *const layer_tensor_names[9] = {
        "ln1", "attn_q", "attn_k", "attn_v", "attn_out", "ln2", "ffn_gate", "ffn_up", "ffn_down"
    };
    for (int i = 0; ok && i < c->n_layer; i++) {
        pn_vision_layer *L = &m->layers[i];
        float **slots[9] = { &L->ln1, &L->wq, &L->wk, &L->wv, &L->wo,
                             &L->ln2, &L->w_gate, &L->w_up, &L->w_down };
        for (int t = 0; t < 9; t++) {
            char name[64];
            snprintf(name, sizeof(name), "v.blk.%d.%s.weight", i, layer_tensor_names[t]);
            *slots[t] = pn_gguf_dequantize_tensor(gguf, name, NULL);
        }
        for (int t = 0; t < 9; t++)
            if (!*slots[t]) ok = 0;
    }

    if (!ok) {
        fprintf(stderr, "vision: missing tensors\n");
        pn_vision_free(m);
        return NULL;
    }
    fprintf(stderr, "vision: clip loaded (hidden=%d layers=%d heads=%d ff=%d patch=%d merge=%d)\n",
            c->hidden, c->n_layer, c->n_head, c->n_ff, c->patch_size, c->merge);
    return m;
}

void pn_vision_free(pn_vision_model *m) {
    if (!m)
        return;
    free(m->patch_embd);
    free(m->pre_ln);
    free(m->mm_input_norm);
    free(m->mm_merger);
    free(m->mm1);
    free(m->mm2);
    if (m->layers) {
        for (int i = 0; i < m->cfg.n_layer; i++) {
            pn_vision_layer *L = &m->layers[i];
            free(L->ln1);
            free(L->wq);
            free(L->wk);
            free(L->wv);
            free(L->wo);
            free(L->ln2);
            free(L->w_gate);
            free(L->w_up);
            free(L->w_down);
        }
        free(m->layers);
    }
    free(m);
}

/* ------------------------------------------------------------------------- */
/* Encode                                                                     */
/* ------------------------------------------------------------------------- */

/*
 * Kept as one function on purpose: under -ffast-math the release bits depend on loop
 * shapes and where they live, so the serial attention fallbacks stay inline here.
 */
int pn_vision_encode(pn_vision_model *m, const uint8_t *rgb, int w, int h,
                     float **out_embd, int *out_dim) {
    if (!m || !rgb || w <= 0 || h <= 0)
        return -1;
    const pn_vision_config *c = &m->cfg;
    const int ps = c->patch_size, mg = c->merge, D = c->hidden, hd = c->head_dim;
    const int nh = c->n_head, F = c->n_ff, PD = c->proj_dim;
    const float eps = c->eps;

    /* Target size: long edge scaled to 1024, rounded up to whole patches, at least one
     * merge block per side. */
    int step = ps * mg;
    int long_edge = w > h ? w : h;
    float scale = (float)PN_VISION_TARGET_LONG_EDGE / (float)long_edge;
    int tw = (int)((float)w * scale);
    int th = (int)((float)h * scale);
    tw = ((tw + ps - 1) / ps) * ps;
    if (tw < step) tw = step;
    th = ((th + ps - 1) / ps) * ps;
    if (th < step) th = step;
    int gw = tw / ps, gh = th / ps, N = gw * gh;

    uint8_t *img = malloc((size_t)tw * th * 3);
    vision_resize_bilinear_rgb8(rgb, w, h, img, tw, th);

    /* Patch matrix [N, 3*ps*ps], each row ordered (channel, ky, kx). */
    int pk = 3 * ps * ps;
    float *P = malloc((size_t)N * pk * sizeof(float));
    for (int gr = 0; gr < gh; gr++) {
        for (int gc = 0; gc < gw; gc++) {
            float *prow = P + (size_t)(gr * gw + gc) * pk;
            int idx = 0;
            for (int ch = 0; ch < 3; ch++) {
                for (int ky = 0; ky < ps; ky++) {
                    for (int kx = 0; kx < ps; kx++) {
                        int px = gc * ps + kx;
                        int py = gr * ps + ky;
                        float v = img[(py * tw + px) * 3 + ch] / 255.0f;
                        prow[idx++] = (v - c->image_mean[ch]) / c->image_std[ch];
                    }
                }
            }
        }
    }
    free(img);

    float *X = malloc((size_t)N * D * sizeof(float));
    vision_linear_forward(P, m->patch_embd, X, N, D, pk);
    free(P);
    for (int i = 0; i < N; i++)
        vision_rms_norm_row(X + (size_t)i * D, m->pre_ln, D, eps);

    float *cosb = malloc((size_t)N * hd * sizeof(float));
    float *sinb = malloc((size_t)N * hd * sizeof(float));
    vision_rope_2d_tables(cosb, sinb, gh, gw, hd, c->rope_theta);

    size_t nd = (size_t)N * D;
    size_t nf = (size_t)N * F;
    float *normed = malloc(nd * sizeof(float));
    float *Q = malloc(nd * sizeof(float));
    float *K = malloc(nd * sizeof(float));
    float *V = malloc(nd * sizeof(float));
    float *AO = malloc(nd * sizeof(float));
    float *tmpD = malloc(nd * sizeof(float));
    float *gate = malloc(nf * sizeof(float));
    float *up = malloc(nf * sizeof(float));
#ifdef PN_VISION_FLASH
    float *scores = NULL;
#else
    float *scores = malloc((size_t)N * N * sizeof(float));
#endif
    float scaleq = 1.0f / sqrtf((float)hd);

#ifdef USE_THREADS
    fw_pool *vpool = fw_pool_new(0);
#endif
#if defined(PN_VISION_FLASH) && defined(__APPLE__)
    /* One thread already saturates the matrix unit for the big matmuls; worker-level
     * threading on top would oversubscribe. Restored before the projector. */
    int saved_blas_threading = (int)BLASGetThreading();
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
#endif

    for (int l = 0; l < c->n_layer; l++) {
        const pn_vision_layer *L = &m->layers[l];

        memcpy(normed, X, nd * sizeof(float));
        for (int i = 0; i < N; i++)
            vision_rms_norm_row(normed + (size_t)i * D, L->ln1, D, eps);
        vision_linear_forward(normed, L->wq, Q, N, D, D);
        vision_linear_forward(normed, L->wk, K, N, D, D);
        vision_linear_forward(normed, L->wv, V, N, D, D);

        for (int i = 0; i < N; i++) {
            for (int hh = 0; hh < nh; hh++) {
                vision_rope_rotate_head(Q + (size_t)i * D + hh * hd, cosb + (size_t)i * hd,
                                        sinb + (size_t)i * hd, hd);
                vision_rope_rotate_head(K + (size_t)i * D + hh * hd, cosb + (size_t)i * hd,
                                        sinb + (size_t)i * hd, hd);
            }
        }

        for (int hh = 0; hh < nh; hh++) {
            int off = hh * hd;
#if defined(PN_VISION_FLASH)
            int ntiles = (N + PN_VISION_FLASH_QUERY_TILE - 1) / PN_VISION_FLASH_QUERY_TILE;
            vision_flash_attention_job job = { Q, K, V, AO, N, D, off, hd, scaleq };
            if (vpool)
                fw_for(vpool, ntiles, vision_flash_attention_tiles, &job);
            else
                vision_flash_attention_tiles(0, ntiles, &job);
#elif defined(PN_VISION_HAVE_CBLAS)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, hd, scaleq,
                        Q + off, D, K + off, D, 0.0f, scores, N);
#ifdef USE_THREADS
            vision_softmax_job sjob = { scores, N };
            if (vpool) {
                fw_for(vpool, N, vision_softmax_rows, &sjob);
            } else
#endif
            {
                for (int qi = 0; qi < N; qi++) {
                    float *row = scores + (size_t)qi * N;
                    float mx = -1e30f;
                    for (int j = 0; j < N; j++)
                        if (row[j] > mx) mx = row[j];
                    float sum = 0;
                    for (int j = 0; j < N; j++) {
                        row[j] = expf(row[j] - mx);
                        sum += row[j];
                    }
                    float inv = 1.0f / sum;
                    for (int j = 0; j < N; j++)
                        row[j] *= inv;
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, hd, N, 1.0f,
                        scores, N, V + off, D, 0.0f, AO + off, D);
#else
#ifdef USE_THREADS
            vision_attention_job ajob = { Q, K, V, AO, scores, N, D, off, hd, scaleq };
            if (vpool) {
                fw_for(vpool, N, vision_attention_rows, &ajob);
            } else
#endif
            {
                for (int qi = 0; qi < N; qi++) {
                    const float *q = Q + (size_t)qi * D + off;
                    float maxs = -1e30f;
                    for (int ki = 0; ki < N; ki++) {
                        const float *k = K + (size_t)ki * D + off;
                        float s = 0;
                        for (int d = 0; d < hd; d++)
                            s += q[d] * k[d];
                        s *= scaleq;
                        scores[ki] = s;
                        if (s > maxs) maxs = s;
                    }
                    float sum = 0;
                    for (int ki = 0; ki < N; ki++) {
                        scores[ki] = expf(scores[ki] - maxs);
                        sum += scores[ki];
                    }
                    float inv = 1.0f / sum;
                    float *o = AO + (size_t)qi * D + off;
                    for (int d = 0; d < hd; d++)
                        o[d] = 0;
                    for (int ki = 0; ki < N; ki++) {
                        float wgt = scores[ki] * inv;
                        const float *v = V + (size_t)ki * D + off;
                        for (int d = 0; d < hd; d++)
                            o[d] += wgt * v[d];
                    }
                }
            }
#endif
        }

        vision_linear_forward(AO, L->wo, tmpD, N, D, D);
        for (size_t i = 0; i < nd; i++)
            X[i] += tmpD[i];

        memcpy(normed, X, nd * sizeof(float));
        for (int i = 0; i < N; i++)
            vision_rms_norm_row(normed + (size_t)i * D, L->ln2, D, eps);
        vision_linear_forward(normed, L->w_gate, gate, N, F, D);
        vision_linear_forward(normed, L->w_up, up, N, F, D);
        for (size_t i = 0; i < nf; i++)
            gate[i] = vision_silu(gate[i]) * up[i];
        vision_linear_forward(gate, L->w_down, tmpD, N, D, F);
        for (size_t i = 0; i < nd; i++)
            X[i] += tmpD[i];
    }

#ifdef USE_THREADS
    if (vpool)
        fw_pool_free(vpool);
#endif
#if defined(PN_VISION_FLASH) && defined(__APPLE__)
    BLASSetThreading((enum BLAS_THREADING)saved_blas_threading);
#endif
    free(normed);
    free(Q);
    free(K);
    free(V);
    free(AO);
    free(gate);
    free(up);
    free(scores);
    free(cosb);
    free(sinb);

    /* Projector: RMSNorm, mg x mg spatial merge (channel-major), linear, GELU, linear.
     * An odd trailing patch row/column is dropped by the merge. */
    for (int i = 0; i < N; i++)
        vision_rms_norm_row(X + (size_t)i * D, m->mm_input_norm, D, eps);
    int mh = gh / mg, mw = gw / mg, M = mh * mw, mk = D * mg * mg;
    float *merged = malloc((size_t)M * mk * sizeof(float));
    for (int br = 0; br < mh; br++) {
        for (int bc = 0; bc < mw; bc++) {
            float *dst = merged + (size_t)(br * mw + bc) * mk;
            for (int ch = 0; ch < D; ch++)
                for (int ky = 0; ky < mg; ky++)
                    for (int kx = 0; kx < mg; kx++)
                        dst[ch * mg * mg + ky * mg + kx] =
                            X[((size_t)(br * mg + ky) * gw + (bc * mg + kx)) * D + ch];
        }
    }
    free(X);
    free(tmpD);

    size_t mpd = (size_t)M * PD;
    float *mo = malloc(mpd * sizeof(float));
    vision_linear_forward(merged, m->mm_merger, mo, M, PD, mk);
    free(merged);
    float *h1 = malloc(mpd * sizeof(float));
    vision_linear_forward(mo, m->mm1, h1, M, PD, PD);
    for (size_t i = 0; i < mpd; i++)
        h1[i] = vision_gelu_tanh(h1[i]);
    float *out = malloc(mpd * sizeof(float));
    vision_linear_forward(h1, m->mm2, out, M, PD, PD);
    free(mo);
    free(h1);

    *out_embd = out;
    if (out_dim)
        *out_dim = PD;
    return M;
}
