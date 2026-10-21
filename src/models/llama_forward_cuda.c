/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_forward_cuda.c
 * @brief Single-die GPU forward for the plain-Llama arch (MiniCPM5, Llama-3).
 *
 * Runs the compute-heavy parts of pn_llama_forward on the GPU — token embedding
 * lookup, per-layer RMSNorm, Q/K/V/O projections, SwiGLU dense FFN and the LM
 * head — via Foundry's fc_* CUDA kernels with GPU-resident (repacked) weights.
 * RoPE and the GQA-softmax attention stay on the host, bit-identical to the CPU
 * oracle (pn_llama_forward): the small Q/K/V vectors are read back, rotated and
 * attended on the host (fp32 KV cache), and the attention output is uploaded for
 * the output projection. This is the same reuse pattern the Qwen3.5 dense path
 * uses (gpu_matvec + fc_rms_norm_ctx + fc_moe_silu_mul_ctx + fc_add_ctx) — no
 * new attention/FFN math is written here.
 *
 * Scope: non-MoE softmax-attention Llama-family models on ONE die. MoE and the
 * F32-legacy path fall back to the CPU forward (the loader never selects this
 * forward for them).
 *
 * @status Correctness-first GPU forward, validated against the CPU oracle by
 *         temp-0 logit parity.
 * @issues None
 * @todo None
 */

#include "llama.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>
#include <unistd.h>

#include "../accel/simd.h"
#include "../core/engine.h"
#include "../core/qweight.h"
/* Foundry headers resolve through -I$(FOUNDRY), NOT a hardcoded ../../foundry:
 * relative to this file that path is Pantheon/foundry, which exists only where
 * someone has made the in-tree symlink. The K80 test machine builds against the sibling
 * checkout (../foundry) and this was the one file that could not build there. */
#include "cuda/fc_cuda.h"
#include "tensor/tensor.h"
#include "../formats/gguf.h"

#ifdef USE_CUDA

/* One GPU-resident quantized weight: repacked device bytes + shape + the CPU
 * fallback pointer (used only for exotic quant types with no kernel). */
typedef struct {
    fc_buffer *bytes;       /* repacked device bytes (NULL => host fallback) */
    pn_gguf_quant type;
    int rows;
    int cols;
    int row_stride;         /* repacked stride when bytes != NULL */
    const pn_qweight *cpu;  /* host weight (fallback) */
} llama_gpu_weight;

/* Per-layer GPU weights. Norms live on the host (tiny F32 vectors) but are
 * uploaded once so RMSNorm can run entirely on the GPU. */
typedef struct {
    fc_buffer *attn_norm;   /* [n_embd] */
    llama_gpu_weight wq, wk, wv, wo;
    fc_buffer *ffn_norm;    /* [n_embd] */
    llama_gpu_weight w_gate, w_up, w_down;
} llama_gpu_layer;

/* Model-level GPU residency handle (stored on pn_model->gpu). */
typedef struct {
    fc_cuda_ctx *ctx;
    int n_layers;
    llama_gpu_layer *layers;
    fc_buffer *output_norm; /* [n_embd] */
    llama_gpu_weight output;/* LM head */
    int n_embd;
    int n_vocab;
    int loaded;             /* 1 once every required weight is device-resident */
    /* Persistent scratch device buffers (reused every token). */
    fc_buffer *d_hidden;    /* [n_embd] running residual stream */
    fc_buffer *d_normed;    /* [n_embd] */
    fc_buffer *d_q;         /* [n_head*head_dim] */
    fc_buffer *d_k;         /* [n_kv_head*head_dim] */
    fc_buffer *d_v;         /* [n_kv_head*head_dim] */
    fc_buffer *d_attn;      /* [n_head*head_dim] attention output */
    fc_buffer *d_attn_proj; /* [n_embd] */
    fc_buffer *d_gate;      /* [n_ff] */
    fc_buffer *d_up;        /* [n_ff] */
    fc_buffer *d_ffn_out;   /* [n_embd] */
    fc_buffer *d_logits;    /* [n_vocab] */
    /* Host staging for the RoPE + attention round-trip. */
    float *h_q, *h_k, *h_v, *h_attn;
    int q_rows, k_rows, v_rows;
} pn_llama_gpu;

/* ============================================================
 * Upload helpers (mirror gpu_weight_upload / gpu_vec_upload)
 * ============================================================ */

/* Repack + upload one quantized weight to a persistent device buffer. Leaves
 * bytes NULL (host fallback) when the quant type has no repacked GPU form. */
static int llama_gpu_weight_upload(fc_cuda_ctx *ctx, llama_gpu_weight *g,
                                   const pn_qweight *w)
{
    g->cpu = w;
    g->type = w->type;
    g->rows = w->rows;
    g->cols = w->cols;
    g->row_stride = (int)w->row_stride;
    g->bytes = NULL;
    if (!w->data || w->rows <= 0)
        return 0;
    size_t rstride = fc_repack_row_bytes((int)w->type, w->cols);
    if (rstride) {
        size_t rtotal = (size_t)w->rows * rstride;
        unsigned char *tmp = malloc(rtotal);
        if (!tmp)
            return -1;
        if (fc_repack_weight((int)w->type, w->data, tmp, w->rows, w->cols) != 0) {
            free(tmp);
            return -1;
        }
        g->bytes = fc_buffer_from_ptr_ctx(ctx, tmp, rtotal);
        free(tmp);
        g->row_stride = (int)rstride;
        return g->bytes ? 0 : -1;
    }
    if ((int)w->type == GGUF_QUANT_F32) {
        size_t total = (size_t)w->rows * w->row_stride;
        g->bytes = fc_buffer_from_ptr_ctx(ctx, w->data, total);
        return g->bytes ? 0 : -1;
    }
    /* Repackable quant type with an unsupported cols, or an exotic type: keep on
     * the host (llama_gpu_matvec falls back to pn_qmv). */
    return 0;
}

/* Upload one F32 vector (norm) to a persistent device buffer. */
static fc_buffer *llama_gpu_vec_upload(fc_cuda_ctx *ctx, const float *src, int n)
{
    if (!src || n <= 0)
        return NULL;
    return fc_buffer_from_ptr_ctx(ctx, src, (size_t)n * sizeof(float));
}

/* GPU quantized matvec y = W @ x. Dispatches by quant type; falls back to the
 * host pn_qmv (via x_host/y_host staging) for types with no device kernel. */
static void llama_gpu_matvec(fc_cuda_ctx *ctx, const llama_gpu_weight *w,
                             fc_buffer *x, fc_buffer *y,
                             float *x_host, float *y_host)
{
    if (w->bytes) {
        switch ((int)w->type) {
        case GGUF_QUANT_Q3_K:
            fc_matvec_q3k_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        case GGUF_QUANT_Q4_K:
            fc_matvec_q4k_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        case GGUF_QUANT_Q5_K:
            fc_matvec_q5k_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        case GGUF_QUANT_Q6_K:
            fc_matvec_q6k_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        case GGUF_QUANT_Q8_0:
            fc_matvec_q80_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        case GGUF_QUANT_F32:
            fc_matvec_f32_ctx(ctx, w->bytes, x, y, w->rows, w->cols, w->row_stride);
            return;
        default:
            break;
        }
    }
    /* Host fallback (exotic quant only). */
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu(x, x_host);
    pn_qmv(w->cpu, x_host, y_host);
    fc_buffer_from_cpu(y, y_host);
}

/* True if every quantized weight of a layer became device-resident (so the
 * layer runs fully on the GPU with no per-op host round-trip). */
static int layer_weight_ok(const llama_gpu_weight *w)
{
    /* A zero-row weight (absent, e.g. MoE gate/up/down on a dense model) is fine
     * — it is simply never used. A present weight must have device bytes. */
    return w->rows == 0 || w->bytes != NULL;
}

/* ============================================================
 * Upload / teardown
 * ============================================================ */

/* Drop the quantized-weight page cache after the GPU upload. Once every weight
 * is device-resident the host mmap of the quant bytes is dead for the GPU path,
 * but on the 8 GB host that page cache is exactly what drains MemAvailable and
 * trips admission. Advise the kernel to drop it (mirror of qwen35moe's
 * gpu_drop_weight_pages so the dense path gets the same reclaim). Safe: the
 * mapping is MAP_PRIVATE + PROT_READ, so a rare host-fallback matvec (exotic
 * quant with no kernel) re-faults the clean pages straight from the file —
 * correct, just slow. The F32 tok_embd/norms are separate heap copies (NOT in
 * this mmap), so they are untouched. Default-ON (the GPU-resident design) since
 * the host-RAM win + decode-neutrality were validated on a K80 host 2026-07-24 on the
 * MoE path; opt OUT with QWEN35_DROP_WEIGHTS=0 for host-fallback debugging. */
static void llama_gpu_drop_weight_pages(const pn_model *model)
{
    const char *e = getenv("QWEN35_DROP_WEIGHTS");
    if (e && e[0] == '0')
        return;
    const pn_llama_qweights *qw = (const pn_llama_qweights *)model->weights;
    if (!qw)
        return;
    pn_gguf_file *gf = qw->gguf;
    if (!gf || !gf->data || gf->data_size == 0)
        return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0)
        return;
    uintptr_t ds = (uintptr_t)gf->data;
    uintptr_t start = (ds + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1); /* page-align up */
    uintptr_t end = ds + gf->data_size;
    if (end <= start)
        return;
    size_t len = (size_t)(end - start) & ~((size_t)pg - 1); /* whole pages only */
    if (len == 0)
        return;
    if (madvise((void *)start, len, MADV_DONTNEED) == 0)
        fprintf(stderr, "llama: dropped %llu MiB host weight page cache "
                "(MADV_DONTNEED; weights resident in VRAM)\n",
                (unsigned long long)(len >> 20));
    else
        fprintf(stderr, "llama: MADV_DONTNEED on weight pages failed "
                "(errno %d); host cache not dropped\n", errno);
}

int pn_llama_gpu_upload(pn_model *model)
{
    if (!model || !model->weights)
        return -1;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    if (!qw->is_quantized)
        return -1;                     /* F32 legacy path stays on the CPU */
    pn_config *cfg = &model->config;
    if (cfg->n_experts > 0)
        return -1;                     /* MoE not handled here */
    if (!fc_available())
        return 0;                      /* graceful skip (no GPU) */
    if (model->gpu)
        return 0;                      /* idempotent */

    pn_llama_gpu *g = calloc(1, sizeof(*g));
    if (!g)
        return -1;
    /* Honour the explicit placement (`--dies N`). Hard-coding ordinal 0 here
     * only worked when CUDA_VISIBLE_DEVICES happened to pin the die to 0; with
     * the full device list visible it put a model asked for die 6 onto die 0 —
     * a die another serve owns. Unplaced models keep the ordinal-0 default. */
    int die = (model->n_gpu_dies > 0) ? model->gpu_dies[0] : 0;
    g->ctx = fc_cuda_init_device(die);
    if (!g->ctx || !fc_cuda_ctx_available(g->ctx)) {
        fprintf(stderr, "llama: fc_cuda_init_device(%d) failed\n", die);
        free(g);
        return -1;
    }
    fprintf(stderr, "llama: GPU upload -> die %d\n", die);
    g->n_layers = qw->n_layers;
    g->n_embd = cfg->n_embd;
    g->n_vocab = cfg->n_vocab;
    g->layers = calloc(qw->n_layers, sizeof(*g->layers));
    if (!g->layers)
        goto fail;

    int head_dim = pn_head_dim(cfg);
    int q_rows = cfg->n_head * head_dim;
    int kv_rows = cfg->n_kv_head * head_dim;

    for (int i = 0; i < qw->n_layers; i++) {
        pn_llama_qlayer *l = &qw->layers[i];
        llama_gpu_layer *d = &g->layers[i];
        d->attn_norm = llama_gpu_vec_upload(g->ctx, l->attn_norm, cfg->n_embd);
        d->ffn_norm  = llama_gpu_vec_upload(g->ctx, l->ffn_norm, cfg->n_embd);
        if (llama_gpu_weight_upload(g->ctx, &d->wq, &l->wq) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->wk, &l->wk) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->wv, &l->wv) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->wo, &l->wo) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->w_gate, &l->w_gate) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->w_up, &l->w_up) != 0 ||
            llama_gpu_weight_upload(g->ctx, &d->w_down, &l->w_down) != 0)
            goto fail;
        if (!d->attn_norm || !d->ffn_norm ||
            !layer_weight_ok(&d->wq) || !layer_weight_ok(&d->wk) ||
            !layer_weight_ok(&d->wv) || !layer_weight_ok(&d->wo) ||
            !layer_weight_ok(&d->w_gate) || !layer_weight_ok(&d->w_up) ||
            !layer_weight_ok(&d->w_down)) {
            fprintf(stderr, "llama: layer %d has a weight the GPU can't hold; "
                            "aborting GPU upload\n", i);
            goto fail;
        }
    }

    g->output_norm = llama_gpu_vec_upload(g->ctx, qw->output_norm, cfg->n_embd);
    if (llama_gpu_weight_upload(g->ctx, &g->output, &qw->output) != 0)
        goto fail;
    if (!g->output_norm || !layer_weight_ok(&g->output)) {
        fprintf(stderr, "llama: LM head not GPU-resident; aborting GPU upload\n");
        goto fail;
    }

    /* Persistent per-token scratch. */
    g->q_rows = q_rows;
    g->k_rows = kv_rows;
    g->v_rows = kv_rows;
    g->d_hidden    = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_embd * sizeof(float));
    g->d_normed    = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_embd * sizeof(float));
    g->d_q         = fc_buffer_alloc_ctx(g->ctx, (size_t)q_rows * sizeof(float));
    g->d_k         = fc_buffer_alloc_ctx(g->ctx, (size_t)kv_rows * sizeof(float));
    g->d_v         = fc_buffer_alloc_ctx(g->ctx, (size_t)kv_rows * sizeof(float));
    g->d_attn      = fc_buffer_alloc_ctx(g->ctx, (size_t)q_rows * sizeof(float));
    g->d_attn_proj = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_embd * sizeof(float));
    g->d_gate      = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_ff * sizeof(float));
    g->d_up        = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_ff * sizeof(float));
    g->d_ffn_out   = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_embd * sizeof(float));
    g->d_logits    = fc_buffer_alloc_ctx(g->ctx, (size_t)cfg->n_vocab * sizeof(float));
    g->h_q    = malloc((size_t)q_rows * sizeof(float));
    g->h_k    = malloc((size_t)kv_rows * sizeof(float));
    g->h_v    = malloc((size_t)kv_rows * sizeof(float));
    g->h_attn = malloc((size_t)q_rows * sizeof(float));
    if (!g->d_hidden || !g->d_normed || !g->d_q || !g->d_k || !g->d_v ||
        !g->d_attn || !g->d_attn_proj || !g->d_gate || !g->d_up ||
        !g->d_ffn_out || !g->d_logits || !g->h_q || !g->h_k || !g->h_v ||
        !g->h_attn)
        goto fail;

    g->loaded = 1;
    model->gpu = g;
    /* Weights are fully device-resident now — drop the host mmap page cache
     * (gated on QWEN35_DROP_WEIGHTS until validated on a K80 host). */
    llama_gpu_drop_weight_pages(model);
    fprintf(stderr, "llama: on GPU (single die, %d layers)\n", g->n_layers);
    return 0;

fail:
    model->gpu = g;          /* let pn_llama_gpu_free reclaim partial state */
    pn_llama_gpu_free(model);
    return -1;
}

int pn_llama_model_gpu_loaded(const pn_model *model)
{
    if (!model || !model->gpu)
        return 0;
    return ((const pn_llama_gpu *)model->gpu)->loaded;
}

void pn_llama_gpu_free(pn_model *model)
{
    if (!model || !model->gpu)
        return;
    pn_llama_gpu *g = (pn_llama_gpu *)model->gpu;
    if (g->layers) {
        for (int i = 0; i < g->n_layers; i++) {
            llama_gpu_layer *d = &g->layers[i];
            fc_buffer_free(d->attn_norm);
            fc_buffer_free(d->ffn_norm);
            fc_buffer_free(d->wq.bytes);
            fc_buffer_free(d->wk.bytes);
            fc_buffer_free(d->wv.bytes);
            fc_buffer_free(d->wo.bytes);
            fc_buffer_free(d->w_gate.bytes);
            fc_buffer_free(d->w_up.bytes);
            fc_buffer_free(d->w_down.bytes);
        }
        free(g->layers);
    }
    fc_buffer_free(g->output_norm);
    fc_buffer_free(g->output.bytes);
    fc_buffer_free(g->d_hidden);
    fc_buffer_free(g->d_normed);
    fc_buffer_free(g->d_q);
    fc_buffer_free(g->d_k);
    fc_buffer_free(g->d_v);
    fc_buffer_free(g->d_attn);
    fc_buffer_free(g->d_attn_proj);
    fc_buffer_free(g->d_gate);
    fc_buffer_free(g->d_up);
    fc_buffer_free(g->d_ffn_out);
    fc_buffer_free(g->d_logits);
    free(g->h_q);
    free(g->h_k);
    free(g->h_v);
    free(g->h_attn);
    if (g->ctx)
        fc_cuda_ctx_free(g->ctx);
    free(g);
    model->gpu = NULL;
}

/* ============================================================
 * Host RoPE + GQA-softmax attention (bit-parity with the CPU oracle)
 * ============================================================ */

/* Interleaved RoPE (Llama/Qwen2 permuted GGUF layout) — identical math to
 * llama.c apply_rope_raw. MiniCPM5 (PN_ARCH_LLAMA) takes this path. */
static void llama_rope_raw(float *data, int n_heads, int head_dim,
                           int rope_dim, int pos, float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < rot; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* NEOX RoPE (Qwen3 unpermuted GGUF layout) — identical to llama.c
 * apply_rope_neox. */
static void llama_rope_neox(float *data, int n_heads, int head_dim,
                            int rope_dim, int pos, float theta_base)
{
    int rot = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = data + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[i], x1 = head[i + half];
            head[i]        = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* GQA softmax attention for all heads — identical math to llama.c
 * compute_attn_heads (full-precision softmax, weighted V sum). Serial: attention
 * is tiny relative to the projections and this runs on 2 host cores. */
static void llama_attn_heads(const float *q, const float *k_cache,
                             const float *v_cache, float *attn_out,
                             float *scores, int n_head, int n_kv_groups,
                             int head_dim, int max_seq, int seq_len, float scale)
{
    for (int qh = 0; qh < n_head; qh++) {
        int kv_h = qh / n_kv_groups;
        const float *q_vec = q + (size_t)qh * head_dim;
        const float *k_base = k_cache + (size_t)kv_h * max_seq * head_dim;
        const float *v_base = v_cache + (size_t)kv_h * max_seq * head_dim;
        float *out_vec = attn_out + (size_t)qh * head_dim;
        for (int s = 0; s < seq_len; s++)
            scores[s] = pn_dot_f32(q_vec, k_base + (size_t)s * head_dim, head_dim) * scale;
        float max_score = scores[0];
        for (int s = 1; s < seq_len; s++)
            if (scores[s] > max_score) max_score = scores[s];
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        memset(out_vec, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            float w_s = scores[s] * inv_sum;
            const float *v_vec = v_base + (size_t)s * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_vec[d] += w_s * v_vec[d];
        }
    }
}

/* ============================================================
 * Forward
 * ============================================================ */

tm_tensor *pn_llama_forward_cuda(pn_model *model, uint32_t token, pn_state *state)
{
    if (!model || !state || !model->weights || !model->gpu)
        return NULL;
    pn_llama_gpu *g = (pn_llama_gpu *)model->gpu;
    if (!g->loaded)
        return NULL;
    pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    fc_cuda_ctx *ctx = g->ctx;

    int n_embd = cfg->n_embd;
    int head_dim = pn_head_dim(cfg);
    int n_kv_groups = cfg->n_head / cfg->n_kv_head;
    int max_seq = kv->max_seq;
    int pos = state->pos;
    float eps = cfg->rms_norm_eps;

    /* 1. Token embedding -> device hidden. */
    if (state->embd_override)
        fc_buffer_from_cpu(g->d_hidden, state->embd_override);
    else
        fc_buffer_from_cpu(g->d_hidden,
                           qw->tok_embd + (size_t)token * n_embd);

    float *scores = malloc((size_t)max_seq * sizeof(float));
    if (!scores)
        return NULL;

    for (int layer = 0; layer < cfg->n_layer; layer++) {
        llama_gpu_layer *d = &g->layers[layer];

        /* === Attention === */
        fc_rms_norm_ctx(ctx, g->d_hidden, d->attn_norm, g->d_normed, n_embd, eps);
        llama_gpu_matvec(ctx, &d->wq, g->d_normed, g->d_q, g->h_q, g->h_q);
        llama_gpu_matvec(ctx, &d->wk, g->d_normed, g->d_k, g->h_k, g->h_k);
        llama_gpu_matvec(ctx, &d->wv, g->d_normed, g->d_v, g->h_v, g->h_v);

        /* RoPE + GQA softmax on the host (parity with the CPU oracle). */
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu(g->d_q, g->h_q);
        fc_buffer_to_cpu(g->d_k, g->h_k);
        fc_buffer_to_cpu(g->d_v, g->h_v);

        if (cfg->arch == PN_ARCH_QWEN3) {
            llama_rope_neox(g->h_q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            llama_rope_neox(g->h_k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        } else {
            llama_rope_raw(g->h_q, cfg->n_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
            llama_rope_raw(g->h_k, cfg->n_kv_head, head_dim, cfg->rope_dim, pos, cfg->rope_theta);
        }

        float *k_cache_data = (float *)kv->k_cache[layer]->data;
        float *v_cache_data = (float *)kv->v_cache[layer]->data;
        int cache_pos = pos % max_seq;
        for (int h = 0; h < cfg->n_kv_head; h++) {
            memcpy(k_cache_data + ((size_t)h * max_seq + cache_pos) * head_dim,
                   g->h_k + (size_t)h * head_dim, head_dim * sizeof(float));
            memcpy(v_cache_data + ((size_t)h * max_seq + cache_pos) * head_dim,
                   g->h_v + (size_t)h * head_dim, head_dim * sizeof(float));
        }

        int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
        llama_attn_heads(g->h_q, k_cache_data, v_cache_data, g->h_attn, scores,
                         cfg->n_head, n_kv_groups, head_dim, max_seq, seq_len,
                         1.0f / sqrtf((float)head_dim));

        /* Output projection + residual, back on the GPU. */
        fc_buffer_from_cpu(g->d_attn, g->h_attn);
        llama_gpu_matvec(ctx, &d->wo, g->d_attn, g->d_attn_proj,
                         g->h_attn, g->h_attn);
        fc_add_ctx(ctx, g->d_hidden, g->d_attn_proj, g->d_hidden, n_embd);

        /* === SwiGLU FFN === */
        fc_rms_norm_ctx(ctx, g->d_hidden, d->ffn_norm, g->d_normed, n_embd, eps);
        llama_gpu_matvec(ctx, &d->w_gate, g->d_normed, g->d_gate, g->h_q, g->h_q);
        llama_gpu_matvec(ctx, &d->w_up, g->d_normed, g->d_up, g->h_q, g->h_q);
        fc_moe_silu_mul_ctx(ctx, g->d_gate, g->d_up, g->d_gate, cfg->n_ff);
        llama_gpu_matvec(ctx, &d->w_down, g->d_gate, g->d_ffn_out, g->h_q, g->h_q);
        fc_add_ctx(ctx, g->d_hidden, g->d_ffn_out, g->d_hidden, n_embd);
    }

    /* Final norm + LM head. */
    fc_rms_norm_ctx(ctx, g->d_hidden, g->output_norm, g->d_normed, n_embd, eps);
    llama_gpu_matvec(ctx, &g->output, g->d_normed, g->d_logits, g->h_q, g->h_q);

    float *logits_raw = malloc((size_t)cfg->n_vocab * sizeof(float));
    if (!logits_raw) {
        free(scores);
        return NULL;
    }
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu(g->d_logits, logits_raw);
    free(scores);

    int64_t logits_shape[] = { cfg->n_vocab };
    tm_tensor *logits_t = tm_from_data(logits_raw, logits_shape, 1, TM_F32);
    tm_tensor *logits = logits_t ? tm_clone(logits_t) : NULL;
    if (logits_t) tm_free(logits_t);
    free(logits_raw);

    state->pos++;
    return logits;
}

#endif /* USE_CUDA */
