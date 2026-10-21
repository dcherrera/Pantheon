/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_internal.h
 * @brief Internal (cross-file) definitions for the Qwen3.5-MoE port: config,
 *        per-layer type map, block structs, and the per-block forward kernels
 *        shared between qwen35moe.c (load/orchestrate) and the compute files
 *        (qwen35moe_attn.c, later qwen35moe_ssm.c / qwen35moe_moe.c).
 *
 * @status A6: dual-state struct (per-layer SSM state + KV cache + per-token
 *         scratch arena) declared here; forward orchestration in
 *         qwen35moe_forward.c. A5: shared-expert MoE FFN kernel (qwen35moe_moe.c).
 *         A4: SSM (Gated-Delta-Net) block kernel + per-layer SSM state declared
 *         here; A3 full-attention block kernel declared here. Structs shared so
 *         each compute block lives in its own file (whale prevention).
 * @issues None
 * @todo None
 */

#ifndef PN_QWEN35MOE_INTERNAL_H
#define PN_QWEN35MOE_INTERNAL_H

#include "../core/qweight.h"
#include "../formats/gguf.h"
#include "../memory/arena.h"

/* Forward declaration (full type from threadpool/threadpool.h under USE_THREADS;
 * NULL-able so non-threaded builds never need the definition). */
struct fw_pool;

/* ============================================================
 * The matvec seam
 * ============================================================
 *
 * EVERY matmul this architecture does goes through qwen35moe_mv(), and an
 * accelerator backend replaces exactly that and nothing else. There are only
 * thirteen call sites in the whole family, so this seam is small enough to
 * read in one sitting — which is the point: an accelerator arm and the CPU arm
 * then differ in the matmuls ALONE, and the SSM recurrence, the conv1d, the
 * attention softmax and every norm are literally the same code executing the
 * same arithmetic in both. A wrong answer therefore has one candidate family
 * rather than two, which is what makes the first card run of a hybrid model
 * diagnosable at all.
 *
 * `fn` returns 0 when it did the work. Anything else means it did NOT, and the
 * host matvec runs instead. A backend that means to serve a weight must never
 * use that path silently: it is there so an UNBOUND weight is still computed
 * correctly, and every backend here counts its uses, because "the card was
 * quietly not used" and "the card was used" have produced identical bytes in
 * this project before.
 */
typedef struct pn_qwen35moe_mv {
    int (*fn)(void *ctx, const pn_qweight *w, const float *x, float *out);
    void *ctx;
} pn_qwen35moe_mv;

/* ============================================================
 * Config
 * ============================================================ */

/* Hybrid Qwen3.5-MoE dimensions parsed from qwen35moe.* GGUF keys. The engine
 * pn_config carries the shared subset; this struct carries the SSM/MoE/hybrid
 * dims that pn_config has no field for. */
typedef struct {
    int n_layer;                 /* block_count (40) */
    int full_attention_interval; /* attn layer when (i+1) % interval == 0 (4) */
    int n_embd;                  /* embedding_length (2048) */
    int n_head;                  /* attention.head_count (16) */
    int n_head_kv;               /* attention.head_count_kv (2) */
    int head_dim;                /* attention.key_length / value_length (256) */
    int rope_dim;                /* rope.dimension_count — partial RoPE (64) */
    int rope_sections[4];        /* rope.dimension_sections (multi-section RoPE) */
    float rope_theta;            /* rope.freq_base (1e7) */
    float rms_norm_eps;          /* attention.layer_norm_rms_epsilon (1e-6) */
    int n_expert;                /* expert_count (256; 0 for dense Bonsai) */
    int n_expert_used;           /* expert_used_count (8; 0 for dense Bonsai) */
    int expert_ff;               /* expert_feed_forward_length (512) */
    int shared_ff;               /* expert_shared_feed_forward_length (512) */
    float expert_weights_scale;  /* expert_weights_scale (routed weight scale, default 1.0) */
    /* Dense-FFN variant (PrismML Bonsai-27B): general.architecture=qwen35, no
     * MoE. When dense_ffn is set the per-layer FFN is a single SwiGLU
     * (ffn_gate/up/down, dim = ffn_length) loaded into the shared-expert slots;
     * the router / routed experts / shared scalar gate are skipped. Nex-N2-mini
     * (qwen35moe) leaves dense_ffn=0 and runs the routed shared-expert MoE. */
    int dense_ffn;               /* 1 = dense SwiGLU FFN (Bonsai), 0 = MoE (Nex-N2) */
    int ffn_length;              /* feed_forward_length (17408) when dense_ffn */
    int ssm_state;               /* ssm.state_size (128) */
    int ssm_n_group;             /* ssm.group_count (16) */
    int ssm_dt_rank;             /* ssm.time_step_rank (32) */
    int ssm_d_inner;             /* ssm.inner_size (4096) */
    int ssm_conv;                /* ssm.conv_kernel (4) */
    int conv_dim;                /* derived: key*2 + value = 8192 */
    int n_vocab;                 /* from token_embd shape */
    int n_ctx;                   /* context_length */
    int ctx_cap;                 /* KV Phase 1: per-model per-session context cap
                                  * in tokens (KV ring window). 0 = unset → fall
                                  * back to QWEN35_MAX_SEQ / PN_QWEN35MOE_MAX_SEQ.
                                  * Mirrors engine pn_config.ctx_cap; lets ornith
                                  * run 40K without moving saber's window. */
    /* Where this model's matmuls run. Zeroed by the loader's calloc, which is
     * exactly the CPU arm; a backend installs itself at upload time. It lives
     * on the config rather than on the model because the config is the one
     * thing already threaded through every block function. */
    pn_qwen35moe_mv mv;
} pn_qwen35moe_config;

/* The one matvec every block in this family calls. See the seam note above. */
static inline void qwen35moe_mv(const pn_qwen35moe_config *cfg, const pn_qweight *w,
                                const float *x, float *out)
{
    if (cfg && cfg->mv.fn && cfg->mv.fn(cfg->mv.ctx, w, x, out) == 0)
        return;
    pn_qmv(w, x, out);
}

/* Per-layer type: which compute block this layer runs. */
typedef enum {
    PN_QWEN35MOE_LAYER_SSM = 0,   /* Gated-Delta-Net SSM */
    PN_QWEN35MOE_LAYER_ATTN = 1,  /* sparse full-attention */
} pn_qwen35moe_layer_type;

/* ============================================================
 * Block structs
 * ============================================================ */

/* Shared-expert MoE block — present on every layer. Routed experts are sliced
 * from merged 3D tensors (zero-copy mmap); the shared expert is its own SwiGLU
 * with a sigmoid scalar gate. */
typedef struct {
    pn_qweight ffn_gate_inp;       /* router logits [n_expert, n_embd] */
    pn_qweight *gate_exps;         /* per-expert gate [n_expert] (sliced) */
    pn_qweight *up_exps;           /* per-expert up   [n_expert] (sliced) */
    pn_qweight *down_exps;         /* per-expert down [n_expert] (sliced) */
    pn_qweight ffn_gate_shexp;     /* shared-expert gate */
    pn_qweight ffn_up_shexp;       /* shared-expert up */
    pn_qweight ffn_down_shexp;     /* shared-expert down */
    pn_qweight ffn_gate_inp_shexp; /* shared-expert scalar gate [n_embd] */
} pn_qwen35moe_moe;

/* SSM (Gated-Delta-Net) layer: norms + 4 projections + conv + recurrence params
 * + gated-norm + output proj, plus the shared MoE FFN. */
typedef struct {
    float *attn_norm;          /* pre-block RMSNorm [n_embd] (F32, owned) */
    pn_qweight attn_qkv;       /* fused q|k|v stream [conv_dim, n_embd] */
    pn_qweight attn_gate;      /* output gate z [ssm_d_inner, n_embd] */
    float *ssm_conv1d;         /* causal conv1d weight [conv, conv_dim] (F32, owned) */
    float *ssm_a;              /* decay A (= -exp(A_log)) [ssm_dt_rank] (F32, owned) */
    pn_qweight ssm_alpha;      /* alpha proj [ssm_dt_rank, n_embd] */
    pn_qweight ssm_beta;       /* beta proj  [ssm_dt_rank, n_embd] */
    float *ssm_dt;             /* dt bias [ssm_dt_rank] (F32, owned) */
    float *ssm_norm;           /* gated RMSNorm weight [ssm_state] (F32, owned) */
    pn_qweight ssm_out;        /* output proj [n_embd, ssm_d_inner] */
    float *post_attention_norm; /* pre-MoE RMSNorm [n_embd] (F32, owned) */
    pn_qwen35moe_moe moe;
} pn_qwen35moe_ssm_layer;

/* Full-attention layer: norms + Q/K/V projections + QK-RMSNorm + output proj,
 * plus the shared MoE FFN. */
typedef struct {
    float *attn_norm;          /* pre-block RMSNorm [n_embd] (F32, owned) */
    pn_qweight attn_q;         /* joint Q+gate [2*head_dim*n_head, n_embd] */
    pn_qweight attn_k;         /* K proj [head_dim*n_head_kv, n_embd] */
    pn_qweight attn_v;         /* V proj [head_dim*n_head_kv, n_embd] */
    float *attn_q_norm;        /* per-head QK-RMSNorm on Q [head_dim] (F32, owned) */
    float *attn_k_norm;        /* per-head QK-RMSNorm on K [head_dim] (F32, owned) */
    pn_qweight attn_output;    /* output proj [n_embd, head_dim*n_head] */
    float *post_attention_norm; /* pre-MoE RMSNorm [n_embd] (F32, owned) */
    pn_qwen35moe_moe moe;
} pn_qwen35moe_attn_layer;

/* ============================================================
 * Full-attention block (A3) — implemented in qwen35moe_attn.c
 *
 * Per-layer KV cache (the only growing state for attention layers). One of
 * these exists per attention layer in the model. Layout is per-head-major,
 * matching llama.c's pn_llama_kv_cache: [n_head_kv][max_seq][head_dim].
 * ============================================================ */

typedef struct {
    float *k;        /* [n_head_kv * max_seq * head_dim] (owned) — fp32 mode */
    float *v;        /* [n_head_kv * max_seq * head_dim] (owned) — fp32 mode */
    /* TurboQuant Q4 KV (QWEN35_KV_Q4): 4-bit cache, ~4-7x smaller. When q4 is
     * set these hold the cache as [n_head_kv][max_seq][pn_kvq4_row_bytes]
     * ggml-Q4_0 rows and k/v above are NULL. See qwen35moe_kv_q4.h. */
    unsigned char *kq;
    unsigned char *vq;
    /* fp16 sink + recent-window shadow (q4 mode only): the attention sink and
     * the last W tokens are kept at fp16 so greedy argmax stops flipping on
     * near-ties. Sized to PN_KVQ4_SINKMAX / PN_KVQ4_WMAX; runtime W/sink pick
     * how much is live. See qwen35moe_kv_q4.h. */
    uint16_t *k_sink, *v_sink;   /* [n_head_kv][PN_KVQ4_SINKMAX][head_dim] */
    uint16_t *k_win, *v_win;     /* [n_head_kv][PN_KVQ4_WMAX][head_dim] ring */
    int q4;          /* 1 = Q4 cache active (kq/vq), 0 = fp32 (k/v) */
    int n_head_kv;
    int head_dim;
    int max_seq;
    /* B2 — deferred KV host mirror: set when the GPU ring has rows that have
     * not been written back to this struct's k/v (or kq/vq) host buffers.
     * The per-decode-token D2H is deferred; the mirror is refreshed lazily by
     * pn_qwen35moe_gpu_state_to_host() before the GPU buffers are reused for a
     * different session (preserve_live in prefix_cache.c).
     * 0 = host mirror current (post-prefill, post-evict-sync, or no GPU path).
     * 1 = GPU ring has new rows not yet in the host mirror. */
    int kv_mirror_stale;
    /* Phase 10b — content identity generation. GPU ring watermarks
     * (kv_host_id / kv_slot_host_id) key on this struct's buffer POINTER,
     * which survives pn_state_reset and snapshot-restore on a reused serve
     * slot: same pointer, different rows. In lazy-mirror mode the bind path
     * no longer blanket-invalidates those watermarks, so the pointer alone
     * is not enough. content_gen is seeded from a global counter at init
     * (so a freed state's recycled buffer address can never alias) and
     * bumped by state_reset and snapshot-restore; a watermark is valid only
     * when BOTH the pointer and the generation match. */
    unsigned content_gen;
} pn_qwen35moe_kv;

/* Upper bound on head_dim for stack scratch (mini 256, Pro 128). */
#define PN_QWEN35MOE_ATTN_HEAD_DIM_MAX 256

/* Runtime QWEN35_KV_Q4 gate (default OFF; keeps the fp32 KV baseline for A/B).
 * Cached on first read. Defined in qwen35moe_forward.c. */
int pn_qwen35moe_kv_q4_on(void);

/* Phase 10b: next value of the global kv content-generation counter (see
 * pn_qwen35moe_kv.content_gen). Defined in qwen35moe_forward.c. */
unsigned pn_qwen35moe_kv_next_content_gen(void);

/**
 * Run one full-attention block (layers where (i+1)%4==0).
 *
 * Op order (matches the Qwen3-Next reference + llama.cpp build_layer_attn):
 *   1. joint Q+gate projection (attn_q) -> split per head: Q = first head_dim,
 *      gate z = second head_dim.
 *   2. QK-RMSNorm: Q via attn_q_norm, K via attn_k_norm (per-head, pre-RoPE).
 *   3. K/V projections; K/V are n_head_kv heads.
 *   4. partial NEOX RoPE on the first rope_dim dims of every head (Q and K),
 *      remaining dims pass through. Multi-section RoPE collapses to standard
 *      NEOX for text (all sections share the linear token position).
 *   5. GQA attention: each KV head broadcast across n_head/n_head_kv Q heads,
 *      scale 1/sqrt(head_dim). Current token's K/V appended to the cache first.
 *   6. output gate: attn_out *= sigmoid(z).
 *   7. output projection (attn_output).
 *
 * @param layer  Attention-layer weights for this block.
 * @param cfg    Model config (dims, rope params, eps).
 * @param x      attn_norm'd input hidden state [n_embd] (read-only).
 * @param kv     Per-layer KV cache for this attention layer (updated in place).
 * @param pos    Absolute token position (0-based) for RoPE + cache slot.
 * @param arena  Per-token scratch arena (reset by the caller, not here).
 * @param out    Output buffer [n_embd]; residual add is the caller's job.
 * @param pool   Optional thread pool for per-head parallel attention (or NULL).
 * @return       0 on success, negative on failure (allocation).
 *
 * Thread-safe: no (mutates kv and arena)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_attn_block(const pn_qwen35moe_attn_layer *layer,
                            const pn_qwen35moe_config *cfg,
                            const float *x,
                            pn_qwen35moe_kv *kv,
                            int pos,
                            pn_arena *arena,
                            float *out,
                            struct fw_pool *pool);

/**
 * Steps 2-6 of the block above, with the projections already done by the caller.
 *
 * The seam an accelerator backend uses: it runs attn_q / attn_k / attn_v and attn_output on
 * its own device and calls this for the arithmetic between them, so the device arm and the
 * CPU oracle share ONE implementation of the Q/gate de-interleave, the QK-RMSNorm, the
 * partial NEOX RoPE, the KV append, the GQA softmax and the sigmoid output gate. A wrong
 * answer from the device arm therefore has the matmuls as its only candidate.
 *
 * @param qg        attn_q's output, `layer->attn_q.rows` floats. Q and its output gate are
 *                  INTERLEAVED PER HEAD — [h0 Q(head_dim), h0 gate(head_dim), h1 Q, ...].
 * @param k         attn_k's output, `n_head_kv * head_dim` floats. MUTATED in place: normed,
 *                  rotated, then appended to the cache.
 * @param v         attn_v's output, `n_head_kv * head_dim` floats. Appended to the cache.
 * @param attn_out  receives `n_head * head_dim` floats — attn_output's input.
 * @return          0 on success, negative on failure (allocation).
 *
 * Thread-safe: no (mutates kv and arena)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_attn_core(const pn_qwen35moe_attn_layer *layer,
                           const pn_qwen35moe_config *cfg,
                           const float *qg, float *k, float *v,
                           pn_qwen35moe_kv *kv,
                           int pos,
                           pn_arena *arena,
                           float *attn_out,
                           struct fw_pool *pool);

/* ============================================================
 * SSM (Gated-Delta-Net) block (A4) — implemented in qwen35moe_ssm.c
 *
 * Per-layer recurrent state. Unlike the attention KV cache, this is a fixed-size
 * state that is updated in place every token (no growth with sequence length).
 * One of these exists per SSM layer in the model.
 *
 * conv_state — the last (ssm_conv - 1) timesteps of the full conv_dim-channel
 *   q|k|v stream, used to prime the causal conv1d for the next token. Layout is
 *   channel-major: conv_state[c * (ssm_conv-1) + t], t=0 oldest .. (conv-2)
 *   newest. Total (ssm_conv-1) * conv_dim floats (e.g. 3 * 8192 = 24576).
 *
 * ssm_state — the recurrent matrix state S, one [ssm_state x ssm_state] matrix
 *   per V-head (ssm_dt_rank heads). Stored TRANSPOSED to match the recurrence's
 *   access pattern (and llama.cpp's gated_delta_net layout): for head h, key dim
 *   i, value dim j,
 *       ssm_state[ h*ssm_state*ssm_state + j*ssm_state + i ] == S_h[i][j].
 *   So row j (contiguous, length ssm_state) holds column j of S, indexed by the
 *   key dim i — making the per-token dot products over i unit-stride. Total
 *   ssm_state * ssm_state * ssm_dt_rank floats (e.g. 128*128*32 = 524288).
 * ============================================================ */

typedef struct {
    float *conv_state;   /* [(ssm_conv-1) * conv_dim] (owned) */
    float *ssm_state;    /* [ssm_state * ssm_state * ssm_dt_rank] (owned, transposed) */
    int conv_dim;
    int conv_minus_1;    /* ssm_conv - 1 (conv_state timesteps) */
    int state_size;      /* ssm_state (per-head matrix is state_size x state_size) */
    int n_v_heads;       /* ssm_dt_rank */
} pn_qwen35moe_ssm_state;

/**
 * Run one Gated-Delta-Net SSM block (layers where (i+1)%4 != 0).
 *
 * Op order (matches docs/architecture/qwen35moe.md SSM section + llama.cpp
 * build_layer_attn_linear / gated_delta_net autoregressive kernel):
 *   1. Projections from the attn_norm'd input: attn_qkv -> fused q|k|v
 *      [conv_dim]; attn_gate -> z [ssm_d_inner]; ssm_beta -> beta[n_v_heads]
 *      then beta = sigmoid(beta); ssm_alpha -> alpha[n_v_heads].
 *   2. Decay gate (scalar per V-head): g = ssm_a * softplus(alpha + ssm_dt).
 *      ssm_a is pre-stored as -exp(A_log) (negative) -> plain multiply.
 *   3. Causal conv1d (k=ssm_conv) over the full conv_dim q|k|v stream primed
 *      with conv_state -> SiLU. Split q/k [state*n_k_heads], v [state*n_v_heads];
 *      L2-normalize q,k over head dim; GQA-repeat q,k from n_k_heads->n_v_heads.
 *   3b. conv_state is updated in place to the newest (ssm_conv-1) timesteps.
 *   4. Per-token recurrence (per V-head, read-AFTER-write, 1/sqrt(state) scale
 *      applied once); ssm_state updated in place.
 *   5. Output: gated = RMSNorm(y, ssm_norm) * silu(z); cur = ssm_out * gated.
 *
 * @param layer  SSM-layer weights for this block.
 * @param cfg    Model config (dims, eps).
 * @param x      attn_norm'd input hidden state [n_embd] (read-only).
 * @param state  Per-layer recurrent SSM state (conv + ssm; updated in place).
 * @param arena  Per-token scratch arena (reset by the caller, not here).
 * @param out    Output buffer [n_embd]; residual add is the caller's job.
 * @param pool   Optional thread pool for per-head parallel recurrence (or NULL).
 * @return       0 on success, negative on failure (allocation/validation).
 *
 * Thread-safe: no (mutates state and arena)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_ssm_block(const pn_qwen35moe_ssm_layer *layer,
                           const pn_qwen35moe_config *cfg,
                           const float *x,
                           pn_qwen35moe_ssm_state *state,
                           pn_arena *arena,
                           float *out,
                           struct fw_pool *pool);

/**
 * Steps 2-5 of the block above, with the four input projections already done by the caller
 * and the output projection left to it.
 *
 * The accelerator seam, and the reason it exists is a control rather than tidiness: the
 * conv1d, the L2 norms, the interleaved GQA repeat, the delta-rule recurrence and the gated
 * norm are ONE implementation shared by the CPU oracle and every device arm, so a device
 * arm that disagrees with the oracle disagrees about the MATMULS and about nothing else.
 *
 * `state` is mutated in place — both the conv window and the transposed recurrent matrix —
 * so a device arm that calls this is carrying the same state the oracle carries, and a
 * comparison at token 48 is a statement about the STATE rather than about one step.
 *
 * @param qkv       attn_qkv's output, `conv_dim` floats (the fused q|k|v stream)
 * @param z         attn_gate's output, `ssm_state * ssm_dt_rank` floats
 * @param beta_raw  ssm_beta's output, `ssm_dt_rank` floats, BEFORE the sigmoid
 * @param alpha     ssm_alpha's output, `ssm_dt_rank` floats
 * @param gated     receives `ssm_state * ssm_dt_rank` floats — ssm_out's input
 * @return          0 on success, negative on failure (allocation/validation).
 *
 * Thread-safe: no (mutates state and arena)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_ssm_recurrence(const pn_qwen35moe_ssm_layer *layer,
                                const pn_qwen35moe_config *cfg,
                                const float *qkv, const float *z,
                                const float *beta_raw, const float *alpha,
                                pn_qwen35moe_ssm_state *state,
                                pn_arena *arena,
                                float *gated,
                                struct fw_pool *pool);

/* ------------------------------------------------------------
 * The three pieces of the recurrence that are NOT the two card phases
 * ------------------------------------------------------------
 * The Gated-Delta-Net op on the card owns `conv1d + SiLU` and the delta rule. The decay gate
 * (a `softplus`, which the SFPU has no `log` for), the q/k L2 norm and the output gated norm
 * stay on the host, and `qwen35moe_gdn_tt.c` runs exactly these three around the two
 * descriptors. They are exported rather than copied for the same reason
 * `pn_qwen35moe_ssm_recurrence` exists: a second implementation of an L2 norm is a second
 * answer, and the oracle is the one that wins.
 */

/** `g[h] = ssm_a[h] * softplus(alpha[h] + ssm_dt[h])`, the pre-exp decay gate per V-head. */
void pn_qwen35moe_decay_gate(const pn_qwen35moe_ssm_layer *layer, const float *alpha,
                             int n_v_heads, float *g_out);

/** `x[d] /= max(sqrt(sum x^2), eps)`, in place over one head. L2, NOT RMS: no mean factor,
 *  and the clamp is `max(.,eps)` rather than `+eps` under the root, matching ggml_l2_norm.
 *  The sum accumulates in double on purpose — this feeds the recurrence directly. */
void pn_qwen35moe_l2_norm(float *data, int head_dim, float eps);

/** `gated = RMSNorm(y, ssm_norm) * silu(z)`, per V-head over `ssm_state` elements. */
void pn_qwen35moe_gated_norm(const pn_qwen35moe_ssm_layer *layer,
                             const pn_qwen35moe_config *cfg, const float *y, const float *z,
                             float *gated);

/** `beta = sigmoid(beta_raw)` per V-head — the one line of the recurrence that is neither a
 *  card phase nor a norm, exported so the two arms cannot spell it differently. */
void pn_qwen35moe_beta_sigmoid(const float *beta_raw, int n_v_heads, float *beta_out);

/**
 * `conv_out = SiLU(conv1d(qkv))` over the whole `conv_dim` stream, and ROLLS `state->conv_state`.
 *
 * PHASE CONV's oracle, exported for one reason: the on-card GDN parity arm needs the host's
 * answer to the SAME sub-step the card ran, and computing it any other way would compare the
 * card against a second implementation instead of against the oracle.
 *
 * **It MUTATES the window.** A caller running it for comparison and then running the whole
 * recurrence over the same state must restore `state->conv_state` in between, or the window
 * rolls twice for one token.
 *
 * @param qkv       `conv_dim` floats, the fused projection
 * @param state     the layer's carried state; `conv_state` is read and rolled
 * @param conv_out  `conv_dim` floats, written
 */
void pn_qwen35moe_conv1d_silu(const pn_qwen35moe_ssm_layer *layer,
                              const pn_qwen35moe_config *cfg, const float *qkv,
                              pn_qwen35moe_ssm_state *state, float *conv_out);

/* ============================================================
 * Shared-expert MoE FFN (A5) — implemented in qwen35moe_moe.c
 *
 * Runs on EVERY layer (both SSM and attention) on the post_attention_norm'd
 * hidden state. Extends the Mixtral-style routed-MoE pattern with two
 * qwen35moe specifics: softmax + norm_topk_prob gating, and an always-on
 * sigmoid-gated shared expert.
 * ============================================================ */

/**
 * Run one shared-expert MoE FFN block (every layer).
 *
 * Op order (matches docs/architecture/qwen35moe.md "Shared-expert MoE FFN"):
 *   1. Routed: logits = ffn_gate_inp . x  [n_expert] -> softmax -> top-k
 *      (n_expert_used) -> norm_topk_prob (renormalize the selected weights to
 *      sum to 1) -> scale by expert_weights_scale. For each selected expert e,
 *      SwiGLU_e(x) = (silu(gate_e . x) * (up_e . x)) . down_e.
 *      moe_out = sum_e weight_e * SwiGLU_e(x).
 *   2. Shared (always on): shexp = (silu(gate_shexp . x) * (up_shexp . x)) .
 *      down_shexp; sgate = sigmoid(ffn_gate_inp_shexp . x) [1 scalar];
 *      shexp *= sgate.
 *   3. Combine: out = moe_out + shexp. (Caller adds the residual.)
 *
 * @param moe    MoE-block weights for this layer (router + experts + shared).
 * @param cfg    Model config (n_expert, n_expert_used, ffs, scale).
 * @param x      post_attention_norm'd hidden state [n_embd] (read-only).
 * @param arena  Per-token scratch arena (reset by the caller, not here).
 * @param out    Output buffer [n_embd]; residual add is the caller's job.
 * @param pool   Optional thread pool (unused for now; accepted for symmetry).
 * @return       0 on success, negative on failure (allocation/validation).
 *
 * Thread-safe: no (mutates arena)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_moe_block(const pn_qwen35moe_moe *moe,
                           const pn_qwen35moe_config *cfg,
                           const float *x,
                           pn_arena *arena,
                           float *out,
                           struct fw_pool *pool);

/* ============================================================
 * Dual state + forward orchestration (A6) — qwen35moe_forward.c
 *
 * Mirrors rwkvx.c's dual top-level state: parallel per-type arrays (one
 * pn_qwen35moe_ssm_state per SSM layer, one pn_qwen35moe_kv per attention
 * layer), indexed by the model's per-type block_index, plus a per-token
 * scratch arena and two ping-pong hidden-state buffers.
 * ============================================================ */

/* Upper bound on the sparse top-k survivor count tracked per state below.
 * Must be >= foundry's FC_TOPK_MAX (checked with a static assert in
 * qwen35moe_forward_cuda.c); kept independent so this header stays free of
 * the CUDA backend headers. */
#define PN_QWEN35MOE_TOPK_MAX 256

typedef struct {
    pn_qwen35moe_ssm_state *ssm;   /* [n_ssm] per-SSM-layer recurrent state */
    pn_qwen35moe_kv *kv;           /* [n_attn] per-attention-layer KV cache */
    int n_ssm;
    int n_attn;
    pn_arena *arena;               /* per-token scratch (reset each token) */
    float *x_norm;                 /* [n_embd] pre-block normed hidden state */
    float *block_out;              /* [n_embd] block/MoE output before residual */
    float *logits;                 /* [n_vocab] owned output buffer */
    int n_embd;
    int n_vocab;

    /* Sparse top-k logits presentation bookkeeping (GPU path, see
     * gpu_fetch_logits in qwen35moe_forward_cuda.c).  `logits` is maintained
     * SPARSELY: -inf everywhere except this step's k survivors, refilled in
     * full only once and thereafter patched incrementally.  This tracking
     * describes THIS state's `logits` buffer, so it must live with it: an
     * earlier version kept it on the shared per-model GPU context keyed by the
     * buffer's ADDRESS, which silently aliased whenever a freed state's
     * address was handed back to a new state — the full refill was skipped and
     * the previous sequence's survivors (EOS included) leaked into the new
     * sequence's step-0 logits.  Zeroed by the calloc in
     * pn_qwen35moe_state_new, which is exactly right: a fresh state's `logits`
     * has not been -inf-filled yet. */
    int topk_filled;               /* 1 = `logits` is -inf-filled and the
                                    * prev-ids below describe its finite entries */
    int topk_prev_n;               /* survivors written by the previous fetch */
    int topk_prev_ids[PN_QWEN35MOE_TOPK_MAX];

    /* Constrained decoding needs the WHOLE logits row.  The die normally
     * reduces the LM head to the top-k (id, logit) pairs so only k pairs cross
     * the 0.4 GB/s host link; everything else arrives as -inf.  A grammar mask
     * has to be able to force a token the model did NOT rank in its top-k — the
     * whole point of constraining — so a request decoding under a grammar sets
     * this and the fetch sends the full row for as long as it is live. */
    int full_logits;
} pn_qwen35moe_state;

/* Opaque model handle — full definition in qwen35moe.c, also typedef'd in
 * qwen35moe.h (C11 permits the identical re-typedef when both are included). */
typedef struct pn_qwen35moe_model pn_qwen35moe_model;

/**
 * Run the full hybrid forward for one token (single-token decode/prefill).
 * embed -> per-layer (attn_norm -> SSM|attn block -> +residual ->
 * post_attention_norm -> shared-MoE -> +residual) -> output_norm -> LM head.
 *
 * @param model  Loaded qwen35moe model (weights + config + index maps).
 * @param token  Input token id.
 * @param state  Dual state (updated in place); pos drives RoPE + caches.
 * @param pos    Absolute token position (0-based).
 * @return       Logits buffer [n_vocab] owned by state, or NULL on failure.
 *
 * Thread-safe: no (mutates state)
 * Reentrant: no
 * Signal-safe: no
 */
float *pn_qwen35moe_forward_token(pn_qwen35moe_model *model, uint32_t token,
                                  pn_qwen35moe_state *state, int pos);

#ifdef USE_CUDA
/* Batched GPU prefill (qwen35moe_forward_cuda.c): n tokens of one sequence,
 * positions pos0..pos0+n-1, in one pass. Returns state->logits holding the
 * LAST position's logits, or NULL if the batched path cannot run (caller
 * falls back to per-token forwards). all_logits (optional, [n * n_vocab])
 * receives every position's full logits. Bit-exact vs n sequential
 * pn_qwen35moe_forward_token_cuda calls (logits, KV cache, SSM state). */
float *pn_qwen35moe_forward_batch_tokens_cuda(pn_qwen35moe_model *m,
                                              const uint32_t *tokens, int n,
                                              pn_qwen35moe_state *s, int pos0,
                                              float *all_logits);

/* Speculative greedy verify (qwen35moe_forward_cuda.c): run tokens[0..n)
 * (committed pending token + drafts) at positions pos0..pos0+n-1 through the
 * batched forward with per-position SSM state snapshots, filling
 * argmax_out[t] with each position's exact greedy argmax (device top-k(1) —
 * identical tokens to the single-token path). Commits NOTHING; refuses
 * pos0 + n > the KV ring size. Returns 0 on success, -1 if the batched path
 * cannot run (caller falls back to plain single-token decode). */
int pn_qwen35moe_spec_verify_tokens_cuda(pn_qwen35moe_model *m,
                                         const uint32_t *tokens, int n,
                                         pn_qwen35moe_state *s, int pos0,
                                         int32_t *argmax_out);

/* Commit the first n_accept (in [1, n]) positions of the LAST verify:
 * restores every SSM layer's conv + recurrent state from its snapshot
 * n_accept-1 (no-op when n_accept == n). The KV cache needs no rollback —
 * rejected entries sit past the committed position and are never read.
 * The caller advances its position bookkeeping by n_accept. */
int pn_qwen35moe_spec_commit_cuda(pn_qwen35moe_model *m, int n_accept);

/* Multi-user decode (qwen35moe_forward_cuda.c): bind user state s to decode
 * slot `slot` of n_slots (snapshots the live GPU SSM state D2D into the
 * slot — call immediately after that user's prefill), then step all K users
 * one token each in ONE batched pass. tokens[u]/pos[u] are user u's pending
 * token and position; each user's logits land in sts[u]->logits. Bit-exact
 * per user vs decoding alone. Returns 0 on success. */
int pn_qwen35moe_decode_slot_bind_cuda(pn_qwen35moe_model *m,
                                       pn_qwen35moe_state *s, int slot,
                                       int n_slots);
/* Rung 2 (reactive ladder): the current memory-pressure-capped decode
 * concurrency ceiling (full PN_QWEN35MOE_GPU_MAX_SLOTS at level OK, halved at
 * TIGHT, 1 at CRITICAL). A batched-decode scheduler should size its slot count
 * to this; bind refuses anything above it. */
int pn_qwen35moe_gpu_ladder_max_slots(void);
int pn_qwen35moe_decode_slots_tokens_cuda(pn_qwen35moe_model *m,
                                          const uint32_t *tokens,
                                          pn_qwen35moe_state *const *sts,
                                          const int *pos, int nslots);

/* Session-state lifetime hook. The GPU layers hold raw back-pointers into a
 * session's state (per-slot kv_slot_kv / kv_slot_host_id, per-SSM-layer
 * host_state). Call this while `s` is still fully alive — pn_qwen35moe_state_free
 * does, as its first act — and every live GPU context drops its references, so
 * no dangling back-pointer can outlive the state. Safe with no GPU loaded.
 *
 * This is why the bind path needs no "is my back-pointer still valid?" test:
 * an invalid one cannot exist. A test that answered by reading the suspect
 * object would be reading freed memory to decide whether it was freed. */
void pn_qwen35moe_gpu_forget_state_cuda(const pn_qwen35moe_state *s);

/* Instrument for the lifetime regression test: how many per-slot kv
 * back-pointers currently land inside [base, base+bytes). Compares addresses
 * only and never dereferences, so it is valid to pass the (already freed)
 * range of a state's kv array. Must be 0 after that state is freed. */
int pn_qwen35moe_gpu_slot_kv_refs_cuda(const void *base, size_t bytes);
#endif

/* Accessors into the opaque model, used by qwen35moe_forward.c (it needs the
 * config, per-type layer counts, index maps, per-type weight arrays, and the
 * model-level embed/norm/head tensors without seeing the struct layout). */
const pn_qwen35moe_config *pn_qwen35moe_model_config(const pn_qwen35moe_model *m);
int pn_qwen35moe_model_n_ssm(const pn_qwen35moe_model *m);
int pn_qwen35moe_model_n_attn(const pn_qwen35moe_model *m);
/* The KV window this MODEL's ring is sized to: cfg->ctx_cap when set, else the
 * global QWEN35_MAX_SEQ. Exported because the TT backend has to size the CARD's
 * KV regions to the same number at arm time, and a second copy of that rule
 * would be a cache the card and the host disagree about the length of. */
int pn_qwen35moe_max_seq(const pn_qwen35moe_config *cfg);
pn_qwen35moe_layer_type pn_qwen35moe_model_layer_type(const pn_qwen35moe_model *m,
                                                      int layer);
int pn_qwen35moe_model_layer_index(const pn_qwen35moe_model *m, int layer);
const pn_qwen35moe_ssm_layer *pn_qwen35moe_model_ssm_layer(
    const pn_qwen35moe_model *m, int idx);
const pn_qwen35moe_attn_layer *pn_qwen35moe_model_attn_layer(
    const pn_qwen35moe_model *m, int idx);
const float *pn_qwen35moe_model_token_embd(const pn_qwen35moe_model *m);
/* Write embedding row `token` into dst[n_embd]. Uses the F32 heap if present
 * (tied/fallback), else dequantizes the row on demand from the mmap (A2c). */
void pn_qwen35moe_model_embed_row(const pn_qwen35moe_model *m, uint32_t token,
                                  float *dst);
const float *pn_qwen35moe_model_output_norm(const pn_qwen35moe_model *m);
const pn_qweight *pn_qwen35moe_model_output(const pn_qwen35moe_model *m);

/* MUTABLE weight accessors. A GPU backend has to WRITE one field of every
 * weight (pn_qweight.gpu, its device-buffer handle), which the const accessors
 * above cannot express — and const-casting their result is a lie about
 * ownership that the compiler can no longer check. Same objects, same
 * lifetime; only the constness differs. */
pn_qwen35moe_ssm_layer *pn_qwen35moe_model_ssm_layer_mut(pn_qwen35moe_model *m,
                                                         int idx);
pn_qwen35moe_attn_layer *pn_qwen35moe_model_attn_layer_mut(pn_qwen35moe_model *m,
                                                           int idx);
pn_qweight *pn_qwen35moe_model_output_mut(pn_qwen35moe_model *m);

/* The config, mutable — a backend installs its matvec seam through this, and
 * the seam is a field of the config for the reason given at its definition. */
pn_qwen35moe_config *pn_qwen35moe_model_config_mut(pn_qwen35moe_model *m);

/* The still-open GGUF the model's weights point into (mmap backing). Lets a
 * backend resolve a weight's bytes back to the tensor that contains them —
 * expert slices are interior offsets into one merged tensor, so pointer
 * containment is the only way to recover tensor identity. NULL if the model
 * was not built from a GGUF. */
pn_gguf_file *pn_qwen35moe_model_gguf(const pn_qwen35moe_model *m);

/* ============================================================
 * Weight enumeration
 *
 * ONE walk over every pn_qweight the model owns, so the "set of weights this
 * model has" is stated in a single place. Every backend that must touch all of
 * them (Metal upload, Metal release, the kernel-coverage gate) goes through
 * this; a private copy of the walk in each of them is how a newly added weight
 * gets uploaded by one and forgotten by another.
 * ============================================================ */

typedef struct {
    pn_qweight *w;      /* the weight itself (mutable: backends attach .gpu) */
    const char *name;   /* e.g. "blk.12.ffn_gate_exps[7]"; valid for this call */
    int layer;          /* global layer index, or -1 for a model-level weight */
    int expert;         /* expert slice index, or -1 when not an expert slice */
} pn_qwen35moe_weight_ref;

typedef void (*pn_qwen35moe_weight_fn)(const pn_qwen35moe_weight_ref *ref,
                                       void *user);

/**
 * Invoke fn once per pn_qweight the model owns, in layer order.
 *
 * @param m     Loaded model (no-op if NULL)
 * @param fn    Callback (no-op if NULL)
 * @param user  Opaque pointer handed back to the callback
 *
 * Thread-safe: no (the callback may mutate the weights)
 * Reentrant: no
 * Signal-safe: no
 */
void pn_qwen35moe_for_each_weight(pn_qwen35moe_model *m,
                                  pn_qwen35moe_weight_fn fn, void *user);

/* Opaque GPU-residency handle accessors (qwen35moe.c stores the pointer; the
 * concrete pn_qwen35moe_gpu struct lives in qwen35moe_forward_cuda.c). */
void *pn_qwen35moe_model_gpu(const pn_qwen35moe_model *m);
void  pn_qwen35moe_model_set_gpu(pn_qwen35moe_model *m, void *gpu);

/* Explicit die placement from pn_load_on (Phase 7): stored on the model before
 * the GPU upload; gpu_die_ids()/gpu_split_layer() prefer it over the
 * QWEN35_GPU_DIE_A/B env knobs. n == 0 clears (env/defaults apply). */
void pn_qwen35moe_model_set_dies(pn_qwen35moe_model *m, const int *dies, int n);
int  pn_qwen35moe_model_dies(const pn_qwen35moe_model *m, const int **dies_out);

#ifdef USE_CUDA
/* ============================================================
 * Single-die GPU forward path (A7 scaffold) — qwen35moe_forward_cuda.c
 *
 * Mirrors the CPU forward (qwen35moe_forward.c) but dispatches the per-layer
 * sandwich to Foundry's fc_* CUDA kernels with GPU-resident weights, dual
 * state, and scratch. Single die only; the 2-die layer-split swap is a stubbed
 * seam (see qwen35moe_forward_cuda.c). Compile-checks on the Mac via dlopen
 * (no GPU); validated layer-by-layer against the CPU reference on a K80 host.
 * ============================================================ */

/**
 * Upload every layer's Q3_K weight bytes + F32 norms/router params to
 * persistent fc_buffers and attach the resulting pn_qwen35moe_gpu handle to the
 * model (via pn_qwen35moe_model_set_gpu). Idempotent: a no-op if already
 * uploaded or if fc_available() is false. The CPU weights are kept live.
 *
 * @param model  Loaded qwen35moe model (CPU weights resident).
 * @return       0 on success (or graceful skip when no GPU), -1 on failure.
 */
int pn_qwen35moe_gpu_upload(pn_qwen35moe_model *model);

/* Free the GPU-residency handle and all its fc_buffers (detaches it from the
 * model). Safe to call with a model that was never uploaded. */
void pn_qwen35moe_gpu_free(pn_qwen35moe_model *model);

/**
 * Single-die GPU forward for one token. Mirrors pn_qwen35moe_forward_token:
 * embed -> per-layer (attn_norm -> SSM|attn block -> +residual ->
 * post_attention_norm -> shared-MoE -> +residual) -> output_norm -> LM head,
 * dispatched to fc_* kernels with GPU-resident weights + dual state. Reads the
 * final logits back to host (state->logits) and returns them.
 *
 * Requires pn_qwen35moe_gpu_upload() to have attached a GPU handle to the model.
 * Reuses the host-side dual state (pn_qwen35moe_state) for KV/SSM bookkeeping
 * and shadows the GPU-resident copies inside the GPU handle.
 *
 * @return  state->logits [n_vocab], or NULL on failure / no GPU handle.
 */
float *pn_qwen35moe_forward_token_cuda(pn_qwen35moe_model *model, uint32_t token,
                                       pn_qwen35moe_state *state, int pos);
#endif /* USE_CUDA */

#ifdef USE_METAL
/* ============================================================
 * Metal (Apple GPU) forward path — qwen35moe_forward_metal.c
 *
 * Same seam as the CUDA path above: a token-level entry the pn_forward_fn
 * adapter (pn_qwen35moe_forward_metal, declared in qwen35moe.h) wraps. Unlike
 * the CUDA one it returns a STATUS, because in Phase 0 "I did not run" is a
 * distinct outcome from "I ran and failed" — the adapter turns the former into
 * a CPU token and the latter into an error.
 * ============================================================ */

enum {
    PN_QWEN35MOE_METAL_OK       =  0,
    PN_QWEN35MOE_METAL_ERROR    = -1,
    PN_QWEN35MOE_METAL_ENOTIMPL = -2,  /* no kernel path yet — use the CPU forward */
};

/**
 * Run one token through the Metal forward.
 *
 * PHASE 0: unimplemented on purpose. Returns PN_QWEN35MOE_METAL_ENOTIMPL
 * without touching model or state, so no half-built GPU path can corrupt the
 * dual state or produce plausible-looking wrong logits. Phases 1-5 fill it in.
 *
 * @param model       Loaded model with Metal weight residency attached.
 * @param token       Input token id.
 * @param state       Dual state (SSM/KV), updated in place once implemented.
 * @param pos         Absolute token position (0-based).
 * @param logits_out  Out: logits [n_vocab] owned by state, NULL on non-OK.
 * @return            PN_QWEN35MOE_METAL_OK / _ENOTIMPL / _ERROR.
 *
 * Thread-safe: no (mutates state)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_forward_token_metal(pn_qwen35moe_model *model, uint32_t token,
                                     pn_qwen35moe_state *state, int pos,
                                     float **logits_out);
#endif /* USE_METAL */

#endif /* PN_QWEN35MOE_INTERNAL_H */
