/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_ENGINE_H
#define PN_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include "../../include/pantheon.h"
#include "../models/gpu_attn_path_tracker.h"
#include "../models/gpu_graph_capture_tracker.h"
#include "tensor/tensor.h"
#include "tokenizer/tokenizer.h"

/*
 * Internal engine types — not part of public API.
 */

/* Forward declarations */
struct pn_arena;
#ifdef USE_THREADS
struct fw_pool;
#endif

/* Supported model architectures */
typedef enum {
    PN_ARCH_LLAMA,
    PN_ARCH_QWEN2,
    PN_ARCH_PHI,
    PN_ARCH_MISTRAL,
    PN_ARCH_MIXTRAL,
    PN_ARCH_RWKVX,
    PN_ARCH_QWEN35MOE,
    PN_ARCH_QWEN3,        /* appended to keep existing enum values stable */
    PN_ARCH_BONSAI,       /* PrismML Bonsai-27B: qwen35-family hybrid, dense SwiGLU FFN + native ternary (own forward entry, shares qwen35moe blocks) */
    PN_ARCH_UNKNOWN,
} pn_arch;

/* Model modality — what kind of input/output the model handles */
typedef enum {
    PN_MODALITY_TEXT,
    PN_MODALITY_AUDIO,
    PN_MODALITY_VISION,
    PN_MODALITY_MULTIMODAL,
} pn_modality;

/* Model configuration (parsed from GGUF metadata or safetensors header) */
typedef struct {
    pn_arch arch;
    pn_modality modality;
    int n_vocab;
    int n_embd;
    int n_head;
    int n_kv_head;
    int head_dim;               /* per-head dim; 0 = derive n_embd/n_head (Qwen3 decouples it) */
    int n_layer;
    int n_ff;
    int n_ctx;
    float rope_theta;
    float rms_norm_eps;
    int sliding_window;         /* 0 = full context, >0 = circular KV cache window */
    int rope_dim;               /* RoPE rotation dim (0 = full head_dim) */
    int qk_norm;                /* 1 = per-head RMSNorm on Q and K before RoPE (Qwen3) */
    int n_experts;              /* 0 for non-MoE, 8 for Mixtral */
    int n_experts_used;         /* top-k experts per token (2 for Mixtral) */
    int ctx_cap;                /* KV Phase 1: per-model per-session context cap
                                 * in tokens (the KV ring / admission window).
                                 * 0 = unset → fall back to QWEN35_MAX_SEQ /
                                 * PN_SESSION_MAX_CTX. Set per model at load so
                                 * ornith can run a 40K window without changing
                                 * saber's (they share the global env otherwise). */
} pn_config;

/* Effective per-head dimension (explicit head_dim, else derived). */
static inline int pn_head_dim(const pn_config *c) {
    return c->head_dim > 0 ? c->head_dim : (c->n_head ? c->n_embd / c->n_head : 0);
}

/* Forward function pointer — each architecture implements one */
typedef tm_tensor* (*pn_forward_fn)(pn_model *model, uint32_t token, pn_state *state);

/* Max dies one model can be explicitly placed on (pn_load_on).
 *
 * 16, matching PN_DM_MAX_DIES / PN_MEM_MAX_DIES (the K80 test machine has 16 GK210
 * dies) and PN_DIE_LAYER_SPLIT_MAX_DIES (what the layer partitioner will
 * plan). These four are the same number on purpose: a placement the die
 * manager accepts must be one the split planner can plan and the memwatch
 * state can account for, or a width gets bound in one layer and refused (or
 * worse, silently narrowed) in another. It was 8 while the CUDA forward only
 * ever split two ways. */
#define PN_MODEL_MAX_DIES 16

/* Model struct (concrete definition, opaque to public API) */
struct pn_model {
    pn_config config;
    void *weights;              /* Architecture-specific weight struct */
    pn_forward_fn forward;      /* Architecture's forward pass */
    ft_tokenizer *tokenizer;    /* BPE tokenizer */
    char *path;                 /* Model file path */
    int chat_style;             /* pn_chat_style enum */
    int chat_add_bos;           /* template prepends bos_token (e.g. MiniCPM5) */
    void *vision;               /* pn_vision_model* for VLMs (NULL otherwise) */
#ifdef USE_THREADS
    struct fw_pool *pool;       /* Thread pool for parallel ops */
#endif
    int gpu_dies[PN_MODEL_MAX_DIES]; /* explicit die placement (pn_load_on) */
    int n_gpu_dies;                  /* 0 = none (env/defaults apply)        */
#if defined(USE_CUDA) || defined(USE_TT)
    void *gpu;                  /* pn_llama_gpu* for the plain-Llama CUDA
                                   forward, or pn_llama_tt* for the Tenstorrent
                                   one (NULL until the matching _upload).
                                   qwen35moe/bonsai keep their handle inside
                                   their own model struct instead.
                                   ONE slot on purpose: a model has ONE
                                   accelerator residency, even in a build that
                                   carries both families. */
#endif
};

/* State struct (concrete definition, opaque to public API) */
struct pn_state {
    void *kv_cache;             /* Architecture-specific KV cache */
    int pos;                    /* Current position in sequence */
    pn_config *config;          /* Reference to model config */
    struct pn_arena *arena;     /* Per-token scratch arena */
    const float *embd_override; /* if set, forward uses this embedding (image token) */
};

/* Attach a vision encoder (clip mmproj GGUF) to a loaded model, for pn_describe. */
int pn_attach_vision(pn_model *model, const char *mmproj_path);

/* Get forward function for an architecture */
pn_forward_fn pn_get_forward_fn(pn_arch arch);

/**
 * Snapshot `model`'s CUDA-graph residency — the graphs_live/graphs_total gauge.
 *
 * A capture that fails degrades to the direct kernel body with CORRECT output,
 * so the only symptom of losing a layer's graph is that the serve is ~1.8
 * ms/step slower across 40 layers. This is the surface that makes that visible,
 * and it separates the three states that look alike from the outside:
 * captured (fine), failed (a real deficit), ineligible (correct behaviour —
 * a non-Q3_K MoE never captures at all).
 *
 * @param model  Any loaded model; architectures with no graph path report 0
 * @param out    Zeroed and filled; `all.total == 0` means "no graph slots"
 * @return       0 if the model has a graph path, -1 otherwise (`out` still
 *               zeroed, so a caller may ignore the return)
 *
 * Thread-safe: yes, lock-free (see pn_qwen35moe_gpu_graph_residency)
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_model_graph_residency_read(const pn_model *model,
                                  pn_model_graph_residency *out);

/**
 * Snapshot which attention path this model's attention layers are on.
 *
 * Same shape and same contract as pn_model_graph_residency_read, for the same
 * reason: an on-die-attention loss returns correct output and reports nothing,
 * so the counter is the only observable.
 *
 * @param model  Any loaded model; architectures with no on-die attention path
 *               report 0
 * @param out    Zeroed and filled; `all.total == 0` means "never evaluated"
 * @return       0 if the model has an instrumented attention path, -1 otherwise
 *               (`out` still zeroed, so a caller may ignore the return)
 *
 * Thread-safe: yes, lock-free (see pn_qwen35moe_gpu_attn_residency)
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_model_attn_residency_read(const pn_model *model,
                                 pn_model_attn_residency *out);

/* Detect model architecture from file */
pn_arch pn_detect_arch(const char *path);

/* Detect file format (GGUF vs safetensors) */
bool pn_is_gguf(const char *path);
bool pn_is_safetensors(const char *path);

/* Load options for pn_load_ex() */
typedef struct {
    const char *vocab_path;   /* RWKV World tokenizer (required for safetensors) */
    const char *config_path;  /* HuggingFace config.json (for Qwen/LLaMA safetensors) */
    bool quantize;            /* INT8 quantize at load time */
    int  ctx_cap;             /* KV Phase 1: per-model per-session context cap in
                               * tokens (KV ring window). 0 = unset. Set BEFORE
                               * weight upload so the ring is sized per model. */
} pn_load_opts;

/* Extended model loading with options (safetensors + GGUF) */
pn_model* pn_load_ex(const char *path, const pn_load_opts *opts);

/* Placement-aware load (Phase 7 die-placement API): as pn_load_ex, but binds
 * the model to an explicit die-set BEFORE any GPU weight upload, so the
 * multi-model host can place disjoint die-sets per model. The placement is
 * recorded on the model (gpu_dies/n_gpu_dies) on every path; today only the
 * qwen35moe CUDA forward consumes it (N dies = an N-way layer split, layers
 * divided as evenly as possible with the LM head on the LAST die;
 * QWEN35_GPU_SPLIT_LAYER still tunes the single boundary of a 2-die split).
 * Explicit placement wins over the QWEN35_GPU_DIE_A/B env knobs. Pass
 * dies == NULL / n_dies == 0 for exact pn_load_ex behavior. Placement is
 * bookkeeping-only without CUDA (the CPU forward ignores it). */
pn_model* pn_load_on(const char *path, const pn_load_opts *opts,
                     const int *dies, int n_dies);

/* GPT-2 byte encoding/decoding (used by chat module for BPE models) */
char* gpt2_bytes_encode(const char *input);
char* gpt2_bytes_decode(const char *input);

/* ============================================================
 * Lossless speculative decode (single sequence, greedy)
 * ============================================================ */

/* Run ONE greedy speculative-decode step for a single sequence.
 *
 * Preconditions: state->pos = N, the KV cache holds committed positions 0..N-1,
 * and `seed` is the last already-emitted token that has NOT yet been written to
 * the cache (it occupies position N). draft[0..K-1] are the draft model's
 * guesses for the K tokens that follow `seed` (K may be 0).
 *
 * Runs one pn_llama_verify pass over [seed, draft[0..K-1]] (M = K+1 tokens) and
 * emits the EXACT greedy continuation of `seed`: with g_i = argmax(logits[i]),
 * it appends g_0 (the true next token), accepts draft[i] while draft[i] == g_i,
 * and at the first mismatch appends the correction g_i and stops; if all K draft
 * tokens are accepted it appends the free bonus g_K. Every appended token is the
 * argmax of the model's own logits for the true accepted prefix, so the emitted
 * stream is bit-identical to plain greedy decoding no matter how good or bad the
 * draft is (a bad draft just yields fewer tokens per pass, never a wrong token).
 *
 * Writes the emitted tokens into out[] (needs room for K+1) and advances
 * state->pos to cover `seed` plus every accepted draft token (their K/V is
 * already committed by the verify pass); the final emitted token (correction or
 * bonus) is left uncommitted and becomes the next call's `seed`. Returns the
 * number of emitted tokens (>= 1), or -1 if the model is unsupported. */
int pn_spec_decode_greedy(pn_model *model, pn_state *state, uint32_t seed,
                          const uint32_t *draft, int K, uint32_t *out);

/* pn_draft_fn (the external draft-provider seam) is declared in pantheon.h and
 * available here via the include above; the batched speculative decoder and the
 * single-sequence pn_spec_generate share that one typedef. */

/* Speedup accounting for a speculative generation run. */
typedef struct {
    long verify_passes;   /* speculative forward passes actually run */
    long tokens_emitted;  /* generated tokens (== plain-decode forward count) */
    long drafted;         /* draft tokens proposed */
    long accepted;        /* draft tokens accepted */
    int  fell_back;       /* 1 if the run dropped to plain decode (low acceptance) */
} pn_spec_stats;

/* Lossless greedy speculative generation for a single sequence.
 *
 * Preconditions: `state` is already prefilled (state->pos = context length) and
 * `seed` is the first generated token, greedily sampled from the prefill logits
 * (argmax), not yet written to the cache. This is exactly the state right after
 * vlm_prefill in the OCR path.
 *
 * Each cycle asks draft_fn for up to K candidate tokens for the slots after the
 * accepted prefix, runs one pn_spec_decode_greedy pass, and appends the exact
 * greedy continuation. Because every emitted token is the model's own argmax and
 * a draft is only accepted when it already equals that argmax, the produced
 * stream is bit-identical to plain greedy decode for ANY draft (see the tests).
 * The draft only changes how many tokens each verify pass yields, hence the
 * speedup: verify_passes <= tokens_emitted, and plain decode would need exactly
 * tokens_emitted forwards.
 *
 * Per-page fallback: after `probe_cycles` speculative cycles, if the running
 * acceptance fraction is below `min_accept_frac` (a structure-heavy page where
 * the draft rarely matches), the run drops to plain single-token decode for the
 * remainder so speculation never costs more than it saves. Pass K<=0 to force
 * plain decode from the start.
 *
 * Stops at `stop_id` (e.g. EOS / <|im_end|>; pass -1 for none) or max_tokens.
 * The stop token is NOT emitted, matching the plain greedy decode loop. Writes
 * generated ids into out[] (capacity >= max_tokens) and returns the count, or -1
 * if the model is not on the quantized verify path. Fills *stats if non-NULL. */
int pn_spec_generate(pn_model *model, pn_state *state, uint32_t seed,
                     int max_tokens, int K, int stop_id,
                     pn_draft_fn draft_fn, void *draft_ud,
                     double min_accept_frac, int probe_cycles,
                     uint32_t *out, pn_spec_stats *stats);

/* ============================================================
 * Sampler seam (for external decode drivers, e.g. the serve layer)
 * ============================================================ */

/* The grammar mask's shortlist decision on its own (grammar_mask_shortlist.h):
 * with params->tool_grammar as the grammar, rewrite `logits` to the masked
 * survivors when the row's top entries provably decide the draw.  Returns 1
 * when rewritten (sample it with the grammar hook removed and the same
 * dense_row setting), 0 when undecidable (row untouched).  Lets a speculative
 * verify decide a masked row from its top-N exactly as plain decode would.
 *
 * Thread-safe: yes (no shared state) */
int pn_sampler_grammar_shortlist(float *logits, int n_vocab, const pn_sampling_params *params,
                                 const uint32_t *recent_tokens, int n_recent);

/* Public wrapper around the internal token sampler so external decode drivers
 * reuse the EXACT sampling path (repeat penalty -> temperature -> top-k ->
 * top-p -> cumulative draw; temperature <= 0 is greedy argmax) instead of
 * reimplementing it. Mutates logits[] in place exactly as the built-in generate
 * loop does. recent_tokens[0..n_recent) is the repeat-penalty ring (may be NULL
 * with n_recent == 0). Returns the sampled token id.
 *
 * Thread-safe: no (uses a shared per-vocab scratch buffer)
 * Reentrant: no */
uint32_t pn_sample_next_token(float *logits, int n_vocab,
                              pn_sampling_params *params,
                              uint32_t *recent_tokens, int n_recent);

/* Sparse-survivor sampler fast path (PN_SPARSE_SAMPLER) — see engine.c.
 * Covers BOTH decode modes: the temperature>0 sampler and the temperature<=0
 * greedy argmax, off the same gather and the same flag.
 *
 * set/clear override the env in process, so a gate can flip arms between two
 * calls on the same logits row.
 *
 * pn_sampler_sparse_hits() counts how many times the fast path actually RAN,
 * not how many times it was enabled.  A gate MUST assert this advances: the
 * previous attempt at this optimisation was silently inert (its row test
 * `x > -INFINITY` is folded to a constant true by -ffast-math, so every row
 * fell through to the slow path) and its bit-exactness gate still passed 18/18
 * — comparing the old path against itself. */
void pn_sampler_set_sparse(int on);
void pn_sampler_clear_sparse(void);
unsigned long pn_sampler_sparse_hits(void);
/** Speculative sampling's chooser state (pn_spec_sample_choose, a pn_spec_choose_fn): the
 *  request's params, an n_vocab row buffer, a copy of the repeat window advanced per draw,
 *  and the verify row being chosen (reset to 0 before each cycle). */
typedef struct pn_spec_sample_ctx {
    pn_sampling_params *params;
    float *row;
    int n_vocab;
    uint32_t *recent;
    int n_recent, max_recent;
    int row_index;
} pn_spec_sample_ctx;
/** Equal to qwen35moe.h's PN_SPEC_CHOOSE_DEFER. */
#define PN_SPEC_CHOOSE_DEFER_TOKEN 0xFFFFFFFFu
/** One verify row's draw: the row as plain sparse decode would hand the sampler, drawn
 *  under the plain repeat window; PN_SPEC_CHOOSE_DEFER_TOKEN when a row past 0 cannot be
 *  decided exactly by its top-N. */
uint32_t pn_spec_sample_choose(void *ctx, const uint32_t *index, const float *value,
                               uint32_t count);
/** Top-N rows the sampler swapped for the dense row (pn_sampling_params.dense_row). */
unsigned long pn_sampler_topn_refills(void);
/** Can this top-N row (-INFINITY outside N entries) decide a sampled draw exactly as the
 *  dense row would (the top-k set after penalties lies inside the N)? Dense rows: true. */
bool pn_sampler_topn_row_exact(const float *logits, int n_vocab,
                               const pn_sampling_params *params,
                               const uint32_t *recent, int n_recent);

/* Grammar-mask density invariant — see the full-logits arming-window block in
 * serve_generation.c.
 *
 * pn_sampler_grammar_mask_steps() counts steps that APPLIED a grammar mask.
 * pn_sampler_grammar_sparse_rows() counts the subset of those whose row was
 * still the device's SPARSE top-k presentation — steps where the mask could
 * only choose among ~k tokens, so a grammar whose legal tokens were all pruned
 * has zero survivors and fails open (the response_format defect).
 *
 * THE SPARSE COUNT MUST STAY ZERO.  It is the load-bearing invariant of the
 * arming window: arming the full row one token late drops the mask on exactly
 * the step that engages the grammar.  Both counters are process-wide across
 * threads (relaxed increments — they are a tripwire, not an accounting record),
 * so a whole serve run is checkable with one number, and a gate can prove its
 * own detector fires by running the deliberately-late arm
 * (PN_GRAMMAR_ARM_STALE=1). */
unsigned long pn_sampler_grammar_mask_steps(void);
/** Masked steps decided from the shortlist / scanned in full, and the wall
 *  seconds spent inside the mask (refill + shortlist + scan, not the draw). */
unsigned long pn_sampler_grammar_shortlist_steps(void);
unsigned long pn_sampler_grammar_fullscan_steps(void);
/** Masked steps applied by fetching only the dense chunks that hold an allowed
 *  token (pn_sampling_params.dense_chunks) instead of the whole row. */
unsigned long pn_sampler_grammar_chunk_steps(void);
double pn_sampler_grammar_mask_seconds(void);
unsigned long pn_sampler_grammar_sparse_rows(void);

/* ============================================================
 * Async forward-step seam (C2 — 2 cores drive 8 cards)
 *
 * Architecture-agnostic wrapper over an OPTIONAL non-blocking single-token
 * step: begin launches the token's GPU work and returns immediately; poll
 * reaps completion (CUDA event query — never blocks). While one card's
 * step is in flight the calling thread sleeps, so concurrent generations
 * on other cards keep the host's 2 cores instead of serializing behind
 * blocking stream syncs. Today only the qwen35moe CUDA forward implements
 * it; every other model (and any non-eligible token) reports "unsupported"
 * and the caller uses model->forward exactly as before.
 * ============================================================ */

/* Launch one token's forward without blocking.
 * Returns 0 = in flight (caller MUST pn_forward_step_poll to completion
 * before any other forward on this model), 1 = unsupported/not eligible
 * (nothing launched — call model->forward instead), -1 = error. */
int pn_forward_step_begin(pn_model *model, uint32_t token, pn_state *state);

/* Reap the in-flight step. Returns 1 = still running (sleep briefly and
 * poll again), 0 = done (*logits_out = owned logits tensor, state->pos
 * advanced — same contract as model->forward), -1 = error. */
int pn_forward_step_poll(pn_model *model, pn_state *state,
                         tm_tensor **logits_out);

#endif /* PN_ENGINE_H */
