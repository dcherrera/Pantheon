/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_LLAMA_H
#define PN_LLAMA_H

#include "../../include/pantheon.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "../formats/gguf.h"

/*
 * Llama model architecture.
 *
 * Covers: Llama 2, Llama 3, Llama 3.1/3.2, CodeLlama, Yi, Solar.
 *
 * Architecture:
 *   embed -> for each layer:
 *     RMSNorm -> Q/K/V proj -> RoPE -> GQA -> O proj -> residual ->
 *     RMSNorm -> SwiGLU FFN (gate*silu(up), down) -> residual
 *   -> RMSNorm -> LM head -> logits
 *
 * Two weight representations:
 *   1. F32 (legacy): all weights dequantized to tm_tensor (pn_llama_weights)
 *   2. Quantized: weights stay in mmap'd quant form (pn_llama_qweights)
 */

/* ============================================================
 * F32 weight structs (legacy, Phase 1)
 * ============================================================ */

/* Llama layer weights (F32) */
typedef struct {
    tm_tensor *attn_norm;       /* RMSNorm weight [n_embd] */
    tm_tensor *wq;              /* Q projection [n_head * head_dim, n_embd] */
    tm_tensor *wk;              /* K projection [n_kv_head * head_dim, n_embd] */
    tm_tensor *wv;              /* V projection [n_kv_head * head_dim, n_embd] */
    tm_tensor *wo;              /* Output projection [n_embd, n_head * head_dim] */
    tm_tensor *ffn_norm;        /* RMSNorm weight [n_embd] */
    tm_tensor *w_gate;          /* Gate projection [n_ff, n_embd] */
    tm_tensor *w_up;            /* Up projection [n_ff, n_embd] */
    tm_tensor *w_down;          /* Down projection [n_embd, n_ff] */
} pn_llama_layer;

/* Llama model weights (F32) */
typedef struct {
    tm_tensor *tok_embd;        /* Token embeddings [n_vocab, n_embd] */
    tm_tensor *output_norm;     /* Final RMSNorm [n_embd] */
    tm_tensor *output;          /* LM head [n_vocab, n_embd] (may share with tok_embd) */
    pn_llama_layer *layers;     /* Per-layer weights [n_layer] */
    int n_layers;
    pn_gguf_file *gguf;         /* Kept open for on-the-fly dequant */
} pn_llama_weights;

/* ============================================================
 * Quantized weight structs (Phase 2)
 * ============================================================ */

/* MoE expert weights (one expert's FFN) */
typedef struct {
    pn_qweight w_gate;          /* Gate projection [n_ff, n_embd] */
    pn_qweight w_up;            /* Up projection [n_ff, n_embd] */
    pn_qweight w_down;          /* Down projection [n_embd, n_ff] */
} pn_moe_expert;

/* Llama layer with quantized projections + F32 norms */
typedef struct {
    float *attn_norm;           /* RMSNorm weight [n_embd] (F32, owned) */
    pn_qweight wq;              /* Q projection (quantized, mmap) */
    pn_qweight wk;              /* K projection (quantized, mmap) */
    pn_qweight wv;              /* V projection (quantized, mmap) */
    pn_qweight wo;              /* Output projection (quantized, mmap) */
    float *bq, *bk, *bv, *bo;  /* Attention biases (NULL for Llama, F32 owned) */
    float *attn_q_norm;         /* Qwen3 per-head Q RMSNorm [head_dim] (F32 owned, NULL otherwise) */
    float *attn_k_norm;         /* Qwen3 per-head K RMSNorm [head_dim] (F32 owned, NULL otherwise) */
    float *ffn_norm;            /* RMSNorm weight [n_embd] (F32, owned) */
    pn_qweight w_gate;          /* Gate projection (quantized, mmap) — zeroed for MoE */
    pn_qweight w_up;            /* Up projection (quantized, mmap) — zeroed for MoE */
    pn_qweight w_down;          /* Down projection (quantized, mmap) — zeroed for MoE */
    /* MoE fields (NULL/zeroed for non-MoE models) */
    pn_qweight ffn_gate_inp;    /* Router [n_experts, n_embd] */
    pn_moe_expert *experts;     /* Array of n_experts (NULL for non-MoE) */
} pn_llama_qlayer;

/* Llama model with quantized weights */
typedef struct {
    int is_quantized;           /* Always 1, used to distinguish from pn_llama_weights */
    float *tok_embd;            /* Token embeddings [n_vocab * n_embd] (F32, owned) */
    float *output_norm;         /* Final RMSNorm [n_embd] (F32, owned) */
    pn_qweight output;          /* LM head (quantized or F32, mmap or owned) */
    float *output_f32;          /* LM head F32 if needed for tied embeddings (owned) */
    pn_llama_qlayer *layers;    /* Per-layer quantized weights [n_layer] */
    int n_layers;
    int n_embd;
    int n_vocab;
    pn_gguf_file *gguf;         /* Kept open (mmap backing) */
} pn_llama_qweights;

/* ============================================================
 * Llama KV cache
 * ============================================================ */

typedef struct {
    tm_tensor **k_cache;        /* Key cache per layer [n_layer][n_kv_head, max_seq, head_dim] */
    tm_tensor **v_cache;        /* Value cache per layer [n_layer][n_kv_head, max_seq, head_dim] */
    int n_layers;
    int max_seq;
    /* WHERE THE LIVE KV ACTUALLY IS.
     *
     * NULL — and it stays NULL on every path but one — means the tensors above hold it and a
     * snapshot is a host memcpy. NON-NULL means the cache lives on an accelerator: the
     * Tenstorrent backend running ON-CARD ATTENTION keeps it in card DRAM as bf16, the
     * tensors above are never written, and a snapshot that read them anyway would export a
     * page of zeros and call it a session.
     *
     * It is a FUNCTION POINTER rather than a device handle for two reasons that both matter.
     * llama_state_snapshot.c then needs no accelerator header and links into every build,
     * including the ones with no card support compiled in; and the off-hardware suite can
     * install a FAKE mover, so the card branch is covered by a test that can go red instead
     * of by an `#ifdef` nothing exercises.
     *
     * Set by the FORWARD that owns the cache, not by whoever allocated it — the forward is
     * the only code that knows which arm is live. */
    void *kv_device;
    /* Move one layer's K (`values` 0) or V (`values` 1) between the device and a packed
     * [head][position][dim] fp32 array of `position_count` positions. `capture` non-zero is
     * device -> host; zero is host -> device. Returns 0, or non-zero on a failed transfer —
     * which the caller must NOT absorb. */
    int (*kv_move)(void *device, int layer, int values, int position_count, float *rows,
                   int capture);
} pn_llama_kv_cache;

/* ============================================================
 * Forward passes
 * ============================================================ */

/* Forward pass — quantized weights + arena (Phase 2, preferred) */
tm_tensor* pn_llama_forward(pn_model *model, uint32_t token, pn_state *state);

/* Forward pass — F32 weights (Phase 1 legacy) */
tm_tensor* pn_llama_forward_f32(pn_model *model, uint32_t token, pn_state *state);

#ifdef USE_CUDA
/* ============================================================
 * Single-die GPU forward (plain-Llama arch: MiniCPM5, Llama-3, etc.)
 *
 * Mirrors pn_llama_forward but runs the compute-heavy projections, norms,
 * SwiGLU FFN and LM head on the GPU via Foundry's fc_* kernels with
 * GPU-resident (repacked) weights, keeping RoPE + GQA-softmax attention on the
 * host (bit-parity with the CPU oracle; the KV cache stays fp32 host-side).
 * Reuses the SAME quant matvec / RMSNorm / SwiGLU kernels the Qwen3.5 CUDA
 * forward uses — no new attention/FFN math. Non-MoE softmax layers only.
 * ============================================================ */

/**
 * Repack + upload every quantized projection (q/k/v/o + gate/up/down + LM head)
 * and the F32 norms/embeddings of a loaded pn_llama_qweights to persistent
 * device buffers, attaching the GPU handle to the model. Idempotent; a no-op
 * (leaving the model un-uploaded) when fc_available() is false or a weight's
 * quant type has no GPU kernel. The CPU weights are kept live (host fallback).
 *
 * @param model  Loaded pn_model whose weights are pn_llama_qweights.
 * @return       0 on success (or graceful skip when no GPU), -1 on failure.
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
int pn_llama_gpu_upload(pn_model *model);

/* 1 if the model has a live GPU-resident weight handle attached. */
int pn_llama_model_gpu_loaded(const pn_model *model);

/* Free the GPU handle + all its device buffers (detaches from the model). */
void pn_llama_gpu_free(pn_model *model);

/**
 * Single-die GPU forward for one token. Requires pn_llama_gpu_upload() to have
 * attached a GPU handle. Returns a freshly-allocated logits tm_tensor [n_vocab]
 * (caller owns it), or NULL on failure / no GPU handle. Advances state->pos.
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
tm_tensor* pn_llama_forward_cuda(pn_model *model, uint32_t token,
                                 pn_state *state);
#endif /* USE_CUDA */

#ifdef USE_TT
/* ============================================================
 * Tenstorrent Blackhole forward (plain-Llama arch)
 *
 * Same shape as the CUDA path — the device does the matrices, the host keeps
 * RoPE, KV append and GQA-softmax for bit-parity with the CPU oracle — with two
 * differences. Weights come from a converted FTTW artifact resident in the
 * card's GDDR6 and NOT from the GGUF directly; and the norms, residual adds and
 * SwiGLU gate stay on the host as well. src/accel/tt/pn_tt_device.h states which
 * capability gaps force that and what closes them.
 *
 * THE ARTIFACT IS PRODUCED AND CACHED FROM THE `.gguf`, NOT ASKED FOR. Tensix
 * eats 32x32 tiles in four 16x16 faces and the matmul kernels read a block-float
 * the unpacker decodes natively, so a GGUF can never be uploaded as it stands —
 * but that conversion is a shader cache, not a question. `--model x.gguf` on its
 * own converts once, caches beside the checkpoint, and is instant thereafter.
 * PN_TT_ARTIFACT still names an artifact to use INSTEAD, as an override.
 * ============================================================ */

/**
 * Open the card, make the checkpoint's matmul weights resident, and attach the
 * handle to the model.
 *
 * Converts the model's `.gguf` into a card-ready artifact if no valid cached one
 * is beside it, choosing the precision rung from what fits — never asking. A
 * first load of a large model prints a progress bar; later loads print one line
 * naming the rung and the fit.
 *
 * Refuses — leaving the model CPU-resident — when the weights are F32 or MoE,
 * the card is absent, or the model does not fit at any rung the card can load.
 * A refusal for not fitting carries the arithmetic; it is never a silent
 * fallback to a cheaper rung than the one it reports.
 *
 * @param model  Loaded pn_model whose weights are pn_llama_qweights and whose
 *               `path` is the source checkpoint.
 * @return       0 on success, -1 on refusal or failure.
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
int pn_llama_tt_upload(pn_model *model);

/* 1 if the model has a live Tenstorrent residency attached. */
int pn_llama_model_tt_loaded(const pn_model *model);

/* Release the residency and detach from the model. Never parks the core. */
void pn_llama_tt_free(pn_model *model);

/**
 * One decode step with every projection on the card. Returns a freshly
 * allocated logits tm_tensor [n_vocab] (caller owns it), or NULL on failure — a
 * failed matmul is never absorbed into a plausible answer. Advances state->pos.
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
tm_tensor* pn_llama_forward_tt(pn_model *model, uint32_t token, pn_state *state);

/**
 * BATCHED multi-token prefill on the card — the same contract as pn_llama_prefill, on the
 * TT route.
 *
 * Ingests n_tokens positions starting at state->pos, advances state->pos by n_tokens and
 * returns the LAST position's logits (caller owns the tm_tensor), or NULL — WITHOUT having
 * mutated state — when the batched path refuses, in which case the caller runs its ordinary
 * per-token loop. Refusals: fewer than 2 tokens, PN_TT_PREFILL=0, on-card attention (its
 * rotary, KV append and flash decode each take ONE position), any of the diagnostic arms
 * (PN_TT_TRACE / ATTN_CHECK / ROPE_CHECK / ROW_CENSUS, all of which read row 0 as if the
 * other 31 were padding), or a prompt that would wrap the KV ring.
 *
 * A tile is 32 rows and a matvec is Mt=1, so the card already multiplies 32 rows against
 * every weight tile it reads and 31 of them are zero padding. This fills them with prompt
 * positions: same descriptor, same weight traffic, same reader floor, 32 tokens. The LAST
 * token is deliberately NOT batched — it goes through pn_llama_forward_tt, so the LM head,
 * the on-card top-N and the sparse-logit presentation are the shipping code unmodified.
 *
 * Thread-safe: no  Reentrant: no  Signal-safe: no
 */
tm_tensor* pn_llama_prefill_tt(pn_model *model, const uint32_t *tokens,
                               const float *const *embd_rows, int n_tokens, pn_state *state);
#endif /* USE_TT */

/* Batched multi-token prefill (quantized, non-MoE Llama/Qwen path).
 *
 * Processes n_tokens positions in one shot using GEMM projections, producing
 * exactly the same KV-cache state and final-token logits as running n_tokens
 * sequential pn_llama_forward() calls. Advances state->pos by n_tokens and
 * returns the logits for the LAST position (caller owns the tm_tensor).
 *
 * For each position i: if embd_rows && embd_rows[i] != NULL, that [n_embd]
 * vector is used as the input embedding (image-token override); otherwise
 * tokens[i] is looked up in the token-embedding table.
 *
 * Returns NULL (without mutating state) if the model is not supported by the
 * batched path (e.g. MoE, F32 legacy weights, ternary). Callers should fall
 * back to the per-token loop in that case. */
tm_tensor* pn_llama_prefill(pn_model *model, const uint32_t *tokens,
                            const float *const *embd_rows, int n_tokens,
                            pn_state *state);

/**
 * Finish one position's q/k/v projections for layer `l`, before RoPE: add the
 * optional attention biases (Qwen2) and apply the per-head Q/K RMSNorm (Qwen3).
 * A no-op for plain Llama. Every forward (CPU, CUDA, Tenstorrent) calls THIS, so
 * a backend that reads q/k/v back to the host cannot skip a step the CPU oracle
 * takes. `q` is [n_head*head_dim], `k`/`v` are [n_kv_head*head_dim], in place.
 *
 * Thread-safe: yes (touches only its arguments)  Reentrant: yes  Signal-safe: no
 */
void pn_llama_qkv_finish(const pn_llama_qlayer *l, const pn_config *cfg,
                         float *q, float *k, float *v);

/* Batched single-token decode across B independent sequences (quantized,
 * non-MoE Llama/Qwen path).
 *
 * Decodes ONE token for each of `batch` sequences in a single M=batch GEMM
 * pass, so every weight is dequantized once and applied to all B rows — the
 * same amortization prefill uses, now for autoregressive decode. Each
 * states[b] supplies sequence b's own KV cache and position (the sequences are
 * fully independent: different lengths and positions are fine, and each
 * attends only its own cache). For row b: if embd_rows && embd_rows[b], that
 * [n_embd] vector is the input embedding; otherwise tokens[b] is looked up.
 *
 * Writes [batch, n_vocab] row-major logits into out_logits (caller-allocated)
 * and advances each states[b]->pos by one. Returns 0 on success, or -1 without
 * mutating any state if the model/quant is unsupported (caller falls back to
 * per-sequence pn_llama_forward). */
int pn_llama_forward_batch(pn_model *model, const uint32_t *tokens,
                           const float *const *embd_rows, pn_state **states,
                           int batch, float *out_logits);

/* Ragged batched speculative-decode verify (multi-sequence, quantized non-MoE
 * path). Verifies n_seqs INDEPENDENT sequences in one forward pass: block b is
 * n_tok[b] candidate tokens for sequence states[b], starting at states[b]->pos.
 * `tokens` is the concatenation of all blocks (M = sum n_tok[b] rows, grouped by
 * sequence, block b occupying rows off[b]..off[b]+n_tok[b]-1).
 *
 * Structured as pn_llama_forward_batch's body with P = M: every projection, FFN
 * and the LM head is a single M=P pn_qmm (each weight row dequantized once and
 * dotted against every activation row independently), so a given output row's
 * logits are bit-identical whether it is verified alone or alongside any other
 * rows or sequences. Only the attention pre-processing (bias, per-head QK-norm,
 * RoPE at each row's true position, KV write into that row's own cache) and the
 * per-query causal attention are ragged: row r uses sequence row_seq[r]'s cache,
 * position row_pos[r] = states[b]->pos + (r - off[b]), and attends exactly its
 * committed prefix 0..states[b]->pos-1 plus its block's rows 0..(r-off[b]) via a
 * seq_len = states[b]->pos + (r-off[b]) + 1 mask. It therefore never sees any
 * other sequence's cache or any row after it. K/V for every row IS written into
 * its sequence's cache at its position, but NO states[b]->pos is advanced (free
 * per-slot rollback: the caller commits accepted rows by advancing pos, and the
 * uncommitted slots are overwritten by the next real write and never read).
 *
 * Writes [M, n_vocab] row-major logits into out_logits (caller-owned); row r
 * predicts the token AFTER candidate r. Returns 0 on success, or -1 (without
 * mutating any state) if the model/quant is unsupported, M > PN_QMM_MAX_B, or any
 * block would wrap its cache (states[b]->pos + n_tok[b] > that state's kv
 * max_seq). Callers fall back to per-sequence pn_llama_verify in that case. */
int pn_llama_verify_batch(pn_model *model, const uint32_t *tokens,
                          pn_state **states, const int *n_tok, int n_seqs,
                          float *out_logits);

/* Speculative-decode verify pass (single sequence, quantized non-MoE path).
 *
 * Runs ONE batched forward over n_tokens candidate tokens starting at the
 * sequence's current position state->pos, using the exact same M=P machinery as
 * pn_llama_prefill (panel-path projections + FFN GEMMs, causal flash attention
 * for the P queries over the existing cache, and the batched LM head), but:
 *
 *   1. it computes the LM head for EVERY one of the P positions and writes the
 *      full [n_tokens, n_vocab] row-major logits into out_logits (caller-owned),
 *      so the caller can verify each drafted token against its own prediction;
 *   2. it does NOT advance state->pos. The K/V for the P positions ARE written
 *      into the cache at slots pos..pos+P-1 (position-indexed), but because pos
 *      is left untouched those slots are uncommitted: rejected drafts are undone
 *      for free by simply not advancing pos, and the stale slots are overwritten
 *      by the next real write and never read (seq_len derives from pos).
 *
 * Row i's logits depend only on tokens 0..i (causal), so the prediction used to
 * accept/reject token i is exactly the logits the model produces for the prefix
 * ending at candidate i, independent of the candidates after it. This is what
 * makes greedy speculative decode lossless: the accepted stream is always the
 * plain-greedy stream, whatever the draft quality.
 *
 * Returns 0 on success, or -1 (without mutating state) if the model/quant is
 * unsupported (caller falls back to plain decode). */
int pn_llama_verify(pn_model *model, const uint32_t *tokens,
                    const float *const *embd_rows, int n_tokens,
                    pn_state *state, float *out_logits);

/* ============================================================
 * Weight loading
 * ============================================================ */

/* Load quantized weights from GGUF (zero-copy mmap).
 * Only dequants tok_embd and norms to F32. Returns NULL if unsupported quant type. */
pn_llama_qweights* pn_llama_load_gguf_quantized(pn_gguf_file *gguf, pn_config *config);

/* Load F32 weights from GGUF (dequants everything). Legacy Phase 1 loader. */
pn_llama_weights* pn_llama_load_gguf(pn_gguf_file *gguf, pn_config *config);

/* Load from HuggingFace safetensors format (BF16 or ternarized).
 * Detects format automatically. Loads tokenizer from vocab.json+merges.txt
 * in the same directory as config_path. */
pn_llama_qweights* pn_llama_load_safetensors(const char *model_path,
                                              const char *config_path,
                                              pn_config *config);

/* Load BPE tokenizer from HuggingFace directory (vocab.json + merges.txt).
 * dir_path: directory containing vocab.json and merges.txt.
 * config: parsed config (used for bos/eos token IDs). */
ft_tokenizer* pn_llama_load_hf_tokenizer(const char *dir_path, const pn_config *config);

/* Free */
void pn_llama_weights_free(pn_llama_weights *w);
void pn_llama_qweights_free(pn_llama_qweights *qw);

/* KV cache */
pn_llama_kv_cache* pn_llama_kv_cache_new(int n_layers, int n_kv_head, int max_seq, int head_dim);
void pn_llama_kv_cache_free(pn_llama_kv_cache *cache);

/* ============================================================
 * Session snapshot — llama-family KV cache at a token position
 * (llama_state_snapshot.c). Pure host memory: this family's whole per-session
 * state IS the KV cache, and on every backend Pantheon has today — CPU, CUDA
 * and Tenstorrent — those rows live in host RAM, so there is no device readback
 * to arrange. Reached through core/session_snapshot.h, which picks this family
 * or the qwen35moe dual-state one from the model's arch; call that, not these.
 * ============================================================ */

/**
 * Deep-copy rows [0, pos) of every layer out of a live KV cache.
 *
 * @param cache_v  pn_llama_kv_cache * (pn_state::kv_cache on this family)
 * @param pos      committed length; must be > 0 and <= the cache's max_seq,
 *                 because the live cache is a RING and a wrapped one no longer
 *                 holds rows [0, pos)
 * @return         Opaque snapshot (free with pn_llama_kv_snapshot_free), or NULL
 *
 * Thread-safe: no (caller serializes against the generation that owns `cache`)
 * Reentrant: yes
 * Signal-safe: no
 */
void *pn_llama_kv_snapshot(const void *cache_v, int pos);

/**
 * Copy a snapshot back into a live KV cache. Validates n_layers / n_kv_head /
 * head_dim exactly and bounds pos against THIS cache's ring; max_seq itself
 * need not match, since the rows are repacked on the way in. The caller sets
 * pn_state::pos to pn_llama_kv_snapshot_pos() afterwards.
 *
 * @return 0 on success, -1 if the snapshot does not fit this cache
 *
 * Thread-safe: no (same discipline as the capture)
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_llama_kv_restore(void *cache_v, const void *snap);

/** The position a snapshot was taken at, or -1. */
int pn_llama_kv_snapshot_pos(const void *snap);

/** Free a snapshot. NULL-safe. */
void pn_llama_kv_snapshot_free(void *snap);

/**
 * Serialize a snapshot into ONE contiguous little-endian blob (magic "PNLK").
 *
 * @param snap             Snapshot from pn_llama_kv_snapshot()
 * @param weights_version  Opaque model-identity tag, echoed in the header
 * @param bytes_out        Receives a malloc'd buffer (caller frees)
 * @param len_out          Receives its length
 * @return                 0 on success, -1 on failure
 *
 * Thread-safe: yes (reads the snapshot only)
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_llama_kv_snap_serialize(const void *snap, uint64_t weights_version,
                               unsigned char **bytes_out, size_t *len_out);

/**
 * Parse a blob back into an opaque snapshot (free with
 * pn_llama_kv_snapshot_free). Validates magic, version, framing, every derived
 * size, and that the blob is consumed EXACTLY — trailing bytes are a refusal.
 *
 * @return Opaque snapshot, or NULL on any validation failure.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: no
 */
void *pn_llama_kv_snap_deserialize(const unsigned char *bytes, size_t len,
                                   uint64_t *weights_version_out);

#endif /* PN_LLAMA_H */
