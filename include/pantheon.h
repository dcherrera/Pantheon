/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PANTHEON_H
#define PANTHEON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pantheon — Pure C Model Hosting Engine
 *
 * Public API. All types are opaque — implementation hidden.
 * Supports GGUF and safetensors model formats.
 * Supports Llama, Qwen, Phi, Mistral, RWKV-X architectures.
 *
 * Text generation: pn_generate(), pn_stream(), pn_stream_tokens()
 * Multi-modal:    pn_infer(), pn_transcribe(), pn_synthesize(),
 *                 pn_describe(), pn_detect(), pn_embed()
 */

/* Opaque types */
typedef struct pn_model pn_model;
typedef struct pn_state pn_state;

/* Output types for multi-modal inference */
typedef enum {
    PN_OUT_LOGITS,      /* Text token logits: float[n_vocab] */
    PN_OUT_EMBEDDING,   /* Embedding vector: float[n_embd] */
    PN_OUT_AUDIO,       /* Audio waveform: float[n_samples] */
    PN_OUT_BBOXES,      /* Bounding boxes: float[n_boxes * 6] (x1,y1,x2,y2,conf,class) */
    PN_OUT_MASK,        /* Segmentation mask: uint8[height * width] */
    PN_OUT_DEPTHMAP,    /* Depth map: float[height * width] */
} pn_output_type;

/* Tagged output from pn_infer() */
typedef struct {
    pn_output_type type;
    float *data;        /* Output buffer (caller frees via pn_output_free) */
    int    n;           /* Number of primary elements (vocab size, n_samples, n_boxes, etc.) */
    int    width;       /* Image width (vision outputs only, 0 otherwise) */
    int    height;      /* Image height (vision outputs only, 0 otherwise) */
} pn_output;

/*
 * Grammar-constrained decoding hook (n2 tool-calling Phase 5) — OPT-IN.
 *
 * When a sampling call carries a non-NULL `tool_grammar` (below), the sampler
 * masks (logit = -INFINITY) every candidate token whose TEXT would take the
 * grammar state off a valid tool-call prefix, then advances the grammar by the
 * chosen token's text.  This forces the model's trained XML tool-call format
 * even under a heavy quant.  It is entirely device-agnostic (a CPU-side logit
 * mutation, like repeat_penalty) and depends only on this callback + an opaque
 * grammar state — the engine does not link the grammar module or the tokenizer
 * into the sampler.
 *
 * The hook is fully self-contained via callbacks, so the engine's sampler does
 * NOT link the grammar module or the tokenizer:
 *   - `st`         : opaque grammar state (a tg_state*, see tool_grammar.h).
 *   - `token_text` : id → UTF-8 text piece for a token (borrowed; NULL/empty →
 *                    the token is treated as always-allowed, a no-op).
 *   - `allowed`    : would appending this text keep a valid grammar prefix?
 *   - `advance`    : walk the grammar state forward by the chosen token's text.
 *   - `user`       : passed to token_text (e.g. the tokenizer/model handle).
 *   - `active`     : gates the mask.  While the model is still in the free
 *                    reasoning/prose region (no tool-call opener yet) the caller
 *                    leaves this 0 so NOTHING is masked (a legit prose answer is
 *                    never forced into a call); it flips to 1 once a tool-call
 *                    opener has appeared and the grammar has engaged.
 *   - `n_vocab`    : vocab bound for the scan.
 * When the whole hook pointer is NULL the sampler is byte-identical to a build
 * without this feature.
 */
typedef struct {
    void       *st;
    const char *(*token_text)(void *user, uint32_t id);
    int         (*allowed)(void *st, const char *text);   /* 1 = allowed */
    void        (*advance)(void *st, const char *text);
    void       *user;
    int         active;
    int         n_vocab;
    /* Optional self-refresh (may be NULL).  When set, the sampler calls it at
     * entry to recompute `active` from the current grammar state, so the hook
     * gates itself WITHOUT the caller refreshing `active` before every step.
     * This is what lets the BATCHED driver (which samples per-slot in its own
     * thread from the slot's copied sampling params) run the grammar correctly:
     * the driver never touches `active`, the hook refreshes it in-thread.  The
     * single-user path can also rely on it (it still refreshes externally too —
     * harmless).  Returns the new active value (1 = mask this step). */
    int         (*refresh)(void *st);
    /* Optional (may be NULL): has the grammar reached a COMPLETE tool call?
     * The BATCHED driver queries this AFTER advancing so it can finish the slot
     * the instant "</function>"/"</tool_call>" closes — the completion-stop for
     * the batched path (mirrors the single-user loop's break-on-complete), so a
     * forced call terminates right after the call instead of running to the
     * max_tokens budget.  Unused (NULL) in the single-user path (its serve loop
     * checks completion itself). */
    int         (*complete)(void *st);
    /* Optional (may be NULL): is a tool-call skeleton currently being emitted
     * under the grammar?  The BATCHED driver's AUTO reasoning-budget force-close
     * queries this so it never injects "</think>" mid-skeleton (which would
     * desync the grammar walk) — the batched twin of the single-user
     * sg_tg_in_call guard.  NULL → treated as "not in a call" (the single-user
     * path checks in_call itself and leaves this NULL). */
    int         (*in_call)(void *st);
    /* Optional (may be NULL): tell the grammar its reasoning close was INJECTED
     * by the serve's reasoning-budget force-close, not chosen by the model.  The
     * grammar cannot tell the two apart on its own — the injected "</think>" is
     * fed through `advance` like any other text — so it would compel the tool
     * opener from the very next char, while the model is still mid-sentence, and
     * the rest of its thought (often a complete call) lands inside a parameter
     * VALUE.  Called by the BATCHED driver once the injected close has drained;
     * the single-user loop calls the grammar directly.  NULL → the pre-fix
     * behavior (compel immediately). */
    void        (*reasoning_end_injected)(void *st);
    /* Fail-CLOSED policy (forced-mode hardening).  On a 0-survivor step (the mask
     * would leave NO legal token) the sampler must NOT fall open to the model's
     * own (garbage) top token — that is how empty/garbage calls escaped.  When
     * `fail_closed` is nonzero the sampler instead forces `forced_token` (an EOS /
     * end id supplied by the caller), terminating the generation cleanly rather
     * than emitting an out-of-grammar token.  EOS is always a safe closed exit
     * (the grammar guarantees a completed-or-abortable call, never a deadlock).
     * When `fail_closed` is 0 the sampler keeps the legacy fail-open behavior
     * (AUTO mode, or callers that did not set this) — so nothing regresses. */
    int         fail_closed;    /* 1 = force forced_token on 0-survivor */
    uint32_t    forced_token;   /* token forced on a 0-survivor step (EOS id) */
    /* Optional (may be NULL): may a token with NO text (an end token: EOS,
     * <|im_end|>) be drawn while the mask is active?  NULL = always, the
     * historical rule.  A "required" walk answers no before its call opens (the
     * turn must make a call) and yes inside it only when the call can still be
     * closed — the serve then closes it (tgj_closing_text). */
    int         (*end_allowed)(void *st);
    /* Optional (may be NULL): a private copy of the walk `st`, which the caller may
     * advance (advance) and query (refresh) without touching the live walk, and its
     * release. The batched MTP driver looks ahead over drafts with it, so speculation
     * stops before any row the mask would touch (batch_sched_spec.h). */
    void       *(*clone)(void *st);
    void        (*clone_free)(void *st);
    /* Optional (may be NULL = no steer, the sampler unchanged): on a masked
     * step the sampler hands this the text of the model's own top token (the
     * unmasked argmax) when `allowed` refuses it.  The grammar may then move
     * `st` so the mask that follows leaves only a forced continuation
     * (field_patterns: a refused close on a patterned value becomes the
     * pattern's next literal bytes, tg_steer).  Returns 1 when it moved `st`. */
    int         (*steer)(void *st, const char *text);
    /* Optional (may be NULL = never): has forced-mode reasoning ended on the walk
     * (the serve's sg_tg_forced_ended)?  The BATCHED driver asks after each token
     * it deposits and records the first ring position where it holds, so the
     * owner flips its emitter out of <think> at that token, not when the
     * driver's walk has already run ahead over a backlog (bs_slot_reasoning_ended). */
    int         (*reasoning_ended)(void *st);
} pn_tool_grammar_hook;

/* One entry of an OpenAI `logit_bias` map: a token id and the bias (nominally
 * -100..100) ADDED to that token's raw logit before any other sampling step. */
typedef struct {
    uint32_t token_id;
    float    bias;
} pn_logit_bias_entry;

/*
 * Per-token logprob sink (OpenAI `logprobs` / `top_logprobs`).
 *
 * OFF by default (NULL in pn_sampling_params).  When set, the sampler computes a
 * log-softmax over the RAW logits — before logit_bias, penalties, temperature,
 * top-k/top-p and the grammar mask — so the reported numbers are the model's own
 * distribution, which is what the OpenAI field means (they are NOT the
 * post-filter probabilities the draw was actually taken from).
 *
 *   - `n_top` : how many ALTERNATIVES to report per step (0..PN_LOGPROBS_MAX_TOP).
 *   - `emit`  : called once per sampled token with the chosen token's logprob and
 *               the top-`n_top` alternatives (descending logprob).  The arrays
 *               are borrowed and valid only for the duration of the call.
 *
 * INJECTED-TOKEN CONTRACT.  A caller that force-decodes a token it did not
 * sample (the reasoning-budget "</think>" close) must call `emit` for it with
 * `chosen_logprob = NAN` and `n_top = 0`.  The sink's record sequence is how a
 * consumer pairs logprobs with the emitted token stream, so a token that
 * silently produced no record would shift every later token's logprobs onto the
 * wrong text — the placeholder keeps the two aligned.
 */
#define PN_LOGPROBS_MAX_TOP 20

typedef struct {
    int   n_top;
    void (*emit)(void *user, uint32_t chosen, float chosen_logprob,
                 const uint32_t *top_ids, const float *top_logprobs, int n_top);
    void *user;
} pn_logprob_sink;

/* Sampling parameters */
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float repeat_penalty;
    int   repeat_last_n;
    /* OFF by default (NULL).  Non-NULL → grammar-constrained tool decoding
     * (Phase 5).  With it NULL, sample_token is byte-identical to before. */
    const pn_tool_grammar_hook *tool_grammar;

    /* ── OpenAI sampling fields (all default to "off" = 0, so a caller that
     * builds params with pn_sampling_defaults() is byte-identical to before) ── */

    /* OpenAI `presence_penalty` / `frequency_penalty`, applied over the same
     * recent-token window `repeat_last_n` feeds the repetition penalty:
     *   logit[t] -= frequency_penalty * count(t in window)
     *   logit[t] -= presence_penalty  * (count(t in window) > 0)
     * Unlike `repeat_penalty` (a SERVER default of 1.1) these are only ever
     * non-zero because the CLIENT asked for them, so — also unlike repeat_penalty
     * — they are applied on the greedy (temperature<=0) path too.  See the
     * ordering note in sample_token_core. */
    float presence_penalty;
    float frequency_penalty;

    /* OpenAI `logit_bias`: borrowed array of id→bias entries (NULL / 0 = off).
     * Added to the RAW logits first, before everything else. */
    const pn_logit_bias_entry *logit_bias;
    int                        n_logit_bias;

    /* OpenAI `seed`: when `rng_seeded` is non-zero the random draw comes from
     * `rng_state` (a per-request splitmix64/xorshift stream advanced in place)
     * instead of the process-global rand(), so the same request + seed replays
     * the same tokens.  Zero → the global RNG, byte-identical to before. */
    int      rng_seeded;
    uint64_t rng_state;

    /* OpenAI `logprobs` / `top_logprobs` (NULL = off). */
    const pn_logprob_sink *logprobs;

    /* A backend that hands the sampler a TOP-N row (-INFINITY outside the N it named)
     * gives a refill here: when the row cannot decide this draw exactly (the request's
     * top-k could reach past the N after penalties), the sampler calls it for the dense
     * row and samples that instead. NULL = rows are taken as given. Returns 0 on success. */
    int  (*dense_row)(void *ctx, float *logits, int n_vocab);
    void  *dense_ctx;
    /* Optional, with dense_row: fetch only PART of the dense row — every 16-entry
     * chunk i with need[i] != 0 (ids 16*i .. 16*i+15) is written into `logits` at
     * its ids, the rest is left as it was.  The grammar mask uses it to read just
     * the chunks holding allowed tokens instead of the whole row.  NULL = whole row
     * only.  Same ctx as dense_row.  Returns 0 on success. */
    int  (*dense_chunks)(void *ctx, const unsigned char *need, float *logits, int n_vocab);
    /* A uniform already drawn for the NEXT draw (0 = none; the default). Speculative
     * sampling sets it when a verify row past 0 needed the dense row after its draw: the
     * row defers WITH the draw and the next draw at that position consumes it, so the RNG
     * stream and the draw's distribution are plain decode's. */
    int   carried_uniform_set;
    float carried_uniform;
    /* Optional, with dense_row: count the RAW logit row `row` (0 = this step's, k = verify
     * row k) into bins on the device — counts[j] = logits whose bf16 order key lies in
     * [keys[j], keys[j-1]), counts[bins] = below keys[bins-1] (keys strictly descending).
     * Lets the sampler bound the mass outside a top-N row without the dense row
     * (sampler_topn_tail.h). Same ctx as dense_row. Returns 0 on success. */
    int  (*tail_counts)(void *ctx, int row, const uint32_t *keys, int bins, uint32_t *counts);
} pn_sampling_params;

/* Generation callback — called for each new token */
typedef void (*pn_token_callback)(const char *token, void *user_data);

/* Stop predicate — polled during generation; return non-zero to abort early. */
typedef int (*pn_stop_callback)(void *user_data);

/* Return default sampling parameters */
pn_sampling_params pn_sampling_defaults(void);

/* Load a model from GGUF or safetensors file */
pn_model* pn_load(const char *path);

/* ============================================================
 * Pre-flight admission control (MemoryAware layer, §6)
 * ============================================================ */

/* Verdict of pn_admit: whether an n_tokens prefill can run as-is, must be
 * capped to a smaller window, or must be refused to protect the host. */
typedef enum {
    PN_ADMIT_FIT    = 0,   /* run it as requested                             */
    PN_ADMIT_CAPPED = 1,   /* run only the first *fit_tokens_out tokens        */
    PN_ADMIT_DENIED = 2,   /* refuse — host/die would be exhausted            */
} pn_admit_status;

/*
 * Decide whether to admit an n_tokens prefill for `model`, BEFORE any
 * memory-growing work starts. Engine-level so both the server (serve_generation)
 * and direct pn_model callers (evals / eval_ppl — the path that crash-looped)
 * are covered. Computes the projected host + per-die cost, compares against the
 * live memory state minus reserved headroom, and returns the verdict. On
 * PN_ADMIT_CAPPED, *fit_tokens_out (if non-NULL) receives the largest prefill
 * length that fits. Never allocates, never blocks. If the memory-watch layer is
 * not running it fails open on memory, but a prompt past the model's configured
 * window (pn_session_ctx_cap) is still PN_ADMIT_CAPPED to that window.
 *
 * Thread-safe: yes
 * Reentrant: yes
 */
pn_admit_status pn_admit(pn_model *model, int n_tokens, int *fit_tokens_out);

/*
 * The CONFIGURED per-session hard context cap for `model`, in tokens — the KV
 * ring / max_seq ceiling clamped to the model's trained context, i.e. what the
 * operator asked for (`--ctx-cap`). Returns 0 if unknown. Pure — never
 * allocates.
 *
 * This is the OPERATOR's number. Never hand it to a client: under memory
 * pressure admission enforces a smaller one — see pn_effective_ctx_window.
 *
 * Thread-safe: yes
 * Reentrant: yes
 */
int pn_session_ctx_cap(const pn_model *model);

/*
 * The EFFECTIVE per-session context window for `model`, in tokens — the
 * configured cap above AFTER the reactive memory ladder's runtime clamp.
 *
 * The two differ exactly when the ladder has shrunk the admitted prefill window
 * under memory pressure, and that difference is the whole reason this call
 * exists: advertising the configured cap while admission enforces a smaller one
 * gives a client a window the serve will then reject (and produced the
 * self-contradictory "prompt 70000 > session cap 131072" message Phase 10
 * fixed). Everything a CLIENT is told — `context_window` on GET /v1/models and
 * GET /admin/models, `kv_cap` in a completion's usage object — comes from here.
 *
 * Returns 0 if unknown. Reads the live memwatch snapshot; never allocates.
 *
 * Thread-safe: yes
 * Reentrant: yes
 */
int pn_effective_ctx_window(const pn_model *model);

/* Free a loaded model */
void pn_model_free(pn_model *model);

/* Keep a loaded model's accelerator resident while it is idle (the serve's idle tick, PM 5).
 * Cheap and safe to call any time no generation is running on the model; a no-op for backends
 * with nothing to keep alive. Returns >= 0, or -1 when a keepalive failed (the next use then
 * re-arms as it always has). */
int pn_model_keepalive(pn_model *model);
/* Warm an accelerator-resident model once after load (re-arms idle families before the first
 * request). Returns 0, or <0 on failure (the model still serves). No-op elsewhere. */
int pn_model_warm(pn_model *model);

/* Create inference state for a model */
pn_state* pn_state_new(pn_model *model);

/* Free inference state */
void pn_state_free(pn_state *state);

/* Reset inference state (clear KV cache, reset position) */
void pn_state_reset(pn_state *state);

/*
 * Generate text from a prompt.
 *
 * Returns a newly allocated string (caller must free with free()).
 * Returns NULL on error.
 */
char* pn_generate(pn_model *model, pn_state *state,
                  const char *prompt, int max_tokens,
                  pn_sampling_params *params);

/*
 * Streaming generation — calls callback for each token.
 *
 * Returns number of tokens generated, or -1 on error.
 */
int pn_stream(pn_model *model, pn_state *state,
              const char *prompt, int max_tokens,
              pn_sampling_params *params,
              pn_token_callback callback, void *user_data);

/*
 * Streaming generation from pre-tokenized input.
 * Prefills from token array, then generates until EOS or max_tokens.
 * Does NOT add BOS — caller controls tokenization entirely.
 * Returns number of tokens generated, or -1 on error.
 */
int pn_stream_tokens(pn_model *model, pn_state *state,
                     const uint32_t *tokens, int n_tokens,
                     int max_tokens,
                     pn_sampling_params *params,
                     pn_token_callback callback, void *user_data);

/* --- Multi-modal inference API --- */

/*
 * Low-level forward pass returning typed output.
 * For text models, returns PN_OUT_LOGITS with float[n_vocab].
 * Caller must free result with pn_output_free().
 * Returns NULL on error.
 */
pn_output* pn_infer(pn_model *model, pn_state *state, uint32_t token);

/*
 * Batched prefill of `n_tokens` prompt tokens (an accelerator's batched path, e.g. the TT
 * card's), returning the LAST position's logits like pn_infer(). Returns NULL when no
 * batched path took the prompt (the state is then untouched and the caller runs pn_infer()
 * per token). Caller frees with pn_output_free().
 */
pn_output* pn_infer_prefill(pn_model *model, pn_state *state, const uint32_t *tokens,
                            int n_tokens);

/* Free a pn_output returned by pn_infer() */
void pn_output_free(pn_output *out);

/*
 * Speech-to-text: transcribe audio to text.
 * Returns newly allocated string (caller frees with free()).
 * Returns NULL if model does not support STT.
 */
char* pn_transcribe(pn_model *model, pn_state *state,
                    const float *audio_samples, int n_samples,
                    int sample_rate);

/*
 * Text-to-speech: synthesize audio from text.
 * Writes audio samples to *out_samples (caller frees with free()).
 * Returns number of samples written, or -1 on error.
 * Returns -1 if model does not support TTS.
 */
int pn_synthesize(pn_model *model, pn_state *state,
                  const char *text, float **out_samples,
                  int *out_sample_rate);

/*
 * Attach a vision encoder (an llama.cpp-style "clip" mmproj GGUF) to a loaded
 * text model so pn_describe() can run. Returns 0 on success, negative on error.
 */
int pn_attach_vision(pn_model *model, const char *mmproj_path);

/*
 * Vision-language: describe an image given a text prompt.
 * Returns newly allocated string (caller frees with free()).
 * Returns NULL if model does not support VLM.
 */
char* pn_describe(pn_model *model, pn_state *state,
                  const uint8_t *image_rgb, int width, int height,
                  const char *prompt, int max_tokens,
                  pn_sampling_params *params);

/*
 * Streaming variant of pn_describe: same result, but as generation proceeds it
 * calls callback(full_text_so_far, user_data) after each token with the full
 * decoded text up to that point (not just the new token). Pass NULL to behave
 * exactly like pn_describe. If should_stop is non-NULL it is polled after each
 * generated token; returning non-zero ends generation early and returns the text
 * produced so far. Returns the final string (caller frees with free()).
 */
char* pn_describe_stream(pn_model *model, pn_state *state,
                         const uint8_t *image_rgb, int width, int height,
                         const char *prompt, int max_tokens,
                         pn_sampling_params *params,
                         pn_token_callback callback, void *user_data,
                         pn_stop_callback should_stop, void *stop_data);

/*
 * Vision-language decode from PRE-SUPPLIED image embeddings, skipping the
 * internal vision encode. Does exactly what pn_describe_stream does after the
 * encode: builds the <|vision_start|> + <|image_pad|> * n_img + <|vision_end|> +
 * prompt sequence, splices the supplied [n_img x embd_dim] decoder-space embeds
 * at the image-pad positions, batched-prefill, and generates. embd_dim must
 * equal the model's decoder embedding dimension. img_embds is NOT freed here
 * (the caller owns it), so an external (e.g. Core ML GPU) vision encoder can
 * supply them. callback / should_stop behave as in pn_describe_stream. Returns
 * newly allocated text (caller frees with free()); NULL on error.
 */
char* pn_describe_embeds(pn_model *model, pn_state *state,
                         const float *img_embds, int n_img, int embd_dim,
                         const char *prompt, int max_tokens,
                         pn_sampling_params *params,
                         pn_token_callback callback, void *user_data,
                         pn_stop_callback should_stop, void *stop_data);

/*
 * Run just the vision encoder for one RGB image, returning decoder-space
 * embeddings [n_img x n_embd] (caller frees *embeds_out). The vision half of
 * pn_describe_stream, split out so a bulk pipeline can stage embeds ahead of the
 * batched decode. Returns 0 on success, -1 on error.
 */
int pn_vision_encode_rgb(pn_model *model, const uint8_t *image_rgb, int width, int height,
                         float **embeds_out, int *n_img_out, int *embd_dim_out);

/* --- Continuous-batch VLM decode (bulk OCR throughput) --- */

/*
 * Prefill a state's KV cache from pre-supplied image embeddings (the ChatML
 * vision turn) and sample the first generated token, WITHOUT running the decode
 * loop. Leaves the state ready to be driven by the batch decoder below (or
 * pn_llama_forward_batch). Writes *first_tok_out and returns 0 on success, -1 on
 * error. img_embds is NOT freed (caller owns it).
 */
int pn_vlm_prefill_embeds(pn_model *model, pn_state *state,
                          const float *img_embds, int n_img, int embd_dim,
                          const char *prompt, pn_sampling_params *params,
                          uint32_t *first_tok_out);

/* Opaque continuous-batch decoder. */
typedef struct pn_batch_decoder pn_batch_decoder;

/* A finished page returned by the decoder: its id, its now-recyclable state, and
 * the decoded UTF-8 transcription (caller frees text with free()). */
typedef struct {
    int       page_id;
    pn_state *state;
    char     *text;
} pn_page_out;

/*
 * Create a decoder that batches up to max_batch prefilled sequences through
 * pn_llama_forward_batch, handling per-sequence sampling, EOS, and
 * detokenization. max_tokens caps each page's output. Returns NULL on error.
 */
pn_batch_decoder* pn_batch_decoder_new(pn_model *model, int max_batch,
                                       int max_tokens, const pn_sampling_params *params);
void pn_batch_decoder_free(pn_batch_decoder *d);

int  pn_batch_decoder_capacity(const pn_batch_decoder *d);   /* max_batch */
int  pn_batch_decoder_active(const pn_batch_decoder *d);      /* slots in use */

/*
 * Admit a prefilled sequence into a free slot. state must already be prefilled
 * (pn_vlm_prefill_embeds) and first_tok is that call's first_tok_out. Returns 0
 * on success, -1 if the decoder is full. The decoder borrows state until it is
 * handed back via a pn_page_out.
 */
int  pn_batch_decoder_admit(pn_batch_decoder *d, pn_state *state, int page_id,
                            uint32_t first_tok);

/*
 * Run ONE batched decode step over all active slots. Pages that finish (EOS or
 * max_tokens) this step are written to out[] (up to max_out) with their text and
 * their recyclable state; returns the number finished (>=0), or -1 if the model
 * does not support batched decode (caller should fall back). Call repeatedly
 * while pn_batch_decoder_active() > 0, admitting fresh pages into freed slots.
 */
int  pn_batch_decoder_step(pn_batch_decoder *d, pn_page_out *out, int max_out);

/*
 * Finalize ALL active slots with whatever they have decoded so far (early stop /
 * drain), returning the number written to out[] (up to max_out).
 */
int  pn_batch_decoder_flush(pn_batch_decoder *d, pn_page_out *out, int max_out);

/* ============================================================
 * Batched lossless speculative decode
 * ============================================================ */

/* Draft provider: given the accepted prefix generated so far for one sequence
 * (prefix[0..n_prefix-1], the model's real tokens), fill draft_out with up to
 * max_k proposed next tokens and return how many were produced (0..max_k).
 * Returning 0 means "no proposal" (that slot takes a single plain greedy step
 * this cycle). This is the seam where an external draft source plugs in. */
typedef int (*pn_draft_fn)(const uint32_t *prefix, int n_prefix,
                           uint32_t *draft_out, int max_k, void *user_data);

/*
 * Configure the decoder for batched speculative decode. max_k is the per-slot
 * draft width, min_accept_frac / probe_cycles drive the per-page fallback (after
 * probe_cycles speculative cycles a slot whose running acceptance fraction is
 * below min_accept_frac stops drafting and rides as a plain single-token slot).
 * REQUIRES the decoder's sampling temperature <= 0 (greedy): speculation is only
 * lossless for greedy decode, so this returns -1 for a sampling temperature.
 * Allocates the ragged verify scratch. Returns 0 on success, -1 on error.
 */
int  pn_batch_decoder_spec_config(pn_batch_decoder *d, int max_k,
                                  double min_accept_frac, int probe_cycles);

/*
 * Admit a prefilled sequence into a free slot with its own draft provider (as
 * pn_batch_decoder_admit, plus draft_fn/draft_ud). Pass draft_fn = NULL to admit
 * a plain (non-speculative) slot that still steps correctly under step_spec (it
 * rides as a K=1 block). Requires pn_batch_decoder_spec_config first. Returns 0
 * on success, -1 if full or not configured.
 */
int  pn_batch_decoder_admit_spec(pn_batch_decoder *d, pn_state *state, int page_id,
                                 uint32_t first_tok, pn_draft_fn draft_fn, void *draft_ud);

/*
 * Run ONE batched speculative decode step. Each active slot proposes up to max_k
 * drafts (via its draft_fn), all slots' [pending, drafts...] blocks are verified
 * in one pn_llama_verify_batch pass, and each slot greedily accepts the drafts
 * that already equal the model's own argmax, emitting the exact greedy
 * continuation. The emitted token stream is therefore BIT-IDENTICAL to plain
 * batched greedy decode (pn_batch_decoder_step) for ANY draft; drafts only change
 * how many tokens each verify pass yields. Finished pages (EOS / max_tokens) are
 * written to out[] (up to max_out); returns the number finished (>=0), or -1 if
 * the model does not support batched verify (caller falls back to plain step).
 */
int  pn_batch_decoder_step_spec(pn_batch_decoder *d, pn_page_out *out, int max_out);

/* Aggregate speculative-decode accounting since pn_batch_decoder_spec_config:
 * verify passes run, draft tokens proposed, and draft tokens accepted (any of
 * the out pointers may be NULL). */
void pn_batch_decoder_spec_stats(const pn_batch_decoder *d, long *passes,
                                 long *drafted, long *accepted);

/*
 * Object detection: find objects in an image.
 * Returns bounding boxes as float[n_boxes * 6]: x1, y1, x2, y2, confidence, class_id.
 * Caller frees returned array with free().
 * Returns NULL if model does not support detection.
 */
float* pn_detect(pn_model *model, pn_state *state,
                 const uint8_t *image_rgb, int width, int height,
                 float confidence_threshold, int *n_boxes_out);

/*
 * Embedding extraction: compute a vector representation.
 * Returns float[n_embd] (caller frees with free()).
 * Returns NULL if model does not support embeddings.
 */
float* pn_embed(pn_model *model, pn_state *state,
                const char *text, int *n_dims_out);

#ifdef __cplusplus
}
#endif

#endif /* PANTHEON_H */
