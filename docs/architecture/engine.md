# Engine Architecture

The engine (`src/core/engine.c`) is Pantheon's central coordinator. It handles model loading, architecture dispatch, tokenization, sampling, and the generation loop.

## Model Loading Pipeline

Two entry points: `pn_load(path)` for GGUF models (legacy), and `pn_load_ex(path, opts)` for all formats including safetensors.

```
pn_load_ex(path, opts)
  ├── pn_is_safetensors(path)?
  │   ├── yes → RWKV-X path:
  │   │   ├── pn_rwkvx_load(path, config, quantize)   load safetensors weights
  │   │   ├── pn_rwkvx_load_tokenizer(weights, vocab)  load trie tokenizer from vocab file
  │   │   ├── forward = pn_rwkvx_forward               set forward function
  │   │   └── tokenizer = NULL                          (RWKV-X uses internal trie tokenizer)
  │   │
  │   └── no → GGUF path:
  │       ├── pn_gguf_open(path)            mmap file, parse header + metadata
  │       ├── detect arch from metadata     "general.architecture" → pn_arch enum
  │       ├── pn_llama_load_qweights(gguf)  load weights (architecture-specific)
  │       ├── extract tokenizer from GGUF   tokens + merges/scores arrays
  │       ├── pn_get_forward_fn(arch)       select forward pass function pointer
  │       └── detect chat template style
  │
  └── create thread pool                    (if USE_THREADS=1)
```

`pn_load(path)` is equivalent to `pn_load_ex(path, NULL)`.

### Load Options

```c
typedef struct {
    const char *vocab_path;   /* RWKV World tokenizer (required for safetensors) */
    bool quantize;            /* INT8 quantize at load time */
} pn_load_opts;
```

### Format Detection

| Function | Detection Method |
|----------|-----------------|
| `pn_is_gguf(path)` | First 4 bytes = `"GGUF"` magic |
| `pn_is_safetensors(path)` | 8-byte LE header length < 10M, followed by `{` JSON |

### Architecture Detection

GGUF models: the `general.architecture` metadata key maps to:

| Value | Enum | Forward Pass |
|-------|------|-------------|
| `"llama"` | `PN_ARCH_LLAMA` | `pn_llama_forward` |
| `"qwen2"` | `PN_ARCH_QWEN2` | `pn_llama_forward` (shared) |
| `"phi"`, `"phi3"` | `PN_ARCH_PHI` | `pn_phi3_forward` (separate) |
| `"mistral"` | `PN_ARCH_MISTRAL` | `pn_llama_forward` (shared) |
| `"llama"` + `expert_count > 0` | `PN_ARCH_MIXTRAL` | `pn_llama_forward` (shared, MoE FFN) |
| unknown | `PN_ARCH_UNKNOWN` | falls back to Llama |

Safetensors models: architecture is always `PN_ARCH_RWKVX` (currently the only safetensors architecture).

Qwen2 and Mistral share the Llama forward pass — architecture differences (attention biases, sliding window, metadata key prefixes) are handled by config flags.

Phi-3 has its own forward pass (`src/models/phi3.c`) due to fused QKV weights, fused gate+up weights, and NeoX-style RoPE that differ significantly from the Llama code path.

RWKV-X has its own complete engine (`src/models/rwkvx.c`) — a hybrid recurrent + sparse attention architecture that is fundamentally different from transformer models. See [RWKV-X Architecture](rwkvx.md) for details.

### Config Parsing

Model dimensions are read from GGUF metadata using architecture-specific key prefixes:

```
llama.*   → Llama, Mistral (Mistral GGUFs use llama.* keys)
qwen2.*   → Qwen2/Qwen2.5
phi3.*    → Phi-3 (read by phi3.c's own config reader)
```

Key parameters:

| Config Field | GGUF Key | Default |
|-------------|----------|---------|
| `n_vocab` | `{arch}.embedding_length` | - |
| `n_embd` | `{arch}.embedding_length` | - |
| `n_head` | `{arch}.attention.head_count` | - |
| `n_kv_head` | `{arch}.attention.head_count_kv` | `n_head` |
| `n_layer` | `{arch}.block_count` | - |
| `n_ff` | `{arch}.feed_forward_length` | - |
| `n_ctx` | `{arch}.context_length` | 2048 |
| `rope_theta` | `{arch}.rope.freq_base` | 10000.0 |
| `rms_norm_eps` | `{arch}.attention.layer_norm_rms_epsilon` | 1e-5 |
| `sliding_window` | `{arch}.attention.sliding_window` | 0 (disabled) |

## Tokenizers

Pantheon supports three tokenizer types, routing automatically based on model architecture.

### GGUF Tokenizer Extraction

For GGUF models, tokenizer data is extracted from metadata arrays:

- `tokenizer.ggml.tokens` — vocabulary (string array)
- `tokenizer.ggml.merges` — BPE merge rules (string array, BPE only)
- `tokenizer.ggml.scores` — unigram scores (float array, SentencePiece only)
- `tokenizer.ggml.bos_token_id` — beginning-of-sequence token
- `tokenizer.ggml.eos_token_id` — end-of-sequence token
- `tokenizer.ggml.model` — tokenizer type (`"gpt2"` = BPE, `"llama"` = SentencePiece)

### Tokenizer Types

| Type | Models | Encoding | BOS | Vocab |
|------|--------|----------|-----|-------|
| BPE | SmolLM2, Qwen2.5, Mistral v0.3 | GPT-2 byte encoding + BPE merges | No | Varies |
| SentencePiece | Mistral v0.1, Llama 1/2 | Leading space prepend + Viterbi unigram | Yes (auto) | Varies |
| RWKV World | RWKV-X | Trie-based greedy longest-match on UTF-8 bytes | No | 65,536 |

BPE and SentencePiece tokenizers use Foundry's `ft_tokenizer` and are stored in `pn_model->tokenizer`.

The RWKV World tokenizer is fundamentally different — it's a trie-based tokenizer that operates on raw UTF-8 bytes with greedy longest-match. It's stored inside the RWKV-X internal model struct (accessed via `pn_rwkvx_encode`/`pn_rwkvx_decode`), and `pn_model->tokenizer` is NULL for RWKV-X models. The engine's `pn_generate()`/`pn_stream()` functions detect `PN_ARCH_RWKVX` and route to the internal tokenizer automatically.

## Generation Loop

```
pn_generate(model, state, prompt, max_tokens, params)
  ├── encode prompt → token IDs
  │   ├── BPE: gpt2_bytes_encode(prompt) → ft_encode()
  │   └── SentencePiece: ft_encode_with_special(prompt, add_bos=true)
  │
  ├── prefill: for each prompt token
  │   └── logits = model->forward(model, token, state)
  │
  ├── decode: for i in 0..max_tokens
  │   ├── next_token = sample_token(logits, params)
  │   ├── if next_token == EOS: break
  │   └── logits = model->forward(model, next_token, state)
  │
  └── decode token IDs → output string
```

### Sampling

Applied in order:

1. **Temperature** — `logits[i] /= temperature` (0 = greedy argmax)
2. **Repetition penalty** — penalize tokens in the recent window (`repeat_last_n`)
3. **Top-k** — keep only the k highest-probability tokens
4. **Top-p (nucleus)** — keep tokens until cumulative probability exceeds p
5. **Softmax** — convert to probabilities
6. **Multinomial sampling** — weighted random selection

## Vision-Language and Bulk OCR

The vision-language path (`pn_describe` / `pn_describe_stream`) is split into a vision-encode seam and a decode seam so a bulk pipeline can stage work and run vision and decode on different units.

```
pn_describe_stream(image, prompt)
  ├── pn_vision_encode_rgb(image) → [n_img x n_embd] decoder-space embeds
  └── pn_describe_embeds(embeds, prompt)
        ├── build ChatML vision turn:
        │     <|vision_start|> + <|image_pad|> * n_img + <|vision_end|> + prompt
        ├── splice the supplied embeds at the image-pad positions
        ├── batched prefill (pn_llama_prefill)
        └── decode loop → text
```

- `pn_vision_encode_rgb` runs just the CPU vision encoder for one image and returns the `[n_img x n_embd]` embeddings (caller owns them). Splitting it out lets a bulk caller encode ahead of decode, or substitute an external encoder (e.g. a Core ML GPU or hand-rolled Metal ViT) that produces the same `[n_img x n_embd]` buffer.
- `pn_describe_embeds` is the decode seam: it takes pre-supplied embeds instead of encoding, and is what `pn_describe_stream` now delegates to after its own encode. The portable single-image path is unchanged.

### Continuous-Batch VLM Decoder

For bulk OCR (many pages at once), the pipeline drives many sequences through the batched decode (`pn_llama_forward_batch`) at the higher aggregate throughput described in [Llama Forward Pass](llama-forward.md#batched-multi-sequence-decode).

- `pn_vlm_prefill_embeds` factors the shared prefill out of `pn_describe_embeds`: it prefills a state's KV cache from image embeds (the ChatML vision turn) and samples the first token, but does not run the decode loop. It leaves the state ready to hand to the batch decoder.
- `pn_batch_decoder` admits prefilled pages into up to `max_batch` slots. Each `step()` runs one `pn_llama_forward_batch` over all active slots with per-sequence sampling, EOS detection, and detokenization, and returns each page that finished this step as a `pn_page_out` (`page_id`, the now-recyclable `pn_state`, and the decoded text). The caller slots a fresh page into each freed slot (continuous batching). `flush()` drains all active slots for early stop.

```
for each page:  pn_vision_encode_rgb → pn_vlm_prefill_embeds → pn_batch_decoder_admit
loop while active > 0:
  n = pn_batch_decoder_step(out, max_out)     # one M=B decode over all slots
  for each finished page in out: emit text, recycle state, admit next page
```

The batched decoder is byte-identical to sequential `pn_describe_embeds`: two pages with different embeds decoded together reproduce each page's standalone greedy transcription and finish at their own lengths (verified by `test_batch_decoder`). If the model does not support batched decode (MoE, F32 legacy, ternary), `pn_batch_decoder_step` returns -1 and the caller falls back to the sequential path.

## Memory Model

```
pn_model
  ├── config         model dimensions (stack allocated)
  ├── weights        architecture-specific (heap, points into mmap or owned memory)
  ├── tokenizer      BPE/SentencePiece tokenizer (heap, NULL for RWKV-X)
  ├── forward        forward pass function pointer
  ├── path           model file path (heap)
  ├── chat_style     chat template enum
  └── pool           thread pool (optional, heap)

pn_state
  ├── kv_cache       architecture-specific state:
  │                    GGUF models: pn_llama_kv_cache (KV tensors per layer)
  │                    RWKV-X: rx_state (recurrent state + KV cache + workspace)
  ├── pos            current sequence position
  ├── config         pointer to model's config
  └── arena          bump allocator for per-token scratch (NULL for RWKV-X)
```

For GGUF models, the arena allocator eliminates per-token `malloc`/`free` calls during inference. All intermediate buffers (projections, activations, attention scores) are allocated from the arena, which resets to zero after each token. This removes ~330 allocations per token in the forward pass.

For RWKV-X models, the engine uses its own pre-allocated workspace with ping-pong buffers. The arena is not used.

## Weight Loading Modes

The engine supports two weight storage modes:

### Quantized (Phase 2, default)

Weights stay in their original GGUF quantized format, pointing directly into mmap'd memory. Dequantization happens per-block during matrix-vector multiply (`pn_qmv`). Only embeddings and layer norms are dequantized to F32 on load.

Benefits: ~4x less RAM for Q4 models, zero-copy weight loading.

### F32 Legacy (Phase 1 fallback)

All weights dequantized to F32 `tm_tensor` on load. Used as fallback for unsupported quant types. Higher memory usage but simpler code path.

## Thread Pool

When built with `USE_THREADS=1`, a thread pool is created at model load. The main thread participates in work (no idle main thread). Currently used for:

- Parallel attention head computation — each head runs independently via `fw_for(pool, n_heads, fn, arg)`

Thread count is auto-detected from hardware.
