# RWKV-X Architecture

RWKV-X is a hybrid model architecture that interleaves RWKV-7 recurrent blocks (~75% of layers) with sparse attention blocks (~25%). It is fundamentally different from transformer models and uses safetensors format instead of GGUF.

## Source

The RWKV-X engine in Pantheon (`src/models/rwkvx.c`, ~2,500 LOC) is copied from the battle-tested Foundry implementation (`foundry/rwkvx/rwkvx.c`), with Metal, CUDA, HybridMem, training, batch, instrumentation, and OpenMP support stripped. A thin adapter layer wraps it into Pantheon's `pn_model`/`pn_state`/`pn_forward_fn` interface.

## Model Architecture

For the 0.2B model (12 RWKV + 4 attention layers), block execution order is:

```
R0, R1, R2, A0, R3, R4, R5, A1, R6, R7, R8, A2, R9, R10, R11, A3
```

Where `R` = RWKV-7 recurrent block, `A` = sparse attention block.

### Forward Pass

```
embed(token)
  → for each block in interleaved order:
      if RWKV-7 block:
        LayerNorm → TimeMix (WKV-7 kernel) → residual
        → LayerNorm → ChannelMix FFN → residual
      if attention block:
        LayerNorm → Multi-Head Attention (with KV cache) → residual
        → LayerNorm → ChannelMix FFN → residual
  → LayerNorm → LM head → logits
```

### WKV-7 Kernel (Generalized Delta Rule)

The core recurrent kernel, computed per-head per-timestep:

```
for each timestep t, for each head h:
  state *= diag(w[t])           // decay
  state += (a[t] @ state) * k^T // delta rule bonus
  state += v[t] @ k[t]^T       // key-value update
  output[t] = r[t] @ state     // readout
```

State is a `[head_size, head_size]` matrix per head per RWKV-7 layer — fixed size regardless of sequence length.

### Attention Blocks

Standard multi-head attention with KV cache. Uses cached keys and values up to a configurable max length (default 2048). No RoPE — position information comes from the surrounding RWKV-7 blocks.

## Weight Format

RWKV-X uses safetensors (not GGUF). Weights are loaded via Foundry's `fs_open()` (zero-copy mmap) and routed by prefix:

| Prefix | Block Type |
|--------|-----------|
| `rwkv.blocks.N.*` | RWKV-7 recurrent block N |
| `moba.blocks.N.*` | Sparse attention block N |
| `emb.weight` | Token embeddings |
| `head.weight` | LM head |
| `ln_out.*` | Final layer norm |

Weight transformations at load time:
- Linear weights are transposed (safetensors stores `[out, in]`, inference needs `[in, out]`)
- Singleton dimensions are squeezed
- `att.r_k` is flattened from `[n_head, head_size]` to `[n_embd]`
- Embeddings are pre-normalized (L2 norm per row)

## INT8 Quantization

When `--quantize` is passed, all linear projection weights are quantized to INT8 at load time using per-row symmetric quantization:

```
scale[row] = max(abs(row)) / 127
quantized[row][col] = round(float[row][col] / scale[row])
```

Inference uses a NEON-accelerated INT8 dot product kernel (`fast_matvec_q8`) that pre-quantizes the input vector once, then computes INT8 dot products for all output rows. This gives ~3x speedup on ARM.

## Tokenizer

RWKV-X uses a **trie-based World tokenizer** with 65,536 tokens. The tokenizer operates on raw UTF-8 bytes with greedy longest-match — fundamentally different from BPE or SentencePiece.

The vocab is loaded from a text file (`rwkv_vocab_v20230424.txt`) via the `--vocab` CLI flag. Each line contains a token ID, token length, and the token's byte representation as a Python byte literal.

The tokenizer is stored inside the RWKV-X model struct (not in `pn_model->tokenizer`). The engine routes to it via `pn_rwkvx_encode()`/`pn_rwkvx_decode()` when `config.arch == PN_ARCH_RWKVX`.

## Dual State

Each inference session maintains two state types in `pn_state->kv_cache`:

- **Recurrent state**: Fixed-size per-layer matrix `[n_heads, head_size, head_size]` for RWKV-7 blocks
- **KV cache**: Key-value store for sparse attention blocks (capped at `max_cache_len`)
- **Workspace**: Pre-allocated scratch buffers (16 buffers + ping-pong pair + INT8 buffer)

All state is encapsulated in a single opaque struct. `pn_state_reset()` zeros both the recurrent state and KV cache.

## Adapter API

The adapter layer in `rwkvx.c` maps Pantheon's interface to internal RWKV-X functions:

| Pantheon API | Internal Function |
|-------------|-------------------|
| `pn_rwkvx_load()` | `rx_model_load_internal()` |
| `pn_rwkvx_free()` | `rx_model_free_internal()` + tokenizer free |
| `pn_rwkvx_forward()` | `rx_forward_internal()` |
| `pn_rwkvx_state_new()` | `rx_state_new_internal()` |
| `pn_rwkvx_state_free()` | `rx_state_free_internal()` |
| `pn_rwkvx_state_reset()` | `rx_state_reset_internal()` |
| `pn_rwkvx_encode()` | `rx_tokenizer_encode()` |
| `pn_rwkvx_decode()` | `rx_tokenizer_decode()` |

## Usage

```bash
# FP32 inference
./pantheon --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" --max-tokens 32

# INT8 quantized inference (~3x faster decode on ARM)
./pantheon --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" --max-tokens 32 --quantize
```

## What Was Stripped

The Foundry RWKV-X engine has additional features not yet in Pantheon:

| Feature | Why Stripped |
|---------|-------------|
| Metal GPU (`fm_*`) | Pantheon doesn't have Metal model integration yet |
| CUDA GPU (`fc_*`) | Pantheon doesn't have CUDA model integration yet |
| HybridMem (`hm_*`) | Separate feature, not part of base inference |
| Training (`rx_forward_ex`, callbacks) | Training not in Pantheon scope yet |
| Batch forward (`rx_forward_batch`) | Prefill optimization, add later |
| Instrumentation (attn_log, embed_capture) | Debug tooling, add later |
| OpenMP (`#pragma omp`) | Pantheon uses threadpool, not OpenMP |

These can be added incrementally in future phases.
