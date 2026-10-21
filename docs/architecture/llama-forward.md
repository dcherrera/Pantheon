# Llama Forward Pass

The Llama forward pass (`src/models/llama.c`) is shared by Llama, Qwen2, Mistral, and Mixtral architectures. Architecture-specific differences (attention biases, sliding window, MoE routing, metadata keys) are handled by config flags, not separate code paths.

Phi-3 has its own forward pass in `src/models/phi3.c` due to fused QKV/gate+up weights and NeoX-style RoPE. See the architecture differences table below.

## Forward Pass Flow

```
token (uint32)
  │
  ├── embed: tok_embd[token] → hidden[n_embd]
  │
  ├── for each layer (0..n_layer-1):
  │   │
  │   ├── RMSNorm(hidden, attn_norm)
  │   │
  │   ├── Attention:
  │   │   ├── Q = Wq @ normed  (+bias for Qwen2)
  │   │   ├── K = Wk @ normed  (+bias for Qwen2)
  │   │   ├── V = Wv @ normed  (+bias for Qwen2)
  │   │   ├── RoPE(Q, K, pos, rope_theta)
  │   │   ├── Write K,V to KV cache at pos
  │   │   ├── GQA: for each query head
  │   │   │   ├── scores = Q_head @ K_cache^T / sqrt(head_dim)
  │   │   │   ├── causal mask (future tokens = -inf)
  │   │   │   ├── softmax(scores)
  │   │   │   └── output = scores @ V_cache
  │   │   └── O = Wo @ concat(heads)  (+bias for Qwen2)
  │   │
  │   ├── residual: hidden += O
  │   │
  │   ├── RMSNorm(hidden, ffn_norm)
  │   │
  │   ├── FFN (standard) or MoE FFN (Mixtral):
  │   │   │
  │   │   ├── Standard SwiGLU (n_experts == 0):
  │   │   │   ├── gate = W_gate @ normed
  │   │   │   ├── up   = W_up   @ normed
  │   │   │   ├── act  = silu(gate) * up
  │   │   │   └── down = W_down @ act
  │   │   │
  │   │   └── MoE routing (n_experts > 0, Mixtral):
  │   │       ├── gate_logits = W_router @ normed    → [n_experts]
  │   │       ├── probs = softmax(gate_logits)
  │   │       ├── top_k = select top n_experts_used
  │   │       ├── weights = renormalize(selected probs)
  │   │       ├── for each selected expert e:
  │   │       │   ├── gate_e = expert[e].W_gate @ normed
  │   │       │   ├── up_e   = expert[e].W_up   @ normed
  │   │       │   ├── act_e  = silu(gate_e) * up_e
  │   │       │   └── out_e  = expert[e].W_down @ act_e
  │   │       └── down = Σ (weight_e * out_e)
  │   │
  │   └── residual: hidden += down
  │
  ├── RMSNorm(hidden, output_norm)
  │
  └── logits = W_output @ normed  →  float[n_vocab]
```

## Key Operations

### RMSNorm

```
rms = sqrt(mean(x^2) + eps)
out = (x / rms) * weight
```

Implemented as a single SIMD kernel (`pn_vrms_norm_f32`). Default `eps = 1e-5`, configurable per model via GGUF metadata.

### RoPE (Rotary Position Embeddings)

Applies learned rotation to Q and K vectors based on position:

```
for each pair (i, i+1) in head:
    freq = 1.0 / (theta ^ (2i / head_dim))
    angle = pos * freq
    q[i], q[i+1] = rotate(q[i], q[i+1], angle)
```

`rope_theta` defaults to 10000 (Llama). Qwen2.5 uses 1,000,000 for extended context.

### Grouped Query Attention (GQA)

GQA reduces KV heads relative to query heads. Each KV head is shared across `n_head / n_kv_head` query heads:

| Model | Q Heads | KV Heads | Ratio |
|-------|---------|----------|-------|
| SmolLM2-135M | 9 | 3 | 3:1 |
| Qwen2.5-0.5B | 14 | 2 | 7:1 |
| Mistral-7B | 32 | 8 | 4:1 |
| Phi-3-mini | 32 | 32 | 1:1 (MHA) |

The forward pass maps query head `h` to KV head `h * n_kv_head / n_head`.

### SwiGLU FFN

```
gate = W_gate @ x
up   = W_up   @ x
out  = W_down @ (silu(gate) * up)
```

The `silu(gate) * up` step is fused into a single SIMD kernel (`pn_vsilu_mul_f32`).

### Sliding Window (Mistral)

When `sliding_window > 0`, the KV cache operates as a circular buffer:

- Write position: `pos % max_seq` (wraps around)
- Readable length: `min(pos + 1, max_seq)` (prevents reading uninitialized slots)
- `max_seq = min(sliding_window, n_ctx)`

This limits memory to a fixed window size regardless of sequence length. Standard Llama/Qwen2 use `sliding_window = 0` (full context).

### Mixture of Experts (Mixtral)

When `n_experts > 0`, the FFN sublayer is replaced with expert routing:

```
normed = RMSNorm(hidden)
gate_logits = W_router @ normed                   → [n_experts]
probs = softmax(gate_logits)
top_idx, top_wt = top_k(probs, n_experts_used)    → select best experts
top_wt = renormalize(top_wt)                       → weights sum to 1

ffn_out = 0
for each selected expert e:
    gate = expert[e].W_gate @ normed
    up   = expert[e].W_up   @ normed
    hid  = silu(gate) * up
    out  = expert[e].W_down @ hid
    ffn_out += top_wt[e] * out

hidden += ffn_out
```

**Expert weight formats in GGUF:**

Merged (most common): 3D tensors sliced into per-expert views (zero-copy mmap)
- `blk.N.ffn_gate_exps.weight` `[n_experts, n_ff, n_embd]`
- `blk.N.ffn_up_exps.weight` `[n_experts, n_ff, n_embd]`
- `blk.N.ffn_down_exps.weight` `[n_experts, n_embd, n_ff]` (transposed)

Split (older): individual tensors per expert
- `blk.N.ffn_gate.E.weight` `[n_ff, n_embd]` for E=0..n_experts-1

The loader probes for merged format first, falls back to split.

## Architecture Differences

### Shared Llama Forward Pass (llama.c)

| Feature | Llama | Qwen2 | Mistral | Mixtral |
|---------|-------|-------|---------|---------|
| Attention biases | No | Yes (Q,K,V,O) | No | No |
| Sliding window | No | No | Yes (4096) | No |
| MoE experts | No | No | No | Yes (n_experts, top-k) |
| GGUF key prefix | `llama.*` | `qwen2.*` | `llama.*` | `llama.*` |
| GGUF arch string | `llama` | `qwen2` | `mistral` | `llama` (detected via `expert_count`) |
| RoPE theta | 10000 | 1000000 | 10000 | 10000 |
| RoPE style | Interleaved | Interleaved | Interleaved | Interleaved |
| Q/K/V weights | Separate | Separate | Separate | Separate |
| FFN weights | Separate | Separate | Separate | Per-expert (merged or split) |
| Tied embeddings | model-dependent | model-dependent | No | No |

Note: Mistral GGUFs use `llama.*` metadata keys (not `mistral.*`), but `general.architecture = "mistral"`. Mixtral GGUFs use `general.architecture = "llama"` and are detected via `llama.expert_count > 0`.

### Phi-3 (phi3.c — separate forward pass)

| Feature | Phi-3 |
|---------|-------|
| GGUF key prefix | `phi3.*` |
| Q/K/V weights | Fused `attn_qkv.weight` [3*n_embd, n_embd] |
| Gate/Up weights | Fused `ffn_up.weight` [2*n_ff, n_embd] |
| RoPE style | NeoX (split-half pairs: `head[i], head[i+half]`) |
| RoPE dim | Configurable via `rope.dimension_count` (96 for mini) |
| Attention | MHA (n_head == n_kv_head = 32) |
| Tied embeddings | No (separate output.weight) |

Phi-3 uses its own forward pass because the fused weight layout requires different matvec + split logic, and NeoX RoPE pairs elements differently from the interleaved style used by Llama/Qwen2/Mistral.

## Weight Storage

Two modes, selected automatically based on quant type support:

### Quantized (default)

All projection weights (`Wq`, `Wk`, `Wv`, `Wo`, `W_gate`, `W_up`, `W_down`, `W_output`) stay in mmap'd quantized form as `pn_qweight` structs. Matrix-vector multiply happens via `pn_qmv()` which dequantizes one block at a time.

Embeddings and norms are always F32 (owned memory), since they're accessed by index (embeddings) or element-wise (norms).

### F32 legacy

All weights dequantized to `tm_tensor` on load. Fallback for unsupported quant types. Uses standard `tm_matmul` for projections.

## Per-Token Memory

All intermediate buffers are allocated from a bump arena that resets after each token:

| Buffer | Size | Used For |
|--------|------|----------|
| `normed` | n_embd | RMSNorm output |
| `q`, `k`, `v` | n_embd | Q/K/V projections |
| `attn_out` | n_embd | Attention output |
| `ffn_gate`, `ffn_up` | n_ff | FFN intermediates |
| `ffn_act` | n_ff | SiLU activation |
| `ffn_out` | n_embd | FFN output |
| `logits_raw` | n_vocab | LM head output |
| `scores` | max_seq per head | Attention scores |

For MoE layers (Mixtral), additional per-layer buffers:

| Buffer | Size | Used For |
|--------|------|----------|
| `gate_logits` | n_experts | Router output |
| `ffn_out` | n_embd | Weighted expert accumulator |
| `exp_gate` | n_ff | Per-expert gate projection |
| `exp_up` | n_ff | Per-expert up projection |
| `exp_hid` | n_ff | Per-expert SiLU activation |
| `exp_out` | n_embd | Per-expert down projection |

Arena is sized at state creation: `sum of all buffers * sizeof(float) * 1.25` (25% safety margin). Typical sizes: 1.5 KB (SmolLM2-135M), 11 KB (Mistral-7B), 19 KB (Mixtral_7Bx2).

## Batched Multi-Sequence Decode

`pn_llama_forward_batch()` decodes one token for each of `B` independent sequences in a single pass, for bulk throughput (e.g. transcribing many pages at once). Each `states[b]` carries its own KV cache and position, so the sequences can have different lengths and each attends only to its own cache; row `b`'s input is `tokens[b]` or, for image-token positions, an `embd_rows[b]` override. It writes `[B, n_vocab]` row-major logits and advances every state's `pos` by one.

The batched path is restricted to the quantized, non-MoE Llama/Qwen weights. For MoE, F32 legacy, or ternary weights it returns without mutating any state so the caller can fall back to per-sequence `pn_llama_forward`. Results are bit-identical to running `B` separate sequential decodes (verified by `test_batch_decode_correct`: a `B=2` batch reproduces each sequence's standalone greedy stream with no cross-contamination).

### Fused AMX-panel M=B matmul (`pn_qmm`)

The projection and FFN matmuls in the batched path go through `pn_qmm` (`src/core/qweight.c`), a fused quantized `M=B` kernel: `out[b, r] = sum_c dequant(W[r, c]) * acts[b, c]`. Each weight row is dequantized once into a small L2-resident panel and multiplied against all `B` activation vectors, so the quantized weight is read and dequantized once per step while the matmul runs at AMX throughput (the panel GEMM lowers to AMX under BLAS). This avoids both the NEON-dot ceiling of the single-vector `pn_qmv` and the full-weight F32 copy to DRAM that a plain BLAS GEMM would need.

Because the per-token matmul FLOPs are irreducible, throughput is compute-bound and plateaus rather than scaling linearly with `B`. Measured on LightOnOCR 1B Q4_K, M4, against 38.8 tok/s single-stream `pn_qmv`:

| path | B=16 | B=32 |
|------|------|------|
| full-BLAS F32 copy | 49.7 | 46.5 (DRAM-bound, regresses) |
| NEON fused dot | 58.4 | - |
| AMX panel (`pn_qmm`) | 72.5 | 84.3 |

The AMX panel gives roughly 2.2x aggregate CPU throughput and settles near ~85 tok/s. `B=1` is intentionally suboptimal (the panel adds a scratch pass over the fused `pn_qmv`), so single sequences should use `pn_llama_forward`. `pn_qmm` falls back to a per-`b` `pn_qmv` loop above `PN_QMM_MAX_B` (64) or for ternary or GPU-resident weights.

## Flash (Blocked) Prefill Attention

Prefill over a long image-token sequence was the bulk-OCR bottleneck. The original per-query attention recomputed over the whole K/V cache once per query position, giving O(P^2) DRAM traffic across the ~1369 image-token queries of a page.

`prefill_flash_range` processes a block of queries (`PF_BQ=32`) together and streams K and V in blocks (`PF_BK=256`) that stay cache-resident and are reused across the whole query block. This cuts K/V reads from about `P` times to about `P/BQ`. The math is unchanged from the per-query path (full-precision softmax, with the scores tile kept L2-resident), and a per-query fallback is retained for the sliding-window wraparound case. It is parallelized over query heads.

Measured: prefill ~4.3 to ~3.2 s/page (~1.33x), bulk OCR ~6.3 to ~5.2 s/page steady state. It is pure CPU, so it helps regardless of the vision backend and without contending for the GPU (unlike GPU prefill below). Output stays correct and batched decode stays bit-identical.

## Optional GPU Paths (off by default)

Two Metal offload paths exist behind build flags and environment switches. Both are **off by default** because, on the current hardware, they lose to the CPU AMX path or contend with the vision GPU. The shipped app builds Pantheon without `USE_METAL`, so both are inert there.

- **GPU batched decode** (`PN_BATCH_GPU`, gated at runtime by `pn_qmm_set_gpu_decode`): uploads the Q4_K decoder weights once (`pn_qmv_gpu_upload`) and runs the `M=B` GEMM on the GPU (`fm_sgemm_q4_k`). Bit-identical to the CPU path, but not a win: `pn_qmm` stages, dispatches, syncs, and copies back per weight, so a step does ~168 CPU/GPU round-trips and the per-call sync latency dominates (~68 tok/s at `B=16` vs ~86 tok/s for the CPU AMX panel). Auto-batching cannot hide it because each GEMM's result is immediately consumed by a CPU op (bias, RoPE, attention). Beating AMX would need a GPU-resident decode (the whole layer chain on GPU, one sync per step); the flag is kept as the seam for that.

- **GPU F16 prefill** (`PN_GPU_PREFILL`): uploads the Q4_K decoder weights once and runs each prefill projection as an F16 MPS GEMM (`fm_gemm_q4_k_f16`) instead of the CPU dequant+AMX path. Also a net loss for bulk OCR, for two reasons: prefill is attention-bound, not GEMM-bound (the O(P^2) decoder attention over the image tokens dominates, so GPU-accelerating only the GEMMs is ~1.3x, not the ~5x a GEMM microbench suggests), and it contends with Core ML vision for the single GPU, breaking the CPU-compute / GPU-vision overlap that makes bulk fast (~6.3 s/page went to ~8-10 s/page). It is kept as an opt-in for the CPU-vision case, where the GPU is otherwise idle. Setting it also forces decode back onto the CPU AMX path (via the `pn_qmm_set_gpu_decode` gate) so the prefill upload does not drag decode onto the slower GPU route.

ABI note: `pn_qweight.gpu` is an unconditional struct field, present in every build (just `NULL` and unused without `USE_METAL`). It must not be wrapped in `#ifdef USE_METAL`: a conditional field makes the struct layout depend on the build flag, so a translation unit compiled the other way reads later fields at the wrong offset.
