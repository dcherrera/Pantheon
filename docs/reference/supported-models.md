# Supported Models

Models verified to load and generate correctly with Pantheon, with the config each one
needs and a runnable command. This list grows as architecture support lands and new
GGUFs are tested.

Hardware in the **Tested** column is what the row was actually measured on — `M4 CPU` is
Apple Silicon, `K80 GPU` is a Tesla K80 (GK210, `sm_37`) via the CUDA backend, `TT p150a` is a
Tenstorrent Blackhole p150a via the `USE_TT=1` backend.

**Tenstorrent: Ornith-1.5-9B is the only supported model.** Other rows marked `TT p150a` ran on
the card during development; they are not supported there. To run Ornith 1.5 on a card, see the
README's "Ornith 1.5 on a Tenstorrent card".

**Tesla K80: Qwen3.6-35B-A3B and Nex-N2-mini (Q3_K_M, one model per card) are the supported
models**, run with `./k80_qwen36.sh` / `./k80_nexn2.sh` (README "Qwen3.6 and Nex-N2-mini on a
Tesla K80"). Other `K80 GPU` rows ran on the card; they are not supported there.

---

## Llama Architecture

| Model | Quant | Params | Layers | Heads (Q/KV) | Vocab | Context | Tested | Notes |
|-------|-------|--------|--------|--------------|-------|---------|--------|-------|
| SmolLM2-135M-Instruct | Q8_0 | 135M | 30 | 9/3 | 49,152 | 8,192 | M4 CPU | 81.3 tok/s decode, 42ms load. Pre-tokenizer falls back with a warning — see [Known Issues](#known-issues). |
| MiniCPM5-1B-SFT | Q8_0 | 1B | 24 | 16/2 | 130,560 | 131,072 | TT p150a | Ran on the card (not supported there) at 4.37 tok/s (see [`../architecture/accel-foundry.md`](../architecture/accel-foundry.md) §7 for its open correctness defect). Orchestrator. arch="llama" GGUF, head_dim=128 (decoupled: n_embd/n_head=96), rope_theta=5e6, rms_eps=1e-6. No new forward — config-only. ctx capped to 4096 for KV. |

**Source (SmolLM2):** `bartowski/SmolLM2-135M-Instruct-GGUF` on HuggingFace
**Source (MiniCPM5-1B):** `openbmb/MiniCPM5-1B-GGUF` (`MiniCPM5-1B-Q8_0.gguf`, ~1.15 GB, Apache-2.0). Loads into `PN_ARCH_LLAMA` — config-only, no new forward pass.

**Best config:** `make USE_BLAS=1` (no threads needed, model is small)

```bash
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/base/

./pantheon --model models/base/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "Hello, my name is" --max-tokens 32
```

---

## Qwen2 Architecture

| Model | Quant | Params | Layers | Heads (Q/KV) | Vocab | Context | Tested | Notes |
|-------|-------|--------|--------|--------------|-------|---------|--------|-------|
| Qwen2.5-0.5B-Instruct | Q8_0 | 494M | 24 | 14/2 | 151,936 | 32,768 | M4 CPU | Has attention biases, tied embeddings |

**Source:** `Qwen/Qwen2.5-0.5B-Instruct-GGUF` on HuggingFace

**Best config:** `make USE_BLAS=1` (threads may help at this size)

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir models/base/

./pantheon --model models/base/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Hello, how are you?" --max-tokens 32
```

---

## Mistral Architecture

| Model | Quant | Params | Layers | Heads (Q/KV) | Vocab | Context | Sliding Window | GGUF Ver | Tested | Notes |
|-------|-------|--------|--------|--------------|-------|---------|----------------|----------|--------|-------|
| Mistral-7B-Instruct-v0.3 | Q4_K_M | 7.2B | 32 | 32/8 | 32,768 | 32,768 | 4,096 | v3 | M4 CPU | BPE tokenizer, coherent output |
| Mistral-7B-Instruct-v0.1 | Q4_K_M | 7.2B | 32 | 32/8 | 32,000 | 32,768 | — | v2 | M4 CPU | SentencePiece tokenizer, GGUF v2 |

**Source:** `bartowski/Mistral-7B-Instruct-v0.3-GGUF`, `TheBloke/Mistral-7B-Instruct-v0.1-GGUF` on HuggingFace

**Best config:** `make USE_BLAS=1 USE_THREADS=1` (7B model needs threads)

```bash
# v0.3 (BPE, GGUF v3)
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models/base/

./pantheon --model models/base/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --prompt "Hello, how are you?" --max-tokens 32

# v0.1 (SentencePiece, GGUF v2)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
  mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir models/base/

./pantheon --model models/base/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  --prompt "Hello" --max-tokens 32
```

---

## Phi-3 Architecture

| Model | Quant | Params | Layers | Heads (Q/KV) | Vocab | Context | Tested | Notes |
|-------|-------|--------|--------|--------------|-------|---------|--------|-------|
| Phi-3-mini-4k-instruct | Q4_K_M | 3.8B | 32 | 32/32 | 32,064 | 4,096 | M4 CPU | MHA, NeoX RoPE, fused QKV + gate/up |

**Source:** `microsoft/Phi-3-mini-4k-instruct-gguf` on HuggingFace

**Best config:** `make USE_BLAS=1` (threads help at this size)

```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir models/base/

./pantheon --model models/base/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 32
```

---

## Mixtral Architecture (MoE)

| Model | Quant | Params | Layers | Heads (Q/KV) | Vocab | Experts (total/used) | Tested | Notes |
|-------|-------|--------|--------|--------------|-------|---------------------|--------|-------|
| Mixtral_7Bx2_MoE | Q4_K_M | ~13B | 32 | 32/8 | 32,000 | 2/2 | M4 CPU | Merged expert tensors, SentencePiece tokenizer |

**Source:** `TheBloke/Mixtral_7Bx2_MoE-GGUF` on HuggingFace

**Best config:** `make USE_BLAS=1 USE_THREADS=1` (large model needs threads)

**Architecture notes:**
- MoE extends Llama: attention path is identical, only FFN differs
- Mixtral GGUFs use `general.architecture = "llama"`, detected via `llama.expert_count` metadata
- Per-layer FFN: router → softmax → top-k expert selection → SwiGLU per expert → weighted sum
- Supports merged (`ffn_gate_exps.weight` 3D) and split (`ffn_gate.E.weight` per expert) tensor formats

```bash
huggingface-cli download TheBloke/Mixtral_7Bx2_MoE-GGUF \
  mixtral_7bx2_moe.Q4_K_M.gguf --local-dir models/base/

./pantheon --model models/base/mixtral_7bx2_moe.Q4_K_M.gguf \
  --prompt "Hello" --max-tokens 32
```

---

## RWKV-X Architecture (Safetensors)

| Model | Format | Params | Layers (RWKV/Attn) | Heads | Embd | Vocab | Tested | Notes |
|-------|--------|--------|-------------------|-------|------|-------|--------|-------|
| RWKV-X-0.2B-64k-Base | safetensors | 0.2B | 12/4 | 12 | 768 | 65,536 | M4 CPU | FP32 and INT8 verified |
| RWKV-X-3.6B-64k-Base | safetensors | 3.6B | 32/8 | 40 | 2,560 | 65,536 | M4 CPU | 17.2 tok/s INT8, 2.6 tok/s FP32, 8.9 GB RSS |

**Source:** `howard-hou/RWKV-X` on HuggingFace

**Best config:**
- 0.2B: `make USE_BLAS=1` (no threads needed). Add `--quantize` for INT8 (~3x faster decode).
- 3.6B: `make USE_BLAS=1 USE_THREADS=1`. Add `--quantize` for INT8 (~6.4x faster decode).

**Requires:** RWKV World tokenizer vocab file (`--vocab` flag)

```bash
# Download models
huggingface-cli download howard-hou/RWKV-X \
  RWKV-X-0.2B-64k-Base.safetensors --local-dir models/base/

huggingface-cli download howard-hou/RWKV-X \
  RWKV-X-3.6B-64k-Base.safetensors --local-dir models/base/

# 0.2B FP32 inference
./pantheon --model models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" --max-tokens 32

# 3.6B INT8 inference (recommended)
./pantheon --model models/base/RWKV-X-3.6B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "The capital of France is" --max-tokens 64 --quantize
```

**Architecture notes:**
- Hybrid model: RWKV-7 recurrent blocks (WKV-7 kernel) + sparse attention blocks
- 0.2B: 12 RWKV + 4 attention blocks, interleaved [R,R,R,A,R,R,R,A,...]
- 3.6B: 32 RWKV + 8 attention blocks, same interleaving pattern
- Dual state: fixed-size recurrent state per RWKV block + KV cache per attention block
- Trie-based World tokenizer (65K vocab, UTF-8 byte-level)
- Uses safetensors format (not GGUF)

---

## RWKV-7 Architecture (Converted from .pth)

| Model | Source Format | Params | Layers | Heads | Embd | Vocab | Tested | Notes |
|-------|-------------|--------|--------|-------|------|-------|--------|-------|
| RWKV-x070-World-0.1B-v2.8 | .pth → safetensors | 0.1B | 12 | 12 | 768 | 65,536 | M4 CPU | BF16, 399 tensors, converted via `pantheon-convert` |

**Source:** `BlinkDL/rwkv-7-world` on HuggingFace (original .pth format)

**Conversion:** Requires `pantheon-convert` to convert from PyTorch .pth to safetensors first. The .pth reader is a sandboxed pickle interpreter in pure C — no Python needed.

**Best config:** `make USE_BLAS=1` (small model, no threads needed)

**Requires:** RWKV World tokenizer vocab file (`--vocab` flag)

```bash
# Download .pth model
huggingface-cli download BlinkDL/rwkv-7-world \
  RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --local-dir models/base/

# Convert to safetensors (no Python needed)
./pantheon-convert models/base/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  -o models/base/RWKV-7-0.1B.safetensors

# Run inference
./pantheon --model models/base/RWKV-7-0.1B.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello" --max-tokens 32
```

**Architecture notes:**
- Pure RWKV-7 (no sparse attention blocks, unlike RWKV-X hybrid)
- WKV-7 kernel with Generalized Delta Rule
- ChannelMix FFN with squared ReLU
- BFloat16 weights preserved through conversion (no precision loss)
- Uses same inference engine as RWKV-X (shared `pn_rwkvx_*` code path)

---

## qwen35moe Architecture (Hybrid Gated-Delta SSM + Shared-Expert MoE)

| Model | Quant | Params | Layers (SSM/Attn) | Heads (Q/KV) | Head dim | Embd | Experts (total/used+shared) | Vocab | Tested | Notes |
|-------|-------|--------|-------------------|--------------|----------|------|-----------------------------|-------|--------|-------|
| Nex-N2-mini | Q3_K_M | 35B-A3B | 30/10 (interval 4) | 16/2 | 256 | 2,048 | 256/8 + 1 | 248,320 | M4 CPU | **Oracle-validated** vs llama.cpp (layer-by-layer, `temp 0`). Reasoning model (`<think>`). load 376 ms, ~1.0 tok/s decode, 9.5 GB RSS — CPU **reference path**. |
| Nex-N2-mini | Q3_K_M | 35B-A3B | 30/10 (interval 4) | 16/2 | 256 | 2,048 | 256/8 + 1 | 248,320 | **K80 GPU (2-die)** | **Supported on K80** (`./k80_nexn2.sh`). Serve, 2026-10-20: ~21 tok/s decode at 1k context, prefill 107-119 tok/s up to 8k, 74 at 14k, 53 at 30k. Earlier single-stream CLI: ~13.7 tok/s decode — Q3_K_M split 20/20 across the two dies of one K80 (small-BAR, both dies mapped at once). GPU output **bit-matches the CPU forward** (argmax + every per-token logit, ulp 0) with `QWEN35_FUSE=0x37 QWEN35_FLASH=0`, measured by the author; default flash prefill is argmax-exact only. ~68× the CPU path; ~15–20 with kernel fusion (Kepler bit-exact ceiling). See [`../architecture/qwen35moe.md`](../architecture/qwen35moe.md). |

| Qwen3.6-35B-A3B | Q3_K_M | 35B-A3B | 30/10 (interval 4) | 16/2 | 256 | 2,048 | 256/8 + 1 | 248,320 | **K80 GPU (2-die)** | **Supported on K80** (`./k80_qwen36.sh`). Serve, 2026-10-20: ~21 tok/s decode at 1k context, prefill 116-121 tok/s up to 8k, 79 at 14k, 60 at 30k. |

**Source:** `nex-agi/Nex-N2-mini` (GGUF: `bartowski/nex-agi_Nex-N2-mini-GGUF`), `Qwen/Qwen3.6-35B-A3B`
(GGUF: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`)

**Best config (M4, oracle):** `make USE_BLAS=1` (Accelerate) — the correctness reference, not a speed target.

**Best config (Tesla K80, the speed path):** `make USE_CUDA=1`; run the 2-die GPU decode with
`QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 ./pantheon --model <gguf> --prompt <text>`.
~13.7 tok/s single-stream; bit-matching the oracle under `QWEN35_FUSE=0x37 QWEN35_FLASH=0`. Multi-card K80 hosts may need platform setup
(PCI BAR sizing, NVIDIA driver options); Foundry's `barswap/k80_quirk/` documents it.

**Architecture notes:**
- Hybrid (Qwen3-Next lineage): **30 Gated-Delta-Net SSM** layers + **10 sparse full-attention** layers, attention when `(i+1) % 4 == 0`
- **Shared-expert MoE every layer:** softmax + top-8/256 routed experts (SwiGLU, `norm_topk_prob`) **plus** an always-on shared expert with a sigmoid gate
- SSM block: causal conv1d (k=4) → SiLU → L2-norm q/k → GQA head-broadcast → read-after-write delta-rule recurrence (scalar GDA decay) → gated RMSNorm
- Attention block: GQA 16/2, head_dim 256, QK-RMSNorm, **partial NEOX RoPE** (first 64 of 256 dims), sigmoid output gate
- Dual state per sequence: fixed recurrent SSM state (30 layers) + KV cache (10 layers), à la RWKV-X
- Tokenizer: Qwen BPE, `tokenizer.ggml.pre = "qwen35"` pre-tokenizer; ChatML with `<think>` reasoning prefill
- **MTP/nextn trap:** GGUF must carry `nextn_predict_layers=0`; the loader never reads `blk.N.nextn.*`
- The A8 gibberish bug was the SSM GQA head-broadcast tiling (`vh = r*n_k_heads + kh`, interleaved like `ggml_repeat`); see [`../architecture/qwen35moe.md`](../architecture/qwen35moe.md)

```bash
# Interactive (applies ChatML + <think>); --temperature 0 for oracle-style greedy
./pantheon --model models/nex-agi_Nex-N2-mini-Q3_K_M.gguf --chat --temperature 0

# One-shot (chat template applied automatically; --raw to send raw text)
./pantheon --model models/nex-agi_Nex-N2-mini-Q3_K_M.gguf \
  --temperature 0 --max-tokens 90 --prompt "How much does the ball cost?"
```

---

## qwen35 Dense Architecture (Bonsai / Ornith)

A GGUF with `general.architecture = "qwen35"` and a dense FFN (`blk.0.ffn_gate.weight`, no
merged expert tensors) loads as `PN_ARCH_BONSAI` (`src/models/bonsai_forward.c`): the same
Gated-DeltaNet + sparse full-attention hybrid as qwen35moe, with a dense SwiGLU FFN in place of
the MoE. A `qwen35` GGUF with routed experts loads as qwen35moe instead.

| Model | Quant | Params | Layers (SSM/Attn) | Tested | Notes |
|-------|-------|--------|-------------------|--------|-------|
| Ternary-Bonsai-27B | Q2_0 (g128 ternary, GGUF type 142) | 27B | 48/16 | CPU | Oracle-equivalent vs PrismML's llama.cpp fork. |
| Ornith-1.5-9B | Q8_0 | 9B | 24/8 | TT p150a | **The supported Tenstorrent model.** `./install.sh && ./run_ornith.sh`. Serve: 42.9 tok/s decode, 1,630-token prompt in 2.43 s (670 tok/s), MTP speculative decoding on. Tool calls use the JSON dialect ([HTTP API](http-api.md#tool-calling)). |
| Ornith-1.0-9B | Q8_0 | 9B | — | K80 GPU, TT p150a | K80: 9.0 tok/s decode. TT (ran, not supported): output byte-identical to the CPU forward, 1.57 tok/s. Tool calls use the JSON dialect ([HTTP API](http-api.md#tool-calling)). |

**Backends:** CPU; CUDA (single die, falls back to CPU if the upload fails); Tenstorrent
(used only when no other accelerator took the model).

---

## Pixtral Vision Encoder (library only)

`src/models/pixtral_vision.c` runs a Pixtral / Mistral-Small-3.1 vision tower with the
LightOnOCR projector from an llama.cpp-style `clip` mmproj GGUF, producing image embeddings
for a ChatML decoder (`<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` tokens).

- **C API only:** `pn_attach_vision`, `pn_describe` / `pn_describe_stream` / `pn_describe_embeds`,
  `pn_vision_encode_rgb`, and a continuous-batch decoder (`pn_batch_decoder_*`) for bulk OCR
  (`include/pantheon.h`). The CLI and `pantheon serve` do not accept images; HTTP message
  `content` must be a string.
- Decoding goes through the Llama-family path (`pn_llama_prefill` / `pn_llama_forward_batch`).
- Exercised by `tests/bench_vision.c`.

---

## Supported Quant Types

| Type | Bits/Weight | Block Size | Status |
|------|-------------|------------|--------|
| F32 | 32 | — | Supported |
| F16 | 16 | — | Supported |
| Q8_0 | 8 | 32 | Supported |
| Q4_0 | 4 | 32 | Supported |
| Q3_K | 3.4 | 256 | Supported |
| Q4_K | 4.5 | 256 | Supported |
| Q5_K | 5.5 | 256 | Supported |
| Q6_K | 6.5 | 256 | Supported |
| BF16 | 16 | — | Supported |
| Q2_0 (g128 ternary, type 142) | 2.1 | 128 | Supported (Bonsai) |

Q3_K_M, Q4_K_M and Q5_K_M GGUFs mix the K-quant tiers (e.g. Q3_K for most layers with Q4_K/Q6_K for attention) — all supported. Q3_K was added for the Nex-N2-mini bring-up.

---

## Hardware Tested

| Hardware | OS | Build Command | Notes |
|----------|----|---------------|-------|
| Apple M4 | macOS 15 | `make USE_BLAS=1` | Accelerate BLAS, NEON SIMD |
| Apple M4 | macOS 15 | `make USE_BLAS=1 USE_THREADS=1` | Add threads for 1B+ models |
| Tesla K80 ×8 (16 GK210 dies, sm_37) | Ubuntu 24.04 | `make USE_CUDA=1` | CUDA 11.4 driver. N2-mini 2-die GPU decode ~13.7 tok/s. |
| Tenstorrent Blackhole p150a | Linux | `make USE_TT=1` | See [`../architecture/accel-foundry.md`](../architecture/accel-foundry.md) §7. |

---

## Not Yet Tested (Should Work)

These models use supported architectures and quant types but haven't been verified yet.

**Llama family:**
- Llama-3.2-1B-Instruct (Q4_K_M, Q8_0)
- Llama-3.2-3B-Instruct (Q4_K_M)
- SmolLM2-360M-Instruct (Q8_0)
- TinyLlama-1.1B (Q4_0, Q8_0)

**Qwen2 family:**
- Qwen2.5-1.5B-Instruct (Q4_K_M, Q8_0)
- Qwen2.5-3B-Instruct (Q4_K_M)
- Qwen2.5-7B-Instruct (Q4_K_M)
- Qwen2.5-Coder-1.5B-Instruct (Q8_0)

**Mistral family:**
- Mistral-7B-Instruct-v0.2 (Q4_K_M, Q8_0)

**Phi-3 family:**
- Phi-3-mini-128k-instruct (Q4_K_M, Q8_0) — same arch, longer context
- Phi-3-small-8k-instruct (Q4_K_M) — may have different head config
- Phi-3-medium-4k-instruct (Q4_K_M) — larger model

**Mixtral family:**
- Mixtral-8x7B-Instruct-v0.1 (Q4_K_M) — full 8 experts, top-2 routing, ~28 GB

---

## Known Issues

- **Unsupported BPE pre-tokenizers fall back with a one-time warning.** Recognised
  `tokenizer.ggml.pre` values are `qwen35`, `qwen2`, `gpt-2`, `default`, `llama-bpe`, `llama3`,
  `llama-v3` (or unset). Anything else (e.g. SmolLM2's) uses the qwen35/GPT-2 pre-tokenizer and
  prints once: `bpe: tokenizer.ggml.pre is unsupported; falling back to the qwen35/GPT-2
  pre-tokenizer (token IDs may not match the reference for this model)`. Prompt token IDs can then
  differ from the reference tokenizer.

---

## Not Yet Tested (RWKV-X)

No untested RWKV-X models remaining. Both 0.2B and 3.6B verified.

## Not Yet Supported

| Architecture | Status | Blocker |
|-------------|--------|---------|
| Phi-3/4 (other variants) | Phase 3.3+ | Tested mini-4k only |
