# qwen35moe — Forward-Pass Design (Pantheon port)

Implementation blueprint for `src/models/qwen35moe.{h,c}`. Hybrid **Gated-Delta-Net SSM + sparse full-attention + shared-expert MoE** (Qwen3-Next lineage). Synthesized from llama.cpp `qwen35moe.cpp` + `delta-net-base.cpp` (math reference) and Pantheon `rwkvx.c` (structural template).

**Structure clones `rwkvx.c`; loading follows the GGUF/llama path (`pn_qweight`, `gguf.c`), NOT rwkvx's safetensors path.** Hybrid of both.

## Config (mini; Pro is the same arch, bigger)

| dim | value | source key |
|---|---|---|
| n_layer | 40 | `block_count` |
| full_attention_interval | 4 → **attn when `(i+1)%4==0`** (layers 3,7,…,39 = 10 attn; rest = 30 SSM) | `full_attention_interval` |
| n_embd | 2048 | `embedding_length` |
| n_head / n_head_kv | 16 / 2 (GQA) | `attention.head_count[_kv]` |
| head_dim | 256 | `attention.key_length`/`value_length` |
| rope_dim | **64** (partial — only first 64 of 256 rotated) | `rope.dimension_count` |
| n_expert / used | 256 / 8 (+1 shared) | `expert_count`/`expert_used_count` |
| expert FF / shared FF | 512 / 512 | `expert_feed_forward_length`/`…shared…` |
| **SSM:** state (S) | 128 | `ssm.state_size` |
| SSM n_group (K-heads) | 16 | `ssm.group_count` |
| SSM dt_rank (V-heads) | 32 | `ssm.time_step_rank` |
| SSM d_inner | 4096 | `ssm.inner_size` |
| SSM conv kernel | 4 | `ssm.conv_kernel` |
| conv_dim | key·2+value = 2048·2+4096 = **8192** | derived |

## Block scaffold — pre-norm sandwich (every layer, both types)

```
x → attn_norm → [SSM | full-attn block] → +x → post_attention_norm → shared-MoE FFN → +(post-attn x)
```
Final `output_norm` once after the stack → LM head. All norms RMS. Residuals add the *un-normalized* input. (Identical to llama.cpp's loop.)

## Full-attention block (layers `(i+1)%4==0`)
Input is `attn_norm`'d. Op order:
1. **Joint Q+gate** proj `attn_q` → `[2·head_dim·n_head]`; split: Q = first `head_dim` per head, gate `z` = second half.
2. **QK-RMSNorm on Q** (`attn_q_norm`, per-head, **before RoPE**).
3. `attn_k`,`attn_v` proj; reshape K to heads; **K-RMSNorm** (`attn_k_norm`, pre-RoPE).
4. **Partial RoPE** on Q,K — rotate only first `rope_dim=64` of each 256-dim head, pass the rest through (multi-section / IMRoPE; sections from `rope.dimension_sections`).
5. **GQA attention** (16 Q / 2 KV heads), scale `1/√head_dim`.
6. **Output gate:** `cur = attn_out * sigmoid(z)`  ← note: **sigmoid** gate here.
7. Output proj `attn_output`. Residual added in the main loop.

## SSM block — Gated-Delta-Net (layers `(i+1)%4!=0`)
Input is `attn_norm`'d. **This is the hard new kernel.**

**Projections** (4 separate, from normed `x`):
- `attn_qkv` → fused **q|k|v** stream `[8192]` (the thing that gets convolved).
- `attn_gate` → **z** `[4096]` (output gate, used at the end).
- `ssm_beta` → `beta [32]`, then **`β = sigmoid(beta)`**.
- `ssm_alpha` → `alpha [32]`.

**Decay gate (scalar per V-head, "GDA"):**
```
g = (-exp(A_log)) · softplus(alpha + dt_bias)      # ssm_a stored as -exp(A_log) (negative); ssm_dt = dt_bias
```
(`ssm_a` is consumed via plain multiply — pre-baked at load.)

**Causal conv1d** (`ssm_conv`, k=4): conv over the full 8192-channel q|k|v stream prepended with `conv_state` (last 3 timesteps) → **SiLU**. Then split into q`[128×16]`, k`[128×16]`, v`[128×32]`; **L2-normalize q and k** over head dim (not RMS); GQA-repeat q,k 16→32 heads.

**Per-token recurrence** (per V-head `h∈32`, state `S` is `[128,128]`, `i`=key dim, `j`=value dim):
```
S'      = exp(g_t) · S_{t-1}                         # (a) scalar decay, whole state
pred_j  = Σ_i S'[i][j] · k_t[i]                       # (b) read decayed state under key
delta_j = (v_t[j] − pred_j) · β_t
S_t[i][j] = S'[i][j] + k_t[i] · delta_j               # (c) rank-1 delta-rule write
y_t[j]  = (1/√128) · Σ_i S_t[i][j] · q_t[i]           # (d) read-AFTER-write under query
```
**Scaling caution:** apply the q/output scale **once** — the autoregressive (single-token) path is what we want for decode; use the form above and do *not* also pre-scale q.

**Output path:**
```
gated   = RMSNorm(y, ssm_norm) · silu(z)             # gated RMSNorm — silu(z), NOT sigmoid
cur     = ssm_out · gated                            # proj [4096→n_embd]
```
Residual added in the main loop.

## Shared-expert MoE FFN (every layer)
On the `post_attention_norm`'d tensor:
```
# routed:
moe_out = MoE(x, ffn_gate_inp, {ffn_gate/up/down_exps}, n_expert=256, used=8,
              gate=SOFTMAX, norm_topk_prob=TRUE, scale=expert_weights_scale)   # SwiGLU experts
# shared (always on):
shexp   = SwiGLU(x, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp)
sgate   = sigmoid(ffn_gate_inp_shexp · x)            # 1 scalar/token
out     = moe_out + sgate · shexp
```
Extends Pantheon's existing Mixtral MoE (routed top-k SwiGLU) with: **softmax+norm_topk_prob gating** and the **sigmoid-gated shared-expert** add.

## Dual state (per sequence, per layer)
- **SSM layers (×30):** conv_state `(4-1)·8192 = 24,576` f32 + ssm_state `128·4096 = 524,288` f32 = **548,864 f32/layer**.
- **Attention layers (×10):** KV cache, sized by `n_head_kv=2 × head_dim=256` per token (the *only* growing state).
- Mirror `rwkvx.c`'s dual top-level struct: parallel per-type arrays indexed by per-type index, + `block_order[]`/`block_index[]`.

## Pantheon mapping (from `rwkvx.c`)

**Clone wholesale:** dual-state top struct; `block_order`/`block_index` interleave + forward dispatch; model = per-type block-struct arrays + emb/output_norm/head; forward skeleton (embed → ping-pong per-layer loop → final norm → head); engine hooks (add `PN_ARCH_QWEN35MOE` enum + branches in the ~6 `engine.c` sites: `pn_load_ex`, `pn_get_forward_fn`/direct-set, `pn_state_new/free/reset`, `pn_model_free`, generate/stream tokenizer routing); adapter shim mirroring `rwkvx.h`; threadpool TLS; own preallocated workspace (not the arena); q8 matvec infra (reuse for expert matrices).

**Add (new):** SSM state struct (conv + ssm); the Gated-Delta kernel (recurrence above); shared-MoE FFN; **RoPE + GQA** on the attention block (rwkvx's attn block has neither — borrow RoPE from the llama path).

**Replace:** RWKV time-mix → SSM kernel; attn internals (add RoPE/GQA, sigmoid output gate); FFN → shared-MoE; RWKV trie tokenizer → Qwen **BPE** (`ft_tokenizer` via the GGUF path); `head_size=64` hardcode → config-driven.

**Loader:** GGUF (not safetensors). Parse all `qwen35moe.*` keys; honor `nextn_predict_layers=0` and never load `blk.N.nextn.*`. Layer type from `(i+1)%4==0`. Per-layer tensors via the gguf/`pn_qweight` conventions.

## Implementation order (= Bucket A / Phase 1 A2–A8)
1. **A2** arch+config+loader (detect, parse, GGUF, layer-type, nextn).
2. **A3** full-attention block (closest to known territory — GQA+QK-norm+partial RoPE+sigmoid gate).
3. **A4** SSM kernel (conv → recurrence → gated-norm). Validate this op in isolation first (hardest).
4. **A5** shared-MoE FFN (extend Mixtral path).
5. **A6** assemble hybrid + dual state + engine hooks.
6. **A8** validate vs the llama.cpp oracle (`temp 0`, `oracle_reference.md`).

## Correctness watch-list
- Decay is **scalar per V-head** (GDA), not per-dim.
- conv runs over the **full 8192** stream, *then* split; q,k get **L2-norm** post-conv.
- Partial RoPE: only first **64** dims.
- Attention layer gate = **sigmoid**; SSM layer gate = **silu** (gated RMSNorm). Don't mix them up.
- Recurrence is **read-after-write** (output reads post-update state).
- Apply the q/√S scale **once**.
- `norm_topk_prob=true` (renormalize the top-8 weights).
- **GQA head broadcast in the SSM (THE A8 gibberish bug — RESOLVED):** the 16 K/Q heads repeat over 32 V-heads via `ggml_repeat`, which **interleaves**: V-head `vh` pairs with K-head `vh % n_k_heads`, i.e. `vh = r*n_k_heads + kh` — **NOT** the blocked `vh = kh*kv_groups + r`. Same trap will appear in the GPU MoE/SSM port. Verified by per-layer activation bisection vs `llama-eval-callback`.
- **ChatML reasoning prefill:** assistant turn opens with `<think>` (`add_generation_prompt`); without it the first generated token diverges from the oracle. Pantheon: `PN_CHAT_CHATML_THINK`.
- ~~QK-norm / ssm_norm `(1+w)`~~ **RESOLVED (not the bug):** GGUF norm weights are ~1-centered (means 0.88–2.59, not ~0), so the converter baked `+1` into them → plain RMS is correct, matches llama.cpp. A8 should look at the **assembly wiring** (residual placement, norm points, kernel-handoff tensor layouts, state init), not the norm convention.

## A2 findings — apply in A3+ (verified against the real GGUF)

- **Exact tensor names** (the §blocks above used shorthand): `ssm_conv1d.weight`, `ssm_a` (no `.weight`), `ssm_dt.bias`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_norm.weight`, `ssm_out.weight`; attention uses `attn_q/k/v.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`, `attn_output.weight`, `attn_gate.weight`, `attn_qkv.weight`; MoE `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `ffn_gate_inp_shexp`. **A2 loads by these real names.**
- **Q3_K added to the CPU path** — the engine lacked Q3_K dequant/`pn_qmv` (only Q4_K/Q5_K/Q6_K). A2 added it (validated: sane stats, no NaN). *Still needs oracle-level verification at A8 — a subtle bit error would only show as output mismatch.*
- **`rope.dimension_sections`** (the multi-section/IMRoPE array) is **not parsed in A2** — only `rope.dimension_count=64`. **A3 must parse the sections array** for partial multi-section RoPE.
- **`token_embd`** kept as dequantized F32 (owned, phi3 convention); `output.weight` (Q6_K) zero-copy `pn_qweight` with tied-embedding fallback.
- **File-split discipline (whale prevention):** A2 is one 607-LOC file. **A3 → `qwen35moe_attn.c`, A4 → `qwen35moe_ssm.c`, A5 → `qwen35moe_moe.c`** — do NOT grow `qwen35moe.c` into a 4th >1000-LOC whale. Keep `qwen35moe.c` for the model struct + load + forward orchestration.

## K80 GPU decode — status & performance

**N2-mini (Q3_K_M) runs on a K80 host** via `pn_qwen35moe_forward_cuda`, split across the
two dies of one card (small-BAR — both dies mapped at once, no barswap): front layers
`[0,split)` + token_embd on die A, back layers + output_norm + LM head on die B. Select with
`QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1` (no barswap acquire —
`QWEN35_GPU_BARSWAP=1` opts back into the legacy single-slot path).

**Verified single-stream decode: ~13.7 tok/s** (~0.073 s/token, 2 dies, Q3_K_M), and the GPU output
**bit-matches the CPU forward** — argmax `11,1496,271,142,79,88,30,81,30,271,220,381`, token-0
logit 13.7252, every per-token logit ulp-0 identical. Up from a 0.2 tok/s first-light baseline (~68×).
K80 autoboost swings this ±~30% run-to-run. Actively being optimized (see headroom).

**What got it there** (all committed):
- GPU matvec kernels for **every quant type** (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/F32) — the mixed-quant
  model was host-falling-back to `pn_qmv` (3.7 s of 5.2 s/token, led by the 248K-vocab Q6_K LM head).
- **Warp-per-row** matvec (one warp/row, 32 lanes coalesce each 256-elem superblock; was
  thread-strided → scattered 32-byte segments at 2–8 GB/s → 14–41 GB/s).
- **Grouped MoE** — all 8 top-k experts' gate+up+SwiGLU in one launch + fused down/accumulate (~52→10 ops/layer).
- **Parallel router top-k** (the serial thread-0 selection sort was ~134 ms/token over 40 layers).
- **GPU-resident SSM state** (kills the per-token state round-trip) + per-die scratch pool (kills ~800 alloc/free per token).
- **Quant weights repacked at upload** (`fc_repack_weight`) into an aligned, scale-pre-unpacked,
  transposed device layout → coalesced `u32` loads + shared-memory activation staging. Matvec
  device-bandwidth ~2× (Q8_0 21→65, Q5_K 16→35, Q3_K 15→30 GB/s), +2 GB/die. Bit-exact (CPU-replica
  verifier, ulp 0).

**Ceiling & headroom:** post-repack the matvecs are **instruction-issue-bound on Kepler** — ~65–70 GB/s
is near the sm_37 limit for *bit-exact* K-quant decode (the earlier ~170 GB/s figure was optimistic).
The remaining single-stream lever is **per-layer kernel fusion** to cut the ~18–20 ms/token of
tiny-kernel launch overhead (~100 F32 router/α/β + 94 Q8_0 launches, ~80 µs each vs ~35 µs of work)
→ ~15–20 tok/s. Beyond that, **"hundreds" is not a single-stream number on Kepler** — it comes from
**speculative block-verify** (DeepSeek DSpark: amortize the bandwidth-bound batch-1 decode across a
verified block) and **continuous batching** across concurrent users (serving-aggregate throughput).

## Batched prefill (the serving foundation)

`pn_qwen35moe_forward_batch_cuda` runs **n tokens of one sequence in a single pass**: token-major
`[n][dim]` activations, every per-token matvec batched into one `fc_matvec_*_nx` launch (each weight
block decoded once per 4-column tile — the dequant instruction stream amortizes over the batch), SSM
conv1d + delta-rule recurrence as **in-kernel scans** over the n positions, causal host GQA attention
per position, per-token MoE routing in batched grouped launches. Single-token decode path untouched.

**Bit-exact check** (2-die split @20, measured with an out-of-tree harness): per-position logits of a batched
12-token prefill are **memcmp-identical (248320 floats × 12)** to the sequential single-token forward
(compare with `QWEN35_FUSE=0x37` — the fused attn qknorm+rope kernel is ~1 ulp off the standalone
pair, a pre-existing fusion contraction property). Prefill → decode continuity verified: one decode
step on top of the batched state matches the fully sequential run exactly.

**Measured prefill throughput** (2-die split @20, boost clocks): n=32 — sequential 15.8 tok/s vs
batched **32.2 tok/s (2.04×)**; n=128 — sequential 17.8 tok/s vs batched **36.5 tok/s (2.05×)**.
Kernel-level nx speedups are 1.7–3.1× per pass (`foundry/cuda/tests/test_matvec_nx`). Remaining
prefill levers: wider column tiles, batching the host attention/Q-split glue onto the die, and
expert-grouped MoE batching (tokens sharing an expert stream its weights once).

**Host bring-up prereqs** (else CUDA falls back to CPU or the box hangs): quirk kernel (16-die
small-BAR map), `nvidiafb` blacklisted, `NVreg_EnableMSI=0`, `NVreg_DynamicPowerManagement=0`
(all on the GRUB cmdline), persistence mode — see `foundry/barswap/k80_quirk/README.md`. Profiler: `QWEN35_PROF=1`.

**Operational:** under sustained high-util load a marginal card can Xid-79 "fall off the bus"
(power/thermal — visible only now that the dies actually work); mitigate with `nvidia-smi -pl 110`
and/or forced-air cooling (K80s are passively cooled).
