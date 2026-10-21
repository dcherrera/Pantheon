# qwen35moe CPU-reference golden vectors

Trusted **CPU-reference** input/output vectors for the four qwen35moe GPU kernels
being written for a K80 host. The CUDA kernels must reproduce these outputs
from the same inputs (within the tolerances below) — the same isolation-test
discipline that already caught the Q3_K dequant high-mask bug and the SSM
head-tiling bug.

- Generator: `tests/gen_golden_qwen35.c`
- Regenerate: `make USE_BLAS=1 golden` (from the Pantheon repo root)
- **No model GGUF is involved.** All inputs are synthetic, produced by a
  fixed-seed xorshift32 PRNG seeded with `0x9E3779B9`. A single RNG is threaded
  through all four kernels in the order they appear below, so the entire output
  set is fully determined by that seed and the generation order.

## File format

Every `*.bin` file is self-describing:

```
line 1 (ASCII, newline-terminated):  PNGOLDEN1 <name> <dtype> <count> seed=0x9E3779B9
remainder:                           <count> raw little-endian <dtype> values
```

`<dtype>` is one of `f32` (4-byte IEEE-754 little-endian), `i32` (4-byte signed
little-endian), or `u8` (raw bytes). Read past the first newline, then read
`count * sizeof(dtype)` bytes of payload. Shapes/semantics are documented
per-file below (the header carries the flat element count only).

## Config constants used

| Constant | Value | Source |
|---|---|---|
| `n_embd` | 2048 | qwen35moe config |
| `n_head` | 16 | qwen35moe config |
| `n_head_kv` | 2 | qwen35moe config |
| `head_dim` | 256 | qwen35moe config |
| `rope_dim` (partial) | 64 | qwen35moe config |
| `rope_theta` (freq base) | 1e7 | model facts (`rope.freq_base`) |
| `rms_norm_eps` | 1e-6 | model facts (`attention.layer_norm_rms_epsilon`) |
| `n_expert` | 256 | qwen35moe config |
| `expert_used` (top-k) | 8 | qwen35moe config |
| `expert_ff` | 512 | qwen35moe config |
| `expert_weights_scale` | 1.0 | GGUF default (no key) |

PRNG seed: `0x9E3779B9`.

---

## Kernel 1 — Q3_K dequant + Q3_K matvec (HIGHEST VALUE)

Golden output produced by the **REAL** CPU functions: `pn_qmv` →
`pn_qdot` → `dot_q3_k` → `pn_dequantize_block_q3_k`
(`src/core/qweight.c`, `src/formats/gguf_dequant.c`). The synthetic weight
matrix is built by filling the real `pn_block_q3_k` struct
(`hmask[32]`, `qs[64]`, `scales[12]`, fp16 `d`) with deterministic
pseudo-random bytes, so every bit pattern is decoded exactly as a GPU kernel
must decode it. Matrix shape: **rows=24, cols=1024** (4 Q3_K super-blocks/row).

| File | dtype | count | shape / meaning |
|---|---|---|---|
| `q3k_matvec_weight.bin` | u8 | 10560 | raw Q3_K blocks, 24 rows × 4 blocks × 110 bytes (row-major) |
| `q3k_matvec_input.bin` | f32 | 1024 | input activation vector (cols) |
| `q3k_matvec_output.bin` | f32 | 24 | `out[r] = W[r,:] · input` (rows) — golden |
| `q3k_dequant_block0.bin` | f32 | 256 | dequantized super-block 0 of the weight (dequant-only isolation) |

**Tolerance:** matvec/dequant on Kepler fp32 — **~1e-4 .. 1e-3 relative**
(per output element). Use the dequant-only file first to localize a mismatch to
the dequant vs. the dot-product.

## Kernel 2 — Partial NEOX RoPE

Golden produced by a **cross-checked replica** of the static helper
`qwen35moe_rope_partial` (`src/models/qwen35moe_attn.c:57-76`) — replicated
term-for-term (cited inline in the generator). Only the first `rope_dim=64`
dims of each `head_dim=256` head are rotated (NEOX pairing `j` vs `j+32`); dims
`[64,256)` pass through unchanged. `theta_base=1e7`. Verified that pos 0 is an
exact identity (angle 0).

Pre-rotation inputs (shared across all positions):

| File | dtype | count | shape |
|---|---|---|---|
| `rope_q_input.bin` | f32 | 4096 | Q, `n_head(16) × head_dim(256)` |
| `rope_k_input.bin` | f32 | 512 | K, `n_head_kv(2) × head_dim(256)` |

Rotated goldens, one pair per position (pos ∈ {0, 1, 7, 100}):

| File | dtype | count | meaning |
|---|---|---|---|
| `rope_q_pos<P>.bin` | f32 | 4096 | Q rotated at position P |
| `rope_k_pos<P>.bin` | f32 | 512 | K rotated at position P |

**Tolerance:** **~1e-4 .. 1e-3 relative** (trig in fp32 on Kepler). Note GPU
`sinf`/`cosf`/`powf` may differ slightly from the host libm; pos 0 should match
to ~0 (identity).

## Kernel 3 — QK-RMSNorm

Golden produced by the **REAL** CPU function `pn_vrms_norm_f32`
(`src/accel/simd.c:198`), applied per head exactly as `qwen35moe_qk_norm_heads`
(`src/models/qwen35moe_attn.c:39-46`) does — one shared `head_dim` norm weight,
`eps=1e-6`. Input is the 16-head Q layout `[n_head(16) × head_dim(256)]`.

| File | dtype | count | shape / meaning |
|---|---|---|---|
| `qknorm_input.bin` | f32 | 4096 | per-head input, `16 × 256` |
| `qknorm_weight.bin` | f32 | 256 | shared per-head norm weight (≈1 ± 0.1) |
| `qknorm_output.bin` | f32 | 4096 | RMSNorm'd output, `16 × 256` — golden |

**Tolerance:** **~1e-4 .. 1e-3 relative**. RMS uses `1/sqrt(mean(x²)+eps)`; the
sum-of-squares reduction order may differ on GPU.

## Kernel 4 — MoE routing

Golden produced by a **cross-checked replica** of the routed-gating path of
`src/models/qwen35moe_moe.c`:
`qwen35moe_router_softmax` (lines 47-61), `qwen35moe_select_topk` (lines 65-89),
`norm_topk_prob` + `expert_weights_scale` (lines 133-136), and the shared-expert
scalar sigmoid gate `sigmoid(gate_w · x)` (lines 179-183). Replicated
term-for-term (cited inline). `n_expert=256`, top-k=8, `expert_weights_scale=1.0`.

Top-k selection is a selection sort with **first-seen tie-break** — the indices
must match **exactly** on GPU; any GPU implementation must use the same
tie-break to be bit-identical here.

Inputs:

| File | dtype | count | meaning |
|---|---|---|---|
| `moe_router_logits.bin` | f32 | 256 | raw router logits (pre-softmax) |
| `moe_shared_gate_weight.bin` | f32 | 2048 | shared-expert gate weight (`n_embd`) |
| `moe_shared_gate_input.bin` | f32 | 2048 | hidden state x for the gate dot (`n_embd`) |

Goldens:

| File | dtype | count | meaning |
|---|---|---|---|
| `moe_topk_indices.bin` | i32 | 8 | selected expert indices, in selection order |
| `moe_topk_weights.bin` | f32 | 8 | renormalized + scaled weights (sum ≈ 1.0) |
| `moe_shared_gate_scalar.bin` | f32 | 1 | `sigmoid(gate_w · x)` scalar gate |

Reference values from this seed: indices `[1, 93, 212, 144, 170, 76, 180, 203]`,
weights sum `1.000000`, shared gate `0.051038`.

**Tolerance:** top-k **indices must match EXACTLY** (integer). Weights and the
gate scalar: **~1e-5 .. 1e-4 relative** (softmax/sigmoid in fp32).

---

## Kernel 5 — SSM per-token delta-rule recurrence (HIGHEST VALUE, the centerpiece)

Golden produced by a **cross-checked replica** of `qwen35moe_compute_recur_heads`
(`src/models/qwen35moe_ssm.c:163-196`) — the static core recurrence, replicated
term-for-term with the source cited inline (the two length-S dots call the public
`pn_dot_f32`, exactly as the source does). Single-token (decode) path.

Config: `n_v_heads=32`, `state_size S=128`, `scale=1/sqrt(128)`. The state is
**transposed**: `s[h*S*S + j*S + i] == S_h[i][j]` (row j contiguous over i). For
each V-head h, each row j:
- (a) `row[i] *= exp(g[h])` — scalar decay of the whole row (ssm.c:180-182)
- (b) `pred = Σ_i row[i]·k[h][i]`; `delta = (v[h][j] − pred)·beta[h]` (ssm.c:184-186)
- (c) `row[i] += k[h][i]·delta` — rank-1 write, in place (ssm.c:188-190)
- (d) `y[h][j] = scale · Σ_i row[i]·q[h][i]` — **read-after-write** (ssm.c:192-193)

q/k are supplied **already L2-normed + GQA-repeated** to `n_v_heads` (the GPU
kernel's input contract; the 16→32 repeat happens on the host before the kernel).
`g` is **pre-exp** (the kernel applies `exp`); `beta` is already sigmoid'd.

Inputs:

| File | dtype | count | shape / meaning |
|---|---|---|---|
| `ssm_recur_state_in.bin` | f32 | 524288 | initial transposed state, `32×128×128` (`h,j,i`) |
| `ssm_recur_q.bin` | f32 | 4096 | q, `32×128` (L2-normed + GQA-repeated) |
| `ssm_recur_k.bin` | f32 | 4096 | k, `32×128` (L2-normed + GQA-repeated) |
| `ssm_recur_v.bin` | f32 | 4096 | v, `32×128` |
| `ssm_recur_g.bin` | f32 | 32 | per-V-head decay (pre-exp; kernel does `exp(g)`) |
| `ssm_recur_beta.bin` | f32 | 32 | per-V-head beta (already sigmoid'd) |

Goldens:

| File | dtype | count | meaning |
|---|---|---|---|
| `ssm_recur_y.bin` | f32 | 4096 | recurrence output y, `32×128` |
| `ssm_recur_state_out.bin` | f32 | 524288 | updated transposed state (decayed + rank-1 write) |

**Tolerance:** two 128-wide dot reductions per (h,j) in fp32 on Kepler —
**~1e-4 .. 1e-3 relative**. The harness diffs **both** y and the updated state.

## Kernel 6 — SSM causal conv1d (k=4) + SiLU

Golden produced by a **cross-checked replica** of `qwen35moe_conv1d_silu`
(`src/models/qwen35moe_ssm.c:110-135`), replicated term-for-term. Depthwise
causal conv over the full `conv_dim=8192` stream with `conv_kernel=4`:
`out[c] = SiLU(history·w[0..k-2] + qkv[c]·w[k-1])`, then the per-channel
`conv_state` is rolled (drop oldest, append raw `qkv[c]`). `conv_state` layout is
`[conv_dim][k-1]` (oldest..newest); weight is row-major `[conv_dim, k]`.

Inputs:

| File | dtype | count | shape / meaning |
|---|---|---|---|
| `ssm_conv_qkv.bin` | f32 | 8192 | raw projected stream this token (`conv_dim`) |
| `ssm_conv_weight.bin` | f32 | 32768 | depthwise conv weights, `8192×4` row-major |
| `ssm_conv_state_in.bin` | f32 | 24576 | conv history, `8192×3` (`[conv_dim][k-1]`) |

Goldens:

| File | dtype | count | meaning |
|---|---|---|---|
| `ssm_conv_out.bin` | f32 | 8192 | SiLU'd convolved stream (`conv_dim`) |
| `ssm_conv_state_out.bin` | f32 | 24576 | rolled conv_state, `8192×3` |

**Tolerance:** small fixed-tap dot + SiLU in fp32 — **~1e-4 .. 1e-3 relative**.
The harness diffs **both** the output and the rolled conv_state.

## Kernel 7 — SSM gated RMSNorm

Golden produced by the **REAL** CPU function `pn_vrms_norm_f32` per head plus a
SiLU replica, mirroring `qwen35moe_gated_norm`
(`src/models/qwen35moe_ssm.c:206-223`):
`gated[h*S+d] = RMSNorm(y_h)[d] · weight[d] · silu(z[h*S+d])`. The norm weight is
shared across heads; the trailing `ssm_out` projection is **out of scope** (the
GPU reuses `fc_matvec_q3k`). `n_v_heads=32`, `state_size S=128`.

Inputs:

| File | dtype | count | shape / meaning |
|---|---|---|---|
| `ssm_gated_y.bin` | f32 | 4096 | recurrence output y, `32×128` |
| `ssm_gated_weight.bin` | f32 | 128 | shared ssm_norm weight (`state_size`) |
| `ssm_gated_z.bin` | f32 | 4096 | attn_gate projection z (pre-silu), `32×128` |

Golden:

| File | dtype | count | meaning |
|---|---|---|---|
| `ssm_gated_out.bin` | f32 | 4096 | `RMSNorm(y)·w·silu(z)`, `32×128` |

**Tolerance:** per-head RMS reduction (128 elems) + silu in fp32 on Kepler —
**~1e-4 .. 1e-3 relative**.
