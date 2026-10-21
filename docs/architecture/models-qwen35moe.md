# Qwen3.5-MoE Model Implementation

Pantheon implementation of the Nex-N2-mini architecture: a 40-layer hybrid Gated-Delta-Net SSM / sparse GQA attention stack with 256-expert MoE on every layer.

Source files:
- `src/models/qwen35moe.c` — loader, config parser, GGUF weight mapping
- `src/models/qwen35moe_internal.h` — config/block structs, dual-state types, cross-file kernel declarations
- `src/models/qwen35moe.h` — public API, `PN_QWEN35MOE_MAX_SEQ_CAP`
- `src/models/qwen35moe_forward.c` — CPU forward, dual-state lifecycle, KV ring init
- `src/models/qwen35moe_ssm.c` — Gated-Delta-Net SSM block kernel
- `src/models/qwen35moe_forward_cuda.c` — CUDA forward: 2-die split, GPU weight residency, decode slots, async step machine (7 348 lines)

---

## Architecture

### Dims (from `pn_qwen35moe_config`, `qwen35moe_internal.h:36–60`)

| Field | Value | GGUF key |
|---|---|---|
| `n_layer` | 40 | `qwen35moe.block_count` |
| `full_attention_interval` | 4 | `qwen35moe.full_attention_interval` |
| `n_embd` | 2048 | `qwen35moe.embedding_length` |
| `n_head` | 16 | `qwen35moe.attention.head_count` |
| `n_head_kv` | 2 | `qwen35moe.attention.head_count_kv` |
| `head_dim` | 256 | `qwen35moe.attention.key_length` |
| `rope_dim` | 64 | `qwen35moe.rope.dimension_count` (partial NEOX) |
| `n_expert` | 256 | `qwen35moe.expert_count` |
| `n_expert_used` | 8 | `qwen35moe.expert_used_count` |
| `expert_ff` | 512 | `qwen35moe.expert_feed_forward_length` |
| `shared_ff` | 512 | `qwen35moe.expert_shared_feed_forward_length` |
| `ssm_state` | 128 | `qwen35moe.ssm.state_size` |
| `ssm_n_group` | 16 | `qwen35moe.ssm.group_count` (K-heads, GQA key/query) |
| `ssm_dt_rank` | 32 | `qwen35moe.ssm.time_step_rank` (V-heads) |
| `ssm_d_inner` | 4096 | `qwen35moe.ssm.inner_size` |
| `ssm_conv` | 4 | `qwen35moe.ssm.conv_kernel` |
| `conv_dim` | 8192 | derived: read from `blk.0.attn_qkv.weight` outermost dim (`qwen35moe.c:163–167`) |

### Layer layout

Layer type is set by `qwen35moe_is_attn_layer()` (`qwen35moe.c:216–220`):

```c
return ((i + 1) % interval) == 0;   /* interval default 4 */
```

With `full_attention_interval=4` and 40 layers, this yields:
- Layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 → **GQA full-attention** (10 layers, `n_attn=10`)
- All other 30 layers → **Gated-Delta-Net SSM** (`n_ssm=30`)

`block_type[i]` and `block_index[i]` in `pn_qwen35moe_model` (`qwen35moe.c:59–60`) map any global layer index to its type and per-type slot. These are the only two arrays that must be consulted to dispatch a layer.

### Per-layer block structure

Every layer (SSM or attention) shares the same outer shell:

```
attn_norm → (SSM block | GQA attn block) → +residual
         → post_attention_norm → shared-MoE FFN → +residual
```

(`qwen35moe_forward.c:316–369`)

### MoE routing

Every layer carries an identical `pn_qwen35moe_moe` block (`qwen35moe_internal.h:75–84`):

1. **Router**: `ffn_gate_inp [n_expert, n_embd]` → logits → softmax → top-8 selection → `norm_topk_prob` renormalization → scaled by `expert_weights_scale` (default 1.0; key absent in current GGUFs, `qwen35moe.c:141–145`).
2. **Routed experts**: 256 experts, zero-copy sliced from merged 3-D GGUF tensors `blk.N.ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight` (`qwen35moe.c:246–293`). Each expert is a SwiGLU: `silu(gate·x) * (up·x)` → `down`. `moe_out = Σ weight_e * SwiGLU_e(x)`.
3. **Shared expert** (always on): separate `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` + a scalar sigmoid gate `ffn_gate_inp_shexp [n_embd]`. `shexp_out = silu(gate·x) * (up·x) → down`, then scaled by `sigmoid(gate_inp_shexp · x)`.
4. **Combine**: `out = moe_out + shexp_out`. Residual add is the caller's responsibility.

(`qwen35moe_internal.h:278–308`)

### GQA attention block (`qwen35moe_internal.h:160–196`)

Op order for attention layers:
1. `attn_q [2*head_dim*n_head, n_embd]` → split: first `head_dim` per head is Q, second is gate z.
2. QK-RMSNorm on Q (per head via `attn_q_norm [head_dim]`) and on K (via `attn_k_norm [head_dim]`), pre-RoPE.
3. `attn_k [head_dim*n_head_kv, n_embd]` and `attn_v [head_dim*n_head_kv, n_embd]`.
4. Partial NEOX RoPE on the first `rope_dim=64` dims of every Q and K head. Multi-section RoPE is parsed (`rope_sections[4]`, `qwen35moe.c:111–130`) but collapses to standard NEOX for text.
5. GQA: 16 Q heads broadcast across 2 KV heads (each KV head serves 8 Q heads), scale `1/sqrt(256)`. K/V appended to per-layer ring before attention.
6. Output gate: `attn_out *= sigmoid(z)`.
7. `attn_output [n_embd, head_dim*n_head]`.

### SSM (Gated-Delta-Net) block (`qwen35moe_ssm.c:229–351`)

Op order for SSM layers:
1. Projections from `attn_norm`-ed input: `attn_qkv → qkv [conv_dim=8192]`; `attn_gate → z [ssm_d_inner=4096]`; `ssm_beta → beta [32]` → sigmoid; `ssm_alpha → alpha [32]`.
2. Decay gate per V-head (`qwen35moe_ssm.c:86–93`): `g[h] = ssm_a[h] * softplus(alpha[h] + ssm_dt[h])`. `ssm_a` is stored as `−exp(A_log)` (negative), so `exp(g) ∈ (0,1)`.
3. Causal depthwise conv1d (kernel size 4) over `conv_dim` channels + SiLU (`qwen35moe_ssm.c:110–135`). Rolls `conv_state` in place (oldest-first, channel-major layout).
4. Split convolved output into q `[128*16]`, k `[128*16]`, v `[128*32]`; L2-normalize q/k per K-head (`qwen35moe_ssm.c:70–76`); GQA-tile from 16 K-heads to 32 V-heads using interleaved tiling (`vh % n_k_heads == kh`, `qwen35moe_ssm.c:311`).
5. Delta-rule recurrence per V-head (`qwen35moe_ssm.c:163–195`): for each of 32 `[128×128]` state matrices (stored transposed, row j = column j of S): decay → pred → delta → rank-1 write → read-after-write y. Scale `1/sqrt(128)` applied once at read.
6. Gated RMSNorm: `RMSNorm(y, ssm_norm) * silu(z)` per V-head (`qwen35moe_ssm.c:206–223`), then `ssm_out [n_embd, ssm_d_inner]`.

The recurrence is parallelized across V-heads via `fw_for` when `USE_THREADS` is set and `n_v_heads >= 4` (`qwen35moe_ssm.c:332–336`).

---

## 2-Die Split and Placement

### Constants (`qwen35moe_forward_cuda.c:89–95`)

```c
#define GPU_DIE_SPLIT_LAYER  40        /* default = disabled (all on one die) */
#define GPU_DIE_A_DEFAULT     0
#define GPU_DIE_B_DEFAULT     1
```

`GPU_DIE_SPLIT_LAYER=40` means no split by default; the single-die path is the validated baseline. A split is engaged only when `QWEN35_GPU_SPLIT_LAYER < n_layer` or explicit placement provides two dies.

### Boundary resolution (`gpu_split_layer`, `qwen35moe_forward_cuda.c:696–711`)

```
pn_load_on(2 dies)  → split = n_layer/2 = 20
pn_load_on(1 die)   → split = n_layer (single-die forced)
no placement        → GPU_DIE_SPLIT_LAYER (40 = disabled)
QWEN35_GPU_SPLIT_LAYER env → overrides boundary (unless 1-die placement)
```

Clamped to `(0, n_layer]`.

### Die id resolution (`gpu_die_ids`, `qwen35moe_forward_cuda.c:717–736`)

Priority: `pn_load_on` explicit placement > `QWEN35_GPU_DIE_A` / `QWEN35_GPU_DIE_B` env vars > `GPU_DIE_A_DEFAULT=0` / `GPU_DIE_B_DEFAULT=1`.

Die ids are ordinals relative to `CUDA_VISIBLE_DEVICES`. On a K80 host, one K80 card exposes two GK110 dies; `CUDA_VISIBLE_DEVICES` maps them to ordinals 0 and 1.

### Weight placement

- Die A (`ctx_a`): layers `[0, split)` + token embedding (read host-side, row uploaded per token).
- Die B (`ctx_b`): layers `[split, n_layer)` + `output_norm` + LM head.

Each `gpu_ssm_layer.ctx` / `gpu_attn_layer.ctx` stores a back-pointer to its owning die context (`qwen35moe_forward_cuda.c:205`, `285`).

### 2-die multiplexing (`gpu_select_die`, `qwen35moe_forward_cuda.c:751–764`)

On legacy large-BAR K80 rigs (single MMIO slot), `barswap` time-multiplexes the two dies. The forward:
1. Acquires die A via `bs_acquire`.
2. Runs front layers on `ctx_a`.
3. Syncs, D2H copies the boundary `hidden [n_embd float32]`.
4. Acquires die B via `bs_acquire`.
5. H2D uploads `hidden` to `ctx_b`.
6. Runs back layers + LM head on `ctx_b`.

In small-BAR mode (modern driver, real dual-device), both contexts are live simultaneously and `g->bs == NULL`; no barswap or per-layer `gpu_select_die` is needed. The decode-slot bind path and the async path only run in small-BAR mode (`qwen35moe_forward_cuda.c:4655–4658`).

---

## KV Cache

### Allocation (`qwen35moe_kv_init`, `qwen35moe_forward.c:129–176`)

One `pn_qwen35moe_kv` per attention layer, 10 total. Layout `[n_head_kv][max_seq][head_dim]` (head-major).

**Q4 mode** (`QWEN35_KV_Q4`, default ON, `qwen35moe_forward.c:47–56`):
- Cache stored as ggml-Q4_0 rows: `kq [n_head_kv * max_seq * pn_kvq4_row_bytes(head_dim)]`, same for `vq`.
- fp16 sink shadow `k_sink/v_sink [n_head_kv * PN_KVQ4_SINKMAX * head_dim]` — attention sink tokens kept at fp16 to prevent argmax flips on near-ties.
- fp16 window shadow `k_win/v_win [n_head_kv * PN_KVQ4_WMAX * head_dim]` — the most recent W tokens similarly kept at fp16.
- PPL delta vs fp32 KV: 0.024% (validated, `qwen35moe_forward.c:45`).
- Disable with `QWEN35_KV_Q4=0`.

**fp32 mode**: `k/v [n_head_kv * max_seq * head_dim float32]`.

The reactive memory ladder rung 4a (`qwen35moe_forward.c:142–152`) auto-forces Q4 if the box is under memory pressure at allocation time even without the env var.

### KV window size

`PN_QWEN35MOE_MAX_SEQ` = 4096 (compile-time default, `qwen35moe_forward.c:70`).

`QWEN35_MAX_SEQ` env var overrides, clamped to:
- minimum: 16
- maximum: `PN_QWEN35MOE_MAX_SEQ_CAP` = 262 144 (`qwen35moe.h:29`)

The cap is the single source of truth shared between the CPU KV ring allocator (`qwen35moe_max_seq`, `qwen35moe_forward.c:76–91`) and the GPU admission gate (`gpu_admit_max_ctx`, `qwen35moe_forward_cuda.c:619–629`). Memory scales linearly: `n_head_kv * max_seq * head_dim * 8` bytes per attention layer on the host (same again on device in GPU mode).

### Ring behavior

KV is a fixed-size ring; `pos % max_seq` gives the slot. There is no dynamic resizing. If `pos >= max_seq` the oldest entry is overwritten. The ring is never grown after allocation.

### Deferred host mirror (B2)

After a GPU decode token, the GPU KV ring is the live source of truth. Writing back to the host mirror (`kv->k/v` or `kq/vq`) per token was too expensive. Instead, `kv->kv_mirror_stale = 1` is set (`qwen35moe_forward_cuda.c:3547`) and the D2H copy is deferred to `pn_qwen35moe_gpu_state_to_host()` (`qwen35moe_forward_cuda.c:6779`), which is called by the prefix cache on session eviction. After `pn_qwen35moe_state_reset()` the flag is cleared to 0 (`qwen35moe_forward.c:308`).

---

## Decode Slots / Multi-User Batching

### Slot capacity

`PN_QWEN35MOE_GPU_MAX_SLOTS` = 16 (`qwen35moe_forward_cuda.c:115`).

Memory cost: ~2.1 MB per SSM layer per slot for the per-slot `conv_slots / ssm_slots` images. At 16 slots across 30 SSM layers ≈ 1 GB on the die pair.

The reactive ladder rung 2 (`pn_qwen35moe_gpu_ladder_max_slots`, `qwen35moe_forward_cuda.c:4605–4613`) caps live slots: OK → 16, TIGHT → 8, CRITICAL → 1.

### Slot protocol

```
1. prefill each user's pn_state (single-token or batched)
2. immediately after each user's prefill:
   pn_qwen35moe_decode_slot_bind(model, state_u, u, K)
   — D2D snapshots the live GPU SSM state into slot u;
     host mirror is stale after GPU prefill so bind must precede
     any other user's forward
3. step all K users: pn_qwen35moe_decode_slots(model, tokens, states, K, logits)
   — ONE forward pass, weights stream once for all K users
```

(`qwen35moe.h:126–138`)

Slot-major per-user SSM images live in `gpu_ssm_layer.conv_slots / ssm_slots` (`qwen35moe_forward_cuda.c:247–250`). Bind snapshots via D2D from the shared `conv_state_g / ssm_state_g` right after prefill (`qwen35moe_forward_cuda.c:4616–4638`). A slot-count change invalidates all existing bindings.

`pn_qwen35moe_decode_slot_move(src, dst, n_slots)` (`qwen35moe.h:154`) compacts the active set via D2D copy without a host round-trip.

### Async step machine (C2, `QWEN35_ASYNC_SCHED`, default ON)

Three-state enum (`qwen35moe_forward_cuda.c:7062–7065`):
- `QW_ASYNC_IDLE` — no step in flight
- `QW_ASYNC_WAIT_A` — 2-die only: front segment (die A) in flight, CUDA event pending
- `QW_ASYNC_WAIT_FINAL` — back segment + LM head in flight (or single-die full step)

`pn_qwen35moe_forward_step_begin()` launches the step and returns without blocking (returns 1 = not async-eligible and runs synchronously in that case). `pn_qwen35moe_forward_step_poll()` reaps via `fc_event_query`, advances to the next phase, and on completion sets `*logits_out` and increments `state->pos`. This lets a 2-core host host keep up to 8 cards' generations pipelined.

Requires: small-BAR mode (real dual-device, `g->bs == NULL`); LM head must have an on-device kernel (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/F32, `qwen35moe_forward_cuda.c:7083–7098`). Disable with `QWEN35_ASYNC_SCHED=0`.

---

## The MTP / nextn Trap

The GGUF metadata for Qwen3.5-MoE may advertise MTP (multi-token prediction) draft-head weights under `blk.N.nextn.*` keys. The weights **do not ship** with public GGUF conversions; `nextn_predict_layers` must always be 0.

The loader explicitly documents this and never touches any `blk.N.nextn.*` tensor:

```c
/* Honor the MTP trap: nextn_predict_layers=0 — never touch blk.N.nextn.* . */
```

(`qwen35moe.c:475`)

If a future GGUF includes these weights, loading them would require a separate code path not yet implemented. Attempting to load them without that path would silently skip them (no tensor lookup is performed) — so correctness is maintained but draft-head acceleration would simply not engage.

---

## Gotchas and Fragile Spots

### KV ring is fixed at allocation — no resize path

`qwen35moe_kv_init` (`qwen35moe_forward.c:129`) allocates the ring once at `max_seq`. There is no resize-on-demand path. If the ring reallocates (e.g., `cache_cap` changes on the GPU side under a geometry change), the segment graph must be dropped and recaptured (`qwen35moe_forward_cuda.c:4941`). Under memory pressure the reactive ladder forces Q4 at init time but cannot shrink the ring after the fact.

**Segfault risk**: if `pos >= max_seq` is not checked before the GPU KV-append kernel writes `pos % max_seq`, and the GPU ring's `cache_cap` was allocated for a different `max_seq` than the host ring, the write goes out of bounds. The GPU ring capacity is tracked separately as `d->cache_cap` (`qwen35moe_forward_cuda.c:306`); a mismatch after a partial realloc failure would be silent until the OOB write.

### kv_mirror_stale + session switch

After a GPU decode step, `kv->kv_mirror_stale = 1`. If a session is evicted without calling `pn_qwen35moe_gpu_state_to_host()` first (e.g., the prefix cache path is bypassed or fails partway), the host mirror remains stale. A subsequent resume re-syncs from the stale host mirror instead of the GPU ring, silently corrupting the KV cache with zeros or old content.

The `pn_qwen35moe_gpu_forget_state()` call (`qwen35moe.h:212`) must be paired with `pn_state_reset()` on session reuse; omitting it leaves the GPU SSM buffers binding to the old session and the next prefill reads stale GPU residue.

### GQA repeat tiling — interleaved, not blocked

The SSM's GQA repeat tiles K-heads to V-heads using `vh % n_k_heads == kh` (interleaved), **not** the blocked `vh / kv_groups` mapping (`qwen35moe_ssm.c:311–312`). This matches `ggml_repeat`'s tiling in llama.cpp. Getting this wrong produces plausible but numerically wrong y vectors — the error is hard to detect without the oracle diff.

### conv_state layout: channel-major, oldest-first

`conv_state[c * (ssm_conv-1) + t]`, t=0 oldest. The roll-forward in `qwen35moe_conv1d_silu` (`qwen35moe_ssm.c:131–133`) shifts t=0..km1-2 forward and appends the new raw input at t=km1-1. The GPU path must replicate this exactly or the conv state diverges after the first token.

### ssm_state is stored transposed

`ssm_state[h * S * S + j * S + i] == S_h[i][j]` — row j (contiguous over i) is column j of S (`qwen35moe_internal.h:214–215`). This matches llama.cpp's `gated_delta_net` layout. Any GPU kernel that reinterprets the layout as standard row-major will compute wrong dot products.

### Batched CUDA prefill and CUDA-graph capture ordering

The per-layer segment graph is only captured after the layer has run DIRECTLY (uncaptured) at least once — `body_ran` gates the attempt (`qwen35moe_forward_cuda.c:229–232`). A batched prefill that precedes the first single-token forward never sets `body_ran` for those layers' fused descriptors, so the first single-token decode after a batch-only session must run uncaptured to build the fused descs. Capturing mid-allocation (before the fused descs exist) would call `cuMemAlloc` inside the capture, invalidating the graph. The `body_ran` gate prevents this.

### Embedding: lazy vs materialized

For untied models (a separate `output.weight` exists), `token_embd` is left `NULL` and each row is dequantized on demand from the mmap (`pn_qwen35moe_model_embed_row`, `qwen35moe.c:620–629`). For tied models, the full ~2 GB F32 embedding is materialized and also serves as the LM head (`qwen35moe.c:437–453`). The fallback for untied models where `n_embd % block_size != 0` also materializes the full heap.

### Numeric status: A8 not done

`qwen35moe_forward.c` and `qwen35moe_forward_cuda.c` both carry an open `@todo A8: validate logits vs the llama.cpp oracle`. End-to-end numeric correctness has not been verified. Several operations in the CUDA path have no dedicated GPU kernel and fall back to host-side computation with a D2H/H2D round trip (marked `TODO kernel:` inline in the CUDA file): joint Q+gate split, GQA head broadcast + L2-norm, decay gate / softplus, sigmoid gates, SwiGLU for non-Q3_K experts, attention softmax.
