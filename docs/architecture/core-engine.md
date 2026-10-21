# Core Engine API

Source: `src/core/engine.c`, `src/core/engine.h`.

Covers the loading surface (`pn_load`/`pn_load_on`), the model and state structs, the async forward-step seam, scratch estimation, and the admission helpers at the engine level. The memory-aware layer that gates admission is documented separately in `memory-aware.md`.

---

## Public API surface

### Loading

```c
// Thin wrappers — all paths converge on pn_load_on.
pn_model* pn_load(const char *path);                              // engine.c:412
pn_model* pn_load_ex(const char *path, const pn_load_opts *opts); // engine.c:549
pn_model* pn_load_on(const char *path, const pn_load_opts *opts,  // engine.c:553
                     const int *dies, int n_dies);
```

`pn_load` and `pn_load_ex` are thin wrappers that call `pn_load_on` with `dies=NULL, n_dies=0`.

`pn_load_on` is the placement-aware entry point added for multi-model serving (Phase 7). When `dies` is non-NULL the die list is recorded on the model before any GPU upload (`model_record_dies` at engine.c:538), so the subsequent `pn_qwen35moe_gpu_upload` call reads the correct target dies via `pn_qwen35moe_model_set_dies`.

**Load opts:**
```c
typedef struct {
    const char *vocab_path;   // RWKV World tokenizer path
    const char *config_path;  // HF config.json for Qwen/LLaMA safetensors
    bool quantize;            // INT8-quantize at load time (RWKV-X only)
} pn_load_opts;              // engine.h:116
```

### pn_model struct (engine.h:74)

```c
struct pn_model {
    pn_config    config;           // parsed dimensions (stack)
    void        *weights;          // arch-specific weight struct
    pn_forward_fn forward;         // forward pass function pointer
    ft_tokenizer *tokenizer;       // BPE/SP tokenizer (NULL for RWKV-X)
    char        *path;
    int          chat_style;
    void        *vision;           // pn_vision_model* for VLMs
    struct fw_pool *pool;          // thread pool (USE_THREADS only)
    int  gpu_dies[PN_MODEL_MAX_DIES];  // explicit die placement (pn_load_on)
    int  n_gpu_dies;                   // 0 = none, env/defaults apply
};
```

`forward` is the architecture dispatch: a function pointer of type `tm_tensor* (*)(pn_model*, uint32_t token, pn_state*)`. The CPU and CUDA forwards are distinct pointers; `pn_admit_gpu_resident` identifies the CUDA path by comparing `model->forward == pn_qwen35moe_forward_cuda` (engine.c:441).

### pn_state struct (engine.h:89)

```c
struct pn_state {
    void         *kv_cache;       // arch-specific: pn_llama_kv_cache / pn_qwen35moe_state / rx_state
    int           pos;            // current sequence position
    pn_config    *config;         // pointer to model config
    struct pn_arena *arena;       // bump-allocator scratch (NULL for RWKV-X and Qwen35MOE)
    const float  *embd_override;  // if set, forward uses this embedding (image token)
};
```

`pn_state_new` allocates architecture-specific state:
- **Llama/Phi/Mistral/Mixtral** (engine.c:870): allocates `pn_llama_kv_cache` and an arena sized by `compute_arena_size`.
- **Qwen3.5-MoE** (engine.c:860): allocates `pn_qwen35moe_state` (dual state: SSM conv/ssm + attention KV). No arena.
- **RWKV-X** (engine.c:849): allocates `rx_state` (recurrent + KV + workspace). No arena.

---

## Loading paths

`pn_load_on` branches on file format:

### Safetensors

1. If `opts->config_path` is set: HF/ternary safetensors loader → `pn_llama_load_safetensors` + `pn_llama_load_hf_tokenizer`. Sets `model->forward = pn_llama_forward` (engine.c:579).
2. Otherwise: RWKV-X path → `pn_rwkvx_load` + optional vocab. Sets `model->forward = pn_rwkvx_forward` (engine.c:627).

### GGUF

Architecture detected from `general.architecture` metadata (engine.c:226). For Qwen3.5-MoE specifically (engine.c:694):

1. `pn_qwen35moe_load(gguf, &config)` — zero-copy mmap weights.
2. Explicit die placement applied to the arch model: `pn_qwen35moe_model_set_dies(qw, model->gpu_dies, model->n_gpu_dies)` (engine.c:704).
3. `fc_init()` — idempotent CUDA init; no-op without a GPU.
4. `pn_qwen35moe_gpu_upload(qw)` — uploads weights to VRAM; on success sets `model->forward = pn_qwen35moe_forward_cuda` (engine.c:716). Falls back silently to `pn_qwen35moe_forward` (CPU) if the upload fails or no GPU is present.

---

## Scratch (arena) estimation

`compute_arena_size` (engine.c:357) sizes the per-state bump allocator for transformer models:

- **Per layer**: `5*n_embd + 3*n_ff + 2*n_kv_head*head_dim` floats. MoE adds `n_experts + 2*n_experts_used + 4*n_embd + 2*n_ff` floats.
- **One-time**: `n_embd + n_head*max_seq + n_embd + n_vocab` floats.
- **max_seq cap**: `min(n_ctx, sliding_window if > 0, 4096)` — the scores buffer is bounded at 4096 (engine.c:374).
- **Alignment overhead**: 16 bytes × (n_layer×11+5) allocations, plus an additional 5 per layer for MoE.
- **Safety margin**: +25% (engine.c:405).

The arena resets between forward calls; nothing in it persists across tokens.

---

## Async forward-step seam (C2)

For the batched GPU serving path a non-blocking single-token step is exposed (engine.h:228):

```c
int pn_forward_step_begin(pn_model *model, uint32_t token, pn_state *state);
int pn_forward_step_poll(pn_model *model, pn_state *state, tm_tensor **logits_out);
```

**begin** (engine.c:1173): dispatches to `pn_qwen35moe_forward_step_begin` when `model->forward == pn_qwen35moe_forward_cuda`; returns 1 ("unsupported") for everything else — the caller uses `model->forward` blocking path instead.

**poll** (engine.c:1182): dispatches to `pn_qwen35moe_forward_step_poll`; returns 1 while running, 0 when done (writes `*logits_out`), -1 on error. Returns -1 for unsupported models.

The design: each connection thread owns one model slot and parks in a poll loop between `begin` and `done`. Because `begin` returns without blocking, the 2-core K80 host can keep multiple cards' steps in flight simultaneously. The per-model `gen_lock` in the serve layer guarantees one in-flight step per model at a time.

---

## Admission helpers (engine-level)

These live in `engine.c:426–534` and are the engine-level face of the memory-aware admission system (full detail in `memory-aware.md`).

### pn_admit (engine.c:481)

```c
pn_admit_status pn_admit(pn_model *model, int n_tokens, int *fit_tokens_out);
```

Called from the serve layer before any prefill work. Returns `PN_ADMIT_FIT`, `PN_ADMIT_CAPPED`, or `PN_ADMIT_DENIED`.

Steps:
1. Fail-open if `!pn_memwatch_running()` — returns `FIT` with no check (engine.c:487).
2. `pn_memwatch_read(&st)` — reads the live seqlock-protected snapshot.
3. Determines `kv_q4` (env `QWEN35_KV_Q4`), `gpu_resident` (`forward == pn_qwen35moe_forward_cuda`), `die_id` (tightest-free-VRAM die), and `max_window` (clamped by `QWEN35_MAX_SEQ` and `model->config.n_ctx`).
4. **Ladder rung 3** (engine.c:504): calls `pn_ladder_plan_from_state` and, if `ladder.prefill_window < max_window`, clamps `max_window` and logs `[ladder] TIGHT -> prefill window clamped 16384 -> 8192`.
5. Calls `pn_admit_request` with the adjusted `max_window` and the full `pn_mem_state`.
6. Maps `PN_ADMIT_CAP` → `PN_ADMIT_CAPPED` (sets `*fit_tokens_out`), `PN_ADMIT_REJECT` → `PN_ADMIT_DENIED`.

### Internal helpers (engine.c:427–479)

| Helper | Purpose |
|---|---|
| `pn_admit_kv_q4()` | Returns `true` unless `QWEN35_KV_Q4=0` |
| `pn_admit_gpu_resident(model)` | `true` iff `model->forward == pn_qwen35moe_forward_cuda` |
| `pn_admit_max_window(model)` | Reads `QWEN35_MAX_SEQ`, clamps to `PN_QWEN35MOE_MAX_SEQ_CAP` and `config.n_ctx` |
| `pn_admit_worst_die(st)` | Finds the die with the least free VRAM (the conservative scheduling target) |

---

## Sampler seam

```c
uint32_t pn_sample_next_token(float *logits, int n_vocab,
                              pn_sampling_params *params,
                              uint32_t *recent_tokens, int n_recent); // engine.c:1160
```

Public wrapper exposing the internal `sample_token` to the serve layer, so it does not re-implement sampling. Mutates `logits[]` in place. Not thread-safe (shared thread-local scratch buffer per vocab size).

---

## Architecture registry

`pn_get_forward_fn(pn_arch arch)` (engine.c:214) maps arch enum → forward function pointer. Llama, Qwen2, Mistral, Mixtral all share `pn_llama_forward`; only Phi and Qwen3.5-MoE have distinct implementations.

---

## State lifecycle

```
pn_state_new(model)   → allocates kv_cache + arena (arch-specific)
pn_state_reset(state) → zeros kv tensors, resets pos to 0 (engine.c:923)
pn_state_free(state)  → frees kv_cache, arena, struct
```

`pn_state_reset` for Llama/Phi explicitly zeros every K and V tensor in the cache (engine.c:936–943). For Qwen3.5-MoE and RWKV-X it delegates to arch-specific reset functions.
