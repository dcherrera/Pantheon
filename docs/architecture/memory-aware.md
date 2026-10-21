# Memory-Aware Execution Layer

Sources: `src/core/mem/pn_memwatch.c`, `src/core/mem/pn_memwatch.h`, `src/core/mem/pn_mem_state.h`, `src/core/mem/pn_host_signals.h`, `src/core/mem/pn_host_signals.c`, `src/core/mem/pn_ladder.c`, `src/core/mem/pn_ladder.h`, `src/core/mem/pn_admission.c`, `src/core/mem/pn_admission.h`. Entry points wired in `src/core/engine.c`, `src/serve/serve_generation.c`, `src/serve/batch_sched.c`, `src/models/qwen35moe_forward_cuda.c`.

---

## Overview

The memory-aware layer tracks host RAM pressure (PSI + `/proc/meminfo`) and per-die VRAM in a continuously updated `pn_mem_state` snapshot, derives a four-level pressure enum, and exposes two behaviors to the rest of the engine:

1. **Pre-flight admission** (`pn_admit_request`, `pn_admit_model_load`): refuse or cap a request before any memory-growing work starts.
2. **Reactive ladder** (`pn_ladder_plan_for`): translate the current pressure level into concrete mitigation rungs (fewer slots, smaller context window, forced Q4 KV, KV eviction flag) that the serve and CUDA layers apply.

---

## Published state: `pn_mem_state` (pn_mem_state.h:46)

One immutable, value-type snapshot written by `pn_memwatch` and read by admission and the ladder. Seqlock-protected so readers never block the writer.

```c
typedef struct {
    uint64_t host_total_bytes;      // MemTotal / sysctl hw.memsize
    uint64_t host_avail_bytes;      // MemAvailable (allocatable without swap)
    double   host_psi_some_avg10;   // /proc/pressure/memory "some avg10" %
    double   host_psi_full_avg10;   // "full avg10" % — whole-box stall
    uint64_t cgroup_current_bytes;  // memory.current (0 if not in cgroup)
    uint64_t cgroup_high_bytes;     // memory.high    (0 == "max"/unset)
    uint64_t cgroup_max_bytes;      // memory.max     (0 == "max"/unset)
    int      n_dies;
    uint64_t per_die_free[16];      // cuMemGetInfo free per die
    uint64_t per_die_budget[16];    // usable budget (total - reserve) per die
    uint64_t per_die_used[16];      // our own alloc tally
    pn_mem_level level;             // derived: worst-of host + any die
    uint64_t seq;                   // monotonic snapshot counter
} pn_mem_state;
```

PSI and cgroup fields are Linux-only; on macOS they read 0 (treated as no pressure).

---

## pn_memwatch — the poll loop (pn_memwatch.c)

One background thread polls host signals every `PN_MEMWATCH_POLL_MS` (250 ms, pn_memwatch.h:46). The GPU side pushes die VRAM numbers via `pn_memwatch_set_die`.

**Seqlock protocol** (pn_memwatch.c:34):
- `g_seq` even = stable. Writer atomically increments to odd, copies state, increments back to even.
- Readers loop until `s1 == s2` (both even, copy stable). Never blocks the writer.

**Start / stop**:
```c
int  pn_memwatch_start(void);   // spawns poll thread; idempotent
void pn_memwatch_stop(void);    // joins thread
bool pn_memwatch_running(void); // true once started
void pn_memwatch_poll_now(void);// force one immediate poll (no wait for next tick)
```

`pn_memwatch_start` runs one synchronous `poll_host_once` before creating the thread so the first `pn_memwatch_read` call returns real numbers (pn_memwatch.c:292).

**Die feed** (pn_memwatch.c:244):
```c
void pn_memwatch_set_die(int die_id, uint64_t free_bytes,
                         uint64_t budget_bytes, uint64_t used_bytes);
```
Called from the GPU side after each cuMemGetInfo query. Holds `g_write_lock`, merges into the state, re-derives the level, and publishes via the seqlock.

---

## Host signals (pn_host_signals.c)

`pn_host_signals_sample` populates a `pn_host_sample` with:
- **Linux**: `/proc/meminfo` (MemTotal, MemAvailable) + `/proc/pressure/memory` (some/full avg10) + cgroup-v2 `memory.{current,high,max}`.
- **macOS / other**: sysctl `hw.memsize` total + `host_statistics64` for available; PSI/cgroup fields stay 0.

All reads are cheap (~1 ms). The poll thread calls `pn_host_signals_sample` inside `poll_host_once` (pn_memwatch.c:217).

---

## Pressure levels (pn_mem_state.h:30)

```c
typedef enum {
    PN_MEM_LEVEL_OK       = 0,  // ample headroom
    PN_MEM_LEVEL_WATCH    = 1,  // headroom shrinking
    PN_MEM_LEVEL_TIGHT    = 2,  // reserve nearly reached / PSI rising
    PN_MEM_LEVEL_CRITICAL = 3,  // at the reserve line / PSI stall — shed
} pn_mem_level;
```

### Host level derivation (pn_memwatch.c:105)

Reserve = `max(PN_HOST_RESERVE_MIN_BYTES=1 GiB, 15% × MemTotal)` (overridable via `PN_HOST_RESERVE_BYTES`).

| Condition | Level |
|---|---|
| `host_avail <= reserve` | CRITICAL |
| slack above reserve < reserve/4 | TIGHT |
| slack above reserve < reserve | WATCH |
| otherwise | OK |

**PSI escalation** (pn_memwatch.c:127):
- `full_avg10 >= PN_PSI_FULL_REJECT_DEFAULT` (15%, env `PN_PSI_FULL_REJECT`) → CRITICAL
- `full_avg10 > 0` → at least TIGHT (sub-threshold blip; see note below)
- `some_avg10 >= 10%` → at least TIGHT
- `some_avg10 > 0` → at least WATCH

**Important**: PSI avg10 decays over ~1 minute, so a transient spike (e.g. the ~90 s model load's dequant/staging burst) leaves `full_avg10` slightly above 0 for a full decay window after the box recovered. The PSI-full reject threshold (15%) is set high enough that this transient blip only escalates to TIGHT, not CRITICAL. Admission then still passes the byte-math check on a recovered box.

### Die level derivation (pn_memwatch.c:142)

Per die (die with `budget == 0` is skipped):

| free VRAM | Level |
|---|---|
| 0 | CRITICAL |
| < budget/16 | TIGHT |
| < budget/8 | WATCH |
| otherwise | OK |

Budget = total VRAM measured at boot − `PN_DIE_RESERVE_BYTES` (1 GiB default, env `PN_DIE_RESERVE_BYTES`).

`derive_level` returns `max(host_level, die_level)` (pn_memwatch.c:163).

### Why TIGHT is normal on the mini

The K80 test machine has 8 GiB of host RAM. The 15% reserve = ~1.2 GiB. The Qwen3.5-MoE model at steady state after load uses ~1.1 GiB RSS on host (measured 2026-07-09). Total host allocation sits well inside the TIGHT band (`slack < reserve`) at rest. This is expected and correct: TIGHT does not refuse requests on the GPU-resident path (PSI-gated, not byte-gated), and the ladder's rung-3 prefill-window clamp and rung-2 slot reduction only engage when the level is TIGHT and the box is actually under pressure. The admission gate at CRITICAL is the shed point for new concurrency (D3), not TIGHT.

---

## Admission control (pn_admission.c / pn_admission.h)

### Cost model

**Host activation peak** (`lut_activation_peak`, pn_admission.c:69): a 6-anchor LUT interpolated linearly. Anchors are conservative CPU-path measurements (the O(N²) attention scores dominate on CPU). On the GPU-resident path the O(N²) scores live on-die in bounded flash tiles — the host-RSS delta measured on a K80 host was ~0 at all tested lengths — so GPU-resident requests bypass this LUT entirely.

**KV per-token cost**: `PN_ADMIT_KV_BYTES_PER_TOKEN_Q4 = 5898` (~5.76 KiB) or `PN_ADMIT_KV_BYTES_PER_TOKEN_FP32 = 40960` (~40 KiB) (pn_admission.h:60).

**Total host cost** (pn_admission.c:98): `kv_per_tok * n_prompt + lut_activation_peak(n_prompt)`.

### pn_admit_request (pn_admission.c:153)

```c
void pn_admit_request(int n_prompt, bool kv_q4, bool gpu_resident,
                      int die_id, int max_window, const pn_mem_state *state,
                      pn_admit_result *out);
```

Decision order:
1. **PSI-full stall guard** (pn_admission.c:174): if `host_psi_full_avg10 >= reject_threshold` → REJECT ("sustained memory stall — retry").
2. **GPU-resident fast path** (pn_admission.c:193): if `gpu_resident`, host cost is declared 0. REJECT if `host_avail` is already inside the reserve; CAP if `n_prompt > max_window`; otherwise ADMIT at full context.
3. **CPU/host-fallback path** (pn_admission.c:208): check host budget (`host_avail − reserve`) and die free VRAM against the LUT cost. Binary-search for the largest fitting window if the full prompt doesn't fit; set CAP or REJECT.

### pn_admit_model_load (pn_admission.c:245)

Gates GPU weight uploads. Checks `weights_bytes + kv@max_ctx + activation_reserve <= die_free − die_reserve`. Used before uploading a model to a die.

### pn_admit_model_load_die (pn_admission.c:266)

Per-die variant for multi-die splits: checks `weights_slice + kv_slice + PN_ADMIT_DIE_SCRATCH_BYTES (1.5 GiB)` against one die's free VRAM. Does not add the host LUT (GPU-path scratch is bounded).

### pn_admit_load_spike (pn_admission.c:292)

Checks that `PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT` (2.4 GiB, dequant/staging burst measured on a K80 host during upload, env `PN_LOAD_SPIKE_BYTES`) fits within host headroom before a multi-minute weight upload starts.

### Engine-level wrapper: pn_admit (engine.c:481)

The serve layer calls `pn_admit(model, n_tokens, &fit_tokens_out)` which:
1. Fail-opens if `!pn_memwatch_running()`.
2. Reads the live state snapshot.
3. Derives `kv_q4`, `gpu_resident`, `die_id` (worst-free-VRAM die), `max_window`.
4. Consults the ladder (rung 3, see below) to possibly clamp `max_window` before calling `pn_admit_request`.
5. Maps the verdict to the public `pn_admit_status` enum.

Called from `sg_run_completion` (serve_generation.c:547) and `sg_run_completion_batched` (serve_generation.c:777) before any prefill starts.

### D3 — admission backpressure into the batch scheduler

When `sg_run_completion_batched` receives `PN_ADMIT_DENIED`, it calls `bs_notify_pressure(sched, (int)ms.level)` (serve_generation.c:786). This updates `sched->pressure_level` and broadcasts to `cv_idle` waiters. `bs_slot_acquire` (batch_sched.c:1493) then holds new requests in the queue if `pressure_level >= BS_PRESSURE_HOLD_LEVEL` (TIGHT/CRITICAL), preventing new work from being admitted until pressure clears. CRITICAL (level 3) triggers a fast-path 503 for queued waiters (batch_sched.c:1541). **D3 gates at CRITICAL, not TIGHT**: TIGHT alone does not block admission; only a DENIED verdict from `pn_admit` (which requires CRITICAL or PSI stall) pushes the pressure signal.

---

## Reactive degradation ladder (pn_ladder.c / pn_ladder.h)

The ladder is a **pure, stateless planner**: given a `pn_mem_level` and an operator config, it returns a `pn_ladder_plan` struct with absolute values (not deltas) the caller applies.

```c
typedef struct {
    pn_mem_level level;
    int  max_slots;       // rung 2: effective concurrency ceiling (>=1)
    bool force_kv_q4;     // rung 4a: force Q4 KV
    bool kv_evict;        // rung 4b: enable low-attention KV eviction (policy flag)
    int  prefill_window;  // rung 3: max prefill tokens (0 = no clamp)
    bool degraded;        // true if any rung reduced a configured value
    const char *note;
} pn_ladder_plan;
```

`pn_ladder_plan_for` (pn_ladder.c:51) and the convenience wrapper `pn_ladder_plan_from_state` are pure — no I/O, no allocation. At `PN_MEM_LEVEL_OK` the plan is byte-identical to the operator config.

Disable via `PN_LADDER_DISABLE` env var (`pn_ladder_disabled`, pn_ladder.c:19).

### Rung thresholds (pn_ladder.h:45)

| Rung | Name | Engages at | Action |
|---|---|---|---|
| 4a | force Q4 KV | WATCH | `force_kv_q4 = true` |
| 2 | shrink concurrency | TIGHT | `max_slots = cfg / 2`; CRITICAL → 1 |
| 3 | clamp prefill window | TIGHT | `prefill_window = cfg / 2`; CRITICAL → `cfg / 4` |
| 4b | KV eviction | TIGHT | `kv_evict = true` (policy flag; SnapKV selection not yet implemented) |

Rungs are reduce-only and monotonic: a higher pressure level never relaxes a rung a lower level engaged.

### Where the ladder is consumed

| Call site | What it reads | File:line |
|---|---|---|
| `pn_admit` (engine.c) | Rung 3 `prefill_window` to clamp `max_window` before admission | engine.c:504 |
| `pn_qwen35moe_gpu_ladder_max_slots` (qwen35moe_forward_cuda.c) | Rung 2 `max_slots` for the GPU decode-slot ceiling | qwen35moe_forward_cuda.c:4605 |
| `bs_new` (batch_sched.c) | Rung 2 ceiling to clamp `cap` at scheduler creation | batch_sched.c:629 |
| `pn_qwen35moe_decode_slot_bind_cuda` | Rung 2 ceiling to enforce the bind guard | qwen35moe_forward_cuda.c:4624 |
| `qwen35moe_kv_init` (qwen35moe_forward.c:142) | Rung 4a `force_kv_q4` at KV cache allocation time | qwen35moe_forward.c:145 |

**Rung 4a acts at KV-cache allocation time**, not reactively. The KV format (fp32 vs Q4) is fixed when the state is created; `force_kv_q4` only fires if the box is already at WATCH when `pn_qwen35moe_kv_init` runs (qwen35moe_forward.c:142). A live fp32 KV is never retroactively converted.

**Rung 4b (`kv_evict`)** is a policy flag only — it is set in the plan at TIGHT but no SnapKV-style row selection code exists yet. Nothing reads this flag today in a path that would evict rows from a live KV cache.

### The logged TIGHT clamp

The line `[ladder] TIGHT -> prefill window clamped 16384 -> 8192 (...)` is emitted by `pn_admit` at engine.c:507 whenever `ladder.prefill_window > 0 && ladder.prefill_window < max_window`. It means the incoming request's allowed context was halved (TIGHT divides by 2, pn_ladder.h:59) before admission checked it. This happens at every request on the mini while the server is at TIGHT.

---

## Concurrent-mutation hazard — segfault suspect

This section catalogs every place the ladder or memory layer could mutate or resize a buffer that an in-flight batched forward (`pn_qwen35moe_decode_slots_cuda` / the async begin/poll step) might be reading.

### 1. `gpu_attn_slot_ring_ensure` — KV ring reallocation (qwen35moe_forward_cuda.c:4946)

**CRITICAL.** Called inside `pn_qwen35moe_decode_slots_cuda` at the start of each attention-layer decode step. If `kv_slots_cap_seq != max_seq` or `kv_slots_cap_n < nslots` or `kv_q4 != last`, it frees and reallocates `k_slots_g`, `v_slots_g`, and the Q4 sink/win buffers (pn_qwen35moe_forward_cuda.c:4958–4989).

The trigger for `kv_slots_cap_n < nslots` is a new slot binding with a different `nslots` argument than the previous bind. The ladder (rung 2) can change the slot ceiling between `bs_new` time (where `cap` is clamped once at scheduler creation) and a subsequent `pn_qwen35moe_decode_slot_bind_cuda` call if the pressure level changed. However, the bind guard at qwen35moe_forward_cuda.c:4624 refuses any bind that would exceed the current ladder ceiling — so the reallocation path inside `gpu_attn_slot_ring_ensure` is only reached when `nslots` is *smaller* than expected (a bind with fewer slots, which can only shrink the allocation, not grow it). This is safe.

**The real hazard**: the async forward step (`pn_forward_step_begin/poll`) holds no lock during its multi-phase GPU operation (qwen35moe_forward_cuda.c:7053: "the per-model gen_lock guarantees one in-flight step per model"). If `gpu_attn_slot_ring_ensure` is somehow entered while an async step is in flight on the same model — e.g. a new connection calls `decode_slot_bind` (which calls `gpu_attn_slot_ring_ensure` via the bind path) before the previous step's poll completes — the reallocation frees buffers (`fc_buffer_free(d->k_slots_g)`) that a CUDA kernel may still be accessing on-device. Whether the serve layer's GPU-turn mechanism (`bs_gpu_acquire`) fully prevents this is the question (see item 3 below).

### 2. `conv_slots` / `ssm_slots` reallocation on slot-count change (qwen35moe_forward_cuda.c:4671)

Inside `pn_qwen35moe_decode_slot_bind_cuda`: if `d->slots_cap < n_slots`, the per-layer conv and SSM slot buffers are freed and reallocated (`fc_buffer_free(d->conv_slots)`, then `fc_buffer_alloc_ctx`). The graph is dropped (`d->slots_graph = NULL`) so the next step recaptures against the new allocation (comment at pn_qwen35moe_forward_cuda.c:4681: "L9 — the slot state buffers just moved"). If a decode step is in flight on the GPU and uses these buffers as kernel arguments via a captured CUDA graph, freeing them while the graph executes is a use-after-free at the CUDA level. The batch_sched comment at batch_sched.c:1144 says "the GPU turn is owned … this whole section is atomic w.r.t. every other forward on this model" — but `bs_slot_bind_common` acquires the GPU turn via `bs_gpu_acquire` (batch_sched.c:1207 calls `pn_qwen35moe_decode_slot_bind` while under the prefill turn, not the decode turn). Whether the `driver_thread`'s decode pass (`decode_slots` at batch_sched.c:480) and the prefill thread's bind path are mutually exclusive depends on the GPU-turn ticket system covering both.

### 3. `qw_defer` staging arena realloc (qwen35moe_forward_cuda.c:5057)

`qw_defer_reset` (called at the top of each decode step at qwen35moe_forward_cuda.c:6320) may `realloc(df->stage, need)` if the staging arena is too small. The comment at pn_qwen35moe_forward_cuda.c:5049 states explicitly: "an in-flight async DtoH writes into it — moving it would corrupt earlier layers' queued copies." The code handles this by pre-sizing up front so `qw_defer_stage` never reallocs mid-step. This is safe **within a single step** but not across steps: if a new step starts and `qw_defer_reset` reallocs `df->stage` while a prior step's async DtoH (from an earlier `begin`) has not been reaped by `poll`, those async writes still point at the old buffer address. The `as_*` phase machine (qwen35moe_forward_cuda.c:572) is supposed to prevent this, but verify that `poll` always fully drains async DtoH before `begin` is allowed to reenter.

### 4. Ladder-triggered max_slots change between scheduler creation and bind

`pn_qwen35moe_gpu_ladder_max_slots()` is called once at `bs_new` time (batch_sched.c:629) to clamp `cap`. It is called again inside `pn_qwen35moe_decode_slot_bind_cuda` (qwen35moe_forward_cuda.c:4629) for the bind guard. If the pressure level changed between these two calls — e.g. the box was OK when the scheduler was created (large cap) but is now TIGHT/CRITICAL (smaller ceiling) — the bind guard fires with `n_slots > slot_ceiling` and returns `-1` (engine.c style: the slot bind fails and the request fails cleanly). This is not a buffer corruption hazard, but it will produce a `[ladder] refusing N decode slots` error on requests that would have been fine at scheduler creation time.

### Summary table

| Location | Buffer mutated | Condition | Guarded by | Hazard for in-flight? |
|---|---|---|---|---|
| `gpu_attn_slot_ring_ensure` (line 4956) | `k_slots_g`, `v_slots_g`, sink/win bufs | `nslots` or `max_seq` or `q4` changed | GPU-turn ticket (prefill) | **Investigate**: does GPU-turn cover the decode step too? |
| `decode_slot_bind_cuda` (line 4671) | `conv_slots`, `ssm_slots` per SSM layer | `slots_cap < n_slots` | GPU-turn ticket (prefill) | **Investigate**: same question |
| `qw_defer_reset` (line 5057) | `df->stage` host staging arena | first step at a new nslots | pre-size contract within a step | Safe within step; verify async DtoH is always reaped before re-entry |
| `gpu_batch_host_ensure` (batch scratch) | `b->hidden` and friends | batch width grows | GPU-turn (prefill/decode) | `slots_graph_scratch` comment (line 271) notes graph invalidation on realloc |
