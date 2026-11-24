# Accelerator / Die-Manager Layer and Foundry CUDA Base

Covers `src/accel/die_manager.c|h` and `foundry/cuda/fc_cuda.c|h`.

---

## 1. Die abstraction

A **die** is one GK210 GPU — the unit of VRAM ownership in Pantheon. A K80
card physically packages **two GK210 dies** connected by an on-card PLX
switch. They appear to CUDA as **two separate devices** with independent
contexts, streams, and VRAM pools. The die manager hard-codes this pairing:
`die_id / 2 = card_id` (`pn_dm_card_of`, `die_manager.h:82`) and
`{2i, 2i+1}` are card-mates (`die_mate`, `die_manager.c:195`). The die
manager supports up to 16 dies (`PN_DM_MAX_DIES`, `die_manager.h:41`; mirroring
`PN_MEM_MAX_DIES`).

### K80 hardware constraints

Reflected in the kernel target (`fc_cuda.h:17`):

- Compute capability sm_37 — no FP16 math, no tensor cores.
- `__shfl_down()` (Kepler) rather than `__shfl_down_sync()` (Pascal+).
- VRAM bandwidth ~240 GB/s; K80 decode is **launch-bound** (~80 µs/launch,
  `fc_cuda.h:562`), not compute-bound — the fused-layer and graph-capture
  mechanisms exist specifically to reduce launch count.

---

## 2. CUDA device → die mapping under `CUDA_VISIBLE_DEVICES`

`pn_dm_init_cuda` (`die_manager.c:72`) enumerates CUDA ordinals `[0, n_dies)`
by calling `fc_cuda_init_device(i)` for each. Because `CUDA_VISIBLE_DEVICES`
re-numbers the visible devices starting at 0, die ordinal `i` inside Pantheon
always equals CUDA device `i` within whatever the variable permits. Dies with
ordinals not visible to the process are not reachable: `cuDeviceGet(i)` returns
error 101 when `i >= CUDA_VISIBLE_DEVICES` count — the benign probe that
terminates the enumeration loop (`die_manager.c:89`):

```c
fc_cuda_ctx *ctx = fc_cuda_init_device(i);
if (!ctx || !fc_cuda_ctx_available(ctx))
    break;   /* ordinals are contiguous — stop */
```

`fc_cuda_init_device` calls `p_cuDeviceGet(&ctx->device, device_id)`
(`fc_cuda.c:910`); on error it logs `cuDeviceGet(%d) failed: %d` and returns
NULL, which is the break condition above.

---

## 3. VRAM budgeting and admission at the die level

### Reserve

The die budget is `total_vram − PN_DIE_RESERVE_BYTES` (default **1 GiB**,
`pn_memwatch.h:42`; overridable via `PN_DIE_RESERVE_BYTES`).
`pn_dm_init_cuda` computes this per die (`die_manager.c:95`):

```c
uint64_t budget = die_total > reserve ? die_total - reserve : 0;
```

### External occupancy

VRAM already consumed by another process at init time is detected and credited
to `vram_used` so Pantheon doesn't over-commit (`die_manager.c:101`):

```c
uint64_t external = (die_total - die_free) > reserve
                  ? (die_total - die_free) - reserve : 0;
g_dm.die[i].vram_used = external < budget ? external : budget;
```

### Two tallies

The die manager maintains two numbers per die (`die_manager.c:37`):

| Field | Meaning |
|---|---|
| `vram_budget` | Usable bytes (total − reserve). Set at init, never changes. |
| `vram_used` | Bytes committed by bindings + external occupancy. Mutated by bind/release. |

Free bytes at any moment = `vram_budget − vram_used` (`die_free_bytes`,
`die_manager.c:183`).

### memwatch feed

After computing each die's budget, `pn_dm_init_cuda` calls
`pn_memwatch_set_die(i, die_free, budget, fc_ctx_alloc_bytes(ctx))`
(`die_manager.c:105`) so `MemoryAware` admission sees the same per-die truth.

---

## 4. Bind / release lifecycle

### `pn_dm_place` (auto-placement, `die_manager.c:200`)

Does not reserve — only selects. Accepts `n_dies_wanted` of 1 or 2.

**2-die request:** Scans even ordinals, looking for a same-card pair `{a, a+1}`
both with `die_free_bytes ≥ per_die`. Prefers a fully-idle pair; among ties
picks the pair with the most aggregate free space (`die_manager.c:219`).

**1-die request:** Prefers a die whose card-mate is already occupied ("pack")
so free same-card pairs remain available for future 2-die models. Among
candidates picks the most-free die (`die_manager.c:244`).

### `pn_dm_bind` (`die_manager.c:270`)

Commits a placement under the global mutex:

1. Finds a free slot in `g_dm.binding[]` (cap `PN_DM_MAX_BINDINGS = 16`).
2. Re-checks `die_free_bytes(id) >= per_die` for every die in the set —
   guards against TOCTOU between `pn_dm_place` and `pn_dm_bind`.
3. Writes the binding (name, die list, `bytes_per_die`).
4. Increments `g_dm.die[id].vram_used` by `per_die` for every die in the set.

The whole operation is atomic under the lock; on any failure nothing is
partially written (`goto refuse`, `die_manager.c:313`).

**Same-name multi-card rule** (`die_manager.c:283`): a model name may be bound
multiple times to DISJOINT die-sets (one entry per copy, for load-balanced
multi-card serving). The code **does not reject duplicate names** at bind time.
Overlap is caught by the per-die room check (step 2): if a second binding would
use the same die, `die_free_bytes` will be too small and the bind refuses.
`pn_dm_release` is first-match on name (`die_manager.c:326`), so N bindings
with the same name require N `pn_dm_release` calls to free all of them.

### `pn_dm_release` (`die_manager.c:318`)

Walks the binding table, finds the first matching name, subtracts
`bytes_per_die` from each die's `vram_used` (clamped to 0), zeros the slot,
and returns. Only releases one binding per call.

---

## 5. Foundry CUDA primitives

### 5.1 Driver loading (`fc_cuda.c:517`)

`fc_driver_load()` (called lazily by `fc_cuda_init_device`):

1. `dlopen("libcuda.so.1")` — falls back to unversioned `libcuda.so`.
2. Resolves all driver API symbols via `fc_load_cuda_symbols`
   (`fc_cuda.c:414`). `_v2` variants are preferred for context/memory ops
   (required by cuBLAS); a fall-back to the unversioned name is attempted for
   each.
3. `cuInit(0)`.
4. `dlopen("libcublas.so.11")` (optional — custom PTX kernels work without it).

The resolved function-pointer table is **process-global** and shared by all
contexts (`fc_cuda.c:152`). The table includes optional symbols for peer copies,
graph capture, and events; missing ones leave their pointer NULL and the
corresponding feature self-disables gracefully.

### 5.2 Per-device context (`fc_cuda_init_device`, `fc_cuda.c:896`)

One `fc_cuda_ctx` per die. Sequence:

1. `p_cuDeviceGet(&ctx->device, device_id)` — binds the ordinal.
2. `p_cuCtxCreate(&ctx->context, 0, ctx->device)` — creates a CUDA context;
   leaves it current.
3. `p_cuStreamCreate(&ctx->stream, 0)` — one stream per context.
4. `cublasCreate` + `cublasSetStream` on the stream (non-fatal if cuBLAS absent).
5. `fc_ctx_load_kernels` — `cuModuleLoadData` of all embedded PTX blobs into
   this context's private module slots. Optional kernels (graph-capture variants,
   8-wide tile variants, slot-attention kernels) are looked up tolerantly and
   left NULL if absent.
6. `fc_ctx_leave()` — pops the new context off the thread's stack.
7. `fc_registry_add(ctx)` — registers in the live-context table for
   crash-time teardown.

The driver function table is shared; module handles and kernel function handles
are **per-context** (copied into each `fc_cuda_ctx`'s module fields,
`fc_cuda.c:224`).

### 5.3 Context isolation (push/pop, `fc_cuda.c:577`)

Every device op calls `fc_ctx_enter(ctx)` → `cuCtxPushCurrent` before issuing
work and `fc_ctx_leave()` → `cuCtxPopCurrent` on return. This keeps concurrent
dies from bleeding into each other even when ops from different dies interleave
on the same host thread.

### 5.4 VRAM accounting (`ctx->alloc_bytes`)

Every `cuMemAlloc` incremented path (`fc_buffer_from_ptr_ctx` at
`fc_cuda.c:1156`, `fc_buffer_alloc_ctx` at `fc_cuda.c:1184`) adds `bytes` to
`ctx->alloc_bytes`. `fc_buffer_free` subtracts it (clamped, `fc_cuda.c:1220`).
This counter is the **predictive** side of admission; `fc_ctx_alloc_bytes(ctx)`
exposes it. `pn_dm_init_cuda` passes it to `pn_memwatch_set_die` so both
tallies start consistent.

### 5.5 Memory operations

| Function | Operation |
|---|---|
| `fc_buffer_from_ptr_ctx` | `cuMemAlloc` + `cuMemcpyHtoD` (blocking) |
| `fc_buffer_alloc_ctx` | `cuMemAlloc` only (output buffer) |
| `fc_buffer_free` | `cuMemFree` + decrement `alloc_bytes` |
| `fc_buffer_to_cpu_n` | `cuMemcpyDtoH` (blocking) |
| `fc_buffer_to_cpu_n_async` | `cuMemcpyDtoHAsync` on ctx stream |
| `fc_buffer_copy_d2d_ctx` | `cuMemcpyDtoDAsync` on ctx stream |
| `fc_buffer_copy_peer` | `cuMemcpyPeer` (blocking, cross-die) |
| `fc_buffer_copy_peer_async` | `cuMemcpyPeerAsync` on stream_ctx stream |
| `fc_buffer_offset_view` | Non-owning wrapper (struct + pointer offset only; no cuMemAlloc/Free) |

Peer copies require `fc_cuda_enable_peer(a, b)` first, which calls
`cuDeviceCanAccessPeer` and `cuCtxEnablePeerAccess` in both directions
(`fc_cuda.c:1400`). Error 704 (already-enabled) is treated as success.

### 5.6 Synchronization

`fc_sync_ctx(ctx)` (`fc_cuda.c:4468`) calls `cuStreamSynchronize` on
`ctx->stream` (falling back to `cuCtxSynchronize` if the stream is NULL).
This blocks the host thread until all work queued on the stream is done.

**Events** (`fc_event_*`, `fc_cuda.c:4391`): created with
`CU_EVENT_DISABLE_TIMING` (no timing overhead). `fc_event_record_ctx` queues
`cuEventRecord` on the ctx stream; `fc_event_query` calls `cuEventQuery` and
returns 0 (done), 1 (CUDA_ERROR_NOT_READY = still running), or -1 (error).
The async serve scheduler uses events to poll all 8 card streams without
blocking any host thread — the alternative, `fc_sync_ctx`, would spin/yield a
core per card and starve other cards' launch turns on a 2-core host
(`fc_cuda.c:4383`).

### 5.7 Graph capture (`fc_graph_*`, `fc_cuda.c:4296`)

Optional (symbols checked at load; all six must be non-NULL or `fc_graph_begin_ctx`
returns -1). `fc_graph_begin_ctx` calls `cuStreamBeginCapture(stream, mode=0
= GLOBAL)`. `fc_graph_end_ctx` calls `cuStreamEndCapture` then
`cuGraphInstantiate`. `fc_graph_launch` replays via `cuGraphLaunch`. Used by the
decode-attention block for K80 launch-bound relief (one graph replay replaces
hundreds of `cuLaunchKernel` calls).

### 5.8 Back-compat single-die path

`fc_init()` creates a process-default context on device 0 (`_fc_default_ctx`).
All non-`_ctx` functions (`fc_buffer_from_ptr`, `fc_sgemm`, `fc_sync`, …)
delegate to it. Multi-die callers use the explicit `fc_cuda_ctx*` variants.

---

## 6. Correctness notes — multi-card same-name bindings

`pn_dm_bind` intentionally **does not prevent duplicate names**
(`die_manager.c:283`): N copies of a model on N different card-pairs all
register under the same name to support load-balanced serving. This is correct
as long as:

1. **Disjoint die-sets:** enforced by the per-die `die_free_bytes < per_die`
   check. If two bindings were attempted on overlapping dies, the second
   would see reduced free space and refuse — provided the first binding's
   `bytes_per_die` correctly covers the die's actual weight bytes.
2. **N releases required:** `pn_dm_release` is first-match; releasing a
   multi-copy model requires exactly N calls (one per binding slot). A single
   call frees only one copy's VRAM accounting. A caller that calls release once
   and assumes the die is fully reclaimed will have a stale `vram_used`.
3. **`pn_dm_get` reports only the first bound model name** for a die
   (`die_manager.c:167`); `n_models` counts all bindings touching that die.
   Status consumers that read only `model_name` will not see co-resident
   same-card models.

No gap in the overlap-rejection path: the per-die room check is atomic under
the global lock, so two concurrent binds on the same die cannot both succeed
if there is only room for one.

---

## 7. Second backend — Tenstorrent Blackhole p150a (`USE_TT=1`)

CUDA is not the only accelerator behind the die manager. `pn_dm_init_tt()`
(`die_manager.h:114`) enumerates a **Tenstorrent Blackhole p150a** instead of
K80 dies; the host driver is `src/accel/tt/pn_tt_{device,residency,oncard}.c`
over Foundry's `tt/` backend.

**Host:** any Linux machine with the card; the reference host is a small 2-core / 8 GB
box (no AVX) that also drives Tesla K80s. With tt-kmd the card appears as
`/dev/tenstorrent/<n>`.

### Build

```bash
make USE_TT=1          # Linux only; adds -DUSE_TT -DFT_TT_PHASE1B_NOC_ENABLED
                       # writes ./pantheon-tt, then points ./pantheon at it
```

This build writes **`pantheon-tt`**, not `pantheon` — the backends have separate
output names now, so a TT build on a host that also runs CUDA can no longer
overwrite a deployed `pantheon-cuda`. It does re-point the `pantheon` symlink,
and on a host that also serves CUDA that has to be put back:
`ln -sfn pantheon-cuda pantheon` (no rebuild needed). Confirm with
`./pantheon --backend` before deploying anything.

`FT_TT_PHASE1B_NOC_ENABLED` is deliberately **not** a second knob: without it
`ft_tt_mem_info()` cannot read a bank count and `pn_dm_init_tt()` refuses every
device, so a build could link the backend and then silently decline to see the
card. Device kernels are **not** built here — the kernel ELFs and Tensix programs
are pre-baked into `foundry/tt/ft_tt_kernels.h` / `ft_tt_programs.h`, the same
trick `cuda/` uses for PTX, so building needs no SFPI (SFPI is only needed to
rebuild those programs).

**Run-time dependency: a way to reach the card, nothing else.** The default
transport is tt-kmd (`/dev/tenstorrent/<n>`); `FT_TT_DRIVER=udrv` uses Foundry's
own userspace driver over PCI sysfs with no kernel module, which needs root or the
udev rule `foundry/tt/tools/60-foundry-blackhole.rules`. No tt-metal runtime, no
TT-NN, no Python, and no C++ anywhere — host or device. That boundary is the
point of the backend, not an incidental property. Foundry's TT layer was rewritten
from public documentation and measured behaviour; no tt-metal, TT-NN, tt-umd or
tt-llk library is linked or vendored (the few Apache-2.0-derived files that remain
are listed in Foundry's NOTICE). The vLLM/tt-metal stack cannot
run on the reference host at all: 2 cores / 8 GB and **no AVX**, so TT's prebuilt
wheels SIGILL.

### Runtime knobs

| Variable | Effect |
|---|---|
| `PN_TT_PROGRAM_DIR` | **override only.** Read Tensix programs from this directory instead of the ones embedded in the binary. Unset — the default — means embedded. An override that cannot be satisfied REFUSES; it never falls back to embedded |
| `PN_TT_DISPATCH` | dispatch-path selection |
| `PN_TT_ONCARD` | on-card op families (reduce / eltwise / top-k) |
| `PN_TT_CORE`, `PN_TT_CORE_{REDUCE,ELTWISE,SAMPLE}` | pin a residency to a Tensix index |

### What it can serve today

`pantheon serve` runs **MiniCPM5-1B** on the card over the OpenAI-compatible
endpoint, multi-turn, across idle gaps: **4.37 tok/s best, 4.14 sustained**
(2026-08-15). Eight op families are green on silicon.

Three limits to plan around:

1. **Everything measured is ONE core of 140. Multi-core is not built.** The
   forward runs 169 matvecs on a single matmul core. The figure quoted
   elsewhere as *8.94 decode tok/s* is **composed** from measured per-op costs
   (3.745 ms/layer × 24 + 22.0 ms lm_head) — **no end-to-end forward pass has
   been run to produce it.** The 4.37 tok/s above is the measured end-to-end
   number.
2. **MoE is not built.** Routed-MoE models (qwen35moe, e.g. Nex-N2) do not run
   on the card. The dense qwen35 hybrid (Gated-DeltaNet + attention, e.g.
   Ornith) does, byte-identical to the CPU forward (`engine.c`, `PN_ARCH_BONSAI`
   branch; Foundry `ft_tt_op_gdn`).
3. **Open correctness defect.** A chat-templated prompt — i.e. every request a
   serve receives — costs **+22.1% perplexity** and disagrees with the CPU
   oracle on **16.9%** of decode steps. Mechanism unidentified.

Weights are not GGUF on the card: `foundry/tt/convert/` turns a GGUF into a
tiled **FTTW** artifact (bf16 / Bfp8_b / Bfp4_b) and the loader is a byte copy.
**BFP4 is refuted on quality** (+40.2% perplexity) and is no faster than BFP8.

### Operational hazards — these cost hardware, not a retry

- **A wrong NoC coordinate hangs the chip and costs a physical power cycle.**
  Coordinates come from ARC telemetry; never probe or guess one.
- **`PHASE1B=1` is mandatory for on-card builds**; off-hardware suites are
  built without the NoC fence so they cannot reach the card even when wrong.
- **Never assert the Tensix BACKEND soft-reset bits** — it silently and
  permanently corrupts that core's compute until a host reboot.
- **Retire by asking, never by reset** (`FT_TT_QUEUE_OP_STOP` / `stop_request`).
  `assert_reset` on a blocked resident kernel is what wedges cores, and a
  resident triad must never be parked.
- **Never more than 4 destinations per multicast** — 5 wedges reliably.
- **On-card runs are detached and polled, never held over an SSH session.**
- **Card→host DMA does not exist on Blackhole.** Host→card is 287 MB/s on the
  reference host's x1 Gen2 link and readback is 2.3–4.7 MB/s, so on-card sampling is
  mandatory rather than an optimisation. **A card reset drops the link to
  Gen1** — retrain and verify afterwards, or every host-involved number is
  silently halved.
- **On-card residencies are bounded** and exit on an idle spin limit — and a
  serve idles by definition. `pn_tt_oncard_rearm()` is what recovers them.

Full detail: `foundry/tt/DEV_NOTES.md` (the authority on hazards),
`foundry/tt/README.md`, `foundry/docs/tenstorrent.md`.
