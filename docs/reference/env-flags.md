# Pantheon — Environment Flag Reference

Every `getenv()` call in `src/` and `foundry/` is listed here. Flags are
grouped by prefix. All flags are **process-constant** unless noted: they are
read once into a static cache on first call and never re-read.

---

## QWEN35\_\* — Qwen3.5-MoE model flags

### Core KV / context window

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_KV_Q4` | `0` or anything | ON | `env[0]=='0'` disables | KV-cache format: Q4\_0 (~4–7× smaller) vs fp32. Default changed to ON (B1). **Only `=0` disables.** | `qwen35moe_forward.c:51`, `engine.c:430` |
| `QWEN35_MAX_SEQ` | integer | `4096` | `atoi`, clamped `[16, 262144]` | KV-window size per attention layer. Memory scales linearly. Ceiling `PN_QWEN35MOE_MAX_SEQ_CAP=262144`. | `qwen35moe_forward.c:81`, `qwen35moe_forward_cuda.c:621`, `engine.c:454` |
| `QWEN35_KV_Q4_WIN` | integer 0–256 | `128` | `atoi`, clamped `[0, 256]` | fp16 recent-window exemption size inside the Q4 KV cache. `0` disables the window (pure Q4). | `qwen35moe_kv_q4.h:54` |
| `QWEN35_KV_Q4_SINK` | integer 0–8 | `4` | `atoi`, clamped `[0, 8]` | fp16 attention-sink prefix kept exempt from Q4 quantization. | `qwen35moe_kv_q4.h:67` |

### GPU placement / split

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_GPU_SPLIT_LAYER` | integer | `40` (`GPU_DIE_SPLIT_LAYER`) | `atoi`, must be `>0`; ignored when explicit 1-die placement | Layer boundary between die A and die B. Layers `[0, split)` → die A; `[split, n_layer)` → die B. Ignored when explicit single-die placement is authoritative. | `qwen35moe_forward_cuda.c:702` |
| `QWEN35_GPU_DIE_A` | integer (CUDA ordinal) | `0` | `atoi` | Die A CUDA device ordinal. Superseded by explicit `--dies` placement (warning printed). | `qwen35moe_forward_cuda.c:732` |
| `QWEN35_GPU_DIE_B` | integer (CUDA ordinal) | `1` | `atoi` | Die B CUDA device ordinal. Same supersession rule as `QWEN35_GPU_DIE_A`. | `qwen35moe_forward_cuda.c:733` |
| `QWEN35_GPU_MAX_LAYERS` | integer | unset (= all layers) | `atoi`, must be `>0` | Single-die only: upload/validate only the first N global layers. Ignored in 2-die mode. Also disables async scheduling (`QWEN35_ASYNC_SCHED`). | `qwen35moe_forward_cuda.c:1407`, `7164` |
| `QWEN35_GPU_BARSWAP` | any | unset | presence check (`== NULL` → small-BAR) | Legacy barswap (single MMIO slot) vs small-BAR mode (all dies live). Default is small-BAR (no barswap). Set to any non-empty value to restore the old barswap 2-die path. | `qwen35moe_forward_cuda.c:1416` |
| `QWEN35_DROP_WEIGHTS` | `0` or unset | unset (= drop) | `env[0]=='0'` opts out | After GPU upload, call `madvise(MADV_DONTNEED)` on the host weight mmap to free its page cache (~15 GB for a 35B MoE). **On by default** (qwen35 and llama CUDA paths); `=0` keeps the pages for host-fallback debugging. Host-fallback matvec refaults clean pages from disk. | `qwen35moe_forward_cuda.c:1935`, `llama_forward_cuda.c:211` |

### Batched decode (C3 scheduler)

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_BATCH_SLOTS` | `1` or anything | OFF | **`env[0]=='1'`** — only the character `'1'` enables | **Enable** multi-user batched decode. Any other value (including `=8`, `=4`, `=on`) silently keeps batching OFF. | `batch_sched.c:51` |
| `QWEN35_BS_STATS` | `1` or anything | OFF | `env[0]=='1'` | Print per-step driver timing to stderr (M6 instrumentation). | `batch_sched.c:110` |
| `QWEN35_BS_DRIVER_SAMPLE` | `0` or anything | ON | `env[0]=='0'` disables | In-driver sampling (default ON). Set `=0` to restore legacy owner-side lockstep (each request thread samples its own token). | `batch_sched.c:651` |
| `BS_QUEUE_DEPTH` | integer `>0` | `32` | `atoi` | Per-model overflow queue depth (D1). Requests queue here when all slots are busy before getting 429. | `batch_sched.c:725` |
| `QWEN35_BS_PREFILL_BATCH` | `0` or anything | ON | `env[0]=='0'` disables | Enable batched prefill (one GPU pass per chunk, ~2× vs token-by-token). `=0` restores the sequential token loop. | `serve_generation.c:471` |
| `QWEN35_BS_PREFILL_CHUNK` | integer `>=0` | `128` | `atoi`; `<0` → 0 | Prefill chunk size in tokens. `0` = entire prompt in one pass (unbounded scratch). | `serve_generation.c:482` |

### Logits / sampling

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_GPU_FULL_LOGITS` | any | unset | presence check | Force full n\_vocab (248K) D2H logits copy per token. **Serve refuses to start if this is set** — it cripples batched decode throughput (K×992 KB/step over host link). Debug/eval only. | `serve_command.c:279`, `qwen35moe_forward_cuda.c:2747` |
| `QWEN35_GPU_TOPK` | integer 0–256 | `40` | `atoi`, clamped `[0, FC_TOPK_MAX=256]` | On-die top-K reduction before D2H. `0` or forced by `QWEN35_GPU_FULL_LOGITS` → full copy. | `qwen35moe_forward_cuda.c:2751` |

### CUDA-graph / fusion

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_GRAPH` | `0` or anything | ON | `env[0]=='0'` disables | Master gate for CUDA-graph replay of SSM layers. Disabling also disables `QWEN35_GRAPH_FULL` and `QWEN35_GRAPH_SLOTS`. | `qwen35moe_forward_cuda.c:1899` |
| `QWEN35_GRAPH_FULL` | `0` or anything | ON | `env[0]=='0'` disables | Full-step segment graphs (C1): capture every layer of a die segment (SSM+attention interleaved) into one graph, replay with one `cuGraphLaunch`. Requires `QWEN35_GRAPH`. | `qwen35moe_forward_cuda.c:1916` |
| `QWEN35_GRAPH_SLOTS` | `0` or anything | ON | `env[0]=='0'` disables | Per-SSM-layer graphs on the batched-decode (slots) path (L9). Sub-gate of `QWEN35_GRAPH`. Attention layers stay direct. | `qwen35moe_forward_cuda.c:1931` |
| `QWEN35_GRAPH_CAPTURE_FAIL_INJECT` | int N | `0` (off) | `atoi(env)` | **Test only.** Forces the first N capture attempts of EVERY graph slot to be treated as failures, so the bounded-retry path can be exercised on real hardware. `N < 5` exercises recovery, `N >= 5` exercises the permanent give-up. Never set on a serving process. | `gpu_graph_capture_tracker.c:pn_graph_capture_inject_failures` |
| `QWEN35_FUSE` | hex bitmask | `0x3F` (all on) | `strtol(env, NULL, 0)` (hex/decimal) | Fusion-site bitmask: bit0=ssm proj4, bit1=ssm out+res, bit2=attn qkv, bit3=attn norm+rope, bit4=moe pre, bit5=moe post. Set specific bits `=0` to bisect regressions. | `qwen35moe_forward_cuda.c:1887` |

### Flash attention

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_FLASH` | `0` or anything | ON | `env[0]=='0'` disables | Flash attention for batched prefill. NOT bit-exact vs host softmax (~1e-6 relative drift). `=0` restores host-softmax path. | `qwen35moe_forward_cuda.c:3257` |
| `QWEN35_FLASH_DECODE` | `0` or anything | ON | `env[0]=='0'` disables | Flash attention for single-token decode. `=0` restores host-softmax decode path. | `qwen35moe_forward_cuda.c:3406` |
| `QWEN35_FLASH_DECODE_KPS` | integer `>=16` | `PN_QWEN35MOE_DEC_KEYS_PER_SPLIT` | `atoi`, floor 16 | Keys per decode split (tune for multi-split combine path in tests). | `qwen35moe_forward_cuda.c:3419` |
| `QWEN35_FLASH_SLOTS` | `0` or anything | ON | `env[0]=='0'` disables | Flash attention on the batched-decode (slots) path (L7). `=0` → host-softmax slots path. | `qwen35moe_forward_cuda.c:4917` |
| `QWEN35_ATTN_WINDOW` | integer `>0` | `0` (full) | `strtol` | StreamingLLM sliding-window attention recent-window size. `0`/unset = full attention. Applies to all 10 global attention layers. | `qwen35moe_forward_cuda.c:3272` |
| `QWEN35_ATTN_SINK` | integer `>=0` | `4` | `strtol` | StreamingLLM attention-sink prefix size (used with `QWEN35_ATTN_WINDOW`). | `qwen35moe_forward_cuda.c:3283` |

### 2-die pipeline / threading

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_ASYNC_SCHED` | `0` or anything | ON | `env[0]=='0'` disables | Non-blocking async dispatch for single-token decode (C2). `=0` restores fully synchronous path. Automatically disabled when `QWEN35_GPU_MAX_LAYERS` is set or LM head needs host fallback. | `qwen35moe_forward_cuda.c:7074` |
| `QWEN35_ASYNC_KVMIRROR` | any non-`0` | OFF | `env[0] && env[0]!='0'` | Defer host-KV-mirror stores in batched decode (L8). Experimental. | `qwen35moe_forward_cuda.c:5017` |
| `QWEN35_TWO_THREAD` | any non-`0` | OFF | `env[0] && env[0]!='0'` | Two-die two-thread decode (small-BAR, 2 dies, nslots≥1). | `qwen35moe_forward_cuda.c:5965` |
| `QWEN35_DIE_PIPELINE` | any non-`0` | OFF | `env[0] && env[0]!='0'` | Software-pipeline two batch groups across the die pair so both dies compute simultaneously (L5). Requires 2 dies + grouped MoE + nslots≥2. | `qwen35moe_forward_cuda.c:5979` |
| `QWEN35_VOCAB_SPLIT` | any non-`0` | OFF | `env[0] && env[0]!='0'` | Split LM head across 2 dies (T3). Only useful in 2-die non-barswap mode with peer access. | `qwen35moe_forward_cuda.c:5991` |

### SSM

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_SSM_2PASS` | any non-`0` | OFF | `env[0] && env[0]!='0'` | Route SSM recurrence to fused-reduction (2-pass) kernel: one LDS sweep, 3×S shared, slightly lossy by reassociation. Default OFF (bit-exact baseline). | `foundry/cuda/fc_cuda.c:3375` |

### Debug / profiling

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `QWEN35_DEBUG` | any | unset | presence check | Enable per-layer activation dumps to stderr. Also forces full-logits D2H (`gpu_topk_k()` returns 0). | `qwen35moe_forward.c:106`, `qwen35moe_forward_cuda.c:991` |
| `QWEN35_PROF` | any | unset | presence check | Per-token wall-clock profiler: accumulates matvec time by quant type and forward phase, printed per token. Adds `fc_sync` around timed ops. | `qwen35moe_forward_cuda.c:794` |
| `QWEN35_SPROF` | any | unset | presence check | Per-step batch-decode profiler: phase breakdown printed per step, cumulative summary every 10 steps. | `qwen35moe_forward_cuda.c:930` |

---

## PN\_\* — Pantheon server / memory flags

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `PN_THREADS` | integer | `0` (auto) | `atoi` | CPU thread-pool size for `USE_THREADS` builds. `0` = auto-detect. | `engine.c:772` |
| `PN_ASYNC_POLL_US` | integer 50–20000 | `500` | `atol`, clamped `[50, 20000]` | Async step poll interval in microseconds (exponential backoff up to this cap). | `serve_generation.c:422` |
| `PN_HOST_RESERVE_BYTES` | integer (bytes) | `max(1 GiB, 15% × MemTotal)` | `strtoull` | Override host RAM reserve held off-limits for admission. Values ≤0 or unparseable fall back to the formula. | `pn_memwatch.c:62` |
| `PN_DIE_RESERVE_BYTES` | integer (bytes) | `1073741824` (1 GiB) | `strtoull` | Override per-die VRAM reserve subtracted from each die's reported total before scheduling. | `pn_memwatch.c:72` |
| `PN_PSI_FULL_REJECT` | float (percent) | `15.0` | `strtod`; values ≤0 ignored | PSI-full avg10 threshold (% wall time) at/above which admission rejects new requests as CRITICAL. | `pn_memwatch.c:78` |
| `PN_LADDER_DISABLE` | any non-`0` | unset (enabled) | `env[0] && env[0]!='0'` | Disable the reactive degradation ladder entirely. The plan mirrors the config at all pressure levels. | `pn_ladder.c:21` |
| `PN_LOAD_SPIKE_BYTES` | integer (bytes) | `2516582400` (~2.4 GB) | `strtoull`; ≤0 falls back to default | Host anon RSS budget for GPU-upload transient (measured on a Tesla K80 host: ~2.4 GB for ~55 s). Admission checks this before starting a load. | `pn_admission.c:280` |
| `PN_VLM_DEBUG` | any | unset | presence check | Print top-3 logits + sampled token for the first 6 steps. Diagnoses empty VLM output. | `engine.c:1686` |
| `PN_GPU_PREFILL` | any | unset | presence check | (llama/Metal) Upload Q4\_K decoder weights + run prefill GEMMs on the GPU. Off by default (measured net loss for bulk OCR vs CPU+GPU overlap). | `llama.c:2068` |
| `PN_BATCH_GPU` | any | unset | presence check | (llama/Metal, batch≤32) Upload Q4\_K decoder weights for GPU-based batched GEMM decode. Experimental. | `llama.c:2398` |

---

## FC\_\* — foundry/cuda MoE kernel tuning

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `FC_MOE_DOWN_WPP` | `1`, `2`, `4`, or `8` | `2` (`FC_MOE_DOWN_WPP_DEFAULT`) | `atoi`; only exact power-of-2 values accepted | Warps-per-pair for the cooperative MoE down-projection CUDA kernel. Values outside {1,2,4,8} are silently ignored. | `foundry/cuda/fc_cuda.c:2645` |
| `FC_MOE_DOWN_THREADS` | `64`, `128`, `256`, or `512` | `128` (`FC_MOE_DOWN_THREADS_DEFAULT`) | `atoi`; must be multiple of 32×wpp | Thread count for the MoE down kernel. Only exact values {64,128,256,512} that are a multiple of `32×wpp` accepted; others silently ignored. | `foundry/cuda/fc_cuda.c:2652` |

---

## Distributed training (foundry/distributed)

| Flag | Values | Default | Parse | Effect | Source |
|------|--------|---------|-------|--------|--------|
| `WORLD_SIZE` | integer | `1` | `atoi` | Distributed training world size. Used by `fd_init_from_env()` in the foundry trainer — not the inference server. | `foundry/distributed/distributed.c:31` |
| `RANK` | integer | `0` | `atoi` | This process's rank. Same scope as `WORLD_SIZE`. | `foundry/distributed/distributed.c:32` |

---

## Key parsing gotchas

1. **`QWEN35_BATCH_SLOTS` is an enable flag, not a slot count.** Only `env[0]=='1'`
   enables batching. `QWEN35_BATCH_SLOTS=8` leaves batching OFF.
   (`batch_sched.c:51`)

2. **`QWEN35_KV_Q4` default is now ON.** Only `=0` disables. The old opt-in
   value `=1` still works but is redundant. Any other value (e.g. `=true`)
   also keeps it ON. (`qwen35moe_forward.c:51`)

3. **`QWEN35_GPU_FULL_LOGITS` is blocked at serve startup.** The server refuses
   to start with this set — it is not silently ignored. (`serve_command.c:279`)

4. **`QWEN35_GPU_SPLIT_LAYER` is ignored when 1-die placement is active.**
   `gpu_split_layer()` returns `n_layer` (single-die path) when `n_placed==1`
   regardless of the env. (`qwen35moe_forward_cuda.c:703`)

5. **`QWEN35_GPU_DIE_A/B` are superseded by `--dies` / `pn_load_on` placement.**
   A warning is printed but they are not honoured. (`qwen35moe_forward_cuda.c:724`)

6. **`FC_MOE_DOWN_WPP` and `FC_MOE_DOWN_THREADS` silently reject invalid values**
   and keep the compiled default. No warning is emitted. (`foundry/cuda/fc_cuda.c:2645–2659`)

7. **`QWEN35_FUSE` parses via `strtol(env, NULL, 0)`** — it accepts hex (`0x3f`)
   and decimal. (`qwen35moe_forward_cuda.c:1888`)

8. **`PN_LOAD_SPIKE_BYTES=0` falls back to the 2.4 GB default**, not zero — a
   zero budget would silently disarm the host-RAM gate. (`pn_admission.c:284`)
