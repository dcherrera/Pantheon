# Pantheon — CLI Reference

Entry point: `cli/main.c`. `pantheon serve` runs the OpenAI-compatible server
(`src/serve/serve_command.c`), `pantheon router` the multi-instance dispatch layer
(`src/serve/router_command.c`); anything else is the one-shot / interactive CLI.

---

## `pantheon` (one-shot and chat)

```
pantheon --model <path> [--prompt <text> [--raw] | --chat [--system <text>]]
         [--max-tokens <n>] [--temperature <f>] [--top-p <f>] [--top-k <n>]
         [--config <path>] [--vocab <path>] [--quantize] [--bench]
pantheon --version | --backend | --help
```

| Flag | Argument | Default | Effect |
|------|----------|---------|--------|
| `--model` | `<path>` | — | GGUF or safetensors model. Without it, prints the version and checks Foundry links. |
| `--prompt` | `<text>` | `"Hello"` | One-shot generation. The model's chat template is applied unless `--raw`. |
| `--raw` | *(flag)* | off | One-shot `--prompt` without the chat template. |
| `--chat` | *(flag)* | off | Interactive multi-turn chat on stdin; template detected from the GGUF. |
| `--system` | `<text>` | unset | System prompt (chat mode only). |
| `--max-tokens` | `<n>` | `128` | Tokens to generate. |
| `--temperature` | `<f>` | `0.8` | Sampling temperature. |
| `--top-p` | `<f>` | `0.9` | Nucleus sampling threshold. |
| `--top-k` | `<n>` | `40` | Top-k sampling. |
| `--config` | `<path>` | unset | HuggingFace `config.json` (safetensors models). |
| `--vocab` | `<path>` | unset | RWKV World tokenizer vocab file. |
| `--quantize` | *(flag)* | off | INT8-quantize weights at load time. |
| `--bench` | *(flag)* | off | Report tok/s and memory. |
| `--version` | *(flag)* | — | `Pantheon v<ver>` then `backend=<b> build=<commit>`. |
| `--backend` | *(flag)* | — | Prints only the compiled backend token (`cuda`, `tt`, `metal`, `cpu`). |
| `-h` / `--help` | *(flag)* | — | Usage. |

---

## `pantheon serve`

```
pantheon serve --model <path> [--name <id>] [--dies <a[,b,...]>]
               [--ctx-cap <n>] [--reasoning-budget <n>]
               [--max-sessions <n>] [--max-concurrent <n>]
               [--model <path> ...]
               [--host <addr>] [--port <n>]
               [--config <path>] [--vocab <path>]
               [--quantize]
               [--served-name <id>]
```

`--model` is repeatable. The per-model flags `--name` and `--dies` bind to
the **most recently seen** `--model`. There is no positional association;
flags are matched by order.

### Global flags

| Flag | Argument | Default | Effect |
|------|----------|---------|--------|
| `--model` | `<path>` (GGUF) | — *required, at least one* | Load and serve the model at this path. Repeatable; each occurrence starts a new model slot. Max `MR_MAX_MODELS=16`. |
| `--host` | `<addr>` | `127.0.0.1` | TCP bind address. Must be a dotted-IPv4 literal (`inet_pton`). |
| `--port` | `<n>` | `8080` | TCP port. Parsed with `atoi`. |
| `--config` | `<path>` | unset | HuggingFace `config.json` for safetensors models. |
| `--vocab` | `<path>` | unset | RWKV World tokenizer vocab file. |
| `--quantize` | *(flag)* | off | INT8-quantize weights at load time. |
| `--served-name` | `<id>` | unset | Back-compat alias: sets the **first** model's served name when it has no `--name`. Silently ignored if `--name` already set for model 0. |
| `-h` / `--help` | *(flag)* | — | Print usage and exit 0. |

### Per-model flags

These flags bind to the model specified by the **preceding** `--model`. If
used before any `--model` the server exits with an error.

| Flag | Argument | Default | Effect |
|------|----------|---------|--------|
| `--name` | `<id>` | basename of path | Served model id (the `model` field clients send in requests). Used for routing in multi-model mode. |
| `--dies` | `<a[,b,...]>` | none (auto-place) | Pin to accelerator dies (CUDA or Tenstorrent builds). CUDA ordinals are relative to `CUDA_VISIBLE_DEVICES`, not physical device numbers. Max `PN_MODEL_MAX_DIES=8`. N dies split the layers N ways (VRAM headroom, not speed). A `--dies` naming dies this binary has no backend for refuses to start; omit it to serve on the CPU. |
| `--ctx-cap` | `<n>` | `0` = engine default | Per-session context cap in tokens (the KV window). A longer prompt is refused with `context_over_budget`; nothing is silently dropped. Max 2^20. |
| `--reasoning-budget` | `<n>` | `0` = unbounded | After `<n>` tokens inside `<think>` without closing it, force `</think>` so the model answers. |
| `--max-sessions` | `<n>` | `0` = default 8 (`BS_MAX_USERS`) | Cap on live prefix-cache sessions; a new session over the cap is refused 429/503. |
| `--max-concurrent` | `<n>` | `0` = the resolved `--max-sessions` | Cap on simultaneous decode slots; surplus requests queue, then get 429/503. |

### Multi-model semantics

```
pantheon serve \
  --model /models/big.gguf   --name big   --dies 0,1 \
  --model /models/small.gguf --name small --dies 2
```

All models share the **single `--port`** (`serve_command.c:333`, one `hs_listen` call).

Each `--model` occupies an independent registry slot (`model_registry.c:117–122`). Per
slot, on load:
- Always: `pn_model` (weights) + `gen_lock` (non-batched serializer).
- Non-batched path (`QWEN35_BATCH_SLOTS` unset): also gets a `pn_state`; `gen_lock`
  serializes single-user access — second concurrent request to the same model → 429.
- Batched path (`QWEN35_BATCH_SLOTS=1`): gets a `pn_batch_sched` instead
  (`model_registry.c:272–282`, `bs_create`); the `gen_lock`+single-`pn_state`
  serializer is bypassed. Up to `BS_MAX_USERS=8` concurrent users share one model.

Requests route by the `model` field in the JSON body (`openai_chat.c:616–629`,
`route_model` → `mr_acquire`). With exactly one loaded model, the field is ignored
(back-compat).

GPU uploads are serialized (one at a time) via `reg->upload_lock` to prevent stacking
the ~2.4 GB host transient.

`--served-name` affects only model 0 when `--name` was not also given for it.

#### Replicas (same name, different dies)

Repeating the **same `--name`** on different `--dies` creates a load-balanced replica
pool (`model_registry.c:117–119`):

```
pantheon serve \
  --model /models/worker.gguf --name worker --dies 2 \
  --model /models/worker.gguf --name worker --dies 3
```

`mr_acquire` (`model_registry.c:322–355`) picks the **least-busy** same-name entry
(fewest in-flight `refs`); each replica has its own die and `gen_lock`, so concurrent
requests run truly in parallel across those dies.

Session affinity across replicas within a single serve process is **not yet
implemented** — requests always go to the least-busy replica. See the
"Multi-Model Smart Routing" section in `docs/README.md`.

### Die index semantics

`--dies 0,1` means CUDA ordinals 0 and 1 as seen by the process — after
`CUDA_VISIBLE_DEVICES` filtering. If `CUDA_VISIBLE_DEVICES=4,5,6,7` is set,
`--dies 0,1` maps to physical cards 4 and 5. **The indices are NOT physical
device numbers.**

### Startup sequence

1. `pn_memwatch_start()` — host RAM poll thread starts.
2. Signal handlers installed (SIGINT/SIGTERM → clean CUDA teardown; crash
   signals → `_exit()` so the kernel closes the nvidia fd and reclaims VRAM).
3. CUDA init (`fc_init()`, `pn_dm_init_cuda()`).
4. Model registry init.
5. For each `--model`: `mr_load()` (blocks; uploads weights to GPU).
6. TCP listen socket bound.
7. Accept loop: each connection spawns a detached thread.

### Runtime behavior

- Cheap routes (GET `/v1/models`, OPTIONS) are never queued behind a
  generation.
- Per-model generation is serialized by `gen_lock`. A second request to the
  same model while one is generating → immediate 429.
- With `QWEN35_BATCH_SLOTS=1`, the scheduler (`batch_sched`) replaces the
  gen\_lock; up to `BS_MAX_USERS=8` concurrent users share one model.
- `QWEN35_GPU_FULL_LOGITS` being set causes immediate exit before any model
  loads.

---

## `pantheon router`

```
pantheon router [--host <addr>] [--port <n>]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8100` | Port. |

A host-side dispatch layer over several `pantheon serve` processes: no model loads, no
GPU. Serve instances register themselves when `PANTHEON_REGISTRY_URL` points at the router,
then heartbeat; instances whose heartbeat lapses are evicted. Requests go to the
least-busy live instance of the requested model, and a session (`X-Session-Id`) sticks to
the instance that served it so its prefix cache is reused. No authentication: bind it to
a trusted interface.

| Route | Purpose |
|-------|---------|
| `POST /registry/register` | A serve announces a model instance |
| `POST /registry/heartbeat` | Refresh liveness and load |
| `DELETE /registry/deregister` | Graceful removal |
| `GET /registry/fleet` | Model-level fleet view |
| `GET /v1/models` | Live models, deduplicated by model id |
| `POST /v1/chat/completions`, `POST /v1/completions` | Least-busy dispatch, SSE pass-through |

---

## `pantheon-convert` (cli/convert.c)

A separate binary (`make pantheon-convert`), not a subcommand: converts PyTorch `.pth`
checkpoints to safetensors. See [`../guides/pth-convert.md`](../guides/pth-convert.md).
