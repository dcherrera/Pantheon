# Pantheon

**An inference engine that assumes you don't have enough.**

Pantheon is a memory-aware inference engine and OpenAI-compatible model server in pure
C11, built for a small host, older or mixed cards, and several models per process: not
enough VRAM, host RAM, cores, context, or budget for current-generation GPUs.

- **Runs on:** CPU, Apple Metal, Tesla K80 (Kepler, 2014), Tenstorrent Blackhole p150a.
- **Headline, measured:** Qwen3.6-35B-A3B at Q8_0 on two Tesla K80s in a 2-core / 8 GB host
  decodes **33.5-38 tok/s for one user at 6-26k tokens of context**, and **24.5 tok/s
  aggregate for four agents** working at once (live server, 2026-11-25;
  [details](#qwen36-35b-a3b-q8_0-on-two-k80s)).
- **Degrades instead of failing:** a ranked memory ladder, admission checked against live
  host RAM and VRAM, and a context-over-budget error that tells the client the real numbers.
- **Quick start:** `git clone --recursive https://github.com/dcherrera/Pantheon.git && cd Pantheon && make USE_BLAS=1 pantheon`
  ([steps](#quick-start)).

It builds on [Foundry](https://github.com/dcherrera/Foundry), its
companion tensor library (a git submodule in this repository). Beyond Foundry it needs only
libc, libm, pthreads and libdl, plus vendored Monocypher (BSD-2-Clause, see NOTICE);
Pantheon Local also runs the system `curl` for model downloads. BLAS (Accelerate on macOS, OpenBLAS on Linux) is
optional, and the optional Metal backend links Apple's frameworks. Each backend builds
to a single binary.

---

## What "memory-aware" actually means

A common behavior when an engine runs out of memory is that the request fails. Pantheon
treats running out as the *normal operating condition* and degrades on a ranked ladder,
least-harmful first (`src/core/mem/pn_ladder.h`):

| Pressure | What gives |
|---|---|
| **WATCH** | force Q4 KV cache — near-zero quality cost, 4–7× VRAM win |
| **TIGHT** | halve concurrency · clamp the prefill window · evict cold KV |
| **CRITICAL** | floor concurrency to one slot — a single request always still completes |

Admission is checked before every load *and* every turn, against live host-RAM and VRAM
pressure, so the engine refuses work it cannot finish instead of accepting it and
falling over halfway. That layer (`src/core/mem/`) is about 2,100 lines of the codebase,
which is a fair measure of how central it is.

**Host RAM is admitted, not hoped for** (`src/core/mem/pn_host_admission.h`). Every large
host allocation — session KV rings, scheduler slots, snapshots, anonymous weight copies —
asks first, against `MemAvailable` minus what the process has already admitted, minus a
reserve. A default context window that does not fit is capped with a log line; an
explicit `--ctx-cap` / `PN_SESSION_MAX_CTX` that does not fit is refused at load with the
numbers; a new session the host cannot hold gets a 503 `server_memory_constrained`.
`PN_HOST_RESERVE_MB` sets the reserve (default 1536), `PN_HOST_ADMIT=0` disables it.
For an ad-hoc job on a shared box, a kernel-enforced backstop on top:
`systemd-run --user --scope -p MemoryMax=5G ./pantheon ...`.

**It tells the client the truth.** A prompt past the context cap is refused with a typed
`context_over_budget` carrying the actual KV numbers, so the caller can compact and
retry. Nothing is truncated silently: for an agent built on top, silent truncation would
be silent corruption.

## What that buys you

- **Many models, one process, many GPUs.** One port hosts N models across N devices,
  routed by the request's `model` field. Repeat a `--name` on another device and it
  becomes a replica, load-balanced least-busy. Load and unload at runtime without a
  restart. This is what lets a 2-core / 8 GB host drive several cards.
- **Portable session state.** Conversation state — recurrent SSM *and* attention KV —
  serializes to a versioned blob stamped with a model-identity tag the restore side
  validates. Hybrid state is hard to serialize because the recurrent half isn't
  position-addressable. It means session state can live on the user's machine instead
  of your disk.
- **Reasoning that terminates.** `--reasoning-budget N` forces `</think>` after N tokens,
  so a reasoning model that runs away still produces an answer instead of burning your
  context.
- **Embeddable.** C11, no C++ runtime, no exceptions, auditable dependency tree, links as
  a static library.

## Old hardware

The CUDA backend targets **`sm_37` — Kepler, 2014**. Measured by the author in a 2-core /
8 GB host:

- **One Tesla K80** (both GK210 dies): a 35B-A3B hybrid MoE at Q3_K_M served at 32K–128K
  context. Qwen3.6-35B-A3B and Nex-N2-mini decode at ~21 tok/s and prefill ~110-120 tok/s up
  to ~8k tokens ([K80 quick start](#qwen36-and-nex-n2-mini-on-a-tesla-k80),
  [supported models](docs/reference/supported-models.md)).
- **Two Tesla K80s:** Qwen3.6-35B-A3B at **Q8_0** decodes at 33.5-38 tok/s for one user
  ([numbers below](#qwen36-35b-a3b-q8_0-on-two-k80s)).

Correctness on that path, stated precisely:

- With the fused attention kernels off (`QWEN35_FUSE=0x37`) and flash prefill off
  (`QWEN35_FLASH=0`), the author measured the 2-die GPU forward's logits **bit-identical
  to the CPU forward** (argmax and every logit).
- The default flash prefill is argmax-exact, **not** bit-exact (~1e-6 relative drift; see
  [env flags](docs/reference/env-flags.md)).
- The in-tree CUDA gates (`tests/test_spec_qwen35`, `tests/test_users_qwen35`,
  `tests/test_prefix_cache_qwen35`) check token-for-token identity between decode paths
  under `QWEN35_FUSE=0x37`. They need a model and a GPU and are not part of `make test`.

Kepler is the extreme case, not the point. The point is that hardware you already own
should keep working.

---

## Quick start

Prerequisites: a C11 compiler and `make`. On Linux, `USE_BLAS=1` also needs OpenBLAS
(`apt install libopenblas-dev`); on macOS it uses Accelerate. The model download below
uses the Hugging Face CLI (`pip install -U huggingface_hub`); any way of fetching the
`.gguf` works. Build on the machine you will run on: the default flags include
`-march=native`, so a binary built on a newer CPU can fail with `Illegal instruction` on
an older one. To build a binary you can copy between x86-64 machines, add `PORTABLE=1`
(`-march=x86-64-v2`, no AVX). Per-platform details: [`docs/guides/install.md`](docs/guides/install.md).

```bash
git clone --recursive https://github.com/dcherrera/Pantheon.git pantheon
cd pantheon && make USE_BLAS=1 pantheon
# already cloned without --recursive?  git submodule update --init
# mirror: https://git.teamide.dev/David/Pantheon.git

hf download bartowski/SmolLM2-135M-Instruct-GGUF \
  SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

./pantheon --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "Hello, my name is" --max-tokens 32
```

Interactive chat, with the model's own template detected automatically:

```bash
./pantheon --model models/SmolLM2-135M-Instruct-Q8_0.gguf --chat
```

### Ornith 1.5 on a Tenstorrent card

Ornith 1.5 9B is the only model supported on Tenstorrent cards for now. On Linux with a
Blackhole p150a:

```bash
git clone --recursive https://github.com/dcherrera/Pantheon.git pantheon
cd pantheon && ./install.sh     # build pantheon-tt, set up card access (asks before sudo)
./run_ornith.sh                 # download the model (~10 GB, once), serve on 127.0.0.1:8080
```

`run_ornith.sh` fetches the files from
[dcherrera/Ornith-1.5-9B-Pantheon-TT](https://huggingface.co/dcherrera/Ornith-1.5-9B-Pantheon-TT)
and checks their sha256. The first start converts the model for the card (~10 GB more on
disk); later starts skip that. It serves an OpenAI-compatible API, model id `ornith`. Options
are in the script's header.

### Qwen3.6 and Nex-N2-mini on a Tesla K80

On Linux with a Tesla K80 and the NVIDIA 470 driver. **Supported K80 models: Qwen3.6-35B-A3B
and Nex-N2-mini (both Q3_K_M), one model per card** (both dies):

```bash
git clone --recursive https://github.com/dcherrera/Pantheon.git pantheon
cd pantheon && ./install.sh --k80   # build pantheon-cuda (no CUDA toolkit needed)
./k80_qwen36.sh                     # or ./k80_nexn2.sh; downloads ~17 GB once, serves on 127.0.0.1
```

Measured on one K80 in a 2-core / 8 GB host (OpenAI streaming client, warm card, runs back to back):

| prompt | Qwen3.6 first token | Nex-N2-mini first token | decode (both) |
|---|---|---|---|
| 1k tokens | 8 s (121 tok/s) | 9 s (107 tok/s) | ~21 tok/s |
| 8k tokens | 69 s (116 tok/s) | 67 s (119 tok/s) | ~18 tok/s |
| 14k tokens | 3.0 min (79 tok/s) | 3.2 min (74 tok/s) | ~15 tok/s |
| 30k tokens | 8.4 min (60 tok/s) | 9.6 min (53 tok/s) | ~10 tok/s |

**Good for:** chat, writing, Q&A and background jobs — short prompts, long answers. Prefix
caching is on: send a `session_id`, and send each earlier assistant turn back with its
`reasoning_content`; later turns then only prefill what's new (a 4k-token conversation's second
turn: ~38 s without the cache, 2-5 s with it). opencode does both on its own: point an
`@ai-sdk/openai-compatible` provider at the server and its second turn (14k-token system prompt)
measured 9 s instead of 170 s. One session id may carry several conversations (opencode's title
request shares it); each keeps its own cache slot.
**Not good for:** pasting long documents or codebases every request; prefill is the K80's weak
spot. Options are in each script's header.

### Qwen3.6-35B-A3B Q8_0 on two K80s

Pantheon Local profile `k80-two-boards`: Qwen3.6-35B-A3B at Q8_0, tensor-parallel across two
K80 cards (four dies), 65,536 context × 4 slots, Q4 KV cache, in the same 2-core / 8 GB host.

**Real use, measured 2026-11-25** on the live server through real agent traffic (read-only
code tasks; tool-call turns, so no MTP). Decode tok/s, median per request, one user:

| prompt | decode | previous build (2026-11-24) |
|---|---|---|
| 6-9k tokens | **38.0** | 33.0 |
| 10-13.5k tokens | **37.1** | 32.2 |
| 14-17k tokens | **35.5** | 31.1 |
| 18-21.5k tokens | **34.7** | 30.3 |
| 22-25.7k tokens | **33.5** | 29.0 |

- Four agents working together on real tasks: **24.5 tok/s aggregate** over a 20-minute
  window (previous build: 22.9). Real use stays prefill-bound: each tool turn prefills 1-2k
  tokens while the others decode, so this sits far below the decode-only figure.
- First token on a fresh 5k-token prompt: ~21 s (~245 tok/s prefill, 2026-11-24). Follow-up
  turns only prefill what's new (130-183 tok/s).
- Requests without a tool grammar use the model's MTP head: 36-39 tok/s at a 3.7k-token
  prompt (2026-11-19, earlier build).
- **Bench, not real use** (8k context, locked clocks): 1 slot 35.8 → 39.0
  tok/s, 4 slots 66.2 → 78.2 tok/s aggregate, from the exact short logits row, the faster
  top-k and the PCIe staging levers, all on by default (`=0` turns each off).

## Serving

One model, no GPU required:

```bash
./pantheon serve --model models/SmolLM2-135M-Instruct-Q8_0.gguf --port 8080
```

That's a complete OpenAI-compatible server. Point any client at
`http://127.0.0.1:8080/v1` and it works.

Everything below is optional. `--name` defaults to the file name, and `--dies` is
CUDA-only — omit it and the engine places the model itself. Repeat `--model` to serve
several from one process:

```bash
./pantheon serve \
  --model models/big.gguf   --name chat  --ctx-cap 32768 \
  --model models/small.gguf --name draft \
  --port 8080
```

Clients choose with the `model` field in a normal OpenAI request body. On a multi-GPU
box you can pin placement, and repeating a `--name` on another device makes it a
**replica** that load-balances least-busy:

```bash
./pantheon serve \
  --model models/big.gguf   --name chat   --dies 0,1 \
  --model models/small.gguf --name worker --dies 2 \
  --model models/small.gguf --name worker --dies 3 \
  --port 8080
```

Two `worker` requests then occupy devices 2 and 3 concurrently, with nothing in front of
the process doing the routing.

### Bringing up more than a couple of models

On a constrained host, launching every model at once is how you OOM. The pattern that
works is to start `serve` with one model and add the rest through
[`POST /admin/models/load`](docs/reference/http-api.md) **one at a time**, each gated on
free VRAM *and* host-RAM headroom, then settle-waited — visible in `/v1/models` — before
starting the next. Bring-up becomes idempotent and restartable: after a failure you
re-run and it picks up where it stopped, keeping what already loaded.

[`examples/deploy.sh`](examples/deploy.sh) is a working template that does exactly this,
driven by a manifest ([`examples/models.conf`](examples/models.conf)):

```
chat  | /models/chat.gguf  | ctx_cap=32768 max_sessions=4 max_concurrent=2
draft | /models/draft.gguf |
```

```bash
./examples/deploy.sh --plan     # dry run — print the plan, change nothing
./examples/deploy.sh            # reconcile (idempotent)
./examples/deploy.sh --prune    # also unload anything not in the manifest
```

It needs **bash and curl, nothing else** — no jq, no Python, no `nvidia-smi`. GPU state
comes from the engine's own `/admin/dies`, so the same script works unchanged on CPU,
Metal and CUDA. Copy it and adapt the gates; every per-model option the `serve` command
line accepts can be set per manifest line.

| | |
|---|---|
| **Endpoints** | `/v1/chat/completions` (SSE), `/v1/models`, `/admin/models/*`, `/admin/dies` |
| **Per-model caps** | `--ctx-cap`, `--max-sessions`, `--max-concurrent`, `--reasoning-budget` |
| **Scheduling** | batch scheduler with a bounded queue, then an honest `429` + `Retry-After` |
| **Caching** | prefix cache with longest-common-prefix reuse across turns |
| **Structured output** | grammar-constrained decoding for tool calls and `response_format` |
| **Ops** | load/unload without restart; `/v1/models` stays live |

A `pantheon router` subcommand fans across separate serve processes.

## Supported models

| Family | Covers |
|---|---|
| **Llama** | Llama, Qwen2, Mistral, Mixtral — one shared forward pass |
| **Phi-3** | fused QKV, NeoX RoPE |
| **Qwen3.5 / 3.6 MoE** | hybrid gated-delta SSM + attention, grouped experts, 2-die split |
| **RWKV-X** | hybrid recurrent + sparse attention, WKV-7 kernel |
| **Pixtral** | vision-language, continuous-batch decoder for bulk OCR |

**Formats:** GGUF (v2/v3, K-quants and legacy) and safetensors, mmap'd zero-copy.
Per-model quants, measured throughput and a runnable command:
[`docs/reference/supported-models.md`](docs/reference/supported-models.md).

## Backends

| Backend | Flag | Notes |
|---|---|---|
| CPU | *(default)* | NEON, AVX2, scalar |
| BLAS | `USE_BLAS=1` | Accelerate on macOS, OpenBLAS on Linux |
| Threads | `USE_THREADS=1` | pthread pool |
| Metal | `USE_METAL=1` | Apple Silicon |
| CUDA | `USE_CUDA=1` | `sm_37`+, `dlopen`ed with kernels embedded as PTX — builds without nvcc. **Supported on Tesla K80: Qwen3.6-35B-A3B and Nex-N2-mini** |
| Tenstorrent | `USE_TT=1` | Blackhole p150a, Linux only; programs embedded, no TT software stack needed to build. **Supported model: Ornith 1.5 9B only** — other models may load on the card but are not supported |

Because CUDA is `dlopen`ed, one binary runs on a machine with no NVIDIA driver at all and
lights up the GPU path on a machine that has one.

```bash
make USE_BLAS=1 pantheon        # CLI
make USE_BLAS=1 libpantheon.a   # static library, to embed
make USE_BLAS=1 test            # test suites
make DEBUG=1 USE_BLAS=1 pantheon
```

### The CLI binary is named for its backend

Each accelerator backend writes its **own** file, and `pantheon` is a symlink to whichever
was built last:

| Build | Binary | `pantheon` symlink |
|---|---|---|
| `make` | `pantheon-cpu` | → `pantheon-cpu` |
| `make USE_CUDA=1` | `pantheon-cuda` | → `pantheon-cuda` |
| `make USE_TT=1` | `pantheon-tt` | → `pantheon-tt` |
| `make USE_METAL=1` | `pantheon-metal` | → `pantheon-metal` |

They used to be one file, so the last build silently owned the deployed path — which is
how a CPU-only binary came to serve a 35B MoE with every GPU die idle. A build for one
backend can no longer overwrite another's binary; at worst it re-points the symlink, and
`ln -sfn pantheon-cuda pantheon` puts that back with no rebuild.

Ask a binary what it is rather than inferring it from its path:

```bash
./pantheon --backend                        # -> cuda|tt|metal|cpu (one token, for scripts)
./pantheon --version                        # -> Pantheon v0.4.0 / backend=cuda build=<commit>
curl -s localhost:8080/ | jq .backend       # a running serve
scripts/assert_backend.sh ./pantheon cuda   # deploy-time assertion; exit 1 on mismatch
```

`pantheon serve` prints `backend=... build=... accelerator=...` as its first line and the
visible die count as its second, and it **refuses to start** when `--dies` names dies this
binary has no backend for. To serve on the CPU deliberately, simply do not pass `--dies`.

## Embedding

Pantheon is a library first and a CLI second. The public API is
[`include/pantheon.h`](include/pantheon.h) — opaque types, explicit ownership, error codes
rather than exceptions, no global state. Link `libpantheon.a` from C or any language with
a C FFI.

## What Pantheon is not

Being clear about this saves everyone time:

- **It is not built for speed on modern datacenter GPUs.** Engines designed for that
  hardware will be faster there.
- **Prefix caching, structured output, n-gram speculative decoding and continuous
  batching are table stakes**, not differentiators. Everyone has them. They are here
  because an engine without them isn't serious, not because they're novel.
- **It is not a distributed serving framework.** One process is self-routing and bounded;
  you scale by adding servers, not by packing more into one.
- **It is one person's project.** Model coverage is narrower than that of larger projects.

## Documentation

[`docs/`](docs/) — start at [`docs/README.md`](docs/README.md).

- [Quickstart](docs/guides/quickstart.md) · [Chat templates](docs/guides/chat.md) · [.pth converter](docs/guides/pth-convert.md)
- [Environment flags](docs/reference/env-flags.md) · [CLI flags](docs/reference/cli.md) · [HTTP API](docs/reference/http-api.md) · [Supported models](docs/reference/supported-models.md)
- Architecture: [engine](docs/architecture/engine.md) · [serve layer](docs/architecture/serve.md) · [memory-aware layer](docs/architecture/memory-aware.md) · [die manager](docs/architecture/accel-foundry.md) · [GGUF](docs/architecture/gguf.md) · [Llama forward](docs/architecture/llama-forward.md) · [Qwen3.5-MoE](docs/architecture/models-qwen35moe.md) · [RWKV-X](docs/architecture/rwkvx.md)

## Built on Foundry

Pantheon uses **Foundry** for tensor compute, safetensors, tokenization, quantization,
attention primitives and the CUDA layer. Same author, same license.

In the public repository Foundry is a git submodule at `foundry/`
([`https://github.com/dcherrera/Foundry`](https://github.com/dcherrera/Foundry); mirror at
[`git.teamide.dev/David/Foundry`](https://git.teamide.dev/David/Foundry)).
The Makefile resolves it as `FOUNDRY=/path`, then `./foundry` (the submodule, or a
symlink in a development checkout), then `../foundry` (a sibling checkout); if none exists it stops with a message
saying so rather than a wall of missing headers.

## License

**AGPL-3.0** — see [`LICENSE`](LICENSE).

Contributions come in under the CLA in [`CLA.md`](CLA.md), with a DCO sign-off on every
commit — see [`CONTRIBUTING.md`](CONTRIBUTING.md). You keep your copyright; signing is one
entry in [`.cla/signatures.md`](.cla/signatures.md), added in your first pull request.

Copyright © 2026 David (Corey) Herrera and the Pantheon contributors. See [`NOTICE`](NOTICE).
