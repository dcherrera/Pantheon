# Pantheon Documentation

Pantheon is a pure-C11 inference engine and OpenAI-compatible model server, built on Foundry (a git submodule). Beyond Foundry it needs libc, libm, pthreads and libdl; BLAS and the Metal backend's Apple frameworks are optional.

## ⭐ Multi-Model Smart Routing — the flagship feature

Pantheon is a **memory-aware, multi-model server**. A **single running instance** hosts **several models across several GPUs at once** and routes every request intelligently — so a tiny host (e.g. 2 cores / 8 GB RAM) can drive a fleet of cards and run models on hardware they were never meant for. Tying *multiple* instances together is a separate future **gateway**; **one instance is fully self-routing** and needs nothing in front of it.

**How routing works (all inside one `pantheon serve` process):**

- One process binds **one port**. Each `--model` is an independent registry slot with its own weights, state, and die placement. A request picks its model by the **`model` field** in the JSON body, matched to each model's `--name` (one model loaded ⇒ the field is ignored).

- **Replicas = the same `--name` on different dies.** `mr_acquire` (`src/serve/model_registry.c`) load-balances across replicas by **least-busy** (fewest in-flight refs); with batching off each replica runs on its own die, so requests execute **concurrently**: two concurrent requests to a model replicated on dies 2 and 3 keep both dies busy at once and finish in about one request's wall-clock.

- **Memory-aware admission** ([Memory-Aware Layer](architecture/memory-aware.md)) gates every load and turn against live host-RAM + VRAM pressure, keeping the instance in a small host footprint.

- **Optional per-model concurrency:** `QWEN35_BATCH_SLOTS=1` adds up to 8 batched slots on a single die (off by default — see [reference/env-flags.md](reference/env-flags.md)).

**Launch it** — one instance, an orchestrator plus a worker replicated across two dies:

```bash
./pantheon serve \
  --model /models/orchestrator.gguf --name orch   --dies 0,1 \
  --model /models/worker.gguf       --name worker --dies 2 \
  --model /models/worker.gguf       --name worker --dies 3 \
  --port 8201
```

Clients POST `/v1/chat/completions` with `"model":"orch"` or `"model":"worker"`; `worker`
requests fan out across dies 2 and 3 by least-busy, concurrently. No router needed.

> **Known gap (roadmap):** replica selection is least-busy but **not yet session-sticky** — under concurrent load a multi-turn session can hop replicas and miss its prefix cache. Session affinity across replicas is designed but not yet shipped.

## ⭐ Reference Library (flags · CLI · HTTP API)

The authoritative, code-audited reference (2026-07-12) — every environment flag, CLI flag, and HTTP endpoint, with values, defaults, effects, source `file:line`, and the **parsing gotchas that bite** (e.g. `QWEN35_BATCH_SLOTS` is an *enable* flag keyed on `env[0]=='1'`, so `=8` silently disables batching; `--dies` indices are relative to `CUDA_VISIBLE_DEVICES`, not physical devices).

- [**Environment Flags**](reference/env-flags.md) — all `QWEN35_*` / `PN_*` / `FC_*` + distributed vars (\~40 flags)

- [**CLI Flags**](reference/cli.md) — `pantheon serve` global + per-model flags, `--dies` semantics, multi-model binding

- [**HTTP API**](reference/http-api.md) — `/v1/models`, `/v1/chat/completions` (SSE), `/admin/models/*`, `/admin/dies`

- [**Supported Models**](reference/supported-models.md) — every verified model with its quant, config, measured throughput and a runnable command

## Architecture — serving-era (code-audited 2026-07-12)

- [Serve Layer](architecture/serve.md) — request lifecycle, model registry + multi-card load-balancing, the batch scheduler, streaming/SSE, the lock table

- [Qwen3.5-MoE Model](architecture/models-qwen35moe.md) — the qwen35moe arch (e.g. Nex-N2-mini), 2-die split/placement, KV cache, decode slots, async step machine *(supersedes the legacy `qwen35moe.md` below)*

- [Core Engine](architecture/core-engine.md) — `pn_load`/`pn_load_on`, the forward-step async seam, admission helpers

- [Memory-Aware Layer](architecture/memory-aware.md) — pressure levels (OK/WATCH/TIGHT/CRITICAL), pre-flight admission, the reactive ladder (why TIGHT is the mini's normal resting state)

- [Accelerator / Die Manager + Foundry CUDA](architecture/accel-foundry.md) — die abstraction, CUDA↔die mapping under `CUDA_VISIBLE_DEVICES`, per-die VRAM budgeting, foundry primitives (streams/events/graphs), and **§7 the second backend: Tenstorrent Blackhole p150a** (`make USE_TT=1`; tt-kmd + SFPI only — no tt-metal, TT-NN, Python or C++) — what it serves today, its runtime flags, and the hazards that cost hardware

## Guides

- [Quickstart](guides/quickstart.md) — Build, download a model, generate text

- [Install](guides/install.md) — Per-platform build and run (CPU, macOS, CUDA, Tenstorrent), troubleshooting

- [Chat Templates & Interactive Chat](guides/chat.md) — Multi-turn chat, template detection, C API

- [PyTorch .pth Converter](guides/pth-convert.md) — Convert .pth to safetensors in pure C, no Python

## Architecture

- [Engine Overview](architecture/engine.md) — How model loading, dispatch, and generation work, plus the vision-language encode/decode seam and continuous-batch VLM decoder for bulk OCR

- [GGUF Format](architecture/gguf.md) — GGUF parser, quant types, v2/v3 support

- [Llama Forward Pass](architecture/llama-forward.md) — Shared forward pass for Llama, Qwen2, Mistral, Mixtral + Phi-3 differences, batched multi-sequence decode, flash prefill, and the opt-in GPU paths

- [Qwen3.5-MoE Forward Pass (legacy)](architecture/qwen35moe.md) — the original forward-pass writeup. **⚠ Predates the A–D serving stack; superseded by **[**architecture/models-qwen35moe.md**](architecture/models-qwen35moe.md)** (the 2026-07-12 code audit). Kept for any unique detail; prefer the new doc.**

- [RWKV-X Forward Pass](architecture/rwkvx.md) — Hybrid recurrent + sparse attention, WKV-7 kernel, safetensors loading

## Serving & multi-tenant

The serving architecture itself is documented in [architecture/serve.md](architecture/serve.md); what follows is a map of the behaviour it implements.

- **Multi-card concurrency** — the host-memory + concurrency stack that lets a small host (8 GB / 2 cores) drive up to 8 cards: the die is the atomic unit, at most one model per die, reliability first, and capacity scales by adding servers rather than packing models.

- **Single vs multi-tenant** — one card serves either one fast user or K slower ones; the two knobs are multi-model placement and `QWEN35_BATCH_SLOTS` batching. See [reference/env-flags.md](reference/env-flags.md).

- **Multi-model serving** — registry, `model`-field routing, runtime load/unload, die placement.

- **Fleet bring-up** — load models one at a time through `/admin/models/load`, gated and settle-waited. [`examples/deploy.sh`](../examples/deploy.sh) is a working template.


## API Reference

- [`include/pantheon.h`](../include/pantheon.h) — Public C API (opaque types, generation, sampling, vision-language describe, and the continuous-batch VLM decoder)

- [`src/core/engine.h`](../src/core/engine.h) — Internal types: config, architecture registry, state, `pn_load_opts`

- [`src/formats/gguf.h`](../src/formats/gguf.h) — GGUF file format types and parser

- [`src/models/llama.h`](../src/models/llama.h) — Llama architecture weight structs and forward pass

- [`src/models/phi3.h`](../src/models/phi3.h) — Phi-3 weight structs (fused QKV, fused gate/up)

- [`src/models/rwkvx.h`](../src/models/rwkvx.h) — RWKV-X adapter (safetensors loading, trie tokenizer, WKV-7 forward)

- [`src/core/chat.h`](../src/core/chat.h) — Chat template detection and formatting

- [`src/core/qweight.h`](../src/core/qweight.h) — Quantized weight matrix (zero-copy mmap)

- [`src/accel/simd.h`](../src/accel/simd.h) — SIMD kernels (NEON, AVX2, scalar)

- [`src/memory/arena.h`](../src/memory/arena.h) — Bump allocator for inference scratch memory

## Contributing

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — build flags, code style, testing bar, and the licensing terms contributions are made under
