# Install

Build and run Pantheon on each supported platform. For a first run, see
[Quickstart](quickstart.md).

## Requirements

- A C11 compiler and `make`.
- Foundry, a git submodule at `./foundry`. Clone with
  `git clone --recursive https://git.teamide.dev/David/Pantheon.git`; in an existing
  clone run `git submodule update --init`. The Makefile looks for Foundry in this order:
  `FOUNDRY=/path/to/foundry`, then `./foundry` (the submodule, or a symlink), then
  `../foundry` (a sibling checkout). If none exists the build stops with
  `Foundry not found at ...`.
- Nothing else for a CPU build. BLAS, threads and accelerators are opt-in flags.

## Linux (CPU)

```bash
sudo apt install libopenblas-dev        # only for USE_BLAS=1
make USE_BLAS=1 USE_THREADS=1 pantheon
```

- `USE_BLAS=1` links OpenBLAS (`-lopenblas`).
- `USE_THREADS=1` adds the compute thread pool; size it with `PN_THREADS` (`0` = auto).

## macOS

```bash
make USE_BLAS=1 USE_THREADS=1 pantheon             # Accelerate
make USE_BLAS=1 USE_THREADS=1 USE_METAL=1 pantheon # + Metal GPU backend
```

`USE_BLAS=1` uses Apple's Accelerate framework, so no extra install is needed.
`USE_METAL=1` (macOS only) compiles Foundry's Objective-C Metal backend and links the
Metal, MetalPerformanceShaders and Foundation frameworks.

## NVIDIA CUDA

```bash
make USE_CUDA=1 USE_BLAS=1 pantheon    # Linux only
```

- **No CUDA toolkit or `nvcc` needed to build.** The kernels are embedded as PTX
  (ISA 7.4, target `sm_37`) and the driver is loaded with `dlopen` at run time.
- **At run time** the process needs `libcuda.so.1` (the NVIDIA driver). The PTX needs a
  driver that supports CUDA 11.4 or newer; Kepler cards such as the Tesla K80 top out at the
  470 driver branch. cuBLAS (`libcublas.so.11`) is optional — without it a warning prints and
  the custom kernels are used.
- With no driver or no card, the same binary runs on the CPU. `GET /` then reports
  `"dies_visible": 0`.
- Pick devices with `--dies 0,1` or `CUDA_VISIBLE_DEVICES`; die numbers are relative to the
  visible devices.

## Tenstorrent

**Ornith 1.5 9B is the only model supported on the card for now**; other models may load but
are not supported. The quick way is the two scripts at the repo root:

```bash
./install.sh       # build pantheon-tt, set up card access (udev rule; asks before sudo)
./run_ornith.sh    # download + check the model, serve on 127.0.0.1:8080
```

By hand:

```bash
make USE_TT=1 pantheon    # Linux only, Blackhole p150a
```

- **No TT software needed to build.** Tensix programs are embedded in the binary
  (`foundry/tt/ft_tt_programs.h`), so neither SFPI nor tt-metal is required.
- **At run time** the card must be reachable: by default through tt-kmd
  (`/dev/tenstorrent/<n>`), or with `FT_TT_DRIVER=udrv` through Foundry's userspace driver,
  which needs no kernel module but needs root or the udev rule
  `foundry/tt/tools/60-foundry-blackhole.rules`. An explicit `FT_TT_DRIVER=udrv` that cannot
  reach the card stops the process rather than falling back to the CPU.
- Read [accel-foundry.md §7](../architecture/accel-foundry.md) before running on the card:
  what it can serve today, known limits, and hazards that cost a power cycle.

## Which binary you built

Each backend writes its own file and `pantheon` is a symlink to the last one built:

| Build | Binary |
|---|---|
| `make` | `pantheon-cpu` |
| `make USE_CUDA=1` | `pantheon-cuda` |
| `make USE_TT=1` | `pantheon-tt` |
| `make USE_METAL=1` | `pantheon-metal` |

Combined flags get combined names (e.g. `pantheon-cuda-tt`). Check what a binary is with
`./pantheon --backend` (prints `cuda`, `tt`, `metal` or `cpu`). Restore the link without
rebuilding: `ln -sfn pantheon-cuda pantheon`.

## Hardware notes

- **Build on the machine you run on.** The default flags include `-march=native`, so a binary
  built on an AVX2 machine dies with `Illegal instruction` (SIGILL) on an older CPU such as a
  no-AVX Pentium. `PORTABLE=1` builds a copyable binary instead (`-march=x86-64-v2` on x86-64,
  the compiler default elsewhere), at some cost in speed.
- **Small hosts.** The `pantheon` binary is a single compiler invocation; `-j` only helps the
  test build. On a 2-core host use `make -j2 ... test`.
- **Host RAM.** 8 GB is the design point: weights are mmap'd, and after a GPU upload the CUDA
  paths drop the weight page cache (`QWEN35_DROP_WEIGHTS`, on by default).

## Running the tests

```bash
make USE_BLAS=1 test
```

Tests that need a model file print `[SKIP]` with the env var to set (e.g. `PN_QWEN35_MODEL`).

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `Illegal instruction` at start | Built on a newer CPU. Rebuild on the target machine. |
| `serve: REFUSED to start — ... has NO accelerator backend compiled in` | `--dies` was given to a binary without that backend. Check `./pantheon --backend`; build with `USE_CUDA=1` / `USE_TT=1`, or drop `--dies` to serve on the CPU. |
| `serve: REFUSED ... the die manager probed 0 dies` | Backend present but no device visible: driver not loaded, card missing, or `CUDA_VISIBLE_DEVICES` empty. |
| HTTP `429 server_busy` with `Retry-After` | Every decode slot is busy and the queue is full (or a 30 s queue wait timed out). Retry after the given seconds. |
| HTTP `429 model_at_session_capacity` | A new `session_id` while all `max_sessions` slots hold live sessions. Release idle sessions (`POST /session/release`) or raise `--max-sessions`. |
| HTTP `503 server_memory_constrained` | Host memory pressure: admission refused the prompt, or the queue is full under critical pressure. Transient; retry later. |
| HTTP `422 context_over_budget` | The prompt exceeds the per-session window (`kv_cap`). Not transient: shorten the conversation, then retry. Nothing is dropped silently. |
| `bpe: tokenizer.ggml.pre is unsupported; falling back to the qwen35/GPT-2 pre-tokenizer` | The model's pre-tokenizer is not implemented (e.g. SmolLM2). It still runs, but prompt token IDs may differ from the reference. See [supported-models.md](../reference/supported-models.md#known-issues). |

Full error list: [HTTP API](../reference/http-api.md#error-responses).
