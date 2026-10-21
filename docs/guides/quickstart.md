# Quickstart

Build Pantheon and generate text in under a minute.

## Build

```bash
git clone --recursive https://git.teamide.dev/David/Pantheon.git
cd Pantheon
make USE_BLAS=1 pantheon
```

Foundry comes in as a git submodule at `foundry/`; in a clone made without
`--recursive`, run `git submodule update --init` first.

Build flags:

| Flag | What | When |
|------|------|------|
| `USE_BLAS=1` | Accelerate (macOS) / OpenBLAS (Linux) | Always |
| `USE_THREADS=1` | Thread pool for parallel attention | 1B+ models |
| `DEBUG=1` | `-O0 -g` debug build | Debugging |

## Download a Model

Any GGUF model from HuggingFace works. RWKV-X safetensors models are also supported. Start small:

```bash
# SmolLM2 135M (fast, good for testing)
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \
  SmolLM2-135M-Instruct-Q8_0.gguf --local-dir ../models/base/

# RWKV-X 0.2B (hybrid recurrent model, safetensors format)
huggingface-cli download howard-hou/RWKV-X \
  RWKV-X-0.2B-64k-Base.safetensors --local-dir ../models/base/

# Phi-3 mini (3.8B, good quality for size)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir ../models/base/

# Mistral 7B v0.3 (larger, needs USE_THREADS=1)
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir ../models/base/
```

See [reference/supported-models.md](../reference/supported-models.md) for the full list of verified models.

## Generate Text

```bash
# GGUF model
./pantheon --model ../models/base/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "The meaning of life is" --max-tokens 64

# RWKV-X safetensors model (requires --vocab)
./pantheon --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" --max-tokens 32

# RWKV-X with INT8 quantization (~3x faster on ARM)
./pantheon --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" --max-tokens 32 --quantize
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <path>` | (required) | Model file (GGUF or safetensors) |
| `--vocab <path>` | none | RWKV World tokenizer vocab file (required for safetensors) |
| `--prompt <text>` | `"Hello"` | Input prompt |
| `--max-tokens <n>` | 128 | Maximum tokens to generate |
| `--temperature <f>` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-p <f>` | 0.9 | Nucleus sampling threshold |
| `--top-k <n>` | 40 | Top-k sampling |
| `--quantize` | off | INT8 quantize weights at load time (RWKV-X) |
| `--chat` | off | Interactive multi-turn chat mode |
| `--system <text>` | none | System prompt (chat mode only) |
| `--bench` | off | Report tok/s, load time, memory |
| `--help` | | Show usage |
| `--version` | | Print version |

## Benchmark

```bash
./pantheon --model ../models/base/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "The meaning of life is" --max-tokens 64 --bench --temperature 0
```

```
=== Benchmark ===
  load:    41.6 ms
  decode:  64 tokens in 787.3 ms (81.3 tok/s)
  memory:  552.7 MB RSS
=================
```

## C API

Pantheon exposes a minimal C API for embedding into applications:

```c
#include "pantheon.h"

int main(void) {
    /* GGUF model */
    pn_model *model = pn_load("model.gguf");

    /* Or: RWKV-X safetensors model */
    pn_load_opts opts = { .vocab_path = "rwkv_vocab.txt", .quantize = true };
    pn_model *rwkvx = pn_load_ex("model.safetensors", &opts);

    pn_state *state = pn_state_new(model);
    pn_sampling_params params = pn_sampling_defaults();
    params.temperature = 0.0f;  /* greedy */

    char *output = pn_generate(model, state, "Hello world", 64, &params);
    printf("%s\n", output);

    free(output);
    pn_state_free(state);
    pn_model_free(model);
    return 0;
}
```

### Streaming

```c
void on_token(const char *token, void *user_data) {
    printf("%s", token);
    fflush(stdout);
}

pn_stream(model, state, "Hello", 128, &params, on_token, NULL);
```

### API Functions

| Function | Description |
|----------|-------------|
| `pn_load(path)` | Load GGUF model, returns `pn_model*` |
| `pn_load_ex(path, opts)` | Load any model (GGUF or safetensors) with options |
| `pn_model_free(model)` | Free model and weights |
| `pn_state_new(model)` | Create inference state (KV cache + arena, or RWKV-X state) |
| `pn_state_free(state)` | Free state |
| `pn_state_reset(state)` | Clear state and position (reuse state) |
| `pn_generate(model, state, prompt, max_tokens, params)` | Generate text, returns `char*` (caller frees) |
| `pn_stream(model, state, prompt, max_tokens, params, cb, ctx)` | Streaming generation via callback |
| `pn_stream_tokens(model, state, tokens, n, max, params, cb, ctx)` | Streaming from pre-tokenized input (for chat) |
| `pn_sampling_defaults()` | Default sampling params (temp=0.8, top_p=0.9, top_k=40) |

## Run Tests

```bash
make USE_BLAS=1 test
```

62 tests across 7 suites: build sanity (5), GGUF dequant (11), arena (4), quantized matvec (4), SIMD kernels (6), chat templates (20), RWKV-X (12).
