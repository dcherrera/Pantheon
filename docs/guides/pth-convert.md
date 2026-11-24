# PyTorch .pth Converter

Convert PyTorch checkpoint files (`.pth`) to safetensors format in pure C, using Foundry's pth and safetensors modules. No Python, no pip.

## Why?

PyTorch's `.pth` format uses Python's `pickle` module, which is fundamentally insecure — `pickle.load()` can execute arbitrary code. Every existing conversion tool requires Python + PyTorch + transformers (2-10 GB of dependencies).

Pantheon reads `.pth` files via a sandboxed C pickle interpreter that extracts tensor data without executing any embedded Python. It then writes safetensors, a secure mmap-friendly format.

## Build

```bash
cd Pantheon
make USE_BLAS=1 pantheon-convert
```

Or from Foundry (standalone, no Pantheon needed):

```bash
cd foundry
make pth-convert
```

## Usage

### Convert

```bash
# Convert .pth to safetensors
./pantheon-convert model.pth -o model.safetensors
```

### Inspect

```bash
# Print tensor table (names, shapes, dtypes, sizes)
./pantheon-convert model.pth --inspect
```

Output:

```
=== PyTorch Tensor Inspection ===
File: RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
Tensors: 399

  #  Name                              DType   Shape                      Size
---  --------------------------------  ------  -------------------------  --------
  0  emb.weight                        BF16    [65536, 768]               96.0 MB
  1  blocks.0.att.x_r                  BF16    [768]                       1.5 KB
  2  blocks.0.att.x_w                  BF16    [768]                       1.5 KB
  ...
```

## Example: RWKV-7 0.1B

```bash
# 1. Download the .pth model from HuggingFace
huggingface-cli download BlinkDL/rwkv-7-world \
  RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  --local-dir models/base/

# 2. Convert to safetensors
./pantheon-convert \
  models/base/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
  -o models/base/RWKV-7-0.1B.safetensors

# 3. Run inference
./pantheon --model models/base/RWKV-7-0.1B.safetensors \
  --vocab foundry/rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello" --max-tokens 32 --temperature 0
```

## How It Works

A `.pth` file is a ZIP archive containing:

```
archive/
  data.pkl          # Pickle: OrderedDict mapping tensor names -> storage refs
  data/0            # Raw bytes for storage #0
  data/1            # Raw bytes for storage #1
  ...
```

The converter:

1. **ZIP reader** — mmaps the file, finds End-of-Central-Directory, parses entries
2. **Pickle interpreter** — stack-based opcode VM that extracts tensor metadata (names, shapes, dtypes, storage references). Security-whitelisted: only allows `torch._utils._rebuild_tensor_v2`, `collections.OrderedDict`, and torch storage classes. Unknown `GLOBAL`/`REDUCE` targets abort immediately.
3. **Storage resolver** — maps storage IDs to ZIP data entries, computes data pointers (zero-copy into mmap)
4. **Safetensors writer** — writes header JSON + aligned tensor data via Foundry's `fs_writer_*` API

## Supported Dtypes

| PyTorch Storage Class | Safetensors Dtype | Bits |
|----------------------|-------------------|------|
| `BFloat16Storage` | BF16 | 16 |
| `HalfStorage` | F16 | 16 |
| `FloatStorage` | F32 | 32 |

## Security

The pickle interpreter is deliberately minimal:

- **No code execution** — only extracts tensor metadata and raw data
- **Whitelisted classes** — `GLOBAL` only allows known torch/collections classes
- **Whitelisted callables** — `REDUCE` only allows `_rebuild_tensor_v2` and `OrderedDict`
- **Immediate abort** — any unrecognized opcode or unauthorized class triggers abort with warning

This is **not** a general-purpose pickle reader. It handles the specific pickle structure that `torch.save()` produces for state dicts. Adversarial pickles with embedded code will be rejected.

## C API

The pth reader is a reusable Foundry module (`pt_` prefix):

```c
#include "pth/pth.h"

pt_file *f = pt_open("model.pth");
int n = pt_tensor_count(f);

for (int i = 0; i < n; i++) {
    const pt_tensor *t = pt_tensor_at(f, i);
    printf("%s: %s [", t->name, fs_dtype_name(t->dtype));
    for (int d = 0; d < t->ndim; d++)
        printf("%s%lld", d ? ", " : "", (long long)t->shape[d]);
    printf("] %zu bytes\n", t->data_size);
    // t->data points directly into mmap'd storage (zero-copy)
}

pt_close(f);
```
