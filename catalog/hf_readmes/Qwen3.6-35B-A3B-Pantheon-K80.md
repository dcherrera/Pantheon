---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags: [pantheon, pantheon-local, k80, gguf]
---

# Qwen3.6-35B-A3B — Pantheon K80 builds

The files Pantheon Local installs for **NVIDIA Tesla K80** boxes.

| file | quant | hardware | notes |
|---|---|---|---|
| `Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf` | Q3_K_M | one board (2 GPUs) | **unmodified** GGUF by [bartowski](https://huggingface.co/bartowski) |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | Q8_0 | two boards (4 GPUs) | our quantization, see below |

Both carry the model's MTP layer, which Pantheon uses for speculative decoding.

## Q8_0 (`Qwen3.6-35B-A3B-Q8_0.gguf`)

- 37,801,097,248 bytes, sha256 `0bd4666292101b41b1965008814f0948c1a6a7e0bd7187e7a12ce6dd1d16212e`.
- Our own Q8_0 quantization (llama.cpp `llama-quantize`) of the BF16 weights of
  [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Every 2-D weight is Q8_0;
  the expert routers and 1-D tensors stay BF16/F32.
- The BF16 conversion had no MTP head, so the MTP layer (`blk.40.*`, Q8_0) is taken unchanged
  from [bartowski's GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) and grafted
  on; every tensor was re-read and compared byte for byte with its source (753/753).
- Measured on Pantheon, 4 K80 GPUs on two boards, MTP on, one user (greedy / temperature 0.7):
  decode **21.3 / 19.1** tok/s at 8k context, **22.0 / 20.9** at 14k, **13.8 / 13.4** at 32k;
  time to first token 51 s for an 8,444-token prompt. Greedy output matches the CPU reference.

## Q3_K_M (`Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf`)

Redistributed unchanged from bartowski. Measured on Pantheon (greedy, MTP on, one board):
decode 24.5 / 19.7 / 13.6 tok/s at 8k / 14k / 32k context.

## License

Apache License 2.0 (see `LICENSE`), the license of the original model:
https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE. Model © the Qwen team, Alibaba
Cloud. Q3_K_M quantization and the Q8_0 file's MTP layer by bartowski. The Q8_0 file is a
modified (quantized) form of the original weights.

Pantheon: https://git.teamide.dev/David/Pantheon
