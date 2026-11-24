---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags: [pantheon, pantheon-local, k80, gguf]
---

# Qwen3.6-35B-A3B — Pantheon K80 build (Q3_K_M)

The file Pantheon Local installs for **NVIDIA Tesla K80** boxes (one board, 2 GPUs).

- `Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf` — **unmodified** Q3_K_M GGUF by
  [bartowski](https://huggingface.co/bartowski) of
  [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team. Includes the
  model's MTP layers, which Pantheon uses for speculative decoding.

**License:** Apache License 2.0 (see `LICENSE`), the license of the original model:
https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE. Model © the Qwen team, Alibaba
Cloud. Quantization by bartowski. Redistributed unchanged.

Measured on Pantheon (greedy, MTP on): decode 24.5 / 19.7 / 13.6 tok/s at 8k / 14k / 32k context.
Pantheon: https://git.teamide.dev/David/Pantheon
