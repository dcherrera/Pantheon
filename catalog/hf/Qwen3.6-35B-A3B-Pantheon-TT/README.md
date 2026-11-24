---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE
base_model: Qwen/Qwen3.6-35B-A3B
tags: [pantheon, pantheon-local, tenstorrent, blackhole, p150]
---

# Qwen3.6-35B-A3B — Pantheon Tenstorrent build (Q8_0 source)

The files Pantheon Local installs for **Tenstorrent Blackhole p150** boxes (one card). They are
for Pantheon only: the `.fttw` files are Pantheon/Foundry's card tile format, not GGUF.

All three are made from bartowski's Q8_0 GGUF
([bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF),
`Qwen_Qwen3.6-35B-A3B-Q8_0.gguf`, sha256 `bc57de09…4434e9`) of
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team.

| file | bytes | what |
|---|---|---|
| `Qwen_Qwen3.6-35B-A3B-Q8_0.meta.gguf` | 15,972,512 | the GGUF's keys (tokenizer, config) and small tensors (norms, conv taps); the matmul weights and embedding are left out. **Modified** (subset). |
| `Qwen_Qwen3.6-35B-A3B-Q8_0.meta.moe_bfp6.fttw` | 28,843,656,936 | every matmul weight and the embedding in the card's tile format: Bfp8 tiles, routed experts Bfp6, expert routers fp32. **Modified** (format conversion; Bfp6 experts are a lossy step beyond Q8_0). |
| `Qwen_Qwen3.6-35B-A3B-Q8_0.meta.moe_bfp6.mtp.fttw` | 697,686,184 | the model's MTP layer, converted the same way, for Pantheon's speculative decoding. **Modified.** |

Measured on Pantheon, one p150, context cap 36,864, one user (greedy, plain decode): decode
12.2 / 11.4 / 9.8 tok/s at 8k / 14k / 32k context; time to first token 192 s for an 8,147-token
prompt. With MTP on: 18.2 tok/s on short prompts.

**License:** Apache License 2.0 (see `LICENSE`), the license of the original model:
https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE. Model © the Qwen team, Alibaba
Cloud. Q8_0 quantization by bartowski. These files are modified (format-converted) forms of the
original weights.

Pantheon: https://git.teamide.dev/David/Pantheon
