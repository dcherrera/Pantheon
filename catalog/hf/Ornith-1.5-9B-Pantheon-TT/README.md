---
license: mit
base_model: ornith-ai/Ornith-1.5-9B
tags: [pantheon, pantheon-local, tenstorrent, blackhole, p150, gguf]
---

# Ornith 1.5-9B — Pantheon Tenstorrent build (Q8_0)

The files Pantheon Local installs for **Tenstorrent Blackhole p150** boxes.

- `Ornith-1.5-9B-Q8_0.gguf` — Q8_0 GGUF of
  [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (from
  `ornith-ai/Ornith-1.5-9B-GGUF`), unmodified.
- `Ornith-1.5-9B-Q8_0.bfp8.fttw` — the same weights converted by Pantheon/Foundry into the card's
  Bfp8 tile format, so the box loads it directly with no conversion step. **Modified** (format
  conversion) from the GGUF above.
- `Ornith-1.5-9B-Q8_0.bfp8.mtp.fttw` — a multi-token-prediction draft head for Pantheon's speculative
  decoding, **fine-tuned by TeamIDE** (`ft_all2`) from the model's MTP head. **Modified.**

**License:** MIT, as stated on the original model card
(https://huggingface.co/ornith-ai/Ornith-1.5-9B). The upstream repositories ship no LICENSE file
and name no copyright holder; the MIT terms apply to these files, including our modified ones.
Ornith is built on Qwen3.5 (Apache-2.0, © the Qwen team).

Measured on Pantheon, one p150 (greedy, MTP on): decode 40.8 tok/s at 7.4k context, 37.9 at 14.6k.
Pantheon: https://git.teamide.dev/David/Pantheon
