# Chat templates of the other served families (reference, no goldens)

Read from the GGUF headers with `tests/host/gguf_extract_chat_template.py` for card 1.8, to list which
families' rendering the D3/D4/D5/D7 changes touch. Not read by any test.

| Key | GGUF | vs Qwen3.6 template |
|---|---|---|
| `saber`, `bonsai` | `Qwen3.6-35B-A3B-NSC-ACE-SABER-Q3_K_M.gguf`, `Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf` | byte-identical |
| `bonsai2` | `Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-Q2_0.gguf` | Qwen3.8-style (`reasoning_effort`) |
| `nex` | `Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf` | think block whenever reasoning is set, `<think>` prefill, scalar args `\|string` |
| `minicpm5` | `MiniCPM5-1B/MiniCPM5-1B-Q8_0.gguf` | own format; no `\|trim`; merges consecutive tool results |
