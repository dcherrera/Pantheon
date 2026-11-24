# Template parity fixtures (Qwen3.6 / Qwen3.8)

Read by `tests/test_template_parity_qwen.c` (in `make test`). Python is only needed to regenerate.

| File | Source |
|---|---|
| `qwen36_chat_template.jinja`, `qwen36_added_tokens.txt` | `models/qwen36-35b-a3b-mtp/` `chat_template.jinja` + `tokenizer.json` `added_tokens` |
| `qwen38_chat_template.jinja`, `qwen38_added_tokens.txt` | `models/qwen3.8-27b/Qwen3.8-27B-Q8_0.N8.gguf` header: `tokenizer.chat_template`, tokens with `token_type` 3 or 4 |
| `ornith15_chat_template.jinja`, `ornith15_added_tokens.txt` | `models/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf` header (reference only, no goldens: checked by hand for card 1.8 — same tool-result merge, `\|trim` and `<tool_response>` tokens as Qwen3.6) |
| `fixtures/*.json` | OpenAI wire request bodies, written by the render script |
| `goldens/<fixture>.<mode>.txt` | reference Jinja render (transformers' environment), `add_generation_prompt` |

Regenerate (from the repo root):

```
python3 tests/host/gguf_extract_chat_template.py models/qwen36-35b-a3b-mtp \
    --template tests/data/template_parity_qwen/qwen36_chat_template.jinja \
    --added-tokens tests/data/template_parity_qwen/qwen36_added_tokens.txt
python3 tests/host/gguf_extract_chat_template.py models/qwen3.8-27b/Qwen3.8-27B-Q8_0.N8.gguf \
    --template tests/data/template_parity_qwen/qwen38_chat_template.jinja \
    --added-tokens tests/data/template_parity_qwen/qwen38_added_tokens.txt
python3 tests/host/render_qwen_template_goldens.py
```

After a fixture or rendering change, the test prints the new pin row for `tests/template_parity_qwen_pins.h`.
