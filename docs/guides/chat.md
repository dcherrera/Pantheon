# Chat Templates & Interactive Chat

Pantheon v0.3.0 adds multi-turn interactive chat with automatic template detection. Each model family uses a different chat template — Pantheon detects the right one from GGUF metadata and formats messages with correct special tokens.

## Quick Start

```bash
./pantheon --model ../models/base/SmolLM2-135M-Instruct-Q8_0.gguf \
  --chat --system "You are a helpful assistant."
```

```
Chat mode (ChatML). Type /quit to exit, /clear to reset.
> What is the capital of France?
The capital of France is Paris.
> And Germany?
The capital of Germany is Berlin.
> /quit
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--chat` | Enter interactive chat mode |
| `--system <text>` | System prompt (optional, chat mode only) |
| `--max-tokens <n>` | Max tokens per response (default: 128) |

Chat mode commands:

| Command | Action |
|---------|--------|
| `/quit` or `/exit` | Exit chat |
| `/clear` | Reset conversation (clears KV cache, starts fresh) |

## Supported Templates

| Style | Models | System Prompt |
|-------|--------|---------------|
| ChatML | Qwen2.5, SmolLM2 | Yes |
| Mistral | Mistral v0.x, Mixtral | No |
| Phi-3 | Phi-3-mini | Yes |

Detection is automatic — Pantheon reads the `tokenizer.chat_template` Jinja2 string from GGUF metadata and matches on key markers. Falls back to architecture-based detection if the template is missing.

## How It Works

### The Special Token Problem

BPE models (Qwen, SmolLM2) have "added tokens" like `<|im_start|>` that exist in the vocabulary but are **not produced by BPE merge rules**. Passing `"<|im_start|>user\nHello"` through the tokenizer splits `<|im_start|>` into individual characters — wrong.

Pantheon solves this by splitting formatted messages into **special** and **text** segments internally. Special tokens are looked up by exact string via `ft_token_to_id()`, text segments are encoded normally. The result is a correct `uint32_t[]` token array.

SentencePiece models (Mistral, Phi-3) don't have this problem — their Viterbi algorithm matches special tokens from the vocab naturally.

### Multi-Turn Flow

1. **Turn 0:** Format `system + user + assistant_prefix` → prefill → generate until EOS
2. **EOS sampled** but not fed through forward pass (existing behavior)
3. **Turn 1+:** Format `EOT + user + assistant_prefix` → prefill from current position → generate until EOS
4. **KV cache persists** across turns — no recomputation of earlier turns

### Template Formats

**ChatML** (Qwen2.5, SmolLM2):
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
<|im_start|>user
{next_user}<|im_end|>
<|im_start|>assistant
```

**Mistral** (Mistral, Mixtral):
```
[BOS][INST] {user} [/INST]{response}</s> [INST] {next_user} [/INST]
```

**Phi-3**:
```
<|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
{response}<|end|>
<|user|>
{next_user}<|end|>
<|assistant|>
```

## C API

### `pn_stream_tokens()` — Pre-tokenized Streaming

```c
int pn_stream_tokens(pn_model *model, pn_state *state,
                     const uint32_t *tokens, int n_tokens,
                     int max_tokens,
                     pn_sampling_params *params,
                     pn_token_callback callback, void *user_data);
```

Like `pn_stream()` but takes pre-tokenized input. No BOS added — caller controls tokenization entirely. Returns number of tokens generated, or -1 on error.

### Chat Template Functions (`src/core/chat.h`)

```c
/* Detect style from Jinja2 template string */
pn_chat_style pn_chat_detect_style(const char *jinja_template);

/* Fallback: detect from architecture enum */
pn_chat_style pn_chat_style_from_arch(int arch);

/* Format first turn (system + user + assistant prefix) */
pn_token_seq pn_chat_format_first(pn_chat_style style, ft_tokenizer *tok,
                                   const char *system, const char *user);

/* Format continuation turn (EOT + user + assistant prefix) */
pn_token_seq pn_chat_format_turn(pn_chat_style style, ft_tokenizer *tok,
                                  const char *user);
```

### Embedding Chat in Your App

```c
#include "pantheon.h"
#include "src/core/engine.h"
#include "src/core/chat.h"

pn_model *model = pn_load("model.gguf");
pn_state *state = pn_state_new(model);
pn_sampling_params params = pn_sampling_defaults();

pn_chat_style style = (pn_chat_style)model->chat_style;
ft_tokenizer *tok = model->tokenizer;

/* First turn */
pn_token_seq seq = pn_chat_format_first(style, tok, "You are helpful.", "Hello");
pn_stream_tokens(model, state, seq.tokens, seq.n_tokens, 128, &params, on_token, NULL);
free(seq.tokens);

/* Second turn — KV cache preserved automatically */
seq = pn_chat_format_turn(style, tok, "Tell me more.");
pn_stream_tokens(model, state, seq.tokens, seq.n_tokens, 128, &params, on_token, NULL);
free(seq.tokens);

/* Reset for new conversation */
pn_state_reset(state);
```

## Tests

20 tests in `tests/test_chat.c` covering style detection (5), architecture mapping (6), style names (1), ChatML formatting (3), Phi-3 formatting (2), Mistral formatting (2), and passthrough (1).

```bash
make USE_BLAS=1 test   # runs all 50 tests including chat
```
