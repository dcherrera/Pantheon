# Security Policy

## Reporting a Vulnerability

**Do not open a public issue for a security vulnerability.**

Email **coreyh55@gmail.com** with:

- a description of the vulnerability
- steps to reproduce
- potential impact
- a suggested fix, if you have one

You will get an acknowledgement within 48 hours. Critical issues are aimed at a
fix within 7 days. If you want credit in the fix commit, say so and give me the
name to use.

## Scope

Pantheon runs untrusted model files and serves untrusted network input, so the
security-relevant surface is larger than the codebase's size suggests:

- **`src/formats/`** — GGUF parsing, quantized-tensor dequantization, and
  memory-mapped I/O. A malformed model file reaching a bad offset or a bogus
  tensor shape is the highest-value bug class here.
- **`src/core/`** — tokenizer and BPE pre-tokenization; chat-template parsing.
  Both consume attacker-controlled text.
- **`src/serve/`** — the HTTP/SSE server: request parsing, JSON handling,
  session and slot management, sampling parameter bounds.
- **`src/memory/`** — KV cache and session state, including the paths that
  export and restore session blobs.

Out of scope: the behavior of the models themselves. Prompt injection,
jailbreaks, and model output content are properties of the weights you load, not
of this engine.

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

Older tags are not patched. Fixes land on `main`.

## Best Practices

When contributing to Pantheon:

- `snprintf`, never `sprintf`
- validate every offset and length read out of a model file before using it —
  treat the file as hostile input, not as data you wrote
- check every `malloc`/`calloc` return for NULL
- bounds-check array indices derived from file or network data
- watch for integer overflow in size calculations, especially
  `count * element_size` on values taken from a header
- no `exit()` in library code — return an error and let the caller decide
