# Clean-room rewrite record

Provenance record for the Pantheon files rewritten from behaviour specifications so that no
ggml or llama.cpp code remains in them. Commits are in Pantheon's history.

## Process

- **Roles.** A *spec writer* (the only party allowed to read the old file) wrote a behaviour
  spec per unit. A separate *implementer* deleted the old file and wrote a new one at the same
  path from the spec alone.
- **Specs** are prose, tables and maths: no copied code or comments. Interface facts that
  callers or file formats depend on (names, signatures, struct and byte layouts, constants,
  error text) are stated verbatim. Odd existing behaviour is kept and marked as-is.
- **Implementers** never read the old file, its history, ggml, llama.cpp or any copy of them,
  nor any other file holding a close paraphrase of the old code.
- **Gate: output identity.** Each rewrite had to reproduce the old code bit for bit (float
  bit patterns, return codes, error text, call sequence) under every build configuration it
  ships in. Each gate was proven able to fail with deliberate mutations before it was trusted.
- Each rewritten file's header names its spec.

## Units

| Round | Date | Files | Spec | Commit |
|---|---|---|---|---|
| 1 | 2026-10-09 | `src/core/bpe_pretokenize.{h,c}`, `src/formats/gguf_dequant.{h,c}` | `cleanroom_bpe_dequant_spec.md` | `7159072` |
| 2 | 2026-10-09 | `src/accel/simd.{h,c}` | `cleanroom_r2_simd_spec.md` | `c5e13d4` |
| 2 | 2026-10-09 | `src/core/qweight.{h,c}` | `cleanroom_r2_qweight_spec.md` | `c5bf531` |
| 2 | 2026-10-09 | `src/models/qwen35moe_ssm.c`, `qwen35moe_attn.c`, `qwen35moe_kv_q4.h` | `cleanroom_r2_qwen35_spec.md` | `0e20f72` |
| 2 | 2026-10-12 | `src/models/pixtral_vision.{h,c}` | `cleanroom_r2_pixtral_spec.md` | `5666039` |

The specs and the equivalence gates are kept in the private repository; they reference the
author's machines and the frozen old code, so they are not part of the public release.
