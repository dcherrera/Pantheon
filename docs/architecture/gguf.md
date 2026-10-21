# GGUF Format Support

Pantheon's GGUF parser (`src/formats/gguf.c`) provides zero-copy model loading via mmap. Both GGUF v2 and v3 are supported.

## File Layout

```
┌──────────────────────────┐
│  Magic: "GGUF" (4 bytes) │
│  Version: uint32         │  v2 or v3
│  Tensor count: uint64    │
│  Metadata count: uint64  │
├──────────────────────────┤
│  Metadata entries         │  typed key-value pairs
│  (variable length)        │
├──────────────────────────┤
│  Tensor info entries      │  name, dims, quant type, offset
│  (variable length)        │
├──────────────────────────┤
│  Padding                  │  aligned to 32 bytes (GGUF_DEFAULT_ALIGNMENT)
├──────────────────────────┤
│  Tensor data (binary)     │  mmap'd, accessed via offset
└──────────────────────────┘
```

### Alignment

The tensor data section starts at a 32-byte aligned offset from the file start. This is the GGUF spec default (`GGUF_DEFAULT_ALIGNMENT = 32`). Files can override this via the `general.alignment` metadata key.

### v2 vs v3

Both versions use the same binary layout. The parser accepts either. v3 added big-endian support (Pantheon assumes little-endian). The version field is stored in `pn_gguf_file.version`.

## Metadata Types

13 value types supported:

| Type | ID | Size |
|------|----|------|
| UINT8 | 0 | 1 |
| INT8 | 1 | 1 |
| UINT16 | 2 | 2 |
| INT16 | 3 | 2 |
| UINT32 | 4 | 4 |
| INT32 | 5 | 4 |
| FLOAT32 | 6 | 4 |
| BOOL | 7 | 1 |
| STRING | 8 | 8 + len |
| ARRAY | 9 | variable |
| UINT64 | 10 | 8 |
| INT64 | 11 | 8 |
| FLOAT64 | 12 | 8 |

Strings are length-prefixed (uint64 length + raw bytes). Arrays contain a type tag + uint64 count + elements.

## Quantization Types

### Implemented (full dequant + fused matvec)

| Type | ID | Bits/Weight | Block Size | Bytes/Block | Notes |
|------|----|-------------|------------|-------------|-------|
| F32 | 0 | 32 | 1 | 4 | Passthrough |
| F16 | 1 | 16 | 1 | 2 | IEEE 754 half |
| Q4_0 | 2 | 4 | 32 | 18 | scale + 4-bit nibbles |
| Q8_0 | 8 | 8 | 32 | 34 | scale + int8 values |
| Q4_K | 12 | 4.5 | 256 | 144 | 8 sub-blocks, 6-bit scales |
| Q5_K | 13 | 5.5 | 256 | 176 | 5-bit values (4 low + 1 high) |
| Q6_K | 14 | 6.5 | 256 | 210 | 6-bit values (4 low + 2 high) |

### Format-only (defined but no dequant kernel)

Q4_1, Q5_0, Q5_1, Q8_1, Q2_K, Q3_K, Q8_K

### Mixed Quantization (Q4_K_M, Q5_K_M)

GGUF files labeled "Q4_K_M" use Q4_K for most layers with Q6_K for attention output and FFN down projections. This is handled transparently — each tensor has its own quant type, and `pn_qmv` dispatches per-row based on the weight's type field.

## Key Metadata Keys

### Model Architecture

| Key | Type | Description |
|-----|------|-------------|
| `general.architecture` | string | `"llama"`, `"qwen2"`, `"phi"`, `"mistral"` |
| `general.name` | string | Human-readable model name |
| `general.alignment` | uint32 | Data section alignment (default 32) |

### Model Dimensions

Prefixed by architecture name (`llama.*`, `qwen2.*`, etc.):

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.embedding_length` | uint32 | Hidden size (n_embd) |
| `{arch}.attention.head_count` | uint32 | Query heads |
| `{arch}.attention.head_count_kv` | uint32 | KV heads (for GQA) |
| `{arch}.block_count` | uint32 | Number of layers |
| `{arch}.feed_forward_length` | uint32 | FFN intermediate size |
| `{arch}.context_length` | uint32 | Max context length |
| `{arch}.rope.freq_base` | float32 | RoPE theta (default 10000) |
| `{arch}.attention.layer_norm_rms_epsilon` | float32 | RMSNorm epsilon |
| `{arch}.attention.sliding_window` | uint32 | Sliding window size (Mistral) |

### Tokenizer

| Key | Type | Description |
|-----|------|-------------|
| `tokenizer.ggml.model` | string | `"gpt2"` (BPE) or `"llama"` (SentencePiece) |
| `tokenizer.ggml.tokens` | string[] | Vocabulary tokens |
| `tokenizer.ggml.merges` | string[] | BPE merge rules (BPE only) |
| `tokenizer.ggml.scores` | float[] | Unigram scores (SentencePiece only) |
| `tokenizer.ggml.bos_token_id` | uint32 | BOS token ID |
| `tokenizer.ggml.eos_token_id` | uint32 | EOS token ID |

## Tensor Naming Convention

GGUF tensors follow the pattern `blk.{layer}.{component}.weight`:

```
token_embd.weight                    embedding table
blk.0.attn_norm.weight               pre-attention RMSNorm
blk.0.attn_q.weight                  query projection
blk.0.attn_k.weight                  key projection
blk.0.attn_v.weight                  value projection
blk.0.attn_output.weight             output projection
blk.0.attn_q.bias                    query bias (Qwen2 only)
blk.0.ffn_norm.weight                pre-FFN RMSNorm
blk.0.ffn_gate.weight                SwiGLU gate projection
blk.0.ffn_up.weight                  SwiGLU up projection
blk.0.ffn_down.weight                down projection
output_norm.weight                   final RMSNorm
output.weight                        LM head (may be tied to token_embd)
```

## Quantized Weight Loading

Weights are loaded as `pn_qweight` structs that point directly into mmap'd data:

```c
typedef struct {
    const void   *data;       /* raw quantized blocks in mmap */
    pn_gguf_quant type;       /* quant format */
    int           rows;       /* output dimension */
    int           cols;       /* input dimension */
    size_t        row_stride; /* bytes per row */
} pn_qweight;
```

GGUF stores dimensions innermost-first. The loader reverses them to C outermost-first: `dims[n_dims-1]` = rows, `dims[0]` = cols. `row_stride` is computed from cols and the quant format's bytes-per-block.

Matrix-vector multiply (`pn_qmv`) iterates rows, dequantizing one block (32 or 256 elements) at a time and accumulating the dot product. No full dequantization ever occurs for projection weights.
