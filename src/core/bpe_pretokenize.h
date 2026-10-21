/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   bpe_pretokenize.h
 * @brief  Byte-level BPE encoding with a pre-token splitter (qwen35/GPT-2 and llama-bpe).
 *
 * Rewritten from a behaviour specification and measured behaviour;
 * the implementer did not see the previous code.
 *
 * The splitter only decides pre-token boundaries. Each pre-token's raw bytes are
 * GPT-2 byte-encoded and handed to ft_encode, and the per-piece IDs are concatenated.
 *
 * @status None
 * @issues - Unknown pre-type warning uses a process-wide flag that is not thread-safe.
 *         - The UTF-8 decoder does not validate continuation bytes; invalid bytes
 *           classify by their Latin-1 value.
 *         - Character classes are a fixed coarse range table, not the Unicode database.
 *         - `length` is written before any argument check (NULL `length` crashes).
 * @todo   None
 */
#ifndef PN_BPE_PRETOKENIZE_H
#define PN_BPE_PRETOKENIZE_H

#include "tokenizer/tokenizer.h"

/** Pre-tokenizer variants. Values are stored via ft_set_pre_type and must not change. */
typedef enum {
    PN_BPE_PRE_QWEN35 = 0,   /**< qwen35 / GPT-2 style split (single digits). */
    PN_BPE_PRE_LLAMA3 = 1,   /**< llama-bpe split (digit groups of up to 3). */
    PN_BPE_PRE_UNKNOWN = 2,  /**< Unrecognised name; encodes as qwen35 with a warning. */
} pn_bpe_pre;

/**
 * Map a `tokenizer.ggml.pre` string to a pn_bpe_pre value (exact, case-sensitive).
 * NULL, "qwen35", "qwen2", "gpt-2", "default" -> QWEN35;
 * "llama-bpe", "llama3", "llama-v3" -> LLAMA3; anything else -> UNKNOWN.
 */
int pn_bpe_pre_from_name(const char *pre);

/**
 * Encode `text` with the pre-tokenizer selected by ft_get_pre_type(tok)
 * (QWEN35 when tok is NULL). LLAMA3 -> pn_bpe_encode_llama3. UNKNOWN prints a
 * one-line warning to stderr on the first such call in the process, then encodes
 * as qwen35. Any other value encodes as qwen35 silently.
 */
int *pn_bpe_encode(ft_tokenizer *tok, const char *text, int *length);

/**
 * Encode `text` with the qwen35/GPT-2 split. `length` must be non-NULL; it is set to
 * 0 first and to the ID count on success. Returns a malloc'd ID array (caller frees);
 * a 1-int zeroed buffer when the result is empty or tok/text is NULL; NULL on failure.
 */
int *pn_bpe_encode_qwen35(ft_tokenizer *tok, const char *text, int *length);

/** As pn_bpe_encode_qwen35, with the llama-bpe split. */
int *pn_bpe_encode_llama3(ft_tokenizer *tok, const char *text, int *length);

#endif /* PN_BPE_PRETOKENIZE_H */
