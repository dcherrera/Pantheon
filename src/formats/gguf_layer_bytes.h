/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_GGUF_LAYER_BYTES_H
#define PN_GGUF_LAYER_BYTES_H

#include "gguf.h"
#include <stdint.h>

/*
 * Per-layer weight accounting straight off the GGUF tensor table.
 *
 * A multi-die split gives each die a *range* of layers, so the admission gate
 * needs "what do layers [lo,hi) actually weigh", not "what fraction of the file
 * is hi-lo layers". The two are not the same number: the file also carries
 * token_embd and output.weight, which belong to no layer, so prorating
 * mmap_size across n_layer charges every die for tensors it does not hold. On
 * Qwen3.6-35B-A3B that is ~700 MiB of phantom weight per die — enough to refuse
 * a load that fits with 1.3 GiB to spare.
 */

/* File bytes of one named tensor. 0 if absent, malformed, or an unsupported
 * quant (callers treat 0 as "don't know", not "free"). */
uint64_t pn_gguf_tensor_file_bytes(pn_gguf_file *gf, const char *name);

/* Summed file bytes of every `blk.<i>.*` tensor with lo <= i < hi. Returns 0
 * when the table holds no matching tensor, which callers use to fall back to
 * an older estimate rather than admit blind. */
uint64_t pn_gguf_layer_range_bytes(pn_gguf_file *gf, int lo, int hi);

/* Highest `blk.<i>.` index in the table, + 1. Read off the tensors rather than
 * `<arch>.block_count` so it needs no architecture prefix and cannot disagree
 * with the tensors a layer range will actually sum. 0 if there are none. */
int pn_gguf_layer_count(pn_gguf_file *gf);

/* Element count of one named tensor, 0 if absent or malformed. Callers that
 * must size a DEQUANTIZED copy need this rather than the file bytes — a tied
 * model's LM head is token_embd materialized to F32, four bytes an element. */
uint64_t pn_gguf_tensor_numel(pn_gguf_file *gf, const char *name);

#endif /* PN_GGUF_LAYER_BYTES_H */
