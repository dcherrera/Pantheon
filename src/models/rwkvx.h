/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rwkvx.h — RWKV-X adapter for Pantheon
 *
 * Wraps the RWKV-X inference engine (copied from foundry/rwkvx/rwkvx.c)
 * into Pantheon's pn_model/pn_state/pn_forward_fn interface.
 */

#ifndef PN_RWKVX_H
#define PN_RWKVX_H

#include "../core/engine.h"

/* Load RWKV-X safetensors model. Populates config from weight shapes.
 * If quantize=true, quantizes linear weights to INT8 at load time.
 * Returns opaque weights pointer (rx_model*), or NULL on failure. */
void* pn_rwkvx_load(const char *safetensors_path, pn_config *config, bool quantize);

/* Free RWKV-X weights. */
void pn_rwkvx_free(void *weights);

/* Forward pass conforming to pn_forward_fn signature.
 * Extracts internal rx_model/rx_state from pn_model/pn_state. */
tm_tensor* pn_rwkvx_forward(pn_model *model, uint32_t token, pn_state *state);

/* State lifecycle — creates/frees RWKV-X dual state (recurrent + KV cache + workspace). */
void* pn_rwkvx_state_new(void *weights);
void  pn_rwkvx_state_free(void *state);
void  pn_rwkvx_state_reset(void *state);

/* Tokenizer access (for engine.c encode/decode routing) */
int   pn_rwkvx_encode(void *weights, const char *text, uint32_t *out, int max);
char* pn_rwkvx_decode(void *weights, const uint32_t *tokens, int n);
int   pn_rwkvx_eos_id(void *weights);     /* returns 0 (<|endoftext|>) */
int   pn_rwkvx_vocab_size(void *weights);  /* returns 65536 */
int   pn_rwkvx_n_attn_layers(void *weights); /* 0 for pure RWKV-7, >0 for RWKV-X hybrid */

/* Load RWKV World tokenizer into the model (separate from weight loading).
 * Must be called after pn_rwkvx_load(). Returns 0 on success, -1 on failure. */
int pn_rwkvx_load_tokenizer(void *weights, const char *vocab_path);

#endif /* PN_RWKVX_H */
