/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_topk_tail.h
 * @brief TAIL COUNTS on the card: one top-k pass over a logit row that also counts every
 *        logit into bins the host's threshold keys define (foundry
 *        kernels/resident_topk_protocol.h §6), so a sampler can bound the mass outside its
 *        top-N without reading the 248k-wide row over the link.
 *
 * @status Used by pn_qwen35moe_tt_tail_counts (the sampler's pn_sampling_params.tail_counts).
 * @issues None
 * @todo None
 */
#ifndef PN_TT_TOPK_TAIL_H
#define PN_TT_TOPK_TAIL_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** Most threshold keys one pass takes (FT_TT_OP_SAMPLE_TAIL_MAX_BINS). */
#define PN_TT_TAIL_MAX_BINS 32u

/**
 * Count logits row `row` of `logits` into `bins` bins: counts[j] = logits whose bf16 order key
 * lies in [keys[j], keys[j-1]) (keys[-1] = +inf), counts[bins] = below keys[bins-1]. Runs one
 * top-k descriptor per split member with the tail flag (its pairs are discarded).
 *
 * @param keys    strictly DESCENDING 16-bit order keys
 * @param counts  bins + 1 entries
 * @return 0, or -1 (nothing counted: bad keys, a refused or aborted member)
 *
 * Thread-safe: no    Reentrant: no
 */
int pn_tt_topk_tail_counts(pn_tt_device *device, const pn_tt_activation *logits, uint32_t row,
                           const uint32_t *keys, uint32_t bins, uint32_t *counts);

/* pn_tt_oncard.c's top-k path, per member: write the keys into the result block and set the
 * request's flag / bin count (before the enqueue); then read back and add its counts. */
struct ft_tt_op_topk_request;
bool pn_tt_topk_tail_prepare(pn_tt_device *device, uint32_t x, uint32_t y,
                             uint32_t result_address, struct ft_tt_op_topk_request *request);
void pn_tt_topk_tail_collect(pn_tt_device *device, uint32_t x, uint32_t y,
                             uint32_t result_address);

#endif /* USE_TT */
#endif /* PN_TT_TOPK_TAIL_H */
