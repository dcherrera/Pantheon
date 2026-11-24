/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_routing_dump.h
 * @brief  `QWEN35_ROUTING_DUMP=<file>`: append every routed-MoE call's top-k selection to a
 *         binary file — the oracle side of the card-vs-oracle routing-agreement gate
 *         and the router-precision measurement.
 *
 * One record a call, in call order (a decode token runs layers 0..L-1 in order, so the
 * reader recovers the layer as call % n_layer):
 *   int32 idx[k]     selected experts, rank order (ties to the lowest index)
 *   float prob[k]    their softmax probabilities (before renormalisation)
 *   float next_prob  the best UNSELECTED probability: prob[k-1] - next_prob is the
 *                    selection margin, ~0 at a near-tie
 * k is the model's n_expert_used (8 for Qwen3.6), so a record is 8k + 4 bytes.
 *
 * @status None
 * @issues Single-threaded writer; the CPU MoE block runs one token at a time.
 * @todo   None
 */
#ifndef PN_QWEN35MOE_ROUTING_DUMP_H
#define PN_QWEN35MOE_ROUTING_DUMP_H

/**
 * Append one record when QWEN35_ROUTING_DUMP is set; a no-op otherwise.
 *
 * @param probs    the n_expert softmax probabilities the selection was made on
 * @param top_idx  the k selected experts
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_qwen35moe_routing_dump(const float *probs, int n_expert, const int *top_idx, int k);

#endif /* PN_QWEN35MOE_ROUTING_DUMP_H */
