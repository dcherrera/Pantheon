/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_metal_kernel_registry.h
 * @brief Which quant types the Metal backend currently has a matvec kernel for.
 *        Answers the question by looking the foundry entry point up in the
 *        running binary's symbol table, so the answer is linked evidence rather
 *        than a claim maintained by hand.
 *
 * @status Phase 1 of foundry/metal/qwen35moe_metal_plan.md. Q4_K and F32 have
 *         entry points today; Q3_K / Q5_K / Q6_K / Q8_0 do not.
 * @issues None
 * @todo None
 */

#ifndef PN_QWEN35MOE_METAL_KERNEL_REGISTRY_H
#define PN_QWEN35MOE_METAL_KERNEL_REGISTRY_H

#include "../formats/gguf.h"

/**
 * Resolve the Metal matvec entry point for a quant type.
 *
 * A type is "covered" when foundry exports an entry point that dispatches a
 * matvec for it. That is tested by symbol lookup in the running image, not by a
 * flag in a table here, for three reasons:
 *
 *   1. It cannot claim a kernel that is not linked. A hand-maintained boolean
 *      can, and the failure it hides — a weight fed to the wrong kernel — is
 *      silent garbage logits, which is the whole reason this gate exists.
 *   2. It flips on its own the day a kernel lands, with no edit here.
 *   3. In a build without USE_METAL nothing resolves, so coverage reports a
 *      truthful 0% instead of crashing.
 *
 * @param type       GGUF quant type to look up
 * @param name_out   Receives the resolved symbol name (NULL when uncovered),
 *                   or NULL if the caller does not want it. Static storage.
 * @return           1 if a kernel is registered for `type`, else 0
 *
 * Thread-safe: yes (read-only lookup)
 * Reentrant: yes
 * Signal-safe: no (dlsym)
 */
int pn_qwen35moe_metal_kernel_for_type(pn_gguf_quant type,
                                       const char **name_out);

#endif /* PN_QWEN35MOE_METAL_KERNEL_REGISTRY_H */
