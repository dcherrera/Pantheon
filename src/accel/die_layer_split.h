/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file die_layer_split.h
 * @brief How a model's transformer layers divide across an N-die placement.
 *        ONE definition of the layer ranges, shared by the pre-load admission
 *        projection and the GPU forward that actually uploads them.
 *
 * The two used to derive the ranges separately — model_registry.c projected
 * `n_layer*idx/n_placed` per die while qwen35moe_forward_cuda.c split at
 * `n_layer/2` regardless of how many dies were placed. They agreed only at
 * n_placed == 2, so a `--dies 5,6,7` placement bound three dies, uploaded a
 * two-way split, under-projected each die's share by a third, and OOM'd. A
 * projection that disagrees with placement is the bug class this module exists
 * to make unrepresentable: admission and the forward call the SAME function.
 *
 * Quant-agnostic by construction: this module partitions LAYER INDICES. It
 * never sees a tensor, a byte count or a quantization type, so the same split
 * carries to Q3_K, Q4_K, Q6_K and F16 unchanged.
 *
 * @status Complete. Ranges, the N=2 compatibility guarantee and the refusal
 *         boundaries are covered by tests/test_die_layer_split.c.
 * @issues None
 * @todo None
 */

#ifndef PN_DIE_LAYER_SPLIT_H
#define PN_DIE_LAYER_SPLIT_H

/* Widest layer split this partitioner will plan. The die manager and the
 * memwatch state carry 16 dies (PN_MEM_MAX_DIES), and PN_MODEL_MAX_DIES caps
 * what one model may be PLACED on; this is the split planner's own ceiling and
 * is deliberately the same number, so "the placement fit" and "the split fit"
 * cannot disagree. Widths above it are refused, never silently narrowed. */
#define PN_DIE_LAYER_SPLIT_MAX_DIES 16

/* A planned N-way layer split. Die i owns global layers [layer_lo[i],
 * layer_hi[i]); the ranges are contiguous, non-empty, cover [0, n_layer)
 * exactly, and differ in length by at most one. The LM head and final norm
 * ride on die n_die-1 (the last), which is a convention this header states
 * rather than a field: nothing here can enforce what the uploader does with
 * the head, so both sides name the same rule. */
typedef struct {
    int n_die;
    int layer_lo[PN_DIE_LAYER_SPLIT_MAX_DIES];
    int layer_hi[PN_DIE_LAYER_SPLIT_MAX_DIES];
} pn_die_layer_split;

/**
 * Plans how `n_layer` layers divide across `n_die` dies.
 *
 * The boundary for die i is `n_layer * i / n_die` in integer arithmetic, which
 * is the flattest split available and — the property the 2-die path depends on
 * — reduces to exactly `n_layer / 2` at n_die == 2, the boundary the CUDA
 * forward has always used. So a 2-die placement plans byte-identically to the
 * hardcoded pair it replaces, for even and odd layer counts alike. Any
 * remainder lands on the LATER dies, which is what makes that identity hold
 * (41 layers over 2 dies is 20 + 21, not 21 + 20).
 *
 * @param n_layer            Total transformer layers (> 0)
 * @param n_die              Dies to split across (1 .. PN_DIE_LAYER_SPLIT_MAX_DIES)
 * @param override_boundary  Manual first boundary, or <= 0 for none. Honoured
 *                           ONLY at n_die == 2 (where a single boundary is the
 *                           whole plan); IGNORED for every other width, because
 *                           one number cannot describe N-1 boundaries. Clamped
 *                           to [1, n_layer-1] so it can never empty a die.
 * @param out                Receives the plan. Untouched on refusal.
 * @return 0 on success, -1 if refused — n_layer <= 0, n_die < 1, n_die above
 *         PN_DIE_LAYER_SPLIT_MAX_DIES, or n_die > n_layer (a die with no layers
 *         is not a narrower split, it is a bug). Callers must REFUSE on -1
 *         rather than fall back to a narrower split: mis-splitting silently is
 *         the failure this replaces.
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 */
int pn_plan_die_layer_split(int n_layer, int n_die, int override_boundary,
                            pn_die_layer_split *out);

/**
 * The layer range for ONE die of an N-way split, without materializing the
 * whole plan — the shape admission wants when it is pricing die `idx` alone.
 * Identical arithmetic to pn_plan_die_layer_split; it is the same function
 * viewed through one index.
 *
 * @param n_layer  Total transformer layers (> 0)
 * @param n_die    Dies to split across (1 .. PN_DIE_LAYER_SPLIT_MAX_DIES)
 * @param idx      Which die (0 .. n_die-1)
 * @param lo_out   Receives the first global layer on this die (may be NULL)
 * @param hi_out   Receives one past the last (may be NULL)
 * @return 0 on success, -1 on the same refusals as pn_plan_die_layer_split
 *         (plus an out-of-range idx). Outputs untouched on refusal.
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 */
int pn_die_layer_range(int n_layer, int n_die, int idx, int *lo_out,
                       int *hi_out);

#endif /* PN_DIE_LAYER_SPLIT_H */
