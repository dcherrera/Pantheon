/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_mask_shortlist.h
 * @brief Apply a grammar mask by judging a SHORTLIST of the highest logits
 *        instead of the whole vocabulary, when that provably decides the draw.
 *
 * The plain grammar mask asks `allowed()` of every one of the ~248k vocab
 * entries, twice (count, then mask) — on a slow host that is the dominant cost
 * of a constrained token.  The draw, though, only ever looks at the top_k
 * survivors (greedy: the argmax).  Penalties only LOWER a logit (presence and
 * frequency subtract, a repeat penalty >= 1 divides a positive and multiplies a
 * negative), so a token outside the shortlist can never rise above `bound` —
 * the largest raw value left out of it.  Hence: if at least K shortlisted tokens
 * are ALLOWED and keep a penalised value strictly above `bound` (with a small
 * margin for the replica's rounding), the top-K of the masked row lies inside
 * the shortlist, and masking everything else changes nothing the draw can see.
 *
 * When it decides, the row is rewritten in place: allowed shortlisted tokens
 * keep their logit, every other entry becomes -INFINITY — a sparse row the
 * sampler's survivor fast path takes.  When it cannot decide (a logit_bias, a
 * penalty that raises, top_k off or above the shortlist, too few allowed
 * tokens) the row is left untouched and the caller runs the full mask.
 *
 * Shortlists: a backend's exact top-N row (every unlisted logit <= its smallest
 * entry) or the top-GMS_MAX entries selected from a dense row.
 *
 * THE OUTSIDE BOUND (gms_sampling.outside, NULL = off).  Fewer than K allowed
 * tokens clear `bound` -- common on a structural step, where a handful of tokens
 * are legal -- so the masked top-K reaches outside the shortlist.  Every allowed
 * token outside has raw logit <= `bound`, so with m the largest penalised
 * allowed value and T the temperature their total mass is at most
 *     B = sum over the outside allowed of exp((logit - m) / T)
 * charged as gms_outside says: `count` tokens at `bound`, or (with the device's
 * tail counts of the raw row) the `count` tokens in the highest bins first.  The
 * allowed shortlisted tokens are then the survivors of sampler_topn_tail.h's
 * planner with that tail: its NUCLEUS verdict proves the top-p cut lands on the
 * same shortlisted token for every outside mass in [0, B] (1e-4 relative margin)
 * and that the cut's value is strictly above `bound` (no outside token joins on
 * a tie).  The dense draw then keeps only shortlisted tokens, and the row is
 * rewritten exactly as above: the sampler's own top-k / top-p on the rewritten
 * row makes the same cut (outside mass 0 is in the range) and the same draw.
 * Sampling with top_p < 1 only.  The planner's survivors are the allowed
 * shortlisted tokens that clear `bound` (fewer than K, so all are in the dense
 * top-K); an allowed shortlisted token penalised below that (the repeat penalty)
 * may or may not make the dense top-K, so its mass joins the uncertain tail.
 *
 * Pure C, libc + libm; tests/test_grammar_mask_shortlist.c compares it with the
 * full mask on random rows, tests/test_grammar_exact_bound.c the outside bound's
 * draws with the dense draws.
 *
 * @status Used by sample_token (engine.c) for every masked step.
 *         PN_GRAMMAR_SHORTLIST=0 turns it off (full mask every step).
 *         The outside bound: PN_GRAMMAR_EXACT_BOUND=0 turns it off (engine.c);
 *         red PN_TT_RED=grammar_bound_blind charges the outside nothing.
 * @issues None
 * @todo None
 */
#ifndef PN_GRAMMAR_MASK_SHORTLIST_H
#define PN_GRAMMAR_MASK_SHORTLIST_H

#include <stdbool.h>
#include <stdint.h>

#include "sampler_topn_tail.h"

/** Largest shortlist judged (also the sampler's sparse survivor cap). */
#define GMS_MAX 256

/** What bounds the ALLOWED tokens outside the shortlist (each has raw logit <= bound). */
typedef struct {
    /** At most this many of them (n_vocab - n_ids when nothing better is known). */
    double count;
    /** Optional device tail counts of the RAW row (sampler_topn_tail.h), NULL = none:
     *  bin_counts[0..bins->bins], the shortlisted tokens still counted in them. */
    const tnt_tail_bins *bins;
    const uint32_t *bin_counts;
    /** Multiplier on B: 1 in production; 0 only for the red arm (blind to the outside). */
    double scale;
} gms_outside;

/** The sampling parameters the decision depends on. */
typedef struct {
    float temperature;        /**< <= 0: greedy (K = 1, no repeat penalty) */
    int top_k;                /**< K when sampling; <= 0 or > shortlist: undecidable */
    float top_p;              /**< the outside bound needs 0 < top_p < 1 */
    const gms_outside *outside; /**< NULL: decide only when K allowed tokens clear `bound` */
    float repeat_penalty;     /**< sampling only; must be >= 1 */
    float presence_penalty;   /**< must be >= 0 */
    float frequency_penalty;  /**< must be >= 0 */
    bool has_logit_bias;      /**< a bias can raise a token: undecidable */
    const uint32_t *recent;
    int n_recent;
} gms_sampling;

/** gms_mask_from_shortlist's verdicts (nonzero = decided, the row rewritten). */
enum { GMS_UNDECIDED = 0, GMS_DECIDED_CLEAR = 1, GMS_DECIDED_BOUND = 2 };

/** Is token `id` allowed by the grammar right now? */
typedef bool (*gms_allowed_fn)(void *ctx, uint32_t id);

/**
 * Try to apply the mask from a shortlist.
 *
 * @param logits   the row (n_vocab), rewritten only on success
 * @param ids      shortlisted token ids (distinct, < n_vocab)
 * @param n_ids    1..GMS_MAX
 * @param bound    the largest raw logit NOT in the shortlist (or any upper bound)
 * @return GMS_UNDECIDED (0: the row is untouched), GMS_DECIDED_CLEAR (K allowed tokens
 *         clear `bound`) or GMS_DECIDED_BOUND (the outside bound decided); nonzero = rewritten
 *
 * Thread-safe: yes (no shared state)
 */
int gms_mask_from_shortlist(float *logits, int n_vocab, const uint32_t *ids, int n_ids,
                             float bound, const gms_sampling *sampling,
                             gms_allowed_fn allowed, void *allowed_ctx);

/**
 * Select the shortlist from a row: every finite entry when the row holds at
 * most GMS_MAX of them (a backend top-N row: `*bound` = its smallest value),
 * else the GMS_MAX largest (`*bound` = the largest value left out).
 * @return the count written to `ids` (0 for an empty row)
 */
int gms_select_shortlist(const float *logits, int n_vocab, uint32_t *ids, float *bound);

#endif /* PN_GRAMMAR_MASK_SHORTLIST_H */
