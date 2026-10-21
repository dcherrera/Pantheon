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
 * Pure C, libc-only; tests/test_grammar_mask_shortlist.c compares it with the
 * full mask on random rows.
 *
 * @status Used by sample_token (engine.c) for every masked step.
 *         PN_GRAMMAR_SHORTLIST=0 turns it off (full mask every step).
 * @issues None
 * @todo None
 */
#ifndef PN_GRAMMAR_MASK_SHORTLIST_H
#define PN_GRAMMAR_MASK_SHORTLIST_H

#include <stdbool.h>
#include <stdint.h>

/** Largest shortlist judged (also the sampler's sparse survivor cap). */
#define GMS_MAX 256

/** The sampling parameters the decision depends on. */
typedef struct {
    float temperature;        /**< <= 0: greedy (K = 1, no repeat penalty) */
    int top_k;                /**< K when sampling; <= 0 or > shortlist: undecidable */
    float repeat_penalty;     /**< sampling only; must be >= 1 */
    float presence_penalty;   /**< must be >= 0 */
    float frequency_penalty;  /**< must be >= 0 */
    bool has_logit_bias;      /**< a bias can raise a token: undecidable */
    const uint32_t *recent;
    int n_recent;
} gms_sampling;

/** Is token `id` allowed by the grammar right now? */
typedef bool (*gms_allowed_fn)(void *ctx, uint32_t id);

/**
 * Try to apply the mask from a shortlist.
 *
 * @param logits   the row (n_vocab), rewritten only on success
 * @param ids      shortlisted token ids (distinct, < n_vocab)
 * @param n_ids    1..GMS_MAX
 * @param bound    the largest raw logit NOT in the shortlist (or any upper bound)
 * @return true when the row was rewritten (decided), false when untouched
 *
 * Thread-safe: yes (no shared state)
 */
bool gms_mask_from_shortlist(float *logits, int n_vocab, const uint32_t *ids, int n_ids,
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
