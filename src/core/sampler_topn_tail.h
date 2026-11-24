/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_topn_tail.h
 * @brief Decide a SAMPLED draw from a backend's top-N row when the request's top-k reaches
 *        past the N (top_k <= 0, i.e. off, or top_k > N) — exactly, never approximately.
 *
 * A top-N row names the N largest raw logits; every other ("tail") token has a raw logit
 * <= `bound`, the smallest of the N. Penalties that only lower (presence/frequency >= 0,
 * repeat penalty >= 1) keep it there. With the N's penalised values v_i, m = max v_i and
 * temperature T, the full softmax normaliser is Z = W + t where
 *     W = sum_N exp((v_i - m) / T)   (known)
 *     t = the tail's share            (unknown, 0 <= t <= B = (V - N) exp((bound - m) / T))
 * With top-k off every token is a candidate. With a top-k between the N and the vocabulary
 * every survivor must keep a penalised value >= `bound` (checked): then no tail token
 * outranks it, the top-k set holds all N, and the dense sampler's set (ties at its threshold
 * included) adds only tail tokens.
 *
 * Two shapes decide from the N alone:
 *   NUCLEUS (top_p < 1): top-p keeps the shortest descending prefix whose mass reaches
 *     top_p * Z. If that prefix is the same j for every Z in [W, W + B] (with a 1e-4
 *     relative margin for the dense path's float sums) and its last value is strictly
 *     above `bound`, the nucleus is j survivors (plus ties) and Z cancels in the
 *     renormalisation: the draw is the survivors' draw at top_k = j, top_p off.
 *   MIXTURE (top_p >= 1): the dense distribution p puts mass W / Z >= a = W / (W + B) on
 *     the N, proportional to exp(v_i / T). So p = a * h + (1 - a) * r, h the N's own
 *     softmax and r = (p - a h) / (1 - a) >= 0. Draw u: u < a -> sample h with u / a;
 *     else the dense row is needed and r is sampled with (u - a) / (1 - a). Each branch
 *     uses a uniform independent of the branch, so the draw is distributed exactly as p.
 * Everything else (the cut ambiguous across [W, W + B], a survivor below `bound` under a
 * partial top-k) is DENSE: the caller refills the row and runs the full path.
 *
 * Pure C, libc + libm only; tests/test_topn_tail_sampler.c checks it against the dense
 * sampler by Monte Carlo and by the exact distribution.
 *
 * @status Used by sample_token_core (engine.c) for top-N rows; PN_TT_SAMPLED_TAIL=0 keeps
 *         such requests on the dense row (pn_qwen35moe_tt_select_rows).
 * @issues The bound B is loose (every tail token at `bound`); the rate of DENSE / residual
 *         draws is what it costs (measured on the card).
 * @todo None
 */
#ifndef PN_SAMPLER_TOPN_TAIL_H
#define PN_SAMPLER_TOPN_TAIL_H

#include <stdbool.h>
#include <stdint.h>

/** Largest survivor set judged (the sampler's sparse survivor cap). */
#define TNT_MAX 256

typedef enum {
    TNT_DENSE = 0,   /**< undecidable from the N: refill the dense row */
    TNT_NUCLEUS,     /**< sample the survivors at top_k = nucleus_k, top_p off */
    TNT_MIXTURE      /**< survivors with probability head_mass, else the residual */
} tnt_kind;

/** The parameters the decision depends on. */
typedef struct {
    float temperature;       /**< > 0 */
    float top_p;             /**< >= 1: off */
    int top_k;               /**< <= 0 or >= n_vocab: off; else > the survivor count */
    int n_vocab;             /**< the dense row's length (every tail token counts) */
    /** The tail bound's multiplier on (V - N) exp((bound - m) / T); 1 in production,
     *  0 only for the distribution test's red arm (a bound that ignores the tail). */
    double tail_scale;
    /** B from the card's tail counts (tnt_tail_from_counts), or < 0 for the trivial bound. */
    double tail_mass;
    /** 1 when the counts prove every survivor stays in a partial top-k set. */
    int survivors_in_topk;
} tnt_sampling;

typedef struct {
    tnt_kind kind;
    int nucleus_k;              /**< NUCLEUS: survivors kept (before ties) */
    double head_mass;           /**< MIXTURE: a = W / (W + B), in (0, 1] */
    double head_prob[TNT_MAX];  /**< MIXTURE: h_i = exp((v_i - m) / T) / W, survivor order */
} tnt_plan;

/**
 * Plan the draw.
 *
 * @param vals   the survivors' PENALISED logits (any order; `head_prob` follows it)
 * @param S      1..TNT_MAX survivors
 * @param bound  the smallest RAW survivor logit (every tail token's raw logit is <= it)
 * @return the plan's kind (also in out->kind)
 *
 * Thread-safe: yes (no shared state)
 */
tnt_kind tnt_plan_draw(const float *vals, int S, float bound, const tnt_sampling *s,
                       tnt_plan *out);

/**
 * MIXTURE's residual branch: the draw u landed at or past head_mass. `probs` is the dense
 * row's final distribution (the full path's), `ids` the survivors' ids in the order
 * `head_prob` follows. Walks ids ascending, as the dense draw does.
 * @return the token for u in [head_mass, 1)
 *
 * Thread-safe: yes
 */
uint32_t tnt_residual_draw(const float *probs, int n_vocab, const int *ids,
                           const tnt_plan *plan, int S, float u);

/* ------------------------------------------------------------------------------------------
 * Tail COUNTS (a backend that can count the row into bins on the device, e.g. the TT card's
 * top-k pass with its tail flag): bins of bf16 order keys from `bound` down
 * T (ln V + 18), so the mass below the last bin is < e^-18 of one token at `bound`, each bin
 * charged at its upper value. B is then within one bin's e^(width / T) of the true tail.
 * ---------------------------------------------------------------------------------------- */

#define TNT_TAIL_BINS 32

typedef struct {
    uint32_t keys[TNT_TAIL_BINS];   /**< strictly descending bf16 order keys; keys[0] = key(bound) */
    double upper[TNT_TAIL_BINS + 1]; /**< the largest value a TAIL logit counted in bin j has */
    int bins;
    /** The bin whose threshold is the smallest key above x_min (bins 0..xmin_index hold exactly
     *  the logits above it), or -1. */
    int xmin_index;
} tnt_tail_bins;

/** The bf16 order key of a bf16-exact float (the kernel's key(): larger value, larger key). */
uint32_t tnt_key_from_float(float value);

/** Thresholds for one row: keys[0] = key(bound), then bound - j * delta, plus the smallest key
 *  above x_min (the smallest PENALISED survivor) when x_min < bound. */
void tnt_tail_bins_make(float bound, float x_min, float temperature, int n_vocab,
                        tnt_tail_bins *out);

/** The bin (0..bins) an order key is counted in. */
int tnt_tail_bin(const tnt_tail_bins *tb, uint32_t key);

/** The device's counting, on the host: counts[j] over `row` (bins + 1 entries). */
void tnt_tail_bins_count(const tnt_tail_bins *tb, const float *row, int n, uint32_t *counts);

/**
 * From the device's counts of the RAW row: B = the tail's mass bound in units of
 * exp((v - m) / T) (the survivors' own counts removed), and *in_topk = 1 when a partial top-k
 * provably keeps every survivor (x_min = the smallest PENALISED survivor).
 *
 * @param raw  the survivors' RAW logits
 * @return 0, or -1 when the counts are inconsistent with the survivors (do not use them)
 */
int tnt_tail_from_counts(const tnt_tail_bins *tb, const uint32_t *counts, const float *raw, int S,
                         double m, double inv_t, float x_min, int top_k, double *B, int *in_topk);

#endif /* PN_SAMPLER_TOPN_TAIL_H */
