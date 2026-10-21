/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_noise.h
 * @brief Reproducible iid-Gaussian perturbations of KNOWN size, so a measured divergence
 *        has a yardstick instead of being read in a vacuum.
 *
 *        Two uses, one generator: perturbing a logit row by sigma (the calibration ladder
 *        that prices a card's ΔPPL against the cost of RANDOM error of the same magnitude),
 *        and drawing an isotropic direction of a given length in hidden space (the null
 *        hypothesis an ALIGNED divergence has to beat).
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_PARITY_NOISE_H
#define PN_TESTS_PARITY_NOISE_H

#include <stdint.h>

/** xorshift64*, so any perturbation is reproducible from its seed alone. */
typedef struct {
    uint64_t state;
} parity_noise_rng;

/** Seed the stream. A zero seed is replaced by the generator's fixed default, because
 *  xorshift64* is stuck at zero and a silently dead RNG is a silently absent control. */
void parity_noise_seed(parity_noise_rng *rng, uint64_t seed);

/**
 * dst = src + iid N(0, sigma), element by element. `dst` may alias `src`.
 * sigma <= 0 copies `src` unchanged, which is what makes arm 0 of a ladder exact.
 */
void parity_noise_add(float *dst, const float *src, int n, double sigma, parity_noise_rng *rng);

/**
 * Fill `dst` with an iid-Gaussian vector rescaled to have exactly Euclidean length `norm` —
 * a uniformly random DIRECTION of a chosen size. This is the isotropic null: error of the
 * same magnitude as a measured divergence but pointing nowhere in particular.
 */
void parity_noise_direction(float *dst, int n, double norm, parity_noise_rng *rng);

#endif /* PN_TESTS_PARITY_NOISE_H */
