/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_noise.c
 * @brief Implementation of the reproducible Gaussian perturbations declared in
 *        parity_noise.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "parity_noise.h"

#include <math.h>
#include <string.h>

/** The default seed when PN_PARITY_SEED is unset or zero. */
#define PARITY_NOISE_DEFAULT_SEED 88172645463325252ULL

#define PARITY_NOISE_TWO_PI 6.283185307179586

static uint64_t noise_next(parity_noise_rng *rng)
{
    uint64_t x = rng->state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

/**
 * One Box-Muller pair, scaled by `sigma` INSIDE the radius rather than afterwards.
 *
 * The association is load-bearing, not stylistic: sigma*(r*cos) and (sigma*r)*cos round
 * differently in double, and the recorded noise-ladder figures were produced by the second.
 * Keeping it means a ladder re-run reproduces the record bit for bit.
 */
static void noise_pair(parity_noise_rng *rng, double sigma, double *a, double *b)
{
    double u1 = ((double)(noise_next(rng) >> 11) + 1.0) / 9007199254740993.0;
    double u2 = (double)(noise_next(rng) >> 11) / 9007199254740992.0;
    double r = sigma * sqrt(-2.0 * log(u1));

    *a = r * cos(PARITY_NOISE_TWO_PI * u2);
    *b = r * sin(PARITY_NOISE_TWO_PI * u2);
}

void parity_noise_seed(parity_noise_rng *rng, uint64_t seed)
{
    rng->state = seed ? seed : PARITY_NOISE_DEFAULT_SEED;
}

void parity_noise_add(float *dst, const float *src, int n, double sigma, parity_noise_rng *rng)
{
    if (dst != src)
        memcpy(dst, src, (size_t)n * sizeof(float));
    if (sigma <= 0.0)
        return;
    for (int j = 0; j + 1 < n; j += 2) {
        double a, b;

        noise_pair(rng, sigma, &a, &b);
        dst[j] = (float)((double)dst[j] + a);
        dst[j + 1] = (float)((double)dst[j + 1] + b);
    }
}

void parity_noise_direction(float *dst, int n, double norm, parity_noise_rng *rng)
{
    double sum_sq = 0.0, scale;

    for (int j = 0; j < n; j += 2) {
        double a, b;

        noise_pair(rng, 1.0, &a, &b);
        dst[j] = (float)a;
        if (j + 1 < n)
            dst[j + 1] = (float)b;
    }
    for (int j = 0; j < n; j++)
        sum_sq += (double)dst[j] * (double)dst[j];
    if (sum_sq <= 0.0)
        return;
    scale = norm / sqrt(sum_sq);
    for (int j = 0; j < n; j++)
        dst[j] = (float)((double)dst[j] * scale);
}
