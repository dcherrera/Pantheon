/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Differential parity test for the optimized token sampler.
 *
 * (2026-10-07: both copies take the llama.cpp/HF order — top-k renormalised before top-p.)
 * This test keeps a VERBATIM copy of the ORIGINAL sample_token implementation
 * (sample_token_ref, the pre-optimization slow path) and a copy of the new
 * OPTIMIZED implementation (sample_token_opt, kept textually in sync with
 * src/core/engine.c). For thousands of random logit vectors and random
 * parameter combinations (temperature including 0 for greedy, top_k, top_p,
 * repeat_penalty, and random recent-token windows), it feeds BOTH samplers the
 * exact same seeded RNG state and asserts they return the identical token id.
 *
 * The two samplers are self-contained (no engine/model dependencies), so this
 * builds and runs anywhere with just a C compiler and libm.
 *
 * Determinism: each sampler calls the global rand() at most once (the final
 * draw). Before each call we srand() to the same per-trial seed, so both draws
 * see the identical random value r. Test data is produced by a private xorshift
 * PRNG so that reseeding rand() never perturbs the generated logits/params.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Mirror of the real sampling params struct (see include/pantheon.h). */
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float repeat_penalty;
    int   repeat_last_n;
} pn_sampling_params;

static float rand_float(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* ==================================================================
 * REFERENCE: verbatim copy of the ORIGINAL sample_token (slow path).
 * ================================================================== */
static uint32_t sample_token_ref(float *logits, int n_vocab, pn_sampling_params *params,
                                 uint32_t *recent_tokens, int n_recent) {
    if (params->repeat_penalty != 1.0f && recent_tokens && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            uint32_t tok = recent_tokens[i];
            if (tok < (uint32_t)n_vocab) {
                if (logits[tok] > 0.0f)
                    logits[tok] /= params->repeat_penalty;
                else
                    logits[tok] *= params->repeat_penalty;
            }
        }
    }

    if (params->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return (uint32_t)best;
    }

    float inv_temp = 1.0f / params->temperature;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] *= inv_temp;
    }

    int top_k = params->top_k;
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float *probs = malloc(n_vocab * sizeof(float));
    if (!probs) return 0;

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] *= inv_sum;
    }

    if (top_k < n_vocab) {
        float *sorted = malloc(n_vocab * sizeof(float));
        memcpy(sorted, probs, n_vocab * sizeof(float));

        for (int k = 0; k < top_k; k++) {
            float best_val = sorted[k];
            int best_idx = k;
            for (int j = k + 1; j < n_vocab; j++) {
                if (sorted[j] > best_val) {
                    best_val = sorted[j];
                    best_idx = j;
                }
            }
            float tmp = sorted[k];
            sorted[k] = sorted[best_idx];
            sorted[best_idx] = tmp;
        }
        float threshold = sorted[top_k - 1];
        free(sorted);

        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        /* llama.cpp / HF order (2026-10-07): renormalise within the top-k set. */
        sum = 0.0f;
        for (int i = 0; i < n_vocab; i++) sum += probs[i];
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
        }
    }

    if (params->top_p < 1.0f) {
        float cum_prob = 0.0f;
        float top_p_threshold = 0.0f;

        float *probs_copy = malloc(n_vocab * sizeof(float));
        memcpy(probs_copy, probs, n_vocab * sizeof(float));

        while (cum_prob < params->top_p) {
            int best = 0;
            for (int i = 1; i < n_vocab; i++) {
                if (probs_copy[i] > probs_copy[best]) best = i;
            }
            if (probs_copy[best] <= 0.0f) break;
            cum_prob += probs_copy[best];
            top_p_threshold = probs_copy[best];
            probs_copy[best] = 0.0f;
        }
        free(probs_copy);

        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < top_p_threshold) probs[i] = 0.0f;
        }
    }

    sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += probs[i];
    if (sum > 0.0f) {
        inv_sum = 1.0f / sum;
        for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
    }

    float r = rand_float();
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[i];
        if (r < cum) {
            free(probs);
            return (uint32_t)i;
        }
    }

    for (int i = n_vocab - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) {
            free(probs);
            return (uint32_t)i;
        }
    }

    free(probs);
    return 0;
}

/* ==================================================================
 * OPTIMIZED: copy kept textually in sync with src/core/engine.c.
 * ================================================================== */
typedef struct {
    float *probs;
    float *work;
    int    cap;
} pn_sampler_scratch;

static _Thread_local pn_sampler_scratch g_sampler_scratch = {0};

static pn_sampler_scratch *sampler_scratch_get(int n) {
    pn_sampler_scratch *s = &g_sampler_scratch;
    if (s->cap < n) {
        free(s->probs); free(s->work);
        s->probs = malloc((size_t)n * sizeof(float));
        s->work  = malloc((size_t)n * sizeof(float));
        if (!s->probs || !s->work) {
            free(s->probs); free(s->work);
            s->probs = NULL; s->work = NULL; s->cap = 0;
            return NULL;
        }
        s->cap = n;
    }
    return s;
}

static float sampler_select_kth(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int i = lo, j = hi;
        float pivot = a[lo + ((hi - lo) >> 1)];
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) {
                float t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

static int sampler_cmp_desc(const void *pa, const void *pb) {
    float x = *(const float *)pa, y = *(const float *)pb;
    return (x < y) - (x > y);
}

static uint32_t sample_token_opt(float *logits, int n_vocab, pn_sampling_params *params,
                                 uint32_t *recent_tokens, int n_recent) {
    if (params->repeat_penalty != 1.0f && recent_tokens && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            uint32_t tok = recent_tokens[i];
            if (tok < (uint32_t)n_vocab) {
                if (logits[tok] > 0.0f)
                    logits[tok] /= params->repeat_penalty;
                else
                    logits[tok] *= params->repeat_penalty;
            }
        }
    }

    if (params->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return (uint32_t)best;
    }

    float inv_temp = 1.0f / params->temperature;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] *= inv_temp;
    }

    int top_k = params->top_k;
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    pn_sampler_scratch *sc = sampler_scratch_get(n_vocab);
    if (!sc) return 0;
    float *probs = sc->probs;

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] *= inv_sum;
    }

    if (top_k < n_vocab) {
        memcpy(sc->work, probs, (size_t)n_vocab * sizeof(float));
        float threshold = sampler_select_kth(sc->work, n_vocab, n_vocab - top_k);
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        /* llama.cpp / HF order (2026-10-07): renormalise within the top-k set. */
        sum = 0.0f;
        for (int i = 0; i < n_vocab; i++) sum += probs[i];
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
        }
    }

    if (params->top_p < 1.0f) {
        float *cand = sc->work;
        int nc = 0;
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] > 0.0f) cand[nc++] = probs[i];
        }
        qsort(cand, (size_t)nc, sizeof(float), sampler_cmp_desc);

        float cum_prob = 0.0f;
        float top_p_threshold = 0.0f;
        for (int t = 0; t < nc; t++) {
            if (!(cum_prob < params->top_p)) break;
            if (cand[t] <= 0.0f) break;
            cum_prob += cand[t];
            top_p_threshold = cand[t];
        }
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < top_p_threshold) probs[i] = 0.0f;
        }
    }

    sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += probs[i];
    if (sum > 0.0f) {
        inv_sum = 1.0f / sum;
        for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
    }

    float r = rand_float();
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[i];
        if (r < cum) return (uint32_t)i;
    }
    for (int i = n_vocab - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) return (uint32_t)i;
    }
    return 0;
}

/* ==================================================================
 * Test driver.
 * ================================================================== */

/* Private xorshift PRNG for generating test data (kept independent of the
 * global rand() used inside the samplers). */
static uint64_t g_rng = 0x123456789abcdef0ull;
static uint32_t xr(void) {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x;
    return (uint32_t)(x >> 32);
}
static float xr_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)xr() / 4294967296.0f);
}
static int xr_range(int lo, int hi) {   /* inclusive */
    return lo + (int)(xr() % (uint32_t)(hi - lo + 1));
}

int main(void) {
    const int TRIALS = 20000;
    int mismatches = 0;
    int greedy_cases = 0, topk_cases = 0, topp_cases = 0, penalty_cases = 0, tie_heavy = 0;

    int max_vocab = 4096;
    float *base   = malloc(max_vocab * sizeof(float));
    float *a      = malloc(max_vocab * sizeof(float));
    float *b      = malloc(max_vocab * sizeof(float));
    uint32_t *recent = malloc(max_vocab * sizeof(uint32_t));

    for (int trial = 0; trial < TRIALS; trial++) {
        /* Vocab size: mix of tiny, small and large. */
        int nv;
        int roll = xr_range(0, 9);
        if (roll < 2)      nv = xr_range(1, 8);       /* tiny edge cases */
        else if (roll < 7) nv = xr_range(16, 512);    /* small */
        else               nv = xr_range(1000, max_vocab);

        /* Decide whether this trial is "tie heavy": draw logits from a small
         * discrete set so lots of tokens share the exact same value. This
         * stresses the top-k threshold and top-p tie handling. */
        int ties = (xr_range(0, 2) == 0);
        if (ties) tie_heavy++;
        for (int i = 0; i < nv; i++) {
            if (ties) {
                int levels = xr_range(2, 6);
                base[i] = (float)xr_range(0, levels) * xr_uniform(0.5f, 3.0f) - 4.0f;
            } else {
                base[i] = xr_uniform(-25.0f, 25.0f);
            }
        }

        /* Random params. Temperature 0 ~1/6 of the time (greedy). */
        pn_sampling_params p;
        int troll = xr_range(0, 5);
        if (troll == 0) { p.temperature = 0.0f; greedy_cases++; }
        else if (troll == 1) p.temperature = 0.2f;   /* production OCR */
        else                 p.temperature = xr_uniform(0.05f, 2.0f);

        int kroll = xr_range(0, 4);
        if (kroll == 0)      p.top_k = 0;             /* -> all */
        else if (kroll == 1) p.top_k = xr_range(nv + 1, nv + 100); /* > vocab */
        else if (kroll == 2) p.top_k = 40;            /* production OCR */
        else                 p.top_k = xr_range(1, nv);
        if (p.top_k > 0 && p.top_k <= nv) topk_cases++;

        int proll = xr_range(0, 4);
        if (proll == 0)      p.top_p = 1.0f;          /* off */
        else if (proll == 1) p.top_p = 0.9f;          /* production OCR */
        else                 p.top_p = xr_uniform(0.05f, 1.0f);
        if (p.top_p < 1.0f) topp_cases++;

        int rpr = xr_range(0, 3);
        if (rpr == 0)      p.repeat_penalty = 1.0f;   /* off */
        else if (rpr == 1) p.repeat_penalty = 1.15f;  /* production OCR */
        else               p.repeat_penalty = xr_uniform(1.0f, 2.0f);

        p.repeat_last_n = 64;

        /* Random recent-token window. */
        int n_recent = 0;
        if (p.repeat_penalty != 1.0f && xr_range(0, 1)) {
            n_recent = xr_range(0, nv < 64 ? nv : 64);
            for (int i = 0; i < n_recent; i++) {
                /* Sometimes emit out-of-range ids to exercise the guard. */
                recent[i] = (xr_range(0, 9) == 0) ? (uint32_t)(nv + xr_range(0, 5))
                                                  : (uint32_t)xr_range(0, nv - 1);
            }
            if (n_recent > 0) penalty_cases++;
        }

        memcpy(a, base, nv * sizeof(float));
        memcpy(b, base, nv * sizeof(float));

        unsigned seed = 0x9E3779B9u ^ (unsigned)(trial * 2654435761u);

        srand(seed);
        uint32_t tok_ref = sample_token_ref(a, nv, &p, n_recent ? recent : NULL, n_recent);
        srand(seed);
        uint32_t tok_opt = sample_token_opt(b, nv, &p, n_recent ? recent : NULL, n_recent);

        if (tok_ref != tok_opt) {
            if (mismatches < 20) {
                fprintf(stderr,
                    "MISMATCH trial=%d nv=%d temp=%.4f top_k=%d top_p=%.4f rp=%.4f "
                    "n_recent=%d ties=%d : ref=%u opt=%u\n",
                    trial, nv, p.temperature, p.top_k, p.top_p, p.repeat_penalty,
                    n_recent, ties, tok_ref, tok_opt);
            }
            mismatches++;
        }
    }

    free(base); free(a); free(b); free(recent);

    printf("sampler parity: %d trials\n", TRIALS);
    printf("  greedy=%d  top_k=%d  top_p=%d  penalty=%d  tie_heavy=%d\n",
           greedy_cases, topk_cases, topp_cases, penalty_cases, tie_heavy);
    if (mismatches == 0) {
        printf("PASS: 100%% token-id match (0 mismatches)\n");
        return 0;
    }
    printf("FAIL: %d mismatches out of %d\n", mismatches, TRIALS);
    return 1;
}
