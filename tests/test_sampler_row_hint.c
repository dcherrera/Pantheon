/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_sampler_row_hint.c
 * @brief Gate for the sampler's SURVIVOR HINT (engine.h pn_sampler_row_hint): a draw that
 *        reads a backend's survivor list must return the token the row scan returns.
 *
 * @status Complete. CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Arm A samples a sparse row (-INFINITY but for k survivors) the way every row is sampled
 * today: the sampler scans it. Arm B builds the row the way the TT forward does under
 * `PN_TT_ROW_RESET=1` — a tensor whose survivor ids follow its floats, `nnz` of them — and
 * hands it to pn_sampler_row_hint() before the same draw. Same row, params and RNG state:
 * the SAME token, over greedy, sampled, top-k past the survivors (tail / refill) and a
 * repeat window, with a `dense_row` refill wired in (a refill must void the hint). Every arm
 * B must have used the hint (pn_sampler_row_hint_uses advances), or the comparison is vacuous.
 *
 * `--selftest`: arm B's list misses the row's best survivor; greedy must then DIFFER — the
 * gate can go red.
 */
#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 20011

static int g_fail, g_checks;

/* -INFINITY everywhere, k survivors at distinct pseudo-random ids. */
static void build_row(float *row, int n, int k, uint64_t seed, int ties)
{
    uint64_t s = seed ? seed : 1;
    int placed = 0;

    for (int i = 0; i < n; i++) row[i] = -INFINITY;
    while (placed < k) {
        uint32_t bits;
        int id;

        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id = (int)((s >> 33) % (uint64_t)n);
        memcpy(&bits, &row[id], sizeof bits);
        if (bits != 0xff800000u) continue;
        row[id] = ties ? 3.25f : 9.0f - 0.37f * (float)placed + (float)((s >> 20) & 7) * 0.01f;
        placed++;
    }
}

/* The refill a top-N backend wires in: a deterministic dense row. */
static int dense_refill(void *ctx, float *logits, int n)
{
    const uint64_t seed = (uint64_t)(uintptr_t)ctx;

    for (int i = 0; i < n; i++)
        logits[i] = (float)((i * 2654435761u + (uint32_t)seed) % 1000u) * 0.004f;
    return 0;
}

/* A tensor the way the TT forward hands it back: the row, then its ascending survivor ids
 * (`drop_best`: the red, the best survivor left out of the list). */
static tm_tensor *hinted_tensor(const float *row, int n, int drop_best)
{
    int64_t shape[1] = { n };
    tm_tensor *t = tm_empty(shape, 1, TM_F32);
    uint32_t *ids;
    float *wide = malloc((size_t)n * sizeof(float) + 256 * sizeof(uint32_t));
    int count = 0, best = -1;

    free(t->data);
    t->data = wide;
    memcpy(wide, row, (size_t)n * sizeof(float));
    ids = (uint32_t *)(wide + n);
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &row[i], sizeof bits);
        if (bits == 0xff800000u) continue;
        if (best < 0 || row[i] > row[best]) best = i;
    }
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &row[i], sizeof bits);
        if (bits == 0xff800000u || (drop_best && i == best)) continue;
        ids[count++] = (uint32_t)i;
    }
    t->nnz = count;
    return t;
}

static uint32_t draw(float *row, int n, pn_sampling_params p, uint32_t *recent, int n_recent,
                     uint64_t seed, tm_tensor *hint)
{
    p.rng_seeded = 1;
    p.rng_state = pn_seed_to_rng_state(seed);
    if (hint) pn_sampler_row_hint(hint);
    return pn_sample_next_token(row, n, &p, recent, n_recent);
}

/* 1 match, 0 mismatch, -1 vacuous (the hint was not used). */
static int compare(int n, int k, int ties, float temp, int top_k, float top_p, float rp,
                   int refill, uint64_t seed, int red)
{
    float *row = malloc((size_t)n * sizeof(float)), *a = malloc((size_t)n * sizeof(float));
    uint32_t recent[16];
    pn_sampling_params p = pn_sampling_defaults();
    tm_tensor *t;
    unsigned long uses;
    uint32_t ta, tb;
    int found = 0;

    build_row(row, n, k, seed * 7919u + 13u, ties);
    for (int i = 0; i < n && found < 4; i++) {
        uint32_t bits;
        memcpy(&bits, &row[i], sizeof bits);
        if (bits != 0xff800000u) recent[found++] = (uint32_t)i;
    }
    for (int i = found; i < 16; i++) recent[i] = (uint32_t)((i * 977u + 5u) % (unsigned)n);
    p.temperature = temp;
    p.top_k = top_k;
    p.top_p = top_p;
    p.repeat_penalty = rp;
    p.repeat_last_n = 16;
    if (refill || temp > 0.0f) { /* a top-N backend always wires the refill */
        p.dense_row = dense_refill;
        p.dense_ctx = (void *)(uintptr_t)seed;
    }
    memcpy(a, row, (size_t)n * sizeof(float));
    ta = draw(a, n, p, recent, 16, seed, NULL);
    t = hinted_tensor(row, n, red);
    uses = pn_sampler_row_hint_uses();
    tb = draw((float *)t->data, n, p, recent, 16, seed, t);
    {
        const int used = pn_sampler_row_hint_uses() > uses;

        tm_free(t);
        free(row);
        free(a);
        if (!used) return -1;
    }
    return ta == tb;
}

int main(int argc, char **argv)
{
    const float temps[] = { 0.0f, 0.6f, 0.8f, 1.0f };
    const int topks[] = { 0, 1, 20, 40, 64 };
    const float topps[] = { 1.0f, 0.95f, 0.9f };
    const float rps[] = { 1.0f, 1.1f };
    const int ks[] = { 1, 8, 40, 64 };
    uint64_t seed = 1;
    int vacuous = 0;

    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        int bad = 0, cases = 0;

        printf("SELFTEST: the hint misses the best survivor — greedy must go RED\n");
        for (uint64_t s = 1; s <= 40; s++, cases++)
            bad += compare(VOCAB, 8, 0, 0.0f, 0, 1.0f, 1.0f, 0, s, 1) == 0;
        printf("  %d cases, %d mismatches\n", cases, bad);
        if (bad == 0) {
            printf("SELFTEST FAILED: a wrong hint still matched\n");
            return 1;
        }
        printf("SELFTEST PASSED\n");
        return 0;
    }
    printf("Sampler survivor hint — token-identity gate (vocab %d)\n", VOCAB);
    for (size_t a = 0; a < sizeof temps / sizeof *temps; a++)
    for (size_t b = 0; b < sizeof topks / sizeof *topks; b++)
    for (size_t c = 0; c < sizeof topps / sizeof *topps; c++)
    for (size_t d = 0; d < sizeof rps / sizeof *rps; d++)
    for (size_t e = 0; e < sizeof ks / sizeof *ks; e++)
    for (int ties = 0; ties < 2; ties++) {
        int r = compare(VOCAB, ks[e], ties, temps[a], topks[b], topps[c], rps[d], 0, seed++, 0);

        g_checks++;
        if (r < 0) { vacuous++; g_fail++; }
        else if (r == 0) {
            g_fail++;
            printf("  FAIL: temp %.2f top_k %d top_p %.2f rp %.2f k %d ties %d\n",
                   (double)temps[a], topks[b], (double)topps[c], (double)rps[d], ks[e], ties);
        }
    }
    printf("Production vocab, greedy and 0.8/40/0.9\n");
    for (uint64_t s = 500; s < 540; s++) {
        int r1 = compare(248320, 8, 0, 0.0f, 0, 1.0f, 1.0f, 0, s, 0);
        int r2 = compare(248320, 64, 0, 0.8f, 40, 0.9f, 1.0f, 1, s, 0);

        g_checks += 2;
        g_fail += (r1 != 1) + (r2 != 1);
        vacuous += (r1 < 0) + (r2 < 0);
    }
    printf("  %d comparisons, %d failures (%d vacuous)\n", g_checks, g_fail, vacuous);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
