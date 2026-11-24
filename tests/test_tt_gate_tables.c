/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_gate_tables.c
 * @brief The exact card gates' tables (src/models/qwen35moe_gate_tables.h) against the host
 *        gates they replace, and the refusal that keeps a load on the host gates.
 *
 * Opens no device. Qwen3.8-shaped layers (48 heads, A = -exp(A_log) and dt biases over the
 * trained ranges):
 *   1. EXACT: for random rows of bf16 codes, the COMPACT run the card's front end would build
 *      from the tables (decay[row] | beta[row], ftgc_table_resolve) equals, word for word, what
 *      gdnc_compact_stage() stages from pn_qwen35moe_gdn_gates() on the same widened codes —
 *      the prefill's host path — including zeros, denormals, infinities and the extremes.
 *   2. THE CAP: every table fits its 32 KiB slot.
 *   3. THE REFUSAL: `gate_table_corrupt` (one entry flipped) makes prepare return NULL — the
 *      caller keeps the host gates and the load goes on (no abort, nothing leaked to free).
 *   4. The build time of all 48 layers, printed.
 *
 * Build: `make tests/test_tt_gate_tables` (any host).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/models/qwen35moe_gate_tables.h"
#include "tt/kernels/ftgc_table.h"

#define HEADS 48u
#define LAYERS 48u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok)
        g_failures++;
}

static uint32_t rng_state = 0xC0FFEE11u;

static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float code_float(uint32_t code)
{
    const uint32_t bits = code << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t random_code(void)
{
    static const uint32_t special[] = {0x0000u, 0x8000u, 0x0001u, 0x807Fu, 0x7F80u,
                                       0xFF80u, 0x7F7Fu, 0xFF7Fu, 0x3F80u, 0xC2C8u};
    uint32_t code;

    if (rng() % 8u == 0u)
        return special[rng() % (sizeof(special) / sizeof(special[0]))];
    do
        code = rng() & 0xFFFFu;
    while ((code & 0x7FFFu) > FTGT_LAST_MAGNITUDE);
    return code;
}

static pn_qwen35moe_ssm_layer g_layers[LAYERS];
static const pn_qwen35moe_ssm_layer *g_layer_ptrs[LAYERS];

/** `PN_GATE_TABLES_PARAMS=<file>` (dump_ssm_gates.py's output): the real model's ssm_a / ssm_dt instead of random ones. */
static bool load_params(const char *path)
{
    FILE *f = fopen(path, "r");
    char name[128];
    unsigned count;
    uint32_t loaded = 0u;

    if (!f)
        return false;
    while (fscanf(f, "%127s %u", name, &count) == 2) {
        unsigned layer;
        float *dst;

        if (count != HEADS || sscanf(name, "blk.%u.", &layer) != 1 || layer >= 64u)
            break;
        /* SSM layers in file order: layer index of the k-th SSM layer is k. */
        dst = strstr(name, "ssm_a") ? g_layers[loaded / 2u].ssm_a : g_layers[loaded / 2u].ssm_dt;
        for (unsigned i = 0; i < count; i++) {
            unsigned bits;

            if (fscanf(f, "%x", &bits) != 1) {
                fclose(f);
                return false;
            }
            memcpy(&dst[i], &bits, 4u);
        }
        loaded++;
        if (loaded == 2u * LAYERS)
            break;
    }
    fclose(f);
    return loaded == 2u * LAYERS;
}

static void make_layers(void)
{
    for (uint32_t l = 0; l < LAYERS; l++) {
        g_layers[l].ssm_a = malloc(HEADS * sizeof(float));
        g_layers[l].ssm_dt = malloc(HEADS * sizeof(float));
        for (uint32_t h = 0; h < HEADS; h++) {
            g_layers[l].ssm_a[h] = -expf((float)(rng() % 1000u) / 200.0f - 2.5f); /* -[0.08,12] */
            g_layers[l].ssm_dt[h] = (float)(rng() % 1000u) / 125.0f - 6.0f;        /* [-6, 2) */
        }
        g_layer_ptrs[l] = &g_layers[l];
    }
    if (getenv("PN_GATE_TABLES_PARAMS")) {
        const bool ok = load_params(getenv("PN_GATE_TABLES_PARAMS"));

        printf("  real parameters from %s: %s\n", getenv("PN_GATE_TABLES_PARAMS"),
               ok ? "loaded" : "NOT LOADED");
        check(ok, "the parameter file loads");
    }
}

/** What gdnc_compact_stage() stages for one head of a 16-row sub-chunk from host gates. */
static void host_run(uint32_t layer, const uint32_t (*alpha)[HEADS],
                     const uint32_t (*beta)[HEADS], uint32_t head, uint32_t run[64])
{
    memset(run, 0, 64u * sizeof(uint32_t));
    for (uint32_t row = 0; row < 16u; row++) {
        float a[HEADS], b[HEADS], decay[HEADS], bet[HEADS];

        for (uint32_t h = 0; h < HEADS; h++) {
            a[h] = code_float(alpha[row][h]);
            b[h] = code_float(beta[row][h]);
        }
        pn_qwen35moe_gdn_gates(g_layer_ptrs[layer], a, b, (int)HEADS, decay, bet);
        memcpy(&run[row], &decay[head], 4u);
        memcpy(&run[32u + row], &bet[head], 4u);
    }
}

/** The same run from the tables, as ftgc_table_head() resolves it. */
static void table_run(const pn_qwen35_gate_tables *t, uint32_t layer,
                      const uint32_t (*alpha)[HEADS], const uint32_t (*beta)[HEADS],
                      uint32_t head, uint32_t run[64])
{
    memset(run, 0, 64u * sizeof(uint32_t));
    for (uint32_t row = 0; row < 16u; row++) {
        run[row] = pn_qwen35_gate_tables_decay(t, layer, head, alpha[row][head]);
        run[32u + row] = pn_qwen35_gate_tables_beta(t, beta[row][head]);
    }
}

int main(void)
{
    pn_qwen35_gate_tables *tables;
    struct timespec t0, t1;
    char what[200];

    printf("test_tt_gate_tables: exact card gates' tables vs the host gates\n");
    make_layers();

    clock_gettime(CLOCK_MONOTONIC, &t0);
    tables = pn_qwen35_gate_tables_prepare(g_layer_ptrs, LAYERS, HEADS, false);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  build + check, %u layers x %u heads: %.2f s\n", LAYERS, HEADS,
           (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9);
    check(tables != NULL, "the tables build, prove and pass the lane check");
    if (!tables)
        return 1;

    printf("(1) exact\n");
    {
        static uint32_t alpha[16][HEADS], beta[16][HEADS];
        uint32_t differ = 0u, runs = 0u, want[64], got[64];

        for (uint32_t trial = 0; trial < 64u; trial++) {
            const uint32_t layer = rng() % LAYERS;

            for (uint32_t row = 0; row < 16u; row++)
                for (uint32_t h = 0; h < HEADS; h++) {
                    alpha[row][h] = random_code();
                    beta[row][h] = random_code();
                }
            for (uint32_t h = 0; h < HEADS; h++) {
                host_run(layer, (const uint32_t (*)[HEADS])alpha,
                         (const uint32_t (*)[HEADS])beta, h, want);
                table_run(tables, layer, (const uint32_t (*)[HEADS])alpha,
                          (const uint32_t (*)[HEADS])beta, h, got);
                differ += memcmp(want, got, sizeof(want)) != 0;
                runs++;
            }
        }
        snprintf(what, sizeof(what), "the tables' compact run is the host's: %u of %u runs differ",
                 differ, runs);
        check(differ == 0u, what);
    }

    printf("(2) the cap\n");
    snprintf(what, sizeof(what), "widest table %u words <= the %u-word slot", tables->widest,
             PN_QWEN35_GATE_TABLE_SLOT_WORDS);
    check(tables->widest > kFtgtHeaderWords && tables->widest <= PN_QWEN35_GATE_TABLE_SLOT_WORDS,
          what);
    pn_qwen35_gate_tables_free(tables);
    free(tables);

    printf("(3) the refusal\n");
    {
        pn_qwen35_gate_tables *red = pn_qwen35_gate_tables_prepare(g_layer_ptrs, 2u, HEADS, true);

        check(red == NULL, "RED gate_table_corrupt: prepare refuses (NULL), the caller stays on "
                           "the host gates and returns normally");
        check(pn_qwen35_gate_tables_prepare(g_layer_ptrs, 0u, HEADS, false) == NULL,
              "no layers: refused, not a crash");
    }
    {
        pn_qwen35_gate_tables t;
        uint64_t bad;

        check(pn_qwen35_gate_tables_build(g_layer_ptrs, 2u, HEADS, &t) == 0,
              "two layers build");
        bad = pn_qwen35_gate_tables_check(&t, g_layer_ptrs, 16u, 7u);
        check(bad == 0u, "...and check clean");
        t.beta[kFtgtHeaderWords + 3u] ^= 0x80000000u;
        bad = pn_qwen35_gate_tables_check(&t, g_layer_ptrs, 16u, 7u);
        snprintf(what, sizeof(what), "RED: one beta entry's sign flipped -> %llu words differ",
                 (unsigned long long)bad);
        check(bad != 0u, what);
        pn_qwen35_gate_tables_free(&t);
    }

    for (uint32_t l = 0; l < LAYERS; l++) {
        free(g_layers[l].ssm_a);
        free(g_layers[l].ssm_dt);
    }
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
