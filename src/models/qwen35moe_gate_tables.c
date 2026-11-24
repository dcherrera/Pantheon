/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gate_tables.c
 * @brief The exact card gates' lookup tables (qwen35moe_gate_tables.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "qwen35moe_gate_tables.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tt/kernels/ftgc_table.h"

#define CODES FTGT_CODES
/** Layers built at once (one thread each). */
#define BUILD_THREADS 8u

static float code_float(uint32_t code)
{
    const uint32_t bits = code << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/** A NaN code: outside the claim (the host refuses a non-finite gate). */
static bool code_is_nan(uint32_t code)
{
    return (code & 0x7FFFu) > FTGT_LAST_MAGNITUDE;
}

typedef struct {
    const pn_qwen35moe_ssm_layer *layer;
    uint32_t heads;
    uint32_t *slots;  /* this layer's heads x SLOT_WORDS */
    uint32_t *beta;   /* [CODES] out, or NULL */
    uint32_t widest;
    int rc;
} build_job;

/** One layer: every code through the host's gate body, then its heads' tables. */
static void *build_layer(void *arg)
{
    build_job *job = (build_job *)arg;
    const uint32_t heads = job->heads;
    uint32_t *values = malloc((size_t)heads * CODES * sizeof(uint32_t));
    float *alpha = malloc(heads * sizeof(float)), *beta_raw = malloc(heads * sizeof(float));
    float *decay = malloc(heads * sizeof(float)), *beta = malloc(heads * sizeof(float));

    job->rc = -1;
    job->widest = 0u;
    if (!values || !alpha || !beta_raw || !decay || !beta)
        goto done;
    for (uint32_t code = 0; code < CODES; code++) {
        const float x = code_float(code);

        for (uint32_t h = 0; h < heads; h++)
            alpha[h] = beta_raw[h] = x;
        pn_qwen35moe_gdn_gates(job->layer, alpha, beta_raw, (int)heads, decay, beta);
        for (uint32_t h = 0; h < heads; h++) {
            values[(size_t)h * CODES + code] = float_bits(decay[h]);
            /* beta is a function of the code alone: a head that disagrees is a lane that saw
             * its neighbours, and the tables would not be the host's. */
            if (!code_is_nan(code) && float_bits(beta[h]) != float_bits(beta[0]))
                goto done;
        }
        if (job->beta)
            job->beta[code] = float_bits(beta[0]);
    }
    for (uint32_t h = 0; h < heads; h++) {
        const uint32_t words = ftgc_table_build(values + (size_t)h * CODES,
                                                job->slots + (size_t)h *
                                                                 PN_QWEN35_GATE_TABLE_SLOT_WORDS,
                                                PN_QWEN35_GATE_TABLE_SLOT_WORDS);

        if (words == 0u)
            goto done;
        if (words > job->widest)
            job->widest = words;
    }
    job->rc = 0;
done:
    free(values);
    free(alpha);
    free(beta_raw);
    free(decay);
    free(beta);
    return NULL;
}

void pn_qwen35_gate_tables_free(pn_qwen35_gate_tables *tables)
{
    if (!tables)
        return;
    free(tables->decay);
    free(tables->beta);
    memset(tables, 0, sizeof(*tables));
}

int pn_qwen35_gate_tables_build(const pn_qwen35moe_ssm_layer *const *layers, uint32_t n_layers,
                                uint32_t heads, pn_qwen35_gate_tables *out)
{
    uint32_t *beta_values = NULL;
    int rc = -1;

    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    if (!layers || n_layers == 0u || n_layers > PN_QWEN35_GATE_TABLE_MAX_LAYERS || heads == 0u)
        return -1;
    out->layers = n_layers;
    out->heads = heads;
    out->decay = calloc((size_t)n_layers * heads * PN_QWEN35_GATE_TABLE_SLOT_WORDS,
                        sizeof(uint32_t));
    out->beta = calloc(PN_QWEN35_GATE_TABLE_SLOT_WORDS, sizeof(uint32_t));
    beta_values = malloc(CODES * sizeof(uint32_t));
    if (!out->decay || !out->beta || !beta_values)
        goto done;
    for (uint32_t first = 0; first < n_layers; first += BUILD_THREADS) {
        build_job jobs[BUILD_THREADS];
        pthread_t threads[BUILD_THREADS];
        bool started[BUILD_THREADS];
        const uint32_t count = n_layers - first < BUILD_THREADS ? n_layers - first : BUILD_THREADS;

        for (uint32_t i = 0; i < count; i++) {
            jobs[i].layer = layers[first + i];
            jobs[i].heads = heads;
            jobs[i].slots = out->decay + (size_t)(first + i) * heads *
                                             PN_QWEN35_GATE_TABLE_SLOT_WORDS;
            jobs[i].beta = (first + i == 0u) ? beta_values : NULL;
            started[i] = pthread_create(&threads[i], NULL, build_layer, &jobs[i]) == 0;
            if (!started[i])
                build_layer(&jobs[i]);
        }
        /* EVERY thread joined before any result is read: a failed layer must not free the
         * slots a sibling is still writing. */
        for (uint32_t i = 0; i < count; i++)
            if (started[i])
                pthread_join(threads[i], NULL);
        for (uint32_t i = 0; i < count; i++) {
            if (jobs[i].rc != 0)
                goto done;
            out->layer_widest[first + i] = jobs[i].widest;
            if (jobs[i].widest > out->widest)
                out->widest = jobs[i].widest;
        }
    }
    {
        const uint32_t words =
            ftgc_table_build(beta_values, out->beta, PN_QWEN35_GATE_TABLE_SLOT_WORDS);

        if (words == 0u)
            goto done;
        if (words > out->widest)
            out->widest = words;
    }
    rc = 0;
done:
    free(beta_values);
    if (rc != 0)
        pn_qwen35_gate_tables_free(out);
    return rc;
}

static uint32_t table_value(const uint32_t *table, uint32_t code)
{
    uint32_t value = 0u;
    const uint32_t index = ftgc_table_resolve(table, code, &value);

    return index == FTGT_CONSTANT ? value : table[kFtgtHeaderWords + index];
}

uint32_t pn_qwen35_gate_tables_decay(const pn_qwen35_gate_tables *tables, uint32_t layer,
                                     uint32_t head, uint32_t code)
{
    return table_value(tables->decay + ((size_t)layer * tables->heads + head) *
                                           PN_QWEN35_GATE_TABLE_SLOT_WORDS,
                       code);
}

uint32_t pn_qwen35_gate_tables_beta(const pn_qwen35_gate_tables *tables, uint32_t code)
{
    return table_value(tables->beta, code);
}

static uint32_t check_rng(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/** A non-NaN code, weighted toward the projections' range. */
static uint32_t check_code(uint32_t *state)
{
    uint32_t code;

    if (check_rng(state) % 2u == 0u)
        return ((check_rng(state) & 1u) << 15) | ((119u + check_rng(state) % 14u) << 7) |
               (check_rng(state) & 0x7Fu);
    do
        code = check_rng(state) & 0xFFFFu;
    while (code_is_nan(code));
    return code;
}

/** One row of codes through the host body; the words that differ from the tables. */
static uint64_t check_row(const pn_qwen35_gate_tables *t, const pn_qwen35moe_ssm_layer *layer,
                          uint32_t l, const uint32_t *alpha_codes, const uint32_t *beta_codes,
                          float *scratch)
{
    const uint32_t heads = t->heads;
    float *alpha = scratch, *beta_raw = scratch + heads, *decay = scratch + 2u * heads,
          *beta = scratch + 3u * heads;
    uint64_t bad = 0u;

    for (uint32_t h = 0; h < heads; h++) {
        alpha[h] = code_float(alpha_codes[h]);
        beta_raw[h] = code_float(beta_codes[h]);
    }
    pn_qwen35moe_gdn_gates(layer, alpha, beta_raw, (int)heads, decay, beta);
    for (uint32_t h = 0; h < heads; h++) {
        bad += float_bits(decay[h]) != pn_qwen35_gate_tables_decay(t, l, h, alpha_codes[h]);
        bad += float_bits(beta[h]) != pn_qwen35_gate_tables_beta(t, beta_codes[h]);
    }
    return bad;
}

uint64_t pn_qwen35_gate_tables_check(const pn_qwen35_gate_tables *tables,
                                     const pn_qwen35moe_ssm_layer *const *layers,
                                     uint32_t random_rows, uint32_t seed)
{
    const uint32_t heads = tables ? tables->heads : 0u;
    uint32_t *alpha_codes = malloc(heads * sizeof(uint32_t) + 1u);
    uint32_t *beta_codes = malloc(heads * sizeof(uint32_t) + 1u);
    float *scratch = malloc(4u * heads * sizeof(float) + 1u);
    uint32_t state = seed ? seed : 0x9E3779B9u;
    uint64_t bad = 0u;

    if (!tables || !layers || !tables->decay || !alpha_codes || !beta_codes || !scratch) {
        bad = UINT64_MAX;
        goto done;
    }
    /* Every code in every head position of layer 0, each head's neighbours on other codes. */
    for (uint32_t code = 0; code < CODES; code++) {
        for (uint32_t h = 0; h < heads; h++) {
            uint32_t a = (code + h * 7919u) & 0xFFFFu, b = (code + h * 104729u + 31u) & 0xFFFFu;

            alpha_codes[h] = code_is_nan(a) ? 0u : a;
            beta_codes[h] = code_is_nan(b) ? 0x8000u : b;
        }
        bad += check_row(tables, layers[0], 0u, alpha_codes, beta_codes, scratch);
    }
    for (uint32_t l = 0; l < tables->layers; l++)
        for (uint32_t r = 0; r < random_rows; r++) {
            for (uint32_t h = 0; h < heads; h++) {
                alpha_codes[h] = check_code(&state);
                beta_codes[h] = check_code(&state);
            }
            bad += check_row(tables, layers[l], l, alpha_codes, beta_codes, scratch);
        }
done:
    free(alpha_codes);
    free(beta_codes);
    free(scratch);
    return bad;
}

pn_qwen35_gate_tables *pn_qwen35_gate_tables_prepare(const pn_qwen35moe_ssm_layer *const *layers,
                                                     uint32_t n_layers, uint32_t heads, bool red)
{
    pn_qwen35_gate_tables *tables = calloc(1u, sizeof(*tables));
    uint64_t bad;

    if (!tables)
        return NULL;
    if (pn_qwen35_gate_tables_build(layers, n_layers, heads, tables) != 0) {
        fprintf(stderr, "qwen35_tt: exact card gates REFUSED — a gate table did not build "
                        "within %u words a head; the gates stay on the host\n",
                PN_QWEN35_GATE_TABLE_SLOT_WORDS);
        free(tables);
        return NULL;
    }
    if (red) {
        /* RED `gate_table_corrupt`: the first window entry of layer 0 head 0's table. */
        uint32_t *t = tables->decay;

        t[kFtgtHeaderWords] ^= 1u;
    }
    bad = pn_qwen35_gate_tables_check(tables, layers, 256u, 0x51ED2701u);
    if (bad != 0u) {
        fprintf(stderr, "qwen35_tt: exact card gates REFUSED — %llu word(s) of the host gates "
                        "differ from the tables on mixed rows; the gates stay on the host\n",
                (unsigned long long)bad);
        pn_qwen35_gate_tables_free(tables);
        free(tables);
        return NULL;
    }
    fprintf(stderr, "qwen35_tt: exact card gate tables — %u layers x %u heads, widest %u words "
                    "(slot %u), every code proved, mixed rows checked\n",
            n_layers, heads, tables->widest, PN_QWEN35_GATE_TABLE_SLOT_WORDS);
    return tables;
}
