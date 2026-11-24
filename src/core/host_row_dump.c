/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file host_row_dump.c
 * @brief Dense decode-row dump for offline sampler tests (host_row_dump.h).
 *
 * @status Instrument only.
 * @issues None
 * @todo None
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "host_row_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_hrd_state = -1;   /* -1 unread, 0 off, 1 on */
static FILE *g_hrd_file;
static int g_hrd_every = 8, g_hrd_max = 64, g_hrd_seen, g_hrd_written;
static float *g_hrd_row;
static int g_hrd_vocab, g_hrd_pending;
static pn_sampling_params g_hrd_sp;
static uint32_t *g_hrd_recent;
static int g_hrd_n_recent, g_hrd_masked;

static int hrd_on(void)
{
    if (g_hrd_state >= 0)
        return g_hrd_state;
    const char *path = getenv("PN_HOST_ROW_DUMP");
    const char *every = getenv("PN_HOST_ROW_DUMP_EVERY");
    const char *max = getenv("PN_HOST_ROW_DUMP_MAX");
    g_hrd_state = 0;
    if (!path || !path[0])
        return 0;
    if (every && atoi(every) > 0) g_hrd_every = atoi(every);
    if (max && atoi(max) > 0) g_hrd_max = atoi(max);
    g_hrd_file = fopen(path, "wb");
    g_hrd_state = g_hrd_file ? 1 : 0;
    return g_hrd_state;
}

int pn_hrd_enabled(void)
{
    return hrd_on();
}

int pn_hrd_begin(const float *row, int n_vocab, const pn_sampling_params *sp,
                 const uint32_t *recent, int n_recent, int masked)
{
    g_hrd_pending = 0;
    if (!hrd_on() || g_hrd_written >= g_hrd_max || !row || !sp || n_vocab < 1)
        return 0;
    const char *gate = getenv("PN_HOST_ROW_DUMP_GATE");
    if (gate && gate[0] && access(gate, F_OK) != 0)
        return 0;   /* not dumping yet: the gate file does not exist */
    if (g_hrd_seen++ % g_hrd_every != 0)
        return 0;
    if (g_hrd_vocab != n_vocab) {
        free(g_hrd_row);
        g_hrd_row = malloc((size_t)n_vocab * sizeof(float));
        g_hrd_vocab = g_hrd_row ? n_vocab : 0;
        if (!g_hrd_row)
            return 0;
    }
    free(g_hrd_recent);
    g_hrd_recent = n_recent > 0 ? malloc((size_t)n_recent * sizeof(uint32_t)) : NULL;
    if (n_recent > 0 && !g_hrd_recent)
        return 0;
    if (n_recent > 0)
        memcpy(g_hrd_recent, recent, (size_t)n_recent * sizeof(uint32_t));
    memcpy(g_hrd_row, row, (size_t)n_vocab * sizeof(float));
    g_hrd_sp = *sp;
    g_hrd_n_recent = n_recent;
    g_hrd_masked = masked;
    g_hrd_pending = 1;
    return 1;
}

void pn_hrd_end(uint32_t token)
{
    if (!g_hrd_pending || !g_hrd_file)
        return;
    const pn_sampling_params *s = &g_hrd_sp;
    const uint32_t version = 1;
    const int32_t ints[] = { g_hrd_vocab };
    const float f1[] = { s->temperature, s->top_p };
    const int32_t top_k = s->top_k;
    const float rp = s->repeat_penalty;
    const int32_t rln = s->repeat_last_n;
    const float pf[] = { s->presence_penalty, s->frequency_penalty };
    const int32_t seeded = s->rng_seeded;
    const uint64_t rng = s->rng_state;
    const int32_t tail[] = { g_hrd_masked, (int32_t)token, g_hrd_n_recent };

    fwrite("PNRD", 1, 4, g_hrd_file);
    fwrite(&version, sizeof version, 1, g_hrd_file);
    fwrite(ints, sizeof ints, 1, g_hrd_file);
    fwrite(f1, sizeof f1, 1, g_hrd_file);
    fwrite(&top_k, sizeof top_k, 1, g_hrd_file);
    fwrite(&rp, sizeof rp, 1, g_hrd_file);
    fwrite(&rln, sizeof rln, 1, g_hrd_file);
    fwrite(pf, sizeof pf, 1, g_hrd_file);
    fwrite(&seeded, sizeof seeded, 1, g_hrd_file);
    fwrite(&rng, sizeof rng, 1, g_hrd_file);
    fwrite(tail, sizeof tail, 1, g_hrd_file);
    if (g_hrd_n_recent > 0)
        fwrite(g_hrd_recent, sizeof(uint32_t), (size_t)g_hrd_n_recent, g_hrd_file);
    fwrite(g_hrd_row, sizeof(float), (size_t)g_hrd_vocab, g_hrd_file);
    fflush(g_hrd_file);
    g_hrd_written++;
    g_hrd_pending = 0;
}
