/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_row_dump.c
 * @brief The masked-row dump instrument (see grammar_row_dump.h).
 *
 * @status Instrument only.
 * @issues None
 * @todo None
 */
#include "grammar_row_dump.h"
#include "grammar_mask_shortlist.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t g_dump_lock = PTHREAD_MUTEX_INITIALIZER;

int pn_grammar_row_dump_enabled(void)
{
    static int armed = -1;

    if (armed < 0) {
        const char *path = getenv("PN_GRAMMAR_ROW_DUMP");
        armed = path && *path;
    }
    return armed;
}

/* The row's top GRO_TOPN in the TT kernel's order: value descending, id ascending. */
static int gro_top_n(const float *logits, int n_vocab, uint32_t *ids)
{
    int n = 0;

    for (int i = 0; i < n_vocab; i++) {
        if (n == GRO_TOPN && !(logits[i] > logits[ids[n - 1]]))
            continue;
        int at = n < GRO_TOPN ? n++ : GRO_TOPN - 1;
        ids[at] = (uint32_t)i;
        for (; at > 0 && logits[ids[at]] > logits[ids[at - 1]]; at--) {
            uint32_t t = ids[at];
            ids[at] = ids[at - 1];
            ids[at - 1] = t;
        }
    }
    return n;
}

static bool gro_allowed(void *ctx, uint32_t id)
{
    return ((const unsigned char *)ctx)[id] != 0;
}

/* Would the top-N shortlist alone (no bound on the outside tokens) decide this draw? */
static int gro_shortlist_decides(const float *logits, int n_vocab, const uint32_t *ids, int n,
                                 const unsigned char *verdict, const gms_sampling *s)
{
    float *sparse = malloc((size_t)n_vocab * sizeof(float));
    const uint32_t neg_inf_bits = 0xff800000u;
    float neg_inf, bound = logits[ids[n - 1]];
    int decided;

    if (!sparse)
        return 0;
    memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));
    for (int i = 0; i < n_vocab; i++) sparse[i] = neg_inf;
    for (int i = 0; i < n; i++) sparse[ids[i]] = logits[ids[i]];
    decided = gms_mask_from_shortlist(sparse, n_vocab, ids, n, bound, s, gro_allowed,
                                      (void *)verdict);
    free(sparse);
    return decided;
}

static void gro_write(FILE *f, const gro_header *h, const uint32_t *recent, const uint32_t *ids,
                      const float *logits, const unsigned char *verdict)
{
    float vals[GRO_TOPN];
    unsigned char allowed[GRO_TOPN];

    for (int i = 0; i < GRO_TOPN; i++) {
        vals[i] = logits[ids[i]];
        allowed[i] = verdict[ids[i]];
    }
    fwrite(h, sizeof(*h), 1, f);
    if (h->n_recent > 0) fwrite(recent, sizeof(uint32_t), (size_t)h->n_recent, f);
    fwrite(ids, sizeof(uint32_t), GRO_TOPN, f);
    fwrite(vals, sizeof(float), GRO_TOPN, f);
    fwrite(allowed, 1, GRO_TOPN, f);
    if (h->has_row) {
        fwrite(logits, sizeof(float), (size_t)h->n_vocab, f);
        fwrite(verdict, 1, (size_t)h->n_vocab, f);
    }
}

void pn_grammar_row_dump(const float *logits, int n_vocab, const unsigned char *verdict,
                         const pn_sampling_params *p, const uint32_t *recent, int n_recent)
{
    uint32_t ids[GRO_TOPN];
    gro_header h;
    FILE *f;
    int n_allowed = 0;

    if (!pn_grammar_row_dump_enabled() || !logits || !verdict || !p ||
        gro_top_n(logits, n_vocab, ids) < GRO_TOPN)
        return;
    for (int i = 0; i < n_vocab; i++) n_allowed += verdict[i] != 0;
    gms_sampling s = {
        .temperature = p->temperature, .top_k = p->top_k, .top_p = p->top_p,
        .repeat_penalty = p->repeat_penalty, .presence_penalty = p->presence_penalty,
        .frequency_penalty = p->frequency_penalty,
        .has_logit_bias = p->logit_bias && p->n_logit_bias > 0,
        .recent = recent, .n_recent = recent ? n_recent : 0,
    };
    memset(&h, 0, sizeof(h));
    h.magic = GRO_MAGIC;
    h.n_vocab = n_vocab;
    h.temperature = p->temperature;
    h.top_p = p->top_p;
    h.top_k = p->top_k;
    h.repeat_penalty = p->repeat_penalty;
    h.presence_penalty = p->presence_penalty;
    h.frequency_penalty = p->frequency_penalty;
    h.has_logit_bias = s.has_logit_bias;
    h.n_recent = s.n_recent;
    h.n_allowed = n_allowed;
    h.has_row = (getenv("PN_GRAMMAR_ROW_DUMP_ALL") && atoi(getenv("PN_GRAMMAR_ROW_DUMP_ALL"))) ||
                !gro_shortlist_decides(logits, n_vocab, ids, GRO_TOPN, verdict, &s);
    pthread_mutex_lock(&g_dump_lock);
    f = fopen(getenv("PN_GRAMMAR_ROW_DUMP"), "ab");
    if (f) {
        gro_write(f, &h, recent, ids, logits, verdict);
        fclose(f);
    } else {
        fprintf(stderr, "grammar_row_dump: cannot open %s\n", getenv("PN_GRAMMAR_ROW_DUMP"));
    }
    pthread_mutex_unlock(&g_dump_lock);
}
