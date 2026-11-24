/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_row_census.c
 * @brief Price and check the grammar shortlist's outside bound on REAL masked rows: the
 *        records PN_GRAMMAR_ROW_DUMP wrote on the Mac (src/core/grammar_row_dump.h).
 *
 *   grammar_row_census [--seeds N] [--no-draws] [--topn N] <dump.grd>...
 *
 * Per file (one request) and in total: masked rows; rows the top-64 shortlist decides today
 * (K allowed tokens clear the 64th logit); of the rest, rows the outside bound decides when
 * it charges the allowed tokens outside the top-64 as
 *   T   every token outside at the 64th logit (n_vocab - 64 of them; no device work),
 *   C   the card's tail counts of the raw row (the T tokens, each in its bf16 bin),
 *   N   the grammar's exact count of allowed tokens outside, at the 64th logit,
 *   NC  that count, filled into the highest bins first.
 * Production today = T then C at row 0, T at a verify row past 0 (engine.c).
 * Draws: every row with a stored dense row (today's undecided ones) is drawn through the real
 * sampler both ways (tests/grammar_exact/grammar_row_compare.c), as row 0 and as a verify row
 * past 0, N seeds each: top-N token vs plain token. Run with PN_TT_RED=grammar_bound_blind
 * for the red arm. Real rows carry ~1e-6 of their mass outside the top-64, so a blind bound
 * rarely draws differently there; `--topn 8` (the draws' row cut to the top 8, a gate knob)
 * leaves real mass outside, where the red must mismatch and production must not.
 *
 * @status Tool (CPU only). Data: tools/grammar_exact/data/ (gitignored).
 * @issues None
 * @todo None
 */
#include "../../src/core/grammar_row_dump.h"
#include "../../src/core/grammar_mask_shortlist.h"
#include "../../src/core/engine.h"
#include "../../tests/grammar_exact/grammar_row_compare.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { V_T, V_C, V_N, V_NC, V_COUNT };
/* Undecided rows by allowed tokens inside the top-64: 0, 1, 2-5, 6-20, 21-39, 40+. */
#define BUCKETS 6

typedef struct {
    unsigned long rows, today, undecided, by[V_COUNT], prod_row0, prod_later;
    unsigned long inside[BUCKETS];
    grc_result row0, later;
} census;

typedef struct {
    gro_header h;
    uint32_t recent[256], ids[GRO_TOPN];
    float vals[GRO_TOPN];
    unsigned char allowed[GRO_TOPN];
    float *row;
    unsigned char *verdict;
} record;

static int read_record(FILE *f, record *r)
{
    if (fread(&r->h, sizeof(r->h), 1, f) != 1) return 0;
    if (r->h.magic != GRO_MAGIC || r->h.n_recent < 0 || r->h.n_recent > 256 || r->h.n_vocab < 1) {
        fprintf(stderr, "grammar_row_census: bad record\n");
        return 0;
    }
    if ((r->h.n_recent && fread(r->recent, 4, (size_t)r->h.n_recent, f) != (size_t)r->h.n_recent) ||
        fread(r->ids, 4, GRO_TOPN, f) != GRO_TOPN || fread(r->vals, 4, GRO_TOPN, f) != GRO_TOPN ||
        fread(r->allowed, 1, GRO_TOPN, f) != GRO_TOPN)
        return 0;
    if (!r->h.has_row) return 1;
    r->row = realloc(r->row, (size_t)r->h.n_vocab * sizeof(float));
    r->verdict = realloc(r->verdict, (size_t)r->h.n_vocab);
    return r->row && r->verdict &&
           fread(r->row, 4, (size_t)r->h.n_vocab, f) == (size_t)r->h.n_vocab &&
           fread(r->verdict, 1, (size_t)r->h.n_vocab, f) == (size_t)r->h.n_vocab;
}

static bool record_allowed(void *ctx, uint32_t id)
{
    return ((const unsigned char *)ctx)[id] != 0;
}

/* Does the outside bound, charged as variant v, decide this undecided row? */
static int decides(const record *r, int v, float *sparse)
{
    const uint32_t neg_inf_bits = 0xff800000u;
    const float bound = r->vals[GRO_TOPN - 1];
    uint32_t counts[TNT_TAIL_BINS + 1];
    tnt_tail_bins tb;
    int inside_allowed = 0;

    for (int i = 0; i < GRO_TOPN; i++) inside_allowed += r->allowed[i] != 0;
    gms_outside o = { v == V_N || v == V_NC ? (double)(r->h.n_allowed - inside_allowed)
                                            : (double)(r->h.n_vocab - GRO_TOPN), NULL, NULL, 1.0 };
    if (v == V_C || v == V_NC) {
        tnt_tail_bins_make(bound, bound, r->h.temperature, r->h.n_vocab, &tb);
        tnt_tail_bins_count(&tb, r->row, r->h.n_vocab, counts);
        o.bins = &tb;
        o.bin_counts = counts;
    }
    gms_sampling s = {
        .temperature = r->h.temperature, .top_k = r->h.top_k, .top_p = r->h.top_p, .outside = &o,
        .repeat_penalty = r->h.repeat_penalty, .presence_penalty = r->h.presence_penalty,
        .frequency_penalty = r->h.frequency_penalty, .has_logit_bias = r->h.has_logit_bias != 0,
        .recent = r->recent, .n_recent = r->h.n_recent,
    };
    for (int i = 0; i < r->h.n_vocab; i++) memcpy(&sparse[i], &neg_inf_bits, sizeof(float));
    for (int i = 0; i < GRO_TOPN; i++) sparse[r->ids[i]] = r->vals[i];
    return gms_mask_from_shortlist(sparse, r->h.n_vocab, r->ids, GRO_TOPN, bound, &s,
                                   record_allowed, r->verdict) == GMS_DECIDED_BOUND;
}

static int inside_bucket(const record *r)
{
    int n = 0;
    for (int i = 0; i < GRO_TOPN; i++) n += r->allowed[i] != 0;
    return n == 0 ? 0 : n == 1 ? 1 : n <= 5 ? 2 : n <= 20 ? 3 : n <= 39 ? 4 : 5;
}

static int g_topn = 0;   /* --topn: the draws' top-N row (a gate knob; 0 = 64) */

static void census_row(census *c, record *r, int seeds, float *sparse)
{
    c->rows++;
    if (!r->h.has_row) { c->today++; return; }
    c->undecided++;
    c->inside[inside_bucket(r)]++;
    for (int v = 0; v < V_COUNT; v++) c->by[v] += (unsigned long)decides(r, v, sparse);
    if (seeds <= 0) return;
    grc_row g = { r->h.n_vocab, r->row, r->verdict, pn_sampling_defaults(), r->recent,
                  r->h.n_recent, g_topn };
    g.params.temperature = r->h.temperature;
    g.params.top_k = r->h.top_k;
    g.params.top_p = r->h.top_p;
    g.params.repeat_penalty = r->h.repeat_penalty;
    g.params.presence_penalty = r->h.presence_penalty;
    g.params.frequency_penalty = r->h.frequency_penalty;
    const unsigned long b0 = c->row0.bound, b1 = c->later.bound;
    grc_compare(&g, GRC_ROW0, seeds, c->rows * 1000003ull, 0, &c->row0);
    grc_compare(&g, GRC_ROW_LATER, seeds, c->rows * 1000003ull, 0, &c->later);
    c->prod_row0 += c->row0.bound > b0;
    c->prod_later += c->later.bound > b1;
}

static void census_print(const char *name, const census *c)
{
    printf("%s rows %lu today %lu undecided %lu | T %lu C %lu N %lu NC %lu | prod row0 %lu "
           "later %lu | inside 0:%lu 1:%lu 2-5:%lu 6-20:%lu 21-39:%lu 40+:%lu\n", name, c->rows,
           c->today, c->undecided, c->by[V_T], c->by[V_C], c->by[V_N], c->by[V_NC], c->prod_row0,
           c->prod_later, c->inside[0], c->inside[1], c->inside[2], c->inside[3], c->inside[4],
           c->inside[5]);
    if (c->row0.draws + c->later.draws)
        printf("%s draws row0 %lu mismatches %lu bound %lu chunks %lu refills %lu | later %lu "
               "mismatches %lu bound %lu deferred %lu\n", name, c->row0.draws,
               c->row0.mismatches, c->row0.bound, c->row0.chunk_steps, c->row0.refills,
               c->later.draws, c->later.mismatches, c->later.bound, c->later.deferred);
}

static void census_add(census *t, const census *c)
{
    unsigned long *a = (unsigned long *)t;
    const unsigned long *b = (const unsigned long *)c;
    for (size_t i = 0; i < sizeof(census) / sizeof(unsigned long); i++) a[i] += b[i];
}

int main(int argc, char **argv)
{
    int seeds = 100, at = 1;
    census total;
    record r;
    float *sparse = NULL;

    memset(&total, 0, sizeof(total));
    memset(&r, 0, sizeof(r));
    for (; at < argc && argv[at][0] == '-'; at++) {
        if (strcmp(argv[at], "--seeds") == 0 && at + 1 < argc) seeds = atoi(argv[++at]);
        else if (strcmp(argv[at], "--no-draws") == 0) seeds = 0;
        else if (strcmp(argv[at], "--topn") == 0 && at + 1 < argc) g_topn = atoi(argv[++at]);
        else { fprintf(stderr, "usage: %s [--seeds N] [--no-draws] [--topn N] <dump.grd>...\n", argv[0]); return 2; }
    }
    for (; at < argc; at++) {
        FILE *f = fopen(argv[at], "rb");
        census c;

        if (!f) { fprintf(stderr, "grammar_row_census: cannot open %s\n", argv[at]); return 1; }
        memset(&c, 0, sizeof(c));
        while (read_record(f, &r)) {
            sparse = realloc(sparse, (size_t)r.h.n_vocab * sizeof(float));
            if (!sparse) return 1;
            census_row(&c, &r, seeds, sparse);
        }
        fclose(f);
        census_print(argv[at], &c);
        census_add(&total, &c);
    }
    census_print("TOTAL", &total);
    free(sparse); free(r.row); free(r.verdict);
    return total.row0.mismatches + total.later.mismatches ? 3 : 0;
}
