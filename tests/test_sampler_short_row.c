/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_sampler_short_row.c
 * @brief A draw handed the SHORT row (the exact top 257 of the full row R, engine.h
 *        pn_sampler_short_row) returns R's token and leaves R's RNG state, for every seed and
 *        setting, on both serve shapes.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Baseline: R through the real sampler (pn_sample_next_token), as the serve draws it today:
 * batched shape (dense_row = the no-op refill, sg_gs_row_already_dense) or single-user shape
 * (dense_row NULL).  Subject: the short row built from R the way the forward builds it
 * (value desc, id asc; not presented on a NaN row or a tie between the 256th and 257th value,
 * where the forward copies R instead), with a fill callback that writes R.  Same token, same
 * RNG state, same carried uniform.  Grammar: a bitmap, token i is "@@<i>".
 *
 * Rows (248320 and 20000 vocab): random with a heavy head; ties inside the list; a plateau on
 * the 256/257 edge (not presented); string content (most tokens allowed: CLEAR decides);
 * structural (few allowed, some outside the list: the BOUND / scan path refills); a penalised
 * head (all 257 listed tokens recent under a large presence penalty: the greedy and top-64
 * checks refill).  Settings: greedy, catalog (T 0.6 / top_p 0.95 / top_k 20), server defaults
 * (T 0.8 / 0.9 / 40, repeat 1.1), top_k 0, top_p 1, presence + frequency, a raising presence,
 * a logit bias, greedy with presence; mask live and not.  `--rows FILE` adds real dumped rows
 * (PN_HOST_ROW_DUMP records, their own settings; masked ones get a synthetic grammar).
 *
 * PASS: 0 mismatches, random/content rows decided from the short row, every refill reason ran.
 * Red (each must mismatch): `--selftest` (the fill writes nothing), `--red-margin`
 * (PN_SHORT_ROW_RED=margin), `--red-greedy` (=greedy). A wrong K is not a red arm: fewer than
 * 64 entries make the margin draw ineligible (fill), and a CLEAR verdict needs its top-k inside
 * the list, so a short list only changes the sparse normaliser's rounding.
 */
#include "../src/core/engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEEDS 24
#define RECENT_MAX 320
#define TEXT_BYTES 12

static uint64_t g_rng = 0x2B7E151628AED2A6ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rnd_unit(void) { return (float)(rnd() % 1000000) / 1000000.0f; }
static float rnd_gauss(void) { return rnd_unit() + rnd_unit() + rnd_unit() + rnd_unit() - 2.0f; }

static char *g_texts;
static int g_texts_n;

static int init_texts(int n_vocab)
{
    if (g_texts_n >= n_vocab) return 0;
    free(g_texts);
    g_texts = malloc((size_t)n_vocab * TEXT_BYTES);
    if (!g_texts) return -1;
    for (int i = 0; i < n_vocab; i++)
        snprintf(g_texts + (size_t)i * TEXT_BYTES, TEXT_BYTES, "@@%d", i);
    g_texts_n = n_vocab;
    return 0;
}
static const char *token_text(void *user, uint32_t id) {
    (void)user;
    return (int)id < g_texts_n ? g_texts + (size_t)id * TEXT_BYTES : NULL;
}
static int allowed(void *st, const char *text) {
    if (text[0] != '@') return 0;
    if (!text[1]) return 1;
    if (text[1] != '@') return 0;
    if (!text[2]) return 1;
    return ((const unsigned char *)st)[strtoul(text + 2, NULL, 10)] != 0;
}
static int refill_already_dense(void *ctx, float *logits, int n_vocab) {
    (void)ctx; (void)logits; (void)n_vocab;
    return 0;
}

/* The fill: R into the row (the forward's D2H of the slot's device copy). */
static int g_selftest;
static int fill_full(void *ctx, float *row, int n_vocab) {
    if (!g_selftest)
        memcpy(row, ctx, (size_t)n_vocab * sizeof(float));
    return 0;
}

static void lp_emit(void *user, uint32_t chosen, float lp, const uint32_t *ids, const float *lps,
                    int n) {
    (void)user; (void)chosen; (void)lp; (void)ids; (void)lps; (void)n;
}
static const pn_logprob_sink g_lp = { 3, lp_emit, NULL };
static const pn_logit_bias_entry g_bias[2] = { { 7, 2.5f }, { 11, -3.0f } };

typedef enum { ROW_RANDOM, ROW_TIES, ROW_EDGE_TIE, ROW_CONTENT, ROW_STRUCTURAL,
               ROW_PENALISED_HEAD, ROW_KINDS } row_kind;
typedef enum { SET_GREEDY, SET_CATALOG, SET_DEFAULTS, SET_TOPK_OFF, SET_TOPP_ONE, SET_PRES_FREQ,
               SET_PRES_RAISE, SET_BIAS, SET_LOGPROBS, SET_GREEDY_PRES, SET_KINDS } setting_kind;

typedef struct {
    unsigned long rows, presented, draws, mismatches;
} tally;

static const float *g_sort_row;
static int cmp_rank(const void *a, const void *b) {
    const int x = *(const int *)a, y = *(const int *)b;
    if (g_sort_row[x] != g_sort_row[y]) return g_sort_row[x] > g_sort_row[y] ? -1 : 1;
    return x - y;
}

static void rank_row(const float *row, int nv, int *rank)
{
    for (int i = 0; i < nv; i++) rank[i] = i;
    g_sort_row = row;
    qsort(rank, (size_t)nv, sizeof(int), cmp_rank);
}

/* -inf from its bits (fast-math does not keep INFINITY reliable; the sampler matches bits). */
static float neg_inf(void) {
    const uint32_t b = 0xff800000u;
    float v;
    memcpy(&v, &b, sizeof v);
    return v;
}

/* The forward's rule: present when the top 257 is clean (no NaN in R, 256th > 257th).
 * Writes the short row (n_list entries finite) and the bound; 0 = the forward copies R. */
static int build_short(const float *row, int nv, const int *rank, float *out, float *bound)
{
    for (int i = 0; i < nv; i++) {
        uint32_t b;
        memcpy(&b, &row[i], sizeof b);
        if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu)) return 0;
    }
    if (!(row[rank[PN_SHORT_ROW_K - 2]] > row[rank[PN_SHORT_ROW_K - 1]])) return 0;
    const float ninf = neg_inf();
    for (int i = 0; i < nv; i++) out[i] = ninf;
    for (int i = 0; i < PN_SHORT_ROW_K; i++) out[rank[i]] = row[rank[i]];
    *bound = row[rank[PN_SHORT_ROW_K - 1]];
    return 1;
}

static void build_row(row_kind kind, int nv, float *row, unsigned char *verdict, int *rank,
                      uint32_t *recent, int *n_recent)
{
    for (int i = 0; i < nv; i++) {
        row[i] = rnd_gauss() * 2.5f;
        verdict[i] = (unsigned char)(rnd() % 100u < 25u);
    }
    for (int i = 0; i < 6; i++) row[rnd() % (unsigned)nv] += 8.0f + (float)i;
    rank_row(row, nv, rank);
    if (kind == ROW_TIES) {     /* plateaus inside the list and in the tail, edge clear */
        for (int i = 20; i < 26; i++) row[rank[i]] = row[rank[20]];
        for (int i = 100; i < 140; i++) row[rank[i]] = row[rank[100]];
        for (int i = 300; i < nv; i++) row[rank[i]] = (float)(int)(row[rank[i]] * 2.0f) * 0.5f;
    }
    if (kind == ROW_EDGE_TIE)   /* plateau across the 256th / 257th place */
        for (int i = 250; i < 262; i++) row[rank[i]] = row[rank[250]];
    if (kind == ROW_CONTENT)
        for (int i = 0; i < nv; i++) verdict[i] = (unsigned char)(rnd() % 100u < 92u);
    if (kind == ROW_STRUCTURAL) {
        memset(verdict, 0, (size_t)nv);
        verdict[rank[rnd() % 3u]] = 1;
        verdict[rank[5u + rnd() % 30u]] = 1;
        verdict[rank[300u + rnd() % (unsigned)(nv - 300)]] = 1;
        verdict[rank[300u + rnd() % (unsigned)(nv - 300)]] = 1;
    }
    rank_row(row, nv, rank);
    *n_recent = 16;
    for (int i = 0; i < 16; i++) recent[i] = (uint32_t)rank[rnd() % 60u];
    if (kind == ROW_PENALISED_HEAD) {
        *n_recent = PN_SHORT_ROW_K;
        for (int i = 0; i < PN_SHORT_ROW_K; i++) recent[i] = (uint32_t)rank[i];
    }
}

static pn_sampling_params make_params(setting_kind s, row_kind kind)
{
    pn_sampling_params p = pn_sampling_defaults();   /* T 0.8, top_p 0.9, top_k 40, rp 1.1 */
    if (s == SET_GREEDY || s == SET_GREEDY_PRES) p.temperature = 0.0f;
    if (s == SET_CATALOG || s == SET_PRES_FREQ || s == SET_PRES_RAISE || s == SET_BIAS ||
        s == SET_LOGPROBS) {
        p.temperature = 0.6f; p.top_p = 0.95f; p.top_k = 20; p.repeat_penalty = 1.0f;
    }
    if (s == SET_TOPK_OFF) p.top_k = 0;
    if (s == SET_TOPP_ONE) p.top_p = 1.0f;
    if (s == SET_PRES_FREQ || s == SET_GREEDY_PRES) {
        p.presence_penalty = kind == ROW_PENALISED_HEAD ? 60.0f : 0.5f;
        p.frequency_penalty = 0.3f;
    }
    if (s == SET_PRES_RAISE) p.presence_penalty = -0.5f;
    if (s == SET_BIAS) { p.logit_bias = g_bias; p.n_logit_bias = 2; }
    if (s == SET_LOGPROBS) p.logprobs = &g_lp;
    p.rng_seeded = 1;
    return p;
}

/* One seed, one shape: baseline on R, subject on the short row. */
static void draw_pair(const float *row, const float *shortrow, float bound, int nv,
                      const unsigned char *verdict, const uint32_t *recent, int n_recent,
                      setting_kind s, row_kind kind, int mask_live, int batched, int seed,
                      tally *t)
{
    static float *base, *subj;
    static int cap;
    static uint32_t ra[RECENT_MAX], rb[RECENT_MAX];
    pn_tool_grammar_hook hook;
    pn_sampling_params a = make_params(s, kind), b;

    if (cap < nv) {
        free(base); free(subj);
        base = malloc((size_t)nv * sizeof(float));
        subj = malloc((size_t)nv * sizeof(float));
        cap = nv;
        if (!base || !subj) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memset(&hook, 0, sizeof(hook));
    hook.st = (void *)verdict;
    hook.token_text = token_text;
    hook.allowed = allowed;
    hook.active = mask_live;
    hook.n_vocab = nv;
    memcpy(ra, recent, (size_t)n_recent * sizeof(uint32_t));
    memcpy(rb, recent, (size_t)n_recent * sizeof(uint32_t));
    a.tool_grammar = verdict ? &hook : NULL;
    a.rng_state = 0xC0FFEEull ^ (0x9E3779B97F4A7C15ull * (uint64_t)(seed + 1));
    a.dense_row = batched ? refill_already_dense : NULL;
    b = a;
    memcpy(base, row, (size_t)nv * sizeof(float));
    memcpy(subj, shortrow, (size_t)nv * sizeof(float));
    const uint32_t want = pn_sample_next_token(base, nv, &a, ra, n_recent);
    pn_sampler_short_row(subj, bound, fill_full, (void *)row);
    const uint32_t got = pn_sample_next_token(subj, nv, &b, rb, n_recent);
    t->draws++;
    t->mismatches += got != want || a.rng_state != b.rng_state ||
                     a.carried_uniform_set != b.carried_uniform_set;
}

static void run_row(const float *row, int nv, const unsigned char *verdict, const int *rank,
                    const uint32_t *recent, int n_recent, row_kind kind, tally *t)
{
    static float *shortrow;
    static int cap;
    float bound;

    if (cap < nv) {
        free(shortrow);
        shortrow = malloc((size_t)nv * sizeof(float));
        cap = nv;
        if (!shortrow) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    t->rows++;
    if (!build_short(row, nv, rank, shortrow, &bound))
        return;
    t->presented++;
    for (int batched = 0; batched < 2; batched++)
        for (int live = 0; live < 2; live++)
            for (int s = 0; s < SET_KINDS; s++) {
                const int greedy = s == SET_GREEDY || s == SET_GREEDY_PRES;
                for (int seed = 0; seed < (greedy ? 1 : SEEDS); seed++)
                    draw_pair(row, shortrow, bound, nv, verdict, recent, n_recent,
                              (setting_kind)s, kind, live, batched, seed, t);
            }
}

static void run_vocab(int nv, int rows_per_kind, tally *by_kind)
{
    float *row = malloc((size_t)nv * sizeof(float));
    unsigned char *verdict = malloc((size_t)nv);
    int *rank = malloc((size_t)nv * sizeof(int));
    uint32_t recent[RECENT_MAX];
    int n_recent;

    if (!row || !verdict || !rank || init_texts(nv) != 0) { fprintf(stderr, "oom\n"); exit(2); }
    for (int k = 0; k < ROW_KINDS; k++)
        for (int r = 0; r < rows_per_kind; r++) {
            build_row((row_kind)k, nv, row, verdict, rank, recent, &n_recent);
            run_row(row, nv, verdict, rank, recent, n_recent, (row_kind)k, &by_kind[k]);
        }
    free(row); free(verdict); free(rank);
}

/* Real rows (host_row_dump.h records): their own settings and recent window; a masked row
 * gets a synthetic grammar (string content or structural, alternating). */
static int run_dump(const char *path, tally *t)
{
    FILE *f = fopen(path, "rb");
    char magic[4];
    int rows = 0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    while (fread(magic, 1, 4, f) == 4 && memcmp(magic, "PNRD", 4) == 0) {
        uint32_t ver;
        int32_t nv, top_k, rln, seeded, masked, token, n_recent;
        float temp, top_p, rp, pres, freq;
        uint64_t rng_state;
        if (fread(&ver, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1 || fread(&temp, 4, 1, f) != 1 ||
            fread(&top_p, 4, 1, f) != 1 || fread(&top_k, 4, 1, f) != 1 ||
            fread(&rp, 4, 1, f) != 1 || fread(&rln, 4, 1, f) != 1 ||
            fread(&pres, 4, 1, f) != 1 || fread(&freq, 4, 1, f) != 1 ||
            fread(&seeded, 4, 1, f) != 1 || fread(&rng_state, 8, 1, f) != 1 ||
            fread(&masked, 4, 1, f) != 1 || fread(&token, 4, 1, f) != 1 ||
            fread(&n_recent, 4, 1, f) != 1 || ver != 1 || nv <= 0 || n_recent < 0 ||
            n_recent > RECENT_MAX)
            break;
        uint32_t recent[RECENT_MAX];
        float *row = malloc((size_t)nv * sizeof(float));
        unsigned char *verdict = malloc((size_t)nv);
        int *rank = malloc((size_t)nv * sizeof(int));
        if (!row || !verdict || !rank ||
            fread(recent, 4, (size_t)n_recent, f) != (size_t)n_recent ||
            fread(row, 4, (size_t)nv, f) != (size_t)nv || init_texts(nv) != 0) {
            free(row); free(verdict); free(rank);
            break;
        }
        rank_row(row, nv, rank);
        const int content = rows % 2 == 0;
        for (int i = 0; i < nv; i++)
            verdict[i] = (unsigned char)(content ? rnd() % 100u < 92u : 0);
        if (!content)
            for (int i = 0; i < 3; i++) verdict[rank[i * 7 + (int)(rnd() % 7u)]] = 1;
        run_row(row, nv, verdict, rank, recent, n_recent,
                content ? ROW_CONTENT : ROW_STRUCTURAL, t);
        free(row); free(verdict); free(rank);
        rows++;
        (void)temp; (void)top_p; (void)top_k; (void)rp; (void)rln; (void)pres; (void)freq;
        (void)seeded; (void)rng_state; (void)masked; (void)token;
    }
    fclose(f);
    return rows;
}

int main(int argc, char **argv)
{
    static const char *const names[ROW_KINDS] = { "random", "ties-inside", "edge-tie",
                                                  "string-content", "structural",
                                                  "penalised-head" };
    static const char *const reasons[PN_SR_REASONS] = { "entry", "mask", "greedy", "top64",
                                                        "margin" };
    const char *mode = argc > 1 ? argv[1] : "";
    const char *dump = NULL;
    tally by_kind[ROW_KINDS], real;
    pn_short_row_stats st;
    unsigned long draws = 0, mismatches = 0;
    int red = 0, fail = 0;

    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], "--rows") == 0) dump = argv[i + 1];
    g_selftest = strcmp(mode, "--selftest") == 0;
    if (strcmp(mode, "--red-margin") == 0) setenv("PN_SHORT_ROW_RED", "margin", 1);
    if (strcmp(mode, "--red-greedy") == 0) setenv("PN_SHORT_ROW_RED", "greedy", 1);
    red = g_selftest || getenv("PN_SHORT_ROW_RED") != NULL;
    if (!freopen("/dev/null", "w", stderr)) return 2;
    memset(by_kind, 0, sizeof(by_kind));
    memset(&real, 0, sizeof(real));
    run_vocab(20000, 6, by_kind);
    run_vocab(248320, 1, by_kind);
    if (dump && run_dump(dump, &real) < 0) return 2;
    pn_sampler_short_row_stats(&st);
    for (int k = 0; k < ROW_KINDS; k++) {
        printf("  %-15s rows %3lu presented %3lu draws %6lu mismatches %4lu\n", names[k],
               by_kind[k].rows, by_kind[k].presented, by_kind[k].draws, by_kind[k].mismatches);
        draws += by_kind[k].draws;
        mismatches += by_kind[k].mismatches;
    }
    if (dump) {
        printf("  %-15s rows %3lu presented %3lu draws %6lu mismatches %4lu\n", "real (dump)",
               real.rows, real.presented, real.draws, real.mismatches);
        draws += real.draws;
        mismatches += real.mismatches;
    }
    printf("  short-row draws %lu, decided %lu, refills:", st.draws, st.decided);
    for (int i = 0; i < PN_SR_REASONS; i++) printf(" %s %lu", reasons[i], st.refill[i]);
    printf(", fill failures %lu\n", st.fill_failed);
    if (red) {
        printf("%s: %lu / %lu draws differ (%s)\n", mode, mismatches, draws,
               mismatches ? "the comparison can fail" : "FAIL");
        return mismatches ? 0 : 1;
    }
    fail = mismatches != 0;
    if (st.draws != draws || st.decided < draws / 4) {
        printf("  FAIL: short rows not handed over, or too few decided from them\n");
        fail = 1;
    }
    for (int i = 0; i < PN_SR_REASONS; i++)
        if (st.refill[i] == 0) {
            printf("  FAIL: refill reason %s never ran\n", reasons[i]);
            fail = 1;
        }
    if (by_kind[ROW_EDGE_TIE].presented != 0) {
        printf("  FAIL: a row with a tie on the 256/257 edge was presented short\n");
        fail = 1;
    }
    printf("%s: %lu draws, %lu mismatches\n", fail ? "FAIL" : "PASS", draws, mismatches);
    return fail;
}
