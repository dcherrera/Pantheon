/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_dense_shortlist.c
 * @brief On an already-dense row, the grammar mask decided from a SHORTLIST (dense_row set to
 *        the no-op refill, as the batched CUDA path now wires it) draws the same token as the
 *        full-vocabulary scan (dense_row NULL, the CUDA behaviour before) for every seed.
 *
 * @status Complete.  CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Each synthetic dense row is drawn through the real sampler (pn_sample_next_token) twice per
 * seed, on copies: the baseline (full scan) and the subject (shortlist, no-op refill, no chunks
 * or tail counts).  Both must return the same token AND leave the same RNG state.  Grammar: a
 * bitmap, token i is "@@<i>" and "@" / "@@" are live prefixes, so the prefix filter and the
 * verdicts see exactly verdict[i].
 *
 * Rows: random; heavy ties (quantised logits, a plateau across the shortlist edge); NO allowed
 * token in the top-GMS_MAX (the shortlist cannot decide: the no-op refill, the retry and the
 * full scan run); no allowed token at all (the full scan fails open); a handful of allowed
 * tokens inside and outside the top 256.  Settings: greedy, the server defaults (T 0.8,
 * top_p 0.9, top_k 40, repeat 1.1 over a window that hits the head), top_k 0 (off), top_p 1,
 * and the mask inactive.  Vocabularies 20000 and 248320 (the real size).
 *
 * PASS: 0 mismatches, the shortlist decided random-row draws, and the fallback ran.
 * `--selftest`: a subject that drops the grammar must mismatch (the comparison can fail).
 * `--red`: PN_TT_RED=grammar_bound_blind (the shortlist's outside bound charged nothing) must mismatch.
 * `--red-hint`: PN_GRAMMAR_HINT_RED=1 (the shortlist's row hint drops a survivor) must mismatch.
 * The subject's draws read their survivors from that hint (engine.c grammar_hint_kept); PASS
 * also requires the hint to have been used.
 */
#include "../src/core/engine.h"
#include "../src/core/grammar_mask_shortlist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEEDS 40
#define RECENT 16
#define TEXT_BYTES 12

static uint64_t g_rng = 0x51D3A7C90B2E4F68ull;
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
/* sg_gs_row_already_dense (serve_generation.c): the row is already on the host. */
static int refill_already_dense(void *ctx, float *logits, int n_vocab) {
    (void)ctx; (void)logits; (void)n_vocab;
    return 0;
}

typedef enum { ROW_RANDOM, ROW_TIES, ROW_NO_ALLOWED_IN_TOP, ROW_NONE_ALLOWED, ROW_FEW_ALLOWED,
               ROW_KINDS } row_kind;
typedef enum { SET_GREEDY, SET_DEFAULTS, SET_TOPK_OFF, SET_TOPP_ONE, SET_MASK_OFF,
               SET_KINDS } setting_kind;

typedef struct {
    unsigned long draws, mismatches, shortlist, fullscan, rows;
} tally;

static const float *g_sort_row;
/* Value descending, id ascending: the order the sampler's tie-breaks use. */
static int cmp_rank(const void *a, const void *b) {
    const int x = *(const int *)a, y = *(const int *)b;
    if (g_sort_row[x] != g_sort_row[y]) return g_sort_row[x] > g_sort_row[y] ? -1 : 1;
    return x - y;
}

static void build_row(row_kind kind, int nv, float *row, unsigned char *verdict, int *rank)
{
    for (int i = 0; i < nv; i++) {
        row[i] = rnd_gauss() * 2.5f;
        if (kind == ROW_TIES) row[i] = (float)(int)(row[i] * 2.0f) * 0.5f;
        verdict[i] = (unsigned char)(rnd() % 100u < 25u);
    }
    for (int i = 0; i < 6; i++) row[rnd() % (unsigned)nv] += 8.0f + (float)i;   /* a heavy head */
    for (int i = 0; i < nv; i++) rank[i] = i;
    g_sort_row = row;
    qsort(rank, (size_t)nv, sizeof(int), cmp_rank);
    if (kind == ROW_TIES)   /* a plateau straddling the 256th place */
        for (int i = 250; i < 262; i++) row[rank[i]] = row[rank[250]];
    if (kind == ROW_NO_ALLOWED_IN_TOP)
        for (int i = 0; i < 300; i++) verdict[rank[i]] = 0;
    if (kind == ROW_NONE_ALLOWED)
        memset(verdict, 0, (size_t)nv);
    if (kind == ROW_FEW_ALLOWED) {
        memset(verdict, 0, (size_t)nv);
        verdict[rank[rnd() % 40u]] = 1;
        verdict[rank[rnd() % 200u]] = 1;
        verdict[rank[256u + rnd() % (unsigned)(nv - 256)]] = 1;
    }
}

static pn_sampling_params make_params(setting_kind s)
{
    pn_sampling_params p = pn_sampling_defaults();   /* T 0.8, top_p 0.9, top_k 40, rp 1.1 */
    if (s == SET_GREEDY) p.temperature = 0.0f;
    if (s == SET_TOPK_OFF) p.top_k = 0;
    if (s == SET_TOPP_ONE) p.top_p = 1.0f;
    p.rng_seeded = 1;
    return p;
}

/* One seed: the baseline (full scan) and the subject (shortlist) on copies of the row. */
static void draw_pair(const float *row, int nv, const unsigned char *verdict, const int *rank,
                      setting_kind s, int seed, int subject_drops_grammar, tally *t)
{
    pn_tool_grammar_hook hook;
    float *base = malloc((size_t)nv * sizeof(float));
    float *subj = malloc((size_t)nv * sizeof(float));
    uint32_t recent_a[RECENT], recent_b[RECENT];
    pn_sampling_params a = make_params(s), b;
    unsigned long shortlist_before, fullscan_before;
    uint32_t want, got;

    if (!base || !subj) { fprintf(stderr, "out of memory\n"); exit(2); }
    memset(&hook, 0, sizeof(hook));
    hook.st = (void *)verdict;
    hook.token_text = token_text;
    hook.allowed = allowed;
    hook.active = s != SET_MASK_OFF;
    hook.n_vocab = nv;
    for (int i = 0; i < RECENT; i++) recent_a[i] = recent_b[i] = (uint32_t)rank[rnd() % 60u];
    a.tool_grammar = &hook;
    a.rng_state = 0xC0FFEEull ^ (0x9E3779B97F4A7C15ull * (uint64_t)(seed + 1));
    b = a;
    b.dense_row = refill_already_dense;
    b.dense_ctx = NULL;
    b.tail_counts = NULL;
    b.dense_chunks = NULL;
    if (subject_drops_grammar) b.tool_grammar = NULL;
    memcpy(base, row, (size_t)nv * sizeof(float));
    memcpy(subj, row, (size_t)nv * sizeof(float));
    want = pn_sample_next_token(base, nv, &a, recent_a, RECENT);
    shortlist_before = pn_sampler_grammar_shortlist_steps();
    fullscan_before = pn_sampler_grammar_fullscan_steps();
    got = pn_sample_next_token(subj, nv, &b, recent_b, RECENT);
    t->shortlist += pn_sampler_grammar_shortlist_steps() - shortlist_before;
    t->fullscan += pn_sampler_grammar_fullscan_steps() - fullscan_before;
    t->draws++;
    t->mismatches += got != want || a.rng_state != b.rng_state ||
                     a.carried_uniform_set != b.carried_uniform_set;
    free(base);
    free(subj);
}

static void run_vocab(int nv, int rows_per_kind, int selftest, tally *by_kind)
{
    float *row = malloc((size_t)nv * sizeof(float));
    unsigned char *verdict = malloc((size_t)nv);
    int *rank = malloc((size_t)nv * sizeof(int));

    if (!row || !verdict || !rank || init_texts(nv) != 0) { fprintf(stderr, "oom\n"); exit(2); }
    for (int k = 0; k < ROW_KINDS; k++)
        for (int r = 0; r < rows_per_kind; r++) {
            build_row((row_kind)k, nv, row, verdict, rank);
            by_kind[k].rows++;
            for (int s = 0; s < SET_KINDS; s++)
                for (int seed = 0; seed < (s == SET_GREEDY ? 1 : SEEDS); seed++)
                    draw_pair(row, nv, verdict, rank, (setting_kind)s, seed, selftest,
                              &by_kind[k]);
        }
    free(row);
    free(verdict);
    free(rank);
}

int main(int argc, char **argv)
{
    static const char *const names[ROW_KINDS] = { "random", "ties", "no-allowed-in-top-256",
                                                  "none-allowed", "few-allowed" };
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
    const int red = argc > 1 && strcmp(argv[1], "--red") == 0;
    const int red_hint = argc > 1 && strcmp(argv[1], "--red-hint") == 0;
    tally by_kind[ROW_KINDS];
    unsigned long draws = 0, mismatches = 0;
    int fail;

    /* The fail-open rows warn once per draw on stderr; the verdict is printed on stdout. */
    if (!freopen("/dev/null", "w", stderr)) return 2;
    if (red) setenv("PN_TT_RED", "grammar_bound_blind", 1);
    if (red_hint) setenv("PN_GRAMMAR_HINT_RED", "1", 1);
    const unsigned long hint_before = pn_sampler_row_hint_uses();
    memset(by_kind, 0, sizeof(by_kind));
    run_vocab(20000, 12, selftest, by_kind);
    run_vocab(248320, 2, selftest, by_kind);
    for (int k = 0; k < ROW_KINDS; k++) {
        printf("  %-22s rows %3lu draws %5lu mismatches %4lu  shortlist-decided %5lu  full-scan %5lu\n",
               names[k], by_kind[k].rows, by_kind[k].draws, by_kind[k].mismatches,
               by_kind[k].shortlist, by_kind[k].fullscan);
        draws += by_kind[k].draws;
        mismatches += by_kind[k].mismatches;
    }
    if (selftest || red || red_hint) {
        printf("%s: %lu / %lu draws differ (%s)\n", selftest ? "selftest, subject without the grammar"
                                                   : red ? "red, outside bound blind"
                                                         : "red, row hint drops a survivor",
               mismatches, draws, mismatches ? "the comparison can fail" : "FAIL");
        return mismatches ? 0 : 1;
    }
    fail = mismatches != 0;
    /* Not vacuous: the shortlist decides random rows, the fallback runs where it cannot. */
    if (by_kind[ROW_RANDOM].shortlist < by_kind[ROW_RANDOM].draws / 4) {
        printf("  FAIL: the shortlist decided too few random-row draws\n");
        fail = 1;
    }
    if (pn_sampler_row_hint_uses() == hint_before) {
        printf("  FAIL: the subject never read the shortlist's row hint\n");
        fail = 1;
    }
    if (by_kind[ROW_NO_ALLOWED_IN_TOP].fullscan == 0 || by_kind[ROW_NONE_ALLOWED].fullscan == 0) {
        printf("  FAIL: the full-scan fallback never ran\n");
        fail = 1;
    }
    printf("%s: %lu draws, %lu mismatches\n", fail ? "FAIL" : "PASS", draws, mismatches);
    return fail;
}
